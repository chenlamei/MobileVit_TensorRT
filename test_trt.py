#!/usr/bin/python

import os
import sys
import ctypes
import argparse
import numpy as np
import tensorrt as trt
from cuda import cudart
from glob import glob 
from time import time_ns
from datetime import datetime as dt


dataFilePath = "./benchmark/data"
planFilePath   = "./target/"
encoderScoreFile = "./target/Score.txt"
soFileList = glob("./plugin/" + "*.so")

tableHead = \
"""
bs: Batch Size
lt: Latency (ms)
tp: throughput (fps)
a0: mean of absolute difference of output 0
a1: median of absolute difference of output 0
a2: maximum of absolute difference of output 0
r0: mean of relative difference of output 0
r1: median of relative difference of output 0
r2: maximum of relative difference of output 0
----+--------+---------+---------+---------+---------+---------+---------+---------+-------------
  bs|      lt|       tp|       a0|       a1|       a2|       r0|       r1|       r2| output check
----+--------+---------+---------+---------+---------+---------+---------+---------+-------------
"""
class MyProfiler(trt.IProfiler):
    def __init__(self):
        super(MyProfiler, self).__init__()
        run_time=0

    def report_layer_time(self, layerName, ms):
        print("Timing: %8.3fus -> %s"%(ms*1000,layerName))

def check(a, b, weak=False, epsilon = 1e-5):
    if weak:
        res = np.all( np.abs(a - b) < epsilon )
    else:
        res = np.all( a == b )
    a0=np.mean(np.abs(a - b))
    a1=np.median(np.abs(a - b))
    a2=np.max(np.abs(a - b))
    
    r0 = np.mean(np.abs(a - b) / (np.abs(b) + epsilon))
    r1 = np.median(np.abs(a - b) / (np.abs(b) + epsilon))
    r2 = np.max(np.abs(a - b) / (np.abs(b) + epsilon))
    #print("check:",res,diff0,diff1)
    return res,a0,a1,a2,r0,r1,r2

#-------------------------------------------------------------------------------
logger = trt.Logger(trt.Logger.ERROR)
trt.init_libnvinfer_plugins(logger, '')

if len(soFileList) > 0:
    print("Find Plugin %s!"%soFileList)
else:
    print("No Plugin!")
for soFile in soFileList:
    ctypes.cdll.LoadLibrary(soFile)

#-------------------------------------------------------------------------------
def testMobileVit(encoderPlanFile,dynamic=True,batch=1,Profiler=False,cudaGraph=False):

    with open(encoderScoreFile, 'w') as f:

        if os.path.isfile(encoderPlanFile):
            with open(encoderPlanFile, 'rb') as encoderF:
                engine = trt.Runtime(logger).deserialize_cuda_engine(encoderF.read())
            if engine is None:
                print("Failed loading %s"%encoderPlanFile)
                return
            print("Succeeded loading %s"%encoderPlanFile)
        else:
            print("Failed finding %s"%encoderPlanFile)
            return

        nInput = np.sum([ engine.binding_is_input(i) for i in range(engine.num_bindings) ])
        nOutput = engine.num_bindings - nInput
        context = engine.create_execution_context()
            
        print(tableHead)  # for standard output
        files=glob(dataFilePath + "/*.npy")
        #tset_files=["benchmark/data/batch1.npy"]
        for ioFile in  sorted(glob(dataFilePath + "/batch*.npy")):
            ioData = np.load(ioFile,allow_pickle=True).item()
            in_tensor=ioData['in_tensor']
            out_tensor=ioData['out_tensor']
            batchSize=in_tensor.shape[0]
            if (Profiler or (not dynamic)) and batchSize!=batch:
                continue
            context.set_binding_shape(0, in_tensor.shape)
            bufferH = []
            bufferH.append( in_tensor.astype(np.float32).reshape(-1) )
            for i in range(nInput, nInput + nOutput):                
                bufferH.append( np.empty(context.get_binding_shape(i), dtype=trt.nptype(engine.get_binding_dtype(i))) )

            bufferD = []
            for i in range(nInput + nOutput):                
                bufferD.append( cudart.cudaMalloc(bufferH[i].nbytes)[1] )

            for i in range(nInput):
                cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

            context.execute_v2(bufferD)

            for i in range(nInput, nInput + nOutput):
                cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
            
            if cudaGraph:
                stream = cudart.cudaStreamCreate()[1]
                cudart.cudaStreamBeginCapture(stream, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal)
                context.execute_async_v2(bufferD, stream)
                _, graph = cudart.cudaStreamEndCapture(stream)
                _, graphExe, _ = cudart.cudaGraphInstantiate(graph, b"", 0)
                # warm up
                for i in range(20):
                    cudart.cudaGraphLaunch(graphExe, stream)
                cudart.cudaStreamSynchronize(stream)

                t0 = time_ns()
                for i in range(50):
                    cudart.cudaGraphLaunch(graphExe, stream)
                    cudart.cudaStreamSynchronize(stream)
                t1 = time_ns()
            else:
                # warm up
                for i in range(20):
                    context.execute_v2(bufferD)

                # test infernece time
                t0 = time_ns()
                for i in range(50):
                    context.execute_v2(bufferD)
                t1 = time_ns()

            
            if Profiler and batchSize==batch:
                context.profiler = MyProfiler()
                context.execute_v2(bufferD)
            
            timePerInference = (t1-t0)/1000/1000/50
            indexEncoderOut = engine.get_binding_index('class')
            
            check0 = check(bufferH[indexEncoderOut],ioData['out_tensor'],True,5e-5)

            string = "%4d,%8.3f,%9.3e,%9.3e,%9.3e,%9.3e,%9.3e,%9.3e,%9.3e, %s"%(batchSize,
                                                        timePerInference,
                                                        batchSize/timePerInference*1000,
                                                        check0[1],
                                                        check0[2],
                                                        check0[3],
                                                        check0[4],
                                                        check0[5],
                                                        check0[6],
                                                        "Good" if check0[1]< 1e-2 and check0[2] < 1e-2 else "Bad")
            print(string)
            f.write(string + "\n")
            
            

            for i in range(nInput + nOutput):                
                cudart.cudaFree(bufferD[i])


if __name__ == "__main__":
    
  parser = argparse.ArgumentParser(description='test trt describe.')
  parser.add_argument(
      "--trt_path",
      type = str,
      default="target/MobileViT_dynamic_final_fp32.trt",
      help="input trt model path, default is target/MobileViT_dynamic_final_fp32.trt.")

  parser.add_argument(
      "--dynamic",
       default=False, action='store_true',
      help="dynamic  model , default is False.")

  parser.add_argument(
      "--ProfilerLayer",
       default=False, action='store_true',
      help="print every layers' latency , default is False.")
  parser.add_argument(
      "--cudaGraph",
       default=False, action='store_true',
      help="use cuda graph , default is False.")

  parser.add_argument(
      "--batch",
      type=int,
      default=1,
      help="batchsize of onnx models, default is 1.")

  args = parser.parse_args()
  print(args)
  testMobileVit(args.trt_path,args.dynamic,args.batch,args.ProfilerLayer,args.cudaGraph)
