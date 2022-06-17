import os
import torch
import ctypes
import argparse
import numpy as np
from glob import glob 
from cuda import cudart
import tensorrt as trt
import MobileViT_Pytorch.utils as utils
from benchmark.imagenet_lmdb_datasets import get_test_data

class TrtMobileVitInfer(object):
    def __init__(self, encoderPlanFile,pluginFileList,batchsize):
        logger = trt.Logger(trt.Logger.VERBOSE)#ERROR INFO  VERBOSE
        trt.init_libnvinfer_plugins(logger, '')
        if len(pluginFileList)>0:
            for pluginFile in pluginFileList:
                ctypes.cdll.LoadLibrary(pluginFile)
                print("load plugin",pluginFile)
        #deserialize_cuda_engine
        if os.path.isfile(encoderPlanFile):
            with open(encoderPlanFile, 'rb') as encoderF:
                self.engine = trt.Runtime(logger).deserialize_cuda_engine(encoderF.read())
            if self.engine is None:
                print("Failed loading %s"%encoderPlanFile)
                return
            print("Succeeded loading %s"%encoderPlanFile)
        else:
            print("Failed finding %s"%encoderPlanFile)
            return
        self.context = self.engine.create_execution_context()
        self.context.set_binding_shape(0, [batchsize,3,256,256])
        self.buffeSizeIn = batchsize*3*256*256 * trt.float32.itemsize
        self.buffeSizeOut =batchsize*1000 * trt.float32.itemsize
        self.bufferD =[] 
        self.bufferD.append(cudart.cudaMalloc(self.buffeSizeIn)[1])
        self.bufferD.append(cudart.cudaMalloc(self.buffeSizeOut)[1])
    def __del__(self):
        cudart.cudaFree(self.bufferD[0])
        cudart.cudaFree(self.bufferD[1])
    def infer(self,input_data,output_data):
        
        cudart.cudaMemcpy(self.bufferD[0], input_data.ctypes.data, input_data.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
        self.context.execute_v2(self.bufferD)

        cudart.cudaMemcpy(output_data.ctypes.data, self.bufferD[1], output_data.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost) 


def test_precision(trt_path,data_path,batch):
    val_acc1 = utils.AverageMeter("Val Acc@1", ":6.2f")
    val_acc5 = utils.AverageMeter("Val Acc@5", ":6.2f")
    val_loader=get_test_data(data_path,batch)
    pluginFileList = glob("./plugin/" + "*.so")
    trt_infer=TrtMobileVitInfer(trt_path,pluginFileList,batch)
    output=np.empty((batch,1000),dtype=np.float32)
    for it, (inputs, targets) in enumerate(val_loader):
        inputs = inputs.numpy()
        targets = targets
        trt_infer.infer(inputs,output)
        acc1, acc5 = utils.accuracy(torch.from_numpy(output), targets, topk=(1, 5))
        val_acc1.update(acc1.item(), batch)
        val_acc5.update(acc5.item(), batch)
    
    print("top1:", val_acc1.avg,"top5:",val_acc5.avg)
    trt_infer.__del__()

if __name__=='__main__':
  parser = argparse.ArgumentParser(description='test trt describe.')
  parser.add_argument(
      "--trt_path",
      type = str,
      default="target/MobileViT_dynamic_final_fp16.trt",
      help="input trt model path, default is target/MobileViT_dynamic_final_fp16.trt.")
  parser.add_argument(
      "--data_path",
      type = str,
      default="/target/test_data/val/val.lmdb",
      help="test data path, default is /root/test_data/val/val.lmdb.")

  parser.add_argument(
      "--batch",
      type=int,
      default=8,
      help="batchsize of onnx models, default is 1.")

  args = parser.parse_args()
  print(args)
  test_precision(args.trt_path,args.data_path,args.batch)
  #valdir="/home/notebook/data/group/imagenet-pytorch-lmdb/val/val.lmdb"
