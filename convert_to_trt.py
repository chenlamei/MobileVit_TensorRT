import os
import ctypes
import onnx
import argparse
import numpy as np
import tensorrt as trt
import onnx_graphsurgeon as gs
from calibrator import MobileVitCalibrator
 
def trt_builder_plugin(onnxFile,trtFile,in_shapes,workspace=22,pluginFileList=[],use_fp16=False,set_int8_precision=False):
    logger = trt.Logger(trt.Logger.VERBOSE)#ERROR INFO  VERBOSE
    trt.init_libnvinfer_plugins(logger, '')
    if len(pluginFileList)>0:
        for pluginFile in pluginFileList:
            ctypes.cdll.LoadLibrary(pluginFile)
            print("load plugin",pluginFile)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    profile = builder.create_optimization_profile()
    config.max_workspace_size = (1 << 30)*workspace
    parser = trt.OnnxParser(network, logger)
    if not os.path.exists(onnxFile):
        print("Failed finding onnx file!")
        exit()
    print("Succeeded finding onnx file!")
    with open(onnxFile, 'rb') as model:
        if not parser.parse(model.read()):
            print("Failed parsing .onnx file!")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            exit()
        print("Succeeded parsing .onnx file!")
    for i in range(network.num_inputs):
        inputTensor = network.get_input(i)
        name=inputTensor.name
        if name in in_shapes:
            profile.set_shape(name, in_shapes[name][0],in_shapes[name][1],in_shapes[name][2])
        
    config.add_optimization_profile(profile)
    if use_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    if set_int8_precision:
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator=MobileVitCalibrator()
        config.set_calibration_profile(profile)       
        
    engineString = builder.build_serialized_network(network, config)
    if engineString == None:
        print("Failed building engine!")
        exit()
    print("Succeeded building engine!")
    with open(trtFile, 'wb') as f:
        f.write(engineString)
        print("Succeeded save engine!")
if __name__=="__main__":
  parser = argparse.ArgumentParser(description='onnx convert to trt describe.')
  parser.add_argument(
      "--input_path",
      type = str,
      default="target/MobileViT_dynamic_final.onnx",
      help="input onnx model path, default is target/MobileViT_dynamic_final.onnx.")

  parser.add_argument(
      "--save_path",
      type=str,
      default="./target/MobileViT_dynamic_final.trt",
      help="save direction of onnx models,default is ./target/MobileViT_dynamic_final.trt.")
      
  parser.add_argument(
      "--dynamic",
       default=False, action='store_true',
      help="export  dynamic onnx model , default is True.")
      
  parser.add_argument(
      "--batch",
      type=int,
      default=1,
      help="batchsize of onnx models, default is 1.")

  parser.add_argument(
      "--fp16",
       default=False, action='store_true',
      help="use fp16, default is False.")
      
  parser.add_argument(
      "--int8", default=False, action='store_true',
      help="use int8 , default is False.")

  args = parser.parse_args()
  print(args)
  if args.dynamic:
    encoder_in_shapes={'input':[(1,3,256,256),(4,3,256,256),(8,3,256,256)]}
  else:
    encoder_in_shapes={'input':[(args.batch,3,256,256),(args.batch,3,256,256),(args.batch,3,256,256)]}
  trt_builder_plugin(args.input_path,args.save_path,encoder_in_shapes,pluginFileList=["plugin/layerNormPlugin.so","plugin/attentionPlugin.so"],use_fp16=args.fp16,set_int8_precision=args.int8)

