import tensorrt as trt
import os
import ctypes
import onnx
import numpy as np
import onnx_graphsurgeon as gs

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
        config.int8_calibrator = None ##need add calibrator
        config.set_calibration_profile(profile)
        #config.set_flag(trt.BuilderFlag.STRICT_TYPES)
    engineString = builder.build_serialized_network(network, config)
    if engineString == None:
        print("Failed building engine!")
        exit()
    print("Succeeded building engine!")
    with open(trtFile, 'wb') as f:
        f.write(engineString)
        print("Succeeded save engine!")
encoder_in_shapes={'input':[(1,3,256,256),(1,3,256,256),(1,3,256,256)]}
trt_builder_plugin("target/MobileViT.onnx","./target/MobileViT_fp32.plan",encoder_in_shapes,pluginFileList=[],use_fp16=False,set_int8_precision=False)
encoder_in_shapes={'input':[(1,3,256,256),(4,3,256,256),(4,3,256,256)]}
trt_builder_plugin("target/MobileViT_dynamic.onnx","./target/MobileViT_dynamic_fp32.plan",encoder_in_shapes,pluginFileList=[],use_fp16=False,set_int8_precision=False)
