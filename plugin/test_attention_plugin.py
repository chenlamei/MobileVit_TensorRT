#https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/kernels/vit_kernels.cu

import os
import torch
import ctypes
import numpy as np
import torch.nn as nn
from cuda import cudart  
import tensorrt as trt

soFilePath= './attentionPlugin.so'
epsilon= 6e-5
use_fp16=True     
B=1
C=4
H=256
W=96


class SelfAttention(nn.Module):
    def __init__(self, dim=144, num_heads = 1, dim_head = 32):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.dim_head = int(dim / num_heads) if dim_head is None else dim_head
        _weight_dim = self.num_heads * self.dim_head
        self.to_qvk = nn.Linear(dim, _weight_dim * 3, bias = False)
        self.scale_factor = dim ** -0.5#0.083333
        self.w_out = nn.Linear(_weight_dim, dim, bias = False)

    def forward(self, x):
        qkv = x.chunk(3, dim = -1)
        q, k, v =qkv[0],qkv[1],qkv[2]
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale_factor
        attn = torch.softmax(dots, dim = -1)
        out = torch.matmul(attn, v)
        return out


def selfAttentionCPU(bufferH):
    model=SelfAttention()
    model.eval()
    #x= torch.rand((1,4,256,96))# (1,4,64,96) (1,4,16,96)
    out=model(torch.from_numpy(bufferH))
    return out.detach().numpy()

def check(a, b, weak = False):
    if weak:
        return np.all( np.abs(a - b) < epsilon)
    else:
        return np.all( a == b )

def getPlugin(scalar=0.083333333):
    for c in trt.get_plugin_registry().plugin_creator_list:
        #print(c.name)
        if c.name == 'Attention':
            parameterList = []
            parameterList.append(trt.PluginField("scale", np.float32(scalar), trt.PluginFieldType.FLOAT32))
            return c.create_plugin(c.name, trt.PluginFieldCollection(parameterList))
        #if c.name == 'Attention':
            #return c.create_plugin(c.name, trt.PluginFieldCollection([]))
    return None

def run():
    logger = trt.Logger(trt.Logger.INFO)#VERBOSE
    trt.init_libnvinfer_plugins(logger, '')
    ctypes.cdll.LoadLibrary(soFilePath)

    builder         = trt.Builder(logger)
    network         = builder.create_network(1<<0)
    config          = builder.create_builder_config()
    config.max_workspace_size = 6 << 30
    config.flags    = 0
    if use_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    inputTensorList = []
    inputTensorList.append( network.add_input('inputT', trt.float32, [-1,4,256,96]) )

    profile = builder.create_optimization_profile()
    profile.set_shape('inputT',[1,4,256,96],[4,4,256,96],[8,4,256,96])
    config.add_optimization_profile(profile)

    pluginLayer = network.add_plugin_v2(inputTensorList, getPlugin())

    network.mark_output(pluginLayer.get_output(0))

    engineString = builder.build_serialized_network(network, config)
    engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)

    context = engine.create_execution_context()
    context.set_binding_shape(0,[B,C,H,W])
    print("Binding all? %s"%(["No","Yes"][int(context.all_binding_shapes_specified)]))
    
    nInput = np.sum([ engine.binding_is_input(i) for i in range(engine.num_bindings) ])
    nOutput = engine.num_bindings - nInput
    for i in range(engine.num_bindings):
        print("input ->" if engine.binding_is_input(i) else "output->",engine.get_binding_dtype(i),engine.get_binding_shape(i),context.get_binding_shape(i))

    bufferH = []
    bufferH.append( np.random.rand(B,C,H,W).astype(np.float32).reshape(B,C,H,W) * 5 - 1)
    bufferH.append(np.empty(context.get_binding_shape(1),dtype=trt.nptype(engine.get_binding_dtype(1))))

    bufferD = []
    for i in range(engine.num_bindings):
        bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

    for i in range(nInput):
        cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

    context.execute_v2(bufferD)

    for i in range(nInput, nInput + nOutput):
        cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

    print("check result:")
    temp1 = bufferH[-1]
    temp2 = selfAttentionCPU(bufferH[0])
    diff=np.abs(temp1 - temp2)
    #print(diff[0,0,24,:])
    print(check(temp1,temp2,True), "max diff=%f"%(np.max(diff)) )
    
    for b in bufferD:
        cudart.cudaFree(b)

if __name__ =='__main__':
    run()
    #bufferH = []
    #bufferH.append( np.random.rand(B,C,H,W).astype(np.float32).reshape(B,C,H,W) * 2 - 1)
    #cpu_out=selfAttentionCPU(bufferH[0])
    print("...")