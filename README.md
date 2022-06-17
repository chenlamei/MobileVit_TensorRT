# mobileVit-TensorRT

# 原始模型
### 模型简介
- mobileVit时一种轻量型的视觉transformer，本项目中的模型应用在ImageNet分类任务中

- 模型网络具体说明可参考 https://mp.weixin.qq.com/s/OoXGZ5pHLMSPZjyriWYstA ，pytoch 实现源码为https://github.com/wilile26811249/MobileViT ，文献 https://arxiv.org/abs/2110.02178 

- 模型的整体结构，如下图所示,MobileViT 中的初始层是一个 3×3 的标准卷积，然后是 MobileNetv2（或 MV2）块和 MobileViT 块，激活函数为Swish。
 
![Image_20220526110045](https://user-images.githubusercontent.com/106289938/170406914-d78b4042-a4bb-4732-902c-5b64dd9969f0.png)

### 模型优化的难点

- 动态网络转成TRT文件时会出现如下错误

python: /root/gpgpu/MachineLearning/myelin/src/compiler/optimizer/kqv_gemm_split.cpp:350: void myelin::ir::kqv_split_pattern_t::check_transpose(): Assertion `in_dims.size() == 3' failed.
  
![image](https://user-images.githubusercontent.com/106289938/170433167-e32e5cbe-af6d-49ae-82cc-d177d9133252.png)

- 动态网络转换后会生成大量shape操作相关节点

- 网络转换成INT8 trt engine时会阻碍tensrrt自动层融合


# 优化过程
- 环境依赖

  - 硬件环境：本次比赛使用nvidia官方的docker，镜像名称为nvcr.io/nvidia/tensorrt:22.04-py3 ，GPU为NVIDIA A10，CPU为4核Intel(R) Xeon(R) Platinum 8369B CPU @ 2.90GHz

  - 软件环境：系统版本：Ubuntu 9.4.0-1ubuntu1~20.04.1； GPU驱动：510.39.01； CUDA:11.6； cudnn:8.4.0； TensrRT:8.2.4； python: 3.8.10

  - dockers设置

    (1)下载git repo:
    
    mkdir trt2022_src;  cd trt2022_src ;   git clone https://github.com/chenlamei/MobileVit_TensorRT.git ;   cd ..

    (2)docker下载及挂载到/tagert/目录运行:

     docker pull nvcr.io/nvidia/tensorrt:22.04-py3

     sudo docker run --gpus all -it --name tensorrt_22_06 -v ~/trt2022_src:/target nvcr.io/nvidia/tensorrt:22.04-py3

     docker start -i  tensorrt_22_06

    (3)docker中安装必要的python包：

     cd /target/MobileVit_TensorRT

     pip config set global.index-url https://pypi.douban.com/simple

     pip install -r requirements.txt

     pip install onnx-graphsurgeon

- pytorch模型转换为onnx

  python convert_to_onnx.py --model_path [pytorch model path] --save_path [save path of onnx file] --batch [batchsize of the model] 

  --opt :optimize pytorch model ；--dynamic: export dynamic model（注：静态网络需要batch 参数，动态网络不需要）

  - 例子：

    转化batchsize为4的静态网络: python convert_to_onnx.py --model_path MobileViT_Pytorch/weights-file/model_best.pth.tar --save_path ./target/MobileViT_b4.onnx --batch 4

    转化batchsize为4的优化后的静态网络: python convert_to_onnx.py --model_path MobileViT_Pytorch/weights-file/model_best.pth.tar --save_path ./target/MobileViT_opt_b4.onnx --batch 4 --opt 

    转化优化后的动态网络: python convert_to_onnx.py --model_path MobileViT_Pytorch/weights-file/qat_checkpoint.pth.tar --save_path target/MobileViT_dynamic_opt.onnx --opt --dynamic 

- onnx模型添加plugin

  python onnx_add_plugin.py --input_path [path of input onnx file] --save_path [save path of onnx file]

  - 例子：python onnx_add_plugin.py --input_path target/MobileViT_dynamic_opt.onnx --save_path target/MobileViT_dynamic_opt_plugin.onnx

     注：(1)会添加layernorm plugin和Attention plugin；（2）若不添加plugin可跳过此步骤

- onnx转化trt

  （1）生成plugin so files (注意需要按照自己硬件环境修改makefile中的TRT_PATH和CUDA_PATH): cd plugin；make -B ； cd ..   

  （2）将onnx模型转换为tensorrt engine

       python convert_to_trt.py --input_path [path of onnx file] --save_path [path of trt file]  --batch [batchsize of the model] 

       --dynamic :the onnx mode is dynamic ; --fp16:use fp16 precision ;--int8:use int8 precision

  - 例子：python convert_to_trt.py --input_path target/MobileViT_dynamic_opt_plugin.onnx --save_path target/MobileViT_dynamic_opt_plugin_int8.trt --dynamic --fp16 --int8


- 优化记录

  （1）简化MultiHeadSelfAttention：通过修改MobileViT_Pytorch/models/model.py源代码，去掉MultiHeadSelfAttention中forward中的两个rearrange的语句得到优化后的网络图，同时解决了动态网络trt转换错误的问题

  （2）4DMM转为2DMM：参考https://github.com/NVIDIA/trt-samples-for-hackathon-cn/tree/master/cookbook/10-BestPractice/Convert3DMMTo2DMM ，实验发现4DMM转为2DMM也会起到加速效果。修改MobileViT_Pytorch/models/model.py 中FFN或Transformer中forward代码，在FFN前后各添加一个reshape节点，将4D矩阵变为2D矩阵，计算完成后再转换为4D

  （3）cuda graph优化：使用nsight system监测原始网络inference发现tensorrt将几十个node转为一个foreignnode，而且进行了优化，例如将layernorm的多个node转换为cuda kernel  __myl__bb0_15_NegExpAddDivMulResTra*。使用nsight system发现gpu kernel submit时间较长，所以使用cuda graph优化cpu占用和gpu kernel submit时间

  （4）int8优化：生成trt engine时添加--fp16 --int8 flag可以生成int8 engine。测试速度发现INT8效果不好，使用test_trt.py测试时添加--ProfilerLayer flag，打印每层信息，发现layernorm被拆分成了ReduceMean +Sub ......等layer。所以int8速度不理想主要是由于使用int8导致trt融合layer失败，layernorm被拆分等原因，后续添加plugin，手动融合layernorm等层。
  
  精度上面，通过test_trt_precision.py测试发现PTQ精度有一定下降，top1 acc约有2个百分点的下降，通过修改IInt8MinMaxCalibrator为IInt8EntropyCalibrator2实现了一定的精度提升。后面通过简化版的QAT（训练中只对weight进行量化，feature map不量化）提高了量化精度，最终量化后top1 acc精度仅下降0.3个百分点，top5下降 0.25个百分点

  （5）添加attention+layernorm plugin：通过修改onnx 模型将layernorm相关的节点融合为一个节点，onnx修改代码为可参考onnx_add_plugin.py中的addLayerNormPlugin。通过修改onnx 模型将MultiHeadSelfAttention相关的节点融合为一个节点，onnx修改代码为onnx_add_plugin.py中的addAttentionPlugin。写对应c++代码，添加tensorrt plugin，代码在plugin文件夹中。


# 精度与加速效果



- 精度

  test_trt.py 可输出不同batchsize下tensorrt 模型和pytorch模型输出差异（包括相对差和绝对差的平均值、中位数、最大值）。实际测试发现由于模型任务为分类，导致大部分输出值数值较小，相对误差较大，所以在判断模型精度时主要以绝对误差为主。pytorch模型是在参考源码给出的训练好的模型，模型文件为 MobileViT_Pytorch/weights-file/model_best.pth.tar。测试数据的生成可参考benchmark/gen_test_data.py，基准pytorch模型的精度测试可参考benchmark/test_torch_precision.py 。

  test_trt_precision.py 可测试tensorrt 模型在imagenet数据集上的正确率，相对test_trt.py更有说服力一些，但需要下载测试数据集，且数据集为LMDB格式，可参考benchmark/imagenet_lmdb_datasets.py。在测试INT8 engine时发现使用test_trt_precision.py更有效，因为分类任务不需要严格保证输出数值的准确性，只需要保证其数值的相对大小，分类结果的正确性，尤其是QAT生成的模型只能使用test_trt_precision.py来测试精度。

- 性能

  test_trt.py 可输出不同batchsize下模型的latency、throughput,在性能测试中，模型会预热20次，然后循环50次来降低系统误差。

- 测试代码

  （1）基本测试：python test_trt.py --trt_path [path of trt file]  --batch [batchsize of the model] 
--dynamic：the trt mode is dynamic; --ProfilerLayer:print every layer's latency ; --cudaGraph:use cuda grapg 

  - 例子： python test_trt.py --trt_path target/MobileViT_dynamic_opt_plugin_int8.trt --dynamic --cudaGraph

  （2）测试imagenet数据集精度（测试集大小为6GB，LMDB格式，若不方便可只进行基本测试）

  python test_trt_precision.py --trt_path [path of trt file]  --batch [batchsize of the model]  --data_path [test data path ]

  #例子：python test_trt_precision.py --trt_path target/MobileViT_dynamic_opt_plugin_int8.trt --data_path /target/test_data/val/val.lmdb --batch 4

- 测试结果

  - latency结果（ms）

   batch|base FP32	trt| FP32 trt|  FP16 trt	| INT8 trt| Plugin INT8 trt
   |---| ----- | ----- | ----- | ----- | ----- |
   1	   |1.577     |	1.223|	0.694	|0.956 |	0.665
   4	   |3.110     |	2.693|	1.354	|1.353 |	0.994
   8	   |5.602     |	5.082|	2.509 |2.362 |	1.813
  - 加速比 (base FP32)/(opt model)

   batch|base FP32	trt| FP32 trt|  FP16 trt	| INT8 trt| Plugin INT8 trt
   |---| ----- | ----- | ----- | ----- | ----- |
   1	   |1.0     |	1.289|	2.272 	| 1.650 |	 2.371
   4	   |1.0     |	1.155 |	2.297 	| 2.299 |	 3.129
   8	   |1.0     |	1.102 |	2.233   | 2.372 |	 3.090

  - imagenet正确率结果

   model	                                       |top1 acc（%）|	top5 acc（%）
   |---| ----- | ----- |
   pytorch base                                 |	68.036      | 	88.35
   FP32 trt                                     |	68.04       | 	88.362
   FP16 trt                                     |	68.044      | 	88.344
   INT8 trt (IInt8MinMaxCalibrator)             |	65.798      | 	86.664
   Plugin INT8 trt(QAT+IInt8EntropyCalibrator2) |	67.78       |	  88.108

