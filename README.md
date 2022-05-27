# mobileVit-TensorRT

# 原始模型
### 模型简介
- mobileVit时一种轻量型的视觉transformer，本项目中的模型应用在ImageNet分类任务中

- 模型网络具体说明可参考 https://mp.weixin.qq.com/s/OoXGZ5pHLMSPZjyriWYstA ，pytoch 实现源码为https://github.com/wilile26811249/MobileViT ，文献 https://arxiv.org/abs/2110.02178 

- 模型的整体结构，如下图所示,MobileViT 中的初始层是一个 3×3 的标准卷积，然后是 MobileNetv2（或 MV2）块和 MobileViT 块，激活函数为Swish。
- 
![Image_20220526110045](https://user-images.githubusercontent.com/106289938/170406914-d78b4042-a4bb-4732-902c-5b64dd9969f0.png)

### 模型优化的难点

- 网络转换后出现很多低效率的ForeignNode
- ![image](https://user-images.githubusercontent.com/106289938/170434773-e41dfcd6-531f-4423-8dcf-44ebf7b336b8.png)

- 动态网络转成TRT文件时会出现如下错误

python: /root/gpgpu/MachineLearning/myelin/src/compiler/optimizer/kqv_gemm_split.cpp:350: void myelin::ir::kqv_split_pattern_t::check_transpose(): Assertion `in_dims.size() == 3' failed.
  
![image](https://user-images.githubusercontent.com/106289938/170433167-e32e5cbe-af6d-49ae-82cc-d177d9133252.png)

- 动态网络转换后会生成大量shape操作相关节点

![image](https://user-images.githubusercontent.com/106289938/170435170-c58feb3f-74c1-40c4-9aa0-794b7e7d3c81.png)

