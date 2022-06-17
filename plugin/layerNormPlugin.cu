#include "layerNormPlugin.h"
#include <numeric>
using namespace nvinfer1;

PluginFieldCollection LayerNormPluginCreator::fc_{};
std::vector<PluginField> LayerNormPluginCreator::attr_;

constexpr int kWarpSize = 32;
__inline__ __device__ void WarpReduceSum(float &local_data) {
#pragma unroll
  for (int mask = kWarpSize / 2; mask > 0; mask /= 2) {
    local_data += __shfl_down_sync(0xffffffff, local_data, mask, kWarpSize);
  }
}

__inline__ __device__ void BlockReduceMean(float &local_data, float *sum_shared,
                                           float *mean_result, int len) {
  float warp_sum;
  const int lid = threadIdx.x % kWarpSize;
  const int wid = threadIdx.x / kWarpSize;
  WarpReduceSum(local_data);
  __syncthreads();
  if (lid == 0) {
    sum_shared[wid] = local_data;
  }
  __syncthreads();

  if (wid == 0) {
    if (threadIdx.x < blockDim.x / kWarpSize) {
      warp_sum = sum_shared[lid];
    } else {
      warp_sum = 0.0f;
    }
  }
  __syncwarp();
  WarpReduceSum(warp_sum);

  if (threadIdx.x == 0) {
    *mean_result = warp_sum / len;
  }
  __syncthreads();
}

template <typename T>
__global__ void LayerNormKernel(const T *__restrict__ pInput,
                                const T *__restrict__ gamma,
                                const T *__restrict__ beta,
                                T *__restrict__ pOutput, const int ld) {
  const int tx = threadIdx.x, index = blockIdx.x * ld + threadIdx.x;
  __shared__ float sum_shared[kWarpSize];
  __shared__ float mean_result;
  __shared__ float var_result;
  float value = 0, value_bak = 0;
  if (tx < ld) {
    value = (float)__ldg(&pInput[index]);
    value_bak = value;
  }
  __syncwarp();
  BlockReduceMean(value, sum_shared, &mean_result, ld);
  if (tx < ld) {
    value = (value_bak - mean_result) * (value_bak - mean_result);
  }
  __syncwarp();
  BlockReduceMean(value, sum_shared, &var_result, ld);
  // if(threadIdx.x==0){ printf("block %d mean:%f var:%f
  // stride:%d\n",blockIdx.x,mean_result,var_result,blockDim.x/2);}
  if (tx < ld) {
    pOutput[index] = (T)((value_bak - mean_result) *
                             rsqrtf(var_result + 1e-5f) * (float)gamma[tx] +
                         (float)beta[tx]);
  }
}

template <typename T>
__global__ void LayerNormNaiveKernel(const T *__restrict__ pInput,
                                     const T *__restrict__ gamma,
                                     const T *__restrict__ beta,
                                     T *__restrict__ pOutput, const int ld) {
  const int tx = threadIdx.x, index = blockIdx.x * ld + threadIdx.x;
  __shared__ float temp[256];
  float value = 0;
  if (tx < ld) {
    value = (float)pInput[index];
    temp[tx] = value;
  } else {
    temp[tx] = 0;
  }
  __syncthreads();

  for (int stride = 128; stride >= 1; stride /= 2) {
    if (tx < stride) {
      temp[tx] += temp[tx + stride];
    }
    __syncthreads();
  }
  float mean = temp[0] / ld;
  __syncthreads();
  if (tx < ld) {
    temp[tx] = (value - mean) * (value - mean);
  } else {
    temp[tx] = 0;
  }
  __syncthreads();

  for (int stride = 128; stride >= 1; stride /= 2) {
    if (tx < stride) {
      temp[tx] += temp[tx + stride];
    }
    __syncthreads();
  }
  float var = temp[0] / ld;
  // if(threadIdx.x==0){ printf("block %d mean:%f var:%f
  // stride:%d\n",blockIdx.x,mean,var,blockDim.x/2);}
  if (tx < ld) {
    pOutput[index] =
        (T)((value - mean) * rsqrtf(var + 1e-5f) * (float)gamma[tx] +
            (float)beta[tx]);
  }
}
int64_t volume(nvinfer1::Dims const& d)
{
    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}
int32_t LayerNormPlugin::enqueue(const PluginTensorDesc *inputDesc,
                                 const PluginTensorDesc *outputDesc,
                                 const void *const *inputs,
                                 void *const *outputs, void *workspace,
                                 cudaStream_t stream) noexcept {
                                 
  int bolck_num =volume(inputDesc[0].dims)/n_;
  
  const int blocksize = ((n_ - 1) / 32 + 1) * 32;

  // printf("ld is %d bolck_num is %d blocksize is %d
  // \n",ld,bolck_num,blocksize);
  // ld should be 144 or 192 or 240
  if (inputDesc[0].type == DataType::kFLOAT) {
    LayerNormKernel<<<bolck_num, blocksize, 0, stream>>>(
        (float *)inputs[0], weight_gpu_, bias_gpu_, (float *)outputs[0], n_);
    // LayerNormNaiveKernel <<<bolck_num, 256, 0, stream>>>((float
    // *)inputs[0],weight_gpu_,bias_gpu_, (float *)outputs[0],ld);
  } else if (inputDesc[0].type == DataType::kHALF) {
    LayerNormKernel<<<bolck_num, blocksize, 0, stream>>>(
        (__half *)inputs[0], weight_half_gpu_, bias_half_gpu_,
        (__half *)outputs[0], n_);
    // LayerNormNaiveKernel <<<bolck_num,256 , 0, stream>>>((__half
    // *)inputs[0],weight_half_gpu_,bias_half_gpu_, (__half *)outputs[0],ld);
  } else {
    printf("Unsupport datatype!\n");
  }
  return 0;
}

REGISTER_TENSORRT_PLUGIN(LayerNormPluginCreator);
