#include "attentionPlugin.h"
using namespace nvinfer1;

PluginFieldCollection AttentionPluginCreator::fc_{};
std::vector<PluginField> AttentionPluginCreator::attr_;

constexpr int kWarpSize = 32;
constexpr float INIT_MIN = -1e10;
template <typename T> __inline__ __device__ void WarpReduceSum(T &local_data) {
#pragma unroll
  for (int mask = 16; mask > 0; mask /= 2) {
    local_data += __shfl_down_sync(0xffffffff, local_data, mask, 32);
  }
}
// cuda10 does not support hmax
__inline__ __device__ __half __myhmax(__half a, __half b) {
  if (a > b)
    return a;
  else {
    return b;
  }
}

__inline__ __device__ void WarpReduceMax(__half &local_data) {
#pragma unroll
  for (int mask = 16; mask > 0; mask /= 2) {
    local_data = __myhmax(local_data,
                          __shfl_down_sync(0xffffffff, local_data, mask, 32));
  }
}
template <typename T> __inline__ __device__ void WarpReduceMax(T &local_data) {
#pragma unroll
  for (int mask = 16; mask > 0; mask /= 2) {
    local_data =
        max(local_data, __shfl_down_sync(0xffffffff, local_data, mask, 32));
  }
}

template <typename T>
__inline__ __device__ void BlockReduceSum(T &local_data, T *sum_shared,
                                          T *sum_result) {
  float warp_sum;
  const int lid = threadIdx.x & 0x1f;
  const int wid = threadIdx.x >> 5;
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
    *sum_result = warp_sum;
  }
  __syncthreads();
}
template <typename T>
__inline__ __device__ void BlockReduceMax(T &local_data, T *max_shared,
                                          T *max_result) {
  float warp_max;
  const int lid = threadIdx.x & 0x1f;
  const int wid = threadIdx.x >> 5;
  WarpReduceMax(local_data);
  __syncthreads();
  if (lid == 0) {
    max_shared[wid] = local_data;
  }
  __syncthreads();

  if (wid == 0) {
    if (threadIdx.x < blockDim.x / kWarpSize) {
      warp_max = max_shared[lid];
    } else {
      warp_max = INIT_MIN;
    }
  }
  __syncwarp();
  WarpReduceMax(warp_max);

  if (threadIdx.x == 0) {
    *max_result = warp_max;
  }
  __syncthreads();
}
// the shape of input must be (b,16,96) or (b,64,96) or (b,256,96);
// the shape of output must be (b,16,16) or (b,64,64) or (b,256,256);
// every block calcuate 8*8  of the result;
template <typename T>
__global__ void
matirxMulKernel(const T *__restrict__ pInput, T *__restrict__ pOutput,
                const int in_batch_size, const int out_batch_size,
                const int out_hw, const float scale) {

  __shared__ T A_shared[8][32];
  __shared__ T B_shared[8][32];
  int idy = blockIdx.y * blockDim.y +
            threadIdx.y; // 0-out_hw.blockDim.y=8 blockDim.x=32
  const int a_idx = blockIdx.z * in_batch_size + idy * 96 + threadIdx.x;
  const int b_idx = blockIdx.z * in_batch_size +
                    (blockIdx.x * 8 + threadIdx.y) * 96 + threadIdx.x + 32;
  A_shared[threadIdx.y][threadIdx.x] = pInput[a_idx];
  B_shared[threadIdx.y][threadIdx.x] = pInput[b_idx];
  __syncthreads();
  T C_local[8];
#pragma unroll
  for (int i = 0; i < 8; i++) {
    C_local[i] = A_shared[threadIdx.y][threadIdx.x] * B_shared[i][threadIdx.x];
    //__syncwarp();
    WarpReduceSum(C_local[i]);
  }
  const int out_idx =
      blockIdx.z * out_batch_size + idy * out_hw + blockIdx.x * 8;
  if (threadIdx.x == 0) {
#pragma unroll
    for (int i = 0; i < 8; i++) {
      pOutput[out_idx + i] = C_local[i] * (T)scale;
    }
  }
}

template <typename T>
__global__ void SoftmaxKernel(T *pInput, T *pOutput, const float scale,const int len) {
  __shared__ T tmp_shared[kWarpSize];
  __shared__ T max_result;
  __shared__ T sum_result;

  // Initialze data to be the smallest  number.
  T data = (T)INIT_MIN;
  const int idx = blockIdx.x * len + threadIdx.x;
  if (threadIdx.x < len) {
    data = __ldg(&pInput[idx])*(T)scale;
  }
  T data_bak = data;
  __syncwarp();
  // find max
  BlockReduceMax(data, tmp_shared, &max_result);
  // exp
  if (threadIdx.x < len) {
    data = (T)__expf(float(data_bak - max_result));
  } else {
    data = 0;
  }
  data_bak = data;
  __syncwarp();
  // exp sum
  BlockReduceSum(data, tmp_shared, &sum_result);
  if (threadIdx.x < len)
    pOutput[idx] = data_bak / sum_result;
}

int32_t AttentionPlugin::enqueue(const PluginTensorDesc *inputDesc,
                                 const PluginTensorDesc *outputDesc,
                                 const void *const *inputs,
                                 void *const *outputs, void *workspace,
                                 cudaStream_t stream) noexcept {
  // the shape of input must be (b,16,96) or (b,64,96) or (b,256,96);

  assert(inputDesc[0].dims.d[2] % 8 == 0);
  const int batch = inputDesc[0].dims.d[0] * inputDesc[0].dims.d[1];
  const int block_dim =
      inputDesc[0].dims.d[2] / 8; // every block calcuate 8*8  of the result;
  const int in_batch_size = inputDesc[0].dims.d[2] * inputDesc[0].dims.d[3];
  const int out_batch_size = inputDesc[0].dims.d[2] * inputDesc[0].dims.d[2];
  const int softmax_batch = batch * inputDesc[0].dims.d[2];
  int softmax_block = inputDesc[0].dims.d[2];
  if (softmax_block < 32)
    softmax_block = 32;
  // printf("scale:%f\n",scale_);
  CHECK(cublasSetStream(handle_, stream));
  cublasSetMathMode(handle_, CUBLAS_TENSOR_OP_MATH);
  const int k = inputDesc[0].dims.d[2];
  const int m = inputDesc[0].dims.d[2];
  const int n = inputDesc[0].dims.d[3] / 3;
  if (inputDesc[0].type == DataType::kFLOAT) {

    float *atten = (float *)workspace;
    
    //matirxMulKernel<<<{block_dim, block_dim, batch}, {32, 8, 1}, 0, stream>>>(\
        (float *)inputs[0], atten, in_batch_size, out_batch_size,\
        inputDesc[0].dims.d[2], scale_);
    const float alpha = 1.0f, beta = 0.0f;
    CHECK(cublasSgemmStridedBatched(
          handle_, CUBLAS_OP_T, CUBLAS_OP_N, m, m, 32, &alpha,
          (float *)inputs[0] + 32, inputDesc[0].dims.d[3], in_batch_size, (float *)inputs[0],
          inputDesc[0].dims.d[3], in_batch_size, &beta,atten, m, out_batch_size,
          batch));
    SoftmaxKernel<<<{softmax_batch, 1, 1}, {softmax_block, 1, 1}, 0, stream>>>(
        atten, atten,scale_, inputDesc[0].dims.d[2]);
    
    CHECK(cublasSgemmStridedBatched(
        handle_, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
        (float *)inputs[0] + 64, inputDesc[0].dims.d[3], in_batch_size, atten,
        k, out_batch_size, &beta, (float *)outputs[0], n, in_batch_size / 3,
        batch));
  } else if (inputDesc[0].type == DataType::kHALF) {

    __half *atten = (__half *)workspace;
    const __half alpha = 1.0f, beta = 0.0f;
    //matirxMulKernel<<<{block_dim, block_dim, batch}, {32, 8, 1}, 0, stream>>>(\
        (__half *)inputs[0], atten, in_batch_size, out_batch_size,\
        inputDesc[0].dims.d[2], scale_);
    CHECK(cublasHgemmStridedBatched(
          handle_, CUBLAS_OP_T, CUBLAS_OP_N, m, m, 32, &alpha,
          (__half *)inputs[0] + 32, inputDesc[0].dims.d[3], in_batch_size, (__half *)inputs[0],
          inputDesc[0].dims.d[3], in_batch_size, &beta,atten, m, out_batch_size,
          batch));
    SoftmaxKernel<<<{softmax_batch, 1, 1}, {softmax_block, 1, 1}, 0, stream>>>(
        atten, atten,scale_, inputDesc[0].dims.d[2]);
    
    CHECK(cublasHgemmStridedBatched(
        handle_, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
        (__half *)inputs[0] + 64, inputDesc[0].dims.d[3], in_batch_size, atten,
        k, out_batch_size, &beta, (__half *)outputs[0], n, in_batch_size / 3,
        batch));
  } else {
    printf("Unsupport datatype!\n");
  }
  return 0;
}

REGISTER_TENSORRT_PLUGIN(AttentionPluginCreator);
