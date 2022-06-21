#include <NvInfer.h>
#include <cassert>
#include <cuda_fp16.h>
#include <iostream>
#include <string>
#include <vector>
// +------- Debug wrapper
// --------------------------------------------------------------------------
#if DEBUG
#define WHERE_AM_I()                                                           \
  do {                                                                         \
    printf("[%s]: this=->%p\n", __func__, this);                               \
  } while (0);
#else
#define WHERE_AM_I()
#endif // DEBUG

inline void check(cudaError_t ret, int line) {
  if (ret != cudaSuccess) {
    std::cerr << "CUDA Error: " << cudaGetErrorString(ret) << ", line: " << line
              << std::endl;
  }
}

#define CHECK(_x) check((_x), __LINE__)

__global__ void FP32ConvertToFP16Kernel(float *pInput, __half *pOutput,
                                        int num) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < num)
    pOutput[index] = __float2half(pInput[index]);
}
// +------- Plguin
// ---------------------------------------------------------------------------------
namespace {
static const char *PLUGIN_NAME{"LayerNorm"};
static const char *PLUGIN_VERSION{"1"};
} // namespace

namespace nvinfer1 {

// +------- Plugin body
// ----------------------------------------------------------------------------
class LayerNormPlugin : public IPluginV2DynamicExt {
private:
  std::string name_;
  std::string namespace_;
  int n_{256}; // max of n_ is 256
  float *weight_gpu_{nullptr};
  float *bias_gpu_{nullptr};

  __half *weight_half_gpu_{nullptr};
  __half *bias_half_gpu_{nullptr};
  nvinfer1::Weights weight_q_{DataType::kFLOAT, nullptr, 0};
  nvinfer1::Weights bias_q_{DataType::kFLOAT, nullptr, 0};

public:
  LayerNormPlugin(const std::string &name, Weights weight_q, Weights bias_q)
      : name_(name) {
    WHERE_AM_I();
    n_ = weight_q.count;
    assert(n_ < 256);
    weight_q_.type = DataType::kFLOAT;
    weight_q_.count = weight_q.count;
    size_t w_size = sizeof(float) * weight_q.count;
    weight_q_.values = malloc(w_size);
    memcpy(reinterpret_cast<char *>(const_cast<void *>(weight_q_.values)),
           weight_q.values, w_size);

    bias_q_.type = DataType::kFLOAT;
    bias_q_.count = bias_q.count;
    size_t b_size = sizeof(float) * bias_q.count;
    bias_q_.values = malloc(b_size);
    memcpy(reinterpret_cast<char *>(const_cast<void *>(bias_q_.values)),
           bias_q.values, b_size);
  }

  LayerNormPlugin(const std::string &name, const void *buffer, size_t length)
      : name_(name) {
    WHERE_AM_I();
    const char *data = reinterpret_cast<const char *>(buffer);
    size_t offset = 0;
    memcpy(&n_, data + offset, sizeof(n_));
    offset += sizeof(n_);
    weight_q_.type = DataType::kFLOAT;
    weight_q_.count = n_;
    size_t w_size = sizeof(float) * n_;
    weight_q_.values = malloc(w_size);
    memcpy(reinterpret_cast<char *>(const_cast<void *>(weight_q_.values)),
           data + offset, w_size);
    offset += w_size;

    bias_q_.type = DataType::kFLOAT;
    bias_q_.count = n_;
    size_t b_size = sizeof(float) * n_;
    bias_q_.values = malloc(b_size);
    memcpy(reinterpret_cast<char *>(const_cast<void *>(bias_q_.values)),
           data + offset, b_size);
  }

  LayerNormPlugin() = delete;

  ~LayerNormPlugin() { WHERE_AM_I(); }
  int initialize() noexcept override {
    WHERE_AM_I();
    size_t w_size = sizeof(float) * weight_q_.count;
    size_t b_size = sizeof(float) * bias_q_.count;
    CHECK(cudaMalloc((void **)&weight_gpu_, w_size));
    CHECK(cudaMalloc((void **)&bias_gpu_, b_size));
    CHECK(cudaMalloc((void **)&weight_half_gpu_, w_size / 2));
    CHECK(cudaMalloc((void **)&bias_half_gpu_, b_size / 2));
    // printf("weight data :%f %f
    // %d\n",*(float*)weight_q_.values,*((float*)weight_q_.values+weight_q_.count-1),n_);
    // printf("bias data :%f %f %d
    // \n",*(float*)bias_q_.values,*((float*)bias_q_.values+bias_q_.count-1),n_);
    CHECK(cudaMemcpy(weight_gpu_, weight_q_.values, w_size,
                     cudaMemcpyHostToDevice));
    CHECK(
        cudaMemcpy(bias_gpu_, bias_q_.values, b_size, cudaMemcpyHostToDevice));

    FP32ConvertToFP16Kernel<<<1, 256>>>(weight_gpu_, weight_half_gpu_,
                                        weight_q_.count);
    FP32ConvertToFP16Kernel<<<1, 256>>>(bias_gpu_, bias_half_gpu_,
                                        bias_q_.count);
    return 0;
  }
  void terminate() noexcept override {
    WHERE_AM_I();
    CHECK(cudaFree(weight_gpu_));
    CHECK(cudaFree(bias_gpu_));
    CHECK(cudaFree(weight_half_gpu_));
    CHECK(cudaFree(bias_half_gpu_));
    return;
  }
  void destroy() noexcept override {
    WHERE_AM_I();
    free(const_cast<void *>(weight_q_.values));
    free(const_cast<void *>(bias_q_.values));
  }

  size_t getSerializationSize() const noexcept override {
    WHERE_AM_I();
    return (sizeof(n_) + sizeof(float) * weight_q_.count +
            sizeof(float) * bias_q_.count);
  }

  void serialize(void *buffer) const noexcept override {
    WHERE_AM_I();
    char *data = reinterpret_cast<char *>(buffer);
    size_t offset = 0;
    memcpy(data + offset, &n_, sizeof(n_));
    offset += sizeof(n_);
    size_t w_size = sizeof(float) * n_;
    memcpy(data + offset, weight_q_.values, w_size);
    offset += w_size;
    size_t b_size = sizeof(float) * n_;
    memcpy(data + offset, bias_q_.values, b_size);
  }

  IPluginV2DynamicExt *clone() const noexcept override {
    WHERE_AM_I();
    LayerNormPlugin *p = new LayerNormPlugin(name_, weight_q_, bias_q_);
    p->setPluginNamespace(namespace_.c_str());
    p->weight_gpu_ = this->weight_gpu_;
    p->bias_gpu_ = this->bias_gpu_;
    p->weight_half_gpu_ = this->weight_half_gpu_;
    p->bias_half_gpu_ = this->bias_half_gpu_;
    return p;
  }

  int getNbOutputs() const noexcept override {
    WHERE_AM_I();
    return 1;
  }

  DimsExprs getOutputDimensions(int32_t outputIndex, const DimsExprs *inputs,
                                int32_t nbInputs,
                                IExprBuilder &exprBuilder) noexcept override {
    WHERE_AM_I();
    return inputs[0];
  }

  bool supportsFormatCombination(int32_t pos, const PluginTensorDesc *inOut,
                                 int32_t nbInputs,
                                 int32_t nbOutputs) noexcept override {
    WHERE_AM_I();
    if (inOut[pos].format != TensorFormat::kLINEAR) {
      return false;
    }

    bool res = false;
    switch (pos) {
    case 0:
      res =
          (inOut[pos].type == DataType::kFLOAT ||
           inOut[pos].type == DataType::kHALF ||
           (inOut[pos].type == DataType::kINT8 && inOut[pos].dims.nbDims >= 3));
      break;
    case 1:
      res = inOut[pos].type == inOut[0].type;
      break;
    default: // should NOT be here
      res = false;
    }
    return res;
  }

  DataType getOutputDataType(int outputIndex, const DataType *inputTypes,
                             int nbInputs) const noexcept override {
    WHERE_AM_I();
    return inputTypes[0];
  }

  void configurePlugin(const DynamicPluginTensorDesc *in, int32_t nbInputs,
                       const DynamicPluginTensorDesc *out,
                       int32_t nbOutputs) noexcept override {
    WHERE_AM_I();
  }

  size_t getWorkspaceSize(const PluginTensorDesc *inputs, int32_t nbInputs,
                          const PluginTensorDesc *outputs,
                          int32_t nbOutputs) const noexcept override {
    WHERE_AM_I();
    return 0;
  }

  void setPluginNamespace(const char *szNamespace) noexcept override {
    WHERE_AM_I();
    namespace_ = szNamespace;
  }
  const char *getPluginNamespace() const noexcept override {
    WHERE_AM_I();
    return namespace_.c_str();
  }
  const char *getPluginType() const noexcept override {
    WHERE_AM_I();
    return PLUGIN_NAME;
  }
  const char *getPluginVersion() const noexcept override {
    WHERE_AM_I();
    return PLUGIN_VERSION;
  }
  int32_t enqueue(const PluginTensorDesc *inputDesc,
                  const PluginTensorDesc *outputDesc, const void *const *inputs,
                  void *const *outputs, void *workspace,
                  cudaStream_t stream) noexcept override;
}; // class LayerNormPlugin

class LayerNormPluginCreator : public IPluginCreator {
private:
  static PluginFieldCollection fc_;
  static std::vector<PluginField> attr_;
  std::string namespace_;

public:
  LayerNormPluginCreator() {
    attr_.emplace_back(
        PluginField("weight", nullptr, PluginFieldType::kFLOAT32));
    attr_.emplace_back(PluginField("bias", nullptr, PluginFieldType::kFLOAT32));
    fc_.nbFields = attr_.size();
    fc_.fields = attr_.data();
  }

  ~LayerNormPluginCreator() {}

  IPluginV2 *createPlugin(const char *name,
                          const PluginFieldCollection *fc) noexcept override {
    WHERE_AM_I();
    nvinfer1::Weights w_q;
    nvinfer1::Weights b_q;
    for (int i = 0; i < fc->nbFields; i++) {
      PluginField field = fc->fields[i];
      std::string field_name(field.name);
      if (field_name.compare("weight") == 0) {
        w_q.values = field.data;
        w_q.count = field.length;
        w_q.type = DataType::kFLOAT;
        // continue;
      }
      if (field_name.compare("bias") == 0) {
        b_q.values = field.data;
        b_q.count = field.length;
        b_q.type = DataType::kFLOAT;
        // continue;
      }
    }
    return new LayerNormPlugin(name, w_q, b_q);
  }

  IPluginV2 *deserializePlugin(const char *name, const void *serialData,
                               size_t serialLength) noexcept override {
    return new LayerNormPlugin(name, serialData, serialLength);
  }

  void setPluginNamespace(const char *szNamespace) noexcept override {
    namespace_ = szNamespace;
  }

  const char *getPluginNamespace() const noexcept override {
    return namespace_.c_str();
  }

  const char *getPluginName() const noexcept override { return PLUGIN_NAME; }

  const char *getPluginVersion() const noexcept override {
    return PLUGIN_VERSION;
  }

  const PluginFieldCollection *getFieldNames() noexcept override {
    return &fc_;
  }
}; // class LayerNormPluginCreator

} // namespace nvinfer1