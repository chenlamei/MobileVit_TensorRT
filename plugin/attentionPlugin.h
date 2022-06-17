#include <NvInfer.h>
#include <cassert>
#include <cublas_v2.h>
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

// +------- Plguin
// ---------------------------------------------------------------------------------
namespace {
static const char *PLUGIN_NAME{"Attention"};
static const char *PLUGIN_VERSION{"1"};
} // namespace
inline void check(cublasStatus_t ret, int line) {
  if (ret != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "cuBLAS Error: " << ret << ", line: " << line << std::endl;
  }
}

inline void check(cudaError_t ret, int line) {
  if (ret != cudaSuccess) {
    std::cerr << "CUDA Error: " << cudaGetErrorString(ret) << ", line: " << line
              << std::endl;
  }
}

#define CHECK(_x) check((_x), __LINE__)
namespace nvinfer1 {

// +------- Plugin body
// ----------------------------------------------------------------------------
class AttentionPlugin : public IPluginV2DynamicExt {
private:
  std::string name_;
  std::string namespace_;
  float scale_{1};
  cublasHandle_t handle_{nullptr};

public:
  AttentionPlugin(const std::string &name, const float scale)
      : name_(name), scale_(scale) {
    WHERE_AM_I();
    CHECK(cublasCreate(&handle_));
  }

  AttentionPlugin(const std::string &name, const void *buffer, size_t length)
      : name_(name) {
    WHERE_AM_I();
    const char *data = reinterpret_cast<const char *>(buffer);
    memcpy(&scale_, data, sizeof(scale_));
    CHECK(cublasCreate(&handle_));
  }

  AttentionPlugin() = delete;

  ~AttentionPlugin() { WHERE_AM_I(); }
  int initialize() noexcept override {
    WHERE_AM_I();
    return 0;
  }
  void terminate() noexcept override {
    WHERE_AM_I();
    return;
  }
  void destroy() noexcept override {
    WHERE_AM_I();
    CHECK(cublasDestroy(handle_));
  }

  size_t getSerializationSize() const noexcept override {
    WHERE_AM_I();
    return sizeof(scale_);
  }

  void serialize(void *buffer) const noexcept override {
    WHERE_AM_I();
    char *data = reinterpret_cast<char *>(buffer);
    memcpy(data, &scale_, sizeof(scale_));
  }

  IPluginV2DynamicExt *clone() const noexcept override {
    WHERE_AM_I();
    AttentionPlugin *p = new AttentionPlugin(name_, scale_);
    p->setPluginNamespace(namespace_.c_str());
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
    DimsExprs out_dim;
    out_dim.nbDims = 4;
    out_dim.d[0] = inputs[0].d[0];
    out_dim.d[1] = inputs[0].d[1];
    out_dim.d[2] = inputs[0].d[2];
    const IDimensionExpr *three = exprBuilder.constant(3);
    out_dim.d[3] = exprBuilder.operation(DimensionOperation::kCEIL_DIV,
                                         *inputs[0].d[3], *three);

    return out_dim;
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
      res = (inOut[pos].type == DataType::kFLOAT ||
             inOut[pos].type == DataType::kHALF);
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
    int att_num = inputs[0].dims.d[0] * inputs[0].dims.d[1] *
                  inputs[0].dims.d[2] * inputs[0].dims.d[2];
    // int v_num =inputs[0].dims.d[0]*inputs[0].dims.d[1] * inputs[0].dims.d[2]*
    // inputs[0].dims.d[3]/3;
    return (att_num) * sizeof(float);
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
}; // class AttentionPlugin

class AttentionPluginCreator : public IPluginCreator {
private:
  static PluginFieldCollection fc_;
  static std::vector<PluginField> attr_;
  std::string namespace_;

public:
  AttentionPluginCreator() {
    attr_.emplace_back(
        PluginField("scale", nullptr, PluginFieldType::kFLOAT32));
    fc_.nbFields = attr_.size();
    fc_.fields = attr_.data();
  }

  ~AttentionPluginCreator() {}

  IPluginV2 *createPlugin(const char *name,
                          const PluginFieldCollection *fc) noexcept override {
    float scale;
    WHERE_AM_I();
    for (int i = 0; i < fc->nbFields; i++) {
      PluginField field = fc->fields[i];
      std::string field_name(field.name);
      if (field_name.compare("scale") == 0) {
        // printf("find scale !!!\n");
        scale = *reinterpret_cast<const float *>(field.data);
        continue;
      }
    }
    // printf("scale create :%f \n",scale);
    return new AttentionPlugin(name, scale);
  }

  IPluginV2 *deserializePlugin(const char *name, const void *serialData,
                               size_t serialLength) noexcept override {
    return new AttentionPlugin(name, serialData, serialLength);
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
}; // class AttentionPluginCreator

} // namespace nvinfer1
