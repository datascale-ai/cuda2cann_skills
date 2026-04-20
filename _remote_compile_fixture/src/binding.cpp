#include <torch/extension.h>
torch::Tensor fused_bias_relu_cuda(const torch::Tensor& x,
                                   const torch::Tensor& bias,
                                   const torch::Tensor& residual,
                                   const torch::Tensor& gamma,
                                   double alpha,
                                   bool fuse_residual);
TORCH_LIBRARY(custom_ops, m) {
  m.def("FusedBiasRelu", &fused_bias_relu_cuda);
}
