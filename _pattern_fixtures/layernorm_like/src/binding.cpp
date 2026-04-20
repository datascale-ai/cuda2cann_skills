#include <torch/extension.h>

torch::Tensor layernorm_like_cuda(const torch::Tensor& x,
                                  const torch::Tensor& gamma,
                                  const torch::Tensor& beta,
                                  double eps);

TORCH_LIBRARY(custom_ops, m) {
  m.def("LayerNormLike", &layernorm_like_cuda);
}
