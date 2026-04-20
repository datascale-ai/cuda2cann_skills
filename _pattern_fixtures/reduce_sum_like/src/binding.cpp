#include <torch/extension.h>

torch::Tensor reduce_sum_like_cuda(const torch::Tensor& x);

TORCH_LIBRARY(custom_ops, m) {
  m.def("ReduceSumLike", &reduce_sum_like_cuda);
}
