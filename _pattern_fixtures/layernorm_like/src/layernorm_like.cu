#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void layernorm_like_kernel(const half* x,
                                      const half* gamma,
                                      const half* beta,
                                      half* out,
                                      int n,
                                      double eps) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
  }
}

torch::Tensor layernorm_like_cuda(const torch::Tensor& x,
                                  const torch::Tensor& gamma,
                                  const torch::Tensor& beta,
                                  double eps) {
  auto out = torch::zeros_like(x);
  int n = static_cast<int>(x.numel());
  layernorm_like_kernel<<<(n + 255) / 256, 256>>>(nullptr, nullptr, nullptr, nullptr, n, eps);
  return out;
}
