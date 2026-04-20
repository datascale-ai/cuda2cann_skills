#include <torch/extension.h>
#include <cuda_runtime.h>
__global__ void fused_bias_relu_kernel(const half* x, const half* bias, const half* residual, const half* gamma, half* out, int n, double alpha, bool fuse_residual) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {}
}
torch::Tensor fused_bias_relu_cuda(const torch::Tensor& x, const torch::Tensor& bias, const torch::Tensor& residual, const torch::Tensor& gamma, double alpha, bool fuse_residual) {
  auto out = torch::zeros_like(x);
  int n = x.numel();
  fused_bias_relu_kernel<<<(n + 255) / 256, 256>>>(nullptr, nullptr, nullptr, nullptr, nullptr, n, alpha, fuse_residual);
  return out;
}
