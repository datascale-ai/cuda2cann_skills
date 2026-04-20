#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void reduce_sum_like_kernel(const float* x, float* out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    atomicAdd(out, x[idx]);
  }
}

torch::Tensor reduce_sum_like_cuda(const torch::Tensor& x) {
  auto out = torch::zeros({1}, x.options());
  int n = static_cast<int>(x.numel());
  reduce_sum_like_kernel<<<(n + 255) / 256, 256>>>(nullptr, nullptr, n);
  return out;
}
