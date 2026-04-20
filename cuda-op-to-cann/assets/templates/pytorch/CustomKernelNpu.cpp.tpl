#include <torch/extension.h>

namespace custom_ops {

torch::Tensor {{op_name}}_npu(const torch::Tensor& x, const torch::Tensor& y) {
  // Replace this with aclnn calls or a custom Ascend C launcher.
  TORCH_CHECK(x.sizes() == y.sizes(), "TODO: replace with the true shape contract.");
  return x;
}

}  // namespace custom_ops

TORCH_LIBRARY_IMPL(custom_ops, PrivateUse1, m) {
  m.impl("{{class_name}}", custom_ops::{{op_name}}_npu);
}
