# PyTorch Adapter Guide

## Scope

Use this guide when the source project is a PyTorch CUDA extension or custom operator package.

## Practical Notes

- Preserve the user-facing op signature first.
- Keep the original CUDA path available until numerical parity is established.
- Replace dispatch and registration incrementally.
- Separate framework binding code from operator implementation code.

## Official PyTorch Path

Use [official-ascend-sources.md](official-ascend-sources.md) when the target is `torch_npu`, OpPlugin, or TorchAir.

- OpPlugin is the official eager-mode adaptation path for PyTorch on Ascend.
- The public schema should stay aligned with Aten IR.
- Official docs describe two main implementation lanes inside OpPlugin:
  - `opapi`: route to `aclnn` or custom op API
  - `aclops`: route to GE-registered operators
- For custom ops, the YAML declaration lives in `op_plugin/config/op_plugin_functions.yaml`.

## Eager-Mode Checklist

- Keep the exported op schema stable.
- Decide whether the generated adapter should target `op_api::xx`, `acl_op::xx`, or a temporary `custom_ops::xx` stub.
- If the op is a packaged custom operator, record that runtime loading usually needs the deployed `op_api/lib` path in `LD_LIBRARY_PATH`.
- If the user needs backward support, record whether gradients are delegated to existing ops, a handwritten backward, or left as a follow-up.

## Graph-Mode Checklist

- If the user needs `torch.compile` or TorchAir, add a Meta registration task.
- Meta registration must be in place before `torch.compile`, otherwise FX tracing cannot infer output shape or dtype.
- Treat graph-mode support as a separate deliverable from eager-mode support; do not imply one automatically gives the other.

## First-Pass Deliverables

- one NPU adapter stub
- one registration stub
- one compare test against a CPU or CUDA oracle
- one list of unresolved shape, dtype, and layout assumptions
- when graph mode matters, one explicit Meta registration stub or TODO
