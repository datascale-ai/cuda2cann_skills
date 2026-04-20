# Migration Playbook

## Goal

Turn a CUDA operator request into a truthful migration package for Ascend.

## Output Contract

Every run should leave behind:

- `inspection.json`
- `strategy.json`
- `migration_report.md`
- `manual_todos.md`
- `generated/signature.json`
- generated scaffolding under `generated/`
- validation artifacts under `tests/`

For `ascendc-custom`, generated scaffolding should include a project tree that resembles the official `msopgen` output:

- `CMakeLists.txt`
- `CMakePresets.json`
- `build.sh`
- `op_host/`
- `op_kernel/`
- `framework/<framework>/`

For PyTorch-facing migrations, generated scaffolding should also include starter framework artifacts that match the official integration split:

- OpPlugin schema starter
- OpApi adapter starter
- TorchAir Meta registration starter when graph mode may matter

## Route Selection

- `aclnn-direct`: use when the source mostly wraps a standard operator.
- `aclnn-composite`: use when the source is a short fused expression or reduction that can be decomposed.
- `ascendc-custom`: use when the behavior is custom but structurally regular enough for a custom operator.
- `manual-high-risk`: use when CUDA-specific tricks dominate the implementation.

## Official Decision Tree

Use the official docs in [official-ascend-sources.md](official-ascend-sources.md) to keep the route selection grounded:

1. Check whether a built-in `aclnn` operator already covers the behavior.
2. If not, check whether the CUDA op can be decomposed into a short chain of built-in `aclnn` calls.
3. If not, move to `msOpGen + Ascend C` and generate a custom operator starter package.
4. If the user needs PyTorch integration, decide whether eager mode only is enough or whether TorchAir graph mode is also required.

## Route-Specific Expectations

- `aclnn-direct`
  Produce a mapping note to the candidate built-in API and record the expected two-stage `GetWorkspaceSize + execute` call flow.
- `aclnn-composite`
  Record the decomposition order and any unresolved broadcast, dtype, or workspace assumptions.
- `ascendc-custom`
  Keep the project tree close to official `msOpGen` output and include deployment notes for `op_api`, `op_impl`, `op_tiling`, and `op_proto`.
- `manual-high-risk`
  Preserve the original CUDA code as the behavior oracle and reduce the scope to analysis plus starter artifacts.

## Non-Negotiables

- Keep unsupported details visible.
- Record assumptions explicitly.
- Prefer a smaller truthful scaffold over a larger misleading one.
- Preserve a path for numerical comparison against the original implementation.
- When the user expects framework integration, distinguish clearly between kernel porting, eager binding, and graph-mode registration.
