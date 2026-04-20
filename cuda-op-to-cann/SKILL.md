---
name: cuda-op-to-cann
description: Migrate CUDA operators, custom kernels, and PyTorch CUDA extensions to CANN on Ascend. Use when the user wants a CUDA op porting plan, ACLNN replacement analysis, Ascend C custom operator scaffolding, msopgen input generation, framework adapter stubs, or validation artifacts for Ascend NPU migration.
---

# CUDA Op to CANN

## Overview

This skill turns a fuzzy "please migrate this CUDA op to Ascend" request into a structured migration workflow. It inspects CUDA sources, chooses the safest migration route, generates starter artifacts, and makes unsupported or high-risk areas explicit instead of pretending the port is finished.

Default outcome: a migration starter package with `inspection.json`, `strategy.json`, `migration_report.md`, `manual_todos.md`, extracted operator signatures, optional `msopgen` spec, a realistic Ascend C project scaffold, framework adapter stubs, and validation test scaffolding.

## Cross-Agent Compatibility

The canonical source of truth is this directory. To make the same skill discoverable by Codex, Cursor, Claude Code, and OpenCode, sync compatibility entrypoints into the repo root:

```bash
python3 scripts/sync_agent_compat.py --root /path/to/repo
```

This generates:

- `.codex/skills/cuda-op-to-cann/SKILL.md`
- `.claude/skills/cuda-op-to-cann/SKILL.md`
- `.agents/skills/cuda-op-to-cann/SKILL.md`
- `.opencode/skills/cuda-op-to-cann/SKILL.md`
- `.cursor/rules/cuda-op-to-cann.mdc`

Those files are wrappers only. Keep edits in the canonical skill and its sibling `references/` and `scripts/`, then re-run the sync script. For tool-specific notes, see [references/agent-compatibility.md](references/agent-compatibility.md).

## When to Use

Use this skill when the request includes any of the following:

- CUDA source files such as `.cu`, `.cuh`, `.cpp`, `.cc`, `.h`, `.hpp`
- PyTorch CUDA extension projects with `TORCH_LIBRARY`, `PYBIND11_MODULE`, `AT_DISPATCH`, or custom launcher code
- Requests to replace CUDA kernels with `aclnn` calls, Ascend C custom operators, or CANN-compatible wrappers
- Requests for migration feasibility analysis, operator decomposition, validation planning, or "one-click" porting scaffolds

Do not over-promise full automatic conversion for:

- inline PTX, `asm`, WMMA or tensor-core-specific kernels
- heavy warp-level collectives, cooperative groups, or custom scheduling tricks
- irregular graph-style kernels, deep atomics contention, or tightly coupled multi-kernel pipelines

In these cases, still use the skill, but keep the result in "analysis plus starter package" mode.

## Workflow

1. Gather input context.
Confirm the source path, target framework, CANN version, SOC target, and whether custom operators are allowed. If those values are missing, make conservative assumptions and record them in the report.

2. Inspect the CUDA implementation.
Run `scripts/inspect_cuda_op.py` or `scripts/run_migration.py` to detect kernels, launcher code, registrations, dispatch macros, memory operations, and high-risk CUDA-only constructs.

3. Choose a migration route.
Use the inspection result plus the decision rules in [references/cuda-to-cann-patterns.md](references/cuda-to-cann-patterns.md), [references/unsupported-patterns.md](references/unsupported-patterns.md), and [references/official-ascend-sources.md](references/official-ascend-sources.md).

- Prefer `aclnn-direct` when the op appears to match a built-in CANN operator.
- Prefer `aclnn-composite` when the CUDA op can be decomposed into a short chain of built-in ops.
- Prefer `ascendc-custom` when the behavior is custom but still structurally regular.
- Escalate to `manual-high-risk` when the op depends on unsupported or architecture-specific CUDA techniques.

When the source project is PyTorch-specific, also decide whether the user needs:

- eager-only integration through OpPlugin or a direct `torch_npu` adapter
- graph-mode integration through TorchAir Meta registration
- both

4. Generate starter artifacts.
Produce the following under an output directory:

- `inspection.json`
- `strategy.json`
- `migration_report.md`
- `manual_todos.md`
- `generated/`
- `tests/`

When `ascendc-custom` is selected and custom operators are allowed, also generate:

- `generated/msopgen/<op_name>.json`
- `generated/ascendc_project/<OpType>/CMakeLists.txt`
- `generated/ascendc_project/<OpType>/op_host/*`
- `generated/ascendc_project/<OpType>/op_kernel/*`
- `generated/ascendc_project/<OpType>/framework/<framework>/*`

5. Keep the output honest.
If the tool cannot infer tensor shapes, attributes, layout assumptions, or workspace behavior, write a placeholder and record it in `manual_todos.md`. Never claim the operator is production ready when it is only scaffolded.

## Command Entry Points

Primary orchestrator:

```bash
python3 scripts/run_migration.py \
  --src /path/to/op \
  --framework pytorch \
  --soc Ascend910B \
  --cann 8.0 \
  --allow-custom \
  --force-custom \
  --output /tmp/cuda-op-port
```

Useful focused tools:

- `scripts/inspect_cuda_op.py`: source inspection and feature extraction
- `scripts/extract_op_signature.py`: infer operator name, wrapper, kernel, tensor inputs, outputs, and attrs
- `scripts/detect_migration_strategy.py`: route selection and rationale
- `scripts/build_msopgen_spec.py`: generate starter `msopgen` JSON
- `scripts/invoke_msopgen.py`: run official `msopgen` when the local CANN environment provides it
- `scripts/patch_msopgen_project.py`: harden generated `build.sh` and inject pattern-aware migration notes
- `scripts/remote_verify_msopgen.py`: use a local machine inventory file such as `machine.local.md` to run `msopgen` and compile verification on a remote Ascend host
- `scripts/generate_ascendc_project.py`: emit a realistic Ascend C project tree
- `scripts/rewrite_framework_adapter.py`: emit framework adapter stubs
- `scripts/generate_pytorch_integration.py`: emit OpPlugin YAML, OpApi starter, and TorchAir Meta starter files
- `scripts/generate_tests.py`: create smoke and compare harnesses

Optional remote verification when the local machine does not provide CANN:

```bash
python3 scripts/run_migration.py \
  --src /path/to/op \
  --framework pytorch \
  --soc Ascend910B2 \
  --cann 8.3.RC1 \
  --allow-custom \
  --force-custom \
  --machine-file ./machine.local.md \
  --machine-keyword 910B \
  --bootstrap-python-deps \
  --output /tmp/cuda-op-port
```

When a remote machine is configured, prefer the downloaded `generated/remote_msopgen_project` as the most realistic project variant. The skill also keeps `generated/project_variants.json` so later steps know which project was selected.

## Decision Rules

Use these fast heuristics before deeper editing:

- If the implementation mostly wraps well-known math or NN behavior, start with `aclnn-direct`.
- If the CUDA kernel is a short fused expression with regular indexing, try `aclnn-composite`.
- If the operator has custom math but regular tiling and limited synchronization, choose `ascendc-custom`.
- If you detect `asm`, `cooperative_groups`, WMMA, deep `atomic*`, or multiple tightly coupled launches, choose `manual-high-risk`.

When the choice is ambiguous, prefer the more conservative path and state why.

## References

Load only the files you need:

- [references/migration-playbook.md](references/migration-playbook.md): end-to-end process and output contract
- [references/cuda-to-cann-patterns.md](references/cuda-to-cann-patterns.md): common mapping patterns
- [references/unsupported-patterns.md](references/unsupported-patterns.md): stop signs and escalation guidance
- [references/pytorch-adapter-guide.md](references/pytorch-adapter-guide.md): PyTorch integration notes
- [references/cann-version-matrix.md](references/cann-version-matrix.md): version-sensitive assumptions
- [references/official-ascend-sources.md](references/official-ascend-sources.md): curated official docs and how to use them
- [references/agent-compatibility.md](references/agent-compatibility.md): Codex, Cursor, Claude Code, and OpenCode entrypoint layout

## Output Standards

Every migration run should leave behind:

- a concise summary of what was detected
- a named migration strategy with confidence
- a pattern family such as `elementwise-binary`, `fused-elementwise-activation`, or `normalization-like`
- a list of assumptions
- a list of manual follow-ups
- at least one validation artifact

When the framework is PyTorch, also prefer leaving behind:

- `generated/integration_plan.json`
- `generated/pytorch_integration/op_plugin_functions.yaml`
- `generated/pytorch_integration/<OpType>KernelOpApi.cpp`
- `generated/pytorch_integration/<op_name>_meta.py`

If the user asks for a full migration, use the generated files as a starting point and then continue implementing the missing pieces directly in the target project.
If the migration depends on version-specific CANN, OpPlugin, or TorchAir behavior, ground the next step in the official Ascend docs before patching code.
