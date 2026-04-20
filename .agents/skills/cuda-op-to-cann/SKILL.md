---
name: cuda-op-to-cann
description: Migrate CUDA operators, custom kernels, and PyTorch CUDA extensions to CANN on Ascend. Use when the user wants a CUDA op porting plan, ACLNN replacement analysis, Ascend C custom operator scaffolding, msopgen input generation, framework adapter stubs, or validation artifacts for Ascend NPU migration.
---

# CUDA Op to CANN

This is a compatibility wrapper for agents that discover project-local skills from tool-specific directories.

Canonical skill:

- `../../../cuda-op-to-cann/SKILL.md`

Canonical references:

- `../../../cuda-op-to-cann/references/migration-playbook.md`
- `../../../cuda-op-to-cann/references/cuda-to-cann-patterns.md`
- `../../../cuda-op-to-cann/references/unsupported-patterns.md`
- `../../../cuda-op-to-cann/references/pytorch-adapter-guide.md`
- `../../../cuda-op-to-cann/references/cann-version-matrix.md`
- `../../../cuda-op-to-cann/references/official-ascend-sources.md`

Canonical scripts:

- `../../../cuda-op-to-cann/scripts/run_migration.py`
- `../../../cuda-op-to-cann/scripts/inspect_cuda_op.py`
- `../../../cuda-op-to-cann/scripts/generate_pytorch_integration.py`
- `../../../cuda-op-to-cann/scripts/remote_verify_msopgen.py`

Use the canonical skill and its sibling files as the source of truth. The most important workflow is:

1. Inspect the CUDA project and extract signatures.
2. Check built-in `aclnn` coverage before committing to a custom operator.
3. If needed, generate `msOpGen + Ascend C` starter artifacts.
4. For PyTorch, distinguish eager OpPlugin integration from TorchAir graph-mode Meta registration.
5. Keep unsupported CUDA details explicit and record manual follow-ups.
