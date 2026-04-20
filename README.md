# CUDA Op to CANN

`cuda-op-to-cann` is a reusable skill and script bundle for migrating CUDA custom operators to CANN on Ascend. It focuses on practical migration scaffolding instead of pretending every CUDA kernel can be translated automatically.

The current project already supports:

- CUDA source inspection and operator signature extraction
- Strategy selection across `aclnn-direct`, `aclnn-composite`, `ascendc-custom`, and `manual-high-risk`
- `msOpGen` starter spec generation
- Ascend C project scaffolding
- PyTorch integration starter generation for OpPlugin and TorchAir Meta
- Optional remote `msopgen + build + package` verification through a user-provided machine inventory file
- Compatibility entrypoints for Codex, Cursor, Claude Code, and OpenCode

## Repository Layout

- [`cuda-op-to-cann/`](./cuda-op-to-cann): canonical skill source, scripts, templates, and references
- [`cuda-op-to-cann/SKILL.md`](./cuda-op-to-cann/SKILL.md): main skill entrypoint
- [`cuda-op-to-cann/scripts/`](./cuda-op-to-cann/scripts): migration pipeline and helper tools
- [`cuda-op-to-cann/references/`](./cuda-op-to-cann/references): migration playbooks and official Ascend notes
- [`_pattern_fixtures/`](./_pattern_fixtures): source-only sample CUDA projects used as local fixtures
- `./.codex`, `./.claude`, `./.opencode`, `./.agents`, `./.cursor`: generated compatibility entrypoints for different coding agents

Generated outputs and remote build artifacts are intentionally ignored by [`.gitignore`](./.gitignore).

## Quick Start

Run the main migration pipeline:

```bash
python3 cuda-op-to-cann/scripts/run_migration.py \
  --src /path/to/cuda/project \
  --framework pytorch \
  --soc Ascend910B \
  --cann 8.0 \
  --allow-custom \
  --output /tmp/cuda-op-port
```

If you want to verify `msopgen` and build packaging on a remote Ascend host, create your own local inventory file such as `machine.local.md`:

```bash
cat > machine.local.md <<'EOF'
机器：示例昇腾910B服务器
IP地址：YOUR_HOST_OR_IP
用户名：YOUR_USERNAME
密码：YOUR_PASSWORD
目录：/home/YOUR_USERNAME
EOF
```

Then run:

```bash
python3 cuda-op-to-cann/scripts/run_migration.py \
  --src /path/to/cuda/project \
  --framework pytorch \
  --soc Ascend910B \
  --cann 8.3.RC1 \
  --allow-custom \
  --force-custom \
  --machine-file ./machine.local.md \
  --machine-keyword 910B \
  --bootstrap-python-deps \
  --output /tmp/cuda-op-port
```

## Multi-Agent Compatibility

The canonical source of truth lives in `cuda-op-to-cann/`. If you update the skill, regenerate the agent-specific entrypoints:

```bash
python3 cuda-op-to-cann/scripts/sync_agent_compat.py --root "$(pwd)"
```

This refreshes:

- `.codex/skills/cuda-op-to-cann/SKILL.md`
- `.claude/skills/cuda-op-to-cann/SKILL.md`
- `.agents/skills/cuda-op-to-cann/SKILL.md`
- `.opencode/skills/cuda-op-to-cann/SKILL.md`
- `.cursor/rules/cuda-op-to-cann.mdc`

## Open-Source Readiness Notes

- Do not commit `machine.local.md` or any file that contains real host credentials.
- Do not commit generated `out*` directories from local or remote verification runs.
- Keep `cuda-op-to-cann/` as the canonical source of truth and regenerate agent-specific wrappers when needed.
