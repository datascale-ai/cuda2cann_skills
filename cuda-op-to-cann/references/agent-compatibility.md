## Agent Compatibility

This skill is maintained canonically in:

- `cuda-op-to-cann/SKILL.md`
- `cuda-op-to-cann/references/*`
- `cuda-op-to-cann/scripts/*`

Do not hand-edit the compatibility mirrors unless you are debugging discovery.

## Target Layouts

Use `scripts/sync_agent_compat.py` to emit these entrypoints at the repo root:

- Codex: `.codex/skills/cuda-op-to-cann/SKILL.md`
- Claude Code: `.claude/skills/cuda-op-to-cann/SKILL.md`
- OpenCode native: `.opencode/skills/cuda-op-to-cann/SKILL.md`
- OpenCode compatible fallbacks: `.agents/skills/cuda-op-to-cann/SKILL.md` and `.claude/skills/cuda-op-to-cann/SKILL.md`
- Cursor: `.cursor/rules/cuda-op-to-cann.mdc`

The generated files are wrappers that point back to the canonical skill with repository-relative paths. This keeps the repo portable across machines and avoids baking local absolute paths into checked-in files.

## Sync Command

From the repository root:

```bash
python3 cuda-op-to-cann/scripts/sync_agent_compat.py --root "$(pwd)"
```

Or from inside the skill directory:

```bash
python3 scripts/sync_agent_compat.py --root /absolute/path/to/repo
```

## Maintenance Rules

- Update the canonical `SKILL.md` first.
- Add new references under `references/` and new helper logic under `scripts/`.
- Re-run the sync script after any canonical skill change.
- Prefer wrapper files over copied full instructions so the four agent families stay aligned.
