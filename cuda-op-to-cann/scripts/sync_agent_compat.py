#!/usr/bin/env python3
"""Sync compatibility entrypoints for Codex, Claude Code, Cursor, and OpenCode."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path


def split_frontmatter(text: str) -> tuple[str, str]:
    if not text.startswith("---\n"):
        raise ValueError("Expected SKILL.md to start with YAML frontmatter.")
    end = text.find("\n---\n", 4)
    if end == -1:
        raise ValueError("Could not find the end of YAML frontmatter.")
    return text[4:end], text[end + 5 :]


def parse_field(frontmatter: str, name: str) -> str:
    match = re.search(rf"^{re.escape(name)}:\s*(.+)$", frontmatter, re.M)
    if not match:
        raise ValueError(f"Could not find {name} in frontmatter.")
    return match.group(1).strip().strip('"').strip("'")


def parse_description(frontmatter: str) -> str:
    return parse_field(frontmatter, "description")


def parse_name(frontmatter: str) -> str:
    return parse_field(frontmatter, "name")


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def relative_path(from_dir: Path, to_path: Path) -> str:
    return os.path.relpath(to_path, start=from_dir)


def render_skill_wrapper(
    name: str,
    description: str,
    canonical_skill_path: Path,
    skill_dir: Path,
    target_path: Path,
) -> str:
    base_dir = target_path.parent
    references_dir = skill_dir / "references"
    scripts_dir = skill_dir / "scripts"
    canonical_display = relative_path(base_dir, canonical_skill_path)
    reference_paths = [
        references_dir / "migration-playbook.md",
        references_dir / "cuda-to-cann-patterns.md",
        references_dir / "unsupported-patterns.md",
        references_dir / "pytorch-adapter-guide.md",
        references_dir / "cann-version-matrix.md",
        references_dir / "official-ascend-sources.md",
    ]
    script_paths = [
        scripts_dir / "run_migration.py",
        scripts_dir / "inspect_cuda_op.py",
        scripts_dir / "generate_pytorch_integration.py",
        scripts_dir / "remote_verify_msopgen.py",
    ]
    rel_reference_lines = "\n".join(f"- `{relative_path(base_dir, path)}`" for path in reference_paths)
    rel_script_lines = "\n".join(f"- `{relative_path(base_dir, path)}`" for path in script_paths)
    return f"""---
name: {name}
description: {description}
---

# CUDA Op to CANN

This is a compatibility wrapper for agents that discover project-local skills from tool-specific directories.

Canonical skill:

- `{canonical_display}`

Canonical references:

{rel_reference_lines}

Canonical scripts:

{rel_script_lines}

Use the canonical skill and its sibling files as the source of truth. The most important workflow is:

1. Inspect the CUDA project and extract signatures.
2. Check built-in `aclnn` coverage before committing to a custom operator.
3. If needed, generate `msOpGen + Ascend C` starter artifacts.
4. For PyTorch, distinguish eager OpPlugin integration from TorchAir graph-mode Meta registration.
5. Keep unsupported CUDA details explicit and record manual follow-ups.
"""


def render_cursor_rule(
    description: str,
    canonical_skill_path: Path,
    skill_dir: Path,
    target_path: Path,
) -> str:
    base_dir = target_path.parent
    references_dir = skill_dir / "references"
    scripts_dir = skill_dir / "scripts"
    canonical_display = relative_path(base_dir, canonical_skill_path)
    reference_paths = [
        references_dir / "migration-playbook.md",
        references_dir / "cuda-to-cann-patterns.md",
        references_dir / "unsupported-patterns.md",
        references_dir / "pytorch-adapter-guide.md",
        references_dir / "cann-version-matrix.md",
        references_dir / "official-ascend-sources.md",
    ]
    script_paths = [
        scripts_dir / "run_migration.py",
        scripts_dir / "inspect_cuda_op.py",
        scripts_dir / "generate_pytorch_integration.py",
        scripts_dir / "remote_verify_msopgen.py",
    ]
    rel_reference_lines = "\n".join(f"- `{relative_path(base_dir, path)}`" for path in reference_paths)
    rel_script_lines = "\n".join(f"- `{relative_path(base_dir, path)}`" for path in script_paths)
    return f"""---
description: {description}
alwaysApply: false
---
# CUDA Op to CANN

Use this rule when the task is to migrate CUDA operators, custom kernels, or PyTorch CUDA extensions to CANN on Ascend.

Canonical skill: `{canonical_display}`

Canonical references:

{rel_reference_lines}

Canonical scripts:

{rel_script_lines}

Load the canonical skill content and follow its workflow. The most important route-selection rules are:

- Prefer built-in `aclnn` when the behavior is already covered.
- Use `msOpGen + Ascend C` when the math is custom but structurally regular.
- For PyTorch, distinguish eager OpPlugin integration from TorchAir graph-mode Meta registration.
- Keep unsupported CUDA details explicit instead of pretending the port is production ready.
"""


def write_if_changed(path: Path, content: str) -> None:
    ensure_parent(path)
    if path.exists() and path.read_text() == content:
        return
    path.write_text(content)


def sync(root: Path) -> list[Path]:
    skill_dir = root / "cuda-op-to-cann"
    canonical = skill_dir / "SKILL.md"
    text = canonical.read_text()
    frontmatter, _body = split_frontmatter(text)
    name = parse_name(frontmatter)
    description = parse_description(frontmatter)

    targets = [
        root / ".codex" / "skills" / "cuda-op-to-cann" / "SKILL.md",
        root / ".claude" / "skills" / "cuda-op-to-cann" / "SKILL.md",
        root / ".agents" / "skills" / "cuda-op-to-cann" / "SKILL.md",
        root / ".opencode" / "skills" / "cuda-op-to-cann" / "SKILL.md",
    ]
    for path in targets:
        write_if_changed(path, render_skill_wrapper(name, description, canonical, skill_dir, path))

    cursor_rule = root / ".cursor" / "rules" / "cuda-op-to-cann.mdc"
    write_if_changed(cursor_rule, render_cursor_rule(description, canonical, skill_dir, cursor_rule))
    return targets + [cursor_rule]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=Path(__file__).resolve().parents[2], type=Path)
    args = parser.parse_args()
    paths = sync(args.root.resolve())
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
