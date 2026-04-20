#!/usr/bin/env python3
"""Invoke msopgen when the local environment provides it."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from common import soc_candidates


def find_msopgen() -> str | None:
    direct = shutil.which("msopgen")
    if direct:
        return direct
    candidates = [
        "/usr/local/Ascend/ascend-toolkit/latest/bin/msopgen",
        "/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/bin/msopgen",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    return None


def detect_project_root(output_dir: str) -> str | None:
    root = Path(output_dir)
    if not root.exists():
        return None
    if root.joinpath("build.sh").exists():
        return str(root)
    candidates = sorted(path.parent for path in root.rglob("build.sh"))
    if candidates:
        return str(candidates[0])
    return None


def invoke_msopgen(spec_path: str, framework: str, soc: str, output_dir: str) -> dict:
    msopgen = find_msopgen()
    if not msopgen:
        return {"invoked": False, "reason": "msopgen not found"}

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    last_error = None
    for candidate_soc in soc_candidates(soc):
        cmd = [
            msopgen,
            "gen",
            "-i",
            spec_path,
            "-f",
            framework,
            "-c",
            f"ai_core-{candidate_soc}",
            "-lan",
            "cpp",
            "-out",
            output_dir,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0:
            return {
                "invoked": True,
                "succeeded": True,
                "msopgen": msopgen,
                "soc": candidate_soc,
                "project_root": detect_project_root(output_dir),
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            }
        last_error = {
            "invoked": True,
            "succeeded": False,
            "msopgen": msopgen,
            "soc": candidate_soc,
            "project_root": detect_project_root(output_dir),
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "returncode": proc.returncode,
        }
    return last_error or {"invoked": True, "succeeded": False, "reason": "unknown failure"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--framework", default="pytorch")
    parser.add_argument("--soc", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    result = invoke_msopgen(args.spec, args.framework, args.soc, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("succeeded") else 1


if __name__ == "__main__":
    raise SystemExit(main())
