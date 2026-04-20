#!/usr/bin/env python3
"""Generate PyTorch OpPlugin and TorchAir starter files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import write_json
from pytorch_integration_templates import (
    render_integration_manifest,
    render_meta_readme,
    render_opapi_cpp,
    render_opplugin_yaml,
    render_torchair_meta,
)


def generate_files(signature: dict, integration_plan: dict | None, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    op_name = signature["op_name_snake"]
    op_type = signature["op_type"]
    output_dir.joinpath("op_plugin_functions.yaml").write_text(render_opplugin_yaml(signature))
    output_dir.joinpath(f"{op_type}KernelOpApi.cpp").write_text(render_opapi_cpp(signature, integration_plan))
    output_dir.joinpath(f"{op_name}_meta.py").write_text(render_torchair_meta(signature, integration_plan))
    write_json(output_dir / "integration_manifest.json", render_integration_manifest(signature, integration_plan))
    output_dir.joinpath("README_SNIPPETS.md").write_text(render_meta_readme(signature))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signature", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--integration-plan")
    args = parser.parse_args()

    signature = json.loads(Path(args.signature).read_text())
    integration_plan = json.loads(Path(args.integration_plan).read_text()) if args.integration_plan else None
    generate_files(signature, integration_plan, Path(args.output_dir).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
