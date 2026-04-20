#!/usr/bin/env python3
"""Patch an msopgen-generated project for more reliable builds and clearer templates."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from ascendc_templates import assumed_expression, render_host_cpp, render_kernel_cpp, render_tiling_header
from pytorch_integration_templates import (
    render_integration_manifest,
    render_opapi_cpp,
    render_opplugin_yaml,
    render_torchair_meta,
)


def load_inspection(path: str) -> dict:
    return json.loads(Path(path).read_text())


def build_preamble() -> str:
    return """#!/usr/bin/env bash
set -eo pipefail

if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
  set +u
  source /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1 || true
fi

if [ -z "${ASCEND_HOME_PATH:-}" ] && [ -d /usr/local/Ascend/ascend-toolkit/latest ]; then
  export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
fi

if [ -z "${ASCEND_CANN_PACKAGE_PATH:-}" ] && [ -d /usr/local/Ascend/ascend-toolkit/latest ]; then
  export ASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/ascend-toolkit/latest
fi

for candidate in \\
  "${ASCEND_CANN_PACKAGE_PATH:-/usr/local/Ascend/ascend-toolkit/latest}/toolkit/toolchain/hcc/bin" \\
  /usr/local/Ascend/ascend-toolkit/latest/toolkit/toolchain/hcc/bin
do
  if [ -d "$candidate" ]; then
    export PATH="$candidate:$PATH"
  fi
done

for candidate in \\
  /usr/local/cmake/cmake-*/bin \\
  /opt/cmake/bin
do
  if [ -d "$candidate" ]; then
    export PATH="$candidate:$PATH"
  fi
done

append_path_var() {
  local var_name="$1"
  local value="$2"
  if [ -d "$value" ]; then
    local current="${!var_name:-}"
    case ":$current:" in
      *":$value:"*) ;;
      *) export "$var_name=${current:+$current:}$value" ;;
    esac
  fi
}

for include_dir in \\
  /usr/include/c++/11 \\
  /usr/include/aarch64-linux-gnu/c++/11 \\
  /usr/include/c++/11/backward \\
  /usr/lib/gcc/aarch64-linux-gnu/11/include \\
  /usr/include/aarch64-linux-gnu \\
  /usr/include \\
  "${ASCEND_CANN_PACKAGE_PATH:-/usr/local/Ascend/ascend-toolkit/latest}/toolkit/toolchain/hcc/aarch64-target-linux-gnu/include/c++/7.3.0" \\
  "${ASCEND_CANN_PACKAGE_PATH:-/usr/local/Ascend/ascend-toolkit/latest}/toolkit/toolchain/hcc/aarch64-target-linux-gnu/include/c++/7.3.0/aarch64-target-linux-gnu" \\
  "${ASCEND_CANN_PACKAGE_PATH:-/usr/local/Ascend/ascend-toolkit/latest}/toolkit/toolchain/hcc/aarch64-target-linux-gnu/include/c++/7.3.0/backward" \\
  "${ASCEND_CANN_PACKAGE_PATH:-/usr/local/Ascend/ascend-toolkit/latest}/toolkit/toolchain/hcc/aarch64-target-linux-gnu/include"
do
  append_path_var CPATH "$include_dir"
  append_path_var CPLUS_INCLUDE_PATH "$include_dir"
  append_path_var C_INCLUDE_PATH "$include_dir"
done
"""


def patch_build_script(project_root: Path) -> None:
    build_sh = project_root / "build.sh"
    if not build_sh.exists():
        return
    original = build_sh.read_text()
    marker = "source /usr/local/Ascend/ascend-toolkit/set_env.sh"
    if marker in original or "ASCEND_CANN_PACKAGE_PATH" in original:
        return
    lines = original.splitlines()
    body = "\n".join(lines[1:]) if lines and lines[0].startswith("#!") else original
    build_sh.write_text(build_preamble() + "\n" + body.lstrip() + "\n")
    build_sh.chmod(0o755)


def pattern_guidance(pattern_family: str, signature: dict) -> list[str]:
    attrs = ", ".join(item["name"] for item in signature.get("attrs", [])) or "none"
    inputs = ", ".join(item["name"] for item in signature.get("inputs", [])) or "x"
    output_name = (signature.get("outputs") or [{"name": "out"}])[0]["name"]
    table = {
        "elementwise-unary": [
            f"- Process `{inputs}` tile by tile and write `{output_name}` with one load/compute/store pass.",
            "- First implementation target: relu, sigmoid, tanh, clamp, cast, or other pure unary math.",
        ],
        "elementwise-binary": [
            f"- Keep `{inputs}` aligned in the same tile schedule and emit `{output_name}` directly.",
            "- If the CUDA source hid a broadcast, move the shape rule into tiling metadata instead of branching in the kernel body.",
        ],
        "broadcast-binary": [
            f"- Encode the broadcast rule for `{inputs}` in tiling data before touching the kernel loop.",
            "- Start with scalar or channel-wise broadcast, then extend to generic ND broadcast if needed.",
        ],
        "fused-elementwise": [
            f"- Reconstruct the fused expression over `{inputs}` in one pass and thread attrs `{attrs}` through tiling and kernel signatures.",
            "- Keep the first version numerically simple, then optimize the memory schedule after correctness is confirmed.",
        ],
        "fused-elementwise-activation": [
            f"- Suggested order: load primary inputs `{inputs}`, apply bias or residual fusion, then activation, then store `{output_name}`.",
            f"- Attrs to preserve while porting: `{attrs}`.",
        ],
        "reduction-like": [
            "- Separate partial reduction from final merge when the original CUDA path relied on atomics.",
            "- Make accumulation dtype, reduction axis, and output shape rules explicit in tiling metadata.",
        ],
        "normalization-like": [
            "- Split statistics and affine phases if that makes the port easier to validate.",
            "- Validate epsilon, accumulation precision, and gamma/beta broadcasting before tuning performance.",
        ],
        "matmul-like": [
            "- Re-check whether built-in aclnn matmul or batch matmul can absorb most behavior before committing to custom kernel math.",
            "- If custom code remains, record layout and transpose semantics in host-side attributes first.",
        ],
        "generic-custom": [
            "- Map the CUDA loop nest into explicit load/compute/store phases before adding optimization.",
            "- Keep synchronization, workspace, and precision assumptions visible in comments until verification is complete.",
        ],
    }
    return table.get(pattern_family, table["generic-custom"])


def detect_add_config_soc(project_root: Path, snake: str) -> str:
    host_cpp = project_root / "op_host" / f"{snake}.cpp"
    if not host_cpp.exists():
        return "ascend910b"
    match = re.search(r'AddConfig\("([^"]+)"\)', host_cpp.read_text())
    return match.group(1) if match else "ascend910b"


def patch_generated_sources(project_root: Path, inspection: dict) -> None:
    signature = inspection.get("primary_signature", {})
    signature["pattern_family"] = inspection.get("pattern_family", "generic-custom")
    op_type = signature.get("op_type") or "CustomOp"
    snake = signature.get("op_name_snake") or "custom_op"
    add_config_soc = detect_add_config_soc(project_root, snake)

    tiling_header = project_root / "op_host" / f"{snake}_tiling.h"
    host_cpp = project_root / "op_host" / f"{snake}.cpp"
    kernel_cpp = project_root / "op_kernel" / f"{snake}.cpp"

    if tiling_header.parent.exists():
        tiling_header.write_text(render_tiling_header(op_type, signature))
    if host_cpp.parent.exists():
        host_cpp.write_text(render_host_cpp(op_type, signature, add_config_soc))
    if kernel_cpp.parent.exists():
        kernel_cpp.write_text(render_kernel_cpp(signature))


def patch_pytorch_integration(project_root: Path, inspection: dict) -> None:
    signature = inspection.get("primary_signature", {})
    framework_dir = project_root / "framework" / "pytorch"
    framework_dir.mkdir(parents=True, exist_ok=True)
    snake = signature.get("op_name_snake") or "custom_op"
    op_type = signature.get("op_type") or "CustomOp"
    framework_dir.joinpath("op_plugin_functions.yaml").write_text(render_opplugin_yaml(signature))
    framework_dir.joinpath(f"{op_type}KernelOpApi.cpp").write_text(render_opapi_cpp(signature, None))
    framework_dir.joinpath(f"{snake}_meta.py").write_text(render_torchair_meta(signature, None))
    framework_dir.joinpath("integration_manifest.json").write_text(
        json.dumps(render_integration_manifest(signature, None), indent=2) + "\n"
    )


def write_notes(project_root: Path, inspection: dict, framework: str, strategy: str) -> None:
    signature = inspection.get("primary_signature", {})
    pattern_family = inspection.get("pattern_family", "generic-custom")
    assumed = assumed_expression(signature, pattern_family)
    notes = [
        f"# Migration Notes for {signature.get('op_type', 'CustomOp')}",
        "",
        f"- Strategy: `{strategy}`",
        f"- Framework: `{framework}`",
        f"- Pattern family: `{pattern_family}`",
        f"- Wrapper function: `{signature.get('wrapper_function') or 'unknown'}`",
        f"- Kernel function: `{signature.get('kernel_function') or 'unknown'}`",
        "",
        "## Porting Focus",
        "",
    ]
    if assumed:
        notes.extend(
            [
                "## First-Pass Assumption",
                "",
                f"- Current generated kernel assumes `{assumed}`.",
                "- This is a migration starter implementation, not a claim that the original CUDA math has been fully recovered.",
                "",
            ]
        )
    notes.extend(pattern_guidance(pattern_family, signature))
    notes.extend(
        [
            "",
            "## Validation Checklist",
            "",
            "- Confirm output shape and dtype inference.",
            "- Confirm dtype coverage and accumulation precision.",
            "- Confirm broadcast, reduction axis, and attribute semantics.",
            "- Confirm workspace and stream behavior before performance tuning.",
        ]
    )
    project_root.joinpath("MIGRATION_NOTES.md").write_text("\n".join(notes) + "\n")


def patch_project(project_root: Path, inspection: dict, framework: str, strategy: str) -> dict:
    patch_build_script(project_root)
    patch_generated_sources(project_root, inspection)
    if framework.lower() == "pytorch":
        patch_pytorch_integration(project_root, inspection)
    write_notes(project_root, inspection, framework, strategy)
    return {
        "project_root": str(project_root),
        "pattern_family": inspection.get("pattern_family", "generic-custom"),
        "patched": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--inspection", required=True)
    parser.add_argument("--framework", default="pytorch")
    parser.add_argument("--strategy", default="ascendc-custom")
    parser.add_argument("--output")
    args = parser.parse_args()

    result = patch_project(Path(args.project).resolve(), load_inspection(args.inspection), args.framework, args.strategy)
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(payload + "\n")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
