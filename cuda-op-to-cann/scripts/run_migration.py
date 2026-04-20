#!/usr/bin/env python3
"""End-to-end migration starter generator for CUDA operators."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from build_msopgen_spec import build_spec
from common import write_json
from detect_migration_strategy import detect_strategy
from generate_tests import write_compare_script, write_smoke_markdown
from inspect_cuda_op import inspect_sources
from invoke_msopgen import invoke_msopgen


def infer_op_name(src_values: list[str]) -> str:
    first = Path(src_values[0])
    stem = first.stem or first.name or "custom_op"
    stem = re.sub(r"[^a-zA-Z0-9_]+", "_", stem)
    return stem.lower().strip("_") or "custom_op"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def sync_preferred_project(source_dir: Path, dest_dir: Path) -> None:
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    shutil.copytree(source_dir, dest_dir, symlinks=True, ignore_dangling_symlinks=True)


def write_markdown_report(
    path: Path,
    op_name: str,
    framework: str,
    soc: str,
    cann_version: str,
    inspection: dict,
    strategy: dict,
    integration_plan: dict,
    assumptions: list[str],
) -> None:
    lines = [
        f"# Migration Report for {op_name}",
        "",
        "## Summary",
        "",
        f"- Framework: `{framework}`",
        f"- SOC: `{soc}`",
        f"- CANN version: `{cann_version}`",
        f"- Project root: `{inspection.get('project_root') or 'unknown'}`",
        f"- Likely source framework: `{inspection['likely_framework']}`",
        f"- Chosen strategy: `{strategy['strategy']}`",
        f"- Confidence: `{strategy['confidence']}`",
        "",
        "## Detected Signals",
        "",
        f"- Source files scanned: `{inspection['source_count']}`",
        f"- Build files detected: `{len(inspection.get('build_files', []))}`",
        f"- Kernel shape hint: `{inspection['kernel_shape']}`",
        f"- Pattern family: `{inspection.get('pattern_family', 'unknown')}`",
        f"- High-risk: `{inspection['high_risk']}`",
        f"- Dtypes: `{', '.join(inspection['dtypes'])}`",
        "",
        "## Reasons",
        "",
    ]
    lines.extend(f"- {reason}" for reason in strategy["reasons"])
    lines.extend(
        [
            "",
            "## Official Checks",
            "",
            f"- Check built-in aclnn first: `{integration_plan['should_check_aclnn_first']}`",
            f"- Built-in lane hint: `{integration_plan['builtin_lane_hint']}`",
        ]
    )
    lines.extend(f"- {item}" for item in integration_plan.get("notes", []))
    pytorch_plan = integration_plan.get("pytorch", {})
    if pytorch_plan.get("enabled"):
        lines.extend(
            [
                "",
                "## PyTorch Integration",
                "",
                f"- Eager lane: `{pytorch_plan['eager_lane']}`",
                f"- Graph-mode Meta required if needed: `{pytorch_plan['graph_meta_required_if_user_needs_graph_mode']}`",
            ]
        )
        lines.extend(f"- {item}" for item in pytorch_plan.get("notes", []))
    lines.extend(
        [
            "",
            "## Assumptions",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in assumptions)
    lines.extend(
        [
            "",
            "## Next Steps",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in strategy["next_steps"])
    path.write_text("\n".join(lines) + "\n")


def build_integration_plan(framework: str, inspection: dict, strategy: dict) -> dict:
    framework_lower = framework.lower()
    pattern_family = inspection.get("pattern_family", "unknown")
    likely_pytorch = framework_lower == "pytorch" or inspection.get("likely_framework") == "pytorch"
    high_risk = inspection.get("high_risk", False)
    strategy_name = strategy.get("strategy", "manual-high-risk")

    builtin_friendly_patterns = {
        "elementwise-unary",
        "elementwise-binary",
        "broadcast-binary",
        "fused-elementwise-activation",
        "reduction-like",
        "normalization-like",
        "matmul-like",
    }
    should_check_aclnn_first = not high_risk
    if strategy_name in {"aclnn-direct", "aclnn-composite"}:
        builtin_lane_hint = strategy_name
    elif pattern_family in builtin_friendly_patterns:
        builtin_lane_hint = "check-aclnn-before-custom"
    else:
        builtin_lane_hint = "custom-likely-after-builtin-check"

    notes = []
    if should_check_aclnn_first:
        notes.append("Use the official built-in aclnn path as the first check before expanding custom operator scope.")
    else:
        notes.append("High-risk CUDA features were detected, so builtin coverage checks may not be sufficient on their own.")
    if strategy_name == "ascendc-custom":
        notes.append("If builtin coverage is incomplete, generate an Ascend C custom operator that stays close to msOpGen packaging.")
    if strategy_name == "manual-high-risk":
        notes.append("Keep the original CUDA path as the behavior oracle while decomposing the port into smaller semantic units.")

    pytorch_plan = {
        "enabled": likely_pytorch,
        "eager_lane": "not-applicable",
        "graph_meta_required_if_user_needs_graph_mode": False,
        "notes": [],
    }
    if likely_pytorch:
        pytorch_plan["eager_lane"] = "opplugin-opapi"
        pytorch_plan["graph_meta_required_if_user_needs_graph_mode"] = True
        pytorch_plan["notes"] = [
            "Keep the public op schema aligned with Aten IR when generating PyTorch bindings.",
            "Prefer the OpPlugin opapi lane for aclnn-backed or packaged custom operators.",
            "Add TorchAir Meta registration when the user needs torch.compile or FX graph capture.",
        ]

    return {
        "should_check_aclnn_first": should_check_aclnn_first,
        "builtin_lane_hint": builtin_lane_hint,
        "notes": notes,
        "pytorch": pytorch_plan,
    }


def write_manual_todos(path: Path, inspection: dict, strategy: dict, integration_plan: dict) -> None:
    todos = [
        "Confirm tensor ranks, shapes, layouts, and broadcast rules.",
        "Confirm dtype support and any accumulation precision requirements.",
        "Confirm workspace behavior and temporary buffer ownership.",
        "Confirm stream semantics and synchronization requirements.",
    ]
    if strategy["strategy"] == "ascendc-custom":
        todos.extend(
            [
                "Implement shape inference and attribute parsing in host code.",
                "Design tiling strategy for Ascend C kernel implementation.",
                "Replace placeholder msopgen metadata before building.",
            ]
        )
    if inspection["high_risk"]:
        todos.append("Manually redesign high-risk CUDA-specific synchronization or PTX behavior.")
    if integration_plan.get("should_check_aclnn_first"):
        todos.append("Check official builtin aclnn coverage before locking in a custom operator implementation.")
    pytorch_plan = integration_plan.get("pytorch", {})
    if pytorch_plan.get("enabled"):
        todos.append("Decide whether the PyTorch eager path should land in OpPlugin opapi or stay as a temporary custom_ops stub.")
        if pytorch_plan.get("graph_meta_required_if_user_needs_graph_mode"):
            todos.append("If torch.compile or TorchAir graph mode is required, add a Meta registration task for shape and dtype inference.")
    path.write_text("\n".join(f"- {item}" for item in todos) + "\n")


def maybe_write_adapter(
    generated_dir: Path,
    framework: str,
    op_name: str,
    strategy_name: str,
    integration_plan_path: Path,
) -> None:
    if framework.lower() != "pytorch":
        return
    signature_path = generated_dir / "signature.json"
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("rewrite_framework_adapter.py")),
            "--framework",
            framework,
            "--signature",
            str(signature_path),
            "--strategy",
            strategy_name,
            "--integration-plan",
            str(integration_plan_path),
            "--output",
            str(generated_dir / f"{op_name}_kernel_npu.cpp"),
        ],
        check=True,
    )


def maybe_write_pytorch_integration(generated_dir: Path, framework: str, integration_plan_path: Path) -> None:
    if framework.lower() != "pytorch":
        return
    signature_path = generated_dir / "signature.json"
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("generate_pytorch_integration.py")),
            "--signature",
            str(signature_path),
            "--integration-plan",
            str(integration_plan_path),
            "--output-dir",
            str(generated_dir / "pytorch_integration"),
        ],
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", nargs="+", required=True, help="Source file(s) or directories")
    parser.add_argument("--framework", default="pytorch")
    parser.add_argument("--soc", default="Ascend910B")
    parser.add_argument("--cann", default="unknown")
    parser.add_argument("--op-name")
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-custom", action="store_true")
    parser.add_argument("--force-custom", action="store_true")
    parser.add_argument("--machine-file")
    parser.add_argument("--machine-keyword", default="910B")
    parser.add_argument("--remote-root")
    parser.add_argument("--bootstrap-python-deps", action="store_true")
    args = parser.parse_args()

    output_dir = ensure_dir(Path(args.output).expanduser().resolve())
    generated_dir = ensure_dir(output_dir / "generated")
    tests_dir = ensure_dir(output_dir / "tests")

    inspection = inspect_sources(args.src)
    signature = inspection.get("primary_signature", {})
    op_name = args.op_name or signature.get("op_name_snake") or infer_op_name(args.src)
    strategy = detect_strategy(inspection, allow_custom=args.allow_custom)
    if args.force_custom:
        strategy = {
            **strategy,
            "strategy": "ascendc-custom",
            "confidence": "user-forced",
            "reasons": strategy.get("reasons", []) + ["Custom operator generation was forced by `--force-custom`."],
            "next_steps": [
                "Generate msopgen input and Ascend C starter files.",
                "Implement shape inference, tiling, and kernel math in iterative passes.",
            ],
        }
    integration_plan = build_integration_plan(args.framework, inspection, strategy)

    assumptions = [
        f"Target framework is `{args.framework}`.",
        f"Target SOC is `{args.soc}`.",
        f"CANN version is `{args.cann}`.",
        "Tensor metadata and attributes still require project-specific confirmation.",
    ]

    write_json(output_dir / "inspection.json", inspection)
    write_json(output_dir / "strategy.json", strategy)
    write_json(generated_dir / "signature.json", signature)
    write_json(generated_dir / "integration_plan.json", integration_plan)
    write_markdown_report(
        output_dir / "migration_report.md",
        op_name,
        args.framework,
        args.soc,
        args.cann,
        inspection,
        strategy,
        integration_plan,
        assumptions,
    )
    write_manual_todos(output_dir / "manual_todos.md", inspection, strategy, integration_plan)

    if strategy["strategy"] == "ascendc-custom" and args.allow_custom:
        msopgen_dir = ensure_dir(generated_dir / "msopgen")
        spec = build_spec(inspection, op_name, args.framework, args.soc, args.cann)
        spec_path = msopgen_dir / f"{op_name}.json"
        write_json(spec_path, spec)
        msopgen_project_dir = generated_dir / "msopgen_project"
        if msopgen_project_dir.exists():
            shutil.rmtree(msopgen_project_dir)
        msopgen_result = invoke_msopgen(str(spec_path), args.framework, args.soc, str(msopgen_project_dir))
        write_json(generated_dir / "msopgen_result.json", msopgen_result)
        if msopgen_result.get("succeeded") and msopgen_result.get("project_root"):
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("patch_msopgen_project.py")),
                    "--project",
                    msopgen_result["project_root"],
                    "--inspection",
                    str(output_dir / "inspection.json"),
                    "--framework",
                    args.framework,
                    "--strategy",
                    strategy["strategy"],
                    "--output",
                    str(generated_dir / "msopgen_patch.json"),
                ],
                check=True,
            )

        remote_result = None
        remote_project_dir = generated_dir / "remote_msopgen_project"
        if args.machine_file:
            remote_args = [
                sys.executable,
                str(Path(__file__).with_name("remote_verify_msopgen.py")),
                "--spec",
                str(spec_path),
                "--inspection",
                str(output_dir / "inspection.json"),
                "--framework",
                args.framework,
                "--soc",
                args.soc,
                "--machine-file",
                args.machine_file,
                "--machine-keyword",
                args.machine_keyword,
                "--download-dir",
                str(remote_project_dir),
                "--output",
                str(generated_dir / "remote_verify.json"),
            ]
            if args.remote_root:
                remote_args.extend(["--remote-root", args.remote_root])
            if args.bootstrap_python_deps:
                remote_args.append("--bootstrap-python-deps")
            remote_proc = subprocess.run(remote_args, text=True)
            remote_result = json.loads((generated_dir / "remote_verify.json").read_text()) if (generated_dir / "remote_verify.json").exists() else None
            if remote_proc.returncode != 0 and remote_result is None:
                raise subprocess.CalledProcessError(remote_proc.returncode, remote_args)

        project_variants = {
            "local_msopgen": msopgen_result.get("project_root"),
            "remote_msopgen": remote_result.get("downloaded_project") if remote_result else None,
            "fallback": None,
            "preferred": None,
        }
        ascendc_project_dir = generated_dir / "ascendc_project"
        if remote_result and remote_result.get("downloaded_project"):
            sync_preferred_project(Path(remote_result["downloaded_project"]), ascendc_project_dir)
            project_variants["preferred"] = str(ascendc_project_dir)
        elif msopgen_result.get("succeeded") and msopgen_result.get("project_root"):
            sync_preferred_project(Path(msopgen_result["project_root"]), ascendc_project_dir)
            project_variants["preferred"] = str(ascendc_project_dir)
        else:
            fallback_dir = generated_dir / "fallback_ascendc_project"
            if fallback_dir.exists():
                shutil.rmtree(fallback_dir)
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("generate_ascendc_project.py")),
                    "--inspection",
                    str(output_dir / "inspection.json"),
                    "--framework",
                    args.framework,
                    "--strategy",
                    strategy["strategy"],
                    "--soc",
                    args.soc,
                    "--output",
                    str(fallback_dir),
                ],
                check=True,
            )
            sync_preferred_project(fallback_dir, ascendc_project_dir)
            project_variants["fallback"] = str(fallback_dir)
            project_variants["preferred"] = str(ascendc_project_dir)
        write_json(generated_dir / "project_variants.json", project_variants)

    maybe_write_adapter(
        generated_dir,
        args.framework,
        op_name,
        strategy["strategy"],
        generated_dir / "integration_plan.json",
    )
    maybe_write_pytorch_integration(generated_dir, args.framework, generated_dir / "integration_plan.json")
    write_compare_script(tests_dir / "compare_tensors.py", signature)
    write_smoke_markdown(tests_dir / "smoke_test.md", signature, strategy["strategy"], integration_plan)

    print(f"Wrote migration starter package to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
