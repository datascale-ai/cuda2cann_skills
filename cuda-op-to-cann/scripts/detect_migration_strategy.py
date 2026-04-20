#!/usr/bin/env python3
"""Choose a migration strategy from an inspection result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_inspection(path: str) -> dict:
    return json.loads(Path(path).read_text())


def detect_strategy(inspection: dict, allow_custom: bool = False) -> dict:
    counts = inspection.get("counts", {})
    signature = inspection.get("primary_signature", {})
    pattern_family = inspection.get("pattern_family", "generic-custom")
    reasons: list[str] = []
    next_steps: list[str] = []

    if inspection.get("high_risk"):
        strategy = "manual-high-risk"
        confidence = "high"
        reasons.extend(inspection.get("risks", []))
        next_steps.extend(
            [
                "Preserve the original CUDA implementation as a behavior oracle.",
                "Decompose the operator into smaller semantic units before porting.",
                "Validate synchronization, precision, and numerical stability manually.",
            ]
        )
    elif counts.get("global_kernel", 0) == 0 and counts.get("kernel_launch", 0) == 0:
        strategy = "aclnn-direct"
        confidence = "medium"
        reasons.append("No explicit CUDA kernel launch found; wrapper-level replacement is plausible.")
        next_steps.extend(
            [
                "Search for an equivalent built-in aclnn operator first.",
                "Replace the CUDA dispatch path with a native CANN call or adapter.",
            ]
        )
    elif (
        pattern_family in {"elementwise-unary", "elementwise-binary", "broadcast-binary", "fused-elementwise-activation"}
        and counts.get("atomics", 0) == 0
        and counts.get("kernel_launch", 0) <= 1
        and len(signature.get("inputs", [])) <= 4
    ):
        strategy = "aclnn-composite"
        confidence = "medium"
        reasons.append(f"Pattern family `{pattern_family}` looks compatible with a composable aclnn implementation.")
        next_steps.extend(
            [
                "Map the kernel into one or more built-in aclnn ops.",
                "Keep a fallback path for any shape or dtype behavior that is still unclear.",
            ]
        )
    elif allow_custom:
        strategy = "ascendc-custom"
        confidence = "medium"
        reasons.append("Custom kernel behavior detected; generate an Ascend C starter package.")
        next_steps.extend(
            [
                "Generate msopgen input and Ascend C starter files.",
                "Implement shape inference, tiling, and kernel math in iterative passes.",
            ]
        )
    else:
        strategy = "manual-high-risk"
        confidence = "medium"
        reasons.append("Custom behavior detected but custom operator generation is disabled.")
        next_steps.extend(
            [
                "Re-run with --allow-custom to scaffold an Ascend C path.",
                "Otherwise, redesign the op in terms of built-in operators.",
            ]
        )

    return {
        "strategy": strategy,
        "confidence": confidence,
        "reasons": reasons,
        "next_steps": next_steps,
        "likely_framework": inspection.get("likely_framework", "generic"),
        "kernel_shape": inspection.get("kernel_shape", "unknown"),
        "op_type": signature.get("op_type"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspection", required=True, help="Path to inspection JSON")
    parser.add_argument("--allow-custom", action="store_true", help="Allow Ascend C custom path")
    parser.add_argument("--output", help="Optional path to write JSON")
    args = parser.parse_args()

    strategy = detect_strategy(load_inspection(args.inspection), allow_custom=args.allow_custom)
    payload = json.dumps(strategy, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(payload + "\n")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
