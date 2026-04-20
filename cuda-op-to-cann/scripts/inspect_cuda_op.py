#!/usr/bin/env python3
"""Inspect CUDA operator sources and extract migration-relevant signals."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from common import discover_sources
from extract_op_signature import extract_project_signature

PATTERNS = {
    "global_kernel": re.compile(r"\b__global__\b"),
    "device_code": re.compile(r"\b__device__\b"),
    "kernel_launch": re.compile(r"<<<[^>]+>>>"),
    "cuda_memcpy": re.compile(r"\bcudaMemcpy(?:Async)?\b"),
    "cuda_stream": re.compile(r"\bcudaStream(?:_t|\b)"),
    "cub": re.compile(r"\bcub::"),
    "thrust": re.compile(r"\bthrust::"),
    "cooperative_groups": re.compile(r"\bcooperative_groups\b"),
    "inline_ptx": re.compile(r"\basm\s*\("),
    "wmma": re.compile(r"\b(?:nvcuda::wmma|wmma::|mma_sync)\b"),
    "atomics": re.compile(r"\batomic(?:Add|Sub|Max|Min|CAS|Exch|Inc|Dec|And|Or|Xor)\b"),
    "at_dispatch": re.compile(r"\bAT_DISPATCH_[A-Z0-9_]+\b"),
    "torch_library": re.compile(r"\bTORCH_LIBRARY(?:_IMPL)?\b"),
    "pybind": re.compile(r"\bPYBIND11_MODULE\b"),
    "tensor_iterator": re.compile(r"\bTensorIterator\b"),
    "half": re.compile(r"\b(?:half|__half|at::Half)\b"),
    "bf16": re.compile(r"\b(?:bfloat16|BFloat16|__nv_bfloat16)\b"),
    "shared_memory": re.compile(r"\b__shared__\b"),
    "warp_intrinsics": re.compile(r"\b__(?:shfl|ballot|syncwarp|activemask)"),
}

HIGH_RISK_FEATURES = {
    "cooperative_groups",
    "inline_ptx",
    "wmma",
    "warp_intrinsics",
}
def classify_kernel_shape(counts: dict[str, int]) -> str:
    if counts["wmma"] or counts["inline_ptx"]:
        return "tensor-core-or-ptx"
    if counts["atomics"] and counts["warp_intrinsics"]:
        return "synchronization-heavy"
    if counts["tensor_iterator"]:
        return "tensor-iterator"
    if counts["cub"] or counts["thrust"]:
        return "library-assisted"
    if counts["kernel_launch"] and not counts["atomics"]:
        return "regular-launch"
    return "unknown"


def infer_pattern_family(primary_signature: dict, counts: dict[str, int]) -> str:
    op_name = (primary_signature.get("op_name") or "").lower()
    input_count = len(primary_signature.get("inputs", []))
    attr_count = len(primary_signature.get("attrs", []))
    if any(token in op_name for token in ("layernorm", "rmsnorm", "batchnorm", "groupnorm", "norm")):
        return "normalization-like"
    if any(token in op_name for token in ("matmul", "gemm", "linear", "bmm", "mm")):
        return "matmul-like"
    if any(token in op_name for token in ("reduce", "sum", "mean", "max", "min", "argmax", "argmin")) or counts["atomics"]:
        return "reduction-like"
    if any(token in op_name for token in ("broadcast", "where", "expand")) or counts["tensor_iterator"]:
        return "broadcast-binary"
    if any(token in op_name for token in ("relu", "gelu", "silu", "sigmoid", "swish", "tanh")) and input_count >= 2:
        return "fused-elementwise-activation"
    if input_count >= 2 and attr_count:
        return "fused-elementwise"
    if input_count >= 2:
        return "elementwise-binary"
    if input_count == 1:
        return "elementwise-unary"
    return "generic-custom"


def inspect_sources(src_paths: Iterable[str]) -> dict:
    files = discover_sources(src_paths)
    counts = {key: 0 for key in PATTERNS}
    hits: dict[str, list[str]] = {key: [] for key in PATTERNS}

    for file_path in files:
        content = file_path.read_text(errors="ignore")
        for name, pattern in PATTERNS.items():
            matched = len(pattern.findall(content))
            if matched:
                counts[name] += matched
                hits[name].append(str(file_path))

    risks = []
    for feature in sorted(HIGH_RISK_FEATURES):
        if counts[feature]:
            risks.append(f"{feature} detected")
    if counts["atomics"] >= 4:
        risks.append("heavy atomic usage")
    if counts["kernel_launch"] >= 3:
        risks.append("multi-launch workflow")

    likely_framework = "generic"
    if counts["torch_library"] or counts["pybind"] or counts["at_dispatch"]:
        likely_framework = "pytorch"

    dtypes = []
    if counts["half"]:
        dtypes.append("fp16")
    if counts["bf16"]:
        dtypes.append("bf16")
    if not dtypes:
        dtypes.append("unknown")

    signature = extract_project_signature(list(src_paths))
    summary = {
        "source_count": len(files),
        "files": [str(path) for path in files],
        "counts": counts,
        "hits": hits,
        "likely_framework": likely_framework,
        "kernel_shape": classify_kernel_shape(counts),
        "high_risk": bool(risks),
        "risks": risks,
        "dtypes": dtypes,
        "build_files": signature["build_files"],
        "project_root": signature["project_root"],
        "primary_signature": signature["primary_signature"],
        "pattern_family": infer_pattern_family(signature["primary_signature"], counts),
        "signatures": {
            "kernels": signature["kernels"],
            "registrations": signature["registrations"],
            "functions": signature["functions"],
        },
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", nargs="+", required=True, help="Source files or directories")
    parser.add_argument("--output", help="Optional path to write JSON")
    args = parser.parse_args()

    inspection = inspect_sources(args.src)
    payload = json.dumps(inspection, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(payload + "\n")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
