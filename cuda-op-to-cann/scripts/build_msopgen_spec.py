#!/usr/bin/env python3
"""Build a starter msopgen JSON spec from the inspection result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import camel_case, normalize_soc_short


def build_tensor_desc(param: dict) -> dict:
    supported_types = param.get("supported_types", ["fp16", "float", "int32"])
    return {
        "name": param["name"],
        "param_type": "required",
        "format": ["ND" for _ in supported_types],
        "type": supported_types,
    }


def build_attr_desc(param: dict) -> dict:
    desc = {
        "name": param["name"],
        "param_type": "required",
        "type": param.get("attr_type", "int"),
    }
    return desc


def build_spec(inspection: dict, op_name: str, framework: str, soc: str, cann_version: str) -> dict:
    signature = inspection.get("primary_signature", {})
    op_title = signature.get("op_type") or camel_case(op_name)
    compute_soc = normalize_soc_short(soc).lower()
    inputs = signature.get("inputs") or [{"name": "x", "supported_types": ["fp16", "float", "int32"]}]
    outputs = signature.get("outputs") or [{"name": "out", "supported_types": ["fp16", "float", "int32"]}]
    attrs = signature.get("attrs") or []
    return [
        {
            "op": op_title,
            "language": "cpp",
            "framework": framework,
            "cann_version": cann_version,
            "compute_unit": f"ai_core-{compute_soc}",
            "input_desc": [build_tensor_desc(param) for param in inputs],
            "output_desc": [build_tensor_desc(param) for param in outputs],
            "attr": [build_attr_desc(param) for param in attrs],
        }
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspection", required=True)
    parser.add_argument("--op-name", required=True)
    parser.add_argument("--framework", default="generic")
    parser.add_argument("--soc", default="Ascend910B")
    parser.add_argument("--cann", default="unknown")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    inspection = json.loads(Path(args.inspection).read_text())
    spec = build_spec(inspection, args.op_name, args.framework, args.soc, args.cann)
    Path(args.output).write_text(json.dumps(spec, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
