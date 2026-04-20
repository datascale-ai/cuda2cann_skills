#!/usr/bin/env python3
"""Generate validation scaffolding for a migrated operator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def write_compare_script(path: Path, signature: dict) -> None:
    op_name = signature["op_name_snake"]
    inputs = signature.get("inputs", [])
    attrs = signature.get("attrs", [])
    init_lines = []
    call_args = []
    for index, item in enumerate(inputs):
        init_lines.append(f'    {item["name"]} = torch.randn(2, 3, dtype=torch.float16)')
        call_args.append(item["name"])
    for item in attrs:
        value = "1" if item.get("attr_type") == "int" else "False"
        init_lines.append(f'    {item["name"]} = {value}')
        call_args.append(item["name"])
    compare_args = ", ".join(call_args)
    path.write_text(
        f"""#!/usr/bin/env python3
\"\"\"Numerical compare harness for {op_name}.\"\"\"

import torch


def reference_impl(x, y):
    raise NotImplementedError("Implement the CUDA or CPU reference for {op_name}.")


def migrated_impl(x, y):
    raise NotImplementedError("Call the migrated CANN path for {op_name}.")


def main():
{chr(10).join(init_lines) if init_lines else "    x = torch.randn(2, 3, dtype=torch.float16)"}
    ref = reference_impl({compare_args or "x"})
    out = migrated_impl({compare_args or "x"})
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    main()
"""
    )


def write_smoke_markdown(path: Path, signature: dict, strategy: str, integration_plan: dict | None = None) -> None:
    op_name = signature["op_name_snake"]
    extra_lines = []
    if integration_plan:
        extra_lines.append(f"- Check builtin aclnn first: `{integration_plan.get('should_check_aclnn_first')}`")
        pytorch_plan = integration_plan.get("pytorch", {})
        if pytorch_plan.get("enabled"):
            extra_lines.append(f"- Preferred PyTorch eager lane: `{pytorch_plan.get('eager_lane')}`")
            extra_lines.append(
                f"- Add Meta registration if graph mode is required: `{pytorch_plan.get('graph_meta_required_if_user_needs_graph_mode')}`"
            )
    path.write_text(
        f"""# Smoke Test Checklist for {op_name}

- Strategy: `{strategy}`
- Inputs inferred: `{len(signature.get("inputs", []))}`
- Outputs inferred: `{len(signature.get("outputs", []))}`
- Scalar attrs inferred: `{len(signature.get("attrs", []))}`
{chr(10).join(extra_lines) if extra_lines else ""}
- Replace placeholder tensor shapes with the real contract.
- Validate forward numerics first.
- Add dtype coverage for fp16, fp32, and bf16 when supported.
- Add shape coverage for scalar, broadcast, and edge-case dimensions.
- Record any unsupported behavior in `manual_todos.md`.
"""
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signature", required=True)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    signature = json.loads(Path(args.signature).read_text())
    write_compare_script(output_dir / "compare_tensors.py", signature)
    write_smoke_markdown(output_dir / "smoke_test.md", signature, args.strategy)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
