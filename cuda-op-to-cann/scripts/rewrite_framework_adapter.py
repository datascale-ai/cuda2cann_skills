#!/usr/bin/env python3
"""Generate a framework adapter stub for the selected migration route."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ascendc_templates import attr_cpp_param_type


def render_pytorch_adapter(signature: dict, strategy: str) -> str:
    return render_pytorch_adapter_with_plan(signature, strategy, None)


def render_pytorch_adapter_with_plan(signature: dict, strategy: str, integration_plan: dict | None) -> str:
    op_name = signature["op_name_snake"]
    class_name = signature["op_type"]
    inputs = signature.get("inputs", [])
    attrs = signature.get("attrs", [])
    input_params = [f'const torch::Tensor& {item["name"]}' for item in inputs]
    attr_params = [f"{attr_cpp_param_type(item)} {item['name']}" for item in attrs]
    args = ", ".join(input_params + attr_params)
    call_args = ", ".join(item["name"] for item in inputs + attrs)
    pytorch_plan = (integration_plan or {}).get("pytorch", {})
    eager_lane = pytorch_plan.get("eager_lane", "opplugin-opapi")
    graph_meta_needed = pytorch_plan.get("graph_meta_required_if_user_needs_graph_mode", False)
    return f"""#include <torch/extension.h>

namespace custom_ops {{

torch::Tensor {op_name}_npu({args or "const torch::Tensor& x"}) {{
  // Strategy selected by the migration skill: {strategy}
  // Preferred PyTorch eager lane: {eager_lane}
  // Replace this stub with either aclnn calls or a packaged custom Ascend C launcher.
  // If torch.compile or TorchAir graph mode is required, add a Meta registration: {graph_meta_needed}.
  auto first = {inputs[0]["name"] if inputs else "x"};
  TORCH_CHECK(first.defined(), "Expected the first tensor input to be defined.");
  TORCH_CHECK(first.device().is_privateuseone() || first.device().type() == c10::DeviceType::CPU,
              "Update device checks for your target NPU runtime.");
  return first;
}}

}}  // namespace custom_ops

TORCH_LIBRARY_IMPL(custom_ops, PrivateUse1, m) {{
  m.impl("{class_name}", custom_ops::{op_name}_npu);
}}
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", required=True)
    parser.add_argument("--signature", required=True)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--integration-plan")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.framework.lower() != "pytorch":
        Path(args.output).write_text(
            f"// TODO: add adapter generation for framework={args.framework}\n"
        )
        return 0

    signature = json.loads(Path(args.signature).read_text())
    integration_plan = json.loads(Path(args.integration_plan).read_text()) if args.integration_plan else None
    Path(args.output).write_text(render_pytorch_adapter_with_plan(signature, args.strategy, integration_plan))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
