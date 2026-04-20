#!/usr/bin/env python3
"""Render PyTorch integration starter files for Ascend custom ops."""

from __future__ import annotations

from ascendc_templates import attr_cpp_param_type
from common import camel_case


def _schema_type(item: dict) -> str:
    if item.get("kind") == "tensor":
        return "Tensor"
    attr_type = item.get("attr_type", "int")
    return {
        "float": "float",
        "bool": "bool",
        "int": "int",
    }.get(attr_type, "int")


def op_api_symbol(signature: dict) -> str:
    return f"aclnn{signature['op_type']}"


def pytorch_function_name(signature: dict) -> str:
    return f"npu_{signature['op_name_snake']}"


def schema_signature(signature: dict) -> str:
    params = [
        f"{_schema_type(item)} {item['name']}"
        for item in signature.get("inputs", []) + signature.get("attrs", [])
    ]
    outputs = signature.get("outputs", [])
    if len(outputs) <= 1:
        returns = "Tensor"
    else:
        returns = "(" + ", ".join("Tensor" for _ in outputs) + ")"
    return f"{pytorch_function_name(signature)}({', '.join(params)}) -> {returns}"


def cpp_signature(signature: dict) -> str:
    params = [
        f"const at::Tensor& {item['name']}" for item in signature.get("inputs", [])
    ]
    params.extend(
        f"{attr_cpp_param_type(item)} {item['name']}" for item in signature.get("attrs", [])
    )
    return ", ".join(params) or "const at::Tensor& x"


def _call_args(signature: dict, output_name: str = "result") -> str:
    names = [item["name"] for item in signature.get("inputs", []) + signature.get("attrs", [])]
    names.append(output_name)
    return ", ".join(names)


def render_opplugin_yaml(signature: dict) -> str:
    return f"""all_version: [v1.11, v2.0, v2.1, v2.2, v2.3, v2.4, v2.5]
official:
custom:
  - func: {schema_signature(signature)}
    op_api: all_version
"""


def render_opapi_cpp(signature: dict, integration_plan: dict | None) -> str:
    fn_name = pytorch_function_name(signature)
    op_api_name = op_api_symbol(signature)
    anchor = signature.get("inputs", [{}])[0].get("name", "x")
    eager_lane = ((integration_plan or {}).get("pytorch") or {}).get("eager_lane", "opplugin-opapi")
    return f"""#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {{
using npu_preparation = at_npu::native::OpPreparation;

at::Tensor {fn_name}({cpp_signature(signature)}) {{
  // Preferred eager lane selected by the migration skill: {eager_lane}
  // Replace the placeholder result allocation and EXEC_NPU_CMD arguments with the true shape contract.
  at::Tensor result = npu_preparation::apply_tensor_without_format({anchor});
  // Example official lane:
  // EXEC_NPU_CMD({op_api_name}, {_call_args(signature)});
  return result;
}}

}}  // namespace op_api
"""


def render_torchair_meta(signature: dict, integration_plan: dict | None) -> str:
    fn_name = pytorch_function_name(signature)
    anchor = signature.get("inputs", [{}])[0].get("name", "x")
    graph_needed = ((integration_plan or {}).get("pytorch") or {}).get(
        "graph_meta_required_if_user_needs_graph_mode", True
    )
    params = ", ".join(item["name"] for item in signature.get("inputs", []) + signature.get("attrs", [])) or "x"
    return f"""import torch
import torch_npu
from torch.library import impl
from torch_npu.op_plugin.meta._meta_registrations import m


@impl(m, "{fn_name}", "Meta")
def {fn_name}_meta({params}):
    # Graph-mode Meta registration required by the migration skill: {graph_needed}
    # Replace this with the true output shape and dtype inference if the op changes rank or dtype.
    return torch.empty_like({anchor})
"""


def render_integration_manifest(signature: dict, integration_plan: dict | None) -> dict:
    return {
        "schema": schema_signature(signature),
        "pytorch_function": pytorch_function_name(signature),
        "op_api_symbol": op_api_symbol(signature),
        "eager_lane": ((integration_plan or {}).get("pytorch") or {}).get("eager_lane"),
        "graph_meta_required_if_user_needs_graph_mode": ((integration_plan or {}).get("pytorch") or {}).get(
            "graph_meta_required_if_user_needs_graph_mode"
        ),
    }


def render_meta_readme(signature: dict) -> str:
    return f"""# PyTorch Integration Starters for {signature['op_type']}

- `op_plugin_functions.yaml`: starter custom schema entry for OpPlugin
- `{camel_case(signature['op_name_snake'])}KernelOpApi.cpp`: starter opapi adapter using `EXEC_NPU_CMD`
- `{signature['op_name_snake']}_meta.py`: TorchAir Meta registration starter for graph mode

Fill in the real output allocation, call arguments, and shape or dtype inference before treating this path as production ready.
"""
