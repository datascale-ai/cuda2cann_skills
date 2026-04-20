#!/usr/bin/env python3
"""Generate a realistic Ascend C custom operator project scaffold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ascendc_templates import attr_cpp_param_type, render_host_cpp, render_kernel_cpp, render_tiling_header
from common import camel_case, normalize_soc_short, snake_case
from pytorch_integration_templates import (
    render_integration_manifest,
    render_opapi_cpp,
    render_opplugin_yaml,
    render_torchair_meta,
)


def normalize_kernel_soc(soc: str) -> str:
    lower = normalize_soc_short(soc).lower()
    if lower.startswith("ascend910b") or lower.startswith("ascend910a"):
        return "ascend910"
    return lower


def normalize_add_config_soc(soc: str) -> str:
    return normalize_kernel_soc(soc)


def make_dirs(root: Path, framework: str) -> dict[str, Path]:
    paths = {
        "root": root,
        "op_host": root / "op_host",
        "op_kernel": root / "op_kernel",
        "framework": root / "framework" / framework.lower(),
        "scripts": root / "scripts",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def render_root_cmake(vendor_name: str) -> str:
    return f"""cmake_minimum_required(VERSION 3.16.0)
project(opp)

set(vendor_name {vendor_name})
find_package(ASC REQUIRED HINTS ${{ASCEND_CANN_PACKAGE_PATH}}/compiler/tikcpp/ascendc_kernel_cmake)

set(package_name ${{vendor_name}})
npu_op_package(${{package_name}}
    TYPE RUN
    CONFIG
        ENABLE_SOURCE_PACKAGE True
        ENABLE_BINARY_PACKAGE True
        INSTALL_PATH ${{CMAKE_BINARY_DIR}}/
)

if(EXISTS ${{CMAKE_CURRENT_SOURCE_DIR}}/op_host)
  add_subdirectory(op_host)
endif()
if(EXISTS ${{CMAKE_CURRENT_SOURCE_DIR}}/op_kernel)
  add_subdirectory(op_kernel)
endif()
"""


def render_cmake_presets(project_name: str, soc: str) -> str:
    compute_soc = normalize_kernel_soc(soc)
    return """{
  "version": 3,
  "cmakeMinimumRequired": {
    "major": 3,
    "minor": 19,
    "patch": 0
  },
  "configurePresets": [
    {
      "name": "default",
      "displayName": "Default",
      "generator": "Unix Makefiles",
      "binaryDir": "${sourceDir}/build",
        "cacheVariables": {
        "ASCEND_CANN_PACKAGE_PATH": "/usr/local/Ascend/ascend-toolkit/latest",
        "ASCEND_COMPUTE_UNIT": "%s",
        "vendor_name": "custom_ops"
      }
    }
  ]
}
""" % compute_soc


def render_build_sh(soc: str) -> str:
    compute_soc = normalize_kernel_soc(soc)
    return """#!/usr/bin/env bash
set -eo pipefail

if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
  # Use the canonical toolkit environment when available.
  set +u
  source /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1 || true
  set -u
fi

if [ -z "${ASCEND_CANN_PACKAGE_PATH:-}" ] && [ -d /usr/local/Ascend/ascend-toolkit/latest ]; then
  export ASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/ascend-toolkit/latest
fi

for candidate in \
  /usr/local/cmake/cmake-*/bin \
  /opt/cmake/bin
do
  if [ -d "$candidate" ]; then
  export PATH="$candidate:$PATH"
  fi
done

generator="Unix Makefiles"
if command -v ninja >/dev/null 2>&1; then
  generator="Ninja"
fi

cmake -S . -B build -G "$generator" \
  -DASCEND_CANN_PACKAGE_PATH="${ASCEND_CANN_PACKAGE_PATH:-/usr/local/Ascend/ascend-toolkit/latest}" \
  -DASCEND_COMPUTE_UNIT="${ASCEND_COMPUTE_UNIT:-%s}" \
  -Dvendor_name="${vendor_name:-custom_ops}"
cmake --build build --target binary -j"$(nproc)"
if cmake --build build --target package -j"$(nproc)" >/dev/null 2>&1; then
  echo "Package target built successfully."
fi
""" % compute_soc


def render_host_cmake(op_type: str, snake: str) -> str:
    return f"""aux_source_directory(${{CMAKE_CURRENT_SOURCE_DIR}} ops_srcs)

npu_op_code_gen(
    SRC ${{ops_srcs}}
    PACKAGE ${{package_name}}
    OUT_DIR ${{ASCEND_AUTOGEN_PATH}}
)

file(GLOB autogen_aclnn_src ${{ASCEND_AUTOGEN_PATH}}/aclnn_*.cpp)
set_source_files_properties(${{autogen_aclnn_src}} PROPERTIES GENERATED TRUE)
npu_op_library(cust_opapi ACLNN
    ${{autogen_aclnn_src}}
)
target_compile_options(cust_opapi PRIVATE
    -fvisibility=hidden
)

file(GLOB proto_src ${{ASCEND_AUTOGEN_PATH}}/op_proto.cc)
set_source_files_properties(${{proto_src}} PROPERTIES GENERATED TRUE)
npu_op_library(cust_op_proto GRAPH
    ${{ops_srcs}}
    ${{proto_src}}
)
target_compile_options(cust_op_proto PRIVATE
    -fvisibility=hidden
)

file(GLOB fallback_src ${{ASCEND_AUTOGEN_PATH}}/fallback_*.cpp)
set_source_files_properties(${{fallback_src}} PROPERTIES GENERATED TRUE)
npu_op_library(cust_optiling TILING
    ${{ops_srcs}}
    ${{fallback_src}}
)
target_compile_options(cust_optiling PRIVATE
    -fvisibility=hidden
)

npu_op_package_add(${{package_name}}
    LIBRARY
        cust_optiling
        cust_opapi
        cust_op_proto
)
"""


def render_framework_adapter(signature: dict, strategy: str) -> str:
    op_type = signature["op_type"]
    op_name = signature["op_name_snake"]
    inputs = signature.get("inputs", [])
    attrs = signature.get("attrs", [])
    input_params = [f'const at::Tensor& {item["name"]}' for item in inputs]
    attr_params = [f'{attr_cpp_param_type(item)} {item["name"]}' for item in attrs]
    args = ", ".join(input_params + attr_params) or "const at::Tensor& x"
    anchor = inputs[0]["name"] if inputs else "x"
    return f"""#include <torch/extension.h>

namespace custom_ops {{

at::Tensor {op_name}KernelNpu({args}) {{
  // Generated for strategy: {strategy}
  // Replace this implementation with op_api::{op_name} or your custom op invocation.
  TORCH_CHECK({anchor}.defined(), "Expected the first tensor input to be defined.");
  return {anchor};
}}

}}  // namespace custom_ops

TORCH_LIBRARY_IMPL(custom_ops, PrivateUse1, m) {{
  m.impl("{op_type}", custom_ops::{op_name}KernelNpu);
}}
"""


def render_kernel_cmake(op_type: str, snake: str) -> str:
    kernel_dir = f"./{op_type}"
    return f"""npu_op_kernel_options(ascendc_kernels ALL OPTIONS --save-temp-files)
npu_op_kernel_sources(ascendc_kernels
    OP_TYPE {op_type}
    KERNEL_DIR {kernel_dir}
    COMPUTE_UNIT ${{ASCEND_COMPUTE_UNIT}}
    KERNEL_FILE {snake}.cpp
)
npu_op_kernel_library(ascendc_kernels
    SRC_BASE ${{CMAKE_SOURCE_DIR}}/op_kernel
    TILING_LIBRARY cust_optiling
)
npu_op_package_add(${{package_name}}
    LIBRARY ascendc_kernels
)
"""


def render_manifest(signature: dict, framework: str, strategy: str) -> dict:
    return {
        "strategy": strategy,
        "framework": framework,
        "op_type": signature["op_type"],
        "pattern_family": signature.get("pattern_family"),
        "kernel_function": signature.get("kernel_function"),
        "wrapper_function": signature.get("wrapper_function"),
        "inputs": signature.get("inputs", []),
        "outputs": signature.get("outputs", []),
        "attrs": signature.get("attrs", []),
    }


def generate_project(inspection: dict, framework: str, strategy: str, soc: str, output_dir: Path) -> None:
    signature = inspection.get("primary_signature", {})
    signature["pattern_family"] = inspection.get("pattern_family", "generic-custom")
    op_type = signature.get("op_type") or "CustomOp"
    snake = signature.get("op_name_snake") or snake_case(op_type)
    paths = make_dirs(output_dir / op_type, framework)
    paths["root"].joinpath("CMakeLists.txt").write_text(render_root_cmake("custom_ops"))
    paths["root"].joinpath("CMakePresets.json").write_text(render_cmake_presets(op_type, soc))
    build_sh = paths["root"].joinpath("build.sh")
    build_sh.write_text(render_build_sh(soc))
    build_sh.chmod(0o755)

    paths["op_host"].joinpath(f"{snake}_tiling.h").write_text(render_tiling_header(op_type, signature))
    paths["op_host"].joinpath(f"{snake}.cpp").write_text(render_host_cpp(op_type, signature, normalize_add_config_soc(soc)))
    paths["op_host"].joinpath("CMakeLists.txt").write_text(render_host_cmake(op_type, snake))
    kernel_subdir = paths["op_kernel"] / op_type
    kernel_subdir.mkdir(parents=True, exist_ok=True)
    kernel_subdir.joinpath(f"{snake}.cpp").write_text(render_kernel_cpp(signature))
    paths["op_kernel"].joinpath("CMakeLists.txt").write_text(render_kernel_cmake(op_type, snake))
    paths["framework"].joinpath(f"{op_type}KernelNpu.cpp").write_text(render_framework_adapter(signature, strategy))
    if framework.lower() == "pytorch":
        paths["framework"].joinpath("op_plugin_functions.yaml").write_text(render_opplugin_yaml(signature))
        paths["framework"].joinpath(f"{op_type}KernelOpApi.cpp").write_text(render_opapi_cpp(signature, None))
        paths["framework"].joinpath(f"{snake}_meta.py").write_text(render_torchair_meta(signature, None))
        paths["framework"].joinpath("integration_manifest.json").write_text(
            json.dumps(render_integration_manifest(signature, None), indent=2) + "\n"
        )
    paths["root"].joinpath("project_manifest.json").write_text(json.dumps(render_manifest(signature, framework, strategy), indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspection", required=True)
    parser.add_argument("--framework", default="pytorch")
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--soc", default="Ascend910B")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    inspection = json.loads(Path(args.inspection).read_text())
    generate_project(inspection, args.framework, args.strategy, args.soc, Path(args.output).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
