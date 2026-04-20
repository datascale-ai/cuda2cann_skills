#!/usr/bin/env python3
"""Shared Ascend C template renderers for generated and patched projects."""

from __future__ import annotations

from common import camel_case, snake_case


def lower_camel_case(name: str) -> str:
    rendered = camel_case(name)
    if not rendered:
        return name
    return rendered[0].lower() + rendered[1:]


def ge_dtype_expr(dtype: str) -> str:
    normalized = dtype.lower()
    table = {
        "fp16": "ge::DT_FLOAT16",
        "bf16": "ge::DT_BF16",
        "float": "ge::DT_FLOAT",
        "double": "ge::DT_DOUBLE",
        "int32": "ge::DT_INT32",
        "int64": "ge::DT_INT64",
        "bool": "ge::DT_BOOL",
    }
    return table.get(normalized, "ge::DT_FLOAT16")


def attr_cpp_param_type(item: dict) -> str:
    cpp_type = (item.get("cpp_type") or "").strip()
    if cpp_type:
        return cpp_type.replace("const ", "").replace("&", "").strip()
    attr_type = item.get("attr_type", "int").lower()
    return {
        "float": "double",
        "bool": "bool",
        "int": "int64_t",
    }.get(attr_type, "int64_t")


def attr_field_name(item: dict) -> str:
    return lower_camel_case(item["name"])


def attr_field_type(item: dict) -> str:
    attr_type = item.get("attr_type", "int").lower()
    return {
        "float": "float",
        "bool": "uint32_t",
        "int": "int64_t",
    }.get(attr_type, "int64_t")


def attr_field_default(item: dict) -> str:
    attr_type = item.get("attr_type", "int").lower()
    return {
        "float": "0.0f",
        "bool": "0U",
        "int": "0",
    }.get(attr_type, "0")


def _supported_types(items: list[dict]) -> list[str]:
    found: list[str] = []
    for item in items:
        for dtype in item.get("supported_types", []):
            if dtype not in found:
                found.append(dtype)
    return found or ["fp16", "float"]


def _format_dtype_block(items: list[dict]) -> tuple[str, str]:
    types = _supported_types(items)
    type_expr = ", ".join(ge_dtype_expr(dtype) for dtype in types)
    format_expr = ", ".join("ge::FORMAT_ND" for _ in types)
    return type_expr, format_expr


def assumed_expression(signature: dict, pattern_family: str) -> str | None:
    input_names = [item["name"] for item in signature.get("inputs", [])]
    if pattern_family == "elementwise-unary" and input_names:
        return f"out = relu({input_names[0]})"
    if pattern_family in {"elementwise-binary", "broadcast-binary"} and len(input_names) >= 2:
        return f"out = {input_names[0]} + {input_names[1]}"
    if (
        pattern_family == "fused-elementwise-activation"
        and input_names[:4] == ["x", "bias", "residual", "gamma"]
    ):
        return "out = relu((x + bias + (fuse_residual ? alpha * residual : 0)) * gamma)"
    if pattern_family == "reduction-like" and input_names:
        return f"out = sum({input_names[0]})"
    if pattern_family == "normalization-like":
        if input_names[:3] == ["x", "gamma", "beta"]:
            return "out = x * gamma + beta (normalization statistics omitted in starter)"
        if input_names[:2] == ["x", "gamma"]:
            return "out = x * gamma (normalization statistics omitted in starter)"
        if input_names:
            return f"out = {input_names[0]} (normalization statistics omitted in starter)"
    return None


def _default_tile_length(pattern_family: str) -> str:
    if pattern_family == "reduction-like":
        return "total_length == 0 ? 1U : total_length"
    return "total_length == 0 ? 1U : (total_length < DEFAULT_TILE_LENGTH ? total_length : DEFAULT_TILE_LENGTH)"


def _default_tile_num(pattern_family: str) -> str:
    if pattern_family == "reduction-like":
        return "total_length == 0 ? 0U : 1U"
    return "total_length == 0 ? 0U : ((total_length + tile_length - 1U) / tile_length)"


def render_kernel_tiling_struct(signature: dict, struct_name: str = "KernelTilingData") -> str:
    attr_lines = [
        f"  {attr_field_type(item)} {attr_field_name(item)};"
        for item in signature.get("attrs", [])
    ]
    return f"""struct {struct_name} {{
  uint32_t totalLength;
  uint32_t tileLength;
  uint32_t tileNum;
{chr(10).join(attr_lines) if attr_lines else ""}
}};"""


def render_tiling_header(op_type: str, signature: dict) -> str:
    guard = f"{snake_case(op_type).upper()}_TILING_H"
    attr_fields = [
        f"  TILING_DATA_FIELD_DEF({attr_field_type(item)}, {attr_field_name(item)});"
        for item in signature.get("attrs", [])
    ]
    return f"""#ifndef {guard}
#define {guard}

#include "register/tilingdata_base.h"

namespace optiling {{
BEGIN_TILING_DATA_DEF(TilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);
  TILING_DATA_FIELD_DEF(uint32_t, tileLength);
  TILING_DATA_FIELD_DEF(uint32_t, tileNum);
{chr(10).join(attr_fields) if attr_fields else ""}
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS({op_type}, TilingData)
}}  // namespace optiling

#endif  // {guard}
"""


def render_host_cpp(op_type: str, signature: dict, add_config_soc: str) -> str:
    snake = signature["op_name_snake"]
    inputs = signature.get("inputs", [])
    outputs = signature.get("outputs", [])
    attrs = signature.get("attrs", [])
    pattern_family = signature.get("pattern_family", "generic-custom")

    infer_shape_lines = [
        "  const gert::Shape* input_shape = context->GetInputShape(0);",
        "  gert::Shape* output_shape = context->GetOutputShape(0);",
        "  *output_shape = *input_shape;",
    ]
    infer_dtype_lines = [
        "  const auto input_dtype = context->GetInputDataType(0);",
        "  context->SetOutputDataType(0, input_dtype);",
    ]

    input_lines = []
    for item in inputs:
        types, formats = _format_dtype_block([item])
        input_lines.extend(
            [
                f'    this->Input("{item["name"]}")',
                "        .ParamType(REQUIRED)",
                f"        .DataType({{{types}}})",
                f"        .Format({{{formats}}})",
                f"        .UnknownShapeFormat({{{formats}}});",
            ]
        )

    output_lines = []
    for item in outputs:
        types, formats = _format_dtype_block(inputs + [item])
        output_lines.extend(
            [
                f'    this->Output("{item["name"]}")',
                "        .ParamType(REQUIRED)",
                f"        .DataType({{{types}}})",
                f"        .Format({{{formats}}})",
                f"        .UnknownShapeFormat({{{formats}}});",
            ]
        )

    attr_lines = []
    for item in attrs:
        attr_method = {
            "int": "Int()",
            "bool": "Bool()",
            "float": "Float()",
        }.get(item.get("attr_type", "int").lower(), "Int()")
        attr_lines.extend(
            [
                f'    this->Attr("{item["name"]}")',
                f"        .{attr_method};",
            ]
        )

    attr_read_lines = ["  const auto* attrs = context->GetAttrs();"]
    for index, item in enumerate(attrs):
        field_name = attr_field_name(item)
        attr_type = item.get("attr_type", "int").lower()
        if attr_type == "float":
            attr_read_lines.extend(
                [
                    f"  float {field_name} = {attr_field_default(item)};",
                    f"  if (attrs != nullptr && attrs->GetFloat({index}) != nullptr) {{",
                    f"    {field_name} = *attrs->GetFloat({index});",
                    "  }",
                    f"  tiling.set_{field_name}({field_name});",
                ]
            )
        elif attr_type == "bool":
            attr_read_lines.extend(
                [
                    f"  bool {field_name} = false;",
                    f"  if (attrs != nullptr && attrs->GetBool({index}) != nullptr) {{",
                    f"    {field_name} = *attrs->GetBool({index});",
                    "  }",
                    f"  tiling.set_{field_name}({field_name} ? 1U : 0U);",
                ]
            )
        else:
            attr_read_lines.extend(
                [
                    f"  int64_t {field_name} = 0;",
                    f"  if (attrs != nullptr && attrs->GetInt({index}) != nullptr) {{",
                    f"    {field_name} = *attrs->GetInt({index});",
                    "  }",
                    f"  tiling.set_{field_name}({field_name});",
                ]
            )

    return f"""#include "{snake}_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {{
constexpr uint32_t BLOCK_DIM = 1;
constexpr uint32_t DEFAULT_TILE_LENGTH = 256;

static ge::graphStatus TilingFunc(gert::TilingContext* context) {{
  TilingData tiling;
  const gert::StorageShape* x_shape = context->GetInputShape(0);
  uint32_t total_length = 0;
  if (x_shape != nullptr) {{
    total_length = static_cast<uint32_t>(x_shape->GetStorageShape().GetShapeSize());
  }}
  uint32_t tile_length = {_default_tile_length(pattern_family)};
  uint32_t tile_num = {_default_tile_num(pattern_family)};

  tiling.set_totalLength(total_length);
  tiling.set_tileLength(tile_length);
  tiling.set_tileNum(tile_num);
{chr(10).join(attr_read_lines)}

  context->SetBlockDim(BLOCK_DIM);
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
  size_t* workspace = context->GetWorkspaceSizes(1);
  if (workspace != nullptr) {{
    workspace[0] = 0;
  }}
  return ge::GRAPH_SUCCESS;
}}
}}  // namespace optiling

namespace ge {{
static graphStatus InferShape(gert::InferShapeContext* context) {{
{chr(10).join(infer_shape_lines)}
  return GRAPH_SUCCESS;
}}

static graphStatus InferDataType(gert::InferDataTypeContext* context) {{
{chr(10).join(infer_dtype_lines)}
  return GRAPH_SUCCESS;
}}
}}  // namespace ge

namespace ops {{
class {op_type} : public OpDef {{
 public:
  explicit {op_type}(const char* name) : OpDef(name) {{
{chr(10).join(input_lines)}
{chr(10).join(output_lines)}
{chr(10).join(attr_lines) if attr_lines else "    // No scalar attrs were inferred from the CUDA project."}
    this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
    this->AICore().SetTiling(optiling::TilingFunc);
    this->AICore().AddConfig("{add_config_soc}");
  }}
}};

OP_ADD({op_type});
}}  // namespace ops
"""


def _fallback_kernel(signature: dict, tiling_class_name: str) -> str:
    snake = signature["op_name_snake"]
    inputs = signature.get("inputs", [])
    outputs = signature.get("outputs", [])
    pattern_family = signature.get("pattern_family", "generic-custom")
    comments = {
        "fused-elementwise": "Port the fused scalar expression into explicit load/compute/store stages.",
        "broadcast-binary": "Encode broadcast semantics in tiling before generalizing the kernel body.",
        "reduction-like": "Wire up the reduction axis, accumulation dtype, and output shape before optimizing the schedule.",
        "normalization-like": "Preserve epsilon, affine weights, and axis semantics before swapping in a tuned normalization kernel.",
        "generic-custom": "Map the original CUDA loop nest into explicit load/compute/store phases.",
    }
    gm_args = [item["name"] for item in inputs + outputs]
    arg_list = ", ".join(f"GM_ADDR {name}" for name in gm_args + ["workspace", "tiling"])
    return f"""#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

using namespace AscendC;
{render_kernel_tiling_struct(signature)}

extern "C" __global__ __aicore__ void {snake}({arg_list}) {{
  GET_TILING_DATA_WITH_STRUCT(KernelTilingData, tiling_data, tiling);
  (void)workspace;
  // Pattern family: {pattern_family}
  // {comments.get(pattern_family, comments["generic-custom"])}
  // Assumed expression: {assumed_expression(signature, pattern_family) or "preserve the CUDA semantics here"}
}}
"""


def _render_unary_kernel(signature: dict, tiling_class_name: str) -> str:
    snake = signature["op_name_snake"]
    inputs = signature.get("inputs", [])
    outputs = signature.get("outputs", [])
    if len(inputs) != 1 or len(outputs) != 1:
        return _fallback_kernel(signature, tiling_class_name)
    x_name = inputs[0]["name"]
    out_name = outputs[0]["name"]
    return f"""#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

using namespace AscendC;
{render_kernel_tiling_struct(signature)}

constexpr int32_t BUFFER_NUM = 2;

class Kernel{camel_case(snake)} {{
 public:
  __aicore__ inline void Init(GM_ADDR {x_name}, GM_ADDR {out_name}, const KernelTilingData& tiling_data) {{
    totalLength_ = tiling_data.totalLength;
    tileLength_ = tiling_data.tileLength;
    tileNum_ = tiling_data.tileNum;
    xGm_.SetGlobalBuffer((__gm__ DTYPE_X*){x_name}, totalLength_);
    outGm_.SetGlobalBuffer((__gm__ DTYPE_OUT*){out_name}, totalLength_);
    if (tileNum_ == 0 || tileLength_ == 0) {{
      return;
    }}
    pipe_.InitBuffer(inQueueX_, BUFFER_NUM, tileLength_ * sizeof(DTYPE_X));
    pipe_.InitBuffer(outQueueOut_, BUFFER_NUM, tileLength_ * sizeof(DTYPE_OUT));
  }}

  __aicore__ inline void Process() {{
    for (uint32_t i = 0; i < tileNum_; ++i) {{
      uint32_t current_length = CurrentLength(i);
      if (current_length == 0) {{
        continue;
      }}
      CopyIn(i, current_length);
      Compute(current_length);
      CopyOut(i, current_length);
    }}
  }}

 private:
  __aicore__ inline uint32_t CurrentOffset(uint32_t progress) const {{
    return progress * tileLength_;
  }}

  __aicore__ inline uint32_t CurrentLength(uint32_t progress) const {{
    uint32_t offset = CurrentOffset(progress);
    if (offset >= totalLength_) {{
      return 0;
    }}
    uint32_t remaining = totalLength_ - offset;
    return remaining < tileLength_ ? remaining : tileLength_;
  }}

  __aicore__ inline void CopyIn(uint32_t progress, uint32_t current_length) {{
    LocalTensor<DTYPE_X> x_local = inQueueX_.AllocTensor<DTYPE_X>();
    DataCopy(x_local, xGm_[CurrentOffset(progress)], current_length);
    inQueueX_.EnQue(x_local);
  }}

  __aicore__ inline void Compute(uint32_t current_length) {{
    LocalTensor<DTYPE_X> x_local = inQueueX_.DeQue<DTYPE_X>();
    LocalTensor<DTYPE_OUT> out_local = outQueueOut_.AllocTensor<DTYPE_OUT>();
    Relu(out_local, x_local, current_length);
    outQueueOut_.EnQue(out_local);
    inQueueX_.FreeTensor(x_local);
  }}

  __aicore__ inline void CopyOut(uint32_t progress, uint32_t current_length) {{
    LocalTensor<DTYPE_OUT> out_local = outQueueOut_.DeQue<DTYPE_OUT>();
    DataCopy(outGm_[CurrentOffset(progress)], out_local, current_length);
    outQueueOut_.FreeTensor(out_local);
  }}

  TPipe pipe_;
  TQue<TPosition::VECIN, BUFFER_NUM> inQueueX_;
  TQue<TPosition::VECOUT, BUFFER_NUM> outQueueOut_;
  GlobalTensor<DTYPE_X> xGm_;
  GlobalTensor<DTYPE_OUT> outGm_;
  uint32_t totalLength_ = 0;
  uint32_t tileLength_ = 0;
  uint32_t tileNum_ = 0;
}};

extern "C" __global__ __aicore__ void {snake}(GM_ADDR {x_name}, GM_ADDR {out_name}, GM_ADDR workspace, GM_ADDR tiling) {{
  GET_TILING_DATA_WITH_STRUCT(KernelTilingData, tiling_data, tiling);
  (void)workspace;
  Kernel{camel_case(snake)} op;
  op.Init({x_name}, {out_name}, tiling_data);
  op.Process();
}}
"""


def _render_binary_kernel(signature: dict, tiling_class_name: str, broadcast_assumption: bool) -> str:
    snake = signature["op_name_snake"]
    inputs = signature.get("inputs", [])
    outputs = signature.get("outputs", [])
    if len(inputs) < 2 or len(outputs) != 1:
        return _fallback_kernel(signature, tiling_class_name)
    x_name = inputs[0]["name"]
    y_name = inputs[1]["name"]
    out_name = outputs[0]["name"]
    extra_note = "  // First pass assumes both inputs are already expanded to the same logical shape.\n" if broadcast_assumption else ""
    return f"""#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

using namespace AscendC;
{render_kernel_tiling_struct(signature)}

constexpr int32_t BUFFER_NUM = 2;

class Kernel{camel_case(snake)} {{
 public:
  __aicore__ inline void Init(GM_ADDR {x_name}, GM_ADDR {y_name}, GM_ADDR {out_name}, const KernelTilingData& tiling_data) {{
    totalLength_ = tiling_data.totalLength;
    tileLength_ = tiling_data.tileLength;
    tileNum_ = tiling_data.tileNum;
    xGm_.SetGlobalBuffer((__gm__ DTYPE_X*){x_name}, totalLength_);
    yGm_.SetGlobalBuffer((__gm__ DTYPE_Y*){y_name}, totalLength_);
    outGm_.SetGlobalBuffer((__gm__ DTYPE_OUT*){out_name}, totalLength_);
    if (tileNum_ == 0 || tileLength_ == 0) {{
      return;
    }}
    pipe_.InitBuffer(inQueueX_, BUFFER_NUM, tileLength_ * sizeof(DTYPE_X));
    pipe_.InitBuffer(inQueueY_, BUFFER_NUM, tileLength_ * sizeof(DTYPE_Y));
    pipe_.InitBuffer(outQueueOut_, BUFFER_NUM, tileLength_ * sizeof(DTYPE_OUT));
  }}

  __aicore__ inline void Process() {{
    for (uint32_t i = 0; i < tileNum_; ++i) {{
      uint32_t current_length = CurrentLength(i);
      if (current_length == 0) {{
        continue;
      }}
      CopyIn(i, current_length);
      Compute(current_length);
      CopyOut(i, current_length);
    }}
  }}

 private:
  __aicore__ inline uint32_t CurrentOffset(uint32_t progress) const {{
    return progress * tileLength_;
  }}

  __aicore__ inline uint32_t CurrentLength(uint32_t progress) const {{
    uint32_t offset = CurrentOffset(progress);
    if (offset >= totalLength_) {{
      return 0;
    }}
    uint32_t remaining = totalLength_ - offset;
    return remaining < tileLength_ ? remaining : tileLength_;
  }}

  __aicore__ inline void CopyIn(uint32_t progress, uint32_t current_length) {{
    LocalTensor<DTYPE_X> x_local = inQueueX_.AllocTensor<DTYPE_X>();
    LocalTensor<DTYPE_Y> y_local = inQueueY_.AllocTensor<DTYPE_Y>();
    DataCopy(x_local, xGm_[CurrentOffset(progress)], current_length);
    DataCopy(y_local, yGm_[CurrentOffset(progress)], current_length);
    inQueueX_.EnQue(x_local);
    inQueueY_.EnQue(y_local);
  }}

  __aicore__ inline void Compute(uint32_t current_length) {{
    LocalTensor<DTYPE_X> x_local = inQueueX_.DeQue<DTYPE_X>();
    LocalTensor<DTYPE_Y> y_local = inQueueY_.DeQue<DTYPE_Y>();
    LocalTensor<DTYPE_OUT> out_local = outQueueOut_.AllocTensor<DTYPE_OUT>();
    Add(out_local, x_local, y_local, current_length);
    outQueueOut_.EnQue(out_local);
    inQueueX_.FreeTensor(x_local);
    inQueueY_.FreeTensor(y_local);
  }}

  __aicore__ inline void CopyOut(uint32_t progress, uint32_t current_length) {{
    LocalTensor<DTYPE_OUT> out_local = outQueueOut_.DeQue<DTYPE_OUT>();
    DataCopy(outGm_[CurrentOffset(progress)], out_local, current_length);
    outQueueOut_.FreeTensor(out_local);
  }}

  TPipe pipe_;
  TQue<TPosition::VECIN, BUFFER_NUM> inQueueX_;
  TQue<TPosition::VECIN, BUFFER_NUM> inQueueY_;
  TQue<TPosition::VECOUT, BUFFER_NUM> outQueueOut_;
  GlobalTensor<DTYPE_X> xGm_;
  GlobalTensor<DTYPE_Y> yGm_;
  GlobalTensor<DTYPE_OUT> outGm_;
  uint32_t totalLength_ = 0;
  uint32_t tileLength_ = 0;
  uint32_t tileNum_ = 0;
}};

extern "C" __global__ __aicore__ void {snake}(GM_ADDR {x_name}, GM_ADDR {y_name}, GM_ADDR {out_name}, GM_ADDR workspace, GM_ADDR tiling) {{
  GET_TILING_DATA_WITH_STRUCT(KernelTilingData, tiling_data, tiling);
  (void)workspace;
{extra_note.rstrip()}
  Kernel{camel_case(snake)} op;
  op.Init({x_name}, {y_name}, {out_name}, tiling_data);
  op.Process();
}}
"""


def _render_fused_bias_relu_kernel(signature: dict, tiling_class_name: str) -> str:
    snake = signature["op_name_snake"]
    inputs = signature.get("inputs", [])
    outputs = signature.get("outputs", [])
    input_names = [item["name"] for item in inputs]
    if input_names[:4] != ["x", "bias", "residual", "gamma"] or len(outputs) != 1:
        return _fallback_kernel(signature, tiling_class_name)
    out_name = outputs[0]["name"]
    return f"""#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

using namespace AscendC;
{render_kernel_tiling_struct(signature)}

constexpr int32_t BUFFER_NUM = 2;

class Kernel{camel_case(snake)} {{
 public:
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR bias, GM_ADDR residual, GM_ADDR gamma, GM_ADDR {out_name},
                              const KernelTilingData& tiling_data) {{
    totalLength_ = tiling_data.totalLength;
    tileLength_ = tiling_data.tileLength;
    tileNum_ = tiling_data.tileNum;
    alpha_ = tiling_data.alpha;
    fuseResidual_ = tiling_data.fuseResidual != 0;

    xGm_.SetGlobalBuffer((__gm__ DTYPE_X*)x, totalLength_);
    biasGm_.SetGlobalBuffer((__gm__ DTYPE_BIAS*)bias, totalLength_);
    residualGm_.SetGlobalBuffer((__gm__ DTYPE_RESIDUAL*)residual, totalLength_);
    gammaGm_.SetGlobalBuffer((__gm__ DTYPE_GAMMA*)gamma, totalLength_);
    outGm_.SetGlobalBuffer((__gm__ DTYPE_OUT*){out_name}, totalLength_);

    if (tileNum_ == 0 || tileLength_ == 0) {{
      return;
    }}

    pipe_.InitBuffer(inQueueX_, BUFFER_NUM, tileLength_ * sizeof(DTYPE_X));
    pipe_.InitBuffer(inQueueBias_, BUFFER_NUM, tileLength_ * sizeof(DTYPE_BIAS));
    pipe_.InitBuffer(inQueueResidual_, BUFFER_NUM, tileLength_ * sizeof(DTYPE_RESIDUAL));
    pipe_.InitBuffer(inQueueGamma_, BUFFER_NUM, tileLength_ * sizeof(DTYPE_GAMMA));
    pipe_.InitBuffer(outQueueOut_, BUFFER_NUM, tileLength_ * sizeof(DTYPE_OUT));
  }}

  __aicore__ inline void Process() {{
    for (uint32_t i = 0; i < tileNum_; ++i) {{
      uint32_t current_length = CurrentLength(i);
      if (current_length == 0) {{
        continue;
      }}
      CopyIn(i, current_length);
      Compute(current_length);
      CopyOut(i, current_length);
    }}
  }}

 private:
  __aicore__ inline uint32_t CurrentOffset(uint32_t progress) const {{
    return progress * tileLength_;
  }}

  __aicore__ inline uint32_t CurrentLength(uint32_t progress) const {{
    uint32_t offset = CurrentOffset(progress);
    if (offset >= totalLength_) {{
      return 0;
    }}
    uint32_t remaining = totalLength_ - offset;
    return remaining < tileLength_ ? remaining : tileLength_;
  }}

  __aicore__ inline void CopyIn(uint32_t progress, uint32_t current_length) {{
    uint32_t offset = CurrentOffset(progress);
    LocalTensor<DTYPE_X> x_local = inQueueX_.AllocTensor<DTYPE_X>();
    LocalTensor<DTYPE_BIAS> bias_local = inQueueBias_.AllocTensor<DTYPE_BIAS>();
    LocalTensor<DTYPE_RESIDUAL> residual_local = inQueueResidual_.AllocTensor<DTYPE_RESIDUAL>();
    LocalTensor<DTYPE_GAMMA> gamma_local = inQueueGamma_.AllocTensor<DTYPE_GAMMA>();
    DataCopy(x_local, xGm_[offset], current_length);
    DataCopy(bias_local, biasGm_[offset], current_length);
    DataCopy(residual_local, residualGm_[offset], current_length);
    DataCopy(gamma_local, gammaGm_[offset], current_length);
    inQueueX_.EnQue(x_local);
    inQueueBias_.EnQue(bias_local);
    inQueueResidual_.EnQue(residual_local);
    inQueueGamma_.EnQue(gamma_local);
  }}

  __aicore__ inline void Compute(uint32_t current_length) {{
    LocalTensor<DTYPE_X> x_local = inQueueX_.DeQue<DTYPE_X>();
    LocalTensor<DTYPE_BIAS> bias_local = inQueueBias_.DeQue<DTYPE_BIAS>();
    LocalTensor<DTYPE_RESIDUAL> residual_local = inQueueResidual_.DeQue<DTYPE_RESIDUAL>();
    LocalTensor<DTYPE_GAMMA> gamma_local = inQueueGamma_.DeQue<DTYPE_GAMMA>();
    LocalTensor<DTYPE_OUT> out_local = outQueueOut_.AllocTensor<DTYPE_OUT>();

    Add(out_local, x_local, bias_local, current_length);
    if (fuseResidual_) {{
      if (alpha_ != 1.0f) {{
        DTYPE_RESIDUAL alpha_value = static_cast<DTYPE_RESIDUAL>(alpha_);
        Muls(residual_local, residual_local, alpha_value, current_length);
      }}
      Add(out_local, out_local, residual_local, current_length);
    }}
    Mul(out_local, out_local, gamma_local, current_length);
    Relu(out_local, out_local, current_length);

    outQueueOut_.EnQue(out_local);
    inQueueX_.FreeTensor(x_local);
    inQueueBias_.FreeTensor(bias_local);
    inQueueResidual_.FreeTensor(residual_local);
    inQueueGamma_.FreeTensor(gamma_local);
  }}

  __aicore__ inline void CopyOut(uint32_t progress, uint32_t current_length) {{
    LocalTensor<DTYPE_OUT> out_local = outQueueOut_.DeQue<DTYPE_OUT>();
    DataCopy(outGm_[CurrentOffset(progress)], out_local, current_length);
    outQueueOut_.FreeTensor(out_local);
  }}

  TPipe pipe_;
  TQue<TPosition::VECIN, BUFFER_NUM> inQueueX_;
  TQue<TPosition::VECIN, BUFFER_NUM> inQueueBias_;
  TQue<TPosition::VECIN, BUFFER_NUM> inQueueResidual_;
  TQue<TPosition::VECIN, BUFFER_NUM> inQueueGamma_;
  TQue<TPosition::VECOUT, BUFFER_NUM> outQueueOut_;
  GlobalTensor<DTYPE_X> xGm_;
  GlobalTensor<DTYPE_BIAS> biasGm_;
  GlobalTensor<DTYPE_RESIDUAL> residualGm_;
  GlobalTensor<DTYPE_GAMMA> gammaGm_;
  GlobalTensor<DTYPE_OUT> outGm_;
  uint32_t totalLength_ = 0;
  uint32_t tileLength_ = 0;
  uint32_t tileNum_ = 0;
  float alpha_ = 1.0f;
  bool fuseResidual_ = false;
}};

extern "C" __global__ __aicore__ void {snake}(GM_ADDR x, GM_ADDR bias, GM_ADDR residual, GM_ADDR gamma, GM_ADDR {out_name},
                                              GM_ADDR workspace, GM_ADDR tiling) {{
  GET_TILING_DATA_WITH_STRUCT(KernelTilingData, tiling_data, tiling);
  (void)workspace;
  Kernel{camel_case(snake)} op;
  op.Init(x, bias, residual, gamma, {out_name}, tiling_data);
  op.Process();
}}
"""


def _render_reduction_kernel(signature: dict, tiling_class_name: str) -> str:
    snake = signature["op_name_snake"]
    inputs = signature.get("inputs", [])
    outputs = signature.get("outputs", [])
    if len(inputs) != 1 or len(outputs) != 1:
        return _fallback_kernel(signature, tiling_class_name)
    x_name = inputs[0]["name"]
    out_name = outputs[0]["name"]
    return f"""#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/reduce/sum.h"

using namespace AscendC;
{render_kernel_tiling_struct(signature)}

constexpr int32_t BUFFER_NUM = 1;

class Kernel{camel_case(snake)} {{
 public:
  __aicore__ inline void Init(GM_ADDR {x_name}, GM_ADDR {out_name}, const KernelTilingData& tiling_data) {{
    totalLength_ = tiling_data.totalLength;
    tileLength_ = tiling_data.tileLength;
    xGm_.SetGlobalBuffer((__gm__ DTYPE_X*){x_name}, totalLength_);
    outGm_.SetGlobalBuffer((__gm__ DTYPE_OUT*){out_name}, 1);
    if (totalLength_ == 0 || tileLength_ == 0) {{
      return;
    }}
    pipe_.InitBuffer(inQueueX_, BUFFER_NUM, tileLength_ * sizeof(DTYPE_X));
    pipe_.InitBuffer(outQueueOut_, BUFFER_NUM, 32);
  }}

  __aicore__ inline void Process() {{
    if (totalLength_ == 0) {{
      return;
    }}
    CopyIn();
    Compute();
    CopyOut();
  }}

 private:
  __aicore__ inline uint32_t AlignedLength() const {{
    return static_cast<uint32_t>(((static_cast<uint64_t>(totalLength_) * sizeof(DTYPE_X) + 31ULL) / 32ULL) * 32ULL / sizeof(DTYPE_X));
  }}

  __aicore__ inline void CopyIn() {{
    LocalTensor<DTYPE_X> x_local = inQueueX_.AllocTensor<DTYPE_X>();
    DataCopy(x_local, xGm_[0], totalLength_);
    inQueueX_.EnQue(x_local);
  }}

  __aicore__ inline void Compute() {{
    LocalTensor<DTYPE_X> x_local = inQueueX_.DeQue<DTYPE_X>();
    LocalTensor<DTYPE_OUT> out_local = outQueueOut_.AllocTensor<DTYPE_OUT>();
    if constexpr (IsSameType<DTYPE_X, half>::value || IsSameType<DTYPE_X, float>::value) {{
      SumParams sum_params{{1U, AlignedLength(), totalLength_}};
      Sum(out_local, x_local, sum_params);
    }} else {{
      // Fallback for non-floating starter dtypes until a dedicated accumulation path is generated.
      DataCopy(out_local, x_local, 1);
    }}
    outQueueOut_.EnQue(out_local);
    inQueueX_.FreeTensor(x_local);
  }}

  __aicore__ inline void CopyOut() {{
    LocalTensor<DTYPE_OUT> out_local = outQueueOut_.DeQue<DTYPE_OUT>();
    DataCopy(outGm_[0], out_local, 1);
    outQueueOut_.FreeTensor(out_local);
  }}

  TPipe pipe_;
  TQue<TPosition::VECIN, BUFFER_NUM> inQueueX_;
  TQue<TPosition::VECOUT, BUFFER_NUM> outQueueOut_;
  GlobalTensor<DTYPE_X> xGm_;
  GlobalTensor<DTYPE_OUT> outGm_;
  uint32_t totalLength_ = 0;
  uint32_t tileLength_ = 0;
}};

extern "C" __global__ __aicore__ void {snake}(GM_ADDR {x_name}, GM_ADDR {out_name}, GM_ADDR workspace, GM_ADDR tiling) {{
  GET_TILING_DATA_WITH_STRUCT(KernelTilingData, tiling_data, tiling);
  (void)workspace;
  if (GetBlockIdx() != 0) {{
    return;
  }}
  Kernel{camel_case(snake)} op;
  op.Init({x_name}, {out_name}, tiling_data);
  op.Process();
}}
"""


def _render_normalization_kernel(signature: dict, tiling_class_name: str) -> str:
    snake = signature["op_name_snake"]
    inputs = signature.get("inputs", [])
    outputs = signature.get("outputs", [])
    if len(inputs) < 2 or len(outputs) != 1:
        return _fallback_kernel(signature, tiling_class_name)
    x_name = inputs[0]["name"]
    gamma_name = inputs[1]["name"]
    beta_name = inputs[2]["name"] if len(inputs) >= 3 else None
    out_name = outputs[0]["name"]
    beta_type = f"DTYPE_{beta_name.upper()}" if beta_name else ""
    beta_queue = f"inQueue{camel_case(beta_name)}_" if beta_name else ""
    beta_global = f"{beta_name}Gm_" if beta_name else ""
    beta_init = ""
    beta_copy_in = ""
    beta_enqueue = ""
    beta_dequeue = ""
    beta_free = ""
    beta_queue_decl = ""
    beta_global_decl = ""
    beta_init_queue = ""
    if beta_name is not None:
        beta_init = f"    {beta_global}.SetGlobalBuffer((__gm__ {beta_type}*){beta_name}, totalLength_);\n"
        beta_copy_in = (
            f"    LocalTensor<{beta_type}> {beta_name}_local = {beta_queue}.AllocTensor<{beta_type}>();\n"
            f"    DataCopy({beta_name}_local, {beta_global}[CurrentOffset(progress)], current_length);\n"
        )
        beta_enqueue = f"    {beta_queue}.EnQue({beta_name}_local);\n"
        beta_dequeue = (
            f"    LocalTensor<{beta_type}> {beta_name}_local = {beta_queue}.DeQue<{beta_type}>();\n"
            f"    Add(out_local, out_local, {beta_name}_local, current_length);\n"
        )
        beta_free = f"    {beta_queue}.FreeTensor({beta_name}_local);\n"
        beta_queue_decl = f"  TQue<TPosition::VECIN, BUFFER_NUM> {beta_queue};\n"
        beta_global_decl = f"  GlobalTensor<{beta_type}> {beta_global};\n"
        beta_init_queue = f"    pipe_.InitBuffer({beta_queue}, BUFFER_NUM, tileLength_ * sizeof({beta_type}));\n"
    return f"""#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

using namespace AscendC;
{render_kernel_tiling_struct(signature)}

constexpr int32_t BUFFER_NUM = 2;

class Kernel{camel_case(snake)} {{
 public:
  __aicore__ inline void Init(GM_ADDR {x_name}, GM_ADDR {gamma_name}, {'GM_ADDR ' + beta_name + ', ' if beta_name else ''}GM_ADDR {out_name}, const KernelTilingData& tiling_data) {{
    totalLength_ = tiling_data.totalLength;
    tileLength_ = tiling_data.tileLength;
    tileNum_ = tiling_data.tileNum;
    xGm_.SetGlobalBuffer((__gm__ DTYPE_X*){x_name}, totalLength_);
    gammaGm_.SetGlobalBuffer((__gm__ DTYPE_GAMMA*){gamma_name}, totalLength_);
{beta_init.rstrip() if beta_name else ''}
    outGm_.SetGlobalBuffer((__gm__ DTYPE_OUT*){out_name}, totalLength_);
    if (tileNum_ == 0 || tileLength_ == 0) {{
      return;
    }}
    pipe_.InitBuffer(inQueueX_, BUFFER_NUM, tileLength_ * sizeof(DTYPE_X));
    pipe_.InitBuffer(inQueueGamma_, BUFFER_NUM, tileLength_ * sizeof(DTYPE_GAMMA));
{beta_init_queue.rstrip() if beta_name else ''}
    pipe_.InitBuffer(outQueueOut_, BUFFER_NUM, tileLength_ * sizeof(DTYPE_OUT));
  }}

  __aicore__ inline void Process() {{
    for (uint32_t i = 0; i < tileNum_; ++i) {{
      uint32_t current_length = CurrentLength(i);
      if (current_length == 0) {{
        continue;
      }}
      CopyIn(i, current_length);
      Compute(current_length);
      CopyOut(i, current_length);
    }}
  }}

 private:
  __aicore__ inline uint32_t CurrentOffset(uint32_t progress) const {{
    return progress * tileLength_;
  }}

  __aicore__ inline uint32_t CurrentLength(uint32_t progress) const {{
    uint32_t offset = CurrentOffset(progress);
    if (offset >= totalLength_) {{
      return 0;
    }}
    uint32_t remaining = totalLength_ - offset;
    return remaining < tileLength_ ? remaining : tileLength_;
  }}

  __aicore__ inline void CopyIn(uint32_t progress, uint32_t current_length) {{
    LocalTensor<DTYPE_X> x_local = inQueueX_.AllocTensor<DTYPE_X>();
    LocalTensor<DTYPE_GAMMA> gamma_local = inQueueGamma_.AllocTensor<DTYPE_GAMMA>();
    DataCopy(x_local, xGm_[CurrentOffset(progress)], current_length);
    DataCopy(gamma_local, gammaGm_[CurrentOffset(progress)], current_length);
{beta_copy_in.rstrip() if beta_name else ''}
    inQueueX_.EnQue(x_local);
    inQueueGamma_.EnQue(gamma_local);
{beta_enqueue.rstrip() if beta_name else ''}
  }}

  __aicore__ inline void Compute(uint32_t current_length) {{
    LocalTensor<DTYPE_X> x_local = inQueueX_.DeQue<DTYPE_X>();
    LocalTensor<DTYPE_GAMMA> gamma_local = inQueueGamma_.DeQue<DTYPE_GAMMA>();
    LocalTensor<DTYPE_OUT> out_local = outQueueOut_.AllocTensor<DTYPE_OUT>();
    Mul(out_local, x_local, gamma_local, current_length);
{beta_dequeue.rstrip() if beta_name else ''}
    outQueueOut_.EnQue(out_local);
    inQueueX_.FreeTensor(x_local);
    inQueueGamma_.FreeTensor(gamma_local);
{beta_free.rstrip() if beta_name else ''}
  }}

  __aicore__ inline void CopyOut(uint32_t progress, uint32_t current_length) {{
    LocalTensor<DTYPE_OUT> out_local = outQueueOut_.DeQue<DTYPE_OUT>();
    DataCopy(outGm_[CurrentOffset(progress)], out_local, current_length);
    outQueueOut_.FreeTensor(out_local);
  }}

  TPipe pipe_;
  TQue<TPosition::VECIN, BUFFER_NUM> inQueueX_;
  TQue<TPosition::VECIN, BUFFER_NUM> inQueueGamma_;
{beta_queue_decl.rstrip() if beta_name else ''}
  TQue<TPosition::VECOUT, BUFFER_NUM> outQueueOut_;
  GlobalTensor<DTYPE_X> xGm_;
  GlobalTensor<DTYPE_GAMMA> gammaGm_;
{beta_global_decl.rstrip() if beta_name else ''}
  GlobalTensor<DTYPE_OUT> outGm_;
  uint32_t totalLength_ = 0;
  uint32_t tileLength_ = 0;
  uint32_t tileNum_ = 0;
}};

extern "C" __global__ __aicore__ void {snake}(GM_ADDR {x_name}, GM_ADDR {gamma_name}, {'GM_ADDR ' + beta_name + ', ' if beta_name else ''}GM_ADDR {out_name}, GM_ADDR workspace, GM_ADDR tiling) {{
  GET_TILING_DATA_WITH_STRUCT(KernelTilingData, tiling_data, tiling);
  (void)workspace;
  Kernel{camel_case(snake)} op;
  op.Init({x_name}, {gamma_name}, {beta_name + ', ' if beta_name else ''}{out_name}, tiling_data);
  op.Process();
}}
"""


def render_kernel_cpp(signature: dict, tiling_class_name: str = "TilingData") -> str:
    pattern_family = signature.get("pattern_family", "generic-custom")
    if pattern_family == "elementwise-unary":
        return _render_unary_kernel(signature, tiling_class_name)
    if pattern_family == "elementwise-binary":
        return _render_binary_kernel(signature, tiling_class_name, broadcast_assumption=False)
    if pattern_family == "broadcast-binary":
        return _render_binary_kernel(signature, tiling_class_name, broadcast_assumption=True)
    if pattern_family == "fused-elementwise-activation":
        return _render_fused_bias_relu_kernel(signature, tiling_class_name)
    if pattern_family == "reduction-like":
        return _render_reduction_kernel(signature, tiling_class_name)
    if pattern_family == "normalization-like":
        return _render_normalization_kernel(signature, tiling_class_name)
    return _fallback_kernel(signature, tiling_class_name)
