# Official Ascend Sources

Load this file when the migration needs version-sensitive or framework-sensitive guidance. Prefer these official documents over memory when the user asks for "latest", framework integration details, or packaging behavior.

## Recommended Reading Order

1. Built-in operator route

- CANN operator library overview and operator index:
  [CANN 算子库接口参考](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/API/aolapi/operatorlist_00001.html)
- Single-op API execution flow and the two-stage `GetWorkspaceSize + execute` contract:
  [单算子调用流程](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha001/appdevg/acldevg/aclcppdevg_000017.html)

Use these when deciding whether a CUDA op should stay in the `aclnn-direct` or `aclnn-composite` lane.

2. Custom operator route

- Ascend C quick start using `msOpGen`:
  [基于自定义算子工程的算子开发](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1alpha001/opdevg/Ascendcopdevg/atlas_ascendc_10_0006.html)
- `msOpGen` compilation, packaging, deployment layout, and package outputs:
  [算子编译部署（msOpGen）](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1alpha002/devaids/optool/atlasopdev_16_0024.html)

Use these when generating or patching `msopgen` specs, `CMakePresets.json`, `build.sh`, deployment notes, or installation instructions.

3. PyTorch eager and graph integration

- OpPlugin adaptation principles and directory structure:
  [基于OpPlugin算子适配开发](https://www.hiascend.com/document/detail/zh/Pytorch/710/ptmoddevg/Frameworkfeatures/featuresguide_00021.html)
- OpPlugin custom op example with `op_plugin_functions.yaml` and `EXEC_NPU_CMD(aclnnXxx, ...)`:
  [OpPlugin 调用样例](https://www.hiascend.com/document/detail/zh/Pytorch/710/ptmoddevg/Frameworkfeatures/featuresguide_00022.html)
- TorchAir graph-mode registration and Meta function requirements:
  [自定义算子入图：算子注册 PyTorch](https://www.hiascend.com/document/detail/zh/Pytorch/710/modthirdparty/torchairuseguide/torchair_00044.html)

Use these when the source is a PyTorch CUDA extension and the user expects `torch_npu`, OpPlugin, or `torch.compile` compatibility.

## Rules to Encode in the Skill

- Prefer built-in `aclnn` when the behavior already exists in the operator library and the migration only needs adapter glue.
- Use Ascend C custom operator generation when the math is custom but the tensor contract is still regular enough for `msOpGen + host tiling + kernel`.
- For PyTorch eager mode, keep the public schema aligned with Aten IR and route actual execution through `op_api::xx` or `acl_op::xx`.
- For PyTorch graph mode, remember that custom ops need a Meta registration before `torch.compile`, otherwise shape and dtype inference for FX tracing will fail.
- For packaged custom ops, remember the deployed layout matters:
  `op_api/include`, `op_api/lib/libcust_opapi.so`, `op_impl/...`, `op_tiling/liboptiling.so`, and `op_proto`.

## What the Skill Should Avoid

- Do not claim every CUDA op should become a custom Ascend C op. Official docs make the built-in `aclnn` path a first-class route.
- Do not emit PyTorch-only stubs without recording the corresponding OpPlugin or TorchAir follow-up when the user needs real framework integration.
- Do not hardcode one CANN branch unless the user explicitly fixes the target version.
