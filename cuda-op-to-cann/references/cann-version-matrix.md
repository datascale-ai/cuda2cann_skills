# CANN Version Matrix

## Version-Sensitive Assumptions

- Header locations and preferred libraries may change across CANN versions.
- Newer CANN releases prefer `aclnnop/aclnn_*.h` style headers instead of older deprecated include paths.
- Library packaging also changes over time, so generated build files should not hardcode a single version unless the user requests it.

## Guidance

- Always record the user target version.
- If the version is unknown, keep paths and build flags as placeholders.
- Regenerate or patch templates when moving between major CANN branches.
- Prefer docs that are close to the user's installed branch, especially for `msOpGen`, packaging, and PyTorch integration details.
- For PyTorch adaptation, record both the PyTorch version and the Ascend Extension for PyTorch version because official branch naming uses both.

## Official Version Clues

- Official PyTorch adaptation examples use branch names such as `v2.1.0-7.1.0`, where the first component is the PyTorch version and the second is the Ascend Extension for PyTorch version.
- Official `msOpGen` and Ascend C docs are published per CANN branch, so the skill should avoid assuming one fixed documentation branch for every project.
- If the user gives a concrete CANN version, prefer matching the generated notes and environment assumptions to that branch before changing build scripts.
