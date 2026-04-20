# CUDA to CANN Patterns

## Common Mappings

- Elementwise unary or binary kernels often map to single `aclnn` ops or short compositions.
- Broadcast-heavy pointwise kernels are good `aclnn-composite` candidates.
- Simple reductions may still be composable if synchronization is limited and the math is standard.
- Custom fused math with regular indexing is a candidate for `ascendc-custom`.

## Pattern Families Used by This Skill

- `elementwise-unary`: one tensor in, one tensor out, no complex synchronization
- `elementwise-binary`: two tensor inputs with direct tile-wise combination
- `broadcast-binary`: binary math where one operand is broadcast across one or more dimensions
- `fused-elementwise`: multiple tensor inputs plus scalar attrs in one fused expression
- `fused-elementwise-activation`: fused bias, residual, and activation style operators
- `reduction-like`: sum, mean, max, min, argmax, argmin, or atomic-heavy patterns
- `normalization-like`: layernorm, rmsnorm, batchnorm, groupnorm families
- `matmul-like`: matmul, gemm, linear, or batched-matmul families

## Signals That Favor Built-In Operators

- No custom `__global__` kernels.
- Wrapper code around tensor checks, dispatch, or registration.
- Semantics that resemble add, mul, relu, gelu, softmax, layernorm, matmul, or indexing helpers.

## Signals That Favor Ascend C Custom Operators

- One main kernel with regular indexing.
- Limited synchronization complexity.
- Predictable shape rules and tensor contracts.
- Clear separation between host-side metadata and device-side math.

## Pattern-to-Template Hints

- `elementwise-unary`: the current starter template emits a queue-based Ascend C kernel that assumes `out = relu(x)`.
- `elementwise-binary`: the current starter template emits a queue-based Ascend C kernel that assumes `out = x + y`.
- `broadcast-binary`: the first-pass template reuses the binary kernel and assumes both inputs are already expanded to the same logical shape; move true broadcast rules into tiling next.
- `fused-elementwise-activation`: the current starter template assumes `out = relu((x + bias + (fuse_residual ? alpha * residual : 0)) * gamma)`.
- `reduction-like`: the current starter template emits a single-block sum starter for one input and one scalar output, and falls back to copy-through for unsupported accumulation dtypes.
- `normalization-like`: the current starter template emits an affine-first kernel (`out = x * gamma [+ beta]`) and keeps normalization statistics as an explicit manual follow-up.
