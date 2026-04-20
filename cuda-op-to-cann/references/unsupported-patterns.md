# Unsupported or High-Risk Patterns

Treat these as escalation points instead of auto-conversion targets:

- inline PTX or `asm`
- WMMA or tensor-core-specific code paths
- `cooperative_groups`
- warp shuffle or ballot heavy logic
- deep atomics contention
- tightly coupled multi-launch workflows with implicit synchronization
- custom allocators, custom stream/event choreography, or hidden side effects

When these appear, generate a starter package plus a redesign plan. Do not mark the migration complete.
