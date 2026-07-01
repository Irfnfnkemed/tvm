# Megakernel DSL Design

## Goal

Megakernel generation is split into agent-facing planning steps and compiler-facing lowering steps.
The agent should describe the structure of the megakernel: stages, tile spaces, tensors, and logical synchronization.
The compiler should validate that structure and lower it to Relax/TIRX/runtime code.

The key boundary is intentional: agents produce logical plans, not low-level CUDA event code.

## Overall Workflow

```text
Stage 1: High-level program -> tile/event plan
Stage 2: Tile/event plan -> KernelSpec DSL
Stage 3: KernelSpec DSL -> Relax representation
Stage 4: Validation and repair diagnostics
Stage 5: Compiler lowering -> executable megakernel
```

## Stage 1: High-Level Program To Plan

Input is a staged high-level program, usually Torch-like code or an operator graph.
The program describes what is computed, but not how it is partitioned or synchronized inside a persistent megakernel.

Example input:

```python
qkv = x @ w_qkv
q, k, v = split(qkv)
k = apply_rope(k)
v = append_v(v)
attn = attention(q, k, v)
out = rmsnorm(attn + residual)
```

The Stage 1 agent preserves the staged dataflow and emits a compact planning artifact:

```text
1. logical tile stages
2. tile instance space for each stage
3. tensors read and written by each stage
4. logical dependencies between producer and consumer tile instances
5. event tensors needed for non-one-to-one dependencies
6. wait/notify coordinate mappings
```

Use [plan.md](plan.md) as the skill specification for this stage.

Stage 1 must not emit CUDA, TIRX, Relax, atomic operations, spin waits, or encoded event counters.
Those details belong to later compiler stages.

## Stage 2: Plan To KernelSpec DSL

The DSL records the Stage 1 plan in Python objects. It is a structural representation, not a lowering implementation.

The current core concepts are:

```text
TensorSpec:
  logical tensor or buffer name, shape, dtype

EventSpec:
  logical event tensor name, shape, initial logical count, dtype

TileImpl:
  local tile implementation: shared memory allocation and tile-local run body

TileSpec:
  tile name, TileImpl, tile_num, reads, writes, waits, notifies

KernelSpec:
  collection of tensors, events, and tiles for one megakernel
```

`TileImpl` describes local computation only. It should not know the whole megakernel graph or encode global scheduling policy.

Example shape:

```python
stage1 = (
    kernel.tile(
        name="stage1_partial_reduce",
        impl=Stage1ReduceTile(A, B, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N),
        tile_num=(NUM_BLOCK_M, NUM_BLOCK_N, 1),
    )
    .read(A)
    .write(B)
    .notify(row_ready, coord_map=lambda m, n, k: (m,))
)

stage2 = (
    kernel.tile(
        name="stage2_final_reduce",
        impl=Stage2ReduceTile(B, C, BLOCK_M=BLOCK_M, NUM_BLOCK_N=NUM_BLOCK_N),
        tile_num=(NUM_BLOCK_M, 1, 1),
    )
    .read(B)
    .write(C)
    .wait(row_ready, coord_map=lambda m, n, k: (m,), expected=NUM_BLOCK_N)
)
```

## Stage 3: KernelSpec DSL To Relax

The DSL should lower to a Relax-level representation before target-specific TIRX/codegen lowering.

The Relax representation should preserve:

```text
1. tile stages and tile instance spaces
2. tensors and intermediate buffers
3. producer-consumer dependencies
4. logical events and readiness conditions
5. scheduler constraints required by the selected lowering strategy
```

## Stage 4: Validation

Validation is required because the Stage 1 plan may be incomplete or internally inconsistent.
Diagnostics should be structured so an agent can repair the plan and retry.

The validator should check at least:

```text
1. every consumed tensor is external or has a producer
2. every written tensor has a well-defined writer policy
3. dependency producer/consumer names exist
4. dependency coordinate maps match event rank
5. wait.expected matches producer multiplicity for count-based events
6. every consumer has a readiness condition when consuming asynchronous producer output
7. tile_num and index axes are compatible with scheduler limits
8. selected scheduler supports each dependency pattern
9. TileImpl signatures match TileSpec metadata
```

Preferred diagnostic shape:

```text
[MissingProducer]
Tile q_reduce reads tensor qkv_partial, but no producer writes it.

[InvalidEventMap]
Dependency gemm_qkv -> q_reduce uses event q_ready with rank 1,
but notify.coord_map returns rank 2.

[UnsupportedDependency]
The selected static scheduler cannot support this dynamic many-to-many dependency.
```

## Stage 5: Compiler Lowering

After validation, compiler passes decide the concrete implementation:

```text
1. scheduler strategy
2. event tensor layout
3. event initialization encoding
4. notify implementation
5. wait implementation
6. shared/global memory allocation
7. final TIRX/CUDA/runtime integration
```

The hand-written [../demo/tirx_ref_version.py](../demo/tirx_ref_version.py) is a reference for the kind of executable code later stages may generate. It is not the Stage 1 agent output.
