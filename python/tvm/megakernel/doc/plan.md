---
name: megakernel-partition-event
description: >
  Analyze a staged computation description and output only the Stage 1
  megakernel tile partition and logical event dependency plan. Use when
  converting high-level staged computation into a megakernel DSL plan before
  TileImpl authoring, validation, and TIRX megakernel lowering.
---

# Megakernel Partition/Event Planner

## Task

Given a staged computation description, produce the Stage 1 megakernel plan.

This plan is Step 2 of the megakernel DSL workflow.  The agent should
plan tile partitioning and logical event dependencies.  The output should be
sufficient for a user or agent to write the megakernel DSL in the next step.

The plan describes:

```text
1. tile stages
2. tile instance spaces
3. tensor flow between stages
4. logical events
5. wait/notify coordinate mappings
```

It must not describe low-level event implementation or tile implementation
bodies.

## Inputs

The user may provide any of these:

```text
1. natural language staged computation description
2. Torch-like staged Python code
3. pseudocode
4. an operator graph
5. a written staged dataflow description
6. shape and block-size symbols for an existing workload
```

Preserve the staged dataflow.  Do not collapse separate user-visible stages
into one mathematically equivalent operation unless the user explicitly asks for
fusion beyond the given stages.

If dimensions, block sizes, or split factors are missing, introduce symbolic
names such as `x` instead of
inventing numeric constants.

## Output Contract

Output YAML only unless the user asks for explanation or code.

The top-level YAML keys are:

```yaml
tiles: {}
tensors: {}
events: {}
dependencies: []
validation: {}
```

Do not output CUDA, TIRX, Python `KernelSpec`, `TileImpl` bodies, atomic
operations, spin waits, memory fences, or encoded event-counter formulas.

## YAML Schema

```yaml
tiles:
  <tile_name>:
    source_stage: <source stage or expression>
    purpose: <short description of local computation>
    tile_impl: <suggested TileImpl class name or null>
    grid: [<m_tiles>, <n_tiles>, <k_tiles>]
    index_axes: [m, n, k]
    reads: [<tensor_name>, ...]
    writes: [<tensor_name>, ...]

tensors:
  <tensor_name>:
    role: input | intermediate | output
    producer: <tile_name_or_null>
    consumers: [<tile_name>, ...]
    shape: <shape_or_symbolic_shape_or_null>
    dtype: <dtype_or_null>

events:
  <event_name>:
    kind: count
    shape: [<event_extent>, ...]
    init: <logical_count>
    dtype: int32
    meaning: <what readiness condition this event represents>

dependencies:
  - producer: <producer_tile>
    consumer: <consumer_tile>
    tensor: <tensor_name>
    event: <event_name_or_null_for_one_to_one>
    relation: <short producer-consumer relation>
    notify:
      tile: <producer_tile>
      coord: [<event_coord_expr>, ...]
    wait:
      tile: <consumer_tile>
      coord: [<event_coord_expr>, ...]
      expected: <logical_count>

validation:
  status: pass | needs_info
  checks:
    - <short completed check>
  assumptions:
    - <assumption introduced because input omitted details>
  questions:
    - <only include if needed to make the plan actionable>
```

The YAML is a planning artifact, not the final Python DSL.  In Step 3 of the
workflow, use [dsl_api.md](dsl_api.md) as the API reference.
`events.<name>.init` maps naturally to `EventSpec.init_count`, and each
wait/notify entry maps to `TileSpec.wait(...)` or `TileSpec.notify(...)`.

For one-to-one dependencies, `event` may be `null` only if no logical readiness
event is required.  Otherwise emit an event with `init: 1`.

## Planning Procedure

1. Identify staged operations in the same order as the input.
2. Assign one logical tile stage per preserved stage.
3. Pick `grid` using `[m, n, k]` axes.  Use extent `1` for unused axes.
4. Record reads and writes for every tile.
5. Build tensor producer/consumer metadata.
6. Classify producer-consumer dependency multiplicity.
7. Create events for dependencies that need logical readiness tracking.
8. Define notify and wait coordinate maps.
9. Validate rank, multiplicity, and staged dataflow preservation.
10. Return YAML only.

## Dependency Rules

Use a count event when several producer tile instances must finish before one
consumer tile instance can run.

Example pattern:

```text
producer tile space: (m, n, k)
consumer tile space: (m, 1, 1)
condition: consumer(m) waits for all producer(m, n, 0)
```

Plan it as:

```yaml
events:
  row_ready:
    kind: count
    shape: [NUM_BLOCK_M]
    init: NUM_BLOCK_N
    dtype: int32
    meaning: all N-block partial reductions for one M-block are ready

dependencies:
  - producer: stage1_partial_reduce
    consumer: stage2_final_reduce
    tensor: B
    event: row_ready
    relation: producer.m == consumer.m, all producer.n
    notify:
      tile: stage1_partial_reduce
      coord: [m]
    wait:
      tile: stage2_final_reduce
      coord: [m]
      expected: NUM_BLOCK_N
```

The rank of `notify.coord` and `wait.coord` must equal the rank of
`events.<event>.shape`.

`wait.expected` must equal the number of producer tile instances mapped to the
same event coordinate.

## What Not To Emit

Never expose low-level implementation details in Stage 1 output like:

```text
atomic add/sub
spin wait loops
CTA role split
shared memory layout
TIRX function bodies
CUDA source snippets
```

Those are compiler-lowering decisions.

## Minimal Example

Input:

```python
B = reduce_n_blocks(A)
C = reduce_block_results(B)
```

Output:

```yaml
tiles:
  stage1_partial_reduce:
    source_stage: B = reduce_n_blocks(A)
    purpose: reduce each A[m-block, n-block] into B[m-block, n]
    tile_impl: Stage1ReduceTile
    grid: [NUM_BLOCK_M, NUM_BLOCK_N, 1]
    index_axes: [m, n, k]
    reads: [A]
    writes: [B]

  stage2_final_reduce:
    source_stage: C = reduce_block_results(B)
    purpose: reduce all partial values for each m-block
    tile_impl: Stage2ReduceTile
    grid: [NUM_BLOCK_M, 1, 1]
    index_axes: [m, n, k]
    reads: [B]
    writes: [C]

tensors:
  A:
    role: input
    producer: null
    consumers: [stage1_partial_reduce]
    shape: [M, N]
    dtype: float32
  B:
    role: intermediate
    producer: stage1_partial_reduce
    consumers: [stage2_final_reduce]
    shape: [M, NUM_BLOCK_N]
    dtype: float32
  C:
    role: output
    producer: stage2_final_reduce
    consumers: []
    shape: [M, 1]
    dtype: float32

events:
  row_ready:
    kind: count
    shape: [NUM_BLOCK_M]
    init: NUM_BLOCK_N
    dtype: int32
    meaning: all partial reductions for one m-block have been written to B

dependencies:
  - producer: stage1_partial_reduce
    consumer: stage2_final_reduce
    tensor: B
    event: row_ready
    relation: producer.m == consumer.m, all producer.n
    notify:
      tile: stage1_partial_reduce
      coord: [m]
    wait:
      tile: stage2_final_reduce
      coord: [m]
      expected: NUM_BLOCK_N

validation:
  status: pass
  checks:
    - every tile has grid and index_axes
    - every tensor consumer has a producer or is an input
    - row_ready has rank 1 and both coord maps have rank 1
    - wait.expected equals NUM_BLOCK_N producer tiles per m-block
    - staged dataflow A -> B -> C is preserved
  assumptions: []
  questions: []
```
