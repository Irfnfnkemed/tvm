---
name: torch-megakernel-partition-event
description: Analyze a staged Torch program or operator graph and output only the Stage 1 megakernel tile partition and logical event dependency plan. Use when converting high-level staged computation into megakernel planning YAML before KernelSpec, Relax, TIRX, CUDA, or runtime lowering.
---

# Torch Megakernel Partition/Event Planner

## Task

Given a staged Torch program, produce a Stage 1 megakernel plan.

The output is a logical planning artifact. It describes tile stages, tensor flow, event tensors, and wait/notify coordinate mappings. It must not describe low-level event implementation.

## Inputs

The user may provide any of these:

```text
1. Torch-like staged Python code
2. an operator graph
3. a written staged dataflow description
4. shape and block-size symbols for an existing workload
```

Preserve the staged dataflow. Do not collapse separate user-visible stages into one mathematically equivalent operation unless the user explicitly asks for fusion beyond the given stages.

If dimensions, block sizes, or split factors are missing, introduce symbolic names such as `NUM_BLOCK_M`, `NUM_BLOCK_N`, `SPLIT_K`, or `NUM_HEADS` instead of inventing numeric constants.

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

Do not output CUDA, TIRX, Relax, Python `KernelSpec`, atomic operations, spin waits, memory fences, or encoded event-counter formulas.

## YAML Schema

```yaml
tiles:
  <tile_name>:
    torch_stage: <source stage or expression>
    purpose: <short description of local computation>
    tile_impl: <suggested TileImpl class name or null>
    tile_num: [<m_tiles>, <n_tiles>, <k_tiles>]
    index_axes: [m, n, k]
    reads: [<tensor_name>, ...]
    writes: [<tensor_name>, ...]

# Include user inputs, intermediates, and outputs.
tensors:
  <tensor_name>:
    role: input | intermediate | output
    producer: <tile_name_or_null>
    consumers: [<tile_name>, ...]
    shape: <shape_or_symbolic_shape_or_null>
    dtype: <dtype_or_null>

# Events represent logical readiness only.
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
      coord_map: [<event_coord_expr>, ...]
    wait:
      tile: <consumer_tile>
      coord_map: [<event_coord_expr>, ...]
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

For one-to-one dependencies, `event` may be `null` only if no asynchronous readiness event is required by the target scheduler. Otherwise emit an event with `init: 1`.

## Planning Procedure

1. Identify staged operations in the same order as the input.
2. Assign one logical tile stage per preserved stage.
3. Pick `tile_num` using `[m, n, k]` axes. Use extent `1` for unused axes.
4. Record reads and writes for every tile.
5. Build tensor producer/consumer metadata.
6. For every producer-consumer edge, classify dependency multiplicity.
7. Create events for non-one-to-one dependencies, especially many-to-one readiness.
8. Define notify and wait coordinate maps in consumer-visible logical coordinates.
9. Validate rank, multiplicity, and staged dataflow preservation.
10. Return YAML only.

## Dependency Rules

Use a count event when several producer tile instances must finish before one consumer tile instance can run.

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
      coord_map: [m]
    wait:
      tile: stage2_final_reduce
      coord_map: [m]
      expected: NUM_BLOCK_N
```

The rank of `notify.coord_map` and `wait.coord_map` must equal the rank of `events.<event>.shape`.

`wait.expected` must equal the number of producer tile instances mapped to the same event coordinate.

## What Not To Emit

Never expose low-level implementation details in Stage 1 output:

```text
EVENT_BASE
atomic add/sub
red.async
ld_global_acquire
spin wait loops
nano_sleep
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
    torch_stage: B = reduce_n_blocks(A)
    purpose: reduce each A[m-block, n-block] into B[m-block, n]
    tile_impl: Stage1ReduceTile
    tile_num: [NUM_BLOCK_M, NUM_BLOCK_N, 1]
    index_axes: [m, n, k]
    reads: [A]
    writes: [B]

  stage2_final_reduce:
    torch_stage: C = reduce_block_results(B)
    purpose: reduce all partial values for each m-block
    tile_impl: Stage2ReduceTile
    tile_num: [NUM_BLOCK_M, 1, 1]
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
      coord_map: [m]
    wait:
      tile: stage2_final_reduce
      coord_map: [m]
      expected: NUM_BLOCK_N

validation:
  status: pass
  checks:
    - every tile has tile_num and index_axes
    - every tensor consumer has a producer or is an input
    - row_ready has rank 1 and both coord maps have rank 1
    - wait.expected equals NUM_BLOCK_N producer tiles per m-block
    - staged dataflow A -> B -> C is preserved
  assumptions: []
  questions: []
```
