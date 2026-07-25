# Megakernel Partition/Event Plan

This document defines the Stage 1 planning artifact used before writing
`KernelSpec` code.  It is compatible with the current DSL and with PR-style
MegaMoE workloads where the important first step is to make tile and event
relationships explicit.

## Task

Given a staged computation description, produce a logical tile, tensor, and
event plan.  The plan must be sufficient to write the DSL spec layer, but must
not include tile implementation bodies or runtime event mechanics.

The plan describes:

1. tile stages;
2. tile instance spaces;
3. tensor flow;
4. logical events;
5. wait/notify coordinate mappings;
6. dynamic reverse mappings when an event should push consumer tasks.

## Inputs

Input may be:

- Torch-like staged Python code;
- an operator graph;
- pseudocode;
- an existing PR-style megakernel/MegaMoE schedule;
- a written staged dataflow description;
- shape and block-size symbols.

Preserve the staged dataflow.  If dimensions, block sizes, or split factors are
missing, introduce symbolic names such as `NUM_BLOCK_M`, `NUM_BLOCK_N`,
`SPLIT_K`, `NUM_EXPERTS`, or `MAX_TOKENS` instead of inventing constants.

## Output Contract

Output YAML only unless the user asks for explanation or code.

Top-level keys:

```yaml
tiles: {}
tensors: {}
events: {}
dependencies: []
validation: {}
```

Do not output CUDA, TIRX, Relax, Python `KernelSpec`, atomics, spin waits,
memory fences, mbarrier layout, queue implementation, or encoded event-counter
formulas.

## YAML Schema

```yaml
tiles:
  <tile_name>:
    source_stage: <source stage or expression>
    purpose: <short description of local computation>
    tile_impl: <suggested TileImpl class name or null>
    grid: [<m_tiles>, <n_tiles>, <k_tiles>]
    index_axes: [m, n, k]
    reads:
      - tensor: <tensor_name>
        region: <optional region expression or dynamic>
    writes:
      - tensor: <tensor_name>
        region: <optional region expression or dynamic>

tensors:
  <tensor_name>:
    role: input | intermediate | output | routing | workspace
    producer: <tile_name_or_null>
    consumers: [<tile_name>, ...]
    shape: <shape_or_symbolic_shape_or_null>
    dtype: <dtype_or_null>

events:
  <event_name>:
    kind: count
    shape: [<event_extent>, ...]
    init_count: <logical_count_or_expression>
    dtype: int32
    meaning: <what readiness condition this event represents>

dependencies:
  - producer: <producer_tile>
    consumer: <consumer_tile>
    tensor: <tensor_name_or_null>
    event: <event_name>
    relation: <short producer-consumer relation>
    notify:
      tile: <producer_tile>
      coord: [<event_coord_expr>, ...]
      notify_num: <number of participating notify workers>
      remote_rank: <remote rank expression or -1>
    wait:
      tile: <consumer_tile>
      coord: [<event_coord_expr>, ...]
      expected: <event init count for this coordinate>
      inv_coord:
        consumer_num: <fan-out count for one event coordinate>
        tile_coord: [<m_expr>, <n_expr>, <k_expr>]
        required_for_dynamic: true | false

validation:
  status: pass | needs_info
  checks:
    - <short completed check>
  assumptions:
    - <assumption introduced because input omitted details>
  questions:
    - <only include if needed to make the plan actionable>
```

`event.shape` defines the number of event dimensions.  The length of every
notify/wait `coord` must match that number when statically known.

## Mapping To DSL

The plan maps to DSL as follows:

```text
tiles.<name>.grid              -> kernel.tile(..., grid=(m, n, k))
tensors.<name>                 -> kernel.tensor(...)
events.<name>.init_count       -> kernel.event(..., init_count=...)
dependencies[*].notify.coord   -> tile.notify(D(event, coord))
dependencies[*].wait.coord     -> tile.wait(D(event, coord, inv_coord=...))
```

DSL dependency functions use this exact shape:

```python
coord = lambda m, n, k, i: (notify_num, remote_rank, *event_coord)
inv_coord = lambda remote_rank, *event_coord, consumer_i: (
    consumer_num,
    consumer_m,
    consumer_n,
    consumer_k,
)
```

For batch notifies, `notify_num` must stay stable for every `notify_i`, each
produced event coordinate must be inside the event shape, and one tile notify
must not produce duplicate `(remote_rank, event_coord)` entries when statically
checkable.

For a wait, `coord(m, n, k, 0)` must describe exactly one local coordinate:
`notify_num == 1` and `remote_rank == -1`.

For dynamic scheduling, every wait that can be triggered by a producer must have
`inv_coord`.  When statically checkable, each generated consumer coord must
round-trip through the wait `coord`, stay inside the consumer grid, and be
unique for that event coord.  Static scheduling may omit `inv_coord`.

## Planning Procedure

1. Identify user-visible stages in order.
2. Assign one tile stage per preserved stage.
3. Choose each tile grid using `[m, n, k]`; use `1` for unused axes.
4. Record reads and writes for every tile, with known regions when possible.
5. Build tensor producer/consumer metadata.
6. Classify every producer-consumer edge: one-to-one, many-to-one,
   one-to-many, many-to-many, or runtime-routed.
7. Create count events for readiness that cannot be represented by pure stage
   order.
8. Define notify event coordinates from producer tile coordinates.
9. Define wait event coordinates from consumer tile coordinates.
10. For dynamic scheduling, define `inv_coord` from event coordinates back to
    consumer tile coordinates.
11. Validate event dimensions, logical counts, tensor flow, and staged dataflow
    preservation.
12. Return YAML only.

## Dependency Patterns

### One-To-One

Producer tile `(m, n, k)` enables consumer tile `(m, n, k)`.

```yaml
events:
  ready:
    kind: count
    shape: [NUM_M, NUM_N, NUM_K]
    init_count: 1
    dtype: int32
    meaning: producer tile result is ready

dependencies:
  - producer: producer
    consumer: consumer
    tensor: tmp
    event: ready
    relation: same m,n,k
    notify:
      tile: producer
      coord: [m, n, k]
      notify_num: 1
      remote_rank: -1
    wait:
      tile: consumer
      coord: [m, n, k]
      expected: 1
      inv_coord:
        consumer_num: 1
        tile_coord: [m, n, k]
        required_for_dynamic: true
```

DSL:

```python
producer.notify(D(ready, lambda m, n, k, i: (1, -1, m, n, k)))
consumer.wait(D(
    ready,
    lambda m, n, k, i: (1, -1, m, n, k),
    inv_coord=lambda remote_rank, m, n, k, ci: (1, m, n, k),
))
```

### Many-To-One Reduction

Producer tile space `(m, n, 1)` enables one consumer tile `(m, 0, 0)` after all
`n` producers for the same `m` are ready.

```yaml
events:
  row_ready:
    kind: count
    shape: [NUM_M]
    init_count: NUM_N
    dtype: int32
    meaning: all n-block producers for one m are ready

dependencies:
  - producer: partial
    consumer: final
    tensor: partial_out
    event: row_ready
    relation: producer.m == consumer.m, all producer.n
    notify:
      tile: partial
      coord: [m]
      notify_num: 1
      remote_rank: -1
    wait:
      tile: final
      coord: [m]
      expected: NUM_N
      inv_coord:
        consumer_num: 1
        tile_coord: [m, 0, 0]
        required_for_dynamic: true
```

### One-To-Many / Runtime Routed

One ready event coordinate pushes multiple consumers.  This is common in
MegaMoE-style routing, where a runtime routing tensor maps event coordinates to
consumer tile coordinates.

```yaml
events:
  expert_ready:
    kind: count
    shape: [NUM_EXPERTS]
    init_count: <producer_count_per_expert>
    dtype: int32
    meaning: all inputs for one expert are ready

dependencies:
  - producer: dispatch
    consumer: expert_gemm
    tensor: routed_tokens
    event: expert_ready
    relation: runtime routing maps expert id to token tiles
    notify:
      tile: dispatch
      coord: [expert_id_from_routing]
      notify_num: 1
      remote_rank: -1
    wait:
      tile: expert_gemm
      coord: [expert_id]
      expected: <producer_count_per_expert>
      inv_coord:
        consumer_num: tokens_per_expert(expert_id)
        tile_coord: [token_for_expert(expert_id, consumer_i), expert_id, 0]
        required_for_dynamic: true
```

In DSL this may use runtime tensor indexing inside `coord` or `inv_coord`.  The
validator will skip static proofs it cannot make and emit warnings; workload
coverage must prove the runtime path.

## Dynamic Scheduler Planning Notes

For dynamic scheduling, the plan must identify entry tiles and one endpoint tile
indirectly through the dependency graph:

- Entry tiles have no waits and are inserted by the dynamic queue-init kernel.
- A producer notify on event `E` can push any consumer tile that waits on `E` and
  provides `inv_coord`.
- The current lowering supports at most one wait dependency per dynamic tile.
- The current lowering requires exactly one endpoint tile, and that endpoint
  grid must be one tile when statically known.

Do not model pre-notify or complete-notify as separate DSL events.  They are the
lowering implementation of one logical notify dependency.

## Validation Checklist

A good plan should state that it checked:

- every tile has a three-dimensional `[m, n, k]` grid;
- every tensor consumer has a producer or the tensor is an input/routing tensor;
- event coordinate length matches event dimensions;
- `init_count` matches the number of producer notifies per event coordinate;
- dynamic waits that need scheduling have `inv_coord`;
- one-to-many `inv_coord` includes a fan-out count;
- staged dataflow from the input is preserved.
