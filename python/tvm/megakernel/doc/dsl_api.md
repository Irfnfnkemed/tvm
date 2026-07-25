# Megakernel DSL User API

This document describes the public API exposed from `tvm.megakernel.dsl`.
The intended user import is:

```python
from tvm.megakernel.dsl import D, KernelSpec, R, TileImpl, SmemManager
```

The API has two layers:

- Spec layer: `KernelSpec`, `R`, and `D`.  It records tensors, tensor regions,
  logical events, tile grids, and wait/notify dependencies.
- Impl layer: `TileImpl` and `SmemManager`.  It provides the local TIRX body for
  each tile kind.

`TensorSpec`, `EventSpec`, `TileSpec`, and `DependencySpec` are builder return
values.  Users may pass them around, but should not construct them directly.

## KernelSpec

```python
kernel = KernelSpec(name: str, attrs: dict[str, Any] | None = None)
```

`KernelSpec` owns one megakernel specification.  It registers vars, tensors,
events, and tile stages.

### Vars

```python
M = kernel.var("M", dtype="int32", bounds=(1, 4096))
```

Vars are symbolic integer values used in tensor shapes, event shapes, and tile
grids.  Bounds are optional, but validation can use them when it samples
symbolic spaces.

### Tensors

```python
A = kernel.tensor("A", (M, 4096), "float16")
```

A tensor shape may contain integers, vars, or DSL expressions such as
`M.ceildiv(128)`.

A bare tensor in a tile declaration means the accessed region is unknown:

```python
kernel.tile("stage", StageTile(), grid=(M, 1, 1), reads=[A])
```

This is equivalent to a dynamic/unknown region.  It is valid, but region-specific
semantic checks cannot prove bounds or overlap.

### Tensor Regions

```python
access = A.region(lambda m, n, k: R[m, 0:128])
```

`tensor.region(...)` returns an access view with a known mapping from tile
coordinate `(m, n, k)` to a tensor region.  Use this when validation should check
region dimensions, bounds, and producer-consumer overlap.

`R[...]` is the public region builder.  Indices may be integers, slices, vars,
DSL expressions, or lower-time expressions.

### Events

```python
ready = kernel.event("ready", (M,), init_count=lambda m: 1)
```

An event is a logical count tensor.  `len(shape)` is the number of event
dimensions.

`init_count` is required.  It may be:

- an integer, used uniformly for every event coordinate;
- a callable, called with the expanded event coordinate.

Examples:

```python
row_ready = kernel.event("row_ready", (NUM_M,), init_count=lambda m: NUM_N)
expert_ready = kernel.event(
    "expert_ready",
    (NUM_EXPERTS,),
    init_count=lambda expert: tokens_per_expert(expert),
)
```

### Tiles

```python
tile = kernel.tile(
    "stage",
    StageTile(),
    grid=(M, 1, 1),
    reads=[A.region(lambda m, n, k: R[m, 0:128])],
    writes=[B.region(lambda m, n, k: R[m, 0:128])],
    attrs={"notify_scope": "cta"},
)
```

The grid is always three-dimensional and follows `(m, n, k)`.  Use `1` for
unused axes.

Event-related tile attrs used by lowering include:

- `wait_scope`: wait scope, usually `"cta"` or `"warp"`.
- `wait_mask`: warp mask for warp-level waits.
- `notify_scope`: scope participating in notify.
- `notify_scope_id`: concrete scope id, or `-1` for current scope.
- `push_scope`: dynamic pre-notify/push scope.  Defaults to `notify_scope`.
- `push_scope_id`: dynamic push scope id.  Defaults to `notify_scope_id`.
- `push_level`: dynamic queue push granularity: `"thread"`, `"warp"`,
  `"warpgroup"`, or `"cta"`.

## Dependencies: D

`TileSpec.wait` and `TileSpec.notify` only accept dependencies built by `D`:

```python
tile.notify(D(event, coord))
tile.wait(D(event, coord, inv_coord=inv_coord))
```

The forward coordinate function has one shape:

```python
coord(tile_m, tile_n, tile_k, notify_i) -> (notify_num, remote_rank, *event_coord)
```

Meaning:

- `notify_i` is the worker index inside the selected notify scope.
- `notify_num` is the number of participating notify workers.
- `remote_rank == -1` means local memory.  Non-negative values use the remote
  atomic path during lowering.
- `event_coord` must have the same number of values as the event has
  dimensions.

For a notify, lowering calls `coord(m, n, k, notify_i)` for workers in the
selected notify scope and notifies when `notify_i < notify_num`.

For a wait, lowering calls `coord(m, n, k, 0)` and waits on `event_coord`.  A
wait must describe exactly one local event coordinate:

```text
notify_num == 1
remote_rank == -1
```

## Dynamic inv_coord

Dynamic scheduling uses a reverse mapping to push consumers after producer
pre-notify:

```python
inv_coord(remote_rank, *event_coord, consumer_i) -> (consumer_num, tile_m, tile_n, tile_k)
```

Meaning:

- `consumer_i` is the fan-out index from one ready event coordinate to one
  consumer tile.
- `consumer_num` is the total number of consumer tiles for this event
  coordinate.
- Lowering reads `consumer_num` from `consumer_i == 0`, then calls `inv_coord`
  for every consumer index to push concrete tasks.
- For statically checkable dependencies, every generated consumer tile coord
  must map back to the original event coord through the wait `coord`, must be
  inside the consumer tile grid, and must not duplicate another consumer tile
  from the same event coord.

Static scheduling may omit `inv_coord`.  Dynamic scheduling requires `inv_coord`
for waits that can be triggered by producer notifies.

## Dependency Examples

Many-to-one reduction:

```python
row_ready = kernel.event("row_ready", (NUM_M,), init_count=lambda m: NUM_N)

partial = kernel.tile(
    "partial",
    PartialTile(),
    grid=(NUM_M, NUM_N, 1),
    writes=[B.region(lambda m, n, k: R[m, n])],
).notify(D(row_ready, lambda m, n, k, i: (1, -1, m)))

final = kernel.tile(
    "final",
    FinalTile(),
    grid=(NUM_M, 1, 1),
    reads=[B.region(lambda m, n, k: R[m, 0:NUM_N])],
).wait(D(
    row_ready,
    lambda m, n, k, i: (1, -1, m),
    inv_coord=lambda remote_rank, m, consumer_i: (1, m, 0, 0),
))
```

One event coordinate pushing multiple consumers:

```python
expert.wait(D(
    expert_ready,
    lambda m, n, k, i: (1, -1, expert_for_tile(m)),
    inv_coord=lambda remote_rank, expert_id, consumer_i: (
        tokens_per_expert(expert_id),
        token_for_expert(expert_id, consumer_i),
        expert_id,
        0,
    ),
))
```

Runtime routing tensors may appear in dependency coordinate closures:

```python
routing = kernel.tensor("routing", (MAX_TOKENS,), "int32")
producer.notify(D(evt, lambda m, n, k, i: (1, -1, routing[i])))
```

This is intended for MegaMoE-style runtime routing.  Semantic validation cannot
prove all dimension and round-trip properties for runtime tensor indexing, so it
skips those static checks with warnings.  Lowering binds the captured
`TensorSpec` to the corresponding TIRX buffer.

Do not capture `TensorSpec` through default arguments or globals:

```python
# Avoid this.
producer.notify(D(evt, lambda m, n, k, i, routing=routing: (1, -1, routing[i])))
```

## TileImpl

Users subclass `TileImpl` for local parser-style TIRX code.  Only `run()` is
required.

```python
class MyTile(TileImpl):
    @T.inline
    def run(self, m_idx, n_idx, k_idx):
        ...
```

Optional hooks:

- `init_shared_resources(cls, smem_manager)`: class-level setup emitted before
  dispatch.
- `finalize_shared_resources(cls, smem_manager)`: class-level cleanup emitted
  after dispatch.
- `device_init(self, smem_manager, m_idx, n_idx, k_idx)`: per-task setup.
- `host_init(self)`: host-side setup.
- `prefetch(self, m_idx, n_idx, k_idx)`: prefetch before waits.
- `run(self, m_idx, n_idx, k_idx)`: required tile body.

Global dependency policy should not be hand-written inside `TileImpl`; it belongs
in `wait(D(...))` and `notify(D(...))`.

## SmemManager

`SmemManager` is the user boundary for managed shared memory:

```python
buf = smem_manager.alloc(shape, dtype="float32", policy="shared")
smem_manager.wait_all(level="cta")
...
smem_manager.release_all(level="cta")
smem_manager.advance()
```

Policies:

- `"shared"`: participates in phase-based reuse.
- `"persistent"`: live for the whole megakernel.
- `"exclusive"`: requests no physical overlap with other concurrently live
  buffers.

Lowering maps these coarse operations to physical shared-memory layout and
chunk-level mbarriers.  Users should not address physical chunks directly.

## Lowering

```python
from tvm.megakernel.transform import LoweringOptions, lower

mod = lower(kernel, LoweringOptions(schedule="static"))
mod = lower(kernel, LoweringOptions(schedule="dynamic", attrs={"num_threads": 256}))
```

`lower()` validates the DSL graph, validates `TileImpl` accesses, prepares a
lowering plan, validates that plan, and emits a TIRX module.

Dynamic lowering follows the old PR-style megakernel runtime model: queue-init
pushes entry tasks, a scheduler warp dequeues tasks with mbarriers, producer
pre-notify pushes newly ready consumers through `inv_coord`, and complete-notify
finishes the event after `run()`.
