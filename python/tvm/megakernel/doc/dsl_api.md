# Megakernel DSL User API

This document describes the user-facing API for writing a megakernel DSL spec.
The stable import boundary is:

```python
from tvm.megakernel.dsl import D, KernelSpec, R, TileImpl, SmemManager
from tvm.megakernel.transform import LoweringOptions, lower
```

Users normally construct a `KernelSpec`, declare symbolic vars, tensors,
events, and tiles, attach wait/notify dependencies through `D(...)`, implement
local tile bodies by subclassing `TileImpl`, and finally call `lower(...)`.

`TensorSpec`, `EventSpec`, `TileSpec`, and `DependencySpec` are returned by the
builders below.  They are part of the object model and can be passed around, but
users should not instantiate them directly.

## Key Concepts

The DSL is split into a spec layer and an impl layer.

The spec layer describes global megakernel semantics.  It records the tile
space, logical tensors, tensor regions, event tensors, and producer-consumer
dependencies.  This layer should be independent of CUDA atomics, queue layout,
shared-memory chunk layout, or the exact TIRX statements used by lowering.

The impl layer describes the local body of one tile kind.  A `TileImpl` should
only implement the work performed by one concrete tile coordinate.  Global
ordering belongs in `tile.wait(D(...))` and `tile.notify(D(...))`, not inside
`TileImpl.run()`.

A tile grid is always three-dimensional and ordered as `(m, n, k)`.  Use `1` for
unused axes.  Event coordinates can have any number of dimensions, but every
wait/notify mapping for an event must return exactly that many event coordinate
values.

A bare tensor in `reads` or `writes` means the accessed region is unknown or
runtime-dynamic.  A tensor region built with `tensor.region(...)` gives
validation a concrete mapping from tile coordinate to tensor region.

## `KernelSpec`

```python
kernel = KernelSpec(name: str, attrs: dict[str, Any] | None = None)
```

`KernelSpec` owns one megakernel spec.  All vars, tensors, events, and tile
stages used by a kernel should be created from the same `KernelSpec` instance.
Validation rejects cross-kernel ownership mismatches.

Parameters:

- `name`: logical kernel name, used by lowering as the generated function name.
- `attrs`: optional kernel-level metadata.  Current lowering does not require
  user attrs for common static/dynamic schedules.

Useful methods:

- `kernel.var(...)`: declare a symbolic integer.
- `kernel.tensor(...)`: declare a logical tensor parameter/workspace.
- `kernel.event(...)`: declare a logical readiness counter tensor.
- `kernel.tile(...)`: declare one tile stage.
- `kernel.validate()`: run validation; it returns the kernel on success and raises on errors.
- `kernel.lower(options=None)`: convenience wrapper around `lower(kernel,
  options)`.

Example:

```python
kernel = KernelSpec("row_reduce")
rows = kernel.var("rows", bounds=(1, 4096))
cols = kernel.var("cols", bounds=(1, 4096))
```

## `kernel.var`

```python
var = kernel.var(name: str, dtype: str = "int32", bounds: tuple[int, int] | None = None)
```

Declares a symbolic integer that can be used in tensor shapes, event shapes,
tile grids, region expressions, event `init_count`, and TileImpl constructors.

Parameters:

- `name`: unique symbolic variable name inside the kernel.
- `dtype`: integer dtype used during lowering, normally `"int32"`.
- `bounds`: optional closed interval `[min, max]` for validation samples and
  lowering-time capacity estimates.  Both endpoints are included, so
  `bounds=(1, 4096)` means `1 <= value <= 4096`.  Bounds must satisfy
  `0 < min <= max`.

Calling rules and notes:

- Duplicate var names are rejected.
- Shapes and grids may use integers, vars, or DSL expressions built from vars.
- Bounds are important when validation samples symbolic spaces and when
  lowering needs a static capacity, such as shared-memory allocation size in
  `device_init`.  Validation may sample the lower endpoint, midpoint, and upper
  endpoint.
- Runtime generated functions receive symbolic vars as scalar parameters when
  they are needed by tensor shapes, event shapes, queue initialization, or tile
  bodies.

Example:

```python
M = kernel.var("M", bounds=(1, 1024))
N = kernel.var("N", bounds=(1, 1024))
BM = 16
num_m_tiles = M.ceildiv(BM)
```

## `kernel.tensor`

```python
tensor = kernel.tensor(name: str, shape, dtype: str)
```

Declares a logical tensor.  The returned tensor object is used in tile
`reads`/`writes`, region declarations, TileImpl constructors, and selected
runtime dependency routing expressions.

Parameters:

- `name`: unique tensor name inside the kernel.
- `shape`: one integer/var/expression, or a tuple/list of them.
- `dtype`: TVM/TIR dtype string such as `"float16"`, `"float32"`, or `"int32"`.

Calling rules and notes:

- Duplicate tensor names are rejected.
- `shape=(M,)` and `shape=M` both mean a one-dimensional tensor.
- Passing a bare tensor in a tile access means the region is unknown/dynamic.
  This is valid and intentionally equivalent to the old explicit dynamic region
  notation.
- Capturing a `TensorSpec` in a `TileImpl` constructor is supported.  Lowering
  replaces it with the corresponding TIRX buffer when emitting hooks.
- Runtime dependency routing may use tensor indexing in a closure, for example
  `routing[i]`.  Validation warns and skips proofs that depend on the runtime
  value.

Example:

```python
A = kernel.tensor("A", (M, N), "float16")
B = kernel.tensor("B", (M,), "float32")
routing = kernel.tensor("routing", (M,), "int32")
```

## `tensor.region` and `R`

```python
access = tensor.region(lambda m, n, k: R[...])
```

`tensor.region(...)` creates a known tile-to-tensor access view.  Use it in
`reads` and `writes` when the region can be described statically from tile
coordinate `(m, n, k)`.

Parameters:

- `region_from_tile`: callable receiving `(m, n, k)` and returning an `R[...]`
  region expression.

Calling rules and notes:

- The callable must accept exactly the tile coordinate convention `(m, n, k)`.
- The returned region must have the same number of dimensions as the tensor.
- `R[...]` entries may be integer points or slices.  Slice syntax follows
  Python semantics: `R[start:stop]` is half-open `[start, stop)`.  Slice bounds
  may use integers, vars, and DSL expressions.
- A bare tensor in `reads` or `writes` means unknown region, so validation cannot
  prove precise bounds, overlap, or region coverage for that access.
- Region declarations describe the semantic access set.  The actual TIRX body in
  `TileImpl` is checked separately against declared reads/writes when it is
  statically visible.

Example:

```python
BM = 16
A_tile = A.region(lambda m, n, k: R[m * BM : (m + 1) * BM, 0:N])
B_tile = B.region(lambda m, n, k: R[m * BM : (m + 1) * BM])
```

Unknown/dynamic region:

```python
kernel.tile("routed", RoutedTile(A), grid=(M, 1, 1), reads=[A])
```

## `kernel.event`

```python
event = kernel.event(name: str, shape, init_count, dtype: str = "int32", attrs=None)
```

Declares a logical readiness event.  Conceptually, an event is a count tensor:
each event coordinate becomes ready after its counter receives enough producer
notifications.

Parameters:

- `name`: unique event name inside the kernel.
- `shape`: event tensor shape.  The number of shape entries is the number of
  event dimensions.
- `init_count`: required readiness count.  It may be an integer, `VarSpec`,
  `ExprSpec`, or callable.
- `dtype`: counter dtype, normally `"int32"`.
- `attrs`: optional event-level metadata.

Calling rules and notes:

- `init_count` has no default.  The spec must say how many producer
  notifications are required.
- Integer and symbolic `init_count` values are normalized to a callable
  internally, so all later validation/lowering paths see one form.
- A callable `init_count` is called with the expanded event coordinate.  It must
  return an integer, var, or expression.
- Use a callable when readiness count varies per event coordinate, such as
  per-expert token counts.
- Validation checks statically provable event counts against producer notifies.
  Runtime-routed cases may require workload tests.

Examples:

```python
row_ready = kernel.event("row_ready", (M,), init_count=N)

expert_ready = kernel.event(
    "expert_ready",
    (NUM_EXPERTS,),
    init_count=lambda expert: tokens_per_expert(expert),
)
```

## `kernel.tile`

```python
tile = kernel.tile(
    name: str,
    impl: TileImpl,
    grid: tuple[ExprLike, ExprLike, ExprLike],
    reads: list[TensorSpec] | None = None,
    writes: list[TensorSpec] | None = None,
    attrs: dict[str, Any] | None = None,
)
```

Declares one logical tile stage.  The returned `TileSpec` is then chained with
`.wait(...)` and `.notify(...)` calls.

Parameters:

- `name`: unique tile-stage name inside the kernel.
- `impl`: `TileImpl` instance implementing one tile body.
- `grid`: three-dimensional tile space `(m, n, k)`.
- `reads`: logical tensor accesses read by this tile.
- `writes`: logical tensor accesses written by this tile.
- `attrs`: optional scheduling/lowering metadata.

Calling rules and notes:

- `grid` must have exactly three dimensions.  Use `1` for unused axes.
- `reads` and `writes` entries must be tensors or tensor-region views from this
  kernel.
- The order of tile declarations gives lowering a stable tile kind order, but
  ordering semantics should still be expressed through events.
- Tile dependency methods only accept `D(...)` objects:
  `tile.wait(D(...))` and `tile.notify(D(...))`.

Event-related attrs currently used by lowering:

- `wait_scope`: wait scope, usually `"cta"` or `"warp"`.
- `wait_mask`: warp mask for warp-level waits.
- `notify_scope`: scope participating in notify.
- `notify_scope_id`: concrete scope id, or `-1` for current scope.
- `push_scope`: dynamic pre-notify/push scope.  Defaults to `notify_scope`.
- `push_scope_id`: dynamic push scope id.  Defaults to `notify_scope_id`.
- `push_level`: dynamic queue push granularity: `"thread"`, `"warp"`,
  `"warpgroup"`, or `"cta"`.

Example:

```python
partial = kernel.tile(
    "partial",
    PartialTile(A, partial_buf, N),
    grid=(M, N, 1),
    reads=[A.region(lambda m, n, k: R[m, n])],
    writes=[partial_buf.region(lambda m, n, k: R[m, n])],
    attrs={"notify_scope": "cta"},
)
```

## `D` Dependencies

```python
dep = D(event, coord, inv_coord=inv_coord)
tile.notify(dep)
tile.wait(dep)
```

`D(...)` builds the only dependency object accepted by tile wait/notify APIs.
It ties a tile stage to one logical event through a coordinate mapping.

Parameters:

- `event`: event created by `kernel.event(...)`.
- `coord`: forward coordinate mapping.
- `inv_coord`: optional keyword-only reverse coordinate mapping used by dynamic
  scheduling to push consumer tasks after a producer notification.

The forward mapping has exactly one API shape:

```python
coord(tile_m, tile_n, tile_k, i) -> (coord_count, remote_rank, *event_coord)
```

Meaning:

- `tile_m`, `tile_n`, `tile_k`: current tile coordinate.
- `i`: mapping index.  For notify lowering, this is carried by the selected
  notify scope worker index.  Waits call this mapping with `i == 0`.
- `coord_count`: number of event coordinates generated by this mapping.
- `remote_rank`: destination rank for the event operation.  `-1` means local.
- `event_coord`: coordinate inside the target event tensor.

Notify rules:

- `tile.notify(D(...))` may use `coord_count > 1` when one tile maps to
  multiple event coordinates.
- Lowering evaluates the mapping for workers in the selected notify scope and
  emits the event operation when `i < coord_count`.
- When statically provable, validation checks `coord_count`, event coordinate
  bounds, event coordinate dimension count, and duplicate notify coordinates.

Wait rules:

- `tile.wait(D(...))` must describe one local event coordinate:
  `coord_count == 1` and `remote_rank == -1`.
- The wait event coordinate must have the same dimension count as the event.
- Static scheduling may omit `inv_coord`.
- Dynamic scheduling requires `inv_coord` for waits that can be triggered by
  producer notifies.

The reverse mapping used by dynamic scheduling has this shape:

```python
inv_coord(remote_rank, *event_coord, consumer_i) -> (consumer_count, tile_m, tile_n, tile_k)
```

Meaning:

- `remote_rank` and `event_coord`: the event coordinate that just became ready.
- `consumer_i`: fan-out index from one ready event coordinate to one consumer
  tile.
- `consumer_count`: total number of consumer tiles generated from this event
  coordinate.
- `tile_m`, `tile_n`, `tile_k`: concrete consumer tile coordinate to push.

Calling rules and notes:

- `inv_coord` is attached to the waiting dependency, because it describes how a
  ready event maps back to consumer tile coordinates.
- Lowering reads `consumer_count` from `consumer_i == 0`, then calls `inv_coord`
  for each `consumer_i` in `[0, consumer_count)`.
- When statically provable, validation checks that generated consumer
  coordinates are inside the consumer grid, unique for the same event
  coordinate, and round-trip through the wait `coord`.
- Runtime tensor indexing inside dependency closures is supported for lowering,
  but validation warns and skips proofs that depend on unknown runtime values.
- Do not capture spec tensors through default arguments or globals.  Capture
  them through normal closure variables so lowering can bind them.

Many-to-one example:

```python
row_ready = kernel.event("row_ready", (M,), init_count=N)

partial.notify(D(row_ready, lambda m, n, k, i: (1, -1, m)))
final.wait(D(
    row_ready,
    lambda m, n, k, i: (1, -1, m),
    inv_coord=lambda remote_rank, m, consumer_i: (1, m, 0, 0),
))
```

One-to-many runtime-routed example:

```python
routing = kernel.tensor("routing", (MAX_TOKENS,), "int32")
expert_ready = kernel.event("expert_ready", (NUM_EXPERTS,), init_count=1)

producer.notify(D(
    expert_ready,
    lambda m, n, k, i: (1, -1, routing[m]),
))

consumer.wait(D(
    expert_ready,
    lambda m, n, k, i: (1, -1, n),
    inv_coord=lambda remote_rank, expert_id, consumer_i: (
        tokens_per_expert(expert_id),
        token_for_expert(expert_id, consumer_i),
        expert_id,
        0,
    ),
))
```

Unsupported capture style:

```python
# Avoid this: default-argument capture hides the spec tensor from binding.
producer.notify(D(evt, lambda m, n, k, i, routing=routing: (1, -1, routing[i])))
```

## `TileImpl`

```python
class MyTile(TileImpl):
    @T.inline
    def run(self, m_idx, n_idx, k_idx):
        ...
```

Subclass `TileImpl` to define parser-style TIRX code for one tile kind.  Only
`run()` is required.  Hooks that emit device statements should be decorated with
`@T.inline`.

Supported hooks:

- `init_shared_resources(cls, smem_manager)`: optional class-level setup emitted
  before tile dispatch.
- `finalize_shared_resources(cls, smem_manager)`: optional class-level cleanup
  emitted after tile dispatch.
- `device_init(self, smem_manager, m_idx, n_idx, k_idx)`: optional per-task
  device setup.  Shared-memory allocation normally happens here.
- `host_init(self)`: optional host-side setup.
- `prefetch(self, m_idx, n_idx, k_idx)`: optional prefetch emitted before waits.
- `run(self, m_idx, n_idx, k_idx)`: required tile body.

Calling rules and notes:

- The hook coordinate arguments are the concrete tile coordinate `(m, n, k)`.
- `TileImpl` should not manually perform global event waits/notifies; use tile
  dependencies for that.
- Constructors may capture `TensorSpec`, `VarSpec`, and `ExprSpec` objects.
  Lowering replaces captured tensors with TIRX buffers.  Captured symbolic
  values become sampled values during validation, upper-bound capacities during
  `device_init`, and runtime TIR values during `prefetch`/`run`.
- For managed shared memory, use the `SmemManager` passed to the hook instead of
  declaring physical shared-memory chunks directly.

Example:

```python
class ReduceTile(TileImpl):
    def __init__(self, source, output, cols):
        self.source = source
        self.output = output
        self.cols = cols

    @T.inline
    def device_init(self, smem_manager, m, n, k):
        self.smem = smem_manager.alloc((self.cols,), "float32", policy="shared")

    @T.inline
    def prefetch(self, m, n, k):
        Tx.copy(self.smem[0:self.cols], self.source[m, n, 0:self.cols])

    @T.inline
    def run(self, m, n, k):
        self.output[m, n] = T.cast("float32", self.smem[0])
```

## `SmemManager`

```python
buf = smem_manager.alloc(
    shape,
    dtype="float32",
    strides=None,
    scope="shared.dyn",
    align=0,
    buffer_type="",
    axis_separators=None,
    layout="default",
    policy="shared",
)
smem_manager.commit()
smem_manager.wait_all(level="cta")
smem_manager.release_all(level="cta")
smem_manager.advance()
```

`SmemManager` is the user boundary for managed shared memory.  It records
logical allocations and coarse lifetime operations; lowering maps them to
physical shared-memory storage and chunk-level synchronization.

`alloc` parameters:

- `shape`: logical shared-memory buffer shape.
- `dtype`: element dtype.
- `strides`, `scope`, `align`, `buffer_type`, `axis_separators`, `layout`: TIRX
  buffer declaration options forwarded by lowering.
- `policy`: allocation lifetime/reuse policy.

Policies:

- `"shared"`: participates in phase-based reuse.
- `"persistent"`: live for the whole megakernel.
- `"exclusive"`: requests no physical overlap with other concurrently live
  buffers.

Calling rules and notes:

- Allocate through the manager in `device_init` when the buffer is per task.
- Use `wait_all` before consuming prefetched/produced shared-memory data that
  may still be in flight.
- Use `release_all` and `advance` to let lowering reuse shared-memory chunks
  across phases.
- Users should not rely on physical chunk ids or emitted mbarrier details.

## `lower` and `LoweringOptions`

```python
mod = lower(kernel, LoweringOptions(schedule="static"))
# or
mod = kernel.lower(LoweringOptions(schedule="dynamic"))
```

`lower(...)` validates and lowers a `KernelSpec` to a TVM `IRModule` containing
the persistent megakernel and supporting initialization functions such as queue
initialization.

`LoweringOptions` fields:

- `schedule`: `"static"` or `"dynamic"`.
- `smem_max_bytes`: shared-memory capacity used by the shared-memory planner.
- `smem_chunk_size`: chunk size used by managed shared-memory lowering.
- `emit_smem_markers`: whether to emit shared-memory lifetime markers.
- `attrs`: optional lowering metadata.

Calling rules and notes:

- Static scheduling emits deterministic tile dispatch over the declared tile
  graph.
- Dynamic scheduling uses entry tiles, event pre-notify, queue push, and endpoint
  completion logic inferred from the dependency graph.
- The current dynamic queue lowering expects enough initial entry tasks relative
  to `sm_count`; robust empty-queue handling is still a lowering/runtime TODO.
- Symbolic vars used by generated functions become scalar parameters.
- Lowering intentionally owns CUDA/TIRX implementation details.  The DSL spec
  should stay focused on logical tile/data/event relationships.
