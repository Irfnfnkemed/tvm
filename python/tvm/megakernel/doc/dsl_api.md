# Megakernel DSL User API

This document lists the user-facing API in `tvm.megakernel.dsl`.

## Public Entry Points

The top-level `tvm.megakernel.dsl` module exposes the user entry points only:

```python
from tvm.megakernel.dsl import KernelSpec, R, TileImpl, SmemManager
```

The DSL has two layers:

- Spec layer: `KernelSpec` and `R`.  This layer describes tile stages, tensor
  inputs/outputs, logical events, and wait/notify dependencies.  Objects such
  as `TensorSpec`, `EventSpec`, and `TileSpec` are returned by `KernelSpec`
  builder methods; users should not construct them directly.
- Impl layer: `TileImpl` and `SmemManager`.  This layer connects a logical tile
  to the concrete implementation of that tile.

Internal spec helper types live under `tvm.megakernel.dsl.spec`.  Internal impl
helper types live under `tvm.megakernel.dsl.impl`.  The top-level module should
remain a small user API facade.

The spec layer corresponds to Step 3 in [workflow.md](workflow.md).  The impl
layer corresponds to Step 4.

## `KernelSpec`

```python
kernel = KernelSpec(name: str, attrs: dict[str, Any] | None = None)
```

Creates one megakernel spec.  Symbolic variables, tensors, events, and tiles
are registered on this object.

Parameters:

- `name`: unique name for the megakernel spec.
- `attrs`: optional metadata reserved for later passes.

Example:

```python
kernel = KernelSpec("two_stage_reduce", attrs={"target": "sm90"})
```

## `KernelSpec.var`

```python
var = kernel.var(name: str, dtype: str = "int32", bounds: tuple[int, int] | None = None)
```

Registers a symbolic integer variable that can be used in tensor shapes, event
shapes, and grid shapes.  The current TIRX lowering emits each symbolic
variable as a local symbolic variable in the PrimFunc body using the `VarSpec`
dtype, for example `M = T.int32()`, instead of exposing it as a kernel
parameter.

Parameters:

- `name`: symbolic variable name, unique inside the kernel.
- `dtype`: scalar dtype used by lowering.  Defaults to `"int32"`.
- `bounds`: optional inclusive `(min, max)` bounds for this symbolic value.  Passes that need static allocation or bounded sampling require this.

Returns: `VarSpec`.

Example:

```python
M = kernel.var("M", dtype="int32", bounds=(1, 1024))
A = kernel.tensor("A", shape=(M, 1024), dtype="float32")
```

## `KernelSpec.tensor`

```python
tensor = kernel.tensor(name: str, shape: ShapeType, dtype: str)
```

Registers a logical tensor.

Parameters:

- `name`: tensor name, unique inside the kernel.
- `shape`: tensor shape.  Each dimension can be an `int`, `VarSpec`, or a small `VarSpec` expression such as `M + 1` or `M.ceildiv(128)`.
- `dtype`: tensor element type.

Returns: `TensorSpec`.

Example:

```python
bs = kernel.var("bs", bounds=(1, 1024))
A = kernel.tensor("A", shape=(bs, 1024), dtype="float32")
```

## `KernelSpec.event`

```python
event = kernel.event(
    name: str,
    shape: ShapeType,
    init_count: int | Callable[..., int],
    dtype: str = "int32",
    attrs: dict[str, Any] | None = None,
)
```

Registers a logical event tensor.

Parameters:

- `name`: event name, unique inside the kernel.
- `shape`: event tensor shape.  Each dimension can be an `int`, `VarSpec`, or a small `VarSpec` expression such as `M + 1` or `M.ceildiv(128)`.
- `init_count`: logical non-negative count for each event coordinate.  This can be a single
  integer for a uniform count, or a callable whose arguments are the expanded
  event coordinate.
- `dtype`: event storage dtype.  Defaults to `"int32"`.
- `attrs`: optional metadata reserved for later passes.

Returns: `EventSpec`.

Examples:

```python
evt1 = kernel.event(
    "evt1",
    shape=(100,),
    init_count=88,
)

evt2 = kernel.event(
    "evt2",
    shape=(100, 200),
    init_count=lambda i, j: i + j,
)
```

## `KernelSpec.tile`

```python
tile = kernel.tile(
    name: str,
    impl: TileImpl,
    grid: GridType,
    reads: list[TensorSpec] | None = None,
    writes: list[TensorSpec] | None = None,
    attrs: dict[str, Any] | None = None,
)
```

Registers one logical tile stage.

Parameters:

- `name`: tile stage name, unique inside the kernel.
- `impl`: local tile implementation object.
- `grid`: grid shape on `(m, n, k)` axes.  Use `1` for unused axes.
- `reads`: tensors or tensor region accesses read by this tile.  Use `tensor.region(...)` when semantic region validation should check the access.
- `writes`: tensors or tensor region accesses written by this tile.  Use `tensor.region(...)` when semantic region validation should check the access.
- `attrs`: optional metadata reserved for later passes.

Returns: `TileSpec`.

Example:

```python
bs = kernel.var("bs", bounds=(1, 1024))
tile_a = kernel.tile(
    "tile_a",
    tile_a_impl,
    grid=(bs, 16, 1),
    reads=[A.region(lambda m, n, k: R[m, n])],
    writes=[B.region(lambda m, n, k: R[m, n])],
)
```

## `TileSpec.wait`

```python
tile.wait(
    event: EventSpec,
    coord: CoordMapType,
    inverse_coord: CoordMapType | None = None,
)
```

Declares that this tile waits on `event` at the coordinate produced by
`coord`.

Parameters:

- `event`: event to wait on.
- `coord`: callable mapping tile index `(m, n, k)` to an event
  coordinate.  If it is a tuple/list instead of a callable, it is used directly
  as the event coordinate.
- `inverse_coord`: optional inverse mapping from event coordinate to
  consumer tile index.  Dynamic scheduling requires this for every wait; static
  scheduling does not.

Returns: `TileSpec`.

Example:

```python
tile_b.wait(
    event=evt1,
    coord=lambda m, n, k: (m,),
)
```

## `TileSpec.notify`

```python
tile.notify(event: EventSpec, coord: CoordMapType)
```

Declares that this tile notifies `event` at the coordinate produced by
`coord`.

Parameters:

- `event`: event to notify.
- `coord`: callable mapping tile index `(m, n, k)` to an event
  coordinate.  If it is a tuple/list instead of a callable, it is used directly
  as the event coordinate.

Returns: `TileSpec`.

Example:

```python
tile_a.notify(
    event=evt1,
    coord=lambda m, n, k: (m,),
)
```

## `SmemManager`

If a tile implementation uses shared memory, it should access shared memory
through `SmemManager`.  The manager is the user-facing boundary between tile
implementation code and the later megakernel lowering pass.

At the DSL level, `SmemManager` has two responsibilities:

- Allocate logical shared-memory buffers and record their metadata.
- Emit coarse shared-memory phase operations.

It does not expose physical pages, chunks, or concrete mbarrier operations to
users.  The current lowering maps the coarse phase operations onto chunk-level
mbarriers internally.

### `SmemManager.__init__`

```python
smem_manager = SmemManager(smem_max_bytes, chunk_size)
```

Creates a manager for one shared-memory pool.

Parameters:

- `smem_max_bytes`: total shared-memory bytes reserved for this manager.
- `chunk_size`: chunk granularity used by the later lowering implementation.

Users should treat `chunk_size` as a manager configuration, not as a physical
page API.  Tile code should still synchronize through phase markers rather than
addressing chunks directly.

### `SmemManager.alloc`

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
```

Allocates one logical shared-memory buffer from the managed pool.  The returned
buffer can be used directly by parser-style TIRX code in `TileImpl.run()`.  The
manager also records the allocation so the lowering pass can decide the final
physical shared-memory layout.

`policy` describes the intended lifetime and reuse behavior:

- `"shared"`: default.  The buffer participates in coarse shared-memory phases
  marked by `wait_all()` and `release_all()`.
- `"persistent"`: the buffer is live for the whole megakernel and is not
  intended to participate in phase-based reuse.  This is useful for long-lived
  runtime state such as barriers or counters.
- `"exclusive"`: the buffer asks lowering to avoid physical overlap with other
  concurrently live buffers.  This is reserved for expert implementations; the
  first DSL version does not infer fine-grained synchronization correctness.

### `SmemManager.commit`

```python
smem_manager.commit()
```

Finalizes the underlying shared-memory pool size annotation after all managed
allocations have been declared.  In normal DSL usage this should be called by
the lowering flow, not manually inside the tile computation body.

### `SmemManager.wait_all`

```python
smem_manager.wait_all(level="cta")
```

Begins a coarse shared-memory phase.  Managed shared-memory buffers used after
this call and before the matching `release_all()` are treated as live in the
same phase.

The current TIRX lowering implements this by waiting on every managed
shared-memory chunk mbarrier at CTA scope.  Only `level="cta"` is supported for
now; users do not address chunks directly.

Any tile that allocates non-persistent managed shared memory must call
`wait_all()` before using that memory and `release_all()` after it is no longer
needed.  The lowering rejects such tiles if either call is missing.

### `SmemManager.release_all`

```python
smem_manager.release_all(level="cta")
```

Ends the current coarse shared-memory phase.  The current TIRX lowering emits
a chunk-level mbarrier arrive for the managed chunks at CTA scope.  Only
`level="cta"` is supported for now.

### `SmemManager.advance`

```python
smem_manager.advance()
```

Advances the logical shared-memory phase.  In the current TIRX lowering this
flips the manager's local mbarrier phase bit.  Tile implementations should call
this after releasing a phase when they intend later work to acquire the next
phase.

## `TileImpl`

Users subclass `TileImpl` to define the local implementation for one tile kind.
Only `run()` is required.

```python
class MyTile(TileImpl):
    def _declare_resources(self, smem_manager):
        self.smem = smem_manager.alloc(...)

    @T.inline
    def device_init(self, smem_manager, m_idx, n_idx, k_idx):
        self._declare_resources(smem_manager)
        ...

    @T.inline
    def run(self, m_idx, n_idx, k_idx):
        ...
```

### `TileImpl.init_shared_resources`

```python
@classmethod
@T.inline
def init_shared_resources(cls, smem_manager): ...
```

Optional.  Emits parser-style initialization for resources shared by all
instances of this tile class.  Class-level resource declaration or
handle-recording logic can live in ordinary Python helpers called from this
hook.

Example:

```python
@classmethod
def _declare_class_resources(cls, smem_manager):
    cls.accum = smem_manager.alloc(...)

@classmethod
@T.inline
def init_shared_resources(cls, smem_manager):
    cls._declare_class_resources(smem_manager)
    warp_id = T.warp_id([...])
    if warp_id == 0:
        T.ptx.tcgen05.alloc(...)
```

### `TileImpl.finalize_shared_resources`

```python
@classmethod
@T.inline
def finalize_shared_resources(cls, smem_manager): ...
```

Optional.  Emits parser-style finalization for resources initialized by
`init_shared_resources()`.

Example:

```python
@classmethod
@T.inline
def finalize_shared_resources(cls, smem_manager):
    warp_id = T.warp_id([...])
    T.tvm_storage_sync("shared")
    if warp_id == 0:
        T.ptx.tcgen05.relinquish_alloc_permit(...)
        T.ptx.tcgen05.dealloc(...)
```

### `TileImpl.device_init`

```python
@T.inline
def device_init(self, smem_manager, m_idx, n_idx, k_idx): ...
```

Optional.  Emits parser-style device initialization for one tile instance.
Use this hook for TIRX statements that must appear in the final kernel body.
Resource declaration or handle-recording logic can live in ordinary Python
helpers called from this hook; those helpers do not need to be part of the
public `TileImpl` API.

### `TileImpl.host_init`

```python
def host_init(self): ...
```

Optional.  Initializes host-side state for one tile instance.  For example,
this hook can set the cuTensorMap used by that tile instance.

Example:

```python
def host_init(self):
    T.call_packed("runtime.cuTensorMapEncodeTiled", ...)
```

### `TileImpl.prefetch`

```python
@T.inline
def prefetch(self, m_idx, n_idx, k_idx): ...
```

Optional.  Prefetches data for one tile instance before `run()`.  This hook
may run before the tile dependency is satisfied, after the tile has been
dispatched to an SM.  For example, it can prefetch weights that do not depend
on activations while previous tasks are still incomplete.

### `TileImpl.run`

```python
@T.inline
def run(self, m_idx, n_idx, k_idx): ...
```

Required.  Defines the computation for one logical tile instance at index
`(m_idx, n_idx, k_idx)`.

## DSL Example

```python
stage1 = kernel.tile(
    "stage1",
    Stage1Tile(),
    grid=(NUM_BLOCK_M, NUM_BLOCK_N, 1),
    reads=[A.region(lambda m, n, k: R[m, n])],
    writes=[B.region(lambda m, n, k: R[m, n])],
).notify(row_ready, lambda m, n, k: (m,))

stage2 = kernel.tile(
    "stage2",
    Stage2Tile(),
    grid=(NUM_BLOCK_M, 1, 1),
    reads=[B.region(lambda m, n, k: R[m, 0:NUM_BLOCK_N])],
    writes=[C.region(lambda m, n, k: R[m])],
).wait(row_ready, lambda m, n, k: (m,), inverse_coord=lambda m: (m, 0, 0))
```
