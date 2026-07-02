# Megakernel DSL User API

This document lists the user-facing API in `tvm.megakernel.dsl`.

## DSL Layers

The DSL has two layers:

- Spec layer: `KernelSpec`, `TensorSpec`, `EventSpec`, `DependencySpec`, and
  `TileSpec`.  This layer describes tile stages, tensor inputs/outputs, logical
  events, and wait/notify dependencies.
- Impl layer: `TileImpl`.  This layer connects a logical tile to the concrete
  implementation of that tile.

The spec layer corresponds to Step 3 in [workflow.md](workflow.md).  The impl
layer corresponds to Step 4.

## `KernelSpec`

```python
kernel = KernelSpec(name: str, attrs: dict[str, Any] | None = None)
```

Creates one megakernel spec.  Tensors, events, and tiles are registered on this
object.

Parameters:

- `name`: unique name for the megakernel spec.
- `attrs`: optional metadata reserved for later passes.


Example:

```python
kernel = KernelSpec("two_stage_reduce", attrs={"target": "sm90"})
```

## `KernelSpec.tensor`

```python
tensor = kernel.tensor(name: str, shape: ShapeType, dtype: str)
```

Registers a logical tensor.

Parameters:

- `name`: tensor name, unique inside the kernel.
- `shape`: tensor shape.  Each dimension can be an `int` or `VarSpec`.
- `dtype`: tensor element type.

Returns: `TensorSpec`.

Example:

```python
bs = VarSpec("bs")
A = kernel.tensor("A", shape=(bs, 1024), dtype="float32")
```

## `KernelSpec.input`

```python
tensor = kernel.input(name: str, shape: ShapeType, dtype: str)
```

Registers an input tensor.  This is currently an alias of `tensor()`, provided
for readability at call sites.

Example:

```python
A = kernel.input("A", shape=(1024, 1024), dtype="float32")
```

## `KernelSpec.intermediate`

```python
tensor = kernel.intermediate(name: str, shape: ShapeType, dtype: str)
```

Registers an intermediate tensor.  This is currently an alias of `tensor()`.

Example:

```python
B = kernel.intermediate("B", shape=(1024, 16), dtype="float32")
```

## `KernelSpec.output`

```python
tensor = kernel.output(name: str, shape: ShapeType, dtype: str)
```

Registers an output tensor.  This is currently an alias of `tensor()`.

Example:

```python
C = kernel.output("C", shape=(1024, 1), dtype="float32")
```

## `KernelSpec.event`

```python
event = kernel.event(
    name: str,
    shape: ShapeType,
    init_count: int | Callable[[tuple[int, ...]], int],
    dtype: str = "int32",
    attrs: dict[str, Any] | None = None,
)
```

Registers a logical event tensor.

Parameters:

- `name`: event name, unique inside the kernel.
- `shape`: event tensor shape.
- `init_count`: logical count for each event coordinate.  This can be a single
  integer for a uniform count, or a callable that returns the count for a given
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
    init_count=lambda coord: coord[0] + coord[1],
)
```

## `KernelSpec.tile`

```python
tile = kernel.tile(
    name: str,
    impl: TileImpl,
    tile_num: TileNumType,
    attrs: dict[str, Any] | None = None,
)
```

Registers one logical tile stage.

Parameters:

- `name`: tile stage name, unique inside the kernel.
- `impl`: local tile implementation object.
- `tile_num`: tile count on `(m, n, k)` axes.  Use `1` for unused axes.
- `attrs`: optional metadata reserved for later passes.

Returns: `TileSpec`.

Example:

```python
bs = VarSpec("bs")
tile_a = kernel.tile(
    "tile_a",
    tile_a_impl,
    tile_num=(bs, 16, 1),
)
```

## `TileSpec.read`

```python
tile.read(*tensors: TensorSpec)
```

Declares tensors read by this tile.  Returns the same `TileSpec`, so calls can
be chained.

Example:

```python
tile_a.read(A)
```

## `TileSpec.write`

```python
tile.write(*tensors: TensorSpec)
```

Declares tensors written by this tile.  Returns the same `TileSpec`.

Example:

```python
tile_a.write(B)
```

## `TileSpec.wait`

```python
tile.wait(event: EventSpec, coord_map: CoordMapType)
```

Declares that this tile waits on `event` at the coordinate produced by
`coord_map`.

Parameters:

- `event`: event to wait on.
- `coord_map`: callable mapping tile index `(m, n, k)` to an event
  coordinate.  If it is a tuple/list instead of a callable, it is used directly
  as the event coordinate.

Returns: `TileSpec`.

Example:

```python
tile_b.wait(
    event=evt1,
    coord_map=lambda m, n, k: (m,),
)
```

## `TileSpec.notify`

```python
tile.notify(event: EventSpec, coord_map: CoordMapType)
```

Declares that this tile notifies `event` at the coordinate produced by
`coord_map`.

Parameters:

- `event`: event to notify.
- `coord_map`: callable mapping tile index `(m, n, k)` to an event
  coordinate.  If it is a tuple/list instead of a callable, it is used directly
  as the event coordinate.

Returns: `TileSpec`.

Example:

```python
tile_a.notify(
    event=evt1,
    coord_map=lambda m, n, k: (m,),
)
```

## `TileImpl`

Users subclass `TileImpl` to define the local implementation for one tile kind.
Only `run()` is required.

```python
class MyTile(TileImpl):
    def run(self, m_idx, n_idx, k_idx):
        ...
```

### `TileImpl.init_shared_resources`

```python
@classmethod
def init_shared_resources(cls): ...
```

Optional.  Initializes resources shared by all instances of this tile class.
For example, this hook can allocate shared memory, mbarriers, or tensor memory
used by all tile instances of the same class.

Example:

```python
@classmethod
def init_shared_resources(cls):
    warp_id = T.warp_id([...])
    if warp_id == 0:
        T.ptx.tcgen05.alloc(...)
```

### `TileImpl.finalize_shared_resources`

```python
@classmethod
def finalize_shared_resources(cls): ...
```

Optional.  Releases resources created by `init_shared_resources()`.  For
example, this hook can release tensor memory shared by all tile instances of
the same class.

Example:

```python
@classmethod
def finalize_shared_resources(cls):
    warp_id = T.warp_id([...])
    T.tvm_storage_sync("shared")
    if warp_id == 0:
        T.ptx.tcgen05.relinquish_alloc_permit(...)
        T.ptx.tcgen05.dealloc(...)
```


### `TileImpl.device_init`

```python
def device_init(self): ...
```

Optional.  Initializes device-side state owned by one tile instance.  For
example, this hook can allocate buffers used by that tile instance.

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
def prefetch(self, m_idx, n_idx, k_idx): ...
```

Optional.  Prefetches data for one tile instance before `run()`.  This hook
may run before the tile dependency is satisfied, after the tile has been
dispatched to an SM.  For example, it can prefetch weights that do not depend
on activations while previous tasks are still incomplete.

### `TileImpl.run`

```python
def run(self, m_idx, n_idx, k_idx): ...
```

Required.  Defines the computation for one logical tile instance at index
`(m_idx, n_idx, k_idx)`.

## DSL Example

```python
stage1 = (
    kernel.tile("stage1", Stage1Tile(), (NUM_BLOCK_M, NUM_BLOCK_N, 1))
    .read(A)
    .write(B)
    .notify(row_ready, lambda m, n, k: (m,))
)

stage2 = (
    kernel.tile("stage2", Stage2Tile(), (NUM_BLOCK_M, 1, 1))
    .read(B)
    .write(C)
    .wait(row_ready, lambda m, n, k: (m,))
)
```
