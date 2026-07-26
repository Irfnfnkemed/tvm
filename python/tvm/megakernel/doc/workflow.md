# Megakernel DSL Workflow

The megakernel DSL is a bridge between a planned staged computation and a TIRX
megakernel implementation.  It is not a parser and it is not the runtime event
implementation itself.  It records the logical graph that lowering can validate
and emit.

## Current Architecture

```text
input program / operator graph / PR-style workload description
  -> Stage 1 tile and event plan
  -> DSL spec layer: KernelSpec, tensors, events, tiles, R, D
  -> DSL impl layer: TileImpl and SmemManager
  -> validation: semantic, impl, lower-plan
  -> prepare: KernelSpec -> KernelLoweringPlan
  -> lower: TIRX module with queue init and persistent megakernel
```

The public user API is intentionally small:

```python
from tvm.megakernel.dsl import D, KernelSpec, R, TileImpl, SmemManager
from tvm.megakernel.transform import LoweringOptions, lower
```

Everything under `dsl.spec`, `dsl.impl`, `transform.validate`, and
`transform.lower` is an implementation/detail package unless a specific symbol
is re-exported by the top-level modules.

## Step 1: Start From Staged Computation

Input may be Torch-like code, an operator graph, pseudocode, an existing PR-style
MegaMoE schedule, or a written staged dataflow description.

Preserve user-visible stages.  Do not collapse stages into a mathematically
equivalent operation unless the user explicitly asks for that transformation.
MegaMoE-style pipelines are especially sensitive to this rule because routing,
expert GEMM, reductions, and communication-adjacent stages may have different
runtime scheduling behavior.

## Step 2: Plan Tiles, Tensors, And Events

Before writing Python DSL, identify:

1. tile stages;
2. tile grid for each stage, always as `(m, n, k)`;
3. tensors read and written by each tile;
4. logical readiness events between producer and consumer stages;
5. event dimensions and `init_count`;
6. wait/notify coordinate mappings;
7. dynamic reverse mappings for any event that should push consumer tasks.

The plan is logical.  It should not contain atomics, spin loops, mbarrier layout,
CUDA source snippets, or TIRX statement bodies.  Those belong to lowering.

Use [plan.md](plan.md) as the natural-language planning guide.

## Step 3: Write The Spec Layer

The spec layer records the plan:

```python
kernel = KernelSpec("my_kernel")
M = kernel.var("M", bounds=(1, 4096))
A = kernel.tensor("A", (M, 4096), "float16")
ready = kernel.event("ready", (M,), init_count=lambda m: 1)

tile = kernel.tile(
    "stage",
    StageTile(),
    grid=(M, 1, 1),
    reads=[A.region(lambda m, n, k: R[m, 0:4096])],
)
```

Bare tensor access, such as `reads=[A]`, means the region is unknown/dynamic.
This is valid but disables region-specific semantic checks for that access.

Known access, such as `reads=[A.region(lambda m, n, k: R[m, 0:4096])]`, enables
region shape, bounds, and producer-consumer overlap validation.

## Step 4: Add Dependencies With `D`

Both waits and notifies use the same dependency builder:

```python
tile.notify(D(event, lambda m, n, k, i: (coord_count, remote_rank, *event_coord)))
tile.wait(D(event, wait_coord, inv_coord=consumer_inverse))
```

The forward mapping is producer/consumer local:

```text
coord(tile_m, tile_n, tile_k, notify_i)
  -> (coord_count, remote_rank, *event_coord)
```

The reverse mapping is dynamic-scheduler only:

```text
inv_coord(remote_rank, *event_coord, consumer_i)
  -> (consumer_count, tile_m, tile_n, tile_k)
```

For statically checkable multi-coordinate notifies, validation checks every `notify_i`:
`coord_count` must be stable, event coordinates must be inside the event shape,
and one tile notify must not generate duplicate `(remote_rank, event_coord)`
entries.

A static schedule only needs enough information to wait and notify logical
events.  A dynamic schedule also needs `inv_coord` on waits so the producer side
can push consumer tasks after an event coordinate becomes ready.  For static
coord mappings, validation checks every fan-out consumer returned by `inv_coord`:
it must round-trip through the wait `coord`, be inside the consumer grid, and
not duplicate another consumer tile for the same event coord.

## Step 5: Implement Tiles

`TileImpl` contains local parser-style TIRX code for one tile instance.  It may
use `SmemManager` for managed shared-memory allocation and phase markers.
TileImpl constructors may capture spec-layer tensors and symbolic expressions;
validation and lowering replace those captures with sample values, upper-bound
capacities, or runtime TIR values depending on the phase.

Dependency policy should stay out of `TileImpl`.  A tile implementation should
not hand-write global waits/notifies for DSL events; those come from
`wait(D(...))` and `notify(D(...))`.

## Step 6: Validate

Validation is split by object boundary:

1. `validate/semantic.py` checks the DSL graph: ownership, dependency shape,
   event producer/waiter consistency, event counts, tensor regions, and
   producer-consumer region coverage.
2. `validate/impl.py` emits each `TileImpl` enough to collect buffer accesses
   and checks actual accesses against declared `reads` and `writes`.
3. `validate/lower.py` checks the prepared lowering plan: reserved job ids,
   event workspace layout, tile-plan coverage, static queue capacity, dynamic
   entry/endpoint structure, and dynamic `inv_coord` round-trip when statically
   checkable.

Runtime tensor indexing inside dependency coordinate functions is allowed for
MegaMoE-style routing, but some static checks cannot prove its dimensions or
round-trip behavior.  Those checks are skipped with warnings and must be covered
by workload tests.  Dependency coordinate functions should capture routing
`TensorSpec` objects through closures; default-argument or global captures are
rejected because lowering cannot safely rewrite them.

## Step 7: Lower

```python
mod = lower(kernel, LoweringOptions(schedule="static"))
mod = lower(kernel, LoweringOptions(schedule="dynamic", attrs={"num_threads": 256}))
```

Lowering prepares a `KernelLoweringPlan` and emits:

- the persistent megakernel body;
- a queue initialization function for static or dynamic schedules;
- event workspace initialization when events exist;
- scheduler, event, and shared-memory runtime helper code.

## Static Schedule Model

The static scheduler receives a precomputed per-SM queue.  Each CTA walks its
queue linearly.  For every tile task, lowering emits:

```text
device_init
prefetch
waits
run
notifies
next_tile
```

Static events use one notify phase.  A notify subtracts `base + 1` from the
event counter.  A wait spins until the counter reaches zero.

## Dynamic Schedule Model

The dynamic scheduler follows the PR/old megakernel execution model:

1. A queue-init function initializes event counters and pushes entry tasks.
2. The persistent kernel starts CTAs that repeatedly dequeue tasks from a global
   MPMC queue.
3. A scheduler warp dequeues one packed task and broadcasts it to the CTA using
   mbarriers.
4. For a tile task, lowering emits `device_init` and `prefetch` first.
5. Producer notifies are split into pre-notify and complete-notify.
6. Pre-notify subtracts `1` from the event counter.  If this pre-notify observes
   that the event coordinate is ready to schedule consumers, the scheduler uses
   the waiting dependency's `inv_coord` to push consumer tasks.
7. The tile waits on its own dependencies, runs, and then complete-notifies by
   subtracting `base`.
8. The endpoint tile pushes end tasks so every CTA can terminate.

This model allows consumers to be scheduled as soon as their producers have
been dispatched far enough to avoid deadlock, while waits still protect the
actual data dependency before `run()`.  The current dynamic queue still assumes
that initial entry tasks are sufficient to keep workers from dequeuing empty
slots; a robust empty-queue protocol is future work.
