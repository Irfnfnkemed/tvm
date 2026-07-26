# Megakernel Partition/Event Plan

This document describes how to plan a megakernel before writing `KernelSpec`
code.  The planning output is intended to be a natural-language design note,
not a rigid schema.  It should make tile spaces, tensor flow, logical events,
and producer-consumer dependencies explicit enough that the DSL spec can be
written and validated.

The plan is compatible with the current DSL and with PR-style MegaMoE workloads.
The key principle is to describe the logical computation and dependency graph
first, while leaving CUDA atomics, spin waits, mbarriers, queue layout, and TIRX
statement bodies to the lowering implementation.

## Scope

A good plan answers these questions:

- What are the user-visible computation stages?
- What tile instances exist for each stage?
- What tensors does each tile read or write, and what regions are known?
- Which producer tiles make which consumer tiles ready?
- Which logical events represent those readiness conditions?
- How does each producer notify an event coordinate?
- How does each consumer wait on an event coordinate?
- For dynamic scheduling, how does one ready event coordinate map back to one
  or more concrete consumer tile coordinates?

The plan should not include TileImpl code, CUDA source snippets, runtime queue
implementation, encoded semaphore formulas, or hand-written event mechanics.
Those belong to `transform.lower`.

## Inputs

The input can be Torch-like staged code, an operator graph, pseudocode, an
existing PR-style megakernel/MegaMoE schedule, or a written staged dataflow
description.

Preserve the staged dataflow unless the user explicitly asks for algebraic
fusion or simplification.  In MegaMoE-style workloads, routing, expert GEMM,
reductions, and communication-adjacent stages may have different runtime
scheduling behavior even when parts of the math could be collapsed.

If a dimension, block size, split factor, expert count, or token bound is not
specified, introduce a symbolic name such as `NUM_BLOCK_M`, `NUM_BLOCK_N`,
`SPLIT_K`, `NUM_EXPERTS`, `MAX_TOKENS`, or `TOKENS_PER_EXPERT`.  Do not invent
concrete constants unless they are given by the workload.

## Planning Narrative

The recommended planning note is written as short sections.

First describe the stages.  Each stage should correspond to a tile kind unless
there is a clear reason to combine stages.  For each stage, state its tile grid
as `(m, n, k)`, using `1` for unused axes.  Also explain what local computation
one tile performs.

Then describe tensors.  Classify each tensor as input, intermediate, output,
routing, or workspace.  Record which stage produces it, which stages consume it,
and the shape or symbolic shape if known.  When a tile accesses a known region,
write the region in terms of its `(m, n, k)` coordinate.  When the region depends
on runtime routing or cannot be expressed statically, say that the region is
unknown/dynamic.

Next describe logical events.  Each event is a count tensor.  State the event
shape, what one event coordinate means, and the `init_count` for each coordinate.
`init_count` may be a constant, a symbolic expression, or a per-coordinate
function.  It should equal the number of producer notifications required before
that event coordinate is ready.

Finally describe dependencies.  For every producer-consumer relationship that
requires ordering, state the event used, the producer notify mapping, and the
consumer wait mapping.  In DSL terms, both mappings become `D(event, coord)`
where `coord(m, n, k, i)` returns `(coord_count, remote_rank, *event_coord)`.
For dynamic scheduling, also state the inverse mapping used to push consumer
tasks: `inv_coord(remote_rank, *event_coord, consumer_i)` returns
`(consumer_count, tile_m, tile_n, tile_k)`.

## Mapping To DSL

The natural-language plan maps to DSL objects directly:

```text
symbolic dimensions      -> kernel.var(...)
tensor declarations      -> kernel.tensor(...)
logical events           -> kernel.event(..., init_count=...)
tile stages              -> kernel.tile(..., grid=(m, n, k), reads=..., writes=...)
known tensor regions     -> tensor.region(lambda m, n, k: R[...])
unknown tensor regions   -> bare tensor in reads/writes
dependencies             -> tile.wait(D(...)) and tile.notify(D(...))
dynamic reverse mapping  -> inv_coord=... on the waiting dependency
```

The dependency coordinate function has one form:

```python
coord = lambda m, n, k, i: (coord_count, remote_rank, *event_coord)
```

For waits, the coordinate must describe one local event coordinate:

```text
coord_count == 1
remote_rank == -1
```

For notifies, `coord_count` tells lowering how many event coordinates this
one tile notify expands to.  If `coord_count > 1`, validation checks every
`notify_i` when the mapping is statically provable.  `coord_count` must be
stable, event coordinates must be inside the event shape, and one tile notify
must not generate duplicate `(remote_rank, event_coord)` entries.

For dynamic scheduling, every wait that can be triggered by a producer must
provide `inv_coord`.  When statically provable, validation checks that each
consumer coordinate generated by `inv_coord` is inside the consumer grid,
round-trips through the wait `coord`, and is unique for that event coordinate.
If dependency routing uses runtime tensor indexing, these static proofs are
skipped with warnings and must be covered by workload tests.

## Common Patterns

### One-To-One

A producer tile `(m, n, k)` enables the matching consumer tile `(m, n, k)`.
Use an event shaped like the tile grid.  The producer notifies event coordinate
`(m, n, k)`, and the consumer waits on the same coordinate.  The event
`init_count` is usually `1`.

DSL shape:

```python
ready = kernel.event("ready", (M, N, K), init_count=1)
producer.notify(D(ready, lambda m, n, k, i: (1, -1, m, n, k)))
consumer.wait(D(
    ready,
    lambda m, n, k, i: (1, -1, m, n, k),
    inv_coord=lambda remote_rank, m, n, k, ci: (1, m, n, k),
))
```

### Many-To-One Reduction

A tile space `(m, n, 1)` produces partial results, and one consumer tile
`(m, 0, 0)` can run after all `n` producers for the same `m` are ready.  Use an
event shaped `(M,)`.  Each producer notifies `row_ready[m]`; the event
`init_count` is `N` or the symbolic variable representing the number of `n`
tiles.

DSL shape:

```python
row_ready = kernel.event("row_ready", (M,), init_count=N)
partial.notify(D(row_ready, lambda m, n, k, i: (1, -1, m)))
final.wait(D(
    row_ready,
    lambda m, n, k, i: (1, -1, m),
    inv_coord=lambda remote_rank, m, ci: (1, m, 0, 0),
))
```

### One-To-Many Or Runtime Routed

One ready event coordinate may enable multiple consumer tiles.  This is common
for MegaMoE-style routing, where a runtime routing tensor maps experts or token
blocks to consumer work.

The plan should state what one event coordinate represents, how many consumers
it may fan out to, and how `consumer_i` selects one concrete consumer tile.
The DSL may use runtime tensor indexing inside dependency coordinate closures;
validation will warn when it cannot statically prove dimension or round-trip
properties.

DSL shape:

```python
expert.wait(D(
    expert_ready,
    lambda m, n, k, i: (1, -1, expert_id_for_tile(m, n)),
    inv_coord=lambda remote_rank, expert_id, ci: (
        tokens_per_expert(expert_id),
        token_for_expert(expert_id, ci),
        expert_id,
        0,
    ),
))
```

## Dynamic Scheduler Notes

For dynamic scheduling, entry tiles are inferred from the dependency graph: they
are tiles with no waits.  A producer notify on event `E` can push any consumer
tile that waits on `E` and provides `inv_coord`.  The endpoint tile is inferred
as a tile with no notifies; the current lowering expects one endpoint tile, and
its grid should be one tile when statically known.

The current dynamic queue implementation has one important limitation: it does
not yet have a robust empty-queue protocol.  The initial entry-task count must
be large enough relative to `sm_count`, otherwise workers may dequeue empty
slots and terminate early.  This is a lowering/runtime limitation, not a DSL
semantic requirement.

## Validation Checklist

Before writing or lowering the DSL, the plan should make these checks clear:

- every tile grid is three-dimensional `(m, n, k)`;
- every tensor read has a producer or is an input/routing tensor;
- known tensor regions have the same number of dimensions as the tensor;
- event coordinate length matches the event dimensions;
- `init_count` matches the number of producer notifies per event coordinate;
- dynamic waits that should be producer-triggered have `inv_coord`;
- one-to-many `inv_coord` includes a stable fan-out count;
- statically checkable `inv_coord` mappings round-trip through the wait mapping;
- runtime-routed mappings are called out as requiring workload coverage;
- the staged dataflow from the original computation is preserved.
