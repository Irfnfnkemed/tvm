# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Semantic validation for megakernel DSL graphs.

Validation performed by this module, in call order:

1. Kernel declaration ownership:
   - Tensor shapes, event shapes, and tile grids may only reference Vars owned
     by the same KernelSpec.

2. Tile access ownership:
   - Every tensor listed in tile.reads and tile.writes must belong to the same
     KernelSpec, after resolving region views to their base tensor.

3. Event dependency ownership and shape:
   - Every waited/notified event must belong to the same KernelSpec.
   - A tile may have at most one wait edge and one notify edge per logical event.
   - Wait/notify coord mappings must return tuple/list values with the same
     number of dimensions as the target event shape.
   - coord_count and rank may only use integers, Vars, and ExprSpecs owned by
     the same KernelSpec.
   - Event coord expressions may also be lower-time/runtime values.  Runtime
     event coords keep shape checks but skip semantic checks that require exact
     static event coordinates.

4. Event producer/waiter consistency:
   - Any waited event must have at least one notifying tile.
   - Any notified event must have at least one waiting tile.

5. Static event count consistency:
   - For each sampled/static event coordinate, the number of notifying tiles
     must match event.init_count(coord).
   - Every waited coordinate must have at least one corresponding notify.
   - Symbolic shapes/grids are checked using deterministic samples.

6. Declared tensor region shape and ownership:
   - Region access dimensions must match tensor dimensions.
   - Region start/extent expressions must be valid expr-like values.
   - Region expressions may only reference Vars owned by the same KernelSpec.

7. Tensor dependency event presence:
   - If one tile writes a tensor and another tile reads the same tensor, there
     must be at least one shared event between the writer's notifies and the
     reader's waits. This is conservative and does not consider disjoint regions.

8. Tensor region dependency coordinates:
   - For sampled/static tile coordinates, overlapping write/read regions of the
     same tensor must be connected by a matching notify/wait event coordinate.
   - Static regions are also checked for tensor bounds.

9. Waited-region source consistency:
   - For each sampled/static read behind a wait, a tile notifying the waited
     event coordinate must write an overlapping region of the same tensor.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import reduce
from types import FunctionType
from itertools import product
from operator import mul
import random
from typing import Any

from ...dsl.spec import (
    EventSpec,
    ExprSpec,
    KernelSpec,
    RegionRange,
    RegionSpec,
    TensorSpec,
    TileSpec,
    VarSpec,
    eval_expr_like,
    expr_vars,
)


@dataclass(frozen=True)
class LogicalEdge:
    """One logical event dependency from a producer tile to a consumer tile."""

    event: EventSpec
    producer: TileSpec
    consumer: TileSpec


def logical_edges(kernel: KernelSpec) -> tuple[LogicalEdge, ...]:
    """Return logical event edges in stable event/tile order."""

    event_notifiers: dict[int, list[TileSpec]] = {
        id(event): [] for event in kernel.events.values()
    }
    event_waiters: dict[int, list[TileSpec]] = {id(event): [] for event in kernel.events.values()}
    for tile in kernel.tiles:
        for dependency in tile.notifies:
            event = dependency.event
            if tile not in event_notifiers.setdefault(id(event), []):
                event_notifiers[id(event)].append(tile)
        for dependency in tile.waits:
            event = dependency.event
            if tile not in event_waiters.setdefault(id(event), []):
                event_waiters[id(event)].append(tile)

    edges: list[LogicalEdge] = []
    for event in kernel.events.values():
        for producer in event_notifiers.get(id(event), []):
            for consumer in event_waiters.get(id(event), []):
                edges.append(LogicalEdge(event, producer, consumer))
    return tuple(edges)


def event_init_count(
    event: EventSpec, coord: tuple[int, ...], env: dict[VarSpec, int] | None = None
) -> int:
    """Evaluate one logical event init count under an optional symbolic-var sample."""

    count = event.init_count(*coord)
    resolved = eval_expr_like(count, env)
    if resolved is None:
        resolved = count
    if isinstance(resolved, bool) or not isinstance(resolved, int):
        raise TypeError("event init_count must produce an integer")
    if resolved < 0:
        raise ValueError("event init_count must produce a non-negative integer")
    return resolved


def validate_kernel(kernel: KernelSpec) -> KernelSpec:
    """Validate DSL-level dependencies before lowering."""

    tensor_ids = {id(tensor) for tensor in kernel.tensors.values()}
    event_ids = {id(event) for event in kernel.events.values()}
    var_ids = {id(var) for var in kernel.vars.values()}
    _validate_kernel_expr_ownership(kernel, var_ids)
    tile_names = [tile.name for tile in kernel.tiles]
    if len(tile_names) != len(set(tile_names)):
        raise ValueError("kernel contains duplicate tile names")

    event_notifiers: dict[int, list[TileSpec]] = defaultdict(list)
    event_waiters: dict[int, list[TileSpec]] = defaultdict(list)
    for tile in kernel.tiles:
        for tensor in tile.reads:
            _validate_tile_tensor_access(tile, tensor, tensor_ids)
        for tensor in tile.writes:
            _validate_tile_tensor_access(tile, tensor, tensor_ids)
        # A tile has at most one notify edge per logical event.
        notified_event_ids: set[int] = set()
        for dependency in tile.notifies:
            event = dependency.event
            coord = dependency.coord
            if id(event) not in event_ids:
                raise ValueError(f"tile {tile.name!r} notifies event outside kernel")
            if id(event) in notified_event_ids:
                raise ValueError(
                    f"tile {tile.name!r} notifies event {event.name!r} more than once"
                )
            notified_event_ids.add(id(event))
            _validate_dependency_shape(
                event, dependency, tile.grid, f"{tile.name}.notify", var_ids, tensor_ids, is_wait=False
            )
            if tile not in event_notifiers[id(event)]:
                event_notifiers[id(event)].append(tile)
        # A tile has at most one wait edge per logical event.
        waited_event_ids: set[int] = set()
        for dependency in tile.waits:
            event = dependency.event
            coord = dependency.coord
            if id(event) not in event_ids:
                raise ValueError(f"tile {tile.name!r} waits on event outside kernel")
            if id(event) in waited_event_ids:
                raise ValueError(
                    f"tile {tile.name!r} waits on event {event.name!r} more than once"
                )
            waited_event_ids.add(id(event))
            _validate_dependency_shape(
                event, dependency, tile.grid, f"{tile.name}.wait", var_ids, tensor_ids, is_wait=True
            )
            if tile not in event_waiters[id(event)]:
                event_waiters[id(event)].append(tile)

    for event in kernel.events.values():
        if event_waiters[id(event)] and not event_notifiers[id(event)]:
            raise ValueError(f"event {event.name!r} is waited on but has no producer")
        if event_notifiers[id(event)] and not event_waiters[id(event)]:
            raise ValueError(f"event {event.name!r} is notified but has no consumer")
        _validate_static_event_counts(
            event, event_notifiers[id(event)], event_waiters[id(event)], tensor_ids
        )

    _validate_region_dependencies(kernel, var_ids)

    return kernel


def _validate_kernel_expr_ownership(kernel: KernelSpec, var_ids: set[int]) -> None:
    for tensor in kernel.tensors.values():
        _validate_expr_ownership(tensor.shape, var_ids, f"tensor {tensor.name!r} shape")
    for event in kernel.events.values():
        _validate_expr_ownership(event.shape, var_ids, f"event {event.name!r} shape")
    for tile in kernel.tiles:
        _validate_expr_ownership(tile.grid, var_ids, f"tile {tile.name!r} grid")


def _validate_expr_ownership(value: Any, var_ids: set[int], label: str) -> None:
    for var in expr_vars(value):
        if id(var) not in var_ids:
            raise ValueError(f"{label} references a VarSpec outside this kernel")


def _validate_tile_tensor_access(tile, access: TensorSpec, tensor_ids: set[int]) -> None:
    tensor = access.base_tensor
    if id(tensor) not in tensor_ids:
        raise ValueError(f"tile {tile.name!r} references tensor outside kernel")


def _validate_region_dependencies(kernel: KernelSpec, var_ids: set[int]) -> None:
    tensor_ids = {id(tensor) for tensor in kernel.tensors.values()}
    for tile in kernel.tiles:
        for access in tile.reads:
            if access.region_from_tile is not None:
                _validate_tensor_region_access(tile, access, tensor_ids, is_write=False, var_ids=var_ids)
        for access in tile.writes:
            if access.region_from_tile is not None:
                _validate_tensor_region_access(tile, access, tensor_ids, is_write=True, var_ids=var_ids)

    _validate_tensor_dependency_events(kernel)
    _validate_region_dependency_coords(kernel)
    _validate_waited_region_sources(kernel)


def _validate_tensor_dependency_events(kernel: KernelSpec) -> None:
    for producer in kernel.tiles:
        write_tensors = {_base_tensor(access) for access in producer.writes}
        if not write_tensors:
            continue
        notify_events = {id(dep.event) for dep in producer.notifies}
        for consumer in kernel.tiles:
            if consumer is producer:
                continue
            shared_tensors = write_tensors & {_base_tensor(access) for access in consumer.reads}
            if not shared_tensors:
                continue
            wait_events = {id(dep.event) for dep in consumer.waits}
            if notify_events & wait_events:
                continue
            tensor_names = sorted(tensor.name for tensor in shared_tensors)
            raise ValueError(
                f"tile {consumer.name!r} reads tensor(s) {tensor_names} written by "
                f"tile {producer.name!r} without an event dependency"
            )


def _validate_region_dependency_coords(kernel: KernelSpec) -> None:
    envs = _sample_kernel_envs(kernel)
    tensor_ids = {id(tensor) for tensor in kernel.tensors.values()}
    for env in envs:
        for producer in kernel.tiles:
            producer_extents = _static_int_tuple(producer.grid, env)
            if producer_extents is None:
                continue
            producer_writes = list(producer.writes)
            if not producer_writes:
                continue
            for consumer in kernel.tiles:
                if consumer is producer:
                    continue
                consumer_extents = _static_int_tuple(consumer.grid, env)
                if consumer_extents is None:
                    continue
                consumer_reads = list(consumer.reads)
                if not consumer_reads:
                    continue
                for producer_idx in product(*(range(extent) for extent in producer_extents)):
                    for write_access in producer_writes:
                        tensor = write_access.base_tensor
                        tensor_shape = _static_int_tuple(tensor.shape, env)
                        if tensor_shape is None:
                            continue
                        write_region = _region_from_access(write_access, env, *producer_idx)
                        _validate_region_bounds(
                            tensor, write_region, tensor_shape, f"{producer.name}.write_region"
                        )
                        for consumer_idx in product(*(range(extent) for extent in consumer_extents)):
                            for read_access in consumer_reads:
                                if read_access.base_tensor is not tensor:
                                    continue
                                read_region = _region_from_access(read_access, env, *consumer_idx)
                                _validate_region_bounds(
                                    tensor, read_region, tensor_shape, f"{consumer.name}.read_region"
                                )
                                if not regions_overlap(write_region, read_region):
                                    continue
                                if _has_matching_event_coord(
                                    producer, producer_idx, consumer, consumer_idx, env, tensor_ids
                                ):
                                    continue
                                raise ValueError(
                                    f"tile {consumer.name!r} idx {consumer_idx} reads tensor "
                                    f"{tensor.name!r} region {region_label(read_region)} that overlaps "
                                    f"tile {producer.name!r} idx {producer_idx} write region "
                                    f"{region_label(write_region)} without an event dependency"
                                )


def _validate_waited_region_sources(kernel: KernelSpec) -> None:
    envs = _sample_kernel_envs(kernel)
    tensor_ids = {id(tensor) for tensor in kernel.tensors.values()}
    for env in envs:
        for consumer in kernel.tiles:
            consumer_extents = _static_int_tuple(consumer.grid, env)
            if consumer_extents is None:
                continue
            consumer_reads = list(consumer.reads)
            if not consumer_reads or not consumer.waits:
                continue
            for consumer_idx in product(*(range(extent) for extent in consumer_extents)):
                for read_access in consumer_reads:
                    tensor = read_access.base_tensor
                    tensor_shape = _static_int_tuple(tensor.shape, env)
                    if tensor_shape is None:
                        continue
                    read_region = _region_from_access(read_access, env, *consumer_idx)
                    _validate_region_bounds(
                        tensor, read_region, tensor_shape, f"{consumer.name}.read_region"
                    )
                    for dependency in consumer.waits:
                        wait_event = dependency.event
                        wait_coord = _static_dependency_coord(
                            dependency, *consumer_idx, env=env, tensor_ids=tensor_ids
                        )
                        if wait_coord is None:
                            continue
                        _validate_waited_region_source(
                            kernel, consumer, consumer_idx, tensor, read_region,
                            wait_event, wait_coord, env, tensor_ids
                        )


def _validate_waited_region_source(
    kernel: KernelSpec,
    consumer,
    consumer_idx,
    tensor: TensorSpec,
    read_region: RegionSpec | None,
    wait_event: EventSpec,
    wait_coord: tuple[Any, ...],
    env,
    tensor_ids: set[int],
) -> None:
    has_same_tensor_event_writer = False
    for producer in kernel.tiles:
        if producer is consumer:
            continue
        producer_writes = [access for access in producer.writes if access.base_tensor is tensor]
        if not producer_writes:
            continue
        if not any(dep.event is wait_event for dep in producer.notifies):
            continue
        has_same_tensor_event_writer = True
        producer_extents = _static_int_tuple(producer.grid, env)
        if producer_extents is None:
            continue
        for producer_idx in product(*(range(extent) for extent in producer_extents)):
            if not _producer_notifies_coord(
                producer, producer_idx, wait_event, wait_coord, env, tensor_ids
            ):
                continue
            for write_access in producer_writes:
                write_region = _region_from_access(write_access, env, *producer_idx)
                _validate_region_bounds(
                    tensor, write_region, _static_int_tuple(tensor.shape, env),
                    f"{producer.name}.write_region"
                )
                if regions_overlap(write_region, read_region):
                    return
    if has_same_tensor_event_writer:
        raise ValueError(
            f"tile {consumer.name!r} idx {consumer_idx} waits on event {wait_event.name!r} "
            f"coord {wait_coord} and reads tensor {tensor.name!r} region "
            f"{region_label(read_region)}, but no producer notifying that coord writes an "
            "overlapping region"
        )


def _producer_notifies_coord(
    producer, producer_idx, wait_event, wait_coord, env, tensor_ids: set[int]
) -> bool:
    for dependency in producer.notifies:
        notify_event = dependency.event
        notify_map = dependency.coord
        if notify_event is not wait_event:
            continue
        notify_coord = _static_dependency_coord(
            dependency, *producer_idx, env=env, tensor_ids=tensor_ids
        )
        if notify_coord is None:
            return True
        if notify_coord == wait_coord:
            return True
    return False


def _base_tensor(access: TensorSpec) -> TensorSpec:
    return access.base_tensor


def _has_matching_event_coord(
    producer, producer_idx, consumer, consumer_idx, env, tensor_ids: set[int]
) -> bool:
    for notify_dep in producer.notifies:
        notify_event = notify_dep.event
        notify_coord = _static_dependency_coord(
            notify_dep, *producer_idx, env=env, tensor_ids=tensor_ids
        )
        for wait_dep in consumer.waits:
            wait_event = wait_dep.event
            if wait_event is not notify_event:
                continue
            wait_coord = _static_dependency_coord(
                wait_dep, *consumer_idx, env=env, tensor_ids=tensor_ids
            )
            if notify_coord is None or wait_coord is None:
                return True
            if wait_coord == notify_coord:
                return True
    return False


def regions_overlap(lhs: RegionSpec | None, rhs: RegionSpec | None) -> bool:
    if lhs is None or rhs is None:
        return True
    if len(lhs.dims) != len(rhs.dims):
        return False
    for lhs_dim, rhs_dim in zip(lhs.dims, rhs.dims):
        if not all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in (lhs_dim.start, lhs_dim.extent, rhs_dim.start, rhs_dim.extent)
        ):
            return False
        lhs_end = lhs_dim.start + lhs_dim.extent
        rhs_end = rhs_dim.start + rhs_dim.extent
        if lhs_dim.start >= rhs_end or rhs_dim.start >= lhs_end:
            return False
    return True


def _validate_tensor_region_access(
    tile, access: TensorSpec, tensor_ids: set[int], is_write: bool, var_ids: set[int]
) -> None:
    kind = "write" if is_write else "read"
    tensor = access.base_tensor
    if id(tensor) not in tensor_ids:
        raise ValueError(f"tile {tile.name!r} has {kind} region for tensor outside kernel")
    if access.region_from_tile is None:
        return
    _validate_region_from_tile_shape(
        tensor, access.region_from_tile, tile.grid, f"{tile.name}.{kind}_region", var_ids
    )


def _region_from_access(access: TensorSpec, env, m_idx, n_idx, k_idx) -> RegionSpec | None:
    if access.region_from_tile is None:
        return None
    return _resolve_region(_region_from_tile(access.region_from_tile, m_idx, n_idx, k_idx), env)


def _region_from_tile(region_from_tile, m_idx, n_idx, k_idx) -> RegionSpec:
    region = region_from_tile(m_idx, n_idx, k_idx) if callable(region_from_tile) else region_from_tile
    if isinstance(region, RegionSpec):
        return region
    if isinstance(region, (tuple, list)):
        return RegionSpec(dims=tuple(RegionRange(value, 1) for value in region))
    raise TypeError(f"region_from_tile must return RegionSpec, tuple, or list, got {region!r}")


def _resolve_region(region: RegionSpec | None, env: dict[VarSpec, int] | None) -> RegionSpec | None:
    if region is None:
        return None
    return RegionSpec(
        dims=tuple(
            RegionRange(
                start=_resolve_expr_value(dim.start, env),
                extent=_resolve_expr_value(dim.extent, env),
            )
            for dim in region.dims
        )
    )


def _validate_region_from_tile_shape(
    tensor: TensorSpec, region_from_tile, grid, label: str, var_ids: set[int] | None = None
) -> None:
    dim = len(_shape_tuple(tensor.shape))
    tile_extents = _static_int_tuple(grid)
    sample = (0, 0, 0)
    if tile_extents is not None:
        sample = tuple(0 for _ in tile_extents)
    region = _region_from_tile(region_from_tile, *sample)
    if region is None:
        return
    if len(region.dims) != dim:
        raise ValueError(
            f"{label} has {len(region.dims)} dims, but tensor {tensor.name!r} has {dim} dims"
        )
    for dim in region.dims:
        if not _is_expr_like(dim.start):
            raise TypeError(f"{label} contains unsupported start value {dim.start!r}")
        if not _is_expr_like(dim.extent):
            raise TypeError(f"{label} contains unsupported extent value {dim.extent!r}")
        if var_ids is not None:
            _validate_expr_ownership((dim.start, dim.extent), var_ids, label)


def _validate_region_bounds(
    tensor: TensorSpec, region: RegionSpec | None, shape: tuple[int, ...], label: str
) -> None:
    if region is None:
        return
    if len(region.dims) != len(shape):
        raise ValueError(
            f"{label} has {len(region.dims)} dims, but tensor {tensor.name!r} "
            f"has {len(shape)} dims"
        )
    for dim, extent in zip(region.dims, shape):
        if not isinstance(dim.start, int) or isinstance(dim.start, bool):
            raise TypeError(f"{label} static region contains non-integer start {dim.start!r}")
        if not isinstance(dim.extent, int) or isinstance(dim.extent, bool):
            raise TypeError(f"{label} static region contains non-integer extent {dim.extent!r}")
        if dim.extent <= 0:
            raise ValueError(f"{label} region {region_label(region)} has non-positive extent")
        if dim.start < 0 or dim.start + dim.extent > extent:
            raise ValueError(
                f"{label} region {region_label(region)} is out of bounds for tensor "
                f"{tensor.name!r} shape {shape}"
            )


def region_label(region: RegionSpec | None) -> str:
    if region is None:
        return "unknown"
    parts = []
    for dim in region.dims:
        if dim.extent == 1:
            parts.append(str(dim.start))
        else:
            parts.append(f"{dim.start}:{dim.start + dim.extent}")
    return "[" + ", ".join(parts) + "]"



def _collect_kernel_vars(kernel: KernelSpec) -> list[VarSpec]:
    seen: set[VarSpec] = set()
    result: list[VarSpec] = []

    def add_from(value: Any) -> None:
        for var in expr_vars(value):
            if var not in seen:
                seen.add(var)
                result.append(var)

    for var in kernel.vars.values():
        add_from(var)
    for tensor in kernel.tensors.values():
        add_from(tensor.shape)
    for event in kernel.events.values():
        add_from(event.shape)
    for tile in kernel.tiles:
        add_from(tile.grid)
    return result


def _sample_kernel_envs(kernel: KernelSpec) -> list[dict[VarSpec, int] | None]:
    vars_seen = list(_collect_kernel_vars(kernel))
    if not vars_seen:
        return [None]

    rng = random.Random(0)
    candidates = []
    for base in (1, 2, 4, 8, 16):
        candidates.append(dict(_sample_value_for_base(var, base) for var in vars_seen))
    bounded_values = [_range_sample_values(var) for var in vars_seen]
    if all(values is not None for values in bounded_values):
        for values in zip(*bounded_values):
            candidates.append(dict(zip(vars_seen, values)))
    for _ in range(5):
        candidates.append({var: _random_sample_value(var, rng) for var in vars_seen})

    unique = []
    seen = set()
    for env in candidates:
        key = tuple((var.name, env[var]) for var in vars_seen)
        if key not in seen:
            seen.add(key)
            unique.append(env)
    return [env for env in unique if _kernel_sample_fits_budget(kernel, env)]


def _kernel_sample_fits_budget(kernel: KernelSpec, env: dict[VarSpec, int]) -> bool:
    budget = 65536
    total = 0
    for tensor in kernel.tensors.values():
        total += _extent_product(_static_int_tuple(tensor.shape, env))
    for event in kernel.events.values():
        total += _extent_product(_static_int_tuple(event.shape, env))
    for tile in kernel.tiles:
        total += _extent_product(_static_int_tuple(tile.grid, env))
    return total <= budget


def _shape_tuple(shape: Any) -> tuple[Any, ...]:
    return tuple(shape) if isinstance(shape, (tuple, list)) else (shape,)


def _static_int_tuple(values: Any, env: dict[VarSpec, int] | None = None) -> tuple[int, ...] | None:
    result = []
    for value in _shape_tuple(values):
        resolved = eval_expr_like(value, env)
        if resolved is None:
            return None
        result.append(resolved)
    return tuple(result)


def _dependency_info_from_map(
    coord_fn,
    m_idx,
    n_idx,
    k_idx,
    notify_i=0,
    *,
    tensor_ids: set[int] | None = None,
) -> tuple[Any, ...] | None:
    if not callable(coord_fn):
        raise TypeError("dependency coord must be callable")
    _validate_dependency_coord_function(coord_fn, tensor_ids)
    try:
        mapped = coord_fn(m_idx, n_idx, k_idx, notify_i)
    except TypeError:
        if tensor_ids is not None and _coord_captures_kernel_tensor(coord_fn, tensor_ids):
            return None
        raise
    if not isinstance(mapped, (tuple, list)):
        raise TypeError(f"dependency coord must return tuple/list, got {mapped!r}")
    if len(mapped) < 2:
        raise ValueError("dependency coord must return (coord_count, rank, *event_coord)")
    return tuple(mapped)


def _coord_from_map(
    coord_fn,
    m_idx,
    n_idx,
    k_idx,
    notify_i=0,
    *,
    tensor_ids: set[int] | None = None,
) -> tuple[Any, ...] | None:
    info = _dependency_info_from_map(
        coord_fn, m_idx, n_idx, k_idx, notify_i, tensor_ids=tensor_ids
    )
    return None if info is None else info[2:]


def _validate_dependency_coord_function(coord_fn, tensor_ids: set[int] | None = None) -> None:
    if not isinstance(coord_fn, FunctionType):
        raise TypeError("dependency coord must be a Python function or lambda")
    if coord_fn.__defaults__ is not None or coord_fn.__kwdefaults__ is not None:
        raise TypeError("dependency coord must not use default arguments")
    if tensor_ids is None:
        return
    for name in coord_fn.__code__.co_names:
        value = coord_fn.__globals__.get(name)
        if isinstance(value, TensorSpec) and id(value.base_tensor) in tensor_ids:
            raise TypeError("dependency coord must capture TensorSpec through closure, not globals")


def _coord_captures_kernel_tensor(coord_fn, tensor_ids: set[int]) -> bool:
    closure = getattr(coord_fn, "__closure__", None)
    if closure is None:
        return False
    for cell in closure:
        try:
            value = cell.cell_contents
        except ValueError:
            continue
        if isinstance(value, TensorSpec) and id(value.base_tensor) in tensor_ids:
            return True
    return False


def _dependency_coord_is_static(coord: tuple[Any, ...]) -> bool:
    return all(_is_expr_like(value) for value in coord)


def _static_dependency_coord(
    dependency, m_idx, n_idx, k_idx, notify_i=0, env=None, tensor_ids: set[int] | None = None
):
    coord = _coord_from_map(
        dependency.coord, m_idx, n_idx, k_idx, notify_i, tensor_ids=tensor_ids
    )
    if coord is None:
        return None
    if not _dependency_coord_is_static(coord):
        return None
    return _resolve_coord(coord, env)


def _event_has_runtime_dependency_coord(
    event: EventSpec, producers, consumers, tensor_ids: set[int]
) -> bool:
    for tile in (*producers, *consumers):
        for dependency in (*tile.notifies, *tile.waits):
            if dependency.event is not event:
                continue
            coord = _coord_from_map(dependency.coord, 0, 0, 0, tensor_ids=tensor_ids)
            if coord is None or not _dependency_coord_is_static(coord):
                return True
    return False


def _validate_dependency_shape(
    event: EventSpec,
    dependency,
    grid,
    label: str,
    var_ids: set[int] | None = None,
    tensor_ids: set[int] | None = None,
    *,
    is_wait: bool,
) -> None:
    dim = len(_shape_tuple(event.shape))
    tile_extents = _static_int_tuple(grid)
    sample = (0, 0, 0)
    if tile_extents is not None:
        sample = tuple(0 for _ in tile_extents)
    info = _dependency_info_from_map(dependency.coord, *sample, 0, tensor_ids=tensor_ids)
    if info is None:
        return
    coord_count, rank, *coord = info
    if len(coord) != dim:
        raise ValueError(
            f"{label} coord has {len(coord)} dims, but event {event.name!r} has {dim} dims"
        )
    if not _is_expr_like(coord_count):
        raise TypeError(f"{label} coord_count contains unsupported value {coord_count!r}")
    if not _is_expr_like(rank):
        raise TypeError(f"{label} rank contains unsupported value {rank!r}")
    if is_wait and coord_count != 1:
        raise ValueError(f"{label} wait dependency must have coord_count == 1")
    if is_wait and rank != -1:
        raise ValueError(f"{label} wait dependency must have rank == -1")
    values = [coord_count, rank]
    for value in coord:
        if not _is_expr_like(value):
            raise TypeError(f"{label} coord contains unsupported value {value!r}")
        values.append(value)
    if var_ids is not None:
        _validate_expr_ownership(tuple(values), var_ids, label)


def _validate_static_event_counts(event: EventSpec, producers, consumers, tensor_ids: set[int]) -> None:
    if _event_has_runtime_dependency_coord(event, producers, consumers, tensor_ids):
        return
    exact_envs = [None]
    if _static_int_tuple(event.shape) is None or any(
        _static_int_tuple(producer.grid) is None for producer in producers
    ) or any(_static_int_tuple(consumer.grid) is None for consumer in consumers):
        exact_envs = _sample_var_envs(event, producers, consumers)
    for env in exact_envs:
        event_shape = _static_int_tuple(event.shape, env)
        if event_shape is None:
            continue
        _validate_event_counts_for_shape(event, producers, consumers, event_shape, env, tensor_ids)


def _validate_event_counts_for_shape(
    event: EventSpec, producers, consumers, event_shape: tuple[int, ...], env, tensor_ids: set[int]
) -> None:
    notify_counts: dict[tuple[int, ...], int] = defaultdict(int)
    for producer in producers:
        tile_extents = _static_int_tuple(producer.grid, env)
        if tile_extents is None:
            return
        for idx in product(*(range(extent) for extent in tile_extents)):
            for dependency in producer.notifies:
                notify_event = dependency.event
                coord = dependency.coord
                if notify_event is not event:
                    continue
                info = _dependency_info_from_map(coord, *idx, 0)
                if info is None:
                    return
                coord_count = _resolve_expr_value(info[0], env)
                if isinstance(coord_count, bool) or not isinstance(coord_count, int) or coord_count < 1:
                    raise ValueError(f"{producer.name}.notify coord_count must be a positive integer")
                for notify_i in range(coord_count):
                    coord_value = _static_dependency_coord(
                        dependency, *idx, notify_i=notify_i, env=env, tensor_ids=tensor_ids
                    )
                    if coord_value is None:
                        return
                    _validate_static_coord(event, coord_value, event_shape, f"{producer.name}.notify")
                    notify_counts[coord_value] += 1

    for coord in product(*(range(extent) for extent in event_shape)):
        expected = event_init_count(event, coord, env)
        actual = notify_counts.get(coord, 0)
        if actual != expected:
            raise ValueError(
                f"event {event.name!r} coord {coord} expects init_count {expected}, "
                f"but has {actual} static notifies"
            )

    for consumer in consumers:
        tile_extents = _static_int_tuple(consumer.grid, env)
        if tile_extents is None:
            return
        for idx in product(*(range(extent) for extent in tile_extents)):
            for dependency in consumer.waits:
                wait_event = dependency.event
                coord = dependency.coord
                if wait_event is not event:
                    continue
                coord = _static_dependency_coord(dependency, *idx, env=env, tensor_ids=tensor_ids)
                if coord is None:
                    continue
                _validate_static_coord(event, coord, event_shape, f"{consumer.name}.wait")
                if notify_counts.get(coord, 0) == 0:
                    raise ValueError(
                        f"{consumer.name!r} waits on event {event.name!r} coord {coord} "
                        "without a producer notify"
                    )


def _sample_var_envs(event: EventSpec, producers, consumers) -> list[dict[VarSpec, int]]:
    vars_seen: list[VarSpec] = []
    seen_ids: set[int] = set()

    def visit(value: Any) -> None:
        for var in expr_vars(value):
            if id(var) not in seen_ids:
                seen_ids.add(id(var))
                vars_seen.append(var)

    visit(event.shape)
    for tile in (*producers, *consumers):
        visit(tile.grid)

    if not vars_seen:
        return []

    rng = random.Random(0)
    candidates = []
    for base in (1, 2, 4, 8, 16):
        candidates.append(dict(_sample_value_for_base(var, base) for var in vars_seen))

    bounded_values = [_range_sample_values(var) for var in vars_seen]
    if all(values is not None for values in bounded_values):
        for values in zip(*bounded_values):
            candidates.append(dict(zip(vars_seen, values)))

    for _ in range(5):
        candidates.append({var: _random_sample_value(var, rng) for var in vars_seen})

    unique = []
    seen = set()
    for env in candidates:
        key = tuple((var.name, env[var]) for var in vars_seen)
        if key not in seen:
            seen.add(key)
            unique.append(env)
    return [env for env in unique if _sample_fits_budget(event, producers, consumers, env)]


def _sample_value_for_base(var: VarSpec, base: int) -> tuple[VarSpec, int]:
    if var.bounds is None:
        return var, base
    lo, hi = var.bounds
    return var, min(max(base, lo), hi)


def _range_sample_values(var: VarSpec) -> tuple[int, ...] | None:
    if var.bounds is None:
        return None
    lo, hi = var.bounds
    mid = (lo + hi) // 2
    return tuple(dict.fromkeys((lo, mid, hi)))


def _random_sample_value(var: VarSpec, rng: random.Random) -> int:
    if var.bounds is None:
        return rng.randint(1, 16)
    lo, hi = var.bounds
    return rng.randint(lo, hi)


def _sample_fits_budget(event: EventSpec, producers, consumers, env: dict[VarSpec, int]) -> bool:
    budget = 65536
    total = _extent_product(_static_int_tuple(event.shape, env))
    for tile in (*producers, *consumers):
        total += _extent_product(_static_int_tuple(tile.grid, env))
    return total <= budget


def _extent_product(extents: tuple[int, ...] | None) -> int:
    if extents is None:
        return 0
    return reduce(mul, extents, 1)


def _resolve_coord(coord: tuple[Any, ...], env: dict[VarSpec, int] | None) -> tuple[Any, ...]:
    return tuple(_resolve_expr_value(value, env) for value in coord)


def _resolve_expr_value(value: Any, env: dict[VarSpec, int] | None) -> Any:
    resolved = eval_expr_like(value, env)
    return value if resolved is None else resolved


def _is_expr_like(value: Any) -> bool:
    return (isinstance(value, int) and not isinstance(value, bool)) or isinstance(
        value, (VarSpec, ExprSpec)
    )


def _validate_static_coord(
    event: EventSpec, coord: tuple[Any, ...], shape: tuple[int, ...], label: str
) -> None:
    if len(coord) != len(shape):
        raise ValueError(
            f"{label} coord has {len(coord)} dims, but event {event.name!r} "
            f"has {len(shape)} dims"
        )
    for value, extent in zip(coord, shape):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{label} static coord contains non-integer value {value!r}")
        if value < 0 or value >= extent:
            raise ValueError(
                f"{label} coord {coord} is out of bounds for event {event.name!r} "
                f"shape {shape}"
            )


__all__ = ["LogicalEdge", "event_init_count", "logical_edges", "validate_kernel"]
