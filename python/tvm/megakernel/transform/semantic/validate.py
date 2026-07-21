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
"""Semantic validation for megakernel DSL graphs."""

from __future__ import annotations

from collections import defaultdict
from functools import reduce
from itertools import product
from operator import mul
import random
from typing import Any
import warnings

from ...dsl.spec import EventSpec, ExprSpec, RegionRange, RegionSpec, TensorSpec, VarSpec, eval_expr_like, expr_vars
from .build import event_init_count
from .impl_access import collect_impl_access
from .model import SemanticPlan
from .region import region_label, region_set_covers, regions_overlap


def validate_semantic_plan(plan: SemanticPlan) -> SemanticPlan:
    """Validate DSL-level dependencies before lowering."""

    tensor_ids = {id(tensor) for tensor in plan.tensors}
    event_ids = {id(event) for event in plan.events}
    var_ids = {id(var) for var in plan.kernel.vars.values()}
    _validate_kernel_expr_ownership(plan, var_ids)
    tile_names = [tile.name for tile in plan.tiles]
    if len(tile_names) != len(set(tile_names)):
        raise ValueError("semantic plan contains duplicate tile names")

    producers: dict[int, list] = defaultdict(list)
    consumers: dict[int, list] = defaultdict(list)
    for tile in plan.tiles:
        for tensor in tile.reads:
            _validate_tile_tensor_access(tile, tensor, tensor_ids)
        for tensor in tile.writes:
            _validate_tile_tensor_access(tile, tensor, tensor_ids)
        notified_events: set[int] = set()
        for dependency in tile.notifies:
            event = dependency.event
            coord = dependency.coord
            if id(event) not in event_ids:
                raise ValueError(f"tile {tile.name!r} notifies event outside kernel")
            if id(event) in notified_events:
                raise ValueError(
                    f"tile {tile.name!r} notifies event {event.name!r} more than once"
                )
            notified_events.add(id(event))
            _validate_coord_shape(event, coord, tile.grid, f"{tile.name}.notify", var_ids)
            if tile not in producers[id(event)]:
                producers[id(event)].append(tile)
        waited_events: set[int] = set()
        for dependency in tile.waits:
            event = dependency.event
            coord = dependency.coord
            if id(event) not in event_ids:
                raise ValueError(f"tile {tile.name!r} waits on event outside kernel")
            if id(event) in waited_events:
                raise ValueError(
                    f"tile {tile.name!r} waits on event {event.name!r} more than once"
                )
            waited_events.add(id(event))
            _validate_coord_shape(event, coord, tile.grid, f"{tile.name}.wait", var_ids)
            if tile not in consumers[id(event)]:
                consumers[id(event)].append(tile)

    for event in plan.events:
        if consumers[id(event)] and not producers[id(event)]:
            raise ValueError(f"event {event.name!r} is waited on but has no producer")
        if producers[id(event)] and not consumers[id(event)]:
            raise ValueError(f"event {event.name!r} is notified but has no consumer")
        _validate_static_event_counts(event, producers[id(event)], consumers[id(event)])

    _validate_region_dependencies(plan, var_ids)
    _validate_impl_access_contract(plan)

    return plan



def _validate_impl_access_contract(plan: SemanticPlan) -> None:
    tensor_ids = {id(tensor) for tensor in plan.tensors}
    envs = _sample_plan_envs(plan)
    for tile in plan.tiles:
        policy = tile.attrs.get("impl_access_validate", "error")
        if policy not in ("error", "warn", "off"):
            raise ValueError(
                f"tile {tile.name!r} has unsupported impl_access_validate policy {policy!r}"
            )
        if policy == "off":
            continue
        impl_tensors = _tile_impl_tensor_attrs(tile, tensor_ids)
        if not impl_tensors:
            continue
        try:
            actual = collect_impl_access(tile.impl, tensors=impl_tensors, hooks=("prefetch", "run"))
        except Exception as err:
            _handle_impl_access_issue(
                tile, policy, f"impl access collection failed: {err}", force_error=True
            )
            continue
        for effect in actual.unknown_effects:
            warnings.warn(
                f"tile {tile.name!r} impl has unknown access effect: {effect}",
                stacklevel=2,
            )
        _validate_actual_accesses(tile, actual.reads, tile.reads, "read", envs, policy)
        _validate_actual_accesses(tile, actual.writes, tile.writes, "write", envs, policy)


def _handle_impl_access_issue(
    tile, policy: str, message: str, *, force_error: bool = False
) -> None:
    full_message = f"tile {tile.name!r} {message}"
    if force_error or policy == "error":
        raise ValueError(full_message)
    warnings.warn(full_message, stacklevel=3)


def _tile_impl_tensor_attrs(tile, tensor_ids: set[int]) -> dict[str, TensorSpec]:
    result: dict[str, TensorSpec] = {}
    for attr_name, value in vars(tile.impl).items():
        if isinstance(value, TensorSpec) and id(value.base_tensor) in tensor_ids:
            result[attr_name] = value.base_tensor
    return result


def _validate_actual_accesses(tile, actual_accesses, declared_accesses, kind: str, envs, policy: str) -> None:
    declared_by_tensor: dict[TensorSpec, list[TensorSpec]] = defaultdict(list)
    for access in declared_accesses:
        declared_by_tensor[access.base_tensor].append(access)

    for actual in actual_accesses:
        tensor = actual.tensor.base_tensor
        candidates = declared_by_tensor.get(tensor, [])
        if not candidates:
            _handle_impl_access_issue(
                tile,
                policy,
                f"impl {kind}s tensor {tensor.name!r} but tile.{kind}s does not declare it",
            )
            continue
        if any(declared.region_from_tile is None for declared in candidates):
            continue
        if actual.tensor.region_from_tile is None:
            warnings.warn(
                f"tile {tile.name!r} impl {kind}s tensor {tensor.name!r} with unknown region; "
                "cannot prove declared region coverage",
                stacklevel=2,
            )
            continue
        for env in envs:
            tile_extents = _static_int_tuple(tile.grid, env)
            if tile_extents is None:
                continue
            for idx in product(*(range(extent) for extent in tile_extents)):
                actual_region = _region_from_access(actual.tensor, env, *idx)
                declared_regions = [
                    _region_from_access(declared, env, *idx) for declared in candidates
                ]
                if region_set_covers(declared_regions, actual_region):
                    continue
                _handle_impl_access_issue(
                    tile,
                    policy,
                    f"impl {kind}s tensor {tensor.name!r} region "
                    f"{region_label(actual_region)} outside declared tile.{kind}s regions",
                )
                break



def _validate_kernel_expr_ownership(plan: SemanticPlan, var_ids: set[int]) -> None:
    for tensor in plan.tensors:
        _validate_expr_ownership(tensor.shape, var_ids, f"tensor {tensor.name!r} shape")
    for event in plan.events:
        _validate_expr_ownership(event.shape, var_ids, f"event {event.name!r} shape")
    for tile in plan.tiles:
        _validate_expr_ownership(tile.grid, var_ids, f"tile {tile.name!r} grid")


def _validate_expr_ownership(value: Any, var_ids: set[int], label: str) -> None:
    for var in expr_vars(value):
        if id(var) not in var_ids:
            raise ValueError(f"{label} references a VarSpec outside this kernel")


def _validate_tile_tensor_access(tile, access: TensorSpec, tensor_ids: set[int]) -> None:
    tensor = access.base_tensor
    if id(tensor) not in tensor_ids:
        raise ValueError(f"tile {tile.name!r} references tensor outside kernel")


def _validate_region_dependencies(plan: SemanticPlan, var_ids: set[int]) -> None:
    tensor_ids = {id(tensor) for tensor in plan.tensors}
    for tile in plan.tiles:
        for access in tile.reads:
            if access.region_from_tile is not None:
                _validate_tensor_region_access(tile, access, tensor_ids, is_write=False, var_ids=var_ids)
        for access in tile.writes:
            if access.region_from_tile is not None:
                _validate_tensor_region_access(tile, access, tensor_ids, is_write=True, var_ids=var_ids)

    _validate_tensor_dependency_events(plan)
    _validate_region_dependency_coords(plan)
    _validate_waited_region_sources(plan)


def _validate_tensor_dependency_events(plan: SemanticPlan) -> None:
    for producer in plan.tiles:
        write_tensors = {_base_tensor(access) for access in producer.writes}
        if not write_tensors:
            continue
        notify_events = {id(dep.event) for dep in producer.notifies}
        for consumer in plan.tiles:
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


def _validate_region_dependency_coords(plan: SemanticPlan) -> None:
    envs = _sample_plan_envs(plan)
    for env in envs:
        for producer in plan.tiles:
            producer_extents = _static_int_tuple(producer.grid, env)
            if producer_extents is None:
                continue
            producer_writes = list(producer.writes)
            if not producer_writes:
                continue
            for consumer in plan.tiles:
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
                                    producer, producer_idx, consumer, consumer_idx, env
                                ):
                                    continue
                                raise ValueError(
                                    f"tile {consumer.name!r} idx {consumer_idx} reads tensor "
                                    f"{tensor.name!r} region {region_label(read_region)} that overlaps "
                                    f"tile {producer.name!r} idx {producer_idx} write region "
                                    f"{region_label(write_region)} without an event dependency"
                                )


def _validate_waited_region_sources(plan: SemanticPlan) -> None:
    envs = _sample_plan_envs(plan)
    for env in envs:
        for consumer in plan.tiles:
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
                        wait_map = dependency.coord
                        wait_coord = _resolve_coord(_coord_from_map(wait_map, *consumer_idx), env)
                        _validate_waited_region_source(
                            plan, consumer, consumer_idx, tensor, read_region,
                            wait_event, wait_coord, env
                        )


def _validate_waited_region_source(
    plan: SemanticPlan,
    consumer,
    consumer_idx,
    tensor: TensorSpec,
    read_region: RegionSpec | None,
    wait_event: EventSpec,
    wait_coord: tuple[Any, ...],
    env,
) -> None:
    has_same_tensor_event_writer = False
    for producer in plan.tiles:
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
            if not _producer_notifies_coord(producer, producer_idx, wait_event, wait_coord, env):
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


def _producer_notifies_coord(producer, producer_idx, wait_event, wait_coord, env) -> bool:
    for dependency in producer.notifies:
        notify_event = dependency.event
        notify_map = dependency.coord
        if notify_event is not wait_event:
            continue
        notify_coord = _resolve_coord(_coord_from_map(notify_map, *producer_idx), env)
        if notify_coord == wait_coord:
            return True
    return False


def _base_tensor(access: TensorSpec) -> TensorSpec:
    return access.base_tensor


def _has_matching_event_coord(producer, producer_idx, consumer, consumer_idx, env) -> bool:
    for notify_dep in producer.notifies:
        notify_event = notify_dep.event
        notify_map = notify_dep.coord
        notify_coord = _resolve_coord(_coord_from_map(notify_map, *producer_idx), env)
        for wait_dep in consumer.waits:
            wait_event = wait_dep.event
            wait_map = wait_dep.coord
            if wait_event is not notify_event:
                continue
            wait_coord = _resolve_coord(_coord_from_map(wait_map, *consumer_idx), env)
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
    rank = len(_shape_tuple(tensor.shape))
    tile_extents = _static_int_tuple(grid)
    sample = (0, 0, 0)
    if tile_extents is not None:
        sample = tuple(0 for _ in tile_extents)
    region = _region_from_tile(region_from_tile, *sample)
    if region is None:
        return
    if len(region.dims) != rank:
        raise ValueError(
            f"{label} rank {len(region.dims)} does not match tensor {tensor.name!r} rank {rank}"
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
            f"{label} rank {len(region.dims)} does not match tensor {tensor.name!r} "
            f"rank {len(shape)}"
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


def _sample_plan_envs(plan: SemanticPlan) -> list[dict[VarSpec, int] | None]:
    vars_seen = list(plan.vars)
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
    return [env for env in unique if _plan_sample_fits_budget(plan, env)]


def _plan_sample_fits_budget(plan: SemanticPlan, env: dict[VarSpec, int]) -> bool:
    budget = 65536
    total = 0
    for tensor in plan.tensors:
        total += _extent_product(_static_int_tuple(tensor.shape, env))
    for event in plan.events:
        total += _extent_product(_static_int_tuple(event.shape, env))
    for tile in plan.tiles:
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


def _coord_from_map(coord_fn, m_idx, n_idx, k_idx) -> tuple[Any, ...]:
    mapped_coord = coord_fn(m_idx, n_idx, k_idx) if callable(coord_fn) else coord_fn
    if not isinstance(mapped_coord, (tuple, list)):
        raise TypeError(f"coord must return tuple/list, got {mapped_coord!r}")
    return tuple(mapped_coord)


def _validate_coord_shape(
    event: EventSpec, coord, grid, label: str, var_ids: set[int] | None = None
) -> None:
    rank = len(_shape_tuple(event.shape))
    tile_extents = _static_int_tuple(grid)
    sample = (0, 0, 0)
    if tile_extents is not None:
        sample = tuple(0 for _ in tile_extents)
    coord = _coord_from_map(coord, *sample)
    if len(coord) != rank:
        raise ValueError(
            f"{label} coord rank {len(coord)} does not match event {event.name!r} rank {rank}"
        )
    for value in coord:
        if not _is_expr_like(value):
            raise TypeError(f"{label} coord contains unsupported value {value!r}")
    if var_ids is not None:
        _validate_expr_ownership(coord, var_ids, label)


def _validate_static_event_counts(event: EventSpec, producers, consumers) -> None:
    exact_envs = [None]
    if _static_int_tuple(event.shape) is None or any(
        _static_int_tuple(producer.grid) is None for producer in producers
    ) or any(_static_int_tuple(consumer.grid) is None for consumer in consumers):
        exact_envs = _sample_var_envs(event, producers, consumers)
    for env in exact_envs:
        event_shape = _static_int_tuple(event.shape, env)
        if event_shape is None:
            continue
        _validate_event_counts_for_shape(event, producers, consumers, event_shape, env)


def _validate_event_counts_for_shape(
    event: EventSpec, producers, consumers, event_shape: tuple[int, ...], env
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
                coord = _resolve_coord(_coord_from_map(coord, *idx), env)
                _validate_static_coord(event, coord, event_shape, f"{producer.name}.notify")
                notify_counts[coord] += 1

    for coord in product(*(range(extent) for extent in event_shape)):
        expected = event_init_count(event, coord)
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
                coord = _resolve_coord(_coord_from_map(coord, *idx), env)
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
            f"{label} coord rank {len(coord)} does not match event {event.name!r} "
            f"rank {len(shape)}"
        )
    for value, extent in zip(coord, shape):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{label} static coord contains non-integer value {value!r}")
        if value < 0 or value >= extent:
            raise ValueError(
                f"{label} coord {coord} is out of bounds for event {event.name!r} "
                f"shape {shape}"
            )


__all__ = ["validate_semantic_plan"]
