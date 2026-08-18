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
"""Validation for prepared lowering plans.

Validation performed by this module, in call order:

1. Lowering job id consistency:
   - Tile job ids may not collide with reserved event-init, wait-init, or end
     job ids used by the lowering runtime.

2. Event workspace layout:
   - Every event workspace region must have a non-negative offset and positive
     size.
   - Event workspace regions, including the optional init-complete region, may
     not overlap.

3. Tile plan coverage and dependency ownership:
   - The lowering plan must contain exactly one TilePlan per KernelSpec tile.
   - TilePlan names must be unique.
   - Every lowered wait/notify dependency must reference an event declared by
     the same KernelSpec.

4. Static schedule shape:
   - Static lowering must have a static schedule and at least one phase.
   - Every tile job id must appear in the static schedule.
   - If phase grids are statically sized, total per-SM queue usage must fit in
     the configured static queue capacity.

5. Dynamic schedule shape:
   - Dynamic lowering must have a dynamic schedule and at least one entry tile.
   - Tile job ids may not collide with the dynamic end job id.
   - A dynamic tile currently supports at most one wait dependency.
   - Dynamic lowering currently requires exactly one endpoint tile, and that
     endpoint grid must contain exactly one tile when statically known.
   - Notify/wait coord mappings must match the target event dimensions when
     statically checkable.  For statically checkable multi-coordinate notify mappings, every
     notify_i entry must keep a stable coord_count, stay inside the event shape,
     and avoid duplicate event notifications within the same tile notify.
   - Dynamic waits must provide inv_coord.  For statically checkable wait
     coords, every fan-out consumer returned by inv_coord must round-trip
     through the wait coord, stay inside the consumer tile grid, and be unique.
   - Runtime TensorSpec indexing in dynamic coord mappings is allowed, but dim
     and round-trip checks that cannot be proven statically are skipped with a
     warning.
   - If entry grids are statically sized, the initial dynamic queue must fit in
     the configured queue capacity.
"""

from __future__ import annotations

import warnings

from ...dsl.spec import TensorSpec
from ..lower.prepare import INIT_EVENT_JOB_ID, WAIT_EVENT_INIT_JOB_ID, LoweringPlan
from ..lower.prepare import raw_shape_product


def validate_lowering_plan(plan: LoweringPlan) -> LoweringPlan:
    """Validate a prepared lowering plan before TIRX emission."""

    _validate_job_ids(plan)
    _validate_event_layout(plan)
    _validate_tile_plans(plan)
    _validate_static_schedule(plan)
    _validate_dynamic_schedule(plan)
    return plan


def _validate_job_ids(plan: LoweringPlan) -> None:
    reserved = {INIT_EVENT_JOB_ID, WAIT_EVENT_INIT_JOB_ID}
    if plan.static_schedule is not None:
        reserved.add(plan.static_schedule.end_job_id)
    for tile_plan in plan.tile_plans:
        if tile_plan.job_id in reserved:
            raise ValueError(
                f"tile {tile_plan.tile.name!r} job id {tile_plan.job_id} "
                "collides with a reserved lowering job id"
            )


def _validate_event_layout(plan: LoweringPlan) -> None:
    regions: list[tuple[int, int, str]] = []
    for event_layout in plan.event_layouts:
        _add_event_region(
            regions, event_layout.workspace_offset, event_layout.size, event_layout.name
        )
    if plan.event_init_complete_layout is not None:
        _add_event_region(
            regions,
            plan.event_init_complete_layout.workspace_offset,
            plan.event_init_complete_layout.size,
            plan.event_init_complete_layout.name,
        )
    for index, (begin, end, name) in enumerate(regions):
        for other_begin, other_end, other_name in regions[index + 1 :]:
            if begin < other_end and other_begin < end:
                raise ValueError(
                    f"event workspace regions {name!r} and {other_name!r} overlap"
                )


def _add_event_region(
    regions: list[tuple[int, int, str]], offset, size, name: str
) -> None:
    if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
        raise ValueError(f"event {name!r} has invalid workspace offset {offset!r}")
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise ValueError(f"event {name!r} has invalid workspace size {size!r}")
    regions.append((offset, offset + size, name))


def _validate_tile_plans(plan: LoweringPlan) -> None:
    if len(plan.tile_plans) != len(plan.kernel.tiles):
        raise ValueError("lowering plan must contain one tile plan per semantic tile")
    names = [tile_plan.tile.name for tile_plan in plan.tile_plans]
    if len(names) != len(set(names)):
        raise ValueError("lowering plan contains duplicate tile plans")
    event_names = {event.name for event in plan.kernel.events.values()}
    for tile_plan in plan.tile_plans:
        for dependency in tile_plan.waits:
            event = dependency.event
            if event.name not in event_names:
                raise ValueError(f"tile {tile_plan.tile.name!r} waits on unknown event")
        for dependency in tile_plan.notifies:
            event = dependency.event
            if event.name not in event_names:
                raise ValueError(f"tile {tile_plan.tile.name!r} notifies unknown event")


def _validate_static_schedule(plan: LoweringPlan) -> None:
    if plan.options.schedule != "static":
        return
    if plan.static_schedule is None:
        raise ValueError("static schedule requires a static schedule plan")
    if not plan.static_schedule.phases:
        raise ValueError("static schedule cannot be empty")

    phase_job_ids = {phase.job_id for phase in plan.static_schedule.phases}
    for tile_plan in plan.tile_plans:
        if tile_plan.job_id not in phase_job_ids:
            raise ValueError(f"tile {tile_plan.tile.name!r} is missing from static schedule")

    sm_count = plan.options.attrs.get("sm_count", 1)
    total_tasks = 0
    for phase in plan.static_schedule.phases:
        count = raw_shape_product(phase.grid)
        if count is None:
            return
        total_tasks += count
    if (total_tasks + sm_count - 1) // sm_count > plan.static_schedule.max_tasks:
        raise ValueError("static schedule exceeds queue max_tasks capacity")


def _validate_dynamic_schedule(plan: LoweringPlan) -> None:
    if plan.options.schedule != "dynamic":
        return
    if plan.dynamic_schedule is None:
        raise ValueError("dynamic schedule requires a dynamic schedule plan")
    if not plan.dynamic_schedule.entry_phases:
        raise ValueError("dynamic schedule requires at least one entry tile")

    reserved = {plan.dynamic_schedule.end_job_id}
    for tile_plan in plan.tile_plans:
        if tile_plan.job_id in reserved:
            raise ValueError(
                f"tile {tile_plan.tile.name!r} job id {tile_plan.job_id} "
                "collides with a reserved dynamic job id"
            )
        if len(tile_plan.waits) > 1:
            raise ValueError("dynamic schedule currently supports at most one wait per tile")

    endpoints = [tile_plan for tile_plan in plan.tile_plans if not tile_plan.notifies]
    if len(endpoints) != 1:
        raise ValueError("dynamic schedule currently requires exactly one endpoint tile")
    endpoint_count = raw_shape_product(endpoints[0].grid)
    if endpoint_count != 1:
        raise ValueError("dynamic schedule currently requires the endpoint grid product to be 1")

    tensor_ids = {id(tensor) for tensor in plan.kernel.tensors.values()}
    for tile_plan in plan.tile_plans:
        for dependency in tile_plan.notifies:
            event = dependency.event
            coord = dependency.coord
            coord_is_static = _validate_coord_dim(
                coord, len(_shape_tuple(event.shape)), "notify", tensor_ids
            )
            if coord_is_static:
                _validate_notify_coord(coord, _shape_tuple(event.shape))
        for dependency in tile_plan.waits:
            event = dependency.event
            coord = dependency.coord
            inv_coord = dependency.inv_coord
            coord_is_static = _validate_coord_dim(
                coord, len(_shape_tuple(event.shape)), "wait", tensor_ids
            )
            if inv_coord is None:
                raise ValueError("dynamic schedule requires inv_coord for every wait")
            _validate_inv_coord(
                coord, inv_coord, _shape_tuple(event.shape), tile_plan.grid, coord_is_static
            )

    entry_tasks = 0
    for phase in plan.dynamic_schedule.entry_phases:
        count = raw_shape_product(phase.grid)
        if count is None:
            return
        entry_tasks += count
    if entry_tasks > plan.dynamic_schedule.max_tasks:
        raise ValueError("dynamic schedule entry tasks exceed queue max_tasks capacity")


def _shape_tuple(shape):
    return tuple(shape) if isinstance(shape, (tuple, list)) else (shape,)


def _validate_coord_dim(coord_fn, dim: int, label: str, tensor_ids: set[int]) -> bool:
    sample = (101, 203, 307)
    try:
        info = coord_fn(*sample, 0)
    except TypeError:
        if _coord_captures_kernel_tensor(coord_fn, tensor_ids):
            warnings.warn(
                f"dynamic schedule {label} coord uses runtime TensorSpec indexing; "
                "skipping static coord dim validation",
                UserWarning,
                stacklevel=2,
            )
            return False
        raise
    if not isinstance(info, (tuple, list)) or len(info) != dim + 2:
        raise ValueError(f"dynamic schedule {label} coord dim must match event dim")
    return True


def _sample_event_coord(event_shape: tuple) -> tuple[int, ...]:
    sample = []
    for axis, extent in enumerate(event_shape):
        if isinstance(extent, int) and not isinstance(extent, bool) and extent > 1:
            sample.append(min(axis + 1, extent - 1))
        else:
            sample.append(0)
    return tuple(sample)


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


def _validate_notify_coord(coord_fn, event_shape: tuple) -> None:
    sample_tile_coord = (0, 0, 0)
    first_info = coord_fn(*sample_tile_coord, 0)
    if not isinstance(first_info, (tuple, list)) or len(first_info) != len(event_shape) + 2:
        raise ValueError("dynamic schedule notify coord dim must match event dim")
    coord_count = first_info[0]
    if isinstance(coord_count, bool) or not isinstance(coord_count, int) or coord_count < 1:
        raise ValueError("dynamic schedule notify coord_count must be a positive integer")

    seen_notifications = set()
    for notify_i in range(coord_count):
        info = coord_fn(*sample_tile_coord, notify_i)
        if not isinstance(info, (tuple, list)) or len(info) != len(event_shape) + 2:
            raise ValueError("dynamic schedule notify coord dim must match event dim")
        if info[0] != coord_count:
            raise ValueError("dynamic schedule notify coord_count must be stable")
        rank = info[1]
        event_coord = tuple(info[2:])
        _validate_event_coord_in_shape(event_coord, event_shape)
        if isinstance(rank, bool):
            raise ValueError("dynamic schedule notify coord rank must not be bool")
        if isinstance(rank, int) and all(isinstance(value, int) and not isinstance(value, bool) for value in event_coord):
            notification = (rank, event_coord)
            if notification in seen_notifications:
                raise ValueError("dynamic schedule notify coord must not produce duplicate event coords")
            seen_notifications.add(notification)


def _validate_event_coord_in_shape(event_coord: tuple, event_shape: tuple) -> None:
    if len(event_coord) != len(event_shape):
        raise ValueError("dynamic schedule notify coord dim must match event dim")
    for axis, (coord, extent) in enumerate(zip(event_coord, event_shape)):
        if not isinstance(coord, int) or isinstance(coord, bool):
            continue
        if isinstance(extent, int) and not isinstance(extent, bool):
            if coord < 0 or coord >= extent:
                raise ValueError(
                    f"dynamic schedule notify coord axis {axis} is outside event shape"
                )


def _validate_inv_coord(
    coord_fn,
    inv_coord,
    event_shape: tuple,
    consumer_grid,
    coord_is_static: bool,
) -> None:
    event_coord = _sample_event_coord(event_shape)
    first_info = inv_coord(-1, *event_coord, 0)
    if not isinstance(first_info, (tuple, list)) or len(first_info) != 4:
        raise ValueError(
            "dynamic schedule inv_coord must return (consumer_count, tile_m, tile_n, tile_k)"
        )
    consumer_count = first_info[0]
    if isinstance(consumer_count, bool) or not isinstance(consumer_count, int) or consumer_count < 1:
        raise ValueError("dynamic schedule inv_coord consumer_count must be a positive integer")
    if not coord_is_static:
        warnings.warn(
            "dynamic schedule wait coord uses runtime TensorSpec indexing; "
            "skipping static inv_coord round-trip validation",
            UserWarning,
            stacklevel=2,
        )
        return

    seen_consumer_idx = set()
    for consumer_i in range(consumer_count):
        consumer_info = inv_coord(-1, *event_coord, consumer_i)
        if not isinstance(consumer_info, (tuple, list)) or len(consumer_info) != 4:
            raise ValueError(
                "dynamic schedule inv_coord must return (consumer_count, tile_m, tile_n, tile_k)"
            )
        if consumer_info[0] != consumer_count:
            raise ValueError("dynamic schedule inv_coord consumer_count must be stable")
        consumer_idx = tuple(consumer_info[1:])
        _validate_consumer_idx_in_grid(consumer_idx, consumer_grid)
        if consumer_idx in seen_consumer_idx:
            raise ValueError("dynamic schedule inv_coord must not produce duplicate consumer tiles")
        seen_consumer_idx.add(consumer_idx)
        roundtrip = coord_fn(*consumer_idx, 0)
        if not isinstance(roundtrip, (tuple, list)) or tuple(roundtrip[2:]) != event_coord:
            raise ValueError("dynamic schedule inv_coord must round-trip through wait coord")


def _validate_consumer_idx_in_grid(consumer_idx: tuple, consumer_grid) -> None:
    grid = _shape_tuple(consumer_grid)
    if len(consumer_idx) != len(grid):
        raise ValueError("dynamic schedule inv_coord tile coord dim must match consumer grid dim")
    for axis, (idx, extent) in enumerate(zip(consumer_idx, grid)):
        if not isinstance(idx, int) or isinstance(idx, bool):
            continue
        if isinstance(extent, int) and not isinstance(extent, bool):
            if idx < 0 or idx >= extent:
                raise ValueError(
                    f"dynamic schedule inv_coord tile coord axis {axis} is outside consumer grid"
                )


validate_static_lowering_plan = validate_lowering_plan

__all__ = ["validate_lowering_plan", "validate_static_lowering_plan"]
