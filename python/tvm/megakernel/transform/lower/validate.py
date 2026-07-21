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
"""Validation for lower-private plans."""

from __future__ import annotations

from .prepare import INIT_EVENT_JOB_ID, WAIT_EVENT_INIT_JOB_ID, LoweringPlan
from .prepare import raw_shape_product


def validate_lowering_plan(plan: LoweringPlan) -> LoweringPlan:
    """Validate the default lowering plan."""

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
    if len(plan.tile_plans) != len(plan.semantic.tiles):
        raise ValueError("lowering plan must contain one tile plan per semantic tile")
    names = [tile_plan.tile.name for tile_plan in plan.tile_plans]
    if len(names) != len(set(names)):
        raise ValueError("lowering plan contains duplicate tile plans")
    event_names = {event.name for event in plan.semantic.events}
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

    for tile_plan in plan.tile_plans:
        for dependency in tile_plan.notifies:
            event = dependency.event
            coord = dependency.coord
            _validate_coord_rank(coord, len(_shape_tuple(event.shape)), "notify")
        for dependency in tile_plan.waits:
            event = dependency.event
            coord = dependency.coord
            inverse_coord = dependency.inverse_coord
            _validate_coord_rank(coord, len(_shape_tuple(event.shape)), "wait")
            if inverse_coord is None:
                raise ValueError(
                    "dynamic schedule requires inverse_coord for every wait"
                )
            _validate_inverse_coord(coord, inverse_coord, len(_shape_tuple(event.shape)))

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


def _validate_coord_rank(coord_fn, rank: int, label: str) -> None:
    sample = (101, 203, 307)
    coord = coord_fn(*sample) if callable(coord_fn) else coord_fn
    if not isinstance(coord, (tuple, list)) or len(coord) != rank:
        raise ValueError(f"dynamic schedule {label} coord rank must match event rank")


def _validate_inverse_coord(coord_fn, inverse_coord, rank: int) -> None:
    event_coord = tuple((101, 203, 307)[:rank])
    consumer_idx = inverse_coord(*event_coord) if callable(inverse_coord) else inverse_coord
    if not isinstance(consumer_idx, (tuple, list)) or len(consumer_idx) != 3:
        raise ValueError("dynamic schedule inverse_coord must return a 3D tile index")
    roundtrip = coord_fn(*consumer_idx) if callable(coord_fn) else coord_fn
    if not isinstance(roundtrip, (tuple, list)) or tuple(roundtrip) != event_coord:
        raise ValueError("dynamic schedule inverse_coord must round-trip through wait coord")


validate_static_lowering_plan = validate_lowering_plan

__all__ = ["validate_lowering_plan", "validate_static_lowering_plan"]
