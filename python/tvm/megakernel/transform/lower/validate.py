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
"""Validation for lower-private static plans."""

from __future__ import annotations

from .prepare import INIT_EVENT_JOB_ID, WAIT_EVENT_INIT_JOB_ID, StaticLoweringPlan
from .prepare import raw_shape_product


def validate_static_lowering_plan(plan: StaticLoweringPlan) -> StaticLoweringPlan:
    """Validate the default static lowering plan."""

    _validate_job_ids(plan)
    _validate_event_layout(plan)
    _validate_tile_plans(plan)
    _validate_static_schedule(plan)
    return plan


def _validate_job_ids(plan: StaticLoweringPlan) -> None:
    reserved = {INIT_EVENT_JOB_ID, WAIT_EVENT_INIT_JOB_ID}
    if plan.static_schedule is not None:
        reserved.add(plan.static_schedule.end_job_id)
    for tile_plan in plan.tile_plans:
        if tile_plan.job_id in reserved:
            raise ValueError(
                f"tile {tile_plan.tile.name!r} job id {tile_plan.job_id} "
                "collides with a reserved static job id"
            )


def _validate_event_layout(plan: StaticLoweringPlan) -> None:
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


def _validate_tile_plans(plan: StaticLoweringPlan) -> None:
    if len(plan.tile_plans) != len(plan.semantic.tiles):
        raise ValueError("static lowering plan must contain one tile plan per semantic tile")
    names = [tile_plan.tile.name for tile_plan in plan.tile_plans]
    if len(names) != len(set(names)):
        raise ValueError("static lowering plan contains duplicate tile plans")
    event_names = {event.name for event in plan.semantic.events}
    for tile_plan in plan.tile_plans:
        for event, _ in tile_plan.waits:
            if event.name not in event_names:
                raise ValueError(f"tile {tile_plan.tile.name!r} waits on unknown event")
        for event, _ in tile_plan.notifies:
            if event.name not in event_names:
                raise ValueError(f"tile {tile_plan.tile.name!r} notifies unknown event")


def _validate_static_schedule(plan: StaticLoweringPlan) -> None:
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
        count = raw_shape_product(phase.tile_num)
        if count is None:
            return
        total_tasks += count
    if (total_tasks + sm_count - 1) // sm_count > plan.static_schedule.max_tasks:
        raise ValueError("static schedule exceeds queue max_tasks capacity")


__all__ = ["validate_static_lowering_plan"]
