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
"""Prepare lower-private state from semantic megakernel plans."""

from __future__ import annotations

from dataclasses import dataclass, field
import keyword
import re
from typing import Any

from ...dsl.spec import (
    EventSpec,
    ExprSpec,
    KernelSpec,
    TensorSpec,
    TileSpec,
    VarSpec,
    expr_bounds,
)
from ..semantic import SemanticPlan

INIT_EVENT_JOB_ID = 29
WAIT_EVENT_INIT_JOB_ID = 30
EVENT_INIT_COMPLETE_NAME = "__event_init_complete__"
DEFAULT_MAX_TASKS = 128


@dataclass(frozen=True)
class LoweringOptions:
    """Configuration for DSL-to-TIRX lowering."""

    smem_max_bytes: int = 228 * 1024
    smem_chunk_size: int = 16 * 1024
    schedule: str = "static"
    emit_event_markers: bool = True
    emit_smem_markers: bool = True
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class VarBinding:
    """Final kernel binding for one symbolic variable."""

    var: VarSpec
    param_name: str
    value: Any | None = None


@dataclass
class TensorBinding:
    """Final kernel binding for one logical tensor."""

    tensor: TensorSpec
    param_name: str
    buffer: Any | None = None


@dataclass
class TileLoweringInfo:
    """Lowering binding for one logical tile."""

    tile: TileSpec
    job_id: int
    class_key: Any
    tensor_bindings: dict[TensorSpec, TensorBinding] = field(default_factory=dict)


@dataclass(frozen=True)
class EventLayout:
    """Event workspace layout consumed by lowering."""

    event: EventSpec | None
    name: str
    shape: Any
    dtype: str
    workspace_offset: Any
    size: Any


@dataclass(frozen=True)
class TilePlan:
    """Tile binding consumed by the TIRX emitter."""

    info: TileLoweringInfo
    grid: Any
    waits: tuple[tuple[EventSpec, Any], ...]
    notifies: tuple[tuple[EventSpec, Any], ...]

    @property
    def tile(self) -> TileSpec:
        return self.info.tile

    @property
    def job_id(self) -> int:
        return self.info.job_id


@dataclass(frozen=True)
class TaskPhase:
    """A task range materialized by the queue init kernel."""

    kind: str
    job_id: int
    grid: Any
    label: str


@dataclass(frozen=True)
class StaticSchedulePlan:
    """Static task phase order consumed by queue initialization."""

    phases: tuple[TaskPhase, ...]
    max_tasks: int
    end_job_id: int

    def normalized_data(self) -> dict[str, Any]:
        return {
            "max_tasks": self.max_tasks,
            "end_job_id": self.end_job_id,
            "phases": [
                {
                    "kind": phase.kind,
                    "job_id": phase.job_id,
                    "grid": _data_value(phase.grid),
                    "label": phase.label,
                }
                for phase in self.phases
            ],
        }


@dataclass(frozen=True)
class DynamicTrigger:
    """One dynamic task push emitted after a producer notification."""

    event: EventSpec
    producer: TilePlan
    consumer: TilePlan
    consumer_coord: Any
    consumer_inverse_coord: Any | None


@dataclass(frozen=True)
class DynamicSchedulePlan:
    """Dynamic task queue plan consumed by the TIRX emitter."""

    entry_phases: tuple[TaskPhase, ...]
    triggers: dict[str, tuple[DynamicTrigger, ...]]
    endpoint: TilePlan | None
    max_tasks: int
    end_job_id: int

    def normalized_data(self) -> dict[str, Any]:
        return {
            "max_tasks": self.max_tasks,
            "end_job_id": self.end_job_id,
            "entry_phases": [
                {
                    "kind": phase.kind,
                    "job_id": phase.job_id,
                    "grid": _data_value(phase.grid),
                    "label": phase.label,
                }
                for phase in self.entry_phases
            ],
            "triggers": {
                name: [
                    {
                        "event": trigger.event.name,
                        "producer": trigger.producer.tile.name,
                        "consumer": trigger.consumer.tile.name,
                        "has_inverse_coord": trigger.consumer_inverse_coord is not None,
                    }
                    for trigger in triggers
                ]
                for name, triggers in self.triggers.items()
            },
            "endpoint": None if self.endpoint is None else self.endpoint.tile.name,
        }


@dataclass
class LoweringPlan:
    """Prepared lower-private plan used by the default TIRX emitter."""

    semantic: SemanticPlan
    options: LoweringOptions
    var_order: list[VarSpec] = field(default_factory=list)
    var_bindings: dict[VarSpec, VarBinding] = field(default_factory=dict)
    tensor_order: list[TensorSpec] = field(default_factory=list)
    tensor_bindings: dict[TensorSpec, TensorBinding] = field(default_factory=dict)
    tiles: list[TileLoweringInfo] = field(default_factory=list)
    tile_job_ids: dict[str, int] = field(default_factory=dict)
    event_layouts: list[EventLayout] = field(default_factory=list)
    event_layout_map: dict[str, EventLayout] = field(default_factory=dict)
    event_init_complete_layout: EventLayout | None = None
    tile_plans: list[TilePlan] = field(default_factory=list)
    tile_plan_map: dict[str, TilePlan] = field(default_factory=dict)
    static_schedule: StaticSchedulePlan | None = None
    dynamic_schedule: DynamicSchedulePlan | None = None
    smem_manager: Any | None = None
    event_bindings: dict[str, Any] = field(default_factory=dict)
    event_init_complete: Any | None = None

    @property
    def kernel(self) -> KernelSpec:
        return self.semantic.kernel


    def normalized_data(self) -> dict[str, Any]:
        return {
            "kernel": self.kernel.name,
            "schedule": self.options.schedule,
            "vars": [
                {"name": var.name, "dtype": getattr(var, "dtype", "int32")}
                for var in self.var_order
            ],
            "tensors": [
                {
                    "name": tensor.name,
                    "shape": _data_value(tensor.shape),
                    "dtype": tensor.dtype,
                    "param": self.tensor_bindings[tensor].param_name,
                }
                for tensor in self.tensor_order
            ],
            "events": [
                {
                    "name": event.name,
                    "shape": _data_value(event.shape),
                    "dtype": event.dtype,
                    "offset": _data_value(event.workspace_offset),
                    "size": _data_value(event.size),
                }
                for event in self.event_layouts
            ],
            "event_init_complete": (
                None
                if self.event_init_complete_layout is None
                else {
                    "offset": _data_value(self.event_init_complete_layout.workspace_offset),
                    "size": _data_value(self.event_init_complete_layout.size),
                }
            ),
            "tiles": [
                {
                    "name": tile_plan.tile.name,
                    "job_id": tile_plan.job_id,
                    "grid": _data_value(tile_plan.grid),
                    "waits": [{"event": dep.event.name} for dep in tile_plan.waits],
                    "notifies": [{"event": dep.event.name} for dep in tile_plan.notifies],
                }
                for tile_plan in self.tile_plans
            ],
            "static_schedule": (
                None if self.static_schedule is None else self.static_schedule.normalized_data()
            ),
            "dynamic_schedule": (
                None if self.dynamic_schedule is None else self.dynamic_schedule.normalized_data()
            ),
        }


def _data_value(value: Any) -> Any:
    if isinstance(value, VarSpec):
        return value.name
    if isinstance(value, tuple):
        return tuple(_data_value(item) for item in value)
    if isinstance(value, list):
        return [_data_value(item) for item in value]
    if isinstance(value, dict):
        return {_data_value(key): _data_value(val) for key, val in value.items()}
    return value


def prepare_lowering_plan(
    semantic: SemanticPlan, options: LoweringOptions
) -> LoweringPlan:
    """Create the lower-private plan for the selected schedule."""

    plan = LoweringPlan(semantic=semantic, options=options)
    _bind_vars(plan)
    _bind_tensors(plan)
    _bind_tiles(plan)
    _build_event_layouts(plan)
    _build_tile_plans(plan)
    _build_static_schedule_plan(plan)
    _build_dynamic_schedule_plan(plan)
    return plan


def plan_event_workspace_size(plan: LoweringPlan) -> Any:
    """Return the reserved event workspace length."""

    if not plan.event_layouts:
        return 0
    result = 0
    for event_layout in plan.event_layouts:
        result = max(result, event_layout.workspace_offset + event_layout.size)
    if plan.event_init_complete_layout is not None:
        result = max(
            result,
            plan.event_init_complete_layout.workspace_offset
            + plan.event_init_complete_layout.size,
        )
    return result


def shape_tuple(
    shape: Any, context: str, plan: LoweringPlan | None = None
) -> tuple[Any, ...]:
    """Lower one DSL shape to a tuple, resolving VarSpec when a plan is bound."""

    if isinstance(shape, (tuple, list)):
        return tuple(lower_expr_like(dim, context, plan) for dim in shape)
    return (lower_expr_like(shape, context, plan),)


def shape_product(shape: tuple[Any, ...]) -> Any:
    result = 1
    for extent in shape:
        result *= extent
    return result


def lower_expr_like(value: Any, context: str, plan: LoweringPlan | None = None) -> Any:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, VarSpec):
        if plan is None or value not in plan.var_bindings or plan.var_bindings[value].value is None:
            raise ValueError(f"{context} uses unbound symbolic VarSpec({value.name!r})")
        return plan.var_bindings[value].value
    if isinstance(value, ExprSpec):
        return _lower_expr_spec(value, context, plan)
    raise TypeError(f"{context} must be an int, VarSpec, or ExprSpec, got {value!r}")


def _lower_expr_spec(expr: ExprSpec, context: str, plan: LoweringPlan | None) -> Any:
    args = [lower_expr_like(arg, context, plan) for arg in expr.args]
    if expr.op == "add":
        return args[0] + args[1]
    if expr.op == "sub":
        return args[0] - args[1]
    if expr.op == "mul":
        return args[0] * args[1]
    if expr.op == "floordiv":
        return args[0] // args[1]
    if expr.op == "mod":
        return args[0] % args[1]
    if expr.op == "neg":
        return -args[0]
    if expr.op == "ceildiv":
        return (args[0] + args[1] - 1) // args[1]
    raise ValueError(f"{context} uses unsupported ExprSpec op {expr.op!r}")


def raw_shape_product(shape: Any) -> Any | None:
    return upper_bound_shape_product(shape, context="shape", require_bounded=False)


def upper_bound_shape_product(
    shape: Any, context: str, require_bounded: bool = True
) -> int | None:
    """Return product of static extents, using VarSpec upper bounds when present."""

    values = shape if isinstance(shape, (tuple, list)) else (shape,)
    result = 1
    for extent in values:
        try:
            bounds = expr_bounds(extent, require_bounded=require_bounded)
        except ValueError as err:
            raise ValueError(f"{context} uses {err}") from err
        except TypeError as err:
            raise TypeError(f"{context} must be an int, VarSpec, or ExprSpec") from err
        if bounds is None:
            return None
        result *= bounds[1]
    return result


def _bind_vars(plan: LoweringPlan) -> None:
    used_names: set[str] = set()
    for var in plan.semantic.vars:
        param_name = _sanitize_identifier(var.name, used_names)
        plan.var_order.append(var)
        plan.var_bindings[var] = VarBinding(var=var, param_name=param_name)


def _bind_tensors(plan: LoweringPlan) -> None:
    used_names: set[str] = set()
    for tensor in plan.semantic.tensors:
        param_name = _sanitize_identifier(tensor.name, used_names)
        plan.tensor_order.append(tensor)
        plan.tensor_bindings[tensor] = TensorBinding(tensor=tensor, param_name=param_name)


def _bind_tiles(plan: LoweringPlan) -> None:
    for job_id, tile in enumerate(plan.semantic.tiles):
        info = TileLoweringInfo(tile=tile, job_id=job_id, class_key=type(tile.impl))
        tensor_bindings = {}
        for access in [*tile.reads, *tile.writes]:
            tensor = access.base_tensor
            if tensor in plan.tensor_bindings:
                tensor_bindings[tensor] = plan.tensor_bindings[tensor]
        info.tensor_bindings = tensor_bindings
        plan.tile_job_ids[tile.name] = job_id
        plan.tiles.append(info)


def _build_event_layouts(plan: LoweringPlan) -> None:
    offset: Any = 0
    for event in plan.semantic.events:
        size = upper_bound_shape_product(
            event.shape, f"event {event.name!r} shape", require_bounded=True
        )
        event_layout = EventLayout(
            event=event,
            name=event.name,
            shape=event.shape,
            dtype=event.dtype,
            workspace_offset=offset,
            size=size,
        )
        plan.event_layouts.append(event_layout)
        plan.event_layout_map[event.name] = event_layout
        offset = offset + size
    if plan.event_layouts:
        plan.event_init_complete_layout = EventLayout(
            event=None,
            name=EVENT_INIT_COMPLETE_NAME,
            shape=(1,),
            dtype="int32",
            workspace_offset=offset,
            size=1,
        )


def _build_tile_plans(plan: LoweringPlan) -> None:
    for tile_info in plan.tiles:
        tile = tile_info.tile
        tile_plan = TilePlan(
            info=tile_info,
            grid=tile.grid,
            waits=tuple(tile.waits),
            notifies=tuple(tile.notifies),
        )
        plan.tile_plans.append(tile_plan)
        plan.tile_plan_map[tile.name] = tile_plan


def _build_static_schedule_plan(plan: LoweringPlan) -> None:
    if plan.options.schedule != "static":
        return
    attrs = plan.options.attrs
    sm_count = attrs.get("sm_count", 1)
    max_tasks = attrs.get("max_tasks", DEFAULT_MAX_TASKS)
    end_job_id = attrs.get("end_job_id", 31)
    phases: list[TaskPhase] = []
    if plan.event_layouts:
        phases.append(
            TaskPhase(
                kind="grid",
                job_id=INIT_EVENT_JOB_ID,
                grid=(len(plan.event_layouts) + 1, 1, 1),
                label="init_event",
            )
        )

    entry_tiles = [tile_plan for tile_plan in plan.tile_plans if not tile_plan.waits]
    rest_tiles = [tile_plan for tile_plan in plan.tile_plans if tile_plan.waits]
    for tile_plan in entry_tiles:
        phases.append(_tile_phase(tile_plan))

    if plan.event_layouts:
        phases.append(
            TaskPhase(
                kind="grid",
                job_id=WAIT_EVENT_INIT_JOB_ID,
                grid=(sm_count, 1, 1),
                label="wait_event_init",
            )
        )

    for tile_plan in rest_tiles:
        phases.append(_tile_phase(tile_plan))

    phases.append(TaskPhase(kind="grid", job_id=end_job_id, grid=(sm_count, 1, 1), label="end"))
    plan.static_schedule = StaticSchedulePlan(
        phases=tuple(phases), max_tasks=max_tasks, end_job_id=end_job_id
    )


def _build_dynamic_schedule_plan(plan: LoweringPlan) -> None:
    if plan.options.schedule != "dynamic":
        return
    attrs = plan.options.attrs
    max_tasks = attrs.get("max_tasks", DEFAULT_MAX_TASKS)
    end_job_id = attrs.get("end_job_id", 31)
    entry_tiles = [tile_plan for tile_plan in plan.tile_plans if not tile_plan.waits]
    entry_phases = tuple(_tile_phase(tile_plan) for tile_plan in entry_tiles)

    tile_plan_by_tile = {id(tile_plan.tile): tile_plan for tile_plan in plan.tile_plans}
    triggers: dict[str, list[DynamicTrigger]] = {}
    for producer in plan.tile_plans:
        producer_triggers: list[DynamicTrigger] = []
        for notify_dep in producer.notifies:
            notify_event = notify_dep.event
            for consumer_tile in plan.semantic.tiles:
                for wait_dep in consumer_tile.waits:
                    wait_event = wait_dep.event
                    if wait_event is not notify_event:
                        continue
                    producer_triggers.append(
                        DynamicTrigger(
                            event=notify_event,
                            producer=producer,
                            consumer=tile_plan_by_tile[id(consumer_tile)],
                            consumer_coord=wait_dep.coord,
                            consumer_inverse_coord=wait_dep.inverse_coord,
                        )
                    )
        if producer_triggers:
            triggers[producer.tile.name] = producer_triggers

    endpoints = [tile_plan for tile_plan in plan.tile_plans if not tile_plan.notifies]
    plan.dynamic_schedule = DynamicSchedulePlan(
        entry_phases=entry_phases,
        triggers={name: tuple(value) for name, value in triggers.items()},
        endpoint=endpoints[0] if endpoints else None,
        max_tasks=max_tasks,
        end_job_id=end_job_id,
    )

def _tile_phase(tile_plan: TilePlan) -> TaskPhase:
    return TaskPhase(
        kind="grid",
        job_id=tile_plan.job_id,
        grid=tile_plan.grid,
        label=tile_plan.tile.name,
    )


def _sanitize_identifier(name: str, used_names: set[str]) -> str:
    candidate = re.sub(r"\W", "_", name)
    if not candidate or candidate[0].isdigit() or keyword.iskeyword(candidate):
        candidate = f"tensor_{candidate}"
    base = candidate
    suffix = 1
    while candidate in used_names:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used_names.add(candidate)
    return candidate


def replace_tensor_specs(value: Any, bindings: dict[TensorSpec, TensorBinding]) -> Any:
    """Replace TensorSpec references with their lowered TIRX buffers."""

    if isinstance(value, TensorSpec):
        tensor = value.base_tensor
        if tensor in bindings:
            return bindings[tensor].buffer
    if isinstance(value, tuple):
        return tuple(replace_tensor_specs(item, bindings) for item in value)
    if isinstance(value, list):
        return [replace_tensor_specs(item, bindings) for item in value]
    if isinstance(value, dict):
        return {
            replace_tensor_specs(key, bindings): replace_tensor_specs(val, bindings)
            for key, val in value.items()
        }
    return value


NormalizedMegakernelPlan = LoweringPlan
KernelLoweringPlan = LoweringPlan
StaticLoweringPlan = LoweringPlan
prepare_static_lowering_plan = prepare_lowering_plan

__all__ = [
    "DEFAULT_MAX_TASKS",
    "DynamicTrigger",
    "DynamicSchedulePlan",
    "EVENT_INIT_COMPLETE_NAME",
    "INIT_EVENT_JOB_ID",
    "WAIT_EVENT_INIT_JOB_ID",
    "EventLayout",
    "KernelLoweringPlan",
    "LoweringOptions",
    "LoweringPlan",
    "NormalizedMegakernelPlan",
    "StaticLoweringPlan",
    "StaticSchedulePlan",
    "TaskPhase",
    "TensorBinding",
    "TileLoweringInfo",
    "TilePlan",
    "VarBinding",
    "lower_expr_like",
    "plan_event_workspace_size",
    "prepare_lowering_plan",
    "prepare_static_lowering_plan",
    "raw_shape_product",
    "upper_bound_shape_product",
    "replace_tensor_specs",
    "shape_product",
    "shape_tuple",
]
