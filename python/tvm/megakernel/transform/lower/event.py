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
"""Event lowering helpers for megakernel DSL transforms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import tvm.tirx.script as T

from ...dsl import EventSpec, ExprSpec, VarSpec
from .prepare import EVENT_INIT_COMPLETE_NAME, INIT_EVENT_JOB_ID, WAIT_EVENT_INIT_JOB_ID
from .scheduler import StaticTileScheduler, TIRXSemaphore


EVENT_WAIT_MARKER = "tirx.megakernel.event.wait"
EVENT_NOTIFY_MARKER = "tirx.megakernel.event.notify"


@dataclass
class EventBinding:
    """Lowering-time storage binding for one logical event."""

    event: EventSpec | None
    buffer: Any
    size: Any


@T.inline
def _wait_event_init_complete(
    buffer, coord: T.constexpr, sm_count: T.constexpr, warp_count: T.constexpr
):
    state = T.alloc_buffer((1,), "int32", scope="local", align=4)
    state[0] = -1
    warp_id = T.warp_id([warp_count])
    lane_id = T.lane_id([32])
    while 1:
        if lane_id == 0:
            T.ptx.ld_global_acquire(state[0], T.address_of(buffer[coord]))
        if T.ptx.any_sync(
            0xFFFFFFFF,
            (state[0] <= sm_count * (TIRXSemaphore.base + 1)) & (state[0] > 0),
        ):
            if (lane_id == 0) & (warp_id == 0):
                T.cuda.atomic_add(T.address_of(buffer[coord]), -(TIRXSemaphore.base + 1))
            break
        T.cuda.nano_sleep(40)


class EventLoweringMixin:
    """Emit event backing storage, initialization, waits, and notifications."""

    def emit_init_event_task(
        self,
        plan,
        scheduler: StaticTileScheduler,
        event_bindings: dict[str, EventBinding],
        event_id,
        _linear_idx,
    ) -> None:
        events = list(plan.kernel.events.values())
        num_threads = plan.options.attrs.get("num_threads", 256)
        tid = T.thread_id([num_threads])

        for static_event_id, event in enumerate(events):
            binding = event_bindings[event.name]
            with T.If(event_id == static_event_id):
                with T.Then():
                    idx = T.alloc_buffer((1,), "int32", scope="local")
                    T.buffer_store(idx, tid, [0])
                    with T.While(idx[0] < binding.size):
                        coord = linear_index_to_coord(
                            idx[0], event_shape_tuple(event.shape, f"event {event.name} shape", plan)
                        )
                        init_count = event_init_count(event, coord)
                        T.buffer_store(
                            binding.buffer,
                            init_count * (TIRXSemaphore.base + 1),
                            list(coord),
                        )
                        T.buffer_store(idx, idx[0] + num_threads, [0])
                    self.emit_event_init_complete_notify(plan, scheduler)

        if plan.event_init_complete is not None:
            with T.If(event_id == len(events)):
                with T.Then():
                    sm_count = plan.options.attrs.get("sm_count", 1)
                    complete_init = (len(events) + 1 + sm_count) * (TIRXSemaphore.base + 1)
                    T.buffer_store(
                        plan.event_init_complete.buffer,
                        complete_init,
                        [event_init_complete_coord(plan)],
                    )
                    self.emit_event_init_complete_notify(plan, scheduler)

    def emit_event_init_complete_notify(self, plan, scheduler: StaticTileScheduler) -> None:
        if plan.event_init_complete is None:
            return
        semaphore = TIRXSemaphore(plan.event_init_complete.buffer)

        def notify_func(_notify_idx):
            return (1, -1, event_init_complete_coord(plan))

        scheduler.notify(semaphore, notify_func, scope="cta")

    def emit_wait_event_init_task(self, plan, scheduler: StaticTileScheduler) -> None:
        if plan.event_init_complete is None:
            return
        _wait_event_init_complete(
            plan.event_init_complete.buffer,
            event_init_complete_coord(plan),
            plan.options.attrs.get("sm_count", 1),
            scheduler.warp_count,
        )

    def bind_event_buffers(self, plan, event_workspace) -> dict[str, EventBinding]:
        events = list(plan.kernel.events.values())
        if not events:
            return {}
        if event_workspace is None:
            raise ValueError("event lowering requires an event workspace argument")

        for event_plan in plan.event_layouts:
            event = event_plan.event
            shape = event_shape_tuple(event_plan.shape, f"event {event_plan.name} shape", plan)
            size = shape_product(shape)
            buffer = T.decl_buffer(
                shape,
                event_plan.dtype,
                data=event_workspace.data,
                elem_offset=event_plan.workspace_offset,
                scope="global",
            )
            plan.event_bindings[event_plan.name] = EventBinding(
                event=event,
                buffer=buffer,
                size=size,
            )

        plan.event_init_complete = EventBinding(
            event=None,
            buffer=event_workspace,
            size=1,
        )
        return plan.event_bindings


def emit_events(
    dependencies,
    marker: str,
    scheduler: StaticTileScheduler | None,
    event_bindings: dict[str, EventBinding],
    m_idx,
    n_idx,
    k_idx,
    options,
) -> None:
    if not dependencies:
        return
    for event, coord_map in dependencies:
        coord = coord_from_map(coord_map, m_idx, n_idx, k_idx)
        if event.name not in event_bindings:
            if options.emit_event_markers:
                emit_marker(marker, event.name, *coord)
        elif marker == EVENT_WAIT_MARKER:
            semaphore = TIRXSemaphore(event_bindings[event.name].buffer)
            scheduler.wait(semaphore, *coord)
        elif marker == EVENT_NOTIFY_MARKER:
            semaphore = TIRXSemaphore(event_bindings[event.name].buffer)

            def notify_func(_notify_idx, coord=coord):
                return (1, -1, *coord)

            scheduler.notify(semaphore, notify_func, scope="cta")
        else:
            raise ValueError(f"Unknown event marker: {marker}")


def event_workspace_size(events: list[EventSpec], plan) -> Any:
    if not events:
        return 0
    return event_init_task_count(events, plan) + 1


def event_init_task_count(events: list[EventSpec], plan) -> Any:
    result = 0
    for event in events:
        result += shape_product(event_shape_tuple(event.shape, f"event {event.name} shape", plan))
    return result


def event_init_complete_coord(plan) -> Any:
    if plan.event_init_complete_layout is not None:
        return plan.event_init_complete_layout.workspace_offset
    return event_init_task_count(list(plan.kernel.events.values()), plan)


def event_shape_tuple(shape: Any, context: str, plan) -> tuple[Any, ...]:
    if isinstance(shape, (tuple, list)):
        return tuple(event_lower_expr_like(dim, context, plan) for dim in shape)
    return (event_lower_expr_like(shape, context, plan),)


def event_lower_expr_like(value: Any, context: str, plan) -> Any:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, VarSpec):
        if value not in plan.var_bindings or plan.var_bindings[value].value is None:
            raise ValueError(f"{context} uses unbound symbolic VarSpec({value.name!r})")
        return plan.var_bindings[value].value
    if isinstance(value, ExprSpec):
        args = [event_lower_expr_like(arg, context, plan) for arg in value.args]
        if value.op == "add":
            return args[0] + args[1]
        if value.op == "sub":
            return args[0] - args[1]
        if value.op == "mul":
            return args[0] * args[1]
        if value.op == "floordiv":
            return args[0] // args[1]
        if value.op == "mod":
            return args[0] % args[1]
        if value.op == "neg":
            return -args[0]
        if value.op == "ceildiv":
            return (args[0] + args[1] - 1) // args[1]
        raise ValueError(f"{context} uses unsupported ExprSpec op {value.op!r}")
    raise TypeError(f"{context} must be an int, VarSpec, or ExprSpec, got {value!r}")


def shape_product(shape: tuple[Any, ...]) -> Any:
    result = 1
    for extent in shape:
        result *= extent
    return result


def linear_index_to_coord(linear_idx, shape: tuple[int, ...]) -> tuple[Any, ...]:
    coord = []
    remaining = linear_idx
    for extent in reversed(shape):
        coord.append(remaining % extent)
        remaining = remaining // extent
    return tuple(reversed(coord))


def event_init_count(event: EventSpec, coord: tuple[Any, ...]):
    if callable(event.init_count):
        return event.init_count(coord)
    return event.init_count


def coord_from_map(coord_map, m_idx, n_idx, k_idx) -> tuple[Any, ...]:
    if callable(coord_map):
        coord = coord_map(m_idx, n_idx, k_idx)
    else:
        coord = coord_map
    if not isinstance(coord, (tuple, list)):
        raise TypeError(f"coord_map must produce a tuple/list coordinate, got {coord!r}")
    return tuple(coord)


def emit_marker(name: str, *args: Any) -> None:
    T.evaluate(T.call_extern("void", name, *args))
