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
from types import FunctionType
from typing import Any

import tvm.tirx.script as T

from ...dsl.spec import DependencySpec, EventSpec, ExprSpec, TensorSpec, VarSpec
from .prepare import (
    EVENT_INIT_COMPLETE_NAME,
    INIT_EVENT_JOB_ID,
    WAIT_EVENT_INIT_JOB_ID,
    TensorBinding,
    replace_tensor_specs,
)
from .scheduler import StaticTileScheduler, StaticTIRXSemaphore


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
            (state[0] <= sm_count * (StaticTIRXSemaphore.base + 1)) & (state[0] > 0),
        ):
            if (lane_id == 0) & (warp_id == 0):
                T.cuda.atomic_add(T.address_of(buffer[coord]), -(StaticTIRXSemaphore.base + 1))
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
                        init_count = event_init_count(event, coord, plan)
                        T.buffer_store(
                            binding.buffer,
                            init_count * (StaticTIRXSemaphore.base + 1),
                            list(coord),
                        )
                        T.buffer_store(idx, idx[0] + num_threads, [0])
                    self.emit_event_init_complete_notify(plan, scheduler)

        if plan.event_init_complete is not None:
            with T.If(event_id == len(events)):
                with T.Then():
                    sm_count = plan.options.attrs.get("sm_count", 1)
                    complete_init = (len(events) + 1 + sm_count) * (StaticTIRXSemaphore.base + 1)
                    T.buffer_store(
                        plan.event_init_complete.buffer,
                        complete_init,
                        [event_init_complete_coord(plan)],
                    )
                    self.emit_event_init_complete_notify(plan, scheduler)

    def emit_event_init_complete_notify(self, plan, scheduler: StaticTileScheduler) -> None:
        if plan.event_init_complete is None:
            return
        semaphore = StaticTIRXSemaphore(plan.event_init_complete.buffer)

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


def event_init_count(event: EventSpec, coord: tuple[Any, ...], plan=None):
    count = event.init_count(*coord)
    if isinstance(count, bool):
        raise TypeError("event init_count must produce an integer")
    if isinstance(count, int):
        if count < 0:
            raise ValueError("event init_count must produce a non-negative integer")
        return count
    if isinstance(count, (VarSpec, ExprSpec)):
        if plan is None:
            raise TypeError("symbolic event init_count requires a lowering plan")
        return event_lower_expr_like(count, f"event {event.name} init_count", plan)
    raise TypeError("event init_count must produce an integer, VarSpec, or ExprSpec")


def dependency_info_from_map(
    dependency: DependencySpec,
    m_idx,
    n_idx,
    k_idx,
    notify_i,
    *,
    tensor_bindings: dict[TensorSpec, TensorBinding] | None = None,
) -> tuple[Any, ...]:
    coord = dependency.coord
    if tensor_bindings is not None:
        coord = bind_dependency_coord(coord, tensor_bindings)
    info = coord(m_idx, n_idx, k_idx, notify_i)
    if tensor_bindings is not None:
        info = replace_tensor_specs(info, tensor_bindings)
    if not isinstance(info, (tuple, list)):
        raise TypeError(f"dependency coord must return tuple/list, got {info!r}")
    if len(info) < 2:
        raise ValueError("dependency coord must return (notify_num, rank, *event_coord)")
    return tuple(info)


def coord_from_map(
    dependency: DependencySpec,
    m_idx,
    n_idx,
    k_idx,
    notify_i=0,
    *,
    tensor_bindings: dict[TensorSpec, TensorBinding] | None = None,
) -> tuple[Any, ...]:
    return dependency_info_from_map(
        dependency, m_idx, n_idx, k_idx, notify_i, tensor_bindings=tensor_bindings
    )[2:]


def bind_dependency_coord(
    coord, tensor_bindings: dict[TensorSpec, TensorBinding]
):
    if not isinstance(coord, FunctionType):
        raise TypeError("dependency coord must be a Python function or lambda")
    if coord.__defaults__ is not None or coord.__kwdefaults__ is not None:
        raise TypeError("dependency coord must not use default arguments")
    _reject_global_tensor_refs(coord)
    closure = coord.__closure__
    if closure is None:
        return coord
    cells = tuple(_bind_dependency_cell(cell, tensor_bindings) for cell in closure)
    bound = FunctionType(coord.__code__, coord.__globals__, coord.__name__, None, cells)
    bound.__dict__.update(getattr(coord, "__dict__", {}))
    bound.__annotations__ = getattr(coord, "__annotations__", {}).copy()
    bound.__qualname__ = getattr(coord, "__qualname__", coord.__name__)
    return bound


def _reject_global_tensor_refs(coord) -> None:
    for name in coord.__code__.co_names:
        value = coord.__globals__.get(name)
        if isinstance(value, TensorSpec):
            raise TypeError(
                "dependency coord must capture TensorSpec through closure, not globals"
            )


def _bind_dependency_cell(cell, tensor_bindings: dict[TensorSpec, TensorBinding]):
    try:
        value = cell.cell_contents
    except ValueError:
        return cell
    if isinstance(value, TensorSpec):
        tensor = value.base_tensor
        if tensor not in tensor_bindings:
            raise ValueError("dependency coord captures TensorSpec outside this kernel")
        return _make_cell(tensor_bindings[tensor].buffer)
    return cell


def _make_cell(value):
    def capture():
        return value

    return capture.__closure__[0]

