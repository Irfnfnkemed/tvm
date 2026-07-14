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
"""Lower megakernel DSL specs to TIRX.

This lowering path intentionally starts simple.  The DSL already records the
logical tile graph, so the first implementation directly emits one static
persistent kernel from the spec instead of building separate tile fragments.

The build owns one ``SmemManager``.  Tile hooks run while the final kernel is
being built, so user parser-style ``TileImpl`` code emits directly into the
current TIRX builder scope.
"""

from __future__ import annotations

import keyword
import re
from dataclasses import dataclass, field
from typing import Any

import tvm
from tvm.ir.module import IRModule
import tvm.tirx.script as T
from tvm.tirx import PrimFunc

from ..dsl import KernelSpec, TensorSpec, TileSpec, VarSpec
from .event import (
    EVENT_NOTIFY_MARKER,
    EVENT_WAIT_MARKER,
    INIT_EVENT_JOB_ID,
    WAIT_EVENT_INIT_JOB_ID,
    EventBinding,
    EventLoweringMixin,
    emit_events,
    event_workspace_size,
)
from .scheduler import StaticTileScheduler
from .smem import TIRXSmemManager


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
    """Lowering-time view of one logical tile."""

    tile: TileSpec
    job_id: int
    class_key: Any
    tensor_bindings: dict[TensorSpec, TensorBinding] = field(default_factory=dict)


@dataclass
class KernelLoweringPlan:
    """Prepared DSL information used by ``KernelBuilder``."""

    kernel: KernelSpec
    options: LoweringOptions
    var_order: list[VarSpec] = field(default_factory=list)
    var_bindings: dict[VarSpec, VarBinding] = field(default_factory=dict)
    tensor_order: list[TensorSpec] = field(default_factory=list)
    tensor_bindings: dict[TensorSpec, TensorBinding] = field(default_factory=dict)
    tiles: list[TileLoweringInfo] = field(default_factory=list)
    tile_job_ids: dict[str, int] = field(default_factory=dict)
    smem_manager: TIRXSmemManager | None = None
    event_bindings: dict[str, "EventBinding"] = field(default_factory=dict)
    event_init_complete: EventBinding | None = None


class _ParserKernelEmitter:
    """Small constexpr object used to emit the kernel inside TIRX parser context."""

    def __init__(self, builder: "KernelBuilder", plan: KernelLoweringPlan):
        self.builder = builder
        self.plan = plan

    def emit(self) -> None:
        self.builder.emit_kernel(self.plan)


@T.jit(check_well_formed=False)
def _megakernel_entry(*, emitter: T.constexpr):
    emitter.emit()


class KernelBuilder(EventLoweringMixin):
    """Directly emit one TIRX megakernel from a prepared plan."""

    def __init__(self, options: LoweringOptions):
        self.options = options

    def build(self, plan: KernelLoweringPlan) -> PrimFunc:
        emitter = _ParserKernelEmitter(self, plan)
        try:
            return _megakernel_entry.specialize(emitter=emitter)
        finally:
            self.restore_tensor_buffers(plan)

    def emit_kernel(self, plan: KernelLoweringPlan) -> None:
        T.func_attr({"global_symbol": plan.kernel.name})
        self.emit_var_args(plan)
        self.bind_tensor_buffers(plan, self.emit_tensor_args(plan))
        event_workspace = self.emit_event_workspace_arg(plan)
        queue = self.emit_static_queue_arg(plan)
        self.emit_profiler_arg(plan)
        T.device_entry()

        smem_manager = self.create_smem_manager(plan)
        event_bindings = self.bind_event_buffers(plan, event_workspace)

        for tile_info in self.unique_tile_classes(plan):
            type(tile_info.tile.impl).init_shared_resources(smem_manager)

        if plan.options.schedule == "static":
            self.emit_static_queue(plan, queue, smem_manager, event_bindings)
        elif plan.options.schedule == "dynamic":
            raise NotImplementedError("dynamic schedule lowering is not implemented yet")
        else:
            raise ValueError(f"Unknown megakernel schedule: {plan.options.schedule!r}")

        for tile_info in reversed(self.unique_tile_classes(plan)):
            type(tile_info.tile.impl).finalize_shared_resources(smem_manager)
        smem_manager.commit()

    def emit_var_args(self, plan: KernelLoweringPlan) -> None:
        _emit_local_symbolic_vars(plan)

    def emit_tensor_args(self, plan: KernelLoweringPlan) -> list[Any]:
        buffers = []
        for tensor in plan.tensor_order:
            binding = plan.tensor_bindings[tensor]
            shape = _shape_tuple(tensor.shape, f"tensor {tensor.name} shape", plan)
            binding.buffer = T.arg(binding.param_name, T.Buffer(shape, tensor.dtype))
            buffers.append(binding.buffer)
        return buffers

    def emit_event_workspace_arg(self, plan: KernelLoweringPlan):
        events = list(plan.kernel.events.values())
        if not events:
            return None
        size = event_workspace_size(events, plan)
        return T.arg("event_workspace", T.Buffer((size,), "int32"))

    def emit_static_queue_arg(self, plan: KernelLoweringPlan):
        if plan.options.schedule != "static":
            return None
        attrs = plan.options.attrs
        return T.arg(
            "queue",
            T.Buffer((attrs.get("sm_count", 1), attrs.get("max_tasks", 128)), "int32"),
        )

    def emit_profiler_arg(self, plan: KernelLoweringPlan):
        profiler_buf_size = plan.options.attrs.get("profiler_buf_size")
        if profiler_buf_size is None:
            return None
        return T.arg("profiler_buf", T.Buffer((profiler_buf_size,), "uint64"))

    def emit_scope_ids(self, attrs: dict[str, Any]) -> dict[str, Any]:
        scope_ids: dict[str, Any] = {}
        if "sm_count" in attrs:
            scope_ids["cta_id"] = T.cta_id([attrs["sm_count"]])
        if "num_threads" in attrs:
            scope_ids["thread_id"] = T.thread_id([attrs["num_threads"]])
        if "warp_count" in attrs:
            scope_ids["warp_id"] = T.warp_id([attrs["warp_count"]])
        if attrs.get("emit_lane_id"):
            scope_ids["lane_id"] = T.thread_id([32])
        return scope_ids

    def emit_static_queue(
        self,
        plan: KernelLoweringPlan,
        queue,
        smem_manager: TIRXSmemManager,
        event_bindings: dict[str, EventBinding],
    ) -> None:
        if queue is None:
            raise ValueError("static scheduling requires a queue argument")
        attrs = plan.options.attrs
        scheduler = StaticTileScheduler(
            "mega_",
            queue,
            smem_manager,
            debug=attrs.get("debug_scheduler", False),
            sm_count=attrs.get("sm_count", 1),
            num_threads=attrs.get("num_threads", 256),
            max_tasks=attrs.get("max_tasks", StaticTileScheduler.MAX_TASKS),
            end_job_id=attrs.get("end_job_id", 31),
        )
        scheduler.init()

        tile_dispatch = {tile_info.job_id: tile_info for tile_info in plan.tiles}
        with T.While(scheduler.valid()):
            idxs, job_id = scheduler.get_idx_and_task_type()
            self.emit_job_dispatch(
                plan, tile_dispatch, scheduler, smem_manager, event_bindings, job_id, *idxs
            )
            scheduler.next_tile()

    def emit_job_dispatch(
        self,
        plan: KernelLoweringPlan,
        tile_dispatch,
        scheduler,
        smem_manager,
        event_bindings,
        job_id,
        m_idx,
        n_idx,
        k_idx,
    ) -> None:
        dispatch_items = []
        if event_bindings:
            dispatch_items.extend([
                (INIT_EVENT_JOB_ID, "init_event"),
                (WAIT_EVENT_INIT_JOB_ID, "wait_event_init"),
            ])
        dispatch_items.extend((static_job_id, tile_info) for static_job_id, tile_info in sorted(tile_dispatch.items()))
        if not dispatch_items:
            return

        if_frames = [T.If(job_id == static_job_id) for static_job_id, _ in dispatch_items]
        then_frames = [T.Then() for _ in dispatch_items]
        else_frames = [T.Else() for _ in dispatch_items]

        for i, (_, item) in enumerate(dispatch_items):
            if_frames[i].__enter__()
            with then_frames[i]:
                if item == "init_event":
                    self.emit_init_event_task(plan, scheduler, event_bindings, m_idx, n_idx)
                elif item == "wait_event_init":
                    self.emit_wait_event_init_task(plan, scheduler)
                else:
                    self.emit_tile(item, scheduler, smem_manager, event_bindings, m_idx, n_idx, k_idx)
            else_frames[i].__enter__()

        T.evaluate(T.cuda.trap_when_assert_failed(False))

        for i in range(len(dispatch_items) - 1, -1, -1):
            else_frames[i].__exit__(None, None, None)
            if_frames[i].__exit__(None, None, None)

    def emit_tile(
        self,
        tile_info: TileLoweringInfo,
        scheduler: StaticTileScheduler,
        smem_manager: TIRXSmemManager,
        event_bindings: dict[str, EventBinding],
        m_idx,
        n_idx,
        k_idx,
    ) -> None:
        tile = tile_info.tile
        smem_manager.set_tile(tile)
        tile.impl.device_init(smem_manager, m_idx, n_idx, k_idx)
        emit_events(tile.waits, EVENT_WAIT_MARKER, scheduler, event_bindings, m_idx, n_idx, k_idx, self.options)
        tile.impl.prefetch(m_idx, n_idx, k_idx)
        tile.impl.run(m_idx, n_idx, k_idx)
        emit_events(tile.notifies, EVENT_NOTIFY_MARKER, scheduler, event_bindings, m_idx, n_idx, k_idx, self.options)

    def bind_tensor_buffers(self, plan: KernelLoweringPlan, buffers: list[Any]) -> None:
        if len(buffers) != len(plan.tensor_order):
            raise ValueError(
                f"Expected {len(plan.tensor_order)} tensor buffers, got {len(buffers)}"
            )
        if getattr(plan, "_tensor_attr_patches", None) is not None:
            return

        patches = []
        for tile_info in plan.tiles:
            impl = tile_info.tile.impl
            for attr_name, value in vars(impl).items():
                new_value = _replace_tensor_specs(value, plan.tensor_bindings)
                if new_value is not value:
                    patches.append((impl, attr_name, value))
                    setattr(impl, attr_name, new_value)
        setattr(plan, "_tensor_attr_patches", patches)

    def restore_tensor_buffers(self, plan: KernelLoweringPlan) -> None:
        patches = getattr(plan, "_tensor_attr_patches", None)
        if not patches:
            return
        for impl, attr_name, value in reversed(patches):
            setattr(impl, attr_name, value)
        delattr(plan, "_tensor_attr_patches")

    def create_smem_manager(self, plan: KernelLoweringPlan) -> TIRXSmemManager:
        smem_manager = TIRXSmemManager(plan.options.smem_max_bytes, plan.options.smem_chunk_size)
        plan.smem_manager = smem_manager
        return smem_manager

    def unique_tile_classes(self, plan: KernelLoweringPlan) -> list[TileLoweringInfo]:
        seen: set[Any] = set()
        result: list[TileLoweringInfo] = []
        for tile_info in plan.tiles:
            if tile_info.class_key in seen:
                continue
            seen.add(tile_info.class_key)
            result.append(tile_info)
        return result



class _StaticQueueInitEmitter:
    """Emit the static queue initialization kernel inside TIRX parser context."""

    def __init__(self, builder: "StaticQueueInitBuilder", plan: KernelLoweringPlan):
        self.builder = builder
        self.plan = plan

    def emit(self) -> None:
        self.builder.emit(self.plan)


@T.jit(check_well_formed=False)
def _static_queue_init_entry(*, emitter: T.constexpr):
    emitter.emit()


class StaticQueueInitBuilder:
    """Emit a static queue fill kernel matching ``StaticTileScheduler`` decoding."""

    def build(self, plan: KernelLoweringPlan) -> PrimFunc:
        return _static_queue_init_entry.specialize(emitter=_StaticQueueInitEmitter(self, plan))

    def emit(self, plan: KernelLoweringPlan) -> None:
        T.func_attr({"global_symbol": f"{plan.kernel.name}_init_queue"})
        attrs = plan.options.attrs
        sm_count = attrs.get("sm_count", 1)
        num_threads = attrs.get("num_threads", 256)
        max_tasks = attrs.get("max_tasks", StaticTileScheduler.MAX_TASKS)
        end_job_id = attrs.get("end_job_id", 31)

        _emit_local_symbolic_vars(plan)
        queue_handle = T.arg("queue", T.handle())
        queue = T.match_buffer(queue_handle, (sm_count, max_tasks), "int32")
        T.device_entry()

        bx = T.cta_id([sm_count])
        tid = T.thread_id([num_threads])
        idx = T.alloc_buffer((1,), "int32", scope="local")
        T.buffer_store(idx, 0, [0])

        phases = self._phases(plan, sm_count, end_job_id)
        for phase_id, phase in enumerate(phases):
            with T.If(bx == phase_id):
                with T.Then():
                    with T.If(tid == 0):
                        with T.Then():
                            self._emit_phase(queue, idx, phase, sm_count)
                with T.Else():
                    T.buffer_store(idx, idx[0] + phase["count"], [0])

    def _phases(self, plan: KernelLoweringPlan, sm_count: int, end_job_id: int) -> list[dict[str, Any]]:
        phases: list[dict[str, Any]] = []
        events = list(plan.kernel.events.values())
        if events:
            phases.append(
                {
                    "kind": "grid",
                    "job_id": INIT_EVENT_JOB_ID,
                    "tile_num": (len(events) + 1, 1, 1),
                    "count": len(events) + 1,
                }
            )

        entry_tiles = [tile_info for tile_info in plan.tiles if not tile_info.tile.waits]
        rest_tiles = [tile_info for tile_info in plan.tiles if tile_info.tile.waits]
        for tile_info in entry_tiles:
            phases.append(self._tile_phase(plan, tile_info))

        if events:
            phases.append(
                {
                    "kind": "grid",
                    "job_id": WAIT_EVENT_INIT_JOB_ID,
                    "tile_num": (sm_count, 1, 1),
                    "count": sm_count,
                }
            )

        for tile_info in rest_tiles:
            phases.append(self._tile_phase(plan, tile_info))

        phases.append(
            {
                "kind": "grid",
                "job_id": end_job_id,
                "tile_num": (sm_count, 1, 1),
                "count": sm_count,
            }
        )
        return phases

    def _tile_phase(self, plan: KernelLoweringPlan, tile_info: TileLoweringInfo) -> dict[str, Any]:
        tile_num = _shape_tuple(tile_info.tile.tile_num, f"tile {tile_info.tile.name} tile_num", plan)
        return {
            "kind": "grid",
            "job_id": tile_info.job_id,
            "tile_num": tile_num,
            "count": _shape_product(tile_num),
        }

    def _emit_phase(self, queue, idx, phase: dict[str, Any], sm_count: int) -> None:
        for_grid = T.grid(*phase["tile_num"])
        m_idx, n_idx, k_idx = for_grid.__enter__()
        packed = _pack_static_task(m_idx, n_idx, k_idx, phase["job_id"])
        T.buffer_store(queue, packed, [idx[0] % sm_count, idx[0] // sm_count])
        T.buffer_store(idx, idx[0] + 1, [0])
        for_grid.__exit__(None, None, None)


class MegakernelLowerer:
    """Top-level DSL lowering driver."""

    def __init__(self, options: LoweringOptions | None = None):
        self.options = options or LoweringOptions()
        self.kernel_builder = KernelBuilder(self.options)
        self.static_queue_init_builder = StaticQueueInitBuilder()

    def prepare(self, kernel: KernelSpec) -> KernelLoweringPlan:
        plan = KernelLoweringPlan(kernel=kernel, options=self.options)
        self._bind_vars(plan)
        self._bind_tensors(plan)
        self._bind_tiles(plan)
        return plan

    def _bind_vars(self, plan: KernelLoweringPlan) -> None:
        used_names: set[str] = set()
        for var in _collect_kernel_vars(plan.kernel):
            param_name = _sanitize_identifier(var.name, used_names)
            plan.var_order.append(var)
            plan.var_bindings[var] = VarBinding(var=var, param_name=param_name)

    def _bind_tensors(self, plan: KernelLoweringPlan) -> None:
        used_names: set[str] = set()
        for tensor in plan.kernel.tensors.values():
            param_name = _sanitize_identifier(tensor.name, used_names)
            plan.tensor_order.append(tensor)
            plan.tensor_bindings[tensor] = TensorBinding(tensor=tensor, param_name=param_name)

    def _bind_tiles(self, plan: KernelLoweringPlan) -> None:
        for job_id, tile in enumerate(plan.kernel.tiles):
            info = TileLoweringInfo(tile=tile, job_id=job_id, class_key=type(tile.impl))
            info.tensor_bindings = {
                tensor: plan.tensor_bindings[tensor]
                for tensor in [*tile.reads, *tile.writes]
                if tensor in plan.tensor_bindings
            }
            plan.tile_job_ids[tile.name] = job_id
            plan.tiles.append(info)

    def lower(self, kernel: KernelSpec) -> PrimFunc:
        return self.kernel_builder.build(self.prepare(kernel))

    def lower_static_queue_init(self, kernel: KernelSpec) -> PrimFunc:
        plan = self.prepare(kernel)
        if plan.options.schedule != "static":
            raise ValueError("static queue init is only available for static scheduling")
        return self.static_queue_init_builder.build(plan)

    def create_tile_infos(self, kernel: KernelSpec) -> list[TileLoweringInfo]:
        return self.prepare(kernel).tiles

    def create_fragments(self, kernel: KernelSpec) -> list[TileLoweringInfo]:
        """Compatibility alias for the previous scaffold name."""

        return self.create_tile_infos(kernel)


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


def _shape_tuple(shape: Any, context: str, plan: KernelLoweringPlan | None = None) -> tuple[Any, ...]:
    if isinstance(shape, (tuple, list)):
        return tuple(_lower_expr_like(dim, context, plan) for dim in shape)
    return (_lower_expr_like(shape, context, plan),)


def _shape_product(shape: tuple[Any, ...]) -> Any:
    result = 1
    for extent in shape:
        result *= extent
    return result


def _replace_tensor_specs(value: Any, bindings: dict[TensorSpec, TensorBinding]) -> Any:
    if isinstance(value, TensorSpec) and value in bindings:
        return bindings[value].buffer
    if isinstance(value, tuple):
        return tuple(_replace_tensor_specs(item, bindings) for item in value)
    if isinstance(value, list):
        return [_replace_tensor_specs(item, bindings) for item in value]
    if isinstance(value, dict):
        return {
            _replace_tensor_specs(key, bindings): _replace_tensor_specs(val, bindings)
            for key, val in value.items()
        }
    return value


def _emit_local_symbolic_vars(plan: KernelLoweringPlan) -> None:
    for var in plan.var_order:
        binding = plan.var_bindings[var]
        binding.value = T.Var(binding.param_name, getattr(var, "dtype", "int32"))


def _pack_static_task(m_idx, n_idx, k_idx, job_id: int):
    return T.bitwise_or(
        T.bitwise_or(job_id, T.shift_left(m_idx, 5)),
        T.bitwise_or(T.shift_left(n_idx, 18), T.shift_left(k_idx, 28)),
    )


def _collect_kernel_vars(kernel: KernelSpec) -> list[VarSpec]:
    seen: set[VarSpec] = set()
    result: list[VarSpec] = []

    def add_from(value: Any) -> None:
        if isinstance(value, VarSpec):
            if value not in seen:
                seen.add(value)
                result.append(value)
        elif isinstance(value, (tuple, list)):
            for item in value:
                add_from(item)

    for var in getattr(kernel, "vars", {}).values():
        add_from(var)
    for tensor in kernel.tensors.values():
        add_from(tensor.shape)
    for event in kernel.events.values():
        add_from(event.shape)
    for tile in kernel.tiles:
        add_from(tile.tile_num)
    return result


def _lower_expr_like(value: Any, context: str, plan: KernelLoweringPlan | None = None) -> Any:
    if isinstance(value, int):
        return value
    if isinstance(value, VarSpec):
        if plan is None or value not in plan.var_bindings or plan.var_bindings[value].value is None:
            raise ValueError(f"{context} uses unbound symbolic VarSpec({value.name!r})")
        return plan.var_bindings[value].value
    raise TypeError(f"{context} must be an int or VarSpec, got {value!r}")


def lower_to_tirx(kernel: KernelSpec, options: LoweringOptions | None = None) -> PrimFunc:
    """Lower a ``KernelSpec`` to a TIRX PrimFunc."""

    return MegakernelLowerer(options).lower(kernel)


def lower_static_queue_init_to_tirx(
    kernel: KernelSpec, options: LoweringOptions | None = None
) -> PrimFunc:
    """Lower static queue initialization for a ``KernelSpec`` to a TIRX PrimFunc."""

    return MegakernelLowerer(options).lower_static_queue_init(kernel)


def lower_to_tirx_module(
    kernel: KernelSpec, options: LoweringOptions | None = None
) -> IRModule:
    """Lower a ``KernelSpec`` to a TIRX module with required helper kernels."""

    lowerer = MegakernelLowerer(options)
    funcs = {kernel.name: lowerer.lower(kernel)}
    if lowerer.options.schedule == "static":
        funcs[f"{kernel.name}_init_queue"] = lowerer.lower_static_queue_init(kernel)
    return IRModule(funcs)


@tvm.transform.module_pass(opt_level=0, name="LowerMegakernelDSL")
class LowerMegakernelDSL:
    """Module pass wrapper for experiments that want a TVM pass object."""

    def __init__(self, kernel: KernelSpec, options: LoweringOptions | None = None):
        self.kernel = kernel
        self.options = options or LoweringOptions()

    def transform_module(self, mod: IRModule, _ctx: tvm.transform.PassContext) -> IRModule:
        lowered = lower_to_tirx_module(self.kernel, self.options)
        return IRModule({**mod.functions, **lowered.functions}, attrs=mod.attrs)
