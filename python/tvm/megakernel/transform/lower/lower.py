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

The DSL records a logical tile graph.  Lowering first prepares that graph into
physical tile, event, and queue metadata, then emits one persistent TIRX kernel
plus any schedule-specific queue initialization functions.

The build owns one ``SmemManager``.  Tile hooks run while the final kernel is
being built, so user parser-style ``TileImpl`` code emits directly into the
current TIRX builder scope.
"""

from __future__ import annotations

from typing import Any
from tvm.ir.module import IRModule
import tvm.tirx.script as T
from tvm.tirx import PrimFunc

from ...dsl.spec import KernelSpec
from ..validate import validate_impl, validate_kernel, validate_lowering_plan
from .prepare import (
    INIT_EVENT_JOB_ID,
    WAIT_EVENT_INIT_JOB_ID,
    KernelLoweringPlan,
    LoweringOptions,
    TaskPhase,
    TileLoweringInfo,
    TilePlan,
)
from .prepare import (
    plan_event_workspace_size,
    prepare_lowering_plan,
    replace_dsl_refs,
    replace_tensor_specs,
    shape_product as _shape_product,
    shape_tuple as _shape_tuple,
)
from .event import (
    EventBinding,
    EventLoweringMixin,
    coord_from_map,
    dependency_info_from_map,
    event_init_count,
    event_shape_tuple,
    linear_index_to_coord,
)
from .scheduler import DynamicTileScheduler, DynamicTIRXSemaphore, StaticTileScheduler, StaticTIRXSemaphore
from .smem import TIRXSmemManager


def lower(kernel: KernelSpec, options: LoweringOptions | None = None) -> IRModule:
    """Lower a ``KernelSpec`` to a TIRX module."""

    return _LoweringPipeline(options).lower(kernel)


class _LoweringPipeline:
    """Validate, prepare, and emit one lowered megakernel module."""

    def __init__(self, options: LoweringOptions | None = None):
        self.options = options or LoweringOptions()

    def lower(self, kernel: KernelSpec) -> IRModule:
        return self.emit_module(self.prepare_plan(kernel))

    def prepare_plan(self, kernel: KernelSpec) -> KernelLoweringPlan:
        validate_kernel(kernel)
        validate_impl(kernel)
        plan = prepare_lowering_plan(kernel, self.options)
        validate_lowering_plan(plan)
        return plan

    def emit_module(self, plan: KernelLoweringPlan) -> IRModule:
        funcs = {plan.kernel.name: self.emit_kernel(plan)}
        if plan.options.schedule in ("static", "dynamic"):
            funcs[f"{plan.kernel.name}_init_queue"] = self.emit_queue_init(plan)
        return IRModule(funcs)

    def emit_kernel(self, plan: KernelLoweringPlan) -> PrimFunc:
        return KernelBuilder(plan.options).build(plan)

    def emit_queue_init(self, plan: KernelLoweringPlan) -> PrimFunc:
        if plan.options.schedule == "static":
            return StaticQueueInitBuilder().build(plan)
        if plan.options.schedule == "dynamic":
            return DynamicQueueInitBuilder().build(plan)
        raise ValueError("queue init is only available for static or dynamic scheduling")


class _ParserKernelEmitter:
    """Small constexpr object used to emit the kernel inside TIRX parser context."""

    def __init__(self, builder: "KernelBuilder", plan: KernelLoweringPlan):
        self.builder = builder
        self.plan = plan

    def emit(self) -> None:
        self.builder.emit_kernel(self.plan)


class _KernelEntry:
    """TIRX parser entry for the main megakernel body."""

    @staticmethod
    @T.jit(check_well_formed=False)
    def emit(*, emitter: T.constexpr):
        emitter.emit()


class KernelBuilder(EventLoweringMixin):
    """Directly emit one TIRX megakernel from a prepared plan."""

    def __init__(self, options: LoweringOptions):
        self.options = options

    def build(self, plan: KernelLoweringPlan) -> PrimFunc:
        emitter = _ParserKernelEmitter(self, plan)
        try:
            return _KernelEntry.emit.specialize(emitter=emitter)
        finally:
            self.restore_impl_attrs(plan)

    def emit_kernel(self, plan: KernelLoweringPlan) -> None:
        T.func_attr({"global_symbol": plan.kernel.name})
        _LoweringUtils.emit_local_symbolic_vars(plan)
        self.bind_impl_attrs(plan, self.emit_tensor_args(plan))
        event_workspace = self.emit_event_workspace_arg(plan)
        queue = self.emit_static_queue_arg(plan)
        dynamic_queue = self.emit_dynamic_queue_args(plan)
        self.emit_profiler_arg(plan)
        T.device_entry()
        self.emit_scope_ids(plan.options.attrs)

        smem_manager = self.create_smem_manager(plan)
        event_bindings = self.bind_event_buffers(plan, event_workspace)
        smem_manager.init()

        tile_classes = self.unique_tile_classes(plan)
        for tile_info in tile_classes:
            type(tile_info.tile.impl).init_shared_resources(smem_manager)

        tile_dispatch = self.tile_dispatch(plan)
        if plan.options.schedule == "static":
            self.emit_static_queue(plan, tile_dispatch, queue, smem_manager, event_bindings)
        elif plan.options.schedule == "dynamic":
            self.emit_dynamic_queue(
                plan, tile_dispatch, dynamic_queue, smem_manager, event_bindings
            )
        else:
            raise ValueError(f"Unknown megakernel schedule: {plan.options.schedule!r}")

        for tile_info in reversed(tile_classes):
            type(tile_info.tile.impl).finalize_shared_resources(smem_manager)
        smem_manager.commit()

    def emit_tensor_args(self, plan: KernelLoweringPlan) -> list[Any]:
        buffers = []
        for tensor in plan.tensor_order:
            binding = plan.tensor_bindings[tensor]
            shape = _shape_tuple(tensor.shape, f"tensor {tensor.name} shape", plan)
            binding.buffer = T.arg(binding.param_name, T.Buffer(shape, tensor.dtype))
            buffers.append(binding.buffer)
        return buffers

    def emit_event_workspace_arg(self, plan: KernelLoweringPlan):
        if not plan.event_layouts:
            return None
        return T.arg("event_workspace", T.Buffer((plan_event_workspace_size(plan),), "int32"))

    def bind_event_buffers(
        self, plan: KernelLoweringPlan, event_workspace
    ) -> dict[str, EventBinding]:
        if not plan.event_layouts:
            return {}
        if event_workspace is None:
            raise ValueError("event lowering requires an event workspace argument")

        for event_plan in plan.event_layouts:
            shape = _shape_tuple(
                event_plan.shape, f"event {event_plan.name} shape", plan
            )
            buffer = T.decl_buffer(
                shape,
                event_plan.dtype,
                data=event_workspace.data,
                elem_offset=event_plan.workspace_offset,
                scope="global",
            )
            size = _shape_product(shape)
            plan.event_bindings[event_plan.name] = EventBinding(
                event=event_plan.event,
                buffer=buffer,
                size=size,
            )

        if plan.event_init_complete_layout is not None:
            plan.event_init_complete = EventBinding(
                event=None,
                buffer=event_workspace,
                size=1,
            )
        return plan.event_bindings

    def emit_static_queue_arg(self, plan: KernelLoweringPlan):
        if plan.options.schedule != "static":
            return None
        attrs = plan.options.attrs
        return T.arg(
            "queue",
            T.Buffer((attrs.get("sm_count", 1), attrs.get("max_tasks", 128)), "int32"),
        )

    def emit_dynamic_queue_args(self, plan: KernelLoweringPlan):
        if plan.options.schedule != "dynamic":
            return None
        if plan.dynamic_schedule is None:
            raise ValueError("dynamic schedule requires a dynamic schedule plan")
        tasks = T.arg("tasks", T.Buffer((plan.dynamic_schedule.max_tasks,), "int32"))
        head = T.arg("head", T.Buffer((1,), "int32"))
        tail = T.arg("tail", T.Buffer((1,), "int32"))
        return tasks, head, tail

    def emit_profiler_arg(self, plan: KernelLoweringPlan):
        profiler_buf_size = plan.options.attrs.get("profiler_buf_size")
        if profiler_buf_size is None:
            return None
        return T.arg("profiler_buf", T.Buffer((profiler_buf_size,), "uint64"))

    def emit_scope_ids(self, attrs: dict[str, Any]) -> dict[str, Any]:
        num_threads = attrs.get("num_threads", 256)
        warp_count = attrs.get("warp_count", max(1, num_threads // 32))
        scope_ids: dict[str, Any] = {
            "cta_id": T.cta_id([attrs.get("sm_count", 1)]),
            "thread_id": T.thread_id([num_threads]),
            "warp_id": T.warp_id([warp_count]),
            "lane_id": T.lane_id([32]),
        }
        if "warpgroup_count" in attrs or "warpgroup_size" in attrs:
            warpgroup_size = attrs.get("warpgroup_size", 128)
            scope_ids["warpgroup_id"] = T.warpgroup_id(
                [attrs.get("warpgroup_count", max(1, num_threads // warpgroup_size))]
            )
            scope_ids["thread_id_in_wg"] = T.thread_id_in_wg([warpgroup_size])
        return scope_ids

    def tile_dispatch(self, plan: KernelLoweringPlan) -> dict[int, TilePlan]:
        return {tile_plan.job_id: tile_plan for tile_plan in plan.tile_plans}

    def emit_static_queue(
        self,
        plan: KernelLoweringPlan,
        tile_dispatch: dict[int, TilePlan],
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

        with T.While(scheduler.valid()):
            idxs, job_id = scheduler.get_idx_and_task_type()
            self.emit_job_dispatch(
                plan, tile_dispatch, scheduler, smem_manager, event_bindings, job_id, *idxs
            )
            scheduler.next_tile()


    def emit_dynamic_queue(
        self,
        plan: KernelLoweringPlan,
        tile_dispatch: dict[int, TilePlan],
        dynamic_queue,
        smem_manager: TIRXSmemManager,
        event_bindings: dict[str, EventBinding],
    ) -> None:
        if dynamic_queue is None:
            raise ValueError("dynamic scheduling requires tasks/head/tail arguments")
        if plan.dynamic_schedule is None:
            raise ValueError("dynamic scheduling requires a dynamic schedule plan")
        tasks, head, tail = dynamic_queue
        attrs = plan.options.attrs
        scheduler = DynamicTileScheduler(
            tasks,
            head,
            tail,
            smem_manager,
            debug=attrs.get("debug_scheduler", False),
            num_threads=attrs.get("num_threads", 256),
            max_tasks=plan.dynamic_schedule.max_tasks,
            end_job_id=plan.dynamic_schedule.end_job_id,
            warp_count=attrs.get("warp_count"),
            warpgroup_count=attrs.get("warpgroup_count"),
            warpgroup_size=attrs.get("warpgroup_size", 128),
            scheduler_warp=attrs.get("scheduler_warp", 7),
        )
        scheduler.init()
        scheduler.next_tile()
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
        if event_bindings and plan.options.schedule == "static":
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
                    self.emit_tile(plan, item, scheduler, smem_manager, event_bindings, m_idx, n_idx, k_idx)
            else_frames[i].__enter__()

        T.evaluate(T.cuda.trap_when_assert_failed(False))

        for i in range(len(dispatch_items) - 1, -1, -1):
            else_frames[i].__exit__(None, None, None)
            if_frames[i].__exit__(None, None, None)

    def emit_tile(
        self,
        plan: KernelLoweringPlan,
        tile_plan: TilePlan,
        scheduler: StaticTileScheduler | DynamicTileScheduler,
        smem_manager: TIRXSmemManager,
        event_bindings: dict[str, EventBinding],
        m_idx,
        n_idx,
        k_idx,
    ) -> None:
        if isinstance(scheduler, DynamicTileScheduler):
            self.emit_dynamic_tile(
                plan, tile_plan, scheduler, smem_manager, event_bindings, m_idx, n_idx, k_idx
            )
        else:
            self.emit_static_tile(
                plan, tile_plan, scheduler, smem_manager, event_bindings, m_idx, n_idx, k_idx
            )

    def emit_static_tile(
        self,
        plan: KernelLoweringPlan,
        tile_plan: TilePlan,
        scheduler: StaticTileScheduler,
        smem_manager: TIRXSmemManager,
        event_bindings: dict[str, EventBinding],
        m_idx,
        n_idx,
        k_idx,
    ) -> None:
        tile = tile_plan.tile
        smem_manager.set_tile(tile)
        patches = self.patch_impl_expr_attrs(plan, tile.impl, expr_mode="upper_bound")
        try:
            tile.impl.device_init(smem_manager, m_idx, n_idx, k_idx)
        finally:
            self.restore_impl_expr_attrs(tile.impl, patches)
        patches = self.patch_impl_expr_attrs(plan, tile.impl, expr_mode="runtime")
        try:
            tile.impl.prefetch(m_idx, n_idx, k_idx)
            _TileEventEmitter.emit_static_waits(
                tile_plan.waits, scheduler, event_bindings, m_idx, n_idx, k_idx, plan.tensor_bindings, tile.attrs
            )
            tile.impl.run(m_idx, n_idx, k_idx)
            smem_manager.validate_tile_phase(tile)
            _TileEventEmitter.emit_static_notifies(
                tile_plan.notifies, scheduler, event_bindings, m_idx, n_idx, k_idx, plan.tensor_bindings, tile.attrs
            )
        finally:
            self.restore_impl_expr_attrs(tile.impl, patches)

    def emit_dynamic_tile(
        self,
        plan: KernelLoweringPlan,
        tile_plan: TilePlan,
        scheduler: DynamicTileScheduler,
        smem_manager: TIRXSmemManager,
        event_bindings: dict[str, EventBinding],
        m_idx,
        n_idx,
        k_idx,
    ) -> None:
        tile = tile_plan.tile
        smem_manager.set_tile(tile)
        patches = self.patch_impl_expr_attrs(plan, tile.impl, expr_mode="upper_bound")
        try:
            tile.impl.device_init(smem_manager, m_idx, n_idx, k_idx)
        finally:
            self.restore_impl_expr_attrs(tile.impl, patches)
        patches = self.patch_impl_expr_attrs(plan, tile.impl, expr_mode="runtime")
        try:
            tile.impl.prefetch(m_idx, n_idx, k_idx)
            _TileEventEmitter.emit_dynamic_pre_notify_and_pushes(
                tile_plan.notifies, scheduler, event_bindings, m_idx, n_idx, k_idx, plan, tile_plan
            )
            _TileEventEmitter.emit_dynamic_waits(
                tile_plan.waits, scheduler, event_bindings, m_idx, n_idx, k_idx, plan.tensor_bindings, tile.attrs
            )
            tile.impl.run(m_idx, n_idx, k_idx)
            smem_manager.validate_tile_phase(tile)
            _TileEventEmitter.emit_dynamic_complete_notifies(
                tile_plan.notifies, scheduler, event_bindings, m_idx, n_idx, k_idx, plan.tensor_bindings, tile.attrs
            )
            _TileEventEmitter.emit_dynamic_endpoint_end_tasks(scheduler, plan, tile_plan, m_idx, n_idx, k_idx)
        finally:
            self.restore_impl_expr_attrs(tile.impl, patches)

    def bind_impl_attrs(self, plan: KernelLoweringPlan, buffers: list[Any]) -> None:
        if len(buffers) != len(plan.tensor_order):
            raise ValueError(
                f"Expected {len(plan.tensor_order)} tensor buffers, got {len(buffers)}"
            )
        if getattr(plan, "_impl_attr_patches", None) is not None:
            return

        patches = []
        for tile_info in plan.tiles:
            impl = tile_info.tile.impl
            for attr_name, value in vars(impl).items():
                new_value = replace_tensor_specs(value, plan.tensor_bindings)
                if new_value is not value:
                    patches.append((impl, attr_name, value))
                    setattr(impl, attr_name, new_value)
        setattr(plan, "_impl_attr_patches", patches)

    def restore_impl_attrs(self, plan: KernelLoweringPlan) -> None:
        patches = getattr(plan, "_impl_attr_patches", None)
        if not patches:
            return
        for impl, attr_name, value in reversed(patches):
            setattr(impl, attr_name, value)
        delattr(plan, "_impl_attr_patches")

    def patch_impl_expr_attrs(
        self, plan: KernelLoweringPlan, impl, *, expr_mode: str
    ) -> list[tuple[str, Any]]:
        patches = []
        for attr_name, value in vars(impl).items():
            new_value = replace_dsl_refs(
                value, plan.tensor_bindings, plan, expr_mode=expr_mode
            )
            if new_value is not value:
                patches.append((attr_name, value))
                setattr(impl, attr_name, new_value)
        return patches

    def restore_impl_expr_attrs(self, impl, patches: list[tuple[str, Any]]) -> None:
        for attr_name, value in reversed(patches):
            setattr(impl, attr_name, value)

    def create_smem_manager(self, plan: KernelLoweringPlan) -> TIRXSmemManager:
        num_threads = plan.options.attrs.get("num_threads", 256)
        smem_manager = TIRXSmemManager(
            plan.options.smem_max_bytes,
            plan.options.smem_chunk_size,
            num_threads=num_threads,
            warp_count=max(1, num_threads // 32),
        )
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


class _StaticQueueInitEntry:
    """TIRX parser entry for static queue initialization."""

    @staticmethod
    @T.jit(check_well_formed=False)
    def emit(*, emitter: T.constexpr):
        emitter.emit()


class StaticQueueInitBuilder:
    """Emit a static queue fill kernel matching ``StaticTileScheduler`` decoding."""

    def build(self, plan: KernelLoweringPlan) -> PrimFunc:
        return _StaticQueueInitEntry.emit.specialize(emitter=_StaticQueueInitEmitter(self, plan))

    def emit(self, plan: KernelLoweringPlan) -> None:
        T.func_attr({"global_symbol": f"{plan.kernel.name}_init_queue"})
        attrs = plan.options.attrs
        sm_count = attrs.get("sm_count", 1)
        num_threads = attrs.get("num_threads", 256)
        max_tasks = attrs.get("max_tasks", StaticTileScheduler.MAX_TASKS)
        _LoweringUtils.emit_symbolic_var_args(plan)
        queue_handle = T.arg("queue", T.handle())
        queue = T.match_buffer(queue_handle, (sm_count, max_tasks), "int32")
        T.device_entry()

        bx = T.cta_id([sm_count])
        tid = T.thread_id([num_threads])
        idx = T.alloc_buffer((1,), "int32", scope="local")
        T.buffer_store(idx, 0, [0])

        if plan.static_schedule is None:
            raise ValueError("static queue init requires a static schedule plan")
        for phase_id, phase in enumerate(plan.static_schedule.phases):
            phase_count = self._phase_count(plan, phase)
            with T.If(bx == phase_id):
                with T.Then():
                    with T.If(tid == 0):
                        with T.Then():
                            self._emit_phase(plan, queue, idx, phase, sm_count)
                with T.Else():
                    T.buffer_store(idx, idx[0] + phase_count, [0])

    def _phase_count(self, plan: KernelLoweringPlan, phase: TaskPhase) -> Any:
        grid = _shape_tuple(phase.grid, f"phase {phase.label} grid", plan)
        return _shape_product(grid)

    def _emit_phase(
        self, plan: KernelLoweringPlan, queue, idx, phase: TaskPhase, sm_count: int
    ) -> None:
        grid = _shape_tuple(phase.grid, f"phase {phase.label} grid", plan)
        for_grid = T.grid(*grid)
        m_idx, n_idx, k_idx = for_grid.__enter__()
        packed = _LoweringUtils.pack_task(m_idx, n_idx, k_idx, phase.job_id)
        T.buffer_store(queue, packed, [idx[0] % sm_count, idx[0] // sm_count])
        T.buffer_store(idx, idx[0] + 1, [0])
        for_grid.__exit__(None, None, None)


class DynamicQueueInitBuilder:
    """Emit dynamic queue initialization and event workspace initialization."""

    def build(self, plan: KernelLoweringPlan) -> PrimFunc:
        return _DynamicQueueInitEntry.emit.specialize(emitter=_DynamicQueueInitEmitter(self, plan))

    def emit(self, plan: KernelLoweringPlan) -> None:
        if plan.dynamic_schedule is None:
            raise ValueError("dynamic queue init requires a dynamic schedule plan")
        T.func_attr({"global_symbol": f"{plan.kernel.name}_init_queue"})
        _LoweringUtils.emit_symbolic_var_args(plan)
        event_workspace = self.emit_event_workspace_arg(plan)
        tasks = T.arg("tasks", T.Buffer((plan.dynamic_schedule.max_tasks,), "int32"))
        head = T.arg("head", T.Buffer((1,), "int32"))
        tail = T.arg("tail", T.Buffer((1,), "int32"))
        T.device_entry()
        self.emit_event_workspace_init(plan, event_workspace)
        self.emit_task_queue_init(plan, tasks, head, tail)

    def emit_event_workspace_arg(self, plan: KernelLoweringPlan):
        if not plan.event_layouts:
            return None
        return T.arg("event_workspace", T.Buffer((plan_event_workspace_size(plan),), "int32"))

    def emit_event_workspace_init(self, plan: KernelLoweringPlan, event_workspace) -> None:
        if not plan.event_layouts:
            return
        if event_workspace is None:
            raise ValueError("dynamic event init requires an event workspace argument")
        attrs = plan.options.attrs
        num_threads = attrs.get("num_threads", 256)
        tid = T.thread_id([num_threads])
        bx = T.cta_id([attrs.get("sm_count", 1)])
        for static_event_id, event_layout in enumerate(plan.event_layouts):
            event = event_layout.event
            with T.If(bx == static_event_id % attrs.get("sm_count", 1)):
                with T.Then():
                    idx = T.alloc_buffer((1,), "int32", scope="local")
                    T.buffer_store(idx, tid, [0])
                    shape = event_shape_tuple(event.shape, f"event {event.name} shape", plan)
                    with T.While(idx[0] < event_layout.size):
                        coord = linear_index_to_coord(idx[0], shape)
                        init_count = event_init_count(event, coord, plan)
                        T.buffer_store(event_workspace, init_count * (StaticTIRXSemaphore.base + 1), [event_layout.workspace_offset + idx[0]])
                        T.buffer_store(idx, idx[0] + num_threads, [0])

    def emit_task_queue_init(self, plan: KernelLoweringPlan, tasks, head, tail) -> None:
        attrs = plan.options.attrs
        sm_count = attrs.get("sm_count", 1)
        num_threads = attrs.get("num_threads", 256)
        bx = T.cta_id([sm_count])
        tx = T.thread_id([num_threads])
        total_entry_tasks = sum(
            _shape_product(_shape_tuple(phase.grid, f"phase {phase.label} grid", plan))
            for phase in plan.dynamic_schedule.entry_phases
        )
        idx = T.alloc_buffer((1,), "int32", scope="local")
        T.buffer_store(idx, 0, [0])
        for i, phase in enumerate(plan.dynamic_schedule.entry_phases):
            with T.If(bx == i):
                with T.Then():
                    with T.If(tx == 0):
                        with T.Then():
                            grid = _shape_tuple(phase.grid, f"phase {phase.label} grid", plan)
                            for_grid = T.grid(*grid)
                            m_idx, n_idx, k_idx = for_grid.__enter__()
                            T.buffer_store(tasks, _LoweringUtils.pack_task(m_idx, n_idx, k_idx, phase.job_id), [idx[0]])
                            T.buffer_store(idx, idx[0] + 1, [0])
                            for_grid.__exit__(None, None, None)
                with T.Else():
                    grid = _shape_tuple(phase.grid, f"phase {phase.label} grid", plan)
                    T.buffer_store(idx, idx[0] + _shape_product(grid), [0])
        clear_idx = T.alloc_buffer((1,), "int32", scope="local")
        T.buffer_store(clear_idx, total_entry_tasks + bx * num_threads + tx, [0])
        with T.While(clear_idx[0] < plan.dynamic_schedule.max_tasks):
            T.buffer_store(tasks, -1, [clear_idx[0]])
            T.buffer_store(clear_idx, clear_idx[0] + sm_count * num_threads, [0])
        with T.If((tx == 0) & (bx == 0)):
            with T.Then():
                T.buffer_store(head, 0, [0])
                T.buffer_store(tail, total_entry_tasks, [0])


class _DynamicQueueInitEmitter:
    def __init__(self, builder: "DynamicQueueInitBuilder", plan: KernelLoweringPlan):
        self.builder = builder
        self.plan = plan

    def emit(self) -> None:
        self.builder.emit(self.plan)


class _DynamicQueueInitEntry:
    """TIRX parser entry for dynamic queue initialization."""

    @staticmethod
    @T.jit(check_well_formed=False)
    def emit(*, emitter: T.constexpr):
        emitter.emit()


class _TileEventEmitter:
    """Emit tile event waits, notifications, dynamic pushes, and end tasks."""

    @staticmethod
    def emit_static_waits(
        waits,
        scheduler: StaticTileScheduler,
        event_bindings: dict[str, EventBinding],
        m_idx,
        n_idx,
        k_idx,
        tensor_bindings,
        attrs: dict[str, Any] | None = None,
    ) -> None:
        for dependency in waits:
            event = dependency.event
            coord = coord_from_map(dependency, m_idx, n_idx, k_idx, tensor_bindings=tensor_bindings)
            if event.name not in event_bindings:
                raise ValueError(f"event {event.name!r} has no lowering buffer for wait")
            attrs = attrs or {}
            scheduler.wait(
                StaticTIRXSemaphore(event_bindings[event.name].buffer),
                *coord,
                wait_level=attrs.get("wait_scope", "cta"),
                mask=attrs.get("wait_mask", 0xFFFFFFFF),
            )


    @staticmethod
    def emit_static_notifies(
        notifies,
        scheduler: StaticTileScheduler,
        event_bindings: dict[str, EventBinding],
        m_idx,
        n_idx,
        k_idx,
        tensor_bindings,
        attrs: dict[str, Any] | None = None,
    ) -> None:
        for dependency in notifies:
            event = dependency.event
            if event.name not in event_bindings:
                raise ValueError(f"event {event.name!r} has no lowering buffer for notify")
            semaphore = StaticTIRXSemaphore(event_bindings[event.name].buffer)

            def notify_func(notify_i, dependency=dependency):
                return dependency_info_from_map(
                    dependency, m_idx, n_idx, k_idx, notify_i, tensor_bindings=tensor_bindings
                )

            attrs = attrs or {}
            scheduler.notify(
                semaphore,
                notify_func,
                scope=attrs.get("notify_scope", "cta"),
                scope_id=attrs.get("notify_scope_id", 0),
            )


    @staticmethod
    def emit_dynamic_waits(
        waits,
        scheduler: DynamicTileScheduler,
        event_bindings: dict[str, EventBinding],
        m_idx,
        n_idx,
        k_idx,
        tensor_bindings,
        attrs: dict[str, Any] | None = None,
    ) -> None:
        for dependency in waits:
            event = dependency.event
            coord = coord_from_map(dependency, m_idx, n_idx, k_idx, tensor_bindings=tensor_bindings)
            if event.name not in event_bindings:
                raise ValueError(f"event {event.name!r} has no lowering buffer for wait")
            attrs = attrs or {}
            scheduler.wait(
                DynamicTIRXSemaphore(event_bindings[event.name].buffer),
                *coord,
                wait_level=attrs.get("wait_scope", "cta"),
                mask=attrs.get("wait_mask", 0xFFFFFFFF),
            )


    @staticmethod
    def emit_dynamic_pre_notify_and_pushes(
        notifies,
        scheduler: DynamicTileScheduler,
        event_bindings: dict[str, EventBinding],
        m_idx,
        n_idx,
        k_idx,
        plan: KernelLoweringPlan,
        tile_plan: TilePlan,
    ) -> None:
        if plan.dynamic_schedule is None:
            return
        for dependency in notifies:
            event = dependency.event
            if event.name not in event_bindings:
                raise ValueError(f"event {event.name!r} has no lowering buffer for notify")
            triggers = [
                trigger
                for trigger in plan.dynamic_schedule.triggers.get(tile_plan.tile.name, ())
                if trigger.event is event
            ]
            if not triggers:
                continue
            semaphore = DynamicTIRXSemaphore(event_bindings[event.name].buffer)

            def notify_func(notify_i, dependency=dependency):
                return dependency_info_from_map(
                    dependency, m_idx, n_idx, k_idx, notify_i, tensor_bindings=plan.tensor_bindings
                )

            attrs = tile_plan.tile.attrs
            for trigger in triggers:

                def push_func(notify_i, consumer_i, trigger=trigger, dependency=dependency):
                    notify_info = dependency_info_from_map(
                        dependency,
                        m_idx,
                        n_idx,
                        k_idx,
                        notify_i,
                        tensor_bindings=plan.tensor_bindings,
                    )
                    rank = notify_info[1]
                    event_coord = notify_info[2:]
                    consumer_num, consumer_m_idx, consumer_n_idx, consumer_k_idx = (
                        _LoweringUtils.consumer_task_info_from_event(
                            trigger, rank, event_coord, consumer_i
                        )
                    )
                    return (
                        consumer_num,
                        _LoweringUtils.pack_task(
                            consumer_m_idx,
                            consumer_n_idx,
                            consumer_k_idx,
                            trigger.consumer.job_id,
                        ),
                    )

                scheduler.pre_notify_and_push(
                    semaphore,
                    notify_func,
                    push_func,
                    push_level=attrs.get("push_level", "cta"),
                    scope=attrs.get("push_scope", attrs.get("notify_scope", "cta")),
                    scope_id=attrs.get("push_scope_id", attrs.get("notify_scope_id", 0)),
                )


    @staticmethod
    def emit_dynamic_complete_notifies(
        notifies,
        scheduler: DynamicTileScheduler,
        event_bindings: dict[str, EventBinding],
        m_idx,
        n_idx,
        k_idx,
        tensor_bindings,
        attrs: dict[str, Any] | None = None,
    ) -> None:
        for dependency in notifies:
            event = dependency.event
            coord = coord_from_map(dependency, m_idx, n_idx, k_idx, tensor_bindings=tensor_bindings)
            if event.name not in event_bindings:
                raise ValueError(f"event {event.name!r} has no lowering buffer for notify")
            semaphore = DynamicTIRXSemaphore(event_bindings[event.name].buffer)

            def notify_func(notify_i, dependency=dependency):
                return dependency_info_from_map(
                    dependency, m_idx, n_idx, k_idx, notify_i, tensor_bindings=tensor_bindings
                )

            attrs = attrs or {}
            scheduler.complete_notify(
                semaphore,
                notify_func,
                scope=attrs.get("notify_scope", "cta"),
                scope_id=attrs.get("notify_scope_id", 0),
            )


    @staticmethod
    def emit_dynamic_endpoint_end_tasks(scheduler, plan, tile_plan, _m_idx, _n_idx, _k_idx) -> None:
        if not isinstance(scheduler, DynamicTileScheduler) or plan is None or tile_plan is None:
            return
        if plan.dynamic_schedule is None or plan.dynamic_schedule.endpoint is not tile_plan:
            return
        sm_count = plan.options.attrs.get("sm_count", 1)
        end_loop = T.serial(0, sm_count)
        end_loop.__enter__()
        scheduler.enqueue(_LoweringUtils.pack_task(0, 0, 0, plan.dynamic_schedule.end_job_id), push_level="cta")
        end_loop.__exit__(None, None, None)

class _LoweringUtils:
    """Small pure helpers shared by kernel and queue builders."""

    @staticmethod
    def consumer_task_info_from_event(trigger, rank, event_coord, consumer_i):
        inv_coord = trigger.consumer_inv_coord
        if inv_coord is None:
            raise ValueError("dynamic trigger requires inv_coord")
        info = inv_coord(rank, *event_coord, consumer_i)
        if not isinstance(info, (tuple, list)) or len(info) != 4:
            raise TypeError("inv_coord must produce (consumer_num, tile_m, tile_n, tile_k)")
        return tuple(info)


    @staticmethod
    def consumer_task_coord_from_event(trigger, rank, event_coord, consumer_i):
        return _LoweringUtils.consumer_task_info_from_event(trigger, rank, event_coord, consumer_i)[1:]


    @staticmethod
    def emit_local_symbolic_vars(plan: KernelLoweringPlan) -> None:
        for var in plan.var_order:
            binding = plan.var_bindings[var]
            binding.value = T.Var(binding.param_name, getattr(var, "dtype", "int32"))


    @staticmethod
    def emit_symbolic_var_args(plan: KernelLoweringPlan) -> None:
        for var in plan.var_order:
            binding = plan.var_bindings[var]
            dtype = getattr(var, "dtype", "int32")
            if dtype != "int32":
                raise TypeError("queue init currently supports only int32 symbolic vars")
            binding.value = T.arg(binding.param_name, T.int32())


    @staticmethod
    def pack_task(m_idx, n_idx, k_idx, job_id: int):
        return T.bitwise_or(
            T.bitwise_or(job_id, T.shift_left(m_idx, 5)),
            T.bitwise_or(T.shift_left(n_idx, 18), T.shift_left(k_idx, 28)),
        )
