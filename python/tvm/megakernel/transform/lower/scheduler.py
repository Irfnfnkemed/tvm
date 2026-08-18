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
"""Runtime scheduler helpers used by megakernel lowering."""

from __future__ import annotations

from typing import Any

import tvm.tirx.script as T

from ...dsl.impl import SmemManager


def _is_const_minus_one(value) -> bool:
    return (isinstance(value, int) and not isinstance(value, bool) and value == -1) or (
        type(value).__name__ == "IntImm" and getattr(value, "value", None) == -1
    )


def _atomic_add_int32_remote(addr, value, rank):
    return T.cuda.func_call(
        "atomic_add_int32_remote",
        addr,
        value,
        rank,
        source_code="""
__forceinline__ __device__ int32_t atomic_add_int32_remote(int32_t* addr, int32_t value, int32_t pe) {
#if defined(NVSHMEM_MAJOR_VERSION)
    if (pe >= 0) {
        int32_t* ptr = (int32_t*)(nvshmem_ptr(addr, pe));
        int32_t old_value;
        asm volatile ("atom.release.gpu.global.add.s32 %0, [%1], %2;"
                      : "=r"(old_value)
                      : "l"(ptr), "r"(value)
                      : "memory");
        return old_value;
    }
#endif
    return atomicAdd(addr, value);
}
""",
        return_type="int32",
    )


def _atomic_add_int32_release(addr, value):
    return T.cuda.func_call(
        "atomic_add_int32_release",
        addr,
        value,
        source_code="""
__forceinline__ __device__ int32_t atomic_add_int32_release(int32_t* addr, int32_t value) {
    int32_t old_value;
    asm volatile ("atom.release.gpu.global.add.s32 %0, [%1], %2;"
                  : "=r"(old_value)
                  : "l"(addr), "r"(value)
                  : "memory");
    return old_value;
}
""",
        return_type="int32",
    )


def _atomic_add_int32(addr, value, rank=-1, *, release: bool = False):
    if _is_const_minus_one(rank):
        if release:
            return _atomic_add_int32_release(addr, value)
        return T.cuda.atomic_add(addr, value)
    return _atomic_add_int32_remote(addr, value, rank)


def _gt(lhs, rhs):
    return T.cuda.func_call(
        "gt",
        lhs,
        rhs,
        source_code="""
__forceinline__ __device__ bool gt(int32_t a, int32_t b) {
    return a > b;
}
""",
        return_type="bool",
    )


class StaticTIRXSemaphore:
    """Semaphore object backed by one event buffer."""

    base = 1 << 16

    def __init__(self, buffer, *, sleep_cycles: int = 40):
        self.buffer = buffer
        self.state = T.alloc_buffer((1,), "int32", scope="local", align=4)
        self.sleep_cycles = sleep_cycles

    @T.inline
    def semaphore_wait(self, *coord, level: str = "cta", mask=0xFFFFFFFF) -> None:
        """Wait until the event counter at ``coord`` reaches zero."""

        if level == "cta":
            while 1:
                T.ptx.ld_global_acquire(
                    self.state[0],
                    self.buffer.access_ptr("r", offset=self.buffer.elem_offset_of(coord)),
                )
                if T.cuda.syncthreads_and(self.state[0] == 0):
                    break
                T.cuda.nano_sleep(self.sleep_cycles)
        elif level == "warp":
            warp_id = T.warp_id([8])
            lane_id = T.lane_id([32])
            if ((mask >> warp_id) & 1) == 1:
                self.state[0] = -1
                while 1:
                    if lane_id == 0:
                        T.ptx.ld_global_acquire(
                            self.state[0],
                            self.buffer.access_ptr("r", offset=self.buffer.elem_offset_of(coord)),
                        )
                    if T.ptx.any_sync(0xFFFFFFFF, self.state[0] == 0):
                        break
                    T.cuda.nano_sleep(self.sleep_cycles)
        else:
            assert False

    @T.inline
    def semaphore_notify(self, *coord, rank=-1, release: bool = False) -> None:
        """Notify one completed producer for the event counter at ``coord``."""

        self.state[0] = _atomic_add_int32(
            self.buffer.ptr_to(coord), -(self.base + 1), rank, release=release
        )
        if self.state[0] <= 0:
            while 1:
                T.ptx.ld_global_acquire(self.state[0], self.buffer.ptr_to(coord))
                if _gt(self.state[0], 0):
                    self.state[0] = _atomic_add_int32(
                        self.buffer.ptr_to(coord), -(self.base + 1), rank, release=release
                    )
                    break
                T.cuda.nano_sleep(self.sleep_cycles)


class DynamicTIRXSemaphore(StaticTIRXSemaphore):
    """Dynamic scheduler semaphore with pre-notify and completion-notify phases."""

    @T.inline
    def semaphore_pre_notify(self, *coord, rank=-1, release: bool = False) -> None:
        """Record that one producer task was dispatched far enough to push consumers."""

        self.state[0] = _atomic_add_int32(self.buffer.ptr_to(coord), -1, rank, release=release)
        if self.state[0] <= 0:
            while 1:
                T.ptx.ld_global_acquire(self.state[0], self.buffer.ptr_to(coord))
                if _gt(self.state[0], 0):
                    self.state[0] = _atomic_add_int32(self.buffer.ptr_to(coord), -1, rank, release=release)
                    break
                T.cuda.nano_sleep(800)

    @T.inline
    def semaphore_complete_notify(self, *coord, rank=-1, release: bool = False) -> None:
        """Record that one producer task has completed its body."""

        self.state[0] = _atomic_add_int32(self.buffer.ptr_to(coord), -self.base, rank, release=release)
        if self.state[0] <= 0:
            while 1:
                T.ptx.ld_global_acquire(self.state[0], self.buffer.ptr_to(coord))
                if _gt(self.state[0], 0):
                    self.state[0] = _atomic_add_int32(self.buffer.ptr_to(coord), -self.base, rank, release=release)
                    break
                T.cuda.nano_sleep(self.sleep_cycles)

    def is_triggered(self):
        """Return whether this pre-notify observed the final outstanding producer."""

        return self.state[0] % self.base == 1


class StaticTileScheduler:
    """Static queue scheduler mirroring the old megakernel scheduler shape."""

    MAX_TASKS = 128

    def __init__(
        self,
        prefix: str,
        exec_queue: Any,
        smem_manager: SmemManager,
        debug: bool = False,
        *,
        sm_count: int = 1,
        num_threads: int = 256,
        max_tasks: int | None = None,
        end_job_id: int = 31,
        warp_count: int | None = None,
        warpgroup_count: int | None = None,
        warpgroup_size: int = 128,
    ):
        self.exec_queue = exec_queue
        self.debug = debug
        self.prefix = prefix
        self.smem_manager = smem_manager
        self.sm_count = sm_count
        self.num_threads = num_threads
        self.max_tasks = max_tasks or self.MAX_TASKS
        self.end_job_id = end_job_id
        self.warp_count = warp_count or max(1, num_threads // 32)
        self.warpgroup_count = warpgroup_count or max(1, num_threads // warpgroup_size)
        self.warpgroup_size = warpgroup_size

    def _alloc(self) -> None:
        self.m_idx = T.alloc_buffer((1,), "int32", scope="local")
        self.n_idx = T.alloc_buffer((1,), "int32", scope="local")
        self.k_idx = T.alloc_buffer((1,), "int32", scope="local")
        self.task_type = T.alloc_buffer((1,), "int32", scope="local")
        self.tile_idx = T.alloc_buffer((1,), "int32", scope="local")
        self.queue_smem = self.smem_manager.alloc(
            (self.max_tasks,), "int32", align=16, policy="persistent"
        )

    @T.inline
    def _update_current_m_n_idx(self) -> None:
        packed = T.alloc_buffer((1,), "int32", scope="local")
        packed[0] = self.queue_smem[self.tile_idx[0]]
        self.task_type[0] = T.bitwise_and(packed[0], 0x1F)
        self.m_idx[0] = T.bitwise_and(T.shift_right(packed[0], 5), 0x1FFF)
        self.n_idx[0] = T.bitwise_and(T.shift_right(packed[0], 18), 0x3FF)
        self.k_idx[0] = T.bitwise_and(T.shift_right(packed[0], 28), 0xF)

    @T.inline
    def init(self) -> None:
        self._alloc()
        bx = T.cta_id([self.sm_count])
        T.warp_id([self.num_threads // 32])
        T.lane_id([32])
        tid = T.thread_id([self.num_threads])
        self.tile_idx[0] = 0
        for k in T.serial(0, (self.max_tasks + self.num_threads - 1) // self.num_threads):
            idx = k * self.num_threads + tid
            if idx < self.max_tasks:
                self.queue_smem[idx] = self.exec_queue[bx, idx]
        T.tvm_storage_sync("shared")
        self._update_current_m_n_idx()

    def get_idx_and_task_type(self):
        return [self.m_idx[0], self.n_idx[0], self.k_idx[0]], self.task_type[0]

    @T.inline
    def next_tile(self) -> None:
        self.tile_idx[0] = self.tile_idx[0] + 1
        self._update_current_m_n_idx()

    @T.inline
    def wait(self, semaphore, *coord, wait_level: str = "cta", mask=0xFFFFFFFF) -> None:
        """Wait on an event semaphore under the static scheduling policy."""

        semaphore.semaphore_wait(*coord, level=wait_level, mask=mask)

    @T.inline
    def notify(
        self,
        semaphore,
        func_notify,
        scope: str = "thread",
        scope_id: int = 0,
        release: bool = False,
    ) -> None:
        """Notify an event semaphore under the static scheduling policy."""

        max_coord_count_map = T.meta_var(
            {
                "thread": 1,
                "warp": 32,
                "warpgroup": self.warpgroup_size,
                "cta": self.num_threads,
            }
        )
        max_scope_id_map = T.meta_var(
            {
                "thread": self.num_threads,
                "warp": self.warp_count,
                "warpgroup": self.warpgroup_count,
                "cta": 1,
            }
        )

        wg_id = T.warpgroup_id([self.warpgroup_count])
        warp_id = T.warp_id([self.warp_count])
        tid = T.thread_id([self.num_threads])
        tid_in_wg = T.thread_id_in_wg([self.warpgroup_size])
        lane_id = T.lane_id([32])
        idx_map = T.meta_var(
            {
                "thread": (tid, 0),
                "warp": (warp_id, lane_id),
                "warpgroup": (wg_id, tid_in_wg),
                "cta": (0, tid),
            }
        )
        idx = idx_map[scope]

        if self.debug:
            T.cuda.trap_when_assert_failed(scope_id == -1 or scope_id < max_scope_id_map[scope])
        if scope_id == -1 or idx[0] == scope_id:
            self._sync_notify_scope(scope, scope_id)
            notify_info = T.meta_var(func_notify(idx[1]))
            coord_count = notify_info[0]
            rank = notify_info[1]
            coord = T.meta_var(notify_info[2:])
            if self.debug:
                T.cuda.trap_when_assert_failed(coord_count <= max_coord_count_map[scope])
            if idx[1] < coord_count:
                semaphore.semaphore_notify(*coord, rank=rank, release=release)

    @T.inline
    def _sync_notify_scope(self, scope: str, scope_id: int = 0) -> None:
        if scope == "thread":
            pass
        elif scope == "warp":
            T.cuda.warp_sync()
        elif scope == "warpgroup":
            T.ptx.bar.sync(6 + scope_id, self.warpgroup_size)
        elif scope == "cta":
            T.tvm_storage_sync("shared")
        else:
            assert False

    def valid(self):
        return (self.tile_idx[0] < self.max_tasks) & (self.task_type[0] != self.end_job_id)


class SchedulerBarrier:
    """One-slot mbarrier used by the dynamic scheduler dequeue path."""

    def __init__(self, smem_manager: SmemManager, *, initial_phase: int, num_threads: int):
        self.smem_manager = smem_manager
        self.initial_phase = initial_phase
        self.num_threads = num_threads

    def _alloc(self) -> None:
        self.mbar = self.smem_manager.alloc((1,), "uint64", align=16, policy="persistent")

    @T.inline
    def init(self, wait_count: int) -> None:
        self._alloc()
        tid = T.thread_id([self.num_threads])
        if tid == 0:
            T.ptx.mbarrier.init(self.mbar.ptr_to([0]), wait_count)

    @T.inline
    def wait(self, phase) -> None:
        T.ptx.mbarrier.try_wait(self.mbar.ptr_to([0]), self.initial_phase ^ phase)

    @T.inline
    def arrive(self) -> None:
        T.ptx.mbarrier.arrive(self.mbar.ptr_to([0]))


class DynamicTileScheduler:
    """Dynamic queue scheduler for megakernel lowering."""

    MAX_TASKS = 32768

    def __init__(
        self,
        tasks: Any,
        head: Any,
        tail: Any,
        smem_manager: SmemManager,
        debug: bool = False,
        *,
        num_threads: int = 256,
        max_tasks: int | None = None,
        end_job_id: int = 31,
        warp_count: int | None = None,
        warpgroup_count: int | None = None,
        warpgroup_size: int = 128,
        scheduler_warp: int = 7,
    ):
        self.tasks = tasks
        self.head = head
        self.tail = tail
        self.smem_manager = smem_manager
        self.debug = debug
        self.num_threads = num_threads
        self.max_tasks = max_tasks or self.MAX_TASKS
        self.end_job_id = end_job_id
        self.warp_count = warp_count or max(1, num_threads // 32)
        self.warpgroup_count = warpgroup_count or max(1, num_threads // warpgroup_size)
        self.warpgroup_size = warpgroup_size
        self.scheduler_warp = min(scheduler_warp, self.warp_count - 1)

    def _alloc(self) -> None:
        self.m_idx = T.alloc_buffer((1,), "int32", scope="local")
        self.n_idx = T.alloc_buffer((1,), "int32", scope="local")
        self.k_idx = T.alloc_buffer((1,), "int32", scope="local")
        self.task_type = T.alloc_buffer((1,), "int32", scope="local")
        self.packed_value = self.smem_manager.alloc(
            (1,), "int32", align=16, policy="persistent"
        )
        self.enqueue_pos = T.alloc_buffer((1,), "int32", scope="local")
        self.dequeue_pos = T.alloc_buffer((1,), "int32", scope="local")
        self.dequeue_phase = T.alloc_buffer((1,), "int32", scope="local")
        self.p2c_dequeue_barrier = SchedulerBarrier(
            self.smem_manager, initial_phase=0, num_threads=self.num_threads
        )
        self.c2p_dequeue_barrier = SchedulerBarrier(
            self.smem_manager, initial_phase=1, num_threads=self.num_threads
        )
        self.push_idx = T.alloc_buffer((1,), "int32", scope="local")
        self.semaphore_state = self.smem_manager.alloc(
            (self.num_threads,), "int32", align=16, policy="persistent"
        )
        self.tail_smem = self.smem_manager.alloc(
            (max(self.warp_count, self.warpgroup_count, 1),),
            "int32",
            align=16,
            policy="persistent",
        )

    @T.inline
    def init(self) -> None:
        self._alloc()
        self.task_type[0] = self.end_job_id
        self.m_idx[0] = 0
        self.n_idx[0] = 0
        self.k_idx[0] = 0
        self.dequeue_phase[0] = 0
        self.p2c_dequeue_barrier.init(1)
        self.c2p_dequeue_barrier.init(self.num_threads)
        T.tvm_storage_sync("shared")
        T.ptx.fence.mbarrier_init()
        T.ptx.fence.proxy_async("shared::cta")

    @T.inline
    def _unpack_current(self) -> None:
        packed = T.alloc_buffer((1,), "int32", scope="local")
        packed[0] = self.packed_value[0]
        self.task_type[0] = T.bitwise_and(packed[0], 0x1F)
        self.m_idx[0] = T.bitwise_and(T.shift_right(packed[0], 5), 0x1FFF)
        self.n_idx[0] = T.bitwise_and(T.shift_right(packed[0], 18), 0x3FF)
        self.k_idx[0] = T.bitwise_and(T.shift_right(packed[0], 28), 0xF)

    @T.inline
    def _dequeue_to_shared(self) -> None:
        self.dequeue_pos[0] = T.cuda.atomic_add(self.head.ptr_to([0]), 1)
        self.packed_value[0] = self.tasks[self.dequeue_pos[0] % self.max_tasks]

    @T.inline
    def next_tile(self) -> None:
        warp_id = T.warp_id([self.warp_count])
        if warp_id == self.scheduler_warp:
            if T.ptx.elect_sync():
                self.c2p_dequeue_barrier.wait(self.dequeue_phase[0])
                self._dequeue_to_shared()
                self.p2c_dequeue_barrier.arrive()
        self.p2c_dequeue_barrier.wait(self.dequeue_phase[0])
        self._unpack_current()
        self.c2p_dequeue_barrier.arrive()
        self.dequeue_phase[0] = self.dequeue_phase[0] ^ 1

    def get_idx_and_task_type(self):
        return [self.m_idx[0], self.n_idx[0], self.k_idx[0]], self.task_type[0]

    @T.inline
    def enqueue(self, packed_task, *, push_level: str = "cta") -> None:
        tid = T.thread_id([self.num_threads])
        lane_id = T.lane_id([32])
        warp_id = T.warp_id([self.warp_count])
        wg_id = T.warpgroup_id([self.warpgroup_count])
        tid_in_wg = T.thread_id_in_wg([self.warpgroup_size])
        if push_level == "thread":
            if tid == 0:
                self.enqueue_pos[0] = T.cuda.atomic_add(self.tail.ptr_to([0]), 1)
                self.tasks[self.enqueue_pos[0] % self.max_tasks] = packed_task
        elif push_level == "warp":
            if warp_id == 0:
                if lane_id == 0:
                    self.tail_smem[0] = T.cuda.atomic_add(self.tail.ptr_to([0]), 1)
                T.cuda.warp_sync()
                if lane_id == 0:
                    self.tasks[self.tail_smem[0] % self.max_tasks] = packed_task
        elif push_level == "warpgroup":
            if wg_id == 0:
                if tid_in_wg == 0:
                    self.tail_smem[0] = T.cuda.atomic_add(self.tail.ptr_to([0]), 1)
                T.ptx.bar.sync(6, self.warpgroup_size)
                if tid_in_wg == 0:
                    self.tasks[self.tail_smem[0] % self.max_tasks] = packed_task
        elif push_level == "cta":
            if tid == 0:
                self.enqueue_pos[0] = T.cuda.atomic_add(self.tail.ptr_to([0]), 1)
                self.tasks[self.enqueue_pos[0] % self.max_tasks] = packed_task
        else:
            assert False
        T.tvm_storage_sync("shared")

    @T.inline
    def _enqueue_many_current_scope(self, push_count, func_push, *, push_level: str = "cta") -> None:
        tid = T.thread_id([self.num_threads])
        lane_id = T.lane_id([32])
        warp_id = T.warp_id([self.warp_count])
        wg_id = T.warpgroup_id([self.warpgroup_count])
        tid_in_wg = T.thread_id_in_wg([self.warpgroup_size])
        if push_level == "thread":
            push_info = T.meta_var(func_push(0))
            if self.debug:
                T.cuda.trap_when_assert_failed(push_count == 1)
            self.enqueue_pos[0] = T.cuda.atomic_add(self.tail.ptr_to([0]), 1)
            self.tasks[self.enqueue_pos[0] % self.max_tasks] = push_info
        elif push_level == "warp":
            if lane_id == 0:
                self.tail_smem[warp_id] = T.cuda.atomic_add(self.tail.ptr_to([0]), push_count)
            T.cuda.warp_sync()
            self.push_idx[0] = lane_id
            while self.push_idx[0] < push_count:
                self.tasks[(self.tail_smem[warp_id] + self.push_idx[0]) % self.max_tasks] = func_push(
                    self.push_idx[0]
                )
                self.push_idx[0] = self.push_idx[0] + 32
        elif push_level == "warpgroup":
            if tid_in_wg == 0:
                self.tail_smem[wg_id] = T.cuda.atomic_add(self.tail.ptr_to([0]), push_count)
            T.ptx.bar.sync(6 + wg_id, self.warpgroup_size)
            self.push_idx[0] = tid_in_wg
            while self.push_idx[0] < push_count:
                self.tasks[(self.tail_smem[wg_id] + self.push_idx[0]) % self.max_tasks] = func_push(
                    self.push_idx[0]
                )
                self.push_idx[0] = self.push_idx[0] + self.warpgroup_size
        elif push_level == "cta":
            if tid == 0:
                self.tail_smem[0] = T.cuda.atomic_add(self.tail.ptr_to([0]), push_count)
            T.tvm_storage_sync("shared")
            self.push_idx[0] = tid
            while self.push_idx[0] < push_count:
                self.tasks[(self.tail_smem[0] + self.push_idx[0]) % self.max_tasks] = func_push(
                    self.push_idx[0]
                )
                self.push_idx[0] = self.push_idx[0] + self.num_threads
        else:
            assert False
        T.tvm_storage_sync("shared")


    @T.inline
    def wait(self, semaphore, *coord, wait_level: str = "cta", mask=0xFFFFFFFF) -> None:
        semaphore.semaphore_wait(*coord, level=wait_level, mask=mask)

    @T.inline
    def pre_notify_and_push(
        self,
        semaphore,
        func_notify,
        func_push,
        *,
        push_level: str = "cta",
        scope: str = "cta",
        scope_id: int = 0,
    ) -> None:
        max_coord_count_map = T.meta_var(
            {
                "thread": 1,
                "warp": 32,
                "warpgroup": self.warpgroup_size,
                "cta": self.num_threads,
            }
        )
        max_scope_id_map = T.meta_var(
            {
                "thread": self.num_threads,
                "warp": self.warp_count,
                "warpgroup": self.warpgroup_count,
                "cta": 1,
            }
        )
        wg_id = T.warpgroup_id([self.warpgroup_count])
        warp_id = T.warp_id([self.warp_count])
        tid = T.thread_id([self.num_threads])
        tid_in_wg = T.thread_id_in_wg([self.warpgroup_size])
        lane_id = T.lane_id([32])
        idx_map = T.meta_var(
            {
                "thread": (tid, 0),
                "warp": (warp_id, lane_id),
                "warpgroup": (wg_id, tid_in_wg),
                "cta": (0, tid),
            }
        )
        idx_in_scope_map = T.meta_var(
            {
                "thread": {"thread": 0},
                "warp": {"thread": lane_id, "warp": 0},
                "warpgroup": {"thread": tid_in_wg, "warp": warp_id, "warpgroup": 0},
                "cta": {"thread": tid, "warp": warp_id, "warpgroup": wg_id, "cta": 0},
            }
        )
        stride_in_scope_map = T.meta_var(
            {
                "warp": {"warp": 1},
                "warpgroup": {"warp": self.warp_count, "warpgroup": 1},
                "cta": {"warp": self.warp_count, "warpgroup": self.warpgroup_count, "cta": 1},
            }
        )
        scope_id_map = T.meta_var(
            {"thread": tid, "warp": warp_id, "warpgroup": wg_id, "cta": 0}
        )
        new_scope_id = T.if_then_else(scope_id == -1, scope_id_map[scope], scope_id)
        idx = idx_map[scope]
        if self.debug:
            T.cuda.trap_when_assert_failed(scope_id == -1 or scope_id < max_scope_id_map[scope])
        if idx[0] == new_scope_id:
            notify_info = T.meta_var(func_notify(idx[1]))
            coord_count = notify_info[0]
            rank = notify_info[1]
            coord = T.meta_var(notify_info[2:])
            if self.debug:
                T.cuda.trap_when_assert_failed(coord_count <= max_coord_count_map[scope])
            if idx[1] < coord_count:
                semaphore.semaphore_pre_notify(*coord, rank=rank)
                self.semaphore_state[tid] = semaphore.state[0]
            else:
                self.semaphore_state[tid] = 0
            T.tvm_storage_sync("shared")
            if push_level == "thread":
                if idx[1] < coord_count:
                    semaphore.state[0] = self.semaphore_state[tid]
                    if semaphore.is_triggered():
                        notify_i = idx[1]
                        push_info = T.meta_var(func_push(notify_i, 0))
                        self._enqueue_many_current_scope(
                            push_info[0],
                            lambda push_i: func_push(notify_i, push_i)[1],
                            push_level=push_level,
                        )
            elif scope == "warp" and push_level == "warp":
                self.push_idx[0] = idx_in_scope_map[scope][push_level]
                while self.push_idx[0] < coord_count:
                    semaphore.state[0] = self.semaphore_state[new_scope_id * 32 + self.push_idx[0]]
                    if semaphore.is_triggered():
                        notify_i = self.push_idx[0]
                        push_info = T.meta_var(func_push(notify_i, 0))
                        self._enqueue_many_current_scope(
                            push_info[0],
                            lambda push_i: func_push(notify_i, push_i)[1],
                            push_level=push_level,
                        )
                    self.push_idx[0] = self.push_idx[0] + stride_in_scope_map[scope][push_level]
            elif scope == "warpgroup" and (push_level == "warp" or push_level == "warpgroup"):
                self.push_idx[0] = idx_in_scope_map[scope][push_level]
                while self.push_idx[0] < coord_count:
                    semaphore.state[0] = self.semaphore_state[
                        new_scope_id * self.warpgroup_size + self.push_idx[0]
                    ]
                    if semaphore.is_triggered():
                        notify_i = self.push_idx[0]
                        push_info = T.meta_var(func_push(notify_i, 0))
                        self._enqueue_many_current_scope(
                            push_info[0],
                            lambda push_i: func_push(notify_i, push_i)[1],
                            push_level=push_level,
                        )
                    self.push_idx[0] = self.push_idx[0] + stride_in_scope_map[scope][push_level]
            elif scope == "cta" and (push_level == "warp" or push_level == "warpgroup" or push_level == "cta"):
                self.push_idx[0] = idx_in_scope_map[scope][push_level]
                while self.push_idx[0] < coord_count:
                    semaphore.state[0] = self.semaphore_state[self.push_idx[0]]
                    if semaphore.is_triggered():
                        notify_i = self.push_idx[0]
                        push_info = T.meta_var(func_push(notify_i, 0))
                        self._enqueue_many_current_scope(
                            push_info[0],
                            lambda push_i: func_push(notify_i, push_i)[1],
                            push_level=push_level,
                        )
                    self.push_idx[0] = self.push_idx[0] + stride_in_scope_map[scope][push_level]
            else:
                assert False
        T.tvm_storage_sync("shared")


    @T.inline
    def complete_notify(
        self,
        semaphore,
        func_notify,
        scope: str = "cta",
        scope_id: int = 0,
        release: bool = False,
    ) -> None:
        max_coord_count_map = T.meta_var(
            {
                "thread": 1,
                "warp": 32,
                "warpgroup": self.warpgroup_size,
                "cta": self.num_threads,
            }
        )
        max_scope_id_map = T.meta_var(
            {
                "thread": self.num_threads,
                "warp": self.warp_count,
                "warpgroup": self.warpgroup_count,
                "cta": 1,
            }
        )
        wg_id = T.warpgroup_id([self.warpgroup_count])
        warp_id = T.warp_id([self.warp_count])
        tid = T.thread_id([self.num_threads])
        tid_in_wg = T.thread_id_in_wg([self.warpgroup_size])
        lane_id = T.lane_id([32])
        idx_map = T.meta_var(
            {
                "thread": (tid, 0),
                "warp": (warp_id, lane_id),
                "warpgroup": (wg_id, tid_in_wg),
                "cta": (0, tid),
            }
        )
        idx = idx_map[scope]
        if self.debug:
            T.cuda.trap_when_assert_failed(scope_id == -1 or scope_id < max_scope_id_map[scope])
        if scope_id == -1 or idx[0] == scope_id:
            StaticTileScheduler._sync_notify_scope(self, scope, scope_id)
            notify_info = T.meta_var(func_notify(idx[1]))
            coord_count = notify_info[0]
            rank = notify_info[1]
            coord = T.meta_var(notify_info[2:])
            if self.debug:
                T.cuda.trap_when_assert_failed(coord_count <= max_coord_count_map[scope])
            if idx[1] < coord_count:
                semaphore.semaphore_complete_notify(*coord, rank=rank, release=release)

    def valid(self):
        return self.task_type[0] != self.end_job_id
