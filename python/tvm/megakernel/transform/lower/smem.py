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
"""Lowering/runtime shared-memory helpers for megakernel codegen."""

from __future__ import annotations

from typing import Literal

import tvm.tirx.script as T

from ...dsl.impl import SmemAllocRecord, SmemManager


@T.inline
def _emit_wait_all(mbar, cur_phase, chunk_num: T.constexpr, warp_count: T.constexpr):
    lane_id = T.lane_id([32])
    warp_id = T.warp_id([warp_count])
    if warp_id == 0:
        if lane_id < chunk_num:
            T.ptx.mbarrier.try_wait(mbar.ptr_to([lane_id]), cur_phase[0])
    T.tvm_storage_sync("shared")


@T.inline
def _emit_release_all(mbar, chunk_num: T.constexpr, warp_count: T.constexpr):
    lane_id = T.lane_id([32])
    warp_id = T.warp_id([warp_count])
    T.tvm_storage_sync("shared")
    if warp_id == 0:
        if lane_id < chunk_num:
            T.ptx.mbarrier.arrive(mbar.ptr_to([lane_id]))


@T.inline
def _emit_advance(cur_phase):
    cur_phase[0] = cur_phase[0] ^ 1


class TIRXSmemManager(SmemManager):
    """Concrete TIRX implementation of the user-facing ``SmemManager`` API."""

    VALID_POLICIES = {"shared", "exclusive", "persistent"}

    def __init__(self, smem_max_bytes, chunk_size, *, num_threads=256, warp_count=None):
        self.smem_max_bytes = smem_max_bytes
        self.chunk_size = chunk_size
        self.chunk_num = smem_max_bytes // chunk_size
        assert self.chunk_num <= 32
        self.num_threads = num_threads
        self.warp_count = warp_count or max(1, num_threads // 32)

        regular_pool_allocator = T.SMEMPool()
        persistent_pool_allocator = T.SMEMPool(regular_pool_allocator.ptr)
        # Keep persistent metadata inside the committed dynamic-smem region.
        # The regular pool starts from the front; persistent allocations start
        # from the last chunk so small always-live buffers, such as the static
        # scheduler queue, do not land past the launch-time smem size.
        persistent_pool_allocator.move_base_to(self.chunk_size * max(0, self.chunk_num - 1))

        self.pool_allocator = {
            "persistent": persistent_pool_allocator,
            "shared": regular_pool_allocator,
            "exclusive": regular_pool_allocator,
        }
        self.tiles = {}
        self.runtime_tile_chunk_count = {}
        self.bufs = {}
        self.persistent_bufs = {}
        self.cur_tile_name = ""
        self.exist_bufs = {}
        self.records: list[SmemAllocRecord] = []
        self.tile_phase_state = {}
        self.mbar = None
        self.shared_count = None
        self.cur_phase = None
        self.reg_count = None

    def set_tile(self, cur_tile) -> None:
        """Start recording allocations for one logical tile."""

        self.cur_tile_name = "default" if cur_tile is None else str(cur_tile)
        self.tiles[self.cur_tile_name] = [
            -1,
            {"exclusive": [], "shared": []},
            [0 for _ in range(self.chunk_num)],
        ]
        self.runtime_tile_chunk_count[self.cur_tile_name] = [
            [0 for _ in range(self.chunk_num)],
            [0 for _ in range(self.chunk_num)],
        ]
        self.tile_phase_state[self.cur_tile_name] = {
            "uses_managed_smem": False,
            "wait_all": False,
            "release_all": False,
        }
        self.pool_allocator["shared"].move_base_to(0)

    def enter_tile_runtime(self, cur_tile) -> None:
        self.cur_tile_name = str(cur_tile)

    def exit_tile_runtime(self) -> None:
        self.cur_tile_name = ""

    def alloc(
        self,
        shape,
        dtype="float32",
        strides=None,
        scope="shared.dyn",
        align=0,
        buffer_type="",
        axis_separators=None,
        layout="default",
        split=1,
        name=None,
        policy: Literal["shared", "exclusive", "persistent"] = "shared",
    ):
        """Allocate a managed shared-memory buffer."""

        if policy not in self.VALID_POLICIES:
            valid = ", ".join(sorted(self.VALID_POLICIES))
            raise ValueError(
                f"Unsupported smem allocation policy {policy!r}; expected one of: {valid}"
            )
        if name is not None:
            if name in self.exist_bufs:
                self.exist_bufs[name] += 1
                name = f"{name}{self.exist_bufs[name] - 1}"
            else:
                self.exist_bufs[name] = 1

        pool_allocator = self.pool_allocator[policy]
        beg = pool_allocator.offset
        if align > 0:
            beg = (beg + align - 1) // align * align

        buffer = pool_allocator.alloc(
            shape,
            dtype,
            strides=strides,
            scope=scope,
            align=align,
            layout=layout,
        )
        end = pool_allocator.offset
        size = end - beg
        assert size % split == 0

        if policy == "persistent":
            self.persistent_bufs[buffer] = (beg, end)
        elif self.cur_tile_name in self.tiles:
            if policy == "shared":
                assert len(self.tiles[self.cur_tile_name][1]["exclusive"]) == 0, (
                    "Cannot use both shared and exclusive smem policies in one tile"
                )
            elif policy == "exclusive":
                assert len(self.tiles[self.cur_tile_name][1]["shared"]) == 0, (
                    "Cannot use both shared and exclusive smem policies in one tile"
                )
            buf_info = (split, beg, size, policy)
            self.tiles[self.cur_tile_name][0] = max(
                self.tiles[self.cur_tile_name][0], (end - 1) // self.chunk_size
            )
            self.tiles[self.cur_tile_name][1][policy].append(buf_info)
            self.bufs[buffer] = buf_info
            self.tile_phase_state.setdefault(
                self.cur_tile_name,
                {"uses_managed_smem": False, "wait_all": False, "release_all": False},
            )["uses_managed_smem"] = True
            if policy == "exclusive":
                for split_idx in range(split):
                    beg_chunk_id = (beg + size // split * split_idx) // self.chunk_size
                    end_chunk_id = (
                        beg + size // split * (split_idx + 1) - 1
                    ) // self.chunk_size
                    for chunk_id in range(beg_chunk_id, end_chunk_id + 1):
                        self.tiles[self.cur_tile_name][2][chunk_id] += 1

        self.records.append(
            SmemAllocRecord(
                buffer=buffer,
                shape=shape,
                dtype=dtype,
                policy=policy,
                attrs={
                    "strides": strides,
                    "scope": scope,
                    "align": align,
                    "buffer_type": buffer_type,
                    "axis_separators": axis_separators,
                    "layout": layout,
                    "split": split,
                    "name": name,
                },
            )
        )
        return buffer

    def check_smem_well_formed(self, debug=False) -> None:
        persistent_buf_list = list(self.persistent_bufs.values())

        for _, buf_info_dict, _ in self.tiles.values():
            checked_chunks = []
            check_overlap = []
            for policy in ["shared", "exclusive"]:
                for split, beg, size, _ in buf_info_dict[policy]:
                    end = beg + size
                    assert end <= self.smem_max_bytes
                    for beg_persistent, end_persistent in persistent_buf_list:
                        assert beg >= end_persistent or beg_persistent >= end, (
                            "Persistent and non-persistent smem allocations overlap"
                        )
                    for beg_other, end_other in check_overlap:
                        assert beg >= end_other or beg_other >= end, (
                            "Overlap detected in smem allocation"
                        )
                    check_overlap.append((beg, end))

                    if policy == "exclusive":
                        for split_idx in range(split):
                            beg_chunk_id = (beg + size // split * split_idx) // self.chunk_size
                            end_chunk_id = (
                                beg + size // split * (split_idx + 1) - 1
                            ) // self.chunk_size
                            for beg_id, end_id in checked_chunks:
                                assert beg_id > end_chunk_id or end_id < beg_chunk_id, (
                                    "Exclusive chunk overlap detected"
                                )
                            checked_chunks.append((beg_chunk_id, end_chunk_id))
                    else:
                        beg_chunk_id = beg // self.chunk_size
                        end_chunk_id = (end - 1) // self.chunk_size
                        checked_chunks.append((beg_chunk_id, end_chunk_id))

        for i, (beg_i, end_i) in enumerate(persistent_buf_list):
            assert 0 <= beg_i <= end_i <= self.smem_max_bytes
            for beg_j, end_j in persistent_buf_list[i + 1 :]:
                assert beg_i >= end_j or beg_j >= end_i, (
                    "Persistent smem allocations overlap"
                )

        if debug:
            self._debug_print()

    def _debug_print(self) -> None:
        for key, value in self.tiles.items():
            print(key, value)
        for key, value in self.bufs.items():
            print(key, value)
        for key, value in self.persistent_bufs.items():
            print(key, value)

    def validate_tile_phase(self, tile) -> None:
        """Require coarse phase synchronization for tiles using managed smem."""

        tile_name = "default" if tile is None else str(tile)
        display_name = "default" if tile is None else getattr(tile, "name", tile_name)
        state = self.tile_phase_state.get(tile_name)
        if not state or not state["uses_managed_smem"]:
            return
        missing = []
        if not state["wait_all"]:
            missing.append("wait_all()")
        if not state["release_all"]:
            missing.append("release_all()")
        if missing:
            raise ValueError(
                f"Tile {display_name} allocates managed shared memory but does not call "
                + " and ".join(missing)
            )

    def commit(self) -> None:
        """Finalize the shared-memory pool size annotation after allocations."""

        self.check_smem_well_formed(debug=False)
        self.pool_allocator["shared"].commit(self.smem_max_bytes)

    def _inner_alloc(self) -> None:
        self.mbar = self.alloc(
            (self.chunk_num,), "uint64", align=8, name="mbar", policy="persistent"
        )
        self.shared_count = self.alloc(
            (1,), "int32", align=4, name="shared_count", policy="persistent"
        )
        self.cur_phase = T.alloc_buffer((1,), "int32", scope="local", align=4)
        self.reg_count = T.alloc_buffer((1,), "int32", scope="local", align=4)

    @T.inline
    def init(self) -> None:
        """Initialize mbarriers for chunk-level shared-memory paging."""

        self._inner_alloc()
        self.cur_phase[0] = 1
        tid = T.thread_id([self.num_threads])
        if tid == 0:
            for i in T.serial(self.chunk_num):
                T.ptx.mbarrier.init(self.mbar.ptr_to([i]), 1)
            self.shared_count[0] = 0
        T.tvm_storage_sync("shared")
        T.ptx.fence.mbarrier_init()
        T.ptx.fence.proxy_async("shared::cta")

    def _phase_state(self):
        return self.tile_phase_state.setdefault(
            self.cur_tile_name,
            {"uses_managed_smem": False, "wait_all": False, "release_all": False},
        )

    def wait_all(self, level="cta") -> None:
        """Wait until every managed shared-memory chunk is ready for CTA reuse."""

        if level != "cta":
            raise ValueError("Only level='cta' is supported for SmemManager.wait_all")
        self._phase_state()["wait_all"] = True
        _emit_wait_all(self.mbar, self.cur_phase, self.chunk_num, self.warp_count)

    def release_all(self, level="cta") -> None:
        """Release every managed shared-memory chunk at CTA scope."""

        if level != "cta":
            raise ValueError("Only level='cta' is supported for SmemManager.release_all")
        self._phase_state()["release_all"] = True
        _emit_release_all(self.mbar, self.chunk_num, self.warp_count)

    def advance(self) -> None:
        """Flip the shared-memory mbarrier phase."""

        _emit_advance(self.cur_phase)
