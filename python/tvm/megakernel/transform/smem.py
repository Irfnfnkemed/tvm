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

from ..dsl.impl import SmemAllocRecord, SmemManager


class TIRXSmemManager(SmemManager):
    """Concrete TIRX implementation of the user-facing ``SmemManager`` API."""

    VALID_POLICIES = {"shared", "exclusive", "persistent"}

    def __init__(self, smem_max_bytes, chunk_size):
        self.smem_max_bytes = smem_max_bytes
        self.chunk_size = chunk_size
        self.chunk_num = smem_max_bytes // chunk_size
        assert self.chunk_num <= 32

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
            strides,
            scope,
            align,
            buffer_type,
            axis_separators,
            layout,
        )
        end = pool_allocator.offset
        size = end - beg
        assert size % split == 0

        if policy == "persistent":
            self.persistent_bufs[buffer] = (beg, end)
        elif self.cur_tile_name in self.tiles:
            buf_info = (split, beg, size, policy)
            self.tiles[self.cur_tile_name][0] = max(
                self.tiles[self.cur_tile_name][0], (end - 1) // self.chunk_size
            )
            self.tiles[self.cur_tile_name][1][policy].append(buf_info)
            self.bufs[buffer] = buf_info

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
        if debug:
            self._debug_print()

    def _debug_print(self) -> None:
        for key, value in self.tiles.items():
            print(key, value)
        for key, value in self.bufs.items():
            print(key, value)
        for key, value in self.persistent_bufs.items():
            print(key, value)

    def commit(self) -> None:
        """Finalize the shared-memory pool size annotation after allocations."""

        self.pool_allocator["shared"].commit(self.smem_max_bytes)

    def wait_all(self, level="cta") -> None:
        """Emit the abstract marker for acquiring the current smem phase."""

        T.evaluate(T.call_extern("void", "tirx.megakernel.smem.wait_all", level))

    def release_all(self, level="cta") -> None:
        """Emit the abstract marker for releasing the current smem phase."""

        T.evaluate(T.call_extern("void", "tirx.megakernel.smem.release_all", level))

    def acquire_all(self, level="cta") -> None:
        self.wait_all(level)

    def advance(self) -> None:
        """Emit the abstract marker for advancing the logical smem phase."""

        T.evaluate(T.call_extern("void", "tirx.megakernel.smem.advance"))
