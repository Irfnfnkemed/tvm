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

"""User-facing implementation hooks for the megakernel DSL."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

import tvm.tirx.script as T


@dataclass(frozen=True)
class SmemAllocRecord:
    """Metadata recorded for one logical shared-memory allocation."""

    buffer: Any
    shape: Any
    dtype: str
    policy: str
    attrs: dict[str, Any] = field(default_factory=dict)


class SmemManager:
    """Shared-memory manager passed to ``TileImpl`` hooks.

    Users receive this object from hook arguments and use it to allocate managed
    shared-memory buffers.  Concrete lowering code provides the implementation.
    """

    VALID_POLICIES = {"shared", "exclusive", "persistent"}

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
        policy: Literal["shared", "exclusive", "persistent"] = "shared",
    ):
        """Allocate managed shared memory and return a buffer."""

        ...

    def commit(self):
        """Finalize managed shared-memory allocation metadata."""

        ...

    def wait_all(self, level="cta"):
        """Wait until all managed shared-memory chunks are ready at CTA scope."""

        ...

    def release_all(self, level="cta"):
        """Release all managed shared-memory chunks at CTA scope."""

        ...

    def advance(self):
        """Advance the shared-memory phase."""

        ...


class TileImpl(ABC):
    """Parser-style implementation of one tile kind.

    `TileImpl` is the user-facing implementation layer.  It should describe the
    local body of one tile.  Global scheduling, event placement, and final
    shared-memory planning remain lowering-pass responsibilities.

    Tile-instance hooks that emit TIRX statements, such as `device_init()`,
    `prefetch()`, and `run()`, should be decorated with `@T.inline` by the user
    implementation.  Resource declaration helpers can be ordinary Python
    methods called from an inline hook.
    """

    @classmethod
    def class_name(cls):
        return cls.__name__

    def __init__(self):
        self._instance_id = id(self)

    def __str__(self):
        return f"{self.__class__.__name__}-{self._instance_id:x}"

    @classmethod
    @T.inline
    def init_shared_resources(cls, smem_manager: SmemManager):
        """Optionally emit parser-style initialization for shared tile-class resources."""

    @classmethod
    @T.inline
    def finalize_shared_resources(cls, smem_manager: SmemManager):
        """Optionally emit parser-style finalization for shared tile-class resources."""

    @T.inline
    def device_init(self, smem_manager: SmemManager, m_idx, n_idx, k_idx):
        """Optionally emit parser-style device initialization for one tile instance."""

    def host_init(self):
        """Run host-side initialization for one tile instance, if needed."""

    @T.inline
    def prefetch(self, m_idx, n_idx, k_idx):
        """Optionally emit parser-style prefetch code for one tile instance."""

    @abstractmethod
    @T.inline
    def run(self, m_idx, n_idx, k_idx):
        """Emit the parser-style body for one logical tile instance."""
