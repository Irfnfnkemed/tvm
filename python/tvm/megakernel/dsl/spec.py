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
"""Core building blocks for the megakernel DSL.

The DSL records the logical specification of a megakernel: tensors, events,
tile instances, and wait/notify relationships.  It stays independent from the
concrete CUDA/TIRX/runtime implementation chosen by later lowering passes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .impl import TileImpl


@dataclass(frozen=True)
class VarSpec:
    """Symbolic variable used in shapes, tile counts, and event counts."""

    name: str
    dtype: str = "int32"


ExprLike = int | VarSpec
ShapeType = ExprLike | tuple[ExprLike, ...] | list[ExprLike]
CoordMapType = Callable[[int, int, int], tuple[int, ...]] | tuple[int, ...] | list[int]
TileNumType = tuple[ExprLike, ExprLike, ExprLike] | list[ExprLike]

# ============================================================
# Tensor / Event
# ============================================================


@dataclass(frozen=True)
class TensorSpec:
    """Logical tensor or buffer in the megakernel spec.

    `TensorSpec` only records the visible interface of a tensor.  Producer and
    consumer relationships are recorded on `TileSpec.reads` and
    `TileSpec.writes`.
    """

    name: str
    shape: ShapeType
    dtype: str


@dataclass(frozen=True)
class EventSpec:
    """Logical readiness event.

    `init_count` describes the logical count needed at each event coordinate.
    It may be a scalar count shared by the whole event tensor, or a callable
    from event coordinate to count.  The concrete counter/barrier mechanism is
    a lowering decision and is not represented here.
    """

    name: str
    shape: ShapeType
    init_count: int | Callable[[tuple[int, ...]], int]
    dtype: str = "int32"
    attrs: dict[str, Any] = field(default_factory=dict)



# ============================================================
# TileSpec
# ============================================================

DependencyType = tuple[EventSpec, CoordMapType]

@dataclass
class TileSpec:
    """One logical tile stage in the megakernel spec.

    `tile_num` is always expressed on three axes `(m, n, k)`.  Use extent `1`
    for unused axes so every stage has a consistent tile index convention.
    """

    name: str
    impl: TileImpl
    tile_num: TileNumType
    reads: list[TensorSpec] = field(default_factory=list)
    writes: list[TensorSpec] = field(default_factory=list)
    waits: list[DependencyType] = field(default_factory=list)
    notifies: list[DependencyType] = field(default_factory=list)
    attrs: dict[str, Any] = field(default_factory=dict)

    def wait(self, event: EventSpec, coord_map: CoordMapType):
        """Declare that this tile waits on `event` at `coord_map`."""

        self.waits.append((event, coord_map))
        return self

    def notify(self, event: EventSpec, coord_map: CoordMapType):
        """Declare that this tile notifies `event` at `coord_map`."""

        self.notifies.append((event, coord_map))
        return self


# ============================================================
# KernelSpec
# ============================================================


class KernelSpec:
    """Container and builder for one megakernel specification."""

    def __init__(self, name: str, attrs: dict[str, Any] | None = None):
        self.name = name
        self.attrs = attrs or {}
        self.vars: dict[str, VarSpec] = {}
        self.tensors: dict[str, TensorSpec] = {}
        self.events: dict[str, EventSpec] = {}
        self.tiles: list[TileSpec] = []

    def var(self, name: str, dtype: str = "int32"):
        """Register a symbolic integer variable."""

        if name in self.vars:
            raise ValueError(f"Duplicate var: {name}")
        var = VarSpec(name=name, dtype=dtype)
        self.vars[name] = var
        return var

    def tensor(self, name: str, shape: ShapeType, dtype: str):
        """Register a logical tensor."""

        if name in self.tensors:
            raise ValueError(f"Duplicate tensor: {name}")
        tensor = TensorSpec(name=name, shape=shape, dtype=dtype)
        self.tensors[name] = tensor
        return tensor

    def event(
        self,
        name: str,
        shape: ShapeType,
        init_count: int | Callable[[tuple[int, ...]], int],
        dtype: str = "int32",
        attrs: dict[str, Any] | None = None,
    ):
        """Register a logical event."""

        if name in self.events:
            raise ValueError(f"Duplicate event: {name}")
        if init_count is None:
            raise ValueError("event init_count must be specified")
        event = EventSpec(
            name=name,
            shape=shape,
            init_count=init_count,
            dtype=dtype,
            attrs=attrs or {},
        )
        self.events[name] = event
        return event

    def tile(
        self,
        name: str,
        impl: TileImpl,
        tile_num: TileNumType,
        reads: list[TensorSpec] | None = None,
        writes: list[TensorSpec] | None = None,
        attrs: dict[str, Any] | None = None,
    ):
        """Register one tile stage."""

        if any(tile.name == name for tile in self.tiles):
            raise ValueError(f"Duplicate tile: {name}")
        tile = TileSpec(
            name=name,
            impl=impl,
            tile_num=tile_num,
            reads=reads or [],
            writes=writes or [],
            attrs=attrs or {},
        )
        self.tiles.append(tile)
        return tile

    def validate(self):
        raise NotImplementedError("Validation is not yet implemented.")

    def lower(self, options=None):
        from tvm.megakernel.transform import lower_to_tirx

        return lower_to_tirx(self, options)
