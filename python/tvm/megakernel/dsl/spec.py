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

from .expr import ExprLike, ExprSpec, VarSpec, eval_expr_like, expr_bounds, expr_vars
from .impl import TileImpl


ShapeType = ExprLike | tuple[ExprLike, ...] | list[ExprLike]
CoordMapType = Callable[[int, int, int], tuple[int, ...]] | tuple[int, ...] | list[int]
TileNumType = tuple[ExprLike, ExprLike, ExprLike] | list[ExprLike]


@dataclass(frozen=True)
class RegionRange:
    """One BufferRegion-style dimension: half-open ``[start, start + extent)``."""

    start: Any
    extent: Any


@dataclass(frozen=True)
class RegionSpec:
    """Logical tensor region, normalized to BufferRegion-style ranges."""

    dims: tuple[RegionRange, ...]


class RegionBuilder:
    """Builder used as ``R[i, j:j+8]`` in tensor region lambdas."""

    def __getitem__(self, indices):
        if not isinstance(indices, tuple):
            indices = (indices,)
        dims = []
        for index in indices:
            if isinstance(index, slice):
                if index.step is not None:
                    raise ValueError("region slices do not support step")
                if index.stop is None:
                    raise ValueError("region slices require a stop")
                start = 0 if index.start is None else index.start
                dims.append(RegionRange(start=start, extent=index.stop - start))
            else:
                dims.append(RegionRange(start=index, extent=1))
        return RegionSpec(dims=tuple(dims))


R = RegionBuilder()
TileRegionResult = RegionSpec | tuple[Any, ...] | list[Any]
TileRegionMap = Callable[[Any, Any, Any], TileRegionResult] | TileRegionResult


@dataclass(frozen=True)
class TensorSpec:
    """Logical tensor or tensor access in the megakernel spec.

    A registered tensor, created by ``KernelSpec.tensor()``, has no ``base``.
    Passing that bare tensor in ``TileSpec.reads`` or ``TileSpec.writes`` means
    the tile may access an unknown region of the tensor.  ``region()`` creates
    an access view with a known tile-index-to-region mapping.
    """

    name: str
    shape: ShapeType
    dtype: str
    region_from_tile: TileRegionMap | None = field(default=None, compare=False)
    base: "TensorSpec | None" = field(default=None, compare=False)

    @property
    def base_tensor(self) -> "TensorSpec":
        """Return the registered tensor behind a region access view."""

        return self.base if self.base is not None else self

    def region(self, region_from_tile: TileRegionMap) -> "TensorSpec":
        """Return an access view with a known tile-index-to-region mapping."""

        base = self.base_tensor
        return TensorSpec(
            name=base.name,
            shape=base.shape,
            dtype=base.dtype,
            region_from_tile=region_from_tile,
            base=base,
        )


@dataclass(frozen=True)
class EventSpec:
    """Logical readiness event."""

    name: str
    shape: ShapeType
    init_count: int | Callable[[tuple[int, ...]], int]
    dtype: str = "int32"
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DependencySpec:
    """Internal event dependency attached to a tile wait or notify."""

    event: EventSpec
    coord_from_tile: CoordMapType
    inverse_coord_from_event: CoordMapType | None = None


@dataclass
class TileSpec:
    """One logical tile stage in the megakernel spec."""

    name: str
    impl: TileImpl
    tile_num: TileNumType
    reads: list[TensorSpec] = field(default_factory=list)
    writes: list[TensorSpec] = field(default_factory=list)
    waits: list[DependencySpec] = field(default_factory=list)
    notifies: list[DependencySpec] = field(default_factory=list)
    attrs: dict[str, Any] = field(default_factory=dict)

    def wait(
        self,
        event: EventSpec,
        coord_from_tile: CoordMapType,
        inverse_coord_from_event: CoordMapType | None = None,
    ):
        """Declare that this tile waits on ``event`` at ``coord_from_tile``."""

        self.waits.append(
            DependencySpec(
                event=event,
                coord_from_tile=coord_from_tile,
                inverse_coord_from_event=inverse_coord_from_event,
            )
        )
        return self

    def notify(self, event: EventSpec, coord_from_tile: CoordMapType):
        """Declare that this tile notifies ``event`` at ``coord_from_tile``."""

        self.notifies.append(DependencySpec(event=event, coord_from_tile=coord_from_tile))
        return self


class KernelSpec:
    """Container and builder for one megakernel specification."""

    def __init__(self, name: str, attrs: dict[str, Any] | None = None):
        self.name = name
        self.attrs = attrs or {}
        self.vars: dict[str, VarSpec] = {}
        self.tensors: dict[str, TensorSpec] = {}
        self.events: dict[str, EventSpec] = {}
        self.tiles: list[TileSpec] = []

    def var(
        self,
        name: str,
        dtype: str = "int32",
        range: tuple[int, int] | None = None,
    ):
        """Register a symbolic integer variable."""

        if name in self.vars:
            raise ValueError(f"Duplicate var: {name}")
        if range is not None:
            if (
                not isinstance(range, tuple)
                or len(range) != 2
                or any(not isinstance(value, int) or isinstance(value, bool) for value in range)
            ):
                raise TypeError("var range must be a tuple of two integers")
            if range[0] <= 0 or range[1] <= 0 or range[0] > range[1]:
                raise ValueError("var range must satisfy 0 < min <= max")
        var = VarSpec(name=name, dtype=dtype, range=range)
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
        read_tensors = _normalize_accesses(reads or [], "reads")
        write_tensors = _normalize_accesses(writes or [], "writes")
        tile = TileSpec(
            name=name,
            impl=impl,
            tile_num=tile_num,
            reads=read_tensors,
            writes=write_tensors,
            attrs=attrs or {},
        )
        self.tiles.append(tile)
        return tile

    def validate(self):
        raise NotImplementedError("Validation is not yet implemented.")

    def lower(self, options=None):
        from tvm.megakernel.transform import lower_to_tirx

        return lower_to_tirx(self, options)


def _normalize_accesses(accesses: list[TensorSpec], label: str) -> list[TensorSpec]:
    tensors: list[TensorSpec] = []
    for access in accesses:
        if not isinstance(access, TensorSpec):
            raise TypeError(f"tile {label} entries must be TensorSpec, got {access!r}")
        tensors.append(access)
    return tensors
