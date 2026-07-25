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
from .region import R, RegionRange, RegionSpec, TileRegionMap, TileRegionResult
from ..impl import TileImpl


ShapeType = ExprLike | tuple[ExprLike, ...] | list[ExprLike]
DependencyCoordMap = Callable[[Any, Any, Any, Any], tuple[Any, ...] | list[Any]]
InvCoordMap = Callable[..., tuple[Any, Any, Any, Any] | list[Any]]
CoordMapType = DependencyCoordMap
GridType = tuple[ExprLike, ExprLike, ExprLike] | list[ExprLike]


@dataclass(frozen=True)
class TensorSpec:
    """Logical tensor or tensor access in the megakernel spec.

    A registered tensor, created by ``KernelSpec.tensor()``, has no ``base``.
    Passing that bare tensor in ``TileSpec.reads`` or ``TileSpec.writes`` means
    the tile may access an unknown region of the tensor.  ``region()`` creates
    an access view with a known tile-index-to-region mapping.
    """

    name: str
    shape: tuple[ExprLike, ...]
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
    shape: tuple[ExprLike, ...]
    init_count: Callable[..., ExprLike]
    dtype: str = "int32"
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DependencySpec:
    """Internal event dependency attached to a tile wait or notify.

    Dependencies are created through the user-facing ``D`` builder.  ``coord``
    is the forward mapping from the current tile coordinate to the event
    coordinate touched by this dependency::

        coord(tile_m, tile_n, tile_k, notify_i) -> (notify_num, rank, *event_coord)

    ``notify_i`` is the worker index inside the selected notify scope.
    ``notify_num`` tells the scheduler how many workers participate in the
    notify operation.  ``rank`` is the destination rank; ``-1`` means local.
    For wait dependencies, lowering calls ``coord(m, n, k, 0)`` and only waits
    on ``event_coord``.  Wait dependencies must therefore describe exactly one
    local event coordinate, i.e. ``notify_num == 1`` and ``rank == -1``.

    ``inv_coord`` is the reverse mapping used only by dynamic scheduling when
    a notified event should push consumer tiles into the dynamic queue::

        inv_coord(rank, *event_coord, consumer_i) -> (consumer_num, tile_m, tile_n, tile_k)

    ``consumer_i`` is the fan-out index for cases where one ready event
    coordinate maps to multiple consumer tile coordinates.  ``consumer_num``
    is the total fan-out count for this ``(rank, *event_coord)``.  Lowering
    reads ``consumer_num`` from ``consumer_i == 0`` and then calls
    ``inv_coord`` for ``consumer_i`` in ``[0, consumer_num)``.
    """

    event: EventSpec
    coord: DependencyCoordMap
    inv_coord: InvCoordMap | None = None


class DependencyBuilder:
    """Build normalized event dependencies for ``TileSpec.wait``/``notify``."""

    def __call__(
        self,
        event: EventSpec,
        coord: DependencyCoordMap,
        *,
        inv_coord: InvCoordMap | None = None,
    ) -> DependencySpec:
        if not isinstance(event, EventSpec):
            raise TypeError("dependency event must be an EventSpec")
        if not callable(coord):
            raise TypeError("dependency coord must be callable")
        if inv_coord is not None and not callable(inv_coord):
            raise TypeError("dependency inv_coord must be callable")
        return DependencySpec(event=event, coord=coord, inv_coord=inv_coord)


D = DependencyBuilder()


@dataclass
class TileSpec:
    """One logical tile stage in the megakernel spec."""

    name: str
    impl: TileImpl
    grid: tuple[ExprLike, ExprLike, ExprLike]
    reads: list[TensorSpec] = field(default_factory=list)
    writes: list[TensorSpec] = field(default_factory=list)
    waits: list[DependencySpec] = field(default_factory=list)
    notifies: list[DependencySpec] = field(default_factory=list)
    attrs: dict[str, Any] = field(default_factory=dict)

    def wait(self, dependency: DependencySpec):
        """Declare an event wait dependency."""

        if not isinstance(dependency, DependencySpec):
            raise TypeError("TileSpec.wait expects a DependencySpec; use D(event, coord, ...)")
        self.waits.append(dependency)
        return self

    def notify(self, dependency: DependencySpec):
        """Declare an event notify dependency."""

        if not isinstance(dependency, DependencySpec):
            raise TypeError("TileSpec.notify expects a DependencySpec; use D(event, coord, ...)")
        self.notifies.append(dependency)
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
        bounds: tuple[int, int] | None = None,
    ):
        """Register a symbolic integer variable."""

        if name in self.vars:
            raise ValueError(f"Duplicate var: {name}")
        if bounds is not None:
            if (
                not isinstance(bounds, tuple)
                or len(bounds) != 2
                or any(not isinstance(value, int) or isinstance(value, bool) for value in bounds)
            ):
                raise TypeError("var bounds must be a tuple of two integers")
            if bounds[0] <= 0 or bounds[1] <= 0 or bounds[0] > bounds[1]:
                raise ValueError("var bounds must satisfy 0 < min <= max")
        var = VarSpec(name=name, dtype=dtype, bounds=bounds)
        self.vars[name] = var
        return var

    def tensor(self, name: str, shape: ShapeType, dtype: str):
        """Register a logical tensor."""

        if name in self.tensors:
            raise ValueError(f"Duplicate tensor: {name}")
        tensor = TensorSpec(
            name=name, shape=_normalize_shape(shape, f"tensor {name!r}"), dtype=dtype
        )
        self.tensors[name] = tensor
        return tensor

    def event(
        self,
        name: str,
        shape: ShapeType,
        init_count: ExprLike | Callable[..., ExprLike],
        dtype: str = "int32",
        attrs: dict[str, Any] | None = None,
    ):
        """Register a logical event."""

        if name in self.events:
            raise ValueError(f"Duplicate event: {name}")
        event = EventSpec(
            name=name,
            shape=_normalize_shape(shape, f"event {name!r}"),
            init_count=_normalize_event_init_count(init_count),
            dtype=dtype,
            attrs=attrs or {},
        )
        self.events[name] = event
        return event

    def tile(
        self,
        name: str,
        impl: TileImpl,
        grid: GridType,
        reads: list[TensorSpec] | None = None,
        writes: list[TensorSpec] | None = None,
        attrs: dict[str, Any] | None = None,
    ):
        """Register one tile stage."""

        if any(tile.name == name for tile in self.tiles):
            raise ValueError(f"Duplicate tile: {name}")
        if not isinstance(impl, TileImpl):
            raise TypeError("tile impl must be a TileImpl")
        read_tensors = _normalize_accesses(reads or [], "reads")
        write_tensors = _normalize_accesses(writes or [], "writes")
        tile = TileSpec(
            name=name,
            impl=impl,
            grid=_normalize_grid(grid, f"tile {name!r} grid"),
            reads=read_tensors,
            writes=write_tensors,
            attrs=attrs or {},
        )
        self.tiles.append(tile)
        return tile

    def validate(self):
        """Validate this kernel."""

        from tvm.megakernel.transform.validate import validate_kernel

        return validate_kernel(self)

    def lower(self, options=None):
        from tvm.megakernel.transform import lower

        return lower(self, options)


def _normalize_accesses(accesses: list[TensorSpec], label: str) -> list[TensorSpec]:
    tensors: list[TensorSpec] = []
    for access in accesses:
        if not isinstance(access, TensorSpec):
            raise TypeError(f"tile {label} entries must be TensorSpec, got {access!r}")
        tensors.append(access)
    return tensors


def _normalize_shape(shape: ShapeType, label: str) -> tuple[ExprLike, ...]:
    values = tuple(shape) if isinstance(shape, (tuple, list)) else (shape,)
    for dim in values:
        if isinstance(dim, bool) or not isinstance(dim, (int, VarSpec, ExprSpec)):
            raise TypeError(f"{label} shape dims must be int, VarSpec, or ExprSpec")
    return values



def _normalize_grid(grid: GridType, label: str) -> tuple[ExprLike, ExprLike, ExprLike]:
    if not isinstance(grid, (tuple, list)):
        raise TypeError(f"{label} must be a tuple/list of three dimensions")
    values = tuple(grid)
    if len(values) != 3:
        raise ValueError(f"{label} must have exactly three dimensions")
    for dim in values:
        if isinstance(dim, bool) or not isinstance(dim, (int, VarSpec, ExprSpec)):
            raise TypeError(f"{label} dims must be int, VarSpec, or ExprSpec")
    return values


def _normalize_event_init_count(
    init_count: ExprLike | Callable[..., ExprLike],
) -> Callable[..., ExprLike]:
    if isinstance(init_count, bool):
        raise TypeError("event init_count must be an int, VarSpec, ExprSpec, or callable")
    if isinstance(init_count, int):
        if init_count < 0:
            raise ValueError("event init_count must be non-negative")

        def uniform_init_count(*_coord, value=init_count):
            return value

        return uniform_init_count
    if isinstance(init_count, (VarSpec, ExprSpec)):

        def uniform_init_count(*_coord, value=init_count):
            return value

        return uniform_init_count
    if callable(init_count):
        return init_count
    raise TypeError("event init_count must be an int, VarSpec, ExprSpec, or callable")
