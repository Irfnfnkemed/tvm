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
    """Symbolic variable used in shapes, tile counts, and event counts.

    ``range`` is an optional inclusive ``(min, max)`` bound.  Lowering uses
    the upper bound when a symbolic shape must reserve static storage, such
    as event workspace layout.
    """

    name: str
    dtype: str = "int32"
    range: tuple[int, int] | None = None

    def __add__(self, other):
        return _binary_expr("add", self, other)

    def __radd__(self, other):
        return _binary_expr("add", other, self)

    def __sub__(self, other):
        return _binary_expr("sub", self, other)

    def __rsub__(self, other):
        return _binary_expr("sub", other, self)

    def __mul__(self, other):
        return _binary_expr("mul", self, other)

    def __rmul__(self, other):
        return _binary_expr("mul", other, self)

    def __floordiv__(self, other):
        return _binary_expr("floordiv", self, other)

    def __rfloordiv__(self, other):
        return _binary_expr("floordiv", other, self)

    def __mod__(self, other):
        return _binary_expr("mod", self, other)

    def __rmod__(self, other):
        return _binary_expr("mod", other, self)

    def __neg__(self):
        return ExprSpec("neg", (self,))

    def ceildiv(self, other):
        return _binary_expr("ceildiv", self, other)


@dataclass(frozen=True)
class ExprSpec:
    """Small integer expression over ``VarSpec`` and integer constants."""

    op: str
    args: tuple["ExprLike", ...]

    def __add__(self, other):
        return _binary_expr("add", self, other)

    def __radd__(self, other):
        return _binary_expr("add", other, self)

    def __sub__(self, other):
        return _binary_expr("sub", self, other)

    def __rsub__(self, other):
        return _binary_expr("sub", other, self)

    def __mul__(self, other):
        return _binary_expr("mul", self, other)

    def __rmul__(self, other):
        return _binary_expr("mul", other, self)

    def __floordiv__(self, other):
        return _binary_expr("floordiv", self, other)

    def __rfloordiv__(self, other):
        return _binary_expr("floordiv", other, self)

    def __mod__(self, other):
        return _binary_expr("mod", self, other)

    def __rmod__(self, other):
        return _binary_expr("mod", other, self)

    def __neg__(self):
        return ExprSpec("neg", (self,))

    def ceildiv(self, other):
        return _binary_expr("ceildiv", self, other)


ExprLike = int | VarSpec | ExprSpec
ShapeType = ExprLike | tuple[ExprLike, ...] | list[ExprLike]
CoordMapType = Callable[[int, int, int], tuple[int, ...]] | tuple[int, ...] | list[int]
TileNumType = tuple[ExprLike, ExprLike, ExprLike] | list[ExprLike]


def _as_expr_like(value: Any) -> ExprLike:
    if isinstance(value, bool):
        raise TypeError("boolean values are not valid megakernel expressions")
    if isinstance(value, (int, VarSpec, ExprSpec)):
        return value
    raise TypeError(f"megakernel expression operands must be int, VarSpec, or ExprSpec, got {value!r}")


def _binary_expr(op: str, lhs: Any, rhs: Any) -> ExprSpec:
    return ExprSpec(op, (_as_expr_like(lhs), _as_expr_like(rhs)))


def expr_vars(value: Any) -> tuple[VarSpec, ...]:
    """Return VarSpec leaves referenced by an expression-like value."""

    result: list[VarSpec] = []
    seen: set[VarSpec] = set()

    def visit(item: Any) -> None:
        if isinstance(item, VarSpec):
            if item not in seen:
                seen.add(item)
                result.append(item)
        elif isinstance(item, ExprSpec):
            for arg in item.args:
                visit(arg)
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)

    visit(value)
    return tuple(result)


def eval_expr_like(value: Any, env: dict[VarSpec, int] | None) -> int | None:
    """Evaluate an expression with a concrete VarSpec environment."""

    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, VarSpec):
        return None if env is None or value not in env else env[value]
    if isinstance(value, ExprSpec):
        args = [eval_expr_like(arg, env) for arg in value.args]
        if any(arg is None for arg in args):
            return None
        return _eval_expr_op(value.op, args)
    return None


def _eval_expr_op(op: str, args: list[int]) -> int:
    if op == "add":
        return args[0] + args[1]
    if op == "sub":
        return args[0] - args[1]
    if op == "mul":
        return args[0] * args[1]
    if op == "floordiv":
        return args[0] // args[1]
    if op == "mod":
        return args[0] % args[1]
    if op == "neg":
        return -args[0]
    if op == "ceildiv":
        return (args[0] + args[1] - 1) // args[1]
    raise ValueError(f"unsupported ExprSpec op {op!r}")


def expr_bounds(value: Any, require_bounded: bool = True) -> tuple[int, int] | None:
    """Return conservative inclusive integer bounds for an expression."""

    if isinstance(value, int) and not isinstance(value, bool):
        return (value, value)
    if isinstance(value, VarSpec):
        if value.range is None:
            if require_bounded:
                raise ValueError(f"symbolic VarSpec({value.name!r}) without a range")
            return None
        return value.range
    if not isinstance(value, ExprSpec):
        if require_bounded:
            raise TypeError(f"expression must be an int, VarSpec, or ExprSpec, got {value!r}")
        return None
    arg_bounds = [expr_bounds(arg, require_bounded=require_bounded) for arg in value.args]
    if any(bound is None for bound in arg_bounds):
        return None
    return _expr_bounds_op(value.op, arg_bounds)


def _expr_bounds_op(op: str, bounds: list[tuple[int, int]]) -> tuple[int, int]:
    if op == "add":
        return (bounds[0][0] + bounds[1][0], bounds[0][1] + bounds[1][1])
    if op == "sub":
        return (bounds[0][0] - bounds[1][1], bounds[0][1] - bounds[1][0])
    if op == "mul":
        products = [a * b for a in bounds[0] for b in bounds[1]]
        return (min(products), max(products))
    if op == "floordiv":
        lo_rhs, hi_rhs = bounds[1]
        if lo_rhs <= 0 <= hi_rhs:
            raise ValueError("floordiv expression divisor range must not include zero")
        values = [a // b for a in bounds[0] for b in bounds[1]]
        return (min(values), max(values))
    if op == "mod":
        lo_rhs, hi_rhs = bounds[1]
        if lo_rhs <= 0 <= hi_rhs:
            raise ValueError("mod expression divisor range must not include zero")
        max_abs = max(abs(lo_rhs), abs(hi_rhs))
        return (0, max_abs - 1)
    if op == "neg":
        return (-bounds[0][1], -bounds[0][0])
    if op == "ceildiv":
        lo_rhs, hi_rhs = bounds[1]
        if lo_rhs <= 0 <= hi_rhs:
            raise ValueError("ceildiv expression divisor range must not include zero")
        values = [(a + b - 1) // b for a in bounds[0] for b in bounds[1]]
        return (min(values), max(values))
    raise ValueError(f"unsupported ExprSpec op {op!r}")


@dataclass(frozen=True)
class RegionRange:
    """One BufferRegion-style dimension: half-open ``[start, start + extent)``."""

    start: Any
    extent: Any


@dataclass(frozen=True)
class RegionSpec:
    """Logical tensor region, normalized to BufferRegion-style ranges."""

    dims: tuple[RegionRange, ...] = ()
    dynamic: bool = False
    reason: str = ""


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
RegionMapType = Callable[[int, int, int], RegionSpec] | RegionSpec


@dataclass(frozen=True)
class TensorSpec:
    """Logical tensor or buffer in the megakernel spec."""

    name: str
    shape: ShapeType
    dtype: str
    region_map: RegionMapType | None = field(default=None, compare=False)
    region_dynamic: bool = field(default=False, compare=False)
    region_reason: str = field(default="", compare=False)
    base: "TensorSpec | None" = field(default=None, compare=False)

    @property
    def base_tensor(self) -> "TensorSpec":
        """Return the registered tensor behind a region access view."""

        return self.base if self.base is not None else self

    @property
    def has_region(self) -> bool:
        """Whether this tensor value represents a region access."""

        return self.region_map is not None or self.region_dynamic

    def region(
        self,
        region_map: RegionMapType | None = None,
        *,
        dynamic: bool = False,
        reason: str = "",
    ) -> "TensorSpec":
        """Return a TensorSpec access view with region metadata."""

        if dynamic:
            if region_map is not None:
                raise ValueError("dynamic tensor region cannot also provide a region_map")
            base = self.base_tensor
            return TensorSpec(
                name=base.name,
                shape=base.shape,
                dtype=base.dtype,
                region_dynamic=True,
                region_reason=reason,
                base=base,
            )
        if region_map is None:
            raise ValueError("tensor.region requires a region_map unless dynamic=True")
        base = self.base_tensor
        return TensorSpec(
            name=base.name,
            shape=base.shape,
            dtype=base.dtype,
            region_map=region_map,
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


DependencyType = tuple[EventSpec, CoordMapType]
TensorAccessType = TensorSpec


@dataclass
class TileSpec:
    """One logical tile stage in the megakernel spec."""

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
        reads: list[TensorAccessType] | None = None,
        writes: list[TensorAccessType] | None = None,
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


def _normalize_accesses(accesses: list[TensorAccessType], label: str) -> list[TensorSpec]:
    tensors: list[TensorSpec] = []
    for access in accesses:
        if not isinstance(access, TensorSpec):
            raise TypeError(f"tile {label} entries must be TensorSpec, got {access!r}")
        tensors.append(access)
    return tensors
