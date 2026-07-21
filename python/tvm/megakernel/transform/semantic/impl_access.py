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
"""Collect read/write tensor regions from generated TileImpl TIRX."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from tvm.ir import Range
import tvm.tirx.script as T
from tvm.tirx import Buffer, BufferLoad, BufferRegion, Call, PrimExpr, Stmt, TilePrimitiveCall, Var
from tvm.tirx.expr_functor import ExprVisitor
from tvm.tirx.stmt_functor import StmtVisitor

from ...dsl import RegionRange, RegionSpec, TensorSpec, TileImpl, expr_bounds


@dataclass(frozen=True)
class Access:
    """One semantic tensor access observed in generated TileImpl TIRX."""

    tensor: TensorSpec
    kind: str
    source: str


@dataclass(frozen=True)
class RawAccess:
    """One raw buffer access observed while visiting TIRX."""

    buffer: Buffer
    region: tuple[Range, ...]
    kind: str
    source: str


@dataclass
class ImplAccess:
    """Read/write summary for one or more generated hook statements."""

    reads: list[Access] = field(default_factory=list)
    writes: list[Access] = field(default_factory=list)
    unknown_effects: list[str] = field(default_factory=list)

    def extend(self, other: "ImplAccess") -> None:
        self.reads.extend(other.reads)
        self.writes.extend(other.writes)
        self.unknown_effects.extend(other.unknown_effects)


class _AccessCollector(StmtVisitor):
    def __init__(self, buffer_tensors: dict[str, TensorSpec] | None = None):
        super().__init__()
        self.buffer_tensors = buffer_tensors or {}
        self.access = ImplAccess()

    def visit_expr(self, expr):
        if isinstance(expr, PrimExpr):
            _ReadExprCollector(self.access, self.buffer_tensors).visit_expr(expr)

    def visit_buffer_store_(self, op):
        self._record_point(op.buffer, op.indices, "write", "BufferStore")
        self.visit_expr(op.value)
        for index in op.indices:
            self.visit_expr(index)
        if op.predicate is not None:
            self.visit_expr(op.predicate)

    def visit_block_(self, op):
        for region in op.reads:
            self._record_region(region, "read", "Block.reads")
        for region in op.writes:
            self._record_region(region, "write", "Block.writes")
        super().visit_block_(op)

    def visit_op_call_(self, op):
        op = TilePrimitiveCall.downcast(op)
        if _is_compose_op(op):
            self.access.unknown_effects.append("generic compose_op has unknown access effects")
            return
        try:
            srcs = op.srcs
            dsts = op.dsts
        except NotImplementedError as err:
            self.access.unknown_effects.append(
                f"tile primitive {getattr(op, 'op', None)} has unknown access effects: {err}"
            )
            return
        for i, src in enumerate(srcs):
            self._record_value(src, "read", f"{_op_name(op)}.srcs[{i}]")
        for i, dst in enumerate(dsts):
            self._record_value(dst, "write", f"{_op_name(op)}.dsts[{i}]")
        for value in op.config.values():
            if isinstance(value, PrimExpr):
                self.visit_expr(value)

    def visit_evaluate_(self, op):
        if _expr_has_unknown_side_effect(op.value):
            self.access.unknown_effects.append(f"opaque call in Evaluate: {op.value}")
        self.visit_expr(op.value)

    def _record_value(self, value: Any, kind: str, source: str) -> None:
        if isinstance(value, BufferRegion):
            self._record_region(value, kind, source)
        elif isinstance(value, BufferLoad):
            self._record_point(value.buffer, value.indices, "read", source)
        elif isinstance(value, Buffer):
            self._record_region(_full_region(value), kind, source)
        elif isinstance(value, PrimExpr):
            self.visit_expr(value)

    def _record_region(self, region: BufferRegion, kind: str, source: str) -> None:
        self._record_raw_access(RawAccess(region.buffer, tuple(region.region), kind, source))

    def _record_point(self, buffer: Buffer, indices: list[Any], kind: str, source: str) -> None:
        region = tuple(Range.from_min_extent(index, 1) for index in indices)
        self._record_raw_access(RawAccess(buffer, region, kind, source))

    def _record_raw_access(self, raw_access: RawAccess) -> None:
        tensor = self.buffer_tensors.get(_buffer_name(raw_access.buffer))
        if tensor is None:
            return
        semantic_access = _semantic_access(tensor, raw_access, self.access)
        (self.access.reads if raw_access.kind == "read" else self.access.writes).append(
            semantic_access
        )


class _ReadExprCollector(ExprVisitor):
    def __init__(self, access: ImplAccess, buffer_tensors: dict[str, TensorSpec] | None = None):
        super().__init__()
        self.access = access
        self.buffer_tensors = buffer_tensors or {}

    def visit_buffer_load_(self, op):
        region = tuple(Range.from_min_extent(index, 1) for index in op.indices)
        raw_access = RawAccess(op.buffer, region, "read", "BufferLoad")
        tensor = self.buffer_tensors.get(_buffer_name(op.buffer))
        if tensor is not None:
            self.access.reads.append(_semantic_access(tensor, raw_access, self.access))
        super().visit_buffer_load_(op)


def collect_impl_access(
    source: Stmt | list[Stmt] | tuple[Stmt, ...] | type[TileImpl] | Callable[..., TileImpl],
    *,
    tensors: dict[str, TensorSpec] | None = None,
    buffer_specs: dict[str, tuple[Any, str]] | None = None,
    hooks: str | tuple[str, ...] = ("prefetch", "run"),
) -> ImplAccess:
    """Collect semantic read/write regions from generated TIRX or a ``TileImpl`` hook.

    The main TileImpl path takes ``tensors``, mapping TileImpl attribute or
    constructor keyword names to ``TensorSpec`` objects.  Those tensors are used
    both to build parser buffers and to bind observed buffer regions back to DSL
    tensor regions.  ``buffer_specs`` is kept for statement-level or unbound
    debugging use.
    """

    if isinstance(source, (list, tuple)):
        result = ImplAccess()
        for stmt in source:
            result.extend(collect_impl_access(stmt))
        return result
    if isinstance(source, Stmt):
        return _collect_stmt_access(source)
    if tensors is None and buffer_specs is None:
        raise TypeError("TileImpl access collection requires tensors or buffer_specs")
    hook_names = (hooks,) if isinstance(hooks, str) else hooks
    return _collect_tile_impl_access(source, tensors or {}, buffer_specs or {}, hook_names)


def _collect_stmt_access(stmt: Stmt, buffer_tensors: dict[str, TensorSpec] | None = None) -> ImplAccess:
    collector = _AccessCollector(buffer_tensors)
    collector.visit_stmt(stmt)
    return collector.access


class _DefaultSmemManager:
    def alloc(self, shape, dtype="float32", scope="shared.dyn", **kwargs):
        del kwargs
        return T.alloc_buffer(shape, dtype, scope=scope)

    def wait_all(self, level="cta"):
        del level

    def release_all(self, level="cta"):
        del level

    def advance(self):
        pass


class _TileImplEmitter:
    def __init__(
        self,
        impl_factory: type[TileImpl] | Callable[..., TileImpl],
        tensors: dict[str, TensorSpec],
        buffer_specs: dict[str, tuple[Any, str]],
        hooks: tuple[str, ...],
    ):
        self.impl_factory = impl_factory
        self.tensors = tensors
        self.buffer_specs = buffer_specs
        self.hooks = hooks
        self.buffer_tensors: dict[str, TensorSpec] = {}

    def emit(self):
        buffers = {}
        for name, tensor in self.tensors.items():
            buffer = T.arg(name, T.Buffer(_buffer_shape(tensor.shape), tensor.dtype))
            buffers[name] = buffer
            self.buffer_tensors[_buffer_name(buffer)] = tensor.base_tensor
        for name, (shape, dtype) in self.buffer_specs.items():
            if name in buffers:
                raise ValueError(f"duplicate TileImpl buffer name {name!r}")
            buffers[name] = T.arg(name, T.Buffer(_buffer_shape(shape), dtype))
        T.device_entry()
        tile, patches = self._create_tile(buffers)
        try:
            m_idx = Var("m_idx", "int32")
            n_idx = Var("n_idx", "int32")
            k_idx = Var("k_idx", "int32")
            tile.device_init(_DefaultSmemManager(), m_idx, n_idx, k_idx)
            for hook in self.hooks:
                getattr(tile, hook)(m_idx, n_idx, k_idx)
        finally:
            for attr_name, old_value in reversed(patches):
                setattr(tile, attr_name, old_value)

    def _create_tile(self, buffers):
        if isinstance(self.impl_factory, TileImpl):
            tile = self.impl_factory
            patches = []
            for attr_name, buffer in buffers.items():
                if hasattr(tile, attr_name):
                    patches.append((attr_name, getattr(tile, attr_name)))
                    setattr(tile, attr_name, buffer)
            return tile, patches
        return self.impl_factory(**buffers), []


@T.jit(check_well_formed=False)
def _tile_impl_access_entry(*, emitter: T.constexpr):
    emitter.emit()


def _collect_tile_impl_access(
    impl_factory: type[TileImpl] | Callable[..., TileImpl],
    tensors: dict[str, TensorSpec],
    buffer_specs: dict[str, tuple[Any, str]],
    hooks: tuple[str, ...],
) -> ImplAccess:
    emitter = _TileImplEmitter(impl_factory, tensors, buffer_specs, hooks)
    stmt = _tile_impl_access_entry.specialize(emitter=emitter).body
    return _collect_stmt_access(stmt, emitter.buffer_tensors)


def _buffer_name(buffer: Buffer) -> str:
    return str(buffer.name or "")


def _semantic_access(tensor: TensorSpec, raw_access: RawAccess, access: ImplAccess) -> Access:
    region_from_tile = _region_from_tile_from_ranges(raw_access.region)
    if region_from_tile is None:
        access.unknown_effects.append(f"cannot normalize region from {raw_access.source}")
        return Access(tensor.base_tensor, raw_access.kind, raw_access.source)
    return Access(tensor.region(region_from_tile), raw_access.kind, raw_access.source)


def _region_from_tile_from_ranges(ranges: tuple[Range, ...]):
    dim_funcs = []
    for dim in ranges:
        start_func = _expr_func(dim.min)
        extent_func = _expr_func(dim.extent)
        if start_func is None or extent_func is None:
            return None
        dim_funcs.append((start_func, extent_func))

    def region_from_tile(m_idx, n_idx, k_idx):
        return RegionSpec(
            dims=tuple(
                RegionRange(start=start(m_idx, n_idx, k_idx), extent=extent(m_idx, n_idx, k_idx))
                for start, extent in dim_funcs
            )
        )

    return region_from_tile


def _expr_func(expr: Any):
    value = _int_value(expr)
    if value is not None:
        return lambda m_idx, n_idx, k_idx, value=value: value
    if isinstance(expr, Var):
        if expr.name == "m_idx":
            return lambda m_idx, n_idx, k_idx: m_idx
        if expr.name == "n_idx":
            return lambda m_idx, n_idx, k_idx: n_idx
        if expr.name == "k_idx":
            return lambda m_idx, n_idx, k_idx: k_idx
        return None
    name = type(expr).__name__
    if name in ("Add", "Sub", "Mul", "FloorDiv", "FloorMod"):
        lhs = _expr_func(expr.a)
        rhs = _expr_func(expr.b)
        if lhs is None or rhs is None:
            return None
        if name == "Add":
            return lambda m_idx, n_idx, k_idx: lhs(m_idx, n_idx, k_idx) + rhs(m_idx, n_idx, k_idx)
        if name == "Sub":
            return lambda m_idx, n_idx, k_idx: lhs(m_idx, n_idx, k_idx) - rhs(m_idx, n_idx, k_idx)
        if name == "Mul":
            return lambda m_idx, n_idx, k_idx: lhs(m_idx, n_idx, k_idx) * rhs(m_idx, n_idx, k_idx)
        if name == "FloorDiv":
            return lambda m_idx, n_idx, k_idx: lhs(m_idx, n_idx, k_idx) // rhs(m_idx, n_idx, k_idx)
        if name == "FloorMod":
            return lambda m_idx, n_idx, k_idx: lhs(m_idx, n_idx, k_idx) % rhs(m_idx, n_idx, k_idx)
    return None


def _int_value(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raw_value = getattr(value, "value", None)
    if isinstance(raw_value, int) and not isinstance(raw_value, bool):
        return raw_value
    return None


def _buffer_shape(shape: Any) -> tuple[int, ...]:
    values = shape if isinstance(shape, (tuple, list)) else (shape,)
    result = []
    for extent in values:
        bounds = expr_bounds(extent, require_bounded=True)
        if bounds is None:
            raise ValueError(f"buffer shape extent {extent!r} is not bounded")
        result.append(bounds[1])
    return tuple(result)


def _full_region(buffer: Buffer) -> BufferRegion:
    return BufferRegion(buffer, [Range.from_min_extent(0, extent) for extent in buffer.shape])


def _is_compose_op(op: TilePrimitiveCall) -> bool:
    return str(getattr(op, "op", "")).endswith("compose_op")


def _op_name(op: TilePrimitiveCall) -> str:
    name = str(getattr(op, "op", "TilePrimitiveCall"))
    return name.removeprefix("Op(tirx.tile.").removesuffix(")")


def _expr_has_unknown_side_effect(expr: Any) -> bool:
    if not isinstance(expr, Call):
        return False
    op_name = str(getattr(expr, "op", ""))
    if "call_extern" in op_name and expr.args:
        func_name = str(expr.args[0]).strip('"')
        if func_name.startswith("tirx.megakernel.") and func_name.endswith(".marker"):
            return False
        return True
    return "call_packed" in op_name


__all__ = ["Access", "ImplAccess", "collect_impl_access"]
