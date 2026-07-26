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
"""Validate TileImpl accesses against declared DSL tensor accesses.

Validation performed by this module:

1. Impl tensor binding discovery:
   - TileImpl attributes that reference tensors from the same KernelSpec are
     used to bind generated TIRX buffers back to DSL TensorSpecs.

2. Impl access collection:
   - TileImpl device_init, prefetch, and run hooks are emitted through the TIRX
     parser.
   - Buffer loads, stores, block regions, and tile primitive srcs/dsts are
     collected as actual read/write accesses.
   - Unknown side effects or regions are recorded so validation can report that
     coverage cannot be proven.

3. Declared access coverage:
   - Actual reads must be covered by tile.reads declarations.
   - Actual writes must be covered by tile.writes declarations.
   - A bare declared tensor access covers any actual region of that tensor.
   - Region coverage is checked on deterministic samples for symbolic grids.

4. Validation policy:
   - impl_access_validate="error" raises on collection or coverage failures.
   - impl_access_validate="warn" reports recoverable failures as warnings.
   - impl_access_validate="off" skips TileImpl access validation for that tile.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from functools import reduce
from itertools import product
from operator import mul
import random
import warnings
from dataclasses import dataclass, field
from typing import Any

from tvm.ir import Call, Expr, Range
import tvm.tirx.script as T
from tvm.tirx import Buffer, BufferLoad, BufferRegion, Stmt, TilePrimitiveCall, Var
from tvm.tirx.expr_functor import ExprVisitor
from tvm.tirx.stmt_functor import StmtVisitor

from ...dsl.impl import TileImpl
from ...dsl.spec import (
    ExprSpec,
    KernelSpec,
    RegionRange,
    RegionSpec,
    TensorSpec,
    VarSpec,
    eval_expr_like,
    expr_bounds,
    expr_vars,
)


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


def validate_impl(kernel: KernelSpec) -> None:
    """Validate generated TileImpl accesses against declared tile accesses."""

    tensor_ids = {id(tensor) for tensor in kernel.tensors.values()}
    envs = _sample_kernel_envs(kernel)
    for tile in kernel.tiles:
        policy = tile.attrs.get("impl_access_validate", "error")
        if policy not in ("error", "warn", "off"):
            raise ValueError(
                f"tile {tile.name!r} has unsupported impl_access_validate policy {policy!r}"
            )
        if policy == "off":
            continue
        impl_tensors = _tile_impl_tensor_attrs(tile, tensor_ids)
        if not impl_tensors:
            continue
        for env in envs:
            patches = _patch_impl_expr_attrs(tile.impl, env)
            try:
                actual = collect_impl_access(tile.impl, tensors=impl_tensors, hooks=("prefetch", "run"))
            except Exception as err:
                _handle_impl_access_issue(
                    tile,
                    policy,
                    f"impl access collection failed: {err}",
                    force_error=True,
                    cause=err,
                )
                continue
            finally:
                _restore_impl_attrs(tile.impl, patches)
            for effect in actual.unknown_effects:
                warnings.warn(
                    f"tile {tile.name!r} impl has unknown access effect: {effect}",
                    stacklevel=2,
                )
            _validate_actual_accesses(tile, actual.reads, tile.reads, "read", [env], policy)
            _validate_actual_accesses(tile, actual.writes, tile.writes, "write", [env], policy)


def _handle_impl_access_issue(
    tile,
    policy: str,
    message: str,
    *,
    force_error: bool = False,
    cause: BaseException | None = None,
) -> None:
    full_message = f"tile {tile.name!r} {message}"
    if force_error or policy == "error":
        raise ValueError(full_message) from cause
    warnings.warn(full_message, stacklevel=3)


def _tile_impl_tensor_attrs(tile, tensor_ids: set[int]) -> dict[str, TensorSpec]:
    result: dict[str, TensorSpec] = {}
    for attr_name, value in vars(tile.impl).items():
        if isinstance(value, TensorSpec) and id(value.base_tensor) in tensor_ids:
            result[attr_name] = value.base_tensor
    return result


def _patch_impl_expr_attrs(impl: TileImpl, env: dict[VarSpec, int] | None):
    patches = []
    for attr_name, value in vars(impl).items():
        new_value = _resolve_impl_expr_refs(value, env)
        if new_value is not value:
            patches.append((attr_name, value))
            setattr(impl, attr_name, new_value)
    return patches


def _restore_impl_attrs(impl: TileImpl, patches) -> None:
    for attr_name, value in reversed(patches):
        setattr(impl, attr_name, value)


def _resolve_impl_expr_refs(value: Any, env: dict[VarSpec, int] | None):
    if isinstance(value, (VarSpec, ExprSpec)):
        resolved = eval_expr_like(value, env)
        if resolved is None:
            return value
        return resolved
    if isinstance(value, tuple):
        resolved = tuple(_resolve_impl_expr_refs(item, env) for item in value)
        return value if resolved == value else resolved
    if isinstance(value, list):
        resolved = [_resolve_impl_expr_refs(item, env) for item in value]
        return value if resolved == value else resolved
    if isinstance(value, dict):
        resolved = {
            _resolve_impl_expr_refs(key, env): _resolve_impl_expr_refs(val, env)
            for key, val in value.items()
        }
        return value if resolved == value else resolved
    return value


def _validate_actual_accesses(tile, actual_accesses, declared_accesses, kind: str, envs, policy: str) -> None:
    declared_by_tensor: dict[TensorSpec, list[TensorSpec]] = defaultdict(list)
    for access in declared_accesses:
        declared_by_tensor[access.base_tensor].append(access)

    for actual in actual_accesses:
        tensor = actual.tensor.base_tensor
        candidates = declared_by_tensor.get(tensor, [])
        if not candidates:
            _handle_impl_access_issue(
                tile,
                policy,
                f"impl {kind}s tensor {tensor.name!r} but tile.{kind}s does not declare it",
            )
            continue
        if any(declared.region_from_tile is None for declared in candidates):
            continue
        if actual.tensor.region_from_tile is None:
            _handle_impl_access_issue(
                tile,
                policy,
                f"impl {kind}s tensor {tensor.name!r} with unknown region; "
                "cannot prove declared region coverage",
            )
            continue
        for env in envs:
            tile_extents = _static_int_tuple(tile.grid, env)
            if tile_extents is None:
                continue
            for idx in product(*(range(extent) for extent in tile_extents)):
                actual_region = _region_from_access(actual.tensor, env, *idx)
                declared_regions = [
                    _region_from_access(declared, env, *idx) for declared in candidates
                ]
                if _region_set_covers(declared_regions, actual_region):
                    continue
                _handle_impl_access_issue(
                    tile,
                    policy,
                    f"impl {kind}s tensor {tensor.name!r} region "
                    f"{_region_label(actual_region)} outside declared tile.{kind}s regions",
                )
                break


def _region_set_covers(
    outers: list[RegionSpec | None] | tuple[RegionSpec | None, ...],
    inner: RegionSpec | None,
) -> bool:
    if any(outer is None for outer in outers):
        return True
    if inner is None:
        return False
    if not outers:
        return False
    if any(_region_contains(outer, inner) for outer in outers):
        return True
    if not _static_region(inner):
        return False
    for point in _region_points(inner):
        point_region = RegionSpec(
            dims=tuple(RegionRange(start=value, extent=1) for value in point)
        )
        if not any(_region_contains(outer, point_region) for outer in outers):
            return False
    return True


def _region_contains(outer: RegionSpec | None, inner: RegionSpec | None) -> bool:
    if outer is None:
        return True
    if inner is None:
        return False
    if len(outer.dims) != len(inner.dims):
        return False
    for outer_dim, inner_dim in zip(outer.dims, inner.dims):
        if not all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in (outer_dim.start, outer_dim.extent, inner_dim.start, inner_dim.extent)
        ):
            return False
        outer_end = outer_dim.start + outer_dim.extent
        inner_end = inner_dim.start + inner_dim.extent
        if inner_dim.start < outer_dim.start or inner_end > outer_end:
            return False
    return True


def _static_region(region: RegionSpec | None) -> bool:
    if region is None:
        return False
    return all(
        isinstance(value, int) and not isinstance(value, bool)
        for dim in region.dims
        for value in (dim.start, dim.extent)
    )


def _region_points(region: RegionSpec):
    return product(*(range(dim.start, dim.start + dim.extent) for dim in region.dims))


def _region_from_access(access: TensorSpec, env, m_idx, n_idx, k_idx) -> RegionSpec | None:
    if access.region_from_tile is None:
        return None
    return _resolve_region(_region_from_tile(access.region_from_tile, m_idx, n_idx, k_idx), env)


def _region_from_tile(region_from_tile, m_idx, n_idx, k_idx) -> RegionSpec:
    region = region_from_tile(m_idx, n_idx, k_idx) if callable(region_from_tile) else region_from_tile
    if isinstance(region, RegionSpec):
        return region
    if isinstance(region, (tuple, list)):
        return RegionSpec(dims=tuple(RegionRange(value, 1) for value in region))
    raise TypeError(f"region_from_tile must return RegionSpec, tuple, or list, got {region!r}")


def _resolve_region(region: RegionSpec | None, env: dict[VarSpec, int] | None) -> RegionSpec | None:
    if region is None:
        return None
    return RegionSpec(
        dims=tuple(
            RegionRange(
                start=_resolve_expr_value(dim.start, env),
                extent=_resolve_expr_value(dim.extent, env),
            )
            for dim in region.dims
        )
    )


def _region_label(region: RegionSpec | None) -> str:
    if region is None:
        return "unknown"
    parts = []
    for dim in region.dims:
        if dim.extent == 1:
            parts.append(str(dim.start))
        else:
            parts.append(f"{dim.start}:{dim.start + dim.extent}")
    return "[" + ", ".join(parts) + "]"


def _collect_kernel_vars(kernel: KernelSpec) -> list[VarSpec]:
    seen: set[VarSpec] = set()
    result: list[VarSpec] = []

    def add_from(value: Any) -> None:
        for var in expr_vars(value):
            if var not in seen:
                seen.add(var)
                result.append(var)

    for var in kernel.vars.values():
        add_from(var)
    for tensor in kernel.tensors.values():
        add_from(tensor.shape)
    for event in kernel.events.values():
        add_from(event.shape)
    for tile in kernel.tiles:
        add_from(tile.grid)
    return result


def _sample_kernel_envs(kernel: KernelSpec) -> list[dict[VarSpec, int] | None]:
    vars_seen = list(_collect_kernel_vars(kernel))
    if not vars_seen:
        return [None]

    rng = random.Random(0)
    candidates = []
    for base in (1, 2, 4, 8, 16):
        candidates.append(dict(_sample_value_for_base(var, base) for var in vars_seen))
    bounded_values = [_range_sample_values(var) for var in vars_seen]
    if all(values is not None for values in bounded_values):
        for values in zip(*bounded_values):
            candidates.append(dict(zip(vars_seen, values)))
    for _ in range(5):
        candidates.append({var: _random_sample_value(var, rng) for var in vars_seen})

    unique = []
    seen = set()
    for env in candidates:
        key = tuple((var.name, env[var]) for var in vars_seen)
        if key not in seen:
            seen.add(key)
            unique.append(env)
    return [env for env in unique if _kernel_sample_fits_budget(kernel, env)]


def _kernel_sample_fits_budget(kernel: KernelSpec, env: dict[VarSpec, int]) -> bool:
    budget = 65536
    total = 0
    for tensor in kernel.tensors.values():
        total += _extent_product(_static_int_tuple(tensor.shape, env))
    for event in kernel.events.values():
        total += _extent_product(_static_int_tuple(event.shape, env))
    for tile in kernel.tiles:
        total += _extent_product(_static_int_tuple(tile.grid, env))
    return total <= budget


def _shape_tuple(shape: Any) -> tuple[Any, ...]:
    return tuple(shape) if isinstance(shape, (tuple, list)) else (shape,)


def _static_int_tuple(values: Any, env: dict[VarSpec, int] | None = None) -> tuple[int, ...] | None:
    result = []
    for value in _shape_tuple(values):
        resolved = eval_expr_like(value, env)
        if resolved is None:
            return None
        result.append(resolved)
    return tuple(result)


def _sample_value_for_base(var: VarSpec, base: int) -> tuple[VarSpec, int]:
    if var.bounds is None:
        return var, base
    lo, hi = var.bounds
    return var, min(max(base, lo), hi)


def _range_sample_values(var: VarSpec) -> tuple[int, ...] | None:
    if var.bounds is None:
        return None
    lo, hi = var.bounds
    mid = (lo + hi) // 2
    return tuple(dict.fromkeys((lo, mid, hi)))


def _random_sample_value(var: VarSpec, rng: random.Random) -> int:
    if var.bounds is None:
        return rng.randint(1, 16)
    lo, hi = var.bounds
    return rng.randint(lo, hi)


def _extent_product(extents: tuple[int, ...] | None) -> int:
    if extents is None:
        return 0
    return reduce(mul, extents, 1)


def _resolve_expr_value(value: Any, env: dict[VarSpec, int] | None) -> Any:
    resolved = eval_expr_like(value, env)
    return value if resolved is None else resolved



class _AccessCollector(StmtVisitor):
    def __init__(self, buffer_tensors: dict[str, TensorSpec] | None = None):
        super().__init__()
        self.buffer_tensors = buffer_tensors or {}
        self.access = ImplAccess()

    def visit_expr(self, expr):
        if isinstance(expr, Expr):
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
            if isinstance(value, Expr):
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
        elif isinstance(value, Expr):
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


__all__ = ["Access", "ImplAccess", "collect_impl_access", "validate_impl"]
