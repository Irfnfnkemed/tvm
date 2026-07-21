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
"""Symbolic integer expressions used by the megakernel DSL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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


__all__ = [
    "ExprLike",
    "ExprSpec",
    "VarSpec",
    "eval_expr_like",
    "expr_bounds",
    "expr_vars",
]
