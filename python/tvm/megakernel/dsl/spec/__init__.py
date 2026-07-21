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

"""Spec-layer building blocks for the megakernel DSL."""

from .core import (
    CoordMapType,
    DependencySpec,
    EventSpec,
    KernelSpec,
    ShapeType,
    TensorSpec,
    GridType,
    TileSpec,
)
from .expr import ExprLike, ExprSpec, VarSpec, eval_expr_like, expr_bounds, expr_vars
from .region import R, RegionBuilder, RegionRange, RegionSpec, TileRegionMap, TileRegionResult

__all__ = [
    "CoordMapType",
    "DependencySpec",
    "EventSpec",
    "ExprLike",
    "ExprSpec",
    "KernelSpec",
    "R",
    "RegionBuilder",
    "RegionRange",
    "RegionSpec",
    "ShapeType",
    "TensorSpec",
    "GridType",
    "TileSpec",
    "TileRegionMap",
    "TileRegionResult",
    "VarSpec",
    "eval_expr_like",
    "expr_bounds",
    "expr_vars",
]
