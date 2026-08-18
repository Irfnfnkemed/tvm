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

"""Tensor region helpers for the megakernel DSL spec layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


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


__all__ = [
    "R",
    "RegionBuilder",
    "RegionRange",
    "RegionSpec",
    "TileRegionMap",
    "TileRegionResult",
]
