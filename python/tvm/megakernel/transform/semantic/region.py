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
"""RegionSpec geometry helpers for semantic validation."""

from __future__ import annotations

from itertools import product

from ...dsl import RegionRange, RegionSpec


def regions_overlap(lhs: RegionSpec | None, rhs: RegionSpec | None) -> bool:
    """Return whether two known or unknown regions may overlap."""

    if lhs is None or rhs is None:
        return True
    if len(lhs.dims) != len(rhs.dims):
        return False
    for lhs_dim, rhs_dim in zip(lhs.dims, rhs.dims):
        if not all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in (lhs_dim.start, lhs_dim.extent, rhs_dim.start, rhs_dim.extent)
        ):
            return False
        lhs_end = lhs_dim.start + lhs_dim.extent
        rhs_end = rhs_dim.start + rhs_dim.extent
        if lhs_dim.start >= rhs_end or rhs_dim.start >= lhs_end:
            return False
    return True


def region_set_covers(outers: list[RegionSpec | None] | tuple[RegionSpec | None, ...], inner: RegionSpec | None) -> bool:
    """Return whether a set of regions covers ``inner``.

    This accepts disjoint point/range declarations whose union covers a single
    actual region.  Non-static symbolic regions are treated as not provable.
    """

    if any(outer is None for outer in outers):
        return True
    if inner is None:
        return False
    if not outers:
        return False
    if any(region_contains(outer, inner) for outer in outers):
        return True
    if not static_region(inner):
        return False
    for point in region_points(inner):
        point_region = RegionSpec(dims=tuple(RegionRange(start=value, extent=1) for value in point))
        if not any(region_contains(outer, point_region) for outer in outers):
            return False
    return True


def region_contains(outer: RegionSpec | None, inner: RegionSpec | None) -> bool:
    """Return whether ``outer`` fully contains ``inner``."""

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


def static_region(region: RegionSpec | None) -> bool:
    """Return whether all range bounds are concrete integers."""

    if region is None:
        return False
    return all(
        isinstance(value, int) and not isinstance(value, bool)
        for dim in region.dims
        for value in (dim.start, dim.extent)
    )


def region_points(region: RegionSpec):
    """Enumerate points in a static region."""

    return product(*(range(dim.start, dim.start + dim.extent) for dim in region.dims))


def region_label(region: RegionSpec | None) -> str:
    """Return a compact user-facing label for a region."""

    if region is None:
        return "unknown"
    parts = []
    for dim in region.dims:
        if dim.extent == 1:
            parts.append(str(dim.start))
        else:
            parts.append(f"{dim.start}:{dim.start + dim.extent}")
    return "[" + ", ".join(parts) + "]"


__all__ = [
    "region_contains",
    "region_label",
    "region_points",
    "region_set_covers",
    "regions_overlap",
    "static_region",
]
