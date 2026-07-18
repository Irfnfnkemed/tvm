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
"""Build semantic plans from megakernel DSL specs."""

from __future__ import annotations

from typing import Any

from ...dsl import EventSpec, KernelSpec, VarSpec, expr_vars
from .model import LogicalEdge, SemanticPlan


def build_semantic_plan(kernel: KernelSpec) -> SemanticPlan:
    """Create a semantic view of the logical DSL graph."""

    return SemanticPlan(
        kernel=kernel,
        vars=tuple(_collect_kernel_vars(kernel)),
        tensors=tuple(kernel.tensors.values()),
        events=tuple(kernel.events.values()),
        tiles=tuple(kernel.tiles),
        logical_edges=logical_edges(kernel),
    )


def logical_edges(kernel: KernelSpec) -> tuple[LogicalEdge, ...]:
    """Return logical event edges in stable event/tile order."""

    producers: dict[int, list] = {id(event): [] for event in kernel.events.values()}
    consumers: dict[int, list] = {id(event): [] for event in kernel.events.values()}
    for tile in kernel.tiles:
        for event, _ in tile.notifies:
            if tile not in producers.setdefault(id(event), []):
                producers[id(event)].append(tile)
        for event, _ in tile.waits:
            if tile not in consumers.setdefault(id(event), []):
                consumers[id(event)].append(tile)

    edges: list[LogicalEdge] = []
    for event in kernel.events.values():
        for producer in producers.get(id(event), []):
            for consumer in consumers.get(id(event), []):
                edges.append(LogicalEdge(event, producer, consumer))
    return tuple(edges)

def _collect_kernel_vars(kernel: KernelSpec) -> list[VarSpec]:
    seen: set[VarSpec] = set()
    result: list[VarSpec] = []

    def add_from(value: Any) -> None:
        for var in expr_vars(value):
            if var not in seen:
                seen.add(var)
                result.append(var)

    for var in getattr(kernel, "vars", {}).values():
        add_from(var)
    for tensor in kernel.tensors.values():
        add_from(tensor.shape)
    for event in kernel.events.values():
        add_from(event.shape)
    for tile in kernel.tiles:
        add_from(tile.tile_num)
    return result


def event_init_count(event: EventSpec, coord: tuple[int, ...]) -> int:
    """Evaluate one logical event init count in semantic validation."""

    return event.init_count(coord) if callable(event.init_count) else event.init_count


__all__ = ["build_semantic_plan", "event_init_count", "logical_edges"]
