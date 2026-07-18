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
"""Semantic megakernel plans derived from the logical DSL."""

from __future__ import annotations

from dataclasses import dataclass

from ...dsl import EventSpec, KernelSpec, TensorSpec, TileSpec, VarSpec


@dataclass(frozen=True)
class LogicalEdge:
    """One logical event dependency from a producer tile to a consumer tile."""

    event: EventSpec
    producer: TileSpec
    consumer: TileSpec

    @property
    def key(self) -> tuple[int, str, str]:
        return (id(self.event), self.producer.name, self.consumer.name)


@dataclass(frozen=True)
class SemanticPlan:
    """Backend-independent meaning of a megakernel DSL graph."""

    kernel: KernelSpec
    vars: tuple[VarSpec, ...]
    tensors: tuple[TensorSpec, ...]
    events: tuple[EventSpec, ...]
    tiles: tuple[TileSpec, ...]
    logical_edges: tuple[LogicalEdge, ...]


__all__ = ["LogicalEdge", "SemanticPlan"]
