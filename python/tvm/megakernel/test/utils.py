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
"""Shared helpers for megakernel tests."""

from __future__ import annotations

from tvm.megakernel.dsl import D, KernelSpec, R, TileImpl
from tvm.megakernel.transform import LoweringOptions, lower
from tvm.megakernel.transform.validate import validate_kernel
from tvm.megakernel.transform.lower.prepare import prepare_lowering_plan


class EmptyTile(TileImpl):
    def run(self, m_idx, n_idx, k_idx):
        pass


def r1(tensor):
    return tensor.region(lambda m, n, k: R[m])


def r2(tensor):
    return tensor.region(lambda m, n, k: R[m, n])


def r2_first_col(tensor):
    return tensor.region(lambda m, n, k: R[m, 0])


def static_options(**attrs):
    base = {"sm_count": 2, "num_threads": 128, "max_tasks": 64, "end_job_id": 31}
    base.update(attrs)
    return LoweringOptions(schedule="static", attrs=base)


def dynamic_options(**attrs):
    base = {"sm_count": 2, "num_threads": 128, "max_tasks": 64, "end_job_id": 31}
    base.update(attrs)
    return LoweringOptions(schedule="dynamic", attrs=base)


def simple_static_kernel(name="static_simple"):
    kernel = KernelSpec(name)
    source = kernel.tensor("source", (8,), "float32")
    tmp = kernel.tensor("tmp", (8,), "float32")
    out = kernel.tensor("out", (8,), "float32")
    ready = kernel.event("ready", (1,), init_count=1)
    kernel.tile("producer", EmptyTile(), (1, 1, 1), reads=[source], writes=[tmp]).notify(
        D(ready, lambda m, n, k, i: (1, -1, 0))
    )
    kernel.tile("consumer", EmptyTile(), (1, 1, 1), reads=[tmp], writes=[out]).wait(
        D(ready, lambda m, n, k, i: (1, -1, 0))
    )
    return kernel


def simple_dynamic_kernel(name="dynamic_simple"):
    kernel = KernelSpec(name)
    source = kernel.tensor("source", (8,), "float32")
    tmp = kernel.tensor("tmp", (8,), "float32")
    out = kernel.tensor("out", (8,), "float32")
    ready = kernel.event("ready", (1,), init_count=1)
    kernel.tile("producer", EmptyTile(), (1, 1, 1), reads=[source], writes=[tmp]).notify(
        D(ready, lambda m, n, k, i: (1, -1, 0))
    )
    kernel.tile("consumer", EmptyTile(), (1, 1, 1), reads=[tmp], writes=[out]).wait(
        D(ready, lambda m, n, k, i: (1, -1, 0), inv_coord=lambda rank, e, i: (1, 0, 0, 0))
    )
    return kernel


def dynamic_plan(kernel, options=None):
    validate_kernel(kernel)
    return prepare_lowering_plan(kernel, options or dynamic_options())


def lowered_text(kernel, options):
    return str(lower(kernel, options))
