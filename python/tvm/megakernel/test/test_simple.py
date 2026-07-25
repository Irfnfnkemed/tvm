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
"""CUDA tests for one simple staged megakernel workload."""

from __future__ import annotations

import warnings


import pytest
import torch
import tvm
import tvm.tirx.script as T
from tvm.tirx.script import tile as Tx

from tvm.megakernel.dsl import D, KernelSpec, R, TileImpl
from tvm.megakernel.transform import LoweringOptions, lower


M = 8
G = 4
K = 8
SM_COUNT = 17
MAX_TASKS = 256
NUM_THREADS = 256

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for megakernel e2e tests"
)


def _torch_simple_reference(source, bias):
    """Torch reference for the staged simple megakernel workload."""
    partial = source.sum(dim=2)
    biased = partial + bias[None, :]
    return biased.sum(dim=1)


def _options(schedule):
    return LoweringOptions(
        schedule=schedule,
        smem_max_bytes=64 * 1024,
        smem_chunk_size=16 * 1024,
        attrs={
            "sm_count": SM_COUNT,
            "num_threads": NUM_THREADS,
            "max_tasks": MAX_TASKS,
            "end_job_id": 31,
        },
    )


class ReduceTile(TileImpl):
    def __init__(self, source, partial, cols):
        super().__init__()
        self.source = source
        self.partial = partial
        self.cols = cols
        self.smem_manager = None
        self.smem = None
        self.reduced = None

    def _declare_resources(self, smem_manager):
        self.smem_manager = smem_manager
        self.smem = smem_manager.alloc((self.cols,), "float32", policy="shared")
        self.reduced = smem_manager.alloc((1,), "float32", policy="shared")

    @T.inline
    def device_init(self, smem_manager, m_idx, n_idx, k_idx):
        self._declare_resources(smem_manager)

    @T.inline
    def run(self, m_idx, n_idx, k_idx):
        self.smem_manager.wait_all("cta")
        Tx.copy(self.smem[0:self.cols], self.source[m_idx, n_idx, 0:self.cols])
        Tx.sum(self.reduced, self.smem[0:self.cols])
        tid = T.thread_id([NUM_THREADS])
        if tid == 0:
            self.partial[m_idx, n_idx] = self.reduced[0]
        self.smem_manager.release_all("cta")
        self.smem_manager.advance()


class BiasTile(TileImpl):
    def __init__(self, partial, bias, biased):
        super().__init__()
        self.partial = partial
        self.bias = bias
        self.biased = biased
        self.smem_manager = None
        self.prefetched_bias = None

    def _declare_resources(self, smem_manager):
        self.smem_manager = smem_manager
        self.prefetched_bias = smem_manager.alloc((1,), "float32", policy="shared")

    @T.inline
    def device_init(self, smem_manager, m_idx, n_idx, k_idx):
        self._declare_resources(smem_manager)

    @T.inline
    def prefetch(self, m_idx, n_idx, k_idx):
        self.smem_manager.wait_all("cta")
        tid = T.thread_id([NUM_THREADS])
        if tid == 0:
            self.prefetched_bias[0] = self.bias[n_idx]
        T.cuda.cta_sync()

    @T.inline
    def run(self, m_idx, n_idx, k_idx):
        tid = T.thread_id([NUM_THREADS])
        if tid == 0:
            self.biased[m_idx, n_idx] = self.partial[m_idx, n_idx] + self.prefetched_bias[0]
        self.smem_manager.release_all("cta")
        self.smem_manager.advance()


class FinalReduceTile(TileImpl):
    def __init__(self, biased, output, groups):
        super().__init__()
        self.biased = biased
        self.output = output
        self.groups = groups
        self.smem_manager = None
        self.smem = None
        self.reduced = None

    def _declare_resources(self, smem_manager):
        self.smem_manager = smem_manager
        self.smem = smem_manager.alloc((self.groups,), "float32", policy="shared")
        self.reduced = smem_manager.alloc((1,), "float32", policy="shared")

    @T.inline
    def device_init(self, smem_manager, m_idx, n_idx, k_idx):
        self._declare_resources(smem_manager)

    @T.inline
    def run(self, m_idx, n_idx, k_idx):
        self.smem_manager.wait_all("cta")
        Tx.copy(self.smem[0:self.groups], self.biased[m_idx, 0:self.groups])
        Tx.sum(self.reduced, self.smem[0:self.groups])
        tid = T.thread_id([NUM_THREADS])
        if tid == 0:
            self.output[m_idx] = self.reduced[0]
        self.smem_manager.release_all("cta")
        self.smem_manager.advance()


class EndpointTile(TileImpl):
    @T.inline
    def run(self, m_idx, n_idx, k_idx):
        T.evaluate(0)


def _maybe_inv(dynamic, inv_coord):
    return inv_coord if dynamic else None


def _build_simple_kernel(name, *, dynamic):
    kernel = KernelSpec(name)
    rows = kernel.var("rows", bounds=(M // 2, M * 2))
    groups = kernel.var("groups", bounds=(G // 2, G * 2))
    cols = kernel.var("cols", bounds=(K // 2, K * 2))

    source = kernel.tensor("A", (rows, groups, cols), "float32")
    bias = kernel.tensor("bias", (groups,), "float32")
    partial = kernel.tensor("partial", (rows, groups), "float32")
    biased = kernel.tensor("biased", (rows, groups), "float32")
    output = kernel.tensor("output", (rows,), "float32")

    partial_ready = kernel.event("partial_ready", (rows, groups), init_count=1)
    row_ready = kernel.event("row_ready", (rows,), init_count=groups)
    done = kernel.event("done", (1,), init_count=rows)

    kernel.tile(
        "reduce",
        ReduceTile(source, partial, cols),
        (rows, groups, 1),
        reads=[source.region(lambda m, n, k: R[m, n, 0:cols])],
        writes=[partial.region(lambda m, n, k: R[m, n])],
    ).notify(D(partial_ready, lambda m, n, k, i: (1, -1, m, n)))

    kernel.tile(
        "bias",
        BiasTile(partial, bias, biased),
        (rows, groups, 1),
        reads=[
            partial.region(lambda m, n, k: R[m, n]),
            bias.region(lambda m, n, k: R[n]),
        ],
        writes=[biased.region(lambda m, n, k: R[m, n])],
    ).wait(
        D(
            partial_ready,
            lambda m, n, k, i: (1, -1, m, n),
            inv_coord=_maybe_inv(dynamic, lambda rank, m, n, i: (1, m, n, 0)),
        )
    ).notify(D(row_ready, lambda m, n, k, i: (1, -1, m)))

    kernel.tile(
        "final",
        FinalReduceTile(biased, output, groups),
        (rows, 1, 1),
        reads=[biased.region(lambda m, n, k: R[m, 0:groups])],
        writes=[output.region(lambda m, n, k: R[m])],
    ).wait(
        D(
            row_ready,
            lambda m, n, k, i: (1, -1, m),
            inv_coord=_maybe_inv(dynamic, lambda rank, m, i: (1, m, 0, 0)),
        )
    ).notify(D(done, lambda m, n, k, i: (1, -1, 0)))

    kernel.tile("endpoint", EndpointTile(), (1, 1, 1)).wait(
        D(
            done,
            lambda m, n, k, i: (1, -1, 0),
            inv_coord=_maybe_inv(dynamic, lambda rank, e, i: (1, 0, 0, 0)),
        )
    )
    return kernel


def _compile_kernel(kernel, options):
    mod = lower(kernel, options)
    with tvm.target.Target("cuda"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return tvm.compile(mod, target="cuda", tir_pipeline="tirx")


def _event_workspace():
    max_rows = M * 2
    max_groups = G * 2
    return torch.empty((max_rows * max_groups + max_rows + 1 + 1,), dtype=torch.int32, device="cuda")


def _make_inputs():
    generator = torch.Generator(device="cuda")
    generator.manual_seed(2)
    source = torch.randn((M, G, K), dtype=torch.float32, device="cuda", generator=generator)
    bias = torch.randn((G,), dtype=torch.float32, device="cuda", generator=generator)
    partial = torch.empty((M, G), dtype=torch.float32, device="cuda")
    biased = torch.empty((M, G), dtype=torch.float32, device="cuda")
    output = torch.empty((M,), dtype=torch.float32, device="cuda")
    for tensor in (partial, biased, output):
        tensor.fill_(float("nan"))
    return source, bias, partial, biased, output


def _run_static(lib, name, tensors):
    event_workspace = _event_workspace()
    queue = torch.empty((SM_COUNT, MAX_TASKS), dtype=torch.int32, device="cuda")
    event_workspace.fill_(-12345)
    queue.fill_(0x7FFFFFFF)

    lib[f"{name}_init_queue"](M, G, K, queue)
    lib[name](*tensors, event_workspace, queue)
    torch.cuda.synchronize()


def _run_dynamic(lib, name, tensors):
    event_workspace = _event_workspace()
    tasks = torch.empty((MAX_TASKS,), dtype=torch.int32, device="cuda")
    head = torch.empty((1,), dtype=torch.int32, device="cuda")
    tail = torch.empty((1,), dtype=torch.int32, device="cuda")
    event_workspace.fill_(-12345)
    tasks.fill_(-1)
    head.fill_(-1)
    tail.fill_(-1)

    lib[f"{name}_init_queue"](M, G, K, event_workspace, tasks, head, tail)
    lib[name](*tensors, event_workspace, tasks, head, tail)
    torch.cuda.synchronize()


def _run_and_check(schedule):
    name = f"{schedule}_simple"
    kernel = _build_simple_kernel(name, dynamic=schedule == "dynamic")
    lib = _compile_kernel(kernel, _options(schedule))
    tensors = _make_inputs()

    if schedule == "static":
        _run_static(lib, name, tensors)
    else:
        _run_dynamic(lib, name, tensors)

    source, bias, _partial, _biased, output = tensors
    reference = _torch_simple_reference(source, bias)
    torch.testing.assert_close(output.cpu(), reference.cpu(), rtol=1e-3, atol=1e-3)


def test_static_simple_matches_torch():
    """Run the simple var-shaped megakernel with static scheduling."""
    _run_and_check("static")


def test_dynamic_simple_matches_torch():
    """Run the simple var-shaped megakernel with dynamic scheduling."""
    _run_and_check("dynamic")
