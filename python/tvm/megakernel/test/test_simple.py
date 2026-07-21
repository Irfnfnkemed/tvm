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
"""Simple end-to-end megakernel test."""

import pytest
import statistics
import warnings
import torch
import tvm
import tvm.tirx.script as T
from tvm.tirx.script import tile as Tx

from tvm.megakernel.dsl import KernelSpec, R, TileImpl
from tvm.megakernel.dsl.spec import DependencySpec
from tvm.megakernel.transform import LoweringOptions, MegakernelLowerer, lower_to_tirx_module


ROWS = 4
GROUPS = 3
K = 8
SM_COUNT = 16
MAX_TASKS = 16
NUM_THREADS = 256


def _options():
    return LoweringOptions(
        smem_max_bytes=64 * 1024,
        smem_chunk_size=16 * 1024,
        attrs={
            "sm_count": SM_COUNT,
            "num_threads": NUM_THREADS,
            "max_tasks": MAX_TASKS,
            "end_job_id": 31,
        },
    )


class SumTile(TileImpl):
    def __init__(self, source, partial):
        super().__init__()
        self.source = source
        self.partial = partial
        self.smem = None
        self.reduced = None
        self.smem_manager = None

    def _declare_resources(self, smem_manager):
        self.smem_manager = smem_manager
        self.smem = smem_manager.alloc((K,), "float32", policy="shared")
        self.reduced = smem_manager.alloc((1,), "float32", policy="shared")

    @T.inline
    def device_init(self, smem_manager, m_idx, n_idx, k_idx):
        self._declare_resources(smem_manager)

    @T.inline
    def run(self, m_idx, n_idx, k_idx):
        self.smem_manager.wait_all("cta")
        Tx.copy(self.smem, self.source[m_idx, n_idx, :])
        Tx.sum(self.reduced, self.smem)
        tid = T.thread_id([NUM_THREADS])
        if tid == 0:
            self.partial[m_idx, n_idx] = self.reduced[0]
        self.smem_manager.release_all("cta")
        self.smem_manager.advance()


class MergeTile(TileImpl):
    def __init__(self, partial_a, partial_b, merged):
        super().__init__()
        self.partial_a = partial_a
        self.partial_b = partial_b
        self.merged = merged
        self.smem = None
        self.smem_manager = None

    def _declare_resources(self, smem_manager):
        self.smem_manager = smem_manager
        self.smem = smem_manager.alloc((2,), "float32", policy="shared")

    @T.inline
    def device_init(self, smem_manager, m_idx, n_idx, k_idx):
        self._declare_resources(smem_manager)

    @T.inline
    def run(self, m_idx, n_idx, k_idx):
        self.smem_manager.wait_all("cta")
        tid = T.thread_id([NUM_THREADS])
        if tid == 0:
            self.smem[0] = self.partial_a[m_idx, n_idx]
            self.smem[1] = self.partial_b[m_idx, n_idx]
            self.merged[m_idx, n_idx] = self.smem[0] + 2.0 * self.smem[1]
        self.smem_manager.release_all("cta")
        self.smem_manager.advance()


class FinalTile(TileImpl):
    def __init__(self, merged, output):
        super().__init__()
        self.merged = merged
        self.output = output
        self.smem = None
        self.reduced = None
        self.smem_manager = None

    def _declare_resources(self, smem_manager):
        self.smem_manager = smem_manager
        self.smem = smem_manager.alloc((GROUPS,), "float32", policy="shared")
        self.reduced = smem_manager.alloc((1,), "float32", policy="shared")

    @T.inline
    def device_init(self, smem_manager, m_idx, n_idx, k_idx):
        self._declare_resources(smem_manager)

    @T.inline
    def run(self, m_idx, n_idx, k_idx):
        self.smem_manager.wait_all("cta")
        Tx.copy(self.smem, self.merged[m_idx, :])
        Tx.sum(self.reduced, self.smem)
        tid = T.thread_id([NUM_THREADS])
        if tid == 0:
            self.output[m_idx] = self.reduced[0]
        self.smem_manager.release_all("cta")
        self.smem_manager.advance()


def _build_kernel():
    kernel = KernelSpec("complex_megakernel")
    tensor_a = kernel.tensor("A", (ROWS, GROUPS, K), "float32")
    tensor_b = kernel.tensor("B", (ROWS, GROUPS, K), "float32")
    partial_a = kernel.tensor("partial_a", (ROWS, GROUPS), "float32")
    partial_b = kernel.tensor("partial_b", (ROWS, GROUPS), "float32")
    merged = kernel.tensor("merged", (ROWS, GROUPS), "float32")
    output = kernel.tensor("output", (ROWS,), "float32")

    pair_ready = kernel.event("pair_ready", (ROWS, GROUPS), init_count=2)
    group_ready = kernel.event("group_ready", (ROWS,), init_count=GROUPS)

    kernel.tile(
        "sum_a",
        SumTile(tensor_a, partial_a),
        (ROWS, GROUPS, 1),
        reads=[tensor_a.region(lambda m, n, k: R[m, n, 0:K])],
        writes=[partial_a.region(lambda m, n, k: R[m, n])],
    ).notify(pair_ready, lambda m, n, k: (m, n))
    kernel.tile(
        "sum_b",
        SumTile(tensor_b, partial_b),
        (ROWS, GROUPS, 1),
        reads=[tensor_b.region(lambda m, n, k: R[m, n, 0:K])],
        writes=[partial_b.region(lambda m, n, k: R[m, n])],
    ).notify(pair_ready, lambda m, n, k: (m, n))
    (
        kernel.tile(
            "merge",
            MergeTile(partial_a, partial_b, merged),
            (ROWS, GROUPS, 1),
            reads=[
                partial_a.region(lambda m, n, k: R[m, n]),
                partial_b.region(lambda m, n, k: R[m, n]),
            ],
            writes=[merged.region(lambda m, n, k: R[m, n])],
        )
        .wait(pair_ready, lambda m, n, k: (m, n))
        .notify(group_ready, lambda m, n, k: (m,))
    )
    kernel.tile(
        "final",
        FinalTile(merged, output),
        (ROWS, 1, 1),
        reads=[
            merged.region(lambda m, n, k: R[m, 0]),
            merged.region(lambda m, n, k: R[m, 1]),
            merged.region(lambda m, n, k: R[m, 2]),
        ],
        writes=[output.region(lambda m, n, k: R[m])],
    ).wait(group_ready, lambda m, n, k: (m,))
    return kernel


PREFETCH_MARKER = "tirx.megakernel.prefetch.marker"
RUN_MARKER = "tirx.megakernel.run.marker"
WAIT_MARKER = "T.ptx.ld_global_acquire"


class OrderProducerTile(TileImpl):
    @T.inline
    def run(self, m_idx, n_idx, k_idx):
        T.evaluate(0)


class OrderConsumerTile(TileImpl):
    @T.inline
    def prefetch(self, m_idx, n_idx, k_idx):
        T.evaluate(T.call_extern("void", PREFETCH_MARKER, m_idx))

    @T.inline
    def run(self, m_idx, n_idx, k_idx):
        T.evaluate(T.call_extern("void", RUN_MARKER, m_idx))


def test_prefetch_is_emitted_before_event_wait():
    kernel = KernelSpec("prefetch_order")
    x = kernel.tensor("x", (1,), "float32")
    ready = kernel.event("ready", (1,), init_count=1)
    kernel.tile(
        "producer", OrderProducerTile(), (1, 1, 1), reads=[x.region(lambda m, n, k: R[0])]
    ).notify(
        ready, lambda m, n, k: (0,)
    )
    kernel.tile(
        "consumer", OrderConsumerTile(), (1, 1, 1), reads=[x.region(lambda m, n, k: R[0])]
    ).wait(
        ready, lambda m, n, k: (0,)
    )

    lowered = str(MegakernelLowerer(_options()).lower(kernel))

    prefetch_pos = lowered.find(PREFETCH_MARKER)
    wait_pos = lowered.find(WAIT_MARKER, prefetch_pos)
    run_pos = lowered.find(RUN_MARKER)
    assert -1 not in (prefetch_pos, wait_pos, run_pos)
    assert prefetch_pos < wait_pos < run_pos



PREFETCH_M = 2
PREFETCH_N = 1
PREFETCH_R = 256
PREFETCH_C = 32
PREFETCH_MAX_TASKS = 4096
GATE_SLEEP_ITERS = 64
PREFETCH_WORK_ITERS = 2048


class PrefetchProducerTile(TileImpl):
    def __init__(self, source, tmp):
        super().__init__()
        self.source = source
        self.tmp = tmp

    @T.inline
    def run(self, m_idx, n_idx, k_idx):
        tid = T.thread_id([NUM_THREADS])
        if tid == 0:
            self.tmp[m_idx, n_idx] = self.source[m_idx, n_idx] * 2.0


class PrefetchGateTile(TileImpl):
    @T.inline
    def run(self, m_idx, n_idx, k_idx):
        for _ in T.serial(GATE_SLEEP_ITERS):
            T.cuda.nano_sleep(80)


class PrefetchConsumerTile(TileImpl):
    def __init__(self, independent, tmp, output):
        super().__init__()
        self.independent = independent
        self.tmp = tmp
        self.output = output
        self.smem_manager = None
        self.prefetch_smem = None
        self.prefetched = None

    def _declare_resources(self, smem_manager):
        self.smem_manager = smem_manager
        self.prefetch_smem = smem_manager.alloc((PREFETCH_R, PREFETCH_C), "float32", policy="shared")
        self.prefetched = smem_manager.alloc((1,), "float32", policy="shared")

    @T.inline
    def device_init(self, smem_manager, m_idx, n_idx, k_idx):
        self._declare_resources(smem_manager)

    @T.inline
    def prefetch(self, m_idx, n_idx, k_idx):
        self.smem_manager.wait_all("cta")
        Tx.cta.copy(self.prefetch_smem, self.independent[m_idx, n_idx, :, :])
        T.cuda.cta_sync()
        Tx.cta.sum(self.prefetched, self.prefetch_smem, axes=(0, 1))
        for _ in T.serial(PREFETCH_WORK_ITERS):
            T.cuda.nano_sleep(80)

    @T.inline
    def run(self, m_idx, n_idx, k_idx):
        tid = T.thread_id([NUM_THREADS])
        if tid == 0:
            self.output[m_idx, n_idx] = self.tmp[m_idx, n_idx] + self.prefetched[0] * 3.0
        self.smem_manager.release_all("cta")
        self.smem_manager.advance()


class NoPrefetchConsumerTile(TileImpl):
    def __init__(self, independent, tmp, output):
        super().__init__()
        self.independent = independent
        self.tmp = tmp
        self.output = output
        self.smem_manager = None
        self.load_smem = None
        self.loaded = None

    def _declare_resources(self, smem_manager):
        self.smem_manager = smem_manager
        self.load_smem = smem_manager.alloc((PREFETCH_R, PREFETCH_C), "float32", policy="shared")
        self.loaded = smem_manager.alloc((1,), "float32", policy="shared")

    @T.inline
    def device_init(self, smem_manager, m_idx, n_idx, k_idx):
        self._declare_resources(smem_manager)

    @T.inline
    def run(self, m_idx, n_idx, k_idx):
        self.smem_manager.wait_all("cta")
        Tx.cta.copy(self.load_smem, self.independent[m_idx, n_idx, :, :])
        T.cuda.cta_sync()
        Tx.cta.sum(self.loaded, self.load_smem, axes=(0, 1))
        for _ in T.serial(PREFETCH_WORK_ITERS):
            T.cuda.nano_sleep(80)
        tid = T.thread_id([NUM_THREADS])
        if tid == 0:
            self.output[m_idx, n_idx] = self.tmp[m_idx, n_idx] + self.loaded[0] * 3.0
        self.smem_manager.release_all("cta")
        self.smem_manager.advance()


def _prefetch_options():
    return LoweringOptions(
        smem_max_bytes=96 * 1024,
        smem_chunk_size=32 * 1024,
        attrs={
            "sm_count": SM_COUNT,
            "num_threads": NUM_THREADS,
            "max_tasks": PREFETCH_MAX_TASKS,
            "end_job_id": 31,
        },
    )


def _build_prefetch_kernel(name, consumer_cls):
    kernel = KernelSpec(name)
    rows = kernel.var("rows", range=(PREFETCH_M, PREFETCH_M))
    cols = kernel.var("cols", range=(PREFETCH_N, PREFETCH_N))
    source = kernel.tensor("A", (rows, cols), "float32")
    independent = kernel.tensor("B", (PREFETCH_M, PREFETCH_N, PREFETCH_R, PREFETCH_C), "float32")
    tmp = kernel.tensor("tmp", (rows, cols), "float32")
    output = kernel.tensor("output", (rows, cols), "float32")
    ready = kernel.event("ready", (PREFETCH_M, PREFETCH_N), init_count=2)

    kernel.tile(
        "producer",
        PrefetchProducerTile(source, tmp),
        (PREFETCH_M, PREFETCH_N, 1),
        reads=[source.region(lambda m, n, k: R[m, n])],
        writes=[tmp.region(lambda m, n, k: R[m, n])],
    ).notify(ready, lambda m, n, k: (m, n))
    kernel.tile(
        "gate",
        PrefetchGateTile(),
        (PREFETCH_M, PREFETCH_N, 1),
    ).notify(ready, lambda m, n, k: (m, n))
    (
        kernel.tile(
            "consumer",
            consumer_cls(independent, tmp, output),
            (PREFETCH_M, PREFETCH_N, 1),
            reads=[
                independent.region(lambda m, n, k: R[m, n, 0:PREFETCH_R, 0:PREFETCH_C]),
                tmp.region(lambda m, n, k: R[m, n]),
            ],
            writes=[output.region(lambda m, n, k: R[m, n])],
        )
        .wait(ready, lambda m, n, k: (m, n))
    )
    return kernel


def _compile_prefetch_kernel(kernel):
    mod = lower_to_tirx_module(kernel, _prefetch_options())
    with tvm.target.Target("cuda"):
        return tvm.compile(mod, target="cuda", tir_pipeline="tirx")


def _run_prefetch_kernel(lib, kernel_name, source, independent, tmp, output, event_workspace, queue):
    lib[f"{kernel_name}_init_queue"](queue)
    lib[kernel_name](source, independent, tmp, output, event_workspace, queue)


def _time_prefetch_kernel(lib, kernel_name, source, independent, tmp, output, event_workspace, queue):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    iterations = 5
    start.record()
    for _ in range(iterations):
        _run_prefetch_kernel(lib, kernel_name, source, independent, tmp, output, event_workspace, queue)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iterations


def _sample_prefetch_timings(prefetch_args, no_prefetch_args):
    for _ in range(3):
        _run_prefetch_kernel(*prefetch_args)
        _run_prefetch_kernel(*no_prefetch_args)
    torch.cuda.synchronize()

    prefetch_samples = []
    no_prefetch_samples = []
    for sample_idx in range(20):
        if sample_idx % 2 == 0:
            prefetch_samples.append(_time_prefetch_kernel(*prefetch_args))
            no_prefetch_samples.append(_time_prefetch_kernel(*no_prefetch_args))
        else:
            no_prefetch_samples.append(_time_prefetch_kernel(*no_prefetch_args))
            prefetch_samples.append(_time_prefetch_kernel(*prefetch_args))
    return prefetch_samples, no_prefetch_samples


def test_prefetch_e2e_matches_no_prefetch_and_reports_timing():
    prefetch_name = "prefetch_e2e"
    no_prefetch_name = "no_prefetch_e2e"
    prefetch_kernel = _build_prefetch_kernel(prefetch_name, PrefetchConsumerTile)
    no_prefetch_kernel = _build_prefetch_kernel(no_prefetch_name, NoPrefetchConsumerTile)
    prefetch_plan = MegakernelLowerer(_prefetch_options()).prepare(prefetch_kernel)
    assert [var.name for var in prefetch_plan.var_order] == ["rows", "cols"]
    assert [tensor.shape for tensor in prefetch_plan.semantic.tensors] == [
        (prefetch_kernel.vars["rows"], prefetch_kernel.vars["cols"]),
        (PREFETCH_M, PREFETCH_N, PREFETCH_R, PREFETCH_C),
        (prefetch_kernel.vars["rows"], prefetch_kernel.vars["cols"]),
        (prefetch_kernel.vars["rows"], prefetch_kernel.vars["cols"]),
    ]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        prefetch_lib = _compile_prefetch_kernel(prefetch_kernel)
        no_prefetch_lib = _compile_prefetch_kernel(no_prefetch_kernel)
    fallback_warnings = [warning for warning in caught if "copy/fallback" in str(warning.message)]
    assert not fallback_warnings

    generator = torch.Generator(device="cuda")
    generator.manual_seed(1)
    shape = (PREFETCH_M, PREFETCH_N)
    source = torch.randn(shape, dtype=torch.float32, device="cuda", generator=generator)
    independent = torch.randn((PREFETCH_M, PREFETCH_N, PREFETCH_R, PREFETCH_C), dtype=torch.float32, device="cuda", generator=generator)

    prefetch_tmp = torch.empty(shape, dtype=torch.float32, device="cuda")
    prefetch_out = torch.empty(shape, dtype=torch.float32, device="cuda")
    no_prefetch_tmp = torch.empty(shape, dtype=torch.float32, device="cuda")
    no_prefetch_out = torch.empty(shape, dtype=torch.float32, device="cuda")
    workspace_shape = (PREFETCH_M * PREFETCH_N + 1,)
    prefetch_event_workspace = torch.empty(workspace_shape, dtype=torch.int32, device="cuda")
    no_prefetch_event_workspace = torch.empty(workspace_shape, dtype=torch.int32, device="cuda")
    prefetch_queue = torch.empty((SM_COUNT, PREFETCH_MAX_TASKS), dtype=torch.int32, device="cuda")
    no_prefetch_queue = torch.empty((SM_COUNT, PREFETCH_MAX_TASKS), dtype=torch.int32, device="cuda")

    _run_prefetch_kernel(
        prefetch_lib,
        prefetch_name,
        source,
        independent,
        prefetch_tmp,
        prefetch_out,
        prefetch_event_workspace,
        prefetch_queue,
    )
    _run_prefetch_kernel(
        no_prefetch_lib,
        no_prefetch_name,
        source,
        independent,
        no_prefetch_tmp,
        no_prefetch_out,
        no_prefetch_event_workspace,
        no_prefetch_queue,
    )
    torch.cuda.synchronize()

    reference = source * 2.0 + independent.sum(dim=(2, 3)) * 3.0
    torch.testing.assert_close(prefetch_out.cpu(), reference.cpu(), rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(no_prefetch_out.cpu(), reference.cpu(), rtol=1e-3, atol=1e-3)

    prefetch_samples, no_prefetch_samples = _sample_prefetch_timings(
        (
            prefetch_lib,
            prefetch_name,
            source,
            independent,
            prefetch_tmp,
            prefetch_out,
            prefetch_event_workspace,
            prefetch_queue,
        ),
        (
            no_prefetch_lib,
            no_prefetch_name,
            source,
            independent,
            no_prefetch_tmp,
            no_prefetch_out,
            no_prefetch_event_workspace,
            no_prefetch_queue,
        ),
    )
    prefetch_ms = statistics.median(prefetch_samples)
    no_prefetch_ms = statistics.median(no_prefetch_samples)
    print(
        "prefetch_e2e timing: "
        f"prefetch_median={prefetch_ms:.4f} ms, "
        f"no_prefetch_median={no_prefetch_ms:.4f} ms, "
        f"ratio={prefetch_ms / no_prefetch_ms:.3f}"
    )
    assert prefetch_ms > 0
    assert no_prefetch_ms > 0


def test_simple_megakernel():
    kernel = _build_kernel()
    physical = MegakernelLowerer(_options()).prepare(kernel)
    assert [(edge.producer.name, edge.consumer.name, edge.event.name) for edge in physical.semantic.logical_edges] == [
        ("sum_a", "merge", "pair_ready"),
        ("sum_b", "merge", "pair_ready"),
        ("merge", "final", "group_ready"),
    ]
    assert [(event.name, event.workspace_offset, event.size) for event in physical.event_layouts] == [
        ("pair_ready", 0, ROWS * GROUPS),
        ("group_ready", ROWS * GROUPS, ROWS),
    ]
    assert physical.tile_job_ids == {"sum_a": 0, "sum_b": 1, "merge": 2, "final": 3}

    bad_kernel = _build_kernel()
    bad_kernel.tiles[1].notifies[0] = DependencySpec(
        bad_kernel.events["pair_ready"],
        lambda m, n, k: (m, 0),
    )
    with pytest.raises(ValueError, match="init_count"):
        MegakernelLowerer(_options()).prepare(bad_kernel)

    plan = physical.normalized_data()
    assert [event["name"] for event in plan["events"]] == ["pair_ready", "group_ready"]
    assert [phase["label"] for phase in plan["static_schedule"]["phases"]] == [
        "init_event",
        "sum_a",
        "sum_b",
        "wait_event_init",
        "merge",
        "final",
        "end",
    ]

    mod = lower_to_tirx_module(kernel, _options())
    with tvm.target.Target("cuda"):
        lib = tvm.compile(mod, target="cuda", tir_pipeline="tirx")

    generator = torch.Generator(device="cuda")
    generator.manual_seed(0)
    tensor_a = torch.randn((ROWS, GROUPS, K), dtype=torch.float32, device="cuda", generator=generator)
    tensor_b = torch.randn((ROWS, GROUPS, K), dtype=torch.float32, device="cuda", generator=generator)
    partial_a = torch.empty((ROWS, GROUPS), dtype=torch.float32, device="cuda")
    partial_b = torch.empty((ROWS, GROUPS), dtype=torch.float32, device="cuda")
    merged = torch.empty((ROWS, GROUPS), dtype=torch.float32, device="cuda")
    output = torch.empty((ROWS,), dtype=torch.float32, device="cuda")
    event_workspace = torch.empty((ROWS * GROUPS + ROWS + 1,), dtype=torch.int32, device="cuda")
    queue = torch.empty((SM_COUNT, MAX_TASKS), dtype=torch.int32, device="cuda")

    for tensor in (partial_a, partial_b, merged, output):
        tensor.fill_(float("nan"))
    event_workspace.fill_(-12345)
    queue.fill_(0x7FFFFFFF)

    lib["complex_megakernel_init_queue"](queue)
    lib["complex_megakernel"](
        tensor_a, tensor_b, partial_a, partial_b, merged, output, event_workspace, queue
    )
    torch.cuda.synchronize()

    reference = (tensor_a.sum(dim=2) + 2.0 * tensor_b.sum(dim=2)).sum(dim=1)
    torch.testing.assert_close(output.cpu(), reference.cpu(), rtol=1e-3, atol=1e-3)
