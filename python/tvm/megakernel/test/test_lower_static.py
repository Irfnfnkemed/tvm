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
"""Static lowering structural smoke tests."""

from __future__ import annotations

import tvm.tirx.script as T

from tvm.megakernel.dsl import D, KernelSpec, R, TileImpl
from tvm.megakernel.transform import LoweringOptions, lower
from tvm.megakernel.transform.lower.lower import _LoweringPipeline


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


class EmptyTile(TileImpl):
    @T.inline
    def run(self, m_idx, n_idx, k_idx):
        T.evaluate(0)


def _simple_static_kernel(name="static_simple"):
    kernel = KernelSpec(name)
    x = kernel.tensor("x", (1,), "float32")
    ready = kernel.event("ready", (1,), init_count=1)
    kernel.tile("producer", EmptyTile(), (1, 1, 1), writes=[x]).notify(
        D(ready, lambda m, n, k, i: (1, -1, 0))
    )
    kernel.tile("consumer", EmptyTile(), (1, 1, 1), reads=[x]).wait(
        D(ready, lambda m, n, k, i: (1, -1, 0))
    )
    return kernel


def test_static_lower_emits_queue_init_function():
    """Check static lowering emits a queue-init function and task packing code."""
    lowered = str(lower(_simple_static_kernel(), _options()))

    assert "static_simple_init_queue" in lowered
    assert "T.shift_left" in lowered


def test_static_lower_emits_single_phase_event_wait_notify():
    """Check static lowering uses the single-phase semaphore notify path."""
    lowered = str(lower(_simple_static_kernel(), _options()))

    assert "T.ptx.ld_global_acquire" in lowered
    assert "-65537" in lowered
    assert "-65536, rank" not in lowered



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
    """Check lowering emits prefetch before wait and run."""
    kernel = KernelSpec("prefetch_order")
    x = kernel.tensor("x", (1,), "float32")
    ready = kernel.event("ready", (1,), init_count=1)
    kernel.tile(
        "producer", OrderProducerTile(), (1, 1, 1), reads=[x.region(lambda m, n, k: R[0])]
    ).notify(D(ready, lambda m, n, k, i: (1, -1, 0,)))
    kernel.tile(
        "consumer", OrderConsumerTile(), (1, 1, 1), reads=[x.region(lambda m, n, k: R[0])]
    ).wait(D(ready, lambda m, n, k, i: (1, -1, 0,)))

    pipeline = _LoweringPipeline(_options())
    lowered = str(pipeline.emit_kernel(pipeline.prepare_plan(kernel)))

    prefetch_pos = lowered.find(PREFETCH_MARKER)
    wait_pos = lowered.find(WAIT_MARKER, prefetch_pos)
    run_pos = lowered.find(RUN_MARKER)
    assert -1 not in (prefetch_pos, wait_pos, run_pos)
    assert prefetch_pos < wait_pos < run_pos
