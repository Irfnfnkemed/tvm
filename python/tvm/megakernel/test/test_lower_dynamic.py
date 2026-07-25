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
"""Dynamic lowering structural smoke tests."""

from __future__ import annotations

import pytest
import tvm.tirx.script as T
from tvm.tirx.script import tile as Tx

from tvm.megakernel.dsl import D, KernelSpec, R, TileImpl
from tvm.megakernel.transform import LoweringOptions, lower
from tvm.megakernel.transform.validate import validate_kernel, validate_lowering_plan
from tvm.megakernel.transform.lower.prepare import prepare_lowering_plan
from tvm.megakernel.transform.lower.lower import _LoweringUtils


class ProducerTile(TileImpl):
    def __init__(self, source, tmp):
        super().__init__()
        self.source = source
        self.tmp = tmp

    @T.inline
    def run(self, m_idx, n_idx, k_idx):
        Tx.copy(self.tmp[0:8], self.source[0:8])

class ConsumerTile(TileImpl):
    def __init__(self, tmp, out):
        super().__init__()
        self.tmp = tmp
        self.out = out

    @T.inline
    def run(self, m_idx, n_idx, k_idx):
        Tx.copy(self.out[0:8], self.tmp[0:8])

class EmptyTile(TileImpl):
    @T.inline
    def run(self, m_idx, n_idx, k_idx):
        T.evaluate(0)

def _options():
    return LoweringOptions(
        schedule="dynamic",
        attrs={"sm_count": 2, "num_threads": 128, "max_tasks": 64, "end_job_id": 31},
    )

def _simple_dynamic_kernel():
    kernel = KernelSpec("dynamic_simple")
    source = kernel.tensor("source", (8,), "float32")
    tmp = kernel.tensor("tmp", (8,), "float32")
    out = kernel.tensor("out", (8,), "float32")
    ready = kernel.event("ready", (1,), init_count=1)

    kernel.tile(
        "producer",
        ProducerTile(source, tmp),
        (1, 1, 1),
        reads=[source.region(lambda m, n, k: R[0:8])],
        writes=[tmp.region(lambda m, n, k: R[0:8])],
    ).notify(D(ready, lambda m, n, k, i: (1, -1, m,)))
    kernel.tile(
        "consumer",
        ConsumerTile(tmp, out),
        (1, 1, 1),
        reads=[tmp.region(lambda m, n, k: R[0:8])],
        writes=[out.region(lambda m, n, k: R[0:8])],
    ).wait(D(ready, lambda m, n, k, i: (1, -1, m,), inv_coord=lambda rank, e, i: (1, e, 0, 0)))
    return kernel

def _dynamic_plan(kernel):
    validate_kernel(kernel)
    return prepare_lowering_plan(kernel, _options())

def test_dynamic_lower_accepts_batch_notify_pre_push():
    """Check dynamic lowering accepts a notify that fans out across multiple notify workers."""
    kernel = KernelSpec("dynamic_batch_notify")
    ready = kernel.event("ready", (2,), init_count=1)
    done = kernel.event("done", (1,), init_count=2)

    kernel.tile(
        "producer",
        EmptyTile(),
        (1, 1, 1),
    ).notify(D(ready, lambda m, n, k, i: (2, -1, i)))
    kernel.tile(
        "consumer",
        EmptyTile(),
        (2, 1, 1),
    ).wait(
        D(ready, lambda m, n, k, i: (1, -1, m), inv_coord=lambda rank, e, i: (1, e, 0, 0))
    ).notify(D(done, lambda m, n, k, i: (1, -1, 0)))
    kernel.tile("endpoint", EmptyTile(), (1, 1, 1)).wait(
        D(done, lambda m, n, k, i: (1, -1, m), inv_coord=lambda rank, e, i: (1, e, 0, 0))
    )

    lowered = str(lower(kernel, _options()))

    assert "dynamic_batch_notify" in lowered

def test_dynamic_lower_emits_tasks_head_tail_and_enqueue():
    """Check dynamic lowering emits queue args and enqueue operations."""
    mod = lower(_simple_dynamic_kernel(), _options())
    lowered = str(mod)

    assert "tasks:" in lowered
    assert "head:" in lowered
    assert "tail:" in lowered
    assert "atomic_add(T.address_of(tail)" in lowered
    assert "dynamic_simple_init_queue" in lowered

def test_dynamic_lower_uses_two_phase_semaphore_notify():
    """Check dynamic lowering emits pre-notify and complete-notify, not static notify."""
    mod = lower(_simple_dynamic_kernel(), _options())
    lowered = str(mod)

    assert "atomic_add" in lowered
    assert "-1, rank" in lowered
    assert "-65536, rank" in lowered
    assert "-65537" not in lowered

def test_dynamic_accepts_old_style_event_scope_attrs():
    """Check old scheduler scope attrs still lower in the dynamic path."""
    kernel = _simple_dynamic_kernel()
    kernel.tiles[0].attrs.update(
        {
            "notify_scope": "warpgroup",
            "notify_scope_id": 0,
            "push_scope": "warpgroup",
            "push_scope_id": 0,
            "push_level": "warp",
        }
    )
    kernel.tiles[1].attrs.update({"wait_scope": "warp", "wait_mask": 0x1})

    plan = _dynamic_plan(kernel)
    validate_lowering_plan(plan)
    lowered = str(lower(kernel, _options()))

    assert "dynamic_simple_init_queue" in lowered
    assert "warp_sync" in lowered

def test_dynamic_default_dequeue_uses_scheduler_warp_mbarriers():
    """Check dynamic dequeue uses the old scheduler-warp mbarrier path."""
    lowered = str(lower(_simple_dynamic_kernel(), _options()))

    assert "T.ptx.elect_sync()" in lowered
    assert "T.ptx.mbarrier.try_wait" in lowered
    assert "T.ptx.mbarrier.arrive" in lowered
    assert "warp_id == 3" in lowered

def test_dynamic_lower_uses_inv_coord_to_push_consumer_tile():
    """Check dynamic triggers use inv_coord to derive concrete consumer tile coords."""
    kernel = KernelSpec("dynamic_inverse")
    tmp = kernel.tensor("tmp", (2, 8), "float32")
    ready = kernel.event("ready", (2,), init_count=1)
    done = kernel.event("done", (1,), init_count=2)

    kernel.tile(
        "producer",
        EmptyTile(),
        (2, 1, 1),
        writes=[tmp.region(lambda m, n, k: R[m, 0:8])],
    ).notify(D(ready, lambda m, n, k, i: (1, -1, m,)))
    kernel.tile(
        "consumer",
        EmptyTile(),
        (1, 2, 1),
        reads=[tmp.region(lambda m, n, k: R[n, 0:8])],
        writes=[tmp.region(lambda m, n, k: R[n, 0:8])],
    ).wait(
        D(ready, lambda m, n, k, i: (1, -1, n,), inv_coord=lambda rank, e, i: (1, 0, e, 0))
    ).notify(D(done, lambda m, n, k, i: (1, -1, 0,)))
    kernel.tile("endpoint", EmptyTile(), (1, 1, 1)).wait(
        D(done, lambda m, n, k, i: (1, -1, m,), inv_coord=lambda rank, e, i: (1, e, 0, 0))
    )

    plan = _dynamic_plan(kernel)
    validate_lowering_plan(plan)
    trigger = plan.dynamic_schedule.triggers["producer"][0]

    assert trigger.consumer.tile.name == "consumer"
    assert trigger.consumer_inv_coord(-1, 1, 0) == (1, 0, 1, 0)
    assert _LoweringUtils.consumer_task_coord_from_event(trigger, -1, (1,), 0) == (0, 1, 0)
    assert "dynamic_inverse_init_queue" in str(lower(kernel, _options()))


def test_dynamic_lower_one_event_coord_pushes_multiple_consumers():
    """Check one ready event coordinate can push multiple consumer tasks."""
    kernel = KernelSpec("dynamic_fanout")
    tmp = kernel.tensor("tmp", (1, 2), "float32")
    ready = kernel.event("ready", (1,), init_count=1)
    done = kernel.event("done", (1,), init_count=2)

    kernel.tile("producer", EmptyTile(), (1, 1, 1), writes=[tmp]).notify(
        D(ready, lambda m, n, k, i: (1, -1, 0))
    )
    kernel.tile("consumer", EmptyTile(), (1, 2, 1), reads=[tmp], writes=[tmp]).wait(
        D(ready, lambda m, n, k, i: (1, -1, m), inv_coord=lambda rank, e, i: (2, e, i, 0))
    ).notify(D(done, lambda m, n, k, i: (1, -1, 0)))
    kernel.tile("endpoint", EmptyTile(), (1, 1, 1)).wait(
        D(done, lambda m, n, k, i: (1, -1, m), inv_coord=lambda rank, e, i: (1, e, 0, 0))
    )

    plan = _dynamic_plan(kernel)
    validate_lowering_plan(plan)
    trigger = plan.dynamic_schedule.triggers["producer"][0]

    assert trigger.consumer_inv_coord(-1, 0, 0) == (2, 0, 0, 0)
    assert trigger.consumer_inv_coord(-1, 0, 1) == (2, 0, 1, 0)
    assert "dynamic_fanout_init_queue" in str(lower(kernel, _options()))


def test_dynamic_lower_runtime_routing_tensor_coord_lowers_with_warning():
    """Check runtime tensor routing coords warn but still lower."""
    kernel = KernelSpec("dynamic_runtime_routing")
    routing = kernel.tensor("routing", (1,), "int32")
    tmp = kernel.tensor("tmp", (1,), "float32")
    ready = kernel.event("ready", (1,), init_count=1)
    done = kernel.event("done", (1,), init_count=1)

    kernel.tile("producer", EmptyTile(), (1, 1, 1), reads=[routing], writes=[tmp]).notify(
        D(ready, lambda m, n, k, i: (1, -1, routing[i]))
    )
    kernel.tile("consumer", EmptyTile(), (1, 1, 1), reads=[tmp]).wait(
        D(ready, lambda m, n, k, i: (1, -1, m), inv_coord=lambda rank, e, i: (1, e, 0, 0))
    ).notify(D(done, lambda m, n, k, i: (1, -1, 0)))
    kernel.tile("endpoint", EmptyTile(), (1, 1, 1)).wait(
        D(done, lambda m, n, k, i: (1, -1, m), inv_coord=lambda rank, e, i: (1, e, 0, 0))
    )

    with pytest.warns(UserWarning, match="runtime TensorSpec indexing"):
        lowered = str(lower(kernel, _options()))

    assert "dynamic_runtime_routing_init_queue" in lowered
    assert "routing" in lowered

