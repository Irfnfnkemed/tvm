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
"""Dynamic scheduler lowering tests for megakernel DSL."""

from __future__ import annotations

import pytest

import tvm.tirx.script as T
from tvm.tirx.script import tile as Tx

from tvm.megakernel.dsl import KernelSpec, R, TileImpl
from tvm.megakernel.dsl.spec import DependencySpec
from tvm.megakernel.transform import LoweringOptions, lower_to_tirx_module
from tvm.megakernel.transform.lower.prepare import prepare_lowering_plan
from tvm.megakernel.transform.lower.tirx import _consumer_task_coord_from_event
from tvm.megakernel.transform.lower.validate import validate_lowering_plan
from tvm.megakernel.transform.semantic import build_semantic_plan, validate_semantic_plan


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
    ).notify(ready, lambda m, n, k: (m,))
    kernel.tile(
        "consumer",
        ConsumerTile(tmp, out),
        (1, 1, 1),
        reads=[tmp.region(lambda m, n, k: R[0:8])],
        writes=[out.region(lambda m, n, k: R[0:8])],
    ).wait(ready, lambda m, n, k: (m,), inverse_coord_from_event=lambda e: (e, 0, 0))
    return kernel


def _dynamic_plan(kernel):
    semantic = validate_semantic_plan(build_semantic_plan(kernel))
    return prepare_lowering_plan(semantic, _options())


def test_dynamic_prepare_records_entry_and_trigger():
    plan = _dynamic_plan(_simple_dynamic_kernel())
    validate_lowering_plan(plan)

    assert plan.dynamic_schedule is not None
    assert [phase.label for phase in plan.dynamic_schedule.entry_phases] == ["producer"]
    assert [(trigger.event.name, trigger.consumer.tile.name) for trigger in plan.dynamic_schedule.triggers["producer"]] == [
        ("ready", "consumer")
    ]
    assert plan.dynamic_schedule.endpoint.tile.name == "consumer"


def test_dynamic_lower_emits_tasks_head_tail_and_enqueue():
    mod = lower_to_tirx_module(_simple_dynamic_kernel(), _options())
    lowered = str(mod)

    assert "tasks:" in lowered
    assert "head:" in lowered
    assert "tail:" in lowered
    assert "atomic_add(T.address_of(tail)" in lowered
    assert "dynamic_simple_init_queue" in lowered


def test_dynamic_lower_uses_two_phase_semaphore_notify():
    mod = lower_to_tirx_module(_simple_dynamic_kernel(), _options())
    lowered = str(mod)

    assert "atomic_add" in lowered
    assert "-1, rank" in lowered
    assert "-65536, rank" in lowered
    assert "-65537" not in lowered


def test_dynamic_rejects_multiple_waits_on_one_tile():
    kernel = KernelSpec("dynamic_bad_multi_wait")
    x = kernel.tensor("x", (1,), "float32")
    e0 = kernel.event("e0", (1,), init_count=1)
    e1 = kernel.event("e1", (1,), init_count=1)

    kernel.tile("p0", EmptyTile(), (1, 1, 1), writes=[x]).notify(e0, lambda m, n, k: (m,))
    kernel.tile("p1", EmptyTile(), (1, 1, 1), writes=[x]).notify(e1, lambda m, n, k: (m,))
    kernel.tile("c", EmptyTile(), (1, 1, 1), reads=[x]).wait(e0, lambda m, n, k: (m,)).wait(
        e1, lambda m, n, k: (m,)
    )

    plan = _dynamic_plan(kernel)
    with pytest.raises(ValueError, match="at most one wait"):
        validate_lowering_plan(plan)


def test_dynamic_rejects_non_identity_wait_without_inverse_coord_map():
    kernel = _simple_dynamic_kernel()
    kernel.tiles[1].waits[0] = DependencySpec(kernel.events["ready"], lambda m, n, k: (n,))

    plan = _dynamic_plan(kernel)
    with pytest.raises(ValueError, match="inverse_coord_from_event"):
        validate_lowering_plan(plan)


def test_dynamic_uses_inverse_coord_map_to_push_consumer_tile():
    kernel = KernelSpec("dynamic_inverse")
    tmp = kernel.tensor("tmp", (2, 8), "float32")
    ready = kernel.event("ready", (2,), init_count=1)
    done = kernel.event("done", (1,), init_count=2)

    kernel.tile(
        "producer",
        EmptyTile(),
        (2, 1, 1),
        writes=[tmp.region(lambda m, n, k: R[m, 0:8])],
    ).notify(ready, lambda m, n, k: (m,))
    kernel.tile(
        "consumer",
        EmptyTile(),
        (1, 2, 1),
        reads=[tmp.region(lambda m, n, k: R[n, 0:8])],
        writes=[tmp.region(lambda m, n, k: R[n, 0:8])],
    ).wait(
        ready,
        lambda m, n, k: (n,),
        inverse_coord_from_event=lambda e: (0, e, 0),
    ).notify(done, lambda m, n, k: (0,))
    kernel.tile("endpoint", EmptyTile(), (1, 1, 1)).wait(
        done, lambda m, n, k: (m,), inverse_coord_from_event=lambda e: (e, 0, 0)
    )

    plan = _dynamic_plan(kernel)
    validate_lowering_plan(plan)
    trigger = plan.dynamic_schedule.triggers["producer"][0]

    assert trigger.consumer.tile.name == "consumer"
    assert trigger.consumer_inverse_coord_map(1) == (0, 1, 0)
    assert _consumer_task_coord_from_event(trigger, (1,)) == (0, 1, 0)

    lowered = str(lower_to_tirx_module(kernel, _options()))
    assert "dynamic_inverse_init_queue" in lowered



def test_dynamic_accepts_old_style_event_scope_attrs():
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
    lowered = str(lower_to_tirx_module(kernel, _options()))

    assert "dynamic_simple_init_queue" in lowered
    assert "warp_sync" in lowered


def test_dynamic_default_dequeue_uses_scheduler_warp_mbarriers():
    lowered = str(lower_to_tirx_module(_simple_dynamic_kernel(), _options()))

    assert "T.ptx.elect_sync()" in lowered
    assert "T.ptx.mbarrier.try_wait" in lowered
    assert "T.ptx.mbarrier.arrive" in lowered
    assert "warp_id == 3" in lowered


def test_dynamic_can_use_single_thread_dequeue_for_debugging():
    options = _options()
    options.attrs["dynamic_dequeue_mode"] = "single_thread"
    lowered = str(lower_to_tirx_module(_simple_dynamic_kernel(), options))

    assert "T.ptx.elect_sync()" not in lowered
    assert "atomic_add(T.address_of(head)" in lowered
