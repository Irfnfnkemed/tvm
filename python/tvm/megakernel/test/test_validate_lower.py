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
"""Prepared lowering-plan validation tests."""

from __future__ import annotations

import pytest

import tvm.tirx.script as T
from tvm.tirx.script import tile as Tx

from tvm.megakernel.dsl import D, KernelSpec, R, TileImpl
from tvm.megakernel.dsl.spec import DependencySpec
from tvm.megakernel.transform.lower import LoweringOptions
from tvm.megakernel.transform.lower.prepare import INIT_EVENT_JOB_ID, WAIT_EVENT_INIT_JOB_ID, prepare_lowering_plan, prepare_static_lowering_plan
from tvm.megakernel.transform.validate import validate_kernel, validate_lowering_plan, validate_static_lowering_plan


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

def test_dynamic_validate_warns_when_runtime_wait_coord_skips_static_checks():
    """Warn when runtime TensorSpec indexing prevents static dynamic coord checks."""
    kernel = KernelSpec("dynamic_runtime_wait_warning")
    buf = kernel.tensor("buf", (1,), "int32")
    ready = kernel.event("ready", (1,), init_count=1)

    kernel.tile("producer", EmptyTile(), (1, 1, 1), reads=[buf]).notify(
        D(ready, lambda m, n, k, i: (1, -1, 0))
    )
    kernel.tile("consumer", EmptyTile(), (1, 1, 1), reads=[buf]).wait(
        D(
            ready,
            lambda m, n, k, i: (1, -1, buf[i]),
            inv_coord=lambda rank, e, i: (1, 0, 0, 0),
        )
    )

    plan = _dynamic_plan(kernel)
    with pytest.warns(UserWarning) as warnings:
        validate_lowering_plan(plan)

    messages = [str(warning.message) for warning in warnings]
    assert any("skipping static coord dim validation" in message for message in messages)
    assert any("skipping static inv_coord round-trip validation" in message for message in messages)

def test_dynamic_rejects_multiple_waits_on_one_tile():
    """Reject dynamic tile plans with more than one wait dependency."""
    kernel = KernelSpec("dynamic_bad_multi_wait")
    x = kernel.tensor("x", (1,), "float32")
    e0 = kernel.event("e0", (1,), init_count=1)
    e1 = kernel.event("e1", (1,), init_count=1)

    kernel.tile("p0", EmptyTile(), (1, 1, 1), writes=[x]).notify(D(e0, lambda m, n, k, i: (1, -1, m,)))
    kernel.tile("p1", EmptyTile(), (1, 1, 1), writes=[x]).notify(D(e1, lambda m, n, k, i: (1, -1, m,)))
    kernel.tile("c", EmptyTile(), (1, 1, 1), reads=[x]).wait(D(e0, lambda m, n, k, i: (1, -1, m,))).wait(D(e1, lambda m, n, k, i: (1, -1, m,)))

    plan = _dynamic_plan(kernel)
    with pytest.raises(ValueError, match="at most one wait"):
        validate_lowering_plan(plan)

def test_dynamic_rejects_non_identity_wait_without_inv_coord():
    """Reject dynamic waits that cannot push consumers because inv_coord is missing."""
    kernel = _simple_dynamic_kernel()
    kernel.tiles[1].waits[0] = DependencySpec(kernel.events["ready"], lambda m, n, k, i: (1, -1, n,))

    plan = _dynamic_plan(kernel)
    with pytest.raises(ValueError, match="inv_coord"):
        validate_lowering_plan(plan)



def _batch_notify_dynamic_kernel(coord, init_count=None):
    kernel = KernelSpec("dynamic_bad_batch_notify")
    tmp = kernel.tensor("tmp", (1,), "float32")
    if init_count is None:
        init_count = lambda e: 2 if e == 0 else 0
    ready = kernel.event("ready", (2,), init_count=init_count)
    done = kernel.event("done", (1,), init_count=1)

    kernel.tile("producer", EmptyTile(), (1, 1, 1), writes=[tmp]).notify(D(ready, coord))
    kernel.tile("consumer", EmptyTile(), (1, 1, 1), reads=[tmp]).wait(
        D(ready, lambda m, n, k, i: (1, -1, 0), inv_coord=lambda rank, e, i: (1, 0, 0, 0))
    ).notify(D(done, lambda m, n, k, i: (1, -1, 0)))
    kernel.tile("endpoint", EmptyTile(), (1, 1, 1)).wait(
        D(done, lambda m, n, k, i: (1, -1, 0), inv_coord=lambda rank, e, i: (1, 0, 0, 0))
    )
    return kernel


def test_dynamic_rejects_duplicate_multi_coord_notify_event_coords():
    """Reject notify mappings that generate the same event coord twice."""
    kernel = _batch_notify_dynamic_kernel(lambda m, n, k, i: (2, -1, 0))

    plan = _dynamic_plan(kernel)
    with pytest.raises(ValueError, match="duplicate event coords"):
        validate_lowering_plan(plan)


def test_dynamic_rejects_unstable_notify_coord_count():
    """Reject notify mappings whose coord_count changes across notify_i."""
    kernel = _batch_notify_dynamic_kernel(lambda m, n, k, i: (2 if i == 0 else 1, -1, i), init_count=1)

    plan = _dynamic_plan(kernel)
    with pytest.raises(ValueError, match="coord_count must be stable"):
        validate_lowering_plan(plan)

def _fanout_dynamic_kernel(inv_coord):
    kernel = KernelSpec("dynamic_bad_fanout")
    tmp = kernel.tensor("tmp", (2, 2), "float32")
    ready = kernel.event("ready", (2,), init_count=1)
    done = kernel.event("done", (1,), init_count=4)

    kernel.tile(
        "producer",
        EmptyTile(),
        (2, 1, 1),
        writes=[tmp.region(lambda m, n, k: R[m, 0])],
    ).notify(D(ready, lambda m, n, k, i: (1, -1, m)))
    kernel.tile(
        "consumer",
        EmptyTile(),
        (2, 2, 1),
        reads=[tmp.region(lambda m, n, k: R[m, 0])],
        writes=[tmp.region(lambda m, n, k: R[m, n])],
    ).wait(D(ready, lambda m, n, k, i: (1, -1, m), inv_coord=inv_coord)).notify(
        D(done, lambda m, n, k, i: (1, -1, 0))
    )
    kernel.tile("endpoint", EmptyTile(), (1, 1, 1)).wait(
        D(done, lambda m, n, k, i: (1, -1, 0), inv_coord=lambda rank, e, i: (1, 0, 0, 0))
    )
    return kernel


def test_dynamic_rejects_duplicate_inv_coord_fanout_tiles():
    """Reject fan-out inv_coord mappings that enqueue the same consumer tile twice."""
    kernel = _fanout_dynamic_kernel(lambda rank, m, i: (2, m, 0, 0))

    plan = _dynamic_plan(kernel)
    with pytest.raises(ValueError, match="duplicate consumer tiles"):
        validate_lowering_plan(plan)


def test_dynamic_rejects_fanout_inv_coord_roundtrip_mismatch():
    """Reject fan-out entries whose consumer tile coord maps to a different event coord."""
    kernel = _fanout_dynamic_kernel(lambda rank, m, i: (2, m if i == 0 else 0, i, 0))

    plan = _dynamic_plan(kernel)
    with pytest.raises(ValueError, match="round-trip"):
        validate_lowering_plan(plan)


def test_dynamic_rejects_inv_coord_outside_consumer_grid():
    """Reject statically known inv_coord tile coords outside the consumer grid."""
    kernel = _fanout_dynamic_kernel(lambda rank, m, i: (2, m, 2 + i, 0))

    plan = _dynamic_plan(kernel)
    with pytest.raises(ValueError, match="outside consumer grid"):
        validate_lowering_plan(plan)

def _r1(tensor):
    return tensor.region(lambda m, n, k: R[m])

def _r2(tensor):
    return tensor.region(lambda m, n, k: R[m, n])

def _r2_first_col(tensor):
    return tensor.region(lambda m, n, k: R[m, 0])

def _lowering_plan():
    kernel = KernelSpec("plan")
    tensor = kernel.tensor("x", (4, 3), "float32")
    ready = kernel.event("ready", (4, 3), init_count=1)
    done = kernel.event("done", (4,), init_count=3)

    kernel.tile("producer", EmptyTile(), (4, 3, 1), reads=[_r2(tensor)]).notify(D(ready, lambda m, n, k, i: (1, -1, m, n)))
    kernel.tile("middle", EmptyTile(), (4, 3, 1), reads=[_r2(tensor)]).wait(D(ready, lambda m, n, k, i: (1, -1, m, n))).notify(D(done, lambda m, n, k, i: (1, -1, m,)))
    kernel.tile("consumer", EmptyTile(), (4, 1, 1), reads=[_r2_first_col(tensor)]).wait(D(done, lambda m, n, k, i: (1, -1, m,)))

    validate_kernel(kernel)
    return prepare_static_lowering_plan(kernel, LoweringOptions(attrs={"sm_count": 2}))

def test_lower_prepare_static_schedule_covers_tile_jobs():
    """Validate static schedules cover tile job ids without reserved-id collisions."""
    plan = _lowering_plan()

    validate_static_lowering_plan(plan)

    assert plan.static_schedule is not None
    phase_job_ids = {phase.job_id for phase in plan.static_schedule.phases}
    tile_job_ids = {tile_plan.job_id for tile_plan in plan.tile_plans}
    reserved_job_ids = {
        INIT_EVENT_JOB_ID,
        WAIT_EVENT_INIT_JOB_ID,
        plan.static_schedule.end_job_id,
    }

    assert tile_job_ids <= phase_job_ids
    assert tile_job_ids.isdisjoint(reserved_job_ids)

def test_lower_prepare_rejects_event_layout_overlap():
    """Reject overlapping event workspace regions in the lowering plan."""
    plan = _lowering_plan()
    plan.event_layouts[1] = type(plan.event_layouts[1])(
        event=plan.event_layouts[1].event,
        name=plan.event_layouts[1].name,
        shape=plan.event_layouts[1].shape,
        dtype=plan.event_layouts[1].dtype,
        workspace_offset=8,
        size=plan.event_layouts[1].size,
    )
    plan.event_layout_map[plan.event_layouts[1].name] = plan.event_layouts[1]

    with pytest.raises(ValueError, match="overlap"):
        validate_static_lowering_plan(plan)
