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
"""Static lowering preparation tests for megakernel lowering."""

import pytest

from tvm.megakernel.dsl import KernelSpec, R, TileImpl
from tvm.megakernel.transform.lower import LoweringOptions
from tvm.megakernel.transform.lower.prepare import (
    INIT_EVENT_JOB_ID,
    WAIT_EVENT_INIT_JOB_ID,
    prepare_static_lowering_plan,
)
from tvm.megakernel.transform.lower.validate import validate_static_lowering_plan
from tvm.megakernel.transform.semantic import build_semantic_plan, validate_semantic_plan


class EmptyTile(TileImpl):
    def run(self, m_idx, n_idx, k_idx):
        pass


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

    kernel.tile("producer", EmptyTile(), (4, 3, 1), reads=[_r2(tensor)]).notify(
        ready, lambda m, n, k: (m, n)
    )
    kernel.tile("middle", EmptyTile(), (4, 3, 1), reads=[_r2(tensor)]).wait(
        ready, lambda m, n, k: (m, n)
    ).notify(done, lambda m, n, k: (m,))
    kernel.tile("consumer", EmptyTile(), (4, 1, 1), reads=[_r2_first_col(tensor)]).wait(
        done, lambda m, n, k: (m,)
    )

    semantic = validate_semantic_plan(build_semantic_plan(kernel))
    return prepare_static_lowering_plan(semantic, LoweringOptions(attrs={"sm_count": 2}))



def test_lower_prepare_uses_var_range_upper_bound_for_event_layout():
    kernel = KernelSpec("plan_symbolic_event")
    rows = kernel.var("rows", bounds=(1, 4))
    groups = kernel.var("groups", bounds=(2, 8))
    tensor = kernel.tensor("x", (rows, groups), "float32")
    ready = kernel.event("ready", (rows, groups), init_count=1)
    done = kernel.event("done", (rows,), init_count=1)

    kernel.tile("producer", EmptyTile(), (rows, groups, 1), reads=[_r2(tensor)]).notify(
        ready, lambda m, n, k: (m, n)
    )
    kernel.tile("middle", EmptyTile(), (rows, 1, 1), reads=[_r2_first_col(tensor)]).wait(
        ready, lambda m, n, k: (m, 0)
    ).notify(done, lambda m, n, k: (m,))
    kernel.tile("consumer", EmptyTile(), (rows, 1, 1), reads=[_r2_first_col(tensor)]).wait(
        done, lambda m, n, k: (m,)
    )

    semantic = validate_semantic_plan(build_semantic_plan(kernel))
    plan = prepare_static_lowering_plan(semantic, LoweringOptions(attrs={"sm_count": 2}))

    assert [(event.name, event.workspace_offset, event.size) for event in plan.event_layouts] == [
        ("ready", 0, 32),
        ("done", 32, 4),
    ]
    assert plan.event_init_complete_layout.workspace_offset == 36


def test_lower_prepare_rejects_unbounded_var_in_event_shape():
    kernel = KernelSpec("plan_unbounded_event")
    rows = kernel.var("rows")
    tensor = kernel.tensor("x", (rows,), "float32")
    ready = kernel.event("ready", (rows,), init_count=1)

    kernel.tile("producer", EmptyTile(), (rows, 1, 1), reads=[_r1(tensor)]).notify(
        ready, lambda m, n, k: (m,)
    )
    kernel.tile("consumer", EmptyTile(), (rows, 1, 1), reads=[_r1(tensor)]).wait(
        ready, lambda m, n, k: (m,)
    )

    semantic = validate_semantic_plan(build_semantic_plan(kernel))
    with pytest.raises(ValueError, match="without bounds"):
        prepare_static_lowering_plan(semantic, LoweringOptions(attrs={"sm_count": 2}))


def test_lower_prepare_uses_var_expression_upper_bound_for_event_layout():
    kernel = KernelSpec("plan_expr_event")
    rows = kernel.var("rows", bounds=(1, 9))
    blocks = rows.ceildiv(4)
    tensor = kernel.tensor("x", (rows + 1,), "float32")
    ready = kernel.event("ready", (blocks,), init_count=1)

    kernel.tile("producer", EmptyTile(), (blocks, 1, 1), reads=[tensor]).notify(
        ready, lambda m, n, k: (m,)
    )
    kernel.tile("consumer", EmptyTile(), (blocks, 1, 1), reads=[tensor]).wait(
        ready, lambda m, n, k: (m,)
    )

    semantic = validate_semantic_plan(build_semantic_plan(kernel))
    plan = prepare_static_lowering_plan(semantic, LoweringOptions(attrs={"sm_count": 2}))

    assert [(event.name, event.workspace_offset, event.size) for event in plan.event_layouts] == [
        ("ready", 0, 3),
    ]
    assert plan.event_init_complete_layout.workspace_offset == 3


def test_lower_prepare_rejects_unbounded_var_expression_in_event_shape():
    kernel = KernelSpec("plan_unbounded_expr_event")
    rows = kernel.var("rows")
    blocks = rows.ceildiv(4)
    tensor = kernel.tensor("x", (rows,), "float32")
    ready = kernel.event("ready", (blocks,), init_count=1)

    kernel.tile("producer", EmptyTile(), (blocks, 1, 1), reads=[tensor]).notify(
        ready, lambda m, n, k: (m,)
    )
    kernel.tile("consumer", EmptyTile(), (blocks, 1, 1), reads=[tensor]).wait(
        ready, lambda m, n, k: (m,)
    )

    semantic = validate_semantic_plan(build_semantic_plan(kernel))
    with pytest.raises(ValueError, match="without bounds"):
        prepare_static_lowering_plan(semantic, LoweringOptions(attrs={"sm_count": 2}))


def test_lower_prepare_static_schedule_covers_tile_jobs():
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
