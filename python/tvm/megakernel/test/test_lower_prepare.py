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
"""KernelSpec to lowering-plan preparation tests."""

from __future__ import annotations

import pytest

import tvm.tirx.script as T
from tvm.tirx.script import tile as Tx

from tvm.megakernel.dsl import D, KernelSpec, R, TileImpl
from tvm.megakernel.transform.lower import LoweringOptions
from tvm.megakernel.transform.lower.prepare import TensorBinding, prepare_static_lowering_plan, prepare_lowering_plan, replace_dsl_refs
from tvm.megakernel.transform.lower.event import dependency_info_from_map
from tvm.megakernel.transform.validate import validate_kernel, validate_lowering_plan


class EmptyTile(TileImpl):
    def run(self, m_idx, n_idx, k_idx):
        pass


class FakeExpr:
    def __init__(self, text):
        self.text = text

    def __add__(self, other):
        return FakeExpr(f"({self.text}+{other})")

    def __eq__(self, other):
        return isinstance(other, FakeExpr) and self.text == other.text

    def __repr__(self):
        return self.text

class FakeBuffer:
    def __init__(self, name):
        self.name = name

    def __getitem__(self, index):
        return FakeExpr(f"{self.name}[{index!r}]")

    def __eq__(self, other):
        return isinstance(other, FakeBuffer) and self.name == other.name

    def __hash__(self):
        return hash(self.name)

def test_dependency_coord_binds_closure_tensors_before_call():
    """Bind closure-captured TensorSpec objects to TIRX buffers before calling coord."""
    kernel = KernelSpec("dependency_bind")
    buf1 = kernel.tensor("buf1", (8,), "int32")
    buf2 = kernel.tensor("buf2", (8,), "int32")
    event = kernel.event("ready", (1,), init_count=1)

    def make_dependency():
        return D(event, lambda m, n, k, i: (1, -1, buf1[buf2[i] + m]))

    dependency = make_dependency()
    bindings = {
        buf1: TensorBinding(buf1, "buf1", FakeBuffer("buf1")),
        buf2: TensorBinding(buf2, "buf2", FakeBuffer("buf2")),
    }

    assert dependency_info_from_map(dependency, 3, 0, 0, 2, tensor_bindings=bindings) == (
        1,
        -1,
        FakeExpr("buf1[(buf2[2]+3)]"),
    )

def test_replace_dsl_refs_lowers_captured_symbolic_exprs():
    """Replace TileImpl-captured VarSpec/ExprSpec as runtime values or static upper bounds."""
    kernel = KernelSpec("captured_expr")
    rows = kernel.var("rows", bounds=(2, 8))
    cols = kernel.var("cols", bounds=(4, 16))
    tensor = kernel.tensor("x", (rows, cols), "float32")
    plan = prepare_lowering_plan(kernel, LoweringOptions())
    plan.var_bindings[rows].value = FakeExpr("rows")
    plan.var_bindings[cols].value = FakeExpr("cols")
    bindings = {tensor: TensorBinding(tensor, "x", FakeBuffer("x"))}

    captured = {"buffer": tensor, "extent": cols, "expr": rows + cols}

    assert replace_dsl_refs(captured, bindings, plan, expr_mode="upper_bound") == {
        "buffer": FakeBuffer("x"),
        "extent": 16,
        "expr": 24,
    }
    assert replace_dsl_refs(captured, bindings, plan, expr_mode="runtime") == {
        "buffer": FakeBuffer("x"),
        "extent": FakeExpr("cols"),
        "expr": FakeExpr("(rows+cols)"),
    }


def test_dependency_coord_replaces_bare_tensor_return_value():
    """Replace bare TensorSpec values returned from dependency coord with bound buffers."""
    kernel = KernelSpec("dependency_bind_bare")
    buf = kernel.tensor("buf", (8,), "int32")
    event = kernel.event("ready", (1,), init_count=1)

    def make_dependency():
        return D(event, lambda m, n, k, i: (1, -1, buf))

    fake = FakeBuffer("buf")
    bindings = {buf: TensorBinding(buf, "buf", fake)}

    assert dependency_info_from_map(make_dependency(), 0, 0, 0, 0, tensor_bindings=bindings) == (
        1,
        -1,
        fake,
    )

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

def test_lower_prepare_uses_var_range_upper_bound_for_event_layout():
    """Use bounded VarSpec upper bounds to size event workspace layouts."""
    kernel = KernelSpec("plan_symbolic_event")
    rows = kernel.var("rows", bounds=(1, 4))
    groups = kernel.var("groups", bounds=(2, 8))
    tensor = kernel.tensor("x", (rows, groups), "float32")
    ready = kernel.event("ready", (rows, groups), init_count=1)
    done = kernel.event("done", (rows,), init_count=1)

    kernel.tile("producer", EmptyTile(), (rows, groups, 1), reads=[_r2(tensor)]).notify(D(ready, lambda m, n, k, i: (1, -1, m, n)))
    kernel.tile("middle", EmptyTile(), (rows, 1, 1), reads=[_r2_first_col(tensor)]).wait(D(ready, lambda m, n, k, i: (1, -1, m, 0))).notify(D(done, lambda m, n, k, i: (1, -1, m,)))
    kernel.tile("consumer", EmptyTile(), (rows, 1, 1), reads=[_r2_first_col(tensor)]).wait(D(done, lambda m, n, k, i: (1, -1, m,)))

    validate_kernel(kernel)
    plan = prepare_static_lowering_plan(kernel, LoweringOptions(attrs={"sm_count": 2}))

    assert [(event.name, event.workspace_offset, event.size) for event in plan.event_layouts] == [
        ("ready", 0, 32),
        ("done", 32, 4),
    ]
    assert plan.event_init_complete_layout.workspace_offset == 36

def test_lower_prepare_rejects_unbounded_var_in_event_shape():
    """Reject event layout sizing when an event shape contains an unbounded var."""
    kernel = KernelSpec("plan_unbounded_event")
    rows = kernel.var("rows")
    tensor = kernel.tensor("x", (rows,), "float32")
    ready = kernel.event("ready", (rows,), init_count=1)

    kernel.tile("producer", EmptyTile(), (rows, 1, 1), reads=[_r1(tensor)]).notify(D(ready, lambda m, n, k, i: (1, -1, m,)))
    kernel.tile("consumer", EmptyTile(), (rows, 1, 1), reads=[_r1(tensor)]).wait(D(ready, lambda m, n, k, i: (1, -1, m,)))

    validate_kernel(kernel)
    with pytest.raises(ValueError, match="without bounds"):
        prepare_static_lowering_plan(kernel, LoweringOptions(attrs={"sm_count": 2}))

def test_lower_prepare_uses_var_expression_upper_bound_for_event_layout():
    """Use bounded VarSpec expression upper bounds to size event layouts."""
    kernel = KernelSpec("plan_expr_event")
    rows = kernel.var("rows", bounds=(1, 9))
    blocks = rows.ceildiv(4)
    tensor = kernel.tensor("x", (rows + 1,), "float32")
    ready = kernel.event("ready", (blocks,), init_count=1)

    kernel.tile("producer", EmptyTile(), (blocks, 1, 1), reads=[tensor]).notify(D(ready, lambda m, n, k, i: (1, -1, m,)))
    kernel.tile("consumer", EmptyTile(), (blocks, 1, 1), reads=[tensor]).wait(D(ready, lambda m, n, k, i: (1, -1, m,)))

    validate_kernel(kernel)
    plan = prepare_static_lowering_plan(kernel, LoweringOptions(attrs={"sm_count": 2}))

    assert [(event.name, event.workspace_offset, event.size) for event in plan.event_layouts] == [
        ("ready", 0, 3),
    ]
    assert plan.event_init_complete_layout.workspace_offset == 3

def test_lower_prepare_rejects_unbounded_var_expression_in_event_shape():
    """Reject event layout sizing for unbounded symbolic expressions."""
    kernel = KernelSpec("plan_unbounded_expr_event")
    rows = kernel.var("rows")
    blocks = rows.ceildiv(4)
    tensor = kernel.tensor("x", (rows,), "float32")
    ready = kernel.event("ready", (blocks,), init_count=1)

    kernel.tile("producer", EmptyTile(), (blocks, 1, 1), reads=[tensor]).notify(D(ready, lambda m, n, k, i: (1, -1, m,)))
    kernel.tile("consumer", EmptyTile(), (blocks, 1, 1), reads=[tensor]).wait(D(ready, lambda m, n, k, i: (1, -1, m,)))

    validate_kernel(kernel)
    with pytest.raises(ValueError, match="without bounds"):
        prepare_static_lowering_plan(kernel, LoweringOptions(attrs={"sm_count": 2}))

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

def test_dynamic_prepare_records_entry_and_trigger():
    """Prepare dynamic entry phases, triggers, and endpoint from the dependency graph."""
    plan = prepare_lowering_plan(_simple_dynamic_kernel(), _options())
    validate_lowering_plan(plan)

    assert plan.dynamic_schedule is not None
    assert [phase.label for phase in plan.dynamic_schedule.entry_phases] == ["producer"]
    assert [(trigger.event.name, trigger.consumer.tile.name) for trigger in plan.dynamic_schedule.triggers["producer"]] == [
        ("ready", "consumer")
    ]
    assert plan.dynamic_schedule.endpoint.tile.name == "consumer"
