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
"""Semantic validation tests for megakernel DSL graphs."""

from __future__ import annotations

import pytest

import tvm.tirx.script as T
from tvm.tirx.script import tile as Tx

from tvm.megakernel.dsl import KernelSpec, R, TileImpl
from tvm.megakernel.dsl.spec import DependencySpec
from tvm.megakernel.transform.semantic import build_semantic_plan, validate_semantic_plan


class EmptyTile(TileImpl):
    def run(self, m_idx, n_idx, k_idx):
        pass


def _validate(kernel):
    return validate_semantic_plan(build_semantic_plan(kernel))


def _r1(tensor):
    return tensor.region(lambda m, n, k: R[m])


def _r2(tensor):
    return tensor.region(lambda m, n, k: R[m, n])


def _r2_first_col(tensor):
    return tensor.region(lambda m, n, k: R[m, 0])


def _basic_kernel(name="semantic"):
    kernel = KernelSpec(name)
    tensor = kernel.tensor("x", (4, 3), "float32")
    ready = kernel.event("ready", (4, 3), init_count=1)
    producer = kernel.tile("producer", EmptyTile(), (4, 3, 1), reads=[_r2(tensor)])
    consumer = kernel.tile("consumer", EmptyTile(), (4, 3, 1), reads=[_r2(tensor)])
    producer.notify(ready, lambda m, n, k: (m, n))
    consumer.wait(ready, lambda m, n, k: (m, n))
    return kernel, tensor, ready, producer, consumer


def _symbolic_kernel(name="semantic_symbolic"):
    kernel = KernelSpec(name)
    rows = kernel.var("rows")
    groups = kernel.var("groups")
    tensor = kernel.tensor("x", (rows, groups), "float32")
    ready = kernel.event("ready", (rows, groups), init_count=1)
    producer = kernel.tile("producer", EmptyTile(), (rows, groups, 1), reads=[_r2(tensor)])
    consumer = kernel.tile("consumer", EmptyTile(), (rows, groups, 1), reads=[_r2(tensor)])
    producer.notify(ready, lambda m, n, k: (m, n))
    consumer.wait(ready, lambda m, n, k: (m, n))
    return kernel, ready, producer, consumer


def test_semantic_accepts_valid_static_graph():
    kernel, _, _, _, _ = _basic_kernel()

    plan = _validate(kernel)

    assert [(edge.producer.name, edge.consumer.name, edge.event.name) for edge in plan.logical_edges] == [
        ("producer", "consumer", "ready")
    ]


def test_semantic_rejects_duplicate_tile_name():
    kernel, _, _, producer, _ = _basic_kernel()
    kernel.tiles.append(producer)

    with pytest.raises(ValueError, match="duplicate tile names"):
        _validate(kernel)


def test_semantic_rejects_foreign_tensor():
    kernel, _, _, producer, _ = _basic_kernel()
    foreign = KernelSpec("foreign_tensor").tensor("foreign", (1,), "float32")
    producer.reads.append(foreign.region(lambda m, n, k: R[0]))

    with pytest.raises(ValueError, match="tensor outside kernel"):
        _validate(kernel)


def test_semantic_allows_bare_tensor_access_with_event_dependency():
    kernel = KernelSpec("semantic_bare_tensor")
    tensor = kernel.tensor("x", (1,), "float32")
    ready = kernel.event("ready", (1,), init_count=1)

    kernel.tile("writer", EmptyTile(), (1, 1, 1), writes=[tensor]).notify(
        ready, lambda m, n, k: (0,)
    )
    kernel.tile("reader", EmptyTile(), (1, 1, 1), reads=[tensor]).wait(
        ready, lambda m, n, k: (0,)
    )

    _validate(kernel)


def test_semantic_rejects_foreign_notify_event():
    kernel, _, _, producer, _ = _basic_kernel()
    foreign = KernelSpec("foreign_event").event("foreign", (1,), 1)
    producer.notifies[0] = DependencySpec(foreign, lambda m, n, k: (0,))

    with pytest.raises(ValueError, match="event outside kernel"):
        _validate(kernel)


def test_semantic_rejects_foreign_wait_event():
    kernel, _, _, _, consumer = _basic_kernel()
    foreign = KernelSpec("foreign_event").event("foreign", (1,), 1)
    consumer.waits[0] = DependencySpec(foreign, lambda m, n, k: (0,))

    with pytest.raises(ValueError, match="event outside kernel"):
        _validate(kernel)


@pytest.mark.parametrize(
    "coord,error,match",
    [
        pytest.param(lambda m, n, k: m, TypeError, "tuple/list", id="not-tuple"),
        pytest.param(lambda m, n, k: (m, n, k), ValueError, "coord rank", id="rank-mismatch"),
        pytest.param(lambda m, n, k: (m, None), TypeError, "unsupported value", id="bad-value"),
    ],
)
def test_semantic_rejects_invalid_coord_shape(coord, error, match):
    kernel, _, ready, producer, _ = _basic_kernel()
    producer.notifies[0] = DependencySpec(ready, coord)

    with pytest.raises(error, match=match):
        _validate(kernel)


def test_semantic_rejects_wait_without_producer():
    kernel, _, _, producer, _ = _basic_kernel()
    producer.notifies.clear()

    with pytest.raises(ValueError, match="no producer"):
        _validate(kernel)


def test_semantic_rejects_notify_without_consumer():
    kernel, _, _, _, consumer = _basic_kernel()
    consumer.waits.clear()

    with pytest.raises(ValueError, match="no consumer"):
        _validate(kernel)


def test_semantic_rejects_notify_count_mismatch():
    kernel, _, ready, producer, _ = _basic_kernel()
    producer.notifies[0] = DependencySpec(ready, lambda m, n, k: (m, 0))

    with pytest.raises(ValueError, match="init_count"):
        _validate(kernel)


def test_semantic_rejects_wait_out_of_bounds():
    kernel, _, ready, _, consumer = _basic_kernel()
    consumer.waits[0] = DependencySpec(ready, lambda m, n, k: (m, 3))

    with pytest.raises(ValueError, match="out of bounds"):
        _validate(kernel)


def test_semantic_rejects_wait_without_matching_notify_coord():
    kernel = KernelSpec("semantic_wait_coord")
    tensor = kernel.tensor("x", (4, 2), "float32")
    ready = kernel.event(
        "ready",
        (4, 2),
        init_count=lambda m, n: 1 if n == 0 else 0,
    )

    kernel.tile("producer", EmptyTile(), (4, 1, 1), reads=[_r2_first_col(tensor)]).notify(
        ready, lambda m, n, k: (m, 0)
    )
    kernel.tile(
        "consumer",
        EmptyTile(),
        (4, 1, 1),
        reads=[tensor.region(lambda m, n, k: R[m, 1])],
    ).wait(ready, lambda m, n, k: (m, 1))

    with pytest.raises(ValueError, match="without a producer notify"):
        _validate(kernel)


def test_semantic_accepts_valid_symbolic_graph():
    kernel, _, _, _ = _symbolic_kernel()

    _validate(kernel)


def test_semantic_samples_symbolic_notify_count_mismatch():
    kernel, ready, producer, _ = _symbolic_kernel()
    producer.notifies[0] = DependencySpec(ready, lambda m, n, k: (m, 0))

    with pytest.raises(ValueError, match="init_count"):
        _validate(kernel)


def test_semantic_samples_symbolic_wait_swapped_coords():
    kernel, ready, _, consumer = _symbolic_kernel()
    consumer.waits[0] = DependencySpec(ready, lambda m, n, k: (n, m))

    with pytest.raises(ValueError, match="out of bounds|without a producer notify"):
        _validate(kernel)


def test_semantic_samples_symbolic_range_bounds():
    kernel = KernelSpec("semantic_symbolic_range")
    rows = kernel.var("rows", bounds=(20, 24))
    tensor = kernel.tensor("x", (rows,), "float32")
    ready = kernel.event("ready", (rows,), init_count=1)

    kernel.tile("producer", EmptyTile(), (16, 1, 1), reads=[_r1(tensor)]).notify(
        ready, lambda m, n, k: (m,)
    )
    kernel.tile("consumer", EmptyTile(), (16, 1, 1), reads=[_r1(tensor)]).wait(
        ready, lambda m, n, k: (m,)
    )

    with pytest.raises(ValueError, match="init_count"):
        _validate(kernel)


def test_kernel_var_rejects_invalid_bounds():
    kernel = KernelSpec("invalid_var_range")

    with pytest.raises(ValueError, match="bounds"):
        kernel.var("rows", bounds=(8, 1))


def test_kernel_normalizes_tensor_and_event_shape():
    kernel = KernelSpec("normalize_shape")
    rows = kernel.var("rows", bounds=(1, 4))

    tensor = kernel.tensor("x", rows, "float32")
    event = kernel.event("ready", rows, init_count=1)

    assert tensor.shape == (rows,)
    assert event.shape == (rows,)


def test_kernel_rejects_invalid_shape_dim():
    kernel = KernelSpec("invalid_shape")

    with pytest.raises(TypeError, match="shape dims"):
        kernel.tensor("x", (True,), "float32")
    with pytest.raises(TypeError, match="shape dims"):
        kernel.event("ready", (object(),), init_count=1)


def test_kernel_normalizes_event_init_count_to_callable():
    kernel = KernelSpec("event_init_count_callable")

    events = [kernel.event(f"ready_{count}", (2,), init_count=count) for count in (0, 3)]
    dynamic = kernel.event("dynamic", (2, 3), init_count=lambda m, n: m + n)

    assert events[0].init_count(0) == 0
    assert events[1].init_count(0) == 3
    assert dynamic.init_count(1, 2) == 3


def test_kernel_rejects_invalid_event_init_count():
    kernel = KernelSpec("invalid_event_init_count")

    for value in (True, -1):
        with pytest.raises((TypeError, ValueError), match="init_count"):
            kernel.event(f"ready_{value}", (1,), init_count=value)
    with pytest.raises(TypeError, match="init_count"):
        kernel.event("ready_bad", (1,), init_count="x")


def test_kernel_normalizes_tile_grid():
    kernel = KernelSpec("normalize_grid")
    tensor = kernel.tensor("x", (1,), "float32")

    tile = kernel.tile("only", EmptyTile(), [1, 1, 1], reads=[tensor])

    assert tile.grid == (1, 1, 1)


def test_kernel_rejects_invalid_tile_grid_and_impl():
    kernel = KernelSpec("invalid_grid")
    tensor = kernel.tensor("x", (1,), "float32")

    with pytest.raises(TypeError, match="TileImpl"):
        kernel.tile("bad_impl", object(), (1, 1, 1), reads=[tensor])
    with pytest.raises(ValueError, match="three dimensions"):
        kernel.tile("bad_rank", EmptyTile(), (1, 1), reads=[tensor])
    with pytest.raises(TypeError, match="grid dims"):
        kernel.tile("bad_dim", EmptyTile(), (1, True, 1), reads=[tensor])


def test_tile_wait_notify_normalizes_constant_coord():
    kernel = KernelSpec("normalize_coord")
    event = kernel.event("ready", (1,), init_count=1)
    tensor = kernel.tensor("x", (1,), "float32")

    tile = kernel.tile("only", EmptyTile(), (1, 1, 1), reads=[tensor])
    tile.wait(event, [0], inverse_coord=[0, 0, 0]).notify(event, [0])

    assert tile.waits[0].coord == (0,)
    assert tile.waits[0].inverse_coord == (0, 0, 0)
    assert tile.notifies[0].coord == (0,)


def test_tile_wait_notify_rejects_invalid_coord():
    kernel = KernelSpec("invalid_coord")
    event = kernel.event("ready", (1,), init_count=1)
    tile = kernel.tile("only", EmptyTile(), (1, 1, 1))

    with pytest.raises(TypeError, match="coord"):
        tile.wait(event, 0)
    with pytest.raises(TypeError, match="inverse_coord"):
        tile.wait(event, (0,), inverse_coord=0)
    with pytest.raises(TypeError, match="coord"):
        tile.notify(event, 0)


def test_semantic_accepts_varspec_expression_in_shape_event_and_grid():
    kernel = KernelSpec("semantic_expr_shape")
    rows = kernel.var("rows", bounds=(1, 9))
    blocks = rows.ceildiv(4)
    tensor = kernel.tensor("x", (rows + 1,), "float32")
    ready = kernel.event("ready", (blocks,), init_count=1)

    kernel.tile(
        "producer", EmptyTile(), (blocks, 1, 1), writes=[tensor.region(lambda m, n, k: R[m])]
    ).notify(ready, lambda m, n, k: (m,))
    kernel.tile(
        "consumer", EmptyTile(), (blocks, 1, 1), reads=[tensor.region(lambda m, n, k: R[m])]
    ).wait(ready, lambda m, n, k: (m,))

    plan = _validate(kernel)

    assert [var.name for var in plan.vars] == ["rows"]


def test_semantic_rejects_foreign_varspec_inside_expression():
    kernel = KernelSpec("semantic_expr_foreign_var")
    rows = kernel.var("rows", bounds=(1, 4))
    foreign = KernelSpec("foreign").var("foreign", bounds=(1, 4))

    kernel.tensor("x", (rows + foreign,), "float32")

    with pytest.raises(ValueError, match="VarSpec outside this kernel"):
        _validate(kernel)


def test_semantic_accepts_region_dependency_covered_by_event():
    kernel = KernelSpec("semantic_region_dep")
    tensor = kernel.tensor("x", (4,), "float32")
    ready = kernel.event("ready", (4,), init_count=1)

    kernel.tile(
        "producer",
        EmptyTile(),
        (4, 1, 1),
        writes=[tensor.region(lambda m, n, k: R[m])],
    ).notify(ready, lambda m, n, k: (m,))
    kernel.tile(
        "consumer",
        EmptyTile(),
        (4, 1, 1),
        reads=[tensor.region(lambda m, n, k: R[m])],
    ).wait(ready, lambda m, n, k: (m,))

    _validate(kernel)


def test_semantic_allows_region_read_from_external_input():
    kernel = KernelSpec("semantic_region_external_input")
    tensor = kernel.tensor("x", (4,), "float32")

    kernel.tile(
        "consumer", EmptyTile(), (4, 1, 1), reads=[tensor.region(lambda m, n, k: R[m])]
    )

    _validate(kernel)


def test_semantic_rejects_waited_coord_that_does_not_write_read_region():
    kernel = KernelSpec("semantic_waited_coord_wrong_region")
    tensor = kernel.tensor("x", (2,), "float32")
    ready = kernel.event("ready", (1,), init_count=1)

    kernel.tile(
        "producer",
        EmptyTile(),
        (1, 1, 1),
        writes=[tensor.region(lambda m, n, k: R[0])],
    ).notify(ready, lambda m, n, k: (0,))
    kernel.tile(
        "consumer",
        EmptyTile(),
        (1, 1, 1),
        reads=[tensor.region(lambda m, n, k: R[1])],
    ).wait(ready, lambda m, n, k: (0,))

    with pytest.raises(
        ValueError,
        match=r"waits on event 'ready' coord \(0,\).*reads tensor 'x' region \[1\]",
    ):
        _validate(kernel)


def test_semantic_accepts_shifted_region_dependency_when_wait_coord_matches_writer():
    kernel = KernelSpec("semantic_shifted_region_dep")
    tensor = kernel.tensor("x", (4,), "float32")
    ready = kernel.event("ready", (4,), init_count=1)

    kernel.tile(
        "producer",
        EmptyTile(),
        (4, 1, 1),
        writes=[tensor.region(lambda m, n, k: R[m])],
    ).notify(ready, lambda m, n, k: (m,))
    kernel.tile(
        "consumer",
        EmptyTile(),
        (3, 1, 1),
        reads=[tensor.region(lambda m, n, k: R[m + 1])],
    ).wait(ready, lambda m, n, k: (m + 1,))

    _validate(kernel)


def test_semantic_rejects_shifted_region_dependency_when_wait_coord_is_wrong():
    kernel = KernelSpec("semantic_shifted_region_wrong_wait")
    tensor = kernel.tensor("x", (4,), "float32")
    ready = kernel.event("ready", (4,), init_count=1)

    kernel.tile(
        "producer",
        EmptyTile(),
        (4, 1, 1),
        writes=[tensor.region(lambda m, n, k: R[m])],
    ).notify(ready, lambda m, n, k: (m,))
    kernel.tile(
        "consumer",
        EmptyTile(),
        (3, 1, 1),
        reads=[tensor.region(lambda m, n, k: R[m + 1])],
    ).wait(ready, lambda m, n, k: (m,))

    with pytest.raises(
        ValueError,
        match=r"reads tensor 'x' region \[1\].*producer' idx \(1, 0, 0\).*without an event dependency",
    ):
        _validate(kernel)


def test_semantic_rejects_region_read_without_event_dependency():
    kernel = KernelSpec("semantic_region_no_event_dep")
    tensor = kernel.tensor("x", (4,), "float32")

    kernel.tile(
        "producer", EmptyTile(), (4, 1, 1), writes=[tensor.region(lambda m, n, k: R[m])]
    )
    kernel.tile(
        "consumer", EmptyTile(), (4, 1, 1), reads=[tensor.region(lambda m, n, k: R[m])]
    )

    with pytest.raises(ValueError, match="without an event dependency"):
        _validate(kernel)


def test_semantic_rejects_region_out_of_bounds():
    kernel = KernelSpec("semantic_region_oob")
    tensor = kernel.tensor("x", (4,), "float32")
    ready = kernel.event("ready", (4,), init_count=1)

    kernel.tile(
        "producer", EmptyTile(), (4, 1, 1), writes=[tensor.region(lambda m, n, k: R[m])]
    ).notify(ready, lambda m, n, k: (m,))
    kernel.tile(
        "consumer", EmptyTile(), (4, 1, 1), reads=[tensor.region(lambda m, n, k: R[4])]
    ).wait(ready, lambda m, n, k: (m,))

    with pytest.raises(ValueError, match="out of bounds"):
        _validate(kernel)


def test_semantic_accepts_range_region_dependency_covered_by_event():
    kernel = KernelSpec("semantic_range_region_dep")
    tensor = kernel.tensor("x", (2, 16), "float32")
    ready = kernel.event("ready", (2,), init_count=1)

    kernel.tile(
        "producer",
        EmptyTile(),
        (2, 1, 1),
        writes=[tensor.region(lambda m, n, k: R[m, 0:16])],
    ).notify(ready, lambda m, n, k: (m,))
    kernel.tile(
        "consumer",
        EmptyTile(),
        (2, 1, 1),
        reads=[tensor.region(lambda m, n, k: R[m, 4:12])],
    ).wait(ready, lambda m, n, k: (m,))

    _validate(kernel)


def test_semantic_accepts_partially_overlapping_region_with_event_dependency():
    kernel = KernelSpec("semantic_range_region_partial_overlap")
    tensor = kernel.tensor("x", (16,), "float32")
    ready = kernel.event("ready", (1,), init_count=1)

    kernel.tile(
        "producer", EmptyTile(), (1, 1, 1), writes=[tensor.region(lambda m, n, k: R[0:8])]
    ).notify(ready, lambda m, n, k: (0,))
    kernel.tile(
        "consumer", EmptyTile(), (1, 1, 1), reads=[tensor.region(lambda m, n, k: R[4:12])]
    ).wait(ready, lambda m, n, k: (0,))

    _validate(kernel)


def test_semantic_unknown_write_conservatively_covers_static_read():
    kernel = KernelSpec("semantic_dynamic_write")
    tensor = kernel.tensor("x", (16,), "float32")
    ready = kernel.event("ready", (1,), init_count=1)

    kernel.tile(
        "producer", EmptyTile(), (1, 1, 1), writes=[tensor]
    ).notify(ready, lambda m, n, k: (0,))
    kernel.tile(
        "consumer", EmptyTile(), (1, 1, 1), reads=[tensor.region(lambda m, n, k: R[4:8])]
    ).wait(ready, lambda m, n, k: (0,))

    _validate(kernel)


def test_semantic_unknown_read_requires_event_dependency_from_writer():
    kernel = KernelSpec("semantic_dynamic_read_no_dep")
    tensor = kernel.tensor("x", (16,), "float32")

    kernel.tile(
        "producer", EmptyTile(), (1, 1, 1), writes=[tensor.region(lambda m, n, k: R[0:16])]
    )
    kernel.tile(
        "consumer", EmptyTile(), (1, 1, 1), reads=[tensor]
    )

    with pytest.raises(ValueError, match="without an event dependency"):
        _validate(kernel)


def test_semantic_accepts_multiple_producers_for_one_event_coord():
    kernel = KernelSpec("semantic_multi_producer")
    tensor = kernel.tensor("x", (4,), "float32")
    ready = kernel.event("ready", (4,), init_count=2)

    kernel.tile("producer_a", EmptyTile(), (4, 1, 1), reads=[_r1(tensor)]).notify(
        ready, lambda m, n, k: (m,)
    )
    kernel.tile("producer_b", EmptyTile(), (4, 1, 1), reads=[_r1(tensor)]).notify(
        ready, lambda m, n, k: (m,)
    )
    kernel.tile("consumer", EmptyTile(), (4, 1, 1), reads=[_r1(tensor)]).wait(
        ready, lambda m, n, k: (m,)
    )

    plan = _validate(kernel)

    assert sorted((edge.producer.name, edge.consumer.name) for edge in plan.logical_edges) == [
        ("producer_a", "consumer"),
        ("producer_b", "consumer"),
    ]


def test_semantic_accepts_multiple_consumers_waiting_on_one_event_coord():
    kernel = KernelSpec("semantic_multi_consumer")
    tensor = kernel.tensor("x", (4,), "float32")
    ready = kernel.event("ready", (4,), init_count=1)

    kernel.tile("producer", EmptyTile(), (4, 1, 1), reads=[_r1(tensor)]).notify(
        ready, lambda m, n, k: (m,)
    )
    kernel.tile("consumer_a", EmptyTile(), (4, 1, 1), reads=[_r1(tensor)]).wait(
        ready, lambda m, n, k: (m,)
    )
    kernel.tile("consumer_b", EmptyTile(), (4, 1, 1), reads=[_r1(tensor)]).wait(
        ready, lambda m, n, k: (m,)
    )

    plan = _validate(kernel)

    assert sorted((edge.producer.name, edge.consumer.name) for edge in plan.logical_edges) == [
        ("producer", "consumer_a"),
        ("producer", "consumer_b"),
    ]


def test_semantic_rejects_duplicate_notify_event_on_one_tile():
    kernel = KernelSpec("semantic_duplicate_notify_event")
    tensor = kernel.tensor("x", (4,), "float32")
    ready = kernel.event("ready", (4,), init_count=2)

    producer = kernel.tile("producer", EmptyTile(), (4, 1, 1), reads=[_r1(tensor)])
    producer.notify(ready, lambda m, n, k: (m,))
    producer.notify(ready, lambda m, n, k: (m,))
    kernel.tile("consumer", EmptyTile(), (4, 1, 1), reads=[_r1(tensor)]).wait(
        ready, lambda m, n, k: (m,)
    )

    with pytest.raises(ValueError, match="notifies event .* more than once"):
        _validate(kernel)


def test_semantic_rejects_duplicate_wait_event_on_one_tile():
    kernel = KernelSpec("semantic_duplicate_wait_event")
    tensor = kernel.tensor("x", (4,), "float32")
    ready = kernel.event("ready", (4,), init_count=1)

    kernel.tile("producer", EmptyTile(), (4, 1, 1), reads=[_r1(tensor)]).notify(
        ready, lambda m, n, k: (m,)
    )
    consumer = kernel.tile("consumer", EmptyTile(), (4, 1, 1), reads=[_r1(tensor)])
    consumer.wait(ready, lambda m, n, k: (m,))
    consumer.wait(ready, lambda m, n, k: (m,))

    with pytest.raises(ValueError, match="waits on event .* more than once"):
        _validate(kernel)


class ImplCopyTile(TileImpl):
    def __init__(self, src, out):
        super().__init__()
        self.src = src
        self.out = out

    @T.inline
    def run(self, m_idx, n_idx, k_idx):
        Tx.copy(self.out[0:8], self.src[0:8])


def test_semantic_accepts_impl_access_matching_declared_regions():
    kernel = KernelSpec("semantic_impl_access_ok")
    src = kernel.tensor("src", (16,), "float32")
    out = kernel.tensor("out", (16,), "float32")

    kernel.tile(
        "copy",
        ImplCopyTile(src, out),
        (1, 1, 1),
        reads=[src.region(lambda m, n, k: R[0:8])],
        writes=[out.region(lambda m, n, k: R[0:8])],
    )

    _validate(kernel)


def test_semantic_rejects_impl_read_missing_from_tile_reads():
    kernel = KernelSpec("semantic_impl_missing_read")
    src = kernel.tensor("src", (16,), "float32")
    out = kernel.tensor("out", (16,), "float32")

    kernel.tile(
        "copy",
        ImplCopyTile(src, out),
        (1, 1, 1),
        reads=[],
        writes=[out.region(lambda m, n, k: R[0:8])],
    )

    with pytest.raises(ValueError, match="impl reads tensor 'src'"):
        _validate(kernel)


def test_semantic_rejects_impl_write_missing_from_tile_writes():
    kernel = KernelSpec("semantic_impl_missing_write")
    src = kernel.tensor("src", (16,), "float32")
    out = kernel.tensor("out", (16,), "float32")

    kernel.tile(
        "copy",
        ImplCopyTile(src, out),
        (1, 1, 1),
        reads=[src.region(lambda m, n, k: R[0:8])],
        writes=[],
    )

    with pytest.raises(ValueError, match="impl writes tensor 'out'"):
        _validate(kernel)


def test_semantic_rejects_impl_region_outside_declared_region():
    kernel = KernelSpec("semantic_impl_region_too_wide")
    src = kernel.tensor("src", (16,), "float32")
    out = kernel.tensor("out", (16,), "float32")

    kernel.tile(
        "copy",
        ImplCopyTile(src, out),
        (1, 1, 1),
        reads=[src.region(lambda m, n, k: R[0:4])],
        writes=[out.region(lambda m, n, k: R[0:8])],
    )

    with pytest.raises(ValueError, match="outside declared tile.reads regions"):
        _validate(kernel)


class ImplUnknownRegionTile(TileImpl):
    def __init__(self, src, out):
        super().__init__()
        self.src = src
        self.out = out

    @T.inline
    def run(self, m_idx, n_idx, k_idx):
        base = m_idx * 8
        Tx.copy(self.out[base : base + 8], self.src[base : base + 8])


def test_semantic_warns_on_unknown_impl_access_by_default():
    kernel = KernelSpec("semantic_impl_unknown_warn")
    src = kernel.tensor("src", (16,), "float32")
    out = kernel.tensor("out", (16,), "float32")

    kernel.tile(
        "copy",
        ImplUnknownRegionTile(src, out),
        (1, 1, 1),
        reads=[src],
        writes=[out],
    )

    with pytest.warns(UserWarning, match="unknown access effect"):
        _validate(kernel)


def test_semantic_warn_policy_downgrades_impl_region_mismatch():
    kernel = KernelSpec("semantic_impl_mismatch_warn")
    src = kernel.tensor("src", (16,), "float32")
    out = kernel.tensor("out", (16,), "float32")

    kernel.tile(
        "copy",
        ImplCopyTile(src, out),
        (1, 1, 1),
        reads=[src.region(lambda m, n, k: R[0:4])],
        writes=[out.region(lambda m, n, k: R[0:8])],
        attrs={"impl_access_validate": "warn"},
    )

    with pytest.warns(UserWarning, match="outside declared tile.reads regions"):
        _validate(kernel)


def test_semantic_off_policy_skips_impl_access_validation():
    kernel = KernelSpec("semantic_impl_mismatch_off")
    src = kernel.tensor("src", (16,), "float32")
    out = kernel.tensor("out", (16,), "float32")

    kernel.tile(
        "copy",
        ImplCopyTile(src, out),
        (1, 1, 1),
        reads=[],
        writes=[],
        attrs={"impl_access_validate": "off"},
    )

    _validate(kernel)


def test_semantic_rejects_invalid_impl_access_validate_policy():
    kernel = KernelSpec("semantic_impl_bad_policy")
    src = kernel.tensor("src", (16,), "float32")
    out = kernel.tensor("out", (16,), "float32")

    kernel.tile(
        "copy",
        ImplCopyTile(src, out),
        (1, 1, 1),
        reads=[src],
        writes=[out],
        attrs={"impl_access_validate": "maybe"},
    )

    with pytest.raises(ValueError, match="unsupported impl_access_validate"):
        _validate(kernel)
