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

from tvm.megakernel.dsl import D, KernelSpec, R, TileImpl
from tvm.megakernel.dsl.spec import DependencySpec
from tvm.megakernel.transform.validate import logical_edges, validate_kernel


class EmptyTile(TileImpl):
    def run(self, m_idx, n_idx, k_idx):
        pass

GLOBAL_DEP_TENSOR = None

def test_semantic_allows_dependency_runtime_coord_from_closure_tensor():
    """Allow dependency coords that capture same-kernel tensors through closures."""
    kernel = KernelSpec("runtime_coord_closure")
    buf = kernel.tensor("buf", (4,), "int32")
    event = kernel.event("ready", (1,), init_count=1)

    def notify_dep():
        return D(event, lambda m, n, k, i: (1, -1, buf[i]))

    def wait_dep():
        return D(event, lambda m, n, k, i: (1, -1, buf[i]))

    kernel.tile("producer", EmptyTile(), (1, 1, 1)).notify(notify_dep())
    kernel.tile("consumer", EmptyTile(), (1, 1, 1)).wait(wait_dep())

    assert validate_kernel(kernel) is kernel

def test_semantic_rejects_dependency_global_tensor_capture():
    """Reject dependency coords that access TensorSpec values through globals."""
    global GLOBAL_DEP_TENSOR
    kernel = KernelSpec("runtime_coord_global")
    GLOBAL_DEP_TENSOR = kernel.tensor("buf", (4,), "int32")
    event = kernel.event("ready", (1,), init_count=1)

    kernel.tile("producer", EmptyTile(), (1, 1, 1)).notify(
        D(event, lambda m, n, k, i: (1, -1, GLOBAL_DEP_TENSOR[i]))
    )
    kernel.tile("consumer", EmptyTile(), (1, 1, 1)).wait(
        D(event, lambda m, n, k, i: (1, -1, 0))
    )

    with pytest.raises(TypeError, match="closure, not globals"):
        validate_kernel(kernel)
    GLOBAL_DEP_TENSOR = None

def test_semantic_rejects_dependency_default_argument_capture():
    """Reject dependency coords that capture TensorSpec values through default args."""
    kernel = KernelSpec("runtime_coord_default")
    buf = kernel.tensor("buf", (4,), "int32")
    event = kernel.event("ready", (1,), init_count=1)

    kernel.tile("producer", EmptyTile(), (1, 1, 1)).notify(
        D(event, lambda m, n, k, i, buf=buf: (1, -1, buf[i]))
    )
    kernel.tile("consumer", EmptyTile(), (1, 1, 1)).wait(
        D(event, lambda m, n, k, i: (1, -1, 0))
    )

    with pytest.raises(TypeError, match="default arguments"):
        validate_kernel(kernel)

def _validate(kernel):
    return validate_kernel(kernel)

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
    producer.notify(D(ready, lambda m, n, k, i: (1, -1, m, n)))
    consumer.wait(D(ready, lambda m, n, k, i: (1, -1, m, n)))
    return kernel, tensor, ready, producer, consumer

def _symbolic_kernel(name="semantic_symbolic"):
    kernel = KernelSpec(name)
    rows = kernel.var("rows")
    groups = kernel.var("groups")
    tensor = kernel.tensor("x", (rows, groups), "float32")
    ready = kernel.event("ready", (rows, groups), init_count=1)
    producer = kernel.tile("producer", EmptyTile(), (rows, groups, 1), reads=[_r2(tensor)])
    consumer = kernel.tile("consumer", EmptyTile(), (rows, groups, 1), reads=[_r2(tensor)])
    producer.notify(D(ready, lambda m, n, k, i: (1, -1, m, n)))
    consumer.wait(D(ready, lambda m, n, k, i: (1, -1, m, n)))
    return kernel, ready, producer, consumer

def test_semantic_accepts_valid_static_graph():
    """Accept a basic producer-consumer event graph and expose logical edges."""
    kernel, _, _, _, _ = _basic_kernel()

    plan = _validate(kernel)

    assert [(edge.producer.name, edge.consumer.name, edge.event.name) for edge in logical_edges(plan)] == [
        ("producer", "consumer", "ready")
    ]

def test_semantic_rejects_duplicate_tile_name():
    """Reject duplicate tile names in one KernelSpec."""
    kernel, _, _, producer, _ = _basic_kernel()
    kernel.tiles.append(producer)

    with pytest.raises(ValueError, match="duplicate tile names"):
        _validate(kernel)

def test_semantic_rejects_foreign_tensor():
    """Reject tile tensor accesses owned by another KernelSpec."""
    kernel, _, _, producer, _ = _basic_kernel()
    foreign = KernelSpec("foreign_tensor").tensor("foreign", (1,), "float32")
    producer.reads.append(foreign.region(lambda m, n, k: R[0]))

    with pytest.raises(ValueError, match="tensor outside kernel"):
        _validate(kernel)

def test_semantic_allows_bare_tensor_access_with_event_dependency():
    """Allow unknown/bare tensor regions when an event dependency connects writer and reader."""
    kernel = KernelSpec("semantic_bare_tensor")
    tensor = kernel.tensor("x", (1,), "float32")
    ready = kernel.event("ready", (1,), init_count=1)

    kernel.tile("writer", EmptyTile(), (1, 1, 1), writes=[tensor]).notify(D(ready, lambda m, n, k, i: (1, -1, 0,)))
    kernel.tile("reader", EmptyTile(), (1, 1, 1), reads=[tensor]).wait(D(ready, lambda m, n, k, i: (1, -1, 0,)))

    _validate(kernel)

def test_semantic_rejects_foreign_notify_event():
    """Reject notify dependencies on events owned by another KernelSpec."""
    kernel, _, _, producer, _ = _basic_kernel()
    foreign = KernelSpec("foreign_event").event("foreign", (1,), 1)
    producer.notifies[0] = DependencySpec(foreign, lambda m, n, k, i: (1, -1, 0,))

    with pytest.raises(ValueError, match="event outside kernel"):
        _validate(kernel)

def test_semantic_rejects_foreign_wait_event():
    """Reject wait dependencies on events owned by another KernelSpec."""
    kernel, _, _, _, consumer = _basic_kernel()
    foreign = KernelSpec("foreign_event").event("foreign", (1,), 1)
    consumer.waits[0] = DependencySpec(foreign, lambda m, n, k, i: (1, -1, 0,))

    with pytest.raises(ValueError, match="event outside kernel"):
        _validate(kernel)

@pytest.mark.parametrize(
    "coord,error,match",
    [
        pytest.param(lambda m, n, k, i: m, TypeError, "tuple/list", id="not-tuple"),
        pytest.param(lambda m, n, k, i: (1, -1, m, n, k), ValueError, "coord has", id="dim-mismatch"),
        pytest.param(lambda m, n, k, i: (1, -1, m, None), TypeError, "unsupported value", id="bad-value"),
    ],
)
def test_semantic_rejects_invalid_coord_shape(coord, error, match):
    """Reject dependency coord callbacks with wrong return type, dimensions, or value types."""
    kernel, _, ready, producer, _ = _basic_kernel()
    producer.notifies[0] = DependencySpec(ready, coord)

    with pytest.raises(error, match=match):
        _validate(kernel)

def test_semantic_rejects_wait_without_producer():
    """Reject waited events that have no producer notify."""
    kernel, _, _, producer, _ = _basic_kernel()
    producer.notifies.clear()

    with pytest.raises(ValueError, match="no producer"):
        _validate(kernel)

def test_semantic_rejects_notify_without_consumer():
    """Reject notified events that have no consumer wait."""
    kernel, _, _, _, consumer = _basic_kernel()
    consumer.waits.clear()

    with pytest.raises(ValueError, match="no consumer"):
        _validate(kernel)

def test_semantic_rejects_notify_count_mismatch():
    """Reject notify mappings whose static counts do not match event init_count."""
    kernel, _, ready, producer, _ = _basic_kernel()
    producer.notifies[0] = DependencySpec(ready, lambda m, n, k, i: (1, -1, m, 0))

    with pytest.raises(ValueError, match="init_count"):
        _validate(kernel)

def test_semantic_rejects_wait_out_of_bounds():
    """Reject static wait event coordinates outside the event shape."""
    kernel, _, ready, _, consumer = _basic_kernel()
    consumer.waits[0] = DependencySpec(ready, lambda m, n, k, i: (1, -1, m, 3))

    with pytest.raises(ValueError, match="out of bounds"):
        _validate(kernel)

def test_semantic_rejects_wait_without_matching_notify_coord():
    """Reject waits whose event coordinate has no matching producer notify."""
    kernel = KernelSpec("semantic_wait_coord")
    tensor = kernel.tensor("x", (4, 2), "float32")
    ready = kernel.event(
        "ready",
        (4, 2),
        init_count=lambda m, n: 1 if n == 0 else 0,
    )

    kernel.tile("producer", EmptyTile(), (4, 1, 1), reads=[_r2_first_col(tensor)]).notify(D(ready, lambda m, n, k, i: (1, -1, m, 0)))
    kernel.tile(
        "consumer",
        EmptyTile(),
        (4, 1, 1),
        reads=[tensor.region(lambda m, n, k: R[m, 1])],
    ).wait(D(ready, lambda m, n, k, i: (1, -1, m, 1)))

    with pytest.raises(ValueError, match="without a producer notify"):
        _validate(kernel)

def test_semantic_accepts_valid_symbolic_graph():
    """Accept a graph whose tensor, event, and tile spaces are symbolic."""
    kernel, _, _, _ = _symbolic_kernel()

    _validate(kernel)

def test_semantic_samples_symbolic_notify_count_mismatch():
    """Catch symbolic notify/init_count mismatches through deterministic samples."""
    kernel, ready, producer, _ = _symbolic_kernel()
    producer.notifies[0] = DependencySpec(ready, lambda m, n, k, i: (1, -1, m, 0))

    with pytest.raises(ValueError, match="init_count"):
        _validate(kernel)

def test_semantic_samples_symbolic_wait_swapped_coords():
    """Catch swapped symbolic wait coordinates through sampled validation."""
    kernel, ready, _, consumer = _symbolic_kernel()
    consumer.waits[0] = DependencySpec(ready, lambda m, n, k, i: (1, -1, n, m))

    with pytest.raises(ValueError, match="out of bounds|without a producer notify"):
        _validate(kernel)

def test_semantic_samples_symbolic_range_bounds():
    """Use bounded symbolic samples to catch missing notify coverage."""
    kernel = KernelSpec("semantic_symbolic_range")
    rows = kernel.var("rows", bounds=(20, 24))
    tensor = kernel.tensor("x", (rows,), "float32")
    ready = kernel.event("ready", (rows,), init_count=1)

    kernel.tile("producer", EmptyTile(), (16, 1, 1), reads=[_r1(tensor)]).notify(D(ready, lambda m, n, k, i: (1, -1, m,)))
    kernel.tile("consumer", EmptyTile(), (16, 1, 1), reads=[_r1(tensor)]).wait(D(ready, lambda m, n, k, i: (1, -1, m,)))

    with pytest.raises(ValueError, match="init_count"):
        _validate(kernel)

def test_semantic_accepts_varspec_expression_in_shape_event_and_grid():
    """Accept VarSpec expressions in tensor shapes, event shapes, and tile grids."""
    kernel = KernelSpec("semantic_expr_shape")
    rows = kernel.var("rows", bounds=(1, 9))
    blocks = rows.ceildiv(4)
    tensor = kernel.tensor("x", (rows + 1,), "float32")
    ready = kernel.event("ready", (blocks,), init_count=1)

    kernel.tile(
        "producer", EmptyTile(), (blocks, 1, 1), writes=[tensor.region(lambda m, n, k: R[m])]
    ).notify(D(ready, lambda m, n, k, i: (1, -1, m,)))
    kernel.tile(
        "consumer", EmptyTile(), (blocks, 1, 1), reads=[tensor.region(lambda m, n, k: R[m])]
    ).wait(D(ready, lambda m, n, k, i: (1, -1, m,)))

    plan = _validate(kernel)

    assert [var.name for var in plan.vars.values()] == ["rows"]

def test_semantic_rejects_foreign_varspec_inside_expression():
    """Reject ExprSpec values that reference vars from another KernelSpec."""
    kernel = KernelSpec("semantic_expr_foreign_var")
    rows = kernel.var("rows", bounds=(1, 4))
    foreign = KernelSpec("foreign").var("foreign", bounds=(1, 4))

    kernel.tensor("x", (rows + foreign,), "float32")

    with pytest.raises(ValueError, match="VarSpec outside this kernel"):
        _validate(kernel)

def test_semantic_accepts_region_dependency_covered_by_event():
    """Accept matching write/read regions connected by the same event coordinate."""
    kernel = KernelSpec("semantic_region_dep")
    tensor = kernel.tensor("x", (4,), "float32")
    ready = kernel.event("ready", (4,), init_count=1)

    kernel.tile(
        "producer",
        EmptyTile(),
        (4, 1, 1),
        writes=[tensor.region(lambda m, n, k: R[m])],
    ).notify(D(ready, lambda m, n, k, i: (1, -1, m,)))
    kernel.tile(
        "consumer",
        EmptyTile(),
        (4, 1, 1),
        reads=[tensor.region(lambda m, n, k: R[m])],
    ).wait(D(ready, lambda m, n, k, i: (1, -1, m,)))

    _validate(kernel)

def test_semantic_allows_region_read_from_external_input():
    """Allow region reads from input tensors with no producer tile."""
    kernel = KernelSpec("semantic_region_external_input")
    tensor = kernel.tensor("x", (4,), "float32")

    kernel.tile(
        "consumer", EmptyTile(), (4, 1, 1), reads=[tensor.region(lambda m, n, k: R[m])]
    )

    _validate(kernel)

def test_semantic_rejects_waited_coord_that_does_not_write_read_region():
    """Reject waited event coords whose producer does not cover the read region."""
    kernel = KernelSpec("semantic_waited_coord_wrong_region")
    tensor = kernel.tensor("x", (2,), "float32")
    ready = kernel.event("ready", (1,), init_count=1)

    kernel.tile(
        "producer",
        EmptyTile(),
        (1, 1, 1),
        writes=[tensor.region(lambda m, n, k: R[0])],
    ).notify(D(ready, lambda m, n, k, i: (1, -1, 0,)))
    kernel.tile(
        "consumer",
        EmptyTile(),
        (1, 1, 1),
        reads=[tensor.region(lambda m, n, k: R[1])],
    ).wait(D(ready, lambda m, n, k, i: (1, -1, 0,)))

    with pytest.raises(
        ValueError,
        match=r"waits on event 'ready' coord \(0,\).*reads tensor 'x' region \[1\]",
    ):
        _validate(kernel)

def test_semantic_accepts_shifted_region_dependency_when_wait_coord_matches_writer():
    """Accept shifted read regions when the wait coord points to the shifted producer."""
    kernel = KernelSpec("semantic_shifted_region_dep")
    tensor = kernel.tensor("x", (4,), "float32")
    ready = kernel.event("ready", (4,), init_count=1)

    kernel.tile(
        "producer",
        EmptyTile(),
        (4, 1, 1),
        writes=[tensor.region(lambda m, n, k: R[m])],
    ).notify(D(ready, lambda m, n, k, i: (1, -1, m,)))
    kernel.tile(
        "consumer",
        EmptyTile(),
        (3, 1, 1),
        reads=[tensor.region(lambda m, n, k: R[m + 1])],
    ).wait(D(ready, lambda m, n, k, i: (1, -1, m + 1,)))

    _validate(kernel)

def test_semantic_rejects_shifted_region_dependency_when_wait_coord_is_wrong():
    """Reject shifted read regions when the wait coord points to the wrong producer."""
    kernel = KernelSpec("semantic_shifted_region_wrong_wait")
    tensor = kernel.tensor("x", (4,), "float32")
    ready = kernel.event("ready", (4,), init_count=1)

    kernel.tile(
        "producer",
        EmptyTile(),
        (4, 1, 1),
        writes=[tensor.region(lambda m, n, k: R[m])],
    ).notify(D(ready, lambda m, n, k, i: (1, -1, m,)))
    kernel.tile(
        "consumer",
        EmptyTile(),
        (3, 1, 1),
        reads=[tensor.region(lambda m, n, k: R[m + 1])],
    ).wait(D(ready, lambda m, n, k, i: (1, -1, m,)))

    with pytest.raises(
        ValueError,
        match=r"reads tensor 'x' region \[1\].*producer' idx \(1, 0, 0\).*without an event dependency",
    ):
        _validate(kernel)

def test_semantic_rejects_region_read_without_event_dependency():
    """Reject producer-to-consumer region flow without a shared event dependency."""
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
    """Reject statically known tensor regions outside tensor bounds."""
    kernel = KernelSpec("semantic_region_oob")
    tensor = kernel.tensor("x", (4,), "float32")
    ready = kernel.event("ready", (4,), init_count=1)

    kernel.tile(
        "producer", EmptyTile(), (4, 1, 1), writes=[tensor.region(lambda m, n, k: R[m])]
    ).notify(D(ready, lambda m, n, k, i: (1, -1, m,)))
    kernel.tile(
        "consumer", EmptyTile(), (4, 1, 1), reads=[tensor.region(lambda m, n, k: R[4])]
    ).wait(D(ready, lambda m, n, k, i: (1, -1, m,)))

    with pytest.raises(ValueError, match="out of bounds"):
        _validate(kernel)

def test_semantic_accepts_range_region_dependency_covered_by_event():
    """Accept range reads covered by a wider producer write region and matching event."""
    kernel = KernelSpec("semantic_range_region_dep")
    tensor = kernel.tensor("x", (2, 16), "float32")
    ready = kernel.event("ready", (2,), init_count=1)

    kernel.tile(
        "producer",
        EmptyTile(),
        (2, 1, 1),
        writes=[tensor.region(lambda m, n, k: R[m, 0:16])],
    ).notify(D(ready, lambda m, n, k, i: (1, -1, m,)))
    kernel.tile(
        "consumer",
        EmptyTile(),
        (2, 1, 1),
        reads=[tensor.region(lambda m, n, k: R[m, 4:12])],
    ).wait(D(ready, lambda m, n, k, i: (1, -1, m,)))

    _validate(kernel)

def test_semantic_accepts_partially_overlapping_region_with_event_dependency():
    """Accept partial region overlap when an event connects producer and consumer."""
    kernel = KernelSpec("semantic_range_region_partial_overlap")
    tensor = kernel.tensor("x", (16,), "float32")
    ready = kernel.event("ready", (1,), init_count=1)

    kernel.tile(
        "producer", EmptyTile(), (1, 1, 1), writes=[tensor.region(lambda m, n, k: R[0:8])]
    ).notify(D(ready, lambda m, n, k, i: (1, -1, 0,)))
    kernel.tile(
        "consumer", EmptyTile(), (1, 1, 1), reads=[tensor.region(lambda m, n, k: R[4:12])]
    ).wait(D(ready, lambda m, n, k, i: (1, -1, 0,)))

    _validate(kernel)

def test_semantic_unknown_write_conservatively_covers_static_read():
    """Treat unknown producer writes as conservatively covering static reads."""
    kernel = KernelSpec("semantic_dynamic_write")
    tensor = kernel.tensor("x", (16,), "float32")
    ready = kernel.event("ready", (1,), init_count=1)

    kernel.tile(
        "producer", EmptyTile(), (1, 1, 1), writes=[tensor]
    ).notify(D(ready, lambda m, n, k, i: (1, -1, 0,)))
    kernel.tile(
        "consumer", EmptyTile(), (1, 1, 1), reads=[tensor.region(lambda m, n, k: R[4:8])]
    ).wait(D(ready, lambda m, n, k, i: (1, -1, 0,)))

    _validate(kernel)

def test_semantic_unknown_read_requires_event_dependency_from_writer():
    """Still require an event dependency when a consumer read region is unknown."""
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
    """Accept multiple producers contributing to one event coordinate."""
    kernel = KernelSpec("semantic_multi_producer")
    tensor = kernel.tensor("x", (4,), "float32")
    ready = kernel.event("ready", (4,), init_count=2)

    kernel.tile("producer_a", EmptyTile(), (4, 1, 1), reads=[_r1(tensor)]).notify(D(ready, lambda m, n, k, i: (1, -1, m,)))
    kernel.tile("producer_b", EmptyTile(), (4, 1, 1), reads=[_r1(tensor)]).notify(D(ready, lambda m, n, k, i: (1, -1, m,)))
    kernel.tile("consumer", EmptyTile(), (4, 1, 1), reads=[_r1(tensor)]).wait(D(ready, lambda m, n, k, i: (1, -1, m,)))

    plan = _validate(kernel)

    assert sorted((edge.producer.name, edge.consumer.name) for edge in logical_edges(plan)) == [
        ("producer_a", "consumer"),
        ("producer_b", "consumer"),
    ]

def test_semantic_accepts_multiple_consumers_waiting_on_one_event_coord():
    """Accept multiple consumers waiting on the same ready event coordinate."""
    kernel = KernelSpec("semantic_multi_consumer")
    tensor = kernel.tensor("x", (4,), "float32")
    ready = kernel.event("ready", (4,), init_count=1)

    kernel.tile("producer", EmptyTile(), (4, 1, 1), reads=[_r1(tensor)]).notify(D(ready, lambda m, n, k, i: (1, -1, m,)))
    kernel.tile("consumer_a", EmptyTile(), (4, 1, 1), reads=[_r1(tensor)]).wait(D(ready, lambda m, n, k, i: (1, -1, m,)))
    kernel.tile("consumer_b", EmptyTile(), (4, 1, 1), reads=[_r1(tensor)]).wait(D(ready, lambda m, n, k, i: (1, -1, m,)))

    plan = _validate(kernel)

    assert sorted((edge.producer.name, edge.consumer.name) for edge in logical_edges(plan)) == [
        ("producer", "consumer_a"),
        ("producer", "consumer_b"),
    ]

def test_semantic_rejects_duplicate_notify_event_on_one_tile():
    """Reject multiple notify edges from one tile to the same logical event."""
    kernel = KernelSpec("semantic_duplicate_notify_event")
    tensor = kernel.tensor("x", (4,), "float32")
    ready = kernel.event("ready", (4,), init_count=2)

    producer = kernel.tile("producer", EmptyTile(), (4, 1, 1), reads=[_r1(tensor)])
    producer.notify(D(ready, lambda m, n, k, i: (1, -1, m,)))
    producer.notify(D(ready, lambda m, n, k, i: (1, -1, m,)))
    kernel.tile("consumer", EmptyTile(), (4, 1, 1), reads=[_r1(tensor)]).wait(D(ready, lambda m, n, k, i: (1, -1, m,)))

    with pytest.raises(ValueError, match="notifies event .* more than once"):
        _validate(kernel)

def test_semantic_rejects_duplicate_wait_event_on_one_tile():
    """Reject multiple wait edges from one tile to the same logical event."""
    kernel = KernelSpec("semantic_duplicate_wait_event")
    tensor = kernel.tensor("x", (4,), "float32")
    ready = kernel.event("ready", (4,), init_count=1)

    kernel.tile("producer", EmptyTile(), (4, 1, 1), reads=[_r1(tensor)]).notify(D(ready, lambda m, n, k, i: (1, -1, m,)))
    consumer = kernel.tile("consumer", EmptyTile(), (4, 1, 1), reads=[_r1(tensor)])
    consumer.wait(D(ready, lambda m, n, k, i: (1, -1, m,)))
    consumer.wait(D(ready, lambda m, n, k, i: (1, -1, m,)))

    with pytest.raises(ValueError, match="waits on event .* more than once"):
        _validate(kernel)
