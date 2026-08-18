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
"""DSL spec builder and public API contract tests."""

from __future__ import annotations

import pytest

from tvm.megakernel.dsl import D, KernelSpec, TileImpl


class EmptyTile(TileImpl):
    def run(self, m_idx, n_idx, k_idx):
        pass



def test_kernel_validate_returns_kernel_spec():
    """Check KernelSpec.validate returns the validated kernel."""
    kernel = KernelSpec("validate_entry")
    x = kernel.tensor("x", (1,), "float32")
    kernel.tile("only", EmptyTile(), (1, 1, 1), reads=[x])

    validated = kernel.validate()

    assert validated is kernel
    assert [tile.name for tile in validated.tiles] == ["only"]

def test_kernel_var_rejects_invalid_bounds():
    """Reject invalid symbolic variable bounds at DSL construction time."""
    kernel = KernelSpec("invalid_var_range")

    with pytest.raises(ValueError, match="bounds"):
        kernel.var("rows", bounds=(8, 1))

def test_kernel_normalizes_tensor_and_event_shape():
    """Normalize scalar tensor/event shapes to one-dimensional tuples."""
    kernel = KernelSpec("normalize_shape")
    rows = kernel.var("rows", bounds=(1, 4))

    tensor = kernel.tensor("x", rows, "float32")
    event = kernel.event("ready", rows, init_count=1)

    assert tensor.shape == (rows,)
    assert event.shape == (rows,)

def test_kernel_rejects_invalid_shape_dim():
    """Reject shape dimensions that are not int, VarSpec, or ExprSpec."""
    kernel = KernelSpec("invalid_shape")

    with pytest.raises(TypeError, match="shape dims"):
        kernel.tensor("x", (True,), "float32")
    with pytest.raises(TypeError, match="shape dims"):
        kernel.event("ready", (object(),), init_count=1)

def test_kernel_normalizes_event_init_count_to_callable():
    """Normalize uniform and per-coordinate event init_count to callables."""
    kernel = KernelSpec("event_init_count_callable")

    events = [kernel.event(f"ready_{count}", (2,), init_count=count) for count in (0, 3)]
    rows = kernel.var("rows", bounds=(1, 8))
    dynamic = kernel.event("dynamic", (2, 3), init_count=lambda m, n: m + n)
    symbolic = kernel.event("symbolic", (rows,), init_count=rows)

    assert events[0].init_count(0) == 0
    assert events[1].init_count(0) == 3
    assert dynamic.init_count(1, 2) == 3
    assert symbolic.init_count(0) is rows

def test_kernel_rejects_invalid_event_init_count():
    """Reject bool, negative, and non-callable/non-integer event init_count values."""
    kernel = KernelSpec("invalid_event_init_count")

    for value in (True, -1):
        with pytest.raises((TypeError, ValueError), match="init_count"):
            kernel.event(f"ready_{value}", (1,), init_count=value)
    with pytest.raises(TypeError, match="init_count"):
        kernel.event("ready_bad", (1,), init_count="x")

def test_kernel_normalizes_tile_grid():
    """Normalize tile grid lists to canonical three-dimensional tuples."""
    kernel = KernelSpec("normalize_grid")
    tensor = kernel.tensor("x", (1,), "float32")

    tile = kernel.tile("only", EmptyTile(), [1, 1, 1], reads=[tensor])

    assert tile.grid == (1, 1, 1)

def test_kernel_rejects_invalid_tile_grid_and_impl():
    """Reject non-TileImpl tile bodies and invalid tile grid shapes."""
    kernel = KernelSpec("invalid_grid")
    tensor = kernel.tensor("x", (1,), "float32")

    with pytest.raises(TypeError, match="TileImpl"):
        kernel.tile("bad_impl", object(), (1, 1, 1), reads=[tensor])
    with pytest.raises(ValueError, match="three dimensions"):
        kernel.tile("bad_dim", EmptyTile(), (1, 1), reads=[tensor])
    with pytest.raises(TypeError, match="grid dims"):
        kernel.tile("bad_dim", EmptyTile(), (1, True, 1), reads=[tensor])

def test_tile_wait_notify_accepts_dependency_builder():
    """Accept D-built dependencies and preserve coord/inv_coord callbacks."""
    kernel = KernelSpec("dependency_builder")
    event = kernel.event("ready", (1,), init_count=1)
    tensor = kernel.tensor("x", (1,), "float32")

    dep = D(
        event,
        lambda m, n, k, i: (1, -1, 0),
        inv_coord=lambda rank, x, i: (1, 0, 0, 0),
    )
    tile = kernel.tile("only", EmptyTile(), (1, 1, 1), reads=[tensor])
    tile.wait(dep).notify(D(event, lambda m, n, k, i: (1, -1, 0)))

    assert tile.waits[0] is dep
    assert tile.waits[0].inv_coord(-1, 0, 0) == (1, 0, 0, 0)
    assert tile.notifies[0].coord(0, 0, 0, 0) == (1, -1, 0)

def test_tile_wait_notify_rejects_invalid_coord():
    """Reject waits/notifies that are not built by D or use invalid dependency callbacks."""
    kernel = KernelSpec("invalid_coord")
    event = kernel.event("ready", (1,), init_count=1)
    tile = kernel.tile("only", EmptyTile(), (1, 1, 1))

    with pytest.raises(TypeError, match="DependencySpec"):
        tile.wait(event)
    with pytest.raises(TypeError, match="DependencySpec"):
        tile.notify(event)
    with pytest.raises(TypeError, match="coord"):
        tile.wait(D(event, 0))
    with pytest.raises(TypeError, match="inv_coord"):
        tile.wait(D(event, lambda m, n, k, i: (1, -1, 0), inv_coord=0))
    with pytest.raises(TypeError, match="coord"):
        tile.notify(D(event, 0))
