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
"""TileImpl access collection and implementation validation tests."""

from __future__ import annotations

import pytest

import tvm.tirx.script as T
from tvm.tirx.script import tile as Tx

from tvm.megakernel.dsl import KernelSpec, R, TileImpl
from tvm.megakernel.transform.validate.impl import collect_impl_access, validate_impl


def _named_regions(accesses, idx=(0, 0, 0)):
    return [(_tensor_label(access), _region_tuple(access.tensor.region_from_tile(*idx))) for access in accesses]


def _tensor_label(access):
    return access.tensor.base_tensor.name


def _region_tuple(region):
    result = []
    for dim in region.dims:
        result.append((dim.start, dim.extent))
    return tuple(result)


class PrefetchRunTile(TileImpl):
    def __init__(self, src, weight, out):
        super().__init__()
        self.src = src
        self.weight = weight
        self.out = out
        self.smem = None

    def _declare_resources(self, smem_manager):
        self.smem = smem_manager.alloc((8,), "float32")

    @T.inline
    def device_init(self, smem_manager, m_idx, n_idx, k_idx):
        self._declare_resources(smem_manager)

    @T.inline
    def prefetch(self, m_idx, n_idx, k_idx):
        Tx.copy_async(self.smem[0:8], self.src[8:16])

    @T.inline
    def run(self, m_idx, n_idx, k_idx):
        Tx.add(self.smem[0:8], self.smem[0:8], self.weight[0:8])
        Tx.copy(self.out[0:8], self.smem[0:8])


class IndexedExprTile(TileImpl):
    def __init__(self, src, weight, tmp, out):
        super().__init__()
        self.src = src
        self.weight = weight
        self.tmp = tmp
        self.out = out

    @T.inline
    def prefetch(self, m_idx, n_idx, k_idx):
        Tx.copy_async(
            self.tmp[m_idx, n_idx, 0:8],
            self.src[m_idx, n_idx * 8 + k_idx * 2 : n_idx * 8 + k_idx * 2 + 8],
        )

    @T.inline
    def run(self, m_idx, n_idx, k_idx):
        Tx.add(
            self.tmp[m_idx, n_idx, 0:8],
            self.tmp[m_idx, n_idx, 0:8],
            self.weight[k_idx, 0:8],
        )
        Tx.copy(self.out[m_idx, n_idx, 0:8], self.tmp[m_idx, n_idx, 0:8])


class RawRunTile(TileImpl):
    def __init__(self, src, out):
        super().__init__()
        self.src = src
        self.out = out

    @T.inline
    def run(self, m_idx, n_idx, k_idx):
        self.out[0] = self.src[0] + 1.0


class GemmRunTile(TileImpl):
    def __init__(self, a, b, c, d):
        super().__init__()
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    @T.inline
    def run(self, m_idx, n_idx, k_idx):
        Tx.gemm(self.d[0:8, 0:8], self.a[0:8, 0:8], self.b[0:8, 0:8], self.c[0:8, 0:8])


class UnknownRunTile(TileImpl):
    def __init__(self, src):
        super().__init__()
        self.src = src

    @T.inline
    def run(self, m_idx, n_idx, k_idx):
        T.evaluate(T.call_extern("void", "unknown_side_effect", self.src.data))


def _prefetch_run_tensors():
    kernel = KernelSpec("prefetch_run")
    return {
        "src": kernel.tensor("src", (32,), "float32"),
        "weight": kernel.tensor("weight", (32,), "float32"),
        "out": kernel.tensor("out", (32,), "float32"),
    }


def _indexed_tensors():
    kernel = KernelSpec("indexed")
    rows = kernel.var("rows", bounds=(2, 5))
    groups = kernel.var("groups", bounds=(1, 3))
    cols = groups * 8 + 16
    return {
        "src": kernel.tensor("src", (rows, cols), "float32"),
        "weight": kernel.tensor("weight", (rows + 1, 16), "float32"),
        "tmp": kernel.tensor("tmp", (rows, groups, 8), "float32"),
        "out": kernel.tensor("out", (rows, groups, 8), "float32"),
    }


def _raw_tensors():
    kernel = KernelSpec("raw")
    return {
        "src": kernel.tensor("src", (16,), "float32"),
        "out": kernel.tensor("out", (16,), "float32"),
    }


def _gemm_tensors():
    kernel = KernelSpec("gemm")
    return {
        "a": kernel.tensor("a", (16, 16), "float32"),
        "b": kernel.tensor("b", (16, 16), "float32"),
        "c": kernel.tensor("c", (16, 16), "float32"),
        "d": kernel.tensor("d", (16, 16), "float32"),
    }


def test_impl_access_collects_tile_prefetch_reads_kernel_tensor_only():
    """Collect tensor reads emitted by the TileImpl prefetch hook."""
    access = collect_impl_access(PrefetchRunTile, tensors=_prefetch_run_tensors(), hooks="prefetch")

    assert _named_regions(access.reads) == [("src", ((8, 8),))]
    assert _named_regions(access.writes) == []
    assert not access.unknown_effects


def test_impl_access_collects_tile_run_reads_and_writes_kernel_tensors_only():
    """Collect tensor reads and writes emitted by the TileImpl run hook."""
    access = collect_impl_access(PrefetchRunTile, tensors=_prefetch_run_tensors(), hooks="run")

    assert _named_regions(access.reads) == [("weight", ((0, 8),))]
    assert _named_regions(access.writes) == [("out", ((0, 8),))]
    assert not access.unknown_effects


def test_impl_access_collects_tile_prefetch_and_run_with_one_interface():
    """Collect combined prefetch and run accesses through one API call."""
    access = collect_impl_access(PrefetchRunTile, tensors=_prefetch_run_tensors(), hooks=("prefetch", "run"))

    assert _named_regions(access.reads) == [("src", ((8, 8),)), ("weight", ((0, 8),))]
    assert _named_regions(access.writes) == [("out", ((0, 8),))]


def test_impl_access_collects_tile_prefetch_with_var_expr_shape_and_indices():
    """Collect prefetch regions whose shapes and indices use symbolic expressions."""
    access = collect_impl_access(IndexedExprTile, tensors=_indexed_tensors(), hooks="prefetch")

    assert [item[0] for item in _named_regions(access.reads)] == ["src"]
    assert [item[0] for item in _named_regions(access.writes)] == ["tmp"]
    assert _named_regions(access.reads, idx=(2, 3, 4)) == [("src", ((2, 1), (32, 8)))]
    assert _named_regions(access.writes, idx=(2, 3, 4)) == [("tmp", ((2, 1), (3, 1), (0, 8)))]


def test_impl_access_collects_tile_run_with_var_expr_shape_and_indices():
    """Collect run regions whose shapes and indices use symbolic expressions."""
    access = collect_impl_access(IndexedExprTile, tensors=_indexed_tensors(), hooks="run")

    assert [item[0] for item in _named_regions(access.reads)] == ["tmp", "weight", "tmp"]
    assert [item[0] for item in _named_regions(access.writes)] == ["tmp", "out"]
    assert _named_regions(access.reads, idx=(2, 3, 4)) == [
        ("tmp", ((2, 1), (3, 1), (0, 8))),
        ("weight", ((4, 1), (0, 8))),
        ("tmp", ((2, 1), (3, 1), (0, 8))),
    ]
    assert _named_regions(access.writes, idx=(2, 3, 4)) == [
        ("tmp", ((2, 1), (3, 1), (0, 8))),
        ("out", ((2, 1), (3, 1), (0, 8))),
    ]


def test_impl_access_collects_tile_raw_buffer_load_store_run():
    """Collect direct Buffer load/store accesses from TileImpl.run."""
    access = collect_impl_access(RawRunTile, tensors=_raw_tensors(), hooks="run")

    assert _named_regions(access.reads) == [("src", ((0, 1),))]
    assert _named_regions(access.writes) == [("out", ((0, 1),))]


def test_impl_access_collects_tile_gemm_run_regions():
    """Collect tile primitive GEMM read/write regions."""
    access = collect_impl_access(GemmRunTile, tensors=_gemm_tensors(), hooks="run")

    assert _named_regions(access.reads) == [
        ("a", ((0, 8), (0, 8))),
        ("b", ((0, 8), (0, 8))),
        ("c", ((0, 8), (0, 8))),
    ]
    assert _named_regions(access.writes) == [("d", ((0, 8), (0, 8)))]


def test_impl_access_records_unknown_effect_in_tile_run():
    """Record opaque calls as unknown access effects."""
    kernel = KernelSpec("unknown")
    tensors = {"src": kernel.tensor("src", (16,), "float32")}
    access = collect_impl_access(UnknownRunTile, tensors=tensors, hooks="run")

    assert access.unknown_effects
    assert "opaque call" in access.unknown_effects[0]


class ImplCopyTile(TileImpl):
    def __init__(self, src, out):
        super().__init__()
        self.src = src
        self.out = out

    @T.inline
    def run(self, m_idx, n_idx, k_idx):
        Tx.copy(self.out[0:8], self.src[0:8])


def test_impl_validate_accepts_matching_declared_regions():
    """Accept TileImpl accesses covered by declared tile reads/writes."""
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

    validate_impl(kernel)


def test_impl_validate_rejects_read_missing_from_tile_reads():
    """Reject actual reads not declared in tile.reads."""
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
        validate_impl(kernel)


def test_impl_validate_rejects_write_missing_from_tile_writes():
    """Reject actual writes not declared in tile.writes."""
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
        validate_impl(kernel)


def test_impl_validate_rejects_region_outside_declared_region():
    """Reject actual access regions outside declared regions."""
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
        validate_impl(kernel)


class ImplUnknownRegionTile(TileImpl):
    def __init__(self, src, out):
        super().__init__()
        self.src = src
        self.out = out

    @T.inline
    def run(self, m_idx, n_idx, k_idx):
        base = m_idx * 8
        Tx.copy(self.out[base : base + 8], self.src[base : base + 8])


def test_impl_warns_on_unknown_access_by_default():
    """Warn on unknown actual access effects under the default policy."""
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
        validate_impl(kernel)


class ImplThreadIndexedTile(TileImpl):
    def __init__(self, src, out):
        super().__init__()
        self.src = src
        self.out = out

    @T.inline
    def run(self, m_idx, n_idx, k_idx):
        tid = T.thread_id([32])
        Tx.copy(self.out[0:8], self.src[tid : tid + 8])


def test_impl_rejects_unknown_actual_region_with_specific_declaration():
    """Reject unknown actual regions when declarations are specific."""
    kernel = KernelSpec("semantic_impl_unknown_region_error")
    src = kernel.tensor("src", (16,), "float32")
    out = kernel.tensor("out", (16,), "float32")

    kernel.tile(
        "copy",
        ImplThreadIndexedTile(src, out),
        (1, 1, 1),
        reads=[src.region(lambda m, n, k: R[0:8])],
        writes=[out.region(lambda m, n, k: R[0:8])],
    )

    with pytest.warns(UserWarning, match="unknown access effect"):
        with pytest.raises(ValueError, match="unknown region"):
            validate_impl(kernel)


def test_impl_warn_policy_downgrades_region_mismatch():
    """Downgrade region mismatch errors to warnings when policy is warn."""
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
        validate_impl(kernel)


def test_impl_off_policy_skips_access_validation():
    """Skip implementation access validation when policy is off."""
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

    validate_impl(kernel)


def test_impl_rejects_invalid_access_validate_policy():
    """Reject unsupported impl_access_validate policy values."""
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
        validate_impl(kernel)
