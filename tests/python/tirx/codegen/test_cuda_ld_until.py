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
"""Contract, codegen and device coverage for the scoped load-until primitive."""

import numpy as np
import pytest

import tvm
from tvm import ir, tirx
from tvm.backend.cuda.codegen.registry import CODEGEN_REGISTRY
from tvm.script import tirx as T
from tvm.testing import env


def make_kernel(dtype="uint64", order="relaxed", space="global", primitive=True):
    @T.prim_func
    def kernel(
        slot: T.Buffer((1,), dtype),
        out: T.Buffer((1,), dtype),
        initial: T.int64,
        generation: T.int32,
    ):
        T.device_entry()
        T.cta_id([1])
        lane = T.thread_id([32])
        observed = T.alloc_local((1,), dtype)
        shared = T.alloc_buffer((2,), dtype, scope="shared")
        if lane == 0:
            if space != "global":
                shared[0] = slot[0]
            observed[0] = T.Cast(dtype, initial)
            if primitive:
                T.cuda.ld_until(
                    observed[0],
                    slot.ptr_to([0]) if space == "global" else shared.ptr_to([0]),
                    predicate=lambda v: (
                        T.Cast("uint32", T.shift_right(v, 32)) == T.uint32(generation)
                        if dtype == "uint64"
                        else v == T.uint32(generation)
                    ),
                    order=order,
                    scope="cluster",
                    space=space,
                )
            else:
                while (
                    T.Cast("uint32", T.shift_right(observed[0], 32)) != T.uint32(generation)
                    if dtype == "uint64"
                    else observed[0] != T.uint32(generation)
                ):
                    T.ptx[f"ld.{order}.cluster.{space}.u{dtype[4:]}"](
                        observed[0], slot.ptr_to([0]) if space == "global" else shared.ptr_to([0])
                    )
            out[0] = observed[0]

    return kernel


def test_registration_and_roundtrip():
    op = ir.Op.get("tirx.cuda.ld_until")
    assert op.get_attr("TCallEffectKind") == tirx.CallEffectKind.Opaque.value
    func = make_kernel()
    calls = []
    tirx.stmt_functor.post_order_visit(
        func.body,
        lambda node: (
            calls.append(node) if isinstance(node, ir.Call) and node.op.same_as(op) else None
        ),
    )
    lowered, _tags = CODEGEN_REGISTRY[op.name](calls[0].args)
    assert lowered.op.same_as(ir.Op.get("tirx.cuda.func_call"))
    script = func.script()
    assert "T.cuda.ld_until" in script
    tvm.ir.assert_structural_equal(func, tvm.script.from_source(script))
    simplify = tirx.transform.StmtSimplify()
    assert "T.cuda.ld_until" in simplify(tvm.IRModule({"main": func})).script()


@pytest.mark.parametrize("dtype", ["uint32", "uint64"])
@pytest.mark.parametrize("order", ["relaxed", "acquire"])
@pytest.mark.parametrize("space", ["global", "shared::cta", "shared::cluster"])
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_device_preserves_initial_and_loaded_values(dtype, order, space):
    executable = tvm.compile(
        make_kernel(dtype, order, space),
        target=tvm.target.Target({"kind": "cuda", "arch": "sm_100a"}),
        tir_pipeline="tirx",
    )
    source = executable.mod.imports[0].inspect_source("cuda")
    assert f"ld.{order}.cluster.{space}.u{dtype[4:]}" in source

    def run():
        dev = tvm.cuda(0)
        word = (1 << 32) | 0xBEEF if dtype == "uint64" else 1
        seed = (1 << 32) | 0xCAFE if dtype == "uint64" else 1
        slot = tvm.runtime.tensor(np.array([word], dtype=dtype), device=dev)
        out = tvm.runtime.tensor(np.zeros(1, dtype=dtype), device=dev)
        for initial, expected in [(0, word), (seed, seed)]:
            executable(slot, out, initial, 1)
            np.testing.assert_array_equal(out.numpy(), np.array([expected], dtype=dtype))

    tvm.testing.run_with_gpu_lock(run)


def test_invalid_contracts():
    def lower(dst, ptr, **options):
        call = T.cuda.ld_until(dst, ptr, **options)
        return CODEGEN_REGISTRY[call.op.name](call.args)

    dst = tirx.decl_buffer((2,), "uint64", scope="local")
    slot = tirx.decl_buffer((2,), "uint64")
    options = dict(
        predicate=lambda v: v == T.uint64(1), order="relaxed", scope="cluster", space="global"
    )
    for field, value in [("order", "release"), ("scope", "warp"), ("space", "local")]:
        with pytest.raises(ValueError, match=field):
            lower(dst[0], slot.ptr_to([0]), **(options | {field: value}))
    with pytest.raises(TypeError, match="thread-local"):
        lower(slot[0], slot.ptr_to([0]), **options)
    with pytest.raises(TypeError, match="boolean"):
        lower(dst[0], slot.ptr_to([0]), **(options | {"predicate": lambda v: v}))
    with pytest.raises(ValueError, match="other than dst"):
        lower(dst[0], slot.ptr_to([0]), **(options | {"predicate": lambda v: v == dst[1]}))
    with pytest.raises(ValueError, match="effectful"):
        lower(
            dst[0], slot.ptr_to([0]), **(options | {"predicate": lambda v: v == T.cuda.clock64()})
        )
    with pytest.raises(ValueError, match="other than dst"):
        lower(dst[0], slot.ptr_to([dst[0]]), **options)
    # A conditional call would materialize a stale bool before the macro.
    with pytest.raises(ValueError, match="predicate"):
        lower(
            dst[0],
            slot.ptr_to([0]),
            **(
                options
                | {
                    "predicate": lambda v: T.if_then_else(
                        v == T.uint64(1), T.bool(True), T.bool(False)
                    )
                }
            ),
        )


if __name__ == "__main__":
    tvm.testing.main()
