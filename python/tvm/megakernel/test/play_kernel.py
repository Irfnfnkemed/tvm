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
"""Runnable playground for the current megakernel DSL.

This file ports the original two-stage static-fusion demo into the current DSL:

1. stage1_partial_reduce:
   A[m-block, n-block] -> B[m-block, n]
2. stage2_final_reduce:
   B[m-block, :] -> C[m-block, 0]

It exercises the current user-facing DSL shape: symbolic vars, tensor regions,
logical events, wait/notify dependencies, dynamic ``inv_coord`` mappings,
TileImpl constructor captures, managed shared memory, static/dynamic lowering,
and optional IR/CUDA source dumps.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import tvm
import tvm.tirx.script as T
from tvm.tirx.script import tile as Tx

from tvm.megakernel.dsl import D, KernelSpec, R, TileImpl
from tvm.megakernel.transform import LoweringOptions, lower


M = 1024
N = 1024
BLOCK_M = 64
BLOCK_N = 64
NUM_BLOCK_M = M // BLOCK_M
NUM_BLOCK_N = N // BLOCK_N
SM_COUNT = 17
MAX_TASKS = 512
ROUTING_MAX_TASKS = 2048
NUM_THREADS = 256
ROUTING_ITEMS = 1024
ROUTING_GROUPS = 16
ROUTING_ITEMS_PER_GROUP = ROUTING_ITEMS // ROUTING_GROUPS


class Stage1PartialReduceTile(TileImpl):
    """Reduce one A[m-block, n-block] tile into B[m-block, n]."""

    def __init__(self, source, partial, *, block_m, block_n):
        super().__init__()
        self.source = source
        self.partial = partial
        self.block_m = block_m
        self.block_n = block_n
        self.smem_manager = None
        self.source_smem = None
        self.partial_smem = None

    def _declare_resources(self, smem_manager):
        self.smem_manager = smem_manager
        self.source_smem = smem_manager.alloc(
            (self.block_m, self.block_n), "float32", policy="shared"
        )
        self.partial_smem = smem_manager.alloc((self.block_m, 1), "float32", policy="shared")

    @T.inline
    def device_init(self, smem_manager, m_idx, n_idx, k_idx):
        self._declare_resources(smem_manager)

    @T.inline
    def run(self, m_idx, n_idx, k_idx):
        self.smem_manager.wait_all("cta")
        Tx.copy(
            self.source_smem,
            self.source[
                m_idx * self.block_m : (m_idx + 1) * self.block_m,
                n_idx * self.block_n : (n_idx + 1) * self.block_n,
            ],
        )
        Tx.sum(self.partial_smem, self.source_smem)
        Tx.copy(
            self.partial[m_idx * self.block_m : (m_idx + 1) * self.block_m, n_idx],
            self.partial_smem,
        )
        self.smem_manager.release_all("cta")
        self.smem_manager.advance()


class Stage2FinalReduceTile(TileImpl):
    """Reduce all B n-block partials for one m-block into C[m-block, 0]."""

    def __init__(self, partial, output, *, block_m, num_block_n):
        super().__init__()
        self.partial = partial
        self.output = output
        self.block_m = block_m
        self.num_block_n = num_block_n
        self.smem_manager = None
        self.partial_smem = None
        self.output_smem = None

    def _declare_resources(self, smem_manager):
        self.smem_manager = smem_manager
        self.partial_smem = smem_manager.alloc(
            (self.block_m, self.num_block_n), "float32", policy="shared"
        )
        self.output_smem = smem_manager.alloc((self.block_m, 1), "float32", policy="shared")

    @T.inline
    def device_init(self, smem_manager, m_idx, n_idx, k_idx):
        self._declare_resources(smem_manager)

    @T.inline
    def run(self, m_idx, n_idx, k_idx):
        self.smem_manager.wait_all("cta")
        Tx.copy(
            self.partial_smem,
            self.partial[m_idx * self.block_m : (m_idx + 1) * self.block_m, 0:self.num_block_n],
        )
        Tx.sum(self.output_smem, self.partial_smem)
        Tx.copy(
            self.output[m_idx * self.block_m : (m_idx + 1) * self.block_m, 0],
            self.output_smem,
        )
        self.smem_manager.release_all("cta")
        self.smem_manager.advance()


class EndpointTile(TileImpl):
    """Dynamic scheduler endpoint task."""

    @T.inline
    def run(self, m_idx, n_idx, k_idx):
        T.evaluate(0)


class RoutingProducerTile(TileImpl):
    """Produce one partial value whose ready event is selected by routing[item]."""

    def __init__(self, source, partial):
        super().__init__()
        self.source = source
        self.partial = partial

    @T.inline
    def run(self, m_idx, n_idx, k_idx):
        tid = T.thread_id([NUM_THREADS])
        if tid == 0:
            self.partial[m_idx] = self.source[m_idx] * T.float32(2)


class RoutingConsumerTile(TileImpl):
    """Consume all partial values routed to one group."""

    def __init__(self, routing, partial, output, *, items):
        super().__init__()
        self.routing = routing
        self.partial = partial
        self.output = output
        self.items = items

    @T.inline
    def run(self, m_idx, n_idx, k_idx):
        tid = T.thread_id([NUM_THREADS])
        if tid == 0:
            acc = T.float32(0)
            for item in T.serial(0, self.items):
                if self.routing[item] == m_idx:
                    acc = acc + self.partial[item]
            self.output[m_idx] = acc


def _dynamic_only(dynamic, inv_coord):
    return inv_coord if dynamic else None


def build_kernel(name: str, *, dynamic: bool) -> KernelSpec:
    """Build the two-stage reduce playground kernel.

    Torch reference:

    ``C[:, 0] = A.sum(dim=1)``
    """

    kernel = KernelSpec(name)
    rows = kernel.var("rows", bounds=(M, M))

    source = kernel.tensor("A", (rows, N), "float32")
    partial = kernel.tensor("B", (rows, NUM_BLOCK_N), "float32")
    output = kernel.tensor("C", (rows, 1), "float32")

    row_ready = kernel.event("row_ready", (NUM_BLOCK_M,), init_count=NUM_BLOCK_N)
    done = kernel.event("done", (1,), init_count=NUM_BLOCK_M)

    kernel.tile(
        "stage1_partial_reduce",
        Stage1PartialReduceTile(source, partial, block_m=BLOCK_M, block_n=BLOCK_N),
        grid=(NUM_BLOCK_M, NUM_BLOCK_N, 1),
        reads=[
            source.region(
                lambda m, n, k: R[
                    m * BLOCK_M : (m + 1) * BLOCK_M,
                    n * BLOCK_N : (n + 1) * BLOCK_N,
                ]
            )
        ],
        writes=[
            partial.region(lambda m, n, k: R[m * BLOCK_M : (m + 1) * BLOCK_M, n])
        ],
    ).notify(D(row_ready, lambda m, n, k, i: (1, -1, m)))

    kernel.tile(
        "stage2_final_reduce",
        Stage2FinalReduceTile(partial, output, block_m=BLOCK_M, num_block_n=NUM_BLOCK_N),
        grid=(NUM_BLOCK_M, 1, 1),
        reads=[
            partial.region(
                lambda m, n, k: R[m * BLOCK_M : (m + 1) * BLOCK_M, 0:NUM_BLOCK_N]
            )
        ],
        writes=[output.region(lambda m, n, k: R[m * BLOCK_M : (m + 1) * BLOCK_M, 0])],
    ).wait(
        D(
            row_ready,
            lambda m, n, k, i: (1, -1, m),
            inv_coord=_dynamic_only(dynamic, lambda rank, m, i: (1, m, 0, 0)),
        )
    ).notify(D(done, lambda m, n, k, i: (1, -1, 0)))

    kernel.tile("endpoint", EndpointTile(), grid=(1, 1, 1)).wait(
        D(
            done,
            lambda m, n, k, i: (1, -1, 0),
            inv_coord=_dynamic_only(dynamic, lambda rank, e, i: (1, 0, 0, 0)),
        )
    )
    return kernel


def build_routing_kernel(name: str, *, dynamic: bool) -> KernelSpec:
    """Build a dynamic runtime-routing dependency example.

    Torch reference, for the fixed routing tensor ``routing[item] = item % groups``:

    ``partial[item] = A[item] * 2``
    ``output[group] = sum(partial[item] for item if routing[item] == group)``

    The important DSL part is the producer notify dependency:

    ``D(ready, lambda m, n, k, i: (1, -1, routing[m]))``

    Here ``routing`` is a KernelSpec tensor captured by closure. During lowering
    the dependency coord closure is rebound so ``routing[m]`` becomes a TIRX
    buffer load from the lowered routing buffer.
    """

    if not dynamic:
        raise ValueError("routing dependency playground is intended for dynamic scheduling")

    kernel = KernelSpec(name)
    items = kernel.var("items", bounds=(ROUTING_ITEMS, ROUTING_ITEMS))
    groups = kernel.var("groups", bounds=(ROUTING_GROUPS, ROUTING_GROUPS))

    source = kernel.tensor("A", (items,), "float32")
    routing = kernel.tensor("routing", (items,), "int32")
    partial = kernel.tensor("partial", (items,), "float32")
    output = kernel.tensor("output", (groups,), "float32")

    # This matches the demo routing tensor routing[item] = item % groups. In
    # general, runtime routing requires init_count(event_coord) to match the
    # routing histogram.
    ready = kernel.event("ready", (groups,), init_count=ROUTING_ITEMS_PER_GROUP)
    done = kernel.event("done", (1,), init_count=ROUTING_GROUPS)

    kernel.tile(
        "producer",
        RoutingProducerTile(source, partial),
        grid=(items, 1, 1),
        reads=[source.region(lambda m, n, k: R[m]), routing.region(lambda m, n, k: R[m])],
        writes=[partial.region(lambda m, n, k: R[m])],
    ).notify(D(ready, lambda m, n, k, i: (1, -1, routing[m])))

    kernel.tile(
        "consumer",
        RoutingConsumerTile(routing, partial, output, items=items),
        grid=(groups, 1, 1),
        reads=[routing, partial],
        writes=[output.region(lambda m, n, k: R[m])],
    ).wait(
        D(
            ready,
            lambda m, n, k, i: (1, -1, m),
            inv_coord=lambda rank, group, i: (1, group, 0, 0),
        )
    ).notify(D(done, lambda m, n, k, i: (1, -1, 0)))

    kernel.tile("endpoint", EndpointTile(), grid=(1, 1, 1)).wait(
        D(done, lambda m, n, k, i: (1, -1, 0), inv_coord=lambda rank, e, i: (1, 0, 0, 0))
    )
    return kernel


def lowering_options(schedule: str, case: str = "two-stage") -> LoweringOptions:
    max_tasks = ROUTING_MAX_TASKS if case == "routing" else MAX_TASKS
    return LoweringOptions(
        schedule=schedule,
        smem_max_bytes=64 * 1024,
        smem_chunk_size=16 * 1024,
        attrs={
            "sm_count": SM_COUNT,
            "num_threads": NUM_THREADS,
            "max_tasks": max_tasks,
            "end_job_id": 31,
        },
    )


def lower_kernel(schedule: str, case: str):
    if case == "two-stage":
        name = f"play_{schedule}_two_stage_reduce"
        kernel = build_kernel(name, dynamic=schedule == "dynamic")
    elif case == "routing":
        name = f"play_{schedule}_runtime_routing"
        kernel = build_routing_kernel(name, dynamic=schedule == "dynamic")
    else:
        raise ValueError(f"unknown playground case: {case}")
    mod = lower(kernel, lowering_options(schedule, case))
    return name, kernel, mod


def _torch_reference(source):
    return source.sum(dim=1, keepdim=True)


def _torch_routing_reference(source, routing):
    import torch

    partial = source * 2
    output = torch.zeros((ROUTING_GROUPS,), dtype=source.dtype, device=source.device)
    for item in range(ROUTING_ITEMS):
        output[routing[item]] += partial[item]
    return output


def _compile(mod):
    with tvm.target.Target("cuda"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return tvm.compile(mod, target="cuda", tir_pipeline="tirx")


def _cuda_source(lib) -> str:
    """Return generated CUDA source from a compiled TIRX module."""

    imports = getattr(lib.mod, "imports", None)
    if not imports:
        raise RuntimeError("compiled module does not contain an imported CUDA module")
    return imports[0].inspect_source()


def _write_text(path, text: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text)
    print(f"wrote {output_path}")


def _event_workspace(torch):
    # Event workspace: row_ready + done + event-init-complete.
    return torch.empty((NUM_BLOCK_M + 1 + 1,), dtype=torch.int32, device="cuda")


def _routing_event_workspace(torch):
    # Event workspace: ready + done + event-init-complete.
    return torch.empty((ROUTING_GROUPS + 1 + 1,), dtype=torch.int32, device="cuda")


def _make_inputs(torch):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(2)
    source = torch.randn((M, N), dtype=torch.float32, device="cuda", generator=generator)
    partial = torch.empty((M, NUM_BLOCK_N), dtype=torch.float32, device="cuda")
    output = torch.empty((M, 1), dtype=torch.float32, device="cuda")
    for tensor in (partial, output):
        tensor.fill_(float("nan"))
    return source, partial, output


def _make_routing_inputs(torch):
    source = torch.arange(1, ROUTING_ITEMS + 1, dtype=torch.float32, device="cuda")
    routing = torch.arange(ROUTING_ITEMS, dtype=torch.int32, device="cuda") % ROUTING_GROUPS
    partial = torch.empty((ROUTING_ITEMS,), dtype=torch.float32, device="cuda")
    output = torch.empty((ROUTING_GROUPS,), dtype=torch.float32, device="cuda")
    partial.fill_(float("nan"))
    output.fill_(float("nan"))
    return source, routing, partial, output


def run_cuda(schedule: str, case: str, name: str, mod, lib=None) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for --run")

    if lib is None:
        lib = _compile(mod)
    if case == "two-stage":
        tensors = _make_inputs(torch)
        event_workspace = _event_workspace(torch)
        init_args = (M,)
    elif case == "routing":
        tensors = _make_routing_inputs(torch)
        event_workspace = _routing_event_workspace(torch)
        init_args = (ROUTING_ITEMS, ROUTING_GROUPS)
    else:
        raise ValueError(f"unknown playground case: {case}")
    event_workspace.fill_(-12345)

    if schedule == "static":
        queue = torch.empty((SM_COUNT, MAX_TASKS), dtype=torch.int32, device="cuda")
        queue.fill_(0x7FFFFFFF)
        lib[f"{name}_init_queue"](*init_args, queue)
        lib[name](*tensors, event_workspace, queue)
    else:
        task_capacity = ROUTING_MAX_TASKS if case == "routing" else MAX_TASKS
        tasks = torch.empty((task_capacity,), dtype=torch.int32, device="cuda")
        head = torch.empty((1,), dtype=torch.int32, device="cuda")
        tail = torch.empty((1,), dtype=torch.int32, device="cuda")
        tasks.fill_(-1)
        head.fill_(-1)
        tail.fill_(-1)
        lib[f"{name}_init_queue"](*init_args, event_workspace, tasks, head, tail)
        lib[name](*tensors, event_workspace, tasks, head, tail)

    torch.cuda.synchronize()
    if case == "two-stage":
        source, _partial, output = tensors
        reference = _torch_reference(source)
    else:
        source, routing, _partial, output = tensors
        reference = _torch_routing_reference(source, routing)
    torch.testing.assert_close(output.cpu(), reference.cpu(), rtol=1e-3, atol=1e-3)
    print(f"{schedule} {case} CUDA result matches Torch reference")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build/lower/run the megakernel DSL playground")
    parser.add_argument("--case", choices=("two-stage", "routing"), default="two-stage")
    parser.add_argument("--schedule", choices=("static", "dynamic"), default="static")
    parser.add_argument("--print-ir", action="store_true", help="print the lowered IRModule")
    parser.add_argument("--ir-out", help="write the lowered IRModule text to this file")
    parser.add_argument("--cuda-out", help="compile and write generated CUDA source to this file")
    parser.add_argument("--run", action="store_true", help="compile CUDA and compare against Torch")
    args = parser.parse_args()

    name, kernel, mod = lower_kernel(args.schedule, args.case)
    print(f"kernel: {kernel.name}")
    print(f"vars: {list(kernel.vars)}")
    print(f"tensors: {list(kernel.tensors)}")
    print(f"events: {list(kernel.events)}")
    print(f"tiles: {[tile.name for tile in kernel.tiles]}")
    print(f"lowered functions: {list(mod.functions)}")

    if args.print_ir:
        print(mod)
    if args.ir_out:
        _write_text(args.ir_out, str(mod))

    compiled_lib = None
    if args.cuda_out:
        compiled_lib = _compile(mod)
        _write_text(args.cuda_out, _cuda_source(compiled_lib))
    if args.run:
        run_cuda(args.schedule, args.case, name, mod, compiled_lib)


if __name__ == "__main__":
    main()
