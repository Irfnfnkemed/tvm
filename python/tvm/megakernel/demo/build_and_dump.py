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

"""Dump and compile the megakernel DSL demo."""

from __future__ import annotations

from pathlib import Path
import argparse
import sys
import traceback

import torch
import tvm

from tvm.megakernel.demo.dsl import kernel
from tvm.megakernel.transform import LoweringOptions, lower_to_tirx_module


DEMO_DIR = Path(__file__).resolve().parent
LOWERED_PATH = DEMO_DIR / "tirx_lowered.py"
CUDA_PATH = DEMO_DIR / "tirx_lowered.cu"

M = 1024
N = 1024
NUM_BLOCK_N = 16
SM_COUNT = 148
MAX_TASKS = 128


def build_options() -> LoweringOptions:
    return LoweringOptions(
        smem_max_bytes=64 * 1024,
        smem_chunk_size=16 * 1024,
        schedule="static",
        attrs={
            "sm_count": SM_COUNT,
            "num_threads": 256,
            "max_tasks": MAX_TASKS,
            "end_job_id": 31,
        },
    )


def dump() -> tvm.IRModule:
    mod = lower_to_tirx_module(kernel, build_options())
    LOWERED_PATH.write_text(mod.script())
    print(f"wrote {LOWERED_PATH}")
    return mod


def compile_module(mod: tvm.IRModule):
    target = tvm.target.Target("cuda")
    with target:
        return tvm.compile(mod, target=target, tir_pipeline="tirx")


def dump_cuda(lib) -> None:
    src = lib.mod.imports[0].inspect_source()
    CUDA_PATH.write_text(src)
    print(f"wrote {CUDA_PATH}")


def run_module(lib, trials: int = 1, base_seed: int = 0) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available for runtime smoke test")

    generator = torch.Generator(device="cuda")
    max_abs_errors = []
    for trial in range(trials):
        generator.manual_seed(base_seed + trial)
        A = torch.randn((M, N), dtype=torch.float32, device="cuda", generator=generator)
        B = torch.empty((M, NUM_BLOCK_N), dtype=torch.float32, device="cuda")
        C = torch.empty((M, 1), dtype=torch.float32, device="cuda")
        event_workspace = torch.empty((NUM_BLOCK_N + 1,), dtype=torch.int32, device="cuda")
        queue = torch.empty((SM_COUNT, MAX_TASKS), dtype=torch.int32, device="cuda")

        B.fill_(float("nan"))
        C.fill_(float("nan"))
        event_workspace.fill_(-12345)
        queue.fill_(0x7FFFFFFF)

        lib["two_stage_reduce_init_queue"](queue)
        lib["two_stage_reduce"](A, B, C, event_workspace, queue)
        torch.cuda.synchronize()

        C_ref = torch.sum(A, dim=1, keepdim=True)
        max_abs_errors.append(torch.max(torch.abs(C - C_ref)).item())
        torch.testing.assert_close(C.cpu(), C_ref.cpu(), rtol=1e-4, atol=1e-4)

    print(
        f"runtime smoke: ok, trials={trials}, "
        f"max_abs_error={max(max_abs_errors):.6g}"
    )


def _compile_single(name, func) -> None:
    try:
        compile_module(tvm.IRModule({name: func}))
        print(f"compile {name}: ok")
    except Exception as err:  # pylint: disable=broad-except
        print(f"compile {name}: failed: {type(err).__name__}: {str(err).splitlines()[-1]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump and compile the megakernel DSL demo.")
    parser.add_argument(
        "--dump-cuda",
        action="store_true",
        help=f"also dump generated CUDA source to {CUDA_PATH}",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="run a CUDA runtime smoke test after compilation",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="number of randomized runtime smoke-test trials",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="base random seed for runtime smoke-test inputs",
    )
    args = parser.parse_args()

    mod = dump()
    try:
        lib = compile_module(mod)
    except Exception:  # pylint: disable=broad-except
        print("compile module: failed")
        traceback.print_exc()
        print("\nper-function compile diagnostics:")
        for gv, func in mod.functions.items():
            _compile_single(gv.name_hint, func)
        sys.exit(1)
    print("compile module: ok")

    if args.dump_cuda:
        dump_cuda(lib)
    if args.run:
        run_module(lib, trials=args.trials, base_seed=args.seed)


if __name__ == "__main__":
    main()
