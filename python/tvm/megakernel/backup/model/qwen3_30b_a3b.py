import json
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Literal
import random


import numpy as np
import torch
from tqdm import tqdm

import tvm
from tvm import dlight, relax, target
from tvm.relax import register_pipeline
from tvm.relax.frontend import nn
from tvm.runtime import ShapeTuple
from tvm.runtime import disco as di
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T

from tvm.tirx.megakernel.utils.base import MegaKernelWrapper
from tvm.tirx.megakernel.utils.dynamic_scheduler import DynamicTileScheduler
from tvm.tirx.megakernel.utils.support import get_max_num_tokens_padded

NUM_HIDDEN_LAYERS = 48


def get_qwen3_30b_a3b_megakernel_relax_mod(
    mk: MegaKernelWrapper,
    scheduler: Literal["static", "dynamic"],
    TP_SIZE: int,
    PROFILER_ON: bool,
    max_batch_size: int,
):
    assert mk.TIE_WORD_EMBEDDINGS == False, "Qwen3-30B-A3B does not support tie word embeddings"

    max_num_tokens_padded = get_max_num_tokens_padded(
        max_batch_size, mk.NUM_EXPERTS_PER_TOK, mk.NUM_EXPERTS, mk.MOE_M_PAD_SIZE
    )
    assert TP_SIZE == 1

    def static_mod():
        # fmt: off
        @R.macro(hygienic=False)
        def call_qwen3_layer(input0, input1, layer_id, max_num_tokens_padded: int):
            with R.dataflow():
                # 9i+1, 9i+2, 9i+3, 9i+4, 9i+5, 9i+6, 9i+7, 9i+9, 9i+17 if i<num_hidden_layers-1 else 9i+10
                model_layers_0_self_attn_c_attn_weight1: R.Tensor(((mk.NUM_ATTENTION_HEADS + 2* mk.NUM_KEY_VALUE_HEADS) * mk.HEAD_DIM, mk.HIDDEN_SIZE), dtype="float16") = packed_params[9*layer_id+1]
                model_layers_0_self_attn_o_proj_weight1: R.Tensor((mk.HIDDEN_SIZE, mk.NUM_ATTENTION_HEADS * mk.HEAD_DIM), dtype="float16") = packed_params[9*layer_id+2]
                model_layers_0_self_attn_q_norm_weight1: R.Tensor((mk.HEAD_DIM,), dtype="float16") = packed_params[9*layer_id+3]
                model_layers_0_self_attn_k_norm_weight1: R.Tensor((mk.HEAD_DIM,), dtype="float16") = packed_params[9*layer_id+4]
                model_layers_0_moe_gating_weight1: R.Tensor((mk.NUM_EXPERTS, mk.HIDDEN_SIZE), dtype="float16") = packed_params[9*layer_id+5]
                model_layers_0_moe_gate_up_proj_weight1: R.Tensor((mk.NUM_EXPERTS, mk.INTERMEDIATE_SIZE * 2, mk.HIDDEN_SIZE), dtype="float16") = packed_params[9*layer_id+6]
                model_layers_0_moe_down_proj_weight1: R.Tensor((mk.NUM_EXPERTS, mk.HIDDEN_SIZE, mk.INTERMEDIATE_SIZE), dtype="float16") = packed_params[9*layer_id+7]
                model_layers_0_post_attention_layernorm_weight1: R.Tensor((mk.HIDDEN_SIZE,), dtype="float16") = packed_params[9*layer_id+9]
                model_norm_weight1: R.Tensor((mk.HIDDEN_SIZE,), dtype="float16") = packed_params[9*layer_id+17 if layer_id < NUM_HIDDEN_LAYERS-1 else 9*layer_id+10]

                default_device = R.call_pure_packed("runtime.disco.device", sinfo_args=[R.Object])
                partital_qkv = R.builtin.alloc_tensor(R.shape([mk.SPLIT_QKV_PROJECT, batch_size, (mk.NUM_ATTENTION_HEADS + 2* mk.NUM_KEY_VALUE_HEADS) * mk.HEAD_DIM]), dtype="float32", runtime_device_index=0)
                qkv = R.builtin.alloc_tensor(R.shape([batch_size, mk.NUM_ATTENTION_HEADS + 2* mk.NUM_KEY_VALUE_HEADS, mk.HEAD_DIM]), dtype="float16", runtime_device_index=0)
                o = R.builtin.alloc_tensor(R.shape([batch_size, mk.NUM_ATTENTION_HEADS, mk.HEAD_DIM]), dtype="float16", runtime_device_index=0)
                o_partial_attn = R.builtin.alloc_tensor(R.shape([mk.MAX_NUM_KV_SPLITS * mk.NUM_KEY_VALUE_HEADS * mk.HEAD_DIM]), dtype="float32", runtime_device_index=0)
                lse_partial = R.builtin.alloc_tensor(R.shape([mk.MAX_NUM_KV_SPLITS * mk.NUM_KEY_VALUE_HEADS]), dtype="float32", runtime_device_index=0)
                partial_o = R.builtin.alloc_tensor(R.shape([mk.SPLIT_O_PROJECT, batch_size, mk.HIDDEN_SIZE]), dtype="float32", runtime_device_index=0)
                before_o_allreduce = R.call_pure_packed("runtime.disco.nvshmem.empty", R.shape([batch_size, mk.HIDDEN_SIZE]), R.dtype("float16"), default_device, sinfo_args=[R.Tensor((batch_size, mk.HIDDEN_SIZE), "float16")]) if TP_SIZE > 1 else R.builtin.alloc_tensor(R.shape([batch_size, mk.HIDDEN_SIZE]), dtype="float16", runtime_device_index=0)
                hidden_state_attn_mlp = R.call_pure_packed("runtime.disco.nvshmem.empty", R.shape([batch_size, mk.HIDDEN_SIZE]), R.dtype("float16"), default_device, sinfo_args=[R.Tensor((batch_size, mk.HIDDEN_SIZE), "float16")]) if TP_SIZE > 1 else R.builtin.alloc_tensor(R.shape([batch_size, mk.HIDDEN_SIZE]), dtype="float16", runtime_device_index=0)
                
                # MoE intermediate buffers
                gating_output = R.builtin.alloc_tensor(R.shape([batch_size, mk.NUM_EXPERTS]), dtype="float32", runtime_device_index=0)
                topk_weights = R.builtin.alloc_tensor(R.shape([batch_size, mk.NUM_EXPERTS_PER_TOK]), dtype="float32", runtime_device_index=0)
                topk_indices = R.builtin.alloc_tensor(R.shape([batch_size, mk.NUM_EXPERTS_PER_TOK]), dtype="int32", runtime_device_index=0)
                sorted_token_ids = R.builtin.alloc_tensor(R.shape([max_num_tokens_padded]), dtype="int32", runtime_device_index=0)
                expert_ids = R.builtin.alloc_tensor(R.shape([max_num_tokens_padded // 128]), dtype="int32", runtime_device_index=0)
                num_valid_tokens = R.builtin.alloc_tensor(R.shape([max_num_tokens_padded // 128]), dtype="int32", runtime_device_index=0)
                num_tokens_post_pad = R.builtin.alloc_tensor(R.shape([1]), dtype="int32", runtime_device_index=0)
                cumsum_buffer = R.builtin.alloc_tensor(R.shape([mk.NUM_EXPERTS + 1]), dtype="int32", runtime_device_index=0)
                reordered_hidden_state = R.builtin.alloc_tensor(R.shape([max_num_tokens_padded, mk.HIDDEN_SIZE]), dtype="float16", runtime_device_index=0)
                gate_up_output = R.builtin.alloc_tensor(R.shape([max_num_tokens_padded, mk.INTERMEDIATE_SIZE * 2]), dtype="float16", runtime_device_index=0)
                silu_mul_output = R.builtin.alloc_tensor(R.shape([max_num_tokens_padded, mk.INTERMEDIATE_SIZE]), dtype="float16", runtime_device_index=0)
                
                output_tensor = R.call_pure_packed("runtime.disco.nvshmem.empty", R.shape([batch_size, mk.HIDDEN_SIZE]), R.dtype("float16"), default_device, sinfo_args=[R.Tensor((batch_size, mk.HIDDEN_SIZE), "float16")]) if TP_SIZE > 1 else R.builtin.alloc_tensor(R.shape([batch_size, mk.HIDDEN_SIZE]), dtype="float16", runtime_device_index=0)
                profiler_buffer = R.builtin.alloc_tensor(R.shape([mk.PROFILER_BUFFER_SIZE]), dtype="uint64", runtime_device_index=0)

                layer_res = R.call_tir_inplace(
                    cls.layer_kernel,
                    (
                        # input and output
                        input0, input1, output_tensor,
                        # attention weights
                        model_layers_0_self_attn_c_attn_weight1, model_layers_0_self_attn_o_proj_weight1,
                        model_layers_0_self_attn_q_norm_weight1, model_layers_0_self_attn_k_norm_weight1,
                        model_layers_0_post_attention_layernorm_weight1, model_norm_weight1,
                        # MoE weights
                        model_layers_0_moe_gating_weight1, model_layers_0_moe_gate_up_proj_weight1, model_layers_0_moe_down_proj_weight1,
                        # page cache, cos_sin cache and plan info
                        cos_sin_cache, rope_pos, kv_data[layer_id], append_pos, q_indptr, kv_indptr,
                        partial_indptr, page_kv_indices, q_len, kv_len, q_start, kv_start, kv_end, kv_head_idx,
                        work_indptr, len_kv_chunk, num_qo_len, merge_indptr, merge_o_indices, inverse_indptr, inverse_indices,
                        # attention intermediate buffer
                        partital_qkv, qkv, o, o_partial_attn, lse_partial, partial_o, before_o_allreduce,
                        hidden_state_attn_mlp,
                        # MoE intermediate buffer
                        gating_output, topk_weights, topk_indices, sorted_token_ids, expert_ids, num_valid_tokens,
                        num_tokens_post_pad, cumsum_buffer, reordered_hidden_state, gate_up_output, silu_mul_output,
                        # event tensor
                        etensor_workspace,
                        # execution queue
                        exec_queue, profiler_buffer
                    ),
                    [2, 1, 54],
                    out_sinfo=[
                        R.Tensor((batch_size, mk.HIDDEN_SIZE), dtype="float16"), # output
                        R.Tensor((batch_size, mk.HIDDEN_SIZE), dtype="float16" if mk.tp_size > 1 else "float32"), # residual
                        R.Tensor((mk.PROFILER_BUFFER_SIZE,), dtype="uint64"), # profiler
                    ]
                )
                R.output(layer_res)
            return layer_res

        @I.ir_module
        class Module:
            @T.prim_func
            def rms_norm(input_ptr: T.handle, weight_ptr: T.handle, out_ptr: T.handle):
                pass

            @T.prim_func(private=True)
            def cast(var_lv4: T.handle, var_compute: T.handle):
                T.func_attr({"op_pattern": 0, "tir.noalias": True})
                batch_size = T.int64()
                lv4 = T.match_buffer(var_lv4, (batch_size, T.int64(1), T.int64(mk.VOCAB_SIZE)), "float16")
                compute = T.match_buffer(var_compute, (batch_size, T.int64(1), T.int64(mk.VOCAB_SIZE)))
                # with T.block("root"):
                for i0, i1, i2 in T.grid(batch_size, T.int64(1), T.int64(mk.VOCAB_SIZE)):
                    with T.block("compute"):
                        v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                        T.reads(lv4[v_i0, v_i1, v_i2])
                        T.writes(compute[v_i0, v_i1, v_i2])
                        compute[v_i0, v_i1, v_i2] = T.Cast("float32", lv4[v_i0, v_i1, v_i2])
                        
            @T.prim_func(private=True)
            def cast_res(var_res: T.handle, var_compute: T.handle):
                T.func_attr({"op_pattern": 0, "tir.noalias": True})
                batch_size = T.int64()
                res = T.match_buffer(var_res, (batch_size, T.int64(mk.HIDDEN_SIZE)), "float16")
                compute = T.match_buffer(var_compute, (batch_size, T.int64(mk.HIDDEN_SIZE)))
                # with T.block("root"):
                for i0, i1 in T.grid(batch_size, T.int64(mk.HIDDEN_SIZE)):
                    with T.block("compute"):
                        v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                        T.reads(res[v_i0, v_i1])
                        T.writes(compute[v_i0, v_i1])
                        compute[v_i0, v_i1] = T.Cast("float32", res[v_i0, v_i1])

            @T.prim_func(private=True)
            def hgemm(A_ptr: T.handle, B_ptr: T.handle, out_ptr: T.handle):
                pass

            @T.prim_func(private=True)
            def cos_sin_cache(cos_sin_cache: T.handle):
                pass

            @R.function
            def cos_sin_cache_func(max_seq_len_: R.Shape(["max_seq_len"])):
                max_seq_len = T.int64()
                cls = Module
                with R.dataflow():
                    cache: R.Tensor((max_seq_len, 128), dtype="float32") = R.call_tir(cls.cos_sin_cache, [], out_sinfo=R.Tensor((max_seq_len, 128), dtype="float32") )
                    R.output(cache)
                return cache

            @T.prim_func(private=True)
            def layer_kernel(
                # input and output
                hidden_state_ptr: T.handle, # input: read-only
                residual_ptr: T.handle, # input & output: inplace update
                output_ptr: T.handle, # output

                # weight
                qkv_proj_weight_ptr: T.handle, # read-only
                o_proj_weight_ptr: T.handle, # read-only
                q_rms_weight_ptr: T.handle, # read-only
                k_rms_weight_ptr: T.handle, # read-only
                attn_add_rms_weight_ptr: T.handle, # read-only
                mlp_add_rms_weight_ptr: T.handle, # read-only

                # MoE weights
                gate_weight_ptr: T.handle, # read-only
                grp_gate_up_weight_ptr: T.handle, # read-only
                grp_down_weight_ptr: T.handle, # read-only

                # page cache, cos_sin cache and plan info
                cos_sin_cache_ptr: T.handle, # read-only
                rope_pos_ptr: T.handle, # read-only
                kv_cache_ptr: T.handle, # inplace update
                append_pos_ptr: T.handle, # read-only
                q_indptr_ptr : T.handle, # read-only
                kv_indptr_ptr : T.handle, # read-only
                partial_indptr_ptr : T.handle, # read-only
                kv_indices_ptr : T.handle, # read-only
                q_len_ptr : T.handle, # read-only
                kv_len_ptr : T.handle, # read-only
                q_start_ptr : T.handle, # read-only
                kv_start_ptr : T.handle, # read-only
                kv_end_ptr : T.handle, # read-only
                kv_head_idx_ptr : T.handle, # read-only
                work_indptr_ptr : T.handle, # read-only
                len_kv_chunk_ptr : T.handle, # read-only
                num_qo_len_ptr: T.handle, # read-only
                merge_indptr_ptr: T.handle, # read-only
                merge_o_indices_ptr: T.handle, # read-only
                inverse_indptr_ptr: T.handle, # read-only
                inverse_indices_ptr: T.handle, # read-only

                # Attention intermediate buffer
                partial_qkv_ptr: T.handle, # intermediate
                qkv_ptr: T.handle,  # intermediate
                o_ptr: T.handle, # intermediate
                o_partial_attn_ptr: T.handle, # intermediate
                lse_partial_attn_ptr: T.handle, # intermediate
                partial_o_ptr: T.handle, # intermediate
                before_o_allreduce_ptr: T.handle, # intermediate
                hidden_state_attn_mlp_ptr: T.handle, # intermediate

                # MoE intermediate buffer
                gating_output_ptr: T.handle, # intermediate
                topk_weights_ptr: T.handle, # intermediate
                topk_indices_ptr: T.handle, # intermediate
                sorted_token_ids_ptr: T.handle, # intermediate
                expert_ids_ptr: T.handle, # intermediate
                num_valid_tokens_ptr: T.handle, # intermediate
                num_tokens_post_pad_ptr: T.handle, # intermediate
                cumsum_buffer_ptr: T.handle, # intermediate
                reordered_hidden_state_ptr: T.handle, # intermediate
                gate_up_output_ptr: T.handle, # intermediate
                silu_mul_output_ptr: T.handle, # intermediate

                # event tensor
                etensor_workspace_ptr: T.handle,

                # execution queue
                exec_queue_ptr: T.handle,
                profiler_buffer: T.Buffer((mk.PROFILER_BUFFER_SIZE,), "uint64")
            ):
                pass


            @R.function(pure=True)
            def batch_decode_inner(
                input_embeds: R.Tensor(("batch_size", 1, mk.HIDDEN_SIZE), dtype="float16"),
                paged_kv_cache: R.Object,
                # rope
                cos_sin_cache: R.Tensor(("max_seq_len", mk.HEAD_DIM), dtype="float32"),
                packed_params: R.Tuple(
                    R.Tensor((mk.VOCAB_SIZE, mk.HIDDEN_SIZE), dtype="float16"),
                    #
                    *([R.Tensor(((mk.NUM_ATTENTION_HEADS + 2* mk.NUM_KEY_VALUE_HEADS) * mk.HEAD_DIM, mk.HIDDEN_SIZE), dtype="float16"),
                    R.Tensor((mk.HIDDEN_SIZE, mk.NUM_ATTENTION_HEADS * mk.HEAD_DIM), dtype="float16"),
                    R.Tensor((mk.HEAD_DIM,), dtype="float16"),
                    R.Tensor((mk.HEAD_DIM,), dtype="float16"),
                    R.Tensor((mk.NUM_EXPERTS, mk.HIDDEN_SIZE), dtype="float16"),
                    R.Tensor((mk.NUM_EXPERTS, mk.INTERMEDIATE_SIZE * 2, mk.HIDDEN_SIZE), dtype="float16"),
                    R.Tensor((mk.NUM_EXPERTS, mk.HIDDEN_SIZE, mk.INTERMEDIATE_SIZE), dtype="float16"),
                    R.Tensor((mk.HIDDEN_SIZE,), dtype="float16"),
                    R.Tensor((mk.HIDDEN_SIZE,), dtype="float16"),] * NUM_HIDDEN_LAYERS),
                    R.Tensor((mk.HIDDEN_SIZE,), dtype="float16"),
                    R.Tensor((mk.VOCAB_SIZE, mk.HIDDEN_SIZE), dtype="float16"),
                ),
            ):
                batch_size = T.int64()
                total_page_num = T.int64()
                max_page_num = T.int64()

                cls = Module
                with R.dataflow():
                    res0 = R.call_pure_packed(
                        "vm.builtin.paged_attention_kv_cache_tensor_retrieve",
                        paged_kv_cache, R.prim_value(1),
                        sinfo_args=[
                            R.Tuple([R.Tensor(None, dtype="float16")] * NUM_HIDDEN_LAYERS),
                            R.Tensor((batch_size + 1,), dtype="int32"),
                            R.Tensor(None, dtype="int32"),
                            R.Tensor((batch_size,), dtype="int32"),
                            R.Tensor((batch_size,), dtype="int32"),
                            R.Tensor((batch_size,), dtype="int32"),
                            R.Tuple([R.Prim("int64")] * 2 + [R.Tuple([R.Tensor(None, dtype="int32")] * 13)] * 2 + [R.Tensor(None, dtype="int32")] * 5),
                            R.Tensor((mk.MAX_TOTAL_NUM_WORKERS,), dtype="int32"),
                            R.Tensor((mk.MAX_TOTAL_NUM_WORKERS,), dtype="int32"),
                            R.Tensor(None, dtype="int32"),
                            R.Prim("int64"),
                        ],
                    )
                    (
                        kv_data,
                        page_kv_indptr,
                        page_kv_indices,
                        page_kv_last_page_len,
                        append_pos,
                        rope_pos,
                        attn_plan_results,
                        inverse_indptr,
                        inverse_indices,
                        etensor_workspace,
                        attn_task_num,
                    ) = (
                        res0[0],
                        res0[1],
                        res0[2],
                        res0[3],
                        res0[4],
                        res0[5],
                        res0[6],
                        res0[7],
                        res0[8],
                        res0[9],
                        res0[10],
                    )
                    # kv_data = R.match_cast(kv_data_, R.Tuple([R.Tensor((max_page_num, 2, mk.NUM_KEY_VALUE_HEADS, mk.PAGE_SIZE, mk.HEAD_DIM), dtype="float16")] * NUM_HIDDEN_LAYERS))
                    # page_kv_indices = R.match_cast(page_kv_indices_, R.Tensor((total_page_num,), dtype="int32"))
                    task, len_kv_chunk, merge_indptr, merge_o_indices, num_qo_len = attn_plan_results[3], attn_plan_results[4], attn_plan_results[5], attn_plan_results[6], attn_plan_results[7]
                    q_indptr, kv_indptr, partial_indptr, q_len, kv_len, q_start, kv_start, kv_end, kv_head_idx, work_indptr = task[0], task[1], task[2], task[3], task[4], task[5], task[6], task[7], task[8], task[9]
                    # len_kv_chunk = R.match_cast(len_kv_chunk_, R.Tensor((2,), dtype="int32"))
                    # merge_indptr = R.match_cast(merge_indptr_, R.Tensor((mk.MAX_NUM_KV_SPLITS,), dtype="int32"))
                    # merge_o_indices = R.match_cast(merge_o_indices_, R.Tensor((mk.MAX_NUM_KV_SPLITS,), dtype="int32"))
                    # num_qo_len = R.match_cast(num_qo_len_, R.Tensor((1,), dtype="int32"))
                    # q_indptr = R.match_cast(q_indptr_, R.Tensor((mk.MAX_TOTAL_NUM_WORKERS,), dtype="int32"))
                    # kv_indptr = R.match_cast(kv_indptr_, R.Tensor((mk.MAX_TOTAL_NUM_WORKERS,), dtype="int32"))
                    # partial_indptr = R.match_cast(partial_indptr_, R.Tensor((mk.MAX_TOTAL_NUM_WORKERS,), dtype="int32"))
                    # q_len = R.match_cast(q_len_, R.Tensor((mk.MAX_TOTAL_NUM_WORKERS,), dtype="int32"))
                    # kv_len = R.match_cast(kv_len_, R.Tensor((mk.MAX_TOTAL_NUM_WORKERS,), dtype="int32"))
                    # q_start = R.match_cast(q_start_, R.Tensor((mk.MAX_TOTAL_NUM_WORKERS,), dtype="int32"))
                    # kv_start = R.match_cast(kv_start_, R.Tensor((mk.MAX_TOTAL_NUM_WORKERS,), dtype="int32"))
                    # kv_end = R.match_cast(kv_end_, R.Tensor((mk.MAX_TOTAL_NUM_WORKERS,), dtype="int32"))
                    # kv_head_idx = R.match_cast(kv_head_idx_, R.Tensor((mk.MAX_TOTAL_NUM_WORKERS,), dtype="int32"))
                    # work_indptr = R.match_cast(work_indptr_, R.Tensor((mk.MAX_TOTAL_NUM_WORKERS,), dtype="int32"))
                    exec_queue = R.call_pure_packed(
                        "vm.builtin.paged_attention_kv_cache_get_exec_queue",
                        paged_kv_cache,
                        R.prim_value(batch_size),
                        attn_task_num,
                        R.prim_value(-1),
                        sinfo_args=[
                            R.Tensor((148, 128), dtype="int32"),
                        ]
                    )

                    model_layers_0_input_layernorm_weight1: R.Tensor((mk.HIDDEN_SIZE,), dtype="float16") = packed_params[8]
                    lm_head_weight1: R.Tensor((mk.VOCAB_SIZE, mk.HIDDEN_SIZE), dtype="float16") = packed_params[NUM_HIDDEN_LAYERS*9+2] # num_hidden_layers*9+2

                    rs0_ = R.reshape(input_embeds, (batch_size, mk.HIDDEN_SIZE))
                    rms_norm = R.call_tir(cls.rms_norm, (rs0_, model_layers_0_input_layernorm_weight1), out_sinfo=R.Tensor((batch_size, mk.HIDDEN_SIZE), dtype="float16"))
                    rs0 = R.call_tir(cls.cast_res, (rs0_,), out_sinfo=R.Tensor((batch_size, mk.HIDDEN_SIZE), dtype="float32")) if mk.tp_size == 1 else rs0_

                    o_layer0 = call_qwen3_layer(rms_norm, rs0, 0, max_num_tokens_padded)
                    o_layer1 = call_qwen3_layer(o_layer0[0], o_layer0[1], 1, max_num_tokens_padded)
                    o_layer2 = call_qwen3_layer(o_layer1[0], o_layer1[1], 2, max_num_tokens_padded)
                    o_layer3 = call_qwen3_layer(o_layer2[0], o_layer2[1], 3, max_num_tokens_padded)
                    o_layer4 = call_qwen3_layer(o_layer3[0], o_layer3[1], 4, max_num_tokens_padded)
                    o_layer5 = call_qwen3_layer(o_layer4[0], o_layer4[1], 5, max_num_tokens_padded)
                    o_layer6 = call_qwen3_layer(o_layer5[0], o_layer5[1], 6, max_num_tokens_padded)
                    o_layer7 = call_qwen3_layer(o_layer6[0], o_layer6[1], 7, max_num_tokens_padded)
                    o_layer8 = call_qwen3_layer(o_layer7[0], o_layer7[1], 8, max_num_tokens_padded)
                    o_layer9 = call_qwen3_layer(o_layer8[0], o_layer8[1], 9, max_num_tokens_padded)
                    o_layer10 = call_qwen3_layer(o_layer9[0], o_layer9[1], 10, max_num_tokens_padded)
                    o_layer11 = call_qwen3_layer(o_layer10[0], o_layer10[1], 11, max_num_tokens_padded)
                    o_layer12 = call_qwen3_layer(o_layer11[0], o_layer11[1], 12, max_num_tokens_padded)
                    o_layer13 = call_qwen3_layer(o_layer12[0], o_layer12[1], 13, max_num_tokens_padded)
                    o_layer14 = call_qwen3_layer(o_layer13[0], o_layer13[1], 14, max_num_tokens_padded)
                    o_layer15 = call_qwen3_layer(o_layer14[0], o_layer14[1], 15, max_num_tokens_padded)
                    o_layer16 = call_qwen3_layer(o_layer15[0], o_layer15[1], 16, max_num_tokens_padded)
                    o_layer17 = call_qwen3_layer(o_layer16[0], o_layer16[1], 17, max_num_tokens_padded)
                    o_layer18 = call_qwen3_layer(o_layer17[0], o_layer17[1], 18, max_num_tokens_padded)
                    o_layer19 = call_qwen3_layer(o_layer18[0], o_layer18[1], 19, max_num_tokens_padded)
                    o_layer20 = call_qwen3_layer(o_layer19[0], o_layer19[1], 20, max_num_tokens_padded)
                    o_layer21 = call_qwen3_layer(o_layer20[0], o_layer20[1], 21, max_num_tokens_padded)
                    o_layer22 = call_qwen3_layer(o_layer21[0], o_layer21[1], 22, max_num_tokens_padded)
                    o_layer23 = call_qwen3_layer(o_layer22[0], o_layer22[1], 23, max_num_tokens_padded)
                    o_layer24 = call_qwen3_layer(o_layer23[0], o_layer23[1], 24, max_num_tokens_padded)
                    o_layer25 = call_qwen3_layer(o_layer24[0], o_layer24[1], 25, max_num_tokens_padded)
                    o_layer26 = call_qwen3_layer(o_layer25[0], o_layer25[1], 26, max_num_tokens_padded)
                    o_layer27 = call_qwen3_layer(o_layer26[0], o_layer26[1], 27, max_num_tokens_padded)
                    o_layer28 = call_qwen3_layer(o_layer27[0], o_layer27[1], 28, max_num_tokens_padded)
                    o_layer29 = call_qwen3_layer(o_layer28[0], o_layer28[1], 29, max_num_tokens_padded)
                    o_layer30 = call_qwen3_layer(o_layer29[0], o_layer29[1], 30, max_num_tokens_padded)
                    o_layer31 = call_qwen3_layer(o_layer30[0], o_layer30[1], 31, max_num_tokens_padded)
                    o_layer32 = call_qwen3_layer(o_layer31[0], o_layer31[1], 32, max_num_tokens_padded)
                    o_layer33 = call_qwen3_layer(o_layer32[0], o_layer32[1], 33, max_num_tokens_padded)
                    o_layer34 = call_qwen3_layer(o_layer33[0], o_layer33[1], 34, max_num_tokens_padded)
                    o_layer35 = call_qwen3_layer(o_layer34[0], o_layer34[1], 35, max_num_tokens_padded)
                    o_layer36 = call_qwen3_layer(o_layer35[0], o_layer35[1], 36, max_num_tokens_padded)
                    o_layer37 = call_qwen3_layer(o_layer36[0], o_layer36[1], 37, max_num_tokens_padded)
                    o_layer38 = call_qwen3_layer(o_layer37[0], o_layer37[1], 38, max_num_tokens_padded)
                    o_layer39 = call_qwen3_layer(o_layer38[0], o_layer38[1], 39, max_num_tokens_padded)
                    o_layer40 = call_qwen3_layer(o_layer39[0], o_layer39[1], 40, max_num_tokens_padded)
                    o_layer41 = call_qwen3_layer(o_layer40[0], o_layer40[1], 41, max_num_tokens_padded)
                    o_layer42 = call_qwen3_layer(o_layer41[0], o_layer41[1], 42, max_num_tokens_padded)
                    o_layer43 = call_qwen3_layer(o_layer42[0], o_layer42[1], 43, max_num_tokens_padded)
                    o_layer44 = call_qwen3_layer(o_layer43[0], o_layer43[1], 44, max_num_tokens_padded)
                    o_layer45 = call_qwen3_layer(o_layer44[0], o_layer44[1], 45, max_num_tokens_padded)
                    o_layer46 = call_qwen3_layer(o_layer45[0], o_layer45[1], 46, max_num_tokens_padded)
                    o_layer47 = call_qwen3_layer(o_layer46[0], o_layer46[1], 47, max_num_tokens_padded)

                    lv4 = R.call_tir(cls.hgemm, (o_layer47[0], lm_head_weight1), out_sinfo=R.Tensor((batch_size, mk.VOCAB_SIZE), dtype="float16"))
                    lv4_rs = R.reshape(lv4, (batch_size, 1, mk.VOCAB_SIZE))
                    astype = R.call_tir(cls.cast, (lv4_rs,), out_sinfo=R.Tensor((batch_size, 1, mk.VOCAB_SIZE), dtype="float32"))

                    profiler = (
                        o_layer0[2], o_layer1[2], o_layer2[2], o_layer3[2], o_layer4[2], o_layer5[2],
                        o_layer6[2], o_layer7[2], o_layer8[2], o_layer9[2], o_layer10[2], o_layer11[2],
                        o_layer12[2], o_layer13[2], o_layer14[2], o_layer15[2], o_layer16[2], o_layer17[2],
                        o_layer18[2], o_layer19[2], o_layer20[2], o_layer21[2], o_layer22[2], o_layer23[2],
                        o_layer24[2], o_layer25[2], o_layer26[2], o_layer27[2], o_layer28[2], o_layer29[2],
                        o_layer30[2], o_layer31[2], o_layer32[2], o_layer33[2], o_layer34[2], o_layer35[2],
                        o_layer36[2], o_layer37[2], o_layer38[2], o_layer39[2], o_layer40[2], o_layer41[2],
                        o_layer42[2], o_layer43[2], o_layer44[2], o_layer45[2], o_layer46[2], o_layer47[2],
                    )

                    gv1 = astype, paged_kv_cache, profiler
                    R.output(gv1)
                return gv1


            @R.function(pure=False)
            def batch_decode(
                input_embeds: R.Tensor(("batch_size", 1, mk.HIDDEN_SIZE), dtype="float16"),
                paged_kv_cache: R.Object,
                # rope
                cos_sin_cache: R.Tensor(("max_seq_len", mk.HEAD_DIM), dtype="float32"),
                packed_params: R.Tuple(
                    R.Tensor((mk.VOCAB_SIZE, mk.HIDDEN_SIZE), dtype="float16"),
                    #
                    *([R.Tensor(((mk.NUM_ATTENTION_HEADS + 2* mk.NUM_KEY_VALUE_HEADS) * mk.HEAD_DIM, mk.HIDDEN_SIZE), dtype="float16"),
                    R.Tensor((mk.HIDDEN_SIZE, mk.NUM_ATTENTION_HEADS * mk.HEAD_DIM), dtype="float16"),
                    R.Tensor((mk.HEAD_DIM,), dtype="float16"),
                    R.Tensor((mk.HEAD_DIM,), dtype="float16"),
                    R.Tensor((mk.NUM_EXPERTS, mk.HIDDEN_SIZE), dtype="float16"),
                    R.Tensor((mk.NUM_EXPERTS, mk.INTERMEDIATE_SIZE * 2, mk.HIDDEN_SIZE), dtype="float16"),
                    R.Tensor((mk.NUM_EXPERTS, mk.HIDDEN_SIZE, mk.INTERMEDIATE_SIZE), dtype="float16"),
                    R.Tensor((mk.HIDDEN_SIZE,), dtype="float16"),
                    R.Tensor((mk.HIDDEN_SIZE,), dtype="float16"),] * NUM_HIDDEN_LAYERS),
                    R.Tensor((mk.HIDDEN_SIZE,), dtype="float16"),
                    R.Tensor((mk.VOCAB_SIZE, mk.HIDDEN_SIZE), dtype="float16"),
                ),
            ):
                cls = Module
                model_output = cls.batch_decode_inner(
                    input_embeds,
                    paged_kv_cache,
                    cos_sin_cache,
                    packed_params,
                )
                if PROFILER_ON:
                    rank = R.call_packed("runtime.disco.worker_id", sinfo_args=[R.Shape()]) if TP_SIZE > 1 else R.prim_value(-1)
                    _ = R.call_packed(
                        "megakernel.export_trace",
                        model_output[2],
                        rank,
                        sinfo_args=[],
                    )
                else:
                    # there must be some dummy thing in else statement to avoid error
                    _ = R.prim_value(-1)

                res = model_output[0], model_output[1]
                return res

        # fmt: on
        return Module

    def dynamic_mod():
        # fmt: off
        @R.macro(hygienic=False)
        def call_qwen3_layer(input0, input1, layer_id, max_num_tokens_padded: int):
            with R.dataflow():
                # 9i+1, 9i+2, 9i+3, 9i+4, 9i+5, 9i+6, 9i+7, 9i+9, 9i+17 if i<num_hidden_layers-1 else 9i+10
                model_layers_0_self_attn_c_attn_weight1: R.Tensor(((mk.NUM_ATTENTION_HEADS + 2* mk.NUM_KEY_VALUE_HEADS) * mk.HEAD_DIM, mk.HIDDEN_SIZE), dtype="float16") = packed_params[9*layer_id+1]
                model_layers_0_self_attn_o_proj_weight1: R.Tensor((mk.HIDDEN_SIZE, mk.NUM_ATTENTION_HEADS * mk.HEAD_DIM), dtype="float16") = packed_params[9*layer_id+2]
                model_layers_0_self_attn_q_norm_weight1: R.Tensor((mk.HEAD_DIM,), dtype="float16") = packed_params[9*layer_id+3]
                model_layers_0_self_attn_k_norm_weight1: R.Tensor((mk.HEAD_DIM,), dtype="float16") = packed_params[9*layer_id+4]
                model_layers_0_moe_gating_weight1: R.Tensor((mk.NUM_EXPERTS, mk.HIDDEN_SIZE), dtype="float16") = packed_params[9*layer_id+5]
                model_layers_0_moe_gate_up_proj_weight1: R.Tensor((mk.NUM_EXPERTS, mk.INTERMEDIATE_SIZE * 2, mk.HIDDEN_SIZE), dtype="float16") = packed_params[9*layer_id+6]
                model_layers_0_moe_down_proj_weight1: R.Tensor((mk.NUM_EXPERTS, mk.HIDDEN_SIZE, mk.INTERMEDIATE_SIZE), dtype="float16") = packed_params[9*layer_id+7]
                model_layers_0_post_attention_layernorm_weight1: R.Tensor((mk.HIDDEN_SIZE,), dtype="float16") = packed_params[9*layer_id+9]
                model_norm_weight1: R.Tensor((mk.HIDDEN_SIZE,), dtype="float16") = packed_params[9*layer_id+17 if layer_id < NUM_HIDDEN_LAYERS-1 else 9*layer_id+10]

                default_device = R.call_pure_packed("runtime.disco.device", sinfo_args=[R.Object])
                partital_qkv = R.builtin.alloc_tensor(R.shape([mk.SPLIT_QKV_PROJECT, batch_size, (mk.NUM_ATTENTION_HEADS + 2* mk.NUM_KEY_VALUE_HEADS) * mk.HEAD_DIM]), dtype="float32", runtime_device_index=0)
                qkv = R.builtin.alloc_tensor(R.shape([batch_size, mk.NUM_ATTENTION_HEADS + 2* mk.NUM_KEY_VALUE_HEADS, mk.HEAD_DIM]), dtype="float16", runtime_device_index=0)
                o = R.builtin.alloc_tensor(R.shape([batch_size, mk.NUM_ATTENTION_HEADS, mk.HEAD_DIM]), dtype="float16", runtime_device_index=0)
                o_partial_attn = R.builtin.alloc_tensor(R.shape([mk.MAX_NUM_KV_SPLITS * mk.NUM_KEY_VALUE_HEADS * mk.HEAD_DIM]), dtype="float32", runtime_device_index=0)
                lse_partial = R.builtin.alloc_tensor(R.shape([mk.MAX_NUM_KV_SPLITS * mk.NUM_KEY_VALUE_HEADS]), dtype="float32", runtime_device_index=0)
                partial_o = R.builtin.alloc_tensor(R.shape([mk.SPLIT_O_PROJECT, batch_size, mk.HIDDEN_SIZE]), dtype="float32", runtime_device_index=0)
                before_o_allreduce = R.call_pure_packed("runtime.disco.nvshmem.empty", R.shape([batch_size, mk.HIDDEN_SIZE]), R.dtype("float16"), default_device, sinfo_args=[R.Tensor((batch_size, mk.HIDDEN_SIZE), "float16")]) if TP_SIZE > 1 else R.builtin.alloc_tensor(R.shape([batch_size, mk.HIDDEN_SIZE]), dtype="float16", runtime_device_index=0)
                hidden_state_attn_mlp = R.call_pure_packed("runtime.disco.nvshmem.empty", R.shape([batch_size, mk.HIDDEN_SIZE]), R.dtype("float16"), default_device, sinfo_args=[R.Tensor((batch_size, mk.HIDDEN_SIZE), "float16")]) if TP_SIZE > 1 else R.builtin.alloc_tensor(R.shape([batch_size, mk.HIDDEN_SIZE]), dtype="float16", runtime_device_index=0)
                
                # MoE intermediate buffers
                gating_output = R.builtin.alloc_tensor(R.shape([batch_size, mk.NUM_EXPERTS]), dtype="float32", runtime_device_index=0)
                topk_weights = R.builtin.alloc_tensor(R.shape([batch_size, mk.NUM_EXPERTS_PER_TOK]), dtype="float32", runtime_device_index=0)
                topk_indices = R.builtin.alloc_tensor(R.shape([batch_size, mk.NUM_EXPERTS_PER_TOK]), dtype="int32", runtime_device_index=0)
                sorted_token_ids = R.builtin.alloc_tensor(R.shape([max_num_tokens_padded]), dtype="int32", runtime_device_index=0)
                expert_ids = R.builtin.alloc_tensor(R.shape([max_num_tokens_padded // 128]), dtype="int32", runtime_device_index=0)
                num_valid_tokens = R.builtin.alloc_tensor(R.shape([max_num_tokens_padded // 128]), dtype="int32", runtime_device_index=0)
                num_tokens_post_pad = R.builtin.alloc_tensor(R.shape([1]), dtype="int32", runtime_device_index=0)
                cumsum_buffer = R.builtin.alloc_tensor(R.shape([mk.NUM_EXPERTS + 1]), dtype="int32", runtime_device_index=0)
                reordered_hidden_state = R.builtin.alloc_tensor(R.shape([max_num_tokens_padded, mk.HIDDEN_SIZE]), dtype="float16", runtime_device_index=0)
                gate_up_output = R.builtin.alloc_tensor(R.shape([max_num_tokens_padded, mk.INTERMEDIATE_SIZE * 2]), dtype="float16", runtime_device_index=0)
                silu_mul_output = R.builtin.alloc_tensor(R.shape([max_num_tokens_padded, mk.INTERMEDIATE_SIZE]), dtype="float16", runtime_device_index=0)
                
                output_tensor = R.call_pure_packed("runtime.disco.nvshmem.empty", R.shape([batch_size, mk.HIDDEN_SIZE]), R.dtype("float16"), default_device, sinfo_args=[R.Tensor((batch_size, mk.HIDDEN_SIZE), "float16")]) if TP_SIZE > 1 else R.builtin.alloc_tensor(R.shape([batch_size, mk.HIDDEN_SIZE]), dtype="float16", runtime_device_index=0)
                profiler_buffer = R.builtin.alloc_tensor(R.shape([mk.PROFILER_BUFFER_SIZE]), dtype="uint64", runtime_device_index=0)
                exec_queue = R.call_pure_packed(
                    "vm.builtin.paged_attention_kv_cache_get_exec_queue",
                    paged_kv_cache,
                    R.prim_value(batch_size),
                    attn_task_num,
                    R.prim_value(layer_id),
                    sinfo_args=[
                        R.Tensor((DynamicTileScheduler.MAX_TASKS,), dtype="int32"),
                        R.Tensor((1,), dtype="int32"),
                        R.Tensor((1,), dtype="int32"),
                    ]
                )
                queue_tasks, queue_head, queue_tail = exec_queue[0], exec_queue[1], exec_queue[2]


                layer_res = R.call_tir_inplace(
                    cls.layer_kernel,
                    (
                        # input and output
                        input0, input1, output_tensor,
                        # attention weights
                        model_layers_0_self_attn_c_attn_weight1, model_layers_0_self_attn_o_proj_weight1,
                        model_layers_0_self_attn_q_norm_weight1, model_layers_0_self_attn_k_norm_weight1,
                        model_layers_0_post_attention_layernorm_weight1, model_norm_weight1,
                        # MoE weights
                        model_layers_0_moe_gating_weight1, model_layers_0_moe_gate_up_proj_weight1, model_layers_0_moe_down_proj_weight1,
                        # page cache, cos_sin cache and plan info
                        cos_sin_cache, rope_pos, kv_data[layer_id], append_pos, q_indptr, kv_indptr,
                        partial_indptr, page_kv_indices, q_len, kv_len, q_start, kv_start, kv_end, kv_head_idx,
                        work_indptr, len_kv_chunk, num_qo_len, merge_indptr, merge_o_indices, inverse_indptr, inverse_indices,
                        # attention intermediate buffer
                        partital_qkv, qkv, o, o_partial_attn, lse_partial, partial_o, before_o_allreduce,
                        hidden_state_attn_mlp,
                        # MoE intermediate buffer
                        gating_output, topk_weights, topk_indices, sorted_token_ids, expert_ids, num_valid_tokens,
                        num_tokens_post_pad, cumsum_buffer, reordered_hidden_state, gate_up_output, silu_mul_output,
                        # event tensor
                        etensor_workspace,
                        # execution queue
                        queue_tasks, queue_head, queue_tail, profiler_buffer
                    ),
                    [2, 1, 56],
                    out_sinfo=[
                        R.Tensor((batch_size, mk.HIDDEN_SIZE), dtype="float16"), # output
                        R.Tensor((batch_size, mk.HIDDEN_SIZE), dtype="float16" if TP_SIZE > 1 else "float32"), # residual
                        R.Tensor((mk.PROFILER_BUFFER_SIZE,), dtype="uint64"), # profiler
                    ]
                )
                R.output(layer_res)
            return layer_res

        @I.ir_module
        class Module:
            @T.prim_func
            def rms_norm(input_ptr: T.handle, weight_ptr: T.handle, out_ptr: T.handle):
                pass

            @T.prim_func(private=True)
            def cast(var_lv4: T.handle, var_compute: T.handle):
                T.func_attr({"op_pattern": 0, "tir.noalias": True})
                batch_size = T.int64()
                lv4 = T.match_buffer(var_lv4, (batch_size, T.int64(1), T.int64(mk.VOCAB_SIZE)), "float16")
                compute = T.match_buffer(var_compute, (batch_size, T.int64(1), T.int64(mk.VOCAB_SIZE)))
                # with T.block("root"):
                for i0, i1, i2 in T.grid(batch_size, T.int64(1), T.int64(mk.VOCAB_SIZE)):
                    with T.block("compute"):
                        v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                        T.reads(lv4[v_i0, v_i1, v_i2])
                        T.writes(compute[v_i0, v_i1, v_i2])
                        compute[v_i0, v_i1, v_i2] = T.Cast("float32", lv4[v_i0, v_i1, v_i2])

            @T.prim_func(private=True)
            def cast_res(var_res: T.handle, var_compute: T.handle):
                T.func_attr({"op_pattern": 0, "tir.noalias": True})
                batch_size = T.int64()
                res = T.match_buffer(var_res, (batch_size, T.int64(mk.HIDDEN_SIZE)), "float16")
                compute = T.match_buffer(var_compute, (batch_size, T.int64(mk.HIDDEN_SIZE)))
                # with T.block("root"):
                for i0, i1 in T.grid(batch_size, T.int64(mk.HIDDEN_SIZE)):
                    with T.block("compute"):
                        v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                        T.reads(res[v_i0, v_i1])
                        T.writes(compute[v_i0, v_i1])
                        compute[v_i0, v_i1] = T.Cast("float32", res[v_i0, v_i1])

            @T.prim_func(private=True)
            def hgemm(A_ptr: T.handle, B_ptr: T.handle, out_ptr: T.handle):
                pass

            @T.prim_func(private=True)
            def cos_sin_cache(cos_sin_cache: T.handle):
                pass

            @R.function
            def cos_sin_cache_func(max_seq_len_: R.Shape(["max_seq_len"])):
                max_seq_len = T.int64()
                cls = Module
                with R.dataflow():
                    cache: R.Tensor((max_seq_len, 128), dtype="float32") = R.call_tir(cls.cos_sin_cache, [], out_sinfo=R.Tensor((max_seq_len, 128), dtype="float32") )
                    R.output(cache)
                return cache

            @T.prim_func(private=True)
            def layer_kernel(
                # input and output
                hidden_state_ptr: T.handle, # input: read-only
                residual_ptr: T.handle, # input & output: inplace update
                output_ptr: T.handle, # output

                # weight
                qkv_proj_weight_ptr: T.handle, # read-only
                o_proj_weight_ptr: T.handle, # read-only
                q_rms_weight_ptr: T.handle, # read-only
                k_rms_weight_ptr: T.handle, # read-only
                attn_add_rms_weight_ptr: T.handle, # read-only
                mlp_add_rms_weight_ptr: T.handle, # read-only

                # MoE weights
                gate_weight_ptr: T.handle, # read-only
                grp_gate_up_weight_ptr: T.handle, # read-only
                grp_down_weight_ptr: T.handle, # read-only

                # page cache, cos_sin cache and plan info
                cos_sin_cache_ptr: T.handle, # read-only
                rope_pos_ptr: T.handle, # read-only
                kv_cache_ptr: T.handle, # inplace update
                append_pos_ptr: T.handle, # read-only
                q_indptr_ptr : T.handle, # read-only
                kv_indptr_ptr : T.handle, # read-only
                partial_indptr_ptr : T.handle, # read-only
                kv_indices_ptr : T.handle, # read-only
                q_len_ptr : T.handle, # read-only
                kv_len_ptr : T.handle, # read-only
                q_start_ptr : T.handle, # read-only
                kv_start_ptr : T.handle, # read-only
                kv_end_ptr : T.handle, # read-only
                kv_head_idx_ptr : T.handle, # read-only
                work_indptr_ptr : T.handle, # read-only
                len_kv_chunk_ptr : T.handle, # read-only
                num_qo_len_ptr: T.handle, # read-only
                merge_indptr_ptr: T.handle, # read-only
                merge_o_indices_ptr: T.handle, # read-only
                inverse_indptr_ptr: T.handle, # read-only
                inverse_indices_ptr: T.handle, # read-only

                # Attention intermediate buffer
                partial_qkv_ptr: T.handle, # intermediate
                qkv_ptr: T.handle,  # intermediate
                o_ptr: T.handle, # intermediate
                o_partial_attn_ptr: T.handle, # intermediate
                lse_partial_attn_ptr: T.handle, # intermediate
                partial_o_ptr: T.handle, # intermediate
                before_o_allreduce_ptr: T.handle, # intermediate
                hidden_state_attn_mlp_ptr: T.handle, # intermediate

                # MoE intermediate buffer
                gating_output_ptr: T.handle, # intermediate
                topk_weights_ptr: T.handle, # intermediate
                topk_indices_ptr: T.handle, # intermediate
                sorted_token_ids_ptr: T.handle, # intermediate
                expert_ids_ptr: T.handle, # intermediate
                num_valid_tokens_ptr: T.handle, # intermediate
                num_tokens_post_pad_ptr: T.handle, # intermediate
                cumsum_buffer_ptr: T.handle, # intermediate
                reordered_hidden_state_ptr: T.handle, # intermediate
                gate_up_output_ptr: T.handle, # intermediate
                silu_mul_output_ptr: T.handle, # intermediate

                # event tensor
                etensor_workspace_ptr: T.handle,

                # execution queue
                queue_tasks_ptr: T.handle,
                queue_head_ptr: T.handle,
                queue_tail_ptr: T.handle,
                profiler_buffer: T.Buffer((mk.PROFILER_BUFFER_SIZE,), "uint64")
            ):
                pass


            @R.function(pure=True)
            def batch_decode_inner(
                input_embeds: R.Tensor(("batch_size", 1, mk.HIDDEN_SIZE), dtype="float16"),
                paged_kv_cache: R.Object,
                # rope
                cos_sin_cache: R.Tensor(("max_seq_len", mk.HEAD_DIM), dtype="float32"),
                packed_params: R.Tuple(
                    R.Tensor((mk.VOCAB_SIZE, mk.HIDDEN_SIZE), dtype="float16"),
                    #
                    *([R.Tensor(((mk.NUM_ATTENTION_HEADS + 2* mk.NUM_KEY_VALUE_HEADS) * mk.HEAD_DIM, mk.HIDDEN_SIZE), dtype="float16"),
                    R.Tensor((mk.HIDDEN_SIZE, mk.NUM_ATTENTION_HEADS * mk.HEAD_DIM), dtype="float16"),
                    R.Tensor((mk.HEAD_DIM,), dtype="float16"),
                    R.Tensor((mk.HEAD_DIM,), dtype="float16"),
                    R.Tensor((mk.NUM_EXPERTS, mk.HIDDEN_SIZE), dtype="float16"),
                    R.Tensor((mk.NUM_EXPERTS, mk.INTERMEDIATE_SIZE * 2, mk.HIDDEN_SIZE), dtype="float16"),
                    R.Tensor((mk.NUM_EXPERTS, mk.HIDDEN_SIZE, mk.INTERMEDIATE_SIZE), dtype="float16"),
                    R.Tensor((mk.HIDDEN_SIZE,), dtype="float16"),
                    R.Tensor((mk.HIDDEN_SIZE,), dtype="float16"),] * NUM_HIDDEN_LAYERS),
                    R.Tensor((mk.HIDDEN_SIZE,), dtype="float16"),
                    R.Tensor((mk.VOCAB_SIZE, mk.HIDDEN_SIZE), dtype="float16"),
                ),
            ):
                batch_size = T.int64()
                total_page_num = T.int64()
                max_page_num = T.int64()
                page_size = T.int64()

                cls = Module
                with R.dataflow():
                    res0 = R.call_pure_packed(
                        "vm.builtin.paged_attention_kv_cache_tensor_retrieve",
                        paged_kv_cache, R.prim_value(1),
                        sinfo_args=[
                            R.Tuple([R.Tensor(None, dtype="float16")] * NUM_HIDDEN_LAYERS),
                            R.Tensor((batch_size + 1,), dtype="int32"),
                            R.Tensor(None, dtype="int32"),
                            R.Tensor((batch_size,), dtype="int32"),
                            R.Tensor((batch_size,), dtype="int32"),
                            R.Tensor((batch_size,), dtype="int32"),
                            R.Tuple([R.Prim("int64")] * 2 + [R.Tuple([R.Tensor(None, dtype="int32")] * 13)] * 2 + [R.Tensor(None, dtype="int32")] * 5),
                            R.Tensor((mk.MAX_TOTAL_NUM_WORKERS,), dtype="int32"),
                            R.Tensor((mk.MAX_TOTAL_NUM_WORKERS,), dtype="int32"),
                            R.Tensor(None, dtype="int32"),
                            R.Prim("int64"),
                        ],
                    )
                    (
                        kv_data_,
                        page_kv_indptr,
                        page_kv_indices_,
                        page_kv_last_page_len,
                        append_pos,
                        rope_pos,
                        attn_plan_results,
                        inverse_indptr,
                        inverse_indices,
                        etensor_workspace,
                        attn_task_num,
                    ) = (
                        res0[0],
                        res0[1],
                        res0[2],
                        res0[3],
                        res0[4],
                        res0[5],
                        res0[6],
                        res0[7],
                        res0[8],
                        res0[9],
                        res0[10],
                    )
                    kv_data = R.match_cast(kv_data_, R.Tuple([R.Tensor((max_page_num, 2, mk.NUM_KEY_VALUE_HEADS, page_size, mk.HEAD_DIM), dtype="float16")] * NUM_HIDDEN_LAYERS))
                    page_kv_indices = R.match_cast(page_kv_indices_, R.Tensor((total_page_num,), dtype="int32"))
                    task, len_kv_chunk_, merge_indptr_, merge_o_indices_, num_qo_len_ = attn_plan_results[3], attn_plan_results[4], attn_plan_results[5], attn_plan_results[6], attn_plan_results[7]
                    q_indptr_, kv_indptr_, partial_indptr_, q_len_, kv_len_, q_start_, kv_start_, kv_end_, kv_head_idx_, work_indptr_ = task[0], task[1], task[2], task[3], task[4], task[5], task[6], task[7], task[8], task[9]
                    len_kv_chunk = R.match_cast(len_kv_chunk_, R.Tensor((2,), dtype="int32"))
                    merge_indptr = R.match_cast(merge_indptr_, R.Tensor((mk.MAX_NUM_KV_SPLITS,), dtype="int32"))
                    merge_o_indices = R.match_cast(merge_o_indices_, R.Tensor((mk.MAX_NUM_KV_SPLITS,), dtype="int32"))
                    num_qo_len = R.match_cast(num_qo_len_, R.Tensor((1,), dtype="int32"))
                    q_indptr = R.match_cast(q_indptr_, R.Tensor((mk.MAX_TOTAL_NUM_WORKERS,), dtype="int32"))
                    kv_indptr = R.match_cast(kv_indptr_, R.Tensor((mk.MAX_TOTAL_NUM_WORKERS,), dtype="int32"))
                    partial_indptr = R.match_cast(partial_indptr_, R.Tensor((mk.MAX_TOTAL_NUM_WORKERS,), dtype="int32"))
                    q_len = R.match_cast(q_len_, R.Tensor((mk.MAX_TOTAL_NUM_WORKERS,), dtype="int32"))
                    kv_len = R.match_cast(kv_len_, R.Tensor((mk.MAX_TOTAL_NUM_WORKERS,), dtype="int32"))
                    q_start = R.match_cast(q_start_, R.Tensor((mk.MAX_TOTAL_NUM_WORKERS,), dtype="int32"))
                    kv_start = R.match_cast(kv_start_, R.Tensor((mk.MAX_TOTAL_NUM_WORKERS,), dtype="int32"))
                    kv_end = R.match_cast(kv_end_, R.Tensor((mk.MAX_TOTAL_NUM_WORKERS,), dtype="int32"))
                    kv_head_idx = R.match_cast(kv_head_idx_, R.Tensor((mk.MAX_TOTAL_NUM_WORKERS,), dtype="int32"))
                    work_indptr = R.match_cast(work_indptr_, R.Tensor((mk.MAX_TOTAL_NUM_WORKERS,), dtype="int32"))
                    model_layers_0_input_layernorm_weight1: R.Tensor((mk.HIDDEN_SIZE,), dtype="float16") = packed_params[8]
                    lm_head_weight1: R.Tensor((mk.VOCAB_SIZE, mk.HIDDEN_SIZE), dtype="float16") = packed_params[NUM_HIDDEN_LAYERS*9+2] # num_hidden_layers*9+2

                    rs0_ = R.reshape(input_embeds, (batch_size, mk.HIDDEN_SIZE))
                    rms_norm = R.call_tir(cls.rms_norm, (rs0_, model_layers_0_input_layernorm_weight1), out_sinfo=R.Tensor((batch_size, mk.HIDDEN_SIZE), dtype="float16"))
                    rs0 = R.call_tir(cls.cast_res, (rs0_,), out_sinfo=R.Tensor((batch_size, mk.HIDDEN_SIZE), dtype="float32")) if mk.tp_size == 1 else rs0_

                    o_layer0 = call_qwen3_layer(rms_norm, rs0, 0, max_num_tokens_padded)
                    o_layer1 = call_qwen3_layer(o_layer0[0], o_layer0[1], 1, max_num_tokens_padded)
                    o_layer2 = call_qwen3_layer(o_layer1[0], o_layer1[1], 2, max_num_tokens_padded)
                    o_layer3 = call_qwen3_layer(o_layer2[0], o_layer2[1], 3, max_num_tokens_padded)
                    o_layer4 = call_qwen3_layer(o_layer3[0], o_layer3[1], 4, max_num_tokens_padded)
                    o_layer5 = call_qwen3_layer(o_layer4[0], o_layer4[1], 5, max_num_tokens_padded)
                    o_layer6 = call_qwen3_layer(o_layer5[0], o_layer5[1], 6, max_num_tokens_padded)
                    o_layer7 = call_qwen3_layer(o_layer6[0], o_layer6[1], 7, max_num_tokens_padded)
                    o_layer8 = call_qwen3_layer(o_layer7[0], o_layer7[1], 8, max_num_tokens_padded)
                    o_layer9 = call_qwen3_layer(o_layer8[0], o_layer8[1], 9, max_num_tokens_padded)
                    o_layer10 = call_qwen3_layer(o_layer9[0], o_layer9[1], 10, max_num_tokens_padded)
                    o_layer11 = call_qwen3_layer(o_layer10[0], o_layer10[1], 11, max_num_tokens_padded)
                    o_layer12 = call_qwen3_layer(o_layer11[0], o_layer11[1], 12, max_num_tokens_padded)
                    o_layer13 = call_qwen3_layer(o_layer12[0], o_layer12[1], 13, max_num_tokens_padded)
                    o_layer14 = call_qwen3_layer(o_layer13[0], o_layer13[1], 14, max_num_tokens_padded)
                    o_layer15 = call_qwen3_layer(o_layer14[0], o_layer14[1], 15, max_num_tokens_padded)
                    o_layer16 = call_qwen3_layer(o_layer15[0], o_layer15[1], 16, max_num_tokens_padded)
                    o_layer17 = call_qwen3_layer(o_layer16[0], o_layer16[1], 17, max_num_tokens_padded)
                    o_layer18 = call_qwen3_layer(o_layer17[0], o_layer17[1], 18, max_num_tokens_padded)
                    o_layer19 = call_qwen3_layer(o_layer18[0], o_layer18[1], 19, max_num_tokens_padded)
                    o_layer20 = call_qwen3_layer(o_layer19[0], o_layer19[1], 20, max_num_tokens_padded)
                    o_layer21 = call_qwen3_layer(o_layer20[0], o_layer20[1], 21, max_num_tokens_padded)
                    o_layer22 = call_qwen3_layer(o_layer21[0], o_layer21[1], 22, max_num_tokens_padded)
                    o_layer23 = call_qwen3_layer(o_layer22[0], o_layer22[1], 23, max_num_tokens_padded)
                    o_layer24 = call_qwen3_layer(o_layer23[0], o_layer23[1], 24, max_num_tokens_padded)
                    o_layer25 = call_qwen3_layer(o_layer24[0], o_layer24[1], 25, max_num_tokens_padded)
                    o_layer26 = call_qwen3_layer(o_layer25[0], o_layer25[1], 26, max_num_tokens_padded)
                    o_layer27 = call_qwen3_layer(o_layer26[0], o_layer26[1], 27, max_num_tokens_padded)
                    o_layer28 = call_qwen3_layer(o_layer27[0], o_layer27[1], 28, max_num_tokens_padded)
                    o_layer29 = call_qwen3_layer(o_layer28[0], o_layer28[1], 29, max_num_tokens_padded)
                    o_layer30 = call_qwen3_layer(o_layer29[0], o_layer29[1], 30, max_num_tokens_padded)
                    o_layer31 = call_qwen3_layer(o_layer30[0], o_layer30[1], 31, max_num_tokens_padded)
                    o_layer32 = call_qwen3_layer(o_layer31[0], o_layer31[1], 32, max_num_tokens_padded)
                    o_layer33 = call_qwen3_layer(o_layer32[0], o_layer32[1], 33, max_num_tokens_padded)
                    o_layer34 = call_qwen3_layer(o_layer33[0], o_layer33[1], 34, max_num_tokens_padded)
                    o_layer35 = call_qwen3_layer(o_layer34[0], o_layer34[1], 35, max_num_tokens_padded)
                    o_layer36 = call_qwen3_layer(o_layer35[0], o_layer35[1], 36, max_num_tokens_padded)
                    o_layer37 = call_qwen3_layer(o_layer36[0], o_layer36[1], 37, max_num_tokens_padded)
                    o_layer38 = call_qwen3_layer(o_layer37[0], o_layer37[1], 38, max_num_tokens_padded)
                    o_layer39 = call_qwen3_layer(o_layer38[0], o_layer38[1], 39, max_num_tokens_padded)
                    o_layer40 = call_qwen3_layer(o_layer39[0], o_layer39[1], 40, max_num_tokens_padded)
                    o_layer41 = call_qwen3_layer(o_layer40[0], o_layer40[1], 41, max_num_tokens_padded)
                    o_layer42 = call_qwen3_layer(o_layer41[0], o_layer41[1], 42, max_num_tokens_padded)
                    o_layer43 = call_qwen3_layer(o_layer42[0], o_layer42[1], 43, max_num_tokens_padded)
                    o_layer44 = call_qwen3_layer(o_layer43[0], o_layer43[1], 44, max_num_tokens_padded)
                    o_layer45 = call_qwen3_layer(o_layer44[0], o_layer44[1], 45, max_num_tokens_padded)
                    o_layer46 = call_qwen3_layer(o_layer45[0], o_layer45[1], 46, max_num_tokens_padded)
                    o_layer47 = call_qwen3_layer(o_layer46[0], o_layer46[1], 47, max_num_tokens_padded)

                    # permute_dims4 = R.permute_dims(lm_head_weight1, axes=None)
                    # lv4: R.Tensor((batch_size, mk.VOCAB_SIZE), dtype="float16") = R.matmul(rms_norm4, permute_dims4, out_dtype="void")
                    lv4 = R.call_tir(cls.hgemm, (o_layer47[0], lm_head_weight1), out_sinfo=R.Tensor((batch_size, mk.VOCAB_SIZE), dtype="float16"))
                    lv4_rs = R.reshape(lv4, (batch_size, 1, mk.VOCAB_SIZE))
                    astype = R.call_tir(cls.cast, (lv4_rs,), out_sinfo=R.Tensor((batch_size, 1, mk.VOCAB_SIZE), dtype="float32"))

                    profiler = (
                        o_layer0[2], o_layer1[2], o_layer2[2], o_layer3[2], o_layer4[2], o_layer5[2],
                        o_layer6[2], o_layer7[2], o_layer8[2], o_layer9[2], o_layer10[2], o_layer11[2],
                        o_layer12[2], o_layer13[2], o_layer14[2], o_layer15[2], o_layer16[2], o_layer17[2],
                        o_layer18[2], o_layer19[2], o_layer20[2], o_layer21[2], o_layer22[2], o_layer23[2],
                        o_layer24[2], o_layer25[2], o_layer26[2], o_layer27[2], o_layer28[2], o_layer29[2],
                        o_layer30[2], o_layer31[2], o_layer32[2], o_layer33[2], o_layer34[2], o_layer35[2],
                        o_layer36[2], o_layer37[2], o_layer38[2], o_layer39[2], o_layer40[2], o_layer41[2],
                        o_layer42[2], o_layer43[2], o_layer44[2], o_layer45[2], o_layer46[2], o_layer47[2],
                    )

                    gv1 = astype, paged_kv_cache, profiler
                    R.output(gv1)
                return gv1


            @R.function(pure=False)
            def batch_decode(
                input_embeds: R.Tensor(("batch_size", 1, mk.HIDDEN_SIZE), dtype="float16"),
                paged_kv_cache: R.Object,
                # rope
                cos_sin_cache: R.Tensor(("max_seq_len", mk.HEAD_DIM), dtype="float32"),
                packed_params: R.Tuple(
                    R.Tensor((mk.VOCAB_SIZE, mk.HIDDEN_SIZE), dtype="float16"),
                    #
                    *([R.Tensor(((mk.NUM_ATTENTION_HEADS + 2* mk.NUM_KEY_VALUE_HEADS) * mk.HEAD_DIM, mk.HIDDEN_SIZE), dtype="float16"),
                    R.Tensor((mk.HIDDEN_SIZE, mk.NUM_ATTENTION_HEADS * mk.HEAD_DIM), dtype="float16"),
                    R.Tensor((mk.HEAD_DIM,), dtype="float16"),
                    R.Tensor((mk.HEAD_DIM,), dtype="float16"),
                    R.Tensor((mk.NUM_EXPERTS, mk.HIDDEN_SIZE), dtype="float16"),
                    R.Tensor((mk.NUM_EXPERTS, mk.INTERMEDIATE_SIZE * 2, mk.HIDDEN_SIZE), dtype="float16"),
                    R.Tensor((mk.NUM_EXPERTS, mk.HIDDEN_SIZE, mk.INTERMEDIATE_SIZE), dtype="float16"),
                    R.Tensor((mk.HIDDEN_SIZE,), dtype="float16"),
                    R.Tensor((mk.HIDDEN_SIZE,), dtype="float16"),] * NUM_HIDDEN_LAYERS),
                    R.Tensor((mk.HIDDEN_SIZE,), dtype="float16"),
                    R.Tensor((mk.VOCAB_SIZE, mk.HIDDEN_SIZE), dtype="float16"),
                ),
            ):
                cls = Module
                model_output = cls.batch_decode_inner(
                    input_embeds,
                    paged_kv_cache,
                    cos_sin_cache,
                    packed_params,
                )
                if PROFILER_ON:
                    rank = R.call_packed("runtime.disco.worker_id", sinfo_args=[R.Shape()]) if TP_SIZE > 1 else R.prim_value(-1)
                    _ = R.call_packed(
                        "megakernel.export_trace",
                        model_output[2],
                        rank,
                        sinfo_args=[],
                    )
                else:
                    # there must be some dummy thing in else statement to avoid error
                    _ = R.prim_value(-1)

                res = model_output[0], model_output[1]
                return res

        # fmt: on
        return Module

    if scheduler == "static":
        return static_mod()
    elif scheduler == "dynamic":
        return dynamic_mod()
    else:
        raise NotImplementedError
