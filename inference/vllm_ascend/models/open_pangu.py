# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import torch
import torch_npu
import vllm.envs as envs
from torch import nn
from transformers import PretrainedConfig
from vllm.compilation.decorators import support_torch_compile
from vllm.attention import Attention, AttentionMetadata, AttentionType
from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              get_tp_group, split_tensor_along_last_dim,
                              tensor_model_parallel_all_gather,
                              tensor_model_parallel_all_reduce,
                              tensor_model_parallel_reduce_scatter)
from vllm.distributed.parallel_state import get_dp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               MergedColumnParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear,
                                               UnquantizedLinearMethod,
                                               QKVParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope, _rotate_gptj
from vllm.model_executor.layers.sampler import get_sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)
from vllm.model_executor.models.utils import (
    make_layers, maybe_prefix, extract_layer_index)
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.distributed.parallel_state import get_ep_group
from vllm_ascend.ops.fused_moe import AscendFusedMoE
from vllm_ascend.quantization.quant_config import AscendLinearMethod
from vllm_ascend.quantization.w8a8_dynamic import AscendW8A8DynamicLinearMethod
from vllm_ascend.utils import dispose_tensor, npu_prefetch, get_fused_moe_state, FusedMoEState
from vllm.model_executor.sampling_metadata import SamplingMetadata


class OpenPanguMergedReplicatedLinear(ReplicatedLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = True,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size,
                         sum(output_sizes),
                         bias=bias,
                         quant_config=quant_config,
                         prefix=prefix)

    def weight_loader(self, param: torch.nn.Parameter,
                      loaded_weight: torch.Tensor, loaded_shard_id: int):
        # With no support for GGUF format yet.
        if getattr(param, "is_gguf_weight", False) or getattr(param, "is_gguf_weight_type", False):
            raise ValueError('With no support for GGUF format yet.')
        if loaded_shard_id >= len(self.output_sizes):
            raise ValueError(f'loaded_shard_id {loaded_shard_id} >= len(self.output_sizes) {len(self.output_sizes)}.')
        shard_offset = sum(self.output_sizes[:loaded_shard_id])
        shard_size = self.output_sizes[loaded_shard_id]
        shard = param.data.narrow(param.output_dim, shard_offset, shard_size)
        if shard.size() != loaded_weight.size():
            raise ValueError(f"Tried to load weights of size {loaded_weight.size()} "
                             f"to a parameter shard of id {loaded_shard_id} size {shard.size()}.")
        shard.copy_(loaded_weight)


class OpenPanguRowParallelLinearReplaceAllreduce(RowParallelLinear):

    def forward(
        self,
        input_,
        is_prefill=True
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Optional[nn.Parameter]]]:
        if self.input_is_parallel:
            input_parallel = input_
        else:
            tp_rank = get_tensor_model_parallel_rank()
            splitted_input = split_tensor_along_last_dim(
                input_, num_partitions=self.tp_size)
            input_parallel = splitted_input[tp_rank].contiguous()

        # Matrix multiply.
        if self.quant_method is None:
            raise ValueError('self.quant_method is None.')
        # Only fuse bias add into GEMM for rank 0 (this ensures that
        # bias will not get added more than once in TP>1 case)
        bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
        output_parallel = self.quant_method.apply(self,
                                                  input_parallel,
                                                  bias=bias_)
        if self.reduce_results and self.tp_size > 1:
            if not is_prefill and output_parallel.shape[0] % self.tp_size == 0:
                output = tensor_model_parallel_reduce_scatter(output_parallel,
                                                              dim=0)
            else:
                output = tensor_model_parallel_all_reduce(output_parallel)
        else:
            output = output_parallel

        output_bias = self.bias if self.skip_bias_add else None

        if not self.return_bias:
            return output
        return output, output_bias


class OpenPanguRowParallelLinear(RowParallelLinear):

    def forward(
        self,
        input_,
        is_prefill=True
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Optional[nn.Parameter]]]:
        return super().forward(input_)


class OpenPanguRotaryEmbedding(nn.Module):
    def __init__(self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ):
        super().__init__()
        self.dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device='npu',
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self,
        seq_len: int,
        device: str,
        dtype: torch.dtype
    ):
        self.max_seq_len = seq_len
        inv_freq = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device='npu') / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        t = torch.arange(seq_len, device='npu', dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self,
                positions: torch.Tensor,
                query: torch.Tensor,
                key: torch.Tensor,
                offsets: Optional[torch.Tensor] = None,
                max_seq_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if max_seq_len is not None and max_seq_len > self.max_seq_len:
            self._set_cos_sin_cache(max_seq_len, query.device, query.dtype)
        idx = torch.add(positions, offsets) if offsets is not None else positions
        cos = self.cos_cached[idx]
        sin = self.sin_cached[idx]
        # Adapt: adapt cos and sin shape
        cos = cos.view(-1, 1, cos.shape[-1])
        sin = sin.view(-1, 1, sin.shape[-1])
        # Adapt end.
        query_rot = query * cos + _rotate_gptj(query) * sin
        if key is not None:
            key_rot = key * cos + _rotate_gptj(key) * sin
        return query_rot, key_rot


class OpenPanguSiluAndMul(SiluAndMul):

    def __init__(self,
                 *,
                 weight_scale: Optional[Callable[[], torch.Tensor]] = None):
        super().__init__()
        self.weight_scale = weight_scale

    def forward_oot(self, x: Union[torch.Tensor, Tuple[torch.Tensor,
                                                       torch.Tensor]]):
        if isinstance(x, tuple):
            if self.weight_scale is None:
                raise ValueError('self.weight_scale is None.')
            quantized_x, dynamic_scale = x
            return torch_npu.npu_dequant_swiglu_quant(
                x=quantized_x,
                weight_scale=self.weight_scale(),
                activation_scale=dynamic_scale,
                activate_left=True,
                quant_mode=1)
        else:
            return super().forward_oot(x)


def check_ffn_act_fn(act_fn: str):
    if act_fn != "silu":
        raise ValueError(
            f"Unsupported activation: {act_fn}. Only silu is supported for now.")


class OpenPanguMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False, 
        reduce_results: bool = True,
        force_replicate: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if not force_replicate:
            self.gate_up_proj = MergedColumnParallelLinear(
                hidden_size, [intermediate_size] * 2,
                bias=bias,
                quant_config=quant_config,
                prefix=f"{prefix}.gate_up_proj")
            self.down_proj = RowParallelLinear(intermediate_size,
                                               hidden_size,
                                               bias=bias,
                                               quant_config=quant_config,
                                               reduce_results=reduce_results,
                                               prefix=f"{prefix}.down_proj")
        else:
            self.gate_up_proj = OpenPanguMergedReplicatedLinear(
                                     hidden_size, [intermediate_size] * 2,
                                     bias=bias,
                                     quant_config=quant_config,
                                     prefix=f"{prefix}.gate_up_proj")
            self.down_proj = ReplicatedLinear(intermediate_size,
                                            hidden_size,
                                            bias=bias,
                                            quant_config=quant_config,
                                            prefix=f"{prefix}.down_proj")

        check_ffn_act_fn(hidden_act)

        quant_method = self.gate_up_proj.quant_method
        if isinstance(quant_method, UnquantizedLinearMethod):
            self.act_fn = OpenPanguSiluAndMul()
        elif (isinstance(quant_method, AscendLinearMethod) and isinstance(
                quant_method.quant_method, AscendW8A8DynamicLinearMethod)):
            # TODO(sdmyzlp): Currently preserved as before:
            # 1. The only quantization supported for silu is W8A8Dynamic
            # 2. Output dtype of gate_up/down is fixed to be int32/bfloat16
            #
            # Maybe one can implement a better and more general configuration
            # scheme, e.g. by somehow passing around the tweaked `quant_config`
            self.act_fn = OpenPanguSiluAndMul(
                # Use lazy binding, for `weight_scale_fp32` is accessible
                # only after `process_weights_after_loading`.
                weight_scale=lambda: self.gate_up_proj.weight_scale_fp32)
            # To be consumed by AscendW8A8DynamicLinearMethod.apply()
            self.gate_up_proj._ascend_quant_config = {
                "output_dtype": torch.int32,
                "pertoken_scale": False,
                "return_scale": True,
            }
            self.down_proj._ascend_quant_config = {
                "output_dtype": torch.bfloat16,
                "pertoken_scale": True,
                "return_scale": False,
            }
        else:
            raise NotImplementedError(
                f"Quantization with [{type(quant_method)}] is NOT supported")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_up_proj(x)[0]))[0]


class OpenPanguMoE(nn.Module):

    top_k: int

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        ascend_config = get_ascend_config()
        self.torchair_graph_enabled = ascend_config.torchair_graph_config.enabled
        self.enable_multistream_moe = \
            ascend_config.torchair_graph_config.enable_multistream_moe
        self.routed_scaling_factor = config.routed_scaling_factor
        check_ffn_act_fn(config.hidden_act)

        self.gate = ReplicatedLinear(config.hidden_size,
                                     config.num_routed_experts,
                                     bias=False,
                                     quant_config=None,
                                     prefix=f"{prefix}.gate")

        self.experts = AscendFusedMoE(
            num_experts=config.num_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            reduce_results=False,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            use_grouped_topk=True,
            num_expert_group=1,
            topk_group=1,
            prefix=f"{prefix}.experts",
            scoring_func='sigmoid',
            e_score_correction_bias=None)

        if config.num_shared_experts is not None:
            self.all_reduce_merge = self.experts.all_reduce_merge
            reduce_results = not self.all_reduce_merge
            intermediate_size = (config.moe_intermediate_size * config.num_shared_experts)
            self.shared_experts = OpenPanguMLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=reduce_results,
                force_replicate=self.enable_multistream_moe,
                prefix=f"{prefix}.shared_experts",
            )
        else:
            self.shared_experts = None  # type: ignore

        self.tp_size = get_tensor_model_parallel_world_size()
        self.dp_size = get_dp_group().world_size
        self.tp_group = get_tp_group().device_group
        self.tp_rank = get_tp_group().rank_in_group
        self.ep_group = get_ep_group()

        self.params_dtype = torch.get_default_dtype()
        self.rm_router_logits = self.experts.rm_router_logits

        self.__class__.top_k = config.num_experts_per_tok

    def forward(self,
                hidden_states: torch.Tensor,
                attn_metadata: Optional[AttentionMetadata] = None,
                replace_allreduce: bool = False) -> torch.Tensor:

        if attn_metadata is None:
            attn_metadata = get_forward_context().attn_metadata
        # when profile runs, force experts to load balanced tokens
        # to avoid high memory consumption on a single rank.
        # TODO: need a better flag to indicate whether in profile run or not.
        if attn_metadata is None:
            # for profile run
            is_prefill = True
            fused_moe_state = get_fused_moe_state(self.ep_group.world_size, is_prefill, True)
            enable_force_load_balance = fused_moe_state != FusedMoEState.AllGatherEP
        else:
            is_prefill = attn_metadata.num_prefills > 0
            enable_force_load_balance = False
            if hasattr(attn_metadata, 'with_prefill_across_dp'):
                is_prefill = is_prefill or attn_metadata.with_prefill_across_dp
            fused_moe_state = get_fused_moe_state(self.ep_group.world_size, is_prefill, True)

        # router_logits: (num_tokens, n_experts)
        router_logits = None
        if not self.rm_router_logits or fused_moe_state == FusedMoEState.All2All:
            router_logits, _ = self.gate(hidden_states.float())

        routed_hidden_states, shared_hidden_states = self.experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            is_prefill=is_prefill,
            top_k=self.__class__.top_k,
            enable_force_load_balance=enable_force_load_balance,
            shared_experts=self.shared_experts,
            gate=self.gate,
            replace_allreduce=replace_allreduce)

        if self.all_reduce_merge and fused_moe_state == FusedMoEState.All2All:
            shared_hidden_states = tensor_model_parallel_all_reduce(shared_hidden_states)
        hidden_states = routed_hidden_states * self.routed_scaling_factor + shared_hidden_states
        if self.all_reduce_merge and fused_moe_state != FusedMoEState.All2All:
            # When all_reduce_merge is in progress, shared_experts does not do all_reduce in mlp, but waits until shared_experts+router_experts are completed before doing all_reduce
            hidden_states = tensor_model_parallel_all_reduce(hidden_states)

        return hidden_states


class OpenPanguMLAAttention(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        attention_qk_dim: int,
        attention_qk_rope_dim: int,
        attention_v_dim: int,
        attention_q_lora_dim: Optional[int],
        attention_kv_lora_dim: int,
        rope_theta: float = 10000,
        max_position_embeddings: int = 8192,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        ascend_config = get_ascend_config()
        self.torchair_graph_enabled = ascend_config.torchair_graph_config.enabled
        self.enable_multistream_mla = ascend_config.torchair_graph_config.enable_multistream_mla

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attention_qk_dim = attention_qk_dim
        self.attention_qk_rope_dim = attention_qk_rope_dim
        self.qk_head_dim = attention_qk_dim + attention_qk_rope_dim
        self.attention_v_dim = attention_v_dim
        self.attention_q_lora_dim = attention_q_lora_dim
        self.attention_kv_lora_dim = attention_kv_lora_dim
        self.rope_theta = rope_theta

        tp_size = get_tensor_model_parallel_world_size()
        if num_heads % tp_size != 0:
            raise ValueError(f'num_heads {num_heads} is not divisible by tp_size {tp_size}.')
        self.num_local_heads = num_heads // tp_size

        self.scaling = self.qk_head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings

        self.prefix = prefix
        self.debug_layer_idx = int(self.prefix.split(".")[-2])

        if self.attention_q_lora_dim is not None:
            self.q_a_proj = ReplicatedLinear(self.hidden_size,
                                             self.attention_q_lora_dim,
                                             bias=False,
                                             quant_config=quant_config,
                                             prefix=f"{prefix}.q_a_proj")
            self.q_a_layernorm = RMSNorm(self.attention_q_lora_dim, eps=config.rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(attention_q_lora_dim,
                                                 self.num_heads * self.qk_head_dim,
                                                 bias=False,
                                                 quant_config=quant_config,
                                                 prefix=f"{prefix}.q_b_proj")
        else:
            self.q_proj = ColumnParallelLinear(self.hidden_size,
                                               self.num_heads * self.qk_head_dim,
                                               bias=False,
                                               quant_config=quant_config,
                                               prefix=f"{prefix}.q_proj")

        self.kv_a_proj_with_mqa = ReplicatedLinear(
            self.hidden_size,
            self.attention_kv_lora_dim + self.attention_qk_rope_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_a_proj_with_mqa")
        self.kv_a_layernorm = RMSNorm(self.attention_kv_lora_dim,
                                      eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.attention_kv_lora_dim,
            self.num_heads * (self.attention_qk_dim + self.attention_v_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj")
        if (config.num_routed_experts is not None
                and self.debug_layer_idx >= config.num_dense_layers and
                ascend_config.torchair_graph_config.enable_multistream_moe):
            self.o_proj = OpenPanguRowParallelLinearReplaceAllreduce(
                self.num_heads * self.attention_v_dim,
                self.hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.o_proj")
        else:
            self.o_proj = OpenPanguRowParallelLinear(
                self.num_heads * self.attention_v_dim,
                self.hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.o_proj")

        self.rotary_emb = OpenPanguRotaryEmbedding(attention_qk_rope_dim,
                                                   rotary_dim=attention_qk_rope_dim,
                                                   max_position_embeddings=max_position_embeddings,
                                                   base=rope_theta)

        self.mla_attn = Attention(
            num_heads=self.num_local_heads,
            head_size=self.attention_kv_lora_dim + self.attention_qk_rope_dim,
            scale=self.scaling,
            num_kv_heads=1,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            use_mla=True,
            # MLA Args
            q_lora_rank=self.attention_q_lora_dim,
            kv_lora_rank=self.attention_kv_lora_dim,
            qk_nope_head_dim=self.attention_qk_dim,
            qk_rope_head_dim=self.attention_qk_rope_dim,
            qk_head_dim=self.qk_head_dim,
            v_head_dim=self.attention_v_dim,
            rotary_emb=self.rotary_emb,
            q_proj=self.q_proj if self.attention_q_lora_dim is None else self.q_b_proj,
            kv_a_proj_with_mqa=self.kv_a_proj_with_mqa,
            kv_a_layernorm=self.kv_a_layernorm,
            kv_b_proj=self.kv_b_proj,
            o_proj=self.o_proj,
        )

    def forward(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            kv_cache: Optional[torch.Tensor] = None,
            attn_metadata: Optional[AttentionMetadata] = None) -> torch.Tensor:
        enable_multistream_mla = (self.enable_multistream_mla
                                  and attn_metadata is not None
                                  and not attn_metadata.with_prefill_across_dp
                                  and attn_metadata.num_decodes > 0)
        forward_kwargs = {"enable_multistream_mla": enable_multistream_mla}
        if self.attention_q_lora_dim is not None:
            npu_prefetch(self.q_a_proj.weight,
                         hidden_states,
                         enabled=enable_multistream_mla)
            ckq = self.q_a_proj(hidden_states)[0]
            hidden_states_or_q_c = self.q_a_layernorm(ckq)
            forward_kwargs['ckq'] = ckq
        else:
            hidden_states_or_q_c = hidden_states
        if self.torchair_graph_enabled:
            if envs.VLLM_USE_V1:
                output_shape = hidden_states.shape
                output = torch.empty(output_shape,
                                     dtype=hidden_states_or_q_c.dtype,
                                     device=hidden_states_or_q_c.device)
                forward_kwargs['output'] = output

            output = self.mla_attn.impl.forward(self.mla_attn,
                                                hidden_states_or_q_c,
                                                hidden_states, None, kv_cache,
                                                attn_metadata,
                                                **forward_kwargs)
            if envs.VLLM_USE_V1:
                output = output.view(-1, output_shape[-1])
            return output
        else:
            kv_c, k_pe = self.kv_a_proj_with_mqa(hidden_states)[0].split(
                [self.attention_kv_lora_dim, self.attention_qk_rope_dim], dim=-1)
            kv_c_normed = self.kv_a_layernorm(kv_c.contiguous())
            return self.mla_attn(hidden_states_or_q_c,
                                 kv_c_normed,
                                 k_pe,
                                 output_shape=hidden_states.shape)


class OpenPanguEmbeddedAttention(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
        bias_o_proj: bool = False,
        cache_config: Optional[CacheConfig] = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
    ) -> None:
        super().__init__()
        layer_idx = extract_layer_index(prefix)
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        if self.total_num_heads % tp_size != 0:
            raise ValueError(f'total_num_heads {total_num_heads} is not divisible by tp_size {tp_size}.')
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size and self.total_num_kv_heads % tp_size != 0:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel NPUs.
            raise ValueError(f'Number of KV heads is less than TP size, but total_num_kv_heads {self.total_num_kv_heads} '
                             f'is not divisible by tp_size {tp_size}.')
        elif self.total_num_kv_heads < tp_size and tp_size % self.total_num_kv_heads != 0:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel NPUs.
            raise ValueError(f'Number of KV heads is less than TP size, but tp_size {tp_size} '
                                f'is not divisible by total_num_kv_heads {self.total_num_kv_heads}.')
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        # MistralConfig has an optional head_dim introduced by Mistral-Nemo
        head_dim = getattr(config, "head_dim", None)
        if head_dim is None:
            head_dim = self.hidden_size // self.total_num_heads
        self.head_dim = head_dim
        # Phi models introduced a partial_rotary_factor parameter in the config
        self.partial_rotary_factor = getattr(config, "partial_rotary_factor", 1)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=bias_o_proj,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self._init_rotary_emb(config,
                              rope_scaling=rope_scaling,
                              quant_config=quant_config)

        if hasattr(config, "interleaved_sliding_window"):
            interleaved_sliding_window = config.interleaved_sliding_window
            if isinstance(interleaved_sliding_window, int):
                sliding_window = interleaved_sliding_window
            elif isinstance(interleaved_sliding_window, list):
                sw_idx = layer_idx % len(interleaved_sliding_window)
                sliding_window = interleaved_sliding_window[sw_idx]
            else:
                raise ValueError(
                    f"{type(interleaved_sliding_window)} is not supported.")
        else:
            sliding_window = None

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=sliding_window,
            attn_type=attn_type,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: Optional[torch.Tensor] = None,
        attn_metadata: Optional[AttentionMetadata] = None
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output

    def _init_rotary_emb(self, config: PretrainedConfig,
                         rope_scaling: Optional[dict[str, Any]],
                         quant_config: Optional[QuantizationConfig]) -> None:
        is_neox_style = True
        is_gguf = quant_config and quant_config.get_name() == "gguf"
        if is_gguf and config.model_type == "Pangu":
            is_neox_style = False

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=self.max_position_embeddings,
            base=self.rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=is_neox_style,
            #partial_rotary_factor=self.partial_rotary_factor,
        )


class OpenPanguDecoderLayer(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str,
        model_config: ModelConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)

        layer_idx = int(prefix.split(sep='.')[-1])
        self.layer_idx = layer_idx
        self.layers = config.num_hidden_layers
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tp_group().rank_in_group
        ascend_config = get_ascend_config()

        self.use_mla = hasattr(config, 'attention_qk_dim') and hasattr(config, 'attention_qk_rope_dim') \
            and hasattr(config, 'attention_v_dim') and hasattr(config, 'attention_kv_lora_dim')
        if self.use_mla:
            self.self_attn = OpenPanguMLAAttention(
                config=config,
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                attention_qk_dim=config.attention_qk_dim,
                attention_qk_rope_dim=config.attention_qk_rope_dim,
                attention_v_dim=config.attention_v_dim,
                attention_q_lora_dim=config.attention_q_lora_dim
                if hasattr(config, "attention_q_lora_dim") else None,
                attention_kv_lora_dim=config.attention_kv_lora_dim,
                rope_theta=rope_theta,
                max_position_embeddings=max_position_embeddings,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
            )
        else:
            attention_bias = getattr(config, "attention_bias", False) or getattr(
                config, "bias", False)
            bias_o_proj = attention_bias
            if hasattr(config, 'qkv_bias'):
                attention_bias = config.qkv_bias
            # By default, PanguEmbedded uses causal attention as it is a decoder-only model.
            # You can override the HF config with `is_causal=False` to enable
            # bidirectional attention, which is used in some embedding models
            if getattr(config, "is_causal", True):
                attn_type = AttentionType.DECODER
            else:
                attn_type = AttentionType.ENCODER_ONLY
            self.self_attn = OpenPanguEmbeddedAttention(
                config=config,
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=getattr(config, "num_key_value_heads", config.num_attention_heads),
                rope_theta=rope_theta,
                rope_scaling=getattr(config, "rope_scaling", None),
                max_position_embeddings=max_position_embeddings,
                quant_config=quant_config,
                bias=attention_bias,
                bias_o_proj=bias_o_proj,
                cache_config=cache_config,
                prefix=f"{prefix}.self_attn",
                attn_type=attn_type,
            )

        if getattr(config, 'num_routed_experts', None) is not None and layer_idx >= config.num_dense_layers:
            self.mlp = OpenPanguMoE(
                config=config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
            self.mla_moe_communication = ascend_config.torchair_graph_config.enable_multistream_moe \
                and model_config.use_mla and envs.VLLM_USE_V1 and self.tp_size > 1
        else:
            self.mlp = OpenPanguMLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                bias=getattr(config, "mlp_bias", False),
                prefix=f"{prefix}.mlp",
            )
            self.mla_moe_communication = False
        self.routed_scaling_factor = getattr(config, 'routed_scaling_factor', None)
        self.num_dense_layers = getattr(config, 'num_dense_layers', None)

        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)
        if getattr(config, 'sandwich_norm', False):
            self.sandwich_norm = True
            self.pre_mlp_layernorm = RMSNorm(config.hidden_size,
                                             eps=config.rms_norm_eps)
            self.post_mlp_layernorm = RMSNorm(config.hidden_size,
                                              eps=config.rms_norm_eps)
        else:
            self.sandwich_norm = False

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        kv_cache: Optional[torch.Tensor] = None,
        attn_metadata: Optional[AttentionMetadata] = None,
        replace_allreduce: bool = False,
    ) -> torch.Tensor:
        # Self Attention
        if self.use_mla and attn_metadata is not None and attn_metadata.num_decodes > 0:
            mla_moe_communication = self.mla_moe_communication and replace_allreduce
        else:
            mla_moe_communication = False
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            previous_hidden_states, previous_residual = hidden_states, residual
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
            # Dispose hidden_states and residual from the previous layer
            # to save npu memory because they're no longer used.
            dispose_tensor(previous_hidden_states)
            dispose_tensor(previous_residual)
        if mla_moe_communication and self.layer_idx > self.num_dense_layers:
            hidden_states = tensor_model_parallel_all_gather(hidden_states,
                                                             dim=0)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )

        if mla_moe_communication and residual.shape[0] != hidden_states.shape[0]:
            chunk_hidden_states = torch.tensor_split(residual,
                                                     self.tp_size,
                                                     dim=0)
            residual = chunk_hidden_states[self.tp_rank]

        if self.routed_scaling_factor is not None and hidden_states.dtype == torch.float16:
            # Fix FP16 overflow
            # We scale both hidden_states and residual before
            # rmsnorm, and rmsnorm result would not affect by scale.
            hidden_states *= 1. / self.routed_scaling_factor
            if self.layer_idx == 0:
                # The residual is shared by all layers, we only scale it on
                # first layer.
                residual *= 1. / self.routed_scaling_factor

        if self.sandwich_norm:
            hidden_states = self.post_attention_layernorm(
                hidden_states)
            hidden_states, residual = self.pre_mlp_layernorm(
                hidden_states, residual)
        else:
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual)

        # Fully Connected
        if isinstance(self.mlp, OpenPanguMoE):
            hidden_states = self.mlp(hidden_states,
                                     attn_metadata,
                                     replace_allreduce=mla_moe_communication)
        else:
            hidden_states = self.mlp(hidden_states)

        if self.routed_scaling_factor is not None and isinstance(self.mlp, OpenPanguMLP) \
            and hidden_states.dtype == torch.float16:
            hidden_states *= 1. / self.routed_scaling_factor

        if self.sandwich_norm:
            hidden_states = self.post_mlp_layernorm(hidden_states)

        if mla_moe_communication and self.layer_idx == self.layers - 1:
            hidden_states = tensor_model_parallel_all_gather(hidden_states,
                                                             dim=0)
            residual = tensor_model_parallel_all_gather(residual, dim=0)

        return hidden_states, residual


@support_torch_compile
class OpenPanguModel(nn.Module):

    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.tp_size = get_tensor_model_parallel_world_size()

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.embed_tokens")

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: OpenPanguDecoderLayer(
                config,
                prefix,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
            ),
            prefix=f"{prefix}.layers")

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: Optional[List[torch.Tensor]] = None,
        attn_metadata: Optional[AttentionMetadata] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.get_input_embeddings(input_ids)
        residual = None

        replace_allreduce = hidden_states.shape[0] % self.tp_size == 0

        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
                kv_caches[i -
                          self.start_layer] if kv_caches is not None else None,
                attn_metadata,
                replace_allreduce=replace_allreduce)

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class OpenPanguForCausalLM(nn.Module):
    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
        "experts":
        ["experts.0.gate_proj", "experts.0.up_proj", "experts.0.down_proj"]
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.model = OpenPanguModel(vllm_config=vllm_config,
                                    prefix=maybe_prefix(prefix, "model"))
        self.lm_head = ParallelLMHead(config.vocab_size,
                                      config.hidden_size,
                                      quant_config=quant_config,
                                      prefix=maybe_prefix(prefix, "lm_head"))
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.sampler = get_sampler()
    
    def load_attn_mlp_weight(self,
                             attn_mlp_replace_mapping: List[Tuple[str, str, int]],
                             params_dict: Dict[str, Any],
                             weight_name: str,
                             loaded_weight: torch.Tensor,
                             loaded_params: set[str]) -> bool:
        for (param_name, origin_name, shard_id) in attn_mlp_replace_mapping:
            if origin_name not in weight_name or \
                (("mlp.experts." in weight_name) and weight_name not in params_dict):
                continue
            weight_name = weight_name.replace(origin_name, param_name)
            if weight_name.endswith(".bias") and weight_name not in params_dict:
                continue
            param = params_dict[weight_name]
            weight_loader = param.weight_loader
            weight_loader(param, loaded_weight, shard_id)
            loaded_params.add(weight_name)
            return True
        return False
    
    def load_expert_weight(self,
                           expert_merge_mapping: List[Tuple[str, str, int, str]],
                           params_dict: Dict[str, Any],
                           weight_name: str,
                           loaded_weight: torch.Tensor,
                           loaded_params: set[str]) -> bool:
        for mapping in expert_merge_mapping:
            param_name, origin_name, expert_id, shard_id = mapping
            if origin_name not in weight_name:
                continue
            weight_name = weight_name.replace(origin_name, param_name)
            param = params_dict[weight_name]
            weight_loader = param.weight_loader
            weight_loader(param,
                          loaded_weight,
                          weight_name,
                          shard_id=shard_id,
                          expert_id=expert_id,
                          return_success=False)
            loaded_params.add(weight_name)
            return True
        return False

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        # (param_name, shard_name, shard_id)
        attn_mlp_replace_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        has_experts = hasattr(self.config, 'num_routed_experts')
        if has_experts:
            expert_merge_mapping = AscendFusedMoE.make_expert_params_mapping(
                ckpt_gate_proj_name="gate_proj",
                ckpt_down_proj_name="down_proj",
                ckpt_up_proj_name="up_proj",
                num_experts=self.config.num_routed_experts)

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if 'layers' in name: # skip spec decode layers for main model
                layer_idx = int(name.split('layers.')[-1].split('.')[0])
                if layer_idx > self.config.num_hidden_layers:
                    continue

            if 'layers' in name and hasattr(self.config, "num_mtp_layers") \
                and (self.config.num_mtp_layers > 0):
                layer_idx = int(name.split('layers.')[-1].split('.')[0])
                mtp_idx = layer_idx - self.config.num_hidden_layers
                if mtp_idx >= 0 and mtp_idx < self.config.num_mtp_layers:
                    continue # skip spec decode layers for main model
            if self.load_attn_mlp_weight(attn_mlp_replace_mapping, params_dict, name, loaded_weight, loaded_params):
                continue
            elif has_experts and self.load_expert_weight(expert_merge_mapping, params_dict, name, loaded_weight, loaded_params):
                continue
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        return loaded_params

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: Optional[List[torch.Tensor]] = None,
        attn_metadata: Optional[AttentionMetadata] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   attn_metadata, inputs_embeds)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states, sampling_metadata)
        return logits


class PanguUltraMoEForCausalLM(OpenPanguForCausalLM):
    pass


class PanguEmbeddedForCausalLM(OpenPanguForCausalLM):
    pass
