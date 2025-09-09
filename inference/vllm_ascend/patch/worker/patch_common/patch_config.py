#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#
from vllm.config import ModelConfig


def get_attr_by_names(src_config, attrs, default_value):
    for attr in attrs:
        value = getattr(src_config, attr, 0)
        if value > 0:
            return value
    return default_value


def _verify_with_expert_parallelism(self) -> None:
        num_expert_names = [
            "moe_num_experts",  # Dbrx
            "num_experts",  # Jamba
            "n_routed_experts",  # DeepSeek
            "num_local_experts",  # Mixtral
            "num_routed_experts",  # Pangu
        ]
        num_experts = 0
        for name in num_expert_names:
            num_experts = getattr(self.hf_text_config, name, 0)
            if num_experts > 0:
                break
        if num_experts < 1:
            raise ValueError(
                "Number of experts in the model must be greater than 0 "
                "when expert parallelism is enabled.")


@property
def is_deepseek_mla(self) -> bool:
    kv_lora_dim_names = ['attention_kv_lora_dim', 'kv_lora_rank']
    kv_lora_dim = get_attr_by_names(self.hf_text_config, kv_lora_dim_names, None)
    if not hasattr(self.hf_text_config, "model_type"):
        return False
    elif self.hf_text_config.model_type in \
        ('deepseek_v2', 'deepseek_v3', 'deepseek_mtp', 'pangu_ultra_moe'):
        return kv_lora_dim is not None
    elif self.hf_text_config.model_type == 'eagle':
        # if the model is an EAGLE module, check for the
        # underlying architecture
        return self.hf_text_config.model.model_type in \
                ('deepseek_v2', 'deepseek_v3', 'pangu_ultra_moe') \
            and kv_lora_dim is not None
    return False


def get_head_size(self) -> int:
    if self.is_deepseek_mla:
        qk_rope_dim_names = ['attention_qk_rope_dim', 'qk_rope_head_dim']
        kv_lora_dim_names = ['attention_kv_lora_dim', 'kv_lora_rank']
        qk_rope_dim = get_attr_by_names(self.hf_text_config, qk_rope_dim_names, 0)
        kv_lora_dim = get_attr_by_names(self.hf_text_config, kv_lora_dim_names, 0)
        if self.use_mla:
            return kv_lora_dim + qk_rope_dim
        else:
            qk_dim_names = ['attention_qk_dim', 'qk_nope_head_dim']
            qk_dim = get_attr_by_names(self.hf_text_config, qk_dim_names, 0)
            if qk_rope_dim and qk_dim:
                return qk_rope_dim + qk_dim
    if hasattr(self.hf_text_config,
                "model_type") and (self.hf_text_config.model_type
                                    == "zamba2"):
        return self.hf_text_config.attention_head_dim

    if self.is_attention_free:
        return 0

    # NOTE: Some configs may set head_dim=None in the config
    if getattr(self.hf_text_config, "head_dim", None) is not None:
        return self.hf_text_config.head_dim

    # FIXME(woosuk): This may not be true for all models.
    return (self.hf_text_config.hidden_size //
            self.hf_text_config.num_attention_heads)


ModelConfig._verify_with_expert_parallelism = _verify_with_expert_parallelism
ModelConfig.is_deepseek_mla = is_deepseek_mla
ModelConfig.get_head_size = get_head_size
