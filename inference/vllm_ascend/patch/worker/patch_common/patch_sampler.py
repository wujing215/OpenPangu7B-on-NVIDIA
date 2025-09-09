#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
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

from typing import Optional

import torch
import torch_npu
from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler, random_sample
from vllm.v1.sample.sampler import Sampler, _SAMPLING_EPS
from vllm.v1.sample.metadata import SamplingMetadata
from vllm_ascend import envs


def apply_top_k_top_p(
    logits: torch.Tensor,
    k: torch.Tensor,
    p: torch.Tensor,
) -> torch.Tensor:
    if p is not None and k is not None:
        # npu_top_k_top_p's parameter order is (logits, p, k), not (logits, k, p)
        return torch_npu.npu_top_k_top_p(logits, p, k)

    probs = logits.softmax(dim=-1)
    probs_sort, _ = probs.sort(dim=-1, descending=False)

    if k is not None:
        top_k_count = probs_sort.size(1) - k.to(torch.long)  # shape: (batch, )
        top_k_count = top_k_count.unsqueeze(dim=1)
        top_k_cutoff = probs_sort.gather(-1, top_k_count)

        # Make sure the no top-k rows are no-op.
        no_top_k_mask = (k == logits.shape[1]).unsqueeze(dim=1)
        top_k_cutoff.masked_fill_(no_top_k_mask, -float("inf"))

        elements_to_discard = probs < top_k_cutoff
        logits.masked_fill_(elements_to_discard, -float("inf"))

    if p is not None:
        cumprob = torch.cumsum(probs_sort, dim=-1)
        top_p_mask = cumprob <= 1 - p.unsqueeze(dim=1)
        top_p_mask[:, -1] = False  # at least one

        top_p_count = top_p_mask.sum(dim=-1).unsqueeze(1)
        top_p_cutoff = probs_sort.gather(-1, top_p_count)
        elements_to_discard = probs < top_p_cutoff
        logits.masked_fill_(elements_to_discard, -float("inf"))

    return logits


def topk_topp_forward_native(
    self,
    logits: torch.Tensor,
    generators: dict[int, torch.Generator],
    k: Optional[torch.Tensor],
    p: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    PyTorch-native implementation of top-k and top-p sampling.

    The logits tensor may be updated in-place.
    """
    logits = apply_top_k_top_p(logits, k, p)
    probs = logits.softmax(dim=-1, dtype=torch.float32)
    return random_sample(probs, generators)


def apply_top_n_sigma(
    logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
):
    if sampling_metadata.no_top_n_sigma:
        return logits

    top_n_sigma = sampling_metadata.top_n_sigma[:, None]
    top_n_sigma_mask = (top_n_sigma != -1)
    filter_value = -3.4028e+38
    max_vals, _ = logits.max(dim=-1, keepdim=True)
    std_vals = logits.std(dim=-1, keepdim=True)
    threshold = max_vals - top_n_sigma * std_vals
    threshold[~top_n_sigma_mask] = filter_value
    mask = (logits < threshold)
    logits = torch.where(mask, filter_value, logits)
    return logits


def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    """Sample logits based on sampling metadata.

    The various logits processing functions called in this method
    may update the logits tensor in-place.
    """

    assert not (sampling_metadata.all_greedy
                and sampling_metadata.all_random)
    if sampling_metadata.all_random:
        greedy_sampled = None
    else:
        greedy_sampled = self.greedy_sample(logits)
        if sampling_metadata.all_greedy:
            return greedy_sampled

    assert sampling_metadata.temperature is not None

    # Apply temperature.
    logits = self.apply_temperature(logits, sampling_metadata.temperature)

    # Apply logits processors that only apply to random sampling
    # (argmax invariant)
    for processor in sampling_metadata.logitsprocs.argmax_invariant:
        logits = processor.apply(logits)

    # Apply top_n_sigma
    logits = apply_top_n_sigma(logits, sampling_metadata)

    # Apply top_k and/or top_p.
    random_sampled = self.topk_topp_sampler(
        logits,
        sampling_metadata.generators,
        sampling_metadata.top_k,
        sampling_metadata.top_p,
    )

    if greedy_sampled is None:
        return random_sampled

    sampled = torch.where(
        sampling_metadata.temperature < _SAMPLING_EPS,
        greedy_sampled,
        random_sampled,
        out=greedy_sampled,  # Reuse tensor
    )
    return sampled


if envs.VLLM_ASCEND_ENABLE_TOPK_TOPP_OPTIMIZATION:
    TopKTopPSampler.forward_native = topk_topp_forward_native

if envs.VLLM_ASCEND_ENABLE_TOP_N_SIGMA:
    Sampler.sample = sample