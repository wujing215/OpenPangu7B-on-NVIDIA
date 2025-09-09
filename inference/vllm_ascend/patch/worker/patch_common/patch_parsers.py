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


from vllm.entrypoints.openai import tool_parsers
from vllm_ascend.entrypoints.openai.tool_parsers import PanguToolParser
tool_parsers.__all__.append("PanguToolParser")


from vllm import reasoning
from vllm_ascend.entrypoints.openai.reasoning_parsers import PanguReasoningParser
reasoning.__all__.append("PanguReasoningParser")