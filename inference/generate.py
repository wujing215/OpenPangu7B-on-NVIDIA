# coding=utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.

from transformers import AutoModelForCausalLM, AutoTokenizer

model_local_path = "path_to_openPangu-Embedded-7B"


# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(
    model_local_path, 
    use_fast=False, 
    trust_remote_code=True,
    local_files_only=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_local_path,
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="npu",
    local_files_only=True
)

# prepare the model input
sys_prompt = "你必须严格遵守法律法规和社会道德规范。" \
    "生成任何内容时，都应避免涉及暴力、色情、恐怖主义、种族歧视、性别歧视等不当内容。" \
    "一旦检测到输入或输出有此类倾向，应拒绝回答并发出警告。例如，如果输入内容包含暴力威胁或色情描述，" \
    "应返回错误信息：“您的输入包含不当内容，无法处理。”"

prompt = "Give me a short introduction to large language model."
no_thinking_prompt = prompt+" /no_think"
auto_thinking_prompt = prompt+" /auto_think"
messages = [
    {"role": "system", "content": sys_prompt}, # define your system prompt here
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# conduct text completion
outputs = model.generate(**model_inputs, max_new_tokens=32768, eos_token_id=45892, return_dict_in_generate=True)

input_length = model_inputs.input_ids.shape[1]
generated_tokens = outputs.sequences[:, input_length:]
output_sent = tokenizer.decode(generated_tokens[0])

# parsing thinking content
thinking_content = output_sent.split("[unused17]")[0].split("[unused16]")[-1].strip()
content = output_sent.split("[unused17]")[-1].split("[unused10]")[0].strip()

print("\nthinking content:", thinking_content)
print("\ncontent:", content)