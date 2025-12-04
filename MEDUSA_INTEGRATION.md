# OpenPangu + Medusa 集成说明

## 概述

本项目将 [Medusa](https://github.com/FasterDecoding/Medusa) 投机解码技术集成到 OpenPangu-Embedded-7B 模型中，实现更快的推理速度。

## 文件修改清单

### 1. 新增文件

| 文件 | 说明 |
|------|------|
| `medusa_model.py` | Medusa 模型封装，支持 Pangu 模型 |
| `medusa_choices.py` | Medusa 树配置，包含 `pangu_stage2` (3 heads) |
| `medusa_compat.py` | 兼容层，处理 transformers 版本兼容性 |
| `train_medusa.py` | Medusa heads 训练脚本 |
| `inference/medusa_generate.py` | Medusa 推理脚本 |
| `patches/medusa_transformers_compat.patch` | 第三方库补丁 |
| `apply_patches.sh` | 补丁应用脚本 |

### 2. 修改的文件

#### `modeling_openpangu_dense.py`
- 支持 Medusa KV Cache (list/tuple 格式)
- 支持自定义 `position_ids` 和 `cache_position`
- 支持 `medusa_mask` 自定义 attention mask

### 3. 第三方库修改 (third_party/Medusa)

#### `medusa/model/modeling_llama_kv.py` 和 `medusa/model/modeling_mistral_kv.py`
- 修复 `is_flash_attn_available` 导入兼容性问题

**注意**：这些修改可能在更新第三方库时丢失，请参见下方的解决方案。

## 第三方库更新后的处理

### 方案 A：使用兼容层（推荐）

`medusa_compat.py` 会在导入 Medusa 之前自动修补 transformers 兼容性问题。只需确保：

```python
import medusa_compat  # 在导入 medusa 之前
from medusa.model.xxx import xxx
```

### 方案 B：重新应用补丁

如果更新了 `third_party/Medusa`，运行：

```bash
chmod +x apply_patches.sh
./apply_patches.sh
```

### 方案 C：手动修改

在 `modeling_llama_kv.py` 和 `modeling_mistral_kv.py` 中，将：

```python
from transformers.utils import is_flash_attn_available
```

替换为：

```python
try:
    from transformers.utils import is_flash_attn_available
except ImportError:
    try:
        from transformers.utils.import_utils import is_flash_attn_available
    except ImportError:
        def is_flash_attn_available():
            return False
```

## 使用方法

### 训练 Medusa Heads

```bash
python train_medusa.py \
    --model_name_or_path /path/to/openPangu-Embedded-7B-V1.1 \
    --data_path /path/to/training_data.json \
    --output_dir ./medusa_output \
    --medusa_num_heads 3 \
    --medusa_num_layers 1
```

### 推理

```bash
cd inference
python medusa_generate.py --prompt "你的问题"

# 交互模式
python medusa_generate.py --interactive

# 流式输出
python medusa_generate.py --prompt "你的问题" --stream
```

## 技术细节

### Medusa 投机解码原理

1. **多头预测**：每个 Medusa head 预测未来不同位置的 token
2. **树形候选**：基于预测构建候选 token 树
3. **批量验证**：基础模型一次性验证所有候选
4. **接受/拒绝**：根据后验概率接受正确预测

### `pangu_stage2` 配置

- 56 条路径，最大深度 3（适配 3 个 Medusa heads）
- 从 `vicuna_7b_stage2` 筛选得到

## 依赖

- PyTorch >= 2.0
- Transformers >= 4.40
- accelerate
- safetensors

## 已知问题

1. 需要足够的 GPU 显存（7B 模型约需 16GB）
2. 第一次加载会显示 "newly initialized" 警告（正常）
