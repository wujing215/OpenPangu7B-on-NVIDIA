#!/bin/bash
# 应用 Medusa 第三方库的兼容性补丁
# 用法: ./apply_patches.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEDUSA_DIR="$SCRIPT_DIR/third_party/Medusa"
PATCHES_DIR="$SCRIPT_DIR/patches"

echo "Applying patches to Medusa..."

# 检查 Medusa 目录是否存在
if [ ! -d "$MEDUSA_DIR" ]; then
    echo "Error: Medusa directory not found at $MEDUSA_DIR"
    exit 1
fi

# 应用补丁
cd "$MEDUSA_DIR"

# 方法 1: 使用 patch 命令（如果补丁格式正确）
# patch -p1 < "$PATCHES_DIR/medusa_transformers_compat.patch"

# 方法 2: 直接修改文件（更可靠）
echo "Patching modeling_llama_kv.py..."
LLAMA_FILE="$MEDUSA_DIR/medusa/model/modeling_llama_kv.py"
if grep -q "from transformers.utils import is_flash_attn_available" "$LLAMA_FILE" && ! grep -q "MODIFIED.*flash attention" "$LLAMA_FILE"; then
    sed -i 's/from transformers.utils import is_flash_attn_available/# [MODIFIED] Handle different transformers versions for flash attention check\ntry:\n    from transformers.utils import is_flash_attn_available\nexcept ImportError:\n    try:\n        from transformers.utils.import_utils import is_flash_attn_available\n    except ImportError:\n        def is_flash_attn_available():\n            return False/g' "$LLAMA_FILE"
    echo "  ✓ Patched modeling_llama_kv.py"
else
    echo "  - modeling_llama_kv.py already patched or has different structure"
fi

echo "Patching modeling_mistral_kv.py..."
MISTRAL_FILE="$MEDUSA_DIR/medusa/model/modeling_mistral_kv.py"
if grep -q "from transformers.utils import is_flash_attn_available" "$MISTRAL_FILE" && ! grep -q "MODIFIED.*flash attention" "$MISTRAL_FILE"; then
    sed -i 's/from transformers.utils import is_flash_attn_available/# [MODIFIED] Handle different transformers versions for flash attention check\ntry:\n    from transformers.utils import is_flash_attn_available\nexcept ImportError:\n    try:\n        from transformers.utils.import_utils import is_flash_attn_available\n    except ImportError:\n        def is_flash_attn_available():\n            return False/g' "$MISTRAL_FILE"
    echo "  ✓ Patched modeling_mistral_kv.py"
else
    echo "  - modeling_mistral_kv.py already patched or has different structure"
fi

echo "Done!"
