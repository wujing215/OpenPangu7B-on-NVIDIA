"""
Medusa 兼容性模块

这个模块在导入 Medusa 之前应用必要的补丁，以确保与新版 transformers 兼容。
使用方法：在导入 medusa 之前先导入此模块

    import medusa_compat  # 先导入此模块
    from medusa.model.medusa_model import MedusaModel  # 然后正常使用 medusa
"""

import sys
import importlib

def patch_flash_attn_import():
    """
    修补 transformers 的 is_flash_attn_available 导入问题
    新版 transformers 将这个函数移到了 transformers.utils.import_utils
    """
    try:
        from transformers.utils import is_flash_attn_available
    except ImportError:
        try:
            from transformers.utils.import_utils import is_flash_attn_available
        except ImportError:
            def is_flash_attn_available():
                return False
        
        # 将函数注入到 transformers.utils 模块
        import transformers.utils
        transformers.utils.is_flash_attn_available = is_flash_attn_available

def setup_medusa_path(medusa_path: str = None):
    """
    设置 Medusa 的 Python 路径
    
    Args:
        medusa_path: Medusa 库的路径，如果为 None 则自动检测
    """
    import os
    
    if medusa_path is None:
        # 自动检测：假设此文件在 openPangu 目录下
        current_dir = os.path.dirname(os.path.abspath(__file__))
        medusa_path = os.path.join(current_dir, "third_party", "Medusa")
    
    if medusa_path not in sys.path:
        sys.path.insert(0, medusa_path)
    
    return medusa_path

def init_medusa_compat(medusa_path: str = None):
    """
    初始化 Medusa 兼容性设置
    
    Args:
        medusa_path: Medusa 库的路径
    
    Returns:
        Medusa 库的路径
    """
    # 1. 首先修补 flash attention 导入
    patch_flash_attn_import()
    
    # 2. 设置 Medusa 路径
    path = setup_medusa_path(medusa_path)
    
    return path

# 自动初始化
_medusa_path = init_medusa_compat()

print(f"[medusa_compat] Medusa compatibility layer initialized")
print(f"[medusa_compat] Medusa path: {_medusa_path}")
