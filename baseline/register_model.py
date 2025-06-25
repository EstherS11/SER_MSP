# ============================================================================
# 文件4: register_model.py - 模型注册脚本
# ============================================================================

#!/usr/bin/env python3
"""
不修改ESP-net源码的模型注册方案
"""

import sys
import os
from pathlib import Path

# 添加当前目录到Python路径
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

# 导入我们的模型
from espnet_ser_model import WavLMECAPAModel

def register_wavlm_ecapa_model():
    """动态注册WavLM-ECAPA模型到ESP-net"""
    
    try:
        from espnet2.tasks.ser import ser_model_choices
        
        # 检查是否已经注册
        if "wavlm_ecapa" in ser_model_choices.choices:
            print("✅ WavLM-ECAPA model already registered")
            return True
        
        # 动态添加到选择字典
        ser_model_choices.choices["wavlm_ecapa"] = WavLMECAPAModel
        
        print("✅ WavLM-ECAPA model registered successfully")
        print(f"Available models: {list(ser_model_choices.choices.keys())}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import ESP-net SER task: {e}")
        print("Please make sure ESP-net is properly installed")
        return False
    except Exception as e:
        print(f"❌ Failed to register model: {e}")
        return False

def test_model_creation():
    """测试模型创建"""
    try:
        model = WavLMECAPAModel(num_class=10)
        print("✅ Model creation test passed")
        print(f"Model type: {type(model)}")
        return True
    except Exception as e:
        print(f"❌ Model creation test failed: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Registering WavLM + ECAPA-TDNN model...")
    
    # 测试模型创建
    if not test_model_creation():
        sys.exit(1)
    
    # 注册模型
    if not register_wavlm_ecapa_model():
        sys.exit(1)
    
    print("🎉 Model registration completed!")
