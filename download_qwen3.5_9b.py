# -*- coding: utf-8 -*-
"""
从ModelScope下载Qwen3.5-9B模型到AutoDL
功能：下载Qwen3.5-9B模型到/autodl-tmp/MODELS/目录
环境：AutoDL环境
"""

import os
import sys
import subprocess

print("=" * 70)
print("Qwen3.5-9B 模型下载工具")
print("=" * 70)

# ========================================
# Step 1: 检查目标目录
# ========================================

print("\nStep 1: 检查目标目录")
print("-" * 70)

target_dir = "/root/autodl-tmp/MODELS"
model_name = "Qwen3.5-9B"
model_path = os.path.join(target_dir, model_name)

print(f"目标目录: {target_dir}")
print(f"模型保存路径: {model_path}")

# 创建目录
os.makedirs(target_dir, exist_ok=True)
print(f"✓ 目标目录已准备")


# ========================================
# Step 2: 安装依赖
# ========================================

print("\nStep 2: 安装必要的依赖库")
print("-" * 70)

# 先升级pip
print("升级pip...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "-i", "https://pypi.tsinghua.edu.cn/simple"])

# 安装modelscope
print("安装modelscope...")
try:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", 
        "modelscope", 
        "-i", "https://pypi.tsinghua.edu.cn/simple"
    ])
    print("✓ modelscope安装成功")
except subprocess.CalledProcessError as e:
    print(f"✗ modelscope安装失败: {e}")
    print("尝试使用官方源...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", 
        "modelscope"
    ])

# 验证安装
print("\n验证依赖安装...")
try:
    from modelscope import snapshot_download
    print("✓ modelscope导入成功")
except ImportError as e:
    print(f"✗ modelscope导入失败: {e}")
    sys.exit(1)


# ========================================
# Step 3: 下载模型
# ========================================

print("\nStep 3: 从ModelScope下载Qwen3.5-9B模型")
print("-" * 70)

model_id = "Qwen/Qwen3.5-9B"
print(f"模型ID: {model_id}")
print(f"下载目标: {model_path}")
print("\n开始下载（这可能需要10-30分钟，请耐心等待）...")
print("-" * 70)

try:
    model_dir = snapshot_download(
        model_id=model_id,
        cache_dir=target_dir,
        revision="master",
    )
    print("-" * 70)
    print(f"✓ 模型下载完成")
    print(f"  实际保存路径: {model_dir}")
    
except Exception as e:
    print(f"✗ 下载失败: {e}")
    print("\n尝试备用方案：使用git clone下载...")
    print("-" * 70)
    
    try:
        os.chdir(target_dir)
        os.system("git clone https://www.modelscope.cn/Qwen/Qwen3.5-9B.git")
        print("✓ 备用下载完成")
    except Exception as e2:
        print(f"✗ 备用方案也失败: {e2}")
        sys.exit(1)


# ========================================
# Step 4: 验证模型
# ========================================

print("\nStep 4: 验证模型文件")
print("-" * 70)

# 查找模型目录
model_dirs = []
for root, dirs, files in os.walk(target_dir):
    if any(f.endswith('.safetensors') or f.endswith('.bin') for f in files):
        model_dirs.append(root)

if model_dirs:
    print(f"✓ 找到 {len(model_dirs)} 个模型目录")
    for model_dir in model_dirs:
        print(f"  - {model_dir}")
        
        # 列出关键文件
        files = os.listdir(model_dir)
        key_files = [f for f in files if f.endswith(('.safetensors', '.bin', '.json', '.py'))]
        print(f"    关键文件数: {len(key_files)}")
        for f in key_files[:5]:
            file_path = os.path.join(model_dir, f)
            file_size = os.path.getsize(file_path) / (1024**3)
            print(f"      - {f} ({file_size:.2f} GB)")
        if len(key_files) > 5:
            print(f"      ... 还有 {len(key_files) - 5} 个文件")
else:
    print("⚠ 未找到模型文件，请检查下载是否成功")


# ========================================
# Step 5: 显示使用说明
# ========================================

print("\n" + "=" * 70)
print("下载完成！")
print("=" * 70)

print("\n在GRPO微调程序中使用模型的方法：")
print("-" * 70)
print("""
在 Qwen3_5_(9B)_GRPO_Lightweight.py 中修改模型路径：

原代码：
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen2.5-9B-Instruct",
        ...
    )

修改为（根据实际下载位置调整）：
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="/autodl-tmp/MODELS/Qwen/Qwen3.5-9B",
        ...
    )
""")

print("\n下载工具执行完成！")
