# -*- coding: utf-8 -*-
"""
从GSM8K数据集提取100条样本用于GRPO微调
功能：从parquet文件中读取数据，提取100条样本，生成JSON格式的训练数据
"""

import json
import os
import pandas as pd
from pathlib import Path

print("=" * 70)
print("GSM8K 数据集样本提取工具")
print("=" * 70)

# ========================================
# Step 1: 定位数据集文件
# ========================================

print("\nStep 1: 定位数据集文件")
print("-" * 70)

dataset_dir = r"D:\AI CASES\30-LLM模型蒸馏与微调实操\【数据集】gsm8k"
parquet_file = os.path.join(dataset_dir, "main", "train-00000-of-00001.parquet")

print(f"数据集目录: {dataset_dir}")
print(f"Parquet文件: {parquet_file}")

if not os.path.exists(parquet_file):
    print(f"✗ 文件不存在: {parquet_file}")
    print("尝试查找其他parquet文件...")
    
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith('.parquet'):
                parquet_file = os.path.join(root, file)
                print(f"✓ 找到: {parquet_file}")
                break

if not os.path.exists(parquet_file):
    print("✗ 未找到parquet文件")
    exit(1)

print(f"✓ 文件已定位")


# ========================================
# Step 2: 读取数据
# ========================================

print("\nStep 2: 读取Parquet数据")
print("-" * 70)

try:
    df = pd.read_parquet(parquet_file)
    print(f"✓ 数据读取成功")
    print(f"  - 总样本数: {len(df)}")
    print(f"  - 列名: {list(df.columns)}")
    print(f"  - 数据类型: {df.dtypes.to_dict()}")
except Exception as e:
    print(f"✗ 读取失败: {e}")
    exit(1)


# ========================================
# Step 3: 提取100条样本
# ========================================

print("\nStep 3: 提取100条样本")
print("-" * 70)

# 随机采样100条
sample_size = min(100, len(df))
df_sample = df.sample(n=sample_size, random_state=42)

print(f"✓ 已提取 {len(df_sample)} 条样本")
print(f"  - 采样方法: 随机采样（seed=42）")


# ========================================
# Step 4: 数据格式转换
# ========================================

print("\nStep 4: 数据格式转换")
print("-" * 70)

def extract_answer_number(answer_str):
    """从答案字符串中提取最终数字答案"""
    if not answer_str:
        return None
    
    # 查找#### 标记后的答案
    if "####" in answer_str:
        answer_part = answer_str.split("####")[-1].strip()
        return answer_part
    
    return None


# 转换为GRPO训练格式
training_data = []

for idx, row in df_sample.iterrows():
    question = row.get('question', '')
    answer = row.get('answer', '')
    
    # 提取最终答案
    final_answer = extract_answer_number(answer)
    
    # 创建训练样本
    sample = {
        "id": len(training_data) + 1,
        "question": question,
        "full_answer": answer,
        "final_answer": final_answer,
        "prompt": [
            {
                "role": "system",
                "content": "Respond in the following format:\n<reasoning>\n...\n</reasoning>\n<answer>\n...\n</answer>"
            },
            {
                "role": "user",
                "content": question
            }
        ]
    }
    
    training_data.append(sample)

print(f"✓ 数据格式转换完成")
print(f"  - 样本数: {len(training_data)}")


# ========================================
# Step 5: 保存为JSON文件
# ========================================

print("\nStep 5: 保存数据文件")
print("-" * 70)

output_dir = r"D:\AI CASES\30-LLM模型蒸馏与微调实操"
output_file = os.path.join(output_dir, "gsm8k_100_samples.json")

try:
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ JSON文件保存成功")
    print(f"  - 文件路径: {output_file}")
    print(f"  - 文件大小: {os.path.getsize(output_file) / 1024:.2f} KB")
except Exception as e:
    print(f"✗ 保存失败: {e}")
    exit(1)


# ========================================
# Step 6: 保存为CSV文件（便于查看）
# ========================================

print("\nStep 6: 保存CSV文件（便于查看）")
print("-" * 70)

csv_file = os.path.join(output_dir, "gsm8k_100_samples.csv")

try:
    # 创建简化版本用于CSV
    csv_data = []
    for sample in training_data:
        csv_data.append({
            "ID": sample["id"],
            "Question": sample["question"],
            "Final_Answer": sample["final_answer"],
            "Full_Answer_Preview": sample["full_answer"][:100] + "..." if len(sample["full_answer"]) > 100 else sample["full_answer"]
        })
    
    df_csv = pd.DataFrame(csv_data)
    df_csv.to_csv(csv_file, index=False, encoding='utf-8')
    
    print(f"✓ CSV文件保存成功")
    print(f"  - 文件路径: {csv_file}")
except Exception as e:
    print(f"✗ CSV保存失败: {e}")


# ========================================
# Step 7: 数据统计与预览
# ========================================

print("\nStep 7: 数据统计与预览")
print("-" * 70)

# 统计信息
question_lengths = [len(s["question"]) for s in training_data]
answer_lengths = [len(s["full_answer"]) for s in training_data]

print(f"问题长度统计:")
print(f"  - 平均长度: {sum(question_lengths) / len(question_lengths):.0f} 字符")
print(f"  - 最短: {min(question_lengths)} 字符")
print(f"  - 最长: {max(question_lengths)} 字符")

print(f"\n答案长度统计:")
print(f"  - 平均长度: {sum(answer_lengths) / len(answer_lengths):.0f} 字符")
print(f"  - 最短: {min(answer_lengths)} 字符")
print(f"  - 最长: {max(answer_lengths)} 字符")

# 显示前3个样本
print(f"\n前3个样本预览:")
print("-" * 70)

for i in range(min(3, len(training_data))):
    sample = training_data[i]
    print(f"\n样本 {i+1}:")
    print(f"  问题: {sample['question'][:80]}...")
    print(f"  最终答案: {sample['final_answer']}")
    print(f"  完整答案预览: {sample['full_answer'][:100]}...")


# ========================================
# Step 8: 生成使用说明
# ========================================

print("\n" + "=" * 70)
print("数据提取完成！")
print("=" * 70)

print("\n生成的文件:")
print(f"  1. {output_file}")
print(f"     - JSON格式，包含完整的prompt结构")
print(f"     - 用于GRPO微调程序")
print(f"\n  2. {csv_file}")
print(f"     - CSV格式，便于Excel查看")
print(f"     - 用于数据审查和统计")

print("\n在GRPO微调程序中使用:")
print("-" * 70)
print("""
修改 Qwen3_5_(9B)_GRPO_Lightweight.py 中的数据加载部分：

原代码：
    dataset = get_gsm8k_questions(num_samples=100)

修改为：
    import json
    with open('gsm8k_100_samples.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    dataset = Dataset.from_dict({
        'prompt': [item['prompt'] for item in data],
        'answer': [item['final_answer'] for item in data]
    })
""")

print("\n✓ 所有操作完成！")



