# -*- coding: utf-8 -*-
"""
Qwen3.5-9B GRPO强化学习训练 - AutoDL完整版本
课程：LLM模型蒸馏与微调实操
功能：使用GRPO对Qwen3.5-9B进行推理能力微调（100样本 x 50步验证）
环境：AutoDL RTX 5090(32GB)，显存优化配置
"""

import re
import torch
import json
from datasets import load_dataset, Dataset
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer

# ========================================
# Step 1: 模型加载（显存优化）
# ========================================

print("=" * 70)
print("Step 1: 加载Qwen3.5-9B模型（4bit量化 + LoRA）")
print("=" * 70)

max_seq_length = 512  # 降低至512以节省显存
lora_rank = 16  # 降低至16（原程序32）

# 使用本地模型路径
model_path = "/autodl-tmp/MODELS/Qwen/Qwen3.5-9B"

print(f"模型路径: {model_path}")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_path,
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    fast_inference=True,
    max_lora_rank=lora_rank,
    gpu_memory_utilization=0.7,
)

print(f"✓ 模型加载完成")
print(f"  - 模型: Qwen3.5-9B")
print(f"  - 最大序列长度: {max_seq_length}")
print(f"  - LoRA Rank: {lora_rank}")


# ========================================
# Step 2: LoRA配置
# ========================================

print("\n" + "=" * 70)
print("Step 2: 配置LoRA适配器")
print("=" * 70)

model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=lora_rank,
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

print("✓ LoRA配置完成")


# ========================================
# Step 3: GSM8K数据准备（100样本）
# ========================================

print("\n" + "=" * 70)
print("Step 3: 加载GSM8K数据集（100条样本）")
print("=" * 70)

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""


def extract_xml_answer(text: str) -> str:
    """从XML格式文本中提取答案"""
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def extract_hash_answer(text: str) -> str | None:
    """从####标记文本中提取答案"""
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def load_gsm8k_from_json(json_file_path, num_samples=100):
    """从JSON文件加载GSM8K数据"""
    print(f"从JSON文件加载数据: {json_file_path}")
    
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"✓ 加载了 {len(data)} 条样本")
        
        # 限制样本数
        if len(data) > num_samples:
            data = data[:num_samples]
            print(f"✓ 限制为 {num_samples} 条样本")
        
        # 转换为Dataset格式
        dataset = Dataset.from_dict({
            'prompt': [item['prompt'] for item in data],
            'answer': [item['final_answer'] for item in data]
        })
        
        return dataset
    
    except FileNotFoundError:
        print(f"✗ 文件不存在: {json_file_path}")
        print("尝试使用在线GSM8K数据集...")
        
        # 备用方案：使用在线数据集
        data = load_dataset('gsm8k', 'main')['train']
        
        if len(data) > num_samples:
            indices = torch.randperm(len(data))[:num_samples].tolist()
            data = data.select(indices)
        
        data = data.map(lambda x: {
            'prompt': [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': x['question']}
            ],
            'answer': extract_hash_answer(x['answer'])
        })
        
        return data


# 尝试从本地JSON加载，如果不存在则使用在线数据集
json_file = "./autodl-tmp/gsm8k_100_samples.json"  # AutoDL上传的文件路径
dataset = load_gsm8k_from_json(json_file, num_samples=100)

print(f"✓ 数据集加载完成")
print(f"  - 样本数: {len(dataset)}")


# ========================================
# Step 4: 奖励函数定义
# ========================================

print("\n" + "=" * 70)
print("Step 4: 定义奖励函数")
print("=" * 70)


def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """正确性奖励：检查答案是否正确（权重最高）"""
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    rewards = [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]
    return rewards


def int_reward_func(completions, **kwargs) -> list[float]:
    """整数奖励：检查答案是否为整数"""
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]


def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """严格格式奖励：完全符合XML格式"""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """宽松格式奖励：基本符合XML格式"""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.search(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def count_xml(text) -> float:
    """计算XML标签完整性得分"""
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count


def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    """XML标签计数奖励"""
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]


print("✓ 5个奖励函数定义完成")
print("  - correctness_reward_func (权重: 2.0)")
print("  - int_reward_func (权重: 0.5)")
print("  - strict_format_reward_func (权重: 0.5)")
print("  - soft_format_reward_func (权重: 0.5)")
print("  - xmlcount_reward_func (权重: 0.5)")


# ========================================
# Step 5: GRPOTrainer配置与训练
# ========================================

print("\n" + "=" * 70)
print("Step 5: 配置GRPOTrainer并开始训练")
print("=" * 70)

max_prompt_length = 256

training_args = GRPOConfig(
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",
    logging_steps=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_generations=4,  # 降低至4（原程序6）
    max_prompt_length=max_prompt_length,
    max_completion_length=max_seq_length - max_prompt_length,
    max_steps=50,  # 50步验证
    save_steps=50,
    max_grad_norm=0.1,
    report_to="none",
    output_dir="outputs_qwen35_9b",
)

print("✓ 训练配置完成")
print(f"  - 训练步数: 50")
print(f"  - 每个问题生成: 4个候选答案")
print(f"  - Batch Size: 1")
print(f"  - 学习率: 5e-6")

# 显存信息
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"\n  GPU信息:")
print(f"  - GPU型号: {gpu_stats.name}")
print(f"  - 总显存: {max_memory} GB")
print(f"  - 已使用: {start_gpu_memory} GB")

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func,
    ],
    args=training_args,
    train_dataset=dataset,
)

print("\n开始训练...")
print("-" * 70)
trainer_stats = trainer.train()
print("-" * 70)

# 训练统计
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)

print("\n✓ 训练完成")
print(f"  - 训练用时: {trainer_stats.metrics['train_runtime']:.2f} 秒")
print(f"  - 训练用时: {trainer_stats.metrics['train_runtime']/60:.2f} 分钟")
print(f"  - 峰值显存: {used_memory} GB")
print(f"  - LoRA训练显存: {used_memory_for_lora} GB")
print(f"  - 显存使用率: {used_percentage}%")


# ========================================
# Step 6: 权重保存
# ========================================

print("\n" + "=" * 70)
print("Step 6: 保存LoRA权重")
print("=" * 70)

model.save_lora("grpo_qwen35_9b_lora")
tokenizer.save_pretrained("grpo_qwen35_9b_lora")

print("✓ 权重保存完成")
print("  - 保存路径: grpo_qwen35_9b_lora/")


# ========================================
# Step 7: 推理测试
# ========================================

print("\n" + "=" * 70)
print("Step 7: 推理测试")
print("=" * 70)

from transformers import TextStreamer

# 测试用例
test_questions = [
    "If a baker makes 3 cakes per hour and works for 8 hours, how many cakes does the baker make?",
    "A store sells apples for $2 each. If you buy 5 apples, how much do you spend?",
]

text_streamer = TextStreamer(tokenizer, skip_prompt=True)

for i, question in enumerate(test_questions, 1):
    print(f"\n--- 测试用例 {i} ---")
    print(f"问题: {question}")
    print(f"回答:")
    
    text = tokenizer.apply_chat_template([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ], tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    
    _ = model.generate(
        **inputs,
        streamer=text_streamer,
        max_new_tokens=256,
        temperature=0.8,
        top_p=0.95,
    )

print("\n" + "=" * 70)
print("✓ 全链路验证完成！")
print("=" * 70)
print("\n总结:")
print("  ✓ 数据加载: 100条GSM8K样本")
print("  ✓ 模型训练: 50步GRPO训练")
print("  ✓ 奖励函数: 5个多维度评估函数")
print("  ✓ 权重保存: LoRA适配器已保存")
print("  ✓ 推理测试: 模型推理验证完成")
print("\n程序名称: Qwen3_5_(9B)_GRPO_AutoDL.py")
print("保存位置: /root/")



