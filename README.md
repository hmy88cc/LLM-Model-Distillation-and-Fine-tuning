# LLM模型蒸馏与微调实操

本项目提供了完整的LLM（大语言模型）蒸馏与微调实践代码，包括监督微调(SFT)、强化学习(GRPO)以及多模态模型微调等多种训练方法。

## 📋 项目概述

本项目基于Unsloth框架，实现了多种高效的LLM微调方案：
- **SFT监督微调**：使用Alpaca数据集对Qwen2.5-7B/Qwen3.5-9B进行指令微调
- **GRPO强化学习**：使用GSM8K数学数据集训练模型的推理能力
- **模型评估**：提供完整的模型性能对比评估工具

## 🚀 快速开始

### 环境要求

- Python 3.8+
- CUDA兼容的GPU（推荐A100/RTX 3090/RTX 5090等）
- 至少16GB显存（使用4bit量化可降低要求）

### 安装依赖

```bash
pip install -r requirements.txt
```

### 主要训练脚本详解

本项目包含5个核心训练脚本，针对不同场景和模型：

#### 1. Qwen2_5_(7B)_Alpaca.py - SFT监督微调
**功能**: 使用Alpaca指令数据集对Qwen2.5-7B进行监督微调(SFT)

**适用场景**:
- 让模型学习遵循指令的能力
- 通用对话和任务完成能力训练
- 快速验证LoRA微调流程

**技术特点**:
- 模型: Qwen2.5-7B-Instruct
- 数据集: Alpaca-cleaned (52K条指令数据)
- 方法: LoRA适配器 + 4bit量化
- 训练步数: 60步(示例配置)
- 显存需求: ~16GB (A100/RTX 3090)

**输出格式**: 标准Alpaca格式
```
### Instruction:
{指令}

### Input:
{输入}

### Response:
{回答}
```

---

#### 2. Qwen2_5_(7B)_R1_GRPO.py - GRPO强化学习(R1推理模型)
**功能**: 使用GRPO强化学习方法训练Qwen2.5-7B的数学推理能力

**适用场景**:
- 提升模型的逻辑推理和思维链(CoT)能力
- 训练类似DeepSeek-R1的推理模型
- 需要结构化推理输出的任务

**技术特点**:
- 模型: Qwen2.5-7B-Instruct
- 数据集: GSM8K数学应用题
- 方法: GRPO(Group Relative Policy Optimization)
- 推理加速: vLLM快速推理引擎
- 生成策略: 每个问题生成6个候选答案
- 训练步数: 250步
- 显存需求: ~24GB (建议A100/A800 40GB+)

**奖励函数** (5个维度):
1. `correctness_reward_func`: 答案正确性 (权重2.0)
2. `int_reward_func`: 答案是否为整数 (权重0.5)
3. `strict_format_reward_func`: 严格XML格式 (权重0.5)
4. `soft_format_reward_func`: 宽松XML格式 (权重0.5)
5. `xmlcount_reward_func`: XML标签完整性 (权重0.5)

**输出格式**: 结构化推理格式
```xml
<reasoning>
详细的逐步推理过程...
</reasoning>
<answer>
最终答案
</answer>
```

---

#### 3. Qwen3_5_(9B)_GRPO_AutoDL.py - Qwen3.5-9B完整训练版
**功能**: 针对AutoDL云平台优化的Qwen3.5-9B完整GRPO训练脚本

**适用场景**:
- 使用更新的Qwen3.5-9B模型
- AutoDL云平台部署 (RTX 5090 32GB)
- 生产环境的完整训练流程
- 需要详细训练监控和统计信息

**技术特点**:
- 模型: Qwen3.5-9B (比7B更强的基座模型)
- 数据集: GSM8K (100条样本用于验证)
- 优化: 显存优化配置，适配32GB显存
- 序列长度: 512 (降低以节省显存)
- LoRA秩: 16 (平衡效果与显存)
- 生成数量: 4个候选答案 (降低显存占用)
- 训练步数: 50步 (验证用)
- 特色: 完整的训练日志和GPU监控

**与Qwen2_5_(7B)_R1_GRPO.py的区别**:
- ✅ 使用更新的Qwen3.5-9B模型 (更强)
- ✅ 针对AutoDL环境优化路径配置
- ✅ 更详细的训练进度显示
- ✅ 显存使用更优化 (适合32GB显卡)
- ❌ 训练步数较少 (50步 vs 250步，可调整)

---

#### 4. Qwen3_5_(9B)_GRPO_Lightweight.py - 轻量化验证版
**功能**: Qwen3.5-9B,用于快速验证全链路

**适用场景**:
- 资源有限的环境
- 快速测试训练流程是否正常
- 调试和开发阶段
- 显存较小的GPU (<24GB)

**技术特点**:
- 模型: Qwen3.5-9B
- 数据集: GSM8K (100条样本)
- 优化策略:
  - 序列长度: 512 (较短)
  - LoRA秩: 16 (较小)
  - 生成数量: 4个 (较少)
  - 训练步数: 50步 (快速验证)
- 显存需求: ~20GB

**与Qwen3_5_(9B)_GRPO_AutoDL.py的区别**:
- ✅ 更轻量，启动更快
- ✅ 适合本地开发和测试
- ✅ 代码结构更简洁
- ❌ 不使用AutoDL特定路径
- ❌ 训练配置更保守

**选择建议**:
- 首次运行 → 用Lightweight版本验证环境
- 正式训练 → 用AutoDL版本获得更好效果

### 📊 训练脚本对比总结

| 脚本 | 模型 | 训练方法 | 数据类型 | 显存需求 | 适用场景 |
|------|------|---------|---------|---------|---------|
| Qwen2_5_(7B)_Alpaca.py | Qwen2.5-7B | SFT | 文本指令 | ~16GB | 通用指令跟随 |
| Qwen2_5_(7B)_R1_GRPO.py | Qwen2.5-7B | GRPO | 数学问题 | ~24GB | 推理能力训练 |
| Qwen3_5_(9B)_GRPO_AutoDL.py | Qwen3.5-9B | GRPO | 数学问题 | ~28GB | AutoDL完整训练 |
| Qwen3_5_(9B)_GRPO_Lightweight.py | Qwen3.5-9B | GRPO | 数学问题 | ~20GB | 快速验证测试 |

### 🔍 如何选择训练脚本?

**根据任务类型**:
- 📝 文本对话/指令遵循 → `Qwen2_5_(7B)_Alpaca.py`
- 🧮 数学推理/逻辑思维 → `Qwen2_5_(7B)_R1_GRPO.py` 或 Qwen3.5系列

**根据硬件条件**:
- 💾 显存 < 20GB → `Qwen3_5_(9B)_GRPO_Lightweight.py`
- 💾 显存 20-30GB → `Qwen3_5_(9B)_GRPO_AutoDL.py`
- 💾 显存 > 30GB → `Qwen2_5_(7B)_R1_GRPO.py` (可调大batch)

**根据训练阶段**:
- 🧪 测试环境 → `Qwen3_5_(9B)_GRPO_Lightweight.py`
- 🚀 生产训练 → `Qwen3_5_(9B)_GRPO_AutoDL.py`
- 📚 学习SFT → `Qwen2_5_(7B)_Alpaca.py`
- 🎓 学习GRPO → `Qwen2_5_(7B)_R1_GRPO.py`

## 📊 数据集

### 内置示例数据
- `gsm8k_100_samples.json` - GSM8K数学问题示例(100条)
  
### 大型数据集获取
由于数据集文件较大，建议通过以下方式获取：
1. 使用提供的`extract_gsm8k_full_samples.py`脚本从原始parquet文件提取
2. 从Hugging Face或ModelScope下载官方数据集

## 🛠️ 工具脚本详解

本项目包含4个辅助工具脚本，用于数据处理、模型下载和评估：

#### 1. model_comparison_eval.py - 模型对比评估工具
**功能**: 对比不同训练阶段模型的输出质量和性能

**适用场景**:
- 比较基座模型 vs SFT微调模型 vs GRPO模型的效果
- 量化评估模型改进程度
- A/B测试不同训练配置
- 生成模型性能报告

**评估维度**:
1. **格式遵循能力** (`format_compliance`)
   - 检查是否遵循XML格式或医疗格式
   - 评分: 0-1分

2. **答案准确率** (`answer_accuracy`)
   - 基于F1分数计算与参考答案的相似度
   - 适用于有标准答案的场景

3. **推理链质量** (`reasoning_quality`)
   - 推理过程长度评分 (30%)
   - 步骤性评分：是否分步推理 (40%)
   - 结论性评分：是否有总结 (30%)
   - 仅适用于GRPO结构化输出

4. **回复长度合理性** (`response_length_score`)
   - 太短或太长都会扣分
   - 合理范围: 20-2000字符

5. **语言匹配度** (`language_match`)
   - 检查回复语言是否与期望一致
   - 中文/英文比例统计

**使用示例**:
```python
# 内置演示模式（使用模拟数据）
python model_comparison_eval.py

# 实际使用时，替换为真实模型推理结果
test_cases = [
    {"question": "问题...", "reference": "参考答案..."}
]
model_results = {
    "基座模型": ["回答1", "回答2"],
    "SFT模型": ["回答1", "回答2"],
    "GRPO模型": ["回答1", "回答2"]
}
compare_models(test_cases, model_results, eval_type="medical")
```

**输出示例**:
```
===== 模型对比评估报告 =====

--- 测试用例 1 ---
问题: 我最近总是感觉头晕，应该怎么办？

  [基座模型(微调前)]
    回复: I can provide general health advice...
    综合评分: 0.3250
    format_score: 0.0
    language_match: 0.0

  [SFT微调模型]
    回复: 头晕的原因很多，可能与低血糖...
    综合评分: 0.8500
    format_score: 1.0
    language_match: 1.0

总体评估汇总:
  基座模型(微调前): 平均综合评分 = 0.3250
  SFT微调模型: 平均综合评分 = 0.8500
  GRPO强化学习模型: 平均综合评分 = 0.9200
```

#### 2. download_qwen35_9b.py - 模型下载工具
**功能**: 从ModelScope自动下载Qwen3.5-9B模型到本地

**适用场景**:
- 首次搭建环境时下载预训练模型
- 批量部署多个训练实例
- 离线环境预先准备模型文件

**技术特点**:
- 目标目录: `/root/autodl-tmp/MODELS/`
- 下载源: ModelScope (国内镜像，速度快)
- 自动验证: 检查模型文件完整性
- 进度显示: 实时下载状态

**使用步骤**:
```bash
# 1. 运行下载脚本
python download_qwen3.5_9b.py

# 2. 等待下载完成（约10-30分钟）
# 3. 验证模型文件
# 4. 在训练脚本中修改模型路径
```

**下载后如何使用**:
```python
# 在其他训练脚本中修改模型路径
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/root/autodl-tmp/MODELS/Qwen/Qwen3.5-9B",
    ...
)
```

**注意事项**:
- ⚠️ 需要稳定的网络连接
- ⚠️ 确保有足够的磁盘空间 (~8GB)
- ⚠️ 建议使用清华镜像源加速

---

#### 3. extract_gsm8k_full_samples.py - GSM8K全量数据提取
**功能**: 从GSM8K原始parquet文件提取并转换为JSON/CSV训练格式

**适用场景**:
- 将Hugging Face数据集转换为自定义格式
- 生成完整的GSM8K训练集 (7473条样本)
- 创建适合GRPO训练的数据格式
- 数据统计和分析

**输入格式**: Parquet文件
```
【数据集】gsm8k/main/train-00000-of-00001.parquet
列: question, answer
```

**输出格式**:
1. **JSON文件** (`gsm8k_full_samples.json`)
   ```json
   {
     "id": 1,
     "question": "问题文本",
     "full_answer": "完整解答（含推理过程）",
     "final_answer": "最终数字答案",
     "prompt": [
       {"role": "system", "content": "..."},
       {"role": "user", "content": "问题"}
     ]
   }
   ```

2. **CSV文件** (`gsm8k_full_samples.csv`)
   - 便于Excel查看和人工审核
   - 包含ID、问题、答案预览

**数据统计功能**:
- 问题长度分布 (平均/最短/最长/中位数)
- 答案长度分布
- 答案类型统计 (数字vs非数字)
- 样本预览 (前5条)

**使用示例**:
```bash
python extract_gsm8k_full_samples.py
```

**输出示例**:
```
Step 1: 定位数据集文件
✓ 文件已定位

Step 2: 读取全量Parquet数据
✓ 数据读取成功
  - 总样本数: 7473
  - 列名: ['question', 'answer']

Step 6: 数据统计与分析
问题长度统计:
  - 平均长度: 120 字符
  - 最短: 35 字符
  - 最长: 312 字符

答案类型统计:
  - 数字答案: 7473 (100.0%)
  - 非数字答案: 0 (0.0%)

生成的文件:
  1. gsm8k_full_samples.json (8098.7 MB)
  2. gsm8k_full_samples.csv (2567.0 MB)
```

**与extract_gsm8k_samples.py的区别**:
- ✅ `extract_gsm8k_full_samples.py`: 提取**全量**7473条数据
- ❌ `extract_gsm8k_samples.py`: 仅提取**部分**样本 (如100条)

---

#### 4. extract_gsm8k_samples.py - GSM8K小样本提取
**功能**: 从GSM8K数据集提取少量样本用于快速测试

**适用场景**:
- 快速验证训练流程
- 调试代码逻辑
- 小规模实验
- 节省存储空间

**技术特点**:
- 可自定义提取数量 (默认100条)
- 随机采样保证代表性
- 生成轻量级JSON/CSV文件
- 适合初学者和测试环境

**使用示例**:
```bash
# 提取100条样本
python extract_gsm8k_samples.py

# 修改代码中的num_samples参数可调整数量
```

**输出文件**:
- `gsm8k_100_samples.json` (105 KB)
- `gsm8k_100_samples.csv` (34 KB)

**与extract_gsm8k_full_samples.py的对比**:

| 特性 | extract_gsm8k_samples.py | extract_gsm8k_full_samples.py |
|------|-------------------------|-------------------------------|
| 样本数量 | 100条 (可配置) | 7473条 (全量) |
| JSON文件大小 | ~105 KB | ~8 GB |
| CSV文件大小 | ~34 KB | ~2.5 GB |
| 处理时间 | < 1秒 | ~30秒 |
| 适用场景 | 测试/调试 | 正式训练 |
| 磁盘占用 | 极小 | 较大 |

**选择建议**:
- 🧪 测试环境 → 用 `extract_gsm8k_samples.py` (100条)
- 🚀 生产训练 → 用 `extract_gsm8k_full_samples.py` (全量)

---

### 🔧 工具脚本使用流程建议

**新手入门流程**:
```bash
# 1. 下载模型
python download_qwen3.5_9b.py

# 2. 提取小样本测试数据
python extract_gsm8k_samples.py

# 3. 运行轻量化训练验证
python Qwen3_5_(9B)_GRPO_Lightweight.py

# 4. 评估训练效果
python model_comparison_eval.py
```

**完整训练流程**:
```bash
# 1. 提取全量训练数据
python extract_gsm8k_full_samples.py

# 2. 运行完整训练
python Qwen3_5_(9B)_GRPO_AutoDL.py

# 3. 对比评估不同版本
python model_comparison_eval.py
```

## 📈 训练特性

### 显存优化技术
- **4bit量化**：大幅降低模型内存占用
- **LoRA适配器**：仅训练少量参数，保持基座模型不变
- **梯度检查点**：减少训练过程中的显存峰值
- **分页优化器**：避免显存碎片化

### 奖励函数设计 (GRPO)
- **正确性奖励**：答案准确性评估
- **格式奖励**：XML标签结构完整性
- **推理质量**：思维链逻辑性评分
- **整数验证**：数值答案格式检查

## 📝 输出格式

GRPO训练采用结构化输出格式：
```xml
<reasoning>
详细的推理过程...
</reasoning>
<answer>
最终答案
</answer>
```

## 🔧 自定义配置

### 修改模型路径
在各训练脚本中修改`model_name`参数指向本地模型路径：
```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/path/to/your/model",
    ...
)
```

### 调整训练参数
- `max_seq_length`：最大序列长度
- `lora_rank`：LoRA适配器秩
- `per_device_train_batch_size`：批次大小
- `max_steps`：训练步数

## 📚 技术栈

- **Unsloth**：高效LLM微调框架
- **Transformers**：Hugging Face模型库
- **TRL**：Transformer强化学习库
- **vLLM**：高性能推理引擎
- **Datasets**：数据处理库
- **PyTorch**：深度学习框架

## ⚠️ 注意事项

1. **GPU要求**：所有训练脚本都需要GPU支持，CPU训练极慢且不实用
2. **显存管理**：如遇显存不足，可降低batch_size或启用4bit量化
3. **数据隐私**：请勿上传包含敏感信息的训练数据到公共仓库
4. **模型版权**：使用预训练模型时请遵守相应的开源协议

## 🤝 贡献

欢迎提交Issue和Pull Request来改进本项目！

## 📄 许可证

本项目仅供学习交流使用，请遵守相关模型的开源许可证要求。

## 🙏 致谢

- Unsloth团队提供的高效微调框架
- Hugging Face社区的优秀工具链
- Qwen系列模型的开源贡献

---

**注意**：大型数据集文件和训练生成的模型权重未包含在仓库中，请根据需要自行下载或训练。
