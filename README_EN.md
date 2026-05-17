# LLM Model Distillation and Fine-tuning Practice

This project provides complete practical code for LLM (Large Language Model) distillation and fine-tuning, including Supervised Fine-Tuning (SFT), Reinforcement Learning with GRPO, and multimodal model fine-tuning.

## 📋 Project Overview

Based on the Unsloth framework, this project implements various efficient LLM fine-tuning approaches:
- **SFT (Supervised Fine-Tuning)**: Instruction fine-tuning of Qwen2.5-7B using Alpaca dataset
- **GRPO (Group Relative Policy Optimization)**: Training reasoning capabilities using GSM8K math dataset
- **Multimodal Fine-Tuning**: Specialized fine-tuning for Qwen2-VL vision-language model
- **Model Evaluation**: Complete model performance comparison and evaluation tools

## 🚀 Quick Start

### Requirements

- Python 3.8+
- CUDA-compatible GPU (Recommended: A100/RTX 3090/RTX 5090)
- At least 16GB VRAM (4bit quantization can reduce requirements)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Main Training Scripts Explained

This project includes 5 core training scripts for different scenarios and models:

#### 1. Qwen2_5_(7B)_Alpaca.py - SFT Supervised Fine-Tuning
**Purpose**: Supervised fine-tuning (SFT) of Qwen2.5-7B using Alpaca instruction dataset

**Use Cases**:
- Teaching models to follow instructions
- General dialogue and task completion capability training
- Quick validation of LoRA fine-tuning workflow

**Technical Features**:
- Model: Qwen2.5-7B-Instruct
- Dataset: Alpaca-cleaned (52K instruction samples)
- Method: LoRA adapter + 4bit quantization
- Training steps: 60 steps (example configuration)
- VRAM requirement: ~16GB (A100/RTX 3090)

**Output Format**: Standard Alpaca format
```
### Instruction:
{instruction}

### Input:
{input}

### Response:
{response}
```

---

#### 2. Qwen2_5_(7B)_R1_GRPO.py - GRPO Reinforcement Learning (R1 Reasoning Model)
**Purpose**: Train mathematical reasoning capabilities of Qwen2.5-7B using GRPO reinforcement learning

**Use Cases**:
- Enhancing logical reasoning and Chain-of-Thought (CoT) capabilities
- Training reasoning models similar to DeepSeek-R1
- Tasks requiring structured reasoning output

**Technical Features**:
- Model: Qwen2.5-7B-Instruct
- Dataset: GSM8K mathematical word problems
- Method: GRPO (Group Relative Policy Optimization)
- Inference acceleration: vLLM fast inference engine
- Generation strategy: 6 candidate answers per question
- Training steps: 250 steps
- VRAM requirement: ~24GB (Recommended: A100/A800 40GB+)

**Reward Functions** (5 dimensions):
1. `correctness_reward_func`: Answer correctness (weight 2.0)
2. `int_reward_func`: Whether answer is integer (weight 0.5)
3. `strict_format_reward_func`: Strict XML format (weight 0.5)
4. `soft_format_reward_func`: Loose XML format (weight 0.5)
5. `xmlcount_reward_func`: XML tag completeness (weight 0.5)

**Output Format**: Structured reasoning format
```xml
<reasoning>
Detailed step-by-step reasoning process...
</reasoning>
<answer>
Final answer
</answer>
```

---

#### 3. Qwen3_5_(9B)_GRPO_AutoDL.py - Qwen3.5-9B Complete Training Version
**Purpose**: Complete GRPO training script for Qwen3.5-9B optimized for AutoDL cloud platform

**Use Cases**:
- Using newer Qwen3.5-9B model
- AutoDL cloud platform deployment (RTX 5090 32GB)
- Production environment complete training workflow
- Requires detailed training monitoring and statistics

**Technical Features**:
- Model: Qwen3.5-9B (stronger base model than 7B)
- Dataset: GSM8K (100 samples for validation)
- Optimization: VRAM-optimized configuration for 32GB GPUs
- Sequence length: 512 (reduced to save VRAM)
- LoRA rank: 16 (balance between performance and VRAM)
- Generation count: 4 candidate answers (reduced VRAM usage)
- Training steps: 50 steps (for validation)
- Feature: Complete training logs and GPU monitoring

**Differences from Qwen2_5_(7B)_R1_GRPO.py**:
- ✅ Uses newer Qwen3.5-9B model (more powerful)
- ✅ Optimized path configuration for AutoDL environment
- ✅ More detailed training progress display
- ✅ Better VRAM optimization (suitable for 32GB GPUs)
- ❌ Fewer training steps (50 vs 250, adjustable)

---

#### 4. Qwen3_5_(9B)_GRPO_Lightweight.py - Lightweight Validation Version
**Purpose**: Lightweight version of Qwen3.5-9B for quick full-pipeline validation

**Use Cases**:
- Resource-constrained environments
- Quick testing of training workflow
- Debugging and development phase
- GPUs with limited VRAM (<24GB)

**Technical Features**:
- Model: Qwen3.5-9B
- Dataset: GSM8K (100 samples)
- Optimization strategies:
  - Sequence length: 512 (shorter)
  - LoRA rank: 16 (smaller)
  - Generation count: 4 (fewer)
  - Training steps: 50 steps (quick validation)
- VRAM requirement: ~20GB

**Differences from Qwen3_5_(9B)_GRPO_AutoDL.py**:
- ✅ Lighter weight, faster startup
- ✅ Suitable for local development and testing
- ✅ Simpler code structure
- ❌ Does not use AutoDL-specific paths
- ❌ More conservative training configuration

**Selection Guide**:
- First run → Use Lightweight version to validate environment
- Formal training → Use AutoDL version for better results

---

#### 5. qwen_vl_car_insurance_train.py - Vision-Language Model Fine-Tuning
**Purpose**: Specialized fine-tuning of Qwen2.5-VL-3B multimodal model

**Use Cases**:
- Image understanding tasks (OCR, object recognition, etc.)
- Odometer reading recognition in car insurance domain
- Multimodal tasks requiring image + text processing
- Vertical domain visual question answering

**Technical Features**:
- Model: Qwen2.5-VL-3B-Instruct (vision-language model)
- Dataset: Excel format (image, prompt, response)
- Task: Vehicle odometer reading extraction
- Fine-tuning scope:
  - ✅ Vision layers
  - ✅ Language layers
  - ✅ Attention modules
  - ✅ MLP modules
- LoRA configuration: r=16, alpha=16
- Training steps: 30 steps
- VRAM requirement: ~16GB

**Input Format**: Multimodal dialogue
```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Please identify the odometer reading"},
        {"type": "image", "image": "image_object"}
      ]
    },
    {
      "role": "assistant",
      "content": [
        {"type": "text", "text": "Odometer reading: 12345 km"}
      ]
    }
  ]
}
```

**Key Differences from Other Scripts**:
- 🖼️ **Only script supporting image input**
- 📊 Uses Excel instead of JSON/Parquet data format
- 🎯 Targeted at specific vertical domain (car insurance)
- 🔧 Fine-tunes both vision and language layers simultaneously

---

### 📊 Training Scripts Comparison Summary

| Script | Model | Training Method | Data Type | VRAM Req. | Use Case |
|--------|-------|----------------|-----------|-----------|----------|
| Qwen2_5_(7B)_Alpaca.py | Qwen2.5-7B | SFT | Text instructions | ~16GB | General instruction following |
| Qwen2_5_(7B)_R1_GRPO.py | Qwen2.5-7B | GRPO | Math problems | ~24GB | Reasoning capability training |
| Qwen3_5_(9B)_GRPO_AutoDL.py | Qwen3.5-9B | GRPO | Math problems | ~28GB | AutoDL complete training |
| Qwen3_5_(9B)_GRPO_Lightweight.py | Qwen3.5-9B | GRPO | Math problems | ~20GB | Quick validation testing |
| qwen_vl_car_insurance_train.py | Qwen2.5-VL-3B | SFT | Image + Text | ~16GB | Multimodal vision tasks |

### 🔍 How to Choose a Training Script?

**By Task Type**:
- 📝 Text dialogue/instruction following → `Qwen2_5_(7B)_Alpaca.py`
- 🧮 Mathematical reasoning/logic → `Qwen2_5_(7B)_R1_GRPO.py` or Qwen3.5 series
- 🖼️ Image understanding/OCR → `qwen_vl_car_insurance_train.py`

**By Hardware Conditions**:
- 💾 VRAM < 20GB → `Qwen3_5_(9B)_GRPO_Lightweight.py`
- 💾 VRAM 20-30GB → `Qwen3_5_(9B)_GRPO_AutoDL.py`
- 💾 VRAM > 30GB → `Qwen2_5_(7B)_R1_GRPO.py` (can increase batch size)

**By Training Phase**:
- 🧪 Testing environment → `Qwen3_5_(9B)_GRPO_Lightweight.py`
- 🚀 Production training → `Qwen3_5_(9B)_GRPO_AutoDL.py`
- 📚 Learning SFT → `Qwen2_5_(7B)_Alpaca.py`
- 🎓 Learning GRPO → `Qwen2_5_(7B)_R1_GRPO.py`

## 📊 Datasets

### Built-in Sample Data
- `gsm8k_100_samples.json` - GSM8K math problem examples (100 samples)
- `images/` - Vehicle odometer sample images

### Large Dataset Acquisition
Due to large file sizes, datasets are recommended to be obtained via:
1. Use the provided `extract_gsm8k_full_samples.py` script to extract from original parquet files
2. Download official datasets from Hugging Face or ModelScope

## 🛠️ Utility Scripts Explained

This project includes 4 auxiliary utility scripts for data processing, model downloading, and evaluation:

#### 1. model_comparison_eval.py - Model Comparison Evaluation Tool
**Purpose**: Compare output quality and performance of models at different training stages

**Use Cases**:
- Compare base model vs SFT fine-tuned model vs GRPO model effects
- Quantitatively evaluate model improvement
- A/B testing different training configurations
- Generate model performance reports

**Evaluation Dimensions**:
1. **Format Compliance** (`format_compliance`)
   - Check if XML format or medical format is followed
   - Score: 0-1

2. **Answer Accuracy** (`answer_accuracy`)
   - Calculate similarity with reference answer based on F1 score
   - Applicable for scenarios with standard answers

3. **Reasoning Chain Quality** (`reasoning_quality`)
   - Reasoning process length score (30%)
   - Step-by-step score: whether reasoning is broken into steps (40%)
   - Conclusion score: whether there's a summary (30%)
   - Only applicable for GRPO structured output

4. **Response Length Reasonableness** (`response_length_score`)
   - Penalize too short or too long responses
   - Reasonable range: 20-2000 characters

5. **Language Match** (`language_match`)
   - Check if response language matches expectation
   - Chinese/English ratio statistics

**Usage Example**:
```python
# Built-in demo mode (using simulated data)
python model_comparison_eval.py

# For actual use, replace with real model inference results
test_cases = [
    {"question": "Question...", "reference": "Reference answer..."}
]
model_results = {
    "Base Model": ["Answer1", "Answer2"],
    "SFT Model": ["Answer1", "Answer2"],
    "GRPO Model": ["Answer1", "Answer2"]
}
compare_models(test_cases, model_results, eval_type="medical")
```

**Output Example**:
```
===== Model Comparison Evaluation Report =====

--- Test Case 1 ---
Question: I've been feeling dizzy lately, what should I do?

  [Base Model (Before Fine-tuning)]
    Response: I can provide general health advice...
    Overall Score: 0.3250
    format_score: 0.0
    language_match: 0.0

  [SFT Fine-tuned Model]
    Response: Dizziness can have many causes...
    Overall Score: 0.8500
    format_score: 1.0
    language_match: 1.0

Overall Evaluation Summary:
  Base Model (Before Fine-tuning): Average Overall Score = 0.3250
  SFT Fine-tuned Model: Average Overall Score = 0.8500
  GRPO Reinforcement Learning Model: Average Overall Score = 0.9200
```

---

#### 2. download_qwen35_4b.py - Model Download Tool
**Purpose**: Automatically download Qwen3.5-4B model from ModelScope to local storage

**Use Cases**:
- Download pre-trained models when setting up environment for the first time
- Batch deployment of multiple training instances
- Pre-prepare model files for offline environments

**Technical Features**:
- Target directory: `/root/autodl-tmp/MODELS/`
- Download source: ModelScope (domestic mirror, fast speed)
- Fallback method: Git clone approach
- Automatic verification: Check model file integrity
- Progress display: Real-time download status

**Usage Steps**:
```bash
# 1. Run download script
python download_qwen35_4b.py

# 2. Wait for download completion (approximately 10-30 minutes)
# 3. Verify model files
# 4. Modify model path in training scripts
```

**How to Use After Download**:
```python
# Modify model path in other training scripts
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/root/autodl-tmp/MODELS/Qwen/Qwen3.5-4B",
    ...
)
```

**Notes**:
- ⚠️ Requires stable network connection
- ⚠️ Ensure sufficient disk space (~8GB)
- ⚠️ Recommended to use Tsinghua mirror source for acceleration

---

#### 3. extract_gsm8k_full_samples.py - GSM8K Full Data Extraction
**Purpose**: Extract and convert GSM8K original parquet files to JSON/CSV training format

**Use Cases**:
- Convert Hugging Face datasets to custom format
- Generate complete GSM8K training set (7473 samples)
- Create data format suitable for GRPO training
- Data statistics and analysis

**Input Format**: Parquet file
```
【数据集】gsm8k/main/train-00000-of-00001.parquet
Columns: question, answer
```

**Output Formats**:
1. **JSON File** (`gsm8k_full_samples.json`)
   ```json
   {
     "id": 1,
     "question": "Question text",
     "full_answer": "Complete solution (with reasoning)",
     "final_answer": "Final numerical answer",
     "prompt": [
       {"role": "system", "content": "..."},
       {"role": "user", "content": "Question"}
     ]
   }
   ```

2. **CSV File** (`gsm8k_full_samples.csv`)
   - Easy to view in Excel and manual review
   - Contains ID, question, answer preview

**Data Statistics Features**:
- Question length distribution (average/min/max/median)
- Answer length distribution
- Answer type statistics (numeric vs non-numeric)
- Sample preview (first 5 samples)

**Usage Example**:
```bash
python extract_gsm8k_full_samples.py
```

**Output Example**:
```
Step 1: Locate dataset files
✓ File located

Step 2: Read full Parquet data
✓ Data read successfully
  - Total samples: 7473
  - Columns: ['question', 'answer']

Step 6: Data Statistics and Analysis
Question Length Statistics:
  - Average length: 120 characters
  - Shortest: 35 characters
  - Longest: 312 characters

Answer Type Statistics:
  - Numeric answers: 7473 (100.0%)
  - Non-numeric answers: 0 (0.0%)

Generated files:
  1. gsm8k_full_samples.json (8098.7 MB)
  2. gsm8k_full_samples.csv (2567.0 MB)
```

**Difference from extract_gsm8k_samples.py**:
- ✅ `extract_gsm8k_full_samples.py`: Extracts **full** 7473 samples
- ❌ `extract_gsm8k_samples.py`: Extracts only **partial** samples (e.g., 100)

---

#### 4. extract_gsm8k_samples.py - GSM8K Small Sample Extraction
**Purpose**: Extract small number of samples from GSM8K dataset for quick testing

**Use Cases**:
- Quick validation of training workflow
- Debugging code logic
- Small-scale experiments
- Saving storage space

**Technical Features**:
- Customizable extraction count (default 100 samples)
- Random sampling ensures representativeness
- Generates lightweight JSON/CSV files
- Suitable for beginners and test environments

**Usage Example**:
```bash
# Extract 100 samples
python extract_gsm8k_samples.py

# Modify num_samples parameter in code to adjust quantity
```

**Output Files**:
- `gsm8k_100_samples.json` (105 KB)
- `gsm8k_100_samples.csv` (34 KB)

**Comparison with extract_gsm8k_full_samples.py**:

| Feature | extract_gsm8k_samples.py | extract_gsm8k_full_samples.py |
|---------|-------------------------|-------------------------------|
| Sample Count | 100 (configurable) | 7473 (full) |
| JSON File Size | ~105 KB | ~8 GB |
| CSV File Size | ~34 KB | ~2.5 GB |
| Processing Time | < 1 second | ~30 seconds |
| Use Case | Testing/debugging | Formal training |
| Disk Usage | Minimal | Larger |

**Selection Guide**:
- 🧪 Test environment → Use `extract_gsm8k_samples.py` (100 samples)
- 🚀 Production training → Use `extract_gsm8k_full_samples.py` (full)

---

### 🔧 Recommended Utility Script Workflow

**Beginner Getting Started Flow**:
```bash
# 1. Download model
python download_qwen35_4b.py

# 2. Extract small sample test data
python extract_gsm8k_samples.py

# 3. Run lightweight training validation
python Qwen3_5_(9B)_GRPO_Lightweight.py

# 4. Evaluate training效果
python model_comparison_eval.py
```

**Complete Training Flow**:
```bash
# 1. Extract full training data
python extract_gsm8k_full_samples.py

# 2. Run complete training
python Qwen3_5_(9B)_GRPO_AutoDL.py

# 3. Compare and evaluate different versions
python model_comparison_eval.py
```

## 📈 Training Features

### VRAM Optimization Techniques
- **4bit Quantization**: Significantly reduces model memory usage
- **LoRA Adapters**: Train only a small number of parameters, keeping base model unchanged
- **Gradient Checkpointing**: Reduces VRAM peaks during training
- **Paged Optimizer**: Avoids VRAM fragmentation

### Reward Function Design (GRPO)
- **Correctness Reward**: Answer accuracy evaluation
- **Format Reward**: XML tag structure completeness
- **Reasoning Quality**: Chain-of-thought logic scoring
- **Integer Verification**: Numerical answer format checking

## 📝 Output Format

GRPO training uses structured output format:
```xml
<reasoning>
Detailed reasoning process...
</reasoning>
<answer>
Final answer
</answer>
```

## 🔧 Custom Configuration

### Modify Model Path
In each training script, modify the `model_name` parameter to point to your local model path:
```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/path/to/your/model",
    ...
)
```

### Adjust Training Parameters
- `max_seq_length`: Maximum sequence length
- `lora_rank`: LoRA adapter rank
- `per_device_train_batch_size`: Batch size
- `max_steps`: Training steps

## 📚 Tech Stack

- **Unsloth**: Efficient LLM fine-tuning framework
- **Transformers**: Hugging Face model library
- **TRL**: Transformer Reinforcement Learning library
- **vLLM**: High-performance inference engine
- **Datasets**: Data processing library
- **PyTorch**: Deep learning framework

## ⚠️ Important Notes

1. **GPU Requirement**: All training scripts require GPU support; CPU training is extremely slow and impractical
2. **VRAM Management**: If encountering VRAM issues, reduce batch_size or enable 4bit quantization
3. **Data Privacy**: Do not upload training data containing sensitive information to public repositories
4. **Model Licensing**: Comply with corresponding open-source licenses when using pre-trained models

## 🤝 Contributing

Issues and Pull Requests are welcome to improve this project!

## 📄 License

This project is for educational purposes only. Please comply with the open-source license requirements of related models.

## 🙏 Acknowledgments

- Unsloth team for providing efficient fine-tuning framework
- Hugging Face community for excellent toolchain
- Qwen series models for open-source contributions

---

**Note**: Large dataset files and trained model weights are not included in the repository. Please download or train as needed.