"""
================================================================================
第四章：使用 GRPO 进行可验证奖励训练
================================================================================

配套文档：04_grpotrainer_tutorial.md

【本文件是什么？】

这是 LLM 微调教程系列的第四个代码文件，展示如何使用 TRL 库的 GRPOTrainer 
进行群组相对策略优化（Group Relative Policy Optimization）训练。

【前置知识】

请先阅读前三章：
- trainer.py（第一章）：理解 Loss Masking 和 SFT 基础
- sfttrainer.py（第二章）：理解 SFTTrainer 的自动化训练
- dpotrainer.py（第三章）：理解 DPO 的偏好对齐

【什么是 GRPO？】

GRPO 是一种更强大的强化学习方法，相比 DPO：
- DPO：只能用离线偏好数据（chosen vs rejected）
- GRPO：支持自定义奖励函数（如数学题的正确性）

【核心学习目标】

1. 奖励模型设计（本章重点）
   - 奖励模型 vs 奖励函数的区别
   - 可验证奖励：数学题、代码测试（基于规则）
   - 启发式奖励：长度、格式、多样性（基于启发式）
   - 模型化奖励：使用神经网络作为奖励模型（高级）
   - 组合奖励：多个奖励信号的加权融合

2. GRPOTrainer 工作原理
   - 在线策略梯度：训练时动态生成响应
   - 群组相对优势：使用同组响应的相对排名估计优势
   - KL 正则化：防止模型偏离参考模型太远
   - 无需价值网络：不同于传统 PPO

3. 关键参数调优
   - num_generations：每个 prompt 生成多少个响应（通常 4-8）
   - kl_coef/beta：KL 正则化系数（类似 DPO 的 beta）
   - temperature：采样温度（越高越多样，影响探索）

4. 监控指标解读
   - rewards/mean：平均奖励（越高越好）
   - rewards/best：最佳奖励（越高越好）
   - objective/kl：KL 散度（不应过大，建议 < 10）
   - completions/clipped_ratio：生成是否被截断

【快速开始】

1. 安装依赖：
   pip install transformers datasets peft bitsandbytes accelerate trl

2. 运行训练：
   python grpotrainer.py

3. 修改配置：在文件底部修改 MODEL_NAME、奖励函数等参数

【训练流程详解】

1. 准备 SFT 模型（先完成第一、二章的训练）
2. 准备 prompt 数据集（只需要 prompt，不需要 response）
3. 设计奖励函数（根据任务需求，这是最关键的步骤）
4. 加载模型（GRPOTrainer 会自动创建参考模型的副本）
5. 配置 GRPOConfig（num_generations, beta 等）
6. 创建 GRPOTrainer 并训练
7. 监控奖励指标和 KL 散度

【奖励模型设计哲学】

奖励模型是 GRPO 的"大脑"，它告诉模型什么是好的、什么是坏的。

核心原则：
1. 对齐性：奖励应该反映真实的任务目标
2. 区分度：好的响应和差的响应应该有明显的奖励差异
3. 可计算性：奖励计算应该高效且可扩展
4. 鲁棒性：对边界情况和噪声有良好的容错性

三种奖励模型范式：
1. 基于规则的奖励（本代码示例）
   - 优点：可解释、可控、无需额外训练
   - 缺点：难以捕捉复杂模式
   
2. 基于模型的奖励（高级用法）
   - 优点：可以学习复杂的评估标准
   - 缺点：需要额外训练奖励模型
   
3. 混合奖励（推荐）
   - 结合规则和模型的优势

【使用建议】

- GRPO 应该在 SFT 之后进行（与 DPO 类似）
- 训练通常需要更多步数（5-10 epochs）
- 奖励函数设计是关键：需要能够准确评估响应质量
- num_generations 越大，训练信号越稳定，但计算成本越高
- 温度设置影响探索：太低导致缺乏多样性，太高导致质量下降

【下一步】

GRPO 是当前最先进的方法之一，被用于 DeepSeek-R1 等推理模型。
进阶方向：多阶段训练、课程学习、自我反思奖励、神经奖励模型
"""

# ============================================================================
# 导入必要的库
# ============================================================================

import torch
from pathlib import Path
import re

from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 尝试导入 flash-attn（可选，如果不可用会自动回退）
try:
    from flash_attn import __version__ as flash_attn_version
    FLASH_ATTENTION_AVAILABLE = True
    print(f"✓ Flash Attention 可用，版本: {flash_attn_version}")
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False
    print("⚠️  Flash Attention 不可用，将使用标准注意力（可以安装: pip install flash-attn）")

# ============================================================================
# 配置参数
# ============================================================================

# 模型配置
MODEL_NAME = "Qwen/Qwen3-8B"  # 使用 Qwen3-8B 模型
USE_FLASH_ATTENTION = True  # 是否使用 Flash Attention（如果可用）

# 数据集配置
# GRPO 只需要 prompt，不需要 response（因为会在训练时动态生成）
DATASET_NAME = "gsm8k"  # 使用 GSM8K 数学问题数据集
DATASET_SUBSET = "main"  # 数据集子集

# 训练配置
OUTPUT_DIR = "./grpo-training-output"
MAX_LENGTH = 2048  # 最大序列长度
MAX_PROMPT_LENGTH = 512  # prompt 的最大长度

# GRPO 特定配置
NUM_GENERATIONS = 4  # 每个 prompt 生成多少个响应
# num_generations 越大：
# - 训练信号越稳定（更多样本用于优势估计）
# - 计算成本越高（生成和奖励计算）
# - 建议范围：4-8

KL_COEF = 0.05  # KL 正则化系数（类似 DPO 的 beta）
# kl_coef 越大：模型越保守，越接近参考模型
# kl_coef 越小：模型越激进，可能偏离参考模型更远
# 建议范围：0.01 - 0.1

TEMPERATURE = 0.7  # 采样温度
# temperature 越高：生成越多样，但可能质量下降
# temperature 越低：生成越确定，但可能缺乏探索
# 建议范围：0.6 - 1.0

# LoRA 配置（如果使用 PEFT）
USE_PEFT = True  # 是否使用 LoRA 进行参数高效微调
LORA_R = 64  # LoRA rank
LORA_ALPHA = 16  # LoRA alpha
LORA_DROPOUT = 0.01  # LoRA dropout

# ============================================================================
# 奖励函数设计（GRPO 的核心）
# ============================================================================
#
# 【奖励模型的重要性】
#
# 在 GRPO 中，奖励函数扮演着"教师"的角色，它告诉模型什么是好的响应。
# 奖励函数的质量直接决定了训练效果：
#   - 好的奖励函数 → 模型朝着正确方向优化
#   - 差的奖励函数 → 模型学到错误的行为模式（reward hacking）
#
# 【奖励函数 vs 奖励模型】
#
# 1. 奖励函数（Reward Function）- 本代码使用的方法
#    - 基于规则和启发式的评分逻辑
#    - 优点：可解释、可控、无需额外训练
#    - 适用：有明确评估标准的任务（数学、代码等）
#
# 2. 奖励模型（Reward Model）- 高级方法
#    - 使用神经网络学习奖励函数
#    - 需要人工标注的偏好数据训练
#    - 适用：主观性强、难以形式化的任务
#    - 例子：ChatGPT 使用的 RLHF 中的 Reward Model
#
# 【本代码的奖励设计策略】
#
# 我们使用"组合奖励"策略，包含三个维度：
#   1. 正确性奖励（主要）：答案是否正确
#   2. 推理质量奖励（次要）：是否包含推理步骤
#   3. 格式规范奖励（次要）：响应长度是否合理
#
# 这种多维度设计的好处：
#   - 即使答案错误，包含推理过程也能获得部分奖励
#   - 鼓励模型生成结构化、可解释的响应
#   - 防止过短或过长的响应
#
# ============================================================================

def extract_answer(text):
    """
    从模型生成的文本中提取最终答案。
    
    【为什么需要答案提取？】
    模型生成的是完整的推理过程，我们需要从中提取最终答案来验证正确性。
    
    【提取策略】
    GSM8K 数据集的标准答案格式是：#### 答案
    我们使用两种方法提取：
      1. 优先查找 #### 后的数字（标准格式）
      2. 回退到查找最后一个数字（宽松匹配）
    
    Args:
        text: 生成的文本
        
    Returns:
        提取的数字答案，如果提取失败返回 None
    """
    # 方法1: 查找 #### 后的数字（标准格式）
    # 正则表达式解释：
    #   ####\s*         匹配 #### 和可选的空格
    #   -?              可选的负号
    #   \d+             一个或多个数字
    #   (?:,\d+)*       可选的千分位逗号和数字（非捕获组）
    #   (?:\.\d+)?      可选的小数部分（非捕获组）
    match = re.search(r'####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)', text)
    if match:
        # 移除逗号并转换为浮点数
        answer_str = match.group(1).replace(',', '')
        try:
            return float(answer_str)
        except ValueError:
            pass
    
    # 方法2: 查找最后一个数字（回退策略）
    # 如果模型没有使用 #### 格式，尝试提取文本中最后出现的数字
    numbers = re.findall(r'-?\d+(?:,\d+)*(?:\.\d+)?', text)
    if numbers:
        answer_str = numbers[-1].replace(',', '')
        try:
            return float(answer_str)
        except ValueError:
            pass
    
    # 提取失败返回 None
    return None


def compute_math_reward(prompt, response, ground_truth):
    """
    计算数学问题的奖励函数（单个样本）。
    
    【奖励函数设计理念】
    
    这是一个"可验证奖励"函数，基于明确的正确性标准。
    与人工偏好标注不同，数学题的答案是客观的、可验证的。
    
    【奖励结构】（三层奖励金字塔）
    
    第一层：正确性奖励（基础奖励 1.0）
      - 答案完全正确 → +1.0
      - 答案错误但有尝试 → -0.5
      - 无法提取答案 → -1.0
    
    第二层：推理质量奖励（+0.3）
      - 包含推理关键词（because, therefore 等）→ +0.3
      - 包含数学符号（+, -, *, / 等）→ +0.3
      → 鼓励模型展示推理过程，而不是直接给答案
    
    第三层：长度规范奖励（+0.2）
      - 响应长度在 50-300 词之间 → +0.2
      → 防止过短（缺乏推理）或过长（冗余）的响应
    
    【奖励范围】
    - 最高奖励：1.0 + 0.3 + 0.2 = 1.5（答案正确 + 有推理 + 长度合适）
    - 最低奖励：-1.0（无法提取答案）
    
    【设计原则】
    1. 主要奖励（正确性）占比最大 → 确保模型优先学习正确答案
    2. 次要奖励（推理、格式）提供额外信号 → 提升响应质量
    3. 负奖励用于惩罚不良行为 → 避免模型学习捷径
    
    Args:
        prompt: 输入问题
        response: 模型生成的响应
        ground_truth: 正确答案
        
    Returns:
        奖励值（浮点数，范围通常在 [-1.0, 1.5]）
    """
    # 步骤 1: 从响应中提取答案
    predicted_answer = extract_answer(response)
    
    # 步骤 2: 如果无法提取答案，给予最低负奖励
    # 这鼓励模型至少要输出一个数字答案
    if predicted_answer is None:
        return -1.0
    
    # 步骤 3: 比较预测答案和正确答案
    try:
        ground_truth_num = float(str(ground_truth).replace(',', ''))
        
        # 步骤 4: 如果答案正确，计算总奖励（基础 + 额外）
        if abs(predicted_answer - ground_truth_num) < 1e-3:
            # 基础奖励：答案正确
            base_reward = 1.0
            
            # 额外奖励 1：响应长度适中
            # 目的：鼓励详细推理，但不过度冗长
            response_length = len(response.split())
            if 50 <= response_length <= 300:
                length_bonus = 0.2
            else:
                length_bonus = 0.0
            
            # 额外奖励 2：包含推理步骤
            # 目的：鼓励模型展示推理过程，提高可解释性
            # 检查是否包含推理关键词或数学符号
            reasoning_keywords = ['because', 'so', 'therefore', 'thus', '+', '-', '*', '/', '=']
            has_reasoning = any(keyword in response.lower() for keyword in reasoning_keywords)
            reasoning_bonus = 0.3 if has_reasoning else 0.0
            
            # 总奖励 = 基础奖励 + 所有额外奖励
            total_reward = base_reward + length_bonus + reasoning_bonus
            return total_reward
        else:
            # 步骤 5: 答案错误，但至少有尝试
            # 给予轻度负奖励（而不是 -1.0）
            # 这样模型不会因为一次错误就受到严厉惩罚
            return -0.5
            
    except (ValueError, TypeError):
        # 处理异常情况（如 ground_truth 格式错误）
        return -1.0


def reward_function(prompts, responses, ground_truths):
    """
    批量计算奖励函数（GRPOTrainer 的接口）。
    
    【这个函数的作用】
    
    GRPOTrainer 会在训练过程中调用这个函数来评估生成的响应。
    具体流程：
      1. 对每个 prompt 生成 k 个不同的响应（k = num_generations）
      2. 调用此函数计算每个响应的奖励
      3. 使用奖励计算群组相对优势（advantage）
      4. 根据优势更新模型参数
    
    【批处理设计】
    
    这个函数接受列表作为输入，可以一次处理多个样本：
      - 输入：n 个 prompts，n 个 responses
      - 输出：n 个 rewards
    
    如果奖励计算很复杂（如调用外部 API），可以在这里实现：
      - 并行化处理（使用 multiprocessing）
      - 批量 API 调用（减少网络开销）
      - 缓存机制（避免重复计算）
    
    【注意事项】
    
    ⚠️ 在实际应用中，这个函数无法直接访问 ground_truth！
    
    原因：GRPOTrainer 只传递 prompts 和 responses。
    
    解决方案：
      1. 在 prompt 中编码 ground_truth 信息（不推荐，会泄露答案）
      2. 维护一个 prompt → ground_truth 的映射字典
      3. 使用不需要 ground_truth 的启发式奖励（见下文）
    
    本代码为了教学目的保留了这个签名，但在实际训练中使用了
    启发式奖励（见下面的 grpo_reward_function）。
    
    Args:
        prompts: prompt 列表
        responses: 对应的响应列表
        ground_truths: 对应的正确答案列表
        
    Returns:
        奖励列表（每个响应对应一个奖励值）
    """
    rewards = []
    for prompt, response, gt in zip(prompts, responses, ground_truths):
        reward = compute_math_reward(prompt, response, gt)
        rewards.append(reward)
    return rewards


# ============================================================================
# 数据预处理函数
# ============================================================================

def format_gsm8k_prompt(sample):
    """
    格式化 GSM8K 数据集样本为 GRPO 所需格式。
    
    GRPO 需要的数据格式：
    - prompt: 输入 prompt（问题）- GRPOTrainer 要求的字段名
    - ground_truth: 正确答案（用于奖励计算）
    
    Args:
        sample: GSM8K 数据集中的一个样本
        
    Returns:
        格式化后的样本
    """
    # GSM8K 数据集格式：
    # - question: 问题文本
    # - answer: 包含推理过程和答案的文本（答案在 #### 之后）
    
    question = sample["question"]
    answer_text = sample["answer"]
    
    # 提取正确答案
    ground_truth = extract_answer(answer_text)
    if ground_truth is None:
        # 如果提取失败，尝试直接使用 answer 字段
        ground_truth = answer_text
    
    # 格式化为对话格式（Qwen 的格式）
    formatted_prompt = f"Question: {question}\n\nPlease solve this step by step and provide your final answer after ####.\n\nAnswer:"
    
    return {
        "prompt": formatted_prompt,  # GRPOTrainer 要求使用 "prompt" 字段名
        "ground_truth": ground_truth
    }


# ============================================================================
# 主训练函数
# ============================================================================

def main():
    """
    主训练函数，执行完整的 GRPO 训练流程。
    """
    print("=" * 80)
    print("GRPO 训练开始")
    print("=" * 80)
    
    try:
        script_dir = Path(__file__).parent.absolute()
    except NameError:
        script_dir = Path.cwd()
    
    dataset_cache_dir = script_dir / "datasets"
    model_cache_dir = script_dir / "models"
    dataset_cache_dir.mkdir(exist_ok=True)
    model_cache_dir.mkdir(exist_ok=True)
    
    print(f"数据集缓存目录: {dataset_cache_dir}")
    print(f"模型缓存目录: {model_cache_dir}")

    # ------------------------------------------------------------------------
    # 1. 加载数据集
    # ------------------------------------------------------------------------
    print("\n[步骤 1] 加载数据集...")
    print(f"数据集: {DATASET_NAME}")
    
    # 加载 GSM8K 数据集
    dataset = load_dataset(DATASET_NAME, DATASET_SUBSET, cache_dir=dataset_cache_dir)
    
    # 获取训练集
    train_dataset = dataset["train"]
    
    # 如果数据集很大，可以只使用一部分进行快速测试
    train_dataset = train_dataset.select(range(min(100, len(train_dataset))))  # 快速测试
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"训练集字段: {train_dataset.column_names}")
    
    # 格式化数据集
    print("\n格式化数据集为 GRPO 格式...")
    train_dataset = train_dataset.map(
        format_gsm8k_prompt,
        remove_columns=train_dataset.column_names,
    )
    
    print("\n示例样本:")
    print("=" * 80)
    sample = train_dataset[0]
    print(f"Prompt: {sample['prompt'][:200]}...")
    print(f"Ground Truth: {sample['ground_truth']}")
    print("=" * 80)

    # ------------------------------------------------------------------------
    # 2. 加载模型和分词器
    # ------------------------------------------------------------------------
    print("\n[步骤 2] 加载模型和分词器...")
    print(f"模型: {MODEL_NAME}")
    
    # 配置量化（使用 bfloat16）
    compute_dtype = getattr(torch, "bfloat16")
    print("使用 4-bit 量化以节省显存...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    
    # ========================================================================
    # 【Flash Attention 配置】
    # ========================================================================
    attn_implementation = None
    if USE_FLASH_ATTENTION and FLASH_ATTENTION_AVAILABLE:
        attn_implementation = "flash_attention_2"
        print("\n✓ 使用 Flash Attention 2")
        print("  优势：")
        print("    - 训练速度提升 2-3 倍")
        print("    - 显存节省 50-80%")
        print("    - 支持更长序列")
    elif USE_FLASH_ATTENTION and not FLASH_ATTENTION_AVAILABLE:
        print("\n⚠️  Flash Attention 不可用，将使用标准注意力")
        print("  提示：可以安装 flash-attn 来启用 Flash Attention")
        attn_implementation = "sdpa"
    else:
        attn_implementation = "sdpa"
        print("\n使用标准注意力（SDPA）")
    
    # 加载模型
    print("加载策略模型...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=compute_dtype,
        device_map="auto",
        quantization_config=quantization_config,
        attn_implementation=attn_implementation,
        trust_remote_code=True,
        cache_dir=model_cache_dir,
    )
    
    # 加载分词器
    print("加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        padding_side="left",  # GRPO 使用 left padding
    )
    
    # 设置 pad_token（如果不存在）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 准备模型进行量化训练
    print("准备模型进行量化训练...")
    model = prepare_model_for_kbit_training(model)
    
    # 添加 LoRA 适配器
    if USE_PEFT:
        print(f"添加 LoRA 适配器 (r={LORA_R}, alpha={LORA_ALPHA})...")
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
        )
    else:
        lora_config = None
    
    # ------------------------------------------------------------------------
    # 3. 配置 GRPO 训练参数
    # ------------------------------------------------------------------------
    print("\n[步骤 3] 配置 GRPO 训练参数...")
    
    # GRPOConfig 包含 GRPO 特定的参数
    training_args = GRPOConfig(
        # 基本训练参数
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,  # GRPO 可能需要更多 epochs
        max_steps=50,  # 快速测试，实际训练可以设置为 -1
        per_device_train_batch_size=1,  # GRPO 因为要生成多个响应，batch_size 通常较小
        gradient_accumulation_steps=8,  # 通过梯度累积增加有效 batch_size
        learning_rate=1e-6,  # GRPO 通常使用很小的学习率
        lr_scheduler_type="cosine",
        warmup_steps=10,
        
        # 优化器配置
        optim="paged_adamw_8bit",
        
        # 日志和保存
        logging_steps=5,
        logging_dir=f"{OUTPUT_DIR}/logs",
        save_strategy="steps",
        save_steps=100,
        
        # 其他配置
        gradient_checkpointing=True,
        report_to="none",
        remove_unused_columns=False,
        bf16=True,
        cast_lm_head_to_fp32=True,  # 修复量化模型生成时的类型不匹配问题
        
        # GRPO 特定参数
        num_generations=NUM_GENERATIONS,  # 每个 prompt 生成的响应数量
        temperature=TEMPERATURE,  # 采样温度
        max_completion_length=256,  # 生成的最大 token 数（正确的参数名）
        beta=KL_COEF,  # KL 正则化系数（使用 beta 参数）
        max_prompt_length=MAX_PROMPT_LENGTH,  # prompt 最大长度
    )
    
    print(f"训练参数配置完成:")
    print(f"  - 每个 prompt 生成数: {NUM_GENERATIONS}")
    print(f"  - KL 正则化系数: {KL_COEF}")
    print(f"  - 采样温度: {TEMPERATURE}")
    print(f"  - 学习率: {training_args.learning_rate}")
    print(f"  - 批次大小: {training_args.per_device_train_batch_size}")
    print(f"  - 梯度累积步数: {training_args.gradient_accumulation_steps}")
    
    # ------------------------------------------------------------------------
    # 4. 创建自定义奖励函数（GRPO 的核心）
    # ------------------------------------------------------------------------
    print("\n[步骤 4] 创建奖励函数...")
    
    # ========================================================================
    # 【为什么需要包装器函数？】
    # ========================================================================
    #
    # GRPOTrainer 的奖励函数接口是固定的：
    #   输入：prompts (List[str]), completions (List[str]), **kwargs
    #   输出：rewards (List[float])
    #
    # 但是我们的 compute_math_reward 需要 ground_truth，而 GRPOTrainer
    # 并不会传递这个信息。因此我们有两种选择：
    #
    # 方案 1：维护 prompt → ground_truth 映射（复杂但精确）
    # 方案 2：使用启发式奖励（简单但近似）← 本代码使用
    #
    # ========================================================================
    
    # ========================================================================
    # 【启发式奖励 vs 精确奖励】
    # ========================================================================
    #
    # 启发式奖励（Heuristic Reward）：
    #   - 不依赖正确答案，只检查响应的形式和结构
    #   - 优点：实现简单，适用于没有标准答案的场景
    #   - 缺点：可能被"欺骗"（模型学会形式而忽略内容）
    #   - 示例：检查格式、长度、关键词等
    #
    # 精确奖励（Exact Reward）：
    #   - 基于正确答案验证，如数学题的答案正确性
    #   - 优点：信号准确，不容易被欺骗
    #   - 缺点：需要维护 prompt → answer 映射
    #   - 示例：compare(predicted, ground_truth)
    #
    # 最佳实践：结合两者
    #   - 主要奖励：精确验证（如答案正确性）
    #   - 次要奖励：启发式检查（如格式、推理过程）
    #
    # ========================================================================
    
    def grpo_reward_function(prompts, completions, **kwargs):
        """
        GRPO 奖励函数包装器（符合 GRPOTrainer 接口）。
        
        【函数签名说明】
        
        这个签名是 GRPOTrainer 要求的标准格式：
          - prompts: 输入的 prompt 文本列表
          - completions: 模型生成的完成文本列表（不包含 prompt）
          - **kwargs: 额外参数，可能包含：
              * completion_ids: 生成的 token IDs
              * metadata: 数据集中的额外信息
        
        【调用时机】
        
        GRPOTrainer 在训练循环中会反复调用这个函数：
        
        for batch in dataloader:
            # 1. 对每个 prompt 生成 k 个响应
            responses = model.generate(prompts, num_return_sequences=k)
            
            # 2. 调用奖励函数评估所有响应
            rewards = reward_function(prompts, responses)  ← 这里！
            
            # 3. 计算群组相对优势
            advantages = rewards - rewards.mean()
            
            # 4. 使用优势更新模型
            loss = -advantages * log_probs
            loss.backward()
        
        【本实现的策略】
        
        我们使用"三维启发式奖励"：
        
        1. 格式奖励（Format Reward）：
           - 检查是否包含 #### 答案标记
           - 目的：鼓励模型使用规范格式
           - 权重：0.5 / -0.5
        
        2. 长度奖励（Length Reward）：
           - 检查响应长度是否在合理范围（50-300 词）
           - 目的：防止过短（无推理）或过长（冗余）
           - 权重：0.3 / -0.2
        
        3. 推理奖励（Reasoning Reward）：
           - 检查是否包含推理关键词
           - 目的：鼓励展示思维过程
           - 权重：0.3 / -0.1
        
        【奖励范围】
        - 最佳情况：0.5 + 0.3 + 0.3 = 1.1
        - 最差情况：-0.5 - 0.2 - 0.1 = -0.8
        
        【局限性与改进方向】
        
        ⚠️ 当前实现的局限：
          - 无法验证答案的正确性（缺少 ground_truth）
          - 可能被"欺骗"（模型学会形式而不是内容）
        
        ✅ 改进方向：
          1. 维护 prompt → ground_truth 映射字典
          2. 在 prompt 中编码额外信息以便提取答案
          3. 使用外部验证器（如代码执行、数学求解器）
          4. 训练一个神经网络奖励模型
        
        Args:
            prompts: prompt 列表
            completions: 完成文本列表（响应）
            **kwargs: 其他参数（如 completion_ids）
            
        Returns:
            奖励列表（每个 completion 对应一个标量奖励）
        """
        rewards = []
        
        for prompt, completion in zip(prompts, completions):
            # ================================================================
            # 奖励维度 1：格式规范性
            # ================================================================
            # 检查是否包含 #### 答案格式（GSM8K 标准）
            has_answer_format = "####" in completion
            format_reward = 0.5 if has_answer_format else -0.5
            
            # ================================================================
            # 奖励维度 2：长度合理性
            # ================================================================
            # 响应应该足够长以包含推理，但不应过于冗长
            response_length = len(completion.split())
            if 50 <= response_length <= 300:
                # 理想长度范围
                length_reward = 0.3
            else:
                # 过短或过长都会受到惩罚
                length_reward = -0.2
            
            # ================================================================
            # 奖励维度 3：推理质量
            # ================================================================
            # 检查是否包含表示推理过程的关键词
            # 这些词通常出现在逻辑推理中
            reasoning_keywords = [
                'because',   # 因果关系
                'so',        # 结论
                'therefore', # 逻辑推导
                'thus',      # 推理结果
                'total',     # 数学总和
                'each'       # 分配问题
            ]
            has_reasoning = any(keyword in completion.lower() for keyword in reasoning_keywords)
            reasoning_reward = 0.3 if has_reasoning else -0.1
            
            # ================================================================
            # 组合所有奖励维度
            # ================================================================
            # 简单加权：所有维度权重相等
            # 高级策略：可以根据任务调整权重，如：
            #   total_reward = 0.5 * format_reward + 0.3 * reasoning_reward + 0.2 * length_reward
            total_reward = format_reward + length_reward + reasoning_reward
            rewards.append(total_reward)
        
        return rewards
    
    print("奖励函数创建完成")
    print("奖励函数组成（三维启发式）：")
    print("  1. 格式奖励：是否包含 #### 答案格式")
    print("     - 有格式：+0.5 | 无格式：-0.5")
    print("  2. 长度奖励：响应长度是否适中（50-300 词）")
    print("     - 合理长度：+0.3 | 过短/过长：-0.2")
    print("  3. 推理奖励：是否包含推理关键词")
    print("     - 有推理：+0.3 | 无推理：-0.1")
    print("\n⚠️  注意：当前使用启发式奖励，未验证答案正确性")
    print("    改进方向：维护 prompt→ground_truth 映射以实现精确验证")
    
    # ------------------------------------------------------------------------
    # 5. 创建 GRPOTrainer（强化学习的执行引擎）
    # ------------------------------------------------------------------------
    print("\n[步骤 5] 创建 GRPOTrainer...")
    
    # ========================================================================
    # 【GRPOTrainer 工作原理详解】
    # ========================================================================
    #
    # GRPOTrainer 是 GRPO 算法的实现，它封装了复杂的强化学习训练循环。
    #
    # 核心流程（每个训练步骤）：
    #
    # 1️⃣ 在线采样（Online Sampling）
    #    for each prompt in batch:
    #        生成 k 个不同的响应（k = num_generations）
    #        使用当前策略模型 π_θ 采样
    #    
    #    示例：
    #      prompt: "计算 2+3"
    #      响应1: "2+3=5，答案是5"  
    #      响应2: "首先2加3等于5"
    #      响应3: "计算得出答案为5"
    #      响应4: "结果是5"
    #
    # 2️⃣ 奖励评估（Reward Evaluation）
    #    调用 reward_function 评估所有响应
    #    
    #    示例：
    #      奖励1: 0.8 (有格式+推理)
    #      奖励2: 0.6 (有推理)
    #      奖励3: 0.5 (较简单)
    #      奖励4: 0.3 (太简单)
    #
    # 3️⃣ 计算群组相对优势（Group Relative Advantage）
    #    这是 GRPO 的核心创新！
    #    
    #    计算群组均值：
    #      mean_reward = (0.8 + 0.6 + 0.5 + 0.3) / 4 = 0.55
    #    
    #    计算相对优势（每个响应 - 均值）：
    #      优势1 = 0.8 - 0.55 = +0.25  ← 好，增加概率
    #      优势2 = 0.6 - 0.55 = +0.05  ← 稍好，轻微增加
    #      优势3 = 0.5 - 0.55 = -0.05  ← 稍差，轻微减少
    #      优势4 = 0.3 - 0.55 = -0.25  ← 差，减少概率
    #    
    #    为什么叫"相对"？
    #      → 优势是相对于同组其他响应计算的
    #      → 不需要知道"绝对好"的标准
    #      → 只需要知道"相对更好"
    #
    # 4️⃣ 策略梯度更新（Policy Gradient Update）
    #    计算损失函数：
    #      L = -E[A(s,a) * log π_θ(a|s)] + β * KL(π_θ || π_ref)
    #    
    #    其中：
    #      - A(s,a): 优势函数（上一步计算的）
    #      - log π_θ(a|s): 生成该响应的对数概率
    #      - β * KL(...): KL 正则化项，防止偏离参考模型
    #    
    #    更新规则：
    #      优势 > 0 → 增加生成该响应的概率
    #      优势 < 0 → 减少生成该响应的概率
    #
    # 5️⃣ KL 正则化（KL Regularization）
    #    防止模型偏离参考模型太远：
    #      KL_penalty = β * KL_divergence(π_θ, π_ref)
    #    
    #    作用：
    #      - 保持训练稳定
    #      - 防止模式崩溃（mode collapse）
    #      - 保留预训练知识
    #
    # ========================================================================
    
    # ========================================================================
    # 【GRPOTrainer vs DPOTrainer 对比】
    # ========================================================================
    #
    # DPOTrainer:
    #   输入：离线数据（prompt, chosen, rejected）
    #   过程：隐式奖励，直接优化偏好
    #   优点：稳定、快速
    #   缺点：需要预先标注的偏好对
    #
    # GRPOTrainer:
    #   输入：只需 prompt（+ 可选的 ground_truth）
    #   过程：在线生成，显式奖励，策略梯度
    #   优点：灵活、支持可验证任务
    #   缺点：计算量大、需要设计奖励函数
    #
    # ========================================================================
    
    # ========================================================================
    # 【参数说明】
    # ========================================================================
    #
    # model: 策略模型（要训练的模型）
    #   - GRPOTrainer 会自动创建一个参考模型的副本（冻结参数）
    #   - 参考模型用于 KL 正则化
    #
    # reward_funcs: 奖励函数（注意是复数形式）
    #   - 可以是单个函数，也可以是函数列表
    #   - 如果是列表，会计算所有奖励的平均值
    #   - 签名：(prompts, completions, **kwargs) -> List[float]
    #
    # args: GRPOConfig 训练配置
    #   - 包含所有训练超参数
    #   - 特别重要：num_generations, beta, temperature
    #
    # train_dataset: 训练数据集
    #   - 必须包含 "prompt" 字段
    #   - 可以包含其他字段（如 ground_truth），通过 **kwargs 传递
    #
    # processing_class: 分词器
    #   - 用于将文本转换为 token IDs
    #   - 也用于解码生成的响应
    #
    # peft_config: LoRA 配置（可选）
    #   - 如果提供，GRPOTrainer 会自动应用 LoRA
    #   - 推荐使用 LoRA，可以大幅降低显存占用
    #
    # ========================================================================
    
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=grpo_reward_function,  # 自定义奖励函数（注意参数名是复数）
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    
    print("GRPOTrainer 创建完成")
    print("\n内部机制：")
    print("  1. 参考模型：已自动创建（用于 KL 正则化）")
    print("  2. 在线采样：每步生成 {} 个响应".format(NUM_GENERATIONS))
    print("  3. 群组优势：使用相对排名（无需价值网络）")
    print("  4. KL 惩罚：系数 = {}".format(KL_COEF))
    
    # ------------------------------------------------------------------------
    # 6. 开始训练
    # ------------------------------------------------------------------------
    print("\n[步骤 6] 开始 GRPO 训练...")
    print("=" * 80)
    
    # 训练过程中会记录以下指标：
    # - loss: GRPO 损失
    # - rewards/mean: 平均奖励
    # - rewards/best: 最佳奖励
    # - rewards/worst: 最差奖励
    # - objective/kl: KL 散度
    # - objective/entropy: 策略熵
    
    trainer.train()
    
    print("\n" + "=" * 80)
    print("GRPO 训练完成！")
    print("=" * 80)
    
    # ------------------------------------------------------------------------
    # 7. 保存模型
    # ------------------------------------------------------------------------
    print("\n[步骤 7] 保存模型...")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"模型已保存到: {OUTPUT_DIR}")
    
    # ------------------------------------------------------------------------
    # 8. 训练后分析（可选）
    # ------------------------------------------------------------------------
    print("\n[步骤 8] 训练指标分析...")
    print("\n关键指标说明:")
    print("  - rewards/mean: 平均奖励（应该逐渐上升）")
    print("  - rewards/best: 最佳奖励（反映模型的最佳表现）")
    print("  - objective/kl: KL 散度（不应过大，建议 < 10）")
    print("  - objective/entropy: 策略熵（反映生成的多样性）")
    print("\n如果平均奖励稳定上升，说明训练成功！")
    
    return trainer, model, tokenizer


# ============================================================================
# 辅助函数：测试训练后的模型
# ============================================================================

def test_model(model, tokenizer, prompt=None):
    """
    测试训练后的模型生成效果。
    
    Args:
        model: 训练后的模型
        tokenizer: 分词器
        prompt: 测试提示
    """
    if prompt is None:
        prompt = """Question: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?

Please solve this step by step and provide your final answer after ####.

Answer:"""
    
    print("\n" + "=" * 80)
    print("测试模型生成效果")
    print("=" * 80)
    print(f"提示: {prompt[:100]}...")
    
    # 设置模型为评估模式
    model.eval()
    
    # 准备输入
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs["input_ids"].shape[1]
    
    # 获取模型所在的设备
    device = next(model.parameters()).device
    model_dtype = model.dtype
    
    # 生成响应
    print("\n生成多个响应进行对比:")
    print("-" * 80)
    
    for i in range(3):  # 生成 3 个不同的响应
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=model_dtype):
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=True,
                    temperature=TEMPERATURE,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
        
        generated_ids = outputs[0][input_length:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        print(f"\n响应 {i+1}:")
        print(response)
        print("-" * 80)


# ============================================================================
# 主程序入口
# ============================================================================

if __name__ == "__main__":
    # 设置随机种子以确保可复现性
    torch.manual_seed(42)
    
    # 运行主训练函数
    trainer, model, tokenizer = main()
    
    # 测试模型
    test_model(model, tokenizer)
    
    print("\n训练脚本执行完成！")
    print(f"模型保存在: {OUTPUT_DIR}")
    print("\n下一步:")
    print("1. 检查训练日志，确认奖励是否上升")
    print("2. 使用训练后的模型进行推理测试")
    print("3. 根据任务调整奖励函数设计")
    print("4. 尝试不同的 num_generations 和 temperature 参数")

