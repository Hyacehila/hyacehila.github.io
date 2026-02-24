"""
================================================================================
第三章：使用 DPO 进行偏好对齐
================================================================================

配套文档：03_dpotrainer_tutorial.md

【本文件是什么？】

这是 LLM 微调教程系列的第三个代码文件，展示如何使用 TRL 库的 DPOTrainer 
进行直接偏好优化（Direct Preference Optimization）训练。

【前置知识】

请先阅读前两章：
- trainer.py（第一章）：理解 Loss Masking 和 SFT 基础
- sfttrainer.py（第二章）：理解 SFTTrainer 的自动化训练

【什么是 DPO？】

DPO 是一种让模型学习人类偏好的训练方法：
- 传统 RLHF：需要训练奖励模型 + PPO 强化学习（复杂）
- DPO：直接用偏好数据训练，无需奖励模型（简单）

【核心学习目标】

1. 偏好数据格式
   {"prompt": "问题", "chosen": "好回答", "rejected": "差回答"}

2. 关键参数
   - beta：KL 正则化强度（0.1 起步，越大越保守）
   - loss_type：损失函数类型（sigmoid/hinge/ipo）

3. 参考模型（Reference Model）
   - 用于防止策略模型偏离太远
   - ref_model=None 时自动创建（推荐，节省显存）

4. 监控指标
   - rewards/margins：chosen - rejected 奖励差（越大越好）
   - rewards/accuracies：正确排序比例（越接近 1.0 越好）

【快速开始】

1. 安装依赖：
   pip install transformers datasets peft bitsandbytes accelerate trl

2. 运行训练：
   python dpotrainer.py

3. 修改配置：在文件底部修改 MODEL_NAME、BETA 等参数

【训练流程】

1. 准备 SFT 模型（先完成第一、二章的训练）
2. 准备偏好数据集（prompt + chosen + rejected）
3. 加载模型，设置 ref_model=None
4. 配置 DPOConfig（beta, loss_type 等）
5. 创建 DPOTrainer 并训练
6. 监控 rewards/margins 和 rewards/accuracies

【使用建议】

- DPO 应该在 SFT 之后进行
- 训练通常只需 1-3 个 epoch
- beta 太小会导致模型偏离过远，太大则改进不明显
- 确保 chosen 确实比 rejected 好

【下一步】

DPO 的局限：只能用离线偏好数据，无法自定义奖励。
进阶方向：GRPO（可自定义奖励）、PPO（传统 RLHF）
"""

# ============================================================================
# 导入必要的库
# ============================================================================


import torch 
from pathlib import Path

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import DPOConfig, DPOTrainer
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
DATASET_NAME = "trl-lib/ultrafeedback_binarized"  # 官方提供的偏好数据集
# 其他可用的偏好数据集：
# - "Anthropic/hh-rlhf" (Helpful and Harmless)
# - "openbmb/UltraFeedback_binarized"
# - "argilla/ultrafeedback-binarized-preferences-cleaned"

# 训练配置
OUTPUT_DIR = "./dpo-training-output"
MAX_LENGTH = 2048  # 最大序列长度（与 sfttrainer.py 保持一致）


# DPO 特定配置
BETA = 0.1  # KL 正则化强度，控制模型偏离参考模型的程度
# beta 越大，模型越保守，越接近参考模型
# beta 越小，模型越激进，可能偏离参考模型更远
# 建议范围：0.05 - 0.5

LOSS_TYPE = "sigmoid"  # 损失函数类型
# 可选值：
# - "sigmoid" (默认): 标准 DPO 损失，基于 Bradley-Terry 模型
# - "hinge": 铰链损失，来自 RSO 论文
# - "ipo": 来自 IPO 论文，解决过拟合问题
# - "exo_pair": 来自 EXO 论文，使用反向 KL 散度
# - "nca_pair": 来自 NCA 论文，优化绝对似然
# - "robust": 鲁棒 DPO，处理噪声标签
# - "bco_pair": 来自 BCO 论文，训练二分类器

LABEL_SMOOTHING = 0.0  # 标签平滑，用于 cDPO（处理噪声标签）
# 如果数据有噪声，可以设置为 0.1-0.2

# LoRA 配置（如果使用 PEFT）
USE_PEFT = True  # 是否使用 LoRA 进行参数高效微调
LORA_R = 64  # LoRA rank（与 sfttrainer.py 保持一致）
LORA_ALPHA = 16  # LoRA alpha（与 sfttrainer.py 保持一致）
LORA_DROPOUT = 0.01  # LoRA dropout（与 sfttrainer.py 保持一致）

# ============================================================================
# 数据预处理函数
# ============================================================================

def format_preference_dataset(sample):
    """
    格式化偏好数据集样本。
    
    DPO 需要的数据格式：
    - prompt: 输入提示
    - chosen: 被选中的（更好的）响应
    - rejected: 被拒绝的（较差的）响应
    
    如果数据集已经是标准格式，这个函数可能不需要修改。
    如果数据集格式不同，需要在这里进行转换。
    
    Args:
        sample: 数据集中的一个样本
        
    Returns:
        格式化后的样本
    """
    # ultrafeedback_binarized 数据集已经是标准格式
    # 如果使用其他数据集，可能需要在这里进行格式转换
    
    # 示例：如果数据集格式不同，可以这样转换：
    # return {
    #     "prompt": sample["instruction"],
    #     "chosen": sample["preferred_response"],
    #     "rejected": sample["rejected_response"]
    # }
    
    return sample


# ============================================================================
# 主训练函数
# ============================================================================

def main():
    """
    主训练函数，执行完整的 DPO 训练流程。
    """
    print("=" * 80)
    print("DPO 训练开始")
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
    print("\n[步骤 1] 加载偏好数据集...")
    print(f"数据集: {DATASET_NAME}")
    
    # 加载偏好数据集
    # ultrafeedback_binarized 数据集已经包含 prompt, chosen, rejected 字段
    dataset = load_dataset(DATASET_NAME,cache_dir=dataset_cache_dir)
    
    # 获取训练集和验证集
    train_dataset = dataset["train"]
    eval_dataset = dataset.get("test", None)  # 尝试获取测试集作为评估集
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"训练集字段: {train_dataset.column_names}")
    if eval_dataset is not None:
        print(f"评估集大小: {len(eval_dataset)}")
    else:
        print("未找到评估集，将禁用评估")
    

    print("\n示例样本:")
    print("=" * 80)
    sample = train_dataset[0]
    print(sample["chosen"])
    print("=" * 80)
    print(sample["rejected"])
    print("=" * 80)

    
    # ------------------------------------------------------------------------
    # 2. 加载模型和分词器
    # ------------------------------------------------------------------------
    print("\n[步骤 2] 加载模型和分词器...")
    print(f"模型: {MODEL_NAME}")
    
    # 配置量化（使用 bfloat16，与 sfttrainer.py 保持一致）
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
    # 如果使用 Flash Attention，需要设置 attn_implementation
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
        attn_implementation = "sdpa"  # 使用 PyTorch 的 SDPA（Scaled Dot Product Attention）
    else:
        attn_implementation = "sdpa"
        print("\n使用标准注意力（SDPA）")
    
    # 加载模型
    # 注意：DPO 需要两个模型：
    # 1. 策略模型（policy model）：我们要训练优化的模型
    # 2. 参考模型（reference model）：用于 KL 正则化，通常与策略模型相同但冻结
    
    print("加载策略模型...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=compute_dtype,  
        device_map="auto",
        quantization_config=quantization_config,
        attn_implementation=attn_implementation,  # 设置注意力实现
        trust_remote_code=True,
        cache_dir=model_cache_dir,
    )
    
    # 加载分词器
    print("加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        padding_side="left",  # DPO 通常使用 left padding
    )
    
    # 设置 pad_token（如果不存在）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 如果使用量化，准备模型进行训练

    print("准备模型进行量化训练...")
    model = prepare_model_for_kbit_training(model)
    
    # 如果使用 LoRA，添加 LoRA 适配器

    print(f"添加 LoRA 适配器 (r={LORA_R}, alpha={LORA_ALPHA})...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Qwen 模型的注意力层
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )


    
    # ------------------------------------------------------------------------
    # 3. 创建参考模型
    # ------------------------------------------------------------------------
    print("\n[步骤 3] 创建参考模型...")
    
    # DPO 需要一个参考模型来计算 KL 散度正则化
    # 参考模型通常是策略模型的副本，但参数冻结
    # 如果 ref_model 为 None，DPOTrainer 会自动创建一个参考模型
    
    # 选项 1：让 DPOTrainer 自动创建参考模型（推荐，节省显存）
    ref_model = None
    
    # 选项 2：手动创建参考模型（如果需要在训练过程中更新参考模型）
    # ref_model = AutoModelForCausalLM.from_pretrained(
    #     MODEL_NAME,
    #     dtype=compute_dtype,
    #     device_map="auto",
    #     quantization_config=quantization_config,
    #     trust_remote_code=True,
    # )
    # if USE_QUANTIZATION:
    #     ref_model = prepare_model_for_kbit_training(ref_model)
    # ref_model.config.use_cache = False
    
    print("参考模型配置完成（将使用策略模型的副本作为参考）")
    
    # ------------------------------------------------------------------------
    # 4. 配置 DPO 训练参数
    # ------------------------------------------------------------------------
    print("\n[步骤 4] 配置 DPO 训练参数...")
    
    # DPOConfig 继承自 TrainingArguments，包含所有标准训练参数
    # 此外还包含 DPO 特定的参数
    training_args = DPOConfig(
        # 基本训练参数
        output_dir=OUTPUT_DIR,
        max_steps=10,  # DPO 通常只需要 1-3 个 epochs， 这是max step加快训练速度
        per_device_train_batch_size=4,  # 根据显存调整
        gradient_accumulation_steps=4,  # 梯度累积步数
        learning_rate=5e-7,  # DPO 通常使用较小的学习率
        lr_scheduler_type="cosine",
        warmup_steps=100,
        
        # 优化器配置
        optim="paged_adamw_8bit" ,
        
        # 日志和保存
        logging_steps=10,
        logging_dir=f"{OUTPUT_DIR}/logs",
        save_strategy="steps",
        save_steps=500,
        # 评估策略：如果有评估集则启用，否则禁用
        eval_strategy="steps" ,
        eval_steps=500 ,
        
        # 其他配置
        gradient_checkpointing=True,  # 节省显存
        report_to="none",  # 不使用 wandb/tensorboard
        remove_unused_columns=False,  # 保留所有列，DPOTrainer 需要
        bf16=True,
        
        # DPO 特定参数
        beta=BETA,  # KL 正则化强度
        loss_type=LOSS_TYPE,  # 损失函数类型
        label_smoothing=LABEL_SMOOTHING,  # 标签平滑（用于 cDPO）
        max_length=MAX_LENGTH,  # 最大序列长度

        
        # 参考模型同步（用于 TR-DPO）
        # sync_ref_model=False,  # 是否同步参考模型权重
        # ref_model_sync_steps=100,  # 同步步数
        # ref_model_mixup_alpha=0.5,  # 混合权重
    )
    
    print(f"训练参数配置完成:")
    print(f"  - Beta (KL 正则化): {BETA}")
    print(f"  - Loss 类型: {LOSS_TYPE}")
    print(f"  - 学习率: {training_args.learning_rate}")
    print(f"  - 批次大小: {training_args.per_device_train_batch_size}")
    print(f"  - 梯度累积步数: {training_args.gradient_accumulation_steps}")
    
    # ------------------------------------------------------------------------
    # 5. 创建 DPOTrainer
    # ------------------------------------------------------------------------
    print("\n[步骤 5] 创建 DPOTrainer...")
    
    # DPOTrainer 是 DPO 训练的核心类
    # 它自动处理：
    # 1. 偏好数据的格式化和分词
    # 2. DPO 损失的计算
    # 3. 奖励指标的计算和记录
    # 4. 参考模型的管理
    
    trainer = DPOTrainer(
        model=model,  # 策略模型（要训练的模型）
        ref_model=ref_model,  # 参考模型（None 则自动创建）
        args=training_args,  # 训练参数
        train_dataset=train_dataset,  # 训练数据集
        eval_dataset=eval_dataset,  # 评估数据集（如果有）
        processing_class=tokenizer,  # 分词器
        peft_config=lora_config,  # PEFT 配置
    )
    
    print("DPOTrainer 创建完成")
    
    # ------------------------------------------------------------------------
    # 6. 开始训练
    # ------------------------------------------------------------------------
    print("\n[步骤 6] 开始 DPO 训练...")
    print("=" * 80)
    
    # 训练过程中会记录以下指标：
    # - loss: DPO 损失
    # - rewards/chosen: chosen 响应的平均奖励
    # - rewards/rejected: rejected 响应的平均奖励
    # - rewards/margins: chosen 和 rejected 奖励的差值（越大越好）
    # - rewards/accuracies: chosen 奖励大于 rejected 奖励的比例（越高越好）
    
    trainer.train()
    
    print("\n" + "=" * 80)
    print("DPO 训练完成！")
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
    print("  - rewards/chosen: chosen 响应的平均奖励（应该为正且较大）")
    print("  - rewards/rejected: rejected 响应的平均奖励（应该为负或较小）")
    print("  - rewards/margins: chosen 和 rejected 的奖励差值（应该逐渐增大）")
    print("  - rewards/accuracies: chosen 奖励大于 rejected 的比例（应该接近 1.0）")
    print("\n如果这些指标趋势良好，说明训练成功！")
    
    return trainer, model, tokenizer


# ============================================================================
# 辅助函数：测试训练后的模型
# ============================================================================

def test_model(model, tokenizer, prompt="What is the capital of France?"):
    """
    测试训练后的模型生成效果。
    
    Args:
        model: 训练后的模型
        tokenizer: 分词器
        prompt: 测试提示
    """
    print("\n" + "=" * 80)
    print("测试模型生成效果")
    print("=" * 80)
    print(f"提示: {prompt}")
    
    # ========================================================================
    # 关键修复 1：设置模型为评估模式
    # ========================================================================
    # 训练后必须将模型设置为评估模式，否则生成结果会不稳定
    model.eval()
    
    # 准备输入
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # 保存输入长度，用于后续只解码生成的部分
    input_length = inputs["input_ids"].shape[1]
    
    # 获取模型所在的设备
    device = next(model.parameters()).device
    
    
    # 使用 torch.autocast
    # 获取模型的数据类型，确保 autocast 使用正确的类型
    # 这解决了启用 Flash Attention 时可能出现的类型不匹配问题
    model_dtype = model.dtype
    
    # 生成响应
    with torch.no_grad():
        # 使用 autocast 来自动处理混合精度，解决类型不匹配问题
        with torch.autocast(device_type=device.type, dtype=model_dtype):
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
    
    # ========================================================================
    # 关键修复 2：只解码生成的部分，而不是整个输出
    # ========================================================================
    # outputs[0] 包含完整的输出（输入提示 + 生成的文本）
    # 我们只需要解码生成的部分，即 outputs[0][input_length:]
    generated_ids = outputs[0][input_length:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(f"响应: {response}")
    print("=" * 80)


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
    print("1. 检查训练日志，确认指标趋势良好")
    print("2. 使用训练后的模型进行推理测试")
    print("3. 如果效果不理想，调整 beta 或其他超参数重新训练")
