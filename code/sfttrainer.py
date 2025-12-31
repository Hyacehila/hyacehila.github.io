"""
================================================================================
第二章：使用 SFTTrainer 简化训练流程
================================================================================

配套文档：02_sfttrainer_tutorial.md

【本文件是什么？】

这是 LLM 微调教程系列的第二个代码文件，展示如何使用 TRL 库的 SFTTrainer 
简化 SFT 训练流程。相比第一章的原生 Trainer，代码更简洁，功能更强大。

【前置知识】

请先阅读 trainer.py（第一章），理解：
- Loss Masking 的核心概念
- DataCollator 的作用
- 为什么需要自动化这些步骤

【核心学习目标】

1. SFTTrainer vs Trainer
   - 自动 Loss Masking（无需手动计算 prompt_len）
   - 自动应用 chat_template
   - 自动配置 DataCollator
   - 自动应用 LoRA

2. 四种数据格式
   - messages：对话格式（推荐），使用 assistant_only_loss=True
   - prompt-completion：提示补全格式，使用 completion_only_loss=True
   - text：纯文本（预训练用）
   - conversational_prompt_completion：混合格式

3. Flash Attention
   - 训练加速 2-3 倍
   - 显存节省 50-80%
   - 需要 Ampere 架构以上 GPU（A100, H100, RTX 30xx/40xx）

【快速开始】

1. 安装依赖：
   pip install transformers datasets peft bitsandbytes accelerate trl
   pip install flash-attn --no-build-isolation  # 可选，加速训练

2. 运行训练：
   python sfttrainer.py

3. 修改配置：在文件底部 __main__ 部分修改 DATA_FORMAT、USE_FLASH_ATTENTION 等参数

【数据格式选择】

- "messages"：多轮对话任务（推荐）
  需要设置 assistant_only_loss=True 和 chat_template_path
  
- "conversational_prompt_completion"：单轮问答任务
  使用默认配置（completion_only_loss=True）

【注意事项】

使用 assistant_only_loss=True 时，需要自定义 chat_template：
- Qwen3 默认 template 不支持 generation 标签
- 修改后的 template 存储在：models/.../final_assistant.jinja
- 详见：https://github.com/HarryMayne/qwen_3_chat_templates

【下一步】

完成本章后，请阅读 dpotrainer.py（第三章），学习如何使用 DPO 进行偏好对齐。
"""


# ============================================================================
# 第一部分：库导入
# ============================================================================

# 导入核心库
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
)
from trl import SFTTrainer,SFTConfig
from functools import partial
import torch
import transformers
from pathlib import Path
import os

# 尝试导入 flash-attn（可选，如果不可用会自动回退）
try:
    from flash_attn import __version__ as flash_attn_version
    FLASH_ATTENTION_AVAILABLE = True
    print(f"✓ Flash Attention 可用，版本: {flash_attn_version}")
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False
    print("⚠️  Flash Attention 不可用，将使用标准注意力（可以安装: pip install flash-attn）")


# ============================================================================
# 第二部分：数据预处理函数 - 展示 SFTTrainer 支持的四种数据格式
# ============================================================================

def process_func_format_1_text(example, tokenizer, max_length):
    """
    格式1：Standard Language Modeling（标准语言建模）
    
    数据格式：{"text": "完整的文本内容"}
    
    适用场景：
    - 预训练任务
    - 续写任务
    - 不需要区分 prompt 和 completion 的场景
    
    特点：
    - 对整个文本计算 Loss
    - 没有 Loss Masking（所有 token 都计算 Loss）
    - 适用于预训练而非 SFT
    
    注意：这种格式通常不用于 SFT，因为会对所有 token 计算 Loss。
    
    Args:
        example: 包含 'dialogue' 和 'summary' 的数据样本
        tokenizer: 分词器实例
        max_length: 最大序列长度
        
    Returns:
        dict: 包含 'text' 字段的字典
    """
    # 直接拼接文本（不推荐用于 SFT，仅用于演示）
    text = f"Please Summarize: {example.get('dialogue', '')}\nSummary: {example.get('summary', '')}"
    return {"text": text}


def process_func_format_2_messages(example, tokenizer, max_length):
    """
    格式2：Conversational Language Modeling（对话语言建模）- 推荐用于 SFT
    
    数据格式：{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    
    适用场景：
    - 聊天模型训练（最推荐）
    - 对话任务
    - 需要自动应用 chat_template 的场景
    
    特点：
    - SFTTrainer 会自动应用 chat_template
    - SFTTrainer 会自动识别 assistant 回复部分
    - SFTTrainer 会自动处理 Loss Masking
    - 这是最推荐的数据格式
    
    SFTTrainer 内部处理流程：
    1. 检测到 messages 格式
    2. 自动调用 tokenizer.apply_chat_template(messages)
    3. 自动识别 assistant 回复的开始位置
    4. 自动将 prompt 部分的 labels 设为 -100
    5. 自动将 assistant 回复部分的 labels 设为对应的 input_ids
    
    相比 trainer_l.py：
    - trainer_l.py 需要手动调用 apply_chat_template
    - trainer_l.py 需要手动计算 prompt_len
    - trainer_l.py 需要手动设置 labels
    - SFTTrainer 全部自动化处理！
    
    Args:
        example: 包含 'dialogue' 和 'summary' 的数据样本
        tokenizer: 分词器实例（必须支持 chat_template）
        max_length: 最大序列长度
        
    Returns:
        dict: 包含 'messages' 字段的字典
    """
    # 构建 messages 格式
    messages = [
        {
            "role": "user",
            "content": f"Please Summarize: {example.get('dialogue', '')}"
        },
        {
            "role": "assistant",
            "content": example.get('summary', '')
        }
    ]
    
    # 直接返回 messages，SFTTrainer 会自动处理
    # 不需要手动调用 apply_chat_template！
    # 不需要手动计算 prompt_len！
    # 不需要手动设置 labels！
    return {"messages": messages}


def process_func_format_3_prompt_completion(example, tokenizer, max_length):
    """
    格式3：Standard Prompt-Completion（标准提示-完成）
    
    数据格式：{"prompt": "提示文本", "completion": "完成文本"}
    
    适用场景：
    - 简单的问答任务
    - 不需要对话格式的场景
    - 需要明确区分 prompt 和 completion 的场景
    
    特点：
    - SFTTrainer 会自动识别 completion 部分
    - 只对 completion 部分计算 Loss
    - 不需要 chat_template
    
    SFTTrainer 内部处理流程：
    1. 检测到 prompt 和 completion 字段
    2. 自动拼接：prompt + completion
    3. 自动识别 completion 的开始位置
    4. 自动将 prompt 部分的 labels 设为 -100
    5. 自动将 completion 部分的 labels 设为对应的 input_ids
    
    Args:
        example: 包含 'dialogue' 和 'summary' 的数据样本
        tokenizer: 分词器实例
        max_length: 最大序列长度
        
    Returns:
        dict: 包含 'prompt' 和 'completion' 字段的字典
    """
    prompt = f"Please Summarize: {example.get('dialogue', '')}"
    completion = example.get('summary', '')
    
    # 直接返回 prompt 和 completion，SFTTrainer 会自动处理
    return {"prompt": prompt, "completion": completion}


def process_func_format_4_conversational_prompt_completion(example, tokenizer, max_length):
    """
    格式4：Conversational Prompt-Completion（对话提示-完成）
    
    数据格式：
    {
        "prompt": [{"role": "user", "content": "..."}],
        "completion": [{"role": "assistant", "content": "..."}]
    }
    
    适用场景：
    - 需要更细粒度控制的对话任务
    - 需要分别处理 prompt 和 completion 的 chat_template 的场景
    - 复杂的多轮对话任务
    
    特点：
    - 结合了对话格式和 prompt-completion 的灵活性
    - 可以对 prompt 和 completion 分别应用 chat_template
    - 提供了最大的控制灵活性
    
    SFTTrainer 内部处理流程：
    1. 检测到 prompt 和 completion 都是列表格式
    2. 对 prompt 应用 chat_template（如果包含 role）
    3. 对 completion 应用 chat_template（如果包含 role）
    4. 拼接处理后的 prompt 和 completion
    5. 自动识别 completion 的开始位置
    6. 自动处理 Loss Masking
    
    Args:
        example: 包含 'dialogue' 和 'summary' 的数据样本
        tokenizer: 分词器实例（必须支持 chat_template）
        max_length: 最大序列长度
        
    Returns:
        dict: 包含 'prompt' 和 'completion' 字段的字典（都是列表格式）
    """
    prompt = [
        {
            "role": "user",
            "content": f"Please Summarize: {example.get('dialogue', '')}"
        }
    ]
    completion = [
        {
            "role": "assistant",
            "content": example.get('summary', '')
        }
    ]
    
    # 直接返回 prompt 和 completion（列表格式），SFTTrainer 会自动处理
    return {"prompt": prompt, "completion": completion}


def process_func_simple(example, tokenizer, max_length):
    """
    简化的数据预处理函数（用于 SFTTrainer）- 使用格式2（messages），他就是前面的函数的翻版
    
    这是本教程主要使用的格式，因为它最适合 SFT 训练。
    
    相比 trainer_l.py 中的 process_func，这个版本更简单：
    - 不需要手动计算 prompt_len
    - 不需要手动设置 labels（SFTTrainer 会自动处理）
    - 只需要返回 messages 格式即可
    
    这是 SFTTrainer 的优势：把复杂的 Loss Masking 逻辑交给训练器处理。
    
    Args:
        example: 包含 'dialogue' 和 'summary' 字段的数据样本
        tokenizer: 分词器实例（必须包含 chat_template）
        max_length: 最大序列长度
        
    Returns:
        dict: 包含 'messages' 字段的字典（SFTTrainer 会自动处理其他字段）
        
    Raises:
        ValueError: 如果 tokenizer 不支持 chat_template
    """
    # 检查 tokenizer 是否支持 chat_template
    if not hasattr(tokenizer, 'chat_template') or tokenizer.chat_template is None:
        raise ValueError(
            "tokenizer 不支持 chat_template。"
            "本代码要求使用 chat_template 进行格式化，请确保使用支持 chat_template 的模型和分词器。"
        )
    
    # ========================================================================
    # 步骤1：构建 messages 格式（格式2：Conversational Language Modeling）
    # ========================================================================
    messages = [
        {
            "role": "user",
            "content": f"Please Summarize: {example.get('dialogue', '')}"
        },
        {
            "role": "assistant",
            "content": example.get('summary', '')
        }
    ]
    
    # ========================================================================
    # 步骤2：直接返回 messages，SFTTrainer 会自动处理
    # ========================================================================
    # SFTTrainer 内部会自动：
    # 1. 应用 chat_template
    # 2. 进行分词
    # 3. 识别 assistant 回复的开始位置
    # 4. 设置 labels（prompt 部分为 -100，completion 部分为 input_ids）
    # 5. 处理截断和填充
    # 
    # 我们不需要做任何这些工作！
    return {"messages": messages}


def get_max_length(model):
    """
    获取模型支持的最大序列长度
    
    需求：统一训练最大长度为 2048（忽略模型自身更长的配置）。
    """
    max_length = 2048
    print(f"固定使用最大长度: {max_length}")
    return max_length


def preprocess_dataset_simple(tokenizer, max_length, seed, dataset, data_format="messages"):
    """
    简化的数据集预处理流程（用于 SFTTrainer）
    
    支持两种数据格式：
    1. "messages"（格式2：Conversational Language Modeling）- 默认格式
       返回格式：{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    2. "conversational_prompt_completion"（格式4：Conversational Prompt-Completion）
       返回格式：{"prompt": [{"role": "user", "content": "..."}], "completion": [{"role": "assistant", "content": "..."}]}
    
    Args:
        tokenizer: 分词器实例
        max_length: 最大序列长度（SFTTrainer 会自动处理）
        seed: 随机种子
        dataset: 原始数据集
        data_format: 数据格式选择，可选 "messages" 或 "conversational_prompt_completion"
        
    Returns:
        预处理后的数据集，格式取决于 data_format 参数
        
    Raises:
        ValueError: 如果 tokenizer 不支持 chat_template
    """
    print("开始预处理数据集（用于 SFTTrainer）...")
    print("="*80)
    print(f"使用数据格式: {data_format}")
    print("重要：Loss Masking 将由 SFTTrainer 自动处理")
    print("="*80)

    if data_format == "messages":
        # 格式2：Conversational Language Modeling
        # 检查 tokenizer 是否支持 chat_template
        if not hasattr(tokenizer, 'chat_template') or tokenizer.chat_template is None:
            raise ValueError(
                "tokenizer 不支持 chat_template。"
                "messages 格式要求使用 chat_template 进行格式化，请确保使用支持 chat_template 的模型和分词器。"
            )
        
        print(f"\n使用 chat_template 格式化提示")
        print(f"chat_template 类型: {type(tokenizer.chat_template).__name__}")
        
        # 使用 process_func_simple 处理数据（messages 格式）
        _processing_function = partial(
            process_func_simple,
            tokenizer=tokenizer,
            max_length=max_length
        )
        
        # 对数据集进行映射处理
        dataset = dataset.map(
            _processing_function,
            remove_columns=['id', 'topic', 'dialogue', 'summary'],  # 移除原始列
        )
        
        # 打乱数据顺序
        dataset = dataset.shuffle(seed=seed)
        
        print("\n数据集预处理完成。")
        print(f"  每个样本包含: messages（对话格式）")
        print(f"  格式示例: {{'messages': [{{'role': 'user', 'content': '...'}}, {{'role': 'assistant', 'content': '...'}}]}}")
        print(f"  SFTTrainer 会自动应用 chat_template 和处理 Loss Masking")
        
    elif data_format == "conversational_prompt_completion":
        # 格式4：Conversational Prompt-Completion
        print(f"\n使用 Conversational Prompt-Completion 格式")
        print(f"注意：此格式不需要 chat_template_path 和 assistant_only_loss")
        
        # 使用 process_func_format_4_conversational_prompt_completion 处理数据
        _processing_function = partial(
            process_func_format_4_conversational_prompt_completion,
            tokenizer=tokenizer,
            max_length=max_length
        )
        
        # 对数据集进行映射处理
        dataset = dataset.map(
            _processing_function,
            remove_columns=['id', 'topic', 'dialogue', 'summary'],  # 移除原始列
        )
        
        # 打乱数据顺序
        dataset = dataset.shuffle(seed=seed)
        
        print("\n数据集预处理完成。")
        print(f"  每个样本包含: prompt 和 completion（对话提示-完成格式）")
        print(f"  格式示例: {{'prompt': [{{'role': 'user', 'content': '...'}}], 'completion': [{{'role': 'assistant', 'content': '...'}}]}}")
        print(f"  SFTTrainer 会自动处理 Loss Masking（使用默认配置）")
        
    else:
        raise ValueError(
            f"不支持的数据格式: {data_format}。"
            f"支持的数据格式: 'messages' 或 'conversational_prompt_completion'"
        )
    
    return dataset


# ============================================================================
# 第三部分：模型训练主函数（使用 SFTTrainer + Flash Attention）
# ============================================================================

def train_with_sfttrainer(
    model_path="Qwen/Qwen3-8B",
    dataset_name="neil-code/dialogsum-test",
    output_dir="./peft-dialogue-summary-training/sfttrainer-checkpoint",
    seed=42,
    max_steps=2000,
    use_flash_attention=True,
    data_format="messages",
):
    """
    使用 TRL SFTTrainer 和 Flash Attention 进行模型训练的主函数
    
    这是 trainer_l.py 的现代化升级版本，展示了如何使用：
    1. SFTTrainer 简化训练流程
    2. 使用 SFTConfig.assistant_only_loss=True 实现完全自动化的 Loss Masking（messages 格式）以及使用completation_only_loss=True(default) 实现完全自动化的 Loss Masking（conversational_prompt_completion 格式）
    3. Flash Attention 加速训练
    
    【数据格式说明】
    
    支持两种数据格式：
    1. "messages"（格式2：Conversational Language Modeling）- 默认格式
       - 使用 assistant_only_loss=True 和 chat_template_path (因为并不是所有模型都支持 generation 关键字，需要手工重新构建jinja文件)
       - 完全自动化的 Loss Masking
       - 适用于大多数对话任务
       
    2. "conversational_prompt_completion"（格式4：Conversational Prompt-Completion）
       - 不使用 assistant_only_loss（在默认配置中 assistant_only_loss=False，completation_only_loss=True，这就符合该任务的需求）
       - 不使用 chat_template_path
       - 非多轮对话任务
    
    Args:
        model_path: 预训练模型路径
        dataset_name: 数据集名称
        output_dir: 模型保存目录
        seed: 随机种子
        max_steps: 最大训练步数
        use_flash_attention: 是否使用 Flash Attention（如果可用）
        data_format: 数据格式选择，可选 "messages" 或 "conversational_prompt_completion"，默认为 "messages"
        
    Raises:
        ValueError: 如果 tokenizer 不支持 chat_template
    """
    
    # ========================================================================
    # 步骤1：设置随机种子（确保实验可复现）
    # ========================================================================
    set_seed(seed)
    print(f"设置随机种子: {seed}")
    
    # ========================================================================
    # 步骤2：设置缓存目录（数据集和模型）
    # ========================================================================
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
    
    # ========================================================================
    # 步骤3：加载数据集
    # ========================================================================
    print("\n" + "="*80)
    print("步骤1：加载数据集")
    print("="*80)
    print(f"数据集名称: {dataset_name}")
    
    dataset = load_dataset(dataset_name, cache_dir=str(dataset_cache_dir))
    
    print(f"数据集信息: {dataset}")
    print(f"训练集样本数: {len(dataset['train'])}")
    print(f"验证集样本数: {len(dataset['validation'])}")
    
    print("\n原始数据示例:")
    print(dataset['train'][0])
    
    # ========================================================================
    # 步骤4：配置量化参数（用于节省显存）
    # ========================================================================
    print("\n" + "="*80)
    print("步骤2：配置量化参数")
    print("="*80)
    compute_dtype = getattr(torch, "bfloat16")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,  # 使用 4-bit 量化
        bnb_4bit_quant_type="nf4",  # 使用 NF4 量化类型
        bnb_4bit_compute_dtype=compute_dtype,  # 计算时使用 bfloat16
    )
    print("量化配置完成：4-bit NF4 量化，计算精度 bfloat16")
    
    # ========================================================================
    # 步骤5：加载预训练模型和分词器
    # ========================================================================
    print("\n" + "="*80)
    print("步骤3：加载预训练模型和分词器")
    print("="*80)
    print(f"模型路径: {model_path}")
    print(f"模型缓存目录: {model_cache_dir}")
    
    # ========================================================================
    # 【Flash Attention 配置】
    # ========================================================================
    # 如果使用 Flash Attention，需要设置 attn_implementation
    attn_implementation = None
    if use_flash_attention and FLASH_ATTENTION_AVAILABLE:
        attn_implementation = "flash_attention_2"
        print("\n✓ 使用 Flash Attention 2")
        print("  优势：")
        print("    - 训练速度提升 2-3 倍")
        print("    - 显存节省 50-80%")
        print("    - 支持更长序列")
    elif use_flash_attention and not FLASH_ATTENTION_AVAILABLE:
        print("\n⚠️  Flash Attention 不可用，将使用标准注意力")
        print("  提示：可以安装 flash-attn 来启用 Flash Attention")
        attn_implementation = "sdpa"  # 使用 PyTorch 的 SDPA（Scaled Dot Product Attention）
    else:
        attn_implementation = "sdpa"
        print("\n使用标准注意力（SDPA）")
    
    # 加载模型（使用量化配置和 Flash Attention）
    original_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        cache_dir=str(model_cache_dir),
        dtype=compute_dtype,
        device_map="auto",
        quantization_config=quant_config,
        attn_implementation=attn_implementation,  # 设置注意力实现
    )
    print("模型加载完成")
    
    # 加载训练用的分词器
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        cache_dir=str(model_cache_dir),
        trust_remote_code=True,
        padding_side="right",  # 训练时：Padding 在右侧（SFT 训练必须）
        add_eos_token=True,
    )
    # 设置 pad_token（如果不存在）
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print("训练分词器加载完成")
    
    # 检查 chat_template 支持情况（必须支持）
    print("\n" + "-"*80)
    print("检查 chat_template 支持情况")
    print("-"*80)
    if not hasattr(tokenizer, 'chat_template') or tokenizer.chat_template is None:
        raise ValueError(
            "tokenizer 不支持 chat_template。"
            "本代码要求使用 chat_template 进行格式化，请确保使用支持 chat_template 的模型和分词器。"
        )
    
    print("✓ tokenizer 支持 chat_template")
    print(f"  chat_template 类型: {type(tokenizer.chat_template).__name__}")
    
    # 打印一个格式化示例（Qwen3-8B 的 chat_template）
    try:
        test_messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        tokenized_output = tokenizer.apply_chat_template(
            test_messages, 
            add_generation_prompt=False,
            return_assistant_tokens_mask=True,
            return_dict=True
        )
        print("Default Template - No Assistant Tokens Masked:\n")

        print(tokenized_output['input_ids'])
        print(tokenized_output['assistant_masks'])
        print(f"Assistant mask sum: {sum(tokenized_output['assistant_masks'])}")
            
    except Exception as e:
        print(f"  警告: 无法测试默认的 chat_template: {e}")
    print("-"*80)

    # 打印另一个格式化示例（自定义的chat_template）

    # Load the all assistant template
    with open('models/models--Qwen--Qwen3-8B/.no_exist/b968826d9c46dd6066d109eabc6255188de91218/final_assistant.jinja', 'r') as f:
        all_assistant_template = f.read()

    tokenizer.chat_template = all_assistant_template
    
    try:
        test_messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        tokenized_output = tokenizer.apply_chat_template(
            test_messages, 
            add_generation_prompt=False,
            return_assistant_tokens_mask=True,
            return_dict=True
        )
        print("Custom Template - All Assistant Tokens Masked:\n")
        print(tokenized_output['input_ids'])
        print(tokenized_output['assistant_masks'])
        print(f"Assistant mask sum: {sum(tokenized_output['assistant_masks'])}")

    except Exception as e:
        print(f"  警告: 无法测试修复后的 chat_template: {e}")
    print("-"*80)


    # 加载评估用的分词器
    eval_tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        cache_dir=str(model_cache_dir),
        add_bos_token=True,
        trust_remote_code=True,
    )
    if eval_tokenizer.pad_token_id is None:
        eval_tokenizer.pad_token_id = eval_tokenizer.eos_token_id
    print("评估分词器加载完成")
    
    # ========================================================================
    # 步骤6：获取模型最大序列长度并预处理数据集
    # ========================================================================
    print("\n" + "="*80)
    print("步骤4：预处理数据集（简化版）")
    print("="*80)
    max_length = 2048
    print(f"使用最大序列长度: {2048}")
    
    # ========================================================================
    # 【核心知识点】SFTTrainer 的数据格式要求
    # ========================================================================
    # SFTTrainer 支持四种数据格式，我们使用 "messages" 格式（推荐）
    # 
    # 数据格式对比：
    # 1. "messages" - 推荐用于对话任务
    #    - 格式：{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}
    #    - SFTTrainer 会自动应用 chat_template
    #    - SFTTrainer 会自动处理 Loss Masking
    #
    # 2. "text" - 适用于预训练
    #    - 格式：{"text": "完整文本"}
    #    - 会对所有 token 计算 Loss（不适用于 SFT）
    #
    # 3. "prompt_completion" - 适用于简单问答
    #    - 格式：{"prompt": "...", "completion": "..."}
    #    - SFTTrainer 会自动识别 completion 部分
    #
    # 4. "conversational_prompt_completion" - 适用于复杂对话
    #    - 格式：{"prompt": [...], "completion": [...]}
    #    - 结合了对话格式和 prompt-completion 的灵活性
    # ========================================================================
    
    # 预处理训练集和验证集（根据 data_format 选择不同的格式）
    # 支持两种格式：
    # 1. "messages" - 使用 process_func_simple（messages 格式）
    # 2. "conversational_prompt_completion" - 使用 process_func_format_4_conversational_prompt_completion
    train_dataset = preprocess_dataset_simple(tokenizer, max_length, seed, dataset['train'], data_format=data_format)
    eval_dataset = preprocess_dataset_simple(tokenizer, max_length, seed, dataset['validation'], data_format=data_format)
    
    print(f"\n预处理后的训练集: {train_dataset}")
    print(f"预处理后的验证集: {eval_dataset}")
    
    # 打印一个预处理后的样本示例
    print("\n预处理后的数据示例:")
    sample = train_dataset[0]
    print(f"  样本键: {list(sample.keys())}")
    if data_format == "messages":
        print(f"  messages 格式: {sample['messages']}")
    elif data_format == "conversational_prompt_completion":
        print(f"  conversational_prompt_completion 格式: {sample['prompt']}, {sample['completion']}")   
    else:
        raise ValueError(f"不支持的数据格式: {data_format}")
    print(f"  说明: SFTTrainer 会自动处理原始数据，得到合适的格式")
    
    # ========================================================================
    # 步骤7：配置训练参数（使用 SFTConfig）
    # ========================================================================
    print("\n" + "="*80)
    print("步骤5：配置训练参数（使用 SFTConfig）")
    print("="*80)
    
    # ========================================================================
    # 【重要说明】SFTConfig vs TrainingArguments
    # ========================================================================
    # SFTTrainer 使用 SFTConfig 来配置训练参数。
    # SFTConfig 是专门为 SFT 训练设计的配置类，继承自 TrainingArguments，
    # 但添加了一些 SFT 特定的参数和优化。
    #
    # 主要区别：
    # - SFTConfig 针对 SFT 训练进行了优化
    # - 提供了更好的默认值和参数验证
    # - 与 SFTTrainer 的集成更加紧密
    # ========================================================================
    
    # ========================================================================
    # 【核心知识点】Loss Masking 的完全自动化方案
    # ========================================================================
    # 
    # 使用 assistant_only_loss=True 或者 completion_only_loss=True
    #   - 不设置 data_collator（留作空白）
    #   - SFTTrainer 会自动处理 Loss Masking
    #   - 完全自动化，用户无需关心 DataCollator
    #   - 这是目前优化的最好的 SFT 方法
    #
    # ========================================================================
    
    # ========================================================================
    # 【核心知识点】根据数据格式设置不同的配置
    # ========================================================================
    # 
    # 1. "messages" 格式：
    #    - 使用 assistant_only_loss=True（完全自动化）
    #    - 使用 chat_template_path（自定义模板）
    #    - 这是推荐的最优方法
    #
    # 2. "conversational_prompt_completion" 格式：
    #    - 不使用 assistant_only_loss（使用默认配置）
    #    - 不使用 chat_template_path（使用默认模板）
    #    - 适用于需要更细粒度控制的场景
    #
    # ========================================================================
    
    if data_format == "messages":
        print("\n【使用 assistant_only_loss=True 适用于多轮对话任务】")
        print("  - 目的：完全自动化的 SFT，用户无需关心 DataCollator")
        print("  - 注意：不需要手动设置 DataCollator，留作空白即可，但对于不支持generation关键字的模型，需要手工重新构建jinja文件")
        
        training_args = SFTConfig(
            # 输出目录
            output_dir=output_dir,
            
            # 训练相关参数
            max_steps=max_steps,  # 最大训练步数
            per_device_train_batch_size=10,  # 每个设备的训练批次大小
            gradient_accumulation_steps=1,  # 梯度累积步数
            learning_rate=2e-4,  # 学习率
            warmup_steps=1,  # 预热步数
            
            # 优化器配置
            optim="paged_adamw_8bit",  # 使用分页的 8-bit AdamW 优化器（节省显存）
            
            # 评估相关参数
            do_eval=True,  # 是否进行评估
            eval_strategy="steps",  # 评估策略：按步数评估
            eval_steps=100,  # 每多少步评估一次
            
            # 保存相关参数
            save_strategy="steps",  # 保存策略：按步数保存
            save_steps=500,  # 每多少步保存一次
            overwrite_output_dir=True,  # 是否覆盖输出目录
            
            # 日志相关参数
            logging_steps=100,  # 每多少步记录一次日志
            logging_dir="./logs",  # 日志目录
            report_to="none",  # 不向任何平台报告
            
            # 其他参数
            group_by_length=True,  # 按长度分组（提高训练效率）
            
            # 【核心参数】assistant_only_loss：是否只对 assistant 回复计算 Loss
            # - True：完全自动化，无需手动设置 DataCollator
            # - SFTTrainer 会自动识别 assistant 回复的开始位置
            # - SFTTrainer 会自动将 prompt 部分的 labels 设为 -100
            assistant_only_loss=True,
            max_length=2048,  # 最大序列长度
            gradient_checkpointing=True,
            packing=False,  # 不打包多个样本到一个序列中（保持简单）
            chat_template_path="models/models--Qwen--Qwen3-8B/.no_exist/b968826d9c46dd6066d109eabc6255188de91218/final_assistant.jinja",
            bf16=True,
        )
        
    elif data_format == "conversational_prompt_completion":
        print("\n【使用 conversational_prompt_completion 格式（格式4）】")
        print("  - 目的：使用默认配置，不设置 assistant_only_loss 和 chat_template_path 也就是completation_only_loss=True")
        print("  - 注意：使用 SFTTrainer 的默认 Loss Masking 行为")
        
        training_args = SFTConfig(
            # 输出目录
            output_dir=output_dir,
            
            # 训练相关参数
            max_steps=max_steps,  # 最大训练步数
            per_device_train_batch_size=10,  # 每个设备的训练批次大小
            gradient_accumulation_steps=1,  # 梯度累积步数
            learning_rate=2e-4,  # 学习率
            warmup_steps=1,  # 预热步数
            
            # 优化器配置
            optim="paged_adamw_8bit",  # 使用分页的 8-bit AdamW 优化器（节省显存）
            
            # 评估相关参数
            do_eval=True,  # 是否进行评估
            eval_strategy="steps",  # 评估策略：按步数评估
            eval_steps=100,  # 每多少步评估一次
            
            # 保存相关参数
            save_strategy="steps",  # 保存策略：按步数保存
            save_steps=500,  # 每多少步保存一次
            overwrite_output_dir=True,  # 是否覆盖输出目录
            
            # 日志相关参数
            logging_steps=100,  # 每多少步记录一次日志
            logging_dir="./logs",  # 日志目录
            report_to="none",  # 不向任何平台报告
            
            # 其他参数
            group_by_length=True,  # 按长度分组（提高训练效率）
            
            # 【核心参数】对于 conversational_prompt_completion 格式：
            # - 不设置 assistant_only_loss（使用默认值）
            # - 不设置 chat_template_path（使用默认模板）
            # - SFTTrainer 会根据 prompt-completion 格式自动处理 Loss Masking
            max_length=2048,  # 最大序列长度
            gradient_checkpointing=True,
            packing=False,  # 不打包多个样本到一个序列中，packing 是有风险的
            bf16=True,
        )
        
    else:
        raise ValueError(
            f"不支持的数据格式: {data_format}。"
            f"支持的数据格式: 'messages' 或 'conversational_prompt_completion'"
        )
    
    print("训练参数配置完成（使用 SFTConfig）")
    print(f"  - 最大训练步数: {max_steps}")
    print(f"  - 批次大小: {training_args.per_device_train_batch_size}")
    print(f"  - 学习率: {training_args.learning_rate}")
    print(f"  - 输出目录: {output_dir}")
    print(f"  - 数据格式: {data_format}")
    if data_format == "messages":
        print(f"  - assistant_only_loss: True")
        print("    ✓ 使用完全自动化的 Loss Masking")
        print(f"  - chat_template_path: 已设置（自定义模板）")
    elif data_format == "conversational_prompt_completion":
        print(f"  - assistant_only_loss: 未设置（使用默认配置assistant_only_loss = False，completation_only_loss=True）")
        print("    ✓ 使用 SFTTrainer 默认的 Loss Masking")
        print(f"  - chat_template_path: 未设置（使用默认模板）")
    if use_flash_attention and FLASH_ATTENTION_AVAILABLE:
        print(f"  - 注意力实现: Flash Attention 2（加速训练）")
    else:
        print(f"  - 注意力实现: 标准注意力（SDPA）")
    
    # ========================================================================
    # 步骤8：配置 PEFT/LoRA
    # ========================================================================
    print("\n" + "="*80)
    print("步骤6：配置 PEFT/LoRA")
    print("="*80)
    
    # 定义 LoRA 配置
    lora_config = LoraConfig(
        r=64,  # LoRA 的秩（rank），控制适配器的大小
        lora_alpha=16,  # LoRA 的 alpha 参数，用于缩放
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 要应用 LoRA 的模块
        bias="none",  # 不对 bias 进行训练
        lora_dropout=0.01,  # LoRA dropout 率
        task_type="CAUSAL_LM",  # 任务类型：因果语言模型
    )
    print("LoRA 配置:")
    print(f"  - Rank (r): {lora_config.r}")
    print(f"  - Alpha: {lora_config.lora_alpha}")
    print(f"  - Target modules: {lora_config.target_modules}")
    
    # 准备模型进行量化训练
    original_model = prepare_model_for_kbit_training(original_model)
    print("模型已准备进行量化训练")
    
    # 禁用缓存（训练时需要）
    original_model.config.use_cache = False
    print("已禁用模型缓存（训练模式）")
    
    # ========================================================================
    # 步骤9：配置数据整理器（DataCollator）
    # ========================================================================
    print("\n" + "="*80)
    print("步骤7：配置数据整理器")
    print("="*80)
    
    print("  说明：")
    print("    - 在 SFTConfig 中已设置 assistant_only_loss=True / completation_only_loss=True")
    print("    - 不需要手动设置 DataCollator，留作空白即可")
    print("    - SFTTrainer 会自动处理 Loss Masking")
    print("  优势：")
    print("    - 代码更简洁，无需关心 DataCollator 的配置")
    print("    - 完全自动化，减少出错可能")
    print("    - 最佳实践，推荐用于生产环境")
    data_collator = None
    
    # ========================================================================
    # 步骤10：创建 SFTTrainer 实例
    # ========================================================================
    print("\n" + "="*80)
    print("步骤8：创建 SFTTrainer 实例")
    print("="*80)
    
    print("SFTTrainer 的优势：")
    print("  ✓ 自动处理 Loss Masking（不需要手动计算 prompt_len）")
    print("  ✓ 自动处理数据格式化（基于 chat_template）")
    print("  ✓ 内置最佳实践和优化")
    print("  ✓ 与 Hugging Face 生态系统深度集成")
    
    # ========================================================================
    # 【核心知识点】SFTTrainer 的内部工作原理
    # ========================================================================
    # 
    # 当我们创建 SFTTrainer 时，它会自动完成以下工作：
    #
    # 1. 数据格式识别和转换：
    #    - 检测数据集中的字段（text/messages/prompt/completion）
    #    - 如果是 messages 格式，自动调用 tokenizer.apply_chat_template()
    #    - 如果是 prompt-completion 格式，自动拼接并识别边界
    #
    # 2. Loss Masking 自动化（这是与普通 Trainer 的核心区别）：
    #    - 自动识别 prompt 和 completion 的边界
    #    - 自动将 prompt 部分的 labels 设为 -100（不计算 Loss）
    #    - 自动将 completion 部分的 labels 设为对应的 input_ids（计算 Loss）
    #    - 不需要手动计算 prompt_len（这是 trainer_l.py 中需要做的）
    #
    # 3. 数据预处理自动化：
    #    - 自动进行分词（tokenization）
    #    - 自动处理截断（truncation）和填充（padding）
    #    - 自动创建 attention_mask
    #    - 自动处理特殊 token（EOS、BOS 等）
    #
    # 4. PEFT 集成自动化：
    #    - 如果提供了 peft_config，自动调用 get_peft_model()
    #    - 自动准备模型进行量化训练
    #    - 自动管理可训练参数
    #
    # 5. 数据整理（Data Collation）自动化：
    #    - 如果提供了 data_collator，使用提供的 collator
    #    - 否则，根据数据格式自动选择合适的默认 collator
    #    - 自动处理批次内的 padding
    #    - 自动处理 labels 的 padding（使用 -100）
    #
    # 对比 trainer_l.py：
    # - trainer_l.py 需要手动实现上述所有步骤（100+ 行代码）
    # - SFTTrainer 自动完成所有步骤（只需要配置参数）
    #
    # ========================================================================
    
    # ========================================================================
    # 【重要说明】SFTTrainer 的参数配置
    # ========================================================================
    # 
    # dataset_text_field: 指定数据集中的文本字段名
    #   - 如果使用 "messages" 格式，SFTTrainer 会自动识别，不需要设置此参数
    #   - 如果使用 "text" 格式，设置为 "text"
    #   - 如果使用 "prompt_completion" 格式，SFTTrainer 会自动识别 prompt 和 completion
    #
    # packing: 是否将多个样本打包到一个序列中
    #   - False: 每个样本独立处理（推荐，更简单）
    #   - True: 将多个短样本打包（更高效，但更复杂）
    #
    # peft_config: LoRA 配置
    #   - SFTTrainer 会自动应用，不需要手动调用 get_peft_model()
    #   - 这是与 trainer_l.py 的另一个重要区别
    #
    # data_collator: 数据整理器（可选）
    #   - 不设置（None），使用 SFTConfig.assistant_only_loss=True
    #     * 完全自动化，用户无需关心 DataCollator
    #     * 这是目前优化的最好的 SFT 方法
    #
    # ========================================================================
    
    
    trainer = SFTTrainer(
        model=original_model,  # 要训练的模型（已准备好量化训练，LoRA 将由 SFTTrainer 自动应用）
        train_dataset=train_dataset,  # 训练数据集（格式取决于 data_format）
        eval_dataset=eval_dataset,  # 验证数据集（格式取决于 data_format）
        args=training_args,  # 训练参数
        peft_config=lora_config,  # LoRA 配置（SFTTrainer 会自动应用，不需要手动调用 get_peft_model）
    )
    
    print("SFTTrainer 创建完成")
    print(f"  - 模型: {type(original_model).__name__}")
    print(f"  - 训练样本数: {len(train_dataset)}")
    print(f"  - 验证样本数: {len(eval_dataset)}")
    print(f"  - 数据整理器: 未设置")
    
    # 打印可训练参数信息
    print("\n可训练参数信息:")
    trainer.model.print_trainable_parameters()
    
    # ========================================================================
    # 步骤11：开始训练
    # ========================================================================
    print("\n" + "="*80)
    print("步骤9：开始训练")
    print("="*80)
    
    # 清空 CUDA 缓存
    torch.cuda.empty_cache()
    
    # 开始训练
    print("训练开始...")
    if use_flash_attention and FLASH_ATTENTION_AVAILABLE:
        print("使用 Flash Attention 2 加速训练...")
    trainer.train()
    print("训练完成！")
    
    # ========================================================================
    # 步骤12：效果验证（训练后的模型生成）
    # ========================================================================
    print("\n" + "="*80)
    print("步骤10：效果验证") # 注意：这里你的注释是步骤10，但标题是步骤12，保持一致即可
    print("="*80)

    # 切换到评估模式
    trainer.model.eval()

    # 拿一条验证数据
    test_sample = dataset['validation'][0]
    prompt_text = f"Please Summarize: {test_sample['dialogue']}"
    messages = [{"role": "user", "content": prompt_text}]

    print("\n测试样本信息:")
    print(f"User 输入:\n{prompt_text[:200]}...")
    print(f"\n标准答案:\n{test_sample['summary']}")

    # 格式化输入（使用 chat_template）
    formatted_input = eval_tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt"
    )

    # 将输入移到模型所在的设备
    device = next(trainer.model.parameters()).device

    if isinstance(formatted_input, dict):
        inputs = {k: v.to(device) for k, v in formatted_input.items()}
        input_length = inputs["input_ids"].shape[1]
    else:
        inputs = {"input_ids": formatted_input.to(device)}
        input_length = inputs["input_ids"].shape[1]

    print(f"\n开始生成（设备: {device}）...")

    # ========================================================================
    # 关键修改点：使用 torch.autocast
    # ========================================================================
    # 获取模型的数据类型，确保 autocast 使用正确的类型
    model_dtype = trainer.model.dtype

    # 生成回复
    with torch.no_grad():
        # 使用 autocast 来自动处理混合精度，解决类型不匹配问题
        with torch.autocast(device_type=device.type, dtype=model_dtype):
            outputs = trainer.model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                pad_token_id=eval_tokenizer.pad_token_id,
                eos_token_id=eval_tokenizer.eos_token_id,
            )

    # 解码生成的回复
    generated_ids = outputs[0][input_length:]
    response = eval_tokenizer.decode(generated_ids, skip_special_tokens=True)

    print(f"\n模型训练后生成:\n{response}")
    print("="*80)

    # ========================================================================
    # 步骤13：清理资源
    # ========================================================================
    print("\n" + "="*80)
    print("步骤11：清理资源") # 同样，注意注释编号
    print("="*80)

    # 释放内存
    del trainer
    torch.cuda.empty_cache()
    print("资源清理完成")

    return output_dir, dataset  # 返回dataset以便后续使用



# ============================================================================
# 第四部分：主程序入口
# ============================================================================

if __name__ == "__main__":
    """
    主程序入口
    
    本代码展示了如何使用 TRL SFTTrainer 和 Flash Attention 进行模型微调。
    这是 trainer_l.py 的现代化升级版本。
    
    主要改进：
    1. 使用 SFTTrainer 简化训练流程
    2. 支持两种数据格式选择：
       - "messages"：使用 assistant_only_loss=True 和 chat_template_path（推荐）
       - "conversational_prompt_completion"：使用默认配置
    3. 集成 Flash Attention 加速训练
    4. 代码更简洁，更易维护
    """
    
    print("="*80)
    print("TRL SFTTrainer + Flash Attention 训练教程")
    print("="*80)
    print("\n本示例展示了如何使用 Hugging Face TRL 库的 SFTTrainer 进行模型微调")
    print("这是 trainer_l.py 的现代化升级版本")
    print("\n主要特性：")
    print("  1. SFTTrainer：自动处理 Loss Masking，代码更简洁")
    print("  2. 支持两种数据格式选择：")
    print("     - messages：使用 assistant_only_loss=True 和 chat_template_path")
    print("     - conversational_prompt_completion：使用默认配置")
    print("  3. Flash Attention：加速训练 2-3 倍，节省显存 50-80%")
    print("  4. 与 Hugging Face 生态系统深度集成")
    print("\n相比 trainer_l.py 的优势：")
    print("  ✓ 不需要手动计算 prompt_len 和 labels")
    print("  ✓ 不需要复杂的 process_func 函数")
    print("  ✓ 不需要手动设置 DataCollator")
    print("  ✓ 训练速度更快（Flash Attention）")
    print("  ✓ 代码更简洁，更易维护")

    print("="*80)
    
    # ========================================================================
    # 配置参数（可以根据需要修改）
    # ========================================================================
    MODEL_PATH = "Qwen/Qwen3-8B"
    DATASET_NAME = "neil-code/dialogsum-test"
    OUTPUT_DIR = "./peft-dialogue-summary-training/sfttrainer-checkpoint"
    SEED = 42
    MAX_STEPS = 40  # 可以根据需要调整训练步数
    USE_FLASH_ATTENTION = True  # 是否使用 Flash Attention（如果可用）
    DATA_FORMAT = "messages"  # 数据格式选择："messages" 或 "conversational_prompt_completion"
    
    # ========================================================================
    # 执行训练
    # ========================================================================
    print("="*80)
    print("训练配置")
    print("="*80)
    print(f"模型路径: {MODEL_PATH}")
    print(f"数据集: {DATASET_NAME}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"最大训练步数: {MAX_STEPS}")
    print(f"使用 Flash Attention: {USE_FLASH_ATTENTION}")
    if USE_FLASH_ATTENTION:
        if FLASH_ATTENTION_AVAILABLE:
            print("  ✓ Flash Attention 可用，将加速训练")
        else:
            print("  ⚠️  Flash Attention 不可用，将使用标准注意力")
            print("     提示：可以安装 flash-attn 来启用 Flash Attention")
            print("     安装命令：pip install flash-attn")
    print(f"\n数据格式: {DATA_FORMAT}")
    if DATA_FORMAT == "messages":
        print("  【assistant_only_loss=True（推荐，最优方法）】")
        print("    - 目的：完全自动化的 SFT，用户无需关心 DataCollator")
        print("    - 配置：使用 chat_template_path 和 assistant_only_loss=True")
    elif DATA_FORMAT == "conversational_prompt_completion":
        print("  【conversational_prompt_completion 格式（格式4）】")
        print("    - 目的：使用默认配置，不设置 assistant_only_loss，此时completation_only_loss=True")
        print("    - 配置：不使用 chat_template_path 和 assistant_only_loss")
    print("="*80)
    
    try:
        output_dir, dataset = train_with_sfttrainer(
            model_path=MODEL_PATH,
            dataset_name=DATASET_NAME,
            output_dir=OUTPUT_DIR,
            seed=SEED,
            max_steps=MAX_STEPS,
            use_flash_attention=USE_FLASH_ATTENTION,
            data_format=DATA_FORMAT,
        )
        
        print("\n" + "="*80)
        print("训练成功完成！")
        print("="*80)
        print(f"模型检查点保存在: {output_dir}")
        print("\n提示：可以使用以下代码加载训练好的模型：")
        print("""
from peft import PeftModel
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

# 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B",
    quantization_config=BitsAndBytesConfig(...)
)

# 加载 LoRA 适配器
model = PeftModel.from_pretrained(
    base_model,
    "{output_dir}",
    is_trainable=False
)
        """.format(output_dir=output_dir))
        
    except Exception as e:
        print("\n" + "="*80)
        print("训练过程中出现错误：")
        print("="*80)
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {str(e)}")
        import traceback
        traceback.print_exc()

