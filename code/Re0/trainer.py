"""
================================================================================
第一章：使用 Transformers Trainer 进行模型微调
================================================================================

配套文档：01_trainer_tutorial.md

【本文件是什么？】

这是 LLM 微调教程系列的第一个代码文件，展示如何使用 Hugging Face 原生 Trainer 
进行监督微调（SFT）。本文件侧重于教学，让你理解 SFT 的核心概念。

【核心学习目标】

1. Loss Masking（★ 最重要的概念 ★）
   - SFT 只计算模型回复部分的 Loss，不计算用户问题部分
   - 通过将 prompt 部分的 labels 设为 -100 实现
   - 如果不做 Loss Masking，模型会学习"续写问题"而非"回答问题"
   
2. 量化（Quantization）
   - 使用 4-bit 量化减少模型大小，让普通显卡也能运行大模型

3. LoRA（Low-Rank Adaptation）
   - 只训练 1% 的参数，大幅降低显存需求

4. DataCollator
   - 理解不同 DataCollator 的区别和选择

【快速开始】

1. 安装依赖：
   pip install transformers datasets peft bitsandbytes accelerate

2. 运行训练：
   python trainer.py

3. 修改配置：在文件底部 __main__ 部分修改 COLLATOR_TYPE、MODEL_PATH 等参数

【DataCollator 选项】

- "seq2seq"       : 生产推荐，保留预处理好的 labels
- "visual_check"  : 学习推荐，打印第一个 batch 的详细信息
- "manual_mask"   : 简化版，便于理解核心逻辑
- "language_modeling" : 仅用于对比教学（会对所有 token 计算 Loss）

【下一步】

完成本章后，请阅读 sfttrainer.py（第二章），学习如何使用 SFTTrainer 简化训练流程。
"""


# ============================================================================
# 补充知识：DataCollator 对比
# ============================================================================
#
# | DataCollator                      | 用途       | labels 处理                    |
# |-----------------------------------|------------|--------------------------------|
# | DataCollatorForLanguageModeling   | 预训练     | 自动复制 input_ids 为 labels   |
# | DataCollatorForSeq2Seq            | SFT (推荐) | 保留预处理好的 labels          |
#
# SFT 必须使用 DataCollatorForSeq2Seq，因为它尊重你的 Loss Masking 设置！
#
# ============================================================================
# 补充知识：量化与 LoRA
# ============================================================================
#
# 量化 (Quantization)：
#   - 将模型参数从 FP32 压缩到 INT4，大小减少 87.5%，在减少存储模型参数所需的显存
#   - 使用 BitsAndBytesConfig(load_in_4bit=True)
#
# LoRA (Low-Rank Adaptation)：
#   - 冻结原模型，只训练少量适配器参数（<1%），减少训练的参数量，从而减少优化器的显存
#   - 使用 LoraConfig + get_peft_model()
#
# 详细说明请参考配套文档：01_trainer_tutorial.md
# ============================================================================

# ============================================================================
# 第一部分：库导入
# ============================================================================


# 导入核心库
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    set_seed,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from functools import partial
import torch
import transformers
from pathlib import Path


# ============================================================================
# 第二部分：数据预处理函数（包含 Loss Masking 核心逻辑）
# ============================================================================

def process_func(example, tokenizer, max_length):
    """
    不仅格式化文本，还负责制作 Labels，实现对 Prompt 部分的 Mask。
    
    这是 SFT（监督微调）训练中最关键的一步！
    
    核心概念：
    - 在 SFT 中，我们只计算模型回复部分的 Loss
    - Prompt 部分的 Label 设置为 -100（PyTorch 中默认忽略的索引）
    - 这样模型只学习如何生成回答，而不是学习如何生成问题
    
    如果不做 Loss Masking，模型实际上在做"预训练续写"而不是"指令微调"。
    
    Args:
        example: 包含 'dialogue' 和 'summary' 字段的数据样本
        tokenizer: 分词器实例（必须包含 chat_template）
        max_length: 最大序列长度
        
    Returns:
        dict: 包含 'input_ids', 'attention_mask', 'labels' 的字典
        
    Raises:
        ValueError: 如果 tokenizer 不支持 chat_template
    """
    # ========================================================================
    # 【代码块】验证 tokenizer 是否支持 chat_template
    # ========================================================================
    # 功能：检查 tokenizer 是否具备 chat_template 功能
    # 原因：本代码强制使用 chat_template 进行格式化，不支持手动拼接字符串，目前较新的语言模型基本都支持了chat_template
    #       这样可以确保训练和推理时使用相同的格式，避免格式不一致导致的性能下降
    if not hasattr(tokenizer, 'chat_template') or tokenizer.chat_template is None:
        raise ValueError(
            "tokenizer 不支持 chat_template。"
            "本代码要求使用 chat_template 进行格式化，请确保使用支持 chat_template 的模型和分词器。"
        )
    
    # ========================================================================
    # 【代码块】步骤1：构建完整的对话文本（包含 user 和 assistant 的回复）
    # ========================================================================
    # 功能：将原始数据转换为标准化的对话格式
    # 原因：使用统一的 messages 格式可以兼容不同模型的 chat_template，提高代码可移植性
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
    # 【代码块】使用 chat_template 生成完整文本
    # ========================================================================
    # 功能：将 messages 格式转换为模型期望的文本格式（包含特殊标记）
    # 原因：apply_chat_template 会自动处理 EOS Token，确保模型学会在合适位置停止生成
    #       如果手动拼接字符串，容易遗漏 EOS Token，导致模型无法正确停止生成
    # tokenize=False: 只进行字符串程度的格式化添加提示模板得到一个字符串，这样逐步进行更适合教程所需
    # add_generation_prompt=False: 训练时不需要生成提示，因为我们已经包含了完整的 assistant 回复
    text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False,  # 只格式化，不进行tokenization
        add_generation_prompt=False  # 训练时不需要生成提示（已包含完整回复）
    )
    
    # ========================================================================
    # 【代码块】步骤2：对完整文本进行分词
    # ========================================================================
    # 功能：将格式化后的文本转换为 token ID 序列
    # 原因：模型只能处理数字序列，需要将文本转换为模型词汇表中的 token ID
    # return_tensors=None: 返回列表而不是 tensor，因为后续需要手动处理 padding
    tokenized = tokenizer(
        text,
        max_length=max_length,
        truncation=True,  # 超过最大长度时截断，避免显存溢出
        return_tensors=None  # 返回列表而不是 tensor（后续需要手动处理 padding）
    )
    input_ids = tokenized["input_ids"]
    
    # ========================================================================
    # 【代码块】步骤3：计算 Loss Masking（这是 SFT 的核心！）
    # ========================================================================
    # 功能：创建 labels，只对 assistant 回复部分计算 Loss
    # 原因：在 SFT 中，我们只希望模型学习如何生成回答，而不是学习如何生成问题
    #       如果不做 Loss Masking，模型实际上在做"预训练续写"而不是"指令微调"
    # 默认所有位置都是 -100（PyTorch 中 -100 表示忽略该位置的 Loss 计算）
    labels = [-100] * len(input_ids)
    
    # ========================================================================
    # 【代码块】定位 assistant 回复的起始位置
    # ========================================================================
    # 功能：通过重新 tokenize 只有 prompt 的部分来确定 assistant 回复的起始位置
    # 原因：不同模型的 chat_template 格式不同，直接字符串匹配不可靠
    #       通过 tokenize 长度对比可以准确找到分界点，即使格式变化也能正确工作
    # 构建只有 User 部分的 prompt（用于计算 prompt 长度）
    # add_generation_prompt=True: 添加 assistant 引导符（如 <|im_start|>assistant）
    #                             这样 prompt_text 的长度就是 prompt 部分的准确长度
    user_messages = [
        {
            "role": "user",
            "content": f"Please Summarize: {example.get('dialogue', '')}"
        }
    ]
    prompt_text = tokenizer.apply_chat_template(
        user_messages,
        tokenize=False,
        add_generation_prompt=True  # 添加 assistant 引导符，确保 prompt 长度计算准确
    )
    
    # 对 prompt 部分进行分词，获取 prompt 的 token 长度
    prompt_tokenized = tokenizer(
        prompt_text,
        max_length=max_length,
        truncation=True,
        return_tensors=None
    )
    prompt_ids = prompt_tokenized["input_ids"]
    prompt_len = len(prompt_ids)  # prompt 部分的 token 数量
    
    # ========================================================================
    # 【代码块】步骤4：将 Prompt 之后的 Labels 设为原本的 input_ids
    # ========================================================================
    # 功能：只对 assistant 回复部分设置 labels，prompt 部分保持 -100
    # 原因：这样模型只学习生成 assistant 回复部分，不学习生成 prompt 部分
    #       如果对 prompt 部分也计算 Loss，模型会学习"续写问题"而不是"回答问题"
    # 处理截断情况：如果 input_ids 被截断，确保只处理实际存在的部分
    if len(input_ids) > prompt_len:
        # 从 prompt_len 开始，将 labels 设置为对应的 input_ids
        # 这样只有 assistant 回复部分会计算 Loss，prompt 部分被忽略
        labels[prompt_len:] = input_ids[prompt_len:]
    
    # ========================================================================
    # 【代码块】步骤5：返回处理后的数据
    # ========================================================================
    # 功能：返回模型训练所需的标准格式数据
    # 原因：Trainer 需要 input_ids、attention_mask 和 labels 三个字段
    #       attention_mask 标记哪些位置是真实内容（1）哪些是 padding（0）
    #       labels 中 -100 的位置不计算 Loss，其他位置计算 Loss
    return {
        "input_ids": input_ids,  # 模型的输入 token ID 序列
        "attention_mask": [1] * len(input_ids),  # 所有位置都参与 attention（尚未 padding）
        "labels": labels  # 只有 assistant 回复部分不是 -100（会计算 Loss）
    }


def get_max_length(model):
    """
    获取模型支持的最大序列长度
    
    不同模型可能使用不同的配置字段来存储最大长度信息。
    此函数尝试从多个可能的字段中获取该值。
    
    Args:
        model: 预训练模型实例
        
    Returns:
        int: 模型支持的最大序列长度，如果找不到则返回默认值 1024
    """
    # ========================================================================
    # 【代码块】尝试从多个可能的配置字段中获取最大长度
    # ========================================================================
    # 功能：从模型配置中提取最大序列长度
    # 原因：不同模型架构使用不同的字段名存储最大长度（如 GPT 用 n_positions，BERT 用 max_position_embeddings）
    #       通过遍历多个可能的字段名，可以提高代码的兼容性
    max_length = None
    # 尝试从不同的配置字段中获取最大长度（不同模型使用不同的字段名）
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            print(f"找到最大长度配置: {max_length}")
            break
    
    # ========================================================================
    # 【代码块】如果找不到配置，使用默认值
    # ========================================================================
    # 功能：当无法从配置中获取最大长度时，使用安全的默认值
    # 原因：1024 是一个常见的默认值，既能处理大多数任务，又不会导致显存溢出
    #       如果使用过大的值，可能导致显存不足；如果使用过小的值，可能截断重要信息
    if not max_length:
        max_length = 1024  # 使用默认值 1024，平衡性能和显存占用
        print(f"使用默认最大长度: {max_length}")
    
    return max_length


def preprocess_dataset(tokenizer, max_length, seed, dataset):
    """
    完整的数据集预处理流程（包含 Loss Masking）
    
    包括：
    1. 使用 chat_template 格式化提示并分词
    2. 实现 Loss Masking（只计算 assistant 回复部分的 Loss）
    3. 过滤过长样本
    4. 打乱数据顺序
    
    Args:
        tokenizer: 分词器实例（必须支持 chat_template）
        max_length: 最大序列长度
        seed: 随机种子
        dataset: 原始数据集
        
    Returns:
        预处理后的数据集（包含 input_ids, attention_mask, labels）
        
    Raises:
        ValueError: 如果 tokenizer 不支持 chat_template
    """
    # ========================================================================
    # 【代码块】打印预处理说明信息
    # ========================================================================
    # 功能：向用户说明 Loss Masking 的重要性
    # 原因：Loss Masking 是 SFT 训练的核心概念，需要让用户理解其作用
    print("开始预处理数据集...")
    print("="*80)
    print("重要：正在实现 Loss Masking（SFT 核心逻辑）")
    print("="*80)
    print("说明：")
    print("  - Prompt 部分的 labels 设置为 -100（不计算 Loss）")
    print("  - 只有 Assistant 回复部分的 labels 设置为对应的 input_ids（计算 Loss）")
    print("  - 这样模型只学习如何生成回答，而不是学习如何生成问题")
    print("="*80)
    
    # ========================================================================
    # 【代码块】验证 tokenizer 是否支持 chat_template
    # ========================================================================
    # 功能：确保 tokenizer 具备 chat_template 功能
    # 原因：本代码强制使用 chat_template，不支持手动格式化，确保训练和推理格式一致
    if not hasattr(tokenizer, 'chat_template') or tokenizer.chat_template is None:
        raise ValueError(
            "tokenizer 不支持 chat_template。"
            "本代码要求使用 chat_template 进行格式化，请确保使用支持 chat_template 的模型和分词器。"
        )
    
    print(f"\n使用 chat_template 格式化提示")
    print(f"chat_template 类型: {type(tokenizer.chat_template).__name__}")
    
    # ========================================================================
    # 【代码块】使用 process_func 进行完整的预处理（包括 Loss Masking）
    # ========================================================================
    # 功能：对数据集中的每个样本应用预处理函数
    # 原因：使用 partial 固定 tokenizer 和 max_length 参数，避免在 map 函数中重复传递
    #       这样可以提高代码可读性，并确保所有样本使用相同的预处理参数
    _processing_function = partial(
        process_func,
        tokenizer=tokenizer,  # 固定 tokenizer，避免在 map 中重复传递
        max_length=max_length  # 固定 max_length，确保所有样本使用相同的长度限制
    )
    
    # 对数据集进行映射处理（应用预处理函数到每个样本）
    # remove_columns: 移除原始列，因为已经转换为 input_ids、attention_mask、labels
    dataset = dataset.map(
        _processing_function,
        remove_columns=['id', 'topic', 'dialogue', 'summary'],  # 移除原始列（已转换为 token 格式）
    )
    
    # ========================================================================
    # 【代码块】过滤掉超过最大长度的样本
    # ========================================================================
    # 功能：移除长度超过 max_length 的样本
    # 原因：超过最大长度的样本会被截断，可能导致信息丢失；直接过滤可以避免训练不稳定
    dataset = dataset.filter(lambda sample: len(sample["input_ids"]) < max_length)
    
    # ========================================================================
    # 【代码块】打乱数据顺序
    # ========================================================================
    # 功能：随机打乱数据集中的样本顺序
    # 原因：打乱顺序可以避免模型学习到数据集的顺序模式，提高泛化能力
    #       使用固定的 seed 可以确保实验可复现
    dataset = dataset.shuffle(seed=seed)  # 使用固定 seed 确保可复现性
    
    print("\n数据集预处理完成。")
    print(f"  每个样本包含: input_ids, attention_mask, labels")
    print(f"  其中 labels 中 -100 表示不计算 Loss，其他值表示需要学习的 token")
    return dataset


# ============================================================================
# 第三部分：自定义 DataCollator（包含可视化教学功能）
# ============================================================================

class VisualCheckDataCollator:
    """
    带可视化教学功能的数据整理器
    
    这个整理器手动实现了 DataCollatorForSeq2Seq 的核心逻辑，并添加了可视化功能。
    它会在第一个 batch 时打印详细的数据结构，帮助理解模型到底"看"到了什么，"学"到了什么。
    
    核心功能：
    1. 对 input_ids 和 labels 进行 padding（保留预处理好的 labels）
    2. 创建 attention_mask
    3. 可视化展示第一个 batch 的数据结构（仅一次）
    
    适用场景：
    - 学习和理解 DataCollator 的工作原理
    - 调试数据格式问题
    - 教学演示
    
    注意：
    - 这个 DataCollator 假设 dataset 中已经包含预处理好的 labels
    - labels 中 -100 表示不计算 Loss，其他值表示需要学习的 token
    - 与 DataCollatorForSeq2Seq 功能相同，但添加了可视化功能
    """
    
    def __init__(self, tokenizer):
        """
        初始化数据整理器
        
        Args:
            tokenizer: 分词器实例
        """
        # ========================================================================
        # 【代码块】初始化数据整理器的基本属性
        # ========================================================================
        # 功能：设置 tokenizer 和 padding token ID
        # 原因：如果 tokenizer 没有 pad_token_id，使用 eos_token_id 作为替代
        #       这样可以确保 padding 操作有可用的 token ID
        self.tokenizer = tokenizer
        # 如果 tokenizer 没有 pad_token_id，使用 eos_token_id 作为替代（确保 padding 有可用 token）
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        self.printed = False  # 只打印一次，防止刷屏（可视化信息只在第一个 batch 打印）
    
    def __call__(self, features):
        """
        处理一批样本，返回模型训练所需的格式
        
        Args:
            features: 包含 'input_ids', 'attention_mask', 'labels' 的样本列表
            
        Returns:
            dict: 包含 'input_ids', 'attention_mask', 'labels' 的字典
        """
        # ========================================================================
        # 【代码块】提取并转换数据格式
        # ========================================================================
        # 功能：将列表格式的 input_ids 和 labels 转换为 PyTorch tensor
        # 原因：pad_sequence 需要 tensor 格式的输入，且需要统一的数据类型（long）
        input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]
        
        # ========================================================================
        # 【代码块】使用 PyTorch 的 pad_sequence 进行 padding
        # ========================================================================
        # 功能：将不同长度的序列填充到相同长度，形成 batch
        # 原因：模型需要固定大小的 batch 输入，不同样本长度不同，必须进行 padding
        from torch.nn.utils.rnn import pad_sequence
        
        # input_ids 的 padding 使用 pad_token_id（模型可以识别并忽略）
        # batch_first=True: 输出格式为 (batch_size, seq_len)，符合大多数模型的输入要求
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        
        # labels 的 padding 使用 -100（PyTorch 默认忽略的索引，不计算 Loss）
        # 原因：labels 中 -100 的位置会被 CrossEntropyLoss 自动忽略，不会影响 Loss 计算
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        
        # ========================================================================
        # 【代码块】创建 Attention Mask
        # ========================================================================
        # 功能：标记哪些位置是真实内容（1），哪些是 padding（0）
        # 原因：模型需要知道哪些位置应该参与 attention 计算，哪些位置应该被忽略
        #       这样可以避免 padding token 影响模型的注意力机制
        attention_mask = (input_ids != self.pad_token_id).long()  # 非 padding 位置为 1，padding 位置为 0
        
        # ====================================================================
        # 【代码块】教学可视化环节（只打印一次）
        # ====================================================================
        # 功能：打印第一个 batch 的详细信息，帮助理解数据格式和 Loss Masking
        # 原因：可视化可以帮助用户理解模型到底"看"到了什么，"学"到了什么
        #       只打印一次避免输出过多信息，影响训练日志的可读性
        if not self.printed:
            print("\n" + "="*80)
            print("【DataCollator 教学透视】第一批数据长什么样？")
            print("="*80)
            
            # ====================================================================
            # 【代码块】提取第一个样本的实际内容（去除 padding）
            # ====================================================================
            # 功能：获取第一个样本的非 padding 部分，用于可视化展示
            # 原因：padding 在右侧（right padding），实际内容在左侧
            #       通过 attention_mask 可以准确找到实际内容的长度
            # 注意：padding 在右侧（right padding），所以实际内容在 input_ids[0][:actual_length]
            #       input_ids[0][actual_length:] 是 padding token，不应该显示
            actual_length = attention_mask[0].sum().item()  # 通过 attention_mask 计算实际长度
            actual_input_ids = input_ids[0][:actual_length]  # 只取非 padding 部分
            actual_labels = labels[0][:actual_length]  # 只取非 padding 部分
            
            # 解码第一个样本的 input_ids（只解码非 padding 部分）
            decoded_input = self.tokenizer.decode(actual_input_ids, skip_special_tokens=False)
            print(f"\n1. 模型输入的 Token ID 序列 (Input IDs):")
            print(f"   实际长度（非padding）: {actual_length}")
            print(f"   Padding后总长度: {len(input_ids[0])}")
            print(f"   前1100个Token ID: {actual_input_ids[:1100].tolist() if actual_length > 1100 else actual_input_ids.tolist()}")
            
            print(f"\n2. 对应的实际文本（包含特殊字符，完整内容）:")
            print(f"   完整文本长度: {len(decoded_input)} 字符")
            if len(decoded_input) > 1100:
                print(f"   前1100字符: {decoded_input[:1100]}...")
                print(f"   后500字符: ...{decoded_input[-500:]}")
            else:
                print(f"   {decoded_input}")
            
            # 解码 labels，把 -100 的地方显示为 [IGNORE]
            readable_labels = []
            label_values = []
            display_length = min(1100, actual_length)
            for i, lab in enumerate(actual_labels[:display_length]):
                if lab.item() == -100:
                    readable_labels.append("[IGNORE]")
                    label_values.append(-100)
                else:
                    readable_labels.append(self.tokenizer.decode([lab.item()]))
                    label_values.append(lab.item())
            
            print(f"\n3. 计算 Loss 的标签 (Labels):")
            print(f"   实际长度（非padding）: {actual_length}")
            print(f"   前{display_length}个Label值: {label_values}")
            print(f"   说明：只有 [IGNORE] 以外的部分会计入 Loss")
            print(f"   前{display_length}个Label文本: {' '.join(readable_labels)}")
            
            # 找到 assistant 回复的起始位置（第一个非 -100 的位置）
            assistant_start_idx = None
            for i, lab in enumerate(actual_labels):
                if lab.item() != -100:
                    assistant_start_idx = i
                    break
            
            if assistant_start_idx is not None:
                # 解码 assistant 回复部分
                assistant_input_ids = actual_input_ids[assistant_start_idx:]
                assistant_labels = actual_labels[assistant_start_idx:]
                assistant_text = self.tokenizer.decode(assistant_input_ids, skip_special_tokens=False)
                
                print(f"\n4. Assistant 回复部分分析:")
                print(f"   Assistant 回复起始位置（Token索引）: {assistant_start_idx}")
                print(f"   Assistant 回复长度（Token数）: {len(assistant_input_ids)}")
                print(f"   Assistant 回复文本（前500字符）: {assistant_text[:500]}...")
                print(f"   说明：这部分在 labels 中不是 -100，会计算 Loss")
            else:
                print(f"\n4. Assistant 回复部分分析:")
                print(f"   ⚠️  警告：未找到 assistant 回复部分（所有 labels 都是 -100）")
            
            # 统计有多少位置需要计算 Loss（只统计非 padding 部分）
            loss_positions = (actual_labels != -100).sum().item()
            total_positions = len(actual_labels)
            print(f"\n5. Loss 计算统计（仅非padding部分）:")
            print(f"   总位置数（非padding）: {total_positions}")
            print(f"   需要计算 Loss 的位置数: {loss_positions}")
            print(f"   忽略的位置数: {total_positions - loss_positions}")
            if total_positions > 0:
                print(f"   Loss 计算比例: {loss_positions/total_positions*100:.2f}%")
            
            print("="*80 + "\n")
            self.printed = True
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


class ManualMaskDataCollator:
    """
    手动处理 Loss Masking 的数据整理器（教学用，简化版）
    
    这个整理器手动实现了 DataCollatorForSeq2Seq 的核心逻辑，但没有可视化功能。
    它保留了预处理好的 labels，只负责 padding 和格式转换。
    
    核心功能：
    1. 对 input_ids 和 labels 进行 padding（保留预处理好的 labels）
    2. 创建 attention_mask
    
    适用场景：
    - 学习和理解 DataCollator 的核心逻辑
    - 需要简洁实现时的参考
    
    注意：
    - 这个 DataCollator 假设 dataset 中已经包含预处理好的 labels
    - labels 中 -100 表示不计算 Loss，其他值表示需要学习的 token
    - 与 DataCollatorForSeq2Seq 功能相同，但代码更简洁，便于理解
    """
    
    def __init__(self, tokenizer, mlm=False):
        """
        初始化数据整理器
        
        Args:
            tokenizer: 分词器实例
            mlm: 是否使用掩码语言模型（Masked Language Modeling）
                 对于 CausalLM，通常设置为 False
        """
        # ========================================================================
        # 【代码块】初始化数据整理器的基本属性
        # ========================================================================
        # 功能：设置 tokenizer 和 padding token ID
        # 原因：如果 tokenizer 没有 pad_token_id，使用 eos_token_id 作为替代
        #       mlm 参数保留以兼容不同场景，但 CausalLM 通常不使用 MLM
        self.tokenizer = tokenizer
        self.mlm = mlm  # 保留 mlm 参数以兼容不同场景（CausalLM 通常为 False）
        # 如果 tokenizer 没有 pad_token_id，使用 eos_token_id 作为替代
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    
    def __call__(self, features):
        """
        处理一批样本，返回模型训练所需的格式
        
        Args:
            features: 包含 'input_ids', 'attention_mask', 'labels' 的样本列表
            
        Returns:
            dict: 包含 'input_ids', 'attention_mask', 'labels' 的字典
        """
        from torch.nn.utils.rnn import pad_sequence
        
        # ========================================================================
        # 【代码块】提取并转换数据格式
        # ========================================================================
        # 功能：将列表格式的 input_ids 和 labels 转换为 PyTorch tensor
        # 原因：pad_sequence 需要 tensor 格式的输入，且需要统一的数据类型（long）
        input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]
        
        # ========================================================================
        # 【代码块】进行 Padding 操作
        # ========================================================================
        # 功能：将不同长度的序列填充到相同长度，形成 batch
        # 原因：模型需要固定大小的 batch 输入，不同样本长度不同，必须进行 padding
        # input_ids 使用 pad_token_id 进行 padding（模型可以识别并忽略）
        # labels 使用 -100 进行 padding（PyTorch 默认忽略，不计算 Loss）
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)  # -100 表示忽略该位置的 Loss
        
        # ========================================================================
        # 【代码块】创建 Attention Mask
        # ========================================================================
        # 功能：标记哪些位置是真实内容（1），哪些是 padding（0）
        # 原因：模型需要知道哪些位置应该参与 attention 计算，避免 padding token 影响注意力机制
        attention_mask = (input_ids != self.pad_token_id).long()  # 非 padding 位置为 1，padding 位置为 0
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


# ============================================================================
# 第四部分：模型训练主函数
# ============================================================================

def train_with_trainer(
    collator_type="seq2seq",
    model_path="Qwen/Qwen3-8B",
    dataset_name="neil-code/dialogsum-test",
    output_dir="./peft-dialogue-summary-training/final-checkpoint",
    seed=42,
    max_steps=2000,
):
    """
    使用 Transformers Trainer 进行模型训练的主函数
    
    注意：本函数始终使用 tokenizer.apply_chat_template 进行格式化，不支持手动格式化。
    
    Args:
        collator_type: DataCollator 类型，可选值：
            - "language_modeling": 使用 DataCollatorForLanguageModeling（预训练用，不适用于 SFT）
            - "seq2seq": 使用 DataCollatorForSeq2Seq（SFT 标准方案，推荐用于生产）
            - "visual_check": 使用 VisualCheckDataCollator（教学用，带可视化）
            - "manual_mask": 使用 ManualMaskDataCollator（教学用，简化版）
            默认值："seq2seq"
        model_path: 预训练模型路径
        dataset_name: 数据集名称
        output_dir: 模型保存目录
        seed: 随机种子
        max_steps: 最大训练步数
        
    Raises:
        ValueError: 如果 tokenizer 不支持 chat_template 或 collator_type 无效
    """
    
    # ========================================================================
    # 【代码块】步骤1：设置随机种子（确保实验可复现）
    # ========================================================================
    # 功能：设置所有随机数生成器的种子
    # 原因：确保每次运行的结果一致，便于调试和对比不同配置的效果
    set_seed(seed)
    print(f"设置随机种子: {seed}")
    
    # ========================================================================
    # 【代码块】步骤2：设置缓存目录（数据集和模型）
    # ========================================================================
    # 功能：确定数据集和模型的缓存位置，如果可以使用HF默认缓存目录可以注释掉相关代码
    # 原因：将缓存放在脚本目录下，便于管理和清理，避免占用系统默认缓存目录
    # 获取当前脚本所在目录（兼容直接运行和作为模块导入的情况）
    try:
        # 如果作为脚本直接运行，可以获取 __file__
        script_dir = Path(__file__).parent.absolute()
    except NameError:
        # 如果作为模块导入，使用当前工作目录（__file__ 不存在时）
        script_dir = Path.cwd()
    
    # 创建数据集和模型的缓存目录（在脚本目录下，便于管理）
    dataset_cache_dir = script_dir / "datasets"
    model_cache_dir = script_dir / "models"
    dataset_cache_dir.mkdir(exist_ok=True)  # 如果目录不存在则创建
    model_cache_dir.mkdir(exist_ok=True)  # 如果目录不存在则创建
    
    print(f"数据集缓存目录: {dataset_cache_dir}")
    print(f"模型缓存目录: {model_cache_dir}")
    
    # ========================================================================
    # 【代码块】步骤3：加载数据集
    # ========================================================================
    # 功能：从 Hugging Face Hub 或本地加载数据集
    # 原因：使用 cache_dir 参数可以控制数据集缓存位置，避免占用系统默认缓存目录
    print("\n" + "="*80)
    print("步骤1：加载数据集")
    print("="*80)
    print(f"数据集名称: {dataset_name}")
    
    # 使用 cache_dir 参数指定数据集缓存位置
    # 原因：将数据集缓存到程序所在目录，便于管理和清理，避免占用系统默认缓存目录
    dataset = load_dataset(dataset_name, cache_dir=str(dataset_cache_dir))
    
    print(f"数据集信息: {dataset}")
    print(f"训练集样本数: {len(dataset['train'])}")
    print(f"验证集样本数: {len(dataset['validation'])}")
    
    # 打印一个示例（注意：这里先加载 tokenizer 后再打印格式化示例）
    print("\n原始数据示例:")
    print(dataset['train'][0])
    
    # ========================================================================
    # 【代码块】步骤4：配置量化参数（用于节省显存）
    # ========================================================================
    # 功能：配置模型量化参数，减少显存占用
    # 原因：4-bit 量化可以将模型大小减少约 87.5%，让普通显卡也能运行大模型
    #       NF4 是一种特殊的 4-bit 量化格式，在保持性能的同时最大化压缩比
    print("\n" + "="*80)
    print("步骤2：配置量化参数")
    print("="*80)
    compute_dtype = getattr(torch, "float16")  # 计算时使用 float16，平衡精度和速度
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,  # 使用 4-bit 量化（模型权重存储为 4-bit）
        bnb_4bit_quant_type="nf4",  # 使用 NF4 量化类型（NormalFloat4，性能更好的 4-bit 格式）
        bnb_4bit_compute_dtype=compute_dtype,  # 计算时使用 float16（推理和反向传播时提升精度）
    )
    print("量化配置完成：4-bit NF4 量化，计算精度 float16")
    
    # ========================================================================
    # 【代码块】步骤5：加载预训练模型和分词器
    # ========================================================================
    # 功能：加载预训练模型和分词器，准备进行微调
    # 原因：使用量化配置可以大幅减少显存占用，让普通显卡也能运行大模型
    print("\n" + "="*80)
    print("步骤3：加载预训练模型和分词器")
    print("="*80)
    print(f"模型路径: {model_path}")
    print(f"模型缓存目录: {model_cache_dir}")
    
    # ========================================================================
    # 【代码块】加载模型（使用量化配置）
    # ========================================================================
    # 功能：从 Hugging Face Hub 或本地加载预训练模型
    # 原因：使用 cache_dir 可以控制模型缓存位置；device_map="auto" 自动分配 GPU/CPU
    #       quantization_config 应用 4-bit 量化，大幅减少显存占用
    original_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        cache_dir=str(model_cache_dir),  # 将模型缓存到指定目录，便于管理
        dtype=compute_dtype,  # 模型的基础数据类型（与量化配置中的计算类型一致）
        device_map="auto",  # 自动分配设备（GPU/CPU），充分利用可用资源
        quantization_config=quant_config  # 应用 4-bit 量化配置
    )
    print("模型加载完成")
    
    # ========================================================================
    # 【代码块】加载训练用的分词器
    # ========================================================================
    # 功能：加载分词器，用于将文本转换为 token ID
    # 原因：训练时需要特定的配置（如 padding_side="right"），确保与训练流程兼容
    # ========================================================================
    # 【注 1】关于 Padding Side (为什么是 Right?)
    # ========================================================================
    # 训练时：必须 Right Padding
    # - 如果 Pad 在左边，你的 labels 对齐逻辑会变得非常复杂
    # - 因为 -100 的位置会变，导致 Loss Masking 无法正确工作
    # - 右侧 Padding 可以确保 labels 和 input_ids 的位置一一对应
    #
    # 推理时：通常建议 Left Padding（虽然本代码演示时用了 Right）
    # - Left Padding 可以让生成的 Token 紧接着 Prompt
    # - 而不是隔着一堆 Pad Token，这样生成效果更好
    #
    # 处理padding对于HF已经是相当底层的工作了，后面我们不会处理很多这些问题，尤其是推理已经被如vllm，sglang等高层次封装框架取代了
    # ========================================================================
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        cache_dir=str(model_cache_dir),  # 使用相同的模型缓存目录
        use_fast=False,  # 使用慢速但更稳定的分词器（某些模型需要）
        trust_remote_code=True,  # 信任远程代码（某些模型需要执行自定义代码）
        padding_side="right",  # 训练时：Padding 在右侧（确保 labels 和 input_ids 位置对齐）
        add_eos_token=True,   # 添加结束符（让模型学会在合适位置停止生成）
    )
    # 设置 pad_token（如果不存在）
    # 原因：某些模型没有 pad_token，需要手动设置（通常使用 eos_token 作为替代）
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id  # 使用 eos_token 作为 pad_token
    print("训练分词器加载完成")
    
    # ========================================================================
    # 【代码块】检查 chat_template 支持情况（必须支持）
    # ========================================================================
    # 功能：验证 tokenizer 是否支持 chat_template
    # 原因：本代码强制使用 chat_template，不支持手动格式化，确保训练和推理格式一致
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
    # 打印一个格式化示例（测试 chat_template 是否正常工作）
    # 原因：通过实际测试可以验证 chat_template 是否配置正确，避免后续训练时出错
    try:
        test_messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        example = tokenizer.apply_chat_template(
            test_messages, 
            tokenize=False,  # 只格式化不分词，用于展示格式
            add_generation_prompt=False  # 训练模式，不需要生成提示
        )
        print(f"  chat_template 示例（前100字符）: {example[:100]}...")
    except Exception as e:
        print(f"  警告: 无法测试 chat_template: {e}")
    print("-"*80)
    
    # ========================================================================
    # 【代码块】打印数据格式化示例（使用实际数据）
    # ========================================================================
    # 功能：展示数据预处理后的格式，帮助理解 Loss Masking 的效果
    # 原因：可视化可以帮助用户理解数据格式，确认 Loss Masking 是否正确工作
    print("\n数据格式化示例（使用 chat_template 和 Loss Masking）:")
    sample_example = dataset['train'][0]
    processed_example = process_func(sample_example.copy(), tokenizer, max_length=512)
    # 解码 input_ids 查看格式化后的文本（包含特殊标记）
    # skip_special_tokens=False: 保留特殊标记，可以看到完整的格式化结果
    decoded_text = tokenizer.decode(processed_example['input_ids'], skip_special_tokens=False)
    print(f"格式化后的文本（前200字符）: {decoded_text[:200]}...")
    # 显示 labels 的分布（统计有多少位置会计算 Loss）
    # 原因：通过统计可以验证 Loss Masking 是否正确，只有 assistant 回复部分会计算 Loss
    labels = processed_example['labels']
    loss_count = sum(1 for l in labels if l != -100)  # 会计算 Loss 的位置数
    ignore_count = sum(1 for l in labels if l == -100)  # 忽略的位置数（prompt 部分）
    print(f"Labels统计: 总长度={len(labels)}, 计算Loss={loss_count}, 忽略={ignore_count}")
    
    # ========================================================================
    # 【代码块】加载评估用的分词器（可以有不同的配置）
    # ========================================================================
    # 功能：加载用于推理评估的分词器
    # 原因：评估时可以使用不同的配置（如 add_bos_token=True），与训练时分词器分离
    #       这样可以灵活调整推理配置，而不影响训练流程
    eval_tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        cache_dir=str(model_cache_dir),  # 使用相同的模型缓存目录
        add_bos_token=True,  # 添加 BOS token（某些模型在推理时需要）
        trust_remote_code=True,
        use_fast=False
    )
    # 设置 pad_token（如果不存在）
    if eval_tokenizer.pad_token_id is None:
        eval_tokenizer.pad_token_id = eval_tokenizer.eos_token_id
    print("评估分词器加载完成")
    
    # ========================================================================
    # 【代码块】步骤6：获取模型最大序列长度并预处理数据集
    # ========================================================================
    # 功能：获取模型支持的最大序列长度，并对数据集进行预处理
    # 原因：不同模型支持的最大长度不同，需要根据模型配置确定合适的长度
    print("\n" + "="*80)
    print("步骤4：预处理数据集")
    print("="*80)
    max_length = get_max_length(original_model)  # 从模型配置中获取最大长度
    print(f"使用最大序列长度: {max_length}")
    
    # 预处理训练集和验证集（应用 Loss Masking）
    # 原因：训练集和验证集都需要相同的预处理流程，确保格式一致
    train_dataset = preprocess_dataset(tokenizer, max_length, seed, dataset['train'])
    eval_dataset = preprocess_dataset(tokenizer, max_length, seed, dataset['validation'])
    
    # ========================================================================
    # 【代码块】步骤6.5：根据 DataCollator 类型处理 dataset
    # ========================================================================
    # 功能：根据选择的 DataCollator 类型，决定是否保留预处理好的 labels
    # 原因：DataCollatorForLanguageModeling 会自动创建 labels，需要移除预处理好的 labels
    #       其他 DataCollator 需要保留预处理好的 labels，确保 Loss Masking 正确工作
    # DataCollatorForLanguageModeling 需要移除 labels（让它自动创建）
    # 其他 DataCollator 需要保留 labels（使用预处理好的 labels）
    if collator_type == "language_modeling":
        print("\n" + "="*80)
        print("⚠️  警告：移除预处理好的 labels")
        print("="*80)
        print("DataCollatorForLanguageModeling 会自动创建 labels = input_ids")
        print("这意味着会对所有 token（包括 prompt）计算 Loss")
        print("这不是 SFT 训练，而是'预训练续写'，仅用于对比教学！")
        print("="*80)
        # 移除 labels，让 DataCollatorForLanguageModeling 自动创建
        train_dataset = train_dataset.remove_columns(['labels'])
        eval_dataset = eval_dataset.remove_columns(['labels'])
    else:
        # 其他 DataCollator 需要保留 labels（使用预处理好的 labels，确保 Loss Masking 正确）
        print(f"\n✓ 保留预处理好的 labels（用于 {collator_type} DataCollator）")
    
    print(f"\n预处理后的训练集: {train_dataset}")
    print(f"预处理后的验证集: {eval_dataset}")
    
    # ========================================================================
    # 【代码块】步骤7：配置训练参数（TrainingArguments）
    # ========================================================================
    # 功能：配置训练过程中的所有超参数和设置
    # 原因：TrainingArguments 集中管理所有训练配置，便于调整和复现实验
    print("\n" + "="*80)
    print("步骤5：配置训练参数")
    print("="*80)
    
    training_args = TrainingArguments(
        # ========================================================================
        # 输出目录
        # ========================================================================
        output_dir=output_dir,  # 模型检查点和日志的保存位置
        
        # ========================================================================
        # 训练相关参数
        # ========================================================================
        max_steps=max_steps,  # 最大训练步数（控制训练时长）
        per_device_train_batch_size=25,  # 每个设备的训练批次大小（根据显存调整）
        gradient_accumulation_steps=1, # 梯度累积步数（实际批次大小 = batch_size * gradient_accumulation_steps）
        # 原因：当显存不足时，可以通过梯度累积模拟更大的 batch size
        learning_rate=2e-4,  # 学习率（2e-4 是微调大模型的常用值）
        warmup_steps=1,  # 预热步数（学习率从 0 逐渐增加到目标值，避免初期震荡）
        
        # ========================================================================
        # 优化器配置
        # ========================================================================
        # 使用分页的 8-bit AdamW 优化器（节省显存）
        # 原因：8-bit 优化器可以大幅减少优化器状态的显存占用，让普通显卡也能训练大模型
        optim="paged_adamw_8bit",
        
        # ========================================================================
        # 评估相关参数
        # ========================================================================
        do_eval=True,  # 是否进行评估（监控模型在验证集上的表现）
        eval_strategy="steps",  # 评估策略：按步数评估（也可以按 epoch 评估）
        eval_steps=100,  # 每多少步评估一次（频繁评估可以及时发现过拟合）
        
        # ========================================================================
        # 保存相关参数
        # ========================================================================
        save_strategy="steps",  # 保存策略：按步数保存（也可以按 epoch 保存）
        save_steps=500,  # 每多少步保存一次（定期保存检查点，避免训练中断导致进度丢失）
        overwrite_output_dir=True,  # 是否覆盖输出目录（如果目录已存在，覆盖旧文件）
        
        # ========================================================================
        # 日志相关参数
        # ========================================================================
        logging_steps=100,  # 每多少步记录一次日志（监控训练进度和 Loss）
        logging_dir="./logs",  # 日志目录（TensorBoard 日志保存位置）
        report_to="none",  # 不向任何平台报告（可选：tensorboard, wandb 等，用于可视化训练过程）
        
        # ========================================================================
        # 其他参数
        # ========================================================================
        # 按长度分组（提高训练效率）
        # 原因：将长度相似的样本放在同一个 batch，减少 padding 浪费，提高训练效率
        group_by_length=True,
    )
    
    print("训练参数配置完成")
    print(f"  - 最大训练步数: {max_steps}")
    print(f"  - 批次大小: {training_args.per_device_train_batch_size}")
    print(f"  - 学习率: {training_args.learning_rate}")
    print(f"  - 输出目录: {output_dir}")
    
    # ========================================================================
    # 【代码块】步骤8：配置 PEFT/LoRA
    # ========================================================================
    # 功能：配置参数高效微调（PEFT），只训练少量参数
    # 原因：LoRA 可以大幅减少可训练参数（通常不到原模型的 1%），让普通显卡也能微调大模型
    print("\n" + "="*80)
    print("步骤6：配置 PEFT/LoRA")
    print("="*80)
    
    # ========================================================================
    # 【代码块】定义 LoRA 配置
    # ========================================================================
    # 功能：配置 LoRA 的超参数和目标模块
    # 原因：通过调整 r、alpha 和 target_modules，可以平衡性能和显存占用
    # ========================================================================
    # 【架构师注脚 3】关于 LoRA 的 Target Modules
    # ========================================================================
    # 代码中设置了 target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    # 这是 Attention 层的四个投影矩阵，适合大多数任务。
    #
    # 对于推理类模型（如 DeepSeek-R1 或 Qwen-Math），逻辑能力主要隐藏在
    # MLP 层（gate_proj, up_proj, down_proj）中。
    #
    # 进阶实验：尝试把 MLP 层也加入训练
    # - 例如：target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
    #                         "gate_proj", "up_proj", "down_proj"]
    # - 你会发现 Loss 下降得更快，模型变聪明了
    # - 但显存占用也会增加，需要根据实际情况权衡
    # ========================================================================
    lora_config = LoraConfig(
        r=64,  # LoRA 的秩（rank），控制适配器的大小（越大参数越多，效果可能更好但显存占用更大）
        lora_alpha=16,  # LoRA 的 alpha 参数，用于缩放（通常设为 r 的 1/4 到 1/2）
        # 要应用 LoRA 的模块（Attention 层的四个投影矩阵）
        # 原因：这些模块对模型性能影响最大，训练这些模块通常能获得最好的效果
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",  # 不对 bias 进行训练（减少参数，通常 bias 对性能影响较小）
        lora_dropout=0.01,  # LoRA dropout 率（防止过拟合，通常设置较小的值）
        task_type="CAUSAL_LM",  # 任务类型：因果语言模型（自回归生成任务）
    )
    print("LoRA 配置:")
    print(f"  - Rank (r): {lora_config.r}")
    print(f"  - Alpha: {lora_config.lora_alpha}")
    print(f"  - Target modules: {lora_config.target_modules}")
    
    # ========================================================================
    # 【代码块】准备模型进行量化训练
    # ========================================================================
    # 功能：为量化模型添加训练所需的辅助模块（如 gradient checkpointing）
    # 原因：量化模型在训练时需要特殊处理，prepare_model_for_kbit_training 会自动处理这些细节
    # 1. 准备模型进行 k-bit 训练
    original_model = prepare_model_for_kbit_training(original_model)
    print("模型已准备进行量化训练")
    
    # 2. 应用 LoRA 适配器
    # 功能：将 LoRA 适配器添加到模型中，冻结原始参数，只训练 LoRA 参数
    # 原因：这样可以大幅减少可训练参数，让普通显卡也能微调大模型
    peft_model = get_peft_model(original_model, lora_config)
    print("LoRA 适配器已应用")
    
    # ========================================================================
    # 【代码块】禁用缓存（训练时需要）
    # ========================================================================
    # 功能：禁用模型的 KV 缓存
    # 原因：训练时不需要缓存（推理时才需要），禁用缓存可以节省显存
    peft_model.config.use_cache = False
    print("已禁用模型缓存（训练模式）")
    
    # 打印可训练参数信息（展示 LoRA 的参数效率）
    peft_model.print_trainable_parameters()
    
    # ========================================================================
    # 【代码块】步骤9：选择数据整理器（DataCollator）
    # ========================================================================
    # 功能：根据 collator_type 参数选择合适的数据整理器
    # 原因：不同的 DataCollator 适用于不同的场景，选择合适的可以提高训练效果
    print("\n" + "="*80)
    print("步骤7：配置数据整理器")
    print("="*80)
    
    # ========================================================================
    # 【代码块】验证 collator_type 参数
    # ========================================================================
    # 功能：检查 collator_type 是否有效
    # 原因：无效的 collator_type 会导致训练失败，提前验证可以给出清晰的错误信息
    valid_collator_types = ["language_modeling", "seq2seq", "visual_check", "manual_mask"]
    if collator_type not in valid_collator_types:
        raise ValueError(
            f"无效的 collator_type: {collator_type}\n"
            f"有效值: {valid_collator_types}\n"
            f"  - 'language_modeling': DataCollatorForLanguageModeling（预训练用，不适用于 SFT）\n"
            f"  - 'seq2seq': DataCollatorForSeq2Seq（SFT 标准方案，推荐用于生产）\n"
            f"  - 'visual_check': VisualCheckDataCollator（教学用，带可视化）\n"
            f"  - 'manual_mask': ManualMaskDataCollator（教学用，简化版）"
        )
    
    # ========================================================================
    # 【代码块】根据 collator_type 创建相应的 DataCollator
    # ========================================================================
    # 功能：根据用户选择的类型创建对应的数据整理器
    # 原因：不同的 DataCollator 适用于不同的场景，需要根据需求选择
    if collator_type == "language_modeling":
        print("⚠️  使用 DataCollatorForLanguageModeling（预训练专用，不适用于 SFT）")
        print("   工作原理：自动创建 labels = input_ids（对所有 token 计算 Loss）")
        print("   ⚠️  警告：会对所有 token（包括 prompt）计算 Loss")
        print("   ⚠️  警告：这不是 SFT 训练，而是'预训练续写'")
        print("   ⚠️  警告：仅用于对比教学，理解为什么需要 Loss Masking")
        # mlm=False: 对于因果语言模型（CausalLM），使用自回归训练而不是掩码语言模型
        # 原因：GPT/Llama 等模型是自回归的，不是 BERT 那样的双向模型
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,  # 对于因果语言模型，设置为 False（自回归训练）
        )
        print("已创建 DataCollatorForLanguageModeling")
        
    elif collator_type == "seq2seq":
        print("✓ 使用 DataCollatorForSeq2Seq（SFT 标准方案，推荐用于生产）")
        print("   工作原理：保留预处理好的 labels，只做 padding（labels 用 -100 padding）")
        print("   ✓ 优点：保留预处理好的 labels，确保 Loss Masking 正确")
        print("   ✓ 优点：这是 Hugging Face 官方推荐的 SFT 训练方式")
        print("   ✓ 优点：适用于生产环境，稳定可靠")
        # model=None: 对于 CausalLM，可以设置为 None（DataCollator 不需要模型信息）
        # padding=True: 启用 padding，将不同长度的序列填充到相同长度
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=None,  # 对于 CausalLM，可以设置为 None（不需要模型信息）
            padding=True,  # 启用 padding（将不同长度的序列填充到相同长度）
        )
        print("已创建 DataCollatorForSeq2Seq")
        
    elif collator_type == "visual_check":
        print("✓ 使用 VisualCheckDataCollator（教学用，带可视化功能）")
        print("   工作原理：手动实现 DataCollatorForSeq2Seq 的逻辑，并添加可视化")
        print("   ✓ 优点：打印第一个 batch 的详细信息，帮助理解数据格式")
        print("   ✓ 优点：完全控制 padding 过程，便于教学")
        # 使用自定义的 VisualCheckDataCollator，带可视化功能
        # 原因：帮助用户理解数据格式和 Loss Masking 的工作原理
        data_collator = VisualCheckDataCollator(tokenizer=tokenizer)
        print("已创建 VisualCheckDataCollator（将显示第一个batch的详细信息）")
        
    elif collator_type == "manual_mask":
        print("✓ 使用 ManualMaskDataCollator（教学用，简化版）")
        print("   工作原理：手动实现 DataCollatorForSeq2Seq 的逻辑（无可视化）")
        print("   ✓ 优点：代码简洁，便于理解核心逻辑")
        # 使用自定义的 ManualMaskDataCollator，代码更简洁
        # 原因：帮助用户理解 DataCollator 的核心逻辑，无需可视化功能时使用
        data_collator = ManualMaskDataCollator(tokenizer=tokenizer)
        print("已创建 ManualMaskDataCollator")
    
    # ========================================================================
    # 【代码块】步骤10：创建 Trainer 实例
    # ========================================================================
    # 功能：创建 Trainer 实例，封装训练循环
    # 原因：Trainer 自动处理训练循环、评估、保存等复杂逻辑，简化训练代码
    print("\n" + "="*80)
    print("步骤8：创建 Trainer 实例")
    print("="*80)
    
    trainer = transformers.Trainer(
        model=peft_model,  # 要训练的模型（已应用 LoRA）
        train_dataset=train_dataset,  # 训练数据集（已预处理，包含 Loss Masking）
        eval_dataset=eval_dataset,  # 验证数据集（用于监控模型性能）
        args=training_args,  # 训练参数（包含所有超参数和配置）
        data_collator=data_collator,  # 数据整理器（负责 batch 的 padding 和格式转换）
    )
    
    print("Trainer 创建完成")
    print(f"  - 模型: {type(peft_model).__name__}")
    print(f"  - 训练样本数: {len(train_dataset)}")
    print(f"  - 验证样本数: {len(eval_dataset)}")
    print(f"  - 数据整理器: {type(data_collator).__name__}")
    
    # ========================================================================
    # 【代码块】步骤11：开始训练
    # ========================================================================
    # 功能：执行训练循环，更新模型参数
    # 原因：清空 CUDA 缓存可以释放未使用的显存，为训练预留更多空间
    print("\n" + "="*80)
    print("步骤9：开始训练")
    print("="*80)
    
    # 清空 CUDA 缓存（释放未使用的显存，为训练预留更多空间）
    torch.cuda.empty_cache()
    
    # 开始训练（Trainer 会自动处理训练循环、梯度更新、评估、保存等）
    print("训练开始...")
    trainer.train()
    print("训练完成！")
    
    # ========================================================================
    # 【代码块】步骤12：效果验证（训练后的模型生成测试）；真实部署一般不使用.generate
    # ========================================================================
    # 功能：使用训练后的模型进行推理，验证训练效果
    # 原因：通过实际生成可以直观地看到模型是否学会了任务，比 Loss 更直观
    print("\n" + "="*80)
    print("步骤10：效果验证")
    print("="*80)
    
    # 切换到评估模式（禁用 dropout 和 batch normalization 的训练模式）
    # 原因：推理时不需要随机性，评估模式可以确保结果稳定
    peft_model.eval()
    
    # ========================================================================
    # 【代码块】准备测试样本
    # ========================================================================
    # 功能：从验证集中选择一个样本进行测试
    # 原因：使用验证集样本可以对比模型生成结果和标准答案，评估训练效果
    test_sample = dataset['validation'][0]
    prompt_text = f"Please Summarize: {test_sample['dialogue']}"
    messages = [{"role": "user", "content": prompt_text}]
    
    print("\n测试样本信息:")
    print(f"User 输入:\n{prompt_text[:200]}...")
    print(f"\n标准答案:\n{test_sample['summary']}")
    
    # ========================================================================
    # 【代码块】格式化输入（使用 chat_template）
    # ========================================================================
    # 功能：将用户输入格式化为模型期望的格式
    # 原因：推理时 add_generation_prompt=True 会添加 assistant 引导符，告诉模型开始生成
    #       使用 chat_template 确保训练和推理格式一致
    formatted_input = eval_tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,  # 添加 assistant 引导符，告诉模型开始生成
        tokenize=True,  # 进行分词，转换为 token ID
        return_tensors="pt"  # 返回 PyTorch tensor 格式
    )
    
    # ========================================================================
    # 【代码块】将输入移到模型所在的设备
    # ========================================================================
    # 功能：确保输入数据在正确的设备上（GPU 或 CPU）
    # 原因：模型和输入必须在同一设备上，否则会报错
    # 重要：formatted_input 可能是字典（包含 input_ids, attention_mask 等）
    #       需要正确处理，确保所有 tensor 都移到正确的设备
    device = next(peft_model.parameters()).device  # 获取模型所在的设备
    
    # 正确的移动方式：如果是字典，需要分别移动每个键值对
    if isinstance(formatted_input, dict):
        inputs = {k: v.to(device) for k, v in formatted_input.items()}
        # 获取输入长度（从 input_ids 获取，用于后续提取生成部分）
        input_length = inputs["input_ids"].shape[1]
    else:
        # 如果是单个 tensor，包装成字典格式（兼容不同返回格式）
        inputs = {"input_ids": formatted_input.to(device)}
        input_length = inputs["input_ids"].shape[1]
    
    print(f"\n开始生成（设备: {device}）...")
    
    # ========================================================================
    # 【代码块】生成回复
    # ========================================================================
    # 功能：使用模型生成回复
    # 原因：generate 方法可以直接接受字典格式的输入，自动处理 attention_mask 等
    #       torch.no_grad() 禁用梯度计算，节省显存和计算资源
    with torch.no_grad():  # 推理时不需要梯度，节省显存和计算资源
        outputs = peft_model.generate(
            **inputs,  # 展开字典作为关键字参数
            max_new_tokens=100,  # 最大生成 token 数（控制生成长度）
            do_sample=True,  # 使用采样（增加多样性，而不是贪心解码）
            temperature=0.7,  # 温度参数（控制随机性，值越大越随机）
            top_p=0.95,  # nucleus sampling（只从概率最高的 95% token 中采样）
            pad_token_id=eval_tokenizer.pad_token_id,  # padding token ID
            eos_token_id=eval_tokenizer.eos_token_id,  # 结束 token ID（遇到时停止生成）
        )
    
    # ========================================================================
    # 【代码块】解码生成的回复（只取新生成的部分）
    # ========================================================================
    # 功能：将 token ID 转换回文本，只提取新生成的部分
    # 原因：outputs 包含完整的序列（输入+生成），我们只需要生成的部分
    #       使用之前保存的 input_length 可以准确提取生成部分
    # outputs[0] 是完整的输出（包含输入+生成），outputs[0][input_length:] 是生成的部分
    generated_ids = outputs[0][input_length:]
    response = eval_tokenizer.decode(generated_ids, skip_special_tokens=True)  # 解码为文本，跳过特殊标记
    
    print(f"\n模型训练后生成:\n{response}")
    
    # ========================================================================
    # 【架构师注脚 2】关于 <|im_end|> (EOS Token)
    # ========================================================================
    # 注意看生成的结果，模型是否学会了在说完话后立刻停止？
    #
    # 如果模型一直在复读或者说话停不下来，通常是因为：
    # - Data Processing 阶段没有给 Assistant 的回复加上 EOS Token
    # - 模型没有学会什么时候该停止生成
    #
    # tokenizer.apply_chat_template 通常会自动处理这个
    # - 在 process_func() 中使用 apply_chat_template 时，会自动添加 EOS Token
    # - 但如果你手动拼接字符串，这是最容易踩的坑
    #
    # 检查方法：
    # - 观察生成结果是否在合适的地方停止
    # - 如果模型一直生成，可能需要检查数据预处理是否正确添加了 EOS Token
    # ========================================================================
    print("="*80)
    
    # ========================================================================
    # 【代码块】步骤13：清理资源
    # ========================================================================
    # 功能：释放不再使用的模型和训练器，清理显存
    # 原因：训练完成后，释放资源可以避免显存占用，为后续操作预留空间
    print("\n" + "="*80)
    print("步骤11：清理资源")
    print("="*80)
    
    # 释放内存（删除大对象，释放显存）
    del original_model  # 删除原始模型（已转换为 peft_model，不再需要）
    del trainer  # 删除训练器（训练已完成，不再需要）
    torch.cuda.empty_cache()  # 清空 CUDA 缓存，释放未使用的显存
    print("资源清理完成")
    
    return output_dir, dataset  # 返回dataset以便后续使用


# ============================================================================
# 第五部分：主程序入口
# ============================================================================

if __name__ == "__main__":
    """
    主程序入口
    
    注意：本代码始终使用 tokenizer.apply_chat_template 进行格式化。
    
    可以通过修改 COLLATOR_TYPE 参数来选择使用哪个 DataCollator：
    - "language_modeling": DataCollatorForLanguageModeling（预训练用，不适用于 SFT，仅用于对比教学）
    - "seq2seq": DataCollatorForSeq2Seq（SFT 标准方案，推荐用于生产）
    - "visual_check": VisualCheckDataCollator（教学用，带可视化功能）
    - "manual_mask": ManualMaskDataCollator（教学用，简化版）
    """
    
    print("="*80)
    print("Transformers Trainer 训练教程")
    print("="*80)
    print("\n本示例展示了如何使用 Hugging Face Transformers 的 Trainer 进行模型微调")
    print("注意：本代码始终使用 tokenizer.apply_chat_template 进行格式化")
    print("\n支持的 DataCollator 类型：")
    print("  1. 'language_modeling': DataCollatorForLanguageModeling（预训练用，不适用于 SFT）")
    print("  2. 'seq2seq': DataCollatorForSeq2Seq（SFT 标准方案，推荐用于生产）")
    print("  3. 'visual_check': VisualCheckDataCollator（教学用，带可视化功能）")
    print("  4. 'manual_mask': ManualMaskDataCollator（教学用，简化版）\n")
    
    # ========================================================================
    # 【代码块】配置参数（可以根据需要修改）
    # ========================================================================
    # 功能：定义训练所需的所有配置参数
    # 原因：集中管理配置参数，便于修改和实验不同的设置
    COLLATOR_TYPE = "visual_check"  # 可选: "language_modeling", "seq2seq", "visual_check", "manual_mask"
    MODEL_PATH = "Qwen/Qwen3-8B"  # 预训练模型路径（必须支持 chat_template）
    DATASET_NAME = "neil-code/dialogsum-test"  # 数据集名称（Hugging Face Hub 上的数据集）
    OUTPUT_DIR = "./peft-dialogue-summary-training/final-checkpoint"  # 模型保存目录
    SEED = 42  # 随机种子（确保实验可复现）
    MAX_STEPS = 40  # 最大训练步数（可以根据需要调整，控制训练时长）
    
    # ========================================================================
    # 【代码块】执行训练
    # ========================================================================
    # 功能：显示训练配置信息，然后执行训练
    # 原因：显示配置信息可以帮助用户确认训练设置是否正确
    print("="*80)
    print("训练配置")
    print("="*80)
    
    # ========================================================================
    # 【代码块】根据 collator_type 显示说明
    # ========================================================================
    # 功能：根据选择的 DataCollator 类型显示相应的说明和警告
    # 原因：不同 DataCollator 有不同的特点和使用场景，需要向用户说明清楚
    collator_descriptions = {
        "language_modeling": {
            "name": "DataCollatorForLanguageModeling（预训练用，不适用于 SFT）",
            "warning": "⚠️  注意：会对所有 token 计算 Loss（包括 prompt 部分）\n  这实际上是在做'预训练续写'，而不是标准的'指令微调（SFT）'\n  仅用于对比教学，理解为什么需要 Loss Masking"
        },
        "seq2seq": {
            "name": "DataCollatorForSeq2Seq（SFT 标准方案，推荐用于生产）",
            "warning": "✓ 只计算 assistant 回复部分的 Loss\n  这是 Hugging Face 官方推荐的 SFT 训练方式"
        },
        "visual_check": {
            "name": "VisualCheckDataCollator（教学用，带可视化功能）",
            "warning": "✓ 只计算 assistant 回复部分的 Loss\n  会打印第一个 batch 的详细信息，帮助理解数据格式"
        },
        "manual_mask": {
            "name": "ManualMaskDataCollator（教学用，简化版）",
            "warning": "✓ 只计算 assistant 回复部分的 Loss\n  代码简洁，便于理解核心逻辑"
        }
    }
    
    # 根据选择的 COLLATOR_TYPE 显示相应的说明
    # 原因：帮助用户理解当前选择的 DataCollator 的特点和适用场景
    if COLLATOR_TYPE in collator_descriptions:
        desc = collator_descriptions[COLLATOR_TYPE]
        print(f"DataCollator 版本：{desc['name']}")
        print(f"  {desc['warning']}")
    else:
        # 如果选择了无效的 COLLATOR_TYPE，使用默认值并给出警告
        # 原因：避免因配置错误导致程序崩溃，提供友好的错误处理
        print(f"⚠️  警告：未知的 COLLATOR_TYPE: {COLLATOR_TYPE}")
        print("  将使用默认值 'seq2seq'")
        COLLATOR_TYPE = "seq2seq"
    
    print("格式化方法：使用 tokenizer.apply_chat_template（必须）")
    print("="*80)
    
    # ========================================================================
    # 【代码块】执行训练函数
    # ========================================================================
    # 功能：调用训练函数，执行完整的训练流程
    # 原因：使用 try-except 捕获异常，提供友好的错误信息，避免程序崩溃
    try:
        output_dir, dataset = train_with_trainer(
            collator_type=COLLATOR_TYPE,  # DataCollator 类型
            model_path=MODEL_PATH,  # 预训练模型路径
            dataset_name=DATASET_NAME,  # 数据集名称
            output_dir=OUTPUT_DIR,  # 输出目录
            seed=SEED,  # 随机种子
            max_steps=MAX_STEPS,  # 最大训练步数
        )
        
        # ========================================================================
        # 【代码块】训练成功完成后的提示信息
        # ========================================================================
        # 功能：显示训练成功信息和使用示例
        # 原因：帮助用户了解如何加载和使用训练好的模型
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
        # ========================================================================
        # 【代码块】错误处理
        # ========================================================================
        # 功能：捕获训练过程中的异常，显示详细的错误信息
        # 原因：提供友好的错误信息，帮助用户快速定位问题，而不是直接崩溃
        print("\n" + "="*80)
        print("训练过程中出现错误：")
        print("="*80)
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {str(e)}")
        import traceback
        traceback.print_exc()  # 打印完整的堆栈跟踪，便于调试

