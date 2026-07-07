---
title: "大语言模型微调实践：显存估算、数据格式与框架选择"
title_en: "LLM Fine-Tuning Practice: Memory Estimation, Dataset Formats, and Framework Selection"
date: 2024-12-01 20:00:00 +0800
categories: ["Foundation Models", "Training & Alignment"]
tags: ["Learning Notes","LLM","Fine-tuning","LoRA","Unsloth","HuggingFace"]
author: Hyacehila
excerpt: "整理微调显存估算方法、各类微调数据集格式（ChatML/Alpaca/ShareGPT）、常用框架（Unsloth/HuggingFace/PyTorch）及公开数据集。"
excerpt_en: "Covers fine-tuning memory estimation, dataset formats (ChatML/Alpaca/ShareGPT), common frameworks (Unsloth/HuggingFace/PyTorch), and public datasets."
mathjax: true
hidden: true
permalink: '/blog/2024/12/01/llm-finetuning-practical-guide/'
---

## 微调的显存开销
### 需要消耗显存的地方
估算微调大型语言模型（LLM）所需的显存大小是一个关键步骤，它直接影响硬件选择和训练效率。显存消耗主要由以下部分组成，不同微调技术会显著影响各部分的大小：

**模型参数：** 存储模型权重本身所需的空间。FP、BF、INT 分别代表浮点数、脑浮点数与整数格式，存储格式只影响大小，不涉及精度本身。

计算模型参数所需的显存公式为 `参数数量 * 每个参数的字节数`。模型参数量用 B（十亿）表示，参数的 bit 数用数字表示，8 bit 意味着 1 个字节。因此 7B 的 INT8 模型需要 $7e^9 \times 1\text{Bytes}$ 总计约 7GB 显存。

**优化器状态：** 优化器（如 Adam、AdamW）需要存储额外的状态（如动量、方差）。估算公式为 `参数数量 * 每个优化器状态所需的字节数 * 状态数量`。

优化器状态只和可训练参数有关，每个可训练参数一般对应 4 Bytes 到 12 Bytes，具体取决于使用的优化器。

**梯度：** 反向传播过程中计算出的损失函数关于每个参数的导数。**估算公式：** `参数数量 * 每个梯度的字节数`，也只计算可训练参数。梯度一般采用 8-16 bit，即 1-2 Bytes。

**激活值：** 前向传播过程中计算出的中间结果，影响因素很多。研究者提出了很多技术来降低激活值的显存开销。在使用各种优化技术后，7B 模型的激活值显存开销可以降低到 1GB 左右。

 激活显存 = `K * batch_size * seq_len * hidden_size * num_layers * bytes_per_activation`。其中 `K` 是一个与架构和是否使用检查点/FlashAttention 相关的因子，保守估计在优化后能达到 1-3。

**框架开销与缓冲区：** 1-3GB 不等。

### 微调策略
**全参数微调：** 更新模型的所有参数，显存开销最高。全参数微调一个 7B 模型的优化器需要消耗约 `7e9 * 16 = 84GB` 显存，加上其余开销至少需要 100GB，因此目前基本不会选择全参数微调。

**参数高效微调（PEFT）：** 冻结大部分参数，降低梯度和优化器的显存开销。当冻结大部分参数以后，优化器和梯度需要消耗 `可训练参数数量 * (每个梯度的字节数 + 优化器字节数)`。前者一般为 1-2 Bytes，后者则是 8-16 Bytes。

而需要微调的参数通常只在 Million 数量级（使用 LoRA 技术时），此时优化器和梯度消耗的显存往往可以忽略不计，约 1-2GB（每 7B 参数）。以 FP16 微调一个 7B 模型配合 LoRA 技术，大约消耗 16GB 显存，基本都是模型本身参数消耗的。

**量化微调：** 以 QLoRA 技术为代表，通过量化存储精度来降低显存消耗，往往会量化为 INT4 或 INT8。此时一个 7B 模型的参数只需要消耗约 3.5GB 显存。

对于 LoRA 和 QLoRA 两种经典的微调技术，后者一般需要消耗**模型大小一半到模型大小之间**的显存，随着模型扩大，需要消耗的显存占比会降低。前者则消耗模型大小的一倍或更多显存。这里的数字均为最小值。


## 微调数据集
在不同的业务场景下解决不同的问题，采取的微调任务类型不一样，所用的数据集格式自然也会有所差别。

**微调数据集的格式帮助我们理解数据的组织形式。在实际训练过程中，我们会将结构化的微调数据集按照一定规则重新拼接为模型可以接受的非结构化数据，这个拼接过程还需要考虑模型的 Tokenizer。**
### 微调数据集类型分类
预训练的数据集格式没有明确要求，无需结构化，只需要将内容整理为连续的文本即可。

所有的**监督微调技术（SFT）**的核心是有输入和输出，即如下的 JSON：
```json
{"input": "Hello", "output": "你好"}
```

OpenAI 建议使用带有系统提示词的数据，将系统提示和用户提示进行区分，以此诱导模型性能提升。JSON 形式为：
```json
[
  {
    "dialogue": [
      {"role": "system", "content": "你现在是一个优秀的对话机器人,会和下面的用户进行友善的对话"},
      {"role": "user", "content": "今天天气怎么样？"},
      {"role": "assistant", "content": "北京今日多云转晴，气温22℃，适合户外活动。"},
      {"role": "user", "content": "那适合去长城吗？"},
      {"role": "assistant", "content": "长城景区海拔较高，建议携带外套，注意防晒。"}
    ]
  },
  ...
]

```

最常用的监督微调技术是指令微调，需要在数据中引入指令（instruction）。基础的 JSON 格式如下。当我们需要强化模型遵循某种指令的能力时，这是最常用且效果较好的微调形式。
```json
[
  {
    "instruction": "将这句英文翻译成法语",
    "input": "Hello, how are you?",
    "output": "Bonjour, comment ça va ?"
  },
  ...
]
```

对话微调（Dialogue Tuning）通过多轮对话数据训练模型生成连贯、符合语境的回复，强调对话历史的上下文理解和回复的自然流畅性。其使用对话形式的 JSON，在内容中包含了参与对话的角色。

```json
[
  {
    "dialogue": [
      {"role": "user", "content": "今天天气怎么样？"},
      {"role": "assistant", "content": "北京今日多云转晴，气温22℃，适合户外活动。"},
      {"role": "user", "content": "那适合去长城吗？"},
      {"role": "assistant", "content": "长城景区海拔较高，建议携带外套，注意防晒。"}
    ]
  },
  ...
]

```

领域适配（Domain Adaptation）是指将模型在特定领域的数据上进行微调，使其更好地适应特定领域的任务和需求。其 JSON 中包含领域描述：
```json
[
  {
    "instruction": "分析患者的症状描述",
    "input": "55岁男性，持续性胸骨后疼痛3小时，含服硝酸甘油无效",
    "output": "可能诊断：急性心肌梗死（STEMI），建议立即行心电图检查及心肌酶谱检测",
    "domain": "医疗"
  },
  {
    "instruction": "解释法律条款",
    "input": "《民法典》第1032条",
    "output": "该条款规定自然人享有隐私权，任何组织或个人不得以刺探、侵扰、泄露、公开等方式侵害他人隐私权",
    "domain": "法律"
  },
  ...
]
```

文本分类（Text Classification），是自然语言处理中的一个经典任务，目的就是通过标注数据训练模型对文本进行类别预测或标签分配。这是使用 LLM 来解决经典 NLP 问题的场景。

```json
[
  {"text": "这款手机续航长达48小时，拍照效果惊艳", "label": "positive"},
  {"text": "系统频繁卡顿，客服响应速度慢", "label": "negative"},
  {"text": "量子计算机突破新型纠错码技术", "label": "science_news"},
  {"text": "央行宣布下调存款准备金率0.5个百分点", "label": "finance_news"}
]
```

推理模型的微调其实是监督微调的一种特殊形式，通过在数据集中显式标注思维链（Chain of Thought, CoT），训练模型不仅给出最终答案，还能生成逻辑推导过程。其核心在于让模型学会「分步思考」。JSON 如下：

```json
[
  {
    "instruction": "解决数学应用题",
    "input": "小明买了3支铅笔，每支2元；又买了5本笔记本，每本比铅笔贵4元。总花费多少？",
    "chain_of_thought": [
      "铅笔单价：2元/支 → 3支总价：3×2=6元",
      "笔记本单价：2+4=6元/本 → 5本总价：5×6=30元",
      "合计花费：6+30=36元"
    ],
    "output": "总花费为36元"
  },
  ...
]
```

**并不是所有任务都适合用推理模型，因为推理模型的幻觉比较大。有些情况下选择推理模型反而会起到相反的效果。在处理简单明确的任务时，推理模型可能会把问题复杂化，导致思考过度、响应较慢，甚至增加幻觉的风险。**

知识蒸馏（Knowledge Distillation）是将复杂模型（教师模型）的知识迁移到轻量级模型（学生模型）的技术，通过优化学生模型使其输出接近教师模型的"软标签"，从而在保持性能的同时降低推理成本。

模型蒸馏的数据集构造应该是最简单的。在完全信任大模型输出的条件下，可以直接将大模型产出的问答对作为数据集。知识蒸馏的数据集可以是任意的 JSON 形式，它们由教师模型直接产生。

强化学习微调通过人类主动反馈来优化模型生成质量。其核心在于引入奖励模型（Reward Model）评估生成结果的合理性，并通过强化学习策略（如 PPO 算法）调整模型参数，使生成内容更符合人类偏好。其 JSON 中需要包含 reward 字段（无法从环境中自动获取）。JSON 如下：

```json
[
  {
    "input": "请推荐一部科幻电影",
    "output": "《星际穿越》是一部经典科幻片，探讨了时间与亲情。",
    "reward_score": 4.5  // 人类标注的质量评分（0-5分）
  },
  {
    "input": "解释黑洞理论",
    "output": "黑洞是由暗物质构成的神秘天体，会吞噬一切物质。",
    "reward_score": 2.0  // 包含错误信息，得分低
  }
]

```

为了更好地与人类对齐，也会采用下面的强化学习微调格式：
```json
{
  "prompt": "写一个关于友谊的短故事。",
  "chosen": "从前有两个好朋友... (被人类标注员评为更好的回复)",
  "rejected": "朋友就是... (被人类标注员评为更差的回复)"
}
```
### 微调数据集格式
微调数据集最常用的组织格式是 JSONL（.jsonl）。它本质上是一个文本文件，其中每一行都是一个独立的、有效的 JSON 对象（字典）。这种格式易于流式读取、处理，并且可以高效地追加新数据。

**内部结构（每行一个 JSON 对象）：** 每个 JSON 对象代表一个独立的训练样本（一个"提示-完成"对或一个对话轮次）。**其内部字段根据微调任务类型可以有所不同，无强制要求。**

绝大多数微调都会将收集到的数据重新整理为 JSONL 后再进行下一步操作。至于具体 JSONL 里面使用什么样的字段，虽然没有强制要求，但有很多习惯可以参考。

Hugging Face 的 `datasets` 库支持加载多种格式，内部使用 Apache Arrow 内存格式，但对外暴露类似 Python 字典的接口。结构同样遵循上述核心原则（`prompt`/`completion`、`instruction`/`input`/`output`、`messages`）。很多数据集采用了 OpenAI 或 Alpaca 的规范形式。


#### ChatML
简单的对话数据集格式。OpenAI 用此格式作为 API 调用的交互方法，后用于其微调 API，逐渐发展为习惯，只包含简单的补全任务。

**文本补全：** 每个样本是 `{"prompt": "...", "completion": "..."}`。

**聊天补全：** 每个样本是 `{"messages": [{"role": "system/user/assistant", "content": "..."}, ...]}`。这是目前最广泛采用的对话格式。

#### 持续预训练
对于继续进行预训练，我们使用没有特定结构的原始文本格式。如：
```json
  "text": "Pasta carbonara is a traditional Roman pasta dish. The sauce is made by mixing raw eggs with grated Pecorino Romano cheese and black pepper. The hot pasta is then tossed with crispy guanciale (cured pork cheek) and the egg mixture, creating a creamy sauce from the residual heat. Despite popular belief, authentic carbonara never contains cream or garlic. The dish likely originated in Rome in the mid-20th century, though its exact origins are debated..."

```
这种格式保留了自然语言流，并允许模型从连续文本中学习，也支持各种 PEFT 技术。
#### Alpaca
**指令微调的代表。** 每个样本包含 `instruction`、`input`（可选）、`output`。

```json
{"instruction": "请解释以下笑话的笑点。", 
 "input": "为什么科学家不相信原子？因为它们构成了一切！",
 "output": "笑点在于双关语。'构成'在科学上指原子的基本组成作用，但在日常语境中也意味着'编造、捏造'。科学家不相信原子，是因为原子'构成了一切'（即科学家认为原子理论是编造出来的谎言），这违背了科学事实，从而产生荒谬的幽默效果。"}
```

在进行指令微调而非对话微调时，我们往往不根据模型更换特定的提示符号，而是直接使用 Alpaca 格式，模型有能力学习到相关指令。但使用非指令格式的 ShareGPT 或 ChatML 时，我们会根据模型切换对话模板，且这是必须的。
#### ShareGPT
ShareGPT 格式是专门为**导出和分享多轮对话**而设计的，最核心的特点是使用 `"conversations"` 数组来组织对话轮次。

```json
{
  "id": "unique_conversation_id_123", // 对话的唯一标识符 (可选，但常见)
  "conversations": [ // **核心字段**：包含对话轮次的有序数组
    {
      "from": "human",    // 表示这条消息来自**人类用户**
      "value": "你好，能介绍一下你自己吗？" // 用户消息的具体内容
    },
    {
      "from": "gpt",     // 表示这条消息来自 **AI 模型 (如ChatGPT)**
      "value": "当然可以！我是一个大型语言模型，由OpenAI训练而成。我的目标是理解和生成人类语言，帮助你完成各种任务，比如回答问题、翻译文本、创作内容等等。有什么我可以为你做的吗？" // AI 回复的内容
    },
    {
      "from": "human",
      "value": "你能写一首关于月亮的短诗吗？"
    },
    {
      "from": "gpt",
      "value": "银盘悬夜幕，清辉洒九州。\n玉兔捣灵药，桂影映琼楼。\n盈亏循天道，阴晴寄客愁。\n千古一轮月，照尽人间秋。"
    }
  ]
}
```

**没有 `"system"` 角色**。如果需要，则应该人工加入 human 的提示词中。
### 公开数据集
Hugging face
Kaggle
Dataset search(数据集搜索工具)
Awesome Public Datasets(开源项目)
Open Data Lab

## 微调常用的框架
### 去微调什么模型
首要决定之一是选择合适的模型。我们需要保证后训练与预训练保持一样的核心任务，靠微调无法获得新能力。

对于大部分开源模型，都面临使用 Instruct 模型还是 Base 模型的抉择。Instruct 模型使用内置指令进行了预训练，无需任何微调即可使用。Base 模型则无法开箱即用，需要单独进行指令微调。一般来说，优质数据越多（1000 行以上），就越可以考虑微调 Base 模型。

至于现在新增的推理能力，根据最终使用的场景抉择即可。
### Unsloth
Unsloth 是一个用于 LLM 微调和强化学习的开源框架，致力于高效率地微调 LLM，官方网站为 [Unsloth](https://docs.unsloth.ai)。其重要工作在于高度可定制的基础上提供新的微调接口，以及使用 Triton 重写的高效内核。

Unsloth 适用于 Linux 和 Windows，并需要 CUDA 进行加速。由于需要使用 Triton 重写内核，其无法第一时间支持新发布的模型。想要使用 Unsloth 进行微调，需要参考支持模型文档 [支持模型](https://docs.unsloth.ai/get-started/all-our-models)。主流的开源模型 Qwen、Gemma、LLaMA 均已获得支持。

对于调整 Unsloth 的参数，参考 [调参指南](https://docs.unsloth.ai/get-started/fine-tuning-guide/lora-hyperparameters-guide)。考虑到大部分的微调工作都会使用 LoRA 技术，这份指南有着很强的参考意义。

对于常见模型微调的一些特性，Unsloth 也提供了一定的指南，参考[常见微调模型](https://docs.unsloth.ai/basics/tutorials-how-to-fine-tune-and-run-llms)即可。

Unsloth 提供了 Colab 的 notebook，大部分微调工作可以基于其模型解决。**Unsloth 实际上是一个比 Hugging Face 生态更加高级的封装，其很多类继承自 Hugging Face，因此它们可以混合使用。**

Unsloth 专注优化模型计算图，使用其 `FastLanguageModel` 得到的模型均已经过其优化，使**任何兼容 HF 的训练器自动获得加速**。因此一般会选择使用 TRL 库来实现训练器。

Unsloth 提供训练器，包括 `UnslothTrainer, UnslothTrainingArguments`，提供了进一步的加速优化流程，以及一些新的参数（如嵌入层独立学习率）。它被使用的并不多，官方模板中[连续预训练](https://docs.unsloth.ai/basics/continued-pretraining)使用了相关训练器。
### Hugging Face 生态
**`transformers` + `datasets` + `trl` + `peft`**

- **`transformers`：** 提供加载、训练和保存各种预训练模型（包括几乎所有主流 LLM）的核心库，是生态的基础。
- **`datasets`：** 提供高效加载、处理和缓存各种数据集的功能，与 `transformers` 无缝集成。
- **`trl`：** 专注于**基于人类反馈的学习**微调（如 SFT、Reward Modeling）。
- **`peft`：** 实现各种**参数高效微调**方法的神器，如 LoRA、Prefix Tuning、P-Tuning、AdaLoRA、IA³ 等。**强烈推荐**，因为它能显著降低显存需求和计算成本，让你在消费级显卡上微调大模型成为可能。

适用于几乎所有 LLM 微调场景，并且继承 **DeepSpeed** 提供多卡训练加速。（DeepSpeed 类似于 PyTorch FSDP，主要用于多卡预训练。）

**提供了非常详细的工具文档，并不限于以上主流包，整个生态的文档库为 [Hugging Face](https://hugging-face.cn/docs)。**

### PyTorch
从底层开始重写，支持张量级的操作。
