---
layout: blog-post
title: 参数高效微调（PEFT）：从 Adapter 到 LoRA 的技术演进
date: 2026-03-05 13:00:00 +0800
categories: [训练与对齐]
tags: [Fine-Tuning, Model Mechanics]
author: Hyacehila
excerpt: 梳理参数高效微调（PEFT）领域的核心方法演进——从 Adapter、Prefix-Tuning 到 LoRA、Prompt Tuning、P-Tuning v2 与 AdaLoRA，理解不同技术路线的设计哲学与适用场景。
math: true
---

# 参数高效微调（PEFT）：从 Adapter 到 LoRA 的技术演进

## 为什么需要参数高效微调

大语言模型的全量微调（Full Fine-Tuning）虽然效果显著，但其计算和存储开销随着模型规模的增长变得越来越难以承受。为了在有限资源下高效适配下游任务，研究者们提出了参数高效微调（Parameter-Efficient Fine-Tuning, PEFT）技术——通过冻结大部分预训练参数，仅训练少量新引入或重参数化的模块，以极低的代价实现接近全量微调的效果。

本文梳理 PEFT 领域几种核心方法的设计思想与技术特点。

## Adapter：在层间插入可训练模块

> Houlsby et al., *Parameter-Efficient Transfer Learning for NLP*, ICML 2019.

Adapter 算得上是整个 PEFT 领域最初的作品。在这个阶段，虽然我们已经开始研究大语言模型，但因果语言模型尚未占据主导地位，因此需要通过微调来适配一些简单的下游任务。Adapter 正是在这个场景下，为了提高模型在下游任务上的性能而被提出的。

Adapter 在模型层之间插入了可训练的结构。在作者的实验中，Adapter 被插入到了解码器的 FFN 之后、Layer Norm 之前，其内部本身也是 FFN 并包含了非线性的激活函数。但在实际场景中，Adapter 插入的位置比较灵活，没有确定的规范。

作者认为 Adapter 的主要优势包括：
- **轻量化且高性能**：只微调更少数量的参数（相较于全量微调），但效果很好
- **模型完整性**：不修改模型结构，核心模型只需要一份
- **作为微调手段**：不需要在新任务上进行全量的训练，符合预训练+微调的范式

## Prefix-Tuning：优化连续前缀

> Li and Liang, *Prefix-Tuning: Optimizing Continuous Prompts for Generation*, ACL 2021.

为了获取更好的特定下游任务效果，我们希望对预训练模型进行微调。由于全量微调成本过高，除了直接冻结原始参数层以外，引入额外的参数适配器也被学者们详细讨论。前缀微调（Prefix-Tuning）便是这样的一种轻量级微调方式。

前缀微调的思想来自于上下文学习（In-Context Learning）。少量的提示词（预先输入的 token）就能获得领域性能，因此作者考虑增加前缀 token。**Prefix 是需要被优化的连续 embedding**，在 Prefix-Tuning 过程中，模型的主参数层被冻结，只需要更换 Prefix 就可以进行模型性能的切换。

所有的 Encoder 层都被加入了 Prefix（作者评估了只在前端加入的情况，效果不能让人满意），因此实际上前缀构成了一个矩阵，总参数数量是 $\text{length}(\text{prefix}) \times \text{num\_layers}$。前缀位于左侧，保证所有的 mask-attention 都可以获得足够的信息。

**同时适用于 NLG 和 NLU 任务，也就是这个技术可以用于微调类 BERT 结构和类 GPT 结构。**

作者认为的优势：
- **轻量化**：只微调更少数量的参数（相较于插入 Adapter Layer）
- **模型完整性**：不修改模型结构，核心模型只需要一份
- **应用层面**：由于足够轻量，实现模型定制非常可行，这也有益于隐私保护

## LoRA：低秩适配的简洁与优雅

> Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR 2022.

LoRA 的思想来自矩阵的低秩分解，认为参数矩阵的修改实际上可以被一个低秩矩阵来概括，因此使用可学习的小矩阵代替原本的全参数修改。

作者认为的主要优势包括：
- **存储空间优势**：原始参数被全部冻结，减少了训练显存消耗以及模型存储与任务切换的开支
- **计算效率**：由于只需要优化新的低秩矩阵，计算效率显著提升
- **零推理延迟**：低秩矩阵可以被直接合并到原始参数中，不会因为新矩阵带来任何推理的延迟
- **灵活组合**：可以与其他很多方法结合使用
- **全量微调的泛化形式**：增加低秩 $r$ 的大小，将逼近全量微调

### 对比其他方案

- Adapter 的策略虽然参数增加很少，但仍旧产生 Inference Latency
- Prefix-Tuning 的前缀难以学习，性能优化速度很低

### 实验发现

- QKVO 四个注意力矩阵最好都被优化。哪怕整体优化需要一个低 rank，但四个矩阵都被更低 rank 的分解优于只高 rank 分解其中一个
- 并不需要很高的 rank 就可以实现足够的性能，在作者考虑的实验中，$r=8$ 和 $r=64$ 的性能没有产生显著的差别，这源于它们共享一维的子空间
- 低秩适配矩阵的核心作用机制——它可能通过增强预训练模型**虽已学习但未被重点利用**的特征，从而有效适配特定下游任务的需求

**LoRA 及其衍生的基于参数冻结与矩阵分解的微调技术，是目前最为重要的 SFT 技术之一。**

## Prompt Tuning：从离散到连续的提示

> Lester et al., *The Power of Scale for Parameter-Efficient Prompt Tuning*, EMNLP 2021.

Prompt Tuning 方法冻结了全部的原始模型参数，在新的 Prompt 上增加了可学习的 soft prompts。其思想是通过在 Prompt 中增加的 soft prompts 来让模型获得适应特定任务的能力。Prompt Tuning 可以被当作是 Prefix-Tuning 的一种特殊形式。

Prompt Tuning 的思想来自于 Prompt Engineering。由于精妙的提示词需要较多的人工参与，因此产生了使用微调模型参数来取代人工提示的思路。只在输入端增加的可学习 soft prompt 被端到端地训练，来浓缩关于任务类型的信息。

Prompt Tuning 相比于人工提示词，一般被称为一种**连续 prompt 技术**，因为其提示 token 在整个嵌入空间上是连续的，而人工提示的方法在嵌入空间上是离散的，因此其性能一般比人工提示更加优秀。

### 有价值的结论

- **随着模型规模增大，Prompt Tuning 与全模型微调的性能差距逐渐缩小**，并且比人工设计的提示要更加优秀
- 前缀 token 的数量不可以太短，也不能太长，控制在几十是比较合适的
- 效果不及 Prefix-Tuning，这引出了后续 P-Tuning v2 的改进
- 由于对原始参数结构的修改很少，Prompt Tuning 可能在输入分布于训练与评估阶段不同时（即领域偏移）更具鲁棒性
- **冻结通用语言理解参数，并将下游学习限制在轻量级参数范围内，有助于避免对特定领域的过拟合**

Prompt Tuning 在思想上的价值远远高于其在应用上的价值。**与此同时，这项技术一般只被用于 NLU 任务，不用于现在主流的类 GPT 结构。**

## P-Tuning v2：深层提示的回归

> Liu et al., *P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks*, ACL 2022.

本文的主要贡献是一项新颖的实证发现：**经过适当优化的 Prompt Tuning** 在各种模型规模和自然语言理解（NLU）任务上，其**性能均可与全参数微调（Fine-Tuning）相媲美**。

P-Tuning v2 放弃了只在输入嵌入层增加 soft prompt 的方案，而是在多个 Encoder 层中都增加了可以学习的前缀。P-Tuning v2 和 Prefix-Tuning 很接近，但 Prefix-Tuning 的高维前缀是通过一个 MLP 层从低维向量上重参数化得到的，而 P-Tuning v2 直接优化高维向量。

**P-Tuning v2** 明确采用**标准分类头**，可以非常自然地支持：
- 文本分类
- 命名实体识别（NER）
- 语义角色标注（SRL）

等涉及 NLU 的任务。相比于 LoRA 适合处理自然语言生成任务，需要各种提示手段来激活其在其他 NLU 任务上的能力，**P-Tuning v2 原生就适合处理这一类任务**。

### P-Tuning v2 的优势与发现

- **适用于小语言模型**：目前的微调着重研究 10B 以上的生成式语言模型，而 P-Tuning v2 在处理 NLU 领域不需要那么多参数就有很好的效果
- **深度提示的价值**：这意味着原本的 Prompt 方案在此处不足
- P-Tuning v2 在所有任务上普遍可与全微调相媲美

**正如 Prompt Tuning 一样，P-Tuning v2 也是适用于类似 BERT 的结构。而大型因果语言模型（类 GPT）在 NLU 任务上已经取得了极强的效果，考虑使用这些微调技术的真实应用价值值得商榷。**

## AdaLoRA：自适应秩分配

> Zhang et al., *AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning*, ICML 2023.

这是对 LoRA 方法的改进。和 QLoRA、LoRA+ 等非常符合直觉且简单的改进不同（分别是量化了原始参数以及修改了学习 LoRA 部分的学习率），AdaLoRA 希望引入一个 Adaptive 的 LoRA rank，减少那些不重要部分的 LoRA，从而提高微调的效率。对于判别重要性，AdaLoRA 使用了奇异值分解（SVD）来实现。

### SVD Adaptation

首先是基于 SVD 的 LoRA 分解，将原本的简单矩阵分解转化为 SVD 形式：

$$
W = W^{(0)} + \Delta = W^{(0)} + P \Lambda Q
$$

并且使用 $\mathcal{G}_i = \{ P_{*i}, \lambda_i, Q_{i*} \}$ 表示奇异值与奇异向量对，其中奇异值使用 0 初始化，向量采用 Gaussian 初始化，并增加了正交软约束：

$$
R(P, Q) = \| P^\top P - I \|_F^2 + \| Q Q^\top - I \|_F^2
$$

所有这些操作都是为了在进行 rank 自适应改变的时候保证优化正常进行，并减少高维矩阵进行 SVD 分解的计算开销。

我们将基于 SVD 的适配方法应用于每个 Transformer 层中的所有权重矩阵，包括 $W_q, W_k, W_v, W_{f_1}$ 和 $W_{f_2}$。为了控制参数预算，在训练过程中根据奇异值的重要性得分迭代地剪枝。

### Importance-based Rank Allocation

使用 $k$ 索引增量矩阵 $\Delta_k = P_k \Lambda_k Q_k$（$k = 1, \dots, n$），将 $\Delta_k$ 的第 $i$ 个三元组记为 $\mathcal{G}_{k,i} = \{P_{*,i}, \lambda_{k,i}, Q_{i,*}\}$，其重要性得分为 $S_{k,i}$。

增加正则项的训练目标函数为：

$$
\mathcal{L}(\mathcal{P}, \mathcal{E}, \mathcal{Q}) = \mathcal{C}(\mathcal{P}, \mathcal{E}, \mathcal{Q}) + \gamma \sum_{k=1}^n R(P_k, Q_k)
$$

在第 $t$ 步，首先执行一个随机梯度步骤来更新参数：

$$
\tilde{\Lambda}_k^{(t)} = \Lambda_k^{(t)} - \eta \nabla_{\Lambda_k} \mathcal{L}(\mathcal{P}^{(t)}, \mathcal{E}^{(t)}, \mathcal{Q}^{(t)})
$$

然后，给定重要性得分 $S_k^{t}$，奇异值按如下方式剪枝：

$$
\mathcal{T}(\tilde{\Lambda}_k^{(t)}, S_k^{(t)})_{ii} =
\begin{cases}
\tilde{\Lambda}_{k,ii}^{(t)} & \text{若 } S_{k,i}^{(t)} \text{ 在 } S^{(t)} \text{ 的前 } b^{(t)} \text{ 名内}, \\
0 & \text{其他情况}
\end{cases}
$$

其中 $S^{(t)}$ 包含了所有三元组的重要性得分，$b^{(t)}$ 是第 $t$ 步剩余奇异值的预算。通过这种方式，剪枝不那么重要的奇异值，将更多预算留给优先级更高的增量矩阵。

### 重要性度量

**奇异值的幅度**是最直接的量化方法，但无法恰当地量化参数对模型性能的贡献。作者提出了基于**敏感性**的重要性评分：

$$
S_{k,i} = s(\lambda_{k,i}) + \frac{1}{d_1} \sum_{j=1}^{d_1} s(P_{k,ji}) + \frac{1}{d_2} \sum_{j=1}^{d_2} s(Q_{k,ij})
$$

采用梯度-权重乘积的敏感性 $I(w_{ij}) = \left\lvert w_{ij} \nabla_{w_{ij}} \mathcal{L} \right\rvert$，并通过**敏感性平滑**和**不确定性量化**来解决波动问题：

$$
\begin{aligned}
\bar{I}^{(t)}(w_{ij}) &= \beta_1 \bar{I}^{(t-1)}(w_{ij}) + (1 - \beta_1) I^{(t)}(w_{ij}) \\
\bar{U}^{(t)}(w_{ij}) &= \beta_2 \bar{U}^{(t-1)}(w_{ij}) + (1 - \beta_2) \left\lvert I^{(t)}(w_{ij}) - \bar{I}^{(t)}(w_{ij}) \right\rvert
\end{aligned}
$$

最后定义重要性为二者的乘积：$s^{(t)}(w_{ij}) = \bar{I}^{(t)}(w_{ij}) \cdot \bar{U}^{(t)}(w_{ij})$。

### 全局预算调度

将预算 $b^{(t)}$ 定义为所有增量矩阵的总秩（即总奇异值数量）。从一个略高于目标预算 $b^{(T)}$ 的初始预算 $b^{(0)}$ 开始（例如 1.5 倍），将每个增量矩阵的初始秩设为 $r = b^{(0)}/n$。预热训练 $t_i$ 步后，遵循一个三次调度策略逐步降低预算 $b^{(t)}$ 直至达到目标。

特别的观察：AdaLoRA 总是将更多预算分配给 FFN 层和顶层的 LM Head，这也符合在 LoRA 微调研究中的发现——注意力层的重要性不及线性层，在条件允许的情况下应该尽可能为全部层提供 LoRA 适配器。

**AdaLoRA 目前是一种和传统 LoRA 一样实用的微调框架，在 PEFT 库中使用 `AdaLoraConfig` 来调用。**

## 小结与展望

本文梳理了 PEFT 领域从 Adapter 到 AdaLoRA 的核心方法演进。这些技术共同揭示了一个核心洞察：**预训练模型的参数空间存在高度冗余，精心设计的低维适配就足以释放其在下游任务上的潜力。**

从技术路线来看，PEFT 方法大体可以分为三个方向：
- **插入式**（Adapter）：在模型层间插入可训练模块
- **前缀/提示式**（Prefix-Tuning, Prompt Tuning, P-Tuning v2）：在输入或各层注入可学习的连续向量
- **重参数化式**（LoRA, AdaLoRA）：通过低秩分解直接修改权重矩阵

其中，LoRA 及其变体凭借零推理延迟和灵活性，已经成为当前最主流的 PEFT 方案。

展望未来，PEFT 的研究方向可能需要从单纯的高效适配向**适配 RL 的优化动力学**，开发新的微调范式。正如 Meta 在 Three-Gate Theory 中揭示的，RL 后训练的梯度更新遵循着与 SFT 截然不同的几何路径——它倾向于修改预训练参数空间中曲率极低的非主方向子空间，而非主成分方向。这意味着传统的基于低秩主权重更新的 LoRA 可能与 RL 的优化动力学存在冲突。**如何设计能够保护和利用这种非主方向更新特性的新型参数高效方法，将是 PEFT 下一阶段的核心问题。**
