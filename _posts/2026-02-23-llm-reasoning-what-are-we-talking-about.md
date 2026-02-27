---
layout: blog-post
title: "当我们在谈论 LLM Reasoning 时，我们在谈论什么"
date: 2026-02-23 20:00:00 +0800
series: "LLM ESSENCE"
categories: [LLM]
tags: [Learning, RL]
author: Hyacehila
math: true
excerpt: "围绕 CoT 与测试时计算（TTS）的一个理解：CoT 通过显式中间步骤把 OOD 难题拆解成更接近 ID 的子任务，从而控制泛化误差；并从策略熵的角度讨论推理与对齐中的探索/利用平衡。"
---

> 本文内容来自 JiaXuan Zou 的博客。

# 当我们在谈论 LLM Reasoning 时，我们在谈论什么

当前，诸如思维链（Chain-of-Thought, CoT，latent CoT）等主流的大语言模型（LLM）推理方法，可以被宽泛地归类为一种测试时计算/扩展（Test-Time Compute/Scaling, TTS）。这一概念的思想源头可以追溯到 Graves 和 Ling et al. 等人的研究。但若仅以 TTS 概括仍显笼统，本文尝试进一步探究其背后的机理。

## CoT 为什么有效？

CoT 通过强制模型分解子任务，将原本难以拟合的 OOD 数据向 ID 数据“拉近”，从而提升系统泛化能力。相比于直接进行端到端训练，采用 CoT 的训练方式能够显著加速模型收敛，并有效提升从 ID 数据到 OOD 数据的推理泛化能力。

我们可以用数学语言更精确地描述这一过程。假设模型经过充分训练，在 ID 测试集上的分布与训练集分布已非常接近，即 $D_{\text{KL}}(P_{\text{test}}^{\text{ID}} \Vert P_{\text{train}}) \to 0$。则期望的泛化误差主要由 OOD 成分主导：

$$
\overline{\text{error}} \leq \sqrt{\frac{2R^2\alpha}{N} D_{\text{KL}}(P_{\text{test}}^{\text{OOD}}(Y \mid X) \Vert P_{\text{train}}(Y \mid X))}.
$$

此公式揭示了一个关键点：即便模型完美学习了 ID 模式，其最终性能仍从根本上受限于在 OOD 数据上的泛化能力，误差与 $\sqrt{\alpha D_{KL}^{OOD}}$ 成正比。

为了改善 OOD 泛化，CoT 引入了中间推理步骤 $C_i$。此时，条件概率 $P(Y \mid X)$ 可以被分解为 $\sum_i P(Y \mid X, C_i) \cdot P(C_i \mid X)$。相应地，泛化误差的上界也可以被分解为两个部分：

$$
\overline{\text{error}}^2 \leq \frac{2R^2\alpha}{N} \left[ D_{\text{KL}}(P_{\text{test}}^{\text{OOD}}(C_i \mid X) \Vert P_{\text{train}}(C_i \mid X)) + \mathbb{E}_{C_i \sim P_{\text{test}}^{\text{OOD}}(C_i \mid X)} \left[ D_{\text{KL}}(P_{\text{test}}^{\text{OOD}}(Y \mid X, C_i) \Vert P_{\text{train}}(Y \mid X, C_i)) \right] \right]
$$

LLM 在 OOD 数据上的泛化挑战，可以拆解为两个更易于控制的子问题：

1. 给定输入 $X$，生成合理的中间推理步骤 $C$ 的能力。
2. 给定输入 $X$ 与中间步骤 $C$，生成最终答案 $Y$ 的能力。

在不采用 CoT 的标准问答模式下，模型必须直接从 $X$ 映射到 $Y$。中间的隐式推理步骤近似于一个难以学习的均匀分布，导致整体的 KL 散度难以控制。

而 **CoT 通过显式地生成中间步骤 $C$，将一个复杂的 OOD 问题转化为一系列更接近 ID 的、更小范围的映射，从而有效地控制了泛化误差。**

## 关于 entropy 的讨论

在探讨了理论框架之后，我们转向经验性的观察：特别是信息论中熵在模型推理过程中的作用。熵衡量了一个概率分布的不确定性。

在通过强化学习（RL）进行模型对齐时，一个普遍观察到的现象是**策略熵的崩溃**。策略熵（平均 token 级熵）定义为：

$$
H(\pi_\theta, D) = -\mathbb{E}_{D, \pi_\theta}[\log \pi_\theta(y_t \mid y_{<t})].
$$

**策略熵的崩溃**：大量实验观察到，在没有进行熵干预的大量 RL 运行中，策略熵在训练初期急剧下降，导致策略模型过于自信，丧失探索能力。研究发现，奖励（performance）的提升往往是以消耗熵为代价的。这表明，为了维持模型的探索能力并持续优化，必须进行有效的熵管理。

**直观地理解：当一个高概率的动作获得了高奖励（高优势函数）时，模型会进一步强化这个选择，从而降低整体熵（利用）；反之，当一个低概率的动作意外获得高奖励时，模型会提升其概率，从而增加整体熵（探索）。**

除了宏观的策略熵，微观的 token 级熵也揭示了推理的奥秘。研究（Wang et al.）观察到，在 CoT 的生成过程中，熵的分布极不均匀：

- 少数 token 以高熵生成：这些 token 通常是连接不同推理片段、充当推理路径分叉点的关键节点，对后续内容的走向有决定性影响。
- 多数 token 以低熵输出：这些 token 多用于完成单词构造或收尾当前句子，其选择相对确定。

更有趣的是，当使用强化学习（如 RLVR）对模型进行微调时，熵的调整主要发生在那些原本就处于高熵的 token 上。基于此的实验发现，如果只对高熵 token 进行强化学习，其性能与对全部 token 进行学习相当，甚至略优。

