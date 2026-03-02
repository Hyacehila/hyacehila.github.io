---
layout: blog-post
title: "Why Language Models Hallucinate"
date: 2026-02-24 20:00:00 +0800
series: "LLM ESSENCE"
categories: [LLM]
tags: [Hallucination]
author: Hyacehila
math: true
excerpt: "基于 OpenAI 团队论文《Why Language Models Hallucinate》：幻觉并非单纯源于数据噪声或模型缺陷，而是现代训练范式与二元评估机制带来的统计压力——系统性惩罚不确定性表达，从而奖励瞎猜。"
---

> 本文内容选自 OpenAI 团队 Paper《Why Language Models Hallucinate》

# Why Language Models Hallucinate

## 简单摘要

大型语言模型（LLMs）的“幻觉”现象——即生成过度自信但事实错误的陈述——长期以来被视为神秘且难以根除的缺陷。从概率论和统计学习理论的角度来看：**幻觉并非源于模型架构或训练算法的内在缺陷，而是现代训练范式中统计压力的自然结果。**特别地，论文证明即使训练数据完全无噪声，交叉熵损失的最优化仍会导致不可避免的生成错误。幻觉在后训练阶段持续存在的根本原因在于：**当前主流评估机制（二元 0-1 评分）系统性地惩罚不确定性表达（如“我不知道”），从而奖励了瞎猜行为。**

## 将生成问题转换为分类问题

### 概率视角下的生成模型

首先将语言模型生成的问题进行概率化。设 $\mathcal{X}$ 为所有 plausible 字符串（文本）的离散空间。我们将 $\mathcal{X}$ 划分为两个不相交集合：

$$
\mathcal{X} = \mathcal{V} \cup \mathcal{E}, \quad \mathcal{V} \cap \mathcal{E} = \emptyset
$$

- **有效串 $\mathcal{V}$**（Valid）：事实正确、逻辑一致的文本
- **错误串 $\mathcal{E}$**（Error）：包含事实错误或矛盾的文本（即幻觉）

设 $p$ 为真实世界的语言分布（训练数据分布），假设训练数据无噪声，即 $p(\mathcal{V}) = 1$。语言模型 $\hat{p}$ 是通过预训练得到的对 $p$ 的估计。

**定义 1（幻觉率）**：模型的幻觉率定义为模型生成错误串的概率：

$$
\text{err} := \hat{p}(\mathcal{E}) = \Pr_{x \sim \hat{p}}[x \in \mathcal{E}]
$$

### Is-It-Valid (IIV) 归约

核心思想：**生成有效输出比判断输出是否有效更困难**。若模型能完美生成有效内容，则它必须能正确回答“这个候选输出是否有效”的二元判断问题。

我们构造一个监督学习问题：IIV 分类任务。

**测试分布 $\mathcal{D}$**：

$$
\mathcal{D}(x) =
\begin{cases}
\frac{1}{2}p(x) & \text{若 } x \in \mathcal{V} \ (+) \\
\frac{1}{2\lvert \mathcal{E} \rvert} & \text{若 } x \in \mathcal{E} \ (-)
\end{cases}
$$

即：以 50% 概率从训练分布 $p$ 采样有效串（正样本），以 50% 概率从错误串集合 $\mathcal{E}$ 中均匀采样（负样本）。

**分类器构造**：给定语言模型 $\hat{p}$，定义 IIV 分类器：

$$
\hat{f}(x) =
\begin{cases}
+ & \text{若 } \hat{p}(x) > \frac{1}{\lvert \mathcal{E} \rvert} \\
- & \text{若 } \hat{p}(x) \leq \frac{1}{\lvert \mathcal{E} \rvert}
\end{cases}
$$

**IIV 分类错误率**：

$$
\text{err}_{\text{iiv}} := \Pr_{x \sim \mathcal{D}}[\hat{f}(x) \neq f(x)]
$$

其中 $f(x)$ 为真实标签（$+$ 表示有效，$-$ 表示错误）。

### 核心定理

**定理 1（无提示情形）**：对任意训练分布 $p$（满足 $p(\mathcal{V})=1$）和任意语言模型 $\hat{p}$，有：

$$
\text{err} \geq 2 \cdot \text{err}_{\text{iiv}} - \frac{\lvert \mathcal{V} \rvert}{\lvert \mathcal{E} \rvert} - \delta
$$

其中：

- $\delta = \lvert \hat{p}(A) - p(A)\rvert$ 为**校准误差**
- $A = \{x \in \mathcal{X} \mid \hat{p}(x) > 1/\lvert \mathcal{E} \rvert\}$ 为高于阈值的响应集合

现在我们需要迁移前面的结论到“有提示词”的生成场景下。

现实场景中，模型根据提示（prompt）$c \in \mathcal{C}$ 生成响应 $r$。设提示分布为 $\mu(c)$，训练分布为条件概率 $p(r \mid c)$。

对每个提示 $c$，定义：

$$
\mathcal{V}_c = \{r \mid (c,r) \in \mathcal{V}\}, \quad \mathcal{E}_c = \{r \mid (c,r) \in \mathcal{E}\}
$$

**关键参数**：

- $K = \min_c \lvert \mathcal{E}_c \rvert$：最简单的提示对应的错误响应数
- $k = \max_c \lvert \mathcal{V}_c \rvert$：最困难的提示对应的正确响应数

针对以上的条件分布变形以及在定理 1 中给出的相关结论，很容易将其内容迁移为条件概率的形式（证明见原论文附录）。

**测试分布**：

$$
\mathcal{D}(c,r) =
\begin{cases}
\frac{1}{2}\mu(c)p(r \mid c) & \text{若 } r \in \mathcal{V}_c \\
\frac{1}{2}\mu(c)\frac{1}{\lvert \mathcal{E}_c \rvert} & \text{若 } r \in \mathcal{E}_c
\end{cases}
$$

**分类器**：

$$
\hat{f}(c,r) = + \iff \hat{p}(r \mid c) > \frac{1}{\min_c \lvert \mathcal{E}_c \rvert}
$$

直接给出定理 1 的推广形式：

**定理 2（含提示情形）**：对任意 $p$（$p(\mathcal{V})=1$）和 $\hat{p}$，有：

$$
\text{err} \geq 2 \cdot \text{err}_{\text{iiv}} - \frac{\max_c \lvert \mathcal{V}_c \rvert}{\min_c \lvert \mathcal{E}_c \rvert} - \delta
$$

其中 $\delta = \lvert \hat{p}(A) - p(A)\rvert$，$A = \{(c,r) \mid \hat{p}(r \mid c) > 1/\min_c \lvert \mathcal{E}_c \rvert\}$。

对于现实世界的诸多情况，**计算学习理论已经给出了对应的 $\text{err}_{\text{iiv}}$ 下界**，这意味着在很多情况下，受制于我们给出的两个不等式，**语言模型的幻觉是不可能消失的：它们存在一个既定的下界，这个下界取决于对应二分类问题的错误率。**

## 后训练阶段幻觉

### 评估机制的激励扭曲

当前语言模型评估普遍采用**二元评分**（0-1 损失）：

- 正确答案：1 分或满分
- 错误答案或 “I don't know”（IDK）：0 分

**观察 1（二元评分的最优策略）**：对任意信念分布 $\rho_c$ 在正确答案上，IDK 响应的期望得分严格低于任何有非零正确概率的猜测响应。

**形式化证明**：设评分函数 $g_c: \mathcal{R}_c \to \{0,1\}$ 满足 $g_c(r) = 0$ 对所有 $r \in \mathcal{A}_c$（IDK 集合）。存在至少一个 $r^* \notin \mathcal{A}_c$ 使得 $\Pr_{g_c \sim \rho_c}[g_c(r^*)=1] > 0$。因此：

$$
\mathbb{E}_{g_c \sim \rho_c}[g_c(r^*)] > 0 = \mathbb{E}_{g_c \sim \rho_c}[g_c(\text{IDK})]
$$

**教学类比**：这就像大部分的标化考试——留空得 0 分，猜错不扣分，因此即使不知道答案，随机猜测也是期望收益最大化的策略。

### 主流评估的现状

**表 1：主流评估基准的评分方式**

| 基准      | 评估方式       | 二元评分 | IDK 得分                          |
| --------- | -------------- | -------- | -------------------------------- |
| GPQA      | 多选准确率     | 是       | 无                               |
| MMLU-Pro  | 多选准确率     | 是       | 无                               |
| IFEval    | 指令遵循验证   | 是       | 无                               |
| Omni-MATH | 数学等价性判断 | 是       | 无                               |
| SWE-bench | 单元测试通过   | 是       | 无                               |
| WildBench | LM 评分（1-10） | 否       | 部分（但低于含幻觉的 "fair" 响应） |

**结论**：论文统计指出，多数主流基准严格惩罚 IDK，导致模型在**整个评估体系中被激励去猜测**。

### 解决幻觉需要着手对所有主流评分 Benchmark 进行修改

现有的主流评估基准基本都采用二元评分方法。用于幻觉评估的 benchmark 虽然存在，但并不受到重视。想要在一定程度上解决 LLM 生成幻觉的问题（乃至彻底解决幻觉问题）的前提，是所有核心 benchmark 都将其纳入考量。只要 LLM 还能在测试时依靠随机猜测获得更高评分，那么幻觉问题就不可能得到解决：因为模型仍旧被奖励猜测，无法学习到回答 IDK。
