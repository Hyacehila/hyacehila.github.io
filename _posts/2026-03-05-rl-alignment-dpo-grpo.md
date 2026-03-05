---
layout: blog-post
title: "LLM 对齐中的强化学习：从 RLHF 到 DPO 与 GRPO"
date: 2026-03-05 13:10:00 +0800
categories: [LLM]
tags: [Reinforcement Learning, Alignment, RLHF]
author: Hyacehila
excerpt: "梳理 LLM 对齐技术，这是一篇等待进一步更新的草稿，我也不知道应该再讨论一些什么，毕竟各种 “PO”实在太多"
math: true
---

# LLM 对齐中的强化学习：从 RLHF 到 DPO 与 GRPO

## 从 RLHF 到更轻量的对齐方案

基于人类反馈的强化学习（RLHF）是当前大语言模型对齐的核心技术路线。传统的 RLHF 方法流程复杂：先训练奖励模型（Reward Model），再使用 PPO 等强化学习算法微调语言模型。这个过程计算量大、超参数多、训练不稳定。PPO 需要同时维护三个网络结构：

- **策略网络（Policy / Actor）**：用于被优化的目标模型
- **奖励网络（Reward）**：为每个输出生成奖励分数，通常使用和策略网络类似的初始化，但修改最后的 Head 来输出单一评分
- **价值网络（Value / Critic）**：用于生成 baseline，避免奖励的随机性导致梯度剧烈变化

在三个网络中，策略网络的规模不能减小，奖励网络有时候可以使用更小规模的网络，而对价值网络的规模缩减会导致算法优化出错。因此 PPO 实际上是一个非常昂贵的算法，需要同时维护三份规模近似原始策略网络的网络。

面对这些挑战，研究者们提出了更轻量的对齐方案。本文梳理其中最具影响力的两种：DPO 和 GRPO。

## DPO：直接偏好优化

> Rafailov et al., *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*, NeurIPS 2023.

### 核心改进

**DPO 发现：奖励函数和最优策略之间存在闭式解关系。**

通过 KL 约束的奖励最大化目标，可推导出：

$$
\pi_r(y \mid x) = \frac{1}{Z(x)}\pi_{ref}(y \mid x)\exp\left(\frac{r(x,y)}{\beta}\right)
$$

变换后，**奖励函数可隐式地表示为策略的函数**，无需单独训练奖励模型。

利用 Bradley-Terry 偏好模型，将奖励差项 $r(x,y_w) - r(x,y_l)$ 替换为策略的 log-ratio：

$$
\sigma\left(r(x,y_w) - r(x,y_l)\right) = \sigma\left(\beta \log\frac{\pi(y_w \mid x)}{\pi_{ref}(y_w \mid x)} - \beta \log\frac{\pi(y_l \mid x)}{\pi_{ref}(y_l \mid x)}\right)
$$

**分母 $Z(x)$ 被完美消去**，使目标仅依赖策略本身。

### 算法优化

将 RLHF 的两阶段（奖励模型学习 + 策略网络学习）压缩为**单阶段分类任务**：

$$
\mathcal{L}_{DPO}(\pi_\theta) = -\mathbb{E}_{(x,y_w,y_l)}\left[\log \sigma\left(\beta \log\frac{\pi_\theta(y_w \mid x)}{\pi_{ref}(y_w \mid x)} - \beta \log\frac{\pi_\theta(y_l \mid x)}{\pi_{ref}(y_l \mid x)}\right)\right]
$$

- **输入**：偏好数据对 $(y_w \succ y_l)$
- **目标**：最大化偏好对的分类概率
- **效果**：同时学习隐式奖励和最优策略

### 核心优势

- **无需 RL 训练循环**：规避 PPO 的采样和价值函数估计
- **无需显式奖励模型**：策略网络隐含奖励函数
- **训练稳定**：仅使用简单的交叉熵损失
- **计算高效**：训练时无需从策略采样
- **理论等价**：在 Bradley-Terry 模型下与 RLHF 优化目标一致
- **数据成本更低**：仅需要通用的偏好数据对，无需人工标注精准奖励

DPO 通过**动态重要性权重**调整偏好对的学习强度：
- 当模型低估人类偏好时（给 $y_l$ 的隐式奖励过高），梯度权重增大
- 当模型已正确排序时，权重自动衰减
- 天然防止过拟合和模式崩溃

## GRPO：群组相对策略优化

> Shao et al., *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*, 2024.

### 核心改进

**GRPO 算法**提出群组相对策略优化，通过组内输出归一化奖励替代传统 Critic 模型，解决 PPO 在 LLM 训练中价值函数难以准确估计的问题以及本身巨大的计算开销。在 GRPO 中，原本的 Policy-Reward-Critic 三网络架构中的 Critic 被移除。

文章还验证了，使用基于规则的 Reward（Rule-Based Reward）的性能不如使用神经网络 Reward。前者的性能会直接退化为类似 RFT（拒绝采样微调）的情形。

由于使用组间相对价值来代替原本的 Critic 实现梯度的平滑，因此原本放在 Reward 中的 KL 散度就不合适了，新的优势计算将 KL 带来的损失放到 Reward 之外。

### 算法优化

**迭代循环**：
- 将当前策略模型设为参考模型
- 对每个训练批次：
  - 从策略模型采样 G 个输出（原文使用 G=64）
  - 用奖励模型为每个输出打分
  - 计算组内相对优势（Group Relative Advantage）：
    - **结果监督**：所有 token 共享归一化的最终奖励
    - **过程监督**：每个 token 累积后续步骤的归一化奖励
  - 使用 GRPO 目标函数更新策略模型
- 在重复了一定批次之后，用 replay 机制（10% 历史数据）持续训练奖励模型，让奖励模型也在训练过程中更新，从而获得更契合新策略网络的奖励

使用数学这个容易基于规则进行评价的领域来构建 Reward，然后基于此训练神经网络的 Reward。在节省计算开销的同时得到了更好的性能。

### 核心优势

移除 Critic 网络带来更优化的性能与完全不落下风的最终精度。作为 Online 优化方法，比 RFT 等 Offline 方法显著获得更好的性能。

### 关于 RL 本身的发现

1. **Online 优于 Offline**：实时探索能防止分布偏移（Distribution Shift），让模型持续在舒适区边缘学习
2. **细粒度反馈优于粗粒度筛选**：RL 方法（GRPO/PPO）不仅提升了模型选出正确答案的能力，还通过惩罚错误路径优化了生成概率分布——即使是 Rule-Based 也要去训练神经网络 Rewarder
3. **RL 主要提升 Maj@K 而非 Pass@K**：RL 更多是在对齐（Alignment）——让模型更倾向于输出它本身知道的正确答案，而不是凭空注入新知识
4. **过程监督的潜力**：GRPO 结合过程奖励（Process Reward）比仅用结果奖励效果更好，因为数学题步骤多，过程监督能提供更密集的梯度信号
5. **迭代式 RL**：随着 Policy 变强，旧的 Reward Model 可能不够用。进行多轮迭代（更新 Policy → 采样新数据训练 Reward Model → 再更新 Policy）能进一步提升效果

## 小结与展望

从 PPO 到 DPO 再到 GRPO，LLM 对齐领域的技术演进呈现出一条清晰的脉络：**不断简化训练流程，同时保持甚至提升对齐效果**。

- **DPO** 通过发现奖励函数与最优策略的闭式解关系，将两阶段的 RLHF 压缩为单阶段分类任务，彻底消除了对显式奖励模型的依赖
- **GRPO** 进一步移除了 Critic 网络，用组内相对优势替代价值函数估计，在大幅降低计算开销的同时维持了训练质量

展望未来，RL 对齐的研究方向将转向**根据任务需求寻找更加有效的通用算法**。当前的 DPO 和 GRPO 虽然在效率上实现了突破，但仍然面临诸多挑战：如何在不同类型的任务（推理、创作、对话）上自适应地调整对齐策略？如何设计能够同时优化多个目标（准确性、安全性、有用性）的统一框架？如何在保持探索能力的同时避免策略熵坍缩？这些问题都指向一个更通用、更鲁棒的 RL 对齐范式。
