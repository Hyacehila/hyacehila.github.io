---
layout: blog-post
title: "隐马尔可夫模型：时间序列的概率解析"
date: 2026-02-10 12:00:00 +0800
categories: [统计学]
tags: [Learning]
series: "概率图模型 (Probabilistic Graphical Models)"
author: Hyacehila
excerpt: 当概率图模型遇上时间序列，便诞生了能够描述动态系统的隐马尔可夫模型 (HMM)。本文深入解析了 HMM 的双重随机过程结构，并详细推导了解决评估、解码与学习这三大核心问题的数学算法。
math: true
---

# 隐马尔可夫模型：时间序列的概率解析

## 引言：从静态到动态

在上一篇关于贝叶斯网络的讨论中，我们处理的数据大多是静态的 (Static)——即假设样本之间是独立同分布 (i.i.d.) 的。然而，现实世界充满了**序列 (Sequential)** 数据：语音是一连串的声波，文本是一连串的单词，股票是一连串的价格。

在这些数据中，前后的观测值往往存在强烈的依赖关系。为了对这种随时间演变的动态系统建模，我们需要引入“时间”维度。**隐马尔可夫模型 (Hidden Markov Model, HMM)** 正是这类**动态贝叶斯网络 (Dynamic Bayesian Network)** 的最简化与典型代表。

## 模型定义：双重随机过程

HMM 描述了一个含有隐含未知参数的马尔可夫过程。它的核心思想在于**双重随机过程**：

1.  **隐状态序列 (Hidden State Sequence)** $Q = \{q_1, q_2, \dots, q_T\}$：
    这是系统在不同时刻所处的真实状态，但我们无法直接观测到。这些状态的转移满足**马尔可夫性**。
2.  **观测序列 (Observation Sequence)** $O = \{o_1, o_2, \dots, o_T\}$：
    这是我们在每个时刻观察到的数据。它仅由当前的隐状态决定。

### HMM 的五元组

一个完整的 HMM 由以下五个要素确定，记为 $\lambda = (N, M, A, B, \pi)$：

*   **状态集合** $S = \{s_1, \dots, s_N\}$：隐状态可能取值的集合（如：天气及其 {晴，雨，阴}）。
*   **观测集合** $V = \{v_1, \dots, v_M\}$：观测值可能取值的集合（如：活动及其 {散步，清洁，购物}）。
*   **状态转移矩阵** $A = [a_{ij}]$：
    $$ a_{ij} = P(q_{t+1} = s_j \mid q_t = s_i) $$
    表示从状态 $i$ 转移到状态 $j$ 的概率。
*   **观测发射矩阵 (Emission Probability)** $B = [b_j(k)]$：
    $$ b_j(k) = P(o_t = v_k \mid q_t = s_j) $$
    表示在状态 $j$ 下观测到符号 $k$ 的概率。
*   **初始状态分布** $\pi = [\pi_i]$：
    $$ \pi_i = P(q_1 = s_i) $$

### 两个基本假设

1.  **齐次马尔可夫假设**：任意时刻 $t$ 的状态只依赖于前一时刻 $t-1$ 的状态，与更早的状态无关。
    $$ P(q_t \mid q_{t-1}, o_{t-1}, \dots, q_1, o_1) = P(q_t \mid q_{t-1}) $$
2.  **观测独立性假设**：任意时刻 $t$ 的观测值只依赖于该时刻的状态 $q_t$，与其他时刻的状态或观测无关。
    $$ P(o_t \mid q_T, o_T, \dots, q_1, o_1) = P(o_t \mid q_t) $$

## HMM 的三个核心问题

HMM 的应用主要围绕三个经典问题展开：

### 概率计算问题 (Evaluation)

**问题**：给定模型 $\lambda = (A, B, \pi)$ 和观测序列 $O$，计算该序列出现的概率 $P(O \mid \lambda)$。

直接计算需要遍历所有可能的隐状态序列 $Q$，复杂度高达 $O(N^T \cdot T)$，这在计算上是不可行的。我们采用动态规划思想的**前向算法 (Forward Algorithm)**。

定义**前向概率** $\alpha_t(i)$：时刻 $t$ 观测序列为 $o_1, \dots, o_t$ 且状态为 $s_i$ 的概率。
$$ \alpha_t(i) = P(o_1, \dots, o_t, q_t = s_i \mid \lambda) $$

**递推公式**：
1.  **初值**：
    $$ \alpha_1(i) = \pi_i b_i(o_1) $$
2.  **递推**：(对于 $t = 1, \dots, T-1$)
    $$ \alpha_{t+1}(j) = \left[ \sum_{i=1}^N \alpha_t(i) a_{ij} \right] b_j(o_{t+1}) $$
3.  **终值**：
    $$ P(O \mid \lambda) = \sum_{i=1}^N \alpha_T(i) $$

该算法将复杂度降低到了 $O(N^2 \cdot T)$。

### 解码问题 (Decoding)

**问题**：给定模型 $\lambda$ 和观测序列 $O$，寻找最有可能产生该观测序列的隐状态序列 $Q^*$。即求 $\arg\max_Q P(Q \mid O, \lambda)$。

这是典型的最优路径规划问题，使用**维特比算法 (Viterbi Algorithm)** 求解。

定义 $\delta_t(i)$：在时刻 $t$ 状态为 $s_i$ 的所有路径中，概率最大的那条路径的概率。

**递推公式**：
1.  **初值**：
    $$ \delta_1(i) = \pi_i b_i(o_1), \quad \psi_1(i) = 0 $$
2.  **递推**：(寻找到达状态 $j$ 的最大概率来源)
    $$ \delta_t(j) = \max_{1 \le i \le N} [\delta_{t-1}(i) a_{ij}] b_j(o_t) $$
    $$ \psi_t(j) = \arg\max_{1 \le i \le N} [\delta_{t-1}(i) a_{ij}] $$ (记录路径回溯点)
3.  **回溯**：
    最优路径终点 $P^* = \max_i \delta_T(i)$，终点状态 $q_T^* = \arg\max_i \delta_T(i)$。
    从 $t=T-1$ 到 $1$ 倒推：$q_t^* = \psi_{t+1}(q_{t+1}^*)$。

### 学习问题 (Learning)

**问题**：已知观测序列 $O$，估计模型参数 $\lambda = (A, B, \pi)$ 使得 $P(O \mid \lambda)$ 最大化。

由于包含隐变量，无法直接使用 MLE。这是 **EM 算法 (Expectation-Maximization)** 的经典应用场景。在 HMM 中，该算法被称为 **Baum-Welch 算法**。

**E 步 (Expectation)**：
计算两个统计量（利用前向变量 $\alpha$ 和后向变量 $\beta$）：
*   $\xi_t(i, j)$：时刻 $t$ 处于状态 $i$ 且时刻 $t+1$ 处于状态 $j$ 的概率。
    $$ \xi_t(i, j) = P(q_t=i, q_{t+1}=j \mid O, \lambda) = \frac{\alpha_t(i) a_{ij} b_j(o_{t+1}) \beta_{t+1}(j)}{\sum_{k=1}^N \sum_{l=1}^N \alpha_t(k) a_{kl} b_l(o_{t+1}) \beta_{t+1}(l)} $$
*   $\gamma_t(i)$：时刻 $t$ 处于状态 $i$ 的概率。
    $$ \gamma_t(i) = \sum_{j=1}^N \xi_t(i, j) $$

**M 步 (Maximization)**：
更新参数：
*   **状态转移概率**：
    $$ \hat{a}_{ij} = \frac{\sum_{t=1}^{T-1} \xi_t(i, j)}{\sum_{t=1}^{T-1} \gamma_t(i)} $$
    (直观理解：从 $i$ 转移到 $j$ 的期望次数 / 从 $i$ 出发的期望总次数)
*   **观测概率**：
    $$ \hat{b}_j(k) = \frac{\sum_{t=1, o_t=v_k}^T \gamma_t(j)}{\sum_{t=1}^T \gamma_t(j)} $$

## 案例：词性标注 (Part-of-Speech Tagging)

HMM 在自然语言处理中有广泛应用，词性标注即为一例。

*   **观测值**：单词序列（如 "I love data"）。
*   **隐状态**：词性标签（如 "Pronoun", "Verb", "Noun"）。
*   **目标**：给定句子，推断最可能的词性序列（解码问题）。

通过在大规模标注语料库上统计单词与词性的共现频率（估计 $B$）以及词性间的转移频率（估计 $A$），我们就可以利用维特比算法从新的句子中恢复出词性结构。

## 总结

隐马尔可夫模型通过引入隐状态和两个独立性假设，巧妙地解决了时间序列的建模问题。前向算法解决了计算效率问题，维特比算法解决了推断问题，Baum-Welch 算法解决了无监督学习问题。

尽管现代深度学习（如 RNN, LSTM, Transformer）在许多任务上已经超越了 HMM，但 HMM 提供的概率图框架和动态规划思想依然是理解序列模型的基石。

在下一篇文章中，我们将把视线从有向图移开，探讨**马尔可夫随机场 (MRF)**，看看当箭头消失、图结构变为无向时，概率图模型又展现出怎样的特性。
