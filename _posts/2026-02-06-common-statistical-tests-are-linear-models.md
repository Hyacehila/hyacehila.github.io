---
layout: blog-post
title: 常见的统计检验本质上都是线性模型 (Common statistical tests are linear models)
date: 2026-02-07 12:00:00 +0800
categories: [统计学]
tags: [Linear Regression, Hypothesis Testing]
excerpt: 转载自 Jonas Kristoffer Lindeløv 的文章。揭示了 t 检验、ANOVA、卡方检验等常用统计方法背后的统一线性模型原理。
---

> 本文转载自 Jonas Kristoffer Lindeløv 的精彩文章 [Common statistical tests are linear models](https://lindeloev.github.io/tests-as-linear/)
> 原文深入浅出地揭示了统计学中一个令人惊讶的简单真理：大多数常用的统计检验（t检验、相关分析、ANOVA、卡方检验等）其实都是线性模型的特例。

# Common statistical tests are linear models

## 核心概念：万物皆线性

大多数常用的统计模型（t-test, correlation, ANOVA; chi-square, etc.）都是线性模型的特例，或者是非常近似的特例。这意味着我们不需要死记硬背每一个检验的假设和公式，因为它们本质上都可以归结为我们高中就学过的公式：

$$y = a \cdot x + b$$

这种简单的美感极大地简化了统计学的理解。不论是频率学派、贝叶斯学派还是基于置换的推断，核心的线性模型都是一致的。

对于所谓的“非参数检验 (non-parametric tests)”，我们也可以用一种更直观的方式来理解：它们通常只是在**秩变换 (rank-transformed)** 数据上运行的对应参数检验。与其认为非参数检验“不需要假设”，不如将其理解为“在排名 (ranks) 上进行计算”。

下图完美总结了这一观点（点击查看[PDF版本](https://lindeloev.github.io/tests-as-linear/linear_tests_cheat_sheet.pdf)）：

![Linear Tests Cheat Sheet](https://lindeloev.github.io/tests-as-linear/linear_tests_cheat_sheet.png)

---

## 相关性 (Pearson and Spearman)

### 理论：作为线性模型

相关性分析的本质，就是寻找最佳拟合直线。模型公式如下：

$$y = \beta_0 + \beta_1 x \qquad \mathcal{H}_0: \beta_1 = 0$$

这其实就是我们熟悉的 $y = ax + b$。在 R 语言中，我们通常写成 `y ~ 1 + x`，这代表 $y = 1 \cdot \beta_0 + x \cdot \beta_1$。不论怎么写，它的核心就是截距 ($\beta_0$) 和斜率 ($\beta_1$)。

### 秩变换 (Rank-transformation) 与 Spearman

Spearman 相关系数实际上就是对 $x$ 和 $y$ 进行**秩变换 (Rank-transformation)** 后的 Pearson 相关系数：

$$rank(y) = \beta_0 + \beta_1 \cdot rank(x) \qquad \mathcal{H}_0: \beta_1 = 0$$

所谓“秩 (Rank)”，就是把数值替换为它们的大小排名（最小的为1，第二小为2...）。虽然 Spearman 的 p 值在小样本时只是近似值，但当 N > 10 时通常已经足够准确。

### R 代码对照：Pearson

运行下面的 R 代码，你会发现线性模型 (`lm`) 产生的 $t$, $p$ 值与内置的 `cor.test` 完全一致。

唯一的小区别是：`lm` 给出的是斜率，而 `cor.test` 给出的是相关系数 $r$。如果我们将数据标准化（使 SD=1），那么斜率就等于 $r$。

```r
# Built-in t-test
a = cor.test(y, x, method = "pearson")

# Equivalent linear model: y = Beta0*1 + Beta1*x
b = lm(y ~ 1 + x)

# On scaled vars to recover r
c = lm(scale(y) ~ 1 + scale(x))
```

### R 代码对照：Spearman

同样的逻辑适用于 Spearman 相关，只需先对数据进行 `rank()` 变换：

```r
# Spearman correlation
a = cor.test(y, x, method = "spearman")

# Equivalent linear model
b = lm(rank(y) ~ 1 + rank(x))
```

---

## 单均值 (One Mean)

### 理论：作为线性模型

单样本 T 检验 (One-sample t-test) 测试样本均值是否显著异于 0。这实际上是一个**只有截距**的线性模型：

$$y = \beta_0 \qquad \mathcal{H}_0: \beta_0 = 0$$

这里没有 $x$（或者说 $x=0$），所以剩下的 $\beta_0$ 就是均值。

对于非参数对应的 **Wilcoxon 符号秩检验 (Wilcoxon signed-rank test)**，原理相同，只是应用于**符号秩 (signed ranks)** 数据上：

$$signed\_rank(y) = \beta_0$$

### R 代码对照：单样本 T 检验

```r
# Built-in t-test
a = t.test(y)

# Equivalent linear model: intercept-only
b = lm(y ~ 1)
```

你会发现 `lm(y ~ 1)` 的输出中，截距项的估计值 (Estimate) 就是均值，t 值和 p 值也与 `t.test` 的结果完全一致。

### R 代码对照：Wilcoxon 符号秩检验

```r
# Built-in
a = wilcox.test(y)

# Equivalent linear model
b = lm(signed_rank(y) ~ 1)

# Bonus: also works for one-sample t-test on signed ranks
c = t.test(signed_rank(y))
```

使用 `lm` 不仅能得到匹配的 p 值，还能直接得到“平均符号秩”，这是一个比单纯的 W 统计量更直观的数字。

---

## 其他常用检验汇总

除了上述两个详细示例，其他常见的统计检验也都可以完美映射到线性模型中。为了保持本文简洁，以下列出摘要对照表，点击链接可查看原文详细推导和代码。

| 统计检验 (Test) | 线性模型公式 (Simulated LM) | 原文链接 |
| :--- | :--- | :--- |
| **双均值 (Two means)** <br> (Independent t-test) | $y = \beta_0 + \beta_1 x$ <br> ($x$ 是二分类变量) | [Link](https://lindeloev.github.io/tests-as-linear/#means2) |
| **方差分析 (Three or more means)** <br> (One-way ANOVA) | $y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ...$ <br> (使用虚拟变量编码 Dummy coding) | [Link](https://lindeloev.github.io/tests-as-linear/#means3) |
| **协方差分析 (ANCOVA)** | $y = \beta_0 + \beta_1 x_{categorical} + \beta_2 x_{continuous}$ | [Link](https://lindeloev.github.io/tests-as-linear/#ancova) |
| **比例与卡方 (Proportions / Chi-square)** | $\ln(y) = \beta_0$ <br> (Log-linear models,使用 Poisson 回归) | [Link](https://lindeloev.github.io/tests-as-linear/#proportions) |

---

## 总结

理解这些检验背后的线性模型本质，能让我们摆脱对特定“检验名称”的依赖，转而关注模型的构建。无论是 t 检验还是复杂的 ANOVA，它们都在回答同一个问题：我的模型参数是否显著不为零？

感谢 **Jonas Kristoffer Lindeløv** 提供的精彩视角。

*   **原文链接**: [Common statistical tests are linear models](https://lindeloev.github.io/tests-as-linear/)
*   **Python 版本**: [Tests as linear (Python)](https://eigenfoo.xyz/tests-as-linear/)
*   **GitHub 仓库**: [lindeloev/tests-as-linear](https://github.com/lindeloev/tests-as-linear)