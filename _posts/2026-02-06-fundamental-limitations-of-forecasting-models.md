---
layout: blog-post
title: 基础预测模型的基本限制：多模态与严谨评估的必要性
title_en: "Fundamental Limits of Foundation Forecasting Models: Multimodality and Rigorous Evaluation"
date: 2026-02-06 12:00:00 +0800
categories: [机器学习]
tags: [Time Series, Methodology]
author: Hyacehila
excerpt: 深度学习在时间序列预测中真的总是有效吗？本文基于 Christoph Bergmeir 在 NeurIPS 2024 的演讲，讨论基础预测模型的局限性、评估中的陷阱，以及为什么时间序列预测需要引入多模态上下文。
excerpt_en: "Are deep learning methods always effective for time-series forecasting? Based on Christoph Bergmeir's NeurIPS 2024 talk, this post discusses model limits and evaluation traps."
---

> 本文内容基于 **Christoph Bergmeir** 在 **NeurIPS 2024** 上的演讲 *"Fundamental limitations of foundational forecasting models - The need for multimodality and rigorous evaluation"*。
>
> Christoph Bergmeir 教授是莫纳什大学时间序列预测库 (Monash Time Series Forecasting Repository) 的主要维护者之一。

# Fundamental limitations of foundational forecasting models - The need for multimodality and rigorous evaluation

## 引言：从随机游走谈起

时间序列预测是数据科学中最基础但也最棘手的话题之一。不同于图像或自然语言处理，时间序列往往面临着信噪比极低的问题。

如果面对的是一个完全随机游走 (Random Walk) 的序列——即没有任何可发现的内在模式——那么理论上对这个序列最好的预测就是 **Naive 预测**，即用前一刻的观测值作为下一刻的预测值 ($y_{t+1} = y_t$)。在这种情况下，无论模型多复杂（如 BP 神经网络、SVM、随机森林等），表现往往都不如最简单的 Naive 方法。

这个看似简单的道理，却在当今的深度学习预测研究中经常被忽视。

## 金融领域的“虚假 SOTA”

股票市场是随机游走的典型例子。金融领域的有效市场假说 (EMH) 认为，股价不是过去价格的函数，而是未来预期的体现。股价所包含的信息几乎已经反映在当前公开信息中，未来变动主要受不可预测的新信息影响。

因此，股价往往被视为具有鞅 (Martingale) 性质。在这种假设下，**超出 Naive 预测的精度在理论上几乎是不可能的**。实际上，金融领域的量化研究往往并不关注单纯的点预测（即明天股价是多少），而是更关注**风险 (Risk)** 与**波动 (Volatility)**。

许多发表在顶尖会议上的文章声称自己在金融预测领域实现了 SOTA (State of the Art)。但仔细审视这些论文，会发现它们往往只与其他深度学习 (DL) 方法比较，却忽略了 Naive 预测这一强基准。模型的计算时间不断膨胀，但性能上未必有实质性突破。

## 气象预测与错误的基准

除了金融数据，深度学习研究者也热衷于在天气与电力数据上展开预测。但这里同样存在常识性的误区。

气象学家普遍认为，由于混沌效应的存在，**超过两周（14天）的长期天气预测是物理上不可能的**。因此，任何声称能进行长期（大于两周）精确逐小时天气预测的模型，基本都在拟合噪声或随机猜测。

翻开相关论文会发现，它们确实比较了 ARIMA 或 ETS 等传统统计方法，但基准常常设置不当。例如，面对具有复杂季节性（如小时级数据）的天气序列时，简单的 ARIMA 并不是合适对手。更强的基准应该是 **DHR-ARIMA (Dynamic Harmonic Regression with ARIMA errors)**。引入这种复杂度较高但更适合该类型数据的统计模型后，那些所谓的 SOTA 深度学习模型往往会败下阵来。

## 评估陷阱：Drop Last Trick

为了让自己的模型效果“看起来”更好，部分学者甚至在评估流程上动起了脑筋。一个典型的例子就是 **"Drop Last Trick"**。

在许多深度学习的时间序列库中，处理数据时往往会将数据集划分为多个 Batch。如果测试集的最后一个 Batch 不满，部分代码库（如错误配置的 `DataLoader`）会默认将其丢弃。

但在时间序列预测中，数据是有序的。**测试集的最后一部分数据，往往是最新、最接近当下的数据，也是最具参考价值的数据**。随意丢弃这部分数据，会导致评估结果严重失真。许多文章里，同一个方法在不同论文中的性能差异巨大，常常正是因为使用了不同的（甚至不严谨的）测试标准来制造所谓的 SOTA，而实际效果可能远不如几十年前的 Baseline。

## 全局模型 (Global Models) 的双刃剑

近年来，利用大量的多来源时间序列构建**全局模型 (Global Models)** 成为了趋势。只有深度学习技术能够有效处理这种海量数据。

有研究表明，即使在毫不相关的数据上训练全局模型，再进行领域微调，也可能优于本地模型。这种思想在基础统计领域早有对应，被称为 **James-Stein 悖论**。它允许我们利用无关数据改进预测效果，通过引入偏差 (Bias) 来换取方差 (Variance) 降低，这也是正则化理论的基础。

**但是，全局上有效的模型并不能保证在特定的局部数据上有效。**

目前的时间序列大模型 (Foundation Model for Time Series) 往往面临一个问题：算法在学习过程中会将各种来源的数据混为一谈。这种“平均化”的处理方式，导致模型无法结合真实情景和对应训练数据做出精准预测。**算法会把不同的隐含模式平均掉，而不是分别加以利用**；数据来源中的信息，也随之被抹平。

虽然语言模型也通过预训练学习通用的语言模式，但我们可以在一轮轮对话中通过 Prompt 纠正模型。而在纯数值的时间序列预测中，这种“即时纠正”要困难得多。

## 出路：上下文与多模态 (Context is King)

那么，基础预测模型的下一步在哪里？

Christoph Bergmeir 指出，**单纯的时间序列缺少足够的可以被利用的信息**。如果仅凭历史数据，我们很难突破随机游走的限制。

想要在 **LLM4TS (Large Language Models for Time Series)** 上解决这个问题，需要利用 **上下文 (Context)**。这里的上下文不只是更长的历史窗口，而是**多模态 (Multimodality)** 信息的引入——包括文本新闻、宏观经济报告、图像数据等。

只有当模型能够理解“公司突发丑闻”这个文本 Context 时，它才有可能预测出股价的暴跌；只有当模型能够结合实时的气象云图，它才可能突破纯数值预测的瓶颈。

**这就是 TS 基准模型更可行的路线：从单模态的数值拟合，走向多模态的上下文理解。**
