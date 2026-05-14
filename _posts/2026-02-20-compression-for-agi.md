---
layout: blog-post
title: Compression for AGI：压缩即智能
title_en: "Compression for AGI: Compression as Intelligence"
date: 2026-02-20 20:00:00 +0800
categories: [基础模型]
tags: [Pre-Training, Model Mechanics]
author: Hyacehila
excerpt: 整理 Jack Rae 在《Compression for AGI》中的观点：基础模型训练可以理解为对有效信息的无损压缩；压缩率越高（loss 越低），模型越可能呈现更强的泛化行为。
excerpt_en: "A summary of Jack Rae's Compression for AGI: foundation-model training as lossless compression of useful information, and why lower loss can imply stronger generalization."
math: true
---

# Compression for AGI：压缩即智能

**主题：压缩即智能：为什么 ChatGPT 拥有智能？** 本文观点来自 OpenAI 研发人员 Jack Rae 的主题分享《Compression for AGI》。文章讨论的主线是：**基础模型训练可以理解为尽可能无损地压缩有效信息。**

## 关于通用人工智能

在解释为什么“压缩”是一种实现通用人工智能（Artificial General Intelligence, AGI）的路径之前，我们先简单回顾一个经典思想实验：中文房间（John Searle, 1980）。

> 将一个对中文毫无了解、只会说英语的人关在一个只有一个小窗的封闭房间里。房间里有一本记录着中英文翻译的手册，还有足够的稿纸和铅笔。写着中文的纸片通过小窗口被送入房间中。房间里的人可以使用手册把中文翻译成英文理解，再用手册把英文翻译回中文作答。虽然他完全不会中文，但房间外的人会以为他能流利地使用中文。

这样一个“巨大手册”显然对应着很低的智能：一旦遇到手册里没有覆盖的输入，它就无法应对。

如果我们能够从大量数据中提取语法与规则，手册就可以变得更精简；与此同时，系统的智能水平更高（泛化能力更强）。

手册越厚，智能越弱；手册越薄，智能越强。就像公司雇一个人：能力越强，你需要解释得越少；能力越弱，你需要解释得越多。

上面的例子直观地解释了为什么“压缩即智能”：去获得更小的描述长度（最短“手册”），系统就更接近智能。

## 生成模型与压缩

对给定数据集 $D$，我们可以用生成模型 $f$ 对其进行压缩：

$$
\lvert D \rvert = -\log P_f(D) + \lvert f \rvert
$$

其中 $\lvert D \rvert$ 表示对数据集进行无损压缩后的大小，它等于对下一个 token 预测的损失总和，加上估计函数的最小描述长度（这里的 $\lvert f \rvert$ 指最小描述长度/编码开销的抽象项，并不等同于参数量）。此时，数据压缩的过程就是训练生成模型的过程。

进一步可以得到一个压缩率的表达式：

$$
r_n = 1 - \frac{S_1}{S_0}
> 1 - \frac{\lvert f_1 \rvert + n + \sum_{t=1}^{n} -\log P(x_{t+1} \mid x_{1:t}, f_1)}{\lvert f_0 \rvert + n \log m}
$$

这也解释了为什么模型越大往往表现出更强的泛化能力：模型更大，通常意味着 loss 更低，从而压缩率更高；在“压缩即智能”的框架下，这对应着更短的有效描述长度。

**Next Token Prediction** 虽然看似简单，但可以用压缩理论解释其合理性：这也是为什么 OpenAI 等团队长期坚持 Next Token Prediction 的原因之一。相对地，从最终应用效果上看，BERT 的“预测中间词”往往难以直接对齐到强生成能力。

## 局限与总结

对所有一切都进行压缩并不现实：例如像素级图像建模的开销就极大。现实中往往需要先确定想要保留和建模的信息片段，再找到方法过滤掉不需要的无关计算和信息片段，从而在无损压缩之前缩小正在处理的数据子集。

从“压缩智能”的角度看，当前单一模态模型的目标是继续提高有效信息压缩能力；而多模态模型的任务，则是寻找对复杂模态信息的压缩建模方法。BPE（Byte-Pair Encoding）可以处理文本的词表建模，基于统计频率的方法不仅高效，最终性能也相当可观，较大的缺点可能是对小众语言不友好；但音频与视频等模态仍需要更合适的离散化与表示。

现实中的许多数据也可能无法直接观测，不能简单指望通过压缩“所有可观测数据”就实现 AGI。更稳妥的理解是：压缩提供了一条解释基础模型泛化能力的路径，但它还需要和数据选择、模态表示、交互环境等问题一起讨论。
