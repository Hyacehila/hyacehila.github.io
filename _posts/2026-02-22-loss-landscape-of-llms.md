---
layout: blog-post
title: "大模型的Loss Landscape是什么样的？"
date: 2026-02-22 20:00:00 +0800
series: "LLM ESSENCE"
categories: [LLM]
tags: [Pre-Training, SFT]
author: Hyacehila
math: true
excerpt: "从 Loss Landscape 视角理解大模型的预训练与微调：pre-training basin 如何保护泛化能力；为何微调会导致能力退化/遗忘；以及为什么少量对抗数据能在最差方向上快速摧毁模型能力。"
---

# 大模型的Loss Landscape是什么样的？

> 本文核心观点来自论文《Understanding Pre-training and Fine-tuning from Loss Landscape Perspective》，内容节选自知乎《大模型的Loss Landscape是什么样的？》。

## Intro

一个好端端的模型，为什么在一个安全数据集上 fine-tune 完之后，它的数学能力、推理能力退化了？或者在数学数据集上 fine-tune 完之后，它的安全能力直接没了？甚至这些数据集只需要几百条就可以完全影响模型的能力。一个不进行 fine-tune 的模型，只需要一些输入上的调整，就可以攻击模型的能力或者安全。以上的问题或许可以使用 Loss Landscape 来进行解释。

## Loss Landscape

Loss landscape 是对神经网络这个高维函数的一个可视化：当参数变化时，模型的 loss 如何变化。

可是模型的参数（如 7B）很大，想要完全画出来不现实。因此我们往往会选择 2 个或者 1 个随机方向来画 loss landscape，即：

$$
L(\alpha) = J_{\mathcal{D}}(\boldsymbol{\theta} + \alpha \boldsymbol{\delta})
$$

其中 $\boldsymbol{\delta}$ 是一个方向向量，$\alpha$ 是沿该方向的标量步长。

为什么这样做是合理的？因为对于深度学习中大多数模型和大多数任务，**大多数方向的 loss 变化几乎没有任何差别，因此随机可视化一个方向，就已经代表了大多数方向上 loss 的变化**。（这是深度学习研究 loss 的经典理论，有相关理论分析、实验证明以及直觉上的解释。）

小模型 loss landscape 往往变化多样，可能同时具备平坦、尖锐与光滑的特征。这类 loss landscape 也在曾经的深度学习研究中指导过相当多的工作：去设计更好的优化方法来得到一个平坦的 loss，以增强模型的泛化能力。

非常不一样的是，大模型的 loss landscape 和小模型截然不同。大模型的 loss landscape 就像一个 basin（盆地）一样：在盆地内部模型的效果基本没有任何变化（这岂不是意味着在盆地内部怎么移动都不怎么影响性能？），出了盆地模型的能力就完全消失，直接输出乱码。

pre-training basin 会给模型一些基础的语言对话能力，而后续的 alignment 都是在 pre-training basin 内部**创造一个又一个的小 basin**，如 math basin、coding basin、safety basin 等。

alignment 做得非常充分时，模型的 alignment basin 很可能和 basic basin 一样大；但如果对齐不足，那么沿着大多数方向走（fine-tune 也好，attack 也好），轻易就走到了数学/coding/safety 能力全没了、但还能正常对话的模型上。

从上面的分析来看，basin 大是一件好事。**因为如果 basin 大，这就意味着沿着大多数方向走，模型的能力在这个范围内不会有任何下降。**而这个“大多数方向”可以通过 Clopper-Pearson Bound（用于给出比例的置信下界）给出下界，例如 99.9% 的方向均是如此。

只要 fine-tune 是在这 99.9% 的方向里，只要你在 basin 内 fine-tune，那么你就一定不会 compromise 任何性能。当且仅当你 fine-tune 的距离太远了，以至于出了 basin，才会 compromise 性能。

我们也可以轻易发现：越大的模型确实 basin 就越大，就越不容易在后续的 fine-tune 中 compromise 之前所得到的性能。

## LLM Worst-Case Loss Landscape

这虽然能解释为什么 fine-tuning 会遗忘，可是如何解释使用仅仅 10 条对抗数据去 fine-tune，总 token 数都不到 1000，模型就直接把安全能力全忘了呢？难道 1000 token 的训练量已经足够让模型走到 basin 外面了吗？

这其实很好解释：因为如果使用对抗的数据，模型 SFT 的走向根本不是这 99.9% 的绝大多数方向（典型方向），而是最差方向（worst-case direction）。它们的 loss landscape 非常尖锐，只要略微偏移，立刻就会让模型的损失到达接近最大的程度，所有学习到的知识全部被遗忘。

---

刚刚我们已经解释了为什么正常的 fine-tuning 会导致遗忘（因为 tune 出 basin 了），以及为什么对抗数据可以迅速遗忘（因为走的最差方向）。那么为什么不是 tune 参数而是优化输入，也能导致模型输出任何你想要的输出呢？

这其实很好理解：对抗样本这种“优化输入”和 fine-tuning 这种“优化参数”（包括第一层参数）其实没有本质区别。只要我们能够通过 fine-tune 的方式修改第一层参数的输出，就一定可以通过提示的方法实现一样的效果：两种扰动在第一层的 activation space 产生相同的向量。

这就解释了在 Intro 中提出的全部问题。
