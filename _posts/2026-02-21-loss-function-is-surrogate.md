---
layout: blog-post
title: "Loss Function is Surrogate"
date: 2026-02-21 20:00:00 +0800
series: "LLM ESSENCE"
categories: [LLM]
tags: [SFT, RL]
author: Hyacehila
excerpt: "整理田渊栋关于“loss 只是 surrogate”的观点：训练的关键不在于把某个数值压到最小，而在于 loss 产生的梯度流如何塑造表征；从这个角度重新审视 SFT 与 RL 的差异。"
---

> 节选自 YuanDong Tian 的一次采访内容。

# Loss Function is Surrogate

损失函数不是所谓的真理（如果真的存在真理，就不会有这么多损失函数都可以 work 了）。田渊栋的一句话我很喜欢：很多 loss function 都是 surrogate（代理），比如 predict the next token……它的目的是产生一个梯度流，让表征往“正确的方向”走，这是最重要的逻辑。

至于这个目标函数“长什么样”，其实并不重要。

我们不在追求最小化损失函数（也就是所谓的收敛），因为优化损失只是寻找正确表示的手段，而不是目的。与其盯着 loss，不如从“我们希望表征学习到的数据结构”出发，去设计符合需求的训练信号。

一个很小的例子是 0-1 评测：如果“答对 = 1 分、I don't know（IDK）= 0 分”，模型就会被激励去猜。很多你以为的“幻觉/投机”，本质上是 surrogate 的激励结构。

从这个角度看，SFT 和 RL 不存在本质的训练质量区别：它们都是在调整超高维参数空间，只是 surrogate 的方向不同，让它们走向了不同的道路。

因此，与其问 loss 有没有收敛，不如问这个 surrogate 是否奖励了你真正想要的行为（以及是否系统性惩罚了“不确定性表达”）。把 loss 当成 surrogate，会更自然地把注意力放回数据、约束与评测。
