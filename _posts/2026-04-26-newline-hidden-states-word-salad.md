---
layout: blog-post
title: '从 `\n\n` 看模型状态：Word Salad Chopper 带来的一个小启发'
title_en: "Reading Model States Through Newline Tokens: A Note from Word Salad Chopper"
date: 2026-04-26 16:00:00 +0800
categories: [基础模型]
tags: [Interpretability, Reasoning, Model Mechanics]
author: Hyacehila
excerpt: Word Salad Chopper 不只是在砍掉推理模型里的重复废话，也提示我们：换行边界 token 的 hidden state 可能是观察模型生成模式的低成本入口。
excerpt_en: "Word Salad Chopper does more than trim repetitive reasoning text. It suggests that hidden states at newline-boundary tokens may offer a cheap window into generation patterns."
featured: false
math: false
---

# 从 `\n\n` 看模型状态：Word Salad Chopper 带来的一个小启发

最近看到一篇 paper：[Word Salad Chopper: Reasoning Models Waste A Ton Of Decoding Budget On Useless Repetitions, Self-Knowingly](https://aclanthology.org/2025.emnlp-main.1705/)。

它表面上讲的是一件很工程的问题：reasoning model 在生成长推理链时，经常会浪费大量 token 在重复、空转、看似还在思考但其实没有新增信息的片段上。作者把这种现象叫作 `word salad`，也就是那种不断重复相似表达、消耗上下文和预算、但对最终答案没有什么帮助的推理废话。

如果只看到这里，这篇 paper 像是在做 token 成本优化：发现模型废话太多，于是设计一个组件把废话砍掉，再让模型重新生成。这个方向当然有用，尤其是在 reasoning model 越来越长、输出 token 越来越贵的情况下。

但我觉得它更值得讨论的地方不在砍掉废话本身，而在它选择的观测位置：每个 reasoning chunk 末尾的 `<\n\n>` token。

## 这篇 paper 做了什么

Word Salad Chopper 的核心思路很简单。作者先把模型的 reasoning trace 按公共分隔符切成一个个 chunk，然后观察每个 chunk 末尾的 `<\n\n>` token 的 hidden state。

直觉上，`\n\n` 好像只是一个排版符。它表示一段结束、下一段开始，最多只是文本格式的一部分。但论文发现，这个边界 token 的 hidden state 里包含了足够多的信号，可以用一个很轻量的线性分类器在线判断模型是不是已经进入了 `word salad` 状态。

一旦检测到模型在重复空转，系统就把这段没有语义价值的输出 chop 掉，然后用一个简单的 regeneration prompt 让模型接着往更有用的方向生成。这样做的目标不是改变模型本身，也不是重新训练一个更会推理的模型，而是在生成过程中加一个很薄的监控和修正层。

这引出了一个更具体的问题。

**在模型每次自然停顿、换段、切 chunk 的位置，hidden state 是否已经暴露了它当下所处的生成模式？**

Word Salad Chopper 给出的答案至少在 `word salad` 这个场景里是肯定的。

## `\n\n` 可能不只是换行

这让我觉得更值得讨论的，是 `\n\n` 这类边界 token 的角色。

在普通文本里，双换行只是段落分隔符。但在生成模型里，它可能同时承担了另一种功能：把前面一段生成过程压缩到一个边界状态里，并为下一段的展开做准备。

换句话说，模型生成到 `\n\n` 的时候，不只是打了一个排版上的句号。这个位置可能天然聚合了几类信息：

- 刚才这一段有没有真正推进推理。
- 下一段大概率是继续推导、换一个角度，还是开始重复。
- 当前生成模式是正常展开、空转、过早收敛，还是某种退化循环。
- 模型是否需要被截断、重试、降温、换策略，或者交给另一个监控器处理。

这不是说 `\n\n` 是什么神秘按钮，也不是说我们可以从一个 token 里读出模型完整的内心活动。更稳的说法是：在一些生成场景里，边界 token 的 hidden state 可能是一个非常便宜的状态摘要。它不是全部真相，但可能已经足够支持一些在线决策。

这个视角和我们平时看文本分段的方式很不一样。过去我们把 `\n\n` 当作外部结构，用它来切文章、切文档、切 RAG chunk。Word Salad Chopper 提示的是另一层东西：边界不只是在文本外面帮助我们切块，它也可能在模型内部形成一个可观测的状态节点。

## 更大的想象空间

如果这个想法能从 `word salad` 推广出去，它可能会变成一类实用的生成监控方法。

第一种用途是推理成本控制。现在很多 reasoning model 的问题不是不会答，而是为了答一个问题绕太远、重复太久。比起等完整答案出来以后再判断质量，边界 token probe 可以在生成中途发现“这一段已经没什么信息增量了”，然后及时截断、重生或切换策略。

第二种用途是 agent 执行监控。Agent 在长任务里也会出现类似空转：反复解释同一个计划、迟迟不调用工具、调用工具后没有整合结果、在错误路径上继续展开。如果工具调用前后、步骤结束处、日志分隔符附近的 hidden state 也有类似信号，那么 monitor 就不一定只能看最终文本，也可以看模型在关键边界处的状态变化。

第三种用途是长文本生成和 RAG。很多长文质量问题不是一句话坏掉，而是段落之间开始漂移、重复、断裂。边界 token 本来就是段落结构的锚点，如果它能反映下一段的生成模式，就有机会做段落级质量检查：这一段要不要继续写、要不要回到证据、要不要重新检索、要不要换一个大纲节点。

第四种用途是轻量 routing。我们通常把 routing 放在请求入口，比如判断这个问题该交给哪个模型。但生成过程本身也可以有 routing：当模型进入正常推导，就继续；当它进入重复，就截断；当它进入不确定状态，就调用工具；当它开始偏离证据，就回到检索。边界 token 的 hidden state 也许可以成为这种动态 routing 的信号之一。

## 我喜欢它的原因

这篇 paper 提供了一个朴素但有启发性的观察方式：

**不要只看模型说了什么，也要看它在结构边界处变成了什么状态。**

在今天的 LLM 系统里，我们已经很习惯用外部组件包住模型：router、verifier、judge、retriever、memory、tool executor。但这些组件大多还是围绕可见文本工作。Word Salad Chopper 这类方法则提醒我们，模型生成过程中的某些自然边界，可能本身就是很好的内部观测点。

这条路线的未来不一定是“读懂模型”。更现实的价值可能是做一批便宜、局部、可插拔的 runtime monitor：不改变模型参数，只在生成过程中读几个关键 hidden state，然后决定要不要继续、砍掉、重试、路由或提醒。

`\n\n` 背后的问题可以继续追下去：模型每次停顿和换段时，是否已经在 hidden state 里留下了它下一步会怎样生成的信号？

## 参考

- [Word Salad Chopper: Reasoning Models Waste A Ton Of Decoding Budget On Useless Repetitions, Self-Knowingly](https://aclanthology.org/2025.emnlp-main.1705/)（Xie et al., EMNLP 2025）
- [Extracting Paragraphs from LLM Token Activations](https://arxiv.org/abs/2409.06328)（arXiv 2024）
- [Future Lens: Anticipating Subsequent Tokens from a Single Hidden State](https://arxiv.org/abs/2311.04897)（arXiv 2023）
