---
title: "JoyAI-VL-Interaction：从 Chat 回到连续交互的视觉语言模型"
title_en: "JoyAI-VL-Interaction: From Chat Back to Continuous Interaction"
date: 2026-06-23 20:00:00 +0800
categories: [基础模型]
tags: [VLM, Multimodality, Interaction Model]
author: Hyacehila
excerpt: "Chat 只是我们给生成式模型套上的一种使用方式，不是模型天然的形状。Interaction Model 看起来很新，但也可以理解成把模型重新放回连续 token、连续感知和连续行动里。JoyAI-VL-Interaction 正好给了一个视觉语言方向的具体例子。"
excerpt_en: "Chat is one way we package generative models, not the natural shape of the model itself. Interaction models return models to continuous tokens, perception, and action, with JoyAI-VL-Interaction as a concrete vision-language example."
permalink: '/blog/2026/06/23/joyai-vl-interaction/'
---

# JoyAI-VL-Interaction：从 Chat 回到连续交互的视觉语言模型

最近看到 [JoyAI-VL-Interaction](https://joyai-vl-video-future-academy-jd.github.io/JoyAI-VL-Interaction/)，一开始只觉得它挺好玩：能看视频流，能判断什么时候说话，还能把复杂问题委派给后台模型或 agent。但再想一层，它其实碰到了一个更基础的问题：Chat 真的是生成式模型天然的交互形态吗？

## Chat 不是模型的天生形态

我们现在太习惯 Chat 了。用户说一句，模型回一句；用户再问，模型再答。对于一个长程的 Agent Runtime，Agent则会在收到命令后默默的执行，然后汇报结果。久而久之，语言模型等价于 Chatbot 。一轮输入，一轮输出，再用一枚 EOS token 结束回答，好像这就是模型本身的形状。

但这更像产品层和训练数据层形成的协议，而不是模型结构本身所决定的。自回归生成模型做的事情很朴素：给定前面的 token，预测下一个 token。EOS token 只是序列里的特殊符号，用来告诉解码器“这段可以停了”。如果场景不要求模型停在这里，而是让它持续接收音频、视频、文本和动作 token，这也是很自然的。

一轮轮 Chat 是把连续世界切成片段的界面设计。它很好用，但不是唯一合理的形态。真实交互里，人会听、看、打断、补充、沉默，也会一边观察一边决定下一句话该不该说。

从这个角度看，Interaction Model 并不是凭空出现的新物种。它更像是把生成式模型从 Chat 产品的约束放开，重新放回连续事件流里：输入不再只有一段用户消息，输出也不再只有一段助手回答，而是持续的感知、判断和行动。

## Interaction Model 到底新在哪里

[Thinking Machines Lab 的 Interaction Models](https://thinkingmachines.ai/blog/interaction-models/) 文章讲的就是这个问题。他们认为，今天很多 AI 系统的协作瓶颈不在于模型完全不会做事，而在于交互界面太窄。用户必须先把意图整理成完整输入；模型生成时又经常卡在自己的输出里，新的语音、画面、打断和反馈进不来。我们可以通过两轮 Tool Calling 之前的时间去 Steer 插入自己的想法，但这只是补丁。

TML 的说法是，interaction models 应该原生处理交互，而不是依赖外部脚手架补丁。模型要能同时处理音频、视频和文本，在实时协作中持续接收信息、回应和行动。他们提到的 multi-stream、micro-turn 设计，就是把大回合切成更小的时间片，让模型不必等完整一句话或完整一段回答结束，才重新感知世界。依旧多轮，但几百毫秒的轮次对人类就意味着连续，远大于普朗克时间。

这里有意思的不是低延迟本身，而是协作结构变了。用户可以打断，模型可以边听边想，画面变化可以触发模型改变计划，后台 agent 可以和前台交互模型分工。interaction model 在架构上没有什么惊天动力的变化，而在训练目标、数据组织、输入输出协议和系统形态。

## 回到生成式模型更早的样子

我更愿意把这个方向理解成生成式模型的回归，而不是对生成式模型的反叛。早期语言模型学习的是序列中的延续关系。后来 instruction tuning、RLHF、聊天模板和 tool call 协议，把模型变成了非常好用的对话助手。副作用是，我们把模型能力和聊天界面绑得太紧了。

一旦进入多模态实时场景，Chat 的边界就会露出来。摄像头画面不是固定 prompt，语音不是已经结束的文字，用户动作也不一定会以清晰命令出现。此时 EOS 反而不是关键，关键是模型能不能学会一组新的 action token 或行为标签：继续听、保持静默、发出提醒、调用工具、委派后台模型。

这样看，Interaction Model 仍然可以是生成式的。只是它生成的不一定都是自然语言，也可能是时机、动作、控制信号，或者给后台系统的任务描述。这也是一种 VLA，而实时交互的 VLA 正是目前研究具身智能的核心技术。

## JoyAI-VL-Interaction 做了什么

JoyAI-VL-Interaction 是一个不错的例子，也正是看到了他我才接触到 Interaction Model 并考虑写了这篇 Blog，它也一定程度上影响了我对 Agent 的想法。它是一个 8B 规模、视觉优先的交互模型。模型每秒都要在三个动作之间做判断：保持静默、直接回应，或者进行委派。这里的静默不是失败输出，而是一种被训练出来的行为。

这和普通视频理解模型不太一样。传统 VLM 更关心“视频里有什么”“请总结这段视频”。JoyAI-VL-Interaction 更像在回答另一个问题：“现在是否值得打断人类？”如果值得，是立刻提醒一句，还是把复杂问题交给后台的复杂长程任务模型。

它的行为来自超过 400 万条时间对齐交互样本，并通过强化学习进一步优化。这个数据形态很关键，因为交互问题天然带时间。一个提醒说得对，但晚了五秒，交互上可能已经失败。系统上，JoyAI-VL-Interaction 也不只是丢出一个模型权重。它开放了模型、训练配方、时间对齐数据和可部署系统，服务侧包括推理、WebUI、ASR、TTS、后台 agent，并兼容 vLLM 生态。

官方还在 58 个真实事件驱动视觉交互场景中做了人工成对比较，评估响应质量和响应时机。这比单纯问答准确率更贴近任务本身，因为交互模型的失败经常是“没有在该出现的时候出现”。总体来看，这是一个还算有趣且有价值的新尝试。

## 为什么这个方向值得看

我不想把 JoyAI-VL-Interaction 说成已经解决了实时助理。8B 模型在知识、复杂推理、长尾请求和个性化上肯定还有限制。TML 的 interaction model 也还处在研究预览阶段，长会话上下文、部署成本、安全边界和后台 agent 协作，都还要继续探索。

但这个方向很值得看。它不是发明了一个和生成式模型完全不同的新结构，而是把我们从 Chat 的惯性里拽出来。模型不一定只能等用户发话；生成也不一定只生成自然语言；EOS 也不必成为交互的边界。我们以前总是在研究 VLA、自动驾驶与机器人的时候讨论实时性，但人与 Agent 之间的交互，未尝不需要这种实时性，现有的 Chat 对齐和 Agent 设计也不一定就是正确的答案。

如果说过去几年基础模型最重要的界面是 Chat，那么下一阶段可能会出现更多 Online 的界面：摄像头里的生活助理、直播流里的实时评论员、机器人身上的观察者、桌面环境里的协作者到所有设计人机交互的地方。它们需要的不是更快地回答我看到了什么，而是判断现在该不该做点什么。

这也是 JoyAI-VL-Interaction 以及所有 Interaction Model 有趣的地方。它不只是一个会看视频的模型，而是一个把交互时机放进模型行为里的尝试。这个看似很新的概念又回到一个很朴素的问题：如果生成式模型本来就可以沿着连续序列往前走，那为什么一定要把它关在一问一答的 Chat 框里？

## 参考资料

- [JoyAI-VL-Interaction 项目页](https://joyai-vl-video-future-academy-jd.github.io/JoyAI-VL-Interaction/)
- [JoyAI-VL-Interaction GitHub 仓库](https://github.com/jd-opensource/JoyAI-VL-Interaction)
- [JoyAI-VL-Interaction 技术报告](https://arxiv.org/abs/2606.14777)
- [JoyAI-VL-Interaction 模型权重](https://huggingface.co/jdopensource/JoyAI-VL-Interaction-Preview)
- [JoyAI-VL-Interaction 数据集](https://huggingface.co/datasets/jdopensource/JoyAI-VL-Interaction)
- [Thinking Machines: Interaction Models](https://thinkingmachines.ai/blog/interaction-models/)
