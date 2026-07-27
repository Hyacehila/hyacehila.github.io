---
title: "JoyAI-VL-Interaction：从 Chat 回到连续交互的视觉语言模型"
title_en: "JoyAI-VL-Interaction: From Chat Back to Continuous Interaction"
date: 2026-06-23 20:00:00 +0800
categories: ["Foundation Models", "Model Mechanics"]
tags: ["Vision-Language Models", "Multimodality"]
author: Hyacehila
excerpt: "Chat 只是我们给生成式模型套上的一种使用方式，不是模型天然的形状。Interaction Model 看起来很新，但也可以理解成把模型重新放回连续 token、连续感知和连续行动里。JoyAI-VL-Interaction 正好给了一个视觉语言方向的具体例子。"
excerpt_en: "Chat is one way we package generative models, not the natural shape of the model itself. Interaction models return models to continuous tokens, perception, and action, with JoyAI-VL-Interaction as a concrete vision-language example."
permalink: '/blog/2026/06/23/joyai-vl-interaction/'
---

最近看到 [JoyAI-VL-Interaction](https://joyai-vl-video-future-academy-jd.github.io/JoyAI-VL-Interaction/)，第一眼只觉得它挺好玩：能一直看视频流，能判断什么时候该说话，还能把一时处理不了的复杂问题委派给后台模型或 agent。看上去像一个很完整的实时助理。后来再看，我觉得它最有意思的地方不是“能做这么多”，而是把一个我已经习惯到不太会怀疑的问题重新摆到了面前：Chat 真的是生成式模型天生就该待着的地方吗？

## Chat 不是模型的天生形态

我们现在对 Chat 太熟了。用户说一句，模型回一句；用户再问，模型再答。运行时间更长的 Agent 也差不多：收到命令后默默把事情做完，再回来汇报结果。久而久之，语言模型好像就等于 Chatbot。一轮输入，一轮输出，末尾放一枚 EOS token 结束回答，好像这就是模型本身的形状。

但这更像产品和训练数据共同养成的规矩，不是模型结构本身逼着我们这么做。自回归生成模型做的事情很朴素：给定前面的 token，预测下一个 token。EOS token 只是序列里的一个特殊符号，用来告诉解码器“这段可以停了”。如果场景不需要模型停在这里，而是让它持续接收音频、视频、文本和动作 token，这同样很自然。

一轮轮 Chat 是一种把连续世界切成片段的界面设计。它确实好用，但不是唯一合理的形态。真实交互里，人会听、看、打断、补充、沉默，也会一边观察一边决定下一句话该不该说。

所以我不太把 Interaction Model 当成凭空出现的新物种。它更像是把生成式模型从 Chat 产品的约束里放出来，重新放回连续事件流：输入不再只有一段用户消息，输出也不再只有一段助手回答，而是持续的感知、判断和行动。

## Interaction Model 到底新在哪里

[Thinking Machines Lab 的 Interaction Models](https://thinkingmachines.ai/blog/interaction-models/) 讲的也是这个问题。他们认为，今天很多 AI 系统卡住的地方不在于模型完全不会做事，而在于交互界面太窄。用户得先把意图整理成完整输入；模型生成时又常常困在自己的输出里，新的语音、画面、打断和反馈进不来。我们当然可以在一次 tool call 结束、下一次开始之前塞进新的状态，或者想办法在模型输出的间隙里 steer 一下，但这些总有点像在现成的 Chat 机制上打补丁。

TML 的说法是，interaction models 应该原生处理交互，不要把关键能力都交给外部脚手架。模型要能同时处理音频、视频和文本，在实时协作中持续接收信息、回应和行动。他们提到的 multi-stream、micro-turn 设计，就是把大回合切成更小的时间片，让模型不用等完整一句话或完整一段回答结束，才重新感知世界。它依旧是多轮，只是时间粒度降到了几百毫秒；对人来说，这已经足够接近连续。

这里真正变掉的不是延迟数字，而是协作结构。用户可以打断，模型可以边听边想，画面变化可以让模型改计划，后台 agent 也可以和前台交互模型分工。Interaction Model 未必在架构上有什么惊天动地的变化，它更多改变的是训练目标、数据组织、输入输出协议和系统形态。

## 回到生成式模型更早的样子

我更愿意把这个方向理解成生成式模型的一次回归，不是什么对生成式模型的反叛。早期语言模型学习的是序列里的延续关系。后来 instruction tuning、RLHF、聊天模板和 tool call 协议，把模型变成了非常好用的对话助手。副作用是，我们也把模型能力和聊天界面绑得太紧了。

一旦进入多模态实时场景，Chat 的边界就露出来了。摄像头画面不是固定 prompt，语音不是已经结束的文字，用户动作也不一定会以清晰命令出现。此时 EOS 反而不是关键，关键是模型能不能学会一组新的 action token 或行为标签：继续听、保持静默、发出提醒、调用工具、委派后台模型。

这样看，Interaction Model 当然仍然是生成式的。只是它生成的不一定都是自然语言，也可能是时机、动作、控制信号，或者给后台系统的任务描述。把输出扩大到这些东西以后，它依旧是一种 VLA 问题，只是动作不一定发生在机械臂上，也可能发生在一个实时交互系统里。

## JoyAI-VL-Interaction 做了什么

JoyAI-VL-Interaction 是一个很直观的例子。也是看到它以后，我才开始认真想 Interaction Model，并动了写这篇文章的念头；它多少也改了我之前看 Agent 的方式。它是一个 8B 规模、视觉优先的交互模型。模型每秒都要在三个动作之间做判断：保持静默、直接回应，或者进行委派。这里的静默不是失败输出，而是一种被训练出来的行为。

这和普通视频理解模型不太一样。传统 VLM 更关心“视频里有什么”“请总结这段视频”。JoyAI-VL-Interaction 更像在反复回答另一个问题：“现在值得打断人吗？”如果值得，是立刻提醒一句，还是把复杂问题交给后台的长程任务模型。

它的行为来自超过 400 万条时间对齐交互样本，并通过强化学习进一步优化。这个数据形态很关键，因为交互问题天然带时间：一句提醒说得对，但晚了五秒，交互上可能已经失败。系统层面，JoyAI-VL-Interaction 也不只是丢出一个模型权重。它开放了模型、训练配方、时间对齐数据和可部署系统；服务侧包括推理、WebUI、ASR、TTS 和后台 agent，并兼容 vLLM 生态。

官方还在 58 个真实事件驱动的视觉交互场景中做了人工成对比较，分别看响应质量和响应时机。这个评估方式很对题，因为交互模型常见的失败不是答错，而是该出现的时候没有出现。整体看下来，这还是一个挺有意思的新尝试。

## GPT-Live-1：语音侧的 Interaction Model

如果说 JoyAI-VL-Interaction 是把 Interaction Model 放进视觉语言流里，那么 GPT-Live-1 更像是同一个问题在语音侧的版本。它不是本文逻辑的反例，反而补上了另一块：实时交互不只发生在摄像头和视频里，也发生在最日常的人类对话里。视觉侧更关心“我看到了什么，现在要不要介入”；语音侧更关心“我听到了什么，现在要不要接话”。

传统语音助手往往是级联系统：先由 ASR 把语音转成文字，再交给语言模型生成答案，最后用 TTS 把答案读出来。这条链路当然能工作，但它天然把对话切成了几个离散阶段。OpenAI 把 GPT-Live 描述为 full-duplex speech-to-speech model，它可以同时听和说，更接近人类对话里那种边听边调整、边说边感知对方反应的状态。重点不只是“支持语音输入输出”，而是交互协议开始从 turn-based chat 往连续语音流靠近。

语音里的 Interaction Model，核心行为也不一定是回答问题。它可能在用户停顿时给一个 backchannel，可能在对方还没说完时保持沉默，也可能在误解快要发生时轻轻打断，或者把更复杂的任务委派给后台系统。模型生成的不只是句子，也包括轮替、沉默、确认、打断和继续听这些对话动作。它们和 JoyAI-VL-Interaction 里的“静默、直接回应、委派”其实很像，只是模态从视觉语言换成了双向语音。

这也是 GPT-Live-1 很适合作为 Interaction Model 参照的原因。JoyAI-VL-Interaction 判断的是“现在该不该看见并行动”，GPT-Live-1 判断的是“现在该不该说、听、等待或委派”。二者的共同点不是多模态本身，而是模型开始把交互时机纳入自己的行为，不再完全交给外部产品脚本处理。

我现在更愿意把 Interaction Model 当成一组正在出现的问题，而不是某一种单一架构名。VL interaction、voice interaction、robot interaction、desktop interaction 都可能长得不一样，但它们都在尝试把模型从离散的 Chat 框里放回连续事件流。JoyAI-VL-Interaction 给了一个视觉语言侧的例子，GPT-Live-1 则提醒我，语音本身也足够构成一个重要的实时交互世界。

## 为什么这个方向值得看

我不想把 JoyAI-VL-Interaction 说成已经解决了实时助理。8B 模型在知识、复杂推理、长尾请求和个性化上肯定还有限制。TML 的 Interaction Model 也还处在研究预览阶段，长会话上下文、部署成本、安全边界和后台 agent 协作，都还有不少问题要解决。

但我还是觉得这个方向值得看。它没有发明一个和生成式模型完全不同的新结构，只是把我们从 Chat 的惯性里拽出来。视觉实时交互和语音实时交互都在说明一件事：模型不一定只能等用户发话，生成也不一定只生成自然语言，EOS 更不一定要成为交互的边界。我们以前总是在研究 VLA、自动驾驶和机器人时讨论实时性，但人与 Agent 的交互未必不需要这种实时性。现有的 Chat 对齐和 Agent 设计，也未必就是最后的答案。

如果 Chat 是过去几年基础模型最成功的界面，接下来也许会有更多模型直接待在持续变化的环境里：摄像头、直播流、机器人、桌面，以及任何需要人机协作的地方。它们面对的不是“我看到了什么”的问答，而是“现在到底要不要做点什么”。

这也是 JoyAI-VL-Interaction、GPT-Live-1 和其他 Interaction Model 有趣的地方。JoyAI-VL-Interaction 是视觉侧的例子，GPT-Live-1 是语音侧的例子；它们都不只是给模型多接了一个输入输出通道，而是在尝试把交互时机放进模型行为里。这个看起来很新的概念，最后又回到了一个很朴素的问题：如果生成式模型本来就可以沿着连续序列往前走，我们为什么一定要把它关在一问一答的 Chat 框里？

## 参考资料

- [JoyAI-VL-Interaction 项目页](https://joyai-vl-video-future-academy-jd.github.io/JoyAI-VL-Interaction/)
- [JoyAI-VL-Interaction GitHub 仓库](https://github.com/jd-opensource/JoyAI-VL-Interaction)
- [JoyAI-VL-Interaction 技术报告](https://arxiv.org/abs/2606.14777)
- [JoyAI-VL-Interaction 模型权重](https://huggingface.co/jdopensource/JoyAI-VL-Interaction-Preview)
- [JoyAI-VL-Interaction 数据集](https://huggingface.co/datasets/jdopensource/JoyAI-VL-Interaction)
- [Thinking Machines: Interaction Models](https://thinkingmachines.ai/blog/interaction-models/)
- [OpenAI: Introducing GPT-Live](https://openai.com/index/introducing-gpt-live/)
- [OpenAI Help: ChatGPT Voice](https://help.openai.com/en/articles/20001274-chatgpt-voice)
- [OpenAI: GPT-Live-1 in the API](https://openai.com/form/gpt-live-1-in-the-api/)
