---
title: "Automatic Testing 与 AI-Native Game Engine"
title_en: "Automatic Testing and AI-Native Game Engines"
date: 2026-09-21 20:00:00 +0800
categories: ["Creative Media & Games", "Game AI & Production"]
tags: ["Automated Testing", "Game Engine", "Game AI", "AI Agent"]
author: Hyacehila
excerpt: "Agentic Testing System，或者说，一套让 AI 真正进入游戏、体验游戏并形成判断的接口，或许会成为下一步 AI4Game 的核心，也可能决定 AI-Native Game Engine 应该是什么样子。"
excerpt_en: "An agentic testing system—or an interface that lets AI genuinely experience a game and draw conclusions from it—may be central to the next stage of AI4Game, and may even determine the shape of an AI-native game engine."
mathjax: false
hidden: false
permalink: '/blog/2026/09/21/automated-testing-ai-native-game-engine/'
---

Agentic Testing System，或者说，一套让 AI 真正进入游戏、体验游戏并形成判断的接口，或许会成为下一步 AI4Game 的核心，也可能决定 AI-Native Game Engine 应该是什么样子。

## 为什么聊到测试？
为什么我会突然把“测试”和“AI 原生游戏引擎”放在一起讨论？

在真正展开这个话题之前，我想先从这篇文章的起点说起。

### FeedBack is all you need
当 AI Coding 正快速渗透前端、后端、客户端和服务端开发，“一句话生成一个 App Demo”已经逐渐成为现实。但当同样的期待来到游戏开发，事情却没有那么顺利。今天的 Coding Agent 可以很快写出角色控制、战斗逻辑，甚至搭起一个完整的游戏工程，可真正让它独立完成一款游戏时，结果往往仍然停留在“能够运行”的 Demo——至于它能不能玩、画面是否正确、流程能不能走通，甚至到底好不好玩，模型自己常常并不知道。

在网易实习的三个月里，我逐渐产生了一个很模糊但越来越强烈的感觉：AI Game Coding 当前缺少的，也许并不是生成能力，而是反馈。

Coding Agent 过去几年的发展已经证明，只要允许模型真正运行自己的代码，读取报错、执行测试、观察结果，再根据反馈不断修改，同一个模型的能力就会发生巨大的变化。SWE-agent 将这种面向模型的交互方式称为 Agent-Computer Interface；而在游戏开发中，这样的闭环却远没有成熟。游戏的重要状态分散在场景、资产、动画、物理、UI 和实时运行过程中，很多错误不会变成一条清晰的 AssertionError，而只会表现为“角色没有走到正确的位置”“镜头穿模了”或者“这一关实际上根本无法通关”。

如果模型只能生成，那么它仍然只是一个生成器；只有当它能够完成“生成—运行—观察—验证—修改”的循环时，它才真正开始成为一个 Game Coding Agent。

这也是我为什么开始对游戏测试产生兴趣。这里所谓的“测试”并不只是传统 QA 意义上的测试用例，而是一整套面向 Agent 的反馈基础设施：让模型能够观察游戏状态、操作运行时、自动完成玩家流程、发现异常，并把这些结果转化成下一轮修改可以理解的反馈。

我的直觉是，这可能会成为 AI 原生游戏开发中非常基础的一层能力。**让Coding Agent 拥有理解游戏的眼睛。**
### Game Benchmark 与现状
GameDevBench 发表于 2026 年的 ICML，测试 Coding Agent 能否在现代游戏引擎（Godot）中完成游戏开发任务。有点像游戏世界的 SWEBench ，任务不复杂但基本覆盖了常见开发场景。

遗憾的是，已经将 SWEbench 以及各种变体刷爆的顶尖智能在游戏领域有一点失灵。Gameplay 类任务成功率约 51.4%，而强调视觉理解的 2D Graphics 类任务只有约 33.0%。**纯 Gameplay Logic 不是最难的。** 代码 Agent 很擅长操作符号世界，游戏大量真实状态存在于视觉世界和运行时世界。

研究者额外给 Agent 两种能力以后，分别是编辑器 MCP 以及 Runtime Video，GPT-5.4 在该测试中的成功率从：41.1% → 52.0%。这已经初步的证明了生成式人工智能在游戏开发领域的不足，**现有 Coding Agent 的感知—行动—验证闭环，并不是为游戏编辑器设计的。**

基于浏览器，也就是 Web 界面的游戏开发与成熟商业引擎之前的差距进一步显现了这个问题。`Three.js` 全是 `canvas` 但 Agent 依旧可以通过它最擅长的 JS 去访问内容；一个 Unreal 项目内部远不止一些代码脚本，海量的状态只存在于编辑器的二进制文件以及运行时的内容变化。GameEngineBench 是针对 UE 的一个 Benchmark ，模型进步，性能倒吸，这就是 Game Dev Agent 的现状。**当 Code 不再是整个世界，我们需要重新表征 World**

### 变得 Model Progress Resistant
GPT6-Astra 的发布让不少人开发者有点害怕，它能够使用 SVG 模拟人类笔触绘画，操作 Blender 工具生成高质量的建模，在场景搭建工具中根据效果图还原位置。一代新的基础模型的发布，吞噬掉了很多生成中间层的位置，也包括我之前做的关于 UI 工作流的工作。

这一轮模型能力跃迁也让我产生了一些选择去做什么的警惕：如果一个方向的主要价值只是弥补今天基础模型在生成能力上的 5%～10% 缺口，那么下一代模型升级就可能迅速压缩它的价值空间。

去找一个伴随模型性能提升越来越好的方向，一个 Model Progress Resistant 的方向，对保住我的饭碗也有不小的价值。

模型的每一轮进步对于生成能力的中间层都是考验，但随着模型能做的越来越多，需要验证的状态更多，Verification infrastructure 仍然存在。模型越强，测试不一定越不重要；它可能反而让更大规模、更高自主性的工程成为可能，从而扩大自动验证的价值。
>“LLMs made generating code cheap. The real bottleneck is verification.” - OpenHands

## Web App 与 Game 的现状

### 从 Web 开始聊聊 Automated 的基础
