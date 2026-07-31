---
title: "当我们构建 AI Agent 时，我们究竟在解决什么问题？"
title_en: "What Problems Are We Really Solving When We Build AI Agents?"
date: 2026-07-28 20:00:00 +0800
categories: ["Agent Systems", "Agent Architecture"]
tags: ["AI Agent", "Agent Architecture", "AI Engineering"]
author: Hyacehila
excerpt: "从几段 AI Agent 开发经历出发，回看过去两年里 AI Agent 的定义，以及开发一个 AI Agent 这件事本身发生的变化。人究竟把什么、以什么形式交给 AI？又该如何约束一个自由的概率模型？本文从第一性原理讨论委派、边界与判断,当我们构建 AI Agent 时，究竟在做什么，又真正解决了什么问题。"
excerpt_en: "Starting from several experiences building AI agents, this essay looks back at how both the definition of an AI agent and the work of building one have changed over the past two years. What are people handing over to AI, in what form, and how should we constrain an inherently open-ended probabilistic model? From first principles, it examines delegation, boundaries, and judgment—what we are actually doing when we build AI agents, and the problems we are truly trying to solve."
mathjax: false
permalink: '/blog/2026/07/28/what-problems-are-we-really-solving-when-building-ai-agents/'
hidden: false
---

## 写在前面

这会是一篇很长的 Blog。我会按一个连我自己都还没想清楚的顺序，聊聊这个标题。文中会插入不少外部资料和旧 Blog 的链接：有的是例子，有的是观点的延伸阅读。它们不负责替我下结论，只给读者多一些可以自行判断的材料。

这篇文章更面向技术出身的读者。我会直接使用不少 CS 和 AI 的概念与名词；如果你不熟悉这些东西，也可以让 AI 陪你一起读。它不会是一篇只抛观点的文章，里面有不少例子和论证。准备读下去之前，最好留一点时间。

这确实是一篇很难开始的文章。我有很多想写的内容，也很确定该写一篇这样的文章，但还没找到真正的切口和行文脉络。太多事都想塞进来，我自己也没想清楚哪些该留、哪些该放在别处，更不知道读者最后应该带走什么。那就先随意一点。想到哪写到哪，慢慢把十几张散乱的稿纸收成一篇完整文章。

让我们从故事开始。
