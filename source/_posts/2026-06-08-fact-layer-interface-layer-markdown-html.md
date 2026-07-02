---
title: "事实层与界面层：Markdown 与 HTML 不是替代关系"
title_en: "The Fact Layer and the Interface Layer: Markdown and HTML Aren't Rivals"
date: 2026-06-08 20:00:00 +0800
categories: ["Work & Society", "AI Engineering Workflows"]
tags: ["Markdown", "HTML", "Documentation", "Interface Design", "Knowledge Management", "Software Engineering"]
author: Hyacehila
excerpt: "HTML 很诱人。它能把 AI 生成的长文档变成一个能读、能点、能分享的界面。但我不想让 HTML 慢慢堆成事实源。对我来说，Markdown 负责留下可 diff、可 grep、可追溯的记录；HTML 负责把复杂系统变得更容易读懂。"
excerpt_en: "HTML is tempting. It can turn long AI-generated documents into something readable, clickable, and shareable. But I do not want HTML to quietly become the source of truth. Markdown should keep the diffable, searchable, auditable record; HTML should make complex systems easier to read."
permalink: '/blog/2026/06/08/fact-layer-interface-layer-markdown-html/'
---

最近有个说法挺容易让人心动：AI 以后不该再输出 Markdown ，而应该直接给 HTML。

这话不是纯粹为了造梗。Claude Code 团队的 Thariq Shihipar 写了《Using Claude Code: The unreasonable effectiveness of HTML》，Simon Willison 又转述了一次，然后 Karpathy 又来吹了吹。那篇文章的例子很实在：代码审查、调研报告、图表、交互式编辑器、PR 说明，都可以做成一个能在浏览器里打开的 `.html` 文件。和一长串 Markdown 比，它确实更像一个可以上手用的工作台。

看起来挺让人心动的，应该有不少人都遇到过手机打不开`.md`的问题。

很多时候，我们不是缺内容，而是缺一个让人愿意读下去的界面。AI 可以吐出两千行分析、二十个风险点、十几个文件之间的调用关系，但人眼面对一整屏等宽字体时，很容易直接进入略读模式。换成有导航、有层级、有展开折叠、有图示的 HTML，情况会好很多。不是因为 HTML 更高级，而是因为人理解复杂系统时，确实需要一些空间和停顿。而HTML就是为了这种交互而设计的。

但HTML 应该取代 Markdown？

对我来说，HTML 适合作为一层交互界面。它让我更愿意读，也更容易看懂复杂系统。可它不适合慢慢积攒在仓库里成事实源。事实源要能被 diff 检查，要能被 grep 找到，要能在几个月后回答一句很具体的问题：这次到底改了什么。

<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px;margin:28px 0 34px;">
  <div style="border:1px solid #e5e7eb;border-radius:14px;padding:20px 18px;background:#fafafa;">
    <strong style="display:block;font-size:1.08em;margin-bottom:8px;">Markdown · 事实层</strong>
    <p style="margin:0;color:#555;font-size:0.95em;line-height:1.6;">留下可 diff、可 grep、可追溯的记录。计划、结论、任务状态和审查意见，最后都要回到这里。</p>
  </div>
  <div style="border:1px solid #e5e7eb;border-radius:14px;padding:20px 18px;background:#fafafa;">
    <strong style="display:block;font-size:1.08em;margin-bottom:8px;">HTML · 界面层</strong>
    <p style="margin:0;color:#555;font-size:0.95em;line-height:1.6;">把复杂关系摊开给人看。它可以生成、刷新、丢弃，也可以把人的选择写回事实层。</p>
  </div>
</div>

## HTML 赢在阅读和理解

复杂系统里最难读的往往不是结论，而是关系：哪个模块依赖哪个模块，状态在哪里流动，改动会碰到哪些边界，风险藏在哪条调用链后面。Markdown 可以描述这些东西，但描述到一定复杂度就开始吃力。HTML 可以把它们变成图、表、分栏、时间线、可折叠区域，甚至是一个可以拖动和过滤的小工具。

这不是装饰。关系本来就带有空间感。你把它总结成段落，读者要在脑子里重新建模；你把它展开成一个界面，读者可以先扫结构，再钻细节。Claude 那篇文章里列出来的示例集，真正有说服力的地方也在这里：很多输出不是「排版更漂亮的 Markdown」，而是把原本很难扫读的材料改成了一个可以操作的视图。

review 也是一样。一个模型给出上千行 markdown 方案，谁能一口气 review 这么多内容呢？ 人的注意力是自然会涣散掉的。HTML 可以让人先看到摘要，再按模块展开；先看风险，再点进证据；先看文件列表，再看每个文件的修改意图。阅读不再只剩一路往下滚，人也比较可能真的读完。

临时协作时，HTML 也顺手。一个自包含文件发出去，对方点开就能看。里面可以有勾选框、过滤器、复制按钮，也可以把某段内容整理成 prompt。对很多一次性分析、会议材料、PR 辅助审查来说，这比 Markdown 省事。

所以我并不反对 HTML。相反，AI 工具应该更擅长生成这种界面。需要的时候，把一份长报告变成一个浏览器里的工作区，把一个调用链变成图，把一堆任务变成可以筛选的列表。而且拿他来做 PPT 也挺好用的，简单快捷，开箱即用。

我的分歧只在下一步：这些界面不能自动变成事实本身。

## 问题出在把交互界面当事实源

HTML 一旦开始承担事实源，麻烦会来得很快。

先说 diff。需求里一句话从「必须支持离线」改成「优先支持离线」，在 Markdown 里通常就是一行差异。放进 HTML，可能混着标签、样式、布局、脚本和生成器的细微变化。review 的人打开 diff，看到的不是意图，而是一堆噪音。版本历史本来应该回答「谁在什么时候改了什么」，结果变成了「这次生成出来的界面和上次有多少不同」。

搜索和长期维护也会变麻烦。Markdown 的优势不是语法多强，而是它简单。你可以用 `rg` 找一段话，用 Git 看历史，用任何编辑器打开，用脚本做简单处理。HTML 当然也能搜索，但当真正的信息被包在结构和样式里面，文本的可操作性就会下降。今天觉得只是多几个 `<div>`，半年后可能就是一堆没人想碰的历史文件。

token 成本更像慢性病。单个 HTML 文件多出的标签和样式也许不算什么，但事实源不是一次性读物。spec、计划、审查记录、任务清单会被模型反复读、反复归档、反复带回上下文。每一次读 HTML，都在为界面付费。大上下文窗口会让这个问题看起来没那么急，但它不会让冗余消失。


## Markdown 的价值不在朴素，而在可审计

Markdown 的价值不是它看起来朴素，而是它把账留得清楚。内容就在文本里，没有额外结构，所有的层级工具都是信息的一部分；变化可以被人和工具低成本检查；你不需要先打开某个运行环境，才能知道文件到底写了什么。

John Gruber 当年对 Markdown 的定位其实很清楚：HTML 是 publishing format，Markdown 是 writing format。Markdown 不是要替代 HTML，而是让写作、阅读和编辑文本本身更容易。这个区分放到 AI 工作流里依然成立，只是 writing 不再只是写博客，也包括写 spec、写计划、写任务状态、写审查结论。

这些东西不应该只活在一个漂亮视图里。

如果一个 Agent 生成了 PR review dashboard，我希望它帮我看懂改动。没问题。可如果我勾掉了一个风险、改了一个判断、重新排序了任务，这些结果最后要回到 Markdown，或者回到另一个同样可审计的事实系统里。HTML 可以是操作台，但操作完要落账。不能让改动只留在 HTML 的交互层里。


## 不是替代，是分层

我理解 HTML-first 的吸引力。对很多人来说，AI 生成的内容终于可以不再是一份难读的报告，而是一个能打开、能点、能分享的小应用。这确实是进步。不同人的工作流也不一样，如果一个团队只需要短期交付、视觉审查和一次性汇报，HTML 作为主要产物完全合理。

但我的日常（或者说大部分系统开发者的日常）不是这样。我的问题不是「模型能不能生成漂亮输出」，而是「这些输出能不能被长期检查」。我需要知道一个判断什么时候变了，为什么变了，谁接受了它，后来有没有被推翻。HTML 能帮我理解这些东西，却不应该替我保存这些东西。

我会把边界划在这里。

需求、约束、计划、review 结论、任务状态、决策记录，这些需要长期存在、需要被版本控制、需要被模型和人反复检查的内容，应该留在 Markdown 里，或者留在同等可审计的结构化存储里。

阅读视图、关系图、PR 审查辅助、研究材料浏览器、临时 dashboard、交互式摘要，这些是给人看的界面。它们的目的不是留下最终记录，而是让人更快看懂、更愿意参与、更容易给出反馈。做成 HTML 很合适。

这条规矩也解释了为什么这篇文章本身可以在 Markdown 里嵌一段 HTML 卡片。卡片是界面层，负责让那组关系更容易看；整篇文章仍然是 Markdown，负责留下可以 diff 的文本。两者不是互相排斥，只是各自待在该待的位置。

最后回到那句话：

> Markdown 负责事实，HTML 负责理解。源文件不变，视图可丢，结论回流。

Skill 没有取代 MCP，CLI 不会取代 GUI，HTML 也不会取代 Markdown， AI Agent 改变很多东西，但炒作的强度比现实的变化更猛。

p.s. 为什么 Anthropic 的团队总是喜欢给我们搞一堆更烧 token 的方案，HTML 意味着我们需要为每一个 `<>` 付钱，多智能体和 Dynamic Workflow 也是 token 消耗大户，吹这玩意是为了赚我钱？

p.s. 从 `.md ` 到 `.html` 背后还有一个更宏伟的概念，等到脑机接口成熟以后，我们应该直接输出表征流（视频流），交互速度直接拉满，`.html` 也只是临时中间体。

## 参考资料

- [Using Claude Code: The Unreasonable Effectiveness of HTML（Thariq Shihipar，claude.com）](https://claude.com/blog/using-claude-code-the-unreasonable-effectiveness-of-html)
- [Simon Willison 转述：Using Claude Code: The Unreasonable Effectiveness of HTML](https://simonwillison.net/2026/May/8/unreasonable-effectiveness-of-html/)
- [The unreasonable effectiveness of HTML 示例集](https://thariqs.github.io/html-effectiveness/)
- [Daring Fireball：Markdown Syntax Documentation](https://daringfireball.net/projects/markdown/syntax)
- [《我们又回到了 CLI：Agent 时代 CLI 与 GUI 的选型》](/blog/2026/05/27/cli-vs-gui-agent-era/)
