---
layout: blog-post
title: "Agent 外接资源收藏册：Skills、MCP Server、插件与实用工具"
title_en: "An Agent Resource Collection: Skills, MCP Servers, Plugins, and Handy Tools"
date: 2026-05-17 21:00:00 +0800
categories: [Agent 基础设施]
tags: [Agents, MCP, Tool Use]
author: Hyacehila
excerpt: "一篇长期滚动更新的收藏册，记录可以给 Agent / Coding CLI 外接的各种资源：Skills、MCP Server、插件，以及顺手好用的小工具。"
excerpt_en: "A rolling collection of resources you can plug into agents and coding CLIs: Skills, MCP servers, plugins, and small tools that are handy enough to keep."
featured: false
math: false
---

# Agent 外接资源收藏册：Skills、MCP Server、插件与实用工具

这篇文章就是一个收藏夹，放一些我喜欢的、可以外接给 Agent / Coding CLI 的资源。

它们不一定很大，也不一定值得单独写一篇文章。可能是一套 Skills，一个 MCP Server，一个插件，或者某段脚本、某份提示词结构、某个看起来有点奇怪、但确实能省事的小办法。共同点是：都能挂到 Agent 或 Coding CLI 上，让它多一点能力。

列表是平铺的，不分区。每条会在开头标一下它是什么类型（Skill / MCP Server / 插件 / 工具），这样收藏变多了也能扫读。

先留在这里。后面遇到新的，再慢慢补。

## 收藏列表

### [General Research Review Agent And Skills](https://github.com/Hyacehila/General-Research-Review-Agent-And-Skills)

**类型：Skill**

这是我自己做的一套 research review skill suite，主要用来辅助综述写作。它不是那种“帮我写一篇综述”的单条 prompt，而是把 scoping review 拆成几个可以接力的步骤：找文献、去重、筛选候选池、抽取论文证据、整理大纲、综合成文，最后检查引用和语言。

我最喜欢的是它把综述从一段聊天，变成了一套有中间产物的流程。以前做这类工作，经常是 Zotero、Google Scholar、手动表格、零散 PDF、临时 prompt 混在一起，最后再让 LLM 帮忙润色。这样当然也能做，但回头查的时候会很痛苦：哪些文章被排除了，为什么排除，某个结论到底靠哪几篇文章支撑，引用有没有贴错，都不太好追。

这套 skills 处理的就是这些地方。candidate pool、selection ledger、evidence notes、outline、citation map，还有最后的 PDF/HTML/Markdown 报告，都会留下来。它不是为了把文章写得更漂亮，而是让综述里最容易散掉的部分有迹可查：检索从哪里来，筛选怎么做，证据写在哪，正文引用支撑了什么。相比临时 prompt 或单纯的文献管理工具，它更适合有范围、有审计要求的综述型研究；如果只是随手总结几篇论文，就没必要上这么重的流程。

### [Humanizer](https://github.com/blader/humanizer) / [Humanizer-zh](https://github.com/op7418/Humanizer-zh)

**类型：Skill**

Humanizer 和 Humanizer-zh 是一组写作清理 skills，用来改掉文本里的 AI 味道。它们不是语法纠错工具，盯的是更烦人的东西：过度总结、宣传腔、三段式排比、破折号滥用、空泛的“关键作用”，还有那种一看就像聊天机器人回答的客套话。

我喜欢它们，是因为这比“帮我润色一下”具体得多。普通润色经常会把文字改得更顺、更满，但也更像 AI；Humanizer 反过来，会删掉那些太工整、太会总结、太像模板的地方。英文内容可以用 Humanizer，中文内容就用 Humanizer-zh。

Humanizer-zh 对中文博客尤其有用。中文模型很容易写出“首先、其次、此外、综上所述”，也很容易把一句普通判断抬成“重要意义”“深刻影响”“复杂格局”。我很多时候不是想让文章更华丽，只是想让它听起来像真的有人写过、删过、犹豫过。写完技术文章、项目介绍、评论和随笔后，都可以拿它过一遍。

### [Beautiful Article](https://github.com/ConardLi/garden-skills/blob/main/skills/beautiful-article/SKILL.md)

**类型：Skill**

Beautiful Article 是一套把现成材料做成网页长文的 skill。网页、PDF、Word、Markdown、截图，甚至一大段粘贴材料，都可以先交给它处理。它会把这些东西整理成干净的源文，再做成一个可以离线打开、可以直接分享的单文件 HTML。它解决的不是生成一个网页，而是让 Agent 真的像编辑一样，先读材料，再重排结构，再把文章变得更好读，并给我们一个更加优雅的交互方式。

漂亮不是堆样式，对于一篇文章而言，更好读需要内容和视觉结构的配合。它会先抽取材料，再写编辑计划，然后停下来让用户确认文章类型、主题、版式、配图和封面。真正开写时，也不是一口气完成，而是先做首屏和第一节，让你看方向对不对。这个节奏很适合长材料：风格可以提前校准，信息也不容易在生成过程中悄悄丢掉。source、plan、review 这些中间文件还会留下来，回头检查时不会只剩一个成品 HTML。

它适合处理那些内容其实不错，但读起来太痛苦的材料，比如报告、教程、访谈、复盘、方案分析、解释文和很长的文章，也可以用于个人自己的分享。最后的成品可以有表格、代码块、图解、封面、目录，也可以加一点交互式小块，但它始终是在做文章，不是在做后台、表单、dashboard 或产品原型。如果手里有一份东西已经够扎实，只是太长、太散、太难读，我会很愿意把它交给 Beautiful Article 收拾一遍。
