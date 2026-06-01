---
layout: blog-post
title: "奇怪 Skills 收藏册：把小能力变成可复用的工具箱"
title_en: "A Collection of Weird Skills: Turning Small Capabilities into a Reusable Toolbox"
date: 2026-05-17 21:00:00 +0800
categories: [Agent 基础设施]
tags: [Agents, Skills, Workflow]
author: Hyacehila
excerpt: "这是一篇长期滚动更新的 skills 收藏册，用来记录那些不一定宏大、但足够有趣、具体、可复用的小能力。"
excerpt_en: "A rolling collection of unusual skills: small, concrete, reusable capabilities that are interesting enough to keep."
featured: false
math: false
---

# 奇怪 Skills 收藏册：把小能力变成可复用的工具箱

这篇文章就是一个收藏夹，用来放一些我喜欢的小工具和 skills。

它们不一定很大，也不一定值得单独写一篇文章。可能是一组脚本，一个工作流，一份提示词结构，或者某个看起来有点奇怪、但真的能省事的小办法。

先留在这里。后面遇到新的，再慢慢补。

## 收藏列表

### [General Research Review Agent And Skills](https://github.com/Hyacehila/General-Research-Review-Agent-And-Skills)

这是我自己做的一套 research review skill suite，主要用来辅助综述写作。它不是那种“帮我写一篇综述”的单条 prompt，而是把 scoping review 拆成几个可以接力的步骤：找文献、去重、筛选候选池、抽取论文证据、整理大纲、综合成文，最后再检查引用和语言。

我最喜欢的是它把综述从一段聊天，变成了一套有中间产物的流程。以前做这类工作，经常是 Zotero、Google Scholar、手动表格、零散 PDF、临时 prompt 混在一起，最后再让 LLM 帮忙润色。这样当然也能做，但回头查的时候会很痛苦：哪些文章被排除了，为什么排除，某个结论到底靠哪几篇文章支撑，引用有没有贴错，都不太好追。

这套 skills 处理的正是这些地方。candidate pool、selection ledger、evidence notes、outline、citation map，还有最后的 PDF/HTML/Markdown 报告，都会作为明确文件留下来。它的优势不在于“更会写”，而在于让综述里最容易散掉的部分有迹可查：检索从哪里来，筛选怎么做，证据写在哪，正文引用支撑了什么。相比临时 prompt 或单纯的文献管理工具，它更适合有范围、有审计要求的综述型研究；如果只是随手总结几篇论文，就没必要上这么重的流程。
