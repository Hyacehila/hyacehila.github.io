---
title: "Google 提示工程实践指南：角色、任务、上下文与格式"
title_en: "Google Prompt Engineering Guide: Roles, Tasks, Context, and Format"
date: 2025-03-01 20:00:00 +0800
categories: ["Work & Society", "AI Engineering Workflows"]
tags: ["Learning Notes","Prompt Engineering","Google","Gemini","Practical Guide"]
author: Hyacehila
excerpt: "整理 Google 官方提示词编写指南的核心框架（角色/任务/上下文/格式）与多场景实践示例，附带 Gemini 提示词双语手册。"
excerpt_en: "Covers Google's official prompt writing framework (Role/Task/Context/Format) with multi-scenario examples, plus the Gemini prompting bilingual handbook."
mathjax: true
hidden: true
permalink: '/blog/2025/03/01/google-prompt-engineering-guide/'
---

## Prompt Guide
本指南提供了在使用 Gemini 时编写有效提示的基础技能。按照规定的指南对提示词进行规范化撰写，可以更好地激活模型的性能并管理提示词。

相对于科学研究，本指南更侧重实际使用，这点与 [提示工程综述](/blog/2025/02/15/prompt-engineering-techniques-survey/) 和 [提示学习与上下文学习](/blog/2024/09/20/llm-prompting-and-in-context-learning/) 中的学术视角有所不同。

编写有效提示时要考虑的四个主要部分是：
* Roles
* Tasks
* Context
* Format

例如：You are a program manager in [industry]. Draft an executive summary email to [persona] based on [details about relevant program docs]. Limit to bullet points.

这个提示词完整地包括了所需的四个部分。并不是每个提示词都需要这四个部分，但考虑这四个方面会对任务处理有帮助。**有着明确动词的任务描述是 Prompt 最重要的部分**。

Quick Tips
1. 使用自然语言，用完整的句子表达完整的思想
2. 使用明确具体的提示词并不断迭代，提供充足的上下文
3. 简单明了，避免过于复杂
4. 结果不满意的时候，迭代微调提示以获得更好效果，这就是多轮对话的意义
5. 补充格式化的文档信息作为上下文
6. **使用提示词 Make this a power prompt 来优化提示词**

最有效的提示平均包含约 21 个单词的相关上下文，但人们尝试的提示通常少于九个单词。因此在认真撰写时尽量保持完整。

生成式 AI 意在帮助人类，但最终输出是你的。永远不要忘记检查它的输出结果。

## Introduction
了解什么构成了一个有效的提示、如何创作一个有效的提示，可以有效改进工作效率、提高生产力与创造力。Gemini 可以做到以下事情：

- 改进你的写作
- 整理数据
- 创建原创图像（视觉语言模型）
- 汇总信息并揭示洞察
- 通过自动记录更好地开会
- 轻松研究不熟悉的主题
- 发现趋势、综合信息并识别商业机会

本指南的后半部分介绍提示的方法，包括优秀的提示设计示例，以辅助理解前面介绍的 Tips。还涵盖了不同角色、用例和潜在提示的场景。

后面的提示示例中将包含多样的风格。有些提示带有括号，表示需要填写具体细节。通过输入 @文件名 标记您自己的个人文件。其他提示则没有突出显示变量，以展示完整提示可能的样子。本指南中的所有提示旨在激发创造力，但最终需要根据具体工作进行定制。

使用角色是提示环节非常重要的一环，我们会在后面不断见到它的应用。

## 基于场景的提示介绍
在阅读以下章节内容的同时，也在不断复习前面介绍过的提示撰写技巧。

### Administrative support
#### Plan agendas
行政支持辅助处理各种各样的行政事务。以 Plan agendas 为例，下面是一个完整的包含四个部分的提示词例子，使用 // 分割了主要部分。

I am an executive administrator to a team director. // Our newly formed team now consists of content marketers, digital marketers, and product marketers. We are gathering for the first time at a three-day offsite in Washington, DC. // Plan activities for each day that include team bonding activities and time for deeper strategic work. // Create a sample agenda for me.

这个提示词自然会生成一个不错的结果，但一定会有我们未能考虑到的细节。现在我们希望为团建活动增加一些具体想法。这个提示词只保留了 Task 以及 Context，因为 Role 在上一轮对话已经使用过了。

Suggest three different icebreaker activities // that encourage people to learn about their teammates' preferred working styles, strengths, and goals. Make sure the icebreaker ideas are engaging and can be completed by a group of 25 people in 30 minutes or less.

由于刚才提示词对结构的疏忽，以及我们希望每一轮模型只处理一个问题，现在来调整一下格式。这个提示词包含了 Task 以及 Format。

Organize this agenda // in a table format. // Include one of your suggested icebreakers for each day.

日程已经安排得不错了，来考虑一下其中的细节吧。这个提示词包含了一个文档作为上下文，并提供了动词引导的 Task。

Use @2024 H2 Team Vision // to generate a summary for the opening remarks on Day 1 of this agenda.

#### Manage multiple email inboxes
处理过去一周内的诸多邮件，进行一些简单的概括，需要模型与场景的结合。

Summarize emails from (manager) from the last week. (Gemini in Gmail)

我们还需要确认重要邮件里的事项以及相关 DDL。

Summarize this email thread and list all action items and deadlines.

基于一些文件来生成相关的回复，如果满意的话就可以直接使用了。

Generate a response to this email and use @file to describe how the initiative can complement the workstream outlined in colleague's message.

#### Travel
规划一个商务旅行中的行程，不包含工作事务，这也是一个很完整的提示。

I am an executive assistant. I need to create an itinerary for a two-day business trip in [location] during [dates]. My manager is staying at [hotel]. Suggest different options for breakfast and dinner within a 10-minute walk of the hotel, and find one entertainment option such as a movie theater, a local art show, or a popular tourist attraction. Put it in a table for me.

现在来创建一个表格追踪预算情况，这需要使用外部软件了。

Create a budget tracker for business travel. It should include columns for: date, expense type (meal, entertainment, transportation), vendor name, and a description.

### Communications
现在我们来看看如何与他人交流。

#### Create a press release
根据现有内容创建一份新闻发布稿，它基本包含了我们前面要求的部分，并且从文档引入了上下文。

I'm a PR manager. // I need to create a press release with a catchy title. // Include quotes from @[VIP Quotes Acquisition].

初稿不可能令人满意，我们根据 Gemini 返回的结果进行进一步的修改。这种延续对话的修改最需要保留的就是 Context 和 Task。

Use @[Biography and Mission Statement] to add more information about the company that is being acquired, its mission, and how it got started.

创建内部沟通文件也很容易。

I need to draft a company-wide memo unveiling our relaunched intranet. // The [new page] addresses [common feedback we heard from employees] and aims to create a more user friendly experience. // Draft an upbeat memo announcing [the new site] using @[Intranet Launch Plan Notes].

#### Prepare for analyst or press briefings
首先需要一个 press briefings 的模板，之后再逐步填充内容。

Generate a brief template to prepare [spokesperson] for an upcoming media and analyst briefing for @[Product Launch]. Include space for a synopsis, key messages, and supporting data.

现在来基于已有的内容填充一下。

Craft a synopsis of the product launch in three main points using @[Product Launch - Notes].

或许也可以考虑使用 AI 帮助我们制作 PPT 以及 Excel 表格，不仅仅是表格的格式，还可以根据文档来组织表格的内容。

Organize my media and analyst contacts from @[Analyst and Journalist Contact Notes] for a new product briefing. I need to keep track of their names, type of contact (analyst or journalist), focus area, the name of the outlet, agency or firm that they work for, and a place where I can indicate the priority level of their attendance at this briefing (low, medium, high).

Create a slide describing what [product] is from @[Product Launch - Notes]. Make sure it is short and easily understood by a broad audience.

#### Interview
来生成一些 Interview 的问题。

I am a [PR/AR] manager at [company name]. We just launched [product] and had a briefing where we discussed [key messages]. I am preparing [spokesperson and role/title] for interviews. Generate a list of mock interview questions to help [spokesperson] prepare. Include a mixture of easy and hard questions, with some asking about the basics of [product] and some asking about the long-term vision of [product].

我们也可以针对这些问题自己回答，或者进一步深化问题。此时省略 Role，侧重后三者，如果没有格式要求则侧重两者。

Use @[Product Launch Notes] to write suggested answers for these questions. Write the talking points as if you are [title of spokesperson] at [company].

### 其他可以在双语PDF手册中获得的内容
#### Customer service
#### Executives

#### Frontline management

#### HR

#### Marketing
营销难免涉及与视觉相关的内容，我们在这里讨论一个关于视觉生成的例子。

Generate ideas for a creative and eye-catching logo for my new business, // a coffee shop combined with a video game cafe. Generate a logo considering the following:
Dual Concept: The logo needs to clearly signal both the coffee and gaming aspects of the business without being too cluttered.
Target Audience: Appeal to a wide range of gamers (casual and enthusiast), as well as coffee lovers seeking a unique hangout spot. //
Style Options: I'm open to these approaches — let's get a few examples in each of these three styles to compare: Modern and Playful: Bold colors, fun graphics, maybe a pixel art aesthetic. Retro-Cool: Think classic arcade style — chunky lettering, neon color inspiration. Sleek and Minimalist: Clean lines, geometric shapes, a more subtle nod to both themes.

整体提示词的主体部分都是对生成风格的要求，最后的 Style 则是对生成格式的要求。对于图片模型的生成，充足的上下文来限制其自由度非常重要。

在第一阶段的生成结束后，可以开始另一阶段的生成（基于第一阶段的结果）。

I like the retro-cool options. Can you provide three more in that same style?

图像生成后，可以开始考虑一些关于文字的 Marketing 问题。

Write a tagline and 10 potential names for the business to go with these logos.

#### Project management

#### Sales
#### Small business owners and entrepreneurs
#### Startup leaders



## 写在结束
本指南旨在提供灵感，下面是一些使用建议。看完示例后或许会对这些建议有更进一步的理解。LLM 的能力几乎是无限的，尝试去利用它。

- 分解任务：如果希望 Gemini 执行多个相关任务，将它们分解为单独的提示。在需要的时候可以整理相关内容后展开新一轮对话，避免长上下文导致的性能下降。
- 提供约束条件：为了生成特定结果，在提示中包括详细信息，如字符数限制或希望生成的选项数量。
- 使用角色：RolePlay 可以提升模型的能力，在需要的时候将 Role 放在提示词的最前面就可以了。
- 请求反馈：明确告诉 Gemini 应该返回什么，详细描述已有的细节和希望的输出。
- 考虑语气：根据目标受众调整提示的语气，这取决于我们要把生成的内容拿去做什么。
- 改进提升：提示词的质量没有最好，根据输出结果来考虑如何进一步改进提示词。有时候问题不应该依靠几轮对话解决，长对话对模型性能是非常不利的。

**Generative AI is meant to help humans, but the final output is yours.**

> 本文配套的 Gemini 提示词双语手册：[Gemini_Prompt.pdf](/assets/Gemini_Prompt.pdf)
