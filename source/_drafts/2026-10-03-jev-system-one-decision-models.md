---
title: "Jev：当软件只需要模型做个判断"
date: 2026-10-03 20:00:00 +0800
categories: ["Foundation Models", "Model Mechanics"]
tags: ["Model Architecture", "AI Agent", "Evaluation"]
author: Hyacehila
excerpt: "从 TypeSafe 的 Jev，到阿里云百炼的决策模型与 OpenAI Decisions API，模型开始把判断直接交给软件。简单回看这几周的时间线，以及这种变化为什么值得关注。"
mathjax: false
permalink: '/blog/2026/10/03/jev-system-one-decision-models/'
---

最近看到 [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)，觉得挺有意思。我们已经习惯让模型先写一段话，再由程序从里面拿到一个判断。但很多软件其实只需要知道：这条请求交给谁，这份材料是否相关，现在要不要调用工具。

之前写 [Interaction Model](/blog/2026/06/23/joyai-vl-interaction/) 时，我关心的是模型能不能走出一问一答，进入连续交互。Jev 又把另一个问题摆到了面前：当模型的使用者是程序时，智能应该以什么形式交出来？

## Jev 提供了什么

Jev 是 TypeSafe AI 的产品名，公司把这类模型叫作 **System One Models**，借用了快思考与慢思考的区分。它面向的是那些范围明确、需要反复执行的判断。

例如，把一条客服消息作为状态输入，再给出财务、技术、销售三个选项，模型就返回选择与各选项的概率。[它的接口](https://docs.typesafe.ai/introduction) 提供三种基本问题：Choice 选一个答案，Score 按给定等级评分，Noul 返回一个命题为真的概率。同一份状态上的多个问题可以并行回答，之后由代码组合结果、决定下一步。

分类器当然早就存在，普通 LLM 也能做这些事。Jev 有意思的地方，是把通用语义判断做成一种可组合的模型接口，并让训练和推理围绕它展开。TypeSafe 将训练方法称为 [RLCD（Reinforcement Learning for Calibrated Decisions）](https://docs.typesafe.ai/introduction/machine-learning-primer)：目标包括判断的正确性，以及概率与实际正确率是否相称。按官方描述，输出也直接并行产生，省去了逐 token 生成答案的过程。

TypeSafe 在[发布文章](https://typesafe.ai/blog/introducing-system-one-models-and-jev)中给出的端到端延迟为 70–500 毫秒，输入价格为每百万 token 0.042 美元，输出免费。这些是厂商公布的数字，实际收益还要看输入长度、网络和任务。它让人感兴趣的原因很直接：如果一次语义判断足够便宜、足够快，软件就能在更多地方用上它。

## 这几周的时间线

以下整理截至 **2026 年 10 月 3 日**能核实的公开信息。

| 时间 | 发生了什么 |
| --- | --- |
| 2026-09-15 | TypeSafe [发布 Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)，开放早期访问，并提出 System One Models。 |
| 2026-09-16 起 | 社区开始尝试类似接口。[Open Jev](https://github.com/JoshuaSP/open-jev) 用 DiffusionGemma 实验结构化决策；作者明确说明，这没有复现 Jev 的训练与概率校准。 |
| 2026-09-24 | 阿里云百炼的[模型更新列表](https://help.aliyun.com/zh/model-studio/newly-released-models)记录了 `decision-model-preview`，支持分类、是非判断与评分。 |
| 2026-09-29 | OpenAI 在 [DevDay](https://openai.com/index/devday-2026-recap/) 公布 Decisions API，让 Luna 回答预设的有限选项；发布时为有限预览，计划随后扩大开放。 |

所以，跟进已经发生了。阿里云的[决策 API](https://www.alibabacloud.com/help/en/model-studio/decision-model-api)甚至明确采用 TypeSafe System One 协议；OpenAI 则把这类能力放进 Luna 的接口，并支持文本和图片上下文。不过，相似的产品目标还不能说明它们使用了相同架构或 RLCD 配方。

开源侧也有 [Laya](https://huggingface.co/convaiinnovations/laya)，以及基于 [Qwen 的社区实验](https://github.com/TheoLeeCJ/SemIf-OpenJev)。Google/Gemma 和阿里/Qwen 的模型被社区拿来使用，与厂商正式推出专用决策产品，是两件不同的事。对于 Anthropic 和 Meta，此次检索还没有找到可确认的同类专用产品公告。

## 它可能改变什么

我更看好的是这种能力在普通软件里的位置。写死的规则很快，但遇到表达变化就容易失效；每次调用完整的生成模型，又可能太慢。决策模型有机会让一些原本需要人工理解的分支，也能被频繁调用：请求路由、材料筛选、结果检查，以及 Agent 下一步动作的选择。

它也可以和更强的生成模型配合。常见判断快速完成，拿不准的任务再升级，复杂问题交给长程推理，最后由代码负责执行。我觉得这会让 Agent 的分工更细，而不必把每个小判断都放进一次完整的思考与生成。

当然，选项固定只能避免输出越出类型，不能保证选得对。概率校准也需要在自己的任务上检验；[接口里的 confidence](https://docs.typesafe.ai/confidence)描述的是概率分布有多集中，不能直接当作正确率。真正有用的指标，是系统能自动处理多少任务，以及其中有多少判断出错。

Jev 的[名字来自 Jevons](https://typesafe.ai/blog/introducing-system-one-models-and-jev)：效率提高以后，使用量可能增长得更多。放到这里，我觉得这个比喻挺贴切。如果模型判断的成本继续下降，它也许会像搜索、排序和数据库查询一样，成为软件里随处可以调用的一种能力。
