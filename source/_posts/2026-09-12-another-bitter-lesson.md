---
title: "可能又是一次 Bitter Lesson"
title_en: "Perhaps Another Bitter Lesson"
date: 2026-09-12 00:00:00 +0800
categories: ["Agent Systems", "Agent Architecture"]
tags: ["AI Agent", "Scaling Laws"]
author: Hyacehila
excerpt: "从上万个智能体探索 NS 问题，到 Astra 建模和操作机械臂，最近的几个事情让我又想起了 Bitter Lesson。通用方法为什么能不断突破边界，硬件彩票又在其中起了什么作用？"
excerpt_en: "From thousands of agents exploring the Navier–Stokes problem to Astra modeling in Blender and operating robot arms, recent events have brought the Bitter Lesson back to mind. Why do general methods keep pushing beyond their boundaries, and what part does the hardware lottery play?"
mathjax: false
hidden: false
permalink: '/blog/2026/09/12/another-bitter-lesson/'
---

最近几天看到的几个事情，让我又想起了 The Bitter Lesson。

9 月 8 日，[OpenAI 公布了 NS（Navier–Stokes）问题的证明方案](https://openai.com/index/navier-stokes-solution/)：约一万个并发智能体参与探索，在带外力的设定下构造有限时间奇点。实现证明的是内部研究模型，GPT-6 Astra 负责后续的 Lean 形式化与验证。这规模看起来确实有点暴力，里面包含了各种分头搜索、交换发现和验证。

另一边，[Astra 已经能在 Blender 里建房子，再放进 Unreal Engine](https://openai.com/index/gpt-6-astra/)；[Robocurve 则让它控制机械臂](https://openai.robocurve.org/gpt-6-astra/)，把积木放进碗里，20 次成功了 19 次。图像输入和工具调用，让通用模型用上了人类已经造好的工具。当然，拼图嵌入只有 2/20，这些实验还不能证明它全面胜过专用建模模型或 VLA。但原来觉得应该为不同任务单独训练模型的地方，现在好像可以重新想想了，Agentic Gen 可能会更好的利用到通用模型能力的高增速。

Sutton 在 2019 年写下的 [Bitter Lesson](https://www.cs.utexas.edu/~eunsol/courses/data/bitter_lesson.pdf)，大意是：从长期看，能够不断利用更多计算的通用方法，尤其是搜索和学习，往往会超过精心加入人类领域知识的方法。苦涩的地方也很好理解。我们认真分析问题，设计结构，加入经验，做出了提升；然后计算规模增长，一种更通用的方法追了上来。之前的投入变得毫无价值。

把这个判断放到今天，**一些我们以为必须专门建模的能力，可能只是通用模型暂时还做不好。** 现在它能看图、写代码、调用工具，训练时有更多计算，解决问题时也可以花更多时间尝试。如果进一步的展开训练，进一步的扩展通用能力，或许越来越多的专用模型以及人的能力范畴会被它侵蚀。

这里还能接上 Sara Hooker 的[“硬件彩票”](https://arxiv.org/abs/2009.06489)。一个研究方向能够胜出，也可能因为它适合当时的硬件和软件。[Transformer 原始论文](https://arxiv.org/abs/1706.03762)就把更容易并行、减少训练时间当作优势。今天的 LM 也许恰好找到了一条适合现有硬件、容易持续扩大投入的路。算法的能力与基础设施的选择，就这样缠在了一起。

所以我现在更想问的是：今天那些看起来必须精心设计的部分，有多少来自问题本身，又有多少只是当前模型能力留下的空缺？我还答不上来。只是每次看到通用方法又往前走了一点，都觉得这个问题应该重新问一遍。可能又是一次 Bitter Lesson。

## 参考资料

1. OpenAI. [On the Navier–Stokes Millennium Prize Problem](https://openai.com/index/navier-stokes-solution/). 2026-09-08，更新于 2026-09-10。
2. OpenAI. [GPT-6 Astra: A new generation of intelligence](https://openai.com/index/gpt-6-astra/). 2026-09-03。Blender 与 Unreal Engine 演示。
3. Robocurve. [GPT-6 Astra on robotic manipulation](https://openai.robocurve.org/gpt-6-astra/). 2026-09-04。两项机械臂任务的结果与实验限制；比较对象为其他通用模型，未包含 VLA。
4. Richard Sutton. [The Bitter Lesson](https://www.cs.utexas.edu/~eunsol/courses/data/bitter_lesson.pdf). 2019-03-13。作者原文的高校存档。
5. Sara Hooker. [The Hardware Lottery](https://arxiv.org/abs/2009.06489). 2020。
6. Ashish Vaswani et al. [Attention Is All You Need](https://arxiv.org/abs/1706.03762). 2017。
