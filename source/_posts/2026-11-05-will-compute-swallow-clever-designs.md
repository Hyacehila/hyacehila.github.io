---
title: "那些精巧的设计，终究会被算力吞掉吗？"
title_en: "Will Computation Eventually Swallow Our Clever Designs?"
date: 2026-11-05 00:00:00 +0800
categories: ["Foundation Models", "Model Mechanics"]
tags: ["Model Architecture", "AI Agent", "Scaling Laws"]
author: Hyacehila
excerpt: "从数学探索、机器人操作到 GPU 上的模型结构，通用方法不断挤进专用设计的地盘。它们为什么能赢，又有什么东西是更多算力无法替我们决定的？"
excerpt_en: "From mathematical exploration and robotics to model architectures on GPUs, general methods keep moving into territory once reserved for specialized designs. Why do they succeed, and what remains for us to decide?"
mathjax: false
hidden: false
permalink: '/blog/2026/11/05/will-compute-swallow-clever-designs/'
---

最近看到的几个事情，让我又想起了 The Bitter Lesson。

9 月，[OpenAI 公布了关于 NS（Navier–Stokes）问题的证明方案](https://openai.com/index/navier-stokes-solution/)。研究中，约一万个并发 Agent 分头探索、交换发现并进行验证，在带外力的设定下构造有限时间奇点。这数学题的解法听起来确实有点暴力。另一边，[Astra 能在 Blender 里建房子，再放进 Unreal Engine](https://openai.com/index/gpt-6-astra/)；[Robocurve 还让它控制机械臂](https://openai.robocurve.org/gpt-6-astra/)，把积木放进碗里。

这些事情离我们平时使用的 AI 好像都有点远，但放在一起看，会冒出一个问题：以前觉得应该为数学、建模、机器人分别设计的系统，现在是否可以让一个通用模型带着工具去尝试？通用模型正在进入更多原本属于专用系统的场景，这让我想重新看看一个老问题：那些精巧的设计，终究会被算力吞掉吗？

## 又是一次 Bitter Lesson？

Richard Sutton 在 2019 年写的 [The Bitter Lesson](https://www.cs.utexas.edu/~eunsol/courses/data/bitter_lesson.pdf) 里总结了一个观察：从长期看，能够持续利用更多计算的通用方法，尤其是搜索和学习，往往会超过依赖人类领域知识精心设计的方法。

这个教训为什么苦涩？因为人类并不是随便设计的。我们理解问题、寻找规律、加入经验，设计出的结构也确实能解决当时的问题。然后计算规模扩大，一种看起来更笨、更通用的方法追了上来，甚至超过了之前的方案。我们花了很长时间教机器怎么做的事情，它可能通过学习和搜索自己找到了做法。

通用不意味着系统什么都不用设计，更不意味着领域知识突然失去了价值。训练方法、工具接口、搜索过程和反馈机制，都需要人去构建；Agent 使用的编程语言、软件和验证工具，也积累了大量人类知识。变化在于，我们未必需要把每个任务的解法都提前写进模型或工作流。我们可以提供工具和反馈，让模型在解决问题时探索更多可能。

Bitter Lesson 或许在提问：**我们精心设计的东西，究竟是问题不可缺少的一部分，还是因为当前模型能力不够，才不得不加上的补丁？** 这个问题没有一个对所有场景都成立的答案。模型能力增长得越快，就越值得重新问一遍。

## 算力从哪里来

等等，上面好像说得很轻松：多用点计算，通用方法就能进步。但计算又不是天上掉下来的，模型得先能在硬件上高效地跑起来。说到这里，就可以接上我之前看到的一个[关于 DeepSeek 与 n-gram 结构的知乎回答](https://www.zhihu.com/question/2031512824917267006/answer/2032447961972597290?share_code=1dybiBtxsSsmS&utm_psn=2032984959640711218)。

它提出的思路很有意思：在大模型工程里，看起来精巧的结构，未必能赢过看起来粗糙的结构。一个新模块也许在理论上更优雅，或者在小实验中提高了指标，但如果它引入不规则的内存访问、难以并行的分支，或让原本连续的大计算变成许多小算子，就可能拖慢整条训练和推理链路。省下了一些计算量，却没省下实际运行时间，甚至还更贵了（听起来有点冤）。

Transformer 的优势之一，是很多计算可以组织成规则的大规模矩阵运算，也更容易并行。[原始论文](https://arxiv.org/abs/1706.03762)就把并行性和训练时间作为设计优势来讨论。后来，硬件、计算内核和分布式训练工具又围绕这类计算不断优化。纸面上精美的设计所带来的优势，有时候难以在工程系统中体现出来。

这就是我理解的“差就是好”：**一个看似笨重但能高效利用现有硬件的方案，可能比精巧却难以运行的方案更容易扩大规模，然后靠规模取胜。** 这不是说简单结构总是更好，也不是说 n-gram 或其他专门结构没有价值。它们能否留下来，仍然要看具体实现、实验结果和系统成本，优雅的结构并不是一切。

## 硬件也买了一张彩票

如果说 Bitter Lesson 关注的是哪些方法能从更多计算中获益，那么 Sara Hooker 的[硬件彩票（The Hardware Lottery）](https://arxiv.org/abs/2009.06489)提醒我们：哪些方法能方便地获得这些计算，本身也受到硬件和软件生态的影响。

一种方法胜出，可能因为它确实有很强的能力，也可能因为它恰好适合现有硬件，能够用成熟工具高效训练。更多人使用它，更多优化围绕它积累，它又因此变得更容易使用。这些因素会相互强化。反过来，某个方向即使有潜力，也可能因为缺少合适的硬件和软件支持，很难在相同投入下展示出来。听起来像不像 CUDA 生态和它的追赶者？

硬件彩票给我们的判断加了一个条件：**今天看到的赢家，是在今天的计算条件下赢的。** 如果硬件、工具或成本结构发生变化，值得尝试的设计也可能改变。

这也解释了为什么“差就是好”和 Bitter Lesson 能放在一起聊，但不能画等号。前者问的是一个方案能否高效使用现有系统；后者关注的是方法能否随着计算增加持续改善。两者可以互相帮助：一个容易高效运行的方法，更有机会获得规模化投入。

## 那些设计还要不要做

说到这里，好像该回答标题里的问题了：精巧的设计会被算力吞掉吗？我的答案是，有些会，但我们现在很难提前知道是哪些。

模型变强以后，一部分为了弥补能力不足而设计的流程可能不再需要。以前需要拆成很多步骤、为每种情况写规则的任务，也许可以交给模型直接处理；以前必须单独训练模型才能完成的操作，也许可以通过通用模型和工具调用实现。这些变化值得期待，毕竟少维护一套复杂系统，大家都能轻松一点。

不过，能完成任务和适合实际使用是两回事。一个专用方案可能更便宜、更快，也可能更容易测试和控制；在需要稳定响应的地方，这些都是真实的需求。即使未来通用模型能完成同一件事，也仍然需要比较质量、成本、延迟和可靠性。不要因为模型能做，就默认它应该负责所有事情（显卡也需要休息）。

所以，我不想把 Bitter Lesson 当成“停止设计，等待更大的模型”这样的结论。今天需要解决的问题，仍然需要今天的设计；只是这些设计也应该有机会随着模型和基础设施的变化被重新审视。有些会被替代，有些会变成工具、约束或反馈机制，继续留在系统里。

那些精巧的设计，终究会被算力吞掉吗？也许一部分会。更让我好奇的是，当模型接过越来越多我们曾经亲手编排的步骤以后，我们会把省下来的时间和注意力放在哪里。我们想解决什么问题，希望得到怎样的结果，这些问题还会留给我们。至少目前，我觉得它们值得比“能不能再加一个模块”多花一点时间。

## 参考资料

1. OpenAI. [On the Navier–Stokes Millennium Prize Problem](https://openai.com/index/navier-stokes-solution/). 2026-09-08，更新于 2026-09-10。
2. OpenAI. [GPT-6 Astra: A new generation of intelligence](https://openai.com/index/gpt-6-astra/). 2026-09-03。
3. Robocurve. [GPT-6 Astra on robotic manipulation](https://openai.robocurve.org/gpt-6-astra/). 2026-09-04。
4. Richard Sutton. [The Bitter Lesson](https://www.cs.utexas.edu/~eunsol/courses/data/bitter_lesson.pdf). 2019-03-13。
5. 关于 DeepSeek 与 n-gram 结构的[知乎回答](https://www.zhihu.com/question/2031512824917267006/answer/2032447961972597290?share_code=1dybiBtxsSsmS&utm_psn=2032984959640711218)。
6. Sara Hooker. [The Hardware Lottery](https://arxiv.org/abs/2009.06489). 2020。
7. Ashish Vaswani et al. [Attention Is All You Need](https://arxiv.org/abs/1706.03762). 2017。
