---
title: "Perhaps Another Bitter Lesson"
title_zh: "可能又是一次 Bitter Lesson"
date: 2026-09-12 00:00:00 +0800
categories: ["Agent Systems", "Agent Architecture"]
tags: ["AI Agent", "Scaling Laws"]
author: Hyacehila
mathjax: false
hidden: false
excerpt: "From thousands of agents exploring the Navier–Stokes problem to Astra modeling in Blender and operating robot arms, recent events have brought the Bitter Lesson back to mind. Why do general methods keep pushing beyond their boundaries, and what part does the hardware lottery play?"
description: "From thousands of agents exploring the Navier–Stokes problem to Astra modeling in Blender and operating robot arms, recent events have brought the Bitter Lesson back to mind. Why do general methods keep pushing beyond their boundaries, and what part does the hardware lottery play?"
excerpt_zh: "从上万个智能体探索 NS 问题，到 Astra 建模和操作机械臂，最近的几个事情让我又想起了 Bitter Lesson。通用方法为什么能不断突破边界，硬件彩票又在其中起了什么作用？"
permalink: '/blog/2026/09/12/another-bitter-lesson/'
lang: en
translation_key: 2026-09-12-another-bitter-lesson
translation_status: machine
translation_source_hash: 7de6233d5c6e89e8cc30fc480a2b9ac38a80497ced8b716f32bf612746431e86
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

A few things I've seen over the past several days have brought The Bitter Lesson back to mind.

On September 8, [OpenAI published a proposed proof resolving the Navier–Stokes problem](https://openai.com/index/navier-stokes-solution/): roughly ten thousand concurrent agents took part in the search, producing a construction of a finite-time singularity under external forcing. An internal research model produced the proof; GPT-6 Astra handled the subsequent Lean formalization and verification. The scale does look a little like brute force, with agents exploring different approaches, exchanging findings, and carrying out verification.

Meanwhile, [Astra can model a house in Blender and bring it into Unreal Engine](https://openai.com/index/gpt-6-astra/). [Robocurve gave it control of robot arms](https://openai.robocurve.org/gpt-6-astra/), and it placed a block into a bowl in 19 out of 20 trials. Image input and tool calling let a general-purpose model use tools people have already built. Of course, it completed the puzzle insertion task only 2 out of 20 times. These demonstrations don't establish that it broadly outperforms specialized 3D models or vision-language-action models (VLAs). Still, some tasks we assumed needed separately trained models seem worth another look. Agentic Gen may be better placed to take advantage of the rapid improvement in general-purpose model capabilities.

Sutton's 2019 essay, [The Bitter Lesson](https://www.cs.utexas.edu/~eunsol/courses/data/bitter_lesson.pdf), makes roughly this observation: over the long run, general methods that can keep using more computation, especially search and learning, tend to surpass methods carefully built around human domain knowledge. It's easy to see why that feels bitter. We study a problem, design structures, add our experience, and achieve improvements. Then computation grows, and a more general approach catches up. The earlier investment becomes worthless.

Looking at that observation today, **some capabilities we think require specialized modeling may simply be things general-purpose models aren't good enough at yet.** They can now inspect images, write code, and call tools. They have more computation during training, and can spend more time trying things when solving a problem. With further training and broader general capabilities, they may increasingly encroach on domains served by specialized models, as well as tasks within the scope of human abilities.

Sara Hooker's [“hardware lottery”](https://arxiv.org/abs/2009.06489) also fits here. A research direction may succeed partly because it suits the hardware and software available at the time. The [original Transformer paper](https://arxiv.org/abs/1706.03762) already presented greater parallelizability and shorter training time as advantages. Perhaps today's LMs have happened upon a path that fits existing hardware and makes it relatively easy to keep increasing investment. Algorithmic capabilities and infrastructure choices become intertwined.

So the question I want to ask now is this: of all the things that seem to demand careful, specialized design today, how many come from the problem itself, and how many fill gaps in current model capabilities? I don't have an answer yet. Every time general methods advance a little further, though, I feel the question deserves to be asked again. Perhaps this is another Bitter Lesson.

## References

1. OpenAI. [On the Navier–Stokes Millennium Prize Problem](https://openai.com/index/navier-stokes-solution/). September 8, 2026; updated September 10, 2026.
2. OpenAI. [GPT-6 Astra: A new generation of intelligence](https://openai.com/index/gpt-6-astra/). September 3, 2026. Blender and Unreal Engine demonstration.
3. Robocurve. [GPT-6 Astra on robotic manipulation](https://openai.robocurve.org/gpt-6-astra/). September 4, 2026. Results and limitations for two robot-arm tasks; the comparison covers other general-purpose models, not VLAs.
4. Richard Sutton. [The Bitter Lesson](https://www.cs.utexas.edu/~eunsol/courses/data/bitter_lesson.pdf). March 13, 2019. A university-hosted copy of the original essay.
5. Sara Hooker. [The Hardware Lottery](https://arxiv.org/abs/2009.06489). 2020.
6. Ashish Vaswani et al. [Attention Is All You Need](https://arxiv.org/abs/1706.03762). 2017.
