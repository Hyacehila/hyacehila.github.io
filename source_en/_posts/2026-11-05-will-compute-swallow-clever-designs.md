---
title: "Will Computation Eventually Swallow Our Clever Designs?"
title_zh: "那些精巧的设计，终究会被算力吞掉吗？"
date: 2026-11-05 00:00:00 +0800
categories: ["Foundation Models", "Model Mechanics"]
tags: ["Model Architecture", "AI Agent", "Scaling Laws"]
author: Hyacehila
mathjax: false
hidden: false
excerpt: "From mathematical exploration and robotics to model architectures on GPUs, general methods keep moving into territory once reserved for specialized designs. Why do they succeed, and what remains for us to decide?"
description: "From mathematical exploration and robotics to model architectures on GPUs, general methods keep moving into territory once reserved for specialized designs. Why do they succeed, and what remains for us to decide?"
excerpt_zh: "从数学探索、机器人操作到 GPU 上的模型结构，通用方法不断挤进专用设计的地盘。它们为什么能赢，又有什么东西是更多算力无法替我们决定的？"
permalink: '/blog/2026/11/05/will-compute-swallow-clever-designs/'
lang: en
translation_key: 2026-11-05-will-compute-swallow-clever-designs
translation_status: machine
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

A few things I've seen recently have brought The Bitter Lesson back to mind.

In September, [OpenAI published a proposed proof concerning the Navier–Stokes problem](https://openai.com/index/navier-stokes-solution/). Around ten thousand concurrent agents explored different approaches, exchanged findings, and carried out verification, constructing a finite-time singularity in a setting with external forcing. That's a pretty brute-force way to tackle a math problem. Meanwhile, [Astra can model a house in Blender and bring it into Unreal Engine](https://openai.com/index/gpt-6-astra/); [Robocurve has also given it control of a robot arm](https://openai.robocurve.org/gpt-6-astra/) to put a block into a bowl.

These examples may feel a little removed from how we normally use AI, but taken together, they raise a question: could a general-purpose model equipped with tools now attempt tasks for which we once thought we needed separate systems for mathematics, modeling, and robotics? General-purpose models are entering more territory once reserved for specialized systems. That makes me want to revisit an old question: will computation eventually swallow our clever designs?

## Another Bitter Lesson?

In [The Bitter Lesson](https://www.cs.utexas.edu/~eunsol/courses/data/bitter_lesson.pdf), written in 2019, Richard Sutton describes an observation: over the long run, general methods that can keep making use of more computation, especially search and learning, tend to surpass methods carefully designed around human domain knowledge.

Why is the lesson bitter? Because those designs weren't arbitrary. We study a problem, look for patterns, and add our experience; the structures we build really do solve the problems of their time. Then computation grows, and a seemingly dumber, more general method catches up or even overtakes the earlier approach. For something we spent a long time teaching machines how to do, they may find their own way through learning and search.

General-purpose doesn't mean a system needs no design, and it certainly doesn't mean domain knowledge has suddenly lost its value. People still have to build training methods, tool interfaces, search processes, and feedback mechanisms. The programming languages, software, and verification tools agents use also contain a great deal of accumulated human knowledge. What changes is that we may not need to write the solution to every task into the model or workflow in advance. We can provide tools and feedback and let the model explore more possibilities as it works.

Perhaps the Bitter Lesson is asking: **are the things we carefully design essential to the problem, or are they patches we need only because current models aren't capable enough yet?** There isn't one answer that applies to every situation. The faster model capabilities grow, the more often this question is worth asking again.

## Where Does the Compute Come From?

Wait, that all sounds rather easy: use more computation and general methods will improve. But compute doesn't fall from the sky. A model first has to run efficiently on hardware. That brings me to a [Zhihu answer about DeepSeek and n-gram structures](https://www.zhihu.com/question/2031512824917267006/answer/2032447961972597290?share_code=1dybiBtxsSsmS&utm_psn=2032984959640711218) that I came across earlier.

It makes an interesting point: in large-model engineering, a clever-looking structure won't necessarily beat a rough-looking one. A new module may be more elegant in theory or improve a metric in a small experiment. But if it introduces irregular memory access, branches that are hard to parallelize, or breaks a large continuous computation into many small operations, it can slow down the entire training and inference pipeline. You've saved some computation but no actual running time, and it might even cost more. (That feels a little unfair.)

One advantage of the Transformer is that much of its computation can be organized into regular, large-scale matrix operations, which are also easier to parallelize. The [original paper](https://arxiv.org/abs/1706.03762) discussed parallelism and training time as design advantages. Hardware, compute kernels, and distributed training tools have since been optimized around this kind of computation. The advantages of an elegant design on paper can sometimes be difficult to realize in an engineering system.

This is how I understand “bad is good”: **a seemingly clumsy approach that uses existing hardware efficiently may be easier to scale than a clever one that's difficult to run, and then win through scale.** That doesn't mean simple structures are always better, or that n-grams and other specialized structures have no value. Whether they earn a place still depends on their implementation, experimental results, and system costs. An elegant structure isn't everything.

## Hardware Bought a Lottery Ticket Too

If the Bitter Lesson asks which methods can benefit from more computation, Sara Hooker's [Hardware Lottery](https://arxiv.org/abs/2009.06489) reminds us that which methods can readily access that computation also depends on the hardware and software ecosystem.

A method may win because it really is powerful, or because it happens to fit existing hardware and can be trained efficiently with mature tools. More people use it, more optimizations accumulate around it, and it becomes easier to use as a result. These factors can reinforce one another. Conversely, even a promising direction may struggle to demonstrate its potential with the same investment if it lacks suitable hardware and software support. Does that sound a little like the CUDA ecosystem and those trying to catch up with it?

The hardware lottery adds a condition to our judgment: **today's winners have won under today's computing conditions.** If hardware, tools, or cost structures change, the designs worth trying may change too.

This also explains why “bad is good” and the Bitter Lesson belong in the same discussion, but aren't equivalent. The former asks whether an approach can use existing systems efficiently; the latter concerns whether a method can keep improving as computation increases. The two can help each other: a method that's easy to run efficiently has a better chance of attracting investment at scale.

## Should We Still Make Those Designs?

By now, I suppose it's time to answer the question in the title: will computation swallow our clever designs? My answer is that some will be swallowed, but it's hard to know in advance which ones.

As models become more capable, some processes designed to make up for their limitations may no longer be needed. Tasks that once had to be broken into many steps, with rules written for every case, might be handled directly by a model. Operations that once required a separately trained model might be possible through a general-purpose model and tool calls. These are changes worth looking forward to. After all, maintaining one less complex system makes life a little easier for everyone.

Still, being able to complete a task and being suitable for practical use are two different things. A specialized approach may be cheaper, faster, or easier to test and control. Where consistent responses are needed, those are real requirements. Even if a general-purpose model can perform the same task in the future, we still need to compare quality, cost, latency, and reliability. Just because a model can do something doesn't mean it should be responsible for everything. (GPUs need a rest too.)

So I don't want to take the Bitter Lesson to mean “stop designing and wait for a bigger model.” Today's problems still need today's designs. But those designs should also have a chance to be reconsidered as models and infrastructure change. Some will be replaced; others will remain in the system as tools, constraints, or feedback mechanisms.

Will computation eventually swallow our clever designs? Perhaps some of them. What interests me more is where we'll put the time and attention we save when models take over more of the steps we once orchestrated by hand. What problems do we want to solve, and what outcomes do we want? Those questions will remain ours. For now, at least, I think they deserve a little more time than “could we add another module?”

## References

1. OpenAI. [On the Navier–Stokes Millennium Prize Problem](https://openai.com/index/navier-stokes-solution/). September 8, 2026; updated September 10, 2026.
2. OpenAI. [GPT-6 Astra: A new generation of intelligence](https://openai.com/index/gpt-6-astra/). September 3, 2026.
3. Robocurve. [GPT-6 Astra on robotic manipulation](https://openai.robocurve.org/gpt-6-astra/). September 4, 2026.
4. Richard Sutton. [The Bitter Lesson](https://www.cs.utexas.edu/~eunsol/courses/data/bitter_lesson.pdf). March 13, 2019.
5. [Zhihu answer on DeepSeek and n-gram structures](https://www.zhihu.com/question/2031512824917267006/answer/2032447961972597290?share_code=1dybiBtxsSsmS&utm_psn=2032984959640711218).
6. Sara Hooker. [The Hardware Lottery](https://arxiv.org/abs/2009.06489). 2020.
7. Ashish Vaswani et al. [Attention Is All You Need](https://arxiv.org/abs/1706.03762). 2017.
