---
title: 'Bad Is Good: Why DeepSeek Did Not Use an n-Gram Structure'
title_zh: 差就是好：从 DeepSeek 未采用 n-gram 结构说起
date: 2026-04-30 00:00:00 +0800
categories:
- Foundation Models
- Model Mechanics
tags:
- Model Architecture
- Pre-Training
author: Hyacehila
excerpt: 'A Zhihu answer offers a useful reminder: in large-model engineering, clever structures do not always beat simple
  ones that keep GPUs full and follow matrix-multiplication pipelines.'
description: 'A Zhihu answer offers a useful reminder: in large-model engineering, clever structures do not always beat simple
  ones that keep GPUs full and follow matrix-multiplication pipelines.'
excerpt_zh: 一个知乎回答提醒我们：在大模型工程里，精巧结构未必胜过简单结构。能稳定吃满显卡、顺着矩阵乘法流水线运行的设计，往往更容易留下来。
permalink: /blog/2026/04/30/deepseek-ngram-poor-is-good/
lang: en
translation_key: 2026-04-30-deepseek-ngram-poor-is-good
translation_status: machine
translation_source_hash: f3d3616dc2233f50f17f5469afff6e6425166f25ad7475430e3fe25bb5a6671e
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<blockquote>
<p>This is the original.<a href="https://www.zhihu.com/question/2031512824917267006/answer/2032447961972597290?share_code=1dybiBtxsSsmS&amp;utm_psn=2032984959640711218">I know what you're saying.</a>The digest and the light sorting.</p>
</blockquote>
<p>The questions in this article can also be addressed<a href="/en/blog/2024/08/15/llm-lifecycle-overview/">LLM Life Cycle Overview: From Data, Pre-Training to Decoding and Deployment</a>、<a href="/en/blog/2026/01/20/from-llm-to-vlm-visual-understanding/">From LLM to VLM, how language models achieve visual understanding</a>How the concept of a relatively close read together is developed in different contexts.</p>
<p>The answer to this question is a very straightforward one of the anti-intuitive patterns in the design of the large model structure:<strong>It's not good.</strong>。</p>
<p>Here, the “discretion” points to a engineering orientation: to abandon excessive sophistication and to give priority to hardware efficiency. A structure can stay in today's large model system, depending first on whether it feeds the graphic cards and runs along the path of mature matrix multiplication, hologram, communication and translation optimization.</p>
<p>The architecture choices of the age of the large model often swing between the two design philosophy. One focuses on theoretical completeness, generalization of bias, interpretable memory mechanisms and more detailed structural expression; the other is more concerned with insinuation, visibility, parallelity, kernel integration and the expansiveness of the project. The former looks more elegant, the latter looks more crude. In large-scale training, the rough side often wins.</p>
<p>The reasons are not complex: models are too big to tolerate too many “precision but difficult” structures. Once theoretical beauty does not translate into actual ingestion, it will be quickly offset by training costs and delays in reasoning.</p>
<h2>Simple overwhelms.</h2>
<p>Complex structures can be easily moved. They commit to greater expression, more natural long-range memory, more rational retrogressive reasoning, and even to embedding controllers, graphics, search modules and hierarchy in models. On paper, such designs tend to be more closely related to the understanding of the mission.</p>
<p>Trouble out on that. Training and reasoning are broken down into small numbers of calculations as long as dynamic control streams, non-ruled memory access, and difficult to integrate branches are introduced. The parameters may be saved, the GPU may not be fully exploited. In large model engineering, savings in mathematics do not necessarily lead to system acceleration.</p>
<p>Transformer's victory was largely due to the simplicity of this project. Its main calculations can be organized into large arrays, with attention, MLP, disability and reclassification all receiving mature bottom-up optimization. FlashAttention, the hologram, the parallel distribution and the optimization of the compilers are all stacked up along this rule.</p>
<p>So the first lesson is:<strong>Simple overwhelms.</strong>I'm sorry. The simple point here is that the algorithm, memory access rules, parallel model rules can allow hardware to operate in a stable and long-term manner.</p>
<h2>Rightness has to be the way to speed.</h2>
<p>Many structures are more “right” in theory. The model would seem to be dependent on a clear expression sequence, should be more modular, should incorporate a general bias towards long-range relationships and should reduce over-parametricization that appears to be heavy.</p>
<p>These judgements are correct, but they are often the first in large-scale training. When data, parameters and algorithms are pushed to a high level of magnitude, local theoretical correctness is swallowed up by the whole body. A heavy but fast-moving structure often reaches its available capacity earlier than a sophisticated but fast-moving structure.</p>
<p>This explains why many of the “looking smarter” modules have not become the backbone. They may be beautiful in small experiments, and once placed in a genuine training chain, they will face problems such as falling through, visible debris, complex communications, kernel incognito, unstable reasoning services.</p>
<p>The so-called “good is good” is a trade-off at the systemic level. A design must be compensated for by a sufficiently large amount of revenue, provided that it significantly reduces the availability of FLOPS. Otherwise, its validity is confined to the blackboard and the small experiment.</p>
<h2>Consistency is more than a fusion.</h2>
<p>The third lesson is:<strong>Consistency is more than a fusion.</strong>。</p>
<p>The cost of complex structures is not limited to individual modules. More problematic is the inconsistency between the parallel models between modules, the memory layout, the mode of communication and the dispatch strategy. System optimization requires a stable calculation map; the more isomer components are assembled, the more the space is optimized.</p>
<p>Transformer's strong, too, is because it's enough to repeat. Linear transformation, non-linearity, disability, fusion, stacking, and stabilization of the computational chart, optimized the stacks around it for many years. Encoders are good, decodors are good, and the subject logic is not frowning, but it is well suited to be pelted over and over again by the entire hardware and software ecology.</p>
<p>For a local target to be inserted into a special structure, it may be possible to take more advantage of a single point indicator or to make the whole training and reasoning chain pay for it. Large models are systems that are co-composed in terms of computing, data, communications and deployment, and individual modules must be fine-tuned to overall efficiency.</p>
<h2>Integrity can be sacrificed.</h2>
<p>The Academy design naturally wants to cover more scenarios: super long files, strict logic, low bit quantification, multimodular alignment, long-range memory, controlled retrieval. Each goal can lead to a dedicated structure, each of which has its own rationale.</p>
<p>The trade-off in the large model project is colder. When a simple decode model already solves most tasks, the remaining special scenarios do not necessarily warrant additional complexity for the backbone structure. Continuous stacking of computing power, stacking of data, context expansion, kernel modification are in many cases more cost-effective than introducing a dedicated module.</p>
<p>That is what it means to be sacrificed. Special scenarios remain important, but the backbone model must prioritize the most common, stable and easily scalable path.</p>
<p>By the time the next generation of graphic cards, context windows and attention are made to move forward, problems that seem to require a dedicated structure today may be covered by cheaper generic calculations. Rather than pursuing a whole-of-the-art architecture, simple structures should continue to grow with the force and data.</p>
<p>This trade-off sounds unprecision, but it fits into the experience of large models: things that can be scaled up efficiently, they can swallow things that can only look pretty on a small scale.</p>
<h2>Return to n-gram structure</h2>
<p>Back to DeepSeek and the n-gram structure, the judgment became clear. Even if it comes from ideas that it has put forward, a new structure does not mean that it should be integrated into the backbone model. It is subject to a common test of theoretical, experimental, training systems, reasoning services and hardware efficiency, and single paper indicators are not sufficient.</p>
<p>Neither the elegance of the concept nor the small-scale benchmark improvements are sufficient. It also needs to be efficient in running on GPUs, with access to training frameworks, parallel strategies, cache mechanisms and reasoning links, and to be sufficiently profitable to cover losses in terms of throughput, delay and complexity.</p>
<p>It is not surprising that it cannot be done.</p>
<p>The engineering reality of the age of the large models is simple: fast-calculated structures, which keep pressing slow structures. The matrix multiplication seems stupid, and as long as it works on the graphic cards, it is more alive than many of the precision techniques.</p>
<p>The final design to be left behind may not be the most elegant, but it must combine hardware, data and size. It is difficult to be a backbone without a sophisticated structure that can reach high-level water lines.</p>
<p>That is what it means to be "good." It recognizes the basic fact in large-scale modelling: size can penalize precision that is difficult to optimize, hardware can reward rules, duplication, and parallel simplicity.</p>
