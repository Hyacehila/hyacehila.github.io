---
title: "从 Token 拟合到概念学习：DFT/RL 现象如何进入大模型可解释性讨论"
date: 2026-04-30 21:00:00 +0800
categories: ["Foundation Models", "Model Mechanics"]
tags: [LLM, DFT, RLHF, Interpretability, Post-training]
author: Hyacehila
excerpt: "DFT/RL 并不是简单地让模型少说连接词，而是在削弱逐 token 仿写的均匀压力。真正值得解释的是：后训练是否让模型少为表面 token 付费，多为概念状态和推理分叉付费。"
---

一个值得认真讨论的现象是：在 SFT、DFT 和 RL 类后训练方法之间，模型似乎不只是“能力分数”不同，它们对不同 token 的训练压力也不同。

标准 SFT 的基本形式是 teacher forcing 下的 next-token prediction。训练时，每个参考答案里的 token 都会进入交叉熵损失：核心实体、推理转折、答案格式、连接词、标点、语气词，都会被要求尽可能贴近示范轨迹。这当然有好处。SFT 很擅长让模型学会格式、语气和任务模板。但问题也在这里：它容易把“这个答案为什么对”和“这个答案长得像什么”绑在一起。

[DFT 论文](https://arxiv.org/abs/2508.05629)提供了一个很好的切入口。它把标准 SFT 的梯度重新解释成一种带有隐式奖励的 policy gradient，并指出低概率参考 token 会在 SFT 中得到过大的更新压力。DFT 的做法很简单：用模型当前给参考 token 的概率去重加权 token-level cross entropy，从而削弱那些模型本来就不确信的 token 对梯度的异常牵引。论文的 token probability 分布分析还观察到：SFT 倾向于把训练集 token 的概率整体推高，而 DFT 会形成更明显的两极化；DPO、GRPO、PPO 也有类似但更温和的趋势。最低概率桶里常见的是诸如连接词、冠词和标点之类的语法性 token。

这并不意味着“连接词没有意义”。更准确的说法应该是：后训练可能在降低一部分**低决策贡献 token**的拟合优先级。

“低意义词汇”这个说法容易误导。否定词、介词、数量词、因果连接词经常会改变整个句子的真值条件。“不是”“除非”“最多”“因为”都不是可以随便忽略的胶水。早期关于 [function word comprehension](https://aclanthology.org/S19-1026/) 的 probing 工作也提醒我们，功能词理解本身就是语言理解的重要组成部分，尤其是否定、介词、疑问词这类结构信号。

所以我更愿意把问题改写为：**哪些 token 在当前任务里真正改变了模型的概念状态、推理路径和最终答案，哪些 token 主要是在维持表面可读性？**

这个改写之后，它就自然进入了可解释性问题。

过去我们讨论可解释性，经常关心“模型有没有某个概念”“某个事实存在哪里”“某个 attention head 在做什么”。例如 [Towards Monosemanticity](https://www.anthropic.com/news/towards-monosemanticity-decomposing-language-models-with-dictionary-learning) 和 [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity) 用 sparse autoencoder 从激活中抽取更可解释的 feature；[Fine-Tuning Enhances Existing Mechanisms](https://arxiv.org/abs/2402.14811) 则展示了微调可能不是创造全新机制，而是增强基座模型里已有的机制。

如果把 DFT/RL 的 token 分配现象放进这个框架，问题会变得更具体：后训练到底增强了哪些 feature？它是在增强语法模板、输出格式和常见措辞，还是在增强实体追踪、条件分支、变量绑定、反事实判断、规划步骤这些更接近“概念操作”的内部特征？

这也是 RL 与 SFT 对比中最有意思的地方。[SFT Memorizes, RL Generalizes](https://proceedings.mlr.press/v267/chu25c.html) 的结论可以粗略概括为：RL 更容易在规则变体和视觉变体中学到可迁移策略，而 SFT 更容易贴合示例轨迹。不过这个判断不能绝对化。2026 年的 [Rethinking Generalization in Reasoning SFT](https://arxiv.org/abs/2604.06628) 已经指出，SFT 的泛化不是完全缺失，而是受优化时长、数据质量和基座能力共同制约。换句话说，SFT 不一定只能记忆，RL 也不天然等于理解；真正要解释的是训练信号如何被分配，哪些内部机制因此被放大。

另一个很贴合的证据来自 [High-Entropy Minority Tokens](https://shenzhi-wang.github.io/high-entropy-minority-tokens-rlvr/) 这条线。它把 CoT 中少数高熵 token 称为 reasoning forks：这些 token 不一定多，但它们会决定推理往哪个方向走。相比之下，大量低熵 token 主要是在补全局部句子、维持可读性或延续已经确定的路径。这和 DFT 的直觉互相呼应：模型不应该对所有 token 使用同一种拟合热情。

一个可执行的实验设计可以这样开始。

选同一个基座模型和同一批推理数据，分别训练 SFT、DFT、DPO 和 RLVR 版本。然后在相同 prompt 上记录四类信号：第一，参考 token 的 probability 和 entropy；第二，每个 token 对参数更新的梯度范数；第三，SAE feature activation，尤其是语法、格式、实体、变量、因果、分支判断相关 feature；第四，通过 activation patching 或 token 替换干预，测量某类 token 被替换后对最终答案和中间推理状态的影响。

然后不要按词表静态地把 token 分成“有意义”和“无意义”，而是按因果贡献分组：

1. 表面维持 token：替换后文本风格变了，但推理状态和答案基本不变。
2. 格式约束 token：替换后答案格式、可解析性或工具调用结构变化。
3. 概念承载 token：替换后实体、变量、关系或事实状态变化。
4. 推理分叉 token：替换后后续路径明显分叉，最终答案概率也改变。

如果 DFT/RL 真的在“少为表面 token 付费，多为概念路径付费”，我们应该看到：相较 SFT，它们对第一类 token 的拟合压力更弱，对第三、第四类 token 的机制强化更明显。这里的“强化”不只看输出概率，也要看内部 feature 是否更稳定、更可干预、更能跨题目迁移。

这个视角还会反过来影响评测。传统 BLEU、ROUGE 依赖表面重合，容易把措辞差异当成质量差异；[BERTScore](https://arxiv.org/abs/1904.09675) 和 [BLEURT](https://aclanthology.org/2020.acl-main.704/) 这类指标已经尝试用上下文表示或学习式评价更接近语义判断。但如果我们进一步关心可解释性，也许还应该问：两个答案是否激活了相似的概念特征？是否经过了相似的关键分叉？是否只是连接词不同，还是内部任务状态真的不同？

最后，这个问题最稳妥的表述不是“模型应该忽略连接词”，而是：

**好的后训练方法应该减少对表面轨迹的均匀仿写，把更多学习能力留给会改变概念状态、推理路径和任务结果的 token。**

DFT/RL 的价值，不只是让模型输出更高分答案，也可能在于它们把模型从“逐字模仿示范答案”推向“学习哪些 token 真正在做决策”。而可解释性的任务，就是把这种变化从输出层拉回到激活、特征、梯度和因果机制层面，看清楚模型到底是在学语言表面，还是在学可迁移的概念操作。
