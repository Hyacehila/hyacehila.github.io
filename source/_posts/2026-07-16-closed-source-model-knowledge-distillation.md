---
title: "闭源模型不能被蒸馏？聊聊生成式语言模型的 Knowledge Distillation"
title_en: "Can Closed-Source Models Be Distilled? Knowledge Distillation for Generative Language Models"
date: 2026-07-16 20:00:00 +0800
categories: ["Foundation Models", "Training & Alignment"]
tags: ["Knowledge Distillation", "LLM Training", "Synthetic Data", "SFT", "On-Policy Learning"]
author: Hyacehila
excerpt: "闭源模型能否被蒸馏，要看教师实际提供什么信号、学生在拟合什么目标，以及这条训练链路是否获得授权；重点是训练链路怎么设计，而不是模型开源与否。"
excerpt_en: "Whether a closed-source model can be distilled depends on the signal it exposes, the student's training objective, and the authorization of the training pipeline—not simply on whether the model is open or closed."
mathjax: false
hidden: true
permalink: '/blog/2026/07/16/closed-source-model-knowledge-distillation/'
---

最近，蒸馏已经不只是一个技术词。Anthropic 把未经授权、批量利用其模型输出训练竞争模型的做法称为 [“distillation attacks”](https://www.anthropic.com/news/detecting-and-preventing-distillation-attacks)。中文讨论里则常有人反问：闭源模型既不公开权重，怎么会被蒸馏？

这看起来像一个 Yes-or-No 问题，实际上把不同层面的事混在了一起。无论直接回答“能”还是“不能”，都不够。

1. 教师模型究竟能提供什么信号；
2. 学生训练时到底在拟合什么；
3. 你是否有权这样使用模型与数据。

如果不把这三件事拆开，讨论很容易滑向口号：一边把所有模型辅助训练都叫蒸馏，另一边又因为模型没有公开权重，就断言它不能被蒸馏。

先把焦点放到训练链路上，暂时不管模型是开源还是闭源。

## 经典 KD：学生直接向教师靠拢

经典 Knowledge Distillation（KD）的核心很朴素：教师已经学会了某种行为，学生训练时直接缩小自己与教师之间的差异。

最典型的是**输出蒸馏**。Hinton 等人提出的 soft targets，不只告诉学生正确类别是什么，也保留了教师认为其他类别离正确答案有多近的信息。在分类任务里，学生拟合的是教师输出层给出的概率分布。放到生成式模型里，这个分布变成下一个 token 的概率分布；学生学习的，是教师在每一步更倾向于接着说什么。

除此以外，还有两类经典做法：

- **特征蒸馏**：对齐隐藏层、attention 或中间表示。FitNets 的直觉是，学生不只应模仿教师最终答题卡，还可以借助教师的中间提示学习。
- **关系蒸馏**：不强求两个模型每层都长得一样，而是让学生保留教师表示空间里的距离、角度、相似性等关系。

这三类方法说到底都在做同一件事：**学生的优化目标，就是更像教师。** 特征蒸馏和关系蒸馏通常需要访问隐藏层或权重，所以更适合白盒场景；经典 token-level KD 的门槛则最低，但仍旧需要访问最后的 logits，并需要保证学生和教师的词表能够对齐。

对 LLM 而言，黑盒模型虽然不能给出完整 logits，仍能给出一串 output tokens。学生把这串 token 当作目标，用交叉熵训练自己继续生成。这个损失形式和自回归预训练相同，差别只是监督信号从原始语料换成了教师输出。教师的最终回答、长推理文本、JSON、函数调用和工具调用轨迹，只要被原样拿来训练学生，都属于输出或序列级的行为模仿。训练信号没有变：学生还是在复刻一条已经给出的序列。这就是 Kim 和 Rush 所说的 sequence-level KD；从数据管线看，它也很像拿教师答案做伪标签 SFT。

这条边界没必要说得过于绝对。更有用的问题是：**学生更新时，是在直接复刻教师给出的输出，还是在一个重新设计过的数据与反馈系统中学习？**

## 用更强模型做数据合成，是另一条训练链路

让更强模型参与训练，不等于把它的输出整段搬进训练集。

模型也可以只是数据生产和评估链路里的一个工具：扩写种子任务、构造反例和难例、生成候选答案、协助标注偏好，或者充当筛选器。训练者再把它和检索证据、规则校验、单元测试、人工审核或奖励模型放在一起，形成 SFT、偏好优化或 RL 所需的数据。

Self-Instruct 的意义就在这里：自生成指令可以扩展 instruction tuning 数据，并不要求每条样本都由人工从零写起。但由模型生成不等于天然高质量。任务分布怎么选、哪些样本保留、什么叫正确、哪些行为应该拒绝、奖励什么、如何验证，仍然要靠训练者做判断。

这正是我想把两条链路分开的原因。

纯 KD 的目标是缩小学生与教师之间的距离。作为技术手段，它当然可以用于自有模型、明确授权的模型，或者团队内部的教师—学生训练链路。但如果语境变成“闭源强模型输出 → 小模型尽量复刻”，它在我看来更像行为复制：训练者几乎没有加入自己的判断，目标只是把教师说过的话再说一遍。这是一种缺少 taste 的行为，你不是在训练一个模型，而是在复制，没有决定什么值得教、什么不该学。

模型辅助的 SFT 或 RL 则是另一回事。这里的价值不只在教师说过什么，还在于人怎样设计课程、约束数据来源、加入验证器、安排难度、定义奖励和失败边界。它不必忠实复制教师的每一句话，甚至可以明确过滤教师的坏答案。这只是一种数据合成的技术，而不是纯粹的复制。

叫法也不能替代内容。把大量教师回答抓下来，原样塞进训练集，又没有新的任务设计和质量控制，哪怕叫合成数据，也还是另一种输出复刻。反过来，同一段 CoT 或工具轨迹，只要经过验证、重组，并被放进新的任务与评估链路里，也可以成为教学材料。**区别不在文本长相，而在训练链路搭建成什么样子。**

## On-policy KD：学生为什么要先自己写一遍？

上面区分的是训练目标。对自回归模型来说，即使目标仍是蒸馏，训练过程里还有一个麻烦：训练和部署时，学生见到的前缀并不一样。

普通的离线 KD 常常是这样：

> 真实数据或教师续写 → 学生模仿。

可是真到部署时，学生面对的不是教师提供的理想前缀，而是自己刚刚写出来的前缀。它可能已经弄错了一个实体、漏了一个条件，或者在工具调用里选错了参数。接下来它只能在这个有偏差的上下文里继续往下走。

On-policy KD 的做法是：

> 学生先生成 → 教师在学生自己的轨迹上给出分布或反馈 → 学生更新。

它处理的是训练和部署之间的状态分布失配，不是再造一份数据集。MiniLLM 从 on-policy 与 reverse-KL 的角度讨论 LLM 蒸馏；GKD 则直接研究学生自生成序列上的教师反馈。

因此，on-policy KD 不是第三种数据合成，也不等于教师生成数据后做 RL。它还是以教师行为为目标，只是让教师在学生实际会走到的轨迹上给反馈。实现时可以借用与策略优化相近的工具，但它讨论的仍是蒸馏。

## 黑盒与白盒：限制的是信号，不是结论

On-policy KD 问的是教师该在哪些轨迹上给反馈。再往下要问的，就是教师到底能给出什么信号。开源和闭源的差别主要落在这里，而不是直接决定能不能蒸馏。

- **白盒教师**：能访问 logits、隐藏层、attention 和关系结构。输出、特征、关系三类 KD 都可能成立，也更容易做细粒度的 on-policy KD。
- **只有最终文本的黑盒教师**：通常能提供答案、推理文本、工具轨迹或偏好判断，但做不了隐藏层蒸馏，也做不了完整词表上的 logits KD。
- **带 logprobs 的接口**：处于中间地带。即使权重不公开，若能拿到足够的 token 概率，仍可能做有限的输出层 KD；但这仍不是白盒特征蒸馏或关系蒸馏。

这也说明，闭源模型不能被蒸馏说得太满。更具体一点：

> 闭源模型未必支持白盒蒸馏；只有文本输出的接口也未必支持经典 token-level KD；但它们仍可能提供可被复刻的行为，或可被加工成 SFT/RL 数据的材料。

“闭源模型能不能被蒸馏”这个问法把几个层面压成了一句话，因此不适合一刀切回答。同一段教师输出，整批倒进训练集、让学生尽量复刻教师，是行为复制；训练者重新决定任务、筛选、验证和奖励时，教师只是材料来源之一。这些选择就是我说的 taste：什么值得教，什么样本可信，哪些错误必须拒绝。是否使用闭源模型只是表层，真正拉开差距的是训练者有没有把这些判断放进训练链路里。

## 参考资料

- Geoffrey Hinton, Oriol Vinyals, Jeff Dean, [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
- Adriana Romero et al., [FitNets: Hints for Thin Deep Nets](https://arxiv.org/abs/1412.6550)
- Wonpyo Park et al., [Relational Knowledge Distillation](https://arxiv.org/abs/1904.05068)
- Yoon Kim, Alexander M. Rush, [Sequence-Level Knowledge Distillation](https://arxiv.org/abs/1606.07947)
- Yizhong Wang et al., [Self-Instruct: Aligning Language Models with Self-Generated Instructions](https://arxiv.org/abs/2212.10560)
- Zuyang Gu et al., [MiniLLM: Knowledge Distillation of Large Language Models](https://arxiv.org/abs/2306.08543)
- Rishabh Agarwal et al., [GKD: Generalized Knowledge Distillation for Auto-regressive Sequence Models](https://arxiv.org/abs/2306.13649)
- Florian Tramèr et al., [Stealing Machine Learning Models via Prediction APIs](https://arxiv.org/abs/1609.02943)
- Anthropic, [Detecting and preventing distillation attacks](https://www.anthropic.com/news/detecting-and-preventing-distillation-attacks)
