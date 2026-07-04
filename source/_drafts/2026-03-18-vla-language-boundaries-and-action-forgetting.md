---
title: "语言即智能？VLA 中语言的必要性、边界与动作后训练遗忘"
date: 2026-03-18 23:20:00 +0800
categories: ["Foundation Models", "Model Mechanics"]
tags: [Embodied AI, Robotics, VLA, RT-2, OpenVLA, pi0, Gemini Robotics, Catastrophic Forgetting, Continual Learning]
author: Hyacehila
excerpt: "语言不是智能本体，但在今天的 VLA 路线里，它仍然是最便宜的任务接口、语义压缩层和 web-scale 先验迁移通道。真正需要拆开的，是语言在机器人里的角色，以及动作后训练为何会伤害、又如何保住这些由语言模型继承来的能力。"
---

> 这篇草稿要回答的，不是“机器人要不要用大模型”这种太宽的问题，而是更窄也更关键的一句：**当我们把一个强大的视觉/语言模型接上动作输出时，语言到底是在提供智能本身，还是只是在提供当下最有效的任务表征接口？**
>
> 截至 **2026 年 3 月 18 日**，我主要核对的一手材料包括 [RT-2](https://arxiv.org/abs/2307.15818)、[Diffusion Policy](https://arxiv.org/abs/2303.04137)、[ACT](https://arxiv.org/abs/2304.13705)、[Octo](https://arxiv.org/abs/2405.12213)、[OpenVLA](https://arxiv.org/abs/2406.09246)、[π0](https://arxiv.org/abs/2410.24164)、[Gemini Robotics](https://arxiv.org/abs/2503.20020)、[Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success](https://arxiv.org/abs/2502.19645)、[Knowledge Insulating Vision-Language-Action Models: Train Fast, Run Fast, Generalize Better](https://arxiv.org/abs/2505.23705)、[Pretrained Vision-Language-Action Models are Surprisingly Resistant to Forgetting in Continual Learning](https://arxiv.org/abs/2603.03818)，以及 [Let’s Talk About Language! Investigating Linguistic Diversity in Embodied AI Datasets](https://openreview.net/forum?id=wWWKPNz6GJ)。文中所有强判断都尽量绑定到论文或官方技术报告；如果是我从多篇材料中抽出来的工程判断，我会明确写成“归纳判断”。

## 研究问题与范围

这篇文章只聚焦两个问题：

1. **语言是不是智能的必要条件？**
2. **在机器人基础模型里，语言究竟是“本体能力”，还是“工程接口”？**

我的范围也故意收窄：

- 不把它写成“动物智能 / 语言哲学 / embodied cognition”的大综述；
- 只把“没有语言的生物仍具备智能”作为开篇引子；
- 主体聚焦 **2023-2026 的 VLA、action-first policy、以及动作后训练引发的遗忘问题**。

换句话说，我真正关心的是：**今天 VLA 里这么高的 L 比例，到底是不是一条正确路线；如果是，它正确到什么阶段；如果不是，问题又具体出在什么地方。**

## 一页式结论

先给我现在最核心的判断。

1. **`confirmed + 归纳判断`：语言不是智能的必要条件。** 如果把智能理解成在环境中感知、预测、规划并达成目标，那么大量动物行为，以及今天很多纯 observation-to-action 的机器人策略，都已经说明“没有语言也能有智能行为”。语言不是智能的本体，但它是人类智能里最强的可组合接口之一。
2. **`confirmed + 归纳判断`：在当前 VLA 路线里，语言最重要的价值不是直接产生动作，而是提供任务表征、开放词汇 grounding，以及 web-scale 先验迁移。** [RT-2](https://arxiv.org/abs/2307.15818)、[OpenVLA](https://arxiv.org/abs/2406.09246)、[π0](https://arxiv.org/abs/2410.24164) 和 [Gemini Robotics](https://arxiv.org/abs/2503.20020) 的共同点，不是“都很会说话”，而是都在把语言当作高层接口，把互联网规模的视觉-语言知识借给机器人控制。
3. **`confirmed`：在窄任务、封闭动作空间、强 in-domain 数据、强调低延迟和高精度控制的场景里，语言不是必需品。** [Diffusion Policy](https://arxiv.org/abs/2303.04137) 和 [ACT](https://arxiv.org/abs/2304.13705) 代表的 action-first 路线，在很多 manipulation 任务上已经很强；这类系统并不依赖自然语言作为核心中间表示。
4. **`confirmed`：但“今天 VLA 高 L 占比”依然是一个阶段性正确的工程选择。** 原因不是语言等于智能，而是机器人数据飞轮太慢：真实采集贵、在线试错慢、硬件受限、安全约束重。与之相比，文本和图像数据、预训练基础设施、推理软件栈、开源社区和评测生态都已经成熟，所以语言模型成了具身智能最便宜的先验注入器。
5. **`confirmed`：关于遗忘，至少要分清两种不同问题。** 第一种是 **动作适配时的表示遗忘**：从 VLM/VLA 继续做动作后训练时，新动作模块会破坏语言和语义知识；[Knowledge Insulating](https://arxiv.org/abs/2505.23705) 讨论的是这个。第二种是 **任务序列中的灾难性遗忘**：模型学会新任务时忘旧任务；[Pretrained VLAs are Surprisingly Resistant to Forgetting in Continual Learning](https://arxiv.org/abs/2603.03818) 讨论的是这个。
6. **`early but important`：大规模预训练 VLA 的灾难性遗忘，可能没有过去小型从零训练 BC policy 那么严重。** 至少在 [2026 年 3 月 4 日提交的 Continual VLA 论文](https://arxiv.org/abs/2603.03818) 中，作者发现简单 `experience replay` 在 VLAs 上就能表现得非常好，甚至在小 buffer 下实现零遗忘。
7. **`speculative but plausible`：长期更可能收敛到“语言 / 空间推理 / 低层控制”分层的模块化架构，而不是把所有东西都塞进一个越来越像 LLM 的单体模型里。** [Gemini Robotics-ER](https://deepmind.google/blog/gemini-robotics-brings-ai-into-the-physical-world/) 和 [π0](https://arxiv.org/abs/2410.24164) 都在往这条路上靠：前者显式把 embodied reasoning 与控制解耦，后者也展示了“高层 VLM 指令 + 低层执行”的价值。

## 检索策略与证据标准

这篇稿子的证据标准尽量保持一致：

- **核心技术判断**：优先使用论文、arXiv 技术报告、官方项目页；
- **产品定位与架构 framing**：补充使用官方技术博客；
- **不使用二手解读来支撑关键结论**；
- 对每条判断标记三种信号层级：
  - `confirmed`：有论文或官方技术报告直接支持；
  - `early`：已有明确技术报告，但仍属于早期结论；
  - `speculative`：基于多篇材料的归纳，不假装已被单篇论文证明。

另外还有一个重要边界：**“语言重要”不等于“语言理解足够强”**。很多 VLA 其实只是在使用一个窄而模板化的任务语言通道，这和通用自然语言能力不是一回事。

## 先把命题拆开：语言不是智能本体，但它是今天最强的任务接口

“语言即智能”这句话最大的问题，在于它把三件不同的事混到了一起：

1. **智能是否必须借由语言出现；**
2. **语言是否是最好的高层任务表示；**
3. **语言模型是否是当前最便宜的先验来源。**

这三件事的答案并不相同。

如果只问第一句，我的答案是：**不是。**  
从比较认知的直观事实，到今天大量非语言控制策略，都说明“实现目标导向行为”并不需要先拥有像人类那样的自然语言系统。机器人里也一样：一个系统可以完全依赖视觉、状态和动作序列，就学会稳定、精确、闭环的操作行为。

但如果问第二句和第三句，答案就变了。  
在今天的机器人基础模型里，语言有三个极其现实的工程价值：

- 它是最统一的 **任务 API**；
- 它是最便宜的 **语义压缩层**；
- 它是接入互联网规模先验的 **迁移通道**。

所以更准确的说法不是“语言即智能”，而是：

> **在当前具身智能阶段，语言不是智能的本体，却是最强、最便宜、最现成的高层任务表示层。**

## 为什么 VLA 现在离不开语言

### 1. 语言给了机器人一个跨任务、跨平台、跨数据源的统一接口

[RT-2](https://arxiv.org/abs/2307.15818) 是这条路线最清楚的起点之一。论文在 **2023 年 7 月 28 日** 提交，核心做法非常直接：把动作也表达成 token，让机器人轨迹数据和互联网规模视觉-语言任务进入同一训练格式。作者明确说，他们的目标是让单个 end-to-end 模型既能学会“从观测到动作”，又能享受 web-scale vision-language pretraining 的好处。

这一招的意义不只是在于“能把动作写成 token”，而在于它把机器人训练问题重新改写成了一个熟悉的问题：**如何把高层语义和低层行为接到同一个序列模型里。**

论文给出的收益也很明确：

- 能更好泛化到新物体；
- 能理解训练中没见过的命令；
- 能出现一些来自互联网训练的语义推理能力。

也就是说，语言在这里不是负责转动电机，而是负责把任务语义装进模型的同一套表示空间里。

### 2. 语言是把互联网规模先验借给机器人最快的方式

[OpenVLA](https://arxiv.org/abs/2406.09246) 在 **2024 年 6 月 13 日** 提交、并于 **2024 年 9 月 5 日** 更新到 v3。它的 abstract 直接把这件事说透了：`Large policies pretrained on a combination of Internet-scale vision-language data and diverse robot demonstrations ... rather than training new behaviors from scratch`.

更具体地说，OpenVLA：

- 基于 `Llama 2` 语言模型骨干；
- 结合 `DINOv2 + SigLIP` 视觉编码器；
- 在 **970k** 条真实机器人示范上训练；
- 在 29 个任务和多种 robot embodiment 上，以 **7x 更少参数** 超过了更大的闭源模型 `RT-2-X`。

我更看重的不是某个 leaderboard 数字，而是这条路线释放出来的信号：**机器人并没有等到自己的数据飞轮成熟之后再做 foundation model，而是先借 VLM/LLM 的成熟飞轮，把语义能力“外挂”进来。**

### 3. 语言是最便宜的人类监督接口

[π0](https://arxiv.org/abs/2410.24164) 在 **2024 年 10 月 31 日** 首次提交，并在 **2026 年 1 月 8 日** 更新到 v4。它使用了预训练 VLM 骨干，再加一个连续动作 `action expert`，目标是兼顾语义知识和精细控制。

π0 这篇最值得注意的一点，不只是“它是个强模型”，而是论文明确把语言放在两个位置：

- 直接来自人的语言指令；
- 来自一个更高层 VLM policy 的中间语言指令。

这意味着语言在 π0 里已经不只是“任务描述文本”，而是 **高层 teacher signal**。模型不是只能吃最终任务名，还能吃“下一步该拿什么、放到哪里”的高层子任务语义。

这很关键，因为在人类给机器人监督时：

- 直接给低层动作很贵；
- 大规模稳定 teleop 很慢；
- 安全在线试错更慢；
- 但给一句高层指令，或者让另一个模型生成高层子任务，成本低得多。

从这个角度看，语言更像一种 **监督压缩格式**。

### 4. 长时程和安全约束也天然偏好语言/代码层接口

Google DeepMind 在 **2025 年 3 月 12 日** 先发布了 [Gemini Robotics / Gemini Robotics-ER 官方博客](https://deepmind.google/blog/gemini-robotics-brings-ai-into-the-physical-world/)，随后在 **2025 年 3 月 25 日** 提交技术报告 [Gemini Robotics: Bringing AI into the Physical World](https://arxiv.org/abs/2503.20020)。

这套系统最重要的地方，不只是“又一个 VLA”，而是它把结构分得更清楚了：

- `Gemini Robotics`：直接控制机器人；
- `Gemini Robotics-ER`：更偏 embodied reasoning，负责空间理解、状态估计、规划和代码生成，并且能对接已有低层控制器。

官方博客甚至明确写到，`Gemini Robotics-ER` 可以和 embodiment-specific 的 low-level safety-critical controllers 对接。这其实已经是一种架构表态：**语言与推理层适合做高层决策与约束表达，低层控制则应该保留更专门、更快、更安全的执行层。**

所以，今天 VLA 高 L 占比为什么“看起来正确”？因为在这个阶段，语言确实能同时承担：

- 任务 specification；
- 开放词汇泛化；
- 人机交互；
- 安全约束与高层计划表达；
- 跨 embodiment 的统一中间层。

## 但别把它神秘化：很多 VLA 的“语言”其实相当模板化

如果上面那几段很容易让人得出“语言就是关键智能”的结论，那么 [Let’s Talk About Language! Investigating Linguistic Diversity in Embodied AI Datasets](https://openreview.net/forum?id=wWWKPNz6GJ) 刚好提供了一个重要反证。

这篇工作发表于 **2025 年 5 月 12 日** 的 Safe-VLMs@ICRA workshop。它最有价值的地方，不是提出了一个更强模型，而是提醒我们：**Embodied AI 数据集里的语言，可能比我们想象中贫瘠得多。**

作者的主要发现包括：

- 很多 EAI 数据集严重依赖重复句式；
- 常见数据集里很少出现否定、条件句、循环结构等复杂语言现象；
- 在 LIBERO-10 上做一个 feature-guided paraphrasing case study 后，OpenVLA 在原始指令上的平均成功率约为 **0.66**，换成改写后的 paraphrase 只剩 **0.3168**；
- 论文在摘要里直接指出：**minor syntactic shifts can cut OpenVLA’s success rate by over 50%**。

这件事的意义非常大。

它说明今天很多 VLA 从语言中获得的，未必是“成熟自然语言理解能力”，而更可能是：

- 稳定的任务标签；
- 模板化动作描述；
- 有限范围的开放词汇 grounding；
- 与 web-pretraining 对齐的语义入口。

也就是说，**VLA 很吃语言，不等于它真的已经掌握了强语言鲁棒性。**  
这也是为什么我更愿意把语言在 VLA 中的角色叫做“任务接口”和“语义先验通道”，而不是直接叫“智能本身”。

## 什么时候语言其实不是必要条件

如果只看 manipulation，本来就存在一整条不以自然语言为核心的强路线。

[Diffusion Policy](https://arxiv.org/abs/2303.04137) 在 **2023 年 3 月 7 日** 提交，论文把机器人策略直接建模成条件扩散过程，在 12 个任务上平均超过已有 SOTA **46.9%**。  
[ACT](https://arxiv.org/abs/2304.13705) 在 **2023 年 4 月 23 日** 提交，用 `Action Chunking with Transformers` 直接学动作序列，在多个精细双臂任务上只用 10 分钟示范数据就能做出 80-90% 成功率。

这两条路线的共同点是：

- 任务空间更窄；
- 行为分布更可控；
- 输入输出更偏 observation-to-action；
- 语言不是核心瓶颈。

甚至在 generalist policy 这边，语言也不是唯一接口。[Octo](https://arxiv.org/abs/2405.12213) 在 **2024 年 5 月 20 日** 提交时就明确支持 **language commands 或 goal images** 两种任务方式。这一点很关键：即便是大规模通用策略，也不一定要把自然语言当唯一高层表示。

所以我对“在纯粹只需要 action 的 model 中，语言其实是不必要的”这句话，基本是同意的，但要补一句条件：

> **只在任务边界足够稳定、目标定义足够明确、泛化需求没有开放到自然语言那种程度时，这句话才成立。**

更具体地说，下面这些场景里，语言通常不是决定性因素：

- 封闭物体集、封闭动作空间的重复操作；
- 高精度、低延迟、接触丰富的局部控制；
- 只追求单一任务成功率、不强调开放词汇指令；
- 已经有足够多本体机器人示范数据；
- 不需要频繁切换任务、不需要与人复杂交互。

这也是为什么很多人第一次看到 VLA 会觉得“是不是把语言用过头了”：因为对一批局部控制问题来说，确实如此。

## VLA vs action-only vs modular reasoning-action

下面这张表，是我目前最想保留下来的结构化判断。

| 路线 | 代表工作 | 主要任务接口 | 语言扮演什么角色 | 最强场景 | 主要短板 |
| --- | --- | --- | --- | --- | --- |
| `action-first policy` | [Diffusion Policy](https://arxiv.org/abs/2303.04137), [ACT](https://arxiv.org/abs/2304.13705) | 视觉 / 状态 / 动作序列 | 可有可无，通常不是核心 | 窄任务、高精度、强 in-domain 数据、低延迟控制 | 开放词汇泛化弱，跨任务统一接口差，人机交互弱 |
| `generalist VLA` | [RT-2](https://arxiv.org/abs/2307.15818), [OpenVLA](https://arxiv.org/abs/2406.09246) | 图像 + 自然语言 + 动作 token | 任务描述、语义 grounding、互联网知识迁移 | 多任务、开放词汇、跨平台适配、快速借用 web prior | 语言鲁棒性未必真强，动作 latency 与连续控制受限 |
| `VLM backbone + action expert` | [π0](https://arxiv.org/abs/2410.24164), [π0.5](https://www.physicalintelligence.company/download/pi05.pdf) | 图像 + 语言 + 连续动作 expert | 高层任务表达、teacher signal、open-world semantics | 既要语义泛化又要连续控制的复杂 manipulation | 动作专家可能伤害语义知识，训练 recipe 更脆弱 |
| `modular reasoning-action split` | [Gemini Robotics](https://arxiv.org/abs/2503.20020), [Gemini Robotics-ER](https://deepmind.google/blog/gemini-robotics-brings-ai-into-the-physical-world/) | 高层语言/推理 + 低层控制器 | 规划、约束、空间推理、代码生成 | 长时程、多阶段、强调安全与人机交互的任务 | 系统复杂度更高，接口设计和延迟预算更难 |

如果只让我给一句话总结：

> **action-first 路线解决“怎么做得准”，VLA 路线解决“怎么做得通用”，而 modular 路线试图把“会想”和“会做”拆开，减少彼此互相伤害。**

## 为什么我仍然认为“今天的高 L 占比”是阶段性正确的

这部分是我的**归纳判断**，不是某一篇论文自己直接证明的结论。

我之所以认为“现在 VLA 中这么高的语言模型占比”在短中期依然是对的，不是因为语言天生更接近智能本体，而是因为它最符合当前具身智能的工程现实：

### 1. 数据飞轮不对称

文本和图像：

- 数据量大；
- 采集便宜；
- 标注成本相对可控；
- 训练基础设施成熟；
- 社区和评测生态完整。

机器人数据：

- 采集慢；
- 硬件贵；
- 安全约束强；
- 场景分布极碎；
- online exploration 成本极高。

在这种不对称下，先把语言/视觉基础模型当 backbone，是最自然的策略。  
**不是因为机器人“本质上应该说话”，而是因为机器人自己的数据飞轮还没转起来。**

### 2. 语言是异构数据最容易对齐的公共层

[π0.5](https://www.physicalintelligence.company/download/pi05.pdf) 在 **2025 年 4 月 22 日** 发布时就把这件事讲得很明确：它通过 co-training 混合多种数据源，包括其他机器人、高层子任务预测、口头语言指令、web data 等，去获得 open-world generalization。

这背后的共同表示层是什么？不是扭矩，也不是接触力，而是更接近 **语义任务层** 的中间表示。

### 3. 语言让高层监督更廉价

人类最擅长给的是：

- 任务描述；
- 约束；
- 意图；
- 错误反馈；
- 子目标。

人类最不擅长大规模给的是：

- 全时序低层动作；
- 长时间高质量 teleop；
- 大规模精细失败恢复轨迹。

所以从“监督经济学”看，语言自然会被推到一个非常中心的位置。

### 4. 但这不应该被误读成终局架构

一旦机器人数据飞轮变快、低层控制更强、world model 和 proprioceptive representation 更成熟，语言在系统中的位置很可能会后移：

- 从“主骨干”退到“高层接口”；
- 从“统一中间表示”退到“规划与约束层”；
- 从“动作生成主通道”退到“任务设定和异常恢复层”。

所以我对“VLA 现在高 L 占比是否正确”的短答是：

> **短中期是，长期大概率不是终局。**

## 先分清两种遗忘：很多讨论其实在说不同问题

我觉得这部分最容易被混淆。

今天大家说“动作后训练导致灾难性遗忘”，至少可能是在说两件不同的事：

### A. 表示遗忘：动作适配把语言/语义知识冲掉了

这是 [Knowledge Insulating Vision-Language-Action Models](https://arxiv.org/abs/2505.23705) 讨论的核心问题。

论文在 **2025 年 5 月 29 日** 提交，问题设定不是“连续学很多任务”，而是：  
**当你把一个预训练 VLM 改造成带 continuous action expert 的 VLA 时，新加的动作模块会不会破坏原来 VLM 的语义知识？**

作者的答案是：**会，而且 naive 做法很容易出问题。**

他们指出，最近很多高性能 VLA 会加入 diffusion / flow matching action expert 来支持连续控制，但这些新模块通常带来大量随机初始化参数；如果梯度直接回写到 backbone，就可能伤害原来从 web-scale VLM 训练里得到的知识。

### B. 序列遗忘：学新任务时把旧任务忘了

这是 [Pretrained Vision-Language-Action Models are Surprisingly Resistant to Forgetting in Continual Learning](https://arxiv.org/abs/2603.03818) 讨论的核心问题。

这篇论文在 **2026 年 3 月 4 日** 提交，关注的是 continual learning：  
模型需要随着时间学新技能，而不灾难性忘掉旧技能。

这里的重点不再是“动作 expert 会不会破坏语义 backbone”，而是：

- 顺序学习时会不会覆盖旧任务；
- replay 是否足够；
- 预训练是否改变 forgetting dynamics。

把这两件事混在一起，会导致很奇怪的讨论：  
有人说“VLA 遗忘很严重”，指的是动作模块伤到语义；  
有人说“VLA 遗忘没那么严重”，指的是 continual learning 下 replay 已经很有效。

这两句话其实都可能是对的，只是它们说的不是同一个问题。

## 动作后训练为什么会伤害语言与世界知识

现在看，至少有四个机制会共同造成这个问题。

### 1. 新动作模块把随机梯度打回了预训练 backbone

[Knowledge Insulating](https://arxiv.org/abs/2505.23705) 的核心发现之一，就是 naive 地把 continuous `action expert` 接到 VLM backbone 上，会显著伤害训练速度与知识迁移。论文明确写到，这类 expert 如果没有隔离好，可能会和预训练 VLM 权重发生不利交互。

这本质上是一个 **梯度干扰** 问题。

### 2. 控制目标和语言建模目标天然不完全一致

语言建模关心的是：

- 语义压缩；
- next-token prediction；
- 跨模态语义对齐；
- 长上下文统计结构。

机器人控制关心的是：

- 连续动作；
- 闭环反馈；
- 低延迟；
- 物理约束；
- 误差累积。

两者并不天然共线。  
所以当模型被大规模拉向动作目标时，原有语言/视觉知识空间会出现变形，并不意外。

### 3. 机器人数据比 web 数据更窄，也更容易诱导 shortcut

[Let’s Talk About Language](https://openreview.net/forum?id=wWWKPNz6GJ) 已经提醒我们，很多 embodied 数据集的语言很模板化。如果动作后训练阶段又主要依赖这些模板数据，模型就更可能把“语言理解”退化成对表面模式的记忆，而不是保留更广泛的语言鲁棒性。

这也解释了为什么 paraphrase 会打得这么狠：  
模型可能记住了任务模板，却没有真的学会足够稳健的语言抽象。

### 4. 为了实时控制而做的工程近似，往往优先保动作、不优先保语义

这是一个很现实的问题。  
机器人上线时，大家最先优化的常常是：

- 速度；
- 延迟；
- 控制稳定性；
- 高频动作质量。

如果一个 recipe 让动作更快、更稳、成功率更高，它就更容易被采用；但这不代表它自动保住了语言理解、开放词汇泛化或 web knowledge transfer。

所以“动作后训练伤害语言知识”并不神秘，它本质上是 **目标错配 + 梯度干扰 + 数据分布变窄 + 工程优化偏置** 的叠加结果。

## 证据对照：今天到底有哪些现象是被直接看到的

| 现象 | 主要证据 | 我对它的判断 |
| --- | --- | --- |
| 语言/视觉预训练能显著帮助机器人泛化 | [RT-2](https://arxiv.org/abs/2307.15818), [OpenVLA](https://arxiv.org/abs/2406.09246), [π0](https://arxiv.org/abs/2410.24164), [Gemini Robotics](https://arxiv.org/abs/2503.20020) | `confirmed` |
| 很多 embodied 数据集的语言表达很模板化 | [Let’s Talk About Language](https://openreview.net/forum?id=wWWKPNz6GJ) | `confirmed` |
| 小的句法变化就会显著打击 VLA 成功率 | [Let’s Talk About Language](https://openreview.net/forum?id=wWWKPNz6GJ) 在 LIBERO-10 上的 paraphrase case study | `confirmed` |
| naive 动作 expert 会伤害知识迁移和语言跟随 | [Knowledge Insulating](https://arxiv.org/abs/2505.23705) | `confirmed` |
| 预训练大 VLA 在 continual learning 下比小 BC policy 更抗遗忘 | [Pretrained VLAs are Surprisingly Resistant to Forgetting in Continual Learning](https://arxiv.org/abs/2603.03818) | `early` |
| 未来更优结构可能是 reasoning / action 分层 | [Gemini Robotics](https://arxiv.org/abs/2503.20020), [Gemini Robotics-ER 官方博客](https://deepmind.google/blog/gemini-robotics-brings-ai-into-the-physical-world/) | `speculative but well-motivated` |

## 怎样缓解动作后训练导致的遗忘

这部分最值得写成一张表，因为不同方法其实在解决不同层面的问题。

| 缓解路线 | 核心机制 | 主要证据锚点 | 更适合解决哪类问题 | 代价与边界 |
| --- | --- | --- | --- | --- |
| `experience replay` | 保留小规模旧任务数据，训练新任务时混合回放 | [Continual VLA](https://arxiv.org/abs/2603.03818) 发现小 replay 在 VLAs 上有时就能实现零遗忘 | 顺序任务学习中的灾难性遗忘 | 依赖 buffer 覆盖度，不能直接解决动作 expert 伤语义的问题 |
| `backbone insulation / stop-gradient` | 阻断 action expert 对预训练 backbone 的破坏性梯度 | [Knowledge Insulating](https://arxiv.org/abs/2505.23705) | 动作适配时的表示遗忘 | 可能限制低层动作模块对 backbone 的联动优化 |
| `VLM data co-training` | 训练时持续混入非动作 VLM 数据和规划数据 | [Knowledge Insulating](https://arxiv.org/abs/2505.23705) | 保语言跟随、保语义迁移、保 OOD generalization | 训练更贵，recipe 更复杂 |
| `adapter isolation / LoRA-style adaptation` | 把更新限制在小模块，减少全骨干漂移 | [OpenVLA](https://arxiv.org/abs/2406.09246) 强调可在 consumer GPU 上通过 low-rank adaptation 做高效微调 | 平台迁移或下游适配时控制破坏范围 | 不是直接的 anti-forgetting 证明，更像风险控制 |
| `动作表示优化` | 并行解码、action chunking、连续动作表示、简单回归目标 | [OpenVLA-OFT](https://arxiv.org/abs/2502.19645) | 在特定平台上又快又稳地做动作适配 | 提升动作侧效率，不自动保证语义知识保留 |
| `reasoning/action 分层` | 高层模型负责计划、空间推理和约束，低层控制器负责执行 | [Gemini Robotics-ER](https://deepmind.google/blog/gemini-robotics-brings-ai-into-the-physical-world/), [π0](https://arxiv.org/abs/2410.24164) | 长时程、多阶段任务，同时减少高层语义和低层执行互相干扰 | 系统复杂度更高，接口设计变成新难题 |

### 我最认可的缓解思路：先别把所有问题交给一个更新通道

如果让我用一句话概括这些方法，我会说：

> **最有效的缓解方式，不是指望“更聪明的 fine-tuning”自动解决一切，而是主动减少高层语义知识与低层动作适配共享同一组脆弱更新通道。**

这也是为什么我会把方法分成三层来理解：

1. **训练层**：`replay`、`co-training`；
2. **参数层**：`LoRA / adapters / stop-gradient / insulation`；
3. **系统层**：`reasoning-action split`。

其中：

- `replay` 最适合对付顺序学习里的 forgetting；
- `insulation` 最适合对付动作 expert 破坏 backbone；
- `split architecture` 最适合从根上减少“会想”和“会做”互相污染。

## OpenVLA-OFT 说明了另一件重要的事：很多提升并不来自“更强语言”，而来自“更好的动作接口”

[Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success](https://arxiv.org/abs/2502.19645) 在 **2025 年 2 月 27 日** 提交，是我觉得很容易被误读的一篇。

它的 headline result 很亮眼：

- 把 OpenVLA 在 LIBERO 四个 task suite 上的平均成功率从 **76.5%** 提到 **97.1%**；
- 同时把 action generation throughput 提升 **26x**；
- 在真实世界评测里，超过了默认 recipe 微调的 `π0`、`RDT-1B`，也超过了从零训练的 [Diffusion Policy](https://arxiv.org/abs/2303.04137) 和 [ACT](https://arxiv.org/abs/2304.13705)。

但我从这篇里读出来的关键点，不是“VLA 必然胜过 action-only”，而是：

1. 一旦任务落到特定平台，**action representation 与 decoding strategy** 往往比“再加一点语言能力”更重要；
2. 很多所谓“VLA 性能不够”其实是 **动作接口设计不对**；
3. 这也反过来说明：语言的主要价值还是高层任务表示，而不是替代精心设计的动作层。

所以 OpenVLA-OFT 很适合被读成一个提醒：

> **当任务转向具体机器人和具体动作频率时，系统瓶颈会从“懂不懂语言”迅速转向“动作表示、解码速度和控制稳定性”。**

## 近一年这条线真正发生了什么变化

如果只看 **2025 年 3 月到 2026 年 3 月** 这一年，我觉得这个领域不是在简单重复“更大 VLA”，而是在往四个更具体的问题收敛：

| 时间 | 代表工作 | 真正推进的问题 |
| --- | --- | --- |
| 2025-02-27 | [OpenVLA-OFT](https://arxiv.org/abs/2502.19645) | 动作接口怎么做得又快又稳 |
| 2025-03-12 / 2025-03-25 | [Gemini Robotics 官方博客](https://deepmind.google/blog/gemini-robotics-brings-ai-into-the-physical-world/), [Gemini Robotics 技术报告](https://arxiv.org/abs/2503.20020) | 高层 embodied reasoning 与低层控制如何分工 |
| 2025-05-12 | [Let’s Talk About Language](https://openreview.net/forum?id=wWWKPNz6GJ) | 今天的 VLA 语言到底有多真、又有多脆 |
| 2025-05-29 | [Knowledge Insulating](https://arxiv.org/abs/2505.23705) | 动作后训练为何伤知识迁移，如何保护 backbone |
| 2026-03-04 | [Continual VLA Forgetting](https://arxiv.org/abs/2603.03818) | 大预训练 VLA 在顺序学习里是否仍会严重遗忘 |

这组变化很说明问题：  
**领域的注意力已经从“要不要把 LLM 接上机器人”转向“接上之后哪里最脆、怎么分层、怎么保住知识、怎么测语言鲁棒性”。**

## 3-5 年综述与里程碑：这条路线是怎么走到今天的

如果把时间拉长到 **2023-2026**，我会把主线读成下面这样：

### 2023：action-first 路线很强，但语义接口还不统一

- [Diffusion Policy](https://arxiv.org/abs/2303.04137) 代表“强动作生成器”路线；
- [ACT](https://arxiv.org/abs/2304.13705) 代表“低成本高精度 imitation”路线。

这两类工作都证明了：**没有自然语言，机器人照样可以很强。**

### 2023 下半年：RT-2 把语言和动作放到同一 token 空间

[RT-2](https://arxiv.org/abs/2307.15818) 的真正历史意义，在于它把“机器人控制”重新连接到互联网规模 vision-language pretraining 上。  
这是 VLA 真正成型的标志。

### 2024：通用策略与开源基础模型开始成形

- [Octo](https://arxiv.org/abs/2405.12213) 让 generalist policy 更开源、更易适配；
- [OpenVLA](https://arxiv.org/abs/2406.09246) 让开源 VLA 真正站到舞台中央；
- [π0](https://arxiv.org/abs/2410.24164) 则把 `VLM backbone + action expert` 这条线推得更深。

### 2025：大家开始重新审视“语言到底贡献了什么”

这一年很关键，因为它不只是继续堆更大的 VLA，而是出现了三种反思：

1. [OpenVLA-OFT](https://arxiv.org/abs/2502.19645)：动作接口怎么优化；
2. [Let’s Talk About Language](https://openreview.net/forum?id=wWWKPNz6GJ)：语言到底有多脆；
3. [Knowledge Insulating](https://arxiv.org/abs/2505.23705)：动作适配怎么不毁掉语义知识。

### 2025-2026：分层与持续学习开始进入主线

[Gemini Robotics](https://arxiv.org/abs/2503.20020) 和 [Continual VLA Forgetting](https://arxiv.org/abs/2603.03818) 把两个长期问题摆到台前：

- 高层 reasoning 和低层 control 是否应该解耦；
- VLA 能否像真正的 agent 一样持续学新技能而不把旧技能全忘掉。

所以，这条线的变化不是“语言越来越重要”，而是：

> **语言从“被动挂在机器人外面的 prompt”变成了系统中的高层接口；与此同时，社区也开始意识到这个接口既有巨大价值，也有明显边界。**

## 我的核心判断：语言模型在机器人里最强的，不是动作，而是任务语言表征

把所有材料合在一起，我现在最确定的一件事是：

> **语言模型在机器人里最具优势的地方，不是直接生成连续动作，而是把任务、约束、对象关系、用户意图和开放词汇语义，组织成一个高可复用的表征空间。**

这也是为什么我会把“语言即智能”改写成下面这句话：

> **语言不是智能本体，但在今天它是最好的任务语义压缩器。**

它强在：

- 任务 specification；
- 约束表达；
- 开放词汇映射；
- 高层计划与中间子目标；
- 跨平台对齐；
- 接入互联网规模先验。

它不强在：

- 高频闭环控制；
- 接触丰富的连续动作；
- 毫秒级 latency budget；
- 纯本体动力学与力控制。

所以如果一个系统只需要后者，语言当然可以退场；  
但如果一个系统需要前者，语言现在仍然几乎没有便宜替代品。

## 争议、局限与开放问题

这条线现在至少还有五个硬问题没有解决。

### 1. 语言鲁棒性 benchmark 还不够强

[Let’s Talk About Language](https://openreview.net/forum?id=wWWKPNz6GJ) 已经证明，paraphrase stress test 会打穿很多现有 VLA。但这类评测还没有成为主流 benchmark 的标准组成部分。

### 2. 还缺少“语言鲁棒性 + 动作适配 + 持续学习”的联合评测

今天的评测往往拆开做：

- 语言泛化单测；
- manipulation success 单测；
- continual learning 单测。

但真实系统里，这三件事是缠在一起的。

### 3. `replay` 的好消息还属于早期结论

[Continual VLA Forgetting](https://arxiv.org/abs/2603.03818) 很重要，但它还是一篇 **2026 年 3 月 4 日** 的新论文。  
“大预训练 VLA 已经不太会灾难性遗忘”这句话，今天还说早了。更稳妥的说法是：**预训练显著改变了 forgetting dynamics，simple replay 可能比我们预期更有效。**

### 4. 高 L 占比是否会拖慢具身自己的原生表示学习

这也是我心里最大的开放问题之一。  
如果机器人长期过度依赖语言与 web priors，会不会让系统更难学出真正适合物理交互的中间表示？目前没有定论。

### 5. 具身智能自己的飞轮还远没有语言/视觉那样成熟

这一点不是单篇论文结论，而是我从整个领域现状得到的工程判断：

- 硬件迭代慢；
- 场景分布碎；
- 安全试错贵；
- 部署回流数据难；
- 线上训练空间有限。

这也是为什么短中期里，VLA 仍会高度借力语言模型，而不是完全靠机器人自生长出一套独立于语言的通用智能栈。

## 可行动研究想法 / 假设候选

如果把这篇文章的判断继续推进成研究问题，我觉得至少有四个方向值得做。

### 1. 做一个真正针对“动作后训练遗忘”的联合 benchmark

同一条评测流水线上同时测：

- 任务成功率；
- paraphrase robustness；
- OOD object grounding；
- continual adaptation 后的旧任务保持率。

今天大家还太容易只看第一项。

### 2. 把“高层语义”和“低层动作”明确拆成不同更新通道

不是所有 fine-tuning 都该更新同一组参数。  
可以系统比较：

- full finetune；
- LoRA / adapter；
- stop-gradient insulation；
- dual-backbone；
- reasoning model + controller 分层。

### 3. 研究何时该用语言，何时该用 goal image、subtask graph 或 latent action

[Octo](https://arxiv.org/abs/2405.12213) 已经提醒我们，语言不是唯一任务接口。  
未来很可能需要做一种 **interface selection**：  
任务越开放、越人机协作、越依赖约束表达，就越适合语言；  
任务越局部、越重复、越讲究实时性，就越适合非语言接口。

### 4. 让 replay 不只回放机器人数据，也回放语义能力

[Knowledge Insulating](https://arxiv.org/abs/2505.23705) 给出的启发非常直接：  
如果要保住 VLM 的知识，就不能只盯着机器人动作数据。  
一个更系统的方向，是在 continual adaptation 里同时回放：

- 旧机器人任务；
- VLM 数据；
- 高层规划数据；
- paraphrased instruction 数据。

这样才能同时保动作、保语言、保开放词汇泛化。

## 持续跟踪清单

接下来我会重点盯下面这些问题：

1. 是否出现专门针对 **VLA 语言鲁棒性** 的标准 benchmark；
2. [Continual VLA](https://arxiv.org/abs/2603.03818) 之后，是否有更多工作验证 replay 在不同模型族上的有效性；
3. [Knowledge Insulating](https://arxiv.org/abs/2505.23705) 之后，是否出现更系统的 `insulated action expert` 或 `dual-path adaptation` 方案；
4. Gemini Robotics 这类 **reasoning / control 分层** 是否成为更主流的工业架构；
5. 未来一代 open-world robot model 是否开始减少“语言主干 + 小动作头”的单体范式，转向更强的模块化。

## 推荐阅读路径

如果要按“先建立主线，再读争议”的顺序读，我会推荐：

1. [RT-2](https://arxiv.org/abs/2307.15818)：先看 VLA 为什么把语言和动作塞进同一模型；
2. [OpenVLA](https://arxiv.org/abs/2406.09246)：再看开源 VLA 怎么把这件事做实；
3. [π0](https://arxiv.org/abs/2410.24164)：看连续动作 expert 如何接上预训练 VLM；
4. [Gemini Robotics](https://arxiv.org/abs/2503.20020)：看 reasoning 与 control 如何开始分层；
5. [OpenVLA-OFT](https://arxiv.org/abs/2502.19645)：看动作接口优化如何决定下游表现；
6. [Let’s Talk About Language](https://openreview.net/forum?id=wWWKPNz6GJ)：看现有 VLA 的语言鲁棒性边界；
7. [Knowledge Insulating](https://arxiv.org/abs/2505.23705)：看动作后训练如何伤害语义知识；
8. [Continual VLA Forgetting](https://arxiv.org/abs/2603.03818)：看大预训练 VLA 是否真的更抗遗忘。

## 参考文献与官方入口

- [RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control](https://arxiv.org/abs/2307.15818)
- [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137)
- [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (ACT)](https://arxiv.org/abs/2304.13705)
- [Octo: An Open-Source Generalist Robot Policy](https://arxiv.org/abs/2405.12213)
- [OpenVLA: An Open-Source Vision-Language-Action Model](https://arxiv.org/abs/2406.09246)
- [π0: A Vision-Language-Action Flow Model for General Robot Control](https://arxiv.org/abs/2410.24164)
- [π0.5: a Vision-Language-Action Model with Open-World Generalization](https://www.physicalintelligence.company/download/pi05.pdf)
- [Open Sourcing π0](https://www.pi.website/blog/openpi)
- [Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success](https://arxiv.org/abs/2502.19645)
- [OpenVLA-OFT project page](https://openvla-oft.github.io/)
- [Gemini Robotics: Bringing AI into the Physical World](https://arxiv.org/abs/2503.20020)
- [Gemini Robotics brings AI into the physical world](https://deepmind.google/blog/gemini-robotics-brings-ai-into-the-physical-world/)
- [How Google built its Gemini robotics models](https://blog.google/products-and-platforms/products/gemini/how-we-built-gemini-robotics/)
- [Knowledge Insulating Vision-Language-Action Models: Train Fast, Run Fast, Generalize Better](https://arxiv.org/abs/2505.23705)
- [Knowledge Insulating PDF / official entry](https://www.physicalintelligence.company/download/pi05_KI.pdf)
- [Pretrained Vision-Language-Action Models are Surprisingly Resistant to Forgetting in Continual Learning](https://arxiv.org/abs/2603.03818)
- [Continual VLA project page](https://ut-austin-rpl.github.io/continual-vla)
- [Let’s Talk About Language! Investigating Linguistic Diversity in Embodied AI Datasets](https://openreview.net/forum?id=wWWKPNz6GJ)

## 结语

如果只让我用一句话总结整篇文章，我会写：

> **语言不是智能的必要条件，但在今天，语言仍然是具身智能最划算的高层接口；真正的挑战不是“要不要语言”，而是“如何让语言、空间推理和低层动作各司其职，而不是在一次动作后训练里互相伤害”。**

这也是我现在对 VLA 的最稳定判断：  
它不是终局，但它是当前这个阶段里最现实、也最有生产力的一座桥。
