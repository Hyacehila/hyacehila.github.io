---
layout: blog-post
title: "Era of Experience：当 AI 的核心问题从「读更多」变成「做更多」"
date: 2026-03-16 20:00:00 +0800
categories: [AI]
tags: [RL, Agents, Reward Design]
author: Hyacehila
excerpt: "Silver 和 Sutton 宣告的「经验时代」，真正改变的不是 RL 又火了，而是 AI 研究的核心问题本身在迁移——从「如何从人类数据中学更多」变成「如何让 agent 在世界中行动并从后果中学习」。而这条路上最关键的瓶颈，是奖励设计。"
featured: false
math: false
---

# Era of Experience ：当 AI 的核心问题从「读更多」变成「做更多」

> "The next generation of AI systems will be shaped primarily by experience rather than human data."
> — David Silver & Richard Sutton, *Welcome to the Era of Experience* (2025)

2025 年 4 月，David Silver 和 Richard Sutton 发表了一篇名为 [Welcome to the Era of Experience](https://storage.googleapis.com/deepmind-media/Era-of-Experience%20/The%20Era%20of%20Experience%20Paper.pdf) 的短文。这两个名字放在一起，本身就是一个信号——AlphaGo 的缔造者和强化学习的教父，联合对 AI 的下一个阶段做出了方向性判断。

**它在说的，不是某种技术又火了，而是 AI 研究的核心问题本身正在发生迁移**——从如何从人类已有的数据中学到更多，变成如何让 agent 在世界中行动、获取反馈、积累经验并持续改进。

这篇博客和Shunyu Yao的The Second Half时间近似，讨论的问题的核心技术也有一点类似，但是他并不侧重于RL Genenralization相关的问题，而是侧重于讨论再下一个Era，我们应该去做什么，什么问题是重要的。

而我自己读完一圈相关资料后，最大的判断是：在这场迁移中，**奖励设计（reward design）是最关键的瓶颈**。其他所有问题——credit assignment、采样效率、安全、评测——都依赖一个前提：我们能不能定义清楚"什么是好的结果"。

这篇文章不是对原文的综述，而是我基于它的思考。我想回答的核心问题是：**在Era of Experience，AI 研究中最关键的问题应该变成什么？**

## 从人类数据时代到经验时代：到底在说什么

先简单交代 Silver 和 Sutton 的核心论证。

过去几年 AI 的主线很清晰：用海量人类生成数据（网页、书籍、代码、论文）做预训练，再用 SFT、偏好数据、RLHF 把模型调成更好的助手。这条路极其成功，造出了今天的大语言模型。

但 Silver 和 Sutton 指出：**这条路的增量收益正在变小**。不是因为模型不够大，而是因为高质量的人类数据本身是有限的，而很多真正重要的新能力——超人水平的数学、科学发现、复杂规划——按定义就还没写进现有数据里。

他们的判断是：下一阶段的核心数据源，不再是人类已经写好的东西，而是 **agent 自己在环境里行动、观察、试错、获取反馈之后产生的经验（experience）**。 这个思路与The Second Half的RL迎来泛化导出的原因不同，但指向的目标一致：**去定义新的有价值的问题并给出有价值的reward**

这里的 experience 不是日常用语里的经验丰富，而是 RL 语境下的精确概念：agent 在环境中采取动作、收到观察、经历后果、拿到奖励，并把这条交互轨迹用于后续改进。

他们把这个新阶段拆成四个核心变化：

1. **Streams**：agent 不再是一问一答的聊天模型，而是活在一条持续的经验流里，跨越数月甚至数年地积累知识和修正策略。
2. **Grounded Actions / Observations**：agent 的输入输出不再局限于文本，而是真正落在环境中——网页、代码执行器、API、机器人传感器。行动开始真的改变环境。
3. **Grounded Rewards**：奖励不再只是"人类觉得这个回答好不好"，而是来自环境后果——代码有没有跑通、实验结果是否更好、任务是否完成。
4. **Planning Beyond Human Traces**：推理不必永远像人类写 chain-of-thought 那样进行；agent 可以发展出不同于人类表述的内部计算与规划方式。

需要先说清楚一个判断：**这篇短文更像研究宣言而不是已验证的定律。** 它提出了非常强的方向性判断，但没有给出精确指标告诉你哪一天人类数据时代正式结束。最好的读法，是把它当作一种新的总纲，思考他如何影响我们该做什么的研究。

## 这不是凭空冒出来的

如果把这篇短文只当作 2025 年的新口号，会低估它。它其实是在给过去多年一条隐含主线命名。

| 时间 | 代表工作 | 为什么重要 |
|---|---|---|
| 2017 | [AlphaGo Zero](https://www.nature.com/articles/nature24270) | 不依赖人类棋谱，自博弈生成经验，证明"自己跟自己下"可以学出超人能力 |
| 2018 | [World Models](https://arxiv.org/abs/1803.10122) | 从交互轨迹里学出潜在动力学模型，在模型里"想象"和规划 |
| 2019 | [The Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html) | Sutton 强调：长期看，最能利用计算的通用方法会压过人类先验注入 |
| 2019 | [OpenAI Five](https://arxiv.org/abs/1912.06680) | 大规模在线交互和自博弈可以扩展到长时程、部分可观测、多智能体环境 |
| 2020 | [MuZero](https://www.nature.com/articles/s41586-020-03051-4) | 不依赖显式规则模型，在学习到的模型上做规划 |
| 2023 | [DreamerV3](https://arxiv.org/abs/2301.04104) | 世界模型 RL 走向更通用的 recipe，一个配置跨很多任务 |
| 2024 | [Genie](https://arxiv.org/abs/2402.15391) | 从无动作标签视频里学习可交互环境，"经验基底"本身可以被生成 |

这条线索里有三个不断汇合的分支：**自博弈/自生成经验**（AlphaGo Zero、OpenAI Five）、**世界模型/规划**（World Models、MuZero、DreamerV3）、**方法论上的反人类先验依赖**（Bitter Lesson）。经验时代就是把这三条线合并后，给整个新阶段起的名字。

## 2024-2026：从观点变成现实

如果前面这些更像研究史，那 2024-2026 真正改变气氛的，是一批系统已经开始把这个思路带进 reasoning、computer use 和 robotics。

| 系统 | 时间 | 它代表什么 |
|---|---|---|
| [OpenAI o1](https://openai.com/index/learning-to-reason-with-llms/) | 2024 | 推理能力提升越来越依赖 RL 与推理计算，而非继续堆预训练数据 |
| [DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1) | 2025 | 开源证明"纯 RL 先冒出推理行为"具有强说服力 |
| [AlphaProof](https://www.nature.com/articles/s41586-025-09833-y) | 2025 | 形式化证明器提供高质量验证信号，是经验时代最理想的奖励场景 |
| [AlphaEvolve](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/AlphaEvolve.pdf) | 2025 | 用自动评估器驱动代码/算法搜索，agent 在"试-评-改"循环里进化方案 |
| [Operator](https://cdn.openai.com/operator_system_card.pdf) | 2025 | agent 直接在 GUI 中做事，从答题转成完成任务 |
| [Gemini Robotics](https://arxiv.org/abs/2503.20020) | 2025 | 多模态模型进入实体环境，动作、观察、空间推理、控制闭环合流 |

这些系统里，有三个特别值得展开说。

### DeepSeek-R1：经验能长出新能力，但不会自动带来好用性

DeepSeek-R1 的震动在于：R1-Zero **先做大规模 RL，而不是先做 SFT 冷启动**。结果是纯 RL 的确自然涌现了 self-verification、reflection、long CoT 等行为。但同时也出现了 endless repetition、poor readability、language mixing。

这恰好说明一件比 benchmark 更重要的事：**经验可以长出新能力，但经验本身不会自动带来好用性、可读性和稳定对齐。** 经验时代不是替代一切人类数据，而是减弱人类数据的必须性。

### AlphaProof：为什么数学是经验时代最先爆发的地方

很多人困惑：数学不是文本任务吗？为什么它被当成经验学习的例子？

答案是：**形式化证明环境提供了极其稀缺但极其珍贵的东西——可靠验证器。** agent 写证明步骤、交给 Lean 检查、立即拿到对/错反馈、据此搜索更好策略。这比人类觉得你解释得不错强得多，因为它给出的是可执行、可重复、可规模化的环境反馈。

AlphaProof 在 2024 年 IMO 上拿到银牌水平（解决 6 题中的 4 题），包括只有 5 名选手做出的最难题目。它的成功核心不在于 RL 有多强，而在于**一旦高质量环境反馈存在，经验学习就可能迅速超越纯人类数据路线**。

这也预示了一个关键问题：如果成功依赖验证器，那没有现成验证器的领域怎么办？

### Operator 和 Gemini Robotics：经验第一次"碰世界"

如果只看 reasoning，会觉得经验时代像是更强的 post-training。但一旦看到 Operator 和 Gemini Robotics，事情变得更大：

- agent 开始自己点网页、浏览界面、执行步骤
- 或者通过视觉-动作闭环操纵机器人
- 错误不再只是"答错一道题"，而是可能变成**误点购买、误删文件、错误控制物体**

experience 的真正含义不是"更会推理"，而是**更会在世界里行动，并从行动后果中学习**。

## 经验时代最关键的问题是什么？

这是我认为这篇短文最值得深想的地方。Silver 和 Sutton 不只是在推销 RL——他们实际上在重写 AI 研究的问题清单。

我把这种迁移整理成五对新旧问题：

| 人类数据时代的核心问题 | 经验时代的核心问题 |
|---|---|
| 如何从海量人类文本中提取更多知识？ | 如何让 agent 在环境中持续生产高价值经验？ |
| 如何用人类偏好对齐模型？ | 如何设计可靠的、基于后果的奖励信号？ |
| 如何扩大预训练规模？ | 如何在真实世界中高效采样且保证安全？ |
| 如何写更好的 prompt / chain-of-thought？ | 如何实现长时程 credit assignment 和持续学习？ |
| 如何在 benchmark 上刷分？ | 如何评测一个在开放环境中长期行动的 agent？ |

这五个问题都很重要，但我认为它们之间存在一个瓶颈关系——**奖励设计是其中最根本的那一个**。

### 为什么奖励设计是瓶颈中的瓶颈

考虑一下：

- 如果你没有好的奖励信号，agent 自己生产的经验就没有方向，再多采样也学不到正确的东西
- 如果你没有好的奖励信号，credit assignment 就无从分配——你不知道什么是成功，怎么知道哪一步导致了成功？
- 如果你没有好的奖励信号，安全约束也很难落地——你无法量化什么行为是危险的
- 如果你没有好的奖励信号，评测就只能退回人工评估——可那正是经验时代想超越的东西

换句话说，**奖励信号的质量决定了经验的质量，而经验的质量决定了 agent 能学到什么。**

AlphaProof 和 AlphaEvolve 之所以成功，恰恰是因为它们有近乎完美的验证器——Lean 证明系统、自动化测试套件。在这些领域，奖励信号是清晰的、即时的、可规模化的。

但一旦换成开放世界的任务，问题立刻爆炸：

- 一个网页操作任务算成功的标准是什么？用户满意？页面状态变化？任务完成率？
- 一个健康建议好不好，是当天有效，还是半年后才能判断？
- 一个科研 agent 的实验计划，怎样量化"有价值"？
- **越接近现实世界，奖励越稀疏、越滞后、越多目标冲突。**

这也是为什么研究社区已经开始对 Silver 和 Sutton 的框架提出修正。Bo Wen 在 [The Missing Reward: Active Inference in the Era of Experience](https://arxiv.org/html/2508.05619v1) 中指出了一个关键的"代理差距"：当代 AI 系统无法自主制定、适应和追求目标，而 Silver 和 Sutton 的框架仍然依赖于广泛的人工奖励设计。他提出用自由能原理（Free Energy Principle）从 Active Inference 的角度提供更原则性的答案。

AlignmentForum 上的 [安全分析](https://www.alignmentforum.org/posts/TCGgiJAinGgcMEByt/the-era-of-experience-has-an-unsolved-technical-alignment) 则从另一个角度指出：随着 agent 能力增强，奖励黑客（reward hacking）和规范博弈的问题会变得更严重。一个足够强的 agent 可能学会"看起来在优化你给的奖励，实际上在优化别的东西"。

**所以我的判断是：经验时代的核心技术挑战，不是怎么做更多 RL，而是怎么在没有完美验证器的开放世界里，设计出足够好的奖励信号。** 这个问题没解决之前，经验时代的承诺只能在可验证的封闭领域（数学、代码、形式化系统）里兑现。而实际上这一点和The Second Half是一致的，定义问题比研究解决问题的方法更重要，只要你有一个完美的Rewarder可以近似人类的需求，RL就能将模型带到那个位置。


## 路线之争：这件事远没有共识

Silver 和 Sutton 的宣言听起来很笃定，但 AI 社区并没有达成统一意见。至少存在三条竞争路线。

**Silver / Sutton ：RL + 经验 + 标量奖励**

核心信念是 [Reward is Enough](https://www.sciencedirect.com/science/article/pii/S0004370221000862)——所有目标都可以被表达为累积标量奖励的最大化。通用智能可以从简单的奖励最大化原则中涌现。Sutton 在 2025 年的 [Dwarkesh Podcast](https://pod.wave.co/podcast/dwarkesh-podcast/richard-sutton-father-of-rl-thinks-llms-are-a-dead-end) 访谈中甚至更激进地将 LLM 称为"世界的瞬间痴迷"（a passing fad），认为它们只是被动的模仿者，RL 才是通向真正智能的路径。

**LeCun 路线：世界模型 + JEPA + 最小训练数据**

Yann LeCun 同样认为 LLM 不够，但他的替代方案不是更多 RL，而是构建**编码物理、因果关系和世界如何随时间变化的世界模型**。他提出的 JEPA（Joint Embedding Predictive Architecture）在抽象表示空间中预测未来，而不是在像素或文本层面生成。2025 年底 LeCun 离开 Meta，创立 AMI Labs 并融资超过 10 亿美元，全面押注这条路线。他的核心论证是：人脑用极少数据就能学会理解世界，说明架构比数据量更重要。

**Hassabis 路线：务实折中，一切可用**

Demis Hassabis 对 LeCun 的LLM 是死胡同判断表示反对，指出从图灵机角度看，通用系统的架构理论上能在有足够时间、内存和数据下学习任何可计算的东西。DeepMind 在实践中也确实混合使用——AlphaProof 用 Gemini + AlphaZero，Gemini Robotics 混合了大模型和机器人控制策略。

这三条路线的分歧核心在于：**通往超人智能，是否需要显式世界模型，还是经验+奖励就够了？还是根本不需要那么多经验，只需要更好的架构？** 这个问题在 2026 年初远没有定论。

但无论哪条路线胜出，有一件事是共识：**纯粹依赖人类已有数据的路线正在接近天花板。** 分歧只在于用什么去补。

**核心问题已经变了。我们还在学怎么回答它。**

