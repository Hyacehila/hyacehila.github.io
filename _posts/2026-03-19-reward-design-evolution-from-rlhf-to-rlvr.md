---
layout: blog-post
title: Reward 设计的演化：从 RLHF 到 RLVR，监督对象如何被重写
date: 2026-03-19 20:00:00 +0800
series: Reward and Training
categories: [训练与对齐]
tags: [Alignment, Reward Modeling, Reinforcement Learning]
author: Hyacehila
excerpt: 这篇文章把 reward 主线一次讲到底：从 OpenAI 的 RLHF 奠基，到 PRM 与 RLVR，再到 LLM as Judge、Rubrics as Rewards 与 ArenaRL。真正的主线不是“谁来打分”，而是 reward 本身应该怎样被给出、被组织、被约束。
featured: false
math: true
---

# Reward 设计的演化：从 RLHF 到 RLVR，监督对象如何被重写

[上一篇文章]({% post_url 2026-03-16-rl-alignment-from-reward-to-advantage %})讲的是，一条 reward 怎样通过 KL、baseline、advantage 和 normalization，最后变成可以更新 policy 的训练信号。那篇真正关心的是 `reward -> advantage` 这条链。

这一篇要把问题往前再推一步，而且我想把它一次讲到底：reward 自己到底从哪里来，它应该以什么形式给出，又为什么会从 OpenAI 式 RLHF 一路演化到 PRM、RLVR、LLM as Judge、Rubrics as Rewards，最后走到 ArenaRL 这种开放式 agent 的相对排序框架。这条主线的核心从来不是”换了谁来打分”，而是监督对象本身一直在被重写——OpenAI 时代把人类偏好变成可学的 surrogate reward；再往后，单一答案级总分既太粗、也太容易被 exploit，reward 开始向多目标、ranking、process、verifier、rubric 继续分化；到了开放 agent 场景，问题甚至已经不再是怎样写一个稳定总分，而是怎样把候选之间的相对结构更可靠地送进优化器。

在训练那条线上我们已经反复看到：后面的 optimizer 再精致，也救不了先天结构模糊的上游信号。很多近年的进展表面上像是在改 loss、改 trainer、改 sampling，真正发生变化的地方却往往在 reward production pipeline：监督对象到底是答案、排序、步骤、轨迹还是未来状态，监督来源到底是人、AI、规则、verifier 还是环境后果，信号组织到底是 pointwise、pairwise、listwise、rubric 还是 tournament。换句话说，reward 不能再被理解成某个单独模型吐出来的一行分数，它更像一条生产线——前面负责把目标改写成某种可监督对象，中间负责把这些对象组织成相对稳定的信号，最后才轮到 optimizer 去消费。

## 第一阶段：OpenAI 的 RLHF，用人工反馈作为奖励

严格说，reward from preferences 的原型当然可以追到 [Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741)。但如果把 LLM 时代的 reward 主线单独拉出来，我还是更愿意从 OpenAI 这条线开始，也就是 [Learning to Summarize from Human Feedback](https://arxiv.org/abs/2009.01325) 和 [InstructGPT](https://arxiv.org/abs/2203.02155)。原因很简单，真正把“偏好数据 -> 奖励模型 -> PPO”写成一条可复用工业流水线的，不是概念上的偏好学习，而是这一组具体工作。

先看 2020 年的 `Learning to Summarize from Human Feedback`。它要解决的问题一点都不抽象：摘要质量很难写成手工 reward，但人类对两份摘要做相对比较却很容易。于是论文采用的做法也非常直接：让模型对同一篇文章生成多个摘要候选，收集人工 pairwise comparison，用这些比较数据去训练 reward model，再把这个 reward model 接到 PPO 上，让 policy 朝着更容易得到高 reward 的方向优化。

真正重要的细节不只是“有了一个 reward model”，而是 reward 的监督对象从人工规则改成了偏好排序，reward 本身也不再等于某个原始人类分数，而是一个对偏好关系的学习的神经奖励模型。更关键的是，OpenAI 很早就在这篇工作里**明确观察到 reward over-optimization：policy 一旦过度追逐 RM 分数，输出就会开始偏离真实人类偏好**。也就是说，language task 的确可以被纳入 preference-based RL，但 proxy gap 从第一代系统里就已经出现了。

再往后就是 `InstructGPT`，它真正做的事情，是把这一套 proof of concept 压缩成后来人人都熟悉的三段式流程：**先用 demonstrations 做 SFT，再用 ranked comparisons 训练 reward model，最后用 PPO 在 RM 指导下继续优化 policy。**这篇论文最值得记住的地方，其实不是又做了一次 RLHF，而是它把几个后来几乎所有 post-training recipe 都会继承的结构写死了。

第一，SFT 不是临时 warm-up，而是 reward 优化的行为起点，它先把基本回答风格、任务格式和动作合法性写进模型，再让 RL 去做细调。

第二，真正进入优化器的从来不是一个孤立的 RM 分数，而更像一个复合信号：
$$
R(x, y) = r_\phi(x, y) - \beta \log \frac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)}
$$

也就是说，reward model 给的是往前拉的力，KL 给的是不许偏离太远的边界，这两者共同决定 policy 能往哪里走。

第三，InstructGPT 那个最经典的结果，经过 RLHF 的 1.3B 模型在人类偏好上可以胜过原始 175B GPT-3，真正说明的也不是参数不重要，而是 reward design 和 post-training pipeline 在用户感知质量上可能比继续堆参数更直接。

如果把 OpenAI 这条线压缩成一句话，它做的不是“用人类给模型打分”这么简单，而是三件彼此绑定的事：**先把复杂目标改写成人类可比较的 pairwise preference，再把这些偏好学成神经奖励模型，最后给这条 reward 加上 reference policy 的边界去优化**。也正是在这里，后面所有关于 reward 的扩展都已经埋下了种子，因为一旦你承认 reward 是 surrogate、而且 surrogate 会被 exploit，你就不得不继续追问：这个 surrogate 到底该怎样组织才更稳。

## 第二阶段：走向多目标与可扩展监督

RLHF 一旦在语言模型里跑起来，问题很快就从能不能做变成这种单一偏好分数到底能覆盖什么。这一阶段最重要的变化不是 trainer 换了什么，而是 reward 对象本身开始变得多维。

[Training a Helpful and Harmless Assistant with RLHF](https://arxiv.org/abs/2204.05862) 很值得放在这里，因为它第一次正面把这个问题摊开：helpful 和 harmless 不是一个维度。如果你只追更符合用户意图的回答，模型可能在危险请求上变得过于迎合；但如果你只强调无害，它又会迅速滑向过度保守。这个工作最重要的意义，不是又复刻了一遍 RLHF，而是它让 alignment 从“单一质量分数”正式变成“多目标之间的折中”。从 reward 设计的角度看，这件事非常关键，因为它说明单一 scalar reward 很多时候只是为了优化方便而做的压缩，并不等于目标本身本来就只有一个轴。后面为什么越来越多系统会显式引入子分数、rubric 和 hard constraints，某种意义上都可以追到这里。

如果说 HH RLHF 暴露的是目标本身不止一个，那么 [Constitutional AI](https://arxiv.org/abs/2212.08073) 和 [RLAIF](https://arxiv.org/abs/2309.00267) 暴露的则是监督来源也不一定总得靠人。Constitutional AI 的关键做法，是先写下一组 constitution principles，让模型先自我批评、再自我修订，然后再把 AI feedback 接回 preference 或 reward 流水线。RLAIF 则把这一思路进一步工程化，把原本完全依赖人工比较的上游部分替换成更规模化的 AI feedback。它们真正改写的，不是谁来做标注，而是 reward production pipeline 的最上游开始被结构化：原则本身成为先验约束，critic 的生成过程开始可程序化，监督来源不再只是人工外包，而变成制度设计的一部分。

这一阶段对 reward 设计留下的启发非常直接。
1. reward 最好别再被想象成一个单目标 utility scalar，它往往是多目标压缩后的产物。
2. 上游 judge 生产线本身也要设计，constitution、AI critique、comparison policy 用什么，都会改变 reward 的形状
3. 一旦目标确实是多维的，未来往 rubric、向子分数、向 hard constraints 走几乎是必然的，因为继续把所有东西揉成平均分，只会把真正的冲突藏起来。

## 第三阶段：代理奖励的结构性诊断——偏好学习能被修好吗

前两个阶段扩展了 reward 的目标维度和监督来源，但有一个更基本的问题一直没有被正面追问：**从偏好数据里学出来的 reward model，作为一个代理（proxy），它自己到底有多可靠？**

这个问题之所以重要，是因为 RLHF（以及后面的Constitution AI） 的核心架构选择就是用学习的方式把偏好压成一个神经奖励函数，再让优化器去追逐这个函数。如果 proxy 本身有结构性缺陷，那么后面无论怎么改 trainer、改 loss、改 sampling，都是在一个先天不牢的地基上修补。2023 年前后的一批工作，恰好从不同层面把这个地基的裂缝逐一暴露了出来。

**信息在进入 RM 之前就被丢了**

[Preference Ranking Optimization](https://arxiv.org/abs/2306.17492) 针对的问题并不复杂：如果你手里天然就有一个候选集合的排序，把它拆成若干独立的 pairwise 胜负再去训练 RM，会不会先丢掉结构信息？PRO 的回答是会，而且丢得很严重。一个 listwise 排序所包含的全局位序关系，比从中拆出的若干 pairwise 二元比较要丰富得多。

[GPO](https://deepmind.google/research/publications/92798/) 暴露的是另一种丢失（GPO是一种对DPO的优化，但是我们还是放到RL里一起讨论，因为他也是偏好学习的典型例子）。很多真实比较并不长成干净的 0 或 1，它更像”A 略好于 B””两者几乎一样”或者”不同标注者之间有稳定分歧”。但传统 RLHF 把这些灰度统一压成了硬标签。GPO 的核心价值，就是把偏好里的不确定性保留下来做 soft preference labels，让优化器不至于在本来就很模糊的边界样本上过度用力。

这一层的问题原则上是可修的——用 listwise loss，用 soft label。但它们共同揭示的模式非常重要：**从原始偏好到 RM 训练数据的每一步压缩，都在丢信息，而丢掉的信息后面再也拿不回来。**

**第二层裂缝：统计建模本身在写 reward**

[A density estimation perspective on learning from pairwise human preferences](https://deepmind.google/research/publications/a-density-estimation-perspective-on-learning-from-pairwise-human-preferences/) 和 [RLHF and IIA: Perverse Incentives](https://deepmind.google/research/publications/63806/) 把问题往底层又推了一层。它们共同提醒的一件事是：pairwise preference 学习并不天然等于恢复一个全局稳定的 utility scalar。你选择用 Bradley-Terry 还是 Thurstone，是否假设 IIA（无关选项独立性），这些看似技术性的建模选择会直接塑造学出来的 reward 的几何形状。尤其是 IIA，虽然让学习更顺手，却可能诱导出扭曲的激励结构和 perverse incentives。

这一层比第一层更深：不只是数据格式的问题，而是**拟合偏好这个行为本身就在用假设替代真实偏好**。即便数据完美，你选的模型家族也在替你做决定。reward model 不是一个中性的统计容器，它本身就在写 reward。

**第三层裂缝：即使修好以上所有，proxy 仍然是 proxy**

前两层裂缝至少还可以在偏好学习范式内修补。但第三层问题是结构性的：无论你把数据组织得多忠实、统计假设选得多审慎，RM 终究是一个学出来的近似，而不是真实目标本身。

这并不是理论上的担忧。从 [Learning to Summarize](https://arxiv.org/abs/2009.01325) 开始，reward over-optimization 就已经被明确观察到：policy 追逐 RM 分数到一定程度后，输出就会开始偏离真实人类偏好。优化器越强，它就越擅长找到 RM 和真实目标之间的缝隙。这不是某个特定 RM 的 bug，而是所有 surrogate reward 的结构性属性——**只要你给优化器的不是目标本身，而是目标的近似，exploit 就只是时间问题。**

**诊断之后的分岔**

这三层诊断叠在一起，指向了一个清晰的分岔口。对于开放式任务（创意写作、开放对话、复杂建议），ground truth 本来就不存在或无法形式化，proxy 是不得不用的——但需要被做得更结构化，这正是后面 LLM as Judge、rubric reward 和 tournament ranking 要解决的问题。

但对于另一类任务——数学推理的对错、代码能不能跑、工具调用的结果是否符合预期——答案的正确性是可以直接验证的。既然可以验证，为什么还要中间夹一个学出来的近似？这正是下一阶段 RLVR 的出发点：不是把 proxy 学得更好，而是对可验证的部分直接绕过 proxy。

## 第四阶段：RLVR——可验证奖励与数学推理的突破

第三阶段的诊断已经指出了方向：对于可验证任务，最彻底的做法不是把 proxy 学得更准，而是直接用 verifier 替代它。而 RLVR 这条线最有说服力的突破，恰恰发生在数学推理领域。

数学题有一个天然优势：答案对错是可以程序化验证的。这意味着你不需要一个学出来的 reward model 去猜分，只需要一个 verifier 去核对最终结果。[DeepSeek-R1](https://arxiv.org/abs/2501.12948) 把这个思路推到了极致：在数学推理任务上，仅靠 outcome-level 的可验证奖励（答案对了给正分，错了给零分）加上基本的格式约束，用 GRPO 做 RL，模型就自发涌现出了长链条推理、自我纠错和反思等能力。这个结果之所以重要，不只是因为效果好，而是因为它清楚地证明了一件事：**当任务可验证时，简单的 outcome reward 配合足够强的 RL 优化，就已经能产生远超预期的行为涌现，根本不需要中间夹一个学出来的 proxy。**

这是 RLVR 真正的核心命题：能验证就直接验证，把 proxy 彻底绕过去。

但紧接着就会遇到两个具体问题。第一，纯 outcome verification 只告诉你最终答案对不对，却不告诉你中间哪一步出了问题，这在长链条推理里意味着 credit assignment 极其稀疏。第二，很多任务并非全部可验证，只有部分结构是可以核查的。

针对第一个问题，[Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) 提出了 process reward model（PRM），把监督单位从 answer-level 改成 step-level。这个思路在 RLVR 语境下最重要的贡献，不是要对每一步做全面的质量打分，而是提供了一种拆分策略：把长链条推理拆成可独立验证的步骤，努力让更多中间环节也变成可验证的对象。[Math-Shepherd](https://arxiv.org/abs/2312.08935) 进一步把 step-level 的验证模型用到了 test-time 的候选排序和路径筛选上，让 reward 不只是训练时的信号，也成为推理时的系统部件。但需要强调的是，PRM 在 RLVR 这条线上的核心价值，是帮助找到更细粒度的可验证对象，而不是退回去做一个更精致的 learned proxy——那样就又回到了第三阶段诊断的老问题。

针对第二个问题，[Crossing the Reward Bridge](https://arxiv.org/abs/2503.23829) 提供了一个关键的方法论判断：很多看起来只能靠 judge 猜分的任务，只要拆得足够细，其实总有一部分结构要求、环境结果或中间约束是可验证的。也就是说，verifiable reward 的适用范围远比”数学题和代码”要广，关键在于你愿不愿意把任务拆开，把可验证的部分先拎出来。[Beyond Outcome Verification](https://arxiv.org/abs/2601.17223) 则从另一个角度补充了这一点：即使最终结果可验证，知道模型究竟在哪一步开始偏航仍然有价值，而这种中间诊断本身也可以尽量做成带明确依据的验证，而不是退回到纯 learned score。

到了 RLVR 这一阶段，reward 的默认写法越来越明确：**如果任务里存在可验证的结构，就先让 verifier 提供核心结果奖励；如果任务是长链条推理，用拆分的方式尽量让更多中间环节也变成可验证对象；如果还有安全、格式、协议等红线，就把它们单独做成 hard constraints，而不是混进平均分里。** RLVR 的本质不是换个缩写继续做 RLHF，而是把 reward source 尽量改写成可验证对象——数学推理的突破已经证明，这条路走得通，而且走得比预期更远。

## 第五阶段：LLM as Judge 的登场——从朴素评分到偏差诊断

但现实任务经常并不这么理想。很多开放任务既没有唯一答案，也没有完全覆盖的 verifier，可你又不能退回纯人工标注，因为规模根本不允许。这就是 `LLM as Judge` 真正进入 reward 主线的地方。它的重要性不在于”最近很火”，而在于它刚好填补了 verifier 和纯人工之间的空位。

这条线的早期代表是 [G-Eval](https://arxiv.org/abs/2303.16634) 和 [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685)。G-Eval 用 `CoT + form-filling` 的方式，让模型先沿着评估逻辑思考，再按结构化表单输出判断；MT-Bench 和 Chatbot Arena 则更进一步证明，在开放对话偏好比较里，强 LLM judge 的确可以在相当程度上近似人类判断，形成一种可规模化的代理评审层。这组工作真正带来的改变，不是”以后不需要人类了”，而是开放任务里终于出现了一种可以大规模近似人类偏好的中间接口。

一旦把 judge 放回 reward 语境，第一个方法论问题就是：judge 到底更适合 direct scoring，还是 pairwise ranking？direct scoring 吞吐高、形式简单、容易批量化，但它要求 judge 自己决定分数尺度，而这在开放任务里通常并不稳定——7 分和 8 分到底差在哪，judge 自己未必讲得清，跨 prompt 的尺度也很容易漂移。pairwise ranking 让 judge 回到它最擅长的动作：比较两个候选谁更好。[Aligning with Human Judgement: The Role of Pairwise Preference in Large Language Model Evaluators](https://arxiv.org/abs/2403.16950) 说得很清楚，把评估视为 ranking problem 而不是 pointwise scoring problem，很多 misalignment 会自然下降。从 reward 设计角度看，这意味着开放任务里的 reward production 更像是在生产相对结构，而不是在发明稳定总分。

但 judge 的风险也必须在这里说清楚。它最大的危险不是”有一点偏”，而是很容易因为表现不错就被误当成神谕。[Large Language Models are not Fair Evaluators](https://arxiv.org/abs/2305.17926) 和 [Judging the Judges](https://arxiv.org/abs/2406.07791) 指出候选顺序一换，judge 的判决就可能翻转（position bias）；MT-Bench / Chatbot Arena 相关研究持续暴露 verbosity bias，judge 常偏爱更长、更像好答案样子的文本；[Self-Preference Bias in LLM-as-a-Judge](https://arxiv.org/abs/2410.21819) 说明 judge 还可能系统性偏爱更像自身分布的答案；而 [JudgeBench](https://arxiv.org/abs/2410.12784) 则提醒得更尖锐——和人类偏好一致并不等于能够可靠裁决事实与逻辑。

把这些偏差放回 reward 主线，结论非常明确：**朴素的 LLM as Judge 可以成为开放任务里的代理评审层，但它不适合单独承担高风险、强事实性任务里的最终 reward oracle。** 如果 judge 的输出只是一个黑盒总分，那它本质上和第三阶段诊断过的 learned proxy 没有根本区别——都是一个你看不清内部结构的近似函数。真正要让 judge 稳下来，不是换一个更大的模型，而是给它加结构。

## 第六阶段：结构化 Judge——从总分到多维可校准的 reward 接口

上一阶段暴露的核心问题可以压成一句话：**一个总分太黑盒，也太粗糙。** judge 知道怎么比较好坏，但让它直接输出一个 pointwise scalar，尺度就会漂移、维度就会坍塌、红线就会被平均分稀释。解决这个问题的思路不是把 judge 替换掉，而是给它加上结构化的评分框架——也就是 rubric。

[LLM-Rubric](https://aclanthology.org/2024.acl-long.745/) 最早把这条路走通：把评价目标拆成多个独立维度，让 LLM 沿每个维度输出概率分布而非点估计，再用校准层映射到真实评审者的尺度。它真正做的事情，是把 judge 从”黑盒打分器”变成了”结构化信号提取器 + 可校准的映射层”。一旦走到这里，rubric 就不只是一张评分表，而是 reward 的结构化接口设计。在工程层面，[Prometheus 2](https://arxiv.org/abs/2405.01535) 进一步证明 judge 不是临时套一层 prompt 的小工具，而是需要被专门训练、专门约束、专门评估的独立系统部件。

真正把 rubric-driven judge 推到 reward 接口终点的，是 [Rubrics as Rewards](https://openreview.net/forum?id=21UFlJrmS2)。它的核心判断非常清晰：如果任务没有唯一答案，但可以写出足够清楚的 rubric，那么 reward 为什么不直接沿 rubric 来给？与之前的工作不同，Rubrics as Rewards 关心的不只是怎样评估，而是怎样把 rubric 评估直接接入 RL 训练循环，而且不需要任何人工标注数据。它在技术上做了三件关键的事：用强 LLM 以黄金参考答案为条件自动生成 rubric criterion，把 criterion 按重要性分级赋权（Essential / Important / Optional / Pitfall），然后选择聚合方式把 criterion-level 判断变成 reward scalar。实验一致显示隐式聚合（把所有 criterion 一起送给 judge 整体评分）优于显式逐条加权求和。更实用的发现是，**弱 judge 配合 rubric 可以逼近强 judge 的直接评分效果**——5B 级别小模型在 rubric 指导下接近 70B 模型的直接 Likert 评分，这对 RL 训练中大量调用 judge 的场景意味着巨大的成本优势。关于 Rubrics as Rewards 的完整技术细节（自动生成流程、分级权重设计、聚合公式、GRPO 集成方式和最小可行实操路径），见附录。

这个设计解决了朴素 judge 的几个核心痛点：**可解释性**——每一分都可以追溯到具体 criterion；**红线可插入**——安全违规、事实性错误可以作为 Essential 或 Pitfall 单独判断，不被高分稀释；**成本可控**——弱模型配合 rubric 就够用。当然，天花板也很明确：**rubric 写得差，reward 质量就会立刻下降。** reward engineering 的重心从训练一个更好的 RM 转移到了写一份更好的 rubric——而后者本质上是一个 specification 问题，不是一个 ML 问题。

到这一阶段，开放任务里 reward 的默认写法已经非常明确：**当 verifier 覆盖不到时，reward 最稳的形态不是让 judge 给一个总分，而是先写 rubric 定义行为规格，再让 judge 沿维度输出结构化判断，最后把 criterion-level signals 聚合成 reward。** 但即便做到了这一步，当轨迹变长、候选变异构、差异变细微时，pointwise 评分仍然容易出现 discrimination collapse——judge 能感觉出谁更好，却说不稳到底值多少分。

## 第七阶段：ArenaRL——当结构化评分也不够时，用相对排序兜底

如果前面所有阶段都还能被理解成怎样更好地生产分数，那 [ArenaRL](https://arxiv.org/abs/2601.06487) 真正推进的一点就在于，它开始怀疑开放式 agent 里根本不该先发明 pointwise 分数。一旦轨迹变长、候选变异构、差异变细微，pointwise scalar reward——哪怕是经过 rubric 结构化的——仍然很容易出现 discrimination collapse：judge 能感觉出谁更好，却说不稳到底值多少分，更别提跨任务、跨轨迹维持统一尺度了。

ArenaRL 的 reward production pipeline 分三层，每一层解决一个具体问题。**第一层是 process-aware pairwise evaluation**：不给单条轨迹打分，而是让 judge 去比较两条轨迹谁更好，同时关注推理链逻辑、工具调用和中间步骤，并做双向评分消除位置偏差。**第二层是 seeded single-elimination tournament**：用锚点轨迹做预排序后构建淘汰赛，把 N 条候选的排序从 O(N²) 的全循环赛压缩到 O(N) 次比较，精度几乎不损失。**第三层是 quantile-based reward mapping**：把离散排名转为归一化分位数值，直接作为 advantage signal 送入 PPO，不需要跨 batch 维护历史状态。关于三层 pipeline 的完整技术细节（双向评分机制、种子淘汰赛流程、quantile vs Elo vs Bradley-Terry 的对比、实验数字），见附录。

这套方案的局限也必须说清楚：tournament 依旧高度依赖 judge 质量，成本仍高于直接 pointwise scoring，而且 quantile mapping 只能产生局部相对信号，无法给出跨 batch 的绝对质量判断。但 ArenaRL 对我最重要的启发非常明确：**在 open-ended agent 里，reward design 的重点已经不再是写一个看似优雅的统一总分，而是怎样让候选之间那些细微但关键的相对差异，更可靠地进入优化器。**

## 所以，reward 本身到底应该怎么给出

写到这里，最值得单独回答的问题还是那个老问题：reward 到底该怎样给。最稳的做法，是先按任务结构决定 reward 的来源和组织形式，再决定最后要不要把这些信号聚合成分数。

| 任务类型 | 首选 reward 形式 | 为什么 | 不建议的起手式 |
| --- | --- | --- | --- |
| 单轮开放回答、摘要、对话 | pairwise / listwise 偏好，必要时加 rubric judge | 绝对分数尺度不稳，相对比较更自然 | 直接让大模型给总分 |
| 多目标助手式任务 | 多维子分数 + hard constraints + KL/风格边界 | helpful、harmless、truthfulness 本来就不是一个轴 | 把所有维度揉成平均分 |
| 数学、代码、工具调用等可验证任务 | verifier-backed outcome reward + process reward 辅助 | 能验证就尽量别只靠猜 | 忽略 verifier、全交给 judge |
| 长链条 reasoning | step-level PRM / process verifier + final outcome | 只看终点会吞掉过程价值 | 只在最后给 0/1 |
| 开放式 agent 轨迹 | pairwise / tournament ranking + process checks + cost/safety penalties | pointwise scalar 容易 discrimination collapse | 试图先发明一个稳定总分 |
| 个性化或多用户产品 | reward features / context-conditioned reward | 不同用户偏好不共享单一 utility | 假设所有人用一个 reward 就够 |

第一步不是问总分怎么算，而是先问什么行为值得鼓励、什么行为必须禁止、哪些部分其实可以验证、哪些地方如果出现幻觉、越权工具调用、安全违规或关键格式错误，就根本不该被“整体表现不错”抵消。很多团队一上来就写平均分，最后最大的坑也往往来自这里：红线没有被拉出来，优化器自然会学会用别的 bonus 去抵消它。

第二步才是决定监督对象到底是什么。单轮开放回答更适合 pairwise 或 listwise，因为相对结构比绝对尺度更稳；多目标助手式任务应该先拆维度，再决定如何聚合；数学、代码和工具调用这种可验证任务，优先让 verifier 做 outcome reward，再用 PRM 或 process verifier 去补过程差分；长链条 reasoning 则不要把所有信用都压到最后一步；开放式 agent 里如果本来就有多候选轨迹，最好尽量保留 ranking 结构，而不是太早把它压扁成 pairwise 0/1。这里还要额外警惕一件事，就是信息在监督组织里被过度丢弃。多候选排序、软偏好、不确定性、用户差异，这些东西如果在进入 reward 前就被压平，后面训练再精致也拿不回来。

最后一步，是把 judge 当成代理层，而不是神谕，并把整个 reward 当成流水线，而不是一个模型。一个更稳的默认模板其实很朴素：先把 hard constraints 单独拉出来，再决定 core task signal 优先来自 verifier、pairwise ranking 还是 rubric，接着才考虑 process signal 要不要补进来，最后用成本、时延、冗余工具调用、可读性这类辅助项做边际修正。

回到最开始的那个比喻：reward 不是一个模型吐出来的一行分数，而是一条生产线。这篇文章从 OpenAI 的 RLHF 一路走到 ArenaRL，真正在讲的始终是同一件事——这条生产线的上游对象、中游组织和下游接口是怎样一步步被重写的。把 reward 的生产结构理清楚之后，下一篇我们就可以进入 training loop 本身：当这些不同形态的 reward 真正送进优化器时，从 SFT 到 PPO 到 GRPO 再到 agentic RL，训练框架自己又经历了怎样的演化。

## 附录：Rubrics as Rewards 技术细节

[Rubrics as Rewards](https://openreview.net/forum?id=21UFlJrmS2) 在技术上做了三件关键的事，这里展开介绍。

**第一件事是 rubric 的自动生成。** 论文不要求你手写每一条 criterion，而是用一个强 LLM（比如 GPT-4o 或 o3-mini）以黄金参考答案为条件，为每个 prompt 自动生成 7-20 个自包含的 criterion items。每个 criterion 本质上是一个二元验证检查——"回答是否清晰阐述了核心概念？""是否避免了常见的逻辑陷阱？""是否提供了有效的支持证据？"。这里有一个重要的实验发现：**用黄金参考答案锚定的 rubric 质量显著高于不带参考的版本**，因为参考答案为 criterion 的生成提供了具体的行为标杆，而不是让 LLM 凭空猜测"好回答应该长什么样"。

**第二件事是 criterion 的分级权重。** 生成的 criterion 不是等权平均的，而是被分为四类：**Essential**（必需，如核心逻辑正确性）、**Important**（重要，如论据充分性）、**Optional**（加分项，如表达优雅度）和 **Pitfall**（陷阱/扣分项，如过度诊断、逻辑循环）。这个分类直接决定了每个 criterion 在最终 reward 里的贡献权重，也意味着红线项（Essential 或 Pitfall）不会被其他维度的高分稀释。

**第三件事是聚合方式的选择。** 论文同时探索了两条路径。**显式聚合**：judge 逐条评判每个 criterion 是否满足（是/否或分数），然后用加权求和 $R = \sum_i (s_i \times w_i) / \sum_i w_i$ 得到最终 reward——可解释性最强，但权重调优脆弱。**隐式聚合**：把所有 criterion 和权重一起送给 judge，让它在理解全部标准后整体输出一个 0-1 分数——灵活性更高，能捕捉 criterion 之间的交互关系。实验结果一致显示**隐式聚合效果更好**，在 HealthBench 上相对提升达 31%，在 GPQA-Diamond 上提升 7%。

在与 RL 训练的集成上，Rubrics as Rewards 直接嫁接到 GRPO（Group Relative Policy Optimization）框架：对每个 prompt 生成一组候选回答（通常 4-8 条），用 rubric judge 对每条回答评分，以组内均值作为 baseline，advantage = 个体分数 - 组内均值，然后用这个 advantage 驱动 PPO 风格的策略更新。这个流程和标准 GRPO 的唯一区别，就是 reward 不再来自一个 learned RM 或简单的 verifier，而是来自 rubric judge 的结构化评估。

**最小可行实操路径：**(1) 为你的任务准备黄金参考答案（哪怕只有部分 prompt 有参考也行）；(2) 用强 LLM 以参考答案为条件自动生成 rubric，每个 prompt 7-20 个 criterion；(3) 把 criterion 按 Essential/Important/Optional/Pitfall 分类赋权；(4) 选择隐式聚合（推荐）或显式聚合；(5) 接入 GRPO 训练循环，用 rubric judge 的评分替代传统 RM 的评分。整个过程不需要训练任何 reward model，reward engineering 的重心完全转移到了 rubric 的设计和迭代上。

## 附录：ArenaRL 技术细节

[ArenaRL](https://arxiv.org/abs/2601.06487) 的三层 pipeline 展开如下。

**第一层：Process-Aware Pairwise Evaluation。** ArenaRL 不给单条轨迹打分，而是让 judge 去比较两条轨迹谁更好。它的 pairwise 比较不只看最终结果——judge 同时关注 CoT 推理链的逻辑紧密度、工具调用的准确性和中间步骤的合理性，这就是所谓的 process-aware。这个设计直接针对的是开放 agent 场景下的一个常见问题：两条轨迹可能最终结果差不多，但一条是通过合理规划到达的，另一条只是碰巧蒙对了，pointwise 评分很难区分这种差异，但 pairwise 比较可以。更关键的是，每对轨迹都会做**双向评分（Bidirectional Scoring）**：先把 A 放前面 B 放后面评一次，再交换顺序评一次，以消除 LLM judge 的位置偏差（position bias）。

**第二层：Seeded Single-Elimination Tournament。** 有了 pairwise 比较能力后，下一个问题是怎么用它处理 N 条候选轨迹。最朴素的做法是 Round-Robin（全循环赛），每两条都比一次，但这需要 O(N²) 次比较——成本不可接受。ArenaRL 的解决方案是 **Seeded Single-Elimination**（有种子的单淘汰赛），只需要 N-1 次比较。具体分两步：先用**贪心解码生成一条基准轨迹作为锚点（anchor）**，用这个锚点对所有候选做一轮快速预排序，得到种子序列——防止高质量样本在早期轮次就相互碰撞而被淘汰；然后基于种子排序构建二叉竞赛树，每轮两两配对做 pairwise evaluation，胜者晋级。在 Open-Travel 基准上，这个 O(N) 的 tournament 方案和 O(N²) 的 Round-Robin 精度基本持平（32.5% vs 32.9%），但计算成本降低了一到两个数量级。

**第三层：Quantile-Based Reward Mapping。** Tournament 产出的是一组离散排名，但 RL 优化器需要连续的 advantage signal。ArenaRL 用**分位数映射**来完成转换：在 N=10 的竞赛中，第 1 名映射到 0.95 quantile，第 10 名映射到 0.05 quantile，中间线性插值，直接作为 advantage signal 送入 PPO。为什么选 quantile 而不是 Elo 或 Bradley-Terry？第一，quantile 在每个 batch 内独立计算，不需要跨 batch 维护历史状态；第二，相对排名对评估噪声更鲁棒；第三，计算效率高，排序 O(N log N) 加线性映射即可。而 Elo 需要迭代求解、对噪声敏感，Bradley-Terry 在小 batch 内数据量不足以可靠拟合。

把三层串起来：**pairwise evaluation 解决 discrimination collapse**；**bidirectional scoring 解决 position bias**；**seeded tournament 解决计算成本**（O(N) vs O(N²)）；**quantile mapping 解决 reward interface**（离散排名转为连续信号）。在 Open-Travel 基准上，SFT baseline 的胜率是 16.4%，传统 GRPO 也只有 16.4%，而 ArenaRL 达到了 41.8%，提升了 155%。

## 附录：当 reward 很难写时，能不能直接借环境后果做监督

这一部分我仍然放在附录里，而不再把它当主线的一部分，因为它更像对 reward 边界的提醒，而不是 reward 主线本身的必经阶段。[Agent Learning via Early Experience](https://arxiv.org/abs/2510.08558) 重要的地方，不是宣布 reward 过时了，而是提醒我们 reward 只是环境信息的一种压缩形式，不是唯一监督方式。

这条线的核心做法，是从 expert trajectory 的状态点出发，对替代动作做受控探索，真实执行这些动作，再直接利用未来状态或者 grounded reflection 做监督。它真正有意思的地方，在于监督对象不再只是“此刻给多少分”，而是把未来环境后果本身拿回来了。也正因为如此，这类方法的信息密度往往会比手工写一个单步 scalar reward 更高。

但我仍然把它放在附录，是因为它更像在告诉我们 reward 的边界在哪里：reward 从来不是目标本身，也不是环境信息的唯一编码方式。当 reward 很难写时，环境后果当然可以成为更高信息密度的中间监督层；可只要我们还在讨论 reward 设计，这条主线真正关心的仍然是如何把目标压成一个尽可能稳的信号接口，而不是彻底绕开 reward 这个问题。

## 附录：当 Judge 不止一个——多智能体辩论与 Evaluation Agent

主线里我们讨论的 LLM as Judge，无论是朴素评分、结构化 rubric 还是 tournament ranking，本质上都还是单一智能体在做评估。但 [ChatEval](https://arxiv.org/abs/2308.07201)（ICLR 2024）提出了一个很自然的追问：人类评估本身就依赖多标注员协作，为什么 LLM 评估不能也这样做？

ChatEval 的核心做法是构建一个多智能体裁判团。多个 LLM agent 各自带着不同的角色设定（例如普通读者、批评家、新闻作者、心理学家、科学家），对同一份文本进行自主辩论，在讨论中暴露单一视角可能忽略的差异，最终通过投票或分数平均产出评估结果。这里有几个设计细节值得关注：

**角色多样性模拟人类标注差异。** 不同角色带着不同的关注重点和评价偏好进入讨论，这和真实人类标注场景里标注者之间的系统性分歧是同构的。单一 judge 即使很强，也只能代表一种视角；多角色协作则试图用 AI 模拟人类评估流程中"多样性产生鲁棒性"的特性，这背后集成了社会学里集体智慧和认知协同的思想。

**沟通策略的设计空间。** ChatEval 同时考虑了三种多智能体沟通策略：顺序发言拼接（agent 依次发言，后者看到前者的完整输出）、异步发言拼接，以及包含总结器的聊天历史——后者让后续 agent 只需阅读前面讨论的摘要而不是完整记录，有效控制了上下文长度。这个设计选择在实际部署中很重要，因为多轮辩论的 token 成本会快速膨胀。

**不要求共识，而是聚合。** 多智能体辩论场景下，最终并不要求所有 agent 形成统一意见。通过投票或者平均全部参与者的分数即可产出最终结果。这个设计选择很务实：强制共识反而可能压缩有价值的分歧信息。

实验结果也证实了这一点：在 TopicalChat 基准上，ChatEval 的多智能体讨论与人类判断的 Kendall Tau 相关性达到 0.57，而单一 GPT-4 evaluator 只有 0.52。更关键的是，简单 ensemble（多次独立评分取平均）并不能显著提升效果，真正起作用的是自然语言交互本身——辩论过程中的论证、质疑和补充才是信号增益的来源。

把 ChatEval 放在附录而不是主线，是因为多智能体辩论评估目前更多还是一个评估方法论的扩展，而不是 reward production pipeline 的直接组成部分。但它指向的方向非常值得关注：**LLM as Judge 几乎必然会成为 rule-based 和 human feedback 之间的中间层，在所有涉及开放式生成任务评估的领域发挥作用。** 从这个角度看，评估的未来不只是单一 context learning 和多智能体辩论的选择，而是构建包含工具调用、多轮交互和复杂结构的 evaluation agent。ChatEval 的多智能体辩论是这条路上的早期探索，它同时也提示了另一个有趣的可能：多智能体协同可以缓解单一 judge 的思维退化（degeneration-of-thought）问题，因为辩论中的质疑和反驳天然构成了一种对抗性的自我纠正机制。

## 参考资料

### 核心入口

- [Learning to Summarize from Human Feedback](https://arxiv.org/abs/2009.01325)
- [InstructGPT](https://arxiv.org/abs/2203.02155)
- [Training a Helpful and Harmless Assistant with RLHF](https://arxiv.org/abs/2204.05862)
- [Constitutional AI](https://arxiv.org/abs/2212.08073)
- [RLAIF](https://arxiv.org/abs/2309.00267)
- [Preference Ranking Optimization](https://arxiv.org/abs/2306.17492)
- [Geometric-Averaged Preference Optimization for Soft Preference Labels](https://deepmind.google/research/publications/92798/)
- [A density estimation perspective on learning from pairwise human preferences](https://deepmind.google/research/publications/a-density-estimation-perspective-on-learning-from-pairwise-human-preferences/)
- [RLHF and IIA: Perverse Incentives](https://deepmind.google/research/publications/63806/)
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948)
- [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)
- [Math-Shepherd](https://arxiv.org/abs/2312.08935)
- [Crossing the Reward Bridge](https://arxiv.org/abs/2503.23829)
- [Beyond Outcome Verification](https://arxiv.org/abs/2601.17223)
- [G-Eval](https://arxiv.org/abs/2303.16634)
- [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685)
- [Prometheus 2](https://arxiv.org/abs/2405.01535)
- [LLM-Rubric](https://aclanthology.org/2024.acl-long.745/)
- [Rubrics as Rewards](https://openreview.net/forum?id=21UFlJrmS2)
- [ArenaRL](https://arxiv.org/abs/2601.06487)

### 延伸阅读

- [Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741)
- [Exploration at Scale using Epistemic Neural Networks](https://deepmind.google/research/publications/exploration-at-scale-using-epistemic-neural-networks/)
- [Capturing Individual Human Preferences with Reward Features](https://arxiv.org/abs/2503.17338)
- [Aligning with Human Judgement: The Role of Pairwise Preference in Large Language Model Evaluators](https://arxiv.org/abs/2403.16950)
- [Large Language Models are not Fair Evaluators](https://arxiv.org/abs/2305.17926)
- [Judging the Judges: A Systematic Study of Position Bias in LLM-as-a-Judge](https://arxiv.org/abs/2406.07791)
- [Self-Preference Bias in LLM-as-a-Judge](https://arxiv.org/abs/2410.21819)
- [JudgeBench](https://arxiv.org/abs/2410.12784)
- [Agent Learning via Early Experience](https://arxiv.org/abs/2510.08558)
- [ChatEval: Towards Better LLM-based Evaluators through Multi-Agent Debate](https://arxiv.org/abs/2308.07201)

