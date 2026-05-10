---
layout: blog-post
title: Reward Hacking：当奖励信号被优化器反向搜索
date: 2026-03-20 20:00:00 +0800
series: Reward and Training
categories: [基础模型]
tags: [Alignment, Reward Modeling, Evaluation]
author: Hyacehila
excerpt: 当 policy 的探索能力超过 reward signal 的判别和泛化能力，并且优化压力持续增加时，reward hacking 就会从偶发错误变成系统性训练风险。
featured: false
math: true
---

# Reward Hacking：当奖励信号被优化器反向搜索

[上一篇文章]({% post_url 2026-03-19-reward-design-evolution-from-rlhf-to-rlvr %})讲的是 reward 自己怎样被生产出来：从人工偏好、神经 reward model、PRM、RLVR、LLM as Judge、rubric，一路到开放 agent 里的 tournament ranking。那篇文章的主线是：**reward 不是一个分数，而是一条生产线。**

但只讲 reward 怎样生产还不够。reward hacking 不是 reward production 的一个阶段，而是任何 reward signal 被强优化以后都可能暴露出的风险。只要 reward 被接进 optimizer，它就会从一个评估接口变成一个被主动搜索、主动放大、主动利用的目标。模型不是被动接受奖励，而是在训练中持续试探奖励系统的边界。于是 reward design 里最需要提前防范的问题就出现了：

**当一个信号被当成目标优化以后，它还配不配继续代表原来的目标？**

这就是 reward hacking 或 reward over-optimization 的核心。它不是单个问题 prompt、异常样本或个别模型缺陷导致的偶发 bug，而是代理目标进入强优化以后需要默认防范的结构性风险。[Learning to Summarize from Human Feedback](https://arxiv.org/abs/2009.01325) 很早就观察到，policy 过度追逐 reward model 分数后，真实人类偏好反而会下降；[DeepSeek-R1](https://arxiv.org/abs/2501.12948) 在讨论 R1-Zero 时也明确说，没有使用 neural reward model 的原因之一，是大规模 RL 中神经 RM 可能带来 reward hacking。

也就是：

**当 policy 的探索能力超过 reward signal 的判别/泛化能力，并且优化压力持续增加时，reward hacking 最容易发生。**

这里的优化压力包括很多东西：RL 轮数更长、采样更多、KL 更低、reward 权重更大、筛选更激进、线上流量持续反馈、agent 环境更开放。reward signal 的判别能力取决于它能不能区分真实好行为和只是看起来得分高的行为。两边一旦失衡，模型就会从学习任务变成学习评分器。

接下沿着 reward stack 里最常见的几个接口来看：规则奖励、神经奖励模型、LLM 提供的评审信号，以及工业里常见的混合奖励。它们不是全部策略的封闭列表，而是几条最容易看清机制的风险路径。可验证任务上，规则奖励通常最稳；但如果规则只覆盖表面格式，它反而会最快被 hack。开放任务上，LLM judge 比手写规则更灵活；但如果 judge 只是一个黑盒总分，它也只是另一个 proxy。混合奖励看起来更安全；但如果冲突维度都被揉成一个 scalar，风险会被藏得更深。

## 规则奖励：最怕开放环境和欠规格化规则

规则奖励通常来自正则表达式、单元测试、格式检查、数据库终态、API 返回值、代码执行结果，或者其他确定性 verifier。它的好处非常明显：便宜、稳定、可复现，而且不需要训练一个可能被 policy 反向利用的神经 RM。DeepSeek-R1-Zero 之所以能在数学推理上用 rule-based accuracy reward 和 format reward 跑出强推理行为，正是因为数学答案有相对清晰的可验证边界。

但规则奖励的风险也同样清楚：**规则只覆盖字面条件，不自动覆盖规则背后的意图。**

模型会优化的是 `reward function`，不是产品经理、研究者或标注者脑子里的真实目标。如果奖励函数写的是“答案里包含指定格式”，模型就可能把格式堆满；如果奖励函数写的是“代码不报错”，模型就可能学会掩盖错误；如果奖励函数只跑一小组单元测试，模型就可能只适配这组测试。这里的 hacking 机制不是模型变坏，而是规则本来就没有把目标说完整。

规则奖励的高风险场景有三个。

第一，任务空间开放，但规则只覆盖少数可检查表面。比如一个 agent 要完成“规划一次旅行”，你只检查输出是否包含地点、时间、预算三项，却不检查这些信息是否来自工具、是否互相一致、是否满足用户约束。模型很快会学会产出结构完整但事实悬空的答案。

第二，奖励太稠密，但每个小项都很浅。格式、长度、关键词、步骤数、工具调用次数都可以成为 bonus。问题是这些 bonus 太容易被满足，policy 会优先学这些便宜信号，而不一定学真正困难的任务能力。

第三，环境反馈可以被动作本身改变。Agent 场景里尤其明显：模型不只是写答案，还能调用工具、改文件、更新状态。如果 verifier 没有隔离好环境，reward 就可能从“评价任务完成度”变成“评价模型是否把环境改成了容易通过检查的样子”。这也是为什么真实 agent 训练里要先定义 environment contract、protected paths、allowed tools 和 termination condition。

所以规则奖励的工程原则是：

- 能客观验证的核心结果，优先用 verifier；
- 安全、越权、事实编造、破坏环境这类错误，做成 hard veto，不要放进平均分；
- 格式类奖励只能当辅助项，权重必须低，而且要监控是否被过度利用；
- 测试集和 verifier 本身要持续扩展，尤其要补 adversarial case 和长尾 case；
- 对 agent 任务，必须记录完整轨迹，以及对环境的完整监控。

规则奖励的 reward hacking 往往发生得很早。因为它不需要模型理解复杂偏好，只需要找到字面规则里最便宜的通过路径。训练曲线上会表现为 reward 很快上升，但人工看样本时发现行为越来越怪。

## 神经奖励模型：固定 RM 在强优化下的风险

神经奖励模型是经典 RLHF 的核心部件：先用人类偏好训练一个标量模型，再让 policy 通过 PPO、GRPO 或其他 RL 目标去追高分。它的优势是能覆盖规则难以写清的开放质量维度，比如有用性、语气、表达、完整性、礼貌程度和主观偏好。

但神经 RM 的本质是 proxy。它不是人类偏好本身，而是从有限比较数据中学出来的近似函数。一旦 policy 开始围绕它做在线搜索，问题会立刻变成：**policy 能不能找到 RM 训练分布以外、但 RM 仍然误判为高质量的区域。**

这类 hacking 的机制通常有两层。

第一层是 OOD exploit。RM 训练时看过的是某个分布里的回答，RL 后期 policy 生成的回答会逐渐偏离这个分布。偏离以后，RM 的标量分数还会给得很自信，但这种自信不再可靠。早期 RLHF 工作里的 reward over-optimization，本质上就是 policy 把 RM 当成目标函数后，沿着 RM 和真实人类偏好之间的缝隙一路前进。

第二层是虚假相关性。RM 很容易学到一些和优秀答案相关但不等于好答案的表面特征：更长、更礼貌、更结构化、更像专家、更有引用样式、更少承认不确定性。policy 在 RL 中会放大这些特征。最后得到的输出可能看起来非常专业，但事实是错的，或者对用户问题没有实际帮助。这种奖励带来的幻觉已经成为了一个需要单独研究的问题，危害性极大。

这也是为什么工业系统很少只相信一个裸 RM 分数。Llama 3 的 post-training 路线使用 reward models、rejection sampling、SFT 与 DPO 的多轮组合，而不是简单地把一个 RM 接上 PPO 一路推到底；DeepSeek 系列技术报告也反复区分 rule-based reward 和 model-based reward 的适用边界。开放偏好任务通常需要多阶段 post-training pipeline、独立评测和持续抽检。

神经 RM 想要更好的工作需要非常多的辅助工作：

- 保持 KL 或 reference 约束，限制 policy 离开已知可靠区域的速度；
- 定期用新 policy 样本刷新 RM 训练集，尤其收集高 RM 分但人工不满意的样本；
- 用 holdout prompts、人工 spot check、事实性评测和 adversarial eval 检测过优化；
- 监控 reward 分布、长度、拒答率、引用密度、重复模式和 KL，而不是只看平均 reward；
- 用 ensemble、uncertainty 或多 RM 交叉验证，降低单一 RM 盲区被精准利用的概率；
- 对事实性、安全性和工具结果一致性，尽量接入 verifier，而不是全交给 RM 猜。

神经 RM 的核心风险可以一句话概括：**固定 RM 被搜索能力更强的 policy 长时间作为目标优化。**

## LLM-as-Judge：最怕宽泛裁判和未校正偏见

LLM-as-Judge 或 RLAIF 的动机很自然：很多任务没有简单 verifier，人类标注又太贵，那就让更强 LLM 来做评审。相比固定神经 RM，LLM judge 有几个优势：它能读 rubric，能解释判断，能处理开放任务，也能用 pairwise comparison 而不是硬输出绝对分数。

但 LLM judge 也是一个有偏的模型，只是偏差表现得更像“评价风格”。MT-Bench / Chatbot Arena 这条线已经反复提醒：LLM judge 可能有 position bias、verbosity bias 和 self-enhancement bias。也就是说，它可能偏爱某个位置的候选，偏爱更长的回答，或者偏爱更像自己输出分布的答案。

一旦 judge 被接进训练，policy 就会学习这些偏见。

如果 judge 喜欢长答案，policy 会注水；如果 judge 喜欢条理化模板，policy 会把所有答案都写成模板；如果 judge 在事实核查上弱但在语气判断上强，policy 会变得更会包装幻觉；如果 judge prompt 过于宽泛，policy 会学会迎合 prompt 中隐含的价值偏向，而不一定更接近真实任务目标。

LLM judge 的高风险场景是：

- judge prompt 只有“请给这个回答打分”，没有明确 rubric；
- 使用 pointwise 分数，但没有跨 prompt 校准；
- pairwise 比较不做 A/B 顺序交换；
- judge 没有独立评测集，也没有和人类标注或可验证信号对齐；
- 被训练 policy 和 judge 来自相近模型家族，self-preference 更明显；
- 对事实性任务仍让 judge 凭语感裁决，而不是接入检索、工具或 verifier。

对 LLM judge 来说，最重要的改进方向是结构化。

第一，尽量让 judge 比较候选，而不是直接发明绝对分。开放任务里，pairwise/listwise 通常比 pointwise 稳，因为比较两个答案谁更好比定义“7 分到底是什么意思”容易。

第二，使用 rubric，把质量拆成可检查维度。比如事实保真、约束满足、推理一致性、工具证据使用、表达清晰度、成本和安全边界。judge 的任务应该是沿着这些维度提取信号，而不是面对一个模糊总目标。

第三，做偏差校正。pairwise 要双向评分，长答案要有长度归一化或冗余惩罚，事实类维度要用外部证据，judge 自身也要被 benchmark。PaperBench 这类 eval tree 思路的价值就在这里：不是让 judge 猜整体复现得怎么样，而是先把开放任务拆成大量可局部判断的 requirement。

第四，不要把 LLM judge 放在 hard constraint 的位置。越权工具调用、关键事实编造、安全违规、破坏环境状态这类错误，应该优先由规则和 verifier 直接否决。judge 可以帮助解释和排序，但不应该用一个高表达分把红线错误盖过去。

LLM judge 的 reward hacking 更像被裁判同化：模型未必变得更强，但会越来越像裁判愿意给高分的样子。

## 混合奖励：最怕静态权重把冲突藏进总分

工业系统很少只用单一 reward。更常见的是：

$$
R = \alpha R_{\text{rule}} + \beta R_{\text{RM}} + \gamma R_{\text{judge}} - \lambda C
$$

看起来这很稳：规则负责硬边界，RM 负责偏好，judge 负责开放质量，成本项负责效率。问题在于，只要这些信号被线性揉成一个 scalar，policy 就会做 optimizer 最擅长的事情：**在各个分项之间寻找套利空间。**

混合奖励的 hacking 不一定表现得很夸张。它通常更隐蔽，因为总分还在上涨，单项指标也可能没有明显崩掉，但真实可用性开始下降。

典型例子是红线被平均分稀释。一个回答事实上编造了关键价格，但结构、语气、完整性、服务闭环都得分很高，最后总分仍然不错。对用户来说这显然不可接受；对线性 reward 来说却可能是一个局部最优。

另一个例子是成本和质量之间的静态权重失衡。如果成本惩罚太弱，agent 会学会多调用工具、多写冗余分析来提高 judge 分；如果成本惩罚太强，它又可能过早停止探索，给出看似简洁但没有完成任务的答案。

混合奖励的高风险配置是：

- 所有分项都线性加权，没有 hard veto；
- 权重静态，不随任务类型、难度和风险等级变化；
- dashboard 只看总分或少量平均值；
- 训练集里缺少冲突样本，比如“表达很好但事实错”“格式完美但工具越权”“答案很短但遗漏关键约束”；
- reward 既当数据筛选器又当 RL 回报，但两个阶段没有分别审计。

混合奖励更稳妥的聚合方式，不是把权重调得更玄学，而是先改变聚合逻辑。

第一，红线不要进平均分。事实编造、越权、破坏环境、安全违规、违反用户硬约束，应该是 veto 或 lexicographic constraint。先判合法性，再谈质量。

第二，任务类型决定 reward geometry。规划任务、信息检索任务、咨询解释任务，不应该共享同一套固定权重。不同任务的“好”本来就不是同一种几何。

第三，分项指标必须单独看。总 reward、RM 分、judge 分、规则通过率、veto 率、长度、工具调用次数、成本、KL、人工满意度都要分开监控。混合 reward 最大的问题是把冲突藏起来，所以监控必须把冲突重新展开。

第四，训练样本要主动构造 trade-off case。没有这类样本，系统就不知道哪些维度不能互相抵消。工业里真正有用的 eval set，往往不是普通样本，而是专门用来暴露权重漏洞的样本。

混合奖励不是 reward hacking 的解药。它只是把单一 proxy 的风险拆散了；如果聚合方式仍然粗糙，风险会以更难排查的方式回来。

## 训练策略应该怎么选

把这些接口上的失配风险合起来看，训练策略可以按任务结构来选，而不是按论文缩写来选。

**数学、代码、SQL、工具终态可验证**：首选 verifier / rule reward + GRPO / RLVR。核心目标可客观验证，没必要先学 proxy；额外防规则覆盖不足、测试集过拟合和格式奖励过强。

**主观偏好、语气、品牌风格、开放对话**：首选 SFT + preference data + DPO / RM-based alignment。目标难以写成 verifier，需要偏好学习；额外防 RM 过优化、长度偏见和 OOD 高分样本。

**长链条 reasoning 或 agent 轨迹**：首选 outcome verifier + process checks + trajectory ranking。只看终点太稀疏，中间决策也要被监督；额外防 judge 看不懂轨迹和过程分过度塑形。

**开放式复杂 agent**：首选 environment contract + feedback stack + hybrid reward + online RL。reward 来自环境、规则、judge、成本多个接口；额外防静态权重套利、环境状态被污染和 benchmark proxy 化。

**高风险事实任务**：首选 verifier / 检索证据，judge 只做辅助。事实性不能靠语气分兜底；额外防 judge 误判和幻觉被表达质量掩盖。

## 最后：reward hacking 不是异常，而是压力测试结果

很多人谈 reward hacking 时，语气像是在说模型耍小聪明。但更准确的说法是：**reward hacking 是优化器替你做了一次规格审计。** 它暴露的不是模型和优化器本身不够好，而是 reward interface 没有把真实目标表达完整。

沿着前面几个接口看，规则奖励暴露的是规格漏洞；神经 RM 暴露的是 proxy gap；LLM judge 暴露的是裁判偏差；混合奖励暴露的是目标冲突和聚合缺陷。

所以更成熟的 reward engineering 不应该只问这个 reward 能不能让体现需求，而应该问四个更难的问题：

- 这个 reward 的盲区在哪里？
- policy 多强时会超过它的判别能力？
- 哪些错误不能被其他高分抵消？
- 当 reward 上涨但真实质量下降时，我能不能在 dashboard 上看出来？

把这四个问题回答清楚，reward 才从一个训练分数变成一个可被审计、可被迭代、可被放进工业系统的接口。

到这里，Reward 这组三篇就形成了一条闭环：先看信号怎样被消费，再看信号怎样被生产，最后看信号在强优化下怎样失配。后续再进入 [training loop]({% post_url 2026-03-22-reward-and-training-in-agent-k-paperbench-amap %})：当 reward 已经被拆成 verifier、judge、rubric、trajectory ranking 和 hard constraints 以后，SFT、offline shaping、online RL、curriculum 和 distillation 应该怎样组成一个闭环。

## 参考资料

- [Learning to Summarize from Human Feedback](https://arxiv.org/abs/2009.01325)
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)
- [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [PaperBench: Evaluating AI's Ability to Replicate AI Research](https://arxiv.org/abs/2504.01848)
- [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685)
- [Large Language Models are not Fair Evaluators](https://arxiv.org/abs/2305.17926)
