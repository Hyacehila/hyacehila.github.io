---
layout: blog-post
title: Reward 与 Training 在真实 Agent 中如何闭环：从数据治理到在线 RL
date: 2026-03-22 20:30:00 +0800
series: Reward and Training
categories: [训练与对齐]
tags: [Reward Modeling, Data Curation, Evaluation, Alignment]
author: Hyacehila
excerpt: 这一篇不再把论文逐篇串讲，而是按真实系统的训练流水线来重写：从 3000 万历史 query 的数据治理，到工具环境、reward 与 verifier 设计，再到 verified trajectories、SFT、curriculum、online RL 与 benchmark audit。
featured: false
math: true
---

# Reward 与 Training 在真实 Agent 中如何闭环：从数据治理到在线 RL

前两篇系列文分别在讲两件事：一篇讲 reward 自己是怎样被生产出来的，另一篇讲这些 reward 进入训练以后，又怎样一路变成 `advantage`、`update` 和 agentic RL 的闭环。但只讲到那里其实还不够，因为读者最容易卡住的问题并不是“某个公式怎么推”，而是更工程化的那个问题：**真实 agent 系统到底从哪里开始长出来？**

答案通常不是“先写一个奖励函数”。更真实的顺序是：先有原始数据和历史 query，再有数据治理和环境构造；环境跑起来以后，系统才会产生日志、tests、数据库状态、工具输出和用户交互；这些反馈再被组织成 verifier、judge、rubric 或 outcome reward；接着才轮到 verified trajectories、SFT、curriculum 和 online RL；最后 benchmark 还要回过头来审计，我们到底有没有在优化正确的目标。

换句话说，reward 在真实系统里不是一个孤零零的分数，而更像一条生产线。为了把这条线讲清楚，我这里不再按论文一篇篇做摘要，而是按系统流水线来重写全文。主线由 `AMAP` 提供，因为它最接近从数据治理一直写到 RL 的完整 recipe；`Agent K`、`MLE-bench`、`tau-bench`、`PaperBench` 和 `Benchmark^2` 则分别嵌到这条链的不同环节，负责回答环境、评估结构、部署目标和 benchmark 审计的问题。

如果读完这篇文章以后，读者能够顺着这条线把从数据处理开始的 reward 与 RL training 全流程复述出来，那这篇文章就算达到目的了。

## 为什么真实系统不是从“一个奖励函数”开始

很多关于 RL 的讨论之所以容易飘，是因为默认系统里先有一个奖励函数，后面的所有工作都只是围绕这个函数做优化。但在真实 agent 里，这个顺序几乎总是反过来的。

因为模型不会直接面对目标本身，它只会面对一个被工程化以后可交互的世界。这个世界里有原始日志、有历史 query、有工具协议、有失败码、有用户反馈、有测试脚本、有数据库状态，也有 judge 和 verifier。reward 只是这整套世界在某个接口上的压缩表达。你如果还没有把数据、环境和评估结构造出来，其实连稳定的 reward 都谈不上，更不用说训练。

这也是为什么我更愿意把真实 agent 里的 reward 理解成一条 `feedback production pipeline`。这条 pipeline 的前半段负责把任务改写成环境、把环境反馈改写成可监督对象；后半段负责把这些对象筛成高价值轨迹，再写回 SFT 和 RL。很多论文在讲 trainer，真正决定上限的地方却往往发生在 trainer 之前，算法和训练确实很重要，但数据的重要性至少与他们等同。

## 各篇论文在这条流水线里的位置

先用一张表把几篇文献的角色钉住。后面展开时就不会再掉回“六篇论文六段摘要”的写法。

| 文献 | 在全流程里的位置 | 真正提供的东西 | 在本文里的篇幅定位 |
| --- | --- | --- | --- |
| [AMAP](https://arxiv.org/abs/2512.24957) | 数据治理 -> 环境 -> reward -> SFT -> curriculum -> RL | 最完整的 agentic training recipe | 主线 |
| [Agent K](https://arxiv.org/abs/2411.03562) | 环境化与 feedback stack | 复杂任务怎样先被改写成可学习环境 | 重点辅助 |
| [MLE-bench](https://arxiv.org/abs/2410.07095) | benchmark curation 与任务标准化 | ML engineering 任务怎样变成公共评测对象 | 辅助 |
| [tau-bench](https://arxiv.org/abs/2406.12045) | 部署级交互环境与 outcome reward | 用户-工具-规则-数据库交互怎样被写成客观奖励 | 重点辅助 |
| [PaperBench](https://arxiv.org/abs/2504.01848) | judge 依赖的评估结构 | 开放研究复现任务怎样拆成 rubrics / eval tree | 辅助 |
| [Benchmark^2](https://arxiv.org/abs/2601.03986) | benchmark 审计 | benchmark 自己是否稳定、可区分、对齐能力层级 | 收束辅助 |

这张表也解释了一个后面会反复出现的判断：**不是每篇论文都在直接讲训练配方。** 有些论文主要在定义环境，有些在定义 benchmark，有些在审计 benchmark。它们都重要，但重要的方式不一样。也正因为如此，后面关于数据治理与训练治理的主体篇幅会明显向 `AMAP`、`Agent K` 和 `tau-bench` 倾斜，而 `MLE-bench`、`PaperBench`、`Benchmark^2` 会更多承担“约束训练目标”的角色。

## 一、从原始数据到可训练 query pool：数据治理与环境准备

### 1. AMAP 先解决的不是 RL，而是 3000 万历史 query 怎么变成训练数据

如果只看二手介绍，很多人会把 `AMAP Agentic Planning Technical Report` 读成一篇地图 agent + RL的文章。但直接读原文以后，很难再这么理解。因为 AMAP 最先解决的，不是 optimizer，而是**数据治理**。

论文把真实世界里的匿名化用户日志作为起点，时间窗是三个月，总量超过3000 万。问题在于，这些日志并不是训练数据，而只是原始世界留下来的痕迹：它们噪声很大、重复很多、分布极不均衡，而且完全没有现成的难度标签、任务类型标签或是否适合 agent 学习的标记。直接拿来训练，只会把噪声和头部高频需求一起喂给模型。

所以 AMAP 做的第一件事，是为这些原始 query 先造一套 `Intent Taxonomy`。这个 taxonomy 不是随便列几个类目，而是一个分层的意图框架：论文里最后收敛成 `5` 个一级类目（Rules and Policies、Discovery、Planning and Decision、Dynamic Information、Application Interaction）、`16` 个二级类目和 `30` 个细粒度叶节点。更重要的是，这套 taxonomy 不是拍脑袋生成的，而是论文所说的 `Seed-Driven Evolutionary Framework`，一步一步长出来的。具体来说分四个阶段：

**Stage 1: Seed Initialization。** 人工挑选一小批高方差的种子 prompt（$D_{seed} \subset D_{pool}$），由专家为每条 query 标注开放式的意图标签。每条 query 被映射成一个元组 $S_i = \langle q_i, T_i \rangle$，其中 $T_i = \{t_{i,1}, t_{i,2}, \ldots, t_{i,k}\}$ 是这条指令覆盖的 $k$ 个正交或互补的意图节点。这一步的关键不是数量，而是让种子集尽可能覆盖领域的核心多样性。

**Stage 2: LLM-Driven Category Induction。** 用标注好的种子集去 prompt LLM，让它归纳出正交的一级类目（Level-1 categories）。LLM 在这里扮演的角色是"从具体样本中抽象出结构"，而不是凭空编造分类体系。

**Stage 3: Iterative Refinement Loop。** 为了防止 LLM 归纳时产生幻觉或遗漏，论文设计了一个严格的 `Tag-Feedback-Correction` 循环：LLM 用当前版本的 taxonomy 重新标注种子集，人类专家审查标注结果，识别歧义和覆盖缺口，再把反馈喂回 LLM 修正 taxonomy。论文把这个过程写成 $T^{(k+1)} = \text{Refine}\big(\text{Annotate}(D_{seed}, T^{(k)}),\; \text{Human Feedback}\big)$，循环持续到 $T^{(k+1)} \approx T^{(k)}$，也就是 taxonomy 收敛为稳定结构 $T^*$。

**Stage 4: Dynamic Taxonomy Expansion。** 前三步只能覆盖种子集能触及的意图空间。为了捕捉长尾意图，论文在每个节点强制保留一个 `Other` 类目。当大规模原始日志被标注后，落入 Other 的样本会被单独分析，从中发现并合并新兴类目。这个机制把 taxonomy 从静态树变成了一个能适应开放世界 query 分布的演化系统。

也就是说，数据治理在这里并不是简单清洗，而是**先把世界切成可管理的意图空间**（俗话说就是先给样本分个类），而且这个分类体系本身就是通过"种子驱动 + LLM 归纳 + 人类校正 + Other 动态扩张"迭代生长出来的。

这一步为什么关键？因为训练真正需要的不是更多数据，而是可控的数据分布。没有 taxonomy，你只能看到一个模糊的 query 海洋；有了 taxonomy，你才能谈覆盖度、长尾比例、困难样本、约束密度，以及哪些区域的监督更稀缺、哪些区域更值得扩充。

### 2. 高质量数据治理不是去重，而是同时保留覆盖度与难度谱系

Taxonomy 稳定以后，AMAP 又做了两类更细的治理。论文把这两步合称为 `Annotation and Data Curation`。

**第一类是多维标注（Precise Multi-Dimensional Annotation）。** 论文把意图理解显式建模为一个多标签分类任务。每条用户指令被映射成一个复合标签向量 $V = \langle I_{primary}, I_{secondary}, C_{constraints} \rangle$。其中 $I_{primary}$ 和 $I_{secondary}$ 对应 Intent Taxonomy 里的叶节点（注意是叶节点，不是一级类目），$C_{constraints}$ 则捕捉辅助约束维度，例如空间范围、时间预算、交通工具偏好等。这样做的意义在于，模型以后不是只学”这是一条导航请求”，而是学”这是一条什么主任务、带着哪些附加约束、需要怎样的工具链”。论文用一个高容量的 teacher LLM 在结构化日志上执行大规模标注，具体的 prompt template 放在了附录 A.2。

**第二类是分层过滤（Controlled Sampling via Funnel Filtering）。** AMAP 设计了一个三级漏斗过滤策略，在词汇、语义、几何三个层面系统性地消除冗余：

- **词汇层（Lexical Redundancy Elimination）：** 在全语料级别做 `Locality-Sensitive Hashing`，高效去掉近重复字符串和字面垃圾。这一步的目标是大幅压缩初始体量，把明显的重复和噪声先清掉。
- **语义层（Semantic Redundancy Elimination）：** 按 $\langle I_{primary}, I_{secondary} \rangle$ 元组把数据分桶，在每个桶内做 embedding 相似度搜索，裁剪掉语义冗余样本——也就是那些措辞不同但在隐空间里距离低于阈值的 query。论文特别提到，为了处理这个规模的数据集，他们用 `Faiss` 加速搜索，把计算复杂度从 $O(N^2)$ 的暴力配对降到了亚线性或近线性。这一步保证的是桶内方差：同一个意图类目下，留下来的样本之间要足够不同。
- **几何层（Geometric Redundancy Elimination）：** 从语义去重后的池子里，用 `K-Center-Greedy` 算法选出最具代表性的样本。这个算法的核心思想是最大化已选样本之间在 embedding 空间里的最小距离，效果是优先保留那些分布在边界和长尾区域的 query。也就是说，它不是简单追求更少的数据，而是在**减少冗余的同时保住分布的形状**——尤其是那些对 agent 鲁棒性至关重要的 corner case。

这一套治理以后，AMAP 从 `3000 万+` 原始历史 query 里筛出大约 `20 万` 候选 query，保留比例不到 `1%`。这正是为什么我前面一直说，真实 agent 训练从来不是”数据很多就行”。在这里，数据治理本身已经是训练 recipe 的一部分。

### 3. 难度标签和负样本不是附属品，而是后续 curriculum 的地基

AMAP 还有一个经常被忽略、但对理解完整流程非常重要的设计：它不只做意图标注，还做难度标注。论文把这部分称为 `Difficulty and Diversity`，核心动机很明确——静态数据分布会导致 RL 训练效率低下：早期模型面对过难的任务只能拿零 reward（梯度消失），晚期模型面对过简单的任务只能拿满分（方差消失）。为了解决这个问题，AMAP 设计了一套 `Execution-Simulation Scoring Mechanism`，从三个正交维度评估难度：

- **Cognitive Load in Tool Selection（工具选择的认知负载）：** 衡量意图映射的模糊程度。从显式映射（Score 1-2，比如”导航到 X”）到隐式推理（Score 4-5，比如”找一个开车 20 分钟内、适合安静读书的地方”，需要抽象意图分解）。
- **Execution Chain Depth（执行链深度）：** 量化解题路径的逻辑复杂度，追踪工具调用的数量、类型和依赖深度（顺序执行 vs 并行执行）。
- **Constraint Complexity（约束复杂度）：** 评估空间、时间、偏好等约束的密度，以及 agent 需要联合优化到什么程度。

论文用一个强 teacher LLM 作为自动评估器，对结构化用户日志按这三个维度打出一个标量难度分数 $r \in \{-1, 0, 1, \ldots, 5\}$。其中 `-1/0` 有特殊用途：`-1` 表示不可执行（Unexecutable），`0` 表示不需要工具（No Tool Needed）。

更值得注意的是，AMAP 专门构造了一个 `Irrelevance Dataset`，用来教模型识别”这件事超出工具边界”或”应该拒绝瞎编”的场景。这些 Score -1 和 0 的样本在 System Control 等领域有明显占比。这一点非常值得记住，因为它说明在真实系统里，**负样本治理也是数据治理的一部分**。如果训练集只有可完成任务，模型通常学不会边界感。

### 4. MLE-bench 的启发：benchmark 也要先经过数据治理

把这个逻辑放大一点看，`MLE-bench` 做的其实是同一件事在 benchmark 层的版本。它从 Meta Kaggle 的 `5673` 个已结束竞赛出发，经过社区竞赛过滤、人工筛选代表性和可复现性，最终只留下 `75` 个测试任务。每个任务还标了 problem type、复杂度分级（经验 ML engineer 需要 <2h / 2-10h / >10h），并用 Kaggle Private leaderboard 的 medal logic 把分数重新绑回可解释的工程水平门槛。

这件事给我们的启发很直接：**不只是训练数据需要治理，benchmark 数据也需要治理。** benchmark 一旦没被整理成稳定的任务合同，后面的分数就不会有解释力。MLE-bench 在本文里不承担训练主线，但它和 AMAP 的数据治理共享同一个原则：先把任务空间结构化，再谈评估和优化。

## 二、先把环境做出来，再谈 reward

如果说数据治理回答的是什么值得学，那么环境构造回答的就是“模型到底在哪个世界里学”。

### 1. AMAP：训练 world model 之前，先把 tool world 搭稳

AMAP 的 environment 部分很像很多人以为不够学术、但在工程里其实最关键的那种工作。论文里先搭了一个高保真的工具环境，覆盖地图、旅行、天气、信息检索四大类、总共 `10` 个专门工具；再用 `FastMCP` 把工具调用协议统一起来；然后把训练基础设施接上去，支持异步 rollout 和异步训练。

这些设计听起来像基础设施细节，但它们实际决定了后面的 RL 能不能发生。因为对 tool-integrated reasoning 来说，环境不是抽象容器，而是奖励和 credit assignment 的生成器。只要 tool 返回慢、协议不稳、轨迹同步不好，后面的 reward signal 就会塌。AMAP 之所以值得拿来讲全流程，恰好是因为它没有把 environment 当成默认存在的黑箱，而是把它明明白白写进了训练 recipe。

### 2. Agent K：复杂任务必须先被 scaffold 成可操作 workspace

`Agent K` 则从另一个角度说明了同一件事。它处理的是 Kaggle 这类极其 messy 的开放数据科学任务，但它并不是直接让模型去“冲分”，而是先把任务拆成 `Workspace Scaffold` 和 `Solution Generation Scaffold` 两层。

在 `Workspace Scaffold` 阶段，agent 先要搞清楚输入输出结构、任务类型、metric、submission format 和 workspace 组织方式。这一步表面上还没进入建模，实际上已经在做最关键的事：把原本杂乱的开放任务，改写成一个可执行、可验证、可多轮修正的工作区。

只有工作区先被搭出来，后面的 `Solution Generation Scaffold` 才有地方落脚：特征工程、模型选择、训练、迭代、调试、再次提交，才不会变成一堆漂浮在 prompt 里的意图。Agent K 给我们的最重要启发，不是“LLM 可以做 Kaggle”，而是**复杂任务一定要先被 scaffold 成环境，反馈才会变得有结构**。

### 3. tau-bench：真实部署环境里，agent 同时面对用户、规则和数据库

`tau-bench` 把环境问题又往现实业务推了一步。它不是只给模型一个任务说明，而是把任务写成一个多轮的 `tool-agent-user` 交互世界：数据库状态是隐藏的，agent 可以通过 API 去读写数据库，同时还要和 LM 模拟的用户反复对话，并且始终遵守 domain-specific policy 文档。

这点特别重要，因为它说明真实环境不是“任务 + 工具”这么简单，而是至少包含四层：用户、工具、规则文档、底层状态。你在这样的环境里训练和评估 agent，真正要学的就不只是“选对一个函数”，而是如何持续补全信息、在多轮里维持上下文、在规则约束下调整路径，并且在必要时拒绝用户请求或者给出替代方案。

也就是说，AMAP、Agent K 和 tau-bench 虽然领域不同，但都在告诉我们同一个事实：**没有环境合同，就没有稳定 reward；没有结构化环境，也就没有可解释的训练。**

## 三、reward / verifier / judge 怎样把目标改写成可监督对象

环境搭好以后，系统才真正进入 reward production 的环节。这里最容易产生误解的一点是：reward 不一定是一个直接拿来优化的总分。很多时候，reward 首先是评估结构。

### 1. AMAP：rubrics-as-reward 不是黑盒打分，而是先拆维度

AMAP 的 reward design 很适合拿来当主例，因为它把 rubric 真正写成了训练接口。论文把 agent 轨迹质量拆成三个维度：

- `Reasoning and Proactive Planning`
- `Information Fidelity and Integration`
- `Presentation and Service Loop`

这个拆法很有代表性。第一维评估 agent 有没有形成经济、有效、主动的执行计划；第二维看它能不能忠实地抽取和整合工具输出；第三维看它是不是把服务闭环真正做完，能不能结构化、清晰地把结果交付给用户。

更关键的是，AMAP 没有给这三维一个固定死权重，而是先按任务类型重新分配权重。复杂规划类任务更重推理和主动性；信息检索类任务更重事实保真；咨询解释类任务更重交付表达和服务闭环。这一步非常关键，因为它意味着系统承认了一个经常被忽略的事实：**不同任务的好答案，本来就不共享同一套固定 reward geometry。**

最值得单独记住的，则是它对 hallucination 的 `hard veto`。论文写得很清楚：只要出现不能由工具输出支撑的事实编造，比如时间、价格、距离等信息幻觉，最终 reward 直接归零。也就是说，AMAP 明确把绝不能犯的错从平均分逻辑里剥离出来了。这个设计对真实 agent 极其重要，因为有些错误本来就不应该靠别的地方表现好一点来抵消。

### 2. tau-bench：有些任务不需要 judge，总结态本身就是 reward

如果说 AMAP 展示的是 rubric reward，那么 tau-bench 展示的就是另一种很重要的写法：`outcome reward`。

它的 episode 结束以后，系统主要看两件事：

- 最终数据库状态是否与唯一正确的目标状态一致
- agent 回答给用户的必要信息是否齐全

这种写法看起来比 rubric 更朴素，但它的意义恰恰在于说明：**并不是所有开放交互任务都必须靠黑盒 judge。** 只要任务设计得足够好，把终态 outcome 写清楚、把必要信息项定义清楚，很多关键反馈仍然可以是客观可验证的，在条件允许时 RLVR 仍是最可靠的思路。

tau-bench 也因此把部署级目标写得非常明确。代码生成世界里常见的是 `pass@k`，也就是跑 k 次至少有一次成功；而 tau-bench 提出的 `pass^k` 则问的是：跑 k 次能不能每次都成功。对真实客服、零售、预订、规则遵循类 agent 来说，后一种指标才更接近部署要求。它测的不是“能不能偶尔解出来”，而是“能不能稳定、守规矩地解出来”。

### 3. PaperBench：开放任务里，judge 能稳定工作的前提是先有 eval tree

`PaperBench` 则把开放任务怎样被 judge 评出来这件事写得非常系统。它评的是研究工程复现，目标天然比普通代码题开放得多，所以论文没有直接让 judge 给总印象分，而是先把每篇论文的复现要求拆成一棵层级化的 rubric tree。整个 benchmark 最终覆盖 `20` 篇论文，并拆出 `8316` 个可独立评分的要求。

更重要的是，它把叶节点明确分成三类：

- `Code Development`
- `Execution`
- `Result Match`

这个拆法非常关键，因为它把三个经常被混成一句复现成功的过程强行拆开了：代码有没有写对、脚本有没有跑起来、结果是不是足以支撑论文结论。PaperBench 的核心启发并不是LLM judge 已经很强，而是**开放任务里 judge 的前提永远是先把任务对象拆成部分可判、可累积、可增量得分的结构**。

它后面再做 `JudgeEval`，其实也在说明同样的原则。论文里最佳 judge 在 JudgeEval 上做到约 `F1=0.83`，这个结果并不是要宣告 judge 无误，而是要提醒我们：judge 也是 proxy，它自己也必须被 benchmark。

到这里，reward / verifier / judge 之间的关系其实已经很清楚了：

- 能写成终态一致性的，尽量写成 objective outcome
- 无法直接写成终态的，尽量拆成 rubrics / eval tree
- judge 不应该直接面对一个模糊总目标，而应该面对结构化 requirement

## 四、verified trajectories 怎样生产、筛选、回流到 SFT

到这一步，系统里已经有数据、有环境、有 reward 结构。接下来真正重要的问题就变成：**训练数据到底是怎么被生产出来的？**

### 1. AMAP：reward 在这里首先不是梯度，而是数据闸门

AMAP 在这一步给出的 recipe 非常完整。它先从前面整理好的高质量 prompt pool 出发，再做两路 data construction。

第一路是离线候选轨迹生成。论文明确写到，它用强模型 `DeepSeek-R1` 对每个 query 生成 `K=8` 条候选 tool-integrated reasoning 轨迹，再用 `Gemini-3-Pro-Preview` 作为 verifier，按照前面那套三维 reward 去打分。最关键的一句是：**只有在所有维度上都拿到满分的轨迹才会被保留。**

这意味着 reward 在这里首先不是 policy gradient 的输入，而是训练数据的准入条件。换句话说，它先决定什么值得学，然后才决定怎样学。这是获取冷启动数据的一个很好的思路。

第二路是 long-tail 数据合成。因为真实分布里高频任务总比复杂规划任务多得多，所以只靠历史 query 回收，模型很容易学会大量导航、检索、短问答，却学不会稀有但高价值的多约束规划。为此，AMAP 会采样真实分布里罕见的复杂工具组合，再通过 ICL 让强模型反向合成必须调用这些工具链的用户 query。然后这些合成 query 仍然要回到同一条离线筛选管线里验证可执行性。

这件事非常重要，因为它说明真实 agent 的数据引擎通常不是回收日志这么简单，而是三件事同时发生：从真实世界回收、从强模型合成、再用 verifier 统一把关。

### 2. Agent K：复杂系统里的多源反馈，本来就适合先做筛选器

如果把这一步和 `Agent K` 放在一起看，逻辑就更清楚了。Agent K 真正依赖的从来不是一个单独标量，而是一整套 feedback stack：unit tests、meta-unit tests、执行日志、validation score、public leaderboard、private leaderboard。这些信号共同承担闸门、诊断、偏置和最终筛选的功能。

这和 AMAP 的 verifier filtering 虽然技术形态不同，但系统含义高度一致：**复杂 agent 的 reward 往往先长成过滤器，再长成优化器。** 换句话说，先决定哪些轨迹值得进入训练集，往往比先讨论它们怎么进入 loss 更重要。

### 3. 数据治理与训练治理在这里开始分叉

走到这一步，可以把整条线先切成两半。前半段偏数据治理，后半段偏训练治理。

| 数据治理 | 训练治理 |
| --- | --- |
| 造 taxonomy，定义任务空间 | 用 seed SFT 建立第一层行为先验 |
| 做去重、代表性选择、长尾补充 | 用 capability probing 衡量当前 policy 会什么、不会什么 |
| 标注意图、约束、难度、边界样本 | 用 signal/noise filtration 决定什么样本还值得继续学 |
| 用 verifier / tests / outcome reward 过滤轨迹 | 用 second-stage SFT 和 RL 接管不同难度区间 |
| 构造 irrelevance / hallucination 负样本 | 把 frontier tasks 留给 online RL |

这张表想说明的核心很简单：reward 与 training 的全流程不止包含训练，它前面还有一整段数据治理。很多系统之所以看起来 recipe 很强，实际差距往往来自它前面那一半做得够不够扎实。

## 五、SFT 与 课程学习

### 1. 多步工具轨迹的 SFT，到底在写什么

AMAP 接下来的设计非常适合用来纠正一个常见误解：SFT 不是 RL 前的热身，而是 agent 的第一层行为写入。

论文把 agent 交互建模成多步轨迹，而不是单轮答案。也就是说，训练对象不是“最终回复文本”这么简单，而是一个交替展开的序列：推理、工具调用、工具观察、再推理、再调用、再总结。模型真正需要学的，是这套多步节奏。

这也是为什么它在 SFT 目标里会特别处理 tool observations。论文里明确提到，工具观测文本不会直接贡献到最终 loss 里，等于说 observation 被当成上下文而不是监督对象。这是一个很好的工程细节：SFT 真正要写进模型的，是如何根据 observation 决策，而不是把 observation 本身背下来。

### 2. seed SFT：先把 policy 拉到一个可以被评估的区域

AMAP 的 SFT 不是一次完成，而是一个分阶段过程。第一阶段是 `seed SFT`。论文的做法是，从前面整理好的 prompt pool 里先随机抽一个 `10% Tiny Dataset`，用强模型生成 ground-truth trajectories，再用 best-of-8 + verifier 选出最优版本，先把 policy warm up 成一个初始策略。

这一步的系统意义非常明确：如果 policy 还弱到连像样轨迹都很难生成，那么后面的“难度估计”“可学性判断”“RL frontier”全都没有可操作性。seed SFT 的任务，就是先把模型拉到一个能自我暴露能力边界的区域。

### 3. capability probing：难度不是样本的静态属性，而是 policy 的相对属性

AMAP 接下来最有意思的一步，是它对 curriculum 的处理方式。论文明确反对把难度理解成一个静态标签，因为对强模型来说容易的任务，对当前 policy 来说可能根本还在分布外；反过来，对 teacher 来说太简单的任务，也可能仍然是 student 需要学习的决策边界。

所以它的做法不是先离线给所有样本分桶，而是先让当前 policy model 自己去 rollout。对每个 query，系统采样 `K=8` 条轨迹，再由 verifier 给出每条轨迹的 reward，进而得到这个 query 的经验均值 `μ̂` 和方差 `σ̂²`。

这一步里，均值和方差扮演的角色很不一样：

- 均值表示“这个任务对当前 policy 来说大致可不可解”
- 方差表示“当前 policy 在这个任务上稳不稳定，有没有可学但还没学稳的信号”

这也是为什么论文把 `σ̂²` 近似当成 uncertainty proxy。高方差但非零均值，通常意味着模型还没学稳，但已经能偶尔做对；这种区域恰好就是最有训练价值的区域。

### 4. signal-to-noise filtration：不是所有有标签的数据都值得学

基于上面的 probing，AMAP 把数据分成三类：

- `Trivial Region`：均值接近 1、方差接近 0，说明模型已经会了，再学只是过拟合
- `Noise Region`：均值接近 0、方差也接近 0，说明当前 policy 基本无能为力，强行学只会高偏差甚至负迁移
- `Learnable Region`：方差高且均值非零，说明任务正好落在当前模型的决策边界上

然后论文只保留 `Learnable Region`。为了量化这个值得学的程度，它还定义了一个 `Learnability Potential Score`，基本思想就是把非零可解性和不稳定性乘起来，优先保留那些既不是 trivially solved、也不是 hopeless noise 的样本。

这一步非常重要，因为它把数据治理真正升级成了训练治理。从这里开始，问题已经不再只是“这条数据好不好”，而是“这条数据在当前 policy 上有没有梯度价值”。

### 5. adaptive trajectory synthesis：第二层 SFT 其实是在修 decision boundary

有了 learnability score 以后，AMAP 还不会立刻结束 SFT。它会进入第四步，也就是 `Adaptive Trajectory Synthesis`。这里强模型被当成一个昂贵 oracle，系统会把更多采样预算分配给 learnability 更高的 query，最多仍然给到 `Kmax = 8` 次生成机会，以便更大概率恢复出一条高质量 reasoning path。

最后聚合这些 verified trajectories，系统再去决定最优混合比例，训练出后续 RL 的 backbone model。

从抽象层看，这就对应了第二层 SFT。它和 seed SFT 的差别并不只是再多训一次，而是角色不同：seed SFT 负责把模型拉进可训练区域；这第二层 SFT 则更像在**修 decision boundary**，把那些当前模型摇摆不定但其实可学的地方写稳。

## 六、online RL：什么时候才该把样本交给 RL

### 1. RL 不是接管一切，而是接管 frontier tasks

到了这里，很多系统会开始上 RL。但真正好的 recipe 不是数据来了就 RL，而是先把哪些问题应该留给 RL 说清楚。

AMAP 在宏观叙事上把这件事概括成：`seed SFT -> 再用更确定的样本稳住行为 -> 把低 certainty 的 frontier tasks 留给 RL`。如果把它和正文里的 capability probing 结合起来读，意思就非常清楚：RL 的职责不是替代全部 SFT，而是去吃那些已经被 reward、verifier 和前两轮 SFT 压缩过、仍然存在探索空间的困难样本。

这和很多人想象中的 RL 很不一样。RL 在这里不是万能细调器，而是一个专门处理长程交互、探索、恢复、决策时机和服务闭环这些静态 SFT 不擅长的问题的模块。

### 2. AMAP 的 RL：沿用同一套 reward 结构，而不是再造黑盒 RM

AMAP 的另一个优点，是它没有在进入 RL 时突然切换监督来源。online RL 用的还是前面那套 reward design：同样的三维 rubric、同样的动态权重、同样的 hallucination veto。也就是说，离线筛轨迹和在线优化共享同一套价值接口。

这点很关键，因为很多训练 recipe 最大的问题，就是离线数据和在线优化在学两件不同的事。AMAP 在这里尽量避免了这种断裂：reward 先当过滤器，后当 RL 的回报信号，但价值定义本身保持连续。

### 3. 算法层：GRPO 是形式，真正重要的是它在什么环境里优化什么

AMAP 在算法层使用的是 `GRPO`，并且在实现上使用 `GSPO` 来稳定训练。论文里的核心做法可以概括成三件事：

- 对每个 query 采样一组轨迹
- 用 reward 组内标准化得到 advantage
- 用 clip 和 KL 约束控制策略更新不要偏离 reference 太远

这些公式当然重要，但这篇文章更想强调的是另一件事：**算法名不是这篇论文的真正重点，重点是 RL 到底在什么环境里、用什么 reward、接管什么样的数据区间。** 如果没有前面的数据治理、环境构造、verified trajectories 和 curriculum 分层，单独把 GRPO 搬进去并不会自然长出一个强 agent。

### 4. RL 真正在这里学的是什么

如果从能力角度总结，AMAP 里的 RL 主要在学四件事：

- 什么时候应该继续探索，而不是过早停止
- 什么时候应该修正用户前提，而不是被动拒绝
- 什么时候需要多工具联动，而不是只做局部检索
- 什么时候应该把前面多步 observation 真正组织成服务闭环

也就是说，online RL 真正补的是 agent 在环境里的决策能力，而不是简单的文本风格或答案模板。

## 七、benchmark 怎样约束训练目标，而 benchmark 自己又怎样被审计

到这里，训练闭环其实已经大体成形了。但还有最后一层经常被忽略的问题：我们怎么知道自己真的在优化正确目标？

### 1. MLE-bench、tau-bench、PaperBench 分别在约束什么

`MLE-bench` 约束的是“完整 ML engineering workflow 能否被公共 benchmark 标准化”。它让我们看到，真正被评估的应该是 description、dataset、grader、leaderboard 一起定义出来的工程任务，而不是单轮答题能力。

`tau-bench` 约束的是“部署级成功到底是什么意思”。它用数据库终态、一致性和 `pass^k` 提醒我们：部署中的 agent 不该只追求至少成功一次，而必须追求稳定、守规矩地成功。

`PaperBench` 约束的则是“开放任务中的 judge 该怎么工作”。它提醒我们，研究复现这种开放任务不能直接交给一个总评 judge 去猜，而要先拆成 rubrics，再把复现进度分解为 `Code Development / Execution / Result Match`。

也正因为如此，这几篇论文在本文里不会承担和 AMAP 同等篇幅的数据治理讨论。原因不是它们不重要，而是它们的主要价值不在直接提供训练配方，而在**定义训练到底应该对齐什么目标**。

### 2. Benchmark^2：reward 是 proxy，judge 是 proxy，benchmark 也是 proxy

最后，`Benchmark^2` 把这个问题再往上提了一层。它问的不是“哪个模型更强”，而是“这个 benchmark 本身到底好不好”。论文提出了三个核心指标：

- `CBRC`：一个 benchmark 给出的模型排序，是否和同领域其他 benchmark 大体一致
- `DS`：这个 benchmark 是否真的能拉开不同模型
- `CAD`：题目层面是否经常出现“同一家族里更弱模型做对了、更强模型反而没做对”的反常 inversion

这个框架的重要性在于，它把 benchmark 从默认真理重新拉回到了 proxy 的位置上。前面几篇文章里我们已经反复看到：reward model 是 proxy，judge 是 proxy；而 Benchmark^2 只是把这件事再推进一步，告诉我们 benchmark 自己也是 proxy。

这也是为什么我会把 benchmark audit 放在整条流水线的最后。训练闭环不是在 policy update 那一刻结束的，而是在 benchmark 也被审计以后才算闭环。否则你很可能只是在一个脆弱评测器上制造了虚假的进步。

## 这篇文章最后想保留的判断

如果只允许我留一句话，我会写成：**真实 agent 里的 reward、verifier、judge、SFT、RL 与 benchmark 不是六个分开的模块，而是一条从数据治理、环境构造、轨迹筛选到训练更新与评测审计的连续系统。**

AMAP 给了我们这条系统里最完整的训练 recipe：从 3000 万历史 query 开始，经由 taxonomy、过滤、难度标注和负样本构造，变成 20 万级的候选 query pool；再通过强模型生成、verifier 评分和 perfect-trajectory 保留，变成 verified trajectories；然后先用 seed SFT 把模型带进环境，再用 capability probing 和 signal/noise filtration 重建 curriculum，最后把真正的 frontier tasks 交给 online RL。`Agent K`、`tau-bench`、`PaperBench`、`MLE-bench` 和 `Benchmark^2` 则分别告诉我们，这条链的前提是任务被环境化、目标被结构化、部署标准被写清楚、benchmark 也被当成 proxy 来审计。

不要把 reward 与 training 理解成两个词。对真实 agent 来说，它们更像同一条系统链的前后半段：前半段负责把目标翻译成可学信号，后半段负责把这些信号真正写回策略。只有把这整条链重新连起来，我们才算真正理解了奖励与训练在实践中的全流程。

## 参考资料

### 主线文献

- [AMAP Agentic Planning Technical Report](https://arxiv.org/abs/2512.24957)
- [Kolb-Based Experiential Learning for Generalist Agents with Human-Level Kaggle Data Science Performance](https://arxiv.org/abs/2411.03562)
- [MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering](https://arxiv.org/abs/2410.07095)
- [tau-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains](https://arxiv.org/abs/2406.12045)
- [PaperBench: Evaluating AI's Ability to Replicate AI Research](https://arxiv.org/abs/2504.01848)
- [Benchmark^2: Systematic Evaluation of LLM Benchmarks](https://arxiv.org/abs/2601.03986)

### 延伸阅读

- [OpenAI PaperBench 介绍页](https://openai.com/research/paperbench/)
- [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)
- [Rubrics as Rewards](https://openreview.net/forum?id=21UFlJrmS2)
- [ArenaRL](https://arxiv.org/abs/2601.06487)
