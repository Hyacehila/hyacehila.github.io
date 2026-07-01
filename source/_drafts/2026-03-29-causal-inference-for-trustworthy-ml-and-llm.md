---
title: 从相关到因果：因果推断如何支撑可信机器学习与大模型
date: 2026-03-29 19:00:00 +0800
categories: ["Machine Learning"]
tags: [Causal Inference, LLM, CLIP, Multimodality, LimiX]
author: Hyacehila
excerpt: 从相关、虚假关联和混杂控制出发，梳理因果推断如何为可信机器学习、大语言模型、CLIP 式多模态系统与结构化数据大模型 LimiX 提供方法论支撑。
---

> 这篇草稿写于 2026 年 3 月 29 日。文中优先使用综述、经典方法论文、官方模型卡和官方仓库入口来整理问题脉络。原始提纲中的核心观点我都保留下来了，但对个别过于绝对的表述做了收敛，例如把“现在的多模态语言模型都基于 CLIP Based Encoder”改写为“当前大量多模态模型仍然依赖 CLIP/SigLIP 一类对比学习视觉编码器或其变体”。

## 研究问题与范围

这篇文章想回答三个连在一起的问题。

第一，为什么今天的深度学习和大语言模型虽然强大，却仍然经常停留在“学到关联”而不是“学到因果”的层面。第二，当我们无法像医学双盲试验那样直接做随机试验时，如何从观测数据中尽可能可靠地估计因果效应。第三，因果推断到底怎样进入今天的大模型体系，尤其是 LLM、CLIP 式多模态模型，以及结构化数据大模型这三条线。

这里不试图完整覆盖因果推断的全部数学细节，而是把问题收束到一条更适合机器学习研究者的主线上：**从“会预测”走向“能解释、能迁移、能支持决策”时，为什么必须把因果性重新请回来。**

## 一页式结论

- 现有深度学习和大语言模型的训练目标，主轴仍然是从海量样本中捕捉统计相关性，而不是显式识别因果机制；这使它们容易学到在训练分布内有效、但在环境变化后失效的虚假关联。[Shortcut Learning in Deep Neural Networks](https://www.nature.com/articles/s42256-020-00257-z) 和 [Causality for Large Language Models](https://arxiv.org/abs/2410.15319) 都把这种问题说得很清楚。
- 纯相关模型当然能做出高分预测，但解释性通常不足，尤其在高风险决策里更明显。后验解释或 IML/XAI 并不是没用，而是常常不足以替代真正可审计的决策依据。[Rudin 2019](https://www.nature.com/articles/s42256-019-0048-x) 对这一点有非常直接的批评。
- 随机对照试验仍然是因果识别的金标准；如果只能使用历史数据或观测数据，核心挑战就变成了如何控制混杂变量，让处理组与对照组在可比条件下估计效应。[A Survey on Causal Inference](https://arxiv.org/abs/2002.02770) 是很好的总入口。
- 观测数据中的经典方法并不只是一种。匹配、倾向得分、IPW/DR、协变量直接平衡，以及像 [D2VD](https://kunkuang.github.io/papers/AAAI17-ATE_DVD.pdf) 这样的数据驱动变量分解方法，都在回答同一个问题：如何尽量把“相关”变成“更接近因果”的估计。
- 因果进入 LLM 的位置并不只在推理时提几个 counterfactual prompt，而是贯穿预训练、微调、对齐、推理和评测全生命周期。[Causality for Large Language Models](https://ar5iv.labs.arxiv.org/html/2410.15319) 明确把它拆成了因果数据、因果 foundation model、因果 SFT、因果 RLHF/DPO、因果 benchmark 等多条路线。
- 在多模态系统里，CLIP 式对比学习提供了强大的统一语义空间，但它并不会自动消除训练数据中的偏差。相反，最新研究提示，预训练数据本身就是 CLIP 偏差的重要来源，因此“因果化地优化 CLIP”确实值得单独研究。[CLIP 原论文](https://arxiv.org/abs/2103.00020)、[Intrinsic Bias is Predicted by Pretraining Data](https://arxiv.org/abs/2502.07957) 和 [Rethinking Misalignment in Vision-Language Model Adaptation from a Causal Perspective](https://arxiv.org/abs/2410.12816) 可以连起来看。
- 你提到的 [LimiX](https://arxiv.org/abs/2509.03505) 值得关注，但需要澄清：它更准确地说是结构化数据 foundation model，也就是 LDM，而不是图文意义上的多模态 VLM。它的价值在于把表格/结构化数据建模推向统一大模型接口，并在官方模型卡里明确强调了 sample retrieval 和 causal inference 方向，这对未来“LLM + 因果分析 + 结构化数据”系统很有启发。[官方模型卡](https://huggingface.co/stable-ai/LimiX-16M)

## 检索策略与证据标准

我这次检索主要围绕六组关键词展开：`causal inference survey`、`matching propensity score confounding`、`D2VD variable decomposition`、`Causality for LLM`、`causal CLIP / vision-language bias`、`LimiX structured-data model`。

证据优先级大致如下：

- `confirmed`：综述、经典方法论文、Nature/统计学期刊文章、arXiv 原始论文页面、官方模型卡与官方仓库。
- `early`：较新的 arXiv 工作，方向可信，但还没有形成稳定共识。
- `speculative`：基于多篇材料串起来的研究判断，而不是单篇论文已经证明的结论。

## 信息源渠道图

| 渠道 | 代表来源 | 用途 | 证据等级 |
| --- | --- | --- | --- |
| 因果推断综述 | [A Survey on Causal Inference](https://arxiv.org/abs/2002.02770) | 统一经典观测因果方法地图 | confirmed |
| 经典方法入口 | [Stuart 2010 Matching Review](https://pubmed.ncbi.nlm.nih.gov/20871802/), [CBPS 官方页](https://imai.fas.harvard.edu/research/CBPS.html), [D2VD](https://kunkuang.github.io/papers/AAAI17-ATE_DVD.pdf) | 对照匹配、倾向得分、直接平衡、变量分解 | confirmed |
| 相关与鲁棒性 | [Shortcut Learning in Deep Neural Networks](https://www.nature.com/articles/s42256-020-00257-z) | 解释虚假关联和分布迁移失败 | confirmed |
| 可解释性边界 | [Rudin 2019](https://www.nature.com/articles/s42256-019-0048-x) | 解释为什么后验解释不足以替代可决策机制 | confirmed |
| LLM 因果化 | [Causality for Large Language Models](https://arxiv.org/abs/2410.15319), [Causal Reasoning and Large Language Models](https://arxiv.org/abs/2305.00050) | 连接 LLM 与因果推断 | confirmed / early |
| 多模态与 CLIP | [CLIP](https://arxiv.org/abs/2103.00020), [Intrinsic Bias...](https://arxiv.org/abs/2502.07957), [Rethinking Misalignment...](https://arxiv.org/abs/2410.12816) | 讨论对比学习、数据偏差与因果优化 | confirmed / early |
| 结构化数据大模型 | [LimiX 论文](https://arxiv.org/abs/2509.03505), [LimiX 模型卡](https://huggingface.co/stable-ai/LimiX-16M) | 理解 LDM 与 retrieval 式推理 | early |

## 核心概念/问题分解

### 1. 相关为什么不等于因果

今天绝大多数深度学习系统的默认目标，都是最小化预测误差或最大化似然。无论是分类器、推荐模型、视觉编码器还是大语言模型，本质上都在学习输入与输出之间的统计依赖。这个范式并没有错，它解释了为什么神经网络在海量数据上能够取得惊人的经验性能。

问题在于，**统计相关性并不自动等于因果关系**。当数据里混有背景偏差、采样偏差、标签污染、社会刻板印象或环境特有的偶然模式时，模型会优先吸收那些最容易降低训练误差的信号。Geirhos 等人在 [Shortcut Learning in Deep Neural Networks](https://www.nature.com/articles/s42256-020-00257-z) 中把这件事概括得很准确：模型经常学到的是在 benchmark 上有效、但在更困难测试条件下无法迁移的 shortcut。

这也是为什么“训练集和测试集看起来都很准”并不意味着模型真的理解了任务。一旦数据环境迁移，原先有效的虚假关联就会失效。对于大语言模型来说，这种问题会表现成幻觉、刻板偏见、错误归因；对于视觉和多模态模型来说，则会表现成背景依赖、语义错配、跨域脆弱性。

### 2. 为什么关联模型的解释性通常不够

很多人会说，既然黑盒模型难解释，那加一层 IML/XAI 就好了。这个思路当然有价值，但它的边界也越来越清楚。解释方法可以帮助我们观察模型关注了什么、哪些特征重要、哪些样本最关键，但这不等于我们已经获得了可以直接用于干预和决策的机制知识。

[Rudin 2019](https://www.nature.com/articles/s42256-019-0048-x) 的核心批评是：在高风险决策里，依赖黑盒再做事后解释，常常是在用“解释”修补一个本来就不该是黑盒的问题。这个判断不必被理解成“IML 没价值”，更准确的说法是：**IML 更像诊断工具，而不是因果保证。**

也就是说，关联模型即便可以被解释，解释出来的也可能只是“模型如何利用数据中的模式”，而不是“世界为什么会这样运作”。这正是原始提纲里“需要结合人类知识才可以用于决策”的含义所在。

### 3. 因果为什么更接近可用于决策的知识

因果推断之所以重要，不是因为它更“高级”，而是因为它直接对应干预问题。我们真正关心的往往不是“X 和 Y 有没有相关”，而是“如果我改变 X，Y 会怎样变化”。这种问题天然面向行动。

当然，这里也要克制一点。因果解释并不是神奇通行证。即使是因果结论，也依赖实验设计、识别假设、变量测量质量以及领域知识。更稳妥的说法是：**相比纯相关，因果分析更接近可用于决策的知识，但它仍然需要明确的识别条件和人为审查。**

### 4. 为什么随机试验仍然是因果识别的金标准

如果有条件，最理想的因果识别方式仍然是随机试验。原因很朴素：随机分配会在期望意义上把处理组和对照组拉到可比条件，使得两组之间系统性的差异尽量只剩下我们关心的干预本身。医学里常说的随机双盲试验，本质上就是在最大限度排除主观偏差、选择偏差和已知或未知混杂。

这也是为什么原始提纲会把双盲试验称为“因果检验的金标准”。在理想条件下，它确实是最接近“只改变一个因素，其余条件保持一致”的设计。

### 5. 做不了随机试验时，观测数据怎样逼近因果

现实问题在于，我们经常拿不到随机试验，只能使用历史数据、平台日志、病历、广告投放数据或政策实施后的观测样本。这时最麻烦的地方就是**混杂变量**。如果处理组和对照组在年龄、疾病严重程度、资源暴露、地理区域或历史行为上本来就不同，那么简单比较两组平均结果，得到的往往只是“组间差异”，而不是处理效应。

因此，观测因果的核心工作就变成了：**如何让两组在关键混杂上尽量可比。** 经典路线主要有几类。

- **匹配方法**：为处理组样本寻找在协变量上最接近的对照样本。[Stuart 2010](https://pubmed.ncbi.nlm.nih.gov/20871802/) 是非常经典的综述入口。
- **倾向得分方法**：先估计“一个样本接受处理的概率”，再用它做匹配、分层、加权或 doubly robust 估计。它的意义不是把问题变简单，而是把高维协变量比较转化为一个更可操作的平衡问题。
- **数据驱动变量分解**：像 [D2VD](https://kunkuang.github.io/papers/AAAI17-ATE_DVD.pdf) 这种方法试图自动区分 confounders、outcome predictors 和 instrumental variables，避免把不该用于调整的变量也一股脑塞进模型。
- **协变量直接平衡方法**：如 [CBPS](https://imai.fas.harvard.edu/research/CBPS.html) 这一类方法，不再只盯着 propensity score 的预测准确率，而是直接把“平衡协变量分布”作为优化目标。

从机器学习视角看，这些方法背后的共同思想非常统一：不是让模型尽可能好地预测结果，而是先努力构造一个更接近“可比较世界”的样本空间，再谈效应估计。

## 近一年进展

如果把时间窗口收缩到 2025 年 3 月到 2026 年 3 月，这个方向最值得注意的几个信号大概是这样：

- 2025 年 2 月的 [Intrinsic Bias is Predicted by Pretraining Data and Correlates with Downstream Performance in Vision-Language Encoders](https://arxiv.org/abs/2502.07957) 系统分析了 131 个 CLIP 模型，指出**预训练数据集选择是偏差的最重要上游因素**，而且偏差与下游性能经常正相关。这直接支持了“不能只靠对比学习 loss，就假设模型会自动学到可靠语义”的担忧。
- 2025 年 3 月的 [CausalCLIPSeg](https://arxiv.org/abs/2503.15949) 是一个较新的 `early` 信号。它把因果干预模块引入 CLIP 驱动的医学 referring segmentation，用来分离 confounding bias 和 causal features。它未必是通用答案，但至少说明“因果化地改造 CLIP”已经开始进入具体任务层。
- 2025 年 9 月发布、11 月更新的 [LimiX](https://arxiv.org/abs/2509.03505) 代表另一条线索：除了语言和视觉 foundation model，结构化数据也开始拥有自己的 foundation-style 大模型接口。这个信号对因果机器学习尤其重要，因为很多因果分析本来就落在表格、医疗、金融、广告和政策数据上。

## 3-5年综述与里程碑

### 1. 从 shortcut learning 到“相关不够用”

过去几年里，一个越来越清楚的共识是：只靠相关学习的模型，往往把“容易学到的信号”误当成“真正该学的信号”。[Geirhos 2020](https://www.nature.com/articles/s42256-020-00257-z) 基本上把这条线正式命名了。

### 2. 从 CLIP 成功到 CLIP 偏差问题显化

[CLIP 2021](https://arxiv.org/abs/2103.00020) 证明了用海量图文对做对比学习，确实可以学到极强的通用表示，并且在自然分布迁移上比很多传统监督模型更稳健。但 CLIP 原论文自己也承认，互联网图文对会把社会偏差一起带进模型。后续研究，如 [Refining Skewed Perceptions in Vision-Language Contrastive Models through Visual Representations](https://arxiv.org/abs/2405.14030) 和 [Intrinsic Bias...](https://arxiv.org/abs/2502.07957)，进一步把这个问题做实了。

### 3. 从“LLM 会说因果话”到“怎样把因果嵌进 LLM 生命周期”

[Causal Reasoning and Large Language Models](https://arxiv.org/abs/2305.00050) 给出的一个重要判断是：LLM 在文本层面已经能生成相当不错的因果论证，甚至能帮助人类更快搭起 causal analysis 的初稿。但这篇工作同样强调，LLM 并没有直接读取真实数据，也无法自动替代正式的因果识别流程。

到 2024 年，[Causality for Large Language Models](https://ar5iv.labs.arxiv.org/html/2410.15319) 把问题又推进了一步。它不再满足于讨论“如何用 prompt 激发 LLM 的因果知识”，而是把因果直接放进 LLM 生命周期的五个阶段：

- 预训练：去偏 token embedding、反事实语料、因果 foundation model。
- 架构：在 Transformer 或 foundation model 中注入显式因果结构、图先验或机制分解，而不是默认让注意力自己从相关性里碰出机制。
- 微调：在特定任务中进行 causal SFT，把模型从“会说相关话”拉向“更会处理干预与反事实”。
- 对齐：把 causal RLHF、counterfactual DPO、causal preference optimization 纳入对齐视角。
- 推理：支持 causal discovery、causal effect estimation 和 counterfactual reasoning。
- 评测：建立真正能测因果能力的 benchmark，而不是只测语言表面相似度。

这里最重要的不是某个具体算法，而是研究问题本身变了。大家开始意识到：**如果只在推理时给 LLM 套一层“因果 prompt”，那么模型底层仍然可能是一个相关性机器。**

## 关键 mixed-source 证据对照

| 判断 | 支撑来源 | 结论强度 | 备注 |
| --- | --- | --- | --- |
| 深度学习容易学到 shortcut 和虚假关联 | [Geirhos 2020](https://www.nature.com/articles/s42256-020-00257-z) | confirmed | 这是今天谈鲁棒性和分布迁移时最稳固的背景判断之一。 |
| 后验解释不足以替代可决策机制 | [Rudin 2019](https://www.nature.com/articles/s42256-019-0048-x) | confirmed | 这更适用于高风险场景，不是说解释方法完全无用。 |
| 观测因果的核心在于控制混杂和设计可比性 | [A Survey on Causal Inference](https://arxiv.org/abs/2002.02770), [Stuart 2010](https://pubmed.ncbi.nlm.nih.gov/20871802/) | confirmed | 匹配、PS、IPW、DR、平衡方法都在回答这一问题。 |
| LLM 可以生成因果论证，但仍需要与正式因果技术结合 | [Causal Reasoning and LLMs](https://arxiv.org/abs/2305.00050) | confirmed | 这是连接“语言能力”和“真实数据因果分析”的关键界面。 |
| 因果应该进入 LLM 全生命周期，而不只是在 prompt 层 | [Causality for LLMs](https://arxiv.org/abs/2410.15319) | confirmed | 目前最系统的路线图之一。 |
| CLIP 偏差与预训练数据高度相关 | [Intrinsic Bias...](https://arxiv.org/abs/2502.07957), [CLIP](https://arxiv.org/abs/2103.00020) | confirmed | 这说明优化目标和数据治理都要改。 |
| 用因果视角优化 CLIP/VLM 可能是一个独立研究方向 | [Rethinking Misalignment...](https://arxiv.org/abs/2410.12816), [CausalCLIPSeg](https://arxiv.org/abs/2503.15949) | early | 已有工作出现，但离形成统一范式还早。 |
| LimiX 可作为“结构化数据底座 + retrieval 推理”参考对象 | [LimiX 论文](https://arxiv.org/abs/2509.03505), [LimiX 模型卡](https://huggingface.co/stable-ai/LimiX-16M) | early | 值得跟踪，但离行业共识还有距离。 |

## 代表论文/系统/数据集对照表

| 类型 | 名称 | 它解决的问题 | 对本文的意义 |
| --- | --- | --- | --- |
| 综述 | [A Survey on Causal Inference](https://arxiv.org/abs/2002.02770) | 系统梳理观测因果方法 | 是“匹配/PS/DR/ML-enhanced causal inference”的总入口 |
| 经典综述 | [Matching Methods for Causal Inference](https://pubmed.ncbi.nlm.nih.gov/20871802/) | 匹配方法的原则与实践 | 适合补“如何从观测数据构造可比组” |
| 经典方法 | [CBPS](https://imai.fas.harvard.edu/research/CBPS.html) | 直接优化协变量平衡的倾向得分 | 对应你提纲中的“混杂变量直接平衡方法” |
| 变量分解 | [D2VD](https://kunkuang.github.io/papers/AAAI17-ATE_DVD.pdf) | 自动分离混杂变量与 adjustment variables | 对应你提纲中的“D 方 VD” |
| LLM 因果综述 | [Causality for Large Language Models](https://arxiv.org/abs/2410.15319) | 因果如何进入 LLM 生命周期 | 对应“因果数据、因果 Transformer、因果 RL 与 SFT 对齐” |
| LLM 能力研究 | [Causal Reasoning and LLMs](https://arxiv.org/abs/2305.00050) | LLM 能否进行因果论证 | 说明 LLM 擅长说出因果论证，但还不能替代因果分析本身 |
| 多模态基础模型 | [CLIP](https://arxiv.org/abs/2103.00020) | 大规模图文对比学习 | 是今天很多 VLM/MM-LLM 视觉前端的重要祖先 |
| VLM 偏差研究 | [Intrinsic Bias...](https://arxiv.org/abs/2502.07957) | 数据偏差与 CLIP 偏差的关系 | 支持“仅靠 contrastive loss 不足以保证无偏” |
| 因果化 VLM | [Rethinking Misalignment...](https://arxiv.org/abs/2410.12816) | 用因果视角解释 CLIP 适配中的 data misalignment | 给“如何用因果优化 CLIP”一个具体技术入口 |
| 结构化数据模型 | [LimiX](https://arxiv.org/abs/2509.03505) | 统一分类、回归、插补等表格任务 | 提醒我们不要只盯语言和图像，结构化数据也需要 foundation model |

## 可行动研究想法 / 假设候选

- **做一个真正面向 spurious correlation 的 CLIP 因果评测集。** 把目标对象、背景、文本提示词和共现偏差显式拆开，测试模型到底在识别对象，还是在识别上下文共现。
- **把 counterfactual data augmentation 引入多模态预训练。** 不只替换 prompt，也替换背景、属性、关系和语义图，让图文正负样本更接近“因果干预”而不是“随机重采样”。
- **把因果一致性纳入 LLM 的 SFT 与偏好对齐。** 奖励模型不只评估答案流畅度，还评估“是否区分了观察、干预和反事实”。
- **做一个 LLM + LimiX 的因果分析 copilot。** 让 LLM 负责问题拆解、变量命名、因果图草拟和报告生成，让 LimiX 一类结构化数据模型负责表格表示、检索式支持和下游预测接口。
- **把 D2VD/CBPS 这一类观测因果方法重新翻译成 foundation-model 时代接口。** 今天很多表格 foundation model 还偏预测导向，真正把“可比性构造”和“confounder control”纳入统一推理接口的工作仍然不多。

## 争议/局限/开放问题

- “因果解释可以直接用于决策”这句话在方向上是对的，但在方法上必须补一句：只有在识别假设合理、变量测量可靠、干预定义明确时，因果估计才适合支持决策。
- 不是所有多模态模型都严格意义上“基于 CLIP”。到 2026 年，很多系统已经转向 CLIP、SigLIP、EVA、InternViT 等更丰富的视觉前端组合。但“依赖大规模对比学习视觉编码器”这个判断依然成立。
- LLM 在文本层面表现出因果论证能力，不等于它已经具备数据层面的因果识别能力。[Causal Reasoning and LLMs](https://arxiv.org/abs/2305.00050) 明确指出，真正值得做的是把 LLM 和现有因果技术结合起来。
- [LimiX](https://arxiv.org/abs/2509.03505) 目前更像一个值得高度关注的 `early` 信号。它说明结构化数据 foundation model 正在成形，但离“因果数据大模型的统一标准答案”还很远。

## 持续跟踪清单

- 跟踪真正把因果嵌进 LLM 训练而非只做 prompting 的工作。
- 跟踪面向 CLIP/VLM 的 counterfactual benchmark 和 causal intervention 方法。
- 跟踪高维观测因果方法在 foundation model 场景下的重写，例如变量分解、协变量平衡和 doubly robust 学习。
- 跟踪 [LimiX 官方模型卡](https://huggingface.co/stable-ai/LimiX-16M) 和 [官方仓库](https://github.com/limix-ldm/LimiX) 的更新，尤其是 retrieval、feature selection、causal inference 接口是否继续扩展。

## 推荐阅读路径

1. 先读 [A Survey on Causal Inference](https://arxiv.org/abs/2002.02770)，建立观测因果的大图景。
2. 再读 [Stuart 2010](https://pubmed.ncbi.nlm.nih.gov/20871802/) 和 [CBPS](https://imai.fas.harvard.edu/research/CBPS.html)，把匹配、倾向得分、协变量平衡这些经典技术真正串起来。
3. 接着看 [D2VD](https://kunkuang.github.io/papers/AAAI17-ATE_DVD.pdf)，理解“并非所有变量都该被同样对待”这件事。
4. 然后读 [Causal Reasoning and Large Language Models](https://arxiv.org/abs/2305.00050) 与 [Causality for Large Language Models](https://arxiv.org/abs/2410.15319)，把因果与 LLM 的关系从“prompt 技巧”提升到“训练与系统设计”。
5. 再回到多模态，连着读 [CLIP](https://arxiv.org/abs/2103.00020)、[Intrinsic Bias...](https://arxiv.org/abs/2502.07957) 和 [Rethinking Misalignment...](https://arxiv.org/abs/2410.12816)。
6. 最后把视角扩展到结构化数据，读 [LimiX](https://arxiv.org/abs/2509.03505) 和 [模型卡](https://huggingface.co/stable-ai/LimiX-16M)。

## 参考文献与官方入口

- [Geirhos et al., Shortcut Learning in Deep Neural Networks, Nature Machine Intelligence, 2020](https://www.nature.com/articles/s42256-020-00257-z)
- [Rudin, Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead, Nature Machine Intelligence, 2019](https://www.nature.com/articles/s42256-019-0048-x)
- [Yao et al., A Survey on Causal Inference, 2020](https://arxiv.org/abs/2002.02770)
- [Stuart, Matching methods for causal inference: A review and a look forward, 2010](https://pubmed.ncbi.nlm.nih.gov/20871802/)
- [Imai and Ratkovic, Covariate Balancing Propensity Score, 2014 官方入口](https://imai.fas.harvard.edu/research/CBPS.html)
- [Kuang et al., Treatment Effect Estimation with Data-Driven Variable Decomposition (D2VD), AAAI 2017](https://kunkuang.github.io/papers/AAAI17-ATE_DVD.pdf)
- [Kiciman et al., Causal Reasoning and Large Language Models: Opening a New Frontier for Causality, 2023](https://arxiv.org/abs/2305.00050)
- [Wu et al., Causality for Large Language Models, 2024](https://arxiv.org/abs/2410.15319)
- [Radford et al., Learning Transferable Visual Models From Natural Language Supervision (CLIP), 2021](https://arxiv.org/abs/2103.00020)
- [Zhang et al., Rethinking Misalignment in Vision-Language Model Adaptation from a Causal Perspective, 2024](https://arxiv.org/abs/2410.12816)
- [Ghate et al., Intrinsic Bias is Predicted by Pretraining Data and Correlates with Downstream Performance in Vision-Language Encoders, 2025](https://arxiv.org/abs/2502.07957)
- [Dai and Joshi, Refining Skewed Perceptions in Vision-Language Contrastive Models through Visual Representations, 2024/2025](https://arxiv.org/abs/2405.14030)
- [Chen et al., CausalCLIPSeg, 2025](https://arxiv.org/abs/2503.15949)
- [Zhang et al., LimiX: Unleashing Structured-Data Modeling Capability for Generalist Intelligence, 2025](https://arxiv.org/abs/2509.03505)
- [LimiX 官方模型卡](https://huggingface.co/stable-ai/LimiX-16M)
- [LimiX 官方仓库](https://github.com/limix-ldm/LimiX)

## 一个更直接的收束

如果把原始提纲概括成一句话，我会把它改写成这样：

**深度学习和大语言模型已经非常擅长从数据中提炼相关模式，但当我们希望模型具备更强的解释性、鲁棒性和决策支持能力时，就不能只满足于“相关做得很好”，而必须进一步追问：这些模式里到底哪些是真正可迁移、可干预、可用于行动的因果结构。**

从这个角度看，因果推断并不是传统统计学留给 AI 的一门旧课，而更像是可信机器学习与可信大模型迟早要补上的底层课程。
