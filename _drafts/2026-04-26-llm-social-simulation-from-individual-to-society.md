---
layout: blog-post
title: "从个体到社会：CNCC2025 讲座中的 LLM Agent 社会模拟研究脉络"
date: 2026-04-26 20:00:00 +0800
categories: [智能体系统]
tags: [LLM Agent, Social Simulation, Multi-Agent System, Agent-Based Modeling]
author: Hyacehila
excerpt: "从 CNCC2025 的大模型智能体社会模拟讲座出发，沿着个体模拟、场景模拟、社会模拟三层脉络，梳理 LLM Agent 如何进入计算社会科学。"
featured: false
math: true
---

# 从个体到社会：LLM Agent 社会模拟的三层研究脉络

这篇文章整理自我在 **CNCC2025 中国计算机大会 Tutorial“基于大模型智能体的社会模拟”** 中听到的一场讲座笔记。讲座中，复旦大学魏忠钰教授团队围绕 LLM Agent 社会模拟的研究进展做了系统梳理，其中一条非常清晰的主线是：**Individual Simulation -> Scenario Simulation -> Society Simulation**。也就是说，研究者首先要让模型能够模拟一个具体的人，再让多个被模拟的人在具体场景中互动，最后才可能讨论大规模社会系统中的舆论、选举、运动、市场和制度实验。

这个脉络和复旦 DISC 团队的综述论文 [From Individual to Society: A Survey on Social Simulation Driven by Large Language Model-based Agents](https://arxiv.org/abs/2412.03563) 高度一致。综述把 LLM Agent 驱动的社会模拟划分为三层：个体模拟、场景模拟和社会模拟。三层之间不是简单的应用分类，而是一个逐渐放大的建模尺度：从“一个人是否像他自己”，到“多个人是否能在一个规则清楚的场景中产生合理互动”，再到“一个由大量异质个体组成的系统是否能涌现出接近真实世界的宏观现象”。

我原本的笔记比较碎片化，但抓住了几个关键问题：个体模拟需要在 prompt 和 fine-tuning 之间寻找成本与保真度的平衡；场景模拟的核心不是堆更多智能体，而是领域任务适配；大型社会模拟最困难的是规模，全部使用 LLM 几乎不现实，因此需要 LLM 与传统 Agent-based Modeling、舆论动力学模型之间的混合框架。本文尝试把这些点整理成一篇更完整的综述式笔记，并结合 arXiv TeX 原文中的方法细节，对几个代表工作做更准确的展开。

## 1. 为什么社会模拟需要 LLM Agent

社会模拟并不是 LLM 出现以后才有的研究方向。传统社会科学、计算社会科学和复杂系统研究中，Agent-based Modeling, ABM 早就被用于研究舆论传播、群体极化、交通流、市场行为、选举预测和传染病传播。ABM 的基本想法是从微观个体出发，让每个个体遵循某种规则，随后观察宏观层面的系统演化。

这种方法的优势是可控、可解释、可扩展。研究者可以清楚地定义每个 agent 的状态变量、交互网络和更新函数，然后用参数扫描或反事实实验观察系统变化。但它的短板也很明显：传统 ABM 中的 agent 往往太“薄”。一个个体可能只有一个态度分数、一个风险偏好、一个收益函数或一个简单规则。这样的个体可以产生宏观模式，却很难表达复杂的人类语言、记忆、身份、情绪、叙事、价值观和情境判断。

LLM Agent 的加入改变了这个局面。大模型天然擅长自然语言理解与生成，可以在 prompt、profile、memory、reflection 和 action module 的组织下表现出更复杂的个体行为。它可以阅读新闻、解释事件、参与对话、表达犹豫、给出理由，也可以根据不同身份或经历产生不同反应。这使得社会模拟不再只是状态变量之间的数值更新，而有机会引入语义、叙事和互动过程。

但这也带来了新的问题。LLM 推理昂贵，长上下文成本高，多智能体交互会迅速放大 token 开销。如果有 $N$ 个智能体，且每个智能体都要和其他智能体通过自然语言交互，最坏情况下交互复杂度会走向 $O(N^2)$。对于十几个或几百个智能体，研究者还可以勉强运行；对于数万、百万甚至千万级社会模拟，全部使用 LLM 生成自然语言显然不可接受。

因此，LLM 社会模拟的核心问题并不是“能不能把每个人都换成一个 ChatGPT”，而是如何在不同建模层级上选择合适的抽象：什么时候需要高保真 LLM 个体，什么时候只需要数学动力学模型，什么时候需要混合建模，什么时候需要把现实人口统计、社交网络和动态事件嵌入模拟环境。CNCC2025 讲座给我的最大启发正在这里：这个领域真正的难点是尺度转换。

## 2. 个体模拟：从“像一个人”开始

个体模拟是整个链条的第一层。它关心的问题是：一个 LLM Agent 能不能模拟某个具体人物、某类人口群体或某种身份组合的行为。如果一个 agent 连单个角色都扮演不好，那么后续的场景互动和社会涌现也很难可信。

综述论文对个体模拟的架构做了一个很有帮助的拆解：一个用于个体模拟的 agent 通常包含 **profile、memory、planning、action** 四类组件。

Profile 决定“这个人是谁”。它可以包括人口统计属性、职业、价值观、人格、关系网络、经历、兴趣、政治立场、专业知识等信息。Memory 决定“这个人记得什么”。它既可以是历史经历，也可以是模拟过程中产生的新观察。Planning 决定“这个人如何在当前情境下思考和计划”。Action 则决定“这个人最后能做什么”，例如回答问题、发帖、转发、投票、购买商品、移动到某个地点、选择合作或背叛。

围绕个体模拟，目前大致可以整理出三条技术路线：非结构化提示词、结构化微调，以及基于身份理论的折中方案。

### 2.1 Prompt-based：把角色写进上下文

最直接的路线是 prompt-based role-playing。研究者把角色背景、人格描述、历史经历或小说片段写进 prompt，让 LLM 在上下文中扮演这个人物。这个方法的好处是简单、灵活、成本低。只要有足够的背景材料，就可以快速构建一个可交互角色。

例如 [Evaluating Character Understanding of Large Language Models via Character Profiling from Fictional Works](https://arxiv.org/abs/2404.12726) 关注的是模型能否从文学作品中提取角色画像。它把 character profile 拆成四个维度：attributes、relationships、events、personality。Attributes 是性别、技能、目标、背景等基本属性；relationships 是角色与他人的关系；events 是角色经历过或受其影响的事件，并且常常需要按时间线重组；personality 则是从行为、选择和互动中抽象出来的稳定人格特征。

这篇工作的重要性不只在于提出了一个评测任务，更在于它揭示了 prompt-based 个体模拟的底层前提：模型必须先把长文本中的人物信息压缩成一个稳定、准确、可使用的画像。角色扮演并不是简单模仿语气，而是要理解一个人在不同事件中的动机、关系和行为逻辑。

但 prompt-based 方法的问题也很明显。第一，它严重依赖上下文窗口。角色越复杂，需要塞进 prompt 的背景材料越多；当对话变长时，早期设定容易被稀释。第二，它依赖模型的 in-context learning 能力，不同基础模型的稳定性差异很大。第三，多轮交互中容易出现 out-of-character，即角色遗忘或角色漂移。模型可能一开始像某个角色，但在受到用户诱导或新上下文干扰后，很快回到通用助手口吻。第四，复杂角色的行为一致性难以评价。我们可以主观觉得“像”，但要系统性比较不同方法，就需要更细的 benchmark。

因此，prompt-based 路线适合低成本快速构建个体，但它很难独立支撑大规模、高可信的社会模拟。它更像个体模拟的原型工具，而不是最终答案。

### 2.2 Fine-tuning-based：把经验写进权重

第二条路线是通过微调把角色知识、经验和表达风格写进模型参数。代表工作是 [Character-LLM: A Trainable Agent for Role-Playing](https://arxiv.org/abs/2310.10158)。

Character-LLM 的核心不是简单收集某个角色说过的话再做 SFT，而是提出了一个 experience upload 的框架。它首先进行 **Experience Reconstruction**：从可靠资料中整理人物 profile，并借助 LLM 把抽象的人物资料重构为具体场景和交互经历。然后进行 **Experience Upload**：用这些重构出的经验数据对基础模型进行监督微调，使模型更稳定地扮演特定角色。最后，它还设计了 **Protective Experience**，用于缓解 character hallucination。比如一个古代人物不应该自信地回答现代编程问题，保护性经验会训练模型在超出角色知识边界时保持一致的拒答或不确定表达。

这条路线的优势是保真度和稳定性更高。把角色经验写入参数后，模型不必每次都依赖长 prompt 重新注入背景，长对话中的角色一致性也更强。对于少数重要角色，例如历史人物、游戏 NPC、虚拟陪伴角色或关键社会模拟 agent，这种方法是有价值的。

但它的问题也很现实。每个新角色都需要构造高质量数据集，可能还要微调一个专门模型。数据、算力、训练工程和评估成本都很高。社会模拟中的个体往往不是一个两个，而是成百上千乃至上百万。如果为每个人都训练一个模型，成本会完全失控。此外，微调后的模型仍然依赖基础模型能力，角色数据质量不够时也可能只是学到表层风格，而不是真正学到行为机制。

因此，fine-tuning-based 路线适合高价值个体和高保真角色，但不适合直接扩展到整个社会。

### 2.3 Identity-driven：从“角色”退一步到“身份组合”

第三条路线试图在 prompt 的低成本和 fine-tuning 的高保真之间取得平衡。讲座中提到的 [Identity-Driven Hierarchical Role-Playing Agents](https://arxiv.org/abs/2407.19412) 就属于这个方向。

这类工作的核心洞察来自社会学中的 identity theory：一个人不是一个不可拆解的整体标签，而是由多个身份共同构成。一个真实个体可能同时是“深度学习研究者”“严厉导师”“温和父亲”“城市中产”“某种政治立场持有者”。当他面对不同情境时，不同身份会被激活、组合或抑制。

HIRPF 的做法是把角色模拟从 role-level 推进到 identity-level。它关注两个关键机制：**identity isolation** 和 **explicit control**。Identity isolation 用来处理互斥身份，例如同一身份类别下的不同人格倾向或职业身份不能随意混合；explicit control 用来处理兼容身份，使模型可以在推理时显式组合多个身份。论文实现上借鉴了 LoRA 和 MoE 思路，用层次化身份模块来支持身份组合。

这条路线非常适合社会模拟。因为社会模拟往往并不需要精确复刻某一个真实人，而是需要构造大量带有差异的人口群体。与其为每个人写一份长 prompt 或训练一个角色模型，不如把个体拆解为人口统计、职业、人格、价值观、兴趣和社会角色等身份维度，再按目标分布组合。这样可以用 zero-shot 或 few-shot 的方式构造大量异质 agent，并且保持一定的可控性。

从研究角度看，identity-driven 方法的重要性在于它把个体模拟从“文学式角色扮演”推向“社会科学中的人口群体建模”。在前者中，我们问的是“这个模型像不像贝多芬或苏格拉底”；在后者中，我们问的是“这个模型能否代表某一类具有稳定统计特征的人”。对于社会模拟，后一个问题往往更关键。

## 3. 场景模拟：从单体拟真到多智能体协作

当个体 agent 能够被基本构造出来之后，下一层就是场景模拟。场景模拟不再只问“一个人像不像”，而是问“多个人在某个具体制度、规则、任务或社交环境中能否产生合理互动”。

综述论文把场景模拟中的 scenario 分为两类：**dialog-driven scenario** 和 **task-driven scenario**。Dialog-driven scenario 以对话和社交互动为核心，例如社交晚宴、问答、谈判、辩论、博弈、角色互动。Task-driven scenario 则以完成具体任务为核心，例如软件开发、科学研究、医疗诊断、法律分析、金融决策、工业流程等。

这一步的关键是多智能体系统，Multi-Agent System, MAS。一个场景通常需要环境、角色、组织结构和通信机制。环境包括场景规则、状态、历史和工具；角色包括参与者和可能的 director；组织结构决定 agent 是平行协作、层级协作，还是由某个协调者调度；通信机制决定 agent 之间用自然语言、结构化消息、工具调用结果还是共享状态进行交互。

我在讲座笔记中写下的一个判断是：**通用 AI 不足以解决我们希望研究的复杂领域问题。** 现在看，这个判断应该放在场景模拟这一层理解。单个通用 LLM 当然可以回答很多问题，但当我们把任务放进真实场景，它就会遇到领域知识、流程规范、工具使用、责任分工、评估标准和错误恢复等问题。一个医疗诊断模拟不能只靠“你是医生”的 prompt；一个法庭模拟不能只靠“你是律师”；一个软件开发团队模拟也不能只靠几个 agent 互相聊天。

因此，场景模拟的核心不是把多个 agent 拼在一起，而是 **领域任务适配**。这包括至少四件事。

第一，构造适合领域的 agent 能力。不同场景需要不同 profile、memory、tools 和 action space。软件开发 agent 需要代码仓库上下文、测试工具和版本控制操作；法律 agent 需要法条、判例和论证结构；医疗 agent 需要症状、检查结果、诊疗指南和风险控制。

第二，构造高质量微调数据或交互轨迹。对于复杂任务，通用模型经常给出看似合理但不符合领域标准的输出。研究者可能需要通过专家标注、LLM 自生成轨迹、拒绝采样微调 Rejection Sampling Fine-Tuning 或 RLHF/RLAIF 方式构造更可靠的 agent。

第三，设计多智能体组织结构。多个 agent 不一定天然优于单个 agent。没有清楚分工时，多 agent 可能只是重复发言、互相附和或把错误放大。有效的 MAS 需要区分 planner、executor、critic、memory manager、tool user、domain expert 等角色，并设计何时通信、通信什么、谁有最终决策权。

第四，加入记忆检索与反思机制。场景模拟往往是多轮的。如果 agent 只看当前轮输入，它就无法保持目标、上下文和责任连续性。因此，RAG、vector database、episodic memory、reflection、自我纠错和历史摘要常常成为标配。

场景模拟是从个体到社会的中间层。它的尺度通常还不算特别大，但复杂度已经从“角色画像”变成“制度化互动”。这一层如果做不好，社会模拟就容易变成大量 agent 的无组织对话。真正的社会不是许多个体随机聊天，而是个体在规则、网络、组织、信息流和制度约束下持续互动。

## 4. 社会模拟：从多体互动到宏观涌现

社会模拟是第三层，也是 CNCC2025 这场讲座中最让我感兴趣的部分。这里的问题不再是几个人完成一个任务，而是成千上万甚至上百万个个体在一个网络和环境中互动，最终产生宏观舆论、选举结果、社会运动、群体极化、市场波动或政策响应。

综述论文把社会模拟的构造元素拆成四类：**composition、network、social influence、outcomes**。

Composition 关心社会由什么样的个体构成。个体是否来自真实人口统计分布？是否包含不同性别、年龄、职业、收入、地区、教育、党派、人格和兴趣？是否需要对 opinion leaders、核心用户、组织节点等 outliers 做特殊建模？这是社会模拟能否具有代表性的基础。

Network 关心个体之间如何连接。线下社会可能是地理邻近、亲属关系、组织关系和职业关系；线上社会则可能是关注、转发、评论、推荐系统和兴趣社群。网络结构决定信息能否传播，也决定局部互动能否放大为宏观现象。

Social influence 关心个体如何影响彼此。一条信息是通过自然语言说服他人，还是通过态度分数更新？影响强度是否取决于相似性、可信度、权威性、重复暴露、情绪强度或平台推荐？不同影响机制会导致完全不同的宏观结果。

Outcomes 关心模拟输出什么。社会模拟既可以输出宏观统计结果，例如选票比例、平均态度、购买率、感染率；也可以输出社会现象的形成过程，例如回音室、极化、谣言扩散、运动动员、信息茧房。

这一层的核心困难是规模。一个直接想法是：既然 LLM Agent 更像人，那就让每个人都是 LLM Agent。但这在大规模社会模拟中基本不可行。假设一百万个用户都要读取上下文、生成文本、更新记忆、与其他用户交互，成本会急剧膨胀。更重要的是，社会模拟不一定每个个体都需要同等精细。现实社会里，大量普通用户只是低频互动、轻量表达或被动接收信息；少数核心用户、意见领袖和组织节点才承担主要内容生成与议程设置功能。

这就引出了混合建模框架。

## 5. HiSim：LLM 与舆论动力学模型的混合破局

我在笔记中写到 [Unveiling the Truth and Facilitating Change: Towards Agent-based Large-scale Social Movement Simulation](https://arxiv.org/abs/2402.16333) “值得参考”。读完 TeX 原文后，这个判断仍然成立。它提出的 HiSim 框架非常适合作为大型社会模拟的入门样例，因为它清楚回答了一个现实问题：如何在不让每个人都调用 LLM 的情况下，模拟社交媒体上的社会运动。

HiSim 的基本假设来自社交媒体中的长尾分布：平台上的大量内容往往由少数活跃且有影响力的人产生。因此，系统把用户分成 **core users** 和 **ordinary users**。核心用户用 LLM Agent 细粒度模拟，普通用户用传统舆论动力学 ABM 模型模拟。这样既保留了关键节点的自然语言表达能力，又避免了对大量普通用户进行昂贵 LLM 推理。

对于核心用户，HiSim 设计了 profile、memory、action 三个模块。Profile 包括 demographics、social traits 和 communication roles。Demographics 包括姓名、性别、政治倾向、账户类型等；social traits 包括活跃度和影响力；communication roles 则借鉴影响拓扑，把用户分为 idea starter、amplifier、curator、commentator、viewer 等传播角色。Memory 分为 personal experience 和 event memory，前者来自事件发生前的历史推文，后者来自模拟过程中的观察。Action module 则覆盖社交媒体行为，例如 post、retweet、reply、like、do nothing。

对于普通用户，HiSim 不让他们生成长文本，而是维护 attitude score，并通过 ABM 的 opinion dynamics 更新态度。论文实验中涉及的模型包括 Bounded Confidence Model, BC、HK variant、Relative Agreement Model, RA、Social Judgement Model, SJ 和 Lorenz model，也就是可以概括为 **BC/HK/RA/SJ/Lorenz** 这一组舆论动力学模型。这些模型的共同点是：个体不需要完整语言生成，只需要根据接收到的信息、相似性、可信度、吸引或排斥机制更新数值态度。

混合框架最关键的接口是异质交互。核心用户输出的是自然语言，例如一条推文；普通用户模型接受的是数值态度。因此，系统需要把文本转换为 attitude score。HiSim 的做法是用外部 LLM 标注 stance direction，再用情感分析工具计算 attitude intensity，最后将文本后处理为普通用户 ABM 可以接收的分数。这就是 LLM 与动力学函数之间的桥。

这里必须做一个重要校准。我原始笔记中写到“底层用户的分数分布变化可以触发离线新闻摘要，再以自然语言形式反馈给核心用户，形成闭环”。这个想法作为未来扩展很有价值，但不能写成 HiSim 已经实现的机制。HiSim 原文明确说，考虑到 ordinary users 对 core users 的影响比较 subtle，当前并没有处理普通用户到核心用户的影响。它的 offline news feed 指的是把现实世界中的离线事件，如 George Floyd 事件，作为自然语言背景信息提供给核心用户 agent。也就是说，HiSim 的已实现闭环主要是核心用户文本影响普通用户态度，而普通用户聚合态度再反向生成自然语言影响核心用户，仍然是一个值得做但尚未在该框架中完成的方向。

HiSim 的评估也值得注意。它构建了 SoMoSiMu-Bench，包含 MeToo、RoeOverturned 和 BlackLivesMatter 三个 Twitter 社会运动数据集。评估分为 micro alignment 和 macro system evaluation。微观层面看 stance alignment、content alignment、behavior alignment；宏观层面看 attitude distribution 的 bias/diversity，以及平均态度时间序列与真实数据之间的 DTW 和 Pearson correlation。这个评估设计说明，社会模拟不能只看某个 agent 的发言像不像，还要看系统层面的趋势是否对齐。

从这篇工作可以得到一个重要结论：LLM Agent 在大型社会模拟中最适合承担语义复杂、影响力高、行为稀疏但重要的部分；传统 ABM 适合承担规模巨大、状态简单、需要高效更新的部分。二者不是替代关系，而是互补关系。

## 6. ElectionSim、SocioVerse、OASIS 与 AgentSociety：从单一框架走向基础设施

HiSim 展示了混合建模的思想，但整个领域正在进一步走向平台化和基础设施化。讲座中提到的 ElectionSim 和 SocioVerse，以及相关的 OASIS、AgentSociety，分别代表了几种扩展方向。

### 6.1 ElectionSim：人口统计分布与选举模拟

[ElectionSim](https://arxiv.org/abs/2410.20746) 面向大规模选举模拟。它的关键问题不是让某个单独 voter agent 说得多像，而是让整个模拟人口的分布接近真实选民结构。选举结果本质上是大量异质个体偏好的聚合，因此 demographic distribution 的对齐至关重要。

ElectionSim 从社交平台收集用户数据，构建大规模 voter pool，并通过 demographic tagging 给用户补充人口统计和政治属性。随后，它使用真实世界数据源，例如 U.S. Census Bureau 和 ANES，来构造目标人群分布。更关键的是，它使用 Iterative Proportional Fitting, IPF，把单变量边际分布近似为多属性联合分布，从而支持按州、性别、年龄、种族、意识形态、党派等维度采样更真实的选民群体。

这件事看起来是工程细节，但对社会模拟非常关键。很多 LLM 社会模拟的失败不是因为模型不会说话，而是因为模拟人口根本不代表真实社会。若样本分布偏了，宏观结果必然偏。ElectionSim 提醒我们：社会模拟不是只做 agent prompt，也是在做抽样、加权、校准和人口统计对齐。

### 6.2 SocioVerse：四个引擎与 10M 用户池

[SocioVerse](https://arxiv.org/abs/2504.10157) 更像一个面向社会模拟的世界模型框架。它提出四个关键部分：social environment、user engine、scenario engine、behavior engine。

Social environment 用于提供现实社会上下文，包括社会结构信息、动态事件和个性化信息流。User engine 用于把模拟 agent 与真实用户样本对齐。论文中构建了覆盖 X 和 Rednote 的 **10M user pool**，并设计人口统计标签体系。Scenario engine 用于把不同任务映射到不同交互结构，例如 questionnaire、in-depth interview、behavior experiment、social media interaction。Behavior engine 则决定 agent 如何产生行为，可以使用 traditional ABM，也可以使用 general LLM、expert LLM 或 domain-specific LLM。

SocioVerse 的意义在于，它不只是在解决某个具体任务，而是在抽象社会模拟所需要的基础设施。一个可信的社会模拟系统需要环境、用户、场景和行为四个层面的对齐。只做用户画像不够，因为用户需要现实环境；只做环境不够，因为不同任务需要不同场景结构；只做 LLM 行为生成也不够，因为某些大规模部分仍然需要传统模型。

这也回应了我在笔记中的另一个直觉：通用 AI 不足以解决复杂领域任务。SocioVerse 的回答不是“训练一个万能社会模型”，而是把社会模拟拆成多个可对齐、可替换、可扩展的引擎。

### 6.3 OASIS：百万级社交媒体平台仿真

[OASIS](https://arxiv.org/abs/2411.11581) 关注的是大规模社交媒体模拟。它的目标是构造一个通用、可扩展的社交平台模拟器，支持 X、Reddit 等平台，并可扩展到一百万用户规模。

OASIS 的组件包括 Environment Server、RecSys、Agent Module、Time Engine 和 Scalable Inferencer。Environment Server 维护用户、帖子、评论、关系、行为轨迹和推荐结果。RecSys 控制 agent 能看到什么信息，这一点非常重要，因为社交平台中的行为并不只由用户内在偏好决定，也由推荐系统塑造。Agent Module 包含 memory 和 action module，支持创建帖子、转发、关注、点赞、评论等多种操作。Time Engine 则用 24 维活动概率控制用户何时被激活，避免所有 agent 同时行动。

OASIS 的实验包括信息传播、群体极化和 Reddit 场景中的 herd effect。它观察到 LLM agent 在某些情况下比人类更容易表现出从众倾向，尤其面对负向初始反馈时更容易继续负向反馈。这与我笔记中的“幻觉与羊群效应衰减”问题相关：LLM agent 在群体交互中可能过度容易被说服，或者过度追随上下文中的多数信号。因此，如何为 agent 设计 cognitive stubbornness、独立判断、来源可信度评估和反操纵机制，是后续非常值得研究的方向。

### 6.4 AgentSociety：真实社会环境与社会科学实验平台

[AgentSociety](https://arxiv.org/abs/2502.08691) 则把社会模拟进一步扩展到城市、社交和经济空间。它强调一个真实社会模拟器应包含 LLM-driven social generative agents、realistic societal environment 和 large-scale simulation engine。论文中模拟了超过 10k agents 的社会生活，并覆盖数百万次 agent 与 agent、agent 与环境之间的交互。

AgentSociety 的一个重要观点是：真实社会中的行为不仅是语言互动，还包括移动、消费、社交、工作、经济交换、对政策和外部冲击的反应。它把 agent 行为分为 mobility behaviors、social behaviors、economic behaviors 等，并用 urban space、social space、economic space 组成环境。这样，社会模拟就不再只是社交媒体上的文本传播，而是更接近日常生活世界中的复杂系统。

它还强调社会科学研究方法的支持，例如 surveys、interviews、interventions。一个有价值的社会模拟平台不仅要能运行，还要能作为实验平台回答问题：如果引入 universal basic income，模拟个体的行为是否变化？如果发生飓风等外部冲击，城市系统和个体行动如何响应？如果有煽动性信息传播，系统是否会极化？

这些工作共同说明，LLM 社会模拟正在从单篇方法论文走向基础设施竞争。未来重要的不只是 agent 写得像不像，而是谁能提供真实人口、动态环境、推荐系统、可控实验、可解释评估和低成本扩展。

## 7. 研究展望：这个方向还能往哪里走

从 CNCC2025 讲座和这些论文来看，LLM Agent 社会模拟还远没有成熟。它的潜力很大，但关键问题也很硬。

第一是 **对齐与评估困境**。社会模拟最难验证。一个聊天机器人是否答对题可以用标准答案评估，但一个社会模拟是否“真实”没有单一答案。我们可以比较宏观统计，比如选票比例、态度均值、传播规模；也可以比较微观行为，比如某个用户是否发帖、立场是否一致、内容类型是否相似。但真正困难的是中间层：微观行为与宏观结果之间的机制是否合理。未来可能需要更多基于历史数据的 backtesting benchmark，用真实社交媒体事件、选举结果、政策变化或市场波动来回测模拟系统。

第二是 **个体保真度与群体代表性的矛盾**。一个高保真角色不等于一个有代表性的社会。Character-LLM 这类工作提高了单个角色质量，ElectionSim 和 SocioVerse 则提醒我们 population distribution 更重要。社会模拟需要同时回答两类问题：agent 是否像某类人，以及这些人按什么比例组成社会。只优化前者会得到漂亮但偏样本的模拟，只优化后者又会得到统计上正确但行为上贫瘠的模型。

第三是 **LLM agent 的从众与极化倾向**。OASIS 中关于 herd effect 的观察非常值得注意。LLM 在多轮交互中可能表现出过强的上下文顺从性：看到多数意见就靠拢，看到负面反馈就继续负面，看到极端表达就生成更极端表达。这可能与 RLHF 后的迎合性、上下文模式匹配、缺乏真实信念稳定性有关。未来可以考虑在 agent profile 或行为模型中显式加入 cognitive stubbornness、source trust、uncertainty、confirmation bias、反驳阈值等参数，让 agent 不只是会被影响，也会抗拒、怀疑或延迟更新。

第四是 **LLM 与 ABM 的接口设计**。HiSim 中的文本到 attitude score 转换只是第一步。更一般地说，未来社会模拟会大量出现跨表示交互：自然语言、数值状态、图结构、推荐分数、地理位置、经济变量、制度规则之间需要互相转换。如何设计稳定、可校准、可解释的接口，会决定混合框架能否扩展。比如核心用户的发言如何影响普通用户的态度？普通用户的分布变化如何反向影响核心用户的信息环境？平台推荐系统如何把全局热度转换成个体可见内容？这些都不是单纯 prompt engineering 能解决的问题。

第五是 **普通用户到核心用户的反馈闭环**。前面提到，HiSim 当前没有实现 ordinary users 对 core users 的反向影响，但这个方向很重要。在真实社会中，意见领袖也会观察群众反应：点赞、转发、评论情绪、民调变化、街头动员、媒体报道都会影响他们下一步行动。未来可以设计一个聚合模块，把普通用户态度分布、局部极化程度、传播热度和反对声音摘要成自然语言或结构化报告，再作为核心 agent 的环境输入。这样，混合模型会从“核心影响大众”变成“双向反馈系统”。

第六是 **伦理与误用问题**。社会模拟天然接近舆论预测、群体操纵和行为干预。如果系统能够模拟哪些信息更容易传播、哪些群体更容易被说服、哪些话术更容易极化，就可能被用于社会科学研究，也可能被用于操纵公众。因此，研究者需要明确数据隐私、模拟对象授权、结果发布边界和防滥用设计。Character-LLM 中对特定个人模拟的伦理提醒，在社会模拟层面会被进一步放大。

## 8. 小结：从“会说话的角色”到“可实验的社会”

回到 CNCC2025 讲座给出的主线，LLM Agent 社会模拟可以被理解为一个逐层放大的问题。

在 **Individual Simulation** 层，我们关心 agent 是否能扮演一个人。Prompt-based 方法成本低但容易漂移；fine-tuning-based 方法稳定但昂贵；identity-driven 方法试图用身份组合在成本和保真度之间取得平衡。

在 **Scenario Simulation** 层，我们关心多个 agent 是否能在具体场景中协作、竞争、讨论或完成任务。这里的核心不只是多智能体数量，而是领域任务适配、组织结构、记忆机制、工具接口和评估方式。

在 **Society Simulation** 层，我们关心大量异质个体能否产生可信的宏观社会现象。这里必须处理人口组成、网络结构、社会影响和系统输出。全部使用 LLM 不现实，因此 HiSim 这类 LLM + ABM 混合框架非常关键；ElectionSim、SocioVerse、OASIS 和 AgentSociety 则进一步把问题推向人口分布、社交平台、推荐系统、真实环境和社会科学实验基础设施。

如果说早期 LLM Agent 研究更多是在证明“模型可以像一个人一样行动”，那么社会模拟方向真正要回答的是：“我们能否用大量可控、可解释、可评估的智能体，构造一个足够接近真实世界的实验场？”这件事的价值不仅在 AI，也在社会科学。它可能让研究者以更低成本、更少伦理风险、更强可复现性的方式研究社会现象；但它也要求我们比一般 agent demo 更严肃地面对验证、校准、偏差和误用问题。

我对这个方向的一个初步判断是：未来有价值的工作不会只是在 prompt 里写更多角色设定，而会在 **个体建模、分布采样、交互网络、混合动力学、平台机制、历史回测和伦理约束** 之间建立更完整的工程和科学闭环。LLM 提供了语义层和行为生成能力，但社会模拟最终仍然是一门关于系统的学问。

## 参考资料

- [CNCC2025 基于大模型智能体的社会模拟 Tutorial](https://www.ccf.org.cn/Media_list/cncc/2025-08-20/847961.shtml)
- [From Individual to Society: A Survey on Social Simulation Driven by Large Language Model-based Agents](https://arxiv.org/abs/2412.03563)
- [Evaluating Character Understanding of Large Language Models via Character Profiling from Fictional Works](https://arxiv.org/abs/2404.12726)
- [Character-LLM: A Trainable Agent for Role-Playing](https://arxiv.org/abs/2310.10158)
- [Identity-Driven Hierarchical Role-Playing Agents](https://arxiv.org/abs/2407.19412)
- [Unveiling the Truth and Facilitating Change: Towards Agent-based Large-scale Social Movement Simulation](https://arxiv.org/abs/2402.16333)
- [ElectionSim: Massive Population Election Simulation Powered by Large Language Model Driven Agents](https://arxiv.org/abs/2410.20746)
- [SocioVerse](https://arxiv.org/abs/2504.10157)
- [OASIS: Open Agent Social Interaction Simulations with One Million Agents](https://arxiv.org/abs/2411.11581)
- [AgentSociety: Large-Scale Simulation of LLM-Driven Generative Agents](https://arxiv.org/abs/2502.08691)
