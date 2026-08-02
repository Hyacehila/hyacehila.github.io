---
title: "从 Build 到 Intervene：CNCC2025 LLM Agent 社会模拟 Paper 阅读路线"
date: 2026-08-02 22:30:00 +0800
categories: ["Agent Systems", "Computational Social Science"]
tags: [LLM Agent, Social Simulation, Agent-Based Modeling, Multi-Agent System, Paper Review]
author: Hyacehila
excerpt: "根据 CNCC2025 社会模拟 Tutorial 的完整转写，沿 Build、Align、Intervene、个体建模与规模化平台五条线，整理一份经过 arXiv 核验的阅读路线。"
---

这份短 review 来自 CCF 数字图书馆的 CNCC2025 Tutorial [基于大模型智能体的社会模拟](https://dl.ccf.org.cn/video/videoDetail.html?id=7730709021640704)。视频包括王翔、魏忠钰、高宸和陈旭的四场报告。顺着完整转写看下来，四场报告实际围绕同一个问题展开：怎样把“会扮演角色的 LLM Agent”变成可以支持社会科学研究的模拟器？

自动转写对英文题名和人名有不少误识别，所以我没有直接照抄。其中提到的论文均在 2026-08-02 按 arXiv 题名和编号重新核验；只有报告中明确出现、且对这条研究路线有帮助的工作才被保留。视频中没有公开题名或找不到可验证 arXiv 条目的内容，会在文末单独说明。

## 视频真正的论证结构

魏忠钰的报告给出了一条很清楚的三步法：

1. **Build**：定义 Agent、动作空间、社交网络、时间、推荐系统和环境规则。
2. **Align**：让外部事件、目标人口、个体行为和宏观分布与现实世界对齐。
3. **Intervene**：改变推荐、审查、信息投放或交换规则，比较反事实结果。

后两场报告继续把这条链向前推：AgentSociety 关心如何把社交、经济与城市物理空间放进统一环境；YuLan-OneSim 关心如何自动构造场景，并把“验证旧理论、提出候选扩展、再交给真人实验验证”连成研究流程。

我最后把判断标准收敛成一句话：

> 个体依据什么被构造，环境机制是否忠实，模拟是否同时通过微观和宏观对齐，干预结论是否能回到真实数据或真人实验。

## 1. 从传统 ABM 到生成式 Agent

传统 Agent-based Modeling, ABM 擅长把微观规则扩展成宏观过程，但其中的个体通常很“薄”：一个态度值、一个收益函数或几条状态转移规则。LLM 带来了语言、记忆、身份和情境判断，却同时引入了成本、随机性、模型先验和难以验证的新问题。

建议先读两篇综述，分别建立 ABM 和社会模拟的地图：

1. [Large Language Models Empowered Agent-based Modeling and Simulation: A Survey and Perspectives](https://arxiv.org/abs/2312.11970)，2023。高宸报告中明确提到的 ABM + LLM 综述。它从 ABM 对交互、自主和动态更新的要求出发，讨论 LLM Agent 能补上什么，以及感知、行动生成、效率和评测仍缺什么。
2. [From Individual to Society: A Survey on Social Simulation Driven by Large Language Model-based Agents](https://arxiv.org/abs/2412.03563)，2024。它按 individual、scenario、society 三层整理任务、组件、数据和评测，适合用来定位后续论文。

然后补两个历史起点：

- [Social Simulacra: Creating Populated Prototypes for Social Computing Systems](https://arxiv.org/abs/2208.04024)，2022。它在 Generative Agents 之前就用预训练语言模型填充社交计算原型。重点是“用合成人群测试产品设计”，还不是严格意义上的社会科学验证。
- [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442)，2023。重点读 observation、memory、reflection、planning 如何维持连续行为，以及消融实验实际证明了什么。25 个 Agent 的小镇是方法原型，不是规模化社会模拟的完成形态。

## 2. Build：先把社会机制写清楚

视频在 Build 部分反复强调四类对象：Agent 能做什么，谁能接触谁，时间如何推进，以及平台如何决定 Agent 看到什么。推荐系统不是外围组件，它本身就是社会机制的一部分。

这一阶段可以连读三篇：

- [S3: Social-network Simulation System with Large Language Model-Empowered Agents](https://arxiv.org/abs/2307.14984)，2023。它用真实社交网络和用户数据构造约 33K 个用户，关注情绪、态度与信息传播，是从小型角色互动进入真实网络结构的重要一步。
- [OASIS: Open Agent Social Interaction Simulations with One Million Agents](https://arxiv.org/abs/2411.11581)，2024。它把环境服务、推荐系统、时间引擎和可扩展推理拆开，能够支持百万级用户。阅读重点应是稀疏激活、信息流和异步执行，而不只是“1M Agents”这个数字。
- [MOSAIC: Modeling Social AI for Content Dissemination and Regulation in Multi-Agent Simulations](https://arxiv.org/abs/2504.07830)，2025。视频在干预页引用的 MOSAIC。它把内容注入、传播、用户互动和监管模块放进同一个社交网络模拟框架，适合观察平台规则如何变成实验变量。

幻灯片的发展路线图还列了三篇更聚焦的信息生态工作：

- [From Skepticism to Acceptance: Simulating the Attitude Dynamics Toward Fake News](https://arxiv.org/abs/2403.09498)，2024，即 FPS。它研究假新闻传播中个体态度的变化与控制。
- [The Stepwise Deception: Simulating the Evolution from True News to Fake News with LLM Agents](https://arxiv.org/abs/2410.19064)，2024，即 FUSE。它关心真实信息如何在多轮传播中逐步失真。
- [TrendSim: Simulating Trending Topics in Social Media Under Poisoning Attacks with LLM-based Multi-agent System](https://arxiv.org/abs/2412.12196)，2024。它把热搜形成与投毒攻击放进同一模拟过程。

三篇可以按“错误信息被接受、内容发生变形、平台趋势被操纵”的顺序选读。它们展示了不同层面的机制，但不能仅凭输出看起来像真实社交媒体，就认定传播因果机制也是真实的。

## 3. Align：对齐的对象不止是输出

视频把 Align 分为环境、目标用户和行为的对齐，评测又分为微观和宏观两层。微观层面看同一个人在同一时刻是否做出相似动作；宏观层面看态度分布、传播范围或时间序列是否接近真实社会。两者必须同时看，因为宏观曲线相似可能来自错误机制的抵消。

主线论文是：

1. [Unveiling the Truth and Facilitating Change: Towards Agent-based Large-scale Social Movement Simulation](https://arxiv.org/abs/2402.16333)，2024。HiSim 只让核心用户使用 LLM，普通用户使用传统舆论动力学模型，是“保真度与成本如何交换”的清楚样例。重点读两类行为引擎的接口，以及 SoMoSiMu-Bench 的微观、宏观评测。
2. [LLM Agents Grounded in Self-Reports Enable General-Purpose Simulation of Individuals](https://arxiv.org/abs/2411.10109)，2024。它用 1,052 名真实参与者的深度访谈构造 Agent，并用两周后的真人复测作为参照。它的重要结论不是某个准确率，而是人口统计标签远不足以支撑个体模拟，稠密的自述经历明显更有价值。
3. [SocioVerse: A World Model for Social Simulation Powered by LLM Agents and A Pool of 10 Million Real-World Users](https://arxiv.org/abs/2504.10157)，2025。它把可信模拟拆成 environment、user、scenario、behavior 四个对齐组件，并以千万级真实用户池支持目标人群采样。读这篇时应重点检查代表性、交互规则和行为模式如何分别被约束，而不是只看用户池规模。

### 个体建模的三种范式

魏忠钰报告随后把个体模拟分成三种路线，这一段是连接王翔“个性化记忆”和群体模拟的关键。

**第一种是非参数方法**：把画像、经历、记忆或相似用户放进上下文。可以配合阅读 [A-MEM: Agentic Memory for LLM Agents](https://arxiv.org/abs/2502.12110) 理解动态记忆组织，但记忆检索得分提高不等于人格和行为已经校准。视频还明确提到：

- [Scaling Synthetic Data Creation with 1,000,000,000 Personas](https://arxiv.org/abs/2406.20094)，2024，即 Persona Hub。它能低成本生成巨大的 persona 空间，但这些是合成多样性，不是真实人口的联合分布，更不是十亿个经验证的真人模型。
- [Persona-DB: Efficient Large Language Model Personalization for Response Prediction with Collaborative Data Refinement](https://arxiv.org/abs/2402.11060)，2024。它通过协作数据提炼和 persona 数据库做响应预测，适合思考冷启动用户如何与已有用户池匹配；论文并没有直接证明这种匹配足以支持社会模拟。

**第二种是参数化方法**：把人物经历或身份写进模型权重。[Character-LLM: A Trainable Agent for Role-Playing](https://arxiv.org/abs/2310.10158) 是典型起点；[Identity-Driven Hierarchical Role-Playing Agents](https://arxiv.org/abs/2407.19412) 则把复杂个体拆成可组合的身份维度，并用参数高效模块控制人格与职业等身份。它比“每个人训练一个完整模型”更可扩展，但仍依赖合成训练数据和量表评测。

**第三种是用户稠密表征或用户码本**：报告把它作为正在推进的 `Ours` 路线，希望用连续表征覆盖更多身份组合。视频没有给出可验证的公开论文题名，因此这里不为它强行匹配一篇相似工作。

这三种范式真正的分歧，不是 prompt 和 fine-tuning 谁更先进，而是个体信息从哪里来、如何压缩、能否跨情境保持稳定，以及能否被现实行为验证。

## 4. Intervene：从复现现象到可证伪实验

建好并对齐模拟社会后，才轮到修改推荐、审查、信息投放、网络连接或交换规则。视频中这条路线有四个层次。

1. [Simulating Social Media Using Large Language Models to Evaluate Alternative News Feed Algorithms](https://arxiv.org/abs/2310.05984)，2023。Törnberg 等人的工作直接比较不同信息流算法，是“把平台机制当实验变量”的早期代表。
2. FPS、FUSE、TrendSim 与 MOSAIC 分别提供假新闻、内容演化、趋势投毒和内容监管的实验对象。它们适合做机制对照，但不能代替真实平台上的因果识别。
3. [Emergence of Human-like Polarization among Large Language Model Agents](https://arxiv.org/abs/2501.05171)，2025。它在 LLM Agent 网络中观察回音室、backfire、selective exposure 和极化，并测试网络干预。最值得追问的是：这些机制由交互涌现，还是由 prompt、网络结构和基础模型偏置预先写入。
4. [Investigating and Extending Homans' Social Exchange Theory with Large Language Model based Agents](https://arxiv.org/abs/2502.12450)，2025。论文先检查 Homans 社会交换理论的六个命题能否在模拟中复现，再修改人格与交换规则寻找候选扩展，最后加入真人实验对齐。这个顺序比直接宣布“发现新社会规律”严谨得多。

我最看重的是 Intervene 这一步。一个逼真的虚拟世界只是演示；能重复运行、能修改机制、也可能被现实数据推翻，才像一套实验系统。

## 5. 规模化：两条路线，不是一个排行榜

视频展示了从一千、一万、十万到一百万 Agent 的规模比较，但 Agent 数量并不是统一的质量指标。不同系统模拟的动作粒度、激活比例、环境复杂度和 LLM 调用频率并不相同。

当前有两条主要扩展路线。

**混合建模**：HiSim 把高影响力用户交给 LLM，把大批普通用户交给 ABM。这条路线承认并非所有个体都需要同样昂贵的认知模型，关键难点是自然语言与数值状态之间的接口，以及普通用户对核心用户的反馈闭环。

**平台与推理基础设施**：OASIS 用模块拆分、稀疏激活和推荐系统扩展社交媒体环境；下面几篇则把环境继续扩展到经济与城市生活：

- [EconAgent: Large Language Model-Empowered Agents for Simulating Macroeconomic Activities](https://arxiv.org/abs/2310.10436)，2023。它用家庭、企业、政府和银行等主体研究劳动、消费和宏观活动，是理解 AgentSociety 经济模块的前置阅读。
- [AgentSociety: Large-Scale Simulation of LLM-Driven Generative Agents Advances Understanding of Human Behaviors and Society](https://arxiv.org/abs/2502.08691)，2025。它把城市物理空间、线下和线上社交、经济活动放进统一环境，并支持调查、访谈和干预。平台的可配置性与论文中具体社会结论应分开评价。
- [GenSim: A General Social Simulation Platform with Large Language Model based Agents](https://arxiv.org/abs/2410.04360)，2024。它是 YuLan-OneSim 的前置平台工作，重点在公共模块、十万级 Agent 和运行错误修正。
- [YuLan-OneSim: Towards the Next Generation of Social Simulator with Large Language Models](https://arxiv.org/abs/2505.07581)，2025。它从自然语言生成 ODD、行为图、规则、代码和分析流程，并提供默认场景。真正的风险在语义编译：生成的场景是否忠实表达了最初的社会科学问题。

陈旭报告最后提出 AI social scientist：从文献和问题出发，自动设计研究、构造场景、运行实验并生成报告。我更愿意把它理解成候选假设筛选器，而不是自动生产结论的机器。平台生成的结论仍然要交给领域专家、真实数据或真人实验审查。

## 最短阅读路线

如果按五轮阅读，我会这样排：

1. **方法地图**：ABM + LLM Survey -> From Individual to Society -> Generative Agents。先弄清传统 ABM 的优势、LLM 补了什么，以及 individual / scenario / society 三层的区别。
2. **构建环境**：S3 -> OASIS -> MOSAIC。比较动作空间、社交网络、时间引擎、推荐系统和监管模块如何进入模拟。
3. **对齐个体与群体**：Grounded in Self-Reports -> HiSim -> SocioVerse。依次看真实个体锚定、LLM + ABM 混合和四引擎对齐。
4. **做干预实验**：Törnberg News Feed -> Human-like Polarization。检查算法与网络干预的因果主张，对模型、prompt、种子和网络拓扑是否稳健。
5. **从平台回到科学**：AgentSociety -> YuLan-OneSim -> Homans SET。最后判断自动化平台如何先复现已知理论，再提出候选扩展，并把结论交回真人验证。

每篇论文都固定回答五个问题：模拟对象是谁；个体依据什么数据被构造；动作、网络和信息流由谁定义；用什么现实基准验证；结论对模型、prompt、随机种子和规模是否稳健。沿着这五问读，比较容易把“像社会的演示”和“能支持研究结论的模拟”分开。

## 核验边界

视频中还有几项值得继续追踪，但截至本次整理没有足够信息给出可信 arXiv 链接：

- **SocioBench**：幻灯片称其基于 ISSP，覆盖 10 个领域、约 5,000 名受访者和 600 个问题，但未给出公开论文题名。
- **用户稠密表征 / 用户码本**：报告标记为 `Ours`，没有给出论文名。
- **Social Simulation as a Service 与 Scenario Market**：报告明确说线上版本仍在收尾，属于平台设想和未发布系统，不能当成已发表论文。

因此，这些内容保留为观察点，而没有用名称相近的论文替代。

## 简评

我更愿意把这些系统看作低成本的假设生成器。虚拟人不能替代真人，但可以先筛掉一批不值得做、代价太高或风险太大的实验设想。视频给出的路径也很克制：先让 Agent 有可追溯的个体依据，再写清环境机制，用真实数据校准微观与宏观结果，最后才做干预。

目前最薄弱的仍是外部效度。更多 Agent、更长对话和更像真人的文本都不能自动解决它。下一阶段更值得关注的不是把规模从十万推到一百万，而是跨模型、跨时间、跨文化、带真人对照的验证协议，以及能够排除 prompt 暗示、训练语料复述和错误机制抵消的实验设计。社会模拟平台只有过了这一关，才可能从工程演示变成可靠的研究工具。
