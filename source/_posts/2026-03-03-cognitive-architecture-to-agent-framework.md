---
title: 从智能体的认知结构到智能体框架
title_en: "From Agent Cognitive Architecture to Agent Frameworks"
date: 2026-03-03 23:36:00 +0800
categories: ["AI & Agents", "Agent Architecture"]
tags: ["Tool Use", "Context Engineering"]
author: Hyacehila
excerpt: 本文从 CoALA 的认知结构出发，讨论 Workflow、Agent、Supervisor、Agent Team 与 MAS 的边界，并分析 LangGraph 等框架的工程价值、抽象代价和未来框架应该提供的基础设施。
excerpt_en: "Starting from CoALA's cognitive architecture, this post discusses Workflow, Agent, Supervisor, Agent Team, and MAS boundaries, plus the engineering value of frameworks such as LangGraph."
permalink: '/blog/2026/03/03/cognitive-architecture-to-agent-framework/'
---

## 引言：Building an LLM Agent

从 2025 年开始，语言模型智能体从论文和 Demo 进入了更多工程场景。学术界在讨论认知结构、记忆和多智能体协作，工业界则更关心工具调用、权限、状态恢复和可观测性。Agent 不再只是“模型多轮对话”，而是开始承担一部分原本很难稳定交付的开放式任务。

这篇文章不专门论证“为什么需要 Agent”。我只先放一个前提：LLM 给 Agent 带来了大量世界先验，让它不必像传统 RL Agent 那样只能在极度受限的环境里从零试错。

LLM Agent 本身并不神秘。它的基本逻辑是把模型从单纯的对话预测，放进一个能观察环境、调用工具、接收反馈并继续行动的循环里。只要会调用语言模型 SDK，让模型输出可解析结构，再把结构映射到工具调用，就已经在某种意义上开始开发 LLM Agent。

难点在循环之外。一个能交付的 Agent 需要回答很多具体问题：记忆如何组织，行动如何发生，反馈如何进入下一轮决策，权限在哪里被约束，失败后怎样恢复。只靠业务直觉做出来的 Agent 很容易停在玩具阶段。

这篇文章把 Agent Framework 看成一层工程翻译：它把 CoALA 里的记忆、行动、反馈、状态和权限，落到可观察、可调试的结构里。框架的职责是降低搭建和维护成本，不是替开发者定义 Agent。

因此本文会沿着一条线展开：先用 CoALA 建立智能体的认知坐标系，再看常见工程模式如何映射到这些认知模块；然后讨论 Single Agent 与 MAS 的边界；最后回到框架本身，解释 LangGraph 等框架为什么有用，为什么也会迅速变成技术债务，以及我心中更好的 Agent Framework 应该是什么样子。

## CoALA：智能体的认知结构

[CoALA（Cognitive Architectures for Language Agents）](https://arxiv.org/abs/2309.02427) 将语言模型本身视为认知架构的核心组成部分，并用附加模块弥补语言模型无记忆、难以与外部交换信息的问题。CoALA 不是一个具体工程框架，而是一套理解语言智能体的认知框架。先理解 Single Agent 的核心结构，才能进一步讨论 Workflow、Supervisor、Agent Team 和 MAS 的区别，也才能选择符合自己需求的技术去构造 Agent。

![CoALA cognitive architecture](/assets/images/agent-framework/coala-architecture.png)

*图源：[Cognitive Architectures for Language Agents](https://arxiv.org/abs/2309.02427) Figure 4。*

结合这张图看，CoALA 可以被拆成两层。左侧 A 是智能体的长期结构：程序记忆、语义记忆、情景记忆、工作记忆、决策程序和外部环境之间如何读写。右侧 B 是一次行动选择的局部循环：观察进入系统，经过 planning、proposal、evaluation、selection，最后执行行动，再把结果反馈回来。

因此 CoALA 认为一个语言智能体至少包含三类核心组成：

1. **LLM 交互**：包括提示、输出解析、结构化调用、模型选择等。LLM 是智能的主要来源，这一层是智能与外部工程系统之间的桥梁。
2. **内部记忆**：包括从长期记忆读取的 Retrieval、在工作记忆中更新信息的 Reasoning，以及把经验和知识写回长期记忆的 Learning。
3. **外部环境**：包括数字环境、物理环境、人类用户、其他智能体，以及各种可以被工具化的 API、文件系统、终端、浏览器、数据库。

从时间顺序上看，一个智能体会先从内部记忆和环境中获得上下文，再用 LLM 进行推理和决策，然后执行行动、观察反馈、更新工作记忆，必要时把重要经验写入长期记忆。这个循环让 LLM 从一次性生成器变成一个能够在环境中持续行动的系统。

CoALA 的核心执行单元可以理解为 ReAct Loop。虽然目前已经提出了多种 Agent 解决问题的方式，比如 Plan-and-Execute、Tree of Thoughts，但 ReAct Loop 仍然是最基础的结构。计划与推理可以被看作语言空间中的行动，计划被验证或修改时也会形成语言空间中的反馈。换句话说，Agent 的本质不是“调用了多少工具”，而是它是否有一个围绕目标、行动、观察和状态更新的闭环。

### Working Memory：当前决策周期里的工作台

工作记忆（Working Memory）将活跃信息保留在当前决策周期里，包括用户输入、当前目标、工具返回、推理产生的中间结论、从长期记忆检索到的知识，以及上一轮决策周期留下来的关键状态。

工程上可以把 Working Memory 理解为 Agent 的核心上下文窗口。它不是所有历史的堆积，而是当前任务切面上的工作台。进入这一工作台的信息，会直接影响模型接下来如何理解任务、选择工具和生成行动。因此 Working Memory 的管理，是 Agent 工程中的核心，大多数附加组件本质上都在围绕 Working Memory 做管理。

有限上下文窗口带来了两个问题。第一，信息放不下。第二，就算放得下，模型也未必能稳定使用它。长上下文并不天然等于有效工作记忆，[Lost in the Middle](https://arxiv.org/abs/2307.03172) 这类研究已经说明，模型对长上下文不同位置的信息利用并不均匀。因此 Working Memory 要决定什么该进场、什么该留下、什么该被压缩、什么该被退回长期记忆。

### Episodic Memory：经验与轨迹

情景记忆（Episodic Memory）存储决策经验和任务轨迹。它关心的是“过去发生过什么”：某个任务如何被拆解，哪个工具调用失败过，哪种 prompt 在某类场景中有效，某次用户偏好如何被表达。

当情景记忆被检索进 Working Memory，它会成为一种 in-context learning。Agent 不一定需要微调参数，也可以通过外部记忆复用历史经验。很多长程 Agent 的持续学习，并不是更新模型权重，而是把可复用的经验写成可检索、可审计、可压缩的外部记忆。

### Semantic Memory：知识与事实

语义记忆（Semantic Memory）存储更稳定的知识、事实、规则和领域材料。情景记忆强调经验，语义记忆强调知识库本身。RAG 中的文档、企业知识库、API 文档、产品规则、领域术语表，都更接近语义记忆。

语义记忆和情景记忆都会通过 Retrieval 进入 Working Memory，但它们的风险不同。情景记忆可能不适合迁移，语义记忆可能检索错误、上下文不完整或者知识过时（如果是一个时间相关的任务）。一个可用的 Agent Memory 系统，必须同时处理召回、排序、压缩、来源追踪、冲突和过期问题。后续关于 Context Engineering 和 Agent Memory 的文章会单独展开这些实现细节。

### Procedural Memory：流程、工具与行动规则

程序记忆（Procedural Memory）包括语言模型内建的程序性知识，也包括在智能体代码中直接编码的流程、工具定义、权限规则、状态转换和提示词模板。

工程框架中的图结构、条件边、工具 schema、system prompt、skills、MCP Server、审批规则，都可以看作程序记忆的一部分。程序记忆是启动智能体所必需的，但它也最容易被误用。自动生成 Workflow 本质上是在修改程序记忆：如果没有专家知识介入和验证，生成出来的流程可能看似合理，却在边界条件下无法纠错。

因此程序记忆既是稳定性的来源，也是僵化和技术债务的来源。Workflow、LangGraph 的 StateGraph、Dify 的工作流、n8n 的自动化流程，本质上都是在把一部分决策逻辑写入程序记忆。而更接近 MAS 的系统，则会把程序记忆从硬约束更多退回到软约束。

### Retrieval、Reasoning、Learning 与 Action

三类记忆的读写构成了内部动作空间，与之对应的外部交互构成外部动作空间。

**Retrieval** 从长期记忆检索内容到工作记忆里。如何理解检索与记忆、如何在工程上实现它，都值得单独讨论。在记忆增强智能体中，长期记忆和检索是一个问题的两个侧面，也是持续学习的核心。

**Reasoning** 读取工作记忆并更新工作记忆。它不只是思考，而是在当前目标、约束、反馈和记忆之间建立下一步行动依据。

**Learning** 将经验更新到情景记忆，将知识更新到语义记忆，也可能通过微调、规则更新或工作流修改影响程序记忆。在 CoALA 中没有特别强调遗忘，但在真实 Agent Memory 系统里，遗忘、冲突处理和版本治理是必须认真对待的问题。

**Action** 则把 Agent 从语言空间推向外部环境。外部环境可以是物理世界，也可以是数字系统、文件系统、终端、浏览器、人类用户，或者另一个 Agent。

**Context 很可能会成为 Agent 工程的核心。** 在线性注意力或更强记忆机制真正成熟之前，注意力机制带来的底层约束仍然是注意力不足和有限上下文。Agent 工程的发展史，很大程度上就是不断用外部系统设计弥补当前 LLM 在记忆、推理和行动上的短板。当这些短板被基础模型吸收，Agent 工程的一部分外壳也会自然消失；但在当下，CoALA 仍然是理解 Agent Framework 的重要起点。

## 智能体工程设计的本质

智能体工程设计要处理一个矛盾：**智能本身的涌现（Emergence）与业务鲁棒性（Robustness）的博弈**。

一个纯粹由 Prompt 驱动、完全自主的 LLM Agent 充满想象力，但也极易在复杂生产环境中产生幻觉、陷入循环或导致流程崩溃。相反，纯粹基于规则的代码虽然稳定，却无法处理开放世界里的模糊需求。智能体工程设计的目标，就是在性能、泛化能力和可控性之间取得平衡。

![Agent Design](/assets/images/agent-framework/agent-design.png)

这张控制流与架构图很好地概括了目前工业界构建 Agent 的核心手段。它覆盖了 Workflow、Map-Reduce、RAG、Agent loop、Supervisor、Multi-Agent 等模式。用 CoALA 的视角重新看，这些模式并不是互不相干的工程技巧，而是在补偿不同认知模块的短板。CoALA 原文也对现有研究（2024 年之前）进行了全面总结；如果你想先从研究者视角进入这个问题，值得先读一读原文，下面的解读会更偏向工程视角。

### Workflow：外置的程序记忆

图中纯粹的 Workflow，比如 `Summarize Email -> Draft Reply`，是单向或带少量分支的流程。

它的认知映射是 Procedural Memory 的外部硬编码。人类把 SOP、业务规则、异常处理和状态流转写进程序，LLM 只在某些节点里做局部判断或文本生成。

这也是 [Anthropic 关于 effective agents 的文章](https://www.anthropic.com/research/building-effective-agents) 中区分 workflow 与 agent 的核心：Workflow 是通过预定义代码路径编排模型和工具；Agent 则让模型动态决定自己的流程和工具使用。生产系统通常更偏向 Workflow，因为它稳定、可测试、可审批。代价是自由度较低，面对未定义异常时容易僵硬。

### Map-Reduce：绕开 Working Memory 限制

Map-Reduce 对应 Working Memory 的分治技巧。面对超大规模文档、长日志、多文件代码库或大批量输入时，一个上下文窗口无法同时承载所有细节。Map-Reduce 将材料拆成多个独立批次，分别让短生命周期的工作记忆处理，再把结果聚合回主工作记忆。

它解决的不是长期记忆问题，而是当前任务的工作记忆容量问题。它的价值在于隔离和压缩：每个 map worker 只看到局部上下文，reduce 阶段只接收结构化摘要、证据和必要引用。这样既降低 context bloat，也减少无关细节污染主 Agent 的决策。

很多 Harness 中的 subagent 也可以从这个角度理解。子 Agent 不一定意味着 MAS，它也可能只是一个工作记忆隔离器：主 Agent 把局部任务交给子 Agent，子 Agent 完成探索后只返回结论、证据和必要状态。

### RAG：长期记忆进入工作记忆

RAG 是长期记忆进入 Working Memory 的常见路径。向量数据库、关键词索引、文档库、代码索引、图数据库，本质上都是长期记忆的不同载体。检索过程负责把当前任务需要的信息召回，再经过排序、压缩和格式化进入工作记忆。**这里的 RAG 不特指向量检索增强生成。**

RAG 的认知映射是 Retrieval：从 Semantic Memory 或 Episodic Memory 中调取当前任务需要的材料。对 RAG 来说，真正困难的部分不只是能不能查到，而是查到的内容是否该进入当前工作台，是否有来源，是否过期，是否和已有记忆冲突。

RAG 是长期记忆进入当前决策周期的一条工程路径。更完整的记忆读写、压缩、更新和遗忘问题，应该放到专门的 Agent Memory 和 Context Engineering 中讨论。

### Agent / Supervisor：处理反馈的闭环控制

图中的 Agent 模式通常包含循环和条件分支，例如 `Draft -> Review -> Revise -> Approve`。Supervisor 模式则在外层引入一个调度者，让它决定下一步交给哪个 worker、是否继续、是否结束。

这两者可以放在同一类模式里理解：**它们都是处理外部反馈和自反馈的闭环控制结构，只是控制权集中程度不同。**

Agent loop 的关键是行动之后有观察，观察之后会更新状态，再根据目标和反馈决定下一步。Supervisor 只是把这个反馈处理中心化：由一个上层控制器读取全局状态，决定调用哪个角色、工具或子流程。

这类结构确实比单向 Workflow 更接近智能体，但它不自动等于 MAS。如果 supervisor、worker、reviewer 全部共享同一个全局 state，并且由同一套中心化条件边决定流转，那它更像一个单体 Agent 的多角色执行，而不是多个独立 Agent 的社会。

### Agent Team：动态认知路由

**当前工业界正在流行 Agent Team / Swarm（handoff 模式）：几个高度专精的 Agent 一起工作并彼此通信。**

这和静态 Workflow 不同。Workflow 是人类提前规定路径；Agent Team 则希望让 Agent 根据任务状态进行动态路由。它更接近人类组织中的专业分工：planner、researcher、coder、reviewer、operator 各自有边界，也通过消息协作。

但它也更容易发散。多个 Agent 对话越自由，越难判断谁拥有最终状态、谁负责终止、谁能覆盖谁的判断、错误从哪里进入系统。这种模式是**智能本身的涌现（Emergence）与业务鲁棒性（Robustness）的博弈，智能在此略胜一筹。**

Agent Team 是通向 MAS 的入口，但它不是魔法。它需要明确的通信协议、状态边界、权限模型和终止条件。我们距离一个真正的多智能体系统可能还很远，也可能就在明天。

## Single Agent or MAS

当工程图谱延伸到 Multi-Agent、Supervisor、Agent Team 时，一个争议问题就出现了：究竟什么是单体智能体，什么又是多智能体系统？

**角色数量、prompt 数量、节点数量都不定义边界。真正的边界在于工作记忆归属权、控制流自主性，以及通信是否被环境化。**

### Workflow：LLM 节点化的流程系统

第一类是 Workflow。它可以是传统 workflow，也可以是 AI-driven workflow。核心特征是流程由人类或代码定义，LLM 只是流程中的局部能力节点。

在这种系统里，LLM 可以做摘要、分类、抽取、判断、重写，也可以调用工具，但它并不拥有整体控制流。它没有决定“自己接下来该做什么”的自治权，也没有独立的工作记忆主权。它更像自动化系统中的智能组件。

这类系统并不低级。相反，它往往是最容易生产落地的 agentic 形态。大多数企业任务并不需要一个完全自主的 Agent，只需要在可靠流程中嵌入少量语义判断。对真实生产环境来说，追求技术上的新奇有时反而会走偏；更重要的还是先把业务理解透，再把其中稳定的部分自动化。

### 伪 MAS：多 prompt、多角色、多节点

第二类是伪 MAS。LangGraph 主导的很多多角色系统都属于这一类：系统里有多个 prompt、多个节点、多个角色名；有 supervisor、researcher、coder、reviewer，但它们共享全局 state，由中心化图结构和条件边控制流转。

这不是贬义。它的价值很明确：角色分工让提示词更清楚，图结构让状态流转可控，checkpoint 让长任务可恢复，分拆机制也显著降低了单个智能体的工作记忆负担。但从认知结构上看，它仍然更接近一个 Single Agent：一个大脑在不同阶段戴上不同面具，通过中心状态机切换工作模式。

典型特征包括：

- 节点是否被调用由 graph/router 决定，而不是由 Agent 自主决定是否响应。
- 通信不是环境中的消息传递，而是函数调用和状态更新。

因此，LangGraph 式 supervisor 是条件边上的多角色 Agent，不是严格意义上的 MAS。它适合工程控制，但不该被误认为多个自治智能体组成的社会。

### 真 MAS：基于消息和边界的 Agent Team

第三类是真 MAS 或更接近 MAS 的 Agent Team。它的关键不是有很多 Agent，而是每个 Agent 至少拥有相对独立的工作记忆、工具边界、目标解释权和消息接口。

在这种结构里，其他 Agent 对某个 Agent 来说属于外部环境。Agent A 看不到 Agent B 的完整内部状态，除非 B 主动通过消息共享。A 发出的请求也不一定强制 B 执行，B 可以回复、拒绝、转发、请求更多信息，或者根据自己的策略调用工具。

这更接近计算机科学中的 Actor Model 或消息社会。它的优势是自治性和可组合性，弱点则是收敛、观测和治理成本。完整开放的 MAS 很容易发散，因此真实生产系统常常会在 MAS 外层再套一层 Workflow。这样会形成一种混合架构：内部有独立 Agent 通过消息协作，外部仍由确定性流程约束任务边界。

### 为什么这个区分重要

前面把 Workflow、伪 MAS 和更接近真实协作的 Agent Team 分开，并不是为了做术语考据，而是因为一旦分类混淆，工程判断就会跟着失真。

如果把所有多 prompt、多角色、多节点系统统称为 MAS，开发者就很容易误判自己面对的问题。你以为自己在设计多智能体协作，实际上可能只是在维护一个中心化状态机；你以为系统失败是因为多个 Agent 没协同好，实际上真正失控的往往只是共享 state 持续膨胀、路由条件越来越脆、上下文被中间噪音污染。

更重要的是，不同结构真正需要的能力也完全不同。Workflow 的核心问题是编排可靠性：顺序是否可控、节点是否可回放、失败是否可恢复、审批与日志是否完备。伪 MAS 的核心问题是状态治理：共享上下文如何裁剪、checkpoint 如何恢复、条件边如何保持透明、局部探索如何不污染全局。更接近真实协作的 Agent Team，才会把问题推向消息协议、handoff 语义、权限边界、冲突解决和终止策略。

所以，这个区分真正决定的不是命名，而是你到底应该把工程精力花在哪一层。

### MAS or Single Agent and Dynamic Workflow

把边界讲清楚以后，再看一些新内容，其实都绕着同一个问题打转：系统应该把不确定性放在哪里？放进一个连续的单体 Agent trace，放进多个彼此隔离的 Agent，还是放进可被动态改写的 Workflow。

2026 年 6 月补：Claude Code 的 [Dynamic Workflows](https://claude.com/blog/introducing-dynamic-workflows-in-claude-code)，本质上是在动态调整程序记忆，让 Agent 在运行中生成、修改和执行流程。我和淚笑在 1 月聊过很接近的东西：人工写一个 DAG 的 YAML 文件，再用一套 Runtime 执行。问题也很快出现：YAML 表达力不足，写着写着就变 DSL；人写坐牢，AI 写费 token。[intentlang](https://github.com/l3yx/intentlang) 更接近我能接受的方向：把专家经验写成硬约束，但别让专家被 Agent 概念绑架。Claude Code 这次只是把载体换成自生成的 JS 和解析 Runtime，路线并不神秘。

危险也在这里：Dynamic Workflow 等于让 Agent 改写程序记忆。模型强了，但还不到默认放权的程度。必须配 subagent 权限、工具边界和验证点；否则它带来的不只是动态流程，还有动态事故入口。

Anthropic 和 Cognition 在多智能体系统与单一智能体的分歧，也可以这里比较。Anthropic 的 [multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) 适合研究任务：lead agent 调度多个 subagent 并行，内部评测比单 agent Claude Opus 4 高 90.2%，复杂查询时间最多缩短 90%。代价是 token，普通 agent 已经比聊天多烧约 4x，多智能体系统约 15x，token 用量本身还能解释 80% 的性能方差。

Cognition 的 [Don't Build Multi-Agents](https://cognition.ai/blog/dont-build-multi-agents) 则提醒编码任务不要默认多智能体。coding agent 往往需要完整 trace，行动又隐含设计决策；一个 subagent 画背景和管道，另一个画小鸟，最后很容易拼不出同一种视觉风格。子任务一旦高度耦合，局部自治就会变成全局冲突。

所以这不是互相矛盾的Blog，他们在具体具体的细节内容中其实也一定程度支持对方的观点但不体现在标题里。研究任务天然可并行，证据可以汇总；编码任务状态耦合高，连续 trace 更值钱。我的判断很简单：信息广、状态松、预算够，用 MAS；决策连续、风格一致、隐含约束多，用 Single Agent；流程清楚且步骤解耦且长的夸张，用 Dynamic Workflow。真正该看的不是角色数量，而是并行度、状态耦合、trace 是否必须共享、权限能否隔离、失败半径多大。

### Workflow or Agent：为什么生产系统最终常常走向混合架构

但理解了边界之后，另一个现实也会变得很清楚：真正的生产系统，往往不会停在某一种纯粹形态上。

纯 Workflow 的优点是稳定、可审计、可控，但它对开放任务、长链探索和未定义异常的处理能力有限。纯 Agent 的优点是灵活、自主、能在模糊问题里试错推进，但代价是上下文容易膨胀、推理容易漂移、系统行为也更难预测。二者各自走到极端，都很难单独支撑复杂生产任务。

因此，现实里的常见解法不是在 Agent 和 Workflow 之间二选一，而是把它们做成分层的混合嵌套架构：在外围用 Workflow 锁定任务边界、预算、审批和终止条件；在局部高不确定性的环节，放入具备自主规划能力的 Agent；当单个主 Agent 的上下文继续膨胀时，再把独立子任务下放给 Subagent，只回收压缩后的结论，而不是把整段探索历史直接灌回主上下文。

这样一来，Workflow 提供的是外层秩序，Agent 提供的是局部智能，Subagent 负责切断上下文污染。我们真正设计的，不再是到底要不要 Agent，而是：哪一层必须刚性约束，哪一层可以释放自主性，哪一层必须隔离记忆与推理过程。

但无论外层选 Workflow，还是在局部放入自主 Agent，真正的前提都不是框架，而是开发者对业务的理解。一个可靠的领域 Workflow，本质上是在把领域专家对问题定义、状态流转、异常分支和验收标准的理解写进系统；一个可信的自主 Agent，也需要有人先说明目标边界、反馈信号、权限和失败恢复。AI 很难替你解决一个你自己都还没有定义清楚的问题。很多时候，开发这类系统最朴素的方法不是再找一个框架，而是每天和业务方喝杯咖啡，把那些只存在于经验里的判断一点点问出来。

这也是如今 Agent 框架让人眼花缭乱的原因之一。用户还没建立对 Agent 本身的理解，就被不同框架的抽象方式干扰，难以形成对 Agent 设计的稳定认识。

但框架的意义就在这里：它不应该替我们模糊这些边界，而应该帮助我们把这些边界显式化。

## 智能体框架：从认知脚手架到工程基础设施

如果用一句话定调：**框架不是在提供智能，而是在冻结如何管理 LLM 不确定性的工程约定。** 不同框架的真正区别，不是名字或界面，而是它们选择把什么当成核心抽象单元，把什么交给模型自由发挥，又把什么当作固定的工作流。

### LangChain / LangGraph：从 Chain 到 StateGraph

LangChain 代表了 Agent 工程最早一波非常典型的思路：先把 Prompt 调用、工具调用、RAG、记忆这些常见动作封装成可拼装组件，再通过 Chain 把它们串起来。它提供的不是某一个固定智能体，而是一套把 LLM 当成应用组件来组合的接口。它第一次把 LLM 放到一个应用软件中。

但当 Agent 从一次性调用走向长任务、Loop 和中途恢复时，单纯的 Chain 很快不够用了。LangGraph 于是把核心抽象升级为显式的 `StateGraph`：共享状态、节点、边、条件路由、循环、检查点恢复。它的强项非常明确：状态机思维、图流转架构、人类审批、长任务恢复、失败重试和可回放。

也正因为它强，LangGraph 很容易反过来塑造开发者的思维，让人先学框架语法，再理解任务本身。它的问题不是框架不够好用，而是**不适合作为默认 Agent 心智模型**。而且智能体技术本身的迭代发展迅速，过早的高层次封装很快变成了技术债务。

这种债务首先会影响 coding agent。人类开发者可以临时读文档、记住接口迁移；模型却更容易被旧版本文档、训练语料和项目局部代码干扰。它修改一个 LangGraph 项目时，不只是修改业务逻辑，而是在修改一套快速演进的控制语言：`StateGraph`、state schema、conditional edge、message reducer、checkpoint、node side effect。

问题不在于这些概念本身复杂，而在于它们把错误空间扩大了。coding agent 必须同时判断 bug 来自业务代码、节点副作用、状态合并、路由条件，还是框架版本差异。高层封装一旦成为额外的解释层，维护成本就会从理解任务变成先理解框架如何命名任务。

这种债务也会影响开发流程本身。`AnalysisPosts` 里第一个阶段是并发清洗。它不能简单使用 OpenAI-compatible 的请求模式，因为项目需要更细粒度的原生 SDK 控制：`ZaiClient` 复用、请求级 `timeout`、视觉输入、`thinking` 参数，以及BatchAPI和KV-Cache。

LangGraph / LangChain 的模型接口没有直接支持这个 SDK，只能在 node 内部手写原生 Python 调用。问题不只是多写几行代码，而是抽象中心发生了转移：真正决定系统可靠性和能力边界的模型请求、重试、并发和参数控制，都已经不在框架里，而在 node 里的原生代码里。当这部分关键链路全部回到原生 Python 后，LangGraph 提供的便利就只剩外层流程编排，不能再承担核心 runtime 的角色。

第二个具体场景是 Stage2 的 DataAgent 和 SearchAgent。它们不是普通函数分支，而是两个小 ReAct Loop：各自会产生工具调用、工具返回、失败尝试、中间观察和局部判断。这些内容应该属于各自的私有工作记忆，而不应该默认写入全局 `messages`。如果按照 LangGraph 的默认心智模型，把协作理解成共享 state 上的节点流转，再让两个分支都更新同一个 `MessagesState`，那么 `add_messages` 或 reducer 会把两边轨迹累积到同一条消息流里。后续 supervisor 或其他节点读到的就不只是最终产物，还包括另一个 Agent 的内部噪音。

这和 `AnalysisPosts` 的需求正好相反。它需要的是**有限信息交换**：DataAgent 和 SearchAgent 可以各自保留局部工作记忆，但主流程只接收 charts、tables、insight_provenance、少量 trace 或压缩摘要。LangGraph 不是完全不能做这种隔离；可以用 subgraph、不同 state schema ，然后手动完成 parent state 到 subgraph input、subgraph output 到 parent state 的映射。但这样一来，真正困难的部分就回到了原生代码：哪些消息允许离开子 Agent，哪些字段应该被过滤，哪些结果可以 merge，哪些中间轨迹必须丢弃。框架提供的是共享状态图，而这个需求需要的是私有工作记忆和显式通信协议。对这种场景，LangGraph 的默认抽象不是优势，反而容易诱导开发者把不该共享的上下文放进全局状态。这是一个常见的Agent Dev的需求，但是Langraph在最初的设计中没有考虑到这个问题，被迫增加一层并不算轻量的补丁，他并不好用。

所以我对 LangGraph 的判断是平衡的。它适合复杂状态机、可恢复长任务、多步骤审计和需要显式持久化的流程。但如果你的核心问题是设计 Agent 本体、控制上下文污染、定义环境反馈、管理权限边界，那么直接从 LangGraph 开始，可能会让你过早把问题翻译成图，而不是先把任务本身想清楚。

### Dify / Coze / n8n：平台式、产品式与自动化式框架

Dify、Coze、n8n 可以放在一起讨论，因为它们都不是以“写一个 Agent runtime”为第一目标，而是把 Agent 能力包装成更易交付的应用或流程。但三者的重心并不相同。

**Dify 更像面向应用交付的 Agent 平台。** 它把知识库、工具、工作流、观测、部署和运营后台放进一个整体产品里。对问答、助手、检索增强、表单处理、内容生产这类业务 Agent，开发团队往往不想从零搭建记忆层、日志面板、发布链路和后台配置系统。Dify 的价值在于让应用先跑起来。

它的代价也来自平台化封装。越是复杂的控制流、越是细粒度的状态设计、越是需要底层透明调试的场景。当任务需要自定义恢复逻辑、精细化上下文裁剪或非标准工具路由时，平台便利性会变成工程自由度的上限。

**Coze 更像面向产品入口的 Agent 工厂。** 它关心的不只是后端编排，而是 Persona、知识、工具、插件、工作流、聊天界面和发布渠道如何组合成一个可被用户直接接触的产品。对很多团队来说，他们真正想做的不是一个后端框架，而是一个能上线、能运营、能分发的 Agent 产品。

Coze 的优势是产品化入口和能力拼装，代价是底层工程细节被平台替你做了选择。对于快速做终端产品，这是好事；对于想深入控制状态结构、协议层、日志系统或跨系统 runtime 的团队，这种便利也意味着更强约束。

**n8n 的出发点则不是 Agent，而是自动化。** 它原本解决的是如何把不同系统和 SaaS 服务可靠连起来，因此真正强项是连接器生态、触发器机制和可视化流程编排。当 AI 节点接入后，n8n 形成的是“确定性流程 + 局部智能节点”的范式。

这类范式特别适合“流程自动化 + 一点智能”的任务：表单收集、消息路由、摘要生成、规则触发后的检索补充、跨系统同步时的语义分类。它非常有价值，但不天然适合复杂记忆管理、长期状态恢复和多 Agent 协商。n8n 擅长把 AI 嵌进流程，不擅长把流程演化成完整 Agent runtime。

类似 n8n 的连接器生态会继续成为重要基础设施，但 MCP、Agent Skills 和 coding agent 的成熟会改变 no-code 平台的边界。曾经我们在 no-code 和编码之间抉择，现在越来越像在 no-code 和 vibe coding 之间抉择。

### AutoGen ：Agent Team 的吸引力与风险

[AutoGen](https://microsoft.github.io/autogen/) 代表的是另一条路线：不先把世界画成图，而是先把世界拆成角色，再让这些角色通过消息传递和对话协作。不同 Agent 拥有不同职责、提示与工具，通过会话、handoff 和任务分解完成目标。

这种范式天然贴近人类对团队协作的直觉。你可以很快搭出 planner、coder、reviewer、tool agent，让它们像开会一样交换信息。这对展示多 Agent 的组织形态、探索开放式任务分工、研究自主协作上限都很有价值。

这类框架的风险也非常直接。Agent 间对话越自由，越容易撞上收敛性、可观测性和生产稳定性问题。任务为什么发散？哪个 Agent 真正拥有状态主权？谁负责停止？某个错误判断被另一个 Agent 接受后，如何追踪责任链？这些问题一旦进入真实业务，调试成本会迅速上升。

因此 AutoGen 式系统非常适合帮助我们理解多 Agent 的潜力，也适合探索真 MAS 的协作边界。但它们不天然等同于最容易落地的生产方案。越接近真 MAS，越需要消息协议、私有记忆、审计、权限和终止条件。而目前这些harness该怎么设计，还没有一个定论。

### OpenAI Agents SDK：轻 runtime 的对照

[OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) 提供了另一个值得对照的方向：Agents、handoffs、guardrails、tracing。它不像 LangGraph 那样把核心抽象压到图和状态机上，而是更接近一个轻 runtime：给模型、工具、交接、安全边界和轨迹观测提供基本结构。

这种轻 runtime 的意义在于，它不急着替开发者画完整状态图，而是先把 Agent 系统中最容易重复的部分做成基础设施。handoff 处理控制权转移，guardrails 处理输入输出边界，tracing 处理可观测性。这些能力并不定义 Agent 本体，却能让开发者更容易构建自己的 runtime。

这并不意味着轻框架一定更好。它也会留下更多设计责任给开发者。但在 Agent 规范尚未稳定时，轻框架的优势是少替你决定，少制造绕不开的 DSL。

### 小结：这些框架冻结了什么约定

| 框架 | 抽象单元 | 控制流 | 状态模型 | 自主性 | 适用场景 | 主要代价 |
| --- | --- | --- | --- | --- | --- | --- |
| LangChain / LangGraph | Chain / Node / StateGraph | 显式图路由与循环 | 共享 State + checkpoint | 中等，可被图精确约束 | 多步闭环、长任务、审批、恢复 | 抽象厚，容易先学框架再理解任务 |
| Dify | 应用、知识库、工具、工作流 | 平台式编排 | 平台托管状态 | 中低 | 问答、业务助手、快速交付 | 灵活度有限，底层细节难伸入 |
| Coze | Persona、插件、工作流、产品入口 | 产品化对话入口 | 平台内聚状态 | 中等 | 面向终端的 Agent 产品 | 平台约束强，工程透明度有限 |
| n8n | 自动化节点 | 确定性流程 + AI 节点 | 流程上下文 | 低到中 | 系统集成、流程自动化 | 不是原生 Agent runtime |
| AutoGen / CrewAI | Agent / Message / Crew | 对话协作或 Flow 控制 | 分散状态或会话状态 | 高到中 | Agent Team、多 Agent 实验 | 易发散，状态主权和终止难治理 |
| OpenAI Agents SDK | Agent / Handoff / Guardrail / Trace | 轻 runtime | 开发者自定义 | 中高 | 自研 Agent runtime、轻框架集成 | 基础设施少替你做决定 |

收束来看，这些框架不是在回答“谁更先进”，而是在分别回答三个问题：如何约束模型，如何组织状态，以及如何处理不确定性。所谓框架差异，归根结底就是这三道题的不同解法。

## 当框架先于规范：智能体框架为何迅速变成技术债务

问题从来不是框架存在，而是智能体设计规范尚未稳定时，框架过早把一套暂时性的做法固化成高抽象。于是很多框架并不是在沉淀已经被证明的工程规律，而是在把某个阶段看起来够好用的经验提前神圣化。

换句话说，框架一旦从工具变成前置心智模型，就会把 Prompt、工具、记忆、检索、状态恢复这些尚未稳定的工程选择，伪装成一套稳定规范。它最危险的地方不在代码多一层，而在失败定位多了一层：开发者必须先判断问题来自模型、业务、环境，还是框架自己冻结的抽象。

### 规范未定，抽象先行

今天我们仍然没有真正稳定下来的 Agent 设计规范。Context 应该如何分层？Memory 应该暴露为数据库、缓存还是显式写回操作？Tool Use 的权限边界如何定义？失败恢复是按节点、按步骤还是按任务？评测到底评过程、评结果还是评轨迹？这些核心原语都还在快速变化。

框架在这种时候进行高层封装，就会把阶段性答案包装成通用真理。比如把 Agent 先翻译成图，把角色先翻译成节点，把交互先翻译成全局 state，把记忆先翻译成某种 store。它们都可能有用，但都不应该被误认为 Agent 的本体和解决问题的唯一方法。

### 过早抽象抬高了心智负担

过早抽象会改变学习顺序。很多开发者在尚未理解 Agent 本体之前，就先被迫学习 `StateGraph`、`Router`、`Node`、`Agent Team`、`Memory Store`、`Reducer`、`Checkpoint` 这些框架语汇。结果不是先理解 Context、工具和状态，再选择合适抽象；而是先学习框架如何命名问题，再倒推 Agent 到底是什么。

对 coding agent 来说，这个问题会更严重。人类至少还能读文档、记住版本差异、理解隐含约定；模型面对高层 DSL 和频繁变化的框架接口时，往往更容易写错。越是高层次和快速变迁的封装，越难被未来的 agentic coding 稳定维护。

### 黑盒封装削弱了可观测性

框架的另一个代价是把关键工程细节藏起来。Prompt 怎么拼出来的，状态什么时候 merge，记忆何时被读取或写回，失败后谁决定重试，某次恢复到底读了哪个 checkpoint，很多时候都被封进框架默认逻辑里。

系统顺畅时，这些封装很省心；系统失控时，它们就变成最难拆开的黑盒。对于 Agent 这种本就充满不确定性的系统，可观测性不是锦上添花，而是生产可用的前提。

### 模型进步会让封装快速过时

模型能力的演进速度会进一步放大这个问题。原生 tool calling、更长上下文、并行工具调用、structured output、供应商 SDK 自带 tracing、模型内置 web search 和代码执行，都会改变哪些抽象是必须的。

很多曾经聪明的封装，几个月后就可能变成多余的一层补丁。开发者开始把大量时间花在怎么绕开框架限制上，而不是花在任务建模和业务逻辑上。尤其当核心问题是环境、权限、反馈、状态治理时，自研原生 runtime 往往比套通用框架更直接。

收束来说，框架依然有用，但它应该帮助我们暴露问题，而不是替我们隐藏问题；应该降低认知负担，而不是制造新的认知债务。

## Claude Code 与 OpenClaw：为什么前沿系统回到原生开发

如果把前面几类框架放在一起看，一个更有意思的对照就会浮现出来：许多真正有效的前沿系统，并不是先选一个通用框架，再把业务塞进去；相反，它们往往是先围绕具体环境搭出原生 runtime，再决定哪些部分值得抽象。

换句话说，LangGraph、AutoGen、OpenAI Agents SDK 这些框架讨论的是“如何抽象 Agent”；而 Claude Code 与 OpenClaw 更像是在回答另一个问题：当任务环境本身已经足够丰富时，系统是否应该先贴着环境长出来，再决定哪些能力值得被框架化。

### Claude Code：围绕编码工作流的原生闭环

Claude Code 的意义，不在于证明某个通用 Agent Framework 足够强，而在于它把编码任务空间组织成了一个高反馈密度的原生闭环。代码库可读，终端可操作，测试和构建能提供快速反馈，Git diff 与 code review 构成天然的人类审查位点。

这套环境本身已经非常适合 Agent：文件系统提供外部记忆，终端提供行动空间，测试提供反馈，Git diff 提供状态边界，review 提供 human-in-the-loop。它不需要先被翻译成一个通用图框架，才能让模型工作。

因此 Claude Code 的成功更像是任务结构先跑通，而不是框架抽象先统一。它让模型直接工作在真实开发环境里：读代码、改代码、运行命令、观察结果、再继续迭代。这是一种原生 runtime 思路，框架在这里只是辅助手段，不是中心。

### OpenClaw：围绕个人环境的原生系统拼装

OpenClaw 暴露的是另一类问题。它面向的不是高度标准化的开发流程，而是聊天、语音、设备、自动化、个人知识、长期记忆、权限控制这些异构而敏感的现实表面。

在这种任务空间里，困难的部分不是先声明几个 Agent 角色，而是怎样把碎片化环境组织成一个可被感知、可被操作、又不至于失控的系统。设备接入怎么做，权限怎么隔离，自动化如何触发，长期记忆怎样管理，技能系统怎样渐进暴露，用户如何随时接管，这些都是 runtime 和系统工程问题。

如果过早套入一个通用 Agent 抽象，反而可能掩盖真正难的问题。更合理的顺序是：先定义环境、反馈、权限、恢复与任务结构，再决定哪些部分值得框架化。

### 轻封装能力单元：Skills 的位置

Skills 这类轻量封装在这一阶段有吸引力，也是这两个产品很重要的组成部分，原因正是它不急着把能力塞进厚重框架。它把说明书、脚本、模板和参考资料打成一个渐进披露的能力目录，适合承载规则、流程和组织经验，却不强行假装自己已经是完整 runtime。关于 Skills 更详细的讨论，可以参考我的另一篇博客 [《从 MCP 到 Agent Skills》](/blog/2026/03/10/from-mcp-to-agent-skills/)。

但 Skills 也提醒我们：轻封装不等于没有代价。模型面对的是半结构化文档与脚本，它仍然需要自主判断何时启用、读取什么、是否执行。这里的问题不只是有没有框架，而是系统是否把不确定性放在了正确的位置：该交给模型的交给模型，该用工程约束锁死的部分必须明确锁死。

## 我理解中的智能体框架：它真正应该提供什么

如果承认重框架很容易制造心智负担，下一步就不是拒绝框架，而是收窄它的职责边界。一个好的框架只该提供那些反复需要、又不值得每个团队重写的能力。

首先是透明的状态管理。上下文里有哪些内容，状态如何更新，检查点在哪里，任务如何恢复，都应该显式可见、可被审计、可被人理解。状态不透明，Agent 就只能停留在看起来会动的阶段。

其次是原生工具与协议接入。MCP、函数调用、外部 API、文件系统、消息系统，乃至 Skills 这样的轻封装能力单元，都应该能被直接接入，而不是被迫再套一层框架自己的专有协议。好框架应该减少适配成本，而不是制造新的兼容层。

第三是可观测性。Prompt 如何被组装，工具何时被调用，状态发生了什么变化，哪个节点触发了重试，整条轨迹是否可以 replay，这些都不该依赖框架黑盒 UI 去猜。Agent 不是传统后端服务，观测能力越弱，调试成本越高。

第四是记忆接口。无论是 episodic memory、semantic memory 还是 procedural memory，框架都不应假装自动帮你管好了。它应该把读写入口、检索策略、压缩、淘汰、冲突和 provenance 暴露出来，让开发者清楚记忆究竟如何影响当前决策。

第五是 human-in-the-loop 与权限边界。审批点、人工接管、回滚点、危险操作确认，不应该靠业务方事后补洞。智能体一旦走进真实工作流，人类不只是兜底者，也是系统的一部分。想要完全替代人，通常只能把人的经验提前封装成 Workflow；对于自主决策 Agent，human-in-the-loop 仍然是可靠性的关键部分。

最后，它还应该对 agentic coding 友好。随着模型编码能力继续增强，越来越多系统会让 Agent 直接阅读、修改甚至扩展自己的运行逻辑。框架越贴近原生语言、原生工程对象和原生工具链，就越容易被人和 Agent 一起理解；越是高层魔法式封装，越容易成为后续自动化维护的阻力。

归根结底，好框架应该减少认知债务，同时保留工程控制权。它不替开发者思考 Agent 应该是什么，而是在开发者已经理解任务结构之后，提供那些值得复用的基础能力。

我更期待的是更薄、更透明、更接近任务环境的框架。它应该让 CoALA 中的记忆、行动、反馈、状态和权限变得可见、可控、可替换，而不是把这些问题包进一个看似高级、最后又必须被绕开的黑盒里。

## 参考资料

- [Cognitive Architectures for Language Agents](https://arxiv.org/abs/2309.02427)
- [Building effective agents](https://www.anthropic.com/research/building-effective-agents)
- [Introducing dynamic workflows in Claude Code](https://claude.com/blog/introducing-dynamic-workflows-in-claude-code)
- [intentlang](https://github.com/l3yx/intentlang)
- Anthropic, [How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system)
- Cognition, [Don't Build Multi-Agents](https://cognition.ai/blog/dont-build-multi-agents)
- [LangGraph documentation](https://langchain-ai.github.io/langgraph/)
- [OpenAI Agents SDK documentation](https://openai.github.io/openai-agents-python/)
- [Microsoft AutoGen documentation](https://microsoft.github.io/autogen/)
- [CrewAI documentation](https://docs.crewai.com/)
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
- [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172)
