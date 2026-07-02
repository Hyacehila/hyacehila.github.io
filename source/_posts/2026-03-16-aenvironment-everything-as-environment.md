---
title: AEnvironment：Agent Dev 为什么需要交互环境层？
title_en: "AEnvironment: Why Agent Development Needs an Interaction Environment Layer"
date: 2026-03-16 21:00:00 +0800
categories: ["AI & Agents", "Agent Architecture"]
tags: ["Agent Environment", "SWE-bench", "Tau-bench", "Verifiers", "Evaluation", "Environment Design"]
author: Hyacehila
excerpt: SWE-bench、SWE-agent 和 Tau-bench 都在提醒一件事：Agent Dev 不能只盯模型和框架。任务、工具、用户、状态、规则和验证器怎样被组织成环境，会直接影响 agent 能不能工作。
excerpt_en: "SWE-bench, SWE-agent, and Tau-bench show that agent development cannot focus only on models and frameworks. Tasks, tools, users, state, rules, and validators shape whether agents work."
permalink: '/blog/2026/03/16/aenvironment-everything-as-environment/'
---

这个 Agent 系列写到现在，绕来绕去其实一直在问同一个问题：Agent 到底怎样接触外部世界。

MCP 讨论协议，Skills 讨论能力封装，认知架构讨论决策循环，RL Agent 那几篇讨论训练和反馈。它们看起来像不同主题，但放在一起看，会发现中间少了一层：agent 运行的那个环境本身。

**Agent = Model + Harness**，我们过去将 Agent Dev 分成两块。模型负责推理，Harness 负责 loop、tool calling、memory、workflow。这个分法没错，只是不够。Agent 不是在真空里推理，它总是被放进某个具体环境：仓库、终端、浏览器、客服系统、数据库、订单系统、测试套件、规则文档，甚至一个会不断追问和改口的用户。这些东西决定了 agent 看见什么、能做什么、动作会造成什么副作用、失败后能不能恢复、最终结果怎样被验证。换句话说，Agent Dev 中的和外部环境交互的那个层面好像在之前的研究中隐形了。（p.s. 当然我们可以说环境就在harness里，这里仅仅是为了强调该概念，你可以在我关于harness的blog中看到详细的关于harness的讨论）

[AEnvironment](https://github.com/inclusionAI/AEnvironment) 是一个不错的可以引出这个问题的最近的例子，它是 inclusionAI 做的环境层项目，**核心主张是 Everything as Environment**。至于是否做到 Everything 暂且不谈。我们来看看环境变成工程核心对象以后的位置。

## 从调工具到进入环境

Tool calling 的默认想象很简单：世界被拆成一组函数，模型选择函数、填参数，系统执行，再把结果交回模型。

这在很多业务自动化里够用。但一旦任务变长，这个抽象就显得薄了。浏览器任务不是一次搜索，软件工程任务不是一次代码生成，客服任务也不是一次数据库查询。它们都有状态变化、连续交互、规则约束和轨迹依赖。

这时，问题不只是“给 agent 什么工具”。还要问：

- 工具背后的世界状态怎样变化；
- agent 每一步能观察到多少状态；
- 错误反馈能不能帮助恢复；
- 多步轨迹怎样记录和重放；
- 成功标准怎样验证；
- 训练、评测和线上运行能不能共享同一套边界。

这些问题更接近 environment，而不是单个 tool schema。

借用 RL 的说法，agent 做的是 `observation -> action -> feedback -> next observation`。LLM Agent 把动作空间从控制信号换成了语言、工具调用和代码操作，但只要任务进入真实系统，它仍然绕不开环境循环。

## SWE-bench：软件工程任务首先是环境任务

[SWE-bench](https://github.com/swe-bench/SWE-bench) 值得反复看，不只是因为它让模型修真实 GitHub issue。它更重要的地方在于，把软件工程能力放进了一个可执行、可验证的环境。

一个 SWE-bench 任务里有真实仓库、issue 描述、代码修改、依赖环境、测试套件、patch 生成和最终验证。它不是问模型“这段代码怎么改”，而是让 agent 进入一个软件工程现场：读代码，定位问题，改文件，跑测试，生成补丁，然后接受测试结果的反馈。

于是 benchmark 设计本身也变成了研究对象。任务怎么抽样，仓库怎么准备，依赖怎么固定，测试怎么判定，patch 怎么应用，执行环境怎样隔离，都会改变 benchmark 实际测到的东西。

如果一个 agent 在 SWE-bench 上表现更好，我们不能只问模型是不是更强。还要看它面对的环境接口：有没有稳定的 shell？有没有文件查看器？错误信息是否简洁？测试反馈是否可恢复？上下文里留下的是有用轨迹，还是一堆已经过期的输出？

这也是为什么 SWE-bench 后来经常和 agent harness、terminal sandbox、evaluation runner 一起被讨论。软件工程 agent 的能力，已经不只是代码生成能力，而是在真实工程环境中持续行动并接受验证的能力。

## SWE-agent / ACI：环境不是裸 shell

[SWE-agent](https://arxiv.org/abs/2405.15793) 把这个问题讲得更直白。它提出了 `ACI`，也就是 `Agent-Computer Interface`。

这个说法很像 HCI 的延伸。人类工程师需要 IDE、语法高亮、搜索、调试器和错误提示，agent 也需要适合自己的界面。裸 Linux shell 对人类很强，但对 LM agent 未必友好。

Shell 的动作空间太宽，命令组合太自由，输出经常太长，错误反馈也不一定指向下一步。人类可以靠经验和视觉上下文过滤噪音，模型却要把这些内容塞进有限上下文里继续推理。同一个计算机环境，换一种接口暴露给 agent，难度会完全不同。

SWE-agent 的 ACI 做得很具体。它用 `find_file`、`search_file`、`search_dir` 降低代码搜索和导航难度；用 file viewer 展示带行号、窗口化、可滚动的文件内容；用 file editor 让模型按行替换代码，并在编辑后立即显示结果；再用 lint guardrails 拦截语法错误和坏编辑。

这些东西看都和模型本身无关，但会直接改变模型的行为。好的交互环境层会把动作空间裁窄，把观测变清楚，把错误变成可恢复信号，把状态压缩成下一步决策需要的信息。

这给 Agent Dev 一个很实际的提醒：接口设计不是产品 UI 的尾活。工具、反馈、上下文管理、guardrails、验证器，合在一起才是 agent 实际面对的世界。一套对 Agent 友好的接口有时候比反复修改 ReAct Loop 与 Workflow 更重要。

## Tau-bench：很多普通任务里的不稳定

SWE-bench 和 SWE-agent 把问题放在软件工程环境里。[Tau-bench](https://arxiv.org/abs/2406.12045) 则换到了客服和企业对话场景。

Tau-bench 的全名是 `A Benchmark for Tool-Agent-User Interaction in Real-World Domains`。它不只是测模型会不会调用函数，而是让 agent 在 retail、airline 这类领域里同时面对用户对话、领域 API 工具和业务规则文档。

这类任务单步看起来并不难。改航班、退货、查订单、更新信息，都不是高深推理题。麻烦在别处：用户信息可能分多轮才给全，agent 必须澄清缺失条件；API 调用会改变数据库状态；业务规则会限制哪些动作可以做；最后还要看数据库状态是否真的正确。

Tau-bench 特别强调一致性。即使是较强的 function calling agent，多次 trial 里也会表现不稳。企业 agent 的麻烦常常不是“完全不会做”，而是有时做对、有时漏规则、有时状态改错。这比单次失败更难处理。

所以客服 agent 的环境不能被理解成一组孤立 API。它是用户、工具、策略、数据库状态、对话历史和 verifier 叠在一起的系统。只优化模型或 tool schema 不够，还要设计环境怎样暴露状态、规则怎样进入上下文、工具副作用怎样被约束、最终状态怎样被验证。

## AEnvironment 的问题意识

在这个背景下再看 AEnvironment，它的 `Everything as Environment` 就没那么像口号了。

我不觉得“所有东西都应该被强行统一成一个巨大抽象”是好方向。这种框架很容易变成最小公倍数，谁都能接，谁都不好用。更合理的理解是：AEnvironment 试图把 agent 运行时、工具、沙箱、benchmark、rollout、reward / verifier 放进同一个环境视角里管理。

AEnvironment 不是孤立的项目。它和 AReaL（完全异步的 RL 训练系统）以及 AWorld（Agent 运行时框架）构成了蚂蚁的”训练-环境-运行”三件套。这个生态背景对理解 AEnvironment 的定位至关重要——它不只是一个通用环境抽象，更是蚂蚁 Agentic RL 闭环中的关键拼图。

AEnvironment 作为一个 `Environment-as-Code` 和环境运行平台，和 Agent Runtime 以及 AReaL 一起工作来处理 RL 任务的全流程。开发者定义可复用环境，把工具或 MCP 服务放进容器化沙箱，通过 SDK 创建环境实例、调用工具、获取结果，并面向 RL / Agent 场景提供标准化的 tool 和 reward 服务。

它踩到的痛点很实在：

- benchmark 不应该只是外部排行榜，也可以是可复用环境；
- 训练环境和线上环境割裂，评测成绩就很难迁移；
- tool execution 需要 sandbox、权限、状态和观测格式；
- reward / verifier 需要和环境状态绑定，不能只靠事后人工贴标签；
- agent 轨迹应该能被记录、回放、分析和改进。

它真正回答的是：当 agent 需要在不同任务世界里反复行动、评测和学习时，环境本身能不能成为一等工程对象。

## Benchmark 正在环境化

从 SWE-bench、Terminal-Bench、BrowserGym 到 Tau-bench，benchmark 已经不太像静态题集了，更像一个个交互环境。

[BrowserGym](https://github.com/ServiceNow/BrowserGym) 用类似 Gym 的接口统一浏览器任务，强调 `reset`、`step`、observation、action 和可复现轨迹。[Terminal-Bench](https://github.com/harbor-framework/terminal-bench) 把终端任务、沙箱和评测 runner 绑在一起。SWE-bench 把真实仓库和测试验证变成软件工程环境。Tau-bench 则把用户模拟器、业务规则、API 工具和数据库状态组合成企业对话环境。

这些项目未必会收敛到同一个框架，但它们承认了同一件事：Agent benchmark 的重点不再只是题目，而是环境接口。

题目描述只是入口。难度和可信度要看环境：agent 能观察到什么，能执行什么，执行后状态如何变化，错误是否可恢复，verifier 能不能判断最终状态，轨迹能不能复现。

这也是为什么 benchmark、任务设计和接口设计本身会变成研究内容。环境设计太松，agent 可能绕过真实能力要求；反馈设计太差，模型可能不是能力不足，而是被坏接口拖垮；verifier 不稳，训练出来的可能只是 reward hacking。

## Reward 其实也是环境产物

benchmark 环境化之后，下一步自然就是 reward/verifier ，在 RL Agent 那篇里我们讨论过 The Second Half 的洞察：当模型已经具备较强泛化能力后，瓶颈会转向 Evaluation。再往前一步看，evaluation 的来源就是环境反馈。

Silver 和 Sutton 在 [Welcome to the Era of Experience](https://storage.googleapis.com/deepmind-media/Era-of-Experience%20/The%20Era%20of%20Experience%20Paper.pdf) 中强调，AI 研究会从“从人类数据中学更多”迁移到“让 agent 在世界中行动并从后果中学习”。如果 agent 要从经验中学习，就必须有环境给出后果。

AlphaProof 和 AlphaEvolve 这类系统之所以强，是因为它们有清晰的验证器：证明是否成立，测试是否通过，程序是否改进。奖励信号清楚，经验学习才可能规模化。

开放世界任务麻烦得多。一个客服 agent 是否成功，不只是回答是否礼貌，而是数据库状态是否正确、规则是否遵守、用户目标是否完成、有没有越权操作。一个软件工程 agent 是否成功，也不只是 patch 能不能编译，还要看测试是否覆盖真实 issue、有没有引入隐藏回归。

**越接近现实世界，奖励越稀疏、越滞后、越多目标冲突**，随着 agent 能力增强，reward hacking 和规范博弈的问题会变得更严重——一个足够强的 agent 可能学会看起来在优化你给的奖励，实际上在优化别的东西。

因此 Reward 不是凭空来的。它是环境设计的产物。环境越能把状态、规则、副作用和验证器组织清楚，agent 就越容易被评测、训练和改进。环境本身混乱，奖励就只能退回人工主观判断，或者被模型钻空子。

## NitroGen：统一环境能解锁什么

游戏领域有一个不错的旁证：[NitroGen](https://nitrogen.minedojo.org/)。它把 Dark Souls III、Sekiro、Black Myth: Wukong、Elden Ring 等商业游戏包装成统一的 Gymnasium API，并把大量游戏映射到同一个 gamepad action space。

这个案例有意思的地方在于，它不是为了统一而统一。统一环境接口之后，研究者才能做跨游戏的视觉-动作预训练，才能让模型从公开视频里的 controller overlay 中学习动作先验，也才能在多个游戏任务上比较迁移效果。

NitroGen 也给统一环境层划了边界。它统一的是 gamepad 动作游戏这个子集，不是所有游戏；它学到的是 system-1 式的动作反射，不是完整长程规划。环境统一的价值取决于统一之后能解锁什么上层能力，而不是抽象听起来多完整。

这个判断同样适用于 AEnvironment。它不需要立刻证明自己能统一所有 agent 场景。更实际的目标是，在某些高价值场景里，把环境做成可定义、可部署、可复现、可评测的工程对象，降低 Agent Dev 的成本。

## 统一环境层真的做得到吗？

Agent 基础设施这类抽象层天然会遇到漏洞抽象风险：传统基建是确定性的，输入 A 必然得到 B；但基于大模型的基建本质上是概率性的。当你试图把浏览器、终端、代码沙箱、移动端、benchmark 全部统一到一个 Environment interface 下，面临的不只是接口层面的差异，而是语义层面的差异。

BrowserGym 只做了 web 一个领域的环境统一（WebArena、MiniWoB、WorkArena），已经够复杂了。NitroGen 只统一了 gamepad 游戏这一个子集，也需要构建从视频数据管线到模拟器封装的完整工程栈。AEnvironment 想做全部。历史上，试图统一一切的抽象层往往要么变得过于臃肿，要么在困难场景下退化为最小公倍数的妥协。

在认知架构那篇里，我们讨论过智能体工程设计里的涌现性与鲁棒性。统一环境接口是在增加确定性，这对训练、评测和复现都有帮助，但它也会约束 Agent 与世界交互的自由度。

同样是在那篇里，我们得出过一个判断：新框架应该是极简的、高度自由的底层设施，而不是高层封装。AEnvironment 是否足够极简？当它试图把 benchmark、RL training、agent deployment、多智能体编排全部收进一个 Environment interface 时，它还能保持底层设施的轻量感吗？

这不是在否定它，而是在指出一个真实张力：环境层越统一，对上层 Agent 的约束就越强。如何在统一性和灵活性之间取得平衡，是 AEnvironment 必须持续回答的问题。

从认知架构的视角看，CoALA 把外部环境定义为物理环境、数字环境、与人类的交互、与其他智能体的交互四种形态。这四种形态的交互语义差异巨大。一个统一的 Environment interface 能在不丢失语义的前提下覆盖它们吗？还是说，它最终只能覆盖其中某几种，而把真正困难的场景留给领域专用的方案？

## 结语

从 SWE-bench 到 SWE-agent，再到 Tau-bench，Agent Dev 的难点正在从让模型会想，扩展到让模型能在正确环境里行动。

SWE-bench 把真实软件工程能力放进仓库、终端、测试和 patch 验证组成的环境里看。SWE-agent / ACI 提醒我们，环境不能只是裸 shell，还要有为 agent 设计过的接口。Tau-bench 则把问题放进企业对话里：大量普通任务、规则约束、工具副作用、状态验证，缺一块都会出问题。

这也是我对 AEnvironment 的理解。它未必是最终答案，但它抓住的问题是真实的。未来 Agent Dev 的一部分工程能力，会变成 environment / interface design。未来做 agent，可能会越来越像在设计一个可行动、可验证、可恢复的环境，而不只是接一个模型和一组工具。

## 参考资料

- [AEnvironment GitHub 仓库](https://github.com/inclusionAI/AEnvironment)
- [AEnvironment Architecture](https://inclusionai.github.io/AEnvironment/architecture/architecture.html)
- [AEnvironment Python SDK Guide](https://www.inclusion-ai.org/AEnvironment/guide/sdk.html)
- [AReaL GitHub 仓库](https://github.com/inclusionAI/AReaL)
- [AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning](https://arxiv.org/abs/2505.24298)
- [AWorld GitHub 仓库](https://github.com/inclusionAI/AWorld)
- [AWorld: Orchestrating the Training Recipe for Agentic AI](https://arxiv.org/abs/2508.20404)
- [SWE-bench GitHub 仓库](https://github.com/swe-bench/SWE-bench)
- Jimenez et al., [SWE-bench: Can Language Models Resolve Real-World GitHub Issues?](https://arxiv.org/abs/2310.06770)
- Yang et al., [SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering](https://arxiv.org/abs/2405.15793)
- Yao et al., [Tau-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains](https://arxiv.org/abs/2406.12045)
- Sierra, [Tau-bench: Benchmarking AI Agents for the Real-world](https://sierra.ai/resources/research/tau-bench)
- [Tau-bench GitHub 仓库](https://github.com/sierra-research/tau-bench)
- Shunyu Yao, [The Second Half](https://ysymyth.github.io/The-Second-Half/)
- DeepMind, [AlphaProof: Olympiad-level formal mathematical reasoning with reinforcement learning](https://www.nature.com/articles/s41586-025-09833-y)
- DeepMind, [AlphaEvolve: A coding agent for scientific and algorithmic discovery](https://arxiv.org/abs/2506.13131)
- Sumers et al., [Cognitive Architectures for Language Agents](https://arxiv.org/abs/2309.02427)
- [Reward Hacking：当奖励信号被优化器反向搜索](/blog/2026/03/20/reward-hacking-four-failure-modes/)
- [Terminal-Bench GitHub 仓库](https://github.com/harbor-framework/terminal-bench)
- [BrowserGym GitHub 仓库](https://github.com/ServiceNow/BrowserGym)
- [NitroGen 项目页](https://nitrogen.minedojo.org/)
- [NitroGen 论文（arXiv:2601.02427）](https://arxiv.org/abs/2601.02427)
- [NitroGen GitHub 仓库](https://github.com/MineDojo/NitroGen)
- [MCP Architecture](https://modelcontextprotocol.io/docs/learn/architecture)
- Silver and Sutton, [Welcome to the Era of Experience](https://storage.googleapis.com/deepmind-media/Era-of-Experience%20/The%20Era%20of%20Experience%20Paper.pdf)
