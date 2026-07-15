---
title: "Spec 不是新范式：Vibe Coding、SDD 与 AI 时代的软件工程转向"
title_en: "Spec Is Not the New Paradigm: Vibe Coding, SDD, and Software Engineering in the AI Era"
date: 2026-04-07 20:00:00 +0800
categories: ["Work & Society", "AI Engineering Workflows"]
tags: ["Software Engineering", "Feedback Loops", "AI Coding"]
author: Hyacehila
excerpt: 代码生成成本下降后，软件工程需要更早获得原型和集成反馈，再把经过验证的约束整理成文档、契约、测试与 ADR。Spec 仍然有用，但它更适合系统逐渐收敛之后。
excerpt_en: "AI-era software engineering is not moving toward spec-first. As code generation costs collapse, it shifts toward feedback-first through prototypes, integration feedback, and living constraints."
permalink: '/blog/2026/04/07/spec-is-not-the-new-paradigm/'
---

如果今天要讨论 AI 时代的软件工程是否出现了新的主导范式，答案大概率不会是 Spec。

`Specification`、`contract`、`ADR` 和 `acceptance criteria` 在高风险系统里仍然重要。我反对的是把前置文档视为 AI 时代的软件工程中心，仿佛需求和设计写得足够完整，模型就能像编译器一样稳定推导出复杂系统。

代码生成成本下降后，工程的难点逐渐转向建立反馈、暴露约束和验证想法。原型、集成和运行结果可以帮助团队区分纸面设想与已经成立的系统行为。`Spec` 在这个过程中仍然重要，但更适合记录逐渐稳定的约束，而不是假设探索开始前就能写清整个系统。

## 为什么 `Vibe Coding` 会崛起

`Vibe Coding` 的流行与原型成本下降有关。开发者可以先让模型生成可运行结果，再决定是否继续投入，这种工作方式适合需求尚不清楚的探索阶段。

`2025-02-06`，Andrej Karpathy 用“完全屈服于氛围，拥抱指数级速度，甚至忘记代码本身的存在”描述这种做法。Simon Willison 后来进一步区分了 `vibe coding` 与一般的 AI 辅助编程：前者包含了**不审查代码，先让模型把东西跑起来**。当自然语言到可运行原型的路径足够短时，这种选择很容易发生，因为开发者能更快看到想法是否值得继续。

早期探索阶段的信息来源因此发生了变化。过去制作原型成本较高，团队会先投入更多时间做文档推演、论证和市场调查，数周或数月后才看到可运行结果。现在，一个想法可能在几十分钟内变成 demo。团队可以更早把系统交给用户、测试和集成环境，用实际反馈修正理解。

Anthropic 在 [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) 中建议从简单、可验证的结构起步，再根据业务价值增加复杂性。对变化很快的 agent 和 coding agent 项目来说，团队需要优先保证每次改动都能得到可观察的反馈，而不是先搭建完整的流程和文档体系。

`Vibe Coding` 把原型和反馈放到了探索期的前面，并把这种做法推到了故意不读代码的程度。它解释了为什么开发者会先追求可运行结果，但不能因此省略后续的 review、测试和人工判断。

Vibe Coding 更适合作为探索期的工作方式。生产环境还需要 review、测试、监控和人工判断，否则生成阶段省下的时间会转化为验证和维护成本。一些原型不会继续进入生产，另一些项目则会在生成之后补上这些工程环节。

## 为什么 `Spec` 不能成为新范式

`Spec-first` 的风险在于，团队可能在接触真实系统复杂性之前，就试图把关键约束全部写清楚。

这件事在复杂软件里通常并不成立。

Paul Ralph 的 `Sensemaking-Coevolution-Implementation` 理论对线性开发过程提出了系统性质疑。它把软件开发描述为在 `sensemaking`、`coevolution` 和 `implementation` 之间反复移动，而不是严格按照“分析 -> 设计 -> 编码 -> 测试”顺序推进。团队在设想和实现解决方案时，会同时改变自己对问题与环境的理解；系统投入使用后，又会产生新的约束和需求。许多需求因此是在软件逐渐成形的过程中被发现的。换句话说，**需求理解、设计重写与实现行为本来就是缠在一起生长的。**

项目刚开始时，业务方通常知道希望解决什么问题，却未必了解系统边界和具体机制。厚重的 `Spec` 可以记录当前假设，但不能让尚未出现的约束提前变得确定。许多边界只有在原型、集成和实际使用中才会暴露。

现代软件的行为边界经常藏在模块之间的耦合关系中。Richard Cook 在 [How Complex Systems Fail](https://how.complexsystems.fail/) 中指出，复杂系统故障很少由单个、孤立原因造成。权限模型、状态竞争、跨服务一致性、缓存失效、速率限制、重试策略和事件顺序等问题，通常要在集成和运行条件叠加后才会暴露。前置文档可以记录已知风险，却难以提前枚举全部交互。

我主要基于以下观察，不把 `Spec` 视为 AI 时代的软件工程中心：

- 需求会随着团队对问题的理解而变化。
- 系统边界和高风险约束经常在集成后才显现。
- 代码变化速度超过文档维护速度时，试图描述全部未来实现细节的文本会迅速过期。

Kiro 的 [Specs 文档](https://kiro.dev/docs/specs/) 也区分了不同场景：`Specs` 适合复杂功能、代价高的 bug、团队协作和需要结构化规划的任务；`Vibe` 更适合 quick exploratory coding 和目标尚不清晰的原型期。这说明 `Spec` 的价值取决于任务阶段和风险，而不是所有项目都采用同一个起点。

一种方法如果主要适用于复杂、高风险或已经明确的任务，就更像工作流中的重要工具，而不是所有软件开发的统一范式。AI Coding 对生产方式的影响更广，但不同阶段仍需要不同的约束和工件。

## 为什么纯 `Vibe` 也一定会撞墙

反馈优先也有明确边界，纯 `Vibe` 无法长期承担生产系统的验证和维护责任。

`Vibe Coding` 缩短了从想法到可交互系统的路径。项目跨过原型期后，主要成本会转向验证、维护和长期一致性，这些工作不会因为生成速度提高而消失。

Simon Willison 对 `vibe coding` 的定义包含不读和不理解生成代码。这种做法会把复杂性从生成前的设计转移到生成后的验证。探索期可以接受一部分转移，因为目标是快速试错；进入生产后，验证能力和生产环境的试错成本会形成限制。

DORA 在 [State of AI-assisted Software Development 2025](https://dora.dev/research/2025/dora-report/) 及后续解读中指出，AI 的效果会受到组织原有能力影响。如果团队没有同时改善小批量变更和验证机制，个体感受到的提速未必会转化为更高的 throughput，也不保证 instability 同步下降。

GitClear 的 [2025 AI Code Quality 研究](https://www.gitclear.com/ai_assistant_code_quality_2025_research) 提供了一组量化观察：它分析了 `2020-2024` 年的 `2.11` 亿行代码变更，发现与重构相关的 changed lines 占比从 `2021` 年的 `25%` 降到 `2024` 年的不到 `10%`，被归类为 `copy/pasted` 的代码则从 `8.3%` 升到 `12.3%`。这些数据不能直接证明 AI 生成代码质量较差，但提示团队需要关注生成速度是否超过了抽象、重构和维护能力。

METR 的 [Measuring the Impact of Early-2025 AI on Experienced Open-Source Developer Productivity](https://metr.org/Early_2025_AI_Experienced_OS_Devs_Study-paper.pdf) 研究了熟悉大型仓库的资深开源开发者。在其任务和实验条件下，允许使用前沿 AI 工具后，平均耗时上升了 `19%`。这个结果不能推广为“AI 没用”，但说明在大型仓库中，理解上下文、审查、测试和修补可能比生成代码更耗时。

纯 `Vibe` 容易低估生成之后的成本：

- 验证成本
- 集成成本
- 长期可读性成本

如果验证、集成和可读性问题在 demo 阶段被跳过，它们会在持续迭代时集中出现。模型仍然能够生成代码，但团队要花更多时间理解和修复已有实现。

*注：上述研究主要反映 `2024` 年至 `2025` 年初的工具与工作流。模型和 coding agent 仍在变化，后续结论需要结合新的实证研究更新。*

## `SDD` 和 `Spec` 在什么阶段有用

讨论 `SDD` 时，更有用的问题是它适合软件生命周期中的哪个阶段，以及它需要与哪些反馈机制配合。我的判断是，现代 `SDD` 适合系统开始收敛之后。此时团队已经发现了一部分稳定约束，需要把它们整理成可执行、可维护的工件。

系统一旦开始进入收敛期，许多内容必须被重新写下来，而且要写得比过去更结构化。比如：

- 权限边界
- API 契约
- 数据模型
- 迁移策略
- 失败处理与回滚规则
- 多团队共享的架构决策
- success criteria

这些内容一旦稳定，就不应该继续只活在聊天记录、prompt 历史和某个工程师的脑子里。它们应该被提炼成 `Spec`、测试、schema、lint、policy、ADR，成为后续人类和 agent 都能读取的活护栏。

现代 `SDD` 可以在开发过程中把已经证明有效的约束整理成机器可读取的护栏。测试、schema、policy、ADR 和文档共同限制后续修改，减少团队反复讨论同一个问题。它负责稳定已经形成的知识，不负责在探索开始前预测全部需求。维护 `docs` 文件夹并让 agent 读取这些文档，也是这种做法的一种轻量形式。

工程工作流可以在原型反馈和结构化工件之间切换。探索阶段允许快速生成，约束逐渐稳定后再提高文档、测试和契约的权重。我把这种工作方式称为“反馈优先的结构化探索”：先通过可运行结果减少不确定性，再把稳定知识整理成下一轮开发可以复用的约束。

成熟团队会根据阶段切换主导工件。探索时，原型和集成结果提供信息；进入收敛期后，`Spec`、测试和 ADR 负责固定约束。许多 coding agent 也开始读取 `docs`、rules 和其他项目级文件，把会话之外的知识带入生成过程。

下面是一种简化的工作流。三个阶段可能重叠，也可能在发现新问题后退回前一步：

第一阶段，是**探索**。  
此时目标是尽快把模糊想法推到可以验证的位置。可以使用 `Vibe Coding`、低保真原型、脚手架式 agent 和快速集成测试探索边界。这个阶段可以容忍部分临时代码，但要记录哪些实现不能直接进入生产，并尽早构建可测试的 MVP。

第二阶段，是**结晶**。  
当某些约束开始反复出现，某些失败模式开始稳定复现，某些架构边界开始显出必要性时，就不能再继续只靠 prompt 历史和临场记忆了。你需要把这些东西提炼成测试、contract、checklist、ADR、schema、policy，把经验从会话里抽出来。

第三阶段，是**收敛**。  
到了这里，`Spec` 才成为主角之一，但它此时扮演的角色已经变了。它不再先验地规定整个系统，而是把已经在反馈里站稳的结构稳定下来，变成 Kiro 的 `requirements.md / design.md / tasks.md`。

Anthropic 在 [Harness design for long-running application development](https://www.anthropic.com/engineering/harness-design-long-running-apps) 中使用 `planner / generator / evaluator` 结构处理长时 agentic coding。规划、生成和评估由不同环节承担，结构化 artifact 可以在会话之间传递。这个设计与结构化探索相似：系统持续生成反馈，并把可复用的信息提取为后续任务的输入。

在这种工作流中，探索和规约会交替进行。每当某项约束被反复验证，就应把它写入测试、契约或文档，而不是继续只保留在聊天记录中。

## 结论

`Spec` 是高价值工件，尤其适合收敛期、治理期和高风险模块，但它无法替团队预先发现所有未知问题。`Vibe Coding` 能降低原型成本，适合用来试探边界，也不能直接成为长期生产纪律。

代码生成越来越便宜后，团队需要更快建立反馈、验证约束，并把已经成立的知识整理成测试、契约和文档。具体流程会因项目风险和阶段而变化，但原型反馈与结构化工件都不可缺少。
AI 时代的软件工程中心，不再是把想象尽可能早地写成 `Spec`，而是尽可能快地把想象送进反馈回路。
> 2026年4月19日最后一次修改，观点仅供参考。

## 参考资料

- Andrej Karpathy quoted by Simon Willison, [A quote from Andrej Karpathy](https://simonwillison.net/2025/Feb/6/andrej-karpathy/)
- Simon Willison, [Not all AI-assisted programming is vibe coding (but vibe coding rocks)](https://simonwillison.net/2025/Mar/19/vibe-coding/)
- OpenAI, [Harness engineering: leveraging Codex in an agent-first world](https://openai.com/index/harness-engineering/)
- Anthropic, [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)
- Anthropic, [Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)
- Anthropic, [Harness design for long-running application development](https://www.anthropic.com/engineering/harness-design-long-running-apps)
- Kiro Docs, [Specs](https://kiro.dev/docs/specs/)
- DORA, [State of AI-assisted Software Development 2025](https://dora.dev/research/2025/dora-report/)
- DORA, [Balancing AI tensions: Moving from AI adoption to effective SDLC use](https://dora.dev/insights/balancing-ai-tensions/)
- GitClear, [AI Copilot Code Quality: 2025 Look Back at 12 Months of Data](https://www.gitclear.com/ai_assistant_code_quality_2025_research)
- METR, [Measuring the Impact of Early-2025 AI on Experienced Open-Source Developer Productivity](https://metr.org/Early_2025_AI_Experienced_OS_Devs_Study-paper.pdf)
- Paul Ralph, [The Sensemaking-Coevolution-Implementation Theory of Software Design](https://arxiv.org/abs/1302.4061)
- Richard I. Cook, [How Complex Systems Fail](https://how.complexsystems.fail/)
