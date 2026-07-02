---
title: "Spec 不是新范式：Vibe Coding、SDD 与 AI 时代的软件工程转向"
title_en: "Spec Is Not the New Paradigm: Vibe Coding, SDD, and Software Engineering in the AI Era"
date: 2026-04-07 20:00:00 +0800
categories: ["Work & Society", "AI Engineering Workflows"]
tags: ["Software Engineering", "Vibe Coding", "Specification", "Feedback Loops", "ADR", "AI Coding"]
author: Hyacehila
excerpt: AI 时代的软件工程并没有走向 Spec-first，而是在代码生成成本坍塌后转向 feedback-first：先用原型和集成反馈发现真实需求，再把已结晶的约束反向提炼成活文档、契约与 ADR。
excerpt_en: "AI-era software engineering is not moving toward spec-first. As code generation costs collapse, it shifts toward feedback-first through prototypes, integration feedback, and living constraints."
permalink: '/blog/2026/04/07/spec-is-not-the-new-paradigm/'
---

如果今天要讨论 AI 时代的软件工程是否出现了新的主导范式，答案大概率不会是 Spec。

这不是因为 `Specification`、`contract`、`ADR`、`acceptance criteria` 这些东西不重要。恰恰相反，它们在很多高风险系统里仍然重要，甚至比过去更重要。我反对的，是把它们抬升成新的时代中心，仿佛只要把需求和设计写得足够完整，AI 就会像一个听话的编译器一样，把复杂系统稳定地推导出来。

我现在更愿意把 AI 时代的软件工程理解成：在代码生成成本下降之后，工程的中心开始从 `Spec-first` 转向 `feedback-first`。稀缺的不再是把代码写出来，而是尽快建立反馈回路，暴露真实约束，分辨哪些想法只是纸面想象，哪些已经在系统里被证明成立。只有在这个基础上，`Spec` 才有资格重新出现，而且是作为收敛期的结果，而不是探索期的起点。

## 为什么 `Vibe Coding` 会崛起

`Vibe Coding` 的爆红，并不意味着工程纪律突然失效，也不意味着开发者集体变懒。它首先是一种对现实约束的自然适配。

`2025-02-06`，Andrej Karpathy 用那条后来被广泛引用的话定义了它：开发者“完全屈服于氛围，拥抱指数级速度，甚至忘记代码本身的存在”。Simon Willison 在后续文章里把这个概念讲得更清楚：`vibe coding` 不是一切 AI 辅助编程，它更接近于**在不审查代码的前提下，让模型先把东西跑起来**。这个定义击中行业，并不是因为它严谨，而是因为它说出了许多人已经开始直觉感受到的事实：当自然语言到可运行原型的路径变得足够短时，人脑会本能地优先选择先看到东西动起来。

这背后的变化，不是开发者突然不再重视质量，而是早期探索阶段的信息结构变了。过去，写一个原型的成本很高，团队倾向于先在文档里做更多推演、论证和市场调查，再拿着数百页需求交付开发，数个月后才能看到原型；今天，一个想法从一句自然语言到一个可运行 demo，可能只需要几十分钟。在这种条件下，最便宜的认知动作，已经不再是先把一切写清楚，而是先把系统推到可以被反馈的位置。

Anthropic 在 [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) 里也提到了这个问题：从最简单、最能闭环的结构起步，让复杂性跟着业务价值一起增长，而不是先把 agent 架构、工作流和文档系统一次性堆满。Agent 相关技术变革和 coding agent 进化都很快，这个判断反而更重要：今天最值得优先优化的，仍然是反馈密度，而不是流程的形式完备性。

`Vibe Coding` 代表的，不是从此不需要工程 discipline，而是探索期的软件工程已经开始默认把原型和反馈排在文档之前。它只是把这个趋势推到了极端，甚至推到了故意不读代码的程度。你可以不喜欢这种极端形式，但很难否认它的出现有现实基础。

这里需要加一条边界：Vibe Coding 更像探索期的工作方式。如果把它原样带进生产环境，数不清的 bug 会很快回来。它之所以爆火，一部分原因是很多项目根本不会进入生产环境，另一部分原因是 Vibe 之后的 review、测试和人工判断仍在正常工作，只是这些纪律被放到了系统外层。

## 为什么 `Spec` 不能成为新范式

如果说 `Vibe Coding` 代表了反馈优先被推到极端，那么 `Spec-first` 代表的则是另一种极端：试图在真正接触系统复杂性之前，就把系统的关键约束预先写清楚。

这件事在复杂软件里通常并不成立。

Paul Ralph 的 `Sensemaking-Coevolution-Implementation` 理论，早就对这种线性想象提出过系统性质疑。它把软件开发描述为在 `sensemaking`、`coevolution` 与 `implementation` 之间反复振荡，而不是在“分析 -> 设计 -> 编码 -> 测试”这些阶段里顺序推进（SCI理论针对性的反对Waterfall Model 与 生命周期过程理论）。换句话说，**需求理解、设计重写与实现行为本来就是缠在一起生长的。** 设想解决方案的过程本身就会改变对环境和问题的理解，从而导致Sensemaking本身发生变化；当创造结束之后，对象随后会改变环境，从而触发进一步的意义建构、协同演化和实现。 很多需求并不是先于系统存在的完整对象，而是在系统逐渐成形、团队逐渐理解问题域之后才被发现的；现代软件是迭代发展的过程。

项目刚开始时，业务方拥有的往往是目标感，而不是边界感；是愿景，而不是机制。你可以把这种模糊性硬写成一份很厚的 `Spec`，但这通常只会制造一种虚假的确定性错觉：看起来一切都被定义了，实际上许多约束还没有被真实系统碰撞出来。

更关键的是，现代软件这类复杂系统的行为边界并不主要藏在单个模块里，而是藏在耦合关系里。Richard Cook 在 [How Complex Systems Fail](https://how.complexsystems.fail/) 里反复强调，复杂系统的问题很少是孤立、单点、可静态枚举的。许多故障只有在多个局部都看起来合理的情况下，经过集成、交互和运行时条件叠加之后才暴露出来。放到软件工程里，这意味着权限模型、状态竞争、跨服务一致性、缓存失效、速率限制、重试策略、事件顺序这些高风险问题，往往不是靠前置文档想出来的，而是靠集成与运行暴露出来的。

所以我不认为 `Spec` 能成为 AI 时代的新范式。理由很简单：

- **第一，需求会演化。** 不是因为团队不专业，而是因为问题本身要通过做出来才会被理解。
- **第二，系统边界会在集成中显形。** 很多真实约束不存在于文档推演里，而存在于模块碰撞之后。
- **第三，文档维护税会快速膨胀。** 代码生成越来越快，最先过期的往往不是代码，而是那些企图穷尽未来实现细节的说明文本；Spec的多层文档体系本质上就是在膨胀文档维护的成本。

有意思的是，连主打 `Specs` 的 Kiro 文档自己都没有把 `Spec` 说成普适起点。它的 [Specs 文档](https://kiro.dev/docs/specs/) 明确写着：`Specs` 适合复杂功能、代价高的 bug、团队协作和需要结构化规划的任务；而 `Vibe` 适合 quick exploratory coding 和 goals 还不清晰的原型期。连最积极推动 `Spec` 工作流的产品都承认，**`Spec` 是条件性工具，不是普适范式。**

这恰恰说明问题所在：如果一种方法需要不断补充“它适合这些情况，但不适合探索期、模糊期、原型期”，那它就不该被叫作“新范式”。范式首先定义的是时代中心，而不是局部场景下的有用工具。从这个角度看，更接近新范式的是 AI Coding，因为它才是放大软件工程生产力的主要因素。

## 为什么纯 `Vibe` 也一定会撞墙

但如果因此把另一边神化，也同样会出错。

`Vibe Coding` 最吸引人的地方，在于它把从想法到可交互系统的路径缩得极短。可一旦项目跨过原型期，问题就会马上从能不能生成转向谁来验证、谁来维护、谁来承担长期一致性。

Simon Willison 对 `vibe coding` 的一个区分很重要：**不读代码、不理解代码，是它定义的一部分，而不是偶发副作用。** 这意味着它在原理上会把大量复杂性从生成前的设计转移到生成后的验证。探索期可以接受这个转移，因为目标是快速试错；生产期不能长期接受，因为验证能力不是无限的，生产环境验证也有高成本。

DORA 在 [State of AI-assisted Software Development 2025](https://dora.dev/research/2025/dora-report/) 及后续解读里反复强调，AI 更像是一个放大器，而不是魔法棒。它确实能提升个体层面的感知效率，但如果组织没有把小批量变更、验证机制一起补上，局部提速很容易转成后续混乱。DORA 在研究里也观察到一种典型情况：AI 让人感觉更快了，但 throughput 与 instability 并不会自动朝着同一方向改善。

GitClear 的 [2025 AI Code Quality 研究](https://www.gitclear.com/ai_assistant_code_quality_2025_research) 则把进行了量化：它分析了 `2020-2024` 的 `2.11` 亿行变更代码，发现与重构相关的 changed lines 占比从 `2021` 年的 `25%` 跌到 `2024` 年的不到 `10%`，而被归类为 `copy/pasted` 的代码则从 `8.3%` 升到 `12.3%`。这不自动等价于AI 写的代码都是垃圾，但它确实说明了一个危险趋势：**当生成速度远快于架构整理、抽象提炼和重构速度时，系统会开始偏向克隆式扩张，而不是可维护的收敛。**

METR 那篇 [Measuring the Impact of Early-2025 AI on Experienced Open-Source Developer Productivity](https://metr.org/Early_2025_AI_Experienced_OS_Devs_Study-paper.pdf) 则更进一步地泼了冷水。对熟悉大型仓库的资深开源开发者来说，在真实任务环境下允许使用前沿 AI 工具，平均并没有带来想象中的加速，反而观察到了 `19%` 的耗时上升。这个结果当然不能被粗暴解读为AI 没用，但它至少提醒了我们：**一旦任务进入真实上下文、真实仓库、真实验证责任，瓶颈就会迅速从写代码转移到理解、审查、测试、修补与确认。**

纯 `Vibe` 最大的问题不是“代码风格不够优雅”，而是它会系统性低估三种成本：

- 验证成本
- 集成成本
- 长期可读性成本

这也是为什么许多 AI 生成系统在 demo 阶段看起来像火箭，在持续迭代阶段却迅速变成泥潭。不是模型突然失灵了，而是**前期被跳过的问题，最终总要以更贵的形式回来。**

*注：以上参考的资料实际上已经过时了，AI Coding在飞速进化，所以Vibe Coding会变成什么样子无人可知。*

## `SDD`、`Spec` 的真实价值，以及新的工程开发模式会长成什么样

**`Spec` 不是新范式，不等于 `Spec` 没价值。** 如果今天要认真讨论 `SDD` 的价值，就不能只停在“它有没有用”这个层面，而要继续追问：**它到底在软件生命周期的什么位置有用，以及 AI 时代更合理的工程工作流会长成什么样。**

我的判断是，现代 `SDD` 确实有现实价值，而且在许多系统里会越来越重要；但它合理的位置，不是探索期的出发点，而是系统开始收敛之后的结构化沉淀层。

系统一旦开始进入收敛期，许多内容必须被重新写下来，而且要写得比过去更结构化。比如：

- 权限边界
- API 契约
- 数据模型
- 迁移策略
- 失败处理与回滚规则
- 多团队共享的架构决策
- success criteria

这些内容一旦稳定，就不应该继续只活在聊天记录、prompt 历史和某个工程师的脑子里。它们应该被提炼成 `Spec`、测试、schema、lint、policy、ADR，成为后续人类和 agent 都能读取的活护栏。

从这个角度看，现代 `SDD` 合理的位置，不是开发前先把世界写完，而是开发中不断把已经证明有效的约束抽取成机器可用的护栏。它更像一种**收敛机制**，而不是世界观本身。它解决的是系统进入稳定经营之后如何降低熵增，而不是未知问题一开始如何被发现。这和目前 Vibe Coding 的相关演进类似：维护一个 `docs` 文件夹，再用这些软文档约束项目发展。我们没有 Spec，但不是没有约束。

一旦把 `SDD` 的位置放对，新的工程开发模式就会变得更清楚。如果 `Spec-first` 不成立，纯 `Vibe` 又不可能长期成立，更合理的答案就不会是两边二选一，而只能是另一种混合形态。

我现在更愿意把它概括成：**反馈优先的结构化探索。**

这个说法的重点在结构化探索。成熟团队并不是在 `Spec` 和 `Vibe` 之间做二选一，而是在不同阶段切换主导工件。`Spec` 的价值没有消失，它只是从起点文档变成了收敛工件；`Vibe` 的价值也没有消失，它只是从长期纪律退回到了探索手段。各个 Coding Agent 正在向这个方向收敛，无论是最新的 `Autodream` 还是 `rules`。

我理解的合理工作流可能是这样的：

第一阶段，是**探索**。  
此时目标不是写出最终系统，而是尽快把模糊想法推到可以被验证的位置。你可以用 `Vibe Coding`、低保真原型、脚手架式 agent、快速集成测试去撞边界。这个阶段容忍代码混乱，前提是你知道自己在买什么：你买的是认知速度，不是长期可维护性。先构建一个 MVP，再考虑其他约束。

第二阶段，是**结晶**。  
当某些约束开始反复出现，某些失败模式开始稳定复现，某些架构边界开始显出必要性时，就不能再继续只靠 prompt 历史和临场记忆了。你需要把这些东西提炼成测试、contract、checklist、ADR、schema、policy，把经验从会话里抽出来。

第三阶段，是**收敛**。  
到了这里，`Spec` 才成为主角之一，但它此时扮演的角色已经变了。它不再先验地规定整个系统，而是把已经在反馈里站稳的结构稳定下来，变成 Kiro 的 `requirements.md / design.md / tasks.md`。

Anthropic 在 [Harness design for long-running application development](https://www.anthropic.com/engineering/harness-design-long-running-apps) 里把长时 agentic coding 收束成 `planner / generator / evaluator` 三代理结构，这点对我很有启发。有效系统的重点，不是让一个 agent 对着一份厚厚的 spec 从头干到尾，而是把规划、生成、评估拆开，并允许结构化 artifact 在会话之间传递。这已经很接近我理解的结构化探索：探索不是无结构的乱跑，而是在持续生成反馈、持续抽取工件、持续降低不确定性。

**探索先于规约，验证先于文档，规约来自反馈沉淀而不是先验想象。**

## 结论

我现在的结论很明确。

`Spec` 不是新范式。它最多是新工作流里的一种高价值工件，而且主要属于收敛期、治理期和高风险模块，而不属于未知问题的起点。

`Vibe Coding` 也不是终局。它代表的是探索期被极端放大的速度红利，适合用来发现问题、试探边界、压低原型成本，但不能直接外推成长期生产纪律。

变化的重点，不是软件工程终于找到了某种更完整的前置文档法。更准确地说，在代码生成成本越来越低之后，软件工程的中心开始从“如何预先描述系统”，转向“如何更快建立反馈、验证约束，并把已证明成立的知识沉淀成护栏”。

AI 时代的软件工程中心，不再是把想象尽可能早地写成 `Spec`，而是尽可能快地把想象送进反馈回路。

这才是我愿意承认的新方向。

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
