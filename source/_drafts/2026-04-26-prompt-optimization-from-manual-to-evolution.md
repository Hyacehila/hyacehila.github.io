---
title: "从 Prompt Optimizer 到 Harness Optimizer：Agent 时代如何优化给模型看的系统"
date: 2026-04-26 18:00:00 +0800
categories: [智能体系统]
tags: [Prompt Optimization, Harness, Agent Evaluation, MCP]
author: Hyacehila
excerpt: "提示词优化没有消失，而是迁移到 Agent Harness：工具描述、Skills、服务器说明、环境反馈、验证器与工作流结构都成了可优化对象。"
---

优化一个系统很困难，当我们想要去优化一个Agent的时候，需要考虑Benchmark应该如何设计，才能精准的反应事实，需要考虑Interface的设计，适合模型的工具才能让模型更好的工作。需要考虑Harness结构的设计，在Agent自主性以及我们认为施加的约束上实现平衡。需要考虑长期记忆与持续学习，让这个系统进步。 这些事情本身已经很难了，但是想要自动的去优化，是更难的问题。

# 从 Prompt Optimizer 到 Harness Optimizer：Agent 时代如何优化给模型看的系统

## 引言：Prompt 工程没有结束，只是换了形态

如果把视角停留在 2023 年，提示词优化看起来像一个很清楚的问题：我们有一条系统提示词，有一个任务数据集，有一个指标，然后想办法把这条提示词改得更好。

这当然是一个真实问题。早期的 Prompt Engineering 之所以重要，就是因为 LLM 对自然语言指令高度敏感。同一个任务，用不同的格式、不同的约束顺序、不同的 few-shot 示例，结果可能差很多。于是自动提示词优化自然出现：既然人类手工试 prompt 太慢，那就让算法系统性地搜索、评估和改写 prompt。

但到了 Agent 时代，问题已经变形了。

今天真正影响模型行为的，早就不只是一条 system prompt。一个能工作的开发型 Agent 通常还会看见这些东西：

- MCP tool 的名称、描述、参数 schema 和返回格式。
- MCP server instructions 里对跨工具关系、操作模式和限制条件的说明。
- Skills 的 `description`、`SKILL.md`、`references/`、`scripts/` 和触发规则。
- 仓库根部的 `AGENTS.md`、`CLAUDE.md`、docs index、项目约定和风格指南。
- 工作流图里的 planner、generator、verifier、subagent 和 handoff artifact。
- 工具执行后的 stdout、错误信息、测试结果、浏览器状态、数据库终态和日志轨迹。

这些东西都在承担相近作用：**它们是给模型看的系统工件。** 下文把它们统一称为 **agent-facing artifacts**。它们不只是文档，也不只是配置，而是在影响模型怎样理解任务、怎样选择工具、怎样解释环境反馈、怎样判断自己是否完成。

所以我现在更愿意把这个问题改写成一句话：

**Prompt optimization 没有过时，它只是从优化一条 prompt，迁移到了优化整个 harness 中所有会进入模型决策边界的工件。**

这篇文章想讨论的不是“某一种提示词优化算法是不是最强”，而是一个更宽的问题：当我们进入 harness 时代以后，MCP 提供的 tool description、Skills 的描述性文字、工作流结构和环境反馈应该如何被评价？LLM-as-judge 是不是解决这类开放问题的唯一方法？有没有可能围绕真实环境设计一个闭环反馈模块，让系统持续发现、归因、优化自己的提示词和结构？

如果把[上下文工程](/blog/2026/03/06/agent-context-engineering/)、[MCP 与 Skills](/blog/2026/03/10/from-mcp-to-agent-skills/)、[统一环境层](/blog/2026/03/16/aenvironment-everything-as-environment/)和 [Agent Harness](/blog/2026/03/20/building-agent-deterministic-constraints/)看作更早的背景，那么这一篇更像是把它们重新接起来：**当 Agent Harness 变成主要工程对象以后，我们到底该怎么优化它？**

[模型路由](/blog/2026/04/25/llm-semantic-routing-compound-ai-systems/)从“调用哪条路径”看 Agent 系统，[Output Token 与 KV Cache](/blog/2026/04/26/output-token-pricing-kv-cache-agent-cost/)从“每次调用内部的 token 成本”继续往下拆；这一篇则继续往外看：这些成本和行为，最终都被 harness 里的系统工件塑造。

## 自动提示词优化的旧问题

先从传统 prompt optimizer 讲起，因为它仍然是理解后面问题的最小原型。

自动提示词优化最早面对的是一个很朴素的目标：给定任务、模型、训练集和评价指标，找到更好的提示词。围绕这个目标，过去几年大致长出了几条路线。

第一类是**候选生成与搜索**。APE（Automatic Prompt Engineer）把 LLM 当作 prompt proposer，让模型根据少量输入输出样例生成候选指令，再用目标任务指标筛选。OPRO（Optimization by PROmpting）更进一步，把优化过程本身写进 LLM 上下文：历史候选和分数被作为上下文，LLM 继续提出新候选。这里的关键不是梯度，而是把自然语言空间当作一个可以被黑盒搜索的空间。

第二类是**自然语言梯度**。ProTeGi / APO 和后来的 TextGrad 都在尝试把失败样本转化成文字反馈，再用这段反馈指导下一轮 prompt 更新。它们背后的直觉很直接：对于黑盒 LLM，我们拿不到参数梯度，但可以拿到错误案例、执行轨迹和自然语言诊断。自然语言反馈虽然不如数值梯度严格，却比一个标量分数包含更多归因信息。

第三类是**编程化 prompt 和模块化优化**。DSPy 把 prompt 从一段手写文本，变成带 signature、module、teleprompter 和 optimizer 的程序对象。MIPRO / MIPROv2 则在 DSPy 生态里优化指令和 few-shot 示例组合。这个方向的重要性不只在于“自动生成提示词”，而在于它把 prompt 从一次性文本变成了可以编译、度量和复用的程序部件。

第四类是**进化式和反思式优化**。PromptBreeder 用自指令和变异机制演化 prompt；GEPA 则把反思式变异和 Pareto 选择结合起来，让 LLM 根据完整轨迹分析失败原因，再生成更有针对性的候选。GEPA 特别值得放在这里，因为它直接把优化信号从“这个候选得了几分”推进到“这条轨迹为什么失败、哪条规则应该被改写”。需要注意的是，论文标题里的 “Can Outperform Reinforcement Learning” 是在其报告的任务、预算和实验设置中成立的比较，不应读成“提示词优化普遍优于 RL”。这也是它和许多纯搜索式 optimizer 的差别。

把这些方法放在一起看，自动提示词优化其实一直在处理三个问题：

| 问题 | 典型方法 | 本质 |
| --- | --- | --- |
| 怎么产生候选 | APE、OPRO、PromptBreeder | 在自然语言空间里搜索 |
| 怎么知道哪里错了 | ProTeGi、TextGrad、GEPA | 把失败样本变成文字诊断 |
| 怎么选择和保留候选 | DSPy/MIPRO、GEPA | 用指标、验证器或 Pareto 前沿筛选 |

这就是传统 prompt optimizer 给我们的第一个启发：**优化从来不是只改文字，而是“候选生成 + 反馈归因 + 选择机制”的闭环。**

但它也留下了一个更重要的问题：反馈到底从哪里来？

## 优化的主要瓶颈是评估

很多关于 prompt optimization 的讨论容易把注意力放在“怎么改 prompt”上，但真正困难的地方通常是“怎么评价改完以后是不是更好”。

如果任务是数学题、代码题、单元测试、数据库终态或格式校验，评估相对清楚。你可以用 verifier，答案对就是对，测试过就是过，JSON schema 合法就是合法。这类任务非常适合自动优化，因为反馈便宜、稳定、可重复。

问题是，Agent 时代的大量任务并不长这样。

一个开发型 Agent 的任务可能是“修复这个 bug，并且不要破坏现有架构”；一个研究型 Agent 的任务可能是“阅读资料后提出一个有价值的研究假设”；一个企业流程 Agent 的任务可能是“在政策约束下处理用户请求”。这些任务往往没有唯一标准答案，甚至很难提前写出完整判分函数。

这时很多人会自然想到 LLM-as-judge。G-Eval、MT-Bench / Chatbot Arena、Prometheus 和大量后续工作说明，LLM 确实可以在许多主观评价任务里提供有用信号。但这不意味着 LLM-as-judge 是唯一答案，更不意味着它是可靠验证器。

更准确的说法应该是：

**LLM-as-judge 是开放任务中不可完全验证部分的代理信号，而不是整个评估系统本身。**

它至少有几个工程风险：

- Judge 会受 rubric 写法、候选顺序、回答长度、模型家族偏好和上下文呈现方式影响。
- Judge 往往擅长判断“看起来合理”，但不一定擅长发现隐藏状态错误、工具误用和长期副作用。
- Judge 自己也需要 meta-evaluation，需要和人类标注、硬验证器、线上结果做校准。
- Judge 的分数很容易被优化器 exploit，尤其当优化器知道评分风格以后。

所以在真实 harness 里，我更倾向于使用一条分层的 grader stack：

```text
hard verifier > execution/process signal > rubric judge > human calibration
```

这里的 `>` 表示开发评估中的优先使用顺序，不是数学意义上的分数大小，也不是所有任务里的绝对等级。

这条优先级的意思很简单：**能验证的事情，不要交给 judge；只有 verifier 覆盖不到的部分，才交给 judge。**

硬验证器包括测试、类型检查、lint、schema validation、数据库终态、文件 diff、权限边界、policy rule 和 replay 成功率。执行过程信号包括工具调用是否有效、是否重复空转、是否读过关键文件、是否正确处理错误、是否在预算内完成。Rubric judge 适合处理“解释是否清楚”“方案是否合理”“是否符合用户偏好”“是否覆盖关键风险”这类开放维度。Human calibration 则用于抽检 judge、修正 rubric、发现系统性 blind spot。

这也是为什么 tau-bench、PaperBench、SWE-bench、WebArena、OSWorld 这类环境化 benchmark 很重要。它们没有把评估简化成一句“让 GPT-4 打分”，而是在构造可交互环境、可复现任务、轨迹日志、终态检查和结构化 rubric。PaperBench 评估开放式研究复现任务时，不是让 judge 直接给一个总分，而是先把目标拆成层级化 rubric tree；tau-bench 把用户、工具、政策文档和数据库状态放进同一个交互环境；SWE-bench 则用真实 GitHub issue 和测试环境把“修好代码”转化为可执行验证。

这里的关键判断是：**评估不是一个模型调用，而是一套环境工程。**

只要接受这一点，LLM-as-judge 的位置就会变清楚。它不是唯一思路，而是 grader stack 中的一层。它有价值，但必须被 verifier、rubric、校准集和失败审计包围。

## 从 Prompt 到 Harness：新的优化对象出现了

如果说传统 prompt optimizer 优化的是“系统提示词”，那么我这里暂且称为 harness optimizer 的东西，要优化的对象就多得多。它更像一种架构提案，而不是已经统一命名的业界产品类别。这里最容易被低估的是那些看起来像文档的东西。

### MCP tool description 不是注释

MCP tool 的 `name`、`description`、参数说明和返回格式，本质上是模型的动作语言。

传统 API 文档的读者是人类工程师；tool description 的读者是非确定性的 Agent。Anthropic 在 tool design 里强调，给 Agent 写 tool 不是普通 API 设计，因为工具描述会直接影响模型是否调用、何时调用、怎样组织参数、如何解释返回结果。

所以一个 MCP tool description 至少应该被评价这些维度：

- **触发精度**：该用这个工具时，Agent 是否能想起来用。
- **触发召回**：不该用这个工具时，Agent 是否会误触发。
- **参数正确率**：模型是否能根据描述稳定填出合法参数。
- **边界清晰度**：描述是否明确说明工具不能做什么、失败时会返回什么。
- **返回可消费性**：工具输出是否足够结构化，是否避免把无关日志污染上下文。
- **成本意识**：描述是否帮助模型选择低成本路径，而不是默认调用最重工具。

这类评价不能只靠“描述写得优雅吗”。真正要看的，是带着这份描述跑一批任务以后，工具选择、参数错误、无效调用、任务成功率和上下文污染有没有改善。

### Server instructions 是工具集的用户手册

MCP 社区在 2025-11 的 [server instructions 文章](https://blog.modelcontextprotocol.io/posts/2025-11-03-using-server-instructions/) 中强调了这类“给 server 的用户手册”能力，很大程度上就是因为单个 tool description 装不下跨工具关系。

一个 server 可能有十几个工具。每个工具单独描述都没问题，但 Agent 仍然可能不知道：应该先查索引还是先读详情？什么时候用 search，什么时候用 fetch？哪些工具会产生副作用？哪些资源只能读不能写？失败后应该重试还是换路径？

这些信息不应该被硬塞进每个 tool description。更合适的位置是 server instructions：它像一本给模型看的用户手册，描述跨功能依赖、推荐操作模式、系统限制和安全边界。

这也带来一个新的优化问题：**server instructions 不只是写给人看的 README，而是影响整个工具集调用策略的控制面。**

### Skills 的 description 是触发信号

在 Skills 里，`description` 的地位更特殊。它更像模型触发和发现信号，而不是给人看的营销介绍。

一个 Skill 的正文可能很长，包含 `SKILL.md`、`references/`、`scripts/` 和 `assets/`。但 Agent 启动时通常不会把所有正文都读进上下文，而是先看到 name 和 description。也就是说，description 决定了这个 Skill 会不会被触发，什么时候被触发，是否会和别的 Skill 冲突。

这和普通文案完全不同。一个 Skill description 写得太宽，会导致过度触发；写得太窄，会导致该用时不用；写得太像营销介绍，模型看不出具体适用条件；写得太长，又会让 skill index 变得臃肿。

后面的研究线索会继续展开这一点：Skills 已经不是“写一份好说明书”这么简单，而是正在变成可以被评估、搜索和优化的系统工件。

### AGENTS.md 和 docs index 是知识路由

仓库里的 `AGENTS.md`、`CLAUDE.md`、docs index 和项目规则，也不应该只被理解成“给 Agent 的备忘录”。它们在做知识路由。

在开发任务里，模型最大的失败之一不是“不聪明”，而是没在正确时间读正确文档。文档如果藏得太深，Agent 会临场猜；文档如果常驻太多，Agent 会被上下文噪声淹没。真正要优化的是暴露顺序：哪些知识常驻，哪些按需读取，哪些只能通过工具返回，哪些必须被强制检查。

这也是我在 Harness 那篇里说的：语言约束会衰减，但知识路由可以被系统化。

### Workflow graph 也是 prompt 的延伸

最后，workflow graph 本身也会影响模型行为。

如果一个系统只有单 Agent 循环，那么所有事情都压在同一个上下文里：计划、执行、验证、总结、恢复都混在一起。引入 planner / generator / evaluator、subagent、hook、checkpoint 和 handoff artifact 后，系统不只是“多了几个模块”，而是改变了模型在每一步看到什么、负责什么、能调用什么工具。

这意味着 harness optimizer 不能只优化文字。它还要优化结构：什么时候拆分 agent，什么时候重置上下文，什么时候插入 verifier，什么时候把经验写回 memory 或 skill，什么时候让某个工具延迟加载。

所以新的优化对象可以概括成一张表：

| 工件 | 它表面上是什么 | 它实际上控制什么 |
| --- | --- | --- |
| Tool description | 工具说明 | 动作选择与参数生成 |
| Server instructions | 服务端说明 | 跨工具策略与约束边界 |
| Skill description | 技能简介 | 能力路由与触发条件 |
| `SKILL.md` | 工作流说明 | 任务执行策略 |
| `AGENTS.md` / docs index | 项目文档入口 | 知识暴露顺序 |
| Tool output format | 返回值格式 | 后续推理可消费性 |
| Workflow graph | 编排结构 | 上下文隔离与验证时机 |
| Verifier / hook | 外部检查器 | 完成声明是否可信 |

这就是从 Prompt Optimizer 到 Harness Optimizer 的关键迁移：**优化对象从一段文本，扩展成了所有会塑造 Agent 行为的模型可见系统表面。**

## 闭环反馈模块可以长什么样

如果接受上面的判断，那么我们就可以设计一个面向开发型 Agent 的闭环优化模块。这个列表是一个最小研究/评估闭环提案，不是现成业界标准。它不需要一开始就像 RL 训练系统那么重，但至少应该具备七个部分：

```text
artifact registry
  -> task bank
  -> rollout runner
  -> grader stack
  -> transcript analyzer
  -> mutation proposer
  -> selection / rollback
```

### 1. Artifact registry：先把可优化对象版本化

系统需要知道自己到底在优化什么。

Artifact registry 里应该保存所有 agent-facing artifacts 的版本：system prompt、tool descriptions、server instructions、Skills、AGENTS.md、docs index、workflow graph、verifier 配置、工具返回模板。每个 artifact 都要有版本号、适用范围、依赖关系和变更说明。

否则后面即使任务成功率变了，也很难知道是哪个描述、哪个工具输出、哪个 skill 触发策略导致的。

### 2. Task bank：评测集必须覆盖真实失败模式

Task bank 不应该只是一组漂亮 demo。它至少要包含：

- 常见成功路径，用来保证基础能力不退化。
- 历史失败案例，用来验证修复是否有效。
- 长尾复杂任务，用来测试组合工具和多步恢复。
- 负样本任务，用来测试拒绝、边界、权限和“不该调用工具”的场景。
- 合成任务，用来补齐真实日志中稀缺但重要的边界条件。

如果任务集只覆盖正例，optimizer 很容易把描述写得越来越激进，导致误触发工具、过度使用 Skills，甚至绕过边界。

### 3. Rollout runner：同一批任务要可重复运行

Rollout runner 负责在固定模型、固定环境、固定权限、固定预算下运行 Agent，并记录完整轨迹。

轨迹不能只保存最终回答。至少要保存：

- 初始任务与可见上下文。
- 每一步模型看到的 artifact 版本。
- 工具候选列表和实际调用。
- 参数、返回值、错误信息和重试。
- 中间计划、状态更新、handoff artifact。
- 终止原因、失败原因和外部验证结果。

没有轨迹，后面就只能看总分；有了轨迹，才能分析失败到底来自工具描述误导、Skill 没触发、上下文过载、workflow 切分不合理，还是 verifier 缺失。

### 4. Grader stack：先验证，再打分

Grader stack 按前面那条优先级执行：

```text
hard verifier > execution/process signal > rubric judge > human calibration
```

对开发型 Agent 来说，hard verifier 可以是测试、构建、类型检查、lint、patch apply、文件权限、受保护路径、数据库终态。过程信号可以是工具调用次数、无效调用比例、重复搜索次数、是否读到关键文件、是否在失败后正确恢复。Rubric judge 则评价补丁说明、风险分析、方案合理性、是否遗漏关键约束。

关键点是：judge 不是第一层。它只处理 verifier 不能覆盖的维度。

### 5. Transcript analyzer：失败归因比总分更重要

如果一次任务失败，只知道“失败了”没有太大价值。系统需要把失败轨迹归因到更细的类别：

- Tool selection failure：该调用的工具没调用，或调用了错误工具。
- Argument failure：选对工具但参数错。
- Context routing failure：该读的文档、Skill 或 memory 没被加载。
- Instruction conflict：不同规则互相冲突，模型选择了错误优先级。
- Verifier gap：模型说完成了，但没有外部检查拦住。
- Over-triggering：Skill 或工具被过度触发，反而污染上下文。
- Under-specification：描述太模糊，模型只能临场猜。
- Environment mismatch：评测环境和真实运行环境语义不一致。

这一步更接近“自然语言梯度”。传统 prompt optimizer 用失败样本反推 prompt 修改；harness optimizer 则要用失败轨迹反推到底该改 tool description、Skill description、server instructions、workflow graph，还是 verifier。

### 6. Mutation proposer：候选变异不只改文字

Mutation proposer 可以由 LLM 生成，但它生成的候选不应该只是一段新 prompt。它应该能提出多种类型的改动：

- 改写 tool description，使触发条件更明确。
- 拆分或合并工具，减少动作空间歧义。
- 调整参数 schema，让常见错误无法表达。
- 重写 Skill description，改善触发精度和召回。
- 把 `SKILL.md` 中的长说明拆成正文、references 和脚本。
- 增加 server instructions，补充跨工具操作顺序。
- 调整 docs index，把关键规则前置。
- 增加 Stop hook 或 verifier，把完成声明外部化。
- 调整 workflow graph，把生成和验证拆开。

换句话说，这一步更像 GEPA 的反思式变异，但变异对象从 prompt 扩展到了 harness。

### 7. Selection / rollback：用 Pareto 前沿防止单目标过拟合

最后，系统需要选择候选。

这里不能只看单一成功率。真实 harness 至少有多目标：

- 任务成功率。
- 工具调用成本。
- 延迟与 token 成本。
- 参数错误率。
- 误触发率与漏触发率。
- 安全边界违规率。
- 回归集表现。
- 跨模型迁移性。
- 人类可维护性。

这就很适合借鉴 GEPA 的 Pareto 思路。不要急着宣称唯一最优版本，而是保留一组在不同维度上有优势的候选，再根据场景选择。更重要的是，所有 artifact 都必须可回滚。优化器一旦把 description 写得过拟合某个 benchmark，线上失败会很隐蔽。

所以，一个更稳妥的 harness optimizer 不应该只是“自动改 prompt”，而应该像小型实验平台：

- 每次改动都可 diff。
- 每个候选都跑相同任务集。
- 每个结果都能追溯到轨迹。
- 每次上线前都跑回归集。
- 每次线上失败都能回流到 task bank。

这才是闭环。

## 开发型 Agent 的研究给了什么启发

围绕开发型 Agent 和环境化 Agent 的研究，已经给这个问题提供了几条很有价值的线索。

### 评测正在环境化

SWE-bench 的重要性不只是“代码修复 benchmark”。它把真实 GitHub issue、仓库环境和测试验证放在一起，让模型必须进入一个可执行软件工程环境。WebArena 和 OSWorld 则分别把网页操作、桌面操作环境化；tau-bench 把用户、工具、政策和数据库状态放进同一个交互世界；PaperBench 把开放研究复现任务拆成 rubric tree。

这些工作共同说明：**越接近真实 Agent，评测越不像静态问答，越像环境中的轨迹验证。**

这对 harness optimizer 的启发很直接。我们不应该只优化最终回答，而应该优化 Agent 在环境里的完整交互行为：它读了什么、调了什么、改了什么、验证了什么、何时停止。

### 经验可以写回系统

Reflexion、Self-Refine、Voyager、ExpeL 这类工作都在说明一个方向：失败经验可以被写回自然语言记忆、技能库或策略说明中。

Voyager 在 Minecraft 环境里维护 skill library，Reflexion 把失败后的语言反思写回后续尝试，ExpeL 强调从经验中抽取可迁移规则。它们不一定直接解决 MCP / Skills 的工程问题，但提供了一个重要思想：**自然语言不只是输入，也可以是经验沉淀的载体。**

放到 harness 里，这意味着每次失败后不一定都要更新模型参数。很多时候，更实际的做法是更新：

- 工具描述。
- 操作手册。
- Skill 触发条件。
- verifier 检查项。
- docs index。
- bad case library。

这是一种比训练更轻、但更贴近工程迭代的学习方式。

### Workflow 本身也可以被搜索

AFlow、ADAS、Maestro 这类工作则把问题推到另一个层次：不只是 prompt 可以被优化，Agent workflow / graph 本身也可以被搜索、生成或改写。

这对开发型 Agent 特别重要。因为很多失败不是某句话写错了，而是结构错了。比如：

- 单 Agent 同时负责生成和验证，导致完成声明不可信。
- 没有 early stop，导致错误轨迹继续滚动。
- 没有 planner / executor 边界，导致工具协议和策略一起漂。
- 没有 context reset，导致长任务后半段被早期错误污染。
- 没有 verifier hook，导致测试没跑也能宣称完成。

这类问题靠改一句 prompt 往往修不好。它们需要改 harness 结构。

### Skills 和 MCP 正在进入可评估阶段

更直接的证据来自 Skills / MCP 生态本身。

截至 2026-05-05，Skills / MCP 的可评估问题已经出现几条线索：

- Agent Skills 相关预印本开始讨论公开 / 第三方 skill 生态中的冗余、安全和质量问题。
- SkillsBench 和 realistic skill usage benchmark 说明 skill 的帮助不是天然成立的，技能质量、触发方式和任务分布都会影响结果。
- Bilevel skill optimization 进一步说明，skill 的结构和内容可以被一起搜索优化。
- MCP tool description smells 和 server description smells 把问题指向工具描述本身：描述写得不好，会影响模型选择工具、组合工具和理解服务器能力。

这意味着我们不宜再把 Skills 和 MCP 描述当成“开发者写完就完”的静态文档。它们正在变成新的 evaluation target。

## LLM-as-judge 不是终点，而是闭环里的一个节点

回到开头的问题：LLM-as-judge 是解决开放式 Agent 任务评估的唯一思路吗？

我的答案是否定的。

更合理的顺序是：

1. 先问任务能不能环境化。能不能放进浏览器、终端、仓库、数据库、沙箱或模拟用户环境里。
2. 再问目标能不能验证。能不能用测试、状态 diff、policy rule、schema、权限边界或执行结果判定。
3. 再问过程能不能结构化。能不能记录工具调用、关键文件读取、错误恢复、预算使用和中间状态。
4. 再问开放部分能不能 rubric 化。能不能拆成独立要求，而不是让 judge 看完整体印象。
5. 最后才问哪些维度必须交给 LLM-as-judge 或人类校准。

这个顺序很重要。因为一旦先把所有事情都交给 judge，系统就会失去很多更硬的反馈来源。Judge 会变成一个看似万能、实际脆弱的总评器；优化器则会学会迎合 judge，而不一定真正修好任务。

真正有价值的闭环，应该尽量把开放任务转化为可验证环境，把不可验证部分整理成 rubric，把 rubric judge 放在可校准的位置上，再让失败轨迹回流到 artifact mutation。

也就是说：

**LLM-as-judge 不是答案本身。更可靠的答案通常是环境、验证器、rubric、轨迹分析和少量 judge 共同组成的反馈生产线。**

## 我把 Harness Optimizer 理解为下一阶段问题

如果只看表面，Prompt Optimizer 和 Harness Optimizer 像是两个不同问题。

前者优化提示词字符串，后者优化工具描述、Skills、服务器说明、验证器和工作流结构。但它们共享同一个内核：都在试图用反馈闭环改进模型行为。

区别在于，Agent 时代的行为不再只由单条 prompt 决定。模型面对的是一个被 harness 组织起来的世界：工具怎样命名，知识怎样暴露，环境怎样返回观察，验证器在哪里拦截，子任务怎样隔离，经验怎样写回系统。这些结构共同决定了 Agent 的能力边界。

所以未来真正重要的 open question 不是“还能不能自动优化 prompt”。这个问题已经有很多答案。更重要的问题是：

- 如何评价一个 MCP tool description 是否真的让模型更会用工具？
- 如何评价一个 Skill description 的触发精度、召回和误触发成本？
- 如何判断一份 `AGENTS.md` 是提供了关键知识，还是污染了上下文？
- 如何在没有唯一标准答案的开发任务里，把 verifier、process signal、rubric judge 和 human calibration 组合起来？
- 如何联合优化文字、工具接口、工作流图、记忆和验证器？
- 如何防止 harness optimizer 过拟合 benchmark，而在真实环境里退化？

这些问题不会被一个更强的 LLM-as-judge 单独解决。它们需要新的评估工程，也需要新的闭环基础设施。

所以我会把这篇文章最后收束成一个判断：

**Prompt Optimizer 的下一阶段问题，不只是更会改 prompt 的算法，而是如何持续优化模型可见系统表面的 Harness Optimizer。**

在 prompt 工程时代，我们优化一句话；在 harness 时代，我们优化模型进入世界的方式。

这个问题如果放进具体行业，会变得更清楚：Agent 不只是优化提示词或工具说明，而是要嵌入真实生产管线。游戏行业就是一个适合观察这种闭环的场景。

## 参考资料

### Prompt optimization

- Zhou et al., [Large Language Models Are Human-Level Prompt Engineers](https://arxiv.org/abs/2211.01910)
- Yang et al., [Large Language Models as Optimizers](https://arxiv.org/abs/2309.03409)
- Pryzant et al., [Automatic Prompt Optimization with "Gradient Descent" and Beam Search](https://arxiv.org/abs/2305.03495)
- Fernando et al., [PromptBreeder: Self-Referential Self-Improvement Via Prompt Evolution](https://arxiv.org/abs/2309.16797)
- Khattab et al., [DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines](https://arxiv.org/abs/2310.03714)
- Opsahl-Ong et al., [Optimizing Instructions and Demonstrations for Multi-Stage Language Model Programs](https://arxiv.org/abs/2406.11695)
- Yuksekgonul et al., [TextGrad: Automatic "Differentiation" via Text](https://arxiv.org/abs/2406.07496)
- Agrawal et al., [GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning](https://arxiv.org/abs/2507.19457)

### Evaluation and judge

- Liu et al., [G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment](https://arxiv.org/abs/2303.16634)
- Zheng et al., [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685)
- Kim et al., [Prometheus: Inducing Fine-grained Evaluation Capability in Language Models](https://arxiv.org/abs/2310.08491)
- Gu et al., [A Survey on LLM-as-a-Judge](https://arxiv.org/abs/2411.15594)
- Anthropic, [Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)

### Agent environments and feedback

- Shinn et al., [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366)
- Wang et al., [Voyager: An Open-Ended Embodied Agent with Large Language Models](https://arxiv.org/abs/2305.16291)
- Zhao et al., [ExpeL: LLM Agents Are Experiential Learners](https://arxiv.org/abs/2308.10144)
- Jimenez et al., [SWE-bench: Can Language Models Resolve Real-World GitHub Issues?](https://arxiv.org/abs/2310.06770)
- Yao et al., [WebArena: A Realistic Web Environment for Building Autonomous Agents](https://arxiv.org/abs/2307.13854)
- Xie et al., [OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments](https://arxiv.org/abs/2404.07972)
- Yao et al., [tau-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains](https://arxiv.org/abs/2406.12045)
- OpenAI, [PaperBench: Evaluating AI's Ability to Replicate AI Research](https://arxiv.org/abs/2504.01848)
- Zhang et al., [AFlow: Automating Agentic Workflow Generation](https://arxiv.org/abs/2410.10762)
- Hu et al., [Automated Design of Agentic Systems](https://arxiv.org/abs/2408.08435)
- Tian et al., [Maestro: Self-Improving Text-to-SQL Agents Through Multi-Agent Collaboration](https://arxiv.org/abs/2509.04642)

### MCP, Skills and harness

- Anthropic, [Writing effective tools for agents](https://www.anthropic.com/engineering/writing-tools-for-agents)
- OpenAI, [Harness engineering: leveraging Codex in an agent-first world](https://openai.com/index/harness-engineering/)
- Model Context Protocol, [Architecture](https://modelcontextprotocol.io/docs/learn/architecture)
- Model Context Protocol Blog, [Server Instructions: Giving LLMs a user manual for your server](https://blog.modelcontextprotocol.io/posts/2025-11-03-using-server-instructions/)
- Agent Skills, [Overview](https://agentskills.io/)
- Model Context Protocol, [Build with Agent Skills](https://modelcontextprotocol.io/docs/develop/build-with-agent-skills)
- Hou et al., [MCP Tool Description Smells: Smell-Aware Evaluation and Selection of Tools in MCP-Based Agents](https://arxiv.org/abs/2602.14878)
- Hou et al., [From Docs to Descriptions: Smell-Aware Evaluation of MCP Server Descriptions](https://arxiv.org/abs/2602.18914)
- Khurana et al., [Agent Skills: A Data-Driven Analysis of Claude Skills for Extending Large Language Model Functionality](https://arxiv.org/abs/2602.08004)
- Wang et al., [SkillsBench: Benchmarking How Well Agent Skills Work Across Diverse Tasks](https://arxiv.org/abs/2602.12670)
- Wu et al., [How Well Do Agentic Skills Work in the Wild?](https://arxiv.org/abs/2604.04323)
- Lin et al., [Bilevel Optimization of Agent Skills via Monte Carlo Tree Search](https://arxiv.org/abs/2604.15709)
