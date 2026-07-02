---
title: "Harness 到底是什么：从 model + harness 到工程、产品与用户友好外壳"
title_en: "What Is Harness? From Model Plus Harness to Engineering, Product, and User-Friendly Shells"
date: 2026-04-04 23:20:00 +0800
categories: ["AI & Agents", "Agent Architecture"]
tags: ["Agent Harness", "Product Design", "Evaluation", "Reliability Engineering"]
author: Hyacehila
excerpt: LangChain 关于 agent = model + harness 的说法只在粗粒度上成立；真实工程里更有解释力的是工程、产品、用户友好外壳与 task interface。
excerpt_en: "LangChain's agent equals model plus harness framing is useful only at a coarse level. In real engineering, harness is better understood through engineering, product, and task interfaces."
permalink: '/blog/2026/04/04/understanding-agent-harness/'
---

上一篇 [《把 LLM 关回笼子里：从 Claude Code 看 Harness 如何把概率 Agent 固化为系统约束》](/blog/2026/03/20/building-agent-deterministic-constraints/) 讨论的是工程事实：为什么 `MCP`、`skills`、`hooks`、`subagents` 这些外壳，会比继续往 `CLAUDE.md` 里堆规则更重要。

这一篇不再重讲那些机制，而只回答一个词的问题：**为什么今天大家开始把这整圈东西叫作 `harness`，这个词到底该怎么用。**

围绕 `harness` 的讨论里最容易同时犯两个方向相反的错误。

第一个错误，是把 Agent 讨论得过于中间，只盯 `planning`、`memory`、`tool use` 和 `reflection`，仿佛 loop 一旦聪明，系统就会自然成立。

第二个错误，是把 `harness` 写得过于宽泛，最后连产品交互、用户信任、协作节奏、任务交付方式、界面默认值都一股脑塞进去，导致这个词虽然听起来包罗万象，却越来越不提供分析力。

正因为如此，对于 `harness`，我的态度是：可以接受它，但不神化它；可以使用它，但在不少场合里要继续拆细。

## 为什么 `harness` 会在现在被高频讨论

先说结论：新的不是外围系统很重要这件事，而是外围系统终于不能再被假装成外围。在 `harness` 概念出来之前，我们也在研究它，只是它还不是讨论重心。

模型已经强到足以让我们把它们放进真实任务里了，而一旦真的这么做，所有原本还能被 Demo 掩盖的问题都会暴露出来。模型会不会漂、工具会不会误用、文档会不会过期、权限会不会越界、验证闭环是不是缺席、坏模式会不会在系统里持续复制——这些都不再是以后再说的问题。

所谓外围代码，根本不是外围。它才是把 LLM 变成可交付 Agent 的主体工程。

如果只从构建可靠 Agent 的技术角度看，决定上限的往往不是中间那颗聪明的大脑，而是周围这一圈是否足够坚硬、足够可验证、足够能治理的外壳。成熟的 Agent，不是更自由的 LLM，而是被精心约束后仍保有足够自由度的 LLM。

这背后至少有五个变化同时发生。

第一，模型能力跨过了够用阈值。
第二，任务开始进入终端、浏览器、仓库、数据库、工单系统和长程任务。
第三，工具副作用变得真实，错误不再只是答错了，而是会改坏文件、污染环境、触发外部系统。
第四，长任务和跨上下文接力开始普遍出现。
第五，产品化规模上来了，团队开始持续面对文档暴露、沙箱、权限、回放、质量治理和 AI slop 这些维护对象。

如果一定要给这个变化找时间锚点，至少有几篇文章值得作为参照：

- `2026-01-09`，Anthropic 在 [Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) 里明确区分了 `agent harness` 和 `evaluation harness`。我们需要外围组件给 `agent loop` 提供足够的校验信息来提升最终输出，这正是因为 `agent` 开始广泛进入真实系统。
- `2026-02-04`，OpenAI 在 [Unlocking the Codex harness](https://openai.com/index/unlocking-the-codex-harness/) 里把 `Codex harness` 写成跨 web、CLI、IDE extension 和 desktop 共享的内核边界，概念开始逐步分离。
- `2026-02-11`，OpenAI 在 [Harness engineering](https://openai.com/index/harness-engineering/) 里直接把外壳工程抬到主叙事位置，模型能力不再是系统能力的唯一瓶颈。
- `2026-03-10`，LangChain 在 [The Anatomy of an Agent Harness](https://www.langchain.com/blog/the-anatomy-of-an-agent-harness) 里把 `Agent = Model + Harness` 明确写成公式，`harness` 概念开始被传播和解读，成为了 `prompt engineering`,`context engineering` 后的新概念。
- `2026-03-11`，OpenAI 在 [From model to agent: Equipping the Responses API with a computer environment](https://openai.com/index/equip-responses-api-computer-environment/) 里通过进一步改善Response API 开始将 shell、container、compaction、state persistence 这类环境操作与Agent能力平台化，Response API意味着 API 本身将从简单的LLM Chat而直接变成Harness Service。

这些时间点连起来看，就会发现：`harness` 的高频出现，本身就是工程重心转移的信号。模型不再只是被接进产品，而是被放进更长、更开放、更高副作用的工作系统里。于是大家开始需要一个词，来统称模型之外那一圈让系统工作起来的外壳。

## 为什么它不是新东西

但如果因此就说 `harness` 发明了一个全新的问题，也不对。

最直接的证据，是 `harness` 在 LLM 领域本来就有旧用法。EleutherAI 的 [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) 很早就在用这个词指代批量跑任务、打分和汇总结果的评测基础设施。在评测世界里，`harness` 并不新。

同样，[BrowserGym](https://github.com/ServiceNow/BrowserGym) 和 [Terminal-Bench](https://github.com/harbor-framework/terminal-bench) 这类项目，也早就在把浏览器、终端、基准任务和可重复运行环境抽成更稳定的边界。哪怕它们不一定每次都高喊 `harness`，它们处理的仍然是同一类问题：环境怎么被统一、动作怎么进入世界、结果怎么被检查、轨迹怎么被重放。

Anthropic 更是一个很好的反例。`2024-12-19` 的 [Building effective agents](https://www.anthropic.com/research/building-effective-agents/) 当时并没有把 `harness` 写成中心词，但里面已经讲清楚了许多后来被打包进这个总称的方法论：

- 什么时候该用 prescriptive workflow
- 什么时候才需要更开放的 agent
- 为什么应从最简单的结构起步
- 为什么工具设计和反馈结构比厚重框架更重要

准确地说，大多数有经验的开发者第一次看到这个词时，第一反应往往是：这又是什么新东西？于是打开 blog 看了一圈。半小时后，他大概率会发现，这不过是把自己以前做过的很多工作重新起了个名字，包成一个新概念，再拿出来做 marketing——AI 圈在造词这件事上，确实很有天赋。

综上所述：

**Harness 不是发明了新问题，而是把 runtime、workflow、tooling、sandbox、eval、ops 这些旧问题重新绑成了一个更显眼的讨论对象。**

新的不是这些问题第一次出现。新的，是它们第一次一起变成 Agent 产品和 Agent 工程的主舞台。

*注：本小节使用的harness概念可以使用`agent = model + harness`理解，因此我们将key loop也称为了harness的一部分，而非仅有环境，后面会进一步回应该问题，这样使用仅仅是为了叙事更为流畅，在开篇引入大量的定义辨析可读性较差。*

## `agent = model + harness` 为什么有用，但不够

LangChain 的 [The Anatomy of an Agent Harness](https://www.langchain.com/blog/the-anatomy-of-an-agent-harness) 最有价值的地方是他直接把 `Agent = Model + Harness` 写成了中心公式。它足够简单，能让读者很快理解什么是 `harness`。这篇文章也确实有不少声量，其中关于 Building harness 以及 model 和 harness 耦合的讨论，都清晰且有指导性。

这条式子以及这篇文章有用且值得一读，我完全同意。

因为它至少纠正了一个常见误解：Agent 从来不是裸模型自己会工作。只要把这条公式摆出来，读者的注意力就会从模型是不是再强一点就够了回到模型之外的系统层。

但这仍然不够。假如你造出了一个看起来不错的 agent，最后效果不好，有人只丢下一句“是 harness 不行”，那这往往是一句正确却没什么操作性的废话。因为对绝大多数团队来说，能有针对性增强 model 的空间本来就极小；如果再把 harness 拆得过宽，它很快就会变成一个几乎什么都能往里装的总括性概念。

当问题一进入真实工程场景，harness 立刻就不再是一个单层对象了。它内部混杂着执行约束、恢复与治理、人类信任锚点、交互默认值、任务交付方式、审批节奏、界面摩擦、任务界面设计等不同层面的内容。毕竟，如果把问题简单划成 model 和 harness 两半，那么除了做 training 的部分，剩下几乎一切都会被归进 harness；而在现实里，不做 training 的团队显然才是绝大多数。这样一来，若继续只用一个 harness 去统摄这些东西，它很快就会因为过于宽泛而失去解释力。

所以我现在更愿意把这组文章的工作式公式写成两条并列的式子：

`agent = model + harness + task interface`

`harness = 工程 harness + 产品 harness + 用户友好 harness`

第一条式子在提醒我们：不少 agent 的竞争力，不只是模型和外壳，还来自任务界面本身是怎么被设计的。是聊天框、表单、固定工件、状态机，还是某种更强结构化的工作台，会直接决定系统的稳定性。将`task interface`变为一等公民，是希望回应Agent Dev中一个经常被忽视的问题————交互方式，只有Chat肯定是不足的，agent 系统应该收敛出一些更加具备场景针对性的`task interface`，比如 notebooklm 或者其他的专用agent。chat 是暂时的，但绝对不是终点。通用走向专用，也是人类社会本身进化的趋势。

第二条式子则是为了防止我们把 `harness` 写成一个无差别的大袋子。只要一进入具体问题，这种继续拆分就会比停在 `agent = model + harness` 更有解释力。这个拆分方式肯定是不正确的，但对于我们后面的理解会有帮助。工程`harness`侧重于系统本身的稳定运转，产品`harness`控制核心loop或者workflow，至于用户友好 `harness`, 则还是对`task interface`的解释，从让chat更好用到离开chat形成新的交互。

## 为什么拆成三种 harness

**在讨论具体系统时，`工程 harness + 产品 harness + 用户友好 harness` 的拆分，比 `agent = model + harness` 更重要。**

原因很简单：后者负责纠偏，前者负责分析。

如果只停在 `agent = model + harness`，你很容易把完全不同层的问题混在一起。Claude 改坏仓库、用户不信任自动执行、界面默认值过强、审批点太后置、结果无法解释，这几件事都能被叫作 “harness problem”，但它们显然不是同一种问题。

所以新的拆分方式如下表所示：

| 类型 | 它主要回答什么 |
| --- | --- |
| `工程 harness` | 约束、恢复、验证、治理、权限、隔离、可观测 |
| `产品 harness` | 信任点、审批点、出处暴露、任务交付方式、用户为什么愿意委托、以及key loop本身 |
| `用户友好 harness` | 默认界面、控制权收回时机、低摩擦修正、是否过度依赖自然语言 |

这三层的区别很具体。

如果 Claude Code 里你在讨论 `MCP`、`hooks`、`sandbox`、`subagents`、回放、验证器，那你主要在讨论 `工程 harness`。

如果你在讨论用户为什么敢把任务交给系统、系统在哪里暴露出处和中间工件、什么时候允许自动执行、什么时候必须要求审批，这个`key loop`是什么设计的，怎么解决用户需要的问题的。那你已经进入 `产品 harness`。

如果你在讨论默认界面到底该是聊天、表单、状态面板还是更结构化的 task board，用户是否可以低成本修正 agent 行为，系统什么时候该自动继续、什么时候该把控制权交回人，那你讨论的就是 `用户友好 harness`。

也正因为如此，我现在不再满足于Agent = model + harness。这句话有用，但它太粗。它帮你把注意力抬回系统层，却没有告诉你系统层内部该怎么继续做下去。

## `task interface` 为什么必须被单独提出来

原因是：很多专用 agent 的竞争力，根本不只是来自更强的 harness，而是来自 harness 和任务界面被一起设计了。

如果一个系统的入口是完全开放的自然语言，任务状态是隐含的，中间工件不固定，成功标准也不稳定，那它天然就会更依赖更厚的运行时和更频繁的人类澄清。

反过来，如果一个系统的入口本来就是表单、schema、固定工件、明确成功标准、受控工具集和状态机，那它的很多可靠性其实在 `task interface` 这一层就已经被提前硬化了。

所以有了那第一条的公式：

`agent = model + harness + task interface`

不是因为 interface 比 harness 更大，而是因为很多看起来像 harness 的优势，其实一半来自任务界面已经被结构化了，从这个结构化任务自然引出的产品harness，就已经解决了问题，而不是自然语言+更多更笨重的产品harness。

所以并不是所有高价值 agent 最终都会长成更像聊天助手的东西（可能在AGI实现那一天是这样的）。但是现在，很多高价值专用 agent 恰恰会越来越不像聊天，而更像一个被约束得很硬的工作台。

## `harness` 与相邻术语的边界

如果 `harness` 真的要保留，它就必须先把边界说清楚。

否则它太容易开始偷走信息量。你本来该说得更具体，却用一个更模糊的总称代替了细节。除了前面的拆分，我们现在来梳理了agent这么久以来的其他术语以及他们与`harness`的关系。

| 术语 | 更准确的职责 | 不该被混写成什么 |
| --- | --- | --- |
| `agent engineering harness` | 把模型、接口、约束、恢复、治理绑成可工作的 agent 系统 | 不该自动等于全部产品层和用户层 |
| `agent evaluation harness` | 运行任务、记录轨迹、打分、汇总结果的评测基础设施，在可信验证变成Agent系统重要组件的现在。`evaluation harness`是产品的一部分 | 不该被当成 agent runtime 全部，他是产品层的一部分 |
| `runtime` | 运行系统本身 | 不该自动吞掉产品交付和界面问题，`task interface`本身也是重要的 |
| `environment` | 动作作用的世界边界，以及观测、结果和副作用的回流 | 不该只被写成工具集合 |
| `framework` | 编排抽象、开发接口、状态图、组件装配、上一篇以及之前的专门文章讨论过这个问题 | 不该直接等于全部 harness |
| `task interface` | 任务入口、工件格式、状态呈现、用户如何指定和修正任务 | 不该被隐身进Chat这个默认前提里 |

这张表最重要的用法，不是废掉 `harness`，而是提醒我自己：**只有在它真的帮我把问题抬到正确层级时，我才该用它；如果它开始遮蔽真正的层次，我就该换回更具体的词。**

## `harness` 什么时候有解释力，什么时候会偷走信息量

我并不想彻底放弃这个词，因为它确实在某些场景下有解释力。

它最有用的时候，通常有三种情况。

第一，当你想讨论模型如何被放进系统时。
只说 `model` 不够，只说 `framework` 也不够。这时 `harness` 可以把工具、文档、环境、验证、治理这些外围系统一起说清楚。

第二，当你想讨论同一套 agent 内核怎样跨多个产品表面复用时。
这正是 OpenAI 那篇 `Unlocking the Codex harness` 最有价值的地方：`harness` 在这里不是一个空泛形容词，而是一个稳定的软件边界。这篇blog其实就一定程度上体现了我对 `task interface` 的介绍，但是他侧重于多平台单一`task interface`，而我强调的`task interface`更多元。

第三，当你想把工程从零散 feature 提升为整体视角时。
这也是上一篇为什么先讲 Claude Code 的原因。单看 `MCP`、`hooks`、`skills`、`subagents`，它们像一堆零散功能；一旦放回 `harness` 语境，你才能看见它们其实是在共同围住模型。

但它也很容易失效。

一旦你本来该说 `evaluation harness`、`runtime`、`environment`、`task interface`等等概念的时候，却懒得继续拆，直接一律写成 `harness`，这个词就会开始偷走信息量。它看起来像在解释问题，实际上是在把问题抹平。


## 通用 harness 与专用 harness

这一点是我现在最想继续往前推的地方，从本章开始的内容更多的是进一步的畅想，一些简单的 brainstorms，作为这篇全是观点的文章的结尾是个不错的选择。

很多讨论默认把所有 agent 放在同一条连续线上：模型越来越强，harness 越来越全，最后就会出现一个通用 agent，能一路吞掉所有专用 agent。

我不太这么看。

我更愿意把通用 harness 和专用 harness 当成两种不同的优化方向。

通用 harness 追求的是跨任务复用。它通常会有更宽的动作空间、更开放的任务表达、更高比例的自然语言接口、更复杂的知识装配问题，也更频繁面对中途澄清和人类接管。

专用 harness 追求的则是高反馈密度、窄动作空间和强验证。它往往更依赖固定工件、结构化输入输出、受控工具集和更硬的成功标准。

**专用 agent 的可靠性优势，通常不是来自它更像人，而是来自它更少依赖语言这个单一接口。**

语言当然还在，但它不再是唯一 interface。表单、schema、工件、状态机、审批位点、验证器、受控工具集，这些东西会一起构成专用 agent 的真实任务界面。

也就是说：

- 通用 agent 不会消失。
- 专用 agent 的优势通常来自更硬的任务界面、更窄的动作空间和更强的验证器。
- 很多专用系统真正竞争的，不只是谁的 harness 更强，而是谁把 harness 和 task interface 设计得更贴合任务本身。

## `harness` 会在哪些地方收敛，哪些地方不会

这也是我现在最关心的另一个问题。

我不相信会出现一种真正意义上的 `perfect harness technology`，像通用数据库协议那样把所有 agent 产品都统一掉。

更合理的判断应该是：**`harness` 会在原语层收敛，不会在完整系统层收敛成一个单一模板。** 也就是我在关于framework的blog以及上一篇blog中提到的，`harness`的收敛进入新的framework，而不收敛的部分则交给后续的开发。

更可能收敛的东西包括：

- 工具接口原语
- 结构化状态与输出
- 沙箱与权限边界
- 文档索引与知识暴露入口
- 轨迹记录、回放与评测基础设施
- browser / terminal / container 这类环境接口

很难完全收敛的东西包括：

- 不同任务域的成功标准
- 人机协作节奏
- 用户控制权分配
- 产品表面与交互设计
- 长任务交接策略
- 专用 agent 的界面设计

这也是为什么我愿意同时保留两种看法：

- 一方面，底层原语一定会越来越标准化。
- 另一方面，完整 harness 仍然要和任务、产品、风险面一起被重新设计。
- 但当AGI甚至ASI到来的时候，这里的一切就都不重要了。

*本节所谓的harness收敛还是在使用Agent = model + harness的概念进行解读，而不是我前面提出了补充概念解释。*

## Tool / ACI 是 harness 的最小接口单元

如果说前面那些收敛中的原语里，有哪一项最值得单独拎出来，那大概就是 `tool`。但更准确地说，真正值得单独讨论的不是 tool 这个对象本身，而是 **harness 暴露给 agent 的 interface layer**。

很多人直到今天仍然会下意识把 tool 理解成给模型加一个函数或者工具。

这句话当然没错，但仍然只说对了一半。

Anthropic 在 [Writing effective tools for agents](https://www.anthropic.com/engineering/writing-tools-for-agents) 里提出了一个判断：**传统函数和 API，主要是确定性系统和确定性系统之间的契约；而 agent tool，则是确定性系统和非确定性模型之间的契约。**

这个差异会直接改写我们该怎么理解 interface design。

在普通软件工程里，一个接口写得好不好，更多是人类开发者与另一个系统之间的可维护性问题；但在 agent 系统里，tool 的名字、描述、参数形状、错误信息、返回字段，都会直接进入模型的决策回路。它们不是纯粹的后端实现细节，而是 harness 暴露给模型的最小接口单元。

这也是 [SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering](https://arxiv.org/abs/2405.15793) 这篇论文最值得被放进 harness 讨论里的原因。它提出了一个很有用的概念：`ACI`，也就是 `Agent-Computer Interface`。如果说 Anthropic 讨论的是 tool 这种动作接口应该怎样写，SWE-agent 讨论的就是这些动作接口怎样被组织成一个 agent 真正能用的计算机接口。

ACI 的核心判断并不复杂：人类需要 IDE、搜索、语法高亮、错误提示、调试器这些界面来更好地操作计算机；LM agent 也不是裸着面对 shell、浏览器、仓库和 API 就会自然表现最好。Agent 是一类新的“终端用户”，它有自己的能力边界和失败模式，所以也需要专门为它设计的计算机接口。

所以这里的层级关系可以先定下来：

- `tool` 是最小动作接口，决定 agent 能采取什么动作。
- `tool description / schema / output` 是最小语义接口，决定 agent 怎样理解这个动作。
- `ACI` 是动作、反馈、guardrails 和 context management 组成的交互接口。
- `harness` 则把这些接口接入 runtime、权限、验证、回放和产品工作流。

也就是说，tool 和 ACI 不是两件孤立的东西。Tool 是 interface layer 的原子，ACI 是这些原子被组织后的工作界面。一个坏 tool 会让模型误判单步动作；一个坏 ACI 会让模型在整个任务过程中持续误判状态、误用动作、丢失上下文。

从这个统一框架看，tool / ACI design 至少同时在回答五个问题。

第一，**动作空间是否可判别。**
工具不是越多越好，而是越可区分越好。一个 agent 看到十个功能高度重叠、命名非常接近、描述又写得很泛的工具时，增加的不是能力，而是路由噪音。工具集设计首先要服务于动作空间的可判别性，而不是功能列表的完整感。

这也是 ACI 的第一层含义：它不是把所有底层能力一股脑暴露出去，而是把复杂环境重新切成模型能稳定选择的动作。SWE-agent 不是让 agent 裸用所有 shell 组合，而是额外提供 `find_file`、`search_file`、`search_dir`、file viewer、file editor 这些更贴近软件工程任务的动作接口。

第二，**动作是否按 agent affordance 设计。**
很多后端接口是按数据库表、服务边界、组织分工长出来的；但 agent 不关心你的微服务拓扑，它关心自己现在要完成什么动作。一个面向 agent 的 tool，不应该只是底层 API 的薄封装，而应该接近 agent 在任务中自然需要的操作。

SWE-agent 的 file editor 就是一个典型例子。它没有让模型自己拼复杂的 shell 编辑命令，而是让模型按行号替换打开文件中的一段内容。这个接口并不底层，但它正好贴近“修改代码片段”这个 agent 动作。

第三，**语义提示是否足够清楚。**
Tool description / schema 本身就是 prompt。很多人把 prompt engineering 和 tool calling 分开写，但在真实系统里，两者经常是同一件事。工具说明不是给人看的补充文档，而是模型理解动作世界的入口文本。参数名写得是否自然，说明里有没有把触发条件、边界条件、危险动作、典型用法说清楚，都会直接影响 tool selection。

命名空间也会反向塑造模型的路由行为。一个工具叫 `fetch`、`query`、`lookup`、`search`、`read_doc`，模型对它们的预期并不一样；前缀、分组、动词风格是否一致，也都会影响模型把哪个动作优先纳入候选集。很多时候，命名就是最轻量的动作先验。MCP 使用文档字符串来构造工具描述是最自然的思路。

第四，**反馈是否高信号、低噪音。**
Tool output 不是拿到什么就全塞回去。如果一个工具每次都把原始网页、完整日志、整段 HTML、整个对象树原封不动地回注进上下文，污染的不是一次调用，而是后续整条推理链。对 agent 来说，好的返回值不是最全的返回值，而是最适合下一步决策的返回值。

ACI 也是同一个问题。SWE-agent 的 file viewer 会展示带行号、窗口化、可滚动的文件片段；编辑完成后，它会立即显示新的文件状态；当没有输出时，系统也会显式告诉 agent 命令成功但没有产生输出。这些都不是额外装饰，而是在设计环境如何把状态反馈给模型。

第五，**错误是否能被恢复和治理。**
Agent tool 不是只要成功时好用就够了。模型一定会选错工具、填错参数、改坏文件、误解反馈。一个好的接口必须让错误尽早暴露，并把错误转成模型可以理解、可以修正的信号。

SWE-agent 在 `edit` 命令里加入 lint guardrails，就是这个原则的具体化。坏编辑会被拦截，语法错误会被反馈，agent 会被要求重新尝试。这里的 guardrail 不只是安全限制，它还是恢复机制：它把一次可能污染仓库的失败，变成一次可以在上下文里修正的观察。

把这五点合起来看，就能解释为什么 Linux shell 对人类工程师很强，却不一定是 LM agent 的好界面。Shell 的动作空间太宽，命令组合太自由，输出经常太长，错误反馈也未必指向下一步恢复。人类可以靠经验、视觉上下文和长期记忆过滤这些噪音，模型却必须把这些内容塞进有限上下文里继续推理。于是，同一个“计算机环境”，通过不同 interface 暴露给 agent，难度会完全不同。

因此，tool / ACI 层其实同时连接了三件事：

- 它向上连接模型决策，因为模型通过它理解“我能做什么”
- 它向下连接系统边界，因为系统通过它决定“只能做什么”
- 它向侧面连接上下文成本，因为 description、output、observation history 都会反向进入工作记忆

综上所述，**tool 不是一个附着在 harness 上的小零件，它是 harness 的最小动作接口；ACI 也不是额外的 UI 包装，而是把动作、反馈、历史和恢复机制组织成可用环境的交互接口。** Harness 真正暴露给模型的，不是一堆后端能力，而是一套被设计过的行动语言。

而且这个判断不只发生在推理时。

一旦 agent 进入训练、评测和持续优化闭环，tool / ACI 设计就会进一步变成训练对象的一部分。一个动作是被写成结构清晰、边界明确、可验证的工具，还是被留给模型通过自然语言和 shell 自己摸索，它最后决定的，不只是推理阶段的稳定性，还决定了 trajectory schema 怎么切、verifier 能覆盖到哪里、credit assignment 能不能落到关键步骤上、benchmark 到底在测什么。

换句话说，**tool / ACI 这种 harness 结构并不只是在 runtime 里包住模型，它还会反过来塑造训练数据、评测接口和能力学习边界。** 你给 agent 什么样的动作语言，它就更可能学会什么样的行为；你把什么样的工具结果做成可验证中间态，你就更容易把这部分能力写进 reward、judge 与 replay。

很多看起来像模型学会了调用工具的进展，背后其实同时包含了另一半——**harness 先把动作世界整理成了一个更适合被学习、被评测、也更适合被人类治理的形状。**

## 一个短的概念延伸：`model harness`

这里我原本只想留一个很短的延伸，但现在觉得它值得稍微展开一点。

`model harness` 不是当前通行术语，而只是我为了理解未来趋势提出的一个工作概念。它想指的是：未来很多 agent 的竞争，可能不再只是更强的通用底模 + 更厚的外壳，而会越来越像更贴近某类任务形态的模型优化 + 更贴近某类任务形态的 harness 优化。正如 Claude Code 针对 `bash` 命令进行针对性学习那样。

我把它留在这里只是为了提醒自己：不要把模型能力和外围工程写成两个彼此孤立的世界。它不是这篇文章的主干，也不是今天必须接受的定义，但它对理解 tool 和 harness 很有帮助。

因为一旦你认真去看 tool design，就会发现不同模型对同一套工具世界的敏感性并不相同。

有的模型更吃动词清晰、参数显式、错误信息短而硬的接口；有的模型更能容忍说明更长、更接近自然语言的工具描述；有的模型在结构化返回上更稳，有的模型在半结构化脚本化工作流上更有优势。现在的模型在训练的时候已经和工具调用与Agentic的能力耦合了。

这意味着切换基础模型往往不是免费的。你不只是换了一颗更强或更弱的大脑，你还经常在同时改变：

- 工具命名和描述的最优写法
- 返回结构的压缩方式
- 错误提示该有多强的引导性
- 哪些动作适合保留成 schema-first，哪些动作适合交给脚本化编排
- 人类审批点、默认自治程度与工作流节奏
- 模型在不同工具上的能力与偏好

模型的差异会直接产生产品包装的差异与工作方式的差异，当我们在构建一个 Agent 的时候，切换基础模型可能不是免费的。

在 OpenAI 将 Response API 交付给大家的时候，model 和 harness 就不是两个更完全割裂开的概念了。当通用基础模型的发展结束，定制模型与 harness 的系统将会是新的方向。

**未来很多竞争，不是 `model vs harness`，而是 `model-harness co-design`。** 谁能把模型偏好、动作接口、返回结构、验证闭环和产品交互一起调到同一个方向，谁才更可能做出真正稳定的 agent 系统。

## 知识该装进参数，还是装进 harness

既然 model 和 harness 要被一起设计，那就还有一根经常被略过的轴值得单独拎出来：**同一份“知识 + 能力”的预算，到底有多少该压进模型权重，又有多少该外置到 harness 和 `skills` 里。** 这一段比 `model harness` 更靠近畅想，但它是我现在最想往前推的一个问题。

先把命题摆清楚。固定参数能承载的信息总是有限的。我在[《从记忆形成到记忆治理：Agent Memory 的全景图》](/blog/2026/03/21/agent-memory-panorama/)里用过 L1 全上下文、L2 外部记忆、L3 参数记忆三层来谈“知识住在哪里”，但那篇的主线是**加法**——怎么把重复经验写回 L3 权重。这里我想问的是反方向的问题，一个**减法**问题：当参数预算本就紧张时，**让模型主动卸下一部分知识、只保留能力（推理、工具使用、指令遵循、长程规划），把知识外置出去**，是不是一个值得认真研究的方向。换句话说，知识和能力在权重里本来是隐式耦合的，能不能把它拆成一根可以显式分配的预算轴。

顺着这根轴往一端推，就是一个很诱人的形态：一个极小的 agent core，加上 `bash`，再外挂一大堆 `skills`，可能就已经能把不少事情做得很好——知识被推到 L2 的 `skills`、文件系统和检索库里，能力留在那颗不大的模型里。这恰好和本文前面那个判断同源：专用 agent 的可靠性优势，往往来自它更少依赖语言这单一接口。微型 agent 的赌注也是一句话——**“小能力 + 外置知识”的组合，可能优于“巨模背下全部知识”。**

但我不打算把这写成一个立论，因为它的代价同样真实。

第一，**能力有地板，知识没有**。`skills` 这种半结构化入口能成立，前提恰恰是模型已经强到能读懂说明书、判断何时触发、何时运行脚本。知识可以外置，能力不能外借——一个能力太弱的小模型，外挂再多 `skills` 也调度不动，最后只会在一堆它分不清的能力前面打转。

第二，**复杂度不会消失，只会搬家**。大 harness 的“笨重”，换成微型 agent 之后，会变成一个大 `skill` 集 + `bash` 编排的另一种笨重：知识检索的延迟、按需加载的上下文成本、触发判别的路由噪音，全都会回来。本文前面讲 tool 时那句话在这里同样成立——**工具不是越多越好，而是越可判别越好**；`skills` 多到一定程度，模型面对的不是能力，而是噪音。

第三，**外置知识自带可靠性与治理负担**。知识一旦搬到模型外面，就要面对检索失败、内容过期、以及 `skills` 投毒与供应链风险——这些我在 Skills 那篇里已经展开过。

第四，**“卸载知识、保留能力”目前更像研究假设，而不是可交付路线**。知识与能力在权重里本就纠缠，要干净地剥离其中一半并不容易。

它们更像 `model-harness co-design` 这片空间里的两个点：预算往模型权重压，还是往 harness 与 `skills` 压，取决于任务的知识更新频率、结果可验证性，以及模型自身的能力门槛——高频更新、强可验证、能力够用的场景，天平会偏向外置；反之则偏向把知识沉进权重。

**一个极小 agent 系统搭配 `bash` 和一堆外挂 `skills`，是值得研究的方案，但不一定是正确的方案。** 我把它留在这里，和前面那句一样：当真正的 `perfect harness technology` 始终不会出现、而 AGI 到来时这一切又都不再重要——在这两者之间的漫长地带，知识与能力怎么在模型和 harness 之间分配，恰恰是现在最值得动手去试的问题之一。

## 结语

如果我要把这一篇总结成最后一句话，那就是：Harness 不是新发明，它是旧问题的新聚光灯。

这个词有用，因为它提醒我们别再把 agent 错认成裸模型；它也危险，因为它太容易宽到什么都能装。我的用法会比较保守：接受 `agent = model + harness` 这条纠偏公式，但进入具体工程问题时，继续拆成 `工程 harness + 产品 harness + 用户友好 harness`，并把 `task interface` 拉到前台。一旦 `harness` 开始偷走信息量，就换回更具体的词。

上一篇回答的是这些外壳在工程上具体怎么工作。这一篇回答的是为什么今天大家把这整圈东西叫作 `harness`，以及这个词该如何拆、如何用、如何避免变空。

对我来说，两篇放在一起，指向的是同一个判断：Agent 产品的难点不只在模型，也不只在框架，而在模型、外壳和任务界面怎样一起被设计成一个可工作的系统。

## 参考资料

- OpenAI, [Harness engineering: leveraging Codex in an agent-first world](https://openai.com/index/harness-engineering/)
- OpenAI, [Unlocking the Codex harness: how we built the App Server](https://openai.com/index/unlocking-the-codex-harness/)
- OpenAI, [The next evolution of the Agents SDK](https://openai.com/index/the-next-evolution-of-the-agents-sdk/)
- Anthropic, [Building effective agents](https://www.anthropic.com/research/building-effective-agents/)
- Anthropic, [Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)
- Anthropic, [Writing effective tools for agents](https://www.anthropic.com/engineering/writing-tools-for-agents)
- LangChain, [The Anatomy of an Agent Harness](https://www.langchain.com/blog/the-anatomy-of-an-agent-harness)
- Yang et al., [SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering](https://arxiv.org/abs/2405.15793)
- [EleutherAI / lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [ServiceNow / BrowserGym](https://github.com/ServiceNow/BrowserGym)
- [harbor-framework / terminal-bench](https://github.com/harbor-framework/terminal-bench)
- [《把 LLM 关回笼子里：从 Claude Code 看 Harness 如何把概率 Agent 固化为系统约束》](/blog/2026/03/20/building-agent-deterministic-constraints/)
- [《从智能体的认知结构到智能体框架》](/blog/2026/03/03/cognitive-architecture-to-agent-framework/)
- [《从 MCP 到 Agent Skills：为什么 Agent 又需要一种新的上下文工程协议？》](/blog/2026/03/10/from-mcp-to-agent-skills/)
- [《从记忆形成到记忆治理：Agent Memory 的全景图》](/blog/2026/03/21/agent-memory-panorama/)
- [《Model Is Good Enough：2026 年，AI 真正稀缺的是应用而不是更大的模型》](/blog/2026/03/18/model-is-good-enough/)
- [《Claude Code or Codex：编码模型差异如何变成产品体验的不同》](/blog/2026/04/10/how-to-choose-the-right-model-for-developers/)
