---
layout: blog-post
title: "给 LLM 戴上确定性枷锁的外围工程：从 Claude Code 看 Agent Harness"
date: 2026-03-20 21:00:00 +0800
series: Agent时代的基础设施
categories: [智能体系统]
tags: [Agents, Harness, Claude Code, MCP, Permissions, Context Engineering]
author: Hyacehila
excerpt: 真正把 Agent 压成可交付系统的，不是核心 loop，而是围绕 LLM 不确定性搭出来的外围工程：把语言请求进一步下沉成工具契约、知识路由、生命周期验证、隔离恢复与自治治理。
featured: true
math: false
---

# 给 LLM 戴上确定性枷锁的外围工程：从 Claude Code 看 Agent Harness

如果今天要研究怎么给 `LLM` 戴上确定性枷锁，我现在更愿意先从 Claude Code 谈起。不是因为它把问题都解决了，而是因为它把 Harness 里最关键的结构件直接暴露给了开发者：`MCP`、`CLAUDE.md` 与 `rules`、`hooks`、`subagents`、`checkpointing`、`permission modes`、`plugins`、`Agent SDK`。在一个真实可用的系统里，你几乎可以直接观察语言约束如何被逐步压成系统约束。

真正把 Agent 变成可交付系统的，不是核心 loop 本身，而是围绕 `LLM` 不确定性搭出来的一整套外围工程。核心循环当然重要，但它往往只是最短的一段。

最长、也最贵的一段，通常是把这个循环关进一个可控系统里。不是让它更会想，而是让它在想错、说错、调错工具、上下文漂移、连续运行几十轮之后，系统仍然不至于一起失控。

几乎每一个认真用过 Claude Code 的人，都经历过同一个阶段：开始疯狂往 `CLAUDE.md` 里写东西。

起因通常很普通。Claude 改了不该改的文件，于是你补一条“不要动这个目录”。它没跑测试就宣布完成，于是你再补一条“必须先跑测试”。它用了错误的包管理器、在奇怪的时机 `commit`、或者把不相关的文档也一起改了，于是 `CLAUDE.md` 越写越长，从几行涨到几十行，再涨到几百行。

然后你发现：它还是会犯。

不完全是同一个错误，而是变种。你写了“不要直接 `push`”，它不 `push` 了，但开始在奇怪的时机 `commit`。你写了“必须先跑测试”，它也跑了，但跑的是不相关的测试文件。你会很自然地产生一个念头：是不是模型还不够强，换一个更强的就好了。

但很快你会意识到，问题不只在模型。问题在别的地方。问题在于：**你仍然在用语言去请求一个概率语言模型按你的方式行动。**

语言请求当然有用，但它天然是概率性的。任务越长、工具越多、上下文越脏、状态越复杂，它就越会衰减。你往 `CLAUDE.md` 里堆的每一条规则，本质上都在和上下文长度、注意力漂移和临场变通赛跑。

这就是我现在想用“确定性枷锁”概括的问题：`LLM Agent` 让系统第一次有能力处理开放世界的不确定性，但工程系统首先要做的，恰恰是给 `LLM` 本身的不确定性戴上枷锁。

也正因为如此，真正让 Agent 可交付的，不是继续往文档里补“下次别再犯”的句子，而是把这些句子一层层压成系统结构：哪些工具存在、哪些工具不存在；哪些知识常驻、哪些知识按需注入；哪些动作在生命周期节点上会被拦截；哪些任务必须在隔离的上下文里完成；哪些“完成声明”会被验证系统直接拒绝。

这正是我现在理解的 `harness` 问题。下一篇我会专门讨论这个词本身，但这一篇我更关心另一个问题：**这些外围工程在产品里到底是怎么长出来的。**

值得注意的是，在这篇文章原始发布日期 `2026-03-20` 之后，Anthropic 又在 `2026-03-24` 和 `2026-03-25` 分别公开了长时应用开发的 Harness 设计与 `auto mode` 的安全栈。补齐了以前尚未展开的两层：**恢复与验证的运行时结构，以及自主性治理。**

## 语言约束为什么会衰减

这不是提示词无效，而是提示词的作用边界很明确。

Claude Code 的官方文档一直把 `CLAUDE.md` 放在 “memory / context” 这条线上讨论，而不是把它当成强制配置文件来讨论。[Store instructions and memories](https://code.claude.com/docs/en/memory) 里说得很直接：`CLAUDE.md` 和 auto memory 都是上下文，而不是 enforced configuration。事实型、常驻型、跨任务稳定成立的约束适合放在 `CLAUDE.md`；一旦进入条件加载、路径匹配、生命周期拦截、权限裁剪和沙箱边界，事情就已经不再是单纯的“写文档”。

这其实已经把问题说透了。

`CLAUDE.md` 的强项是让模型在会话开始时拿到一组重要事实。它的弱项是：**它仍然活在上下文里，而不是活在系统结构里。**

所以只要任务复杂起来，语言约束就会同时面临几个老问题：

- 它会被遗忘。
- 它会被误解。
- 它会被临场“合理变通”。
- 它会在长链任务后半段权重下降。

写好提示词不再应该是 Agent 工程的主线。提示词当然重要，但它更像是最靠近模型、也最软的一层约束。真正成熟的系统，一定会把一部分原本靠提示词维持的希望，迁移到模型之外。

## 技术工程的分层与 Claude Code 的 Harness

如果把这件事再拆开，我现在更愿意把给 `LLM` 戴上确定性枷锁理解成七层技术工程。它们不是某家框架的 `feature list`，也不是某个厂商的 `marketing terminology`，而是所有可靠 Agent 系统迟早都会长出来的控制层。

前两层处理的是“系统到底接收什么状态、沿着什么流程推进”。结构化输入输出层负责让系统只接受被 `parser`、`validator` 和 `schema` 验过的状态，而不是看起来像答案的东西；控制流与任务闭环层负责把 Agent 关进一个有终点、有 `budget`、有 `checkpoint`、可回退也可插手的流程，不让规划无限发散。

中间三层处理的是“它能碰什么、看见什么、谁能拦住它”。工具与运行时层把 `shell`、`browser`、`API` 这类能力改造成有参数约束、副作用边界和上下文预算的契约；上下文与系统提示词层决定哪些知识常驻、哪些按需注入、哪些以前置索引和文档入口的形式进入工作记忆；网关、流量与权限层则负责 `rate limit`、`auth`、`quota`、审计和访问边界，用更强的确定性去对冲模型引入的新攻击面和成本面。

最后两层处理的是“它做完没有、长期会不会慢慢烂掉”。评测、验证与恢复层负责在外部环境里证明任务真的完成，并在失败后能 `replay`、`rollback`、继续推进；可观测、运维与治理层负责盯住漂移、坏模式复制、`AI slop` 和熵增，不让一次错误在系统里被规模化复用。Claude Code 暴露给开发者看到的是产品化表面，而这些表面背后对应的，正是更抽象的控制工程。

Claude Code 值得分析，就在这里。官方甚至在 [How Claude Code works](https://code.claude.com/docs/en/how-claude-code-works) 里直接把它定义成 “the agentic harness around Claude”。问题于是就不再是 Claude 能做什么，而是：**这个 Harness 把哪些本来靠语言维持的约束，下沉成了产品里可见的结构件。**

先理解给 `LLM` 戴上确定性枷锁的抽象控制层，再看 Claude Code 在产品里把哪些层做成了可见设计面：

| Claude Code 可见设计面 | 在产品里长什么样 | 它真正解决的 Harness 问题 | 主要映射到哪些工程层 |
| --- | --- | --- | --- |
| 工具契约层 | `MCP`、`built-in tools`、`tool search` | 定义动作契约、裁剪能力边界、控制工具进入上下文的成本 | 工具与运行时层、部分权限层、部分上下文层 |
| 上下文路由层 | `CLAUDE.md`、`.claude/rules/`、`imports`、`auto memory`、`managed settings` | 决定哪些知识常驻、哪些按条件注入、哪些由系统强制 | 上下文层、部分治理层 |
| 生命周期验证层 | `hooks`、`prompt hooks`、`agent hooks` | 在动作前、动作后、结束前和 `compaction` 前后插入验证器与状态回注 | 控制流层、验证恢复层 |
| 隔离与恢复层 | `subagents`、`sessions`、`checkpointing`、`--fork-session` | 隔离上下文、分叉任务路径、回滚文件状态、显式 `handoff` | 控制流层、上下文层、恢复层 |
| 自主性治理层 | `permission modes`、`protected paths`、`sandbox`、`auto mode` | 调节自治程度、对冲 `approval fatigue`、阻断高风险越界 | 权限层、网关与治理层、安全层 |
| 可分发 Harness | `plugins`、`GitHub Actions`、`Agent SDK` | 把本地 Harness 原语打包成可复用、可部署、可集成的部件 | 平台化、运维、分发层 |

这里故意不是把七层工程抽象机械地摊成七个产品功能。前面的七层，是按控制能力在系统内部怎么分工来切；这里的六个设计面，则是按开发者在产品里实际能看见什么、能配置什么、能调试什么来切。两者是一对多映射，而不是一一对应。

也正因为如此，有两层不会以独立标题直接出现在后文里。结构化输入输出层更多藏在 `tool schema`、`hook/verifier`、`checkpoint` 和任务状态约束里；可观测、运维与治理层则分散在 `permissions`、`sessions`、审计、`OpenTelemetry`、插件分发和 `SDK` 接口这些表面上。后文继续按六个可见设计面展开，但每一节都会回到它背后的控制层。


## 一. 工具契约层：`MCP` 不只是扩能力，也是压缩动作世界

很多人第一次接触 `MCP`，会把它理解成“让 Claude 会更多事”。

这个理解不算错，但只理解了一半。`MCP` 更深的价值，不在能力扩展，而在能力边界的划分。

Anthropic 在 [Writing effective tools for agents](https://www.anthropic.com/engineering/writing-tools-for-agents) 里有一个非常重要的判断：传统函数和 API，是确定性系统和确定性系统之间的契约；而 tool，是确定性系统和非确定性 agent 之间的新契约。这意味着工具层最重要的不是暴露更多动作，而是**把动作写成 agent 可以安全使用、系统可以稳定消费的契约**。

放到 Claude Code 上，这个判断会变得非常具体。

没有 `MCP` 时，Claude 想查数据库、碰内部服务或者读 Issue，很可能要在终端里现场猜路径、猜认证方式、猜命令序列。它当然可能猜对，但每次执行路径都带着临场推断。

有了 `MCP` 之后，事情被改写了。Claude 不再需要自己猜“数据库在哪、如何认证、失败时怎么恢复”，它看到的是一组已经被裁剪、命名、约束过的工具接口。`query_database(sql: string)`、`get_ticket(id)`、`search_internal_docs(query)` 这种接口，真正重要的不是功能更多，而是**实现细节已经被收走了，剩下的是系统愿意让 agent 看到的那部分能力边界。**

**能力边界即架构边界。** 没有 `delete_user_data` 这个工具，这个动作就不只是“不建议做”，而是在技术层面“不存在”。这和传统软件里的最小权限原则是同一件事，只不过现在这个原则被应用到了 agent 工具设计上。

但只把 `MCP` 理解成权限边界也还不够。Claude Code 官方还把这层和上下文预算直接绑在了一起。[How Claude Code works](https://code.claude.com/docs/en/how-claude-code-works) 与 [Connect Claude Code to tools via MCP](https://code.claude.com/docs/en/mcp) 都明确提到：`MCP` tool definitions 默认会通过 tool search 延迟加载，Claude 先只看到工具名，真正的 schema 在需要时才进入上下文。这意味着工具层除了定义能力边界，还在定义**上下文成本边界**。

这件事其实很关键。工具的“出口设计”最后会进入模型的上下文。如果一个工具把整页数据库记录、整份 HTML 或完整日志一股脑塞回来，污染的不是单次调用，而是后续整条推理链。好的工具设计，会在服务端先做过滤、压缩和字段选择；好的工具发现机制，则会控制多少工具描述真正进入上下文。

所以 `MCP` 这一层解决的，从来不只是 Claude 怎么调用外部世界，而是三件事情：

- 工具契约必须由系统定义，不能交给模型现场猜。
- 动作空间必须被主动裁剪，而不是靠 prompt 事后约束。
- 工具进入上下文的时机和粒度，也属于 Harness 设计，因为工具描述与工具输出都会反向塑造后续推理。

这也是为什么 OpenAI 在 [From model to agent: Equipping the Responses API with a computer environment](https://openai.com/index/equip-responses-api-computer-environment/) 里会把 shell、hosted container、skills、compaction、状态持久化这些东西直接下沉成平台原语。词可以不一样，但工程事实是一致的：**真正难的从来不是“给模型一个动作”，而是“给模型一个被约束过的动作世界”。**

从控制层的角度看，这一节的落点主要是工具与运行时层，同时还牵动权限边界和上下文成本控制。

## 二. 上下文路由层：`CLAUDE.md`、`rules`、`auto memory` 与 `docs index` 的分工

有了更可靠的工具边界之后，第二类问题很快会浮出来：Claude 还是会做出不符合团队习惯的决定。

比如用了你们已经废弃的 `API` 版本，没有遵循既有的 `Repository` 模式，或者把本来只该用于排查告警的流程拿去处理日常开发任务。这个时候，最自然的动作还是继续往 `CLAUDE.md` 里补规则。

方向没错，但载体往往不够好。

Claude Code 的记忆系统现在其实已经分成了几层，而不是单一的 `CLAUDE.md`：

- `CLAUDE.md` 和 `CLAUDE.local.md` 负责常驻说明。
- `.claude/rules/*.md` 负责把说明切成模块，并允许用 `paths:` `frontmatter` 做条件加载。
- `@path` `import` 负责把 `repo docs`、`README`、流程文档稳定挂进入口文件。
- `auto memory` 负责让 Claude 自己沉淀跨会话学习。

这背后的工程逻辑是：**每一次 Agent 的失败，都是某条隐性知识还没有被显式编码进系统，或者还没有被放在正确加载层级上的信号。**

也正是在这里，Vercel 那篇 [AGENTS.md outperforms skills in our agent evals](https://vercel.com/blog/agents-md-outperforms-skills-in-our-agent-evals) 仍然有很大参考价值。它最重要的结论不是 `AGENTS.md` 打败了 `skills`，而是：**知识暴露顺序本身就是约束强度的一部分，设计这套暴露机制/上下文路由本身很重要。**

他们的结果很直接：

- 没有文档时，`baseline` 通过率是 53%。
- 默认 `skills` 触发时，结果几乎没有改善，仍是 53%。
- 显式提示模型去用 `skills` 时，提升到 79%。
- 把压缩后的文档索引直接放进仓库根部 `AGENTS.md` 时，做到了 100%。

这里真正发生的事情，不是哪种格式更高级，而是：**系统有没有把要不要去读这份知识继续交给模型临场决定。**

所以这一层我现在更愿意下一个更硬的结论：

- `CLAUDE.md` 解决常驻事实。
- `.claude/rules/` 解决条件注入和路径匹配。
- `auto memory` 解决跨会话学习。
- `repo docs` 继续做 `system of record`。
- `managed settings` 负责把不该靠语言维持的约束收回到客户端强制层。

从这个意义上说，真正值得学是**把知识装配改造成知识路由。**

换回抽象层语言，这里讨论的核心是上下文与系统提示词层，只是它在产品里表现成了更细的知识路由机制。

## 三. 生命周期验证层：`hooks` 不只是脚本，而是外部 `verifier` 接口

到了第三层，问题会再次收紧。

假设你已经有了合理的工具边界，也把关键流程写进了 `skills` 或仓库文档里，Claude 仍然可能在长任务末尾说一句“完成了”，但你一看，测试没跑，或者跑错了。

这时你就会发现：`请确保测试通过` 这种句子，本质上还是语言请求。

Claude Code 的 [Hooks reference](https://code.claude.com/docs/en/hooks) 很重要，因为它把这件事从提醒直接推进到了生命周期事件。你不再只是对模型说“请这样做”，而是在 `SessionStart`、`InstructionsLoaded`、`PreToolUse`、`PostToolUse`、`Stop`、`PreCompact`、`PostCompact` 这些节点上挂接外部逻辑。

这件事的意义，不在脚本自动化本身，而在于：**完成声明、工具调用、上下文压缩这些原本只活在模型叙事里的事件，第一次被外部系统拿回来了。**

`PreToolUse` 的价值，是把越界动作变成可以在发生前被阻断的事情。比如禁止高风险命令、限制危险路径、阻止某类写操作。这时约束不再是“请不要这么做”，而是“你根本不能做”。

`Stop` 的价值更大。它把模型说自己做完了，变成一个可以被验证器拒绝的系统事件。官方文档现在已经不只支持 `command hook`，还支持 `prompt hook` 和 `agent hook`。也就是说，`Stop` 不只是跑个 `shell` 脚本，它还可以拉起一个带工具的 `verifier subagent` 去检查测试、读文件、比对工件，再决定要不要允许会话结束。

这一步非常关键，因为它说明 Claude Code 的生命周期层，已经不只是“事件回调”，而是在产品层面正式打开了 **`generator / verifier` 分离** 的接口。

这也正好和 Anthropic 在 `2026-03-24` 发布的 [Harness design for long-running application development](https://www.anthropic.com/engineering/harness-design-long-running-apps) 形成互文。那篇文章明确把长时应用开发写成 `planner / generator / evaluator` 三代理结构：生成器负责推进，评估器负责打分、找 `bug`、把输出拉回规格。这其实就是一个更通用的 Harness 原理：**不要让同一个 `agent` 既负责产出，又负责无限信任自己的完成声明。**

从这个角度看，Claude Code 的 `hooks` 就不只是“自动化小脚本”，而是把 `verifier` 接进运行时的标准口。

`InstructionsLoaded`、`SessionStart`、`PreCompact`、`PostCompact` 则处理另一个更隐蔽的问题：上下文压缩和状态蒸发。长任务里最危险的事情之一，不是模型一开始没拿到约束，而是它在压缩、恢复和切换阶段把约束丢了。把关键状态外置到文件、脚本、检查点和生命周期注入里，才能避免前面明明说过的知识在后面自然蒸发。

这一层最重要的工程判断，我现在会写得非常重：

- `PreToolUse` 负责在动作发生前阻断越界行为。
- `Stop` 负责把“完成声明”变成可拒绝的系统事件。
- `prompt hooks` 和 `agent hooks` 说明验证器本身也已经被产品化，而不只是 `shell glue`。
- `PreCompact / PostCompact / SessionStart / InstructionsLoaded` 负责把关键状态从上下文窗口外置，并允许你调试“它到底加载了什么”。
- 越接近安全和强制性约束，越应使用确定性实现，而不是再交回给 `LLM` 判断。

这也是为什么 Anthropic 在 [Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) 里要非常清楚地区分 `agent harness` 和 `evaluation harness`。前者让模型以 `agent` 的方式行动，后者运行任务、记录轨迹、打分和汇总结果。两者分开之后，你才会发现一个关键事实：**没有验证闭环的 Agent，本质上只是把一次采样结果直接暴露给用户。**

`hooks` 的价值，就在于它让一部分原本只能在评测或人工 `review` 里完成的检查，被提前嵌进了运行时，并且它一定会被执行，而不是 `CLAUDE.md` 里的一句“请进行测试并保证测试通过”。

在七层工程里，这一节对应的是控制流与任务闭环，以及评测、验证与恢复之间的接口。

## 四. 隔离、分叉与恢复：`subagents` 只是入口，不是全部

`subagents` 很容易被讲成并行化或者提速功能。

这当然不是错，但如果只看到这里，你会错过它更核心的工程意义：**把一个大而脏的解空间拆成多个更窄、更可控、更容易验证的解空间。**

Claude Code 的 [Create custom subagents](https://code.claude.com/docs/en/sub-agents) 文档强调的是专门化、隔离的上下文和定制工具访问。对工程来说，这件事真正值钱的地方，不是多开几个 `agent`，而是每个 `agent` 可以在更干净的上下文里、用更受限的工具集，只解决一个更窄的问题。

一个 `reviewer subagent` 如果只有读权限，那么“不要修改文件”就不再是一条依赖模型自觉的提示词，而是它的动作空间里根本没有 `Edit`。一个在隔离工作区里运行的 `subagent`，即便失败，也不会把主工作区一起拖脏。一个只负责测试和验证的 `subagent`，不需要背着前面几十轮探索历史继续工作。

但如果把视角再放大一点，你会发现 Claude Code 官方这两个月已经把隔离这件事从 `subagent` 扩到更完整的一组恢复原语上了：

- [Checkpointing](https://code.claude.com/docs/en/checkpointing) 会在每次编辑前自动快照文件状态。
- [How Claude Code works](https://code.claude.com/docs/en/how-claude-code-works) 说明会话是本地 `JSONL`，可 `resume`，也可 `--fork-session` 分叉。
- `session-scoped permissions` 在 `resume` 或 `fork` 时不会继承，说明权限状态本身也被当成显式边界，而不是隐式延续。
- `session` 与目录绑定，官方直接建议用 `git worktrees` 跑并行会话，避免同一 `session` 在多个终端里互相污染。

这组设计放在一起，才是这一层真正完整的样子：它不只是开 `subagent`，而是**把长任务拆成可隔离、可回滚、可分叉、可 `handoff` 的局部状态机**。

这也正好和 Anthropic 在 2026-03-24 那篇 Harness 文里讲的重点完全一致。那篇文章明确区分了 `compaction` 和 `context reset`：`compaction` 只是原地压缩，同一个 `agent` 继续跑；`reset` 则是给下一个 `agent` 一个干净上下文，只通过结构化 `handoff artifact` 交接必要状态。原文甚至直接写道：`compaction` 保留连续性，但不提供 `clean slate`；而 `reset` 虽然要付出 `handoff` 成本，却能真正切断 `context anxiety`。

从这个角度看，这一层的真正价值是四件事：

- 独立上下文，减少噪音污染。
- 缩窄工具集，减少越界路径。
- 明确 `handoff`，让每一轮都知道自己接的是什么。
- 结合 `checkpoint`、`resume`、`fork` 和 `worktree`，把失败限制在更小的局部里。

**限制不是缺陷，而是可靠性的来源。**

自由度越高，失控路径越多；解空间越窄，行为越容易预测。很多时候，真正把 Agent 做稳的方法，不是继续放大它的自主性，而是把任务拆给多个更受限、更好验证、而且可分叉恢复的执行单元。

如果改用控制层视角来描述，这一节谈的是控制流、恢复和上下文隔离如何一起收窄失败半径。

## 五. 自主性治理：`permission modes`、沙箱与 `auto mode`

因为如果没有这一层，Claude Code 看起来就还像一个默认要不断弹权限框的 `agent` 工具。可现在已经不是这样了。它实际上已经长出了一条非常清晰的**自主性梯度**。

[Choose a permission mode](https://code.claude.com/docs/en/permission-modes) 里现在已经把日常最常见的四个模式排成一条明确梯度：

- `default`：只默认读。
- `acceptEdits`：自动接受工作目录内的文件编辑和常见文件系统命令。
- `plan`：研究和提出方案，但不改源文件。
- `auto`：不弹人工确认，由后台安全检查代替。

如果再加上更极端的 `bypassPermissions`，就能更清楚地看到这是一个策略层，而不是几个 `UI` 开关。

更重要的是，这条梯度不是“随你点哪个都一样”。官方文档里已经明确把一些边界硬编码成了模式无关的 `guardrail`：

- `protected paths` 在任何模式都不会被无条件放行。
- `acceptEdits` 只自动批准工作目录内的编辑和有限的文件系统命令。
- `bypassPermissions` 也仍然会对 `protected paths` 保留最后一道提示。
- `auto` 则不是“什么都放”，而是把高风险动作送给分类器。

这意味着 `permission modes` 已经不是“用户体验偏好”，而是**自治程度如何被制度化管理**的问题。

再往下看，Anthropic 在 [Making Claude Code more secure and autonomous with sandboxing](https://www.anthropic.com/engineering/claude-code-sandboxing) 里公开的东西就更值得注意了。它不是把沙箱写成一个安全附件，而是直接把它写成更安全也更 `autonomous`的基础条件。

如果说沙箱回答的是“能在哪些硬边界内放权”，那么 [Claude Code auto mode](https://www.anthropic.com/engineering/claude-code-auto-mode) 回答的就是另一个问题：**怎样在不回到人工审批疲劳的前提下，给高自治一个更细粒度的策略栈。**

这篇文章提到：用户实际会批准 93% 的 `permission prompt`。也就是说，单纯把一切风险都压回人工确认，最后很容易演化成 `approval fatigue`。

它介绍了 `auto mode` 的两层防线：

- 输入层有一个 `prompt-injection probe`，在工具结果进入 `agent` 上下文前先扫一遍。
- 输出层有 `transcript classifier`，在动作执行前判断这一步是否真的被用户授权，以及是否越过了信任边界。

它不是单层分类器，而是两阶段结构：先用一个快速、偏保守的单 `token filter` 做初筛，再只对被 `flag` 的动作启用带推理的第二阶段，尽量把成本和误杀率压下来。

它把多代理 `handoff` 也纳入了审查范围。官方文里明确写了，`classifier` 会在 `subagent delegation` 的出去和回来两端都做检查，因为是否用户真授权了这个任务在 `handoff` 处最容易失真。

当你想提高自治（自主智能体一定是我们想要的），不应该只想着“怎么让主 `agent` 更聪明”，还应该同时设计“谁来判断这一步是否越权”。

所以这一层最值得保留的判断是：

- `default / acceptEdits / plan / auto` 不是几个便利模式，而是一条自治梯度。
- `protected paths`、`managed settings`、沙箱和分类器一起构成了自治的硬边界。
- 真正成熟的自治，不是去掉审批，而是把审批背后的判断逻辑重写成系统结构。

更抽象地说，这一层是在回答自治如何被治理，因此它落在权限控制，也自然外溢到可观测与治理。

## 六. 可分发 Harness：`plugins`、`GitHub Actions`、`Agent SDK`

前面几层都还比较像本地工具怎么约束自己。但 Claude Code 其实还有一个更容易被低估的设计面：它已经不只是一个 `CLI`，而是开始把自己的 Harness 打包成可分发组件。

先看 [Create plugins](https://code.claude.com/docs/en/plugins)。官方现在给插件的定义已经非常明确：插件不是只装 `skill` 的小扩展，而是可以打包 `skills`、`agents`、`hooks`、`MCP servers`、`LSP servers`、`monitors`、`bin/` 可执行文件和默认 `settings` 的复合单元。也就是说，一个插件本质上就是一包可搬运的 Harness 配置。

Claude Code 已经不把 Harness 视为用户机器上的零散私有配置，而是视为可版本化、可分享、可装配的部件。你今天在本地调好的技能、`agent`、`hook` 和监控器，明天可以封成插件，变成团队级基础设施。

再看 [Claude Code GitHub Actions](https://code.claude.com/docs/en/github-actions)。官方文档写得很清楚：它允许你在 `GitHub workflow` 里运行 Claude Code，本身又建立在 Claude Agent SDK 之上，而且会尊重仓库里的 `CLAUDE.md` 标准。这意味着同一套本地 Harness 逻辑，可以被搬去做 `PR` 创建、`issue` 实现、`code review` 和自动化修复。

最后是 [Agent SDK overview](https://code.claude.com/docs/en/agent-sdk/overview)。它提供的是“the same tools, agent loop, and context management that power Claude Code”，只是被做成了 `Python` 和 `TypeScript` 可编程接口。更重要的是，它不是只暴露一个 `query API` 就结束了，而是连 `hooks`、`subagents`、`MCP`、`permissions`、`sessions`、`checkpointing`、`OpenTelemetry observability` 都一起暴露出来。在通用任务Single Agent 任务上，复用Claude Code SDK 或者类似 PI 之类的成品 Harness 可以让我们利用现成的不确定性压缩以及工程收敛的结果，Harness as a service（Haas）

这就意味着 Claude Code 背后的 Harness 已经出现了一个很清楚的平台化趋势：

- `CLI` 是交互表面。
- 插件是配置与能力分发单元。
- `GitHub Actions` 是 `CI` 运行表面。
- `Agent SDK` 是程序化嵌入表面。

换句话说，Claude Code 不是一个孤立产品，而是**一组被产品化的 Harness 原语**。这也是为什么我会把这一层叫作可分发 Harness。

> 题外话：OpenAI Response API开始将大量的工具调用，Shell，代码解释器等放到API内部处理，这也可以理解为一种 Harness as a service.

这一层不是新增的第八层，而是把前面几层已经形成的控制能力，进一步做成平台化、运维化和可分发的基础设施表面。

## 如果你还在写语言请求，更好的结构通常会长成什么

把前面六层压回工程语言之后，我现在更愿意把对照关系写成下面这样：

| 如果你只写语言请求 | 更硬的结构约束通常会长成什么 |
| --- | --- |
| “让它自己去查系统、自己调工具” | `MCP` 契约 + `tool search` + 服务端过滤后的工具输出 |
| “遵守团队规范和目录边界” | `CLAUDE.md` 入口 + `.claude/rules/` 条件加载 + `managed settings` 的强制层 |
| “完成前记得检查一下” | `Stop` `hook` + `prompt/agent verifier` + `generator/evaluator` 分离 |
| “别把主上下文和工作区搞脏” | `subagents` 隔离 + `checkpoints` + `resume` / `fork-session` + `worktree` |
| “尽量少打扰我，但别胡来” | `permission modes` + `protected paths` + `sandbox` + `auto mode` `classifier` |
| “把这套经验复用到别的仓库和运行场景” | `plugins` + `GitHub Actions` + `Agent SDK` |

这张表最想说明的一点其实很简单：**语言请求没有消失，但真正决定系统可靠性的，已经不是语言请求本身。**

## 这也是框架真正替你做掉的大头工作

当你把这些问题摊开之后，`framework` 这个词就会自动去魅。

框架并不是因为大家不会写 `while loop` 才存在，也不是因为 `planner` 特别难写才存在。框架真正应该替你做掉的大头工作，是把这些一遍遍出现、又不能总靠人工重讲的 Harness 技术冻结成可复用约定：

- 状态如何定义。
- 工具如何包装和延迟装载。
- 文档如何前置、切片和路由。
- 生命周期节点如何拦截、校验和回注。
- 子任务如何隔离、分叉、恢复与 `handoff`。
- 自治等级如何被治理，而不是只靠人工点确认。
- 这些约束如何被封装成插件、`CI` 集成和 `SDK`。

**框架不是智能本身，而是 Harness 技术的收敛形态。**

你当然可以不用框架，但你不用框架，不代表这些问题会消失。更常见的现实是：你最后还是会把它们自己补回来。区别只在于，是每个项目都重造一遍，还是有人已经把其中一部分抽成了稳定部件。

> 题外话：现有的框架包括极简框架（如Pocketflow）为用户提供Agent架构的理解，高度集成框架（LangChain/Graph）为用户提供封装Agent结构，但单纯提供Harness本身是不足的，Agent时代需要新的基础设施帮助开发者更好的开发，但不需要过多封装导致留下技术债。真正通用且好用的框架离我们还很远。

## 结语

回到文章开头那个把 `CLAUDE.md` 写得越来越长的工程师。

他做的事情并没有错，只是停在了一个更软的层次上。用语言去约束语言模型，就像把“不要犯错”写进员工手册：有帮助，但远远不够。真正的工程答案，是给它搭一套工作系统：它能碰什么工具、哪些规范会在正确时机被加载、哪些动作在生命周期节点会被拦下、哪些任务要在隔离环境里做、它说“完成”时谁来验证、它可以被放权到什么程度、以及这套约束怎么被带进 CI 和别的运行表面。

这也是我现在最想保留的一句判断：**真正重要的不是让模型更会听话，而是让系统在模型不听话时仍然有边界。**

这一篇回答的是：这些外围工程在产品里具体怎么工作，以及为什么 Claude Code 是观察它们的绝佳入口。

下一篇我会把角度切开，专门讨论另一个问题：为什么今天大家把这整圈东西集中叫作 `harness`，这个词到底该怎么拆、什么时候有解释力、什么时候会因为过宽而失效，以及为什么在讨论具体工程问题时，把它继续拆成 `工程 harness + 产品 harness + 用户友好 harness` 往往比停在 `agent = model + harness` 更重要。

> 本文经历了多轮修订重写，发布日期和实际完整日期不同步。

## 参考资料

- Anthropic Docs, [How Claude Code works](https://code.claude.com/docs/en/how-claude-code-works)
- Anthropic Docs, [Explore the context window](https://code.claude.com/docs/en/context-window)
- Anthropic Docs, [Store instructions and memories](https://code.claude.com/docs/en/memory)
- Anthropic Docs, [Connect Claude Code to tools via MCP](https://code.claude.com/docs/en/mcp)
- Anthropic Docs, [Hooks reference](https://code.claude.com/docs/en/hooks)
- Anthropic Docs, [Create custom subagents](https://code.claude.com/docs/en/sub-agents)
- Anthropic Docs, [Checkpointing](https://code.claude.com/docs/en/checkpointing)
- Anthropic Docs, [Choose a permission mode](https://code.claude.com/docs/en/permission-modes)
- Anthropic Docs, [Create plugins](https://code.claude.com/docs/en/plugins)
- Anthropic Docs, [Claude Code GitHub Actions](https://code.claude.com/docs/en/github-actions)
- Anthropic Docs, [Agent SDK overview](https://code.claude.com/docs/en/agent-sdk/overview)
- OpenAI, [Harness engineering: leveraging Codex in an agent-first world](https://openai.com/index/harness-engineering/)
- OpenAI, [From model to agent: Equipping the Responses API with a computer environment](https://openai.com/index/equip-responses-api-computer-environment/)
- Anthropic, [Writing effective tools for agents](https://www.anthropic.com/engineering/writing-tools-for-agents)
- Anthropic, [Harness design for long-running application development](https://www.anthropic.com/engineering/harness-design-long-running-apps)
- Anthropic, [Claude Code auto mode: a safer way to skip permissions](https://www.anthropic.com/engineering/claude-code-auto-mode)
- Anthropic, [Making Claude Code more secure and autonomous with sandboxing](https://www.anthropic.com/engineering/claude-code-sandboxing)
- Anthropic, [Effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)
- Anthropic, [Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)
- Vercel, [AGENTS.md outperforms skills in our agent evals](https://vercel.com/blog/agents-md-outperforms-skills-in-our-agent-evals)
- 知乎，[从 Claude Code 看 Harness Engineer 的设计](https://zhuanlan.zhihu.com/p/2021603278606087058?share_code=11INMrHCLcWKE&utm_psn=2025010367944730281)
- [《让 Agent 变得可行，大模型结构化输出与受限解码技术》](/blog/2026/03/01/语言模型的结构化输出/)
- [《Context is All You Need：智能体的上下文工程》](/blog/2026/03/06/agent-context-engineering/)
- [《从 MCP 到 Agent Skills：为什么 Agent 又需要一种新的上下文工程协议？》](/blog/2026/03/10/from-mcp-to-agent-skills/)
- [《AEnvironment：Agent 需要一个统一的环境层吗？》](/blog/2026/03/16/aenvironment-everything-as-environment/)
