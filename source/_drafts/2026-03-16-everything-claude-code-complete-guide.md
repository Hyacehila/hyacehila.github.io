---
title: "Everything Claude Code 中文上手与实现拆解：它到底给 Claude Code 加了什么？"
date: 2026-03-16 20:00:00 +0800
categories: [智能体系统]
tags: [Claude Code, Everything Claude Code, Codex, Agent, Hooks, Rules, Skills, MCP]
author: Hyacehila
excerpt: "以 ECC 官方中文 README 与两份官方教程为线索，拆开 commands、agents、rules、hooks、scripts 与 Codex 迁移路径，解释它到底怎样把 Claude Code 变成长期系统。"
---

如果你第一次看到 Everything Claude Code（后文简称 **ECC**），最容易产生两个误解：

- 它是不是一个新的 agent 产品？
- 它是不是在“替代” Claude Code？

都不是。

**ECC 本质上是一套围绕 Claude Code 原生扩展点搭起来的 repo。** 它没有改 Claude Code 内核，而是把 Claude Code 已经给出的 command、agent、skill、rule、hook、plugin、MCP 这些接口几乎全都填满，再把它们串成一套长期运行的工作系统。

所以如果你对 Claude Code 本体还不熟，直接看 ECC 很容易产生一种错觉：它好像什么都会，像是“另一个更强的 Claude Code”。但你真正需要建立的是下面这条理解链：

```text
Claude Code 本体
-> 提供原生扩展点（commands / skills / subagents / hooks / plugins / MCP / memory）
-> ECC 把这些扩展点系统化，做成一个可安装、可复用、可迁移的工程化配置仓库
-> 后续之所以能迁移到 Codex，是因为 ECC 不是改内核，而是在组织这些能力
```

这篇文章就按这个顺序讲。

- 先补你理解 ECC 所必需的 Claude Code 基础；
- 再用 ECC 官方中文 README 和两份官方教程解释“作者到底想做什么”；
- 再拆开 repo 本身，看它哪些是说明、哪些是真正执行的实现；
- 最后再单独讲：如果以后迁移到 Codex，哪些能直接带走，哪些不能照搬。

> 说明：本文所有动态状态都以 **2026-03-19** 的公开资料为准。ECC 的中文 README 适合入门，但它的数字快照已经落后于主 README 和当前仓库实现，后文会专门说明。

## 1. 先补 Claude Code 必要基础：不懂这些，你很难看懂 ECC

ECC 是“挂”在 Claude Code 上的，所以先要知道 Claude Code 原生到底给了哪些接口。

### 1.1 Claude Code 的几个入口

Anthropic 官方把 [Claude Code](https://code.claude.com/docs/en/overview) 定义为一个 *agentic coding tool*：它能读代码库、改文件、运行命令并接入工具。它不是普通聊天框，而是一个带文件系统、终端、工具调用和长期上下文机制的 coding agent。

最先要记住的几个入口是：

| 入口 | 作用 | 你什么时候会用到 |
| --- | --- | --- |
| `claude` | 打开交互式主会话 | 日常开发主入口 |
| `claude -p` | print mode，非交互输出 | 脚本、流水线、批处理 |
| `claude -c` | 继续当前目录最近一次会话 | 接着昨天的工作继续做 |
| `claude -r <name-or-id>` | 恢复指定会话 | 你有多个 session 时 |

按照 Anthropic 的 [CLI reference](https://code.claude.com/docs/en/cli-reference)，`claude -p`、`claude -c`、`claude -r` 都是正式支持的工作方式。也就是说，Claude Code 天然就是 **session-based** 的，这一点跟后面 ECC 为什么重视 memory、session summary、context compaction 直接相关。

### 1.2 Claude Code 常用内置命令

如果你对 Claude Code 本身不熟，先记这几个：

| 命令 | 作用 |
| --- | --- |
| `/plan` | 先进入 plan mode，再动手 |
| `/memory` | 查看和编辑 `CLAUDE.md`、rules、auto memory |
| `/plugin` | 管理插件 |
| `/mcp` | 管理 MCP 连接与认证 |
| `/agents` | 管理 agent / subagent |
| `/compact` | 压缩当前上下文 |
| `/cost` | 查看 token 花费 |

ECC 之所以能成立，不是因为它发明了这些能力，而是因为它**把这些原生入口做成了更完整的系统**。

### 1.3 你真正要分清的是这些原生扩展点

很多人第一次接触 Claude Code，会把 `CLAUDE.md`、rules、skills、subagents、hooks、plugins、MCP 混成一团。其实它们解决的问题完全不同：

| Claude Code 原生扩展点 | 原生作用 | ECC 对应组件 | ECC 做了什么 |
| --- | --- | --- | --- |
| `CLAUDE.md` / rules | 长期说明与约束 | `rules/`、`examples/CLAUDE.md` | 把编码规范、测试标准、安全边界模块化 |
| skills | 按需加载的任务说明包 | `skills/` | 把 TDD、security review、verification、continuous learning 做成可复用技能 |
| subagents | 独立上下文的小代理 | `agents/` | 预设 planner、architect、reviewer、security reviewer 等角色 |
| slash commands | 快速入口 | `commands/` | 把常见工作流封装成 `/plan`、`/tdd`、`/code-review` 等命令 |
| hooks | 生命周期自动化 | `hooks/` + `scripts/hooks/` | 自动提醒、自动格式化、质量门、session summary、cost tracking |
| plugins | 打包与分发 | `.claude-plugin/` | 让 ECC 能作为 Claude Code 插件安装 |
| MCP | 接外部工具 | `mcp-configs/` | 给出推荐 MCP 配置与接入方式 |
| memory / compact | 长会话上下文管理 | `contexts/`、`skills/strategic-compact/`、continuous learning | 把长期上下文和会话总结做成系统策略 |

这一张表基本就是理解 ECC 的钥匙。

**一句话说透：Claude Code 给的是“原生插槽”，ECC 做的是“把这些插槽全部插满，并让它们协同工作”。**

## 2. 中文入门文档到底怎么定义 ECC？

如果你只读 ECC 的 [中文 README](https://github.com/affaan-m/everything-claude-code/blob/main/README.zh-CN.md)，你会得到一个非常直观的印象：

> 它是“来自 Anthropic 黑客马拉松获胜者的完整 Claude Code 配置集合”。

这份中文 README 的价值，不在于它最“新”，而在于它最适合第一次建立全局地图。它会先告诉你三件事：

1. **这不是单一 prompt，而是完整配置集合。**
2. **它包含 agents、skills、hooks、commands、rules、MCP。**
3. **阅读顺序应该是先看精简指南，再看详细指南。**

中文 README 里的快速开始也很清楚：

```text
/plugin marketplace add affaan-m/everything-claude-code
/plugin install everything-claude-code@everything-claude-code
```

然后它马上强调一个关键限制：

> **Claude Code 插件当前不能自动分发 `rules`，所以 rules 必须手动安装。**

这个限制不是细枝末节，而是 ECC 的一条主线：**插件可以分发 commands、agents、skills、hooks，但 rules 仍然需要单独落盘。** 所以 ECC 从第一步开始就不是“只装一个插件就结束”，而是“插件 + 规则 + 工作流”的组合。

### 2.1 中文 README 的最大价值：先把仓库结构讲出来

中文 README 非常适合作为索引，因为它会先把这个仓库按层拆开：

- `.claude-plugin/`
- `agents/`
- `skills/`
- `commands/`
- `rules/`
- `hooks/`
- `scripts/`
- `contexts/`
- `mcp-configs/`

这其实已经暗示了 ECC 的真实形态：**它不是一个“配置文件”，而是一个“多层系统仓库”。**

而且这份中文 README 还特别强调两件对新手很友好的事：

- 当前插件支持 **Windows、macOS、Linux**；
- 包管理器检测、跨平台脚本和 `/setup-pm` 这类辅助能力，都是它想降低接入门槛的一部分。

### 2.2 但要注意：中文 README 的数字是旧快照

这里有一个很重要、但很多人第一次会忽略的事实。

截至 **2026-03-19**：

- 中文 README 的 quick start 仍写着：**13 agents、43 skills、31 commands**；
- 当前主 README 则写的是：**25 agents、108 skills、57 commands**。

这说明什么？

**说明中文 README 适合建立第一印象，但不是所有动态数字的最终真相。** 真正要判断 ECC 今天到底做到哪一步，你还得结合主 README、changelog 和仓库里的实际目录一起看。

也正因为如此，这篇文章后面会把“中文 README 的教学价值”和“当前实现的真实状态”分开来写。

## 3. 两份官方教程分别在讲什么？

ECC README 会反复提醒你：**真正解释这个项目的，不是 README 一份文件，而是两份 guide。**

- [The Shorthand Guide](https://github.com/affaan-m/everything-claude-code/blob/main/the-shortform-guide.md)
- [The Longform Guide](https://github.com/affaan-m/everything-claude-code/blob/main/the-longform-guide.md)

你可以把它们理解成两层：

| 教程 | 它主要讲什么 | 你读完会得到什么 |
| --- | --- | --- |
| Shorthand Guide | skills、commands、hooks、subagents、rules、MCP、plugins、并行工作流 | 搭起 ECC 的基础设施地图 |
| Longform Guide | token economics、memory persistence、continuous learning、verification loops、parallelization | 学会怎样把这套系统真正跑顺 |

如果再用一句话概括：

- **Shorthand Guide = 搭系统**
- **Longform Guide = 跑系统**

### 3.1 Shorthand Guide 讲的是“ECC 的零件”

Shorthand Guide 里最重要的几件事，是它把 ECC 的基本积木讲清楚了：

- **Skills** 是工作流定义；
- **Commands** 是以 slash command 形式触发的 skills；
- **Hooks** 是事件触发自动化；
- **Subagents** 是被主 agent 委派出去的有限作用域代理；
- **Rules** 是长期遵循的说明；
- **MCP** 是连接外部服务的桥；
- **Plugins** 则是更易安装的打包方式。

这里最关键的一句其实是它对 commands 的定义：

> **Commands are skills executed via slash commands.**

这句话对理解 ECC 非常重要。因为它说明 `/plan`、`/tdd`、`/code-review` 这些表面上是“命令”，背后其实是在调用某套预设 workflow，而不是在运行一段神秘黑盒代码。

### 3.2 Longform Guide 讲的是“ECC 的运行哲学”

Longform Guide 的重点已经不是“有哪些组件”，而是“怎样让这些组件长期工作而不烂掉”。

它主要补的是五件事：

1. **Token optimization**：模型选择、何时 compact、如何减少上下文浪费；
2. **Memory persistence**：如何跨 session 保存、恢复和提炼上下文；
3. **Continuous learning**：把重复出现的模式提炼成可复用 skill；
4. **Verification loops / evals**：不要只生成代码，要有验证与评分回路；
5. **Parallelization**：何时用多实例、何时用 git worktree、何时扩展子 agent。

这就解释了为什么 ECC 在 README 里把自己从“完整配置集合”升级成了 **agent harness performance optimization system**：因为它真正想解决的，已经不是“怎么装几个命令”，而是“怎么让 agent 长期稳定工作”。

## 4. ECC 仓库到底实现了什么：哪些是说明，哪些是真正在执行？

这是整篇文章最重要的一段。

很多人第一次点开 ECC 仓库，会以为它主要是很多 `.md` 文件。这个印象只对了一半。

**ECC 既有“说明层”，也有“执行层”。** 你可以先看这张结构图：

```text
everything-claude-code/
├── .claude-plugin/     # 插件与市场包装层
├── agents/             # 子代理定义
├── commands/           # slash command 定义
├── skills/             # workflow 与知识层
├── rules/              # 长期约束
├── hooks/              # hook 事件配置
├── scripts/            # 真正执行的 Node 运行时与安装器
├── contexts/           # 动态上下文模板
├── mcp-configs/        # MCP 配置建议
└── .codex/             # Codex 侧配置与角色映射
```

再把“哪些是说明、哪些是真正执行”拆得更具体一点：

| 目录 / 文件 | 主要内容 | 更偏“说明”还是“执行” | 在系统里的角色 |
| --- | --- | --- | --- |
| `.claude-plugin/plugin.json` | 插件元数据 | 说明 | 让 Claude Code 把它识别成可安装插件 |
| `.claude-plugin/marketplace.json` | 市场清单 | 说明 | 让 `/plugin marketplace add` 能找到它 |
| `commands/*.md` | slash command 提示模板 | 说明 | 定义工作流入口 |
| `agents/*.md` | 子代理定义 | 说明 | 定义角色、模型、工具范围、输出格式 |
| `skills/*` | 任务说明、参考资料、脚本 | 说明为主 | 定义可复用 workflow / domain knowledge |
| `rules/*` | 长期规则 | 说明 | 提供持久约束 |
| `hooks/hooks.json` | hook 配置 | 执行调度 | 决定什么事件触发什么动作 |
| `scripts/hooks/*.js` | Node hook 逻辑 | 执行 | 真正做格式化、类型检查、总结、成本记录 |
| `scripts/install-apply.js` | 安装运行时 | 执行 | 负责跨平台安装与目标分发 |
| `.codex/config.toml` / `.codex/agents/*.toml` | Codex 配置与角色 | 执行配置 | 把 ECC 的一部分迁移到 Codex |

所以如果你问“ECC 究竟做了什么”，最准确的回答不是“它写了很多命令”，而是：

**它把 Claude Code 的说明层和执行层一起搭起来了。**

### 4.1 `.claude-plugin/`：让 ECC 先变成一个“可安装物”

ECC 能用 `/plugin marketplace add affaan-m/everything-claude-code` 安装，不是凭空来的。

- `.claude-plugin/marketplace.json` 提供市场清单；
- `.claude-plugin/plugin.json` 提供插件身份和元数据。

也就是说，**`.claude-plugin/` 做的是“包装层”**：把这个仓库从“一堆散文件”变成 Claude Code 可以识别和安装的插件。

这里很重要的一点是：plugin 本身只是“分发容器”。真正的能力还是在 `commands/`、`agents/`、`skills/`、`hooks/` 这些目录里。

### 4.2 `commands/` + `agents/`：把 prompt 工程变成稳定的工作流入口

ECC 最显眼的体验，往往是 `/plan`、`/tdd`、`/code-review` 这些命令。但它们背后的实现方式其实很“Claude Code 原生”：

- `commands/*.md` 负责定义 slash command 的行为；
- `agents/*.md` 负责定义专门的子代理；
- command 再去调用 agent 或 workflow。

最典型的例子就是 `commands/plan.md`。

这份文件明确写着：`/plan` 的任务是**重述需求、识别风险、创建实施计划，而且在用户确认之前不要动代码**。这说明它不是“万能规划 AI”，而是一个被写得非常明确的 workflow prompt。

再看 `agents/planner.md`，你会发现 planner agent 也不是抽象概念，而是一个带 frontmatter 的正式定义：

- `name: planner`
- `description`: 复杂功能和重构的规划专家
- `tools: ["Read", "Grep", "Glob"]`
- `model: opus`

这告诉你两件事：

1. **ECC 没有改 Claude Code 的 agent 机制，它是在用 Claude Code 原生的 agent 机制。**
2. **它的工作方式是先把角色写死、把边界写清、把输出格式标准化。**

所以 `/plan` 不是 ECC 自己发明了一个新系统，而是“命令 + 子代理”的组合：**用 command 作为入口，用 planner agent 提供稳定规划输出。**

### 4.3 `rules/`：把长期约束从口头提醒改成模块化规则

ECC 的 `rules/` 目录是理解它“系统感”的另一个关键。

它不是只放一个巨大的 `CLAUDE.md`，而是拆成：

- `common/`
- `typescript/`
- `python/`
- `golang/`
- `perl/`
- 以及更多语言/框架扩展

这种拆法对应的是 Claude Code 的原生 rule 思路：**长期说明不要全堆在一个文件里，而要按主题、按语言、按作用域拆开。**

而且 ECC 安装时专门要求你手动安装 rules，也暴露出一个很重要的现实：

> Claude Code 当前的 plugin 机制不能自动分发 rules。

这意味着 rules 在 ECC 里不是边角料，反而是它非常核心的一层：**它们太重要了，以至于即使插件装好了，你还得亲自把它们落盘。**

### 4.4 `hooks/hooks.json` + `scripts/hooks/*.js`：这才是 ECC 最像“系统”的地方

如果说 commands、agents、skills、rules 主要还是“说明层”，那么 ECC 真正像操作系统的一层，基本就在 hooks。

当前 `hooks/hooks.json` 里你能看到这些生命周期事件：

- `PreToolUse`
- `PreCompact`
- `SessionStart`
- `PostToolUse`
- `Stop`
- `SessionEnd`

而且这些 hook 干的不是简单提醒，而是实实在在的自动化：

- 开发命令前提醒你用 `tmux`
- `git push` 前提醒审查
- 编辑文档时告警非标准文档文件
- 编辑后建议 compact
- 编辑后自动 format
- TypeScript 编辑后自动 typecheck
- 编辑后提示 `console.log`
- session 结束时保存状态
- 提取 continuous learning 模式
- 记录 token / cost 指标

更关键的是：**这些并不是写在一个大 prompt 里的幻想能力，而是由真正的脚本执行。**

举个具体例子。

`scripts/hooks/run-with-flags.js` 这个运行时 wrapper 会做几件很工程化的事：

1. 读取 hook ID 和目标脚本；
2. 根据 `ECC_HOOK_PROFILE=minimal|standard|strict` 决定这个 hook 是否启用；
3. 支持 `ECC_DISABLED_HOOKS` 临时关闭指定 hook；
4. 自动解析 plugin root；
5. 拒绝越过 plugin root 的 path traversal；
6. 优先直接 `require()` 导出 `run()` 的 hook，避免每次都额外 spawn 一个 Node 进程。

这说明什么？

说明 ECC 的 hook 层不是“写几个 shell 一把梭”，而是有一个正式的运行时包装器。换句话说，它不只是自动化，而且在认真处理：

- 性能开销
- 安全边界
- 开关控制
- 插件路径解析
- 跨平台兼容

顺带一提，中文 README 说“所有 hooks 和 scripts 都已用 Node.js 重写”，从当前实现来看，**核心运行时确实已经 Node 化**，但配置里仍能看到少量 bash wrapper 作为兼容层。这个细节也很能说明 ECC 的工程取向：它不是追求绝对纯粹，而是优先让系统在真实环境里可用。

### 4.5 `scripts/install-apply.js`：ECC 已经不只是 Claude Code 插件，而是跨目标安装器

另一个很容易被忽略，但非常说明问题的文件是 `scripts/install-apply.js`。

如果你只从 README 看 ECC，可能会以为它的安装逻辑无非是“复制几个 rules 过去”。但这份脚本暴露了另一层现实：

- 它支持 `claude`、`cursor`、`antigravity` 等不同 target；
- 支持 `--profile`、`--modules`、`--with`、`--without`；
- 支持 `--dry-run` 和 `--json`；
- `package.json` 里还暴露了 `ecc` 和 `ecc-install` 这两个 bin entry；
- npm 包名本身已经是 `ecc-universal`，而不是只叫某个 Claude 专属名字。

这意味着 ECC 已经不仅仅是在“给 Claude Code 补配置”，而是在把自己做成一个 **跨 harness、跨安装目标的分发系统**。

这也是为什么 README 里会把它重新定位成 harness performance system，而不是 configuration pack。

### 4.6 `contexts/`、continuous learning、verification-loop：它想解决的是长期运行问题

ECC 最容易被低估的部分，不是命令，而是这些偏“方法论基础设施”的目录：

- `contexts/`
- `skills/continuous-learning/`
- `skills/continuous-learning-v2/`
- `skills/strategic-compact/`
- `skills/verification-loop/`
- `skills/eval-harness/`

这些东西共同在做一件事：

**把 Claude Code 从“一次性完成当前任务”推进到“长期会话、长期验证、长期沉淀”的系统。**

也就是说，ECC 不只是问“你今天怎么写功能”，它更在乎：

- 你的会话怎么收尾；
- 经验怎么进入下次会话；
- 什么时候该 compact；
- 什么时候该开始验证；
- 怎样把重复出现的模式提炼成 skill；
- 怎样减少 token 浪费。

这也是 Longform Guide 的真正重心。

## 5. ECC 到底怎样改造 Claude Code？

到了这里，我们终于可以回答一个核心问题：

**ECC 到底给 Claude Code 加了什么？**

最准确的回答是：**它没有改 Claude Code 内核，而是把 Claude Code 的原生扩展点填满，并串成体系。**

你可以把这个“改造”过程理解成下面六步：

| 原始 Claude Code 体验 | ECC 的改造方式 | 结果 |
| --- | --- | --- |
| 每次从空白 prompt 开始 | `commands/` + `skills/` | 常见工作流有稳定入口 |
| 主 agent 什么都自己做 | `agents/` | 规划、评审、安全、TDD 等有专门角色 |
| 规范全靠口头提醒 | `rules/` | 长期约束可以复用和共享 |
| 检查全靠你手工想起 | `hooks/` + quality gates | format、typecheck、summary、cost 进入自动化 |
| 经验留在聊天记录里 | continuous learning / instincts | 经验可沉淀、可导出、可演化 |
| 长会话越来越乱、越来越贵 | strategic compact / token optimization / eval loops | 会话更可控，验证更体系化 |

所以 ECC 的“改造”并不是黑科技，而是很工程化的：

- 把临时 prompt 变成命令和技能；
- 把临时分工变成角色代理；
- 把口头规范变成规则；
- 把手工检查变成 hooks；
- 把零散经验变成可学习、可复用的模式；
- 把上下文问题变成一套会话管理策略。

这也是为什么我更愿意把 ECC 叫做“**Claude Code 的系统化工程层**”，而不是“Claude Code 增强插件”。

## 6. 我该怎么用 ECC：从安装到第一个闭环

如果你现在已经知道 ECC 是什么，下一步最重要的不是把所有功能都打开，而是跑通第一条最小闭环。

### 6.1 推荐安装顺序

我建议你按下面这个顺序来：

#### 第一步：先装插件

```text
/plugin marketplace add affaan-m/everything-claude-code
/plugin install everything-claude-code@everything-claude-code
```

这一步的意义是：先把 commands、agents、skills、hooks 进入 Claude Code 可见范围。

#### 第二步：再装 rules

如果你用 Windows PowerShell，最直接的方式是：

```powershell
git clone https://github.com/affaan-m/everything-claude-code.git
cd everything-claude-code
npm install
.\install.ps1 typescript
```

如果你想走兼容入口，也可以用：

```bash
npx ecc-install typescript
```

这一步的意义是：把 `rules/common/` 和你当前语言的规则真正落到 Claude Code 会读取的位置。

**没有 rules，你装上的只是 ECC 的“表层入口”，不是它的完整工作方式。**

### 6.2 plugin 安装和手动安装，命令形态不同

这是新手最容易搞混的一点。

- **plugin 安装**时，命令通常是 namespaced 形式：

```text
/everything-claude-code:plan "Add user authentication"
```

- **手动安装**时，命令通常是短形式：

```text
/plan "Add user authentication"
```

这不是语法细节，而是排错时首先要查的点之一。

### 6.3 你的第一个最小闭环

如果你今天只想先把 ECC 跑起来，我建议就做这一条：

```text
/everything-claude-code:plan "Add user authentication"
/tdd
/code-review
```

这条闭环分别在干什么？

- **`/plan`**：先用 planner agent 把需求、风险和步骤讲清；
- **`/tdd`**：把实现路径切换到测试驱动；
- **`/code-review`**：用 reviewer workflow 检查质量和安全问题。

这一条链路已经能很好代表 ECC 的工作方式：

**先规划，再实现，再验证。**

### 6.4 哪些东西不要一上来就全开

ECC 功能很多，但我非常不建议新手一开始全部打开。

| 能力 | 建议时机 | 为什么 |
| --- | --- | --- |
| MCP 全家桶 | 后面再开 | 太多工具会吃掉上下文 |
| continuous learning | 先理解后再开 | 不然你不知道它学进去了什么 |
| security scan / AgentShield | 跑顺基础闭环后再开 | 它更适合进入“稳定开发期” |
| 严格 hook profile | 等你习惯默认行为后 | 否则一开始会觉得系统“太爱打断你” |
| Codex 迁移 | Claude Code 跑顺后再做 | 不要同时排两套系统 |

Shorthand Guide 和 Longform Guide 都反复强调一个主题：**上下文窗口是宝贵资源。** MCP 太多、插件太多、工具太多，都会让 Claude Code 的有效上下文下降，甚至让行为开始发散。

所以 ECC 的正确打开方式不是“能开多少开多少”，而是“先跑顺最小系统，再加重装备”。

## 7. 如果以后迁移到 Codex，该怎么理解 ECC？

这部分放在最后讲，是因为它本质上不是 ECC 的核心，而是 ECC 为什么能迁移的证明。

ECC 在当前仓库里已经自带 `.codex/`，说明作者确实在认真做 Codex 兼容，而不是只在 README 里口头提一下。

### 7.1 `.codex/` 里到底有什么

当前仓库里最关键的三个部分是：

- `.codex/config.toml`
- `.codex/AGENTS.md`
- `.codex/agents/*.toml`

它们分别扮演不同角色：

#### `.codex/config.toml`

这是 Codex 的 reference config。当前文件里你能看到：

- `approval_policy = "on-request"`
- `sandbox_mode = "workspace-write"`
- `web_search = "live"`
- 一组默认 MCP servers：GitHub、Context7、Exa、Memory、Playwright、Sequential Thinking
- `features.multi_agent = true`
- `strict` 与 `yolo` 两套 profile
- `explorer`、`reviewer`、`docs_researcher` 三个 agent role

而且它有一个非常重要的设计选择：**默认不 pin `model` 和 `model_provider`**，让 Codex 使用自己的当前默认模型，只有在你明确想锁定时才覆盖。

#### `.codex/AGENTS.md`

这份文件不是替代根目录 `AGENTS.md`，而是 **Codex supplement**。

它主要在补：

- Codex 侧推荐模型；
- skills 的发现机制；
- 多 agent 支持方式；
- 与 Claude Code 的关键差异。

尤其重要的是它明确写出了这条限制：

> **Codex 目前还没有 Claude-style hook execution parity。**

而且这份文件还明确说明，Codex 侧的 skills 会从 `.agents/skills/` 自动发现，每个 skill 由 `SKILL.md` 和 `agents/openai.yaml` 这类元数据组成。也就是说，**ECC 在 Codex 侧保留的是“技能层”和“说明层”，而不是 Claude Code 式的 hook 自动化层。**

#### `.codex/agents/*.toml`

这些是 Codex 侧的角色配置。

例如 `reviewer.toml` 里就很典型：

- `model = "gpt-5.4"`
- `model_reasoning_effort = "high"`
- `sandbox_mode = "read-only"`
- `developer_instructions` 里要求它优先关注 correctness、security、behavioral regression、missing tests

这说明 ECC 在 Codex 侧的做法不是“原样复制 Claude Code agent”，而是**把角色重新映射到 Codex 原生的 config/role 机制里。**

### 7.2 迁移到 Codex，哪些能带走，哪些不能？

最简单的看法是下面这张表：

| Claude Code + ECC | Codex + ECC | 迁移结论 |
| --- | --- | --- |
| `commands/` slash commands | 更偏 instruction / AGENTS / skill 触发 | 不能 1:1 照搬 |
| `agents/*.md` | `.codex/agents/*.toml` + `/agent` | 可以迁移思路，但实现形式不同 |
| `rules/` + `CLAUDE.md` | `AGENTS.md` + `.codex/AGENTS.md` + rules | 可以迁移原则 |
| `hooks/hooks.json` + `scripts/hooks/*.js` | 暂无 Claude-style parity | 不能直接搬，需靠 instruction + sandbox + approval 替代 |
| `skills/` | `.agents/skills/` | 可以迁移核心知识层 |
| MCP 配置 | `.codex/config.toml` / `codex mcp add` | 可以迁移，但接线方式不同 |
| quality gate / verification 思路 | reviewer、sandbox、instructions、手动流程 | 能迁移方法论，不能完全迁移自动化 |

### 7.3 真正合理的迁移策略

如果未来你要从 Claude Code 迁移到 Codex，我建议按下面这条顺序：

1. **保留根目录 `AGENTS.md` 作为共享基础层**；
2. **把 ECC 的 skills、workflow、review 规则尽量上移到跨工具层**；
3. **把 Claude Code 专属的 hooks、plugin、rules 继续留在 `.claude/`**；
4. **在 `.codex/` 里补 Codex 专属配置、approval、sandbox 和 roles**；
5. **不要试图 1:1 复刻 hooks，而是接受 Codex 目前更偏 instruction-based 的现实。**

换句话说，ECC 迁移到 Codex 的关键不是“复制配置”，而是：

**把通用方法论抽出来，把宿主特有能力留给宿主自己实现。**

这也是 ECC 之所以值得研究的一点：它本质上不是在教你“怎么堆配置”，而是在教你“怎么把 agent harness 分层”。 

## 8. 最后怎么读这套项目，才最不容易走歪？

如果你现在对 ECC 还是有点复杂感，我建议你按这个顺序读和用：

### 读法顺序

1. **先读中文 README**：建立全局地图；
2. **再读 Shorthand Guide**：理解 skills / commands / hooks / agents / rules / MCP 的关系；
3. **再读 Longform Guide**：理解 memory、token、verification、parallelization 为什么是系统问题；
4. **最后回到仓库实现本身**：看 `commands/plan.md`、`agents/planner.md`、`hooks/hooks.json`、`scripts/install-apply.js`、`.codex/config.toml`。

### 上手顺序

1. **先把 Claude Code 原生入口用熟**：`claude`、`/plan`、`/plugin`、`/memory`、`/compact`、`/cost`
2. **安装 ECC 插件**
3. **手动安装 rules**
4. **先跑通 `/plan -> /tdd -> /code-review`**
5. **再逐步引入 hooks、verification、continuous learning**
6. **最后才考虑 Codex 迁移**

如果你只想看全文的一句结论，那就是：

**ECC 最值得学的不是“它有哪些命令”，而是“它怎样把 Claude Code 的原生扩展点变成一套长期、可验证、可迁移的系统”。**

这也是为什么它比很多“提示词合集”更值得认真读一遍实现本身。

## 参考文献与官方入口

### ECC 官方入口

- [ECC 中文 README](https://github.com/affaan-m/everything-claude-code/blob/main/README.zh-CN.md)
- [ECC README](https://github.com/affaan-m/everything-claude-code)
- [The Shorthand Guide](https://github.com/affaan-m/everything-claude-code/blob/main/the-shortform-guide.md)
- [The Longform Guide](https://github.com/affaan-m/everything-claude-code/blob/main/the-longform-guide.md)
- [ECC CHANGELOG](https://github.com/affaan-m/everything-claude-code/blob/main/CHANGELOG.md)

### 文中提到的关键实现文件

- [hooks/hooks.json](https://github.com/affaan-m/everything-claude-code/blob/main/hooks/hooks.json)
- [commands/plan.md](https://github.com/affaan-m/everything-claude-code/blob/main/commands/plan.md)
- [agents/planner.md](https://github.com/affaan-m/everything-claude-code/blob/main/agents/planner.md)
- [scripts/install-apply.js](https://github.com/affaan-m/everything-claude-code/blob/main/scripts/install-apply.js)
- [.codex/AGENTS.md](https://github.com/affaan-m/everything-claude-code/blob/main/.codex/AGENTS.md)
- [.codex/config.toml](https://github.com/affaan-m/everything-claude-code/blob/main/.codex/config.toml)

### Claude Code 官方文档

- [Overview](https://code.claude.com/docs/en/overview)
- [CLI reference](https://code.claude.com/docs/en/cli-reference)
- [Built-in commands](https://code.claude.com/docs/en/commands)
- [Memory](https://code.claude.com/docs/en/memory)
- [Skills](https://code.claude.com/docs/en/skills)
- [Subagents](https://code.claude.com/docs/en/sub-agents)
- [Hooks](https://code.claude.com/docs/en/hooks)
- [Plugins](https://code.claude.com/docs/en/plugins)
- [MCP](https://code.claude.com/docs/en/mcp)

### Codex 官方文档

- [Custom instructions with AGENTS.md](https://developers.openai.com/codex/guides/agents-md)
- [Configuration Reference](https://developers.openai.com/codex/config-reference)
- [Agent approvals & security](https://developers.openai.com/codex/agent-approvals-security)
- [Rules](https://developers.openai.com/codex/rules)
