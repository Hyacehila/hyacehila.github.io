---
title: 我们又回到了 CLI：Agent 时代 CLI 与 GUI
title_en: "Back to the CLI: Choosing Between CLI and GUI in the Agent Era"
date: 2026-05-27 22:00:00 +0800
categories: ["Agent Systems"]
tags: [Software Engineering, Agents, Tool Use]
author: Hyacehila
excerpt: Agentic Coding 让很多开发者重新打开终端。CLI 现在确实顺手：轻、能脚本化，也方便 Agent 直接接管工具。但这不代表 GUI 已经过时。本文从 worktree、Ghostty、Zellij、Neovim 这套终端组合讲起，再看桌面端、浏览器插件和 Computer Use 为什么会重新变重要。我的结论是：CLI 更适合留在工具复用层，人这边的编排和 review 还是会回到 GUI。
excerpt_en: "Agentic coding has pushed many developers back to the terminal. The CLI is convenient today because it is light, scriptable, and easy for agents to drive. That does not mean the GUI is dead. This post starts from a terminal-centered workflow and argues that CLI belongs in the tool-composition layer, while review and orchestration will move back to GUI."
permalink: '/blog/2026/05/27/cli-vs-gui-agent-era/'
---

CLI 在 Agent 时代重新成了一种受关注的交互方式。

这有点反直觉。过去十几年，开发工具的大方向一直是把复杂命令藏进界面里：Vim、Emacs 之后有 VS Code、JetBrains，`git` 命令之外也有 diff 面板、分支图和冲突解决器。我们花了很长时间，把写代码从黑框里搬进编辑器，还顺手补上了插件、调试器和重构工具。

可 Agentic Coding 起来之后，第一批跑得起来、也足够好用的工具，又大多从终端长出来。Claude Code、Codex、Aider、OpenCode、Hermes、Openclaw 这些编码或通用 Agent，最早的使用方式几乎都是一行命令。

我不觉得这是倒退。现阶段把 Coding Agent 放在 CLI 里很合理，后面会展开。但另一个问题也绕不开：CLI 是眼下顺手，还是未来也应该一直这样？

所以先从一套我觉得够用的终端工具链说起。它能解释为什么 CLI 现在舒服；后半段再谈我更关心的问题：当人的工作从写代码变成 review 和编排，承载它的到底应该是 CLI 还是 GUI。

我现在常用的是下面这套组合，各自负责的事很清楚。
<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px;margin:28px 0 34px;">
  <div style="border:1px solid #e5e7eb;border-radius:14px;padding:18px 16px;background:#fafafa;min-height:190px;display:flex;flex-direction:column;align-items:center;text-align:center;">
    <div style="height:56px;display:flex;align-items:center;justify-content:center;margin-bottom:10px;font-size:34px;line-height:1;">🌿</div>
    <strong style="display:block;min-height:30px;font-size:1.08em;line-height:1.3;">git worktree</strong>
    <p style="margin:10px 0 0;color:#555;font-size:0.95em;line-height:1.55;">让每个 Agent / 分支 / 实验拥有独立目录。</p>
  </div>
  <div style="border:1px solid #e5e7eb;border-radius:14px;padding:18px 16px;background:#fafafa;min-height:190px;display:flex;flex-direction:column;align-items:center;text-align:center;">
    <div style="height:56px;display:flex;align-items:center;justify-content:center;margin-bottom:10px;"><img src="https://raw.githubusercontent.com/ghostty-org/website/main/public/ghostty-logo.svg" alt="Ghostty official logo" style="width:42px;height:42px;object-fit:contain;"></div>
    <strong style="display:block;min-height:30px;font-size:1.08em;line-height:1.3;">Ghostty</strong>
    <p style="margin:10px 0 0;color:#555;font-size:0.95em;line-height:1.55;">负责字体、主题、响应速度和长期终端观感。</p>
  </div>
  <div style="border:1px solid #e5e7eb;border-radius:14px;padding:18px 16px;background:#fafafa;min-height:190px;display:flex;flex-direction:column;align-items:center;text-align:center;">
    <div style="height:56px;display:flex;align-items:center;justify-content:center;margin-bottom:10px;"><img src="https://zellij.dev/img/logo.png" alt="Zellij official logo" style="width:42px;height:42px;object-fit:contain;"></div>
    <strong style="display:block;min-height:30px;font-size:1.08em;line-height:1.3;">Zellij</strong>
    <p style="margin:10px 0 0;color:#555;font-size:0.95em;line-height:1.55;">把 Agent、测试、服务和日志整理成 workspace。</p>
  </div>
  <div style="border:1px solid #e5e7eb;border-radius:14px;padding:18px 16px;background:#fafafa;min-height:190px;display:flex;flex-direction:column;align-items:center;text-align:center;">
    <div style="height:56px;display:flex;align-items:center;justify-content:center;margin-bottom:10px;"><img src="https://neovim.io/logos/neovim-mark.svg" alt="Neovim official logo" style="width:42px;height:42px;object-fit:contain;"></div>
    <strong style="display:block;min-height:30px;font-size:1.08em;line-height:1.3;">Neovim</strong>
    <p style="margin:10px 0 0;color:#555;font-size:0.95em;line-height:1.55;">在终端 session 里完成快速编辑和小范围修补。</p>
  </div>
</div>

## 一套以终端为中心的 Agentic Coding 工具链

先说工具。如果你已经把 Claude Code、Codex CLI 或 Aider 放进日常，终端体验会突然变得重要：一个 Agent 在跑任务，一个窗口在看测试，一个窗口在启动本地服务，另一个还在对比日志、查 Git 状态、临时改配置，可能还有窗口连着服务器，跑一个时间很长的训练。下面这套组合解决的不是酷炫，而是把「并行任务、终端会话、临时编辑、分支隔离」管住。

```text
Git worktree   负责代码上下文隔离
Ghostty        负责现代终端窗口体验
Zellij         负责终端 workspace / pane / tab 管理
Neovim         负责终端内快速编辑文件
```

**`git worktree`** 是 Git 自带的能力，我觉得在 Agentic Coding 里尤其有用。我们经常会同时在不同分支上调度 AI 工作，再把结果合回来。同一个仓库可以在不同目录下 checkout 不同分支，共享同一份对象数据库，但工作区彼此独立。一个 Agent 在 `feature/search` 上实现搜索，另一个 Agent 在 `fix/login-test` 上修测试，主目录仍然保持干净。不用反复 `git switch`，也不用 clone 多份仓库。Claude Code、Codex、OpenCode 现在基本都内置了 worktree 支持。

**Ghostty** 负责最外层的窗口观感。Agentic Coding 往往意味着长时间盯着流式输出、日志和 diff，字体渲染、滚动、主题、快捷键都会直接影响疲劳感。你当然可以继续用 iTerm2、WezTerm、Windows Terminal 或 Kitty，Mac 自带终端也够用。但如果正好在重整一套以终端为中心的环境，这一层值得考虑。谁不想看得舒服一点呢。

**Zellij** 是个现代化的终端复用器：把窗口切成多个 pane、组织 tab、保存 layout、恢复 session，默认 UI 比传统 `tmux` 直观。Agent 输出不和测试日志混在一起，本地服务长期挂在一个 pane 里，session 可以恢复，不必每次重新摆窗口。给每个 worktree 一个独立 session 是个顺手的习惯：

**Neovim** 解决「Agent 写了 90%，你只想立刻改一行」的场景。当 Agent 在一个 pane 里跑、测试在另一个 pane 里输出，你不必切回图形编辑器，终端内就能完成快速修补和 review。它更像一把随身小刀，不负责整场开发。

这套组合不神秘。它只是把 Agent 已经待在终端这件事整理得没那么乱。问题还在后面：这会不会就是最终形态？

## 为什么 Agent 时代我们又回到了终端

要回答 CLI 是不是未来，得先想清楚我们为什么会回到 CLI。

我的判断是：**CLI 是当下的最优解，但主要因为它短期最省事，而不是因为它本质上更适合人。**

传统开发里，编辑器是为人写代码设计的。光标、补全、边栏文件树、单文件聚焦的编辑视角，都是围绕「一个人逐行敲代码、逐处改 bug」来优化的。这套设计在过去很成功。但 Agentic Coding 改变了人的工作内容：你不再是主要的代码生产者，更像是在调度几个半自动队友。一个 Agent 在实现功能，一个 Agent 在修测试，你自己随时 review diff、补充说明、打断错误方向，同时本地服务、数据库和日志还在旁边跑。

当人的工作从写变成看与编排，编辑器就有点不合身了。边栏插件挤在一个为单人编码设计的界面里，diff 是给「我刚改的那几行」准备的，不是给「五个 Agent 各自改了一摊」准备的。所以退回终端并不奇怪：CLI 轻、可脚本化，Agent 可以直接寄宿，不需要先迁就一套旧工作方式留下来的 GUI。

它更像过渡期的应急方案：现有 GUI 不趁手，而终端正好够用、够快、够灵活。但现有 GUI 不趁手，不等于 GUI 这条路走不通。

## CLI 杀死 GUI？

这几个判断混在一起后，就容易出现 2026 年那波很会传播的口号。从 X 到 YouTube 到小红书，一种时髦口径同时冒出来，最有代表性的是[一篇同名檄文](https://cn.ai.cc/blogs/the-app-is-dead-agentic-cli-killed-gui-2026/)：

> The App Is Dead. Agentic CLI Killed the GUI in 2026.

常见的叙述内容基本有以下几点：

- IDE 边栏插件已死；
- 桌面 App 是上一个时代的产物；
- 真正的 AI 工程师都在终端里完成所有事情；
- CLI = 极客 / 高效 / 未来，GUI = 小白 / 低效 / 过去。

这种说法好转发，因为它把工具选择变成身份标签：用 CLI 的是先锋，用 GUI 的是落伍。但看投入方向，事情没那么简单。AI 大厂并没有只押 CLI，它们同样在做桌面端、浏览器插件和 Computer Use。

**Codex** 一边保留 CLI，一边把桌面端和浏览器接入做得更重。它的 [Chrome 扩展](https://developers.openai.com/codex/app/chrome-extension)能直接跑在你自己的浏览器 profile 里，复用你已经登录好的会话。访问 Salesforce、Gmail 或内网工具时，只需在 prompt 里用 `@Chrome` 把任务交给它，登录、cookie、token、二次验证那堆麻烦都省掉了；本地 localhost 的预览与验证则交给内置浏览器，两边互不打扰。Codex App 也在往 ChatGPT 入口里靠。

**Claude Code** 从 2025 年 5 月的 CLI-first，演变成了一个全家桶：CLI、多桌面 App（Mac/Windows）、Web 应用和 IDE 扩展。像 fast mode 这类能力，在桌面 App 里点一下就能切换。它没有放弃 CLI，但也显然没把宝全押在 CLI 上。CLI 里的斜杠命令当然还会增加，可一旦切到桌面应用，这些命令就不再是用户必须记住的东西。

所以我更愿意看投入方向，而不是口号。博主可以喊 CLI 万岁，厂商会把钱投到用户会长期停留的地方。

再看复杂工作流。最需要 GUI 的，是编排型工作流：一个父需求挂多个子模块，多人各切一片；iOS / Android / Server / Web 多端并发；每端再挂一个或多个 Agent，外加 bug 列表、知识沉淀、子代理调度。这种工作流如果纯 CLI，你每开一个模块、一个端、一个 Agent 都得开一个终端窗口。三个模块乘五个端就是一屏铺满终端，有管理工具也会让人头晕；GUI 至少可以把它们折叠成项目、任务、状态面板和筛选器。

我不是说终端不能做。只是当业界开始盘点[「2026 最好的 AI 编码 Agent 桌面应用」](https://www.augmentcode.com/tools/best-ai-coding-agent-desktop-apps)时，迁移理由和这里很接近：多 Agent 编排需要专门的基础设施，任务时长从「分钟」拉长到「小时」甚至更久，Agent 管理和代码编辑最好各用一套界面。IDE 边栏则很容易把 Agent 困在单一上下文和同步、编辑器内的交互里。这些都不是一个终端窗口适合独自扛下来的。

GUI 的用处也不是把按钮画漂亮，而是让人看见复杂度，又不被复杂度淹没：多窗口隔离让每个需求互不串线，状态面板让进度、测试状态、子代理输出、bug 列表一屏可见，流式输出推一段显示一段，可以边读边追问，而不是滚屏看完早就忘了上下文。工具越常用，易用性就越要命。CLI 能撑住早期用户，但很难成为所有人的主入口。

## 那么，定制 GUI 能不能吞下整套 CLI 工具链？

回到开头那套终端工具链。

worktree + Ghostty + Zellij + Neovim，其实是在用四件分散的 CLI 工具，拼出「上下文隔离 + 终端观感 + workspace 管理 + 快速编辑」四种能力。但这四件事，一个为 Agent 定制的 GUI 也可以一并吞下：

- 切 worktree？不用敲命令，点两下就切，甚至自动为每个 Agent 建好独立工作区；
- 看 diff？直接开一个 diff 面板，而不是依赖零散的 CLI 文本返回；
- 多终端？像编辑器那样开多个终端面板，状态各自独立；
- 观感？交给设计师定制，大概率比手动美化过的 Ghostty 更耐看、更一致。

Zellij、Ghostty、Neovim 解决的体验问题，一个像样的 GUI 几乎都能覆盖，而且通常会更顺。

问题就剩一个：CLI 除了能复用现成命令行工具，还剩多少优势？

在人和工具交互这件事上，我不觉得还有多少。CLI 现在的优势，很大程度上是生态优势：存量工具都在命令行里，Agent 直接复用很省事。这当然重要。只是这个优势属于工具复用，不属于交互体验。

## 可能的分工：CLI 回归工具复用层，GUI 回归人机交互

把工具复用和人机交互拆开看，CLI 的位置其实在之前讨论 Skills 的时候就出现过。

我在[《从 MCP 到 Agent Skills：为什么 Agent 又需要一种新的上下文工程协议？》](/blog/2026/03/10/from-mcp-to-agent-skills/)里聊过，Skills 是一种很轻的能力封装：一个目录、一份说明、几段脚本和若干参考资料，让模型按需读取。Skills 基本上就是在使用 CLI 来执行这些命令，而非 MCP，从而换取一个灵活编排，他很有存在的价值。

因此，CLI 最后会更像工具复用层，而不是人的主界面。它会继续和 MCP 竞争「能力怎么交给模型」这个位置，作为 Agent 调用、脚本拼装、能力封装的底座存在。在这一层，CLI 的可组合性是长期优势，没有谁能轻易替代。

人机交互这一层，则更可能回到 GUI。GUI 本来就是为人看状态、对比差异、点选切换而生的。现在人还要并行监控多个 Agent，这些诉求更适合图形界面。CLI 目前占着人机交互的位置，更多是先顶上来：编辑器还没为「人 review、人编排」这套新工作方式重新设计，而 CLI 恰好够用。一旦专门为 Agent 编排设计的 GUI 成熟起来，人这一侧的交互没有理由一直停在终端。

两者不是谁杀死谁，而是各归各位。

## 小结

我们确实又回到了 CLI，但回到终端不等于终点就是终端。

CLI 是 Agent 时代当下的最优解，这没问题。编辑器为旧的工作方式而设计，在人转向 review 和编排之后有点不合身；CLI 轻、可脚本化，Agent 可以直接寄宿，短期内自然会被拿来当承载层。前面那套 worktree + Ghostty + Zellij + Neovim 的组合，正是这个阶段里很务实的选择，今天用它不亏。

但厂商的真实投入、编排型工作流的体验上限，还有 Skills 暴露出来的那条线索，都在提醒我：CLI 的长期价值在工具组合与复用，不在人的主交互入口；人机交互这一层，还是会回到为人设计的 GUI。当一个定制 GUI 能点两下切 worktree、开 diff 面板、并排多个终端，并且做得比手动拼起来的 CLI 工具链更好看、更顺手时，CLI 未必还是未来的答案。

## 参考资料

- [Git 官方文档：git-worktree](https://git-scm.com/docs/git-worktree)
- [Ghostty 官方网站](https://ghostty.org/)
- [Zellij 官方网站](https://zellij.dev/)
- [Neovim 官方网站](https://neovim.io/)
- [OpenAI Codex 文档：Chrome 扩展](https://developers.openai.com/codex/app/chrome-extension)
- [Augment Code：9 Best AI Coding Agent Desktop Apps in 2026](https://www.augmentcode.com/tools/best-ai-coding-agent-desktop-apps)
- [The App Is Dead: Agentic CLI Killed the GUI in 2026](https://cn.ai.cc/blogs/the-app-is-dead-agentic-cli-killed-gui-2026/)
- [《从 MCP 到 Agent Skills：为什么 Agent 又需要一种新的上下文工程协议？》](/blog/2026/03/10/from-mcp-to-agent-skills/)
- [《Spec 不是新范式：Vibe Coding、SDD 与 AI 时代的软件工程转向》](/blog/2026/04/07/spec-is-not-the-new-paradigm/)
- [《Claude Code or Codex：编码模型差异如何变成产品体验的不同》](/blog/2026/04/10/how-to-choose-the-right-model-for-developers/)
- [《让 Agent 操作浏览器：从自动化脚本到浏览器基础设施的演进》](/blog/2026/05/22/agent-browser-tools-comparison/)
