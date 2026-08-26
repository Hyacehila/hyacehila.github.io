---
title: 'Back to the CLI: Choosing Between CLI and GUI in the Agent Era'
title_zh: 我们又回到了 CLI：Agent 时代 CLI 与 GUI
date: 2026-05-27 22:00:00 +0800
categories:
- Work & Society
- AI Engineering Workflows
tags:
- AI Coding
- Developer Tools
- Tool Use
author: Hyacehila
excerpt: Agentic coding has pushed many developers back to the terminal. The CLI is convenient today because it is light,
  scriptable, and easy for agents to drive. That does not mean the GUI is dead. This post starts from a terminal-centered
  workflow and argues that CLI belongs in the tool-composition layer, while review and orchestration will move back to GUI.
description: Agentic coding has pushed many developers back to the terminal. The CLI is convenient today because it is light,
  scriptable, and easy for agents to drive. That does not mean the GUI is dead. This post starts from a terminal-centered
  workflow and argues that CLI belongs in the tool-composition layer, while review and orchestration will move back to GUI.
excerpt_zh: Agentic Coding 让很多开发者重新打开终端。CLI 现在确实顺手：轻、能脚本化，也方便 Agent 直接接管工具。但这不代表 GUI 已经过时。本文从 worktree、Ghostty、Zellij、Neovim
  这套终端组合讲起，再看桌面端、浏览器插件和 Computer Use 为什么会重新变重要。我的结论是：CLI 更适合留在工具复用层，人这边的编排和 review 还是会回到 GUI。
permalink: /blog/2026/05/27/cli-vs-gui-agent-era/
lang: en
translation_key: 2026-05-27-cli-vs-gui-agent-era
translation_status: machine
translation_source_hash: fb3e994801467d724847692b087078afae5ffbc5080a7953541c332606108ed7
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<p>CLI was a new interactive approach in the Age of Age.</p>
<p>It's a little counterintuitive. The main direction of the tools over the last decade has been to hide complex commands in the interface: Vim, Emacs, followed by VS Code, Jetbrains, and others, who have been able to use them for their own benefit.<code>git</code> The diff panel, branch chart and conflict resolutionr are also added to the command. We spent a long time moving the writing code into the editor from the black box, and then adding plugs, debuggers and re-engineering tools.</p>
<p>But after the AgeCating, the first ones that could run, and that were good enough, were mostly grown from the terminals. Claude Code, Codex, Aider, OpenCode, Hermes, Openclaw, these codes or common Agent, were used almost exclusively by a line of commands.</p>
<p>I don't think it's backwards. It's a reasonable time to put Coding Agen in the CLI at this stage, and it'll be spread out back. But another question is not going to get around: is the CLI going right now, or should it always be the future?</p>
<p>So start with a set of terminals that I think are useful. It explains why CLI is comfortable now; I'll talk about my more important concerns later: When people move from writing codes to review and organize, should it be CLI or GUI.</p>
<p>What I'm using now is the following combination, and the responsibility is clear.</p>
<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px;margin:28px 0 34px;">
  <div style="border:1px solid #e5e7eb;border-radius:14px;padding:18px 16px;background:#fafafa;min-height:190px;display:flex;flex-direction:column;align-items:center;text-align:center;">
    <div style="height:56px;display:flex;align-items:center;justify-content:center;margin-bottom:10px;font-size:34px;line-height:1;">🌿</div>
    <strong style="display:block;min-height:30px;font-size:1.08em;line-height:1.3;">git worktree</strong>
    <p style="margin:10px 0 0;color:#555;font-size:0.95em;line-height:1.55;">Allows each Agent/ Branch/ Experiment to have a separate directory.</p>
  </div>
  <div style="border:1px solid #e5e7eb;border-radius:14px;padding:18px 16px;background:#fafafa;min-height:190px;display:flex;flex-direction:column;align-items:center;text-align:center;">
    <div style="height:56px;display:flex;align-items:center;justify-content:center;margin-bottom:10px;"><img src="https://raw.githubusercontent.com/ghostty-org/website/main/public/ghostty-logo.svg" alt="Ghostty official logo" style="width:42px;height:42px;object-fit:contain;"></div>
    <strong style="display:block;min-height:30px;font-size:1.08em;line-height:1.3;">Ghostty</strong>
    <p style="margin:10px 0 0;color:#555;font-size:0.95em;line-height:1.55;">Responsible for fonts, themes, speed of response and long-term terminal perception.</p>
  </div>
  <div style="border:1px solid #e5e7eb;border-radius:14px;padding:18px 16px;background:#fafafa;min-height:190px;display:flex;flex-direction:column;align-items:center;text-align:center;">
    <div style="height:56px;display:flex;align-items:center;justify-content:center;margin-bottom:10px;"><img src="https://zellij.dev/img/logo.png" alt="Zellij official logo" style="width:42px;height:42px;object-fit:contain;"></div>
    <strong style="display:block;min-height:30px;font-size:1.08em;line-height:1.3;">Zellij</strong>
    <p style="margin:10px 0 0;color:#555;font-size:0.95em;line-height:1.55;">The blog has been published by the Global Voices Online.</p>
  </div>
  <div style="border:1px solid #e5e7eb;border-radius:14px;padding:18px 16px;background:#fafafa;min-height:190px;display:flex;flex-direction:column;align-items:center;text-align:center;">
    <div style="height:56px;display:flex;align-items:center;justify-content:center;margin-bottom:10px;"><img src="https://neovim.io/logos/neovim-mark.svg" alt="Neovim official logo" style="width:42px;height:42px;object-fit:contain;"></div>
    <strong style="display:block;min-height:30px;font-size:1.08em;line-height:1.3;">Neovim</strong>
    <p style="margin:10px 0 0;color:#555;font-size:0.95em;line-height:1.55;">Quick editing and small range patches in terminal session.</p>
  </div>
</div>

<h2>A terminal-centred set of Agenic Coding tool chains</h2>
<p>First, tools. If you have placed Claude Code, Codex CLI or Aider in daily life, terminal experience suddenly becomes important: one Agent on a running mission, one window watching tests, one window starting local services, the other is still comparing logs, Zia Git status, temporary configuration, and possibly a window connected to a server, running a long training. The next set of solutions is not so cool, but rather, it is a "parallel task, terminal session, temporary editing, branch isolation" solution.</p>
<pre><code class="language-text">Git worktree   负责代码上下文隔离
Ghostty        负责现代终端窗口体验
Zellij         负责终端 workspace / pane / tab 管理
Neovim         负责终端内快速编辑文件
</code></pre>
<p><strong><code>git worktree</code></strong> It's Git's own ability, and I think it's particularly useful in the Agenic Coding. We often move AI on different branches at the same time, and then we put the results back together. The same warehouse can be used in different branches of the checkout directory to share the same object database, but the work area is independent of each other. A person named Agent. <code>feature/search</code> Find on top, another Agent is in <code>fix/login-test</code> The overhaul test, the master catalogue remains clean. Don't repeat it. <code>git switch</code>, and not a cline multiple warehouse. Claude Code, Codex, OpenCode is now largely embedded in the worktree support.</p>
<p><strong>Ghostty</strong> The most important thing is to look at the outermost window. Agenic Coding often means that long-term staring at stream output, logs and diffs directly affects fatigue. Of course you can continue with iTerm2, WezTerm, Windows Telminal or Kitty, Mac's self-contained terminal. But this layer deserves consideration if it is just in the process of re-establishing a set of end-centred environments. Who doesn't want to see comfort.</p>
<p><strong>Zellij</strong> A modern terminal reuser: cut windows into more than one pane, organize tab, save playout, restore session, default UI over tradition <code>tmux</code> Intuitive. Agent output is not mixed with test logs, local services are hanging in a pane for a long time, and the session can be restored without having to reset the window each time. To give every worktree an independent session is a pass-through:</p>
<p><strong>Neovim</strong> The scene of "Agent wrote 90%, you just want to change the line immediately." When Agent runs in one pane and tests in another pane, you do not have to cut back to the graphic editor to complete quick fixes and reviews in the terminal. It's more like a knife with a knife, not a whole development.</p>
<p>This combination is not mysterious. It just sorted out the fact that Agent was already at the terminal. The question is: Is this the final form?</p>
<h2>Why, Agent, we're back at the terminal.</h2>
<p>To answer CLI is not the future, you have to figure out why we're back to CLI.</p>
<p>My judgment is:<strong>CLI is the best solution in the present, but mainly because it is the least expensive in the short term, not because it is more human in nature.</strong></p>
<p>In traditional development, the editor is designed for human writing. The cursor, completion, sidebar file tree, and single file focus editorial perspectives are optimized around "one-man-on-one, line-by-line code, bug-by-bug" This design was successful in the past. But Actic Coding changed people's work: you're not the main code producer anymore, more like a few semi-automatic teammates. One Agent is performing functionality, one Agent is fixing tests, you're ready to review diff, add, interrupt the wrong direction, while local services, databases and logs are running.</p>
<p>When people move from writing to reading and organizing, the editor is a little out of step. The sidebar plugin is packed in an interface designed for single-person coding, diff for "the lines I just changed" and not for "five Agents have changed their share." So it's not surprising to return to the terminal: CLI is light, script, Agent can board directly, without having to move to the old way of working the GUI.</p>
<p>It's more like a transitional emergency: the existing GUI doesn't take advantage of it, and the terminal is adequate, fast and flexible. But the existing GUI doesn't take advantage of it, it doesn't mean it can't get through this road.</p>
<h2>CLI killed Gui?</h2>
<p>And when these judgments are mixed, they're prone to the slogans that spread in 2026. From X to YouTube to Little Red, a fashionable caliber comes out at the same time.<a href="https://cn.ai.cc/blogs/the-app-is-dead-agentic-cli-killed-gui-2026/">One of the same names.</a>：</p>
<blockquote>
<p>The App Is Dead. Agentic CLI Killed the GUI in 2026.</p>
</blockquote>
<p>The most common narratives are essentially the following:</p>
<ul>
<li>IDE border plugin is dead;</li>
<li>Desktop App is a product of the previous era;</li>
<li>The real AI engineer did everything in the terminal;</li>
<li>CLI = Polar/ Efficient/ Future, GUI = White / Inefficiency / Past.</li>
</ul>
<p>This is a good thing to repeat, because it turns tool selection into identity tags: CLI is for pioneers, and GUI is for old. But looking at the direction of the input, it's not that simple. The AI major plant does not only place CLI, but also does desktop end, browser plugin and Computer Use.</p>
<p><strong>Codex</strong> Keep the CLI while making the desktop end and browser more accessible. It's... <a href="https://developers.openai.com/codex/app/chrome-extension">Chrome Extension</a>You can run directly into your own browser, re-enact a session you already log in. Use only the prompt when accessing Salesforce, Gmail or Intranet tools <code>@Chrome</code> Leave the task to it, and the problems of login, cookies, token, double validation are saved; the local local localhost preview and validation are given to the built-in browser, with no disturbance between the two sides. Codex App is also leaning towards the ChatGPT entrance.</p>
<p><strong>Claude Code</strong> From CLI-first in May 2025, it evolved into a family barrel: CLI, multi-desk app (Mac/Windows), Web application and IDE extension. A power like fast Mode can be switched by clicking in the desktop App. It didn't give up on the CLI, but apparently it didn't put all the treasure on the CLI. The slash commands in the CLI will certainly be added, but once you cut to the desktop application, they will no longer be something that users must remember.</p>
<p>So I'd rather see the direction of the investment than the slogan. The owner can shout "Bilm CLI" and the factory's chambers invest money in places where users will stay for long.</p>
<p>And then look at the complex workflow. Most needed is the organization of workflows: one parent needs multiple submodules, with multiple cuts; iOS / Android / Server / Web multiple-end; one or more Agents, plus bug lists, knowledge deposition, sub-agent movements. If this workflow is pure, you have to open a terminal window every module, one end, one end. Three modules, five-end, are a screen full of terminals, and management tools can make you dizzy; GUI can at least fold them into projects, tasks, status panels and filters.</p>
<p>I'm not saying the terminal can't do it. It's just that when the industry starts to count,<a href="https://www.augmentcode.com/tools/best-ai-coding-agent-desktop-apps">2026 Best AI Encoding Agent Desktop Application</a>The reason for the move is very close to here: more Agent is a specialized infrastructure, the mission is longer than "minutes" to "hours" and it is better to have an interface for Agent management and code editing. The IDE borderbar can easily trap Agent in a single context and in a synchronous, editor interaction. None of these are a single end window suitable for carrying it off.</p>
<p>The utility of GUI is not to draw buttons beautifully, but rather to show complexity without being overwhelmed by complexity: multiple windows separate each demand from one another, status panels allow progress, test status, sub-agent output, bug list to be visible, stream output to a section shows a section, which can be read and asked, rather than the context is forgotten long after the screen is over. The more common the tools are, the more easy to use it. CLI can hold the early users, but it's hard to be the main entrance for everyone.</p>
<h2>So, custom GUI, can you swallow the whole CLI tool chain?</h2>
<p>Back to the beginning of the terminal tool chain.</p>
<p>Worktree + Ghostty + Zellij + Neovim, actually using four scattered CLI tools, spell out four capabilities: "Segregated context + terminal perception + workspace management + quick editing". But one of the four things, a GUI custom-made for Agent, can swallow together:</p>
<ul>
<li>Cut worktree? Without a knock, cut it by two, and even automatically build an independent workspace for every Agent;</li>
<li>See? Directly opens a diff panel instead of relying on a fragmented CLI text to return;</li>
<li>Multi-end? Multiple terminals are open as editors, and stand alone;</li>
<li>- A look? It's customised by designers, and probably more durable and consistent than a manual Ghostty.</li>
</ul>
<p>Zellij, Ghostty, Neovim solves the problem of experience, a decent GUI that is almost always covered and usually smoother.</p>
<p>There's only one problem: how many advantages do CLI have left, apart from reusing the ready-made command line tools?</p>
<p>I don't think there's much more to this about people and tools interacting. CLI's advantage is now, to a large extent, ecological: stock tools are in the command line, and it's very easy to re-use it directly. Of course it matters. It's just that the advantage is the reuse of tools, not the experience of interactions.</p>
<h2>Possible division of labour: CLI Return Tool Reuse Layer, GUI Returner Interactive</h2>
<p>Turn the tool back and the machine off. The CLI position actually appeared when Skills was discussed.</p>
<p>I'm here.<a href="/en/blog/2026/03/10/from-mcp-to-agent-skills/">From MCP to Agent Skills: Why does Agent need a new context work protocol?</a>As you have discussed, Skills is a light-capacity wrapper: a catalogue, a note, some scripts and a number of references, allowing models to read on demand. Skills is basically using CLI to carry out these orders, not MCP, in exchange for a flexible arrangement, which is valuable.</p>
<p>So, CLI will end up more like a tool reuse layer than a human primary interface. It will continue to compete with MCP for the position of "how to hand over the power to the model" as the bottom of the Agent call, script assembly, capacity seal. On this level, the composition of the CLI is a long-term advantage, and no one can easily replace it.</p>
<p>The human plane interacts with this layer, and it is more likely to return to the GUI. The GUI was originally created for the purpose of seeing the state, comparing the differences, and choosing the switch. Now people are monitoring multiple Agents in parallel, and these claims are more appropriate for the graphical interface. CLI is now in the position of human-powered interaction, more so at the top: the editor has not been redesigned for the new "people review, people to organize" approach, and CLI is just enough. Once the GUI, designed specifically for Agent, matures, there is no reason why interaction on this side of the human being should remain at the terminal.</p>
<p>They are not who kills who, but who gets what you want.</p>
<h2>Summary</h2>
<p>We did return to CLI, but returning to the terminal is not the end.</p>
<p>CLI is the most excellent solution of the Age of Age, which is fine. The editor is designed for old working methods and is not fit after people have turned to review and organize; CLI light, script, Agent can board directly and will naturally be used as a carrier in the short term. The combination of the worktree + Ghostty + Zellij + Neovim in the front is the pragmatic choice that is used today.</p>
<p>But the firm's real input, the organized workflow experience cap, and the lead that Skylls revealed, are reminding me that the long-term value of CLI is in the combination and reuse of tools, not in the main human interface; or that humans are going to cross the layer and return to the GUI that was designed for people. When a custom GUI can click two-trip worktrees, open a diff panel, line up multiple terminals and do better and more smooth than a manually fusion CLI tool chain, CLI may not be the answer for the future.</p>
<h2>References</h2>
<ul>
<li><a href="https://git-scm.com/docs/git-worktree">Git Official Document: Git-worktree</a></li>
<li><a href="https://ghostty.org/">Ghostty Official Website</a></li>
<li><a href="https://zellij.dev/">Zellij official website</a></li>
<li><a href="https://neovim.io/">Neovim official website</a></li>
<li><a href="https://developers.openai.com/codex/app/chrome-extension">OpenAI Codex Document: Chrome Extension</a></li>
<li><a href="https://www.augmentcode.com/tools/best-ai-coding-agent-desktop-apps">Augment Code：9 Best AI Coding Agent Desktop Apps in 2026</a></li>
<li><a href="https://cn.ai.cc/blogs/the-app-is-dead-agentic-cli-killed-gui-2026/">The App Is Dead: Agentic CLI Killed the GUI in 2026</a></li>
<li><a href="/en/blog/2026/03/10/from-mcp-to-agent-skills/">From MCP to Agent Skills: Why does Agent need a new context work protocol?</a></li>
<li><a href="/en/blog/2026/04/07/spec-is-not-the-new-paradigm/">Spec is not a new paradigm: Video Coding, SDD and AI-era software engineering shift</a></li>
<li><a href="/en/blog/2026/04/10/how-to-choose-the-right-model-for-developers/">Claude Code or Codex: How differences in code models translate into differences in product experiences</a></li>
<li><a href="/en/blog/2026/05/22/agent-browser-tools-comparison/">Let Agent operate the browser: from automated scripts to browser infrastructure evolution</a></li>
</ul>
