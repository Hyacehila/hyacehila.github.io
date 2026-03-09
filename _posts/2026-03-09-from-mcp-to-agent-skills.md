---
layout: blog-post
title: "从 MCP 到 Agent Skills：为什么 Agent 又需要一种新的能力封装？"
date: 2026-03-10 20:00:00 +0800
categories: [LLM]
tags: [Agent, MCP, Skills, Context Engineering, Tooling]
author: Hyacehila
excerpt: "Agent Skills 的爆火不是因为它比 MCP 更先进，而是因为它把能力封装做得足够简单。它是上下文工程中重要的一环，但不是新的统一协议，更不是 Agent 的终局。"
series: "Agent时代的基础设施"
featured: false
math: false
---

# 从 MCP 到 Agent Skills：为什么 Agent 又需要一种新的能力封装？

Agent Skills 的爆火，不是因为它比 MCP 更先进，而是因为它比 MCP 更容易被用起来。

这句话听起来像是在故意贬低 Skills，但我真正想说的是：**Skills 的价值非常真实，只是它的价值主要不在协议创新，而在工程压缩。** 它把一件原本需要 Host、Client、Server、Schema、权限模型与安装链路共同完成的事情，压缩成了一个目录、一份 `SKILL.md`、几段脚本和若干参考资料。它没有发明一个全新的世界，却精准踩中了 Agent 开发里最痛的那个点：**能力应该如何被交给模型。**

MCP 在 2024 年 11 月 25 日公开之后，很多人以为 Agent 世界终于有了统一接口，后面的事情无非是补生态、补客户端、补分发。

但现实并没有这么线性。

2025 年 10 月 16 日，Anthropic 把 Agent Skills 推到台前；2025 年 12 月 18 日，Skills 变成开放标准；同一天 GitHub Copilot 宣布支持 Agent Skills；2026 年 2 月 2 日，OpenAI 在介绍 Codex app 时也公开提到 skills 机制。这个时间线本身已经说明了一件事：**即便协议已经存在，开发者仍然在寻找更轻、更软、更少约束的能力封装。**

所以这篇文章不是一篇“Agent Skills 简介”，而是一篇判断：**Skills 是上下文工程里重要且有价值的一环，但它相对 MCP 不是跨越式进步，而是模型能力上升之后，对能力封装方式的一次轻量修补。它之所以会火，也不只是因为它足够好，而是因为它足够简单。**

## 引子：Skills 的爆火，不是因为它比 MCP 更先进

技术世界最容易犯的一个错误，就是把“流行”误写成“更先进”。

MCP 比 Skills 更完整，这几乎没有什么争议。它从一开始就不是单纯的 Tool Calling 协议，而是一个真正意义上的上下文协议：工具、资源、提示模板、生命周期、能力协商、客户端权限边界、后续的 Registry 与 `.mcpb` bundle，都说明它想解决的是 Agent 与外部世界之间的接口标准化问题。

而 Skills 根本不是沿着这条路线长出来的。它没有试图统一 transport，没有试图解决多端一致权限，也没有试图把所有能力都压进一套严密的协议原语里。它做的事情非常朴素：**把 Prompt、脚本、局部资料和工作流说明捆成一个能力包，让模型按需读取。**

这也是为什么我不认为 Skills 是“MCP 的下一代”。它更像是一条工程旁路：当协议太完整、使用太重、心智负担太高时，开发者会自然地回到更接近文件系统、更接近脚本、更接近说明书的那种封装方式。

这并不丢人。恰恰相反，这说明 Agent 开发的核心矛盾并不在于我们能不能设计出更优雅的协议，而在于我们能不能让能力被真正交付给模型、被团队真正维护、被项目真正复用。

## Agent 为什么总在寻找更轻的能力封装

从 Tool Call 到 MCP，再到 Skills，表面上看是在换技术名词，本质上讨论的是同一个问题：**能力应该以什么形式暴露给模型。**

最早的 Tool Call 思路非常干净。给模型一个工具名、一个工具描述、一个参数 Schema，它就知道何时调用、如何调用。这套范式的好处是明确、稳定、容易审计，坏处也同样明显：它特别适合函数，不特别适合流程。

但 Agent 世界里，很多重要能力恰恰不是函数，而是套路。

比如“写一份市场分析报告”“阅读仓库后输出技术方案”“按品牌语气生成一篇博客”“检查一个项目是否适合上线”，这些东西不是几个参数就能说清楚的。它们通常包含：

- 输入应该如何被理解
- 哪些资料需要先读
- 哪些步骤必须按顺序执行
- 哪些脚本值得直接运行
- 输出应该符合什么格式

这类能力如果强行压成 Tool Schema，会显得生硬；如果纯靠 Prompt，又会因为太长、太脆弱、太难维护而迅速失控。于是中间地带自然出现了：**不如把能力写成一个可以被读取、被展开、被分发的包。**

Skills 之所以成立，不是因为它突然发现了一个全新的技术方向，而是因为它刚好长在这个中间地带上。它比 Tool Call 更柔软，比 MCP Server 更轻，比大段系统提示更稳定。对今天的大多数 Agent 项目来说，这个生态位本来就存在，只是迟早会有人把它包装出来。

## Skills 到底是什么：Prompt、脚本与局部记忆的联合打包

如果只从文件结构看，Skills 甚至显得有些过于普通：

```text
market-research/
├── SKILL.md
├── scripts/
│   └── analyze.py
├── references/
│   └── checklist.md
└── assets/
    └── report-template.md
```

这套结构没有什么神秘之处，但它非常符合 Agent 的使用直觉。一个 Skill 里最重要的不是脚本本身，而是**它把能力的不同层级摆在了不同位置上**。

- `SKILL.md` 负责告诉模型“我是什么”“我适合在什么问题里被调用”
- `references/` 负责承载那些不值得默认注入、但在特定场景下必须读取的知识
- `scripts/` 负责把重复、脆弱、易错的步骤从 token 推理里剥离出来
- `assets/` 负责提供模板、样例与静态资源

于是，一个 Skill 实际上同时承担了三种角色：

1. **它是 Prompt 的入口**：告诉模型这项能力存在。
2. **它是局部记忆的容器**：让知识按需暴露，而不是一次性塞满上下文。
3. **它是工作流的执行器**：把部分步骤交给脚本，而不是继续让模型临场发挥。

这也是为什么我认为 Skills 不是简单的“提示词文件化”。它真正做的是把 Prompt、局部记忆与脚本执行放在同一个封装面里。这个设计谈不上宏大，但非常接近真实开发。

为了把这几个概念压缩得更清楚，可以用一张表来理解它们的分工：

| 机制 | 主要封装对象 | 最大优点 | 主要问题 |
| --- | --- | --- | --- |
| Tool Call | 动作接口 | 边界清晰、参数明确、易审计 | 不擅长流程和组织经验 |
| MCP | 工具、资源、提示与协议生命周期 | 结构完整、适合互操作 | 使用链路重、客户端体验参差 |
| Skills | Prompt、脚本、资料、模板 | 轻量、易分发、贴近项目工作流 | 标准弱、边界模糊、维护债务高 |
| Memory | 跨步骤或跨会话状态 | 处理保留、回忆与压缩 | 不负责定义能力本身 |

这张表里最值得注意的一点是：**Skills 解决的不是一切，而是“能力如何进入上下文”这个局部问题。** 这就是它重要的原因，也是它有限的原因。

## 为什么是现在：模型变强以后，不确定性的工具终于能被使用

如果把时间拨回更早，Skills 这种东西未必会这么成立。

因为它天生带着一种不确定性：模型拿到的不是一个严格的 JSON Schema，也不是一个完全标准化的远程服务，而是一组说明书、参考资料和脚本。它需要自己判断该不该用、先读什么、接着读什么、什么时候运行脚本、什么时候只保留模板不用执行。

这其实是非常高的要求。

Schema-based tool calling 的本质，是把不确定性消灭在接口层；Skills 的本质，则是接受一部分不确定性，并把这部分不确定性重新交还给模型能力。这也是为什么我认为 Skills 的成立条件，根本不只是 `SKILL.md` 这个格式，而是**模型已经强到可以消化这种半结构化能力入口了。**

换句话说，Skills 不是凭空出现的设计创新，而是模型能力提升之后的一种“自然副产物”。当模型已经足够擅长读文档、读脚本、理解工作流、按描述路由能力时，开发者当然会倾向于用更松、更软的封装方式来交付能力。因为这时候，严格 Schema 的边际收益开始下降，而轻量封装的工程收益开始上升。

这也是 Skills 和早期 Tool Use 研究之间最有趣的关系。过去的工具调用研究一直在想办法让模型学会在海量 API 中找对工具；今天，前沿模型已经足够强，问题逐渐从“如何训练模型会用工具”迁移成“如何让开发者更低摩擦地把能力交给模型”。Skills 恰好踩在这条迁移线上。

## MCP 的未竟事业：协议早就写在那里，体验却没有长出来

如果只看概念，Skills 并没有超出 MCP 太多。

MCP 一开始就不只是 tools。 resources、prompts、后来的 instructions，都说明它本来就想把模型所需的外部上下文系统化。甚至从理念上看，MCP 比今天很多人理解的还要更大：它讨论的不是“怎样给模型加函数”，而是“怎样把模型需要的外部能力、外部知识与交互边界统一组织起来”。

问题不在协议想得不够多，而在于协议写在那里，不等于体验就自动长出来。

MCP 的抽象太完整了。完整意味着强大，也意味着更高的接入负担。开发者要处理 server、client、安装、权限、生命周期、能力暴露、客户端支持矩阵，还要面对不同产品对 prompts、resources、instructions 的实际呈现并不一致。**一个协议可以在设计上非常先进，但只要它没有长出足够低摩擦的产品体验，它就不会自动成为开发者的第一选择。**

这正是 Skills 命中的地方。它没有试图回答所有问题，只回答了那个最眼前的问题：**我怎么把这套能力塞给 Agent，让它今天就能用。**

所以我会把 Skills 理解成 MCP 的一个旁路补丁，而不是替代方案。它不是在协议层赢了，而是在体验层赢了。更准确一点说，它是在 MCP 还没把“轻量封装 + 轻量分发 + 轻量激活”做到极致之前，先把开发者的耐心赢走了。

这也是为什么 MCP 后来继续补 Registry、补 `.mcpb`、补 server instructions。因为协议世界终究也意识到了：**接口定义不是全部，分发、安装、心智负担和默认体验同样是协议的一部分。**

## Skills 为什么会赢得开发者

如果只从纯技术完备性看，Skills 不应该这么火；但如果从真实开发流程看，它火得非常合理。

### 它更适合团队知识

团队最常沉淀下来的，从来不只是 API，而是规则、模板、风格和流程。怎么写报告、怎么做代码审查、怎么整理一次调研、怎么把输出对齐到组织语气，这些东西本来就更像一个 Skill，而不是一个 Tool。

### 它更适合项目本地工作流

很多能力根本不值得被做成一个独立服务。一个 repo 里的几个脚本、一份检查清单、一套产出模板，本来就和项目代码、项目规范、项目上下文绑在一起。Skills 把这些东西原地打包，反而更符合工程直觉。

### 它足够宽松

这是我认为最重要的一点。Skills 的火，不只是因为它好，而是因为它**要求得少**。它不要求开发者先理解一整套严格协议，不要求每个能力都写成规范化接口，也不要求所有宿主行为先达成一致。正因为它宽松，所以它容易被采用；正因为它容易被采用，所以它迅速扩散。

标准化世界总有一个悖论：越标准，往往越重；越轻，往往越不标准。**Skills 眼下明显站在“先让人用起来”这一边。**

## 简单的代价：Skills 的标准不足、安全风险与维护债务

但简单从来不是免费的。

我对 Skills 最大的保留，不在于它有没有价值，而在于它把大量原本应该由接口、协议和宿主共同承担的约束，重新推回给了模型理解能力和团队维护纪律。

### 模型负担更重

Tool Call 的优势在于明确：名称、描述、参数，一个模型即便不那么聪明，也比较容易走对路。Skills 则不同。模型要从 `SKILL.md` 里理解何时启用能力，要决定是否继续读 `references/`，还要判断脚本是不是该运行。这当然更灵活，但也显然更依赖模型质量。

简单说，**Skills 省掉的不是复杂度，而是接口层的复杂度；被省掉的那部分复杂度，最后会转移到模型理解层。**

### 安全边界更模糊

MCP 不是绝对安全，但它的风险暴露面比较明确：你知道 server 暴露了什么工具，知道客户端的权限确认在哪一层发生，也知道哪些调用是协议的一部分。Skills 的风险则更容易藏进普通文件和脚本里。一个能力包同时携带 instructions、references、scripts 时，它已经很接近供应链问题，而不仅仅是 Prompt 问题。

这也是为什么项目级 Skills 在工程上必须谨慎。它们看起来像是“知识包”，实际上往往带着执行权。

### 可移植性比想象中弱

开放标准不代表统一运行时。Claude、Copilot、Codex 说自己支持 Skills，并不意味着它们在目录扫描、权限控制、网络访问、依赖安装、脚本环境和 UI 呈现上完全一致。Skill 可以被传播，不代表 Skill 可以被无损执行。

换句话说，Skills 眼下更像**可迁移的封装格式**，还不是严格意义上的统一接口标准。

### 它很容易变成新的文档工程

最早你以为自己只是写了一份 `SKILL.md`，后来你会发现自己在维护一整套信息架构：描述不能冲突、引用不能失效、脚本不能漂移、模板不能过期、不同模型上的触发效果还可能不同。Skill 越多，这个问题越明显。

2026 年 2 月有一篇针对公开 Claude Skills 的实证研究，统计了 40,285 个公开 skills，结论之一就是生态里存在明显的冗余和非平凡安全风险。这件事并不让我惊讶，因为只要一种能力封装足够轻、足够宽松，它就一定会先经历一次野蛮生长，然后才开始补治理。

所以我对 Skills 的判断一直都不是“危险，不要用”，而是：**它值得用，但必须带着工程警惕去用。**

## Skills 在 Agent 版图中的位置：它是上下文工程的一环，不是终局

我觉得讨论 Skills 最容易走偏的地方，就是把它写成一个足以代表 Agent 未来的大概念。

它不是。

Skills 解决的是能力如何进入上下文，解决的是能力封装与能力分发，解决的是“把什么知识、什么流程、什么脚本交给模型”。它非常重要，但它只负责加载，不负责全部上下文治理。

这也是为什么 Skills 出现以后，Memory、Context Editing、Subagent 这些概念并没有消失，反而越来越重要。

- **Memory** 处理的是保留、压缩和回忆，不是定义能力。
- **Subagent** 处理的是上下文隔离和任务拆分，不是封装能力本身。
- **A2A** 处理的是 Agent 与 Agent 的互操作，不是单个运行时内部的能力打包。

尤其是 A2A，必须顺手澄清一下。A2A 讨论的是远程 agent 如何通信、交换任务、管理生命周期，它解决的是跨主体协作问题；Skills 讨论的是一个 agent 在本地或单运行时里，如何拥有一组可按需展开的能力包。两者的抽象层级根本不同，不应该放在一个层面上比较谁替代谁。

从这个角度看，Skills 的准确位置其实很明确：**它是 Context Engineering 的一层，是 Agent 能力暴露与能力加载的一层。** 它很重要，因为上下文管理本来就重要；但它不是终局，因为上下文管理从来不只有加载。

真正困难的问题始终在后面：

- 什么时候应该忘记已经读过的资料
- 什么时候只保留结论，不保留试错过程
- 什么时候应该把复杂任务拆给 subagent，而不是继续污染主上下文

这些问题，Skills 本身回答不了。

## 结语：这项技术值得用，但不值得神化

如果一定要用一句话给 Skills 下定义，我会这样说：

**它不是新的统一协议，而是模型能力提升之后，一种非常有效的能力打包技巧。**

这个评价听起来不够宏大，但我认为它反而更接近现实。Skills 的确重要，它让 Agent 真正开始拥有“项目私有能力”“团队私有流程”“组织私有经验”这些以前很难优雅暴露的东西。它比纯 Prompt 稳，比纯 Tool Call 柔软，比自建 MCP Server 省事。光凭这一点，它就已经值得被认真对待。

但我不愿意把它写成一场技术飞跃。因为它不是。

它没有重新定义协议边界，没有自动统一权限模型，没有天然解决长期记忆，也没有消灭上下文治理的复杂度。它做得最对的一件事，是承认了一件很多工程师早就隐约感觉到的事实：**在 Agent 时代，标准不一定先赢，简单往往先赢。**

所以我的结论非常明确：

- **Skills 值得采用。** 尤其当你要封装团队知识、本地流程和轻量自动化时，它几乎是天然合适的。
- **Skills 不值得神化。** 它不是 MCP 的全面升级，也不是 Agent 架构的终局。
- **Skills 最终会被放回更大的上下文工程框架里理解。** 它是其中一环，而且是重要的一环，但它终究只是一环。

对我来说，这恰恰是它最迷人的地方。它不伟大，但它有用；它不彻底，但它击中了现实；它不是答案本身，却逼着我们重新去问那个更重要的问题：**一个 Agent 的能力，到底应该如何被组织进上下文。**

## 参考资料

- Anthropic, [Equipping agents for the real world with Agent Skills](https://claude.com/blog/equipping-agents-for-the-real-world-with-agent-skills)
- Anthropic, [Introducing Agent Skills as an open standard](https://www.anthropic.com/news/agent-skills)
- Anthropic Docs, [Agent Skills](https://docs.claude.com/en/docs/agents-and-tools/agent-skills)
- Anthropic Claude Code Docs, [Extend Claude with skills](https://code.claude.com/docs/en/skills)
- Anthropic Claude Code Docs, [Create custom subagents](https://code.claude.com/docs/en/sub-agents)
- Anthropic, [Managing context on the Claude Developer Platform](https://claude.com/blog/context-management)
- Agent Skills, [Specification](https://agentskills.io/specification)
- GitHub Changelog, [GitHub Copilot now supports Agent Skills](https://github.blog/changelog/2025-12-18-github-copilot-now-supports-agent-skills/)
- OpenAI, [Introducing the Codex app](https://openai.com/index/introducing-the-codex-app/)
- Model Context Protocol, [Specification](https://modelcontextprotocol.io/specification/2025-06-18)
- Model Context Protocol Blog, [Introducing the MCP Registry](https://blog.modelcontextprotocol.io/posts/2025-09-08-mcp-registry-preview/)
- Model Context Protocol Blog, [Server Instructions: Giving LLMs a user manual for your server](https://blog.modelcontextprotocol.io/posts/2025-11-03-using-server-instructions/)
- Model Context Protocol Blog, [Adopting the MCP Bundle format (.mcpb) for portable local servers](https://blog.modelcontextprotocol.io/posts/2025-11-20-adopting-mcpb/)
- Google Developers Blog, [Announcing the Agent2Agent Protocol (A2A)](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/)
- arXiv, [Agent Skills: A Data-Driven Analysis of Claude Skills for Extending Large Language Model Functionality](https://arxiv.org/abs/2602.08004)
