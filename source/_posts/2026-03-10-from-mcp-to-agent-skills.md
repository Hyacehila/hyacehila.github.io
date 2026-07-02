---
title: 从 MCP 到 Agent Skills：为什么 Agent 又需要一种新的上下文工程协议？
title_en: "From MCP to Agent Skills: Why Agents Need a New Context Engineering Protocol"
date: 2026-03-10 20:00:00 +0800
categories: ["AI & Agents", "Agent Infrastructure"]
tags: ["MCP", "Agent Skills", "Context Engineering", "Protocols", "Capability Packaging"]
author: Hyacehila
excerpt: Agent Skills 的流行主要来自低摩擦的能力封装。它是上下文工程中有价值的一层，但不是新的统一协议，也不是 Agent 的终局。
excerpt_en: "Agent Skills are popular because they package capabilities with low friction. They are a useful layer of context engineering, but not a new universal protocol or final agent form."
permalink: '/blog/2026/03/10/from-mcp-to-agent-skills/'
---

2024 年 11 月 25 日 Anthropic 推出了 Model Context Protocol（MCP），很多人以为 Agent 世界终于有了统一接口，后面的事情无非是补生态、补客户端、补分发。

但现实并没有这么线性。

2025 年 10 月 16 日，Anthropic 把 Agent Skills 推到台前；2025 年 12 月 18 日，Skills 变成开放标准；同一天 GitHub Copilot 宣布支持 Agent Skills；2026 年 2 月 2 日，OpenAI 在介绍 Codex app 时也公开提到 skills 机制。这个时间线本身已经说明了一件事：**即便协议已经存在，开发者仍然在寻找更轻、更软、更少约束的能力封装。**

所以这篇文章会给出一个判断：**Skills 是上下文工程里有价值的一环。相对 MCP，它不是跨越式进步，更像是模型能力上升之后，对能力封装方式的一次轻量修补。它会火，主要因为足够简单。**

这句话听起来像是在故意贬低 Skills，但我想说的是：**Skills 的价值很真实，只是主要不在协议创新，而在工程压缩。** 它把一件原本需要 Host、Client、Server、Schema、权限模型与安装链路共同完成的事情，收束成了一个目录、一份 `SKILL.md`、几段脚本和若干参考资料。它没有发明一个全新的世界，却踩中了 Agent 开发里最痛的那个点：**能力应该如何被交给模型。**

## 引子：Skills 的爆火，不是因为它比 MCP 更先进

技术世界最容易犯的一个错误，就是把流行误写成更先进。

MCP 比 Skills 更完整，这几乎没有什么争议。它从一开始就不是单纯的 Tool Calling 协议，而是一个上下文协议：`tools` 、`resources`、`prompts`、`Registry`、 `.mcpb`、生命周期、能力协商、客户端权限边界、认证授权等等，都说明它想解决的是 Agent 与外部世界之间的接口标准化问题。无论是 SDK 还是文档本身，MCP 都比 Skills 复杂得多。

MCP 的设计是真正意义上的 **Model Context Protocol**。所有 AI 外部的上下文（提示词、工具、资源、人机交互）都在协议设计中被全面考虑过。理想情况下，一个 AI 模型（不需要额外的提示词）加上一个 MCP Server 就可以化身完整的 AI Agent。

而 Skills 根本不是沿着这条路线长出来的。它没有试图统一 transport，没有试图解决多端一致权限，也没有试图把所有能力都纳入一套严密的协议里。它做的事情很朴素：**把 Prompt、脚本、局部资料和工作流说明捆成一个能力包，让模型按需读取。**

Skills 给人的感觉是：既然 MCP 没能实现预期的效果（或者被误读为仅是工具标准），那我们就索性提一个新的标准出来。

这也是为什么我不认为 Skills 是 MCP 的下一代。它更像一条工程旁路：当协议太完整、使用太重、心智负担太高时，开发者会自然回到更接近文件系统、脚本和说明书的封装方式，**也就是更容易理解、安装和维护的能力包。**

这并不丢人。恰恰相反，这说明 Agent 开发的矛盾并不在于我们能不能设计出更优雅的协议，而在于我们能不能让能力交付给模型、被团队维护、被项目复用，并为项目本身创造真实价值。

## Agent 为什么总在寻找更轻的能力封装

从 Tool Call 到 MCP，再到 Skills，表面上看是在换技术名词，实际讨论的是同一个问题：**能力应该以什么形式暴露给模型。**

最早的 Tool Call 思路非常干净，给模型一个工具名、一个工具描述、一个参数 Schema，它就知道何时调用、如何调用。这套范式的好处是明确、稳定、容易审计，坏处也同样明显：它特别适合函数，不特别适合流程。（关于MCP为什么相对于传统Function Calling是有一定程度的进步的在之前的blog中讨论过了，不重复介绍）

但 Agent 世界里，很多重要能力恰恰不是函数，而是套路。

比如“写一份市场分析报告”“阅读仓库后输出技术方案”“按照我的风格生成一篇博客”“检查一个项目是否适合上线”，这些东西不是几个参数就能说清楚的。它们通常包含：

- 输入应该如何被理解
- 哪些资料需要先读
- 哪些步骤必须按顺序执行
- 哪些脚本值得直接运行
- 输出应该符合什么格式

这类能力如果强行改写成 Tool Schema，会显得生硬；如果纯靠 Prompt，又会因为太长、太脆弱、太难维护而失控。不是每个人都能长期维护一份千字提示词。于是中间地带自然出现了：**不如把能力写成一个可以被读取、被展开、被分发的包。**

Skills 成立，是因为它刚好长在这个中间地带上。它比 Tool Call 更柔软，比 MCP Server 更轻，比大段系统提示更稳定。对今天的大多数 Agent 项目来说，这个生态位一直存在，只是迟早会有人把它包装出来。

## Skills 到底是什么：说明书、脚本与按需知识的联合打包

### Making a Skill

如果把 MCP 看成 Agent 接外部世界的标准插座，那 Skill 更像一个工作包。它更像是把某一类任务的 SOP、脚本、模板、参考资料和使用规则，收拾成一个可以随身携带、随时分发的小目录。任务来了就打开，任务不相关就先别打扰上下文。

从格式上看，一个 Skill 最少只是一个带 `SKILL.md` 的文件夹；从运行上看，它是三层东西叠在一起：内容层、调度层、宿主层。内容层解决“这里面放什么”，调度层解决“什么时候触发”，宿主层解决“谁来加载、谁来执行、谁来约束权限”。这和 MCP 一样，也有多层结构。

可以先看一个最典型的样子：

```text
pdf-skill/
├── SKILL.md
├── references/
│   ├── forms.md
│   └── reference.md
├── scripts/
│   └── fill_form.py
└── assets/
    └── template.pdf
```

`SKILL.md` 是主说明书，但它自己也分两部分。开头的 YAML frontmatter 至少要有 `name` 和 `description`；后面才是给模型看的 Markdown 正文。这个设计很朴素：`name` 是标识，`description` 是触发条件。按照 Anthropic 的文档和 Agent Skills 开放规范，Agent 启动时通常不会把所有 Skill 正文都塞进上下文，而是先读取每个 Skill 的 `name` 和 `description`，把它们作为技能目录放进系统提示。用户请求命中某个 `description` 后，模型才去读取完整的 `SKILL.md`。这就是 Skills 的工程思想：**先暴露索引，再按需加载正文，使用渐进披露来缓解上下文的腐烂。** 更关键的是，这种渐进披露是默认路径。

接下来是第二层：为什么 Skill 里经常还有 `scripts/`。有些事情靠模型临场发挥不划算，也不稳定。比如解析 PDF 表单、批量改文件、调用固定 CLI、生成标准报告。每次都让模型现场拼命令，浪费 token，也容易翻车。Skill 的做法是让 `SKILL.md` 负责讲方法，让 `scripts/` 负责高频、机械、确定性要求高的部分。Anthropic 在介绍 Skills 架构时也明确说过，模型可以在触发 Skill 后用 bash 读取说明、选择运行脚本；脚本代码本身不一定要进上下文，模型只需要拿到执行结果。更准确地说，Skills 没有重新发明 tool calling 协议，而是让模型先读说明文档，再调用宿主已经提供的 `bash`、文件系统、代码执行能力去跑脚本，必要时自己组织命令、参数乃至一小段 Python 代码，再根据输出决定下一步。

第三层是 `references/` 和 `assets/`。`references/` 解决的是大块知识放哪儿，`assets/` 解决的是模板和素材放哪儿。Skill 的优秀之处，恰恰不在于它给模型喂了更多文本，而在于它让文本、脚本、模板和静态资源终于可以分开存放、分开加载。这也是渐进式披露：启动时先给目录；命中后再给正文；正文里如果提到表单说明、schema、模板，再继续去读相应文件。这样一来，Skill 可以很大，但上下文不必一上来就开始膨胀。

到了这里，Skill 的“内容层”其实就很清楚了：

- `SKILL.md`：主说明书，告诉模型什么时候用、怎么做、注意什么。
- `scripts/`：可执行的确定性操作，负责把高频动作做稳。
- `references/`：按需打开的参考资料，负责塞下那些不适合常驻上下文的大块知识。
- `assets/`：模板、样例和静态资源，负责支撑最终交付物。

这套结构没有什么神秘之处，但它非常符合 Agent 的使用直觉。一个 Skill 里最重要的不是脚本本身，而是**它把能力的不同层级摆在了不同位置上**。

### 启用Skill

但光有目录结构，Skill 还跑不起来。让它活起来的是调度层，也就是宿主系统额外塞给 Agent 的那层系统提示词与工具。Agent Skills 官方集成指南把这件事说得很直接：启动时扫描技能目录、解析 frontmatter，然后把可用 Skill 的元数据注入 system prompt，让模型知道自己有哪些能力。换句话说，Skill 不会主动跳出来；宿主先在系统提示里给模型一张技能清单，模型再根据用户请求决定要不要展开其中一个。

这层系统提示词通常至少要解决几件事：有哪些 Skill 可用；用户显式点名某个 Skill 时怎么处理；没有点名但任务明显匹配时是否自动触发；是把整个 `SKILL.md` 注入上下文，还是先去文件系统里读；多个 Skill 同时相关时能不能组合；如果 Skill 里有脚本，哪些工具默认允许、哪些需要额外确认。也正因为这里有明确的“发现—触发—加载—执行”链路，Skill 从来都不只是一个提示词文件夹，而是一种轻量的运行时机制。

Prompt 工程只是在想一句话怎么写，Skill 工程想的是：**怎样把一整类任务交给模型，而且以后每次都能复用。** 前者解决一次对话，后者开始碰工程化。

进一步往下看，不同产品其实是在同一套实现方向外面套了不同的壳。在 Claude Code 里，Skill 可以放在 `~/.claude/skills/` 作为个人能力，也可以放在项目里的 `.claude/skills/`，跟着 git 一起走，变成团队共有的工作流，Anthropic 还支持插件把 Skill 一起打包带进来。

在 OpenAI 这边，截至 2026 年 3 月 11 日，Help Center 已经明确写到 ChatGPT 里的 Skills 是可复用、可分享的工作流，可以自动使用一个或多个 Skill，也能在工作区里创建、安装、分享。你会发现这些产品外观差别很大，但底层直觉是一致的：**不管外壳怎么变，Skills都在做同一件事——把专有流程和组织知识交给模型。**

### 一点题外话

另外一个很有意思的细节是，Skill 的标准格式虽然故意做得很轻，但并不是完全没有边界。开放规范对 `SKILL.md` 的 frontmatter 其实有不少约束：`name` 要和目录匹配、长度有限、通常用短横线命名；`description` 不只是介绍文案，它直接影响模型能不能在合适的时候触发这个 Skill。

还有一些实现会支持 `compatibility`、`license`、`metadata`、甚至实验性的 `allowed-tools` 之类字段，用来表达运行环境要求、授权信息或工具边界。也就是说，Skills 的轻量不是随便写点 Markdown，而是“只把最关键的结构标准化，其余留给宿主实现”。

如果要用一句不那么绕的话总结这一节，我会说：**Skill 给 Agent 的不是新器官，而是说明书、工具箱和按需翻阅的附录。** 它的价值不在宏大设计，而在工程上顺手：人能写、团队能传、模型能用、上下文还不会立刻爆炸。

当然，写到这里你也能看出来，Skills 有一个隐含前提：模型得足够强，强到能看懂这份说明书、知道什么时候翻附录、什么时候运行脚本、什么时候只借用流程而不死搬硬套。也正因为如此，Skills 看起来像个目录结构，背后其实吃的是模型能力的红利。这个问题，我们下一节再展开。

## 为什么是现在：模型变强以后，不确定性的工具终于能被使用

如果把时间拨回更早，Skills 这种东西未必会这么成立。

因为它天生带着一种不确定性：模型拿到的不是一个严格的 JSON Schema，也不是一个完全标准化的远程服务，而是一组说明书、参考资料和脚本。它需要自己判断该不该用、先读什么、接着读什么、什么时候运行脚本、什么时候只保留模板不用执行。

这其实是非常高的要求。

Schema-based tool calling 把不确定性压到接口层；Skills 接受一部分不确定性，并把它交给模型理解能力。这也是为什么我认为 **Skills 的成立条件，不只是 `SKILL.md` 这个格式，而是模型已经强到可以消化这种半结构化能力入口了。**

Skills 不是凭空出现的设计创新，更像模型能力提升之后的副产物。当模型已经足够擅长读文档、读脚本、理解工作流、按描述路由能力时，开发者自然会倾向于用更松、更软的方式交付能力。严格 Schema 的边际收益开始下降，轻量封装的收益开始上升。

或者说：**Skills 与 Tool Call 的差别，在于谁来承担接口确定性。** `schema-based tool calling` 把确定性更多写进接口层；Skills 把其中一部分确定性转移给模型理解层。前者对模型友好，后者对开发者友好；前者更适合强约束动作，后者更适合宽松工作流与团队知识。

这也是 Skills 和早期 Tool Use 研究之间最有趣的关系。过去的工具调用研究一直在想办法让模型学会在海量 API 中找对工具；今天，前沿模型已经足够强，问题逐渐从如何训练模型会用工具迁移成如何让开发者更低摩擦地把能力交给模型。Skills 恰好踩在这条迁移线上。

> Anthropic 官方文档里已经提供了 `programmatic tool calling`。它允许 Claude 在 `code execution` 容器里写 Python 代码，把工具当函数来调用；中间的多次工具调用、过滤和聚合都发生在脚本里，而不是每一步都重新回到模型采样。官方给出的价值也很直接：多步工作流可以减少延迟，并降低中间结果对上下文窗口的持续占用。

> 这个细节说明 schema-first 一侧也在向脚本化编排靠拢：Skills 让模型读 `SKILL.md` 后调用 bash/CLI 脚本，programmatic tool calling 则让模型在受控的代码执行环境里自己写脚本去调工具。它不是 MCP core 的规范字段，而是 Anthropic Tool Use 基础设施的一部分；但它回应的是同一个工程问题：如何让模型不用把每一步都写成一条僵硬的 JSON 函数调用，仍然能够自己组织流程、处理中间结果，再继续往下做。

> 这也源于基础模型能力的进步，MCP也将拥有了CLI脚本的使用能力，只要主Agent本身（接入Skills和MCP的agent）拥有相关命令的权限，CLI script 源于模型能力进步，而不是Agent Skills的发明，

## MCP 的未竟事业：协议早就写在那里，体验却没有长出来

如果只看概念，Skills 并没有超出 MCP 太多。

MCP 一开始就不只是 `tools`。`resources`、`prompts`和后来的 `instructions`，都说明它本来就想把模型所需的外部上下文系统化。甚至从理念上看，MCP 比今天很多人理解的还要更大：它讨论的不是怎样给模型加函数，而是怎样把模型需要的外部能力、外部知识与交互边界统一组织起来。

问题不在协议想得不够多，而在于协议写在那里，不等于体验就自动长出来。

如果只谈能力覆盖，MCP 其实并不缺先给目录、再按需展开这类渐进披露机制。早在 2024 年 11 月 5 日公开的规范版本里，`tools/list`、`prompts/list`、`resources/list` 就都已经支持 `pagination`，而且各自可以通过 `listChanged` 通知告诉客户端目录发生了变化。协议层完全可以先暴露一部分工具、Prompt 或资源，再按 cursor 继续向后取，而不是一次把整个能力面都摊开。

到了 2025 年和 2026 年，Anthropic 又在产品层把这件事继续往前推：MCP tool search 默认会在工具描述占上下文超过 10% 时启用，把 MCP 工具标记为 `defer_loading: true`，先搜索、后展开。

MCP connector 和 tool search tool 也把 `tool_reference`、`default_config.defer_loading` 这些机制做成了现成能力。换句话说，MCP 并不缺渐进披露机制，缺的是把它包装成普通开发者一眼能懂、一用顺手的默认体验。目前还不够顺手。

MCP 的抽象太完整了，Anthropic 的工程团队几乎考虑了当时能考虑到的一切问题：协议、教程、多语言 SDK、分发社区，还在持续补充能力。完整意味着强大，也意味着更高的接入负担。开发者要处理 server、client、安装、权限、生命周期、能力暴露、客户端支持矩阵，还要面对不同产品对 `prompts`、`resources`、`instructions` 的呈现差异。**协议设计得再完整，如果产品体验不够低摩擦，也不会自动成为开发者的第一选择。** 理解 MCP 和开发 MCP 本身是一项专业工作。

这正是 Skills 命中的地方。它没有试图回答所有问题，只回答了最眼前的问题：**我怎么把这套能力塞给 Agent，让它今天就能用；我怎么让每个人都能给 Agent 增加能力，而不让这项能力继续停留在少数开发者手里。**

前者由 Skills 本身解决：它足够简单，容易理解，也更依赖模型能力而不是精妙工程。很多 Agent 只要有文件读取能力和少量提示词调整，就可以接入 Skills，不必先实现一个 Client。后者则更直接：造一个用于生成 Skills 的 Skill，并把它放到每个用户面前。

所以我会把 Skills 理解成 MCP 的一个旁路补丁，而非替代方案。它赢的不是协议完整性，而是体验。更准确一点说，在 MCP 还没把“轻量封装 + 轻量分发 + 轻量激活”做到极致之前，Skills 先拿走了开发者的耐心。

这也是为什么 MCP 后来继续补 `Registry`、补 `.mcpb`、补 `server instructions`，也在 SDK / API 层补 `tool search` 与 `defer_loading`。因为协议世界终究也意识到了：**接口定义不是全部，分发、安装、心智负担和默认体验同样是协议的一部分。**

> 关于MCP的 `server instructions`: MCP在第一次发布的时候不包含该功能，Anthropic的开发团队意识到了我们需要补充对Server的整体介绍而不是只有工具的description。

> 新增的 `instructions` 字段就是一本专门写给 AI 模型的用户手册（User Manual）, 也可以理解Skills的文本描述部分，它介绍这个Server里的各个功能，以避免之前的开发将全局规则被迫写到工具描述里。

> Anthropic 建议将 instructions 的使用重点放在 Tools 和 Resources 本身无法传达的信息上，主要包括以下三类：跨功能依赖关系（Cross-feature relationships），最佳操作模式（Operational patterns）以及系统约束与硬性限制（Constraints and limitations）

> 但要注意，`server instructions` 解决的是 server 级用户手册，不是渐进披露本身。真正负责先给目录、再按需展开的，是 `MCP tool search`、`tool_reference` 和 `defer_loading`。

如果把 Skills 拆开看，它的几块核心部件在 MCP 这套栈里其实都能找到近似对应物：
- `SKILL.md` 这样的说明性工作流更接近 `server instructions + prompts`
- `references/` 更接近 `prompts`
- `scripts/` 更接近 Host 开放的 `bash`、普通 `tools` 以及更进一步的 `programmatic tool calling` 的融合产物
- `assets` 则和 `resources` 有着一定的延续关系
- `metadata` 和 渐进读取则对应 `tool search + defer_loading`。

它们并不逐项同构，但组合起来，确实已经覆盖了 Skills 最核心的能力形态。从这个角度看，MCP 的能力面并不缺，缺的是让普通用户自然用到这些能力的路径。

## Skills 为什么会赢得开发者

如果只从纯技术完备性看，Skills 不应该这么火；但如果从真实开发流程看，它火得非常合理。

**它更适合团队知识**

团队最常沉淀下来的，从来不只是 API，而是规则、模板、风格和流程。怎么写报告、怎么做代码审查、怎么整理一次调研、怎么把输出对齐到组织语气，这些东西本来就更像一个 Skill/Claude.md，而不是一个 Tool。

**它更适合项目本地工作流**

很多能力根本不值得被做成一个独立服务。一个 repo 里的几个脚本、一份检查清单、一套产出模板，本来就和项目代码、项目规范、项目上下文绑在一起。Skills 把这些东西原地打包，反而更符合工程直觉。

**它提供了一种足够轻的能力分发格式**

很多团队想分发的不是标准化 API，而是一整套做事方式：一段说明、一份模板、几个脚本、一些参考资料。MCP 可以覆盖其中不少能力，但 Skills 把这件事做得更轻：一个 `SKILL.md` 加几个目录，就已经足够开始共享。它赢在打包和分发成本低。

通过 `SKILL.md` 这个入口，提示词、操作说明和工作流第一次被当作可传播的工程工件来对待。对很多团队来说，这种分发方式比严格协议更接近真实需求，也更接近日常协作。今天回头看，Skills 赢的不是独占能力，而是把原本分散在 MCP 协议、宿主 runtime 和 Tool Use 基础设施里的能力，整理成了一个所有人都看得懂的目录。

**它足够宽松**

这是我认为最关键的一点。Skills 的火，不只因为它好，也因为它**要求得少**。它不要求开发者先理解一整套严格协议，不要求每个能力都写成规范化接口，也不要求所有 Host 行为先达成一致。它宽松，所以容易被采用；容易被采用，所以扩散很快。

**它使用起来太简单了**

用户几乎不需要理解背后的协议细节：装上、描述清楚、让 Agent 自己发现，Skill 就可以开始工作。层次化暴露带来的额外 token 成本也通常可控。如果你想要分享自己的能力，甚至还可以使用生成 Skills 的 Skill 来辅助创建它，只需要几句提示词和几轮对话，而不是先学完一整套 server/client/schema 的技术栈。

到今天再看，这并不意味着 MCP 做不到同样的能力，而是它们默认没有把这些能力包装到同样低的理解门槛里。

标准化世界总有一个悖论：越标准，往往越重；越轻，往往越不标准。**Skills 眼下明显站在先让人用起来这一边。**

## 简单的代价：Skills 的标准不足、安全风险与维护债务

但简单从来不是免费的。

我对 Skills 最大的保留，不在于它有没有价值，而在于它把大量原本应该由接口、协议和宿主共同承担的约束，重新推回给了模型理解能力和团队维护纪律。

**模型负担更重**

Tool Call 的优势在于明确：名称、描述、参数，一个模型即便不那么聪明，也比较容易走对路。Skills 则不同。模型要从 `SKILL.md` 里理解何时启用能力，要决定是否继续读 `references/`，还要判断脚本是不是该运行。这当然更灵活，但也显然更依赖模型。

这也是为什么我对脚本化工具一直保留一点警惕。脚本 CLI 看起来像省掉了 Schema，实际上只是把一部分接口复杂度转移给模型：模型要自己读说明、辨认脚本能力、决定参数和执行时机。它更灵活，但并不比 Tool Call 更省心。

简单说，**Skills 省掉的是接口层复杂度；这部分复杂度会转移到模型理解层。**

**安全边界更模糊**

MCP 不是绝对安全，但它的风险暴露面比较明确：你知道 server 暴露了什么工具，知道客户端的权限确认在哪一层发生，也知道哪些调用是协议的一部分。Skills 的风险则更容易藏进普通文件和脚本里。一个能力包同时携带 instructions、references、scripts 时，它已经很接近供应链问题，不再只是 Prompt 问题。

这也是为什么项目级 Skills 在工程上必须谨慎。它们看起来像是知识包，实际上往往带着执行权。

事实上Skills投毒已经不是只存在想象中的事情了，尤其是Openclaw爆火之后，有毒的Skills充斥在整个社区，而且无人在意。而在2024年的MCP浪潮中，虽然MCP也很可能被下毒，但起码还有企业在为它们提供的MCP Server背书，Skills却是人人都在制造，并依靠几乎没有审核机制的Hub分发。

**可移植性比想象中弱**

开放标准不代表统一运行时。Claude、Copilot、Codex 说自己支持 Skills，并不意味着它们在目录扫描、权限控制、网络访问、依赖安装、脚本环境和 UI 呈现上完全一致。Skill 可以被传播，不代表 Skill 可以被无损执行。

换句话说，Skills 眼下更像**可迁移的封装格式**，还不是严格意义上的统一接口标准。

**它很容易变成新的文档工程**

最早你以为自己只是写了一份 `SKILL.md`，后来你会发现自己在维护一整套信息架构：描述不能冲突、引用不能失效、脚本不能漂移、模板不能过期、不同模型上的触发效果还可能不同。Skill 越多，这个问题越明显。

渐进式披露也不是白送的。作者必须控制各层级的信息分布和信息密度：写少了，模型看不懂，能力会误用，甚至没法继续读；写多了，渐进加载就失去意义，额外 token 消耗又会涨回来。于是你会发现，Skills 不是消灭复杂度，而是把一部分复杂度从接口设计换成信息设计。

2026 年 2 月有一篇针对公开 Claude Skills 的实证研究，统计了 40,285 个公开 skills，结论之一就是生态里存在明显的冗余和非平凡安全风险。这件事并不让我惊讶，因为只要一种能力封装足够轻、足够宽松，它就一定会先经历一次野蛮生长，然后才开始补治理。

所以我对 Skills 的判断一直都不是“危险，不要用”，而是：**它值得用，但必须带着工程警惕去用。**

## Skills 在 Agent 版图中的位置：它是上下文工程的一环，不是终局

我觉得讨论 Skills 最容易走偏的地方，就是把它写成一个足以代表 Agent 未来的大概念。

它不是。

Skills 解决的是能力如何进入上下文：什么知识、流程、脚本应该交给模型，以及如何封装和分发。它有价值，但只负责加载，不负责全部上下文治理。

这也是为什么 Skills 出现以后，Memory、Context Engineering、Subagent 这些概念并没有消失，反而越来越重要。

- **Memory** 处理的是保留、压缩和回忆，不是定义能力。
- **Context Engineering** 处理的是上下文隔离和任务拆分，不是封装能力本身。
- **MCP** 标准化的工具不会失去价值，它将能力封装给确定性的函数，对于可以标准化的场景，开发MCP仍旧非常有意义。

从这个角度看，Skills 的位置很明确：**它是 Context Engineering 的一层，是 Agent 能力暴露与能力加载的一层。** 上下文管理不只包括加载，还包括压缩、隔离、遗忘和恢复，所以 Skills 不会是终局。

更困难的问题在后面：

- 什么时候应该忘记已经读过的资料
- 什么时候只保留结论，不保留试错过程
- 什么时候应该把复杂任务拆给 subagent，而不是继续污染主上下文

如果加载阶段的优化空间有限，遗忘就会变得更关键，也就是 **Context 或者说 Memory 的管理**。系统需要及时把非关键的信息移出上下文，把注意力空间留给当前任务。**加载和遗忘是一组配套动作，它们和记忆、检索一起回应同一个开发问题：Context。**

这些问题，Skills 本身回答不了。它是上下文工程的范畴，是单一通用性 Agent 都要去回答的问题。

## 结语：这项技术值得用，但不值得神化

如果一定要用一句话给 Skills 下定义，我会这样说：

**它不是新的统一协议，而是一种能力打包技巧：把说明书、参考资料、脚本和按需加载组织成默认体验。**

这个评价听起来不够宏大，甚至有点普通，但我认为它更接近现实。Skills 的确有价值，它让 Agent 可以拥有项目私有能力、团队私有流程和组织经验。过去这些东西很难优雅暴露给模型。它比纯 Prompt 稳，比纯 Tool Call 柔软，比自建 MCP Server 省事。光凭这一点，它就值得认真对待。

但我不愿意把它写成一场技术飞跃。

它没有重新定义协议边界，没有自动统一权限模型，没有天然解决长期记忆，也没有消灭上下文治理的复杂度。它做得最对的一件事，是承认了一件很多工程师早就隐约感觉到的事实：**在 Agent 时代，标准不一定先赢，简单往往先赢。**

所以我的结论非常明确：

- **Skills 值得采用。** 尤其当你要封装团队知识、本地流程和轻量自动化时，它几乎是天然合适的。
- **Skills 不值得神化。** 它不是 MCP 的全面升级，也不是 Agent 架构的终局；在 MCP 这套技术里，它最突出的几项能力今天基本都能找到对应物。
- **Skills 最终仍要被放回更大的上下文工程框架里理解。** 它是其中一环，但终究只是一环。

Skills 不是答案本身，却逼着我们重新追问一个更实际的问题：**我们应该如何把能力组织进上下文，并构造一个真正有用的智能体。**

## 参考资料

- Anthropic, [Equipping agents for the real world with Agent Skills](https://claude.com/blog/equipping-agents-for-the-real-world-with-agent-skills)
- Anthropic, [Introducing Agent Skills as an open standard](https://www.anthropic.com/news/agent-skills)
- Anthropic Docs, [Agent Skills overview](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview)
- Anthropic Docs, [Skill authoring best practices](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices)
- Anthropic Claude Code Docs, [Extend Claude with skills](https://code.claude.com/docs/en/skills)
- Anthropic Docs, [Bash tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/bash-tool)
- Anthropic Docs, [Programmatic tool calling](https://platform.claude.com/docs/en/agents-and-tools/tool-use/programmatic-tool-calling)
- Anthropic Docs, [Tool search tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool)
- Anthropic Docs, [Connect to external tools with MCP](https://platform.claude.com/docs/en/agent-sdk/mcp)
- Anthropic Docs, [MCP connector](https://platform.claude.com/docs/en/agents-and-tools/mcp-connector)
- Anthropic, [Managing context on the Claude Developer Platform](https://claude.com/blog/context-management)
- Model Context Protocol, [Pagination (2024-11-05)](https://modelcontextprotocol.io/specification/2024-11-05/server/utilities/pagination)
- Model Context Protocol, [Tools (2024-11-05)](https://modelcontextprotocol.io/specification/2024-11-05/server/tools)
- Model Context Protocol, [Prompts (2024-11-05)](https://modelcontextprotocol.io/specification/2024-11-05/server/prompts)
- Model Context Protocol, [Resources (2024-11-05)](https://modelcontextprotocol.io/specification/2024-11-05/server/resources)
- Model Context Protocol Blog, [Introducing the MCP Registry](https://blog.modelcontextprotocol.io/posts/2025-09-08-mcp-registry-preview/)
- Model Context Protocol Blog, [Server Instructions: Giving LLMs a user manual for your server](https://blog.modelcontextprotocol.io/posts/2025-11-03-using-server-instructions/)
- Model Context Protocol Blog, [Adopting the MCP Bundle format (.mcpb) for portable local servers](https://blog.modelcontextprotocol.io/posts/2025-11-20-adopting-mcpb/)
- OpenAI Help Center, [Skills in ChatGPT](https://help.openai.com/en/articles/20001066-skills-in-chatgpt)
- Agent Skills, [How to add skills support to your agent](https://agentskills.io/client-implementation/adding-skills-support)
- Agent Skills, [Specification](https://agentskills.io/specification)
- GitHub Changelog, [GitHub Copilot now supports Agent Skills](https://github.blog/changelog/2025-12-18-github-copilot-now-supports-agent-skills/)
- OpenAI, [Introducing the Codex app](https://openai.com/index/introducing-the-codex-app/)
- arXiv, [Agent Skills: A Data-Driven Analysis of Claude Skills for Extending Large Language Model Functionality](https://arxiv.org/abs/2602.08004)
