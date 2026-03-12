---
layout: blog-post
title: "从 MCP 到 Agent Skills：为什么 Agent 又需要一种新的上下文工程协议？"
date: 2026-03-10 20:00:00 +0800
categories: [LLM]
tags: [Agent, MCP, Skills, Context Engineering, Tooling]
author: Hyacehila
excerpt: "Agent Skills 的爆火不是因为它比 MCP 更先进，而是因为它把能力封装做得足够简单。它是上下文工程中重要的一环，但不是新的统一协议，更不是 Agent 的终局。"
series: "Agent时代的基础设施"
featured: True
math: false
---

# 从 MCP 到 Agent Skills：为什么 Agent 又需要一种新的上下文工程协议？

2024 年 11 月 25 日 Anthropic 推出了 Model Context Protocol（MCP），很多人以为 Agent 世界终于有了统一接口，后面的事情无非是补生态、补客户端、补分发。

但现实并没有这么线性。

2025 年 10 月 16 日，Anthropic 把 Agent Skills 推到台前；2025 年 12 月 18 日，Skills 变成开放标准；同一天 GitHub Copilot 宣布支持 Agent Skills；2026 年 2 月 2 日，OpenAI 在介绍 Codex app 时也公开提到 skills 机制。这个时间线本身已经说明了一件事：**即便协议已经存在，开发者仍然在寻找更轻、更软、更少约束的能力封装。**

所以这篇文章不是一篇“Agent Skills 简介”，而是一篇判断：**Skills 是上下文工程里重要且有价值的一环，但它相对 MCP 不是跨越式进步，而是模型能力上升之后，对能力封装方式的一次轻量修补。它之所以会火，也不只是因为它足够好，或者比MCP更为先进，而是因为它足够简单。**

这句话听起来像是在故意贬低 Skills，但我真正想说的是：**Skills 的价值非常真实，只是它的价值主要不在协议创新，而在工程压缩。** 它把一件原本需要 Host、Client、Server、Schema、权限模型与安装链路共同完成的事情，压缩成了一个目录、一份 `SKILL.md`、几段脚本和若干参考资料。它没有发明一个全新的世界，却精准踩中了 Agent 开发里最痛的那个点：**能力应该如何被交给模型。**

## 引子：Skills 的爆火，不是因为它比 MCP 更先进

技术世界最容易犯的一个错误，就是把流行误写成更先进。

MCP 比 Skills 更完整，这几乎没有什么争议。它从一开始就不是单纯的 Tool Calling 协议，而是一个真正意义上的上下文协议：`tools` 、`resources`、`prompts`、生命周期、能力协商、客户端权限边界、认证授权、后续的 Registry 与 `.mcpb` 等等，都说明它想解决的是 Agent 与外部世界之间的接口标准化问题。无论是SDK还是文档本身，MCP都比Skills复杂得多。

MCP 的设计是真正意义上的 **Model Context Protocol**。所有 AI 外部的上下文（提示词、工具、资源、人机交互）都在协议设计中被全面考虑过。理想情况下，一个 AI 模型（不需要额外的提示词）加上一个 MCP Server 就可以化身完整的 AI Agent。

而 Skills 根本不是沿着这条路线长出来的。它没有试图统一 transport，没有试图解决多端一致权限，也没有试图把所有能力都压进一套严密的协议原语里。它做的事情非常朴素：**把 Prompt、脚本、局部资料和工作流说明捆成一个能力包，让模型按需读取。**

Skills 给人的感觉是：既然 MCP 没能实现预期的效果（或者被误读为仅是工具标准），那我们就索性提一个新的标准出来。

这也是为什么我不认为 Skills 是MCP 的下一代。它更像是一条工程旁路：当协议太完整、使用太重、心智负担太高时，开发者会自然地回到更接近文件系统、更接近脚本、更接近说明书的那种封装方式，**或者说一种更为简单，无需心智负担就能使用的那种方式。**

这并不丢人。恰恰相反，这说明 Agent 开发的核心矛盾并不在于我们能不能设计出更优雅的协议，而在于我们能不能让能力被真正交付给模型、被团队真正维护、被项目真正复用。并为项目本身创造真实的价值。

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

这类能力如果强行压成 Tool Schema，会显得生硬；如果纯靠 Prompt，又会因为太长、太脆弱、太难维护而迅速失控（我不想并不是每个人都能写一个千字长文的提示词）。于是中间地带自然出现了：**不如把能力写成一个可以被读取、被展开、被分发的包。**

Skills 之所以成立，不是因为它突然发现了一个全新的技术方向，而是因为它刚好长在这个中间地带上。它比 Tool Call 更柔软，比 MCP Server 更轻，比大段系统提示更稳定。对今天的大多数 Agent 项目来说，这个生态位本来就存在，只是迟早会有人把它包装出来。

## Skills 到底是什么：说明书、脚本与按需知识的联合打包

### Making a Skill

如果把 MCP 看成 Agent 接外部世界的标准插座，那 Skill 更像一个上岗包。别把它想成给模型偷偷换脑子，它更像是把某一类任务的 SOP、脚本、模板、参考资料和使用规则，收拾成一个可以随身携带、随时分发的小目录。任务来了就打开，任务不相关就先别打扰上下文。

从格式上看，一个 Skill 最少只是一个带 `SKILL.md` 的文件夹；从运行上看，它却是三层东西叠在一起：内容层、调度层、宿主层。内容层解决“这里面放什么”，调度层解决“什么时候触发”，宿主层解决“谁来加载、谁来执行、谁来约束权限”。这也是为什么我前面说它不是 Prompt 的别名。Prompt 只是其中一层，而且通常只是入口层。

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

`SKILL.md` 是主说明书，但它自己也分两部分。开头的 YAML frontmatter 至少要有 `name` 和 `description`；后面才是给模型看的 Markdown 正文。这个设计很朴素，但极其关键：`name` 是标识，`description` 是触发条件。按照 Anthropic 的文档和 Agent Skills 开放规范，Agent 在启动时通常不会把所有 Skill 正文都塞进上下文，而是先只读每个 Skill 的 `name` 和 `description`，把它们放进系统提示里，当作一份“可用能力目录”。当用户请求和某个 `description` 对上号时，模型才会真正去读取完整的 `SKILL.md`。这就是 Skills 最核心的工程思想：**先暴露索引，再按需加载正文，使用渐进披露来缓解上下文的腐烂**

接下来是第二层：为什么 Skill 里经常还有 `scripts/`。原因很简单，有些事情靠模型临场发挥不划算，也不稳定。比如解析 PDF 表单、批量改文件、调用某个固定 CLI、生成标准报告，这些动作每次都让模型现场拼命令，不仅浪费 token，也容易翻车。于是 Skill 的做法是：让 `SKILL.md` 负责讲方法，让 `scripts/` 负责做那些高频、机械、确定性要求高的部分。Anthropic 在介绍 Skills 架构时甚至明确说过，模型可以在触发 Skill 后用 bash 读取说明、选择运行脚本，而脚本代码本身不一定要进上下文，模型只需要拿到执行结果就够了。这个设计很像“说明书 + 工具箱”的分工：说明书告诉你什么时候用工具，工具负责把重复劳动稳定做完。

第三层是 `references/` 和 `assets/`。这两个目录特别像“别把整本附录印在首页”。`references/` 解决的是大块知识放哪儿，`assets/` 解决的是模板和素材放哪儿。Skill 的优秀之处，恰恰不在于它给模型喂了更多文本，而在于它让文本、脚本、模板和静态资源终于可以分开存放、分开加载。开放规范和 Anthropic 文档都把这套思路叫 progressive disclosure，也就是渐进式披露：启动时先给目录；命中后再给正文；正文里如果提到表单说明、schema、模板，再继续去读相应文件。这样一来，Skill 可以很大，但上下文不必一上来就开始膨胀。最初的MCP没有将其内嵌在每个用户的认知中，大家还是以为MCP越多上下文就会开始膨胀，后续的`instructions `处理了这个问题但只有少部分开发者认识到了。

到了这里，Skill 的“内容层”其实就很清楚了：

- `SKILL.md`：主说明书，告诉模型什么时候用、怎么做、注意什么。
- `scripts/`：可执行的确定性操作，负责把高频动作做稳。
- `references/`：按需打开的参考资料，负责塞下那些不适合常驻上下文的大块知识。
- `assets/`：模板、样例和静态资源，负责支撑最终交付物。

这套结构没有什么神秘之处，但它非常符合 Agent 的使用直觉。一个 Skill 里最重要的不是脚本本身，而是**它把能力的不同层级摆在了不同位置上**。

### 启用Skill

但光有目录结构，Skill 还跑不起来。真正让它活起来的是“调度层”，也就是宿主系统额外塞给 Agent 的那一层系统提示词。Agent Skills 官方集成指南把这件事说得很直接：宿主在启动时应该扫描技能目录、解析 frontmatter，然后把可用 Skill 的元数据注入 system prompt，让模型知道自己有哪些能力可以调用。换句话说，Skill 不是自己站起来说“用我用我”，而是宿主先在系统提示里给模型一张技能清单，模型再基于用户请求决定要不要展开其中一个。

这层系统提示词通常至少要解决几件事：有哪些 Skill 可用；用户显式点名某个 Skill 时怎么处理；没有点名但任务明显匹配时是否自动触发；是把整个 `SKILL.md` 注入上下文，还是先去文件系统里读；多个 Skill 同时相关时能不能组合；如果 Skill 里有脚本，哪些工具默认允许、哪些需要额外确认。也正因为这里有明确的“发现—触发—加载—执行”链路，Skill 从来都不只是一个提示词文件夹，而是一种轻量的运行时机制。

这也是我为什么更愿意把 Skills 理解成“内容包 + 调度规则”，而不是“Prompt 工程的别名”。Prompt 工程只是在想一句话怎么写，Skill 工程想的是：**怎样把一整类任务交给模型，而且以后每次都能复用。** 前者解决一次对话，后者开始碰工程化。

进一步往下看，不同产品其实是在同一套实现方向外面套了不同的壳。在 Claude Code 里，Skill 可以放在 `~/.claude/skills/` 作为个人能力，也可以放在项目里的 `.claude/skills/`，跟着 git 一起走，变成团队共有的工作流；Anthropic 还支持插件把 Skill 一起打包带进来。在 OpenAI 这边，截至 2026 年 3 月 11 日，Help Center 已经明确写到 ChatGPT 里的 Skills 是“可复用、可分享的工作流”，可以自动使用一个或多个 Skill，也能在工作区里创建、安装、分享。你会发现这些产品外观差别很大，但底层直觉是一致的：**Skill 要么是文件系统里的能力目录，要么是产品里的可安装工作流；不管外壳怎么变，它都在做同一件事——把专有流程和组织知识交给模型。**

### 一点题外话

另外一个很有意思的细节是，Skill 的标准格式虽然故意做得很轻，但并不是完全没有“工程边界”。开放规范对 `SKILL.md` 的 frontmatter 其实有不少约束：`name` 要和目录匹配、长度有限、通常用短横线命名；`description` 不只是介绍文案，它直接影响模型能不能在合适的时候触发这个 Skill。

还有一些实现会支持 `compatibility`、`license`、`metadata`、甚至实验性的 `allowed-tools` 之类字段，用来表达运行环境要求、授权信息或工具边界。也就是说，Skills 的轻量不是“随便写点 Markdown”，而是“只把最关键的结构标准化，其余留给宿主实现”。

如果要用一句不那么绕的话总结这一节，我会说：**Skill 不是给模型加新器官，而是给 Agent 发一本说明书、一个工具箱和一叠按需翻阅的附录。** 它真正厉害的地方，不是技术上有多宏大，而是工程上足够顺手：人能写、团队能传、模型能用、上下文还不会立刻爆炸。

当然，写到这里你也能看出来，Skills 有一个隐含前提：模型得足够强，强到能看懂这份说明书、知道什么时候翻附录、什么时候运行脚本、什么时候只借用流程而不死搬硬套。也正因为如此，Skills 看起来像个目录结构，背后其实吃的是模型能力的红利。这个问题，我们下一节再展开。

## 为什么是现在：模型变强以后，不确定性的工具终于能被使用

如果把时间拨回更早，Skills 这种东西未必会这么成立。

因为它天生带着一种不确定性：模型拿到的不是一个严格的 JSON Schema，也不是一个完全标准化的远程服务，而是一组说明书、参考资料和脚本。它需要自己判断该不该用、先读什么、接着读什么、什么时候运行脚本、什么时候只保留模板不用执行。

这其实是非常高的要求。

Schema-based tool calling 的本质，是把不确定性消灭在接口层；Skills 的本质，则是接受一部分不确定性，并把这部分不确定性重新交还给模型能力。这也是为什么我认为 **Skills 的成立条件，根本不只是 `SKILL.md` 这个格式，而是模型已经强到可以消化这种半结构化能力入口了。**

换句话说，Skills 不是凭空出现的设计创新，而是模型能力提升之后的一种副产物。当模型已经足够擅长读文档、读脚本、理解工作流、按描述路由能力时，开发者当然会倾向于用更松、更软的封装方式来交付能力。因为这时候，严格 Schema 的边际收益开始下降，而轻量封装的工程收益开始上升。

这也是 Skills 和早期 Tool Use 研究之间最有趣的关系。过去的工具调用研究一直在想办法让模型学会在海量 API 中找对工具；今天，前沿模型已经足够强，问题逐渐从如何训练模型会用工具迁移成如何让开发者更低摩擦地把能力交给模型。Skills 恰好踩在这条迁移线上。

## MCP 的未竟事业：协议早就写在那里，体验却没有长出来

如果只看概念，Skills 并没有超出 MCP 太多。

MCP 一开始就不只是 `tools`。 `resources`、`prompts`、后来的 `instructions`，都说明它本来就想把模型所需的外部上下文系统化。甚至从理念上看，MCP 比今天很多人理解的还要更大：它讨论的不是“怎样给模型加函数”，而是“怎样把模型需要的外部能力、外部知识与交互边界统一组织起来”。

问题不在协议想得不够多，而在于协议写在那里，不等于体验就自动长出来。

MCP 的抽象太完整了,Anthropic的工程团队几乎考虑了当时能考虑到的一切问题，不仅规定了协议，教程，还提供了各种语言的SDK与分发社区，并且还在经常补充它。完整意味着强大，也意味着更高的接入负担。开发者要处理 server、client、安装、权限、生命周期、能力暴露、客户端支持矩阵，还要面对不同产品对 `prompts`、`resources`、`instructions` 的实际呈现与支持情况并不一致。**一个协议可以在设计上非常先进，但只要它没有长出足够低摩擦的产品体验，它就不会自动成为开发者的第一选择。** 理解MCP和开发MCP本身是一项专业的工作而不只是业余用户的随手行为。

这正是 Skills 命中的地方。它没有试图回答所有问题，只回答了那个最眼前的问题：**我怎么把这套能力塞给 Agent，让它今天就能用；我怎么让每个人都能去为Agent增加能力，而不是这项能力限制为少数人的特权**

前者是Skills本身解决的，它足够简单从而容易理解。它依赖模型的能力而不是精妙的工程。是一个Agent都可以轻松接入Skills而不是去制造一个Client。后者则利用了一个更粗暴的解决办法， 造一个用于生成Skills的Skills，并且把这个Skills丢在每个用户面前。

所以我会把 Skills 理解成 MCP 的一个旁路补丁，而不是替代方案。它不是在协议层赢了，而是在体验层赢了。更准确一点说，它是在 MCP 还没把“轻量封装 + 轻量分发 + 轻量激活”做到极致之前，先把开发者的耐心赢走了。

这也是为什么 MCP 后来继续补 Registry、补 `.mcpb`、补 server instructions。因为协议世界终究也意识到了：**接口定义不是全部，分发、安装、心智负担和默认体验同样是协议的一部分。**

> 关于MCP的 `server instructions`: MCP在第一次发布的时候不包含该功能，Anthropic的开发团队意识到了我们需要补充对Server的整体介绍而不是只有工具的description。

> 新增的 `instructions` 字段就是一本专门写给 AI 模型的用户手册（User Manual）, 也可以理解Skills的文本描述部分，它介绍这个Server里的各个功能，以避免之前的开发将全局规则被迫写到工具描述里。

> Anthropic 建议将 instructions 的使用重点放在 Tools 和 Resources 本身无法传达的信息上，主要包括以下三类：跨功能依赖关系（Cross-feature relationships），最佳操作模式（Operational patterns）以及系统约束与硬性限制（Constraints and limitations）

## Skills 为什么会赢得开发者

如果只从纯技术完备性看，Skills 不应该这么火；但如果从真实开发流程看，它火得非常合理。

**它更适合团队知识**

团队最常沉淀下来的，从来不只是 API，而是规则、模板、风格和流程。怎么写报告、怎么做代码审查、怎么整理一次调研、怎么把输出对齐到组织语气，这些东西本来就更像一个 Skill，而不是一个 Tool。

**它更适合项目本地工作流**

很多能力根本不值得被做成一个独立服务。一个 repo 里的几个脚本、一份检查清单、一套产出模板，本来就和项目代码、项目规范、项目上下文绑在一起。Skills 把这些东西原地打包，反而更符合工程直觉。

**它提供了一种足够轻的能力分发格式**

很多团队真正想分发的，并不是一个标准化 API，而是一整套做事方式：一段说明、一份模板、几个脚本、一些参考资料。MCP 并不是做不到这些，但 Skills 把这件事压缩得更轻：一个 `SKILL.md` 加几个目录，就已经足够开始共享。它在这里真正赢的不是“更强的标准化”，而是“更低成本的打包和分发”。

通过 `SKILL.md` 这个显眼入口，提示词、操作说明和工作流第一次被当作可传播的工程工件来对待。对很多团队来说，这种分发方式比严格协议更接近真实需求，也更接近日常协作。

**它足够宽松**

这是我认为最重要的一点。Skills 的火，不只是因为它好，而是因为它**要求得少**。它不要求开发者先理解一整套严格协议，不要求每个能力都写成规范化接口，也不要求所有宿主行为先达成一致。正因为它宽松，所以它容易被采用；正因为它容易被采用，所以它迅速扩散。

**它使用起来太简单了**

用户什么都不需要了解，安装，命令就可以在Agent里使用Skills，享受近乎免费的能力（层次化暴露带来的额外token消耗并不高）。如果你想要分享自己的能力，就可以使用构造Skills的Skills，只需要几句提示词，聊天就好，而不是写一天代码。

标准化世界总有一个悖论：越标准，往往越重；越轻，往往越不标准。**Skills 眼下明显站在先让人用起来这一边。**

## 简单的代价：Skills 的标准不足、安全风险与维护债务

但简单从来不是免费的。

我对 Skills 最大的保留，不在于它有没有价值，而在于它把大量原本应该由接口、协议和宿主共同承担的约束，重新推回给了模型理解能力和团队维护纪律。

**模型负担更重**

Tool Call 的优势在于明确：名称、描述、参数，一个模型即便不那么聪明，也比较容易走对路。Skills 则不同。模型要从 `SKILL.md` 里理解何时启用能力，要决定是否继续读 `references/`，还要判断脚本是不是该运行。这当然更灵活，但也显然更依赖模型质量。

这也是为什么我对“脚本化工具”一直保留一点警惕。脚本 CLI 看起来像省掉了 Schema，实际上只是把一部分接口复杂度转移给模型：模型要自己读说明、辨认脚本能力、决定参数和执行时机。它更灵活，但并不天然比 Tool Call 更省心。

简单说，**Skills 省掉的不是复杂度，而是接口层的复杂度；被省掉的那部分复杂度，最后会转移到模型理解层。**

**安全边界更模糊**

MCP 不是绝对安全，但它的风险暴露面比较明确：你知道 server 暴露了什么工具，知道客户端的权限确认在哪一层发生，也知道哪些调用是协议的一部分。Skills 的风险则更容易藏进普通文件和脚本里。一个能力包同时携带 instructions、references、scripts 时，它已经很接近供应链问题，而不仅仅是 Prompt 问题。

这也是为什么项目级 Skills 在工程上必须谨慎。它们看起来像是知识包，实际上往往带着执行权。

事实上Skills投毒已经不是只存在想象中的事情了，尤其是Openclaw爆火之后，有毒的Skills充斥在整个社区，而且无人在意。而在2024年的MCP浪潮中，虽然MCP也很可能被下毒，但起码还有企业在为它们提供的MCP Server背书，Skills却是人人都在制造。

**可移植性比想象中弱**

开放标准不代表统一运行时。Claude、Copilot、Codex 说自己支持 Skills，并不意味着它们在目录扫描、权限控制、网络访问、依赖安装、脚本环境和 UI 呈现上完全一致。Skill 可以被传播，不代表 Skill 可以被无损执行。

换句话说，Skills 眼下更像**可迁移的封装格式**，还不是严格意义上的统一接口标准。

**它很容易变成新的文档工程**

最早你以为自己只是写了一份 `SKILL.md`，后来你会发现自己在维护一整套信息架构：描述不能冲突、引用不能失效、脚本不能漂移、模板不能过期、不同模型上的触发效果还可能不同。Skill 越多，这个问题越明显。

渐进式披露也不是白送的。它要求作者非常仔细地控制各层级的信息分布和信息密度：写少了，模型看不懂，能力会误用，甚至没法一步一步继续读；写多了，渐进加载就失去意义，额外的 Token 消耗又会重新涨回来。于是你会发现，Skills 省掉的并不是复杂度本身，而是把一部分复杂度从“接口设计”换成了“信息设计”。

2026 年 2 月有一篇针对公开 Claude Skills 的实证研究，统计了 40,285 个公开 skills，结论之一就是生态里存在明显的冗余和非平凡安全风险。这件事并不让我惊讶，因为只要一种能力封装足够轻、足够宽松，它就一定会先经历一次野蛮生长，然后才开始补治理。

所以我对 Skills 的判断一直都不是“危险，不要用”，而是：**它值得用，但必须带着工程警惕去用。**

## Skills 在 Agent 版图中的位置：它是上下文工程的一环，不是终局

我觉得讨论 Skills 最容易走偏的地方，就是把它写成一个足以代表 Agent 未来的大概念。

它不是。

Skills 解决的是能力如何进入上下文，解决的是能力封装与能力分发，解决的是“把什么知识、什么流程、什么脚本交给模型”。它非常重要，但它只负责加载，不负责全部上下文治理。

这也是为什么 Skills 出现以后，Memory、Context Editing、Subagent 这些概念并没有消失，反而越来越重要。

- **Memory** 处理的是保留、压缩和回忆，不是定义能力。
- **Subagent** 处理的是上下文隔离和任务拆分，不是封装能力本身。
- **MCP** 标准化的工具不会失去价值，它将能力封装给确定性的函数，对于可以标准化的场景，开发MCP仍旧非常有意义。

从这个角度看，Skills 的准确位置其实很明确：**它是 Context Engineering 的一层，是 Agent 能力暴露与能力加载的一层。** 它很重要，因为上下文管理本来就重要；语言模型不从简洁中受益，它需要上下文来理解你的目标。但它不是终局，因为上下文管理从来不只有加载。

真正困难的问题始终在后面：

- 什么时候应该忘记已经读过的资料
- 什么时候只保留结论，不保留试错过程
- 什么时候应该把复杂任务拆给 subagent，而不是继续污染主上下文

如果载入阶段的优化空间有限，那么遗忘就是更重要的研究方向。即 **Context 的压缩或者说 Memory 的管理**。系统需要正确且及时地将非关键的信息从上下文中移出，腾出宝贵的注意力空间给核心任务。**加载和遗忘是与记忆和检索一样的一体两面的内容，他们都在回应 Agent 最核心的开发概念——Context。**

这些问题，Skills 本身回答不了。它是上下文工程的范畴，是单一通用性 Agent 都要去回答的问题。

## 结语：这项技术值得用，但不值得神化

如果一定要用一句话给 Skills 下定义，我会这样说：

**它不是新的统一协议，而是模型能力提升之后，一种非常有效的能力打包技巧。**

这个评价听起来不够宏大，相较于它的热度而言这句评价看起来太普通了，但我认为它反而更接近现实。Skills 的确重要，它让 Agent 真正开始拥有项目私有能力、团队私有流程、组织私有经验，这些以前很难优雅暴露的东西。它比纯 Prompt 稳，比纯 Tool Call 柔软，比自建 MCP Server 省事。光凭这一点，它就已经值得被认真对待。

但我不愿意把它写成一场技术飞跃。因为它不是。

它没有重新定义协议边界，没有自动统一权限模型，没有天然解决长期记忆，也没有消灭上下文治理的复杂度。它做得最对的一件事，是承认了一件很多工程师早就隐约感觉到的事实：**在 Agent 时代，标准不一定先赢，简单往往先赢。**

所以我的结论非常明确：

- **Skills 值得采用。** 尤其当你要封装团队知识、本地流程和轻量自动化时，它几乎是天然合适的。
- **Skills 不值得神化。** 它不是 MCP 的全面升级，也不是 Agent 架构的终局。
- **Skills 最终会被放回更大的上下文工程框架里理解。** 它是其中一环，而且是重要的一环，但它终究只是一环。

对我来说，这恰恰是它最迷人的地方。它不伟大，但它有用；它不彻底，但它击中了现实；它不是答案本身，却逼着我们重新去问那个更重要的问题：**我们应该如何将能力组织进入上下文，如何构造一个真正有用的智能体。**

## 参考资料

- Anthropic, [Equipping agents for the real world with Agent Skills](https://claude.com/blog/equipping-agents-for-the-real-world-with-agent-skills)
- Anthropic, [Introducing Agent Skills as an open standard](https://www.anthropic.com/news/agent-skills)
- Anthropic Docs, [Agent Skills](https://docs.claude.com/en/docs/agents-and-tools/agent-skills)
- Anthropic Claude Code Docs, [Extend Claude with skills](https://code.claude.com/docs/en/skills)
- Anthropic Claude Code Docs, [Create custom subagents](https://code.claude.com/docs/en/sub-agents)
- Anthropic, [Managing context on the Claude Developer Platform](https://claude.com/blog/context-management)
- OpenAI Help Center, [Skills in ChatGPT](https://help.openai.com/en/articles/20001066-skills-in-chatgpt)
- Agent Skills, [What are skills?](https://agentskills.io/what-are-skills)
- Agent Skills, [How to add skills support to your agent](https://agentskills.io/client-implementation/adding-skills-support)
- Agent Skills, [Specification](https://agentskills.io/specification)
- GitHub Changelog, [GitHub Copilot now supports Agent Skills](https://github.blog/changelog/2025-12-18-github-copilot-now-supports-agent-skills/)
- OpenAI, [Introducing the Codex app](https://openai.com/index/introducing-the-codex-app/)
- Model Context Protocol, [Specification](https://modelcontextprotocol.io/specification/2025-06-18)
- Model Context Protocol Blog, [Introducing the MCP Registry](https://blog.modelcontextprotocol.io/posts/2025-09-08-mcp-registry-preview/)
- Model Context Protocol Blog, [Server Instructions: Giving LLMs a user manual for your server](https://blog.modelcontextprotocol.io/posts/2025-11-03-using-server-instructions/)
- Model Context Protocol Blog, [Adopting the MCP Bundle format (.mcpb) for portable local servers](https://blog.modelcontextprotocol.io/posts/2025-11-20-adopting-mcpb/)
- Google Developers Blog, [Announcing the Agent2Agent Protocol (A2A)](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/)
- arXiv, [Agent Skills: A Data-Driven Analysis of Claude Skills for Extending Large Language Model Functionality](https://arxiv.org/abs/2602.08004)
