---
title: "Agent 外接资源收藏册：Skills、MCP Server、插件与实用工具"
title_en: "An Agent Resource Collection: Skills, MCP Servers, Plugins, and Handy Tools"
date: 2026-05-17 21:00:00 +0800
categories: ["Agent Systems", "Agent Infrastructure"]
tags: ["Agent Skills", "MCP", "Tool Use"]
author: Hyacehila
excerpt: "一篇长期滚动更新的收藏册，记录可以给 Agent / Coding CLI 外接的各种资源：Skills、MCP Server、插件，以及顺手好用的小工具。"
excerpt_en: "A rolling collection of resources you can plug into agents and coding CLIs: Skills, MCP servers, plugins, and small tools that are handy enough to keep."
permalink: '/blog/2026/05/17/agent-resource-collection/'
---

这篇文章就是一个收藏夹，放一些我喜欢的、可以外接给 Agent / Coding CLI 的资源。

它们不一定很大，也不一定值得单独写一篇文章。可能是一套 Skills，一个 MCP Server，一个插件，或者某段脚本、某份提示词结构、某个看起来有点奇怪、但确实能省事的小办法。共同点是：都能挂到 Agent 或 Coding CLI 上，让它多一点能力。

这篇来源分两块：一块是**大厂 / 平台方基础服务 Skill**，多半是官方或平台团队维护的 CLI、产品文档和工程最佳实践；另一块是**独立开发者捣鼓出来的东西**，更像试验田，有编辑流程、创作工作流，也有一些小而顺手的工具。每条还是会标类型（Skill / MCP Server / 插件 / 工具），以后攒多了也不至于翻半天。

先留在这里。后面遇到新的，再慢慢补。

## 大厂 / 平台方基础服务 Skill

### [vercel-labs/skills](https://github.com/vercel-labs/skills)

**类型：Skill 分发工具 / CLI**

`vercel-labs/skills` 不是某个具体 skill，而是装 skill 的工具。入口是 `npx skills`，可以从 GitHub 仓库、本地目录或其他 git source 拉取 skills，再放到不同 Agent 能识别的位置。它不教 Agent 怎么写页面、做视频、跑实验，只管一件事：这些能力怎么装、怎么分享、怎么迁移。

它也要和 [vercel-labs/agent-skills](https://github.com/vercel-labs/agent-skills) 区分开。`vercel-labs/skills` 是 CLI / 分发工具，像基础设施；`vercel-labs/agent-skills` 才是 Vercel 官方维护的内容型 skills 集合。前者管怎么装，后者管装什么。这两个仓库放在一起看，能看出 Vercel 不只想写几份好用的 `SKILL.md`，也想把分发这件事做顺。Vercel 官方也写了一些关于他们在 Skills 方面研究的 blogs，他们的观点是值得参考的。

### [ClawHub](https://github.com/openclaw/clawhub)

**类型：Skill / Plugin 注册表 / 分发目录**

ClawHub 是 OpenClaw 生态里的公共 skill registry。它把围绕 `SKILL.md` 和配套文件的发布、版本、搜索、安装、评论、收藏和审查收进同一套目录里。具体任务怎么做，仍然交给各个 skill 自己；ClawHub 关心的是这些 skills 和 plugins 怎么被发现、更新和治理。

把它和 `vercel-labs/skills` 放在一起看会更清楚。`vercel-labs/skills` 偏安装和迁移：从 GitHub、本地目录或 git source 把 skill 放到合适的位置。ClawHub 偏 OpenClaw 自己的 registry 和 package catalog：既有技能目录，也开始覆盖 code plugins、bundle plugins 这类更重的扩展单元。对于 OpenClaw 来说，它承担的是公共市场和索引层的角色。

用这类注册表时，别只看名字就装。安装前先 inspect：来源、版本、changelog、metadata、扫描状态和 moderation 状态都值得扫一眼。Skill 的门槛低是好事，但它会把脚本、环境变量、外部服务和本地权限一起带进 Agent 工作流里。越顺手，越要先看清楚。

### [Remotion Agent Skills](https://www.remotion.dev/docs/ai/skills)

**类型：Skill**

Remotion Agent Skills 是 Remotion 给 Agent 准备的视频工程说明书。它让 Agent 按 Remotion 的方式写 React composition、frame-based 动画、音频、字幕、素材和转场，最后渲染成 MP4。这里的时间不是一页一页推进，而是一帧一帧算；产物也不是网页舞台，而是可以交付、复用、自动导出的视频文件。

它把视频生成拉回了工程层。很多 AI 视频工具适合输出气氛和镜头感（自动匹配图片素材），但技术内容、数据动画、字幕节奏、批量模板和 CI 渲染，还是需要更确定的结构。Remotion 让 Agent 不再是想象一段视频，而是用 React 和时间轴把视频做出来。真要把技术视频做成稳定栏目，或者在 CI 里批量 render，它更像生产引擎。在视频生成模型成本降低之前，我们还是需要依赖一下这些确定性的工具。

## 独立开发者捣鼓出来的东西

### [General Research Review Agent And Skills](https://github.com/Hyacehila/General-Research-Review-Agent-And-Skills)

**类型：Skill**

这是我自己做的一套 research review skill suite，主要用来辅助综述写作。它不是那种“帮我写一篇综述”的单条 prompt，而是把 scoping review 拆成几个可以接力的步骤：找文献、去重、筛选候选池、抽取论文证据、整理大纲、综合成文，最后检查引用和语言。

我最喜欢的是它把综述从一段聊天，变成了一套有中间产物的流程。以前做这类工作，经常是 Zotero、Google Scholar、手动表格、零散 PDF、临时 prompt 混在一起，最后再让 LLM 帮忙润色。这样当然也能做，但回头查的时候会很痛苦：哪些文章被排除了，为什么排除，某个结论到底靠哪几篇文章支撑，引用有没有贴错，都不太好追。

这套 skills 处理的就是这些地方。candidate pool、selection ledger、evidence notes、outline、citation map，还有最后的 PDF/HTML/Markdown 报告，都会留下来。它不是为了把文章写得更漂亮，而是让综述里最容易散掉的部分有迹可查：检索从哪里来，筛选怎么做，证据写在哪，正文引用支撑了什么。相比临时 prompt 或单纯的文献管理工具，它更适合有范围、有审计要求的综述型研究；如果只是随手总结几篇论文，就没必要上这么重的流程。

### [Humanizer](https://github.com/blader/humanizer) / [humanizer-zh-next](https://github.com/Hyacehila/humanizer-zh-next)

**类型：Skill**

Humanizer 和 humanizer-zh-next 是一组写作清理 skills，用来改掉文本里的 AI 味道。它们不是语法纠错工具，盯的是更烦人的东西：过度总结、宣传腔、三段式排比、破折号滥用、空泛的“关键作用”，还有那种一看就像聊天机器人回答的客套话。humanizer-zh-next 是我自己开发和维护的新版中文 Skill，在同类功能上做了更新和增强。

我喜欢它们，是因为这比“帮我润色一下”具体得多。普通润色经常会把文字改得更顺、更满，但也更像 AI；Humanizer 反过来，会删掉那些太工整、太会总结、太像模板的地方。英文内容可以用 Humanizer，中文内容就用 humanizer-zh-next。

humanizer-zh-next 对中文博客尤其有用。中文模型很容易写出“首先、其次、此外、综上所述”，也很容易把一句普通判断抬成“重要意义”“深刻影响”“复杂格局”。我很多时候不是想让文章更华丽，只是想让它听起来像真的有人写过、删过、犹豫过。写完技术文章、项目介绍、评论和随笔后，都可以拿它过一遍。

### [Beautiful Article](https://github.com/ConardLi/garden-skills/blob/main/skills/beautiful-article/SKILL.md)

**类型：Skill**

Beautiful Article 是一套把现成材料做成网页长文的 skill。网页、PDF、Word、Markdown、截图，甚至一大段粘贴材料，都可以先交给它处理。它会把这些东西整理成干净的源文，再做成一个可以离线打开、可以直接分享的单文件 HTML。它解决的不是生成一个网页，而是让 Agent 真的像编辑一样，先读材料，再重排结构，再把文章变得更好读，并给我们一个更加优雅的交互方式。

漂亮不是堆样式，对于一篇文章而言，更好读需要内容和视觉结构的配合。它会先抽取材料，再写编辑计划，然后停下来让用户确认文章类型、主题、版式、配图和封面。真正开写时，也不是一口气完成，而是先做首屏和第一节，让你看方向对不对。这个节奏很适合长材料：风格可以提前校准，信息也不容易在生成过程中悄悄丢掉。source、plan、review 这些中间文件还会留下来，回头检查时不会只剩一个成品 HTML。

它适合处理那些内容其实不错，但读起来太痛苦的材料，比如报告、教程、访谈、复盘、方案分析、解释文和很长的文章，也可以用于个人自己的分享。最后的成品可以有表格、代码块、图解、封面、目录，也可以加一点交互式小块，但它始终是在做文章，不是在做后台、表单、dashboard 或产品原型。如果手里有一份东西已经够扎实，只是太长、太散、太难读，我会很愿意把它交给 Beautiful Article 收拾一遍。

### [Web Video Presentation](https://github.com/ConardLi/garden-skills/tree/main/skills/web-video-presentation)

**类型：Skill**

Web Video Presentation 走的是内容导演路线。它先把文章或口播稿拆成 script、outline 和章节 step，再做成一个 16:9、可点击推进、适合录屏的网页舞台。它不负责把 React 代码渲染成 MP4，它关心的是怎么把一篇文章组织成值得录下来的讲解。

它不是一上来就写动画，而是要求先保留原文，产出口播稿和 outline，在稿子、章节、主题、素材、开发模式上停下来对齐。它的 `narrations.ts` 把 step 数和口播文本绑成唯一真相源，每一步独占整屏，画面细节还要回到 article 里抽。这个思路对技术博客特别有用：技术内容最怕被压成几句漂亮废话，它逼着 Agent 把论证链、数字、案例和节奏都留下来。

所以做单条技术分享时，它会比直接上 Remotion 顺手：先把博客拆成口播、章节和视觉节拍，再用浏览器录屏，得到一个足够清楚的视频。如果以后需要更稳定的字幕、音频和批量导出，再把这些中间产物迁到 Remotion 里做最终渲染也不迟。Web Video Presentation 给了很好的内容层和协作节奏，尤其适合把文章变成可讲、可录、还能保留细节的网页演示。也无需承担视频生成的高昂成本和抽卡风险。

### [yinyo-image2-prompt](https://github.com/xiaoshiyilangzhao1996-droid/yinyo-image2-prompt) / [GPT Image 2 Skill](https://github.com/ConardLi/garden-skills/blob/main/skills/gpt-image-2/SKILL.md)

**类型：Skill**

这两个 skills 都围绕同一个问题：怎么让 Agent 更稳定地使用 GPT Image 2 生图和改图，而不是每次临时堆一段提示词。新一代图像模型对“8K、ultra detailed、masterpiece”这类旧咒语已经不太吃了，真正影响结果的反而是主体顺序、场景结构、文字约束、镜头语言、编辑时哪些要变和哪些不能变。它们处理的就是这层翻译工作：把人的模糊想法，变成模型更容易执行的视觉指令。

yinyo-image2-prompt 小而专。它把 GPT Image 2 的常见任务拆成 35 个子模板，用 5-Phase 流程引导用户先判断场景、再补足关键参数，背后还整理了 33 组盲评和几组 PK 赛的经验。它会提醒你什么时候不要套模板：App UI、Logo、编辑工作流这种结构强的任务，模板很有帮助；Pixar 3D、概念插画这类更靠叙事和画面感的任务，过度工程化反而会把模型的自由度压坏。这个判断比单纯多塞几个 prompt 更有用。

ConardLi 的 GPT Image 2 Skill 则更像一条生图生产线。它不只负责写 prompt，还把运行环境分成三种模式：本地有 API key 时可以直接调用脚本出图并落盘；宿主 Agent 自带图像工具时，就把渲染好的 prompt 委托给宿主；如果什么图像工具都没有，它也能退化成纯 prompt 顾问。它还带有 `check-mode.js`、`generate.js`、`edit.js` 这些脚本，模板库按 UI、产品、信息图、学术图、技术图、编辑工作流等目录展开，prompt 和图片也会分别归档，适合真正接进 Agent 的工作流里反复使用。

yinyo-image2-prompt 更适合用来学习和优化 GPT Image 2 的提示词判断：该问什么、该省什么、什么场景别过度写。GPT Image 2 Skill 更适合作为工程入口：让 Agent 先判断模式，再选模板、渲染 prompt、保存产物，必要时直接出图。前者像脑子，后者像手脚；真正想把生图变成稳定工作流时，两个放在一起看会更完整。

生图得玩，感觉拿来搞钱还是很有前景的。
