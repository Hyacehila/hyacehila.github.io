---
title: AI 自动生成 PPT Skills：HTML Slides 与原生 PPTX
title_en: "AI-Generated PPT Skills: HTML Slides and Native PPTX"
date: 2026-06-15 12:00:00 +0800
categories: ["Work & Society", "AI Engineering Workflows"]
tags: ["HTML", "PPTX", "Agent Skills", "Workflow"]
author: Hyacehila
excerpt: "AI 做 PPT 的选择，不只在主题和动效，也在交付物。HTML slides 擅长快速探索视觉与演示体验；原生 PPTX 更适合继续修改、交接和正式交付。领域化 Skill 提供内容组织的思路，但还不能替代人对受众与内容的判断。"
excerpt_en: "Choosing an AI PPT Skill is not only about themes and effects, but also about the deliverable. HTML slides are strong for visual exploration and presenting; native PPTX is better for editing, handoff, and formal delivery. Domain-specific Skills offer useful content frameworks, but do not replace human judgment about audience and material."
permalink: '/blog/2026/06/15/ai-html-ppt-agent-skills/'
hidden: true
---

前几天我写过一篇[《事实层与界面层：Markdown 与 HTML 不是替代关系》](/blog/2026/06/08/fact-layer-interface-layer-markdown-html/)，里面顺手提了一句：HTML 拿来做 PPT 也挺好用。


HTML PPT 当然不是新东西。Reveal.js、Slidev、Marp 已经用了很多年，浏览器早就可以当演示播放器。变化发生在 Agent 这一层：Claude Code、Codex、Cursor 这类工具已经能稳定写前端，而 Agent Skills 把风格、版式、演讲体验和部分工作流约束写成可复用的说明书、模板、脚本和约束。原生 PPTX 路线则把同一件事接回传统办公软件的编辑和交付流程。

所以，选 Skill 的顺序可以稍微倒过来：先确定最后需要的是 HTML slides 还是原生 PPTX，再看视觉约束、可编辑性和人工复核的成本。这篇就按这个顺序看几个相关 Skill 分别解决了生成过程中的哪一环。

## 通用 Skill：风格、约束与交付格式

通用 Skill 解决的仍然是最常见的问题：用户说不清自己想要什么风格，只知道别像 AI 模板。没有约束时，结果很容易滑向紫蓝渐变、圆角卡片、空洞图标和一堆熟悉但没有性格的布局。每个用 AI 生成过前端界面的人能理解这是什么。

`frontend-slides` 的切入点很聪明。它没有继续堆 CSS，而是把风格选择改成视觉预览：Agent 先生成几种真实封面，用户看图来选，不用先在脑子里描述「设计流派」。大多数人说不出自己要哪种风格，但一眼能看出哪张不对味。

不过，这更适合作为探索方向的流程，而不是保证一次成稿的生产线。输入材料本身的密度、选中的风格分支，以及后续生成的轮次，都会放大结果差异。短 deck 或前期试稿时，它能快速帮人排除错误方向；直接拿来生成较长、较重要的演示稿，仍然需要逐页验收和返工。这个判断不是说它没有价值，而是把它放回更合适的位置：先帮人找到视觉方向，再决定是否继续投入。

`guizang-ppt-skill` 更依赖用户主动选择风格。它有两套互不混用的视觉系统：一套是「电子杂志 × 电子墨水」，偏复古和风格化；另一套是瑞士国际主义，强调无衬线与现代简洁。

两种风格的配色和布局都被限制得比较紧。这种强约束不妨碍创作，反而很适合 Agent。模型越自由，越容易发明不存在的结构；deck 越长，风格漂移越明显。`guizang-ppt-skill` 用一部分局部自由换整体一致性，长稿尤其受益。

`html-ppt-skill` 更像一个完整的 HTML PPT 作者系统。它提供主题、完整 deck 模板、单页布局、CSS 动画和 Canvas 特效，覆盖面很广；按 `S` 还能打开独立的 Presenter Mode，查看当前页、下一页、逐字稿和计时器。它解决的不只是「做出页面」，而是把浏览器演示这件事补成一个比较完整的演讲工具。


HTML 并不是唯一的终点。如果交付物还要进入 PowerPoint 或 WPS 的既有流程，`ppt-master` 值得单独看。它把重点放在原生 PPTX：既可以从材料开始做一份新的 deck，也可以向既有模板填充内容，或者在现成的 PPT 上继续增强。

我更愿意把它当作正式交付时优先尝试的 Skill。它的价值不只在于做出一份看起来不错的 PPT，而在于把成品留在原生文件的编辑链路里：生成之后仍能继续改文案、套公司模板、调整细节、交给其他人接手。HTML slides 和原生 PPTX 并不是高下关系，前者适合浏览器里的演示和视觉探索，后者更适合需要反复修改与交接的文档流程。

## 专用 Skill：有价值的内容框架，但还不是主流程

通用 Skill 更关心怎么做出一份能看的 PPT。专用 Skill 想解决的是另一个问题：在某个领域里，什么才算讲清楚。这个方向值得继续尝试，但它们目前更像可借鉴的内容框架和检查清单，还不能替代讲者对材料、受众和叙事的把控。

教育、科普场景里的 `visual-cognition-slides` 提供了一个很好的提醒：先明确受众是谁、观众最后只需要记住什么，再按知识类型选择解释方式。概念性知识可以用类比，程序性知识可以用步骤，关系性知识可以用连接图，数据性知识可以用比例和趋势；一张 slide 只承担一个认知单元。它的价值不在于自动替你完成教学设计，而在于让制作 PPT 的人重新检查：这一页到底在帮观众理解什么。

`ppt-director` 则把正式汇报中的前置思考拆得更细：一份不绑定具体风格的页面描述，和一份带画布、坐标、字体与组件映射的生成稿，分别回答这页如何被理解和这页如何被生成。这套区分很有用，但它更像一层导演稿，而不是任何场景都可以直接拿来生产的默认工作流。

这些尝试说明，领域经验当然可以写进 Skill；但把内容按领域逻辑展开做成稳定、易用、可迁移的工作流，离真正成熟还有距离。

## 结语

AI 做 PPT，最直观的好处是省时间。把大纲丢进去，几分钟后拿到一份能看的 slides。但真正影响后续体验的，往往不是第一次生成得有多快，而是生成之后还能不能继续改、能不能稳定地把关、能不能顺利交给下一个人。

所以我现在会先问交付物是什么：需要一份适合浏览器演示和快速探索的 HTML slides，还是一份进入 PowerPoint 或 WPS 流程的原生 PPTX？答案确定后，再选强调视觉风格、强约束一致性或原生编辑能力的 Skill，返工成本会低很多。

专用 Skill 依然值得关注。它们把教学、汇报和企业框架里的隐性经验摊开，让人看见一份 PPT 在排版之前还需要哪些判断。不过在当前阶段，它们更适合作为内容组织的辅助，而不是自动替代人做叙事决策。人提供材料、意图和最后的验收；Skill 负责缩短探索、制作和修改的路径。这已经足够有价值。

## 参考资料

- [frontend-slides](https://github.com/zarazhangrui/frontend-slides)
- [guizang-ppt-skill](https://github.com/op7418/guizang-ppt-skill)
- [html-ppt-skill](https://github.com/lewislulu/html-ppt-skill)
- [ppt-master](https://github.com/hugohe3/ppt-master)
- [visual-cognition-slides](https://github.com/edu-ai-builders/visual-cognition-slides)
- [ppt-director](https://github.com/Hermess/ppt-director)
- [KingDee-PPT-Skill](https://github.com/WayneZhon/KingDee-PPT-Skill)
- [《事实层与界面层：Markdown 与 HTML 不是替代关系》](/blog/2026/06/08/fact-layer-interface-layer-markdown-html/)
- [《从 MCP 到 Agent Skills：为什么 Agent 又需要一种新的上下文工程协议？》](/blog/2026/03/10/from-mcp-to-agent-skills/)