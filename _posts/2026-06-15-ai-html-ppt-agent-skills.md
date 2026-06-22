---
layout: blog-post
title: AI 自动生成 HTML PPT Skills
title_en: "AI-Generated HTML PPT Skills"
date: 2026-06-15 12:00:00 +0800
categories: [Agent 基础设施]
tags: [Agents, Agent Skills, HTML]
author: Hyacehila
excerpt: "AI 生成 HTML PPT 的问题，已经从多写几段 CSS，转向把设计规则、叙事顺序和领域判断交给 Skill。通用 Skill 管视觉和演示体验，专用 Skill 管教学、汇报这类场景里的内容组织。"
excerpt_en: "AI-generated HTML PPTs become interesting when Skills encode design rules, narrative sequencing, and domain judgment. General Skills handle visual quality and stage behavior; specialized Skills encode how different domains organize a talk."
featured: false
math: false
---

# AI 自动生成 HTML PPT Skills

前几天我写过一篇[《事实层与界面层：Markdown 与 HTML 不是替代关系》](/blog/2026/06/08/fact-layer-interface-layer-markdown-html/)，里面顺手提了一句：HTML 拿来做 PPT 也挺好用。

这里稍微展开一下。相关项目现在确实很实用：学习成本低，出片快，效果也不差。更重要的是，它不像很多 AI 小工具那样只热闹一阵。只要浏览器还适合展示，只要 Agent 还擅长写前端，它就有继续存在的理由。

HTML PPT 当然不是新东西。Reveal.js、Slidev、Marp 已经用了很多年，浏览器早就可以当演示播放器。变化发生在 Agent 这一层：Claude Code、Codex、Cursor 这类工具已经能稳定写前端，而 Agent Skills 把风格、叙事顺序和领域习惯写成可复用的说明书、模板、脚本和约束。于是「做一份能看、能讲、符合场景的 slides」开始变成一种可以打包和分享的能力。

这篇就看几个相关 Skill 分别在解决生成过程中的哪一环。

## 通用 Skill：来点艺术风格

通用 HTML PPT Skill 解决的问题是：用户说不清自己想要什么风格，只知道别像 AI 模板。没有约束时，结果很容易滑向紫蓝渐变、圆角卡片、空洞图标和一堆熟悉但没有性格的布局。每个用 AI 生成过前端界面的人能理解这是什么。

`frontend-slides`（21,714 stars）算是目前最受欢迎的相关 Skill。它的切入点很聪明。它没有继续堆 CSS，而是把风格选择改成了视觉预览。Agent 会先生成三种真实封面，用户看图来选，不用先在脑子里描述「设计流派」。大多数人说不出自己要哪种风格，但一眼能看出哪张不对味。

它用多风格模板和用户描述做匹配，再让用户在几个方案里挑。这个流程很省心。项目内置 12 种风格，包括浅色、暗色和四个特殊风格，也支持 beautiful-html-templates 提供的 34 个模板。对多数普通需求来说，`frontend-slides` 已经够用了。

`guizang-ppt-skill`（17,311 stars）更依赖用户主动选择风格。它有两套互不混用的视觉系统：一套是「电子杂志 × 电子墨水」，偏复古和风格化；另一套是瑞士国际主义，强调无衬线与现代简洁。

两种风格加起来只有 9 套配色，布局也被限制得比较紧。这种强约束不妨碍创作，反而很适合 Agent。模型越自由，越容易发明不存在的结构；deck 越长，风格漂移越明显。`guizang-ppt-skill` 用一部分局部自由换整体稳定性，长稿尤其受益。

`html-ppt-skill`（6,039 stars）更像一个完整的 HTML PPT 作者系统。它有 36 个主题、15 个完整 deck 模板、31 种单页布局、27 种 CSS 动画和 20 种 Canvas 特效。功能很全，特点没有前两个鲜明。

它的 Presenter Mode 挺有意思。按 `S` 可以打开独立演讲者窗口，里面有当前页、下一页、逐字稿和计时器。对于演讲者比较实用。

## 专用 Skill：让内容按领域逻辑展开

通用 Skill 解决的是“看起来像一份好的 PPT”。专用 Skill 处理另一个问题：在某个领域里，什么才算讲清楚。

教育、科普和正式汇报场景里，`visual-cognition-slides`（70 stars）和 `ppt-director` 很适合放在一起看。它们都先处理同一个问题：这一页为什么成立，观众应该从哪里理解它。当然他们目前做的工具仍旧很有限，是否应该取代讲者对内容的把控将思考也外包给 AI 值得商榷，与此同时他们本身也不怎么好用，能力很是有限。

`visual-cognition-slides` 更偏教学设计。它会先问受众是谁、最后只能记住一件事是什么，再按知识类型选择解释方式：概念性知识用类比动画，程序性知识用步骤动画，关系性知识用连接图，数据性知识用比例和趋势。它的硬规则也很明确：一张 slide 一个认知单元，文字只是标签，图形才是主体。对教育内容来说，这比换一套漂亮主题管用。

`ppt-director` 则更像正式汇报里的总导演。它关心受众/评审校准、页面结构导演稿、设计语言对表、HTML 预览和 PPTX 审查门禁；尤其是把不绑定风格的 `页面描述_优化版`，和带画布、坐标、字体字号、组件映射的 `生成就绪导演稿` 分开。前者解决“这页如何被理解”，后者解决“这页如何被生成”。这说明专用 Skill 的价值不只在领域知识，也在把生成前的判断过程固化下来。

商业汇报里，`KingDee-PPT-Skill`（54 stars）把企业品牌规范和商业模型绑在一起。他的内容分析阶段可以作为参考：识别金字塔/MECE、PDCA、SWOT、黄金圈、5W1H、SCQA、IPD 五看等结构，再映射到对应版式。它也采用 HTML-first 工作流，先做可演示的 HTML deck，再按需要导出 PPTX。

这种 Skill 不需要适合所有公司。它就是为某一类汇报习惯服务的：管理层看到 SWOT、PDCA、IPD，知道该从哪里读、该问什么问题。反过来讲，说服这些细分领域的人接受 HTML 而不是 PPTX，可能比继续打磨 Skill 本身更难。

## 结语

把这些项目放在一起看，通用 Skill 和专用 Skill 的分工很清楚。

通用 Skill 更像设计总监。它关心主题、字体、动效、舞台、演讲者模式和视觉节奏。它要解决的是空白页恐惧：我有材料，但不知道怎么把它做成一份像样的 PPT。

专用 Skill 更像领域协作者。它关心受众是否理解、教学内容有没有降低理解成本、商业逻辑是否落进熟悉框架。它主要处理信息组织问题，美观只是入口。

AI 做 PPT，最直观的好处是省时间。把大纲丢进去，几分钟后拿到一份能看的 slides。但 HTML PPT 和 Agent Skills 放在一起，重点不止是快。领域框架可以封装成 Skill，模型开始在受约束的流程里帮人组织表达。

主题当然重要，但主题不会单独存在。不同领域有不同的习惯和范式。以前这些东西要靠人自己学、自己记住，下次做 PPT 的时候再想起来。

如果 Skill 同时封装领域理解和专业风格，AI 生成 HTML PPT 就不只是自动排版。人给出材料和意图，Skill 带来领域规范与风格要求，Agent 再把它编译成一份可以观看、演讲、继续修改的界面。说白了，模板只是外壳，真正有价值的是里面那套可复用的判断。

## 参考资料

- [frontend-slides](https://github.com/zarazhangrui/frontend-slides)
- [guizang-ppt-skill](https://github.com/op7418/guizang-ppt-skill)
- [html-ppt-skill](https://github.com/lewislulu/html-ppt-skill)
- [visual-cognition-slides](https://github.com/edu-ai-builders/visual-cognition-slides)
- [ppt-director](https://github.com/Hermess/ppt-director)
- [KingDee-PPT-Skill](https://github.com/WayneZhon/KingDee-PPT-Skill)
- [《事实层与界面层：Markdown 与 HTML 不是替代关系》](/blog/2026/06/08/fact-layer-interface-layer-markdown-html/)
- [《从 MCP 到 Agent Skills：为什么 Agent 又需要一种新的上下文工程协议？》](/blog/2026/03/10/from-mcp-to-agent-skills/)
