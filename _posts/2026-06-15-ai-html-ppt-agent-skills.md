---
layout: blog-post
title: AI 自动生成 HTML PPT Skills
title_en: "AI-Generated HTML PPT Skills"
date: 2026-06-15 12:00:00 +0800
categories: [Agent 基础设施]
tags: [Agents, Agent Skills, HTML]
author: Hyacehila
excerpt: "AI 生成 HTML PPT 的问题，已经从多写几段 CSS，转向把设计规则、叙事顺序和领域判断交给 Skill。通用 Skill 托住审美和演示体验，专用 Skill 则把教学和商业汇报的组织逻辑写进生成过程。"
excerpt_en: "AI-generated HTML PPTs become interesting when Skills encode design rules, narrative sequencing, and domain judgment. General Skills handle visual quality and stage behavior; specialized Skills encode how different domains organize a talk."
featured: false
math: false
---

# AI 自动生成 HTML PPT Skills

前几天我写过一篇[《事实层与界面层：Markdown 与 HTML 不是替代关系》](/blog/2026/06/08/fact-layer-interface-layer-markdown-html/)，里面顺手提了一句：HTML 拿来做 PPT 也挺好用。

这里我们来展开一下这个问题，毕竟相关的项目现在算是非常实用的工具，学习成本很低，还能让我们很快的做出来相当美观且优秀的展示，如此低投入高回报的工作性价比不要太高，而且在短期可预见的未来，他应该还能存活挺久，保质期在 AI 项目里大概率算是比较长的那一批。

HTML PPT 当然不是新东西。Reveal.js、Slidev、Marp 已经用了很多年，浏览器早就可以当演示播放器。变化发生在 Agent 这一层：Claude Code、Codex、Cursor 这类工具已经能稳定写出高质量高水平的前端界面，无论是 HTML + CSS 还是 React 等框架都相当的强，而 Agent Skills 开始把「怎么做一份能看的、能讲的、符合某个领域习惯的演示」写成可复用的说明书、模板、脚本和约束，从而允许我们低成本的分享能力。当美观和低成本低门槛放在了一起，HTML Slides 爆火也不足为奇。

所以这篇文章来聊聊目前 AI 都能做到些什么，以及未来还有什么值得进一步推进的任务。各个 Skills 都在努力解决生成过程中的什么问题。

## 通用 Skill：来点艺术风格

通用 HTML PPT Skill 解决的问题是：用户说不清自己想要什么风格，只知道别像 AI 模板。没有约束时，结果很容易滑向紫蓝渐变、圆角卡片、空洞图标和一堆熟悉但没有性格的布局。每个用 AI 生成过前端界面的人能理解这是什么。

`frontend-slides`（21,714 stars）算是目前最受欢迎的相关 Skill 。它的切入点很聪明。它没有只堆更多 CSS，而是把风格选择改成了视觉预览。Agent 会先生成三种真实封面预览，用户通过看图来选，而不是先在脑子里描述「设计流派」。大多数人说不出自己要哪种风格，但一眼能看出哪张不对味。

通过一系列成熟的多风格设计模板参考以及根据用户描述的匹配机制，再搭配不同方案的选择，绝大多数用户都能找到一个符合自己内容的 HTML 风格。非常的傻瓜化操作。作者原生提供了 12 种风格，包括浅色，暗色以及四个特殊风格，也支持了 beautiful-html-templates 提供的 34 个模板。`frontend-slides` 可以满足大多数用户的大多数需求且使用起来非常干饭容易。

`guizang-ppt-skill`（17,311 stars）更需要用户来选择自己需要的风格，也更加适合各种分享和研究，帮助展示者更好的输出自己的观点与产品。它有两套互不混用的视觉系统：一套是「电子杂志 × 电子墨水」，提供复古质感和风格化的输出；另一套是瑞士国际主义，强调无衬线与现代简洁风格。方便我们客观的对比和陈述。

两种风格总共加起来只有9套配色设计，布局本身被严格限制来控制自由发挥。这种强约束并不妨碍创作，反而很适合 Agent。模型越自由，越容易发明不存在的结构；deck 越长，风格漂移越明显。`guizang-ppt-skill` 用一部分局部自由换整体稳定性，长稿尤其受益。作者很好的体现了其在前端美学上的理解。

`html-ppt-skill`（6,039 stars）则更像一个完整的 HTML PPT 作者系统。它有 36 个主题、15 个完整 deck 模板、31 种单页布局、27 种 CSS 动画和 20 种 Canvas 特效。功能足够丰富但没那么有特色。

它的 Presenter Mode 挺有意思。按 `S` 可以打开独立演讲者窗口，里面有当前页、下一页、逐字稿和计时器。对于演讲者比较实用。

## 专用 Skill：让内容按领域逻辑展开

通用 Skill 解决看起来像一份好的 PPT。专用 Skill 处理另一个问题：在某个领域里，什么才算讲清楚。

教育和科普场景里，`visual-cognition-slides`（70 stars）很有启发。它不急着生成页面，而是先问受众是谁、最后只能记住一件事是什么。然后再按知识类型选择解释方式：概念性知识用类比动画，程序性知识用步骤动画，关系性知识用连接图，数据性知识用比例和趋势。

这其实是在把教学设计塞进生成过程。它的硬规则也很明确：一张 slide 一个认知单元，文字只是标签，图形才是主体。对教育内容来说，这比换一套漂亮主题更管用。课堂和科普最怕的不是丑，而是观众只是读完了屏幕，却没有在脑子里建立概念。

但他目前还不够好用，真的适合科研汇报场景的 PPT 或许还是 Beamer，在不难看的同时保持信息量与理解难度，是一件有难度的事情。`guizang-ppt-skill` 做的不错，但是当我们需要大规模图表对比的时候，他目前提供的模板仍旧无能为力，当然风格是对的。

商业汇报里，`KingDee-PPT-Skill`（54 stars）把企业品牌规范和商业模型绑在一起。页面当然有金蝶蓝，但更大的动作发生在内容分析阶段：识别金字塔/MECE、PDCA、SWOT、黄金圈、5W1H、SCQA、IPD 五看等结构，再映射到对应版式。它也采用 HTML-first 工作流，先做可演示的 HTML deck，再按需要导出 PPTX。

这种 Skill 不需要适合所有公司。他的结构为这个领域的一些问题而定制。方便管理层看到 SWOT、PDCA、IPD，知道该从哪里读、该问什么问题。但说服这些细分领域的人接受我们使用 HTML 而非 PPTX 可能比做一个更好的 Skill 要更难。

## 结语

把这些项目放在一起看，通用 Skill 和专用 Skill 的分工很清楚。

通用 Skill 更像设计总监。它关心主题、字体、动效、舞台、演讲者模式和视觉节奏。它要解决的是空白页恐惧：我有材料，但不知道怎么把它做成一份像样的 PPT。

专用 Skill 更像领域协作者。它关心受众是否理解、教学内容有没有降低理解成本、商业逻辑是否落进熟悉框架。它主要处理信息组织问题，美观只是入口。

AI 做 PPT，最浅的一层当然是省时间。把大纲丢进去，几分钟后拿到一份能看的 slides。但 HTML PPT 和 Agent Skills 放在一起，事情会更有意思。领域框架可以封装成 Skill，模型开始在受约束的流程里帮人组织表达。

主题当然重要，但主题绝不会单独存在。不同领域有着不同的习惯与范式，在这之前人们需要学习它，记到脑子里，然后在下一次做 PPT 的时候想起来。

但如果 Skills 同时封装了领域理解与专业的风格，AI 生成 HTML PPT 就越过了自动排版。 人给出材料和意图，Skill 带来领域规范与风格要求，Agent 把它编译成一份可以观看、演讲、继续修改的界面。Skill 内部不再是模板，而是可以被 Agent 直接利用的心智模型，为用户直接提供最终的表达层。

## 参考资料

- [frontend-slides](https://github.com/zarazhangrui/frontend-slides)
- [guizang-ppt-skill](https://github.com/op7418/guizang-ppt-skill)
- [html-ppt-skill](https://github.com/lewislulu/html-ppt-skill)
- [visual-cognition-slides](https://github.com/edu-ai-builders/visual-cognition-slides)
- [KingDee-PPT-Skill](https://github.com/WayneZhon/KingDee-PPT-Skill)
- [《事实层与界面层：Markdown 与 HTML 不是替代关系》](/blog/2026/06/08/fact-layer-interface-layer-markdown-html/)
- [《从 MCP 到 Agent Skills：为什么 Agent 又需要一种新的上下文工程协议？》](/blog/2026/03/10/from-mcp-to-agent-skills/)
