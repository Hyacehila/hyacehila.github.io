# Hyacehila - Personal Portfolio & Blog (Jekyll)

> 一个静态个人主页（Portfolio）+ Jekyll 技术博客。

English: A static personal portfolio + Jekyll-powered blog.

## 目录

- [项目简介](#项目简介)
- [功能特性](#功能特性)
- [项目结构](#项目结构)
- [本地预览](#本地预览)
- [部署到 GitHub Pages](#部署到-github-pages)
- [博客写作指南（给 AI Agent）](#博客写作指南给-ai-agent)
- [License](#license)

## 项目简介

本仓库是我的个人站点，包含：

- **主页（Portfolio）**：单页站点，支持 **中英双语切换**，并包含旅行足迹地图等组件。
- **博客（Blog）**：基于 **Jekyll** 的 Markdown 博客系统，提供分类/标签筛选、系列文章聚合、代码高亮、Mermaid、MathJax 等能力。

线上预览（当前仓库配置）：

- 主页：`https://hyacehila.github.io/`
- 博客：`https://hyacehila.github.io/blog/`

项目最初 fork 自 `ivansaul/personal-portfolio`（GitHub：`https://github.com/ivansaul/personal-portfolio`），随后进行了较大幅度改造以适配 Jekyll 博客与自定义页面。

## 功能特性

### 主页（Portfolio）

- **响应式**：适配桌面/平板/移动端。
- **中英双语切换**：`index.html` 内通过 `data-i18n` + JS 字典渲染。
- **旅行足迹地图**：ECharts 渲染，支持 China/World 模式切换（数据在 `assets/travelmap/cities.json`）。
- **联系信息**：Email/Phone/Address/LinkedIn/GitHub 等（当前无“提交式”Contact Form）。

### 博客（Blog）

- **Markdown 写作**：Kramdown（GFM 输入）。
- **代码高亮**：Rouge（Monokai）。
- **数学公式**：MathJax（`$...$`/`$$...$$`）。
- **图表**：Mermaid（```mermaid 代码块）。
- **分类/标签筛选**：博客主页使用 7 个固定中文主分类 + 英文标签白名单筛选（仍以 `categories` 的第 1 个元素作为主分类）。
- **系列文章**：通过 Front Matter 的 `series` 字段聚合，并提供系列页：`/blog/series/?name=...`。
- **得意之作（精选）**：`featured: true` 的文章可在“得意之作”视图筛选。
- **阅读时长**：在博客卡片上按汉字统计（300 字/分钟，英文占比高的文章可能显示为 0）。
- **碎碎念**：在博客主页“碎碎念”视图中随机展示 `assets/murmur/murmur.json`。
- **上一篇/下一篇**：文章页底部导航（按时间排序）。
- **草稿（Drafts）**：草稿放在 `_drafts/`，本地可用 `--drafts` 预览，默认不会发布（见 `_config.yml`）。

## 项目结构

```
PersonelPage/
├── index.html                  # 主页（Portfolio，含 i18n & travelmap）
├── _config.yml                 # Jekyll 配置（permalink, future/show_drafts 等）
├── _posts/                     # 已发布文章（YYYY-MM-DD-slug.md）
├── _drafts/                    # 草稿（本地 --drafts 预览）
├── _layouts/
│   └── blog-post.html          # 文章页布局（固定 header、上一篇/下一篇、标签等）
├── _includes/
│   └── mathjax.html            # MathJax 配置与加载
├── blog/
│   ├── index.html              # 博客主页（筛选/系列卡片/碎碎念）
│   └── series/
│       └── index.html          # 系列页（按 series 聚合）
├── assets/
│   ├── css/style.css           # 全站样式
│   ├── js/script.js            # 主页交互脚本
│   ├── gitbook/                # 排版/高亮资源
│   ├── images/                 # 图片资源
│   ├── murmur/murmur.json      # 碎碎念数据
│   └── travelmap/              # 旅行足迹（ECharts + 数据）
├── code/                       # 文章配套代码/资料（会被静态发布）
├── README.md
└── LICENSE
```

## 本地预览

本仓库是纯静态文件 + Jekyll 站点结构。

如果你已经有 Ruby/Jekyll 环境：

```bash
jekyll serve
```

预览草稿与未来日期文章：

```bash
jekyll serve --drafts --future
```

校验博客分类/标签元数据：

```bash
python scripts/validate_taxonomy.py
```

说明：`_config.yml` 中默认 `future: false`、`show_drafts: false`，用于生产环境不发布草稿/未来文章。

## 部署到 GitHub Pages

1. 推送到 GitHub 仓库
2. 在仓库设置中启用 GitHub Pages（选择分支作为 Source）
3. 等待 Pages 构建完成后访问站点（通常为 `https://<username>.github.io/` 或 `https://<username>.github.io/<repo>/`）

## 博客写作指南（给 AI Agent）

本章用于指导后续 AI Agent：**将“原始草稿”加工为符合本仓库规范、可直接放入 `_posts/` 发布的 Markdown 博客文章**。

### 0. 最终交付物（Definition of Done）

- 产出 1 个 Markdown 文件：放入 `_posts/`（或先放 `_drafts/`），并满足文件命名与 YAML Front Matter 规范
- 文章结构清晰：`#` / `##` / `###` 层级合理，排版不破坏样式
- 元数据可用于博客主页筛选：`categories` / `tags` / `series` / `featured`
- `excerpt` 可直接用于列表卡片展示（短、准、无 Markdown）

### 1. 文件路径与命名

#### 1.1 草稿阶段（推荐）

- 放到：`_drafts/`
- 文件名：可自由（例如 `topic.md`），便于快速迭代
- 本地预览：`jekyll serve --drafts`

#### 1.2 发布阶段（必须）

- 放到：`_posts/`
- 文件名：`YYYY-MM-DD-<slug>.md`

`<slug>` 的建议规则（结合现有文章命名习惯做约束）：

- 推荐使用 **小写英文 + 连字符**：`high-dimensional-data-and-statistics`
- 系列文章可加入编号：`re0hf-01` / `re0hf-02`（也可以保留既有风格 `Re0HF-01`，但不建议混用大小写）
- **避免**：空格、逗号、引号、括号等复杂符号（会增加 URL slugify 与转义风险）
- 中文标题建议：用英文 slug（或拼音）+ 在 `title` 里写中文全称

### 2. YAML Front Matter 规范（强制）

每篇文章文件顶部必须包含 Front Matter：

```yaml
---
layout: blog-post
title: "文章标题（建议与正文 H1 完全一致）"
date: 2026-02-27 20:00:00 +0800
categories: [统计学]                           # 必须且只能有 1 个主分类
tags: [Statistical Inference, Resampling]     # 2~4 个英文标签，必须来自白名单
author: Hyacehila             # 推荐保留；可省略
excerpt: "给列表卡片用的一句话摘要，不要写 Markdown"

# 可选：系列文章
series: "概率图模型 (Probabilistic Graphical Models)"

# 可选：精选文章（会进入“得意之作”视图）
featured: true

# 可选：数学标记（当前页面会全量加载 MathJax，但建议保留该语义字段）
math: true
---
```

字段说明（结合现有页面逻辑）：

- `layout`: 固定为 `blog-post`（对应 `_layouts/blog-post.html`）。
- `title`: 用于页面标题与列表卡片标题；如包含冒号/引号等特殊字符，务必用双引号包裹。
- `date`: 用于排序、URL 与“上一篇/下一篇”。注意 `_config.yml` 默认 `future: false`，**未来日期文章不会发布**。
- `categories`: 必须为列表（`[...]`），且长度固定为 1。允许值仅有：`基础模型`、`训练与对齐`、`智能体系统`、`机器学习`、`统计学`、`数据科学`、`随笔与观察`。
- `tags`: 必须为列表（`[...]`），长度固定为 2~4，且全部使用英文白名单标签；博客主页用 `,` 拼接 tags 做筛选，因此 **单个 tag 名称不要包含逗号**。
- `excerpt`: 列表卡片摘要；建议 60~140 字（中文）或 20~40 words（英文），不换行、不写 Markdown。
- `series`: 相同字符串会被聚合到同一系列；系列页按 `date` 排序。
- `featured`: `true` 时会出现在“得意之作”。

固定 taxonomy：

- 主分类（7 个）：`基础模型`、`训练与对齐`、`智能体系统`、`机器学习`、`统计学`、`数据科学`、`随笔与观察`
- 英文标签白名单（30 个）：
  `Pre-Training`、`Fine-Tuning`、`Alignment`、`Reinforcement Learning`、`Reward Modeling`、`Reasoning`、`Multimodality`、`Model Mechanics`、`Agents`、`Tool Use`、`MCP`、`Context Engineering`、`Retrieval`、`Evaluation`、`Data Curation`、`Ensemble Learning`、`Interpretability`、`Imbalanced Learning`、`Scientific ML`、`Embeddings`、`Statistical Inference`、`Linear Models`、`Graphical Models`、`Time Series`、`Dimensionality Reduction`、`Resampling`、`Spatial Data`、`Data Visualization`、`Society`、`Methodology`
- 标签顺序建议：领域标签 -> 方法标签 -> 任务/视角标签 -> 可选补充标签
- 统一映射约束：
  - `SFT`/`PEFT`/`LoRA`/`Post-Training` 统一写为 `Fine-Tuning`
  - `RL`/`Agentic RL` 统一写为 `Reinforcement Learning`
  - `RLHF`/`RLVR`/`RLAIF` 优先归到 `Alignment`，奖励机制为主角时再补 `Reward Modeling`
  - `Agent`/`LLM Agent`/`Agent Framework`/`AEnvironment` 统一写为 `Agents`
  - `Tool Use`/`Skills`/`Workflow` 统一写为 `Tool Use`
  - `Context Engineering`/`Agent Memory` 统一写为 `Context Engineering`
  - `Data Pipeline`/`Data Cleaning`/`Deduplication`/`Data Curation` 统一写为 `Data Curation`
  - `Ensemble Learning`/`Tree Models` 统一写为 `Ensemble Learning`
  - `Explainability`/`Model Interpretability` 统一写为 `Interpretability`

### 3. 正文标题层级与排版逻辑（强约定）

#### 3.1 标题层级

- 正文第一行使用 `#`（H1）：**必须与 `title` 一致**
- 主要章节使用 `##`（H2）：用于“章节级”逻辑分割（引言/方法/推导/实验/结论…）
- 小节使用 `###`（H3）：用于展开细节（定义/推导步骤/实现细节/示例…）

#### 3.2 常用文章结构模板（按类型选用）

随笔/观点型（示例：`_posts/2026-01-04-does-llm-bring-equality.md`）：

1. `## 引子/背景`：用 1~2 段说明问题与动机
2. `## 核心观点`：用小标题拆分论点（每点 2~4 段）
3. `## 反例/边界条件`：说明何时不成立、有哪些争议
4. `## 结论`：回扣问题，给出 takeaway
5. `## 参考资料`：外链/论文/博客（可选）

教程/笔记型（示例：`_posts/2025-12-27-Re0HF-01.md`）：

1. `## 前言`：为什么需要这项技术
2. `## 本章学习目标`：列出 3~6 条明确目标
3. `## 核心概念/原理`：定义 + 直觉 + 公式（可选）
4. `## 实现细节/代码`：给出可运行片段、关键参数解释
5. `## 常见坑/最佳实践`：列 checklist
6. `## 小结`：本章总结 + 下一步

翻译/整理型（示例：`_posts/2026-02-08-why-im-not-a-fan-of-r-squared.md`）：

- 开头用 blockquote 标注来源：
  `> 本文核心观点翻译/整理自 ...（给出链接）`
- 适当加入 `## Take Home Message` 或 `## TL;DR`，先给结论再展开

#### 3.3 排版细则

- 段落：以“短段落”为主（2~5 句/段），避免一整屏的长段。
- 强调：用 `**加粗**` 突出关键结论；避免连续多段全加粗。
- 列表：优先用无序列表（`-`）表达要点；列表前后留空行。
- 表格：适合对比概念/参数（参考 `anscombes-quartet` 的数据表）。

### 4. 代码、公式、图表、图片的写法（按需）

#### 4.1 代码块（Rouge）

使用 fenced code block，并写清语言：

````markdown
```python
print("hello")
```
````

#### 4.2 数学公式（MathJax）

- 行内：`$a^2+b^2=c^2$`
- 行间：

```markdown
$$
E[X] = \mu
$$
```

已知坑（来自示例文章约定）：**公式里尽量不要直接写 `|`**，需要“竖线/条件”时用 `\mid`，避免被 Markdown 当成表格分隔符。

#### 4.3 Mermaid 图表

直接写：

````markdown
```mermaid
graph TD
  A --> B
```
````

#### 4.4 图片

Markdown 语法：`![alt](url)`。

- 当前仓库文章多使用外链图片；如需要更稳定，建议把图片放到 `assets/images/` 下并用相对路径引用：
  `![]({{ site.baseurl }}/assets/images/xxx.png)`

### 5. 从原始草稿到可发布文章：操作流程

1. **理解草稿意图**：确定文章类型（随笔/教程/整理），明确 1 句话核心结论。
2. **确定元数据**：`title`、`excerpt`、主分类（固定 7 选 1）、tags（2~4 个英文白名单标签）、是否系列/精选。
3. **搭建大纲**：用 `##` 写 4~8 个主章节；每个 `##` 下用 `###` 拆成 2~5 个小节（需要时）。
4. **填充内容**：优先写“结论/Takeaways”，再补论证与例子；技术文建议加入可运行代码片段与图表。
5. **润色与一致性**：统一术语（中英文大小写）、统一编号格式（`1.` / `1.1`）、删掉重复句。
6. **自检（见下方 checklist）**：确保 Front Matter 与 Markdown 不会破坏页面渲染。
7. **发布**：移动到 `_posts/`，按规范重命名，提交并 push，等待 Pages 构建。

### 6. 发布前自检 Checklist（必须逐条通过）

- [ ] 文件位于 `_posts/`，文件名为 `YYYY-MM-DD-<slug>.md`
- [ ] Front Matter 有且只有一段：以 `---` 开始/结束
- [ ] `layout: blog-post`
- [ ] `title` 与正文第一个 `#` 完全一致
- [ ] `date` 不是未来时间（除非你明确知道 `_config.yml` 的 `future` 行为）
- [ ] `categories`/`tags` 均为 `[...]` 列表，且 `categories` 恰好 1 个、`tags` 为 2~4 个英文白名单标签
- [ ] `excerpt` 为纯文本、无换行、可做列表摘要
- [ ] 所有 fenced code block 成对闭合；Mermaid 块使用 `mermaid` fenced block
- [ ] 公式块 `$$...$$` 成对出现；必要时避开 `|`（用 `\mid`）
- [ ] 外链图片/引用链接可访问
- [ ] `python scripts/validate_taxonomy.py` 通过

## License

本项目基于 [MIT License](LICENSE) 开源。
