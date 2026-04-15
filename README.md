# Hyacehila - Personal Portfolio & Blog (Jekyll)

> 一个静态个人主页（Portfolio）+ Jekyll 技术博客。

English: A static personal portfolio + Jekyll-powered blog.

## 目录

- [项目简介](#项目简介)
- [功能特性](#功能特性)
- [项目结构](#项目结构)
- [本地预览](#本地预览)
- [部署到 GitHub Pages](#部署到-github-pages)
- [博客发布约定](#博客发布约定)
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
- **分类/标签筛选**：博客主页使用 6 个固定中文主分类 + 基于文章 Front Matter 自动生成的英文标签筛选（仍以 `categories` 的第 1 个元素作为主分类）。
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
├── scripts/
│   └── validate_taxonomy.py    # taxonomy 校验脚本（分类、系列、标签格式）
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

特别的，在目前的开发环境中不包含Jekyll，在任何博客修改完成后都不进行预览，而是等待人工审阅和部署后检查。

## 部署到 GitHub Pages

1. 推送到 GitHub 仓库
2. 在仓库设置中启用 GitHub Pages（选择分支作为 Source）
3. 等待 Pages 构建完成后访问站点（通常为 `https://<username>.github.io/` 或 `https://<username>.github.io/<repo>/`）

## 博客发布约定

本章只记录**当前站点模板、博客索引页和 `scripts/validate_taxonomy.py` 真实依赖的约定**。此前偏“AI 写稿模板”的内容已移除，避免文档与系统实现脱节。

### 1. 发布路径与文件命名

- 草稿可放在 `_drafts/`，本地用 `jekyll serve --drafts --future` 预览。
- 正式发布文章放在 `_posts/`。
- 已发布文章文件名使用 `YYYY-MM-DD-<slug>.md`。
- `slug` 建议使用小写英文加连字符，例如 `high-dimensional-data-and-statistics`。
- 中文标题可以直接写在 `title`，无需为了 URL 强行改标题语言。

### 2. Front Matter 约定

每篇文章顶部都需要 YAML Front Matter：

```yaml
---
layout: blog-post
title: "文章标题"
date: 2026-02-27 20:00:00 +0800
categories: [数据科学]
tags: [Statistical Inference, Resampling]
author: Hyacehila
excerpt: "给列表卡片用的一句话摘要，不要写 Markdown"
series: "概率图模型 (Probabilistic Graphical Models)"
featured: true
math: true
---
```

当前系统实际依赖如下：

- `layout`: 固定为 `blog-post`。
- `title`: 用于页面标题与列表卡片标题；建议与正文第一个 `#` 保持一致。
- `date`: 用于排序、URL 与上一篇/下一篇；`_config.yml` 默认 `future: false`，未来日期不会发布。
- `categories`: 必须是单元素列表，且值只能是 `基础模型`、`训练与对齐`、`智能体系统`、`机器学习`、`数据科学`、`随笔与观察`。
- `tags`: 必须是列表；博客主页会按 Front Matter 自动生成标签筛选。当前校验脚本要求标签存在、标签非空且单个 tag 不含逗号。数量上建议控制在 2~4 个。
- `excerpt`: 建议使用单行纯文本；博客列表直接展示该字段。
- `series`: 可选；相同字符串会聚合到同一系列页，且同一系列文章应保持同一主分类。
- `featured`: 可选；`true` 时会出现在“得意之作”筛选视图。
- `math`: 可选；当前页面会全量加载 MathJax，这个字段保留语义用途。

### 3. 标签与 taxonomy

- 博客主页当前使用 6 个固定主分类按钮。
- 标签按钮不再依赖手写白名单，而是从所有文章的 `tags` 自动汇总并按自然顺序排序。
- 仍建议优先复用现有常见标签词汇，避免同义词重复造轮子；例如 `Agents`、`Tool Use`、`Context Engineering`、`Statistical Inference`、`Data Curation`。
- `统计学` 已不再作为主分类使用；相关内容统一归入 `数据科学`。

### 4. 正文与资源

- 正文第一行建议使用 `#` 标题，并与 `title` 保持一致。
- 代码块使用 fenced code block，并尽量标注语言。
- 数学公式使用 MathJax 语法；如果公式中需要条件竖线，优先写 `\mid`，避免和 Markdown 表格语法冲突。
- Mermaid 图直接使用 ```` ```mermaid ```` 代码块。
- 图片可以使用外链；如果需要稳定托管，放到 `assets/images/` 下并在文章中引用。

### 5. 发布前检查

建议在提交前至少完成以下检查：

- [ ] 文件位于 `_posts/`，文件名符合 `YYYY-MM-DD-<slug>.md`
- [ ] Front Matter 以 `---` 包裹且字段完整
- [ ] `categories` 为单元素列表，且属于当前 6 分类 taxonomy
- [ ] `tags` 为列表，且单个 tag 不含逗号
- [ ] `excerpt` 为适合列表展示的单行摘要
- [ ] `python scripts/validate_taxonomy.py` 通过
- [ ] 如本地具备 Ruby/Jekyll 环境，执行 `jekyll serve --drafts --future` 做渲染检查

## License

本项目基于 [MIT License](LICENSE) 开源。
