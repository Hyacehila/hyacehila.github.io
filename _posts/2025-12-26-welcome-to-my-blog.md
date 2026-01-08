---
layout: blog-post
title: 第一篇博客（An Example）
date: 2025-12-26 19:00:00 +0800
categories: [随笔]
tags: [Tech]
author: Hyacehila
excerpt: 这是博客的第一篇文章，也是后续写作规范的一个示例与说明。
---

# 第一篇博客（An Example）

这是一篇示例文章：一方面作为博客的开篇，另一方面也用来记录后续写作的一些约定。

## 项目背景

2025 年 12 月 26 日， Claude Code、GLM4.7 与 Cursor（Opus 4.5）完成了站点组件的搭建。之后只要编写 `.md` 文档，Jekyll 就能将其渲染为 HTML。

## 关于这个博客

这里主要记录技术笔记、项目复盘，也会偶尔写点随想。

## 技术栈

- **Jekyll**：静态站点生成器
- **Markdown**：内容编写格式

## 如何写一篇新的文章

1. 在 `_posts/` 下新建文件，命名为 `YYYY-MM-DD-title.md`。
2. 在文件顶部添加 YAML Front Matter（如下）。`tags` 和`categories`会被筛选系统使用；二级标题（`##`）建议用于组织正文结构。
3. 正文直接使用 Markdown 编写：支持代码块语法高亮、常见 LaTeX 公式与 Mermaid 图表。
   - 公式里尽量不要直接使用 `|` 作为分隔符；需要“竖线”时用 `\mid`，避免被误判为表格。

```yaml
---
layout: blog-post
title: 欢迎来到我的博客
date: 2025-12-26 10:00:00 +0800
series: 不知道在写什么
categories: [生活, 随笔]
tags: [欢迎, 博客]
author: Hyacehila
excerpt: 这是我博客的第一篇文章，欢迎大家来访！
---
```
