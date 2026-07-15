# Hyacehila 个人网站与博客

这是一个基于 [Hexo](https://hexo.io/) 与 [Redefine](https://redefine.ohevan.com/) 主题构建的个人主页和技术博客，使用 GitHub Pages 托管。

- 在线地址：https://hyacehila.github.io
- 默认语言：英文界面，支持中英文切换
- 部署方式：GitHub Actions 构建并发布到 GitHub Pages
- 搜索方式：浏览器端 MiniSearch BM25 全文检索，不依赖后端服务

## 站点页面

| 路径 | 说明 |
| --- | --- |
| `/` | 首页 Banner、文章列表、分类、标签、侧栏和搜索入口 |
| `/archives/` | 按时间排列的全部文章归档 |
| `/categories/` | 分类索引 |
| `/tags/` | 标签索引 |
| `/me/` | 个人介绍、经历、研究方向和荣誉 |
| `/projects/` | 项目与案例展示 |
| `/murmur/` | 碎碎念时间线 |
| `/footprints/` | 基于 Globe.gl 的旅行足迹页面 |
| `/friends/` | 友情链接 |
| `/comments/` | 留言页面 |
| `/cv/` | 简历预览与下载 |
| `/contact/` | 邮箱、GitHub 和 LinkedIn 等联系方式 |

顶部导航包含 `Home`、`Archives`、`Me`、`Project` 和 `About`。其中 `About` 下拉菜单包含旅行足迹、友情链接、留言和简历页面。

旧文章链接格式保持不变：

```text
/blog/:year/:month/:day/:title/
```

## 项目结构

```text
_config.yml                 # Hexo 站点配置
_config.redefine.yml        # Redefine 主题、导航、搜索和资源注入配置
package.json                # Node.js 依赖与构建命令
scripts/
  search-index.js           # 生成 MiniSearch BM25 静态索引
  search-ui.js              # 替换主题搜索弹窗模板
  validate-search.js        # 搜索完整性、质量和性能校验
  validate-i18n.js          # 中英文内容与分类标签校验
source/
  _posts/                   # 已发布文章
  _drafts/                  # 草稿，不参与正式构建
  _data/
    essays.yml              # 碎碎念数据
    links.yml               # 友情链接数据
    projects.yml            # 项目数据
    search-aliases.yml      # 搜索同义词和别名
  assets/
    css/                    # 自定义样式
    js/                     # i18n、搜索 Worker 和其他客户端脚本
    data/                   # 浏览器端读取的静态数据
    images/                 # 图片资源
  me/ projects/             # 个人介绍与项目页面
  murmur/ footprints/       # 碎碎念与旅行足迹页面
  friends/ comments/        # 友情链接与留言页面
  cv/ contact/              # 简历与联系方式页面
.github/workflows/
  deploy.yml                # GitHub Pages 自动构建和部署流程
public/                     # Hexo 生成的静态站点，不手动编辑
```

## 本地开发

推荐使用 Node.js 24，与 GitHub Actions 的构建环境保持一致。

安装依赖：

```bash
npm ci
```

启动 Hexo 开发服务器：

```bash
npm run server
```

默认访问地址：

```text
http://localhost:4000/
```

如果需要按照 GitHub Pages 的最终静态产物进行预览：

```bash
npm run clean
npm run build
npx hexo server --static -p 4173
```

然后访问：

```text
http://localhost:4173/
```

## 构建与校验

```bash
npm run clean          # 删除 Hexo 缓存和 public 目录
npm run build          # 生成静态站点并执行搜索质量校验
npm run check:i18n     # 校验中英文内容、分类和标签
npm run check:search   # 单独校验已生成的搜索索引
```

`npm run build` 不只是生成页面，还会验证搜索索引的完整性、排序质量、分类标签召回率和查询性能。校验失败时构建会直接失败，避免把不可用的搜索功能部署到 GitHub Pages。

## 写作流程

创建正式文章：

```bash
npx hexo new post "文章标题"
```

创建草稿：

```bash
npx hexo new draft "文章标题"
```

发布草稿：

```bash
npx hexo publish draft "文章标题"
```

文章 Front Matter 示例：

```yaml
---
title: "中文标题"
title_en: "English title"
date: 2026-06-25 10:00:00
categories:
  - Work & Society
  - Builder & Product Thinking
tags:
  - Builder Mindset
  - Product Thinking
  - Software Engineering
excerpt: "中文摘要"
excerpt_en: "English excerpt"
hidden: false
---
```

### 写作约定

- 正文开头不要重复编写一级标题，文章标题只放在 Front Matter 的 `title` 字段中。
- `title_en` 和 `excerpt_en` 用于英文界面的首页、归档、分类、标签和搜索结果。
- `hidden: true` 只会让文章从首页列表隐藏，文章链接、归档、分类、标签和全文搜索仍然保留。
- 分类最多使用两级结构，并优先复用 `_config.yml` 中已有的分类映射。
- 数学公式使用 `$...$` 和 `$$...$$`，构建时由 KaTeX 转换为静态 HTML。
- Mermaid 图表使用标记为 `mermaid` 的代码块。
- 提交前建议运行 `npm run check:i18n` 和 `npm run build`。

## 内容数据

更新碎碎念：

```text
source/_data/essays.yml
```

每条记录主要包含 `content` 和 `date` 字段。

更新友情链接：

```text
source/_data/links.yml
```

数据按 `links_category` 和 `list` 组织，每个链接可配置 `name`、`link`、`description` 和 `avatar`。

更新项目数据：

```text
source/_data/projects.yml
```

## 静态全文搜索

搜索系统完全运行在浏览器中，兼容 GitHub Pages，不需要数据库、搜索服务器或后端 API。

### 索引内容

搜索索引覆盖以下字段：

- 中英文文章标题
- 正文标题和章节标题
- 分类与标签
- 中英文摘要
- 清洗后的正文段落
- 代码块中的技术词汇
- `source/_data/search-aliases.yml` 中配置的同义词

### 检索流程

1. `scripts/search-index.js` 在 Hexo 构建阶段读取文章内容。
2. 使用 MiniSearch 7.2.0 生成 BM25 索引和文档元数据。
3. 索引按照核心文章和历史归档拆分为两个静态分片。
4. 浏览器打开搜索框后，由 Web Worker 加载核心索引。
5. 核心结果可以先展示，历史归档索引在后台继续加载。
6. 查询结果按照标题、章节、标签、分类、摘要和正文命中情况进行加权排序。

生成文件位于：

```text
public/assets/search/
public/assets/vendor/minisearch-7.2.0.js
```

浏览器端代码位于：

```text
source/assets/js/search-worker.js
source/assets/js/search-engine.js
source/assets/js/search-tokenizer.js
source/js/build/tools/localSearch.js
```

### 搜索质量门禁

`npm run check:search` 会检查：

- 索引和文档分片是否存在且与 Manifest 一致
- MiniSearch 浏览器 Bundle 是否是有效 JavaScript
- 非法控制字符是否已清理
- 精确标题查询的 MRR@10 和 Top-3 排名
- 分类与标签召回率
- 人工维护的关键词评测集
- 常见错误子串匹配边界
- 暖态查询 P95 延迟
- 单个压缩索引分片是否超过体积限制

## 中英文切换

站点界面默认使用英文，并提供中英文切换功能。语言选择保存在浏览器 `localStorage` 中，并兼容 Redefine 的 Swup 单页导航。

主要实现文件：

```text
source/assets/js/i18n.js
source/assets/js/lang-toggle.js
scripts/validate-i18n.js
```

文章正文可以只提供中文，但建议为文章补充 `title_en` 和 `excerpt_en`，保证英文界面的文章列表与搜索结果具有良好可读性。

## 部署到 GitHub Pages

推送到 `master` 分支后，`.github/workflows/deploy.yml` 会自动执行：

1. 检出仓库和 Git LFS 文件。
2. 安装 Node.js 24。
3. 使用 `npm ci` 安装依赖。
4. 清理并生成 Hexo 静态站点。
5. 执行搜索质量门禁。
6. 上传 `public/` 目录并部署到 GitHub Pages。

仓库需要在 GitHub 中进行一次性配置：

```text
Settings → Pages → Build and deployment → Source → GitHub Actions
```

## 许可证

本项目使用 [MIT License](LICENSE)。