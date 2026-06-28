# Hyacehila — Personal Site & Blog (Hexo + Redefine)

> 个人主页 + 技术博客，基于 [Hexo](https://hexo.io/) 与 [Redefine](https://redefine.ohevan.com/) 主题。
> A personal site + tech blog, built with Hexo and the Redefine theme.

线上地址 / Live: https://hyacehila.github.io

## 站点结构

| 路径 | 说明 |
|------|------|
| `/` | Redefine 默认文章流（全部文章 + 分类 / 标签 / 搜索 / 侧栏） |
| `/archives/` | 时间线归档 |
| `/me/` | 正式自我介绍 |
| `/projects/` | 案例式项目介绍 |
| `/about/` | 杂物间：碎碎念 · CV · 足迹(Globe.gl) · 友链 · 联系方式 |
| `/categories/` `/tags/` | 分类 / 标签索引（不在顶栏，从文章卡片进入） |

旧文章外链保持不变：`/blog/:year/:month/:day/:title/`。

## 项目结构

```
_config.yml             # Hexo 站点配置
_config.redefine.yml    # Redefine 主题配置
package.json            # 依赖（含主题与插件）
source/
  _posts/               # 已发布文章
  _drafts/              # 草稿（不发布，但随仓库 git 同步）
  _data/projects.yml    # 项目数据（规范记录）
  me/ projects/ about/  # 个人页（含 categories/ tags/ 索引页）
  assets/
    css/ js/            # 自定义样式与脚本（i18n 切换、Globe.gl、murmur）
    data/               # 客户端读取的数据（cities.json, murmur.json）
    images/             # 文章图片资产
code/                   # 杂物：训练器代码片段、写作/环境笔记（不发布）
.github/workflows/      # GitHub Actions 自动构建 + 发布
migration/              # 一次性迁移脚本与校验工具（i18n 字典源等）
```

## 本地预览

```bash
npm install
npx hexo clean && npx hexo generate
npx hexo server        # http://localhost:4000
```

## 写作

```bash
npx hexo new post "标题"          # 新文章 -> source/_posts/
npx hexo new draft "标题"         # 新草稿 -> source/_drafts/（不发布）
npx hexo publish draft "标题"     # 草稿转正式
```

文章 front matter（Redefine 友好格式）：

```yaml
---
title: "标题"
date: 2026-06-25 10:00:00
categories: [随笔与观察]
tags: [Agents, Design]
excerpt: "摘要"
mathjax: false      # 数学渲染由 KaTeX 全站处理，此字段保留但不再依赖
---
```

- 数学公式：`$...$` 行内、`$$...$$` 块级，构建期由 `hexo-filter-katex` 渲染为静态 HTML。
- 流程图：` ```mermaid ` 代码块。
- 双语：个人页通过 `data-i18n="key"` 复用 `migration/home.i18n.source.json` 中的字典，右下角按钮切换 EN/中，选择持久化在 localStorage。

## 部署（GitHub Actions）

推送到 `master` 即触发 `.github/workflows/deploy.yml`：安装依赖 → `hexo generate` → 发布到 GitHub Pages。

> ⚠️ 一次性设置：GitHub 仓库 **Settings → Pages → Build and deployment → Source** 需从
> “Deploy from a branch” 改为 **“GitHub Actions”**，否则 Pages 仍按 Jekyll 构建。

## License

MIT，详见 [LICENSE](LICENSE)。
