# Hyacehila — Personal Site & Blog (Hexo + Redefine)

> 个人主页 + 技术博客，基于 [Hexo](https://hexo.io/) 与 [Redefine](https://redefine.ohevan.com/) 主题。
> A personal site + tech blog, built with Hexo and the Redefine theme.

线上地址 / Live: https://hyacehila.github.io

## 站点结构

| 路径 | 说明 |
|------|------|
| `/` | 固定封面 banner（山水图 + 高斯模糊，标题为 ID）+ Redefine 默认文章流（分类 / 标签 / 搜索 / 侧栏） |
| `/archives/` | 时间线归档（不在 Home 的历史文章） |
| `/me/` | 正式自我介绍：教育/经历 Timeline · 研究枚举 · 奖项 |
| `/projects/` | 案例式项目介绍 |
| **About（顶栏下拉）** | 下拉菜单，包含以下子页： |
| &nbsp;&nbsp;`/murmur/` | 碎碎念（Redefine 原生「说说」时间线） |
| &nbsp;&nbsp;`/footprints/` | 旅行足迹（Globe.gl 3D 地球） |
| &nbsp;&nbsp;`/friends/` | 友情链接（Redefine 原生「友链」模板） |
| &nbsp;&nbsp;`/cv/` | CV（PDF 预览 + 下载，缺文件时优雅占位） |
| &nbsp;&nbsp;`/contact/` | 邮箱 / GitHub / LinkedIn |
| `/categories/` `/tags/` | 分类 / 标签索引（不在顶栏，从文章卡片进入） |

顶栏：**Home / Archives / Me / Project / About（下拉）**。
旧文章外链保持不变：`/blog/:year/:month/:day/:title/`。

## 项目结构

```
_config.yml             # Hexo 站点配置
_config.redefine.yml    # Redefine 主题配置（banner / 导航下拉 / inject 等）
package.json            # 依赖（含主题与插件）
source/
  _posts/               # 已发布文章（正文不含开头 # 标题，标题只在 front-matter）
  _drafts/              # 草稿（不发布，但随仓库 git 同步）
  _data/
    essays.yml          # 碎碎念数据（说说模板消费）
    links.yml           # 友链数据（友链模板消费）
    projects.yml        # 项目数据（规范记录）
  me/ projects/         # 个人页（结构化 HTML + data-i18n）
  murmur/ footprints/   # About 下拉子页
  friends/ cv/ contact/
  categories/ tags/     # 分类 / 标签索引页
  assets/
    css/ js/            # 自定义样式与脚本（i18n 切换、Globe.gl、CV 占位）
    data/               # 客户端读取的数据（cities.json）
    images/             # 文章图片资产
code/                   # 杂物：训练器代码片段、写作/环境笔记（不发布）
.github/workflows/      # GitHub Actions 自动构建 + 发布
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
title_en: "English title"   # 切到 EN 时列表/归档/分类/标签里显示的英文标题
date: 2026-06-25 10:00:00
categories: ["Essays"]
tags: [Agents, Design]
excerpt: "摘要"
excerpt_en: "English excerpt"  # 切到 EN 时首页卡片显示的英文摘要
mathjax: false      # 数学渲染由 KaTeX 全站处理，此字段保留但不再依赖
archived: false     # true => 不在 Home 显示，进入 Archives；Categories/Tags 保持全量可发现
---
```

约定：
- **不要在正文开头写 `# 标题`**；标题只写在 front-matter 的 `title:`，主题会渲染页面标题。
- **归档/隐藏首页**：给文章加 `archived: true`，它就不再出现在 Home 文章流，而会进入 `/archives/`；分类、标签仍保持全量索引，单篇链接也不变。
- **语言策略**：英文是系统 UI 默认语言；博客正文、碎碎念等可以作为中文内容岛保留。列表/归档/分类/标签里的文章标题与首页摘要会用 `title_en`/`excerpt_en`；个人页用 `data-i18n`；导航、侧栏、页脚、工具入口等 UI 文案跟随语言切换。语言入口位于右侧小齿轮工具栏，选择持久化在 localStorage。
- **i18n 校验**：提交前运行 `npm run check:i18n`，检查 `data-i18n` key、文章英文标题/摘要、英文默认 UI 中文残留和 CJK permalink 映射。
- 数学公式：`$...$` 行内、`$$...$$` 块级，构建期由 `hexo-filter-katex` 渲染为静态 HTML。
- 流程图：` ```mermaid ` 代码块。

### 更新碎碎念 / 友链

- 碎碎念：编辑 `source/_data/essays.yml`（字段 `content` + `date`）。
- 友链：编辑 `source/_data/links.yml`（`links_category` → `list`，每项 `name/link/description/avatar`）。

## 部署（GitHub Actions）

推送到 `master` 即触发 `.github/workflows/deploy.yml`：安装依赖 → `hexo generate` → 发布到 GitHub Pages。

> ⚠️ 一次性设置：GitHub 仓库 **Settings → Pages → Build and deployment → Source** 需从
> “Deploy from a branch” 改为 **“GitHub Actions”**，否则 Pages 仍按 Jekyll 构建。

## License

MIT，详见 [LICENSE](LICENSE)。
