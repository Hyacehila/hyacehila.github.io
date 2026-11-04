# 项目视觉卡片 Demo

## 设计与范围

北极星：保留原站柔和的玻璃层与蓝灰色调，让项目封面成为第一阅读层，文字退到清晰、可展开的介绍。

用户路径：浏览封面 → 识别项目与问题 → 在段落末尾展开介绍，或点击图片访问真实仓库。

仅修改中英文项目页、项目区样式与共享词典中的 gallery 键，添加五张项目封面和仅在项目页加载的展开脚本。未调整文章、导航、部署或依赖。主分支原有四个未跟踪文件保存在 `Before project gallery demo 2026-11-03` stash，开发位于 `codex/project-gallery-demo`。

## 布局与交互

- 容器宽度超过 850px 时三列，531–850px 两列，530px 及以下一列。按可用容器宽度换列，避免侧边空白挤压内容。
- 桌面封面 3:2，手机 16:9；标题完整换行，简介默认展示第一句概述，段落末尾的原生 button 展开全文。展开只增加当前卡片高度，同排卡片保持顶部对齐。
- 外链仅包裹图片，保留新标签页和键盘语义。标题、正文与卡片留白不跳转；只有段末按钮展开与收起。无 JS 时直接展示完整介绍，隐藏无效按钮。
- 只有 AnalysisPosts、Novel Evaluation 有已提供的 GitHub 地址；其余保留静态图片，不伪造地址。移除卡片底部跳转提示、关键词、编号与项目计数。
- 卡片 hover / focus-within 上浮 5px、边框提亮、阴影加深；鼠标 hover 时封面放大 2.5%。减少动态效果时禁用位移和过渡；触屏不依赖 hover。
- 没有添加 React、Tailwind、shadcn 或新运行依赖。参考其组件分层与响应式思路，以现有 Hexo 主题和原生 HTML/CSS 实现。

## 配图

五张封面均由 OpenAI imagegen 在本任务生成，用作概念配图，不是项目真实截图。原始文件保留在本机 Codex generated_images 目录，站点使用 `source/assets/images/projects/` 内 960×640 WebP，单张约 50–78 KB，总计 312272 字节。

统一提示方向：纸张、象牙白、磨砂玻璃、低饱和蓝灰、少量黄铜，柔和自然光，无文字/标志。各图主题分别是知识档案与罗盘、三方协作的证据透镜、小说与批注、可穿戴设备与脉搏、城市与统计数据。

## 参考

- https://ui.shadcn.com/docs/components/base/card
- https://ui.shadcn.com/docs/components/radix/collapsible
- https://tailwindcss.com/docs/grid-template-columns
- https://tailwindcss.com/docs/responsive-design

## 首版验收记录（精简版另见下方）

最终 `npm run build` 成功，包含 i18n、搜索、SEO、publication 四项检查。浏览器 38 项断言连续三轮通过：1440/1024/768/390/320px 的列数、无横向溢出、封面加载、全部五张卡片展开与 Enter 收起、完整正文可见、展开不跳转、封面打开准确的仓库地址、无伪造外链、英文长文案窄幅排版、减少动态效果以及无 JavaScript 的内容与展开。另行验证 CSS 200% 放大，无横向溢出且介绍可展开。

外链测试拦截 GitHub 请求来验证打开的新窗口地址，不宣称验证了远端仓库服务可用性。英文截图是将现有英文词典应用到卡片的排版压力测试，不代表新增英文路由或更改全站语言切换行为。没有修改既有语言路由策略。

独立视觉复核一轮（存量界面、不打分）：卡片区没有明显重叠、溢出或布局阻断；保留配图材质、简短摘要和留白。评审指出英文压力截图页标题仍为中文，这是只替换词典内容的测试方式导致的，并非新上线英文页面。减法保留项目名、单段介绍和真实外链状态，不恢复长篇技术字段，未发现需要进一步删除的填充文案。

本机截图与日志归档在 `.codex-artifacts/project-gallery/`，核心截图为 `screenshots/projects-1440.png`、`projects-390.png`、`projects-dark.png`、`projects-zoom-200.png`。可重复验证脚本在 `tools/tests/project-gallery.playwright.js`。运行前启动本地静态预览并创建 `output/playwright`，再用 `npx --yes --package @playwright/cli playwright-cli open http://localhost:4000/projects/` 和 `npx --yes --package @playwright/cli playwright-cli run-code --filename=tools/tests/project-gallery.playwright.js` 执行。脚本返回断言列表；任何失败会抛出异常。

首轮发现主题文章标题和图片样式侵入卡片、图片灯箱截获点击；通过主题已有 not-markdown 标记和局部 CSS 解除冲突。未修改全站文章样式。

## 精简版（2026-11-04）

按用户反馈，移除所有关键词、编号、项目计数与底部外链/未公开提示。简介以首句准确描述项目问题，展开按钮紧随段落，完整介绍只在点击按钮后显示。图片成为唯一外链入口；无地址的图片保持静态。新增脚本只负责段落显隐与 aria-expanded 状态，不改变导航或全站行为。

悬浮参考 https://tailwindcss.com/docs/hover-focus-and-other-states 与 https://tailwindcss.com/docs/transition-property ，采用克制的上浮、阴影、边框和图片缩放。

本轮完整构建及 i18n、搜索、SEO、publication 检查通过；浏览器 47 项断言通过，覆盖五档宽度、图片唯一跳转入口、正文点击不展开、按钮展开与键盘收起、整体卡片 hover、英文排版、减少动态效果及无 JS 完整正文降级。主 Agent 检查桌面、手机与展开截图，确认段末按钮排版及卡片悬浮效果。截图与日志保存在 `.codex-artifacts/project-gallery-refined/`。
