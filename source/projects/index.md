---
title: 项目
title_en: Projects
date: 2026-06-28 00:00:00
permalink: /projects/
comment: false
comments: false
---

<section class="projects-page project-gallery" aria-label="Projects">
<div class="gallery-grid">
<article class="gallery-card" style="--card-order:0">
<div class="gallery-cover"><img class="not-markdown" src="/assets/images/projects/lenny-compass.webp" alt="" width="960" height="640" loading="lazy" decoding="async"></div>
<div class="gallery-body">
<h2 class="gallery-title not-markdown" id="gallery-lenny-compass-title"><span data-i18n="gallery-lenny-compass-title">Lenny Compass</span></h2>
<p class="gallery-description" id="gallery-lenny-compass-description"><span class="gallery-preview" hidden></span><span class="gallery-full" data-i18n="gallery-lenny-compass-desc">帮助读者从 Lenny 的文章与播客中找到产品问题的答案，并核对每条结论的原始依据。Lenny Compass 把提问、资料检索、原文阅读和引用检查放进同一个本地工作区，减少在零散内容中反复翻找的时间。简单问题可以快速检索，复杂问题由智能体多轮查找和筛选证据；回答生成后，再检查结论是否得到引用支持，修复或省略缺少依据的内容。读者可以展开引用附近的原文，区分资料中的结论与模型建议，也能保存会话、切换模式和继续追问，让一次查询积累成可以回看的研究记录。</span> <button class="gallery-toggle" type="button" aria-expanded="true" aria-controls="gallery-lenny-compass-description" hidden><span class="gallery-more" data-i18n="gallery-more">展开</span><span class="gallery-less" data-i18n="gallery-less">收起</span></button></p>
</div></article>
<article class="gallery-card" style="--card-order:1">
<a class="gallery-cover gallery-link not-markdown" href="https://github.com/Hyacehila/AnalysisPosts" target="_blank" rel="noopener noreferrer" aria-labelledby="gallery-analysisposts-title"><img class="not-markdown" src="/assets/images/projects/analysisposts.webp" alt="" width="960" height="640" loading="lazy" decoding="async"></a>
<div class="gallery-body">
<h2 class="gallery-title not-markdown" id="gallery-analysisposts-title"><span data-i18n="gallery-analysisposts-title">AnalysisPosts</span></h2>
<p class="gallery-description" id="gallery-analysisposts-description"><span class="gallery-preview" hidden></span><span class="gallery-full" data-i18n="gallery-analysisposts-desc">把帖子中的统计规律与外部事实放在一起分析，生成有来源、可追溯的舆情报告。AnalysisPosts 通过多智能体协作完成数据准备、分析和报告生成：统计分析智能体处理标注数据与量化工具，检索智能体补充外部事实，调度中枢组织多轮补证、交叉质疑和冲突检测，减少单一证据来源造成的片面判断。报告中的核心结论绑定来源与证据，再由核查智能体复查引用。使用者还可以在可视化界面中查看任务步骤、工具输入输出和调度路径，追查某条结论如何形成，或定位分析过程中的异常。</span> <button class="gallery-toggle" type="button" aria-expanded="true" aria-controls="gallery-analysisposts-description" hidden><span class="gallery-more" data-i18n="gallery-more">展开</span><span class="gallery-less" data-i18n="gallery-less">收起</span></button></p>
</div></article>
<article class="gallery-card" style="--card-order:2">
<a class="gallery-cover gallery-link not-markdown" href="https://github.com/Hyacehila/novel_evaluation" target="_blank" rel="noopener noreferrer" aria-labelledby="gallery-novel-eval-title"><img class="not-markdown" src="/assets/images/projects/novel-eval.webp" alt="" width="960" height="640" loading="lazy" decoding="async"></a>
<div class="gallery-body">
<h2 class="gallery-title not-markdown" id="gallery-novel-eval-title"><span data-i18n="gallery-novel-eval-title">Novel Evaluation</span></h2>
<p class="gallery-description" id="gallery-novel-eval-description"><span class="gallery-preview" hidden></span><span class="gallery-full" data-i18n="gallery-novel-eval-desc">为小说提供标准明确、分项可查的评测报告，帮助读者理解评分依据，并在不同作品之间复用评测方案。这套本地大模型评测系统会先筛查输入、选择分析模式并识别作品类型，再按八个维度评分，补充针对作品类型的评估，检查结果的一致性后汇总报告。使用者可以查看各项判断、调整提示词和评测标准，并通过历史记录比较结果。系统还支持任务创建、批量评测、输出格式校验与回归检查，便于反复检验方案是否稳定。评测的重点是呈现可以讨论和修改的判断依据，让总分背后的取舍有迹可循。</span> <button class="gallery-toggle" type="button" aria-expanded="true" aria-controls="gallery-novel-eval-description" hidden><span class="gallery-more" data-i18n="gallery-more">展开</span><span class="gallery-less" data-i18n="gallery-less">收起</span></button></p>
</div></article>
<article class="gallery-card" style="--card-order:3">
<div class="gallery-cover"><img class="not-markdown" src="/assets/images/projects/mental-health.webp" alt="" width="960" height="640" loading="lazy" decoding="async"></div>
<div class="gallery-body">
<h2 class="gallery-title not-markdown" id="gallery-mental-health-title"><span data-i18n="gallery-mental-health-title">可穿戴心理健康研究</span></h2>
<p class="gallery-description" id="gallery-mental-health-description"><span class="gallery-preview" hidden></span><span class="gallery-full" data-i18n="gallery-mental-health-desc">探索用可穿戴设备的日常记录预测心理健康状态，为阶段性问卷之外的状态观察提供数据依据。项目将生理与行为信号整理成可用于建模的数据，从四年观测记录中筛选四十三个变量，构建约三万行的多时间尺度数据集，处理缺失值、离群值和偏态分布。随后建立线性回归、随机森林与梯度提升模型，通过集成学习结合预测结果，并用网格搜索调整超参数，形成从原始记录到状态预测的完整流程。这项研究用于分析可穿戴信号与心理健康状态的关系，项目获得国家级大学生创新创业训练计划支持，于二〇二四年五月结题。</span> <button class="gallery-toggle" type="button" aria-expanded="true" aria-controls="gallery-mental-health-description" hidden><span class="gallery-more" data-i18n="gallery-more">展开</span><span class="gallery-less" data-i18n="gallery-less">收起</span></button></p>
</div></article>
<article class="gallery-card" style="--card-order:4">
<div class="gallery-cover"><img class="not-markdown" src="/assets/images/projects/health-metrics.webp" alt="" width="960" height="640" loading="lazy" decoding="async"></div>
<div class="gallery-body">
<h2 class="gallery-title not-markdown" id="gallery-health-metrics-title"><span data-i18n="gallery-health-metrics-title">中国居民健康测度</span></h2>
<p class="gallery-description" id="gallery-health-metrics-description"><span class="gallery-preview" hidden></span><span class="gallery-full" data-i18n="gallery-health-metrics-desc">把分散的居民健康数据整理成可比较、可解释的综合指标，并分析与健康水平相关的影响因素。项目整合国家统计局与商业数据库中的健康指标，结合医疗研究主题的变化信息，用主成分分析梳理指标间的关联，根据实际含义选取两个主要成分，减少单项指标难以反映整体状况的问题。再通过稀疏回归筛选相关因素，利用网格搜索平衡模型拟合程度与变量数量，为后续比较和分析提供依据。最终模型拟合优度约为零点八九，项目获得全国大学生统计建模大赛赛区一等奖。结果用于解释数据中的关联，供进一步研究与检验。</span> <button class="gallery-toggle" type="button" aria-expanded="true" aria-controls="gallery-health-metrics-description" hidden><span class="gallery-more" data-i18n="gallery-more">展开</span><span class="gallery-less" data-i18n="gallery-less">收起</span></button></p>
</div></article>
</div></section>

<script defer src="/assets/js/project-gallery.js"></script>
