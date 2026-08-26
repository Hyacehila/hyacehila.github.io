---
title: 关于我
title_en: Me
date: 2026-06-28 00:00:00
permalink: /me/
comments: false
---

<div class="me-page">
<section class="me-section me-about-section" aria-labelledby="me-about-title">
  <h2 id="me-about-title" class="me-section-title"><span class="me-section-icon" aria-hidden="true"><i class="fa-solid fa-seedling"></i></span><span>关于我</span></h2>
  <div class="me-prose">
    <p>hyacehila 是我长期使用的网络 ID。它最早来自我喜欢的植物 hyacinth，后来被我改成了一个更轻、更像名字的形式。<code>-hila / -ila</code> 对我来说有一点虚构生物的感觉，所以我有时把 hyacehila 想象成一只生活在复杂结构里的风信子精灵：在系统、工具链、工作流和不确定环境之间穿行，试着找到一条自然、可解释，而且真正能解决问题的路径。你也可以叫我 Julian 或 Jules。</p>
    <p>这也很接近我理解技术和工程的方式。我不太相信一个技术从出现的那一刻起，就已经知道自己应该被放在哪里。很多时候，真正有意义的位置需要经过一段时间才能找到：先定义问题，形成一个足够明确、可以被证伪的判断，然后把它放进现实里，观察失败、吸收反馈、修正假设，再继续迭代。相比从某一种固定的 Agent 架构出发，我更习惯先拆解业务：哪些地方真正需要模型判断，哪些地方应该交给 Workflow、Retrieval、Tool 或外部验证来保证确定性，再据此设计适合问题本身的 AI 系统。对我来说，好的工程不是消除所有不确定性，而是知道应该把不确定性留在哪里，又该怎样让它持续接受现实的反馈。</p>
    <p>最近我主要关注知识系统、AI Agent 在金融与游戏研发/发行中的应用、长期记忆，以及 AI NPC 与 World Simulation。我尤其感兴趣的是模型如何从一次性的生成工具，逐渐成为能够理解环境、积累状态、行动、接受反馈并长期协作的系统。这个博客记录的也不只是技术教程，而是我在 Agent Architecture、Context Engineering、Evaluation、AI Native Game 等问题上不断形成又修正的判断——有些会成为工程实践，有些会被后来新的经验推翻。写作对我来说，是保存这段迭代过程的一种方式。商业小说则是另一项长期兴趣。</p>
  </div>
</section>

<section class="me-section" aria-labelledby="me-intern-title">
  <h2 id="me-intern-title" class="me-section-title"><span class="me-section-icon" aria-hidden="true"><i class="fa-solid fa-briefcase"></i></span><span>实习经历</span></h2>
  <div class="me-entry-list">
    <article class="me-entry">
      <h3>AI Agent 研发工程师（实习）</h3>
      <p class="me-entry-company">网易互娱（上海） · 2026 年 6 月—2026 年 8 月 · <a href="/blog/2026/08/08/ui-pipeline-automation-thinking/" data-i18n-preserve-label>实习复盘</a></p>
      <ul class="me-detail-list">
        <li><strong>项目背景与目标：</strong>开发基于 Agent 的 UI 工程自动生成系统，减少把游戏 UI 设计稿转成可上线工程时的重复搭建工作，降低流程对个人开发经验的依赖，并探索 Agent 在更完整 UI 工作流中的作用。</li>
        <li><strong>UI Agent 工作流：</strong>独立设计并实现混合架构，将确定性的 Agent Workflow 与基于 OpenAI Agents SDK 的自主节点结合起来。系统可以从 Figma/PSD 设计稿生成 NeoX <code>.uiprefab</code> 工程，同时支持从零生成和基于蓝图生成。整套流程约有 50 个节点，其中 26 个在不同流程间复用。LLM 只处理结构推断、素材选择等确实需要模型判断的任务，确定性工作流负责兜底；多模态检索与项目规范注入则让生成结果遵循既有工程规范，并尽量复用已有控件。蓝图模式能够复用历史工程。我与 GUI 设计师协作整理了约 10 套商业蓝图，覆盖周签到、商店弹窗等常见界面。部分 B/C 类 UI 任务由约 0.5—1.5 人日缩短到 30 分钟，已有 4 个生成工程上线。</li>
        <li><strong>UI Agent 评测与回归测试：</strong>建立“启发式规则筛查、Ground Truth 对比、人工终审”三层校验流程，让 Agent 生成结果更稳定，也更容易验证。使用 AI 从已有工程中挖掘结构聚类、子 Prefab 候选等模式，再按照 UI 开发规范校准，并通过多轮 Skill Demo 把这些模式加入工作流。针对 10 余个真实 UI 样本，用启发式评测和编辑器检查持续迭代，同时为蓝图生成和从零生成的基础能力补充回归测试。一次蓝图中间件迁移中，测试发现了字段与脚本遗漏；这些遗漏会导致整体画面比例失真。</li>
        <li><strong>内部知识库：</strong>针对类型复杂的内部文档和游戏领域知识，尝试不同的知识建模与检索方案。对已分类、跨多种文件格式的文档，使用 Docling 建立统一的结构化事实层，并补充图表信息，再结合向量检索、BM25 和 PageIndex 风格的结构化检索。对游戏知识，则从源码和配置表中抽取约 20,000 个实体节点与 30,000 条程序关系，整理成 JSON Wiki，以处理策划文档过期和内容冲突的问题。在此基础上，用 LLM Wiki 风格的方法和查询改写构建 Agentic RAG，在测试集上实现全部问题召回。</li>
      </ul>
    </article>
    <article class="me-entry">
      <h3>算法研究员（实习）</h3>
      <p class="me-entry-company">绿盟科技（武汉） · 2025 年 12 月—2026 年 3 月</p>
      <ul class="me-detail-list">
        <li><strong>漏洞挖掘 Agent 与 CodeQL 验证循环：</strong>复杂代码漏洞分析中，LLM 容易被早期判断锚定，之后给出不可靠的结论。针对这个问题，我搭建了用于漏洞挖掘的 Single-Agent harness，把漏洞情报检索、源码定位、污点流建模、CodeQL 查询生成和引擎验证组织成可追踪的多轮分析循环。系统重点处理状态表示、工具调用协议和验证反馈，而不是预先拆成多个角色。模型可以围绕候选 source/sink 对、失败路径、工具输出和 CodeQL 验证结果持续迭代，从而减少长程分析中的路径依赖与误报。</li>
        <li><strong>训练数据与 Agent 轨迹整理：</strong>基于开源项目和内部数据库，为漏洞挖掘 Agent 的后训练搭建漏洞数据清洗与标注流程。抽取并验证 8,000 余个 CVE 实体，收集 4,000 条高质量样本，按漏洞类型和编程语言标注，并平衡训练集构成。从 Agent 执行记录中蒸馏出 2,500 条高置信度工具使用 SFT 轨迹和 300 条污点流记录，用于评测集构建、奖励设计及后续 RL 训练探索。</li>
      </ul>
    </article>
  </div>
</section>

<section class="me-section" aria-labelledby="me-research-title">
  <h2 id="me-research-title" class="me-section-title"><span class="me-section-icon" aria-hidden="true"><i class="fa-solid fa-microscope"></i></span><span>研究</span></h2>
  <ul class="me-research-list">
    <li>
      <span class="me-paper-title">Unveiling the Drivers of PTSD: An Interpretable Machine Learning Approach with SHAP</span>
      <span class="me-paper-venue">International Conference on Intelligent Computing and Data Analysis 2025 ; EI</span>
      <span class="me-link-row">
        <a href="https://doi.org/10.1145/3772726.3772849" target="_blank" rel="noopener">DOI</a>
      </span>
    </li>
  </ul>
</section>

<section class="me-section" aria-labelledby="me-awards-title">
  <h2 id="me-awards-title" class="me-section-title"><span class="me-section-icon" aria-hidden="true"><i class="fa-solid fa-trophy"></i></span><span>奖项与证书</span></h2>
  <ul class="me-awards-list">
    <li>全国大学生统计建模大赛陕西省一等奖</li>
    <li>SAS 中国高校数据分析大赛全国三等奖</li>
    <li>美国大学生数学建模竞赛（MCM/ICM）二等奖</li>
    <li>大学英语四级：510｜大学英语六级：513</li>
  </ul>
</section>
</div>
