---
title: Me
date: 2026-06-28 00:00:00
permalink: /me/
comments: false
---

<div class="me-page">
<section class="me-section me-about-section" aria-labelledby="me-about-title">
  <h2 id="me-about-title" class="me-section-title"><span class="me-section-icon" aria-hidden="true"><i class="fa-solid fa-seedling"></i></span><span>About Me</span></h2>
  <div class="me-prose">
    <p>hyacehila is my long-term online ID. It comes from hyacinth, my favorite plant. I later reshaped it into a lighter, more name-like form: hyacehila. The ending -hila / -ila gives it a small, airy, fictional texture, so to me it is not only a username, but also a hyacinth sprite living inside complex structures.</p>
    <p>This ID is close to how I understand technology: moving through dense systems, toolchains, workflows, and uncertain environments to find a natural, explainable path that actually solves the problem. I care about general problem-solving patterns and about whether technology can transfer and generalize across real scenarios.</p>
    <p>Today I mainly focus on AI Agent deployment and Evaluation, which I see as two of the most important technologies for bringing large models into engineering practice and industrial pipelines. I study both Single-Agent and Multi-Agent systems: how they are designed, evaluated, constrained, and eventually embedded in real business workflows rather than left at the benchmark level. Writing commercial fiction is a side interest.</p>
    <p>You can call me Julian or Jules.</p>
  </div>
</section>

<section class="me-section" aria-labelledby="me-intern-title">
  <h2 id="me-intern-title" class="me-section-title"><span class="me-section-icon" aria-hidden="true"><i class="fa-solid fa-briefcase"></i></span><span>Intern</span></h2>
  <div class="me-entry-list">
    <article class="me-entry">
      <h3>AI Agent R&amp;D Engineer (Intern)</h3>
      <p class="me-entry-company">NetEase Interactive Entertainment (Shanghai) · June 2026 -- August 2026 · <a href="/blog/2026/08/08/ui-pipeline-automation-thinking/" data-i18n-preserve-label>Internship Reflection</a></p>
      <ul class="me-detail-list">
        <li><strong>Project Background &amp; Objective:</strong> Developed an agent-based system for automated UI project generation to reduce the repetitive setup cost of translating game UI designs into production-ready implementations and the reliance on individual developer experience, while exploring how agents could support the broader UI workflow.</li>
        <li><strong>UI Agent Workflow:</strong> Independently designed and implemented a hybrid architecture combining an <strong>Agent Workflow with autonomous nodes built on the OpenAI Agents SDK</strong>. The system generates NeoX <code>.uiprefab</code> projects from Figma/PSD designs in both from-scratch and blueprint-based modes. The workflow comprises approximately 50 nodes, including 26 reused across processes. LLMs are reserved for necessary tasks such as structural inference and asset selection, while deterministic workflow logic provides fallbacks; multimodal retrieval and project-guideline injection align outputs with established engineering conventions and existing control reuse. The blueprint mode reuses legacy projects, and collaboration with GUI designers produced approximately 10 commercial blueprints covering recurring interfaces such as weekly check-ins and store pop-ups. It reduced selected Class B/C UI tasks from approximately 0.5--1.5 person-days to 30 minutes, with four generated projects already deployed to production.</li>
        <li><strong>UI Agent Evaluation &amp; Regression Testing:</strong> Established a three-layer validation process—<strong>heuristic rule screening, Ground Truth comparison, and final human review</strong>—to improve the stability and verifiability of agent-generated projects. Used AI to mine structural clusters, sub-Prefab candidates, and other patterns from existing projects, calibrated them against UI development standards, and incorporated them into the workflow through iterative Skill Demos. Iterated against real UI projects using heuristic evaluations and editor inspections across more than 10 samples, and built regression tests covering blueprint-based generation and foundational from-scratch capabilities. During a blueprint middleware migration, the tests identified omitted fields and scripts that had distorted the overall screen proportions.</li>
        <li><strong>Internal Knowledge Base:</strong> Explored different knowledge-modeling and retrieval strategies for heterogeneous internal documentation and game-domain knowledge. For categorized documents spanning multiple file types, used Docling to create a unified structured fact layer and augmented charts and tables, then combined vector search, BM25, and PageIndex-style structured retrieval. For game knowledge, addressed stale and conflicting design documents by extracting approximately 20,000 entity nodes and 30,000 program relationships from source code and configuration tables into a JSON Wiki. Built Agentic RAG with an LLM Wiki-style approach and query rewriting, achieving full recall coverage on the test set.</li>
      </ul>
    </article>
    <article class="me-entry">
      <h3>Algorithm Researcher (Intern)</h3>
      <p class="me-entry-company">NSFOCUS Technology (Wuhan) · Dec 2025 -- Mar 2026</p>
      <ul class="me-detail-list">
        <li><strong>Vulnerability Mining Agent &amp; CodeQL Verification Loop:</strong> To address the tendency of LLMs to produce unreliable conclusions in complex code-vulnerability analysis after becoming anchored to early judgments, built a Single-Agent harness for vulnerability mining. Organized vulnerability-intelligence retrieval, source-code localization, taint-flow modeling, CodeQL query generation, and engine validation into a traceable multi-round analysis loop. Emphasized state representation, tool-calling protocols, and validation feedback rather than predefined multi-role decomposition, enabling the model to continuously iterate around candidate source/sink pairs, failed paths, tool outputs, and CodeQL validation results while mitigating path dependence and false positives in long-horizon analysis.</li>
        <li><strong>Training Data &amp; Agent Trajectory Curation:</strong> Built a vulnerability-data cleaning and annotation pipeline over open-source projects and internal databases for post-training the vulnerability-mining agent. Extracted and verified 8,000+ CVE entities, collected 4,000 high-quality samples, labeled them by vulnerability type and programming language, and balanced the training-set composition. Distilled 2,500 high-confidence tool-use SFT trajectories and 300 taint-flow records from agent executions for evaluation-set construction, reward design, and subsequent RL training exploration.</li>
      </ul>
    </article>
  </div>
</section>

<section class="me-section" aria-labelledby="me-research-title">
  <h2 id="me-research-title" class="me-section-title"><span class="me-section-icon" aria-hidden="true"><i class="fa-solid fa-microscope"></i></span><span>Research</span></h2>
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
  <h2 id="me-awards-title" class="me-section-title"><span class="me-section-icon" aria-hidden="true"><i class="fa-solid fa-trophy"></i></span><span>Awards &amp; Certificates</span></h2>
  <ul class="me-awards-list">
    <li>First Prize in Shaanxi Province, National College Students Statistical Modeling Competition</li>
    <li>Third Prize National, SAS China University Data Analysis Competition</li>
    <li>Second Prize, Mathematical Contest in Modeling (MCM/ICM)</li>
    <li>CET4: 510 | CET6: 513</li>
  </ul>
</section>
</div>
