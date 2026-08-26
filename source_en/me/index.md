---
title: Me
date: 2026-06-28 00:00:00
permalink: /me/
lang: en
---


<div class="me-page">
<section class="me-section me-about-section" aria-labelledby="me-about-title">
  <h2 id="me-about-title" class="me-section-title"><span class="me-section-icon" aria-hidden="true"><i class="fa-solid fa-seedling"></i></span><span>About me.</span></h2>
  <div class="me-prose">
    <p>hyacehila is the online ID I have used for a long time. It began with a plant I like, hyacinth, and I later reshaped it into something lighter and more name-like. The suffixes <code>-hila / -ila</code> feel a little like a fictional creature to me, so I sometimes imagine hyacehila as a hyacinth sprite living in complex structures: moving between systems, toolchains, workflows, and uncertain environments, trying to find a path that is natural, explainable, and genuinely solves the problem. You can also call me Julian or Jules.</p>
    <p>This is also close to how I understand technology and engineering. I do not really believe that a technology knows where it belongs from the moment it appears. Often, it takes time to find a place that is genuinely meaningful: first define the problem, form a sufficiently clear and falsifiable judgment, put it into reality, observe failures, absorb feedback, revise the hypothesis, and iterate again. Rather than starting from a fixed Agent architecture, I prefer to decompose the business first: which parts truly require model judgment, which should be handed to Workflow, Retrieval, Tool, or external validation to preserve determinism, and then design an AI system suited to the problem itself. To me, good engineering is not about eliminating all uncertainty, but about knowing where uncertainty should remain and how to keep it continuously exposed to feedback from reality.</p>
    <p>Recently, I have focused on knowledge systems, applications of AI Agents in finance and game development and publishing, long-term memory, and AI NPCs and World Simulation. I am especially interested in how models can evolve from one-off generation tools into systems that understand their environments, accumulate state, act, receive feedback, and collaborate over the long term. This blog records more than technical tutorials. It records the judgments I keep forming and revising around Agent Architecture, Context Engineering, Evaluation, AI Native Game, and related questions. Some of those judgments will become engineering practices; others will be overturned by later experience. Writing is one way for me to preserve this process of iteration. Commercial fiction is another long-term interest.</p>
  </div>
</section>

<section class="me-section" aria-labelledby="me-intern-title">
  <h2 id="me-intern-title" class="me-section-title"><span class="me-section-icon" aria-hidden="true"><i class="fa-solid fa-briefcase"></i></span><span>Internship experience</span></h2>
  <div class="me-entry-list">
    <article class="me-entry">
      <h3>AI Age Research and Development Engineer (internship)</h3>
      <p class="me-entry-company">Internet Recreation (Shanghai) June 2026 August 2026 <a href="/en/blog/2026/08/08/ui-pipeline-automation-thinking/" data-i18n-preserve-label>Intern Rewind</a></p>
      <ul class="me-detail-list">
        <li><strong>Project background and objectives:</strong>Develop an Agent-based UI Project Auto-Generation System, reduce duplication of construction work when converting UI designs to online projects, reduce process dependence on personal development experience, and explore the role of Agent in a more complete UI workflow.</li>
        <li><strong>UI Agent Workstream:</strong>Design and implement a hybrid structure independently, combining the definitive Age Workflow with the OpenAI Actors SDK based autonomous nodes. System can be generated from the Figma/PSD script NeoX <code>.uiprefab</code> Project, while supporting generation from zero and blueprint-based. The process package has about 50 nodes, 26 of which are reused between different processes. LLM only handles tasks that do require model judgement, such as structural extrapolations, selection of materials, and determines that sex workflows are responsible for the bottom-out; multi-modular retrieval and project specifications allow the results to follow established engineering specifications and re-use existing controls as much as possible. Blueprint models can re-use historical engineering. I worked with the GUI designer to organize about 10 business blueprints covering common interfaces like weekly sign-ins, shop windows, etc. Part B/C UI missions have been reduced from about 0.5 to 1.5 person days to 30 minutes, and 4 generation projects have been online.</li>
        <li><strong>UI Agent evaluates and returns:</strong>Create a three-tier validation process for "inspiring rules screening, Ground Truth comparison, manual final review" to make Agent's results more stable and more easily validated. Use AI to dig up structural clustering and sub-prefab candidate models from existing projects, then develop standard calibrations according to UI and add these models to the workflow through multiple rounds of Skill Demo. For more than 10 real UI samples, continuous iteratives are checked with an inspirational rating and editor, supplemented by regression tests for blueprint generation and base capacity from zero. In a blueprint intermediate migration, field and script omissions were detected; these omissions could lead to a distorted overall picture ratio.</li>
        <li><strong>Internal knowledge base:</strong>Try different knowledge modelling and retrieval options for knowledge in complex types of internal documentation and game fields. For documents that are classified, in multiple file formats, a unified structured factual layer is created using Docling, which complements chart information, then integrates vector search, BM25 and PageIndex style structured search. For game knowledge, about 20,000 physical nodes and 30,000 program relationships are extracted from the source code and configuration table and organized into JSON Wiki to address the problem of planning for expired documents and conflict of content. On this basis, the LLM Wiki style and query rewriting is used to construct the Agenic RAG to achieve the recall of all questions on the test set.</li>
      </ul>
    </article>
    <article class="me-entry">
      <h3>Algorithm Fellow (internship)</h3>
      <p class="me-entry-company">Green League Technology (Wuhan) December 2025-March 2026</p>
      <ul class="me-detail-list">
        <li><strong>Hole-duging Agent and CodeQL authentication cycle:</strong>In complex code loophole analysis, LLM is easily anchored by early judgement and then gives unreliable conclusions. In response, I created the Single-Agent Harness for the glare-mining, which forms a traceable multi-cycle analysis cycle of leak intelligence retrieval, source positioning, stain flow modelling, CodeQL query generation and engine validation. The system focuses on status, tool call protocols and validation feedback rather than pre-dismantling multiple roles. The model can be used to keep iterative the results of the validation of candidates, source/sink pairs, failed paths, tool output and CodeQL, thereby reducing path dependence and misreporting in long-range analysis.</li>
        <li><strong>Training data and Agent track preparation:</strong>Based on open-source projects and internal databases, a gap-cleaning and labelling process was put in place for the back-training of Agent in the hole-duging process. Take and validate more than 8,000 CVE entities, collect 4,000 high-quality samples, mark them by type of loophole and programming language, and balance the composition of the training set. Distilling 2,500 high-confidence tools from the Agent execution records using SFT tracks and 300 stain stream records for evaluation of the architecture, reward design and subsequent RL training exploration.</li>
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
  <h2 id="me-awards-title" class="me-section-title"><span class="me-section-icon" aria-hidden="true"><i class="fa-solid fa-trophy"></i></span><span>Awards and certificates</span></h2>
  <ul class="me-awards-list">
    <li>First prize for the National University of Statistics Modelling Championship</li>
    <li>SAS China University Data Analysis National Third-class Award</li>
    <li>Second-class award for the Mathematics Modelling Competition for American University Students (MCM/ICM)</li>
    <li>University English IV: 510 | University English VI: 513</li>
  </ul>
</section>
</div>
