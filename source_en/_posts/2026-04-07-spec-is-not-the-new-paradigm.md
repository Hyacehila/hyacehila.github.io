---
title: 'Spec Is Not the New Paradigm: Vibe Coding, SDD, and Software Engineering in the AI Era'
title_zh: Spec 不是新范式：Vibe Coding、SDD 与 AI 时代的软件工程转向
date: 2026-04-07 20:00:00 +0800
categories:
- Work & Society
- AI Engineering Workflows
tags:
- Software Engineering
- Feedback Loops
- AI Coding
author: Hyacehila
excerpt: AI-era software engineering is not moving toward spec-first. As code generation costs collapse, it shifts toward
  feedback-first through prototypes, integration feedback, and living constraints.
description: AI-era software engineering is not moving toward spec-first. As code generation costs collapse, it shifts toward
  feedback-first through prototypes, integration feedback, and living constraints.
excerpt_zh: 代码生成成本下降后，软件工程需要更早获得原型和集成反馈，再把经过验证的约束整理成文档、契约、测试与 ADR。Spec 仍然有用，但它更适合系统逐渐收敛之后。
permalink: /blog/2026/04/07/spec-is-not-the-new-paradigm/
lang: en
translation_key: 2026-04-07-spec-is-not-the-new-paradigm
translation_status: machine
translation_source_hash: 4e7be4733de97592d1af2b434f1d4c2e843ff802b0b52bc662c998db16e87fc9
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<p>If we are going to discuss today whether there is a new dominant paradigm for the AI-era software project, the answer is probably not Spec.</p>
<p>The questions in this article can also be addressed<a href="/en/blog/2026/07/08/after-code-is-cheap/">After the code was getting cheaper,</a>、<a href="/en/blog/2026/05/27/cli-vs-gui-agent-era/">We're back to the CLI: Age of the Age of the Age of the CLI and the GUI</a>How the concept of a relatively close read together is developed in different contexts.</p>
<p><code>Specification</code>、<code>contract</code>、<code>ADR</code> and <code>acceptance criteria</code> It remains important in high-risk systems. What I object to is seeing pre-documentation as the center of the software engineering of the AI era, as if the need and design were sufficiently complete to allow models to stabilize and export complex systems like compilers.</p>
<p>After the cost of code generation has decreased, the difficulty of the project has gradually shifted to the idea of establishing feedback, exposure constraints and validation. Prototypes, integrated and operational results can help teams distinguish between paper ideas and established system behaviour.<code>Spec</code> While still important in this process, it is better to record the constraints of gradual stabilization than to assume that the entire system can be written before exploration begins.</p>
<h2>Why? <code>Vibe Coding</code> It's gonna rise.</h2>
<p><code>Vibe Coding</code> The epidemic is associated with the decline in prototype costs. The developers can allow the model to generate operational results before deciding whether to continue with the input, which is appropriate for the exploratory phase where the needs are not clear.</p>
<p><code>2025-02-06</code>Andrej Karpathy describes this approach in terms of “total submission to the atmosphere, embraces the speed of the index, and even forgets the existence of the code itself”. Simon Willison later made a further distinction. <code>vibe coding</code> Auxiliary programming: the former includes<strong>Let the model run the thing.</strong>I'm sorry. This choice is easily made when the natural language is short enough to run the prototype, as developers can see more quickly whether the idea is worth continuing.</p>
<p>The sources of information for the early stages of exploration have changed as a result. The prototype was previously costly, and teams devote more time to document promotion, validation and market surveys, and see operational results for weeks or months. Now, an idea could be demo in a few minutes. Teams can hand over the system to users, test and integrate environment earlier, and revise understandings with actual feedback.</p>
<p>Anthropic is here. <a href="https://www.anthropic.com/engineering/building-effective-agents">Building effective agents</a> It is proposed to start with a simple, verifiable structure, which would add complexity based on the value of the business. For fast-changing agent and coding anent projects, the team needs to prioritize ensuring that each change receives a perceptible feedback rather than first complete process and documentation systems.</p>
<p><code>Vibe Coding</code> Prototypes and feedback were placed ahead of the exploratory period and pushed to the point where the code was intentionally not read. It explains why developers would seek operational results first, but this cannot omit subsequent review, testing and manual judgement.</p>
<p>Vibe Coding is more suitable for the working methods of the exploration period. The production environment also requires review, testing, monitoring and manual judgement, otherwise the time saved during the generation phase will translate into certification and maintenance costs. Some prototypes will not continue to be produced, while others will be completed after generation.</p>
<h2>Why? <code>Spec</code> It's not a new paradigm.</h2>
<p><code>Spec-first</code> The risk is that the team may try to spell out all the key constraints before it can be exposed to the complexity of the real system.</p>
<p>This is not usually a matter of complex software.</p>
<p>Paul Ralph. <code>Sensemaking-Coevolution-Implementation</code> Theories systematically question the linear development process. It describes software development as <code>sensemaking</code>、<code>coevolution</code> and <code>implementation</code> There's a lot of movement between them, not just the "analyzing--"&gt; Design -&gt; Encoding -&gt; Test "sequencely advance. The team simultaneously changes its own understanding of the problem and the environment when it conceives and realizes solutions; when the system is put into operation, it creates new constraints and demands. Many of the requirements were therefore identified as software was gradually being shaped. In other words,<strong>Demand understanding, rewrite and realization are inherently intertwined.</strong></p>
<p>At the outset of the project, the operator usually knows what problems he wishes to solve, but does not necessarily know the boundaries of the system and the specific mechanisms. Heavy. <code>Spec</code> Current assumptions can be documented, but no constraints that have not yet emerged can be established in advance. Many boundaries are exposed only in prototypes, integration and actual use.</p>
<p>The behavioural boundaries of modern software are often hidden in a coupling relationship between modules. Richard Cook is in <a href="https://how.complexsystems.fail/">How Complex Systems Fail</a> It was noted that complex system failures were rarely caused by individual and isolated causes. The issues of competency models, state competition, cross-service consistency, cache failure, speed limits, re-testing strategies and event sequences are usually not disclosed until integration and operating conditions are superimposed. Pre-document can record known risks, but it is difficult to list all the interactions in advance.</p>
<p>I'm based on the following observations. I don't want to. <code>Spec</code> The software engineering center that is considered to be the AI age:</p>
<ul>
<li>Needs change as the team understands the problem.</li>
<li>System boundaries and high-risk constraints often appear after integration.</li>
<li>The text that attempts to describe all future realization details quickly expires when the code changes faster than the document maintenance speed.</li>
</ul>
<p>Kiro's <a href="https://kiro.dev/docs/specs/">Specs Documents</a> The video shows the following:<code>Specs</code> (a) Tasks that are suitable for complex functions, high-cost bugs, teamwork and the need for structured planning;<code>Vibe</code> It is more appropriate for a prototype period that is short example and unclear about the goals. That means... <code>Spec</code> Value depends on the mission phase and risks, rather than the same starting point for all projects.</p>
<p>An approach that is primarily applicable to complex, high-risk or identified missions is more like an important tool in the workflow than a uniform paradigm for all software development. AI Coding has a wider impact on production methods, but different constraints and work are still required at different stages.</p>
<h2>Why is it pure? <code>Vibe</code> I'll hit the wall.</h2>
<p>The feedback priority is clear boundaries, pure. <code>Vibe</code> Responsibility for the validation and maintenance of production systems cannot be assumed over the long term.</p>
<p><code>Vibe Coding</code> Shortening the path from thought to interactive systems. After the project has passed its prototype period, the main costs will shift to validation, maintenance and long-term consistency, and these will not disappear as a result of increased production speed.</p>
<p>Simon Willison. Yeah. <code>vibe coding</code> The definition contains unreading and ununderstanding generation codes. This approach shifts complexity from pre-generation design to post-generation validation. A partial transfer is acceptable during the exploratory period, as the objective is to test the error quickly; the test cost of the certification capacity and the production environment is limited when it enters production.</p>
<p>Dora is here. <a href="https://dora.dev/research/2025/dora-report/">State of AI-assisted Software Development 2025</a> The results of AI are influenced by the organization's original ability to do so, as indicated in subsequent readings. Without the team improving small batch change and validation mechanisms simultaneously, the speeds felt by individuals do not necessarily translate into higher throughput, or guarantee a simultaneous decline in instability.</p>
<p>GitClar <a href="https://www.gitclear.com/ai_assistant_code_quality_2025_research">2025 AI Code Quality Research</a> A set of quantitative observations were provided: it analysed <code>2020-2024</code> Year <code>2.11</code> Billion line code changes, changed lines associated with re-engineering found from <code>2021</code> Year <code>25%</code> Down to <code>2024</code> Not in years. <code>10%</code>classified as <code>copy/pasted</code> . The code from <code>8.3%</code> Raise <code>12.3%</code>I'm sorry. These data do not directly support the poor quality of AI generation codes, but suggest that teams need to look at whether the production speed exceeds abstract, re-constructed and maintained capabilities.</p>
<p>METR <a href="https://metr.org/Early_2025_AI_Experienced_OS_Devs_Study-paper.pdf">Measuring the Impact of Early-2025 AI on Experienced Open-Source Developer Productivity</a> Senior open source developers familiar with large warehouses were studied. The average time has increased after the use of the front-line AI tool was allowed under mission and experimental conditions <code>19%</code>I'm sorry. This result cannot be extended to "AI " , but it suggests that in large warehouses it may be more time-consuming to understand context, review, test and repair than to generate code.</p>
<p>Pure <code>Vibe</code> It is easy to underestimate the costs that follow generation:</p>
<ul>
<li>Validation cost</li>
<li>Integration costs</li>
<li>Long-term readability costs</li>
</ul>
<p>If validation, integration and readability problems are bypassed at the demo stage, they will converge over time. The model still generates codes, but teams spend more time understanding and repairing what has been achieved.</p>
<p><em>Note: The main reflections of the above studies <code>2024</code> Period <code>2025</code> The tools and work streams at the beginning of the year. Models and coding parties are still changing, and subsequent conclusions need to be updated in conjunction with new empirical studies.</em></p>
<h2><code>SDD</code> and <code>Spec</code> What stage does it work?</h2>
<p>Discussion <code>SDD</code> The more useful question then is what stage of the software life cycle it is suitable for and what feedback mechanisms it needs to work with. My judgment is, modern. <code>SDD</code> Suits the system to start stifling. At this point, the team has found some stabilization constraints that need to be organized into enforceable and sustainable work.</p>
<p>Once the system begins to enter a leaning period, many elements must be rewritten and more structured than in the past. For example:</p>
<ul>
<li>Permission Border</li>
<li>API Compact</li>
<li>Data Model</li>
<li>Migration policy</li>
<li>Failed to process and roll back</li>
<li>Multi-team shared structure decision-making</li>
<li>success criteria</li>
</ul>
<p>Once it's stable, it's not supposed to be just talking records, prompt history, or the brain of an engineer. They should be refined. <code>Spec</code>The following are the living bars that are accessible to both humans and delegates.</p>
<p>Modern <code>SDD</code> In the development process, the constraints that have been demonstrated to be effective can be organized into machine-readable columns. Testing, schema, policy, ADR and documentation together limit subsequent changes and reduce the number of repeated discussions on the same issue by teams. It is responsible for stabilizing the knowledge that has been developed and not for forecasting the full demand before the exploration begins. Maintenance <code>docs</code> The folder and the access of angent to these documents is also a light form of this practice.</p>
<p>The work stream of the project can be switched between prototype feedback and structured work. The exploration phase allows for rapid generation, binding gradually to stabilize before increasing the weight of documents, tests and contracts. I call this working method “Structuralized exploration of feedback priorities”: first, reducing uncertainty through operational results, then organizing stable knowledge into a reusable constraint for the next round of development.</p>
<p>A mature team will switch lead items according to the stage. When exploring, prototypes and integration results provide information; after the harvest period,<code>Spec</code>, test and ADR are responsible for fixing the restriction. Many of the coding parties also started reading <code>docs</code>, rules and other project-level documents, bringing knowledge beyond the session into the generation process.</p>
<p>The following is a simplified workflow. Three phases may overlap or may be followed by a step back when new problems are identified:</p>
<p>Phase one, yes.<strong>Explore</strong>。<br>The goal at this point is to move vague ideas to verifiable positions as soon as possible. Available <code>Vibe Coding</code>, prototype low-level security, scaffolding anent and rapid integration testing to explore boundaries. This stage allows for the tolerance of some temporary codes, but records are to be kept of what is achieved without direct access to production and the buildable MVPs are to be established as early as possible.</p>
<p>Phase two, yes.<strong>Crystal</strong>。<br>When certain constraints begin to recur, certain models of failure begin to stabilize and certain structural boundaries begin to emerge as necessary, it is no longer possible to continue to rely solely on prompt history and improvisation. You need to refine these things into tests, tests, checklists, ADRs, schema, polity, and extract the experience from the conversation.</p>
<p>Phase three, yes.<strong>Stay down.</strong>。<br>Here.<code>Spec</code> It was just one of the main players, but the role it played had changed. It no longer a priori defines the system, but stabilizes the structure that is already standing in the feedback and becomes Kiro. <code>requirements.md / design.md / tasks.md</code>。</p>
<p>Anthropic is here. <a href="https://www.anthropic.com/engineering/harness-design-long-running-apps">Harness design for long-running application development</a> Use <code>planner / generator / evaluator</code> Structure handles lengthy processes. Planning, generation and evaluation are carried out by different elements and structured artifact is transmitted between sessions. This design is similar to the structured exploration: the system generates continuous feedback and extracts reusable information as input for subsequent missions.</p>
<p>In such a workflow, exploration and the Statute will alternate. Whenever a constraint is repeatedly verified, it should be written into a test, contract or document, rather than kept in a chat record.</p>
<h2>Conclusions</h2>
<p><code>Spec</code> It is a high-value job, especially suitable for the drying-up, governance and high-risk modules, but it is not able to prediscover all unknown issues for the team.<code>Vibe Coding</code> It reduces the costs of prototypes and is suitable for testing borders, and it cannot be directly a long-term production discipline.</p>
<p>With the increasingly inexpensive generation of codes, teams need to establish feedback, validate constraints and organize established knowledge into tests, compacts and documents more quickly. Specific processes can vary depending on project risks and phases, but both prototype feedback and structured work are essential.
AI, the software engineering center of the era, no longer write the imagination as early as possible. <code>Spec</code>And instead, it sends the imagination back to the feedback loop as quickly as possible.</p>
<blockquote>
<p>Last amended 19 April 2026, for information purposes only.</p>
</blockquote>
<h2>References</h2>
<ul>
<li>Andrej Karpathy quoted by Simon Willison, <a href="https://simonwillison.net/2025/Feb/6/andrej-karpathy/">A quote from Andrej Karpathy</a></li>
<li>Simon Willison, <a href="https://simonwillison.net/2025/Mar/19/vibe-coding/">Not all AI-assisted programming is vibe coding (but vibe coding rocks)</a></li>
<li>OpenAI, <a href="https://openai.com/index/harness-engineering/">Harness engineering: leveraging Codex in an agent-first world</a></li>
<li>Anthropic, <a href="https://www.anthropic.com/engineering/building-effective-agents">Building effective agents</a></li>
<li>Anthropic, <a href="https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents">Demystifying evals for AI agents</a></li>
<li>Anthropic, <a href="https://www.anthropic.com/engineering/harness-design-long-running-apps">Harness design for long-running application development</a></li>
<li>Kiro Docs, <a href="https://kiro.dev/docs/specs/">Specs</a></li>
<li>DORA, <a href="https://dora.dev/research/2025/dora-report/">State of AI-assisted Software Development 2025</a></li>
<li>DORA, <a href="https://dora.dev/insights/balancing-ai-tensions/">Balancing AI tensions: Moving from AI adoption to effective SDLC use</a></li>
<li>GitClear, <a href="https://www.gitclear.com/ai_assistant_code_quality_2025_research">AI Copilot Code Quality: 2025 Look Back at 12 Months of Data</a></li>
<li>METR, <a href="https://metr.org/Early_2025_AI_Experienced_OS_Devs_Study-paper.pdf">Measuring the Impact of Early-2025 AI on Experienced Open-Source Developer Productivity</a></li>
<li>Paul Ralph, <a href="https://arxiv.org/abs/1302.4061">The Sensemaking-Coevolution-Implementation Theory of Software Design</a></li>
<li>Richard I. Cook, <a href="https://how.complexsystems.fail/">How Complex Systems Fail</a></li>
</ul>
