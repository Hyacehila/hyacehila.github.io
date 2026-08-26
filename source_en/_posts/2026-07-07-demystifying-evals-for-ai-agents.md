---
title: Demystifying evals for AI agents (Anthropic)
title_zh: Demystifying evals for AI agents — Anthropic
date: 2026-07-07 23:30:00 +0800
categories:
- Agent Systems
- Agent Evaluation & Governance
tags:
- Evaluation
- AI Engineering
- LLM
author: Hyacehila
mathjax: false
excerpt: The capabilities that make agents useful also make them difficult to evaluate. The strategies that work across deployments
  combine techniques to match the complexity of the systems they measure.
description: The capabilities that make agents useful also make them difficult to evaluate. The strategies that work across
  deployments combine techniques to match the complexity of the systems they measure.
excerpt_zh: 让 agent 变得有用的那些能力，也让它们变得难以 eval。跨部署行之有效的策略会组合多种技术，以匹配它们所衡量系统的复杂度。
permalink: /blog/2026/07/07/demystifying-evals-for-ai-agents/
lang: en
translation_key: 2026-07-07-demystifying-evals-for-ai-agents
translation_status: machine
translation_source_hash: 0ba6597ffe66b6067668576764008199d59cffdbceb1c7d8bea862568419931d
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<blockquote>
<p>This post is a translation from Anthropic Engineering Blog, published on 9 January 2026. <a href="https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents">Demystifying evals for AI agents</a>I'm sorry. This article was kept in January of this year, and it was not read in detail until July when a complete set of anatomy systems was actually built. It was of great quality after reading it, from the basic concept of eval, the different types of eval strategy, to the creation of a zero-sum eval map to the co-operation of eval with other quality tools (project monitoring, A/B testing, etc.). A record is kept here to facilitate his checking. The text was translated without being deleted, the technical terms were not translated to ensure readability, the original CDN address was directly quoted in the picture and the links were kept as they were.</p>
</blockquote>
<p>The questions in this article can also be addressed<a href="/en/blog/2026/03/17/behavior-auditing-and-decoding-beginners-guide/">Behaviour Audit and Decoded Behaviour: From Reward to Agent Observation</a>、<a href="/en/blog/2026/03/18/from-black-box-predictors-to-traceable-medical-agents/">From Black Box Forecast to Retroactive Medicine</a>How the concept of a relatively close read together is developed in different contexts.</p>
<h2>Introduction</h2>
<p>Good eval can help the team deliver AI ant. No, eval, the team is easily caught in a passive cycle-- – The problem is only found in the production environment, and repairing one malfunction triggers another. Eval lets problems and behavioural changes become visible before they affect users, and their value will accumulate throughout the life cycle of an individual.</p>
<p>Just as we are. <a href="https://www.anthropic.com/engineering/building-effective-agents">Building effective agents</a> The description describes how angents cross-turn operations: call tool, change state, adjust to intermediate results. It's these powers that make AI anent useful -- autonomy, intelligence and flexibility -- that make it harder.</p>
<p>Through our work internally and in collaboration with clients at the forefront of the development of angent, we learned how to design a more rigorous and useful eval for angent. The following are practices that have been validated in various contexts and in real deployment scenarios.</p>
<h2>The structure of an evaluation</h2>
<p>An evaluation&quot;eval&quot;) is a test of the AI system: give AI an input and apply the Grading logic to its output to measure success. In this paper, we focus on automation that can be run without real users in the development process.</p>
<p>Single-turn eval is very direct: a prompt, a response, and a grading logic. For the early LLM, single-turn, non-aggression eval is the main method of assessment. With the progress of AI, multi-turn eval became more common.</p>
<p><img src="https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2Fbd42e7b2f3e9bb5218142796d3ede4816588dec0-4584x2834.png&amp;w=3840&amp;q=75" alt="simpl yval vs multi-turn val"></p>
<p>In a simple eval, an angent handles a prompt, grader check the output for expectations. And in more complex multi-turn eval, a coding agent receives tool, a task (e.g. building a MCP server) and an environment, executes &quot;agent loop&quot;(tool call and updating) and update in the environment. Grading then uses unit test to verify whether MCP server is working.</p>
<p>The blog also has a more complex version of the video. Agent uses tool in multiple turn-ons, changes state in the environment and adjusts continuously - meaning that errors can spread and accumulate. The front model can also find creative solutions that go beyond static eval limits. For example, Opus 4.5 found a loophole in the policy in addressing a problem of booking tickets for two-bench. Follow the written standard of eval.&quot;Failed&quot;Yes, but in fact it found a better solution for users.</p>
<p>When constructing ant eval, we use the following definitions:</p>
<ul>
<li>One. <strong>task</strong>(also known as problem or test case) is a single test with clearly defined input and success criteria.</li>
<li>Every attempt of each task is called once <strong>trial</strong>I'm sorry. Because the output of the model changes between different operations, we will run multiple trials to produce more consistent results.</li>
<li>One. <strong>grader</strong> It is the logic of scoring certain aspects of ant's performance. One of the tasks can have more than one grader, each with more than one assertion (sometimes called check).</li>
<li>One. <strong>transcript</strong>(also known as track or projectory) is a complete record of a trial, including output, tool call, resoning, intermediate results and all other interactions. For Anthropic API, this is the complete message group at the end of the operation of eval - contains all calls to API and all returns.</li>
<li><strong>Outcome</strong> It's the end state of the trial environment. A ticket booking an anent may end up at the transcript.&quot;Your flight has been booked.&quot;, but outcome is whether the booking is actually in place in the environment SQL database.</li>
<li>One. <strong>evaluation harness</strong> It's the infrastructure that runs eval from end to end. It provides instructions and tool, and runs the task, records all steps, and drives the output and summarizes the results.</li>
<li>One. <strong>agent harness</strong>(or scaffer) is a system that allows a model to run as an anent: it processes input, organizes tool call and returns the result. When we, eval &quot;An anent&quot;When we eval is the effect of the joint work of the Harness and Model. Claude Code, for example, is a flexible agent with the idea that we build our long-run agent with the core original language of Agent SDK.</li>
<li>One. <strong>evaluation suite</strong> It is a set of tasks designed to measure specific capabilities or behaviours. The table in Suite usually shares a broad goal. For example, a customer support area may test refunds, cancellations and upgrades.</li>
</ul>
<p><img src="https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F0205b36f9639fc27f2f6566f73cb56b06f59d555-4584x2580.png&amp;w=3840&amp;q=75" alt="angent eval component"></p>
<h2>Why build evaluations?</h2>
<p>When the team started building angent, they could move pretty far through a combination of manual tests, dogfood and intuition. More stringent eval may even be considered as additional expenses to slow down delivery. But after the early prototype phase, once angent goes online and starts scale, the development without eval starts to collapse.</p>
<p>The critical point usually appears: user feedback angent after change&quot;It's not working.&quot;♪ And the team ♪&quot;♪ Fly blind ♪&quot;— There is no way to verify it except for speculation and inspection. When there is a lack of eval, the debug is passive: waiting for a complaint, reproducing it manually, repairing the bug, and then hoping that nothing else will come back. The team was unable to distinguish between real regreasing and noise, to automatically test changes with hundreds of scenes before they were released, or to measure improvements.</p>
<p>We've seen this evolution go on and on. For example, Claude Code started with a rapid iterative process based on feedback from Anthropic staff and external users. And then we joined eval -- initially in narrow areas like concise and file edit, then in more complex acts like over-engineering. These evals help to identify problems, guide improvements and focus on research-project collaboration. Together with the tools of prevention monitoring, A/B test, user research, etc., eval provides a signal for continuous improvement of Claude Code.</p>
<p>It's useful to write eval at any stage of the life cycle of an individual. Early, eval forced the product team to clarify what angent's success means; later, eval helped maintain consistent quality standards.</p>
<p>Discript's angent helps users edit videos, so they build eval around three dimensions of successful editing workflow: don't screw up, do what I ask, do what I ask. They evolved from manual grading to LLM grader, which is defined by the product team and is regularly manually calibrated, and now runs two separate sets of seine: one for quality, one for quality, and one for regrementation test. The Bolt AI team started building eval after a widely used anent. In three months, they built an eval system: run their anent and use status anallysis to grow output, use the Browner anent to test applications and use LLM judge to process behaviors such as this.</p>
<p>Some teams created eval at the beginning of their development; others joined when they reached a certain scale and eval became a bottleneck for improvement of angent. Eval was particularly useful in the early stages of the development of the anent and could be used to encode expected behaviour in a visible manner. The two engineers, reading the same initial spec, may have different understandings of how AI should deal with the marginal situation. An eval suit can solve this ambiguity. Whenever it is created, eval can accelerate development.</p>
<p>Eval decided how soon you could adopt the new model. When the stronger model is published, the team without eval faces several weeks of testing, and the competitors with eval can quickly determine the model advantage, adjust the prompt and upgrade in a few days.</p>
<p>Once eval is in position, you get free access to baseline and regreasing test:latency, token usage, costs and error rates for each task can be tracked in a static task collection. Eval can also be the highest bandwidth channel between the produdct and the research team, defining indicators that can be optimized by researcher. Obviously, the benefits of eval are much more than tracking progress and improving. Their value will accumulate, and this is easily overlooked, as costs are visible in the early stages and gains are only visible in the later stages.</p>
<h2>How to evaluate AI agents</h2>
<p>We see that there are several types of current large-scale deployments: coding ant, research ant, campaign use ant and general anent. Each type may be deployed in a wide variety of industries, but they can be performed using similar technologies eval. You don't need to invent an eval from scratch. The following sections describe mature technologies for several categories of angent. Please build on these approaches and then expand to your field.</p>
<h3>Types of graders for agents</h3>
<p>Agent eval usually combines three types of grader: code-based, model-based and human. Each grader assesses a part of a transcript or outcome. A key element of an effective eval design is the selection of the right grader.</p>
<h4>Code-based graders</h4>
<table>
<thead>
<tr>
<th>Methodology</th>
<th>Advantages</th>
<th>Disadvantages</th>
</tr>
</thead>
<tbody><tr>
<td>Bring match check (precision, regularity, vagueness, etc.)<br>Binary test（fail-to-pass、pass-to-pass）<br>Static anallysis (lint, type, security)<br>Outlook Authentication<br>Tool call validation (what tools are used, parameters)<br>Transcript analysis (turn number, token use)</td>
<td>Come on.<br>Cheap.<br>Objective<br>Revertible<br>Easy debug<br>Verifiable Specific Conditions</td>
<td>We're vulnerable to effective variants that do not match the exact pattern.<br>Lack of Nuance<br>I'm not sure I'm gonna be able to do anything.</td>
</tr>
</tbody></table>
<h4>Model-based graders</h4>
<table>
<thead>
<tr>
<th>Methodology</th>
<th>Advantages</th>
<th>Disadvantages</th>
</tr>
</thead>
<tbody><tr>
<td>Rubric-based Rating<br>Natural language<br>Pairwise comparison<br>Reference-based evaluation<br>Multi-judge consensus</td>
<td>Flexibility<br>But scale<br>Catch<br>Processing Open<br>Process free format output</td>
<td>Insertity<br>It's more expensive than code.<br>Need to calibrate with human grader to maintain accuracy</td>
</tr>
</tbody></table>
<h4>Human graders</h4>
<table>
<thead>
<tr>
<th>Methodology</th>
<th>Advantages</th>
<th>Disadvantages</th>
</tr>
</thead>
<tbody><tr>
<td>SME review<br>Crowdsourced's judgment<br>Spot-check Sample<br>A/B testing<br>Inter-annotator agreement</td>
<td>Standard quality of gold<br>Match expert user judgement<br>For calibration model-based grader</td>
<td>Expensive<br>Slow<br>Usually, it takes a massive acquisition of human experts.</td>
</tr>
</tbody></table>
<p>For each task, the rating can be weighted (the combination of the grader scores must reach the threshold), binary (all graders must pass) or mixed.</p>
<h3>Capability vs. regression evals</h3>
<p>Capital eval (or&quot;quality&quot; The question is:&quot;What can you do with this angent?&quot;They should start with a lower passrate, aim at an anent that's unmanageable, and give the team a climbable slope.</p>
<p>The question is:&quot;Can ant still handle all the things it used to handle?&quot;They should have been close to 100% of the pass rate. They prevent retreats, and the drop in scores indicates that something is wrong and needs to be repaired. It is also important to run the process when the team climbs the slopes of Capability eval, to ensure that the change does not cause problems elsewhere.</p>
<p>Pass rank high level eval when angent goes online and optimizes&quot;Graduated&quot;Be regremention suit, running continuously to capture any drift. Used to measure&quot;Can we do this?&quot;The task, now measured by&quot;Can we do this reliably?&quot;</p>
<h3>Supplement: On the Agent Trust and Security Assessment</h3>
<p>Trust and safety assessments are important for intelligence bodies entering the production environment, and most of them focus on research capabilities, which are also the focus of developers in their development. The CSA can be a special assessment perspective that incorporates both the results and the process assessment.</p>
<p>This helps to assess the reliability and adaptability of intelligent bodies under less than ideal conditions. This is done to avoid poor interaction between intelligent bodies and systems. In fact, when intelligence bodies are put into practical application, they may face unexpected tests. It is therefore important to ensure that intelligence bodies are able to respond appropriately to such situations.</p>
<p>We are concerned about reliability in harsh conditions. Focus on stability (perfect capacity), safety (resistance of command injection) and fairness (reduce prejudice), as well as any security problems that may be encountered after being online.</p>
<h3>Evaluating coding agents</h3>
<p>Coding anent prepares, tests and debugs code, browses code and runs commands like human developers. Effective eval for modern coding anent usually relies on clearly specified tests, stable testing environments and adequate testing for the resulting code.</p>
<p>Deterministic grader is natural for the company anent because software is usually more direct: can the code run? Did the test pass? Two widely used committees, coding ant-bunkmark...<a href="https://www.swebench.com/SWE-bench/">SWE-bench Verified</a> and <a href="https://www.tbench.ai/">Terminal-Bench</a>— That's the way it is. SWE-bench Verified provides angent with GitHub issue from popular Python warehouses and adds solutions by running a test suit; they are only passed when the solution fixes failed tests without disrupting existing ones. LLM's grades on this eval have improved from 40% in just one year to over 80%. The main reason for this is that the government is not going to be able to do anything about it. It tests end-to-end technology for task, for example, to build Linux kernel from source code or train an ML model.</p>
<p>Once you have a pass-or-fail test to verify the key of the code challenge outside, it is usually necessary to make a grading of the transcript. For example, the inspired codeQuality rule can go beyond evaluating the resulting code by testing this dimension, while the model-based grader with a clear rubric can assess how ants call tool or interact with users.</p>
<p><strong>Example: A theory for coding anent eval</strong></p>
<p>Consider a coding task, angent must fix a gap between the two. This agent can be assessed using a combination of grader and tracked metric, as shown in the indicative YamL file below.</p>
<pre><code class="language-yaml">task:
  id: &quot;fix-auth-bypass_1&quot;
  desc: &quot;Fix authentication bypass when password field is empty and ...&quot;
  graders:
    - type: deterministic_tests
      required: [test_empty_pw_rejected.py, test_null_pw_rejected.py]
    - type: llm_rubric
      rubric: prompts/code_quality.md
    - type: static_analysis
      commands: [ruff, mypy, bandit]
    - type: state_check
      expect:
        security_logs: {event_type: &quot;auth_blocked&quot;}
    - type: tool_calls
      required:
        - {tool: read_file, params: {path: &quot;src/auth/*&quot;&#125;&#125;
        - {tool: edit_file}
        - {tool: run_tests}
  tracked_metrics:
    - type: transcript
      metrics:
        - n_turns
        - n_toolcalls
        - n_total_tokens
    - type: latency
      metrics:
        - time_to_first_token
        - output_tokens_per_sec
        - time_to_last_token
</code></pre>
<p>Please note that this example presents a full picture of the various types of graders available for illustration. In practice, coding eval usually relies on unit test for accuracy and LLM rubric for overall code quality, and additional grader and metric add only as needed.</p>
<h3>Evaluating conversational agents</h3>
<p>The users interact with users in areas such as support, sales or coaching. Unlike traditional Chatbot, they maintain state, use tool and act in the middle of dialogue. Although coding ant and research ant may involve multiple rounds of interaction with users, a unique challenge is faced by the following:<strong>The quality of interaction itself is part of your desire to eval.</strong>I'm sorry. An effective eval for general anent usually relies on a final state outcome, and a rubric that captures both the completion and interactive quality of the task. Unlike most other evals, they usually need a second LLM to simulate users. We're here. <a href="https://alignment.anthropic.com/2025/automated-auditing/">alignment auditing agents</a> Using this method, pressure testing of models is conducted through extended confrontational dialogue.</p>
<p>The success of the conservation anent can be multidimensional: state check, has it been completed in no more than 10 turn-points, is it appropriate to speak (LLLM rubric)? Two benchmarks with multidimensional dimensions. Yes. <a href="https://arxiv.org/abs/2406.12045">τ-Bench</a> and his successors <a href="https://arxiv.org/abs/2506.07982">τ2-Bench</a>I'm sorry. They simulate multi-turn interactions in areas such as retail support and airline booking, one of which model plays the role of user, and angent navigates in the real scene.</p>
<p><strong>Example: A theory of general anent eval</strong></p>
<p>Consider a support task, angent has to process a disloyalty client.</p>
<pre><code class="language-yaml">graders:
  - type: llm_rubric
    rubric: prompts/support_quality.md
    assertions:
      - &quot;Agent showed empathy for customer&#39;s frustration&quot;
      - &quot;Resolution was clearly explained&quot;
      - &quot;Agent&#39;s response grounded in fetch_policy tool results&quot;
  - type: state_check
    expect:
      tickets: {status: resolved}
      refunds: {status: processed}
  - type: tool_calls
    required:
      - {tool: verify_identity}
      - {tool: process_refund, params: {amount: &quot;&lt;=100&quot;&#125;&#125;
      - {tool: send_confirmation}
  - type: transcript
    max_turns: 10
tracked_metrics:
  - type: transcript
    metrics:
      - n_turns
      - n_toolcalls
      - n_total_tokens
  - type: latency
    metrics:
      - time_to_first_token
      - output_tokens_per_sec
      - time_to_last_token
</code></pre>
<p>Like the example of coding anent, this task shows a variety of categories of graders for illustration. In practice, the general application of the model-based grader is used to assess both the quality of communication and the degree of completion of the objectives, because many of the questions, such as answering one question, may have several correct solutions.</p>
<h3>Evaluating research agents</h3>
<p>Research ant collects, synthesizes and analyses information, and then produces outputs such as answers or reports. Unlike the two-dollar pass/fail signal that is available for coding anent, the quality of the search can only be judged relative to the task. What is it?&quot;Comprehensive&quot;、&quot;Sources are reliable&quot;Even.&quot;Correct.&quot;Depending on context: a market scan, a due diligence acquisition report and a scientific report each require different standards.</p>
<p>Research eval faces a unique challenge: experts may disagree on whether a comprehensive report is comprehensive, ground truth will drift as reference content evolves, and longer, more open output leaves more room for error. For example,<a href="http://arxiv.org/abs/2504.12516">BrowseComp</a> This is a banchmark test if AI anent is open on the network.&quot;A needle in a haystack.&quot;The answer is found in it — these questions are designed to be easy to prove but difficult to resolve.</p>
<p>One of the strategies to build a research event eval is to combine multiple categories of grader. The certificate checks to verify whether the statement is supported by a source that has been retrieved, defines the key facts that a good answer must contain, and confirms that the source cited is authoritative, not just the first result that has been retrieved. For those with objective correct answers&quot;What's X's third quarter income?&quot;- That's what I'm talking about. A LLM can mark gaps in statements and coverage that lack support, but at the same time validate the consistency and completeness of the open synthesis report.</p>
<p>Given the subjective nature of the quality of the research, LLM-based rubric should be regularly calibrated with expert manual judgement to effectively carry out the grading of such an anent.</p>
<h3>Computer use agents</h3>
<p>The Computer use antenna to interact with software through the same interface as humans - screenshots, mouse clicks, keyboard input and scroll - rather than through API or code. They can use any application with a graphical user interface (GUI), from design tools to legacy enterprise software. Eval needs to run antent in real or sandbox environments, to use software applications and check if it reaches expected exit. For example,<a href="https://arxiv.org/abs/2307.13854">WebArena</a> Tests the browser-based tool for the navigational correctness of ant, using URLs and page state check, and back-end state verification of the tool for modifying data (validation that the order was actually placed, not just that the page was created).<a href="https://os-world.github.io/">OSWorld</a> Extend this to complete operating system controls, using the eval scripts that check products after the completion of the tool: file system state, application configuration, database content and UI element properties.</p>
<p>Browner use ant-party needs to balance token efficiency and latency. DOM-based interactively executes fast but consumes a lot of token, while screenshot-based interacts slowly but token is more efficient. For example, when Claude summarizes Wikipedia, it is more efficient to extract text from DOM; and when Amazon finds a new laptop package, it is more efficient to screen a screenshot (because the whole DOM is very consuming token). In our Claude for Crome product, we developed an eval to check whether ant had chosen the right tool for each context. This allows us to complete the browser-based task faster and more accurately.</p>
<h3>How to think about non-determinism in evaluations for agents</h3>
<p>Whatever the type of angent, the behaviour of angent will change between different run-offs, making the eval result more difficult to interpret than it looks at first glance. Each tsk has its own success rate -- it may be 90% on one tsk, and the other 50% -- and an eval running through the tsk could fail in the next tsk. Sometimes we want to measure the frequency of angent success on a task.</p>
<p>Two indicators help capture this type of thing.</p>
<p><strong>pass@k</strong> Measure angent's probability of getting at least one correct solution in a k attempt. As k increases, pass@k scores rise: more&quot;Shooting opportunities.&quot;This means that the probability of success is higher at least once. 50% of pass@1 points means model can succeed with the first attempt on half of the eval's task. In the coding, we usually care most about angent finding solutions at first attempt -- pass@1. In other cases, it was acceptable to propose multiple solutions, provided that there was one effective one.</p>
<p><strong>pass^k</strong> Measure the probability that all k times will succeed. As k increases, pass^k falls because demanding more trial consistency is a more difficult criterion to reach. If your anent each time a Trial is 75% success rate, and you run three times a Trial, the probability through all three is (0.75) 3 ≈ 42%. This indicator is particularly important for client-oriented angents, as users expect reliable behaviour every time.</p>
<p><img src="https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F3ddac5be07a0773922ec9df06afec55922f8194a-4584x2580.png&amp;w=3840&amp;q=75" alt="Diagrams of divergence for pass@k and pass^k"></p>
<p>Pass@k and pass^k split up with the increase in the number of trials. They are the same when k=1 (equal to the success rate of each trial). By k=10 they tell the opposite story: pass@k approaches 100%, pass^k falls to 0%.</p>
<p>Both indicators are useful, and which one is used depends on product demand: For a successful tool, use pass@k; an anent, key to consistency, use pass^k.</p>
<h2>Going from zero to one: a roadmap to great evals for agents</h2>
<p>This section offers a practical and empirical proposal that we have drawn from practice to help you move from having no eval to having a trusted eval. Consider this as a road map for eval-driving anent development: defining it well in advance, measuring it clearly and continuing in an iterative manner.</p>
<h3>Collect tasks for the initial eval dataset</h3>
<p><strong>Step 0. Start early</strong></p>
<p>We saw a lot of teams delay building eval because they think it's gonna take hundreds of tasks. Actually, 20-50 simple tasks from real failures are a good starting point. After all, in the early days of the development of the agency, each change to the system usually had a clear and detectable impact, and this large effect size meant that small sample sizes were enough. A more mature anent may need a bigger, more difficult eval to test for smaller effects, but it is best to start with an 80/20 method. The longer you wait, the harder it is to build an eval. Early on, product demand naturally translates into best case. You wait too long, you'll have to get the test of success from a linear system.</p>
<p><strong>Step 1. Start with what you already test manually</strong></p>
<p>Starting with the manual checks you've been doing in the development process -- the behavior you validate before each release, and the common test of end-user attempts. If you're already in the production environment, look at your bug tracker and support queue. Turning a user report failure into a test case ensures that your suit reflects real usage; sorting by user impact priority helps you to focus your efforts where you are most worth it.</p>
<p><strong>Step 2: Write unambiguous tasks with reference solutions</strong></p>
<p>Put the task mass right, it's much harder than it looks. A good task is like this: two fields of experts will independently draw the same pass/fail conclusions. Can they pass through this? If not, this challenge needs improvement. The ambiguity in the Task spec will become the noise in the indicator. The same principle applies to the standard of model-based grader: vague rubric produces inconsistent judgements.</p>
<p>Every task should be able to be passed by an ant who's following the instructions correctly. This may be delicate. For example, it was found during the audit of Terminal-Bench that if a task requires an agent to write a script without specifying a file path and testing assumes that the script is on a particular file path, an anent may not have failed if it was not its own fault. Everything that Grader checks should be clearly visible from the task description; angent should not fail because of the vague spec. For front-line model, 0% pass rate (i.e. 0% pass@100) most often means a problem problem rather than an incompetent ant-- This is the signal you should recheck the name and the name of the graph. For each task, it is useful to create a reference solution: a known output that works through all the jobs of the grader. This proves that the challenge is understandable and that the grader configuration is correct.</p>
<p><strong>Step 3: Build balanced problem sets</strong></p>
<p>At the same time, it is necessary to test what should and should not happen. One-sided eval will lead to one-sided optimization. For example, if you only test whether angent is searching when it should be, you may eventually get an angent that search almost everything. Try to avoid it. <a href="https://developers.google.com/machine-learning/crash-course/overfitting/imbalanced-datasets">class-imbalanced eval</a>I'm sorry. We experienced this first-hand when we built Websearch for Claude.ai. The challenge is to prevent the search of Model when it is not appropriate to search, while retaining its ability to conduct extensive research where appropriate. The team built an eval:Model that covers two directions, which should search for (e.g., looking for weather), and a query that should be answered from the knowledge available (e.g.,&quot;Who started Apple?&quot;I'm not sure. It is very difficult to find the right balance between undesired and unsearched, and to find the right balance, which requires multiple rounds of prompt and eval. As more examples of problems arise, we continue to add to eval to improve coverage.</p>
<h3>Design the eval harness and graders</h3>
<p><strong>Step 4: Build a robust eval harness with a stable environment</strong></p>
<p>It is essential that the agent behaviour in eval is roughly the same as that used in the production environment and that the environment itself should not introduce additional noise. Every time, the trial should be...&quot;Isolated.&quot;- Start with a clean environment. Unnecessary sharing of the state (residual documentation, cache data, depletion of resources) between operations may lead to associated failures due to the instability of infrastructure rather than to the performance of ant. Share state and may be artificially high. For example, in some internal evals, we observe Claude gaining some unfair advantage over some of the tests by passing the test git histoory. If multiple independent trials fail because of the same limitations in the environment (e.g. limited CPU memory), these trials are not independent because they are affected by the same factors, evals become unreliable and unable to measure the performance of angent.</p>
<p><strong>Step 5: Design graders thoughtfully</strong></p>
<p>As noted above, the excellent eval design involves selecting the best grader for angent and challenge. We suggest that, where possible, the choice be made between deterministic graders, the use of LLM graders where necessary or required additional flexibility, and the careful use of human graders for additional validation.</p>
<p>There is a common hunch that antent is being executed in accordance with very specific steps, such as a sequence of tool calls in the right order. We find this method too rigid, and it makes the test too fragile, because angent often finds effective methods that the designer of evals did not anticipate. In order to punish creativity unnecessarily, it is usually better to make a move on what angent produces, rather than on the path it takes.</p>
<p>For a task with multiple components, introduce a partial credit. A properly identified and identified customer, but unable to process refunds is better than an instant failed agent. It is important to reflect this continuity of success in the outcome.</p>
<p>Model gathering usually requires careful and iterative validation of accuracy. LLM-as-judge should be calibrated closely with human opert to build confidence that there is no significant difference between human grading and model grading. To avoid hallucinations, give LLM one.&quot;Way out.&quot;For example, give an instruction to return when it does not have sufficient information &quot;Unknown&quot;I'm sorry. It is also helpful to create a clear, structured rubric to do the drawing of each dimension of the task, and then to use a separate LLM-as-judge to do the drawing of each dimension, instead of a single dimension. Once the system is robust, it's enough to use human review occasionally.</p>
<p>Some evals have delicate mode, which can lead to low scores even in cases where anent is performing well - anent is unable to resolve it because of grating bugs, angent constraint or ambiguity. Even a seasoned team could miss these problems. For example, Opus 4.5 scored 42% on CORE-Bench until an Anthropic researcher discovered several problems: rigid grading will &quot;96.12&quot; I'm looking forward to it. &quot;96.124991……&quot;, blurry task spec, and random task that cannot be accurately reproduced. Upon repairing bugs and using less bound scaffold, the fraction of Opus 4.5 jumped to <a href="https://x.com/sayashk/status/1996334941832089732">95%</a>I'm sorry. Similarly, METR found several configuration errors in its time horizon benchmark: they asked anent to optimize the score threshold for a declaration, but then it was required to exceed it. This punishes a model that follows instructions like Claude, while ignoring the stated goal of the mark gets better scores. Carefully double-checking the questions and the graders can avoid these problems.</p>
<p>Get your graders to have resistance to bypass or hack. Agent should not be able to be so easy. Land&quot;Cheating.&quot;Pass eval. Task and grader should be designed to make it really necessary to solve problems through eval, not to exploit unexpected loopholes.</p>
<h3>Maintain and use the eval long-term</h3>
<p><strong>Step 6: Check the transcripts</strong></p>
<p>You're not gonna know if your grader works well unless you read a lot of trail's transcript and glade. In Anthropic, we invest in the construction of tools to view eval transports and we regularly spend time reading them. When a task failed, Transcript told you that angent was really making a mistake or that your grader refused a valid solution. It also often reveals the key details of the behavior of angent and eval.</p>
<p>Failure should look fair: angent should be clear about what's wrong and why is wrong. When the score does not rise, we need confidence is the reason why angent is acting, not the reason why eval is. Reading translate is the way you verify that eval is really measuring something that really matters, and it is a key skill in the development of an individual.</p>
<p><strong>Step 7: Monitor for capability eval saturation</strong></p>
<p>A 100% eval can track return, but cannot provide an improved signal. Eval satellite takes place when an individual passes all the solvency of the task, leaving no room for improvement. For example, the SWE-bench Verified scores started at 30% early this year, while the front line model is now approaching saturation, over 80%. As eval approaches saturation, progress will slow, as only the most difficult is the task. This may result in deceptive results, as the enormous increase in capacity is reflected in a small increase in scores. For example, Qodo, a code review start-up company, initially had little impression of Opus 4.5 because one-shot working eval had not captured a longer, more complex upgrade on the table. In response, they developed a new framework for anagentic eval, which provides a clearer picture of progress.</p>
<p>As a matter of principle, we will not consider the eval fraction as a surface value until there is a deep dig-in detail and a reading of some transcript. If the grating is unfair, the task is vague, effective, and the solution is punished, or the harms limit the model, then the eval should be revised.</p>
<p><strong>Step 8: Keep evaluation suites healthy long-term through open contribution and maintenance</strong></p>
<p>An eval suit is a living piece of work that requires constant attention and clear ownership to remain useful.</p>
<p>In Anthropic, we tested various methods of maintaining eval. The most effective way to do this is to create a dedicated eval team to have a core infrastructure, while field experts and product teams contribute most of the eval task and run their own eval.</p>
<p>For the AI product team, ownership and iterative eval should be as routine as maintaining unit test. The team could be...&quot;It works in early testing.&quot;But AI failed to meet unspecified expectations, which were functionally wasted weeks -- and a well-designed eval could have revealed them earlier. Defines eval task as one of the best ways to test the demand for a pressure test product to be specific enough to start building.</p>
<p>We recommend practice eval-driven development: build an eval to define these capabilities before an individual can meet the desired capabilities, and then it's done well in succession until an individual. Inside, we often build today.&quot;That's good.&quot;And the functions, but they are actually a bet on the power of the models a few months later. The low pass rate of capability eval makes this visible. When the new model is released, running suit can quickly reveal which bets have been rewarded.</p>
<p>The closest to product demand and user is the best person to define success. With current model capabilities, product managers, customer successful managers or salesmen can use Claude Code to contribute an eval task in the form of PR -- let them do it! Or, better yet, give them the initiative.</p>
<p><img src="https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F0db40cc0e14402222a179fc6297b9c8818e97c8a-4584x2580.png&amp;w=3840&amp;q=75" alt="Create an effective eval process chart"></p>
<h2>How evals fit with other methods for a holistic understanding of agents</h2>
<p>Automation eval can run thousands of tasks on angent without having to deploy to the production environment or affect real users. But it's just one of many ways to understand angent's performance. The complete picture also includes the description monitoring, user feedback, A/B testing, human translation review and systematistic human evaluation.</p>
<table>
<thead>
<tr>
<th>Methodology</th>
<th>Advantages</th>
<th>Disadvantages</th>
</tr>
</thead>
<tbody><tr>
<td><strong>Automated evals</strong>: programmable run tests without real user</td>
<td>Faster iteration.<br>It's completely recapable.<br>Without prejudice to users<br>Run on every session<br>Large-scale testing without production deployment</td>
<td>More upfront investment to build.<br>As the product and model evolve, it needs to be maintained continuously to avoid drift<br>If true usage patterns are not matched, false confidence can be created.</td>
</tr>
<tr>
<td><strong>Production monitoring</strong>: Tracking indicators and errors in the online system</td>
<td>The massive behavior of the real users.<br>Capture the problem of the synthetic eval<br>Provide ant actual performance ground truth</td>
<td>Passive; problem reached user before you knew.<br>The signal may have a noise.<br>Need investment<br>Lack of ground truth for grading</td>
</tr>
<tr>
<td><strong>A/B testing</strong>: Compare variants with real user flows</td>
<td>Measure real users outcome (retention, task completion)<br>Control of mixed factors<br>Can scale and systematize</td>
<td>Slow; it takes days or weeks to achieve visibility and requires sufficient flow<br>Only test your deployment changes.<br>Bottom of indicator change without careful review of transcript&quot;Why?&quot;There's less signal.</td>
</tr>
<tr>
<td><strong>User feedback</strong>: visible signals, such as thumbs-down or bug report</td>
<td>To expose problems you didn't expect.<br>A true example of a true human user<br>Feedback is usually related to product objectives</td>
<td>Slight and self-selected.<br>I think it's a serious problem.<br>Users rarely explain why something failed.<br>Non-automated<br>Relying primarily on users to detect problems could have negative user impacts</td>
</tr>
<tr>
<td><strong>Manual transcript review</strong>: Human Reading angent Dialogue Record</td>
<td>Create a hunch for fair Mode<br>Capture the minor quality of the omission of automated checks<br>Help calibration&quot;Okay.&quot;And take care of the details.</td>
<td>Time-intensive<br>Cannot scale<br>Inconsistent coverage<br>Examiner fatigue or different reviewers may affect signal quality<br>Usually only give a qualitative signal, not a clear quantification</td>
</tr>
<tr>
<td><strong>Systematic human studies</strong>: structured by trained evaluators for angent output</td>
<td>Standard quality judgement of gold from multiple human evaluators<br>dealing with subjective or vague<br>Signals for improvement of model-based grader</td>
<td>Relatively expensive and slow to recycle<br>It's hard to run on a lot of times.<br>Inter-rader Difference Needed Conciliation<br>Complex areas (legal, financial, medical) require human experts to conduct research</td>
</tr>
</tbody></table>
<p>These methods correspond to different stages of the development of angent. Automated eval is particularly useful before release and in CI/CD as a line of defence against quality, running at every occasion change and model upgrade. The project monitoring process is launched after the release, detecting drift and unexpected real world failure. A/B testing to verify major changes after you have enough traffic. The User feedback and Transcript review are ongoing practices to fill the gap: continuous disaggregated feedback, weekly sample reading of transcript and in-depth excavation as required. Leave the stymatic human studies to calibration of LLM grader or to assess subjective output - in these scenarios, human consensus is used as a reference.</p>
<p><img src="https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2Fb77b8dbb7c2e57f063fbc8a087a853d5809b74b0-4584x2580.png&amp;w=3840&amp;q=75" alt="Swiss Cheese Model"></p>
<p>It's like in the security program. <a href="https://en.wikipedia.org/wiki/Swiss_cheese_model">Swiss Cheese Model</a>No single layer can capture every problem. When multiple methods are combined, failure through one layer is captured at the other.</p>
<p>The most effective team uses these methods: automated eval for fast iterative, programing monitoring for ground truth, periodic human review for calibration.</p>
<h2>Conclusion</h2>
<p>Without eval, the team will be caught in a passive cycle -- repairing one failure and creating another failure, and unable to distinguish between real regretion and noise. The team that worked early found the opposite: development accelerated as failure became a test case, test case prevented regretion, indicators replaced speculation. Eval, give the whole team a clear slope to climb, will&quot;I'm not feeling well, Agent.&quot;Turned into something that was operational. It's gonna accumulate, but it's only if you treat eval as a core component, not as an ex post remedy.</p>
<p>The pattern varies according to the type of anent, but the basic principles described here are constant. Start early, don't wait for the perfect suit. Get real from the failures you see. Define a non-mistakable and robust success criterion. Carefully designed and grouped multiple types. Ensuring that the problem is hard enough for the model. The word "val" is used to raise the noise ratio. Read it!</p>
<p>AI anent eval remains an emerging, rapidly evolving field. As angent assumes longer task, collaborates in multi-agent systems and handles increasingly subjective work, we will need to adapt our technology. As we learn more, we will continue to share best practices.</p>
<h2>Acknowledgements</h2>
<p>Written by Mikaela Grace, Jeremy Hadfield, Rodrigo Olivares, and Jiri De Jonghe. We&#39;re also grateful to David Hershey, Gian Segato, Mike Merrill, Alex Shaw, Nicholas Carlini, Ethan Dixon, Pedram Navid, Jake Eaton, Alyssa Baum, Lina Tawfik, Karen Zhou, Alexander Bricken, Sam Kennedy, Robert Ying, and others for their contributions. Special thanks to the customers and partners we have learned from through collaborating on evals, including iGent, Cognition, Bolt, Sierra, Vals.ai, Macroscope, PromptLayer, Stripe, Shopify, the Terminal Bench team, and more. This work reflects the collective efforts of several teams who helped develop the practice of evaluations at Anthropic.</p>
<h2>Appendix: Eval frameworks</h2>
<p>There are several open sources and commercial trades that can help teams implement an individual eval without building infrastructure from zero. The right choice depends on your anent type, the existing technology warehouse, and whether you need offline evaluation, regulation or both.</p>
<p><strong><a href="https://harborframework.com/">Harbor</a></strong> Designed to operate ant for containerized environments, provide infrastructure for large-scale operation of trial across cloud providers and standardized formats for defining task and grader. The popular benchmark, like Terminal-Bench 2.0, is published via Harbor registry, making it easier to run existing benchmarks and custom suit.</p>
<p><strong><a href="https://www.braintrust.dev/">Braintrust</a></strong> It's a platform for combining offline education with protection against exploitation and acquisition tracking -- It is useful for teams that need to evolve in the development process while monitoring the quality of the production environment. Other <code>autoevals</code> The library contains pre-construction scorer for practice, relevance and other common dimensions.</p>
<p><strong><a href="https://docs.langchain.com/langsmith/evaluation">LangSmith</a></strong> The project provides access to technologies, offline and online education, and data management, closely integrated with the Langchain ecology.<strong><a href="https://langfuse.com/">Langfuse</a></strong> As an open source alternative to hosting, a similar capability is provided and suitable for teams with data presence needs.</p>
<p><strong><a href="https://arize.com/">Arize</a></strong> Provides Phoenix - an open source platform for LLM traching, debugging and offline/online evaluation, and AX - a SaaS product for scale, optimation and monitoring extension of Phoenix.</p>
<p>Many teams use multiple tools to build their own eval framework, or simply use simple eval scripts as a starting point. We found that while the framwork can be a valuable way to accelerate progress and standardize, their good or bad will depend on the eval task you run through them. It's usually best to quickly choose a framework that suits your workflow, and then to focus on eval itself - a high-quality test case and a grader.</p>
