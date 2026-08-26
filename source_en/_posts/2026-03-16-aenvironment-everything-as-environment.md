---
title: 'AEnvironment: Why Agent Development Needs an Interaction Environment Layer'
title_zh: AEnvironment：Agent Dev 为什么需要交互环境层？
date: 2026-03-16 21:00:00 +0800
categories:
- Agent Systems
- Agent Architecture
tags:
- Verifiers
- Evaluation
- Environment Design
author: Hyacehila
excerpt: SWE-bench, SWE-agent, and Tau-bench show that agent development cannot focus only on models and frameworks. Tasks,
  tools, users, state, rules, and validators shape whether agents work.
description: SWE-bench, SWE-agent, and Tau-bench show that agent development cannot focus only on models and frameworks. Tasks,
  tools, users, state, rules, and validators shape whether agents work.
excerpt_zh: SWE-bench、SWE-agent 和 Tau-bench 都在提醒一件事：Agent Dev 不能只盯模型和框架。任务、工具、用户、状态、规则和验证器怎样被组织成环境，会直接影响 agent 能不能工作。
permalink: /blog/2026/03/16/aenvironment-everything-as-environment/
lang: en
translation_key: 2026-03-16-aenvironment-everything-as-environment
translation_status: machine
translation_source_hash: 8692a972ebea27ef9023555da396c138df1953761600f3328e8e0da52de935bc
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<p>This Agent series has been written so far, and has been asking the same question: how does Agent actually reach the outside world?</p>
<p>The questions in this article can also be addressed<a href="/en/blog/2026/03/09/from-rl-agent-to-language-agent-v2/">Model shift and uncertainty modelling after RL Agent to LLM Agent: The Second Half</a>How the concept of a relatively close read together is developed in different contexts.</p>
<p>MCP Discussion Agreements, Skills Discussion Capability Encapsulation, Cognitive Structure Discussion Decision Cycles, RL Agent, those Discussion Trainings and Feedback. They look different, but together, they find a layer missing: the environment in which the angent runs.</p>
<p><strong>Agent = Model + Harness</strong>We used to split Agent Dev into two pieces. Models are responsible for reasoning, Harness is responsible for looping, tool calling, memoory, workworkworkwork. That's the right division, just not enough. Agent, not in a vacuum, is always placed in a specific environment: a warehouse, terminal, browser, passenger service system, database, order system, test package, rule document, and even a user who will keep asking and changing. These things determine what angent sees, what he can do, what side effects his actions can cause, whether he recovers after failure, and how the end result is verified. In other words, the dimension of the interaction with the external environment in Agent Dev seems to have been invisible in the previous study. We can say that the environment is in the Harness, of course, and here is just to emphasize the concept, and you can see in my blog on Harness the detailed discussion of Harness.</p>
<p><a href="https://github.com/inclusionAI/AEnvironment">AEnvironment</a> A good recent example of how this problem can be raised is the environmental layer of the project that is being done by inclusionai,<strong>Everything as Envirronment</strong>I'm sorry. Everything is not discussed. Let's see where the environment is going to become the core object of the project.</p>
<h2>From tools to environment</h2>
<p>The default image of Tool calling is simple: the world is broken down into a set of functions, model selection functions, filling parameters, system execution, and return the results to the model.</p>
<p>This is sufficient for many business automations. But once the task has grown longer, the abstraction is thin. Browser missions are not a search, software engineering tasks are not a code generation and passenger service missions are not a database query. They all have changed status, continuous interaction, rules constraints and trajectories.</p>
<p>And then the question is not just "What tools are given to angent?" And ask:</p>
<ul>
<li>How the world is changing behind the tool;</li>
<li>How many states can be observed at each step;</li>
<li>Can the feedback help restore?</li>
<li>How multi-step tracks are recorded and re-played;</li>
<li>How the success criteria are validated;</li>
<li>The same boundary can be shared by training, assessment and online operation.</li>
</ul>
<p>These questions are closer to environment than tool schema.</p>
<p>Use RL, angent. <code>observation -&gt; action -&gt; feedback -&gt; next observation</code>I'm sorry. LLM Agent switched the action space from control signals to language, tool call and code operation, but it still does not loop the environment as long as the mission enters the real system.</p>
<h2>SWE-bench: Software engineering tasks begin with environmental tasks</h2>
<p><a href="https://github.com/swe-bench/SWE-bench">SWE-bench</a> It's worth looking at again, not just because it makes the model look real GitHub issue. It is more important to place software engineering capabilities in an enforceable and verifiable environment.</p>
<p>A SWE-bench task includes real repository, assue description, code modification, environment reliance, testing packages, patch generation and final validation. It's not about "How to change this code," but about getting angent into a software project site: Read the code, position problems, change the file, run the tests, generate patches, and then receive feedback on the results.</p>
<p>So the benchmark design itself became the subject of research. The way the mission is sampled, the way the warehouse is prepared, the way it is fixed, the way the tests determine, the way the Patch is applied, the way the environment is isolated, changes what the benchmark actually detects.</p>
<p>If an anent is doing better on the SWE-bench, we can't just ask if the model is stronger. It depends on the environmental interface it faces: is there a stable shell? Do you have a file viewer? Is the information of the error simple? Can the feedback from the test be restored? Is there a useful trajectory in the context, or is there a pile of expired outputs?</p>
<p>This is why SWE-bench was often discussed with the other parties, the technical Sandbox, the evaluation runner. Software engineering angent capabilities are no longer just code generation capabilities, but the ability to act and be validated in a real engineering environment.</p>
<h2>SWE-atent / ACI: Environment is not nudity shell</h2>
<p><a href="https://arxiv.org/abs/2405.15793">SWE-agent</a> Putting this issue more clearly. It's made. <code>ACI</code>That's... <code>Agent-Computer Interface</code>。</p>
<p>This is a very similar extension of HCI. Human engineers need IDE, syntax highlighting, search, debugger and error tips, and angent needs an interface that suits you. Naked Linux shell is strong for humans, but not necessarily friendly for LM anent.</p>
<p>Shell's motion space is too wide, command combinations are too free, output is often too long, and error feedback does not necessarily point to the next step. Humans can filter noise through experience and visual context, but models continue to extrapolate it into a limited context. The same computer environment, where an interface is exposed to angent, will be completely different.</p>
<p>SWE-agent's ACI does very specifically. It's working. <code>find_file</code>、<code>search_file</code>、<code>search_dir</code> Reduces the difficulty of searching and navigation by using the code; displays the contents of files with line numbers, window-based, scrollable files with file numbers; uses file edito to replace the code by line and displays the results immediately after editing; and stops syntax errors and bad editing with the Lint Guardrails.</p>
<p>These things are not related to the model itself, but they change the model's behavior directly. A good interactive layer of environment narrows the motion space, makes observations clear, turns errors into recoverable signals and compresses the state into information for the next decision.</p>
<p>This gives Agent Dev a very practical reminder: interface design is not the end of the product UI. Tools, feedback, context management, guardrails, certifiers, together, are the world that angent actually faces. A friendly interface for Agent sometimes matters more than repeated changes to React Loop and WorkFlow.</p>
<h2>Tau-bench: Instability in many ordinary missions</h2>
<p>SWE-bench and SWE-agent put the problem in the software engineering environment.<a href="https://arxiv.org/abs/2406.12045">Tau-bench</a> The first time I saw the news, I was told that I was in a business conversation with a guest.</p>
<p>Tau-bench's full name is <code>A Benchmark for Tool-Agent-User Interaction in Real-World Domains</code>I'm sorry. It is not just a measurement that the model will call a function, but it allows angent to face user dialogue, field API tools and business rules documents in areas like retail, airline.</p>
<p>Such missions do not seem to be difficult to take. Re-routing, return of goods, order checks and updates are not highly deduced. Trouble elsewhere: User information may be given in multiple rounds, and angent must clarify the missing conditions; API calls change the status of the database; business rules limit what can be done; and lastly, whether the database is really correct.</p>
<p>Tau-bench placed special emphasis on coherence. Even strong funcing anent, multiple trials will be unstable. Business actors are often troubled not by “never doing it”, but by doing the right thing, sometimes by missing the rules, sometimes by changing the wrong situation. This is more difficult than a single failure.</p>
<p>So the client's service andent environment cannot be understood as an isolated group of API. It is a system of users, tools, strategies, database status, dialogue history and verifier stacking. It is not enough to optimize models or tool schema, to design how the environment is exposed, how the rules are put into context, how the side effects of the instrument are restrained, and how the end state is verified.</p>
<h2>AEnvirronment Awareness of Problem</h2>
<p>And in this context, look at AEnvirronment, it's... <code>Everything as Environment</code> It's not like a slogan.</p>
<p>I do not think it would be good to “all things should be forced to be unified into a great abstract”. Such frameworks can easily be reduced to the lowest common multiple, and they are not easily accessible to anyone. More rationally, AEnvirronment tries to manage the time of angent running, tools, sandboxes, benchmark, rollout, reward/verifier in the same environmental perspective.</p>
<p>AEnvirronment is not an isolated project. It and AReaL (a completely different RL training system) and AWorld (Agent Time Framework) form the three-part “training-environment-run” set of ants. This ecological context is essential to understand the positioning of AEnvirron -- it is not just a generic abstraction of the environment, but a key puzzle in the ants' closed circle.</p>
<p>AEnvirronment as <code>Environment-as-Code</code> and the environment run platform, working with Age Runme and AreaL to process the entire RL task. Developer defines a reusable environment, places tools or MCP services in containerized sandboxes, creates environmental examples through SDK, calls tools, captures results, and provides standardized tool and reward services for RL / Agent scenarios.</p>
<p>It's a real pain:</p>
<ul>
<li>Benchmark should not be just an external ranking, but also a reusable environment;</li>
<li>The training environment is fragmented and the assessment of achievements is difficult to migrate;</li>
<li>Tool evaluation requires Sandbox, permissions, status and observation formats;</li>
<li>(a) Need and environmental state of reward / verifeer, not merely artificial tagging;</li>
<li>Angent tracks should be able to be recorded, replayed, analysed and improved.</li>
</ul>
<p>It really answers: can the environment itself become a first-class project when angent needs to re-engage, assess and learn in different mission worlds?</p>
<h2>Benchmark is environmentalizing</h2>
<p>From SWE-bench, Terminal-Bench, BrownserGym to Tau-Bench, benchmark is less like a static set of questions, more like an interactive environment.</p>
<p><a href="https://github.com/ServiceNow/BrowserGym">BrowserGym</a> Harmonize browser tasks with an interface similar to Gym, emphasize <code>reset</code>、<code>step</code>, observation, action and replicable trajectory.<a href="https://github.com/harbor-framework/terminal-bench">Terminal-Bench</a> Tie the terminal mission, sandbox and evaluation runner together. SWE-bench transforms real warehouse and test validation into a software engineering environment. Tau-bench combines user simulators, business rules, API tools and database status into an enterprise dialogue environment.</p>
<p>These projects do not necessarily constrict into the same framework, but they acknowledge the same: the focus of Agent benchmark is no longer just a topic, but an environmental interface.</p>
<p>The title description is just an entry. The difficulty and credibility depend on the environment: what angent can observe, what can be done, how the post-implementation state changes, whether the error is restored, whether the final state can be judged and the trajectory can be repeated.</p>
<p>This is why benchmark, mission design and interface design themselves become research content. Environmental design is too loose, angent may bypass real capability requirements; feedback design is too poor, models may not be inadequate, but may be dragged down by bad interfaces; and the training may be just rewarding, given the unstable verifier.</p>
<h2>Reward is actually an environmental product.</h2>
<p>After the environmentalization of benchmark, the next step is naturally to reward/verifier, and in the RLAgent, we discussed the insight of the Second Half: when models are already more broadly developed, bottlenecks turn to Evaluation. And one step further, the source of the evaluation is environmental feedback.</p>
<p>Silver and Sutton are in <a href="https://storage.googleapis.com/deepmind-media/Era-of-Experience%20/The%20Era%20of%20Experience%20Paper.pdf">Welcome to the Era of Experience</a> It was emphasized that AI would move from “learning more from human data” to “leting angent act in the world and learn from its consequences”. If angent is to learn from experience, there must be an environment that gives consequences.</p>
<p>AlphaProof and AlphaEvolve are strong because they have clear certifications: proof of validity, test pass and improvement of the procedure. The rewards signal is clear, and learning from experience is possible.</p>
<p>Opening the world is a much more difficult task. Whether a client's service is successful is not just whether the answer is polite, but whether the database is in the right state, whether the rules are complied with, whether the user's target is completed, whether the action is ultra vires. Whether a software project anent has been successful, and not only whether the Patch can be compiled, but also whether the test covers real assue, whether it has been introduced to hide returns.</p>
<p><strong>The closer we get to the real world, the less we reward, the less we delay, the more we're in conflict with our targets.</strong>And as angent's ability grows, the problem of rewarding and regulating games becomes more serious -- a strong enough anent may learn to look like you're optimizing the rewards you give, actually optimizing something else.</p>
<p>So Reward didn't come in empty. It's an environmental design. The more the environment is organized to understand the state, rules, side effects and the certification, the more likely it is to be assessed, trained and improved. The environment itself is chaotic, and rewards can only be returned to artificial subjective judgement or drilled into the void by models.</p>
<h2>NitroGen: What can a unified environment unlock?</h2>
<p>There's a good side to the game:<a href="https://nitrogen.minedojo.org/">NitroGen</a>I'm sorry. It packages business games like Dark Souls III, Sekiro, Black Myth: Wukong, Elden Ring into a single Gymnasium API and maps a large number of games to the same game action space.</p>
<p>What is interesting about this case is that it is not uniform. The integrated environmental interface allows researchers to do cross-play visual-action pre-training, to learn action a priori from the model in the open video, and to migrate to more than one game task.</p>
<p>NitroGen has also drawn boundaries for the unified environment. It unites the gamepad action game as a subset, not all; it learns system-1 action reflection, not full long-range planning. The value of environmental unity depends on what upper-level capacity can be unlocked after unification, rather than how complete it sounds in abstraction.</p>
<p>This judgment applies equally to AEnvirronment. It doesn't need to prove immediately that it can unify all the angent scenes. More realistically, the goal is to reduce the costs of Agent Dev by making the environment a identifiable, deployable, recapable and measurable target in some high-value scenarios.</p>
<h2>Can the unified environmental layer really do it?</h2>
<p>Aquant infrastructure, such as infrastructure, is naturally exposed to the abstract risk of a loophole: traditional infrastructure is definitive, and input A necessarily gets B; but capital based on a large model is essentially probabilistic. When you try to unify browsers, terminals, code sandboxes, moving end, benchmarks under one Environment intoterface, you face not just differences at the interface level, but at the semantic level.</p>
<p>BrownserGym has only made web-based environmental unification in one area (Webarena, MiniWoB, WorkArena), which is complicated enough. NitroGen only unites the subset of the game game and needs to build a complete engineering warehouse from video data lines to simulators. AEnvirment wants to do it all. Historically, the abstract layer that attempts to unify has tended to become either over-heavy or to degenerate into a compromise with a minimum common factor in a difficult scenario.</p>
<p>In the cognitive architecture, we discussed the emergence and the power of intelligent engineering. The integrated environmental interface is increasing certainty, which can help with training, evaluation and recurrence, but it will also limit the freedom of Agent to interact with the world.</p>
<p>Also in that article, we came to the conclusion that the new framework should be a very simple, highly liberal bottom facility, not a high-level seal. Is AEnvirron sufficient to be brief? When it tries to integrate the benchmark, RL training, anent development, multi-intelligence into one enterprise into the facility, can it maintain the lightness of the bottom facility?</p>
<p>This is not a denial, but a point of truth: the more uniform the environment, the more binding the upper Agent. How to balance unity and flexibility is a question that AEnvirron must answer on an ongoing basis.</p>
<p>From the perspective of cognitive structures, CoALA defines the external environment as the physical environment, the digital environment, interaction with humans, and interaction with other intelligent bodies. The four forms of interactive semantics vary greatly. Can a unified Envirronment interface cover them without losing their semantics? Or is it ultimately only covering some of them, leaving the real challenge to the field-specific programmes?</p>
<h2>Concluding remarks</h2>
<p>The difficulty from SWE-bench to SWE-ent, and then to Tau-bench, Agent Dev is extending from the model to the model to act in the right environment.</p>
<p>SWE-bench places real software engineering capabilities in the environment of warehouse, terminal, test and Patch validation. SWE-agent/ACI reminds us that the environment cannot be just naked shell, but also an interface designed for angent. Tau-bench puts the problem in business dialogue: a lot of ordinary tasks, rules, tools side effects, status verification, one missing one can be problematic.</p>
<p>That's what I understand about AEnvirronment. It may not be the final answer, but it captures the real question. A part of the engineering capability of Agent Dev will become an environment/interface signature. Future agents may increasingly be like designing an actionable, verifiable and recoverable environment, rather than simply a follow-up model and set of tools.</p>
<h2>References</h2>
<ul>
<li><a href="https://github.com/inclusionAI/AEnvironment">AEnvironnement GitHub repository</a></li>
<li><a href="https://inclusionai.github.io/AEnvironment/architecture/architecture.html">AEnvironment Architecture</a></li>
<li><a href="https://www.inclusion-ai.org/AEnvironment/guide/sdk.html">AEnvironment Python SDK Guide</a></li>
<li><a href="https://github.com/inclusionAI/AReaL">AReaL GitHub Repository</a></li>
<li><a href="https://arxiv.org/abs/2505.24298">AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning</a></li>
<li><a href="https://github.com/inclusionAI/AWorld">AWorld GitHub repository</a></li>
<li><a href="https://arxiv.org/abs/2508.20404">AWorld: Orchestrating the Training Recipe for Agentic AI</a></li>
<li><a href="https://github.com/swe-bench/SWE-bench">SWE-bench GitHub repository</a></li>
<li>Jimenez et al., <a href="https://arxiv.org/abs/2310.06770">SWE-bench: Can Language Models Resolve Real-World GitHub Issues?</a></li>
<li>Yang et al., <a href="https://arxiv.org/abs/2405.15793">SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering</a></li>
<li>Yao et al., <a href="https://arxiv.org/abs/2406.12045">Tau-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains</a></li>
<li>Sierra, <a href="https://sierra.ai/resources/research/tau-bench">Tau-bench: Benchmarking AI Agents for the Real-world</a></li>
<li><a href="https://github.com/sierra-research/tau-bench">Tau-bench GitHub repository</a></li>
<li>Shunyu Yao, <a href="https://ysymyth.github.io/The-Second-Half/">The Second Half</a></li>
<li>DeepMind, <a href="https://www.nature.com/articles/s41586-025-09833-y">AlphaProof: Olympiad-level formal mathematical reasoning with reinforcement learning</a></li>
<li>DeepMind, <a href="https://arxiv.org/abs/2506.13131">AlphaEvolve: A coding agent for scientific and algorithmic discovery</a></li>
<li>Sumers et al., <a href="https://arxiv.org/abs/2309.02427">Cognitive Architectures for Language Agents</a></li>
<li><a href="/en/blog/2026/03/20/reward-hacking-four-failure-modes/">Reward Hacking: When reward signals are optimized and retroverted</a></li>
<li><a href="https://github.com/harbor-framework/terminal-bench">Terminal-Bench GitHub Repository</a></li>
<li><a href="https://github.com/ServiceNow/BrowserGym">BrownserGym GitHub repository</a></li>
<li><a href="https://nitrogen.minedojo.org/">NitroGen Project Page</a></li>
<li><a href="https://arxiv.org/abs/2601.02427">NitroGen Thesis (arXiv:2601.02427)</a></li>
<li><a href="https://github.com/MineDojo/NitroGen">NitroGen GitHub Repository</a></li>
<li><a href="https://modelcontextprotocol.io/docs/learn/architecture">MCP Architecture</a></li>
<li>Silver and Sutton, <a href="https://storage.googleapis.com/deepmind-media/Era-of-Experience%20/The%20Era%20of%20Experience%20Paper.pdf">Welcome to the Era of Experience</a></li>
</ul>
