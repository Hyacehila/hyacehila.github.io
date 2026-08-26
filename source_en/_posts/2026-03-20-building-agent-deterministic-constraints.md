---
title: 'Putting Deterministic Guardrails Around LLMs: Agent Harness Engineering from Claude Code'
title_zh: 给 LLM 戴上确定性枷锁的外围工程：从 Claude Code 看 Agent Harness
date: 2026-03-20 21:00:00 +0800
categories:
- Agent Systems
- Agent Architecture
tags:
- Agent Harness
- Context Engineering
- Reliability Engineering
- Claude Code
author: Hyacehila
excerpt: What makes agents deliverable is not only the core loop, but the surrounding engineering that turns language requests
  into tool contracts, routing, validation, isolation, recovery, and governance.
description: What makes agents deliverable is not only the core loop, but the surrounding engineering that turns language
  requests into tool contracts, routing, validation, isolation, recovery, and governance.
excerpt_zh: 真正让 Agent 变成可交付系统的，不是核心 loop，而是围绕 LLM 不确定性搭出来的外围工程：把语言请求进一步下沉成工具契约、知识路由、生命周期验证、隔离恢复与自治治理。
permalink: /blog/2026/03/20/building-agent-deterministic-constraints/
lang: en
translation_key: 2026-03-20-building-agent-deterministic-constraints
translation_status: machine
translation_source_hash: b85974e82a107ad9eae9d4c08865314dfa3eb213372586e43e0e800858888ff3
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<p>If we're going to study today, <code>LLM</code> I'd rather start with Claude Code now, with the certainty shackles. The reason is not that it solved the problem, but that it exposed the key structure in Harness directly to the developers:<code>MCP</code>、<code>CLAUDE.md</code> and <code>rules</code>、<code>hooks</code>、<code>subagents</code>、<code>checkpointing</code>、<code>permission modes</code>、<code>plugins</code>、<code>Agent SDK</code>I'm sorry. In a system that is really available, you can almost see directly how language constraints are being gradually consolidated into systems.</p>
<p>Turning Agent into a deliverable system is often not the core loop itself, but the center around. <code>LLM</code> Uncertainty comes out of a whole set of peripheral works. The core cycle is of course important, but it is only the shortest part.</p>
<p>The longest and most expensive part is to put this cycle in a controlled system. The focus is not on making it think better, but on keeping the system out of control after misthinking, making mistakes, mistook tools, drifting context and running dozens of rounds.</p>
<p>Almost everyone who used Claude Code seriously went through the same phase: starting to go crazy. <code>CLAUDE.md</code> It's written in.</p>
<p>The causes are usually ordinary. Claude changed the unchangeable document, so you added "Don't move this directory." It was announced that it had been completed without running tests, so you added the "must run first" test. It's using the wrong package manager, at strange times. <code>commit</code>Or change the unconnected document together. <code>CLAUDE.md</code> The longer it goes, the more it goes from several lines to dozens, the more it goes to hundreds.</p>
<p>And then you find out: it's still gonna do it.</p>
<p>Not exactly the same mistake, but the variant. You wrote, "Don't be direct. <code>push</code>♪ It's not ♪ <code>push</code> Yes, but it's starting to happen at a strange time. <code>commit</code>I'm sorry. You wrote "You have to run the test first," and it ran, but it ran off without the relevant test files. You're naturally thinking about a model that's not strong enough to be stronger.</p>
<p>But soon you'll realize that the problem is not just a model. The problem is you're still asking for a probabilistic language model to do it your way.</p>
<p>Language requests are certainly useful, but they are natural and probabilistic. The longer the mission, the more tools, the more dirty the context and the more complex it becomes, the more it will decline. You go. <code>CLAUDE.md</code> Each rule is in the context of the length and attention drift and the alternative race.</p>
<p>This is the question I'm trying to summarize with the "confirmity shackles":<code>LLM Agent</code> For the first time, the system has the capacity to deal with the uncertainties of an open world, but the engineering system is the first to do exactly what it does. <code>LLM</code> The uncertainty itself is shackled.</p>
<p>And that is why Agent can deliver, not simply by continuing to fill the "next time" sentence in the document, but by consolidating the layers of those sentences into a system structure: which tools exist and which ones do not exist; which ones are permanently present and which ones are injected as needed; which actions are intercepted at the life cycle nodes; which tasks must be performed in the isolated context; and which "declarations of completion" are directly rejected by the certification system.</p>
<p>That's exactly what I understand now. <code>harness</code> Problem. I'll discuss the word itself in the next one, but this one is more concerned about how these peripherals grow in the product.</p>
<p>It's worth noting that the article was originally published on the date of release. <code>2026-03-20</code> And then, Anthony was there. <code>2026-03-24</code> and <code>2026-03-25</code> The long-time application of the Harness design and the long-time development were published separately. <code>auto mode</code> The safe house. Two layers were completed that had not been previously developed:<strong>Recovery and validation of operational time structures and autonomous governance.</strong></p>
<h2>Why is language constraints declining?</h2>
<p>This is not an invalid reminder, but a clear line of meaning.</p>
<p>Claude Code's official file keeps on <code>CLAUDE.md</code> Discuss it on the line “memoory/context”, rather than as a mandatory configuration document.<a href="https://code.claude.com/docs/en/memory">Store instructions and memories</a> It's very straightforward:<code>CLAUDE.md</code> And auto memoory is the context, not the context. The constraints of de facto, permanent and cross-mission stabilization fit into the <code>CLAUDE.md</code>; once entry conditions are loaded, path matching, life cycle interception, permission to cut and sandbox boundaries, the matter is no longer simply “writing a document”.</p>
<p>That's a good point.</p>
<p><code>CLAUDE.md</code> The strength of this is that the model gets a set of important facts at the beginning of the session. And its weaknesses are clear:<strong>It still lives in context, not in system structures.</strong></p>
<p>So as long as the task is complicated, language constraints are faced with several old problems at the same time:</p>
<ul>
<li>It'll be forgotten.</li>
<li>It's gonna be misunderstood.</li>
<li>It'll be "reasonablely adapted" when it comes to the field.</li>
<li>It will decrease weights in the second half of the long chain.</li>
</ul>
<p>Writing a hint should no longer be the main line of the Agent project. The hint is of course important, but it is more like the closest and softest layer of constraint to the model. A mature system moves some of the hopes that were originally sustained by the hints out of the model.</p>
<h2>Technical engineering layers and Harolds of Claude Code</h2>
<p>If I take this thing apart again, I'd rather give it to you now. <code>LLM</code> The placing of a definitive shackle is understood as a seven-layer technical engineering. They're not part of a framework. <code>feature list</code>It's not from a certain manufacturer. <code>marketing terminology</code>And it's a controlled layer that's gonna grow out of a reliable Agen system sooner or later.</p>
<p>The first two layers deal with “what status the system receives and what process it moves along”. Structured input output layer to allow the system to accept only the recipient <code>parser</code>、<code>validator</code> and <code>schema</code> The test states, not something that looks like an answer; the control flow and mission ring is responsible for putting Agent in a terminal, yes. <code>budget</code>Yes. <code>checkpoint</code>, back-up or interferocity processes that do not allow planning to spread indefinitely.</p>
<p>The three layers of treatment are “what it touches, what it sees, who can stop it”. Tools and Runtime Layers <code>shell</code>、<code>browser</code>、<code>API</code> Such capabilities result in contracts with parameters, side effects boundaries and context budgets; context and system hint layers determine which knowledge is permanently located, which is injected as required, which previously indexed and document entry forms enter working memory; gateways, traffic and access layers are responsible <code>rate limit</code>、<code>auth</code>、<code>quota</code>, audit and access to borders, with greater certainty, to the new offensive and cost aspects introduced by the hedge model.</p>
<p>The last two layers deal with “if it's finished and if it's not rotting in the long run”. The Assessment, Certification and Recovery Level is responsible for proving that the mission is actually completed in the external environment and that it is capable of being successful after failure <code>replay</code>、<code>rollback</code>; progress; observational, operational and governance levels responsible for drifting, bad patterns replication,<code>AI slop</code> And entropy, so that one error is not re-used in the system. Claude Code exposed to developers is the product-based surfaces that correspond to the more abstract control projects.</p>
<p>Claude Code is worth analysing right here. Officially even. <a href="https://code.claude.com/docs/en/how-claude-code-works">How Claude Code works</a> It is defined directly as “the anticipatory harms around Claude”. So the question is no longer what Claude can do, but:<strong>This Harness, which language-based constraints were used to sink into the structural elements that were visible in the product.</strong></p>
<p>I understand. <code>LLM</code> The abstract control layer, with the certainty chain, then looks at what Claude Code has made visible design in the product:</p>
<table>
<thead>
<tr>
<th>Claude Code Visible Design Noodles.</th>
<th>What does it look like in the product?</th>
<th>It really solves the Harness problem.</th>
<th>Which layers are the main maps of the project?</th>
</tr>
</thead>
<tbody><tr>
<td>Tool Compact Level</td>
<td><code>MCP</code>、<code>built-in tools</code>、<code>tool search</code></td>
<td>Defines the cost of action compacts, cut-off capability boundaries, control tools entering the context</td>
<td>Tools and running time layer, partial access layer, partial context layer</td>
</tr>
<tr>
<td>Context route layer</td>
<td><code>CLAUDE.md</code>、<code>.claude/rules/</code>、<code>imports</code>、<code>auto memory</code>、<code>managed settings</code></td>
<td>Decide which knowledge is resident, which is fed on condition, which is mandatory by the system</td>
<td>Context layer, partially governance layer</td>
</tr>
<tr>
<td>Life cycle certification layer</td>
<td><code>hooks</code>、<code>prompt hooks</code>、<code>agent hooks</code></td>
<td>Before, after, before and after <code>compaction</code> Insert a certifier and state recall before and after</td>
<td>Control the troposphere, verify the recovery layer</td>
</tr>
<tr>
<td>Segregation and recovery floor</td>
<td><code>subagents</code>、<code>sessions</code>、<code>checkpointing</code>、<code>--fork-session</code></td>
<td>Segregate context, cross-task path, roll back file state, visible <code>handoff</code></td>
<td>Control troposphere, context layer, recovery layer</td>
</tr>
<tr>
<td>Autonomous governance</td>
<td><code>permission modes</code>、<code>protected paths</code>、<code>sandbox</code>、<code>auto mode</code></td>
<td>Reconciliation of autonomy, hedge <code>approval fatigue</code>Interrupting high-risk cross-border movements</td>
<td>Authority, gateway and governance level, security level</td>
</tr>
<tr>
<td>Available Harness</td>
<td><code>plugins</code>、<code>GitHub Actions</code>、<code>Agent SDK</code></td>
<td>Pack local Harness original language into reusable, deployable, integrated components</td>
<td>Platform, transport, distribution level</td>
</tr>
</tbody></table>
<p>This is not the distribution of seven layers of engineering in abstract mechanically, into seven product functions. The first seven layers are cut by control capacity and division within the system; the six design layers here are cut by what developers actually see in the product, what they can configure, what they can debug. The two are multiple maps, not one.</p>
<p>And that is why two layers do not appear directly in the later text under separate headings. More of the structured input output layer is hidden in <code>tool schema</code>、<code>hook/verifier</code>、<code>checkpoint</code> and mission status constraints; the observation, transport and governance layers are scattered <code>permissions</code>、<code>sessions</code>audit,<code>OpenTelemetry</code>and distribution of plugins and <code>SDK</code> Interface these surfaces. The later text continues to be based on six visible design surfaces, but each section returns to the control layer behind it.</p>
<h2>i. Tool compact level:<code>MCP</code> It's not just an expansion, it's a world of compressed action.</h2>
<p>Many people first come in contact. <code>MCP</code>And it's going to be like, "Let Claude do more."</p>
<p>That is not a misreading, but only half.<code>MCP</code> Deeper value, not in terms of capacity expansion, but in the delineation of capacity boundaries.</p>
<p>Anthropic is here. <a href="https://www.anthropic.com/engineering/writing-tools-for-agents">Writing effective tools for agents</a> It is a very important judgement: traditional functions and API, which are the contracts between a certainty system and a certainty system; and a tool, which is the new contract between a certainty system and antagon. Which means the tool layer is not about exposing more movements, but about...<strong>Write action into an agreement that ant can be safely used and systems can stabilize consumption.</strong>。</p>
<p>Put it on Claude Code, and this judgment will become very specific.</p>
<p>No, I'm not. <code>MCP</code> Claude is likely to be able to guess the path, the authentication method, the order sequence at the terminal. It may be true, of course, but every implementation path carries a field inference.</p>
<p>Got it. <code>MCP</code> After that, things were rewritten. Claude no longer needs to guess himself “where, how, how, how to restore the database when it fails”, and it sees a set of tools that have been cut, named, and bound.<code>query_database(sql: string)</code>、<code>get_ticket(id)</code>、<code>search_internal_docs(query)</code> The real importance of this interface is not more functional, but...<strong>The details of the realization have been taken away, and the rest is the part of the capability that the system would like to see.</strong></p>
<p><strong>The capacity boundary is the structured boundary.</strong> No, I'm not. <code>delete_user_data</code> This tool, this move is not just "no recommendation" but "no existence" at the technical level. The minimum-authority principle in this and traditional software is the same thing, except that it is now applied to the design of the agent tool.</p>
<p>But only the ones that... <code>MCP</code> It is not enough to understand the boundaries of competence. Claude Code also tied the floor directly to the context budget.<a href="https://code.claude.com/docs/en/how-claude-code-works">How Claude Code works</a> and <a href="https://code.claude.com/docs/en/mcp">Connect Claude Code to tools via MCP</a> The following are clearly mentioned:<code>MCP</code> Tool solutions default to delay loading by tool search, Claude sees only the toolname, and the real schema enters the context when needed. This means that the tool layer is defined beyond the capability boundary.<strong>Context Cost Boundaries</strong>。</p>
<p>This is really the key. The export design of the tool will eventually enter the context of the model. If a tool is to plug back a whole page of database, whole HTML or complete log, the contamination is not a single call, but a subsequent entire chain of reasoning. Good tool design, filtering, compression and field selection at the service end; good tool discovery mechanism, controls how many tools describe the actual entry context.</p>
<p>The blogger says:<code>MCP</code> The layers are not just narrow motion spaces, but they're defining them in advance. <strong>Action language for angent</strong>I'm sorry. Whether the language is structured, borders are stable, returns are verifiable, determines directly how the tracks are divided, how behaviour is audited, how procedurally it is evaluated, and which steps in the closed circle are firmly written into verifier and reward. The tool compact is not only for one call, but it is also pre-forming a syntax for learning, assessable, auditable agent behaviour.</p>
<p>So... <code>MCP</code> This floor has never been just Claude's call to the outside world, but three things:</p>
<ul>
<li>Tool compacts must be defined in a system and cannot be given to model live guesses.</li>
<li>The action space must be proactively cut, not be restricted by prompt after the event.</li>
<li>The timing and particle size of the tool entering the context are also part of the Harness design, as both the tool description and the tool output reverse the subsequent reasoning.</li>
</ul>
<p>That's why OpenAI is here. <a href="https://openai.com/index/equip-responses-api-computer-environment/">From model to agent: Equipping the Responses API with a computer environment</a> The site will sink these things into the original platform language. The words may be different, but the engineering facts are identical:<strong>The real hard part is never "for a model to do it," but "for a model to do it in a world of bound action."</strong></p>
<p>From the perspective of the control layer, the drop point of this section is primarily the tool and running time layer, while also influencing the control of the cost of the permission boundary and context.</p>
<h2>II. Context router layers:<code>CLAUDE.md</code>、<code>rules</code>、<code>auto memory</code> and <code>docs index</code> Division of labour</h2>
<p>With more reliable tools, the second category will soon emerge: Claude will still make decisions that are not in line with team habits.</p>
<p>Like what you've abandoned. <code>API</code> Version, not following what is already in place <code>Repository</code> The model, or the process that should have been used only to screen the alarms, is used to handle routine development tasks. At this point, the most natural move continues. <code>CLAUDE.md</code> Refilling rules.</p>
<p>The direction is correct, but the carrier is often not good enough.</p>
<p>Claude Code's memory system is now divided into layers, not a single one. <code>CLAUDE.md</code>：</p>
<ul>
<li><code>CLAUDE.md</code> and <code>CLAUDE.local.md</code> Responsible for permanent statements.</li>
<li><code>.claude/rules/*.md</code> It's responsible for cutting the instructions into modules and allowing them to be used <code>paths:</code> <code>frontmatter</code> Make conditions for loading.</li>
<li><code>@path</code> <code>import</code> Take care of it. <code>repo docs</code>、<code>README</code>, process document is firmly attached to the entry.</li>
<li><code>auto memory</code> And I'm in charge of making Claude himself sink across sessions.</li>
</ul>
<p>The engineering logic behind this is:<strong>Every time Agent fails is a hidden knowledge that has not been encoded into the system or placed on the correct loading level.</strong></p>
<p>That's right here, Vercel. <a href="https://vercel.com/blog/agents-md-outperforms-skills-in-our-agent-evals">AGENTS.md outperforms skills in our agent evals</a> There is still a great value for reference. The most important conclusion is not that <code>AGENTS.md</code> We're beat. <code>skills</code>And it is:<strong>The knowledge exposure sequence is itself part of the binding strength and the design of the exposure mechanism/text route itself is important.</strong></p>
<p>Their results are straightforward:</p>
<ul>
<li>When there is no document,<code>baseline</code> The pass rate is 53%.</li>
<li>Default <code>skills</code> When triggered, the result was almost no improvement, still 53 per cent.</li>
<li>Visible tip model to use <code>skills</code> , up to 79%.</li>
<li>Put the compressed document index directly into the base of the warehouse <code>AGENTS.md</code> When did it do 100%.</li>
</ul>
<p>What really happened here, not in a higher format, was:<strong>Did the system continue to give the model a decision on whether to read it or not?</strong></p>
<p>So I'd rather go to the next harder conclusion:</p>
<ul>
<li><code>CLAUDE.md</code> Resolving the facts of the resident status.</li>
<li><code>.claude/rules/</code> Resolves the injection and path matching.</li>
<li><code>auto memory</code> Address cross-session learning.</li>
<li><code>repo docs</code> Keep doing it. <code>system of record</code>。</li>
<li><code>managed settings</code> It is responsible for taking back constraints that should not be maintained by language to the mandatory level of the client.</li>
</ul>
<p>In that sense, it's really worth learning.<strong>The adaptation of knowledge assembly to knowledge pathways.</strong></p>
<p>In the abstract, the central point of discussion here is context and system hint layers, but it is a more sophisticated knowledge route in the product.</p>
<h2>III. Life cycle certification layers:<code>hooks</code> It's not just a script, it's an external one. <code>verifier</code> Interface</h2>
<p>The third floor, the problem will be tightened again.</p>
<p>Suppose you have a reasonable tool boundary and a key process in it. <code>skills</code> Or in the warehouse file, Claude may still say "completed" at the end of the long mission, but you look at it, the test didn't run, or you ran wrong.</p>
<p>And then you'll find:<code>请确保测试通过</code> The sentence is essentially a language request.</p>
<p>Claude Code. <a href="https://code.claude.com/docs/en/hooks">Hooks reference</a> It is important because it takes this matter directly from the reminder to the life cycle event. You're not just saying "Do it" to the model, but you're just saying, <code>SessionStart</code>、<code>InstructionsLoaded</code>、<code>PreToolUse</code>、<code>PostToolUse</code>、<code>Stop</code>、<code>PreCompact</code>、<code>PostCompact</code> These nodes are attached to external logic.</p>
<p>The significance of this is not to automate the script itself, but to:<strong>The completed statements, the call of tools and the context compression of these events, which were originally only in the model narrative, were taken back for the first time by the external system.</strong></p>
<p><code>PreToolUse</code> The value is to turn cross-border movements into things that can be blocked before they occur. For example, prohibiting high-risk orders, limiting dangerous paths, and preventing certain types of writing. The constraints are no longer “don't do this”, but “you can't do it”.</p>
<p><code>Stop</code> It's worth more. It has completed the model itself into a system event that can be rejected by the certifier. Official documents are now more than just supported <code>command hook</code>And you're still supporting it. <code>prompt hook</code> and <code>agent hook</code>I'm sorry. The blogger adds:<code>Stop</code> Not just running. <code>shell</code> Script. It can pull up a tool. <code>verifier subagent</code> Check the tests, read the files, compare the work, and decide whether to allow the session to end.</p>
<p>This step is critical because it shows that Claude Code's life cycle is not just an event echo, but it's officially opened at the product level. <strong><code>generator / verifier</code> Separation</strong> the interface.</p>
<p>It's just in time with Anthropic. <code>2026-03-24</code> Issued <a href="https://www.anthropic.com/engineering/harness-design-long-running-apps">Harness design for long-running application development</a> The issue of the "Mixed" is a matter of concern. The article clearly describes the long-term application as <code>planner / generator / evaluator</code> Three proxy structures: generator for propulsion, evaluator for scoring, search <code>bug</code>, pull back the output to the specification. This is actually a more common Harness principle:<strong>Don't let one. <code>agent</code> It is responsible for both output and unlimited confidence in its own statement of completion.</strong></p>
<p>From this perspective, Claude Code. <code>hooks</code> It's not just an automated little script, it's a little bit of a... <code>verifier</code> Enter the standard run-time port.</p>
<p><code>InstructionsLoaded</code>、<code>SessionStart</code>、<code>PreCompact</code>、<code>PostCompact</code> Another more subtle problem is addressed: context compression and evaporation. One of the most dangerous things in a long mission is not that the model was initially not bound, but that it lost the constraint during the compression, recovery and switching phases. The key state is placed in documents, scripts, checkpoints and life cycle injections to avoid the natural evaporation of the knowledge that was clearly said earlier.</p>
<p>The most important engineering judgment on this level is that I'm going to write it very heavily:</p>
<ul>
<li><code>PreToolUse</code> The police have been responsible for blocking cross-border movements before the movement occurs.</li>
<li><code>Stop</code> Responsible for turning the “complete declaration” into a system event that can be rejected.</li>
<li><code>prompt hooks</code> and <code>agent hooks</code> It means that the certifier itself has been manufactured, not just... <code>shell glue</code>。</li>
<li><code>PreCompact / PostCompact / SessionStart / InstructionsLoaded</code> It is responsible for putting critical states outside the context window and allowing you to debug “what exactly it loads”.</li>
<li>The closer security and mandatory constraints are, the more certainty should be used to achieve, rather than revert to <code>LLM</code> - Judge.</li>
</ul>
<p>That's why Anthropic is here. <a href="https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents">Demystifying evals for AI agents</a> It's very clear. <code>agent harness</code> and <code>evaluation harness</code>I'm sorry. The former lets the model <code>agent</code> The results are measured and aggregated. You'll find a key fact when the two are separated:<strong>The non-validation of the closed loop, Agent, is essentially simply exposed to the user for a single sample.</strong></p>
<p><code>hooks</code> It's worth it to make a part of it be only for evaluation or manual. <code>review</code> The checks that were completed were embedded in the operation in advance and it was certain that they would be executed, not <code>CLAUDE.md</code> , and then insert the phrase "test and ensure that the test passes".</p>
<p>In the seven-layer project, this section corresponds to the control flow and task loops, as well as the interface between assessment, validation and restoration.</p>
<h2>IV. Segregation, forklifting and restoration:<code>subagents</code> Just the entrance, not all of it.</h2>
<p><code>subagents</code> It is easy to be described as parallel or speed-up.</p>
<p>Of course it's not wrong, but if you see it here, you'll miss its more central engineering meaning:<strong>Disassembly a large and dirty decomposition space into a number of narrower, more manageable and more verifiable decomposition spaces.</strong></p>
<p>Claude Code. <a href="https://code.claude.com/docs/en/sub-agents">Create custom subagents</a> The document emphasizes specialized, isolated context and customized tool access. For engineering, this is a real thing, not a few more. <code>agent</code>♪ And it's every one ♪ <code>agent</code> Only one narrower problem can be solved in a cleaner context and with a more restricted set of tools.</p>
<p>One. <code>reviewer subagent</code> If only read permission, then "do not modify the file" is no longer a self-inflicted reminder based on a model, but it has no action space at all. <code>Edit</code>I'm sorry. One that runs in a quarantine. <code>subagent</code>If it fails, it will not be dirty together in the main work area. A guy who only tests and validates. <code>subagent</code>There is no need to continue work behind the back of dozens of rounds of exploration.</p>
<p>But if you zoom in a little bit, you'll find Claude Code officially isolated the whole thing for two months. <code>subagent</code> The group was expanded to a more complete group of originals:</p>
<ul>
<li><a href="https://code.claude.com/docs/en/checkpointing">Checkpointing</a> Automatically snapshot file status before each editing.</li>
<li><a href="https://code.claude.com/docs/en/how-claude-code-works">How Claude Code works</a> It means the session is local. <code>JSONL</code>, <code>resume</code>, or maybe. <code>--fork-session</code> Fork.</li>
<li><code>session-scoped permissions</code> Yes. <code>resume</code> or <code>fork</code> It is not always inherited, and the state of competence itself is treated as a visible boundary, not as a hidden continuation.</li>
<li><code>session</code> Tie to the directory, officially recommended. <code>git worktrees</code> Run parallel sessions, avoid the same. <code>session</code> Contamination in multiple terminals.</li>
</ul>
<p>This is the whole floor that's really complete: It's not just open. <code>subagent</code>And...<strong>Breaking long missions into quarantine, roll-back, fork in, and <code>handoff</code> Local status machines</strong>。</p>
<p>And that's exactly what Anthony said in 2026-03-24 in Harness. The article clearly distinguished. <code>compaction</code> and <code>context reset</code>：<code>compaction</code> Just a compression. Same. <code>agent</code> Keep running;<code>reset</code> And for the next one. <code>agent</code> A clean context, only structured. <code>handoff artifact</code> Handover is necessary. The text even goes directly:<code>compaction</code> Retain continuity, but not provide <code>clean slate</code>; and <code>reset</code> I'll pay for it. <code>handoff</code> Cost, but it's really cut. <code>context anxiety</code>。</p>
<p>From this perspective, the real value of this layer is four things:</p>
<ul>
<li>Independent context to reduce noise pollution.</li>
<li>Shrink tool sets to reduce cross-border routes.</li>
<li>Clear <code>handoff</code>Let each round know what it's about.</li>
<li>Combined <code>checkpoint</code>、<code>resume</code>、<code>fork</code> and <code>worktree</code>, limit the failure to a smaller part.</li>
</ul>
<p><strong>The limitation is not a defect, but a source of reliability.</strong></p>
<p>The more freedom the more the path is out of control; the narrower the space for decomposition, the easier the behaviour to predict. Many times, the way to really make it steady is not to continue to expand its autonomy, but to tear down tasks into more restricted, more validated and more strung-up implementation units.</p>
<p>If the control layer perspective is used instead, this section deals with how the control stream, restoration and context isolation narrow the failure radius together.</p>
<h2>V. Autonomous governance:<code>permission modes</code>Sandboxes and <code>auto mode</code></h2>
<p>Because without this layer, Claude Code still looks like a default to keep playing the permission box. <code>agent</code> Tools. But it's not like that anymore. It's actually already got a very clear one.<strong>Auto-Gradient</strong>。</p>
<p><a href="https://code.claude.com/docs/en/permission-modes">Choose a permission mode</a> The four most common models in the day-to-day world have now been organized into a clear gradient:</p>
<ul>
<li><code>default</code>: Read only by default.</li>
<li><code>acceptEdits</code>: Automatically accept the file editing and common file system commands in the work directory.</li>
<li><code>plan</code>: research and present programmes without changing source documents.</li>
<li><code>auto</code>: Not manually confirmed, replaced by a backstage security check.</li>
</ul>
<p>If you add more extremes <code>bypassPermissions</code>And it's a tactical layer, not a few. <code>UI</code> Switches.</p>
<p>More importantly, this gradient is not "anyone you want." The official document clearly makes some hard-coding of the boundary irrelevant. <code>guardrail</code>：</p>
<ul>
<li><code>protected paths</code> In any mode, they will not be allowed to go without conditions.</li>
<li><code>acceptEdits</code> Only edit and limited file system commands in the work directory are automatically approved.</li>
<li><code>bypassPermissions</code> And still will. <code>protected paths</code> Keeps the last reminder.</li>
<li><code>auto</code> Instead of "all" it is "all" to send to the sorter the high-risk action.</li>
</ul>
<p>Which means... <code>permission modes</code> It's not a user experience preference anymore, but...<strong>How autonomy is institutionalized</strong>Problem.</p>
<p>And look down, Anthropic is here. <a href="https://www.anthropic.com/engineering/claude-code-sandboxing">Beyond permission prompts: making Claude Code more secure and autonomous</a> And what's public about it is even more noteworthy. It doesn't make the sandbox a safety attachment, but it makes it safer and safer. <code>autonomous</code>* The basic conditions.</p>
<p>If the sandbox answers “which hard borders can be decentralized”, then <a href="https://www.anthropic.com/engineering/claude-code-auto-mode">Claude Code auto mode</a> The answer is another question:<strong>How to give a more fine-tracked strategy to the high self-government without returning to manual approval fatigue.</strong></p>
<p>This article says that users actually approve 93% of the total number of users. <code>permission prompt</code>I'm sorry. So, just push back all risks to manual confirmation, and then it's easy to evolve. <code>approval fatigue</code>。</p>
<p>It's introduced. <code>auto mode</code> Two layers of defense:</p>
<ul>
<li>There's one in the input layer. <code>prompt-injection probe</code>, enter in the tool result <code>agent</code> Scanning the context before it.</li>
<li>There's an output layer. <code>transcript classifier</code>, determine whether this step is actually authorized by the user and whether it crosses the trust boundary before the action is executed.</li>
</ul>
<p>It's not a unilayer, but a two-stage structure: a quick, conservative one first. <code>token filter</code> First sift, then only the bee. <code>flag</code> The actions enable the second stage with reasoning, and minimize costs and by-catch.</p>
<p>It's got a lot of agents. <code>handoff</code> It was also included in the review. The government has written that the government is not a party to the law.<code>classifier</code> Yes. <code>subagent delegation</code> The end of the mission is checked both at the end and back, because the user actually authorized the mission. <code>handoff</code> The most vulnerable is the location.</p>
<p>When you want to be self-governing, you should not just think about "How to make the Lord" <code>agent</code> More intelligent, it should be designed simultaneously to “who will judge whether this step is beyond his power”.</p>
<p>So the most valuable judgement to be retained on this level is:</p>
<ul>
<li><code>default / acceptEdits / plan / auto</code> Not only are there a few facilitation models, but it is also an autonomy gradient.</li>
<li><code>protected paths</code>、<code>managed settings</code>Together with sandboxes and sorters, they constitute a hard border of self-government.</li>
<li>The mature autonomy is not to remove approval, but to recast the logic of judgement behind approval into a system structure.</li>
</ul>
<p>More abstractly, this layer is an answer to how autonomy is governed, and therefore falls under the control of its powers and naturally spills over to detectability and governance.</p>
<h2>VI. Have you got Harness:<code>plugins</code>、<code>GitHub Actions</code>、<code>Agent SDK</code></h2>
<p>The first few floors are more like local tools to control themselves. But Claude Code has a design that is more easily underestimated: It's not just one. <code>CLI</code>And it started to wrap its Harness into distribution components.</p>
<p>Look first. <a href="https://code.claude.com/docs/en/plugins">Create plugins</a>I'm sorry. The official definition of plugins is now clear: plugins are not just loaded. <code>skill</code> A small extension, but a pack. <code>skills</code>、<code>agents</code>、<code>hooks</code>、<code>MCP servers</code>、<code>LSP servers</code>、<code>monitors</code>、<code>bin/</code> Executable and Default <code>settings</code> . A plugin can be understood as a package of portable Harness configurations.</p>
<p>Claude Code does not regard Harness as a fragmented private configuration on a user machine, but rather as a versionable, shared, assembled component. You've been doing your skills locally today.<code>agent</code>、<code>hook</code> And the monitor, it can be sealed into a plug-in tomorrow, into a team-level infrastructure.</p>
<p>Look again. <a href="https://code.claude.com/docs/en/github-actions">Claude Code GitHub Actions</a>I'm sorry. The official document is clear: it allows you to be <code>GitHub workflow</code> Run Claude Code, which is built on Claude Agent SDK and respects the warehouse <code>CLAUDE.md</code> Standards. It means the same local Harness logic that can be moved to it. <code>PR</code> Create, create,<code>issue</code> Achieved,<code>code review</code> And automate repair.</p>
<p>And finally, <a href="https://code.claude.com/docs/en/agent-sdk/overview">Agent SDK overview</a>I'm sorry. It offers "the same tools, ant loop, and context management that power Claude Code" and it's just done. <code>Python</code> and <code>TypeScript</code> Programmable interface. More importantly, it's not just one. <code>query API</code> It's over. It's the company. <code>hooks</code>、<code>subagents</code>、<code>MCP</code>、<code>permissions</code>、<code>sessions</code>、<code>checkpointing</code>、<code>OpenTelemetry observability</code> All exposed together. In the generic Single Agent task, reuse Claude Code SDK or something like PI-like Harness allows us to use readily available uncertainties to compress and the results of project contraction, Harness as a service</p>
<p>This means that the Harness behind Claude Code has emerged as a clear trend towards platformization:</p>
<ul>
<li><code>CLI</code> It's interactive surface.</li>
<li>The plugin is the configuration and capability distribution module.</li>
<li><code>GitHub Actions</code> Yes. <code>CI</code> Run the surface.</li>
<li><code>Agent SDK</code> It's programmable to embed the surface.</li>
</ul>
<p>Claude Code is not an isolated product, but...<strong>A product-based Harness Original language</strong>I'm sorry. That's why I call this floor a distribution.</p>
<blockquote>
<p>Extra-curricular: OpenAI Reponse API started to put a lot of tools, Shell, code interpreter, etc., into API, which can be understood as a kind of Harness as a service.</p>
</blockquote>
<p>This layer is not the eighth layer, but the additional layer of control that has been built up in the first layers, further placizing, developing and distributing infrastructure surfaces.</p>
<h2>If you're still writing language requests, what's a better structure usually?</h2>
<p>I would now prefer to write the following in the first six layers of the translation back into the engineering language:</p>
<table>
<thead>
<tr>
<th>If you only write language requests</th>
<th>What does a tougher structure usually grow?</th>
</tr>
</thead>
<tbody><tr>
<td>"Let it check the system, make its own tools."</td>
<td><code>MCP</code> Compact + <code>tool search</code> + Service-based filtered tool output</td>
</tr>
<tr>
<td>“Comply with team norms and catalogue boundaries”</td>
<td><code>CLAUDE.md</code> Entry + <code>.claude/rules/</code> Load Condition <code>managed settings</code> Force Layer</td>
</tr>
<tr>
<td>"Remember to check before completion."</td>
<td><code>Stop</code> <code>hook</code> + <code>prompt/agent verifier</code> + <code>generator/evaluator</code> Separation</td>
</tr>
<tr>
<td>"Don't mess up the main context and the workspace."</td>
<td><code>subagents</code> Segregation+ <code>checkpoints</code> + <code>resume</code> / <code>fork-session</code> + <code>worktree</code></td>
</tr>
<tr>
<td>"Don't bother me, but don't be silly."</td>
<td><code>permission modes</code> + <code>protected paths</code> + <code>sandbox</code> + <code>auto mode</code> <code>classifier</code></td>
</tr>
<tr>
<td>"Reuse this experience to other warehouses and operating scenes."</td>
<td><code>plugins</code> + <code>GitHub Actions</code> + <code>Agent SDK</code></td>
</tr>
</tbody></table>
<p>The table is simple:<strong>Language requests have not disappeared, but it is no longer the language requests themselves that determine the reliability of the system.</strong></p>
<h2>It's the big job the frame actually did for you.</h2>
<p>When you spread out these problems,<code>framework</code> The word automatically goes to the ghost.</p>
<p>The frame is not because people don't. Write <code>while loop</code> It's not because it's there. <code>planner</code> It's a very difficult story to write. The framework should be a big job for you, and it is a reusable agreement to freeze the Harness technology that has been seen over and over and over and over and over again:</p>
<ul>
<li>State how it is defined.</li>
<li>How the tools are packaged and delayed.</li>
<li>How the document presets, slices and routers.</li>
<li>How life cycle nodes are intercepted, verified and rebroadcast.</li>
<li>How the submissions are isolated, split, restored and <code>handoff</code>。</li>
<li>How autonomy is governed, rather than being artificially identified.</li>
<li>How these constraints are sealed into plugins,<code>CI</code> Integrated and <code>SDK</code>。</li>
</ul>
<p><strong>The framework is not the intelligence itself, but the form of the condensation of Harness technology.</strong></p>
<p>Of course you can use the frame, but you don't use the frame, which doesn't mean that the issues will disappear. And the more common reality is that you'll have to make them up to yourself. The difference is only between the re-engineering of each project or whether some of it has been pumped into a stable component.</p>
<blockquote>
<p>Extra-curricular: The existing framework includes a very simple framework (e.g. PocketFlow) to provide users with an understanding of the Agent architecture, a highly integrated framework (Langchain/Graph) to provide users with a sealed Agent structure, but simply the provision of Harness is not sufficient, and the Age of Agent requires new infrastructure to help developers develop better, but does not need too much containment to leave behind technical debt. The framework that really works and is really common is far from us.</p>
</blockquote>
<h2>Concluding remarks</h2>
<p>Back to the first one. <code>CLAUDE.md</code> It's getting longer and longer.</p>
<p>He did nothing wrong, but stopped at a softer level. Language to confine language models is like writing "no mistakes" in the staff manual: it helps, but it's not enough. The engineering answer is to put it on a working system: what tools it can touch, what norms are loaded at the right time, what actions are stopped at the life cycle nodes, what tasks are to be done in a segregated environment, who is to verify when it says “complete”, to what extent it can be decentralized, and how the restraints are brought into CI and other operating surfaces.</p>
<p>That is the judgment I would like to retain:<strong>It is not important to make models more responsive, but to keep systems bound when models do not.</strong></p>
<p>And this one says, "How exactly do these peripherals work in the product, and why Claude Code is the perfect portal to watch them."</p>
<p>I'll cut the angles off next time and discuss another question: Why do you focus on this whole circle today? <code>harness</code>And how does it actually go down, when it's explained, when it's too broad to be out of shape, and why it continues to be broken down when discussing specific engineering issues? <code>工程 harness + 产品 harness + 用户友好 harness</code> It's always better than stopping. <code>agent = model + harness</code> More important.</p>
<blockquote>
<p>This paper has undergone several rounds of revision, with no synchronization between the date of publication and the actual date of completion.</p>
</blockquote>
<h2>References</h2>
<ul>
<li>Anthropic Docs, <a href="https://code.claude.com/docs/en/how-claude-code-works">How Claude Code works</a></li>
<li>Anthropic Docs, <a href="https://code.claude.com/docs/en/context-window">Explore the context window</a></li>
<li>Anthropic Docs, <a href="https://code.claude.com/docs/en/memory">Store instructions and memories</a></li>
<li>Anthropic Docs, <a href="https://code.claude.com/docs/en/mcp">Connect Claude Code to tools via MCP</a></li>
<li>Anthropic Docs, <a href="https://code.claude.com/docs/en/hooks">Hooks reference</a></li>
<li>Anthropic Docs, <a href="https://code.claude.com/docs/en/sub-agents">Create custom subagents</a></li>
<li>Anthropic Docs, <a href="https://code.claude.com/docs/en/checkpointing">Checkpointing</a></li>
<li>Anthropic Docs, <a href="https://code.claude.com/docs/en/permission-modes">Choose a permission mode</a></li>
<li>Anthropic Docs, <a href="https://code.claude.com/docs/en/plugins">Create plugins</a></li>
<li>Anthropic Docs, <a href="https://code.claude.com/docs/en/github-actions">Claude Code GitHub Actions</a></li>
<li>Anthropic Docs, <a href="https://code.claude.com/docs/en/agent-sdk/overview">Agent SDK overview</a></li>
<li>OpenAI, <a href="https://openai.com/index/harness-engineering/">Harness engineering: leveraging Codex in an agent-first world</a></li>
<li>OpenAI, <a href="https://openai.com/index/equip-responses-api-computer-environment/">From model to agent: Equipping the Responses API with a computer environment</a></li>
<li>Anthropic, <a href="https://www.anthropic.com/engineering/writing-tools-for-agents">Writing effective tools for agents</a></li>
<li>Anthropic, <a href="https://www.anthropic.com/engineering/harness-design-long-running-apps">Harness design for long-running application development</a></li>
<li>Anthropic, <a href="https://www.anthropic.com/engineering/claude-code-auto-mode">Claude Code auto mode: a safer way to skip permissions</a></li>
<li>Anthropic, <a href="https://www.anthropic.com/engineering/claude-code-sandboxing">Beyond permission prompts: making Claude Code more secure and autonomous</a></li>
<li>Anthropic, <a href="https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents">Effective harnesses for long-running agents</a></li>
<li>Anthropic, <a href="https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents">Demystifying evals for AI agents</a></li>
<li>Vercel, <a href="https://vercel.com/blog/agents-md-outperforms-skills-in-our-agent-evals">AGENTS.md outperforms skills in our agent evals</a></li>
<li>I'm not sure.<a href="https://zhuanlan.zhihu.com/p/2021603278606087058?share_code=11INMrHCLcWKE&amp;utm_psn=2025010367944730281">Look at the design of Harness Engineering from Claude Code</a></li>
<li><a href="/en/blog/2026/03/01/structured-output-and-constrained-decoding/">"Making Agen Work, Large Model Structure Output and Limited Decoding Technologies"</a></li>
<li><a href="/en/blog/2026/06/11/agent-context-engineering/">"Context is All You Need: Context Project for Smart Bodies"</a></li>
<li><a href="/en/blog/2026/03/10/from-mcp-to-agent-skills/">From MCP to Agent Skills: Why does Agent need a new context work protocol?</a></li>
<li><a href="/en/blog/2026/03/16/aenvironment-everything-as-environment/">"AEnvirronment: Agent Need a Unified Environmental Level?</a></li>
</ul>
