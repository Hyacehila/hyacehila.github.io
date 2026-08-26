---
title: What Is Harness? From Model Plus Harness to Engineering, Product, and User-Friendly Shells
title_zh: Harness 到底是什么：从 model + harness 到工程、产品与用户友好外壳
date: 2026-04-04 23:20:00 +0800
categories:
- Agent Systems
- Agent Architecture
tags:
- Agent Harness
- Product Design
- Evaluation
- Reliability Engineering
author: Hyacehila
excerpt: LangChain's agent equals model plus harness framing is useful only at a coarse level. In real engineering, harness
  is better understood through engineering, product, and task interfaces.
description: LangChain's agent equals model plus harness framing is useful only at a coarse level. In real engineering, harness
  is better understood through engineering, product, and task interfaces.
excerpt_zh: LangChain 关于 agent = model + harness 的说法只在粗粒度上成立；真实工程里更有解释力的是工程、产品、用户友好外壳与 task interface。
permalink: /blog/2026/04/04/understanding-agent-harness/
lang: en
translation_key: 2026-04-04-understanding-agent-harness
translation_status: machine
translation_source_hash: 7b7b4f4f642d027775765d2a9218f632387ed7e5438ff9befb11e72db489216e
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<p>Previous <a href="/en/blog/2026/03/20/building-agent-deterministic-constraints/">"Turn LLM back in the cage: From Claude Code, see how Harness is able to fix the probability into a system-bound system."</a> It's about the engineering facts: Why? <code>MCP</code>、<code>skills</code>、<code>hooks</code>、<code>subagents</code> These shells will be better than going. <code>CLAUDE.md</code> The rules are more important.</p>
<p>This article no longer rephrases those mechanisms, but answers the question of one word:<strong>Why is everyone calling this whole circle today? <code>harness</code>And what exactly does that word do?</strong></p>
<p>Around <code>harness</code> The most likely combination of opposite mistakes in the discussions is that of the opposite.</p>
<p>The first mistake was to talk about Agent too much, just focus. <code>planning</code>、<code>memory</code>、<code>tool use</code> and <code>reflection</code>It's like loop, once smart, the system will be set up.</p>
<p>The second mistake is to put <code>harness</code> The broadness of the text, which ended with a brain-drinking of product interaction, user trust, collaborative rhythm, task delivery, interface defaults, led to the word becoming less analytical, although it sounded all-encompassing.</p>
<p>That's why, yes. <code>harness</code>My attitude is that it is acceptable, but not deviating; it is available, but it continues to be decomposed on many occasions.</p>
<h2>Why? <code>harness</code> They'll be discussed at the high frequency right now.</h2>
<p>First, it is concluded that the new peripheral system is not important, but that it can no longer be pretended to be peripheral. Yes. <code>harness</code> Before the concept came out, we were also studying it, but it was not the focus of the discussion.</p>
<p>The model is strong enough to get them into the real task, and once it's done, all the problems that Demo would have hidden. The question of whether the model will drift, whether the tool will be misused, whether the document will expire, whether the permission will cross, whether the closed loop is absent, and whether the bad pattern will continue to be replicated in the system — none of which is later.</p>
<p>So-called outer-coding codes are not peripherals at all. It's the main project that turns LLM into a deliverable agent.</p>
<p>If only from the technical point of view of building a reliable Agent, it is not the smart brain in the middle that often decides the upper limit, but the shell around it that is strong enough to prove and to govern. The mature Agen is not a freer LLM, but a LLM that is carefully restrained and remains sufficiently free.</p>
<p>There are at least five simultaneous changes behind this.</p>
<p>First, model capabilities have crossed sufficient thresholds.
Second, missions begin to enter terminals, browsers, warehouses, databases, worksheet systems and long-range missions.
Third, the side effects of the tool become real, and errors are no longer just wrong answers, but rather the document is corrupted, the environment is polluted and the external system is triggered.
Fourthly, long missions and cross-boundary relays are beginning to become widespread.
Fifth, the scale of productization has come up, and the team is constantly facing file exposure, sandboxes, privileges, playback, quality governance and AI slop, which are the maintenance objects.</p>
<p>If there is to be a time anchor for this change, at least a few articles are worth referring to:</p>
<ul>
<li><code>2026-01-09</code>Anthropic is here. <a href="https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents">Demystifying evals for AI agents</a> It's a clear distinction. <code>agent harness</code> and <code>evaluation harness</code>I'm sorry. We need an outer component. <code>agent loop</code> Give enough proof to boost the final output, which is precisely because <code>agent</code> It's starting to get a wide access to the real system.</li>
<li><code>2026-02-04</code>OpenAI is in <a href="https://openai.com/index/unlocking-the-codex-harness/">Unlocking the Codex harness</a> Liang! <code>Codex harness</code> It's written across the inner nuclear boundary shared by web, CLI, IDE extension and desktop, and the concept begins to be gradually separated.</li>
<li><code>2026-02-11</code>OpenAI is in <a href="https://openai.com/index/harness-engineering/">Harness engineering</a> The interior is being carried directly to the main narrative, and modelling capacity is no longer the only bottleneck of system capability.</li>
<li><code>2026-03-10</code>Lang Chain is here. <a href="https://www.langchain.com/blog/the-anatomy-of-an-agent-harness">The Anatomy of an Agent Harness</a> Liang! <code>Agent = Model + Harness</code> The idea is to make the formula clear.<code>harness</code> The concept is being disseminated and interpreted. <code>prompt engineering</code>,<code>context engineering</code> New concept after that.</li>
<li><code>2026-03-11</code>OpenAI is in <a href="https://openai.com/index/equip-responses-api-computer-environment/">From model to agent: Equipping the Responses API with a computer environment</a> Response API started to placate these environmental operations with Agent's capabilities by further improving the Résponse API, which itself will become Harness Service directly from a simple LLM Chat.</li>
</ul>
<p>When you look at them together, you find:<code>harness</code> The presence of HF is in itself a signal of a shift in the engineering focus. The model is no longer simply being fed into the product, but is being placed in a longer, more open, more reactive work system. So you started to need a word to describe the shell that was working on the system outside the model.</p>
<h2>Why isn't it new?</h2>
<p>But if that's why you say <code>harness</code> I invented a whole new problem, and it's not right.</p>
<p>The most direct evidence is... <code>harness</code> The old usage of LLM is the same. EleutherAI <a href="https://github.com/EleutherAI/lm-evaluation-harness">lm-evaluation-harness</a> The term was used early to refer to the assessment infrastructure for mass running missions, scoring and synthesizing results. The world is being evaluated.<code>harness</code> Not new.</p>
<p>The blogger adds:<a href="https://github.com/ServiceNow/BrowserGym">BrowserGym</a> and <a href="https://github.com/harbor-framework/terminal-bench">Terminal-Bench</a> Such projects are also long overdue in drawing out browsers, terminals, reference missions and re-operable environments into more stable borders. Even if they don't always shout. <code>harness</code>They still deal with the same kind of questions: how the environment is integrated, how the movement moves into the world, how the results are checked, how the trajectory is re-played.</p>
<p>Anthropic is a good counter-example.<code>2024-12-19</code> Yes. <a href="https://www.anthropic.com/research/building-effective-agents/">Building effective agents</a> I didn't put it in the bag. <code>harness</code> It's written in the middle of the word, but it's made clear a lot of methodological theories that were then wrapped in this general term:</p>
<ul>
<li>When should we use the prescript workwork</li>
<li>When do you need to be more open?</li>
<li>Why should we start with the simplest structure?</li>
<li>Why tools design and feedback structures are more important than a heavy framework</li>
</ul>
<p>To be precise, when most experienced developers first see the word, the first reaction is often: what is it? So open the blog and look around. In half an hour, he probably finds that it's just a new name for many of his previous work, a new concept, and then it's a very talented thing to do -- "marketing-AI" -- in the word making.</p>
<p>In conclusion:</p>
<p><strong>Harness not invented new issues, but re-engineered old issues of runtime, workworkwork, stuffing, sandbox, eval, ops, into a more visible discussion.</strong></p>
<p>These problems are not the first to emerge. New, is that they're the first stage where they're becoming the Agent products and the Agent Project.</p>
<p><em>Note: The concept of harness used in this subsection is applicable<code>agent = model + harness</code>Understand, we therefore call key loop part of the harnes, not just the environment, and then we will respond to the problem further, simply to make narratives more fluid and introduce a large number of definitions that are less readable at the beginning.</em></p>
<h2><code>agent = model + harness</code> Why is it working? But not enough.</h2>
<p>LangChain's <a href="https://www.langchain.com/blog/the-anatomy-of-an-agent-harness">The Anatomy of an Agent Harness</a> The most valuable thing is that he's been putting it right in the face. <code>Agent = Model + Harness</code> It's written as a central formula. It's simple enough to make the reader understand what it is. <code>harness</code>I'm sorry. The article does have a lot of voice, and the discussions about the association of Building Harness and Model and Harness are clear and instructive.</p>
<p>This style and this article are useful and worth reading, and I fully agree.</p>
<p>Because it corrects at least one common misconception: Agent never works naked models. Just put this formula out, the reader's attention will be enough to go back to the system layer outside the model from whether the model is stronger or not.</p>
<p>But that is not enough. If you make a good-looking angent, and it doesn't work, someone just drops the "harness no," and it's usually the right one without any operational bullshit. Because for most teams, there is a very small space for targeted enhancement of Model; if you tear it too wide, it will soon become an all-inclusive concept that can almost anything.</p>
<p>As soon as the problem entered the real engineering scene, harness was no longer a single-layered object. It is composing various levels of implementation constraints, restoration and governance, human trust anchors, interactive defaults, mission delivery modalities, approval rhythms, interface frictions, mission interface design, etc. After all, if the problem is simply divided into models and harnes, then almost everything else is going to be put into harnes; in reality, it's obviously the majority of teams that don't. So if you keep taking all of this with just one one harness, it'll soon be too broad to explain.</p>
<p>So I would rather write the article in two parallel formulas:</p>
<p><code>agent = model + harness + task interface</code></p>
<p><code>harness = 工程 harness + 产品 harness + 用户友好 harness</code></p>
<p>The first is a reminder of how many of the players are competitive, not just in models and shelles, but also how the mission interface itself was designed. The dialogue box, the form, the fixed work, the status machine or some more structured workstation directly determines the stability of the system. Will<code>task interface</code>To become a first-class citizen is to respond to a problem that has been neglected in Agent Dev - The only thing that matters is that Chat is not enough. The system should be able to get something more relevant.<code>task interface</code>Like, notebookm or something special. Chat is temporary, but definitely not the end. Universalization towards exclusive use is also a trend towards the evolution of human society itself.</p>
<p>The second is to prevent us from turning <code>harness</code> It's written in a big bag without any distinction. Once we get to the point, this continues to split. <code>agent = model + harness</code> More explanatory. This split is certainly incorrect, but it would help with the understanding behind us. Engineering<code>harness</code>Focus on the stable operation of the system itself, products<code>harness</code>Control core loop or workflow, user-friendly <code>harness</code>, and it's still true.<code>task interface</code>The explanation ranges from making the chat better to leaving the chat to create new interactions.</p>
<h2>Why are you breaking into three kinds?</h2>
<p><strong>In the discussion of specific systems,<code>工程 harness + 产品 harness + 用户友好 harness</code> ♪ The split, the ratio ♪ <code>agent = model + harness</code> More important.</strong></p>
<p>The reason is simple: the latter is responsible for correction, the former for analysis.</p>
<p>If only it stopped <code>agent = model + harness</code>You can easily mix completely different layers of problems. Claude changed the warehouse, users did not trust automatic execution, the interface default was too strong, the approval point was too late and the result was not explained, all of which could be called “harness problem”, but they were clearly not the same problem.</p>
<p>So the new split is as follows:</p>
<table>
<thead>
<tr>
<th>Type</th>
<th>What does it answer?</th>
</tr>
</thead>
<tbody><tr>
<td><code>工程 harness</code></td>
<td>Containment, restoration, certification, governance, competence, segregation, observational</td>
</tr>
<tr>
<td><code>产品 harness</code></td>
<td>Point of trust, point of approval, exposure, how the job was delivered, why the user was willing to commission, and keep the loop itself</td>
</tr>
<tr>
<td><code>用户友好 harness</code></td>
<td>Default interface, timing of right of control recovery, low friction correction, overreliance on natural language</td>
</tr>
</tbody></table>
<p>The differences between these three levels are very specific.</p>
<p>If Claude Code is talking about... <code>MCP</code>、<code>hooks</code>、<code>sandbox</code>、<code>subagents</code>You're talking about the backplay, the authentication. <code>工程 harness</code>。</p>
<p>If you're discussing why users dare to give the task to the system, where the system is exposed and intermediate, when it is allowed to execute automatically, when approval is required,<code>key loop</code>What's the design, how is it to solve the user's needs? Then you've entered. <code>产品 harness</code>。</p>
<p>If you're discussing whether the default interface should be chatting, forms, status panels or more structured, whether users can modify ant-act at low cost, when the system should continue automatically, when control should be returned to the person, then you're talking about it. <code>用户友好 harness</code>。</p>
<p>And that's why I'm not satisfied with Agent = model + Harness. That's a useful phrase, but it's too thick. It helps you to get your attention back to the system level, but it doesn't tell you how to keep it going inside the system level.</p>
<h2><code>task interface</code> Why must it be raised alone?</h2>
<p>The reason is that many of the dedicated anent's competitiveness is not just from stronger Harness, but from which the task interface was designed together.</p>
<p>If a system is entered into a fully open natural language, with an implicit mission state, with irregular intermediate work and unstable success criteria, it naturally relies more on thicker human clarification than ever before.</p>
<p>In turn, if a system is inherently a form, a schema, a fixed piece, a clear success standard, a controlled tool set and a status machine, then much of its reliability is in the end. <code>task interface</code> This floor has been hardened earlier.</p>
<p>So there's the formula of the first:</p>
<p><code>agent = model + harness + task interface</code></p>
<p>Not because it is bigger than it is because of the advantages of many of the things that look like naturals, but because half of the task interface has been structured, and the product that comes out of this structured task is already solved, not natural language + more heavy products.</p>
<p>So not all high-value parties end up being more like chat assistants (probably on the day AGI is realized). But now, many of the high-value specials are going to be more and more like a station that's bound to be hard.</p>
<h2><code>harness</code> Boundary with adjacent terminology</h2>
<p>If <code>harness</code> To be truly true, it must first make its borders clear.</p>
<p>Otherwise it's too easy to start stealing the amount of information. You should have said more, but replaced the details with a more vague overall name. In addition to the splits in the front, we've come to combe the other terms that have been ingenent for so long and that they're with<code>harness</code>Relationship.</p>
<table>
<thead>
<tr>
<th>Terminology</th>
<th>More precise duties</th>
<th>It's not supposed to be a mix.</th>
</tr>
</thead>
<tbody><tr>
<td><code>agent engineering harness</code></td>
<td>Tie models, interfaces, binding, restoration, governance to a working angent system</td>
<td>It's not automatically equal to all product and user layers.</td>
</tr>
<tr>
<td><code>agent evaluation harness</code></td>
<td>The measurement infrastructure for operational tasks, recording the trajectory, scoring and synthesizing results is now being validated in a credible manner to become an important component of the Agent system.<code>evaluation harness</code>It's part of the product.</td>
<td>It's not supposed to be an anent runtime. He's part of the product layer.</td>
</tr>
<tr>
<td><code>runtime</code></td>
<td>Run the system itself</td>
<td>The government should not automatically swallow up the delivery and interface problems.<code>task interface</code>It's important in itself.</td>
</tr>
<tr>
<td><code>environment</code></td>
<td>The world boundary of action, and the return of observations, results and side effects</td>
<td>It's not just about being written together as a tool.</td>
</tr>
<tr>
<td><code>framework</code></td>
<td>The issue was discussed in an abstract, development interface, state maps, assembly of components, previous and previous specialized articles</td>
<td>It's not exactly the same as everything, Harris.</td>
</tr>
<tr>
<td><code>task interface</code></td>
<td>Task entry, working format, status presentation, how users assign and amend tasks</td>
<td>I shouldn't have been invisible to the default premise of Chat.</td>
</tr>
</tbody></table>
<p>The most important use of this watch is not to be scrapped. <code>harness</code>And remind myself:<strong>I should use it only if it really helps me to get the problem to the right level; if it starts to cover the real level, I should change the more specific word back.</strong></p>
<h2><code>harness</code> When there's an explanation, when there's a chance to steal the information.</h2>
<p>I do not want to give up this word completely, because it does have an interpretive effect in certain contexts.</p>
<p>When it is most useful, there are usually three situations.</p>
<p>First, when you want to discuss how models are put into the system.
Just say it. <code>model</code> Not enough. Just say it. <code>framework</code> Not enough. And then... <code>harness</code> Tools, files, environment, validation, governance of these peripheral systems can be described together.</p>
<p>Second, when you want to discuss how the same kernel goes over multiple product surfaces.
This is the one about Openai. <code>Unlocking the Codex harness</code> The most valuable places:<code>harness</code> It is not an empty adjective here, but a stable software boundary. This blog is a part of my right. <code>task interface</code> The introduction, but he focused on a single multiplatform.<code>task interface</code>And I stress it.<code>task interface</code>More dollars.</p>
<p>And third, when you want to move the project from a fragmented view to a holistic perspective.
That's why Claude Code was first mentioned in the last article. Look at this. <code>MCP</code>、<code>hooks</code>、<code>skills</code>、<code>subagents</code>They're like a bunch of loose functions; once they're released, Back <code>harness</code> The way you see it, it's actually a common frame of work.</p>
<p>But it's easy to fail.</p>
<p>Once you were supposed to say, <code>evaluation harness</code>、<code>runtime</code>、<code>environment</code>、<code>task interface</code>When you wait for the concept, you don't want to go on and write it down. <code>harness</code>And that word starts stealing the amount of information. It looks like it's explaining the problem, actually leveling it out.</p>
<h2>General Harness and Special</h2>
<p>This is where I would like to move forward, starting with this chapter, more than anything else, and a few simple brinstores, as the end of this article, which is full of ideas, is a good option.</p>
<p>Many discussions have implicitly put all the anent on the same continuum: the model is getting stronger, the harmness is getting more complete, and then a generic anent will emerge, and it will swallow all the special anents all the way.</p>
<p>I don't think so.</p>
<p>I prefer to use generics and exclusive ones as two different directions of excellence.</p>
<p>General Harness is seeking cross-mission reuse. It usually has broader motion spaces, more open mission expression, a higher proportion of natural language interfaces, more complex knowledge assembly issues, and more frequent situations of mid-way clarification and human takeover.</p>
<p>Specialized harness is seeking high feedback density, narrow action space and strong validation. It tends to rely more on fixed works, structured inputs and outputs, controlled tool sets and more rigid success criteria.</p>
<p><strong>The reliability advantage of a dedicated agent is usually not from its being more human, but from its being less dependent on language as a single interface.</strong></p>
<p>Language is of course still there, but it's no longer the only interface. Forms, schema, works, status machines, approval points, certifiers, controlled tool sets, which together constitute a real task interface for an individual agent.</p>
<p>That means:</p>
<ul>
<li>General angent will not disappear.</li>
<li>The advantages of dedicated anent are usually derived from a more rigid task interface, narrower action space and a stronger authentication device.</li>
<li>Many specialized systems are really competitive, not just whose laws are stronger, but who designed them more closely to the task itself.</li>
</ul>
<h2><code>harness</code> Where will it be? Where won't it be?</h2>
<p>That is another issue of greatest concern to me now.</p>
<p>I don't believe that there's a real meaning to it. <code>perfect harness technology</code>And, like the General Database protocol, it's a common set of all the other products.</p>
<p>A more reasonable judgement should be:<strong><code>harness</code> It will be constricted at the source level and will not condense into a single template at the full system level.</strong> The blog I wrote about the framework and the last blog, which is a blog that is used to share the story of the framework.<code>harness</code>The harvests entered the new framework, while the undefended portion was left to subsequent development.</p>
<p>More likely to be constricted are:</p>
<ul>
<li>Tool interface original</li>
<li>Structured Status and Output</li>
<li>Sandboxes and Permission Border</li>
<li>Document Index and Knowledge Exposure Portal</li>
<li>Track records, playback and evaluation infrastructure</li>
<li>Browner / technical / environment interface</li>
</ul>
<p>Among the things that are difficult to fully absorb are:</p>
<ul>
<li>Criteria for success in different task areas</li>
<li>Human collaboration rhythm</li>
<li>Distribution of user control</li>
<li>Product surface and interactive design</li>
<li>Long-term mission transfer strategy</li>
<li>Special anent interface design</li>
</ul>
<p>That is why I would like to keep two points:</p>
<ul>
<li>On the one hand, the lower-level original language will certainly become more standardized.</li>
<li>On the other hand, complete history is still to be redesigned with the mission, product, risk dimension.</li>
<li>But when AGI and even ASI came, everything here was irrelevant.</li>
</ul>
<p><em>This section is called harness condense or is it interpreted using the concept of Agent = model + harness, rather than the additional conceptual interpretation I have put forward earlier.</em></p>
<h2>Tool/ ACI is the smallest interface unit for Harness</h2>
<p>If there's one of those in the pretenses that's worth taking out, then it's probably... <code>tool</code>I'm sorry. But more precisely, it's not the Tool itself that really deserves to be discussed alone, but the object itself. <strong>The naturals are exposed to the interface player.</strong>。</p>
<p>Many people still understand tool as a function or tool to the model.</p>
<p>That is certainly true, but it is still only half right.</p>
<p>Anthropic is here. <a href="https://www.anthropic.com/engineering/writing-tools-for-agents">Writing effective tools for agents</a> It gives a judgment:<strong>Traditional functions and APIs are mainly contracts between a certainty system and a certainty system; antttools are contracts between a certainty system and an indeterminate model.</strong></p>
<p>This difference will be rewritten directly how we understand it.</p>
<p>In the normal software project, one interface is written well, more about the maintenance of the human developer and another system; but in the angent system, the name, description, parameter shape, error information, return field will go directly into the model's decision loop. They are not purely back-end realization details, but the smallest interface units exposed to the model by the Harness.</p>
<p>And so is it. <a href="https://arxiv.org/abs/2405.15793">SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering</a> The reason why this paper is most worth being put in the discussion of the Harness. It offers a useful concept:<code>ACI</code>That's... <code>Agent-Computer Interface</code>I'm sorry. If Anthony is talking about how tol this action interface should be written, and the SWE-agent is talking about how these actions interfaces are organized into a really usable computer interface for an individual.</p>
<p>The core ACI judgement is not complex: humans need IDE, search, syntax, error tips, debuggers to better operate computers; and LM anent is not the best person to be able to perform in the face of shell, browser, warehouse and API. Agent is a new type of “end user” with its own capacity boundaries and failure patterns, and therefore also requires a computer interface designed specifically for it.</p>
<p>So the hierarchy here can be settled:</p>
<ul>
<li><code>tool</code> It's the minimum action interface, deciding what angent can do.</li>
<li><code>tool description / schema / output</code> It's the smallest semantic interface, deciding how antent understands this move.</li>
<li><code>ACI</code> It is an interactive interface of actions, feedback, guardrails and contact context.</li>
<li><code>harness</code> These interfaces are connected to runtime, privileges, authentication, re-display and product workflows.</li>
</ul>
<p>That means, Tool and ACI are not two isolated things. Tool is the atom of an interfacer, and ACI is the working interface of these atoms after they are organized. A bad tool makes the model misjudge a one-step move; a bad ACI makes the model continue to misjudge the state, misuse the action, lose the context throughout the mission.</p>
<p>From this unified framework, the tool/ACI signature answers at least five questions at the same time.</p>
<p>First,<strong>Whether or not action space is different.</strong>
The tools are not as good as they are, but as different as possible. An agent saw 10 highly overlapping, very close-named, and broadly written tools, adding not to the ability, but to the traffic noise. The tool set design is designed to serve, first and foremost, the symmetry of the action space, not the integrity of the functional list.</p>
<p>This is also the first meaning of ACI: it is not a brain exposure of all the bottom capabilities, but a move that re-cuts complex environments into models that can stabilize choices. SWE-agent is not a group of all shells that you want to use naked, but an extra supply. <code>find_file</code>、<code>search_file</code>、<code>search_dir</code>The action interfaces are closer to the software engineering mission.</p>
<p>Second,<strong>Whether or not to press the action design.</strong>
Many back-end interfaces have grown by database tables, service boundaries, division of work; but angent doesn't care about your microservices, he cares about what he's doing now. A tool for anent should not be just a thin cover for the bottom API, but should be close to the operation that anent naturally needs in the mission.</p>
<p>The SWE-ent file editor is a typical example. It does not allow the model to spell complex shell edit commands, but instead allows the model to replace a section of the file with a line number. This interface is not bottom, but it is close to the "modify the code segment" action.</p>
<p>Thirdly,<strong>Is semantic hint sufficiently clear?</strong>
Tool description/ schema is a prompt in itself. Many people have separated the prompt engagement from the tool calling, but in the real system, the two are often the same thing. The tool description is not a visualized additional document, but a model that understands the entry text of the action world. Whether the parameter name is natural or not, the description of whether the trigger condition, boundary conditions, dangerous action, and typical usage are clearly stated, directly affects the tol selection.</p>
<p>Naming space also reverses the path-by-path behaviour of models. A tool called <code>fetch</code>、<code>query</code>、<code>lookup</code>、<code>search</code>、<code>read_doc</code>The model does not expect the same for them; prefixes, grouping, verb styles are consistent, and influence which actions the model prioritizes in the candidate set. And often, naming is the lightest action a priori. MCP uses document strings to construct a tool description as the most natural idea.</p>
<p>Fourth,<strong>• Whether feedback is high, low noise.</strong>
Tool output or whatever it is you're getting put back in. If a tool is always reposting the original web page, the full log, the entire section of HTML, the whole object tree in its original form, the contamination is not a call, but a subsequent entire chain of reasoning. For angent, the good return value is not the full return value, but the best return value for the next decision.</p>
<p>ACI is the same issue. The fileViewer of SWE-agent displays a file clip with line numbers, window-based, scrollable; once the editing is completed, it displays a new file status; when no output is available, the system also shows the angent command successfully but does not produce an output. These are not extra decorations, but rather how the environment is designed to feed back to the model.</p>
<p>Fifthly,<strong>Whether mistakes can be restored and governed.</strong>
Agent tool is not enough to use when it works. The model will always be wrongly selected, wrong parameters filled, bad files changed, misunderstanding feedback. A good interface must allow errors to be exposed as soon as possible and convert them into understandable, reversible signals.</p>
<p>SWE-atent is in <code>edit</code> The inclusion of lint guardrails in the command is the concreteization of this principle. Bad editors are stopped, grammar errors are fed back, angent is asked to try again. Here, the guardrail is not just a security constraint, it is a mechanism for restoring: it turns a failure of a potential warehouse into an observation that can be modified in the context.</p>
<p>If you look at these five points together, you can explain why Linux shell is strong for human engineers, but not necessarily a good interface for LM anent. Shell's motion space is too wide, command combinations are too free, output is often too long, and false feedback does not necessarily point to the next recovery. Humans can filter these noises through experience, visual context and long-term memory, but models must continue to reason with these elements in limited contexts. So the same computer environment, exposed to angent through different intelfaces, would be completely different.</p>
<p>So, the Tool/ ACI layer actually connects three things at the same time:</p>
<ul>
<li>It decides on the uplink model, because it understands "what I can do."</li>
<li>It connects down the system boundaries because the system determines through it what it can do.</li>
<li>It connects the context cost to the side because description, output, observation history reverses the work memory</li>
</ul>
<p>The blogger says:<strong>Tool is not a small part attached to aharness, it is the smallest action interface for aharness; ACI is not an additional UI package, but organizes action, feedback, history and restoration mechanisms into interactive interfaces in an available environment.</strong> Harness is not a bunch of back-end capabilities, but a set of designed operational languages that are actually exposed to models.</p>
<p>And this judgement does not take place only in the context of reasoning.</p>
<p>Once angent enters training, evaluation and continuous optimization of the closed loop, the tool/ACI design becomes a further part of the target audience. One action is written as a structured, border-specific, verifiable tool, or is left to models to search through natural languages and shell themselves, and it finally decides not only the stability of the reasoning phase, but also how to cut, where to cover, how to cover, how to reach the critical level, and what to measure in Benchmark.</p>
<p>In other words,<strong>This structure of naturals does not simply wrap models in runtime, but it in turn shapes training data, assessment interfaces and capacity to learn boundaries.</strong> You give angent the kind of action language that it's more likely to learn what to do; you make the intermediate results of what you use, and you're easier to write that in reward, judge and replay.</p>
<p>Many of them look like models have learned how to use the tools, and they actually have the other half behind them.<strong>Well, it's a good thing that you can't be a better shape than being taught, judged, and governed by humans.</strong></p>
<h2>A short conceptual extension:<code>model harness</code></h2>
<p>I was just going to leave a short extension here, but now I think it's worth a little bit of expansion.</p>
<p><code>model harness</code> It is not the current generic terminology, but just a working concept that I have proposed to understand future trends. It is meant to mean that many future competitions of the anggens may no longer be just a stronger generic base model + thicker shell, but will increasingly be like a model optimization closer to a given task pattern + a natural optimization closer to a given task pattern. Just like Claude Code. <code>bash</code> Orders for targeted learning.</p>
<p>I left it here just to remind myself not to write into two isolated worlds of modeling and peripheral engineering. It is not the backbone of this article, nor is it the definition that must be accepted today, but it is very helpful to understand the tool and the Harness.</p>
<p>Because once you look at the tool design, you find that different models are not as sensitive to the same set of tools.</p>
<p>Some models prefer verbs to be clear, obvious, short and hard-line; others tolerate tools that describe longer and closer to natural languages; others are more stable in their return to structure, and others are more advantageous in semi-structured script workflows. The current model has been linked to the ability of Agenic to be used in the training.</p>
<p>This means that the switchover of the base model is often not free. You don't just change a stronger or weaker brain, you change it at the same time:</p>
<ul>
<li>Best spell for tool naming and describing</li>
<li>Returns the compression of the structure</li>
<li>How powerful a hint is a bug.</li>
<li>Which actions are suitable for schema-first and which actions are suitable for scripting Line</li>
<li>Human approval points, default autonomy and workflow rhythm</li>
<li>The capabilities and preferences of models in different tools</li>
</ul>
<p>The differences in models directly result in differences in product packaging and working methods, and when we build an Agent, switching the base model may not be free.</p>
<p>When OpenAI handed over the Reponse API to you, Model and Harness were not the two more completely separated concepts. When the development of the universal basic model is completed, customizing the model with the system of Harness will be a new direction.</p>
<p><strong>There's a lot of competition in the future, no. <code>model vs harness</code>And... <code>model-harness co-design</code>。</strong> Who is more likely to make a truly stable anent system if they can move models in the same direction, action interfaces, return structures, validate rings and products in the same direction.</p>
<h2>Should knowledge be factored in, or should knowledge be loaded in,</h2>
<p>Now that Model and Harness are designed together, there is a single axis that is often omitted:<strong>How many of the same "knowledge + capacity" budgets should press the model weight and how many of them should be placed in the habitat and <code>skills</code> Lee.</strong> This is the difference. <code>model harness</code> Closer to the horizon, but it's the one thing I'd like to push forward now.</p>
<p>Let's get this straight. The information that fixed parameters can carry is always limited. I'm here.<a href="/en/blog/2026/03/21/agent-memory-panorama/">From Memory Generation to Memory Governance: A Panorama of Age Memoory</a>The L1 is used in three layers of L1 context, L2 external memory, L3 parameter memory, but the main line is:<strong>Add</strong>- How to write back the L3 weight of the repeat experience. Here's the question of the opposite direction. One.<strong>Subtract</strong>Question: When the parameters are already under strain, the budget is not the same as the budget.<strong>To allow the model to proactively remove a portion of knowledge, retain only capacity (debate, use tools, follow instructions, plan long distances) and remove knowledge from the outside.</strong>It's not a direction worth studying. In other words, knowledge and capacity are supposed to be a hidden combination of weights, and can it be broken down into a budget axis that can be allocated in a visible way.</p>
<p>Pushing one end of the axis, it's a very attractive form: a tiny angst core, plus <code>bash</code>There's a lot of them out there. <code>skills</code>Maybe we can do a lot of good already. <code>skills</code>The ability to be in that small model is left in the file system and the retrieval library. This coincides with the previous judgement: the reliability advantage of a dedicated agent, often from its single interface, which is less dependent on language. The bet of microent is also a sentence:<strong>The combination of “small capabilities + foreign knowledge” may be better than “full knowledge under a giant model”.</strong></p>
<p>But I do not intend to put it in a single statement, because its cost is equally true.</p>
<p>First,<strong>Power has floors, knowledge doesn't exist.</strong>。<code>skills</code> This semi-structured portal can be established on the assumption that the model is strong enough to read the instructions, determine when the trigger is, and run the script. Knowledge can be alienated, not able to borrow -- a small model with too little power, hangs more. <code>skills</code> And it's not moving, and it's only gonna turn in front of a bunch of things that it can't tell.</p>
<p>Second,<strong>The complexity doesn't disappear. It just moves.</strong>I'm sorry. Big Harness's "big" and then microent, it becomes a big one. <code>skill</code> Set+ <code>bash</code> Another heavy organization: delayed retrieval of knowledge, contextual costs as required, and triggers another route noise, all of which will return. And that's what happened here when we said tool earlier.<strong>The tools are not as good as the tools, but the better they are.</strong>；<code>skills</code> To a certain extent, models are not dealing with power, but with noise.</p>
<p>Thirdly,<strong>Self-reliability of external knowledge and the burden of governance</strong>I'm sorry. Once knowledge moves outside the model, it is subject to failure of retrieval, lapse of content, and <code>skills</code> The poison and the supply chain risk -- these were already in the Skills section.</p>
<p>Fourth,<strong>“Landing knowledge, retaining capacity” is more a research assumption than a deliverable route</strong>I'm sorry. Knowledge and capacity are entangled in the weight of power, and it is not easy to strip half of them clean.</p>
<p>They're more like <code>model-harness co-design</code> Two points in this space: budget to model weight, or to harness and <code>skills</code> The balance is biased against the outside, depending on the frequency of the mission's knowledge update, the validation of results and the model's own capability threshold — the high frequency update, robust validation and adequate capability scenario; the reverse is biased towards the weight of the knowledge.</p>
<p><strong>A tiny agent system mix <code>bash</code> And a bunch of hangers. <code>skills</code>It is a programme worth studying, but not necessarily the right one.</strong> I left it here, like the one before: when it was real. <code>perfect harness technology</code> And it never happened, and it didn't matter when the AGI came -- in the long space between them, how knowledge and capacity are distributed between models and naturals is one of the most worthwhile questions to try.</p>
<h2>Concluding remarks</h2>
<p>If I want to wrap this up in the last sentence, it's: Harness is not new, it's a new spotlight on old problems.</p>
<p>It's a useful word because it reminds us not to mistook an angent as a naked model; it's dangerous because it's too easy to fit anything. My use is conservative: yes. <code>agent = model + harness</code> This is a correction formula, but when it goes into specific engineering, it continues to be broken down. <code>工程 harness + 产品 harness + 用户友好 harness</code>And put <code>task interface</code> Pull to the front desk. ♪ Once ♪ <code>harness</code> Start stealing the amount of information, and change more specific words.</p>
<p>The last answer was how the shell works in the works. This one says why people call this whole circle today. <code>harness</code>And how the word should be broken down, how it should be used, how to avoid becoming empty.</p>
<p>For me, two parts are together, pointing to the same judgment: the hard point for Agent products is not just a model, not just a frame, but how the model, the shell and the mission interface are designed together as a working system.</p>
<h2>References</h2>
<ul>
<li>OpenAI, <a href="https://openai.com/index/harness-engineering/">Harness engineering: leveraging Codex in an agent-first world</a></li>
<li>OpenAI, <a href="https://openai.com/index/unlocking-the-codex-harness/">Unlocking the Codex harness: how we built the App Server</a></li>
<li>OpenAI, <a href="https://openai.com/index/the-next-evolution-of-the-agents-sdk/">The next evolution of the Agents SDK</a></li>
<li>Anthropic, <a href="https://www.anthropic.com/research/building-effective-agents/">Building effective agents</a></li>
<li>Anthropic, <a href="https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents">Demystifying evals for AI agents</a></li>
<li>Anthropic, <a href="https://www.anthropic.com/engineering/writing-tools-for-agents">Writing effective tools for agents</a></li>
<li>LangChain, <a href="https://www.langchain.com/blog/the-anatomy-of-an-agent-harness">The Anatomy of an Agent Harness</a></li>
<li>Yang et al., <a href="https://arxiv.org/abs/2405.15793">SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering</a></li>
<li><a href="https://github.com/EleutherAI/lm-evaluation-harness">EleutherAI / lm-evaluation-harness</a></li>
<li><a href="https://github.com/ServiceNow/BrowserGym">ServiceNow / BrowserGym</a></li>
<li><a href="https://github.com/harbor-framework/terminal-bench">harbor-framework / terminal-bench</a></li>
<li><a href="/en/blog/2026/03/20/building-agent-deterministic-constraints/">"Turn LLM back in the cage: From Claude Code, see how Harness is able to fix the probability into a system-bound system."</a></li>
<li><a href="/en/blog/2026/03/03/cognitive-architecture-to-agent-framework/">From the cognitive structure of an intelligent body to the framework of an intelligent body</a></li>
<li><a href="/en/blog/2026/03/10/from-mcp-to-agent-skills/">From MCP to Agent Skills: Why does Agent need a new context work protocol?</a></li>
<li><a href="/en/blog/2026/03/21/agent-memory-panorama/">From Memory Generation to Memory Governance: A Panorama of Age Memoory</a></li>
<li><a href="/en/blog/2026/03/18/model-is-good-enough/">"Model Is Good End: 2026, AI is really scarcely applied, not larger models."</a></li>
<li><a href="/en/blog/2026/04/10/how-to-choose-the-right-model-for-developers/">Claude Code or Codex: How differences in code models translate into differences in product experiences</a></li>
</ul>
