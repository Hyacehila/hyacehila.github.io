---
title: 'From MCP to Agent Skills: Why Agents Need a New Context Engineering Protocol'
title_zh: 从 MCP 到 Agent Skills：为什么 Agent 又需要一种新的上下文工程协议？
date: 2026-03-10 20:00:00 +0800
categories:
- Agent Systems
- Agent Infrastructure
tags:
- MCP
- Agent Skills
- Context Engineering
- Protocols
author: Hyacehila
excerpt: Agent Skills are popular because they package capabilities with low friction. They are a useful layer of context
  engineering, but not a new universal protocol or final agent form.
description: Agent Skills are popular because they package capabilities with low friction. They are a useful layer of context
  engineering, but not a new universal protocol or final agent form.
excerpt_zh: Agent Skills 的流行主要来自低摩擦的能力封装。它是上下文工程中有价值的一层，但不是新的统一协议，也不是 Agent 的终局。
permalink: /blog/2026/03/10/from-mcp-to-agent-skills/
lang: en
translation_key: 2026-03-10-from-mcp-to-agent-skills
translation_status: machine
translation_source_hash: a3e679de0333379d977f733c129ae9c6c5d03573ec483300dafd6e78459b9bc3
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<p>On November 25, 2024, Anthropic launched Model Context Protocol (MCP), which many thought that the Agent world had finally a single interface, and that the next thing was to make up for ecology, customers, distribution.</p>
<p>The questions in this article can also be addressed<a href="/en/blog/2026/02/16/mcp-model-context-protocol/">MCP (Model Context Protocol)</a>、<a href="/en/blog/2026/05/17/agent-resource-collection/">Agent Extra Resource Collection: Skills, MCP Server, Plugins and Practical Tools</a>How the concept of a relatively close read together is developed in different contexts.</p>
<p>But reality is not that linear.</p>
<p>On October 16, 2025, Anthropic pushed Agent Skills to the stage; on December 18, 2025, Skills became open standards; on the same day GitHub Copilot announced support for Agent Skills; on February 2nd, 2026, OpenAI made public reference to the Skills mechanism in introducing Codex app. This timeline itself has shown one thing:<strong>Even when agreements already exist, developers are still looking for lighter, softer, less restrained capacity envelopes.</strong></p>
<p>So this article will give a judgment:<strong>Skills is a valuable link in context engineering. It is not a leap forward relative to MCP, more like a light patch to the capacity containment approach after the model capacity has risen. It'll fire, mainly because it's simple enough.</strong></p>
<p>That sounds like a deliberate denigration of Skills, but I'd like to say:<strong>The value of Skills is real, but it is not primarily about agreement innovation, but about project compression.</strong> It made a catalogue, a copy of a story that it would have done together with Host, Clit, Server, Schema, the license model and the installation link. <code>SKILL.md</code>and several references. It didn't invent a whole new world, but it stepped on Agent's worst point in development:<strong>How the capacity should be handed over to the model.</strong></p>
<h2>The introduction: the fire of Skills, not because it is more advanced than MCP</h2>
<p>One of the most easily made mistakes in the technological world is to write the popular error more advanced.</p>
<p>MCP is more complete than Skills, which is hardly controversial. It was not a Tool Calling agreement from the beginning, but a context agreement:<code>tools</code> 、<code>resources</code>、<code>prompts</code>、<code>Registry</code>、 <code>.mcpb</code>The life cycle, competency consultation, client-based permission boundaries, authentication authorizations, etc. all indicate that it is trying to address the standardization of interfaces between Agent and the outside world. Whether SDK or the document itself, MCP is much more complex than Skills.</p>
<p>MCP design is really a real thing. <strong>Model Context Protocol</strong>I'm sorry. All AI external context (tributions, tools, resources, human interaction) is considered comprehensively in the protocol design. Ideally, an AI model (without additional hints) and an MCP Server can be fully accompanied by AI Agent.</p>
<p>And Skills didn't grow up along this route. It did not attempt to unify its translation, did not attempt to resolve multiple-end consistent competencies, and did not attempt to incorporate all capabilities into a tight set of agreements. It's very simple:<strong>(b) Tie the Prompt, scripts, local information and workstream descriptions into a capacity package to allow the model to be read on demand.</strong></p>
<p>Skills gives people the feeling that, since MCP has not achieved the desired results (or is misread as a tool standard only), we will be asking for a new standard.</p>
<p>That's why I don't think Skills is the next generation of MCP. It's more like a project sidewalk: when the protocol is too complete, too heavy, too heavy a mental burden, the developers naturally go back to the way they cover the file system, scripts and instructions.<strong>This is the capability package that is easier to understand, install and maintain.</strong></p>
<p>It's not a shame. On the contrary, this suggests that the paradox of Agent's development is not whether we can design more elegant agreements, but whether we can deliver our capabilities to models, be maintained by teams, reused by projects, and create real value for the project itself.</p>
<h2>Why is it always looking for lighter capacity seals, Agent?</h2>
<p>From Tool Call to MCP, to Skills, ostensibly changing technical terms, the same issue is actually being discussed:<strong>What form should the capabilities be exposed to the model?</strong></p>
<p>The earliest Tool Call is very clean, giving the model a tool name, a tool description, a parameter Schema, which knows when and how to call. The benefits of this paradigm are clear, stable, easy to audit, and the disadvantages are equally clear: it is particularly appropriate for functions and not for processes. (On why MCP has made some progress in relation to traditional Fund Action, discussed in previous logs, without repetition)</p>
<p>But in Agen, many of the important capabilities in the world are not just functions, but set.</p>
<p>For example, the "Current Analysis" "Reading the Warehouse Output Technology Program" "Creating a blog in my style" "Checking if a project is fit to go online" is not a few parameters that can be made clear. They typically include:</p>
<ul>
<li>How should input be understood</li>
<li>What information needs to be read first?</li>
<li>Steps that must be implemented in sequence</li>
<li>Which scripts are worth running directly?</li>
<li>What format should output match?</li>
</ul>
<p>Such capabilities would appear rigid if they were forcibly rewritten as Tool Schema; if they were purely Prompt, they would be out of control because they were too long, too fragile and too difficult to maintain. Not everyone can maintain a word of a word for a long time. So the middle zone naturally emerged:<strong>It would be better to write capacity into a package that can be readable, expanded and distributed.</strong></p>
<p>Skills was set up because it just grew up in this middle zone. It's softer than Tool Call, lighter than MCP Server, more stable than a large system hint. For most of today's Agent projects, this ecological position has been in existence, but sooner or later someone will wrap it up.</p>
<h2>Skills, what is it? A description, script and a needs-based knowledge package.</h2>
<h3>Making a Skill</h3>
<p>If MCP is considered to be the standard plugin for Agent to connect to the outside world, then Skyll is more like a work bag. It is more like a small catalogue of items that can be carried and distributed at any time, like SOP, scripts, templates, references and rules of use for a given type of mission. The mission opens when it arrives and the mission is irrelevant, so leave the context undisturbed.</p>
<p>From the way it's presented, a Skyll is at least just a belt. <code>SKILL.md</code> ; from the operational point of view, it is a three-storey stack of objects and objects: the content layer, the movement layer, the host layer. The content layer resolves what is in it, the movement layer resolves when it triggers, and the host layer resolves who loads, who executes and who binds the authority. It's like MCP, it's multilayered.</p>
<p>You can see the most typical of them:</p>
<pre><code class="language-text">pdf-skill/
├── SKILL.md
├── references/
│   ├── forms.md
│   └── reference.md
├── scripts/
│   └── fill_form.py
└── assets/
    └── template.pdf
</code></pre>
<p><code>SKILL.md</code> It is the Master Note, but it is also in two parts. The first one, the first one, the first one, the first one, the first one, the first one. <code>name</code> and <code>description</code>;the last thing that's left is the Markdown text for the model. This design is simple:<code>name</code> It's a logo.<code>description</code> It's a trigger condition. According to the Anthropic document and the Agent Skills open code, Agent starts without putting all Skill text into the context, but with every Skill first read <code>name</code> and <code>description</code>, as a skills directory, put them in the system alert. User requests to hit a certain <code>description</code> After that, the model will read the whole thing. <code>SKILL.md</code>I'm sorry. This is how Skills works:<strong>The index is exposed and then the text is loaded as necessary, using progressive disclosure to mitigate the corruption in the context.</strong> More crucially, this progressive disclosure is the default path.</p>
<p>Next up is the second floor: Why is there a lot of room in Skyll? <code>scripts/</code>I'm sorry. Some things are not cost-effective and unstable by modeling. For example, the PDF forms, batch changes, calls fixed CLIs, and generation of standard reports. Every time, the model is used to spell orders, waste token, and easily roll over. Skill's way of getting <code>SKILL.md</code> I'm in charge of the way, Jean. <code>scripts/</code> Responsible for high frequency, machinery, high certainty components. In introducing the Skills architecture, Anthropic also made it clear that the model can read instructions and choose running scripts after Skill has been triggered; the script code itself does not have to be context-specific, and the model needs to get the results of the execution. More precisely, Skills did not recreate the Tool calling protocol, but instead allowed the model to read the description document and then call the host that had already provided it. <code>bash</code>, file system, code execution capability to run scripts, organize commands, parameters and even a small Python code, if necessary, and then decide on the next step based on the output.</p>
<p>Third floor is... <code>references/</code> and <code>assets/</code>。<code>references/</code> The solution is where the big pieces of knowledge are put.<code>assets/</code> The solution is where the templates and materials are put. The beauty of Skill is precisely not that it fed the model more text, but that it allowed text, scripts, templates and static resources to be stored and loaded separately. This is also a gradual disclosure: a directory is given on start-up; a text is given after hit; and a document is read in the body when reference is made to the statement, schema, template. So, Skyll can be big, but the context doesn't have to start to swell.</p>
<p>And here, the Skill's "content layer" is clear:</p>
<ul>
<li><code>SKILL.md</code>: Master Notes, tell the models when they are used, how they are done, what they are taken into account.</li>
<li><code>scripts/</code>: An enforceable certainty operation, responsible for the steady operation of high frequency.</li>
<li><code>references/</code>: as needed, for the reference material, which is responsible for filling in large amounts of knowledge that are not suitable for permanent presence in the context.</li>
<li><code>assets/</code>Templates, sample and static resources to support the final delivery.</li>
</ul>
<p>There's nothing mysterious about this structure, but it's very in tune with the Intuitive Use of Agen. The most important thing in a Skill is not the script itself, but...<strong>It places different levels of capability in different places. Go, go, go!</strong>。</p>
<h3>Enable Skill</h3>
<p>But Skill can't get up with the catalogue structure alone. It's alive with dispatch layers, the layer of system hints and tools that host systems are stuffed to Agent. The official integration guide for Age Skills makes this very straightforward: scan skills catalogues at startup, decompose frontmatters, and then inject Skill-usable metadata into system programt to let models know what they can do. In other words, Skill will not jump out; the host will first give the model a list of skills in the system alert and then the model will decide whether to expand one of them as requested by the user.</p>
<p>The system hints usually address at least a few things: what Skill is available; how to handle a user-specific name for a Skill; whether to automatically trigger a job without a name but when the job clearly matches; and whether to trigger the whole thing. <code>SKILL.md</code> Injecting context or going to the file system first; whether multiple Skills can be combined when relevant; and, if Skill has scripts, which tools are defaulted to allow and which require additional confirmation. And because there's a clear “discovery-trigger-load-execution” link, Skyll is never just a hint folder, but a light running time mechanism.</p>
<p>Prompt just thought about what to say, and Skyll thought:<strong>How to give a whole category of tasks to a model and to be reused every time.</strong> The former addressed a dialogue, while the latter began to be engineering.</p>
<p>Further down, different products are actually placing different shells outside the same set of directions of realization. In Claude Code, Skill can put it in <code>~/.claude/skills/</code> As a personal ability, it can be placed in a project. <code>.claude/skills/</code>And followed the Guit to become a team-wide workflow, and Anthony supported the plugin to bring Skill in with it.</p>
<p>On OpenAI, as of March 11, 2026, HelpCenter has clearly written that the Skylls in ChatGPT are reusable, shared workflows that can be used automatically, one or more Skills can be created, installed, shared in the workspace. You'll find the products to be very different in appearance, but the bottom instinct is the same:<strong>Whatever the shell changes, Skills is doing the same thing -- handing over proprietary processes and organizational knowledge to models.</strong></p>
<h3>A little bit off the record.</h3>
<p>Another interesting detail is that the Skill standard format, although deliberately light, is not entirely without borders. Open norms are right. <code>SKILL.md</code> The frontmatter actually has a lot of constraints:<code>name</code> To match the directory, to limit the length, and usually to name with a short line;<code>description</code> Not just the text, it directly affects whether the model can trigger this Skyll at the right time.</p>
<p>And some of the results will support it. <code>compatibility</code>、<code>license</code>、<code>metadata</code>And even experimental. <code>allowed-tools</code> fields, such as fields, which are used to express operational environmental requirements, authorized information or tool boundaries. That is, the lightness of Skills is not just a little bit of Markdown, but rather “only the most critical structures are standardized, leaving the rest to the host”.</p>
<p>If I were to sum up this section in a less nuanced phrase, I would say:<strong>Skill is not a new organ for Agent, but a brochure, toolbox and an appendix to be searched on demand.</strong> It is not a large design, but it works in a way that is easy: people can write, teams can pass, models can work, context doesn't explode immediately.</p>
<p>And of course, you can see from this, Skills, that there's an implicit premise: the model is strong enough to read the instructions, know when to turn over the appendices, run the script, and when to use the process and not the hard-ass. And that's why Skills looks like a catalogue structure, and behind it is a modeling dividend. We will proceed with this issue in the next section.</p>
<h2>Why now? When the models get stronger, the tools of uncertainty are finally used.</h2>
<p>If we turn back the time earlier, Skills, this kind of thing doesn't necessarily work that way.</p>
<p>Because it was born with an uncertainty: the model was not a rigorous JSON Schema, nor a fully standardized remote service, but a set of instructions, references and scripts. It needs to decide for itself whether it should be used, what to read first, what to read, when to run scripts, when to keep templates and not implement them.</p>
<p>It's actually very high demand.</p>
<p>Schema-based tool calling to press uncertainty onto the interface; Skills accepts a portion of the uncertainty and gives it to the model understanding capability. That's why I think it is. <strong>The conditions for Skills to be established are not just <code>SKILL.md</code> This format, instead, the model is strong enough to digest this semi-structured capability portal.</strong></p>
<p>Skills is not a design innovation that emerges in a vacuum, more like a byproduct of an enhanced model capacity. When models are already good at reading documents, scripts, understanding work streams and describing routers, developers naturally tend to deliver capacity in a lighter and softer way. The marginal gains of strict Schema began to decline and the gains of light-volume seals began to rise.</p>
<p>Or say:<strong>The difference between Skills and Tool Call is who is to assume the certainty of the interface.</strong> <code>schema-based tool calling</code> More certainty is written into the interface layer; Skills transfers a portion of certainty to the model understanding layer. The former are model-friendly, the latter are developers friendly; the former are more suitable for binding actions, and the latter for loose work streams and team knowledge.</p>
<p>This is also the most interesting relationship between Skills and the early Tool Use study. The tools used to be used have been trying to get models learned to find the right tools in the API; today, front-line models are strong enough to gradually shift from training models to tools to lower frictions in the development of the capabilities. Skills just stepped on this migration line.</p>
<blockquote>
<p>Anthropic has been provided in official files. <code>programmatic tool calling</code>I'm sorry. It allows Claude to be in <code>code execution</code> The Python code is written in the container and tools are called as functions; multiple tools in the middle are called, filtered and aggregated in scripts, rather than being re-scoped in model at every step. Officially, the value is also straightforward: multiple-step workflows can reduce delays and the continued use of contextual windows for intermediate results.</p>
</blockquote>
<blockquote>
<p>This detail is also moving towards scriptization on the side of the schema-first: Skylls lets models read <code>SKILL.md</code> , then call the bash/ CLI script, programmatic tool calling, and let the model write its own script in the controlled code execution environment. It is not a standard field for MCP core, but part of the Anthropic Tool Use infrastructure; it responds to the same engineering issue: How to make a model do not have to be called in a rigid JSON function, but still manage its own processes, process intermediate results and continue to do it down.</p>
</blockquote>
<blockquote>
<p>It also stems from the progress in basic modelling capabilities, and MCP will have the ability to use CLI scripts, provided that the main Agent himself (agents to access Skills and MCP) has the authority to order, CLI script is the result of the advancement of model capabilities, not the invention of Agent Skills.</p>
</blockquote>
<h2>MCP unfinished business: the agreement was written there long ago, and experience didn't grow.</h2>
<p>If only the concept was in, Skills wasn't going beyond MCP too much.</p>
<p>MCP, it's not just from the beginning. <code>tools</code>。<code>resources</code>、<code>prompts</code>And later. <code>instructions</code>This indicates that it was intended to systematize the external context required for the model. Even conceptually, MCP is much bigger than many people today understand: It discusses not how models are added to functions, but how external capabilities, external knowledge and cross-border convergence are organized.</p>
<p>The problem is not that the agreement is not thought out enough, but that it is written there and does not automatically grow up to be experienced.</p>
<p>If only capacity coverage is mentioned, MCP does not in fact lack a catalogue before developing such progressive disclosure mechanisms as needed. The law was published in the public version of November 5, 2024, and the law was published in the public version of November 5, 2024.<code>tools/list</code>、<code>prompts/list</code>、<code>resources/list</code> I've already supported it. <code>pagination</code>And they can each do it. <code>listChanged</code> The notification informs the client that the directory has changed. The protocol layer is well placed to expose a part of the tool, Prompt or resource, and then press cursor to continue to take it back, instead of spread the entire capability at once.</p>
<p>In 2025 and 2026, Anthropic continued to push the matter forward on the product layer: the MCP tool search default will be activated when the tool description is more than 10% of the context, and the MCP tool will be marked as <code>defer_loading: true</code>, search first, then spread.</p>
<p>MCP Connéctor and tool search tool also <code>tool_reference</code>、<code>default_config.defer_loading</code> These mechanisms have become ready-made capacities. In other words, MCP does not lack a gradual disclosure mechanism, but rather a simple, easy-to-understand default experience that can be understood by ordinary developers. It is not yet enough.</p>
<p>MCP is too abstract, and the engineering team in Anthropic has considered almost everything that could be considered at the time: protocols, curricula, multilingual SDK, distribution communities, and continuous capacity-building. Integrity means strong and means a higher burden of access. Developer handles server, client, installation, privileges, life cycle, capability exposure, client support matrix, and faces different products Yeah. <code>prompts</code>、<code>resources</code>、<code>instructions</code> The appearance of the difference.<strong>The protocol must be designed so that if the product experience is not low enough, it will not automatically become the first option for developers.</strong> Understanding MCP and developing MCP is a professional task in itself.</p>
<p>That's where Skills hit. It did not attempt to answer all the questions, but only the most immediate ones:<strong>How do I put this power into Agent so it can be used today? How do I allow everyone to give Agent a power that doesn't stay in the hands of a few developers?</strong></p>
<p>The former is addressed by Skills itself: it is simple enough to understand and more dependent on modelling capabilities than on fine engineering. Many Agent can access Skills without having to achieve a Client, with file reading and a few hint adjustments. The latter is more direct: to build a Skill for generating Skills and to put it in front of every user.</p>
<p>So I'm going to understand Skills as a side patch for MCP, not a substitute. It does not win the integrity of the agreement, but the experience. More precisely, Skills took the patient from the developers before MCP had reached the extreme of "light sealing + light distribution + light activation."</p>
<p>That's why MCP went on. <code>Registry</code>Refill <code>.mcpb</code>Refill <code>server instructions</code>, also replace SDK/ API <code>tool search</code> and <code>defer_loading</code>I'm sorry. The world of the deal finally realized that:<strong>Interfaces are not defined in their entirety and distribution, installation, mental burden and default experience are equally part of the agreement.</strong></p>
<blockquote>
<p>MCP <code>server instructions</code>MCP did not include this feature when it was first released, and the development team in Anthropic realized that we needed to complement the whole presentation of Server rather than the tool-only description.</p>
</blockquote>
<blockquote>
<p>New <code>instructions</code> The field is a user manual dedicated to the AI model (User Manual) and it is understandable that Skills has a text description section that describes the functions of this Server so that the previous development would not force global rules into the tool description.</p>
</blockquote>
<blockquote>
<p>Anthropic suggests focusing on information that is not transmitted by Tools and Resources per se, mainly in three categories: cross-functional dependencies, best operating mode (Operational patters), and system restraints and limitations (Constabilitys and limitations)</p>
</blockquote>
<blockquote>
<p>But be careful,<code>server instructions</code> The solution is a service-level user manual, not a gradual disclosure itself. The real responsibility is to give the catalogue first, then to expand it as needed. <code>MCP tool search</code>、<code>tool_reference</code> and <code>defer_loading</code>。</p>
</blockquote>
<p>If you open up Skills, it's a few pieces of core parts that are actually found in the MCP.</p>
<ul>
<li><code>SKILL.md</code> This is a more descriptive stream of work. <code>server instructions + prompts</code></li>
<li><code>references/</code> Closer. <code>prompts</code></li>
<li><code>scripts/</code> Closer, Host, open. <code>bash</code>Normal <code>tools</code> And more. <code>programmatic tool calling</code> The product of integration.</li>
<li><code>assets</code> and <code>resources</code> There's a certain continuity.</li>
<li><code>metadata</code> and progressive reading <code>tool search + defer_loading</code>。</li>
</ul>
<p>They're not itemized, but they're combined and they're already covering the core of Skills' capability. From this perspective, MCP has no shortage of capabilities, but the missing path for ordinary users to use them naturally.</p>
<h2>Why did Skills win the developer?</h2>
<p>If it's purely technical, it shouldn't be; but if it's real, it's very reasonable.</p>
<p><strong>It's better for team knowledge.</strong></p>
<p>The most often settled team is never just API, but rules, templates, styles and processes. How to write a report, how to review the code, how to organize a study, how to align the output to the organizational tone, these things are more like a Skyll/Claude.md than a Tool.</p>
<p><strong>It's more suitable for local work streams on projects.</strong></p>
<p>Many of these capabilities are not worth being treated as independent services. Several scripts in a repo, a checklist, a set of output templates, tied together with project codes, project specifications, project context. Skills packed these things in situ, but more intuitively.</p>
<p><strong>It provides a light-weight capability distribution format</strong></p>
<p>Many teams wanted to distribute not standardized API, but a set of ways to do things: a note, a template, a few scripts, some references. MCP can cover a lot of these capabilities, but Skills makes it easier: one. <code>SKILL.md</code> A few directories would be enough to start sharing. It wins at low cost of packing and distribution.</p>
<p>Pass. <code>SKILL.md</code> This portal, the alerts, instructions and work streams are treated for the first time as diffuse works. For many teams, this distribution is closer to real needs than to strict agreements, and to daily collaboration. Looking back today, Skills won not by monopolistic power, but by putting together a catalogue of capabilities that were scattered across the MCP, host runtime and Tool Use infrastructure.</p>
<p><strong>It's loose enough.</strong></p>
<p>That is the most important point I have. Skills' fire, not just because it's good, but also because it's good.<strong>Less than that.</strong>I'm sorry. It does not require developers to understand a set of strict agreements, does not require that each capability be written into a normative interface, or that all Host behaviour be agreed first. It is loose, so it is easy to be adopted; it is easy to be adopted, so it spreads fast.</p>
<p><strong>It's too simple to use.</strong></p>
<p>Users hardly need to understand the details of the protocol behind them: load it, describe it clearly, let Agent find out for himself, and Skyll can start working. Additional token costs of hierarchical exposure are also usually manageable. If you want to share your abilities, even using Skill, which generates Skills, to help create it, you need only a few hints and rounds of dialogue, rather than a full set of service/client/schema technology bars.</p>
<p>So, as of today, this does not mean that MCP cannot do the same, but they implicitly fail to wrap these capabilities into the same low threshold of understanding.</p>
<p>There is always a paradox in a standardized world: standards are often heavier than standards; and the lighter they are, they are often less standard.<strong>Skills is clearly standing on the side of the front.</strong></p>
<h2>Simple costs: Skills standards are inadequate, security risks and maintenance liabilities</h2>
<p>But simple is never free.</p>
<p>My biggest reservation to Skills is not its worth, but its re-introduction of many constraints that should have been shared by the interface, agreement and host, to model understanding and team discipline.</p>
<p><strong>Models are heavier.</strong></p>
<p>The strength of Tool Call is clarity: name, description, parameters, a model that is not so smart as to be more easily versed in the right direction. Skills is different. Models to be from <code>SKILL.md</code> It understands when to use it, and it decides whether to continue reading. <code>references/</code>, and then decide whether the script is supposed to run. This is certainly more flexible, but it is also clearly more dependent on models.</p>
<p>That is why I keep a little bit of vigilance about the scripting tool. The script CLI looks like it saved Schema, and in fact simply transferred part of the complexity of the interface to the model: the model needs to read itself, identify the script capability, decide parameters and the timing of implementation. It's more flexible, but not more sober than Tool Call.</p>
<p>I'm not sure if I'm going to be able to do this.<strong>Skills saves the complexity of the interface layer; this complexity will shift to the model understanding layer.</strong></p>
<p><strong>The border is even more blurred.</strong></p>
<p>MCP is not absolutely safe, but its risk exposure is clear: you know what tools sserver exposes, what level of permission the client has to confirm, and what calls are part of the agreement. The risk of Skills is easier to hide in ordinary documents and scripts. When a capacity package carries both instituments, references, and scripts, it is already close to the supply chain, and is no longer just a Prompt problem.</p>
<p>That's why project level Skills has to be careful about the project. They appear to be knowledge packages, often with enforcement powers in practice.</p>
<p>In fact, the poison of Skills is not the only thing that exists in the world, especially after the fire of Openclaw, where toxic Skills are filled with the whole community and nobody cares. In the MCP wave of 2024, while MCP was also likely to be poisoned, at least there were companies that had endorsed the MCP Servers for them, and Skills was being manufactured by everyone and was being distributed by Hub, which had little or no audit mechanism.</p>
<p><strong>Portability is weaker than I imagined.</strong></p>
<p>Open standards do not represent a uniform running time. Claude, Copilot, Codex, who says they support Skills, does not mean that they are completely consistent with directory scanning, access control, network access, reliance on installation, script environment and UI appearances. Skill can be spread, doesn't mean Skill can be executed without harm.</p>
<p>In other words, Skills is more like it now.<strong>Removable cover format</strong>, not the standard of a uniform interface in the strict sense.</p>
<p><strong>It's easy to turn into a new document project.</strong></p>
<p>At first you thought you only wrote one. <code>SKILL.md</code>And then you find yourself maintaining a whole set of information structures: not conflicting descriptions, not invalid references, not drifting scripts, not expired templates, and possibly different trigger effects on different models. The more Skill, the more obvious the problem is.</p>
<p>Progressive disclosure is also not in vain. Authors must control the distribution and density of information at all levels: less writing, less modeling, more misperception, more inability to read, more writing, more time load, more time, more token consumption, more consumption. So you're gonna find that, instead of eliminating complexity, Skills is going to switch some of that complexity from interface design to information design.</p>
<p>One of the findings of an empirical study in February 2026 on the disclosure of Claude Skills, which counted 40,285 public skills, was that there was a clear risk of redundancy and extraordinary security in the ecology. This is not surprising to me, because as long as a capacity is contained in a light and relaxed capacity, it will certainly experience a brutal growth before it begins to re-establish governance.</p>
<p>So my judgment on Skills has never been "dangerous, don't use it" but:<strong>It is worth it, but it must be used with engineering vigilance.</strong></p>
<h2>Skills' position in Agent: it's part of the context project, not the end.</h2>
<p>I think the easiest place to discuss Skills is to write it into a big idea that represents the future of Agent.</p>
<p>It's not.</p>
<p>Skills addresses how capacity enters the context: what knowledge, processes, scripts should be given to models, and how to seal and distribute. It is valuable, but only loaded, and not fully contextualized.</p>
<p>And that's why, after Skills emerged, the concepts of memoory, Contex Engineering, Subagent, have not disappeared, but have become more and more important.</p>
<ul>
<li><strong>Memory</strong> It deals with reservations, compression and memory, not with definitional capacity.</li>
<li><strong>Context Engineering</strong> The treatment is context segregation and task split, not encapsulation capacity per se.</li>
<li><strong>MCP</strong> Standardized tools will not lose value, and they will encapsulate capacity to a certain function, and the development of MCP remains very relevant for a standardized scenario.</li>
</ul>
<p>From this perspective, Skills is well located:<strong>It's the first layer of Context Engineering, the first layer of Agent's ability exposure and capacity loading.</strong> Context management includes not only loading, but also compression, isolation, oblivion and restoration, so Skills will not be the end.</p>
<p>The more difficult questions are in the back:</p>
<ul>
<li>When should we forget what we read?</li>
<li>When do you keep the conclusion and not the error of the test process?</li>
<li>When should complex tasks be removed to subagent instead of continuing to contaminate the main context?</li>
</ul>
<p>If the loading phase is limited in the optimal space, forgetability becomes more critical, that is, <strong>Context, or the management of memoory</strong>I'm sorry. The system needs to move non-critical information out of context in a timely manner, leaving the focus space for the current task.<strong>Loading and forgetting are a set of side actions that respond to the same development issue with memory, retrieval: Context.</strong></p>
<p>These questions, Skills, cannot answer. It's the context project, it's the single generic Agent's question to answer.</p>
<h2>Conclusion: This technology is useful, but not divine.</h2>
<p>If it is necessary to define Skills in one sentence, I would say:</p>
<p><strong>It is not a new uniform agreement, but a competency-packing technique: to organize instructions, references, scripts and loads on demand into a default experience.</strong></p>
<p>This assessment does not sound ambitious, even somewhat ordinary, but I think it is closer to reality. Skills is really valuable, and it gives Agent the ability to have private project capabilities, team-based processes and organizational experience. It's hard to exaggerate these things in the past. It's more stable than pure Prompt, softer than pure Tool Call, less than self-constructed MCP Server. It is worth taking seriously.</p>
<p>But I don't want to write it as a technological leap.</p>
<p>It does not redefine the agreed boundaries, does not have an automatic unified competency model, does not have natural long-term memory of solutions, and does not eliminate the complexity of context governance. The most appropriate thing it did was to admit a fact that many engineers had felt about:<strong>In the Age of Age, standards do not necessarily win first, but simple often win first.</strong></p>
<p>So my conclusion is very clear:</p>
<ul>
<li><strong>Skills deserves to be adopted.</strong> Especially when you're going to seal up team knowledge, local processes and light automation, it's almost natural.</li>
<li><strong>Skills is not worth dementing.</strong> It is not a full upgrade of MCP, nor is it the final end of the Agent architecture; in the MCP technology, its most prominent capabilities are today largely replicating.</li>
<li><strong>Skills is still ultimately to be read in the larger context engineering framework.</strong> It's one of them, but it's just one of them.</li>
</ul>
<p>Skills is not the answer to the question, but is forcing us to ask a more practical question:<strong>How we should organize our capabilities into context and construct a truly useful intelligence.</strong></p>
<h2>References</h2>
<ul>
<li>Anthropic, <a href="https://claude.com/blog/equipping-agents-for-the-real-world-with-agent-skills">Equipping agents for the real world with Agent Skills</a></li>
<li>Anthropic, <a href="https://www.anthropic.com/news/agent-skills">Introducing Agent Skills as an open standard</a></li>
<li>Anthropic Docs, <a href="https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview">Agent Skills overview</a></li>
<li>Anthropic Docs, <a href="https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices">Skill authoring best practices</a></li>
<li>Anthropic Claude Code Docs, <a href="https://code.claude.com/docs/en/skills">Extend Claude with skills</a></li>
<li>Anthropic Docs, <a href="https://platform.claude.com/docs/en/agents-and-tools/tool-use/bash-tool">Bash tool</a></li>
<li>Anthropic Docs, <a href="https://platform.claude.com/docs/en/agents-and-tools/tool-use/programmatic-tool-calling">Programmatic tool calling</a></li>
<li>Anthropic Docs, <a href="https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool">Tool search tool</a></li>
<li>Anthropic Docs, <a href="https://platform.claude.com/docs/en/agent-sdk/mcp">Connect to external tools with MCP</a></li>
<li>Anthropic Docs, <a href="https://platform.claude.com/docs/en/agents-and-tools/mcp-connector">MCP connector</a></li>
<li>Anthropic, <a href="https://claude.com/blog/context-management">Managing context on the Claude Developer Platform</a></li>
<li>Model Context Protocol, <a href="https://modelcontextprotocol.io/specification/2024-11-05/server/utilities/pagination">Pagination (2024-11-05)</a></li>
<li>Model Context Protocol, <a href="https://modelcontextprotocol.io/specification/2024-11-05/server/tools">Tools (2024-11-05)</a></li>
<li>Model Context Protocol, <a href="https://modelcontextprotocol.io/specification/2024-11-05/server/prompts">Prompts (2024-11-05)</a></li>
<li>Model Context Protocol, <a href="https://modelcontextprotocol.io/specification/2024-11-05/server/resources">Resources (2024-11-05)</a></li>
<li>Model Context Protocol Blog, <a href="https://blog.modelcontextprotocol.io/posts/2025-09-08-mcp-registry-preview/">Introducing the MCP Registry</a></li>
<li>Model Context Protocol Blog, <a href="https://blog.modelcontextprotocol.io/posts/2025-11-03-using-server-instructions/">Server Instructions: Giving LLMs a user manual for your server</a></li>
<li>Model Context Protocol Blog, <a href="https://blog.modelcontextprotocol.io/posts/2025-11-20-adopting-mcpb/">Adopting the MCP Bundle format (.mcpb) for portable local servers</a></li>
<li>OpenAI Help Center, <a href="https://help.openai.com/en/articles/20001066-skills-in-chatgpt">Skills in ChatGPT</a></li>
<li>Agent Skills, <a href="https://agentskills.io/client-implementation/adding-skills-support">How to add skills support to your agent</a></li>
<li>Agent Skills, <a href="https://agentskills.io/specification">Specification</a></li>
<li>GitHub Changelog, <a href="https://github.blog/changelog/2025-12-18-github-copilot-now-supports-agent-skills/">GitHub Copilot now supports Agent Skills</a></li>
<li>OpenAI, <a href="https://openai.com/index/introducing-the-codex-app/">Introducing the Codex app</a></li>
<li>arXiv, <a href="https://arxiv.org/abs/2602.08004">Agent Skills: A Data-Driven Analysis of Claude Skills for Extending Large Language Model Functionality</a></li>
</ul>
