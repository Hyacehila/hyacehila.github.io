---
title: BettaFish, MiroFish, OpenClaw, and Agent Trust Boundaries
title_zh: BettaFish、MiroFish、OpenClaw 与 Agent 的信任边界
date: 2026-04-12 20:00:00 +0800
categories:
- Agent Systems
- Agent Evaluation & Governance
tags:
- Security
- Product Design
- Human Control
author: Hyacehila
excerpt: 'BettaFish/MiroFish and OpenClaw push agents against two trust boundaries: how much we can trust what AI says, and
  how much we let AI do inside our environments.'
description: 'BettaFish/MiroFish and OpenClaw push agents against two trust boundaries: how much we can trust what AI says,
  and how much we let AI do inside our environments.'
excerpt_zh: BettaFish/MiroFish 和 OpenClaw 分别把 Agent 推到两条信任边界上：我们能相信 AI 说到什么程度，以及我们愿意让 AI 在自己的环境里做到什么程度。
permalink: /blog/2026/04/12/agent-trust-boundary-openclaw-bettafish/
lang: en
translation_key: 2026-04-12-agent-trust-boundary-openclaw-bettafish
translation_status: machine
translation_source_hash: 93b23fce13892c7a22bf54016612c54ed8aa43bec5c7e3c6661e7298aea9f932
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<p>The boundaries of language model intelligence are not just the boundaries of capacity, but also the boundaries of commission.</p>
<p>The questions in this article can also be addressed<a href="/en/blog/2026/03/17/behavior-auditing-and-decoding-beginners-guide/">Behaviour Audit and Decoded Behaviour: From Reward to Agent Observation</a>、<a href="/en/blog/2026/03/18/from-black-box-predictors-to-traceable-medical-agents/">From Black Box Forecast to Retroactive Medicine</a>How the concept of a relatively close read together is developed in different contexts.</p>
<p>Ask if the model can plan, call the tools, see the browser, have a long-term memory, and not touch the core of Agent. More crucial is what users are willing to give it.</p>
<p>BettaFish and MiroFish touch the boundaries of understanding: to what extent do we want to believe in AI's description, summation and evolution of the world? OpenClaw touches the operational boundary: the extent to which we are willing to get AI into computers, browsers, communication portals and long-term memories to do real things for us.</p>
<p>The two boundaries end up on the same issue: the value of Agent often comes after users are willing to let it cross a line that should not have been easily crossed. The gains that would flow from the past would also magnify the risks. Okay, Agent's product is not to destroy this line, but to draw it out.</p>
<h2>BettaFish and MiroFish: Believe in AI to what extent</h2>
<blockquote>
<p>The project was just about the same time I was doing something like love work, but not so successful from Stars' point of view.</p>
</blockquote>
<p>BettaFish is called "micro-synthesis" in Chinese and is a multi-Agent intelligence assistant available to all. It does not simply bring LLM to search API, but it breaks down the profile into a set of Agent and Engineering modules: Query Agent understands questions and retrieval directions, Media Agent handles media and material, Insight Agent does analytical abstracting, Report Agent produces reports; ForumEngine listens to different Agent logs, and hosts structured phase discussions.</p>
<p>These complexities are largely invisible to users. Users face a more natural entry point: What do public opinion say about this now, who is talking about, where emotions are, where trends may go. BettaFish has turned its work, which is originally part of the intelligence company, the data analysis team and the professional research process, into a direct access point for ordinary users.</p>
<p>MiroFish is an anatomy of him. BettaFish says "What do you think?" MiroFish says "What happens next." It allows users to upload seed materials, such as data reports, stories, event contexts, and then input into the forecast demand; the system extracts entities and relationships from the material, constructs knowledge maps, generates intelligence bodies with personality, memory and behavioral logic, puts them into a parallel evolutionary simulation environment, and ultimately produces prediction reports by ReportAgent, and users can continue to interact with role models in the world.</p>
<p>MiroFish is closer to the world simulation. It uses Zep to carry entities and relationships in ted materias, and then converts entities to usable smart body images. The reporting period is not a one-off summary, but a tool-based ReACT Review. It provides access to maps, interview simulations, insights.</p>
<p>BettaFish and MiroFish's heat has nothing to do with the technology behind it. The average user doesn't care about the system's Agent, whether there's a Graphragm, what simulation engine. The real diffusion is that they translate complex technologies into two instincts:<strong>Look how and what will happen.</strong></p>
<p>That is also their risk. The more natural the interface, the more complete the report, the more specific the simulation is, the easier the user will believe him, but the 100 per cent of the content is generated by language binding a probability-generation model. Multiple Agent debates, knowledge mapping, group simulations and reporting templates add to the level of interpretation, but do not automatically translate uncertainty into certainty. They at best make assumptions more visible, process more open to discussion and conclusions more open to challenge.</p>
<p>BettaFish/MiroFish is thus a valuable and dangerous direction: They make AI from answering questions to organizational awareness. The better they are, the more they need to remind users that they see a model of the world that has been processed by the system, not the world itself. I think the topic of the border is clear when we see on the Internet that more and more people are taking what AI says as truth and trying to convince others with an AI screenshot. It is now just ordinary people who do this, and it is not certain that when AI becomes stronger, when AI can prove the mathematical assumptions that have been unresolved for decades, then the most advanced human brain may not be able to discern the truth or falsehood of what AI says in the short term, and people will believe or question.</p>
<h2>OpenClaw: Authorizing AI to what extent</h2>
<p>OpenClaw touches another border.</p>
<p>Its predecessor included a bunch of names changed, and finally, OpenClaw. The project aims to be an individual AI assistant running on the user ' s own equipment, supporting a wide range of platforms. This location is not new, but OpenClaw is complete enough.</p>
<p>OpenClaw's core is Gateway. Gateway connects to the message portal, session, tool, node and client. The entrances to whatsApp, Telegram, Slack, Discord, Signal, iMesseage/BrueBubbles, Matrix, Teams, Feishu, LINE, Mattermost, WeChat, QQ, WebChat, etc. can all be places for dialogue with Agent. It's not about pulling the user into a new app, but about putting the agent into even the communication software that the user is already using.</p>
<p>The IM portal is not a matter of the number of channels, but of the location of the relationship. The web page, IDE, terminal and independent App require users to enter a tool at their own initiative; the IIM portal makes Agent more like an implementer in the user ' s daily environment. It's closer to the user, and it's easier to be my assistant, or someone in an IM tool.</p>
<p>OpenClaw's system design can be written on several levels:</p>
<table>
<thead>
<tr>
<th>Level</th>
<th>OpenClaw</th>
<th>Meaning to Agent Developer</th>
</tr>
</thead>
<tbody><tr>
<td>The entrance.</td>
<td>Multiple IM / WebChat / Voice / Companion app</td>
<td>The entrance is where the mission happened, not just the UI.</td>
</tr>
<tr>
<td>Control</td>
<td>Gateway daemon + WebSocket API + Nodes</td>
<td>Agent becomes local control plane</td>
</tr>
<tr>
<td>Session</td>
<td>DM Default sharing, group segregation, new session, session transfer</td>
<td>Context, authority and continuity of the mission</td>
</tr>
<tr>
<td>Memory</td>
<td><code>AGENTS.md</code>、<code>SOUL.md</code>、<code>USER.md</code>、<code>MEMORY.md</code>、<code>memory/YYYY-MM-DD.md</code></td>
<td>Long-term memory is being made into a working-area state.</td>
</tr>
<tr>
<td>Tools</td>
<td>Shell, files, browsers, Canvas, cron, external nodes</td>
<td>Agent capabilities from tools and privileges</td>
</tr>
<tr>
<td>Extension</td>
<td>Skills、plugins、ClawHub</td>
<td>Reusable capacity becomes an installed, shared ecological object</td>
</tr>
<tr>
<td>Clear.</td>
<td>Host-first default, sandbox, managed-opist mode</td>
<td>Place personal trust boundaries</td>
</tr>
</tbody></table>
<p>The last line is the key to understanding OpenClaw. OpenClaw documents clearly state that workspace is the home and private memory of Agent, but not hard Sandbox; the relative path is defaulted on workspace, and the absolute path may still access other locations of the host if the Sandbox is not enabled. Sandbox is optional configuration, mode including <code>off</code>、<code>non-main</code>、<code>all</code>, the backend could be Docker, SSH or OpenShell.</p>
<p>This is not a small footnote, but a product judgement. The real useful personal assistant cannot remain in a harmless but incompetent box forever. It requires some exposure to real computers, real documents, real browsers, real communications portals and real long-term memory in certain settings. The question therefore went from “how to avoid risks altogether” to “how to place them within borders that users can understand, configure and recover”. Perhaps the border is not well controlled, and OpenClaw's user size means a large influx of lay developers who are not familiar with these elements and who are more empowered to acquire the maximum possible capacity.</p>
<p>This is not the default path for many enterprise-level Agent platforms. The business platform first asks for segregation, auditing, approval, minimum permission; OpenClaw first asks how an individual user can keep Agent in his own environment. The former are more secure, while the latter are more likely to create a vibrancy when it is first seen by ordinary users.</p>
<h2>OpenClaw's cause of fire</h2>
<p>OpenClaw is already hundreds of thousands of star sizes on GitHub, and npm has been downloading millions of sizes in the last month. The numbers do not prove a long-term success, but they show that it stepped on the strong mood of early 2026: Users are no longer satisfied with an AI that talks, or with an Agent that runs tasks only in browsers. They started to want an AI that is really around, that really works, that really belongs to themselves.</p>
<p>OpenClaw stepped on at least six things at the same time.</p>
<p>First, the IM entrance puts Agent in the routine. Users do not need to open IDE, backstage or specialized tools to schedule Agent like a message in Telegram, Slack, Twitter, QQQ or WebChat. The closer the entrance to everyday, the more Agent was not like a tool, the more a part of life.</p>
<p>Secondly, long-term memory and personality documents bring emotional distance.<code>AGENTS.md</code>、<code>SOUL.md</code>、<code>USER.md</code>、<code>MEMORY.md</code>, the daily memory file allows Agent to stop being a model in the current conversation and to be a local object with a continuous status. This is context engineering for developers, and it's closer to me for ordinary users. Default Memoory is much more valuable to ordinary users than to professional users who are willing to audit themselves <code>Claude.md</code> The former may not know what it's called. <code>.md</code></p>
<p>Thirdly, high-level competencies bring capacity. Claude Code has given developers the "AI can work in warehouses and terminals" but Claude Code is still a development tool, with little contact with ordinary users. OpenClaw moves similar capabilities to personal help scenarios: shell, file, browser, external nodes, timed tasks, and so on. And for the first time, a common user saw "this AI really can move my environment," not just chatting. In particular, CLI is also a more secure entry point than GUI, with a higher lower limit of capability, in the current scenario.</p>
<p>Fourth, Skills and ClawHub give the power the sharing attributes. Skills package reusable operations into installed, diffuseable, manageable capability modules. Without ecology, Agent has high-authority privileges more like dangerous toys; with ClawHub, personal practice can become a transferable skill asset. As to why Skills is available, there is also a significant relationship to the upgrading of base model capabilities.</p>
<p>Fifth, the local-first and own-your-data reduce the psychological burden. A person who can access a personal environment can be naturally nervous if it is fully operated in a distant cloud. OpenClaw’s local Gateway, workspace, private memory, and sandbox, which brings it closer to my assistant in narratives than a company that operates on my behalf.</p>
<p>Sixth, a large number of ordinary users in the market are being hit by AI, who have a high degree of anxiety about AI, but before that, many products were not developed for them. Claude Code needs CLI and Git knowledge, Manus is simple enough to be limited to the extremes. Users are extremely anxious, and OpenClaw is a cure that tries to use it to make users feel that they are not out of date. The larger models further promote the related anxiety to sell Token, and further magnify his users.</p>
<p>OpenClaw's fire is not a single-point innovation, but a result of the combined advances in access, memory, authority, ecology, ownership and underlying modelling capabilities.</p>
<h2>It's not necessarily good to have a fire.</h2>
<p>But the fire itself is not necessarily good. It shows that OpenClaw captured the real desire of the individual Agent and that it was pushed to a point where it was easily debatable. In the words of Gartner Hype Cycle, it is more like standing near “the peak of expectations: discussion, download, Star and successful intercepts run ahead, and steady delivery, boundaries of access and general user education are behind.</p>
<p>This is often the case for technological breakthroughs. The first ones are not mature products, but demo, ranking, social communication and "you don't have to lag behind." Thermals are useful, they attract developers, expose demand, accelerate ecology; and heat is dangerous because OpenClaw is not a chat-to-talk toy, it touches me, files, browsers, shell, long-term memory and external skills. The faster the transmission, the easier it is to have a group of users who do not understand the boundaries of the permission, to open up all the capabilities, just to get the vibrating sense of "AI can finally do something for me."</p>
<p>So it is necessary to untangle the technical value and heat of OpenClaw. Local Gateway, multiple entrances, work area memory, Skills, ClawHub, optional Sandbox, all deserve to be seen in these directions. But hundreds of thousands of Stars and millions of npm downloads only prove that many people are willing to try, not that they are mature personal assistants. OpenClaw is not the key to the future, but whether it is understandable and revokable, whether the skill ecology is credible and manageable, and whether users can move from “dress to see” to a stable daily commission.</p>
<h2>Operating environment is not the same as a product.</h2>
<p>The products or frameworks of Manus, Bean Packer, GLM Agent / Open-AutoGM are proving that models can perform quite a lot of actions in GUI, browser or mobile phones. But being able to operate the environment is only a capacity and not a product.</p>
<p>Manus is more like a remote task portal. The user gives it the task, which is performed in the cloud or browser environment. This pattern is suitable for demonstrating the ability of the autonomous anent, but it still has a distance from the users ' own daily access, private long-term memory, local documentation systems and tool ecology. It can do things, but not necessarily like my assistant in my environment. And one Agent can do more than one thing, and one that Agent knows how much I know, a coded task can be placed in the clouds, but not in the back of a micro-letter.</p>
<p>As for the bean bag phone, it's more expensive to trust. The address book, album, payment, authentication code, private chat, location, and App access to the phone means that privacy information is completely exposed. Mobile phones UI automation is also more vulnerable, with App anti-automic mechanisms and rights tips. At the same time, the ToC properties of bean-bags are clearly weaker, and users are unable to continue to communicate with fissions without natural means.</p>
<p>Open-AutoGM as an open source &amp; The framework is technical, and even the most popular of these frameworks (and Manus' cakes are, of course, huge, and 2 billion are really expensive). But more like a capacity demonstration and research/development framework. It proves that AI Phone is viable, but does not address access, memory, ecology and personal belonging.</p>
<p>Claude Code is another reference. Its high-level privileges are absorbed by the boundaries known to those developers by warehouses, terminals, Git diff, testing and code review. Developer knows it can change files and how to roll back, test, and look at diff. OpenClaw is trying to make a more ordinary, private, broader version, and it's difficult here: ordinary users don't have the natural boundary of the developers. And that, of course, means that ordinary users don't use Claude Code, not their product.</p>
<p>The competition for Agent products depends not only on who lets the model point buttons first, but on who can place the dot buttons in a system that users are willing to use on a continuous basis, to take risks and to share their capabilities.</p>
<h2>Agent's border and breaking it</h2>
<p>The boundaries of Agent should not be drawn by capacity alone, nor by fear of security alone. More precisely, it should follow a chain of trustees: what the users give, what the system does, what the benefits are, how to stop, roll back and take back when a mistake is made.</p>
<p>The border is not a static wall, but an implicit contract. BettaFish/MiroFish allows users to hand over some of their cognitive sovereignty. Users who would have read their own materials, found information, judged emotions, compared views, imagined the future, now hand over some of them to AI. The returns are structured cognitive, trend interpretation and interactive simulation. The risk is that users may take the description generated by the system as a reality itself and over-rebel what Agent said.</p>
<p>OpenClaw allows users to hand over some of their operational sovereignty. Users who would have to open the App, read the file, run the command, sort the information, and maintain the long-term context, now hand over some of it to Agent. The rewards are direct implementation, cross-entry responses, long-term memory and scalable skills. The risk is that Agent may make a mistake, and that the error will not be limited to the wrong answer, but will be the direct wrong and the file broken.</p>
<p>One is to let AI look at the world for me, and the other is to let AI move the world for me. They're finally asking the same thing:<strong>I am ready to give much judgement and action to a system that is not fully reliable.</strong></p>
<p>This is something that needs to be argued. It is certainly safer to leave Agent within the border: read-not write-only, just recommend-not-implement, summarize-not-forward, do-not-in-sand-in-sand, not-in-the-real-world. This is more in line with the engineer instincts, as errors are restricted to text, reporting or temporary environments. But the Agent within the border is easily turned into a heavy advice machine. It tells you what you can do, leaving you with real friction of implementation; it gives careful conclusions, but it does not necessarily make people trust; it reduces risk and benefits.</p>
<p>I made it myself. <a href="https://github.com/Hyacehila/AnalysisPosts">AnalysisPosts</a> It's close to the reverse. It also aims at profiling, and it also introduces Agent: Stig 1 for data enhancement, Stig 2 for in-depth analysis, and Stig 3 for report generation; among them QuerySearchFlow, DataAgent uses statistical analysis functions to draw conclusions, SeachAgent for supplementary information based on search engines, ForumHost dynamic cycle debates, graphic analysis, insight generation and trace evidence chain. A complete monitoring system has been added to help verify which requests and where errors were made.</p>
<p>These designs are designed to serve reliability. Stage2 will write down the search, data analysis and discussion process in the forum. <code>trace.json</code>;Stage3 requests paragraph references <code>[E#]</code> The evidence corner signs, followed by a hard-check of Review Chapters, were produced; methodological appendices and evidence indices were also inserted in the report. In other words, it has been trying to answer one question: how credible is this conclusion.</p>
<p>But this also illustrates the cost of systems within borders. AnalysisPosts is more like a serious water line for analysts than a product that ordinary users will naturally open. Users need to understand the phase, use reptile tools, select the portal, wait for multiple cycles, view the trail, read the report, trace the evidence and consider the targeted involvement in the process of Agent's thinking. It breaks the uncertainty down very thinly, and it takes a heavy chain of evidence, without the complex beading of "I ask a question, you give me a judgment" like BettaFish/MiroFish, or the ability to put it directly into the user's daily environment like OpenClaw.</p>
<p>It is within borders, and is therefore safe, retroactive and auditable. It also does not give the ordinary user the sense of being able to “return the huge gains immediately with a little trust”. The biggest reminder of this project is that reliability is not a product experience per se. Making conclusions more credible and allowing users to give their assignments to the system are two related but distinct things. Users do not necessarily need this reliability.</p>
<p>A strong Agent product experience, often across this border. BettaFish/MiroFish will never create the imagination of the future sandpad if it is always only summarized; OpenClaw will only be degraded into another chat window if it never touches files, browsers, IM and long-term memories. The potential benefits of breaking borders are not low: it turns recommendations into commission, turns one-off questions and answers into long-term relationships, and pushes what AI can say to AI, what I can do on a sustainable basis.</p>
<p>The danger is coming from here. When the cognitive boundary is broken, the fluid narrative conceals assumptions, evidentiary gaps and simulations; when the operational boundary is broken, model errors, tips, authority misalignment and plug trust enter the real environment. A person who can read your long-term memory, control your login browser, respond to IM, call shell, is no longer just a mischaracteristic system, but an implementer who can change the state of the environment.</p>
<p>Breaking borders is not a justification for reckless excesses of power. The closer the border, the more seriously it will be designed. Whether users know what they have given up, whether the system clearly shows what it can see, change, remember, whether the operation is observed, cut off, roll back, whether the wrong explosion radius is restricted, whether memory and skills are sourced, versioned and detended, and whether the confidence of the border is redrawn when the individual scene enters a multi-person scene, questions whether cross-border is product capacity or dangerous illusion.</p>
<p>Without these answers, cross-borders are creating risks. With these answers, cross-border capacity can become a product. The boundary is not something that is pushed down at once, but it is a redrawn of each authorization and every scene.</p>
<h2>Concluding remarks</h2>
<p>BettaFish, Mirofish and OpenClaw can be read together because they push Agent to the same center, from the "Believe AI narrative" and "Approve AI actions" sides:<strong>Agent is not just a question of intelligence, but a question of commission, which is how much authority AI has gained and how you can trust it, and which has been solved three years ago, but which has only just surfaced.</strong></p>
<p>Model capabilities continue to be important, but before ASI comes, the division of Agent products will increasingly be in border design. BettaFish and Mirofish are not just questions about accuracy rates, but how to make assumptions, evidence and uncertainty understandable; OpenClaw is not just about whether to implement, but how to make privileges, memories and trust clearly managed by users.</p>
<p>OpenClaw's controversy is here. It makes personal AI assistants a system that can be installed, connect to IM, write memories, install skills, run tools. It does not solve the problem of the Agent border, but pushes it onto the table: when Agent enters a real personal environment, product attraction and safety risks come from the same place. The more useful it is, the more dangerous it is; the more restricted it is, the more vulnerable it is to losing its sense of magic.</p>
<p>The key to the next generation of Agent products is not just to make AI smarter, but to let users know what they have given up and to be able to hand over a part of the world to it responsibly.</p>
<h2>References</h2>
<ul>
<li><a href="https://github.com/openclaw/openclaw">OpenClaw GitHub repository</a></li>
<li><a href="https://docs.openclaw.ai/concepts/architecture">OpenClaw Document: Architecure</a></li>
<li><a href="https://docs.openclaw.ai/concepts/agent-workspace">OpenClaw Document:</a></li>
<li><a href="https://docs.openclaw.ai/gateway/sandboxing">OpenClaw Document: Sandboxing</a></li>
<li><a href="https://github.com/openclaw/openclaw/blob/main/SECURITY.md">OpenClaw SECURITY.md</a></li>
<li><a href="https://docs.openclaw.ai/tools/skills">OpenClaw Document: Skills</a></li>
<li><a href="https://clawhub.ai/">ClawHub</a></li>
<li><a href="https://www.npmjs.com/package/openclaw">npm: openclaw</a></li>
<li><a href="https://www.gartner.com/en/research/methodologies/gartner-hype-cycle">Gartner: Hype Cycle Research Methodology</a></li>
<li><a href="https://github.com/666ghj/BettaFish">BettaFish GitHub Repository</a></li>
<li><a href="https://github.com/666ghj/MiroFish">MiroFish GitHub Repository</a></li>
<li><a href="https://github.com/camel-ai/oasis">OASIS: Open Agent Social Interaction Simulations</a></li>
<li><a href="https://github.com/zai-org/Open-AutoGLM">Open-AutoGM GitHub repository</a></li>
<li><a href="https://github.com/Hyacehila/AnalysisPosts">AnalysisPosts GitHub repository</a></li>
<li><a href="https://manus.im/">Manus, official entrance.</a></li>
<li><a href="https://docs.anthropic.com/en/docs/claude-code/overview">Claude Code Document</a></li>
</ul>
