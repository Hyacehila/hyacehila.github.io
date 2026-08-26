---
title: When Code Becomes Cheap
title_zh: 在代码逐渐廉价之后
date: 2026-07-08 23:30:00 +0800
categories:
- Work & Society
- AI Engineering Workflows
tags:
- AI Coding
- Software Engineering
- Engineering Judgment
author: Hyacehila
mathjax: false
excerpt: Coding agents make code and prototypes cheap, shifting the scarce resources of software development toward judgment,
  verification, coordination, learning, and responsibility.
description: Coding agents make code and prototypes cheap, shifting the scarce resources of software development toward judgment,
  verification, coordination, learning, and responsibility.
excerpt_zh: Coding Agent 让代码和原型迅速变得廉价，但软件工程的成本没有消失。它正在转移到问题定义、验证、协作、学习和责任上。
permalink: /blog/2026/07/08/after-code-is-cheap/
lang: en
translation_key: 2026-07-08-after-code-is-cheap
translation_status: machine
translation_source_hash: 1c3b51325ee271f64f98a0ca0a80aefcc91e439dff8e563a060d731b669732b5
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<blockquote>
<p>“Talk is cheap. Show me the code.” — Linus Torvalds, August 2000</p>
</blockquote>
<p>In the past, the realization of an idea was often screened first at development costs. Even if the PRD had been written, the full vision was expressed, and it would really be a MVP, and it would take weeks, even months, to develop, test and collaborate. Many ideas were not proven to be worthless, but had been stopped on the wish list before it had reached the validation stage.</p>
<p>By 2026, Coding Age had begun to rewriting the screening mechanism. We may not have figured out the problem, but it's written the first edition of the test, added it and ran Demo. From Idea to Code, the link could have been shrunk to a few hours.</p>
<p>Change is not just development fast. When the scarce codes of the past began to become redundant, when realization was no longer the main bottleneck, the question lay elsewhere: what should we produce and on what basis should we believe in it?</p>
<h2>Just Code is Cheap</h2>
<p>Coding Agent can quickly generate and modify codes, run tests, or combo with developers a vague idea. The cost of code generation has been significantly reduced for a large number of matured and relatively clear tasks. It is indeed easier for an thinker to make a runable version than in the past.</p>
<p>Just Code is Cheap. More precisely, what becomes cheap is the process of creating, copying, modifying and mistesting codes, and quickly transforming an idea into Prototype.</p>
<p>Here's Cheap's border. The more vague the need, the more special the system, the more binding the real environment, the harder it is for Agent to provide a reliable answer from the existing Context. New algorithms, complex legacy systems, security-critical software, and issues that are not adequately covered by documents and open source codes, are not suddenly made simple by Prompt.</p>
<p>More importantly, the cheap code does not mean the software is cheap. A functioning Demo, and a system that can be used by a real user for a long time, still separates needs judgement, architecture trade-offs, testing, deployment, observability, safety, maintenance, and a large number of constraints that are not written in the PRD. Coding Agent has reduced the cost of realization, but has not automatically eliminated it.</p>
<p>The speed of code production and code validation has not been synchronized. Agent can modify dozens of documents in a very short time, and Reviewer still needs to understand what these changes have changed, what they have left out, and whether they will become accidents in the future. The time saved by the generator can easily be diverted to the burden of the maintainer and the examiner.</p>
<h2>Code and Prototypes Are Cheap, Engineering Isn&#39;t</h2>
<p>Manu Singh Chauhan is here. <a href="https://medium.com/@dhandedhan/code-is-cheap-engineering-isnt-0dd2756a1874">Code Is Cheap. Engineering Isn&#39;t.</a> The blogger says that the government is not a party to the law. <a href="https://nadh.in/blog/code-is-cheap/">Code is cheap. Show me the talk.</a>I'm sorry. His core point is that Coding has never been the whole, not even the most difficult, of Software Engineering.</p>
<p>Software Engineering is a complete Pipeline. Codes are only one important link from understanding needs, designing programmes, developing and testing, to monitoring, maintenance and iterativeization after they are online. For the development of such a longer production process, codes also need to be coordinated with planning, art, animation, audio, content production and distribution. Accelerating one of these links increases local throughput without necessarily causing a qualitative deterioration of the system as a whole. If demand recognition, asset production, evaluation and integration do not change simultaneously, the time saved will only become a new waiting and backlog downstream.</p>
<p>Engineers usually have to answer many questions before writing down the first line of code: Why does this need exist? Who really needs it? What's the difference between the systems and the architecture that are currently in operation? How much time do we have? What debt is provisionally acceptable? How do you roll back after losing? How do you judge it when it's online? These questions do not necessarily appear directly in the code, but they determine what the code should look like.</p>
<p>These are not just Coding Production, but Judgment Production. Agent can quickly give many technically sound options, but it cannot determine which risk the organization is willing to take, which complexity the team is capable of maintaining, and what results really solve the problem.</p>
<p>Communication in the organization is not just about exchanging information. It also includes consultations on priorities, exposure to conflict, building commitment and bearing consequences. Agent can help to sort information and find gaps, but cannot do these relationship work for the team.</p>
<p>The faster the code is generated, the more often these judgments are made. Programmes that were constrained by development costs and did not have the opportunity to be tried can now quickly become a seemingly viable version. Agent didn't reduce the choices we needed, it just greatly increased what we could choose.</p>
<p>And that's why Code is Cheap will soon become Prototypes are Cheap. In the past, an idea would take weeks of development; now, one person might complete the prototype in one afternoon and display an interactively complete Demo the next day. Exploration is thus more liberal, and outputs in the wrong direction are equally cheaper.</p>
<p>A pretty Prototype can easily create a illusion that progress has been made. The interface can click, the Agent can answer, the core process can run, but only proves that this technology path can be achieved, but it does not prove that users really need it, that it is better than the existing scheme, or that it can operate in a real environment. Buildable, Desirable and Reliable are three different things.</p>
<p>The cheaper the prototype, the more it is to be clear before it does, what it is prepared to verify: what observational support continues to invest, what results indicate we should stop, whether users are willing to move, pay or change existing processes, and whether models can achieve the lowest acceptable quality in target data and boundary scenarios. Otherwise, we're just packing an uncertified guess into a demonstrationable product at a faster rate.</p>
<p>I was before.<a href="/en/blog/2026/05/11/from-engineer-to-builder-opc-product-thinking/">From Engineering to Builder</a>Similar issues were discussed: AI expanded the scope for engineers to be able to do it independently, and required engineers to reach out to users, needs and distribution earlier. When costs are reduced, we can no longer prove a direction worth continuing with "a lot of development time has been invested." Making it cheaper and choosing what is worth realizing is becoming more expensive.</p>
<h2>The scarce things are moving.</h2>
<p>Code is Cheap doesn't make everything cheap. It changes the distribution of scarcity in software development.</p>
<table>
<thead>
<tr>
<th>Relatively scarce capacity in the past</th>
<th>Now, more scarce capabilities.</th>
</tr>
</thead>
<tbody><tr>
<td>Translation of needs into code</td>
<td>To judge whether demand is worth achieving</td>
</tr>
<tr>
<td>Knowledge of languages, frameworks and grammar</td>
<td>Understanding business, systems and historical constraints</td>
</tr>
<tr>
<td>Make a prototype that can run</td>
<td>Design to verify hypocritical assumptions</td>
</tr>
<tr>
<td>Production of more codes</td>
<td>Read, filter and reject codes</td>
</tr>
<tr>
<td>Showcasing the workload achieved</td>
<td>Demonstrate the basis for decision-making and validate evidence</td>
</tr>
<tr>
<td>Completion of one functional delivery</td>
<td>Long-term responsibility for operation and evolution</td>
</tr>
</tbody></table>
<p>This does not mean that Coding lost its value, but rather that it is declining as evidence of scarce capacity and workload. Engineers' values are beginning to appear more at both ends of the code.</p>
<p>Before codes are introduced, engineers need to work with products, design, operation and other developers to turn vague needs into negotiable targets, constraints and trade-offs. If input is just a vague wish, Agent will turn it into a large number of specific codes very efficiently, without automatically judging whether the original question is correct or not.</p>
<p>After the code, the team is faced with integration, access, observation, maintenance and responsibility. Agent's output is usually directed at the current Prompt, current warehouse and current acceptance conditions; engineers also consider the next migration, maintenance after six months, on-call late at night, and who can take over when the system fails.</p>
<p>If the organization understood engineers as code producers only and weakened engineering capacity because codes became cheaper, the costs saved were likely to be re-emerged in integration, accidents and long-term maintenance.</p>
<h2>Validation and learning don't automatically become cheap.</h2>
<h3>Validation becomes a new bottleneck</h3>
<p>The most interesting question for AI Coding is not whether it will generate the wrong code, but rather whether it will create the wrong code, and humans will also make mistakes, but the speed of generation and authentication is not being balanced.</p>
<p>One of the Agents can try multiple scenarios at the same time, modify dozens of files and complete the tests. In theory, this has expanded the search space for developers; in reality, it has also produced more candidate results for reading, comparison and rejection. If the team still follows the Review process, which is premised on the speed of artificial production, it will end up getting more and more PPR, more and more superficial scrutiny, and more late exposure.</p>
<p>Thus, what is required after Code is Cheap is not production per se, but uncertified production. The team needs to control the scope of single changes to enable Agent to submit smaller, more easily perjury changes; to prioritize high-risk state changes, data boundaries, security conditions and irreversible operations, rather than be confused by a full code style and complete commentary.</p>
<p>A complete delivery cannot be a single code. Problems and constraints, the rationale for choosing the current option, the results of tests and validations, known failure patterns, monitoring signals, rollback and ultimately Owner are all part of the delivery. When codes can be generated in large quantities, what really is worth more than one achievement, is why we believe that this can be achieved in the real world.</p>
<h3>Don't outsource the learning process.</h3>
<p>Code is Cheap, a power amplifier for experienced engineers. We can skip familiar work on models and put more time on structures, experiments and ideas that were not energyful in the past. But it can also be a shortcut to the process of capacity formation for those who have not yet developed a systematic understanding.</p>
<p>A beginner can keep asking Agent to change the code until the test is passed, but he still doesn't know why the problem is happening and why the repairs are working. In the short term, the task was accomplished; in the long term, he did not develop a mental model that would take over the problem when Agent made a mistake.</p>
<p>I'm here. <a href="/en/blog/2026/06/03/dont-outsource-the-learning/">Don&#39;t Outsource the Learning</a> It is called a cognitive liability: we trade the judgement of the future for today ' s delivery speed. After Code is Cheap, learning doesn't happen naturally. The default goal of the tool is to close the task, not to develop a person who can independently judge.</p>
<p>This requires us to consciously put some friction back into the work stream. Write down its own assumptions before requests are made; ask for an explanation of the programme and a trade-off before accepting codes; and before merging, at least one of the teams will have to be able to explain how the critical state changes, where the path to failure is, and why the current option is chosen instead of another.</p>
<p>Vibe Coding certainly fits into exploring, personal tools and low-risk prototypes. But it remains important to understand what we are doing when software requires long-term maintenance, processing of data from real users or when it fails to do so. Otherwise, we may have more and more codes, but we may lose those who can judge them.</p>
<h2>After Code is Cheap, what should we do?</h2>
<p>Before the code, define intent and boundaries. Don't let Agent's first realization determine the scope of the problem for us. The user, the target, the constraint, the act that cannot be compromised, and what results would justify the change. In the prototype, the signal of continuation and cessation is also written in advance.</p>
<p>In the code, limiting the size of generation challenges its assumptions. Consider the code created by AI as an untrustworthy PR, not an answer. Read it, run it, look for a reverse example and ask it to explain why it chose the current option. Smaller tasks, narrower subject and shorter feedback cycles are usually more reliable than generating 10,000 lines at a time.</p>
<p>After the code, the evidence is delivered and responsibility is retained. Tests are only part of the evidence, especially when they are realized and tested from the same Agent, and check whether they share the same set of false assumptions. Changes involving funds, privileges, user data and irreversible status require clear monitoring, rollback and Owner. Agent can do his job, but he can't be the one responsible for the accident.</p>
<h2>Show Me the Evidence</h2>
<p>"Talk is cheap. Show me the code." And then, turning ideas into codes is a powerful proof of that.</p>
<p>Now, a functioning version can be produced in a very short time, and 10 different kinds of realization can occur simultaneously. Codes are no longer sufficient to demonstrate understanding, input and quality. And we have to continue to ask: What is it that solves? How do we know it's right? Under what conditions would it fail? Who understands it and who bears the burden?</p>
<p>So today, it can be changed to:</p>
<blockquote>
<p>Code is cheap. Show me the evidence.</p>
</blockquote>
<p>Evidence still needs Judgment, and a owner willing to bear the consequences. Software works will not end because Code is Cheap. Problems definitions, trade-offs, validation, communication, maintenance, teaching and accountability, which were often overshadowed by Coding ' s visible outputs, are now back in the forefront of the work.</p>
<p>After Code is Cheap, what we have to do is to determine which codes are worth living and to take responsibility for what happens when they enter the real world.</p>
<h2>References</h2>
<ul>
<li>Kailash Nadh, <a href="https://nadh.in/blog/code-is-cheap/">Code is cheap. Show me the talk.</a></li>
<li>Manu Singh Chauhan, <a href="https://medium.com/@dhandedhan/code-is-cheap-engineering-isnt-0dd2756a1874">Code Is Cheap. Engineering Isn&#39;t.</a></li>
</ul>
