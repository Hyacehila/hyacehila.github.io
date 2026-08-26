---
title: 'The Fact Layer and the Interface Layer: Markdown and HTML Aren''t Rivals'
title_zh: 事实层与界面层：Markdown 与 HTML 不是替代关系
date: 2026-06-08 20:00:00 +0800
categories:
- Work & Society
- AI Engineering Workflows
tags:
- HTML
- Software Engineering
author: Hyacehila
excerpt: HTML is tempting. It can turn long AI-generated documents into something readable, clickable, and shareable. But
  I do not want HTML to quietly become the source of truth. Markdown should keep the diffable, searchable, auditable record;
  HTML should make complex systems easier to read.
description: HTML is tempting. It can turn long AI-generated documents into something readable, clickable, and shareable.
  But I do not want HTML to quietly become the source of truth. Markdown should keep the diffable, searchable, auditable record;
  HTML should make complex systems easier to read.
excerpt_zh: HTML 很诱人。它能把 AI 生成的长文档变成一个能读、能点、能分享的界面。但我不想让 HTML 慢慢堆成事实源。对我来说，Markdown 负责留下可 diff、可 grep、可追溯的记录；HTML 负责把复杂系统变得更容易读懂。
permalink: /blog/2026/06/08/fact-layer-interface-layer-markdown-html/
lang: en
translation_key: 2026-06-08-fact-layer-interface-layer-markdown-html
translation_status: machine
translation_source_hash: 1aa1b8b88c7142100cd01e72305ab5001f5c39b15e8ad804ded4dfaf7108ef89
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<p>Recently there was a pretty moving saying: AI should not be going to Marktown anymore, but should be going to HTML.</p>
<p>The questions in this article can also be addressed<a href="/en/blog/2026/04/07/spec-is-not-the-new-paradigm/">Spec is not a new paradigm: Video Coding, SDD and AI-era software engineering shift</a>How the concept of a relatively close read together is developed in different contexts.</p>
<p>This is not just about the incriminating. Tharik Shihipar of Claude Code wrote "Using Claude Code: The Unreasible Effecution of HTML" and Simon Willison re-transmitted it, and Karpathy blew it again. The article is a very practical example: code review, research reports, charts, interactive editor, PR notes, all of which can be made into a browser that can be opened. <code>.html</code> Documentation. And a long list of Markdowns is more like a working table that can be used.</p>
<p>It looks very moving. I think a lot of people have been unable to open their phones.<code>.md</code>Problem.</p>
<p>And many times, we don't have a missing content, but rather a missing interface that makes people want to read. AI can spit out a call relationship between 2,000 lines of analysis, 20 risk points, and a dozen files, but when people look at a wide font such as a whole screen, they can easily enter a read-in mode. It would be much better to have a navigational, hierarchical, folding, illustrated HTML. Not because HTML is more advanced, but because people really need some space and pause when they understand complex systems. And HTML is designed for this interaction.</p>
<p>But HTML should replace Markdown?</p>
<p>For me, HTML fits as an interface. It makes me more willing to read and more accessible to complex systems. But it is not suitable to slowly accumulate into a warehouse to become a reality. The source of the facts is examined by diff, and if it is found by grep, then in a few months it will answer a very specific question: what has changed this time?</p>
<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px;margin:28px 0 34px;">
  <div style="border:1px solid #e5e7eb;border-radius:14px;padding:20px 18px;background:#fafafa;">
    <strong style="display:block;font-size:1.08em;margin-bottom:8px;">Markdown . Factual level</strong>
    <p style="margin:0;color:#555;font-size:0.95em;line-height:1.6;">The blog also shows how the government is doing this: Plans, conclusions, mandate status and review are all to be brought back to this point.</p>
  </div>
  <div style="border:1px solid #e5e7eb;border-radius:14px;padding:20px 18px;background:#fafafa;">
    <strong style="display:block;font-size:1.08em;margin-bottom:8px;">HTML & Interface Level</strong>
    <p style="margin:0;color:#555;font-size:0.95em;line-height:1.6;">Let's spread the complex relationship. It can generate, refresh, discard or write back people ' s choices.</p>
  </div>
</div>

<h2>HTML Wins Reading and Understanding</h2>
<p>The hardest to read in complex systems is often not the conclusion, but the relationship: which module depends on, where the state moves, where the change will touch, and which chain of call is the risk hidden behind. Markdown can describe these things, but it starts to work at a certain level of complexity. HTML can turn them into charts, tables, columns, time lines, foldable areas, or even a small tool that can be dragged and filtered.</p>
<p>It's not decoration. Relationships are inherently spatial. You make it into paragraphs, you remodel it in your head; you make it an interface, you can sweep the structure, then you can drill the details. Claude, the examples that are listed in that article, are here: a lot of outputs are not "a much better markdown" but a really operational view of materials that are hard to read.</p>
<p>Same thing happened to the review. A model gives thousands of lines of markdown, who can breathe back on that much? People's attention is just a matter of course. HTML allows for summary summaries to be seen before being carried out by modules; first, risk-based, then evidence-based; first, a list of documents, then a description of the intent of each document to be modified. Reading is no longer a journey down, and people are more likely to actually finish reading.</p>
<p>When you work together on an ad hoc basis, HTML goes along. One of them is sent out with the file, and the other side can see it by clicking. This can include a tick box, filter, copy button, or a section of the text can be organized into a programt. This is less expensive than Markdown for many one-off analyses, meeting materials, PR ancillary reviews.</p>
<p>So I'm not against HTML. Instead, AI tools should be better at generating this interface. When needed, a long report is turned into a workspace in a browser, a call chain is turned into a map, and a bunch of jobs is turned into a filterable list. And it's very useful to use him as a PPT, easy and fast, open the box.</p>
<p>My differences are only the next step: these interfaces cannot automatically become the facts themselves.</p>
<h2>The problem is that the interface is a fact.</h2>
<p>Once the HTML starts taking over the source, the trouble will come soon.</p>
<p>First diff. The sentence in demand has been changed from "must support offline" to "prior support offline", and in Marktown, it is usually a line of difference. Puts it in HTML, which may mix the fine changes in labels, styles, layouts, scripts and generators. The guy who opened diff, saw not intent, but a bunch of noise. The version of history should have answered, "Who changed what when?" and it turned out, "How different the interfaces that were generated and how different was the last time."</p>
<p>Search and long-term maintenance can also be problematic. Markdown has the advantage not of how strong it is, but of being simple. You can use it. <code>rg</code> Find a word, look at history with Git, open it with any editor, and do simple processing with scripts. HTML can certainly search, but when the real information is wrapped in structures and styles, the operationality of the text decreases. I thought it was just a few more today. <code>&lt;div&gt;</code>Six months later, it could be a bunch of historical documents that nobody wants to touch.</p>
<p>Token costs more like chronic disease. The individual HTML file may not be a single label or style, but the fact is not a one-time reading. Spec, plan, review records, task lists are rereaded, archived and re-referenced. Every reading of HTML is paying for the interface. The context window would make the problem less urgent, but it would not make redundancy disappear.</p>
<h2>Markdown is not a simple value, but auditable</h2>
<p>Markdown's value is not that it looks simple, but that it keeps the account clear. The content is in the text, without any additional structure, and all the tier tools are part of the information; change can be checked by people and tools at low cost; you don't need to open a certain operating environment to know what the document is writing.</p>
<p>John Gruber was actually very clear about the location of Marktown: HTML was publishing format, Markdown was writing format. Markdown is not a substitute for HTML, but makes writing, reading and editing the text itself easier. This distinction is still valid in the AI Workstream, except that writing is no longer just a blog, but also a spec, a plan, a mission status, and a review.</p>
<p>These things should not live in a beautiful view.</p>
<p>If an Agent generates PR review dashboard, I hope it helps me understand the changes. No problem. But if I had identified a risk, changed a judgement, reordered a mission, those results would end up going back to Markdown or to another equally auditable fact-system. HTML can be an operating table, but it is to be paid for after the operation. The change cannot be left in the HTML interactive layer only.</p>
<h2>Not a substitute. A layer.</h2>
<p>I understand the attraction of HTML-first. For many people, the content that AI produces can finally be a little more than a hard-to-read report, but rather a small application that can be opened, made, shared. This is indeed progress. The workflow of different people is different, and HTML is perfectly reasonable as the main product if a team requires short delivery, visual review and one-off reporting.</p>
<p>But my daily life (or most system developers) is not. My question is not "can models produce beautiful outputs," but "can these outputs be permanently checked?" I need to know when a judgement has changed, why it has changed, who accepted it and then was not overturned. HTML helps me understand these things, but it shouldn't be kept for me.</p>
<p>I'll draw the border here.</p>
<p>Demand, constraints, plans, review findings, mission status, decision-making records, which require long-term presence, version control, model and cross-check, should remain in Markdown or in an equally auditable structured storage.</p>
<p>Reading views, relationship charts, PR review aids, research materials browsers, temporary dashboards, interactive summaries, which are visible interfaces. Their aim is not to leave a final record, but to make it easier to understand, to be more forthcoming and to give feedback. Make HTML fit.</p>
<p>This rule also explains why the article itself can embed a HTML card in Marktown. The card is a interface level that makes the relationship easier to read; the whole article is still Markdown, leaving text that can diff. The two are not mutually exclusive, but are each in the same place.</p>
<p>And finally, to that:</p>
<blockquote>
<p>Markdown is responsible for the facts, HTML is responsible for understanding. Source file is unchanged, view can be thrown, conclusion flow back.</p>
</blockquote>
<p>Skill doesn't replace MCP, CLI doesn't replace GUI, HTML or Markdown, AI Agent changes a lot, but the flaming is more intense than the real world.</p>
<p>Why is the Anthropic team always likes to make us a lot more token-like, HTML means we need to do every one of these things? <code>&lt;&gt;</code> Paying for the multi-mix and Dynamic Workflow, too.</p>
<p>I'm sorry, p.s. <code>.md </code> Present. <code>.html</code> And there's a much more ambitious concept behind it, and when the brain interface matures, we should just output the flow of the display, and the interaction speeds are full.<code>.html</code> And it's just a temporary middle.</p>
<h2>References</h2>
<ul>
<li><a href="https://claude.com/blog/using-claude-code-the-unreasonable-effectiveness-of-html">Using Claude Code: The Unreasonable Effectiveness of HTML（Thariq Shihipar，claude.com）</a></li>
<li><a href="https://simonwillison.net/2026/May/8/unreasonable-effectiveness-of-html/">Simon Willison: The Unreasible Impact of HTML</a></li>
<li><a href="https://thariqs.github.io/html-effectiveness/">The unreasible example of HTML</a></li>
<li><a href="https://daringfireball.net/projects/markdown/syntax">Daring Fireball：Markdown Syntax Documentation</a></li>
<li><a href="/en/blog/2026/05/27/cli-vs-gui-agent-era/">"We're back to the CLI: Age CLI and GUI"</a></li>
</ul>
