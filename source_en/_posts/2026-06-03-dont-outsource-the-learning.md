---
title: Don't Outsource the Learning
title_zh: Don't Outsource the Learning（Addy Osmani）
date: 2026-06-03 10:00:00 +0800
categories:
- Work & Society
- Career & Learning
tags:
- Learning
- AI Coding
- Repost
author: Hyacehila
excerpt: 'A Chinese translation and commentary of Addy Osmani''s essay: the default AI coding loop is optimized for closing
  tasks, not for keeping you sharp.'
description: 'A Chinese translation and commentary of Addy Osmani''s essay: the default AI coding loop is optimized for closing
  tasks, not for keeping you sharp.'
excerpt_zh: 转载 Addy Osmani 的一篇文章：默认的 Agentic Coding Loop 优化的是“完成编码任务”，而非“增加开发者能力”。将任务完全委托给 AI 总会有失效的一天，而解药藏在你怎么提问里。
permalink: /blog/2026/06/03/dont-outsource-the-learning/
lang: en
translation_key: 2026-06-03-dont-outsource-the-learning
translation_status: machine
translation_source_hash: cc63d2d130d8898b027d77934758fd14622cd8681e0d62fea97523e053c042f8
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<blockquote>
<p>This post is part of our special coverage of the World Cup.<a href="https://x.com/addyosmani/status/2056078124346228860">《Don&#39;t Outsource the Learning》</a>The language is duly modulated and localized without changing its original intent. Original author:<a href="https://x.com/addyosmani">Addy Osmani</a>I'm sorry. The first person in the text, who said "I" was the original author.</p>
</blockquote>
<p>The questions in this article can also be addressed<a href="/en/blog/2026/01/31/the-illustrated-guide-to-a-phd/">The Illustrated Guide to a Ph.D.</a>、<a href="/en/blog/2025/12/26/welcome-to-my-blog/">How did this blog get to be like this?</a>How the concept of a relatively close read together is developed in different contexts.</p>
<h1>Don&#39;t Outsource the Learning</h1>
<p>Now one thing is too easy: let AI finish writing the code and save himself the learning step. Bug is fixed, but your mental model is still unscrewed -- for a long time, even back. We are in fact quietly using our future capabilities for today's speed, and the tools will not stop us. It's up to you to stop.</p>
<p>Most of us have slipped into the same pattern: stick a demand or a misstatement in it, hand over a model for a fix, the symptoms disappear, and you go straight to the hip. At some point in this cycle, the mess between the problem and the solution has been completely eliminated.</p>
<p>I wrote "cognition surrender" -- the moment when AI's judgment unwittingly took your own judgment. Today, it's a single version of it: you and the model. The model is faster than you, so you stop fighting it like you know it better. But in this thousands of little interactions, those that you could have made alone without AI are being weakened a week. And none of these moments, on that day, seems to be a problem.</p>
<p>I'm not against AI. I have used these tools daily, and they have delivered more results over the past year than have been combined in previous years. But we used their default approach to optimize one thing: to shut off the task.</p>
<p>And this is two goals, and "to keep yourself sharp enough to keep them in the grip of their careers."</p>
<h2>These studies are pointing to the same conclusion.</h2>
<p>Several studies over the past year have yielded some remarkable convergence.</p>
<p><strong>Anthropic</strong> A random comparison was done in early 2026: an engineer was given a new Python library, half with AI assistance and half without. The speed of the two groups to complete their tasks is not the same. But in the subsequent understanding test, AI lost -- 50 percent of the opponent's action group was 67 percent, and the more debugging the question, the bigger the gap. More interesting is the AI Group<strong>Internal</strong>The difference: the engineer who asks for the concept with AI scored more than 65%, and the one who directly duplicated the code that was created was less than 40%. It's not the tool itself that can be determined, but you use it in your position.</p>
<p><strong>MIT</strong> The study, " Your Brain on ChatGPT " , divided the author into three groups: LLM, search engine, brain-based. EEG data show that for each layer of external aids, the brain is less connected than a point, with LLM groups being the weakest. After the article was written, 83% of the LLM users could not repeat a sentence from what they had just written. The researchers named this phenomenon — Cognition Debt: Today saves the brain, and tomorrow comes with critical thinking.</p>
<p><strong>CHI 2026</strong> And one of the studies adds a related one: when people use LLM at the beginning of the mission, LLM will set the framework for the whole problem. Even if the work that follows is done by the people themselves, this initial "fence" will make final decision-making significantly worse. In other words, it's working.<strong>Order</strong>It's more important than how much I've spent.</p>
<p>The approaches vary, but the conclusions differ:<strong>Using AI without a proactive learning intent will quietly erode the skills on which you are eating.</strong></p>
<h2>The tool is defaulted for "delivery", not "learning you."</h2>
<p>Opens an encoded agent, everything goes by default, and all designs are aligned to only one indicator: finish the job. Model writing code, you do it, you go back and forth. At any point, the tool won't stop asking you: "What do you think is wrong with it?" Or, "Do you want to write your first five lines first?"</p>
<p>This is the direction of the product and the interactive design. The product team is rewarded with combined changes and shorter cycles, not "to grind you into a sharper engineer." We all wanted to knock a few less keys, so the tool was a little bit smooth. The problem is,<strong>Learning is exactly what's hidden in those frictions.</strong>。</p>
<p>There are also companies that want to put a brake on the cycle. Anthropic has a Learning Mode for Claude: using Socrates-style cross-examination and having you write a code before you go down. OpenAI and Google have similar features. But, to be honest, they are hardly used in real production. We quietly put them in the class of "students for use" -- a miscalculation. The same thing that helps the React of the junior college, and it helps a senior engineer, Rust, is that you're willing to be a freshman again.</p>
<h2>"If AI can do it, why should I understand?"</h2>
<p>That's a question. For some things, the answer is indeed — maybe you don't understand. If it's a prototype code, a glue code, or a one-time CI script that you're never gonna look at again, then hand it over. It's too expensive to put that grammar back into your head.</p>
<p>But instead of software that really needs to be maintained for a long time, it is a very important tool for the people who are living in the country.<strong>The pure commission will collapse in a few specific places.</strong>：</p>
<ul>
<li><strong>Something went wrong.</strong> AI wrote the code, falling up and writing. "This is written by angent" doesn't help you debug, and someone in the team has to really eat through this architecture.</li>
<li><strong>It makes a mistake in one book.</strong> ILM's still gonna be bullshitting. The only line of defence against a reasonable and factual error is that you know enough and you can see through it. The "skills" and "stamps" are only for a while.</li>
<li><strong>The base changes when.</strong> The code is temporary, the system is long. The framework upgrade, or the security review, pulls out a structural problem, and you can't get past it by "prompt again" and you have to really understand the system and move it to the past.</li>
<li><strong>You deviate from the median.</strong> For those who have been solved millions of times on GitHub, AI is more than a blade; but the farther you get from the median, the harder it gets. The difficult, undocumented problems — and it is precisely the problem of being able to afford a senior salary — still depend on a deep understanding.</li>
<li><strong>Market repricing.</strong> The engineers who only deliver and leave AI are entering a market where "how much is the professional capacity worth" is already beginning to be revalued. Taking AI to skip school is like taking the future competitive, taking a little less twirling afternoon.</li>
</ul>
<h2>The antidote is in your question.</h2>
<p>The good news is that tools to create perceptions of indebtedness can also sharpen people. The difference is only what you ask them to do.</p>
<ul>
<li><strong>Let's just say it first, then ask.</strong> Before requesting a fix, write down what you think is the problem, and then take the model answer.<strong>Test</strong>Your judgment, not your judgment.<strong>Replace</strong>Drop it.</li>
<li><strong>First we have to explain, then we need the code.</strong> In the strangest realm, the first sentence might ask: "Tell me how this works, what alternatives are available, and what are the trade-offs. "and then we'll get the code when the concept is over.</li>
<li><strong>When you can't eat, open up your learning mode.</strong> It's gonna be a little slow, but slow is what it means.</li>
<li><strong>Use AI output as a PR by junior colleagues.</strong> Read it, pick out its problems, and be more serious with it. Would you merge it directly because you tested it? If not, it's not here.</li>
<li><strong>Three to five, push things back with your own hands.</strong> Pick a model to write for you, try to build it again from zero. This is a calibration that will show you how much you've lost.</li>
<li><strong>Let the model tell you what it just did.</strong> It writes a beautiful function, and asks what concepts it uses, what it wants to read. And if you ask me, you're taking something from this conversation that's completely different.</li>
</ul>
<p>It's not a big deal, it's just a few little gestures you've been using.</p>
<h2>Two indicators, not one</h2>
<p>I'm used to asking myself when I close the code:<strong>Did I learn something today or just shut down a few issues?</strong></p>
<p>Sometimes the honest answer is "I just closed a few issues" -- that's nothing. But if that's the case for months, then the cognitive debt is accumulated in the back of the door where you can't see it.</p>
<p>The two lines of delivery and learning are different. Your boss and client will always ask the first question, the second one will only be on your own.</p>
<blockquote>
<p>I'd rather deliver only 80% of what I could have delivered, and then complete the learning 100%, and not the other way around. The two alternatives bring in a very different engineer.</p>
</blockquote>
<p>You don't have to choose between "Ai" and "learning" but you do have to choose a combination of jobs -- because default settings don't pick you up. The tools are always ready, just waiting for you.</p>
<p>The next boring thing you're going to throw out is a good start.</p>
