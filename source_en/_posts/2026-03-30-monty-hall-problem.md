---
title: Three Gate Problem
title_zh: Three Gate Problem
date: 2026-03-30 00:00:00 +0800
categories:
- Data Science
- Statistical Thinking
tags:
- Statistical Thinking
author: Hyacehila
mathjax: true
excerpt: The Monty Hall problem is confusing not because 2/3 is hard to compute, but because after the host opens a door people
  instinctively see the situation as 50/50.
description: The Monty Hall problem is confusing not because 2/3 is hard to compute, but because after the host opens a door
  people instinctively see the situation as 50/50.
excerpt_zh: 三门问题最迷惑人的地方，不是算不出 2/3，而是主持人开门以后，人会本能地把局面看成 50/50。真正关键在于，主持人的行动不是随机事件，而是一次带约束的信息披露。
permalink: /blog/2026/03/30/monty-hall-problem/
lang: en
translation_key: 2026-03-30-monty-hall-problem
translation_status: machine
translation_source_hash: 4a90da2d2d6bb4a205475c0a8fd91cb104e491a7a4881d837cffb24e116e25d4
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<h2>First, why do instincts always get stuck?</h2>
<p>The questions in this article can also be addressed<a href="/en/blog/2026/01/10/statistics-and-truth/">Statistics and Truth: How to use the accident (Statistics and Truth)</a>、<a href="/en/blog/2026/01/14/anscombes-quartet/">Anscom Quartet: Visualized power and statistical illusion</a>How the concept of a relatively close read together is developed in different contexts.</p>
<p>Three of the most hesitant points are not counting, but emotions.</p>
<p>You chose one door, the host opened the other, and there were only two doors left. The first reaction of a person is almost always:<strong>Isn't that a choice now?</strong> And then there's an additional psychological discomfort: If the door was right, it would be particularly bad if it was right and wrong.</p>
<p>So many people are stopped not by the probability, but by the feeling that they should not betray their first choice.</p>
<h2>The probability is never reset.</h2>
<p>The point is, before the host opens the door, the probability of your first selection is set:</p>
<p>&#36;&#36;
P(\text{第一次就选中}) = \frac{1}{3}
&#36;&#36;</p>
<p>This isn't just a matter of being a part of the host's life when he opens an empty door. &#36;\frac{1}{2}&#36;I'm sorry. Your door is still that door. Your first guess is still only a chance. &#36;\frac{1}{3}&#36;。</p>
<p>So,</p>
<p>&#36;&#36;
P(\text{坚持原门获胜}) = \frac{1}{3}
&#36;&#36;</p>
<p>And the other two doors were together at the beginning. &#36;\frac{2}{3}&#36; And the winning probability. After the host opens one of the sheep doors, this... &#36;\frac{2}{3}&#36; It didn't disappear, but it was concentrated on the only remaining door:</p>
<p>&#36;&#36;
P(\text{换门获胜}) = \frac{2}{3}
&#36;&#36;</p>
<p>From this perspective, the moderator is not turning the three doors into two, but rather<strong>Put the probability of being spread over two doors and focus on one door.</strong>。</p>
<h2>What did the host really bring?</h2>
<p>The three questions are not about a door being opened, but about a door that's not about to be opened.<strong>Who opened it and what rules did he open it?</strong>。</p>
<p>The conclusions were established on three conditions:</p>
<ul>
<li>The host knows where the prize is.</li>
<li>The host will open a door with sheep.</li>
<li>The host will give you a chance to change doors once they open.</li>
</ul>
<p>It means that the host's actions are not random, but a single one.<strong>Binding disclosure of information</strong>I'm sorry. He cannot open the trophy door, so once you have not chosen at first, the facilitator is able to rule out the wrong option for you and keep the remaining unopened door.</p>
<p>If the host opens the door randomly and even accidentally opens the prize, the conclusion is no longer simple. &#36;\frac{2}{3}&#36;I'm sorry. Many arguments have confused this.</p>
<h2>Zoom in to 100. The door is not too bad.</h2>
<p>Zooming three questions to 100 doors, the instincts will be clear immediately.</p>
<p>You pick a door, you pick the prize for the first time.</p>
<p>&#36;&#36;
\frac{1}{100}
&#36;&#36;</p>
<p>The remaining 99 doors are:</p>
<p>&#36;&#36;
\frac{99}{100}
&#36;&#36;</p>
<p>The probability is hidden in prizes. Then the host knew the answer and opened 98 doors consecutively, leaving only your original door and another open door.</p>
<p>If you stick to the door, the odds are still... &#36;\frac{1}{100}&#36;If you change doors, you win. &#36;\frac{99}{100}&#36;。</p>
<p>The three versions were miscalculated because <code>1/3</code> and <code>2/3</code> The gap does not seem to be as obvious as it is. After zooming into 100 doors, you'll see more easily:<strong>You're comparing your first very unreliable guess with the results of the host's sift once.</strong></p>
<h2>Concluding remarks</h2>
<p>Three questions are a little trick, and it admits one thing:<strong>Information is not delivered by words alone, but by the rules of conduct themselves.</strong></p>
<p>After the host opened the door, the world didn't turn into an average of 50/50. What happened was that part of the original scattered probability was re-focused by a limited movement. And that's why the change of doors is better.</p>
