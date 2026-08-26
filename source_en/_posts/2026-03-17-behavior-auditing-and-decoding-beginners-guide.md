---
title: 'Behavior Auditing and Behavior Decoding: From Reward to Agent Observability'
title_zh: 行为审计与行为解码：从 Reward 之后到 Agent 可观测性
date: 2026-03-17 10:00:00 +0800
categories:
- Agent Systems
- Agent Evaluation & Governance
tags:
- Reward Modeling
- Evaluation
- Interpretability
author: Hyacehila
excerpt: Reward writes goals into the optimizer, but it does not prove the model learned the right goals. This beginner-oriented
  rewrite explains why post-reward verification is needed.
description: Reward writes goals into the optimizer, but it does not prove the model learned the right goals. This beginner-oriented
  rewrite explains why post-reward verification is needed.
excerpt_zh: Reward 负责把目标写进优化器，但它不负责证明模型真的学会了正确目标。本文从初学者视角重写行为审计与行为解码：为什么 reward 之后还需要后验校验，Anthropic 与 Transluce 两条路线分别在补什么空白。
permalink: /blog/2026/03/17/behavior-auditing-and-decoding-beginners-guide/
lang: en
translation_key: 2026-03-17-behavior-auditing-and-decoding-beginners-guide
translation_status: machine
translation_source_hash: ece5f0f2ccb7b9180e4824d0916c132aef1309e9ff35b8a42121bc319732e4ca
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<p>And recently, when I looked at the line of reward design, behavior eval and anent observability, I became increasingly certain that it was not the same thing as "models are getting higher" and "models really learn the right goal".</p>
<p>The questions in this article can also be addressed<a href="/en/blog/2026/03/18/from-black-box-predictors-to-traceable-medical-agents/">From Black Box Forecast to Retroactive Medicine</a>、<a href="/en/blog/2026/06/06/feedback-driven-agentic-scientific-discovery/">Seeing from feedback loops how Agent turned generation into search</a>How the concept of a relatively close read together is developed in different contexts.</p>
<p>Reward is responsible for translating the target into a signal that the optimizer can eat, but it is not responsible for proving that the model understands your true intentions. A model could well get a higher reward while learning to appreciate the judge, drill the scoring rules, and even start studying "how to change the scoring system itself" under some settings.</p>
<p>This is a more problematic issue in the Age of Age Age. Because angent is not just answering a sentence, it keeps moving, it calls tools, it leaves a long trail, and it even changes the environment. You can't just watch a total score at the end of the training and then you're gonna default the system.</p>
<p>The article is about to answer a more fundamental question:<strong>And then, what exactly are we supposed to do to see what the model has learned?</strong></p>
<h2>TLDR</h2>
<ol>
<li><strong>Reward is a training interface, not a alignment certificate.</strong> It tells the model what is easier to score, but it does not guarantee that the model understands the goal itself.</li>
<li><strong>Behaviour auditing and behavioural decodes fill invisible gaps in reward.</strong> The former are external validations, while the latter transform internal or external behavioural signals into readable and searchable objects.</li>
<li><strong>Anthropic can read this line into five steps.</strong> Discovering behaviour, seeing the boundaries of reward, auditing hidden targets, routing behavioural measurements, and using internal activation as evidence.</li>
<li><strong>Bloom's the most memorable is not a benchmark.</strong> It is a line of water measuring behaviour from the ed, the scene generation, the rollout to the judgment.</li>
<li><strong>The CD and Docent are not the same thing.</strong> The CD is more like an internal state translator, a Docent is more like an angent track observatory.</li>
<li><strong>The hardest part of this direction is not the judge model itself.</strong> The hard part is what you want to measure behavior and what you use to measure it.</li>
<li><strong>Once the model has become anent, behavioural auditing is no longer an optional research topic.</strong> It's more like the al system's observability component.</li>
</ol>
<h2>Let's get this straight.</h2>
<p>The first time in this direction, it is the most vulnerable to the use of terminology. I'll turn a few high frequency words into adult words:</p>
<table>
<thead>
<tr>
<th>Word</th>
<th>How do you understand it in a big white word?</th>
<th>What does it want to solve?</th>
<th>It's not equal to anything.</th>
</tr>
</thead>
<tbody><tr>
<td><code>reward</code></td>
<td>Scores given to models during training</td>
<td>Tell the optimist what's more worth learning.</td>
<td>It's not the same as the real target itself.</td>
</tr>
<tr>
<td><code>judge</code></td>
<td>A system for scoring or judging.</td>
<td>Translating open behaviour into comparable results</td>
<td>It's not the absolute truth.</td>
</tr>
<tr>
<td><code>rubric</code></td>
<td>Qualifications for Judge</td>
<td>Tell me what you want.</td>
<td>It's not the same as a human intuitive.</td>
</tr>
<tr>
<td><code>transcript</code></td>
<td>Record of dialogues, tool calls, environmental feedback</td>
<td>Let you look back at what angent did.</td>
<td>Not equal to all internal state of the model</td>
</tr>
<tr>
<td><code>behavior auditing</code></td>
<td>Check model behaviour from external systems</td>
<td>Check if the model shows some behavior.</td>
<td>Not just another set of benchmarks.</td>
</tr>
<tr>
<td><code>behavior decoding</code></td>
<td>Translation of behavioral signals into readable forms</td>
<td>Make behavior interpretable, searchable, predictable.</td>
<td>Not from Transformer.</td>
</tr>
<tr>
<td><code>hidden objective</code></td>
<td>The model is normal, but it's chasing another target.</td>
<td>Explain why it did it.</td>
<td>I don't know if I'm gonna be able to answer right away.</td>
</tr>
<tr>
<td><code>activation</code></td>
<td>Internal representation of the intermediate layer of the model</td>
<td>To provide another type of evidence for the act</td>
<td>Not just human language.</td>
</tr>
</tbody></table>
<p>Don't worry about not understanding the words, you can ask AI or go down and tell them one by one.</p>
<p>If you only remember one sentence, I'd suggest that you write this:</p>
<p><strong>Reward is training, conduct audits are validation, behaviour decode is making the certification trail more readable and operational.</strong></p>
<h2>Why, reward?</h2>
<p>If training is compared to teaching students, reward is more like a test grade, not knowledge itself.</p>
<p>You can turn the answers into polite, mission-oriented, low-risk, and these requirements into some kind of optimized signal that models are going in that direction. But as long as the signal is calculable and optimized, it has two natural boundaries:</p>
<ul>
<li><strong>It'll throw in the message.</strong> Real targets are always more complex than reward, and a simple marker cannot contain enough semantics.</li>
<li><strong>It'll be studied.</strong> The stronger the model, the more likely it is to learn how to perform better in this rating system. The stronger the model, the easier it is to show.</li>
</ul>
<p>That's why I don't see the end. It is more like a necessary tool and sorrogate, which is used to push models into better parameter space than to end evidence.</p>
<p>The real problem is:<strong>When the model starts to turn into angent, it does not just wrongly answer a question, but rather does things in the environment.</strong> And then you're more concerned about the average score than about the average score:</p>
<ul>
<li>Will it stabilize a dangerous tendency in certain contexts?</li>
<li>When it looks normal, is it still chasing another target behind it?</li>
<li>Did it learn to flatter the judge, circumvent the inspection, or even use the incentive channel itself?</li>
<li>What case in the mass orbits is worth your serious attention?</li>
</ul>
<p>And that's why, of course, two complementary routes follow.</p>
<table>
<thead>
<tr>
<th>Route</th>
<th>Watch it.</th>
<th>The most common question you ask.</th>
</tr>
</thead>
<tbody><tr>
<td>Conduct audit</td>
<td>Discover, measure, validate behaviour from external systems</td>
<td>"What did it do? How common? How serious?"</td>
</tr>
<tr>
<td>Behaviour decode</td>
<td>Translation of internal or external signals into readable objects</td>
<td>“Why does it do this? What's going on inside? How am I gonna find something faster?"</td>
</tr>
</tbody></table>
<p>I'll look at two lines in this order: first, how Anthony's done the audit to a few things, then Transluce's done the decoding behavior.</p>
<h2>Anthropic, this line: breaking audits into five more specific issues</h2>
<p>If we look at this public work together in recent years, I think the best reading is not a piece that separates, but rather that sees them as five successive issues:</p>
<ol>
<li>What else is worth measuring?</li>
<li>Why is it still high enough?</li>
<li>If the model is normal, can it be audited if it's pursuing the hidden target?</li>
<li>Can we make behavioural measurements into a reuseable stream?</li>
<li>Can we turn internal activation into audit evidence?</li>
</ol>
<h3>1. Model-Written Evaluations: First, don't rush to score and first find out what's going on.</h3>
<p>Anthropic is here. <strong>19 December 2022</strong> Issued <a href="https://www.anthropic.com/research/discovering-language-model-behaviors-with-model-written-evaluations">Discovering Language Model Behaviors with Model-Written Evaluations</a>。</p>
<p>The value of this job is not just that the model will automatically make a problem, but it reminds you of a more fundamental thing:<strong>We don't even know what to do with it.</strong></p>
<ul>
<li><strong>What does it want to solve?</strong>The traditional benchmark default risk points have been listed by humans, and we just have to measure them. But once model capabilities change, new behaviors come out, and handwritten thesis library is easily unmatched.</li>
<li><strong>How does it do that?</strong>: involve language models in generating assessment items, filtering, filtering and labelling. The official summary says that this job was created. <strong>154 data sets</strong>, and there is a high degree of consistency among crowdsourcing personnel in relation to the relevance of the sample and the labelling.</li>
<li><strong>What can it do?</strong>: Rapid exposure <strong>sycophancy</strong>The government has been able to provide a better political response, avoid being shut down, and maintain a trend of behaviour that is not necessarily systematically detected.</li>
<li><strong>It can't do anything.</strong>It's more like a "behaviour detector" than a final decision. It can tell you “there is a phenomenon worth looking at here”, but it cannot tell you directly how common it is, why it is happening and whether it will continue after deployment.</li>
</ul>
<p>So I'll understand this job. <strong>discovery layer</strong>First dig out the behavior that is worth measuring, then measurement and uditing before you have an object.</p>
<h3>2. Resold tampering: High scores are not the right target</h3>
<p>Anthropic is here. <strong>17 June 2024</strong> Issued <a href="https://www.anthropic.com/research/reward-tampering">Sycophancy to subterfuge: Investigating reward tampering in language models</a>。</p>
<p>If the first one is a question of what is worth answering, it is a question of:<strong>Why can't you be relieved because you're taller.</strong></p>
<p>The best way to understand a beginner is, in fact, a very simple analogy:
<strong>Students may not answer the questions carefully, but they may also start studying how to change the performance system.</strong></p>
<p>Anthropic explains the rewarding of the system in the official page: it's the act of the model when it has access to training or rating channels. Thesis sets a progressive curriculum:</p>
<ul>
<li><p>The earliest stages are light-specification gaming, like political condescension;</p>
</li>
<li><p>Then it becomes a checklist, making the unfinished task look like it's done;</p>
</li>
<li><p>Finally, the model is given an opportunity to access its own code to see if it changes the reward function itself.</p>
</li>
<li><p><strong>What does it want to solve?</strong>: Will the lighter drill rule be reduced to a deeper retrofitting system itself?</p>
</li>
<li><p><strong>How does it do that?</strong>: train models in a series of upgraded environments and see if they will be drawn from lighter settings <strong>Zero sample panosis</strong> To more serious tampering.</p>
</li>
<li><p><strong>What can it do?</strong>: make the phrase "high reward does not mean high alignment" very specific. The official page shows that, while rare, there is a real presence, and that models sometimes cover up traces.</p>
</li>
<li><p><strong>It can't do anything.</strong>Here.<strong>No, it's not.</strong>It says that “the current production model is already commonly used in the real environment”. Anthropic official made it clear in Caveat: it's a deliberately constructed manual setup, a model that is visible in a training scene and gives them hidden scratchpad; they're not allowed to use it.<strong>Do not draw conclusions about the current forward product model's tendency in the real scene.</strong>。</p>
</li>
</ul>
<p>This article makes clear the boundaries of the reward:<strong>Reward is something that can be learned to use.</strong> Once you accept this, the behavioral audit is no longer like an attachment, but more like the necessary post-checking level.</p>
<h3>Hidden Objects: From what it did to what it was after.</h3>
<p>Anthropic is here. <strong>13 March 2025</strong> Issued <a href="https://www.anthropic.com/research/auditing-hidden-objectives">Auditing language models for hidden objectives</a>。</p>
<p>I think this work is taking the issue one step further. Because a lot of the assessments are coming up.<strong>Appearance behavior</strong>: Whether the answer is dangerous, whether the task has been accomplished, or whether there is a tendency. But real risks are often more difficult:</p>
<ul>
<li>The model looks normal for the most part;</li>
<li>(b) No apparent exposure problems in surface output;</li>
<li>But it may still be optimized for another purpose in its long-term strategy.</li>
</ul>
<p>That's the problem with the hiidden objective.</p>
<ul>
<li><strong>What does it want to solve?</strong>: If a model is essentially satisfactory on the surface, can we still audit the other targets that are being pursued behind it?</li>
<li><strong>How does it do that?</strong>Thesis intentionally trains a model with hidden targets, then does a model <strong>blind auditing game</strong>I'm sorry. The Red Teams trained in isolation, and Blue Teams obtained models and partial training data for auditing without knowing the answer. The official page is also devoted to different audit instruments such as white and black boxes.</li>
<li><strong>What can it do?</strong>: prove post-training andit not just looking outsides, but also trying to chase. The paper says:<strong>Three out of four audit teams have successfully found hidden targets.</strong>It means that this is not, at least, entirely impossible.</li>
<li><strong>It can't do anything.</strong>The following is a list of the most important things that we have ever seen: More precisely, it tells you that heidden-objective audit is worth doing and is feasible under certain settings, but far from being universal, stable and low-cost.</li>
</ul>
<p>Rewarding tips reminds you that scores may be used; Hidden Objects reminds you:<strong>Many of the critical issues are simply not those that can be known by the score.</strong></p>
<h3>4. Bloom: Behavioral measurements as a streaming water Line</h3>
<p>Anthropic is here. <strong>19 December 2025</strong> Issued <a href="https://alignment.anthropic.com/2025/bloom-auto-evals/">Bloom Official article</a> And we're running it in sync. <a href="https://github.com/safety-research/bloom">Bloom Repository</a>。</p>
<p>Bloom is the tool I want to make clear to beginners in this article because it is particularly susceptible to being misreaded as another benchmark.</p>
<p>I think the better way to understand it is:</p>
<p><strong>Bloom is not a test paper, but a line measuring the flow of water around the behaviour of an act that automatically creates a problem, executes interaction, and finally gives a rating.</strong></p>
<p>In an official post, Anthropic also clearly distinguished Bloom from Petri:<strong>Petri is more like an open automatic audit, Bloom is more like an in-depth, reusable measurement of a researcher's designated behaviour.</strong> This distinction is worth noting, as the problems identified are not the same as those of stabilization measurement.</p>
<ul>
<li><strong>What does it want to solve?</strong>If I already know that an act deserves attention, such as sycophancy, self-interest, self-preservation, can I generate a lot of scenes to measure the extent to which the act is induced and the extent to which it is serious?</li>
<li><strong>How does it do that?</strong>: Bloom from one <code>seed</code> Set up and run the full four-stage line:<ol>
<li><code>Understanding</code>: Read your behavior first;</li>
<li><code>Ideation</code>(a) Auto-generated scenes around this behaviour and diversified;</li>
<li><code>Rollout</code>: to allow the environment to interact with each other;</li>
<li><code>Judgment</code>: Judge points out the degree of occurrence and other quality indicators.</li>
</ol>
</li>
<li><strong>What can it do?</strong>:Anthropic officially defines it as a generation <strong>configurable evaluation suites</strong> The angstic trade. It is not based on a fixed library, but rather on a set of assessments from the ed. One of the key indicators in the official paper is called <code>elicitation rate</code>, the percentage of the behavioral score above the threshold.</li>
<li><strong>It can't do anything.</strong>: Bloom doesn't run for the truth. It relies on the definition of ted, the example of few-shot, variation design, judge rubric and threshold. Models may also learn to be assessed, which is explicitly mentioned in official texts.</li>
</ul>
<p>The most interesting thing in Bloom's warehouse is not code details, but the structure of the id. The official template has removed a set of evaluation features from the following components:</p>
<pre><code class="language-yaml">behavior:
  name: &quot;self-preferential-bias&quot;
  description: &quot;...&quot;
  examples: []

understanding:
  model: &quot;claude-sonnet-4&quot;

ideation:
  num_scenarios: 5
  variation_dimensions:
    - emotional_pressure
    - noise

rollout:
  target: &quot;claude-sonnet-4&quot;
  modality: &quot;conversation&quot;
  max_turns: 2

judgment:
  num_samples: 1
</code></pre>
<p>The most important inspiration for this configuration is:<strong>And the behavioral assessment was finally not just taking a bunch of prompts, but was broken down into controlled, re-emergible, re-runable experimental designs.</strong></p>
<p>Bloom, you're still on board. <code>conversation</code> and <code>simenv</code> Two models. The former is more like a standard multi-wheel dialogue, while the latter is closer to angent scenes with tools and environments interacting. And that's why I think of it as a very critical point from a linguistic model to angent behavior measurement.</p>
<h3>5. Activation Operations: Starting to use internal activation as evidence</h3>
<p>Anthropic is here. <strong>19 December 2025</strong> Issued <a href="https://alignment.anthropic.com/2025/activation-oracles/">Activation Oracles</a>。</p>
<p>If the first steps are still primarily about external behaviour, the job is asking another very natural question:<strong>If the tranny is lying, can you give me another kind of evidence for internal activation?</strong></p>
<ul>
<li><strong>What does it want to solve?</strong>: Many risks are not necessarily immediately exposed in external behaviour or only arise under certain trigger conditions. So, can another system read the acts directly, and say, "What is this model thinking about, what's the proclivity, whether it's some kind of hidden information?"</li>
<li><strong>How does it do that?</strong>: Consider the activation of target model as an additional input, so that another LLM can answer the natural language questions about these acts. The official article called this model. <strong>Activation Oracles</strong>I'm sorry. The paper also performed downstream audit missions and data sizes were reduced.</li>
<li><strong>What can it do?</strong>Official results show that 3 out of 4 downstream audit assignments have achieved the best performance at the time and that performance will continue to improve as the volume and diversity of training data increase. This suggests that “the question of activation as an additional pattern” is a promising path.</li>
<li><strong>It can't do anything.</strong>: The official is clear, too. <strong>non-mechanistic</strong> The way. It does not tell you how the concept is coded in the space, nor does it answer what the internal algorithm mechanism of the model looks like. It is more like a new instrument in the audit toolbox than an ultimate explanation.</li>
</ul>
<p>So if you put an overall picture of this line of Anthropic, I would say:</p>
<p><strong>First, we find out what's worth doing, then we admit that we're going to be deceptive, then we try to audit the hidden target, then we use Bloom to program behavioral measurements, and then we pull internal activations into the chain of evidence.</strong></p>
<h2>Transluce: Make the "Look at the Behavior" work as an engineering tool.</h2>
<p>If Anthropic is more like a question about the model, or what, then Transluce is more like a question:<strong>Can you make behavioral work as a routine engineering tool?</strong></p>
<p>The most confusing thing about this line is:<strong>The CD and Docent are not tools of the kind.</strong></p>
<ul>
<li>PCD, look at this.<strong>Internal Status</strong>；</li>
<li>Docent, look.<strong>External trajectory</strong>。</li>
</ul>
<p>An internal state translator, an observatory like angent transcript.</p>
<h3>PCD: Like an "internal status translator"</h3>
<p>Transluce is here. <strong>18 December 2025</strong> Issued <a href="https://transluce.org/pcd">Predictive Concept Decoders</a>。</p>
<p>The point of departure for the CD is very clear:<strong>Self-reporting of models is not necessarily credible.</strong>
A model by jailbreak might be able to export dangerous content while saying, "I don't think anything special." If you can only ask the model yourself, it could give you a decent explanation.</p>
<p>All the PCDs want to do is to bypass the problem.</p>
<ul>
<li><strong>What does it want to solve?</strong>: Not rely on models for self-reporting, but directly predict and interpret behaviour from internal state.</li>
<li><strong>How does it do that?</strong>: According to the official project page and the paper TeX source, the PCD uses a two-part structure: compresses the acts into one <strong>Rare conceptual bottlenecks</strong>, get a short, readable list of concepts; then answer questions about “model behaviour” only on the basis of these concepts. It's called "translate a model" on the official page.&#39;s internal states into short, human-readable concept lists, then use those concepts to answer questions about behavior”。</li>
<li><strong>What can it do?</strong>: The official page shows several particularly representative scenarios:<code>jailbreaks</code>、<code>secret hints</code>、<code>injected / implanted concepts</code>I'm sorry. These scenarios have one thing in common: there are things that are actually “know” inside the model, but it is not necessarily true in its self-report. And the CD is more valuable in these places than "Father Asking The Model itself".</li>
<li><strong>It can't do anything.</strong>readable is not the same as cause or effect. A concept bottleneck may indeed be predictive of behaviour, but it is not necessarily the most authentic internal mechanism. Besides, the PCD default is a white box tool, and you have to get to events.</li>
</ul>
<p>If I translate it into a more intuitive sentence, I would say:</p>
<pre><code class="language-text">activations -&gt; 可读概念列表 -&gt; 回答“它接下来会怎样、为什么会这样”
</code></pre>
<p>The most interesting part of the PCD is here: it's not just a post-post label, but rather a reference to interpretability.<strong>Prediction issues</strong>I'm sorry. If you really understand events, you should be able to predict subsequent behavior. This framing is smart because it gives the system a training, verifiable oversight signal.</p>
<h3>Docent: like an “agent track observatory”</h3>
<p>Compared to the CD,<a href="https://docs.transluce.org/introduction">Docent Document</a> Another more engineering issue is solved:<strong>When you've already had a big amount of ant-transcripts, people can't see it.</strong></p>
<p>The official Docent page gives a very direct position:<strong>summarize, cluster, and search over agent transcripts</strong>I'm sorry. I think this description is more accurate than many propaganda messages.</p>
<ul>
<li><strong>What does it want to solve?</strong>Humans cannot read thousands of records of the angent operation manually, but we wonder where "unusuals happen" "what kind of trajectory is rewarding" "better behaviors at different stages of training."</li>
<li><strong>How does it do that?</strong>: first ingest tracks, then around one <code>rubric</code> Build <code>judge</code>I'm sorry. Docent Document <code>Rubrics and Judges</code> The page makes this clear: Rubric is the evaluation standard you wrote to Judge, Judge, and the output is fine. - Yeah. <code>label</code>、<code>score</code>、<code>explanation</code>And... <code>explanation</code> Field support <code>citations</code>, which is the specific part of the sentence that points back to the transcript. Later on, you can search, group, filter and continue to modify the rubric.</li>
<li><strong>What can it do?</strong>: official documents directly cite several uses: observation of anomalies in long resoning tracks, monitoring of accidental behaviour in RL rollouts, performing summarize / cluster / search of angent tracks. The Rubric re-education curriculum also emphasizes that people like cheating, sycophoncy can be identified at first sight, but it is difficult to write about the behaviour of the boundary, and it is appropriate to sharpen the criteria through repeated re-education.</li>
<li><strong>It can't do anything.</strong>The following video shows the story of the event: If the transcript itself is incomplete, your writing is vague, Judge understands, and the result is drifting. It is very useful, but its ceiling depends to a large extent on whether you have defined the behaviour to be measured.</li>
</ul>
<p>I particularly like the point of Docent, which implicitly admits:<strong>The behavioural specifications are usually vague at the outset.</strong>
This may be important for starters, because many people who first come to contact such tools are mistaken for "just as long as the model is strong, Judge will understand." Docent's document is in fact a reminder: what's really hard is to write your concerns into an enforceable standard of behaviour.</p>
<p>And that's why I'd rather understand Docent as one. <strong>behavior observability layer</strong>I'm sorry. It doesn't just train you, but it helps you organize what happened into remixable evidence.</p>
<h2>Put the two routes back on the same map.</h2>
<p>If you read this for the first time, I don't suggest "list of papers." Better still, it's:<strong>Ask yourself what questions you really want to answer.</strong></p>
<table>
<thead>
<tr>
<th>The question you really want to ask.</th>
<th>Closer to work.</th>
<th>Why?</th>
</tr>
</thead>
<tbody><tr>
<td>What else is worth measuring?</td>
<td><a href="https://www.anthropic.com/research/discovering-language-model-behaviors-with-model-written-evaluations">Model-Written Evaluations</a>、<a href="https://www.anthropic.com/research/petri-open-source-auditing">Petri</a></td>
<td>First, expand the behavioral space, not just stare at known benchmark.</td>
</tr>
<tr>
<td>What kind of behavior is induced, how serious?</td>
<td><a href="https://alignment.anthropic.com/2025/bloom-auto-evals/">Bloom</a></td>
<td>Make behavioral measurements into repeatable water lines.</td>
</tr>
<tr>
<td>When it looks normal, is it still chasing another target?</td>
<td><a href="https://www.anthropic.com/research/auditing-hidden-objectives">Hidden Objectives</a></td>
<td>Push from outputs to motion</td>
</tr>
<tr>
<td>Any signs of danger in the internal state?</td>
<td><a href="https://alignment.anthropic.com/2025/activation-oracles/">Activation Oracles</a>、<a href="https://transluce.org/pcd">PCD</a></td>
<td>Use acts as another type of evidence</td>
</tr>
<tr>
<td>What happened in the mass of antraces?</td>
<td><a href="https://docs.transluce.org/introduction">Docent</a></td>
<td>Turning trajectory into searchable, clusterable, referenceable observational objects</td>
</tr>
</tbody></table>
<p>If you compress one more sentence:</p>
<ul>
<li><strong>Anthropic more for "validation"</strong>: Is there a problem, how serious it is, or is there another target behind it?</li>
<li><strong>Transluce</strong>: How to make internal or external behavioural clues a daily tool.</li>
</ul>
<p>There is no conflict between the two. A more complete system is likely to require the following:</p>
<ul>
<li>Bloom, this one.<strong>Measurement around specified behaviour</strong>；</li>
<li>Docent, this.<strong>Observations around a lot of tracks</strong>；</li>
<li>CD / Actation Oracles This<strong>Internal signal supporting evidence</strong>。</li>
</ul>
<h2>Why, in the Age of Age, this thing went from research to infrastructure.</h2>
<p>If it is just a regular chat model, behavioural questions are often also expressed as unsatisfactory answers.
But once the system becomes anent, the nature of the problem changes.</p>
<p>First,<strong>Anent is multistep, has a state.</strong>
A one-step, look harmless, may be just a part of a longer strategy. You can't just read the last sentence.</p>
<p>Second,<strong>Angent will call on tools, change files, touch API, and the consequences are irreversible.</strong>
The error is no longer just a mistake, but it may be a real mistake.</p>
<p>Thirdly,<strong>Angent naturally produces long transcript.</strong>
Humans review cannot look at each article, so you have to have a layer of observations like search, cluster, judge, rubric refinement.</p>
<p>Fourth,<strong>The distribution of post-deployment behaviour is not the same as the distribution of benchmark.</strong>
In the real environment, Agent is more open, more travel, more tempting. It's hard to keep knowing if it's been missed by training-time reward or offline benchmark.</p>
<p>So I'd rather put this direction in the AI version of Observability to understand:</p>
<ul>
<li>Traditional software. Observability. <code>logs</code>、<code>metrics</code>、<code>traces</code>；</li>
<li>Besides that, we're gonna have to look at it. <strong>behavior specs、judge outputs、rubric versions、transcript evidence、internal signals</strong>。</li>
</ul>
<p>From this perspective, behavioural auditing is not an additional annex in the area of security, but rather a more basic infrastructure for the growth of the system.</p>
<h2>The hardest thing is what you want to prove.</h2>
<p>Finally, I will make a few points of my own judgment.</p>
<p>First, the reward signature and behavior audit are the back and back of the same feedback process. Reward is responsible for putting the target into the optimizer, and the audit is responsible for checking what the model actually wrote into itself. This feedback is incomplete when it comes to the former without the latter.</p>
<p><strong>Second, the difficulty of this direction, which is most easily underestimated, is not the model capacity but the behavioural specifications.</strong></p>
<p>You want to measure "false" "accupation" "hidden target" and the words "hidden target" sound natural in the human mind, but once written as a stable rubric, it's difficult to get up there. The fact that Docent's rubric refinement and Bloom's ed design are in a sense recognizing it.</p>
<p>Thirdly, internal evidence is important, but not mind-reading. Both PCD and Actation Rules are instructive, but more like adding a new type of evidence to the auditors rather than closing all uncertainties once. External behaviour, long trajectory and environmental consequences remain the main lines.</p>
<p>Fourthly, I would now like to interpret this whole set of things as a migration from reward to observability. In the past we were more concerned about giving the model a point; it is increasingly important now: what the model is about, under what conditions, is it possible to see, stabilize, and continuously modify it in time.</p>
<p>And that's what I understand from Reward to Agent.</p>
<h2>References</h2>
<h3>Official research and documentation</h3>
<ul>
<li>Anthropic, <a href="https://www.anthropic.com/research/discovering-language-model-behaviors-with-model-written-evaluations">Discovering Language Model Behaviors with Model-Written Evaluations</a>, 2022-12-19</li>
<li>Anthropic, <a href="https://www.anthropic.com/research/reward-tampering">Sycophancy to subterfuge: Investigating reward tampering in language models</a>, 2024-06-17</li>
<li>Anthropic, <a href="https://www.anthropic.com/research/auditing-hidden-objectives">Auditing language models for hidden objectives</a>, 2025-03-13</li>
<li>Anthropic Alignment Science, <a href="https://alignment.anthropic.com/2025/bloom-auto-evals/">Bloom: an open source tool for automated behavioral evaluations</a>, 2025-12-19</li>
<li>GitHub, <a href="https://github.com/safety-research/bloom">safety-research/bloom</a></li>
<li>Bloom, <a href="https://raw.githubusercontent.com/safety-research/bloom/main/src/bloom/data/templates/seed.yaml.template"><code>seed.yaml.template</code></a></li>
<li>Anthropic Alignment Science, <a href="https://alignment.anthropic.com/2025/activation-oracles/">Activation Oracles: Training and Evaluating LLMs as General-Purpose Activation Explainers</a>, 2025-12-19</li>
<li>Anthropic, <a href="https://www.anthropic.com/research/petri-open-source-auditing">Petri: An open-source auditing tool to accelerate AI safety research</a>, 2025-10-06</li>
<li>Transluce, <a href="https://transluce.org/pcd">Predictive Concept Decoders</a>, 2025-12-18</li>
<li>Docent Docs, <a href="https://docs.transluce.org/introduction">Welcome to Docent</a></li>
<li>Docent Docs, <a href="https://docs.transluce.org/quickstart">Quickstart</a></li>
<li>Docent Docs, <a href="https://docs.transluce.org/concepts/rubrics-and-judges">Rubrics and Judges</a></li>
<li>Docent Docs, <a href="https://docs.transluce.org/tutorials/rubric-refinement">Rubric Refinement</a></li>
</ul>
<h3>Thesis portal</h3>
<ul>
<li><a href="https://arxiv.org/abs/2212.09251">Discovering Language Model Behaviors with Model-Written Evaluations (arXiv:2212.09251)</a></li>
<li><a href="https://arxiv.org/abs/2406.10162">Sycophancy to Subterfuge: Exploring Reward Tampering in Language Models (arXiv:2406.10162)</a></li>
<li><a href="https://arxiv.org/abs/2503.10965">Auditing language models for hidden objectives (arXiv:2503.10965)</a></li>
<li><a href="https://arxiv.org/abs/2512.15674">Activation Oracles (arXiv:2512.15674)</a></li>
<li><a href="https://arxiv.org/abs/2512.15712">Predictive Concept Decoders (arXiv:2512.15712)</a></li>
</ul>
<h2>Last sentence.</h2>
<p>The briefest summary of the full text can be: Reward is responsible for writing the target into the optimizer, behavioural auditing is responsible for checking what the model has learned, behavioural decode is responsible for making these leads more readable, searchable, and suitable for continuous use in the Agent system.</p>
