---
title: MineCLIP, Visual Signals, and Reward Design
title_zh: MineCLIP、视觉信号与奖励函数
date: 2026-03-23 21:00:00 +0800
categories:
- Agent Systems
- Agent Training
tags:
- Vision-Language Models
- Reinforcement Learning
- Reward Modeling
author: Hyacehila
mathjax: true
hidden: true
excerpt: A short appendix on MineCLIP as a vision-language reward model for Minecraft skill learning, and how Plan4MC uses
  this idea in its reward stack.
description: A short appendix on MineCLIP as a vision-language reward model for Minecraft skill learning, and how Plan4MC
  uses this idea in its reward stack.
excerpt_zh: Minecraft agent 的目标往往很清楚，麻烦在于中间步骤没有奖励。MineCLIP 把视频片段和任务文本的相似度变成 dense reward，Plan4MC 则展示了这种视觉奖励在技能训练里怎样被使用。
permalink: /blog/2026/03/23/mineclip-visual-reward-appendix/
lang: en
translation_key: 2026-03-23-mineclip-visual-reward-appendix
translation_status: machine
translation_source_hash: 4f4764c45d9cb61ef6fb1e8064c8808576fd59036f605b11203cdcf75a9102b9
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<p>Minecraft AGent's ultimate goal is often clear: milla cow, craft an iron pickaxe, dig a hole. Trouble in the middle. It is difficult for the environment to give valuable feedback before the mission is completed. RL knows the end, but doesn't know if that step is closer to the end.</p>
<p>The questions in this article can also be addressed<a href="/en/blog/2026/03/22/reward-and-training-in-agent-k-paperbench-amap/">How Reward and Training close the loop in real Agent: from data governance to online RL</a>、<a href="/en/blog/2026/03/21/from-sft-to-agentic-rl-training-loop/">Actic RL: Why is training closed rings more important than training algorithms?</a>How the concept of a relatively close read together is developed in different contexts.</p>
<p>When long-range missions are detached into skills, the problem is smaller, but it will not disappear.<code>find a cow</code>、<code>harvest milk_bucket</code>、<code>place crafting_table</code> The task is shorter than the full one and feedback is still needed. Handwritten for each skill, will soon become a bunch of rules of vulnerability. MineCLIP is here: train visual-linguistic models with video and subtitles from players to judge whether this recent image is like this skill description.</p>
<p><a href="https://arxiv.org/abs/2206.08853">MineDojo</a> The MineCLIP is proposed to cut in from this gap. It uses YouTube video clips and time to train video-language compatible mode. It's like the CLIP, just a short video.</p>
<p>Give Video Window &#36;V_t&#36; and task text &#36;G&#36;Video encoder &#36;\phi_V&#36; Output &#36;v_t=\phi_V(V_t)&#36;, text encoder &#36;\phi_G&#36; Output &#36;g=\phi_G(G)&#36;I'm sorry. Reward head to subsynthetic cosine similarity, multiplied by a learning temperature:</p>
<p>&#36;&#36;
s(V_t,G)=\exp(\alpha)\left\langle
\frac{\phi_V(V_t)}{|\phi_V(V_t)|},
\frac{\phi_G(G)}{|\phi_G(G)|}
\right\rangle .
&#36;&#36;</p>
<p>During the training, every video clip in the battling is subtitled. The positive sample is the same video-text pair, while the negative sample is from other text or video in the bat. InfoNCE:</p>
<p>&#36;&#36;
\mathcal{L}<em>{v\rightarrow g}
=-\frac{1}{B}\sum</em>{i=1}^{B}
\log
\frac{\exp(s(V_i,G_i))}
{\sum_{j=1}^{B}\exp(s(V_i,G_j))}.
&#36;&#36;</p>
<p>After training,&#36;s(V,G)&#36; Not just service search, but also as a soft judgement: is this observation consistent with the target description?</p>
<p>When entering RL, the smart body takes the nearest 16 frames at each step &#36;V_t&#36;I'm sorry. If the candidate text is &#36;\mathcal{G}={G,G_1^-,\ldots,G_{N_T-1}^-}&#36;MineCLIP first converts the similarity of the target text to softmax probability:</p>
<p>&#36;&#36;
P_{G,t}=
\frac{\exp(s(V_t,G))}
{\sum_{G&#39;\in\mathcal{G&#125;&#125;\exp(s(V_t,G&#39;))}.
&#36;&#36;</p>
<p>The appendix to the paper discussed two ways to convert the target amount. The first one is direct reward:</p>
<p>&#36;&#36;
r_t=\max\left(P_{G,t}-\frac{1}{N_T},0\right).
&#36;&#36;</p>
<p>&#36;1/N_T&#36; It's a random guess baseline. Visual matching below the baseline is not rewarded to send the models themselves to the optimizers. The second one is delta reward:</p>
<p>&#36;&#36;
r_t=P_{G,t}-P_{G,t-1}.
&#36;&#36;</p>
<p>This is more like a progress reward. It does not reward standing on the same thing, but it rewards the visual change closer to the goal. Direct is more effective for moving animal missions; static targets may simply allow angent to stare at the target, but forget to continue interacting.</p>
<p>Change is simple: traditional rare rewards are given only when the mission is completed. &#36;1&#36; or &#36;100&#36;; MINECLIP gives a similarity of visual language at each time step. It eases the problem of exploration and saves a lot of handwritten work. But it's still proxy. Visually, it's like a Sheep, not the wool actually goes into the backpack.</p>
<p><a href="https://arxiv.org/abs/2303.16563">Plan4MC</a> It's a good combination of traditional RL and LLM Agent. It does not use MineCLIP as a reward, but it starts by tearing the Minecraft skills into three categories: Finding-kills, Manipulation-kills dig, kill, place, collect, Crafting-skills synthesis. The upper level produces skill graph graph, then the graphics are used to search for skill sequences; only the bottom skills are still RL-based.</p>
<p>When training Manipulation-skills, Plan4MC uses MineCLIP to train inspiric reward. Approach and MineDojo are close: take past 16 frames, score with current skill sample and 31 negative samples, and get the target proft max probability &#36;p&#36;, and then to:</p>
<p>&#36;&#36;
r_{\mathrm{CLIP&#125;&#125;=\max\left(p-\frac{1}{32},0\right).
&#36;&#36;</p>
<p>Plan4MC also writes the border very well. MineCLIP rewarded is useful for some visually recapable skills, but does not cover all behaviours. Add distance and attack reward, log / cobblestone, iron ore / diamund, and depth reward. VLM provides some signals, but not a complete rewardback.</p>
<p>Visual rewards are also easy to drill into. Optimizers look for the state that makes the model feel like, not necessarily for the state that actually enables the environment to be completed. Angent may have aimed its perspective at an entity, creating a high-level image of similarity without correctly interacting. Visual signals are dense, but no stocks, formulations, durable tools and long-term causal chains are visible.</p>
<p>MineCLIP is a partial answer: it solves the problem of signal density in skills training, does not solve the whole of the program, state validation and long term credit accreditation. Plan4MC just showed it. Pre-trained visual-linguistic models can turn a target into a reward, but available anent has to do more, reward.</p>
<p><a href="https://arxiv.org/abs/2303.10571">CLIP4MC</a> Similar studies, with similar overall logic, could be used as a reference. It's not going to start here.</p>
<p>References:</p>
<ul>
<li><a href="https://arxiv.org/abs/2206.08853">MineDojo</a></li>
<li><a href="https://github.com/MineDojo/MineCLIP">MineCLIP GitHub</a></li>
<li><a href="https://arxiv.org/abs/2303.16563">Plan4MC</a></li>
<li><a href="https://arxiv.org/abs/2303.10571">CLIP4MC</a></li>
</ul>
