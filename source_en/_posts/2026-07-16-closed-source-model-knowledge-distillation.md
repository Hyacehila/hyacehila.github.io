---
title: Can Closed-Source Models Be Distilled? Knowledge Distillation for Generative Language Models
title_zh: 闭源模型不能被蒸馏？聊聊生成式语言模型的 Knowledge Distillation
date: 2026-07-16 20:00:00 +0800
categories:
- Foundation Models
- Training & Alignment
tags:
- Knowledge Distillation
- LLM Training
- Synthetic Data
- SFT
- On-Policy Learning
author: Hyacehila
mathjax: false
hidden: true
excerpt: Whether a closed-source model can be distilled depends on the signal it exposes, the student's training objective,
  and the authorization of the training pipeline—not simply on whether the model is open or closed.
description: Whether a closed-source model can be distilled depends on the signal it exposes, the student's training objective,
  and the authorization of the training pipeline—not simply on whether the model is open or closed.
excerpt_zh: 闭源模型能否被蒸馏，要看教师实际提供什么信号、学生在拟合什么目标，以及这条训练链路是否获得授权；重点是训练链路怎么设计，而不是模型开源与否。
permalink: /blog/2026/07/16/closed-source-model-knowledge-distillation/
lang: en
translation_key: 2026-07-16-closed-source-model-knowledge-distillation
translation_status: machine
translation_source_hash: 193e96bfa65d238b0b140ebed329e8e8701026282f7d49abc83c7f1f09a5c372
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<p>Recently, distillation has become more than a technical word. Anthropic describes the unauthorized, bulk-based use of its model output to train competition models as <a href="https://www.anthropic.com/news/detecting-and-preventing-distillation-attacks">“distillation attacks”</a>I'm sorry. In Chinese, the question is often answered: how can the closed-source model be distilled without the public weight?</p>
<p>The questions in this article can also be addressed<a href="/en/blog/2024/11/01/llm-post-training-and-finetuning/">Post-linguistic training and fine-tuning of practice: from SFT, LoRA to human alignment</a>、<a href="/en/blog/2026/07/03/sft-synthetic-data-engineering/">Data synthesis is becoming a project: from Terminal-Corpus</a>How the concept of a relatively close read together is developed in different contexts.</p>
<p>It looks like a yes-or-no problem, actually mixing up different levels. Neither the direct answer to “can” nor “can” is sufficient.</p>
<ol>
<li>The teacher model provides a signal.</li>
<li>What exactly is the student training plan?</li>
<li>Do you have the right to use models and data like this?</li>
</ol>
<p>If these three things are not removed, it is easy to slip the discussion to the slogan: while all model aids are called distillation, the other side, because the model has no public weight, asserts that it cannot be distilled.</p>
<p>Focus on the training chain, whether the model is open or closed.</p>
<h2>Classic KD: Students come in direct to teachers</h2>
<p>The core of the classic Knowledge Division (KD) is simple: teachers have learned to behave and students are trained to reduce their differences directly from teachers.</p>
<p>The typical thing is...<strong>Output distillation</strong>I'm sorry. The question raised by Hinton and others about the soft classes not only tells students what the correct categories are, but also retains information on how close the teachers think the other categories are to the correct answers. In classification tasks, students are drawn up by the probability distribution given by the teacher ' s output layer. In the generation model, this distribution becomes the next probability distribution for token; it is the teacher who is more inclined to say what he/she will say at every step.</p>
<p>In addition, there are two types of classic practices:</p>
<ul>
<li><strong>Characteristic distillation</strong>: Alignment hide layer, attitudinal or intermediate. FitNets' instinct is that students should not only copy the final teacher response card, but also use the teacher ' s intermediary advice.</li>
<li><strong>Relationship distillation</strong>Instead of forcing the two models to look the same at each level, students are left to the teacher to express the distance, angle, similarity, etc. in space.</li>
</ul>
<p>These three methods are all doing the same thing:<strong>The goal of student excellence is more like a teacher.</strong> Characteristic distillation and relationship distillation usually require access to the hidden layer or weight, so it is more appropriate for the white box scene; the classical token-level KD has the lowest threshold, but still requires access to the final logits and needs to ensure that the vocabulary of students and teachers is aligned.</p>
<p>For LLM, the black box model, although not complete logits, can give a string of outputs tokens. Students target this token and train themselves to continue to produce with cross-critics. This form of loss is the same as self-repatriation pre-training, and the difference is simply that the monitoring signal is replaced by the original language into the teacher's output. The final answer, long-debate text, JSON, functional call and tool call track, as long as it is used to train students, is an output or sequence-level behavioural imitation. The training signal has not changed: students are still reproducing a sequence that has been given. This is what Kim and Rush call a security-level KD; from the data line, it's also like using teachers' answers as a false label for SFT.</p>
<p>This border need not be too categorical. The more useful question is:<strong>When students are updated, are they learning directly from teachers or in a redesigned data and feedback system?</strong></p>
<h2>It's another training link.</h2>
<p>To involve stronger models in training does not mean that the entire output is moved into the training set.</p>
<p>Models can also be just a tool in the data production and assessment chain: scaling up seed tasks, constructing counter-scenes and dilemmas, generating candidate answers, helping to mark preferences, or acting as filters. The trainers then put it together with the search for evidence, rule-checking, unit testing, manual auditing or incentive models to generate data required for SFT, preference optimization or RL.</p>
<p>Self-Instract is here: the command can expand input reporting data without requiring that each sample be written manually from zero. But model generation is not the same as natural high quality. The choice of the distribution of tasks, which samples are retained, what is correct, which acts should be rejected, what to reward, and how to validate, still depends on the trainers to judge.</p>
<p>That is why I want to separate the two links.</p>
<p>The goal of the pure KD is to reduce the distance between students and teachers. As a technical tool, it can certainly be used in home-grown models, clearly mandated models, or in the training chain for teachers-students within the team. But if the context becomes "Closed-source-Power Model Output" and small models, as much as possible, it looks more like a behavioral reproduction: Trainers are hardly part of their judgment, and the goal is to repeat what the teacher said. It's a lack of taste, you're not training a model, but is replicating, not deciding what to teach, what not to learn.</p>
<p>Model-aided SFT or RL are different things. The value here is not only what teachers say, but also how people design curricula, constrain data sources, join certification machines, organize difficulties, define rewards and fail borders. It does not have to faithfully replicate every word of the teacher, and it can even clearly filter out the teachers ' bad answers. It is a technology of data synthesis, not a mere reproduction.</p>
<p>Nor can it be called as a substitute for content. A large number of teachers were arrested, the same text was inserted into the training set, and no new mission design and quality control was available, even if it was called synthetic data, which was another output. Conversely, the same COT or tool tracks can also be teaching materials, provided they are validated, reorganized and placed in a new task and assessment chain.<strong>The difference is not what the text looks like, but what the training links look like.</strong></p>
<h2>On-policy KD: Why do students write first?</h2>
<p>The above distinguishes between training objectives. Even if the target remains distilled for the self-regression model, there is a problem in the training process: students see different prefixes when they are trained and deployed.</p>
<p>Normal offline KDs are often this way:</p>
<blockquote>
<p>Real data or teachers continue to copy students imitating.</p>
</blockquote>
<p>But when deployed, students do not face ideal prefixes provided by teachers, but rather those they have just written. It may have misled an entity, missed a condition, or picked the wrong parameter in the tool call. It can only continue to go down in this distorted context.</p>
<p>On-policy KD:</p>
<blockquote>
<p>The student teacher gives a distribution or feedback on the student's own trajectory.</p>
</blockquote>
<p>It deals with a mismatch in the distribution of the state between training and deployment, rather than with the creation of a further data set. MiniLM discusses LLM distillation from the point of view of on-policy and reverse-KL; GKD directly studies teacher feedback on student generation.</p>
<p>Therefore, the fact that the data are not the third type of data synthesis is not equal to the fact that the teachers produce the data and then make RL. It also targets teacher behaviour, but only allows teachers to give feedback on the trajectory that students actually reach. This can be achieved by using tools close to the tactical optimization, but it is still discussed in distillation.</p>
<h2>Black and white boxes: only limited to signals, not conclusions</h2>
<p>On-policy KD asks teachers on which tracks to give feedback. And then, what we're going to ask is what kind of signal the teacher can give. The difference between open and closed sources is mainly here, rather than deciding directly whether or not to distill.</p>
<ul>
<li><strong>The white box teacher.</strong>: access logits, hide layers, attention and relationship structure. The three types of KD that are output, feature, relationship can be established and can be more easily on-policy KD.</li>
<li><strong>Only the final text of the black box teacher.</strong>: usually provides answers, reasoning texts, tool tracks or preferred judgements, but cannot be distilled from the hidden layer or the full vocabulary of logits KD.</li>
<li><strong>Interface with logprobs</strong>: In the middle. Even if weights are not disclosed, KD may still be limited if sufficient token probabilities are obtained; this is not the white box feature distillation or relationship distillation.</li>
</ul>
<p>This also means that closed source models cannot be distilled too full. More specifically:</p>
<blockquote>
<p>Closed-source models do not necessarily support white box distillation; only text-output interfaces do not necessarily support classic token-level KD; however, they may still provide information that can be recalculated or processed into SFT/RL data.</p>
</blockquote>
<p>The question of whether the closed source model can be distilled has reduced several layers to a single sentence and is therefore not suitable for a one-size-fits-all answer. The same paragraph, which is exported by the teacher, is a whole-student training package, allowing the student to reset the teacher as much as possible, is a replica of behaviour; when the trainee re-decides the task, screens, validates and rewards, the teacher is only one source of material. These choices are what I say is about: what is worth teaching, what is credible, what mistakes must be rejected. Whether the closed-source model is used is only a surface layer, and the real gap is whether the trainers put these judgments in the training chain.</p>
<h2>References</h2>
<ul>
<li>Geoffrey Hinton, Oriol Vinyals, Jeff Dean, <a href="https://arxiv.org/abs/1503.02531">Distilling the Knowledge in a Neural Network</a></li>
<li>Adriana Romero et al., <a href="https://arxiv.org/abs/1412.6550">FitNets: Hints for Thin Deep Nets</a></li>
<li>Wonpyo Park et al., <a href="https://arxiv.org/abs/1904.05068">Relational Knowledge Distillation</a></li>
<li>Yoon Kim, Alexander M. Rush, <a href="https://arxiv.org/abs/1606.07947">Sequence-Level Knowledge Distillation</a></li>
<li>Yizhong Wang et al., <a href="https://arxiv.org/abs/2212.10560">Self-Instruct: Aligning Language Models with Self-Generated Instructions</a></li>
<li>Zuyang Gu et al., <a href="https://arxiv.org/abs/2306.08543">MiniLLM: Knowledge Distillation of Large Language Models</a></li>
<li>Rishabh Agarwal et al., <a href="https://arxiv.org/abs/2306.13649">GKD: Generalized Knowledge Distillation for Auto-regressive Sequence Models</a></li>
<li>Florian Tramèr et al., <a href="https://arxiv.org/abs/1609.02943">Stealing Machine Learning Models via Prediction APIs</a></li>
<li>Anthropic, <a href="https://www.anthropic.com/news/detecting-and-preventing-distillation-attacks">Detecting and preventing distillation attacks</a></li>
</ul>
