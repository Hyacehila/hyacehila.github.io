---
title: Why Language Models Hallucinate
title_zh: Why Language Models Hallucinate
date: 2026-02-24 20:00:00 +0800
categories:
- Foundation Models
- Model Mechanics
tags:
- Evaluation
- Training Dynamics
- Model Behavior
- Paper Notes
author: Hyacehila
mathjax: true
excerpt: A reading of OpenAI's Why Language Models Hallucinate, arguing that hallucination is tied to training paradigms and
  binary evaluation pressure, not only data noise or model flaws.
description: A reading of OpenAI's Why Language Models Hallucinate, arguing that hallucination is tied to training paradigms
  and binary evaluation pressure, not only data noise or model flaws.
excerpt_zh: 基于 OpenAI 团队论文《Why Language Models Hallucinate》：幻觉并非单纯源于数据噪声或模型缺陷，也和现代训练范式与二元评估机制带来的统计压力有关。
permalink: /blog/2026/02/24/why-language-models-hallucinate/
lang: en
translation_key: 2026-02-24-why-language-models-hallucinate
translation_status: machine
translation_source_hash: ba5a4cf3a93a414e9ac44c9d54cbe28cc1a9ede18ef111b21da1e16ab136a939
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<blockquote>
<p>This post is taken from the OpenAI team Paper "Why Language Models Hallucinate"</p>
</blockquote>
<p>The questions in this article can also be addressed<a href="/en/blog/2026/01/26/neural-scaling-laws/">Neal Scaling Laws: From Kaplan to Chinchilla</a>、<a href="/en/blog/2026/02/20/compression-for-agi/">Compression for AGI: compression is intelligence</a>How the concept of a relatively close read together is developed in different contexts.</p>
<h1>Why Language Models Hallucinate</h1>
<h2>Summary</h2>
<p>The “satisfaction” of the large language models (LLMs) refers to the model producing statements of overconfidence but of error of fact. From the perspective of probabilistic and statistical learning theory:<strong>The illusion is not just a data noise or model structure flaw, but also a statistical pressure in a modern training paradigm.</strong> The paper proved that even if training data were completely non-noise, the optimization of cross-breath losses would lead to inevitable production errors. The illusions persist in the post-training phase, and are related to the assessment of incentives:<strong>The current mainstream assessment mechanism (Dual dollar 0-1 rating) will reward speculation by systematically punishing uncertainty expressions (e.g. “I do not know”).</strong></p>
<h2>Convert generation problems to classification issues</h2>
<h3>Generating models in a probabilistic perspective</h3>
<p>First, the problem of the formation of language models is de-probable. Set &#36;\mathcal{X}&#36; Dispersed space for all plausible strings (text). We'll... &#36;\mathcal{X}&#36; It is divided into two separate collections:</p>
<p>&#36;&#36;
\mathcal{X} = \mathcal{V} \cup \mathcal{E}, \quad \mathcal{V} \cap \mathcal{E} = \emptyset
&#36;&#36;</p>
<ul>
<li><strong>Effective string &#36;\mathcal{V}&#36;</strong>(Valid): Text that is correct and logical</li>
<li><strong>Error String &#36;\mathcal{E}&#36;</strong>(Error): Text containing errors or contradictions of fact (i.e. hallucinations)</li>
</ul>
<p>Set &#36;p&#36; For the real world's linguistic distribution (training data distribution), assuming training data are no noise, i.e. &#36;p(\mathcal{V}) = 1&#36;I'm sorry. Language Model &#36;\hat{p}&#36; It's through pre-training. &#36;p&#36; estimate.</p>
<p><strong>Definition 1 (sight rate)</strong>: The hallucinating rate of the model is defined as the probability of the model producing an error string:</p>
<p>&#36;&#36;
\text{err} := \hat{p}(\mathcal{E}) = \Pr_{x \sim \hat{p&#125;&#125;[x \in \mathcal{E}]
&#36;&#36;</p>
<h3>Is-It-Valid (IIV) Convention</h3>
<p>Core ideas:<strong>It is more difficult to generate a valid output than to judge whether it is effective</strong>I'm sorry. If the model produces a good content, it must be able to answer correctly the dualistic determination of whether this candidate output is valid.</p>
<p>We're building a monitoring learning problem: IV classification tasks.</p>
<p><strong>Test Distribution &#36;\mathcal{D}&#36;</strong>：</p>
<p>&#36;&#36;
\mathcal{D}(x) =
\begin{cases}
\frac{1}{2}p(x) &amp; \text{if}x\mathcal{V}\(+) \
\rvert rvert rvert rvert rvert rvert rvert rv rvert rv rv rv rv rv r\t r\rv r\rv r\rv r\rv r\r\rv r\rv r\r\rv}r\r\rv t\r\rv t\r\rv t\r\r\f t\r\fsc\fsc\f\f\f\f\f\f\f\f\f\f\f\f\f\f\f\f\f\f\f\f\f\f\f\f\f\f\t\f\f\f\f\f\f\f\t\f\f\f\t\f\f\ft\ft\ft\t\t\ft\t\f\t\t\t\t\t\t\t\t\t\t\rt\rt\ &amp; \text{if}x\mathcal{E}\(-)
The next thing I know, I'm not sure.
I'm sorry.</p>
<p>That is, 50% probability of distribution from training &#36;p&#36; Sample active string (positive sample), 50% probability from error string - Yeah. &#36;\mathcal{E}&#36; Medium even sample (negative sample).</p>
<p><strong>Catalogue Structure</strong>: A given language model &#36;\hat{p}&#36;, define IIV classification:</p>
<p>&#36;&#36;
\hat{f}(x) =
\begin{cases}</p>
<ul>
<li>&amp; \text{if}\hat{p}(x) &gt; \frac{1}{\lvert \mathcal{E} \rvert} \</li>
</ul>
<ul>
<li>&amp; \text \leq\frac \lvert\mathcal{E}\rvert}
The next thing I know, I'm not sure.
I'm sorry.</li>
</ul>
<p><strong>IIV Classification Error Rate</strong>：</p>
<p>&#36;&#36;
\text{err}<em>{\text{iiv&#125;&#125; := \Pr</em>{x \sim \mathcal{D&#125;&#125;[\hat{f}(x) \neq f(x)]
&#36;&#36;</p>
<p>of which &#36;f(x)&#36; As Real Tab (Pretty)&#36;+&#36; The blogger says:&#36;-&#36; The blogger says that the government is not in a position to do so.</p>
<h3>Core Theorem</h3>
<p><strong>Theorem 1 (no hints)</strong>: distribution of training at random &#36;p&#36;(Fulfilled) &#36;p(\mathcal{V})=1&#36;) and any language model &#36;\hat{p}&#36;, by:</p>
<p>&#36;&#36;
\text{err} \geq 2 \cdot \text{err}_{\text{iiv&#125;&#125; - \frac{\lvert \mathcal{V} \rvert}{\lvert \mathcal{E} \rvert} - \delta
&#36;&#36;</p>
<p>Of which:</p>
<ul>
<li>&#36;\delta = \lvert \hat{p}(A) - p(A)\rvert&#36; Yes<strong>Calibration error</strong></li>
<li>&#36;A = {x \in \mathcal{X} \mid \hat{p}(x) &gt; 1/lvert \\mathcal{E}&#36;&#36;&#36; for response collection above threshold</li>
</ul>
<p>Now we need to move the previous conclusions to the "with hint" scenario.</p>
<p>In reality, the model is based on the hint&#36;c \in \mathcal{C}&#36; Generate Response &#36;r&#36;I'm sorry. Set the hint distribution to &#36;\mu(c)&#36;Training is distributed to the probability of conditions &#36;p(r \mid c)&#36;。</p>
<p>For each hint &#36;c&#36;, defines:</p>
<p>&#36;&#36;
\mathcal{V}_c = {r \mid (c,r) \in \mathcal{V&#125;&#125;, \quad \mathcal{E}_c = {r \mid (c,r) \in \mathcal{E&#125;&#125;
&#36;&#36;</p>
<p><strong>Key parameters</strong>：</p>
<ul>
<li>&#36;K = \min_c \lvert \mathcal{E}_c \rvert&#36;: Number of error responses for simple tips</li>
<li>&#36;k = \max_c \lvert \mathcal{V}_c \rvert&#36;: Number of correct responses to the most difficult tips</li>
</ul>
<p>The conditions are deformed and the conclusions given in the inferences are easily transposed into the form of a probability of conditions (see the appendix to the original paper for proof).</p>
<p><strong>Test Distribution</strong>：</p>
<p>&#36;&#36;
\mathcal{D}(c,r) =
\begin{cases}
\frac{1}{2}\mu(c)p(r \mid c) &amp; \text{r\mathcal{V} c \c
\mu\frac \lvert\mathcal{E} c\rvert} &amp; \text{r\mathcal{E} c
The next thing I know, I'm not sure.
I'm sorry.</p>
<p><strong>Catalogue</strong>：</p>
<p>&#36;&#36;
\hat{f}(c,r) = + \iff \hat{p}(r \mid c) &gt; \frac{1}{\min_c \lvert \mathcal{E}_c \rvert}
&#36;&#36;</p>
<p>Directly give theorem 1 extension form:</p>
<p><strong>Theorem 2 (with hints)</strong>: For any &#36;p&#36;（&#36;p(\mathcal{V})=1&#36;and &#36;\hat{p}&#36;, by:</p>
<p>&#36;&#36;
\text{err} \geq 2 \cdot \text{err}_{\text{iiv&#125;&#125; - \frac{\max_c \lvert \mathcal{V}_c \rvert}{\min_c \lvert \mathcal{E}_c \rvert} - \delta
&#36;&#36;</p>
<p>of which &#36;\delta = \lvert \hat{p}(A) - p(A)\rvert&#36;，&#36;A = {(c,r) \mid \hat{p}(r \mid c) &gt; 1/\min_c \lvert \mathcal{E}_c \rvert}&#36;。</p>
<p>For many of the real world,<strong>The theory of computing learning has given us a corresponding answer. &#36;\text{err}_{\text{iiv&#125;&#125;&#36; Bottom</strong>I'm sorry. So in these cases, it's subject to two of the above-mentioned variations,<strong>The illusion of a language model cannot be completely eliminated: they exist in a defined sub-class of people, depending on the error rate for the question of the second classification.</strong></p>
<h2>Post-training hallucinations.</h2>
<h3>Stimulation distortions in the assessment mechanism</h3>
<p>Current language model assessment is widely used<strong>Double score.</strong>(0-1 loss):</p>
<ul>
<li>Correct answer: 1 point or full score</li>
<li>Wrong answer or "I don't know."&#39;t know(IDK): 0 minutes</li>
</ul>
<p><strong>Observation 1 (optimal strategy for binary scoring)</strong>: Distribution of beliefs &#36;\rho_c&#36; In the right answer, IDRK responded with a strict rating of expectations below any speculative response with a non-zero probability.</p>
<p><strong>Formalization of certificates</strong>: Set rating functions &#36;g_c: \mathcal{R}_c \to {0,1}&#36; Satisfied &#36;g_c(r) = 0&#36; For All &#36;r \in \mathcal{A}_c&#36;(IDK ASS) Exists at least one &#36;r* \notin \mathcal{A}<em>c&#36; 使得 &#36;\Pr</em>{g_c \sim \rho_c}[g_c(r^*)=1] &gt; &#36;0. Therefore:</p>
<p>&#36;&#36;
\mathbb{E}<em>{g_c \sim \rho_c}[g_c(r^*)] &gt; 0 = \mathbb{E}</em>{g_c \sim \rho_c}[g_c(\text{IDK})]
&#36;&#36;</p>
<p><strong>Teaching-like</strong>This is like most of the calibration tests -- zero points left in the air, and no points taken off by miscalculation, so random speculation is the strategy to maximize the benefits, even if the answers are not known.</p>
<h3>Current status of mainstream assessments</h3>
<p><strong>Table 1: Rating of the baseline for the mainstream assessment</strong></p>
<table>
<thead>
<tr>
<th>Benchmark</th>
<th>Modalities of assessment</th>
<th>Double score.</th>
<th>IDK Score</th>
</tr>
</thead>
<tbody><tr>
<td>GPQA</td>
<td>Multiple accuracy rate</td>
<td>Yes.</td>
<td>None</td>
</tr>
<tr>
<td>MMLU-Pro</td>
<td>Multiple accuracy rate</td>
<td>Yes.</td>
<td>None</td>
</tr>
<tr>
<td>IFEval</td>
<td>Command Follow Verification</td>
<td>Yes.</td>
<td>None</td>
</tr>
<tr>
<td>Omni-MATH</td>
<td>Mathematical Equivalence</td>
<td>Yes.</td>
<td>None</td>
</tr>
<tr>
<td>SWE-bench</td>
<td>Unit test passed.</td>
<td>Yes.</td>
<td>None</td>
</tr>
<tr>
<td>WildBench</td>
<td>LM Rating (1-10)</td>
<td>Yes</td>
<td>Partial (but lower than hallucinogenic) &quot;fair&quot; Response)</td>
</tr>
</tbody></table>
<p><strong>Conclusions</strong>: The paper statistics point out that most mainstream benchmarks severely penalize IDK, leading to models in<strong>The assessment system is more speculative</strong>。</p>
<h3>The relief of hallucinations requires a change in the mainstream rating, Benchmark.</h3>
<p>Most of the existing mainstream assessment benchmarks use a dual rating. Benchmark, which is used for hallucinogenic assessment, exists but is not always included in the core indicators. To mitigate LLM's hallucinations, at least the core benchmark needs to include the expression of uncertainty in the rating design. So long as LLM can also obtain higher ratings on random speculation during testing, the problem of illusions is difficult to solve: because models are still rewarded for speculation, it is impossible to stabilize the IDRK answer.</p>
