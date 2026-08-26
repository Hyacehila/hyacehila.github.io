---
title: 'Neural Scaling Laws: From Kaplan to Chinchilla'
title_zh: Neural Scaling Laws：从 Kaplan 到 Chinchilla
date: 2026-01-26 21:00:00 +0800
categories:
- Foundation Models
- Model Mechanics
tags:
- Pre-Training
- Model Mechanics
- Paper Notes
author: Hyacehila
mathjax: true
excerpt: A summary of empirical scaling laws from Kaplan to Chinchilla, with a brief discussion of tree models and scaling
  laws.
description: A summary of empirical scaling laws from Kaplan to Chinchilla, with a brief discussion of tree models and scaling
  laws.
excerpt_zh: 本文从 Kaplan 定律到 Chinchilla 修正，整理 Scaling Law 的经验规律，并简单讨论 Tree Model 与 Scaling Law 的关系。
permalink: /blog/2026/01/26/neural-scaling-laws/
lang: en
translation_key: 2026-01-26-neural-scaling-laws
translation_status: machine
translation_source_hash: 54ad74ffe0e674fe961e75bb9c1f3b4b12270efea4a0b551eebbe10040c6c933
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<blockquote>
<p>I'm not studying Deep Learning Theory, and I'm not familiar with the content; the following is just a brief summary of a theoretical field. Write anything about fun.</p>
</blockquote>
<p>The questions in this article can also be addressed<a href="/en/blog/2026/02/20/compression-for-agi/">Compression for AGI: compression is intelligence</a>How the concept of a relatively close read together is developed in different contexts.</p>
<p>If you want to see Scaling Law first, you can read it first, between data, pre-training, post-training and deployment.<a href="/en/blog/2024/08/15/llm-lifecycle-overview/">LLM Life Cycle Overview</a>;This paper expands only the size pattern itself.</p>
<h2>From 0 to 1 from Neal Scaling Laws</h2>
<p>The Neurocaling Laws are a common set of empirical patterns that understand large-scale in-depth learning (especially the LLM model). It describes<strong>The relationship between model performance and Power Law calculations of resources, data and parameters.</strong></p>
<p>Within a given range, multiple increases in the algorithm, data or parameters, and model Loss (loss function) usually drops at a relatively stable rate. This has moved the refining from a purely intuitive approach to the process, to an additional tool for estimating work that is not necessarily accurate but sufficiently useful (P.s. The material of the new age is refined and extremely friendly to the experimenters, except for their hair).</p>
<p>The most basic formula for Neal Scaling Laws is usually:
&#36;&#36; L(x) \propto x^{-\alpha} &#36;&#36;
of which &#36;L&#36; It's on the test set, Los.&#36;x&#36; is a size variable (e.g., parameter volume) &#36;N&#36;Data volume &#36;D&#36; Or, uh, the math. &#36;C&#36;），&#36;\alpha&#36; is the scaling index.
<strong>Main findings:</strong> Within the scope of the experiment, model performance is mainly influenced by size (Scale) and has a weaker relationship to specific model structures (e.g., layers, width ratios) (as long as the architecture is not too disproportionate).</p>
<h3>Baseline: Kaplan Law for OpenAI - Multiplication and Parameter Prefer</h3>
<p>Kaplan Team Hypothetical Test Set Los (&#36;L&#36;) with the amount of parameters ( )&#36;N&#36;) and the size of the data set (&#36;D&#36;Following the laws of independence and the two are bound together. Original article is <em>Scaling Laws for Neural Language Models</em> (Kaplan et al., OpenAI) </p>
<p>It's often compared to the age of big models.&quot;Moore's Law.&quot;I'm sorry. More precisely, it uses experiments to combine performance and performance of Transformer models. &#36;N&#36;(Agrilateral quantities),&#36;D&#36;(Size data),&#36;C&#36;There is a stable arctic relationship (calculated amount).</p>
<p>Its single variable forms:
&#36;&#36; L(N) \approx \left( \frac{N_c}{N} \right)^{\alpha_N}, \quad L(D) \approx \left( \frac{D_c}{D} \right)^{\alpha_D} &#36;&#36;
of which &#36;\alpha_N \approx 0.076&#36;, &#36;\alpha_D \approx 0.095&#36;, using experiments to prepare.</p>
<p>To enable Scaling Laws' experience formula to further guide the next stage of architecture design, Kaplan proposed a coupling.<strong>Joint Zoom Formula</strong>To describe &#36;N&#36; and &#36;D&#36; Time limits:
&#36;&#36; L(N, D) = \left[ \left( \frac{N_c}{N} \right)^{\frac{\alpha_N}{\alpha_D&#125;&#125; + \frac{D_c}{D} \right]^{\alpha_D} &#36;&#36;
 <strong>Math meaning:</strong> This is a similar, even-handed form. Because &#36;\alpha N &lt; \alpha_D&#36;，这意味着随着算力增加，参数 &#36;N&#36; 的边际收益递减速度比数据 &#36;D.L.L., slow down.</p>
<p> <strong>Inference:</strong> To minimize the loss of Los,<strong>Arguments &#36;N&#36; It should grow faster than the amount of data. &#36;D&#36;</strong>(i.e., &#36;N \propto C^{0.73}, D \propto C^{0.27}&#36;I'm not sure. They therefore suggested that, in increasing the ability to calculate, priority should be given to the model being larger than to the unlimited increase in data.</p>
<p>This directly led to the early large model (e.g. GPT-3, PLM, MT-NLG) crazy stack parameters (175B, 540B), but training data were relatively small (usually only 1 Epoch trained).</p>
<h3>Amendment time: Chinchilla Law of DeepMind - Added in equal scaling Fire!</h3>
<p>The Chinchira team is here <em>Training Compute-Optimal Large Language Models</em> (Hoffmann et al., DeepMind) points to the loophole in the Kaplan method when it's ready to fit the hyperparameter and suggests a more intuitive one.<strong>Adding Model</strong>, contains non-acceptable errors.</p>
<p><strong>Core formula:</strong>
&#36;&#36; L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta} &#36;&#36;</p>
<ul>
<li>&#36;E&#36;: Irreducible Los, i.e. the entropy of the natural language itself (Beyers error), even if the model is perfect.</li>
<li>&#36;\frac{A}{N^\alpha}&#36;: <strong>Model Approximate Error</strong>, which results from inadequate model capacity.</li>
<li>&#36;\frac{B}{D^\beta}&#36;: <strong>Data estimate error</strong>, from the differences caused by limited samples.</li>
</ul>
<p>It's the last thing I can do. &#36;C \approx 6ND&#36; Under a fixed constraint, use the Lagrams multiplier to reach the extreme, found in DeepMind &#36;\alpha \approx 0.5, \beta \approx 0.5&#36;I'm sorry. Because &#36;\alpha \approx \beta&#36;，&#36;N&#36; and &#36;D&#36; It's a similar contribution to Los.<strong>Symmetry</strong>I'm sorry. The best strategy under the fixed sum is to... <strong>&#36;N&#36; and &#36;D&#36; It should be on the scale.</strong>（&#36;N \propto C^{0.5}, D \propto C^{0.5}&#36;I'm not sure. That's famous.&quot;Chinchilla Scaling Laws&quot;。</p>
<p>Chinchilla Scaling Laws overturned parts of Kaplan's conclusions. DeepMind found that previous models (e.g. GPT-3) were all<strong>Undertraining</strong>I'm sorry. For the current Transformer,<strong>Each parameter requires approximately 20 trainings</strong>(i.e. number of Tokens/ argument numbers) &#36;\approx 20&#36;I'm not sure. This has dramatically changed the direction of the industry away from the blind pursuit of trillion parameters, but rather of relatively small models + very large amounts of data.</p>
<h3>The power of the Scaling Laws&quot;Inconsistence&quot; (The Discontinuity)</h3>
<p>In the laws of Kaplan and Chinchilla, the decline of Loss is silky. But in practical applications, engineers have discovered an incomprehensible phenomenon:<strong>Losing the lower does not mean that models learn to do things.</strong></p>
<p>Models are going through one of these tasks (e.g. triple digit plus, complex reasoning). <strong>Phase Transmission</strong>: From&quot;Totally random.&quot;♪ And suddenly jumps to ♪&quot;Close to humans.&quot;。 <em>Emergent Abilities of Large Language Models</em> (Wei et al., Google, 2022) partly answered the question -- why is Loss smooth, but the result is steep?</p>
<p>We can use a simple probability model to extrapolate this non-linear.</p>
<p>Suppose a logical task needs to be continuous. &#36;L&#36; Steps is all right to score.
You're the one who's gonna get you. &#36;p&#36; Forecasting for models<strong>Single Token (or one-step reasoning)</strong> The correct rate. According to Scaling Laws, with size &#36;N&#36; Increase, one-step correct rate &#36;p&#36; It's smooth. &#36;p \propto 1 - N^{-\alpha}&#36;）。</p>
<p>But the probability of success for the whole mission &#36;P(\text{Task})&#36; is the product of all steps:
&#36;&#36; P(\text{Task}) = p^L &#36;&#36;</p>
<p>Here, a non-linear dimension of a trimester is introduced. Assumptions of mission requirements &#36;L=5&#36; Step:
<strong>Small model</strong> (&#36;p=0.5&#36;): &#36;P(\text{Task}) = 0.5^5 \approx 0.03&#36; (near 0, as shown)&quot;No, I won't.&quot;)</p>
<p><strong>Medium Model</strong> (&#36;p=0.8&#36;): &#36;P(\text{Task}) = 0.8^5 \approx 0.32&#36; ( still failing)</p>
<p><strong>Critical points</strong> (&#36;p \to 0.95&#36;): &#36;P(\text{Task}) \approx 0.77&#36; (<strong>The power is bursting.</strong>)</p>
<p><strong>Conclusions</strong> It's not magic, it's magic.<strong>Projection of microprobability multiplier effects on macro-indicators</strong>I'm sorry. That explains why we need to do the model so great -- because only one step is accurate. &#36;p&#36; The success rate of long chain reasoning when it's extremely close to 1. &#36;p^L&#36; It's only meaningful.</p>
<p>From the perspective of Metric, Stanford's Schaeffer (2023) points out that if we put the evaluation indicators from the perspective of the&quot;All right.&quot;In the smooth.&quot;Token Edit Distance&quot;, the curves rise back to smooth curves. It tells us:<strong>Capabilities have been built up, only because of the harsh assessment criteria, which have led us to lose sight of them at the tipping point.</strong></p>
<h3>Data repetition and quality: Data-Constrained Scaling</h3>
<p>Scaling Laws is demanding too much data, and the Internet is running out of high-quality text (Data Wall). We have a new problem:<strong>If the data are not sufficient, how many times can we learn from old data?</strong></p>
<p><em>Scaling Data-Constrained Language Models</em> (Muennighoff et al.) Amends the Chinchilla formula to include repeated training (Epochs, as &#36;R&#36;as a variable. He found out that the data...<strong>Validity</strong>With the number of repetitions declining.</p>
<p>We can create a simplified conceptual formula:
&#36;&#36; D_{eff} \approx D_{unique} \cdot (1 + \lambda \cdot \log R) &#36;&#36;
Or more intuitively, the conclusion is:
<strong>&#36;R \le 4&#36; (4 Epochs in):</strong> Data returns are almost non-depleted. Models extract residual value from duplicate data.</p>
<p><strong>&#36;R &gt; 40&#36;:</strong> The revenue is almost zero, and even over-compatibility leads to a backsliding of Los.</p>
<p>In an era of data depletion, we can only repeat four times at most the high-quality data in our hands. After that, new ways of finding solutions (e.g. synthetic data) must be found. Since... &#36;D&#36;The only way out of the constraints (quantity) is to increase the data quality coefficient. Microsoft Phi series proves:
&#36;&#36; L \propto \frac{1}{(Q \cdot D)^\beta} &#36;&#36;
If data quality &#36;Q&#36; High enough (e.g., synthetic data at textbook level), very small &#36;D&#36; And it's going to be very low, Los. It breaks blind Scaling superstition.</p>
<h3>Mixing expert model: MoE Scaling</h3>
<p>Why did GPT-4, Gemini 1.5, Mixtral and DeepSeek all turn to MoE structures? Because Scaling Law showed up on MoE.<strong>It's called a bit of a bad efficiency.</strong>。</p>
<p>Traditional Dense model, parameters &#36;N&#36; It's a direct calculation. &#36;C&#36;(FLOPS) The bigger the model, the slower it goes.
MoE broke this tie, and it introduced two dimensions. &#36;N&#36;：</p>
<ul>
<li>&#36;N_{total}&#36;: Total parameter volume (decided on model)&quot;Knowledge capacity&quot;And memory.</li>
<li>&#36;N_{active}&#36;: Active parameter (calculating cost/velocity at the time of reasoning).</li>
</ul>
<p><em>Unified Scaling Laws for Routed Language Models</em> (DeepMind, Google) found that MoE's Loss dropped following a law that included two items:
&#36;&#36; L(N_{total}, N_{active}) \approx \frac{A}{(N_{active})^\alpha} + \frac{B}{(N_{total})^\beta} &#36;&#36;</p>
<p>There's one key here.<strong>Asymmetricality</strong>：
<strong>Costs of reasoning</strong>Main &#36;N_{active}&#36; Decision.</p>
<p><strong>Model performance</strong>But I can enjoy it at the same time. &#36;N_{total}&#36; The dividends (although marginal gains are lower than Dense, they are significant on a large scale).</p>
<p>This gives MoE a chance to be <strong>Paretto Frontier (Pareto front)</strong> Better than Dense model:<strong>MoE often mobilizes a larger total parameter capacity with equal reasoning.</strong></p>
<h3>Concluding remarks</h3>
<p>The laws of neuroscaling have written some of the patterns of experience in large model training into an estimateable engineering relationship. Kaplan emphasizes the size of the parameters, Chinchilla revises the focus to the calculus-data ratio;&quot;♪ Emergence ♪&quot;The data quality and MoE architecture were all problems that were revealed as the scaling line continued to advance. As Training Scaling’s marginal gains become more expensive, the follow-up route will continue to seek better quality and synthetic data, while also focusing more attention on the Inference Time Scaling.</p>
<h2>Bias-Variance No Trade-off：Neural Scaling Laws and Statistical Learning Theory</h2>
<p>The Neurotic Scaling Laws are important not only because they guide engineering estimates, but also because of the obvious tension in the intuition of the Statistic Learning Theory, SLT.</p>
<p>The traditional SLT tells us&quot;The model is too well developed.&quot;And making a better model requires controlling the complexity of the model, and Scaling Laws tells us that&quot;The bigger the better.&quot;I'm sorry. This tension comes from...<strong>Defeat and re-engineering in the age of deep learning</strong>。</p>
<h3>Classical Review: Traditional Perspectives of Dividence and Differences</h3>
<p>Before we go into the conflict in depth, we need to look at how the classical statistical learning theory views generalization errors.</p>
<p><strong>Offset-Equal Disaggregation (Bias-Variance Decomposition)</strong> It is a common tool for interpreting the generalization of learning algorithms. For a learning algorithm, its expectations on the test set are generalized error. &#36;E(f;D)&#36; It can be broken down into three sumes:</p>
<p>&#36;&#36;
E(f;D) = \text{Bias}^2(\boldsymbol{x}) + \text{Variance}(\boldsymbol{x}) + \text{Noise}
&#36;&#36;</p>
<ul>
<li><strong>Distortion (Bias)</strong>: The capability to develop models is measured. High deviations mean that models are too simple to capture the true pattern of data (see figure 1).<strong>Outstanding</strong>）。</li>
<li><strong>Difference (Variance)</strong>: Measurement of the sensitivity of the model to changes in training data. The difference means the model is too complex to remember the noise in the training data.<strong>Compromise</strong>）。</li>
<li><strong>Noise</strong>: The data itself is non-accuracy, and is a general sub-continuation.</li>
</ul>
<p><strong>Offset - Square Discretion (Bias-Variance Dilemma)</strong>:
This is the most painful trade-off in traditional theory.</p>
<ul>
<li><strong>When training is not enough</strong>: Bias leads, Varance low. The learning device is not sufficiently formulated and the disturbance of training data is not sufficient to make a significant difference in the learning device.</li>
<li><strong>When you're overtrained.</strong>: Bias low, Varance dominated. The learning device was so well designed for training data that it captured the noise in the data.</li>
</ul>
<p>This leads to famous. <strong>The U-type learning curve</strong>: The total error drops and rises as the complexity of the model increases. Therefore, training needs to stop at a middle point in search of a less-than-high “Sweet Point” (Trade-off) for Bias and Varance.</p>
<h3>Collision with traditional SLT: from uniform condensation to the paradox of scale</h3>
<p>In classic SLT, we tried to find one.&quot;Worst case scenario&quot;The guarantee. The broad error (Risk) is usually broken down into:
&#36;&#36; \text{Risk} = \text{Bias}^2 + \text{Variance} + \text{Noise} &#36;&#36;</p>
<p>Classic. <strong>VC Panorama (VC Generalization Bund)</strong> Tell us about the complexity of the model. &#36;h&#36;(approximate to parameter volume) &#36;N&#36;) Hypothetical space, generalization error &#36;R(f)&#36; Training error &#36;\hat{R}(f)&#36; There is a relationship between:
&#36;&#36; R(f) \leq \hat{R}(f) + \underbrace{\sqrt{\frac{h (\log(2n/h) + 1) + \log(1/\delta)}{n&#125;&#125;}_{\text{Complexity Penalty&#125;&#125; &#36;&#36;</p>
<p><strong>The prophecy of the classic theory:</strong> With the Arguments &#36;N&#36; Increase, training error &#36;\hat{R}&#36; It'll drop to zero, but in &#36;N &gt; n&#36; 时，复杂度惩罚项（包含 &#36;\sqrt{h/n} will tend to be endless. It's a direct result of the famous. <strong>U-type curve</strong> Inference: The total error must have been first down and then up.</p>
<p>But in 2017, Zhang et al. <em>Understanding deep learning requires rethinking generalization</em> A very powerful phenomenon was suggested. They found:<strong>Even if you put the label on the training set. &#36;y&#36; All randomly disrupted. Deep network still reaches. &#36;\hat{R}(f)=0&#36;</strong>I'm sorry. This means that the model is sufficiently big to remember pure noise, but it can still be generalized in real data; tradition is based on&quot;- We're all in.&quot;It is difficult to explain what is actually happening in depth learning.</p>
<h3>Escape the curse: a virtuous match</h3>
<p>To explain Scaling Law &#36;N \to \infty&#36; And then, Ross, he fell in one-way, and he was told, <strong>It's a good thing.</strong> Theory (Bartlett et al., 2020). This theory is trying to re-understand. <strong>Varance</strong> The behavior in the high space.</p>
<p>Amount of time &#36;d&#36; Much greater than the sample &#36;n&#36; And then the SSD will be hidden. <strong>&#36;\ell_2&#36; Minim Norm</strong> ♪ The world ♪
&#36;&#36; \hat{\theta} = \arg\min_{\theta} |\theta|_2 \quad \text{s.t.} \quad X\theta = y &#36;&#36;</p>
<p>With this setup, the breakdown of Risk has changed:</p>
<ul>
<li><p><strong>Bias single-hut drop:</strong> As the model becomes larger and the subspace cover becomes stronger, the model is better able to approach the real function.</p>
</li>
<li><p><strong>Varance disappears (not explodes):</strong> That is the most important part. Bartlett proves that as long as the data match the matrix to meet a specific spectrum decay,<strong>Noise energy will be used.&quot;Paint&quot;It spreads across countless extra dimensions. Go, go, go!</strong>。</p>
</li>
</ul>
<h3>Full mathematical image: double drop (Double Descent)</h3>
<p>Combining the theory, we got Scaling Laws complete. <strong>Double down.</strong> Image:</p>
<ul>
<li><p><strong>Arrear parameters range (&#36;N) &lt; n&#36;)：</strong> Observed by the classic VC-dimensional boundary, presenting the U-type curve. Bias falls, Varance rises.</p>
</li>
<li><p><strong>Critical Range (Classory Range)&#36;N \approx n&#36;)：</strong> &#36;XX^T&#36; The smallest feature value is close to 0, and the counter-format is extremely large, resulting in <strong>Risk explosion</strong>I'm sorry. This is the region most feared by traditional statistics.</p>
</li>
<li><p><strong>Good over integration (in %2)&#36;N \gg n&#36;)：</strong> Enter the Scaling Law field. Bias continues to decline because of dimensions &#36;d&#36; It's huge, the anti-formation is becoming good and the noise is spread to high-dimensional zero.<strong>Varance, instead of blowing up, is approaching zero.</strong>。</p>
</li>
</ul>
<p>Scaling Laws proves that, under the combination of deep nervous networks + SSD, we are on the right side of the double-down curve. Increase &#36;N&#36; And so, the fact that the two of us are able to reduce the Bias and Variance at the same time, and thus the Loss is showing a one-way drop in the law.</p>
<h3>Theory Reconstruction: Scaling Laws Revision of Math Intuitives</h3>
<p>Scaling Laws proves that under the combination of deep nervous networks + SSD, the government is not going to be able to use the Internet to control the situation.<strong>The differential gains (soft and integrated) from over-parametricization exceed the differential risks associated with it</strong>。</p>
<p>Let's compare the mathematical perspectives of the old and new ages:</p>
<table>
<thead>
<tr>
<th align="left">Feature</th>
<th align="left">Traditional Statistical Learning (SLT)</th>
<th align="left">Scaling Laws</th>
<th align="left">The mathematical nature difference.</th>
</tr>
</thead>
<tbody><tr>
<td align="left"><strong>Curve form</strong></td>
<td align="left">U-shape curve (U-shape)</td>
<td align="left">The law is falling.</td>
<td align="left">Mono-telephone vs non-uni-telephone</td>
</tr>
<tr>
<td align="left"><strong>Main contradictions</strong></td>
<td align="left">Vs variance</td>
<td align="left">Calculate resource allocation (allorcative Efficacy)</td>
<td align="left">Optimization issues vs statistical extrapolation issues</td>
</tr>
<tr>
<td align="left"><strong>Overparameter</strong></td>
<td align="left">Dangerous (over-sizing)</td>
<td align="left">Must.</td>
<td align="left">Benign Overfitting</td>
</tr>
<tr>
<td align="left"><strong>Error limit</strong></td>
<td align="left">Dependency &#36;\sqrt{N/D}&#36;</td>
<td align="left">Dependency &#36;N^{-\alpha} + D^{-\beta}&#36;</td>
<td align="left">Here. &#36;N&#36; It's a denominator.</td>
</tr>
</tbody></table>
<p>Formula for the Chinchila Law &#36;L = E + A/N^\alpha + B/D^\beta&#36; It actually rewritten the general error: it no longer contained the explosion. &#36;N/D&#36; item, but will &#36;N&#36; and &#36;D&#36; Considers two independent variables that have a positive contribution to reducing Ross. This is a major revision of the traditional statistical learning theory in the context of deep learning.</p>
<h2>United: Chartering Laws ' structure irrelevant and computational scaling down laws</h2>
<p>These two elements bring Scaling Laws from the empirical formula to the allocation of engineering resources and historical evolution.</p>
<p>If you say so.&quot;Smooth&quot;Scaling Law exists.<strong>Math Base</strong>Well, then...&quot;Calculate Zoom&quot;It's how you build a building.<strong>Construction drawings</strong>and&quot;Structure irrelevant&quot;And that explains why we finally chose Transformer.<strong>Construction materials</strong>。</p>
<h3>Compute scaling: Optimizing resource allocation issues</h3>
<p>In practical, large model training, we're not just concerned with...&quot;Parameters &#36;N&#36;&quot;or&quot;Data &#36;D&#36;&quot;In itself, it's... <strong>&quot;I have &#36;100 million in the budget. What am I supposed to do?&quot;</strong>  And then Scaling Law became a resource optimization issue.</p>
<p>Train a total calculation volume for a Transformer model &#36;C&#36;(Floating-point operations) can be approximated as:
&#36;&#36; C \approx 6 \cdot N \cdot D &#36;&#36;</p>
<ul>
<li>&#36;N&#36;: Model parameter volume.</li>
<li>&#36;D&#36;: Token number of training data.</li>
<li>&#36;6&#36;: Empirical factor (upward transmission of the approximation) &#36;2N&#36;Inverse transmission of the contract &#36;4N&#36;）。</li>
</ul>
<p>Optimizing geometric interpretation of the problem: Imagine a two-dimensional system:</p>
<ul>
<li>X-axis: calculated quantity &#36;C&#36;( logarithmic coordinates).</li>
<li>Y axis: Los (relative coordinates).</li>
</ul>
<p>If we fix the size of the model, &#36;N&#36;(e.g. 10B parameter) increasing data &#36;D&#36;And we'll get a curve. ♪ With &#36;D&#36; Add, and Los drops and slows down (limited to model capacity). If we draw curves of 1B, 10B, 100B models of different parameters, they're like a line down.</p>
<p><strong>The Scaling Law curve (Compute Frontier) is the lower condensed line of this curve.</strong></p>
<p>It means that it's in any given budget. &#36;C&#36; I'm not sure if you're going to be able to do this.<strong>It's inevitable.</strong>The best (N^)<em>, D^</em>) &#36;, which brings Los to the smallest global level.</p>
<p>There is a corresponding cost of departing from this optimal combination.<strong>Over-large model (over-sized)</strong> It means that if you use a huge model but with little data (e.g., Kaplan early proposal), you waste your calculus to multiply the matrix without giving the model sufficient information (Undertrained). The blogger adds:<strong>Sub-models</strong> It means that if you run through big data with a model of tiny (e.g., a few small models before LLAMA), the model is full and the return on continuing to read is low.</p>
<p>In the early stages of deep learning, bottlenecks are often the same.<strong>Algorithm</strong>"Don't know how to train deep webs" or "I don't know how to train deep webs"<strong>Data</strong>(no ImageNet).
But in the age of Scaling Law:</p>
<ul>
<li><strong>Algorithms are known:</strong> Transformer + SGD。</li>
<li><strong>Data adequacy:</strong> CommonCrawl contains the entire Internet.</li>
<li><strong>The only limit is &#36;C&#36;：</strong> Each unit of Los is falling, and needs an index increase in FLOPS.</li>
</ul>
<p><strong>The Chinchila law can be written as this optimised equation:</strong>
&#36;&#36; \min_{N,D} L(N,D) = \frac{A}{N^\alpha} + \frac{B}{D^\beta} \quad \text{s.t.} \quad C = 6ND &#36;&#36;
The best part of the job is... &#36;N \propto C^{0.5}, D \propto C^{0.5}&#36;I'm sorry. This means that under this scenario, the budget for computing will directly limit the amount of work to be done by models and the amount of data to be fed.</p>
<h3>Structure irrelevant: Why is Transformer the last winner?</h3>
<h4>Structure irrelevant (Universality)</h4>
<p>Scaling Law, a very useful observation:<strong>The trend of law enforcement is not just one structure.</strong></p>
<p>Research by OpenAI and DeepMind shows that it is not just Transformer, but:</p>
<ul>
<li><strong>LSTM / RNN</strong>(Circulatory Neurological Network)</li>
<li><strong>CNN</strong>(CROWD NRUNET)</li>
<li><strong>Linear Transformers</strong></li>
<li>Even some pure connections.</li>
</ul>
<p>Their Loss can be seen with the scale. &#36;L \propto C^{-\alpha}&#36; The curator form. This means that structures such as LSTM can also benefit theoretically from more calculus and parameters.<strong>The scaling up of the gains is not exclusive to Transformer.</strong></p>
<p>If you're following the law, why is LSTM dead?<strong>Theoretical performance caps vs. Engineering efficiency</strong>。</p>
<p>Although they all follow the formula of the law. &#36;L = a C^{-k}&#36;but coefficients &#36;a&#36;(transmitting) and &#36;k&#36;It's different, it's more important.<strong>Feasibility of practical training</strong>。</p>
<h4>Efficiency is everything: serial bottlenecks and gradients Stream</h4>
<p><strong>Serial vs. Parallel (The Wall-Clock Time Wall)</strong></p>
<p>This is the key to the Transformer victory.<strong>LSTM/RNN</strong> The Mathistic Form &#36;h_t = f(h_{t-1}, x_t)&#36; It was decided that it had to be calculated in a serial line: to calculate the status of 1000 token, it had to be calculated first, 999 before. This one. &#36;O(T)&#36; Time dependence is difficult to reconcile, and it leads to a very slow physical training to the same scale even if it is theoretically capable of Scaling.</p>
<p>The blogger adds:<strong>Transformer</strong> Attention mechanism &#36;Softmax(QK^T)V&#36; Allows a one-time parallel calculation of all token relationships. Although the complexity is... &#36;O(T^2)&#36;However, it is more likely to eat the array computing capacity of large GPU clusters, and the training efficiency is clearly superior to the current hardware.</p>
<p><strong>Gradient flow and long distance dependence</strong></p>
<p>Apart from calculating efficiency, gradient flows are important. Yes. <strong>LSTM</strong> , the gradient is to be transmitted back through time steps (BPTT) and is vulnerable to early information loss when dealing with extremely long sequences (e.g., a book), resulting in a Scaling curve being flater than Transformer. And... <strong>Transformer</strong> , any distance between either token is 1 (connected through the Attention), and a smoother gradient flows make it more efficient to use data in Scaling.</p>
<h4>Hardware Lottery</h4>
<p>Google Fellow Sara Hooker presented <strong>&quot;Hardware lottery&quot;</strong> Theory:
Scaling Law is not about Transformer being mathematically perfect, but about... <strong>Transformer is the most suitable structure for current GPU hardware (matrix multiplication accelerator)</strong>。</p>
<p>If we use some kind of brain chip, maybe Scaling Law wins LSTM or Spiking Natural Networks.</p>
<p>But the reality is, we have a GPU that's good at multiplication of dense arrays. Transformer's architecture features (high parallels, dense calculations) make it in this race of Scaling, and it is a very important way to get the world to the bottom of the world.<strong>Decline the speed of Loss in unit time (Loss per GPU-hour)</strong> Far beyond other structures.</p>
<h3>Wrap-up: The cruel truth about Scaling Law</h3>
<p>Together, we can draw a cruel conclusion about the modern AI development:</p>
<p><strong>The structure is no longer the core moat:</strong> If the architecture supports the gradient decline and has some expression, it can be scale. Transformer's victory is based on the fact that<strong>Engineering efficiency</strong>The victory (it can run Scaling Law as fast as possible).</p>
<p><strong>The calculation of intelligence:</strong> Under the optimal resource allocation strategy (Cinchiilla), the model ' s intellectual level (los) is almost entirely accounted for by input (in the form of a budget).&#36;C&#36;(b) Decision.</p>
<p><strong>Rules of the game:</strong> Deep learning becomes one.<strong>Transforming electricity and chips into intelligent industrial processes</strong>I'm sorry. Whoever runs more efficiently (higher parallels and higher hardware utilization) along the Scaling Law curve, is the winner.</p>
<p>That's why the AI lab is more like<strong>Systems Engineering</strong>Not the traditional algorithm lab. Competitiveness from&quot;Design a fine network structure&quot;Turn&quot;Building infrastructure to operate the Vanka cluster in a stable manner&quot;。</p>
<h2>The negative: GDDT&quot;Hard Cut&quot;With invalid Scaling</h2>
<p>This is a very real issue. Scaling Laws does not form directly in front of classic gradients such as the GGBDT, LightGBM, which helps us to clarify what conditions Scaling relies on.</p>
<p>The brief answer is:<strong>GDDT is still stuck in the U-type of classic statistical learning theory, and deep neural network (DNN) is passed&quot;It's a good thing.&quot;Run away.</strong></p>
<p>To understand why Scaling Law is only interested in deep learning, we need to go deep into the bottom of the Function Application.</p>
<h3>U-size curve vs. & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & &  and & &  and & & & </h3>
<p>If you keep increasing the number of trees in XGBoost (<code>n_estimators</code>Or deep, you'll see the textbook.<strong>Distortion - Square trade-off (Bias-Variance Trade-off)</strong>：</p>
<ul>
<li><strong>Initial:</strong> As the trees increase, the model captures data characteristics, Bias drops, test set Los decreases.</li>
<li><strong>Later:</strong> Once the threshold has been crossed, the model begins to compress noise. While the training set is close to zero, the test set is quick to bounce back.</li>
<li><strong>End of story:</strong> That's classic. <strong>U-type curve</strong>I'm sorry. GGBT relies heavily on Earthly Stopping to prevent a crash.</li>
</ul>
<p>LLM is in contrast&quot;Double down.&quot;Modern spaces. When the amount of data is sufficient (in line with the Chinchilla ratio), the increase in the parameter volume usually results in a continuous decline in Los. This one.&quot;Stacking arguments&quot;The dividends are more evident in in-depth learning.</p>
<h3>Division constant vs continuous flow: a general neutral war</h3>
<p>Why? The point is both mathematically. <strong>Summarize bias (Industive Bias)</strong>It's the model that's pre-set.&quot;Worldview&quot;Different.</p>
<h4>GDDT: Space Partitioning</h4>
<p>The tree model can be understood as cutting up the high space into a lot of irrelevants.<strong>Hyper-cubes</strong>。
&#36;&#36; f(x) = \sum c_m \cdot \mathbb{I}(x \in R_m) &#36;&#36;
It assumes that the world is made up of countless unrelated flat slices, and then gives a constant projection (sub-consistent function) in each block, which is expected to jump (StepFunction) at the regional boundary.</p>
<ul>
<li><strong>Deadback:</strong> Increasing parameters (deepening trees) is tantamount to cutting space even more. As the tree becomes deeper, the space is cut into powder, and each leaf node may contain only one sample.</li>
<li><strong>Non-slipble:</strong> For unseen areas (Gap), the model can only output rigid constants and cannot be inserted according to gradients. This one.<strong>Inconsistent, non-smooth</strong>The approach led to the direct explosion of the difference by the parameterization.</li>
</ul>
<h4>DNN: Fluid Soft Approach</h4>
<p>The neural network is constructed by a layered matrix multiplication and activation function.<strong>A smooth, smooth, continuous flow of height</strong>。
&#36;&#36; f(x) = \phi_L(\dots \phi_1(W_1 x)) &#36;&#36;
It assumes that the world is continuous and micro-synthetic.</p>
<ul>
<li><strong>Good idea:</strong> Even though the big models are capable,&quot;Remember.&quot;Noise, but with the driver of SSD, the model tends to constrict to <strong>Minim Norm</strong> The solution.</li>
<li><strong>Inhibited regularization:</strong> Math instincts tell us that SPD automatically chooses among the countless solutions that can be drawn.<strong>Minimum curve, smoothest</strong>That one. Adding parameters gives the model more freedom to draw more fine and smooth curves than to make sawn teeth.</li>
</ul>
<h3>Conclusion: Smoothness is a ticket to Scaling.</h3>
<p>Smoothness will have a direct impact on whether Scaling Law is effective. In high-dimensional space, data is extremely thin. Models must guess the value of the blank area (plug-in value) based on the training point. If the model is...&quot;Jump.&quot;, for example, for blank areas (high variance). If the model is...&quot;Smooth&quot;, (e.g. DNN) allows smoothing of the plugs using existing data points, thus maintaining low error in the blank area.</p>
<p>We can sum up by a visual analogy:</p>
<ul>
<li><strong>Scaling for GDDT</strong> It's like stacking blocks. To make a curve, you're going to pile it with smaller and smaller blocks. The smaller the wood, the more visible the sawn teeth on the edge, the more sensitive the deviation from position (oversizing).</li>
<li><strong>Scaling for DNN</strong> It's like pulling a rubber band. To draw up the data points, you add the elasticity of the rubber band (freedom). Under SSD tension, the rubber band not only crosses the data points, but also maintains a smooth transition in the blanks.</li>
</ul>
<p><strong>Scaling Law is very dependent on smoothness.</strong> GDDT lacks global smoothing, overparametrics lead to greater space debris decoupling; and DNN has global flow structure, which makes it easier to convert past parameters into real functions<strong>High Precision Smooth Plugin Value</strong>。</p>
<p>That's why GDDT is still the master of the tables (low, non-continuous), but on the road to AGI, their fate is very different:</p>
<p><strong>GBDT</strong>: Tree models are good at processing low-dimensional, dense, table-type data. However, in very high dimensions (e.g. text, pixel-level characteristics of images), the depth and number of trees required to cover the entire space are exploded exponentially. This increase in size does not bring about a generalization of capacity, but rather a convergence.</p>
<p><strong>DNN</strong>: The network of neurons (especially Transformer) is good at compressing high-dimensional thin data into low-dimensional currents through Embedding, and its characteristic combination capabilities are enhanced exponentially with depth and remain general.</p>
<p>Only deep nervous networks can break the dimension curse and set off on the Scaling road to AGI.</p>
<h2>Scaling: The nature of non-parametricism versus parametricization</h2>
<p><strong>Since tree models can increase the number of trees,<code>n_estimators</code>Isn't that a Scaling?</strong></p>
<p>The answer is:<strong>Yes, it's a Scaling, but it's closer.&quot;Non-parametric Scaling&quot;, with deep learning&quot;Parametric Scaling&quot;It's different in kinetics.</strong></p>
<p>We can try to establish this connection, but we have to see them.<strong>Maths</strong>Differences over. The following is a further summary of the discussions that took place earlier:</p>
<h3>Re-examine: Are the tree models really only U-shaped curves? (Connects Double Discent)</h3>
<p>The tree model mentioned earlier usually follows the U-shaped curve, but this is just one&quot;The conditional truth.&quot;I'm sorry. The latest research (e.g. the work of the Mikhail Belkin team) shows that the government is not willing to take the initiative to stop the violence.<strong>Under certain conditions, tree models can also show a single-tune drop or a scaling Law-style drop or&quot;Double down.&quot;The phenomenon.</strong></p>
<h4>Random Forest&quot;It's a good thing.&quot;</h4>
<p>Random forests are the traditional way to get closer to Scaling Law instincts.</p>
<p><strong>Mathistic form:</strong> RF is Bagging.
&#36;&#36; f_{RF}(x) = \frac{1}{M} \sum_{m=1}^M T_m(x) &#36;&#36;</p>
<p><strong>Scaling Behaviour:</strong> With the number of trees &#36;M \to \infty&#36;, the RF test error is usually<strong>Unargued</strong>and constrict to a constant (non-approximate error + model deviation). It hardly fits because of too many trees (U-style right side not to go up).</p>
<p><strong>Links to Deep Learning:</strong> It's like a deep study.<strong>Width Scaling</strong>I'm sorry. The hyper-widening neural network can also be understood as the convergence of many sub-networks from the perspective of Ensemble.</p>
<p><strong>Variance:</strong> RF's Los, the speed of decline usually follows statistics. &#36;1/\sqrt{M}&#36; or &#36;1/M&#36; It's a very fast harvest, but the ceiling is hard to break. And Deep Learning Scaling Law often crosses more orders.</p>
<h4>GDDT&quot;Double down.&quot;Possibilities</h4>
<p>For Boosting (e.g. XGBoost), the traditional view was that it was too much to say. But recent experiments have found that if ** completely uncut** and with very low learning (Shrinkage) &#36;\to 0&#36;Boosting can also observe a similar double decline.</p>
<p><strong>Explanation:</strong> When the tree is extremely deep (over parametrically) each tree is over-composed to some of the defects. But if the learning rate is low enough, this is a match.&quot;Slow&quot;And the next tree will be going on the noise in front.&quot;Align&quot;I'm sorry. This simulates in some way the iterative process of SSD.</p>
<h3>Core spread: two distinct&quot;Scaling Paradigm&quot;</h3>
<p>Even though they can be Scaling, tree models and nervous networks are climbing two different peaks. We need to understand the geometry of the difference between the efficiency of the two Scaling.</p>
<h4>Tree model 's Scaling (Tiling / Partitioning):</h4>
<p><strong>Operation:</strong> Tree models approach target functions by cutting space continuously (Axis-aligned splits).</p>
<p><strong>Complexity:</strong> It's ** Local**. To get close to a high-dimensional sphere, the tree model needs to cut out thousands of tiny squares to take the sphere.&quot;Spell&quot;Come out.</p>
<p><strong>Scaling Trouble:</strong> <strong>Curse of Demension</strong>I'm sorry. For each additional characteristic dimension, the number of trees (parameters) required to maintain the same near accuracy will need to be increased exponentially (in the case of the number of parameters).&#36;2^D&#36;)。</p>
<p><strong>Conclusions</strong> The Scaling of the tree model is extremely effective at low dimensions (tables) but very low at high levels (images/text).</p>
<h4>Neurosciente Scaling (Commonposition / Folding):</h4>
<p><strong>Operation:</strong> Neural networks distort space through linear transformation (rotation/extension) and non-linear activation (crash).</p>
<p><strong>Complexity:</strong> It's...<strong>Global</strong> and <strong>Composition</strong> Yeah. Deep-end networks do not need to cut space, but simply learn a function to learn the flow structure of data.</p>
<p><strong>Scaling Advantages:</strong> This one. <strong>Combiningability</strong> This allows the neural network to express the complex function of the index (as evidenced by Telgarsky et al. in 2016) by linearly increasing the amount of parameters.</p>
<p><strong>Conclusions</strong> The tree model. Scaling is...&quot;Add French&quot;The neuronet Scaling is...&quot;Multiplication/complex&quot;) (approach efficiency with depth enhancement).</p>
<h4>Memory vs. Understanding (Memoration vs. Internationalization)</h4>
<p>Back to what we discussed.<strong>Smooth</strong>。</p>
<p><strong>Tree extension:</strong> When we extend the GDDT to the extreme, it actually becomes one. <strong>k-Nearest Neighbor (kNN)</strong> Or check the tab. It remembers training data perfectly.</p>
<p><strong>On test:</strong> For the new sample, it's just going to check.&quot;Which leaves are closest to me?&quot;I'm sorry. The plug is...<strong>Non-slippin steps</strong>。</p>
<p><strong>Network expansion:</strong> When network parameters are sufficient, it is more likely to learn through the sample point under suitable optimization and normal conditions.<strong>Relative smooth flow</strong>。</p>
<p><strong>On test:</strong> For new samples, it is moving upstream in the flow.</p>
<p><strong>Conclusions</strong> Scaling Law really needs to explain, not just&quot;Bigger&quot;And it's also...&quot;Why do you keep a relatively smooth forecast structure when you get bigger?&quot;I'm sorry. When the tree model becomes larger (if not made more normal), space becomes easier to make.<strong>Broken</strong>。</p>
<h3>Scaling Law?</h3>
<p>What do you want to do if the tree model has the same Scaling capability as Transformer? This points to some of the integration directions of the current AI study.</p>
<h4>Soft tree / Neural tree (Differentiable / Soft Trees)</h4>
<p>If we put the tree in the model,&quot;Hard&quot;Yes. <code>if x &gt; 0.5 then left else right</code> I'm gonna...&quot;Soft&quot;Yes. <code>Sigmoid(x - 0.5)</code>(that is, to the left with this probability)</p>
<ul>
<li>And the tree becomes nuanced.</li>
<li>Trees become a special form of neural network (a complete layer of connections).</li>
<li><strong>Results:</strong> This one.&quot;Neural tree.&quot;You can use SSD training, and you can get closer to the Zooming Conditions of the Neural Network.</li>
<li><strong>A revelation:</strong> This suggests that the benefits associated with Scaling Law may depend not only on the name of the structure (tree vs web) but also on the name of the structure.<strong>Micro-diverse<strong>Bring it here.</strong>Gradient Optimization</strong>and<strong>Continuous flow pattern</strong>。</li>
</ul>
<h4>Missing Representation Learning</h4>
<p>Scaling Law's success on LLM is largely due to the fact that models grow with size and learn better. <strong>Embeding (identity)</strong>。</p>
<ul>
<li><strong>Tree model:</strong> Usually, it is directly divided on the original characteristic. It does not create new features, but only combines old ones.</li>
<li><strong>Reform:</strong> If we add a Transformer to the GDDT front to make the feature extraction, then then the GDDT to the header. So this whole system is compatible with Scaling Law. But at this point, it is thanks to Transformer, not to GGBDT.</li>
</ul>
