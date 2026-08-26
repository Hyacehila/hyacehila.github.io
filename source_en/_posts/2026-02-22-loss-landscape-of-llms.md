---
title: What Does the Loss Landscape of LLMs Look Like?
title_zh: 大模型的 Loss Landscape 是什么样的？
date: 2026-02-22 20:00:00 +0800
categories:
- Foundation Models
- Model Mechanics
tags:
- Fine-Tuning
- Alignment
- Model Mechanics
- Paper Notes
author: Hyacehila
mathjax: true
excerpt: Based on Unveiling the Basin-Like Loss Landscape in Large Language Models, this post explains basin-like loss landscapes
  and implications for fine-tuning and alignment.
description: Based on Unveiling the Basin-Like Loss Landscape in Large Language Models, this post explains basin-like loss
  landscapes and implications for fine-tuning and alignment.
excerpt_zh: 基于论文 Unveiling the Basin-Like Loss Landscape in Large Language Models，解读大模型 loss landscape 的 basin 现象及其对微调、对齐、越狱与预训练的启示。
permalink: /blog/2026/02/22/loss-landscape-of-llms/
lang: en
translation_key: 2026-02-22-loss-landscape-of-llms
translation_status: machine
translation_source_hash: e5bab0c04308661231f01d183d794002e0080959a27c80ade54c66c2c0a98fef
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<blockquote>
<p>tldr: Pre-training does not just push the model to a certain level of merit, but also creates a high-dimensional basin in parameter space that tolerates disturbances. The subsequent alignment, SFT and even escape attacks can be understood in this landscape.</p>
</blockquote>
<p>The questions in this article can also be addressed<a href="/en/blog/2026/01/26/neural-scaling-laws/">Neal Scaling Laws: From Kaplan to Chinchilla</a>、<a href="/en/blog/2026/02/20/compression-for-agi/">Compression for AGI: compression is intelligence</a>How the concept of a relatively close read together is developed in different contexts.</p>
<p>If you watch the big model after-training, you get a few problems that seem contradictory: The model is already strong, why does it suddenly decline in mathematical, reasoning or safety after a normal SFT? Why do models sometimes have to have a little bit of data to fight, like amnesia, to throw away their original abilities? And why can the model be directed to a hazardous output by prompt alone without changing the parameters at all?</p>
<p>This article is based on thesis. <a href="https://arxiv.org/abs/2505.17646v2">Unveiling the Basin-Like Loss Landscape in Large Language Models</a>, attempts to answer these questions with a unified geometry perspective: the larger model is seen as a point in the high-dimensional parameter space, and the modelling capacity is determined by the landscape of the space. You see, the question is not just “does it have a moving parameter” but also “how far along what direction”.</p>
<p>Also, I have added a section at the end of the text that quotes and responds to the blog Sun X published at 2026-03-10.<a href="https://chenxing-xuan.github.io/blog/2026/LossLandscape/">Second think about the big model, Los Landscape: the dynamic depth of gaming intelligence and universal intelligence</a>Further, ask whether the “public valley floor is equal to General Intelligence”, why multiple missions compress search space and how this extends to multiple models and Agent.</p>
<h2>The phenomenon: why the model suddenly forgets, escapes, falls in power Zoom</h2>
<p>The paper focuses on a very specific phenomenon:<strong>alignment brittleness</strong>That is, the vulnerability of alignment. It is mainly characterized by three types of issues:</p>
<ul>
<li>When normal fine-tuning is done, the old capabilities of the model can be accidentally compromised, such as loss of safety and loss of mathematical capability.</li>
<li>The counter-motion requires little data and little step to push the model to completely different modes of behaviour.</li>
<li>The Jailbreak in the input space looks like another kind of problem, but the results and parameter attacks are strikingly similar.</li>
</ul>
<p>If one can only understand “how good is the data” or “how high is the learning rate”, one can certainly explain some, but not all. Further questions are:<strong>Why are some directions updated almost without harm to models, while others are as steep as cliffs?</strong></p>
<p>Los Landscape gives the perspective that models move not in a flat space, but in a highly uneven landscape. Most directions may be smooth, but once they hit the worst, the capacity will collapse as quickly as it fell from the edge of the basin.</p>
<h2>Los Landscape: high-dimensional slices, random direction and benchmark</h2>
<p>The so-called "loss landscape" is visualization of "how changes in parameters affect model performance". The most basic formulation is:</p>
<p>&#36;&#36;
L(\alpha) = J_{\mathcal{D&#125;&#125;(\boldsymbol{\theta} + \alpha \boldsymbol{\delta})
&#36;&#36;</p>
<p>of which &#36;\boldsymbol{\theta}&#36; is the current model parameter,&#36;\boldsymbol{\delta}&#36; It's a direction vector,&#36;\alpha&#36; It's the long walk in this direction.&#36;J_{\mathcal{D&#125;&#125;&#36; It's in the data set. &#36;\mathcal{D}&#36; Benchmark loss.</p>
<p>Here's one thing:<strong>Thesis is not a training-based cross-entropy curve, but a result-based benchmark loss.</strong> The author will then consolidate the results of the assessments on the different tasks into a single scale and then draw a one-dimensional slice. So the vertical axis in the figure is closer to "Whether the power is preserved" rather than "Whether the probability of a given token is slightly fine-tuned." This is why the figure is very "in the literally sense of a basin" shape: In a large section, benchmark is almost completely unchanged; once out of bounds, performance suddenly deteriorates.</p>
<p>In this framework, the paper distinguishes between the two most important types of landscape:</p>
<ul>
<li><strong>Most-case landscape</strong>: To be cut in random directions to observe changes in “most directions”.</li>
<li><strong>Worst-case landscape</strong>: Take the initiative to find the most steep and easily disabling direction for the model.</li>
</ul>
<p>Why would a random direction be a point? The author's experience has found that for LLMs of sufficiently large, the curves obtained in different random directions are very similar, so that individual random slices can approximate the landscape of “most directions”. Then they use the Clipper-Pearson base to raise this empirical observation to statistically lower scales.</p>
<p>But there is one border condition:<strong>This basin conclusion relies on a production-based benchmark.</strong> Landscape tends to be smoother if you switch to more continuous indicators like log-likelihood. The basin here says not "the real loss of the nervous network is itself a box of squares," but "LLM will be a distinct basin of power from the point of view of whether or not the generating capacity is maintained."</p>
<h2>Most-Case Landscape: Why big models grow basin</h2>
<p>First, look at the most important picture in the paper. Here's Qwen2.5-7B in most-case direction Landscape, with a axis of unified benchmark loss, which is closer to 0, which means that the power is complete:</p>
<p><img src="/assets/images/loss-landscape-llm/qwen-most-case.png" alt="Qwen2.5-7B most-case landscape"></p>
<p>This map basically summarizes the main lines of the paper. The four curves of safety, math, basic, coding are close to 0 in the centre area for a long time, indicating that the model is almost “no drop” within the perturbation of this parameter; it can be quickly lifted to close to 1 once it continues to go out, meaning that the model-related capabilities are significantly degraded.</p>
<p>That's what the paper says. <strong>basin</strong>Not the smooth, round, and constantly changing bowl curves in the traditional small models, but a broad zone of stability. The author's instinct is that pre-training has pushed the model into a sufficiently large high-level stabilization subspace where small-scale moves are made, and most benchmarks are not going to break right away.</p>
<p>Another observation is that this type of basin is not “prevent”, but will emerge as the size of the model grows. Qwen:</p>
<p><img src="/assets/images/loss-landscape-llm/qwen-scale-0_5b.png" alt="Qwen2.5-0.5B most-case landscape"></p>
<p>This is more like our familiar little model Landscape: The center has a low valley, but the plateau is narrow, and the four curves are more like a continuously constricted V-shaped valley than a vast and stable basin.</p>
<p>And then look at 32B:</p>
<p><img src="/assets/images/loss-landscape-llm/qwen-scale-32b.png" alt="Qwen2.5-32B most-case landscape"></p>
<p>The landscape is very different. While it does not require full symmetry of the left and left, the low-damaged area in the middle is significantly wider and the model can maintain its original capacity in a wider parameter neighbourhood. One of the core observations of the paper was:<strong>The bigger the model, the bigger the basin, the more obvious and the wider it is.</strong></p>
<p>This explains a common empirical phenomenon: large models are usually more “resisting” than small ones. The “resistance” here is not that it is fine, but that it is updated in most random directions and within a certain range, and it is not easy to sacrifice its capacity immediately.</p>
<h2>Basic Basin and Capitalism Basin: What's changed for pre-training and alignment?</h2>
<p>If we continue to follow the above figure, the paper gives an inspiring statement:<strong>Pre-training first creates the Basic Basin, then then engraves the carving inside.</strong></p>
<p>The so-called basic basin is a stable area where the language that is the most basic of the models is understood, continued and dialogue. Once the model arrives through extensive pre-training, it has the minimum capability to work as a language model. The subsequent instructions were aligned, secure alignment, mathematical fine-tuning, code fine-tuning, not to transfer the model to another completely unrelated location, but more like building further narrow, more specialized sub-basins in some directions within the larger basin.</p>
<p>So many things are logical:</p>
<ul>
<li>If a certain ability corresponds to a large base, then the model is less likely to forget it when it is fine-tuned.</li>
<li>If a certain capacity is narrow, then a slight improvement in the direction of the capacity may be a first-time disruption.</li>
<li>The size of the base of the different model families varies, so the same data and hyperparameters can create completely different side effects in different base models.</li>
</ul>
<p>It also reminds us not to interpret the language as "covering a thin skin with pre-training." More appropriately, alignment is re-formed in the existing landscape. Some of the plastics are sufficiently large, so the new capacity is wide enough; others are not stable enough, so once SFT continues, the most vulnerable substructures will be wiped out.</p>
<h2>World-Case and SFT-Case: Why normal fine fine fine fine fine fine fine-tuning but can destroy a few steps</h2>
<p>If you look at most-case, you can easily draw an overly optimistic conclusion: since most directions are safe, why is there a reality of “ten data fine-tuning one model”? The answer is... <strong>worst-case direction</strong>。</p>
<p>First-case landscape:</p>
<p><img src="/assets/images/loss-landscape-llm/qwen-worst-case.png" alt="Qwen2.5-7B worst-case landscape"></p>
<p>This picture is almost like a needle. In addition to the extremely narrow central areas, the four capacity curves have almost instantaneously reached the high level of the loss. The conclusions it conveys are very straightforward:<strong>While most directions are good, some extremely bad directions do exist in parameter space, and the model quickly loses almost all its capabilities as long as there is a slight deviation.</strong></p>
<p>This provides a geometric explanation for “the high level of lethality of the small number of opposing data”. The counter-division is not going slowly in a random direction, but in the most negative direction. It is not about building up a large volume of updates, but about finding the export that is most likely to undermine existing capabilities.</p>
<p>The paper further drew the SFT direction. The author divides it into three scenarios: benign, normal and adversarial. In the first version of the text we keep the first two, because the original is basically a reconnect with the world-case.</p>
<p>See first, benign SFT, which is a finer finer and more moderate in the direction of the original training distribution:</p>
<p><img src="/assets/images/loss-landscape-llm/qwen-sft-benign.png" alt="Qwen2.5-7B benign SFT landscape"></p>
<p>This curve remains very much like most-casebasin: the meso-stabilized zone is wide enough to indicate that the fine-tune direction is at a high altitude in the original large basin. The paper uses official versions of Qwen2.5-7B-1M to construct this direction, which you can visualize as "continue to follow the direction that the bottom model is familiar with."</p>
<p>And then look at the normal SFT, which is a regular downstream fine tune with clear distribution differences but not malicious data:</p>
<p><img src="/assets/images/loss-landscape-llm/qwen-sft-normal.png" alt="Qwen2.5-7B normal SFT landscape"></p>
<p>The basin became significantly narrower. It's not as exaggerating as much as the world-case, but it's no longer as relaxed as beign SFT. This means:<strong>Normal fine-tuning is still within the controlable range, but its safety residual is much smaller than most-case.</strong> As long as data distribution is more volatile, learning rates more radical and training steps longer, it is more likely that capacity will be reduced.</p>
<p>So, a more precise understanding is that the SFT is not in a non-black-and-black-coated division of “security” and “danger”, but rather in a continuum. The closer the training distribution, the closer the fine-tune to the original model, the closer the moste-case; the closer the deviant, the more the target is drawn, the closer the world-case.</p>
<h2>Prompt Attack looks like Fine-Tune Attack</h2>
<p>The original text concludes with a question: why not change the parameters, optimize the input, and also cause similar damage and fine-tuning?</p>
<p>The difference between the two is not as large as it appears from the first layer of activation. Set embedding &#36;\boldsymbol{W}&#36;, the input is &#36;\boldsymbol{x}&#36;。</p>
<p>If we disturb parameters, the first level of output becomes:</p>
<p>&#36;&#36;
(\boldsymbol{W} + \Delta \boldsymbol{W})\boldsymbol{x} = \boldsymbol{W}\boldsymbol{x} + \Delta \boldsymbol{W}\boldsymbol{x}
&#36;&#36;</p>
<p>If we disturb the input, the first level of output becomes:</p>
<p>&#36;&#36;
\boldsymbol{W}(\boldsymbol{x} + \Delta \boldsymbol{x}) = \boldsymbol{W}\boldsymbol{x} + \boldsymbol{W}\Delta \boldsymbol{x}
&#36;&#36;</p>
<p>So the question turns to: can you find a input disturbance? &#36;\Delta \boldsymbol{x}&#36;♪ That makes ♪</p>
<p>&#36;&#36;
\boldsymbol{W}\Delta \boldsymbol{x} = \Delta \boldsymbol{W}\boldsymbol{x}
&#36;&#36;</p>
<p>So long as this can be done, the effects of the two attacks are the same in the first layer of activity space. The paper quoted the view that many LLM layers of embedding are now sufficiently “full” in the column space, so that the equivalent is geometrically achievable. The blogger adds:<strong>Prompt optimation can be considered a projection of parameter attack in input space.</strong></p>
<p>This certainly does not mean that “all jailbreaks are worth a fine-tuning exercise”, but it explains why they often display similar vulnerabilities: They're all trying to get the model out of the stable basin, except one in the parameter space and the other in the input space for equivalent disturbances.</p>
<h2>Basin's theoretical meaning: Clipper-Pearson, rannamozed moving and down the power line</h2>
<p>Here, basin is still an empirical observation. The paper went further: it tried to turn this geometry into a measurable and probative object.</p>
<p>First, the author defined a much more soft one. &#36;\sigma&#36;-Basin. Intuitively, if the model parameters are added to the standard deviation &#36;\sigma&#36; The Goss noise, the model's expectations are almost constant, which means it has a size. &#36;\sigma&#36; BASIN:</p>
<p>&#36;&#36;
J_{f,\mathcal{D&#125;&#125;(\boldsymbol{\theta}) - \mathbb{E}<em>{\boldsymbol{\epsilon} \sim \mathcal{N}(0, \sigma^2 \boldsymbol{I})}[J</em>{f,\mathcal{D&#125;&#125;(\boldsymbol{\theta}+\boldsymbol{\epsilon})] \leq \varepsilon
&#36;&#36;</p>
<p>The advantage of this definition is that it allows us to turn the “basin size” into a statistical object. Then there are two levels of theory.</p>
<p><strong>The first level, the Clipper-Pearson base points down the directional scale.</strong></p>
<p>The author will test in a large number of random directions: how many directions remain within the given radius. Because it is essentially a two-scale distribution success estimate, the Clopper-Pearson can give a strict confidence interval. So the phrase "most directions seem safe" is not just a visual image, but a "believing level." &#36;1-\gamma&#36; At least... &#36;p_{\text{lower&#125;&#125;&#36; The ratio is in the base condition.</p>
<p><strong>The second layer, the ranmomid smoothing turned the base size into a sub-performance line.</strong></p>
<p>The anticipatory sense given was that smoother benchmark changes would become more stable in relation to parameters once they were smoothed. Corresponds, the model is from &#36;oldsymbol {\theta}<em>0&#36; 走到 &#36;\boldsymbol{\theta}</em>After that, the performance drop can be restrained by the size of the base:</p>
<p>&#36;&#36;
\mathbb{E}<em>{\boldsymbol{\epsilon&#125;&#125;[J(\boldsymbol{\theta}</em>{\text{sft&#125;&#125;+\boldsymbol{\epsilon})]
\ge
\Phi\left(
\Phi^{-1}(\mathbb{E}_{\boldsymbol{\epsilon&#125;&#125;[J(\boldsymbol{\theta}<em>0+\boldsymbol{\epsilon})])-
\frac{\lVert \boldsymbol{\theta}</em>{\text{sft&#125;&#125;-\boldsymbol{\theta}_0 \rVert_2}{\sigma}
\right)
&#36;&#36;</p>
<p>No need to remember the formula.<strong>The same fine-tuned distance.&#36;\sigma&#36; The bigger the base, the wider the performance is.</strong></p>
<p>But restraint is also needed here. The paper itself acknowledges that the theoretically guaranteed “certified region” is usually much smaller than the empirically observed basin. In other words, the theoretical certificate is a conservative sub-level, not a “model that must be broken as long as it comes out of the theoretical safety zone”. Many of the conventional SFTs in reality still fall into experience, but are not fully covered by strong certificates.</p>
<p>Finally, emphasis is placed on border conditions: the basin discussed here is based mainly on the production of benchmark. If you switch to "likewood-based evaluation, loss landscape" you tend to re-slip the curve. This does not overturn the Basin perspective, but reminds us that the landscape you see is always about the way you choose to assess it.</p>
<h2>Basin can be inspired by the initiative: GO optimizer</h2>
<p>If basin means "more difficult to forget and less difficult to get hit in the worst direction," then there is naturally a question of follow-up:<strong>Can the basin be expanded?</strong></p>
<p>The paper gives a directional answer that can be tried. They introduced a Gaussian-augmented Optimizer (GO optimizer), which was trained to optimize not only single-point parameters but rather expectations in the adjacent area of the parameters:</p>
<p>&#36;&#36;
L_{\text{train&#125;&#125;(\boldsymbol{x}, \boldsymbol{\theta}) = -\mathbb{E}_{\boldsymbol{\epsilon} \sim \mathcal{N}(0, \sigma^2\boldsymbol{I})}[\log p(\boldsymbol{x} \mid \boldsymbol{\theta}+\boldsymbol{\epsilon})]
&#36;&#36;</p>
<p>Intuitively, it is a small disturbance in the adaptation of the model to the parameter neighbourhood, thus optimizing single points into a relatively good decomposition of a neighbourhood.</p>
<p>Here's a map of GPT2-127M, compared to Landscape, which was pretrained on OpenWebText, with the red line GO, the green line AdamW:</p>
<p><img src="/assets/images/loss-landscape-llm/go-pretrain-landscape.png" alt="GO optimizer pretraining landscape"></p>
<p>As you can see in the figure, the Go curve is evener and wider, which means that it learns not a sharp point, but a more stable region.</p>
<p>This wider basin also shows the benefits in the subsequent fine-tuning. The following is a comparison of the performance of the subsequent training on Alpaca: the left axis is NLL for the old capability OpenWebText and the right axis is NLL for the new capability Alpaca.</p>
<p><img src="/assets/images/loss-landscape-llm/go-pretrain-benchmark.png" alt="GO optimizer benchmark comparison"></p>
<p>Not absolute values, but trends: the model that GO pre-trained is not learning much slower when new assignments are being taught, but the old capacity is falling even less. This corresponds to the main thesis line:<strong>If you're going to be a little broad in pre-training, then the chances of a catastrophic memory of SFT follow-up are even lower.</strong></p>
<p>Of course, this part is far from being an industrial conclusion. The paper was validated only on smaller models and was more like a clear research inspiration than a direct recipe to all production systems. But it did bring the matter of “reduce the forget”, from the question of post-training techniques to the question of pre-training geometry.</p>
<p>To bind the whole article in three words, it's probably:</p>
<ul>
<li>The parameter space of the large model is not even and pre-training will shape a huge basin.</li>
<li>Alignment and conventional fine-tuning are usually shaped inside the base, while anti-data can lead the model in the worst direction.</li>
<li>If the model is to be more stable, the focus is not only “a little less than a parameter”, but also “can it be possible to make the space around the model more securely mobile”.</li>
</ul>
<h2>Supplement: Second reflections on the dynamics of the public valley to multitasking</h2>
<p>The following section is an additional one. I quote and respond to the blog Sun X published at 2026-03-10.<a href="https://chenxing-xuan.github.io/blog/2026/LossLandscape/">Second think about the big model, Los Landscape: the dynamic depth of gaming intelligence and universal intelligence</a>I'm sorry. The following are mainly re-transposing and extending the blog ' s views, not representing the direct claims of the original author; the discussion of multi-mission dynamics, the Hessian neighbourhood size and the data mix is more appropriate as an inspiration than as a conclusion that the original paper has strictly proved.</p>
<h3>Is the public valley floor equal to universal intelligence?</h3>
<p>The most valuable thing about this blog is that it does not directly equate "sharing the bottom of the valley on the proximity parameters of different tasks" with "models have found some sort of pure universal intelligence kernel." Bloggers are very vigilant about this optimistic interpretation: If the public valley really means that there is a “pure public knowledge base” that can be independently obtained from multi-mission training, then, by this logic, we would seem to need only a few pre-trainings, leaving the remaining tasks to be activated. But this is not the case, and the large models of good performance almost all rely on large-scale, heterogeneity, cross-cutting data mix pre-training.</p>
<p>Linking this suspicion to the Basin narrative, I prefer to interpret the "public valley floor" as one.<strong>A stable compromise area under multi-mandate constraints</strong>Instead of a single mandate, there is ample evidence that everything can be taken from it. The original paper showed us that different capabilities can be maintained at the same time in the nearest region at the low of benchmark loss; but it suggests at best that the model found a set of shared signs and robust parameterizations in the region, which is not sufficient to introduce “a purely public knowledge that is not related to multitasking language material”.</p>
<p>If we continue to follow this perspective, it is easier to understand why the pre-training phase needs to be as rich, heterodox and broad as possible: not because the model has a purely generic knowledge core before being inserted into another mission; but because<strong>A sufficient number of tasks and data distributions are part of the plasticization that makes it more likely to form a stable public area.</strong>。</p>
<h3>Why would multitask reduce the exploration dimension of search parameters space?</h3>
<p>The instinct of blogs is that multi-mission training is an important function, not only of learning a few more things about models, but also of learning about them.<strong>Reduce the dimension of exploration in search parameters space</strong>I'm sorry. This is a very natural statement, if it is taken from the geometry of the parametric model.</p>
<p>For a large, highly parametric model, single-mission training is usually not just one of the best, but a whole low-loss stream. Intuitively, the model has a large amount of parameters that can compensate for each other: You change the parameters in some directions, the loss of task A is almost unchanged, so the optimizer can "show" in this wide flat area. Geometrically, these directions are the cutting space for low-floating single-tasks and the freedom that is most easily wasted during model searches.</p>
<p>Once mission B, mission C folds in, things change. Each mission defines its own low-loss flow, and multi-task learning seeks the intersection of these streams. Once multiple high-dimensional low-intensity areas are turned over, the direction that can be preserved at the same time will be significantly reduced; many of the “free-sliding” that was previously harmless only for mission A, which appears to be steeply upward, and they will be erased directly by new gradient signals.</p>
<p>So I'll read "reduce the dimension of exploration" as follows:<strong>Reducing the degree of ineffectiveness rather than turning the optimisation problem magically into a low-dimensional linear problem.</strong> From the point of view of optimising dynamics, search no longer occurs on a vast plain that can run around, but more like entering a narrower and more restricted public passage.</p>
<h3>The mathematical instinct of the curve, the Hessian and the optimal neighbourhood size.</h3>
<p>The most critical mathematical instinct in the blog comes from the inclusion of the total loss of multiple tasks as a weight:</p>
<p>&#36;&#36;
\mathcal L(\theta) = \sum_{i=1}^m \alpha_i \mathcal L_i(\theta)
&#36;&#36;</p>
<p>If it's the best part of the world, &#36;\theta^*&#36; The next step is to do a second-order approximation of individual tasks.</p>
<p>&#36;&#36;
\mathcal L_i(\theta) \approx \mathcal L_i(\theta^<em>) + \frac{1}{2}(\theta-\theta^</em>)^\top H_i(\theta-\theta^*)
&#36;&#36;</p>
<p>Here. &#36;H_i&#36; It's the first. &#36;i&#36; Hessian, a mission, it's painted the loss local curvature in all directions. So, after supersing the multi-mission losses together, the model is allowed to remain "no more than error" &#36;\epsilon&#36;"the best neighbourhood, which can be written as:</p>
<p>&#36;&#36;
\delta(\epsilon)=\left{\theta:(\theta-\theta^<em>)^\top\left(\sum_i \alpha_i H_i\right)(\theta-\theta^</em>)\le 2\epsilon\right}
&#36;&#36;</p>
<p>This collection can be seen as a super ellipse decided by the General Hessian. If you remember</p>
<p>&#36;&#36;
H_{\text{total&#125;&#125; = \sum_i \alpha_i H_i = Q\Lambda Q^\top
&#36;&#36;</p>
<p>of which &#36;\Lambda = \mathrm{diag}(\lambda_1,\dots,\lambda_k)&#36; It's a characteristic value in the curvature space, so in the main axis coordinates, the upper bounds become</p>
<p>&#36;&#36;
\sum_{j=1}^k \lambda_j z_j^2 \le 2\epsilon
&#36;&#36;</p>
<p>So the radius of each main axis is...</p>
<p>&#36;&#36;
r_j = \sqrt{\frac{2\epsilon}{\lambda_j&#125;&#125;
&#36;&#36;</p>
<p>That is, the larger the characteristic value in a given direction, the smaller the radius of parameters that can be tolerated in this direction. Further, in the chorus, the size of the super ellipse is satisfied.</p>
<p>&#36;&#36;
V(\epsilon) \propto \prod_{j=1}^k \sqrt{\frac{2\epsilon}{\lambda_j&#125;&#125; = \frac{(2\epsilon)^{k/2&#125;&#125;{\sqrt{\prod_{j=1}^k \lambda_j&#125;&#125;
&#36;&#36;</p>
<p>This is the mathematical intuitive source of blogs that say "multitasks will compress the best neighbourhood size": when Hessian is almost independent and not exactly in the same direction as he is bound by multiple tasks, he is not the only one who is the best in the world to be able to use his own tools.&#36;H_{\text{total&#125;&#125;&#36; The active feature values will be raised as a whole and many of the flat directions that were close to zero will be activated by additional constraints. As a result, the viable neighbourhood that meets the same tolerance for error is significantly reduced.</p>
<p>Of course, it must be clear here: It's a...<strong>Understand the math instincts of blogging.</strong>, not the complete conclusion that has been rigorously proved in the present paper. The practical training is not only about the near-defunct of the local second stage, but also about the complexity of non-compression, re-alignment of parameters, and non-exchangeability of different tasks. So it's more like a approximation that helps us understand why multiple missions compress ineffective freedom than a theory that can be mechanically applied.</p>
<h3>Is a multi-module considered a stronger multi-task?</h3>
<p>My judgment is:<strong>It can, and usually can, be seen as a more binding, more isomeric multitask.</strong> For VLM or more generic multi-modular models, a simple joint goal can be written as:</p>
<p>&#36;&#36;
\mathcal L_{\text{total&#125;&#125; = \alpha \mathcal L_{\text{text&#125;&#125; + \beta \mathcal L_{\text{vision&#125;&#125; + \gamma \mathcal L_{\text{align&#125;&#125;
&#36;&#36;</p>
<p>These include both text-modeling losses and visual modelling losses, as well as cross-modular alignment losses. These constraints are much more isomeric than the multiple tasks in the text: the text is a discrete symbol structure, the visual is a continuous, dense signal, and cross-modular alignment requires that both are mapd in shared semantic spaces.</p>
<p>From the Hessian perspective, the multimodules are more binding than “more tasks”, but rather because the different mosaics are cutting more closely to the cross-section of parameter space. Some direction is almost zero-costed in pure text missions, and in visual missions it may immediately become high curvature; while some of the redundant direction of visuality may be re-locked by language alignment losses. So the overall Hessian active feature values are more likely to be raised as a whole, with the original large numbers of nulk-space and near-zero curvatures being cut, and the optimal neighbourhood size to meet the same error threshold is more likely to continue to contract.</p>
<p>But I'd like to add a conservative sentence here:<strong>Smaller neighbourhoods do not automatically mean that training is easier.</strong> Multiple models bring not only greater restraint, but also greater optimization of instability, model conflicts and training rigidity. It is true that it may be more conducive to the development of cross-model public concepts, but it is equally likely to drag models into more optimised states if the loss matching, sampling sequence, architecture interfaces and alignment of targets are sufficiently rational.</p>
<p><strong>The value of isomer feedback may lie not only in providing more information to the model but also in remodeling the los landscape, the ineffective, speculative and cheating exploration channels in compressed policy space.</strong> If this judgement is generally true, the future is better in terms of data mixing data by knowledge coverage or geometrically combined data — that is, which tasks, which models, which feedback should come together to move the model more effectively to the public flat valley floor, which is difficult to cheat and which is migratory.</p>
<h2>References</h2>
<ul>
<li>Huanran Chen et al., <em>Unveiling the Basin-Like Loss Landscape in Large Language Models</em>. <a href="https://arxiv.org/abs/2505.17646v2">arXiv Summary Page</a></li>
<li>Sun X, <em>Second think about the big model, Los Landscape: the dynamic depth of gaming intelligence and universal intelligence</em>. <a href="https://chenxing-xuan.github.io/blog/2026/LossLandscape/">Original Link</a> （2026-03-10）</li>
</ul>
