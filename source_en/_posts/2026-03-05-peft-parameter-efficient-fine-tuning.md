---
title: 'Parameter-Efficient Fine-Tuning (PEFT): From Adapter to LoRA'
title_zh: 参数高效微调（PEFT）：从 Adapter 到 LoRA 的技术演进
date: 2026-03-05 13:00:00 +0800
categories:
- Foundation Models
- Training & Alignment
tags:
- LoRA
- Fine-Tuning
- Survey
author: Hyacehila
mathjax: true
hidden: true
excerpt: A survey of PEFT methods including Adapter, Prefix-Tuning, LoRA, Prompt Tuning, P-Tuning v2, and AdaLoRA, with their
  design logic and use cases.
description: A survey of PEFT methods including Adapter, Prefix-Tuning, LoRA, Prompt Tuning, P-Tuning v2, and AdaLoRA, with
  their design logic and use cases.
excerpt_zh: 梳理参数高效微调（PEFT）领域的代表性方法——从 Adapter、Prefix-Tuning 到 LoRA、Prompt Tuning、P-Tuning v2 与 AdaLoRA，理解不同技术路线的设计思路与适用场景。
permalink: /blog/2026/03/05/peft-parameter-efficient-fine-tuning/
lang: en
translation_key: 2026-03-05-peft-parameter-efficient-fine-tuning
translation_status: machine
translation_source_hash: a49ae18668129324c7fd5c4b98a71dd8335efdd39d77654530a590b7f90d8f01
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<h2>Why do you need to fine-tune the parameters efficiently?</h2>
<p>The questions in this article can also be addressed<a href="/en/blog/2025/12/27/Re0HF-01/">Re0-01 : HuggingFace Transformers Trainer</a>、<a href="/en/blog/2024/11/01/llm-post-training-and-finetuning/">Post-linguistic training and fine-tuning of practice: from SFT, LoRA to human alignment</a>How the concept of a relatively close read together is developed in different contexts.</p>
<p>Full Fine-Tuning is usually a good effect of large language models, but computing and storing costs will increase rapidly with model size. To fit downstream missions with limited resources, researchers proposed the high-efficiency fine-tuning of parameters (Parameter-Efficient Fine-Tuning, PEFT) technology: freeze most pre-training parameters, train only a few newly introduced or re-parametric modules and use lower-cost effects to approach full fine-tuning.</p>
<p>This paper summarizes the design ideas and technical features of several representative methods in the PEFT field.</p>
<h2>Adapter: Insert training modules into layers</h2>
<blockquote>
<p>Houlsby et al., <em>Parameter-Efficient Transfer Learning for NLP</em>, ICML 2019.</p>
</blockquote>
<p>Adapter can be an early representation in the PEFT field. At this stage, although large-scale pre-training models have emerged, causal language models have not yet taken precedence and many downstream tasks still need to be fine-tuned. Adapter was raised in this context to reduce the cost of fine-tuning.</p>
<p>Adapter has inserted a training structure between the model layers. In the author 's experiment, Adapter was inserted after the FNN of the coding device, before the Layer Norm, and the internal FFN itself contained non-linear activation functions. But in the actual scene, Adapter is inserted in a more flexible position, with no defined norm.</p>
<p>The authors consider that the main advantages of Adapter include:</p>
<ul>
<li><strong>Light Quantification and High Performance</strong>: fine-tuned only a smaller number of parameters (to a full measure), but very good.</li>
<li><strong>Model reuse</strong>: the main model only needs one copy, without having to copy the full amount of parameters for each task</li>
<li><strong>As a means of fine-tuning</strong>: no full training required on new assignments, consistent with the pre-training + fine-tuning paradigm</li>
</ul>
<h2>Prefix-Tuning: Optimizing Continuous Prefixes</h2>
<blockquote>
<p>Li and Liang, <em>Prefix-Tuning: Optimizing Continuous Prompts for Generation</em>, ACL 2021.</p>
</blockquote>
<p>In order to obtain better off-the-road mission effects, we would like to fine-tune the pre-training model. The introduction of additional parameter adapters, in addition to the direct freezing of the original parameter layers, is also discussed in detail by scholars, given the high full fine-tuning costs. Prefix-Tuning is one such lightweight fine-tuning.</p>
<p>The idea of prefix fine-tuning comes from context learning (In-Context Learning). A small number of hints (token pre-input) can be used to acquire field performance, so the author considers adding prefixes to the token.<strong>Prefix is a continuum that needs to be optimized</strong>, during Prefix-Tuning, the main parameter layer of the model is frozen, and the replacement of Prefix is sufficient to allow the model to be converted.</p>
<p>All Encoder layers were added to Prefix, so the prefix actually formed a matrix, the total number of parameters, and the number of prefixes was the same. &#36;\text{length}(\text{prefix}) \times \text{num_layers}&#36;I'm sorry. Prefixes are left-hand to ensure that all Mask-attents have sufficient information.</p>
<p><strong>This technology is also applied to NLG and NLU tasks, i.e., can be used to fine-tune the BERT structure and class GPT structure.</strong></p>
<p>The advantages the author believes are:</p>
<ul>
<li><strong>LightQuantification</strong>: fine-tune only a smaller number of parameters (as compared to the insertion of Adapter Layer)</li>
<li><strong>Model integrity</strong>: Without changing the model structure, only one copy is required for the core model</li>
<li><strong>Application level</strong>: Modeling is feasible because of its lightness and it also benefits privacy protection</li>
</ul>
<h2>LoRA: Low-fitness and grace</h2>
<blockquote>
<p>Hu et al., <em>LoRA: Low-Rank Adaptation of Large Language Models</em>, ICLR 2022.</p>
</blockquote>
<p>LoRA's idea is derived from the low-level breakdown of the matrix, which is based on the belief that changes to the parameter matrix can actually be summarized by a low-level matrix, and therefore uses a small learning matrix instead of the original full-parameter modification.</p>
<p>The main advantages identified by the authors include:</p>
<ul>
<li><strong>Storage space advantage</strong>: original parameters frozen completely, reducing training visible consumption and expenditure on model storage and task switching</li>
<li><strong>Calculate efficiency</strong>: significant improvement in computing efficiency due to the need to optimize only the new low-caliber matrix</li>
<li><strong>Zero reasoning delay</strong>: The low-swirl matrix can be directly merged into the original parameter without any delay in reasoning as a result of the new matrix</li>
<li><strong>Flexible combinations</strong>: It can be used in combination with multiple methods</li>
<li><strong>Full fine-tuning of the generalization</strong>: Increase low tumble &#36;r&#36; The size of the size will be nearly fully fined.</li>
</ul>
<h3>Comparison of other programmes</h3>
<ul>
<li>Adapter's strategy, although it has rarely increased, still produces Inference Latecy</li>
<li>Prefix-Tuning prefixes are hard to learn, performance optimization is low</li>
</ul>
<h3>The lab found out.</h3>
<ul>
<li>QKVO's four attention matrices are better all optimized. Even if overall optimization takes a low runk, the four matrices are broken down by lower runks over only one of the runks.</li>
<li>It doesn't need a high-level runk to achieve enough performance in the experiments that the author considers.&#36;r=8&#36; and &#36;r=64&#36; The performances do not differ significantly, because they share a dimension of the subspace.</li>
<li>The mechanism for the central role of the Low Spectrofit Matrix - it may be enhanced by enhancing the pre-training model<strong>Learning but not focused</strong>and thereby effectively match the needs of specific downstream missions</li>
</ul>
<p><strong>LoRA and its derived fine-tuning techniques based on parameter freezing and matrix decomposition are among the most important SFT technologies at present.</strong></p>
<h2>Prompt Tuning: A hint from dispersing to continuous</h2>
<blockquote>
<p>Lester et al., <em>The Power of Scale for Parameter-Efficient Prompt Tuning</em>, EMNLP 2021.</p>
</blockquote>
<p>Prompt Tuning freezes all original model parameters and adds learning-able soft prompts to the new Prompt. The idea is to make models fit for specific tasks by adding soft prompts to Prompt. Prompt Tunning can be considered a special form of Prefix-Tunning.</p>
<p>Prompt Tunning's idea comes from Prompt Engineering. The idea of using fine-tuning model parameters to replace artificial tips has emerged as a result of the need for more manual involvement in fine-tuning of the phrases. Only the additional learning soft programt is trained from end to end to enrich information about the type of task.</p>
<p>Prompt Tuning, compared to manual hints, is generally known as a<strong>Continuous prompt technology</strong>, because the hints are continuous throughout the embedded space, the method of the artificial hints is discrete in the embedded space and therefore generally perform better than the artificial hints.</p>
<h3>Valuable conclusions</h3>
<ul>
<li><strong>As the size of the model grows, the gap between Prompt Tuning and the whole model fine-tuning is gradually narrowing.</strong>And better than artificially designed tips.</li>
<li>The prefix token should not be too short, too long, and it would be more appropriate to control at dozens.</li>
<li>Less effective than Prefix-Tuning, which leads to subsequent improvements in P-Tuning v2</li>
<li>Prompt Tuning may be more robust because of the few changes to the structure of the original parameter, which is distributed at different stages of the training and assessment (i.e., field deviation)</li>
<li><strong>Freezing of common language understanding parameters and limiting downstream learning to lightweight parameters would help avoid over-representation of specific areas - Yeah.</strong></li>
</ul>
<p>Prompt Tunning is much more valuable in thought than it is in application.<strong>At the same time, this technology is generally used only for NLU tasks and not for the type GPT structures that are now in the mainstream.</strong></p>
<h2>P-Tuning v2: Deep tip return</h2>
<blockquote>
<p>Liu et al., <em>P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks</em>, ACL 2022.</p>
</blockquote>
<p>The main contribution of this paper is an innovative empirical finding:<strong>Prompt Tuning</strong> In the context of the various model scales and natural language understanding (NLU) missions,<strong>Performance can be comparable to full-parameter fine-tuning</strong>。</p>
<p>P-Tuning v2 has abandoned the option of adding soft programt only to the embedded layer, but has added learning prefixes to the multiple Encoder layers. P-Tuning v2 and Prefix-Tuning are close, but Prefix-Tuning's high-dimensional prefix is obtained by a lower-dimensional weight parameter from an MLP layer, and P-Tuning v2 directly optimizes the high-dimensional vector.</p>
<p><strong>P-Tuning v2</strong> Clear application<strong>Standard Header</strong>, which can very naturally support:</p>
<ul>
<li>Text Classification</li>
<li>Named Entity Identification (NER)</li>
<li>Semantic Role Description (SRL)</li>
</ul>
<p>. The task of NLU is covered. Compared to the LoRA, which is suitable for natural language generation tasks, it requires a variety of tips to activate its capabilities in other NLU missions.<strong>P-Tuning v2 is fit for this type of task</strong>。</p>
<h3>P-Tuning v2 strengths and discoveries</h3>
<ul>
<li><strong>For small language models</strong>: The current fine-tuning focuses on the study of the production language model of 10B or more, while P-Tuning v2 has a good effect on the fact that it does not require so many parameters in the NLU field</li>
<li><strong>Value of Depth Tips</strong>: The original Prompt scheme is still inadequate here</li>
<li>P-Tuning v2 is generally comparable to a whole fine tune for all tasks Beautiful.</li>
</ul>
<p><strong>As with Prompt Tunning, P-Tuning v2 is also a structure that applies to a similar BERT. The large cause-effect language model (GPT) has had a very strong effect on NLU missions, and the real application value of considering using these fine-tuning techniques is questionable.</strong></p>
<h2>Adalora: Self-adaptation distribution</h2>
<blockquote>
<p>Zhang et al., <em>AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning</em>, ICML 2023.</p>
</blockquote>
<p>This is an improvement on the LoRA method. Unlike the visual modifications of QLora, LoRA+ (quantitative original parameters and modified learning rates for the LoRA part), Adalora wishes to introduce an Adaptive LoRA Rank, which reduces the low-value portion of LoRA, thereby increasing fine-tuning efficiency. For the importance of the distinction, Adalora was achieved using the oddly foreign decomposition (SVD).</p>
<h3>SVD Adaptation</h3>
<p>First, LoRA decomposition based on SVD, which converts the original simple matrix decomposition into SVD form:</p>
<p>&#36;&#36;
W = W^{(0)} + \Delta = W^{(0)} + P \Lambda Q
&#36;&#36;</p>
<p>And use &#36;\mathcal{G}<em>i = { P</em>{<em>i}, \lambda_i, Q_{i</em>} That indicates a strange value against a strange direction, where the odd value is used to initialize, the vector is to initialize with Gaussian and add a positive-soft constraint:</p>
<p>&#36;&#36;
R(P, Q) = | P^\top P - I |_F^2 + | Q Q^\top - I |_F^2
&#36;&#36;</p>
<p>All these operations are designed to ensure that normal processes are optimized when making the rank adaptation and reduce the calculation costs of the high-dimensional matrix for SVD breakdown.</p>
<p>We will apply the SVD-based appliance to each ownership matrix in the Transformer layer, including &#36;W_q, W_k, W_v, W_{f_1}&#36; and &#36;W_{f_2}&#36;I'm sorry. In order to control the budget of the parameters, the branches are cut in order to be divided over time during the training process, based on the importance of the odd values.</p>
<h3>Importance-based Rank Allocation</h3>
<p>Use &#36;k&#36; Indexing incremental matrix &#36;\Delta_k = P_k \Lambda_k Q_k&#36;（&#36;k = 1, \dots, n&#36;) , will &#36;\Delta_k&#36; No. No. &#36;i&#36; Three-dollar group to &#36;matcal{G}<em>{k,i} = {P</em>{<em>,i}, \lambda_{k,i}, Q_{i,</em>&#125;&#125;&#36;，其重要性得分为 &#36;S_{k,i}&#36;。</p>
<p>The training target function for adding regular items is:</p>
<p>&#36;&#36;
\mathcal{L}(\mathcal{P}, \mathcal{E}, \mathcal{Q}) = \mathcal{C}(\mathcal{P}, \mathcal{E}, \mathcal{Q}) + \gamma \sum_{k=1}^n R(P_k, Q_k)
&#36;&#36;</p>
<p>At the end of the day &#36;t&#36; Step, first to implement a random gradient step to update the parameters:</p>
<p>&#36;&#36;
\tilde{\Lambda}<em>k^{(t)} = \Lambda_k^{(t)} - \eta \nabla</em>{\Lambda_k} \mathcal{L}(\mathcal{P}^{(t)}, \mathcal{E}^{(t)}, \mathcal{Q}^{(t)})
&#36;&#36;</p>
<p>Then, you score the importance. &#36;S_k^{t}&#36;, the odd value is cut as follows:</p>
<p>&#36;&#36;
\mathcal{T}(\tilde{\Lambda}<em>k^{(t)}, S_k^{(t)})</em>{ii} =
\begin{cases}
\tilde{\Lambda}<em>{k,ii}^{(t)} &amp; \text{if}S</em>\text{ before \t\text{}b^t}\text{name},\
Photo by Flickr user @un.org &amp; \\text{Other circumstances}
The next thing I know, I'm not sure.
I'm sorry.</p>
<p>of which &#36;S^{(t)}&#36; The first one was a three-dollar score, which was a three-dollar score.&#36;b^{(t)}&#36; No. No. &#36;t&#36; The remaining odd budget. In this way, cuttings are less important and leave more budgets to higher-priority incremental matrices.</p>
<h3>The material measure</h3>
<p><strong>Range of the odd value</strong>It is the most direct quantitative method, but it is not possible to quantify appropriately the contribution of parameters to model performance. The author has proposed a basis for the<strong>Sensitivity</strong>The importance of the scoring:</p>
<p>&#36;&#36;
S_{k,i} = s(\lambda_{k,i}) + \frac{1}{d_1} \sum_{j=1}^{d_1} s(P_{k,ji}) + \frac{1}{d_2} \sum_{j=1}^{d_2} s(Q_{k,ij})
&#36;&#36;</p>
<p>Sensitivity using gradient-weight multiplier &#36;I(w_{ij}) = \left\lvert w_{ij} \nabla_{w_{ij&#125;&#125; \mathcal{L} \right\rvert&#36;and adopted<strong>Sensitivity Smooth</strong>and<strong>Quantified Uncertainty</strong>Addressing fluctuations:</p>
<p>&#36;&#36;
\begin{aligned}
\bar{I}^{(t)}(w_{ij}) &amp;= \beta_1 \bar{I}^{(t-1)}(w_{ij}) + (1 - \beta_1) I^{(t)}(w_{ij}) \
\bar{U}^{(t)}(w_{ij}) &amp;= \beta_2 \bar{U}^{(t-1)}(w_{ij}) + (1 - \beta_2) \left\lvert I^{(t)}(w_{ij}) - \bar{I}^{(t)}(w_{ij}) \right\rvert
\end{aligned}
&#36;&#36;</p>
<p>The final definition of importance is the product of both:&#36;s^{(t)}(w_{ij}) = \bar{I}^{(t)}(w_{ij}) \cdot \bar{U}^{(t)}(w_{ij})&#36;。</p>
<h3>Global budget movement</h3>
<p>Budget to be budgeted &#36;b^{(t)}&#36; Defines the sum of all incremental matrices (i.e. the total oddly foreign number). From a slightly higher than target budget &#36;b^{(T)}&#36; Initial budget &#36;b^{(0)}&#36; Start (e.g. 1.5 times) with the initial lid of each incremental matrix as &#36;r = b^{(0)}/n&#36;I'm sorry. Preheating training &#36;t_i&#36; After that, the budget is gradually reduced following a three-way movement strategy &#36;b^{(t)}&#36; Until the goal is achieved.</p>
<p>Special observation: Adalora always allocates more budget to the LM Head of the FNN and top floors, which is also consistent with the finding in the LoRA fine-tuning study that the concentration level is less important than the linear layer, and should provide the entire layer with the LoRA adaptor as far as possible, subject to the conditions.</p>
<p><strong>Adalora is currently a fine-tuning framework as useful as traditional LoRA, used in PEFT libraries <code>AdaLoraConfig</code> To call.</strong></p>
<h2>Summary and outlook</h2>
<p>This post has been used to review the representative approach of the PEFT field from Adapter to Adalora. These technologies together illustrate one thing:<strong>The parameter space of the pre-training model is heavily reusable and is sufficiently effective for many downstream missions, either low-dimensional or locally.</strong></p>
<p>From the technical route, the PEFT approach can be broadly divided into three directions:</p>
<ul>
<li><strong>Insert</strong>(Adapter): Insert training modules between models</li>
<li><strong>Prefix/tip</strong>(Prefix-Tuning, Prompt Tuning, P-Tuning v2) Injecting learning-able continuous vectors at input or at all levels</li>
<li><strong>Reparatic</strong>(LoRA, AdaloRA): Directly modify the weight matrix by low-stealing decomposition</li>
</ul>
<p>Of these, LoRA and its variants have become one of the most commonly used PEFT programmes at present, with low additional reasoning costs and flexibility.</p>
<p>Looking ahead, the PEFT research may need to be directed from simple efficient adaptation to a shift from a more efficient to a more efficient one.<strong>Optimizing RL dynamics for adaptation</strong>I'm sorry. As Meta discussed in Three-Gate Theory, the gradient update for the RL post-training may follow a different geometric path than SFT: It prefers to modify the non-main directional subspace with a lower curvature in the pre-training parameter space rather than the main ingredient direction. Thus, the traditional LoRA, which is based on low sovereignty renewal, does not always match RL ' s optimisation dynamics.<strong>How to design efficient methods of protecting and using parameters that can update this non-main orientation will be a matter of concern for the next phase of PEFT.</strong></p>
