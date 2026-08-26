---
title: 'LLM Lifecycle Overview: From Data and Pretraining to Decoding and Deployment'
title_zh: LLM 生命周期总览：从数据、预训练到解码与部署
date: 2024-08-15 20:00:00 +0800
categories:
- Foundation Models
- Model Mechanics
tags:
- LLM
- Pre-Training
- Tokenization
- Decoding
- Deployment
author: Hyacehila
mathjax: true
hidden: true
excerpt: An end-to-end map of the LLM lifecycle, covering data preparation, pretraining, post-training, decoding, quantization,
  and inference serving.
description: An end-to-end map of the LLM lifecycle, covering data preparation, pretraining, post-training, decoding, quantization,
  and inference serving.
excerpt_zh: 从完整生命周期理解大语言模型：数据怎样进入预训练，模型如何经过后训练获得可用行为，以及解码、量化与推理服务分别解决什么问题。
permalink: /blog/2024/08/15/llm-lifecycle-overview/
lang: en
translation_key: 2024-08-15-llm-lifecycle-overview
translation_status: machine
translation_source_hash: 5792840924fcb7977f373b61309ef1bb772b0aab9d0fb444aff5b50ecf6c2235
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<h2>Let's finish the life cycle.</h2>
<p>When I first sorted out the LLM basics, I disassembled development history, pre-trained data and code deployment. This is appropriate for chapter-by-chapter notes, but it is easy to lose contact with several stages: It is difficult to see why the cleansing of data affects post-training, why the syllabi limits reasoning services, and to quantify whether it occurs at the training or deployment stage.</p>
<p>This article re-organises the content along model life cycles. It is not a training manual and does not follow a model list that changes every month. Here's a more stable question:<strong>How the raw set of data was transformed into a model that could be used and what decisions were to be made at each stage.</strong></p>
<pre><code class="language-mermaid">graph LR
    A[&quot;原始数据&quot;] --&gt; B[&quot;过滤 / 去重 / 配比&quot;]
    B --&gt; C[&quot;Tokenizer 与训练样本&quot;]
    C --&gt; D[&quot;大规模预训练&quot;]
    D --&gt; E[&quot;SFT / 偏好学习 / RL&quot;]
    E --&gt; F[&quot;评估与安全检查&quot;]
    F --&gt; G[&quot;解码策略&quot;]
    G --&gt; H[&quot;量化与推理服务&quot;]
    H --&gt; I[&quot;线上反馈与数据回流&quot;]
    I --&gt; B
</code></pre>
<p>The chain is not a strict one-way street. The failure of the assessment leads the team back to the data and training formula, online feedback to the next round of post-training, and deployment restrictions may in turn change the size of the model, the length of context and selection of tokenizer. The so-called life cycle focuses on such pre- and post-restraint, rather than on training, reasoning and application in three sets of works.</p>
<h2>Data: Models first learn about data distribution</h2>
<p>Pre-training usually organizes mass text, code or multi-modular sequences into token, then the model predicts the next token. Target functions look simple, but data determine what language, knowledge, code style and behaviour patterns the model can see. The size of the parameters does not make it possible to recover what does not exist in the training distribution.</p>
<h3>Data sources aren't as good as they are.</h3>
<p>Common sources include web pages, books, papers, codes, multilingual materials and authorized operational data. They vary: the web page is wide-ranging, but there are many templates, advertisements and duplicates; books and essays are structured in such a way that they may be concentrated in a few areas; code data can be executed, validated and mixed into automated documents, licensing issues and security deficiencies.</p>
<p>The data list is long enough to replace data mixing. The team has to decide which languages and competencies the model is intended to cover before assigning sampling weights to the various categories of data. High-quality small data can be duplicated and low-quality sources of large volumes need to be reduced. The percentage of data in code, mathematics or target areas may also be increased at a later stage of training, but such movement changes the distribution of capabilities and can also be forgotten.</p>
<h3>Clean up at least four lines.</h3>
<ol>
<li><strong>Mass Filter</strong>: Text that handles sprawl, template pages, keyword stacking, abnormal symbol proportions and missing semantic content. Rules, classifications and models can be used in combination, but filters themselves can introduce deviations.</li>
<li><strong>Heavy.</strong>: Accurate repetition of waste calculations, and similar repetitions allow models to remember fixed expressions. Even more troublesome is the overlap between the training set and the evaluation set, which makes the results look much better than the real ones.</li>
<li><strong>Security and compliance</strong>Privacy, licences, harmful content and source records are not statements to be completed after training. Data need to be available from the point of entry into the pipe, knowing where it comes from, what it is processed and whether it can be deleted.</li>
<li><strong>Distribution check</strong>Total number of tokens does not indicate whether the data are healthy. Language ratios, area coverage, document length, time ranges and repetition rates from different sources need to be seen separately.</li>
</ol>
<p><a href="https://arxiv.org/abs/2107.06499">Deduplicating Training Data Makes Language Models Better</a> The impact of duplicate data on memory and generalization was discussed. Large-scale data engineering in recent years continues to validate the same thing over and over again: the filtering rules are not moats per se, they are capable of tracking data, replicating experiments and rematching assessments, and they are the available data lines.</p>
<h3>Tokenizer is part of the model.</h3>
<p>Tokenizer maps the string as discrete token ID. BPE, WordPiece, Unigram, etc. have different details, but are dealing with the same set of trade-offs: how big the vocabulary is, whether the common clips can be used less token to indicate that rare words and multilingual texts can be cut much.</p>
<p><a href="https://arxiv.org/abs/1508.07909">BPE</a> (a) Repeatedly merge HF clips from character or byte units;<a href="https://arxiv.org/abs/1808.06226">SentencePiece</a> This provides for direct training of BPE or Unigram tokenizer in the original text. In the actual system, more important than the algorithm name are the following:</p>
<ul>
<li>Tokenizer must be matched with model weights, and any change in the total input space will change;</li>
<li>Token compression rates in different languages have a direct impact on available context and reasoning costs;</li>
<li>Special token, chat templates and tool call formats are aligned at the training and reasoning stages;</li>
<li>Data cleansing must take place at two angles before and after tokenation, and normal characters do not represent normal token distribution.</li>
</ul>
<p>A model claims to support a long context, which does not mean that the same amount of information is available in each language. Tokenizer cuts more, and the same material fills the window earlier. This is a easily overlooked link between data engineering and reasoning services.</p>
<h2>Pre-training: digesting complex distribution with simple targets</h2>
<p>Today's common generation LLM uses decoder-only Transformer. For token series &#36;x_1, x_2, \ldots, x_T&#36;, probability of model learning conditions:</p>
<p>&#36;&#36;
p(x_{1:T}) = \prod_{t=1}^{T} p(x_t \mid x_{&lt;t})
&#36;&#36;</p>
<p>The most common target for training is to minimize the next token crossbow. Models are not taught “grammatically” “common sense” or “programming” on a case-by-case basis, but lower prediction errors on a large number of sequences. The capacity that ultimately emerges depends on the combination of architecture, data, budgeting and optimization processes, which cannot simply be attributed to the number of parameters.</p>
<h3>Why is Decoder-only mainstream?</h3>
<p><a href="https://arxiv.org/abs/1706.03762">Transformer</a> Replacing the cycle structure with a focus mechanism makes it easier to parallel serial calculations. The decoder-only model uses a causal mask, and each location can only see its previous token, so the training target is naturally consistent with the token decode at the time of generation.</p>
<p>It doesn't mean that encoder, encoder-decoder or other structures have lost value. Retrieving coding, classification, translation and multi-modular systems still use different structures. It's just that, when discussing generic generation models, the most mature training and reasoning ecology has developed. Attention and guidance from Transformer itself can be consulted.<a href="/en/blog/2024/11/14/self-attention-and-transformer-architecture/">Self-Focused Mechanisms and Transformer Architecture</a>。</p>
<h3>Scaling Law is a budget tool, not a smart formula.</h3>
<p><a href="https://arxiv.org/abs/2001.08361">Kaplan et al.</a>The empirical relationship between the loss of the language model and the amount of parameters, data and calculations was demonstrated;<a href="https://arxiv.org/abs/2203.15556">Chinchilla</a> It was also explained that, under a fixed calculation budget, model parameters and training token needed to grow more evenly. The relevant extrapolation and controversy can be found at the reference point<a href="/en/blog/2026/01/26/neural-scaling-laws/">Neal Scaling Laws: From Kaplan to Chinchilla</a>。</p>
<p>The real value of Scaling Law is to help the team estimate “more than double the calculation, and probably how much loss can be reduced” “Is the model too big to have enough data to feed”? It does not guarantee that a certain capability emerges suddenly at the size of a given parameter, let alone directly introducing universal intelligence. Discrepancies, warning templates and scoring thresholds may also express continuous upgrading as so-called capability leaps.</p>
<h3>When it's on scale, the training becomes system engineering.</h3>
<p>When a single GPU cannot accommodate model weights, gradients, optimizers and activated values, multiple parallel policies are required:</p>
<ul>
<li>Data in parallel allows multiple cards to process different bats and synchronize gradients;</li>
<li>A single layer matrix operation in parallel with squares to multiple cards;</li>
<li>The current line divides different layers into different equipment in parallel;</li>
<li>ZeRO/FSDP split weights, gradients and optimizer state, reducing reproduction per card;</li>
<li>Combining precision, activation of recalculation and kernel optimization such as FlashAttention, with numerical precision or additional calculation for display and throughput.</li>
</ul>
<p><a href="https://arxiv.org/abs/1909.08053">Megatron-LM</a> and <a href="https://arxiv.org/abs/1910.02054">ZeRO</a> It's two classic entry points to understand the division of work. Upon entering genuine training, data break points, optimizer state recovery, gradient anomalies, Loss Spike, hardware malfunctions and cross-node communications are also processed. The model's strength is often determined by these insatiable details.</p>
<h2>Post-training: transforming the base model into a usable model</h2>
<p>Pre-training optimizes serial probabilities and does not automatically receive stable instructions to follow, reject boundaries and dialogue formats. The training will then use monitoring data, preference data or environmental feedback to reshape the behaviour of the model in response to user requests.</p>
<p>Common paths include continuous pre-training, supervisory fine-tuning (SFT), high-efficiency fine-tuning of parameters (PEFT), direct preference optimization and incentive-based intensive learning. They address different issues: continuous pre-training to adjust the distribution of knowledge and fields, SFT teaching models to imitate target responses, preference for learning and RL to change the model ' s choice between a number of feasible answers.</p>
<p>This part is enough to form another full article.<a href="/en/blog/2024/11/01/llm-post-training-and-finetuning/">Post-Language Training and Fine-tuning Practice: From SFT, LoRA to Human Alignment</a>I don't know. This entry point is maintained here to serve as a reminder that the “model capabilities” observed in the application are usually the result of a combination of pre- and post-training capabilities.</p>
<h2>Decoding: The same set of probability can generate completely different text</h2>
<p>After training, the model gives the probability distribution of the next token at each step. The decoding device is to select a token from this distribution and to continue to generate it back in context. Model weights remain unchanged, and decode strategies are sufficient to significantly change output certainty, repetition rate and diversity.</p>
<h3>Uncertainty search</h3>
<p>Greedy choosing the highest probability token at every step. It is fast, re-emergible, suitable for classification labels, short answers and a number of structured tasks, but the local best is not the best for the whole sequence, and it is easily repetitious.</p>
<p>Beam search also keeps several candidate sequences and continues to expand after comparing cumulative probabilities. It is common in relatively closed missions such as machine translation, but open dialogue does not always benefit from higher serial probabilities. When it's too big, the output may be more conservative and more similar.</p>
<h3>Probability sampling</h3>
<p>Temperatures will resize logits. If original logits are &#36;z_i&#36;Temperature &#36;T&#36;, the probability of sampling is:</p>
<p>&#36;&#36;
p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
&#36;&#36;</p>
<p>&#36;T&#36; Lower distribution is sharper and output is more stable; higher, lower probability token increases opportunities, but errors and problems increase. Temperature close to zero is usually treated as greeny and is not an ordinary sampling point suitable for a direct proxy formula.</p>
<p>Top-k only highest probability &#36;k&#36; sample in token; Top-p (nucleus sampling) selects the cumulative probability to reach the threshold &#36;p&#36; The smallest pool of candidates. The former are fixed, while the latter vary according to context. The two can be combined with temperature, but the more parameters the controls are, in many cases, just twisting a few interactive buttons at the same time.</p>
<p><a href="https://huggingface.co/docs/transformers/main/en/generation_strategies">Transformers generation policy document</a>It's a common view code. The actual selection model is best mission-driven: there is a need for re-emergible structured fields to reduce randomity; there is a need for multiple creative candidates to increase sampling diversity and then for additional screening.</p>
<h3>Stop, length and repeat controls</h3>
<p>Decodering also has a number of seemingly minor but common parameters on line:</p>
<ul>
<li><code>max_new_tokens</code> Control the maximum amount of token generated, which is not a substitute for the correct cessation conditions;</li>
<li>EOS and stop security determine when to terminate, and chat templates and tool protocols must match them;</li>
<li>Mechanisms such as regulation penalty, freency penalty can discourage repetition, but excessive force can undermine terminology, codes and fixed formats;</li>
<li>The length penalty is primarily used for comparison of candidate sequences and is not appropriate to be understood as “the longer the answer, the better”.</li>
</ul>
<p>The hint requires model output JSON, but does not guarantee the validity of each brackets. Where a machine is required to stabilize consumption results, the grammatical constraints of the decoder phase should be incorporated into the design of the system. This is a limited decodement with JSON Schema.<a href="/en/blog/2026/03/01/structured-output-and-constrained-decoding/">Making Agent Work: Large Model Structured Output and Restricted Decoding Technologies</a>。</p>
<h2>Deployment: parameters can be loaded, only start</h2>
<p>The most intuitive cost of reasoning is the model weight. If parameter is &#36;N&#36;, use each parameter &#36;b&#36; Bit, the theoretical storage of weights is approximately:</p>
<p>&#36;&#36;
M_{weights} = N \times \frac{b}{8}
&#36;&#36;</p>
<p>7B parameters weigh only about 14 GB, INT8 about 7 GB and INT4 about 3.5 GB under FP16/BF16. Real deployment will also require KV Cache, temporary capacity, operational buffer zones and framework costs. The long context, the high confluence and the bigger bats make KV Cache a major constraint very quickly, so "weighting into visibility" is not equivalent to running services.</p>
<h3>Quantification: First ask what was quantified.</h3>
<p>Quantify the weight or activate map to a lower precision expression. Common routes include post-training quantification (PTQ) and quantitative sensor training (QAT). More frequently in LLM deployment, PTQ is not required to undergo complete re-training, but different methods vary between the calibration data, grouping, abnormal value processing and hardware cores.</p>
<p><a href="https://arxiv.org/abs/2208.07339">LLM.int8()</a>、<a href="https://arxiv.org/abs/2210.17323">GPTQ</a> and <a href="https://arxiv.org/abs/2306.00978">AWQ</a> They represent several influential low-bit routes. Quantitative effects must be assessed in terms of target hardware and target tasks: a smaller document does not necessarily result in an equivalent percentage of delayed reductions, nor does a format load any real acceleration of the internal core.</p>
<p>QLora is easily misplaced.<a href="https://arxiv.org/abs/2305.14314">QLoRA</a> The focus is on storing frozen base models in the form of 4 bits to retrain high-precision LoRA parameters, thereby reducing the fine-tune memory. It is a training programme, not a generic reasoning quantification algorithm. A separate decision on the format of deployment after the training is completed is required.</p>
<h3>The distillation and cutting changes the model itself.</h3>
<p>The distillation allows smaller models to learn the output, distribution or reasoning trajectory of the teacher model. It may reduce the cost of reasoning or replicate teachers ' mistakes and styles together. Cuts remove weights, corridors, layers or attention; unstructured thinning translates real speed gains only when supported by hardware and kernels.<a href="https://arxiv.org/abs/2301.00774">SparseGPT</a> It's one of the delegates' jobs after ILM's training.</p>
<p>Quantification of the main change values indicates that distillation retrains a student, cuttings the model structure or valid parameters. All three are called model compressions, but the engineering path is different.</p>
<h3>The reasoning service has to be stale.</h3>
<p>The online service also handles substantive monitoring, requests for dispatch, KV Cache reuse, prefill/decode separation, separation and failure recovery. Enter a token that usually completes the prefill in parallel, and the output token has to be decoded in a different mode of calculation. You can continue reading about the relationship between KV Cache and token costs<a href="/en/blog/2026/04/26/output-token-pricing-kv-cache-agent-cost/">Why Output Token is more expensive: from KV Cache to Agent Cost Project</a>。</p>
<p>The most likely areas of error during the deployment phase are options selected using offline accuracy or single-request speed. At a minimum, a real service is to measure both the visibility, the initial token delay, each token delay, the adsorption, the context degradation and the output quality.</p>
<h2>Look back at the life cycle with a watch.</h2>
<table>
<thead>
<tr>
<th>Phase</th>
<th>Main object</th>
<th>Decisions to be made</th>
<th>It's good to keep reading.</th>
</tr>
</thead>
<tbody><tr>
<td>Data readiness</td>
<td>Documents, codes, metadata</td>
<td>Sources, filtering, weighting, licensing, data mixing</td>
<td>The “Data” section of this paper</td>
</tr>
<tr>
<td>Tokenization</td>
<td>String and token ID</td>
<td>Glossary, Language Overwrite, Special Token, Chat Template</td>
<td><a href="https://arxiv.org/abs/1808.06226">SentencePiece</a></td>
</tr>
<tr>
<td>Pre-training</td>
<td>Model parameters and optimizers</td>
<td>Structure, data/parameters/power matching, parallel strategy</td>
<td><a href="/en/blog/2026/01/26/neural-scaling-laws/">Neural Scaling Laws</a></td>
</tr>
<tr>
<td>Post-training</td>
<td>Directives, preferences and environmental feedback</td>
<td>SFT, PEFT, optimisation, RL, evaluation</td>
<td><a href="/en/blog/2024/11/01/llm-post-training-and-finetuning/">Post-training and fine-tuning practices</a></td>
</tr>
<tr>
<td>Prompt and Context</td>
<td>Current requests and references</td>
<td>Command, Example, Context Selection, Output Contract</td>
<td><a href="/en/blog/2024/09/20/prompt-engineering-and-in-context-learning/">Prompt Project and Context Learning</a></td>
</tr>
<tr>
<td>Decoding</td>
<td>Probability distribution for next token</td>
<td>Greedy, sampling, suspension conditions, grammar constraints</td>
<td><a href="/en/blog/2026/03/01/structured-output-and-constrained-decoding/">Structure output and restricted decoding</a></td>
</tr>
<tr>
<td>Deployment</td>
<td>Weight, KV Cache, Request Queue</td>
<td>Precision, quantification, batch processing, co-production, hardware</td>
<td><a href="/en/blog/2026/04/26/output-token-pricing-kv-cache-agent-cost/">KV Cache and Agent Costs</a></td>
</tr>
</tbody></table>
<p>When these stages are linked, much of the debate will be simpler. Models are not effective and need not change structures immediately; they may be data distribution, post-training targets, hint context or decoding strategies. Deployment costs are high and not necessarily quantifiable; shorter output, better monitoring and less ineffective context are sometimes more direct. It is much more reliable to judge at what level problems occur and to select tools than to pursue a technical term.</p>
<h2>References</h2>
<ul>
<li><a href="https://arxiv.org/abs/1706.03762">Attention Is All You Need</a></li>
<li><a href="https://arxiv.org/abs/2001.08361">Scaling Laws for Neural Language Models</a></li>
<li><a href="https://arxiv.org/abs/2203.15556">Training Compute-Optimal Large Language Models</a></li>
<li><a href="https://arxiv.org/abs/2104.04473">Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM</a></li>
<li><a href="https://arxiv.org/abs/1910.02054">ZeRO: Memory Optimizations Toward Training Trillion Parameter Models</a></li>
<li><a href="https://arxiv.org/abs/2107.06499">Deduplicating Training Data Makes Language Models Better</a></li>
<li><a href="https://arxiv.org/abs/2305.14314">QLoRA: Efficient Finetuning of Quantized LLMs</a></li>
</ul>
