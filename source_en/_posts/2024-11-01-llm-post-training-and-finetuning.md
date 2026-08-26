---
title: 'LLM Post-Training and Fine-Tuning Practice: From SFT and LoRA to Human Alignment'
title_zh: 大语言模型后训练与微调实践：从 SFT、LoRA 到人类对齐
date: 2024-11-01 20:00:00 +0800
categories:
- Foundation Models
- Training & Alignment
tags:
- LLM
- Post-Training
- SFT
- LoRA
- RLHF
- Alignment
author: Hyacehila
mathjax: true
hidden: true
excerpt: An engineering-oriented guide to post-training, covering data formats, chat templates, memory estimation, SFT, LoRA,
  QLoRA, preference optimization, and RLHF.
description: An engineering-oriented guide to post-training, covering data formats, chat templates, memory estimation, SFT,
  LoRA, QLoRA, preference optimization, and RLHF.
excerpt_zh: 把后训练放回完整工程链：从数据格式、chat template、loss mask 和显存估算出发，理解 SFT、LoRA、QLoRA、偏好优化与 RLHF 分别改变什么。
permalink: /blog/2024/11/01/llm-post-training-and-finetuning/
lang: en
translation_key: 2024-11-01-llm-post-training-and-finetuning
translation_status: machine
translation_source_hash: af7ac206c1b11f258c6765f3fb5a75f932105a8264da42f9403ee61d080ae5c6
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<h2>Post-training is not an algorithm.</h2>
<p>At the end of the pre-training exercise, we got a base model that's good at predicting the next token. It may have a large body of language and code models, but it may not necessarily answer in the way the user expects: unstable formats, vague mission boundaries, and not necessarily what should be the priority when faced with conflicting instructions. What is to be done after training is to re-engineer this behaviour with smaller and more targeted data.</p>
<p>I had two earlier sessions of fine-tuning of instructions, human alignment, visible estimates, data formats and framework options. The problem is that when training methods and engineering conditions are removed, it is easy to conclude that LoRA is only discussed without discussion of activation values, compared with data formats without looking at cat template, or that QLora is considered to be a quantitative deployment. Now put them back in the same chain.</p>
<pre><code class="language-mermaid">graph LR
    A[&quot;预训练模型&quot;] --&gt; B[&quot;持续预训练&quot;]
    A --&gt; C[&quot;监督微调 SFT&quot;]
    B --&gt; C
    C --&gt; D[&quot;偏好数据&quot;]
    D --&gt; E[&quot;DPO 等直接偏好优化&quot;]
    D --&gt; F[&quot;奖励模型 + RL&quot;]
    C --&gt; G[&quot;评估 / 安全 / 领域测试&quot;]
    E --&gt; G
    F --&gt; G
    G --&gt; H[&quot;部署与数据回流&quot;]
</code></pre>
<p>The routes are different:</p>
<ul>
<li><strong>Ongoing pre-training</strong>(b) Continuing to study the text distribution of the fields, suitable for language, knowledge and code style;</li>
<li><strong>SFT</strong>(a) Simulate quality input pairs and create command compliance, format and task behaviour;</li>
<li><strong>PEFT</strong>(b) Determining which parameters are involved in updating is a training resource programme and not a stand-alone data target;</li>
<li><strong>Prefer Optimization</strong>More multiple responses to make models more selective;</li>
<li><strong>Enhanced learning</strong>Converting incentives or environmental feedback into strategic updates that are appropriate for interactive, verifiable or issues to explore.</li>
</ul>
<p>There is no fixed streaming line for post-training. A field completion model may require only continuous pre-training, a taxonomy may be only SFT, and Agent may use SFT tracks and environmental incentives in turn. The order cannot be reversed before identifying what behaviour is to be changed and deciding on data and algorithms.</p>
<h2>Data Engineering: Define the target for training first</h2>
<p>The training script ends with token sequences and a loss mask, but the team should maintain a structured sample on a daily basis. The more common target groups are three.</p>
<h3>Ongoing pre-training text</h3>
<p>Ongoing pre-training usually retains the continuous structure of natural text or code:</p>
<pre><code class="language-json">{&quot;text&quot;: &quot;领域文档、代码或其他连续语料……&quot;}
</code></pre>
<p>There are no natural data for this type of data. The training target remains the projection of the next token, so it is closer to pre-training, except for more centralized data distribution and computing. If the goal is to teach models to adhere to a certain output format, continuous pre-training alone is often not direct enough.</p>
<h3>Command and single-wheel samples</h3>
<p>Alpaca styles are common <code>instruction</code>、<code>input</code>、<code>output</code> For a single task:</p>
<pre><code class="language-json">{
  &quot;instruction&quot;: &quot;找出日志中的直接故障原因&quot;,
  &quot;input&quot;: &quot;&#123;&#123;日志内容&#125;&#125;&quot;,
  &quot;output&quot;: &quot;连接池耗尽导致请求持续排队。&quot;
}
</code></pre>
<p>It is suitable for data production and manual review, but these fields are not the format that the model eventually reads. They are still token sequences of target models to be rendered before training.</p>
<h3>Multiple rounds of information</h3>
<p>Modern dialogue data usually use message arrays:</p>
<pre><code class="language-json">{
  &quot;messages&quot;: [
    {&quot;role&quot;: &quot;system&quot;, &quot;content&quot;: &quot;只依据给定材料回答。&quot;},
    {&quot;role&quot;: &quot;user&quot;, &quot;content&quot;: &quot;这次故障影响了哪些服务？&quot;},
    {&quot;role&quot;: &quot;assistant&quot;, &quot;content&quot;: &quot;记录确认支付和订单查询受到影响。&quot;}
  ]
}
</code></pre>
<p>ChatML, ShareGPT, Alpaca are common agreements, not uniform standards across models. ShareGPT Common <code>conversations/from/value</code>Message format is common <code>messages/role/content</code>The different training frameworks would also accept their own listing. Before entering tokenizer, you should convert to an internal uniform scheme, then replay it with the model's cattemplate.</p>
<h3>Chat template decides what the model actually sees.</h3>
<p>Chat texture uses characters, messages and special tokens as strings for modeling training. The following two messages are synonymous with different graphs and can be given completely different starting and starting marks and role token after different tokenizers. When templates do not match, the model sees controlrs that were never seen during the training phase, and performance is often more hidden than the problem of the data content itself.</p>
<p>The Hugging Face <a href="https://huggingface.co/docs/transformers/main/en/chat_templating">Chat Templates</a> The document is given. <code>apply_chat_template</code> - How it's used. The training and reasoning should be re-enacted with the same template and make it clear whether to add a reference protocol, where to place EOS, how to encode the tool message. Do not manually guess a format that looks like the model.</p>
<h3>Los Mask decides who the model is.</h3>
<p>A conversation can calculate the loss of all tokens or only train antsant to respond. The latter are often called resonese-only or "assistant-only loss: system " and "user token " , which are still entered as conditions without requiring models to predict them.</p>
<p>There is no correct answer for all the tasks. Training only asistant is more in line with the goal of “generated answers on request” and avoids wasteful model capacity restatement users; some training that requires learning complete interactive structures or special controls token may also leave teams with more positions to lose. It's important to record the mask clearly. The two texts are identical, and the training objectives are different as long as the los mask is different.</p>
<h3>Packing up utilization and changing sample boundaries</h3>
<p>A large number of short samples, if article by article, would waste a significant portion of the calculations. Packing toss multiple samples into the same sequence to increase the effective token ratio. This is done by confirming whether attention is allowed to cross the sample, how the location ID is handled, whether EOS is correctly inserted, and whether the loss mark is at the wrong place at the border.</p>
<p>Packing is a stale optimization, and should not change the semantic of the data. If a model emerges after training that brings the last answer to the next question, the sample boundaries and templates are checked instead of the immediate learning rate.</p>
<h3>High quality is not a model rating</h3>
<p>Post-training data should be checked at least for:</p>
<ul>
<li>Whether the Directive is enforceable and whether the entry contains the information required to complete the task;</li>
<li>The answer is correct, complete and in the same way as the desired model;</li>
<li>(a) Coherence of roles, tools and environmental status in the multi-cycle dialogues;</li>
<li>Whether to mix the rating collection, template leaks, error references and unrecoverable external state;</li>
<li>Whether data sources, generation models, filtering rules and history modification can be traced.</li>
</ul>
<p>Synthetic data reduce the cost of writing samples and make it easier to scale up the same error. See more complete data synthesis process.<a href="/en/blog/2026/07/03/sft-synthetic-data-engineering/">Data synthesis is becoming a project: from Terminal-Corpus</a>I'm sorry. The article discusses how to reverse training data from mission world, environment and verifier; here is the concern about how data is coded and calculated for the loss.</p>
<h2>Visible estimate: Discount first, then training.</h2>
<p>The "7B model requires a certain amount of visibility" without a fixed answer that is off the configuration. The full parameter is also LoRA, weight accuracy, optimizer, sequence length, bat size, whether to save master weights, checkpointing and fractions. It would be more stable to break the apparent deposits into several books.</p>
<h3>Weights</h3>
<p>Arguments in &#36;N&#36;, store accuracy is &#36;b&#36; When bit, the theoretical size of weight is:</p>
<p>&#36;&#36;
M_{weights} = N \times \frac{b}{8}
&#36;&#36;</p>
<p>7B parameters are about 14 GB, INT8 about 7 GB, 4 bit about 3.5 GB using BF16/FP16. This is only parameter data per se, and quantitative scale, zero point, grouping metadata and running buffers add additional occupancy.</p>
<h3>Gradient and Optimizer State</h3>
<p>If the training parameter is &#36;P&#36;, the gradient is approximated to:</p>
<p>&#36;&#36;
M_{grad} = P \times bytes_{grad}
&#36;&#36;</p>
<p>Adam W, usually, keeps the first and second steps for each training parameter, and if both are used FP32, this is only part of it. &#36;8P&#36; Bytes. Some blending precisions will retain FP32 master weights, and add more. &#36;4P&#36; Bytes. The framework, optimizer and fractional policy are different and cannot be considered a "fixed byte number per parameter" as a common constant.</p>
<p>Take the full 7B BF16 training as an example, without consideration for fraction and activation: weights of about 14 GB, BF16 gradients of about 14 GB, two FP32 Adam states of about 56 GB; if FP32 master weights are saved, add about 28 GB. That is, these states alone could be 84 GB or 112 GB, active values, communication buffers and CUDA workspaces that have not yet started to be counted.</p>
<p>The difference between LoRA is clear here: the base weight still needs to be loaded, but only low-skull parameters are involved in gradients and optimizer updates, and therefore &#36;P \ll N&#36;I'm sorry. But the "Landitude and Optimizer Smaller" does not mean that the active value disappears, and long sequences and large bats may still fill the display.</p>
<h3>Activate Value</h3>
<p>Activated values relate to the size of the bat, the length of the sequence, the hidden dimensions, the layers, the attention achieved and the intermediate results that are saved. It will not be determined by the amount of parameters alone and it will be difficult to estimate using a fixed factor across structures.</p>
<p>These techniques are commonly used in practice to control the activation of the presence:</p>
<ul>
<li>(a) Gradient checkpointing does not save all intermediate results and recalculates when they are disseminated in reverse;</li>
<li>(a) The physicalization of a matrix of reduced attention, such as FlashAttention;</li>
<li>(a) Increase the effective token ratio by setting a limit ton-up, but also by activation with the total token increase;</li>
<li>A few small micro-batchs simulated by a larger, effective watch;</li>
<li>Shortening the length of the sequence usually reduces activation costs more directly than fine-tuning a few LoRA rank.</li>
</ul>
<h3>Split and Frame Costs</h3>
<p>FSDP, Zero, takes parameters, gradients and optimizer status fractions to multiple cards. They reduce the use of single cards, but increase the complexity of communications, movement control and configuration. CUDA constex, Kirnel workspace, temporary load, dataloader pre-emission and visible debris also require residual capacity.</p>
<p>So the visual memory estimate should be written in a compartment and validated by a short run with a target configuration. It is not enough to complete forward/backward, to see if peaks are visible, swallowed, offload, and if they are frequent, and if they are saved by checkpoint, OOM will not be repeated.</p>
<h2>SFT: Make models imitate target behavior</h2>
<p>Monitors fine-tuning the calculation of cross-cabins using input and target output, allowing the model to increase the probability of target token.<a href="https://arxiv.org/abs/2203.02155">InstructGPT</a>、<a href="https://arxiv.org/abs/2109.01652">FLAN</a> and <a href="https://arxiv.org/abs/2212.10560">Self-Instruct</a> Shows the role of manual commands, task mixing and synthetic commands in the context of integration.</p>
<p>SFT is best suited to learn exemplary behaviour: answer structure, tool trajectories, domain terminology, refusal mode, code modification mode and conversation style. It does not guarantee reliable inclusion of facts in training samples in models, nor does it allow models to stabilize tasks that are completely impossible for the base model. If training data are themselves guessed, models can only be more skilled in imitating such guesses.</p>
<h3>Learning rates, bats and epoch do not exit the data.</h3>
<p>Post-training usually uses a learning rate lower than pre-training, but the specific range is influenced by model size, full parameters/PEFT, data volume, optimizer and target mission. What is more useful than observation of the default set of parameters:</p>
<ul>
<li>Training for loss of productivity and the absence of improvement in validation tasks;</li>
<li>Whether the model is rapidly losing its original generic capability;</li>
<li>Whether or not a few templates occupy the output, causing style collapse;</li>
<li>Whether long and short answers receive different weights because of the token difference;</li>
<li>After a mixture of multiple data sources, which sample dominates the gradient.</li>
</ul>
<p>Small data repeats multiple epochs easily. When data is big, a part of the training epoch may be sufficient. The final choice should be made by the set of tasks left, the pass rate and the regression test, rather than by training only, the loss.</p>
<h2>PEFT: Reduced number of updated parameters without changing training objectives</h2>
<p>Efficient fine-tuning of parameters (PEFT) freezes most of the base parameters and only trains a small number of additional parameters or the selected parameters. It reduces the presence of gradients and optimizers and allows the preservation of adapter for multiple tasks. More complete technology spectrum.<a href="/en/blog/2026/03/05/peft-parameter-efficient-fine-tuning/">PEFT: Technology Evolution from Adapter to LoRA</a>。</p>
<h3>LoRA</h3>
<p><a href="https://arxiv.org/abs/2106.09685">LoRA</a> It is assumed that the weights needed to update downstream adaptations have a lower inherent value. For original weights &#36;W_0&#36;It's frozen. &#36;W_0&#36;, with two low-swipe matrices:</p>
<p>&#36;&#36;
W = W_0 + \Delta W, \qquad \Delta W = BA
&#36;&#36;</p>
<p>If &#36;W_0 \in \mathbb{R}^{d_{out} \times d_{in&#125;&#125;&#36;♪ I'm so sorry ♪ &#36;r&#36;, and &#36;A \in \mathbb{R}^{r \times d_{in&#125;&#125;&#36;、&#36;B \in \mathbb{R}^{d_{out} \times r}&#36;I'm sorry. When? &#36;r&#36; Training parameters are significantly reduced when much smaller than input output dimensions.</p>
<p>The main decisions of LoRA include target Modeles, rank, alpha, dropout and training for embedding, lm head or bias. The only word for "use" is still lacking enough information. The amount of argument may be significantly different from the effect of the attention projector, the effect of the profile, the effect of the profile and the addition of the profile to all linear layers.</p>
<p>Adapter can combine base weights before reasoning, or maintain independent loading. The consolidation does not normally add additional matrix calculations, but loses the convenience of running multiple adapter; the non-merger is more flexible and requires the correct support of the reasoning framework.</p>
<h3>QLoRA</h3>
<p><a href="https://arxiv.org/abs/2305.14314">QLoRA</a> Store and calculate the frozen base weights in 4 bits, while training higher precision LoRA parameters. It further subpresss the weight, but training still requires counterquantification of the calculation, activation and LoRA optimization.</p>
<p>QLora is not " train all parameters with INT4 " , nor is it a generic term for deployment of a quantitative format. It addresses the low resource fine-tuning; the decision whether to merge an adapter after the training is completed, what precision to export, and whether the target service supports the corresponding quantitative kernel remains another group.</p>
<h3>Adapter and parameter selection</h3>
<p>Classic Acapter inserts small bottlenecks networks in Transformer layers; prompt turning, prefix turning trains continuous vectors, and puts learning information in input or attention prefixes. Another method selects a partial layer or parameter update directly. They're all diminishing. &#36;P&#36;, but the placement, reasoning costs and multi-task combinations are different.</p>
<p>PEFT saves on training status and does not automatically fix data problems. If full fine-twining does not learn about targeted behaviour, LoRA is not usually solved in a vacuum; in turn, the task requires only light-scale behaviour and full-parameter updates may simply increase costs and forget risks.</p>
<h2>Framework selection: group by function, rather than choosing a universal framework</h2>
<p>The fine-tuning tools are often placed in the same ranking table, but they are at different levels. Before comparing them, look what they are responsible for.</p>
<table>
<thead>
<tr>
<th>Component</th>
<th>Main duties</th>
<th>Official entrance</th>
</tr>
</thead>
<tbody><tr>
<td>Transformers</td>
<td>Models, tokenizer, chat template, basic training interface</td>
<td><a href="https://huggingface.co/docs/transformers/main/en/training">Transformers</a></td>
</tr>
<tr>
<td>Datasets</td>
<td>Data loading, mapping, fluid processing and cache</td>
<td><a href="https://huggingface.co/docs/datasets">Datasets</a></td>
</tr>
<tr>
<td>PEFT</td>
<td>Efficient method for parameters such as LoRA, IA3, prompt/prefix turning</td>
<td><a href="https://huggingface.co/docs/peft">PEFT</a></td>
</tr>
<tr>
<td>TRL</td>
<td>SFT, DPO, incentive model, PPO/GRPO etc.</td>
<td><a href="https://huggingface.co/docs/trl">TRL</a></td>
</tr>
<tr>
<td>Accelerate</td>
<td>Single-line Doca, hybrid precision and distribution startup</td>
<td><a href="https://huggingface.co/docs/accelerate">Accelerate</a></td>
</tr>
<tr>
<td>PyTorch FSDP</td>
<td>Parameters, Gradients and Optimizer Status Spectrometers</td>
<td><a href="https://docs.pytorch.org/docs/stable/fsdp.html">FSDP</a></td>
</tr>
<tr>
<td>DeepSpeed</td>
<td>ZeRO, offload, parallel training and reasoning component</td>
<td><a href="https://www.deepspeed.ai/tutorials/zero/">ZeRO</a></td>
</tr>
<tr>
<td>Unsloth</td>
<td>Optimizing the performance of the seals for common models and the LoRA/RL workflow</td>
<td><a href="https://docs.unsloth.ai/">Unsloth Docs</a></td>
</tr>
</tbody></table>
<p>A common combination is Transformers for models and tokenizer, Datasets for data, PEFT for LoRA, TRL for SFT Trainer or preferred trainers, and Accelerate, FSDP or DeepSpeed processing equipment and fractions. Unsloth provides this eco-based set with optimized models for loading, kernel and training portals.</p>
<p>The selection of the frame depends on whether the target model supports, the chatch testmate is correct, the trainers are able to express the loss of the mark and the packing, the distributed checkpoint is restored, and the final weight is readable in the deployment frame. API writes short just as part of the experience. The higher the abstract layer, the more it needs to be prepared to go back to the bottom to confirm what it did when it met with new models and special ross.</p>
<h3>Base is still a model that has been incorporated-tuned</h3>
<p>The Base model retains a more primitive pre-training distribution, suitable for teams with sufficient data to want to redefine interactive behaviour; the interstasis-tuned model already has dialogue and guidelines to follow, and a small number of field data are usually more readily available, and may inherit original templates, refusals and style preferences.</p>
<p>There is no "how many data above must select the Base" common threshold. More practically, two starting points are compared with the same validation missions: if the INSTRUCTION model is already in place, it will be necessary to supplement the field behaviour and continue to fine-tune it, which is usually more economical; and if the original alignment seriously interferes with target formats or language distribution, then the Base model and the more complete data formulation will be assessed.</p>
<h2>Human alignment: from imitation of answers to comparative behavior</h2>
<p>SFT data tells the model "Here is a target answer." The preferred data give multiple answers and their relative selection under the same prompt, and allow training objectives to move from imitating a single answer to adjusting the probability relationship between the answer.</p>
<h3>Classic RLHF</h3>
<p><a href="https://arxiv.org/abs/2203.02155">InstructGPT</a> The processes frequently cited include three phases:</p>
<ol>
<li>(a) Manual demonstration of SFT to obtain the initial strategy to follow the directive;</li>
<li>(b) Sorting of multiple model responses and training of incentive models;</li>
<li>Use the PPO optimization policy to give higher rewards to answers while using KL binding to avoid a deviation from the reference model.</li>
</ol>
<p>The reward model is not a “human values function”. It learns about preference agents in specific labelling norms, samples and model distributions. The strategy may exploit the incentive gap when it is continually targeted to optimize it, and therefore there is a need to retain manual assessments, stand-alone task sets and behavioural constraints.</p>
<p>The PPO's hardness is not just an algorithm formula. The system is complicated by the generation of samples, computing incentives, estimations, training in value mode, control of KL, processing of long deviations and maintenance of multi-machine ingestion. For a full discussion of the issues of reward, baseline, advantage and numaration, see<a href="/en/blog/2026/03/16/rl-alignment-from-reward-to-advantage/">Intensive Learning in LLM Alignment: From Incentive Signals to Estimation of Strengths</a>。</p>
<h3>Direct optimisation</h3>
<p><a href="https://arxiv.org/abs/2305.18290">DPO</a> The goal of changing preference modelling to a direct optimization strategy and reference model does not require separate training incentive models and online PPOs. It brings training closer to regular supervisory learning, but does not eliminate data problems: Whether the difference between the size of the box indicates the preference of the labeler, whether the length of the response is short and whether the training distribution covers the line request, still determines the final effect.</p>
<p>The names of DPO, KTO, ORPO will continue to be added. When reading these methods, three questions can be asked: what feedback data are used, how reference models or hidden incentives enter the loss, whether training is offline or does online sampling are needed. This makes it easier to discern differences in the project than to group them simply by “Is it RL or not”.</p>
<h3>SFT, Prejudice and RL. How to divide the job.</h3>
<ul>
<li>SFT is used first when there is a clear standard answer or a demonstration trajectory;</li>
<li>Multiple responses are available, but preference data are used when learning style, security or quality sequencing;</li>
<li>The award is from the environment, from testing or from interactive results, and is then considered online RLs when different strategies need to be explored;</li>
<li>In any case, independent assessments are maintained to prevent training signals from becoming more skilled.</li>
</ul>
<p>Many projects are entering complex preference training too early, and the practical problems are SFT data format errors, chat test inconsistencies or weak assessment sets. The more sophisticated the post-training methodology, the more it is necessary to prove that the previous phase has stabilized.</p>
<h2>An implementation checklist</h2>
<table>
<thead>
<tr>
<th>Problem</th>
<th>Decisions requiring recording</th>
</tr>
</thead>
<tbody><tr>
<td>Objective</td>
<td>What kind of behaviour is expected to change and which capabilities must remain intact?</td>
</tr>
<tr>
<td>Base</td>
<td>Base is still intract mode, whether license matches tokenizer</td>
</tr>
<tr>
<td>Data</td>
<td>How are schema, source, ratio, weight, quality doors and assessment sets defined</td>
</tr>
<tr>
<td>Templates</td>
<td>Chat text, special token, EOS and tool message code</td>
</tr>
<tr>
<td>Loss</td>
<td>Which token calculates the loss, how long the sample is added Rights</td>
</tr>
<tr>
<td>Training</td>
<td>Full/LoRA/QLoRA, target modules, precision, bat, serial length</td>
</tr>
<tr>
<td>Organisation</td>
<td>What are the weights, gradients, optimizers, activated, fractions and residuals?</td>
</tr>
<tr>
<td>Evaluation</td>
<td>Mission quality, pass rate, security, return, delay and cost</td>
</tr>
<tr>
<td>Export</td>
<td>Whether or not to merge and whether the final precision is compatible with the deployment framework</td>
</tr>
<tr>
<td>Roll back</td>
<td>Checkpoint, complete copy and training configuration</td>
</tr>
</tbody></table>
<p>Post-training is often described as the selection of a framework, preparation of a JSONL, initiation of training. Actual results depend on consistency between objectives, data, templates, Loses, resources and assessments. LoRA can reduce the number of parameters updated, and cannot define tasks for you; RLHF can use preference signals or turn unreliable labels into reliable values.</p>
<p>If only one order of work is retained, I choose: to write an assessment before sorting out the data; to run through SFT before deciding whether to opt for optimization; to unload the visible accounts before choosing the frame. This is not a fancy exercise, but it can make most of the failures more visible earlier.</p>
<h2>References</h2>
<ul>
<li><a href="https://arxiv.org/abs/2203.02155">Training Language Models to Follow Instructions with Human Feedback</a></li>
<li><a href="https://arxiv.org/abs/2106.09685">LoRA: Low-Rank Adaptation of Large Language Models</a></li>
<li><a href="https://arxiv.org/abs/2305.14314">QLoRA: Efficient Finetuning of Quantized LLMs</a></li>
<li><a href="https://arxiv.org/abs/2305.18290">Direct Preference Optimization: Your Language Model is Secretly a Reward Model</a></li>
<li><a href="https://arxiv.org/abs/2212.10560">Self-Instruct: Aligning Language Models with Self-Generated Instructions</a></li>
<li><a href="https://arxiv.org/abs/2109.01652">Finetuned Language Models Are Zero-Shot Learners</a></li>
</ul>
