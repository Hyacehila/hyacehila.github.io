---
title: 'Reinforcement Learning in LLM Alignment: From Reward Signals to Advantage Estimation'
title_zh: LLM 对齐中的强化学习：从奖励信号到优势估计
date: 2026-03-16 17:30:00 +0800
categories:
- Foundation Models
- Training & Alignment
tags:
- Reinforcement Learning
- Alignment
- Reward Modeling
- Advantage Estimation
- Training Dynamics
author: Hyacehila
mathjax: true
excerpt: Following the signal chain from reward to baseline, advantage, and normalization, this post explains why RL algorithms
  for LLM alignment keep rewriting reward signals.
description: Following the signal chain from reward to baseline, advantage, and normalization, this post explains why RL algorithms
  for LLM alignment keep rewriting reward signals.
excerpt_zh: 从 reward、baseline、advantage 与 normalization 这条信号链出发，解释为什么 LLM 对齐中的 RL 算法总在重写奖励信号。
permalink: /blog/2026/03/16/rl-alignment-from-reward-to-advantage/
lang: en
translation_key: 2026-03-16-rl-alignment-from-reward-to-advantage
translation_status: machine
translation_source_hash: 0b5e1ae58fc0e359f98024f45cf20e304d534fcbde5719305e1b5d229e4e9676
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<p>For a long time, we talked about enhanced learning in LLM alignment, often in algorithmic names: PPO, DPO, RLOO, GRPO, REINFORCE++. It is certainly convenient to remember this, but it is also easy to lose a comprehensive understanding of the field, as if the progress in this area is mainly about changing abbreviations, changing a loss, changing a training formula, increasing points (and not necessarily really rising), pub, and then going to the next. The strategies are perfect, but it seems that outsiders don't know what they're up to.</p>
<p>This article defaults that readers already know SFT, preference data and classic RLHF processes. If you need to see the training first, then you can go back to the training, LoRA and the visual estimates.<a href="/en/blog/2024/11/01/llm-post-training-and-finetuning/">Post-Language Training and Micro-Adjusting Practice</a>; here is the special tracking of how reward becomes an optimised signal.</p>
<p>It'll be clearer from another angle.<strong>LLM is constantly changing in Quri, the way to reward signal entry to the optimizer.</strong> A primitive reward, how it was bound, distributed, reduced, consolidated and eventually transformed into a stable and updated strategy, is the main disagreement of these approaches.</p>
<p>Once we have this main line, the way we cut each other apart can be linked:</p>
<ul>
<li>PPO provides early LLM RLHF project templates with high impact reward-to-advantage;</li>
<li>DPO bypassing the online RL to write preferences directly into the optimization target;</li>
<li>GRPO and REINFORCE++ re-examine:<strong>No critic after, baseline and numaration should come from.</strong></li>
</ul>
<p>Here is the conceptual framework for how to optimizer consumption; if you care about TRL <code>GRPOTrainer</code> The code-level principle and training interface that allows comparison <a href="/en/blog/2025/12/30/Re0HF-04/">Re0-04: TRL GRPO Trainer (principle)</a>。</p>
<h2>Plugin: Why is it not the RL in LLM that has no reward, but rather whether it can be a reliable training signal?</h2>
<p>Reward in traditional intensive learning, at least in many classical environments, is relatively clear: game scores, goal hits, collisions, success in mission. But in LLM-System, reward is not usually the same.</p>
<p>First of all, it's often...<strong>Slight</strong>Yeah. In many scenarios, the reward model gives you a total score after the model produces a complete section of the answer, or the certifier tells you the answer.<a href="https://proceedings.neurips.cc/paper/2020/file/1f89885d556929e98d3ef9b86448f951-Paper.pdf">Learning to Summarize from Human Feedback</a> and <a href="https://arxiv.org/abs/2203.02155">InstructGPT</a> This type of classic RLHF process is largely addressing this final feedback.</p>
<p>Second, it's often...<strong>Delay</strong>Yeah. The value of a single answer token of the dozens that lie ahead is probably only judged after the entire answer has been completed. One of the difficulties of the strategy gradient is to redistribute the late signal back to the process of generation.<a href="https://doi.org/10.1007/BF00992696">Williams 1992</a> / <a href="https://proceedings.neurips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation">Policy Gradient Theorem</a></p>
<p>And third, it's often...<strong>Noisey.</strong>I'm sorry. The reward model is not a truth judge, it's just a sorrogate. It can learn from some patterns of human preference, or from others. OpenAI has long observed in the wrap-up mission that if strategies are over-optimised, the end result is a departure from the real human preference.</p>
<p>Fourth, it could be.<strong>Utilization</strong>I'm sorry. Models may not really be more helpful for open missions, but they may be better at following reward models. And that's why many RLHF articles introduce mechanisms like KL constraint, length control, reference models: not because they look more elegant, but because without them, rewards are too easy to hack.</p>
<p>So, the RL in the LLM is never just a reward, but:</p>
<p><strong>This is a training signal that can be processed into a low differential, but not a perfect, not easily expluited, and can keep the scale stable across the prompt.</strong></p>
<h2>TLDR</h2>
<p>Write the judgment I want to put at the top of the line.</p>
<ol>
<li><strong>LLM-RL evolution is constantly rewrite <code>reward -&gt; advantage</code> This chain.</strong> It is not the algorithm that changes, but who is better at translating raw feedback into stable estimates of strengths.<a href="https://arxiv.org/abs/1707.06347">PPO</a> / <a href="https://aclanthology.org/2024.acl-long.662/">Back to Basics</a> / <a href="https://arxiv.org/abs/2501.03262">REINFORCE++</a></li>
<li><strong>Classic RLHF reward, not a nudity score from the beginning, but a composite signal of "mission reward plus behavioral restraint".</strong> <a href="https://proceedings.neurips.cc/paper/2020/file/1f89885d556929e98d3ef9b86448f951-Paper.pdf">Learning to Summarize from Human Feedback</a> / <a href="https://arxiv.org/abs/2203.02155">InstructGPT</a></li>
<li><strong>The clitic-free method does not eliminate the problem of advantage estability, but simply transfers it from the clitic network to the reward processing and statistical construction.</strong> You don't train value head, but you have to answer: where is baseline from, how do you measure it, how do you control the local noise?<a href="https://aclanthology.org/2024.acl-long.662/">Back to Basics</a> / <a href="https://arxiv.org/abs/2402.03300">DeepSeekMath / GRPO</a></li>
<li><strong>The key innovation of GRPO is not only the use of critić, but also the adaptation of learning signals to group comparative advantage.</strong> This is very effective in the reasoning task, but it also exposes the problem: what about the small difference in local standards? Is better in the group equal to better in the whole world?<a href="https://arxiv.org/abs/2402.03300">DeepSeekMath / GRPO</a></li>
<li><strong>REINFORCE++ deserves separate attention because it brings the issue of metrics to the global level.</strong> It is no longer merely a matter of baseline, but rather: whether the scale of advantage should be defined in the prompt or in the whole of the bat.<a href="https://arxiv.org/abs/2501.03262">REINFORCE++</a></li>
<li><strong>If you read the new RL alignment paper later, the first thing to ask is not what it is called PO, but five questions: where did you get it from, where did you put it, where did you get it, where you did it, what was the ultimate optimization.</strong> This is more useful than the acronym of the back calculation.</li>
</ol>
<h2>Dissociate the concept first: reward, return, baseline, advantage, KL, normalization</h2>
<p>If you do not open these words first, you will almost certainly be confused.</p>
<h3>Reward: Raw feedback, not necessarily suitable for training</h3>
<p>The most original source of feedback for the mission is reward. In LLM paired up, it could come from:</p>
<ul>
<li>Scores of reward models;</li>
<li>(a) The determination of the correctness of the rule certifier;</li>
<li>Step points of the process reward model;</li>
<li>Evaluation of LLM as Judge;</li>
<li>Signs of success/failure in the return of the external environment.</li>
</ul>
<p>But from the history of RlHF,<strong>Reward is not usually used directly.</strong> It also needs to be combined with mechanisms such as KL restraint, length punishment, and cessation of punishment to become a signal of excellence. While the quality of reward itself is critical, the processing of signals at the RL algorithm level is equally a major investment to make rewards available for training.</p>
<h3>Return: Rewarding the future and the return</h3>
<p>The strategy gradient is not just about the immediate reward, but about whether the current action will bring the rewards behind it. The most simple cumulative return is:</p>
<p>&#36;&#36;
G_t = \sum_{t&#39; = t}^{T} \gamma^{t&#39; - t} r_{t&#39;}
&#36;&#36;</p>
<p>If there's only a final reward for an LLM mission, this formula means:<strong>Although the reward was given at the end, the front token still gets the training signal through the cumulative return.</strong></p>
<p>The idea is simple, but it is the basis for retrofitting the thin incentive to a token level signal for optimization. And then, with the return, the next step is to update the parameters with it -- The simplest thing is that REINFORCE is going to be: &#36;G_t&#36; Multiply the gradient of log probability, high return token increases the probability, low return pressure lower the probability. But this rough distribution also leads to a lack of quality of the signal itself (the whole trajectory is randomly stacked), a wide range of differences, and each step later — baseline, advantage, nonmalization — is compensating for this rough signal.</p>
<h3>Baseline: not an additional incentive rule, but a reference to the reduction margin Yes</h3>
<p>A number of beginners understand the baseline as another incentive function, but more precisely it is more like a statistical reference system. You're not asking how much you scored this time, but how much you're asking how much higher this time than usual.</p>
<p><a href="https://www.jmlr.org/papers/volume5/greensmith04a/greensmith04a.pdf">Greensmith et al. 2004</a> This is clearly illustrated: the main function of Baseline is to do control variate, which is the drop margin, not to change the gradient expectations. The simple reward is of limited significance, and the update is on advantage, and the baseline is on it.</p>
<h3>Advantage: The real newer is not usually the reward itself</h3>
<p>Once the baseline is introduced, the trainers are really interested in not just nudity but in advantages:</p>
<p>&#36;&#36;
A_t = G_t - b_t
&#36;&#36;</p>
<p><strong>In introducing the policy gradient method for baseline / advantage, the updated signal is usually not a nudity score, but rather an advantage over the reference level.</strong></p>
<p>In the PPO era, the most common method of estimation is that of the old vantage. <strong>GAE（Generalized Advantage Estimation）</strong>I'm sorry. The idea is to see only one step of the TD error equation but rely heavily on clitic. The Monte Carlo is no bias but the difference is big. GAE goes through one parameter. &#36;\lambda&#36; Inserting values between the two, and balancing them.</p>
<h3>KL: It's not an attachment, it's a behavioural boundary.</h3>
<p>Another particularly easy-to-penetrating target in RLF is KL. Many people would understand it as a simple addition to the regularizer, but from the early RLHF work, KL's role is closer to the behavioral boundary:</p>
<ul>
<li>I'm gonna tell you where you want to go.</li>
<li>KL tells you not to deviate from the reference model too far.</li>
</ul>
<p>Without this layer of boundary, reward model can be easily over-optimised. In the LLM scenario, it may be behavioural drift, preference for degradation, speculation in format or length, degradation of capacity, or, more typically, rewarding. If understood from an optimised perspective, it can be seen as a strategy that moves away from the high-quality areas of behaviour represented by reference mode; but it is not the only mechanism, nor is it necessary to attribute all problems to catastrophic oblivion or to a lost base.</p>
<p>It's not KL that needs to be watched here, it's KL, the super-parameter itself.<strong>Optimizing pressure</strong>I'm sorry. The same reward signature may be reliable in light reranking or short-wheeling preferences; once you enter the line RL, multi-spectrum sample, low KL or high weight rewarding, the border of this signal is detected more frequently. DPO, rejection sampling, online RL, use different ways of rewarding, but they may all increase the selective pressure on proxy signal defects; the real risk is that strong online optimization continues to target a blind area as a signal.</p>
<p>So the point of rewarding is not that "rewarded is wrong" but:<strong>The policy explores beyond the definition and generalization of the reward signature, but the optimizer continues to use this signature as a target.</strong> This is first KL/ Optimised Risk Volkage; more systematic uncomfort to Title III. Training monitoring, KL, reward distribution, human rating or validation indicators, and length of behaviour, should be seen together; average reward is easy to miscalculate as an improvement in capability.</p>
<h3>It determines who the model compares with.</h3>
<p>Normaration looks like a project detail, but it determines the scale of comparison. Because it determines the scale at which the above- and below-average are defined:</p>
<ul>
<li>Is it more than average for a sample of the same prompt?</li>
<li>Or is all the whole of the bats above average?</li>
</ul>
<p>This difference, the center of the GRPO divide with REINFORCE++.</p>
<p>And the next general map is the first one I think I've ever been able to build.</p>
<table>
<thead>
<tr>
<th>Link</th>
<th>What's it answering?</th>
<th>Common objects</th>
<th>Representative practice</th>
<th>Common failure patterns</th>
</tr>
</thead>
<tbody><tr>
<td>original</td>
<td>What's the better answer?</td>
<td>R.M., Verifier, Environmental Feedback</td>
<td>Awarding model scores, rule validation, process awards</td>
<td>Reward noise, mislabelling, non-prescriptive.</td>
</tr>
<tr>
<td>Binding Item</td>
<td>How far away can we go to the reference strategy?</td>
<td>KL, length constraints, end of punishment</td>
<td>KL Penalty, length/cut-off penalty</td>
<td>Rewarding, shifting style</td>
</tr>
<tr>
<td>credit assignment</td>
<td>How do we get the final signal back to the front?</td>
<td>return、token reward</td>
<td>Cumulative return, end end reward refill</td>
<td>Scatter, delay, long-range noise</td>
</tr>
<tr>
<td>baseline / advantage</td>
<td>How much higher than the baseline this time?</td>
<td>critic、group mean、leave-one-out</td>
<td>GAE、leave-one-out、group mean</td>
<td>High variance, baseline distortion</td>
</tr>
<tr>
<td>normalization</td>
<td>Where do you define the margin of advantage?</td>
<td>local group、global batch</td>
<td>group std、batch std、z-score</td>
<td>Local explosions, cross-prompt.</td>
</tr>
<tr>
<td>policy update</td>
<td>How to hold each parameter up to date</td>
<td>actor / old policy / reference policy</td>
<td>PPO clip、KL loss、critic-free update</td>
<td>Tactical shakes, training unstable.</td>
</tr>
</tbody></table>
<p>If these layers are separated, much of the debate that follows will be clear. To be honest, many paper-based rewards, the real ones are the fourth and fifth lines, and this should be the algorithm itself.</p>
<p>We'll use a line to tie the whole chain of signals:</p>
<pre><code>raw reward → +KL约束 → return(分配到token) → -baseline → advantage → normalize → ×∇log π → 更新参数
</code></pre>
<p>The difference between each algorithm in the end can be compared to the line by which it makes a different choice.</p>
<h2>First generation answer: how the PPO RLHF spell reward model, critic and KL as a training system</h2>
<p><a href="https://arxiv.org/abs/1707.06347">PPO</a> It was not originally designed for LLM, but it gave RLHF a particularly important project template:</p>
<ul>
<li>An estimate of the size of the chemical;</li>
<li>Do not over-renew with clipping control tactics... Specifically, the PPO is usually used &#36;\epsilon \approx 0.2&#36; (b) Crop the probability ratio of old and new strategies, remove incentives to continue moving in a more favourable direction, and prevent single updates from taking too many steps;</li>
<li>Provides a stable reference to training with old policy/reference policy.</li>
</ul>
<p>Here we are. <a href="https://proceedings.neurips.cc/paper/2020/file/1f89885d556929e98d3ef9b86448f951-Paper.pdf">Learning to Summarize from Human Feedback</a> and <a href="https://arxiv.org/abs/2203.02155">InstructGPT</a> This RLHF route, reward Model and PPO are beginning to really integrate. A good enough mind model is:</p>
<p>&#36;&#36;
R(x, y) = r_\phi(x, y) - \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\mathrm{ref&#125;&#125;(y \mid x)}
&#36;&#36;</p>
<p>It does not have to be understood as a uniform formula that is used word for word in all the papers, but it captures the spirit of the classic RLHF:</p>
<ul>
<li>Part I &#36;r_\phi(x, y)&#36; (b) Expressing reward models or mission feedback;</li>
<li>The second part shows the cost of deviating from the reference strategy.</li>
</ul>
<p>This is a crucial step because it shows that<strong>Classic RLHF from the beginning making composite reward designs</strong>I'm sorry. And not all of them, KL, and not all of them, together, constitute a real goal of optimization.</p>
<p>The role of clitic here is also important. It is not telling the model what is true, but it is estimating as steadily as possible the advantage so that training does not spread out because the reward noise is too loud. This is the first generation answer for the PPO RLHF:</p>
<p><strong>Provide direction with reward Model, boundaries with KL, low differential advantage estimates with critic, and a PPO update mechanism to stabilize the system.</strong></p>
<p>The program is strong, but it is not light:</p>
<ul>
<li>Structure heavy - under LLM scenario, clitic usually shares backbone with an actor plus a value head, plus reference policy and reward model, training with the need to maintain a network of four large models at the same time, with almost doubling of the amount of presence and calculation;</li>
<li>When critic is bad, it becomes a new source of error;</li>
<li>The training formula is complex (learning rate, GAE parameters, clipping range, KL coefficients).</li>
</ul>
<p>And it's because of these problems that the back of the critic-free route is back on the rise.</p>
<h2>Sideline: DPO why is not the main line of the article</h2>
<p><a href="https://arxiv.org/abs/2305.18290">DPO</a> A key relationship was found: the most effective strategy is closed to hidden rewards, with KL constraints, so that it can be directly optimized in classification on preferences, bypassing online rollout, critic and advantage estimates. It makes a lot of people realize for the first time that<strong>Alignment does not have to be done online through RLs.</strong>I'm sorry. But it's because it's bypassed. <code>reward -&gt; advantage</code> This chain, rather than continuing to rewriting it, is used here as a cross-reference. If you're concerned about the online RL itself, how to handle the reward signal, the main line is still PPO, GRPO and REINFORCE++.</p>
<h2>clitic-free main line: comparative advantage within the group from the REINFORCE prototype to GRPO</h2>
<p>Once you remove the critic, the problem doesn't disappear, just reposition it on the table. The new issue becomes:</p>
<ul>
<li>No, no, no, no, no. Where's the baseline from?</li>
<li>No GAE. What do you think?</li>
<li>No critic to smooth your scale, nomalization.</li>
</ul>
<h3>REINFORCE: Minimum prototype, directly passing the results back to the trajectory</h3>
<p><a href="https://doi.org/10.1007/BF00992696">Williams 1992</a> The ReINFORCE is the starting point for all of this. Its most simple new instinct is that the high return trajectory rises logprob and the low return trajectory is down.</p>
<p>&#36;&#36;
\nabla_\theta J(\theta) \approx \sum_t G_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)
&#36;&#36;</p>
<p>On LLM, this can be understood in a visual sense as the token option that makes up a whole paragraph that achieves a high final score should be more easily sampled again in a similar context.</p>
<p>REINFORCE's value is not that it can still be used today to train large models, but that it makes the point clear:<strong>Strategic learning is the reverse attribution of results to the whole sample trajectory.</strong></p>
<p>But it also immediately exposed the oldest problem: the square difference is too wide, the fraction itself has noise, and the allocation token will introduce more noise later; training will often be difficult to stabilize if the baseline is not reduced, but rather if absolute scores are directly influenced by the output probability. The simple baseline REINFORCE is rarely directly used for modern LLM alignment, but the REINFORCE-style critic-free method is re-enacted and continues to be adapted on this very prototype of the idea.</p>
<h3>GRPO: Push local relativization to group average + group standard Bad</h3>
<p><a href="https://aclanthology.org/2024.acl-long.662/">Back to Basics</a> This kind of work re-places the REINFORCE style approach back to the LLM RLHF discussion center, and reminds us that the critic-free is not going to baseline, but instead moving baseline from value head to sample statistics. For example, a multiple sample of the same question gives the average of the other answers under the same prompt as the reference system. GRPO continues the road: it not only subtracts the group average but divides it by group deviation.</p>
<p><a href="https://arxiv.org/abs/2402.03300">DeepSeekMath / GRPO</a> The typical way to sample the same prompt &#36;k&#36; One answer, a set of rewards. &#36;r_{1:k}&#36;and then construct:</p>
<p>&#36;&#36;
A_i = \frac{r_i - \mathrm{mean}(r_{1:k})}{\mathrm{std}(r_{1:k}) + \epsilon}
&#36;&#36;</p>
<p>The idea behind this is intuitive:<strong>For the same topic, what is worth learning is not absolute scores, but advantages relative to the level within the group.</strong> If you take it apart, GRPO has done three things at the same time:</p>
<ul>
<li>Group means baseline, which no longer compares with zero points or global averages, but rather with the average of candidates for the same subject;</li>
<li>The group std is local networking, the margin of advantage is determined by this group of candidates of its own;</li>
<li>Multiple sampling of the same subject defines comparative space, and the training signal first answers “Who among these candidates is more worthy of increasing probability”.</li>
</ul>
<p>This is the most critical shift in the critic-free phase:<strong>Less training on a critic does not mean that the advantage improvement is gone, but it is being pushed into the statistical structure of the Rollout group.</strong> You no longer need a value head, and you have reduced a possible training estimate; the cost is more reliance on the quality of the same subject multiple samples, more reliance on the verifier / reward model to create differences within the group and more reliance on the group statistics themselves to be sufficiently stable.</p>
<p>This is effective in the case of tasks such as mathematical reasoning, which are so diverse and clearly sequenced as to the same subject. And that's why GRPO is fast becoming an influential representational method in the reasing scene. It is particularly appropriate for training settings that can generate multiple candidates, reuser certifications or incentive models for the same subject matter.</p>
<p>But if you look at the reward-signal perspective, the GRPO also exposes the problem of local relativization:</p>
<ol>
<li>The group is too hourly, the average and standard deviations are per se unstable, and the advantage is heavily dependent on the candidate for this sample.</li>
<li>If the reward in the group is almost the same, group std will get the signal right. &#36;\epsilon&#36; / Clipping is very sensitive; in some settings close to zero signals, in others the noise may be magnified.</li>
<li>This prompt is better than the candidate, not necessarily equal to "better distribution of the data as a whole".</li>
</ol>
<p>I'd rather understand these questions as the historical value of GRPO: it's not the end of the problem, it's the end of it. <strong>local normalization</strong> This thing pushed into the middle of the stage. The subsequent REINFORCE++, which follows this line, continues to follow.</p>
<h2>REINFORCE++: From local  Normalization Debate to Global Scale</h2>
<p>If the key to GRPO is local relativization, then <a href="https://arxiv.org/abs/2501.03262">REINFORCE++</a> The key is that it is:<strong>It's not just the baseline itself, it's a measure of partial integration.</strong></p>
<p>In the author ' s narrative, there are at least three risks to prompt-level regulation such as GRPO: small local groups, uneven averages and standard differences; molecular and denominators from the same group, statistically linked; models learn more like winning themselves in the group than necessarily forming a global scale across the prompt.</p>
<p>So REINFORCE++ gives a response not to replace a new model, but to pull the range of the local group from the global watch:</p>
<p>&#36;&#36;
A^{\text{norm&#125;&#125; = \frac{A - \mathrm{mean}<em>{\text{batch&#125;&#125;(A)}{\mathrm{std}</em>{\text{batch&#125;&#125;(A) + \epsilon}
&#36;&#36;</p>
<p>This step seems to be just a change of location, but it recast the training signal for a comparator:</p>
<ul>
<li>In GRPO, the model is mainly a comparison with other candidates on the same subject;</li>
<li>In REINFORCE++, the model is compared with the distribution of the advantage of the whole bat.</li>
</ul>
<p>The normal REINFORCE++ and REINFORCE++-Baceline in the paper can be understood in this framework: the former places greater emphasis on the integration of the same topic into the "advantage" at the catch scale, even without multiple samples; the latter retains the average value impairment of the group, allowing the same issue to continue to be taken together, but moves the std from the local group to the catch level, avoiding using the same group to decide both "who is higher" and "how much higher".</p>
<p>What is more worth remembering here is not terminology, but three dimensions of decision-making:</p>
<ul>
<li>- Where did you get that?</li>
<li>Where does that happen?</li>
<li>KL is either blending into advantage, or alone into loss.</li>
</ul>
<p>And that's exactly the value of REINFORCE++ in this main line: it's not adding a complex branch, but it's continuing the core debate of the clitic-free debate from “baseline where” to “where does the scale define”?</p>
<h2>End to end: how a signal from prompt was converted from reward to advantage</h2>
<p>A number of concepts have been removed, and the chain is being followed by a specific numerical example.</p>
<p>Suppose there's a prompt:&quot;One sentence explains quantum tangles.&quot;We took it three times, and the reward model scored it:</p>
<table>
<thead>
<tr>
<th>Answer!</th>
<th>RRM Score &#36;r_i&#36;</th>
</tr>
</thead>
<tbody><tr>
<td>Answer A.</td>
<td>8.2</td>
</tr>
<tr>
<td>Answer B.</td>
<td>6.5</td>
</tr>
<tr>
<td>Answer C.</td>
<td>9.1</td>
</tr>
</tbody></table>
<p><strong>Return/Return</strong>: In the traditional RL intuition, if only the final reward, you can put &#36;G_t = \gamma^{T-t} \cdot r_T&#36; Different steps are allocated. But in the common LLM RLHF / GRPO realization, it is more common to convert the security-level response to the same command advantage and broadcast it to the updated version of the protocol tokens, with token-level KL or radio constraints. The concept of return is retained here to explain "how the result is attributed in reverse to the trajectory" and not that all LLM realizations are allocated on a discount basis.</p>
<p><strong>Baseline (average / leave-one-out style)</strong>: When looking at answer A, a reference system can take the average of the remaining answers, baseline = &#36;(6.5 + 9.1) / 2 = 7.8&#36;I'm sorry. If written as GRPO Group Mean, the average of three answers is used directly &#36;7.93&#36; As this prompt's baseline. The details are different, but in common: baseline is no longer from critic, but rather from the same subject matter sample.</p>
<p><strong>Advantage</strong>: If written in leave-one-out, answer A &#36;A = 8.2 - 7.8 = +0.4&#36;(lightly better than reference level for the same subject; small increase probability); answer B &#36;A = 6.5 - 8.65 = -2.15&#36;(Specificly different from other samples, press down probability). If GRPO is used, the average of the group is then reduced and then the standard deviation within the group is standardized.</p>
<p><strong>Normalization (GRPO style vs REINFORCE++ style)</strong>：</p>
<ul>
<li>GRPO: With these three answers to their average and standard differences z-score,&#36;A_i^{\text{norm&#125;&#125; = (r_i - 7.93) / 1.07&#36;I'm sorry. This is the local scale of the prompt, which is not stable when there are only 3 samples in the group.</li>
<li>REINFORCE++: Calculate the average and standard deviation of this three responses with the whole bat (e.g. 256 prompt x 3 = 768 answers) and the average and standard deviations, with a more stable scale.</li>
</ul>
<p><strong>Policy Update</strong>: Finally, REINFORCE-type update multiplied by &#36;\nabla_\theta \log \pi_\theta&#36;— The probability of a token answering A is slightly increased (advantage positive), the probability of a token answering B is significantly lowered (advantage negative), and the answer is C obtains the greatest positive update.</p>
<p>That's the whole thing. <code>reward → return → baseline → advantage → normalization → policy update</code> Chain. All the differences in the algorithms that follow can be returned to this example, and see where they make different choices.</p>
<h2>Read the paper checklist: which five interfaces should be asked first</h2>
<p>If only one reusable frame is left at the end of the article, this section is probably the case.</p>
<p>See the new RL alignment algorithm later, so don't hurry up and remember what it's called. The following five questions are first asked:</p>
<table>
<thead>
<tr>
<th>Problem</th>
<th>What does it really decide?</th>
<th>Typical answer.</th>
</tr>
</thead>
<tbody><tr>
<td>Original where did you come from?</td>
<td>What are you perfecting?</td>
<td>Incentive models, rule-certifiers, process rewards, environmental feedback</td>
</tr>
<tr>
<td>KL writes "reward" or "single"?</td>
<td>Whether or not to bind directly to the advantage construction</td>
<td>Classic RLHF for the former, GRPO / REINFORCE++ for some variants</td>
</tr>
<tr>
<td>- Where's the baseline from?</td>
<td>How do we lower the difference?</td>
<td>critic、moving average、leave-one-out、group mean</td>
</tr>
<tr>
<td>What range does that have to be?</td>
<td>Who are the models comparing themselves to?</td>
<td>local group、global batch、running statistics</td>
</tr>
<tr>
<td>What was the final update?</td>
<td>What is the object that is really being optimized?</td>
<td>sequence reward、cumulative return、normalized advantage</td>
</tr>
</tbody></table>
<p>These five issues can almost reorder all mainstream approaches:</p>
<ul>
<li><strong>PPO RLHF</strong>: critic + KL + PPO update, reward mode signal;</li>
<li><strong>DPO</strong>: Just bypass the online advantage assessment;</li>
<li><strong>GRPO</strong>: group mean baseline + group std, defining signals as local comparative advantage;</li>
<li><strong>REINFORCE++</strong>: Move the integration scale to the catch/ global level.</li>
</ul>
<p>Ultimately, the discussion is not about complete rewarding, but rather about rewarding signal shaping / rewarding consummation: how to be restrained, compared, normalized and interpreted before and after the reward signal enters the optimist. It includes at least:</p>
<p><strong>How the original feedback was processed through a range of statistics and optimizations was eventually interpreted as a powerful signal that could be trained in strategy.</strong></p>
<p>In other words, the main line is... <code>reward-to-advantage design</code>: nominally <code>reward</code>And actually is improving the RL algorithm itself.</p>
<h2>Summary</h2>
<p>If LLM alignment is seen as a series of algorithms, then what happens over the years is like changing abbreviations; but if it is seen as a signal chain from reward to advantage, many things suddenly become consistent.</p>
<ul>
<li>The PPO RLHF addresses how to combine reward Mode, critic and KL into a system template that can be used on a large scale;</li>
<li>DPO tells us that there are some problems that can be bypassed from the online RL;</li>
<li>GRPO Note: After no critic, baseline and numaration are transferred to the group for statistical purposes;</li>
<li>REINFORCE++ further narrows the issue to a specific debate:<strong>The measure of advantage should be defined locally or globally.</strong></li>
</ul>
<p>This line of study can be summed up in one sentence:</p>
<p><strong>At least on the RL algorithm line, the focus of the improvement is not on constantly creating new incentives, but on how to be explained before and after rewrite the reward signal into the optimizer.</strong></p>
<p>If the chain is sufficiently mature, the subsequent bottlenecks will shift further towards rollout selection, sample efficiency and system ingestion. The training system problems of the PODS are placed in the appendix; the main line goes back to the next step: how did the reward actually produce itself?</p>
<h2>Appendix: PODS and the side support of the training system - not all rollout is worth updating</h2>
<p><a href="https://arxiv.org/abs/2504.13818">PODS</a> It's the side of the training-system, not the main line in the next chapter of this series. It's interesting, not to rewrite, not to rewrite, but to ask a question closer to the training system itself:<strong>Which rollout is worth sending in an expensive parameter update.</strong> On the online RL, generating rollout tends to be easier to create in parallel, and the real cost is the reverse transmission, cross-calorized and optimised state maintenance, so the problem starts with&quot;How do you define the signal?&quot;Turn&quot;Which samples are worth consuming the updated budget&quot;。</p>
<p>It is also a simple idea: Mr. Rollout becomes more, sifts less informative samples, and only gives more training-worthy subsets to the original GRPO/PPO target. It is not just who is the top, but who is better able to pull the training signal; the max-variance down-sampling highlighted in the paper prefers to retain a portion of both high and low incentive samples instead of a top-k-only reservation.</p>
<p>If you return to the main line of this article, the PODS is not about the chain itself, but rather how the bottlenecks continue to shift to rollout selection, sample efficiency and system ingestion, when the chain is mature enough. The main line remains how the signal is explained, and the PODS discussion is about what samples are worth updating the model after the explanation has been completed.</p>
<h2>References</h2>
<h3>Foundation theory</h3>
<ul>
<li><a href="https://doi.org/10.1007/BF00992696">Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning</a></li>
<li><a href="https://proceedings.neurips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation">Policy Gradient Methods for Reinforcement Learning with Function Approximation</a></li>
<li><a href="https://www.jmlr.org/papers/volume5/greensmith04a/greensmith04a.pdf">Variance Reduction Techniques for Gradient Estimates in Reinforcement Learning</a></li>
</ul>
<h3>Classic RLHF / PPO</h3>
<ul>
<li><a href="https://arxiv.org/abs/1707.06347">Proximal Policy Optimization Algorithms</a></li>
<li><a href="https://proceedings.neurips.cc/paper/2020/file/1f89885d556929e98d3ef9b86448f951-Paper.pdf">Learning to Summarize from Human Feedback</a></li>
<li><a href="https://arxiv.org/abs/2203.02155">Training language models to follow instructions with human feedback</a></li>
</ul>
<h3>clitic-free and homogenization</h3>
<ul>
<li><a href="https://aclanthology.org/2024.acl-long.662/">Back to Basics: Revisiting REINFORCE-Style Optimization for Learning from Human Feedback in LLMs</a></li>
<li><a href="https://arxiv.org/abs/2402.03300">DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models</a></li>
<li><a href="https://arxiv.org/abs/2501.03262">REINFORCE++: Stabilizing Critic-Free Policy Optimization with Global Normalization</a></li>
</ul>
<h3>System side support</h3>
<ul>
<li><a href="https://arxiv.org/abs/2305.18290">Direct Preference Optimization: Your Language Model is Secretly a Reward Model</a></li>
<li><a href="https://arxiv.org/abs/2504.13818">Not All Rollouts are Useful: Down-Sampling Rollouts in LLM Reinforcement Learning</a></li>
</ul>
