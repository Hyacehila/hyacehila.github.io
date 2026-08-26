---
title: 'Practical Handling of Class Imbalance: When and How to Act'
title_zh: 样本不均衡的实践处理：何时处理、怎么处理
date: 2026-03-14 20:00:00 +0800
categories:
- Machine Learning
- Classical Machine Learning
tags:
- Sampling
- Class Imbalance
author: Hyacehila
mathjax: true
hidden: true
excerpt: 'A step-by-step guide to handling class imbalance: decide whether treatment is needed, fix the evaluation, start
  with threshold moving and class weights, and only then resample; covers rare events, the pretrained-model era, and common
  pitfalls.'
description: 'A step-by-step guide to handling class imbalance: decide whether treatment is needed, fix the evaluation, start
  with threshold moving and class weights, and only then resample; covers rare events, the pretrained-model era, and common
  pitfalls.'
excerpt_zh: 遇到样本不均衡时按步骤处理：先判断要不要处理，再守住评估，然后从阈值移动和权重开始，最后才改数据；附稀有事件、预训练时代与常见误区。
permalink: /blog/2026/03/14/training-imbalance-solutions/
lang: en
translation_key: 2026-03-14-training-imbalance-solutions
translation_status: machine
translation_source_hash: 46f710562b09b9d50cd0977784f04d23b4b726bef42e4863e05543a7a6e1d007
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<p>And the first thing that happens to people is that they're not even.&quot;The few are too few.&quot;I'm sorry. But if the understanding is to stop at this point, it is easy to follow a series of costly operations that do not necessarily serve business objectives. The real available frame is<strong>Just follow the sequence and take a few steps.</strong>First, to decide whether to process it, then to hold the assessment, then to start with the cheapest means, and then to consider changing the data.</p>
<p>The questions in this article can also be addressed<a href="/en/blog/2024/03/28/machine-learning-introduction-supervised-learning/">Introduction to Machine Learning: Monitoring Learning and the Bayesian Approach</a>How the concept of a relatively close read together is developed in different contexts.</p>
<h2>Step one: to decide whether to deal with it or not.</h2>
<p>Before addressing imbalances, make a shift:<strong>Don't ask first.&quot;Should I, SMOTE?&quot;, and ask&quot;I'm going to optimize ranking, probability, or the final benefit of decision-making.&quot;。</strong> These three objectives are not evenly priced in the uneven scenario:</p>
<ul>
<li>Just care.<strong>Sort</strong>(Is the formality above the negative ones: with indicators such as AUROC/PR-AUC, many times without doing anything;</li>
<li>Care.<strong>Probability</strong>(0.8 of output is really close to 80%): a calibrated probability output is required;</li>
<li>Care.<strong>Final proceeds</strong>(A different cost of misreporting than of underreporting): the movement of thresholds and sensitivity of costs are key.</li>
</ul>
<p>- Put it on.&quot;Samples are uneven&quot;Dismantling into three phenomena that are often mixed should be approached differently:</p>
<ol>
<li><strong>Uneven training cluster categories</strong>: Most common scenes, many of the head type samples, few of the tail type.</li>
<li><strong>Estimates of probability of rare events</strong>: a small probability of event (failure rate, risk of complications, failure rate) is to be estimated steadily.</li>
<li><strong>Pre-deployment deviation</strong>: Training sets differ from test sets, type a priori (label Shift) for online traffic.</li>
</ol>
<h3>When is it really not necessary to deal with it?</h3>
<ul>
<li><strong>The character is sufficiently different.</strong>: Standard models can be learned well when a few categories and most categories are virtually non-overlapping in characteristic space, even at a ratio of 110000;</li>
<li><strong>There's enough of them.</strong>: When the total is a million, the tail class is 0.1% and there are thousands of them, not bottlenecks;</li>
<li><strong>The tree model can carry a part of it.</strong>XGBoost / LightGBM, which enhances trees, is highly adaptive to moderate imbalances;</li>
<li><strong>Business indifferent tail-type performance</strong>: When you only care about the whole sort or the type of accuracy of the head, you deliberately raise tail and lower the main target.</li>
</ul>
<p><strong>Conclusion: First, to verify whether the imbalance is really hurting the indicators of your concern, then to decide to stay.</strong> Sometimes the problem is not proportional, but the quality of the feature or the noise of the label.</p>
<h2>Step two: hold the evaluation. Don't be fooled by Accuracy.</h2>
<p>The blogger says that the government is not taking the initiative.<strong>The quality of ranking, probability and decision-making must be separated Talk.</strong>I'm sorry. Details of the algorithms and selection of specific indicators are available.<a href="/en/blog/2025/10/02/supervised-learning-model-evaluation/">Monitoring of learning performance assessment</a>Here is the conclusion of practice:</p>
<ul>
<li><strong>Under the strong imbalance, the PR curve tends to reflect actual pressure more than ROC</strong>: ROC takes in a lot of real negatives, and many models are going to look like&quot;It's nice.&quot;;PR Space more directly exposes the misstatement of costs.</li>
<li><strong>At least report concurrently on the selection of indicators, PRUC, per-class and thresholds</strong>, multi-classification long end with the macro indicator and head/tail layer results. The only way to get a report is to be too optimistic.</li>
</ul>
<h2>Step 3: Starting with the cheapest - threshold movement and weight</h2>
<p>As long as the model gives a reasonable probability sequence, many questions simply do not require data modification:</p>
<ul>
<li><strong>Threshold moving</strong>: Move decision-making thresholds from default 0.5 to business cost equivalent. The most cost-effective first step is not to repeat training or disrupt training distribution. Attention. <strong>F1 Best Threshold is not&quot;Common threshold&quot;</strong>It has some anti-intuitive nature - the threshold is decision-making, not default.</li>
<li><strong>Category weights</strong>XGBoost <code>scale_pos_weight</code>LightGBM <code>is_unbalance</code> / Category weight, equal to the weight given to the sample, to preserve the original distribution and to avoid pre-testing the test set.</li>
<li><strong>Cost-sensitive learning</strong>: If the cost of the omission and the misrepresentation are different, the default 0.5 threshold is not natural. The absence of a high-risk transaction in the fraud detection is far more expensive than reviewing several normal transactions; medical early screening is more likely to be called back in exchange for false reporting. The costs are directly written into the loss function or decision threshold and the weight sample is cleaner.</li>
</ul>
<h2>Step 4: Change data before it is enough - re-sampling</h2>
<p>The first two steps are not a solution (extreme imbalance, small categories surrounded by most categories, high cost asymmetries) before data are moved. Three types of approach:</p>
<ul>
<li><strong>Undesired sampling (underersampling)</strong>: Remove some negatives that bring the number of positives and negatives closer. The disadvantage is that most types of information are lost, and the improved version is <strong>EasyEnsemble</strong>• Disaggregation of the inverse into several sets for different learning devices, each of which is unsampled, but information is not lost across the board.</li>
<li><strong>Oversampling</strong>: Add some positive examples. Note that the initial rule cannot be simply copied, otherwise it would cause serious overload; normally, the plug is used to generate additional rule.<strong>SMOTE</strong> It is the insertion of composite samples of values in a few local adjacent areas. It works if there are plugged structures in a few types of neighborhoods. - I'm sorry. If a few categories are caught in most categories or are labelled with noise, synthetic samples only contaminate the boundary.</li>
<li><strong>Integrated + Sample</strong>: Embedding samples into integration processes, such as SMOTEBoost, which is more focused on hard-to-train tail samples per round.</li>
</ul>
<p><strong>The most critical sequence discipline: first split the training/test set and then sampled only within the training set.</strong> SMOTE, then split, will allow training and testing to share local neighbourhood structures, and the results are very high -- the most classic data leak.</p>
<h2>Special target: Probability is what rare events require.</h2>
<p>If the target is not...&quot;Distinguishing the positive and negative.&quot;and estimates the probability of rare events (failure rate, failure rate, risk of complications),<strong>Standard models systematically underestimate the probability of events when the normal pattern is scarce.</strong>— The uneven impact is not only the classification of boundaries, but also the parameter estimates themselves. The blind re-sampling is not the first option at this time:</p>
<ul>
<li>Keep original aforecheck, do it <strong>prior correction</strong>(Assisting the sample back to real a priori);</li>
<li>After training, do it alone.<strong>Probability calibration</strong>（Platt / Isotonic / Temperature Scaling）；</li>
<li><strong>Don't take the rebalancing score directly as a risk probability.</strong>— Oversampling and weighting are all rewritten training a priori, and model output scores may not be interpreted as a real world backsliding probability.</li>
</ul>
<h2>Deep model and pre-training age: Longtail is not that important.</h2>
<p>The core difficulty of traditional long-term learning is that <strong>tail class means unstable</strong>- There are too few samples to learn reliable characterizations. But the massive pre-training model (CLIP, DINOV2, Big Language Model, etc.) changed this premise:</p>
<ul>
<li><strong>tail class means instability is mitigated by pre-training</strong>(a) Pre-training has learned fairly common expression space and there is a good enough starting point for less downstream samples of the tail type;</li>
<li><strong>Zero samples / Few samples capacity directly bypassed long tails</strong>: Visual language models can even describe text description of the type of training that has never been seen before in the training.</li>
</ul>
<p>So in the scene that I'm watching,<strong>Longtail study is not so important anymore.</strong>I'm sorry. If the depth model still has to address the imbalance, the point is:&quot;The decorating means learning and categorizing. Set&quot;And focus on calibration, not stacking of Focal Los, LDAM, such weighting techniques.</p>
<h2>Common error areas (checklist)</h2>
<ol>
<li><strong>I took samples before the splitting.</strong>- Data leak. Always cut first, then take samples only inside the training set.</li>
<li><strong>Only Accuracy or AUROC</strong>— may be overly optimistic; at least report both the provalence, PR indicators, per-class indicators.</li>
<li><strong>Telling stories on the balance test set, but defaulting on the actual deployment a priori.</strong>- Balanced benchmark is not a realistic a priori.</li>
<li><strong>Consider the increase of tail-class precision as a probability-based increase</strong>— Improved classification and improved probability must be tested separately.</li>
<li><strong>Make imbalance the only problem, ignoring overlaps and noises.</strong>— When a few categories are mixed in most categories or with noise, the weighting is only more rigorous in the learning of noise.</li>
</ol>
<h2>Page Selection</h2>
<table>
<thead>
<tr>
<th>scene</th>
<th>First option</th>
<th>What can be folded?</th>
<th>Don't come up here and do it.</th>
<th>Suggested indicators for reporting</th>
</tr>
</thead>
<tbody><tr>
<td>Small sample table II classification</td>
<td>Layer-specification, class weight, threshold-adjusted</td>
<td>Moderate sampling or EASYEnsemble in training set</td>
<td>SMOTE before split; only Accuracy</td>
<td>PR-AUC、Balanced Accuracy、MCC</td>
</tr>
<tr>
<td>Raise Tree-Major Data Task</td>
<td>Verify first whether the processing is really needed; move the scale pos wait + threshold if necessary</td>
<td>Cost-sensitive, small sample</td>
<td>Blind SMOTE</td>
<td>PR-AUC, per-class, confusion matrix</td>
</tr>
<tr>
<td>Probability of rare events</td>
<td>Keep original aforesee, logical regression / GGBDT+ calibration</td>
<td>Case-control sampling, project control, sootonic/temperature scaling</td>
<td>Make the rebalancing fractions the probability of risk.</td>
<td>Brier、Calibration、PR-AUC</td>
</tr>
<tr>
<td>Wind control / Medical care / required for reliable probability output</td>
<td>Define deployment a priori and cost before training</td>
<td>calibration、threshold moving、label-shift correction</td>
<td>I'll just watch my own, and I'll be on the line.</td>
<td>Calibration curve、Brier、ECE</td>
</tr>
</tbody></table>
<p>Default order of operation:<strong>Define the target first</strong>(Sorting / Probability / Proceeds) <strong>Hold the evaluation.</strong>(Accessorator set close to deployment) <strong>First, make a simple baseline.</strong>(class weight, threshold movement, correct indicators, header correction) <strong>And then the complicated way.</strong>(sampling, weighting) <strong>Last chance of re-entry</strong>( calibration).</p>
<h2>Concluding remarks</h2>
<p>Let's just say the first line:<strong>A truly stable imbalance is not a one-man skill, but a one-man skill.&quot;Training objectives, evaluation indicators, deployment decisions&quot;Three-man consistent design.</strong> In most cases, holding assessments, moving thresholds and re-aligning weights have resolved most of the problems and changing data is always at the end. When you see the imbalance as a whole link from data distribution to deployment decision-making, the method is easier to choose.</p>
