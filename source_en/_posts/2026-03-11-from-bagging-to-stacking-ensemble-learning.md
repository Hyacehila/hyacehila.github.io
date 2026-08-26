---
title: 'From Bagging to Stacking: A Map of Ensemble Learning Methods'
title_zh: 从 Bagging 到 Stacking：集成学习方法图谱
date: 2026-03-11 22:30:00 +0800
categories:
- Machine Learning
- Classical Machine Learning
tags:
- XGBoost
- Machine Learning
author: Hyacehila
mathjax: true
hidden: true
excerpt: Maps Random Forest, GBDT, XGBoost, and Stacking onto the same conceptual frame.
description: Maps Random Forest, GBDT, XGBoost, and Stacking onto the same conceptual frame.
excerpt_zh: 把 Random Forest、GBDT、XGBoost 与 Stacking 放回同一张图里，梳理集成学习的核心脉络。
permalink: /blog/2026/03/11/from-bagging-to-stacking-ensemble-learning/
lang: en
translation_key: 2026-03-11-from-bagging-to-stacking-ensemble-learning
translation_status: machine
translation_source_hash: 36bbf2f33ab5abaa4db4b9284c8ba5451305acc650274c2278ba30675cdad494
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<p>In a previous article on table data and tree models, I have already mentioned Random Forest, GGBT, XGBoost, LightGBM and CatBoost. But if we remember only that “they are a combination of many trees”, one more question that is worth asking is:<strong>Why are they all integrated and the training logic is completely different?</strong></p>
<p>The questions in this article can also be addressed<a href="/en/blog/2024/03/28/machine-learning-introduction-supervised-learning/">Introduction to Machine Learning: Monitoring Learning and the Bayesian Approach</a>、<a href="/en/blog/2024/04/06/advanced-machine-learning-unsupervised-learning/">Machine learning progression: unsupervised and semi-supervised learning</a>How the concept of a relatively close read together is developed in different contexts.</p>
<p>Random Forest's idea is "co-production of diversity, then averages," and the idea of GDDT is "colleges to continue the uneducated part of the previous round." And then back, Stacking doesn't even ask that the base model be the same algorithm, which is more like asking:<strong>Can I train a model, specializing in "When should I trust who?"</strong></p>
<p>This article is about to complete the picture. The whole text is based on a simple route:</p>
<ul>
<li>Avaging / Voting: Minimised Integration Prototype</li>
<li>Bagging: stability through re-sampling</li>
<li>Boosting: Continuously uplifting the model through serial error correction</li>
<li>STacking: Let me get a metamodel to learn the combination rules</li>
<li>DCS / DES: From "overall weights" to "local selection"</li>
<li>Refusal / Dumpbin label: A border technique similar to integration but not standard Ensemble</li>
</ul>
<h2>Why is integration worth learning alone? Speak.</h2>
<p>I'll take two key words to sum up the integration. <strong>Diversity</strong> and <strong>Offset - Specimen</strong>。</p>
<p>First, diversity is an effective precondition for integration. Multiple models do not magically improve on average if they make the same kind of mistakes; what is valuable is that different models are strong and weak in different samples, different characteristic subspaces, different summation preferences, and their mistakes are offset by each other.</p>
<p>Second, many integration methods can be understood by putting back the deviation-variance framework. Intuitively:</p>
<ul>
<li><strong>Bagging</strong> It's more like trying to lower the differences and make the models more stable.</li>
<li><strong>Boosting</strong> More like fixing deviations on a continuous basis, bringing models closer to better functions.</li>
<li><strong>Stacking</strong> Instead of directly corresponding to a single direction, a further layer of combinations is developed on existing models.</li>
</ul>
<p>If you just want to remember one sentence, you can first distinguish between the following:</p>
<ul>
<li><strong>Bagging</strong>: training multiple models in parallel, and then doing average or voting.</li>
<li><strong>Boosting</strong>• Multiple models for symmetrical training, with each round continuing to learn about the shortcomings of the previous round.</li>
<li><strong>Stacking</strong>Training a group of base models and a meta-model to learn how to combine them.</li>
</ul>
<p>So, ILLM is not just a toolbox label, but a way to understand model relationships. Random Forest and GDDT, as seen in the tree model article, are two typical branches of this framework.</p>
<p>This section is not about to go back to the algorithm name, but about the endble. <strong>How to make diversity</strong>And it.<strong>Trying to solve the problem of deviation or deviation</strong>I'm sorry. Once these coordinates are fixed, the subsequent method is easily returned to the same map.</p>
<h2>Starting with simple integration: Avaging and Voting</h2>
<p>Many times, integration learning starts with a very simple question:<strong>If I've got multiple models, can I just put them together?</strong></p>
<p>For return missions, the most natural approach is average (Averaging). If the importance of the different models varies, weighted averages can be used:</p>
<p>&#36;&#36;
\hat{y}(x) = \sum_{m=1}^{M} w_m \hat{y}<em>m(x), \qquad \sum</em>{m=1}^{M} w_m = 1
&#36;&#36;</p>
<p>The hunch behind it is simple: individual models fluctuate, but the average of multiple models tends to be more stable. In practice, in addition to the average, there are also those who use median aggregations when the abnormalities are higher, as it is less sensitive to extreme predictions.</p>
<p>For classification tasks, the corresponding is voting. The most common forms are three:</p>
<ul>
<li>Majority voting: selection of the most numerous votes</li>
<li>Weighted voting: giving more credible models more power Heavy</li>
<li>Average probability: direct average of the type of probability for each model output, with the largest</li>
</ul>
<p>Here's a problem that can be easily ignored:<strong>The output of different taxonomyrs is not necessarily natural comparable.</strong> Some model outputs are probabilities, others are margins, some are just fractions. If these volumes are not on the same scale, the direct weighting and the meaning are often not clear. A more conservative approach would be to harmonize them into comparable probabilities or uniform definitions; where needed, comparability could also be improved by means of probability calibration (e.g. Platt scaling, sootonic revision).</p>
<p>Averaging and Voting seem "very basic," but many of the frameworks that follow contain a step of integration. The difference is just that some methods are only one last step of integration, and others are designed together with the "how the submodels are produced".</p>
<p>Avaging / Voting is the smallest prototype of an ensemble. They show that even if the base model itself is not changed,<strong>As long as it's a good combination, the predictions are more stable.</strong>I'm sorry. But if we want to keep the performance gap open, we need to continue to think: How did these models come to be made?</p>
<h2>Bagging: stability by re-sampling</h2>
<p>The idea of Bagging is:<strong>Since diversity was important, it was a voluntary effort to produce a number of different training data.</strong></p>
<p>The classic approach is Bootstream re-sampling: original training sets are sampled back and multiple, similar sizes are obtained over and over again, but samples form different data sets. The base learners are then trained separately on these data sets and are then put on average or voted on.</p>
<p>The mechanism is effective because it is well suited to the high-variant model. For example, decision trees are often sensitive to minor disturbance in training samples if they are not too restrictive: a slight change in the sample results in a marked change in the split structure of the tree. Bagging used this to grow each tree on slightly different data and eventually evenly reduce these fluctuations.</p>
<p>Bootstream also gave Bagging a practical by-product: extra-pack sample (out-of-bag, OOB). After a round of bootstream samples, it appears that on average about 36.8% of the samples were not taken, and they could be used to make a near-feature set to assess broad-based performance in a rough manner, without necessarily cutting an extra piece of hold-out data.</p>
<h3>From Bagging to Random Forest</h3>
<p>Random Forest can be seen as a reinforced version of Bagging on the decision tree. Normal Bagging has made the difference through "sampling random" and Random Forest is a step forward: When each node is split, the tree is not the most excellent of all characteristics, but the first of these is a randomly selected feature subset, which is then selected from the child's concentration.</p>
<p>So Random Forest introduced two layers of diversity:</p>
<ul>
<li><strong>Sample Random</strong>: Different data subsets per tree</li>
<li><strong>Feature Random</strong>: The candidate characteristics for each node are different</li>
</ul>
<p>This is why Random Forest is often a good place to be used as baseline. Its training can be parallel, its routing is relatively clear and it is generally less sensitive to abnormalities and characterization scaling. In many structured forms, if you haven't figured out whether to go straight to Boosting, Random Forest is always a good first step.</p>
<p>Of course, Bagging has its borders. It is more skilled at “stabilizing unstable models on average”, but if there is a strong systemic bias in individual models themselves, it is often difficult to break this ceiling on parallel averages alone. This is exactly where Boostling's gonna take over.</p>
<p>The key word for Bagging is <strong>Parallel, re-sampling, drop-off</strong>I'm sorry. Random Forest is the most classic and durable route for decision-making tree integration by adding a more random dimension to the characteristic dimension based on Bagging.</p>
<h2>Boosting: Keep the last part of the cycle that didn't work.</h2>
<p>Bagging is concerned with "How to make the model more stable," Boosting is more concerned about:<strong>How to make the model a whole round stronger.</strong></p>
<p>It's a sort of a sequence of base learning devices. The latter cycle of learning machines is not retrained from scratch, but focuses on the parts of the previous cycle that were not yet completed.</p>
<h3>AdaBoost: Start with sample empowerment</h3>
<p>AdaBoost is the most classic introductory model of the Boosting family. It can be written as a combination of weak learners:</p>
<p>&#36;&#36;
H(x) = \sum_{t=1}^{T} \alpha_t h_t(x)
&#36;&#36;</p>
<p>The focus here is not on the formula itself, but on two layers of the “weighted” idea:</p>
<ul>
<li>Increased attention to the next training cycle for the misspectled samples of the previous round</li>
<li>For a weak learning machine that behaves better, give it a higher final combination. Heavy</li>
</ul>
<p>So, AdaBoost's focus is not simply average, but rather,<strong>Let the learning process be remembered.</strong>: The wrong place in front, the focus of the post.</p>
<p>However, AdaBoost is mainly promoting learning through the “sample empowerment”. This is a very good idea, but when the loss function is more general and the task more complex, we would like a more unified view of what “is to be learned in the next round”. This leads to the raising tree and the GDDT.</p>
<h3>Raise Tree & GGBDT: Approaching from error correction to function</h3>
<p>In raising trees, we see the final model as an additional model of multiple trees:</p>
<p>&#36;&#36;
F_M(x) = \sum_{m=1}^{M} T_m(x)
&#36;&#36;</p>
<p>The task of each new tree at this time is not to complete a forecast on its own, but to supplement it with the current model. The problem is:<strong>What should we do with this round?</strong></p>
<p>The answer given by GDDT is to make the negative gradient for the current loss function. At the end of the day &#36;m&#36; Wheel, what we care about is...</p>
<p>&#36;&#36;
r_{im} = -\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]<em>{F(x)=F</em>{m-1}(x)}
&#36;&#36;</p>
<p>When the loss function is a square error, it is degraded into the most familiar “discretion”; and when the loss function is replaced by a more general form of loss, such as a logarithmic loss, a “to-composed negative gradient” becomes a uniform and natural description.</p>
<p>This is a key step, as it moves Boostling from a perspective of how the sample weight changes to a perspective of how the function approaches the target step by step. It is therefore the common denominator of subsequent projects such as XGBoost, LightGBM, CatBoost, etc. The frameworks discussed in the previous tree model article, although they vary in terms of regularization, divisive tactics, type characterization and engineering optimization, remain on this Boostling main line.</p>
<p>The value of GDDT is not understood to be simply an old algorithm, but to see why the most common powerful models in the form modelling are derived from the line of “step-by-step preparation of the gap/gravity”.</p>
<p>The key word for Boosting is <strong>Serial, error-correction, addition model</strong>I'm sorry. AdaBoost expresses this idea with sample weighting, and GDDT further expands it to "A negative gradient for each round of the loss function is prepared". And to understand this, it's a lot better to know.</p>
<h2>Stacking: Let the model learn who to write.</h2>
<p>Bagging and Boosting are often producing a large number of base learning devices, which often come from the same training mechanism. The Stacking is a different angle:<strong>If I had some very different types of strong models, could I train a model that would learn their combination rules?</strong></p>
<p>At this point, the base model is not necessarily a "weak learning machine". They can be logical regression, random forests, XGBoost, neural networks and even rule systems. The focus is not on the weakness of the underlying models, but on whether they have complementary information.</p>
<p>Standard training processes for Stacking are usually:</p>
<ol>
<li>Split the training into one. &#36;K&#36; Off.</li>
<li>For every base model, do it. &#36;K&#36; Training in discounts and generation of off-trading forecasts for each sample (out-of-fold, OOF).</li>
<li>These OOF predictions are combined to form new meta-features.</li>
<li>Train a secondary learning device with meta-Learner.</li>
<li>The projection is made by giving all the underlying models an output before giving them to the meta-model to produce the final projection.</li>
</ol>
<p>Here, OOF is the most critical detail of the Stacking. If the meta-model is trained directly using “projection results from the base model on the training set”, the meta-model sees a severely overestimated performance, which can easily be considered a real capability for data leakage, thus clearly reversing the vehicle on the validation set or test set.</p>
<h3>Why is a metamodel usually not too complicated?</h3>
<p>Many people think about Stacking, and they instinctively think, "Well, since they're all stacked to the second level, the stronger the model, of course." But the experience in the field is quite the opposite:<strong>The meta-models usually do not require particular complexity.</strong></p>
<p>The reason is simple. The input that the meta-model faces is no longer a raw feature, but a set of predictive signals that have been compressed by the base model. These signals are usually low in dimension and easily co-linear. If an overly complex meta-model is then, it is often not a robust combination pattern that is learned, but rather a test of the noise.</p>
<p>So, many of the streams of work are given priority:</p>
<ul>
<li>Catalogue tasks use logical regression as a meta-model</li>
<li>Re-entry missions use simple models such as linear return, and the return of the Ridge.</li>
<li>Use the probability output of the base model as a meta feature, rather than just keep a hard label</li>
</ul>
<p>Blinging can be seen as a simplified version of Stacking: it does not do the full OOF generation, but it simply leaves a validation set dedicated to training meta-models. This would be simpler and faster, but at the expense of some training data and more sensitive to the splitting approach.</p>
<p>In the data competition, Stacking is often an effective means of further extracting performance caps; it can also be valuable in the production environment, but it is important to consider additional training complexities, delays in reasoning, difficulties in monitoring and the cost of barriers. So Stacking is more like a strategy of "continue to do group learning on the basis of a strong model" than defaulting on the first option.</p>
<p>The key word for Stacking is <strong>Meta-characterization, OOF, second-tier learning</strong>I'm sorry. It is not the difference between Bagging / Boosting, it is not "more models," but it is the question of how to combine it itself.</p>
<h2>Dynamic selection: when the “overall weight” is insufficient</h2>
<p>The first ones, Averaging, Voting, Bagging, Boosting, Stacking, though different, have one thing in common:<strong>Once model training is completed, integration rules are usually fixed across the board.</strong></p>
<p>But in reality, a situation often arises: one model is very good in one sample but very common in another. It is not always reasonable to give it only one overall weight, which is to be implied that it is equally reliable in all local areas.</p>
<p>This leads to the idea of dynamic sorter selection (Dynamic Classifier Security, DCS) and dynamic integration selection (Dynamic Ensemble Security, DES). Instead of establishing a uniform combination rule in advance, they would first go to the local area adjacent to a new sample and look at it:<strong>Which models are the most credible in this local space, or which ones are the most credible?</strong></p>
<p>Intuitively, DCS is more like "local best modeling" and DES is more like "local best little integration." Such methods make model selection itself a dynamic process dependent on input positions, and therefore they are more flexible than the integration of fixed weights and are more closely aligned with the original motivation of “different models are good at different regions”.</p>
<p>However, the reasons for the absence of the first types of methods in actual landings are also realistic:</p>
<ul>
<li>The projections are often accompanied by additional area search, with higher costs calculated.</li>
<li>Local neighborhoods may be unstable, especially in the high-dimensional space.</li>
<li>Training, referral, deployment and interpretation are more complex</li>
</ul>
<p>So I prefer to see DCS/DES as an extended perspective: it reminds us that an ensemble is not necessarily a "one-size-fits-all rule"; but in most engineering scenarios, the fixed-rule Bagging/Booting/ Stateing is still more common and manageable.</p>
<p>DCS / DES contributed by moving the problem from "how to weighting the whole picture" to "who should trust the part." They are instructive, but are not usually the first priority in the form modelling workflow.</p>
<h2>Borders and Errors: Refusal / Dumpbin label does not match standard Ensemble</h2>
<p>Some techniques look like integrated learning, but not strictly within the standard insemble framework. A typical example, that is.<strong>Rejection or trash label (subject option / value class)</strong>。</p>
<p>It does not focus on “combining multiple models”, but rather allows the system to avoid high confidence judgements when uncertainty exists, or to place samples in a “suspension of decision” category. This is common in high-risk classification tasks, as it is better to recognize “not sufficiently certain” in a visible manner than to impose a potentially wrong label.</p>
<p>Such strategies sometimes appear in conjunction with two-stage processes: the first stage, which starts with normal classification, and the second stage, which decides whether to refuse based on confidence thresholds, abnormality detection results or additional rules. It is often used in combination with models, but it is more like a kind of thing.<strong>Decision-making strategy</strong>, instead of the integration method in the sense of Bagging / Boosting / Stacking.</p>
<p>Along this border, several common error areas can also be clarified:</p>
<ul>
<li><strong>The more models the better.</strong> If the model is highly relevant, it is simply a repetition of the same error, and the benefits of integration are limited.</li>
<li><strong>Diversity is not a pile of models.</strong> It is valuable to summarize differences at the level of preferences, training data, characteristic perspectives or loss functions.</li>
<li><strong>Stacking is the worst thing about leaking data.</strong> Without OOF, the "clip" is often just a more beautiful set of training.</li>
<li><strong>The refusal is not a delivery of precision.</strong> It is usually rebalancing coverage, recall rates and the cost of error.</li>
</ul>
<p>So it makes sense to put these boundary approaches in an integrated learning article, but it is best to keep in mind that they are complementary to “how to make final decisions” rather than defining standard lines of the ensemble.</p>
<h2>How to choose: Bagging, Boosting, Stacking and DCS</h2>
<p>If the preceding elements are to be organized into a “methodical selection table”, I would like to conclude:</p>
<table>
<thead>
<tr>
<th>Methodology</th>
<th>Training relations</th>
<th>Core objectives</th>
<th>Main strengths</th>
<th>Main costs</th>
<th>Typical representative.</th>
</tr>
</thead>
<tbody><tr>
<td>Bagging</td>
<td>Parallel</td>
<td>Lower the margin and increase stability</td>
<td>Steady, easy to run, good for strong</td>
<td>Limited improvement in systemic deviations</td>
<td>Bagging, Random Forest</td>
</tr>
<tr>
<td>Boosting</td>
<td>Serial</td>
<td>Over and over again, we'll be able to do this.</td>
<td>It's often stronger, it's very effective for table tasks.</td>
<td>It's easier to rely on the involvement and to make noise.</td>
<td>AdaBoost, GBDT</td>
</tr>
<tr>
<td>Stacking</td>
<td>Two-tier studies.</td>
<td>The rules of the mix between learning models</td>
<td>It integrates the isomer model, often to hit higher upper limits.</td>
<td>Training process complex, data leak prone</td>
<td>Stacking, Blending</td>
</tr>
<tr>
<td>DCS / DES</td>
<td>Dynamic Selection</td>
<td>To determine who to write to.</td>
<td>The rules of integration are more flexible and closer to "locally good"</td>
<td>High complexity and deployment, narrow application</td>
<td>DCS, DES</td>
</tr>
</tbody></table>
<p>If your main scene is a structured form model, I'd prefer this:</p>
<ol>
<li><strong>First, a steady, fast, easy-to-explain baseline.</strong>: Give priority to the Random Forest type Bagging thinking.</li>
<li><strong>I want to keep the single model going up.</strong>: Prioritize and use the GGBDT route, and select XGBoost, LightGBM or CatBoost according to the size and characteristic type of the task.</li>
<li><strong>There are already several powerful models of distinctly different styles that want to continue to squeeze the ceiling.</strong>Considering Stacking and carefully addressing OOF, delay and maintenance complexity.</li>
<li><strong>The mission is clearly local. And you're willing to accept higher costs.</strong>: DCS / DES can be used as a complementary route to a research or special scenario.</li>
</ol>
<p>Back to the question at the beginning of the article: Why is Random Forest and GDDT all "integrated trees," but behavioral differences are so great? The answer is clear.<strong>The former are doing parallel averages, while the latter are doing a series of wrongs.</strong> Once they're back on the two main lines Bagging and Boosting, it's easier to understand the context of the specific frames in the tree model article.</p>
<h2>References</h2>
<ul>
<li>Zhou Ji-hwa, Machine Learning, Integrated Learning Chapter Section</li>
<li>Li, "Statistical Learning Methodology", chapter on "Scaling Methods" Section</li>
<li>Leo Breiman, <em>Bagging Predictors</em>, 1996</li>
<li>Leo Breiman, <em>Random Forests</em>, 2001</li>
<li>Yoav Freund and Robert E. Schapire, <em>A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting</em>, 1997</li>
<li>Jerome H. Friedman, <em>Greedy Function Approximation: A Gradient Boosting Machine</em>, 2001</li>
<li>David H. Wolpert, <em>Stacked Generalization</em>, 1992</li>
</ul>
