---
title: Machine Learning Training Tricks Memo
title_zh: 机器学习炼丹技术备忘
date: 2025-11-25 12:33:02 +0800
permalink: /blog/2025/11/25/machine-learning-training-tricks/
categories:
- Machine Learning
- Deep Learning
tags:
- Machine Learning
- Model Training
- Research Methods
excerpt: A memo on machine learning training and experiment tricks, covering compute, hyperparameters, small architecture
  changes, incremental design, evaluation methods, and common training tactics.
description: A memo on machine learning training and experiment tricks, covering compute, hyperparameters, small architecture
  changes, incremental design, evaluation methods, and common training tactics.
lang: en
translation_key: 2025-11-25-machine-learning-training-tricks
translation_status: machine
translation_source_hash: fd6ecbbb92dffe75708f2bbeda86c3b8ea79a2a628f2361e4f82e5e844a1795d
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<h2>The force crush.</h2>
<p>1.1 Larger batchsize, pretend to be multiple-timed

1.2 Multiple training, but it is not clear that replacing the length of training with an iterative report, and vice versa, cannot be seen as a single sight.

1.3 epoch numbers remain constant, but a sample is used several times, thus sneaking over data

1.4 Reduced number of samplings in the model, many times larger than the number of models calculated, but only against the number of parameters

1.5 Massive computing power in areas that do not care about the amount and the amount of reference

1.6 Description of the highly economical components, and efficiency analysis only for other components

1.7 The model is very large with heavy parametrics, slow training, but it's more expensive than reasoning.

1.8 EMA / Multi-model convergence points, with conditions that allow self-distillation

1.9. Select a super-small set of training so that you can focus on the match.</p>
<h2>Superparameter</h2>
<p>2.1. Obtain desired experimental results by adjusting the COSINE learning rate change to a fixed learning rate, or vice versa (the final part of COsine ' s reduced learning rate usually makes models rapidly rising and early drops in learning rates seem to be highly training-efficient)

2.2 A little higher learning rate, a little higher learning rate for baseline Small

2.3 Hide all kinds of superparameters into code as magnetic number

2.4 Select random seeds</p>
<h2>- I'm gonna fix it.</h2>
<p>3.1 Replace relu with swish or leaky relu / prelu

3.2 Sneaking around the world, basically up points; cheaper access

3.3 Replace components like poping, reese without parameters with learning parameters.

3.4 Faults between the teams, multiple concat, some features, anyway.

3.5 Add BN where there is no BN, remove BN where there is BN, and replace GN/ IN/ LN/ WN, etc.

3.6 Expansion of the training package and reprogramming of the training package in response to differences between the training and testing sets</p>
<h2>Incremental Design</h2>
<p>4.1 Gage's weird, Gan Los, Convergence, Los, it's hard to say that there are many formulas.

4.2. Detailed technology of the words of the paper, plus some magic formulas to make half a page of papers

4.3 When you want to design components x to add to the model, create a learning beta parameter with an initial value of 0, and then add Beta*x to the model, the worst case of which is beta=0.

4.4 Extension of the previous article to design a set of components in a learning parameter format

4.5. Continue to expand. Add a NAS in.

4.6 Take some pre-training parameters from other models so that the starting point of the model becomes higher and the upper limit becomes higher because it is equivalent to data added and labeled

4.7. A very complex course, varnishing, or not, that people work to get involved.

4.8. Improved learning framework, whether useful or not, to increase ownership of models</p>
<h2>Test Method</h2>
<p>5.1 Ten indicators, three of which reported progress

5.2 Experiment 10 data sets, throw out five useless ones.

5.3. Deliberately mischaracterizing the test and other people's training, doing low baselines, like putting RGB tunnels off the back of the other people's head.

5.4 New indicators for innovation evaluation developed; magic indicators such as Y Channel PSNR, but with others RGB - Yeah.

5.5 Looking for a situation that no one else has thought about, making a huge promotion.

5.6 Large models are smaller than others, without reporting on large models; models trained for certain indicators are more untrained than others

5.7 Speeding on different hardware and reporting together

5.8 Recent language large model, sneaking in to test prompt Riga hint, few-shot and zero-shot

5.9 Consistency of the disguised test set, such as data leaking, random seed leaking; test samples placed in upstream pre-training

5.10 Test data sets with real scenes, OOD samples, baseline drops a lot, then add a little more or dropout and add the dots back, but count the contribution to the dots elsewhere

5.11 Private testing sets, manual evaluation, and improvement can be done with more visibility

5.12 Objective comparison is no more than subjective, no more subjective than chirry pick</p>
<h2>End of Method</h2>
<p>6.1. Copy one other person's method, but change the name.

6.2 High performance, open source is only README

6.3. Start writing papers without experimenting. It's just as high as sota. Points </p>
<p>The questions in this article can also be addressed<a href="/en/blog/2024/09/02/deep-learning-basics/">In-depth learning base: neuronet, optimization approach and integration</a>、<a href="/en/blog/2024/11/13/deep-learning-network-architectures/">In-depth learning network architecture: CNN, RNN and Seq2Seq</a>How the concept of a relatively close read together is developed in different contexts.</p>
<ol>
<li>Self-gating basically up points</li>
</ol>
<p>Variables include context gating and SE modules
The core idea is to use itself to get itself into the game.

The basic form is y = sigmoid(wx)x

2. Reconstructions, which involve entering the corrupt and then rebuilding with the autoencoder, will generally make the feature more robust, as will the MaE of Ho Kemin.

All kinds of dropout, a place where you can try to add a dropout, embeding can add a dropout, attent, ffn could add, mlp could add, or input it directly, equivalent to some sort of currupt
4. Miixup, also a god-class ideaa, enter a mixture of the upper a class +b, then label becomes a +b mixture, basically brainless, which is bound to increase.

5. In contrast to learning the Great One, the core looks at how to construct positive and negative samples. There's a amazing idea, the same input twice, because the dropout is different, and it's a positive sample, and it's a brainless rise.</p>
<h2>References</h2>
<ul>
<li><a href="/assets/docs/deep-learning-tuning-guide-zh.pdf">In-depth learning adaptation guide in Chinese</a></li>
</ul>
