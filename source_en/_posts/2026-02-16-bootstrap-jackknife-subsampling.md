---
title: 'The Computational Revolution in Statistical Inference: Jackknife, Bootstrap, and Subsampling'
title_zh: 统计推断的计算革命：详解 Jackknife, Bootstrap 与 Subsampling
date: 2026-02-16 12:00:00 +0800
categories:
- Data Science
- Statistical Modeling & Inference
tags:
- Statistical Inference
author: Hyacehila
hidden: true
excerpt: How can we estimate statistical uncertainty without heavy distributional assumptions? This post explains Jackknife,
  Bootstrap, and Subsampling from principles to asymptotic behavior.
description: How can we estimate statistical uncertainty without heavy distributional assumptions? This post explains Jackknife,
  Bootstrap, and Subsampling from principles to asymptotic behavior.
excerpt_zh: 无需繁冗的分布假设，如何估计统计量的误差？本文深入剖析 Jackknife、Bootstrap 与 Subsampling 三种重抽样方法，从数学原理到渐进性质，探讨计算力如何替代解析推导成为统计推断的新引擎。
permalink: /blog/2026/02/16/bootstrap-jackknife-subsampling/
lang: en
translation_key: 2026-02-16-bootstrap-jackknife-subsampling
translation_status: machine
translation_source_hash: 3f1f10395f2770984faa99e815a8292d8a1dc4983141fee6155762769aa6e4ab
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<h2>Introduction: Inference crises in application</h2>
<p>The questions in this article can also be addressed<a href="/en/blog/2026/01/16/dimensionality-reduction-high-dimensional-data/">From Oscillospace to Flowing: The Declining of High Data</a>How the concept of a relatively close read together is developed in different contexts.</p>
<p>In today's data science applications, the common scenario is that you have designed an operational indicator to measure algorithmic effects or have trained an in-depth neuronet model. In addition to obtaining a point estimate (e.g. CTR up 2%, or model accuracy 85%), operators or scientific reviewers often ask:</p>
<p><strong>"What is the confidence of this result? How much is it in the range of fluctuations?"</strong></p>
<p>That sounds like a standard statistical inference. In the traditional statistics course, we learned to process the mean of the function with CLT (centre of extreme limits), and the difference of the function with Delta Method (Delta method). But in the face of modern applications, these classic tools are often not effective:</p>
<ul>
<li><strong>The indicators are too complex.</strong>: In the A/B test, many of the indicators we are concerned with (e.g., the decay factor for retention rates, the fractional value of the LTV for the user life cycle) are difficult to process, and it is impossible to even write a decomposition expression, let alone ask for a guide.</li>
<li><strong>The hypothesis is too fragile.</strong>: Many extrapolations rely on the normality or independence of data (IID). However, in financial time series or social networking data, these assumptions are often broken.</li>
<li><strong>Blackbox Model</strong>: For deep learning models, we do not even know what the specific "parameters" distribution is, much less the broad error boundary that can be extrapolated through the Hessian matrix.</li>
</ul>
<p>In the face of these “unpredictable” difficulties, statistics have undergone an important shift in the second half of the 20th century:<strong>Computer statistics</strong>Rise. It's a direct idea, but it changes the way it is extrapolated:</p>
<p><strong>If distribution is not easily extrapolated by mathematical formulas, can we use the powerful computing power to "calculate" distribution directly by means of repeated simulations of data?</strong></p>
<p>The three leading actors in this revolution are discussed here:<strong>Jackknife</strong>、<strong>Bootstrap</strong> and <strong>Subsampling</strong>I'm sorry. We start with the pain of their application, and see how computing can replace the decomposition and become a new engine of modern statistical extrapolation.</p>
<h2>2. Jackknife: sharp instrument for correction of deviations</h2>
<h3>Application scenario: deviation from ratio estimates</h3>
<p>In sample surveys and econometrics, we often need to estimate the ratio of the two variables. For example, to estimate the “input output ratio” of society as a whole, we may divide the average output of the sample by the average input:&#36;\hat{R} = \bar{y} / \bar{x}&#36;。</p>
<p>However, statistics tell us,<strong>Rate expectations do not match expectations</strong>(i.e., &#36;E[\bar{y}/\bar{x}] \neq E[y]/E[x]&#36;I'm not sure. This means that the government is not a party to the law.&#36;\hat{R}&#36; It's one.<strong>Biased Estimator</strong>I'm sorry. In the case of small samples, this deviation may seriously mislead decision-making.</p>
<p>How do you correct this deviation? If you're going to deduce &#36;\hat{R}&#36; Taylor has begun to make corrections, which are cumbersome and prone to errors. This time,<strong>Jackknife (scissor)</strong> It's going to work.</p>
<h3>Algorithms and principles</h3>
<p>Jackknife's idea is,<strong>Insulate the nature of the whole by observing the impact of the absence of some data on the whole</strong>。</p>
<p>Suppose we're gonna take a sample. &#36;\mathcal{X}_n = {X_1, \dots, X_n}&#36; Medium estimate parameters &#36;\theta&#36;。</p>
<ol>
<li><strong>Build a Go-I sample</strong>: Yes &#36;i = 1, \dots, n&#36;♪ We'll be &#36;i&#36; A observation value removed from the sample and the size of the observation was &#36;n-1&#36; The sample. &#36;\mathcal{X}_{(i)}&#36;。</li>
<li><strong>Calculates the volume of copies</strong>: recalculated statistics on each “Done One” sample, obtained &#36;\hat{\theta}_{(i)}&#36;。</li>
</ol>
<p>Tukey (1958) proves that through these &#36;\hat{\theta}_{(i)}&#36;, we can construct a revised estimate of deviation:</p>
<p>&#36;&#36;
\hat{\theta}<em>{jack} = n\hat{\theta} - (n-1)\bar{\theta}</em>{(\cdot)}
&#36;&#36;</p>
<p>Where's the bar?<em>{(\cdot)} = \frac{1}{n}\sum \hat{\theta}</em>{(i)}&#36;。</p>
<p>It seems simple, but strong: It can take the deviation from the estimate. &#36;O(n^{-1})&#36; Directly from elimination to &#36;O(n^{-2})&#36;I'm sorry. In applications with small sample volumes (e.g. early medical clinical trials), Jackknife can provide much more accurate estimates.</p>
<p>At the same time, Jackknife gave a non-parameter estimate of the variance:</p>
<p>&#36;&#36;
\widehat{Var}<em>{jack}(\hat{\theta}) = \frac{n-1}{n} \sum</em>{i=1}^n (\hat{\theta}<em>{(i)} - \bar{\theta}</em>{(\cdot)})^2
&#36;&#36;</p>
<h3>Limitations and application of borders</h3>
<p>The calculation cost for Jackknife is low (just need &#36;n+1&#36; (a) A positive performance in correction. But in modern applications, it has obvious limitations:</p>
<p><strong>It will fail on non-smooth indicators.</strong></p>
<p>The most typical example is<strong>Medium (Median)</strong>I'm sorry. Even the sample. &#36;n&#36; The median difference estimated by Jackknife is not consistent (it will not shrink to the real difference). This limits its application in modern wind control indicators based on fractional numbers (e.g. VaR, Value at Risk). We need a more universal approach.</p>
<h2>Bootstream: Swiss military knife, general extrapolation</h2>
<h3>Application scenario: confidence-building for complex operational indicators</h3>
<p>In the Internet company A/B testing platform, analysts often define complex “artic star indicators”. For example:</p>
<p>&#36;&#36;
\text{Metric} = \frac{\text{GMV&#125;&#125;{\text{DAU&#125;&#125; \times \log(\text{Retention}_{7\text{day&#125;&#125;)
&#36;&#36;</p>
<p>Is this real or random increase when the test group and control group have a 1% difference? To answer that question, we need to draw a composite indicator. <strong>95% confidence compartment</strong>I'm sorry. It is almost impossible to extrapolate the distribution of this indicator.</p>
<p>1979, by Bradley Efron. <strong>Bootstrap (self-help)</strong> A common solution to the problem was provided.</p>
<h3>Core idea: Plug-in Prince (inclusion principle)</h3>
<p>The philosophy of Bootsrap is very intuitive:<strong>Since we can't get the real distribution, &#36;F&#36;Then you can tell the distribution of the experience you've seen. &#36;\hat{F}_n&#36; Consider it a true sum.</strong></p>
<ul>
<li><strong>Real world.</strong>: From the general &#36;F&#36; Data from the Chinese sample &#36;\mathcal{X}_n&#36;, calculate the amount of statistics &#36;\hat{\theta}&#36;。</li>
<li><strong>Bootstrap World</strong>: Distribution from experience &#36;\hat{F}_n&#36; Medium<strong>- Yes.</strong> Sampled &#36;matcal{x}<em>_n&#36;，计算统计量 &#36;\hat{\theta}^</em>&#36;。</li>
</ul>
<p>Efron proves that under quite a wide range of conditions, the world of Bootslap Medium &#36;\hat{\theta}^*&#36; Around &#36;\hat{\theta}&#36; The distribution of the world is perfect for the real world. Medium &#36;\hat{\theta}&#36; Around True Values &#36;\theta&#36; the distribution.</p>
<h3>Algorithms & Credibles</h3>
<p>With this principle, it becomes very simple to apply:</p>
<ol>
<li><strong>Re-sampling</strong>: Using computers, put back to extract from raw data &#36;B&#36; Group data (e.g., &#36;B=1000&#36;）。</li>
<li><strong>Redecount</strong>Compute your complex indicators on each set of data, and get 1000 &#36;\hat{\theta}^*&#36;。</li>
<li><strong>Distribution extrapolation</strong>This is the distribution of 1,000 values, the simulation sample distribution of your complex indicator.</li>
</ol>
<p>For confidence compartments, most commonly.<strong>Bitmap</strong>Take the first 2.5% of the 1000 digits and the 97.5% fraction point as the upper and lower boundary between the zones.</p>
<p>We can also use scenarios that require a high degree of precision (e.g. biopharmaceuticals). <strong>BCa (Bias-Corrected and Accelerated) Method</strong>I'm sorry. Using the bias and deviation information estimated by Jackknife, it fine-tuned the fractions to obtain a second-order accuracy.</p>
<p>Bootstrap has significantly liberated the productivity of data scientists and has become a standard tool for modern statistical extrapolation of this "Swiss Army Sword".</p>
<p>Here's another side: in the machine learning model assessment, Bootstream was used as a data-segregation strategy -- &#36;D with a drop sample.&#39;&#36; Trained, tested with approximately 36.8% of undiscovered samples (out-of-bag), suitable for small data sets and integrated learning. This usage complements the extrapolation perspective of this section, and a more systematic model assessment approach is presented in the following paragraphs:<a href="/en/blog/2025/10/02/supervised-learning-model-evaluation/">Monitoring of learning performance assessment</a>The self-help approach section of the Law.</p>
<h2>Subsampling: Last line of defence in extreme cases</h2>
<h3>Apply scene: when Bootstream is invalid Time</h3>
<p>Bootstream is useful, but it's not almighty. In some high-risk applications, blind use of Bootstream can have serious consequences:<strong>It may give a confidence interval that seems precise and completely incorrect.</strong></p>
<p>Typical failures include:</p>
<ol>
<li><strong>Extreme estimate</strong>: For example, the distribution boundary for estimating peak traffic in cybersecurity. The maximum value of the Bootslap sample will never exceed that of the original sample, leading to a degradation of its distribution at the end.</li>
<li><strong>Parameters at Borders</strong>: When the true parameter is located at the boundary of the parameter space (e.g. a 0-square test).</li>
<li><strong>Strong reliance on data</strong>: Simple Bootslap destroys the time-series structure of the time series.</li>
</ol>
<p>At this point, we need to ask the most theoretical and robust members of the heavy sample family:<strong>Subsampling (subsampling)</strong>。</p>
<h3>Core thinking and operational guidelines</h3>
<p>Subsampling's logic is clearly different from Bootstream:<strong>Bootstream tries to simulate the real world.&#36;n \to n&#36;The real world is only a microcosm (the real world).&#36;m \to \infty&#36;）。</strong></p>
<p>Its core operation is based on<strong>No Return Sample (Sampling Without Replacement)</strong>and sample quantities &#36;m&#36; Much less than &#36;n&#36;。</p>
<h4>Specific Operational Steps (Algoritthm)</h4>
<p>Subsampling is used in the actual project and can be operated as follows:</p>
<ol>
<li><p><strong>Determine subsampling size &#36;m&#36;</strong>:
This is the most critical step.&#36;m&#36; It has to follow. &#36;n&#36; Increase and increase, but faster than &#36;n&#36; Slow.<em>Common experience:&#36;m = \sqrt{n}&#36; or &#36;m = n^{2/3}&#36;</em>。</p>
</li>
<li><p><strong>Build subsamples</strong>:
From the original &#36;n&#36; The blog is a good example of the situation.<strong>No return</strong>Extract &#36;m&#36; Data. That means you're generating a much smaller data set than the original one.</p>
</li>
<li><p><strong>Calculate statistics</strong>:
Count your numbers on each subsampling. &#36;\hat{\theta}^*_{m}&#36;。</p>
</li>
<li><p><strong>Repeat and Distribution Build</strong>:
Repeat the above process &#36;B&#36; Number of times (e.g. &#36;B=1000&#36;(c) The distribution of experience in obtaining statistics.</p>
</li>
<li><p><strong>Rescaling</strong>:
This step is often forgotten. Because you're in &#36;m&#36; The distribution calculated on a sample is definitely more varied than the one calculated on the sample. &#36;n&#36; A sample on the big. To extrapolate the original sample, &#36;n&#36; The nature of the process, you need to use the speed of extraction. &#36;\tau_n = \sqrt{n}&#36;) To scale up.
What we need to see is... &#36;\tau_m (\hat{\theta}^*_m - \hat{\theta}_n)&#36; The distribution of the problem. Use it to simulate it. &#36;\tau_n (\hat{\theta}_n - \theta)&#36; the distribution.</p>
</li>
</ol>
<h3>Why is it more "strong" in theory?</h3>
<p>The effectiveness of Bootstream depends on a stronger assumption:<strong>The distribution of statistics must be smooth.</strong>(i.e., &#36;\hat{F}_n&#36; Weak harvests &#36;F&#36; At times, the distribution of statistics must also be reduced). This is often not valid when non-slipper parameters (e.g. polar, median, etc.) or when parameters are located at the boundary.</p>
<p>By contrast, Politis &amp; Romano (1994) proves that <strong>Subsampling's effectiveness requires only that the statistical volume itself be distributed at limits.</strong></p>
<p>This is a very weak condition:</p>
<ul>
<li>If Bootstream works, Subsampling works (although efficiency is somewhat lower because it is not a whole sample).</li>
<li>Subsampling is still valid if Bootstream is invalid (e.g., the polar issue).</li>
</ul>
<p>It is thus often called the “last line of defence”.</p>
<h3>Modern front application cases</h3>
<h4>1. General error estimates in depth learning</h4>
<p>In in-depth study theory, we often need to estimate the boundaries of the model across error. The standard progressive normality assumption is completely invalid because the loss function curvature of the nervous network is non-compressed, non-silent and very high in parameters.
The blogger says:<strong>Subsampling</strong> An effective means of constructing such non-ruled statistical confidence-building areas is provided. It does not need to assume second-order guidance for loss functions and is able to capture more robustly model fluctuations resulting from training data disturbances.</p>
<h4>2. Enhanced learning and time series: Block Subsampling</h4>
<p>In the Policy Assessment, the trajectory data generated by intelligent bodies are highly relevant in time. The random and shattering of heavy samples by Bootstream directly undermines this time dependence and leads to a serious underestimation of the Value Function equation. You'll think the strategy is stable, not really.</p>
<p>The solution is to use <strong>Block Subsampling</strong>I'm sorry. We are not taking single sample points, but a continuous “block of time”. This preserves the local dependency structure within the data, which allows for the correct estimation of the differential returns of Long-horizon and provides a real security boundary for the tactical trajectories.</p>
<h4>3. Extreme value theory (EVT) and financial regulation</h4>
<p>In calculating the value of the financial market at risk (VaR) or expected loss (ES), we are concerned with the distribution of the “tails” — the black swan event that occurs at one-tenth of the 10,000.
Subsampling has become the preferred statistical extrapolation of extreme values due to the inconsistency of Bootstrap ' s estimates of boundaries. Select the appropriate subsampling size &#36;m&#36;We can use it. &#36;m&#36; The polar distribution pattern in the sample is extrapolated by the scaling relationship of the polar theory. &#36;n&#36; Extreme risks in samples even on larger scales in the future.</p>
<h2>5. Summary</h2>
<p>From Jackknife to Bootstream, and then Subsampling, this evolution clearly shows how statistics, with their calculus, conquer the difficulty of applying:</p>
<ul>
<li>♪ When we face ♪<strong>Small sample deviation</strong>The problem is that the government is not a party to the law.<strong>Jackknife</strong> The simplest way to go is to give a graceful solution.</li>
<li>♪ When we face ♪<strong>Common extrapolation of complex statistics</strong>The problem is that the government is not a party to the law.<strong>Bootstrap</strong> Using “simulations instead of extrapolation”, it has become a standard weapon for data scientists.</li>
<li>♪ When we face ♪<strong>Extreme, non-sliding, heavy dependence</strong>The blogger says that the government is not the only one who is not a member of the opposition.<strong>Subsampling</strong> Relying on weaker theoretical assumptions, it is often more stable than Bootstream.</li>
</ul>
<p>In today ' s time, where the algorithms are within reach, it is far more valuable for data science practitioners to understand the rationale and boundaries of these methods than to remember a few normal distribution formulas. The real world is often not a normal distribution, but it can always be counted.</p>
