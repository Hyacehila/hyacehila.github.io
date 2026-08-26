---
title: Common Statistical Tests Are Linear Models
title_zh: 常见的统计检验本质上都是线性模型 (Common statistical tests are linear models)
date: 2026-02-07 12:00:00 +0800
categories:
- Data Science
- Statistical Modeling & Inference
tags:
- Linear Models
- Repost
author: Hyacehila
excerpt: A repost of Jonas Kristoffer Lindelov's classic article showing the unified linear-model logic behind t-tests, ANOVA,
  chi-square tests, and other common methods.
description: A repost of Jonas Kristoffer Lindelov's classic article showing the unified linear-model logic behind t-tests,
  ANOVA, chi-square tests, and other common methods.
excerpt_zh: 转载自 Jonas Kristoffer Lindeløv 的文章。揭示了 t 检验、ANOVA、卡方检验等常用统计方法背后的统一线性模型原理。
permalink: /blog/2026/02/07/common-statistical-tests-are-linear-models/
lang: en
translation_key: 2026-02-07-common-statistical-tests-are-linear-models
translation_status: machine
translation_source_hash: 838333ddbda267547a969b188353b2df8be1c50c9175207637e18b9ec258afb0
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<blockquote>
<p>This post is a good article from Jonas Kristoffer Lindeløv. <a href="https://lindeloev.github.io/tests-as-linear/">Common statistical tests are linear models</a>
The original text reveals in a very shallow way an amazing simple truth in statistics: most commonly used statistical tests (t-test, related analysis, ANOVA, calculator tests, etc.) are exceptions to linear models.</p>
</blockquote>
<p>The questions in this article can also be addressed<a href="/en/blog/2023/09/04/linear-regression-basics-notes/">Linear regression base: linear model, minimum 2x2 estimate and regression diagnosis</a>、<a href="/en/blog/2025/09/23/advanced-linear-regression-notes/">Linear regression step: proposed alignment, model selection and co-line Sex</a>How the concept of a relatively close read together is developed in different contexts.</p>
<h1>Common statistical tests are linear models</h1>
<h2>Core concept: Linearness of everything</h2>
<p>Most commonly used statistical models (t-test, coordination, ANOVA; chi-square, etc.) are either linear models or similar ones. We do not have to remember the assumptions and formulas of each test in the back, because they can be attributed to the formulas that were taught in high school:</p>
<p>&#36;&#36;y = a \cdot x + b&#36;&#36;</p>
<p>This simple aesthetic sense lowers the threshold for understanding statistics. The linear models at the bottom are consistent, whether they are frequency, Bayesian or on the basis of replacement.</p>
<p>For what is called the "non-parametric tests", we can also understand in a more intuitive way: They're usually just...<strong>Zirconium (rank-transformed)</strong> The corresponding parameter tests run on the data. Instead of considering the non-parametric test “no assumptions are required”, it is understood as “calculating in rankings (ranks)”.</p>
<p>This view is summarized in the figure below (click on page). Look.<a href="https://lindeloev.github.io/tests-as-linear/linear_tests_cheat_sheet.pdf">PDF Version</a>）：</p>
<p><img src="https://lindeloev.github.io/tests-as-linear/linear_tests_cheat_sheet.png" alt="Linear Tests Cheat Sheet"></p>
<hr>
<h2>Relevance (Pearson and Spearman)</h2>
<h3>Theory: as a linear model</h3>
<p>The essence of the relevance analysis is to find the best possible line. The model formula is as follows:</p>
<p>&#36;&#36;y = \beta_0 + \beta_1 x \qquad \mathcal{H}_0: \beta_1 = 0&#36;&#36;</p>
<p>That's what we know. &#36;y = ax + b&#36;I'm sorry. In R languages, we usually write <code>y ~ 1 + x</code>That means... &#36;y = 1 \cdot \beta_0 + x \cdot \beta_1&#36;I'm sorry. It's got to be cut by the cut, no matter what.&#36;\beta_0&#36;) and tilt ( )&#36;\beta_1&#36;(c) Composition.</p>
<h3>Rank-Transformation and Spearman</h3>
<p>Spearman, the coefficient is actually right. &#36;x&#36; and &#36;y&#36; Conduct<strong>& Change (Rank-Transformation)</strong> Post-Pearson correlation coefficient:</p>
<p>&#36;&#36;rank(y) = \beta_0 + \beta_1 \cdot rank(x) \qquad \mathcal{H}_0: \beta_1 = 0&#36;&#36;</p>
<p>The term "Rank" is used to replace the value with their size ranking (minimum 1 and second, small 2...). Although Spearman's p value was only approximate at the time of the small sample, when N &gt; The time is usually sufficiently accurate.</p>
<h3>R-Class: Pearson</h3>
<p>Run the R code below, you find a linear model (<code>lm</code>) &#36;t&#36;, &#36;p&#36; Value & & & & Inline <code>cor.test</code> Exactly.</p>
<p>The difference is:<code>lm</code> It gives a slope, and... <code>cor.test</code> The relevant coefficient is given. &#36;r&#36;I'm sorry. If we standardize the data (SD=1), the slope is equal to &#36;r&#36;。</p>
<pre><code class="language-r"># Built-in t-test
a = cor.test(y, x, method = &quot;pearson&quot;)

# Equivalent linear model: y = Beta0*1 + Beta1*x
b = lm(y ~ 1 + x)

# On scaled vars to recover r
c = lm(scale(y) ~ 1 + scale(x))
</code></pre>
<h3>R Code Contrast: Spearman</h3>
<p>The same logic applies to Spearman's connection, just to do the data first. <code>rank()</code> Change:</p>
<pre><code class="language-r"># Spearman correlation
a = cor.test(y, x, method = &quot;spearman&quot;)

# Equivalent linear model
b = lm(rank(y) ~ 1 + rank(x))
</code></pre>
<hr>
<h2>Average (One Mean)</h2>
<h3>Theory: as a linear model</h3>
<p>Single sample T test (One-sample t-test) tested whether the average sample value is significantly different from 0. It's actually a...<strong>Only the cut.</strong>Linear model:</p>
<p>&#36;&#36;y = \beta_0 \qquad \mathcal{H}_0: \beta_0 = 0&#36;&#36;</p>
<p>Not here. &#36;x&#36;Or... &#36;x=0&#36;♪ So the rest ♪ &#36;\beta_0&#36; It's the average.</p>
<p>For non-parameters <strong>Wilcoxon Symbolic Test (Wilcoxon signed-rank test)</strong>It's the same principle. It's just applied.<strong>Symbolic (signed ranks)</strong> Data:</p>
<p>&#36;&#36;signed_rank(y) = \beta_0&#36;&#36;</p>
<h3>R Code Contrast: Single Sample T Test</h3>
<pre><code class="language-r"># Built-in t-test
a = t.test(y)

# Equivalent linear model: intercept-only
b = lm(y ~ 1)
</code></pre>
<p>You'll find out. <code>lm(y ~ 1)</code> , the estimated value of the amplitude item (Estimate) is the mean, the t and p values are also equal to <code>t.test</code> The results are perfectly consistent.</p>
<h3>R Code Contrast: Wilcoxon Symbol Check</h3>
<pre><code class="language-r"># Built-in
a = wilcox.test(y)

# Equivalent linear model
b = lm(signed_rank(y) ~ 1)

# Bonus: also works for one-sample t-test on signed ranks
c = t.test(signed_rank(y))
</code></pre>
<p>Use <code>lm</code> Not only do you get a p-value matching, but you get a "mean sign" directly, which is a more intuitive number than a simple W statistical figure.</p>
<hr>
<h2>Other common testing summary</h2>
<p>In addition to the two examples mentioned above, other common statistical tests can be mapped into linear models. To keep the paper simple, a summary cross-reference is provided below, with a link to the original text for detailed extrapolations and codes.</p>
<table>
<thead>
<tr>
<th align="left">Statistical Test (Test)</th>
<th align="left">Linear Model Formula (Simpleted LM)</th>
<th align="left">Original Link</th>
</tr>
</thead>
<tbody><tr>
<td align="left"><strong>Double means</strong> <br> (Independent t-test)</td>
<td align="left">&#36;y = \beta_0 + \beta_1 x&#36; <br> (&#36;x&#36; (is a subcategory variable)</td>
<td align="left"><a href="https://lindeloev.github.io/tests-as-linear/#means2">Link</a></td>
</tr>
<tr>
<td align="left"><strong>Three or more means</strong> <br> (One-way ANOVA)</td>
<td align="left">&#36;y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ...&#36; <br> (Perfect variable coded)</td>
<td align="left"><a href="https://lindeloev.github.io/tests-as-linear/#means3">Link</a></td>
</tr>
<tr>
<td align="left"><strong>Analysis of the differences (ANCOVA)</strong></td>
<td align="left">&#36;y = \beta_0 + \beta_1 x_{categorical} + \beta_2 x_{continuous}&#36;</td>
<td align="left"><a href="https://lindeloev.github.io/tests-as-linear/#ancova">Link</a></td>
</tr>
<tr>
<td align="left"><strong>Scale to calorie (Portions / Chi-square)</strong></td>
<td align="left">&#36;\ln(y) = \beta_0&#36; <br> (Log-linear Models, returns using Poisson)</td>
<td align="left"><a href="https://lindeloev.github.io/tests-as-linear/#proportions">Link</a></td>
</tr>
</tbody></table>
<hr>
<h2>Summary</h2>
<p>Understanding the linear model relationships behind these tests allows us to reduce our reliance on specific "test names" and to focus on model construction. Whether t or complex ANOVA, they answer the same question: Are my model parameters significantly not zero?</p>
<p>Thank you. <strong>Jonas Kristoffer Lindeløv</strong> Provides a brilliant perspective.</p>
<ul>
<li><strong>Original Link</strong>: <a href="https://lindeloev.github.io/tests-as-linear/">Common statistical tests are linear models</a></li>
<li><strong>Python Version</strong>: <a href="https://eigenfoo.xyz/tests-as-linear/">Tests as linear (Python)</a></li>
<li><strong>GitHub Repository</strong>: <a href="https://github.com/lindeloev/tests-as-linear">lindeloev/tests-as-linear</a></li>
</ul>
