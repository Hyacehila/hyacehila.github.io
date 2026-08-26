---
title: Why I Am Not a Fan of R-Squared
title_zh: 统计学常用评估指标R方，它从不衡量模型与真实世界的拟合程度
date: 2026-02-08 12:00:00 +0800
categories:
- Data Science
- Statistical Thinking
tags:
- Linear Models
- Evaluation
- Statistical Thinking
author: Hyacehila
excerpt: R-squared is not a simple model error function. Its definition hides a comparison against a constant model, so it
  never directly measures fit to the real world.
description: R-squared is not a simple model error function. Its definition hides a comparison against a constant model, so
  it never directly measures fit to the real world.
excerpt_zh: R 方不是单纯的模型误差函数，它的定义中还隐含了两个模型的比较：一个是当前被分析的模型，一个是所谓的常数模型，即只利用因变量均值进行预测的模型。基于此，R 方从不衡量模型与真实世界的拟合程度。
permalink: /blog/2026/02/08/why-im-not-a-fan-of-r-squared/
lang: en
translation_key: 2026-02-08-why-im-not-a-fan-of-r-squared
translation_status: machine
translation_source_hash: 928047995745c48f268cad56c5dfef722e71de3bbb0b6b09c713bae281631fc4
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<blockquote>
<p>The core view of this post is translated from John Myles White's article. <a href="https://www.johnmyleswhite.com/notebook/2016/07/23/why-im-not-a-fan-of-r-squared/">Why I&#39;m Not a Fan of R-Squared</a></p>
</blockquote>
<p>The questions in this article can also be addressed<a href="/en/blog/2026/01/10/statistics-and-truth/">Statistics and Truth: How to use the accident (Statistics and Truth)</a>、<a href="/en/blog/2026/01/14/anscombes-quartet/">Anscom Quartet: Visualized power and statistical illusion</a>How the concept of a relatively close read together is developed in different contexts.</p>
<h2>Take Home Message</h2>
<p>&#36;R^2&#36; Not simply model error functions, but it also implies a comparison of two models: One is the model currently analysed, and one is the so-called constant model (i.e., the model that only uses the average of variables to predict). The blogger says:&#36;R^2&#36; The answer is one question:</p>
<blockquote>
<p><strong>"Is my model better than a constant model?"</strong></p>
</blockquote>
<p>But we usually want to answer another completely different question:</p>
<blockquote>
<p><strong>"Is my model worse than the real model?"</strong></p>
</blockquote>
<p>The case of some people who have been constructed shows that the answers to the two questions are not interchangeable. One example can be constructed: our models are not much better than constant models, nor are they much worse than real models. Similarly, another example can be constructed that makes our models far better than constant models, but far worse than real models.</p>
<p>The way the model compares with all the models,<strong>&#36;R^2&#36; Not only is it a function of a comparable model, it is also a function of observational data.</strong>I'm sorry. Almost all models have a data set that makes it impossible to distinguish between constant models and real models. Specifically, when using a model to distinguish between data sets that are less effective,&#36;R^2&#36; It can be anywhere near zero -- even if we calculate the real model. &#36;R^2&#36; The same goes for me.</p>
<p>We must therefore always remember:</p>
<p><strong>&#36;R^2&#36; It's not like we're talking about models that are a good approximation of real models.&#36;R^2&#36; Just tell us if our models are much better under current data than a constant model.</strong></p>
<h2>A theoretical example</h2>
<p>To understand how “comparison of models with constant models” leads to a conclusion that is distinct from “comparison of models with real models”, a simple example is considered: we want to compare functions. &#36;f(x)&#36; Modelling and &#36;n&#36; Noise-bearing data were observed at the point of the equation.</p>
<p>First, it is assumed that:</p>
<ul>
<li>&#36;f(x) = \log(x)&#36;。</li>
<li>&#36;x_{min} = 0.99&#36;。</li>
<li>&#36;x_{max} = 1.01&#36;。</li>
<li>Yes. &#36;x_{min}&#36; and &#36;x_{max}&#36; We've seen 1,000 points of even distribution. &#36;y_i = f(x_i) + \epsilon_i&#36;, of which &#36;\epsilon_i \sim \mathbf{N}(0, \sigma^2)&#36;。</li>
</ul>
<p>Based on these data, we try to use single variables, OLS, to learn from it. &#36;f(x)&#36; and a secondary model. An example of the modelling process is as follows:</p>
<p><img src="/assets/images/statistical-thinking/r-squared-small-range.png" alt="Small examples"></p>
<p>In this picture,&#36;f(x)&#36; It can be well approximated by a straight line, so both linear and secondary regression models are fairly close to real models. It's because... &#36;x_{min}&#36; and &#36;x_{max}&#36; Very close, the target function can be approximated in this area, especially when considering the level of noise observed.</p>
<p>If you calculate these models, &#36;R^2&#36;It'll find a linear model. &#36;R^2 = 0.007&#36;It's a real model. &#36;R^2 = 0.006&#36;I'm sorry. This is a very low value, suggesting that our model is no better than that of constant, although it is already the best linear approximation in this local area.</p>
<p><strong>&#36;R^2&#36;The reason for this is that constant models are already good at this time; because the differences are large, they are difficult to match with the noise.</strong></p>
<p>Now, if &#36;x_{max}&#36; and &#36;x_{min}&#36; What happens when the gap gets bigger? For like &#36;f(x) = \log(x)&#36; This is a monotonous function. &#36;R^2&#36; There's gonna be a strange change.</p>
<p>And then look at a specific example of it. &#36;x_{min} = 1&#36; and &#36;x_{max} = 1000&#36;：</p>
<p><img src="/assets/images/statistical-thinking/r-squared-large-range.png" alt="Large examples"></p>
<p>In this case, it is clear from visual examination that both linear and secondary models are systematically inaccurate (because the logarithmic functions are clearly not linear), but they are. &#36;R^2&#36; And the value went up dramatically: linear models. &#36;R^2 = 0.760&#36;It's a real model. &#36;R^2 = 0.997&#36;。</p>
<p><strong>These examples show that even most people agree that linear models are becoming a near-feature of the real models (i.e., increasingly systemic deviations) and linear models. &#36;R^2&#36; The government has been able to increase the number of people in the country. This means that,&#36;R^2&#36; It could be misleading.</strong></p>
<h2>Conclusions</h2>
<p>These are not content that can be stopped by users. &#36;R^2&#36;, but use the following premise to fully understand:</p>
<ul>
<li>&#36;R^2&#36; The value of the data set used depends to a large extent on the value used;</li>
<li>Even if your model is becoming a growing approximation of the real model,&#36;R^2&#36; The value may also decline (and vice versa).</li>
</ul>
<p>When deciding whether a model is useful, it's high. &#36;R^2&#36; Not necessarily. Low. &#36;R^2&#36; It is not necessarily undesirable.</p>
<p>It is an inescapable question: whether a misdeed model is useful is always dependent on the area where the model is applied and on the way we assess all possible errors in that area. Because... &#36;R^2&#36; It contains an implicit model comparison, which is subject to this general dependence on data sets.</p>
<p>In contrast, proposed convergence indicators such as MSE (average error) and MD (average absolute deviation) are not the only ones that are not the most important.<strong>Its “defect” is also its advantage.</strong>I'm sorry. They lack implicit homogenization and are ostensibly “arbitrary numbers” that are totally affected by the field. This characteristic forces analysts to face up to the sensitivity of the indicators to the application field — compared to the fact that the data are not available in the country.&#36;R^2&#36; The integration of these figures makes them less arbitrary and may also make it seem that model assessments of data dependency are not easy, so that the high R side is blindly perceived as good.</p>
