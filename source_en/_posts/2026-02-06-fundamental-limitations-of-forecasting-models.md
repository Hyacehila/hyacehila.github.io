---
title: 'Fundamental Limits of Foundation Forecasting Models: Multimodality and Rigorous Evaluation'
title_zh: 基础预测模型的基本限制：多模态与严谨评估的必要性
date: 2026-02-06 12:00:00 +0800
categories:
- Data Science
- Time Series & Spatial Data
tags:
- Time Series
- Evaluation
- Multimodality
author: Hyacehila
excerpt: Are deep learning methods always effective for time-series forecasting? Based on Christoph Bergmeir's NeurIPS 2024
  talk, this post discusses model limits and evaluation traps.
description: Are deep learning methods always effective for time-series forecasting? Based on Christoph Bergmeir's NeurIPS
  2024 talk, this post discusses model limits and evaluation traps.
excerpt_zh: 深度学习在时间序列预测中真的总是有效吗？本文基于 Christoph Bergmeir 在 NeurIPS 2024 的演讲，讨论基础预测模型的局限性、评估中的陷阱，以及为什么时间序列预测需要引入多模态上下文。
permalink: /blog/2026/02/06/fundamental-limitations-of-forecasting-models/
lang: en
translation_key: 2026-02-06-fundamental-limitations-of-forecasting-models
translation_status: machine
translation_source_hash: 5f08736e81d29ec4a61dc39a38b5da23150c471b5e11c8879cbb1226c98d5f2d
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<blockquote>
<p>This post is based on <strong>Christoph Bergmeir</strong> Yes. <strong>NeurIPS 2024</strong> The speech. <em>&quot;Fundamental limitations of foundational forecasting models - The need for multimodality and rigorous evaluation&quot;</em>。</p>
<p>Professor Christoph Bergmeir is one of the main maintainers of Monash Time Series Forecasting Library of Monash Series Forecasting Repository.</p>
</blockquote>
<p>The questions in this article can also be addressed<a href="/en/blog/2024/01/30/linear-time-series-analysis-notes/">Linear time series analysis: smooth sequence, ARMA and ARIMA</a>、<a href="/en/blog/2024/05/06/univariate-financial-time-series-analysis-notes/">Financial time series analysis: ARCCH/GARCH effect and volatility modelling</a>How the concept of a relatively close read together is developed in different contexts.</p>
<h2>Introduction: Start with random travel</h2>
<p>Time-series prediction is one of the most fundamental but also the most difficult topics in the data sciences. Unlike image or natural language processing, time series often face extremely low levels of belief.</p>
<p>If it's a completely random sequence of Random Walker -- there's no hidden pattern -- - So the best prediction of the sequence is... <strong>Naive Forecast</strong>, that is, the first-minute observations are projected at the next-minute projection (&#36;y_{t+1} = y_t&#36;I'm not sure. In such cases, however complex models (e.g. BP Neural Networks, SVM, Random Forests, etc.) are often less performing than the simplest of the Naive methods.</p>
<p>This seemingly simple logic is often overlooked in today ' s in-depth learning prediction studies.</p>
<h2>Financial “false SOTA”</h2>
<p>Stock markets are a typical example of random migration. The EMH (EmH) is of the view that stock prices are not a function of past prices, but a reflection of future expectations. The information contained in the equity price is almost already reflected in the current public information, and future changes are mainly influenced by new and unpredictable information.</p>
<p>Thus, stock prices are often considered to be of a moistique (Martingale) nature. The blogger says that the government is not a party to the law.<strong>The accuracy of the Naive prediction is almost impossible in theory.</strong>I'm sorry. In fact, quantitative research in the financial field tends to focus less on mere predictions (i.e., what the stock price is tomorrow) than on more.<strong>Risk (Risk)</strong> and<strong>Volatility</strong>。</p>
<p>Many articles published at the top-of-the-art conference claim to have achieved SOTA (State of the Art) in the area of financial forecasting. But looking at these papers, they tend to be compared only with other in-depth learning (DL) methods, while ignoring the strong benchmark of the Naive prediction. The calculation time of the model is expanding, but performance does not necessarily lead to a substantial breakthrough.</p>
<h2>Weather predictions and wrong benchmarks</h2>
<p>In addition to financial data, in-depth learning researchers are also keen to forecast weather and electricity data. But there is also a common sense error.</p>
<p>The meteorologists generally believe that due to the confusion effects,<strong>Long-term weather predictions over two weeks (14 days) are physically impossible.</strong>I'm sorry. Thus, any model that claims to be capable of producing a long-term (greater than two weeks) accurate hour-by-hour weather prediction is largely acoustic or random speculation.</p>
<p>The paper found that they did compare traditional statistical methods such as ARIMA or EDS, but that benchmarks were often inappropriately set. For example, when faced with weather sequences with complex seasonal (e.g. hourly data), simple ARIMA is not the right opponent. The stronger benchmark should be <strong>DHR-ARIMA (Dynamic Harmonic Regression with ARIMA errors)</strong>I'm sorry. The introduction of this statistical model, which is more complex but more suitable for this type of data, often leads to the failure of the so-called SOTA deep learning model.</p>
<h2>Assessing the trap: Drop Last Trick</h2>
<p>In order to make their models look better, some scholars have even been brain-drived in the assessment process. A typical example is... <strong>&quot;Drop Last Trick&quot;</strong>。</p>
<p>In many of the time series of in-depth learning, data are processed in multiple Batchs. If the last Batch of the test set is not satisfied, some of the coding libraries (such as those that were wrongly configured) <code>DataLoader</code>It's a big deal.</p>
<p>However, in time series projections, the data are orderly.<strong>The last part of the test set is often the most up-to-date, closest and most relevant data</strong>I'm sorry. The arbitrary discarding of this part of the data could lead to a serious distortion of the results of the assessment. In many articles, the same method has a wide variation in performance in different papers, often because different (even less stringent) test criteria are used to create so-called SOTA, and the actual effect may be much less than in Baseline, decades ago.</p>
<h2>Double-edged global Models Sword.</h2>
<p>In recent years, a large number of multi-source time series have been used to construct<strong>Global Models</strong> It's becoming a trend. Only in-depth learning techniques can effectively process this mass of data.</p>
<p>Studies have shown that even training in global models on irrelevant data, with further fine-tuning of the field, may be better than local models. This idea is a long-standing counterpart in basic statistics, called <strong>James-Stein Paradox</strong>I'm sorry. It allows us to improve the predictions using irrelevant data, and to reduce them by introducing deviations (Bias) in exchange for differentials (Variance), which is the basis of the theory of normality.</p>
<p><strong>However, models that are valid across the board do not guarantee their effectiveness in specific local data.</strong></p>
<p>The current large time series model (Foundation Mode for Time Series) often faces a problem: Algorithms combine data from various sources during learning. This “averageization” approach has resulted in models not being able to predict accurately in combination with real scenarios and corresponding training data.<strong>Algorithms can average different hidden models instead of using them separately.</strong>; data sources are also being erased.</p>
<p>While language models are also taught through pre-training of common language models, we can correct models through Prompt in a round of dialogue. This “immediate correction” is much more difficult in the pure time series projection.</p>
<h2>Way out: Context is King</h2>
<p>So, what's the next step in the base prediction model?</p>
<p>The blogger says:<strong>Simple time series lacks enough information to be used</strong>I'm sorry. If we rely on historical data alone, it is difficult to break the random travel restrictions.</p>
<p>♪ Want to be in <strong>LLM4TS (Large Language Models for Time Series)</strong> It's a problem that needs to be addressed. <strong>Context</strong>I'm sorry. The context is not just a longer historical window, but a longer one.<strong>Multimodity</strong> The introduction of information - including text news, macroeconomic reports, image data, etc.</p>
<p>Only when the model understands the text of the “corporate scandal” Context will it be possible to predict a collapse in stock prices; only when the model combines real-time weather cloud maps can it break the bottleneck of pure numerical prediction.</p>
<p><strong>This is the more feasible route for the TS baseline model: from the numerical combination of a single modulus to the context of a multi-modular modulation.</strong></p>
