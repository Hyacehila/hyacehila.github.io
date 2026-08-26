---
title: SVD Decomposition and Patterns in Multivariate Time Series
title_zh: 多元时间序列的 SVD 分解与模式
date: 2026-01-12 12:00:00 +0800
categories:
- Data Science
- Time Series & Spatial Data
tags:
- Time Series
- Dimensionality Reduction
author: Hyacehila
mathjax: true
hidden: true
excerpt: For an n by T time-series matrix X, SVD decomposes it into U, Sigma, and V transpose. This post explains spatial
  patterns, temporal patterns, and singular values.
description: For an n by T time-series matrix X, SVD decomposes it into U, Sigma, and V transpose. This post explains spatial
  patterns, temporal patterns, and singular values.
excerpt_zh: 对于一个 n×T 的时间序列矩阵 X，SVD 分解可以得到 X=UΣV^T。本文解释得到的矩阵在时空模式挖掘中的物理意义，包括空间模式 U、时间模式 V 以及奇异值 Σ 的含义。
permalink: /blog/2026/01/12/svd-decomposition-and-patterns-of-multivariate-time-series/
lang: en
translation_key: 2026-01-12-svd-decomposition-and-patterns-of-multivariate-time-series
translation_status: machine
translation_source_hash: 20a3f7e2bd2e7489cc142473e4eb9f1682788a2dc0075dea49be0772d7096738
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<p>For one. &#36;n\times T&#36; Multiple Time Series Matrix &#36;X&#36;, of which &#36;n&#36; is the number of channels (or variables),&#36;T&#36; is the time node (observation number). Disassemble it from SVD. &#36;X=U\Sigma V^T&#36;I'm sorry. The implications of the decomposition matrices and their application are explained below.</p>
<p>The questions in this article can also be addressed<a href="/en/blog/2024/01/30/linear-time-series-analysis-notes/">Linear time series analysis: smooth sequence, ARMA and ARIMA</a>、<a href="/en/blog/2024/05/06/univariate-financial-time-series-analysis-notes/">Financial time series analysis: ARCCH/GARCH effect and volatility modelling</a>How the concept of a relatively close read together is developed in different contexts.</p>
<h3>Meaning of the whole</h3>
<p>For Matrix &#36;U&#36;It represents the data.<strong>Space model</strong>, each column is a linear combination of a characteristic channel.</p>
<p>For Matrix &#36;V&#36;It represents the data.<strong>Time Mode</strong>, each column corresponds to the temporal evolution of the above spatial pattern.</p>
<p>For the Odd Matrix &#36;\Sigma&#36;It represents.<strong>Importance of space-time patterns</strong>I'm sorry. The greater the odd value, the greater the corresponding pattern in the original data and the more the variance in interpretation.</p>
<h3>A visual example of physical meaning.</h3>
<p>For Space Mode &#36;U&#36; , and then the column vectors. &#36;u_i&#36;We're considering the most exotic column vector. &#36;u_1&#36;The most important space model:</p>
<ul>
<li>Yes.<strong>Multi-area temperature sequence studies</strong>We might find out. &#36;u_1&#36; All the elements that indicate the northern region are larger, while the southern region is smaller. That means most of all.<strong>Temperature change patterns are national spatial differences.</strong>(e.g. North-South differences).</li>
<li>Yes.<strong>EEG data</strong>，&#36;u_1&#36; It may mean that the activity is more intense in a given brain zone.</li>
</ul>
<p>In short, space mode. &#36;U&#36; I've measured us.<strong>Overall differences in observation space</strong>。</p>
<p>And look at the most important time patterns. &#36;v_1&#36;I'm sorry. It's a time series:</p>
<ul>
<li>Yes.<strong>Temperature issues</strong>，&#36;v_1&#36; It may be seasonal, representing temperature fluctuations at any time.</li>
<li>Yes.<strong>EMPLE EMPLOYMENT DATA</strong>，&#36;v_1&#36; It may be the moment when a given brain region is activated.</li>
</ul>
<p>As for the eccentric matrix, &#36;\Sigma&#36;It's on the line.<strong>Measuring the contribution of this (space-time) effect to the overall data</strong>。</p>
<h3>In-depth analysis and application</h3>
<h4>1. Study of differences and linkages between corridors (focus on matrix) &#36;U&#36;）</h4>
<p>When we focus on research,<strong>Differences and linkages between time series corridors</strong>♪ And when we should ♪<strong>Focus on the study matrix &#36;U&#36;</strong>。</p>
<ul>
<li><strong>Community Discovery</strong>: In the most important vectors, the absolute values of some elements are large, often strongly correlated, forming corresponding communities.</li>
<li><strong>Key Node Identification</strong>: those in multiple &#36;u_i&#36; Accessors that have a large weight in vectors are often key nodes of the system.</li>
<li><strong>Unusual detection</strong>: If we know that certain corridors should be closely linked, but the data analysis shows that they are &#36;U&#36; The performance was inconsistent, which may indicate that the data had produced anomalies.</li>
</ul>
<h4>2. Study of dynamic characteristics of time (focus on matrix) &#36;V&#36;）</h4>
<p>When we focus on research,<strong>Dynamic characteristics of time</strong>And when it's time, it's time to focus on the matrix. &#36;V&#36;。</p>
<ul>
<li><strong>Frequency analysis</strong>: For the most important time mode vector &#36;v_i&#36; The FFT is used to determine the main frequency of the shock.</li>
<li><strong>Incident monitoring</strong>: A space model may represent a particular event. The occurrence of the incident can be confirmed by monitoring the time pattern of the incident.</li>
<li><strong>Trends capture</strong>: The most important time pattern typically captures the slowest and sustained global trend. As the oddly odd values decline, the degree of persistence of the corresponding pattern tends to decrease, and the importance thereof decreases (high frequency noise is usually at the end).</li>
</ul>
<h4>3. Data compression</h4>
<p>SVD breakdown<strong>It can also be used for data compression.</strong>, similar to other matrix decomposition. We can choose to lower the matrix. &#36;U&#36; Or a matrix. &#36;V&#36; Other Organiser</p>
<ul>
<li>Lower &#36;U&#36; : Focus on reducing the number of features (passages).</li>
<li>Lower &#36;V&#36; Dictional: Focus on reducing the length of time series, with a small number of representative moments representing the entire sample.</li>
</ul>
<p>It's a long observation.&#36;T&#36; It's a big one, which means that when the sample is too big, it can be considered.</p>
