---
title: 'Anscombe''s Quartet: Visualization and Statistical Illusions'
title_zh: 安斯库姆四重奏：可视化的力量与统计错觉
date: 2026-01-14 12:00:00 +0800
categories:
- Data Science
- Statistical Thinking
tags:
- Statistical Thinking
author: Hyacehila
mathjax: true
excerpt: Are calculations precise while charts are rough? Francis Anscombe's quartet challenges that belief and shows why
  exploratory data analysis matters in statistical inference.
description: Are calculations precise while charts are rough? Francis Anscombe's quartet challenges that belief and shows
  why exploratory data analysis matters in statistical inference.
excerpt_zh: 数值计算是精确的，图表是粗略的？统计学家 Francis Anscombe 用四组特殊的数据集打破了这一成见。本文通过安斯库姆四重奏（Anscombe's Quartet）讨论探索性数据分析（EDA）在统计推断中的必要性。
permalink: /blog/2026/01/14/anscombes-quartet/
lang: en
translation_key: 2026-01-14-anscombes-quartet
translation_status: machine
translation_source_hash: 3b2831dca8c49f1ac8ff4cbd09ed502d4868beedabc3c59c0f174e5e279c1844
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<p>The questions in this article can also be addressed<a href="/en/blog/2026/01/10/statistics-and-truth/">Statistics and Truth: How to use the accident (Statistics and Truth)</a>、<a href="/en/blog/2026/02/04/benfords-law-and-statistical-fraud/">Figures don't lie, but the liars make them up: talk about statistical fraud from Benford's particular law.</a>How the concept of a relatively close read together is developed in different contexts.</p>
<h2>Introduction: Deceptiveness of values</h2>
<p>In data science and statistics, beginners are often caught in the wrong direction:<strong>Overdependence on numerical indicators (Metrics) ignores the distribution patterns of data.</strong></p>
<p>When data are available, averages, differences and related coefficients are calculated immediately, and even the regression model is applied directly. The view that earlier was popular in the statistical community:&quot;The numerical calculations are accurate and the charts are not very useful.&quot;（Numerical calculations are exact, but graphs are rough）。</p>
<p>To counter this, in 1973, the statisticians Francis Anscombe built four special data sets, known as the "Facebooks"<strong>Anscombe Quartet&#39;s Quartet）</strong>I'm sorry. These four sets of data are almost identical in statistical characteristics, but vary considerably in graphic presentation.</p>
<p>It states:<strong>Before statistical extrapolations are made, visualized exploratory data analysis (EDA) is a necessary step to test the validity of the extrapolation.</strong></p>
<h2>Anscom Quartet Data Set</h2>
<p>Look at these four sets of data first. They're four pairs. &#36;(x, y)&#36; Composition of variables:</p>
<table>
<thead>
<tr>
<th align="left">Observations</th>
<th align="left">&#36;x_1&#36;</th>
<th align="left">&#36;y_1&#36;</th>
<th align="left">&#36;x_2&#36;</th>
<th align="left">&#36;y_2&#36;</th>
<th align="left">&#36;x_3&#36;</th>
<th align="left">&#36;y_3&#36;</th>
<th align="left">&#36;x_4&#36;</th>
<th align="left">&#36;y_4&#36;</th>
</tr>
</thead>
<tbody><tr>
<td align="left">1</td>
<td align="left">10.0</td>
<td align="left">8.04</td>
<td align="left">10.0</td>
<td align="left">9.14</td>
<td align="left">10.0</td>
<td align="left">7.46</td>
<td align="left">8.0</td>
<td align="left">6.58</td>
</tr>
<tr>
<td align="left">2</td>
<td align="left">8.0</td>
<td align="left">6.95</td>
<td align="left">8.0</td>
<td align="left">8.14</td>
<td align="left">8.0</td>
<td align="left">6.77</td>
<td align="left">8.0</td>
<td align="left">5.76</td>
</tr>
<tr>
<td align="left">3</td>
<td align="left">13.0</td>
<td align="left">7.58</td>
<td align="left">13.0</td>
<td align="left">8.74</td>
<td align="left">13.0</td>
<td align="left">12.74</td>
<td align="left">8.0</td>
<td align="left">7.71</td>
</tr>
<tr>
<td align="left">4</td>
<td align="left">9.0</td>
<td align="left">8.81</td>
<td align="left">9.0</td>
<td align="left">8.77</td>
<td align="left">9.0</td>
<td align="left">7.11</td>
<td align="left">8.0</td>
<td align="left">8.84</td>
</tr>
<tr>
<td align="left">5</td>
<td align="left">11.0</td>
<td align="left">8.33</td>
<td align="left">11.0</td>
<td align="left">9.26</td>
<td align="left">11.0</td>
<td align="left">7.81</td>
<td align="left">8.0</td>
<td align="left">8.47</td>
</tr>
<tr>
<td align="left">6</td>
<td align="left">14.0</td>
<td align="left">9.96</td>
<td align="left">14.0</td>
<td align="left">8.10</td>
<td align="left">14.0</td>
<td align="left">8.84</td>
<td align="left">8.0</td>
<td align="left">7.04</td>
</tr>
<tr>
<td align="left">7</td>
<td align="left">6.0</td>
<td align="left">7.24</td>
<td align="left">6.0</td>
<td align="left">6.13</td>
<td align="left">6.0</td>
<td align="left">6.08</td>
<td align="left">8.0</td>
<td align="left">5.25</td>
</tr>
<tr>
<td align="left">8</td>
<td align="left">4.0</td>
<td align="left">4.26</td>
<td align="left">4.0</td>
<td align="left">3.10</td>
<td align="left">4.0</td>
<td align="left">5.39</td>
<td align="left">19.0</td>
<td align="left">12.50</td>
</tr>
<tr>
<td align="left">9</td>
<td align="left">12.0</td>
<td align="left">10.84</td>
<td align="left">12.0</td>
<td align="left">9.13</td>
<td align="left">12.0</td>
<td align="left">8.15</td>
<td align="left">8.0</td>
<td align="left">5.56</td>
</tr>
<tr>
<td align="left">10</td>
<td align="left">7.0</td>
<td align="left">4.82</td>
<td align="left">7.0</td>
<td align="left">7.26</td>
<td align="left">7.0</td>
<td align="left">6.42</td>
<td align="left">8.0</td>
<td align="left">7.91</td>
</tr>
<tr>
<td align="left">11</td>
<td align="left">5.0</td>
<td align="left">5.68</td>
<td align="left">5.0</td>
<td align="left">4.74</td>
<td align="left">5.0</td>
<td align="left">5.73</td>
<td align="left">8.0</td>
<td align="left">6.89</td>
</tr>
</tbody></table>
<p>The figures in the table are not per se visible, but when statistics are calculated, a discrepancy emerges.</p>
<h2>Statistical trap: perfect disguise</h2>
<p>If no maps are drawn, the results will be highly consistent, relying only on the usual descriptive statistical volumes and linear regression models to analyse the four data sets.</p>
<h3>1. Descriptive statistics</h3>
<p>The average and the variance are calculated for each of the four sets of data, as follows (to 2-3 decimal places):</p>
<ul>
<li><strong>Mean (Mean)</strong>：<ul>
<li>&#36;E(x) = 9.00&#36; (Four groups are identical)</li>
<li>&#36;E(y) = 7.50&#36; (Four groups are identical)</li>
</ul>
</li>
<li><strong>Sample variance (Variance)</strong>：<ul>
<li>&#36;Var(x) = 11.00&#36; (Four groups are identical)</li>
<li>&#36;Var(y) \approx 4.12&#36; (Four groups are essentially the same)</li>
</ul>
</li>
<li><strong>Related coefficient (Correlation)</strong>：<ul>
<li>&#36;Corr(x, y) \approx 0.816&#36; (Four groups are identical)</li>
</ul>
</li>
</ul>
<h3>2. Return analysis</h3>
<p>If you're assuming &#36;y&#36; and &#36;x&#36; Linear relationships exist and the lowest hyperbolic method (OLS) is used to develop linear regression models &#36;y = \beta_0 + \beta_1 x + \epsilon&#36;The parameters of the four models will also be very consistent:</p>
<ul>
<li><strong>Intersection (%2)&#36;\beta_0&#36;)</strong>: About 3.00</li>
<li><strong>Slope (%2)&#36;\beta_1&#36;)</strong>: About 0.50</li>
<li><strong>Proposed Preference (Present)&#36;R^2&#36;)</strong>: About 0.67</li>
</ul>
<p><strong>Conclusions</strong>: If we look only at the above figures, we are likely to think that the four sets of data reflect the same pattern:&#36;x&#36; and &#36;y&#36; There are significant positive and relevant linear relationships and the extent to which models are ready to be developed is good.</p>
<p><strong>However, this conclusion is not valid.</strong></p>
<h2>Visualizing the revelation: Charts don't lie</h2>
<p>The differences between these four sets of data are clearly apparent when they are drawn into a scattered map.</p>
<p><img src="https://upload.wikimedia.org/wikipedia/commons/e/ec/Anscombe%27s_quartet_3.svg" alt="Anscombe&#x27;s Quartet"></p>
<p>Look at these four images one by one:</p>
<ol>
<li><strong>Dataset I (top left):</strong> Data points are roughly evenly distributed on both sides of the regression line, and random error appears to be subject to normal distribution. For this group of data, linear regression models are appropriate.</li>
<li><strong>Dataset II (top right):</strong> It's a clear one.<strong>Non-linear relationships</strong>(Looks like a reverse parabolic line. There is a strong relationship of certainty between the data, but the curve is combined with linear models (lines) that ignore the structure between variables.&#36;R^2=0.67&#36; It's of limited significance here.</li>
<li><strong>Dataset III (bottom left):</strong> This is a...<strong>Strong Impact Point (Outlier)</strong> The typical case. The vast majority of data points exhibit near-perfect linear relationships (the correlation coefficient is almost 1) but an off-group value changes the slope of the entire regression line and reduces the correlation coefficient. Without visualization, it is difficult to identify this anomaly and to decide whether it should be removed or analysed separately.</li>
<li><strong>Dataset IV (bottom right):</strong> This is a...<strong>High Leverage Point</strong> The extreme case.&#36;x&#36; All but one point of the same value (%2)&#36;x=8&#36;I'm not sure. Linear relationships depend on this extreme observation. If we remove this point,&#36;x&#36; and &#36;y&#36; There's no lineage between them. &#36;x&#36; is the constant. The results of the model are determined almost by a single data point and are therefore very vulnerable.</li>
</ol>
<h2>In-depth exploration: Why is visualization important for statistical inferences?</h2>
<p>Anscom Quartet shows the risk of being ignored in statistical inferences.</p>
<h3>1. Validation of model assumptions (Assumptions Checking)</h3>
<p>Classic statistical models (e.g. linear regression) are usually based on a series of strict assumptions, such as:</p>
<ul>
<li><strong>Linearity</strong>: Linear relationships between variables and cause variables.</li>
<li><strong>Homoscedastability</strong>: The error item is constant.</li>
<li><strong>Normality (Normality)</strong>: The error items are subject to normal distribution.</li>
<li><strong>Independence</strong>: Samples are independent of each other.</li>
</ul>
<p><strong>The numerical indicators themselves cannot judge whether these assumptions are valid or not.</strong>I'm sorry. For example, in Datat II, values show correlation, but the graphics immediately negated&quot;Linear&quot;Assumptions. ♪ With the help of<strong>Residual Plt</strong> We can only diagnose the applicability of the model when visualization is done.</p>
<h3>2. Identification of discrete values and strong effects Points</h3>
<p>As shown in Dataet III and IV, individual ioning values are sufficient to change the statistical characteristics of the entire model. Averages and squares are sensitive to discrete values (no. Before calculating and extrapolating, by<strong>Boxline (Boxplot)</strong> Or the identification and treatment of anomalies in the sprawl map is an essential step in increasing the inference of robustness.</p>
<h3>3. &quot;Data Form&quot;Better than&quot;Data indicators&quot;</h3>
<p>In addition to focusing on central trends (average values), modern data analysis needs to observe the distribution patterns of data. Skeweness and Kurtosis, while describing distribution, is still not as good as<strong>Histogram (Histogram)</strong> or <strong>Estimates of nuclear density (KDE)</strong> Intuitive.</p>
<h2>Concluding remarks</h2>
<p>The caution of Francis Anskum, which was expressed 50 years ago, remains valid in today ' s big data age. The more common automation machine learning (AutoML) and black box models are, the more easily people lose sight of the raw data.</p>
<p><strong>Do not rely on statistics without charts. No, I'm not.</strong></p>
<p>I'm pressing it. &quot;Run Model&quot; button, draw a picture. This is both a good working habit and a first confirmation that the data really support subsequent inferences.</p>
<h2>Appendix: Python Code Achieved</h2>
<p>If you want to reproduce the four sets of data and charts in person, you can use the following Python code:</p>
<pre><code class="language-python">import matplotlib.pyplot as plt
import numpy as np

# Anscombe&#39;s Quartet Data
x = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
y1 = [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]
y2 = [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74]
y3 = [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73]
x4 = [8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8]
y4 = [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89]

datasets = [(x, y1), (x, y2), (x, y3), (x4, y4)]

fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
axes = axes.flatten()

for i, (xi, yi) in enumerate(datasets):
    ax = axes[i]
    # 绘制散点
    ax.scatter(xi, yi, color=&#39;orange&#39;, edgecolor=&#39;k&#39;, s=60)
    # 绘制回归线
    m, b = np.polyfit(xi, yi, 1)
    ax.plot(np.array([4, 19]), m * np.array([4, 19]) + b, color=&#39;blue&#39;, alpha=0.6)

    ax.set_title(f&#39;Dataset {i+1}&#39;)
    ax.set_xlim(3, 20)
    ax.set_ylim(3, 13)

plt.tight_layout()
plt.show()
</code></pre>
