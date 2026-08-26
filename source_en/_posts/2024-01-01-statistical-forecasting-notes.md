---
title: 'Statistical Forecasting: Qualitative Methods, Quantitative Methods, and Error Evaluation'
title_zh: 统计预测：定性预测、定量预测与趋势外推
date: 2024-01-01 15:06:38 +0800
categories:
- Data Science
- Time Series & Spatial Data
tags:
- Statistics
- Forecasting
author: Hyacehila
mathjax: true
hidden: true
excerpt: Covers qualitative and quantitative forecasting, trend extrapolation, smoothing methods, and forecast error evaluation.
description: Covers qualitative and quantitative forecasting, trend extrapolation, smoothing methods, and forecast error evaluation.
excerpt_zh: 整理定性预测、定量预测、趋势外推、平滑预测和预测误差评估等基础方法。
permalink: /blog/2024/01/01/statistical-forecasting-notes/
lang: en
translation_key: 2024-01-01-statistical-forecasting-notes
translation_status: machine
translation_source_hash: 8d3a8011163a59e4bd1661483788b49b08304758f4b013f70f5e019697d753d8
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<h2>Overview of statistical policymaking</h2>
<h3>Concept of statistical projections</h3>
<p>From a working point of view, statistical projections refer to the use of statistical methods to judge what is uncertain or unknown in the future, that is, to analyze what is unknown in the past.
In terms of results, it's statistical projections. Value
In terms of discipline, he's a discipline of science that reveals objective patterns of things.
Currently, the use of statistical predictions is very extensive, breaking the boundaries of natural and social sciences, and theoretical research is constantly improving, absorbing a lot of nutrients from other disciplines to improve itself.</p>
<h3>Classification of statistical projection methods</h3>
<p>Qualitative predictions are based on subjective judgement, often relying on expert experience.</p>
<ul>
<li>To make descriptive predictions about the nature of things, we can't always give quantitative models for very complex questions.</li>
<li>Emphasis on trends and trends in the development of things, major shifts (for the reasons given above)
Qualitative predictions rely on historical and current data to create mathematical models to calculate accurate predictions.</li>
<li>Emphasis on accurate projections of quantitative values</li>
<li>Emphasizing the role of historical information and mathematical models
Quantitative predictions do not give good results when the structure of the system of predicting objects changes qualitatively and when statistics fluctuate dramatically.</li>
</ul>
<h3>Principles and steps for statistical forecasting</h3>
<h4>Principles</h4>
<ul>
<li>The principle of consistency: the regularity of the statistical projection of objects applies to the past, the present and the future for some time.</li>
<li>Relevant analogy principle: changes in the statistical projection audience can be studied in the light of changes in the statistical projection audience and other related volumes</li>
<li>The principle of probability: the statistical projection of changes is both accidental and inevitable.</li>
<li>Systemic principles: looking at the problem from a global perspective, looking at not only nature but also other quantities of systems in the system and other related systems</li>
</ul>
<h4>Steps</h4>
<ul>
<li>Targeting of projections</li>
<li>Collection of historical and practical information</li>
<li>Establishment of appropriate statistical forecasting models</li>
<li>Forecast and evaluate models</li>
</ul>
<h3>State of development of statistical projections</h3>
<p>In the decades that have passed, statistical projections have developed hundreds of prediction methods that can address a wide range of issues.
But in the real world, predictors can be very complex systems, looking for adaptive and effective prediction models. Far
So far, some of the more innovative predictions have been made.</p>
<ul>
<li>Fuzzy projections in uncertainty projections</li>
<li>Gray projections in uncertainty projections</li>
<li>Portfolio projections
In this book, we will only study the concepts of more traditional and mature statistical prediction methods and statistical predictions, and those elements worth separate studies will be studied separately, such as time series analysis, regression predictions, neuronet predictions, etc.</li>
</ul>
<h3>Relationship of statistical projections to decision-making</h3>
<ul>
<li>Accurate statistical projections guide decision-making</li>
<li>Decision-making will be counterproductive to predictions.</li>
<li>The two interact with each other.</li>
</ul>
<h2>Qualitative projection methods</h2>
<h3>Overview</h3>
<p>The concept of a qualitative projection method has been introduced earlier; his core point is to rely on human subjectivity to achieve a projection, with the greatest advantage being flexibility; we will then present more commonly used qualitative projection methods, but not many, because of the flexibility of the qualitative projection method, which is essentially a qualitative projection in all sentences, without a very rigorous approach.</p>
<h3>Diffle.</h3>
<h4>Overview</h4>
<p>Diffel is a classic method of predicting by relying on expert opinion, and he gives a set of advices at night.</p>
<h4>Process</h4>
<ul>
<li>Creation of thematic planning groups (including professionals and statisticians)</li>
<li>Selection of assessment experts 10-50 most suitable persons could be appropriately expanded</li>
<li>Use of correspondence for expert advice </li>
<li>In the first round of consultations, only the projection targets and related information were given without restriction</li>
<li>Second round of consultations with experts on the results of the ongoing previous round of statistics</li>
<li>Until the results of the predictions are almost exhausted.</li>
</ul>
<h4>Characteristics</h4>
<ul>
<li>Anonymous: all experts conceal each other First Name</li>
<li>Feedback: multiple rounds of feedback giving forecast results</li>
<li>Authority: selection of authoritative experts</li>
<li>Condensity: final forecast results should be reduced after multiple rounds of feedback</li>
<li>Quantitative: The views of a small number of experts cannot be ignored, with half of them projecting results beyond two four-point points to ensure that experts are not too homogeneous.</li>
</ul>
<h4>Statistical processing</h4>
<p>It's customary for two quartiles and medians of Diffel's return projections.
Medium as the projection result, four-point to measure the degree of fragmentation of the projection</p>
<h2>Extrapolation of trends</h2>
<h3>Summary of extrapolation of trends</h3>
<h4>Concept</h4>
<p>Trends extrapolation is an analytical method of time-series data that should have been part of the time-series analysis but was singled out because it was too simple; it can be described as an extension and contraction of the multi-formulation of this mathematically combined approach, where only time-series data can be processed, and promotion is no longer subject to multiple functions, but new scenarios are introduced.
Its rationale is:</p>
<ul>
<li>The factors that determine what's going on in the past determine what's going to happen in the future.</li>
<li>Things tend to evolve gradually, not leapfrog.
Therefore, calculating the time series over time&#36;t&#36; Changing Functions&#36;f(t)&#36; So the moment of the future.&#36;t&#36; There's a scientific principle to the function.</li>
</ul>
<h4>Classification</h4>
<ul>
<li>Multiple curve prediction model</li>
<li>Index curve prediction model</li>
<li>A logarithmic curve prediction model</li>
<li>Growth curve prediction model</li>
</ul>
<h4>Model selection</h4>
<h5>Graphical Recognition</h5>
<p>Directly draw break-up maps of time-series data Comparison of graphic features with commonly used trend extrapolations</p>
<h5>Difference calculation method</h5>
<p>We can define the difference.
xx
Same thing.</p>
<p>The questions in this article can also be addressed<a href="/en/blog/2023/03/18/stochastic-process-basics-notes/">Random process basis: random process definition, digital characteristics and smooth process</a>、<a href="/en/blog/2024/01/30/linear-time-series-analysis-notes/">Linear time series analysis: smooth sequence, ARMA and ARIMA</a>How the concept of a relatively close read together is developed in different contexts.</p>
<p>By comparing the differential characteristics of time series data to be studied and the differential characteristics of common models for extrapolation of trends, we can choose the models that we want to use.</p>
<ul>
<li>The 2nd grade is divided into zero.</li>
<li>A multi-formula model with an equal margin for each stage.</li>
<li>The margin of the two-step difference equals the index model</li>
</ul>
<h3>Multiple Curve Extrapolation</h3>
<p>The basic shape of the polycurve model is
&#36;&#36;y_{t}=\beta_{0}+\beta_{1}t+\beta_{2}t^{2}+\cdots+\beta_{n}t^{n}&#36;&#36;
There's no need for a single multiform study of the meta-regression and the use of the minimum binary directly presented in our online statistical model.</p>
<h4>A binary polycurve projection</h4>
<p>The basic form of the problem is
&#36;&#36;y_{t}=\beta_{0}+\beta_{1}t+\beta_{2}t^{2}&#36;&#36;
There's a minimum of two times and three points for more common treatment.
For the minimum binary, we just need to swap the binary and construct a binary regression model.
The basic idea of the three-point approach is to take a long distance, and close to three points are the point where the predictive model determines the parameters.<br>The three-point operation is:</p>
<ul>
<li>The total number of observations in the time series is odd (the first observation is deleted in even numbers) </li>
<li>Based on&#36;n&#36;The size of the selection is far and wide.&#36;k&#36; Group</li>
<li>Usage weight&#36;w_i=i&#36; From far to near to empower three points on the conic.</li>
<li>General &#36;n&gt;15&#36;时 &#36;k=5&#36; 否则&#36;k = &#36;3.00 General time series data less than 8 non-analysis
There's a dollar down there.&gt;A three-point calculation example of &#36;15.
Let's say we all share.&#36;n&#36; Medium&#36;d=\frac{n+1}{2}&#36;
Five in the forward.
&#36;bar}<em>{1}=\frac{y</em>{1}+2y_{2}+3y_{3}+4y_{4}+5y_{5&#125;&#125;{1+2+3+4+5}.&#36;&#36;
&#36;&#36;\overline{t}<em>{1}=\frac{1\times1+2\times2+3\times3+4\times4+5\times5}{1+2+3+4+5}=\frac{11}3&#36;&#36;
中期五个有
&#36;&#36;\bar{y}</em>{2}=\frac{y_{d-2}+2y_{d-1}+3y_{d}+4y_{d+1}+5y_{d+2&#125;&#125;{1+2+3+4+5}.&#36;&#36;
时间同理计算
近期五个有
&#36;&#36;\bar{y}<em>{3}=\frac{y</em>{n-4} +2y {n-3} +3y {n-2} +4y n m m n} +5y {n} {1+2+3+4+5}. &#36;
Time-to-time calculations
Three points, three.</li>
</ul>
<h4>Triple polycurve</h4>
<p>It's possible to use the exchange rate to analyze the minimum two-fold method.
Select the left, the middle, the left, the middle, the right, the right, the four dots to determine multiple functions and then analyze them.
Normal &#36;n at this time&gt;20&#36;时 &#36;k=5&#36; 否则&#36;k = &#36;3
The appropriate deletion of one or two of the data allows us to construct four segments of the same distance, and the weight is still going from far to near. Empowerment
<strong>The analysis using the Minimal Double Multiplication method is often more accurate.</strong></p>
<h3>Index curve extrapolation</h3>
<h4>Index curve forecast</h4>
<p>The normal form of the function is as
That's right.<em>{t}=\beta</em>{0}\beta_{1}^{t}\left(\beta_{0},\beta_{1}&gt;0\right)&#36;&#36;
对数化处理有
&#36;&#36;Other Organiser
Now you can use the exchange method to convert it to a linear model, and you can use the minimum binary.
All we have to do is finally restore the data.</p>
<h4>Amending the index curve forecast</h4>
<p>The normal form of the function is as
That's right.<em>{t}=\beta</em>Other Organiser
First calculate&#36;\beta_0&#36;  The transfer is then determined by using a minimum of two multipliers.</p>
<h3>Logarithmic Extrapolation</h3>
<p>The normal form of the function is as
That's right.<em>{t}=\frac{L}{1+\beta</em>{0}e^{-\beta_{1}t&#125;&#125;,&#36;&#36;
&#36;L&#36; It's the limit of growth.
Usually after indexing or logarithmic deformation, use the minimum binary. Break</p>
<h3>Growth curve extrapolation</h3>
<p>The growth curve is the curve of the general pattern of development of things.</p>
<h4>Gompertz Curve</h4>
<p>The function of the curve is as
&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36; K==K&gt;Yeah, right.
It's calculated by taking natural logarithms on both sides and turning it into a modified index model.</p>
<h4>Logistic Curve Model</h4>
<p>The function form of the logistic model is the solution of a differential equation.
&#36;&#36;y_{t}=\frac{k}{1+ae^{-bt&#125;&#125;,&#36;&#36;
We can turn the problem into a modified index model using a minimum of two times the calculation.</p>
<h3>Curved Preference Analysis</h3>
<p>Brief introduction of some indicators used to measure the proposed advantages of the curve
In fact, we've introduced a lot of indicators into the regression analysis to measure the proposed advantages of the curve, including:
Discrepancies squared and RSS average error MSE, and we'll introduce a few of the time series.
&#36;STE=sqrt{frac{sum}\left}<em>{t}\right)^{2&#125;&#125;{T&#125;&#125;&#36;&#36;
&#36;&#36;R</em>{\mathrm{NL&#125;&#125;=\sqrt{\frac{\sum_{t=1}^{T}(y_{t}-\hat{y}<em>{t})^{2&#125;&#125;{\sum</em>{t=1}^{T}y_{t}^{2&#125;&#125;},&#36;&#36;
&#36;&#36;FR=\frac{\sum_{t=1}^{T}y_{t}\hat{y}<em>{t&#125;&#125;{\sqrt{\sum</em>{t=1}^{T}y_{t}^{2}\sqrt{\sum_{t=1}^{T}\hat{y}_{t}^{2&#125;&#125;&#125;&#125;.&#36;&#36;</p>
<h2>Markov Forecast Method</h2>
<h3>Overview</h3>
<p>The Marcov chain and the Markov process are two of the most fundamental elements of a random process of non-revalence, in which we have presented their basic theory; in the real world, non-revalidity is a very common feature, which also marks the important role that the Marx chain will play in statistical prediction.
In most studies, we tend to think that the model matches a Zilong Mask. Chain</p>
<h3>Forecast methodology</h3>
<h4>Split Status Space</h4>
<p>First, we need to determine the medium-state of the target to be projected; it can be distinguished from the apparent state boundaries of the target to be projected, or it can be judged by the actual situation; and it is important to consider the intended purpose and ensure the comprehensiveness of the state.</p>
<h4>Determining the status of information</h4>
<p>Determines the status of the data series for each time period according to the predefined status</p>
<h4>Statistical analysis</h4>
<p>We're going to do a statistical analysis of what's available, calculate a one-step transfer probability matrix.
Use frequency instead of probability.
&#36;&#36;p_i=P{X=i}=\frac{n_i}M,&#36;&#36;
Use frequency instead of probability.
&#36;&#36;p_{ij}=\frac{n_{ij&#125;&#125;{n_{i&#125;&#125;&#36;&#36;</p>
<h4>Further analysis</h4>
<p>Determine the initial distribution, calculate the absolute distribution, discuss the history, determine the smooth distribution.
We've studied all of this at random. </p>
<h2>Portfolio projection methodology</h2>
<h3>Overview</h3>
<p>Portfolio predictions are similar to the idea of integrated learning, and we need to build on the advantages of multiple individual methods to achieve better predictions by combining multiple individual methods.
The combined projection methodology allows for the following broad classifications:</p>
<ul>
<li>Linear combination predictions and non-linear combination predictions based on the function relationship of the combination projection to the individual projection (very common classification, more linear use)</li>
<li>Depending on the method of calculation of the weighted factor, the optimal combination projection is divided into the non-optimal combination projection, the difference being whether or not a very small word is used or a certain target function is significantly modified to calculate the weight value</li>
<li>The variable portfolio forecast and the non-variable portfolio forecast, whether our weight values will change according to time or some variable, and the current variable portfolio projection is still limited due to the complexity of the results</li>
<li>Categorized into a non-poor combination of projections and an excellent combination of projections based on whether they are better than individual projections</li>
</ul>
<h3>Methodology for determining models that are not optimal combinations of projections</h3>
<h4>Algorithm</h4>
<p>Give them the same weight.
Use linear models</p>
<h4>Error Square and Last</h4>
<p>We can use it to improve weights.&#36;E_{ii}&#36; is the predictive error squared and weighting
&#36;&#36;l_{i}=\frac{E_{ii}^{-1&#125;&#125;{\sum_{i=1}^{m}E_{ii}^{-1&#125;&#125;,&#36;&#36;</p>
<h4>Average error countdown</h4>
<p>Remember&#36;E_{ii}&#36; is the predictive error squared and weighting
&#36;&#36;l_{i}=\frac{E_{ii}^{-\frac{1}{2&#125;&#125;}{\sum_{i=1}^{m}E_{ii}^{-\frac{1}{2&#125;&#125;},&#36;&#36;</p>
<h4>Simple weighted average method</h4>
<p>We sequence the error of the different error models so that the smallest error models have the greatest weight and naturally give an improved weight.
&#36;&#36;l_{i}=\frac{i}{\sum_{i=1}^{m}i}=\frac{2i}{m(m+1)}&#36;&#36;</p>
<h3>Projection error squared and minimum linear combination prediction model</h3>
<p>It's a linear planning problem.
&#36; \begin{aligned}
&amp;\min J_{1}=\sum_{t=1}^{N}\sum_{i=1}^{m}\sum_{j=1}^{m}l_{i}l_{j}e_{it}e_{jt}, \
&amp;I don't know, s.t.
I'm sorry.
Use software to solve it.</p>
<h3>Optimal combination projection model based on relevant coefficients</h3>
<p>Agreed symbol
&#36;.A^mathrm{T}A=left[\begin{array}c}e 1\mathrm}e}mathrm{e}m^mathrm}e}end}right.\right [e 1,e 2,\cdots,e m]=left[\begin{array}e 1\mathrm}e{T}e 1 &amp;e_1^\mathrm{T}e_2&amp;\cdots&amp;e_1^\mathrm{T}e_m\e_2^\mathrm{T}e_1&amp;e_2^\mathrm{T}e_2&amp;\cdots&amp;e_2^\mathrm{T}e_m\\vdots&amp;\vdots&amp;&amp;\vdots\e_m^\mathrm{T}e_1&amp;e_m^\mathrm{T}e_2&amp;\cdots&amp;e_m^\mathrm{T}e_m\end{array}\right]=E.&#36;&#36;
 &#36;&#36;\begin{aligned}
J_{1}&amp; =\sum_{t=1}^{N}\sum_{i=1}^{m}\sum_{j=1}^{m}l_{i}l_{j}e_{it}e_{jt}=\sum_{i=1}^{m}\sum_{j=1}^{m}\left[l_{i}l_{j}\left(\sum_{t=1}^{N}e_{it}e_{jt}\right)\right]  \
&amp;=\sum_{i=1}^{m}\sum_{j=1}^{m}\left[l_{i}l_{j}E_{ij}\right]=L^{T}EL.
\end{aligned}&#36;&#36;
那么机遇相关系数的最优线性组合预测模型为以下规划问题
&#36;&#36;\max R(l_1,l_2,\cdots,l_m)=\frac{\sum_{i=1}^ml_i\sum_{t=1}^Ne_te_{it&#125;&#125;{\sqrt{\sum_{t=1}^Ne_t^2\sqrt{L^\mathrm{T}EL&#125;&#125;},&#36;&#36;
&#36;&#36;\left.\mathrm{s.t.}\left{\begin{array}{l}\sum_{i=1}^{m}l_i=1,\i=1\l_i\geqslant0,i=1,2,\cdots,m.\end{array}\right.\right.&#36;&#36;</p>
<h3>A combination projection method based on IOWA algorithms</h3>
<h4>OWA and IOWA.</h4>
<p>American scholars proposed orderly weighted averages and induced weighted averages
The conventional weighted arithmetic averages are their exception, and they have been widely applied in many fields;
Definitions: set &#36;mathrm{OWA}<em>W:\mathbb{R}^n\to\mathbb{R}&#36; 是&#36;n&#36;维函数&#36;W=\left[w</em>{1},w_{2},\cdots,w_{n}\right]^{\mathrm{T&#125;&#125;&#36; 是加权向量 满足&#36;{\cHFFFFFF}{\cH00FFFF} {\cHFFFFFF}&#36;1 if
&#36; \\mathrm{OWA}<em>W[a_1,a_2,\cdots,a_n]=\sum</em>{i=1}^nw_ib_i,&#36;&#36;
其中&#36;b_i&#36; 是从大到小的第&#36;i&#36;个量 则称这是一个OWA算子
能看出 OWA算子无关于&#36;a_i&#36;的具体值 只和他们的位置相关
定义 设&#36;\left[\left\langle v_1,a_1\right\rangle,\left\langle v_2,a_2\right\rangle,\cdots,\left\langle v_n,a_n\right\rangle\right]&#36; 是&#36;n&#36;个二维数组 则
&#36;&#36;\mathrm{IOWA}<em>W\left[\left\langle v_1,a_1\right\rangle,\left\langle v_2,a_2\right\rangle,\cdots,\left\langle v_n,a_n\right\rangle\right]=\sum</em>=nw i v-\mathrm{lindex}, &#36;
It's IWOA.&#36;v-index\left(i\right)&#36; It's based on&#36;v_i&#36; First in the order of size to small&#36;i&#36;Subscript</p>
<h4>Combination projection based on IOWA algorithms</h4>
<p>The weights of each individual projection method are not changed in the traditional combination prediction; but this projection is certainly different at different times; this is the flaw of the traditional weighted combination projection; but if the algorithms induce orderly weighted averages based on their predictions at all times, they can create new models that work better.
Yes.&#36;t&#36;Traditional weighted arithmetic average model of time
That's right.<em>t=\sum</em>{i=1}^ml_ix_{it},&#36;&#36;
令
&#36;&#36;\left.a_{it}=\left{\begin{array}{ll}1-\left|\dfrac{x_t-x_{it&#125;&#125;{x_t}\right|,&amp;\left|\dfrac{x_t-x_{it&#125;&#125;{x_t}\right|&lt;1,\\0,&amp;\left|\dfrac{x_t-x_{it&#125;&#125;{x_t}\right|\geqslant1,\end{array}\right.\right.&#36;&#36;
则 &#36;a_{it}&#36; 表示第&#36;i&#36;种预测方法在第&#36;t&#36;时刻的预测精度 我们可以把它看作预测值&#36;x_{it}&#36; 的诱导值 现在我们就可以使用IOWA模型了 如下
&#36;&#36;\text{ IOWA}<em>{L}\left[\left\langle a</em>{1t},x_{1t}\right\rangle,\left\langle a_{2t},x_{2t}\right\rangle,\cdots,\left\langle a_{mt},x_{mt}\right\rangle\right]=\sum_{i=1}^{m}l_{ii}x_{a-\mathrm{index}(it)},&#36;&#36;</p>
