---
title: 'Linear Time Series Analysis: Stationarity, ARMA, and ARIMA'
title_zh: 线性时间序列分析：平稳序列、ARMA与ARIMA
date: 2024-01-30 23:46:29 +0800
categories:
- Data Science
- Time Series & Spatial Data
tags:
- Statistics
- Time Series
author: Hyacehila
mathjax: true
hidden: true
excerpt: Covers stationarity, ARMA, ARIMA, model identification, parameter estimation, residual diagnostics, and regression
  with time series errors.
description: Covers stationarity, ARMA, ARIMA, model identification, parameter estimation, residual diagnostics, and regression
  with time series errors.
excerpt_zh: 整理平稳序列、ARMA、ARIMA、模型识别、参数估计、残差诊断和含时间序列误差的回归。
permalink: /blog/2024/01/30/linear-time-series-analysis-notes/
lang: en
translation_key: 2024-01-30-linear-time-series-analysis-notes
translation_status: machine
translation_source_hash: a07c6f7f8ae2167dc415b1d601b48e2e5fe51064a481596cd3ce4ec7e388e865
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<h2>Introduction and some basic concepts</h2>
<h3>Basic definitions</h3>
<p>Time series is a series of time points that can be arranged in an orderly fashion.</p>
<p>The data from observation of a series of time points is very common in research, so time series analysis is used in a wide variety of ways.</p>
<p>The research objectives of time series analysis are twofold.</p>
<ul>
<li>Study of the mechanisms for the generation of time series <strong>Get to know the past.</strong></li>
<li>Forecasting future possibilities based on historical data and other relevant factors <strong>Forecasting the future</strong></li>
</ul>
<p>In order to achieve time series analysis, we often need to require that time series exist in some built-in structure, and if he is completely random, we often lack the need to study time series.</p>
<h3>An example of a time series</h3>
<p>Here's a time series.
<img src="/assets/images/probability-statistics-notes/linear-time-series-analysis-notes-01.png" alt="Time series analysis">
It contains a time (which may be the year, the hour, etc.) as a cross-reference, an observation value as a vertical coordinate, and this is the most basic graphic of the time series.</p>
<p>Another common time series chart
<img src="/assets/images/probability-statistics-notes/linear-time-series-analysis-notes-02.png" alt="Time series analysis 1">
It uses the data of the previous year as cross-references, the data of the current year as vertical coordinates, which allows us to study whether there is some influence between the two-year observations.</p>
<p><strong>Time series analysis requires more of our research maps than other areas, and we need to map them against the research targets, and the information in the analysis maps is important to develop an understanding of the charts, and we often choose the appropriate time series analysis model based on the information that the maps react to.</strong></p>
<h3>Linear regression and time series analysis</h3>
<p>The time series cannot be equated to regression analysis. </p>
<p>The characteristics of the time series data are:</p>
<ul>
<li>Self-relevance (linear or non-linear)</li>
<li>Non-exchangeability (samples sequence is not interchangeable)</li>
</ul>
<p>The simplest data structure in the application scene of regression analysis is interchangeable.<strong>Data for independent and distributed</strong>When time series data meet some conditions, you can use regression analysis to handle it.&#36;AR&#36;But we can't think of them as identical.</p>
<p>There are a lot of unique ways to do time series analysis, and they have no connection with regression analysis, so...<strong>The time series cannot be equated to regression analysis.</strong></p>
<h3>Random process and time series analysis</h3>
<p>Time series is a set of observations from random processes: We call it the time series process of the random process.</p>
<p>The time series process is a special random process.</p>
<h3>Digital features of time series</h3>
<p>More commonly used as follows:
Mean function defined as
&#36;&#36;\mu_{t}=E(Yt)&#36;&#36;
Self-conciliation difference function defined as
&#36;&#36;Cov(X_s,X_t)=\operatorname{E}\left[(X_s-m_X(s))(X_t-m_X(t))\right]&#36;&#36;
The function is defined as
&#36;&#36;rt s, =mathrm{Corr} (Y t}, Y Y mathrm{Cov} (Y t}, Y s} }{\sqrt{\mathrm{Var} (Y t}\mathrm{Var} (Y})<em>{s})&#125;&#125;=\frac{\gamma</em>{t,s&#125;&#125;{\sqrt{\gamma_{t,t}\gamma_{s,s&#125;&#125;}&#36;&#36;
他们有一些基础的性质为
&#36;&#36;\begin{aligned}\gamma_{\iota.\iota}&amp;=\operatorname{Var}(Y_{\iota})&amp;\rho_{\iota.\iota}&amp;=1\\gamma_{t.s}&amp;=\gamma_{s.t}&amp;\rho_{t,s}&amp;=\rho_{s,t}\\mid\gamma_{\iota,s}|&amp;\leqslant\sqrt{\gamma_{\iota.t}\gamma_{s,s&#125;&#125;&amp;\mid\rho_{t.s}\mid\leqslant1\end{aligned}&#36;&#36;
我们这里给出一个有用的定理
&#36;&#36;\mathrm{Cov}\bigg[\sum_{i=1}^{m}c_{i}Y_{t_{i&#125;&#125;,\sum_{j=1}^{n}d_{j}Y_{s_{j&#125;&#125;\bigg]=\sum_{i=1}^{m}\sum_{j=1}^{n}c_{i}d_{j}\mathrm{Cov}(Y_{t_{i&#125;&#125;,Y_{s_{j&#125;&#125;)&#36;&#36;</p>
<p>Let's just go over the concept of random process smoothness here.
<a href="/en/blog/2023/03/18/stochastic-process-basics-notes/">Random Process Basis</a> “Definition of smooth process” section</p>
<h3>Disaggregation of time series</h3>
<p>The time series needs an internal structure to be analysed.
&#36;&#36;X_t=T_t+S_t+R_t,t=1,2,\ldots &#36;&#36;</p>
<ul>
<li>Trends</li>
<li>Seasonal item</li>
<li>Random Item
We have a lot of natural ideas about how to estimate these items, like using retrogressive convergence trends, and the rest of the items using quarterly average catch seasons, and so on, we'll have a detailed study later on.</li>
</ul>
<p>Trends and seasons can be treated as non-random time series, and their prediction problems are often simple. Random items are usually smooth sequences.</p>
<h2>Examples of common time series</h2>
<h3>I'm just gonna swim around.</h3>
<p>You!&#36;e_1,e_2,...&#36;The difference is&#36;\sigma^2&#36;I'm sorry. I'm looking at the time series of independent and distributed random variables. &#36;&#123;Y_t:t=&#36; &#36;1,:2,:...}&#36; Construct as follows:
&#36;&#36;\left.\left.\begin array}Y 1&amp;=e_1\Y_2&amp;=e_1+e_2\&amp;\vdots\Y_t&amp;=e 1+e 2+\cdots+e t\end{array}\right.\right}&#36;&#36;
We can easily calculate it.
Mean Functions&#36;\mu_{t}=0&#36;
Difference Functions &#36;\mathrm{Var}(Y_{\imath})=t\sigma_{e}^{2}&#36;
Custom Related Functions &#36;\rho_{t,s}=\frac{\gamma_{t,s&#125;&#125;{\sqrt{\gamma_{t,t}\gamma_{s,s&#125;&#125;}={\sqrt{\frac{t}{s&#125;&#125;}&#36;</p>
<p>And we can give you some understanding of randomly moving processes.</p>
<p>Over time, in the next few minutes. Go, go, go!&#36;Y&#36;The value is becoming more relevant, and on the other hand, the time is far away.&#36;Y&#36;Values, which are becoming less relevant, and ac-frograms of the unit root process, which are slow and slow to decline. </p>
<p>Although the theoretical average is zero, the difference increases over time, so the process is expected to swing away from zero, which also shows that the model is unpredictable.</p>
<h3>Slide Average</h3>
<p>Or the assumptions that lie ahead?
&#36;&#36;Y_t=\frac{e_t+e_{t-1&#125;&#125;2&#36;&#36;
Mean Functions&#36;\mu_{t}=0&#36;
Difference Functions &#36;\mathrm{Var}(Y_{\imath})=0.5\sigma_{e}^{2}&#36;
Custom Related Functions
&#36;US&#36;US&#36;US&#36;&#36;US&#36;&#36;US&#36;&#36;US&#36;US&#36;US&#36;&#36;US&#36;US&#36;&#36;US&#36;US&#36;&#36;US&#36;&#36;US&#36;US&#36;US&#36;&#36;US&#36;US&#36;US&#36;US&#36;US&#36;&#36;US&#36;US&#36;US&#36;&#36;US&#36;US&#36;US&#36;US&#36;US&#36;&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;&amp;\mid t-s\mid=0\0.5&amp;\mid t-s\mid=1\0&amp;\mid t-s\mid&gt;You're not gonna get away with this?
The average process is a smooth process that is usually used as an example of introduction.</p>
<h3>White Noise</h3>
<p>An important example of this smoothing is the so-called white noise process, defined as the series of random variables in the same distribution.&#36;\langle e_i\rangle&#36;It's not because it's interesting, but because many useful processes can be constructed by white noise processes.</p>
<p>For white noise processes, there are &#36;E[e_t]=\mu&#36;   &#36;cov(e_t,e_s)=\begin{cases}\sigma^2&amp;t=s\0&amp;I'm sorry, I'm sorry, but I'm sorry, but I'm sorry, but I'm sorry, but I'm sorry, but I'm sorry.
If random variables are independent of each other, it's called independent white noise.
If the average is 0, then it's called the zero-average independent white noise, and then if the difference is 1 it's called the standard independent white noise. </p>
<p>We usually study the most general standard of independent white noise.
We can see that randomly moving and sliding are constructed on average according to white noise processes. </p>
<h3>Random Cosine Wave</h3>
<p>&#36;&#36;Y_{t}=\cos\left[2\pi(\frac{t}{12}+\Phi)\right]&#36;&#36;
of which &#36;\Phi\sim U(0,1)&#36;
And we can see that this random process has a strong degree of certainty, and it's cyclical, and his only randomity is how we pick and choose our first date.
We can figure it out.
Mean Functions &#36;\mu_{t}=0&#36;
Custom Related Functions  &#36;\rho_k=\cos(2\pi\frac k{12})&#36;
So, we can judge it as a smooth time series.
The simulator of the average and random cosine wave of observation shows that it is unrealistic to judge the stability of the time series by relying on our observation time series alone, and we need to find other ways to process it later.</p>
<h3>Stable differences</h3>
<p>We know that random migration sequences are not stable.
But we're not alone. &#36;Y_t&#36;It's about his differences, which is...
&#36;&#36;Z_t=Y_t-Y_{t-1}&#36;&#36;
It's easy to see that the sequence after the differential is flat.
So we can use the simple technique of differentials to get statistically stable sequences that were not stable.</p>
<h2>Trends</h2>
<p>This is some of the explanations for the section on the breakdown of time series, which explains the meaning of the trend item.</p>
<h3>Trends of certainty and randomity</h3>
<p>Trends are the product of observations of reality, but we know that time series are random and that the trends we see are not necessarily the true characteristics of the sequence, so we need to model them over time and find real information from multiple surface trends.</p>
<ul>
<li>Random trends: multiple simulations show completely different trends in time series</li>
<li>Trends of certainty: trends in multiple simulations show near time series</li>
</ul>
<p>It's very obvious:<strong>If we only have one chance to observe, there is no way to judge whether the trend is random or definitive, and we will determine sexual trends in the later studies.</strong></p>
<h3>Constant mean</h3>
<p>As a simple case, it's also called a smooth time series, so we assume that the average function is constant.
&#36;&#36;Y_t=\mu+X_t&#36;&#36;
Random disturbance of which&#36;X_t&#36;Yes.&#36;EX=0&#36;
We can estimate the average value below without adding any other assumptions.
&#36;&#36;\overline{Y}=\mu&#36;&#36;
To study the accuracy of this estimate, we need to be right.&#36;X_t&#36;And there's not much to be done here.</p>
<h3>Very equal</h3>
<p>We need to look at the uneven sequences and make a series of assumptions about changes in the averages to be specific to the analysis.</p>
<h4>Linear trends</h4>
<p>Consider
&#36;&#36;\mu_t=\beta_0+\beta_1t&#36;&#36;
To estimate this situation, we need the minimum two-fold method most commonly used in regression analysis.</p>
<h4>Seasonal trends</h4>
<p>It's our basic model.
&#36;&#36;Y_t=\mu_t+X_t&#36;&#36;
The idea of a more common seasonal model is to follow the monthly pattern.
&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;USE&amp;t=1,13,25,\cdots\\beta_2&amp;t=2,14,26,\cdots\\vdots\\beta_{12}&amp;t= 12, 24, 36, \cdots\end{cases}
The estimates of this trend are based on experience and some of the statistical methods that we have not been able to address, and the use of statistical software is needed, and we know that it is enough to analyse them.</p>
<h4>Cosine trend</h4>
<p>The seasonal average model contains many separate parameters, but it does not take into account the shape of seasonal trends, and only a few time series models are not relevant when they are close, so we introduced cosine trend models, which are very common when they exist.
Consider the average model below
&#36;&#36;\mu_{t}=\beta\mathrm{cos}(2\pi ft+\Phi)&#36;&#36;</p>
<p>We can deform it below, and we can use regression to deal with it.
&#36;&#36;\mu_{t}=\beta_{0}+\beta_{1}\cos(2\pi ft)+\beta_{2}\sin(2\pi ft)&#36;&#36;</p>
<h2>Steady Random Time Series ARMA</h2>
<p>We're talking about the basic concept of a large class of parameter time series models, which are retrogressive average models (ARMAs), which play an important role in modelling real processes, and their core characteristics are:<strong>Steady.</strong></p>
<h3>General linear process</h3>
<p>I think...&#36;Y_t&#36;It's the time series we've been observing, and we think...&#36;e_t&#36;is a white noise sequence that is not observed;
<strong>You can set the mean of this white noise sequence to zero, which means we erase the average information, and in a real model, we can be considered to have lost the average.</strong>
So the general linear process can be seen as a weighted linear combination of past and present white noise.
&#36;&#36;Y_{t}=e_{t}+\psi_{1}e_{t-1}+\psi_{2}e_{t-2}+\cdots &#36;&#36;
On the right is an infinite number, and we will be able to limit it more to the goal we want to study.</p>
<h3>Slide Average Process (MA)</h3>
<p>One very natural idea is that the impact of the very distant noise can be ignored, so we can transform the general linear process into a process that is very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very,
&#36;&#36;Y_t=e_t-\theta_1e_{t+1}-\theta_2e_{t-2}-\cdots-\theta_qe_{t-q}&#36;&#36;
<strong>The coefficient is random, the positive.</strong>
We call it the equation.&#36;q&#36;Slipping average process as&#36;MA(q)&#36;
The slide average is the average based on the weight, then the average is re-enacted at a time, and so on.</p>
<h4>MA（1）</h4>
<p>Model as
&#36;&#36;Y_{t}=e_{t}-\theta e_{t-1}.&#36;&#36;
It's easy to calculate.
&#36;&#36;E(Y_{t})=0&#36;&#36;
&#36;&#36;\mathrm{Cov}(Y_t,Y_{t-1})=\mathrm{Cov}(e_t-\theta e_{t-1},e_{t-1}-\theta e_{t-2})=\mathrm{Cov}(-\theta e_{t-1},e_{t-1})=-\theta \sigma_t^2&#36;&#36;
&#36;&#36;\mathrm{Cov}(Y_t,Y_{t-2})=\mathrm{Cov}(e_t-\theta e_{t-1},e_{t-2}-\theta e_{t-3})=0&#36;&#36;
It can be seen that there is no self-relevance after the process is older than level one.
We can base our actions on specifics.&#36;\theta&#36;The value is taken to analyse the magnitude of the coefficient, and it can also be analysed for relevance, as we have shown at the beginning of the section on "Sequence-Sequence Examples"</p>
<h4>MA（2）</h4>
<p>Model as
&#36;&#36;Y_{t}=e_{t}-\theta_{1}e_{t-1}-\theta_{2}e_{t-2}&#36;&#36;
Compute the synergetic difference function has
&#36;&#36;\gamma_0=\mathrm{Var}(Y_t)=\mathrm{Var}(e_t-\theta_1e_{t-1}-\theta_2e_{t-2})=(1+\theta_1^2+\theta_2^2)\sigma_{t}^{2}&#36;&#36;
&#36;&#36;\begin{aligned}
\gamma_{1}&amp; =\mathrm{Cov}(Y_{i},Y_{i-1})=\mathrm{Cov}(e_{i}-\theta_{1}e_{i-1}-\theta_{2}e_{i-2},e_{i-1}-\theta_{1}e_{i-2}-\theta_{2}e_{i-3})  \
&amp;=\mathrm{Cov}(-\theta_1e_{t-1},e_{t-1})+\mathrm{Cov}(-\theta_1e_{t-2},-\theta_2e_{t-2}) \
&amp;=[-\theta_1+(-\theta_1)(-\theta_2)]\sigma_e^2=(-\theta_1+\theta_1\theta_2)\sigma_e^2
\end{aligned}&#36;&#36;
&#36;&#36;\begin{gathered}
\gamma 2}
== sync, corrected by elderman ==
I'm sorry, I'm sorry.
That's right.&#36;MA(2)&#36; Models larger than second tier lags are not self-relevant</p>
<h4>MA（q）</h4>
<p>We'll give you a direct conclusion on the function.
&#36;US&#36;\rho k=begin{cases}\fra{k+theta k+theta \theta \ \q^}&amp;\quad k=1,2,\cdots,q\0&amp;\quad k&gt;q\end{cases}
The conclusion is clear.
&#36;MA(q)&#36;Process is greater than&#36;q&#36;There's no correlation when you're lagging behind.
We've done enough research on the average slide process. We'll go on to another important model.</p>
<h3>Self-Return Process (AR)</h3>
<p>By definition, self-return refers to self-returning as a regression variable. This is in the form of:&#36;p&#36;Step-by-step process meets equations
&#36;&#36;Y_t=\phi_1Y_{t-1}+\phi_2Y_{t-2}+\cdotp\cdotp\cdotp+\phi_pY_{t-p}+e_t&#36;&#36;
That's right.&#36;p&#36;A lag item and new message entry&#36;e_t&#36; </p>
<h4>AR（1）</h4>
<p>Or do you start with a simple form?
&#36;&#36;Y_{t}=\phi Y_{t-1}+e_{t}&#36;&#36;
<strong>We'll assume the average is zero, and it'll be on a smooth condition.</strong>
Let's calculate.
&#36;&#36;\gamma_0=\frac{\sigma_e^2}{1-\phi^2}&#36;&#36;
&#36;&#36;\gamma_k=\phi^k\frac{\sigma_e^2}{1-\phi^2}&#36;&#36;
&#36;&#36;\rho_k=\frac{\gamma_k}{\gamma_0}=\phi^k&#36;&#36;
Because the difference must be right, &#36;1&lt;\phi&lt;&#36;1.
So we know that the function of the function in question is characterized by an index decrease, and that, depending on the positive and negative, it's a serene type.
We might as well take it.&#36;\phi=0.9&#36; It can be seen that even a third-level lag has a strong self-relevance.
We can easily prove it.&lt;\phi&lt;1&#36;  的时候 &#36;AR(1) process is smooth.</p>
<h4>AR（2）</h4>
<p>Model in the form of
&#36;&#36;Y_t=\phi_1Y_{t-1}+\phi_2Y_{t-2}+e_t&#36;&#36;
To make the model stable, we need to meet the conditions.
&#36; \phi 1+\phi 2&lt;1,\quad\phi_2-\phi_1&lt;1\text{,}\quad|\phi_2|&lt;1&#36;&#36;
为了研究自相关函数 我们沿用上面的思路得到下面的递推方程 Yule-Walker 方程
&#36;&#36;\rho_{k}=\phi_{1}\rho_{k-1}+\phi_{2}\rho_{k-2},\quad k=1,2,3,\cdotp\cdotp\cdotp &#36;&#36;
其中为了进行递推有
&#36;&#36;\rho_1=\frac{\phi_1}{1-\phi_2}&#36;&#36;
&#36;&#36;\rho 2 = \ \ \ \ \ \ \ \ \ \ \ \ \ \ } } } } } } } } } } } } } } } } = = = = = &#36; &#36; &#36; &#36; &#36; &#36; &#36; &#36; \ &#36; &#36; &#36; &#36; &#36; &#36; \ \ &#36; &#36; &#36; &#36; \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ &#36; &#36; &#36; \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \
We study the incremental formula of the function to analyse the nature of the function, and positive or negative changes are possible as the lag step increases from the index of the relevant coefficient.
This decline could be index decline or block the nymphosphate.</p>
<h4>AR（p）</h4>
<p>Consider the model as
&#36;&#36;Y_t=\phi_1Y_{t-1}+\phi_2Y_{t-2}+\cdotp\cdotp\cdotp+\phi_pY_{t-p}+e_t&#36;&#36;
And give the necessary conditions for stability:
&#36;&#36;\begin{aligned}\phi 1+\phi 2+\cdots+\pi p&lt;1\|\phi_p|&lt;1\end{aligned}&#36;&#36;
给出Yule-Walker方程为
&#36;&#36;\begin{aligned}\rho_1&amp;=\phi_1+\phi_2\rho_1+\phi_3\rho_2+\cdots+\phi_p\rho_{p-1}\\rho_2&amp;=\phi_1\rho_1+\phi_2+\phi_3\rho_1+\cdots+\phi_{p,p-2}\&amp;\vdots\\rho_p&amp;=cdots+\ft p\d{aligned} I'm sorry.
If we give a specific coefficient, we can solve the coefficients by the equation.
The nature of the relevant coefficients is:<strong>The coefficient is a linear combination of some resistance to nectar and some resistance to swirl fluctuations.</strong></p>
<h3>Average Slipper Process (ARMA) for self-return</h3>
<p>If one of the models is self-regressive, the other is sliding average, there's a more general model form (or flat)
&#36;&#36;Y_t=\phi_1Y_{t-1}+\phi_2Y_{t-2}+\cdots+\phi_pY_{t-p}+e_t-\theta_1e_{t-1}-\theta_2e_{t-2}-\cdots-\theta_qe_{t-q}&#36;&#36;
We call it&#36;ARMA(p,q)&#36;
We'll just introduce a simple form.</p>
<h4>ARMA（1）</h4>
<p>Model defined
&#36;&#36;Y_t=\phi Y_{t-1}+e_t-\theta e_{t-1}&#36;&#36;
Because of the presence&#36;AR&#36;The ingredients, we need to consider the condition of stability.
&#36;&#36;\mid\pi\mid&lt;1&#36;&#36;
通过一系列运算有 自相关函数为
&#36;&#36;\rho k=\frac(1-\theta\i) (\theta)\1-2\theta\pi+\theta^)\rho=,\fad k\geqslant&#36;1
He's also a form of index decline.&#36;p_0&#36; (Dependantly)&#36;\theta&#36;) This is the one that makes it&#36;AR(1),MA(1)&#36; It's different.
One of them is a step behind.&#36;\theta&#36; The other one is index decline but starting with 1.</p>
<p>For the general ARMA model, we're meeting the level of stability.
&#36;&#36;\text{当月仅当 AR 特征方程 }\phi(x)=0\text{ 的根的模大于 }1&#36;&#36;
Satisfactory to the relevant function at this time
&#36;rho k=k 1}k=k\k\k\&gt;q&#36;
A form similar to the Yule-Walker equation when &#36;k&lt;q&#36;的时候 自相关函数会含有&#36;\theta&#36; ingredients</p>
<h3>Reversibility</h3>
<p>We can actually see it in front of us.&#36;MA&#36;Models. We use them.&#36;MA(1)&#36; For example,
&#36;&#36;\mathrm{Cov}(Y_t,Y_{t-1})=\mathrm{Cov}(e_t-\theta e_{t-1},e_{t-1}-\theta e_{t-2})=\mathrm{Cov}(-\theta e_{t-1},e_{t-1})=-\theta \sigma_t^2&#36;&#36;
And then there was...
&#36;&#36;\rho_{1}=(-\theta)/(1+\theta^{2})&#36;&#36;
It's easy to find.&#36;\theta,\frac{1}{\theta}&#36; The relevant coefficients are the same.</p>
<p>And even if we take the value of a known correlation coefficient, the coefficients that we get are not the only ones that are the same.</p>
<p>We know that the AR process can be described as a general linear process. What about the MA process?
We'll think about it.
&#36;&#36;Y_t=e_t-\theta e_{t-1}&#36;&#36;
Convert
&#36;&#36;e_{t}=Y_{t}+\theta e_{t-1}&#36;&#36;
And then they keep changing.&#36;e_{t-1}&#36; Yes.
&#36;&#36;e_t=Y_t+\theta Y_{t-1}+\theta^2Y_{t-2}+\cdotp\cdotp\cdotp &#36;&#36;
So there is.
&#36;&#36;Y_t=(-\theta Y_{t-1}-\theta^2Y_{t-2}-\theta^3Y_{t-3}-\cdots)+e_t&#36;&#36;
Which means if we have a t-shirt,&lt;1&#36; 则&#36;MA(1)&#36; 可以转化为一个自回归模型 此时我们称其为可逆的&#36;MA&#36; Model
For the general&#36;MA,ARMA&#36; Models, they can reverse if the root of the characteristic equation is more than one.
We can easily prove:<strong>For reversible MA processes, a single set of parameters can be obtained in the case of a given function</strong>
<strong>We're working on it.&#36;ARMA&#36;Models are both smooth and reversible.</strong></p>
<h2>Unstable Random Time Series ARIMA</h2>
<p>Not all event sequence models are stable; in fact, most time series models in the real world are non-stable, and the forced use of the method modelling in the "Stable and Random Time Series ARMA" section of this paper can only produce absurd conclusions.</p>
<p>Fortunately, we need only a simple way to study the problem of instability.</p>
<p>AR model involves smoothing (factors)
MA model involves reversible issues (or is it related to coefficients)
And then the ARMA model, the ARIMA model, will study them at the same time.</p>
<h3>Stable differences</h3>
<p>Let's think about one.&#36;AR(1)&#36; Model
&#36;&#36;Y_t=3Y_{t-1}+e_t&#36;&#36;
This doesn't fit the description.&#36;AR(1)&#36; The conditions for stabilization studied by the model
Actually...
&#36;&#36;\mathrm{Var}(Y_{t})=\frac{1}{8}(9^{t}-1)\sigma_{e}^{2}&#36;&#36;
The difference has an exponential explosion.
&#36;&#36;\mathrm{Corr}(Y_{t},Y_{t-k})=3^{k}\sqrt{\frac{9^{t-k}-1}{9^{t}-1&#125;&#125;\approx1.&#36;&#36;
For the larger&#36;t&#36; And medium strength.&#36;k&#36; The index explosion is the cause of this correlation.
And that's that, in any case, this time series will be exponential.</p>
<p>We'll think about it.
&#36;&#36;Y_t=Y_{t-1}+e_t&#36;&#36;
Compute its first-class differential.
&#36;&#36;\nabla Y_t=e_t&#36;&#36;
It's easy to see that the difference can stabilize the original unstable model.
<strong>The phenomena described above are widespread, the differentials are conducive to smoothing the unstable model, and if the first step is not enough, then the difference is empirically enough.</strong></p>
<h3>ARIMA Model</h3>
<p>If a time series model&#36;&#123;Y_t}&#36; Yes.&#36;d&#36;The sub-division is a flat ARMA model, which is called an ARIMA model if the differential obeys&#36;ARMA(p,d)&#36; Name &#36;Y_t&#36; Obey.&#36;ARIMA(p,d,q)&#36;
Think about one next.&#36;ARIMA(p,1,q)&#36; You!&#36;W_t=Y_t-Y_{t-1}&#36;
&#36;&#36;W_t=\phi_1W_{t-1}+\phi_2W_{t-2}+\cdots+\phi_pW_{t-p}+e_t-\theta_1e_{t-1}-\theta_2e_{t-2}-\cdots-\theta_qe_{t-q}&#36;&#36;
Or as an expression
&#36;&#36;00\&amp;\phi_1(Y_{t-1}-Y_{t-2})+\phi_2(Y_{t-2}-Y_{t-3})+\cdots+\phi_p(Y_{t-p}-Y_{t-p-1})\&amp;+e_t-\theta_1e_{t-1}-\theta_2e_{t-2}-\cdots-\theta_qe_{t-q}\end{aligned}&#36;&#36;
可以改写为
&#36;&#36;\begin{aligned}Y_t&amp;=(1+\phi_1)Y_{t-1}+(\phi_2-\phi_1)Y_{t-2}+(\phi_3-\phi_2)Y_{t-3}+\cdots\&amp;+ (\p-\ph) Y t-p}- \\ph p t-p e t-\theta 1e t t \theta 2 t}-\cdots-\theta qe  {t-q} I'm sorry.
To understand these three forms
The first one is one.&#36;ARMA(p,q)&#36; The second is just a change of name, and the third is a change of name. &#36;ARMA(p+1,q)&#36; </p>
<h3>IMA(1,1)</h3>
<p>We're still thinking about the simplest form.
&#36;&#36;Y_{t}=Y_{t-1}+e_{t}-\theta e_{t-1}&#36;&#36;
The differences and related factors in calculating the model are:
&#36;&#36;\left.\mathrm{Var}(Y_{i})=\left[\begin{matrix}{1+\theta^{2}+(1-\theta)^{2}(t+m)}\\end{matrix}\right.\right]\sigma_{\epsilon}^{2}&#36;&#36;
Showing explosion.
&#36;&#36;00\
\\mathrm{Corr} (Y ({\ota}, Y t-k}&amp; =\frac{1-\theta+\theta^2+(1-\theta)^2(t+m-k)}{[\operatorname{Var}(Y_t)\operatorname{Var}(Y_{t-k})]^{1/2&#125;&#125;\approx\sqrt{\frac{t+m-k}{t+m&#125;&#125;  \
&amp;\\approx1
I'm sorry, I'm sorry.
The reason for the explosion is the strong correlation.</p>
<h3>IMA(2,2)</h3>
<p>Model as
&#36;&#36;\nabla^2Y_t=e_t-\theta_1e_{t-1}-\theta_2e_{t-2}&#36;&#36;
We do not calculate, answer directly, and the difference and the related coefficients are presented in the same way as in this paper, "IMA (1, 1)." Section</p>
<h3>Constants in ARIMA</h3>
<p>In the section on "Arma for a smooth, random time series" here, constants are not affecting our research, less averages, and zero values are studied, and all results plus averages are sufficient.
But in ARIMA, it's not that simple.
It's easy to introduce the model into the differential model, as follows:
&#36;&#36;00\
W t}-mu=&amp; \phi_{1}(W_{t-1}-\mu)+\phi_{2}(W_{t-2}-\mu)+\cdots+\phi_{p}(W_{t-p}-\mu)  \
&amp;+e_{t}-\theta_{1}e_{t-1}-\theta_{2}e_{t-2}-\cdots-\theta_{q}e_{t-q}
\end{aligned}&#36;&#36;
或者
&#36;&#36;W_t=\theta_0+\phi_1W_{t-1}+\phi_2W_{t-2}+\cdots+\phi_pW_{t-p}+e_t-\theta_1e_{t-1}-\theta_2e_{t-2}-\cdots-\theta_qe_{t-q}&#36;&#36;
其中
&#36;&#36;Other Organiser
They're actually equal. </p>
<p>We need to consider the impact of this non-0-average on the original ARIA model.
IMA (1,1).&#36;Y_t&#36;Yes.
&#36; \begin{aligned}Y t=&amp;e_t+(1-\theta)e_{t-1}+(1-\theta)e_{t-2}+\cdots+(1-\theta)e_{-m}-\theta e_{-m-1}\&amp;+ (t+m+1)\theta
The fact that the IMA is not a zero average has led to a linear trend over time.
Yes.&#36;d=2&#36; Time Non-0-average leads to a double time trend item</p>
<h3>Data transformation in time series analysis</h3>
<p>In many realities, we're going to show a percentage increase, especially in economic and biological data, which is actually an exponential growth in time series.</p>
<p>We're doing a logarithmic variant that works very well at this point, and a very important shift in numbers. Group</p>
<p><strong>BoxCox certainly is not just for regression analysis and variance analysis, but there are many things that help us to choose specifics.&#36;\lambda&#36;The way they do it, they usually study normality and variance.</strong></p>
<p>Data conversion is also a very common operation in data analysis, not binding on TSA, and when we choose to change and not study it, as a mere reminder:<strong>The time series of data changes are also a very useful technique.</strong></p>
<h2>Model recognition</h2>
<p>We have developed a large array of time series models, ARIMA; now we have to learn to do statistical extrapolation from the research we have done before, and our work is divided into four parts.</p>
<ul>
<li>Select the appropriate time series data for the given time series&#36;p,d,q&#36;Value</li>
<li>Parameters for estimating the selected model</li>
<li>Test the proposed model and make improvements</li>
<li>Forecast future data using previously identified models
So that's model recognition, parameter estimates, model tests, model predictions, four parts, and we'll be talking about these in four chapters, including this chapter.</li>
</ul>
<h3>Identification of MA models</h3>
<p>We can estimate that the sample has its own function.
&#36;&#36;r_{k}=\frac{\sum_{t=k+1}^{n}\left(Y_{t}-\overline{Y}\right)\left(Y_{t-k}-\overline{Y}\right)}{\sum_{t=1}^{n}\left(Y_{t}-\overline{Y}\right)^{2&#125;&#125;,\quad k=1,2\cdots &#36;&#36;
Our purpose is to identify the samples from their own relevant functions. Out&#36;r_k&#36;The pattern of the ARIMA model, which is already known, is the appropriate model to be selected.&#36;p,d,q&#36;</p>
<p>Yes.&#36;MA(q)&#36; Model when the lag step exceeds&#36;q&#36;. This means that the sample is already the relevant function.&#36;MA&#36;A good indicator of the process.</p>
<p>Use sample ACF to identify</p>
<h3>Identification of AR Models</h3>
<p>But, yes,&#36;AR(p)&#36;Models are becoming progressively decaying from the function in question.&#36;AR(p)&#36; Model
We introduce. &#36;k&#36;The relevant function of the step lag is
&#36;&#36;\begin{aligned}\phi_{kk}=\mathrm{Corr}(Y_t-\beta_1Y_{t-1}-\beta_2Y_{t-2}-\cdots-\beta_{k-1}Y_{t-k+1},\Y_{t-k}-\beta_1Y_{t-k+1}-\beta_2Y_{t-k+2}-\cdots-\beta_{k-1}Y_{t-1})\end{aligned}&#36;&#36;
The following is the calculation of the selected function of the sample.
&#36;&#36;\phi_{kk}=\frac{\rho_k-\sum_{j=1}^{k-1}\phi_{k-1,j}\rho_{k-j&#125;&#125;{1-\sum_{j=1}^{k-1}\phi_{k-1,j}\rho_j}.&#36;&#36;
What are the characteristics of the function?
Yes.&#36;AR(p)&#36;Model
&#36;US&#36; k=,\dk=&gt;P&#36;
That's right. &#36;AR(p)&#36; Models are given preference from the good indicator of the function
<em>&#36;MA(q)&#36;The model has no indication of preference for the function. He'll index decay instead of zero.</em></p>
<p>Use sample PACF to identify</p>
<h3>Identification of ARMA models</h3>
<p><strong>ARMA(p,q) does not have a tailing nature for either indicator function</strong> We need to come up with new ways to achieve this, and there is more than one way we've put forward to solve the problem of the ARAMA model, and we've introduced the EACF method, which is currently being simulated as being of a more positive nature.</p>
<p>The core idea of the EACF law is:<strong>If the AR of a model is known, the observation sequence filtering out of the regression section will get a pure MA model that can be used to determine the order of magnitudes by ACF.</strong></p>
<p>Consider&#36;ARMA(1,1)&#36;To describe the use of the EACF.
&#36;&#36;Y_t=\phi Y_{t-1}+e_t-\theta e_{t-1}&#36;&#36;
In this case, apply&#36;Y_t&#36;Yeah.&#36;Y_{t-1}&#36; Simple linear regression is possible.&#36;\phi&#36;Unsatisfactory estimates (systematic deviations because the quantity of regression factor estimates by nature contains&#36;\theta&#36;But this re-entry is a real disability that can help us analyze.
Second Re-entry&#36;Y_t&#36; The coefficient of return for the first step of the first degree of disability&#36;\tilde{\phi}&#36; Yeah.&#36;\phi&#36;The same estimate, that is,
&#36;&#36;W_{t}=Y_{t}-\tilde{\phi}Y_{t-1}&#36;&#36;It's one.&#36;MA(1)&#36;Process</p>
<p><strong>For ARMA models, it's a good way to think about EACF and the corresponding zero-point point.</strong>
<img src="/assets/images/probability-statistics-notes/linear-time-series-analysis-notes-03.png" alt="Time series analysis 4">
Consider the zero triangle.&#36;MA(1),AR(1or2)&#36; It's all acceptable.</p>
<h3>ARIMA Model recognition</h3>
<h4>The idea of recognition of models</h4>
<p>We present here in the section “ARIMA of the non-stable random time series” the unstableness that can be explained by the ARIMA model;
All ACF calculations using unstable sequences do not mean anything (ACF calculations are themselves assumed to be smooth)</p>
<p>But we have found nature in a lot of research:<strong>The ACF of the non-stable sequence shows a slow downward trend in non-indexed decline</strong> </p>
<p>This is not even a ma ARMA model, which can be considered a good way to judge the Alima model.</p>
<p>After judging the ARIMA model, the difference is our only choice.</p>
<h4>About the excess differential</h4>
<p>We've been talking about the ARIMA section of the non-stable random time series, and the difference in the smooth sequence is still flat; but the excess is not desirable.</p>
<p>When we do the difference, we need to estimate it once more.&#36;\theta&#36;Values, he's also seriously affecting the parameters.</p>
<p>Models should be built as concisely as possible, and excessive differentials are not desirable, but we must also be decisive in the absence of a clear margin.</p>
<h4>Other model identification methods</h4>
<p>We have a lot of pure-value-based methods.
Dickey - Fuller Root Check
Select the best model using information volume guidelines such as AIC BIC</p>
<h2>Parameter estimation</h2>
<p>We've solved the problem of modeling.&#36;p,d,q&#36; Now we've identified the parameters in the model, and we've only been able to consider the ARMA model, and then the ARMA model, and then the ARMA model, which is not the average, is reduced to their average.</p>
<h3>Rectangular estimate</h3>
<p>The rectangular estimate is not considered the most efficient method, but it's the simplest way, or the sampler is the theoretical rectangular method, and it's made up of equations that can be estimated by any unknown parameter, the most classic example of which is the estimate of the overall average by the sample average.</p>
<h4>Self-Regression Model AR</h4>
<p>Yes.&#36;AR(1)&#36;We have models.
&#36;&#36;\rho_{1}=\phi.&#36;&#36;
So we calculate the sample from its own function.&#36;r_k&#36;Yes. &#36;r_1=\phi&#36;
Yes.&#36;AR(2)&#36;Model
&#36;&#36;r_1=\phi_1+r_1\phi_2,\quad r_2=r_1\phi_1+\phi_2&#36;&#36;
Higher&#36;AR&#36;The model is the same thing.</p>
<h4>Slide Average Model MA</h4>
<p>For the slide average model, rectangular estimation is not very good.&#36;MA(1)&#36;Yes.
&#36;&#36;\rho_1=-\frac\theta{1+\theta^2}&#36;&#36;
We'll take over.&#36;r_1&#36; We're dealing with a double equation.
&#36;&#36;\hat{\theta}=\frac{-1+\sqrt{1-4r_1^2&#125;&#125;{2r_1}&#36;&#36;
This is at &#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;&#36;US&#36;&#36;US&#36;US&#36;US&#36;&#36;US&#36;US&#36;&#36;&#36;&#36;&#36;&#36;&#36;US&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&lt;If you can solve it with 0.5 dollars, otherwise there's no real solution to this equation;
For higher-grade MMA models, it's rapidly becoming more complex.</p>
<h4>ARMA</h4>
<p>We'll think about it.&#36;ARMA(1,1)&#36;Situation
&#36;&#36;\hat{\phi}=\frac{r_{2&#125;&#125;{r_{1&#125;&#125;&#36;&#36;
&#36;&#36;r_1=\frac{(1-\theta\hat{\phi})(\hat{\phi}-\theta)}{1-2\theta\hat{\phi}+\theta^2}&#36;&#36;
We still need to deal with a secondary equation that is more problematic when it exists.</p>
<h4>Noise difference</h4>
<p>The last amount we need to estimate is the noise. Bad&#36;\sigma_e^2&#36; First we know we can estimate the sequences with sample differences. Bad
&#36;&#36;s^{2}=\frac{1}{n-1}\sum_{\iota=1}^{n}{(Y_{t}-\overline{Y})^{2&#125;&#125;&#36;&#36;
And then we use the text of the difference that was presented in the smooth model to estimate the noise. Bad
&#36;AR(p)&#36;
&#36;&#36;\hat{\sigma}<em>{e}^{2}=(1-\hat{\phi}</em>{1}r_{1}-\hat{\phi}<em>{2}r</em>{2}-\cdots-\hat{\phi}<em>{p}r</em>{p})s^{2}&#36;&#36;
&#36;MA(q)&#36;
&#36;&#36;\hat{\sigma}<em>{\epsilon}^{2}=\frac{s^{2&#125;&#125;{1+\hat{\theta}</em>{1}^{2}+\hat{\theta}<em>{2}^{2}+\cdots+\hat{\theta}</em>{q}^{2&#125;&#125;&#36;&#36;
&#36;ARMA(1,1)&#36;
&#36;&#36;\hat{\sigma}_e^2=\frac{1-\hat{\phi}^2}{1-2\hat{\phi}\hat{\theta}+\hat{\theta}^2}s^2&#36;&#36;</p>
<h4>Summary</h4>
<p>Based on the results of the previous calculations, we can easily draw the following conclusions.</p>
<ul>
<li>The rectangular estimates of the self-regression model are acceptable.</li>
<li>It's hard to get a rectangular estimate of sliding the average model.</li>
<li>The results of the rectangular estimates of the hybrid model are unacceptable.</li>
<li>The MA component causes the result of the rectangular estimation to be very poor.</li>
</ul>
<h3>Minimal 2x10 estimate</h3>
<p>And at this point, we're introducing an average in the flat model.&#36;\mu&#36; From that point on, the lowest two-fold estimate is a good way to estimate it.</p>
<h4>Self-Regression Model AR</h4>
<p>Consider&#36;AR(1)&#36;Situation
&#36;&#36;Y_t-\mu=\phi(Y_{t-1}-\mu)+e_t&#36;&#36;
We can see it.&#36;Y_t&#36;Cause variable &#36;Y_{t-1}&#36; As a regression model for the variable, the lowest quadrilateral study is the minimization of the deviation squared, which is
&#36;&#36;S_{\epsilon}(\phi,\mu)=\sum_{\iota=2}^{\pi}\bigl[(Y_{\iota}-\mu)-\phi(Y_{t-1}-\mu)\bigr]^{2}&#36;&#36;
Minimize
Based on the method of calculating the minimum two-fold factor, we can estimate it.
&#36;&#36;\mu=\overline{Y}&#36;&#36;
&#36;&#36;\hat{\phi}=\frac{\sum_{t=2}^n{(Y_t-\overline{Y})(Y_{t-1}-\overline{Y})&#125;&#125;{\sum_{t=2}^n{(Y_{t-1}-\overline{Y})^2&#125;&#125;\quad.&#36;&#36;
They're not accurate estimates, but the errors caused by the missing items can be ignored in terms of smoothing the process.
We can easily promote these results to higher-level AR models. Medium
<strong>The results given by the minimum quadrilateral are not very different from those estimated by AR model and rectangular.</strong></p>
<h4>Slide Average Model MA</h4>
<p>Think of the simplest.&#36;MA(1)&#36;Situation
&#36;&#36;Y_t=e_t-\theta e_{t-1}&#36;&#36;
It doesn't look like the least double-drive.
But we can give the MA model the nature of a near-regressive model, if reversible, as follows:
&#36;&#36;Y_t=-\theta Y_{t-1}-\theta^2Y_{t-2}-\theta^3Y_{t-3}-\cdots+e_t&#36;&#36;
Because of the parameters that need to be addressed&#36;\theta&#36;There's a non-linear presence, so we need to use some numerical solvers.
For higher-level situations, it's possible to solve it in an iterative manner.</p>
<h4>Mixed Model</h4>
<p>Consider&#36;ARMA(1,1)&#36;
&#36;&#36;Y_t=\phi Y_{t-1}+e_t-\theta e_{t-1}&#36;&#36;
We're still considering the squared-out minimization of the difference.
&#36;&#36;e_t=Y_t-\phi Y_{t-1}+\theta e_{t-1}&#36;&#36;
At this point, we're involved in selection.&#36;e_i&#36;The initial value, but we can choose freely, and in the case of a large sample, it has little effect on the final outcome.</p>
<h3>Very similar estimates.</h3>
<p>For the sequences and random season models of the length, the selection of the initial values has a significant impact on the final estimate of the parameters, so we have introduced the best method of estimation -- the MLE.</p>
<p>Appearance Functions&#36;L&#36;Defines the probabilistic density function for obtaining actual observations, which is also considered as the function of unknown parameters of the model when observations are fixed;&#36;ARIMA&#36;Models. After the observations.&#36;L&#36;It's the function of model parameters.</p>
<p>The exact estimation method is omitted here.</p>
<h2>Model diagnostics</h2>
<p>Now, let's consider the advantages of testing models, analysing the residuals, analysing over-parametric models, and they're also the usual ideas for the retrogressive diagnosis.</p>
<h3>Disability analysis</h3>
<p>The disability analysis is the most extensive analytical method involved in the synthesis of the problem, and the most basic part of the diagnosis of the entire model. <a href="/en/blog/2023/09/04/linear-regression-basics-notes/">Linear regression base</a> And here we're going to show some of the time series analysis methods.
The definition of disability is still the same as before.
&#36;&#36;\text{残差}=\text{实际值}-\text{预测值}&#36;&#36;</p>
<h4>Time series of disability</h4>
<p>An ideal, no pattern of the gap figure should show a rectangular dispersion of the line around zero horizontal lines without trend, as follows:
<img src="/assets/images/probability-statistics-notes/linear-time-series-analysis-notes-04.png" alt="Time series analysis 5">
This is a basic ideal time series. Figure
<strong>The time series of the residuals is usually used for the abnormal value test.</strong></p>
<h4>Normality test for disability</h4>
<p>In the process of modelling, we assumed that the disability had a normal analysis, and that the normality test for the disability was appropriate at this time, and that the more common methods of normality testing were the following:</p>
<ul>
<li>QQ Q Chart</li>
<li>Non-parameter tests such as Shapiro-Wilk Normality Test</li>
</ul>
<h4>Self-relevance of the disability</h4>
<p>We were looking at time series models and we were asking that the noise item be independent, and this noise item was shown in the sample as a disability.
For real white noise and bigger than that.&#36;n&#36;  (a) The normal distribution of the sample from a function that is almost unrelated and has zero equal value;
But even the correct identification model that is effective in estimating the parameters has different characteristics.
The normal difference is almost equal to the normal anger distribution of zero.&#36;\frac{1}{n}&#36;  For a large lag, the approximation difference.&#36;\frac{1}{n}&#36;  </p>
<h5>Intuitive approach</h5>
<p>Now we need to look at the specific problem; we can draw the ACF curve of sample differences, and if it's less than the criticality of the gap, then basically the disability is not relevant.</p>
<p><strong>It is particularly important to note that the ACF, which lags behind 4 (quarterly data)12 (monthly data), is likely to exceed the threshold, as we will describe in more detail where the seasonal model is located.</strong></p>
<h5>Ljung-Box Test</h5>
<p>We studied the self-relevant coefficients of the residuals that are lagging alone, and it's actually interesting to see these factors together.<a href="/en/blog/2023/03/18/mathematical-statistics-notes/">Mathematical statistics</a> and the study in the “Relevance coefficient” section
We're putting in statistics.
&#36;Q=n<em>{1}^{2}+\hat{r}</em>+\cdots+\hat{r}&#36;&#36;
If the estimate is correct,&#36;ARMA(p,q)&#36;Models, in large sample cases, it's almost submissive.&#36;\chi^2(K-p-q)&#36; </p>
<p>Unfortunately, this conclusion is not good for smaller samples, so Ljung and Box have proposed amendments to the typical sample capacity.
&#36;&#36;Q_{\star}=n(n+2)\Big(\frac{\hat{r}_1^2}{n-1}+\frac{\hat{r}_2^2}{n-2}+\cdots+\frac{\hat{r}_k^2}{n-K}\Big)&#36;&#36;
It's better than the statistics above are similar to the calorie distribution.</p>
<p>In practice, we'll reduce the freedom of the parameters according to the number of free parameters and adjust the parameters of the Ljung-Box test.</p>
<h3>Overcompatibility and argument redundancy</h3>
<p>This is another diagnostic method that is more important in time series analysis; the question we want to end is, in part, like&#36;AR(2)&#36;The model is already working well in small amounts, but we've been over the edge as well.&#36;AR(3)&#36;And this is the time to correct this error.
<strong>We need to minimize the complexity of models, as conditions permit.</strong></p>
<p>We believe there is an over-compatibility in the following circumstances.</p>
<ul>
<li>Additional parameters are not significant or 0</li>
<li>There is no change in the common parameters compared to the original estimate
In part, we'll use model numbers, like AIC values, to aid some of these judgments.</li>
</ul>
<p>We also have some suggestions at the design stage of the model to avoid overcompatibility.</p>
<ul>
<li>Use simple models to the extent possible and, if not, consider reprocessing by means such as disability analysis, if possible</li>
<li>Do not add the ranks of AR MA I</li>
</ul>
<h2>Model prediction</h2>
<p>In fact, predictions are the purpose of the time series modelling, and time series analysis is not at all as coefficients as retrogressive analysis can analyse, so so predictions are made and the accuracy of the predictions is the final part of the whole time series analysis.</p>
<h3>Minimal average error forecast</h3>
<p>Sequences can be obtained until time&#36;t&#36;The data, that's...&#36;Y_1,...Y_t&#36;  We want to predict the future.&#36;l&#36;The data of the period, that's...&#36;Y_{t+l}&#36;  We call time.&#36;t&#36;As the starting point for projections &#36;l&#36;For predicting step or advance time
In some of the certificates we omitted, we concluded that&#36;l&#36;The minimum average error of step is projected
&#36;&#36;&#36;&#36;1 }<em>t(\ell)=E(Y</em>|Y 1, Y 2, \\cdotp\cdotp, Y t) &#36;
This is a very important conclusion, and all the projections we take are the smallest MLE projections, and it is expected that this will simplify many of our problems.</p>
<h3>Trends in certainty</h3>
<p>Here, our certainty trends are not about observations that represent the real situation, but about the mechanisms that are inside our known models.
Here's the example of how we can understand the basics of our predictions.
Consider
&#36;&#36;Y_t=\mu_t+X_t&#36;&#36;
of which&#36;X_t&#36;is the known difference white noise of zero averages
&#36;&#36;00\
What's wrong with you?<em>{t}(\ell)&amp; =E(\mu</em>{t+\ell}+X_{t+\ell}|Y_{1},Y_{2},\cdotp\cdotp\cdotp,Y_{\iota})  \
&amp;=E(\mu_{t+\ell}\mid Y_1,Y_2,\cdots,Y_t)+E(X_{t+\ell}\mid Y_1,Y_2,\cdots,Y_t) \
&amp;=\mu_{t+\ell}+E(X_{t+\ell})
\end{aligned}&#36;&#36;
也就是
&#36;&#36;\hat{Y}<em>{\iota}(\ell)=\mu</em>{\iota+\ell}&#36;&#36;</p>
<p>Projected error is
&#36;e t}(\ell)=Y t+ell}-hat{Y}<em>{t}(\ell)=\mu</em>{t+\ell}+X_{t+\ell}-\mu_{t+\ell}=X_{t+\ell}&#36;&#36;
研究预测误差有
&#36;&#36;E(e_{t}(\ell))=E(X_{t+\ell})=0&#36;&#36;
&#36;&#36;\mathrm{Var}(e t}( \ell)=mathrm{Var} (X t+ell}=gamma &#36;&#36;&#36;)
That's what the prediction is, neutral and the difference is fixed.</p>
<h3>ARIMA projections</h3>
<p>Now we're extrapolating the models from data, then estimating the parameters, and then predicting them, and there's a different way of doing it, and we need to add a few sections of the carrying, and the rest of the narrative will be a whole, and we'll have to look at the whole picture.</p>
<h4>AR Model</h4>
<p>We're from non-zero averages.&#36;AR(1)&#36;The model begins, and then the natural extension understands the entire AR model.
Model form
&#36;&#36;Y_{t}-\mu=\phi(Y_{t-1}-\mu)+e_{t}&#36;&#36;
We're gonna make a step-by-step prediction.
&#36;&#36;Y_{t+1}-\mu=\phi(Y_t-\mu)+e_{t+1}&#36;&#36;
Based on the minimum MSE projections, we expect to have
&#36;&#36;&#36;&#36;1 }<em>{t}(1)-\mu=\phi[E(Y</em>{t}\mid Y_{1},Y_{2},\cdots,Y_{t})-\mu]+E(e_{t+1}\mid Y_{1},Y_{2},\cdots,Y_{t})&#36;&#36;
根据条件期望的性质我们可以实现化简
&#36;&#36;\hat{Y}<em>{t}(1)=\mu+\phi(Y</em>- You're not gonna get it.
I can see that. <strong>The first-order AR model is essentially compressed (coefficient absolutes less than 1) and the last model will conceivably reach the average.</strong>
Wants to predict more steps.<strong>Organisation</strong>Our predictions are good.</p>
<p>The prediction error of the model is also very well studied.
&#36;&#36;&#36;1=Y=<em>{t}(1)=\left[\phi(Y</em>{t}-\mu)+\mu+e_{t+1}\right]-\left[\phi(Y_{t}-\mu)+\mu\right]&#36;&#36;
也就是
&#36;&#36;e_t(1)=e_{t+1}&#36;&#36;
一步向前预测误差是AR噪声
我们也可以轻松的给出预测误差的方差（期望为0必然）
&#36;&#36;\mathrm{Var}(e_{t}(1))=\sigma_{e}^{2}&#36;&#36;
研究多步预测的误差有
&#36;&#36;=t(\ell)=e t+t+t\t\t\t\t\t\t\t\psi t t t+t+t+t\cdots+psi e t+t+t+t+t+t+t+
Of which&#36;\psi&#36;It's the form of a coefficient deformation. <strong>This form is set for the Alima model.</strong>
It's easy to judge.
&#36;&#36;\mathrm{Var}(e_t(\ell))=\sigma_e^2(1+\psi_1^2+\psi_2^2+\cdotp\cdotp\cdotp+\psi_{\ell-1}^2)&#36;&#36;
<strong>This form is also set up for all ARIMA models.</strong> </p>
<p>As for the other nature of this value, we're going to come to the conclusion that
<strong>For all the flat ARMA models, there are...</strong>
&#36;&#36;\mathrm{Var}(e_{\iota}(\ell))\approx\mathrm{Var}(Y_{\iota})=\gamma_{0},\text{对较大的 }\ell &#36;&#36;
We've come to three conclusions that go beyond the AR model.</p>
<h4>MA Model</h4>
<p>We're here to consider how to deal with the average slide ingredient.&#36;MA(1)&#36; Yes.
&#36;&#36;Y_t=\mu+e_t-\theta e_{t-1}&#36;&#36;
One step forward, and the conditions are expected to be.
&#36;&#36;&#36;&#36;1 }<em>{\iota}(1)=\mu-\theta E(e</em>{t}|Y_{t},Y_{2},\cdots,Y_{t})&#36;&#36;
 然而我们知道（这是一个在&#36;t&#36;较大的时候成立的逼近结论）
&#36;&#36;E(e_t\mid Y_1,Y_2,\cdotp\cdotp\cdotp,Y_t)=e_t&#36;&#36;
 因此 一步预测的表达式为
&#36;&#36;\t(1)=mu \theta e t&#36;
Of which&#36;e_t&#36;As a first&#36;t&#36;The difference in the steps was determined when the model was ready.</p>
<p>The MMA model's multistep predictions show some slight changes, as follows:
&#36;&#36;&#36;&#36;1 }<em>t(\ell)=\mu+E(e</em>\midY 1, Y 2, \cdotp\cdotp, Y t) \theta e(e t+ell )\midY 1, Y 2, \cdotp\cdotp, Y t) &#36;
We found out we were right over&#36;t&#36;The steps are broken without any knowledge. <strong>There's no real value. No disability.</strong> But we know.
Disabled&#36;e_{t+l}&#36; and&#36;Y_i&#36;So the condition is zero, so the MMA model's multistep prediction is,
&#36;&#36;&#36;&#36; =m,&#125;&#125;Y &gt;1&#36;&#36;
<strong>We don't introduce the M.A.'s prediction error. We'll just follow the three conclusions that AR gave us.</strong></p>
<h4>ARMA Model</h4>
<p>We're giving the predictions directly in the form of:
&#36; \begin{aligned}<em>i(\ell)&amp;=\phi_1\hat{Y}<em>i(\ell-1)+\phi_2\hat{Y}<em>i(\ell-2)+\cdots+\phi_p\hat{Y}<em>t(\ell-p)+\theta_0-\theta_1E(e</em>{t+t-1}\mid Y_1,Y_2,\cdot\cdot\cdot,Y_t)\&amp;-\theta_2E(e</em>{t+t-2}\mid Y_1,Y_2,\cdot\cdot\cdot,Y_t)-\cdot\cdot\cdot-\theta_eE(e</em>{t+t-q}\mid Y_1,Y_2,\cdot\cdot\cdot,Y_t)\end{aligned}&#36;&#36;
其中有
&#36;&#36;E(e</em>{t+j}\mid Y_1,Y_2,\cdots,Y_t)=\begin{cases}0&amp;j&gt;0\e_{t+j}&amp;I'm not gonna leave you alone.
This is a reasonable estimate of the value of the error requirement at &#36;j.&gt;We had no disability to use at the time of &#36; 0, and the expectation of error was zero, of course, which means that...
<strong>When the pace is large enough, the impact of the noise item will be weak, mainly influenced by self-regressive parameters.</strong>
In conclusion,
When the predicted step is less than&#36;q&#36;And sometimes, the noise item can be directly associated with our prediction model, or else the noise item can influence the later model by influencing the self-regressive item take-off in the front.</p>
<p>Based on the formula below
&#36;&#36;00begin{aligned}\t(\ell)\mu&amp;=\phi_1[\hat{Y}_t(\ell-1)-\mu]+\phi_2[\hat{Y}_t(\ell-2)-\mu]+\cdots\&amp;+\phi_p[\hat{Y}_t(\ell-p)-\mu],\quad\ell&gt;q\end{aligned}
Similar to&#36;ARMA&#36;Model&#36;p_k&#36; We can tell the conclusion based on Yule-Walker's deductions.
<strong>The formula in front will be index decay combined with a fast decline of the sineline to zero.</strong>
That's right.
<strong>The steady ARMA model long-term projections are constricted to the average.</strong> It also responds to the relative nature of the ARMA models.</p>
<p>We're going to repeat here the conclusions we've already made about the disability.
&#36;&#36;\mathrm{Var}(e_{\iota}(\ell))\approx\mathrm{Var}(Y_{\iota})=\gamma_{0},\text{对较大的 }\ell &#36;&#36;</p>
<h4>Random Swim with drift</h4>
<p>To deal with the Alima model, we'll start by introducing some of the necessary models.
&#36;&#36;Y_t=Y_{t-1}+\theta_0+e_t&#36;&#36;
At this point, one step forward, and the conditions are expected to be met.
&#36;&#36;&#36;&#36;1 }<em>\iota(1)=Y</em>\ota+\theta &#36;0
The constant gap allows for multi-step predictions.
If&#36;\theta_{0}\ne 0&#36; So no matter how many steps we make, we're not gonna shrink, but we're going straight ahead.</p>
<p>Because the constants will change the nature of the predictions, the non-stable ARIMA model should avoid the presence of the constants as much as possible when studying after the differentials, unless we clearly find that the average of the difference sequence is zero.</p>
<p>We don't study the prediction errors alone, but we'll introduce them in the ARIMA model.</p>
<h4>ARIMA Model</h4>
<p>The ARIMA model's predictions are not a difficult one to solve.</p>
<h5>Unstable trends</h5>
<p>If the margin is not zero and the ARMA model is not zero, then the ARIMA model has a trend in the margin, not a steady trend.</p>
<h5>About the error</h5>
<p>Gives a direct conclusion on the error.
&#36;&#36;E(e_{t}(\ell))=0,\ell\geqslant1&#36;&#36;
&#36;&#36;\mathrm{Var}(e_t(\ell))=\sigma_e^2\sum_{j=0}^{\ell-1}\psi_i^2,\ell\geqslant1&#36;&#36;
The latter is a non-consumable grade, which is...
<strong>The difference between the unstable prediction error will increase and not be on the line.</strong>
It's very reasonable, after all, that the future of the unstable sequence is quite uncertain.</p>
<h3>Projections after changing sequences</h3>
<h4>Difference</h4>
<p>We have a very natural idea.
<strong>Forecast a smooth sequence after the margin, then add the value of the original sequence.</strong>
Actually, it worked very well.</p>
<h4>Like the Box Cox conversion.</h4>
<p>And one very natural idea is that we're modelling the sequences after the transformation and then reverse the predictions.
&#36;&#36;E(Y_{t+t}|Y_{t},Y_{t-1},\cdotp\cdotp\cdotp,Y_{1})\geqslant\exp[E(Z_{t+t}|Z_{t},Z_{t-1},\cdotp\cdotp\cdotp,Z_{l})]&#36;&#36;
That's the idea that there's no guarantee of the smallest MSE. </p>
<p>Well, good thing we don't need much more work, but according to the rectangular function, there's a lot of work to be done.
If&#36;X&#36;Subject to normal distribution
&#36;&#36;E[\exp(X)]=\exp[\mu+\frac{\sigma^2}{2}]&#36;&#36;
So the smallest MSE of the original sequence is projected to be
&#36;&#36;\exp\left(hat})<em>{t}(\ell)+\frac{1}{2}\mathrm{Var}[e</em>(♪ ♪ ♪ ♪ I'm not sure what I'm gonna do ♪
The latter is the difference between the predicted error and the expected error.
<strong>Not so much as a direct change of approach to MSE.</strong></p>
<h2>Unit Root Process</h2>
<p>Typical non-stable time series model is unit root non-smooth time series</p>
<h3>I'm just gonna swim around.</h3>
<p>See the section “Swam randomly” in this paper.</p>
<h3>Random Swimloads with Floating Items</h3>
<p>We're thinking about the randomly moving model.
&#36;&#36;p_t=\mu+p_{t-1}+\varepsilon_t,t=1,2,\ldots &#36;&#36;</p>
<p>And considering the features we've been thinking about,</p>
<ul>
<li>The difference remains unchanged.</li>
<li>Average added a trend item
<strong>The model remains unpredictable, and the sequences that are theoretically swinging near the line are out of regularity because of the large variance.</strong></li>
</ul>
<p>Random Swimming with Drifting&#36;p_t&#36;, can be broken down into two parts:
&#36;&#36;p t=(p 0+\mu)+p t^<em>I'm sorry.
Of which &#36;p t^</em>=\sum_{j=1}^t\varepsilon_t&#36;是从0出发的不带漂移的随机游动，&#36;p 0+\mu t&#36; is a non-random linear trend.</p>
<h3>Fixed trend model</h3>
<p>The presentation of the constant trend model is available for reference in this paper, Trends I. Section</p>
<h3>The link between the constant trend model and random migration</h3>
<p> Randomly Swim&#36;p_t=p_{t+1}+\varepsilon_t&#36;Disturbing with fixed trends &#36;Y_t=a+bt+X_t&#36;(of which)&#36;&#123;X_t}&#36;The trend is slow. </p>
<p>The difference is:</p>
<ul>
<li>The randomly moving differentials are linear increases, and the observed differences in fixed trends are constant;</li>
<li>The impact of randomly moving disturbances is permanent, and the impact of fixed trends is only at one moment (if disturbed)&#36;X_t&#36;It's white noise or very short time.&#36;X_t&#36;is a linear time series);</li>
<li>(a) The trend of randomly moving is not fixed and the shape of the change of the fixed trend is fixed;</li>
<li>Fixed trend model&#36;Y_t&#36;minus a fixed regression function&#36;Y=a+bt&#36;The blogger says that the government is not a party to the law.<strong>Randomly moving minus any non-random function cannot be stabilized and can be smoothed by differentials.</strong></li>
</ul>
<h3>ARIMA Model</h3>
<p>The description of the ARMA model is based on the section on "Stable and Random Time Series ARMA" here.
The ARIMA model adds a margin to the ARAMA model.</p>
<p>If&#36;Y_t&#36;It's already weak and smooth, and it's not right.&#36;Y_{t}&#36;Make a difference. (very naturally)
If&#36;Y_t&#36;Yes.<strong>The linear trend of non-random is smoother, and although the differentials can be smoother, they should not be used to make the difference, but rather to make it back.</strong>, the difference is used to introduce unnecessary unit roots in the MA section of the ARMA model.</p>
<h3>Index smoothing model</h3>
<p>The index smoothing is the first method of predicting a simple method: predicting the value of the next point in the linear combination of historical data, the linear combination coefficient declines by negative index (geometric) over distance.
&#36;&#36;&#36;500<em>h(1)\approx wx_h+w^2x</em>{h-1}+\cdots=\sum_{j=1}^\infty w^jx_{h+1-j}&#36;&#36;
加权平均需要满足权重和为1 则
&#36;&#36;\hat{x}<em>h(1)=(1-w)(x_h+wx</em>=h^x h}ldots=1-w\sum\j=
So for index smoothing, we can use the ARIMA model to study index smoothing without wasting time in modelling alone.</p>
<h3>Organisation</h3>
<p>We can use the unit root test to determine whether a process is a unit root; it assumes that a process is a unit root (with unit root) and that if the original assumption is not rejected, it can be considered a unit root process.</p>
<p>The unit root process is a good way to determine that the model is flat. Steady.</p>
<p>We can select the pattern of the unit root process, which will determine whether the remaining residuals support the non-parity after the trend is formulated. Steady.</p>
<h2>Seasonal Model</h2>
<p>Seasonal data, or life cycle data, should be common in time series analysis; the models we have described before cannot explain these data. </p>
<p>We'll find that the simpleness of the gap is still very relevant in many ways.</p>
<h3>Season ARMA Model</h3>
<h4>Seasonal MA Model</h4>
<p>We'll start with the flat model.&#36;s&#36;The data are general.&#36;s=12&#36; Quarterly data are general&#36;s=4&#36;
Consider the following form of model
&#36;&#36;Y_t=e_t-\Theta e_{t-12}&#36;&#36;
We can easily verify:
&#36;&#36;\mathrm{Cov}(Y_{t},Y_{t-1})=\mathrm{Cov}(e_{t}-\Theta e_{t-12},e_{t-1}-\Theta e_{t-13})=0&#36;&#36;
&#36;&#36;\mathrm{Cov}(Y_{t},Y_{t-12})=\mathrm{Cov}(e_{t}-\Theta e_{t-12},e_{t-12}-\Theta e_{t-24})=-\Theta\sigma_{e}^{2}&#36;&#36;
It's easy to see that the sequence is stable and only 12 steps behind is self-relevance.</p>
<p>According to the above, we define the season cycle as&#36;s&#36;Yes.&#36;Q&#36;Step MA Model &#36;MA(Q)&#36;
&#36;&#36;Y_t=e_t-\Theta_1e_{t-s}-\Theta_2e_{t-2s}-\cdots-\Theta_Qe_{t-Qs}&#36;&#36;
The condition for reversibility is the same as the one for the front.
It's the same function as the one before it, and it turns zero after several steps.</p>
<h4>Seasonal AR Model</h4>
<p>Very naturally, it defines the season AR model.&#36;AR(P)&#36;
&#36;&#36;Y_{\iota}=\Phi_{1}Y_{t-s}+\Phi_{2}Y_{t-2s}+\cdots+\Phi_{P}Y_{t-Ps}+e_{t}&#36;&#36;
We'll come to a conclusion.</p>
<ul>
<li>The conditions for stability remain unchanged.</li>
<li>The associated function is the combination of index decay and blocking nitro sine.</li>
</ul>
<h3>Multiplication season ARMA model</h3>
<p>It's not worth considering the previous models that only have their own relevance in terms of seasonal lag, and it's completely the same as ARIMA, which is not really meaningful.</p>
<p>Now we want to combine the thinking on the season model Alima and the previous study of the Alima, the models that contain relevance not only in the season lag, but also in the near future.</p>
<p>We'll give you two examples.
&#36;&#36;Y_t=e_t-\theta e_{t-1}-\Theta e_{t-12}+\theta\Theta e_{t-13}&#36;&#36;
&#36;&#36;Y_{t}=\Phi Y_{t-12}+e_{t}-\theta e_{t-1}&#36;&#36;
Give two typical ACF curves at the same time.
<img src="/assets/images/probability-statistics-notes/linear-time-series-analysis-notes-05.png" alt="Time series analysis 6">
They're all typical, and they fit the model we've been talking about.</p>
<p>We'll take the model we're introducing right now.
&#36;&#36;ARMA(p,q)\times(P,Q)&#36;&#36;
What does it mean?</p>
<ul>
<li>Non-seasonal portion&#36;ARMA(p,q)&#36; </li>
<li>The season itself.&#36;ARMA(P,Q)&#36;</li>
</ul>
<h3>ARIMA model for unstable seasons</h3>
<p>The difference, or the non-stable core step.&#36;s&#36;♪ The season differentials
&#36;&#36;\nabla_{s}Y_{t}=Y_{t}-Y_{t-s}&#36;&#36;
And then you combine the difference in your own model, and you can multiply the ARAMA model for it has a model. </p>
<ul>
<li>Seasonal cycle&#36;s&#36;</li>
<li>Non-seasonal steps&#36;ARIMA(p,d,q)&#36;</li>
<li>The order of the season&#36;ARIMA(P,D,Q)&#36;
It's a big model.</li>
</ul>
<h3>Seasonal ARIMA model recognition, preparation, testing, predictions.</h3>
<p>The core approach has been described in the "Structure Identification" section of this paper, the "parameter estimation" section, the "Model Diagnostic" section of this paper.
Here's a few separate presentations on the seasonal model.</p>
<h4>Seasonal ARIMA model recognition</h4>
<ul>
<li>Study Time Series Chart ACF PACF Determination of Stable and Periodicity</li>
<li>Consider normal differentials and try to capture stability.</li>
<li>Consider the seasonal differentials and try to eliminate cyclicality.</li>
<li>Study whether the sample ACF is free of self-relevant (the difference is to eliminate all self-relevant)</li>
</ul>
<p><strong>The sequences of the season ARIMA model after a first-order differential and the seasonal differential often lag behind the positions 1, 4, 5 (in the case of monthly data, by 1, 12, 13) to synthesize the ACF PACF time-series judgement, and have not eliminated the smoothness and cyclicality</strong></p>
<h4>Seasonal ARIMA model preparation</h4>
<p>We're looking for the MLE formula, and the code will help us calculate the results.</p>
<h4>Seasonal ARIMA model diagnosis</h4>
<p>Or is it a study of the disability?</p>
<h4>Seasonal ARIMA model prediction</h4>
<p>As expected, the best way to predict the season model is to use the margin to predict.
Consider &#36;ARIMA(0,1,1)\times (1,0)<em>Yeah.
&#36;Y t-Y</em>{t-1}=\Phi(Y_{t-12}-Y_{t-13})+e_t-\theta e_{t-1}-\Theta e_{t-12}+\theta\Theta e_{t-13}&#36;&#36;
一步向前预测就有
&#36;&#36;\hat{Y}<em>t(1)=Y_t+\Phi Y</em>Other Organiser&#36;&#36;&#36;US&#36;US&#36;
More steps are the same, and we still need to consider the fact that noise items sometimes include projections in the form of disability, sometimes in the form of self-return.
<strong>The ARIMA model of the seasons has two parts, one of the pre-time trends, the other of the cyclical segments, and they're our ARIMA seasons.</strong></p>
<h3>Seasonal virtual variable</h3>
<p>Another method of expressing seasonality is to express a fixed seasonal pattern using non-random regression items. It's possible that this pattern can be eliminated by seasonal differences, but it's also relevant to dynamic models and<strong>Similarity of non-random linear trend models</strong>, fixed seasonal models should not be treated with seasonal differentials.
<strong>If we can find a certain trend, and we can smooth it down, then we don't need to use differential treatment.</strong></p>
<p>Non-random seasonal factors are expressed in the return to the mute variable.&#36;s=4&#36;, the fixed level of the four different seasons is expressed using three dumb variables. To determine whether the non-random season model is used, it is possible to prepare dynamic season ARIMA&#36;(1,0,1)(1,0,1)_s&#36;Models, where seasonal factors are found to be negligible, may consider non-random seasonal models. Examples are given below.</p>
<p>Give Sequence Chart
<img src="/assets/images/probability-statistics-notes/linear-time-series-analysis-notes-06.png" alt="Linear Time Series Analysis">
Can't see a clear cycle.
Make an ACF map.
<img src="/assets/images/probability-statistics-notes/linear-time-series-analysis-notes-07.png" alt="Linear Time Series Analysis 1">
The 12th grade lag is clearly not zero, reflecting a cyclical pattern.
ARIMA season for dynamic&#36;(1,0,1)(1,0,1)_{12}&#36;
Found <code>sar1 = 0.9882,sma1 = -0.9142</code>  Write as a physical model
&#36;&#36;(1+0.0639B)(1-0.9882B^{12})(X_t-0.0117)=(1+0.2508B)(1-0.9142B^{12})\varepsilon_t&#36;&#36;
They can be approximated, which means they can consider a regression model for the seasonal dummy variable, which is essentially a regression problem.<strong>Returns to calculate trend</strong> The time series features are not considered at this time.</p>
<p><strong>This seems very credible, but in fact, time series data are highly self-relevant, and we'll find this in our analysis of the retrogressive residuals, and it is hasty and irresponsible to take a direct look at trends in the time series and end the analysis.</strong></p>
<h2>Retrieval model with time series errors</h2>
<p>In statistical analysis, linear regression analysis is one of the most commonly used analytical tools, a linear regression is as follows:
&#36;&#36;Y_t=\beta_0+\beta_1X_t+e_t,t=1,2,\ldots,T&#36;&#36;
We're going back to the model and we're going to ask for the disability.&#36;e_t&#36; Independent as normal distribution </p>
<p>But in the regression analysis of financial implications, time series are frequently present, including the regression analysis we use when going to trends, where they all have non-independent disabilities, based on the disability.&#36;e_t&#36; The independent and normal distribution of the estimates, like the standard error estimates, the hypothetical tests, are no longer valid. <strong>The regression factor estimates are still credible.</strong></p>
<p>Studies have shown that when the disability is positively correlated, the standard error estimate for the regression factor is low, making it corresponding.&#36;t&#36;and&#36;F&#36;Test absolute values for statistics </p>
<p>When?&#36;&#123;e_t}&#36;When the ARAMA sequence is flat and reversible, linear regression models can be estimated at the same time as the smooth reversible ARAMA sequence, and standard error estimates, hypothetical tests and projections are available.  <code>arima()</code>Function provides <code>xreg=</code> to introduce a regression variable.</p>
<p>If you don't care,&#36;&#123;e_t}&#36;The correct estimate of the regression factor for the SE and the correctness of the hypothetical test can be assumed only&#36;&#123;e_t}&#36;The structure of the agreement is not considered.&#36;&#123;e_t}&#36;Modelling. Like<a href="/en/blog/2023/09/04/linear-regression-basics-notes/">Linear regression base</a> Section on “Minimum 2x (weighted OLS)”</p>
<p>The basic steps for modelling are:</p>
<ol>
<li>To integrate a linear regression model and test the sequence relevance of the residual</li>
<li>If the disability sequence is non-stable in unit, the difference is one-ordered for both the variable and the self-variant. Then we take the first step in the sequence after the difference. If the residue sequence is flat, identify an ARMA model for the residue sequence and modify the linear regression model accordingly.</li>
<li>The regression model is estimated jointly with the ARMA model using the maximum semblance estimation method and the model is tested for improvement. The main use is for white noise testing of the disability using Ljung-Box.</li>
</ol>
<h2>Long Memory Model</h2>
<p>ACF is an important reference for time series modelling.</p>
<ul>
<li>For ARMA sequences, when delayed&#36;k\to\infty&#36;The sample ACF is now zero negative index.</li>
<li>Theoretically, ACF is not defined for unit root, since the SACF is defined for weak smooth column, and its sample ACF is in sample volume&#36;T\to\infty&#36;Every time&#36;\hat{\rho}_k&#36;Both trend &#36;1.00 (k)&gt; 0)&#36; 。</li>
</ul>
<p>There are some flat time series ACFs, though they also lag behind.&#36;k\to\infty&#36;It's zero, but it's slow to zero, only negative.&#36;k^{-\alpha}&#36; This speed. This represents a slow reduction in the self-relevance of the sequence as it evolves from distance to distance, which is described as a long-term memory time series.</p>
<p>Note that the time series of long memory remains weak and stable, and the unit roots are not called long memory, although they are highly self-relevant from a distance.</p>
<p>In financial time series modelling, long memory models can be considered if the sample ACF values are small but the decline is particularly slow. If the values are large and slow, it could be the unit roots that are not flat, or the ARAMA sequence that has very close to the one.</p>
<p>The typical model of the long memory time series is the fractional balance flat column, the model is the
&#36;&#36;1-B ^dX t=\xi t,:-0.5&lt;d&lt;&#36; 0.5 million
of which&#36;&#123;\xi_t}&#36;It's zero-average-symmetrical, and white noise.</p>
<p>If we talk about this white noise spreading to the point where we can get it,&#36;ARMA(p,q)&#36;Sequences.&#36;ARFIMA(p,d,q)&#36; Model called the fractional grade differential ARMA model.</p>
<h2>Unstable from mutations</h2>
<p>The second reason for non-stableness is that the overall regression function has changed during the sample period. In economics, many of the reasons for this are the sudden changes in an industry as a result of changes in economic policy, changes in economic structure, and innovation. If these changes or “matures” occur, but these factors are not taken into account in the model, they will shake the basis for our predictions and extrapolations.</p>
<p>This section presents the thinking for the two time series regression models to detect mutations.</p>
<p>The first idea is to look for possible mutations from the perspective of the hypothetical test and to pass&#36;F&#36; The statistics test the significant change in the regression factor.</p>
<p>The second idea is to look for possible mutations from the perspective of prediction: The projection is based on the assumption that the sample was completed by the actual end of the sample period and the results of the projection are evaluated. If the forecast results are significantly reduced in accuracy, a mutation is assumed.</p>
<p>We have just described the unstableness of trends without considering mutations.</p>
<h3>What's mutation?</h3>
<p>Mutant variations may occur in the general regression function, sudden changes at a given time, or in a long-term evolution.</p>
<p>Unbridled changes in macroeconomic data can result from large-scale changes in macro-policy.</p>
<p>Mutant changes may also result from the evolution of the overall regression function over time, such as slow economic policy reforms and gradual changes in the economic structure.</p>
<p><strong>The methods described in this section for detecting mutations can be used to test sudden mutations as well as mutations caused by long periods of evolution.</strong></p>
<h3>The mutation test</h3>
<p>One of the methods used to detect mutations is to test the discrete or mutation of the regression coefficient. The specific test will depend on whether the moment of the mutation point occurs.</p>
<h4>Known Time</h4>
<p>Test of known time mutations. In some cases, you may suspect that a mutation has occurred at a given point in time.</p>
<p>If the date of the possible mutation is known, the double variable cross-entry model can be used to test the zero scenario, which should be non-maritime. For the sake of simplicity, we will consider the ADL (1,1) model, which includes the interpretation variables, the cut-off, the cut-off, and the other variables.&#36;Y_t&#36;♪ And the first step behind ♪ &#36;X_{\iota}&#36;The first step is lagging. Use&#36;\tau&#36;The event is a very important one for the world.&#36;D_\iota(\tau)&#36; is a binary variable with a value of 0 before the mutation period and 1 after the mutation period, i.e., when&#36;t\leqslant\tau&#36;The blog is also available.&#36;D_t(\tau)=0&#36;, when &#36;t&gt;\tau&#36;时， &#36;D (\tau)=&#36;1.00. The regression equations with binary mutation indicator variables and all cross-cutting items are:
&#36;&#36;Y_t=\beta_0+\beta_1Y_{t-1}+\delta_1X_{t-1}+\gamma_0D_t(\tau)+\gamma_1\begin{bmatrix}D_t(\tau)\times Y_{t-1}\end{bmatrix}+\gamma_2\begin{bmatrix}D_t(\tau)\times X_{t-1}\end{bmatrix}+u_t&#36;&#36;
If no mutation occurs during the sample period, the overall regression function should be the same in both stages, i.e. all of which are contained&#36;D_t(\tau)&#36;, and the coefficient for each item should be zero. That is, the zero assumption should be no mutation during the sample period, i.e.&#36;\gamma_0=\gamma_1=&#36; &#36;\gamma_2=0&#36;I'm sorry. The alternative scenario is that mutations exist during the sample period, which means that the overall regression function is at the mutation point&#36;\tau&#36;Different, i.e.&#36;\gamma_0&#36;、&#36;\gamma_1&#36;、&#36;\gamma_2&#36;At least one is not zero. So, if there was a mutation in the sample period, it could be...&#36;F&#36;Statistical testing</p>
<h4>Unknown time</h4>
<p>If we don't know the specific time point at which the mutation occurs, then we can consider multiple mutations of known time, and fortunately, the form of integration has been studied.</p>
<p>This improved Zou is usually called Quant Likelihood Ratio (QLR) Statistic (which we will use to indicate this test below), or sup-Wald statistics.</p>
<h4>Hypothetical test for external prediction</h4>
<p>The test of the accuracy of the model's prediction ultimately depends on its ability to predict outside the sample, i.e., its ability to predict in the “actual projection range” after the model's estimation has been completed.</p>
<p>Hypothetical external predictions (Psueudo Out-of-sample Forecasting) are a method used to simulate prediction models for predicting performance in actual projection ranges. Its approach is simple: select a point of time at the end of the sample range, use data from before that point of time to estimate the model and then use the estimated model to project the observations at the end of the sample. Repeating the above steps at multiple points of time at the end of the sample can provide multiple false predictions and multiple false prediction errors. These errors are used to test whether we are satisfied with the stability of the predicted relationship.</p>
<p>It's also a way to help us judge whether mutation is not a source of stability.</p>
<h3>Handle mutations</h3>
<p>The handling of mutations is a very complex issue, and we don't have any information here about how to deal with mutations.</p>
