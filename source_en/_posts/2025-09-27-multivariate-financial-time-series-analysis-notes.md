---
title: 'Multivariate Financial Time Series Analysis: VAR, cointegration, and state space models'
title_zh: 多元金融时间序列分析：向量自回归、协整与状态空间模型
date: 2025-09-27 18:50:29 +0800
categories:
- Data Science
- Time Series & Spatial Data
tags:
- Statistics
- Time Series
- Financial Time Series
author: Hyacehila
mathjax: true
hidden: true
excerpt: Covers VAR, cointegration, state space models, Kalman filtering, and multivariate volatility modeling.
description: Covers VAR, cointegration, state space models, Kalman filtering, and multivariate volatility modeling.
excerpt_zh: 整理向量自回归、协整、状态空间模型、卡尔曼滤波和多元波动建模。
permalink: /blog/2025/09/27/multivariate-financial-time-series-analysis-notes/
lang: en
translation_key: 2025-09-27-multivariate-financial-time-series-analysis-notes
translation_status: machine
translation_source_hash: 4cf1f8955ae7a58bc481e94f01aa57ba321e920aa7bff1edd07d6f2063441709
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<h2>Vector self-regression model</h2>
<p>The global integration of the economy and the development of information dissemination have linked financial markets, and price changes in one market can spread quickly to another. Investors holding multiple assets also wanted to know about the relationship between returns on multiple assets. These issues fall within the context of a multi-temporal sequence analysis. We start with this chapter by looking at multiple time series analysis, rather than treating them as individual analyses.</p>
<h3>Multi-time series basic concept</h3>
<h4>Weak Smooth Row</h4>
<p>When a multiple time series &#36;r_{t}= {r_{1t}...r_{nt&#125;&#125;&#36; Call it a plural smooth column if the following conditions are met
&#36;&#36;\begin{cases}E\bardsymbol{r} t=\bardsymbol{mu}\text{not }t\text{){mathrm{<del>= = \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ = = = = = = { { { { { { { { { { { { { { { { { { { { { { {</del>Cov}(\boldsymbol{r}<em>t,\boldsymbol{r}</em>== sync, corrected by elderman == @elder man&amp;You're not gonna get away with this?
It can be seen that the concept of multiple and weak stratification is naturally transformed from the one dollar wide and smooth.<a href="/en/blog/2023/03/18/stochastic-process-basics-notes/">Random Process Basis</a> "Soft Process" section</p>
<h4>Interrelate Matrix</h4>
<p>A single-dollar time series would require only a study of the factors associated with the deviation and lag, but the diversity would need to be considered more.</p>
<p>Let's remember.
&#36;&#36;\rho_{ij}(0)=\mathrm{corr}(r_{it},r_{jt})=\frac{\mathrm{Cov}(r_{it},r_{jt})}{\sqrt{\mathrm{Var}(r_{it})\mathrm{Var}(r_{jt})&#125;&#125;=\frac{\Gamma_{ij}(0)}{\sqrt{\Gamma_{ii}(0)\Gamma_{jj}(0)&#125;&#125;&#36;&#36;
To delay the 0-sync multi-time series interconnective matrix, he was a symmetric matrix of all 1 symmetrical elements of the diagonal sequence, studying the relevance of the various sub-series of the multi-time series, which he had modified from the normal synoptic matrix.</p>
<p>To study the lag relationship, we define it.&#36;k&#36; Weak and smooth sequences &#36;r_t&#36; Delay&#36;l&#36; The matrix of the mutual agreement is
&#36;&#36;00-&#36;-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-0-0-00-0-00-00-0-<em>t-\boldsymbol{\mu})(\boldsymbol{r}</em>&#36; ^t-l}bardsymbol(m)
He's also a natural extension of the one dollar self-conciliation difference, relying on delay rather than time.</p>
<p>Delay in obtaining amendments from them&#36;l&#36; The interconnective matrix is
&#36;&#36;\rho_{ij}(l)=\mathrm{corr}(r_{it},r_{j,t-l})=\frac{\Gamma_{ij}(l)}{\sqrt{\Gamma_{ii}(0)\Gamma_{jj}(0)&#125;&#125;&#36;&#36;
He's not normally symmetrical, and when the lag-related matrix is not zero, we usually call it a pioneer.</p>
<p>The method of calculating the matrix of sample interconnectivity is easy to imagine.
&#36;a ..<em>l=\frac1T\sum</em>{t=l+1}^T(\boldsymbol{r}<em>t-\bar{\boldsymbol{r&#125;&#125;)(\boldsymbol{r}</em>{tl}-bar(baroldsymbol{r}) ^T&#36;
Sample interconnective matrix can be calculated from the inter-coordinate matrix.</p>
<h4>Classification of linear dependencies between time series</h4>
<p>The interconnectivity of multiple time series reflects the linear dependencies of the time series, and here is a simple summary chapter.</p>
<p>We've recorded the interconnectivity of multiple time series as &#36;p_l&#36; The elements are: &#36;r_{ij}(l)&#36; Then we can give it to you.</p>
<ul>
<li>Diagonal elements &#36;r_{ii}(l)&#36;  It's a one-dollar time series. &#36;r_{it}&#36; ACF</li>
<li>&#36;p_{ij}(0)&#36; It's two points. &#36;r_{it},r_{jt}&#36; Synchronise Linear Relationships</li>
<li>&#36;p_{ij}(l)&#36;  It's reeling. &#36;r_{it}&#36; Yeah.&#36;r_{jt}&#36;  And zero is not linear.</li>
</ul>
<p>Based on the difference. &#36;p_{ij}(l)&#36; And in the case of a multi-temporal sequence, we can divide it into one.</p>
<ul>
<li>&#36;p_{ij}(l)=p_{ji}(l)=0&#36;  For Any&#36;l&#36; The two sequences are irrelevant.</li>
<li>&#36;p_{ij}(l)=p_{ji}(l)=0&#36;  For Any &#36;l&gt;It's a zero-dollar split.</li>
<li>One is not a zero, called a one-way guide and lag.</li>
<li>Neither of them is zero, called mutual guidance and lag.</li>
</ul>
<h4>Multiple-compositing Test</h4>
<p>The one-dollar Ljung-Box white noise test was extended to a variety of situations. Test zero for a multi-series.
&#36;&#36;H_0:\boldsymbol{\rho}_1=\cdots=\boldsymbol{\rho}_m=\boldsymbol{0}&#36;&#36;
The opposing assumption is not all zero matrix.</p>
<p>Use test statistics
&#36;&#36;Q_k(m)=T^2\sum_{l=1}^m\frac1{T-l}\mathrm{tr}(\hat{\Gamma}_l^T\hat{\Gamma}_0^{-1}\hat{\Gamma}_l\hat{\Gamma}_0^{-1})&#36;&#36;
You can achieve a white noise test similar to the Ljug-Box, and determine if the sequences are white noise.</p>
<h3>VAR Model Foundation</h3>
<h4>VAR Model Structure</h4>
<p>The most common of multiple asset-rate joint models is the Vector Autoregression, VAR model, which we give you.&#36;k&#36;Won's&#36;VAR(1)&#36; The model structure is...
&#36;&#36;\bardsymbol{r}<em>0+\boldsymbol{\Phi}\boldsymbol{r}</em>\bardsymbol}t&#36;
of which&#36;\phi_0&#36; Yes.&#36;k&#36;Other Organiser &#36;\Phi&#36; Yes.&#36;k&#36; Array &#36;a_t&#36; It's a error column. It's usually assumed to be zero.&#36;k&#36;Normal distribution</p>
<p>Consider &#36;k=2&#36; The model structure is becoming
&#36;&#36;\left{\begin{array}{l}r_{1t}=\phi_{10}+\phi_{11}r_{1,t-1}+\phi_{12}r_{2,t-1}+a_{1t}\r_{2t}=\phi_{20}+\phi_{21}r_{1,t-1}+\phi_{22}r_{2,t-1}+a_{2t}\end{array}\right.&#36;&#36;
If &#36;\phi_{12}=\phi_{21}=0&#36; If you're separated, you can call the two sequences separate.&#36;a_t&#36; It's not relevant, and we call it non-conformity.
Conversely, if the coefficient for determining the separation is not zero, it is called that the two sequences are feedback-to-respond.</p>
<p>Statistics have their own way of explaining the relationship between these feedbacks, when two sequences of feedbacks are made.&#36;a_{1t}&#36; and &#36;a_{2t}&#36; It's not relevant, it's called a transmission function. We can adjust it.&#36;r_1&#36; To adjust.&#36;r_2&#36; In econometrics, it's called Granger's Causation.</p>
<p>Granger has a more detailed explanation for this: considering a binary sequence ahead of schedule.&#36;l&#36;Step prediction problems, using VAR models and one-dimensional models, respectively, to predict if&#36;r_{2t}&#36; The two-dollar projection is more accurate than his one-dollar projection.&#36;r_{1t}&#36; It's the Granger cause. Of course, it could be for the Grande cause.</p>
<p>We do not explain in detail the reasoning behind the prediction error, which is essentially the simplest MSE, and returns to the example before it.</p>
<p>When?&#36;\phi_{12}=0&#36;♪ Time, predict ♪&#36;r_2&#36; It's needed.&#36;r_1&#36; ♪ Information, so ♪&#36;r_1&#36; Yes.&#36;r_2&#36; The reasons for the Granger are the same as for the other. When the new interest item of the sequence&#36;a_t&#36; The alignment matrix is not the time of the diagonal, and the two sequences are synchronized, i.e., the transient grange causality.</p>
<p>The way it is, it's all about the way ahead.&#36;VAR(1)&#36; Research does not allow for the discovery of grange causality in a practical application from such a simple coefficient relationship. But that's enough to understand the model itself.</p>
<h4>Simplified structure of VAR</h4>
<p>In the model structure used earlier &#36;\Phi&#36; Reflects dynamic dependencies, and synchronized dependencies are used&#36;a_t&#36; . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . &#36;\Sigma&#36; . This form is usually called the simplified form of the VAR model, because it does not clearly show the synchronous linear dependency between the spectrospecies.</p>
<p>We can use matrix variations to express the synchronous dependencies in a visible way.&#36;a_t&#36; . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . &#36;\Sigma&#36;  Exists in Cholesky Disaggregated by &#36;\Sigma = LGL^T&#36;  of which&#36;G&#36;Yes.&#36;k&#36; The Quarters,&#36;L&#36;It's a lower triangle matrix with an angle of 1.
&#36;&#36;\bardsymbol{b}<em>t=\boldsymbol{L}^{-1}\boldsymbol{a}<em>t=(b</em>{1t},\ldots,b</em>{kt})^T&#36;&#36;
则有
&#36;&#36;\begin{aligned}E\boldsymbol{b}<em>t=&amp;0\\mathrm{Var}(\boldsymbol{b}<em>t)=&amp;\boldsymbol{L}^{-1}\mathrm{Var}(\boldsymbol{a}<em>t)L^{-T}=\boldsymbol{G}\end{aligned}&#36;&#36;
因此我们可以对原本的VAR模型进行同时左乘&#36;L^{-1}&#36; 得到
&#36;&#36;\begin{aligned}\boldsymbol{L}^{-1}\boldsymbol{r}</em>{t}=&amp;\boldsymbol{L}^{-1}\boldsymbol{\phi}</em>{0}+\boldsymbol{L}^{-1}\boldsymbol{\Phi}\boldsymbol{r}</em>{t-1}+\boldsymbol{L}^{-1}\boldsymbol{a}<em>{t}\=&amp;\boldsymbol{\phi}</em>{0}^{<em>}+\boldsymbol{\Phi}^{</em>}\boldsymbol{r}<em>{t-1}+\boldsymbol{b}</em>{t}\end{aligned}&#36;&#36;
他的最后一个子方程为
&#36;&#36;* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * &#36; &#36; &#36; * * * * * * * &#36; &#36; &#36; &#36; &#36; &#36; &#36; &#36;
Because&#36;b_{kt}&#36; Must be with&#36;b_{ki}&#36; So this equation is a direct reflection of the simultaneous dependency we call the structure equation.</p>
<p><strong>Simplified forms are commonly used in time series analysis</strong>, because</p>
<ul>
<li>Simplified forms are easier to estimate;</li>
<li>At the time of the projection, the synchronized form is not available;</li>
</ul>
<h4>Steady conditions and rectangularness</h4>
<p>We analyzed the problem with the stability of the AR model in the online time series analysis. <a href="/en/blog/2024/01/30/linear-time-series-analysis-notes/">Linear Time Series Analysis</a> The section on "Stable and random time series ARMA / Self-Regression Process (AR)" and uses the feature multi-form thinking to study its stability.</p>
<p>There are similar problems in VAR models, but they are still complex and are not presented here.</p>
<h4>VAR(p) model</h4>
<p>Here we expand the VAR(1) to VAR(p) model, we call it.&#36;k&#36;The time series obeys.&#36;VAR(p)&#36; When?
&#36;&#36;\bardsymbol{r}t=\bardsymbol{\pi}0+\bardsymbol{Phi}<em>1\boldsymbol{r}</em>{t-1}+\cdots+\boldsymbol{\Phi}<em>p\boldsymbol{r}</em>\bardsymbol{a}
The rules for all of these coefficients are not changed.</p>
<p>Yes.&#36;VAR(p)&#36; , coefficient of model&#36;\Phi&#36; It's a precursor to each weight, but it's complicated. </p>
<h3>VAR Modeling</h3>
<h4>Estimates and rankings</h4>
<p>VAR model modelling also follows largely the repeated trial processes of ranking, model estimation and model testing. One dollar of PACF can be extended to multiple situations to support rankings.</p>
<p>For a real data, we consider the next step progressive VAR model.
&#36;&#36;00\
&amp;\boldsymbol{r}_t= \boldsymbol{\phi}_0+\boldsymbol{\Phi}<em>1\boldsymbol{r}</em>{t-1}+\boldsymbol{a}_t  \
&amp;\boldsymbol{r}_t= \boldsymbol{\phi}_0+\boldsymbol{\Phi}<em>1\boldsymbol{r}</em>{t-1}+\boldsymbol{\Phi}<em>2\boldsymbol{r}</em>{t-2}+\boldsymbol{a}_t  \
&amp;\text{:} \
&amp;\boldsymbol{r}_t= \boldsymbol{\phi}_0+\boldsymbol{\Phi}<em>1\boldsymbol{r}</em>{t-1}+\cdots+\boldsymbol{\Phi}<em>p\boldsymbol{r}</em>{t-p}+wordsymbol{a} t
I'm sorry, I'm sorry.
Model parameters can be estimated using OLS (minimal 2 times) for each equation, i.e. multi-linear regression problems.</p>
<p>We're the first of them.&#36;i&#36;The equation is estimated to be the difference.
&#36; }<em>{t}^{(i)}=\boldsymbol{r}</em>{t}-\hat{\boldsymbol{\Phi&#125;&#125;<em>{1}^{(i)}\boldsymbol{r}</em>{t-1}-\cdots-\hat{\boldsymbol{\Phi&#125;&#125;<em>{i}^{(i)}\boldsymbol{r}</em>{t-i}&#36;&#36;
他的协方差矩阵为
&#36;&#36;\hat{\boldsymbol{\Sigma&#125;&#125;<em>i=\frac1{T-(k+1)i-1}\sum</em>^That(boldsymbol{a)}t&#36;
So we can go one by one.&#36;l&#36;Performance of hypothetical tests &#36;H 0:\bardsymbol<em>l=\mathbf{0}\leftrightarrow H_a:\boldsymbol{\Phi}<em>Other Organiser
Test statistics at &#36;M(1)=- (T-k-\frac{5} \ln\frac{hat{\bardsymbol{\Sigma}</em>{1}|}{|\hat{\boldsymbol{\Sigma&#125;&#125;</em>He was following the C.O. when the hypothesis was set</p>
<p>Or we could use information guidelines like AIC to determine that he needs to use the synoptic matrix of the much-approached margin in the form of
&#36; \tilde(boldsymbol}<em>{i}=\frac{1}{T}\sum</em>{t=i+1}^{T}\hat{\boldsymbol{a&#125;&#125;<em>{t}^{(i)}[\hat{\boldsymbol{a&#125;&#125;</em>You're not gonna get it.
The form of the definition of the volume of information is not presented here, and the selection results of these guidelines are not influenced by the matrix.</p>
<h4>Model testing</h4>
<p>Model differences can be calculated and multiple white noise tests (multiple mixing tests) are performed for the residuals. The multiple-compositing tests of the disability are reduced by using the estimated parameters.&#36;k^2p&#36;, this is the coefficient matrix &#36;\Phi _j, j= 1, 2, \ldots , p&#36;.</p>
<p>If some parameters in the coefficient matrix are fixed to zero, the freedom to be deducted should be calculated in the number of parameters without binding.</p>
<h4>Simplified Model</h4>
<p>When VAR is measured&#36;k&#36;When larger, the model has many parameters, and the number of parameters in the coefficient matrix is&#36;k^2p&#36;A few. If there is no a priori knowledge requirement that the parameters are not zero, the less significant parameters can be bound to zero and then estimated.</p>
<p>This is consistent with our one-dollar time-series model in practical applications, based on&#36;t=test&#36; And visualize to fix certain coefficients to zero.</p>
<h4>Grange's karma test.</h4>
<p>If the model can be simplified to include a coefficient equal to zero for some of the GL/R, then the GL/R test can be performed accordingly. In the binary VAR(1) model, if bound&#36;\phi_{12}(1)=0&#36;The post-module model does not differ significantly from the unbounded model, and&#36;r_{2t}&#36;No, it's not.&#36;r_{1t}&#36;The Granger cause.&#36;p&#36;Levels and&#36;k&#36;The dollar is similar.</p>
<p>To compare unbound and unbound models, the logarithmic test is used to test the statistically obtained amounts closer to the calculator distribution under the zero assumption that the binding parameters are equal to zero.</p>
<p>The test of Grange's causality is based on VAR, and the limitation is that the weights must be smooth, not support the co-ordinated model. So the co-ordination model can't use the function that's here.</p>
<h4>Projections</h4>
<p>If VAR&#36;p&#36;) Models are known, smooth conditions are met, set&#36;{a_t}&#36;It's a separate, stable time series. Use it.&#36;F_t&#36;Other Organiser&#36;t&#36;It's been so long.&#36;r_s,s\leq t&#36;, and then &#36;E (\bardsymbol{a}<em>t|F</em>{t-1})=0&#36;。基于&#36;t&#36;时刻的信息进行超前&#36;I&#36; 1 trot forecast, projected
&#36;&#36;\bardsymbol{r}<em>t(l)=E(\boldsymbol{r}</em>{t+l}|F_t)&#36;&#36;
当&#36;l=1&#36;时
&#36;&#36;\boldsymbol{r}_t(1)=\boldsymbol{\phi}_0+\boldsymbol{\Phi}_1\boldsymbol{r}_t+\cdots+\boldsymbol{\Phi}<em>p\boldsymbol{r}</em>{t+1-p}&#36;&#36;
当&#36;l=2&#36;时
&#36;&#36;\begin{aligned}\boldsymbol{r}<em>t(2)=&amp;E(\boldsymbol{r}</em>{t+2}|F_t)\=&amp;\boldsymbol{\phi}_0+\boldsymbol{\Phi}<em>1E(\boldsymbol{r}</em>{t+1}|F_t)+\boldsymbol{\Phi}<em>2\boldsymbol{r}<em>t+\cdots+\boldsymbol{\Phi}<em>p\boldsymbol{r}</em>{t+2-p}\end{aligned}&#36;&#36;
若记
&#36;&#36;\left.\boldsymbol{r}<em>t(l)=\left{\begin{array}{ll}E(\boldsymbol{r}</em>{t+l}|F_t),&amp;l&gt;0\\boldsymbol{r}</em>{t+l},&amp;l\leq0\end{array}\right.\right.&#36;&#36;
则超前&#36;l&#36;步预报可以写成
&#36;&#36;r_t(l)=E(r</em>{t+l}|F_t)=\boldsymbol{\phi}<em>0+\sum</em>I'm sorry, I'm sorry.
The visible advance multistep projection can be calculated incrementally.</p>
<p>For VAR that meets the condition of stability&#36;(p)&#36;Models, that can be proved.
&#36;&#36;\lim_{l\to\infty}\boldsymbol{r}_t(l)=\boldsymbol{\mu}=E\boldsymbol{r}_t&#36;&#36;That's what predicts the average regression.</p>
<p>The prediction error is easy to write.
&#36;&#36;\bardsymbol{e}<em>t(l)=\boldsymbol{r}</em>{t+l}-\boldsymbol{r}<em>t(l)=\boldsymbol{r}</em>{t+l}-E(\boldsymbol{r}_{t+l}|F_t)&#36;&#36;</p>
<h2>Accomplishment and vector error correction model</h2>
<h3>Fake return.</h3>
<p>Linear regression analysis is one of the most common models of statistics, but if the self-variant and the variable that returns is a time series, then the time series is the one that is the only one that can be used to make a statistical model. <strong>The return does not satisfy the basic assumption of the regression analysis: the model error item is distributed independently.</strong></p>
<p>When such a false return problem arises, the return may not be compatible or the standard error estimates and hypothetical tests that coincide with the return result are incorrect. We're here.<a href="/en/blog/2024/01/30/linear-time-series-analysis-notes/">Linear Time Series Analysis</a> The section on “Regression models with time series errors” describes a more common situation and treatment of false regressions.</p>
<p>We will continue to discuss the design of false returns more closely and to refine the relevant theory.</p>
<p><a href="/en/blog/2024/01/30/linear-time-series-analysis-notes/">Linear Time Series Analysis</a> The original sequence is given sufficient differentials to ensure its smoothness in the section on regression models containing time series errors so that the final error series must be in the form of time series, the latter part of this chapter <strong>Coordination and analysis</strong> It'll be a little more complicated.</p>
<h3>Coordination and analysis</h3>
<h4>Concept of coordination and analysis</h4>
<p>For the binary time series<em>t=(x</em>{1t},x_{2t})^T&#36;,如果&#36;x_{1t}&#36;和&#36;x_{2t}&#36;都是一元单位根过程，但存在非零线性组合&#36;\beta=(\beta_1,\beta_2)&#36;使得 &#36;z_t=\beta_1x_{1t}+\beta_2x_{2t}&#36;弱平稳，则称两个分量&#36;x_{1t}&#36;和&#36;x_{2t}&#36;存在协整关系(cointegration) , &#36;(\beta_1,\beta_2)^T&#36;称为&#36;x t&#36; integer. </p>
<p>Multiple multiple time series of points can similarly define the concretization relationship, and multiple concretization vectors can be present in multiple cases.</p>
<h4>The two-staged Engle and Granger method</h4>
<p>Want to look at multiple time series&#36;r_t&#36;The unit root test is required to confirm that both parts are unit root processes, and that there is no unit root after the difference, which is called "single-size"</p>
<p>Second, I'll...&#36;x_{1t}&#36;The blogger says:&#36;x_{2t}&#36;As a variable, as a linear regression, it gets disabled.&#36;e_t&#36;Sequences, and regression factor&#36;\beta_1&#36;, the equation is
&#36;&#36;x_{1t}=\beta_0+\beta_1x_{2t}+e_t&#36;&#36;</p>
<p>According to the study by Engle and Granger, the parameters are estimated to be the same at the time the conciliation relationship is formed, but the coefficient estimates are not normal, so the estimate is obtained using a linear minimum of two times the estimate.<strong>Point estimates are available, but the t and F tests in the result are not valid</strong>。</p>
<p>To verify that the association is established, only a unit root test of the returning defect is required, and when he does not have a unit root, we call the two points a combination. But because...&#36;e_t&#36; It's a return disability, so we need to use the Phillips-Ouliaris co-program.</p>
<p>The second stage of the two-stage Engle and Granger approach is the need to find all the condensed vectors in a number of situations, which requires the modification of the vector error model (VECM), which we will present later.</p>
<h4>VARMA Model</h4>
<p>Following the one dollar ARMA model, the VAR model can be extended to VARMA model in the form of
&#36;P(B)\bardsymbol{r}<em>t=Q(B)\boldsymbol{a}<em>t&#36;&#36;
其中
&#36;&#36;\begin{aligned}P(z)=&amp;\boldsymbol{I}-\boldsymbol{\Phi}</em>{1}z-\cdots-\boldsymbol{\Phi}</em>{p}z^{p}\Q(z)=&amp;\boldsymbol{I}+\boldsymbol{\Theta}<em>{1}z+\cdots+\boldsymbol{\Theta}</em>You're not gonna get a job.
VARMA has a problem with the same model that can be expressed as different parameter forms, so use VAR as much as possible to avoid VARMA.</p>
<h3>Error correction model and reconciliation</h3>
<h4>Error fixer model</h4>
<p>Because in the system, the number of units of non-stable weights is more than the number of units of root (the linear combination can reduce the amount of unit roots not even), so if the difference is calculated for each unit of non-stable weights, it is smooth, but it causes excessive differences.</p>
<p>This excess is a country differential that we will not see in a one-dollar model.</p>
<p>To correct this excess, we propose a vector error correction model.</p>
<p>For a VARMA model if it contains&#36;m&#36;A complication factor.&#36;m&#36; The following form of error correction (VECM)
&#36;&#36;&#36;&#36;&#36;\Delta\boldsymbol}<em>t=\boldsymbol{\alpha}\boldsymbol{\beta}^T\boldsymbol{x}</em>{t-1}+\sum_{j=1}^{p-1}\boldsymbol{\Phi}<em>j^*\Delta\boldsymbol{x}</em>{t-j}+\boldsymbol{a}<em>t+\sum</em>{j=1}^q\boldsymbol{\Theta}<em>j\boldsymbol{a}</em>&#36;
of which&#36;\boldsymbol{\alpha}&#36;and&#36;\beta&#36;Both.&#36;k\times m&#36;The ma is full of matrixes, no roots in the MA.&#36;m&#36;D-time series.&#36;\bardsymbol{y}<em>t=\boldsymbol{\beta}^T\boldsymbol{x}<em>t&#36;是平稳列(没有单位根),&#36;\boldsymbol{\beta}&#36;的每一列都是&#36;\boldsymbol{x}<em>t&#36;的一个协整系数。&#36;\boldsymbol{\Phi}<em>j^<em>&#36;和&#36;\boldsymbol{\alpha},\boldsymbol{\beta}&#36;都依赖于原来的AR部分的系数矩阵 &#36;\boldsymbol{\Phi}<em>j., in relation to:
&#36; \begin{aligned}\bardsymbol{\</em>{j}^{</em>}=&amp;-\sum</em>{i=j+1}^{p}\boldsymbol{\Phi}</em>{i},j=1,2,\ldots,p-1\\boldsymbol{\alpha}\boldsymbol{\beta}^{T}=&amp;\boldsymbol{\Phi}</em>{p}+\cdots+\boldsymbol{\Phi}</em>Other Organiser
coefficient&#36;\alpha,\beta&#36; It's not the only one.</p>
<h4>Related uses</h4>
<p>The VEPM model is determined using the maximum semblance estimate</p>
<p>The VECM model needs to be tested using Johansen's co-processing test, which is the essence of the test.&#36;\mathbf{\Pi}=\boldsymbol{\alpha}\boldsymbol{\beta}^T&#36; Yes.&#36;rank(\Pi)&#36; The test is the number of coordinated relationships.</p>
<p>The estimated VEPM model can be used for prediction. First, you can get it from the model.&#36;\Delta\boldsymbol{x}_t&#36;The prediction of the sequence, then from&#36;\Delta\boldsymbol{x}_t&#36;- I can solve it.&#36;\boldsymbol{x}_t&#36; . The difference between VEPM and VAR projections is that VEPM permits unit roots and unit roots, and VAR forecasts do not allow unit roots.</p>
<p><strong>VECM is the only time series model we've been learning to allow the root of the unit to exist, which is his unique place.</strong></p>
<h2>State spatial model</h2>
<h3>A brief introduction.</h3>
<p>State-spatial models are powerful, flexible and diverse models in the area of time series analysis, which, in conjunction with Kalman filtering techniques, can cover ARIMA models, many non-stable models with external variables, and many models with a different range of variables. <strong>More than before (includes)<a href="/en/blog/2024/01/30/linear-time-series-analysis-notes/">Linear Time Series Analysis</a>Linear time series models described in ) are more flexible</strong>。 </p>
<p>R Extension<strong>statespacer</strong>And many models based on linear Goss State spatial models have been achieved, and can be customised. The state space model is a relatively independent knowledge, but it's quite powerful, compared to<a href="/en/blog/2024/05/06/univariate-financial-time-series-analysis-notes/">Financial time series analysis (one dollar)</a> and the "Version self-repeat model" section <a href="/en/blog/2024/05/06/univariate-financial-time-series-analysis-notes/">Financial time series analysis (one dollar)</a> I'm sure the "Assessment and Vector Errors Model" section is more used than anything else.</p>
<p>As an introduction, we'll start with a local horizontal model. The model is simple, so it can be used to demonstrate the expression and estimate of the state space model. And then we're looking at the whole state space model.</p>
<h3>Local horizontal models</h3>
<p>Set&#36;{y_t,t=1,2,\ldots,T}&#36;For time series, meet the following models
&#36; \begin{aligned}y t}=&amp;\mu_t+e_t,:{e_t}\sim\mathrm{iid<del>N}(0,\sigma_e^2),:t=1,2,\ldots,n,\\mu_{t+1}=&amp;\mu_t+\eta_t,:{\eta_t}\sim\mathrm{iid</del>♪ I'm not gonna let you go ♪
of which&#36;{e_t}&#36;and&#36;{\eta_t}&#36;Independent, initial&#36;\mu_1&#36;For a given value or random variable subject to normal distribution, with &#36;{e t,\eta t,t&gt;0}&#36;相互独立。称&#36;{\mu_t}&#36;为&#36;{y_t}&#36;的水平，模型中&#36;{y_t}&#36;可观测而&#36;It's not visible.</p>
<p>This equation is a special case of a linear Gospel State spatial model. We can see from this model structure a similar place to the various time series structures studied earlier.</p>
<p>Of which&#36;{\mu_t}&#36;Called <strong>State equation</strong> &#36;{y_t}&#36; Called <strong>Observation equation</strong> &#36;{e_t}&#36;It's observational error. It's instantaneous error or noise.</p>
<p>This model is called a local horizontal model, and it is a special case of a structural time series model.</p>
<p>We can notice.
&#36;&#36;y_t-y_{t-1}=\eta_{t-1}+e_t-e_{t-1},&#36;&#36;
The difference between a first-order differential is the sum of random errors in the average value and a first-order lag, which means the original. &#36;y_t&#36; Obey. &#36;ARIMA(0,1,1)&#36; </p>
<p>Local horizontal models can handle multiple time series, but they are only processed in a single-dollar sequence, and there's nothing special about them.</p>
<h3>Filter, Smooth and Forecasting</h3>
<p>We continue to use local-level models as examples of the various analytical and modelling techniques of state-of-the-art spatial models, and filtering, smoothing and forecasting are the issues most frequently considered in the state-of-the-spatial models. The following is the text:</p>
<ul>
<li><strong>Filter</strong>: From&#36;{y_1,\ldots,y_t}&#36; estimate &#36;\mu_t&#36;</li>
<li><strong>Smooth</strong>: From&#36;{y_1,\ldots,y_n}&#36; estimate &#36;{\mu_1,...\mu_n}&#36;</li>
<li><strong>Forecast</strong>: From&#36;{y_1,\ldots,y_t}&#36; estimate &#36;\mu_{t+h}&#36;
So we can give it to you.</li>
<li>Filtering as&#36;E(\mu_t|y_{1:t});&#36;</li>
<li>Smoothly desolate as&#36;E(\mu_t|y_{1:n});&#36;</li>
<li>Forecast to&#36;E(\mu_{t+h}|y_{1:t})&#36;or&#36;E(y_{t+h}|y_{1:t})&#36;。</li>
</ul>
<p>For local horizontal models, set&#36;\mu_1\sim\mathbb{N}(a_1,P_1)&#36;and separates from the disturbance sequence. By the nature of the normal distribution, the local horizontal model is the Gaussian process, and the condition distribution is still the Gaussian distribution, so...&#36;\mu_t|y_{1:s}&#36;and&#36;y_t|y_{1:s}&#36;Relying on Goss distribution (multiple-normal distribution), the conditionality is expected to be a minimal-equivalence error estimate and a linear one.</p>
<p>&#36;\mu_t&#36;Yes.&#36;y_1:s&#36;The distribution of the conditions below is determined solely by expectations and differences in conditions. Remember&#36;\mu_t|s=E(\mu_t|y_{1:s})&#36;Remember&#36;\Sigma_t|s=Var(\mu_t|y_{1:s})&#36;。</p>
<p>Remember&#36;y_{t|s}=E(y_t|y_{1:s})&#36;I'm sorry. In particular, remember&#36;a_t=E(\mu_t|y_{1:t-1}),P_t=&#36;Var&#36;(\mu_t|y_{1:t-1})&#36;I'm sorry. By the nature of the distribution of Goss, the conditions are non-random. Remember
&#36;&#36;v_t=y_t-E(y_t|y_{1:t-1}),&#36;&#36;
That's right.&#36;y_t&#36;The error in making the best prediction, obviously.&#36;Ev_t=0&#36;You...
&#36;&#36;F_t=Ev_t^2=\mathrm{Var}(v_t),&#36;&#36;
The country is a country with a large number of different kinds of countries.&#36;v_t&#36;and&#36;y_{1:t-1}&#36;Independence, so there are.
&#36; \begin{aligned}F t}=&amp;\mathrm{Var}(v_t)=E(v_t^2)=E(v_t^2|y_{1:t-1})\=&amp;E[(y_t-E(y_t|y_{1:t-1}))^2|y_{1:t-1}]\=&amp;\mathrm{Var}(y_t|y_{1:t-1}).\end{aligned}&#36;&#36;</p>
<h3>Kalman filter.</h3>
<p>Kalman filtering is an infernal algorithm.&#36;t=1,2,\ldots&#36;based on&#36;\mu_t|y_{1:t-1}&#36;Distribution of conditions and newly available observations&#36;y_t&#36;Please.&#36;\mu_t|y_{1:t}&#36;Conditional distribution, which equals&#36;\mu_t|(y_{1:t-1},v_t)&#36;The distribution of conditions requires only expectations and differences in the distribution of conditions in Goss.</p>
<p>The blogger adds:&#36;\mu_t|y_{1:t-1}\sim\mathbb{N}(a_t,P_t),v_t\sim\mathbb{N}(0,F_t)&#36;and&#36;y_{1:t-1}&#36;Independence. Attention.&#36;{e_t}&#36;and&#36;{\eta_t}&#36;Independent, so...&#36;{e_t}&#36;and&#36;{\mu_t}&#36;Independent, with the expectation of a filter.
&#36;&#36;00\
&amp;\mu_{t|t}=E(\mu_t|y_{1:t}) \
&amp;= E(\mu_t|y_{1:t-1},v_t)  \
&amp;= E(\mu_t|y_{1:t-1})+E(\mu_t-E\mu_t|v_t)  \
&amp;= a_t+\frac{\mathrm{Cov}(\mu_t-E\mu_t,v_t)}{\mathrm{Var}(v_t)}v_t  \
&amp;= a_t+\frac{\mathrm{Cov}(\mu_t,v_t)}{F_t}v_t.
\end{aligned}&#36;&#36;
其中
&#36;&#36;\begin{aligned}
&amp;\mathrm{Cov}(\mu_t,v_t) \
&amp;=E(mu tv t)\quad(\text{Note}Ev t=0)\
&amp;= E[\mu_t(y_t-a_t)]  \
&amp;= E[\mu_t(\mu_t+e_t-a_t)]  \
&amp;= E[\mu_t(\mu_t-a_t)]+E[\mu_te_t]  \
&amp;= E[\mu_t(\mu_t-a_t)]+0  \
&amp;= E\left{E\left[(\mu_t-a_t)^2|y_{1:t-1}\right]\right}  \
&amp;= E{P_t}=P_t.
\end{aligned}&#36;&#36;
化简有
&#36;&#36;E(\mu_t|y_{1:t-1},v_t)=a_t+\frac{P_t}{F_t}v_t,&#36;&#36;
记
&#36;&#36;K_{t}=\frac{P_{t&#125;&#125;{F_{t&#125;&#125;=\frac{P_{t&#125;&#125;{P_{t}+\sigma_{e}^{2&#125;&#125;,&#36;&#36;
所以有滤波的条件期望为
&#36;&#36;(mu t 1 ,v t) =a t+K tv t, &#36;
That means we'll put &#36;y_1,...,y_t&#36; Yeah. &#36;\mu_t&#36; The best forecast (i.e. filtering) formula is broken down into two parts; the first part is:&#36;y_1,...,y_{t-1}&#36; Yeah. &#36;\mu_t&#36; The best forecast. The second part is a new pair.&#36;y_t&#36;The best forecast, the latter factor being the Kalman gain.&#36;K_t&#36;The best forecast is linear.</p>
<p>In practice, the Kalman filtering operation is carried out in a round, each of which is structured as follows, and in the cycle structure of the round, the entire filter sequence is obtained.
&#36;&#36;\left{aligned}v t=&amp;y_t-a_t,\F_t=&amp;P_t+\sigma_e^2,\K_t=&amp;P_t/F_t,\a_{t+1}=&amp;\mu_{t+1|t}=a_t+K_tv_t,\P_{t+1}=&amp;\Sigma t+1|t}P t(1-K t)+\sigma \eta^2,\mathrm{t=1,2,\ldots, n.\end{aligned}\right.&#36;
Arguments for initial distribution of algorithms &#36;a_{1}&#36; and &#36;P_1&#36; The selection has a significant impact on the entire Kalman filter, and we'll be able to introduce it separately.</p>
<h3>One step for errors</h3>
<p>When we were ahead of Kalman filtering, one step forecast was made and one step error was studied.
&#36;&#36;00\
&amp;v_{1}= y_1-a_1,  \
&amp;v_2= y_2-a_2=y_2-a_1-K_1(y_1-a_1),  \
&amp;v_3= y_3-a_3=y_3-a_1-K_2(y_2-a_1)-K_1(1-K_2)(y_1-a_1),
\end{aligned}&#36;&#36;</p>
<p>We can write it in matrix form, like
&#36;&#36;\bardsymbol{bardsymbol{(Y n a mathbf )<em>n)&#36;&#36;
其中
&#36;&#36;K=\begin{pmatrix}1&amp;0&amp;0&amp;\cdots&amp;0\k</em>{21}&amp;1&amp;0&amp;\cdots&amp;0\k_{31}&amp;k_{32}&amp;1&amp;\cdots&amp;0\\vdots&amp;\vdots&amp;\vdots&amp;\ddots&amp;\vdots\k_{n1}&amp;k_{n2}&amp;k_{n3}&amp;\cdots&amp;I'm sorry, I'm sorry.
We'll use it later.</p>
<h3>Smoothness of Status and Disturbation</h3>
<h4>Smooth state</h4>
<p>In the filter, we want to predict with the observations we have. &#36;\mu_t|y_{1:t}&#36;  When we get all the observations, &#36;{y_1,...y_n}&#36;  Using all observations to estimate &#36;\mu_t&#36;  That's what I get. &#36;\mu_t|Y_n&#36; This is called a smoothing problem.</p>
<p>We're going to give the local horizontal model's state smooth calculation method without proving it does.</p>
<p>To get what you want.<em>t=\mu</em>{t|n}&#36;,需要先进行卡尔曼滤波求出&#36;a_t,P_t,v_t,F_t,K_t,L_t&#36;,然后令&#36;r n=0&#36;, calculated by inverse inverse:
&#36; \begin{aligned}r=&amp;\frac{v_t}{F_t}+L_tr_t,\\hat{\mu}<em>{t}=&amp;\mu</em>{t|n}=a_t+P_tr_{t-1},:t=n,n-1,\ldots,2,1.\end{aligned}&#36;&#36;
同理，可以反向递推计算状态平滑方差有
&#36;&#36;\begin{aligned}
N_{t-1}=&amp; \frac1{F_t}+L_t^2N_t,  \
V_{t}=&amp; \Sigma_{t|n}=P_t-P_t^2N_{t-1},t=n,n-1,\ldots,2,1.
\end{aligned}&#36;&#36;</p>
<h4>Disturbing Smoothness</h4>
<p>We can also estimate the difference between smoothing and smoothing. &#36;e_{t},\eta_t&#36; The condition distribution, the problem is called disturbance smooth. He can use it to model, to find a mutation point in the state (a leap or change point for local horizontal models equivalent to a level), to find abnormal values for observed errors.</p>
<p>Remember
&#36;a }<em>t=E(e_t|y</em>{1:n}),\quad\hat{\eta}<em>t=E(\eta_t|y</em>{1:n}),\mathrm{~}t=1,2,\ldots,n.&#36;&#36;
因为 &#36;e_t=y_t-\mu_t&#36; 所以有
&#36;&#36;e_t\left|y_{1:n}\right.\sim\mathrm{N}(y_t-\mu_{t|n},\Sigma_{t|n})=\mathrm{N}(y_t-\hat{\mu}<em>t,V_t).&#36;&#36;
&#36;&#36;\hat{\eta}<em>t=E(\mu</em>{t+1}|y</em>{1:n})-E(\mu_t|y_{1:n})=\mu_{t+1|n}-\mu_{t|n}=\hat{\mu}_{t+1}-\hat{\mu}_t,&#36;&#36;</p>
<p>We can just give the formula there is.
&#36;&#36;\begin{gathered}
E(e_t|y_{1:n})= \sigma_e^2\left(F_t^{-1}v_t-K_tr_t\right), \
\mathrm{Var}(e_t|y_{1:n})= \sigma_e^2-\sigma_e^4\left(\frac1{F_t}+K_t^2N_t\right),
\end{gathered}&#36;&#36;
For the state equation disturbance,
&#36;&#36;00\
=&amp; \sigma_\eta^2r_t,  \
\mathrm{Var}(\eta_t|y_{1:n})=&amp; \sigma_\eta^2-\sigma_\eta^4N_t,\mathrm{~}t=n,n-1,\ldots,2,1.
\end{aligned}&#36;&#36;</p>
<h3>Processing and forecasting of missing values</h3>
<p>It is difficult for a general time series model to process missing values that appear within the time horizon. A major advantage of the state spatial model is that it is easier to observe missing values.</p>
<p>In Local Horizontal Models, set&#36;{y_t}_{t=\ell+1}^{\ell+h}&#36;Missing. The state space model can solve the problem of missing values in a number of ways, using a method that does not change the time step and model form.</p>
<p>Yeah. &#36;t\in{\ell+1,\ldots,\ell+h}&#36; We can give you the formula of a local horizontal model.
&#36;&#36;\mu_t=\mu_{t-1}+\eta_{t-1}=\cdots=\mu_{\ell+1}+\sum_{j=\ell+1}^{t-1}\eta_j,&#36;&#36;</p>
<p>We can give you filter structure as follows:
&#36;&#36;00\
♪ I'm not sure I'm gonna be able to do this ♪&amp; E(\mu_t|Y_\ell)=a_{\ell+1},  \
\mathrm{Var}(\mu_t|Y_{t-1})=&amp; \mathrm{Var}(\mu_t|Y_\ell)=P_{\ell+1}+(t-\ell-1)\sigma_\eta^2,
\end{aligned}&#36;&#36;
于是有递推式
&#36;&#36;\begin{aligned}a_t=&amp;\mu_{t|t-1}=\mu_{t-1|t-2}=a_{t-1},\P_t=&amp;♪ I'm not gonna let you go ♪ I'm sorry.
Which means that the Kalman filter we've been doing is still working, for the missing ones.&#36;y_t&#36; We should take the corresponding.&#36;v_{t}= 0&#36; At the same time&#36;K_{t}= 0&#36; That means no Kalman gain.</p>
<p>In fact, the prediction we're making is basically Kalman filter, and the results are the same as those for future values, which are given for missing direct filters.</p>
<h3>Selection of primary value distribution parameters and model parameter estimates</h3>
<h4>Selecting the parameters of the primary distribution</h4>
<p>Kalman filters need to be presumed to know. &#36;\mu_1\sim\operatorname{N}(a_1,P_1)&#36;  Actually, one of them.&#36;a_1,P_1&#36; It's all unknown.</p>
<p>Use filter formula
&#36;&#36;00\
♪ The world is so full of shit ♪&amp; y_1-a_1,\quad F_1=P_1+\sigma_e^2,  \
a_{2}=&amp; a_1+\frac{P_1}{F_1}v_1=a_1+\frac{P_1}{F_1}(y_1-a_1)  \
\rightarrow &amp; y_1\quad(P_1\to\infty),  \
P_{2}=&amp; P_1\left(1-\frac{P_1}{P_1+\sigma_e^2}\right)+\sigma_\eta^2  \
=&amp; \frac{P_1}{P_1+\sigma_e^2}\sigma_e^2+\sigma_\eta^2  \
\rightarrow &amp; The blogger says that the government is not a party to the law.
I'm sorry, I'm sorry.
And so...&#36;P_1\to\infty&#36;It's like a moment of thought.&#36;y_1&#36;is a non-random defined value, and&#36;\mu_1\sim\mathbb{N}(y_1,\sigma_e^2)&#36;I'm sorry. This initialization method is called proliferation initiation or diffusion a priori. The a priori proliferation is equivalent to a lack of knowledge of the initial state distribution.</p>
<h4>Model parameter estimates</h4>
<p>Filtering and smoothing are hypothetical model parameters.&#36;\sigma_e^2&#36;and&#36;\sigma_\eta^2&#36;Known. For the estimation of parameters, the maximum semblance method can be used, and the filter algorithm can be used to calculate the apparent function.</p>
<h3>State spatial model</h3>
<p>All of our previous presentations are related knowledge of local horizontal models, which are a simple exception to linear Gospel spatial models. This section gives a state spatial model, examples of other models that this model can represent, and gives filters, smoothing, forecasting formulas and parameter estimation methods.</p>
<p>- Take care of your reference. <a href="/en/blog/2024/05/04/r-time-series-analysis-learning-notes/">R TSA</a> The “State-Spatial Model” section of this section is very important for modelling in the form of model marks, which are followed by the State-Spatial Model.</p>
<p><strong>Many models can be presented as state-spatial models, but researching this expression is not very meaningful in applications.</strong></p>
<h4>Linear Goss State Space Model</h4>
<p>The state spatial model has many different expressions, according to the formula (Durbin and Koopman 2012), and the linear Goss model is: ^b0460f
&#36;&#36;00begin{gathered}
\boldsymbol{t=t\boldsymbol{\alpha}t+\boldsymbol{\varepsilon}t,\boldsymbol{\varepsilon}<em>t\sim\mathrm{N}(0,H_t), \
\boldsymbol{\alpha}</em>{t+1}= T_t\boldsymbol{\alpha}_t+R_t\boldsymbol{\eta}_t,\boldsymbol{\eta}_t\sim\mathrm{N}(0,Q_t),
\end{gathered}&#36;&#36;
其中
&#36;&#36;\alpha_1\sim\mathrm{N}(a_1,P_1).&#36;&#36;</p>
<p>of which&#36;y_t&#36;Yes.&#36;t&#36;The time value of the observation is&#36;p\times1&#36;Vector;&#36;\boldsymbol{\alpha}_t&#36;Yes.&#36;t&#36;The state of the time system is non-observable.&#36;m\times1&#36;Random vector, first equation called observation equation, second equation called state equation.</p>
<p>&#36;{\boldsymbol{\varepsilon}_t}&#36;and&#36;{\boldsymbol{\eta}_t}&#36;The two are independent and distributed white noise columns.&#36;\boldsymbol{\varepsilon}_t&#36;Yes&#36;p\times1&#36;Random vector,&#36;\boldsymbol{\eta}_t&#36;Yes&#36;r\times1&#36;Random vector,&#36;r\leq m&#36;。</p>
<p>Set Matrixs&#36;Z_t,T_t,R_t,H_t,Q_t&#36;The blogger says:&#36;Z_t&#36;and&#36;T_{t-1}&#36; Allows dependency on \\bardsymbol{y}<em>1,\ldots,\boldsymbol{y}</em>{t-1}&#36;,初始状态&#36;\boldsymbol{\alpha}_1&#36;服从&#36;N(\boldsymbol{a}_1,P_1)&#36;,设&#36;\boldsymbol{a}_1,P_1&#36;已知，&#36;\boldsymbol{\alpha}_1&#36;与&#36;{\boldsymbol{\varepsilon}_t}&#36;和&#36;I'm not gonna be able to get a chance to get a chance to get a chance to get a better chance.</p>
<p>Set when parameters are unknown&#36;\psi&#36;Unknown parameter, matrix&#36;Z_t,T_t,R_t,H_t,Q_t&#36;You can rely on unknown parameters.&#36;\psi&#36;。</p>
<p>in the Model&#36;R_t&#36;The government has been able to control the situation.&#36;r=m&#36;Some of the models of teaching materials are not available.&#36;R_t&#36;This one. Organisation&#36;R_t&#36;The good news is,&#36;R_t&#36;It's often a team formation.&#36;I_m&#36;♪ Some columns make up one ♪&#36;m\times r&#36;Matrix, called Selection Matrix, which allows for an error of zero in the equation for some state mass, and&#36;\boldsymbol{\eta}_t&#36;Square Range&#36;Q_t&#36;It could be full of stuff.&#36;r\times r&#36;Stand by, if not.&#36;R_t&#36;Matrix&#36;Q_t&#36;I'm not sure if I'm gonna be happy. If&#36;R_t&#36;It's normal.&#36;m\times r&#36;The matrix, most of the conclusions on the state spatial model, remain valid.</p>
<h4>Promulgated state spatial model</h4>
<p>This is a succession to the previous section, which extends the stateal Goss spatial model to the state equation, which is still linear Gossic, and the observation equation is distributed in non-Gas, or the observational equation is not linear in relation to the state variable, and it is also extended to the state equation in non-linear, and is distributed in non-Gs.</p>
<p>The generic non-linear, non-Gross State spatial model is in the form of a generic non-linear, non-Gross-state spatial model.
&#36;&#36;00\begin{aligned}\bardsymbol{y}<em>{t}\sim&amp;f</em>{t}(\boldsymbol{\alpha}<em>{t};\boldsymbol{\beta}),\\boldsymbol{\alpha}</em>{t+1}\sim&amp;g t}(\bardsymbol {\alpha};\bardsymbol{\theta},\end{aligned} I'm sorry.
Such models typically require random simulation methods such as MCMC, sequenced and important samples for filtering, smoothing and estimation.</p>
<h4>MASS package model</h4>
<p><strong>MARSS</strong>It's a more common R-state space model software package, and he has some agreement on the model format, which we need to understand to facilitate our use of the software package. MARSS is the abbreviation of multiple self-regression spatial models, which are, in fact, linear Gaussian spatial models.</p>
<p>The basic model formula is:
&#36;&#36;00\&amp;\boldsymbol{x}<em>{t}=B\boldsymbol{x}</em>{t-1}+\boldsymbol{u}+\boldsymbol{w}<em>{t},\quad&amp;\boldsymbol{w}</em>{t}\sim\mathrm{N}(0,Q),\&amp;\boldsymbol{y}<em>{t}=Z\boldsymbol{x}</em>{t}+\boldsymbol{a}+\boldsymbol{v}<em>{t},\quad&amp;\boldsymbol{v}</em>{t}\sim\mathrm{N}(0,R),\&amp;\bardsymbol0}sim\mathrm{\ (\bardsymbol{\pi},\Lambda).\end{aligned} I'm sorry.
Here and<a href="/en/blog/2024/05/06/univariate-financial-time-series-analysis-notes/">Financial time series analysis (one dollar)</a> The structure of the “Line Gosse State Space Model” section is essentially the same, with only a change of marking. The special feature is the matrix. &#36;B,Z,u,a&#36; Time change is allowed.</p>
<p> More complex models can also add a section on the impact of external variables to two equations. The parameterization and estimation methods of the MARSS extension package differ considerably from other state-of-the-art spatial model extension packages.</p>
<p>The regression part of the external variable and the models that can be written in each matrix allow for variations
&#36;&#36;00\&amp;\boldsymbol{x}<em>{t}=B</em>{t}\boldsymbol{x}<em>{t-1}+\boldsymbol{u}</em>{t}+C_{t}\boldsymbol{c}<em>{t}+\boldsymbol{w}</em>{t},\quad&amp;\boldsymbol{w}<em>{t}\sim\mathrm{N}(0,Q</em>{t}),\&amp;\boldsymbol{y}<em>{t}=Z</em>{t}\boldsymbol{x}<em>{t}+\boldsymbol{a}</em>{t}+D_{t}\boldsymbol{d}<em>{t}+\boldsymbol{v}</em>{t},\quad&amp;\boldsymbol{v}<em>{t}\sim\mathrm{N}(0,R</em>{t}),\&amp;\bardsymbol0}sim\mathrm{\ (\bardsymbol{\pi},\Lambda).\end{aligned} I'm sorry.
of which&#36;c_t&#36;It's in the equation.&#36;p&#36;External variable data, which can be entered into one&#36;p\times T&#36;Matrix;</p>
<p>&#36;C_t&#36;is the corresponding regression load matrix, which can contain unknown quantities, if it is non-temporal, if entered as&#36;m\times p&#36;matrices; if time changes, enter as&#36;m\times p\times T&#36;3-D array, with the last subscript for time&#36;t&#36;。</p>
<p>&#36;\boldsymbol d_t&#36;It's in the observation equation.&#36;q&#36;The data on the external variables,&#36;D_t&#36;is the corresponding load matrix.</p>
<p>This is not just a regression model, but also an internal state variable.&#36;x_t&#36;. A time series model with an external variable (return from the variable).</p>
<h2>The Herma Model HMM</h2>
<p>The Hemama model is similar to the state spatial model, but it follows a chain of horsees, usually in dispersive form. The model is also used extensively, for example, in biological research, model identification, financial modelling, etc.</p>
<ul>
<li>(<a href="https://www.math.pku.edu.cn/teachers/lidf/course/fts/ftsnotes/html/_ftsnotes/hmm.html#ref-Zucchini2016:HMM-TS-R">Zucchini, MacDonald, and Langrock 2016</a>): Hidden Markov Models for Time Series - An Introduction Using R. 2nd ed., 2016, CRC Press.</li>
</ul>
<h3>HMM Basic Introduction</h3>
<h4>Preparatory knowledge</h4>
<p>The observational values variable of the Hema model is subject to a simple, marginal distribution in a combination of separate distributions. Set&#36;\delta_1,\ldots,\delta_m&#36;It is a weighted average factor.&#36;p_j(x)&#36;,&#36;j=1,2,\ldots,m&#36;Yes.&#36;m&#36;Density (or probability mass function),
&#36;&#36;p(x)=\sum_{j=1}^m\delta_jp_j(x),&#36;&#36;
then&#36;p(x)&#36;is a density (or probability mass function) with a distribution called a stand-alone hybrid distribution or abbreviated distribution. Set&#36;X_j\sim p_j&#36;,&#36;X\sim p&#36;, then
&#36;&#36;E(X)=\sum_{j=1}^m\delta_jE(X_j).&#36;&#36;
and
&#36;&#36;E(X^k)=\sum_{j=1}^m\delta_jE(X_j^k).&#36;&#36;</p>
<p>We can use the description of the Marseilles. <a href="/en/blog/2023/03/18/stochastic-process-basics-notes/">Random Process Basis</a> The "Markov Chain of Dispersed Time" section </p>
<h4>The Consort of the Consorture</h4>
<p>Set&#36;{C_t}&#36;For the chain of horse,&#36;{X_t}&#36;For random processes,&#36;X_t&#36; Yes.&#36;X_1,\ldots,X_{t-1},C_1,\ldots,C_t&#36;The conditions are equal to&#36;X_t&#36;Yes.&#36;C_t&#36;, which is the&#36;{X_t}&#36;Follow the Hema model. In fact, the state space model is also the hide-and-map model, but the state equation in the state space model is not normally a discrete horse-map chain.</p>
<p>♪ If the chain is ♪&#36;{C_t}&#36;# The only state space #&#36;m&#36;a value, then the model is called&#36;m&#36;Status HMM. Other names of the Hema process, we see names that naturally come to mind.</p>
<p>Set&#36;p_i(x)&#36;Organisation&#36;X_t&#36;Yes.&#36;C_t=i&#36;under conditions, the probability mass function at discrete distribution and the probability density function at continuous distribution.</p>
<h4>The simple nature of the Cascade chain.</h4>
<p>For the distribution of a dollar, we can directly state:
&#36;&#36;P(X_t=x)=[\boldsymbol{u}(1)]^T\Gamma^{t-1}P(x)\mathbf{1}.&#36;&#36;</p>
<p>The distribution of the two dollars is
&#36;&#36;00\
&amp;P(X_t=v,X_{t+k}=w) \
&amp;= \sum_{i=1}^m\sum_{j=1}^mu_i(t)p_i(v)\gamma_{ij}(k)p_j(w)  \
&amp;= \boldsymbol{u}(t)^TP(v)\Gamma^kP(w)\mathbf{1}.
\end{aligned}&#36;&#36;</p>
<p>We can give you the nature of the rectangular.
&#36;&#36;00\
E (X t)&amp; =\sum_{i=1}^mu_i(t)E(X_t|C_t=i)  \
&amp;== sync, corrected by elderman == @elder man
I'm sorry, I'm sorry.</p>
<h4>The Cascade chain is a function.</h4>
<p>The observation sequence of the Henmar model is&#36;T&#36;Yeah, yeah.&#36;\boldsymbol{X}^{( t) }= ( X_1, \ldots , X_t) ^T&#36;, &#36;\boldsymbol{x}^{( t) }= ( x_1, \ldots , x_t) ^T。( x_1, \ldots , x_T)&#36; , i.e.&#36;P(\boldsymbol{X}^{(T)}=\boldsymbol{x}^{(T)})&#36;, needs to be&#36;P(X_1=x_1,\ldots,X_T=x_T,C_1=c_1,\ldots,C_T=c_T)&#36;Every one of them&#36;C_t&#36;Item about&#36;c_t&#36;Peace, yes.&#36;T&#36;And all that is to be reconciled.&#36;2T&#36;the product of the entry, so the apparent function of the surface is calculated to be&#36;O(Tm^T)&#36;,&#36;T&#36;It is not feasible to calculate at a larger time; however, there is a general measure of calculation in practice&#36;O(Tm^2)&#36;The algorithm.</p>
<p>We give the apparent function as a matrix.
&#36;&#36;L_T=\boldsymbol{\delta}^TP(x_1)\Gamma P(x_2)\Gamma P(x_3)\cdots\Gamma P(x_T)\mathbf{1}.&#36;&#36;
of which&#36;P(x)=\operatorname{diag}((p_1(x),\ldots,p_m(x))),p_j(x)=P(X_t=x|C_t=j)&#36;Not dependent.&#36;t&#36;value.</p>
<p>Maximum apparent estimation method</p>
<p><strong>It's enough to get a simple theory about what this section is about.</strong></p>
<h3>Observation values projections, status estimates</h3>
<p>The maximum semblance of the values after the given observations is estimated, and then the missing observations can be estimated, the predicted observations, the estimated marsal chain state, etc. This is based on the calculation of the condition distribution.</p>
<p>We don't want the chain to be smooth. &#36;\delta&#36; Yes. &#36;t=1&#36; Status&#36;C_1&#36; Distribution</p>
<h4>Conditional distribution of observations</h4>
<p>Remember&#36;\boldsymbol{x}^{(-t)}&#36;In the&#36;\boldsymbol{x}^{(t)}=(x_1,\ldots,x_T)^T&#36;Delete&#36;x_t&#36;.&#36;\boldsymbol{X}^{(-t)}&#36;Meanings similar. Consider&#36;X_t&#36;Yes.&#36;\boldsymbol{x}^{(-t)}&#36;condition distribution, which can be used to fill the missing value.</p>
<p>To calculate
&#36;&#36;P(X_t=x|\boldsymbol{X}^{(-t)}=\boldsymbol{x}^{(-t)})=\frac{P(X_t=x,\boldsymbol{X}^{(-t)}=\boldsymbol{x}^{(-t)})}{P(\boldsymbol{X}^{(-t)}=\boldsymbol{x}^{(-t)})},&#36;&#36;</p>
<p>And finally, the structure can be written.
&#36;&#36;P(X_t=x|\boldsymbol{X}^{(-t)}=\boldsymbol{x}^{(-t)})=\sum_{j=1}^mw_j(t)p_j(x),\mathrm{~}t=1,2,\ldots,T.&#36;&#36;
of which
&#36;&#36;w_j(t)=\frac{d_j(t)}{\sum_{k=1}^md_k(t)}.&#36;&#36;</p>
<h4>Projected distribution of observations</h4>
<p>&#36;&#36;\text{预测分布指条件概率}P(X_{T+h}=x|\boldsymbol{X}^{(T)}=\boldsymbol{x}^{(T)}),\quad\text{可以看成是}X_{T+1},\ldots,X_{T+h}\text{缺失情况下的计算。}&#36;&#36;
At this point,
&#36;&#36;P(X_{T+h}=x|\boldsymbol{X}^{(T)}=\boldsymbol{x}^{(T)})=\frac{P(\boldsymbol{X}^{(T)}=\boldsymbol{x}^{(T)},X_{T+h}=x)}{P(\boldsymbol{X}^{(T)}=\boldsymbol{x}^{(T)})},&#36;&#36;
The final projection distribution can be simplified into the form of a mixed distribution of the observed conditions below.</p>
<p>&#36;&#36;P(X_{T+h}=x|\boldsymbol{X}^{(T)}=\boldsymbol{x}^{(T)})=\sum_{j=1}^m\xi_j(h)p_j(x),&#36;&#36;</p>
<h4>Decoding</h4>
<p>Decoding is a state of restoration based on observations.&#36;C_t&#36; * The present document was not edited before being sent to the United Nations translation services. &#36;C_t&#36;</p>
<p>The way to decode it, we'll skip it.</p>
<h4>State forecast</h4>
<p>It can be shown that the state prediction can be decoded at an equal price.</p>
<h3>Model selection and diagnosis</h3>
<p>Increase in status&#36;m&#36;It changes the alignment, but it will.&#36;m^{2}&#36;Speed increases the number of parameters and there is a risk of over-composed. Some of the special models may streamline the status transfer matrix or condition distribution so that it relies only on a small number of parameters.</p>
<p>The AIC, BIC guidelines can be used to make different models. For the adequacy of model formulation, a false disability can be calculated and a residual diagnosis can be performed.</p>
<h4>Model selection using AIC, BIC</h4>
<p>Use of information volume guidelines to select models whose ideas need not continue to be repeated</p>
<h4>Modelled diagnosis using false disability</h4>
<p>After the model is prepared, the adequacy of the alignment needs to be assessed, and the anomalies identified, which are particularly poor. When modelling a normal linear regression model, a model can be used to diagnose the disease; in more general cases, a "false disability" or a "specific defect" can be defined to make model diagnosis.</p>
<p>Set&#36;X&#36;Subject to a continuous distribution, the distribution function is&#36;F(\cdot)&#36;, then&#36;U=F(X)&#36;Obey U(0,1) distribution. Random Variables&#36;X_t&#36;If the observation value is&#36;x_t&#36;, the distribution function is calculated under the presumed model
&#36;&#36;u_t=P(X_t\leq x_t)=F_{X_t}(x_t),&#36;&#36;
, and when the model is correct,&#36;u_t&#36;It should be subject to U(0,1) distribution.&#36;u_t&#36;Near zero or one. Because of the conversion of the different distributions to 0 and 1, the observations from the different distributions are comparable.</p>
<p>Set Data As&#36;x_1,\ldots,x_T&#36;, the model is&#36;X_t\sim F_t&#36;,&#36;x_t&#36;Yes.&#36;X_t&#36;And the observations, because of the different distributions, are these&#36;x_t&#36;It's not comparable. Calculate&#36;u_t=F_t(x_t)&#36;,&#36;u_1,\ldots,u_T&#36;For the sake of eveny, these are comparable. Yes, I can.&#36;u_1,\ldots,u_T&#36;The histogram and the QQQ graph, which are not evenly distributed, indicate that the model was wrongly set if there were significant differences with the balanced distribution performance.</p>
<p>The flatness margin is not easily used to identify the anomaly, and the 0.01 and 0.05 fractions are only 0.04 different, and are already very different in the normal distribution. Because we know the distribution of normal, we define the pseudo-psychological.
&#36;&#36;z_t=\Phi^{-1}(u_t)=\Phi^{-1}(F_t(x_t)),&#36;&#36;
It is easier to identify abnormalities. The normal-state falseness should be represented in the standard-normal distribution sample when the model is correct. The value of the normal false margin reflects the&#36;x_t&#36;The degree of deviation from the median (not the average) of its distribution. Hetograms, normal QQ Q Q Q Q Q Q Q Q Q Q Q Q Q Q Q Q Q Q Q Q Q Q Q Q Q Q Q Q Q Q Q Q Q Q Q Q Q Q Q Q Q Q Q Q Q Q Q Q Q Q Q Q Q Q Q Q Q Q Q Q Q Q Q Q Q Q Q Q Q Q Q Q Q Q Q Q Q Q Q Q Q   Q Q Q Q   Q Q Q   Q Q  Q  Q     Q  Q Q          Q  Q  Q   Q      Q                </p>
<p>The most important nature of the pseudo-disability is that it is distributed in a similar manner as the standard distribution (or the standard normal distribution), and it cannot be assumed that it is independent of each other, and that the falseness is not independent of each other.</p>
<h3>Concordant variables and other dependencies</h3>
<p>Impacts such as time trends, seasonal items, can be introduced into the model as non-random competitors. Consideration could also be given to the case of a hidden state model being a second or higher chain of horse. The presumption of conditionality could also be relaxed.</p>
<h4>HMM with competitive variables</h4>
<p>The probability of observing condition distribution parameters or a marzipan chain transfer can be operated on the basis of a competitor. This would still allow for the maximum semblance of estimates. The value of the competitor is known.</p>
<h4>HMM based on a second-grade chain</h4>
<h3>HMM in a continuum</h3>
<p>Status Number&#36;m&#36;Sometimes it's hard to choose objectively, when&#36;m&#36;Too many unknown parameters are available when it is large. So sometimes the Hema model of continuity may be more advantageous. This is very close to the state space model.</p>
<h3>The semi-hidden mast model.</h3>
<p>The state variable is sometimes not accurate by using a first-order chain. The shift to a high-strate marzipan increases the number of parameters. Another extension is the status process, which is half-map chain.</p>
<p>Set&#36;Y_t&#36;Is the status space as&#36;{1,\ldots,m}&#36;The Time Zima chain, the state transfer matrix.&#36;\Omega&#36;Angular elements are equal to zero. This is the way to make a difference in the state series when you are two times adjacent.</p>
<p>Set&#36;d_i&#36;It's a probability distribution on a positive integer set, yeah.&#36;i=1,2,\ldots,m&#36;Yes.&#36;m&#36;This is a distribution called the time-suspension distribution. From&#36;Y_t&#36;and&#36;{d_i}&#36;Construct Process&#36;{C_t}&#36;See below. Every one.&#36;Y_t&#36;Value represents several consistent states.&#36;C_s&#36;, and the number of constants, when&#36;Y_t=i&#36;Timeline Time Distribution&#36;d_i&#36;。</p>
<p>That's how it got here.&#36;{C_t}&#36;It is not usually a marina chain, known as the SMC. If&#36;{C_t}&#36;All&#36;d_i&#36;It's all geometrical, but it's still a marina chain.</p>
<p>The status of the Hema model&#36;C_t&#36;Replaced with a half-map chain, known as the hidden half-map model (HSMM). The hidden half-marticulation is much more complex than HMM. It is also difficult to introduce compost variables.</p>
<p>HMM can be used to expand the state space of HMM to be approximately any HSM.</p>
<h3>HMM for Vertical Data</h3>
<p>Existing&#36;K&#36;Every individual, every individual, on a continuous basis.&#36;T&#36;A point of time, and the observation is...&#36;{x_{tk},t=1,\ldots,T,i=1,\ldots,K}&#36;I'm sorry. This is called vertical data and economics is called panel (pane) data. It is noted that there is a correlation between multiple observations from the same individual. The same model is used for setting the time series for each individual, but the parameters can vary.</p>
<p>In some cases, it can be assumed.&#36;K&#36;Each sequence depends on a common potential status sequence.&#36;C_t&#36;, the conditions for independentness between the sequences after the given status sequence may be considered for HMM. One example is considering the return on a large number of equities, which are collectively affected by the same potential market position. This can be seen as a multidimensional HMM.</p>
<p>In some cases, it is not possible to assume that there are common state sequences, for example, multiple measurements at different time points for different patients. If it is assumed that the sequences, and the corresponding state, are independent, the function appears to be the product of the apparent functions of each series. If one assumes that some of the parameters in the observation sequence models are the same, the model can be estimated using data from all observation sequences together, increasing the accuracy of the estimate, and this combined approach can model models that cannot be estimated by individual modelling if the data are short.</p>
<p>Changes between individuals can be distinguished by the covariant value.</p>
