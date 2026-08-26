---
title: 'Advanced Linear Regression: Goodness of Fit, Model Selection, and Collinearity'
title_zh: 线性回归进阶：拟合优度、模型选择与共线性
date: 2025-09-23 12:02:07 +0800
categories:
- Data Science
- Statistical Modeling & Inference
tags:
- Statistics
- Regression
- Linear Models
author: Hyacehila
mathjax: true
hidden: true
excerpt: Covers goodness of fit, model selection, collinearity, heteroskedasticity, autocorrelation, robust standard errors,
  and related regression extensions.
description: Covers goodness of fit, model selection, collinearity, heteroskedasticity, autocorrelation, robust standard errors,
  and related regression extensions.
excerpt_zh: 整理拟合优度、模型选择、共线性、异方差、自相关、稳健标准误和相关回归扩展问题。
permalink: /blog/2025/09/23/advanced-linear-regression-notes/
lang: en
translation_key: 2025-09-23-advanced-linear-regression-notes
translation_status: machine
translation_source_hash: 54c08b94b75035c11d5bd5c6549254855f4a1fcc6a66c90b68d32872729decdd
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<h2>Evaluation of the regression equation</h2>
<h3>Proposed eugenicity indicator</h3>
<h4>RSS</h4>
<p>The difference of the model error item&#36;\sigma^{2}&#36; Reflects the size of the error and observation error of the model </p>
<p>Definitions:&#36;\hat{e}=y-x\hat{\beta}&#36;It's an estimate of the difference vector.
Definitions: &#36;RSS=e^{\Lambda}\acute{e}^{\Lambda}=\sum_{i\cdots1}^{n}e_{i}^{\Lambda2}&#36; It's the squared balance of the cripple.
RSS reflects the ability of models to aggregate data.</p>
<p>Because&#36;RSS&#36;The more you incorporate the variables, the smaller they are, the less we can use the RSS for comparison between the same selected models of the number of variables, and the less meaningful the rest (and not for comparison of RSS in multiple regression equations)</p>
<p>Theoretically:</p>
<ul>
<li>&#36;RSS=y^{\prime}(I-X(X^{\prime}X)^{-1}X^{\prime})y;&#36;</li>
<li>&#36;\hat{\sigma}^2=\frac{RSS}{n-p}&#36;Yes.&#36;\sigma^{2}&#36;A non-event estimate</li>
<li>&#36;RSS = y&#39;y-\hat{\beta}&#39;X&#39;y.&#36; is the difference squared and equals the total squared and the reduction of the squared and the reduction of the sum of the squared</li>
</ul>
<h4>&#36;R^{2}&#36;</h4>
<p>Definition: Returns squared &#36;SS_{\text{回&#125;&#125;=\hat{\beta}^{\prime}X_{c}^{\prime}y=y^{\prime}X_{c}(X_{c}^{\prime}X_{c})^{-1}X_{c}^{\prime}y&#36;
Definitions: Amending the total squared to &#36;SS_{\text{总&#125;&#125;=(y-\hat{\alpha}^{\wedge}\mathbf{1})^{\prime}(y-\hat{\alpha}^{\wedge}\mathbf{1})=(y-\bar{y}\mathbf{1})^{\prime}(y-\bar{y}\mathbf{1})&#36;
<strong>Attention, we need to go into advanced centralization here to start our research here, so all the tags are centralized tagging formats.</strong></p>
<p>Definitions: Decision coefficients&#36;R^{2}&#36;Yes
&#36;R^2=\frac(mathrm{SS}<em>{\cHFFFFFF}{\cH00FF00} What's wrong with you?</em>\text{total}, &#36;
Actually, we'll use it in front.&#36;RSS&#36;Explain the degree of model alignment.&#36;R^{2}&#36;It's the same thing. It's just...&#36;R^{2}&#36;The bigger the better.</p>
<p>I can see that. &#36;R^2&#36;The most important value is one, the closer one is the better the match, and of course he's not below zero.</p>
<p>There are amendments.&#36;R^2&#36; Because we're thinking about it.&#36;R^2&#36; Changes in (positive) sample capacity&#36;R^2&#36; We usually use both volumes in statistical analysis.</p>
<p><strong>In essence, it's not different from RSS, but it's comparable in multiple models, which is the advantage of RSS, and we're now almost off the balance and assessing the degree of alignment.</strong></p>
<p><strong>&#36;R^2&#36;No value applied when the regression equation does not have a cutout</strong></p>
<p>&#36;R^2&#36;Quantitatively describes the extent to which the variation of the explained variable is described or explained by the regression variable, so it is used frequently in applications&#36;R^2&#36; 。</p>
<p>But it's too dependent.&#36;R^2&#36; It could put people in a difficult position, and in practical application, "maximize"&#36;R^2&#36; There is little economic or statistical significance. On the contrary, whether or not to add a variable to the multiple regression depends on whether the inclusion of this variable will allow a better estimate of the causal effects of interest, i.e., the reference behind the reference.<a href="/en/blog/2023/09/04/linear-regression-basics-notes/">Linear regression base</a> and the “Selection of Model Indicators” section <a href="/en/blog/2023/09/04/linear-regression-basics-notes/">Linear regression base</a> and the “Information quantity indicators” section of the</p>
<h4>Return standard error SER</h4>
<p>Standard regression error (Standard Error of the Regresion, SER) is the standard error estimate for the regression, with the same error for SER and the regression result, so we also measure the degree of fragmentation of the regression error with SER;</p>
<p>We use the disability to estimate the SER formula as
&#36;&#36;SER=s_{\hat{u&#125;&#125;\text{,其中 }s_{\hat{u&#125;&#125;^2=\frac{1}{n-2}\sum_{i=1}^{n}\hat{u}_{i}^2=\frac{\mathrm{SS}R}{n-2}&#36;&#36;</p>
<p><strong>SER measured the size of the deviation from the regression line, i.e. the size of the regression error</strong> Too big for the SER. It's very bad.</p>
<h4>Coefficient S.E.</h4>
<p>For a sample of a whole number of times, each sample size is&#36;n&#36;So each sample has its own average, and the standard deviations for these averages are called standard errors.</p>
<p>For the regression coefficient, SE reflects the estimated effects of our coefficient in the same unit, and too large SE means that the coefficient estimate is not accurate and therefore meaningless.</p>
<p><strong>The standard error is itself a term, with different interpretations of coefficients and regressions, and the standard error is a difference in the standard estimate used to measure the effects of the estimate</strong></p>
<h3>Select Model Indicators</h3>
<p>We'll use it in this chapter.<strong>MSEP</strong>The average prediction error instead of the normal average error is the measure of the prediction, just to make a distinction, and there is no change in the actual definition.</p>
<h4>Full and selected models</h4>
<p>The whole model is the regression model we were thinking of.
&#36;&#36;y_{i}=\beta_{0}+\beta_{1}x_{i1}+\cdots+\beta_{p-1}x_{i,p-1}+e_{i}&#36;&#36;
And we're just choosing a part of the model that we're going to choose from to make a new model.
&#36;&#36;Y=\beta_0+\beta_1X_1+\cdots+\beta_{q-1}X_{q-1}+e&#36;&#36;
It's perfectly normal that when we switch from a whole model to a model, the variables associated with the prediction deviations change, and we need to focus on the average E, the difference Var, the average projection error MSP.
Same as the RSS that was used to evaluate errors.
&#36;&#36;MSEP(\hat{y})=E(y-\hat{y})^2=E(\hat{e}^2)&#36;&#36;
We can do the full model and the selection of the equation error.</p>
<ul>
<li>Even if the whole model is correct, the chosen model can get smaller MASS and smaller regression factors, but it's cost-effective.</li>
<li>Even if the whole model is correct, the selection of models sometimes increases the accuracy of the prediction, often because a regression factor is difficult to estimate accurately.
So, it's good for us to have a few of the self-variant elements that are not very active or difficult to observe in the regression equation, so we're going to analyze the different sub-variant sets.<strong>Select the best subset</strong>It's what we do behind us.
Now we're going to start looking at the criteria for evaluating the regression equation.</li>
</ul>
<h4>&#36;RMS_q&#36;Guidelines</h4>
<p>&#36;RMS_q&#36;The idea of the code is to use the difference squared and RSS because RSS reflects the actual data and theoretical predictions of deviations, and even smaller RSS means a better alignment.</p>
<p>We'll take the model RSS as&#36;RSS_q&#36;
Based on the formula, here are two models.&#36;RSS_q&#36;
&#36;&#36;RSS_q=y^{\prime}(I-X_q(X_q^{\prime}X_q)^{-1}X_q^{\prime})y&#36;&#36;
&#36;&#36;RSS_{q+1}=y^{\prime}(I-X_{q+1}(X_{q+1}^{\prime}X_{q+1})^{-1}X_{q+1}^{\prime})y&#36;&#36;
Directly deviating can be done.
&#36;&#36;RSS_{q+1}\leq RSS_q&#36;&#36;
The more the model expands, the better? This is clearly contrary to the point of our selection model, so we're used to introducing a penalty factor.
&#36;&#36;RMS_q\overset{}{\operatorname*{=&#125;&#125;\frac1{n-q}\overset{}{\operatorname*{RSS&#125;&#125;_q&#36;&#36;
I can do it now.&#36;RMS_q&#36;The smallest rule is to choose the model we want.
It's called average disability squared and standard. </p>
<h4>&#36;C_p&#36;Guidelines</h4>
<p>&#36;C_p&#36; Guidelines are considered in terms of the accuracy of the projection of the variables
&#36;&#36;MSEP(\hat{y})=E(\hat{y}-y)^2=Var(x_q^{\prime}\hat{\varphi}_q)+(E\hat{e})^2&#36;&#36;
It's the statistics we want to use.
&#36;&#36;C_p=\frac{RSS_q}{\hat{\sigma}^2}-(n-2q)&#36;&#36;
&#36;C_p&#36;  The smaller the model, the more worthwhile it is to choose.</p>
<h3>Indicators of volume of information</h3>
<h4>&#36;AIC&#36;Guidelines</h4>
<p>We'll give it straight here.&#36;AIC&#36;He's based on the most obvious model code.
&#36;&#36;AIC=-2\ln(\text{模型似然度})+2(\text{模型自由参数个数})&#36;&#36;
Akaike, a Japanese statistician, proposed a code for the volume of information of Akaike.
Get the smallest one.&#36;AIC&#36;And that set of parameters is the best, and he's using very broad statistical guidelines, and a lot of areas are using them to study best models.
For selecting models
&#36;&#36;Y=\beta_0+\beta_1X_1+\cdots+\beta_{q-1}X_{q-1}+e,~e\sim N(0,\sigma^2I)&#36;&#36;
The function appears to be
&#36;&#36;L(\varphi_q,\sigma_q^2|y)=(2\pi\sigma_q^2)^{-\frac n2}\exp\left{-\frac1{2\sigma_q^2}\sum_{i=1}^n(y_i-\sum_{j=0}^{q-1}\beta_jx_{ij})^2\right}&#36;&#36;
The logarithmic function is
&#36;US&#36; (\,\x\^2|)=-\frac{((22(2)=((2)=((2piv)=(1})=(})=(})=(})= (=)= (= \ ^ ^ ^)= (= = ^ ^)= (=)= (= = = ^ ^ ^ ^)= (= } })= (= = } } })= (1} } } } } } } })= (1} } } } } })= (} } } } } })&#39;(y-X_q\varphi_q)&#36;&#36;
代入
&#36;&#36;\begin{aligned}\hat{\varphi}<em>q&amp;=(X_q^{\prime}X_q)^{-1}X_qy\\hat{\sigma}<em>q^2&amp;=\frac{RSS_q}n\end{aligned}&#36;&#36;
得到
&#36;&#36;\ln L(\hat{\varphi}</em>{q},\hat{\sigma}</em>{q}^{2}\left|y\right)=\left[-\frac{n}{2}+\frac{n}{2}\ln(\frac{n}{2\pi})\right]-\frac{n}{2}\ln(RSS_{q})&#36;&#36;
因此
&#36;&#36;AIC = n\n(RSS {)+2q&#36;
We'll use this directly from now on.&#36;AIC&#36;The formula is fine.
Akaike's information we haven't learned, so it's hard to understand his mind.</p>
<h4>BIC Guidelines</h4>
<p>Bates Information Guidelines
Calculate formulae as
&#36;&#36;\mathrm{BIC}=-2\cdot\ln(\hat{L})+k\cdot\ln(n)&#36;&#36;
We don't have much of a presentation here. We'll do it later.</p>
<h2>Assumptions test</h2>
<p>The purpose of this chapter is to study the problem of prediction of variables in the linear statistical model, or at the core of the method of the lowest two-fold estimation, and then we'll give you a little bit of a little bit of the prediction of variables in the linear statistical model, which is the common way of organizing it.
We're still studying the lowest-based two-fold estimate of the experience that we've got back to the equation.
In fact, given the speciality of the hypothetical test, it's difficult to do the research in other cases.
The main issues addressed in this chapter are as follows:</p>
<ul>
<li>Re-entry equations for re-entry.</li>
<li>Visibility test for regression coefficients</li>
<li>Anomalous Point Test
The hypothesis test is introduced because the experience that has been gained is not really the relationship between the equation and the variable.</li>
</ul>
<h3>General linear assumptions</h3>
<h4>Basic thinking</h4>
<p>We're starting with our core linear regression model.
&#36;&#36;y=X\beta+e,e\sim N(0,\sigma^2I)&#36;&#36;
of which&#36;X&#36;Yes.&#36;n\times p&#36;The matrix, which is...&#36;n&#36;Sub-observation &#36;p&#36;Parameters
For general linear assumptions, we add linear assumptions for parameters to be estimated
&#36;&#36;\boldsymbol{H}{:}\boldsymbol{A\beta}=\boldsymbol{b}&#36;&#36;
It's a minimum 2-fold estimate.
of which&#36;A&#36;Yes.&#36;m\times p&#36;. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . &#36;m&#36;The number of bound equations
We can easily give the balance of the difference before we increase the constraint.
&#36;&#36;RSS=(y-A\hat{\beta})^{\prime}(y-A\hat{\beta})=y^{\prime}(I-X(X^{\prime}X)^{-1}X^{\prime})y&#36;&#36;
And of course we can give the balance of the post-restriction balance.
&#36;&#36;\hat{\beta}_{H}=\hat{\beta}-(X^{\prime}X)^{-1}A^{\prime}(A(X^{\prime}X)^{-1}A^{\prime})^{-1}(A\hat{\beta}-b)&#36;&#36;
&#36;&#36;RSS_H=(y-A\hat{\beta}_H)^{\prime}(y-A\hat{\beta}<em>H).&#36;&#36;
增加了约束条件后 非常明显的会增大残差的平方和  也就是
&#36;&#36;RSS</em>& US&#36;US&#36;
If the parameters of the model do satisfy the constraints, then the ordinary and limited amount of the post-conditional disability should be increased, so when the squares of the residual increase are increased to a certain degree, we can reject the assumption.</p>
<h4>Core Theorem</h4>
<p>For normal linear regression models&#36;y=X\beta+e,e\sim N(0,\sigma^2I)&#36; Yes.</p>
<ul>
<li>&#36;{RSS}/\sigma^{2}\sim\chi_{n-p}^{2}&#36;</li>
<li>Assumptions&#36;{A\beta}={b}&#36; &#36;(RSS {)<em>H}-RSS</em>{}^{})/\sigma^2   \sim \chi_{_m}^2;&#36;</li>
<li>&#36;RSS&#36;and&#36;RSS-RSS_H&#36; Independence</li>
<li>Assumptions&#36;{A\beta}={b}&#36; &#36;F_{<em>H}=\frac{(RSS</em>{<em>H}-RSS)/m}{RSS/(n-p)}\sim F</em>{_{m,n-p&#125;&#125;&#36;</li>
</ul>
<h4>Brief form</h4>
<p>We gave a simple calculation of RSS, which is the difference squared and equals the total squared and the reduction of the reduction of the squared.
&#36;RSS = y&#39;y-\hat{\beta}&#39;X&#39;y&#36;&#36;
使用类似的手法还是可以计算&#36;RSS_H&#36; 带入前面的&#36;F_H&#36;有
&#36;&#36;F_{H}=\frac{(A\hat{\boldsymbol{\beta&#125;&#125;-b)^{\prime}(A(X^{\prime}X)^{-1}A^{\prime})^{-1}(A\hat{\boldsymbol{\beta&#125;&#125;-b)/m}{RSS/(n-p)}&#36;&#36;</p>
<h4>Quit Fields Confirm</h4>
<p>The basic idea before us is that when the balance is squared and increased to a certain degree, we can reject the assumption.
And that's what we're testing for, you know, the numbers are so big that we can just say no to the assumptions.
So we should choose a single-side hypothetical test and reject the field selection as
&#36;F H}&gt;\mathbf{F}_{m,n-p}(\alpha)&#36;&#36;</p>
<h4>Concluding remarks</h4>
<p>Why do you call it a general linear hypothesis?
It's because many of the assumptions that follow are converted to some particular case of this general linear hypothesis, and then we apply the formula here to solve it.
For example:
&#36; \begin{aligned}y 1&amp;=\beta_1+e_1\y_2&amp;=2\beta_1-\beta_2+e_2\y_3&amp;=\beta_1+2\beta_2+e_3\end{aligned}&#36;&#36;
检验
&#36;&#36;{H:\beta 1=beta 2} &#36;
You can get a shape change. &#36;\beta_1-\beta_2=0&#36;
Which means...&#36;n=3,p=2,m=1&#36;General linear assumptions</p>
<p>Or his promotion.
&#36; \begin{aligned}y 1&amp;=X_1\beta_1+e_1,e_1\sim N(\theta,\sigma^2I_{\mathrm{n}<em>1})\y_2&amp;=X_2\beta_2+e_2,e_2\sim N(\theta,\sigma^2I</em>{\mathrm{n}<em>2})\end{aligned}&#36;&#36;
检验
&#36;&#36;\boldsymbol{H}:\boldsymbol{\beta}<em>1=\boldsymbol{\beta}<em>2&#36;&#36;
能看出参数这里等价于
&#36;&#36;I</em>{p}\beta</em>{1}-I</em>=&#36;0
So this is...&#36;n=n_1+n_2;p=2p;m=p&#36;General linear assumptions</p>
<h3>Re-entry equations for re-entry.</h3>
<h4>Basic thinking and testing statistics</h4>
<p>What are we trying to prove? It is certainly proof that the regression equation is really significant; so based on the idea of the hypothetical test, our original assumption is that all the regression coefficients are zero.&#36;\beta_0&#36;It's constants that don't add zero to the test here.
&#36;H:\beta scdots=beta sc<em>{p-1&#125;&#125;=0&#36;&#36;
如果我们同意了原假设 就意味着所有自变量的影响都不重要
此时我们的约束可以看作如下的形式
&#36;&#36;A=(0,I</em>{p-1}),b=0\quad(A\beta=\mathbf{b})&#36;&#36;
这就化简成我们上一节研究的一般线性假设问题了
通过一系列化简运算我们得到原本的检验&#36;F&#36;统计量变形为如下
&#36;&#36;F \text{back}=\frac{SS text{back}/(p-1)}
We can judge whether we reject the original hypothesis by formula.</p>
<h4>Analysis of the perspective of the equation</h4>
<p>Under the original assumptions of this section,
&#36;RSS <em>H}=y^{\prime}y-\beta</em>{0}^{^{\star&#125;&#125;\mathbf{1}^{\prime}y=\sum_{i=1}^{\mathrm{n&#125;&#125;(y_{i}-\overline{y})^{2}&#36;&#36;
也就是在本节中
&#36;&#36;RSS_H=TSS&#36;&#36;
又
&#36;&#36;TSS=RSS+SS \text{
So this section's test statistics can be considered as
The test of statistics compares the two parts when the return to squares and the greater error of the test, rejecting the original hypothesis that the return equation is significant.</p>
<p>If we choose to accept the original hypothesis,
This means that the error in the model is the one that's different from the one that's different.&#36;Y&#36;The impact can be ignored.
The possibility is as follows:</p>
<ul>
<li>The model is very different, and it's possible that some of the variables have been missing, some of the variables that have returned to the variables are not linear.</li>
<li>Back to the variable is true.&#36;Y&#36;The impact is small.</li>
</ul>
<h3>Visibility test for regression coefficients</h3>
<p>We can determine the re-entry equation's re-entry profile.&#36;Y&#36;It depends on a range of self-variant variables that we have chosen, but it is not possible to rule out that some of the self-variant variables are not actually being used.&#36;Y&#36;Dependency, that's hypothetical.
&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;&#36;US&#36;&#36;US&#36;US&#36;US&#36;US&#36;&#36;US&#36;US&#36;US&#36;&#36;US&#36;&#36;US&#36;&#36;US&#36;US&#36;US&#36;&#36;US&#36;US&#36;US&#36;US&#36;&#36;US&#36;&#36;US&#36;US&#36;&#36;US&#36;&#36;US&#36;&#36;US&#36;&#36;US&#36;&#36;&#36;&#36;&#36;US&#36;&#36;&#36;US&#36;&#36;&#36;US&#36;&#36;&#36;&#36;US&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;...&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;...&#36;&#36;&#36;&#36;&#36;...&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;...&#36;&#36;&#36;&#36;&#36;...<em>&#123;&#123;i&#125;&#125;:{\beta}</em>Other Organiser
Here we're going to look at the hypothetical test of this hypothesis.</p>
<h4>Basic thinking</h4>
<p>We can still choose to see this as a special case of a general linear hypothesis.
&#36;&#36;\beta_i=0\Leftrightarrow A\beta=\mathbf{0},A=(0,...,0,1,0,..,0)&#36;&#36;</p>
<p>Then bring in the formula we're using in the first section.
&#36;&#36;F_{H}=\frac{(A\hat{\boldsymbol{\beta&#125;&#125;-b)^{\prime}(A(X^{\prime}X)^{-1}A^{\prime})^{-1}(A\hat{\boldsymbol{\beta&#125;&#125;-b)/m}{RSS/(n-p)}&#36;&#36;</p>
<h4>Core thinking</h4>
<p>The calculation is still too complicated. Let's do something simple; look at the obvious test of regression coefficients in a different perspective.
We know the minimum 2x2 estimate of the unknown parameter is satisfied.
&#36;&#36;\hat{\beta}\sim N(\beta,\sigma^2(X^{\prime}X)^{-1}).&#36;&#36;
<em>When the disability is simple, you can turn back to the nature.</em>
If I remember
&#36;&#36;C_{p\times p}=(c_{ij})=(X^{\prime}X)^{-1}&#36;&#36;
Then it is.
What's that?<em>i\sim N(\beta_i,\sigma^2c</em>{ii}).&#36;&#36;
因此当假设&#36;H_i:\beta_i=0&#36;成立的时候有
&#36;&#36;\frac{\hat{\beta}<em>i}{\sigma{\sqrt{c</em>{ii&#125;&#125;&#125;&#125;={N}(0,1)&#36;&#36;
又因为我们知道
&#36;&#36;{RSS}/\sigma^{2}\sim\chi_{n-p}^{2}&#36;&#36;
因此有
&#36;&#36;t_i=\frac{\hat{\beta}<em>i}{\hat{\sigma}\sqrt{c</em>{ii&#125;&#125;}\sim t_{n-p}&#36;&#36;
其中&#36;\hat{\sigma}^2=RSS/(n-p)&#36;
非常自然的 我们应该使用双侧拒绝域的&#36;t&#36;检验 选择的拒绝域为
&#36;&#36;|t_i|\geq t_{n-p}(\alpha/2).&#36;&#36;</p>
<h4>Special analysis</h4>
<p>We know.
&#36; \\mathrm{Var} (hat(beta})<em>i)=\sigma^2c</em>{ii}&#36;&#36;
所以可以给出标准误差
&#36;&#36;I'm sorry, I'm sorry, but I'm sorry, but I'm sorry, but I'm sorry, but...
All you have to do is change the estimate.&#36;\hat{\sigma}&#36;
That's ours.&#36;t&#36;The test is...
<strong>The lowest 2x2 estimate and the difference between the standard error</strong>
The very natural thought is that if you're a variable,&#36;X&#36;For regression variables&#36;Y&#36;Without significant influence, you can be detached from the regression equation. And then you can use the rest of the regression from the variable to re-entry, and that's the updated coefficient, and that's the idea of chapter 5 to re-entry from the variable to the selection of the variable.</p>
<h3>Anomalous Point Test</h3>
<p>The sum of the anomaly is the point of departure from the main data, and this section is the one where we study how to study and judge the anomaly.
Here we use the idea of a return equation to study anomalies, which is to use the handicapped mind.
<strong>Use the size of the difference in the estimated value of the entire model to determine whether this point is abnormal</strong>
The idea of the anomaly test in pre-processing of data is different.</p>
<h4>Mean drift linear regression model</h4>
<p>The normal linear regression model is as follows:
&#36;&#36;y_{i}=x_{i}^{\prime}\beta+e_{i},e_{i}\sim N(0,\sigma^{2})&#36;&#36;
If one of the data is abnormal, he's bound to be off the equation, which is,
&#36;&#36;y_{j}=x_{j}^{\prime}\beta+\eta+e_{j}&#36;&#36;
His matrix is...
&#36;&#36;y=X_i\beta+d_j^{\prime}\eta+e&#36;&#36;
We call this model the mean drift linear regression model.
Directly gives an estimate of the minimum two-fold of the average drift linear regression model.
&#36;&#36;&#36;200,000<em>}=\hat{\beta}_{(j)},\eta^{</em>}=\frac1{1-h_{jj&#125;&#125;\hat{e}<em>&#36;
Where's the £2500beta?</em>{(j)}&#36; 为从原回归模型中剔除第&#36;Minimum 2x2 estimate for j&#36; group data
&#36;h_{jj}&#36;Yes.  &#36;H=X(X^{\prime}X)^{-1}X^{\prime}&#36; No. No.&#36;j&#36;Diagonal
&#36;e_j&#36;It's the first one from the original regression model.&#36;j&#36;A cripple.</p>
<h4>Test statistics</h4>
<p>We're dropping the complicated extrapolation and giving the test statistics directly.
If you're assuming&#36;H:\eta=0&#36;  There is.
&#36;&#36;F_j=\frac{(n-p-1)r_j^2}{n-p-r_j^2}\sim F_{1,n-p-1}&#36;&#36;
Of which&#36;r_j&#36;The student disability we want to test.
&#36;n&#36;It's the number of observations. &#36;p&#36;is the number of the variable that returns to the variable.
For random&#36;\alpha&#36; We use a single-side hypothetical test, which means we reject the field as
&#36;F j=frac{(n-p-1)r n-p-r i}&gt;F_{<em>{1,n-p-1&#125;&#125;(\alpha)&#36;&#36;
这是因为我们非常自然的思想 学生化残差越大越偏离越应该拒绝
因此我们选择单侧假设检验
由于&#36;t&#36;分布和&#36;F&#36;分布的关系 可以给出等价的检验统计量（双侧的）
&#36;&#36;t_j=\left(F_j\right)^{1/2}&#36;&#36;
拒绝域为
&#36;&#36;\left|t_i\right|\geq t</em>{n-p-1}(\alpha^{}/2).&#36;&#36;</p>
<h4>Some notes</h4>
<p>The only way we can handle this is by one anomaly.
But the number of anomalies in reality is uncertain, and if there are too few, then there are some points that are not suspected and wrongly introduced, and if there are too many assumptions, there are no normal points.
The anomaly test is a lead, and there's a lot to follow.</p>
<h3>Projections due to variables</h3>
<p>The prediction of regression models is a very simple question.
Just like we did in Bayesian statistics, put him in the back of the hypothetical test for a simple introduction.</p>
<h4>Simple theoretical presentation and point prediction</h4>
<p>In theory, we have two predictions.
One is the average of projections.
&#36;&#36;Ey_0=x_0^{\prime}\beta &#36;&#36;
It's not necessary to consider random errors at this point.
The other is the projection.
&#36;&#36;\hat{y}_0=x_0^{\prime}\hat{\beta}&#36;&#36;
Unfortunately, we can't know what the error is, and we give the specifics of our predictions based on the principle of non-selectivity, which is the actual application we're using.
They're the same form, but they mean different things.
We'll come to a conclusion that the projection is the lowest-short-linear no-speculation.</p>
<h4>Inter-sector projections</h4>
<p>Give theorem.
&#36;&#36;\hat{y}_0-y_0\sim N(0,\sigma^2(1+x_0^{\prime}(X^{\prime}X)^{-1}x_0))&#36;&#36;
He's only got a random error subject to a uniform estimate of zero.
So export the core amount
&#36;&#36;&#36;\frac(hat)<em>0-y_0}{\hat{\sigma}\sqrt{1+x_0^{\prime}(X^{\prime}X)^{-1}x_0&#125;&#125;\sim t</em>{n-p}&#36;&#36;
所以对于给定的显著性水平 可以给出预测区间
&#36;&#36;\left[\hat{y}<em>0-t</em>{n-p}(\frac\alpha2)\hat{\sigma}\sqrt{1+x_0^{\prime}(X^{\prime}X)^{-1}x_0},\hat{y}<em>0+t</em>\n-p\ (\frac\alpha2) \sqrt{1+x 0{\({\){\(x^)\right]&#36;
It's actually a situation that's estimated in mathematical statistics.</p>
<h2>Selecting the regression equation</h2>
<p>There should be some sort of problem with the original equation of return.<strong>Linear testing of regression models</strong> And we're not looking at this in online statistics, and we're defaulting on linear models, just taking the choice of retrogressive variables.</p>
<p>In chapter three, we're going to describe the gradual regression of MSEs in the context of optimizing the convergence of linear variables by removing them, and we're going to be going to tell you how to judge and eliminate the convergence of linear variables, and eventually achieve a gradual return equation, and we're going to need to use an entire chapter to describe the problem, and we need to look at a number of small questions to give us the final picture.<strong>Progressive return approach</strong></p>
<h3>Calculate all possible returns</h3>
<p>For one of them, there is.&#36;p-1&#36;A linear regression model for a variable, a subset of a variable, and a factor.&#36;y&#36;It's a linear regression model, so we actually get it.&#36;2^{p-1}-1&#36;Linear regression model
We're looking for a reasonable order and method of calculation, and controlling the increase in the number of calculating indices and controlling errors is what we're looking at in this section.</p>
<h4>&#36;S_p&#36;Sequence Method</h4>
<p>Let's put&#36;p&#36;A sub-assemble of variables&#36;p&#36;Vector&#36;(u_1,u_2,...,u_p)&#36;To indicate which&#36;u_i=0\text{或}1&#36; Which means the subset does not contain this variable.
Now calculate all possible returns from where they started:
Along the edge of the Govt-Step, through every vertex without repetition.
I can see that.&#36;S_p&#36;The sequence is characterized by:
Each regression subset happens once, and one variable is different from the next.
Let's go down to the next one.&#36;S_p&#36;The method of construction of the sequence
First of all, we'll define it as follows: &#36;S_p&#36;Sequence &#36;i&#36;This is the introduction of variables&#36;x_i&#36;   &#36;-i&#36;Means the removal of variables&#36;x_i&#36;   That's...
&#36;&#36;S_{2}={1,2,-1}&#36;&#36;
Which means that the first calculation is only a variable.&#36;x_1&#36;And then the second time, it was introduced.&#36;x_2&#36; Third Excerpt&#36;x_1&#36; Just do it.&#36;x_2&#36;♪ And the return ♪
Here's what we'll do if we start with zero.&#36;S_p&#36;Sequence
First definition &#36;T_i&#36;Yes.&#36;S_i&#36;Rewind and change the sequence of the symbol, which is
&#36;S_{2}={1,2,-1}&#36;   &#36;T_{2}={1,-2,-1}&#36;
Now we give you the iterative rule.
&#36;S_1 ={1}&#36;  &#36;S_2={S_{1}<del>2</del>T_{1&#125;&#125;&#36; &#36;S 3}S 2}, 3, T 2}
Just push it like that.</p>
<h4>Matrix to Change</h4>
<p>To calculate all possible returns, we need to give an algorithm that works.
For the square.
&#36;&#36;A_{n^*n}=(a_{ij})&#36;&#36;
Introduce a new Square&#36;B=(b_{ij})&#36; Remember
&#36;&#36;\begin{cases}b_{ii}=1/a_{ii}\b_{ij}=a_{ij}/a_{ii},i\neq j,j=1,...,n\b_{ji}=-a_{ji}/a_{ii},j\neq i,j=1,...,n\b_{kl}=a_{kl}-a_{il}a_{ki}/a_{ii},k\neq i,l\neq i\end{cases}&#36;&#36;
It's called&#36;a_{ii}&#36;It's a central axial change.&#36;B=T_iA&#36;
We give the following nature for the matrix's elimination and transformation.</p>
<ul>
<li>&#36;T_iT_iA=A&#36;</li>
<li>&#36;T_iT_jA=T_jT_iA&#36;</li>
<li>&#36;A=\begin{bmatrix}A_{11}&amp;A_{12}\A_{21}&amp;A_{22}\end{bmatrix}&#36;  为&#36;q&#36;阶方阵 &#36;T_{1}T_{2}...T_{q}=\begin{bmatrix}A_{11}^{-1}&amp;A_{11}^{-1}A_{12}\-A_{21}A_{11}^{-1}&amp;A_{22}-A_{21}A_{11}^{-1}A_{12}\end{bmatrix}&#36;</li>
</ul>
<p>Now we're going to use the matrix's digestive transformation to study linear regression models.
&#36;&#36;y=X\beta+e,E(e)=0,Co\nu(e)=c^2I&#36;&#36;
Put&#36;X&#36;- Yes, I do.&#36;&#36;X=(X_q,X_r)\quad\beta=(\varphi_q^\prime,\varphi_r^\prime)&#36;&#36;Remember
&#36;B=X&#39;X=\begin{pmatrix}X&#39;<em>q\X&#39;<em>q\end{pmatrix}(X_q,X_r)=\begin{pmatrix}X&#39;X_q&amp;X&#39;X_r\X_r&#39;X_q&amp;X&#39;X_r\end{pmatrix}=\begin{pmatrix}B</em>{11}&amp;B</em>{12}\B_{21}&amp;B_{22}\end{pmatrix}&#36;&#36;
则有
&#36;&#36;A=\begin{pmatrix}B&amp;X^{\prime}y\y^{\prime}X&amp;y^{\prime}y\end{pmatrix}=\begin{pmatrix}B_{11}&amp;B_{12}&amp;X_q^{\prime}y\B_{21}&amp;B_{22}&amp;X_r^{\prime}y\y^{\prime}X_q&amp;y^{\prime}X_r&amp;y^{\prime}y\end{pmatrix}&#36;&#36;
那么就有
&#36;&#36;\left.T_{1}T_{2}\cdots T_{q}\boldsymbol{A}=\left(\begin{array}{ccc}{\boldsymbol{B}<em>{11}^{-1&#125;&#125;&amp;{\star}&amp;&#123;&#123;\boldsymbol{B}</em>{11}^{-1}\boldsymbol{X}<em>{q}^{\prime}\boldsymbol{y&#125;&#125;}\{\star}&amp;{\star}&amp;{\star}\{\star}&amp;{\star}&amp;&#123;&#123;\boldsymbol{y}^{\prime}\boldsymbol{y}-\boldsymbol{y}^{\prime}\boldsymbol{X}</em>{q}\boldsymbol{B}<em>{11}^{-1}\boldsymbol{X}</em>{\bord0\shad0\alphaH3D}The \blur0}showing of the \blur0}A new \blur0 \blur0 \blur0 \blur0 \blur0 \blur0 \blur0 \blur0 \blur0 \blur0 \blur0 \blur0 \blur30 \blur0 \blur30 \blur0 \blur0 \blur0 \blur30 \bl30 \bl30 \bl30 \bl30 \bl30 \bl30 \bl30 \bl30 \bl30 \bl30 \bl30 \bl30 \bl30 \bl30 \bl30 \bl30 \bl \ \ \bl30 \bl30 \bl30 \bl30 \bl30 \bl \ \ \ frfr \ \ \ fr \ fr fr \ \ \ fr \ \
I can see that.
&#36;B_{11}^{-1}X_{q}^{\prime}\mathbf{y}&#36; The minimum two-fold estimate for the model selection.
&#36;y^{\prime}y-y^{\prime}X_{q}B_{11}^{-1}X_{q}^{\prime}y&#36; The difference between the squares of the model.
So we can see it.
Calculates the new minimum 2x2 estimate variable for the matrix elimination conversion problem
If you want to introduce new variables, you can do a new matrix to change it.
To remove a variable, you just have to do the distillation that you did when you introduced him again.
So that each time a new regression model is calculated, it's the result of the last transformation.
This iterative method is important for calculating the selection model.</p>
<h3>Calculates the best subset returns</h3>
<p>The very natural idea is that we should study together with the first two sections.
One section has selected all the sub-sets for us, and the other section gives us the best way to assess them.
I'll take both.</p>
<h4>Return to the curve of the variable subset</h4>
<p>For each of the components&#36;q-1&#36;Calculate statistics from a subset of variables&#36;U_{q-1}&#36;Value
And what this value is assessing is that under the criteria used, the good and bad of this subset is actually assessing a bunch of the ones that are included in the calculations we've been making.&#36;q-1&#36;A selected model of the variable
If&#36;U_{q-1}&#36; The smaller the better, the smallest subset is called a tatter one.
If&#36;U_{q-1}&#36; The bigger the better, the bigger the one, the bigger the one.
Now the Zen reflects the good and bad of this subset under the same criteria as we used.
We'll make all the tarts one.&#36;U_{q-1}&#36; Make a picture of it.&#36;U_{q-1}&#36; Figure</p>
<h4>Practical</h4>
<p>The way we actually do it in practice is the way we think about it.
It's just a combination of what I've been talking about.&#36;U_{q-1}&#36; Figure
For every evaluation, we can give them our own.&#36;U_{q-1}&#36; Figure
Then we can compare them manually to multiple models.&#36;U_{q-1}&#36; I'm gonna figure out the best subset I can think of.
It's actually a subjective Voting. </p>
<h3>Progressive return</h3>
<p>It's certainly good to calculate the best subsets back, but when the number of variables is quite large (generally 10 or more), although the matrix is reduced to a large amount of cost, the way backsup is too large to calculate the cost, so we're proposing a way to do not count all the subsets back, and gradual return is the most common of applications.
The most excellent sub-regression also cannot address the issue of conjunctive linearity, and gradual regression can address some degree of concentrism, but in a highly linear situation, gradual regression cannot directly affect the high degree of correlation of the variable, and therefore not very well.</p>
<h4>The idea of gradual return</h4>
<p>Introduce a variable one by one, provided that its partial re-square and (PRSS) are tested to be significant and then the old variable is tested one by one, and the less significant variable is excluded.
The effect of the last result is that all selected variables are significant and that variables that are not in the regression equation are not tested significant.</p>
<h4>Back to Square</h4>
<p>The deviation is the square and measures the degree of interpretation of the variation of the variable from a given variable. In the multiple regression model, the degree of interpretation of variability due to the variable after controlling other variables ' impact
For the original model
&#36;&#36;{y=X_{q+1}\varphi_{q+1}+e}&#36;&#36;
Model after removing a variable
&#36;&#36;{y=X_q\varphi_q+e}&#36;&#36;
Back to Square&#36;PRSS=RSS_q-RSS_{q+1}=\hat{\beta}_q^2(x_q^{\prime}N_qx_q)&#36; </p>
<h4>Remove from Variables</h4>
<p>Is there anything to be removed from the existing model?
We use the retrogressive coefficients in the hypothetical section.
&#36;&#36;{H_\text{剔}{ : \beta _ q }=0}&#36;&#36;
Then it's available.
&#36;&#36;&#36;&#36;&#36;500{\<em>i}{\hat{\sigma}\sqrt{c</em>\sim  n-p}&#36;
Of course he can be used.&#36;F&#36;Form of the test</p>
<p>In fact, the statistical test of this hypothesis is in the form of an equivalent price, which is based on a partial regression and extrapolation.
&#36;&#36;\left.F \text{q)=\left(\begin{matrix}n-q-dond{matrix}\right.\right)\frac(hat{beta} (\bardsymbol{x}<em>q^{\prime}N_q\boldsymbol{x}<em>q)}{R\mathrm{SS}</em>I'm sorry, I'm sorry.
In general, it's a habit.
We'll calculate all the variables for &#36;F.</em>\text{clip} Found the smallest of them, and if the smallest is significant enough (greater than the threshold), then the cut is over and can't continue. Except...</p>
<h4>Introduction of variables from the</h4>
<p>Or exactly the same idea. Let's do the test, see if we can assume.
&#36;&#36;US&#36;US&#36;&#36; text{ \bardsymbol{\beta}<em>q=\boldsymbol{0}&#36;&#36;
是否会被拒绝
我们还是使用刚才导出的那个检验统计量
&#36;&#36;\left.F</em>\text{quote} =left(\begin{matrix}n-q-1\ \ \matrix}\frac(hat}{\bardsymbol&#123;&#123;\prime}<em>q)}{R\mathrm{SS}</em>I'm sorry, I'm sorry.
Or, as we did just now, we're going to calculate all the variables that were not introduced and get a series of tests.</p>
<p>The square and the square must be significantly reduced, otherwise the negatives are not possible.&#36;F&#36;Test
<strong>The molecule is a squared and form.</strong>
This time, the biggest of these, if the largest is less than the selected prominence, is that all variables are not significant enough, or else a variable is selected, and the rest of them are redoing the process.</p>
<h4>Other gradual thinking</h4>
<h5>Forward.</h5>
<p>I've been introduced to the point where I know that all the amounts are tested.</p>
<h5>Backwards</h5>
<p>Put all the variables directly into the regression model and then pick them up one by one. Except for self-variant that contributes less to RSS
Until there's no amount to be removed.</p>
<h2>And then we'll talk about the regression hypothesis and the heretic.</h2>
<p>The previous equation of return begins with the assumption that Gauss Markov, where we explain how we got this assumption, is not actually the form of our original scenario of return.</p>
<h3>Basic assumptions for the regression equation</h3>
<p>A regression problem, we want to predict some variable characteristics from some variables, which we're looking at in the form of.<a href="/en/blog/2023/09/04/linear-regression-basics-notes/">Linear regression base</a> The "Line Regression Model" section of the paper is described.</p>
<p>So we give four basic returns assumptions that if they are not, then the return cannot be achieved.</p>
<ul>
<li>When x is given, the condition of the disability is zero, which means we ask that the disability is irrelevant to the variable.
<strong>If the disability is associated with the variable, it means it is not random, that is, it does not capture the characteristics that need to be used for prediction.</strong></li>
<li>Sample i.i.d (time series data generally do not satisfy i.i,d)
<strong>This is the basic assumption for statistical estimates.</strong></li>
<li>No great anomalies.
<strong>OLS is not a roust, so be careful with the abnormalities.</strong></li>
<li>No complete junctive
<strong>We need to decompose the matrix when we solve the OLS, and if we're fully communicative, we can't solve the coefficient.</strong></li>
</ul>
<p>It can be seen that the assumptions here do not meet the requirements of the Gass Markov hypothesis, and the difference is not equal at this point.</p>
<p>Especially if we go against the first hypothesis, which is internal.<a href="/en/blog/2023/09/04/linear-regression-basics-notes/">Linear regression base</a>The various methods of regression analysis described in the report are problematic; we need to consider <a href="/en/blog/2024/05/24/generalized-linear-regression-notes/">Broad linear regression</a> "Regression of fixed effects" section <a href="/en/blog/2024/05/24/generalized-linear-regression-notes/">Broad linear regression</a> ..to analyze the tool variable returns section</p>
<p>If we go against the assumption of the same difference, Gauss Markov, we need to consider the question of the difference between the same and the different. Section</p>
<p>If we do not comply with the second assumption of the sample i.i.d, it will lead to time-series analysis and the creation of self-regression models.<a href="/en/blog/2024/01/30/linear-time-series-analysis-notes/">Linear Time Series Analysis</a> <a href="/en/blog/2024/05/24/generalized-linear-regression-notes/">Broad linear regression</a> "Return models with disabilities" section</p>
<p>If we break the third hypothesis, we're gonna have to. <a href="/en/blog/2024/05/24/generalized-linear-regression-notes/">Broad linear regression</a> "Stand-up Robust Return Technology" section</p>
<p>If we do not comply with the fourth hypothesis, then we will be able to draw up the present "Estimation of regression parameters (under cominglinear)" Section</p>
<h3>Same difference and differentiating</h3>
<p>I'm giving you a hand. &#36;X&#36; , About&#36;e&#36;The only assumption for distribution is that the average is zero (the first minimum two-fold scenario) and goes further if the difference in the distribution of the condition is not dependent on&#36;X&#36;, the error is described as the same.</p>
<p>This chapter deals with the theoretical meaning of the difference, the theoretical risk of which we have been assessing in the first place based on the hypothesis of the difference.</p>
<p>If for anything&#36;i=1,2,...,n&#36;I'll give it to you.&#36;X_i&#36;Time&#36;u_i&#36;Difference in the condition distribution &#36;var(u_i|X_i=x)&#36;As Constant
And not dependent.&#36;x&#36;, the error item is referred to&#36;u_i&#36;is the same difference; otherwise, the error is described as an alien difference.</p>
<p>The difference is met by the Guass-Markov hypothesis.</p>
<ul>
<li>OLS estimates are neutral and similar to normal, and this is also true for the heretic model.</li>
<li>The validity of the OIS estimate when the error is equal is the BLUE.</li>
<li>The same difference applies to the difference formula, which is the formula for the difference in the estimated amount.
<strong>Especially, the standard SE formula we're applying is a staggered and robust equation, so the results of the software are directly trusted, whether or not they are.</strong></li>
</ul>
<h3>Heterogeneity and Application</h3>
<p>In practical models, there are few models that can determine the difference; therefore, we use as much as possible the various analytical methods that are robust in the differential.</p>
<p>Unfortunately, the variance analysis is completely unsettled, so we need to be extra careful when we do the differential analysis. <a href="/en/blog/2024/02/29/experimental-design-methods-notes/">Pilot design methodology</a></p>
<p>The existence of a homogenous margin requires a specific analysis of the problem and, where it is not possible to judge, the use of an exotic model is possible. There are special GLS models when the variance is known. <a href="/en/blog/2023/09/04/linear-regression-basics-notes/">Linear regression base</a> Or we're considering using the above-mentioned standard error of the heterogeneity.</p>
<h3>Alien Test</h3>
<p>We know that the heretic model is problematic, and we need to know when there is an heretic effect, so we can take note of it. </p>
<p>When retrospect diagnosis is performed, visual graphics are available to observe the problem of the heterogeneity.</p>
<p>The formalization test is:
<strong>Breush-Pagan (BP) test</strong>  <strong>White test</strong></p>
<h2>Reverting the essence of analysis</h2>
<h3>What's the essence?</h3>
<p>All regression analyses want to study the dependency between random variables, and we know that, of course,<strong>There's no connection between dependency and causation.</strong>The causal inference is a separate course in statistics.</p>
<p>Description by probability distribution, i.e. known&#36;X=x&#36;I'm not sure if you're going to be able to do this.&#36;Y&#36;Probability density of conditions&#36;f_Y|X(y|x)&#36;It describes&#36;X&#36;How much to decide?&#36;Y&#36;But we are more concerned with their expectations than with their probability density:&#36;E( Y| X= x)&#36; This condition is a random variable.&#36;Y&#36;Yeah.&#36;X&#36;.</p>
<p>Why are we concerned with expectations because of:<strong>Because under the principle of average error, the function of the average error is the expectation of condition.</strong> Which means...
&#36;&#36;g(X)=E(Y|X)\text{ 是 }\underset{g\in\mathbb{F&#125;&#125;{\operatorname*{\arg\min&#125;&#125;E[Y-g(X)]^2\text{ 的最优解}&#36;&#36;</p>
<p>In fact, there are different optimal decomposition functions under different criteria, and they are also widely applied in other regression models: For example, the best solution under the MAE code is the median, and the best solution under the Logit return is the largest entropy rule.&#36;Sigmoid&#36;Functions</p>
<p>So if we use MAE as an indicator to judge the fit, then the essence of regression analysis is that it uses all known parameter models.&#36;\mathbb{F}&#36;Yeah. &#36;E( Y| X)&#36;Approximate. And...<strong>We usually use linear functions as&#36;E(Y|X)&#36;And this is the approximate, this is the linear regression model.</strong></p>
<h3>Proof of theorem (certification of the best predictor)</h3>
<p>The title is set by &#36;X = (X 1, \dots, X p)&#39;&#36; 为 &#36;p&#36; 维随机向量，&#36;Y&#36; 为随机变量。对任意函数 &#36;f: \mathbb{R}^p \to \mathbb{R}&#36;，有： &#36;&#36;E(Y - E(Y|X))^2 \leq E(Y - f(X))^2&#36;&#36; 即条件期望 &#36;E(Y|X)&#36; 是 &#36;Y&#36; is the best predictor (under the mean of average error). </p>
<p>Proof: By expanding squared items and using the nature of the expectations: </p>
<p>Split the margin value to &#36;Y - f(X)&#36; Split &#36;(Y - E(Y|X)) + (E(Y|X) - f(X))&#36;, and: &#36;&#36;E(Y - f(X))^2 = E\left[(Y - E(Y|X)) + (E(Y|X) - f(X))\right]^2&#36;&#36;
Expand squared and spread formulae according to squared &#36;(a+b)^2 = a^2 + b^2 + 2ab&#36;- It's okay. &#36;&#36;= E(Y - E(Y|X))^2 + E(E(Y|X) - f(X))^2 + 2E\left[(Y - E(Y|X))(E(Y|X) - f(X))\right]&#36;&#36;
Deal with cross-cutting items, use expectations&quot;The Code of Rectitude Expectations&quot;(Law of Iran Projects), extracting the conditions of the cross-section: &#36;&#36;E\left[(Y - E(Y|X))(E(Y|X) - f(X))\right] = E\left[ E\left[(Y - E(Y|X))(E(Y|X) - f(X)) \mid X\right] \right]&#36;&#36;
Because &#36;E(Y|X) - f(X)&#36; It's about... &#36;X&#36; function (recognised as &#36;g(X)&#36;) It may be suggested from the expectations of the conditions: &#36;&#36;= E\left[ (E(Y|X) - f(X)) \cdot E\left[Y - E(Y|X) \mid X\right] \right]&#36;&#36;
Imminent conditions for the implementation of the &#36;E(Y|X)&#36; It's given. &#36;X&#36; Down &#36;Y&#36; , and therefore: &#36;&#36;E\left[Y - E(Y|X) \mid X\right] = E(Y|X) - E(Y|X) = 0&#36;&#36; The cross item after the sub-item becomes: &#36;&#36;E\left[ (E(Y|X) - f(X)) \cdot 0 \right] = 0&#36;&#36;
Merge result. Cross-sections are zero-generation. &#36;&#36;E(Y - f(X))^2 = E(Y - E(Y|X))^2 + E(E(Y|X) - f(X))^2&#36;&#36; Because &#36;E(E(Y|X) - f(X))^2 \geq 0&#36;(non-negative of squares), so that: &#36;&#36;E(Y - f(X))^2 \geq E(Y - E(Y|X))^2&#36;&#36;
The equals condition is the right and only the right. &#36;E(E(Y|X) - f(X))^2 = 0&#36; Time, the equal sign is set. Since the square item is not negative, the equivalent is: &#36;&#36;E(Y|X) - f(X) = 0 \quad \text{几乎必然（almost surely）}&#36;&#36;That's... &#36;f(X) = E(Y|X)&#36;</p>
<p><strong>Conclusion Expectations &#36;E(Y|X)&#36; In the sense of minimizing the average error, yes &#36;Y&#36; The best predictor. Any other basis &#36;X&#36; , and then select the projection function for the &#36;f(X)&#36; It's gonna cause a bigger average error unless... &#36;f(X)&#36; and &#36;E(Y|X)&#36; Almost everywhere.</strong></p>
<h3>Optimistic linear predictor</h3>
<p>Set Random Vector &#36;\begin{pmatrix} Y \ X \end{pmatrix}&#36; The expectations are:
&#36;&#36;E\begin{pmatrix} Y \ X \end{pmatrix} = \begin{pmatrix} \mu_y \ \mu_x \end{pmatrix}&#36;&#36;
The matrix of the agreement is:
&#36;&#36;D\begin{matrix}Y \X\end{matrix} =Sigma=\begin{matrix} \Sigma yy} &amp; \Sigma_{yx} \ \Sigma_{xy} &amp; \Sigma_{xx} \end{pmatrix} &gt; &#36;0.00
(Calling)</p>
<p>For Any &#36;\alpha \in \mathbb{R}&#36;、&#36;\beta \in \mathbb{R}^p&#36;, by:
&#36;E\left (Y-\left (y-\sigma yx)\Y-(alpha+\bita)^2&#39;- What?
That's...<strong>Optimistic linear predictor</strong>Is:
&#36;&#36;\hat{Y} = \alpha^* + \beta^{*\prime}X = \mu_y - \Sigma_{yx}\Sigma_{xx}^{-1}\mu_x + \Sigma_{yx}\Sigma_{xx}^{-1}X&#36;&#36;</p>
<p>If Random Vector &#36;\begin{pmatrix} Y \ X \end{pmatrix}&#36; Comply with multiple normal distribution: &#36;US&#36;\left (\begin{matrix}\mu end{matrix},\begin{matrix}\sigma y} &amp; \Sigma_{yx} \ \Sigma_{xy} &amp; \Sigma_{xx} \end{pmatrix} \right)&#36;&#36; 则对任意函数 &#36;f: \mathbb{R}^p \to \mathbb{R}&#36;，有： &#36;&#36;E (Y-(y-\Sigma) ^ (Y-f) ^2 US&#36;2 million under the distribution of multiple normals,<strong>The best linear predictor is the best predictor.</strong>。</p>
<h3>Return to squared and decomposition</h3>
<p>In multiple regressions, the total squares can be broken down to: &#36;Y&#39;Y = Y&#39;MY + Y&#39;(I-M)Y = SSR(X)+SSE&#36;&#36;, of which: </p>
<ul>
<li>&#36;SSR(X) = Y&#39;My&#36; is the return to squared </li>
<li>&#36;SSE = Y&#39;(I-M)Y&#36; is the sum of the error squared</li>
</ul>
<p>Return squared and can be further decomposed to: &#36;&#36;SSR(X) = SSR(X_1) + SSR(X_2|X_1) + \ldots + SSR(X_p|X_1, \ldots, X_{p-1})&#36;&#36;Definition of the condition squared &#36;SSR(X_j|X_1, \ldots, X_{j-1})&#36; This post is part of our special coverage Syria Protests 2011. </p>
<ul>
<li>Contained in Model &#36;X_1, \ldots, X_{j-1}&#36; In the case </li>
<li>Add &#36;X_j&#36; ♪ And the squared and the increased ♪</li>
</ul>
<p>This breakdown reflects the contribution levels of the various self-variant variables in the multiple regressions: </p>
<ol>
<li>&#36;SSR(X_1)&#36;: Independent contribution of the first variable </li>
<li>&#36;SSR(X_2|X_1)&#36;: under control &#36;X_1&#36; The blogger says:&#36;X_2&#36; and Add.1 - 2 </li>
<li>By such extrapolation, each variable ' s conditional contribution excludes the impact of the preceding variable</li>
</ol>
