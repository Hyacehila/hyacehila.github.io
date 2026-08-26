---
title: 'Logistic Regression: Binary Responses, Linear Probability Models, and Logit Models'
title_zh: Logistic 回归：二分类因变量、线性概率模型与Logit 模型
date: 2024-03-13 13:23:55 +0800
categories:
- Data Science
- Statistical Modeling & Inference
tags:
- Statistics
- Regression
- Logistic Regression
author: Hyacehila
mathjax: true
hidden: true
excerpt: Covers binary response variables, linear probability models, logit models, parameter estimation, interpretation,
  and related extensions.
description: Covers binary response variables, linear probability models, logit models, parameter estimation, interpretation,
  and related extensions.
excerpt_zh: 整理二分类因变量、线性概率模型、Logit 模型、参数估计、模型解释和相关扩展。
permalink: /blog/2024/03/13/logistic-regression-notes/
lang: en
translation_key: 2024-03-13-logistic-regression-notes
translation_status: machine
translation_source_hash: b8a23d4b1dd48b1f05c54abdbf6e478326640b5fa9e64ce2fe3b9bfe8cf73ffb
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<h2>Category 2 returns by variable and Logistic</h2>
<h3>Introduction</h3>
<p>The linear regression model is one of the most popular methods in modern statistical analysis.<a href="/en/blog/2023/09/04/linear-regression-basics-notes/">Linear regression base</a>There's been a lot of introductions; but in a lot of cases, linear regression can be limited, and it's very common. <strong>Because variables are classified variables</strong> </p>
<p>The most common way to deal with this situation is the loglinear model, and we're here to present a special logarithmic model, the Logistic model, which, for ease of understanding, will not start with the theory of a logarithmic model, but will be similar to the linear regression model that we have learned to describe the Logistic regression model.</p>
<h3>Linear Probability Model</h3>
<p>In the traditional regression analysis, we do not limit the type of variable that we can accept, whether it's a continuous variable or a second-class variable.
If you can only take classification values because of variables, now we're using the most common 0-1 classification, and see what happens with the minimum two-fold method.
&#36;&#36;y_{i}=\alpha+\beta x_{i}+e_{i},&#36;&#36;
of which&#36;y_i&#36;It's a two-class variable, only 0 and 1
It's given.&#36;x&#36;in case of calculation&#36;y&#36;There are expectations.
&#36;&#36;\begin{gathered}
E\left(y_{i}|x_{i}\right) =E\left(\alpha+\beta x_{i}+e_{i}\right) \
=\alpha+\beta x_{i}.
\end{gathered}&#36;&#36;
At this point, there is.
&#36;&#36;E\left(\left.y_{i}\right|x_{i}\right)=P\left(\left.y_{i}=1\right|x_{i}\right)&#36;&#36;
Which means...  &#36;y_i&#36;The expectation is no.&#36;i&#36;A family, the probability of a return to the calculated event, is also a linear probability model. The linear regression model for the second classification because of the variable is also called the linear probability model.
That's the probability that things don't happen.
&#36; \begin{aligned}&amp;P\left(\left.y_i=0\right|x_i\right)=1-\left(\alpha+\beta x_i\right)=1-\alpha-\beta x_i.\end{aligned}&#36;&#36;
据此 我们可以计算残差有
&#36;&#36;\begin{aligned}&amp;e_{i}=y_{i}-\alpha-\beta x_{i}\end{aligned}&#36;&#36;
计算残差的方差有
&#36;&#36;\begin{aligned}
&amp;=\left(\alpha+\beta x_{i}\right)\left(1-\alpha-\beta x_{i}\right) \
&amp;== sync, corrected by elderman == @elder man
I'm sorry.
Which means... <strong>The difference depends on the variation in the value of the variable, and the difference in observations is different, which is statistically referred to as unsatisfactory variance.</strong></p>
<p>The LPM projections are still problematic due to the special nature of the values of the variables.</p>
<ul>
<li>The difference means the difference in the parameter estimate is biased and any hypothetical test is invalid.</li>
<li>The probabilities are likely to exceed.&#36;[0,1]&#36;It's against common sense.</li>
<li>A linear function does not match the true form of this model.</li>
</ul>
<p>In conclusion, we need to look for a new model to study this subclassic variable, or to promote a step towards the regression of the subclassical variable.</p>
<h3>Logistic regression model</h3>
<p>It's very realistic that we need to look for a function that is appropriate for our predictive model, most commonly the Logistics distribution.</p>
<p>Assuming there's a continuous variable that describes the possibility of an event, he can change freely, and when it reaches zero, it happens, so...<em>&gt;0&#36; &#36;y_i=1&#36; 其他情况下 &#36;y_i=0&#36; 我们只能观测到&#36;y i&#36;Assuming a linear model
&#36;0.00</em>}=\alpha+\beta x_{i}+\varepsilon_{i}.&#36;&#36;
研究条件概率有
&#36;&#36;\begin{aligned}
P\left(y_{i}=1|x_{i}\right)&amp; =P\left[\left(\alpha+\beta x_{i}+\varepsilon_{i}\right)&gt;0\right]  \
&amp;=P\left[\varepsilon_{i}&gt;\left(-\alpha-\beta x_{i}\right)\right].
\end{aligned}&#36;&#36;
通常我们假设误差项&#36;\varepsilon_{i}&#36; 服从logistic分布或者标准正态分布（分别对应logistics模型和probit模型） 因此我们可以改写出累积分布函数的形式
&#36;&#36;\begin{aligned}
P\left(y_{i}=1|x_{i}\right)&amp; =P\left[\varepsilon_{i}\leq\left(\alpha+\beta x_{i}\right)\right]  \
&amp;=F\left(\alpha+\beta x_{i}\right),
\end{aligned}&#36;&#36;
我们后面主要研究logistics模型 对于probit模型只在最后进行简单的介绍
那么现在假设误差项&#36;\varepsilon_{i}&#36; 服从logistic分布有
&#36;&#36;\begin{aligned}
P\left(y_{i}=1\left|x_{i}\right)\right.&amp; =P\left[\varepsilon_{i}\leqslant\left(\alpha+\beta x_{i}\right)\right]  \
&amp;== sync, corrected by elderman == @elder man
I'm sorry.
His range of values is in the range of 0-1. </p>
<p>Now we're going to work on how to design specific models.
&#36;&#36;P(y_{i}=1\left|x_{i}\right)=\frac{1}{1+e^{-\left(\alpha+\beta x_{i}\right)&#125;&#125; &#36;&#36;
Here, we define &#36;\alpha+\beta x_{i}&#36; It is a linear function of a series of factors that influence the probability of an event.
And replacing the probability of an event would be a model for re-entry.
&#36;&#36;\begin{gathered}
p_{i} =\frac{1}{1+e^{-\left(\alpha+\beta x_{i}\right)&#125;&#125; \
=\frac{e^{\alpha}+\beta x_{i&#125;&#125;{1+e^{\alpha+\beta x_{i&#125;&#125;},
\end{gathered}&#36;&#36;
The probability of an event not occurring can be derived from a discrepancy with one.
&#36;&#36;\frac{p_{i&#125;&#125;{1-p_{i&#125;&#125;=e^{(\alpha+\beta x_{i})}.&#36;&#36;
It's what we call a ratio.
It's a match.
&#36;&#36;\ln\left(\frac{p_{i&#125;&#125;{1-p_{i&#125;&#125;\right)=\alpha+\beta x_{i}.&#36;&#36;
It means that the logarithmic logarithmics of the odds are linear; we can use the meaning of the original set of explanation coefficients, and we don't have to worry about the magnitude of the model that leads to excess probability.</p>
<p>The multivariant model changes naturally to
&#36;&#36;p_{i}=\frac{e^{a+\sum_{i=1}^{k}\beta_{k}x_{ki&#125;&#125;}{1+e^{a}+\sum_{i=1}^{k}\beta_{k}x_{ki&#125;&#125;.&#36;&#36;
&#36;&#36;\ln\left(\frac{p_{i&#125;&#125;{1-p_{i&#125;&#125;\right)=a+\sum_{k=1}^{k}\beta_{k}x_{ki}.&#36;&#36;</p>
<p><strong>The regression model that we're presenting because the variable is classified is the classic non-linear regression model, which, while describing a linear process, is not because of the variable.</strong>
<strong>When we have a sample value and know if the event is going to happen, we can study the probability of happening in a given situation.</strong></p>
<h2>Parameter estimation of the Logistic regression model</h2>
<p>It's a very natural idea, and after the model's presentation, we need to start looking at how to calculate the values of the parameters in the model based on known information.</p>
<h3>Very similar estimates (MLE)</h3>
<p>The most important parameter estimate we're presenting in the linear regression model is the minimum hyperbolic method (OLS).
But it's not working well in the Logistic regression model, and the most important method we're going to use here is a very similar estimate; in the online regression model, it's a very similar estimate and a minimum of two multipliers, but a very similar estimate can also be applied to the estimates of the non-linear model.</p>
<p>The idea of a very specious estimate is to create an apparent function that selects the right parameter to get the maximum value from the apparent function. Value<a href="/en/blog/2023/03/18/mathematical-statistics-notes/">Mathematical statistics</a> The “Big Appearances” section
Set&#36;p_i&#36;It's given.&#36;x_i&#36;Under the circumstances, the probability of 1 is observed.
&#36;&#36;\begin{gathered}
p_{i} =\frac{1}{1+e^{-\left(\alpha+\beta x_{i}\right)&#125;&#125; \
=\frac{e^{\alpha}+\beta x_{i&#125;&#125;{1+e^{\alpha+\beta x_{i&#125;&#125;},
\end{gathered}&#36;&#36;
Well, the probability of getting an observation is...
&#36;&#36;P\left(y_{i}\right)=p_{i}^{y_{i&#125;&#125;\left(1-p_{i}\right)^{1-y_{i&#125;&#125;,&#36;&#36;
Multiple observations.
&#36;&#36;L\left(\theta\right)=\prod_{i=1}^{n}p_{i}^{y_{i&#125;&#125;\left(1-p_{i}\right)^{\left(1-y_{i}\right)}.&#36;&#36;
The parameters of the Logistic regression model can be estimated by a large degree of cosmopolitan function that we calculate.</p>
<h3>Assumptions</h3>
<p>As OLS requires that Gauss-Markov assume that the Logistics model has its own assumptions, they do not have the same assumptions as OLS.</p>
<ul>
<li>Data from random samples</li>
<li>Because the variable is assumed to be a non-linear function form of the variable (requires an intrinsic link)</li>
<li>Sensitivity to reconnectivity.</li>
<li>Because variables can only be classified as category II.</li>
<li>Don't ask for normality on your own.</li>
</ul>
<h3>Sample size and MLE nature</h3>
<p>The LOGistic regression model is very much estimated to have</p>
<ul>
<li>Coherence</li>
<li>Progressive effectiveness</li>
<li>Progressive normality
They are all large samples;
This means that, in the case of small samples, the statistical nature of the Logistic regression model cannot be determined as to how we should extend the sample appropriately, as conditions permit; there is a certain amount of research in the academic community that has not led to a definitive result.</li>
<li>The sample number is over 100, and the effect is better.</li>
<li>When the number of samples is over 500, it's part of the big sample.</li>
<li>More parameters rely on more observations to estimate that each parameter should be accompanied by more than 10 samples.</li>
<li>When it's of a reconnective linear nature, you need to expand the sample appropriately.</li>
<li>When there are more classifications in the model, the sample should be expanded appropriately.</li>
</ul>
<h3>Group data and Logistic regression model</h3>
<p>It's a special Logistics regression model so that he's a little closer to linear regression models, as described below;</p>
<p>We're looking at the relationship between university advancements and students' gender, the focus on high school attendance, and the three classification variables of the level of achievement of students, which is the secondary classification variable of the regression factor.
From this point of view, this is an ordinary Logistics regression model.</p>
<p>Now, we're going to change the angle from one person to another; we're going to calculate the percentage of all students in a high school with a particular gender.&#36;f_i&#36; Calculating a group regression model for prediction
At this point in time, the variable remains unchanged because the variable becomes one&#36;[0,1]&#36; The amount between them, if you use the minimum of two times directly in a continuous pattern, will inevitably cause a problem, so we're going to have another deformation.
&#36;&#36;\begin{aligned}\ln\left(\frac{f_j}{1-f_j}\right)=a+\beta_1\text{GENDER}_j+\beta_2\text{KEYSCH}_j+\beta_3\text{GRADE2}_j+\varepsilon_j\end{aligned}&#36;&#36;
Now it's possible to estimate the regression coefficient by using the minimum hyperplication; it's essentially a non-linear regression model.</p>
<p>Because of the heterogeneity associated with the direct use of OLS, we need to use the difference in the margin of the difference for a minimum of two times the estimate of which the weight is the inverse of the error in the standard.
&#36;&#36;{S_{j&#125;&#125;^{2}=\mathrm{Var}\left(\frac{\varepsilon_{j&#125;&#125;{p_{j}\left(1-p_{j}\right)}\right)=\frac{1}{n_{j}f_{j}\left(1-f_{j}\right)}&#36;&#36;
&#36;&#36;\begin{aligned}\left(\frac{1}{S_{j&#125;&#125;\right)\ln\left(\frac{f_{j&#125;&#125;{1-f_{j&#125;&#125;\right)&amp;=\left(\frac{1}{S_{j&#125;&#125;\right)\alpha^{<em>}+\beta_{1}^{</em>}\mathrm{GENDER}<em>{j}\left(\frac{1}{S</em>{j&#125;&#125;\right)+\beta_{2}^{<em>}\mathrm{KEYSCH}<em>{j}\left(\frac{1}{S</em>{j&#125;&#125;\right)\&amp;+\beta_{3}^{</em>{\cHFFFFFF}{\cH00FF00} \mathrm{GRADE}2}\loft(s{right)+u j}.\end{aligned}&#36;&#36;
Software can help us complete these complex calculations.</p>
<h2>Evaluation of regression models</h2>
<p>The proposed eulogy test is designed to test whether there is a sufficiently small gap between the proposed model and the actual observations; we have more than one indicator that will help us study how the proposed eugenics are.</p>
<h3>Pearson &#36;\chi^2&#36; Proposed Preference</h3>
<p>First of all, we need to introduce the concept of covariate pattern, which is the type of competitor that translates into Chinese;
Accompaniment variables are defined as variables that affect the variables and are not controlled by the tester.
The number of calibrated variables (used only when the full classification is due to the variable)
We can give a breakdown of observations and projections in different cavariate scales, and the effect is a high-dimensional classification.
We're here.<a href="/en/blog/2023/03/18/mathematical-statistics-notes/">Mathematical statistics</a> A similar situation is described in the section on "The idea of a Peason-Carp Probability Test"
This test will help us compare the predictions with the observations.&#36;\chi^2&#36;When you're older, it's not good. &#36;\chi^2&#36;Proposed Preference</p>
<h3>Deviance</h3>
<p>And we use the approximation to describe the comparison of observations and projections, which seems to mean the probability of producing observations under certain parameter estimates.
We'll use it.&#36;L_s&#36;So we need to give a baseline -- the likeness of saturation models -- that's what makes perfect predictions. &#36;L_f&#36;  By comparing the two apparent models, you can judge the sum of the models.&#36;D&#36;Statistics
&#36;D=2ln\left<em>{s&#125;&#125;{\hat{L}</em>{f&#125;&#125;\right)=-2\left(\ln\hat{L}<em>{s}-\ln\hat{L}</em>{\fnH00FFFF}
It almost obeys when the sample size is large enough.&#36;\chi^2&#36;It describes the deviation between our model and the perfect model, which is Devance.
And when it's small enough, it means that the model is ready to work. Okay.
<strong>When MLE formulation models are used, deviations and calibrations generally have near-extract values</strong></p>
<p>We're using deviation &#36;\chi^2&#36; The following is required as an indicator to measure the proposed profile of the model:</p>
<ul>
<li>Over 10 observations per variable type
It's easy to see from here. <strong>Deviance Pearson &#36;\chi^2&#36; None of the proposed advantages apply to the test of the intended advantages of the logistics regression model with a continuous variable.</strong></li>
</ul>
<h3>Hosmer-Lemeshow Probability</h3>
<p>It's kind of like Pearson. &#36;\chi^2&#36; It's a test to test the merits, but it circumvents some of the competitor type observations by artificial grouping.
&#36;HL=sum g}g}\frac{\left\n g\widehat{p}<em>{g}\right)}{n</em>{g}\widehat{p}<em>{g}\left(1-\widehat{p}</em>}right }
Or with Pearson? &#36;\chi^2&#36; The smaller it is, the better it is;
<strong>If not significant (&#36;p)&gt;The difference is a good match.</strong></p>
<h3>Information measurement indicators</h3>
<p>Just as we are.<a href="/en/blog/2023/09/04/linear-regression-basics-notes/">Linear regression base</a> ".&#36;AIC&#36;The AIC (Akaike) BIC (Beyers) that we can model based on two levels of information is mentioned in the section of the Code.
The smaller they are, the better.
<strong>Information volume judgement models are intended to serve as guidelines for the wider application of modern mathematical statistics</strong></p>
<h3>Category&#36;R^2&#36;</h3>
<p>Online sexual regression in progress &#36;R^2&#36; It's the most widely used indicator to measure the effect of the model; because the Logit model's variables are classified, we were originally&#36;R^2&#36;Statistics can't continue to be used; but we can still create similar statistics as if
&#36;LRI=\left(\frac{2L\hat{L}<em>{0}-\left(-2L\hat{L}</em>{\cHFFFFFF}{\cH00FF00}
of which &#36;-LL_0&#36; Summarized by analogy &#36;-LL_s&#36; Similarity is the sum of deviations
and&#36;R^2&#36;The LRI-like range is 0 to 1 and the closer to 1 means the better.</p>
<p>Mathematicians always wanted to create indicators that would unite them, but because of the specificity of the logit model, there was no expression that would cover both, but we still defined the logit model.&#36;R^2&#36; It's kind of like the above, and then we'll use it as a measure to assess the proposed effect of the logit model, which is the most commonly used statistically of proposed advantages.</p>
<h2>Study of regression factors</h2>
<p>When the model has a good coding effect, there's a point in studying the coefficient.
Because of the specific nature of the logic model</p>
<h3>Odds and Odds Ratio</h3>
<p>What happens is the frequency of events.
&#36;&#36;\mathrm{odds}=(\text{事件发生频数})/(\text{事件不发生频数}).&#36;&#36;
It can also be divided by the total number of events.
&#36;&#36;odds_{k}=\left[p_{k}/\left(1-p_{k}\right)\right]&#36;&#36;
We can also immediately compare the probability of an event happening and the probability of not happening. That's right.</p>
<p>Because of his margin composition, the upper limit of the range has no boundary, when the margin is greater than one, and vice versa.</p>
<p>The way to compare the ratio should be to divide it by comparing it to the Odds Ratio or to describe the relationship between the same event ratio of different groups, which is very common in later logit returns, especially in the interpretation of the effect of the variable on the variable in the logit model;</p>
<ul>
<li>If his OR is more than one, it means that self-variant has a positive effect on events. </li>
<li>For multi-category variables and multiple models
<strong>After that, we used the OR to evaluate logit coefficients.</strong></li>
</ul>
<h3>Explain logit coefficient according to OR</h3>
<h4>Rate of occurrence of logit models</h4>
<p>In our online regression model, the coefficient of the model is a better explanation.</p>
<p>But in the logit model, what we're dealing with is a non-linear model, and the direct effect of the coefficient is that it's difficult to estimate the results in a logarithmic unit, so we need to use the rate of occurrence to study it.</p>
<p>In the previous example,
&#36;&#36;\begin{aligned}\ln\left(\frac{f_j}{1-f_j}\right)=a+\beta_1\text{GENDER}_j+\beta_2\text{KEYSCH}_j+\beta_3\text{GRADE2}_j+\varepsilon_j\end{aligned}&#36;&#36;
Ours.&#36;\alpha&#36; That's the logarithm of the benchmark ratio, that's when all the parameters are zero.</p>
<p>It's easier to understand than to understand logarithms.
&#36; \begin{aligned}\mathrm{odds}=frac{p}{p}&amp;=\exp\left(\alpha+\beta_1\text{GENDER}+\beta_2\text{KEYSCH}+\beta_3\textbf{MEANGR}\right)\&amp;== sync, corrected by elderman == @elder man &#36;
And right now, we can see that when the coefficient is positive, he's having a positive effect; he's having a positive effect.&#36;e^{\beta}&#36; He's actually the ratio.</p>
<h4>Ratio of occurrence of continuous variables</h4>
<p>When?&#36;x_k&#36;When you add a unit, the number of times you change.&#36;e^{\beta_k}&#36;<br>Same thing.&#36;e^{\beta_k}-1&#36;<br>The coefficients that we're talking about are the influence factors when we control the other variables.&#36;e^{\beta_k}&#36;
To adjust the incidence ratio (adjusted odds radio) to AOR</p>
<p>In many cases, we don't want to study the effects of a change in a unit of continuous variables, more about multiple units of change.&#36;a&#36;Change to&#36;b&#36; The AOR at this time is
&#36;&#36;e^{\beta_k(b-a)}&#36;&#36;
At this point, our AOR is an amount that only relates to the variation margin, which is that our modelling is actually linear, and in many cases it shouldn't be linear.</p>
<h4>Rate of occurrence of the category II from variables</h4>
<p>There are only two changes between 0 and 1 or between 1 and 0 in the second classification.
Very easy to calculate.
&#36;&#36;AOR=e^\beta&#36;&#36;
<strong>AOR is equal to and less than 1.</strong></p>
<h4>Rate of occurrence of multi-category variables</h4>
<p>Based on our basic techniques in regression analysis, we need to create virtual variables.<a href="/en/blog/2024/05/24/generalized-linear-regression-notes/">Broad linear regression</a> And in principle, if we have a classification variable, we're going to be able to use it as a source of information.&#36;m&#36;It's a category.&#36;m&#36;A variable to describe what he belongs to.&#36;m-1&#36;A virtual variable.
&#36;&#36;\ln\left(\frac{p}{1-p}\right)=\alpha+\beta_{1}\text{GENDER}+\beta_{2}\text{SCH}1+\beta_{3}\text{SCH2}+\beta_{4}\textbf{MEANGR}.&#36;&#36;
At this point, our original variable SCH has three categories, and we have chosen one of them as a reference category, so the equation has two virtual variables.
This is our time. &#36;AOR=e^\beta&#36; is the occurrence ratio of category from reference to 1 or 2
<strong>The current statistical software has the ability to automatically generate virtual variables from multi-classical variables and will automatically calculate the relevant coefficients and AOR values for us, while performing a high-profile test.</strong></p>
<p><strong>The AOR value only reflects the ratio problem of 0-1.</strong></p>
<h3>Standardization of logit models</h3>
<p>In retrogression analysis, we're showing how standardized coefficients make sense.
But there's no point in standardizing classification variables.
But for the logit model, his variable classification, but it's non-linear, and it's actually standardized, and at this point we're going to have to do extra work on the coefficients of the individual variables that we're going to have to do, and we're going to give the formula that we're going to have to do.
&#36;&#36;\beta^{*}=\frac{\widehat{\beta}s_{x&#125;&#125;{\sqrt{s_{logit}^{2}/R^{2&#125;&#125;}=\frac{\widehat{\beta}s_{x}R}{s_{logit&#125;&#125;&#36;&#36;
That means a logit regression, then a calibration.
<strong>Standardized coefficients allow for comparison, the same thinking and linear regression.</strong></p>
<h3>Visibility test for regression coefficients</h3>
<p>The remarkable level of the regression factor, which is what we usually use.&#36;p&#36;The value is the same as the linear regression.&#36;p&#36;Value less than&#36;0.05&#36;When it's the equivalent, we're judging it to be sufficiently obvious that we reject the assumption that the coefficient is valuable.</p>
<h4>Wald Test</h4>
<p>For a large sample, it's possible to test the overall coefficient for zero.&#36;Z&#36;Statistics
&#36;Z=hat(beta)<em>{k}/SE</em>{\beta_{k&#125;&#125;.&#36;&#36;
使用双侧的&#36;t&#36;检验
在大多数统计软件中侧重于使用Wald检验 也就是
&#36;&#36;\dot{W}=\left(\hat{\beta}<em>{k}/\mathrm{SE}</em>It's okay.
Obey.&#36;\chi^2&#36;Distribution
<strong>When the absolute value of the regression factor is large, the Wald statistical value becomes small, which is not applicable at this time.</strong></p>
<h4>Like a test.</h4>
<p>Statistically, there's a proof of semblance; two times the value of the logarithmic between the two models is subject to the calorie distribution.
Using software to calculate what appears to be a direct line to the calorie distribution, there's no difference between tests and Wald's statistics.</p>
<h4>Subset of test coefficients</h4>
<p>Sometimes we don't just want to look at a certain coefficient's level of prominence, but we want to know if some coefficients are significant in general and in multiple regressions.&#36;F&#36;It's the same thing to test ideas; of course, if the problem with virtual variables is involved, it's more important that the profile of the coefficient subset is high.
It's obvious that the LR test only needs to compare the approximation of changes in models, and it's very appropriate to test the coefficients subset, except for the freedom changes.</p>
<h3>Forecast probability</h3>
<p>The logit model, of course, can give the predictions that go into the calculated logit regression formula for our variables, and then the associated non-linear changes can calculate the probability that the variables will occur.
&#36;&#36;p_{i}=\frac{e^{a+\sum_{i=1}^{k}\beta_{k}x_{ki&#125;&#125;}{1+e^{a}+\sum_{i=1}^{k}\beta_{k}x_{ki&#125;&#125;.&#36;&#36;
We can do a variety of studies on the probability of being calculated, for example, by misusing multiple odds.
We'll introduce the results to the confidence zone.</p>
<h3>Confidence interval for regression parameters</h3>
<p>Visibility tests can tell us if a coefficient is significant.
Probability predicts a specific predictive probability of an event.
But the parameters can't be very precise.</p>
<h4>Confidence interval of regression factor</h4>
<p>For Selected&#36;\alpha&#36; The confidence interval is where SE is the standard error of the corresponding coefficient.
What's that?<em>{k}\pm Z</em>{\alpha/2}\times SE_{\beta_{k&#125;&#125;&#36;&#36;</p>
<h4>Confidence interval for which rate is due</h4>
<p>Most researchers are not paying much attention to the confidence zone of the regression coefficient.
With us.<a href="/en/blog/2023/09/04/linear-regression-basics-notes/">Linear regression base</a> And we're more concerned about the confidence zone in the prediction probabilities section of this paper.</p>
<p>After the adjustment factor (0-1), we can give two predicted occurrence rates, and the corresponding confidence interval is from the factor to the ratio.
&#36;&#36;(e^{0.509},e^{1.223})\longrightarrow(1.664,3.397).&#36;&#36;</p>
<h4>Confidence interval for event probability</h4>
<p>The confidence zone with the ratio is still less than the confidence zone with the probability of an event; the latter is the confidence zone that really reflects our ultimate projection.
His form is
&#36;&#36;(e^{\mathrm{logit}\left(y\right)-1.96\sqrt{\mathrm{Var}\left[\mathrm{logit}\left(y\right)\right]&#125;&#125;,e^{\mathrm{logit}\left(y\right)+1.96\sqrt{\mathrm{Var}\left[\mathrm{logit}\left(y\right)\right]&#125;&#125;).&#36;&#36;
Of which logit(y) is harder to calculate, but it doesn't have to be done manually.</p>
<h2>Diagnosis of regression</h2>
<h3>Variable Selection</h3>
<p>Our mission is to identify variables that can work very well on predictions and incorporate them into our models.</p>
<h4>Filter from Variables</h4>
<p>We'll first consider which variables are worthy of inclusion.
In online regression, we've incorporated relevant quantities into our models; but there's no correlation in logit return.<a href="/en/blog/2023/11/05/descriptive-statistics-and-visualization-notes/">Descriptive statistics and visualization</a> The `relevant analysis' section does not provide a methodology for measuring the correlation between qualitative and quantitative amounts
<strong>We should do a one-dollar logit return to see if it has any significant effect.</strong></p>
<p>Only significant quantities tested should be included in regression models.</p>
<h4>Progressive return</h4>
<p>Just as we are.<a href="/en/blog/2023/09/04/linear-regression-basics-notes/">Linear regression base</a> The concept of gradual return is presented in the section on gradual return, which is just what we need to describe in the section on evaluation of regression models in which we compare the results of models.
The effect of forward and backward approaches is still less than the cumulative gradual return.</p>
<p><strong>The deletion of meaningful variables and the retention of meaningless variables would have negative effects on models, as would linear regression, which is also a question of choosing variables.</strong></p>
<h3>Non-linear</h3>
<p>The logit regression model we're using right now. Medium
&#36;&#36;\ln\left(\frac{p_{i&#125;&#125;{1-p_{i&#125;&#125;\right)=a+\sum_{k=1}^{k}\beta_{k}x_{ki}.&#36;&#36;
The function on the right is the right half of a classic linear regression model.
We sometimes need to change the right to a non-linear form, just like the possibility of non-linear regression.</p>
<h3>Interaction</h3>
<p>The same linear regression model we used.
The logit model may also have a self-variant interaction that requires special treatment.
We don't have much of an introduction here.</p>
<h3>Extrusion.</h3>
<p>Oversegregation is a special case leading to a reduction in the intended benefit; also known as two variations
It only occurs when the following models are defective, and the consequence is that the models are not working well.
<strong>There's no point in dealing with fragmentation. We need to think about the following after we find the problem.</strong></p>
<ul>
<li>Too few observations in some variable types (recommended to merge)</li>
<li>Some important variables or cross-sections are not included in the model</li>
<li>It's non-linear but not considered.</li>
<li>There is an odd value</li>
<li>Change in data preprocessing not in place</li>
</ul>
<h3>Empty Unit</h3>
<p>The number of observations in some competitor type is 0
Usually because of too many variables.
If there's a large number of empty units, they need to be merged.
<strong>The most obvious phenomenon is that the coefficient is too large and the standard is very wrong.</strong></p>
<h3>Full separation.</h3>
<p>The change in one variable is a direct factor in our prediction.
<strong>The most obvious phenomenon is that the coefficient is too large and the standard is very wrong.</strong>
It's a small sample, but a lot of arguments.</p>
<h3>Reunification Linear</h3>
<p>The Cyclops are the classic problem of linear regression.<a href="/en/blog/2023/09/04/linear-regression-basics-notes/">Linear regression base</a> “Estimation of the regression parameters (under cholinear)” I Section
This is also an important issue in the return of logit.
Rycolinearity also manifests itself as a greater standard error in logit return.
And unlike a series of loss-regressive methods, we have no solution but to try to resolve the lack of a solution to the problem of communism.</p>
<ul>
<li>Use of reduction methods such as PC return</li>
<li>Try to remove the variables that lead to the recoherence. Volume</li>
</ul>
<h3>Odd</h3>
<p>We've been working on two important regression diagnostics in online re-entry.</p>
<ul>
<li>Impact analysis</li>
<li>Disability analysis
They're all closely related to those extremely special values.
<strong>There's a change in statistics and linear regression here.</strong></li>
</ul>
<h4>Residue in logit model</h4>
<p>Here are the more frequent disabilities we use.</p>
<h5>Non-standardised disability</h5>
<p>Difference in probabilities and reality
&#36;&#36;y-\hat{P}\left(y=1\right)&#36;&#36;</p>
<h5>Pearson disability (standardized disability)</h5>
<p>Standardized adjusted residuals
&#36;&#36;z=\frac{y-\hat{P}\left(y=1\right)}{\sqrt{P\left(y=1\right)\left(1-P\left(y=1\right)\right)&#125;&#125;,&#36;&#36;</p>
<h5>Logit disability</h5>
<p>&#36;&#36;L=\frac{y-\widehat{P}\left(y=1\right)}{\widehat{P}\left(y=1\right)\left(1-\widehat{P}\left(y=1\right)\right)}.&#36;&#36;</p>
<h5>Devance</h5>
<p>&#36;&#36;d=\pm\sqrt{-2\left[y\ln\left(P\right)+\left(1-y\right)\ln\left(1-P\right)\right].}&#36;&#36;</p>
<h5>Students with disabilities</h5>
<p>Student logit disability is no longer related to disability.
It's a situation where, after the model changes, the deviation is to be combined to measure whether a parameter is important.</p>
<h4>Impact analysis</h4>
<h5>Leverage statistics</h5>
<p>&#36;&#36;H=X\left(X^{\prime}X\right)^{-1}X^{\prime},&#36;&#36;
&#36;H&#36;Elements of the matrix diagonal</p>
<h5>Cook Statistics</h5>
<p>It's a combination of standardized disability and leverage statistics that reflects the impact of this parameter on the model.
&#36;Cook&#39;sD_{i}=\left(Z_{i}^{2}\times h_{i}\right)/\left(1-h_{i}\right)^{2},&#36;&#36;</p>
<h4>How to test an odd value</h4>
<p>We have the basic idea.</p>
<ul>
<li>One observation has a large, non-systemic disability.</li>
<li>Too much leverage.</li>
<li>Too big Cook statistics</li>
</ul>
<h2>Extension of Logistic regression model</h2>
<h3>Probit Model</h3>
<h4>Probit Model</h4>
<p>We introduced it at the beginning of the whole text, except for the Logistics function.
The probability of an event can be described as
&#36; \begin{aligned}
\text{P}&amp; =P\left(y=1|x\right)  \
&amp;=F\left(\alpha+\beta x\right) \
&amp;=\int_{-\infty}^{\alpha+\beta x}f\left(z\right)dz,
\end{aligned}&#36;&#36;
其中&#36;F,f(z)&#36; 分别是正态概率密度的CDF和PDF
变形可以得到
&#36;&#36;=P=alpha+x, &#36;
This is our probit model.
The parameters can be estimated using MLE.</p>
<h4>Interpretation of the model</h4>
<p>Because the variables are changed for the reverse effect of the normal CDF, this explanation is clearly intuitive, and we can introduce odds in the logit model to help us calculate, but the probit model is not so good.</p>
<h5>Forecast probability</h5>
<p>As long as the value of the variable is replaced directly and the CDF is reversed, it is possible to calculate a specific probability value.</p>
<h5>Effect on probability</h5>
<p>The most intuitive approach is to calculate the change in probability values and then calculate the percentage of change to determine the effect of a self-variant on the variable.</p>
<h4>Probit model for grouping data</h4>
<p>As we explained in the section on Cluster Data and Logistic Return Models, we need</p>
<ul>
<li>Use the frequency to calculate the corresponding regression factor variable</li>
<li>Completing with OLS </li>
<li>Consider weighted OLS weights as the last of the difference squared</li>
</ul>
<h4>Comparison of Logit model and Probit model</h4>
<p>Logit model and Probit model have extremely close CDF curves in the case of subcategories
That means their return is basically the same.</p>
<p><strong>But we can't compare the proposed advantages of the Logit model, the obvious tests, which means that the Probit model is less explanatory, and we actually use less.</strong></p>
<h3>Order returned by Logit of the variable</h3>
<p>It's very common to return because the variable is sequenced.
In the view of some scholars, order can be considered as a continuous variable as long as the number of variables is greater than 5, although there are occasional problems;
In order to deal with a smaller number of sequences, or we don't really want to do transformational continuums, we introduced the Logit model of sequence-based variables.</p>
<h4>Cumulative Logit regression model (Cumulative LRM)</h4>
<p>The basic form of the model is as follows:
&#36;0.00<em>== sync, corrected by elderman == @elder man
When the actual observation variable has&#36;J&#36;When it's a class, we'll give it to the dollar.</em>&#36; 设定&#36;J-1&#36;个未知的门槛 当他们达到门槛后 就意味着自动进入下一个级别 这些门槛被记为&#36;\mu_j&#36; </p>
<p>At this point, we can give the following form to the CDF:
&#36; \begin{aligned}
P\left (y\leqslant j\right)&amp; =P\left(y^{<em>}\leqslant\mu_{j}\right)  \
&amp;=P\left[\left(\alpha+\sum_{k=1}^{K}\beta_{k}x_{k}+\varepsilon\right)\leqslant\mu_{j}\right] \
&amp;=P\left[\varepsilon\leqslant\mu_{j}-\left(\alpha+\sum_{k=1}^{K}\beta_{k}x_{k}\right)\right] \
&amp; =F\left[\left.\mu_{j}-\left(\alpha+\sum_{k=1}^{K}\beta_{k}x_{k}\right)\right].\right.
\end{aligned}&#36;&#36;
根据它就可以计算出累计概率 然后推出各个类别的概率
&#36;&#36;P\left(y\leqslant j\mid x\right)=P\left(y^{</em>}leqslant\mu=mid x\right}\frac{\mathrm}e^left[\j}-\left(a+sum =k=k}k}right\right}{1+mathrm{e}^left^\a\sum k=k}k}k{
The statistical software now helps us to calculate what we need.
<strong>Neither Pearson's Preference nor Devance is a continuous variable.</strong></p>
<h3>Multiclass Logit model</h3>
<p>Apply to this model without an order of multiple classifications
We basically don't use multiple probit models because multiple normal distributions are not easy to calculate.
The multiple Logit model also has a distinct disadvantage.
<strong>It requires a choice between either category, assuming that the choice is unrelated to the other, i.e. independent of the unrelated category. ]</strong>
Which means that if there's an alternative to the amount available, the most common way to deal with it is to deal with it in advance.<strong>Merge Alternatives</strong>
Model as
&#36;&#36;\ln\left[\frac{P\left(y=j|x\right)}{P\left(y=J|x\right)}\right]=\alpha_{j}+\sum_{k=1}^{K}\beta_{jk}x_{k}.&#36;&#36;
This means that the multiple Logit model is essential. Let's go. <strong>Builds multiple non-repeated classes of Logit models</strong>Made
Essentially exists in multiple models&#36;J-1&#36;An ordinary Logit model
Which means... <strong>Each variable has a set of coefficients and a complete set of analyses.</strong>
I don't think so.&#36;J&#36;Category selected for reference
You can use the formula below to calculate the probability of a certain category.
&#36;&#36;P\left(y=j\mid x\right)=\frac{e^{\alpha_{j}+\sum_{k=1}^{K}\beta_{jk}x_{k&#125;&#125;}{1+\sum_{j=1}^{J-1}\mathrm{e}^{a_{j}+\sum_{k=1}^{K}\beta_{jk}x_{k&#125;&#125;}.&#36;&#36;
The exact calculation process is still done by software</p>
