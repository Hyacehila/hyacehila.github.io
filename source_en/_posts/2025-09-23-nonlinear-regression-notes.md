---
title: 'Nonlinear Regression: Intrinsic Nonlinearity, Polynomial Models, and Nonlinear Least Squares'
title_zh: 非线性回归：内在线性模型、内在非线性模型与多项式回归
date: 2025-09-23 12:00:46 +0800
categories:
- Data Science
- Statistical Modeling & Inference
tags:
- Statistics
- Regression
- Nonlinear Regression
author: Hyacehila
mathjax: true
hidden: true
excerpt: Covers intrinsically linear and nonlinear models, polynomial regression, orthogonal polynomial regression, and nonlinear
  least squares.
description: Covers intrinsically linear and nonlinear models, polynomial regression, orthogonal polynomial regression, and
  nonlinear least squares.
excerpt_zh: 整理内在线性模型、内在非线性模型、多项式回归、正交多项式回归和非线性最小二乘。
permalink: /blog/2025/09/23/nonlinear-regression-notes/
lang: en
translation_key: 2025-09-23-nonlinear-regression-notes
translation_status: machine
translation_source_hash: 12fa08096c7f94c722d5107d8834b18723e2f3107b8bc5106a1cd29114bbf9bb
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<h2>Non-linear regression</h2>
<p>We know that some non-linear models can be converted into linear regression models by some transformations, which we call internal online; some models cannot be converted to linear models, and no matter how they are, we call them internal non-linear models, and we introduce them separately.</p>
<h3>Internal online model</h3>
<h4>Multiple Returns</h4>
<p>Multi-form regression is the most classic inlinear model; we'll only introduce one-dollar multi-linear regression models in the form of the following:
&#36;&#36;y_i=\beta_0+\beta_1x_i+\beta_2x_i^2+\cdots+\beta_kx_i^k+\varepsilon_i,\quad i=1,2,\cdots,n&#36;&#36;
It's easy to know.</p>
<h4>Multiple-transactional Returns</h4>
<p>Multiple returns have a problem, number of times&#36;k&#36;When it was older. &#36;x,x^2...x^k&#36;The proximity of linear correlations, which can cause a great deal of difficulty in solving the normal equation, leads to a greater computational error; the corresponding difference is the difference between the equations we estimate, so we introduced the active multi-turning model.
&#36;&#36;y_i=\beta_0+\beta_1\varphi_1(x_i)+\beta_2\varphi_2(x_i)+\cdots+\beta_k\varphi_k(x_i)+\varepsilon_i,\quad i=1,2,\cdots,n,&#36;&#36;
of which
&#36;&#36;0&#36;begin=0=\vvvv==================================================================================================================================================================================================================================================&amp;j=1,2,\cdots,k,\\\sum_{i=1}^n\varphi_j(x_i)\varphi_q(x_i)=0,&amp;\neq=1,2,\cdots, k.end{cases}
We don't give a specific extrapolation formula. </p>
<h3>Inlinear nonlinear regression</h3>
<p>We'll use only the non-linear minimum two-fold and very large-like models.
Model as
&#36;&#36;Y=f(X_1,X_2,\cdots,X_p,\theta_1,\theta_2,\cdots,\theta_k)+\varepsilon,&#36;&#36;
We can use a very seemingly nuanced approach to squared and minimized the difference, which is,
&#36;&#36;\min\quad Q(\theta)=\sum_{i=1}^n\left(y_i-f(X^{(i)},\theta)\right)^2.&#36;&#36;
The exact method of calculation is not pushed to the point where R provides the corresponding function, which involves the optimal question.</p>
<p>The questions in this article can also be addressed<a href="/en/blog/2023/09/04/linear-regression-basics-notes/">Linear regression base: linear model, minimum 2x2 estimate and regression diagnosis</a>、<a href="/en/blog/2024/03/13/logistic-regression-notes/">Logistic regression: 2-classic because of variables, linear probabilities model and Logit model</a>How the concept of a relatively close read together is developed in different contexts.</p>
<p>The median coefficient for the non-linear regression model is not as intuitive as linear regression, and we would be better advised to explain a regression model by drawing or tabulating changes in target variables when they change, regardless of the formula, which should be the same for the logarithmic transformation regression model.</p>
<h2>Some ideas for a disability analysis.</h2>
<p>Now we've probably understood the meaning of disability, and we can use it to try to deal with the real problem.
According to the proposal, if we create a cross-reference to the proposed value vector,&#36;\hat{y}&#36; The vertical axis is the student disability.&#36;r_{i}&#36;The difference between the two is that the point on the plane should be roughly evenly set in a horizontal zone. Internal
<img src="/assets/images/probability-statistics-notes/nonlinear-regression-notes-01.png" alt="Linear statistical model">
This is a picture that means we're assuming.&#36;e\sim N(0,\sigma^2I)&#36;  It's basically reasonable.
<img src="/assets/images/probability-statistics-notes/nonlinear-regression-notes-02.png" alt="Linear statistical model 1"></p>
<p><img src="/assets/images/probability-statistics-notes/nonlinear-regression-notes-03.png" alt="Linear statistical model 2"></p>
<p><img src="/assets/images/probability-statistics-notes/nonlinear-regression-notes-04.png" alt="Linear statistical model 4">
These three pieces of the gap figure mean that the error and the difference are not valid.
<img src="/assets/images/probability-statistics-notes/nonlinear-regression-notes-05.png" alt="Linear statistical model 5">
Both of these means that our assumptions are problematic, and it's probably a non-linear model, which can be a very large cause of the disability analysis, which requires experience to address.
And of course, we can also create other disability analysis maps, like research disabilities over time, based on a self-variant, and so on, and disability analysis is a very important part of regression diagnosis, and we're not doing enough here.
The problem with disability analysis needs to be addressed gradually.
For example, if you think you lack a retrogressive item, you can start a transformation to make it linear when you think it's non-linear.
In case of error differentials, you can use a variation like Box-Cox (choose to change to high freedom) and apply a weighted minimum two-fold estimate, etc.
<strong>The disability analysis is not just a small part of what we're presenting here, but we'll learn more about disability analysis later, but not now.</strong>
The most common disability analysis is histograms to analyze the disability normality, QQ to analyze the disability normality, SW to test the normality of the disease, travel to analyse the independence of the anomaly, etc.</p>
<p>The disability is determined by the relevant judgment.
The treatment is GLS technologies like Cochrane Orcutt and Prais-Winsten.</p>
<p>Add lags to the model for example we could specify an autoregressive
distributed lag (ARDL) model</p>
<p>Leave model alone but fix the standard errors using Newey-West/HAC
standard error</p>
<p>There's no detail here. There's a chance to add.</p>
<h3>Hartley Test (Hartley)&#39;s F-max Test)</h3>
<p><strong>Formula:</strong></p>
<p>&#36;&#36;
F_{\text{max&#125;&#125; = \frac{\max(s_1^2, \dots, s_k^2)}{\min(s_1^2, \dots, s_k^2)}
&#36;&#36;</p>
<ul>
<li><strong>Meaning</strong>: Calculates the margin between the maximum and the minimum values for the difference in the sample range of each group.</li>
<li><strong>Conditions of application</strong>: Requires equal size of the sample groups (balanced design).</li>
<li><strong>Original assumption (H0)</strong>: The overall variance is equal for all groups (&#36;\sigma_1^2 = \sigma_2^2 = \dots = \sigma_k^2&#36;）。</li>
<li><strong>Deny Field</strong>: When &#36;F_{\text{max&#125;&#125;&#36; When the threshold is exceeded, the original assumption is rejected as not having been the same.</li>
</ul>
<hr>
<h3>Cochran Test (Cochran)&#39;s C Test)</h3>
<p><strong>Formula:</strong></p>
<p>&#36;&#36;
G = \frac{\max(s_1^2, \dots, s_k^2)}{\sum_{i=1}^{k} s_i^2}
&#36;&#36;</p>
<ul>
<li><strong>Meaning</strong>: Calculated the maximum sample variance as a proportion of the sum of all sample differences.</li>
<li><strong>Conditions of application</strong>: Same requirement for the groups of samples.</li>
<li><strong>Original assumption (H0)</strong>: The overall variance of all groups is equal.</li>
<li><strong>Deny Field</strong>: When &#36;G&#36; When the threshold is exceeded, the original assumption is rejected as having a significant bias.</li>
</ul>
<hr>
<h3>Bartlett Test (Bartlett)&#39;s Test)</h3>
<p><strong>Formula:</strong></p>
<p>&#36;&#36;
\chi^2 = \frac{1}{c} \left[ \sum f_i \ln s_i^2 - f_e \ln s_e^2 \right] \sim \chi_{k-1}^2
&#36;&#36;</p>
<ul>
<li><strong>Meaning</strong>: Based on the amount of the calculator that is constructed with a logarithmic value, the difference in the total of multiple normal distributions is tested as equal.</li>
<li><strong>Conditions of application</strong>: Data need to be near-normal; not-normally-sensitive.</li>
<li><strong>of which</strong>：<ul>
<li>&#36;f_i = n_i - 1&#36; It's the first. &#36;i&#36; Freedom of group (%1)&#36;n_i&#36; As No. &#36;i&#36; Group sample volume)</li>
<li>&#36;s_i^2&#36; It's the first. &#36;i&#36; Sample differences for group</li>
<li>&#36;f_e = \sum f_i&#36; It's total freedom.</li>
<li>&#36;s_e^2 = \frac{\sum f_i s_i^2}{f_e}&#36; is the merger difference (weighted average difference)</li>
<li>&#36;c = 1 + \frac{1}{3(k-1)} \left( \sum \frac{1}{f_i} - \frac{1}{f_e} \right)&#36; is the correction factor (for small sample correction)</li>
</ul>
</li>
<li><strong>Original assumption (H0)</strong>: The overall variance of all groups is equal.</li>
<li><strong>Deny Field</strong>: When &#36;\chi^2&#36; Value greater than &#36;\chi_{k-1}^2&#36; When the top-side fraction of the distribution is selected, the original assumption is rejected.</li>
</ul>
<ul>
<li><p>00:43.
The difference is the difference between the data to be compiled and the real data to be studied, and the appropriate display and study of the difference can draw attention to certain pre-neglected systemic behaviours in the data.</p>
<p>  Research disability</p>
<p>  Data is a superseding of the composition and the residuals, and we want to keep analysing the residuals and revise our proposeds.</p>
<p>  The way back functions are used may be wrong, and we're thinking about the pattern from the disability analysis.
Some informal criteria are: no pattern in the disability, all caused by random error, smaller margin squared, better proposed advantager r</p>
<p>  We often use the assumption of disability in mathematical statistics, where this well-defined distribution of disability is best used to achieve the desired alignment, and vice versa to examine whether the proposed margin is consistent with the assumption, the most important of which is the study of the disability map.</p>
<p>  A more classic residual analysis is an off-group site test.</p>
<p>  So long as we use a robust method of extreme tolerance, the model will largely ignore the presence of the discrete points, and the residual analysis will be a good way of exposing the presence of the disempowerment points.</p>
<p>  The difference in calculations is a complete set of data that we can continue to put into the analysis of eda, do what we do in the description statistics, study distribution, concentration separation, shape, etc.</p>
<p>  Most of our methods of alignment tend to minimize a function of the disability, like Les, when we study the pattern of the disability, we can change functions like the weighted disability squared and the difference of the difference of the difference.</p>
</li>
</ul>
<h2>Restrictive cube</h2>
<p>In scientific research, we often construct regression models to analyse the relationship between variables and variables. A major assumption of most regression models is that the variables are linearly related.</p>
<p>When there are non-linear relationships, we need to use some non-linear means of return. For example, multiple regression methods or the question of converting a continuous variable into an discrete variable for classification.</p>
<p>Unfortunately, the direct construction of multiple returns may have problems of overcompatibility, co-linearity and so forth. Dispersion factors can result in loss of part of the information and the location of nodes is more complex than the choice of categories.</p>
<p>I'm not sure.<strong>"Restricted cubes."</strong>One of the most common methods of analysing non-linear relationships is the RCS.</p>
<p>The stylist (spline) is originally a flexible piece of wood or metal used to draw smooth curves.<strong>The root of a bar curve is a multi-segment function</strong>, this function is limited to certain control points and is called ""<strong>Nodes</strong>, the type of the line curve is determined by the number and location of the nodes, the multiple types, and the number and location of nodes.<strong>Cube</strong>is the function of a three-time polygon.<strong>Limits</strong>is an additional requirement based on the regression of the sample: the sample function is in two compartments at both ends of the variable data range <code>[X1,X2)</code> and <code>(Xn-1,Xn]</code> is a linear function.</p>
<p>The above presentation allows us to understand what the Cube is about, he is just a non-linear way of returning.</p>
<p><strong>The number of RCS nodes is more important than location.</strong>I'm sorry. Because the choice of nodes is related to freedom, more nodes can be taken when the sample is larger. But the more nodes, the more freedom, the more complex models, the harder to solve.</p>
<p>It is suggested that the number of nodes be 4 and that the proposed effect of the model be better, i.e., it is possible to balance the smoothness of the curve and avoid the reduction of precision caused by over-choice. When the sample is larger, five nodes are a better option. Small sample (&#36;n)&lt;30) you can choose three nodes. When the number of nodes is 2 the condensed curve is a straight line. Most researchers recommend 3-5 nodes.</p>
<p>When drawing RCS, we often draw 95% of CIs, and RCS can only deal with one-on-one returns with great limitations, but in many areas it is very practical. </p>
<h2>Cox Return Profile</h2>
<p><strong>Cox regression, also known as Cox Proportional Risk Model (Cox Production-Hazards Model) or Survival Analysis Model, we can also call it Cox Model.</strong>This is a semi-parametric regression model, first introduced in 1972 by British statisticians, D.R. Cox, and a statistical method widely used in the field of survival analysis. It is used to study the time of the event, which can be of any kind, such as the time of survival, the time of unemployment, the time of recurrence of the disease, etc. The main objective of Cox's return is to analyse the risks or risks of the event and to explore the factors associated with these risks.</p>
<p>That is, as a regression model, our variables are time indicators of the time of survival, the time of recurrence of disease, which means the risk of the event. We want to study this risk, as well as the factors that influence it (from the variables).
<em>The end-of-class 2 variable can also be used as the Cox regression factor, but it's not used that widely.</em></p>
<p>The greatest feature of survival time is normal.<strong>Not matching normal distribution</strong>(usually right-side distribution) and may contain<strong>Ending Data</strong>(Because some individuals are still alive at the end of the study), the traditional regression method cannot process this data, which is what Cox returns mean.</p>
<p>The Cox model is based on a proportional risk assumption that there are different individuals.<strong>The risk ratio is constant.</strong>, that is, risks will not change over time. This assumption allows the model to estimate the relative risk of different factors without the need to know the specific form of the risk function.</p>
<h2>The Cox Returning Principle</h2>
<h3>Basic concepts</h3>
<p>We first have a few basic concepts:</p>
<ul>
<li>Survival Time: indicates the time between a particular point of entry (e.g., beginning of treatment, time of diagnosis, etc.) and the occurrence of an event (e.g. death, relapse).</li>
<li>Survival Function: is the probability of survival before a certain point in time, usually using&#36;S(t)&#36;- Show.</li>
<li>Hazard function (HazardFunction): This is the core concept of the Cox model, which is the probability of an event occurring within a unit time near a point of time, i.e. the instant risk of a given point of time, usually used&#36;\lambda(t)&#36;- Show. Risk (Hazard) indicates the probability of an event occurring over a period of time.</li>
<li>Covariates: Self-variant variables that may affect the lifetime or incidence of events, such as age, sex, treatment, etc.</li>
<li>Cox scale risk assumption: The core assumption of the Cox model is that the hazard function is a time function, but the hazard function ratio is constant between individuals.</li>
</ul>
<p>Cox Retaliation Model.<strong>Basic form</strong>The text reads as follows:
&#36;&#36;\lambda(t)=\lambda_0(t)\cdot e^{\beta_1x_1+\beta_2x_2+\ldots+\beta_px_p}&#36;&#36;
of which &#36;\lambda(t)&#36; Time.&#36;t&#36; . the dangerous function, &#36;\lambda_0(t)&#36; is the baseline hazard function, which is usually the average hazard function for the entire sample.&#36;x_i,\beta_i&#36; It's the normal regression from variables and the corresponding coefficient.</p>
<p>Characteristics of Cox's return: </p>
<ul>
<li>Cox regression is a semi-parametric model, which does not require a baseline risk function&#36;\lambda_0(t)&#36;A specific assumption is made and therefore very flexible.</li>
<li>Survival data for right-sided distributions can be processed without the need to meet normal distribution assumptions.</li>
<li>Allow analysis of the impact of multiple co-variant on the lifetime and estimate their relative risk.</li>
<li>The data reviewed and the end-of-pipe data can be processed.</li>
</ul>
