---
title: 'Generalized Linear Regression: Categorical Predictors, Dummy Variables, and Fixed Effects'
title_zh: 广义线性回归：分类自变量、虚拟变量与固定效应
date: 2024-05-24 17:32:12 +0800
permalink: /blog/2024/05/24/generalized-linear-regression-notes/
categories:
- Data Science
- Statistical Modeling & Inference
tags:
- Statistics
- Regression
- Generalized Linear Models
excerpt: Covers categorical predictors, dummy variables, fixed effects, instrumental variables, robust regression, and GLM-related
  topics.
description: Covers categorical predictors, dummy variables, fixed effects, instrumental variables, robust regression, and
  GLM-related topics.
lang: en
translation_key: 2024-05-24-generalized-linear-regression-notes
translation_status: machine
translation_source_hash: 7f15f19630927537bfa79bef423ee1ed26bbf7a8fc69fac3d71c88dd3d9b8012
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<h2>Overview</h2>
<p>In the broad linear regression chapter, we are not limited to the study of traditional GLM models, but to the introduction of many linear transformations, which are presented in the broad linear regression chapter.</p>
<p>In particular, the Logit series model using defined variables will be presented separately <a href="/en/blog/2024/03/13/logistic-regression-notes/">Logistic returns</a></p>
<p>The contents of this chapter include:</p>
<ul>
<li>Returns model for classification from variables</li>
<li>The fixed effect returns to the tool variable returns</li>
<li>Steady return technology</li>
<li>Wait.</li>
</ul>
<h2>Classification of variables and virtual variables</h2>
<p><strong>Classification Variables</strong>Values are of limited type, e.g. gender: male, female. The classification variable cannot be used directly in the regression model, i.e. to use 1 for men, to use 0 for women, and this 1 and 0 can still only be used to distinguish categories. If they are used without processing, the logic and results of the entire model are incorrect.</p>
<p>In programming tools such as R, the type of variable classified is a factor, not a value.
When we incorporate the factor variable into the regression model, the R language automatically helps us with the final regression analysis.
Now what we're trying to tell you is how he's handling this factor variable internally.</p>
<h3>One-hot encoding</h3>
<p>One-hot coding is a very natural idea, using a bit to indicate a possible category, and a variable cannot normally belong to more than one category at the same time, so only one bit is 1 and none other means that it is not.</p>
<p>Use one-of-K or one-hot encoding (one-hot encoding). It can take every one of them.&#36;m&#36;The characteristics of the category are converted to&#36;m&#36;Binary Characters</p>
<p>We can get these 01s right into regression models or more sophisticated machine learning models.</p>
<h3>Virtual Encoding</h3>
<p>The so-called "dummy variables" are actually similar in principle to a single thermal code.</p>
<p>The biggest difference between virtual coding on a single thermal coding is a degree of freedom between selection and reduction of reference classes, each having&#36;m&#36;The characteristics of the category are converted to&#36;m-1&#36;The remaining feature of the classification will be used as a reference, and the other features are meant to be compared with the reference class, and their coefficients are more explanatory.</p>
<p>The effect of the classification variables on regression models is to make comparisons between clusters, and only to act as group comparisons.</p>
<p>For one.&#36;q&#36;Horizontally classified variables that we have to create if the regression model has a cutout entry in betweenept &#36;q-1&#36;And the other items will use this item as a basis for their own meaning because <strong>If we add a dummy variable to this regression model with a cut-off, and add a factor horizontally, we can cause a comprehensible linear problem, which leads to a failure to solve.</strong></p>
<p>When the model does not have an amplitude item, set a horizontal number of virtual variables, but it's almost impossible for the model to have an amplitude item.</p>
<p>At this point, we randomly select one volume as the baseline, and other virtual variables can be seen as the result of those variables when they are based on this quantity.</p>
<p><strong>Different virtual variables affect all regression factors and cut-off items simultaneously.</strong></p>
<h3>Effect Encoding</h3>
<p>The effect coding is also the way to select the reference class.&#36;-1&#36;</p>
<p>At this point, the amplitude item represents the overall average, the coefficients for each variable reflect the difference before his or her overall average, and in the effect coding, no separate feature is presented as a reference category, and we need to calculate it separately, which is the sum of the opposite of the coefficients for all other categories.</p>
<p>Because of the computational advantages of the large-scale coefficient matrix, we do not use the effects code very often.</p>
<h3>Order dumb variable</h3>
<p>We also have a sequenced mute variable, which preserves the sequence of the data while coding, without any difference between the idea and the one-heat code.</p>
<table>
<thead>
<tr>
<th>Features</th>
<th>Encoded</th>
</tr>
</thead>
<tbody><tr>
<td>bad</td>
<td>&#36;(1,0,0)&#36;</td>
</tr>
<tr>
<td>normal</td>
<td>&#36;(1,1,0)&#36;</td>
</tr>
<tr>
<td>good</td>
<td>&#36;(1,1,1)&#36;</td>
</tr>
</tbody></table>
<p>We've got a way to code the target rate for more fine particles. We'll learn later.</p>
<h2>Fixed Effect Return</h2>
<p>The technology of multiple returns is very powerful, but the issue of missing variables has been a very important issue of return technology, and he will significantly influence our estimates of effectiveness.</p>
<p>This chapter presents the technology for re-entry of panel data, which, by the type of panel data, allows us to control those variables that are not actually observed and to explain the omission effect.</p>
<p>If the missing variable does not change over time (i.e. the fixed amount of the category) or if the missing variable changes over time only, we have panel data that can be used to return;</p>
<h3>Panel Data</h3>
<p><strong>Panel data returns, requiring both cross-section data and time series data</strong> For example, government policy data sets on the number of traffic accidents and alcohol taxes caused by drunk driving in several states of the United States over the years. It's also a very common type of data. <a href="/en/blog/2023/11/05/descriptive-statistics-and-visualization-notes/">Descriptive statistics and visualization</a> Section on Measurement</p>
<p>When describing cross-section data, we use subscripts to represent individuals, for example.&#36;Y_i&#36;is the number of the variable Y&#36;i&#36; An individual. In the description of panel data, we need additional symbols to track both individuals and periods. For this, we use two subscripts instead of single subscripts: first of all, we need to get to the bottom of this.&#36;i&#36; It's an individual; second,&#36;t&#36; Indicates the period of observation. Therefore...&#36;Y_it&#36;Organisation&#36;n&#36; No. of individuals&#36;i&#36;An individual.&#36;T&#36;No. of the period&#36;t&#36;Variables observed over time&#36;Y&#36;value.</p>
<p>Other terms on panel data describe the absence of observations. Balanced Panel refers to the availability of all observations; that is, variables are observed for each individual and for every period. If data is missing for at least one period or at least one individual, the panel is called the non-balanced panel</p>
<h3>Two-time panel returns</h3>
<p>By comparing the two-time panel data, we can deal with some of the missing variables involved in the regression and fix the effects of non-observable variables that change between individuals and over time.</p>
<p>You!&#36;Z_i&#36;No. No.&#36;i&#36;A unique individual that doesn't vary over time from a non-observable variable, we can build a regression equation to
&#36;&#36;\widehat{FatalityRate_{it&#125;&#125;=\beta_0+\beta_1BeerTax_i+\beta_2Z_i+u_i&#36;&#36;
Because we mentioned the effects.&#36;Z_i&#36;It doesn't change over time, and we have two stages of panel data, so we're not.
&#36;&#36;FR_{i1988}-FR_{i1982}=\beta_1(BeerTax_{i1988}-BeerTax_{i1982})+u_{i1988}-u_{i1982}&#36;&#36;</p>
<p>Discrepancies are set in such a way that the effects of non-observable variables that do not change over time are eliminated, when estimates of the regression analysis of the equations that follow the differentials allow for the use of conventional estimation methods. It can also be analysed in a way that is directly based on conventional models.</p>
<p>This “pre-post” analysis can be used when data are observed in two different years. However, our data sets contain observations for seven different years, and it would be unwise to discard these additional data that may be useful. This leads to a return of the fixed effects back there.</p>
<h3>Fixed-effect regression model</h3>
<p>The return of fixed effects is a method of control panel data that changes the missing variables as individuals (states) do not change over time. Unlike the “pre-ex post” comparison, fixed-effect regression can be used in situations where each individual has observations for two or more periods</p>
<p>The fixed effect regression model has&#36;n&#36;A different cut, one for an individual. These amplitudes can be expressed as a collection of binary (or indicator) variables. These binary variables absorb the effects of all missing variables that vary between individuals but do not change over time. </p>
<p>The basic form of the return equation is one of them.&#36;Z_i&#36;It's the fixed amount of individual observations that don't change over time.
&#36;&#36;Y_u=\beta_0+\beta_1X_{it}+\beta_2Z_i+u_{it}&#36;&#36;</p>
<p>If we build each individual into a model with its own cut-off,
&#36;&#36;Y_{it}=\beta_1X_i+\alpha_i+u_{it}&#36;&#36;
Of course, we're asking that the regression factor be consistent with the individual regression equation, which is the fixed effect regression model.</p>
<p><strong>We can't use the OLS at this time. We need more changes.</strong></p>
<p>The cut-off range of a given individual in a fixed-effect regression model can also be expressed as a binary variable for a single individual, i.e. we can introduce virtual variables to harmonize the regression equation as follows:
&#36;&#36;Y_u=\beta_0+\beta_1X_u+\gamma_2D2_i+\gamma_3D3_i+\cdotp\cdotp\cdotp+\gamma_nDn_i+u_i&#36;&#36;
At this point, the introduction of virtual variables will also consider the problems that lead to multiple co-linearity, without losing sight of the fact that this paper "Segregated self-variant and virtual variable issues" Section
At this point, the model can be reclassified as a problem for traditional regression models.</p>
<p>It's very easy to introduce this type of return into the problem of multiple returns.
&#36;&#36;Y_{it}=\beta_1X_{1,it}+\cdotp\cdotp\cdotp+\beta_kX_{k,it}+\alpha_i+u_{it}&#36;&#36;</p>
<h3>Time fixed effect returns</h3>
<p>For each individual, the fixed effects represent the effects of variables that do not change over time, but change over time. Similarly, the fixed effects of time represent the effects of variables that do not change over time. It's not appropriate to have model effects if you miss the time back.</p>
<p>It's natural to give a model that only has time effects, or to use one dollar for example.
&#36;&#36;Y_{it}=\beta_1X_{it}+\lambda_t+u_{it}&#36;&#36;
Of which&#36;\lambda_t&#36; Called Time Fixed Effects</p>
<p>Just as an individual fixed-effect regression model can be used, one or more binary indicator variables can be used. &#36;T-1&#36;A binary indicator variable indicates a time fixed effect regression model:
&#36;&#36;Y_{it}=\beta_0+\beta_1X_{it}+\delta_2BT_t+\cdotp\cdotp\cdotp+\delta_TBT_t+u_{it}&#36;&#36;
We're ignoring a problem that avoids convergence.</p>
<p>Time fixation allows us to eliminate the deviations from the missing variables, such as the national introduction of safety standards, over time but consistent across all states in a given year.</p>
<p>For time effects, reference may be made<a href="/en/blog/2024/01/30/linear-time-series-analysis-notes/">Linear Time Series Analysis</a> <a href="/en/blog/2024/05/06/univariate-financial-time-series-analysis-notes/">Financial time series analysis (one dollar)</a> It gives us better time-effect analysis tools.</p>
<h3>Mixed time fixed and fixed effects</h3>
<p>If some of the missing variables are fixed for time but vary with the individual (e.g. cultural norms) while others are fixed for the individual but change over time (e.g. national security standards), it is appropriate to attract both individual and temporal effects.</p>
<p>Integration of individual and time fixed effect regression models (Entity and Time Fixed Effects ReviewModel) as
&#36;&#36;Y_{it}=\beta_1X_{it}+\alpha_i+\lambda_t+u_{it}&#36;&#36;</p>
<p>We can still break down a lot of virtual variables to solve the problem.</p>
<h3>Panel regression factor estimate</h3>
<p>We know that the equation ahead can be determined using the traditional OLS estimated realization factor, but in practice this OLS regression is redundant or some software packages cannot be implemented when there are a very large number of individuals.</p>
<p>As a result, econometric software has special treatments for fixed-effect regression model OLS estimates. These special treatment methods are estimated to be OLS with the return of the whole binary variable, but they operate faster by using some mathematically simplified algebras that are suitable for the return of fixed effects.</p>
<p>Here, we don't explain the problem of the algorithm, just as a hint, we can consider a special package for the acceleration of the algorithm when we take the panel data back.</p>
<p>A special panel regression estimation method is
In the balance panel,&#36;X&#36;The coefficient can be used first.&#36;Y&#36;and&#36;X&#36; Less the average of the individual and the time, then estimate the centralized.&#36;Y&#36;It's a centralization.&#36;X&#36; The multiple regression equation. This algorithm, which is usually used in the regression software, avoids the construction of a collection of all the binary indicator variables.</p>
<h3>Standard error for panel regression</h3>
<p>Because the panel returns to include time series data, it must be self-relevant, which is the general feature of time series data; therefore, the omission factor associated with the regression error is likely to contain time variability, and they are self-relevant.
<strong>In general, as long as the omission factor is self-relevant, the error will be self-relevant.</strong></p>
<p>If the regression error is self-relevant, then the original heterogeneity deviation standard error is no longer applicable, just as the original homogeneity difference SE is no longer applicable.</p>
<p>For potential heterogeneity-related errors of the individual, a cluster standard error should be used, with his exclusive return to the panel data, still assuming that the individual is not related, and that he can also be used for the problem of heterogeneity and the difference.</p>
<p>Finally, the return of panel data requires panel data, which is difficult to obtain, so we need to consider other ways of dealing with non-observable variables, i.e., the return of tool variables in the next chapter.</p>
<h2>Tool variable returns</h2>
<h3>Tool Variable Return Introduction</h3>
<p>The return of panel data can only address some of the disability and self-variant problems, and sometimes we cannot collect panel data. Similarly, the return of fixed effects has no effect on the dichotomy of the two-way causal factors, and we need new ways to address this.</p>
<p>We still want to solve the regression variable.&#36;X&#36;With disability&#36;e&#36;But at this point in time, we don't think about how this is happening, but we measure him directly by a variable, which is the return of tool variables.</p>
<p>The return of a tool variable (IVR) is a general method to obtain consistent estimates of regression functions when the variable and error items are relevant. Let's put&#36;X&#36;The change is considered to consist of two parts, one not related to the disability and the other related to the disability; we use the separated tool variable (IV) to separate those changes associated with the disability, thus allowing for a consistent estimation of the regression equation.</p>
<p><strong>The return of the tool variable is essentially the introduction of a new variable, but the variable is not used for estimation as before.</strong></p>
<h3>Tool variable estimates for single regression and single tool variables</h3>
<p>We start with individual regressions from variables.</p>
<h4>Tool variable regression model and internality</h4>
<p>A standard general regression model takes the form of
&#36;&#36;Y_i=\beta_0+\beta_1X_i+u_i,i=1,\cdots,n&#36;&#36;
Current Variable&#36;X&#36;And disability.&#36;u&#36;We need to find a way to fix this.</p>
<p>Tool variable returns with a specific term to distinguish between the general error item&#36;u&#36;is a relevant or unrelated variable. The variables associated with the error items are called inner variables (Endogenous Variables), while those unrelated to the error items are called external variables (Exogenous Variables).</p>
<p>Effective tool variables must meet two conditions, i.e. the tool-related condition (Instruation Relevance condition) and the tool-generated condition (Instruation Analysis condition):</p>
<ul>
<li>Tool variables relevant: corr&#36;(Z_i,X_i)\neq0&#36;</li>
<li>Tool variable external: corr&#36;(Z_i,u_i)=0&#36;</li>
</ul>
<p>If the tool is relevant, then the change in the tool&#36;X_i&#36;Change is relevant. Besides, if the tool is alien, then the tool variable captures it.&#36;X_i&#36;This part of the change is also alien.</p>
<p>Therefore, relevant and external tools can capture&#36;X_i&#36;The external changes. This external change can be used in turn to estimate the overall coefficient&#36;\beta&#36; </p>
<h4>Minimal two-stage multiplication TSLS (one dollar)</h4>
<p>If Tools&#36;Z&#36;The condition of relevance and externality is met, then the coefficient&#36;\beta_{\mathrm{l&#125;&#125;&#36;Estimates can be made by means of a tool variable estimate called Two Stage List Squares, TSLS.</p>
<p>As in its name, the minimum two-stage estimate is calculated in two phases. Phase I will be&#36;X&#36;Decompose into two parts: the one relating to and causing problems with regression errors, and the other not related to errors that do not cause problems. Phase II uses unproblematic parts to estimate&#36;\beta_{1}&#36; </p>
<p>Phase 1 from X and&#36;Z&#36; The general return begins:
&#36;&#36;X_i=\pi_0+\pi_1Z_i+v_i&#36;&#36;
The function of this regression is to decompose the original self-variant. of which cut-outs and&#36;Z&#36; It is external, and this part has nothing to do with the error. The last part.&#36;v_i&#36;It's about the part of the original variable that caused the problem.</p>
<p>The idea behind the minimum double is the part where the use doesn't cause problems.&#36;X_i&#36;i.e.&#36;\pi_0+\pi_1Z_i&#36;, while ignoring&#36;\upsilon_i&#36;I don't know. The only complication is that...&#36;\pi_0&#36;and&#36;\pi_1&#36;The value is unknown, so...&#36;\pi_0+\pi_1Z_i&#36;Can't be calculated.</p>
<p>Therefore, the first stage of the minimum double multiplier applies OLS to the previous equation, using the OLS regression projection, i. e.&#36;\hat{X}_i=\hat{\pi}_0+\hat{\pi}_iZ_i&#36;, where&#36;\hat{\pi}_0&#36;and&#36;\hat{\pi}_1&#36;For OLS estimates.
The second stage of the minimum quadrilateral is very simple: using OLS&#36;Y_i&#36;Yeah.&#36;\hat{X}_i&#36;Come back. The estimate for the second stage of return is the lowest 2x2 estimate.<em>0^\mathrm{TSLS}&#36;和&#36;\hat{\beta}</em>\mathrm{l}^\mathrm{TSLS}&#36;。</p>
<h4>Sample distribution of TSLS</h4>
<p>The precise distribution of TSL estimates in small samples is complex. However, as with OLS estimates, their distribution is relatively simple in large samples: TSL estimates are consistent and subject to normal distribution.</p>
<p>Although two phases of TSLS make the estimates seem more complex, only one regression variable is considered &#36;X&#36; and a tool variable&#36;Z&#36;The TSLS estimate has a simpler formula, as we assume in this chapter. You! &#36;s_{ZY}&#36;Yes&#36;Z&#36; and&#36;Y&#36;♪ the sample is the difference between, and ♪ &#36;x_{ZX}&#36;Yes&#36;Z&#36; and &#36;X&#36; The difference between the samples. TSLS estimated to contain a one-dollar tool variable
What's that?<em>1^{TSLS}=\frac{s</em>{ZY&#125;&#125;{s_{ZX&#125;&#125;&#36;&#36;</p>
<p>More extrapolation work is done with specific instructions from software, and no further description is provided here.</p>
<h3>General tool variable returns</h3>
<h4>General tool variable returns to model form</h4>
<p>The general tool variable regression model contains four forms of variable: the explained variable &#36;Y&#36;;the internal regression variable that causes the problem is recorded as&#36;X&#36;; other regression variables referred to as including external variables as&#36;W&#36;; and tool variables&#36;Z&#36;。</p>
<p>Generally, there may be multiple built-in regression variables.&#36;X&#36;), multiple containing external variables (&#36;W&#36;) and several tool variables (&#36;Z&#36;)。</p>
<p>To make it work, the tool variable ()&#36;Z&#36;) number must be at least internal regression variable ( )&#36;W&#36;Just as much. In the previous section, only in-kind regression and monist tool variables were considered. For an IUD variable, a (at least) tool variable is required. Without it, we will not be able to calculate the tool variable estimate: there is no first-stage regression in TSLS.</p>
<p>The relationship between the number of tool variables and the number of internal regression variables has their own specialized terminology. If the number of tools&#36;m&#36;equals the number of internal regression variables (&#36;k&#36;), which is&#36;m=k&#36;, which states that the regression factor is correctly recognized. If the number of tools is greater than the number of inner regression variables, &#36;m&gt;k&#36;,则称系数是过度识别 (Overidentified) 的。如果工具的个数小于内生回归变量的个数，即&#36;m&lt;k, is unrecognized.</p>
<p>If it is to estimate the return of the tool variable, the coefficient must be correctly or over-identified.</p>
<p>Therefore, we should present the model as
&#36;&#36;Y_i=\beta_0+\beta_1X_{1i}+\cdotp\cdotp\cdotp+\beta_kX_{ki}+\beta_{k+1}W_{1i}\cdotp\cdotp\cdotp+\beta_{k+r}W_n+u_i,i=1,\cdotp\cdotp,n&#36;&#36;</p>
<h4>Minimum two times two stages (general)</h4>
<p>If there's only one variable, then it should take the form of
&#36;&#36;Y_i=\beta_0+\beta_1X_i+\beta_2W_{1i}+\cdotp\cdotp\cdotp+\beta_{1+r}W_{ri}+u_i&#36;&#36;
The corresponding minimum two times the first stage should be
&#36;&#36;X_i=\pi_0+\pi_1Z_{1i}+\cdotp\cdotp\cdotp+\pi_mZ_{mi}+\pi_{m+1}W_{1i}+\cdotp\cdotp\cdotp+\pi_{m+r}W_n+v_i&#36;&#36;
In the second phase, all we have to do is use the first-phase projections.</p>
<p>Extending to the issue of multiple returns requires only the minimum of two times the first of multiple self-variant variables in the first phase, with the second remaining unchanged.</p>
<p>More about the hypothetical tests, the error estimates are not presented here.</p>
<h3>ToolVariance Validity Test</h3>
<p><strong>Whether or not the tool variable returns works in the given application depends on the effectiveness of these tools</strong>, the invalid tool variable produces meaningless results. It is therefore particularly important to evaluate the effectiveness of a set of tools in a given application.</p>
<h4>Tool Variable Relevance and Weak Tool</h4>
<p>Tool-relevance conditions play a very delicate role in the return of tool variables. One way to see the relevance of the tool is to compare its role to sample capacity: the more relevant the tool variable - the more it is.&#36;X&#36;The change is explained by the tool - meaning that more information can be used in the return of the tool variable.</p>
<p>Less explanation.&#36;X&#36;Change tools are called weak tools. In the case of cigarettes, the distance between the state and the place where the cigarette is produced can prove to be a weak tool: Although the distance increases transport costs (thus moving the supply curve to the left and increasing even prices), transport costs account for only a small portion of cigarette prices because of the low weight of cigarettes. Thus, the part of price changes that is explained by transport costs, i.e. the distance between origin and origin, may be very small.</p>
<p>Let's talk about relevance and weak tools in this section.</p>
<p>In the case of weak tools, the use of normal distributions to approximate the sample distribution of ISLS estimates is not optimal. Thus, despite being a large sample, there is also a lack of theoretical justification for the use of normal statistical extrapolations. In fact, if the tool is weak, the TSL estimate is highly biased in the direction of the OLS estimate.</p>
<p>In addition, the probability of the true value of the coefficient contained in the 95 per cent confidence interval, constructed using a standard error of 1.96 times the TSLS estimate, may be much smaller than 95 per cent. In short, if the tools are weak, TSL is no longer reliable.</p>
<p>Very few cases are encountered where the tool variables are completely irrelevant, but the extent to which the tool variables are relevant can be considered to be comparable. We have an empirical rule: <strong>Assuming a return equation of 0 for a minimum 2x2 return in phase I, if the F count is greater than 10, then there is no need to worry about weak tools</strong></p>
<p>If we have few powerful tools and many weak ones, we better ignore the weaker tool variables; while this will increase the standard error of TSLS, the SE itself, which contains the weak tools, will not work.</p>
<p>If the coefficients are correctly identified, even over-identified might not find so many powerful tools available, then we cannot ignore the weak ones. Find some new tool variables or use IVR options for weak tool files</p>
<h4>Externality of tool variables</h4>
<p>If the tool is not external, then TSL is inconsistent: i.e. TSL estimates yield other values based on probability to non-overall regression factors. After all, the idea that the tool variable returns is that the tool contains an error item.&#36;u_i&#36;It doesn't matter.&#36;X_i&#36;Information on changes.</p>
<p>Is it statistically possible to test the externality of the tool? Yes, but possibly not at the same time. On the one hand, when the coefficient is correctly identified, it is not possible to test the tool as an external hypothesis. On the other hand, if the coefficient is over-identified, then the over-identification constraint can be tested, assuming that there are enough effective tools to identify the factors of interest, so that the “additional” tools are external.</p>
<p>First of all, consider the exact circumstances, when you have the same number of tools and variables as the number of internal regression variables. In that case, it was not possible to conduct statistical tests of the assumptions that the tool was in fact external. This means that empirical evidence cannot be used to answer the question of whether the tool variable meets the external constraints.</p>
<p>In such cases, the only way in which the evaluation tool is external is through expert advice and your experience on the issue.</p>
<p>The externality of the assessment tool variables must be judged professionally on the basis of individual knowledge of the application of the practice. However, if there are more tools than internal regression variables, there are statistical tools that can help with the process: the over-identification of binding tests.</p>
<p>The idea of over-identification tests is to use multiple tool variables for different TSLs, and if they are external, the estimated results must be close, and if they are not, it means that at least one variable has an external problem. We have a special hypothetical test to solve this problem.&#36;J&#36;Statistics</p>
<h3>Search for valid tool variables</h3>
<p>In practice, the most difficult part of the tool variable estimates is to find both relevant and external tool variables. Two approaches are presented here, reflecting two different perspectives of econometric and statistical modelling.</p>
<p>The first approach is to find tools based on economic theory. For example, Philip Wright ' s knowledge of the agricultural market economy prompted him to find a tool to move the supply curve without moving the demand curve; it also led him to consider the weather conditions in the agricultural region. However, economic theory is abstract and often fails to take into account the small and detailed differences in the data concentration of a specific analysis, so the methodology is not always effective.</p>
<p>The second construction tool variable is to find it.&#36;X&#36;Some of the exogenous causes of change are in fact caused by a random phenomenon leading to the movement of internal regression variables. Such an approach often requires a full understanding of the issues studied, while the details of the data are carefully explored, preferably through specific cases.</p>
<p>We'll use three cases to see how other scholars found the right IV.</p>
<h4>Is keeping criminals in prison a deterrent?</h4>
<p>This is a question that only economists can ask. After all, criminals are unable to commit crimes outside prison while serving their sentences, while arresting some of them also helps to deter others from committing crimes. But the magnitude of the combined effect — that is, the change in crime rates caused by the 1 per cent increase in the number of people in prison — is an empirical problem.</p>
<p>One strategy for estimating this effect is the return of crime rates (number of offenders per 100,000 population) to prison rates (number of prisoners per 100,000 population) based on data at appropriate levels of jurisdiction (e.g. state of the United States).</p>
<p>This return may include certain measures of economic conditions. When the overall economic situation deteriorates, crime increases. However, there may be serious problems arising from potential two-way causal deviations: If the crime rate increases and the police act impartially, more prisoners will emerge. On the one hand, an increase in the number of prisoners would lead to a decrease in the crime rate; the return of OLS, which established the crime rate against the prison rate, would estimate the complex combination of the two effects, due to the two-way causality. And the problem can't be solved by finding better control variables.</p>
<p>However, we can overcome the two-way causal bias by finding the right tool variables and using TSLS. The tool must be related to the rate of imprisonment (the relevant conditions must be met), but not to the error in the crime rate equation of interest (the external conditions must be met).<strong>In other words, the instrument must be capable of influencing the rate of incarceration, but at the same time not related to any determining factor in the unobserved crime rate.</strong></p>
<p>How do we find tools and variables? Because the construction of prisons is time-consuming, short-term content restrictions may force states to release prisoners early or reduce prison rates. For this reason, Levitt, 1966, considered that legal proceedings aimed at reducing prison overcrowding could be used as a tool variable, while he undertook an empirical analysis based on data from the United States State panels of 197211993.</p>
<p>We can feel that this observation of prison overcrowding has nothing to do with crime rates, but it does have a bearing on the rate of incarceration. At the same time, it is relevant and external. Multiple tool variables can be obtained if we continue to dig up more observations of prison overcrowding on this side.</p>
<h4>Does reducing class size increase test scores?</h4>
<p>As we have done in the case of empirical analysis, small classes produce richer schools, while their students receive more intensive learning opportunities both inside and outside the school.</p>
<p>We have overcome the threat of missing variables by controlling variables such as student wealth, English language proficiency and multiple regressions. However, skeptics will wonder if we are in position: if we still miss something important, then our estimates of the class size effect will remain biased.</p>
<p>This potential omission of variables can be overcome by the way they are correctly controlled, but if they are not available (e.g., difficult to measure out-of-school learning opportunities), the tool variable regression should be used as an alternative. The return requires a tool variable related to class size (relevance), but not to test achievement determinants that constitute errors (e.g. parents ' interest in learning, opportunities for out-of-school learning, teacher level and school facilities, etc.) (externality).</p>
<p>Hoxby, 2000 suggests the use of biological theory. Due to random fluctuations in the timing of birth, the number of new entrants to kindergarten varies from year to year. Although the actual number of children entering kindergarten may be internal (there have been recent reports that schools may affect parents' ability to send their children to private schools), she points out that the number of potential children entering kindergarten — that is, the number of children aged 4 in the school district — is mainly due to random fluctuations in the date of birth of the child.</p>
<h4>Can a positive effect on heart disease prolong life?</h4>
<p>Active treatment of patients with heart disease (professionally referred to as acute myocardial infarction, AMI) helps to prolong life. Before new treatments can be extended to general applications, clinical experiments, i.e. a series of random control experiments designed to measure their effects and side effects, are required. However, good clinical trials are one thing, and the effects in practical applications are another.</p>
<p>A natural starting point for studying the effectiveness of the treatment is to compare the situation of patients who have received the treatment with those who have not. This entails establishing a re-entry of the patient ' s life expectancy to the binary treatment variable (whether the patient has undergone a heart tube implant) and other control variables affecting mortality (e.g. age, weight, other health conditions, etc.).</p>
<p>The overall coefficient of the indicator variable is the increased life expectancy of the patient receiving the treatment. Unfortunately, the OLS estimate is biased: the operation is not “randomly” performed on the patient; rather, it is performed only when the doctor and the patient believe it may be effective. If the decision depends in part on non-observed factors that are not centralized but are associated with health outcomes, then the treatment decision will be related to regression error. If the most healthy patients are treated, the OLS estimate will be biased (the treatment is related to missing variables), and the treatment will appear to be more effective than its real effects.</p>
<p>We need a tool variable related to treatment but not to health factors that affect life expectancy to address this problem.</p>
<p>McClellan, McNeil, and Newhouse, 1994 believe that geographical location should be used. Most of the hospitals in their data sets are not special hospitals for heart catheters, so many patients go closer to the “conventional” hospital, which does not provide such treatment, than to the heart catheters hospital. MacLelon, McNeil and Newhouse therefore use as a tool variable the distance between the home of a heart disease patient and the nearest heart catheter insulation hospital to the nearest general hospital, and zero if the nearest hospital were a heart catheter insulation hospital, otherwise positive. If the relative distance above affects the probability of treatment, then it is relevant. If relative distance is randomly distributed among heart patients, it is external.</p>
<h4>Some lessons learned on IV</h4>
<ol>
<li><strong>Don't go back two steps manually.</strong></li>
<li><strong>The first stage must contain all external variables.</strong></li>
<li><strong>Carefully use the ageing entry as a tool variable</strong> It usually does not satisfy the physical assumption.</li>
<li><strong>The tools are not as good as the variables.</strong> </li>
<li><strong>Tool variables are not a panacea.</strong> It's very difficult to find tool variables that simultaneously satisfy strong relevance and strict externalities.</li>
</ol>
<h2>Robust's Return to the Indian Ocean</h2>
<h3>Source of Information Services</h3>
<p><strong>Impacted observations</strong>That's what we're talking about.<strong>Big enough.</strong>So much to remove it.<strong>Substantive changes in parameter estimates and findings</strong>This makes conclusions unreliable.</p>
<p>It could come from two sources.</p>
<ul>
<li><strong>Leverage</strong> Explain Variable&#36;X&#36;Unusual.</li>
<li><strong>Organisation</strong> Because of variables&#36;Y&#36;Unusual.</li>
</ul>
<p>We can't ignore it.<strong>Influential Observations</strong> So we need to study his tests and Robust's regression techniques.</p>
<h3>Students with disabilities</h3>
<p>References <a href="/en/blog/2023/09/04/linear-regression-basics-notes/">Linear regression base</a> The "minimal 2 times estimated disability" section of the study is the most basic tool for studying abnormalities.</p>
<h3>Leverage</h3>
<p>We know that the magnitude of the impact of the estimates of each of the observations that build the OLS regression equation is in fact different. Leverage is intended to study the magnitude of the impact of the various observations on the regression equation estimates.</p>
<p>The Leverage of OLS returns is defined as
&#36;&#36;h_i=\frac{1}{n}+\frac{(x_i-\overline{x})^2}{\sum_j(x_j-\overline{x})^2}&#36;&#36;</p>
<p>For larger leverage points, the slope of the regression line tends more to the slope from that point to the average, i.e., the point of greater leverage is most important in determining the slope of the regression line. Small leverage points are not that many, and we can delete them without changing the regression line.</p>
<h3>Cook's distance</h3>
<p>The leverage is still not intuitive, and we need to give statistics from generic regression models, that is, Cook's distance.
&#36;&#36;D_{i}=\frac{e_{i}^{2&#125;&#125;{k\hat{\sigma}^{2&#125;&#125;\left(\frac{h_{i&#125;&#125;{(1-h_{i})^{2&#125;&#125;\right)&#36;&#36;</p>
<p>&#36;k&#36;is the number of regression variables. It gives a measure of the difference between the estimation parameters and the group after they were deleted.</p>
<p>The empirical approach is to consider those observation groups with Cook statistics larger than one; to see if they are too influential.</p>
<p>Coker's distance is only available at a single impact point, which is theoretically clear.</p>
<h3>Breakdown value</h3>
<p>The blow-through value is a great robust measure for an estimate; he shows the maximum anomaly required for the estimate to be safe.</p>
<p>For example, as a measure of concentration, his penetration is zero, and any extreme impact sample will destroy our estimated effects.</p>
<p>It is clear that the penetration point of any reasonable estimate cannot exceed 50% (because if more than half of the observations are contaminated, it is impossible to distinguish between potential distribution and contamination distribution)</p>
<h3>The OLS returns.</h3>
<p>When the error distribution is normal, the minimum binary (LS) is the most efficient regression estimate. However, LS is very sensitive to the abnormalities of high leverage points. As a result, OLS often collapses when such abnormal values exist.</p>
<p>While individual abnormalities can normally be detected through sensitivity analysis, they can be “covered” if there is a set of abnormalities. This may lead to a rewinding error distribution. So we need some of Robust's return technology.</p>
<h4>Minimum absolute deviation returns</h4>
<p>Change Optimization Method to
&#36;&#36;minimise\sum_{i=1}^N|y_i-\beta_1x_{1i}-\beta_2x_{2i}-\ldots-\beta_kx_{ki}|&#36;&#36;
It's possible.&#36;x&#36;(b) Sensitivity;
In some cases, it could collapse if there was only one abnormal value with a damaged y value.</p>
<h4>Median Returns</h4>
<p>&#36;&#36;r_{i}^{2}(\beta)=\left(y_{i}-\beta_{1}x_{1i}-\beta_{2}x_{2i}-\ldots-\beta_{k}x_{ki}\right)^{2}&#36;&#36;
LMS Estimater is looking for minimization &#36;r_{i}^{2}(\beta)&#36; The median beta (i.e., the median of its minimization of the orderly square difference). LMS Estimator has a high penetration value but is difficult to calculate.</p>
<h2>A broad linear model GLM</h2>
<h3>What's GLM?</h3>
<p>The broad linear model (Generalized Linear Model) is a direct extension of the common normal linear model, which can be applied directly to continuous and discrete data; the broad linear model requires that the corresponding variable be linearly dependent on local variables; it maintains the idea of linear variables; and it expands in two ways.</p>
<ul>
<li>A linear relationship between the expectations of the responding variable and the interpretation variable is created by a connection function, i.e., a change in expectations for the responding variable.</li>
<li>By an error function, describe the last part of the random item of the broad linear model; there are different forms of disability.</li>
</ul>
<p>After the roll-out, we can summarize the three points of the GLM.</p>
<ul>
<li>Linear dependency</li>
<li>Connect Functions</li>
<li>Error Functions</li>
</ul>
<p><strong>The model is as follows:</strong>
Our original linear regression model can be described as
&#36;&#36;\mu_Y=\beta_0+\sum_{j=1}^p\beta_jX_j&#36;&#36;
In the traditional linear regression model,&#36;\mu_Y&#36; It's the observation of the average of conditions for variables, and we're asking...&#36;Y&#36; Subject to normal distribution</p>
<p>In GLM ' s broad linear model, the ready form becomes
&#36;&#36;g(\mu_Y)=\beta_0+\sum_{j=1}^p\beta_jX_j&#36;&#36;
of which&#36;g(\mu_Y)&#36; It's a function of condition average, we call it.<strong>Connect Functions</strong>And you can relax.&#36;Y&#36;For the assumption of normal distribution read&#36;Y&#36;Obey.<strong>A distribution in the index distribution group</strong>It's okay. When you set the connection function and the probability distribution, the parameters can be derived from the maximum seemingly multiple inverted generation. </p>
<h3>Selection of connecting functions and distributions</h3>
<p>We can see that at the moment we have two important super-parameters to choose for the GLM problem.</p>
<ul>
<li>Probability distribution type (because of probability distribution of variables)</li>
<li>Connection function type (what changes are made)</li>
</ul>
<p>In fact, different probability distributions have their own default connecting functions, as shown below.</p>
<table>
<thead>
<tr>
<th>Distribution Type</th>
<th>Connect Functions</th>
<th>Name of return</th>
</tr>
</thead>
<tbody><tr>
<td><code>binomial</code></td>
<td><code>(link = &quot;logit&quot;)</code></td>
<td>Logit returns</td>
</tr>
<tr>
<td><code>gaussian</code></td>
<td><code>(link = &quot;identity&quot;)</code></td>
<td>Normal linear regression</td>
</tr>
<tr>
<td><code>gamma</code></td>
<td><code>(link = &quot;inverse&quot;)</code></td>
<td>GLM Return</td>
</tr>
<tr>
<td><code>inverse.gaussian</code></td>
<td><code>(link = &quot;1/mu^2&quot;)</code></td>
<td>GLM Return</td>
</tr>
<tr>
<td><code>poisson</code></td>
<td><code>(link = &quot;log&quot;)</code></td>
<td>Porsche returns.</td>
</tr>
<tr>
<td><code>quasi</code></td>
<td><code>(link = &quot;identity&quot;, variance = &quot;constant&quot;)</code></td>
<td>A quasi-distribution of GLM</td>
</tr>
<tr>
<td><code>quasibinomial</code></td>
<td><code>(link = &quot;logit&quot;)</code></td>
<td>Normal distribution Logit</td>
</tr>
<tr>
<td><code>quasipoisson</code></td>
<td><code>(link = &quot;log&quot;)</code></td>
<td>It's almost distributed.</td>
</tr>
</tbody></table>
<p>^d089b1</p>
<p>We need to make the right model selection based on the distribution of the variables.</p>
<h3>Classic GLM.</h3>
<p>We're here.<a href="/en/blog/2024/03/13/logistic-regression-notes/">Logistic returns</a>It describes in detail the most classic form of GLM return, so that we can understand what we're doing, and so that we can compare to other GLM returns.</p>
