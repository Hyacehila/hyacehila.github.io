---
title: 'Linear Regression Basics: Linear Models, Least Squares, and Diagnostics'
title_zh: 线性回归基础：线性模型、最小二乘估计与回归诊断
date: 2023-09-04 23:00:57 +0800
permalink: /blog/2023/09/04/linear-regression-basics-notes/
categories:
- Data Science
- Statistical Modeling & Inference
tags:
- Statistics
- Regression
- Linear Models
excerpt: Covers linear models, least squares estimation, regression diagnostics, model evaluation, ANOVA, and generalized
  least squares.
description: Covers linear models, least squares estimation, regression diagnostics, model evaluation, ANOVA, and generalized
  least squares.
lang: en
translation_key: 2023-09-04-linear-regression-basics-notes
translation_status: machine
translation_source_hash: 361927c294117149e90d53bad01463a83ba348ae3a46e49b83a90bff5c1f2a42
hidden: true
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<h2>Introduction</h2>
<p>Linear statistical models are one of the most widely used models of modern statistics and are the basis for a very large number of statistical studies; because there are many linear dependencies in the real world and many dependencies that can be converted; linear models are very easy to build and process, and we present only a few simple examples in the introductory section to facilitate understanding of the various approaches behind us.</p>
<p>At the beginning of the introduction, we need to know the process and purpose of creating linear statistical models.</p>
<p>A linear statistical model would allow us to understand the relationship between the variables and the variables; to understand the potential linkages between the variables, and finally to use the models to achieve the prediction functions that we need most, decision-making, understanding linkages, etc.;</p>
<p>Of course, it's not that easy to build a model, and we need constant testing to correct the methods used before and until we're satisfied.</p>
<p><strong>Return analysis is a classic statistical issue, and we should not lack statistical thought.<a href="/en/blog/2026/01/14/anscombes-quartet/">Anscom Quartet</a></strong></p>
<h3>Linear regression model</h3>
<h4>A brief introduction.</h4>
<p>In the real world, there are a lot of situations where two variables, like X and Y, have some dependencies, which X can partially determine the value of Y, such as height and weight, temperature and city consumption; we can't determine precisely (and the exact function is not the same).<strong>Relationship</strong>
For the study of relationships, regression models are an important part.</p>
<p>In the example above, Y's response variable X's predictive variable, we know that part X determines Y, so we can give a model that includes both X-Y's decision and those factors that are not taken into account and random.
&#36;&#36;Y=f(X)+e&#36;&#36;
Here.&#36;f(X)&#36;It's what makes decisions. &#36;e&#36;It's a random error.&#36;E(e)=0&#36;</p>
<p>When?&#36;f(X)&#36;Take Special Linear Functions&#36;f(X)=\beta_{0}+\beta_{1}X&#36; At the time, we got a linear regression model that we're looking at here.&#36;f(X)=\beta_{0}+\beta_{1}X+e&#36; It's a linear regression model.</p>
<p>After applying appropriate statistical methods to determine these regression factors, we call the result linear regression equations.</p>
<p>In practical applications, there's often more than one variable, so it's easy to give multiple forms of linear regression.
&#36;&#36;Y=\beta_0+\beta_1X_1+\cdots+\beta_{p-1}X_{p-1}+e&#36;&#36;</p>
<h4>Matrix</h4>
<p>We...&#36;n&#36;The linear regression model of the dollar is performed.&#36;n&#36;Sub-observation.
&#36;&#36;x_{i1},\cdotp\cdotp\cdotp,x_{i,p-1},y_i,\quad i=1,\cdotp\cdotp\cdotp,n&#36;&#36;
They're satisfied.
&#36;&#36;y_{i}=\beta_{0}+x_{i1}\beta_{1}+\cdots+x_{i,p-1}\beta_{p-1}+e_{i} &#36;&#36;
Include Matrix
{\cHFFFFFF}{\cH00FFFF} {\cHFFFFFF}{\cH00FFFF} {\cH00FFFF} {\cHFFFFFF}{\cH00FFFF}&amp;x_{11}\cdots x_{1,p-1}\1&amp;x_{21}\cdots x_{2,p-1}\\vdots&amp;\vdots&amp;\vdots\1&amp;x_{n1}\cdots x_{n,p-1}\end{array}\right|,\mathbf{\beta}=\left|\begin{array}{c}\beta_0\\beta\mathbf{\lambda}\\vdots\\beta_{p-1}\end{array}\right|,\mathbf{e}=\left|\begin{array}{c}e_1\e_2\\vdots\\vdots\e_n\end{array}\right|,&#36;&#36;
得
&#36;&#36;y=X\beta+e&#36;
In the form of a matrix,&#36;y&#36;It's called the observation vector. &#36;X&#36;It's called Model Matrix. &#36;\beta&#36;Unknown parameter vector &#36;e&#36;A random error vector. </p>
<p>Random error generally requires the following conditions (Gauss-Markov hypothesis)</p>
<ul>
<li>&#36;E(e_{i})=0&#36; Average error value is 0</li>
<li>&#36;Var(e_{i})=\sigma^{2}&#36; Error Equation</li>
<li>&#36;Cov(e_{i},e_{j})=0&#36;Irrelevance of errors</li>
</ul>
<h4>A linear regression model</h4>
<p>For Functions
&#36;&#36;Q_{t}=aL_{t}^{b}K_{t}^{c}&#36;&#36;
Logarithmic
&#36;&#36;
\ln\left(Q_{t}\right)=\ln a+b\ln\left(L_{t}\right)+c\ln\left(K_{t}\right)
&#36;&#36;
If you change it, you get a linear model.
&#36;&#36;y_{t}=\ln(Q_{t}),x_{t1}=\ln(L_{t}),x_{t2}=\ln(K_{t})&#36;&#36;
The heart of linearity is the exchange of dollars.</p>
<h3>Difference Analysis Model</h3>
<p>The dependence of regression analysis models on variables is mostly continuous.
The difference in this section is that we're looking at indicative variables as self-variant, and he tends to indicate whether an effect exists and takes only one or two values; this model is very relevant to the size of the effect of two or more factors; it's generally called a differential analysis model.</p>
<h4>Single variable variance analysis model</h4>
<p>We use one example to describe the model that we're trying to describe.</p>
<p>We wanted to study the merits of the three wheat varieties, so we arranged six identical plots of land, each of which planted one wheat, and managed the same.</p>
<p>We'll use it.&#36;y_{ij}&#36;Description&#36;i&#36;Wheat&#36;j&#36;Production of plots of land can easily be broken down as follows:
&#36;&#36;y_{ij}=\mu+\alpha_{i}+e_{ij},&#36;&#36;
of which due to variable mean&#36;\mu&#36; Impact factors&#36;\alpha_{i}&#36;  Random error is included in the model.
Matrixed
&#36;US&#36;00begin{bmatrix}y=egin{vmatrix}1&amp;1&amp;0&amp;0\1&amp;1&amp;0&amp;0\1&amp;0&amp;1&amp;0\1&amp;0&amp;1&amp;0\1&amp;0&amp;1&amp;0\1&amp;0&amp;0&amp;1\1&amp;0&amp;0&amp;1\end{vmatrix}\begin{bmatrix}\mu\\alpha_{1}\\alpha_{2}\\alpha_{3}\\end{bmatrix}+\begin{vmatrix}e_{11}\e_{12}\e_{21}\e_{21}\e_{22}\e_{31}\e_{31}\end{vmatrix}&#36;&#36;</p>
<p> Very naturally, it can be expressed in the following forms:
 &#36;&#36;y=X\beta+e&#36;&#36;
 We can see the matrix form, the differential analysis model and the linear regression model in similar forms.</p>
<h4>Multivariant variance analysis model</h4>
<p> Our natural single-factor differential model has been extended to multiple-factor differential models, adding two different kinds of fertilizer to the preceding examples, and natural expansion.
 &#36;&#36;y_{ij}=\mu+\alpha_{i}+\beta_{i}+e_{ij}&#36;&#36;
 of which&#36;\alpha_{i}&#36; There are three kinds of extracts. &#36;\beta_{i}&#36;There are two kinds of extracts.&#36;X&#36; Yes.
&#36; \bardsymbol=\left\begin{matrix}1&amp;1&amp;0&amp;0&amp;1&amp;0\1&amp;1&amp;0&amp;0&amp;1&amp;0\1&amp;1&amp;0&amp;0&amp;0&amp;1\1&amp;1&amp;0&amp;0&amp;0&amp;1\1&amp;0&amp;1&amp;0&amp;1&amp;0\1&amp;0&amp;1&amp;0&amp;1&amp;0\1&amp;0&amp;1&amp;0&amp;0&amp;1\1&amp;0&amp;1&amp;0&amp;0&amp;1\1&amp;0&amp;0&amp;1&amp;1&amp;0\1&amp;0&amp;0&amp;1&amp;1&amp;0\1&amp;0&amp;0&amp;1&amp;0&amp;1\1&amp;0&amp;0&amp;1&amp;0&amp;I don't know.
Clearly, the matrix mentioned above indicates that no change is required; the multifactorial differential model and the plural linear regression model are similar in form, which means that we can follow a similar approach to these two different models.</p>
<h4>Coordinated differential analysis model</h4>
<p>Analysis of Covariance, ANCOVA is a combination.<strong>Difference Analysis (ANOVA)</strong> and<strong>regression</strong> The statistical model is used to compare differences in the average values of the different groups (classifications from variables) due to the variables after controlling the impact of one or more of the continuum variables (cavarians).</p>
<p>A typical coordinated differential analysis model consists of an average value item, a process effect item, and a returnee variable item and a residual. <strong>In essence, ANCOVA is an extension of ANOVA</strong>A linear adjustment to the co-variant has been added to the ANOVA. He has the hypotheses of slantness, i.e. a consistent and fixed coefficient for each group.</p>
<p>&#36;&#36;y_{ij} = \mu + \alpha_i + \gamma x_{ij} + \varepsilon_{ij}, \quad i = 1,2,\ j = 1,2,3&#36;&#36;</p>
<p>ANCOVA is essentially...<strong>Linear regression model with classification variables</strong> It can therefore also be explained from the perspective of multiple regressions, except that they emphasize different effects, a comparison of focus groups and an estimate of variable effects.</p>
<p><strong>The introduction of ANCOVA here is really irrelevant. <a href="/en/blog/2024/05/24/generalized-linear-regression-notes/">Broad linear regression</a> Section on classification of variables and virtual variables</strong> </p>
<p><strong>While we're here to present the variance analysis model that can be treated in a similar way as regression analysis, in fact it has its own way of dealing with it, and we're going to study the problem of differential analysis in detail in the pilot design, which is omitted.</strong> <a href="/en/blog/2024/02/29/experimental-design-methods-notes/">Pilot design methodology</a></p>
<h2>Overview of the regression analysis</h2>
<h3>Overview</h3>
<p>Here is a brief summary of the regression analysis.
<strong>A similar function relationship between regression analysis variables</strong>
We're looking at a connection, but it's not a strict decision.
We use variables and variables to name them in regression analysis, but there is no causality.
The general form of the regression model is
&#36;&#36;Y=f(X_1,\cdotp\cdotp\cdotp,X_p)+e&#36;&#36;
If we knew...&#36;f&#36;It's the form, but the parameters are unknown, and it's called retrogression.
Normally, we study linear regressions in the form of...
&#36;&#36;f(x_1,\cdotp\cdotp\cdotp,x_p)=b_0+b_1x_1+\cdotp\cdotp\cdotp+b_px_p&#36;&#36;</p>
<h3>Apply</h3>
<p>The application or step of the regression analysis is usually as follows:</p>
<ul>
<li>Description: A descriptive analysis of the raw data to reveal the pattern of the sample, etc.</li>
<li>Estimate: Estimated regression functions reveal hidden relationships</li>
<li>Forecast: Forecast based on regression function</li>
<li>Control: Based on the regression function, we want to keep the variable within a given range and how to select the value of the variable is more appropriate</li>
</ul>
<h3>Some explanations.</h3>
<ul>
<li>The regression equation is valuable only if the variable is within reasonable range.</li>
<li>We should minimize the re-application of the regression equation to factors other than variables.</li>
<li>If both the variable and the variable are the result of observing the world, then he reveals a natural pattern, and if the artificial interference pattern does not exist.</li>
<li>The regression equation should avoid extrapolation as much as possible, and if so, control the extrapolation and be careful whether it is reasonable.</li>
<li>Return equations cannot be reversed when used for prediction, but a new regression equation is created (control allows)</li>
<li>A single linear regression coefficient reflects the magnitude of the impact of the variable, but in multiple regressions, this analysis lacks real value due to interaction.</li>
</ul>
<h2>Estimated 1 for regression parameters (minimal 2 times estimate)</h2>
<p>Now that we know the basic shape of linear regression models, the question is how to calculate the parameters of regression models based on observations, which is what this chapter needs to deal with, and of course we know that differential analysis models are actually a particular linear regression model, so actually the contents of this chapter can be used to process differential analysis models.
This chapter actually describes the core of the entire linear statistical model.</p>
<p>For the form of linear statistical models we've given you before.
&#36;&#36;\mathbf{y}=\mathbf{X}\mathbf{\beta}+\mathbf{e}&#36;&#36;
We think it meets the Gauss-Markov hypothesis, which is...
&#36;&#36;E(e)=0,\quad\mathrm{Cov}(e)=\sigma^2I_n&#36;&#36;
This is the basic linear statistical model we're looking at right now.
Now we need to study the parameter vector.&#36;\beta&#36; Yeah.</p>
<h3>Minimal 2x10 estimate (LSE)</h3>
<p>The average minimum two times is estimated at OLS.</p>
<h4>Core content</h4>
<p>The most basic and common method to obtain the parameter vector to be estimated is the minimum two-fold method.
That's finding the right parameter vector.&#36;\beta&#36; Make deviation vectors&#36;e=y-X\beta&#36; square sum of&#36;\parallel y-x\beta\parallel^2&#36; This is the idea that the minimum two-fold method of execution is justified.
<strong>There are many other ways to minimize it.</strong>
Expand the equation that we need to minimize.
&#36;&#36;Q(\beta)=y^{\prime}y-2y^{\prime}X\beta+\beta^{\prime}X^{\prime}X\beta.&#36;&#36;
So that's the problem with multi-function miniaturization, using the theory of polarity in mathematical analysis, which requires a deviation of zero to get the regular equation. <em>This deflection requires the knowledge of a matrix micromerger.</em>
&#36;&#36;X^{\prime}X\beta=X^{\prime}y.&#36;&#36;
This equation has only one solution.&#36;rank(X)=p&#36; Which means...&#36;\beta&#36;And we always thought that we would actually meet that condition and finally get the equation to solve it.
&#36;&#36;\hat{\beta}=(X^{\prime}X)^{-1}{X}^{\prime}y.&#36;&#36;
We can prove it.&#36;\hat{\beta}&#36;It's not just a station, it's a small one.<br>Now, we can get the experience back to the equation.
That's right.<em>{0}+\hat{\beta}</em>{1}X_{1}+\cdots+\hat{\beta}<em>{p-1}X</em>It's not a good idea.
Note that the return equation is not what we think is true, and that further statistical tests are required.</p>
<h4>Apply examples</h4>
<p>For the one dollar linear regression problem &#36;y_{i}=\alpha+\beta x_{i}+e_{i},i=1,\cdots,n&#36;  Let's do it.&#36;n&#36;Sub-observation.
&#36;X=\begin{bmatrix}1&amp;x_1\1&amp;x_2\\vdots&amp;\vdots\1&amp;x_n\end{bmatrix},\beta=\begin{bmatrix}\alpha\\beta\end{bmatrix},y=\begin{bmatrix}y_1\y_2\\vdots\y_n\end{bmatrix}&#36;&#36;
计算正则方程可以得到
&#36;&#36;\left.\left(\begin{matrix}n&amp;\Sigma x_i\\Sigma x_i&amp;\Sigma x_i^2\end{matrix}\right.\right)\left(\begin{matrix}\alpha\\beta\end{matrix}\right)=\left(\begin{matrix}\Sigma y_i\\Sigma x_iy_i\end{matrix}\right),&#36;&#36;
最后能化简得到答案为
&#36;&#36;\begin{aligned}\hat{\beta}&amp;=\frac{\sum x_i\mathbf{y}_i-\sum y_i\overline{x&#125;&#125;{\sum x_i^2-n\overline{x}^2}\\hat{\alpha}&amp;== sync, corrected by elderman == @elder man
That's the lowest two-fold estimate of our entire dollar return, the same as the one in high school math.</p>
<h4>Centralization and standardization</h4>
<p>Both operations are statistically significant.</p>
<h5>Centralization</h5>
<p>Amend the original regression model to the following format:
&#36;y i=alpha+(x BAR x)<em>{1})\beta</em>{1}+\cdots+(x_{i,p-1}-\bar{x}<em>{p-1})\beta</em>{p-1}+e_{i}&#36;&#36;
事实上 完成中心化以后 我们可以改写线性回归模型为以下的形式
&#36;&#36;\mathbf{y}=a\mathbf{1}<em>{n}+X</em>{c}\mathbf{\beta}+e&#36;&#36;
此时我们的设计矩阵为
&#36;&#36;\boldsymbol{X}<em>{c}=\begin{bmatrix}x</em>{11}-\bar{x}<em>{1}&amp;x</em>{12}-\bar{x}<em>{2}&amp;\cdots&amp;x</em>{1,p-1}-\bar{x}<em>{p-1}\x</em>{21}-\bar{x}<em>{1}&amp;x</em>{22}-\bar{x}<em>{2}&amp;\cdots&amp;x</em>{2,p-1}-\bar{x}<em>{p-1}\\vdots&amp;\vdots&amp;&amp;\vdots\x</em>{n1}-\bar{x}<em>{1}&amp;x</em>{n2}-\bar{x}<em>{2}&amp;\cdots&amp;x</em>{n,p-1}-\bar{x}_{p-1}\\end{bmatrix}&#36;&#36;
<strong>What have we achieved?
We separated the regression factor from the regression constant.</strong>
We can get the results back.
&#36;&#36;\begin{cases}\hat{\boldsymbol{\alpha&#125;&#125;=\bar{\boldsymbol{y&#125;&#125;,\\hat{\boldsymbol{\beta&#125;&#125;=({\boldsymbol{X&#125;&#125;_c^{\prime}\mathbf{X}_c)^{-1}\mathbf{X}_c^{\prime}\mathbf{y}.\end{cases}&#36;&#36;
The effect of centralization is to separate the regression factor from the regression constant.
In practical studies, our relationship regression factor is much greater than our return constant, which is what we're all about.</p>
<h5>Standardization</h5>
<p>We're here to introduce our main use.<strong>z-score standardization</strong>
&#36;&#36;\begin{aligned}s_j^2&amp;=\sum_{i=1}^n(x_{ij}-\bar{x}<em>j)^2\\z</em>{ij}&amp;=\frac{x_{ij}-\bar{x}<em>I'm sorry.
After subtracting the average, we divide all the data by their standard. Bad
<strong>We can get two more useful results.</strong>
First
&#36;R=Z{\r</em>{ij}).&#36;&#36;
&#36;&#36;r_{ij}=\frac{\sum_{k=1}^{n}(x_{ki}-\overline{x}<em>{i})(x</em>{kj}-\overline{x}<em>{j})}{s</em>{i}s j}, &#36;
This means that a standardized design matrix can directly calculate the matrix from the variable.&#36;R&#36;
Second
The standardized data eliminates the differences in the range of units and values from the variables, and the estimates of the regression factor become easier for statistical analysis and visuality</p>
<h4>Estimated expectations and differences</h4>
<p>For a minimum 2x10 estimate &#36;\hat{\beta}=\left(X^{\prime}X\right)^{-1}X^{\prime}y&#36;  We do.</p>
<ul>
<li>&#36;E(\hat{\beta})=\beta;&#36;</li>
<li>&#36;\mathrm{Cov}(\hat{\beta})=\sigma^2(X^{\prime}X)^{-1}.&#36; of which&#36;\sigma^{2}&#36;It's a deviation.&#36;e&#36;The difference is a constant amount.</li>
</ul>
<p>Additional definitions:</p>
<ul>
<li>If the estimate is a linear function of observations, then he's a linear estimate.</li>
<li>Best linear neutral estimate (BLUE) is the smallest variance in the total linear neutral estimate of this parameter (validity explains the BLUE definition).</li>
</ul>
<p><strong>For linear regressions that satisfy the Gauss-Markov hypothesis, the lowest two-fold estimate is BLUE.</strong>
<strong>This provides a rational explanation for the minimum quadrillion.</strong></p>
<h4>More nature</h4>
<p>The nature of this is based on meeting the linear regression of the Gauss-Markov hypothesis.&#36;e\sim N(0,\sigma^2I)&#36; Give more nature. </p>
<p>When?&#36;e\sim N(0,\sigma^2I)&#36; Give theorem.</p>
<ul>
<li>&#36;\hat{\beta}\sim N(\beta,\sigma^2(X^{\prime}X)^{-1});&#36;</li>
<li>&#36;\frac{\mathrm{RSS&#125;&#125;{\sigma^2}\sim\chi_{n-p}^2;&#36;</li>
<li>&#36;\beta\text{ 与 RSS 相互独立.}&#36;</li>
</ul>
<p>For a centralized linear regression model</p>
<ul>
<li>&#36;E\left(\stackrel{\wedge}{\alpha}\right)=\alpha,\quad E\left(\hat{\beta}\right)=\beta,&#36; &#36;\text{这里 }\hat{\alpha}=\overline{y},\hat{\beta}=(X_{c}^{\prime}X_{c})^{-1}X_{c}^{\prime}\mathbf{y}.&#36;</li>
<li>&#36;\left.\text{ Cov}\left[\begin{array}{c}{\hat{\alpha&#125;&#125;\{\hat{\beta&#125;&#125;\\end{array}\right.\right]=\sigma^{2}\left[\begin{array}{cc}{\frac{1}{n&#125;&#125;&amp;{0}\{0}&amp;{(X_{c}^{\prime}X_{c})^{-1&#125;&#125;\\end{array}\right].&#36;</li>
<li>If further requests&#36;e\sim N(0,\sigma^2I)&#36;  Yes. &#36;\hat{\alpha}\sim N\left(\alpha,\frac{\sigma^{2&#125;&#125;{n}\right),\hat{\beta}\sim N(\beta,\sigma^2(X_c^{\prime}X_c)^{-1}),&#36;  And they're separate.
Centralization or separation of regression constants and regression factors</li>
</ul>
<h4>Minimal 2 times estimated disability</h4>
<p>Definitions: proposed combination vector&#36;\hat{y}=X\hat{\beta}&#36;  So the lowest two-fold calculation with the parameter vector is again
&#36;&#36;\hat{\mathbf{y&#125;&#125;=\boldsymbol{X}(\boldsymbol{X}^{\prime}\boldsymbol{X})^{-1}\boldsymbol{X}^{\prime}\boldsymbol{y}=\boldsymbol{H}\mathbf{y}~~~H=X(X^{\prime}X)^{-1}X^{\prime}&#36;&#36;
Easy to see, matrix.&#36;H&#36;The effect was to put a hat on the observations, so he was also called the hat matrix.</p>
<p>Can verify the nature of the hat matrix, symmetrical, etc. Sex
&#36;H^&#39;}=H,\quad H^{2}=H&#36;&#36;</p>
<p>So we can use this to show the difference vector.
&#36;&#36;\hat{e}=y-\hat{y}=(I-H)y=(I-H)e&#36;&#36;</p>
<p>Then we can verify the nature below.</p>
<ul>
<li>&#36;E(\hat{e})=0,{Cov}(\hat{e})=\sigma^2(I-H)&#36;</li>
<li>&#36;e\sim N(0,\sigma^2I)&#36; Time&#36;\hat{e}\sim N(0,\sigma^2(I-H))&#36;</li>
</ul>
<p>The nature of the disability we get here is not good enough.
&#36; \\mathrm{Var}<em>{i}^{})=\sigma^{2}(1-h</em>{ii}, \text{here} \text{i nm}.&#36;&#36;
因此我们修正这个残差得到学生化残差（T化残差）
&#36;&#36;r_{i}=\frac{\hat{e}^{}<em>{i&#125;&#125;{\hat{\sigma}\sqrt{1-h</em>{ii&#125;&#125;}&#36;&#36;</p>
<p>Now let's make a statement when the error is satisfied.&#36;e\sim N(0,\sigma^2I)&#36; Students with disabilities obey.&#36;N(0,1)&#36;
And it's supposed to match the vector.&#36;\hat{y}&#36;Discrepancies with vectors&#36;\hat{e}&#36;Independence, at a time when student disability will be a good disability analysis tool, and we will use it often in later analyses.</p>
<h3>Minimal limit 2 times estimate</h3>
<p>We didn't have parameter variables in the discussion of the two preceding subsections.&#36;\beta&#36;Here, we give the estimate of a minimum of two times the bound in the case of linear binding.
Theoretically: He's restraining the linear regression that meets the Gauss-Markov hypothesis.&#36;A\beta=b&#36; Minimal 2 times bound down estimated
&#36;&#36;\hat{\beta}_c=\hat{\beta}-(x^{\prime}x)^{-1}A\left(A\left(x^{\prime}x\right)^{-1}A^{\prime}\right)^{-1}(A\hat{\beta}-b),&#36;&#36;
of which &#36;\hat{\beta}=(X^{\prime}X)^{-1}X^{\prime}y&#36; A minimum of two times the estimate without constraints
The theorem's proof has been omitted.</p>
<h3>Minimal 2x10 (weighted OLS)</h3>
<p>We've mentioned in the disability analysis that many of the errors of linear regression models are not relevant and must have been established, so we need to give the corresponding treatment, except for the Box-Cox mutation described earlier, which can also be solved by a wide range of minimum two-folds.</p>
<p>It's also called a weighted minimum of two-fold estimation, in which the weight is related to the error equation.
What this section needs to address is the question of return as follows.&#36;\Sigma&#36;Known
&#36;&#36;y=X\beta+e,E(e)=0,\mathrm{Cov}(e)=(\sigma^2\Sigma).&#36;&#36;
In fact, this section is less practical and focuses more on the introduction of theoretical research.
Because of the matrix&#36;\Sigma&#36;It's known, and it's positive, so you can get a diagonal matrix.&#36;\boldsymbol{\Sigma}=\boldsymbol{P}^{\prime}\boldsymbol{\Lambda P}&#36; of which&#36;\Lambda&#36;It's a feature matrix.
\bardsymbol(Sigma)^&#39;\mathrm{diag}( \lambda  \frac1}2} \cdots,\lambda  \frac )\bardsymbol{P}
Left on original linear regression problem Multiplication&#36;\Sigma^{-\frac{1}{2&#125;&#125;&#36;   Got it.
&#36;&#36;z=U\beta+\varepsilon,\quad E(\varepsilon)=0,\quad\mathrm{Cov}(\varepsilon)=\sigma^2I,&#36;&#36;
So the problem turns into the lowest two-fold estimate we've dealt with before.
&#36;&#36;\beta^{\star}=(U^{\prime}U)^{-1}U^{\prime}z=(X^{\prime}\Sigma^{-1}X)^{-1}X^{\prime}\Sigma^{-1}y.&#36;&#36;
Below are some of the underlying properties of the broad minimum 2 times estimate.</p>
<ul>
<li>&#36;E(\beta^*)=\beta^;&#36;</li>
<li>&#36;\mathrm{Cov}(\boldsymbol{\beta}^{*})=\sigma^{2}(\boldsymbol{X}^{\prime}\boldsymbol{\Sigma}^{-1}\boldsymbol{X})^{-1}&#36;</li>
<li>In terms of the form of the problem mentioned at the beginning of this section, the broadest two-fold estimate is that the scope of the BLUE, or the Guass-Markov theorem, has been expanded.</li>
</ul>
<p>It can easily be seen that the assumption added at the beginning of this section is actually difficult to fulfil.&#36;\Sigma&#36;It's known that it's not a good assumption at the application level.</p>
<p>We usually use the standard minimum two-fold estimate to get some information on the error vector from the residual analysis. </p>
<p>Of course, some of the special problems do give the error vector some special structures, and this is a time for special study of special problems.</p>
<h2>Estimation of regression parameters 2 (under cholinear)</h2>
<h3>Multicollinearity</h3>
<p>The M2R estimate is widely used because he has the smallest variance in the online impartial estimation class, but with modern computer technology, people have the ability to deal with some of the super-large linear regressions, and in many cases the M2R is highly deviating from people's presumption (absolutely too high or the symbol and materially different).</p>
<p>Research suggests that the central part of these problems is that there's a near linear relationship between variables, i.e., pluralolinearity.</p>
<p>In this section, we'll look at the existence and effects of reconnective linearity and then we'll start thinking about reconnective linear solutions.</p>
<h4>Average error (MSE)</h4>
<p>We're here to present a very important criterion for evaluating estimates.
Definitions:
&#36; \begin{aligned}MSE&amp;=E\parallel\tilde{\boldsymbol{\theta&#125;&#125;-\theta\parallel2\&amp;=E(\tilde{\boldsymbol{\theta&#125;&#125;-\theta)^{\prime}(\tilde{\boldsymbol{\theta&#125;&#125;-\theta).\end{aligned}&#36;&#36;
定理：
&#36;&#36;== sync, corrected by elderman == @elder man
Inference: &#36;\tilde{\theta}<em>{1},\tilde{\theta}</em>{2},\cdots,\tilde{\theta}<em>{p}(p)}&#36;)
&#36;&#36;\pepratorname{trcov}=sum</em>{i=1}^p\operatorname{Var}(\tilde{\theta}<em>i).&#36;&#36;
&#36;&#36;\parallel E\tilde{\theta}-\theta\parallel^2=\sum</em>I don't know.
Equivalent error is made up of two parts.</p>
<p>This assessment is reasonable, and it takes into account both estimates and deviations, which is why it is more important than impartiality and validity. </p>
<h4>Average error assessment minimum two times</h4>
<p>Consider linear regression models
&#36;&#36;y=\alpha\mathbf{1}+X\mathbf{\beta}+e,E(e)=\mathbf{0},\mathrm{Cov}(e)=\sigma^2\mathbf{I}.&#36;&#36;
We've given a minimum of two-fold estimate.
That's right.&#39;x)^{-1}X&#39;y.&#36;&#36;
我们这里直接计算MSE能知道（无偏估计后半部分为0，只用研究方差和）
 &#36;&#36;== sync, corrected by elderman == @elder man &#36;
It tells us if&#36;(X^{\prime}X)^{-1}&#36; There's a very small feature value, so considering the minimum two-multiplier in terms of average error is not a good estimate.
Parameters estimate vector at this time&#36;\hat{\beta}&#36; There's gonna be an absolute overvalue.</p>
<p> This is the Gauss-Markov theorem that we've given you before, and we're just saying that the lowest two-fold estimate is the lowest difference between all linear neutral estimates, but this is also very large, which actually means that all linear impartial estimates are not good estimates in this case.</p>
<p>At this point, there's a similar linear relationship between the column vectors of the design matrix, equal to the linear relationship of the regression from the variable.</p>
<p><strong>Full converbence makes the model unsolveable, incomplete contortivity leads to inaccuracy (extremely inaccuracies) in the solution of partial coefficients, and we want to minimize the contortism before OLS.</strong></p>
<h4>Measuring co-linear</h4>
<h5>Characteristic Root</h5>
<p>Study Matrix &#36;X^&#39;Characteristic root of X&#36;
If one or more of the characteristics is close to zero, the linear algebra theory ensures the existence of linear combinations.
There's reconnectivity at this time.</p>
<h5>Conditionality</h5>
<p>We usually use squares.&#36;(X^{\prime}X)&#36; The number of conditions to measure the size of the conjunctivity defines the condition as
&#36;&#36;k=\frac{\lambda_1}{\lambda_p}&#36;&#36;
The ratio of the maximum feature value to the minimum feature value.&lt;100&#36;可以认为不存在复共线性 &#36;100&lt;k&lt;1000&#36;认为存在较强的复共线性 &#36;k&gt;&#36;1,000 says there's a non-serious reconnectivity.</p>
<p>That very small characteristic value corresponds to a characteristic vector that reflects a conjunctive linear relationship.</p>
<p>Assume that the minimum feature value corresponds to the characteristic vector as&#36;\phi&#36; So there is.
&#36;&#36;X\varphi\approx0&#36;&#36;
&#36;X&#36;It's using column vectors, or X for re-entry variables.</p>
<h5>Moderate inflation factor (VIF)</h5>
<p>Varance Information Action
Define
&#36;&#36;VIF_{j}=\frac{1}{1-{R_j}^2}&#36;&#36;
of which&#36;R_{j}&#36; As Variable&#36;j&#36; For the variable, the remaining is the minimum two-fold re-entry factor.
General considered &#36;VIF&gt;Ten dollars means there's a strong reconnectivity.</p>
<h3>Progressive return</h3>
<p>The idea of gradual return is modified at a minimum of two times.
By removing the comingolinear variable, the minimum binary avoids the excess MSE caused by the smallest diplex
We're going to go into this in chapter five, and we're going to look at options for multiple regression equations.</p>
<h3>PC returns</h3>
<p>Full name of Prince Component Review
Return of Main Component (PCR)</p>
<p>The main ingredient returns themselves are very well understood.</p>
<ol>
<li>Execute PAA to get main ingredient</li>
<li>Select the main ingredients to remove some of the smaller ones.</li>
<li>Minimum 2x2 return of the remaining main ingredient</li>
<li>Restore the main ingredient variable to the original variable</li>
</ol>
<p>Now let's talk about the nature of some of the main ingredients estimates.</p>
<ul>
<li>The main ingredient is estimated to be biased.</li>
<li>When the design matrix has a conjunctive linear relationship, proper selection of the main ingredient reduces the average error
In fact, the hard part of the main ingredient estimate is the implementation and understanding of the PA.
The return itself is a very common minimum two-fold estimate.</li>
</ul>
<p><strong>The return of the main ingredient is essentially a way of reducing the reduction of the return of the conjunctivity. Corresponding factors such as the analysis of the factors, and the reduction of the model of the structural equation can be achieved by using the idea of a similar return of the main ingredient.</strong></p>
<h3>Ridge, come back.</h3>
<p>Back to the Ridge.</p>
<h4>Summary of models</h4>
<p>Now we're going to start looking at a linear regression, and we're going to add a regular item to deal with some of the problems encountered in the smallest two-fold regression.
From the results of the lowest 2 times the estimate.
&#36;&#36;\hat{\beta}=(X^\top X)^{-1}X^\top y&#36;&#36;
When the data collected are reconnective, the design matrix is likely to be dissatisfied.
&#36;&#36;\hat{\beta}(k)=(X^{\prime}X+kI)^{-1}X^{\prime}y,&#36;&#36;
This adds to the matrix.&#36;kI&#36;It's called the Ridge.&#36;k&#36;We need to study.
At the same time, it should minimize the function as
&#36;&#36;\begin{aligned}\text{minimize}|y-X\beta|_2^2+\lambda|\beta|_2^2\end{aligned}&#36;&#36;
That's right. It's equal to adding one to the original optimisation.&#36;L2&#36;The rule of punishment.
We gave some theory, after all.&#36;L2&#36;There's still something to be done about regulars.
&#36;&#36;\mathcal{L}(\hat{w})=||X\hat{w}-Y||_2^2+\lambda||\hat{w}||_2^2=(X\hat{w}-Y)^T(X\hat{w}-Y)+\lambda\hat{w}^T\hat{w}&#36;&#36;
Yeah.&#36;\omega&#36;Yes, sir.
&#36;&#36;\frac{\partial\mathcal{L}(\hat{w})}{\partial\hat{w&#125;&#125;=2X^TX\hat{w}-2X^TY+2\lambda\hat{w}=0&#36;&#36;
Solve the equation.
&#36;&#36;(X^TX+\lambda I)\hat{w}=X^TY&#36;&#36;
Which means...
&#36;&#36;\hat{w}=(X^TX+\lambda I)^{-1}X^TY&#36;&#36;</p>
<p>Now we're going to study the nature of some of the ridge estimates.
It's a miscalculation.
&#36; \begin{aligned}E\hat{boldsymbol{beta}(k)&amp;=(X^{\prime}\boldsymbol{X}+k\boldsymbol{I})^{-1}\boldsymbol{X}^{\prime}\boldsymbol{E}\boldsymbol{y}\&amp;=(X^{\prime}\boldsymbol{X}+k\boldsymbol{I})^{-1}\boldsymbol{X}^{\prime}\boldsymbol{X}\boldsymbol{\beta}\&amp;\neq\bardsymbol{\beta}, \end{aligned}
Ridge probably has a smaller average error in some cases.
Existing &#36;k&gt;&#36;0.00 made
&#36;&#36;MSE (hat(beta}(k))&lt;MSE (hat(beta}). &#36;
It's not a thin model, but it's not going to reduce the coefficients to zero, it's going to be the regression factor for the balanced colinear variable, and finally it's going to reduce the MSE, which means that Ridge tends to spread the weight to the characteristics.
Ridge is better suited to a plan with a large correlation between features and a limited number of features without compression.</p>
<h4>Superparameter Determination</h4>
<p>For Ridge's estimate, we need to clarify a concept that's with&#36;k&#36;RSS increases.</p>
<h5>The Ridge Method</h5>
<p>The core of Ridge's existence is to solve the problem of colinearity that leads to a partial coefficient that is too small.&#36;k&#36;Changed drawings, that's the way it's done.
<img src="/assets/images/probability-statistics-notes/linear-regression-basics-notes-01.png" alt="Linear statistical model 6">
I can see it.&#36;k&#36;And the increase, the coefficient is nearing the level of stability, and we're going to have to broadly select the points that have just reached the level of stability as our estimate, because at this point we can try to balance the level of stability with the RSS.</p>
<h5>Control the balance squared</h5>
<p>The RSS will follow.&#36;k&#36;We choose a constant.&#36;c&#36; Control Ridge regression factor satisfied &#36;RSS(k)&lt;cRSS(LS)&#36;</p>
<h5>Control spread factor</h5>
<p>It's the same as the gradual return.
We're testing whether or not we're continuing to recoherent.</p>
<h5>Hoerl-Kennard Formula</h5>
<p>&#36;&#36;\dot{\hat{k&#125;&#125;=\frac{(p-1)\hat{\sigma}^{2&#125;&#125;{\sum_{i=1}^{p}(\hat{\beta}_{i}^{*})^{2&#125;&#125;&#36;&#36;
of which&#36;p&#36; is the number of variables participating in the return
The rest of the variables are estimates of the return of the LS.</p>
<h3>LASSO returns.</h3>
<h4>Summary of models</h4>
<p>Least Absolute Security and Shrinkage Advisor<br>Minimal absolute value enrichment and algorithm regression
Adding an estimated optimization function in Ridge&#36;L2&#36;Under the positive influence, Lasso estimates that it's coming.&#36;L1&#36;As punishment.
&#36;&#36;\operatorname*{minimize}_{}|y-X\beta|_2^2+\lambda|\beta|_1&#36;&#36;
For LASSO's return due to the addition&#36;L1&#36;It's complicated to continue to work on the analysis.</p>
<p>The LASSO model is characterised by a thin model, which quickly compresses some parameters to zero, and leaves him out of our regression model.
In fact, LASSO has played a very powerful role in character selection, and if there's a strong correlation between characteristics, LASSO would prefer to choose one of them and reduce the others to zero.
So LASSO is better suited to the small correlation between the features, but the problem of the large number of features.</p>
<h4>Superparameter Determination</h4>
<p>LASSO's super-parameters are a little more complicated than Ridge's, because we've got a problem with the compression of the number of features, and we've got to combine the two figures below.
<img src="/assets/images/probability-statistics-notes/linear-regression-basics-notes-02.png" alt="Linear statistical model 7">
This diagram shows how the variables are selected.&#36;log(\lambda)&#36; It shows the amount left of the signature at this time.
<img src="/assets/images/probability-statistics-notes/linear-regression-basics-notes-03.png" alt="Linear statistical model 8">
This is a map of MSE's involvement.&#36;log(\lambda)&#36;  A combination of MSE and variable compression is what we need to do.
We actually have to cross-check the machine learning field to determine the end.&#36;\lambda&#36;
Using a few discounts to cross-check a normal computer will help us decide.
Finally we get a minimum MSE.&#36;\lambda&#36; I can get it.&#36;\lambda&#36; Your own standard error.
If MSE doesn't guarantee a better compression, then a standard wrong scaling will work.</p>
<h3>ElasticNet returns</h3>
<p>Flex-net regression</p>
<h4>Summary of models</h4>
<p>So ElasticNet was quickly studied by mathematicians and made the following corrections to the optimisation function, and at this point we need to determine two super-parameters.
&#36;&#36;\min Q(\beta)=|y-X\beta|^2+\lambda\alpha\sum_{j=0}^n|\beta_j|+\lambda(1-\alpha)\sum_{j=0}^n\beta_j^2\color{red}&#36;&#36;
ElasticNet was created to combine the strengths of LASSO and Ridge.
By combining the L1, L2 model, ElasticNet preserves both LASSO ' s easy-to-resort characteristics and L2 ' s retrogressive nature, while also addressing the problem of LASSO ' s method producing multiple decompositions when multiple variables are highly relevant, effectively removing irrelevant variables from the selection of variables and retaining relevant variables of high relevance. </p>
<h4>Superparameter Determination</h4>
<p>ElasticNet models need to identify two super-parameters at the same time, but our core thinking hasn't changed.
For just making sure&#36;\lambda&#36; We've already described the treatment in LASSO, which means K-turn cross-check to find MSE.
Then add the variable.&#36;\alpha&#36;What's the situation?
First select 0.01 to 0.99. &#36;\alpha&#36; Mixed superparameters
I'm sure.&#36;\alpha&#36; Under the circumstances, we can still use the K-turn cross-check to find the right place.&#36;\lambda&#36; (minimal MSE) and corresponding standard errors
We can match every one of them.&#36;\alpha&#36;The MSE down there found the smallest MSE.
That's two super-parameter choices.
In fact, in R languages, ElasticNet LASSO uses glment packages for processing.
They're doing exactly the same thing.
In fact, this is in the assigned parameter domain.&#36;[0,1]&#36;Just re-state the process here.</p>
