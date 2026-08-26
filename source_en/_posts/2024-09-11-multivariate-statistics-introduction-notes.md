---
title: 'Multivariate Statistics Introduction: Random Vectors, Covariance Matrices, and Multivariate Normal Distributions'
title_zh: 多元统计引论：随机向量、协方差矩阵与多元正态分布
date: 2024-09-11 18:36:49 +0800
categories:
- Data Science
- Statistical Modeling & Inference
tags:
- Statistics
- Multivariate Statistics
author: Hyacehila
mathjax: true
hidden: true
excerpt: Introduces multivariate statistics through random vectors, covariance matrices, multivariate normal distributions,
  generalized variance, distances, and core concepts.
description: Introduces multivariate statistics through random vectors, covariance matrices, multivariate normal distributions,
  generalized variance, distances, and core concepts.
excerpt_zh: 整理随机向量、协方差矩阵、多元正态分布、广义方差、距离和相关基础概念。
permalink: /blog/2024/09/11/multivariate-statistics-introduction-notes/
lang: en
translation_key: 2024-09-11-multivariate-statistics-introduction-notes
translation_status: machine
translation_source_hash: 1021eb4c8c49e3a52309ad48367b4dd9586da7243d8fa825b02e4cfb31da5dce
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<h2>Additional knowledge of random vectors</h2>
<p>In the study of linear statistical models, random vectors appear in various places, where there is some addition to random vectors that are not described in detail in the probability theory.</p>
<h3>Random vector numerical feature</h3>
<p>For random vectors, we define his average as
&#36;&#36;E(X)=(EX_1,\cdots,EX_n)^{\prime}&#36;&#36;
If &#36;Y=AX+b&#36; Well...
&#36;&#36;E(Y)=AE(X)+E(b).&#36;&#36;
And there is.
&#36;&#36;{E}(АXB)=\mathrm{A}{E}(X)B&#36;&#36;</p>
<p>For random vectors, we define his compromise matrix as
&#36;&#36;\mathrm{Cov}(X)=E[(X-EX)(X-EX)^{\prime}].&#36;&#36;
It's easy to see that the matrix is symmetrical, and each dollar is the one that's often seen in probabilities, and that's the difference on the diagonal.&#36;Var(X_{i})&#36; The balance can be used to determine the relevance.
<strong>(Only linear correlation can be judged, the difference and the related coefficient do not mean independence from each other)</strong>
There's something about the matrix.
&#36;&#36;\mathrm{trCov}(X)=\sum_{i=1}^{n}\mathrm{Var}(X_{i})&#36;&#36;
The positive properties of the study matrix secondary are:
&#36;&#36;协方差矩阵是对称的半正定矩阵&#36;&#36;
If&#36;Y=AX&#36;
&#36;&#36;\operatorname{Cov}(\boldsymbol{Y})=A\operatorname{Cov}(\boldsymbol{X})\boldsymbol{A}^{\prime}.&#36;&#36;
It's extended to the collusive matrix between two random vectors.
&#36;&#36;\mathrm{Cov}(X,Y)=E[(X-EX)(Y-EY)^{\prime}].&#36;&#36;</p>
<p>It's common to add to the computational nature of
&#36;&#36;\mathrm{Cov}(AX,BY)=A\mathrm{Cov}(X,Y)B^{\prime}.&#36;&#36;</p>
<h3>Random Vector Secondary</h3>
<p>Defines the default of the random vector as
&#36;&#36;X^{\prime}AX=\sum^{n}\sum^{n}a_{ij}X_{i}X_{j}&#36;&#36;
Our random vector replaced the original.&#36;X&#36;, instead of replacing the original symmetric matrix with a collusive matrix&#36;A&#36;
Please note that the recoil of the random vector is essentially a random variable.</p>
<p>For this random variable, we can give the desired formula.
&#36;&#36;\text{设 }\operatorname{E}(X)=\boldsymbol{\mu},\operatorname{Cov}(X)=\boldsymbol{\Sigma},\text{则}E(X^{\prime}AX)=\boldsymbol{\mu}^{\prime}A\boldsymbol{\mu}+\mathrm{tr}(A\boldsymbol{\Sigma}).&#36;&#36;
&#36;tr&#36; The operator is the sum of the diagonal elements used to trace the matrix.
Based on this formula, we can provide some inferences.
&#36;&#36;\text{f}\mu=0,\text{E}&#39;AX)=\mathrm{tr}(A\Sigma);&#36;&#36;
&#36;&#36;\\textrm=(\bardsymbol{\sigma^I},\text{E^({\prime}A\mu+sigma}(}n}a}ii})=mu^}A\prime^A;&#36;&#36;
&#36;&#36;\text \mathbf(=mathbf} \mathbf},\mathbf(=mathbf{I},\text{E(\bardsymbol{X}&#39;AX})=\mathrm{tr}\boldsymbol{A}.&#36;&#36;</p>
<h3>Normal Random Vector</h3>
<h4>Basic knowledge</h4>
<p>We've studied the normal random variable in the front, which means the random variable that follows the normal distribution.&#36;N(\mu,\sigma^{2})&#36;
&#36;&#36;f(x)=\frac{1}{\sqrt{2\pi}\sigma}\mathrm{e}^{-\frac{1}{2\sigma^{2&#125;&#125;(x-\mu)^{2&#125;&#125;,-\infty&lt;x&lt;+\infty &#36;&#36;
他的二维推广应该是
&#36;&#36;f(x_1,x_2)=\frac1{2\pi\sigma_1\sigma_2\sqrt{1-\rho^2&#125;&#125;e^{-\frac1{2(1-\rho^2)}\left(\frac{(x_1-\mu_1)^2}{\sigma_1^2}-2\rho\cdot\frac{x_1-\mu_1}{\sigma_1}\cdot\frac{x_2-\mu_2}{\sigma_2}+\frac{(x_2-\mu_2)^2}{\sigma_2^2}\right)}&#36;&#36;
现在，我们自然的推广这个一维的形式到随机向量上
定义：&#36;\text{设 }n\text{ 维随机向量 }{X}=(X_1,\cdotp\cdotp\cdotp,X_n)`\text{具有密度函数}&#36;
&#36;&#36;f(x) =\frac^(\det\bardsymbol{\(\mathrm}e}\frac&#125;&#125;(x-\bardsymbol})/\bardsymbol{\Sigma^(x-\bardsymbol{\mbol{\&#125;&#125;, &#36;&#36;&#36;
We usually write it down.&#36;N(\boldsymbol{\mu},\boldsymbol{\Sigma})&#36; These are their distribution parameters.
Theorem: if the normal vector of the top distribution is satisfied &#36;E(X)=\mu,\quad\mathrm{Cov}(X)=\boldsymbol{\Sigma}.&#36;</p>
<p>As can be seen from the top core description, the multiple normal distribution is his average vector.&#36;\mu&#36;A matrix of differences&#36;\Sigma&#36; Absolutely.
The other one.&#36;\mathbf{\mu}=\mathbf{0},\boldsymbol{\Sigma}=\mathbf{I}&#36; At the time, we called it a multiple standard normal distribution.</p>
<p>His feature function is
&#36;&#36;\Phi_{X}\left(t\right)=\exp\left[\mathrm{i}t^{\prime}\mu-\frac{1}{2}t^{\prime}\Sigma t\right]&#36;&#36;
Characteristic functions can be determined with distribution functions (density functions)</p>
<h4>Decision Theorem</h4>
<p>We know that each weight of the normal vector is a normal variable, but the combined distribution of the two normal variables is not necessarily a normal vector, so we give the following determinations.</p>
<p>Random vector &#36;X=(X_1,\quad X_2,\cdots,\quad X_n)^T&#36; Obey. &#36;n&#36; Normal distribution &#36;N(a,B)&#36; The only necessary condition is any linear combination of it.&#36;Y=\sum_{i=1}^nl_iX_i&#36;Obey the one-dimensional normal distribution.&#36;&#36;N(\sum_{i=1}^nl_ia_i,\sum_{i=1}^n\sum_{k=1}^nl_il_k\operatorname{cov}(X_i,X_k))&#36;&#36;
Based on this theorem, it is easy to see the previously proven conclusion that the joint distribution of the two separate normal variables must be the normal vector.</p>
<h4>Decomposition Theorem</h4>
<p>We talked about the decomposition of the dichotomy when we studied it, and now we're promoting it.
A matrix with multiple normal distributions.&#36;\Sigma&#36;Meet the split diagonal matrix form
&#36;&#36;\left.\bardsymbol<em>{11}&amp;\boldsymbol{0}\\boldsymbol{0}&amp;\boldsymbol{\Sigma}</em>{22}\end{matrix}\right.\right],&#36;&#36;
那么我们可以进行对应的分解
&#36;&#36;\boldsymbol{X}=\begin{pmatrix}\boldsymbol{X}<em>1\\boldsymbol{X}<em>2\end{pmatrix},\quad\boldsymbol{\mu}=\begin{pmatrix}\boldsymbol{\mu}<em>1\\boldsymbol{\mu}<em>2\end{pmatrix},&#36;&#36;
这时候我们能验证到
&#36;&#36;f(x)=f</em>{1}(x</em>{1})f</em>{2}(x</em>{2}),&#36;&#36;
当
&#36;&#36;\begin{aligned}f_1(x_1)&amp;=\frac{1}{(2\pi)^{\frac{m}{2&#125;&#125;\det\Sigma_{11&#125;&#125;\mathrm{e}^{-\frac{1}{2}(x_1-\mu_1)/\Sigma_{11}^{-1}(x_1-\mu_1)},\f_2(x_2)&amp;=\frac{1}{(2\pi)^{\frac{n-m}{2&#125;&#125;\det\Sigma_{22&#125;&#125;\mathrm{e}^{-\frac{1}{2}(x_2-\mu_2)\Sigma_{22}^{-1}(x_2-\mu_2)}.\end{aligned}&#36;&#36;
其实我们在二元正态分布里面提到的相关系数&#36;\rho&#36;    就是决定协方差矩阵副对角元是否为0的关键
根据上面的推论我们能够给出一个重要的正态分布分解定理
&#36;&#36;\begin{aligned}\text{(a) setx)\sim\(mu,\sigma),\text{and X\text{and \mu\text},\text{and}\boldsymbol{Sigma}\text{syggle},\text{x}\bardsymbol{X} i\sim (\boldsymbol{\mu}<em>i,\boldsymbol{\Sigma}</em>{ii}, i=1,2\text{and independent}. \end{aligned}&#36;&#36;
&#36;&#36;\begin{aligned} (\mathrm{b}\text{\bardsymbol{\Sigma}2\bardsymbol{I},\text{and }\bardsymbol}X}(X 1,\cdots,X n)^prime},\bardsymbol{mu&#125;&#125;(\mu 1,\cdots,\mu^n)^,\text{x}, \text{x^x(i, \sigma^2, i)&amp;=1,\cdots, n\text{and independent} .\end{aligned}
<em>This theorem tells us that independence and irrelevance are equal to the weight of a normal vector, and we just need to verify that we have a zero difference to guarantee independence.</em></p>
<h4>Aggregated normal vector</h4>
<p>We demand that all normal variables be independent.
&#36;&#36;\sum_{r=1}^na_rX_r\sim N{\left(\sum_{r=1}^na_r\mu_r,\sum_{r=1}^na_r^2\sigma_r^2\right)}&#36;&#36;
We want all normal vectors to be independent.
&#36;&#36;\sum_{r=1}^nX_rA_r\sim N_m\left(\sum_{r=1}^n\mu_rA_r,\sum_{r=1}^n(A_r^{\prime}\sum_rA_r)\right)&#36;&#36;</p>
<h4>The change of the dimension to the normal vector</h4>
<p>Let's start with a theory of the core of normal vector variation.
&#36;&#36;\begin{gathered}
\text{ 设 n 维随机向量}\mathbf{X}\sim N(\boldsymbol{\mu},\boldsymbol{\Sigma}),\boldsymbol{A}\text{ 为}n\times n\text{ 非随机可逆阵},\boldsymbol{b} \
\text{为 }n\times1\text{ 向量,记 }Y=AX+b\text{ ,则} \
\boldsymbol{Y}\sim N(\boldsymbol{A}\boldsymbol{\mu}+\boldsymbol{b},\boldsymbol{A}\boldsymbol{\Sigma}\boldsymbol{A}^{\prime}).
\end{gathered}&#36;&#36;
It's obvious that we can use some of the more specific variations to achieve some of the special effects with random reversible matrices.
<strong>Amending Arguments</strong>
&#36;&#36;\text{设 }X\sim N(\boldsymbol{\mu},\boldsymbol{\Sigma}),\text{则 }Y=\boldsymbol{\Sigma}^{-\frac{1}{2&#125;&#125;\boldsymbol{X}\sim N(\boldsymbol{\Sigma}^{-\frac{1}{2&#125;&#125;\boldsymbol{\mu},\boldsymbol{I}).&#36;&#36;
Changed to render irrelevant all normal weights that could have been relevant, and the difference was one.
<strong>Switching</strong>
&#36;&#36;\text{设 }\boldsymbol{X}\sim N_n(\boldsymbol{\mu},\sigma^2\boldsymbol{I}),\boldsymbol{Q}\text{ 为 }n\times n\text{ 正交阵,则 }\boldsymbol{Q}\boldsymbol{X}\sim N_n(\boldsymbol{Q}\boldsymbol{\mu},\sigma^2I)&#36;&#36;
It's a transformation that guarantees the original independence and equation.
<strong>Regenerative</strong>
&#36;&#36;\begin{aligned}\text{}\bardsymbol{x}-\bardsymbol{n(\bardsymbol},\bardsymbol{\Sigma},\bardsymbol{\chi}\bardsymbol{X},\barysymbol{x}, \bardsymbol}Sigma}\text{segmbol}&amp;=\begin{bmatrix}\boldsymbol{X}<em>1\\boldsymbol{X}<em>2\end{bmatrix},\boldsymbol{\mu}=\begin{bmatrix}\boldsymbol{\mu}<em>1\\boldsymbol{\mu}<em>2\end{bmatrix},\boldsymbol{\Sigma}:=\begin{bmatrix}\boldsymbol{\Sigma}</em>{11}&amp;\boldsymbol{\Sigma}</em>{12}\\boldsymbol{\Sigma}</em>{21}&amp;\boldsymbol{\Sigma}</em>\end{bmatrix},\text{it}\boldsymbol{x}<em>The text is called m\timestext, while \bardsymbol{\Sigma}</em>\text{m\times\boldsymbol{m}\text{format, \boldsymbol{x} 1-\boldsymbol{m}<em>1,\boldsymbol{\Sigma}</em>That's right.
This theorem tells us that any dimension of a normal vector is also a normal vector.</p>
<h4>Change in the normal vector of changes in dimensions</h4>
<p>The core theorem is just a small change ahead.
&#36;US&#36;\begin{array}\text{set}X\sim n\left(\bardsymbol},\bardsymbol{\right),\bardsymbol{A}text{m\tates n\text{format)}m\lex(m\left}&lt;{\bord0\shad0\alphaH3D}You know, \boldsymbol \boldsymbol \sim\boldsymbol}mbol}mft (\boldsymbol{, \boldsymbol{, \boldsymbol{A} \boldsymbol{\Sigma}\boldsymbol{A^ {\prime}\right.\end{array} &#36;
This theorem applies to the increase in dimensions, which is &#36;m.&gt;Case of &#36;0.00
It makes him look at normal processes at random.
<strong>Lower the dimension to 1</strong>
&#36;&#36;\begin{array}{cc}&amp;\text{x\x\sim{n(\bardsymbol{mu},\bardsymbol{Sigma},\bardsymbol{c}\text{n\text{non-zero vectors} &amp; ^(\prime}\bardsymbol{x}\sim(\pendsymbol^c},\bardsymbol{c&#125;&#125;.\end{array} &#36;
Linear combinations of normal vectors are normal variables
<strong>And turn the extraction into one.</strong>
&#36;&#36;\begin{aligned}\quad&amp;\text{x\sim n\left(\bardsymbol{mu},\bardsymbol(sigma}\right),\bardsymbol{mu}=left(,\cdots,\bardsymbol}<em>n\right)^{\prime},\boldsymbol{\Sigma}=\left(\boldsymbol{\sigma}</em>\right, \text{&amp;N\left(\boldsymbol{\mu}<em>i\right.,\\sigma</em>{ii}\left.\right),&amp;i =1,\cdots, n.end{aligned}
This theorem tells us that each weight of a normal vector is a normal vector.
But please note that the inverse theorem is not valid.</p>
<h4>Distribution of conditions and expectations</h4>
<p>&#36; \\begin{aligned}\text{Theoretical 2.3.2]&amp;\text{set}X=begin{bmatrix}X^x}<em>{p-r}\sim N</em>{\rho}(\mu,\Sigma)(\Sigma&gt;0), \\text{, then \\text{to}\text{time}, X^(1)}\text{condition distribution} (X^(1)}XX^(2)}&amp;\sim N_{r}(\mu_{1,2},\Sigma_{11,2}),\end{aligned}&#36;&#36;
其中
&#36;&#36;\{\mu_{1\cdot2}=\mu^{(1)}+\Sigma_{12}\Sigma_{22}^{-1}(x^{(2)}-\mu^{(2)})}\~~~{\Sigma_{11\cdot2}=\Sigma_{11}-\Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}.}&#36;&#36;</p>
<p>Corresponds.
&#36;&#36;(X^{(1)}|X^{(2)})\sim N_{r}(\mu_{1,2},\Sigma_{11,2})&#36;&#36;
Name
&#36;&#36;\mu_{1\cdot2}=\mu^{\left(1\right)}+\sum_{12}\sum_{n2}^{-1}\left(x^{\left(2\right)}-\mu^{\left(2\right)}\right)&#36;&#36;
It's an expectation.&#36;E\left(X^{\left(1\right)}|X^{\left(2\right)}\right)&#36;</p>
<h3>Carside distribution</h3>
<p>&#36;\begin{aligned}&amp;\\text{order}Z 1, Z 2, \\cdots, Zn\text{and}Z\text{is the standard normal distribution random variable for independent and distributed distribution,}&amp;\text \ and =1^2Z2^cdots+\n^text{, which we call X is subject to the caloric distribution of freedom }end{aligned}
It's easy to know that the density function of the calf distribution is
&#36;g(x)=\begin{cases}\dfrac{1}Gamma\Big(\dfrac{n}2^ {\mathrm{e}-\frac{2&#125;&#125;&amp;\text \x&gt;0,\\0,&amp;\text \xleqslant0\end{cases}&#36;&#36;
定理
&#36;&#36;\text{set}\bardsymbol{sim\mathrm{<em>0}bardsymbol{\symbol{sim\chi 2.&#36;&#36;
定理
&#36;&#36;\begin{aligned}X\sim n(0,I n), A\text{n\times\text{r\text{symmetric},\text{\text{}\matbf{A^mathbf{A}\mathbf{A}\text{second}X^&#39;}AX\sim\chi</em>{r}^{2}\end{aligned}&#36;&#36;
定理
&#36;&#36;\begin{aligned}&amp;\text{X}\sim n (\mathbf}, \mathbf{I} n), A\text{n\tmes\text{symmetric array}, B\text{m\times\text}. \text{Fang}&amp;Other Organiser&#36;&#36;
定理
&#36;&#36;\begin{gathered}
X\sim n (mathbf{I}), A\text{and B}n\text{symmetrical},\text{and}A\bardsymbol{B}=mathbf,\text{second}
\\text{type X&#39;AX and X&#39;BX Independent. ♪ I'm sorry ♪
I'm sorry.</p>
<h2>Multiple statistical bases</h2>
<h3>Parameter estimates for multiple normal distribution</h3>
<h4>Additional definitions</h4>
<p>Common digital features of multiple normals in general have been introduced in probabilistic theory. <a href="/en/blog/2023/03/18/elementary-probability-notes/">Primary probability theory</a> It's enough to know the next few.</p>
<ul>
<li>Mean vector</li>
<li>Arguments</li>
<li>Related coefficient arrays</li>
</ul>
<p>Additional definitions<strong>Sample deviation matrix</strong>Is:
&#36; \begin{aligned}A&amp;=\sum_{a=1}^n\left(X_{(a)}-\overline{X}\right)\left(X_{(a)}-\overline{X}\right)^{\prime}=X^{\prime}X-n\overline{X}\overline{X^{\prime&#125;&#125;\&amp;=X^{\prime}\left[I_n-\frac{1}{n}\mathbf{1}<em>n\mathbf{1}<em>n^{&#39;}\right]X\xrightarrow{\mathrm{def&#125;&#125;\left(a</em>{ij}\right)</em>{p\times p}\end{aligned}&#36;&#36;
其中
&#36;&#36;a_{ij}=\sum_{a=1}^{n}\left(x_{ai}-\overline{x}<em>{i}\right)\left(x</em>&#36;-overline{x}
That's where the unit matrix didn't separate &#36;n.<del>or</del> n-1&#36;  </p>
<h4>Largely specious estimates of averages and alignment matrix</h4>
<p>Now, let's figure out how to estimate two core parameters in a multi-state normal analysis. &#36;\mu,\Sigma&#36; </p>
<p>Yes.&#36;\Sigma&#36;Yeah.
&#36; \begin{aligned}
L\left (mu, Sigma\right)&amp; =-\frac{np}{2}\ln2\pi-\frac{n}{2}\ln\left|\Sigma\right|  \
&amp;-\frac{1}{2}\mathrm{tr}\left[\Sigma^{-1}\sum_{i=1}^{n}\left(x_{\left(i\right)}-\mu\right)\left(x_{\left(i\right)}-\mu\right)^{\prime}\right] \
&amp;=C-\frac{1}{2}\mathrm{tr}\left[\Sigma^{-1}A+n\Sigma^{-1}\left(\overline{X}-\mu\right)\left(\overline{X}-\mu\right)^{\prime}\right] \
&amp;=C-\frac{1}{2}\mathrm{tr}\left(\Sigma^{-1}A\right)-\frac{n}{2}\left[\left(\overline{X}-\mu\right)^{\prime}\Sigma^{-1}\left(\overline{X}-\mu\right)\right] \
&amp;\leqslant C-\frac{1}{2}\mathrm{tr}\left(\Sigma^{-1}A\right).
\end{aligned}&#36;&#36;
等号只在&#36;\mu=\overline{X}&#36; 的时候取到 也就是
&#36;&#36;\ln L\left(\overline{X},\Sigma\right)=\max_{\mu}\ln L\left(\mu,\Sigma\right).&#36;&#36;</p>
<p>We can prove it with similar ideas.
When?&#36;&#36;\hat{\Sigma}=\frac{1}{n}A&#36;&#36;Sometimes.
&#36; \\mathrm}L\left(overline{x),\right.) =max {\{\xXx},\Sigma=&gt;♪ L\left (overline{X}, Sigma\right): &#36;
That's how the parameters are estimated. </p>
<h4>The export of other very seemingly significant estimates and the related coefficients are very similar.</h4>
<p>We've been telling you how to do this in the course of learning the math. &#36;\mu,\Sigma&#36; It's a very similar estimate.&#36;\phi(\mu,\Sigma)&#36; The whole idea is a function of nature.</p>
<p>Well, we can naturally export the MLE.
There's a huge estimate of what's going on.
What's that?<em>{ij}=\frac{1}{n}\sum</em>{t=1}^{n}\left(x_{ti}-\overline{x}<em>{i}\right)\left(x</em>{tj}-\overline{x}<em>{j}\right)=\frac{1}{n}a</em>{ij}.&#36;&#36;
根据相关系数定义式可以给出
&#36;&#36;r_{ij}=\frac{\hat{\sigma}<em>{ij&#125;&#125;{\sqrt{\hat{\sigma}</em>{ii}\cdot\hat{\sigma}<em>{jj&#125;&#125;}=\frac{a</em>{ij&#125;&#125;{\sqrt{a_{ii}\cdot a_{jj&#125;&#125;}.&#36;&#36;</p>
<h4>Nature of the estimate</h4>
<p>It's very apparent that the estimates have the following characteristics.</p>
<ul>
<li>&#36;\overline{X}\sim N_{p}\left(\mu,\frac{1}{n}\sum\right)&#36;</li>
<li>&#36;\overline{X}和A相互独立&#36;</li>
<li>&#36;A\xrightarrow{d}\sum_{i=1}^{n-1}Z_{i}Z_{i}^{\prime},\text{其中}Z_{1},\cdots,Z_{n-1}\text{独立同 }N_{p}(0,\Sigma)\text{分布}&#36;</li>
<li>&#36;P\left{A&gt;0\right}=1\Leftrightarrow n&gt;p.&#36;</li>
<li>The value of the average is very much estimated, neutral, effective, compatible, gradual normality, sufficient statistically.</li>
<li>It's a very similar estimate of the alignment matrix, the correction is neutral, effective, compatible, gradual normality is a sufficient measure of statistics.</li>
<li>Estimation of relevant coefficients is gradual and neutral</li>
<li>The estimate of the simple average against the reconciliation matrix does not require a normal sum</li>
</ul>
<h3>Total sample distribution of multiple normals</h3>
<p>In a one-dollar normal general, the hypothetical test involves an overall problem of differential analysis of the two aggregates that have been extended to multiple aggregate averages; we've designed a lot of fine statistics to help us solve these problems; here we extend those sample statistics to multiple normal aggregates, including common ones.&#36;\chi^2,t,F&#36;Multiple normal forms of three statistics</p>
<h4>Wishart&#36;W&#36;Distribution</h4>
<p>We know that two important statistics are being given. Medium
&#36;&#36;\overline{X}\sim N_{p}\left(\mu,\frac{\sum}{n}\right).&#36;&#36;
So, what's the estimate? &#36;S=\frac{1}{n-1}A&#36; What's the distribution?</p>
<p>Definitions&#36;X(a)\sim ~N_p\left(0,\Sigma\right)\left(\alpha=1,\cdots,n\right)&#36;Independent, remember.&#36;X=&#36;
&#36;\left(X_{\left(1\right)},\cdots,X_{\left(n\right)}\right)^{\prime}为n\times p矩阵，则称随机阵&#36;
&#36;&#36;W=\sum_{a=1}^{n}X_{\left(a\right)}X_{\left(a\right)}^{\prime}=X^{\prime}X&#36;&#36;
It's distributed between the West and the West.&#36;\sim W_{p}\left(n,\Sigma\right).&#36;</p>
<p>When?&#36;p=1&#36;There are times.
&#36;&#36;W=\sum_{a=1}^{n}X_{\left(a\right)}^{2}\sim\sigma^{2}\chi^{2}\left(n\right),&#36;&#36;
In other words, this is the spread of the calorie distribution in the total of multiple normals.</p>
<h4>Hotelling&#36;T^2&#36;Distribution</h4>
<p>We can come straight to the conclusion. It's original.&#36;t&#36;Extension of distribution</p>
<p>Set &#36;X\sim N_{p}\left ( 0, \Sigma \right )&#36;,W\simW p}\left(n,Sigma\right)\left(\Sigma)&gt;0,\right.&#36; &#36;\gqslant p) and X and W are independent of each other.&#36;为霍特林&#36;T^{2}&#36;统计量，其分布称为服从&#36;n&#36;个自由度的&#36;T^2&#36;分布，记为&#36;I'm sorry.
More generally, if&#36;X\sim N_{\rho}\left(\mu,\Sigma\right)\left(\mu\neq0\right)&#36;, or&#36;T^2&#36;The distribution is non-centre-hotrin.&#36;T^2&#36;Distribution, as&#36;T^2\sim T^2(p,n,\mu).&#36;</p>
<h4>Wilks.&#36;A&#36;Distribution</h4>
<p>When we estimate the parameters, use them. &#36;A&#36; As an estimate of the coordinated matrix; we defined the broad range in multiple statistics as the array of the coordinated matrix <a href="/en/blog/2024/01/30/multivariate-statistical-analysis-notes/">Multiple statistical analysis</a> and the “wide range” section</p>
<p>Set &#36;A \left( , , , , , , , , , )&gt;0,n_{1}\geq\right.&#36;&#36;p),且A_{1}与A_{2}独立，则称广义方差之比&#36;
&#36;&#36;\Lambda = \\right}{\left|A  +right}{\right}A}text}Stat{Stat{Stat{Stat{Stat{Stat{Stat{Stat{Specific }Specific }Wilx's Distribution, as }Lambda\sim\Lambda\ft(p, n , n}right}.\ when p=1, \LambdaStat\Species is precisely the parameter in the one-dollar statistics is n  {(m}\n}2}text{Spect\text{Spect{(}beta}, n}, {2}).</p>
<h4>Special conclusions</h4>
<p>&#36;&#36;\Delta\left(p,n,1\right)\frac{d}{1+\frac{1}{n}T^{2}\left(p,n\right)}&#36;&#36;</p>
<p>&#36;&#36;T^{2}\left(p,n\right)=n\cdot\frac{1-\Lambda\left(p,n,1\right)}{\Lambda\left(p,n,1\right)}&#36;&#36;</p>
<p>&#36;&#36;\frac{n-p+1}{np}T^{2}=\frac{n-p+1}{p}\frac{1-A}{\Lambda}=F\left(p,n-p+1\right).&#36;&#36;</p>
<h2>Multiple statistical extrapolations</h2>
<h3>Insulation of individual aggregate mean vectors</h3>
<p>Want to test
&#36;&#36;H_0:\boldsymbol{\mu}=\boldsymbol{\mu}_0,\quad H_1:\boldsymbol{\mu}\neq\boldsymbol{\mu}_0&#36;&#36;</p>
<h4>The AC matrix is known.</h4>
<p>Construct statistics
&#36;T 0^2=(\overline{x}-\bardsymbol{0}^prime}\left(\frac1n\bardsymbol{\right)^(\overline{bardsymbol{)}n(\overline{\bardsymbol})=(\barysymbol{x&#125;&#125;}<em>0)^{\prime}\boldsymbol{\Sigma}^{-1}(\overline{\boldsymbol{x&#125;&#125;-\boldsymbol{\mu}<em>0)&#36;&#36;
原假设为真时则有
&#36;&#36;T</em>{0}^{2}\sim\chi^{2}\left(p\right)&#36;&#36;
使用单侧检验有
&#36;&#36;\text\</em>\alpha^2(p), \\text{rejected}H &#36;0</p>
<h4>The Accompany Matrix is unknown</h4>
<p>Construct Statistics (Hotlyn)&#36;T^2&#36;Statistics)
&#36;&#36;T^{2}=n\left(\bar{x}-\mu_{0}\right)^{\prime}S^{-1}\left(\bar{x}-\mu_{0}\right)&#36;&#36;
When the assumption is true, there is.
&#36;&#36;\frac{n-p}{p(n-1)}T^{2}\sim F(p,n-p)&#36;&#36;
One-sided check.
&#36;&#36;\text{若}\frac{n-p}{p\left(n-1\right)}T^2\geqslant F_a(p,n-p),\text{则拒绝 }H_0&#36;&#36;</p>
<h4>Large sample extrapolation</h4>
<p>The previous proofs are based on multiple normal assumptions, but sometimes multiple normal assumptions are not satisfied; but when the sample capacity is large enough, multiple centers can solve problems.
When the sample size is large enough, the following approximation relationships occur.
&#36;T 0^2=(\overline{x}-\bardsymbol{0}^prime}\left(\frac1n\bardsymbol{\right)^(\overline{bardsymbol{)}n(\overline{\bardsymbol})=(\barysymbol{x&#125;&#125;}<em>0)^{\prime}\boldsymbol{\Sigma}^{-1}(\overline{\boldsymbol{x&#125;&#125;-\boldsymbol{\mu}<em>0)&#36;&#36;
&#36;&#36;T</em>{0}^2=n(\overline{x}-\mu)^{\prime}S^{-1}(\overline{x}-\mu)&#36;&#36;
&#36;&#36;T</em>Other Organiser
All hypothetical tests and area estimations follow the formula above.</p>
<p>&#36;&#36;S^{-1}=A^{-1}(n-1)&#36;&#36;</p>
<h3>It seems to be more than statistics.</h3>
<p>A very large number of important tests of statistics in multiple statistics are derived from the maximum approximation principle, rather than by extending the largest approximation principle in mathematical statistics. And here we're presenting the principle of seemingly statistical and maximum comparison.</p>
<p>Set &#36;p&#36; The density function of the sum of the elements is&#36;f\left(x,\theta\right)&#36;, where&#36;\theta&#36;is an unknown parameter, and&#36;\theta\in\Theta&#36;
&#36;\left(参数空间\right),又设\Theta_{0}是\Theta 的子集，我们希望对下列假设：&#36;
&#36;&#36;H_{0}:\theta\in\Theta_{0},H_{1}:\theta\in\Theta_{0}&#36;&#36;
To judge, that's a hypothetical test.</p>
<p>From General&#36;X&#36;The extraction capacity is&#36;n&#36;The sample.&#36;X_{(t)}(t=1,\cdots,n).&#36;Use the joint density function of the sample
&#36;&#36;L\left(x_{\left(1\right)},\cdots,x_{
\left(n\right)};\theta\right)=\prod_{t=1}^{n}f\left(x_{\left(t\right)};\theta\right)&#36;&#36;
As&#36;L\left(X;\theta\right)&#36;and calls it the apparent function of the sample</p>
<p>Include statistics
&#36;&#36;\lambda=\max_{\theta\in\theta_{0&#125;&#125;L\left(X;\theta\right)/\max_{\theta\in\theta}L\left(X;\theta\right),&#36;&#36;
It's a sample.&#36;X_{(t)}\left(t=1,\cdots,n\right)&#36;function, commonly called&#36;\lambda&#36;It's like we're out of statistics.</p>
<p>Known by the principle of maximum approximation if&#36;\lambda&#36;It's too small.&#36;H_0&#36;This sample was observed for real.&#36;X_{(\omega)}(t=1,...,n)&#36;Probability ratio &#36;H_0&#36; To observe this sample when not true&#36;X_{(\omega)}&#36; It's much less likely.&#36;H_0&#36;Not established</p>
<p>According to traditional methods, we need to calculate the exact sample distribution that appears to be statistically comparable to the hypothetical test results, but multiple statistics are too complex and we give a large sample that resembles the theorem.
When sample size n is large
&#36;&#36;-2ln\lambda=-2ln\left[\max_{\theta\in\Theta_{0&#125;&#125;L\left(X;\theta\right)/\max_{\theta\in\Theta}L\left(X;\theta\right)\right]&#36;&#36;
It's like obeying freedom.&#36;f&#36;Yes.&#36;\chi^2&#36;Distribution, where&#36;f=\Theta&#36;Number of dimensions&#36;-\Theta_0&#36;Dimensions (i.e., the margin where freedom is restricted)</p>
<h3>Extrapolation of two overall averages</h3>
<h4>Aligning Matrix</h4>
<p>Two totals. &#36;N_{p}(\mu_1,\Sigma),N_{p}(\mu_2,\Sigma)&#36; Take two separate samples.&#36;x_{n_1},y_{n_2}&#36;
We want tests.
&#36;H 0:\bardsymbol}<em>1=\boldsymbol{\mu}<em>2,\quad H_1:\boldsymbol{\mu}<em>1\neq\boldsymbol{\mu}<em>2&#36;&#36;
我们可以自然的从一元统计的情形中得到霍特林&#36;T&#36;统计量
&#36;&#36;\begin{aligned}T^2=&amp;\left(\frac{1}{n_1}+\frac{1}{n_2}\right)^{-1}\left(\overline{x}-\overline{y}\right)^{\prime}S^{-1}\left(\overline{x}-\overline{y}\right)\=&amp;\frac{n_1n_2}{n_1+n_2}\left(\overline{x}-\overline{y}\right)^{\prime}S^{-1}\left(\overline{x}-\overline{y}\right)\end{aligned}&#36;&#36;
当原假设成立的时候
&#36;&#36;\frac{n</em>{1}+n</em>{2}-p-1}{p\left(n</em>{1}+n</em>{2}-2\right)}T^{2}\sim F(p,n_{1}+n_{2}-p-1)&#36;&#36;
可以自然的进行单侧检验量 方向和我们在前面介绍的一样
其中
&#36;&#36;S^{-1}=(\frac{A_1+A_2}{n_1+n_2-2})^{-1}&#36;&#36;
<strong>There's a significant difference between the two mean vectors, which doesn't mean there must be a significant difference between them.</strong>; that is to say, the equal rejection of the mean vector does not mean that we will be able to detect significant differences when we test each weight separately; </p>
<p>But this difference is still the main reason for the average vector difference, and it is customary for us to examine the significant differences between the weights separately after testing the significant differences in the overall vector.</p>
<h4>It's a pair.</h4>
<p>It is assumed that two samples are independent in certain circumstances; in a number of experiments, two samples may exist in pairs but are not independent; and pairing data often leads to better statistical extrapolations
You!
&#36;&#36;d_i=x_i-y_i,\quad i=1,2,\cdots,n&#36;&#36;
There is.&#36;d_{i}&#36;Obey the new distribution.
&#36;&#36;N_{p}\left(\delta,\Sigma\right)&#36;&#36;
of which
&#36;&#36;\delta=\mu_{1}-\mu_{2}&#36;&#36;
So the original assumption is...
&#36;&#36;H_0:\boldsymbol{\mu}_1=\boldsymbol{\mu}_2,\quad H_1:\boldsymbol{\mu}_1\neq\boldsymbol{\mu}_2&#36;&#36;
To
&#36;&#36;H_0:\boldsymbol{\delta}=\boldsymbol{0},\quad H_1:\boldsymbol{\delta}\neq0&#36;&#36;
The problem turned into a single whole.</p>
<h3>Comparison of multiple aggregate averages (multiple variance analysis)</h3>
<p>Assumptions
&#36;&#36;H_0:\boldsymbol{\mu}_1=\boldsymbol{\mu}_2=\cdot\cdot\cdot=\boldsymbol{\mu}_k,\quad H_1:\boldsymbol{\mu}_i\neq\boldsymbol{\mu}_j,\text{至少存在一对 }i\neq j&#36;&#36;
One of ours.&#36;\mu&#36;They're all vectors, not single-dollar variables presented in the variance analysis.</p>
<p>Remember
&#36;&#36;T=SST=\sum_{i=1}^k\sum_{j=1}^{n_i}(x_{ij}-\overline{x})(x_{ij}-\overline{x})^{\prime}&#36;&#36;
&#36;&#36;E=SSE=\sum_{i=1}^k\sum_{j=1}^{n_i}(x_{ij}-\overline{x_i})(x_{ij}-\overline{x_i})^{\prime}&#36;&#36;
&#36;&#36;H=SSTR=\sum_{i=1}^kn_i\left(x_i-\overline{x}\right)\left(x_i-\overline{x}\right)^{\prime}&#36;&#36;
There is.
&#36;&#36;T=E+H&#36;&#36;
Use the approximation test to get Wilks statistics.
&#36;&#36;\Lambda=\frac{|E|}{|E+H|}&#36;&#36;
When the original assumption is true, the statistics follow the parameters.&#36;(p,k-1,n-k)&#36;The Wilkes Distribution</p>
<p>The rule for rejection is:
&#36;&#36;\text{若}\Lambda\leqslant\Lambda_{1-a}(p,k-1,n-k),\text{则拒绝 }H_0&#36;&#36;
The absence of significant differences in multiple tests does not mean that their weights do not differ significantly; in turn, they are the same; in custom, if multiples detect significant differences, we still have to do a one-dollar variance analysis to see where the differences generally come from.</p>
<h3>The Inference of the Arranged Matrix</h3>
<p>We don't take into account the hypothetical tests of the single-sum matrix because it's complicated and not unique.</p>
<p>Assumptions
&#36;H 0:\bardsymbol{\<em>1=\boldsymbol{\Sigma}<em>2=\cdots=\boldsymbol{\Sigma}<em>k,\quad H_1:\boldsymbol{\Sigma}<em>i\neq\boldsymbol{\Sigma}<em>j,\text{at least one pair exists}i\neq j&#36;&#36;
修正的似然比统计量为
&#36;&#36;\lambda=\frac{\prod</em>{i=1}^k|S_i|^{(n_i-1)/2&#125;&#125;{|S_p|^{(n-k)/2&#125;&#125;&#36;&#36;
其中
&#36;&#36;S_i=\frac1{n_i-1}\sum</em>{j=1}^{n_i}{(x</em>{ij}-\bar{x_i})\left(x</em>{ij}-\bar{x_i}\right)}^{\prime}&#36;&#36;
&#36;&#36;S_p=\frac1{n-k}\sum</em>{i=1}^k{(n_i-1)S_i}=\frac1{n-k}E&#36;&#36;
构造&#36;M&#36;统计量有
&#36;&#36;M=-2\mathrm{ln}\lambda=\left.(n-k)\ln|S_p|-\sum_{i=1}^k{(n_i-1)}\ln|S_i\right. &#36;&#36;
当原假设为真的时候
&#36;&#36;u=(1-c)M&#36;&#36;
近似服从自由度为&#36;\frac{1}{2}(k-1)p(p-1)&#36; 的卡方分布
其中
&#36;&#36;c=\Big(\sum_{i=1}^{k}\frac{1}{n_{i}-1}-\frac{1}{n-k}\Big)\frac{2p^{2}+3p-1}{6\left(p+1\right)\left(k-1\right)}&#36;&#36;
拒绝规则为
&#36;&#36;\text \text \ if \geqslant\chi\</p>
<h2>Preparatory knowledge of multiple statistics</h2>
<h3>Difference and Broadest</h3>
<p>In the one-dollar probability theory, the variance is used to measure the dissegregation of a random variable (variability level) and we explain the difference in the sample in the mathematical statistics; the multiple parts of the probability theory are used to explain the difference before the two random variables, although not explained in the mathematical statistics, but it is not difficult to introduce the difference in the sample (which also needs to be corrected)&#36;n-1&#36;The concept of mathematical software can also serve us;
The ACSM is a matrix; how to measure the total variability of random vectors in one number is the question we have to answer;</p>
<h4>Total variance</h4>
<p>The total variance is defined as
&#36;&#36;\mathrm{tr}\left(\boldsymbol{\Sigma}\right)=\sum_{i=1}^{p}\sigma_{ii}&#36;&#36;
He didn't consider the effects of the correlation between the samples; that was one of his flaws, but we would still use it in some places back there.
&#36;p=1&#36;And then it turns into a difference.</p>
<h4>Broad Difference</h4>
<p>The most commonly used definition for wide range is
&#36;&#36;\left|\Sigma\right|&#36;&#36;
The broad variance takes into account the correlation between variables; however, it may be misleading that the same broad difference is derived from two completely different matrixes.
&#36;p=1&#36;And then it turns into a difference.</p>
<h3>O'Hara and Ma's.</h3>
<h4>Orchid distance.</h4>
<p>&#36;p&#36;The distance of two o'clock in Vior's space is
&#36;&#36;d(x,y)=\sqrt{(x_{1}-y_{1})^{2}+(x_{2}-y_{2})^{2}+\cdots+(x_{p}-y_{p})^{2&#125;&#125;&#36;&#36;
In order to avoid rooting, we're used to using square distance.
&#36;&#36;d^2(x,y)=(x-y)^{\prime}(x-y)=(x_1-y_1)^2+(x_2-y_2)^2+\cdotp\cdotp\cdotp+(x_p-y_p)^2&#36;&#36;
In the case of different weight units, the OSD is of little practical value;
Even where units are the same, we need to standardize the data (usually z-scores); otherwise the calculation of the Euros distance is not meaningful; standardization is also the basis for the data pre-processing process.</p>
<h4>Ma's distance.</h4>
<p>When there is a linear relationship between random variables; the oscillation distance loses its original role and does not allow for the correct judgement of the isolation and proximity;</p>
<p>The Ma's distance was proposed to solve the problem; his essence was to rotate the axis, so that the correlation would disappear and then use the result of the A's distance.</p>
<p>Two vectors are defined as follows:
&#36;&#36;d^{2}\left(x,y\right)=\left(x-y\right)^{\prime}\boldsymbol{\Sigma}^{-1}\left(x-y\right)&#36;&#36;
The range of vector to total is defined as follows:
&#36;&#36;d^{2}\left(x,\pi\right)=\left(x-\mu\right)^{\prime}\Sigma^{-1}\left(x-\mu\right)&#36;&#36;
of which&#36;x,y&#36;A random vector. &#36;\mu,\Sigma&#36; The average vector and the all-inclusive matrix, respectively.</p>
<p>Let's give you a brief description of the nature of Mars' distance.</p>
<ul>
<li>About change and change&#36;y=Cx+b&#36;
Changes in scale can be expressed as&#36;y=Cx&#36;  Change the value of some plus or minus to&#36;y=x+b&#36;</li>
</ul>
<p>Ma's distance doesn't mean that the change doesn't affect our judgment about Ma's distance.</p>
<p>Standardized transformations are also a special form of transformations in the front, with no change in the distance between and after standardization
<strong>When the weights are not relevant, the horse's distance is the standardized distance.</strong></p>
