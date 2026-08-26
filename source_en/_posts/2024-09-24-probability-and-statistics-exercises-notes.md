---
title: 'Probability and Statistics Exercises: Sampling Distributions, Estimation, and Statistics'
title_zh: 概率与统计例题：抽样分布、参数估计与统计量性质
date: 2024-09-24 00:14:54 +0800
categories:
- Data Science
- Probability & Statistical Foundations
tags:
- Statistics
- Probability
author: Hyacehila
mathjax: true
hidden: true
excerpt: A overview of probability and statistics exercises, covering classical probability, sampling distributions, moment
  estimation, maximum likelihood estimation, and properties of statistics.
description: A overview of probability and statistics exercises, covering classical probability, sampling distributions, moment
  estimation, maximum likelihood estimation, and properties of statistics.
excerpt_zh: 整理古典概型、抽样分布、矩估计、极大似然估计和统计量性质等例题。
permalink: /blog/2024/09/24/probability-and-statistics-exercises-notes/
lang: en
translation_key: 2024-09-24-probability-and-statistics-exercises-notes
translation_status: machine
translation_source_hash: a7dcbf72932a32d8a07bbf55050955ed42dad5350f9a9135200c2af8440b4a03
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<h2>Classic examples of classical generalizations.</h2>
<h3>Throwing coins.</h3>
<p>A-a-n plus a coin, a-n-n-n, a-a-a-a-a-a-a- more than a-a-a-a-b?
<em>The study of the sample point is absolutely complex and requires complex probabilistic calculations, and the subject gives us an interesting idea.</em>
Remember&#36;A&#36;It's the probability of our target event.&#36;\bar{A}&#36; It's a less likely one to have a head than a b, obviously, and he's also more likely to have a side than a side, with symmetry. &#36;P(A)=P(\bar{A})且P(A)+P(\bar{A})=1&#36; And the answer is 0.5.</p>
<h3>Throwing in the ball (birthday issue, abstract form of Maxwell-Boltzman statistics)</h3>
<p>&#36;n for N-bars&gt;n The probability of each ball falling into each grid is equal.</p>
<ul>
<li>Each of the n grids has a ball. &#36;P=n!/N^{n}&#36;</li>
<li>Every single one of the n-bars has a ball. &#36;P=N!/N^{n}(N-n)!&#36;
<em>The calculations are not very complicated, but here's a description.</em></li>
</ul>
<h3>Probability explanation for drawing lots that are not sequential</h3>
<p>&#36;a个黑球 b个白球，求第k次摸出黑球的概率&#36;
<strong>Before we hit the ball, it was different.</strong> It needs to be discussed separately.<strong>It's the same situation before the ball was determined.</strong>
&#36;总数m=C_{a+b}^{k} 样本点数量n=C_{a+b-1}^{k-1}&#36;
<em>Both results are the same.</em>
<em>It's important to study classical generals, and you have to make sure you're consistent.</em></p>
<p>The questions in this article can also be addressed<a href="/en/blog/2023/03/18/elementary-probability-notes/">Primary probability theory: random events, probability models and random variables</a>、<a href="/en/blog/2024/10/09/advanced-probability-notes/">High probability theory: probability space, random variables and measurement base</a>How the concept of a relatively close read together is developed in different contexts.</p>
<h2>The mathematical statistical example of probabilities</h2>
<p>In the study of practical problems, we should not know the number or the rate of sub-products, and we need to look at him in the light of the situation, which is an example;</p>
<p>(b) Capture 1,200 fish from the pool, mark it and release it back, then re-eating 1,000 fish, 100 of which were marked and extrapolated to the total;</p>
<p>Obviously, it's a problem of hypergeometric distribution because he's a non-release sample, and he's given the total number of Ns; we know that the probability of these events is that
&#36;&#36;P=\left(\begin{array}{l}
n_{1} \
k
\end{array}\right)*\left(\begin{array}{l}
n-n_{1} \
r-k
\end{array}\right)/\left(\begin{array}{l}
n \
r
\end{array}\right)&#36;&#36;
of which&#36;n_{1}&#36;Special total,&#36;n&#36;is the total number,&#36;k&#36;The number of fish caught is special.&#36;r&#36;It's the number of the fish.</p>
<p>Based on<strong>It's a very similar estimate.</strong>We're looking at P-max. &#36;n=n_{1}*r/k&#36; It's our answer.</p>
<h2>Examples of mathematical statistical sampling distribution</h2>
<h3>1</h3>
<p>Set Master&#36;\xi\sim b(1,p)&#36;(two points distribution),&#36;(\xi_1,\xi_2,\cdots,\xi_n)&#36;To take a sub-species from this parent,&#36;\bar{\xi}&#36;If the value is equal to the child,&#36;p=0.2&#36; So, the sample capacity...&#36;n&#36; How much do you need to take to satisfy?
&#36;&#36;P(|\bar{\xi}-p|\leq0.1)\geq0.75;&#36;&#36;</p>
<p>If we do normal problems, we can get the distribution, but that's not gonna work.
I can see that.&#36;\sum\limits X_{i}&#36; It's two distributions. It's the basis of our back check.
&#36;&#36;P(|\bar{\xi}-p|\leq0.1)=P(0.1\leq\bar{\xi}\leq0.3)\=P(0.1n\leq\sum_{i=1}^{n}X_{i}\leq0.3n)\geq0.75&#36;&#36;
Direct checks are available.&#36;n=10&#36; </p>
<h3>2</h3>
<p>Subspecies (in the case of&#36;\xi_1,\xi_2,\xi_3)&#36;From Normal Matrix&#36;N(0,1)&#36;And...&#36;\eta_1=0.8\xi_1+0.{6}\xi_2&#36;, &#36;\eta_{2}=\sqrt{2}\left(0.3\xi_{1}-0.4\xi_{2}-0.5\xi_{3}\right),\eta_{3}=\sqrt{2}\left(0.3\xi_{1}-0.4\xi_{2}+0.5\xi_{3}\right)&#36;,Page&#36;\eta_1,\eta_2&#36;, &#36;\eta_{3}&#36;) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .&#36;\eta_1,\eta_2,\eta_3&#36; Marginal distribution</p>
<p>It's a subject that requires eyes.</p>
<p>First we can see it.  &#36;\eta_1,\eta_2,\eta_3&#36; Or are you obeying?&#36;N(0,1)&#36; Normal distribution
Subspecies (in the case of&#36;\xi_1,\xi_2,\xi_3)&#36;From the normal matrix, they make sure they're a normal vector and independent of each other.
New&#36;\eta_1,\eta_2,\eta_3&#36; It's an original linear combination.</p>
<p>We know if it was an independent normal vector, then we'd have to multiply the active matrix or an independent normal vector.
We can really verify that this shift in coefficient matrix is a positive matrix.</p>
<p>And so...&#36;\eta_1,\eta_2,\eta_3&#36; The combined density is the sum of the density functions.&#36;N(0,1)&#36; ）</p>
<h3>3</h3>
<p>Set Master&#36;\xi&#36;The distribution function is&#36;F(x),(\xi_1,\xi_2,...,\xi_n)&#36;It's a sub-spectrum from this matrix.&#36;F(x)&#36;The second-class is there.&#36;\overline{\xi}&#36;Subsistence average, test (&#36;\xi_i-\bar{\xi})&#36; with (&#36;\xi_j-\bar{\xi})&#36; the relevant coefficient is
&#36;&#36;\rho=-\frac{1}{n-1}&#36;&#36;</p>
<p>It still requires a basic definition of the subject. </p>
<p>The formula for the relevant coefficient is
&#36;&#36;Corr\left(X,Y\right)=\frac{Cov\left(X,Y\right)}{\sqrt{Var\left(X\right)}\sqrt{Var\left(Y\right)&#125;&#125;=\frac{Cov\left(X,Y\right)}{\sigma_{X}\sigma_{Y&#125;&#125;&#36;&#36;
And... &#36;Cov(X,Y)=E(XY)-E(X)E(Y)&#36;<br>So our problem has turned into a problem of desired resolution.
Because the averages of the extraction and sample and sample are not independent, we need to untangle the averages and expect the small items and then we can get the results of the coefficients in the subject.</p>
<h2>Examples of mathematical statistical rectangulation estimates</h2>
<h3>1</h3>
<p>Set total &#36;x&#36; Average and variance, respectively &#36;\mu&#36; and &#36;\sigma^{2}&#36;,&#36;X_1,X_2,\cdotp\cdotp,X_n&#36;It's the whole thing.&#36;X&#36; If the whole of the originals of the first and second steps are present, please &#36;\mu&#36;and &#36;\sigma^2&#36; The rectangular estimate.</p>
<p>Based on the rectangular estimation, we know that two equations are required.
&#36;&#36;\mu=EX,\quad E(X^{2})=DX+E^{2}X=\mu^{2}+\sigma^{2}&#36;&#36;
So the whole thing is the same as the sample.
&#36;&#36;\begin{cases}\mu=\frac1n\sum_{i=1}^nX_i,\\\mu^2+\sigma^2=\frac1n\sum_{i=1}^nX_i^2,\end{cases}&#36;&#36;
Expressions of classic processing of simplicitys can be obtained (square equals second-order rectangular minus desired squares)
&#36;&#36;\left.\left{\begin{matrix}{\mu=\overline{X},}\{\sigma^{2}=\frac{1}{n}\sum_{i=1}^{k}(X_{i}-\overline{X})^{2&#125;&#125;\\end{matrix}\right.\right.&#36;&#36;
This example tells us that, regardless of the distribution,
<strong>The rectangular estimate for the overall average is the sample average, and the rectangular estimate for the overall variance is the sample variance</strong>
This is going to be a very common theorem.</p>
<p><em>Look, the rectangular estimates are biased, and we'll be back to that.</em></p>
<h3>2</h3>
<p>Sets the overall probability density as&#36;f(x)=\frac1{2\sigma}e^{-\frac{|x|}\sigma}&#36; Rectangular estimate of solver parameters</p>
<p>A parameter requires only a rectangular equation to calculate the average of the total.
&#36;&#36;E(X)=\int_{-\infty}^{+\infty}x\cdot\frac1{2\sigma}e^{-\frac{|x|}\sigma}dx=0&#36;&#36;
The fractions of the odd function that you can get without counting.
Obviously, in this case, we can't get a rectangular estimate from this equation, so we calculate the second-order rectangular.
&#36;&#36;00begin{gathered}
=(x^fty^x\frat\frac^e^\x\sigma}dx=dx=dfx=int}x^x^xxcdot\frac^xx
Other Organiser<em>{0}^{+\infty}+\int</em>Other Organiser
== sync, corrected by elderman == @elder man
I'm sorry, I'm sorry.
Or is it about symmetric simplification and segmentation?</p>
<h2>A mathematically very similar estimate.</h2>
<h3>1</h3>
<p>Overall&#36;X&#36;And the density is...
&#36;&#36;\left.f(x;\lambda)=\left(begin{matrix}\lambda e^-\lambda x},&amp;x&gt;0\0,&amp;\\text{Other,} \\right.\right. ;&#36;
Of which &#36;lambda&gt;0&#36;为未知参数，&#36;X_1,X_2,...,X_n&#36; 是取自总体&#36;X&#36;的一组样本， 求&#36;\lambda's \mbda's \spectacular and rectangular estimate</p>
<p>Calculate Appearance Functions
&#36;&#36;\left.L (\lambda)=\prod (i)=f(x}i;\lambda)=\left(begin{array&#125;&#125;\lambda^(i)=e^lambda x i},&amp;x_{i}&gt;0,i=1,2,\cdots,n;\0,&amp;\text \right.\right.&#36;&#36;
明显如果取等于0的时候没法计算估计量 所以找全部大于0 的样本 计算对数似然函数
&#36;&#36;\ln L(\lambda)=n\ln\lambda-\lambda\sum_{i=1}^{n}x_{i}&#36;&#36;
求导计算得到似然方程
&#36;&#36;\frac{d\ln L(\lambda)}{d\lambda}=\frac{n}{\lambda}-\sum_{i=1}^{n}x_{i}=0&#36;&#36;
得到极大似然估计量
&#36;&#36;\hat{\lambda}=\frac{1}{\overline{X&#125;&#125;&#36;&#36;
使用上一节的手段的计算矩估计量
&#36;&#36;EX=\int_{-\infty}^{+\infty}xf(x;\lambda)dx=1/\lambda &#36;&#36;
&#36;&#36;{\fnH00FFFF}
It's clear that the exact estimate of the problem is consistent with the very similar estimate.</p>
<h3>2</h3>
<p>Set total&#36;x\sim N(\mu,\sigma^2)&#36;, of which &#36;\mu,\sigma^2&#36; None known, set&#36;X_1,X_2,...X_n&#36; It's from...&#36;X&#36;A sample of the &#36;\mu&#36;and &#36;\sigma^2&#36; And it's a very, very similar estimate.</p>
<p>The multi-parameter approach is exactly the same or the radicalization that's presented in the mathematical analysis.
Calculate Appearance Functions
&#36;&#36;L(\mu,\sigma^{2})=\prod_{i=1}^{n}f(x_{i};\mu,\sigma^{2})=\prod_{i=1}^{n}\frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{(x_{i}-\mu)^{2&#125;&#125;{2\sigma^{2&#125;&#125;}&#36;&#36;
Logarithmic function
&#36;&#36;\ln L(\mu,\sigma^{2})=-\frac{n}{2}\ln2\pi\sigma^{2}-\frac{1}{2\sigma^{2&#125;&#125;\sum_{i=1}^{n}(x_{i}-\mu)^{2}&#36;&#36;
Two parameters, you get the semblance of equation, you get two different directions.
&#36;&#36;00\&amp;\frac{\partial\ln L(\mu,\sigma^2)}{\partial\mu}=\frac1{\sigma^2}\sum_{i=1}^{n}(x_i-\mu)=0,\&amp;\frac{\partial\ln L(\mu,\sigma^2)}{\partial\sigma}=-\frac n{\sigma}+\frac1{\sigma^3}\sum_{i=1}^{n}(x_i-\mu)^2=0,\end{aligned}&#36;&#36;
解方程得到
&#36;&#36;{\cHFFFFFF}{\cH00FF00} {\cHFFFFFF} {\cHFFFFFF}{\cH00FF00} {\cHFFFFFF} {\cHFFFFFF} {\cHFFFFFF} {\cHFFFFFF}{\cH00FF00} {\cH00FFFF} {\cH00FFFF} {\cHFFFFFF} {\cH00FF00} {\cH00FF00} {\cH00FF00} {\cH00FF00} {\cH00FF00} {\cH00FF00} {\cH00FF00} {\cH00FF} {\cH00FF} {\cH303030D3F4} {\cH00F4} {\cH3030D3D} {\cH00} {\cH30303030303030D} \cH303030303030303030303030303030303030303030303030303030 \cH30303030303030303030303030303030303030303030303030303030
The large, seemingly normal distribution is equal to the rectangular estimation.</p>
<h3>3</h3>
<p>Set total&#36;x&#36;Obedience evenly distributed&#36;U[0,\theta]&#36;..and&#36;\theta&#36;Unknown parameter, &#36;X_1,X_2,...,X_n&#36; It's the whole thing.&#36;X&#36;A sample of the sample, please.&#36;\theta&#36;And it's a very, very similar estimate.</p>
<p>Samples Appearingly Function
&#36;&#36;\left.L (\theta)=\prod i=n}f(x{i};\theta)=\left{begin{array}ll}\frac{1\theta^n}&amp;♪leq x theta, i=1,2,\cdots, n; ♪text{, other}. ♪end{array}\right.\right.&#36;</p>
<p>Obviously, direct guidance to an apparent function or to a logarithmic function cannot be greatly enhanced. This function is the weakness of the method presented in the mathematical analysis, and he's not the almighty one.</p>
<p>Obviously, the function is reduced in a single tone. We need to make it as far as possible.&#36;\theta&#36; Smaller
But we must have to meet. &#36;&#36;0\leq x_{i}\leq\theta({i=1,2,\cdots,n})&#36;&#36;Well, we can take this, in that case, a very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very,
&#36;&#36;\hat{\theta}=\max_{1\leq i\leq n}{X_{i&#125;&#125;&#36;&#36;</p>
<h3>4</h3>
<p>Set&#36;\xi_1,\xi_2,\cdots,\xi_n&#36; It's from the logarithmic normal distribution matrix&#36;\xi&#36; A sub-section, in &#36;\xi&#36; - &#36;N(\mu,\sigma^2),-\infty&lt;\mu&lt;+\infty&#36; ,&#36;0&lt;\sigma&lt;+\infty&#36;.试求 &#36;\xi&#36; 的期望值 &#36;E\xi&#36; 和方差 &#36;D\xi&#36; is a very similar estimate</p>
<p>This is not about our parameters.&#36;\mu,\sigma^2&#36;And the distributional characteristic numbers are, in fact, an expression that contains the distribution parameters, and we need to get this expression and then bring in the parameters a very significant estimate, which is the very similar estimate of the characteristics.</p>
<p>The distribution of the need for research&#36;\xi&#36; It's a function of a normal distribution that needs to be used to simplify the calculation of expectations and differences.
Remember&#36;\eta=ln\xi&#36; then&#36;\eta\sim N(\mu,\sigma^2)&#36;  &#36;\xi=e^{\eta}&#36;  Calculate using the desired and differential function formula
&#36;&#36;E\xi=\exp\left{\frac{1}{2}(2\mu+\sigma^{2})\right}&#36;&#36;
&#36;&#36;D\xi=e^{2\mu+\sigma^{2&#125;&#125;[e^{\sigma^{2&#125;&#125;-1].&#36;&#36;
Calculate using theorem in probability theory&#36;\xi&#36;The distribution is
&#36;&#36;\left.p \y\right=\left{begin{array}\sqrt{1\spi\sigma}\xp{2sigma^2}\frac}1}y}&amp;y&gt;0,\0,&amp;\text{else}\end{array}\right.\right.&#36;&#36;
他的极大似然估计前面已经计算过了
&#36;&#36;\begin{cases}\hat{\mu}=\frac{1}{n}\sum_{i=1}^{n}X_{i}=\overline{X},\\hat{\sigma}^{2}=\frac{1}{n}\sum_{i=1}^{n}(X_{i}-\mu)^{2}\end{cases}&#36;&#36;
将&#36;X_{i}&#36; 变化为 &#36;ln\xi_{i}&#36;  把他们的结果带入到
&#36;&#36;E\xi=\exp\left{\frac{1}{2}(2\mu+\sigma^{2})\right}&#36;&#36;
&#36;&#36;D\xi=e2m+sigma}[e^sigma^1]. &#36;
It's a very large estimate of the number of features.</p>
<h2>Guidelines for the assessment of mathematical statistical estimates</h2>
<h3>No bias</h3>
<h4>1</h4>
<p>Set total&#36;x\sim N(\mu,\sigma^2)&#36;, of which &#36;\mu,\sigma^2&#36; None known, set&#36;X_1,X_2,...X_n&#36; It's from...&#36;X&#36;A sample of the &#36;\mu&#36;and &#36;\sigma^2&#36; The large, seemingly neutral estimates and rectangular estimates of the
For Rectangular Estimates
That's a good idea.<em>{M}=2\bar{X}&#36;&#36;
则有
&#36;&#36;E(\hat{\theta}</em>{M})=2E(\bar{X})=2E(X)=2\cdot\frac{\theta}{2}=\theta &#36;&#36;
对于极大似然估计量
&#36;&#36;\hat{\theta}=\max_{1\leq i\leq n}{X_{i&#125;&#125;&#36;&#36;
如何研究&#36;E[\hat\theta]&#36;
能看出来 实际上&#36;\hat\theta&#36; 是一个顺序统计量 因此我们是能够给出他的密度函数的 所以我们可以从此计算其期望
密度函数为
&#36;&#36;\left.f_{\max}(z)=\left{\begin{matrix}n\frac{1}{\theta}\bigg(\int_0^z\frac{1}{\theta}\mathrm{d}z\bigg)^{n-1},0&lt;z&lt;\theta\0,&amp;\textbf \textbf \matrix}right.\right.=left{begin{matrix}n\frac{1theta}\big^&lt;z&lt;\theta\0,&amp;\textbf \textbf \right.&#36;&#36;
计算期望有
&#36;&#36;\begin{array}{c}{E(\hat{\theta}<em>{L})=\int</em>{-\infty}^{\infty}z\cdot f_{\max}(z)\mathrm{d}z}\{=\int_{0}^{\theta}z\cdot n\frac{1}{\theta}\biggl(\frac{z}{\theta}\biggr)^{n-1}\mathrm{d}z=\frac{n}{n+1}\theta&lt;It's not a good idea.
Actually, this is a partial estimate, which can be converted to a neutral estimate by correcting the coefficient.</p>
<h4>2</h4>
<p>All that explains is how to judge whether a quantity is a neutral estimate.</p>
<p>Set Random Variables&#36;\xi&#36;Subscribe to the two distributions
&#36;&#36;
\left.P\left(\begin{array}{c}{\xi=x}\\end{array}\right.\right)=\left(\begin{array}{c}{n}\{x}\\end{array}\right)\theta^{x}\left(\begin{array}{c}{1-\theta}\\end{array}\right)^{n-x},x=0:,1:,\cdots
&#36;&#36;
Test &#36;\theta^2&#36; No-Effort estimate</p>
<p>We can't imagine a neutral estimate, so all similar questions are based on some estimates we know of, and we use some computational properties to arrive at a neutral estimate.
We know.
&#36;&#36;E\bar{\xi}=E\xi=n\theta,ES^{*2}=D\xi=n\theta(1-\theta)=n\theta-n\theta^{2}&#36;&#36;
This is the most basic estimate.&#36;\theta^2&#36;What about the form? Watch.
&#36;&#36;E\frac{\vec{\xi}-S^{*2&#125;&#125;{n}=\frac{E\vec{\xi}-ES^{*2&#125;&#125;{n}=\frac{n\theta-(n\theta-n\theta^{2})}{n}=\theta^{2}&#36;&#36;
And so...&#36;\theta^2&#36;The no-even estimate is
&#36;&#36;\frac{\vec{\xi}-S^{*2&#125;&#125;{n}&#36;&#36;
We use that to solve other problems, and sometimes the subject gives us some estimates.</p>
<h3>Average error</h3>
<p>Normal Total&#36;N(\mu,\sigma^2)&#36;Mean&#36;\mu&#36;Difference&#36;\sigma^{2}&#36;The average MLE error study averages clearly we know that the MLE estimate is neutral, so...&#36;bias=0&#36; &#36;MSE=var(\hat\mu)=\frac{\sigma^2}{n}&#36;
MLE estimation for differentials
&#36;&#36;\begin{gathered}
b(\theta,\hat{\sigma}^{2})=E(\hat{\sigma}^{2})-\sigma^{2}=-\frac{\sigma^{2&#125;&#125;{n}, \
Var(\hat{\sigma}^{2})=Var\biggl(\frac{(n-1)S_{n}^{*2&#125;&#125;{n}\biggr)=Var\biggl(\frac{(n-1)S_{n}^{*2&#125;&#125;{\sigma^{2&#125;&#125;\frac{\sigma^{2&#125;&#125;{n}\biggr) \
=\frac{\sigma^{4&#125;&#125;{n^{2&#125;&#125;Var\Bigg(\frac{(n-1)S_{n}^{*2&#125;&#125;{\sigma^{2&#125;&#125;\Bigg)=\frac{\sigma^{4&#125;&#125;{n^{2&#125;&#125;2(n-1)
\end{gathered}&#36;&#36;
So...
&#36;&#36;MSE=Var(\hat{\sigma}^2)+b^2(\theta,\hat{\sigma}^2)=\frac{\sigma^4(2n-1)}{n^2}.&#36;&#36;
Study of impartial revision estimates
&#36;&#36;00\
&amp;MSE = Var\Bigg ((n-1) S (n)* (x)}sigma^ (x)} (x)0 (x)= (x)= (x)= (x)= (x)= (x)= (x)= (x)= (x)= (x)= (x)= (x)= (x)= (x)= (x) (x) (x) (x) (x) (x) (x) (x) (x) (x) (x) (x) (x) (x) (x) (x) (x) (x) (x) (x) (x) (x) (x) (x) (x) (x) (x) (x) (x) (x) (x) (x) (x) (x) (x) (x) (x) (x) (x) (x) (x) (x) (x) (x) (x) (x) (x) (x) (x) (x) (x) (x) (x) (x) (x) (
I'm sorry, I'm sorry.
I can see that the MLE is slightly smaller than MSE.</p>
<h2>Mathistics Fisher Information DataQuote</h2>
<p>Fisher information volume for index distribution
&#36;&#36;p(x; \theta)=\frac{1 \text\left{-\frac{x\theta},\quad x&gt;0,\theta&gt;0&#36;&#36;
没说就按照只抽取一个样本进行计算
&#36;&#36;\frac{\partial}{\partial\theta}\ln p(x;\theta)=-\frac{1}{\theta}+\frac{x}{\theta^{2&#125;&#125;=\frac{x-\theta}{\theta^{2&#125;&#125;&#36;&#36;
则有
&#36;&#36;I(\theta)=E\Bigg(\frac{x-\theta}{\theta^2}\Bigg)^2=\frac{\mathrm{Var}(x)}{\theta^4}=\frac{1}{\theta^2}&#36;&#36;</p>
<h2>Mathematical Statistical Illustrative Case</h2>
<p>Set&#36;\xi_1,\xi_2,\cdots,\xi_n&#36;To take from the normal matrix&#36;N(\mu,\sigma^2)&#36;A sub-protest.
&#36;(1)\hat{\mu}=\overline{\xi}&#36;Yes.&#36;\mu&#36;A valid estimate;
(2) If&#36;\mu&#36;Known, then&#36;S_1^2=\frac1n\sum_{i=1}^n(\xi_i-\mu)^2&#36;Yes.&#36;\sigma^{2}&#36;Effective estimates
If&#36;\mu&#36;unknown&#36;S_{2}^{2}=\frac{1}{n-1}\sum_{i=1}^{n}(\xi_{i}-\overline{\xi})^{2}&#36;No, it's not.&#36;\sigma^{2}&#36;Effective estimates</p>
<p>Inferences on average estimates of previously learned
&#36;&#36;D\overline{\xi}=\frac{\sigma^2}n&#36;&#36;
Use inference to calculate Fisher's information.
&#36;&#36;I(\mu)=\frac{1}{\sigma^{2&#125;&#125;&#36;&#36;
It's easy to prove effectiveness.</p>
<p>Estimates of variance
If&#36;\mu&#36;- I know. - Research is impartial.
&#36;&#36;S_1^2=\frac1n\sum_{i=1}^n(\xi_i-\mu)^2&#36;&#36;
&#36;&#36;ES_{1}^{2}=\frac{1}{n}E(\xi_{i}-E\xi_{i})^{2}=\frac{1}{n}\sum_{i=1}^{n}D\xi_{i}=\sigma^{2}&#36;&#36;
There is no doubt that there is a prerequisite for discussing effective estimates.
The final calculation of Fisher's information based on the results of the promotion is obtained from the lower level of CR
We're counting on...&#36;\sigma^2&#36; The amount of information you're going to need to be able to get a guide.&#36;\sigma^2&#36; As a whole.
&#36;&#36;I(\mu)=\frac{1}{2\sigma^{4&#125;&#125;&#36;&#36; It's easy to know.
&#36;&#36;\frac{1}{\sigma^{2&#125;&#125;\sum_{i=1}^{n}(\xi_{i}-\mu)^{2}\sim\chi^{2}(n)&#36;&#36;
Difference&#36;2n&#36;
And there is.
&#36;&#36;D(S_1^2)=\frac{2\sigma^4}n=\frac1{nI(\sigma^2)}&#36;&#36;
It's a valid estimate.</p>
<p>If&#36;\mu&#36;Unknown. We know it's impartial.
Fisher's information remains the same as the questions.&#36;S^2&#36; The distribution has changed.
&#36;&#36;\frac1{\sigma^2}\sum_{i=1}^n(\xi_i-\overline{\xi})^2=\frac{(n-1)S_2^2}{\sigma^2}\sim\chi^2(n-1)&#36;&#36;
So there is.
&#36;&#36;D(S_2^2)=\frac{2\sigma^4}{n-1}\neq\frac1{nI(\sigma^2)}&#36;&#36;
Not a good estimate, but a good estimate.</p>
<h2>The subject of adequate statistical data in mathematical statistics</h2>
<p>Set total &#36;x&#36; You're gonna have to follow two points.&#36;B(1,p)&#36;that is&#36;P\left(\mathrm{X=x}\right)=p^{x}\left(1-p\right)^{1-x},x=0,1&#36;, zero of which&lt;p&lt;1&#36; &#36;\quad(X_1,X_2,...,X_n)&#36; 为来自总体&#36;X&#36;一个样本， 研究统计量&#36;\overline{x}=frac1\sum i=nX i&#36; ^nx i&#36; \nx i i i i i i i i i i i i i i i i i i i i i i i i i i i i i i i i i i i i i i i i i i i i i i i i i i i i i i i i i i n i n i i n i i n i i i i i n i i i i i i i i i i </p>
<p>The most basic definition used here to verify it.
Let's start with the statistical distribution. It's easy to know.
&#36;&#36;n\overline{X}=\sum_{i=1}^{n}X_{i}-B(n,p),&#36;&#36;
So the condition is... &#36;\overline{X}=\frac{k}{n}&#36;   That's what statistics take for a particular value.
Conditions distribution of study samples
&#36;&#36;\begin{gathered}
P\left(X_{1}=x_{1},X_{2}=x_{2},\cdots,X_{n}=x_{n}\left|\overline{X}=\frac{k}{n}\right)\right.  \
=\frac{P\left(X_{1}=x_{1},X_{2}=x_{2},\cdots,X_{n}=x_{n},\overline{X}=\frac{k}{n}\right)}{P\left(\overline{X}=\frac{k}{n}\right)}
\end{gathered}&#36;&#36;
&#36;&#36;=\begin{cases}\dfrac{P(X_1=x_1,X_2=x_2,\cdots,X_n=x_n)}{P(n\overline{X}=k)},\text{如果}\sum_{i=1}^nx_i=k,\0,\text{如果}\sum_{i=1}^nx_i\neq k,\end{cases}&#36;&#36;
The definition of the introduction to the distribution is simplified
&#36;\left. \left{begin{aligned}&amp;\frac1},\text \sif \sum i=k,\save=&amp;\textbf{0, if \sum i\neq k, \right.\right.&#36;
It's obvious that our final results and parameters in distribution&#36;p&#36; It's not, so...&#36;\overline{X}&#36; is the parameter&#36;p&#36; Quantified</p>
<h2>Statistically complete case</h2>
<p>Set&#36;X_1,X_2,...,X_n&#36;It's from the two-point distribution.&#36;B(1,p)&#36;The sample. From the previous case.&#36;\overline{X}=\frac1n\sum_{i=1}^nX_i&#36; Yes.&#36;p&#36; Adequately measured. Verify Down&#36;\overline{X}&#36; It's a full-scale count.
Easy to give.&#36;\overline{X}&#36; The Law of Distribution
&#36;&#36;P\left{\overline{X}=\frac kn\right}=C_n^kp^k\left(1-p\right)^{n-k}&#36;&#36;
Assumptions exist&#36;g(X)&#36;  Meet the requirements above (expected function conclusions used)
&#36;&#36;E_p[g(X)]=\sum_{k=0}^ng{\left(\frac kn\right)}C_n^kp^k(1-p)^{n-k}=0&#36;&#36;
Equivalent to
&#36;&#36;(1-p)^n\sum_{k=0}^ng{\left(\frac kn\right)}C_n^k{\left(\frac p{1-p}\right)}^k=0&#36;&#36;
Equivalent to
&#36;&#36;\sum_{k=0}^ng(\frac kn)C_n^k\left(\frac p{1-p}\right)^k=0&#36;&#36;
You can see that to satisfy this equation, you need to...
&#36;&#36;g\left(\frac kn\right)=0&#36;&#36;
That's what we're gonna do.</p>
<h2>Mathematical Statistics UMVUE</h2>
<h3>1</h3>
<p>Set two points to be distributed as a whole&#36;p(x;\theta)=p^{\mathrm{x&#125;&#125;(1-p)^{1-x},x=0,1&#36;Please.&#36;p&#36;UMVUE
It's easy to know.
&#36;&#36;
\overline{X}=\frac{1}{n}\sum_{i=1}^{n}X_{i}\text{}.
&#36;&#36;
is the parameter&#36;p&#36;Quantified and complete statistics
&#36;&#36;\overline{X}=\frac{1}{n}\sum_{i=1}^{n}X_{i},&#36;&#36;
is the parameter&#36;p&#36; A non-event estimate
So this is the only UMVUE.</p>
<h3>2</h3>
<p>Overall Porcelain distribution parameters&#36;\lambda&#36;UMVUE
&#36;&#36;p(k,\lambda)=\frac{\lambda^{k&#125;&#125;{k!}e^{-\lambda}&#36;&#36;
It's easy to know.
&#36;&#36;T(X_{1},X_{2},\cdots,X_{n})=\sum_{i=1}^{n}X_{i}.&#36;&#36;
is the parameter&#36;\lambda&#36;Yes.<strong>Fully statistically complete</strong>
And we know that.
&#36;&#36;\bar{X}=\frac{1}{n}\sum_{i=1}^{n}X_{i}&#36;&#36;
is the parameter&#36;\lambda&#36;No-Effort estimate
This is a full statistical function, so he's UMVUE.</p>
<h3>3</h3>
<p>Overall&#36;X&#36; The density function is
&#36;p(x)=\left(begin{array}ll}
The \lambda e^, \lambda x, \ambda \ &amp; x\ge0, \
0, &amp; \text {Other.}
I'm sorry, I'm sorry.
Please.&#36;\frac{1}{\lambda}&#36;UMVUE
We know according to the theory of the index distribution. &#36;\sum\limits X_i&#36; It's a full-fledged statistical volume.
And because... &#36;\overline{X}&#36; It's a neutral estimate of the parameters to be asked for, so he's a UMVUE.</p>
<h3>4</h3>
<p>Set two distributions as a whole&#36;X\sim B(n,p),&#36; Please.&#36;{p}(1-p)&#36; UMVUE
According to the theory of the index distribution, we know that&#36;X=\sum\limits X_i&#36; It's a full-fledged statistical volume.
Remember&#36;\overline{X}=\frac1n\sum_{i=1}^nX_i=\frac1nX&#36;, and&#36;X&#36;Subscribe to the two distributions&#36;B(n,p).&#36; And...
<strong>Why do you think that? We know.&#36;\overline{X}&#36;Yes.&#36;p&#36;So I want to build a new estimate from here.</strong>
&#36;&#36;E(\frac Xn(1-\frac Xn))\color{}{=\frac{n-1}np(1-p)}&#36;&#36;
So...
&#36;&#36;\varphi(\bar{X})=\frac{n}{n-1}\frac{X}{n}(1-\frac{X}{n})=\frac{n}{n-1}\bar{X}(1-\bar{X})&#36;&#36;
It's one.&#36;p(1-p)&#36;No-Effort estimate
It's a full count.&#36;\overline{X}&#36; The function (the mean of the distribution of two points) is therefore UMVUE</p>
<h3>5</h3>
<p>Set total&#36;X&#36;Yes.&#36;[0,\theta]&#36;Observance evenly distributed, where&#36;\theta&#36;It is an unknown parameter.&#36;X_1,X_2,\cdots,X_n&#36;A sample from the whole population, asking for parameters&#36;\theta&#36;UMVUE
&#36;&#36;00begin{aligned}p (x 1,x 2,\cdots,x n;\theta)&amp;=\begin{cases}\frac1{\theta^n},&amp;\mathbf{0}\leq x_{(1)}\leq x_{(n)}\leq\theta,\0,&amp;\mathrm{otherwise}.&amp;\end{cases}\&amp;=\frac1{\theta^n}I_{<em>{(X</em>{(n)}\leq\theta)&#125;&#125;I_{<em>{(X</em>{(1)}\geq0)&#125;&#125;\end{aligned}&#36;&#36;
根据因子分解定理知
&#36;&#36;X{<em>{(n)&#125;&#125;=\max{x</em>{1},x_{2},\cdots,x_{n&#125;&#125;&#36;&#36;
是一个参数的充分统计量 能证明它也是完备的，证明方法是用定义；从函数期望为0去推导&#36;g(X)&#36; 为0 就可以了 需要一步对积分上限函数求导
&#36;&#36;\int_{0}^{\theta}[g(X)]{\cdot}n\frac{x^{n-1&#125;&#125;{\theta^{n&#125;&#125;dx=0&#36;&#36;
所以
&#36;&#36;[g(X)]\theta^{n-1}=0&#36;&#36;
因此
&#36;&#36;g(X)=0&#36;&#36;
是完备统计量 证毕
能得到它的无偏估计
&#36;&#36;{\fnH00FFFF} {\fnH00FF}
It's a fully statistical function, so it's UMVUE.</p>
<h3>6</h3>
<p>Set total&#36;X&#36;Subscribe to normal distribution&#36;N(\mu,\sigma^2)&#36;, &#36;\theta=(\mu,\sigma^2)&#36;I don't know.&#36;X_1,X_2,\cdots,X_n&#36;It's a sample from the whole population.&#36;\mu&#36;and&#36;\sigma^{2}&#36;Yes.&#36;UMVUE.&#36; And verify whether the difference in the UMVUE of the parameters is below the C-R boundary
It's not actually below the CR.</p>
