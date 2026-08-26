---
title: 'Mathematical Statistics: Populations, Samples, and Sampling Distributions'
title_zh: 数理统计：总体与样本、统计量与抽样分布
date: 2023-03-18 21:28:08 +0800
categories:
- Data Science
- Probability & Statistical Foundations
tags:
- Statistics
- Mathematical Statistics
- Statistical Inference
author: Hyacehila
mathjax: true
hidden: true
excerpt: Covers populations and samples, statistics, sampling distributions, parameter estimation, hypothesis testing, ANOVA,
  and common tests.
description: Covers populations and samples, statistics, sampling distributions, parameter estimation, hypothesis testing,
  ANOVA, and common tests.
excerpt_zh: 整理总体与样本、统计量、抽样分布、参数估计、假设检验、方差分析和常见检验方法。
permalink: /blog/2023/03/18/mathematical-statistics-notes/
lang: en
translation_key: 2023-03-18-mathematical-statistics-notes
translation_status: machine
translation_source_hash: 66a3e72e7502a0ebd2d1225e55596522dcb981fddec567712136ff79a0fe0b6f
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<h2>Basic concepts</h2>
<p>Statistics are the reverse of probabilism, and in probabilism we have the essence of what happens, to study the results of what happens, and in statistics we reverse the principles by looking at the real world.
Mathematical statistics are the foundational knowledge of the entire statistical system, and a lot of the rest goes from here.
Mathematical statistics: study of the collection of data with random errors using probabilistic and mathematical methods and analysis of the collected data (statistical analysis) under a set model (statistical analysis) to infer the problems studied (statistical extrapolation)
And we'll start with the most basic concept of statistics, and we'll build the foundation of classical statistics, which is the frequency school; the corresponding Bayes school will be presented separately later.</p>
<p>Before embarking on a specific narrative of statistical content, we need to distinguish between another division of statistical science, which is dedicated to the study of observations per se, and statistical inferences, which focuses on the analysis of subjects per se from the point of view.</p>
<h3>Overall and sample</h3>
<h4>Total and individual</h4>
<p>The body of the subject is referred to as the sum; the elements that make up the whole are referred to as the individual
Indicators that are often of limited interest in statistical research, so the total number of indicators to be studied is at this point in time, and the corresponding number of individuals still present.
It's obvious that we can see these numbers as random variables.
At this point, the total is a random variable, and his distribution and numerical characteristics are called the total distribution of digital characteristics.
The central point is that it's a probability distribution.
Because the parameters are unknown, we can also say that overall is a probability distribution group.
Sometimes the aggregate cannot be expressed in a parameter distribution, which we call non-parameter aggregate, and at this point applies to the treatment of non-parametric statistics.</p>
<h4>Samples and samples</h4>
<p>In order to extrapolate the overall distribution and characteristics, a number of individuals are drawn from the general population for observation tests to obtain information about the overall population, which is referred to as “sampling”;
The number of individuals included in the sample is called sample capacity.
Because the sample is random, each individual is a random variable with a capacity of&#36;n&#36;The sample can be considered as&#36;n&#36;V-random variable
Once we get a sample, we get it.&#36;n&#36;A specific number, referred to as an observation value for a sample, a short sample value
Those samples that give exact values are called complete samples.
Only given the range of sample observations called group samples</p>
<h4>Simple random sample</h4>
<p>Since the purpose of sampling is to provide statistical inferences to the population as a whole and in order for the sample taken to reflect the overall information well, consideration must be given to the most common sampling method.<strong>Simple random sampling</strong> He's satisfied.</p>
<ul>
<li>All samples taken and total equal distribution</li>
<li>The samples are independent of each other.
A simple random sample is called a simple random sample.
In later studies, all samples are considered simple random.</li>
</ul>
<p>Finally.
In fact, the information we obtained from the samples was specific and determined; they were sample values rather than samples;
The job of statistics is to look at the general characteristics from the available data, and the sample is our bridge; the reason we're able to do this is because the sample values are determined by the total distribution.</p>
<h3>Statistics and their distribution</h3>
<h4>Statistics</h4>
<p>The sample is the basis for statistical extrapolation. But when applied in practice, it is generally not the use of the sample itself, but the construction of appropriate functions, i.e. statistics, that are tailored to the specific issue, using these functions for research.</p>
<p>Definitions:&#36;X_{i}&#36;It's a general sample. &#36;x_{i}&#36; It's a sample value.&#36;g(X_{1},...X_{n})&#36;For statistics.&#36;g(x_{1}...x_{n})&#36;For the corresponding observation value
Note that statistics must not contain any unknown parameters in the overall distribution.
Like what? &#36;\frac1{\sigma^2}\sum_{i=1}^n(X_i-\mu)^2&#36; Whether or not statistics depend on parameters&#36;\sigma,\mu&#36;Known
The average of the samples, the difference, the standard difference, the steps of the square, the center of the square are important statistics, and their expressions are no longer repeated. </p>
<h4>Fix sample differences</h4>
<p>The difference we're using in probability is
&#36;&#36;S^{2}=\frac{1}{n}\sum_{i=1}^{n}(X_{i}-\mu)^{2}&#36;&#36;
But in statistics, in order to meet the problem of unbiased estimates that we will point out later, we need to make the following amendments, in addition to replacing the overall average with sample averages.
&#36;&#36;S^{2}=\frac{1}{n-1}\sum_{i=1}^{n}(X_{i}-\overline{X})^{2}=\frac1{n-1}[\sum_{i=1}^nX_i^2-n\overline{X}^2]&#36;&#36;</p>
<h4>Three sample distributions</h4>
<p>The statistics are a function determined entirely by the template, and he's also a random variable, and his distribution is called sample distribution, and here's three very important sample distributions, all of which come from the most classic general pattern of normality.</p>
<h5>Carside Distribution&#36;\chi ^{2}&#36;Distribution)</h5>
<p>Definition: Setup&#36;X_{1},X_{2}...X_{n}&#36; It's from the standard normal general.&#36;N(0,1)&#36;The sample is statistical.
&#36;&#36;\chi^2=X_1^2+X_2^2+\cdots+X_n^2&#36;&#36;
Call it freedom.&#36;n&#36;I'm not sure I'm going to be able to do that.&#36;\chi^2\sim\chi^2(n).&#36;
We give the probability density function for the distribution of the card directly.
The probability density function is
&#36;US&#36;\left.f(x)=\left}begin{array&#125;&#125;frac}1}&amp;x&gt;0,\0,&amp;\text Other Organiser.&#36;&#36;
同时给出一些性质
卡方分布具有可加性（特征函数理论可证明）&#36;&#36;X\sim\chi^2(n 1), Y\sim\chi^2(n 2), \\text{and}X, Y\text{independent}\Rightarrow X+Y\thicksim\chi^2(n 1+n 2)&#36;&#36;
卡方分布的期望和方差为（中心矩，原点矩理论可证明）
&#36;&#36;E(\chi^2)=n,D(\chi^2)=2n.&#36;&#36;
定义 某点&#36;x&#36;是&#36;\chi^2\sim\chi^2(n).&#36; 的上&#36;\alpha&#36; 分位点 当且仅当
&#36;&#36;P{\chi^{2}&gt;\chi_{\alpha}^{2}(n)}=\alpha(0&lt;\alpha&lt;&#36;1
(very basic definition of location)</p>
<h5>T distribution</h5>
<p>Set&#36;X\sim N(0,1),Y\sim\chi^2(n)&#36; The two are independent of each other.
&#36;&#36;t=\frac X{\sqrt{Y/n&#125;&#125;&#36;&#36;
Satisfied freedom by&#36;n&#36;The T distribution is recorded as&#36;t(n)&#36;
Gives a probability density function for T-distribution
&#36;f(x)=\frac{\Gamma [(n+1)/2}sqrt{\pi}\Gamma(n)}left(+\frac{x^2}^right)^-\frac{n+2} (-\info)}&lt;x&lt;+infty
It's easy to see that the probability density function of the T-distribution is an even function, and the standard normal probability density is the T-distribution.&#36;n&#36; Approaching. &#36;\infty&#36; The limit of time</p>
<p>T-distribution as an even function
&#36;&#36;t_{1-\alpha/2}(n)=-t_{\alpha/2}(n)&#36;&#36;</p>
<h5>F distribution</h5>
<p>Set&#36;X\sim\chi^{2}(n_{1}),Y\sim\chi^{2}(n_{2})&#36; The two are independent of each other.
&#36;&#36;F=\frac{X/n_1}{Y/n_2}&#36;&#36;
Obey freedom.&#36;n_{1},n_{2}&#36;F distribution as recorded&#36;F(n_{1},n_{2})&#36;
Gives a probability density function for F distribution
&#36;f(x)=\begin{cases}\Gamma<a href="n_1/n_2">(n_1+n_2)/2</a>^{\frac{n_1}2}x^{\frac{n_1}2-1}\\hline\Gamma(n_1/2)\Gamma(n_2/2)[1+(n_1x/n_2)]^{\frac{n_1+n_2}2}\0,&amp;\text{Other}.\end{cases}&#36;&#36;
容易知道
F分布有关于交换构成部分分子和分母的性质
&#36;&#36;F\sim F(n_1,n_2)\Rightarrow\frac1F\sim F(n_2,n_1)&#36;&#36;
关于F分布的分位点有这样的性质
&#36;&#36;F_{1-\alpha}(n_1,n_2)=\frac1{F_\alpha(n_2,n_1)}&#36;&#36;</p>
<h5>About freedom.</h5>
<p>The original freedom of freedom was originally introduced in the Carabinieri distribution.
He expressed the number of stand-alone random variables contained in our statistics.
In computing, it's often the sample capacity minus the number of binding equations, and we continue to see many expressions of freedom in many places back there.</p>
<h4>Distribution of sample averages and differences in sample size</h4>
<h5>Theorem I</h5>
<p>Assumptions of average and variance overall
&#36;&#36;E(X)=\mu,D(X)=\sigma^2,&#36;&#36;
So whatever the distribution of the total, for the sample from this total,
There must be a sample average.
&#36;&#36;E(\overline{X})=\mu,D(\overline{X})=\frac{\sigma^{2&#125;&#125;{n}.&#36;&#36;
The expectations and differences of the sample (amendment) range can be addressed by the distribution given by Theorem II
Combined&#36;\chi^2&#36; Average and variance of distribution calculated</p>
<h5>Theorem II</h5>
<p>For a single normal aggregate &#36; \\mathrm{N)<del>\mu</del>,<del>\sigma</del>2)}&#36; 其样本均值&#36;\overline{X}&#36;和样本修正方差&#36;Satisfied with &#36;2</p>
<ul>
<li>&#36;\overline{X}\sim N(\mu,\frac{\sigma^{2&#125;&#125;{n}).&#36;</li>
<li>&#36;\frac{(n-1)S^2}{\sigma^2}\sim\chi^2(n-1);&#36; </li>
<li>If there are no amendments to&#36;\frac{1}{n}&#36;The difference in the coefficient is&#36;\frac{nS^2}{\sigma^2}\sim\chi^2(n-1);&#36; </li>
<li>If the difference is calculated using a real average instead of a sample average &#36;\chi^2&#36;The freedom of distribution becomes&#36;n&#36; ;fixing deviations does not involve the use of real averages, the correction is intended to resolve the problem of neutrality, and true averages do not involve this</li>
<li>&#36;\frac{\overline{X}-\mu}{S^{\color{red&#125;&#125;/\sqrt{n&#125;&#125;\sim t(n-1);&#36;</li>
<li>Sample average&#36;\overline{X}&#36;and sample correction variance&#36;S^2&#36;Independence</li>
</ul>
<h5>Theorem III</h5>
<p>For a double-normal sum of the same difference &#36; \\mathrm{N)<del>\mu_{1}</del>,<del>\sigma^2)}&#36;  &#36;\mathrm{N(</del>\mu_{2}<del>,</del>The average difference is
&#36;&#36;\frac{(\overline{X}-\overline{Y})-(\mu_1-\mu_2)}{S_w\cdot\sqrt{\frac1{n_1}+\frac1{n_2&#125;&#125;}\sim t(n_1+n_2-2)&#36;&#36;
of which
&#36;&#36;S_{w}^{2}=\frac{(n_{1}-1)S_{X}^{2}+(n_{2}-1)S_{Y}^{2&#125;&#125;{n_{1}+n_{2}-2}.&#36;&#36;
If not amended
&#36;&#36;S_{w}^{2}=\frac{(n_{1})S_{X}^{2}+(n_{2})S_{Y}^{2&#125;&#125;{n_{1}+n_{2}-2}.&#36;&#36;
Where's that? <del>\overline{Y}</del> ~S X^2}~S Y^2}&#36; for the average sample and difference
And there is.
&#36;&#36;\frac{S_X^2}{S_Y^2}\thicksim F(n_1-1,n_2-1)&#36;&#36;
If not amended
&#36;&#36;\frac{n_{1}(n_{2}-1)S_X^2}{n_2(n_1-1)S_Y^2}\thicksim F(n_1-1,n_2-1)&#36;&#36;
<em>When the difference is different, but both take the form of correction.</em>
&#36;&#36;\frac{\frac{S_X^2}{\sigma_{x}^2&#125;&#125;{\frac{S_Y^2}{\sigma_{y}^2&#125;&#125;\thicksim F(n_1-1,n_2-1)&#36;&#36;
The key formulas and the three distributions given in this section are very important, and then they are often used to deal with hypothetical tests.</p>
<p>Most of the inducts involved are fundamental deformations and the application of the theorem to the front.
Be careful to make a strict distinction between sample differences and sample correction differences.</p>
<h4>Examples</h4>
<p><a href="/en/blog/2024/09/24/probability-and-statistics-exercises-notes/">Examples of this section</a> Section entitled “Examples of mathematical statistical sampling distribution”</p>
<h3>Order statistics and their distribution</h3>
<h4>Order Count</h4>
<p>Assumptions&#36;X_{1}...X_{n}&#36;From the general distribution function&#36;F(x)&#36;If we sort these samples from bottom to big, we get a sequenced sample. &#36;X_{(1)}...X_{(n)}&#36;<br>We call it No.&#36;i&#36;Order counts from small to large&#36;i&#36;Amount &#36;X_{i}&#36;
Obviously. &#36;X_{1}&#36; Called Minimum Order Statistics &#36;X_{n}&#36; Called maximum order statistics</p>
<p>It's very clear that order statistics should be distributed separately, because sample numbers must be limited to a few points, and we can study the distribution of order statistics, but obviously, the distribution of order statistics should not be independent.
It's a very good order for the distribution that's separated.
Here's what we're going to do.</p>
<h4>Distribution of order statistics</h4>
<p>General&#36;X&#36;The density function is&#36;p(x)&#36; Distribution function is&#36;F(x)&#36;  &#36;X_{1}...X_{n}&#36;From the general distribution function&#36;F(x)&#36;And the sample is no.&#36;k&#36;Order statistics&#36;X_{k}&#36;Distribution is
&#36;&#36;p_{k}\left(x\right)=\frac{n!}{\left(k-1\right)!\left(n-k\right)!}\left(F\left(x\right)\right)^{k-1}\left(1-F\left(x\right)\right)^{n-k}p(x)&#36;&#36;</p>
<p>For the joint distribution of multiple order statistics, we directly give the binary formula:
&#36;00begin{aligned}p (y,z)&amp;=\frac{n!}{\left(i-1\right)!\left(j-i-1\right)!\left(n-j\right)!}\left[F(y)\right]^{i-1}\left[F(z)-F(y)\right]^{j-i-1}\\&amp;\cdot\left[1-F(z)\right]^n-j}p(y)p(z),\cdot\leq z\end{aligned} &#36;
We don't give formulas for the wider picture.</p>
<h3>Empirical Distribution Functions</h3>
<p>Assumptions&#36;x_{1}...x_{n}&#36;From the general distribution function&#36;F(x)&#36;If we sort these samples from bottom to big, we get a sequenced sample. &#36;x_{(1)}...x_{(n)}&#36; Define the following functions with an orderly sample
&#36;F n(x)=\begin{cases}0,&amp;x&lt;x_{(1)}\k/n,&amp;x_{(k)}\le x&lt;x_{(k+1)},\1,&amp;x (n)\le \end{cases}\quad k=1,2,... n-1
Obviously, this function meets all the conditions of a distribution function, which we call an experience distribution function.&#36;F_{n}(x)&#36;     He's an ordinary jumping function.
According to Bernouli's law. &#36;F_{n(x)}&#36;Condense to distribution function by probability&#36;F(x)&#36; </p>
<p>Empirical distribution functions are derived from samples, even if the same sample size varies.
For sure.&#36;x&#36; The empirical distribution function is a random variable. It's an event.&lt;Frequency of occurrence
We gave a theory about the empirical distribution function, and he explained that as long as the sample was large enough, the empirical distribution function was a good approximation of the distribution function.
Set&#36;x_1,x_2,\cdots,x_n&#36;is the total distribution function as&#36;F(x)&#36;The sample, &#36;F_n(x)&#36;for their experience distribution function, when&#36;n\to+\infty&#36;There is.
&#36;&#36;P{ \lim sup\mid F_n( x) - F( x) \mid = 0} =1&#36;&#36;</p>
<h3>Compilation and presentation of sample data</h3>
<p>References <a href="/en/blog/2023/11/05/descriptive-statistics-and-visualization-notes/">Descriptive statistics and visualization</a></p>
<h2>Parameter estimation</h2>
<h3>Parameter estimation issues</h3>
<h4>Definitions</h4>
<p>The basic problem with mathematical statistics is to extrapolate the overall distribution and some numerical characteristics of the distribution, based on the information provided by the sample. One of the types of the problem is that the overall distribution is known, while some of its parameters are unknown and, based on the samples obtained, these parameters are extrapolated and referred to as parameter estimates</p>
<h4>Estimates and estimates of unknown parameters</h4>
<p>Suppose we have a normal distribution overall.&#36;X&#36; Obey.&#36;N(\mu,\sigma^{2})&#36; Parameters are unknown
How do we estimate the unknown parameters?
Yeah.&#36;\mu&#36; You can use the average sample, median sample, etc.&#36;\sigma&#36; We can estimate the difference.
I can see that we need to construct a sample function. &#36;\hat{\theta}(X_1,X_2,\cdotp\cdotp\cdotp X_n)&#36; It's obviously a statistical amount that we call the statistical amount used to estimate the parameters.
This method of directly giving values for unknown parameters is called<strong>Point estimate</strong>
estimate&#36;(0,1)&#36; Overwrite&#36;\mu&#36;  The probability is 95%.
The method to give the value range of unknown parameters is called<strong>Estimates</strong></p>
<h4>Common estimation methods</h4>
<ul>
<li>Rectangular estimation method</li>
<li>It's a very similar estimate.</li>
<li>Minimal 2x2 estimation method</li>
<li>The Bayesian method.</li>
<li>Scale-neutral method</li>
<li>Maximum Minimum Estimated Method
The usual estimates are basically these.
In mathematical statistics, we're going to introduce rectangular estimations and very similar methods.</li>
</ul>
<h3>Rectangular estimate</h3>
<h4>Theory</h4>
<p>The idea of rectangular estimation is a simple alternative thought, proposed by the statistician Pearson.
Theoretically, the law of Sinchin.
&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;&#36;US&#36;&#36;&#36;&#36;&#36;&#36;US&#36;US&#36;&#36;US&#36;&#36;US&#36;&#36;US&#36;&#36;US&#36;US&#36;&lt;\varepsilon)=1.&#36;&#36;
样本矩依概率收敛于总体矩
由于 &#36;X_{i}^{k}&#36; 仍然保证了独立同分布 因此高阶原点矩也可以使用辛钦大数定律
&#36;&#36;\lim_{n\to\infty}P(\frac{1}{n}\sum_{i=1}^{n}X_{i}^{k}-E(X^{k})|&lt;\varepsilon) = &#36;1
This method of estimating the overall rectangular using the corresponding sample rectangular to determine the estimated value of the parameters to be determined is called<strong>Rectangular estimation method</strong>  </p>
<h4>Methodology</h4>
<p>General&#36;X&#36;Probability function&#36;f(x;\theta_1,\theta_2,...,\theta_l)&#36;Organisation&#36;l&#36;Unknown parameter&#36;\theta_{1}...\theta_{l}&#36;
&#36;X_1,X_2,\cdotp\cdotp,X_n&#36;It's total.&#36;X&#36; , and in general,&#36;l&#36;Rectangular E (X^k)\left()<em>{k=1,2,..,l}\right)&#36;存在，则它们应是这&#36;Functions for \iota&#36;:
&#36;E (X^k}=g</em>{k}(\theta_{1},\cdots,\theta_{l}),\quad k=1,2,\cdots,l&#36;&#36;
又 样本的&#36;k&#36; 阶原点矩为&#36;A_{k}=\frac{1}{n}\sum_{i=1}^{n}X_{i}^{k}&#36;
因此我们可以建立并解以下的方程来确定参数的矩估计值
&#36;&#36;\begin{cases}g 1(\theta 1,\cdots,\theta l)=\frac1n\sum=i,\g 2(theta 1,\cdots,\theta l)=\frac1\sum 1}i^,\cdots\cdots\g l(\theta 1,\cdots,\theta l)=\frac1n\sum i^l,\cdots\
It's very obvious that we should select the number of equations that we have to create, the number of the highest steps in the rectangular.</p>
<h4>Examples</h4>
<p><a href="/en/blog/2024/09/24/probability-and-statistics-exercises-notes/">Examples of this section</a> Section entitled “Examples of mathematical statistical rectangles”</p>
<p>The advantage of the rectangular approach is that it's simple, and it doesn't need to know what the distribution is. </p>
<p>The disadvantage is that when the overall type is known,<strong>Inadequate use of information provided by the distribution</strong> . </p>
<p>At the same time, in general, rectangular estimates are not unique, the main reason being that when establishing a rectangular equation, the general rectangles are selected with the corresponding sample rectangles instead of the belt. </p>
<h3>Very similar estimates.</h3>
<p>A very similar estimate (MLE) is a parameter estimation method used under conditions known for the overall distribution type.
It was originally proposed by mathematician Gauss, but the real success is due to statisticians Fisher.
It's a very simple idea.
Select the one with the greatest probability that the sample will appear in all parameter selections as its estimate. Value
It's a very intuitive idea.</p>
<h4>It's a very similar principle.</h4>
<p>In general, if the probability and parameters of event A occur&#36;\theta\in\Theta&#36;About,&#36;\theta&#36; The values vary, as do the P(A). the probability of event A occurring&#36;P(A|\theta).&#36;If one test, event A happens, it's considered at this time.&#36;\theta&#36;It's supposed to be a medium P.&#36;\theta&#36;To the biggest one. That's how it works.
His advantage is that the information given by the overall distribution when it's known is of better quality, but the difficulty of calculating it has clearly increased.
It's very apparent that the assumption is that there's an overall known distribution in the form of parameters, which is also a parameter method.</p>
<h4>Methodology</h4>
<h5>It's a great estimate of the total separation.</h5>
<p>If the total&#36;X&#36;It's discrete. &#36;P{X=x}=p(x;\theta)&#36; Form known but parameters unknown&#36;X_1,\cdots,X_n&#36;It's from&#36;X&#36;the sample; or&#36;X_1,\cdots,X_n&#36;Joint distribution laws
&#36;&#36;\prod_{i=1}^np(x_i;\theta)&#36;&#36;
Again.&#36;x_1,\cdots,x_n&#36;Yes.&#36;X_1,\cdots,X_n&#36;A sample value: an easy sample&#36;X_1,...,X_n&#36;Remove&#36;x_1,...,x_n&#36;the probability of an event&#36;&#123;X_1=x_1,\cdots,X_n=x_n}&#36;The probability is that:
&#36;&#36;L(\theta)=L(x_1,\cdots,x_n;\theta)=\prod_{i=1}^np(x_i;\theta),\theta\in\Theta.&#36;&#36;
of which&#36;L(\theta)&#36; We just have to find the right parameters to select the right ones under the determined sample values to make the apparent functions extremely significant as the estimate of the parameter, that is,
&#36;&#36;L(x_1,\cdots,x_n;\hat{\theta})=\max_{\theta\in\Theta}L(x_1,\cdots,x_n;\theta)&#36;&#36;
&#36;\hat{\theta}(x_1,\cdotp\cdotp,x_n)&#36; It's called a very similar estimate.
&#36;\hat{\theta}(X_1,\cdotp\cdotp\cdotp,X_n)&#36; It's called a huge estimate of parameters.</p>
<h5>It's a long line of estimates.</h5>
<p>Use exactly the same principle to get a function.
&#36;&#36;L(\theta)=L(x_{1},\cdots,x_{n};\theta)=\prod_{i=1}^{n}f(x_{i};\theta)&#36;&#36;
Or is it the construction that makes the apparition of function great?
&#36;&#36;L(x_1,\cdots,x_n;\hat{\theta})=\max_{\theta\in\Theta}L(x_1,\cdots,x_n;\theta)&#36;&#36;
Claims&#36;\hat{\theta}(x_1,\cdots,x_n)&#36;Yes&#36;\theta&#36;It's a huge estimate.
Claims&#36;\hat{\theta}(X_1,\cdots,X_n)&#36;Yes&#36;\theta&#36;It's a huge estimate.</p>
<h5>Larger methods</h5>
<p>If density functions&#36;f(x;\theta),p(x;\theta)&#36; About&#36;\theta&#36; Then we can use the analysis's differentials to do a great deal of research.
&#36;&#36;\frac{dL(\theta)}{d\theta}=0&#36;&#36;
Solution.&#36;\theta&#36; That's our final parameter.</p>
<p>Because of the apparent function&#36;L(\theta)&#36; It's the sum of multiple functions, so it looks like a logarithmic function.&#36;\ln(L(\theta))&#36;It'll simplify the guidance, simplify the operation, and the result will be the same, because...&#36;\ln(x)&#36; Single</p>
<p>This stems from the theorem:
If&#36;\hat\theta&#36;Unknown parameter&#36;\theta&#36;It's a huge estimate, and...&#36;g(\theta)&#36;Yes&#36;\theta&#36;One-to-Tune Functions&#36;&#123;g}(\hat\theta)&#36; Yeah.&#36;g(\theta)&#36;It's a huge estimate. </p>
<h4>Examples</h4>
<p><a href="/en/blog/2024/09/24/probability-and-statistics-exercises-notes/">Examples of this section</a> and the section on “Suspicions of numerical estimates”</p>
<h3>Medium estimate</h3>
<p>Let's try to estimate the parameters of a Cauchy distribution.&#36;\theta&#36; The density function is
&#36;&#36;f\left(x,\theta\right)=\frac{1}{\pi\left[1+\left(x-\theta\right)^{2}\right]}&#36;&#36;
We know that Cauchy's rectangles don't exist, so the rectangles don't work.
&#36;&#36;\sum_{i=1}^{n}\frac{X_{i}-\theta}{1+\left(X_{i}-\theta\right)^{2&#125;&#125;=0&#36;&#36;
This equation has a lot of roots and it's not easy to take root.
But there's a simpler way.</p>
<p>&#36;\theta&#36;It's the median of Cauchy's distribution.</p>
<h3>Guidelines for the excellence of estimates</h3>
<p>So when there are multiple estimates, which one is better, that's the question that we're going to study now, if we judge the good or the bad, if we get a better estimate.</p>
<h4>No bias</h4>
<h5>Non-selective introduction</h5>
<p>We know that the dot estimate is actually a random variable, so as a volatile estimate, the result of many of its fluctuations, if it's around the real value of the parameter, he might be considered a good estimate.
That's the system error estimated by the impartial study.</p>
<p>General&#36;X\sim F(x;\theta)(\theta\in\Theta)&#36;As parameter space.&#36;X_1,X_2,...,X_n&#36;Overall&#36;X&#36;The sample,&#36;\hat{\theta}=\hat{\theta}(X_1,X_2,...,X_n)&#36;Unknown parameter &#36;\theta&#36; I'm just trying to figure it out.
If the estimate &#36;\hat\theta&#36; The mathematical expectations exist and there is.
&#36;&#36;E_{\theta}(\hat{\theta})=\theta &#36;&#36;
Name&#36;\hat\theta&#36; Yes.&#36;\theta&#36; It's called bias.
Claims
&#36;&#36;b_{n}(\hat{\theta},\theta)=E_{\theta}(\hat{\theta})-\theta &#36;&#36;
For deviation of estimate </p>
<p>If &#36;b_{n}(\hat{\theta},\theta)\ne0&#36; Name&#36;\hat\theta&#36; Yes.&#36;\theta&#36; Estimated bias
If &#36;\pepratorname*lim}<em>{n\to\infty}b</em>{n}({\hat{\theta&#125;&#125;)=0&#36;   则称&#36;\hat\theta&#36; 是&#36;\theta&#36;.</p>
<p>No bias requires no system error, which is certainly good in theory, but in practical application, the value of neutrality also needs to be determined by specific events.</p>
<h5>Theorem</h5>
<p>No matter what.&#36;X&#36;Obey what distribution, if
&#36;&#36;
\mu\overset{\Delta}{\operatorname*{=&#125;&#125;E(X):,:\sigma^{2}\overset{\Delta}{\operatorname*{=&#125;&#125;D(X)
&#36;&#36;
Both exist, then.&#36;\hat{\mu}=\overline{X},\hat{\sigma}^2=S^2&#36;The difference is... &#36;\mu,\sigma^2&#36; The impartial estimate.</p>
<p>Here.&#36;S^2&#36; It's a sample variance. It's modified. &#36;\frac{1}{n-1}&#36;is the difference of the coefficient</p>
<p>We introduced it at the beginning of mathematical statistics.
&#36;&#36;E(\overline{X})=\mu,D(\overline{X})=\frac{\sigma^{2&#125;&#125;{n}.&#36;&#36;
The former is proof of the neutrality of the mean estimate.
&#36; \begin{aligned}
E\left (S{2}right)&amp; =\frac{1}{n-1}E\left[\sum_{i=1}^{n}\left(X_{i}-\overline{X}\right)^{2}\right]  \
&amp;=\frac{1}{n-1}E\left[\sum_{i=1}^{n}X_{i}^{2}-n\overline{X}^{2}\right] \
&amp;=\frac{1}{n-1}\left[\sum_{i=1}^{n}E\left(X_{i}^{2}\right)-nE\left(\overline{X}^{2}\right)\right] \
&amp; =\frac{1}{n-1}\left[\sum_{i=1}^{n}\left(\sigma^{2}+\mu^{2}\right)-n\left[\operatorname{Var}(\bar{X})+\left(E(\bar{X})\right)^{2}\right]\right]  \
&amp;=\frac{1}{n-1}\left[n\left(\sigma^{2}+\mu^{2}\right)-n\left[\frac{\sigma^{2&#125;&#125;{n}+\mu^{2}\right]\right] \
&amp;=\sigma^{2}.
\end{aligned}&#36;&#36;</p>
<p>Note for uncorrected sample variance (in&#36;\frac{1}{n}&#36; It's only gradual.
&#36;&#36;\begin{gathered}
{S_{n&#125;&#125;^{2}=\frac{1}{n}\sum_{i=1}^{n}(X_{i}-\bar{X})^{2}=\frac{n-1}{n}S_{n}^{*2} \
E({S_{n&#125;&#125;^{2})=\frac{n-1}{n}E(S_{n}^{*2}) \
=\frac{n-1}{n}\sigma^{2}\rightarrow\sigma^{2}~(n\rightarrow\infty)
\end{gathered}&#36;&#36;
Which means, both rectangular and MLE, that their estimates of the overall variance are based directly on the sample center rectangular approach, which is not neutral.</p>
<h5>Examples</h5>
<p><a href="/en/blog/2024/09/24/probability-and-statistics-exercises-notes/">Examples of this section</a> “Guidelines for the assessment of mathematical statistical estimates” Section</p>
<h4>Average error</h4>
<p>Assumptions used&#36;T(x)&#36;As Parameter&#36;q(\theta)&#36;The estimates, a natural criterion for evaluating the merits of the estimates, could be defined as follows:
&#36;&#36;
MSE_\theta(T)=R(\theta,T)=E(T(x)-q(\theta))^2
&#36;&#36;
Call the top the equation the average error and short the MSE</p>
<p>It's a very natural guideline for estimating the magnitude of errors, and it's common in a lot of statistical places as long as it involves point estimates.
Equivalent errors can normally be broken down in the following forms:
&#36;&#36;\begin{gathered}
MSE=E_\theta\left[(\hat{\theta}-\theta)^2\right]=E_\theta\left[\hat{\theta}^2+\theta^2-2\hat{\theta}\theta\right] \
=E_\theta\left[\hat{\theta}^2\right]-E_\theta\left[\hat{\theta}\right]^2+E_\theta\left[\hat{\theta}\right]^2+\theta^2-2\theta E_\theta\left[\hat{\theta}\right] \
=V_\theta\left[\hat{\theta}\right]+(\theta-E_\theta\left[\hat{\theta}\right])^2
\end{gathered}&#36;&#36;
If we remember the deviation of the bias
&#36;&#36;bias=E_\theta[\hat{\theta}]-\theta &#36;&#36;
There is.
&#36;&#36;MSE=V_\theta[\hat{\theta}]+bias^2&#36;&#36;
That is the sum of the squares of the average error or deviation estimated</p>
<p>You can see that if the estimate is neutral, the deviation should be 0 MSE equals the method of estimation.
In fact, the theory still contains equations of real values, and in actual calculations we need estimates instead of MSEs.</p>
<p>Define if for all&#36;\theta\in\Theta&#36; There's always.&#36;R(\theta,T)\leq R(\theta,S)&#36; Then we call it&#36;S&#36;It's an unacceptable estimate.&#36;T&#36;That's right.&#36;S&#36;Better. We don't usually choose unacceptable estimates.</p>
<p>Obviously, we would like to find an estimate with minimal MSE values for all parameters, but that's not possible; the best estimate of uniform error to a minimum does not exist.
We usually choose to make some reasonable demands on estimates and select good estimates in estimates that meet the reasonable requirements. </p>
<p><a href="/en/blog/2024/09/24/probability-and-statistics-exercises-notes/">Examples of this section</a> Section on "Equal Errors"</p>
<h4>Validity</h4>
<p>Definitions&#36;X_1,X_2,\cdot\cdot,X&#36;It's total.&#36;X\sim F(x,\theta);\theta\in\Theta&#36; The sample,
What's that?<em>{1}=\hat{\theta}</em>{1}(X_{1},X_{2},\cdots,X_{n}),:\hat{\theta}<em>{2}=\hat{\theta}</em>{2} (X , X 2, \cdotsX n}
Both.&#36;\theta&#36; {\cHFFFFFF}{\cH00FF00}<em>1)=E(\hat{\theta}<em>2)=\theta\quad&#36;.若&#36;\forall\theta\ta&#36;
&#36;
♪ And I'm so sorry ♪</em>{1})\leq D(\hat{\theta}</em>{2}
&#36;
Name&#36;\hat\theta_{1}&#36; That's right.&#36;\hat\theta_{2}&#36; More effective.
It can be seen that effectiveness is just a narrower MSE.
We introduced the concept of effective estimates in the C-R variant, and in fact the whole C-R variant and UMVUE were derived from the validity.</p>
<h4>Compatibility (consistency)</h4>
<h5>Definitions</h5>
<p>This is a good estimate of how the estimates will change as the sample capacity increases.
A very natural idea is that with the increase in sample capacity, estimates should be more precise.
Define. Set &#36;hat(theta}<em>n=\hat{\theta}(X_1,X_2,...,X_n)&#36;是未知参数 &#36;\theta&#36; 的点估计， 若 &#36;\forall\theta\in\Theta&#36; 满足： &#36;\forall\varepsilon&gt;0&#36; 有 &#36;\lim P{|\hat{\theta}</em>{\cHFFFFFF}{\cH00FF00}
Name &#36;\hat{\theta}_n&#36; Yes.&#36;\theta&#36; Combined estimates</p>
<h5>Theorem</h5>
<p>Whatever it is,<em>{X}&#36;服从什么分布，若 &#36;\mu\triangleq E(X), \sigma^2\triangleq D(X)&#36;
Both exist, then.&#36;\hat{\mu}=\overline{X},\hat{\sigma}^2=S_n^{*2}&#36;The difference is... &#36;\mu,\sigma^2&#36; Combined estimates
The core of research compatibility is the Sinchin Big Number Law, which means that the sample rectangles by probability.
It's based directly on Sinchin's law.
That's right.</em>{i=1}^{n}X_{i}\xrightarrow{P}\mu(n\rightarrow\infty)&#36;&#36;
研究方差&#36;&#36;\begin{gathered}
== sync, corrected by elderman ==
== sync, corrected by elderman ==
== sync, corrected by elderman ==
I'm sorry.
Obviously, the result is constricted by probability, because the difference is equal to the two-step square and one-step square. Bad</p>
<h5>Conclusions</h5>
<ul>
<li>Rectangular estimates are consistent estimates.</li>
<li>Very similar estimates are generally consistent.</li>
<li>Compatibility estimates may not be neutral, or even consistent.</li>
<li>If&#36;\hat\theta&#36; It's an impartial estimate.&#36;\lim_{n\to\infty}D(\hat{\theta})=0&#36;It's a confluence of estimates.<strong>It's a good conclusion.</strong>I'm not sure.
Based on the theory here in Chebishev, there are sufficient conditions to estimate:<strong>(progressive) unbiased estimates and close to zero.</strong>
Proof: Chebishev is different
&#36;&#36;P\left(\left|\xi- E(\varepsilon)\right|\geqslant\varepsilon\right)\leqslant\frac{D\left(\xi\right)}{\varepsilon^{2&#125;&#125;&#36;&#36;
If there is.&#36;\lim_{n\to\infty}D(\hat{\theta})=0&#36; And the difference left is close to zero, based on the pressure theorem.</li>
</ul>
<h4>Progressive normality</h4>
<p>A lot of complicated numbers in the&#36;n&#36;Close&#36;\infty&#36; At the same time, they're moving towards normal distribution.
Gradual normality is the extrapolation of the very restrictive rationale;
What statistics are gradual and normal, and how to judge is not the focus of our research here.
Gradual normality and compatibility are equally large samples.</p>
<h3>C-R heterogeneity</h3>
<p>One parameter tends to have multiple impartial estimates.
We naturally hope that the difference will be smaller, but whether or not the difference is lower, under what conditions.
The C-R heterogeneity explains this.
Proof that under certain conditions there is no bias in the estimates.&#36;\hat\theta&#36;There's a positive bottom line. </p>
<h4>Fisher Information Volume</h4>
<h5>Definitions</h5>
<p>For logarithmic functions&#36;ln(\xi;\theta)&#36;  Define the amount of Fisher's information as follows:
&#36;I(\theta)=E (\frac(partial\f(\xi;\theta)\partial\theta}^2}&gt;&#36;0.00
The expectation here is to see the parameters as defined values.&#36;X&#36; (samples) Seeking expectations, that's...&#36;E_{X|\theta}&#36;
Various properties indicate that the larger the Fisher information, the more information the sample can be considered to contain about unknown parameters.</p>
<h5>Examples</h5>
<p><a href="/en/blog/2024/09/24/probability-and-statistics-exercises-notes/">Examples of this section</a> "Mathematic statistics Fisher information volume case" section</p>
<h5>Conclusions</h5>
<p>If (generally most distribution meets)
&#36;&#36;\frac\partial{\partial\theta}\int\frac{\partial f(x;\theta)}{\partial\theta}dx=\int\frac{\partial^2f(x;\theta)}{\partial\theta^2}dx,&#36;&#36;
then
&#36;&#36;I(\theta)=-E[\frac{\partial^2\ln f(\xi;\theta)}{\partial\theta^2}]&#36;&#36;</p>
<h4>C-R heterogeneity</h4>
<p>Set&#36;\xi_1,\xi_2,\cdots,\xi_n&#36;To take from a probability function&#36;f(x;\theta),\theta\in\Theta&#36; &#36;={\theta:a&lt;\theta&lt;b}&#36;的母体的一个子样 ,其中&#36;a,b&#36;为已知常数， 且可设&#36;a=-\infty,b=+\infty.&#36; 又&#36;\eta=u(\xi_1,\xi_2,\cdots,\xi_n)&#36;是&#36;g (\theta) a neutral estimate and meets normal conditions
&#36;\text{collect}{x:f(x;\theta)&gt;\text {not \theta\text}&#36;&#36;
&#36;&#36;\begin{aligned}g^{\prime}(\theta)&amp;\text{and}Partial f(x; \theta)}Partial\theta}\text{and for everything}\theta\Theta,\fra\partial{\theta}&amp;\int f(x;\theta)dx=\int\frac{\partial f(x;\theta)}{\partial\theta}dx\end{aligned}&#36;&#36;
&#36;&#36;\begin{aligned}\frac\partial{\partial\theta}&amp;{\int\cdots\int u(x_1,x_2,\cdots,x_n)f(x_1;\theta)\cdots f(x_n;\theta)dx_1\cdots dx_n}\&amp;=\int\cdots\int u(x_1,x_2,\cdots,x_n)\frac\partial{\partial\theta}[\prod_{i=1}^nf(x_i;\theta)]dx_1\cdots dx_n\end{aligned}&#36;&#36;
则对于Fisher信息量
&#36;&#36;I(\theta)=E(\frac{\partial\ln f(\xi;\theta)}{\partial\theta})^{2}&gt;0&#36;&#36;
有
&#36;&#36;D_\theta\eta\geq{\frac{[g^{\prime}(\theta)]^2}{nI(\theta)&#125;&#125;&#36;&#36;
对于&#36;g(\theta)=\theta&#36; 的情形
&#36;&#36;D theta\eta\geq\frac1}&#36;&#36;
So, we gave an estimate of the lower CR range, which is also known as the information variable.</p>
<p>We define the estimated amount that meets the normal condition as a formal estimate.
It's easy to see that the lower level of the CR is the difference between a formal unbiased estimate. Bottom
For other estimates that are not formal or impartial, the difference cannot be given in the CR-I. Border</p>
<h4>Application of CR insularity</h4>
<p>Definitions&#36;\theta&#36;An impartial estimate&#36;\hat\theta&#36; Makes the CR indifferent.
&#36;&#36;D(\hat{\theta})=\frac1{nE[(\frac{\partial\ln f(\xi;\theta)}{\partial\theta})^2]}=\frac1{nI(\theta)}&#36;&#36;
Yes, it is referred to as a valid estimate (the amount of information here, regardless of multiple sampling, only one sample is taken)
Definition&#36;e=\frac{1}{nI(\theta)}/D(\hat\theta)&#36; It's called neutral efficiency. &#36;e=1&#36; It's called a valid estimate.
Definitions&#36;e\ne1&#36; If&#36;lim(e)=1&#36; It's called a progressive and effective estimate.</p>
<p><a href="/en/blog/2024/09/24/probability-and-statistics-exercises-notes/">Examples of this section</a> Section entitled “Mathematical statistics, CR-influences”</p>
<h3>Adequate statistics</h3>
<h4>Adequate statistics</h4>
<h5>Introduction</h5>
<p>The samples come from the general, and they contain the general information, but we often use the function of the tectonic sample -- statistical data -- to extrapolate statistically, how to extract the general information from the sample, whether or not we take the total information from the entire sample, which is the solution here.
<em>Here we usually consider only the information on the general distribution parameters contained in the sample, not the general distribution type.</em>
We use one example to introduce the concept of adequate statistics.
For example, in order to study the fatality rate of an athlete, we tested the athletes 10 times and found that the remaining eight were hit except for the third and sixth.
Now we're going to look into the parameters of the hit rate.
It's very obvious.&#36;T=x_{1}+x_{2}+...+x_{n}&#36; The amount of statistics constructed under this scenario will not be lost at all.&#36;\theta&#36; That's the idea that statisticians Fisher is proposing. <strong>Adequate statistics</strong>
With sufficient statistics, our statistical inferences on this parameter can be converted to statistics that no longer require sample data.</p>
<h5>Definitions</h5>
<p>Sample&#36;X_{1}...X_{n}&#36; There's a sample distribution.&#36;F_{\theta}(x)&#36; It contains all the parameters.&#36;\theta&#36; Statistics&#36;T&#36; There's a sample distribution. &#36;T_{\theta}(t)&#36; The full measure of nature means...&#36;T_{\theta}(t)&#36; All of it.&#36;F_{\theta}(x)&#36;About parameters&#36;\theta&#36; The information, the distribution of the samples.&#36;F_\theta(x|T=t)&#36; Other Organiser&#36;\theta&#36; Information
In that sense, we can give a full statistical definition.</p>
<p>Definitions&#36;X_1,X_2,...,X_n&#36;For Total&#36;X&#36; The sample,&#36;X&#36; The distribution function is&#36;F(x;\theta),\quad T{=}T(X_1,X_2,...,X_n)&#36;for a statistical amount, when T=t is given, if the sample&#36;X_1,X_2,...,X_n)&#36; The distribution of conditions (probability of conditions at the time of separation and density at the time of continuity) and parameters&#36;\theta&#36;It's not relevant, it's called&#36;T&#36;As Parameters&#36;\theta&#36;Adequate statistics </p>
<p>Definition of equivalence&#36;X_1,X_2,..,X_n&#36;For Total&#36;x&#36; The sample,&#36;X&#36; The probability function is&#36;f(x,\theta),\quad T{=}T(X_1,X_2,...,X_n)&#36;The probability function for a statistical amount&#36;g(t,\theta)&#36;If&#36;\frac{f(x_1,\theta)f(x_2,\theta)\text{L }f(x_n,\theta)}{g(T(x_1,x_2,\text{L },x_n),\theta)}=h(x_1,x_2,\text{L },x_n)&#36;Establishment;
And...&#36;t=T(x_1,x_2,\mathcal{L},x_n)&#36;When taking a fixed value,&#36;T=t&#36;Conditional probabilities under conditions of occurrence&#36;h(x_1,x_2,\mathcal{L},x_n)&#36;Not dependent&#36;\theta&#36;and &#36;T&#36; As Parameters&#36;\theta&#36; Adequate statistics</p>
<h5>Examples</h5>
<p><a href="/en/blog/2024/09/24/probability-and-statistics-exercises-notes/">Examples of this section</a> Section entitled “Question of adequate statistical data”</p>
<h4>Factorial Theorem</h4>
<p>By definition, it's cumbersome to determine whether a statistical measure is sufficient, so we've given the factor decomposition theorem, which can greatly simplify the search for a full statistical matter.</p>
<p>Factorial Theorem
&#36;T&#36; Yes.&#36;\theta&#36; The most important condition for a full measure of statistics is that the joint distribution of samples can be broken down into the following forms:&#36;h&#36; Non-negative and&#36;\theta&#36; Not relevant &#36;g&#36; Passes Only&#36;T&#36; Link to the sample.
&#36;&#36;L(\theta)=\prod_{i=1}^nf(x_i;\theta)=h(x_1,x_2,\cdots,x_n)g(T(x_1,x_2,\cdots,x_n);\theta)&#36;&#36;
It can be used very simply.
All we have to do is get the combined density function properly deformed and decompose.
The core is decomposition.</p>
<p>Theoretically, if&#36;T&#36; Yes.&#36;\theta&#36; A full statistical volume &#36;f(t)&#36; It's a single reversible function.&#36;f(T)&#36; Yeah.&#36;\theta&#36; Adequate statistics </p>
<h4>Full statistics</h4>
<p>First we introduce the concept of a full distribution function.
General&#36;X&#36;The distribution function family is&#36;&#123;F(x;\theta),\theta\in\Theta}&#36; For any satisfaction&#36;E_{\theta}[g(X)]=0&#36;♪ To everything ♪&#36;\theta\in\Theta&#36;Random variable&#36;g(X)&#36;Always.
&#36;&#36;
P_{\theta}{g(X)=0}=1,:\text{对一切}\theta\in\Theta,
&#36;&#36;
Name&#36;&#123;F(x;\theta),\theta\in\Theta}&#36;As Full Distribution Functions
Definitions&#36;(X_1,X_2,...,X_n)&#36;For Total&#36;F(x;\theta)(\theta\in\Theta)&#36;A sample, if measured&#36;T=T(X_1,X_2,...,X_n)&#36;Distribution Functions&#36;&#123;F_{\tau}(x;\theta),\theta\in\Theta}&#36; is a full distributed function family, which is&#36;T=T(X_1,X_2,\cdots,X_n)\text{}&#36; To complete statistics
You can see the following characteristics of the full statistical data.
&#36;US&#36;00begin{aligned}P {\theeta}\big{g}=g=(2}(T)\big}&amp;=1,\quad\forall\theta\in\Theta\\Leftrightarrow E_{\theta}\big[g_{1}(T)\big]&amp;=E_{\theta}\big[g_{2}(T)\big],\forall\theta\in\Theta\text{。}\end{aligned}&#36;&#36;
<a href="/en/blog/2024/09/24/probability-and-statistics-exercises-notes/">Examples of this section</a> Section entitled “Question of full and mathematical statistics”</p>
<h4>Indicator Distribution</h4>
<p>It's a sort of widely used distributed community.</p>
<h5>Single Parameter Index Distribution</h5>
<p>Definitions:
General &#36;X&#36; or &#36;X|\theta&#36; Distribution density &#36;p(x|\theta)&#36; Is:
&#36;&#36;
p(x|\theta)=g(x)h(\theta)\exp{t(x)\phi(\theta)}
&#36;&#36;
The functions of which are the normal known functions are referred to&#36;p(x|\theta)&#36; In the single parameter index distribution group
Examples
Study normal distribution&#36;N(\mu,\sigma^{2})&#36; When?&#36;\sigma^2&#36;When known
&#36; \begin{aligned}
p (x|m)&amp; =\left(2\pi\sigma^2\right)^{-\frac12}\exp\left{-\frac1{2\sigma^2}(x-\mu)^2\right}  \
&amp;== sync, corrected by elderman ==
I'm sorry.
A definition that meets a single parameter index distribution group
Similar, if the average is thought to be known, the difference is unknown, which is also part of the single-parameter index distribution.
It's similar to the porcelain distribution, two distributions, Gamma distributions, Beta distributions, all of which are index distribution groups.
In fact, the index is very wide-ranging.
But the very basic distribution of evenly distributed forms is not an index distribution group, because his definition of clusters is related to parameters that cannot be summarized as an index.
<strong>The definition of index distribution is only one of these forms, and it can actually be defined in many different ways, but they're all equal.</strong></p>
<h5>Biparameter index distribution</h5>
<p>Definitions
General &#36;X&#36; or &#36;X|\theta,\varphi&#36; The density of the distribution is &#36;p(x|\theta,\varphi),\theta,\varphi&#36; Unknown parameter if
&#36;&#36;
p(x|\theta,\varphi)=g(x)h(\theta,\varphi)\exp{t(x)\phi(\theta,\varphi)+u(x)\chi(\theta,\varphi)}
&#36;&#36;
Name &#36;X&#36; The distribution belongs to the two-parameter index distribution group</p>
<h5>A theorem.</h5>
<p>Set Random Variables&#36;x&#36; With a single parameter index distribution,&#36;X_1,X_2&#36;,L ,&#36;X_n&#36; It's from the general.&#36;x&#36;. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .&#36;\sum_{i=1}^nt(X_i)&#36;is the parameter&#36;\theta&#36;Adequate statistics
Here.&#36;t(X_i)&#36; It's the function inside when defining the index distribution group.
If&#36;\Theta^*&#36;As&#36;\mathbb{R}^k&#36;And at this point, the full amount of statistics is complete.
That is, for the index population, there's a very close line between full and complete statistics.
<strong>It's a very good theory to give a full statistical breakdown.</strong></p>
<h3>Unanimous Minimal Estimation</h3>
<p>The C-R-I-R-I-R-I-I-I told us.</p>
<ul>
<li>There's a difference in the amount of statistics.</li>
<li>In fact, not every parameter has a valid estimate, because not every impartial estimate can reach the lower C-R horizon.
So we asked two questions.</li>
<li>It's known that a neutral estimate can construct a new neutral estimate that is smaller than the original one.</li>
<li>An impartial estimate, even if it doesn't work, but his equation is minimal.</li>
</ul>
<h4>Rao-Blackwell Theorem</h4>
<h5>Rao-Blackwell Theorem</h5>
<p>Set&#36;X&#36;and&#36;Y&#36;It's two random variables.&gt;\boldsymbol{0}&#36;  定义&#36;&#36;\varphi(y)=E(X\mid Y=y)&#36;It's okay.
There is.&#36;E\varphi(Y)=\mu,\mathrm{Var}(\varphi(Y))\leq\mathrm{Var}(X)&#36;
It's a condition.&#36;&#123;:}X\text{和 }\phi(Y)\text{几乎处处相等}.&#36;
He offered a way to reduce the difference.
Prove it.
Set&#36;X,Y&#36; combined density is&#36;p(x,y)&#36; &#36;X&#36; The condition density is&#36;h(x|y)&#36; There is.
&#36;&#36;\varphi(y)=E\left(X\mid Y=y\right)=\int xh(x\mid y)dx=\int x\frac{p(x,y)}{p_{Y}(y)}dx&#36;&#36;
So you can give it to me.
&#36; \begin{aligned}E\phi(Y)&amp;=\int\varphi(y)p_Y(y)dy\&amp;=\int\int xp(x,y)dxdy=EX=\mu\end{aligned}&#36;&#36;
&#36;&#36;\begin{aligned}\operatorname{Var}(X)&amp;=E\left[(X-\varphi(Y))+(\varphi(Y)-\mu)\right]^2\&amp;=E\left(X-\varphi(Y)\right)^2+E\left(\varphi(Y)-\mu\right)^2\&amp;\color{}{+}2E[(X-\varphi(Y))(\varphi(Y)-\mu)]\end{aligned}&#36;&#36;
针对后半部分单独计算有
&#36;&#36;\begin{aligned}
&amp;E[(X-\varphi(Y))(\varphi(Y)-\mu)] \
&amp;=\int\int[x-\varphi(y)][\varphi(y)-\mu]p(x,y)dxdy \
&amp;=\int\int[x-\varphi(y)][\varphi(y)-\mu]p_Y(y)h(x\mid y)dxdy \
&amp;=\int[\phi(y)-\mu]{\int[x-\phi(y)]h(x|y)dx}p_Y(y)dy=0
\end{aligned}&#36;&#36;
因此
&#36;&#36;\begin{aligned}\operatorname{Var}(X)&amp;=E\left(X-\varphi(Y)\right)^2+\operatorname{Var}(\varphi(Y))\\operatorname{Var}(X)&amp;=\operatorname{Var}(\varphi(Y))\quad\Leftrightarrow P\left(X-\varphi(Y)=0\right)=1\end{aligned}&#36;&#36;</p>
<h5>Use of adequate statistics</h5>
<p>Set an overall probability density function as&#36;p(x;\theta),X_1,X_2&#36;, ,&#36;X_{n}&#36;  It's a sample.&#36;T=T(X_{1}...X_{n})&#36; Yes.&#36;\theta&#36;Adequate statistics &#36;S(X_{1}...X_{n})&#36; is the parameter&#36;g(\theta)&#36;An impartial estimate
&#36;&#36;\varphi(T)=E(S(X)|T(X))&#36;&#36;
Yes.&#36;g(\theta)&#36; An impartial estimate and
&#36;&#36;Var_{\theta}(\varphi(T))\leq Var_{\theta}(S(X)),&#36;&#36;
When and only when&#36;P(\varphi(T)=S(X))=1&#36; Time is set.
This theory tells us that when the expectations are reduced, using a good choice when fully measured, it tells us to choose the conditions.&#36;Y&#36;How
More generally:
If the unbiased estimate is not a function of sufficient statistical volumes, it is expected that the requirement for sufficient statistical volumes will yield a new neutral estimate that is smaller than the original estimate, thereby reducing the partial estimate. In other words, consider&#36;\theta&#36;The problem of estimating needs only to be performed in a function based on sufficient statistical data, and the statement is correct for all statistical inferences, which is called<strong>The principle of adequacy</strong></p>
<h5>An example</h5>
<p>Set&#36;X_{1}...X_{n}&#36;It's from&#36;b(1,p)&#36; the sample&#36;\overline{X}&#36; Yes.&#36;p&#36; Adequacy of statistics&#36;\theta=p^2&#36;Uneven estimates
We know that the sample average and the sample range are neutral estimates of the overall average and the range.
&#36;&#36;E\overline{X}=E(X)=p;ES^{*^2}=D(X)=p(1-p)&#36;&#36;
There's a difference between the two.&#36;p^2&#36; This is what we're building for.
It's very common to use similar techniques to construct impartial estimates in the later study of UMVUE.</p>
<h4>Minimal variance estimation</h4>
<p>Now, let's answer another question:</p>
<h5>UMVUE Definition</h5>
<p>For parameter estimates, set&#36;\hat{\theta}&#36;Yes.&#36;\theta&#36;A neutral estimate, if any&#36;\theta&#36;Uneven estimates&#36;\tilde{\theta}&#36;, in parameter space&#36;\Theta&#36;Both.
&#36;&#36;Var(\hat{\theta})\leq Var(\tilde{\theta})&#36;&#36;
It's called the Minimal Specimetric Estimates (UMVUE)</p>
<ul>
<li>If UMVUE exists, it must be a function of sufficient statistical weight.</li>
<li>If the variance reaches the lower C-R limit, it must be UMVUE.</li>
<li>The variance of the UMVUE does not necessarily reach the lower C-R boundary</li>
</ul>
<h5>UMVUE judgement</h5>
<p>Set&#36;X=(X_1,X_2,\cdots,X_n)&#36;It's a sample from a certain group.&#36;\hat{\theta}=\hat{\theta}(X)&#36;Yes.&#36;\theta&#36;An impartial estimate, Var. &lt; + \infty.&#36;如果对任意满足&#36;E(\phi(X))=0&#36; 的&#36;\phi(X)&#36; both
&#36;&#36;\color{}{\mathrm{Cov}_\theta(\widehat{\theta},\varphi)=0},\quad\forall\theta\in\Theta,&#36;&#36;
then&#36;\hat\theta&#36;Yes.&#36;\theta&#36;UMVUE</p>
<h5>Construct UMVUE</h5>
<p>Set&#36;T(X)&#36;It's a full-fledged count.&#36;S(X)&#36;Yes.&#36;g(\theta)&#36;, and&#36;\varphi(T)=E_{\theta}(S(X)|T(X))&#36;Yes.&#36;g(\theta)&#36;UMVUE<br>Further
If for all &#36;\theta\in\Theta&#36;,&#36;Var_{\theta}(\varphi(T))&lt;\infty&#36;, 则&#36;\varphi(T)&#36;是&#36;g(\theta)&#36;唯一的&#36;UMVUE&#36;
Theorem tells us that the UMVUE can be constructed using a full measure of statistics.
And UMVUE is the only one in probability.
In fact, this theorem tells us two ways to find UMVUE, but first we need to find a full measure of statistics.&#36;T(X)&#36;</p>
<ul>
<li>Statistical quantum function method: if&#36;\varphi(T(X))&#36;Yes.&#36;g(\theta)&#36;Quantified, then&#36;\varphi(T(X))&#36; Yeah.&#36;g(\theta)&#36;The UMVUE.&#36;g(\theta)&#36;The impartial estimate.</li>
<li>Expectations: If available&#36;g(\theta)&#36;An impartial estimate&#36;\varphi(X)&#36;,&#36;E(\varphi(X)|T(X))&#36;Yeah.&#36;g(\theta)&#36;UMVUE.</li>
</ul>
<h5>UMVUE and C-R lower bounds</h5>
<p>Some UMVUE can reach the lower C-R level, but others cannot.
Normally, the neutral estimate for reaching the lower C-R is UMVUE, provided the lower C-R is present.
You can't deny UMVUE without reaching the lower C-R level.
It's important that we have an impartial estimate of how to reach the lower C-R level.</p>
<h5>Examples</h5>
<p><a href="/en/blog/2024/09/24/probability-and-statistics-exercises-notes/">Examples of this section</a> "Mathematic Statistics UMVUE" section</p>
<h3>Estimates</h3>
<p>The section is expected to introduce some of the elements of the hypothetical tests and to apply some of the elements that will be discussed later in advance, although still within the parameters estimate.
We use average error to measure the deviation; but the concept of reliability is still lacking, the average error is much less accurate, and there are no fixed criteria; the inter-area estimate gives a range of parameters according to a certain level of reliability, which is what the entire section of the estimation is about.</p>
<h4>Basic concepts of spatial estimates</h4>
<h5>Definition of confidence interval</h5>
<p>General&#36;X&#36;Distribution Functions&#36;F(x;\theta)&#36;Contains an unknown parameter&#36;\theta&#36;, for given value &#36;\alpha\left (0)&lt;\alpha&lt;1\right)&#36;,若由样本&#36;X_1,X_2,\cdots&#36;, &#36;Two statistics determined by X n&#36;
&#36;\underline{\theta}=\underline{\theta}(X_1,X_2,\cdots,X_n)&#36;and&#36;\text{}\bar{\theta}=\overline{\theta}(X_1,X_2,\cdots,X_n)\text{ }&#36;Satisfied
&#36;
(X 1, X 2, \cdots, X n)&lt;\theta&lt;== sync, corrected by elderman == @elder man
&#36;
Name of random space (%2)&#36;\underline{\theta},\overline{\theta})&#36;Yes.&#36;\theta&#36;The confidence is 1-&#36;\alpha&#36;It's the confidence zone.&#36;\underline{\theta}&#36; and &#36;\overline{\theta}&#36;It's called confidence 1.&#36;-\alpha&#36; The two-sided confidence interval is called the upper limit and lower limit.</p>
<ul>
<li>The parameters to be determined are certain, but the unknown is random.</li>
<li>So we can't say there are parameters.&#36;1-\alpha&#36;We should say that there is random space.&#36;1-\alpha&#36;Probability includes parameters</li>
<li>If there's a lot of areas that are sampled over and over again, then the range that contains the parameters is about as much as&#36;1-\alpha&#36;It's the law of Bernoulie's great numbers.</li>
</ul>
<h5>Solution steps between confidence zones</h5>
<h6>One.</h6>
<p>Find a sample &#36;X_1,X_2,...,X_n&#36;Function:
&#36;&#36;
Z=Z(X_1,X_2,\cdotp\cdotp,X_n;\theta)
&#36;&#36;
 Include only parameters to be assessed &#36;\theta&#36;and &#36;Z&#36; The distribution is known and does not depend on any unknown parameters (including&#36;\theta&#36;I'm not sure.
Obviously, this doesn't fit the statistical definition. </p>
<h6>Two.</h6>
<p> For given confidence 1-&#36;\alpha&#36;set two constants&#36;a,b&#36;, make &#36;P{a)&lt;Z(X_1,X_2,\cdots,X_n;\theta)&lt;b}=1-\alpha.&#36;&#36;
 &#36;a,b&#36;It's all we need to do.</p>
<h6>III</h6>
<p>If you can get from &#36;a&lt;Z(X_1,X_2,\cdots,X_n;\theta)&lt;b&#36;得到等价的不等式 &#36;\underline{\theta}&lt;\theta&lt;\overline{\theta}&#36;, 其中 &#36;\theta=\theta(X_1,X_2,\cdots,X_n)&#36;, &#36;\overline{\theta}=\overline{\theta}(X_1,X_2,...,X_n)&#36;都是统计量，那么 &#36;It's fine.
Yes. &#36;\theta&#36;A confidence is 1- &#36;\alpha&#36; Other Organiser
It's a sort of equation.</p>
<h6>A few notes.</h6>
<ul>
<li>Different levels of confidence&#36;\alpha&#36; Parameters&#36;\theta&#36;The corresponding confidence interval is different.</li>
<li>The smaller the confidence interval, the more accurate the estimates, the lower the corresponding confidence level, and vice versa. Yeah.</li>
<li>If you want to reduce the confidence level, you have to increase the sample capacity.</li>
<li>The same confidence, the same confidence zone.</li>
<li>The value of the axis is usually derived from statistical distortion, which is a test of the level of thinking, so how to construct the axis is not the focus we need to have.</li>
</ul>
<h4>Estimates of total normal averages</h4>
<h5>The overall variance is known.</h5>
<p>If the total&#36;X&#36;Obey.&#36;N(\mu,\sigma^2)&#36; of which&#36;\sigma_{2}&#36;as known&#36;U&#36;Statistics
&#36;&#36;U=\frac{\bar{X}-\mu}{\sigma/\sqrt{n&#125;&#125;&#36;&#36;
Parameters&#36;\mu&#36;Make an estimation.
For given confidence level&#36;1-\alpha&#36; Yes.
&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;&#36;US&#36;US&#36;US&#36;US&#36;US&#36;&#36;US&#36;&#36;&#36;US&#36;US&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;...&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;...&#36;&#36;&#36;&#36;&#36;&#36;&#36;...&#36;&#36;&#36;&#36;&#36;&#36;...&lt;u_{1-\frac{\alpha}{2&#125;&#125;}=1-\alpha &#36;&#36;
<em>Why is that? We sometimes need theory.</em>
And there's a confidence zone.
&#36;&#36;\color{}{\left(\bar{X}-u_{\alpha/2}\frac\sigma{\sqrt{n&#125;&#125;,\bar{X}+u_{\alpha/2}\frac\sigma{\sqrt{n&#125;&#125;\right)}&#36;&#36;
<em>What happens between the confidence zones is a constant repetition of the problem.</em>
Understood.&#36;u&#36;The value means you know how to calculate.
Is this the only kind of confidence zone?
&#36;&#36;\left(\overline{X}-\frac\sigma{\sqrt{n&#125;&#125;u_{0.01},\overline{X}+\frac\sigma{\sqrt{n&#125;&#125;u_{0.04}\right)&#36;&#36;
The reason we don't do that is because this confidence zone is much longer, so it's not precise enough. High
In fact, there are countless options in the confidence zone, but we're only going to choose the one that's the shortest.</p>
<h5>The overall equation is unknown</h5>
<p>This is the time to choose.&#36;t&#36;Statistics are for testing. It's easy to know.
&#36;&#36;\frac{\overline{X}-\mu}{S^{\color{red&#125;&#125;/\sqrt{n&#125;&#125;\sim t(n-1);&#36;&#36;
So give confidence as&#36;1-\alpha&#36;Other Organiser
&#36;US&#36; \\left(x}-\frac{S n^)<em>}{\sqrt{n&#125;&#125;\cdot t_{\alpha/2}(n-1),\quad\bar{X}+\frac{S_n^</em>\cdottt (n-1)\right)
Understood.&#36;t&#36;The value means how to calculate.</p>
<h4>Estimates of the difference between the two normal overall mean values</h4>
<h5>Both differences are known.</h5>
<p>&#36;&#36;\overline{X}\sim N(\mu_1,\frac{\sigma_1^2}n),\quad\overline{Y}\sim N(\mu_2,\frac{\sigma_2^2}m)&#36;&#36;
It's easy to construct a core number.
&#36;&#36;\frac{(\bar{X}-\bar{Y})-(\mu_1-\mu_2)}{\sqrt{\frac{\sigma_1^2}n+\frac{\sigma_2^2}m&#125;&#125;\sim N(0,1)&#36;&#36;
So you can get it.&#36;\mu_1-\mu_2&#36;There's a confidence zone.
&#36;&#36;\left((\overline{X}-\overline{Y})-u_{1+\frac a2}\sqrt{\frac{\sigma_1^2}n+\frac{\sigma_2^2}m},\quad(\overline{X}-\overline{Y})+u_{1+\frac a2}\sqrt{\frac{\sigma_1^2}n+\frac{\sigma_2^2}m}\right)&#36;&#36;</p>
<h5>The difference is unknown, but the difference is equal.</h5>
<p>&#36;&#36;\overline{X}\sim N(\mu_1,\frac{\sigma^2}n),\quad\overline{Y}\sim N(\mu_2,\frac{\sigma^2}m)&#36;&#36;
Give the pivotal amount
&#36;&#36;\frac{(\overline{X}-\overline{Y})-(\mu_1-\mu_2)}{\sqrt{\frac1n+\frac1m}\sqrt{\frac{(n-1)S_1^2+(m-1)S_2^2}{n+m-2&#125;&#125;}\sim t(n+m-2)&#36;&#36;
Theoretical reasoning.
&#36;P\left(\left|left.\frac{(overline{X}----overline{Y&#125;&#125; }{\sqrt{\rac1n+\frac1m}\sqrt{(n-1)S 1^2+(m-1)S 2^n+m2&#125;&#125;&#125;&#125;right|&lt;t_{1-\frac\alpha2}\right)=1-\alpha\right. &#36;&#36;
给出置信区间
&#36;&#36;\left((\overline{X}-\overline{Y})\pm t_{1+\frac\alpha2}\sqrt{\frac1n+\frac1m}\sqrt{\frac{(n-1)S_1^2+(m-1)S_2^2}{n+m-2&#125;&#125;\right)&#36;&#36;</p>
<h5>The difference is unknown, but the sample is large enough.</h5>
<p>When the sample is large enough (which is generally considered to be more than 50) we can replace the difference in the sample correction with the difference in the actual difference and then return to the situation where the difference is known.
&#36;&#36;\left((\overline{X}-\overline{Y})\pm u_{1-\frac\alpha2}\sqrt{\frac{S_1^2}n+\frac{S_2^2}m}\right)&#36;&#36;</p>
<h5>The difference is unknown, but the sample is equal.</h5>
<p>&#36;&#36;\overline{X}\sim N(\mu_1,\frac{\sigma_1^2}n),\quad\overline{Y}\sim N(\mu_2,\frac{\sigma_2^2}m)&#36;&#36;
And there is.
&#36;&#36;n=m&#36;&#36;
You!&#36;Z_{i}=X_{i}-Y_{i}&#36;
We can think of the sample now.&#36;Z_{i}&#36;It all comes from
&#36;&#36;Z\sim N(\mu_{1}-\mu_{2},\sigma_{1}^{2}+\sigma_{2}^{2})&#36;&#36;
We can assume that at this point we are in the process of estimating the mean of the normal distribution of the unknown difference in the sample.
Select&#36;t&#36;Statistics
&#36;&#36;\frac{\overline{Z}-\mu}{S^{\color{red&#125;&#125;/\sqrt{n&#125;&#125;\sim t(n-1);&#36;&#36;
The formula before transformation is
&#36;US&#36;\left(overline{Z}-\frac{S z^)<em>}{\sqrt{n&#125;&#125;\cdot t_{\alpha/2}(n-1),\quad\overline{Z}+\frac{S_
z^</em>}{\sqrt{n&#125;&#125;\cdot t_{\alpha/2}(n-1)\right)}&#36;&#36;
换入我们需要的量得到估计区间
&#36;&#36;\left((\overline{X}-\overline{Y})\pm t_{1+\frac\alpha2}(n-1)\frac{S_Z}{\sqrt{n&#125;&#125;\right)&#36;&#36;</p>
<h4>Estimated range of normal aggregate differences</h4>
<p>Overall&#36;X&#36;Obey.&#36;N(\mu,\sigma^2)&#36;
We just need to introduce.&#36;\mu&#36;Unknown
<em>For what is already known about the average, a similar form is given in the section on statistics that can be used to construct the axis, and freedom and coefficients change.</em>
One of the caloric distributions that we've described earlier is a pivotal amount.
&#36;&#36;\frac{(n-1)S^2}{\sigma^2}\sim\chi^2(n-1)&#36;&#36;
 So there is.
&#36;P\left{\chi {\alpha/2)^2(n-1)&lt;\frac{(n-1)S^2}{\sigma^2}&lt;\chi_{1-\alpha/2}^2(n-1)\right}=1-\alpha &#36;&#36;
 计算得到置信区间为
 &#36;&#36;\left(\frac{(n-1)S^2}{\chi_{1-\alpha/2}^2(n-1)},\frac{(n-1)S^2}{\chi_{\alpha/2}^2(n-1)}\right)&#36;&#36;
 开方就可以得到标准差的置信区间
 &#36;&#36;\loft(\1-\alpha/2^2(n-1)},\frac{sqrt{1}(n-1)&#125;&#125;
This is the first asymmetric axis we've presented.&#36;F&#36;So is the core.
We still picked a symmetrical point on both sides to determine the confidence interval.
It's just a matter of choice.</p>
<h4>Estimates of the difference between the two normal aggregates</h4>
<p>Let's just talk about the fact that the overall average is unknown.
<em>Although we can still construct new cores based on the known variance estimates of the average above, it is easy to guess that freedom increases when the average is known.</em>
Or do you think you're going to give us a pivotal point based on the conclusions we've presented in the statistics section? Volume
&#36;&#36;\frac{\frac{S_X^2}{\sigma_{x}^2&#125;&#125;{\frac{S_Y^2}{\sigma_{y}^2&#125;&#125;\thicksim F(n_1-1,n_2-1)&#36;&#36;
So, give us an inequity.
&#36;P\left{F  {\alpha/2} (n 1-1, n 2-1)&lt;\frac{S_1{}^2/{\sigma_1}^2}{S_2{}^2/{\sigma_2}^2}&lt;F_{1-\alpha/2}(n_1-1,n_2-1)\right}=1-\alpha &#36;&#36;
置信区间为
&#36;&#36;\color{}{\left(\frac{S_1^2}{S_2^2}\frac1{F_{1-\alpha/2}(n_1-1,n_2-1)},\frac{S_1^2}{S_2^2}\frac1{F_{\alpha/2}(n_1-1,n_2-1)}\right)}.&#36;&#36;</p>
<h4>Single-side confidence interval</h4>
<p>At this point, we're sure that the core of the confidence zone will be changed to
&#36; \begin{aligned}P&lt;\theta)&amp;=1-\alpha\quad (\text{or}P(\theta)&lt;\overline(theta} =1-\alpha)\end{aligned}
It's understandable.
At this point,&#36;\underline{\theta}&#36; It's called a one-sided lower limit. &#36;\overline{\theta}&#36;It's called a one-sided confidence cap.
The manner in which the core amount is constructed will not change the one-sided confidence zone, but the determination of the difference.
The single-side confidence zone still meets our definition of confidence.
He actually has the same value as the double confidence zone.</p>
<h4>Proportional confidence interval (large sample capacity)</h4>
<p>As a whole&#36;X&#36;The distribution is unknown, but the sample is very large.
&#36;&#36;\overline{X}\sim N(\mu,\frac{\sigma^{2&#125;&#125;{n})&#36;&#36;
Actually, we're back to the whole problem of normality.
Use estimate&#36;\overline{X}&#36;To construct the core amount required to finalize the estimated range of parameters</p>
<p>Give an example of a ratio (often attributed to two-point distribution parameters)&#36;p&#36;It's a very common problem in statistics.</p>
<p>Set Distribution&#36;X&#36;Obedience parameter is&#36;p&#36;The two-point distribution of the sample is&#36;X_{1},...,X_{n}&#36; &#36;n&gt;50&#36;  求参数&#36;p&#36;置信度为&#36;1-\alpha-dollar confidence interval</p>
<p>It's not a problem with a normal distribution, but it's easy to know that we can be assisted in our research with very limited logic.
&#36;&#36;\overline{X}\sim N(p,\frac{p(1-p)}{n})&#36;&#36;
<em>Attention, we're bringing in two-point distribution expectations and differences, not two distributions.</em>
Now the question is, is there an unknown variance in the range estimate of the normal distribution mean?</p>
<p>The difference is unknown, but he only contains the parameters we want to estimate.&#36;U&#36;Statistics as pivotal
&#36;P\left.&lt;\frac{\sqrt{n}(\overline{X}-p)}{\sqrt{p(1-p)&#125;&#125;&lt;u_{1-\frac\alpha2})\approx1-\alpha &#36;&#36;
得到关于&#36;p&#36;的方程（和前面的思路还是有点区别的）
&#36;&#36;\begin{aligned}0\leq\frac{n(\overline{X}-p)^2}{p(1-p)}&lt;u_{1-\frac{\alpha}2}^2\end{aligned}&#36;&#36;
化简
&#36;&#36;(n+u_{1-\frac\alpha2}^2)p^2-(2n\overline{X}+u_{1-\frac\alpha2}^2)p+n\overline{X}^2&lt;&#36;0.00
The equation can be estimated at the desired range.</p>
<h2>Assumptions test</h2>
<h3>Introduction of assumptions and assumptions</h3>
<p>When we don't know anything about parameters, we usually use the parameter estimation method in the previous chapter.
But when the parameter estimates are completed, we have a basic understanding of the parameters, and we want to know if our estimates are correct, and that's what this chapter's hypothetical test is about. </p>
<h4>What's hypothetical?</h4>
<p>Presentation of specific values for overall parameters
For example, the overall average is greater than a certain number, the overall equation is less than a certain amount, and so on.</p>
<h4>What's a hypothetical test?</h4>
<p>Some assumption of the overall parameter (or distribution form) is first made, and then the determination of whether or not the hypothesis is established is made using sample information</p>
<p>There are two types of tests of parameters and non-parametric tests, the difference between non-parametric and non-parametric statistics.</p>
<p><strong>Logical application of counter-evidence, statistically based on the principle of small probability.</strong></p>
<h4>Original and alternative assumptions</h4>
<p>null hypothesis     alternative hypothesis</p>
<p><strong>The assumption is that we're gathering evidence that we want to object to.</strong></p>
<p>In the hypothetical tests behind us,
Thinks it's usually an equation with an equal sign like:&#36;\mu=10,\mu\ge10&#36; </p>
<p>The alternative scenario is the one we want to support.
In the hypothetical tests behind us,
An equation that does not normally contain equivalents is as follows:&gt;0&#36; </p>
<p>The original and alternative assumptions must be opposed to each other. Group</p>
<h3>A hypothetical test example</h3>
<p>Let's start with an example of the process of hypothetical testing.</p>
<p>Snails produced at a plant, with a standard strength of 68, and actual production strength&#36;X&#36; Yeah.&#36;N(m,3.6^2 )&#36;If&#36;E(X)=\mu=68&#36;If the average values are: 69.5 and 67.5 respectively, is the same?</p>
<p>We make two assumptions. &#36;\mu=68&#36; Alternative assumptions &#36;\mu\ne68&#36; </p>
<p>Now we're going to select one of the two scenarios for a hypothetical test. &#36;\mu=68&#36; Correct (select original assumption)</p>
<p>Then there is.
&#36;&#36;\overline{X}\sim N(68,3.6^2/36)&#36;&#36;
Now we're building a contradiction between a small probability event, and...
It's based on a small probability.&#36;\alpha=0.05&#36;
&#36;&#36;P\left(\left|\frac{\overline{X}-68}{3.6/6}\right|&gt;== sync, corrected by elderman ==
It's possible to decipher the odds and get the acceptance and rejection. </p>
<h3>Additional explanation of some terms</h3>
<h4>Two types of errors</h4>
<p>It's very clear that our test results are based on the fact that it's completely incorruptible.</p>
<ul>
<li>The assumption is true, but the sample was rejected.</li>
<li>The original hypothesis was false, but it was accepted for sampling reasons.</li>
</ul>
<p>We usually write down the probability of a first-class error.&#36;\alpha&#36; The probability of a second type of error is...&#36;\beta&#36;</p>
<h4>Small probability principle</h4>
<p>In one trial, the probability of an almost impossible event is called a small probability.
Once a small and medium probability event occurs, we have reason to reject the assumption.
The small probability is what we decide.</p>
<h4>High profile level</h4>
<p>That's the small probability we identified as a small probability.&#36;\alpha&#36;
That's what we decided in advance. <del>0.05</del> ~0.1 million
And that's why we're using the symbol here because it's the same value.</p>
<h4>Test statistics</h4>
<p>Test statistics are based on sample observations.
The sample statistics used to make decisions about the original and alternative assumptions are also a statistical amount, but we use it for a specific purpose.
He's a hypothetical.&#36;H_0&#36;For real, a certain amount of statistical data that is known to be constructed.
Core questions on testing classical statistical assumptions when selecting the appropriate statistical volume</p>
<h4>Deny Field</h4>
<p>Denied domain is a collection of all possible values that can be taken to test the statistical amount as originally assumed, and the boundary of rejected domain is called the threshold</p>
<h3>The probability of two types of error</h3>
<h4>Qualitative analysis</h4>
<p>Type two error probability </p>
<ul>
<li>Increases as the overall parameters assumed decrease</li>
<li>Increase with first-class errors</li>
<li>Increase with overall standard deviation</li>
<li>Increase with reduced sample capacity</li>
</ul>
<p>We need to know. </p>
<p><strong>It's impossible to reduce the probability of two types of error at the same time, at the established sub-sample capacity.</strong>
<strong>By increasing the size of the subs, you can reduce the second type of error.</strong></p>
<p>The calculation below can be explained.</p>
<h4>Calculating the probability of two errors</h4>
<p>Let's just say one example.
Example:&#36;X_1,X_2...X_{n}\sim N(\mu,\sigma^2)&#36; of which&#36;\sigma^2&#36; Known
&#36;H 0: \mu=h 1: \mu&gt;&#36;0.00
Deny Domain As &#36;\bar{x}\ge c_{0}&#36;  </p>
<ul>
<li>The probability of two types of error. </li>
<li>Yes.&#36;\mu_0=0.5,\sigma=0.2,\alpha=0.05,n=9,\mu=0.65&#36;Time-based calculations of the probability of not making category two errors</li>
</ul>
<p>We just need to start with a definitional analysis, and use tests to make some complementary simplicity.</p>
<p>The first type of error is a waiver of the truth, which is the original hypothesis, but a refusal.
The second type of error is hypocrisy, which means the original hypothesis is not valid, but acceptance.
&#36;&#36;00\
&amp;\alpha=P\left(\c 0H 0真right)\
&amp;=P_{\mu_{0&#125;&#125; \left(\overline{x}\geq c_{0}\right) \
&amp;=P\left(\frac{\overline{x}-\mu_{0&#125;&#125;{\sqrt{\frac{\sigma^2}{n&#125;&#125;}\geq\frac{c_0-\mu_{0&#125;&#125;{\sqrt{\frac{\sigma^{2&#125;&#125;{n&#125;&#125;}\right)=1-\phi\left(\frac{c_{0}-\mu_{0&#125;&#125;{\sqrt{\frac{\sigma^{2&#125;&#125;{n&#125;&#125;}\right)
\end{aligned}&#36;&#36;
&#36;&#36;\begin{aligned}
&amp;\beta=P\left(\c 0HH 1真right)\
&amp;=P_{\mu} \left(\overline{x}\leq c_{0}\right) \
&amp;=P\left(\frac{\overline{x}-\mu}{\sqrt{\frac{\sigma^2}{n&#125;&#125;}\leq\frac{c_0-\mu}{\sqrt{\frac{\sigma^{2&#125;&#125;{n&#125;&#125;}\right)=1-\phi\left(\frac{c_{0}-\mu}{\sqrt{\frac{\sigma^{2&#125;&#125;{n&#125;&#125;}\right)
\end{aligned}&#36;&#36;</p>
<p><strong>From this point on, one pattern of one type of error and two type of error is shown; therefore, the probability of one type of error cannot be reduced indefinitely, corresponding to the probability of another type of error explodes.</strong></p>
<p>We're using the basic definition to calculate the probability of two errors.
And then, by testing the amount of statistics to simplify the distribution of some standard, you get the probability of two types of error.</p>
<p><strong>I can see the first and second types of errors.&#36;c_0&#36;It's completely unknown, because the rejection field is determined by the probability of choosing the first type of error, and we control the probability of the first type of error first in the actual hypothetical test.</strong></p>
<p><strong>We don't need to calculate the specific values of two types of errors, for the first type, which is actually an equation, and the probability of the first type of errors is artificially assigned; for the second category of errors, the probability of the first type is assigned.&#36;\alpha&#36;Send&#36;c_0&#36;Then we'll take over.&#36;\beta&#36;expression that you can get the result of</strong></p>
<p><strong>If we give the rejection field directly (or calculate the result in the question before), then the probability of a category I-II error (re-inferred by definition) can be calculated, which is often the case when the probability of a type I error is properly controlled.</strong></p>
<h3>Assumed testing of the overall average</h3>
<h4>A hypothetical test for the known difference average value extraction</h4>
<p>Give two hypotheses.
&#36;&#36;H_0\colon\mu=\mu_0;\quad H_1\colon\mu\neq\mu_0&#36;&#36;
Construct the number of tests
&#36;&#36;U=\frac{\overline{X}-\mu_0}{\sigma/\sqrt{n&#125;&#125;\sim N(0,1)&#36;&#36;
When you accept the original hypothesis, test the statistics.&#36;\overline{X}&#36; The situation is known.&#36;N(0,1)&#36; </p>
<p>The area of rejection, the area where the small probability of an event occurs.
&#36;&#36;P_{H_{0&#125;&#125;(\left|\frac{\overline{X}-\mu_{0&#125;&#125;{\sigma/\sqrt{n&#125;&#125;\right|\geq u_{\frac{\alpha}{2&#125;&#125;)=\alpha &#36;&#36;
You can get statistics by deforming.&#36;\overline{X}&#36;Other Organiser</p>
<p>Called&#36;U&#36;Tests, because the numbers are...&#36;U&#36;Statistics</p>
<h4>Hypothetical test for the difference unknown</h4>
<p>Construct the number of tests
&#36;&#36;T=\frac{\overline{X}-\mu_{0&#125;&#125;{S^{*}/\sqrt{n&#125;&#125;  \sim t(n-1)&#36;&#36;
It's easy to give a rejection field based on the principle of small probability.
Just check if this is in the rejected field.
Called&#36;t&#36;Test</p>
<h4>Assumed test of equal dual-normal matrix averages</h4>
<h5>When the difference is unknown but equal</h5>
<p>Test the number of statistics to be
&#36;&#36;\frac{\overline{X}-\overline{Y&#125;&#125;{\sqrt{\frac1n+\frac1m}\sqrt{\frac{(n-1)S_1^{*2}+(m-1)S_2^{*2&#125;&#125;{n+m-2&#125;&#125;}\sim t(n+m-2)&#36;&#36;</p>
<h5>In the case of Big Son</h5>
<p>According to the central limits, both values are subject to normal distribution, so they can be constructed.&#36;U&#36;Statistics
&#36;&#36;U=\frac{\bar{X}-\bar{Y&#125;&#125;{\sqrt{\frac{S_1^{*2&#125;&#125;n+\frac{S_2^{*2&#125;&#125;m&#125;&#125; \sim N(0,1)&#36;&#36;</p>
<h3>Assumptions of overall variance</h3>
<h4>Hypothetical test of the squared-out value if the average is known</h4>
<p>Test statistics
&#36;&#36;\chi^2=\frac{\sum_{i=1}^n(X_i-\mu)^2}{\sigma_0^2}\sim\chi^2(n)&#36;&#36;
Here's the difference.&#36;n&#36;And the auxiliary construction.&#36;n&#36;I've got a split.</p>
<h4>Hypothetical test for squared-out values in the event of unknown averages</h4>
<p>Test statistics
&#36;&#36;\chi^{2}=\frac{(n-1)S^{*2&#125;&#125;{\sigma_{0}^{2&#125;&#125;\sim\chi^{2}(n-1)&#36;&#36;</p>
<h4>In the case of large samples</h4>
<p>For the tests we've been able to construct,&#36;\chi^2(n)&#36;In the case
When the sample is large enough, it's very limited by the center.
&#36;&#36;\frac{\chi^2-n}{\sqrt{2n&#125;&#125;\overset{\text{&#125;&#125;{ \operatorname* { \sim &#125;&#125;N(0,1)&#36;&#36;
So, the test statistics that are calculated above are brought here to get new tests.
All the calorie tests can be converted to large samples.&#36;U&#36;Test</p>
<h4>Assuming tests for a two-normal parent-rate difference</h4>
<p>Test statistics
&#36;&#36;F=\frac{S_1^{*2}/\sigma_1^2}{S_2^{*2}/\sigma_2^2}=\frac{S_1^{*2&#125;&#125;{S_2^{*2&#125;&#125;\sim F(n-1,m-1)&#36;&#36;
Use this to test whether the difference is equal and then return to check the balance;
It can be understood that the difference is an essential part of the examination of the mean equivalent.</p>
<h3>Match data&#36;t&#36;Test</h3>
<p>Match data&#36;t&#36;Testing is one of the most widely applied hypothetical tests in statistics.</p>
<h4>How to match</h4>
<p>We often want to get two subjects close to the test.
The aim is to avoid the influence of factors other than experimental treatment.</p>
<h4>Three scenarios with design information</h4>
<ul>
<li>The same pairing of pairs is given two different treatments to the subject subject, the aim being to infer whether the effects of the two treatments differ.</li>
<li>The same subject was treated differently, with the aim of insulating that the effects of the two were different</li>
<li>Comparison of treatment before and after the same subject is tested, the purpose is to infer whether a certain treatment is effective
<strong>You can see, it's a pair.&#36;t&#36;The tests should be very extensive for processing of experimental data.</strong></li>
</ul>
<h4>&#36;t&#36;Test</h4>
<p>Designed with pairs&#36;t&#36;The test was to examine the difference in the average between the two data sets.
If we're going to make a difference between two sets of data, then we're going to be a whole.
What we're looking at is a hypothetical test of whether the average of the margin values in this group is zero.
&#36;&#36;t=\frac{\overline{d}-\mu_{d&#125;&#125;{s_{\overline{d&#125;&#125;}=\frac{\overline{d}-0}{s_{d}/\sqrt{n&#125;&#125;=\frac{\overline{d&#125;&#125;{s_{d}/\sqrt{n&#125;&#125;\quad\sim&#123;&#123;t(n-1)&#125;&#125;&#36;&#36;
It's a hypothetical test of the unknown difference in the overall average.
The specific example is not to describe here the idea of understanding this test is enough.</p>
<h3>Single-side hypothetical tests</h3>
<h4>Double and single-sided hypothetical tests</h4>
<p>The alternative is no directional.&#36;\ne&#36; It's called a double-sided test or a double-tail test.
The alternative scenario is specific in orientation and contains a symbol&gt;~&lt;&#36; Called a single-tailed test or a single-tail test</p>
<p>Use symbol&lt;&#36; 的称为左侧检验 使用&#36;&gt;Called right test</p>
<h4>One-sided testing is achieved</h4>
<p>The one-sided hypothesis test is very natural, and we need to convert the rejection of the domain to the same level of statistical data.</p>
<p>Whether a single-sided test is used is required to see whether the need for such use exists for the specific subject of the hypothetical test;</p>
<p>Requires that the standard deviation of resistance of a certain conductor should not exceed 0.2.
Obviously, we should have a default scenario of &#36;H 1:\sigma^2&gt;0.2 GH2
Our aim is to deny the bias, so we should take it as a hypothetical.</p>
<p>The standard weight of each bag packed with salt-eating automatic packaging is not less than 500, and the machine adjusts to take a subs and tests whether the average weight of the bag is significantly lower.
Our goal is to deny the smallness of the dollar.&lt;}500.&#36;</p>
<p>That's the situation where one-sided hypothetical tests are required.</p>
<h3>Non-normal overall hypothetical tests</h3>
<h4>Assumptions of scale (situation of large samples)</h4>
<p>Or is it the use of the central, very limited logic that is the subject of a normal general hypothesis test, which we'll present only one scenario, that is, the hypothetical test of the 0-1 distribution parameter?
Set Overall as
&#36;&#36;P\bigl{X=x\bigr}=p^{x}\left(1-p\right)^{1-x},\quad x=0,1&#36;&#36;
Two scenarios are selected as
&#36;&#36;H_{0}:p=p_{0},\quad H_{1}:p\neq p_{0}&#36;&#36;
We know by the very limits of the center when the original hypothesis is established and the sample is sufficiently large.
&#36;&#36;U=\frac{\overline{X}-p_{0&#125;&#125;{\sqrt{p_{0}(1-p_{0})/n&#125;&#125;.&#36;&#36;
Approximate to standard normal distribution&#36;N(0,1)&#36; </p>
<p>Available&#36;U&#36;Test export hypothetically tested rejection field
The change will give you the same rejection field as the sample average.</p>
<h4>Hypothetical testing of the value of the index distribution parameter</h4>
<p>The core of the hypothetical test is that the construction of the statistics is not a matter of perception, and first we need to make sure that the distribution of the statistics is known, and then we need to probably at least test the way the statistics are being shifted.
&#36;p(x)=\left(begin{array}ll}
The \lambda e^, \lambda x, \ambda \ &amp; x\ge0, \
0, &amp; \text {Other.}
I'm sorry.&#36;&#36;
由于我们已知&#36;E(X)=\frac{1}{\lambda}&#36;
所以非常自然的思想是 根据均值的偏移方向和均值变形后可能的分布来构造检验统计量量
不妨设原假设为 &#36;H_0:\lambda=\lambda_1&#36;
那么 均值过大或者过小的时候 拒绝原假设  又
&#36;&#36;2\lambda(1}cdots+x n}) =2\lambda\bar}sim\chi 2 &#36;&#36;
It's not that hard to refuse the field.
The construction tests are very dependent on people's experience and knowledge of the distributions.</p>
<h3>Practical application of hypothetical tests</h3>
<h4>Assumed p-test</h4>
<p>In the hypothetical tests we conducted before us, we gave permission to make the first type of error and then determined the rejection field;</p>
<p>In the statistical software currently in use, we often give it to the&#36;p&#36;Value for hypothetical tests.</p>
<p>&#36;p&#36;The method is to calculate the theoretical boundary position of the rejected domain at this point, based on the values we observe, and then calculate the probability of error under our control in the first category based on the assumption that the sample value is on the edge of the rejected domain.&#36;\alpha&#36;  And then the value of the calculations is...&#36;p&#36;Value, therefore&#36;p&#36;Less than the required visibility can be judged to be significant.</p>
<p>Under the original assumption, the P-value is subject to the rule.&#36;[0,1]&#36;The flat distribution of the area requires extrapolation based on the testing of the statistics.</p>
<h4>On sample capacity</h4>
<p>Yes.&#36;p&#36;When the value is getting smaller, the results of the tests are getting more and more obvious.</p>
<p>It's very obvious.&#36;p&#36;The value is smaller as the sample capacity increases.&#36;p&#36;The value sample size is different from what we understand, and the hypothetical tests must be conducted in such a way as to give the size of the sample capacity, and in fact we need to discuss the statistical efficacy of a statistical function that we normally use the probability of a second type of error.&#36;\beta&#36; . To calculate&#36;1-\beta&#36; To be statistically effective as a hypothetical test.</p>
<h3>Effects and minimum sample amount tested</h3>
<p>In the hypothetical test,<strong>Minimum Technical Effects, MDE</strong> is the level of visibility in the given sample (in the case of the&#36;\alpha&#36;), statistical effectiveness ()&#36;1-\beta&#36;The conditions. Tests can be significant in terms of the minimum effect size.</p>
<p>The minimum detectable effect is:
&#36;&#36;\Delta_{\min}=\sqrt{\frac{2\sigma^2}{n&#125;&#125;\cdot(z_{\alpha/2}+z_\beta)&#36;&#36;
The formula is based on the corresponding distribution of the statistically relevant content, the most common two-normal aggregate differences are known, the mean difference Z test is essentially similar to the rest. of which</p>
<ul>
<li>&#36;z_{\alpha/2}&#36;is the normal distribution fraction of the profile level. If it's a unilateral study,&#36;z_\alpha&#36; If other tests are considered, the corresponding distribution will need to be changed.&#36;\alpha&#36;It's a significant level of probability of a first type of error, because it's studied here.&#36;z&#36;The symmetry of distribution will&#36;z_{1-\alpha/2}&#36;Writing&#36;z_{\alpha/2}&#36;</li>
<li>&#36;z_\beta&#36;is the normal distribution fraction of the statistical effect, and the statistical effect is:&#36;1-\beta&#36; The second type of error is a difference between probability and probability, because of the research here.&#36;z&#36;The symmetry of distribution will&#36;z_{1-\beta}&#36;Writing&#36;z_{\beta}&#36;</li>
<li>&#36;\sigma ^{2}&#36;It's the overall variance.&#36;n&#36;is the number of single groups of samples.</li>
</ul>
<p>If the T test is considered, it needs to be replaced with a T-distribution, which is still short-cut because of its symmetrical nature, but needs to be supplemented by the freedom of the T-distribution, which varies from one degree to another.</p>
<p>The card-side tests are used primarily for the independent testing of disaggregated data or for the test of proposed eugenics, and are not usually defined as minimal detectable effects directly by formulas similar to those of Z and T. - It's a Cramer.&#39;s V, etc. measure the effect size. Like<a href="/en/blog/2023/11/05/descriptive-statistics-and-visualization-notes/">Descriptive statistics and visualization</a> / Correlation factors in the /Legation Tables &#36;V&#36;Related coefficients” section</p>
<p>The minimum detectable effect is determined by the level of visibility, statistical efficacy, sample size and standard deviation of the sample.</p>
<ul>
<li>When the sample volume increases or when the standard deviation decreases,&#36;\Delta_{\min}&#36;And then it drops, which means the experiment is more sensitive.</li>
<li>The government is not a party to the law, but it is a party to the law.&#36;\Delta_{\min}&#36;The subsequent rise indicates that greater effects are needed to be tested.</li>
<li>MDE helps researchers to assess the capabilities of the experimental design to ensure a reasonable sample allocation and optimization of resources.</li>
</ul>
<p>Give us the minimum expected effect we want to detect in advance, based on the minimum detectable effect size formula.&#36;\Delta&#36; , the formula can be converted to the formula used to calculate the minimum sample amount required, i.e.
&#36;&#36;n=\frac{2\sigma^2}{\Delta_{\min}^2}\cdot(z_{\alpha/2}+z_\beta)^2&#36;&#36;</p>
<h3>Non-parametric hypothetical tests</h3>
<h4>Introduction</h4>
<p>Non-parametric hypothesis tests are part of non-parametric statistics
And not the key point of parameter statistics is that <strong>Unknown for the overall distribution</strong> We're here to present a few non-parametric hypothetical tests that are very rare.<strong>A quote from non-parametric statistics</strong>
Probability drawings are a classic non-parametric test, and probability drawings make the point from the normal general a straight line on the drawings, and he is the normal QQQ (Quantile-Quantile) map to test whether the sample from the normal aggregate is studied using a fractional number.
In addition, we'll present the Pesason Carp's proposed eugenicity test, which is the non-parametric test on which the comparison is based.
<strong>This is a way to test whether a sample comes from a distribution we've chosen.</strong>
It's a classic problem in non-parametric statistics.</p>
<h4>The idea of Peason's party to match the eugenic test</h4>
<p>We divide the results of random experiments into a complete event. Group &#36;A_{1},A_2,...,A_k&#36;
That's the rim up.<em>{i=1}^{\kappa}\mathbf{A}</em>\, \A\A=A=mathbf}, i,j=1,2, \mathbf{L},k.&#36;
We're assuming that.&#36;H_0&#36;Here we go.
&#36;&#36;p_i=P(A_i)&#36;&#36;
So there's an analysis of the actual and theoretical frequency.
&#36;&#36;\chi^{2}=\sum_{i=1}^{k}\frac{(f_{i}-np_{i})^{2&#125;&#125;{np_{i&#125;&#125;&#36;&#36;
When he was big enough to deviate a lot, then the assumption was fake, the assumption was true.</p>
<h4>Theories of the Peason Cartridges to the Presbyterian Presumptuous Test</h4>
<p>When?&#36;H_0&#36;For real and&#36;n&#36;When you're big enough to count.
&#36;&#36;\chi^{2}=\sum_{i=1}^{k}\frac{(f_{i}-np_{i})^{2&#125;&#125;{np_{i&#125;&#125;=\sum_{i=1}^{k}\frac{f_{i}^{2&#125;&#125;{np_{i&#125;&#125;-n&#36;&#36;
Close to obedience.&#36;\chi^{2}(k-1)&#36;
From this test of the distribution of statistics, we can export the rejection field we need to use.
Because we explained earlier that when the original assumption was false when the deviation was large, it was necessary to select a single-side hypothesis test to reject the field as being a non-existent one-sided one-sided test.&#36;\chi^{2}\geq\chi_{\alpha}^{2}(k-1)&#36;</p>
<p>If we're choosing the hypothesis,&#36;H_0&#36;♪ And if ♪&#36;X&#36;The distribution function contains unknown parameters
First, use the sample to obtain the maximum semblance of the unknown parameter, using the estimate as the parameter value.
We'll come to conclusions.&#36;H_0&#36;For real and&#36;n&#36;When you're big enough to count.
&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;<em>{i}\right)^{2&#125;&#125;{n\hat{p}</em>{i&#125;&#125;=\sum_{i=1}^{k}\frac{f_{i}^{2&#125;&#125;{n\hat{p}<em>-&#36;US&#36;&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;&#36;US&#36;&#36;US&#36;&#36;US&#36;&#36;US&#36;&#36;US&#36;&#36;US&#36;&#36;US&#36;&#36;&#36;US&#36;&#36;&#36;US&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;US&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;
Close to obedience.&#36;\chi^{2}(k-1-r)&#36;
of which&#36;r&#36;is the number of unknown parameters contained in the distribution function
Deny domain &#36;\q\chi</em>{\alpha}^{2}(k-1-r)&#36;</p>
<h4>Operating habits</h4>
<p>Using the Peason Cartrix Probability Test will generally require assurance that the following requirements are used to classify the samples</p>
<ul>
<li>Large sample generally considered &#36;n&gt;50&#36;</li>
<li>Requesting theoretical frequency for groups &#36;np i&gt;5&#36;</li>
<li>General data are divided into 7-14 groups, which can be smaller than 7 groups to satisfy Article II.
The proper grouping of all sample data is in fact the core of the Peason Carpenter's proposed eugenicity test.</li>
</ul>
<h4>Additional elements</h4>
<p>We're testing the calculator's proposed merits only for the theoretical distribution with limited values.</p>
<p>If you want to process a continuous variable, we need to divide the compartments, amend them to a limited range and calculate the probability size of the compartments.</p>
<p>The idea of the Person card for the quality test is very well applied in the luminum table, and we will introduce it in the related and other analyses of the luminous table, all of which have a classification, frequency structure.</p>
