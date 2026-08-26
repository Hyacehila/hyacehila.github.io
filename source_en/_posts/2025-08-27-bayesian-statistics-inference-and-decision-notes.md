---
title: 'Bayesian Statistics: Inference and Decision'
title_zh: 贝叶斯统计：推断与决策
date: 2025-08-27 00:18:46 +0800
categories:
- Data Science
- Probability & Statistical Foundations
tags:
- Statistics
- Bayesian Statistics
- Statistical Decision
author: Hyacehila
mathjax: true
hidden: true
excerpt: Covers conditional methods, Bayesian estimation, hypothesis testing, multiple hypotheses, and decision theory.
description: Covers conditional methods, Bayesian estimation, hypothesis testing, multiple hypotheses, and decision theory.
excerpt_zh: 整理条件方法、贝叶斯估计、假设检验、多假设问题和决策理论。
permalink: /blog/2025/08/27/bayesian-statistics-inference-and-decision-notes/
lang: en
translation_key: 2025-08-27-bayesian-statistics-inference-and-decision-notes
translation_status: machine
translation_source_hash: dbf64b85fec7ecec728f8c35798460532e672e769752b3d7324d8c862a3eb1e6
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<h2>Bayesian statistical inference</h2>
<p>The opening of this chapter will review some of the elements we have been exposed to earlier; then we will conduct a study of statistical extrapolations.</p>
<h3>Conditional approach</h3>
<p>The posteriori distribution is the combined a priori distribution, the total distribution, the sample distribution, the distribution of three types of information in one body.</p>
<p>We have a variety of statistical inferences, such as parameter estimates and hypothetical tests, that extract information from the back distribution.<strong>All statistical inferences have to start from a posteriori distribution.</strong>It's easier to extract information than classical statistics.</p>
<p><strong>The Bayesian method is based on the idea that only the data that have emerged (sampling observations) are considered irrelevant to extrapolation.</strong></p>
<p>Classical statistics tend to think that the estimates of parameters should be neutral, that is,
&#36;&#36;E[\hat{\theta}(x)]=\int_x\hat{\theta}(x)p(x\mid\theta)dx=\theta &#36;&#36;
The average of these is for all possible samples in the sample space, but the vast majority of the samples in the actual sample space are still present, so the Beyers school of the holder's condition view is not biased, which is understandable.</p>
<h3>It seems like a principle.</h3>
<p>It seems that the principle will help us to better understand the ideas of Bayesian statistics as well as the entire system of probability statistics.</p>
<p>Add the following point: Likeness and probability are interchangeable in English. But in statistics, they're very different.</p>
<p>Probability describes the output of random variables when the parameters are known; it seems to describe the output of known random variables.<strong>Possible value of unknown parameter</strong></p>
<h4>Appearance Functions</h4>
<p>If&#36;\mathbf{x}=(x_1,...,x_n)&#36;It's from the density function.&#36;\mathrm{p( x|\theta) }&#36;for a sample, the product:
&#36;&#36;
p(\mathbf{x}\mid\theta)=\prod_{i=1}^np(x_i\mid\theta)
&#36;&#36;
There are two explanations:</p>
<ul>
<li>When?&#36;\theta&#36;Timelines.&#36;p(\mathbf{x}|\theta)&#36;is the joint density function of the sample x;</li>
<li>When the observation of the sample x is given,&#36;p(\mathbf{x}|\theta)&#36;It's a function of an unknown parametric acoustic.&#36;L(\theta)&#36;</li>
</ul>
<h4>It seems like a principle.</h4>
<ul>
<li>Observes&#36;x&#36;After that, doing something about&#36;\theta&#36;of all tests&#36;\theta&#36;Information is contained in an apparent function&#36;L(\theta)&#36;Centre</li>
<li>If two seemingly proportional functions, the ratio constant is&#36;θ&#36;It doesn't matter.&#36;\theta&#36;Contains the same information</li>
</ul>
<h4>An example of the principle.</h4>
<h5>Introduction</h5>
<p>Description of the problem:&#36;\theta&#36;In order to have a positive probability of throwing a coin upward, the following two assumptions are tested:
&#36;&#36;US&#36;US&#36;US&#36;&#36;US&#36;&#36;US&#36;&#36;US&#36;&#36;US&#36;US&#36;US&#36;&#36;US&#36;&#36;US&#36;&#36;US&#36;&#36;US&#36;US&#36;US&#36;US&#36;US&#36;&#36;US&#36;US&#36;&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;&#36;&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;&#36;US&#36;US&#36;US&#36;US&#36;&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;&#36;US&#36;US&#36;US&#36;US&#36;US&#36;&#36;US&#36;US&#36;US&#36;US&#36;&#36;US&#36;US&#36;US&#36;&#36;US&#36;US&#36;&#36;&#36;&#36;US&#36;&#36;&#36;&#36;&#36; US&#36; US&#36;&#36;&#36;&#36;&gt; 1/2
As a result, a series of separate tests of the coin were conducted, resulting in nine positive and three negatives. How to make a reasonable judgment?</p>
<p>The important question is:<strong>A series of separate experiments.</strong>He may have two scenarios.</p>
<p>We decided to do 12 experiments in advance, subject to the two distributions, to give the corresponding approximation.&#36;&#36;L_1(\theta)=P_1(X=x\mid\theta)=\begin{pmatrix}n\x\end{pmatrix}\theta^x\left(1-\theta\right)^{n-x}=220\theta^9\left(1-\theta\right)^3&#36;&#36;
We want to terminate the experiment after three failures, that is, the negative two-part distribution, and give the corresponding approximation that there is.
&#36;&#36;L_2\left(\theta\right)=P_2\left(X=x\mid\theta\right)=\binom{k+x-1}{x}\theta^x\left(1-\theta\right)^{n-x}=55\theta^9\left(1-\theta\right)^3&#36;&#36;</p>
<p><strong>It seems that the principle tells us that the sample information in this case is the same, which is consistent with our previous guess, after all, only the differences in experimental methods.</strong></p>
<h5>A hypothetical test for classical statistics</h5>
<p>Use the hypothetical tests of classical statistics to deal with two questions.&#36;0.05&#36;As a visible level </p>
<ul>
<li>Use two distribution models not rejected&#36;H_0&#36;</li>
<li>Using negative two-part distribution model rejected&#36;H_0&#36;</li>
</ul>
<p>That's a contradiction to the apparent principle.</p>
<h5>Hypothetical tests of Bayesian statistics</h5>
<p>Obviously simple versus complicated, using an a priori distribution without information
&#36;&#36;\pi(\theta)=\pi_{0}I_&#123;&#123;0.5&#125;&#125;(\theta)+\pi_{1}g_{1}(\theta)&#36;&#36;
of which&#36;\pi_0=\pi_1=1/2,\mathrm{~g_1(\theta)=U(0.5,1)}&#36;
Calculating the Bayesian factor.
&#36;&#36;B_i^{\pi}\left(x=9\right)=\frac{\alpha_0\pi_1}{\alpha_1\pi_0}=\frac{P_i\left(X=9\mid\theta=1/2\right)}{m_i\left(x=9\right)}&#36;&#36;
Molecular
&#36;&#36;P_i\left(X=9\mid\theta=1/2\right)=k_i\theta^9\left(1-\theta\right)^3=0.000244k_i&#36;&#36;
Factor
&#36; \begin{aligned}
m i}(x=9)&amp;=\int_{1/2}^{1}P_{i}\left(X=9\mid\theta=1/2\right)g_{1}(\theta)d\theta   \
&amp;=\int_{1/2}^1k_i\theta^9\left(1-\theta\right)^3\cdot2d\theta  \
&amp;=2k_i\int_{1/2}^1(\theta^9-3\theta^{10}+3\theta^{11}-\theta^{12})d\theta  \
&amp;=0.000666k_i
\end{aligned}&#36;&#36;
因此两种情况下的贝叶斯因子实际上相同
&#36;&#36;== sync, corrected by elderman ==
Bayesian refusal.&#36;H_0&#36; We choose to accept.&#36;H_0&#36;</p>
<h5>Answering the contradiction.</h5>
<p>The Bayesian schools of statistics support the apparent principle, so they believe that the hypothetical test results given in classical statistics are wrong;</p>
<p>For classic statistical schools: in fact, many statistical methods do not satisfy the approximation principle, they support the approximation principle when using a very similar estimate, but not when they find MLE.</p>
<p>Some statisticians think they need to know.&#36;f(x|\theta)&#36;It's a very reasonable requirement that these gaps lead to a difference in the results of the final statistical inference; they demand that<strong>The method of experimental design is known.</strong></p>
<h3>Bayesian point estimates</h3>
<h4>Definition of Bayesian estimates</h4>
<p>Bayes usually has three.</p>
<ul>
<li>Post-calculations estimate&#36;\hat{\theta}_{MD}&#36;</li>
<li>After-check median estimate<em>{</em>{Me&#125;&#125;&#36; </li>
<li>Post-expected estimate&#36;\hat{\theta}_E&#36;</li>
</ul>
<p>They can be used to estimate parameters.</p>
<ul>
<li>Post-calculations estimate&#36;\hat{\theta}_{MD}&#36;This is calculated by using the knowledge in mathematical analysis to greatly characterize the post-distribution density function (e.g., by looking for a post-numeric bias and looking for a point of bias to zero).</li>
<li>Post-expected estimate&#36;\hat{\theta}_E&#36; The method of calculation is to calculate the expectations of a posteriori distribution using the techniques of probabilistic theory.</li>
<li>After-check median estimate<em>{</em>Because it's not easy to calculate.</li>
</ul>
<h4>Several examples</h4>
<p>Estimated failure rate &#36;\theta&#36;, randomly extracted from a product today&#36;n&#36; items, of which unsatisfactory&#36;X&#36;Obey.&#36;B(n,\theta)&#36;, general selection&#36;Be(\alpha,\beta)&#36; Yes&#36;\theta&#36;A priori distribution, set&#36;\alpha&#36;Beta is known, please.&#36;\theta&#36; Bayes estimates
Based on co-examining the distribution, the later distribution is as follows:
&#36;&#36;Be(\alpha+x,\beta+n-x)&#36;&#36;</p>
<p>There is.
What's that?<em>{MD}=\frac{\alpha+x-1}{\alpha+\beta+n-2},\quad\hat{\theta}</em>{E}=\frac{\alpha+x}{\alpha+\beta+n}&#36;&#36;</p>
<p>As you can see, if we choose the Bayesian hypothesis for a priori distribution,&#36;\alpha=\beta=1&#36;
&#36;&#36;\hat{\theta}<em>{</em>== sync, corrected by elderman ==
As you can see, the post-absorption estimates are very similar.</p>
<p>Some of the methods of estimation in classic statistics are the special case of Bayesian estimates in certain circumstances, as evidenced by this example.</p>
<p>At the same time, this posteriori estimate of expectations is more reasonable:&#36;x&#36;All at 0:00.</p>
<p>Set&#36;x&#36;From the following index distribution<strong>An observation value</strong>
&#36;&#36;
p(x|\theta)=e^{-(x-\theta)},\quad x\geq\theta
&#36;&#36;
It also uses Cossi's distribution as an a priori distribution of the calf, namely:
&#36; \pi (\theta)=\frac{1}{\1+theta^2},:-\info&lt;\theta&lt;That's a good idea.
Maximum posterior estimate for the scavenger&#36;\hat{\theta}_{MD}&#36;</p>
<p>Easy to calculate later distribution to
&#36;&#36;\pi(\theta|x)=\frac{e^{-(x-\theta)&#125;&#125;{m(x)(1+\theta^2)\pi},\theta\le x&#36;&#36;
<strong>When analysing the numerical estimates, the marginal density of the denominator is not important because it does not contain&#36;\theta&#36;</strong></p>
<p>The result of the logarithmic bias is 0.
&#36;&#36;\theta=1&#36;&#36;
It's clearly unreasonable.
Or is that our estimate always this value, regardless of the sample? That doesn't make any sense. </p>
<p>Directly search for back-density and no more logarithmics.
&#36;&#36;\frac{d}{d\theta}\pi(\theta|x)=\frac{e^{-x&#125;&#125;{m(x)\pi}\biggl[\frac{e^\theta}{1+\theta^2}-\frac{2\theta e^\theta}{\left(1+\theta^2\right)^2}\biggr]=\frac{e^{-x}e^\theta\left(\theta-1\right)^2}{m(x)(1+\theta^2)^2\pi}\ge0&#36;&#36;
Which means...&#36;\theta&#36;It's a single increase.&#36;\theta\le x&#36; Therefore...
&#36;&#36;\hat{\theta}_{MD}=x&#36;&#36;</p>
<h4>The precision of the Bayesian dot.</h4>
<p>In mathematical statistics, we use the average error as a measure of the estimated error, and in the Bayesian estimate, the later average difference is used to consider the estimated error.</p>
<p>Set parameters&#36;\theta&#36;Backup distribution is&#36;\pi(\theta|x)&#36;The Bayesian estimate.&#36;\hat{\theta}&#36;,&#36;(\theta-\hat{\theta})^2&#36;The Retrospective Expectations
&#36;&#36;
PMSE(\hat{\theta}\Big|x)=E^{\theta|x}(\theta-\hat{\theta})^{2}
&#36;&#36;
Called&#36;\hat{\theta}&#36;, the square root is referred to as the standard balance Bad</p>
<p>There's an explanation for PMSE.</p>
<ul>
<li>&#36;E^{\theta|x}&#36; Indicating a condition distribution &#36;\pi(\theta|x)&#36;Expectations.</li>
<li>♪ When ♪<em>{E}=\boldsymbol{E}(\theta|x)&#36;时，则&#36;PMSE(\hat{\theta}</em>== sync, corrected by elderman ==</li>
<li>There's a relationship between the ex post average and the ex post.
&#36; \begin{aligned}
PMSE&amp; =E^{\theta|x}(\theta-\hat{\theta})^{2}  \
&amp;=E^{\theta|x}[(\theta-\hat{\theta}<em>{E})+(\hat{\theta}</em>{E}-\hat{\theta}<em>{})]^{2} \
&amp;=Var(\theta|x)+(\hat{\theta}</em>{E}-\hat{\theta})^{2} \
&amp;\geq Var(\theta|x)
\end{aligned}&#36;&#36;</li>
</ul>
<p>Which means...<strong>Retrospective expectations are the lowest PMSE estimates and therefore the most common method of estimation.</strong></p>
<p>As you can see, the calculation and evaluation of Bayesian estimates is much simpler than classical statistics.</p>
<p>A discrete example.
Non-conformity rate for the set of products &#36;\theta&#36; The inspection is a one-off exercise until the first non-conformity is discovered, if&#36;X&#36;For the number of products inspected when the first non-conformity was discovered, then&#36;X&#36;Subject to geometric distribution, distribution is classified as:
&#36;&#36;
P(X=x|\theta)=\theta(1-\theta)^{x-1},x=1,2,\cdots
&#36;&#36;
Set &#36;\theta&#36; The a priori distribution is &#36;P(\theta=\frac{t}{4})=\frac{1}{3},i=1,2,3&#36; , only one now&#36;x&#36;
Sample observation values&#36;x=3&#36;Please.&#36;\theta&#36; Maximum post-examining estimates, post-examining expectations estimates and calculation of their errors
There's less back-to-back exercise on individual discrete samples.
&#36;&#36;P(\theta=i/4|X=3)=\frac{P(X=3,\theta=i/4)}{P(X=3)}=\frac{4i}{5}(1-\frac{i}{4})^{2},i=1,2,3&#36;&#36;
As a result, the back count is estimated at &#36;hat(theta}<em>= &#36;1/4
Post-expected estimate</em>== sync, corrected by elderman ==
Calculating the squaredness of the two-step original and one-step original rectangles Bad
&#36; \begin{aligned}
Var (\theta|x)&amp; =E(\theta^{2}\Big|x)-E^{2}(\theta\Big|x)  \
&amp;=17/80-(17/40)^2=51/1600
\end{aligned}&#36;&#36;
计算后验均方误差PMSE
&#36;&#36;\begin{aligned}
PMSE(\hat{\theta}|x)&amp; =Var(\theta\big|x)+(\hat{\theta}<em>{MD}-\hat{\theta}</em>{E})^{2}  \
&amp;=51/1600+(1/4-17/40)^{2}=\frac{1}{16}
\end{aligned}&#36;&#36;</p>
<h3>Estimates</h3>
<h4>Trustable spaces</h4>
<p>The Bayesian estimation and core is the construction of a credible zone, that is, two statistics. &#36;\hat{\theta}_L=\hat{\theta}_L(x)&#36; and&#36;\hat{\theta}_U=\hat{\theta}_U(x)&#36; Make
&#36;&#36;
P(\hat{\theta}_L\leq\theta\leq\hat{\theta}_U\mid x)\geq1-\alpha
&#36;&#36;
There is a difference between the levels of trust and trust between the levels of trust and those in classic statistics, although they are similar concepts.</p>
<ul>
<li>Credible ranges are for random variables.&#36;\theta&#36;It's a study. The confidence interval is for a certain number.&#36;\theta&#36;Research, several times using this confidence room to cover it.&#36;\theta&#36; The frequency interpretation is not meaningful for one or two uses, and in fact, in the actual study, the confidence zone is often used as a credible zone.</li>
<li>The number of tectonic axes is not easy to estimate in classical statistics, but it is not necessary to have such a tectonic structure in a credible range, which is better calculated using a lateral distribution.</li>
</ul>
<h4>A credible zone at the end.</h4>
<p>In fact, since Bayes' estimates can only be accomplished by using a posteriori density, they are actually less difficult than mathematical estimates.</p>
<p>If you're aware of a posteriori distribution, you just need a checklist to get the probability everywhere.
&#36;&#36;\theta_L=\theta_{0.01},\theta_R=\theta_{0.91};\
\\theta_L=\theta_{0.05},\theta_R=\theta_{0.95}&#36;&#36;
Which one should we choose?</p>
<p>In this subsection, we ask for the distribution of the steroids, which is
&#36;&#36;\theta_L=\theta_{\frac{\alpha}{2&#125;&#125;,\theta_R=\theta_{1-\frac{\alpha}{2&#125;&#125;&#36;&#36;
All we have to do is give the floor and the ceiling.</p>
<p>When the distribution is relatively simple, we can get the results we need by tabulation; when the distribution is more complex than direct research, computer technology can help us to do the calculations.</p>
<h4>Maximum post-density (HPD) credible range</h4>
<h5>Definitions</h5>
<p>Waiting at the end is the best place to be trusted? In fact, we've already explained in mathematical statistics that the best confidence should have the shortest length, and that we use the symmetry because it's the shortest length and it's simple enough to symmetry the distribution function. </p>
<p>Here we're going to tell you how to find the shortest credible zone, the HPD.</p>
<p>Definitions: setting parameters&#36;\theta&#36;Post-density as&#36;\pi(\theta|x)&#36;, for given probability1-&#36;\alpha(0{)&lt;}\alpha{&lt;}1)&#36;, 若在直线上存在这样一个子集&#36;C&#36;, meets the following two conditions:</p>
<ul>
<li>&#36;\mathrm{P(C|x)=1-\alpha}&#36;</li>
<li>&#36;\text{对任给}\theta_1{\in}\mathbb{C}\text{和 }\theta_2\notin C\text{,总有}\pi(\theta_1|\mathbf{x}){\geq}\pi(\theta_2|\mathbf{x})&#36;
And call C is&#36;\theta&#36;The credible level is the maximum back-density of (1-alpha), the short (1-alpha) HPD and, if C is a zone, the C is also called&#36;1-\alpha&#36; HPD Trustable Area</li>
</ul>
<p>A simple understanding of this HPD-responsible zone, which is very much understood, is to look for a larger segment of the density function, which would minimize the length of the credible zone, as follows: Icon
<img src="/assets/images/probability-statistics-notes/bayesian-statistics-inference-and-decision-notes-01.png" alt="Bayesian Statistics 2"></p>
<h5>Some explanations.</h5>
<p>Some basic explanations for a credible HPD zone</p>
<ul>
<li>The reprocessing of discrete random variables is difficult to calculate and generally does not study.</li>
<li>A single post-peak density of HPD is always available.</li>
<li>Multi-peak post-density often gives HPD credibility to multiple unconnected zones. Set
About multi-peak post-density</li>
<li>The emergence of multi-peak post-density is often caused by inconsistencies between a priori and sample information, which are important for Bayesian statistics.</li>
<li>Co-prospecting is mostly single-peak, which must lead to a single-facing distribution, which may conceal the many-facility resistance that should have been generated, so be careful to use co-prospecting.
You should be careful when using a credible HPD.</li>
<li>HPD Trusted Areas are not strictly Trusted Areas</li>
<li>The HPD trust zone is also a symmetrical zone at single peaks;</li>
<li>(b) When single peaks, asymmetrics, solve by computer numerical methods;</li>
<li>When multiple peaks, recommend abandoning HPD guidelines and using a connected symmetrically credible zone estimate</li>
</ul>
<h5>Computer asymmetrical HPD trustable range for the solver of a single-scale value</h5>
<p>In fact, the process of the intertinction of computers is very well understood.</p>
<ul>
<li>Give the initial&#36;k&#36;Value</li>
<li>Calculate&#36;\pi(\theta|x)=k&#36; Got it.&#36;\theta_1,\theta_2&#36;</li>
<li>Calculate&#36;\theta_1,\theta_2&#36;Go, go, go!&#36;\pi(\theta|x)&#36;%1 points </li>
<li>If the confidence is greater than what is required, increase it.&#36;k&#36; Otherwise, it's down.&#36;k&#36; Continue with the iterative phase two.
It's enough to understand that.</li>
</ul>
<h5>Large sample method</h5>
<p>In the case of large samples, use a near HPD-like trustable zone.</p>
<p>Proof of: &#36;n&#36; The blogger says:&#36;\pi_n(\theta|x)&#36;Close to obedience. &#36;N(\mu^\pi(x),V^\pi(x))&#36; Here&#36;\mu^\pi(x)&#36;and&#36;V^\pi(x)&#36;Aftervalid averages and backer averages, respectively Bad</p>
<p>He's a symmetrical single-peak distribution, a consistent HPD-respondent zone and a symmetrical zone. Out&#36;\theta&#36; The level of credibility is similar to 1-&#36;\alpha&#36; And the HPD zone is
&#36;&#36;(\mu^\pi(x)-u_{\alpha/2}\sqrt{V^\pi(x)},\mu^\pi(x)+u_{\alpha/2}\sqrt{V^\pi(x)})&#36;&#36;</p>
<p><strong>Big samples don't need to calculate the posteriori distribution.</strong>
Take a rather interesting example of a large sample of HPD.
Number of weekly fire incidents in a currency X Subject to Porpine distribution &#36;P(\theta)&#36;, no information a priori distribution of the thorium is known, and no information a priori is considered&#36;\pi(\theta)=\theta^{-1}I_{(0,\infty)}(\theta)&#36; is appropriate. Set the total number of fire incidents in 5 weeks to be 3 with a mean distribution of porcelain &#36;\theta&#36; The level of credibility is 90% of the HPD trustable area, using a large sample method;
We should be able to calculate the post-censorship distribution, but five observations in five weeks, and we only have three total observations.
The core of the Porcelain distribution is
&#36;&#36;\theta^ke^{-\theta}&#36;&#36;
We might want to take a look at 11,100 over the next five weeks.
So the sample combines to be
&#36;&#36;\theta^3e^{-5\theta}&#36;&#36;
So the back-up distribution is positive.
&#36;&#36;\theta^2e^{-5\theta}&#36;&#36;
This is the base of Gamma's distribution, and the average of the posteriori distribution is calculated as &#36;\frac{3}{5}&#36; Square difference&#36;\frac{3}{25}&#36;
Based on the large sample method, there's a study.&#36;N(\frac{3}{5},\frac{3}{25})&#36;90% HPD, the trustable zone, because the normal distribution is symmetrical, the study can wait for the trustable zone.</p>
<h3>Assumptions test</h3>
<p>The hypothetical test is also a major category of questions studied throughout classical statistics, and after the estimates are completed, it is natural to test the reasonableness of our estimates, and to test the statistical levels of assumptions in the classical statistics, which are more than the need to construct them;</p>
<h4>General hypothetical test method</h4>
<h5>Assumptions in classic statistics</h5>
<ol>
<li>Create original assumptions&#36;H_0&#36;and alternative assumptions&#36;H_1&#36;</li>
<li>Select the number of tests&#36;T=T(x)&#36; When the original assumption is&#36;H_0&#36;When it's true, it's known.</li>
<li>Visibility of the given&#36;\alpha&#36; The probability of making a first-class error is less than the probability of a denial.&#36;\alpha&#36; </li>
<li>Sample observation values&#36;x&#36;When you fall in the rejected domain, you reject the original hypothesis.&#36;H_0&#36; Otherwise, the original assumption is retained</li>
</ol>
<p>As with the construction of the pivotal, it is difficult to test the determination of statistics in classical statistics.</p>
<h5>Assumptions in Bayes statistics</h5>
<ol>
<li>Probability of post-testing&#36;\pi(\theta|x)&#36; Then calculate assumptions separately&#36;H_0&#36; &#36;H_1&#36; Post-probability &#36;\alpha_i=P(\theta_i|x)&#36; </li>
<li>When the probability ratio is back-checked&gt;1&#36; 时不拒绝&#36;H_0&#36;  &#36;\frac{\alpha_0}{\alpha_{1&#125;&#125;&lt;1&#36;  时不拒绝&#36;H_{1}&#36; 接近&#36;A &#36;1-million period without judgment, without any conclusion.</li>
</ol>
<h5>A comparison of hypothetical thinking between schools</h5>
<p>It's easy to see.</p>
<ul>
<li>The Bayesian hypothesis test is easier to understand, simpler.</li>
<li>The Bayesian hypothetical test does not need to select the statistical test to determine the sample distribution</li>
<li>No prior indication of a significant level of visibility to determine the area of rejection</li>
<li>It's easy to extend to multiple hypotheticals or to look for the highest probability of a posteriori.</li>
</ul>
<p>In fact, the Beyers statistical hypothetical test is the same as the classic statistical principle of probability, but it doesn't require counter-proofing.</p>
<h5>Details of the Bayesian hypothetical test</h5>
<p>We're here to tell you how to do the Bayesian hypothetical tests.</p>
<p>Post-probability density calculations: a little
Assumptions&#36;H_0&#36;  Assumptions&#36;H_1&#36; &#36;H=H_{0}\cup H_1&#36; For the total space, all the assumptions mean&#36;\theta\in H&#36; </p>
<ol>
<li>Calculation assumptions&#36;H_0&#36;  Post-probability &#36;P(H_0|x)=\int_{H_0}^{}\pi(\theta|x)d\theta\triangleq\alpha_0&#36; </li>
<li>Calculation assumptions&#36;H_1&#36;  Post-probability &#36;P(H_1|x)=\int_{H_1}^{}\pi(\theta|x)d\theta\triangleq\alpha_1&#36; </li>
<li>Calculate the probability ratio for post-test&#36;\frac{\alpha_0}{\alpha_{1&#125;&#125;&#36;</li>
</ol>
<ul>
<li>&#36;\frac{\alpha_0}{\alpha_{1&#125;&#125;&gt;1&#36; 时不拒绝&#36;H_0&#36;  也就是接受&#36;H_0&#36;</li>
<li>&#36;\frac{\alpha_0}{\alpha_{1&#125;&#125;&lt;1&#36;  时不拒绝&#36;H_{1}&#36; 也就是接受&#36;H_1&#36;</li>
<li>&#36;\frac{\alpha_0}{\alpha_{1&#125;&#125;\approx 1&#36; No judgement, no conclusion, further sampling or a priori correction.</li>
</ul>
<p>The hypothetical test in Bayes statistics is being translated into a problem of points.</p>
<ul>
<li>Simple assumption: the assumption we're making at this time is&#36;\theta=x&#36; </li>
<li>Complex assumptions: assuming the corresponding parameters are valued as a space</li>
</ul>
<h4>Beyes Factor and Presumption Test</h4>
<p>The Beyers factor can help us understand better the Beyers hypothetical test.</p>
<p>Two assumptions&#36;\Theta_{0}&#36;and&#36;\Theta_{1}&#36;The probabilities are the same.&#36;\pi_0&#36;and&#36;\pi_1&#36;, the probability of a later examination is different&#36;\alpha_0&#36;and&#36;\alpha_1&#36;, and then
&#36;&#36;B^\pi(x)=\frac{\text{后验机会比&#125;&#125;{\text{先验机会比&#125;&#125; = \frac { \alpha _ 0 / \alpha _ 1 }{ \pi _ 0 / \pi _ 1 }=\frac{\alpha_0\pi_1}{\alpha_1\pi_0}&#36;&#36;
Bayes factor</p>
<p>I can see that.</p>
<ul>
<li>The Beyers factor depends on data at the same time.&#36;x&#36;& A priori Distribution&#36;\pi(\theta)&#36;</li>
<li>Two opportunities, compared to dichotomy, reduce the a priori distribution, and highlight the impact of data.</li>
<li>Bates factor reflects data&#36;x&#36;Support the original assumption&#36;H_0&#36;The extent of the system (as with the opportunity to divide by one)</li>
</ul>
<h4>Simple assumptions versus simple assumptions</h4>
<p>Now we're looking at the Beyers factor in a few different scenarios, and we're continuing to strengthen the most central hypothesis test we've given: the probability ratio.</p>
<p>First, look at simple assumptions.
Assumptions are:
&#36;&#36;H_{0}:\Theta_{0}={\theta_{0&#125;&#125;\leftrightarrow H_{1}:\Theta_{1}={\theta_{1&#125;&#125;.&#36;&#36;
There's a chance of a corresponding post-test.
&#36; \begin{aligned}\alpha 0&amp;=P(\Theta_0|\boldsymbol{x})=\frac{f(\boldsymbol{x}|\theta_0)\pi_0}{f(x|\theta_0)\pi_0+f(\boldsymbol{x}|\theta_1)\pi_1},\\alpha_1&amp;=P(\Theta_1|\boldsymbol{x})=\frac{f(\boldsymbol{x}|\theta_1)\pi_1}{f(\boldsymbol{x}|\theta_0)\pi_0+f(\boldsymbol{x}|\theta_1)\pi_1},\end{aligned}&#36;&#36;
<em>It's a definition of the probability, and it's easy to calculate because it corresponds to the discrete probability space.</em></p>
<p>So, the chance is the one.
&#36;&#36;\frac{\alpha_0}{\alpha_1}=\frac{\pi_0f(\boldsymbol{x}|\theta_0)}{\pi_1f(\boldsymbol{x}|\theta_1)}&#36;&#36;
Calculating the Beyers Factor
&#36;&#36;B^\pi(\boldsymbol{x})=\frac{\alpha_0/\alpha_1}{\pi_0/\pi_1}=\frac{f(\boldsymbol{x}|\theta_0)}{f(\boldsymbol{x}|\theta_1)}.&#36;&#36;
You want to reject the original hypothesis, which is to ask for &#36;\frac{\alpha 0}&lt;1&#36; 也就是&#36;&#36;\frac{f(\boldsymbol{x}|\theta_1)}{f(\boldsymbol{x}|\theta_0)}&gt;\frac{\pi_0}{\pi_1}.&#36;&#36;</p>
<p>Intuitive understanding is that the density function is more than the threshold, which is similar to the basic result of N-P reasoning.</p>
<ul>
<li>From now on, it's obvious. &#36;B^\pi(\boldsymbol{x})&#36; It's supposed to be a ratio of opportunity to data, and he's not dependent on a priori distribution, but on the sample.</li>
<li>So we're taking the Beyers.&#36;B^\pi(\boldsymbol{x})&#36; Consider Data&#36;x&#36;- Yes.&#36;H_0&#36;Level of support</li>
</ul>
<h4>Complex assumptions versus complex assumptions</h4>
<h5>Calculating the Beyers Factor</h5>
<p>Considering the following hypothetical test issues:
&#36;&#36;H_0:\theta\in\Theta_0\leftrightarrow H_1:\theta\in\Theta_1,&#36;&#36;
And we can re-write the a priori density function in the following form.<em>The rewriting is intended to facilitate the subsequent calculation and representation.</em>
&#36;&#36;\left.\pi(\theta)=\left{\begin{array}{ll}\pi_0g_0(\theta),&amp;\theta\in\Theta_0,\\pi_1g_1(\theta),&amp;\theta\in\Theta_1,\end{array}\right.\right.&#36;&#36;</p>
<p>Rewrite the probability ratio under this mark.
&#36;&#36;\frac{\alpha_0}{\alpha_1}=\frac{\int_{\Theta_0}f(\boldsymbol{x}|\theta)\pi_0g_0(\theta)\mathrm{d}\theta}{\int_{\Theta_1}f(\boldsymbol{x}|\theta)\pi_1g_1(\theta)\mathrm{d}\theta},&#36;&#36;
<em>The form of the fraction is that complex density functions do not need to process single points.</em>
Give me the Bates.
&#36;&#36;B^\pi(\boldsymbol{x})=\frac{\alpha_0/\alpha_1}{\pi_0/\pi_1}=\frac{\int_{\Theta_0}f(\boldsymbol{x}|\theta)g_0(\theta)\mathrm{d}\theta}{\int_{\Theta_1}f(\boldsymbol{x}|\theta)g_1(\theta)\mathrm{d}\theta}=\frac{m_0(\boldsymbol{x})}{m_1(\boldsymbol{x})}.&#36;&#36;
The ratio of the border-distribution.</p>
<h5>Explain the Beyers.</h5>
<ul>
<li>The Beyers factor at this time is not an apparent comparison, but it can be seen as an apparent weighting form, partially eliminating the effects of the a priori distribution, emphasizing the sample.</li>
<li>If you set up a new one,<em>0&#36; 与&#36;\hat{\theta}<em>1&#36; 分别是&#36;\theta&#36;在&#36;\Theta</em>{0}&#36;与&#36;\Theta</em>{1}&#36;上的极大似然估计(MLE), 那么经典统计中所使用的似然比统计量是贝叶斯因子&#36;Special case of &#36; ^ (\mathbf{x}</li>
<li>The response of the Beyers factor to changes in sample information is sensitive, while the reaction to changes in a priori information is slow (this is explained by the complexity and complexity of the information, and simply by the fact that the Beyers factor and the a priori are completely irrelevant to the simple assumption)</li>
</ul>
<h4>Simple assumptions versus complex assumptions</h4>
<p>Considering the following hypothetical test issues:
&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;&#36;US&#36;&#36;US&#36;US&#36;US&#36;US&#36;&#36;US&#36;US&#36;US&#36;&#36;US&#36;&#36;US&#36;&#36;US&#36;US&#36;US&#36;&#36;US&#36;US&#36;US&#36;US&#36;&#36;US&#36;&#36;US&#36;US&#36;&#36;US&#36;&#36;US&#36;&#36;US&#36;&#36;US&#36;&#36;&#36;&#36;&#36;US&#36;&#36;&#36;US&#36;&#36;&#36;US&#36;&#36;&#36;&#36;US&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;...&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;...&#36;&#36;&#36;&#36;&#36;...&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;...&#36;&#36;&#36;&#36;&#36;...<em>0：\theta=\theta</em>1 \theta\neq\theta &#36;0
This is the most complicated situation.</p>
<p>If we use a continuous density function directly, then the a priori probability of a single point must be zero, and there's no way to calculate it.
So we need to solve the problem by rewriting the density function in a complex and complex way to add parameters. </p>
<p>- There's a deformation.
&#36;&#36;\pi(\theta)=\pi_0I_{\theta_0}(\theta)+\pi_1g_1(\theta)&#36;&#36;
Of which&#36;I&#36;It's a token function that's added to it.&#36;\theta=\theta_0&#36;. Time to take to 1  &#36;\pi_0+\pi_1=1&#36; We can assume that the a priori density at this time is composed of two separate and continuous parts, so there is a certain number of people who are separated from each other.
&#36;&#36;\left.\pi(\theta)=\left{begin{array}ll}pi 0,&amp;\theta=\theta_0,\\pi_1g_1(\theta),&amp;\theta\neq\theta_0,\end{array}\right.\right.&#36;&#36;
计算边缘密度有
&#36;&#36;m(\boldsymbol{x})=\int_\Theta f(\boldsymbol{x}|\theta)\pi(\theta)\mathrm{d}\theta=\pi_0f(\boldsymbol{x}|\theta_0)+\pi_1m_1(\boldsymbol{x}),&#36;&#36;
其中&#36;m_1(x)&#36;为&#36;&#36;m_1(\boldsymbol{x})=\int_{\theta\neq\theta_0}f(\boldsymbol{x}|\theta)g_1(\theta)\mathrm{d}\theta.&#36;&#36;
分别计算两个假设下的后验密度有
&#36;&#36;\alpha_0=\pi(\Theta_0|\boldsymbol{x})=\frac{\pi_0f(\boldsymbol{x}|\theta_0)}{m(\boldsymbol{x})},\quad\alpha_1=\pi(\Theta_1|\boldsymbol{x})=\frac{\pi_1m_1(\boldsymbol{x})}{m(\boldsymbol{x})}.&#36;&#36;
因此可以计算后验机会比
&#36;&#36;\frac{\alpha_0}{\alpha_1}=\frac{\pi_0f(\boldsymbol{x}|\theta_0)}{\pi_1m_1(\boldsymbol{x})}.&#36;&#36;
计算贝叶斯因子有
&#36;&#36;B^\pi(\boldsymbol{x})=\frac{\alpha_0/\alpha_1}{\pi_0/\pi_1}=\frac{f(\boldsymbol{x}|\theta_0)}{m_1(\boldsymbol{x})}.&#36;&#36;</p>
<p>It is easier to see the Beyers factor in its presentation, and there are no parameters that we have added to the study to support it, so we tend to calculate the Beyers factor in the actual study, and using the Beyers factor to calculate the probability later is a simple equation.</p>
<h4>One example</h4>
<p>This example is not about the knowledge of computation. We have to explain the results of the calculations.</p>
<p>Set From Normal General&#36;\mathbb{N}(0,1)&#36;randomly extract a capacity as&#36;10&#36;The sample x, the average sample value &#36;\overline{x}=1.5&#36;, test two assumptions:
I'm sorry.
H 0: \theta\leq 1, \quadH 1:\theta&gt;1
&#36;&#36;
&#36;\text{取}\theta\text{的共轭先验分布为N}(0.5,2)\text{}&#36;</p>
<p>Obviously complex, complex, and the chances of a later check on formula calculations. - Yeah.
&#36; \begin{aligned}\alpha 0=&amp;\mathrm{P(\theta\leq1|x)=0.0708}\\alpha_1=&amp;\mathrm{P(\theta&gt;1|lpha 0=0/922}end{aligned}&#36;&#36;&#36;
Post-opportunity versus support assumptions&#36;H_1&#36;</p>
<p>Calculate a priori chances against
&#36;&#36;\pi=0.6368,\quad\pi_1–0.3632&#36;&#36;
A priori opportunity against support&#36;H_0&#36;</p>
<p>Calculating the Beyers.
&#36;&#36;B^\pi(x)=0.0434&#36;&#36;
The Beyers factor supports the hypothesis.&#36;H_1&#36;</p>
<p>It's a contradiction between our beyers and our a priori judgment, and it's a response to our conclusions. <strong>The Beyers factor is more concerned with sample information, and in fact this phrase is valid for any type of beyers hypothetical test.</strong> </p>
<h3>Projections extrapolation</h3>
<p>We're making statistical inferences about the future observations of random variables, and there's no chapter in mathematical statistics that corresponds to them.</p>
<h4>A brief introduction.</h4>
<p>What we need to do is estimate the future observations of random variables based on the situation that is known, which will basically be divided into the following:</p>
<ul>
<li>No observations, parameters&#36;θ&#36;Unknown, forecast&#36;X&#36;（&#36;X&#36;Organisation&#36;\theta&#36;(parameters)</li>
<li>Observation information, parameters&#36;θ&#36;Unknown, forecast&#36;X&#36;（&#36;X&#36;Organisation&#36;\theta&#36;(parameters)</li>
<li>Observation information, parameters&#36;θ&#36;Unknown, forecast&#36;Z&#36;（&#36;Z&#36;Organisation&#36;\theta&#36;(parameters)</li>
</ul>
<h4>Projections in the absence of observations</h4>
<p>And although we don't have any observations at this point, there's a sample distribution and a priori distribution of parameters, which naturally gives a marginal distribution.
&#36;&#36;m(x)=\int_{\Theta}p(x|\theta)\pi(\theta)d\theta &#36;&#36;
As our forecast distribution, this is called a priori projection distribution.</p>
<p>Projections methods:
Use the projection of the expected, median or agglomerations of the projected distribution (as we did in the Bayesian point estimate)</p>
<p>Use a certain confidence to calculate the confidence interval for the projected distribution (as we did in the Bayesian estimation)</p>
<h4>If you have X-observation data, predict X.</h4>
<p>Calculates post-femination&#36;\pi(\theta|{x})&#36;  We calculate our projected distribution using a posteriori density.
&#36;&#36;m(x\mid\mathbf{x})=\int_{\Theta}p(x\mid\theta)\pi(\theta\mid\mathbf{x})d\theta &#36;&#36;
Called the Post-Examining Forecast Distribution
The projection methodology remains unchanged</p>
<h4>When X observations are available, predict Z.</h4>
<p>Calculates post-femination&#36;\pi(\theta|{x})&#36;  We calculate our projected distribution using a posteriori density.
&#36;&#36;m(z\mid\mathbf{x})=\int_{\Theta}g(z\mid\theta)\pi(\theta\mid\mathbf{x})d\theta &#36;&#36;Called the Post-Examining Forecast Distribution
The projection methodology remains unchanged</p>
<h3>Bates Hypothesis and Model Selection</h3>
<h4>Multi-scenario Bayesian hypothetical test</h4>
<p>The previous hypothetical test is limited to the link between the original and alternative assumptions <a href="/en/blog/2023/10/28/bayesian-statistics-posterior-distributions-notes/">Bayesian Statistics 1 (Beyes Statistics and Post-Assessment Distribution)</a> The Beyes Hypothetical Test and Model Choice section, however, an important advantage of the Beyes Hypothetical Test is that it is very easy to extend to multiple scenarios;</p>
<p>We just need to calculate the back-probability ratio or the Beyers factor between the multiple assumptions, and we can only decide whether we accept the original assumption according to its size;</p>
<p>And as for the size of the Beyers factor, the relationship to the assumptions on the model support molecule, Jeffeys made some suggestions.</p>
<table>
<thead>
<tr>
<th>Bayesian</th>
<th>Interpretation</th>
</tr>
</thead>
<tbody><tr>
<td>&#36;B&lt;1&#36;</td>
<td>Negative molecular assumptions</td>
</tr>
<tr>
<td>&#36;1&lt;B&lt;3&#36;</td>
<td>Insufficient evidence of hypothetical evidence on the supporting molecule</td>
</tr>
<tr>
<td>&#36;3&lt;B&lt;10&#36;</td>
<td>Strong support</td>
</tr>
<tr>
<td>&#36;10&lt;B&lt;30&#36;</td>
<td>Strong support.</td>
</tr>
<tr>
<td>&#36;30&lt;B&lt;100&#36;</td>
<td>Very strong support.</td>
</tr>
<tr>
<td>&#36;100&lt;B&#36;</td>
<td>I'm sure you'll support it.</td>
</tr>
</tbody></table>
<h4>Assessment of the Bayesian model</h4>
<h5>Importance of the Bayesian model evaluation</h5>
<p>Both the Beyers statistical extrapolation and decision-making are dependent on the later distribution; therefore the results of the extrapolation study are dependent on the quality of the later distribution; it is therefore important to evaluate our Beyers model; the usual Bayes model evaluation method includes not only the AIC BIC guidelines introduced from classical statistics, but also the BPIC.</p>
<h5>AIC and BIC</h5>
<p>They're all based on the principle of great apparition, the MLE estimate.<a href="/en/blog/2023/03/18/mathematical-statistics-notes/">Mathematical statistics</a> The "Big Appearances" section.
Build</p>
<p>The AIC Guidelines are in the form of
&#36;AIC=2\f\lft(x)\widinghat(theta)}<em>+2p, &#36;
Where's the \\wideehat(theta)?</em>{MLE}is the largest estimate of \theta \left (MLE\right)&#36; &#36;P&#36; is the dimension of the estimation parameter</p>
<p>The BIC guidelines take the form of:
&#36;&#36;BIC=-2\ln f\left(x_{n}|\widehat{\theta}_{MLE}\right)+p\ln n&#36;&#36;
We're both aiming to be small.</p>
<h5>BPIC BPIC BEC BEC BEC BIENZ BEYES</h5>
<p>Consider the following two assumptions: (a) Parameter model &#36;f(x|\theta)&#36; It contains a real model. &#36;g(x)=f(x;\theta_0)&#36; &#36;\theta_0\in\Theta&#36;, and the specified model is not far from the real model; (b) the logarithmic a priori is &#36;\ln\pi(\theta)=O_p(1).&#36; Ando (2007) presents the Bayesian forecast information guidelines (the Bayesian forecast profile, BPIC) under the two above-mentioned assumptions and certain normal conditions.
&#36;&#36;
BPIC=-2\int_{\Theta}\ln f\left(x_{n}|\theta\right)\pi\left(\theta|x_{n}\right)d\theta+2p,
&#36;&#36;</p>
<p>Because the logarithmic posterior averages are not usually analyzed, we usually approach the MC method.
&#36;&#36;\int_{\Theta}\ln f\left(x_{n}|\theta\right)\pi\left(\theta|x_{n}\right)d\theta\approx\frac{1}{L}\sum_{j=1}^{L}\ln f\left(x_{n}|\theta^{\left(j\right)}\right),&#36;&#36;
<strong>This is a less a priori.</strong></p>
<h5>DIC Code for Distortion Information</h5>
<p>You!&#36;D\left(\theta\right)=-2\ln f\left(x_{n}|\theta\right)&#36;It's a measure of the usual model deviation. Spiegelhalter et al. (2002) points to a similar backsight. &#36;\bar{D}=E[D(\theta)|x_n]&#36;The higher the model's data is, the higher the model's data is, the higher the model's data is, the higher the model's data is, the more the model's data is, the more the model's data is, the more the model is, the more the data is, the more the data is, the more the data is, the more the data is, the more the data is, the more the data is, the more the data is, the more the data is, the more the data is, the more the data is, the more the data is, the more the data is, the more the data is, the more the data is, the more the data is, the more the data is, the more the data is, the more the data is, the more the data is, the more the data is, the more the data is, the more the data is, the more the data are, the more the data are, the more the data are, the more the data are the data are the data are the data, the more the data are the data are the more the data are the data are the data that are the data are the data are the data.&#36;\bar{D}&#36; The smaller the number of valid parameters is defined below to characterize the complexity of the model:
I'm sorry.
P D=overline{D}-D(overline{theta}<em>{n})=2\ln f(\boldsymbol{x}</em>{n}|\overline{\boldsymbol{\theta&#125;&#125;<em>{n})-2\int</em>{\Theta}\ln f(\boldsymbol{x}<em>{n}|\boldsymbol{\theta})\pi(\boldsymbol{\theta}|\boldsymbol{x}</em>{n})\mathrm{d}\boldsymbol{\theta},
&#36;&#36;</p>
<p>of which&#36;\vec{\theta}_n&#36; Defines the deviation information guidelines for the later average. Spiegelhalter et al. (2002) (Deviance Information profile, DIC) as</p>
<p>&#36;&#36;
DIC=\overline{D}+p_{D}=-2\int_{\Theta}\ln f\left(x_{n}|\theta\right)\pi\left(\theta|x_{n}\right)\mathrm{d}\theta+p_{D},\left(4.6.13\right)
&#36;&#36;
The first one. &#36;\tilde{D}&#36; It can be explained as a measure of the degree of model alignment, the smaller the better; second &#36;p_D&#36; Considered a measure of the complexity of the model, as defined above &#36;DIC&#36; Could be rewritten to &#36;DIC=D\left(overline {\theta}<em>{n}\right)+&#36; &#36;2p</em>{D}=-2\ln f\left(\boldsymbol{x}<em>{n}|\overline{\boldsymbol{\theta&#125;&#125;</em>{n}\right)+2p_{D}&#36;,其中&#36;\overline{\boldsymbol{\theta&#125;&#125;<em>{n}&#36; 为后验均值，从形式上看，它与 AIC 很相似，因此可以认为 &#36;DIC&#36; 是 &#36;AIC=D(\widehat{\theta}</em>{MLE})+2p&#36; 的一个推广，此处 &#36;\widehat{\theta}<em>{MLE}&#36; 为&#36;\theta&#36; 的最大似然估计，对非分层模型而言，当&#36;n&#36; 充分大时有 &#36;p\approx p_D,\widehat{\theta}</em>♪ And I'm gonna be a big fan of the world ♪</p>
<p><strong>DEC can easily calculate the results by using the MCMC method. So the DIC guidelines are used for a variety of beyers model selection problems.</strong></p>
<h2>Statistical decision-making</h2>
<h3>Introduction</h3>
<p>In addition to the three elements of statistical decision-making: sample space and distributed family space loss function, the Beyers statistical decision-making introduced the fourth element of pre-spect distribution function based on statistical decision-making.&#36;F(\theta)&#36; </p>
<p>The Beyers statistical decision-making introduced a fourth element based on the Beyers statistical extrapolation.&#36;L&#36;</p>
<p>Here we make a deal:
If data for decision-making are not randomly influenced, it is called a general decision-making issue (all defined decisions fall into this category).
On the contrary, if you get random effects, it's called statistical decision-making.</p>
<h3>Minimum principle of post-risk</h3>
<p>The principle of the lowest risk is as important for statistical decision-making as the probability of post-mortem is for statistical inference.</p>
<h4>Definition of post-examining risk</h4>
<p>We call the loss function a posteriori risk function.
&#36;&#36;R(\delta(x)|x)=E^{\boldsymbol{\theta}|x}[L(\theta,\delta(x))]&#36;&#36;
&#36;&#36;\left.=\left{\begin{array}{l}\int_\Theta L(\theta,\delta(x))\pi(\theta|x)\mathrm{d}\theta,\\sum_iL(\theta_i,\delta(x))\pi(\theta_i|x),\end{array}\right.\right.&#36;&#36;
He and Bayes expected to lose in the same vein, a probabilities using a priori and a probabilities using a posteriori.</p>
<p>If the decision function is to minimize the risk of post-risk, we call it the best Beyers decision-making function under the Minimal Risk Standard.</p>
<h4>Relationship between post-risk and Bayesian risk</h4>
<p>We know that in the Bayesian statistical inferences.
&#36;&#36;f(x,\theta)=f(x|\theta)\pi(\theta)=\pi(\theta|x)m(x)&#36;&#36;
<strong>This is the late distribution formula migration.</strong></p>
<p>Use it to transform the Beyers risk.
&#36;&#36;R_{\pi}(\delta(x))=E^{\theta}\bigl[R(\theta,\delta(x))\bigr]=E^{X}\bigl[R(\delta(x)|x)\bigr]&#36;&#36;
The core of the Beyers solution, which is used to calculate the Beyers solution, is expressed in two equals. Pattern</p>
<p><strong>One is to calculate the risk function and then use a priori probability density.&#36;\pi(\theta)&#36;Average</strong>
<strong>The other one is to calculate the risk and then use the edge distribution.&#36;m(x)&#36;Average</strong> </p>
<p>It proves that
&#36;&#36;00\
R (\delta)&amp; =E^{\theta}[R(\theta,\delta(x))]  \
&amp;=\int_{\Theta}R(\theta,\delta(x))\pi(\theta)d\theta  \
&amp;=\int_{\Theta}\int_{\chi}L(\theta,\delta(x))f(x\mid\theta)\pi(\theta)dxd\theta  \
&amp;=\int_{\chi}\biggl[\int_{\Theta}L(\theta,\delta(x))\pi(\theta\mid x)d\theta\biggr]m(x)d\mathbf{x} \
&amp;=E^{\mathrm{x&#125;&#125;\Big[R(\delta(x)|x)\Big],
\end{aligned}&#36;&#36;</p>
<h4>Minimum principle of post-risk</h4>
<p>We will prove:<strong>The decision-making function under the principle of minimal risk is Bayes. Break</strong>
Theorem: There exists a non-random decision-making function &#36;\delta_{\pi}(x)&#36;,Fulfilment of Conditions
I'm sorry.
R (\) =pectorname* *inf*<em>{\delta}R(\delta(x)|x)=\operatorname*{inf}</em>I'm not a real guy.
I'm sorry.
then &#36;\delta_\pi(x)&#36; A priori distribution &#36;\pi(\theta)&#36; The Bhaius solution. &#36;\pi(\mathrm{d}\theta|x)=\pi(\theta|x)\mathrm{d}\theta.&#36;</p>
<p> If&#36;\pi(\theta)&#36;We've got a broad beyers solution, but the a priori formula doesn't change.
 <strong>We understand the complexity of the concept.</strong></p>
<p> Proofs that:
&#36;&#36;\begin{aligned}R(\delta(x)|)&amp;=\int_\Theta L(\theta,\delta(x))\pi(\mathrm{d}\theta|x)\&amp;\geqslant\int_\Theta L(\theta,\delta_\pi)\pi(\mathrm{d}\theta|x)=R(\delta_\pi(x)|x),\end{aligned}&#36;&#36;
两边同时对边缘分布&#36;m(x)&#36;做积分有
&#36;&#36;\begin{aligned}
R_{\pi}(\delta(x))&amp; =\int_{\mathcal{X&#125;&#125;R(\delta(x)\mid x)m(x)\mathrm{d}x  \
&amp;\geqslant\int {\mmathcal{x)m(x)\mathrm{d}x=R\pi(\delta \pi(x)).
I'm sorry, I'm sorry.
It proves that the lowest risk of post-examining is Bayesian, the least risk of the post-examining is Bayesian. Break</p>
<h4>A simple example.</h4>
<p>Set&#36;\theta&#36;The a priori distribution is &#36;\pi(\theta_1)=0.6~\pi(\theta_2)=0.4&#36; Set Random Variables&#36;X&#36;Take 0-1 2 values&#36;p(i|\theta_j)=P(X=i|\theta=\theta_j)&#36;  Yes.&#36;X&#36;is the probability distribution
&#36;&#36;p(1|\theta_1)=0.1,\quad p(1|\theta_2)=0.2,\quad p(0|\theta_1)=0.9,\quad p(0|\theta_2)=0.8.&#36;&#36;
Calculating the probability of a posteriori if we know the loss function is&#36;L&#36; Calculating Post-Aspect Risk</p>
<p>The post-exposure of discretes is always more round, not as much as a continuum, but essentially a formula for calculation.
&#36;&#36;\pi(\theta_i|x)=\frac{f(x|\theta_i)\pi(\theta_i)}{\sum_if(x|\theta_i)\pi(\theta_i)}\quad(i=1,2,\cdots).&#36;&#36;
According to the theory, we're going to calculate the probability of a back check in two samples.&#36;X=0&#36; And the other one is...&#36;X=1&#36;
<strong>When we calculate the edge density, don't replace it with specifics.&#36;\theta&#36;, 'cause I'm gonna take all of it</strong></p>
<p>The risk of post-loss is the loss function's after-feed density points, both in profit and loss matrix and in continuous functions.</p>
<h3>Bates estimate under the general loss function</h3>
<p>We use the decision-making method to consider the question of the beyers point estimates in statistical extrapolation.</p>
<p>And the last beyers that you get is the estimate of the parameters that you want to be asked for in the question.</p>
<h4>Bayesian estimate under the square loss function</h4>
<p>If the loss function is used
&#36;&#36;L(\theta,\delta)=(\delta-\theta)^{2}&#36;&#36;
Then we know.&#36;\theta&#36;The Bates estimate is the posteriori average, which is
&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;......&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;......&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;...&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;...&#36;&#36;&#36;...&#36;&#36;......&#36;&#36;&#36;.........&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;...........................&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;<em>B}(x)=E(\theta|x)&#36;&#36;
证明如下：
&#36;&#36;\begin{gathered}
R(a|\boldsymbol{x}) =E[(\theta-a)^2|x]=\int</em>\Theta(\theta-a)^2\pi(\theta|\boldsymbol{x})\mathrm{d}\theta  \
=\int_{\Theta}(\theta^{2}-2a\theta+a^{2})\pi(\theta|\boldsymbol{x})\mathrm{d}\theta.
\end{gathered}&#36;&#36;
我们需要找到合适的&#36;a&#36; 让后验风险最小 对&#36;a&#36;求偏导有
&#36;&#36;\mathrm{\mathrmbl}{\d}theta+2a=&#36;0.00=
And so...&#36;a&#36; Minimise when equal to the posteriori average </p>
<p><strong>Right there.&#36;\Theta&#36;The upper-to-prevalence density function, the after-test density function is all one, which is determined by the nature of the density function.</strong></p>
<p>Form of the weighted square loss function
If the loss function is used
&#36;&#36;L(\theta,\delta)=w(\theta)(\delta-\theta)^{2}&#36;&#36;
Then we know.&#36;\theta&#36;The Bayesian is probably...
&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;......&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;......&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;...&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;...&#36;&#36;&#36;...&#36;&#36;......&#36;&#36;&#36;.........&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;...........................&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;<em>B}(x)=\frac{E[w(\theta)\theta|x]}{E[w(\theta)|x]}&#36;&#36;
他的计算式为 也就是对后验密度求期望
&#36;&#36;\frac{\int_a\theta w(\theta)\pi(\theta\mid x)\mathrm{d}\theta}{\int_aw(\theta)\pi(\theta\mid x)\mathrm{d}\theta}&#36;&#36;
这个期望的计算还是比较复杂的 一般是构造新的分布积分为1实现
证明如下
&#36;&#36;\begin{aligned}
R(a|\boldsymbol{x})&amp; =E\bigl[w(\theta)(\theta-a)^2|\boldsymbol{x}\bigr]  \
&amp;=\int</em>{\boldsymbol{\Theta&#125;&#125;\left[\theta^2w(\theta)-2a\theta w(\theta)+a^2w(\theta)\right]\pi(\theta|\boldsymbol{x})\mathrm{d}\theta.
\end{aligned}&#36;&#36;
求偏导有
&#36;&#36;\mathrm{\mathrm{d}a}ta\bardsymbol=)=mathrm=d}
The equation will lead to the conclusion.</p>
<p>When the parameter vector is multiple &#36;\theta^{\prime}=(\theta_1,\cdotp\cdotp\cdotp,\theta_k)&#36; For Multiple Diplesity Loss Functions
&#36;&#36;L(\theta,\delta)=(\delta-\theta)^{\prime}Q(\delta-\theta)&#36;&#36;
Bates estimates the post-mortem average.
&#36;&#36;\delta_B(x)=E(\theta\mid x)=\begin{pmatrix}E(\theta_1\mid x)\\vdots\E(\theta_k\mid x)\end{pmatrix}&#36;&#36;</p>
<h4>Bates estimate under the linear loss function</h4>
<p>We take the loss function as linear.
&#36;&#36;L(\theta,\delta)=\begin{cases}k 0(theta-\delta),\delta\leq\theta\k 1 (\delta-\theta),\delta&gt;\theta&amp;You're not gonna get away with this?
His Bayes is probably a posteriori distribution.&#36;\pi(\theta|x)&#36; Yes.&#36;\frac{k_0}{k_{0}+k_{1&#125;&#125;&#36;Bits</p>
<p>Special if the loss function is (i.e., the absolute value loss function)
&#36;&#36;{L}(\theta,\delta)=|\theta\text{-}\delta|&#36;&#36;
Bates is estimated to be a posteriori median.</p>
<h3>Practicability and limited operational issues</h3>
<p>In estimating the problem, action is often a matter of choice, but many statistical decision-making issues are only available in a limited number of actions, such as hypothetical testing, and the question of the Bayesian statistical decision-making is very well handled.</p>
<p>Space of operation &#36;A={a_{1},a_{2},...,a_{r&#125;&#125;,&#36;  Losses are as at &#36;L(\theta,a_i)&#36; To find the best, we have to get back to the expected loss. &#36;E^{\theta|\mathbf{x&#125;&#125;[L(\theta,a_i)]&#36; Minimum</p>
<p>Two issues are examined below: two actions (assuming tests) multi-action (classification) issues</p>
<h4>Presumption test issues</h4>
<p>Consider the following hypothetical test issues:
&#36;&#36;H_0:\theta\in\Theta_0\leftrightarrow H_1:\theta\in\Theta_1\quad(\Theta_0\cup\Theta_1=\Theta).&#36;&#36;
Use action&#36;a_0&#36;- To accept the original assumption. - Action.&#36;a_1&#36; To deny the original assumption</p>
<p>Select&#36;0-k_i&#36;The loss function is as follows:
&#36;&#36;\left.L (\theta,a 0)=\left(begin{array}ll}0,&amp;\theta\in\Theta_0,\k_0,&amp;\theta\in\Theta_1,\end{array}\right.\right.&#36;&#36;
&#36;&#36;\left.L(\theta,a_1)=\left{\begin{array}{ll}k_1,&amp;\theta\in\Theta_0,\0,&amp;\theta\allay.\right.\right.&#36;
It's a function of operational space and parameter space, of course.</p>
<p>Post-risk
&#36;&#36;\begin{gathered}
R(a_0\mid x)=E^{\theta\mid x}[L(a_0,\theta)]=\int_{\Theta_1}k_0\pi(\theta\mid x)d\theta=k_0P(\Theta_1\mid x) \
R(a_1\mid x)=E^{\theta\mid x}[L(a_1,\theta)]=\int_{\Theta_0}k_1\pi(\theta\mid x)d\theta=k_1P(\Theta_0\mid x)
\end{gathered}&#36;&#36;
You can just use the back-up risk criterion to determine the best course of action when comparing the size of the back-up risk.</p>
<p>There's a presumption against it.
&#36;&#36;k_0P\left(\Theta_1|x\right)\geqslant k_1P\left(\Theta_0|x\right),&#36;&#36;
Equivalent
&#36;&#36;P\left(\Theta_{1}|x\right)\geqslant\frac{k_{1&#125;&#125;{k_{0}+k_{1&#125;&#125;.&#36;&#36;
This is the rejection of the Beyers hypothetical test in the classic statistics.
&#36;&#36;D=\left{X=\left(X_{1},X_{2},\cdots,X_{n}\right):P\left(\Theta_{1}|X=x\right)\geqslant\frac{k_{1&#125;&#125;{k_{0}+k_{1&#125;&#125;\right},&#36;&#36;
<em>We've seen this form in the seemingly comparable test.</em></p>
<h4>Multi-action issues</h4>
<p>The way we deal with the problem of multiple operations has not changed, and for each operation we give the expression of the loss function independently.</p>
<p><strong>The loss function is based on a relatively reasonable choice.</strong></p>
<p>And then we calculate the risk of post-examining, and compare the size of the post-examining risk to make the final decision, the same idea as we did when we studied the hypothesis test directly in front.
<a href="/en/blog/2023/10/28/bayesian-statistics-posterior-distributions-notes/">Bayesian Statistics 1 (Beyes Statistics and Post-Assessment Distribution)</a> and the “Beyers Presumption and Model Selection” section</p>
<h4>Inter-district estimates in statistical policymaking</h4>
<p>Consider applying statistical decision-making methods to consider the issue of credible inter-temporal or inter-temporalization</p>
<p>At this point, the operational space is the assembly of all possible inter-zone formations.&#36;C(x)=[d_{1}(x),d_{2}(x)]&#36;
Loss function is used to
&#36;&#36;L(\theta,C(x))=m_1[d_2(x)-d_1(x)]+m_2[1-I_{C(x)}(\theta)]&#36;&#36;
Of which&#36;m_1~m_2&#36;It's a constant given in advance.
The first half measures the loss from the length of the zone, the greater the loss.
The second half of the story is that&#36;\theta&#36;Losses from deviations </p>
<p>Compare the risk of post-examining between multiple zones, and find the least risk of post-examining.</p>
<h3>Minimax Guidelines</h3>
<p>Consistent optimal decision-making functions may not exist, or often do not; then we need a new guideline that considers which best from the point of view of risk functions.
<strong>This is what we do when we're not sure about a priori distribution. In other cases, it's better to study the later risk.</strong></p>
<p>Consider risk functions&#36;R(\theta,\delta)&#36;  <a href="/en/blog/2023/10/28/bayesian-statistics-posterior-distributions-notes/">Bayesian Statistics 1 (Beyes Statistics and Post-Assessment Distribution)</a> ..the "Risk Functions and Consistency Best Decision-Making Functions" section
&#36;&#36;M(\delta)=\sup_{\theta\in\Theta}R(\theta,\delta).&#36;&#36;
And we look at the most risks in a decision-making situation, and we choose the least risk-taking decision-making.
This decision-making rule we call...<strong>Minimax Guidelines</strong> </p>
<p><strong>The Minimax Code is a mind that doesn't demand much, but that doesn't want much to be lost.</strong> </p>
<p>It's more difficult to calculate the Minimax solution.
Set &#36;\widiehat{g} k=\widiehat{g}<em>k(\boldsymbol{x})&#36; 为在先验分布 &#36;\pi_k(\theta)&#36; 下 &#36;g(\theta)&#36; 的一列贝叶斯估 计，&#36;k=1,2,\cdots;&#36; 假定 &#36;\widehat{g}<em>k&#36; 的贝叶斯风险为 &#36;r_k,k=1,2,\cdots&#36;, 且有&#36;\lim</em>{k\to\infty}r</em>{k}=r&lt;\infty&#36;,<br>Set &#36;\widiehat{g}<em>}=\widehat{g}^{</em>}(x)&#36; 为 &#36;g (\theta) &#36;&#36;1 = an estimate, condition fulfilled
&#36;M\left (\wideehat{g}<em>{\cHFFE7C5}right, {\cHFFE7C5}
^</em>Minimax estimates for decision-making</p>
<h2>Bayesian statistical calculations</h2>
<h3>Introduction</h3>
<p>(a) The expectations, differences, fractions or numbers of post-square distributions are often calculated in the Bayesian statistical methodology;
For example, the usual post-mortem average, which is estimated by Beyers at the square loss, is measured by the difference between the post-test examination ... the post-censorship number, the post-test median and the post-test fractional number are also often used as a factor in the Yates estimation or in the establishment of a Beneath trustable zone;
If the a priori distribution is not its a priori distribution (which is often encountered in many cases), then the later distribution is often no longer a standard distribution. Thus, the numerical characteristics of the posteriori distribution that need to be calculated are often not expressed in a dominant way, which requires some special methods of calculation.</p>
<p>Like what?
&#36;&#36;\pi(\theta|x)\propto\exp{-(\theta-x)^{2}/(2\sigma^{2})}[\tau^{2}+(\theta-\mu)^{2}]^{-1}.&#36;&#36;
His posteriori expectations and differences are complex, undissolved points.
We can also solve it with some numerical weights.</p>
<p>Think about another question.
Aforecast distribution given with logarithmic distribution
&#36;&#36;\nu=\left(\ln\theta_{1},\ln\theta_{2},\cdots,\ln\theta_{k}\right)^{T}\sim N\left(\mu1_{k},\tau^{2}\left{\left(1-\rho\right)I_{k}+\rho J_{k}\right}\right)&#36;&#36;
So you can give the later distribution to
&#36;&#36;00\&amp;\pi\left(\nu|x\right)\propto f\left(x|\nu\right)\pi\left(\nu\right)\propto g\left(\nu|x\right)\&amp;=xp\left{-\sum l\l\left&#125;&#125;l}l}l}l}l}l l l l mu l\l^l^l\l^l^l\l}l}l}l}l}l}l}l}l}l}l}l}l}l}l}l}l}l}l}l}l}l}l}l}l}l}l}l}l}l}l}l}l\l\l}l\l\l\l\l\l}l}l}l}l}l}l}l}l}l}l}l}l}l}l\l\l}l}l}l}l}l}l}l}l}l\l\l}l}l}l}l}l}l}l}l}l}l}l}l}l}l}l}l}l}lignlignlignlignlignlignlignlignlignlignlignlignlignlignl I'm sorry.
His expectations are two.&#36;k&#36;The value of the re-scoring; the value-scoring method is not helping the high-dimensional sub-coordinate (as it relates to the high-dimensional disaster) and now we need some new treatment techniques to solve this problem; of which MCMC algorithms are the most central instrument.</p>
<h3>Em algorithm</h3>
<p>The EM algorithm is a very important kind of statistical algorithm that he uses to solve two very important statistical computational problems, one of which is a very similar estimation, and the other of the late-calculations in Bayes statistics, which is in fact a special case of the latter, and we focus on the EM algorithm from the perspective of Bayes statistics.
EEM algorithms are an extended algorithm (data addition algorithm) because it is very difficult to calculate (extensive) the number of late-calculations directly, but rather to expand some of the original data by adding some potential data, so that a series of large or simulated can be simply achieved.
These potential data may be missing data or unknown parameters </p>
<p><strong>The estimate of the post-mortem number is the main use of our EEM algorithm.</strong></p>
<h4>Algorithm Process</h4>
<p>We're still going to be able to get a later analysis of the distribution.&#36;p(\theta|Y)&#36; We'll expand into a variable.&#36;Z&#36; Getting easy&#36;{p(\theta|Z,Y)}&#36; So we can simplify the computation process and then we can add it to the list.&#36;Z&#36;And then we'll make it work. </p>
<p>Em algorithm is an iterative algorithm divided into E step and M step.&#36;p(\theta|Y)&#36; The post-censorship distribution that means that needs to be studied &#36;p(\theta|Z,Y)&#36;To Add Post-Examining Distribution&#36;p(Z|\theta,Y)&#36; To expand the condition density function of the variable, our goal is to study the number of numbers that are distributed after the results.
Remember&#36;\theta^{i}&#36;It's the first.&#36;i+1&#36;And the next step in the iterative process is to get to the bottom of the count.
Step E: Expectation.
Yeah.&#36;{p(\theta|Z,Y)}&#36; Or...&#36;log({p(\theta|Z,Y)})&#36;  About&#36;Z&#36;The conditions are based on expectations.&#36;Z&#36; I'm gonna score it.
&#36;&#36;00\
Q (\theta|theta^(i), Y)&amp; \hat{=}E_{Z}\big[\log p(\theta|Z,Y)|\theta^{(i)},Y\big]  \
&amp;=\int\big[\log p(\theta|Y,Z)\big]p(Z|\theta^{(i)},Y)dZ.
\end{aligned}&#36;&#36;
<strong>Here's the expectation of an expanded variable, and we're using an estimate of the number of parameters that are being iterative.</strong>
Step M, it's huge.
- Put it on.&#36;Q(\theta|\theta^{(i)},Y)&#36; Make it great. Get some.&#36;\theta^{i+1}&#36; Make
&#36;&#36;Q(\theta^{(i+1)}|\theta^{(i)},Y)=\max Q(\theta|\theta^{(i)},Y).&#36;&#36;
Repeat this EEM process until we've been in it.&#36;\theta&#36;Sequence collapse
<strong>It's huge. By looking for the right thing.&#36;\theta&#36; Achieved&#36;Q&#36;♪ The great</strong></p>
<h4>Theoretically,</h4>
<p>Theoretically:
If&#36;f(x)&#36;It's a cam function.&#36;&#36;E[f(X)]\leq f[E(X)]&#36;&#36;
It's called Jensen's Instinct.</p>
<p>Theoretically:
EEM algorithms can increase the value of a post-density function at every inch.
&#36;&#36;{p(\theta^{(i+1)}|Y)\geq p(\theta^{(i)}|Y)}&#36;&#36;
Theoretically:
If Em's algorithm is in the middle of&#36;\theta&#36;Sequence Satisfaction</p>
<ul>
<li>&#36;\left.\frac{\partial Q(\theta|\theta^{(i)},Y)}{\partial\theta}\right|_{\theta=\theta^{(i+1)&#125;&#125;=0;&#36;</li>
<li>&#36;\text{d}p(Z\theta^(i), Y\text{softly smooth,}\theta^(i)}\text{deep to certain values}\theta^()<em>♪ I'm not gonna let you go ♪
then
&#36;&#36;\partial\log p(\theta|)</em>=&#36;0.00
The EM algorithm must have constricted to a level of stability, but it may not be a maximum value point, but if you want to keep the maximum, you need to select multiple starting values to model stability over time.</li>
</ul>
<h4>Example of an EM algorithm</h4>
<h5>The missing data is a very significant estimate.</h5>
<p>For the general&#36;X\sim N(\mu,\sigma^{2})&#36; &#36;X_{1},X_{2},X_{3}&#36; It's a sample from the whole population. &#36;X_{2}&#36;Missing Parameters for determining the overall distribution using a very similar estimate
For this type of problem, EEM algorithms can be handled by adding missing data.</p>
<p>Extension&#36;X_{2}&#36; Get full and symmetrical functions
&#36;&#36;\log p(\theta\mid X_1,X_2,X_3)=-3\ln\sigma-\frac{\sum_{i=1}^3(X_i-\mu)^2}{2\sigma^2}.&#36;&#36;
Execute E step, which is directed at&#36;X_{2}&#36; Expectations.
It's obvious.&#36;X_{2}&#36;I'm not expecting anything.&#36;X_{2}&#36; All of them can be considered constants, so we actually just need to calculate a small fraction.
&#36;&#36;E_{X_{2&#125;&#125;[(X_{2}-\mu)^{2}\mid\theta^{(i)},X_{1},X_{3}]=(\mu_{i}-\mu)^{2}+\sigma_{i}^{2}&#36;&#36;
Yeah.&#36;X_{2}&#36;We have the last time we've given an iterative estimate of the parameters. Value&#36;\theta^i&#36; It's so easy to know.&#36;X_{2}&#36;And finally, what we're looking for is the square of the normal distribution, which is a second-order problem, which is easy to calculate.
Then we can get the final results of step E.
&#36;&#36;00\
Q (\theta\mid\theta^, X (), X (3),&amp; \left.\hat{=}\left.E_{X_{2&#125;&#125;\right[\log p(\theta\mid X_{1},X_{2},X_{3})\mid\theta^{(i)},X_{1},X_{3}\right]  \
&amp;=-3\mathrm{~ln}\sigma-\frac{(X_1-\mu)^2+(X_3-\mu)^2+(\mu_i-\mu)^2+\sigma_i^2}{2\sigma^2}.
\end{aligned}&#36;&#36;
M步
找到合适的&#36;\theta&#36;取值 让Q极大 我们只需要研究对&#36;\theta&#36;的偏导数
&#36;&#36;\left.\left[\begin{aligned}&amp;\frac{\partial Q}{\partial\mu}=\frac{(X_1-\mu)+(X_3-\mu)+(\mu_i-\mu)}{\sigma^2}=0,\&amp;\frac(X 1-\m2)^2+ (i-\m2)^sigma 3}=end{aligned}\right.\right.&#36;&#36;
The solution will be the next iterative result.
Be careful. We're in the middle of a...&#36;\theta^i&#36;In&#36;\theta^{i+1}&#36;of which&#36;\theta^i&#36;It was something that had to be given when you studied E step.
The sequences that are quickly reduced by the overlap are the meaning of the EEM algorithm.</p>
<h5>Post-research distribution</h5>
<p>In fact, it's very much like studying the distribution of the number of people after studying the function of the function, and the large amount of the missing item is about the problem of the later distribution, and the large amount of the missing item is about a special case of the later distribution of the population.</p>
<p>Assuming there are four possible outcomes, the probability of each happening is that the two of us will be able to do the same. &#36;\frac{1}{2}+\frac{\theta}{4},\frac{1}{4}(1-\theta),\frac14(1-\theta),\frac\theta4,&#36;  of which&#36;\theta&#36;Yes.&#36;(0,1)&#36;The results of the four results were measured as follows:&#36;Y=(y_{1},y_{2},y_{3},y_{4})=(125,18,20,34).&#36;</p>
<p>Now let's study it.&#36;\theta&#36;And the distribution of the original amount is assumed to be the original.&#36;\pi(\theta)&#36; The aforecast distribution is flat.
&#36;&#36;00\
P\left (\theta\mid Y\right)&amp; \propto\pi(\theta)p(Y\mid\theta)  \
&amp;=\left(\frac{1}{2}+\frac{1}{4}\right)^{y_{1&#125;&#125;\left[\frac{1}{4}(1-\theta)\right]^{y_{2&#125;&#125;\left[\frac{1}{4}(1-\theta)\right]^{y_{3&#125;&#125;\left(\frac{1}{4}\theta\right)^{y_{4&#125;&#125; \
&amp;\infty\left(2+\theta\right)y_1(1-\theta)^{y_2+y_3}\theta^{y_4}.
\end{aligned}&#36;&#36;
这个后验分布众数可不好研究 因此我们假定第一种结果可以分成两部分 概率分别为&#36;\frac{1}{2}&#36;和&#36;\frac{\theta}{4}&#36; 用&#36;Z&#36;和&#36;y_{1}-Z&#36; 表示试验结果落入其中的次数（Z是我们补充的隐藏数据）那么添加后验分布为
&#36;&#36;\begin{aligned}
p(\theta\mid Y,Z)&amp; \propto\pi(\theta)p(Y,Z\mid\theta)  \
&amp;=\left(\frac{1}{2}\right)^{z}\left(\frac{\theta}{4}\right)^{y_{1}-z}\left[\frac{1}{4}(1-\theta)\right]^{y_{2&#125;&#125;\left[\frac{1}{4}(1-\theta)\right]^{y_{3&#125;&#125;\left(\frac{1}{4}\theta\right)^{y_{4&#125;&#125; \
&amp;\infty(\theta)^{y_1-Z+y_4}(1-\theta)^{y_2+y_3}.
\end{aligned}&#36;&#36;
对于这样的添加后验分布 求众数明显就简单了 所以我们使用EM算法继续计算
E步 对添加后验分布的添加量的对数求期望 和添加量Z无关的算常数
&#36;&#36;\begin{aligned}Q(\theta\mid\theta^{(i)},Y)&amp;=E^Z[(y_1-Z+y_4)\mathrm{log}\theta+(y_2+y_3)\mathrm{log}(1-\theta)\mid\theta^{(i)},Y]\&amp;=[y_1-E^Z(Z\mid\theta^{(i)},Y)+y_4]\mathrm{log}\theta+(y_2+y_3)\mathrm{log}(1-\theta).\end{aligned}&#36;&#36;
而&#36;Z&#36;的条件分布很明显是一个二项分布 &#36;Z\sim b\left(y_1,\frac2{\theta^{(i)}+2}\right)&#36;
因此&#36;&#36;= (Z) \ (Z) \ (=), Y\right = \ \ \ \ \ \ }{\ }{\ } } + + + + + + + + } } } + + + + + + + + + + + + + + + + + + + + + + + = \ + + + \ \ \ \ \ \ \ \ } \ } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } }
Now we can get a big Q.
M step for & Q on & Wide&#36;\theta&#36; And then you can do it in an iterative manner.</p>
<h5>Mixed distribution issues</h5>
<p>For Distribution Functions&#36;&#36;f_{X}\left(x\right)=\sum_{j=1}^{K}p_{j}f_{X_{j&#125;&#125;\left(x\right).&#36;&#36;
We have &#36;USum\litits p j=1,p j&gt;0&#36; &#36;f_{X}(x)&#36;是总体密度函数 &#36;f x {j}(x)&#36; is a subtotal density function
And for a problem like this mixed distribution, it's important to solve the parameter estimates.
But using the rectangular estimation method would be very complicated, Pearson, and it's been proven by an attempt.
In follow-up, it was found that using EEM algorithms can easily estimate the parameters of mixed distribution, which is a powerful boost to the study of mixed distribution models. </p>
<h3>Monte Carlo method of crediting</h3>
<p>The Beyers statistics have a lot of goals that are a fraction.
<strong>The MC method is essentially a numerical score method, and he's not effective in the high-dimensional disaster we just mentioned, and he's got a lot of better MC methods.</strong></p>
<h4>Theory Foundation</h4>
<p><strong>Benuli's law of big numbers.</strong>
&#36;&#36;\underset{n\to\infty}{\operatorname*{lim&#125;&#125;P\left{\left.\frac{\mu_{n&#125;&#125;{n}-p\right|&lt;\varepsilon\right}&#36;1.&#36;1
Benuli's law of big numbers tells us that frequency is constricted by probability, which means that the problem of fraction is directly transformed into a problem of proportionality when it's calculated by the size or volume of the measure.
<strong>The Law of the Great Count of Sinchin.</strong>
&#36;&#36;\lim_{n\to\infty}P\left{\left|\frac{1}{n}\sum_{i=1}^{n}X_{i}-\mu\right|&lt;\varepsilon\right}&#36;1.&#36;1
The Sinchin Big Numbers Law tells us that our mathematical expectations for random variables can be measured in the near-image statistical order using sample averages.
Select the appropriate density function to turn the points into the desired calculation problem and then use the sample to solve it.</p>
<h4>Random Pointing Method</h4>
<p>Calculate
&#36;&#36;\theta=\int_{a}^{b}f\left(x\right)\mathrm{d}x.&#36;&#36;
It's turned into a calculator curve.&#36;f(x)&#36;Question of the size of the areas below
In&#36;D=[a,b]\times [0,M]&#36;Center Medium Random Droppoint If it falls on a curve&#36;f(x)&#36;, and then separate the lower of the item.
Finally, statistical random drop points fall on the curve.&#36;f(x)&#36;The probability below the&#36;P&#36;
Getting under the formula below&#36;\theta&#36;Estimated value
&#36;&#36;P=P\langle Z_i\in\Omega\rangle=\frac{S(\Omega)}{S(D)}=\frac{\theta}{M(b-a)}&#36;&#36;
Benuli's law of big numbers ensures that our estimate can increase as the number of drop points increases.</p>
<h4>Average method</h4>
<p>The average method has increased the efficiency of the score estimation
The average method is based on the law of the Sinchin Big Numbers, on the one hand, and on the mathematical expectations of random variable functions, on the other, it is a significant distortion.
When?&#36;X&#36;Subject to the probability density function as&#36;g(x)&#36;. When distribution
&#36;&#36;\theta=\int_{a}^{b}f\left(x\right)\mathrm{d}x=\int_{a}^{b}\frac{f\left(x\right)}{g\left(x\right)}g\left(x\right)\mathrm{d}x=E\Bigl[\frac{f\left(X\right)}{g\left(X\right)}\Bigr].&#36;&#36;
To simplify the problem, we'll introduce a supplementary distribution.&#36;X&#36;Take As&#36;X\sim U(a,b)&#36;<br>So...
&#36;&#36;\theta=(b-a)E[f(X)].&#36;&#36;
Now we've turned the problem into a mathematically desired calculation, and we can easily use statistical simulation methods to generate the corresponding random numbers. And then we'll do the desired calculations.</p>
<p>For infinity curves, the problem can be converted into a limited range using a fractional transformation.
It's the same idea as in the chapter on the study of broad points in mathematical analysis.</p>
<h4>GVM</h4>
<p>The core idea of the Govt point is not changed.
&#36;&#36;P=P\langle Z_{i}\in\Omega\rangle=\frac{V(\Omega)}{V(D)}=\frac{\theta}{MV(C)}=\frac{\theta}{M\prod_{j=1}^{d}\left(b_{j}-a_{j}\right)}&#36;&#36;
The core formula for the high-dimensional mean method is changed to
&#36;&#36;\widetilde{\theta}=\prod_{j=1}^{d}\left(b_{j}-a_{j}\right)\frac{1}{n}\sum_{i=1}^{n}f(x_{i}).&#36;&#36;</p>
<h3>Marcov Chain Monte Carlo (MCMC) methodology</h3>
<p>The MC approach that preceded was not able to deal with the problem of the high-dimensional disaster on the one hand, and the statistical problem of the Beyers, which was not well known in many of the problems, on the other; but the former was more common in the Byces, which we introduced.<strong>Marcov Chain Monte Carlo (MCMC) methodology</strong>Value</p>
<h4>Markov's Grand-Cultural Law.</h4>
<p>Theorem: assumptions &#36;{X_n,n\geqslant0}&#36; As a person with numerical space &#36;S&#36; The marzipan chain, the transfer probability matrix is &#36;P&#36;... further assuming that it is not available and is distributed smoothly&#36;\pi={\pi_i:i\in S}&#36;, and to any boundary function &#36;h:S\to\mathbf{R}&#36; and start value &#36;X_{0}&#36; Any initial distribution of</p>
<p>&#36;&#36;
\frac{1}{n}\sum_{i=0}^{n-1}h(X_{i})\to\sum_{j}h(j)\pi_{j},\quad n\to\infty
&#36;&#36;
Probability. When the space is inexcusable, the chain. &#36;{X_n,n\geqslant0}&#36; It's impossible and it's evenly distributed.&#36;\pi&#36;Sometimes, too.
&#36;&#36;
\frac{1}{n}\sum_{i=0}^{n-1}h(X_{i})\to\int_{S}h(x)\mathrm{d}\pi(x),\quad n\to\infty.
&#36;&#36;</p>
<p>Theorem is very useful, for example, in a given collection. &#36;S&#36; The probability distribution of the thorium, and &#36;s&#36; On-act Functions &#36;h(\theta)&#36;Suppose we're calculating the points. <em>Sh( \theta) d\pi( \theta|x)&#36;, 当从后验分布 &#36;\pi (\theta|x) &#36;&#36; is difficult to sample directly,
And you can build a chain of horses, and you can make it a space. &#36;S&#36; And it's distributed smoothly. &#36;\pi&#36; It's the target's back-situation. &#36;\pi(\cdot|x)&#36;, from a start value &#36;\theta</em>{0}&#36; 出发，将此链运行一段时间，比如 &#36;0,1,2,\cdots,n-1&#36;,生成随机数 (样本) &#36;\theta 0, \theta 1, \cdots, \theta &#36;, as understood by the previous theorem
&#36;overline<em>{n}=\frac{1}{n}\sum</em>Other Organiser
The points that are required &#36;\mu&#36; A compatible estimate, which is called the MCMC method, for the calculation of points</p>
<p><strong>The MCMC will provide us with a series of samples, which are sampled from the end of the target.</strong></p>
<h4>Some of the terms that you're gonna use.</h4>
<h5>Initial value</h5>
<p>It's used to initialize a Marcov chain; if the primary principle is more dense than the number of algorithms; then our final score could be wrong.</p>
<p>To avoid the impact of opening values, we suggest that we...</p>
<ul>
<li>Drop some of the first-in-a-time samples.</li>
<li>From multiple openings</li>
</ul>
<p>It could be considered as a starting value based on a priori expectations or numbers if the a priori information is sufficient</p>
<h5>Brush in pre-burn</h5>
<p>We mentioned earlier that we were going to drop some of the first iterative samples and then record them when they're in a state of calm, and this part of the iterative that was removed is called preburning, and that removal of preburning does not theoretically affect our results if the chain is running long enough.</p>
<h5>Sampling lag</h5>
<p>The samples from the Ma's chain cannot be completely independent, but we need to be independent; we can find the right space by watching the ACF map, and we can make sure the samples are almost independent.</p>
<h5>Number of its constants</h5>
<p>Difference between total and pre-burner iterative</p>
<h5>Algorithms are impregnable.</h5>
<p>The Marcov chain is in a state of calm, and the samples after the stables can be approximated as the samples in the posterioris.</p>
<h5>Monte Carlo error.</h5>
<p>Report if our random simulations are approaching a smooth distribution.</p>
<h4>Condensation diagnosis</h4>
<p>There is no single indicator that can help us study the MCMC method's robustness.</p>
<ul>
<li>MC error means a contraction.</li>
<li>The sample road map is not in a defined trend in one area.</li>
<li>Cumulative average stable</li>
<li>ACF Chart</li>
<li>Some diagnostic methods, such as the Gelman-Rubin diagnosis.</li>
</ul>
<h3>Metropolis - Hasting Algorithm</h3>
<p>The core of his sampling from the general posteriori distribution is the use of the MCMC algorithm, which is to create a chain of marzies that meets a set of predefined conditions; so the most central is the rules of how to move in each state.</p>
<p>Metropolis - Hasting algorithm is one of the most classic algorithms.</p>
<h3>Gibbs algorithm</h3>
<p>Promotion of Metropolis - Hasting algorithms to high-dimensional sampling</p>
