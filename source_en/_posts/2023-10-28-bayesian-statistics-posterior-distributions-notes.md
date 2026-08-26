---
title: 'Bayesian Statistics: Posterior Distributions'
title_zh: 贝叶斯统计：后验分布
date: 2023-10-28 17:36:48 +0800
categories:
- Data Science
- Probability & Statistical Foundations
tags:
- Statistics
- Bayesian Statistics
author: Hyacehila
mathjax: true
hidden: true
excerpt: Covers Bayes formula, priors, posteriors, marginal distributions, conjugate priors, and Bayesian computation basics.
description: Covers Bayes formula, priors, posteriors, marginal distributions, conjugate priors, and Bayesian computation
  basics.
excerpt_zh: 整理 Bayes 公式、先验分布、后验分布、边缘分布、共轭先验和贝叶斯计算基础。
permalink: /blog/2023/10/28/bayesian-statistics-posterior-distributions-notes/
lang: en
translation_key: 2023-10-28-bayesian-statistics-posterior-distributions-notes
translation_status: machine
translation_source_hash: 151b86def54302132076cadc7c8a93dcbac38ee9724738a3e62bb87235a67e14
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<h2>Introduction</h2>
<h3>Introduction</h3>
<h4>Bayes Formula</h4>
<p>We learned the full probability formula in probabilistic theory.
&#36;&#36;P(A)=P\bigg(\sum_{i=1}^{n}AB_{i}\bigg)=\sum_{i=1}^{n}P(A|B_{i})P(B_{i}).&#36;&#36;</p>
<p>And on that basis, the Bayes formula of probabilism was introduced.
&#36;&#36;P(B_i|A)=\frac{P(A|B_i)P(B_i)}{P(A)}=\frac{P(A|B_i)P(B_i)}{\sum_{j=1}^nP(A|B_j)P(B_j)}.&#36;&#36;</p>
<p>This formula was a general inference, but Bayes' research gave it a deeper thought.</p>
<p>We just watch.&#36;P(B_{1}),\cdots,P(B_{n}),&#36; He just didn't have any further information.&#36;A&#36;♪ The time people are having ♪&#36;B&#36;I know, but when we have the incident,&#36;A&#36;When we get the information, we're gonna have to go through the incident.&#36;B&#36;I've got an update on this.&#36;P(B_1|A),\cdots,P(B_n|A)&#36;</p>
<p>If we see A as the result of an event, then the full probability formula is the result of the event, and the Bayes formula is the opposite of what he did.&#36;A&#36; To speculate about the probability of a state's causes. </p>
<p>In fact, this is a very common issue in modern statistics. Down
A reagent for the diagnosis of a cancer, which has been recorded in clinical trials as follows: The test results for cancer patients are 95% positive, and for non-cancer patients 95% negative. How do you judge whether a person is suffering from cancer when a community with this reagent is surveyed for cancer, with a cancer incidence rate of 0.5% in the community?
At this point, the reason for the disease is the positive indicator is the result.
&#36;&#36;P(B_1|A)=0.087&#36;&#36;
The odds are...&#36;0.913&#36; Based on the probability, this patient should not be sick.</p>
<h4>Three messages</h4>
<p>For the whole sample we're doing, he has his own distribution and distribution parameters, and the information about it is called the aggregate information.</p>
<p>We can get sample information for samples taken from the population.</p>
<p>The whole information and the sample information together get the sample information (sampling information)</p>
<p>The theoretical method of statistical extrapolation based on aggregate and sample information is called classic or frequency statistics.</p>
<p>Another information is pre-information, which is that before we sampled, we had a certain understanding of the statistical inferences that we wanted to understand, which often come from experience and historical information, and can be used by us to extrapolate statistics. Medium</p>
<p>We've been able to get back-up information by correcting our a priori information with sample information.</p>
<p>This statistical school using a priori information is Bayes Statistics. Learn.</p>
<h4>History</h4>
<p>Now let's look back at the history of Bayes statistics.</p>
<p>Formally, Bayes is a theory of a full probability formula, but Bayes finds the idea of inference that is embedded in it and publishes it, and later scholars have evolved to develop it into a systematic theory and methodology of statistical inference, called the Bayes Method. </p>
<p>These methods form Bayes Statistics, and the scholars who support Bayes Statistics form Bayes School of Math Statistics.</p>
<p>The idea of the Bayes school is that the difference between the two is that the two are very different.
<strong>Bayes schools think that the parameters to be estimated are random variables, while frequency schools think they are a certain number, and the differences that follow are all due to this.</strong></p>
<h4>The debate between two schools of statistics</h4>
<p>Frequency schools (classical schools) and Bayesian schools are the two largest schools of statistics today.
The frequency of probabilities is the same as the frequency of scholars who insist on researching through a lot of repetitions.
All scholars who insist on the meaning of information are of Bayesian origin.
So, the debate between them has not yet been resolved. Points</p>
<h5>The Bates criticism of the frequency parties.</h5>
<p>They think that the determination of a priori information is particularly problematic when the probability is different from the one in humans and the frequency of probability is at odds with the scientific lack of objective scientific value.</p>
<p>Besides, Bayes, the school also uses sample distribution as its starting point, which is the frequency of probability.</p>
<p>The Bayes school response is the following.</p>
<ul>
<li>Subjective probabilities are common, custom-based, understandable.</li>
<li>Statistical inferences and decision-making are themselves consequences for actors, and the emphasis on objectivity is meaningless when people have different levels of perception and natural trade-offs.</li>
<li>A lot of frequency schools have an inference that the Bayes solution is unique.</li>
</ul>
<h5>The Beyers School of Frequency Criticism.</h5>
<ul>
<li>A lot of tests can't be repeated.</li>
<li>It's not reasonable to assume that the medium accuracy of the test was determined in advance and not related to the sample.</li>
</ul>
<h3>Basic concepts of the Bayesian statistical extrapolation</h3>
<p>The difference between Bayesian and classic statistics is the use of a priori information on parameters.
From the point of view of the Bayesian schools of statistics,<strong>All statistical assumptions must be based on a posteriori distribution.</strong></p>
<h4>A prior and a post-specify distribution</h4>
<p>Any probability distribution in parameter space is a prior distribution</p>
<p>We use it all the time.&#36;\pi(\theta)&#36; This is the a priori distribution, the density function or the distribution column. </p>
<p>How to determine a prior distribution is presented later. </p>
<p>As the sample went on, we got the sample.&#36;X&#36; We'll adjust our numbers to the sample.&#36;\theta&#36; And the idea is that you can get a posterior distribution, which is the general information, sample information, pre-check information.</p>
<p>Define the posteriori distribution as
&#36;&#36;\pi(\theta|x)=\frac{h(x,\theta)}{m(x)}=\frac{f(x|\theta)\pi(\theta)}{\int_{\Theta}f(x|\theta)\pi(\theta)\mathrm{d}\theta}&#36;&#36;
Of which&#36;m(x)&#36; Called the edge distribution in the form of
&#36;&#36;m(x)=\int_\Theta h(x,\theta)\mathrm{d}\theta=\int_\Theta f(x|\theta)\pi(\theta)\mathrm{d}\theta &#36;&#36;
In case of separation, you can use the distribution column as a symbol
&#36;&#36;\pi(\theta_i|x)=\frac{f(x|\theta_i)\pi(\theta_i)}{\sum_if(x|\theta_i)\pi(\theta_i)}\quad(i=1,2,\cdots).&#36;&#36;</p>
<p>In fact, the back distribution of the separation is the Bayes formula, which means that our back distribution is essentially a promotion of the Bayes formula.</p>
<p>In here. &#36;\pi(\theta)&#36; A priori information &#36;f(x|\theta)&#36; It's sample information. It's sample information and general information.
&#36;\pi(\theta|x)&#36; The whole equation is a replica of the underlying Bayes formula.</p>
<p>The formulae for the individual samples are described earlier, and when we take multiple samples, we explain them in detail when we present statistical estimates later.</p>
<h4>Point estimate</h4>
<p>From the point of view of the Bayesian schools of statistics,<strong>All statistical assumptions must be based on a posteriori distribution.</strong>Our Bayesian point is probably the same thing.</p>
<p>Definition: make post-feasibility &#36;\pi(\theta|x)&#36; Maximum value reached<em>{MD}&#36; 称为&#36;\theta&#36; 的后验众数(Mode)估计； 后验分布的中位数 &#36;\hat{\theta}</em>{_{Me&#125;&#125;&#36;称为 &#36;\theta&#36;的后验中位数(Median)估计； 后验分布的期望(Expectation)值 &#36;\hat{\theta}<em>E&#36; 称为 &#36;\theta&#36; 的后验期望值估计，这三个估计都称为贝叶斯估计,记为&#36;\hat\theta</em>{B}&#36;</p>
<ul>
<li>The post-censorial estimates are also known as the maximum post-censorship estimates.</li>
<li>These three estimates are usually different, but when the back-density is symmetrical, the three estimates overlap.</li>
</ul>
<p>Using these three beyers dot estimates to estimate the unknown parameters is the idea of beyers dots dots dots.</p>
<h4>Estimates</h4>
<p>From the point of view of the Bayesian schools of statistics,<strong>All statistical assumptions must be based on a posteriori distribution.</strong>And so is our Bayesian region. </p>
<p>Definition: Credible interval
Parameters&#36;\theta&#36;Post-separation distribution is&#36;\pi(\theta|x)&#36;- For given samples. &#36;x&#36; and probability &#36; 1-\alpha (0)&lt;\alpha&lt;1)&#36;,若存在这样的两个统计量 &#36;\hat{\theta}_L=\hat{\theta}_L(x)&#36; 与&#36;♪ And the world ♪
&#36;&#36;
P(\hat{\theta}_L\leq\theta\leq\hat{\theta}_U\mid x)\geq1-\alpha
&#36;&#36;
And it's called the interval.&#36;\hat{\theta}_L,\hat{\theta}_U&#36;As Parameters&#36;\theta&#36;The level of credibility is 1- &#36;\alpha&#36; Beyes' credible inter-area estimates, or abbreviations&#36;\theta&#36;Yes. &#36;1-\alpha&#36; Credible area</p>
<p>Satisfied&#36;P(\theta\geq\hat{\theta}_L\mid x)\geq1-\alpha&#36; Yes.&#36;\hat{\theta}_L&#36;Called&#36;\theta&#36;Yes. &#36;1-\alpha&#36; Credibility threshold (one side)
Satisfied&#36;P(\theta\leq\hat{\theta}_U\mid x)\geq1-\alpha&#36; Yes.&#36;\hat{\theta}_U&#36;Called&#36;\theta&#36;Yes.&#36;1-\alpha&#36; Credibility ceiling (one-sided)</p>
<p>This is the idea of the Bayesian region, and the rest of the idea is to present different Bayesian methods of estimation on this basis.</p>
<h4>Assumptions test</h4>
<ol>
<li>Probability of post-testing&#36;\pi(\theta|x)&#36; After that, calculate assumptions separately&#36;H_0&#36; &#36;H_1&#36; Post-probability &#36;\alpha_i=P(\theta_i|x)&#36; </li>
<li>When the probability ratio is back-checked (opportunity ratio)&gt;1&#36; 时不拒绝&#36;H_0&#36;  &#36;\frac{\alpha_0}{\alpha_{1&#125;&#125;&lt;1&#36;  时不拒绝&#36;H_{1}&#36; 接近&#36;A &#36;1-million period without judgment, without any conclusion.
That's the core idea of the Bayesian hypothetical test.</li>
</ol>
<p>Then we'll introduce the Beyers factor, which will help us study the hypothetical tests in Bayesian statistics.</p>
<h4>Forecast extrapolation</h4>
<p><strong>Here we have a response to the meaning of the margin distribution.</strong></p>
<p>In fact, we use the predictive extrapolation section to do our predictions using the margin distribution, and in this section we call the margin distribution the predictive distribution, the margin distribution.&#36;m(x)&#36;Expectations, medians, numbers, numbers, as our predictions.</p>
<p>The line of thought is as follows:
Because &#36;\pi(\theta|\boldsymbol{x})&#36; Yes &#36;\theta&#36; ♪ the back-up distribution, so ♪ &#36;g(z|\theta)\pi(\theta|\boldsymbol{x})&#36; As a given &#36;x&#36; Conditions&#36;(Z, θ)&#36;And then we'll put it right. &#36;\theta&#36; Score, get a score. &#36;x&#36; Random variable for time &#36;Z&#36; * The term "regulated" means "regulated" or "regulated" means "pregnated density".</p>
<h3>Basic concepts for statistical decision-making in Bayes</h3>
<h4>Three elements of a statistical decision</h4>
<p><strong>Sample space</strong> &#36;\chi&#36; It's a collection of potential values for samples. <strong>Sample distribution group</strong>&#36;f(x|\theta)&#36; It's the density function of the sample.
An event in sample space&#36;A&#36;
&#36;&#36;\left.P(A|\theta)=\left{\begin{array}{ll}\int_Af(x|\theta)\mathrm{d}x,&amp;x\text{is a continuous random variable}, \\sum x\inA}f (x|theta),&amp;x\text{for discrete random variable}.\end{array}\right.\right.&#36;</p>
<p><strong>Space for action</strong>: the non-empty collection of actions we can take on a statistical decision-making issue
For parameter estimation: operational space is a collection of estimates For hypothetical test questions: only two actions in operational space accept and reject the original hypothesis</p>
<p><strong>Loss Functions</strong>: Definition in parameter space and operational space&#36;\Theta\times D&#36; Dictatorial function above; assessed loss of action under a parameter extraction value </p>
<p>There are many types of loss function where we distinguish between the income matrix (function) and the loss matrix (function) yield positive and negative, and often the monetary unit means the gain, and negative means the loss. Loss</p>
<p>The loss function is only positive. We use it to describe the gains we should have received but not received, which is,
&#36;&#36;L(\theta,a)=\max Q(\theta,a)-Q(\theta,a)&#36;&#36;
Of which&#36;Q&#36;It's a revenue function. &#36;L&#36;It's a loss function.</p>
<h4>Risk function and consistency optimal decision-making function</h4>
<p><strong>Decision-making Functions</strong> : Defines the function that is to be valued in the decision-making space in the sample space &#36;\delta=\delta(\boldsymbol{x})&#36; </p>
<p><strong>Risk Functions</strong> : the average loss to measure a decision-making function to replace the loss function in the front and to expect a sample distribution
&#36;&#36;R(\theta,\delta)=E[L(\theta,\delta(\boldsymbol{X}))]=\int_{\mathcal{E&#125;&#125;L(\theta,\delta(\boldsymbol{x}))\mathrm{d}F(\boldsymbol{x}|\theta)&#36;&#36;
&#36;&#36;\left.=\left{\begin{array}{l}\int_{\mathcal{X&#125;&#125;L(\theta,\delta(\boldsymbol{x}))f(\boldsymbol{x}|\theta)\mathrm{d}\boldsymbol{x},\\sum_{\boldsymbol{x}\in\mathscr{X&#125;&#125;L(\theta,\delta(\boldsymbol{x}))f(\boldsymbol{x}|\theta),\end{array}\right.\right.&#36;&#36;</p>
<p>The only criterion for evaluating the decision-making function is his risk function, based on Wald's theory of statistical decision-making.
The smaller the risk function, of course.</p>
<p>If a risk function&#36;R(\theta,\delta)&#36; In all of them.&#36;\theta&#36;The value is smaller than the other risk function
We call his decision-making function more superior.</p>
<p>If you can find the least risk decision-making function, it's called<strong>Consistency and Optimal Decision Functions</strong></p>
<h4>BEYES' EXPERIENCES AND BEYES' RISK</h4>
<p>Definitions: Establishment &#36;\delta(\boldsymbol{x})&#36; Yes &#36;\theta&#36; . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .&#36;L(\theta,\delta(\boldsymbol{x}))&#36; For the loss function,&#36;F^{\pi}(\theta)&#36; Yes&#36;\theta&#36; The a priori distribution function, we call the next form the Bayesian expected loss.
&#36;&#36;
R(\pi,\delta(\boldsymbol{x}))=\int_{\Theta}L(\theta,\delta(\boldsymbol{x}))\mathrm{d}F^{\pi}(\theta)
&#36;&#36;
&#36;&#36;\left.=\left{\begin{array}{l}\int_{\Theta}L(\theta,\delta(\boldsymbol{x}))\pi(\theta)\mathrm{d}\theta,\\sum_iL(\theta_i,\delta(\boldsymbol{x}))\pi(\theta_i),\end{array}\right.\right.&#36;&#36;</p>
<p>It's not the same concept as the risk function because the average value is for the&#36;\theta&#36;The first step in the process is to calculate the expected distribution of the sample.</p>
<p>Definition: Risk function&#36;R(\theta,\delta)&#36;  &#36;F^{\pi}(\theta)&#36; Yes&#36;\theta&#36; A priori distribution function
&#36;&#36;\begin{gathered}
R_{\pi}(\delta(\boldsymbol{x})) =\int_\Theta R(\theta,\delta(\boldsymbol{x}))\mathrm{d}F^\pi(\theta)=E^\pi[R(\theta,\delta(\boldsymbol{X}))] \
=\int_\Theta\int_{\mathscr{X&#125;&#125;L(\theta,\delta(\boldsymbol{x}))\boldsymbol{f}(\boldsymbol{x}|\theta)\mathrm{d}\boldsymbol{x}\mathrm{d}F^\pi(\theta)
\end{gathered}&#36;&#36;
It's the Bayesian risk.</p>
<p>He's re-examining the risk function against the a priori density function, which is not the same as the Bayesian expected loss.</p>
<h4>Bhaith.</h4>
<p>If a decision function minimizes the risk to the Bayesian,
We call this decision-making function the Bayesian of Statistical Policy-Making. Break
If the a priori distribution is broad, the corresponding beyes solution is called the broad beyes. Break</p>
<h3>Beyes statistical calculations</h3>
<p>We need some statistical calculations from Bayesian statistics. </p>
<p>The statistical algorithms are designed to solve some of the computational problems in Bayes, and the central problem in Bayes is the computation of the later distribution and the digital characteristics of the later distribution, which we are presenting mainly the EM algorithms and the MCMC.</p>
<h3>It seems like a test.</h3>
<p>We've learned various tests in mathematical statistics, such as the hypothetical test for normal aggregates, yes.&#36;p&#36;The value test, and there are some non-parametric tests;<a href="/en/blog/2023/03/18/mathematical-statistics-notes/">Mathematical statistics</a> “Assumptions” section
However, we did not present a very important test after we presented a very seemingly seemingly seemingly seemingly seemingly disproportionate estimate:<strong>It seems like a test.</strong> We added him to Bayesian statistics.</p>
<h4>It seems to be more than statistically measured.</h4>
<p>For the most basic hypothetical test questions:
&#36;&#36;H_{0}:\theta\in\Theta_{0}\leftrightarrow H_{1}:\theta\in\Theta_{1},&#36;&#36;</p>
<p>We're thinking of thinking that's a very similar estimate. <a href="/en/blog/2023/03/18/mathematical-statistics-notes/">Mathematical statistics</a> The "extremely speciistic estimate" section considers two hypothetical functions under the sample.
&#36;&#36;L_{\Theta_{0&#125;&#125;\left(x\right)=\sup_{\theta\in\Theta_{0&#125;&#125;f\left(x,\theta\right),\L_{\Theta_{1&#125;&#125;\left(x\right)=\sup_{\theta\in\Theta_{1&#125;&#125;f\left(x,\theta\right).&#36;&#36;
It seems to be constructed more than statistically.
&#36;&#36;\lambda\left(X\right)=\frac{\sup_{\theta\in\Theta}f\left(X,\theta\right)}{\sup_{\theta\in\Theta_{0&#125;&#125;f\left(X,\theta_{0}\right)}&#36;&#36;
And when it seems to be bigger than statistics, we naturally have a tendency to reject the original assumption because it seems smaller.
We can naturally give the test function as:
&#36;&#36;\left.\varphi\left=begin{cases}1,&amp;\lambda\left(x\right)&gt;c,\r,&amp;\lambda\left(x\right)=c,\0,&amp;\lambda\left(x\right)&lt;c\end{cases}\right.&#36;&#36;</p>
<p>The core of the problem is now.<strong>Studying the distribution of statistically comparable amounts, or their equivalent, to determine our rejection field</strong></p>
<h4>It seems like a test.</h4>
<p>General &#36;\lambda(X)&#36; The expression is complex and it is very difficult to calculate his distribution; so we conclude:<strong>If&#36;\lambda(X)=g(T(X))&#36; Yes&#36;T(X)&#36;So the test function can be naturally deformed to</strong>
&#36;&#36;\left.\varphi\left(x\right)=\begin{cases}1,&amp;T\left(x\right)&gt;c,\r,&amp;T\left(x\right)=c,\0,&amp;T\left(x\right)&lt;- I'm not gonna get you out of here.
When the decline is made,&#36;\phi(x)&#36;Medium altimeter Reverse</p>
<p>If distribution is not specified, the maximum distribution is acceptable, and this is presented later in the section on "More than the maximum distribution."</p>
<h4>It seems to be an example of statistically different.</h4>
<p>Set&#36;X=\left(X_{1},X_{2},\cdots,X_{n}\right)&#36; It's from the normal distribution. &#36;&#123;N\left(\mu,\sigma^{2}\right),&#36; &#36;-\infty&lt; \mu&lt;+\infty,\sigma^{2}&gt;I.i.d. sample taken from the middle axis, ask for the following questions
&#36;&#36;
H_{0}:\mu=\mu_{0}\leftrightarrow H_{1}:\mu\neq\mu_{0}
&#36;&#36;
The seemingly comparable test.</p>
<p>The function appears to be
&#36;&#36;f\left(x,\theta\right)=\left(2\pi\sigma^{2}\right)^{-\frac{n}{2&#125;&#125;\exp\left{-\frac{1}{2\sigma^{2&#125;&#125;\sum_{i=1}^{n}\left(x_{i}-\mu\right)^{2}\right},&#36;&#36;</p>
<p>The two scenarios are estimated to be very similar.
&#36;&#36;\widehat{\mu}=\overline{X},\widehat{\sigma}^{2}=\frac{1}{n}\sum_{i=1}^{n}(x_{i}-\overline{X})^{2};&#36;&#36;
&#36;&#36;\tilde{\sigma}^{2}=\frac{1}{n}\sum_{i=1}^{n}\left(x_{i}-\mu_{0}\right)^{2}.&#36;&#36;</p>
<p>So, two hypotheses are given.
&#36;&#36;\sup_{\theta\in\Theta}f\left(x,\theta\right)=f\left(x,\widehat{\mu},\widehat{\sigma}^{2}\right)=\left(\frac{2\pi e}{n}\right)^{-\frac{n}{2&#125;&#125;\left(\sum_{i=1}^{n}\left(x_{i}-\bar{x}\right)^{2}\right)^{-\frac{n}{2&#125;&#125;,
\\sup_{\theta\in\Theta_{0&#125;&#125;f\left(x,\theta\right)=f\left(x,\mu_{0},\tilde{\sigma}^{2}\right)=\left(\frac{2\pi e}{n}\right)^{-\frac{n}{2&#125;&#125;\left(\sum_{i=1}^{n}\left(x_{i}-\mu_{0}\right)^{2}\right)^{-\frac{n}{2&#125;&#125;&#36;&#36;
So there's a statistical comparison.
&#36;&#36;\lambda\left(X\right)=\left(1+\frac{1}{n-1}T^{2}\right)^{\frac{n}{2&#125;&#125;&#36;&#36;
Of which&#36;T&#36;Yes&#36;&#36;T=\sqrt{n}\left(\overline{x}-\mu_{0}\right)/\sqrt{\frac{1}{n-1}\sum_{i=1}^{n}\left(x_{i}-\overline{x}\right)^{2&#125;&#125;.&#36;&#36;
So we can use statistics.&#36;T&#36;To conduct an apparent comparison.
&#36;&#36;P\left (\left|T\right|)&gt;c\left|H_{0}\right)=\alpha.\right.&#36;&#36;
当原假设成立的时候 &#36;T\sim t_{n-1}&#36;  因此
&#36;&#36;\varphi\left(X\right)=\left{\begin{matrix}1,&amp;\left|T\right|\geqslant t_{n-1}\left(\alpha/2\right)\0,&amp;\left|T\right|&lt;t_{n-1}\left(\alpha/2\right)\end{matrix}\right.&#36;&#36;</p>
<h4>The maximum distribution of the apparent comparison</h4>
<p>It seems that the distribution is not always easy to calculate than it was when the original hypothesis was established, and its precise distribution may sometimes be unsolved, which is also the situation that happens in the hypothetical tests.</p>
<p>But if our sample is i.i.d, we can study it with an approximate maximum distribution.</p>
<p>Theorem: Set &#36;\Theta&#36; The dimension is &#36;k,\Theta_0&#36; The dimension is &#36;s&#36; if &#36;k-s=t&gt;0&#36;, 且样本分在满足一定的正则条件，则对似然比检验问题,在原假设 &#36;H_{0}&#36; 成立之下，当样本 &#36;When I was a little bit old,
&#36;&#36;
2\ln\lambda\xrightarrow{}\chi_{t}^{2}.
&#36;&#36;</p>
<h2>Selecting aforecast distribution</h2>
<h3>Subjective probability</h3>
<h4>Introduction</h4>
<p><strong>Subjective probabilities are the chances of people to speculate about the probability of an event.</strong> A bet on a game, a stock boom, and these random phenomena are not repeated, and we can't use frequencies to study probability.</p>
<p>At this point we actually abandoned the definition of frequency of probability in classical statistics, but it is a complement to the traditional definition of probability (frequency is not visible) and is consistent with our visual perception (in fact, subjective probability is often used naturally).</p>
<p>We need to use subjective probabilities only if we don't have any information to do it a priori. </p>
<h4>Use relative approximation</h4>
<p>Probability has a fair definition </p>
<p>If we know there's only two sides to an event, <del>and</del>A.&#36;. And the probability of the former is twice as high as the probability of the latter.
So we can get the probability of a equation by definition of justice. &#36;\frac{1}{3}~ \frac{2}{3}&#36;  That's the relative approximation of the use.</p>
<p><em>It seems that English is a possibility that the term "probability" in statistics cannot be confused with probability, which is a characteristic of our existence in the theory of probability, which seems to be derived from sample statistics.</em></p>
<p>This method is usually only theoretically valuable.</p>
<h4>Use of expert advice</h4>
<p>This is the central method for determining the probability of a supervisor.
Assessing the recommendations of experts in multiple related areas and synthesizing them
The subjective probabilities of experts are generally more accurate than those of ordinary people.</p>
<h4>Use of historical information</h4>
<p>Assessment of the historical situation of research on similar issues</p>
<p>Is that a probability frequency? Not really.
We use historical information on similar events here, not multiple observational studies of the current events.</p>
<p>The fact that the events we're studying cannot be repeated is that the Bayesian statistics are produced. Learn.</p>
<h3>Use a priori information</h3>
<p>Using a priori information to determine a priori distribution requires some knowledge of a priori. </p>
<p>Histograms and nuclear density curves need to be supplemented by subjective probabilities (experts) or historical information
Relative approximation is an extension of the use of this subjective information in the front.</p>
<p>The scores and the variation method require expert judgment many times.</p>
<p>To determine a prior distribution by superparameters or some historical information</p>
<p>If we replace historical information with real sample information about a priori, then we're really using a priori information.
<strong>The combination of subjective probabilities and the a priori information section to understand their thinking is close.</strong></p>
<h4>Histogram and nuclear density curve</h4>
<p>Applicable to a limited sub-area in parameter space and sufficient a priori information, and the probability of a priori distribution in a zone based on subjective probability or historical information</p>
<p>Based on these a priori information, heterographs or nuclear density curves, they're all a non-parametric estimate of the original distribution.</p>
<p>And at this point we can study the probabilities that are required in the light of the original distribution of the theory.</p>
<h4>Relatively Appearances</h4>
<p>This approach is generally applied to cases where the a priori distribution is in a limited range, and it is an extension of the relatively semblance of the front-end use, with the goal of obtaining a first-spectrum distribution.</p>
<p>For example, we know that the a priori distribution is in the zone.&#36;[0,1]&#36; So, we can draw a map of the possibilities, and we'll use the smallest one as a relative one. Here's the picture. We need to regularize it.
<img src="/assets/images/probability-statistics-notes/bayesian-statistics-posterior-distributions-notes-01.png" alt="Bayesian statistics"></p>
<h4>Parity and Variable Method</h4>
<p>The method of scoring and the method of variing are the methods of obtaining subjective probabilities based on expert advice and then processing them into probabilities curves, which are actually a continuation of the notion of subjective probabilities. </p>
<p>The scoring method is the division of the possible range of parameters into equal lengths, and the inviting of experts in each small area to give subjective probabilities</p>
<p>The variation method is to divide all the zones into two sub-divisions of equal opportunity (no length of the zone) and experts need to give points.</p>
<p>We usually use the fractional method more so that we can better drive experts to think and give better answers.</p>
<p>With the division given by the experts, we can easily complete the construction of the probability histogram based on this information.</p>
<h4>First, make sure you're a priori distributed and then determine the hyperparameter.</h4>
<p>It's a very broad method.</p>
<h5>Superparameter</h5>
<p>We call the parameters in the aforecast distribution hyperparameter</p>
<p><strong>In machine learning, the concept of hyperparameters is derived from Bayesian statistics, where parameters refer to the number that will be automatically determined during model learning, and superparameters refer to those that will need to be determined in advance in those models, so that the design of an automated superparameter adjuster will automatically determine the cross-reference.</strong></p>
<p>And this section is the first one we're thinking of.&#36;\theta&#36;The a priori density is... &#36;\pi(\theta)&#36; Among them are pending parameters &#36;\mu&#36;We're just gonna have to set this super-parameter behind us.&#36;\mu&#36; I'll know about the a priori distribution.</p>
<p>Of course, it's easy to see. The core of this method is the selection of a priori distribution. &#36;\pi(\theta)&#36;  If this is the wrong place to choose, then the estimates will be very different.</p>
<h5>Determine aforecast distribution</h5>
<p>Based on the characteristics of the parameter space
Parameter Space &#36;(-\infty,\infty)&#36; Distribution: Normal distribution, Cosy distribution (average and variance not present), students &#36;t&#36; Distribution, etc.;
The distribution of parameter space (0, \infty) is: index distribution, Wable distribution, gamma distribution, etc.;
Parameter Space &#36;0,1,\ldots&#36; Distribution: Porcelain distribution, geometry, etc.;</p>
<h5>Determine the hyperparameter</h5>
<h6>Rectangular estimate superparameters</h6>
<p>The sample rectangles that were processed from the a priori information, and the equation that was used to determine the hyperparameters using the sample rectangulars equal to the total rectangulars, is the rectangular estimates, which are just used in the Bayesian statistics.</p>
<h6>Estimated decimals</h6>
<p>It's another idea of the super-parametrics.</p>
<p>We know that the overall fraction must be a function of the hyperparameter, and the sample fraction can be processed using sample information, and we can equalize the equation to determine the hyperparameter.</p>
<p><strong>The score estimate is actually an important branch of classical statistics, but it is not presented in most mathematical statistics materials.</strong></p>
<p><strong>There are certainly more than two ways to determine the super-parameters than one, and we can estimate them together or use ideas from different mathematical statistics, such as those that are very similar.</strong></p>
<h4>Use a decimal to determine a priori CDF</h4>
<p>If there is a greater a priori distribution of medians, it's like a proposed CDF curve, and the CDF curve, which is designed to reflect the a priori distribution, and the section of this paper "Standing with a Nuclear Density Curve" is close to thinking.</p>
<h3>Use edge distribution</h3>
<p><strong>Marginal distribution lacks practical meaning, and the sampling of marginal distribution is theoretically possible, and in fact the knowledge of this subsection is less used in practical application</strong></p>
<h4>Marginal distribution</h4>
<h5>Definitions</h5>
<p>We've been working on the definition of the margin distribution, and we've been working on this one. Section</p>
<p>When Random Variable&#36;X&#36;Probability density function&#36;f(x|\theta)&#36; The pre-distribution density function is&#36;\pi(\theta)&#36;  can be defined as the distribution of random variables as
&#36;&#36;00\
m\left(x\right)&amp; =\int_{\Theta}f(x|\theta)\mathrm{d}F^{\pi}(\theta)  \
&amp;\left.=\left{\begin{array}{ll}\int_\Theta f(x|\theta)\pi(\theta)\mathrm{d}\theta,&amp;\\theta\text{is a continuous random variable}, \\sum if(x\theta i)\pi(\theta i),&amp;\theta\text{for discrete random variables}. \end{array}\right.\right.\right.
I'm sorry, I'm sorry.
The margin distribution is obtained by combining sample and a priori information. </p>
<h5>Mixed distribution</h5>
<p>When Random Variable&#36;X&#36;With probability.&#36;p&#36;Yes.&#36;F({x|\theta_{1&#125;&#125;)=F_{1}&#36;Median Value &#36;1-p&#36;The probability is that&#36;F({x|\theta_{1&#125;&#125;)=F_{2}&#36;We know that the combination is the only way to get a distribution. &#36;X&#36;The hybrid distribution function is&#36;&#36;F(x)=pF(x|\theta_1)+(1-p)F(x|\theta_2)&#36;&#36;
And it turns out that the distribution of the edges is actually a form of promotion of mixed distribution, a limitation.&#36;\theta&#36;For the discrete variable, the edge distribution is the result of a combination of probabilistic density functions.</p>
<p>When?&#36;\theta&#36;When the continuous variable is &#36;m(x)&#36;It's a mixture of infinity infinity.</p>
<h5>An example of the calculation of the marginal distribution</h5>
<p>Set Aspect&#36;\theta&#36;♪ A sample of time ♪&#36;X&#36; Subject to normal distribution&#36;N(\theta,\sigma^{2})&#36; of which&#36;\sigma&#36;Known.&#36;\theta&#36;A priori distribution is&#36;N(\mu_\pi,\sigma_\pi^2)&#36; Calculate edge distribution&#36;m(x)&#36;
&#36;&#36;\begin{aligned}
m(x)&amp; =\int_{-\infty}^{\infty}f(x|\theta)\pi(\theta)\mathrm{d}\theta   \
&amp;=\frac{1}{2\pi\sigma\sigma_{\pi&#125;&#125;\int_{-\infty}^{\infty}\exp\left{\left.-\frac{1}{2}\left[\frac{(x-\theta)^{2&#125;&#125;{\sigma^{2&#125;&#125;+\frac{(\mu_{\pi}-\theta)^{2&#125;&#125;{\sigma_{\pi}^{2&#125;&#125;\right]\right}\mathrm{d}\theta\right.  \
&amp;=\frac{1}{2\pi\sigma\sigma_{\pi&#125;&#125;\int_{-\infty}^{\infty}\exp\left{-\frac{A}{2}\left(\theta-\frac{B}{A}\right)^{2}\right}\cdot\exp\left{-\frac{1}{2}\left(C-\frac{B^{2&#125;&#125;{A}\right)\right}\mathrm{d}\theta  \
&amp;=\frac1{\sqrt{2\pi(\sigma^2+\sigma_\pi^2)&#125;&#125;\exp\left{-\frac{(x-\mu_\pi)^2}{2(\sigma^2+\sigma_\pi^2)}\right}
\end{aligned}&#36;&#36;
其中
&#36;&#36;A=\frac{1}{\sigma^2}+\frac{1}{\sigma_{\pi}^2},\quad B=\frac{x}{\sigma^2}+\frac{\mu_{\pi&#125;&#125;{\sigma_{\pi}^2},\quad C=\frac{x^2}{\sigma^2}+\frac{\mu_{\pi}^2}{\sigma_{\pi}^2}&#36;&#36;
也就是
&#36;&#36;N(\mu_\pi,\sigma^2+\sigma_\pi^2)&#36;&#36;</p>
<p>Or do you construct a density function with a final value of one, which is the classic method of processing the fractions in probability and statistics?</p>
<h5>Perception Probability</h5>
<p>Time of lapse of setting an electronic component &#36;X&#36; Obedience index distribution &#36;Exp(1/\theta)&#36;, the density function is &#36;f (x|theta)=\theta^-1}mathrm{e^-x/\theta}x&gt;0)&#36;, 若未知参数 &#36;\theta&#36; 的先验分布为逆伽马分布 &#36;\\Gamma^(1,100) calculates the probability that the component will expire on the edge before 200 hours.
<strong>The probability of the edge is the fraction of the margin distributed between the corresponding zones.</strong>
So the result of the calculations here is that
&#36;&#36;\int_{0}^{200}m(x)dx =\frac{2}{3}&#36;&#36;</p>
<h4>Select ML-II method for afore-distribution</h4>
<p>Now, back to the point of this section, we're looking at a priori distribution selection, so let's talk about how to use the edge distribution to determine a priori distribution.</p>
<p>Our core thinking is still more likely to occur in the sample.</p>
<p>So we're going to select a priori distribution that gives the sample distribution a higher probability of the current situation. </p>
<h5>Definitions and methods</h5>
<p>Definitions:
Set&#36;\Gamma&#36; It's a priori class we're considering.&#36;x=(x_{1},x_{2}...x_{n})&#36; If it exists &#36;\hat{\pi}\in\Gamma&#36; Make
&#36;&#36;m(\boldsymbol{x}|\hat{\pi})=\sup_{\pi\in\Gamma}\prod_{i=1}^nm(x_i|\pi)&#36;&#36;
Name&#36;\hat{\pi}&#36;The largest a priori for type II, or ML-II. </p>
<p>"In fact, it's a...&#36;m(x)&#36;Considers it a very, very a priori function)
If the a priori density function is known to be just an unknown hyperparameter, then we can simplify the problem above to the following form, which is the one that we're looking at.&#36;\Lambda&#36; It's a collection of values for hyperparameters.
&#36;&#36;m(\boldsymbol{x}|\hat{\lambda})=\sup_{\lambda\in\Lambda}m(\boldsymbol{x}|\lambda)=\sup_{\lambda\in\Lambda}\prod_{i=1}^nm(x_i|\lambda)&#36;&#36;
That's the big, obvious question of studying our parametrics.&#36;\Lambda&#36;I'll take the value.</p>
<p>Why a bunch of serials?
The margin distribution is calculated by the formula based on the base edge distribution for a sample of the margin distribution
Of course, we can't have only one exterior distribution.
<strong>We can understand it with a very specious idea of how to construct a function in a sample.</strong></p>
<p>And simple random samples are naturally independent (i.i.d). The approximation is itself a series of probabilities and then it's dramatically increased.
We usually call multiple multiplications. <strong>Joint Edge Density Function</strong></p>
<h5>Examples</h5>
<p>Set Random Variables &#36;X\sim N(\theta,\sigma^2)&#36;, of which &#36;\sigma^2&#36; Known, reset. &#36;\theta\sim N(\mu_\pi,\sigma_\pi^2)&#36;If... &#36;X=(X_1,\cdots,X_n)&#36; To Distribution From Margins &#36;m(x|\lambda)&#36; i.i.d. sample taken, test confirmed &#36;\theta&#36; A priori distribution</p>
<p>First we'll study the distribution of the edges.  &#36;X&#36;The distribution of the edges is&#36;N(\mu_\pi,\sigma^2+\sigma_\pi^2)&#36; This paper, for example, "An example of the calculation of the distribution of the edges" section</p>
<p>According to the method before, we give the functions that we need to do to be radicalized.&#36;\bar{x}&#36; and&#36;S^{2}&#36;The average and difference of the sample taken is
&#36;&#36;00\
&amp;L(\mu_\pi,\sigma_\pi^2|\boldsymbol{x}) =m(\boldsymbol{x}|\boldsymbol{\lambda})  \
&amp; =\left[2\pi(\sigma^2+\sigma_\pi^2)\right]^{-n/2}\exp\left{-\frac{1}{2(\sigma^2+\sigma_\pi^2)}\cdot\sum_{i=1}^{n}(x_i-\mu_\pi)^2\right}  \
&amp;=\left[2\pi(\sigma^2+\sigma_\pi^2)\right]^{-n/2}\exp\left{\frac{-nS^2}{2(\sigma^2+\sigma_\pi^2)}\right}\cdot\exp\left{-\frac{n(\bar{x}-\mu_\pi)^2}{2(\sigma^2+\sigma_\pi^2)}\right},
\end{aligned}&#36;&#36;</p>
<p>It's easy to see.
If&#36;\sigma_{\pi}^{2}&#36;- Fixed. - So...&#36;\mu_{\pi}=\bar{x}&#36;When you're trying to maximize it, you don't need to be partial.</p>
<p>Bring in&#36;\mu_{\pi}=\bar{x}&#36;
&#36;&#36;\phi(\sigma_{\pi}^{2})=\big[2\pi(\sigma^{2}+\sigma_{\pi}^{2})\big]^{-n/2}\exp\bigg{\frac{-nS^{2&#125;&#125;{2(\sigma^{2}+\sigma_{\pi}^{2})}\bigg}.&#36;&#36;
The logarithmic guidance study was extremely successful.
&#36;&#36;\hat{\sigma}_\pi^2=S^2-\sigma^2&#36;&#36;
Obviously, it's impossible to take the burden.&#36;S^{2}&#36;When I was little.&#36;0&#36;Yeah.</p>
<h4>Select the rectangular method for a prior distribution</h4>
<p>The core here is to study the relationship between the margin distribution and the a priori distribution rectangular, so that the rectangular estimates of the a priori distribution parameters are achieved by a sample of reality, and the prior direct study of the a priori rectangular estimates are different, and at this point we lack direct knowledge of the a priori. </p>
<p><strong>The core idea is to estimate the edge distribution rectangular, which is a function of the margin distribution rectangular.</strong></p>
<h5>Theory leads</h5>
<p>Calculate sample distribution&#36;f(x|\theta)&#36;The expectations and the differences.&#36;\theta&#36;It's a constant, yeah.&#36;x&#36;The hope is different from the difference)
&#36;&#36;\mu(\theta)=E^{X|\theta}(X),\quad\sigma^{2}(\theta)=E^{X|\theta}[X-\mu(\theta)]^{2},&#36;&#36;
Calculate edge distribution&#36;m(x)=m(x|\lambda)&#36;Expectations and differences&#36;\lambda&#36;(overparameters) is constant pair&#36;x&#36;The hope is different from the difference)</p>
<p>&#36;&#36;\begin{aligned}
&amp;\mu_{m}(\lambda) =E^{X|\lambda}(X)=\int_{\mathcal{C&#125;&#125;xm(x|\lambda)\mathrm{d}x=\int_{\mathcal{C&#125;&#125;\int_{\Theta}xf(x|\theta)\pi(\theta|\lambda)\mathrm{d}\theta\mathrm{d}x  \
&amp;=\int_\Theta\left[\int_{\mathscr{E&#125;&#125;xf(x|\theta)\mathrm{d}x\right]\pi(\theta|\lambda)\mathrm{d}\theta=\int_\Theta\mu(\theta)\pi(\theta|\lambda)\mathrm{d}\theta  \
&amp;=E^{\theta|\lambda}[\mu(\theta)], \
&amp;\sigma_{m}^{2}(\lambda) =E^{X|\lambda}\left{[X-\mu_{m}(\lambda)]^{2}\right}=\int_{\mathcal{E&#125;&#125;[x-\mu_{m}(\lambda)]^{2}m(x|\lambda)\mathrm{d}x  \
&amp;=\int_{\mathscr{X&#125;&#125;\int_{\Theta}[x-\mu_m(\lambda)]^2f(x|\theta)\pi(\theta|\lambda)\mathrm{d}\theta\mathrm{d}x \
&amp;=\int_{\Theta}\left{\int_{\mathscr{X&#125;&#125;[x-\mu_{m}(\lambda)]^{2}f(x|\theta)\mathrm{d}x\right}\pi(\theta|\lambda)\mathrm{d}\theta  \
&amp;=\int_{\Theta}E^{X|\theta}\left[x-\mu_{m}(x)\right]^{2}\pi(\theta|\lambda)\mathrm{d}\theta,
\end{aligned}&#36;&#36;
其中
&#36;&#36;\begin{aligned}
&amp;E^{X|\theta}\left{\left[x-\mu_m(\lambda)\right]^2\right} =E^{X|\theta}\left(\left{\left[x-\mu(\theta)\right]+\left[\mu(\theta)-\mu_{m}(\lambda)\right]\right}^{2}\right)  \
&amp;=E^{X|\theta}\left{\left[x-\mu(\theta)\right]^2\right}+E^{X|\theta}\left{\left[\mu(\theta)-\mu_m(\lambda)\right]^2\right} \
&amp; =\sigma^2(\theta)+\left[\mu(\theta)-\mu_m(\lambda)\right]^2.
\end{aligned}&#36;&#36;
因此方差实际上的表示为
&#36;&#36;\begin{gathered}
\sigma= (m)= (m)= (m)= (m)= (m)= (m)= (m)= (m)= (m)= (m)= (m)= (m)= (m){ (m){ (m){ (m){ (m){ (m){ (m){ (m){ (m){ (m){ (m){ (d)\theta{ (m){ (m){ (m){ (m){ (m){ (m){ (m){ (m){ (m){ (m){ (m){ (m){ (m){ (m){ (m){ (m){ (m){ (m){t){ (m){t){ (m){tta){ (m){ta (m){tta){ (m){ (m) (m) (m){tta){ (m) (m) (m) (m) (m) (m)
♪ It's a good thing you're not a good guy ♪
I'm sorry, I'm sorry.
Parameters here&#36;\lambda&#36;It's a generic description of the parameters in the aforethought distribution, not just the parameters.</p>
<p>If we're only two super-parameters in a priori distribution, &#36;\lambda_{1},\lambda_{2}&#36; So the rectangular estimate only takes two volumes, which is easy to calculate for two margin distribution rectangles (the most basic expectations and the difference)
What's that?<em>{m}=\overline{X}=\frac{1}{n}\sum</em>♪ I'm not gonna let you go ♪<em>{m}^{2}=S^{2}=\frac{1}{n-1}\sum</em>{i=1}^{n}(X_{i}-\overline{X})^{2}&#36;&#36;</p>
<p>The equation of the extricated sample rectangular has the results we need to know about the hyperparameter.
&#36;&#36;\left.\left{\begin{array}{l}\hat{\mu}_m=E^{\theta|\boldsymbol{\lambda&#125;&#125;\big[\mu(\theta)\big],\\hat{\sigma}_m^2=E^{\boldsymbol{\theta}|\boldsymbol{\lambda&#125;&#125;\big[\sigma^2(\theta)\big]+E^{\boldsymbol{\theta}|\boldsymbol{\lambda&#125;&#125;\big[\mu(\boldsymbol{\theta})-\mu_m(\boldsymbol{\lambda})\big]^2\bigg}.\end{array}\right.\right.&#36;&#36;</p>
<p>Language expression of the final formula:</p>
<ul>
<li>Estimates of the margin distribution average: we calculate the sample distribution average for parameters&#36;\lambda&#36; Expectations</li>
<li>Estimates of differentials in the margin distribution: the difference in the sample distribution is about parameters&#36;\lambda&#36; The expectations and the sample distribution averages about the parameters&#36;\lambda&#36; the difference between</li>
</ul>
<p><strong>The averages and the differences in the sample distributions are their own, and they're their parameters.&#36;\lambda&#36;That's the principle of calculation.</strong></p>
<h5>Examples</h5>
<p>Set &#36;X|\theta\sim N(\theta,1)&#36;,parameters &#36;\theta&#36; A priori distribution to &#36;N(\mu_\pi,\sigma_\pi^2)&#36;, of which &#36;\lambda=(\mu_\pi,\sigma_\pi^2)&#36; Unknown. &#36;X=(X_1,\cdots,X_n)&#36; To Distribution From Margins &#36;m(x|\lambda)&#36; i.i.d. sample from which the sample is calculated as the average sample value &#36;\bar{X}=10,S^{2}=3.&#36; Try to be sure. &#36;\theta&#36; A priori distribution</p>
<p>Study using rectangular estimation</p>
<p>The expected distribution and the difference in the sample distribution are different &#36;\theta&#36; and &#36;1&#36;</p>
<p>The expectations and differences in calculating the marginal distribution are:
&#36;&#36;\left.\left{\begin{array}{l}\mu_m(\lambda)=E^{\theta|\lambda}(\theta)=\mu_\pi,\\sigma_m^2(\lambda)=E^{\theta|\lambda}(\sigma^2(\theta))+E^{\theta|\lambda}[\theta-\mu_\pi]^2=1+\sigma_\pi^2.\end{array}\right.\right.&#36;&#36;</p>
<p>Average and variance of the sample taken into the sample
&#36;&#36;\left.\left{\begin{array}{l}10=\overline{X}=\mu_\pi,\3=S^2=1+\sigma_\pi^2.\end{array}\right.\right.&#36;&#36;
And so...&#36;\theta&#36;The a priori distribution is&#36;N(10,2)&#36;</p>
<h3>No information a priori distribution</h3>
<p>The Beyers statistics are characterized by the use of a priori information in statistical extrapolation.</p>
<p>But sometimes, there's little or no information, and we still want to use the Beyers statistics idea.<strong>No information a priori (noninformation preor) is that there is no preference for any parameter space at all</strong></p>
<h4>Bayesian hypothetical and broad a priori distribution</h4>
<p>No information a priori distribution means our a priori distribution does not contain any&#36;\theta&#36;There is no preference for any value in the information. </p>
<h5>Definitions</h5>
<p>It's natural. We can put it in.&#36;\theta&#36;As a flat distribution in the range of values taken, it is considered a priori distribution, which is the Bayes assumption, which usually follows:</p>
<ul>
<li>Disperse evenly if&#36;\Theta&#36; It's a limited set, so the dispersive distribution is evenly distributed. &#36;P(\theta=\theta_i)=1/n&#36;</li>
<li>A limited area is evenly distributed if&#36;\Theta&#36; It's a limited area.&#36;[a,b]&#36; So the limited area is evenly distributed&#36;U(a,b)&#36;</li>
<li>Broad a priori distribution if&#36;\Theta&#36; - No bounds. - So?&#36;\pi(\theta)\equiv1&#36; He doesn't meet probabilistic criteria, so it's broad.</li>
</ul>
<p>Definitions:
If&#36;\theta&#36;A priori distribution&#36;\pi(\theta)&#36; Satisfied</p>
<ul>
<li>&#36;\pi(\theta)\equiv1&#36;  And...&#36;\int_{\Theta}\pi(\theta)\mathrm{d}\theta=\infty;&#36;</li>
<li>Post-density&#36;\pi(\theta|x)&#36;It's normal density function.
And then, "Could"&#36;\pi(\theta)&#36; Yes.&#36;\theta&#36; Broad a priori density (improper prier density)</li>
</ul>
<p><strong>It's easy to know whether a priori density is broad enough to multiply any constant or a priori density is broad enough to be a priori.</strong></p>
<h5>Bates' hypothesis is not enough.</h5>
<p>The Beyers hypothesis is a disadvantage, the biggest one being uncertainty.</p>
<p>If we're right...&#36;p&#36;I'm equally ignorant.&#36;p^2,p^3&#36;So, theoretically, we take the Bayesian hypothesis.&#36;U(0,1)&#36;The distribution of the three of them should not change; obviously, in many cases it is not.</p>
<p><strong>Bates' assumption is not enough for the change of the constant.</strong>
For example: consider normal standard deviations &#36;\sigma\in(0,\infty)&#36;, define a change
&#36;&#36;
\eta=\sigma^2\in(0,\infty)
&#36;&#36;
then &#36;\eta&#36; Normal difference
Set &#36;\sigma&#36; The a priori density function is &#36;\pi(\sigma)&#36;, &#36;\eta&#36; A priori density function is &#36;\pi^<em>(\eta)&#36;, 那么&#36;The \eta&#36; density function can be expressed as
&#36;&#36;&#36;2 million</em>(\eta)=\pi(\sqrt{\eta})\left|\frac{d\sigma}{d\eta}\right|&#36;&#36;</p>
<p>As you can see, you cannot set a constant for a priori distribution of a parameter, which means that the Bayesian hypothesis cannot be used at random.</p>
<h4>No information a priori for the position parameter family</h4>
<p>General &#36;X&#36; . The density function is as follows: &#36;f(x-\theta)&#36;, its sample space &#36;\mathscr{X}&#36; and parameter space &#36;\Theta&#36; The distribution of these density functions is called the positional parameter group (localization modeler family), and the location function is a series of sites that are located in the country. &#36;\theta\in\Theta&#36; Called position parameters.</p>
<p>Here are two examples of the two position parameter communities.
Normal distribution
&#36;&#36;\frac{1}{\sqrt{2\pi}\sigma}\exp\Big{\frac{1}{2\sigma^2}(x-\theta)^2\Big}=f(x-\theta)&#36;&#36;
Couchy distribution
&#36;&#36;\frac1\pi\cdot\frac\lambda{\lambda^2+(x-\mu)^2}=f(x-\mu)&#36;&#36;</p>
<p>It's easy to see that the position parameter community is not in the same shape as the lateral variant.</p>
<p>Yeah.&#36;X&#36;I can do the transposition. &#36;Y=X+c&#36;  And also against arguments&#36;\theta&#36;I can do the transposition. &#36;\mu=\theta+c&#36; It's easy to know. &#36;Y&#36;The density function is as follows: &#36;f(y-\mu)&#36; Or is it a member of the position parameter community and the sample space and parameter space is unchanged? </p>
<p>So the statistical problems of both studies are the same, and we should think they have the same no-information a priori, and we'll go to prove that no-information a priori density is the same.&#36;\pi(\theta)\equiv1&#36; </p>
<p>You! &#36;\pi&#36; And &#36;\<em>}&#36;  分别表示 &#36;\theta&#36; 与 &#36;\eta&#36; 的无信息先验密度，以上论点说明 &#36;\pi&#36; 和  &#36;\pi^{</em>It should have the same a priori density, that is
&#36; \ pi (\ pi)= pi^<em>(\tau)&#36;&#36;
由于前面的线性关系我们知道
&#36;&#36;\pi(\eta)=\pi^</em>(\eta)=\pi(\eta-c)&#36;&#36;
特别的 我们取&#36;\eta=c&#36;
&#36;&#36;\pi(c)=\pi(0)=\text{ constant}.&#36;
So we take it.&#36;\pi(\theta)\equiv1&#36; It's reasonable.
I'm not sure what I'm talking about.&#36;\theta&#36; When a position parameter is not previously obtained as constant or 1</p>
<h4>No information a priori for the spectrometers</h4>
<p>General &#36;X&#36; . The density function is as follows: &#36;\sigma^{-1}\varphi(x/\sigma)&#36;,&#36;\sigma of which&gt;0&#36; 为刻度参数，参数空间为 &#36;\matbb{R} + (0, \infty) &#36;, and the distribution of these density functions is called the scale parameter family</p>
<p>Here are a few examples.
Normal distribution with an average of 0
&#36;&#36;00\
f (x|sigma)&amp; =\frac1{\sqrt{2\pi}\sigma}\exp\left{-\frac{x^2}{2\sigma^2}\right}  \
&amp;=\sigma^{-1}\left[\frac1{\sqrt{2\pi&#125;&#125;\exp\Big{\left.-\frac12\left(\frac x\sigma\right)^2\right}\right]=\sigma^{-1}\varphi\Big(\frac x\sigma\Big),
\end{aligned}&#36;&#36;
伽马分布
&#36;&#36;f(x|\lambda)=\frac{\lambda^{-r&#125;&#125;{\Gamma(r)}x^{r-1}\mathrm{e}^{-x/\lambda}=\lambda^{-1}\Big[\frac{1}{\Gamma(r)}\Big(\frac{x}{\lambda}\Big)^{r-1}\mathrm{e}^{-x/\lambda}\Big]=\lambda^{-1}\varphi\Big(\frac{x}{\lambda}\Big),&#36;&#36;</p>
<p>And the same principle, which shows the constantity of the tic parameter family in the tics.</p>
<p>Yeah.&#36;X&#36;Change &#36;Y=cX&#36; Yeah.&#36;\theta&#36;Make the corresponding changes &#36;\eta=c\sigma&#36;  Got it.&#36;Y&#36; Or a member of the spectroparameters, so it's reasonable for both to choose the same as the non-information a priori.</p>
<p>According to the same means of proof as before, we take
&#36;&#36;\ (\sigma)=firc1}{\sigma=&gt;(0) &#36;
Uninfoted aforecast distribution as a symmetric parameter family</p>
<h4>Position-scale parameter family</h4>
<p>Let's take a position-scale parameter in the context of the previous text. Group</p>
<p>Set density functions with two parameters &#36;\mu&#36; and &#36;\sigma&#36;, and density takes the following forms:
&#36;&#36;
p(x;\mu,\sigma)=\frac1\sigma f\left(\frac{x-\mu}\sigma\right),\mu\in(-\infty,\infty),\sigma\in(0,\infty)
&#36;&#36;
of which &#36;f( x)&#36; Is a fully established function, &#36;\mu&#36; Called position parameters,&#36;\sigma&#36; Called a measure parameter, and this sort of distribution group called a position-scale parameter Group</p>
<p>The normal distribution, the Cauchy distribution, the index distribution, is evenly distributed in this category.</p>
<p><strong>I am not sure if I am.&#36;\sigma=1&#36; is often called the position parameter, and&#36;\mu=0&#36; The current term is the scale parameter family, the position-scale parameter group being the combination of the two above</strong></p>
<p>His counterpart's no information a priori. &#36;\pi(\theta,\sigma)=\frac{1}{\sigma^{2&#125;&#125;&#36; </p>
<h4>General Uninformation Precursion</h4>
<p>The uninfomatic a priori of the general situation is the most common method of Jeffreys, because its extrapolation involves a lot of information about abstract algebra variations and Harr measurements, and we will only present the method below.</p>
<h5>Jeffreys has no information a priori.</h5>
<p>Assumptions for sample distribution &#36;&#123;f(x|\theta),\theta\in\Theta}&#36; Meet CR. &#36;\theta=(\theta_1,\cdots,\theta_p)&#36; Yes &#36;p&#36; Width vector. Set &#36;\boldsymbol{X}=(X_1,\cdots,X_n)&#36; From the whole. &#36;f(x|\theta)&#36; Simple sample taken.
When?&#36;\theta&#36; When no a priori information is available, Jeffreys uses the square root of the Fisher Info array in a row &#36;\theta&#36; ..no information a priori called</p>
<p> Jeffreys has no information to solve the problem.</p>
<ol>
<li>Write Parameters&#36;\theta&#36;logarithmic function
&#36;&#36;l(\boldsymbol{\theta}|\boldsymbol{x})=\ln\left[\prod_{i=1}^nf(x_i|\boldsymbol{\theta})\right]=\sum_{i=1}^n\ln f(x_i|\boldsymbol{\theta}).&#36;&#36;</li>
<li>Calculating Fisher Information Frame
&#36;I (\bardsymbol(theta}) =\left(I ij}(\bardsymbol(theta})\right)<em>{p\times p},\quad I</em>{ij}(\boldsymbol{\theta})=E_{\boldsymbol{X}\mid\boldsymbol{\theta&#125;&#125;\Big{-\frac{\partial^2l}{\partial\theta_i\partial\theta_j}\Big}\quad(i,j=1,\cdots,p).&#36;&#36;
<em>For a single parameter scenario Fisher Info array is&#36;1\times1&#36;Matrix</em>
&#36;&#36;I(\boldsymbol{\theta})=E_{\boldsymbol{X}|\boldsymbol{\theta&#125;&#125;\Big{-\frac{\partial^{2}l}{\partial\boldsymbol{\theta}^{2&#125;&#125;\Big}.&#36;&#36;
<strong>Take this down.&#36;E&#36;- Yeah.&#36;X&#36;It's good. There's got to be a sample in there.</strong></li>
<li>The square root of the column of the Calculating Info array as a priori without information
&#36;&#36;\pi(\theta)=\left[\det I(\theta)\right]^{1/2}&#36;&#36;
<em>For a single parameter situation</em>
&#36;&#36;\pi(\theta)=[I(\theta)]^{1/2}.&#36;&#36;</li>
</ol>
<p>Fisher's Information Volume Definition is
&#36;&#36;I(\theta)=E_\theta\left[\frac\partial{\partial\theta}\ln p(x;\theta)\right]^2&#36;&#36;
The form we're using above is a price equivalent that meets the notion of a detailed study of Fisher's information volume, which will give us a detailed description of the definition and the reasoning.
<strong>If you don't stress the number of samples, you should take only one sample, if the number of samples is not good.</strong></p>
<h5>Examples</h5>
<p>Set &#36;X=(X_1,\cdots,X_n)&#36; From the whole. &#36;N(\mu,\sigma^2)&#36; Simple sample taken. Remember&#36;\theta=(\mu,\sigma)&#36;Please. &#36;(\mu,\sigma)&#36; Joint without aforeword</p>
<p>The calculation logarithmic function is as if
&#36;&#36;l(\boldsymbol{\theta}|\boldsymbol{x})=-\frac n2\ln2\pi-n\mathrm{ln}\sigma-\frac1{2\sigma^2}\sum_{i=1}^n(x_i-\mu)^2.&#36;&#36;
The elements of the Fisher Info-Founding for Division
&#36;&#36;00begin{aligned}I 11(\bardsymbol{\theta}&amp;=E_{\boldsymbol{X}|\boldsymbol{\theta&#125;&#125;\Big{-\frac{\partial^2l(\boldsymbol{\theta}|\boldsymbol{x})}{\partial\mu^2}\Big}=\frac{n}{\sigma^2},\I_{22}(\boldsymbol{\theta})&amp;=E_{\boldsymbol{X}|\boldsymbol{\theta&#125;&#125;\Big{-\frac{\partial^2l(\boldsymbol{\theta}|\boldsymbol{x})}{\partial\sigma^2}\Big}=-\frac{n}{\sigma^2}+\frac{3}{\sigma^4}E\Big{\sum_{i=1}^n(X_i-\mu)^2\Big}=\frac{2n}{\sigma^2},\I_{12}(\boldsymbol{\theta})&amp;=I_{21}(\boldsymbol{\theta})=E_{\boldsymbol{X}|\boldsymbol{\theta&#125;&#125;\Big{-\frac{\partial^2l(\boldsymbol{\theta}|\boldsymbol{x})}{\partial\mu\partial\sigma}\Big}=E\Big{\frac{2}{\sigma^3}\sum_{i=1}^n(X_i-\mu)\Big}=0,\end{aligned}&#36;&#36;
计算Fisher信息阵和行列式的平方根有
&#36;&#36;\left.I(\boldsymbol{\theta})=\left(\begin{array}{cc}\frac n{\sigma^2}&amp;0\0&amp;\frac{2n}{\sigma^2}\end{array}\right.\right),\quad[\det I(\boldsymbol{\theta})]^{1/2}=\frac{\sqrt{2}n}{\sigma^2}.&#36;&#36;
由于广义先验可以自由调整常数 联合无信息先验为
&#36;&#36;\pi(\mu,\sigma)=1/\sigma^2,&#36;&#36;</p>
<p>Or this example, we can see.</p>
<p>When?&#36;\sigma&#36;When known, belonging to the position parameter family, no information a priori should read&#36;\pi(\theta)\equiv1&#36; The fact that the Felher information array was used to calculate Jeffreys' lack of information a priori is the same result.</p>
<p>When?&#36;\mu&#36;When known, the data should be \pi(\sigma) = \frac{1}{\sigma}\quad(\sigma)&gt;This is also the result of the fact that Jeffreys' lack of information was calculated using Fisher's Info array.</p>
<p>So when they were independent, the joint no-information aforema density was &#36;\pi(\sigma)=\frac{1}{\sigma}\sigma&gt;(0)
When they're not independent, no information is combined a priori density&#36;\pi(\mu,\sigma)=1/\sigma^2,&#36;</p>
<p>I can see that. <strong>No information a priori density is not unique</strong></p>
<p><strong>In fact, the impact of the different information a priori on the Bayesian inference is small.</strong></p>
<p><strong>The absence of information a priori is one of the most successful parts of Bayesian statistics.</strong></p>
<p><strong>Many of the estimates in classical statistics can be considered as some kind of anaesthesia without information.</strong></p>
<h3>Co-examining distribution</h3>
<p>It's a theoretical way of determining a priori, and in the case of known samples, for theoretical research purposes.</p>
<p>In fact, the distribution of the edges in the high-dimensional context is very difficult to calculate.</p>
<h4>Definitions</h4>
<p>Definitions&#36;\mathscr{F}&#36; Other Organiser &#36;\theta&#36; A priori distribution &#36;\pi(\theta)&#36; Composition of the distributed community. If you're willing to take it,&#36;\pi\in\mathscr{F}&#36; and sample values &#36;x&#36;,Performing Distribution&#36;\pi( \theta|x)&#36;  Still belongs to &#36;\mathscr{F}&#36; So, what do you say? &#36;\mathscr{F}&#36; A co-prospective distribution (conjugate prefix distribution family)</p>
<p>It's obvious that the a priori values of the distributional and sampled are relevant.</p>
<p>At the same time, the a priori distribution is for one parameter in the distribution, and it is important to understand the circumstances that contain multiple parameters.</p>
<h4>Examples of co-prospecting distributions</h4>
<p>Set&#36;X\sim B(n,\theta)&#36; </p>
<ul>
<li>If &#36;\theta&#36; Obedience evenly distributed &#36;U(0,1)&#36;, attests: &#36;\theta&#36; The posteriori distribution is the Beta distribution;</li>
<li>If &#36;\theta&#36; A priori distribution is the beta distribution &#36;Beta(a,b)&#36;, of which &#36;a,b&#36; Known, attests to: &#36;\theta&#36; The posteriori distribution is still the Beta distribution, i.e. &#36;\theta&#36; The co-prospecting distribution is the Beta distribution.</li>
</ul>
<p>Two questions in a row are really a question of a posteriori distribution.</p>
<ol>
<li><p>The following is the text:
Sample distribution is in two forms:
&#36;&#36;f(x|\theta)=\binom nx\theta^x(1-\theta)^{n-x}\quad(x=0,1,\cdots,n)&#36;&#36;
The aforecast distribution is evenly distributed &#36;\pi(\theta)=1&#36; The post-calculation distribution is as follows:
&#36;&#36;\pi(\theta|x)=\frac{\theta^x(1-\theta)^{n-x&#125;&#125;{\int_0^1\theta^x(1-\theta)^{n-x}\mathrm{d}\theta}&#36;&#36;
Calculating the points is the mathematical analysis of the Gamma function's points. Pass.
&#36;&#36;\int_0^1\theta^x(1-\theta)^{n-x}\mathrm{d}\theta=\frac{\Gamma(x+1)\Gamma(n-x+1)}{\Gamma(n+2)}.&#36;&#36;
So they get a posteriori distribution.
&#36;&#36;\pi(\theta|x)=\frac{\Gamma(n+2)}{\Gamma(x+1)\Gamma(n-x+1)}\theta^{(x+1)-1}(1-\theta)^{(n-x+1)-1}&#36;&#36;</p>
</li>
<li><p>The following is the text:
The sample distribution is unchanged, the pre-specification distribution is Beta, and the same principle is applied to the post-calculation distribution.
&#36;&#36;\pi(\theta|x)=\frac{\theta^{x+a-1}(1-\theta)^{n-x+b-1&#125;&#125;{\int_0^1\theta^{x+a-1}(1-\theta)^{n-x+b-1}\mathrm{d}\theta}&#36;&#36;
And we're going to calculate the points and bring in the results and we're going to have a back-up density.
&#36;&#36;\pi(\theta|x)=\frac{\Gamma(n+a+b)}{\Gamma(x+a)\Gamma(n-x+b)}\theta^{(x+a)-1}(1-\theta)^{(n-x+b)-1}&#36;&#36;
So we're sure it's a co-prospecting distribution.</p>
</li>
</ol>
<h4>Determine the distribution after co-checking</h4>
<p>We can think of a sample.&#36;X&#36;..the density of the edge&#36;f_{m}(x)&#36; and&#36;\theta&#36;It's not about that, which means he's a constant.
&#36;&#36;\pi(\theta|x)=\frac{f(x|\theta)\pi(\theta)}{f_m(x)}\propto f(x|\theta)\pi(\theta)&#36;&#36;</p>
<p>Definition: The kernel of the probability function is the part of the probability function that is only relevant to the parameters
*Eg. Sample Probability Functions&#36;f(x|\theta)&#36;- The core. - Yes.&#36;f(x|\theta)&#36;only with&#36;\theta&#36;Relevant part</p>
<p>For co-prospecting distributions, the following steps can be taken to test back density:</p>
<ol>
<li>Write sample probability function&#36;f(x|\theta)&#36;. . . . . . . . . and pre-density functions &#36;\pi(\theta)&#36; Nuclear</li>
<li>Using the formula at the end of the line, you give back-density cores, which are the mass of sample cores and a priori sum.</li>
<li>Add a regular constant factor and get a posteriori density.&#36;x&#36;(Refer to)</li>
</ol>
<p>How to add a regular constant factor: the distribution of the back-density of co-scoping is known. We can just go to the corresponding back-density function.</p>
<p>This method is generally only available for a priori distribution. </p>
<p>Because we know that the back check is the same as the a priori density function, and we know the distribution of the back check is easy to give constants.</p>
<p>When a non-cooperative aforesee, but a later check, we know that it's a core of a common distribution, and it can be given.</p>
<p>In other cases, we can't determine the constant factor, but we need to calculate it according to the basic formula of the later check.</p>
<p>Specific examples can be found in the section “Accounting for later distribution” of this paper</p>
<h3>Multi-layer a priori (phased a priori)</h3>
<h4>Basic thinking</h4>
<p>If the parametric parameter is not certain, then the second is called superprecursor, and if it's still difficult to determine whether we can continue the pre-precursion, the new a priori is called multi-layer a priori based on a priori and superpresence.</p>
<p>The multilayered aforethought is as follows:
First level a priori
&#36;F_1={\pi_1(\theta|\lambda):\pi_1&#36;. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .&#36;\lambda\in\Lambda}&#36;I'm not sure.
of which &#36;\Lambda&#36; As Superparameter &#36;\lambda&#36; range of values to be taken from, and λ unknown</p>
<p>Second floor a priori.
&#36;\lambda&#36;The a priori distribution is&#36;\pi(\lambda)&#36; Without any unknown parameters</p>
<p>The two levels of the calculation code are pre-established
&#36;&#36;\pi(\theta)=\int_{\Lambda}\pi_1(\theta|\lambda)\pi_2(\lambda)\mathrm{d}\lambda=\int_{\Lambda}\pi(\theta,\lambda)\mathrm{d}\lambda&#36;&#36;</p>
<p>The core is to get our standard a priori distribution through multilayered compound a priori, and then use this standard a priori distribution for the Bayesian statistical extrapolation.</p>
<h4>Pre-specify the stratification</h4>
<p>To study the a priori distribution of the failure rate, we first consider the failure rate to be a priori.&#36;U(0,1)&#36; But the failure rate is low, so this a priori is not a reasonable one, so it's a choice to use multilayer a priori to study it.</p>
<p>We think the failure rate is... &#36;U(0,\lambda)&#36; of which super-parameters&#36;\lambda&#36;A priori as&#36;U(0.1,0.5)&#36; </p>
<p>Calculates a priori for regularization.
&#36;&#36;\pi(\theta)=\int_{\Lambda}\pi_{1}(\theta|\lambda)\pi_{2}(\lambda)\mathrm{d}\lambda=\frac{1}{0.5-0.1}\int_{0.1}^{0.5}\lambda^{-1}I_{[0,\lambda]}(\theta)\mathrm{d}\lambda&#36;&#36;
of which&#36;I&#36;is a specter function that calculates a priori results in several different cases.</p>
<p>&#36;&#36;\begin{aligned}&amp;(a)\text{bet }0&lt;\theta&lt;..the time of the ..&amp;\pi\left(\theta\right)=\frac{1}{0.4}\int_{0.1}^{0.5}\lambda^{-1}d\lambda=2.5\ln5\approx4.0236;\end{aligned}&#36;&#36;
&#36;&#36;\begin{aligned}&amp;\text{b) \leqslant\theta&lt;0.5\text{t}, \\&amp;\pi\left(\theta\right)=\frac{1}{0.4}\int_{\theta}^{0.5}\lambda^{-1}d\lambda=2.5\left(\ln\left(0.5-\ln\theta\right)\approx-1.7329-2.5\ln\theta\right);\end{aligned}&#36;&#36;
&#36;&#36;(c) \\theta\geqslant0.5, \\theta)=&#36;0.00</p>
<p>All right, all right.
&#36;&#36;\left.\pi(\theta)=\left{begin{array}ll}4.0236,&amp;0&lt;\theta&lt;0.1,\-1.7329-2.5\ln\theta,&amp;0.1\leqslant\theta&lt;0.5,\0,&amp;0.5\leqslant\theta&lt;I'm sorry, I'm sorry, but I'm sorry.
Just to meet the normative requirements of the probabilistic density function</p>
<h4>A priori ideological characteristics of the hierarchy</h4>
<p>A priori stratification model allows for the conversion of relatively complex situations into a series of cartridges when modelling
As we have seen in the preceding example, although we are still making a stratification a priori norm, the stratification Beyers model allows us to decompose relatively complex situations into a series of simple situations that make modelling less difficult. <strong>Sometimes our norms are complicated to the point where they don't even have a visible expression, but the layers of Bayes still help us to model.</strong></p>
<p>Another feature of the stratification a priori model is that it is easy to calculate.
Sometimes the posterior density is too complicated, which makes it difficult to calculate him and some of his digital features, and leads to the Beyers statistical inferences that statistical decisions are difficult to make.
But if we use the hiercurate of multi-layer structures to indicate laterals, even if the outer layer is not represented by the expression, we can calculate it using methods like MCMC.</p>
<h3>Co-prospecting of the index distribution</h3>
<p>The necessary description of the mathematical statistics of the index distribution is given.<a href="/en/blog/2023/03/18/mathematical-statistics-notes/">Mathematical statistics</a> and the “Indicate distribution” section</p>
<h5>Co-a prior distribution of single-parameter index distributions</h5>
<p>If &#36;X|\theta&#36; The distribution is of an indexed group (samples distribution): &#36;x=(x_1,\cdots,x_n)&#36; The sample is i.i.d. The apparent function can be expressed as
&#36;&#36;
l(\theta|x)\propto[h(\theta)]^n\exp\left{\sum_{i=1}^nt(x_i)\phi(\theta)\right}
&#36;&#36;
 That is, the index distribution family parameter &#36;\theta&#36; The Co-Protesters &#36;\Pi&#36; Yes &#36;\pi(\theta)&#36; Is:
(Use of co-prospecting density functions to give the same form as the density functions of the sample)
&#36;&#36;
\pi(\theta)\propto[h(\theta)]^{\gamma}\exp&#123;&#123;\tau\phi(\theta)&#125;&#125;
&#36;&#36;
of which super-parameters &#36;\gamma,\tau&#36; Known.</p>
<h5>Co-prospective distribution of the two parameter index distributions</h5>
<p>If &#36;x=(x_1,\cdots,x_n)&#36; For the i.i.d. sample,&#36;X&#36; , which is an index distribution family with two parameters, is an approximation &#36;l(\theta,\varphi|x)&#36; May be read:
&#36;&#36;
l(\theta,\varphi|x)\propto[h(\theta,\varphi)]^n\exp\left{\sum[t(x_i)\phi(\theta,\varphi)]+\sum[u(x_i)\chi(\theta,\varphi)]\right}
&#36;&#36;
 Parameters &#36;(\theta,\varphi&#36; A priori. &#36;\Pi&#36; Is this the following:
&#36;&#36;
\pi(\theta,\varphi)\propto[h(\theta,\varphi)]^\gamma\exp\left{\alpha\phi(\theta,\varphi)+\beta\chi(\theta,\varphi)\right}
&#36;&#36;
of which super-parameters &#36;\gamma,\alpha,\beta&#36; Known</p>
<h3>Multiparameter model</h3>
<h4>The idea of a multiparametric model</h4>
<p>The idea of solving the backsizing of a given parameter has been described before.</p>
<p>The basic idea is to give a priori and then use formulas to make a later calculation.</p>
<p>In a large number of practical questions, it is common to have multiple unknown parameters, and we can give a back-sizing by a method similar to the single parameter.</p>
<p>Sometimes we focus on only a fraction of all the parameters, and at this point, to get the marginal back-density of the parameters that we're interested in, we need to get a fraction of the whole back-density of the parameters.</p>
<h4>Examples of multiparametric models</h4>
<p>The government has also been able to provide information on the situation.&#36;\underline{x}=(x_1,\cdots,x_n)&#36; As Sample Capacity with&#36;n&#36; i.i.d sample, unknown parameter &#36;(\theta,\sigma^2)&#36; for a 2-D random variable, if&#36;(\theta,\sigma^2)&#36; A priori density is &#36;\pi(\theta,\sigma^2)\propto\frac1{\sigma^2}&#36;,</p>
<p>&#36;(1)\left(\theta,\sigma^{2}\right)|x&#36; Joint Postdensity Function &#36;\pi(\theta,\sigma^2|x)&#36; Is:
&#36;&#36;
\pi(\theta,\sigma^2|x)\propto(\sigma^2)^{-\frac{\gamma+1}2-1}\exp\left{-\frac1{2\sigma^2}\left(S+n(\bar{x}-\theta)^2\right)\right}
&#36;&#36;
of which &#36;\gamma=n-1,:S=\sum_{i=1}^{n}\left(x_{i}-\bar{x}\right)^{2},:\bar{x}=\frac{1}{n}\sum_{i=1}^{n}x_{i}&#36; </p>
<p>(2) &#36;\theta&#36; The marginal lateral distribution is:</p>
<p>&#36;&#36;
t=\frac{\theta-\bar{x&#125;&#125;{s/\sqrt{n&#125;&#125;\sim t(\gamma)
&#36;&#36;
of which &#36;s^2=\frac1{n-1}S&#36;, &#36;t(\gamma)&#36; It's freedom. &#36;\gamma&#36; Yes. &#36;t&#36; Distribution</p>
<p>(3) Difference &#36;\sigma^2&#36; Other Organiser
&#36;&#36;00\
\pi (sigma^)&amp; =\int\pi(\theta,\sigma^{2}|x)d\theta   \
&amp;=\int_{-\infty}^{+\infty}\left(\sigma^2\right)^{-\frac{\gamma+1}2-1}\exp\left{-\frac1{2\sigma^2}\left(S+n(\theta-\bar{x})^2\right)\right}d\theta  \
&amp;\propto\left(\sigma^{2}\right)^{-\frac{\gamma}{2}-1}\exp\left{-\frac{S}{2\sigma^{2&#125;&#125;\right} \
&amp;\times\int_{-\infty}^{+\infty}\left(\frac{2\pi\sigma^2}n\right)^{-\frac12}\exp\left{-\frac1{2\sigma^2}\cdot n(\theta-\bar{x})^2\right}d\theta  \
&amp;== sync, corrected by elderman ==
I'm sorry, I'm sorry.
The equation to the right is the reverse of Gamma distribution core, so the difference is &#36;\sigma^2&#36; The after-check margin distribution is&#36;IGamma(\frac{\gamma}{2},\frac{S}{2})&#36;</p>
<h2>Calculation of the posteriori distribution</h2>
<h3>Calculation of the posteriori distribution</h3>
<h4>Theory Introduction</h4>
<p><strong>The calculations of the post-spectrum distribution are the basis for all the Beyers statistical inferences.</strong></p>
<p>The formula for the posteriori distribution is:
&#36;&#36;\pi(\theta|x)=\frac{h(x,\theta)}{m(x)}=\frac{f(x|\theta)\pi(\theta)}{\int_{\Theta}f(x|\theta)\pi(\theta)\mathrm{d}\theta}&#36;&#36;
Which means we need to make a score for the edge distribution.</p>
<p>This step is not always a good measure.</p>
<h4>Common Post-Assessment Method</h4>
<p>We have three ways of calculating the posteriori distribution.</p>
<ul>
<li>General calculation based on the Bayesian formula: definition of a later distribution</li>
<li>Simplified calculation method based on distribution nuclear</li>
<li>Calculation methods based on adequate statistical data
There are two types of pre-selection that are more common.</li>
<li>No information a priori distribution</li>
<li>Co-examining distribution</li>
</ul>
<h3>Examples of calculations for the posteriori distribution</h3>
<h4>Example 1</h4>
<p>We're still counting the front set.&#36;X\sim B(n,\theta)&#36;   &#36;\theta&#36; A priori distribution is the beta distribution &#36;Beta(a,b)&#36;
Sample density core&#36;\theta^x(1-\theta)^{n-x}&#36;  The pre-density core is &#36;\theta^{a-1}(1-\theta)^{b-1}&#36;
So the posterioris content is satisfied.
&#36;&#36;\pi(\theta|x)\propto f(x|\theta)\pi(\theta)\propto\theta^{x+a-1}(1-\theta)^{n-x+b-1}.&#36;&#36;
Apparently, he's also a beta distribution core, with a positive factor to supplement Beta distribution.
&#36;&#36;\pi(\theta|x)=\frac{\Gamma(n+a+b)}{\Gamma(x+a)\Gamma(n-x+b)}\theta^{(x+a)-1}(1-\theta)^{(n-x+b)-1}&#36;&#36;</p>
<h4>Example 2</h4>
<p>Set&#36;X&#36;Subject to normal distribution&#36;N(\theta,\sigma^{2})&#36; The difference is known, but the average is unknown.&#36;\theta&#36;The a priori is...&#36;N(\mu,\tau^2)&#36; All parameters are known.
&#36;&#36;\pi(\theta|x)\propto f(x|\theta)\pi(\theta)\propto\exp\left{-\frac{1}{2}\Big[\frac{(x-\theta)^2}{\sigma^2}+\frac{(\theta-\mu)^2}{\tau^2}\Big]\right}.&#36;&#36;
You!
&#36;&#36;\rho=\frac{1}{\tau^{2&#125;&#125;+\frac{1}{\sigma^{2&#125;&#125;=\frac{\sigma^{2}+\tau^{2&#125;&#125;{\sigma^{2}\tau^{2&#125;&#125;.&#36;&#36;
Simpliciting the squared construction
&#36;&#36;\pi(\theta|x)\propto\exp\Big{-\frac\rho2\Big[\theta-\frac1\rho\Big(\frac\mu{\tau^2}+\frac x{\sigma^2}\Big)\Big]^2-\frac{(x-\mu)^2}{2(\sigma^2+\tau^2)}\Big}\propto\exp\Big{-\frac{\rho}{2}\Big[\theta-\frac{1}{\rho}\Big(\frac{\mu}{\tau^{2&#125;&#125;+\frac{x}{\sigma^{2&#125;&#125;\Big)\Big]^{2}\Big}&#36;&#36;
I can see that. &#36;N(\mu(x),\eta^{2})&#36; ..to add a regularised factor to the core to obtain a post-feasibility.
&#36;&#36;\pi(\theta|x)=\frac{1}{\sqrt{2\pi}\eta}\exp\Big{-\frac{1}{2\eta^2}[\theta-\mu(x)]^2\Big}.&#36;&#36;
of which
&#36;&#36;\begin{aligned}\mu(x)&amp;=\frac{1}{\rho}\left(\frac{\mu}{\tau^2}+\frac{x}{\sigma^2}\right)=\frac{\sigma^2}{\sigma^2+\tau^2}\mu+\frac{\tau^2}{\sigma^2+\tau^2}x,\\eta^2&amp;♪ I'm not gonna let you go ♪
When the sample is distributed to the known normal distribution of the variance, the mean parameter is&#36;\theta&#36;The co-prospecting distribution is normal.</p>
<h4>Example 3</h4>
<p>And here we give some examples of co-prospecting distribution, not counting, but simply giving some examples.</p>
<ul>
<li>Sample distribution is Porpine distribution&#36;P(\theta)&#36;And then, if the density is in line with the gamma distribution, then the later distribution is in the gamma distribution, which is the co-prospecting distribution group is in the gamma distribution.</li>
<li>The sample distribution is gamma. &#36;\Gamma(r,\lambda)&#36; of which&#36;r&#36;Known&#36;\lambda&#36;The co-prospecting distribution is the gamma distribution.</li>
<li>The index distribution is a special case of gamma distribution, so the sample distribution is as follows:&#36;Exp(\lambda)&#36;♪ When ♪  &#36;\lambda&#36;The co-prospecting distribution is the gamma distribution.</li>
<li>The difference is when the sample distribution is the known normal distribution of the average.&#36;\sigma^{2}&#36;The co-prospecting distribution is anti-Gamma distribution.</li>
</ul>
<h3>Brief summary</h3>
<p>There's a fixed idea to be made of the distributional community. </p>
<ol>
<li>Writes the core of the sample probability function</li>
<li>The probability function for selecting a nuclear sample has a priori distribution of the same core (in similar forms) as a co-prospecting distribution, and thus a co-prospecting distribution.</li>
</ol>
<p><strong>The nuclear of the probability density of the sample, which contains&#36;\theta&#36;, but at this point we're distributed&#36;X&#36;It's given as a variable if we put&#36;\theta&#36;It's a self-variant. He's supposed to be another distribution core.</strong></p>
<p>I can see that.</p>
<ul>
<li>Co-examining distribution is easy to calculate and then distribute.</li>
<li>There are many parameters for a posteriori distribution that can be explained very well.</li>
</ul>
<p>For example, in the case of the normal distribution, the average after-checking was determined by a priori and sample, and as the amount of sample information increased, the role of the precursor was weakened by nature.</p>
<p>We then turn to the section on "Assumption and sufficiency" of a method that uses sufficient statistical data and the section on "Beyers statistical calculations" of a numerical method.</p>
<h3>Post-scenes distribution and adequacy</h3>
<h4>Sufficiency in mathematical statistics</h4>
<p>The adequacy of statistics is one of the most important concepts in mathematical statistics, which intuitively is defined as the statistical amount of information that is not lost.</p>
<p>Theoretically defined.&#36;T(x)=t&#36; , and then you can have it. &#36;X&#36;Conditions distribution and parameters&#36;\theta&#36; Not relevant</p>
<p>The best way to judge is to use the factor to decompose the theorem.</p>
<h4>Sufficiency in Bayesian statistics</h4>
<p>Actually, the way we define it is perfectly consistent.
Intuitive definition: use of sample distribution and statistics&#36;T(x)&#36;The distribution calculated afterward distribution is consistent</p>
<p>Theory definition: &#36;\mathbf{x}=(x_1,\cdots,x_n)&#36; It's from the density function.&#36;p(x|\theta)&#36;A sample of,&#36;T=T(x)&#36;It's statistical. It's density function is&#36;q(t|θ)&#36;It's not working.&#36;\mathbf{H}={\pi(\theta)}&#36;Yes.&#36;\theta&#36;, which is a priori distributed,&#36;\mathrm{T( x) }&#36;Yes&#36;\theta&#36;A sufficient statistical amount is most likely to be distributed a priori to any given&#36;\pi(\theta)\in H&#36; Yes.
&#36;&#36;\pi(\theta\mid\mathrm{T}(\mathbf{x}))=\pi(\theta\mid\mathbf{x})&#36;&#36;</p>
<p>What's the theorem for? — Computation of simplified post-scenary distribution</p>
<p>If we determine that a statistical amount is sufficient, then we can calculate the post-censor distribution by using a full statistical amount, not by using a sample distribution.</p>
<p>The way to judge whether this is a full measure or the exact same factor decomposition theorem.
&#36;&#36;L(\theta)=\prod_{i=1}^nf(x_i;\theta)=h(x_1,x_2,\cdots,x_n)g(T(x_1,x_2,\cdots,x_n);\theta)&#36;&#36;
The meaning of statistics has not changed.</p>
<h4>Application of adequacy in Bayesian statistics</h4>
<p>Discard pre-selection&#36;\pi(\mu)&#36; Calculate normal distribution using full statistical volume&#36;\mathbb{N}(\mu,1)&#36;Medium Parameters&#36;\mu&#36; Post-examining distribution
We know the numbers.&#36;\overline{x}&#36; is the full statistical amount of the parameter and
&#36;&#36;\overline{x}\sim N(\mu,\sigma^2/n)&#36;&#36;
Calculates the post-examining distribution using the formula for the post-examining distribution
&#36;&#36;\pi(\theta\mid\overline{x})=\frac{\exp\left{-\frac{n(\overline{x}-\theta)^2}{2\sigma^2}\right}\pi(\theta)}{\int_{-\infty}^{\infty}\exp\left{-\frac{n(\overline{x}-\theta)^2}{2\sigma^2}\right}\pi(\theta)d\theta}&#36;&#36;
I can see that we're actually the ones who put it in the original.&#36;f(x|\theta)&#36; It's become... &#36;\bar{x}&#36; density function
Nothing else has changed. His role is to circumvent.&#36;\prod_{i=1}^nf(x_i;\theta)&#36; The complex operations that you bring.</p>
<p>Of course we can extend to the situation of two parameters.
Set the overall distribution to a normal distribution &#36;N(\mu,\sigma^2)&#36;,Samp&#36;\mathbf{x}=(x_1,...,x_n)&#36; i.i.d sample, average &#36;\mu&#36; Difference&#36;\sigma^2&#36; Unknown, calculated after-scenes distribution using full statistical data
Give
&#36;&#36;\overline{x}=\frac1n\sum_{i=1}^nx_i\quad\quad Q=\sum_{i=1}^n\left(x_i-\overline{x}\right)^2&#36;&#36;
Easy to know, two-dimensional statistics.&#36;(\overline{x},Q)&#36; Yes.&#36;(\mu,\sigma^2)&#36; ♪ and the full statistical ♪
&#36;&#36;\bar{x}\sim N(\mu,\sigma^2/n),Q/\sigma^2\sim\chi^2(n-1)&#36;&#36;
The density function is as follows:
&#36;&#36;00\
&amp;p(\bar{x}\mid\mu,\sigma^2) =\frac{\sqrt{n&#125;&#125;{\sqrt{2\pi\sigma&#125;&#125;\exp\left{-\frac{n(\overline{x}-\mu)^2}{2\sigma^2}\right}  \
&amp;p(Q|\mu,\sigma^2) =\frac1{\Gamma(\frac{n-1}2)(2\sigma^2)^{\frac{n-1}2&#125;&#125;Q^{\frac{n-3}2}\exp{-\frac Q{2\sigma^2&#125;&#125;
\end{aligned}&#36;&#36;
两者本身独立 得到联合分布有
&#36;&#36;\begin{aligned}
p(\bar{x},Q\mid\mu,\sigma^2)&amp; =\frac{\sqrt{n&#125;&#125;{\sqrt{2\pi\sigma&#125;&#125;\frac1{\Gamma(\frac{n-1}2)(2\sigma^2)^\frac{n-1}2}Q^\frac{n-3}2  \
&amp;\times\exp{-\frac1{2\sigma^2}[Q+n(\overline{x}-\mu)^2]}
\end{aligned}&#36;&#36;
计算后验分布
&#36;&#36;=(\ \ \ = = = = = = = = = = = = = = = = = = = \ \ \ \ \ \ } } } } } } } } } } } } } } } = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = ^ \ ^ ^ ^ ^ \ ^ ^ ^ ^ ^ ^ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ } } } } } } ^ ^ ^ ^ ^ } } } } } \ \ \ \ \ \
It's the same thing we did with the sample.</p>
<h3>Post-examining distribution of multiple samples</h3>
<p>All the examples we've had before are sample samples.&#36;1&#36;(a) A presentation, including the formula given in the preamble;</p>
<p>But in fact, the post-speciator distribution of multiple samples is the most common case in practice; we've been diluting it to reduce the difficulty of thinking, so we have learned the techniques to explain how the post-speciation distribution is calculated in multi-samples and how the previous methods will change.</p>
<h4>Multi-sampling post-calculation method</h4>
<p>Original Post-Specific Formula
&#36;&#36;\pi(\theta|x)=\frac{h(x,\theta)}{m(x)}=\frac{f(x|\theta)\pi(\theta)}{\int_{\Theta}f(x|\theta)\pi(\theta)\mathrm{d}\theta}&#36;&#36;</p>
<p>After taking multiple i.i.d samples, we give the combined probability density of multiple samples (the amount of the individual probability density, but the change in the sample is to be distinguished).
&#36;&#36;f\left(\boldsymbol{x}|\theta\right)=\prod_{i=1}^{n} f(x_{i}|\theta)&#36;&#36;</p>
<p>Use&#36;f\left(\boldsymbol{x}|\theta\right)=\prod_{i=1}^{n} f(x_{i}|\theta)&#36;Replace the original.&#36;f(x|\theta)&#36; The calculation using the Bayesian hiercurate formula is the posteriori distribution of multiple samples.</p>
<p>It's obvious that the direct fraction method can be fully applied to the nuclear-based approach, which requires a fresh study of the nuclear, without a memory conclusion. </p>
<p>As for the method of full statistical counts, it exists to solve the problem.&#36;\prod_{i=1}^nf(x_i;\theta)&#36; The complex calculations that come with it are the most appropriate for multiple samples.</p>
<h4>Nuclear methods and multiple samples</h4>
<p>If you further assume &#36;X_1,\cdots,X_n&#36; i.i.d. &#36;\sim N(\theta,\sigma^2),\sigma^2&#36; The blogger says:&#36;\theta\sim N(\mu,\tau^2)&#36;Try &#36;\theta&#36; The posteriori density. </p>
<p>Because of the sample. &#36;X=(X_1,\cdots,X_n)&#36; And the combined density is...
&#36;&#36;00\
\left (\bardsymbol{theta\right)&amp; =(2\pi\sigma^2)^{-n/2}\exp\left{-\frac1{2\sigma^2}\sum_{i=1}^n(x_i-\theta)^2\right}  \
&amp;\propto\exp\left{-\frac1{2\sigma^2}\Big[\sum_{i=1}^n(x_i-\bar{x})^2+n(\bar{x}-\theta)^2\Big]\right} \
&amp;\propto\exp\left{-\frac{n(\bar{x}-\theta)^2}{2\sigma^2}\right},
\end{aligned}&#36;&#36;</p>
<p>And we can use the nuclear method to give the back-density function in the case of normal a priori.
&#36;&#36;\pi(\theta|\boldsymbol{x})=\frac1{\sqrt{2\pi}\eta_n}\exp\left{-\frac1{2\eta_n^2}[\theta-\mu_n(\boldsymbol{x})]^2\right},&#36;&#36;
of which
&#36;&#36;\mu_n(\boldsymbol{x})=\frac{\sigma^2/n}{\sigma^2/n+\tau^2}\mu+\frac{\tau^2}{\sigma^2/n+\tau^2}\bar{x},&#36;&#36;
&#36;&#36;\eta_n^2=\frac{\tau^2\cdot\sigma^2/n}{\sigma^2/n+\tau^2}=\frac{\sigma^2\tau^2}{n\tau^2+\sigma^2}.&#36;&#36;
Actually, you can replace the results of the previous calculations.&#36;\sigma^2=\frac{\sigma^2}n,x=\overline{x}&#36; And that's what you get.</p>
<h4>Full statistical and multiple samples</h4>
<p>In fact, the method of post-quantifiable calculations can be superimposed with the method of the nuclear calculations, and then a little more streamlined. </p>
<p>For normal aggregate &#36;N(\theta,1)&#36; Three observations, with the specific observations of 2, 4, 3. If the a priori distribution of the thorium is normal &#36;N(3,1)&#36;Please. &#36;\theta&#36; Post-density</p>
<p>If we do it in a normal nuclear way, then we can consider a full measure of the three samples that will eventually be a very long, very difficult nuclear to calculate.</p>
<p>We know. &#36;\bar{x}&#36; It's a full statistical amount of normal distribution, easily visible.
&#36;&#36;\bar{x}\sim N(\theta,1/3)&#36;&#36;
His observation is 3.</p>
<p>So the shape-distorting sample is
&#36;&#36;e^{-\frac{(3-\theta)^{2&#125;&#125;{\frac{2}{3&#125;&#125;}&#36;&#36;
The pre-test is...
&#36;&#36;e^{-\frac{(3-\theta)^{2&#125;&#125;{2&#125;&#125;&#36;&#36;
So the posterioris check is
&#36;&#36;e^{-\frac{(3-\theta)^{2&#125;&#125;{\frac{6}{7&#125;&#125;}&#36;&#36;
Post-check is still a normal distribution</p>
<h2>Beyes formula, full probability formula, link between conditional probability formula</h2>
<p>First, we need to look at how these formulas are extrapolated.</p>
<p>The most important is the formula of total probability, which is one of the applications of the idea of classification discussion, described as:
&#36;&#36;P(A)=\sum_nP(A\mid B_n)P(B_n).&#36;&#36;
It's natural that he doesn't have to consider his proof.</p>
<p>Now we'll consider the probability formula.
&#36;&#36;P(A|B)=\frac{P(AB)}{P(B)}=\frac{P(B|A)P(A)}{P(B)}&#36;&#36;
The first step is the most fundamental notion of probability, the second is the application of natural probabilistic equations, which take the following full form:
&#36;&#36;P(A B)=P(A|B)P(B) , P(AB)=P(B|A)P(A) ,P(B|A)P(A)=P(A|B)P(B)&#36;&#36;</p>
<p>Then we can naturally get the Bayesian formula.
&#36;&#36;P(A\mid B)=\frac{P(A)P(B\mid A)}{P(B)}&#36;&#36;
Get the full Bayesian formula. (The denominator generally needs to be formulated in a simple manner using full probability)</p>
<p>In the form of a formula,<strong>The Bayesian formula is not derived from the probability of a condition, but from the probability of a condition itself.</strong> It's not natural that there is no need to name a formula alone, but as a basic inference, whether the Beyers formula really makes a separate and important formula or its mathematical thinking.</p>
<p>The basic idea of the full probability formula is the partitionist idea of classification, which naturally implies the sequence of events,&#36;B&#36; It happens first. &#36;A&#36; That's why it happened. &#36;P(A|B)&#36;</p>
<p>The idea of a probabilistic formula is the most basic cause and consequence. We start with it.&#36;B&#36;Middlely extrapolate the result&#36;A&#36; - The probability.</p>
<p>The Bayesian formula turns the original causal thinking, which we observe.&#36;B&#36; The result of the incident is that the police were not able to report the incident.&#36;A&#36; It's unknown. He can use normal causality.&#36;P(A)P(B\mid A)&#36;  Come on, push back.</p>
<p>&#36;A&#36;For that reason, there was a priori in the initial study.&#36;P(A)&#36; Now we have observations. &#36;P(B|A)&#36; And then, after updating the reasoning behind it, it's the Beyers' idea that we constantly revise the initial probability to get more real, and that's the probability of new causes that are in line with the current reality.&#36;P(A|B)&#36;</p>
