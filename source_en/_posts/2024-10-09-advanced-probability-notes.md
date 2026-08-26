---
title: 'Advanced Probability: Probability Spaces, Random Variables, and Measure-Theoretic Foundations'
title_zh: 高等概率论：概率空间、随机变量与测度基础
date: 2024-10-09 21:29:48 +0800
categories:
- Data Science
- Probability & Statistical Foundations
tags:
- Statistics
- Probability
- Measure Theory
author: Hyacehila
mathjax: true
hidden: true
excerpt: Covers probability spaces, random variables, measure-theoretic foundations, convergence, independence, conditional
  expectation, and related theory.
description: Covers probability spaces, random variables, measure-theoretic foundations, convergence, independence, conditional
  expectation, and related theory.
excerpt_zh: 整理概率空间、随机变量、测度基础、收敛性、独立性、条件期望和相关概率理论。
permalink: /blog/2024/10/09/advanced-probability-notes/
lang: en
translation_key: 2024-10-09-advanced-probability-notes
translation_status: machine
translation_source_hash: c51be6054b3b48910ccd0030b0c6bae1256f94dc9d34e95070d59d130dab80e6
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<h2>Probability space versus random variables</h2>
<p>The high probability theory study of a system of probabilities for justice is also an important step towards modern probabilities, the basic development of which is as follows:</p>
<ol>
<li>Classical definition of probability (two broad)</li>
<li>Introduction of the theory of aggregation and measurement</li>
<li>Establishment of a just structure</li>
</ol>
<p>This chapter, which complements basic mathematical knowledge in complete probabilistic theory, as well as the creation of our basic probabilistic systems, will include some basic mathematical knowledge, as well as precise mathematical definitions of the most basic concepts in probabilistic theory, probability space and random variables.</p>
<h3>&#36;\sigma&#36;Algebra</h3>
<p>In order to introduce the probabilistic theory of probabilism, we begin by adding knowledge of algebra.</p>
<p>Definitions: Clusters, also referred to as aggregations, are the sum of their composition</p>
<p>Definition: All in space&#36;\Omega&#36;The upper grouping of closed transaction is called &#36;\pi&#36; Category</p>
<p>Definitions: If cluster&#36;\mathcal{A}&#36; Satisfied</p>
<ul>
<li>&#36;\Omega \in \mathcal{A}&#36; (&#36;\phi \in \mathcal{A}&#36;)</li>
<li>&#36;\text{若}A\in \mathcal{A}\text{则}A^{c}\in \mathcal{A}&#36;</li>
<li>&#36;\text{若}A_1,A_2,\ldots,A_n\in \mathcal{A},\text{则}\sum_{i=1}^nA_i\in \mathcal{A}&#36;
otherwise referred to as cluster&#36;\mathcal{A}&#36;  An algebra.</li>
</ul>
<p>From definition: <strong>Algebra is &#36;\pi&#36; category, and &#36;\pi&#36; Classes may not be algebras</strong></p>
<p>Definitions: Establishment&#36;F&#36;is cluster and satisfied</p>
<ul>
<li>&#36;\Omega \in F&#36; (&#36;\phi \in F&#36;)</li>
<li>&#36;\text{若}A\in F\text{则}A^{c}\in F&#36;</li>
<li>&#36;\text{若}A_1,A_2,\ldots\in F,\text{则}\sum_{i=1}^{\infty} A_i\in \mathcal{A}&#36;
Name&#36;F&#36; It's one.&#36;\sigma&#36;Algebra</li>
</ul>
<p><strong>We changed the definition from a limited and closed one to a closed one, so&#36;\sigma&#36;Algebra must be an algebra.</strong></p>
<p>We can give.&#36;\sigma&#36;The two characteristics of algebra, they can prove it in combination with the definition.</p>
<ul>
<li>&#36;\sigma&#36;And the algebra is...&#36;\sigma&#36;Algebra</li>
<li>&#36;\sigma&#36;Algebra doesn't have to be.&#36;\sigma&#36;Algebra Algebra</li>
</ul>
<p>Definitions: Establishment&#36;\mathcal{A}&#36;Yes.&#36;\Omega&#36;A subset,&#36;F&#36;Yes.&#36;\sigma&#36;Algebra if</p>
<ul>
<li>&#36;\mathcal{A}\in\mathcal{F}&#36;</li>
<li>&#36;任意包含A的\sigma 代数F^{\prime},均有F\subset F^{\prime}&#36;
Name&#36;F&#36;Yes.&#36;A&#36;Generated&#36;\sigma&#36;Algebra&#36;F=\sigma(A)&#36;</li>
</ul>
<p>For understanding generation&#36;\sigma&#36;The algebra concept, we can give you two things.</p>
<ul>
<li>&#36;\sigma(A)&#36;Yes. &#36;A&#36; Minimum&#36;\sigma&#36;Algebra</li>
<li>&#36;\sigma(A)&#36;is all contained &#36;A&#36; Yes.&#36;\sigma&#36;Algebra surrender</li>
</ul>
<p>It's generated from the whole containment.&#36;\sigma&#36;Algebra is called the Borel algebra. He'll be the most common after us when we study random variables.&#36;\sigma&#36;Algebra</p>
<h3>&#36;\pi - \lambda&#36; Theorem</h3>
<p>Definition: referred to as cluster&#36;F&#36;Yes.&#36;\lambda&#36;Category</p>
<ul>
<li>&#36;\Omega\in F&#36;</li>
<li>&#36;对任意A,B\in F 且A\subset B,则 B/A\in F&#36;</li>
<li>&#36;A_{1}A_{1},A_{2},\ldots\in F A_{n}\uparrow A=\cup_{i=1}^{\infty}A_{i},则A\in F&#36;</li>
</ul>
<p>Obviously.</p>
<ul>
<li>&#36;\sigma&#36;Algebra is&#36;\lambda&#36;Category</li>
<li>&#36;\lambda&#36;Class-to-Current Operations Closed</li>
</ul>
<p>Theorem: We give theorem easily.
&#36;&#36;集类F是\sigma代数\Longleftrightarrow F是\pi类且是\lambda类&#36;&#36;</p>
<p>Give me the most important part of this section. &#36;\pi - \lambda&#36; Theorem</p>
<p>If there are two assembly categories&#36;P&#36;and&#36;L&#36;, where&#36;P&#36;It's one.&#36;\pi&#36;The system (i.e. closed aggregates with limited delivery)&#36;L&#36;It's one.&#36;\lambda&#36;- Systems (incorporated in empty collections, for patches and for clusters that do not intersect and are closed) and&#36;P&#36; Yes.&#36;L&#36; Subset, then,&#36;P&#36; Generated&#36;\sigma&#36;- Algebra.&#36;\sigma(P)&#36;Yes.&#36;L&#36;Subset.</p>
<h3>Probability measure</h3>
<p>Starting with this section, we start with the probabilistic theory of justice, and we start with probabilistic measurements and probability space.</p>
<p>Definition: The set function is a map from the collection to the actual number</p>
<p>&#36;P&#36;  It's defined as  &#36;\sigma&#36;Algebra &#36;\mathcal{F}&#36;  The probability measure, if it's a set function  &#36;P: \mathcal{F} \rightarrow[0,1]&#36; and meets the following three principles:</p>
<ol>
<li>Non-negative: for arbitrary events  &#36;A \in \mathcal{F}&#36;  Yes.  &#36;P(A) \geq 0&#36;  </li>
<li>Unitivity: The probability of the entire sample space is 1, i. e.  &#36;P(\Omega)=1&#36;  </li>
<li>Can add: for any number of non-intersectional events  &#36;A_{1}, A_{2}, \ldots \in \mathcal{F} , 有  P\left(\bigcup_{i=1}^{\infty} A_{i}\right)=   \sum_{i=1}^{\infty} P\left(A_{i}\right)&#36;</li>
</ol>
<p>Probability is a special measure, the object of our main study in probability theory, the concept of probability being the first one to be measured.<a href="/en/blog/2023/03/18/real-analysis-notes/">Variable Functions</a>It was introduced.</p>
<p>Naturally, we can give some probability measurements.</p>
<ul>
<li>Monophonic:&#36;A \subset B&#36; then &#36;P(A) \le P(B)&#36;</li>
<li>Complementation rule: &#36;P(A^{c}) = 1-P(A)&#36;</li>
<li>Limited addition: Arbitrary incidents&#36;A_{1}, A_{2}, \ldots \in \mathcal{F}&#36;  Yes. &#36;P\left(\bigcup_{i=1}^{\infty} A_{i}\right) \le\sum_{i=1}^{\infty} P\left(A_{i}\right)&#36;</li>
<li>Continuity: &#36; \lim if \subseteq A {2}\subsetq \ldots is an incremental sequence of events and \bigcup i=(infoty}A i}=A <em>{n \rightarrow \infty} P\left(A</em>{n}right) = P(A) &#36; decrease equals</li>
<li>Complimentary principles:&#36;P(A\cup B) = P(A)+P(B)-P(AB)&#36;</li>
</ul>
<p>Having defined the concept of probability measurement, we can give a precise definition of probability space:</p>
<p>A probability space is set by a triad. &#36;(\Omega,F,P)&#36; Composition, of which:</p>
<ul>
<li>Sample space&#36;\Omega&#36; It's not empty and contains all the results (sampling points)</li>
<li>Event Field&#36;F&#36;It's sample space.&#36;\Omega&#36; The last one.&#36;\sigma&#36;Algebra</li>
<li>Probability measure&#36;P&#36;Satisfied with the definition we just gave, is one&#36;\sigma&#36;Algebra set functions</li>
</ul>
<p>They provide the necessary structure and rules for the analysis and calculation of probability of random events, i.e. sample size, measurable range of events and probability of events.</p>
<p>Now, let's introduce the theory of perfecting probability space, which is the basis for some of the questions that follow.</p>
<p>Definition: In Probability Space &#36;(\Omega, \mathcal{F}, P)&#36; , if&#36;A \in \mathcal{F}&#36; And... &#36;P(A) = 0&#36; Name&#36;A&#36;For a zero, we usually use it.&#36;\mathcal{N}&#36;The whole population of zero events in probability space</p>
<p>Especially, not all the probability spaces.<strong>Zero is a zero.</strong>But we call it the probability space to satisfy this nature.<strong>Full probability space</strong>(and completeness in general analysis is not a concept)</p>
<p>While the random probability space may not be complete, it must be able to expand it to a full probability space, along the following lines:</p>
<p>Theorem: to establish a probability space &#36;(\Omega, \mathcal{F}, P)&#36; There must be a perfect probability space. &#36;(\Omega, \overline{\mathcal{F&#125;&#125;, \overline{P})&#36; Satisfied &#36;\mathcal{F} \subset \overline{\mathcal{F&#125;&#125;,P = \overline{P}&#36; </p>
<h3>Random variable</h3>
<p>With so much discussion before us, we can begin to discuss the most important concept of random variables in the probability theory.</p>
<p>Definitions&#36;\left(\Omega,\mathcal{F},\mathbb{P}\right)&#36;For a probability space and&#36;(S,\mathcal{S})&#36;is randomly measurable space (i.e.&#36;\mathcal{S}&#36;Yes&#36;S&#36; Made from certain subsets &#36;\sigma&#36;-Algebra) , a function defined in sample space &#36;X(\omega)&#36;: &#36;\Omega\to S&#36;It's one.&#36;(\Omega,\mathcal{F},\mathbb{P})&#36;Top&#36;S&#36;-A random variable (r.v.) if any &#36;B\in S&#36; Yes.
&#36;&#36;X^{-1}\left(B\right):=\left{\omega\in\Omega;X\left(\omega\right)\in B\right}\in\mathcal{F}.&#36;&#36;</p>
<p>It's a natural definition. <strong>The size of the event field determines whether a measurable function is a random variable</strong> </p>
<p>We're here.<a href="/en/blog/2023/03/18/elementary-probability-notes/">Primary probability theory</a>is a random variable of the actual value of the general study at this time.&#36;(S,\mathcal{S})&#36; Replace &#36;(R^d,B_{R^d})&#36;  &#36;d = 1&#36;It's a random variable, or it's a random vector.</p>
<p>We can turn the original problem to the theory below.
&#36; \begin{aligned}&amp;\text{X:\Omega\to S\quad\text{and (S,\mathcal{S}\quad\text{is Detectablespace}.\text}J=x^B\S}\text{is}\sigma\text{alges}.\end{aligned} &#36;
So we remember&#36;\sigma(X) = {X^{-1}(B)|B\in S}&#36; To Map&#36;X&#36;Generated&#36;\sigma&#36;Algebra. He's a function of the sample space.&#36;X&#36;The smallest event field that makes up a random variable</p>
<p>Here's a simple picture of the multi-dimensional random variables.&#36;\sigma&#36;Algebra
Set&#36;X=\left(X_1,\cdots,X_n\right):\left(\Omega,\mathcal{F}\right)\to\left(\mathbb{R}^n,B_{\mathbb{R}^n}\right)&#36;It's one.&#36;\mathbb{R}^d&#36;A random variable with a value
&#36;&#36;\sigma\left(X\right)=\sigma\left(\bigcup_{i=1}^{n}\sigma\left(X_{i}\right)\right).&#36;&#36;
In other words, a multi-dimensional random variable.&#36;\sigma&#36;Algebra. It's all his weight.&#36;\sigma&#36;Algebra and Generated&#36;\sigma&#36;Algebra</p>
<p>Because it's measurable, it's easy to give down the theorem.
Set&#36;f:\left(\mathbb{R}^n,B_{\mathbb{R}^n}\right)\to (R,B_{R})&#36;Measurable  &#36;X_i&#36;Yes.&#36;\left(\Omega,\mathcal{F},\mathbb{P}\right)&#36;. There are random variables&#36;f(X_1,X_2...X_n)&#36; Yes. &#36;\left(\Omega,\mathcal{F},\mathbb{P}\right) \to (R,B_R)&#36; Random variable on</p>
<p>The proof is simple, just because...&#36;f&#36;It's a combination of two tiers of measurable function, so he can measure it, so it's a random variable.</p>
<h2>Distribution & Scores</h2>
<h3>Distribution of Random Variables</h3>
<p>The random variable is from Probability Space.<strong>Three dollars.</strong>To detectable space.<strong>Two dollars.</strong>It's a strange thing to do from the three-dollar map to the two-dollar map, actually because we're missing a concept.<strong>Distribution of Random Variables</strong>He'll add two to three.</p>
<p>Definitions  &#36;X:(\Omega, \mathcal{F}) \rightarrow(S, \mathcal{S})&#36; For Probability Space  &#36;(\Omega, \mathcal{F}, \mathbb{P})&#36; A random variable on top.  &#36;B \in \mathcal{S}&#36;  , defines:
&#36; \mathcal{P}<em>{X} (B): =\mathbb{P}\left(X^ (B)\right) &#36;
Called &#36;\mathcal{P}</em>{X}&#36; 为随机变量 &#36;Distribution of X&#36;</p>
<p><strong>This defines the distribution of random variables, not the distribution function.</strong></p>
<p>We're easy to verify. &#36;(S, \mathcal{S},\mathbb{P}_X)&#36; It's also a probability space.<strong>We've come from the original event probability space to a more abstract distribution probability space.</strong>, thereby simplifying some studies</p>
<p>If (S, \\mathcal{S} = \\left(\mathbb{R}, \\mathcal{B}<em>The following function is called:
&#36;F</em>{X} (x): =\mathcal{P} \X} (-intty, x) \quad x \mathbb{R} &#36;
Random variable for real value  &#36;X&#36;  ^ac85f9</p>
<p>Now we can make it clear that the so-called distribution function is a function that is constructed using the distribution of random variables, that is, the distribution function is induced by the distribution of random variables.</p>
<p>Here we finally have a precise definition of distribution, and in the theory of primary probability we have only studied distribution functions, and only one definition of distribution is felt.</p>
<p>Here's the definition of random variables and distribution.</p>
<p>Definitions: Establishment&#36;X,Y&#36;It's worth the same space.&#36;S&#36;Yes.&#36;S&#36;A random variable with \\mathcal{P}<em>{X}=\mathcal{P}</em>{Y}&#36; 也就是任意&#36;B\in S&#36; 有 &#36;\mathcal{P}<em>{X}(B)=\mathcal{P}</em>{Y}(B)&#36;  则称 &#36;X&#36; 与 &#36;Y&#36; 同分布 写作 &#36;X \overset{d}{=} Y&#36;</p>
<p>Special <strong>Same distribution does not define the same probability space Let's go.</strong>, distribution is a much more abstract concept than probabilistic space, studying only the extraction space of random variables rather than the original probability space</p>
<p>We're going to add a theory to this.</p>
<p>Theorem: Set&#36;X,Y&#36;It's worth the same space.&#36;S&#36;, meet&#36;\mathcal{S} = \sigma(\mathcal{A})&#36; And...  &#36;\mathcal{A}&#36; It's one.&#36;\pi&#36; Category:&#36;B\in \mathcal{A}&#36; There is &#36;matcal{P}<em>{X}(B)=\mathcal{P}</em>{Y}(B)&#36;  则  &#36;X \overset{d}{=} Y&#36;</p>
<p>Inference:&#36;(S, \mathcal{S})=\left(\mathbb{R}, \mathcal{B}_{\mathbb{R&#125;&#125;\right)&#36;   &#36;\mathcal{A} = {(-\infty,x]|x\in R}&#36; And...&#36;\sigma(A) = B_R&#36;  According to the theory above, whatever.&#36;x\in R&#36; Yes. &#36;F_X(x)= F_Y(x)&#36;    then  &#36;X \overset{d}{=} Y&#36;</p>
<p>Continues with the inference that almost the same distribution function produces the same distribution.</p>
<p><strong>This abstract process of random variables can lead to a compression of information, but at the same time it means a simplification of the problem, and it's important to set the right random variables.</strong></p>
<h3>Nature of Distribution Functions</h3>
<p>We have a definition of random variable distribution functions.</p>
<p>See the “relevant paragraphs” section of this paper.</p>
<p>In fact, in conjunction with the introduction of random variables in the primary probability theory, it is easy to give him three characteristics.</p>
<ul>
<li>&#36;F(-\infty) = 0,F(\infty) = 1&#36;</li>
<li>&#36;F(x)&#36; It's a one-to-one increment.</li>
<li>&#36;F(x)&#36; Right at every point.</li>
</ul>
<p>In fact, we can also give another perspective to this question: All functions that satisfy the three above-mentioned characteristics are distributed functions, which can inversely induce distribution</p>
<h3>Classification and breakdown of distributed functions</h3>
<h4>Separated and consecutive parts</h4>
<p>Sufficient proof has been given to ensure that the break points of a distribution function can be listed at most.&#36;{a_n}&#36; So we can define a leap for whatever.&#36;n&#36; Yes.</p>
<p>&#36;&#36;b_{n}=\Delta F\left(a_{n}\right)=F\left(a_{n}\right)-F\left(a_{n}-\right)&#36;&#36;
of which&#36;b_n&#36;It's called a leap. &#36;F\left(a_{n}-\right)&#36; It's given because of the right continuous nature of the function.</p>
<p>Further, we can define
&#36;&#36;F_{d}\left(x\right):=\sum_{n\in\mathbb{Z&#125;&#125;b_{n}l_{\left[a_{n},\infty\right)}\left(x\right),x\in\mathbb{R}&#36;&#36;
of which&#36;l&#36;It's the symbolic part. The function means the leap and we'll...&#36;F_d(x)&#36;The discrete part called the distribution function</p>
<p>It's easy to verify.&#36;F_d(x)&#36;Satisfied with the nature of</p>
<ul>
<li>&#36;F_d(-\infty) = 0,F_d(\infty) \le  1&#36;</li>
<li>&#36;F_d(x)&#36; It's a one-to-one increment.</li>
<li>&#36;F_d(x)&#36; Right at every point.</li>
</ul>
<p>If &#36;\sum_{n\in\mathbb{Z&#125;&#125;b_{n} = 1&#36;  then&#36;F_d(x)&#36;It's also a distribution function.&#36;F_d(\infty) =  1&#36;This is what we call a discrete distribution function. </p>
<p>In particular, we can define the continuous part of the distribution function by
&#36;&#36;F_{c(x)}= F(x) - F_d(x)&#36;&#36;
It's easy to verify. He's satisfied.</p>
<ul>
<li>&#36;F_c(-\infty) = 0,F_c(\infty) \le  1&#36;</li>
<li>&#36;F_c(x)&#36; It's a one-to-one increment.</li>
<li>&#36;F_c(x)&#36; Right at every point.</li>
</ul>
<p>Again, if he also forms a distribution function, we call it a continuous distribution function.</p>
<h4>Jordan has broken up.</h4>
<p>Theoretically (<strong>Jordan breakdown of distributed functions</strong>Construction&#36;F(x),x\in\mathbb{R}&#36;is any distribution function, which exists and which exists only&#36;\alpha\in[0,1]&#36;Make
&#36;&#36;F\left(x\right)=\alpha F_{1}\left(x\right)+\left(1-\alpha\right)F_{2}\left(x\right),x\in\mathbb{R},&#36;&#36;
of which&#36;F_1&#36;is the discrete distribution function and&#36;F_2&#36;Is a continuous distribution function</p>
<h4>It's absolutely continuous.</h4>
<p>We already have the concepts of continuity and separation. Is that enough? We need to think about this in the context of the theory of primary probabilities.</p>
<p>What about the discrete part of the primary probability theory that we've been talking about? </p>
<p>If we want him to have a continuous distribution, we need to find a corresponding density function, which simply meets the nature of the front, so we define an absolute continuous distribution function.</p>
<p>Definition (absolute continuous distribution function)&#36;F(x),x\in\mathbb{R}&#36;for a distribution function. If &#36;F&#36;It's absolutely continuous.&#36;(AC)&#36;, i. e. any &#36;-\infty&lt;x_1&lt;y_1&lt;x_2&lt;y_2&lt;\cdots&lt;x_m&lt;y_m&lt;+\infty&#36; 和任意&#36;\varepsilon&gt;0, present \delta&gt;Zero makes
&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;&lt;\delta\Longrightarrow\sum_{i=1}^{m}\left|F\left(y_{i}\right)-F\left(x_{i}\right)\right|&lt;\varepsilon, &#36;
Name&#36;F&#36;An absolute continuous distribution function</p>
<p>Depending on the nature of the absolute continuous function, we also have:&#36;F&#36;is an absolute continuous distribution function, then
A non-negative function exists&#36;f\in L^1(\mathbb{R})&#36;Make, for any &#36;x 1&lt;X &#36;2 content
&#36;&#36;F\left(x_{2}\right)-F\left(x_{1}\right)=\int_{x_{1&#125;&#125;^{x_{2&#125;&#125;p\left(x\right)dx,&#36;&#36;
Of which &#36;0.00<em>{L^1}:=\int</em>{\mathbb{P&#125;&#125;|p(x)|&#36;d&#36;x=1.&#36;于是&#36;F^\prime=p\geqslant0&#36;, a.e. (因为分布函数&#36;F&#36;是单增的),以后我们称&#36;x\to p(x)&#36;为绝对连续型分布函数&#36;F.O.R.I.D.</p>
<p>We can propose the concept of reciprocity, which is a strange distribution.</p>
<p>Definitions: If&#36;F(x)&#36;A distributed function and &#36;F^&#39;}=0&#36; a.e. 则称&#36;F&#36;是奇异型分布函数，进一步的，如果&#36;F.A.A. is continuous. It's called a continuous odd distribution function.</p>
<p>Any discrete distribution function is a strange distribution function, and a continuous odd distribution function requires the use of the Contractor tricentiary</p>
<h4>Lebesgue decomposition</h4>
<p>Set&#36;F&#36;Is any distribution function, then exists&#36;\alpha,\beta \in [0,1]&#36; Make
&#36;&#36;F=\alpha F_{1}+\beta(1-\alpha)F_{2}+(1-\alpha)(1-\beta)F_{3}&#36;&#36;
of which&#36;F_1&#36;It's part of the separation.&#36;F_d&#36; &#36;F_2&#36;It's an absolute continuum.&#36;F_{a.c}&#36;  &#36;F_3&#36;It's the weird part of the continuum.&#36;F_{c.s.}&#36; </p>
<p>There must be one and only such breakdown.</p>
<h3>Score (mathematical expectations)</h3>
<p>Let's discuss random variables in this section.&#36;X&#36;And we're looking at the definition of expectations, the nature of expectations, and some of the principles that promote them.</p>
<h4>Definition of points</h4>
<p>We have introduced the concept of probability measurement on random variables, and it's very natural that we should consider the Lebesgue points in the system of justice, which is called<strong>Random variables happen to be a measurable function.</strong>So it's very natural to define his Lebesgue points, and we'll introduce them in four steps.</p>
<p>For random variables and detectable functions: each value of a random variable (or&#36;B_{R}&#36;According to the relevant theorem, one can best be found.&#36;\Omega&#36;Up Subset&#36;A&#36;It's a probability measure.&#36;P(A)&#36;It's a measure of the measurable values of random variables.</p>
<p>With these descriptions, we can imitate the classic four-step definition of the Lebesgue points for random variables.</p>
<p>STEP1
For indicative random variables
&#36;&#36;I_{A}(w)=\begin{cases}1,w\in A\0,w \notin A\end{cases}&#36;&#36;
His expectations are...
&#36;&#36;E(I_{A})= 1 \times P(A)+0 \times P(A^c)&#36;&#36;</p>
<p>STEP2
A non-negative random variable exists&#36;\Omega&#36;Non-intersectional division&#36;A_i&#36; and corresponding weights &#36;b_i&#36; Random variable content
&#36;&#36;X\left(\omega\right)=\sum_{i=1}^{n}b_{i}I_{A_{i&#125;&#125;\left(\omega\right)&#36;&#36;
We can define it naturally.&#36;X&#36;About&#36;P&#36;Score
&#36;&#36;\mathbb{E}\left[X\right]=\int_{\Omega}X\left(\omega\right)\mathbb{P}\left(d\omega\right):=\sum_{i=1}^{n}b_{i}\mathbb{P}\left(A_{i}\right).&#36;&#36;</p>
<p>STEP3
For any non-negative random variable&#36;X&#36;, we can find a column of one-on-one non-negative simple random variable columns &#36;X^{m}&#36; Satisfied
&#36;&#36;0\leqslant\left|X\left(\omega\right)-X^{\left(m\right)}\left(\omega\right)\right|\leqslant2^{-m},\forall\omega\in\Omega.&#36;&#36;
So we can define the points by the limits.
&#36;&#36;\mathbb{E}[X]:=\sup\left{\mathbb{E}[\xi];\xi\text{ 是非负简单随机变量且 }\xi\leqslant X\right}\in[0,+\infty].&#36;&#36;
If&#36;E[X] \to \infty&#36; It's called the lack of points.</p>
<p>STEP4
For any real-value random variable &#36;X = X^{+}+X^{-}&#36; Define the score has
&#36;&#36;\mathbb{E}[X]=\int_{\Omega}X\left(\omega\right)\mathbb{P}\left(d\omega\right):=\mathbb{E}\left[X^{+}\right]-\mathbb{E}\left[X^{-}\right].&#36;&#36;</p>
<p>In conclusion, the definition of our points, or mathematical expectations, is over.
&#36; \begin{aligned}
E [X]&amp; =\int_{\Omega}X(\omega)P(d\omega)=\int_{\Omega}X(\omega)dP(\omega) \
&amp;=\int_{x}XdP
\end{aligned}&#36;&#36;</p>
<h4>Nature of expectation</h4>
<ol>
<li>Linear Addability&#36;\mathbb{E}[aX+bY]=a\mathbb{E}[X]+b\mathbb{E}[Y]&#36;</li>
<li>Set&#36;X&#36;It's a random variable and&#36;X\geqslant0&#36;,a.e.&#36;\mathbb{P}(X\geqslant0)=1)&#36;, then&#36;\mathbb{E}[X]\geqslant0.&#36;Especially if&#36;X=0&#36;And, a.e.,&#36;\mathbb{E}[X]=0.&#36;</li>
<li>Set&#36;X,Y&#36;For two scramble random variables and&#36;X\leqslant Y&#36;And, a.e.,&#36;\mathbb{E}[X]\leqslant\mathbb{E}[Y]&#36;</li>
<li>&#36;\text{设 }X,Y\text{为两个可积随机变量，那么}|X+Y|\text{也是可积的，以及}&#36; &#36;\mathbb{E}[|X+Y|]\leqslant\mathbb{E}[|X|]+\mathbb{E}[|Y|].&#36;</li>
<li>Set Random Variables&#36;X&#36;Compilation&#36;|\mathbb{E}[X]|\leq\mathbb{E}[|X|]&#36; </li>
<li>Set Random Variables&#36;X&#36;Cumulative &#36;A\in F&#36; If constant exists &#36;a \le X(w) \le b&#36; Whatever it takes.&#36;a\mathbb{P}\left(A\right)\leqslant\mathbb{E}\left[X11_{A}\right]\leqslant b\mathbb{P}\left(A\right).&#36;</li>
</ol>
<h4>Important theorizing of expectations</h4>
<p>We're looking at the condensation of random variable sequences in this section. <a href="/en/blog/2023/03/18/elementary-probability-notes/">Primary probability theory</a> "Concentration almost everywhere, probability 1 constriction" Section </p>
<p><strong>(Single-concentrating theorem)</strong> Set &#36;&#36;X n}<em>{n\geqslant1}&#36;为概率空间&#36;(\Omega,\mathcal{F},\mathbb{P} &#36; = non-negative, monogamable and amplifiable random variable, then there is
&#36;E\left[\lim]</em>{n\to\infty}X_{n}\right]=\lim_{n\to\infty}E[X_{n}].&#36;&#36;
 <strong>(Fatou Introduction)</strong> Set &#36;&#36;X n}<em>{n\geqslant1}&#36;是概率空间&#36;(\Omega, \mathcal{F}, \mathbb{P}) &#36; a column of non-negative and random variables, if
&#36; \matbb{E}\left[\lim]</em>{n\to\infty}X_{n}\right]\leqslant\lim_{n\to\infty}\mathbb{E}\left[X_{n}\right].&#36;&#36;</p>
<p><strong>(Crossing incipient theorem)</strong> Set &#36;&#36;X n}<em>{n\geqslant1}&#36;是概率空间&#36;(\Omega, \\mathcal{F}, \\mathbb{P}) &#36; a column of random variables and satisfied
&#36;\left|X</em>{n}\right|\leqslant Y,\forall n\geqslant1,&#36;&#36;
其中&#36;Y&#36;是一个独立于&#36;n&#36; 的非负可积随机变量.进一步，如果
&#36;&#36;\mathbb{ft x=right=x\text=xright=x\light{\x}&#36;&#36;
那么有
&#36;&#36;\mathbb{E}\left[\lim_{n\to\infty}X_{n}\right]=\lim_{n\to\infty}\mathbb{E}[X_{n}].&#36;&#36;</p>
<p> <strong>♪ There's a strong theorem ♪</strong> Set &#36;\left{X n}right}<em>{n \geqslant 1}&#36;  和  &#36;X&#36;  是概率空间  &#36;(\Omega, \mathcal{F}, \mathbb{P})&#36;  上的一列有界随机变量且满足  &#36;X</em>X, n \rightarrow(a.e.)
&#36;lim <em>{n \rightarrow \infty} \mathbb{E}\left[X</em>{n}\right]=\mathbb{E}\left[\lim <em>{n \rightarrow \infty} X</em>{n}\right] .&#36;&#36;</p>
<p>We give two very natural inferences.</p>
<ul>
<li>Set&#36;X&#36;is a non-negative real value random variable,&#36;P(A)=0&#36; Launch &#36;E[X;A]=0&#36;</li>
<li>Set&#36;X&#36;is a random variable with a positive physical value, or&#36;E[X;A]=0&#36; Launch &#36;P(A)=0&#36;
They're all very natural.</li>
</ul>
<h4>Probability inequality</h4>
<p><strong>Hölder heterogeneity</strong> Set &#36;1&lt;p,q&lt;+\infty&#36; 和 &#36;= &#36;1 and yes
&#36;&#36;|\mathbb{E}[XY]|\leqslant\mathbb{E}[|XY|]\leqslant{\mathbb{E}[|X|^{p}]}^{p^{-1&#125;&#125;{\mathbb{E}[|Y|^{q}]}^{q^{-1&#125;&#125;.&#36;&#36;</p>
<p><strong>Minkovski</strong> For any &#36;p&gt;&#36; 0, there is
&#36;&#36;\left{\mathbb{E}[|X+Y|^{p}]\right}^{p^{-1&#125;&#125;\leqslant\left{\mathbb{E}[|X|^{p}]\right}^{p^{-1&#125;&#125;+\left{\mathbb{E}[|Y|^{p}]\right}^{p^{-1&#125;&#125;.&#36;&#36;</p>
<p> <strong>Lyapunov hex</strong> For Any &#36;1&lt;p&lt;q&lt;+infty, yes
&#36;&#36;\left{\mathbb{E}\left[|X|^{p}\right]\right}^{p^{-1&#125;&#125; \leqslant\left{\mathbb{E}\left[|X|^{q}\right]\right}^{q^{-1&#125;&#125; .&#36;&#36;</p>
<p><strong>Jensen heterogeneity</strong> Set  &#36;\varphi(x): \mathbb{R}^{d} \rightarrow \mathbb{R}&#36;  For a convex, there is
&#36;&#36;\varphi(\mathbb{E}[X]) \leqslant \mathbb{E}[\varphi(X)]&#36;&#36;</p>
<p><strong>Cr Instinct</strong> For any &#36;p &gt; Yes.
&#36;&#36;|X_{1}+X_{2}+...X_{n}|^{p}\le Cr(|X_{1}|^{p}+|X_{2}|^{p}+...|X_{n}|^{p})&#36;&#36;
of which &#36;Cr = 1 <del>if</del> p\le 1<del>else</del>Cr = n^{p-1}&#36;   </p>
<h4>Variable variant formula (calculation of points)</h4>
<p>It's not easy to calculate the scores of the random variables of European spatial values on probability measurements, and here's the theory that converts such probability measurements to Riemann-Stiltjes points for easy calculation. The variant formula is in its broader form, but here we just describe the types of probability measurements.</p>
<p><strong>Theorem (Variance Change Formula)</strong> Set  &#36;X:(\Omega, \mathcal{F}) \rightarrow(S, \mathcal{S})&#36;  For Probability Space  &#36;(\Omega, \mathcal{F}, \mathbb{P})&#36;  A random variable and  &#36;h&#36;  Define in  &#36;S&#36;  a function of the actual value above, which allows  &#36;h(X)&#36;  It's colossal, then.
&#36;US&#36;\mathbb{E}[h(X)]=int (Omega}h(X(omega))\mathbb{P}(\mathrm{d}\omega)=\int S}h(x)\mathcal{P}<em>{X}(\mathrm{~d} x)&#36;&#36;
如果  &#36;S=\mathbb{R}^{d}&#36;  ，则有
&#36;&#36;\mathbb{E}[h(X)]=\int</em>{\Omega} h(X(\omega)) \mathbb{P}({d} \omega)=\int_{S} h(x) \mathcal{P}<em>{X}({~d} x)=\int</em>I'm sorry.
of which  &#36;\mathcal{P}_{X}&#36;  For random variables  &#36;X&#36;  the distribution, and when  &#36;S=\mathbb{R}^{d}&#36;  I don't know.  &#36;F(x), x \in \mathbb{R}&#36;  is the distribution function.</p>
<p>This formula allows us to convert the measure to the Rs.</p>
<p>If&#36;F&#36;It's absolutely continuous.&#36;F(dx)&#36;Yeah.&#36;d(F(x))&#36;  The original problem can be converted to Riemann points.</p>
<p>If he is separated,
&#36;&#36;F\left(x\right):=\sum_{n\in\mathbb{Z&#125;&#125;b_{n}l_{\left[a_{n},\infty\right)}\left(x\right),x\in\mathbb{R}&#36;&#36;
The original problem is the expectation of a discrete distribution.<a href="/en/blog/2023/03/18/elementary-probability-notes/">Primary probability theory</a> . The Mathematical Expectations of the Functions of Random Variables section</p>
<p>In the event of a strange continuum, it would not be within the scope of the solution at present, and would not be calculated.</p>
<p>If the original distribution function is too complex, the distribution function is broken down using the " Lebesgue decomposition " section of this paper, which is calculated separately.</p>
<p>Definition of Riemann - Steeltjes points, reference<a href="/en/blog/2024/10/17/financial-stochastic-analysis-notes/">Financial random analysis</a> "Reemann-Stiltjes" section</p>
<h3>Independence</h3>
<p>We're here.<a href="/en/blog/2023/03/18/elementary-probability-notes/">Primary probability theory</a>It looks at the independence of events and the independence of random variables, but their definition always makes us feel less mathematical, and here it is to bridge this shortcoming.</p>
<h4>Independence of the two incidents</h4>
<p>Let's refer to the introduction to the primary probability theory.<a href="/en/blog/2023/03/18/elementary-probability-notes/">Primary probability theory</a> “The Independence of the Two Events” section.</p>
<h4>Two.&#36;\sigma&#36; Independence of algebra</h4>
<p>Definitions (normal)&#36;\sigma&#36;Algebra: Later&#36;\mathcal{H}&#36;It's an ordinary thing. &#36;\sigma&#36;algebra, if
&#36;&#36;\mathbb{P}\left(H\right)=0或1,\forall H\in H.&#36;&#36;</p>
<p>Definitions&#36;\sigma&#36; Independence of algebra: establishment&#36;(Ω,\mathcal{F},\mathbb{P})&#36;It's a probability space.&#36;\mathcal{H},\mathcal{G}\subset\mathcal{F}&#36; For two.&#36;\sigma&#36;Algebra ... if any&#36;H\in\mathcal{H}&#36;and&#36;G\in\mathcal{G}&#36;We did.
&#36;&#36;\mathbb{P}(G\cap H)=\mathbb{P}(G)\mathbb{P}(H),&#36;&#36;
Name&#36;\sigma&#36; Algebra&#36;\mathcal{H},\mathcal{G}&#36;It's independent.</p>
<p>Easy to see: ordinary&#36;\sigma&#36;Algebra and other space.&#36;\sigma&#36;Algebras are independent. We'll use it in the back.</p>
<h4>Independence of two random variables</h4>
<p>Definitions (independentness of random variables) Set&#36;X,Y&#36;For Probability Space&#36;(\Omega,\mathcal{F},\mathbb{P})&#36;. If&#36;\sigma\left(X\right)&#36;and&#36;\sigma\left(Y\right)&#36;It's independent.&#36;X&#36;and&#36;Y&#36;Be independent.</p>
<p>It's a very natural definition. We created it with random variables.&#36;\sigma&#36;Algebra independence to study the independence of random variables.</p>
<h4>Multiple independence</h4>
<p>We can naturally give the following reasoning.
See<a href="/en/blog/2023/03/18/elementary-probability-notes/">Primary probability theory</a> “Independent Multiple Events” section.</p>
<p>&#36;\sigma&#36;Algebras are a cluster, so we can start with definitions of cluster independence based on the definition of event independence:</p>
<p>Definition (independentness of a cluster): Establishment&#36;(\Omega,\mathcal{F},\mathbb{P})&#36;It's a probabilistic space and &#36; {\mathcal{A}<em>\alpha}</em>{\alpha\in I}\subset\mathcal{F}&#36;为一族集类 (其中&#36;I&#36;可以不可数).如果对任意的正整数&#36;L&gt;0&#36;和互不相同的&#36;\alpha_{1},\cdots,\alpha_{L}\in I&#36;,
&#36;&#36;\mathbb{P}\left(\bigcap_{k=1}^{L}A_{k}\right)=\prod_{k=1}^{L}\mathbb{P}\left(A_{k}\right),\forall A_{k}\in\mathcal{A}<em>{\alpha</em>K=1,\cdots, L, &#36;
And it's called &#36; {\matcal{<em>\alpha}</em>It's independent of each other.</p>
<p>Definitions&#36;\sigma&#36;Independence of algebra: We allow that&#36;n&#36;individual&#36;\sigma&#36;The definition of algebra independence is optimal
&#36;&#36;\mathbb{P}\left(\bigcap_{k=1}^{n}A_{k}\right)=\prod_{k=1}^{n}\mathbb{P}\left(A_{k}\right)&#36;&#36;
In other words, there is no longer any demand for arbitrary combinations of independence, and total independence is mutual independence.</p>
<p>Definition: One family random variable &#36; {X alpha}<em>{\alpha\in I}&#36;是相互独立的如果&#36;\left{\sigma\left(X</em>It's independent.</p>
<h4>Difference between primary probabilities and independence in higher probabilities</h4>
<p>We're studying the independence of random variables in our primary probability theory.
See<a href="/en/blog/2023/03/18/elementary-probability-notes/">Primary probability theory</a> "Independent of random variables".</p>
<p>What's the difference between this and independence in the theory of higher probability?</p>
<p>The high probability theory allows non-real random variables, and distribution functions allow real random variables only. From this point of view, the definition is price unequal.</p>
<p>Even if we limit the random variables in the high probability theory to the real range, the definition is not seemingly equal.</p>
<ul>
<li>Study distribution functions in primary probability only&#36;[-\infty,x]&#36;Independence of the original in scope</li>
<li>The high probability theory is limited to real value.&#36;B_R&#36; Clear range.&#36;[-\infty,x]&#36;Large
<strong>But we've come to the conclusion that these two definitions are essentially equivalent and can be used.&#36;\pi-\lambda&#36;Evidence of achievement of definitions</strong></li>
</ul>
<h3>Multiplication</h3>
<p>Definition&#36;\sigma&#36;-Algebra: set&#36;F_1和F_2&#36;Two.&#36;\sigma&#36;- Algebra. Defines the following matrix set
&#36;R:=lift{A\times B; A\mathcal{F}<em>{1},B\in\mathcal{F}</em>Right, &#36;
Then... &#36;\mathcal{F} _1\otimes \mathcal{F} _2: = \sigma \left ( \mathcal{R} \right )&#36;Yes&#36;\sigma&#36;- Algebra.&#36;\mathcal F_1,\mathcal{F}_2&#36;Product&#36;\sigma&#36;- Algebra. Further, we say.&#36;R&#36;Medium
, which is in a measurable rectangle.</p>
<p>Definition (multiplier measurable space and intercept): set (\Omega k,\mathcal{F}<em>k),k=12.00 is two detectable spaces, then
&#36;&#36;\left (\Omega,\mathcal{F}\right): =\left(\Omega)</em>{1}\times\Omega_{2},\mathcal{F}<em>{1}\otimes\mathcal{F}</em>{2}\right)&#36;&#36;
为乘积可测空间.对任意&#36;E\subset\Omega_1\times\Omega_2&#36;和&#36;w_i\in\Omega_i,i=1,2&#36;,我们定义：
&#36;&#36;\begin{cases}E {\omega}; \left{w\in’omega 2}; \left(1}omega right}\subset\Omega}2}, \e omega}: \left{w\in’omega }; \left(\omega,\omega {rights}\subset\Omega}. \end{cases}&#36;&#36;
Further, separate&#36;E_{\omega_1}&#36;Yes&#36;E&#36;Yes.&#36;\omega_1&#36;- Cut.&#36;\left(\omega_1\text{-section)和 }E_{\omega_2}\text{ 为 }E\text{ 的 }\omega_2\text{-截口}\right.&#36;</p>
<p>Theorem: any&#36;\Omega_1\times\Omega_2&#36;It's all measurable.</p>
<p>Theorem: Set&#36;f:\Omega_1\times\Omega_2\to\overline{\mathbb{R&#125;&#125;&#36;It's a measurable function.&#36;\omega_i\in\Omega_i,i=1,2&#36;, define the amputation function:
&#36;&#36;f_{\omega_{1&#125;&#125;(\omega_{2}):=f(\omega_{1},\omega_{2}),\quad f_{\omega_{2&#125;&#125;(\omega_{1}):=f(\omega_{1},\omega_{2}).&#36;&#36;
So there is.
&#36;\left(\mathrm{i}\right)对任意\omega_1\in\Omega_1,f_{\omega_1}:\Omega_2\to\overline{\mathbb{R&#125;&#125;&#36;is measurable;
&#36;\left(\mathrm{ii}\right)对任意\omega_{2}\in\Omega_{2},f_{\omega_{2&#125;&#125;:\Omega_{1}\rightarrow\overline{\mathbb{R&#125;&#125;&#36;It's measurable.</p>
<p>Theorem: For &#36;i=1,2, set \nu <em>{i}&#36;为可测空间&#36;\left(\Omega_i,\mathcal{F}<em>i\right)&#36;上的&#36;\sigma-limited, then
Only &#36;\left (\Omega,\mathcal{F}\right): =\left(\Omega)</em>{1}\times\Omega</em>{2},\mathcal{F}<em>{1}\otimes\mathcal{F}</em>\sigma-limited \nu on \right)
&#36;&#36;\nu\left(\biguplus_{k=1}^{m}A_{k}\times B_{k}\right)=\sum_{k=1}^{m}\nu_{1}\left(A_{k}\right)\nu_{2}\left(B_{k}\right),&#36;&#36;
Where &#36;A k\mathcal{F} 1, B k\mathcal{F}<em>2&#36;使得&#36;{A_k\times B_k}</em>{k=1}^m&#36;是互不相交的。对于这样的&#36;v&#36;我们称为乘积测度，&#36;v 1, v &#36;2 is known as the margin measure.</p>
<p>And finally, we'll connect multiplication and joint distribution.</p>
<p>Theoretically, random variables&#36;X_1,\cdots,X_n&#36;It is independent of each other and is established only when:
&#36;&#36;P_{\left(X_{1},\cdots,X_{n}\right)}=\prod_{i=1}^{n}P_{X_{i&#125;&#125;,\text{在}S^{\otimes n}上.&#36;&#36;</p>
<h2>Expectations</h2>
<h3>Disconnected time conditions expect</h3>
<p>We're here.<a href="/en/blog/2023/03/18/elementary-probability-notes/">Primary probability theory</a>The theory of conditionality, which is being studied, is actually flawed and focuses only on calculation, but indeed on understanding its essence, and there is a need to add some element of expectation to facilitate this.<a href="/en/blog/2024/10/17/financial-stochastic-analysis-notes/">Financial random analysis</a>(c) To initiate discussions on the subject.</p>
<p>Every moment.&#36;n&#36; For each coin-throwing sequence, we can price the shares for a fork tree, which is the essence of the options pricing model described above, as follows:
&#36;&#36;S_n(\omega_1\cdots\omega_n)=\frac1{1+r}[\widetilde{p}S_{n+1}(\omega_1\cdots\omega_nH)+\widetilde{q}S_{n+1}(\omega_1\cdots\omega_nT)]&#36;&#36;</p>
<p>To simplify marking, we define
&#36; \tilde<em>n<a href="%5Comega_1%5Ccdotp%5Ccdotp%5Ccdotp%5Comega_n">S_{n+1}</a>=\tilde{p}S</em>{n+1}(\omega_1\cdotp\cdotp\cdotp\omega_nH)+\tilde{q}S_{n+1}(\omega_1\cdotp\cdotp\cdotp\omega_nT)&#36;&#36;
这样就可以简化原本的式子为
&#36;&#36;S_n=\frac1{1+r}\mathbb{E}<em>n[S</em>{n+1}]&#36;&#36;</p>
<p>Here.&#36;E[S_{n+1}]&#36; Call it time-based.&#36;n&#36;Information&#36;S_{n+1}&#36;Conditional expectations.</p>
<p>So we can give it a little more.</p>
<p>Definitions&#36;n&#36;Satisfied 1&#36;\leqslant n\leqslant N&#36;,for given sequence &#36;\omega_1\cdots\omega_n&#36;,Existing 2&#36;^{N-n}&#36;Possible follow-up &#36;\omega_n+1\cdots\omega_N&#36;I don't know. Use &#36;\sharp H(\omega_{n+1}\cdots\omega_N)&#36;It's a follow-up. &#36;\omega_{n+1}\cdots\omega_N&#36; The number of positives,&#36;\sharp T(\omega_n+1\cdots\omega_N)&#36;is the number of times the back appears. We define:
&#36; \tilde<em>n<a href="%5Comega_1%5Ccdots%5Comega_n">X</a>=\sum</em>== sync, corrected by elderman == @elder man
It's time-based.&#36;n&#36;Information&#36;X&#36;Expectations</p>
<p>Conditional expectations are also random variables, depending on which information is available at zero.&#36;n&#36;Results of the sub-test. We're at the bottom of the expected symbol in this section. Indicators&#36;n&#36;The condition is indicated as to whether the results of previous experiments are known to be problem-based.</p>
<h3>Nature of expectations of discrete time conditions</h3>
<p>We're not making false statements about the nature of our expectations.</p>
<p>Set N as a positive integer number,&#36;X&#36; and&#36;Y&#36; To rely on the front&#36;N&#36;Random variable of the result of a coin tossing. For given 0&#36;\leqslant n\leqslant N&#36; The following are established:</p>
<p><strong>The linear nature of conditionality</strong>: For all constants&#36;c_1&#36; and&#36;c_2&#36;Yes.
&#36;&#36;\mathbb{E}_n\begin{bmatrix}c_1X+c_2Y\end{bmatrix}=c_1\mathbb{E}_n\begin{bmatrix}X\end{bmatrix}+c_2\mathbb{E}_n\begin{bmatrix}Y\end{bmatrix}&#36;&#36;</p>
<p><strong>Extract known amount</strong>: If&#36;X&#36;It depends on the past.&#36;n&#36;Once a coin throws, then:
&#36;&#36;\mathbb{E}_n[XY]=X\cdot\mathbb{E}_n[Y]&#36;&#36;</p>
<p><strong>Repetitive expectations</strong>: "It's the formula of all expectations" if &#36;0\leqslant n\leqslant m\leqslant N&#36;And then:
&#36;&#36;\mathbb{E}_n[\mathbb{E}_m[X]]=\mathbb{E}_n[X]&#36;&#36;</p>
<p><strong>Independence</strong>: If&#36;X&#36;♪ Just depend on the first ♪&#36;n+1&#36;Second to &#36;N&#36; The result of a coin throw, then:
&#36;&#36;\mathbb{E}_n[X]=\mathbb{E}X&#36;&#36;</p>
<p><strong>The terms and conditions of Jason are not equal.</strong>: If&#36;\varphi(x)&#36;For the dumb variable&#36;x&#36;, then:
&#36;&#36;\mathbb{E}_n[\varphi(X)]\geqslant\varphi(\mathbb{E}_n[X])&#36;&#36;</p>
<p>These natures will facilitate the development of some of the subsequent proofs, referring to “the nature of the expectations of conditions” in this paper, in relation to the broader nature of the expectations. Section</p>
<h3>Symbolic measure</h3>
<p>Symbolic measurements are intended to complement the measurement base in the high probability theory and are useful in demonstrating the relevance of the “independent” section of this paper and in examining the “uniqueness of expectations” section of this paper.</p>
<h4>Definition of symbol measure</h4>
<p>In math analysis, we've studied the problem of variable points and conductors, for continuity.&#36;f&#36;
&#36;&#36;F(x) = \int_{a}^{x}f(y)dy &#36;&#36;
It's called an indeterminate fraction and a guide.</p>
<p>Correspondingly, measurable functions for the fractions in measure space&#36;f&#36; We can define a set of functions.&#36;\varphi&#36;
&#36;&#36;\varphi(A) = \int_{A}fd\mu~~~A\in F&#36;&#36;
Unpredictable points and measurements.&#36;\mu&#36;Other Organiser </p>
<p>Whether the conductor of a set function has a problem with guidance on measurements is the object studied in this section
Definition: Measurable functions for the fractions in the measure space&#36;f&#36;Uncertain points
&#36;&#36;\varphi(A) = \int_{A}fd\mu~~~A\in F&#36;&#36;
We know by the definition of points,&#36;\varphi&#36;Meet all the basic conditions for measurements other than non-negative, we call them<strong>Symbolic measure</strong>  And all measurements that meet these conditions are called symbol measurements.</p>
<h4>Hannah and Jordan split up.</h4>
<p>The fact that symbols are not non-negative is very annoying. Is there any way we can get him to be non-negative again? That's what it looks like.</p>
<p>The definition of detectable functions is divided according to the following rules, taking into account the uncertain points of the front
&#36;&#36;X=&lt;0\rangle &#36;&#36;
则可以把原始空间&#36;X&#36;分成下面的两部分
&#36;&#36;A\mathscr{F}, A\subset X^, \Raightarrow\varpi(A)\geqslant0; \A\mathscr{F}, A\subsetX^}Longrightarrow\varphi(A)\leqslant0.&#36;
We'll be like this in primitive space.&#36;X&#36;It's called Hann.</p>
<p>Again.
&#36;&#36;\varphi^{\pm} (A)=\int_{A}f^{\pm} \mathrm{d}\mu &#36;&#36;
You can measure it.&#36;\varphi&#36;Decomposed to two degrees, with decomposition
&#36;&#36;\varphi=\varphi^+-\varphi^-&#36;&#36;
This is called Jordan decomposition, which also becomes the total variation of the symbol measurements.</p>
<p><strong>There's a breakdown between Hann and Jordan for normal symbol measurements.</strong></p>
<h4>Radon-Nikodym Theorem</h4>
<p>Set &#36;\varphi&#36;is the measure space.&#36;X,\mathscr{F},\mu)&#36;We'll start defining it. &#36;\varphi&#36; The basic idea is really simple: if this symbol measure is only in the form of an uncertain fraction of the surface,
&#36;&#36;\varphi(A) = \int_{A}fd\mu~~~A\in F&#36;&#36;
They can then be considered to have a guide and an element of uncertainty.</p>
<p>Definitions: Establishment &#36;\varphi&#36;It's measuring space (X,&#36;\mathscr{F},\mu)&#36;. If the only detectable function in the meaning of a.e.&#36;f&#36;The preceding is established, in other words,&#36;f&#36;Yes&#36;\varphi&#36;Yeah.&#36;\mu&#36;R-N (Radon-Nikodym)&#36;\frac{\mathrm{d}\varphi}{\mathrm{d}\mu}\overset{\mathrm{def&#125;&#125;{\operatorname*{=&#125;&#125;f.&#36;</p>
<p>Just as not all functions in calculus can be directed, not every symbol measure has a R-N handle. What kind of symbol measure does it have a R-N handle?<strong>Only if &#36;\varphi&#36;Yeah. &#36;\mu&#36; It's possible in absolute continuity.</strong> </p>
<p>Definitions &#36;\varphi&#36;and &#36;\mu&#36; It's measurable.&#36;X,\mathscr{F}&#36;. if any&#36;A\in\mathscr{F}&#36;Both.
&#36;&#36;\mu(A)=0\Rightarrow\varphi(A)=0:&#36;&#36;
Name &#36;\varphi&#36;Yeah. &#36;\mu \textbf{}&#36;It's absolutely continuous.&#36;\varphi \ll \mu .&#36; </p>
<h4>Lebegue decomposes.</h4>
<p>The theme of this section is the decomposition of Lebesgue. The purpose of this section is to prove that any constrictive measure degrees&#36;\varphi&#36;For any one &#36;\sigma&#36;Limited Measure &#36;\mu&#36;, which can be broken down into two parts: a part of it.&#36;\mu&#36;Absolute continuous; another part with&#36;\mu&#36;It's weird.</p>
<h3>Uniqueness of conditionality</h3>
<p>The expectations of conditions in the original probabilistic theory are difficult to satisfy, so here we look at the expectations of conditions of justice.</p>
<p>Definitions  &#36;X, Y&#36;  It's defined in probability space. &#36;(\Omega, \mathscr{F}, \mathbb{P})&#36;  on the buildable variable,  &#36;\mathscr{G}&#36;  Yes.  &#36;\mathscr{F}&#36; Son.  &#36;\sigma&#36; - Algebra. If</p>
<ul>
<li>&#36;Y \in \mathscr{G}&#36;  (i)  &#36;Y&#36;  Yes.  &#36;\mathscr{G}&#36;  Detectable, i.e. &#36;\sigma(Y)\subset \mathscr{G}&#36;）；</li>
<li>To Any  &#36;A \in \mathscr{G}&#36; ,&#36;&#36;\int_{A} Y(\omega) d \mathbb{P}(\omega)=\int_{A} X(\omega) d \mathbb{P}(\omega),&#36;&#36;
Name  &#36;Y&#36;  Yes.  &#36;X&#36;  Yes.  &#36;\mathscr{\mathscr { G &#125;&#125;&#36;  The condition is expected, as  &#36;Y = \mathbb{E}[X \mid \mathscr{G}]&#36;</li>
</ul>
<p>It's a little bit different from what you've been studying? All with note below</p>
<p><strong>Note 1</strong> : If&#36;\mathscr{G}&#36;It's a random variable.&#36;Z&#36;Generated&#36;\sigma&#36;algebra, then
&#36;&#36;\mathbb{E}[X|\mathscr{G}]=\mathbb{E}[X|\sigma(Z)]=\mathbb{E}[X|Z].&#36;&#36;</p>
<p><strong>Note 2</strong> : For probability space&#36;(\Omega, \mathscr{F}, \mathbb{P})&#36;  Any real value variable&#36;X&#36; And whatever.&#36;\sigma&#36; - Algebra.&#36;\mathscr{G}  \subset \mathscr{F}&#36; There must be measurable random variables&#36;Y&#36;Meets the above conditions and the only random variable.</p>
<p><strong>Note 3</strong> : the conditional expectation of random variables is a random variable or the original probability space&#36;(\Omega, \mathscr{F}, \mathbb{P})&#36; on a detectable function we can remember as&#36;f(Z)&#36;Because it was created.&#36;\sigma&#36;Algebra's getting smaller.</p>
<p><strong>Note 4</strong> Other Organiser&#36;A&#36;About one.&#36;\sigma&#36;Algebra's conditional probability is defined as&#36;E[I_{A}\mid \mathscr{G}]&#36;</p>
<h3>Nature of expectation</h3>
<p>Set&#36;X,Y&#36;It's defined in probability space.&#36;(\Omega,\mathscr{F},\mathbb{P})&#36;and&#36;\mathcal{H}\subset\mathscr{G}&#36; Yes.&#36;\mathscr{F}&#36; Son. &#36;\sigma&#36;Algebra
It's the following.</p>
<ol>
<li>To Any&#36;a,b\in\mathbb{R}&#36;Both.&#36;\mathbb{E}[aX+bY|\mathscr{G}]=a\mathbb{E}[X|\mathscr{G}]+b\mathbb{E}[Y|\mathscr{G}].&#36;</li>
<li>If &#36;X\in \mathscr{G}&#36;,  &#36;\mathbb{E} [ XY| \mathscr{G} ] = X\mathbb{E} [ Y| \mathscr{G} ] .&#36; </li>
<li>If &#36;X\bot \mathscr{G}&#36;,  &#36;\mathbb{E} [ X| \mathscr{G} ] \Longrightarrow \mathbb{E} [ X] .&#36;</li>
<li>&#36;\mathbb{E} [ \mathbb{E} [ X| \mathscr{G} ] \mid \mathscr{H} ] = \mathbb{E} [ X| \mathscr{H} ] .&#36; Full expectation formula</li>
<li>If&#36;\varphi&#36;is a convex, then&#36;\varphi(\mathbb{E}[X|\mathscr{G}])\leq\mathbb{E}[\varphi(X)|\mathscr{G}].&#36;</li>
</ol>
<p>They're dealing with each other.</p>
<ol>
<li>Linear</li>
<li>Take out what you know.</li>
<li>Independence</li>
<li>Reciprocal expectations (i.e. the broad form of the full expectation formula)</li>
<li>The terms and conditions of Jason are not equal.</li>
</ol>
