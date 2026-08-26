---
title: 'Real Analysis: Sets, Point Sets, and Measure Theory'
title_zh: 实变函数：集合与点集、测度与可测函数
date: 2023-03-18 21:27:38 +0800
permalink: /blog/2023/03/18/real-analysis-notes/
categories:
- Mathematics
- Mathematical Analysis
tags:
- Real Analysis
- Measure Theory
excerpt: Covers sets and point sets, measures, measurable functions, Lebesgue integration, integration, and differentiation.
description: Covers sets and point sets, measures, measurable functions, Lebesgue integration, integration, and differentiation.
lang: en
translation_key: 2023-03-18-real-analysis-notes
translation_status: machine
translation_source_hash: c0ea7ab8b53b6e3235ef1adc6935091030c70f654c419e2aab1f291fa93292a1
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<h2>Introduction</h2>
<p>The content of the whole course is intended to address the limitations of Riemann points in mathematical analysis. Some pre-knowledge has been introduced to address the problem with regard to measurements, although the theory of high probability also needs to be based on measurement knowledge, which has been studied together.</p>
<p>The limits of Riemann's points are as follows:</p>
<ul>
<li>Too much requirement for continuity of functions, resulting in the lack of accumulation of the Dirichlet function Riemann</li>
<li>Too many requirements for the order of limits and points exchange</li>
<li>It's too complicated.</li>
<li>Use only in real terms</li>
</ul>
<p>That's why we decided to introduce Lebegue points.&#36;x&#36;The axes have been divided.&#36;y&#36;Axis division, circumvented.&#36;x&#36;The effect of an axis break point (inconsistent points of a function) on the result of the fraction.</p>
<p>Of course, because it's right.&#36;y&#36;The change in the way the axes are divided is more complex.&#36;x&#36;We naturally need a new set of theories to deal with the clustering of axes.</p>
<p>So we can introduce the narrative logic of this book.</p>
<ol>
<li>The theory of the collection and the dots helps us understand the new ones.&#36;x&#36;Axis Division</li>
<li>Study measurement theory to quantify new classification</li>
<li>Study of Lebegue points and related theorems</li>
</ol>
<h2>Gathering and Points</h2>
<h3>Gather!</h3>
<p>The basic theory of assembly has been introduced in the mathematical analysis, which is simply a review, while complementing the new concepts we need.</p>
<h4>The expression of assembly</h4>
<p>Here are some examples of a collection, some of which are familiar and others relatively unknown.</p>
<p>The most basic expression</p>
<ul>
<li>&#36;f(E)={f(x)|x\in E}&#36;</li>
<li>&#36;D\subset R\quad f^{-1}(D)={x\in E\mid f(x)\in D}&#36;</li>
</ul>
<p>Convergence of inclusion</p>
<ul>
<li>&#36;[a,b]\subset {x:f(x)\leq m}&#36;</li>
</ul>
<p>Group Columns &#36;{f_n(x)}&#36; Is a column function</p>
<ul>
<li>&#36;{x:\sup f_{n}(x)\leq c}=\cap_{n=1}^{\infty}{x:f_{n}(x)\leq c}&#36;</li>
<li>&#36;{x:\sup f_{n}(x)&gt;c}=\cup_{n=1}^{\infty}{x:f_{n}(x)&gt;c}&#36;</li>
<li>&#36;(a,b)=\bigcup_{n=1}^{\infty}[a+\frac{1}{n},b-\frac{1}{n}]&#36;</li>
<li>&#36;{x|f(x)&gt;0}=\cup_{n=1}^{\infty}{x|f(x)&gt;\frac{1}{n&#125;&#125;&#36;</li>
</ul>
<p>And the assembly column shall be a smaller representation than it has seen before, and it shall be presented as a collection by a row of delivery. There will be an innumerable replacement.</p>
<h4>The limit of the collection</h4>
<p>For a row of groups&#36;{A_n}&#36;  The definition is as follows:</p>
<ul>
<li>&#36;\cup_{n=1}^{\infty}\cap_{m=n}^{\infty}A_{m}=\underline{\lim_{n\to\infty}A_{n&#125;&#125;&#36;  For the bottom line, it means from a moment later. &#36;x&#36; More than one.&#36;A_n&#36;  (only in limited collections)</li>
<li>&#36;\cap_{n=1}^{\infty}\cup_{m=n}^{\infty}A_{m}=\overline{\lim_{n\to\infty}A_{n&#125;&#125;&#36;  The upper limit means &#36;x&#36; More than one.&#36;A_n&#36;</li>
<li>When? &#36;\underline{\lim_{n\to\infty}A_{n&#125;&#125; = \overline{\lim_{n\to\infty}A_{n&#125;&#125;&#36; It's called a collection with limits. &#36;lim _{n \to \infty }A_n&#36;</li>
</ul>
<p>There's a monotonous combination. &#36;A_{1}\subset A_{2}\subset A_{3}\cdots&#36;  And it's called a one-to-one-to-one-to-one-to-one-to-one-to-one-to-one-to-one-to-one-to-one-to-one.</p>
<p>If the pool increases or decreases, they must have a limit, which is the sum line or surrender.</p>
<h4>Number of elements assembled</h4>
<p>Definition: If there is a correspondence between the two pools (two-shot), the two pools are described as having the same base figure or representation &#36;|A|=|B|&#36;</p>
<p>Definitions: and collection&#36;N&#36;The homogeneity collection is called numeric/columnable in which the number of elements is recorded &#36;\aleph_0&#36;</p>
<p>Theoretically:&#36;A&#36;It's quantifiable, so he can be numbered into endless sequences.</p>
<p><strong>Equivalence is an equal value relationship, self-reverse, symmetry, transmission of three properties</strong></p>
<p>Theoretically:&#36;R&#36;It's incalculable, which means he has more than one element.&#36;\aleph_0&#36;  We'll take the base figure.&#36;c&#36;</p>
<p>They can all prove it by definition.</p>
<ul>
<li>Rational series is numbered.</li>
<li>A limited set and a quantifiable set</li>
<li>Any infinite set contains a quantifiable set (the smallest infinite set)</li>
<li>Quantifiable collections, at most.</li>
<li>The limited number of flutes is quantifiable.</li>
<li>Countable collections of wireless subsets</li>
<li>The whole composition of the algebra is numeric (not the number of algebras is the excess)</li>
</ul>
<p>Definition: A subset of A and B is equivalent to a base number less than the equivalent of B&#36;\bar{A}\le\bar{B}&#36;
Definition: A subset of A and B is equivalent but AB is not equal&lt;\bar{B}&#36;</p>
<p>Theoretically: the base number for the accumulation is&#36;c&#36;</p>
<p>Theorem (Berstein): Set&#36;A,B&#36;Two, if&#36;A \le B&#36;and &#36;A\ge B&#36;  then &#36;\overline{A} =\overline{B}&#36;</p>
<p>Cortor Theorem: The base of the collection is greater than the base of the original collection</p>
<p>For the continuous base, there's the following proposition:</p>
<ul>
<li>Comprising several consecutive bases</li>
<li>The aggregate of several consecutive bases has a continuous base</li>
<li>&#36;B_{n}= {0,1}&#36; But a few.&#36;B_n&#36;The straight amount is the continuous base</li>
<li>&#36;R^n&#36;The base figure is&#36;c&#36;</li>
<li>&#36;c&#36;Consistency base with a continuous base</li>
</ul>
<h3>Points</h3>
<p>We're here. <a href="/en/blog/2023/09/11/functional-analysis-notes/">The sprawl nature of measuring space in general correspondence analysis</a> and <a href="/en/blog/2024/10/16/point-set-topology-notes/">The space and continuity of tectonics.</a> The relevant theories are described in detail and need not be repeated here. Focus on the nature of points that can be used behind the measurement theory.</p>
<p>Theorem: Any non-empty opening on a straight line can be expressed as a maximum number of open areas and (refer to the cluster column that we described earlier)</p>
<p>Theoretically: For a closed set on the construction of a straight line, up to a few open spaces can be dug from the straight line.</p>
<p>Contor's Triple: By digging up the middle of each segment of the assembly.&#36;\frac{1}{3}&#36;Partly, it's very magical, and it seems contradictory.</p>
<ul>
<li>Contractor's trilogy is a complete set.</li>
<li>There's nothing in the Contor III episode, which means he's nowhere dense. - Yeah.</li>
<li>Consequencing base for the Contor III.&#36;c&#36;</li>
<li>The measure of the Contor tri-point is 0, which means the point is 1</li>
</ul>
<p>It's dense and not dense enough, with enough points but zero, two contradictions.</p>
<h2>Measure</h2>
<p>Thermological study of measurements is broader "long."</p>
<ul>
<li>Non-negative</li>
<li>Limited/quantifiable</li>
<li>Air mass is 0</li>
</ul>
<p>Of course, what we're after is Lebegue's measure, not the simplest one, and Lebegue's measure will have some unique properties that will help us study it.</p>
<h3>External measures</h3>
<p>Why are we talking about external measurements?</p>
<p>There are two types of measurements that are inadequate and surplus. They correspond to internal and external measurements, which are equally called measurable. Once we have the relevant knowledge of external measurements, we can continue to study them in depth.</p>
<p>Definitions:&#36;(a,b)&#36; Length is &#36;b-a&#36; Let's just remember &#36;|(a,b)|&#36;</p>
<p>Definitions:&#36;I=I_{1}\times I_{2}\times\cdots\times I_{n}&#36; of which&#36;I_i&#36;Yes. &#36;R&#36; Central &#36;|I|=|I_1|\times|I_2|\times\cdots\times|I_n|&#36;is its volume. When the compartments are open, they are called the squares.</p>
<p>Realistic functions are generally based on European-style space research, which is why this definition is given here.</p>
<p>Definitions:&#36;A\subset R^{n}&#36;  &#36;{I_k}&#36;Yes &#36;A&#36; , and only &#36;A\subset\cup_1^{\infty}I_{k}&#36;</p>
<p>Definitions:&#36;A&#36;Other Organiser &#36;v^{\star}(A)=\inf{\Sigma|I_k|}&#36; of which&#36;I_k&#36;A maximum of several overwhelms&#36;V,m&#36; They're all symbols.</p>
<p>It's obvious that the outer measure is a positive number.</p>
<p>Proof:&#36;R&#36;The outer measure of the limited and quantifiable set is 0, and the properties can be extended to&#36;R^n&#36; The nature of this is an expression of rational scarcity.</p>
<p>Does the external measure satisfy some of the characteristics we need?</p>
<ul>
<li>Non-negative: satisfaction</li>
<li>Addability: Not satisfactory, only the number of times that can be added</li>
<li>Airset is 0: Satisfactory</li>
<li>Regularity&#36;(0,1)&#36;Measure 1: Satisfactory</li>
</ul>
<p>Here's some of the arguments.</p>
<ul>
<li>Empty collection zeroing:&#36;v^{\star}(\phi) = 0&#36;</li>
<li>Single: &#36;A&lt;B\quad V^{<em>}(A)\leq V^{</em>}(B)&#36;</li>
<li>Alteration:&#36;{V}^{\star}(x_{0}+A)={V}^{\star}(A)&#36;</li>
<li>The following may be added:&#36;{V}^{\star}(\cup_1^{\infty}A_{k})\boxed{\leq}\sum_{1}^{\infty}V^{\star}(A_k)&#36;</li>
</ul>
<p><strong>It's annoying that the external measure doesn't match the amount of cosmopolitanity we like, but we don't have any better options, so we still use this theory.</strong></p>
<h3>Measurable</h3>
<p>With the concept of external measurements, how can measurable measurements be studied?</p>
<p>Definitions (measurable)&#36;E\subset R^n&#36; E can be measured.&#36;T\subset R^n&#36; Both.<em>(T)=m^</em>(T\cap E)+m^*(T\cap E^c)&#36; 此时称 &#36;^(E) (E external measure)</p>
<p>We can give the following theory:</p>
<ul>
<li>&#36;R^n,\phi&#36; It's measurable.</li>
<li>&#36;S&#36;Measurable rules &#36;S^c&#36; Measurable</li>
<li>&#36;S_1,S_2\text{可测 }\Rightarrow S_{1}\cup S_2\text{ 可测}&#36;</li>
<li>&#36;S_1S_2\text{ 可测 }S_1\cap S_2=\phi&#36; Do what you want.&#36;T&#36;  &#36;m^{\star}(T\cap {(S_1\cup S_2)}=m^{\star}(T\cap S_1)+m^{\star}(T\cap S_2)&#36;\</li>
<li>&#36;S_1S_2\text{ 可测 }S_1\cap S_2=\phi&#36;   &#36;m^{\star}(S_{1}\cup S_{2})=m^{\star}(S_1)+m^{\star}(S_2)&#36;</li>
<li>A limited set of detectable intersections.</li>
<li>&#36;S_1,S_2&#36;Measurable &#36;S_1-S_2&#36;Measurable and &#36;m (S 2)&lt;\infty&#36; 则 &#36;m(S_1-S_2)=m(S_1)-m(S_2)&#36;</li>
</ul>
<p>We can still take measurements to the limit.
&#36;S_i&#36;An incremental collection.&#36;S=\cup S_{i}= limS_n&#36; then&#36;m(S) = lim(m(S_n))&#36;  Declining is the same thing. It's essentially a collection and a shift in order.</p>
<h3>Measurable Clusters</h3>
<p>In abstraction of the nature that we have studied before, we explain which clusters are measurable.</p>
<p>The basic nature of measurable aggregations is</p>
<ul>
<li>&#36;\Omega \in \mathcal{A}&#36; (&#36;\phi \in \mathcal{A}&#36;)</li>
<li>&#36;\text{若}A\in \mathcal{A}\text{则}A^{c}\in \mathcal{A}&#36;</li>
<li>&#36;\text{若}A_1,A_2,\ldots,A_n\in \mathcal{A},\text{则}\sum_{i=1}^nA_i\in \mathcal{A}&#36;
That's actually what we're talking about in the high probability theory.&#36;sigma&#36;The algebra.&#36;\sigma&#36;Algebra</li>
</ul>
<p>So we can say, "Specific cluster is one."&#36;R^n&#36;Top&#36;\sigma&#36;Algebra</p>
<p>It's natural. We know.</p>
<ul>
<li>It's measurable anywhere.</li>
<li>Any collection is measurable</li>
<li>The Borel set from the opening is measurable.</li>
</ul>
<p>Theoretically: zero-measurement, zero-measurement subsets, capable of up to a few zero-measurements, all of which are measurable</p>
<p>Intuitively: a measurable source has an excellent boundary from which we can give theorem: If&#36;E&#36;is measurable</p>
<ul>
<li>&#36;mE=\inf{mG:G\text{是开集 }E\subset G}&#36; That's outside formality.</li>
<li>&#36;mE=\sup{mK,k\text{是紧集} K\subset E}&#36;  It's internal formality.</li>
</ul>
<p>On the contrary: if a collection meets internal and external formality, he can be measured.</p>
<p>We acknowledge the following:</p>
<ul>
<li>For any collection with a positive measure, there must be an incalculable object. Set</li>
<li>We can't find a better measure to allow the subset to be measured and meet the Legegue three.</li>
</ul>
<h2>Denumerable and Lebegue points</h2>
<h3>Definition of measurable functions</h3>
<p>The study of measurable functions is designed to ensure the existence of Lebegue points and lay the groundwork for the later study.</p>
<p>Our traditional real-value research.&#36;f:R\to R&#36;  This is not enough to study measurable functions, and we need to introduce a broad concept of real functions.</p>
<p>Definitions (extensive real functions): We allow research&#36;f:E\to R\cup(-\infty,+\infty)&#36;  Here.&#36;\infty&#36;It's a big number, bigger than a real number.&#36;E&#36;It's measurable. function. We studied it before.&#36;f:E\to R&#36;  Called a limited function (not a boundary function)</p>
<p>Definitions (measurable functions):&#36;f:E\to R\cup(-\infty,+\infty)&#36; and only if &#36;{x\inE; f(x)&gt;a}&#36; 是可测集 记作 &#36;E[f&gt;a]&#36;  &#36;\forall a\in R&#36;</p>
<p>Theoretically: The detectable function has the following qualification:</p>
<ul>
<li>&#36;E[f\le a]可测\forall a\in R&#36;</li>
<li>&#36;E[f\ge a]可测\forall a\in R&#36;</li>
<li>&#36;E[f&lt; \\forall a\in R&#36;</li>
</ul>
<p>Inference:&#36;f可测\Rightarrow E[f=a]是可测集&#36;  <strong>That's what research can do.</strong></p>
<p>In particular, we can approach non-negative detectable functions with non-negative simple functions, and use two non-negative detectable functions similar to the main and negative of the detectable functions, ultimately bringing all detectable functions closer with simple functions.<strong>It's because simple functions have very good Lebegue points, and that's a good way to define the Lebegue points and even all the measurements.</strong></p>
<h3>Measurable Operations</h3>
<p>Easy to know:&#36;f(x)=c&#36; Constant function is measurable</p>
<p>By definition, we can easily prove that:<strong>Four operations of measurable functions remain measurable</strong></p>
<p>Use definitions to prove that the absolute value of measurable functions is still measurable</p>
<p>Now let's look at a few special questions.</p>
<p>For detectable function columns&#36;{f_n(x)}&#36;  Research&#36;sup(f_n(x)),inf(f_n(x))&#36; The detectability.
&#36;&#36;E[\sup f_n(x)\geq a]=\cap E[f_n(x)\geq a]&#36;&#36;
So detectable.&#36;sup,inf&#36;Keeps it closed, which is a good character not available for continuous functions. It can be extended to <strong>The upper and lower limits of the detectable function are also measured</strong></p>
<p>For detectable functions&#36;f(x)&#36;  Research&#36;f^+(x),f^-(x)&#36;For me, it is possible to verify by definition that the nature does exist.</p>
<p>In conclusion, we can give some classic examples of measurable functions.</p>
<ul>
<li>Constant function detectable</li>
<li>Continuous function detectable</li>
<li>&#36;[a,b]&#36;Single-tom function to measure</li>
<li>Simple function detectable (Dirichlet function is also a simple function)</li>
</ul>
<h3>Consistency of measurable function columns</h3>
<p>We extend the experience of all studies of function tiers to detectable functions.</p>
<p>Retrospect: Consistency is stronger than one. In order to ensure continuity, the narrowing of the zone may lead to a convergence of convergence.</p>
<p>Supplement: Almost everywhere, after the elimination of a zero-measurement, recorded as a.e.</p>
<p>Yegorov Theorem: For a column of measurable functions&#36;f_k(x)&#36; One in...&#36;E&#36;Detectable Functions on&#36;f&#36; , they are all a.e. limited and &#36;m(E)&lt;\infty&#36; 。若&#36;f_k(x)&#36;几乎处处收敛于&#36;f&#36;，则&#36;f_k(x)&#36;几乎处处一致收敛于&#36;f&#36;</p>
<p>This is Yegorov's Theorem, which looks at the point-by-point connection of detectable functions to coherent contraction.</p>
<p>Now, let's look at a kind of condensity.</p>
<p>Definitions (depression by measure):&#36;{f_n}&#36;The definition is measurable.&#36;E&#36;A limited number of detectable function sequences up there, if any&#36;E&#36;Limited detectable functions above&#36;f&#36;Meets the following:&gt;Yes, I do.
&#36;&#36;\lim_{n\to\infty}\mu({x\in E:|f_n(x)-f(x)|\geq\varepsilon})=0,&#36;&#36;
function series&#36;{f_n}&#36;Condense by measure&#36;f&#36;, or measure constriction&#36;f&#36;。</p>
<p>In a more intuitive way, it's with.&#36;n&#36;The increase, the deviation.&#36;f&#36;More than precision&#36;\varepsilon&#36;Yes.&#36;f_n&#36;The measure of the formation pool is zero.</p>
<p><strong>As can be seen, a degree of concentration is weaker than a point-by-point one, and can be seen in the context of primary probability.</strong></p>
<p>While we say that the point-by-point reduction is almost complete, and the convergence is weak, the latter cannot deduce the former without additional conditions, and we cannot withdraw anything. If, however, conditions are to be attached, the following is the theorem.</p>
<p>(Leberg) Theorem: a column detectable functions&#36;f_n(x)&#36;,&#36;f&#36;Yeah.&#36;E&#36;upper detectable functions, which are almost everywhere, &#36;m(E)&lt;\infty&#36;, 则若&#36;f_n(x)&#36;几乎处处收敛于&#36;f&#36;,则&#36;f_n(x)&#36;依测度收敛于&#36;f&#36;。</p>
<p>(Ris) Theorem:&#36;f_n&#36;Condense by measure&#36;f&#36;, there are sub-bars&#36;f_{n_k}&#36;a.e. Repression&#36;f&#36;。</p>
<h3>Definition of Lebegue points</h3>
<p>The Lebegue points were introduced to deal with the inexhaustible circumstances of Riemann.&#36;x&#36;And the division is right.&#36;y&#36;Division.</p>
<p>We refer here to the previous approach to simple functions approaching detectable functions: non-negative simple functions can be used to approach non-negative detectable functions, and the positive and negative parts of detectable functions can be separated from the non-negative detectability approximations and ultimately from simple functions to general detectable functions. A simple function has a good Lebesgue crediting properties and is therefore suitable as a starting point for defining the Lebesgue crediting as well as the general measure score.
After the definition of points, let's study the nature of some Lebegue points. And the section on calibration in this paper entitled "Specific and calibration"</p>
<h4>Non-negative simple function</h4>
<p>&#36;&#36;E=U_{i=1}^{n}E_{i}\quad f(x)=C\quad\forall x\in E_i&#36;&#36;
From the vertical axis, we can naturally define Lebegue's size.
&#36;&#36;\int_{E}f(x)dx=\sum c_i\cdot m(E_i)&#36;&#36;</p>
<p>So we can naturally give some of the Lebegue points, and many of them can be extended back to what we study.</p>
<ul>
<li>&#36;A\subset E&#36; There is. &#36;\int_{A}f(x)dx=\sum c_{i}m(E_{i}\cap A)&#36;</li>
<li>&#36;A,B\subset E\quad A\cap B=\phi \quad\text{则}\int_{A\cup B}f(x)dx=\int_{A}f(x)dx+\int_{B}f(x)dx&#36;</li>
<li>For a row&#36;A_n,E=limA_n&#36;  then&#36;\int_{E}f(x)dx=\lim_{x\to\infty}\int_{An}f(x)dx&#36;</li>
<li>&#36;\int_{E}(\alpha f(x)+\beta g(x)dx=\alpha\int_{E}f(x)dx+\beta\int_{E}g(x)dx&#36;</li>
<li>Zero-measured accumulation divided into zero.</li>
</ul>
<h4>Non-negative detectable functions</h4>
<p>Naturally, we can find a column of non-negative functions that makes &#36;lim_{k\to \infty}\phi_k(x)=f(x)&#36; So we can define it.
&#36;&#36;\int E}f(x)dx=sup{E}\phi(x)dx,\phi(x)\text{is a non-negative function} phil(x)&lt;f(x)}&#36;&#36;</p>
<p>We can naturally find nature if &#36;\varphi(x)&lt; \psi(x)&#36; 则 &#36;\int_{E}\varphi(x)dx&lt;\int E}\psi(x)dx&#36; is established for all non-negative detectable functions.</p>
<p>Definitions: If &#36;\int E}f(x)dx&lt;\infty&#36; 则称为&#36;f(x)&#36;在&#36;E.A. Lebegue.</p>
<p>Theoretically: If&#36;f(x)&#36;Lebegue has accumulated, and he must be almost everywhere.</p>
<p>Nature: The five properties we have studied earlier are established in non-negative detectability functions.</p>
<p>Inference:</p>
<ul>
<li>Zero measurements do not affect Lebegue points, so the nature described above remains valid at a.e.</li>
<li>If&#36;f(x)=g(x)&#36; a.e. Establishment&#36;\int_{E}f(x)dx=\int_{E}g(x)dx&#36;</li>
<li>If&#36;f(x)=0&#36;  a.e. Establishment&#36;\int_{E}f(x)dx=0&#36;</li>
</ul>
<p>We can give three important theorems when we're looking at the non-negative leg of Lebegue.</p>
<p><strong>Livil Theorem</strong>: For an incremental function column, limits and points can be exchanged in order
&#36;&#36;\lim_{n\to\infty}\int_Ef(x)dx=\int_E\lim_{n\to\infty}f_n(x)dx&#36;&#36;</p>
<p><strong>Item by Item</strong>: A cumulative function can be converted to an item-by-item fraction
&#36;&#36;\int_E(\sum f_n(x))dx=\sum\int_{E}f_n(x)dx&#36;&#36;</p>
<p><strong>Fatou's reasoning</strong>: The fraction of the lower limit of the function series does not exceed the lower limit of the function sequence.</p>
<p>Set&#36;(f_n)&#36;The definition is measurable.&#36;E&#36;. The following variations are then established:
&#36;&#36;\begin{aligned}\int_E\liminf_{n\to\infty}f_n:d\mu\leq\liminf_{n\to\infty}\int_Ef_n:d\mu.\end{aligned}&#36;&#36;
Here.&#36;\lim\inf_{n\to\infty}f_n&#36;For function series&#36;(f_n)&#36;The point-by-point limit, for each&#36;x\in E,\lim\inf_n\to\infty f_n(x)=\lim_{n\to\infty}\inf_{k\geq n}f_k(x)&#36;</p>
<h4>General detectable functions</h4>
<p>Naturally, we make a distinction between straight and negative.</p>
<p>Definitions:&#36;\int_{E}f(x)dx=\int_{E}f^{+}(x)dx-\int_{E}f^{-}(x)dx&#36; When one of the two points is limited, we call the points certain, and when the margin is limited, the Lebegue points.</p>
<p>The nature of the points proposed above can be promoted naturally, and we give a distinct character from Riemann's.</p>
<p>Theoretically:&#36;f&#36;When it's measurable,&#36;f&#36; - Lebegue, yes. &#36;|f|&#36; Lebegue</p>
<p>E.g. &#36;|f(x)|&lt;g(x)\quad a.e.E\quad\quadg(x)\text{non-negative L build} It's okay.
There are,&#36;f(x)\in L(E)&#36;</p>
<p>Prove it.
&#36;&#36;|\int_{E}f(x)dx|\leq\int_{E}|f(x)|dx\leq\int_{E}g(x)dx&#36;&#36;
That is, ** L-accumulation is guaranteed by the capacity control. That's not what Riemann points are.</p>
<h2>Scores and Scores</h2>
<h3>Lebegue controls the theorem and application</h3>
<p>This is an extension of the three theorems of the Livil theorem.</p>
<p><strong>Lebesgue's theorem for containment.</strong>: Set&#36;{f_n}&#36;The definition is measurable.&#36;E&#36;, with the following conditions:</p>
<ul>
<li>For all&#36;n&#36;Yes.&#36;f_n(x)\to f(x)&#36;Almost everywhere (a.e.)&#36;E&#36;There's a collection of zeros.&#36;N\subset E&#36;, make it all&#36;x\in E\setminus N&#36;♪ When ♪&#36;n\to\infty&#36;I don't know.&#36;f_n(x)\to f(x)&#36;。</li>
<li>There is an amplifiable function&#36;g&#36;(i)&#36;g\in L^1(E)&#36;) makes for all&#36;n&#36;And almost everything.&#36;x\in E&#36;Yes.&#36;|f_n(x)|\leq g(x)&#36;
Well, the function.&#36;f&#36;It's stoked and:
&#36;&#36;\lim\limits_{n\to\infty}\int\limits_{E}f_{n}:d\mu=\int\limits_{E}f:d\mu.&#36;&#36;</li>
</ul>
<p>The intuitive meaning of this theory is that if a function sequence is condensed almost everywhere to a function and the sequence is controlled by an amplifiable function, the integral limit of the series is equal to the consumable function. Which means...<strong>The points and the limits can be exchanged.</strong></p>
<p>In accordance with Lebugue's theory of containment control, we can come up with two more propositions about the severing order.</p>
<p>For infinity:&#36;\sum f_{n}(x)=\lim_{n\to\infty}\sum f_{k}(x)&#36;  Remove
&#36;&#36;F(x)=\sum_{1}^{\infty}|f_{k}(x)|&#36;&#36;
As a control function, then just
&#36;US&#36;\int E}sum|f (x)|dx&lt;\infty&#36;&#36;
就可以保证原始无穷级数收敛，并且可以进行换序有
&#36;&#36;\int_{E}\sum f_{n}(x)dx=\sum\int_{E}f_{n}(x)dx&#36;&#36;</p>
<p>For guidance and crediting sequences:&#36;\lim_{n\to+\infty}\frac{f(x,t+h)-f(x,t)}{h}&#36; We can also prove by looking for control functions that use Lebegue to control the insinuation theorem, that this control function can be found, so guidance and points can be changed.</p>
<h3>Riemann points and Lebegue points</h3>
<p>We talked about it while studying Riemann points.</p>
<ul>
<li>When the amplitude is almost zero, Riemann can build up.</li>
<li>When it's almost continuous, Riemann can build up.
On this basis, we can prove that a single-modular function, Riemann, is available because it does not have a point of zero. Measure</li>
</ul>
<p>So what's the link between Lebegue and Riemann?</p>
<p>We know that Lebegue points can deal with some of Riemann ' s inaccumulations, but the current method of relying on definitions makes the Lebegue points difficult to calculate, so we can give theoretics.</p>
<p><strong>Riemann's staggered function is certain that Lebugue is available and its Lebegue and Riemann's are equals</strong></p>
<p>In other words, Lebegue is the wider Riemann score, which is compatible with the Riemann score previously studied.</p>
<p>For an abnormal fraction there is</p>
<p>Theorem: Yes.&#36;[a,b]&#36;non-negative functions on&#36;f&#36;  If \forall A&gt;0&#36; 都有 &#36;[a,A]&#36; 上 &#36;f&#36; 是Riemann可积的并且反常积分收敛，则&#36;f&#36;在&#36;[a, \\intty] &#36; on Legebue is available and has an equal value. Legebue is not available if the anomaly distribution is dispersed</p>
<p>For functions that are not negative&#36;f&#36;  The Legebue score is not an extension of the Riemann score. Indeed, the abnormal Riemann points themselves do not fall within the Riemann points and do not satisfy their definition.</p>
<h3>Lebegue geometry and Fubini theorem</h3>
<p>Let's start with the conclusion that the measured volume is still measurable and satisfactory.
&#36;&#36;m(A\times B)=m(A)\times m(B)&#36;&#36;</p>
<p>Definition: Below Graphics&#36;G(E,f)&#36;Is a function&#36;f&#36;In defined range&#36;E&#36;Graphical area below the curve above</p>
<p>Theoretically: the geometrical meaning of the Lebegue points of a non-negligible function is</p>
<ul>
<li>&#36;f&#36;It's measurable. &#36;\Longleftrightarrow&#36;  Graphical area below is measurable</li>
<li>Lebegue points are the size of the next graphic.</li>
</ul>
<p>Inference: function that is general and is measurable in Legebue&#36;f&#36;</p>
<ul>
<li>&#36;\int_{E}f(x)dx=G(E,f^{+})-G(E,f^{-})&#36;</li>
<li>&#36;G(E,f^{+}),G(E,f^{-})&#36; The area is limited.</li>
</ul>
<p><strong>Fubini Theorem</strong>: The core theorem for the calculation of the rejuvenation using a step-by-step fraction</p>
<p>Yeah.&#36;A\subset R^{p},B\subset R^q&#36; ..of the detectable collection</p>
<p>If&#36;f(D)=f(x,y)&#36; Yes.&#36;A\times B&#36; A.e. &#36;\forall x\in A&#36; &#36;f(x,y)&#36;As&#36;y&#36;function in&#36;B&#36;And...
&#36;&#36;\int_{A\times B}f(d)dd=\int_{A}dx\int_{B}f(x,y)dx&#36;&#36;</p>
<p>If&#36;f(D)&#36;Yes.&#36;A\times B&#36;A.e. &#36;\forall x\in A&#36; &#36;f(x,y)&#36;As&#36;y&#36;function in&#36;B&#36;A.e. &#36;\forall y\in B&#36; &#36;f(x,y)&#36;As&#36;y&#36;function in&#36;B&#36;And...
&#36;&#36;\int_{A\times B}f(d)dd=\int_{A}dx\int_{B}f(x,y)dx&#36;&#36;</p>
<h3>Lebegue Theorem</h3>
<p>Starting with this section, we're looking at the issue of Lebegue's points of reference.</p>
<p>As we know earlier, the point of discontinuity in a single-telephone function is to be numbered at most.</p>
<p>Now give.<strong>Lebegue Theorem</strong>: Set&#36;f(x)&#36;Yes.&#36;[a,b]&#36;, and then</p>
<ul>
<li>&#36;f(x)&#36;Yes.&#36;[a,b]&#36;Virtually everywhere is a guide number&#36;f^{\prime}(x)&#36; / &#36;f(x)&#36;Almost everywhere.</li>
<li>&#36;f(x)&#36;Yes.&#36;[a,b]&#36;Cumulative</li>
<li>If&#36;f(x)&#36;is an addition, then&#36;\int_{a}^{b}f^{\prime}(x)\leq f(b)-f(a)&#36;</li>
</ul>
<h3>Variable function</h3>
<p>In the area of Riemann ' s points, the points and conductors are counter-measured, and this is a possible extension to Lebegue ' s points.</p>
<p>We know that direct extension is completely impossible, and exactly what kind of function would satisfy the superior nature of opposite calculations.</p>
<p>Definition (with variation function): If&#36;f(x)&#36;Yes.&#36;[a,b]&#36; , if&#36;f(x)=g(x)-h(x)&#36;  and&#36;g(x),h(x)&#36;Both.&#36;[a,b]&#36;Add function to&#36;f(x)&#36;is a variation function.</p>
<p>Equivalent definition (with variable functions):&#36;f(x)&#36;Yes.&#36;[a,b]&#36; limited function, give split&#36;\Gamma&#36; If&#36;{\sum|f(x_i)-f(x_{i-1})|}&#36; There's a line under any division.&#36;f(x)&#36;is a variation function.</p>
<p>According to the Lebegue Control Consortance Theorem: if there is a mutation function, then&#36;f(x)&#36;Yes.&#36;[a,b]&#36;There's almost a wizard up there.&#36;f^{\prime}(x)&#36; It's I can build.</p>
<p>The mutation function has the following properties.</p>
<ul>
<li>There's a mutation.</li>
<li>If&#36;f,g&#36;is a variation function, and&#36;cf,f+g&#36;It's a mutation function.</li>
<li>If&#36;f,g&#36;, and then&#36;fg&#36;It's a mutation function.</li>
<li>If&#36;f&#36;Yes.&#36;A&#36;there is a mutation function above, then&#36;f&#36; Yes. &#36;A&#36; There's a mutation function on the subset.</li>
</ul>
<h3>calculus basic theorem</h3>
<p>We know we've got an L-enabled guide.</p>
<p>In fact, we cannot guarantee that all the variable functions satisfy the N-L formula, but only that the functions that satisfy the N-L formula are variable, so we need to study it further.</p>
<p>We're here to continue to study the issues of points and calibration sequences, which are at the heart of the N-L formula.</p>
<h4>I'll take it first.</h4>
<p>Theoretically:&#36;f(x)&#36;Yes.&#36;[a,b]&#36;upper L-capable function, then
&#36;&#36;\frac{\mathrm{d&#125;&#125;{\mathrm{d}x}[\int_{a}^{x}f(t)dt]=f(x)&#36;&#36;
a.e. &#36;[a,b]&#36;</p>
<h4>Numerical and integral.</h4>
<p>Which means...
&#36;&#36;\int_{a}^{x}f^{\prime}(t)dt=f(x)-f(a)&#36;&#36;
This nature is not valid for all L-capable variable functions.</p>
<h4>It's absolutely continuous.</h4>
<p>Definition (unusual function): the yield is almost zero, but not a constant function (hard to construct, requires a set of Contractors)</p>
<p>Definition (absolute continuous function): a continuous function of the same kind as an odd function</p>
<p>Theoretically:&#36;g(x)\in L[a,b]&#36; Time&#36;\int_{a}^{x}g(t)dt&#36; Absolutely continuous.</p>
<p>Theoretically:&#36;[a,b]&#36;Absolute continuous functions constitute linear space</p>
<p>Theorem: The absolute continuous function is a mutation function</p>
<h4>calculus basic theorem</h4>
<p>In the light of the above, we give you<strong>The basic calculus theorem of Lebegue points</strong>：&#36;f(x)&#36; Yes.&#36;[a,b]&#36;In absolute continuity, yes.
&#36;&#36;\int_{a}^{x}f^{\prime}(t)dt=f(x)-f(a)&#36;&#36;</p>
