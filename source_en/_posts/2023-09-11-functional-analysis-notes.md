---
title: 'Functional Analysis: Metric Spaces, Compactness, and Banach Spaces'
title_zh: 泛函分析：度量空间、紧性与可分性
date: 2023-09-11 21:36:59 +0800
categories:
- Mathematics
- Mathematical Analysis
tags:
- Functional Analysis
- Metric Spaces
author: Hyacehila
mathjax: true
hidden: true
excerpt: Covers metric spaces, compactness, separability, completeness, normed linear spaces, Banach spaces, and linear operators.
description: Covers metric spaces, compactness, separability, completeness, normed linear spaces, Banach spaces, and linear
  operators.
excerpt_zh: 整理度量空间、紧性、可分性、完备性、线性赋范空间、Banach 空间和线性算子等内容。
permalink: /blog/2023/09/11/functional-analysis-notes/
lang: en
translation_key: 2023-09-11-functional-analysis-notes
translation_status: machine
translation_source_hash: de7d0270bb9f4c9a6971cd214046bb4f1213d3ee8c0b9094def719964f7ddf44
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<p>In the mathematical analysis, we studied core calculus of the whole higher mathematical theory; the subsequent multiple functions extended the theory to the plural, and the real-life function introduced the broader Leberg points; and the more numerous functions of general analysis, studying them as they are of the same nature as calculus, which is one of the most important basic lessons to follow up on a very large number of courses.</p>
<p>The questions in this article can also be addressed<a href="/en/blog/2023/03/16/mathematical-analysis-limits-continuity-notes/">Mathematical analysis: the theory of limits and continuity</a>How the concept of a relatively close read together is developed in different contexts.</p>
<h2>Measurement Space</h2>
<p>The calculus theory is about functions, and we start with a lot of things from the limit and the continuous function; the boundaries of the painting need to be based on distance; this is what needs to be studied in the measurement space; we want to study the distance more broadly.</p>
<h3>Definition of measuring space</h3>
<h4>Definition of measuring space</h4>
<p>Set non-empty collections&#36;X&#36;, binary map exists&#36;d(x,y)\to R&#36; Make it random.&#36;x,y&#36;belong&#36;X&#36; The following conditions are met:</p>
<ol>
<li>Non-negative (Positive):&#36;d(x,y)\ge 0 当且仅有x=y时取等号&#36;</li>
<li>Symmetry:&#36;d(x,y)=d(y,x)&#36;</li>
<li>Triangular inequity:&#36;d(x,y)\le d(x,z)+d(y,z)&#36;</li>
</ol>
<p>Name&#36;d&#36;Time&#36;x&#36;A distance function in the previous&#36;(x,d)&#36;It's a distance.&#36;d&#36;The result is distance.</p>
<h4>Examples</h4>
<h5>European Space</h5>
<p>Take Space&#36;R^{n}={(x_{1},x_{2},x_{3}...x_{n}|x_{i}\in R)}&#36;
Take Distance Functions&#36;d(x,y)=\sqrt{\sum\limits(x_{i}-y_{i})^{2&#125;&#125;&#36;
Proves he's a measuring space.</p>
<p>(a) Use of definitions to verify;
&#36;&#36;\left[\sum_{i=1}^{n}\left(x_{i}-y_{i}\right)^{2}\right]^{\frac{1}{2&#125;&#125;=\left[\sum_{i=1}^{n}\left(x_{i}-z_{i}+z_{i}-y_{i}\right)^{2}\right]^{1/2}\leqslant\left[\sum_{i=1}^{n}\left(x_{i}-z_{i}\right)^{2}\right]^{1/2}+\left[\sum_{i=1}^{n}\left(z_{i}-y_{i}\right)^{2}\right]^{1/2}&#36;&#36;
The axiom and symmetry are clearly satisfied.
We need to use Minkowski's iniquities, and we can prove it.</p>
<p>The formula is as follows:
&#36;&#36;00\&amp;&amp;(\sum_{i=1}^n|\left.a_i+b_i\right|^k)^{\frac1k}\leqslant(\sum_{i=1}^n|\left.a_i\right|^k)^{\frac1k}+(\sum_{i=1}^n\left|\left.b_i\right|^k\right)^{\frac1k}.\end{aligned}&#36;&#36;
k偶数的时候就可以去除绝对值符号了
这个公式还有一个积分形式 使用勒贝格积分 也非常常用 如下
&#36;&#36;\\left^f\left^g\left^right^s}mathrm^d^right^(s}matft^xright^)^ I'm sorry.
I can see this is a very useful equation.</p>
<p>We call the above defined distance function the standard European distance; there are other ways of defining, for example, &#36;d(x,y)=max{|x_{k}-y_{k}|}&#36;   &#36;d(x,y)=\sum\limits{|x_{k}-y_{k}|}&#36;   (a) Wait;</p>
<p><strong>There are many ways to define the same space, to form a variety of measures, and measure space is the whole of space and measurement functions.</strong></p>
<h5>Dispersive measure space</h5>
<p>Defines the distance function for any non-empty collection X as follows:
&#36;&#36;d_{0\left(x,y\right)}=\left|\begin{matrix}0,x=y,\1,x\neq y.\end{matrix}\right.&#36;&#36;
It's easy to verify that it meets three conditions for measuring space, which we call dispersive measure space.
Any kind of non-empty collection always defines measuring space like this, and he's a very special measuring space, and we'll use it a lot later.</p>
<h5>Continuous Function Space</h5>
<p>Set&#36;C[a,b]={f:[a,b]\to R|f连续}&#36; Defines the distance function as
&#36;&#36;d\left(f,g\right)=\max_{t\in\left[a,b\right]}\left|f\left(t\right)-g\left(t\right)\right|&#36;&#36;
Very clearly meets symmetry and non-negativeness, and only needs to be used to prove triangulation.<strong>Classic absolute value triangles are being expanded</strong>(a) The use of the Internet;
We call this space a continuous function.
There's a way of defining distance in the continuous function space.
&#36;&#36;d\left(f,g\right)=\int_{a}^{b}\left|f(x)-g(x)\right|\mathrm{d}x&#36;&#36;</p>
<h5>There's a boundary array space</h5>
<p>=lts, \cdots, \cdots\right =sup \i\gqslant1}&lt;\infty\right.&#36;  对于&#36;x=(x i}, y=(y i}\in l^(info}&#36;
&#36;&#36;d\left(x,y\right)=\sup\left|x_{i}-y_{i}\right|,&#36;&#36;
It's a measuring space, as it is.&#36;l^{\infty}&#36;</p>
<h5>P-salent array space</h5>
<p>It's a promotion of the original European space.
&#36;&#36;00\(x 1,\cdots,\cdots)=(x n,\cdots)\sum sum=sm sm sxsm sm}sm}sm}sm}sm}sm}sm}sm}sm}sm}sm}sm}sm}sm}sm}sm}sm}sm}sm}sm}sm}sm}sm}sm}sm}sm}sm}sm}sm}sm}sm}sm}sm}sm}sm}sm=sm}sm}sm}sm}sm}sm}sm}sm}sm}sm}sm}sm}sm}sm}sm}sm}sm}sm}s}sm}s}sm}s\sm}s\sm}sm}s\s\s\s\s\s\s\s\s\s\s&lt;\alpha\text{, where)&lt;+alpha,\text{for}forforall x=(x i), y=(y i)\in l^p,\text{Definition}\end{aligned}&#36;&#36;
&#36;&#36;d p(x,y)=(sum i i y i i^p) {frac1p}
It's a measuring space, and it's generally recorded as...&#36;l^{p}&#36;</p>
<h5>P-scrutinized function space</h5>
<p>It's a promotion of the form of points in continuous function space.
&#36;&#36;00\&amp;\text{L^p}a,b}f(t)\int\[a,b]}\midf(t)\mid^p\mathrm{d}t&lt;+\infty}\&amp;\text{inter alia}&lt;+\infty,\int [a,b]}\midf(t)\mid^pmathrm{d}t\text{d}(f(t)^p\text{in \a,b]\text{) \Lerberg \&amp;L^p[a,b]\text{is almost everywhere equivalent to the same function. For}f, g\in L^s[a,b]\text{, define distance}&amp;d(f,g)=(\int [a,b]}|f(t)-g(t)\mid^s\mathrm{d})^(frac1p},\end{aligned}&#36;&#36;&#36;
Could verify the definition of the satisfaction measure space</p>
<p>So all the important measures and some of the proofs they're involved in have been given, and we'll start to study the nature of the space.</p>
<h3>Measuring the expanse of space</h3>
<p>This section requires reference<a href="/en/blog/2024/10/16/point-set-topology-notes/">I'll give you a little space for the poking in the thaw.</a></p>
<h4>Basic expansional nature</h4>
<p>Definition (near): &#36;O (0},\sigma)=x|d (x,x)&lt;It's called a neighborhood.</p>
<p>Definition (Inline): an area within a pool is an adjacent area within this pool, and this point is an inner point.</p>
<p>Definition: All points of a collection are his inner points, which is&#36;intG=G&#36;</p>
<p>Definitions (closed): If&#36;F^{C}=X\setminus F&#36; It's the opening.&#36;F&#36;It's closed.</p>
<p>Nature of the opening collection</p>
<ul>
<li>Empty collections and space-wide collections.</li>
<li>Any more, and all of it.</li>
<li>The limited openings are the openings.</li>
</ul>
<p>Closed meeting:</p>
<ul>
<li>Empty and full space are closed.</li>
<li>Any kind of closed encounter is closed.</li>
<li>The limited closed collections are closed.</li>
</ul>
<h4>More thawing.</h4>
<p>Definitions (units and closed)
Set &#36;A&#36; It's a space for poking. &#36;X&#36; Subsets,&#36;xin X&#36;I'm sorry. If &#36;x&#36; Every neighborhood contains &#36;Aackslash{x}&#36; midpoint, then called &#36;x&#36; Yes &#36;A&#36; (a) The focal point for the implementation of the Convention;&#36;A&#36; All the gatherings of the world's most popular places are called &#36;A&#36; ♪ The collection of the world's most famous &#36;A^{prime}&#36;I'm sorry. Call it a rally. &#36;overline A:=Aigcup A^{prime}&#36; Yes &#36;A&#36; - The lockdown.
I can see that the gathering points are not necessarily in the assembly.</p>
<p>Theorem: The following propositions are equivalent</p>
<ul>
<li>&#36;x_{0}\in A^{&#39;}&#36;当且仅当存在&#36;&#123;x_{n&#125;&#125;\in A,s.t~ ~\lim_{x \to \infty} x_{n}=x_{0}&#36;</li>
<li>&#36;\bar{A}&#36;It's closed.</li>
<li>&#36;A&#36;It's closed and it's just...&#36;A=\bar{A}&#36;</li>
<li>If it exists&#36;F\subset X&#36;Make&#36;A\subset F&#36; then&#36;A\subset\bar{A}\subset F&#36;</li>
</ul>
<p><strong>This part of the theory and the inference tells us that the collection of closed packages is the smallest of his closed collections, and all the closed encounters older than him.</strong></p>
<h3>Measurement of space limits and continuity</h3>
<h4>Limits</h4>
<p>The theory of limits is the basis of the calculus theory.</p>
<p>Definition (limits):&#36;\lim_{x \to \infty}d(x_{n},x_{0})=0&#36; - It's just a condolence.&#36;x_{0}&#36;Yes.&#36;x_{n}&#36;♪ The limits of the other side ♪</p>
<p><strong>The array is relevant to the selection of the measured space, and&#36;X&#36;and&#36;d&#36;It's all about it.</strong></p>
<p>Definition (subspace): For measuring space&#36;(X,d)&#36; Obviously for the subset&#36;A&#36;There's room for measurement.&#36;(A,d)&#36; It's called subspace.</p>
<p>Define (range): Point to assembly distance is&#36;d(x,A)=inf_{y\in A}{d(x,y)}&#36;</p>
<p>Definition (in diameter): Pooling&#36;A&#36;The diameter is&#36;diaA=sup_{x,y\in A}{d(x,y)}&#36;</p>
<p><strong>The only thing that's limited in diameter is a boundary, and the opposite is a borderless one. Set</strong></p>
<p>Theorem (the nature of the limit):</p>
<ul>
<li>Pointing in, and the only limit.</li>
<li>The line is also constricted and has the same limits.</li>
<li>The Tilt Line is a collection. He's a part of the world. Set</li>
</ul>
<h4>Continuous and consistent</h4>
<p>This question needs to be addressed to the mapping.</p>
<p>Definitions:
Set&#36;(X,d),(Y,d)&#36;It's two measuring spaces. &#36;f&#36;It's a map above them.
&#36;&#36;00\&amp;\text{for \xx,\text{if}forforall\varepsilon&gt;0,\exists\delta&gt;0,\text{x\text{and}d(x,x 0)&lt;\delta\&amp;\\text{f\text{x\text{x\text{x\text{x}[x\text{x]}f\text{x ]}Presistence. If f\text{sync}Presistence at each point x}, f\text{x\text{x\text{symper}.&amp;\text{if \form\epsilon&gt;0,\exists\delta&gt;0, {forall x, y\in X, \\text{d(x,y)\leq\delta}\text{, \rho(x), f(y))&lt;\\varepsilon\text{, otherwise }&amp;f\text{x\text{x\text{consequence}end{aligned} &#36;
<strong>Like mathematical analysis, the concept of continuity is local, and the concept of consistency is holistic.</strong>
Function Continuous and consistent continuous equivalents in the closed interval</p>
<p>Theorem (continuous equivalent)
The following propositions are equivalent</p>
<ul>
<li>Map&#36;f&#36;Yes.&#36;x_{0}&#36;Point-in-line</li>
<li>Existence&#36;f[O(x_0,\delta)]\subset O(f(x_0),\varepsilon).&#36;</li>
<li>&#36;\lim_{n\to\infty}f\left(x_{n}\right)=f\left(x_{0}\right)<del>if</del> Other Organiser
<em>The intuitive thought is proof.</em>
Theorem (continuous filling)
&#36;&#36;\text{对}Y\text{中的任一}\text{开集}G,\text{其原像}f^{-1}\left(G\right)={x|x\in X,f\left(x\right)\in G}\text{是开集}&#36;&#36;
It's a continuous requirement.</li>
</ul>
<p>There is also an equivalent form of the theory that closed-sets are a constant and essential condition.</p>
<h3>Measuring the spectrometry of space</h3>
<h4>Bulk definition</h4>
<p>Definitions (breathing)
Set&#36;X&#36;It's measuring space.&#36;A,B\subset X&#36; If&#36;B&#36;Any area of any point contains&#36;A&#36;Middle Point is called&#36;A&#36;Yes.&#36;B&#36;♪ Middle dense if ♪&#36;A&#36;Exactly.&#36;B&#36;The subset, so called&#36;A&#36;Yes.&#36;B&#36;A dense subset
It's very easy to understand that the rational numbers are dense in the irrational and the actual numbers, the irrational numbers are dense in the reasonable and the real numbers. Cyclical
The meaning of the denseness is very close to what he literally means.
Theoretically:&#36;(X,d)&#36;It's measuring space.&#36;A,B\subset X&#36; The following propositions are equivalent</p>
<ul>
<li>&#36;A&#36;Yes.&#36;B&#36;Middle density</li>
<li>&#36;\forall x\in B,\exists\left{x_{n}\right}\subset A,\text{使得}\lim_{n\to\infty}d\left(x_{n},x\right)=0&#36;</li>
<li>&#36;B\subset\overline{A}&#36;</li>
<li>&#36;\text{fp}\delta&gt;0, \\text{possessed}B\subset\cup x\inA}O\left(x,\delta\right).</li>
</ul>
<p>Theorem (bulk transmission)
Set&#36;X&#36;It's measuring space.&#36;A,B,C\subset X&#36; If &#36;A&#36;Yes.&#36;B&#36;Middle density &#36;B&#36;Yes.&#36;C&#36;Middle density rule&#36;A&#36;Yes.&#36;C&#36;Middle density</p>
<h4>Delineable definitions</h4>
<p>Definitions (dependable)
Set&#36;X&#36;It's measuring space.&#36;A\subset X&#36; If&#36;A&#36;Existence<strong>A row of dense subsets</strong> Name&#36;A&#36;It's a partable collection if&#36;X&#36;It's a set of points, and it's called&#36;X&#36;It's a fractional measure space.</p>
<p>It's very easy to verify that the real set is a fractional space, and he has a dense subset like the rational set, and the rational set is a columnable sequence.</p>
<h4>Examples of subspaces</h4>
<ul>
<li>European Space&#36;R^{n}&#36;Score</li>
<li>Continuous Function Space&#36;C[a,b]&#36;Score</li>
<li>&#36;p&#36;Secondary sacramental function space&#36;L^{p}[a,b]&#36; Score</li>
<li>&#36;p&#36;Substantiate array space&#36;l^{p}&#36;Score</li>
<li>There's a boundary array space&#36;l^{\infty}&#36;It's indistinguishable.</li>
<li>Set&#36;X=[0,1]&#36; Dispersive measure space&#36;(X,d)&#36;It's indistinguishable.</li>
<li>&#36;X&#36;When columnable, dispersible measure space&#36;(X,d)&#36;Score</li>
</ul>
<h4>Inferences</h4>
<p>Theoretically:
&#36;&#36;设X是可分的度量空间Y是X的子空间 则Y是可分的子空间&#36;&#36;
Inferences:
&#36;&#36;00\&amp;\\text{}X\text{measured space, \subset X\text{is non-column space and exists}\delta&gt;\text,}forall x\text,}y\Y,\text{satisfaction}d(x,y)\sigma\text}X\text{not subspace}\end{aligned} I'm sorry.</p>
<h3>Metric Space Completeness</h3>
<p>Here we're promoting the Cauchy-Calcinator principle to measurement space. Medium</p>
<h4>Definitions and cross-references</h4>
<p>Definitions (basic columns)
&#36;&#36;\begin{aligned}\text{x n}\text{is a bar in the measurement space},\text{if any}\epsilon&gt;0,\text{existing}, n,\text{m, n}&gt;\text{, has}&amp;d(x_n,x_n)&lt;\epsilon,\text{n},\text{is } a basic column of x\text{) \end{aligned} &#36;</p>
<p>Theorem: The nature of the basic column in the measurement space&#36;(X,d)&#36;Medium</p>
<ul>
<li>The consolidation is the basic line.</li>
<li>The basic columns are made up of a collection of boundaries.</li>
<li>If the basic column contains a condensation column, he's condensed and condensed in the same way as the sub-column.
<strong>Theoretically, the holding up must be the basic column, which in fact does not necessarily form the measuring space, but only the European space.</strong></li>
</ul>
<h4>Full measure space</h4>
<p>Definitions
For a measure space&#36;X&#36; If any of her basic columns are abated, we call it.&#36;X&#36;It's a perfect measurement space.</p>
<ul>
<li>&#36;n&#36;Vior's full of space.</li>
<li>Continuous function space is complete (when the first definition of distance is used)</li>
<li>Continuous function space is incomplete (when the second definition of distance is used)</li>
</ul>
<h4>Some inferences.</h4>
<p>Theorem (Closer Ball Set)
&#36;&#36;00\&amp;(X, d)\text{is a full measurement space,}B n=overline{\matcal{O}(x n,\delta n)\text{is a closed ball, the latter is a subset of the former}&amp;\text{if the radius of the ball}\delta n\to0, n\toify,\text{so at the only point}\bit bigcap^n.\end{aligned} I'm sorry.
Theorem of the closed ball is another way to measure space.
As we presented in the mathematical analysis, the theory of the integrity of the actual is a lot of the same.</p>
<p>Theoretically.
Set&#36;(X,d)&#36;is a full measure space,&#36;M\subset X&#36;It's a complete collection and only a set of things.&#36;M&#36;It's closed.</p>
<p>Summary of the score and completeness of commonly used measurement space
<img src="/assets/images/mathematics-notes/functional-analysis-01.png" alt="Summary of common measure spatial spectrometry and completeness">
This is a map we need to remember;
Many of the previous proofs are actually proof of the conclusion.</p>
<h3>Measuring the tightness of space</h3>
<p>Intuitively, it means that each element is sufficiently dense in the set.
In mathematical analysis, the inter-segregation continuum has its most valuable value, and it also stems from tightness.
Actually... <strong>The tight mathematical language describes the existence of a line of bounds, which is the intimacy of the theory.</strong></p>
<p>Definitions:
Set&#36;X&#36;It's measuring space.&#36;A\subset X&#36; If&#36;A&#36;Any of the points is constricted.&#36;X&#36;The sub-column, then called&#36;A&#36;It's a tight set.&#36;A&#36;It's also closed, and it's called&#36;A&#36;It's a tight set.&#36;X&#36;It's a tight set of itself.&#36;X&#36;It's tight space.
<em>The whole space and the empty set are both opening and closing.</em>
<strong>The only thing that's contained is the holding point that's out of the pool.</strong></p>
<p>Nature</p>
<ul>
<li>Set&#36;X&#36;It's measuring space.&#36;A\subset X&#36; then&#36;A&#36;It's tight and it's just random.&#36;A&#36;There is a condensation line in the middle bar&#36;A&#36;Middle Point</li>
<li>Any limited set is a tight set</li>
<li>The sub-set of the tight set is tight.</li>
<li>Any number of tight encounters are tight.</li>
<li>The limited number of tights and tights.</li>
<li>&#36;A&#36;Yes.&#36;X&#36;♪ And the line is tight and the line is tight ♪&#36;\bar{A}&#36;It's a tight set.</li>
</ul>
<p>Inferences&#36;X&#36;It's tight space.&#36;A\subset X&#36;</p>
<ul>
<li>There's a boundary in the tight space.</li>
<li>Tight space is a perfect measure space.</li>
<li>&#36;A&#36;It's tight.&#36;A&#36;It's closed.</li>
</ul>
<p>Theorem: Set&#36;A&#36;Yes.&#36;n&#36;A subset of Vior's space.</p>
<ul>
<li>&#36;A&#36;It's a tight set and just a tight set.&#36;A&#36;There's a boundary.</li>
<li>&#36;A&#36;It's tight and it's just...&#36;A&#36;It's closed. Set</li>
</ul>
<p>Theoretically:
Continuous mapping will make the close mapping a close one
Definitions
Continuous mapping of real numbers is also called generic.
Theoretically:
Set&#36;X&#36;It's tight space. Close.&#36;A\subset X&#36; &#36;f&#36;It's a general letter on the X.&#36;f&#36;Yes.&#36;A&#36;Maximum and minimum value to be taken from above</p>
<h3>All-Boundary Collections of Measurement Space</h3>
<p>The concept is also used to paint the tightness of the picture;
(b) Strictness and all-margin equivalence in the full measure space;
In the normal measurement space, the column close must be full of boundaries.</p>
<p>Definitions (%1)&#36;\epsilon&#36;Net)
Set&#36;X&#36;It's measuring space. &#36;A,B\subset X&#36; For given&#36;\epsilon&#36; What if...&#36;B&#36;Any point in the middle&#36;x&#36; It must be there.&#36;A&#36;Midpoint &#36;x{&#39;}&#36;使得&#36;d(x,x^{&#39;})&lt;\epsilon&#36;则称A是B的一个 &#36;\epsilon&#36; 网 也就是&#36;B\subset \cup_{x\in A} O(x,\epsilon)&#36;
<strong>It's obvious.&#36;\epsilon&#36;The net exists, but the other way around is not possible.**
Because denseness requires random distance, but...&#36;\epsilon&#36;The net is a limited distance.
Definitions (all bounds)
Set&#36;X&#36;It's measuring space.&#36;A\subset X&#36; If for any given &#36; \epsilon&gt;0&#36; &#36;A&#36;总存在有限的&#36;\epsilon&#36;网 则称A是&#36;All-Boundary Sets in X&#36;
There's two at the core.</strong> Always exist and limited networks**
Introduction
&#36;A&#36;It's a full-fledged collection of measuring space and only when &#36;\forall \epsilon &gt; 0 ~\exists{x_{1},x_{2}...x_{n&#125;&#125; \subset A&#36;  使得 &#36;A\subset\cup \scene}O (xix) \scuse \scene \scene \scene \scene \scene \scene \scene \scene \scene \scene \scene \scene \scene \scene \scene \scene \screech \scene \scene \scene \scene \scene \scene \scene \scene \scmscene \scmscene \scmscmscm
The proof of the reasoning is just the use of the definition.</p>
<p>Theoretically.
Set&#36;X&#36;It's measuring space.&#36;A\subset X&#36; If A is&#36;X&#36;All Boundaries in the</p>
<ul>
<li>A is a boundary.</li>
<li>A is divided.</li>
</ul>
<p>Theoretically.
A is a full-fledged addition to the condition that any entry in A has a basic sub-bar</p>
<p>Hausdorf Theorem
Set&#36;X&#36;It's measuring space.&#36;A\subset X&#36; then</p>
<ul>
<li>A is a tight set, and A is a bounded one. Set</li>
<li>If X is full measure space, then A is the column tight and only when A is the full boundary.</li>
</ul>
<p>Summary
&#36;&#36;\begin{aligned}\\text{紧集}\Rightarrow\text{列紧集}\Rightarrow\text{全有界集}\Rightarrow\text{有界集}+\text{可分集}\\text{紧集}\Leftarrow_{\text{闭&#125;&#125;\text{列紧集}\Leftarrow_{\text{完备&#125;&#125;\text{全有界集}\end{aligned}&#36;&#36;</p>
<h3>Open-covered space in measuring space</h3>
<p>Definitions:
&#36; \\begin{aligned}\text{set}X\text{is the measurement space},&amp;\\Lambda\text{is an indicator set,}A\subset X\text{,}forall\lambda\in\Lambda, G \lambda\text{is the opening collection of x\text{, if A}subset\bigcup  {\lambda\inA}G lambda, \\text{&amp;\text \is A open-covered}
There is no requirement for the columnability of the indicator set.
Introduction:
Set&#36;X&#36;is the measurement space A is the tight fraction of X; &#36;&#123;G_\lambda|\lambda\in\Lambda}&#36;  It's an open-covered A, and there's &#36; epsilon.&gt;0&#36; 使得任意&#36;x\in A&#36; 存在&#36;G_{x}\in {G_{\lambda&#125;&#125;&#36; 满足&#36;O (x,\epsilon)\subset G x}
Theoretically:
Set&#36;X&#36;It's measuring space.&#36;A\subset X&#36; A is a close and limited coverage when A is a free-covered cover.
<strong>This is a very important theory, and it's the only core of this section.</strong>
Nature:
The continuous mapping in tight space is consistent continuous mapping.
Nature:
&#36;&#36;\quad\text{设}\left(X,d\right)\text{为度量空间},\text{则 }X\text{ 为紧空间的充要条件是:对 }X\text{ 中的任意闭集}\\text{族}F_{\lambda},\left(\lambda\in\Lambda\right)\text{,若其中任意有限个闭集 }F_{\lambda}\text{的交集都为非空集,则}\bigcap_{\lambda\in\Lambda}F_{\lambda}\text{也必为非空集}&#36;&#36;</p>
<h2>Linear enabling and built-in spaces</h2>
<h3>Definition and nature of linear enabling spaces</h3>
<h4>Definitions</h4>
<p>Set&#36;X&#36;It's digital.&#36;F&#36;If for each of them,&#36;x\in X&#36; One of the positives is the number of the number. &#36;||x||&#36;  And there is.</p>
<ul>
<li>Non-negative &#36;||x||\ge0&#36; And there's only one thing.&#36;x=0&#36; Yes.  &#36;||x||=0&#36;</li>
<li>Rectangular &#36;||ax||=a||x||&#36;</li>
<li>Triangular Instinct &#36;||x+y||\le||x||+||y||&#36;
Name&#36;||x||&#36; It's a model for x.&#36;B^{*}&#36;Space)
If we define &#36;&#36;d\left(x,y\right)=\left|x-y\right|&#36;&#36;
It's easy to verify that the distance must be a non-negative symmetry, triangulation, so it can form a measurement space called the meso-enabled space--
And now all the questions we've been looking at in the chapter on measuring space can be used.</li>
</ul>
<h4>Concealed by standard</h4>
<p>Set&#36;X&#36;For linear enabling space,&#36;&#123;x_n}&#36;Yes.&#36;X&#36; ,&#36;x\in X&#36;, if&#36;lim||x_{n}-x||=0&#36; Consistency in standard numbers&#36;x&#36;(Abbreviations)&#36;&#123;x_n}&#36;"Stand down"&#36;\lim x_n=x&#36;or&#36;x_n\to x,n\to\infty&#36;
It's clear that the same amount of distance is derived from the same number of scales.
Gives a few things.
(1) Continuity of the model: the model&#36;||x||&#36;From&#36;X&#36;Present.&#36;R&#36;Up on the continuous map.
(2) Boundaries: if&#36;(x_n)&#36;♪ And hold on to the&#36;x&#36;, and then&#36;||x_{n}||&#36;There's a boundary.
(3) Continuity of linear operations:&#36;x_n\to x,y_n\to y,n\to\infty&#36;, and&#36;x_n+y_n\to x+y,ax_n\to ax,n\to\infty&#36;, of which&#36;a&#36;is the constant.
From these natures, the prototype must be a continuous, generic letter.</p>
<h4>Banach Space</h4>
<p>Set&#36;X&#36;It's a linear enabling space.&#36;d\left(x,y\right)=\left|x-y\right|&#36; So we call it Banach Space.&#36;B&#36; Space
This theory is very useful for introducing a very important theory behind us.
We can study the linear enabling space in the back of the last chapter about the adequacy of some space.
<img src="/assets/images/mathematics-notes/functional-analysis-02.png" alt="Summary of space measurement and linear enabling spatial relationships">
Note: Not all measurement spaces can find a linear enabling space, which is a one-way equivalent.
For example, the dispersive measure space could not find the corresponding linear enabling space because it violated his three rules of justice.
&#36;&#36;\left|x\right|=d_{0}\left(x,\theta\right)=1,\left|2x\right|=d_{0}\left(2x,\theta\right)=1&#36;&#36;
Here are the conditions for measuring space to make linear enabling space.
&#36;&#36;d(x-y,\theta)=d(x,y),d(ax,\theta)=\big|\alpha\big|d\left(x,\theta\right).&#36;&#36;</p>
<h4>Level</h4>
<p>Set&#36;X&#36;Linear enabling space, line&#36;&#123;x_n}\subset X&#36;,Performance &#36;x_1+x_2+\cdots+x_n+\cdots=\sum x_n&#36; Yes&#36;X&#36;. If Partially and Pointbar&#36;S_n=x_1+x_2+\cdotp\cdotp+x_n&#36;Consistency in standard numbers&#36;s\in X&#36;, and then the number of grades&#36;\sum x_n&#36;♪ And hold on to the&#36;s&#36;- What?&#36;s&#36;And the sum of the steps, as it is,&#36;s=\sum_{n=1}^{\infty}x_n&#36;...if several levels&#36;\sum_{n=1}^{\infty}|x_n|&#36;Consistency, rank&#36;\sum_{n=1}^{\infty}x_n&#36;Absolutely.
Theorem: Set&#36;X&#36;It's linear enabling space, and...&#36;X&#36; Yes, Banach. Space is only.&#36;X&#36; The absolute insulation of any level of middle is the total insulation of the grade.
Theorem: Set X to Banach space,&#36;\left{x_{n}\right},\left{y_{n}\right}\subset X&#36; , when n&gt;When \mathbb{&#36;}
&#36;||x_{n}\parallel=c\parallel y_{n}\parallel&#36;, of which&#36;c&#36;It's a constant, then if&#36;\sum^{\infty}y_{n}&#36;Absolutely.&#36;\sum_{n=1}^{\infty}x_{n}&#36;And definitely.</p>
<h3>Sub-units and commercial spaces of linear enabling spaces</h3>
<h4>Crumb</h4>
<p>Set&#36;X&#36;Linear space on a digital F, a subset of C for X, if&#36;\forall x,y\in C&#36;Yes.
&#36;&#36;\left{\alpha x+\left(1-\alpha\right)y|0\leqslant\alpha\leqslant1\right}\subset C&#36;&#36;
C for X
Proof of the closed units in linear enabling spaces Ball.&#36;B(0,1)={x|~||x||\le1}&#36;</p>
<h4>Subspace</h4>
<p>Subspace Setup&#36;X,|\cdot|)&#36;For linear enabling spaces, V is a linear subspace of X and elements in V &#36;x&#36; The standard is based on its&#36;X&#36; vantage point in&#36;|x|&#36;, or&#36;(V,|\cdot|)&#36;Or...&#36;V&#36;It's linear enabling space.&#36;X&#36; Subspace
Nature:
Set&#36;X&#36;It's linear enabling space, and it's closed to the subspace.&#36;\bar{V}&#36; It's linear.
The lockdown is to ensure linearity.
Set&#36;X&#36;It's Banach. &#36;M&#36;Yes.&#36;X&#36;. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .&#36;M&#36;Banach's subspace is closed.
We just need to know his completeness.</p>
<p><em>We know that the linear enabling space R is closed, so is it true that, in general, the linear enabling space is closed? If the linear enabling space&#36;X&#36;The dimensions of this space as linear space are:&#36;n&#36;, or&#36;X&#36;For n-dimensional linear space, if&#36;X&#36; It's a limited-dimensional linear enabling space. &#36;X&#36; Any subspace is closed. However, for infinity linear enabling spaces, subspace is not necessarily closed.
We'll explain this in the next section.</em></p>
<h4>Linear</h4>
<p>Set&#36;E&#36;It's digital F-line space.&#36;X&#36;The whole linear combination of all the limited series in E is the Linear Span, and it's a simple, non-empty subset.
&#36;&#36;\left.\mathrm{span}E=left&lt;a_{1}x_{1}+a_{3}x_{2}+\cdots+a_{n}x_{n}\right|x_{1}+x_{2},\cdots,x_{n}\in E,a_{1},a_{2},\cdots,a_{n}\in\mathbb{R}\right&gt;I'm sorry.
The equivalent is, spanE is the intersection of all linear subspaces that contain E. In fact, spanE is the smallest linear subspace that contains E.</p>
<p>Set&#36;X&#36;Linear F-based space, E is a non-empty subset of X, and all closed subspaces containing E are interlocked into E-closed graphs, as follows:&#36;\overline{\mathrm{span&#125;&#125;E&#36;</p>
<p>E is a non-empty collection of X, if the linear enabling space on X-Streen-F is available
&#36;\left(1\right)\overline{span}E\in X\text{的闭线性子空间}.&#36;
&#36;\left(2\right)\overline{span}E=\overline{spanE},&#36;</p>
<h4>Business space</h4>
<p>A linear enabling space on X-Ten-Cities F, V is a closed space for X if &#36;x-y\in V&#36;, or &#36;x&#36; and y belong to the same class of prices, as&#36;[x]&#36;Or...&#36;\widetilde{x}&#36;, the entire group of these equals is recorded as &#36;X/V=\langle[x]|[x]=x+V\rangle&#36; X/V is defined as X plus, multiply and model for commercial space of V, commercial space X/V
&#36;\forall\left[x\right],\left[y\right]\in X/V,\alpha\in\mathbb{F},\text{有}&#36;
&#36;\left[x\right]+\left[y\right]=\left(x+V\right)+\left(y+V\right)=x+y+V=\left[x+y\right]&#36;
&#36;a\left[x\right]=a\left(x+V\right)=ax+V=\left[ax\right]&#36;
&#36;\left|\left[x\right]\right|=\left|x+V\right|=\inf\left{d\left(x,v\right)|v\in V\right}.&#36;
Understand business space. Every element of business space.&#36;[x]&#36;It's all a piece of the original linear enabling space. Set
&#36;&#36;\left.\left[x\right]=x+V=\left&lt;x+v\right|v\in V\right&gt;\subset X.&#36;&#36;
也就是&#36;x&#36;是&#36;[x]&#36;中的一个代表 我们可以在商空间定义范数
&#36;&#36;\begin{aligned}|\begin{bmatrix}x\end{bmatrix}|&amp;=\inf\langle d(x,v)\mid v{\in}V\rangle=\inf{d(x,x-y+v)\mid x-y+v{\in}V\rangle\&amp;=inf{d(y,v)\mid v(in)}V\rangle=[y]|.\end{aligned}
It's a linear enabling space.</p>
<p>Some nature
Set X as linear enabling space, V is the closed space of X
&#36;\left(1\right)设Q；X\to X/V&#36;The blog is a natural map. &#36;Q\left(x\right)=\left[x\right]=x+V&#36;,&#36;\forall x\in X&#36;,&#36;\left|Q\left(x\right)\right|\leqslant\left|x\right|,Q&#36;To a continuous map
(2) If X is Banach space, commercial X/V is also Banach space,
♪ W is&#36;X/V&#36;♪ The beginning of the world ♪&#36;Q^{-1}\left(W\right)=\left{x\left|Q\left(x\right)=\left[x\right]\in W\right}\in X\right.&#36;The opening of the festival,
(4) If&#36;U&#36;Yes.&#36;X&#36;♪ The opening, then ♪&#36;Q\left(U\right)&#36;It's commercial space.&#36;X/V&#36;The beginning of the collection</p>
<h3>Synonyms and protometric equivalents of linear enabling spaces</h3>
<p>The symmetry in the online algebra is a reflection of the presence of a multiplication and multiplication in two limited dimensions of linear space, which in the same sense is of exactly the same nature.</p>
<h4>Linear equidistance</h4>
<p>Set (X, \cdot|)<em>x),(Y,|\cdot|<em>Y)&#36;是同一数域F上的两个线性赋范空间，如果存在一一映射&#36;T:X\toY, &#36;Fulfilled:
(1) Linear: &#36;\forall x</em>{1},x</em>{2}\in X,\alpha,\beta\in\mathbb{F},T(\alpha x_{1}+\beta x_{2})=\alpha T(x_{1})+\beta T(x_{2});&#36; &#36;\left(right) \forall x, \parallel Tx\parallel =y}=parallel x parallel x}&#36;,
and then it's called&#36;Y&#36;Linear equidistance and map&#36;T&#36;It's a linear equal range map.
We have the same dimensions as they have in the online algebra, and in fact, the online enabling spaces can make similar conclusions.
&#36;\text{Theoret 2.3.1 setting X is the n-dimensional grant space on R in real-digit domains, X and R&quot;Linear equidistance.
Based on this easy inference: the subspace of limited-dimensional linear enabling spaces must be closed (because)&#36;R^n&#36;(SINGING)</p>
<h4>The equivalent of the equivalent</h4>
<p>With the same components of linear enabling space, we must be interested in the relationship to the same timescale.
Definitions
Set&#36;|\cdot|_1和|\cdot|_2&#36;It's defined as a linear space.&#36;X&#36;Two of the above-scaled, pointer&#36;\left{x_n\right}\subset X&#36;, if by&#36;x_n\parallel_1\to0&#36; Available&#36;|x_n|_2\to0&#36;, or&#36;|\cdot|_1比|\cdot|_2&#36;# I'm not sure I'm gonna be able to do this #
If&#36;|\cdot|_1比|\cdot|_2&#36;Strong, and&#36;|\cdot|_2比|\cdot|_1&#36;Strong, then call it the standard \cdot 1 and \cdot|<em>&#36;2.00 equal.
Theoretically.
Linear Enabling Space&#36;X&#36;The two-faulted vans.<em>1&#36;和&#36;|\cdot|<em>2&#36; 等价当且仅当存在正实数a和&#36;b&#36;,使得&#36;\forall X, yes
&#36;a\left|right|</em>{2}\leqslant\left|x\right|</em>{1}\leqslant b\left|x\right|</em>I'm sorry.
Theoretically.
A random standard equivalent in limited-dimensional linear enabling spaces
This theory tells us that we can think of different ranges or simplest parameters.</p>
<h3>Dimensions and constraints of linear enabling space</h3>
<h4>Dimensions of linear enabling space</h4>
<p>If the linear enabling space&#36;X&#36;The dimensions of this space as linear space are:&#36;n&#36;, or&#36;X&#36;Yes&#36;n&#36;A perimeter-based enabling space.
If his dimensions are infinite, then we call them infinite.</p>
<h4>The relationship between the tightness and dimensions of linear enabling space</h4>
<p>We've introduced it in the measurement space. &#36;R^{n}&#36;Subspace is priced at its own level.
Theoretically:
Set&#36;X&#36;It's linear enabling space.&#36;X&#36;%1 %2 %2 %2 %2 %2 %2 %2 %2 %2 %2 %2 %2 %2 %2 %2 %2 %2 %2 %2 %2 %2 %2 %2 %2 %2 %2 %2 %2 %2 %2 %2 %2 %2 %2 %2 %2 %2 %1 %2 %2 %1 %1 %1 %2 %1 %2 %1 %1 %1 %1 %1 %1 %1 %1 %1 %1 %1 %&#36;X&#36;Each of these clusters is a tight one.
Equivalent proposition:
Set&#36;X&#36;It's linear enabling space.&#36;X&#36;It's infinite. At least.&#36;X&#36;One of the boundaries is not a tight one.</p>
<p>Riesz Introduction:&#36;A&#36;It's linear enabling space.&#36;X&#36; Closed space, and &#36;A\neq X,0&lt;\alpha&lt;1&#36;,则存在&#36;x_a\in X&#36;,使得&#36;\parallel=1, and d\left(x{a}, A\right)&gt;\alpha.&#36;</p>
<h3>Definition of built-in space</h3>
<p>Or is it the online algebra that we're going to study the interior space after we've finished the linear space, and we're going to try to add geometry to the original linear space, and now we're going to re-create these studies in online enabling spaces?</p>
<h4>Definitions</h4>
<p>Set&#36;X&#36;It's digital.&#36;F&#36;On the linear space, if there is a map (.. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .&#36;X×X→F&#36;♪ And make ♪&#36;\forall x,y,z\in\mathbb{X},\alpha~\beta\in F&#36; Satisfied</p>
<ul>
<li>Non-negative:&#36;\left(x,x\right)\geqslant0,\left(x,x\right)=0\text{ 当且仅当 }x=0&#36;</li>
<li>Symmetry in cosmmetry.&#36;\left(x,y\right)=\overline{\left(y,x\right)}&#36;</li>
<li>First metalinearity.&#36;\left(\alpha x+\beta x,y\right)=\alpha\left(x,y\right)+\beta\left(z,y\right)&#36;
We call it&#36;F=R&#36;It's time to build up the interior. &#36;F=C&#36;Time is for the interior.
The inner space of limited dimensions is called the O'Shock space, the inner space of limited dimensions is called the condensed space.
It's easy to prove for continuous function space&#36;C[a,b]&#36; Define Internal As
&#36;&#36;\left(f,g\right)=\int_{a}^{a}f\left(x\right)g\left(x\right)dx&#36;&#36;
It's easy to verify that he is a real, solid space.</li>
</ul>
<h4>Export linear enabling spaces</h4>
<p>Define the standard number for inner space as&#36;\left|x\right|=\left(x,x\right)^{\frac{1}{2&#125;&#125;&#36;
Give another Cauchy equation as the point of argument.
&#36;&#36;\left.\text{设}X\text{为内积空间,证明}\forall x,y\in X,\text{有}|\left(x,y\right)\right|\leqslant\left|x\right|\cdot\left|y\right|&#36;&#36;
<strong>He built a relationship between the internality and the standard.</strong> This formula is very important.
The Cauchy iniquities can be proved.
The range from which the internal volume is exported is sufficient for positive characterization and alignment.
&#36;&#36;00\&amp;|x+y|^2=|(x+y,x+y)|=|(x,x+y)+(y,x+y)|\&amp;\leqslant|(x,x+y)|+|(y,x+y)|\leqslant|x|\cdot|x+y|+|y|\cdot|x+y\&amp;\leqslant(|x|+|y|)|x+y|,\&amp;\text \xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx&#36;&#36;
核心点就是使用第一变元线性性进行拆分 结合新的Cauchy不等式进行计算，这在内积空间后面的研究中会非常的常用
现在我们可以知道
&#36;&#36;\\text{innerspace}\longrightarrow \text{linear enabling space}\longrightarrow\text{measure space.} &#36;
Three spaces can be exported in sequence.</p>
<h4>Hilbert Space</h4>
<p>Set&#36;X&#36;It's digital.&#36;F&#36;The interior space on the inside is in range from the inside.&#36;\left|x\right|=\left(x,x\right)^{\frac{1}{2&#125;&#125;&#36;   If you follow this range,&#36;X&#36;It's Banach, and we call it Inner Space.&#36;X&#36;It's Hilbert Space, called "H."
<strong>In fact, both H and B are studying whether the distance from which they are produced is sufficient, and the research is basically consistent.</strong>
It's easy to give the theorem down.
&#36;H&#36;So is the space.&#36;H&#36;Space and only when he's closed.
Or do we follow the idea of studying B space?
Real built-in space&#36;R^n&#36; Define Internal As&#36;(x,y)=x_{1}y_{1}+x_{2}y_{2}+\cdots+x_{4}y_{n}&#36;
Recovery space&#36;C^n&#36; Define Internal As&#36;\left(x,y\right)=x_{1}\overline{y_{1&#125;&#125;+x_{2}\overline{y_{2&#125;&#125;+\cdots+x_{s}\overline{y_{n&#125;&#125;&#36;
Recovery space&#36;l^2&#36; Define Internal As&#36;\left(x,y\right)=x_{1}\overline{y_{1&#125;&#125;+x_{2}\overline{y_{2&#125;&#125;+\cdots+x_{s}\overline{y_{n&#125;&#125;+...&#36;
Recovery space&#36;L^2[a,b]&#36;  Define Internal As &#36;(x,y)=(L)\int_{[0,\delta]}x(t)\overline{y(t)}dt.&#36;
We can only see how we can prove it by actually looking back with the full range of known space. The original Banach space is not able to induce Hilbert to continue.</p>
<h3>Inner space and linear enabling spaces</h3>
<p>Inner space can export linear entanglements, and we're going to do some inverses at the end, but is that something that can be found? Actually, it's not necessarily possible.</p>
<h4>Polar Synopsis</h4>
<p>For the actual amount of space and the typical amount it produces
&#36;&#36;\left(x,y\right)=\frac{1}{4}\left(\parallel x+y\parallel^{2}-\parallel x-y\parallel^{2}\right).&#36;&#36;
For the compound space and the frame it produces
&#36;&#36;(x,y)=\frac{1}{4}(\parallel x+y\parallel^2-\parallel x-y\parallel^2+\mathrm{i}\parallel x+\mathrm{i}y\parallel^2-\mathrm{i}\parallel x-\mathrm{i}y\parallel^2).&#36;&#36;</p>
<h4>Parallel quadrilateral formula</h4>
<p>Linear Enabling Space&#36;X&#36;It's a place of internal space and only&#36;\forall x,y\in&#36; X, a model to meet parallel quadrilateral formulas
&#36;&#36;\left|x+y\right|^{2}+\left|x-y\right|^{2}=2\left|x\right|^{2}+2\left|y\right|^{2}.&#36;&#36;</p>
<h3>The decomposition of the built-in space</h3>
<p>In three-dimensional space, all vectors can be broken down into two vertical vectors and we're just creating the base system simplification by active decomposition.</p>
<h4>Done</h4>
<p>If&#36;(x,y)=0&#36; Call two vectors positive.
If we're gonna have two,&#36;A,B&#36;The vectors are two-to-two, and the two-to-one are set. Hand it over.
If for the interior space&#36;X&#36;  ，&#36;E&#36;and&#36;X&#36;Corrected&#36;E&#36;Yes.&#36;X&#36;A cross-section</p>
<p>We can naturally promote the principle of equity by nature.
&#36;&#36;\text{设}X\text{是内积空间},x,y\in X,\text{ 若 }x\perp y,\text{ 则 }\parallel x+y\parallel^2=\parallel x\parallel^2+\parallel y\parallel^2.&#36;&#36;
His inverse theory has
In real-input space, the charades theorem knows two vectors are in the right direction.
There's a theory to promote this.
&#36;&#36;\parallel a_{1}x_{1}+a_{2}x_{2}+\cdots+a_{n}x_{n}\parallel^{2}=\left|a_{1}\right|^{2}\parallel x_{1}\parallel^{2}+\left|a_{2}\right|^{2}\parallel x_{2}\parallel^{2}+\cdots+\left|a_{n}\right|^{2}\parallel x_{n}\parallel^{2} &#36;&#36;
<strong>And this is a very important theory, a polarized constant equation, a parallel quadrilateral formula, a typology of the equation, the only abstracted paradigm we know of.</strong></p>
<h4>- I'm on it.</h4>
<p>The ongoing replenishment is an extension of the concept of the replenishment.
Set&#36;X&#36;It's built-in space.&#36;M\subset X&#36;Remember&#36;M^{\perp}=\left{x\mid x\perp M,x\in X\right}&#36;, or&#36;M^{\perp}&#36;It's obvious there's a real deal for the M-set. &#36;X^{\perp}=\left{0\right},\left{0\right}^{\perp}=X,以及 M^{\perp}\bigcap M=\left{0\right}.&#36;
The definition being supplemented can be of the following nature:</p>
<ul>
<li>&#36;\text{若}M\bot N,则M\subset N^{\perp}&#36;</li>
<li>&#36;若M\subset N,则M^{\perp}\supset N^{\perp}&#36;</li>
<li>&#36;M\subset\left(M^{\perp}\right)^{\perp}&#36;
Theoretically:
 &#36;&#36;\text{设 }X\text{ 是内积空间},{M\subset X,\text{则}M^\perp\text{是 }X\text{ 的闭线性子空间&#125;&#125; .&#36;&#36;
Because we know that the requirement for the complete measurement of the subspace is the closing of the subspace.
So this theory can be used to study the perfect subspace for the Banach Hilbert properties.
The positive replacement of Hilbert's subspace must be Hilbert's space.</li>
</ul>
<h4>It's decomposing.</h4>
<p>Set&#36;M&#36;It's built-in space.&#36;X&#36; The subspace,&#36;x\in X&#36;, if exists &#36;x_0\in M,z\in M^\perp&#36;♪ And make ♪ &#36;x=x_0+z&#36;, or&#36;x_{0}&#36;Yes&#36;x&#36;Yes.&#36;M&#36;♪ Up the positive projection or the positive decomposition ♪
<strong>Actually, that's the problem we started with. It's split into two vectors in the right direction.</strong>
Introduction
X is built up in the interior, M is a linear subspace of X,&#36;x\in\mathbb{X}&#36;, if existing &#36;y\in M&#36;♪ And make ♪
&#36;||x-y||=d(x,M),那么 x-y\perp M.&#36;
<em>The reasoning is only for proof of the projection theorem below.</em>
Project Theorem
Set&#36;M&#36;It's Hilbert Space.&#36;H&#36;Up the closed subspace, then&#36;H&#36; Elements in &#36;x&#36; There's only positive projection in M, which is &#36;\forall x\in H&#36;  Yes.&#36;x=x_{0}+z&#36; of which&#36;x_{0}\in M,z\in M^\perp&#36;
The projection theory tells us that Hilbert space and closed subspace can achieve the disintegration we want and the only one that can be broken.</p>
<h4>Straightness of subspace</h4>
<p>Set M and N to be two subspaces of the Linear Space U, called &#36;M+N={m+n\mid m\in M,n\in\mathbb{N&#125;&#125;&#36; Yes&#36;M&#36; And with N and Sum. If &#36;M\bigcap N=\left{\theta\right}&#36;, or&#36;&#123;m+n\mid m\in M,n\in N}&#36; Yes&#36;M&#36; The sum of the sum of N, which is then recorded as
&#36;&#36;M\oplus N=\left{m+n|m\in M,n\in N\right},M\bigcap N=\left{\theta\right}.&#36;&#36;
According to Projectiving Theorem if M is Hilbert Space &#36;H&#36; Up the closed subspace, then &#36;&#36;H=M\oplus M&#36;&#36;
Give a theory.
&#36;&#36;\text{设 H 是 Hilbert空间, }M\subset H,\text{那么}M\text{ 是闭子空间当且仅当 }M=(M^\perp)^\perp &#36;&#36;
&#36;&#36;\text{设 }H\text{是 Hilbert空间},MH,\text{那么}M\text{是 }H\text{ 的稠密子集当且仅当 }M^\perp=\langle\theta\rangle.&#36;&#36;</p>
<h3>Positive intersection of built-in space</h3>
<p>The concept of "transaction" is just to get the foundation.</p>
<h4>Definition of the standard cross-cutting basis</h4>
<p>The X is built up as an interior space.&#36;E=\left{e_{\lambda}|\lambda\in\Lambda\right}&#36; Yes. &#36;X&#36; , where tow is the indicator set.&#36;\forall e_{i},e_{i}\in E&#36;Satisfied
&#36;&#36;\left.\left(e i},e j}\right)=\left{begin{matrix}1,&amp;i=j,\0,&amp;I'm not a real guy.
E is called the standard log in X or the standard login Yes
<strong>Note that the standard is based on two main points, standards and cross-section, but never complete.</strong>
Nature:</p>
<ul>
<li>Any standard logarithmic distance between the logimetric space is&#36;\sqrt{2}&#36;</li>
<li>Set&#36;H&#36;It's a divided Hilbert space, and his random standard log is a collection.</li>
</ul>
<h4>Export of standard positive transfer</h4>
<p>We have described in the online algebra that any linear independent system can be transformed into a standard log; in turn, the standard log is a linear independent. Yes
<strong>Theorem</strong>: Set&#36;X&#36;It's an internal space. &#36;E&#36;It's his standard, right? &#36;\left{e_{n_{1&#125;&#125;,e_{n_{2&#125;&#125;,\cdots,e_{n_{k&#125;&#125;\right|\subset E&#36; Remember
&#36;&#36;M=span\left{e_{n_{1&#125;&#125;,e_{n_{3&#125;&#125;,\cdots,e_{n_{k&#125;&#125;\right},&#36;&#36;</p>
<p>&#36;\forall x\in X,x_{k}=\sum\left(x,e_{n_{k&#125;&#125;\right)e_{n_{k&#125;&#125;&#36;Yes.&#36;x&#36;Yes.&#36;M&#36;The top positive projection, which is...&#36;x_{k}\in M~x=x_k+z&#36;  &#36;\left(x-x_{k}\right)\perp M.&#36;
This theorem tells us how to calculate the coefficient of vectors on this base.</p>
<h4>Schmidt is conversing.</h4>
<p>Theoretically.
If&#36;\left{x_n\right}&#36;For any group of linear independent systems in the built-in space X, the following is the list of the main lines of action that can be used:&#36;&#123;x_n}&#36;The Gram-Schmidt method is used as the standard logarithmic&#36;&#123;e_{n&#125;&#125;&#36;and for any natural number&#36;n&#36;, exists&#36;\alpha_i^{(n)},\beta_k^{(n)}\in\mathbb{F}&#36;♪ And make ♪
&#36;&#36;
x_{n}=\sum_{k=1}^{n}\alpha_{k}^{\left(n\right)}e_{k},e_{n}=\sum_{k=1}^{n}\beta_{k}^{\left(n\right)}x_{k},
&#36;&#36;
Meanwhile... &#36;span{e_{1},e_{2},\cdots,e_{n&#125;&#125;=span\left{x_{1},x_{2},\cdots,x_{n}\right}.&#36;</p>
<p>The intersectional theory is not the most important; actually, the important thing is how we are doing it, and the following is both proof of this and how we are doing it.
You're the one who's gonna get you.&#36;e_1=\frac{x_{1&#125;&#125;{||x_{1}||}&#36; There is.&#36;M_1=span{e_1}&#36; It's the first step of the hierarchy.
We know.&#36;x_2&#36;It's got to be a target.&#36;M_1&#36;- I'm in a real dissectation.&#36;x_2=(x_2,e_1)e_1+v_2&#36;
I got it right now.&#36;v_2&#36;There must be a positive link to the current iterative standard, so it can be given.
&#36;e_2=\frac{v_{2&#125;&#125;{||v_{2}||}&#36;  &#36;M_2=span{e_1,e_2}&#36;
Repeat this process and continue to decompose.&#36;v_{3}....v_n&#36; Then we can standardize, and we can get a set.
&#36;&#36;e_1,e_{2}.....e_n&#36;&#36; That's the standard price we get.
If you're going to make him a model, it's going to be the following.
&#36;&#36;begin{aligned}e s&amp;=\frac{x_{1&#125;&#125;{\parallel x_{1}\parallel},e_{2}=\frac{x_{2}-(x_{2},e_{1})e_{1&#125;&#125;{\parallel x_{2}-(x_{2},e_{1})e_{1}\parallel},e_{0}=\frac{x_{3}-(x_{3},e_{1})e_{1}-(x_{3},e_{2})e_{2&#125;&#125;{\parallel x_{3}-(x_{3},e_{1})e_{1}-(x_{3},e_{2})e_{2}\parallel},\cdots,\e_{n}&amp;=\frac{x_{n}-(x_{n},e_{1})e_{1}-(x_{n},e_{2})e_{1}-\cdots-(x_{n},e_{n-1})e_{n-1&#125;&#125;{\parallel x_{n}-(x_{n},e_{1})e_{1}-(x_{n},e_{2})e_{2}-\cdots-(x_{n},e_{n-1})e_{n-1}\parallel},\cdots.\end{aligned}&#36;&#36;</p>
<h3>Fulley leaf grade and compulsiveness</h3>
<p>In the mathematical analysis, we've been able to introduce the Fully leaf class, which is used extensively in various applications, and now we're promoting the Fully leaf grade tools into our inner space, and, more specifically, Hilbert space.</p>
<h4>Definition of the number of Fully leaves</h4>
<p>Set&#36;e_n&#36;It's built-in space.&#36;X&#36;The standard is the top of the line.&#36;x\in X&#36;, and then the number of grades
&#36;&#36;\sum\left(x,e_{k}\right)e_{k}=\sum c_{k}e_{k}&#36;&#36;</p>
<p>Yes&#36;x&#36;About&#36;e_n&#36;The number of Fullie leaves.&#36;c_{i}=\left(x,e_{i}\right)&#36;Yes&#36;x&#36;About&#36;e_i&#36;Fuliber coefficient</p>
<h4>Best proximity theorem and Bézier's.</h4>
<p>Set as&#36;e_n&#36;Inner space&#36;X&#36;Standard Logic&#36;x\in X,c_k=\left(x,e_k\right),k=1,2,...,则对任何数组&#36; &#36;\left{\alpha_{1},\alpha_{2},\cdots,\alpha_{n}\right}\subset\mathbb{F}&#36;Yes.
&#36;&#36;\left|x-\sum_{k=1}^{n}c_{k}e_k\right|\leq\left|x-\sum_{k=1}^{n}\alpha_{k}e_{k}\right|&#36;&#36;
The best proximity theory tells us the characteristics of the error that goes by the Frei leaf class.</p>
<p>&#36;&#36;\text{设}e_n\text{为内积空间 }X\text{ 的标准正交基,则 }\forall x\in X,\text{有}\sum_{k=1}|(x,e_k)|^2\leqslant|x|^2.&#36;&#36;
The geometry of the Bézier-Illegitimate is the squares of the projection length and the squares of the original length.
He studied the equation of the Frei leaf coefficient.</p>
<h4>Repression as a requirement</h4>
<p>Set&#36;&#123;e_n}&#36;It's built-in space.&#36;X&#36; Standard Logic &#36;x\in{X}&#36;, and&#36;x&#36; About&#36;&#123;e_n}&#36;Fulley Levels&#36;\sum\left(x,e_k\right)e_k&#36;♪ And hold on to the&#36;x&#36; . The condition of the condition is
&#36;&#36;&#123;\parallel x\parallel^{2}=\sum_{k=1}^{\infty}\left|c_{k}\right|^{2},}&#36;&#36;
And this is what the Bézier formula is like.</p>
<p>In fact, this is influenced by one factor, which is that our standard logarithmic is sufficient in number, so just take the same spatial dimension as the standard logarithmic is the same, and the Paseva formula is set.</p>
<h4>Full standard is active.</h4>
<p><strong>Definitions</strong> : Set&#36;E={e_{\lambda}|\lambda\in\Lambda}&#36; It's a full standard of the H of Hilbert, and there is.&#36;\forall x\perp e_{\lambda}&#36;  then&#36;x=0&#36; Explain when the real deal is all over.
Introduction: Set-up&#36;E={e_{\lambda}|\lambda\in\Lambda}&#36; It's a full standard of H Hilbert. &#36;M=spanE&#36; There is. &#36;H=\overline{M}&#36;
Theorem: Set&#36;H&#36;It's a Hilbert space.&#36;c_k=(x,e_k)&#36; The following is the same deal.</p>
<ul>
<li>&#36;&#123;e_k}&#36;Yes.&#36;H&#36;The full standard is on the line.</li>
<li>&#36;\forall x\in H&#36; &#36;x&#36;About&#36;&#123;e_n}&#36; The Fulley Leafs are down.</li>
<li>&#36;\forall x\in H&#36; &#36;||x||^{2}=\sum\limits |c_k|^2&#36;
Nature: Establishment&#36;E={e_{\lambda}|\lambda\in\Lambda}&#36; It's a standard log H of Hilbert Space, and he's a standard log and only.&#36;E^{\perp}=0&#36;
Theorem: any non-zero-zero built-in space has a full standard logarithmic base</li>
</ul>
<h3>Symkinesis of Hilbert Space</h3>
<p>We have linear equidistance equivalents in online enabling spaces, which guarantee that linearity and the paradigm remain unchanged.
Set&#36;X_1,X_2&#36; It's the same digital domain.&#36;F&#36; The interior space on the inside, if there's a single map,&#36;\phi&#36; Promise.
&#36;&#36;00\
&amp;\varphi\left(\alpha x+\beta y\right)=\alpha\varphi\left(x\right)+\beta\varphi\left(y\right), \
&amp;\\left(x,y\right)=x,y\right,
I'm sorry, I'm sorry.
And then, "Could"&#36;X_1,X_2&#36; Linear equidistance
Theorem: Set&#36;H&#36;Yes.&#36;n&#36;Wear Hilbert Space&#36;H&#36;and build-up of space&#36;C^n&#36; Symbiotic
Theorem: Infinite-dimensional Hilbert Space&#36;H&#36;It's a good deal.
Theoretically: if the infinite dimension of Hilbert space&#36;H&#36;He's got a point.&#36;l^2&#36;Symbiotic
For the first theory: just make it happen.
For the third theorem: we know he has a perfect standard logarithmic base, and we'd like to verify whether the map is the linearity and internality we need.</p>
<h2>Linear</h2>
<p>The concept, which we barely used to have, actually applied a lot;
Any linear-to-line mapping of the enabling space is called the algorithm.</p>
<p>So the calculus we're learning before us, the points, are all kind of algorithms, and in this chapter we want to study abstract algorithms, but in order to make it simple enough, we're just going to study linear algorithms.</p>
<h3>Definition and nature of linear algorithms</h3>
<h4>Definition of linear algorithms</h4>
<p><strong>Definitions (calculator)</strong>:
Sets X and Y are two linear enabling spaces, if T is a map of a subset of X D to Y &#36;T&#36; Called D as the definition city for the count in the subset D to Y, and recorded as D & T; and the subset for the name Y&#36;R(T)={y|y=T(x),x\in D}&#36;For the counter.&#36;T&#36;. Yes.&#36;x\in D&#36;, usually remember&#36;x&#36; Like&#36;T(x)&#36;Yes&#36;Tx&#36;
Especially if&#36;X=Y=R&#36;. If&#36;Y&#36;It's a digital field, and it's called a general letter.&#36;Y&#36;"Door has decided whether they are letters or letters."
Definition (continuous number):
For the number D to Y algorithms, X and Y are two linear enabling spaces.&#36;x_{0}\in D&#36; &#36;\forall \varepsilon~ ~\delta &gt;&#36; 0 million
For random&#36;x\in D&#36;when &#36;xx 0|&lt;\delta&#36;时，有&#36;|Tx-Tx_0|\leq\varepsilon&#36;,则称算子&#36;T&#36;在点&#36;x&#36; continuous. If every point of the count T is continuous in D, then T is the continuous count on D.
&#36;f\left(x\right)&#36;Yes.&#36;x_{0}&#36;Point continuous equal value&#36;\forall\left{x_{n}\right}\subset D&#36;If&#36;x_{n}\to x_{0}&#36;, and there is&#36;f\left(x_{n}\right)\to f\left(x_{0}\right)&#36;
Definition (linear algorithms):
X and Y are two linear enabling spaces, D.&#36;\subset X,T&#36;for the count in D to Y if&#36;\forall x,y\in D&#36;,
&#36;&#36;T\left(\alpha x+\beta y\right)=\alpha T\left(x\right)+\beta T\left(y\right)&#36;&#36;
Definition (linear spectrometer):
X and Y are two linear enabling spaces, D.&#36;\subset X,T:D\to Y&#36;It's a linear algorithm, if M exists.&gt;0,
&#36;\forall x\in D,有\parallel Tx\parallel\leqslant M\parallel x\parallel&#36;The T is a linear algorithm on D, which is a very important tool for the development of the country.
Note that there is no consistency between the notion and function of the algorithm, for example, in the calculus.&#36;f(x)=x&#36; Obviously in real-number domains&#36;R&#36;It's unbounded, but there is.
&#36;&#36;||f\left(x\right)||=||x||\leqslant M\left|x\right|,M=1&#36;&#36;
It's a non-conscriptive function that may be a general communication.</p>
<p>Examples of frequent algorithms</p>
<ul>
<li>Constant Alcalculator&#36;I&#36;</li>
<li>Zero.&#36;0&#36;</li>
<li>Micro-calculator&#36;T&#36;And the score count.&#36;T&#36;</li>
<li>Matrix Transferr &#36;T&#36;</li>
</ul>
<h4>Nature of linear algorithms</h4>
<p>Theorem: Linear algorithms in&#36;D&#36;So, what's the point of the continuous and just one point?
Theorem: Linear algorithms are linear and only when it's a&#36;D&#36;There are boundaries that are mapped and bounds that are assembled.
Theorem: Linear algorithms continue and are linear algorithms.
Theoretically:&#36;X&#36;When it's limited-dimensional linear, linear, linear, it must be linear.
<strong>This means that limited-dimensional linear enabling domains, linear algorithms, linear horizons, linear mono-point continuumrs, linear continuum equations.</strong></p>
<h3>Zero space for linear algorithms</h3>
<h4>Definition of zero space</h4>
<p>Set&#36;X&#36;and&#36;Y&#36;It's two linear enabling spaces called a rally.&#36;\ker(T)={x\mid Tx=0,x\in X}&#36;For the counter.&#36;T,X\to Y&#36; Zero space or neural of the algorithm T
Easy to prove, Count.&#36;T&#36;Zero space must be.&#36;X&#36;A linear subspace</p>
<p>Here's to the very common nature of zero space, which is basically the only thing at their core, is zero space. Close</p>
<h4>Nature of zero space</h4>
<p>Theoretically:
Set&#36;T&#36;It's linear enabling space.&#36;X&#36;Linear Cursor on the top, zero space.
<strong>Linear Scattered Logs Launching Zero Space Close</strong>
Attention, this is a proposition that doesn't work.
Theoretically:
The X is a linear enabling space on the F-area.&#36;f:X\to\mathbb{F}&#36;For linear general letters, map the theorem 3.2.1
&#36;G:X/\ker(f)\to\mathbb{F}&#36;* The present document is being issued without formal editing.&#36;G([x])=G(x+\ker(f))=f(x)&#36;, at the same time&#36;G&#36; It's from commercial space. &#36;X/\ker\left(f\right)&#36; Present. &#36;f&#36; Value &#36;R\left(f\right)\subset\mathbb{F}&#36;The linear map on it.
<em>This is the theorem to supplement the evidence below.</em>
Theoretically:
The X is a linear enabling space on the F of several cities, and it is a very important opportunity for the government to create a new system of free speech.&#36;f:X\to\mathbb{F}&#36;For linear general communications, &#36;f&#36; For linear continuum<strong>When and only when</strong>Zero space.
<strong>Linear, continuous general correspondence and zero spatial closing prices;</strong>
Theoretically:
The X is a linear enabling space on the digital F.&#36;f:X\to\mathbb{F}&#36;* As non-zero-linear general communication, &#36;f&#36; For a continuous, generic, zero space, the ker(f) is dense in X Central Africa.
If it's dense, it's not linear.
<strong>And finally, it's connected to the denseness of zero space.</strong></p>
<h3>Linear boundary algorithm space</h3>
<p>The algorithm is defined in space, but he can't make a new space without taste.</p>
<h4>Definitions</h4>
<p>Set&#36;X&#36;and&#36;Y&#36;It's two linear spaces.&#36;L(X\to Y)&#36;Other Organiser&#36;X&#36; Present.&#36;Y&#36;The collection of all linear algorithms, that's...
&#36;L\left ( X\to Y\right ) = \left { T\right | T&#36; It's a linear item for X-Y.
We can verify that the additions and multipliers are defined in the following way, and we can determine that linear algorithms form a linear algorithm space.
&#36;&#36;00\
&amp;\left(T_{1}+T_{2}\right)\left(x\right)=T_{1}\left(x\right)+T_{2}\left(x\right), \
&amp;\left(\alpha T_{1}\right)\left(x\right)=\alpha T_{1}\left(x\right).
\end{aligned}&#36;&#36;
非常自然的 我们可以同样的定义线性有界算子空间
&#36;&#36;B\left(X\to Y\right)=\left{T|T\text{Yes}x\to Y\text{Line boundary }&#36;&#36;</p>
<p>It's a linear space, and it's a very natural idea. Let's explain this.
Definitions:
Set&#36;T\in B\left(X\to Y\right),T&#36;is defined as
&#36;&#36;\parallel T\parallel\triangle\sup_{x\neq0}{\frac{\parallel Tx\parallel}{\parallel x\parallel&#125;&#125;&#36;&#36;
Linear Enabling Space &#36;B\left(X\rightarrow Y\right)&#36;It's called linear algorithmic space, special memory.&#36;B\left(X\right)=B\left(X\to X\right)&#36;</p>
<h4>Nature</h4>
<p>Here are some explanations of the linear algorithmic space that is defined above, and here are very important, and finally, a few examples for practice.
When?&#36;X&#36;and&#36;Y&#36;It's two linear enabling spaces.</p>
<ul>
<li>&#36;T\in B\left(X\to Y\right)\text{当且仅当}\sup_{x\neq0}{\frac{\parallel Tx\parallel}{\parallel x\parallel&#125;&#125;\text{是有限值}.&#36;</li>
<li>&#36;\text{通过}\parallel T\parallel=\sup_{x\neq0}{\frac{\parallel Tx\parallel}{\parallel x\parallel&#125;&#125;\text{定义的范数满足“范数”三条公理}.&#36;</li>
<li>&#36;当x\in X时,\text{有}\left|T\left(x\right)\right|\leqslant\left|T\right|\cdot\left|x\right|.&#36;</li>
<li>&#36;\left|T\right|=\sup_{x\neq0}\left{\frac{\left|Tx\right|}{\left|x\right|}\right}=\sup_{\left|x\right|\to1}\left{\left|Tx\right|\right}=\sup_{\left|x\right|\leq1}\left{\left|Tx\right|\right}.&#36;</li>
</ul>
<p>Theoretically:
&#36;&#36;\text{设 }X\text{ 是有限维线性藏范空间,}Y\text{是任意的线性赋范空间,则}L\left(X\rightarrow Y\right)=B\left(X\rightarrow Y\right).&#36;&#36;
&#36;&#36;设 X 是线性服范空间,Y 是 Banach 空间,那么 B(X\to Y)是 Banach空间&#36;&#36;
The meaning of these two theories is still well understood because the numbers in limited-dimensional linear spaces must be continuous and well-defined, and the latter are indeed sufficient to prove their completeness.</p>
<h4>Projection</h4>
<p>Definitions:
M is the closed space on Hilbert Space H, map &#36;P:H\to M&#36; Define&#36;\forall x\in H,P\left(x\right)=x_{0},x-Px=x-x_{0}\in M^{\perp}&#36;
of which&#36;x_{0}&#36;Yes.&#36;x&#36;Yes.&#36;M&#36;Up the positive projection, called P is the projection or the positive projection on M ( Orthography Production Manager) &#36;P_M&#36;
It's easy to understand the projection algorithm, and we need to distinguish between a projection not on a full standard positive base, but a normal subspace.</p>
<p>Some of the nature of the projection algorithms.
M is Hilbert's space.&#36;H&#36;On the non-zero closed space, P-to&#36;&#123;M}&#36;on the projection, then
&#36;\left(1\right)P&#36;Zero space. &#36;\ker\left(P\right)=M^{\perp}&#36;,Range &#36;R\left(P\right)=M.&#36;
(2) P is the linear algorithm on H.
&#36;\left(3\right)\parallel P\parallel=1.&#36;</p>
<p>M is Hilbert's space.&#36;H&#36;On the non-zero closed space, P for the projection of M, then
&#36;H=\ker\left(P\right)\bigoplus R\left(P\right)&#36;
This theory tells us that the original space is the straightness of the projection range.
It's very easy to understand.</p>
<p>Set M is Hilbert Space &#36;H&#36; On non-zero closed space, P for a projection item on M, then
P for the soothsayer.
This is the theory that tells us to project the calculus.</p>
<p>H is Hilbert space. &#36;P\inB &<del>ker(P)\perp R(P)</del> P=P^2&#36; 则&#36;P-&#36; is a projection.
That's the reverse of the two theorems.</p>
<h3>Theorem for Ivoryspace and Risez.</h3>
<p>We've been studying all the linear algorithms that make up the linear enabling space, and now we're just a little bit more specific.
Research of linear general communications&#36;B(X\to F)&#36;The linear enabling space that makes up</p>
<h4>Pairspace</h4>
<p>Set&#36;X&#36;For a linear enabling space, the X-based liner broad message assembly&#36;B(X-(E))&#36;Mark as &#36;X^<em>♪ That's right ♪
&#36;X{</em>== sync, corrected by elderman == @elder man&#36;&#36;
称线性赋范空间 &#36;X^*&#36;为 X 的对偶空间或共轭空间(Conjugate Space)
为了方面后面的研究 规定一类函数
&#36;&#36;\left.\delta_{ij}=\left{\begin{matrix}1,i=j,\0,i\neq j.\end{matrix}\right.\right.&#36;&#36;
非常明显 对偶空间很复杂不适合我们研究 后面的工作就是简化对偶空间方便我们的研究
&#36;&#36; Sets the x-dimensional grant space, \\langle e  1, e   2, \\cdots, e  n\range is the base of X\text, \\text{there exists its foundation for the veneer space&#125;&#125;f 1, ...f n}&#36;&#36; 使得
&#36;&#36;=left(e j}right)=delta=j}&#36;&#36;
We know that the original space is a linear-based-enabled space, and this theory tells us that the results of mapping the original space-based base are the same.</p>
<p>Another theorem.
Pair space X-ray<em>It's Banach.&#36;&#36;
并且给出
再线性等距同构的意义下
&#36;&#36;(R^n)^</em>=R^n~(C^n)^*=C^n&#36;&#36;</p>
<h4>Risez for theorem</h4>
<p>Set H for Hilbert Space,&#36;f&#36;Yes.&#36;H&#36; There's only one line of letters on the line. &#36;z\in H&#36;, meet: &#36;\forall x\in H&#36;Yes.
&#36;f\left(x\right)=\left(x,z\right),\left|f\right|=\left|z\right|.&#36;</p>
<p>Risez says the meaning of the theorem is that
For all Hilbert space, linear broad letters can always be converted into a Hilbert space, the inner volume of space, and this linear broad letter is the typical of the point that the inner volume has been selected.
It's a very simple line of communication on Hilbert's space.</p>
<h3>Calculator Multiplication and Countercalculator</h3>
<p>Here we continue to study the structure of the linear enabling space of linear algorithms.</p>
<h4>Calculator Multiplication</h4>
<p><strong>Define Alcalculation product</strong>
Set&#36;X,Y,Z&#36;It's a linear enabling space in the same digital domain.
&#36;T_{1}\in B(X\to Y),T_{2}\in B(Y\to Z),\forall x\in X&#36;,Definition&#36;(T_{2}T_{1})x\bigtriangleup T_{2}(T_{1}x)&#36;, or&#36;T_{2}T_{1}&#36;Yes&#36;T_{1}&#36;Right times&#36;T_{2}&#36;or&#36;T_{2}&#36;Left times&#36;T_{1}&#36;
We can easily get the nature of the calculator product.
&#36;&#36;\begin{aligned}\quad\text{设 }X,Y,Z\text{ 是同一数域上的线性赋范空间,若 }T_1\in\mathbb{B}(X\to Y),T_2\in\mathbb{Q}(Y\to Z),\T_2T_1\in\mathbb{B}(X\to Z),\parallel T_2T_1\parallel\leqslant\parallel T_2\parallel\parallel T_1\parallel.\end{aligned}&#36;&#36;
That's the product of a calculator product with a range less than a scale.</p>
<p><strong>Definitions</strong> Algebra to exchange
Set&#36;X&#36; It's digital.&#36;F&#36;A linear space on it if it's for any element&#36;x,y,z\in X&#36; and&#36;\lambda\in\mathcal{F}&#36;, Existing multiplication meets&#36;xy\in X,x\left(yz\right)=\left(xy\right)z,x\left(y+z\right)=xy+xz,\left(x+y\right)z=xz+yz,\lambda\left(xy\right)=x\left(\lambda y\right)&#36;and&#36;X&#36;An algebra. If there is a non-zero element &#36;e\in X,\forall x\in X&#36; Yes. &#36;ex=xe=x&#36;, e is called the unit of the algebra X. If&#36;\forall x,y\in X&#36;Yes.&#36;xy=yx&#36;, or&#36;X&#36;is the algebra that can be exchanged. If online enabling spaces&#36;X&#36;and&#36;\forall x,y\in X&#36;Yes.&#36;|xy|\leqslant|x||y|&#36;, or&#36;X&#36;For the grant of the algebra, the perfect enabling algebra is called Banach Algebra.
We have defined some of the characteristics of the algorithms in the given algebra.
&#36;&#36;\left(T_{1}T_{2}\right)T_{3}=T_{1}\left(T_{2}T_{3}\right)&#36;&#36;
&#36;&#36;I:X\to X\text{ 为, I}x=x.\parallel I\parallel=1,\forall T\in B(X),有IT=TI=T.&#36;&#36;
&#36;&#36;P\left(T\right)x=a_{0}x+a_{1}Tx+a_{2}T^{2}x+\cdots+a_{n}T^{n}x.&#36;&#36;</p>
<p>The numbering of algorithms is combined, the number of units, the number of algorithms.</p>
<h4>Counter-calculations</h4>
<p>Set&#36;X,Y&#36;It's a linear enabling space in the same digital domain, and&#36;T\in B(X\to Y)&#36;, if exists&#36;S\in B(Y\to X)&#36;- Yeah.&#36;ST=I_X,TS=I_Y&#36;, T is called a reversible counter-calculate and S and T are counter-calculated, as &#36;T^{-1}=S&#36; - Where?&#36;I_{X}&#36; &#36;I_{Y}&#36;The difference is...&#36;X,Y&#36;The constant number of the count.</p>
<p>Easy to prove that T's reversible counter-calculate S only exists and (&#36;T^{-1})^{-1}=T&#36; and (I)&#36;T)^{-1}=-T^{-1}&#36;, the product of reversible algorithms can also reverse
It's easy to draw the following conclusions:</p>
<ul>
<li>&#36;\left(T^{-1}\right)^{-1}=T&#36;</li>
<li>&#36;\left(ST\right)^{-1}=T^{-1}S^{-1}&#36;
Theoretically:
&#36;&#36;00\&amp;\\text{x\text{is Banach space if \bardsymbol{in}\bardsymbol{B},\underline{t|T|&lt;I'm not a real guy.&amp;♪ I'm a good girl ♪
Theoretically:
If&#36;X&#36;It's linear enatch space, Y is Banach space, then.&#36;&#123;B}(X\to Y)&#36;All the reversible counter-calculators form an opening of it.</li>
</ul>
<h3>Theorem of the Baire</h3>
<p>The two consecutive sections below are the theorems of the reverse counter-calculations.
Theoretically:
Set&#36;X,Y&#36;It's linear enabling space, and&#36;T\in L(X\to Y)&#36;Well, then...&#36;T\in B(X\to Y)&#36;When and only when&#36;\left{x\in X\right|\parallel Tx\parallel\leqslant1 }的内部为非空集&#36;。</p>
<p>Definitions:
Set X is measuring space, A드X, if the inner of A's closed (internal: internal assembly of points) is empty, i.e.&#36;(\overline{A})^{\circ}=\phi&#36;if A can be expressed as a sum of up to a few fractions, i.e.&#36;A=\bigcup_{n}^{\infty}A_{n},A_{n}&#36;- It's a rare collection, and it's called A as the first set;
The word for thinness is literally, it's not dense in any kickoff.
The determination of thinness gives the following characteristics:</p>
<ul>
<li>&#36;A&#36;To Rare Sorrow</li>
<li>&#36;\bar{A}&#36;A neighbourhood with no point.</li>
<li>&#36;\bar{A}^C&#36;Thin
Attention, although the original silt can be supplemented by dense, the dense remix may not be thin, and one cannot be judged by this as a collection of thin ones.
So we give the common thinness below. Set</li>
</ul>
<p>O'Culture Space R.&quot;, and in particular, single dot {x} is thin, so any of the old columns is the first. As (in)&#36;\overline{\mathbb{Q&#125;&#125;)^{\circ}=\mathbb{R}^{\circ}=\mathbb{R}&#36;So the Rational Numerical Set Q in R is not a thin collection.
Because&#36;&#123;x}=O(x,0.5)&#36;It's dispersive measure space (X,&#36;d_0&#36;) is a single-point set, so {x} is the second-stage collection.
The X is a measure of space.&#36;x_0\in X&#36;, if there is a neighbourhood&#36;O(x_0,\delta)&#36;♪ And make ♪&#36;O(x_0,\delta)\bigcap X={x_0}&#36;, or&#36;x_{0}&#36;Isolain Point.
And give some of the nature.</p>
<ul>
<li>Subsets of thin and closed-pack thin</li>
<li>A limited and thin collection.</li>
<li>The measure space, without isolated points, is thinner than the limited collection.</li>
</ul>
<p>&#36;&#36;\begin{aligned}\text{ 设 }X\text{ 是度量空间,}A\subseteq X,\text{那么 }A\text{ 是稀疏集的充要条件是对于任意开球}\O(x,\epsilon),\text{ 存在 }O(y,r)\subset O(x,\epsilon),\text{使得 A}\bigcap Q(y,r)=\phi.\end{aligned}&#36;&#36;
&#36;&#36;\text{完备的度量空间(X,d)是第二纲集.}&#36;&#36;</p>
<h3>An open map theorem and a countercalculator theorem</h3>
<h4>Open Map</h4>
<p>X.Y. is linear, if you're a counter.&#36;T:X\to Y&#36;It's gonna work.&#36;X&#36; , whichever is an open map of the Y, is called the Albino T.
It's easy to know that all the collections in the dispersive measure space are open, so...&#36;Y&#36;When the discrete measure space is, all the maps are open; it's easy to know that the complex of the open is open.
Inferences:
&#36;&#36;\begin{aligned}\text{set}X&amp;\text{is linear enabling space,}A, B\subseteq X.\text{xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx&amp;\in\left(A+B\right)^{\circ}.\end{aligned}&#36;&#36;</p>
<p>An open map theorem:
&#36;&#36;\text{设 }X,Y\text{ 是 Banach 空间, 算子 }T\in B(X\rightarrow Y),R(T)=Y,\text{则 }T\text{ 为开映射}.&#36;&#36;
As long as the linear continuum is full of numbers, it must be a screen opening.</p>
<h4>Counter-calculations</h4>
<p>Inverse algorithm:
Set X. Y is Banach space, Count.&#36;T\in\mathbb{B}(X\to Y)&#36;If the count is full of fire (single and full of fire), then &#36;T^{-1}\in B\left(Y\to X\right).&#36;
The core point is Banach.
Inferences:
Linear Enabling Space&#36;X_{2}&#36;There are two of them.<em>♪ And the chords ♪</em>{2}&#36;  这两个范数定义下均是Banach空间，而且&#36;|\cdot|<em>{2}&#36;比&#36;|\cdot|</em>{1}&#36;强)那么范数&#36;|\cdot|<em>♪ And the chords ♪</em>Equivalent
Inferences
(1) Constant exists &#36;M&gt;0&#36;,使得 &#36;\forall x\in D\left(T\right)&#36; 有&#36;\parallel Tx\parallel\geq M\parallel x\parallel&#36;,则 &#36;T&#36; 可逆&#36;^ (R\left)\rightarrow D\left (T\right)\right, and \\forall y\inR\left (T\right) has \\right
&#36;\left|T^{-1}y\right|\leq\frac{1}{M}\left|y\right|.&#36;
&#36;&#36;00begin{aligned}\text{if \text{and }t\t\t\t(R\left(T\right)\t\d\right),\text{th)}M&gt;\text{en \light\text\ I'm sorry.</p>
<h3>Linear General Letters Extension Theorem</h3>
<p>Set&#36;M&#36;For linear enabling space.&#36;X&#36; Subspace, for definition in subspace&#36;M&#36;Linear General Letters Up&#36;f\in M^{\circ}&#36;, if there's space&#36;X&#36;Linear General Letters Up&#36;F\in X^*&#36;♪ Makes when ♪&#36;x\in M&#36;The blog is also available.&#36;F(x)=f(x)&#36;, and call it "F."&#36;f&#36; The extension on X, f is the F on M, is recorded as&#36;F|_{M}=f&#36;
We're going to discuss the existence of this expansion.</p>
<p><strong>Hahn-Bananch Extension Theorem</strong>
Subspace with M as Linear Enabling Space X, &#36;f\in M^<em>}&#36;,则存在 &#36;F\in X^{</em>}&#36;,使得&#36;F|
&#36;||F||=||f||&#36;
This is what we finally gave you.<strong>Hahn-Bananch Extension Theorem</strong> So, as long as it's linear, it can be extended and expanded to match the original algorithm.</p>
<p>Inferences:
Set&#36;X&#36;For a linear fragrance, for whatever.&#36;x_0\in X&#36;,&#36;x_0\neq0&#36;♪ And there's always ♪&#36;x&#36;Linear Consequences Up&#36;f&#36;,Fulfilled&#36;f(x_0)=|x_0|&#36;and&#36;|f|=1.&#36;</p>
<p>Finally, there must be plenty of space for any linear enabling space.</p>
<h3>Closed Image Theorem</h3>
<p>We've been telling the calculus that the function is a curve, and if the function is a continuous, the corresponding point in the image is a closed collection, we want to study similar properties in online enabling spaces.</p>
<h4>Linear enabling spaces in the product space</h4>
<p>Set&#36;X&#36; and&#36;Y&#36;It's a linear F-based space on the same digital field.&#36;X\times Y=\left{\left(x,y\right)\left|x\in X,y\in Y\right}\right.&#36;Yes.&#36;X\times&#36; Y defines additions and multipliers as follows:&#36;\forall(x_1,y_1),(x_2,y_2)\in X\times Y&#36; and&#36;\forall_a\in\mathbb{F}&#36;Yes.
&#36;(x_{1},y_{1})+(x_{2},y_{2})=\left(x_{1}+x_{2},y_{1}+y_{2}\right),\alpha\left(x_{1},y_{1}\right)=\left(\alpha x_{1},\alpha y_{1}\right)&#36;I'm not sure.
Well...&#36;X\times Y&#36;Composition of linear space
Set&#36;x\in X,y\in Y&#36;, with the corresponding numbers&#36;|x|,|y|&#36;♪ So then ♪&#36;X\times Y&#36;Predefinable
&#36;&#36;0000\begin{aligned}\left(x,y\right)|<em>{p}&amp;=\left(|x|^{p}+|y|^{p}\right)^{\frac{1}{p&#125;&#125;,1\leq p&lt;+\infty,\|\left(x,y\right)|</em>{\infty}&amp;♪ I'm a little bit old ♪
Most commonly,&#36;p=1,p=2&#36;And the infinite number
Obviously, the test of the product space is also a linear enabling space, all of which can be studied.</p>
<h4>Closed algorithm</h4>
<p>Set&#36;X&#36; and&#36;Y&#36;is the linear enabling space on the same digital field F, if T is an image (Graph)
&#36;G\left(T\right)=\left{\left(x,y\right)\mid y=Tx,x\in D\left(T\right)\right}&#36;
It's the product space.&#36;X\times Y&#36;closed verse of the&#36;T&#36;It's a closed-line algorithm. It's called a closed-accounter.</p>
<p>&#36;T:D(T)(\subset X)\to Y&#36;It's the first of the linear universe, if D(T) is the closed subspace of X, then T is the closed algorithm.
This is the theorem, Jean.&#36;D(T)=X&#36; That means:
<strong>Linear algorithms are closed.</strong></p>
<h4>Closed Image Theorem</h4>
<p>The X and Y are both Banach spaces.&#36;T:D(T)(\subset X)\to Y&#36; It's a closed algorithm. D(T) is a closed subspace for X. T is a linear algorithm.
This is the theorem, Jean.&#36;D(T)=X&#36; That means:
<strong>The CBS is a linear one.</strong> If he's the counterman in Banach space,</p>
<p>Inferences:
&#36; \begin{aligned}\textbf}&amp;\\text{}X\text{and Y are both Banach space, \t\L(X\toY),\text{so}T\text{and}Linelinear algorithms. \text{T only is T}Closed algorithms. \Box\end{aligned} I'm sorry.
This is a very good theory, and we've been working on a lot of questions about linear algorithms, and now as long as he's Banach, he's a closed counter.
In other cases, there's no bounds.</p>
<h3>It's a definition.</h3>
<p>There's a problem with a number of numbers.
The X and Y are linear enabling spaces on the same digital F.&#36;F\subset B(X\to Y)&#36;, if&#36;||T||~T\in F&#36;There's a boundary, and the count is F.</p>
<p>X is Banach Space, Y is linear enamel space, the countenance &#36;F\subset B(X\to Y)&#36; So the count is a fair and simple F.&#36;\forall x\in X,\left{\parallel Tx\parallel\parallel T\in F\right}&#36;For a boundary set
<strong>The problem of the conglomerate is the result of the mapping.</strong>
That's what we call a consistent definition. It's called a resonance theory.</p>
<h3>Banach won't move the theorem.</h3>
<h4>Basic concepts</h4>
<h5>Don't move.</h5>
<p>Set&#36;x&#36; It's a non-empty collection if it's for mapping.&#36;A{:}X\to X&#36; ,&#36;x^<em>\in X&#36;满足&#36;A(x^</em>)=x^<em>&#36;,则称&#36;x^</em>&#36;为映射&#36;A.A.'s fixed points</p>
<p>If the map is known, the no-motion point is the equation, but at this point we want to do nothing about the equation and study the existence of the no-motion point.</p>
<h5>Compression Map</h5>
<p>Set&#36;x\text{为度量空间，如果映射}_{A:X\to X}&#36;, constant&#36;\alpha\in(0,1)&#36;, &#36;\forall x,y\in X&#36;Yes.&#36;d(Ax,Ay)\leq\alpha d(x,y)&#36;and&#36;A&#36;Yes&#36;X&#36;Up the compression map. Call constant&#36;\alpha&#36;Compression factor.
It's obviously of the following nature.</p>
<ul>
<li>Compression map is continuous map</li>
<li>Compression map compound or compression map
Now we want to know what happens to a problem that uses compression mapping again and again.
That's right.&#36;\text{记}x_n=A^n(x_0)\text{,那么点列}{x_n}\text{有什么特点}?&#36;</li>
</ul>
<h4>Banach won't move the theorem.</h4>
<p><strong>Banach won't move the theorem.</strong>
Set&#36;X&#36; It's a perfect measure space.&#36;A{:}X\to X&#36; is a compressed map, then&#36;A&#36;Yes.&#36;X&#36;with the only no-motion point&#36;x&#36;♪ Make &#36;x ♪<em>=A(x^</em>I'm not sure.
So for a full measure space, the no-motion point of the compressed map exists and is the only one that can be used to measure the space.
This conclusion is only a sufficient condition, not necessary.
Now, let's look at the proof of the theorem, which is actually the process of finding no action.
<em>The overall idea of proof is to construct Corsile with a compressed map, to use completeness to get a draw point, and to discuss whether the draw point is a no-go point.</em>
Compressed map construction basic columns:
&#36;\text{supplier}<em>\text{, \levft{x n\right},\text{x n=A(x)</em>I'm sorry, I'm sorry.
It's easy to know.
&#36;&#36;00\
d (x n}, x n+k})&amp; \leq d(x_{n},x_{n+1})+d(x_{n+1},x_{n+2})+\cdots+d(x_{n+k-1},x_{n+k})  \
&amp;\lpha^lpha^lpha}c=frac{\lpha}lpha}c}lpha}
I'm sorry, I'm sorry.
So the compressed map does get the basic column.
Give me a draw point:
Because X is a full measure space, so the basic column is constricted.<em>(n\rightarrow\infty)
&#36;&#36;x{</em>}=\lim_{n\to\infty}x_{n}=\lim_{n\to\infty}A(x_{n-1})=A(\lim_{n\to\infty}x_{n-1})=Ax^{<em>}.&#36;&#36;
证明唯一性
假设存在第二个不动点 则
&#36;&#36;d(x_{1}^{</em>},x^{<em>})=d(Ax_{1}^{</em>},Ax^{<em>})\leq\alpha d(x_{1}^{</em>},x^{<em>})&#36;&#36;
因此
&#36;&#36;(1-\alpha)d(x_{i}^{</em>},x^{<em>\leq0,\text{and so}d(x i}^</em>},x^{<em>{\l7F4}\leq0,\text{</em>You know, it's a good idea to have a good time.
That's the way it's gonna be.</p>
