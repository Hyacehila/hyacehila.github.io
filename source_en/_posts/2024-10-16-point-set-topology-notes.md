---
title: 'Point-Set Topology: Topological Spaces, Continuity, and Subspaces'
title_zh: 点集拓扑学：拓扑空间、连续性与子空间
date: 2024-10-16 23:06:09 +0800
categories:
- Mathematics
- Geometry & Topology
tags:
- Topology
author: Hyacehila
mathjax: true
hidden: true
excerpt: Covers topological spaces, continuity, subspaces, product spaces, quotient spaces, connectedness, and compactness.
description: Covers topological spaces, continuity, subspaces, product spaces, quotient spaces, connectedness, and compactness.
excerpt_zh: 整理拓扑空间、连续性、子空间、积空间、商空间、连通性和紧致性等内容。
permalink: /blog/2024/10/16/point-set-topology-notes/
lang: en
translation_key: 2024-10-16-point-set-topology-notes
translation_status: machine
translation_source_hash: ac9eea60e6931fd6d11b59910c072ea479d7a9aa7c127f537fef49f90a29a6c2
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<h2>Introduction</h2>
<p>The megapolymic study of geometry, which is completely different from that studied in geometry in Europe, has nothing to do with measurement and shape, but with the overall structural properties of the graphic.</p>
<p>So, we can give a change that doesn't change the nature of the polo.<strong>Toggle Change</strong>) Definition:</p>
<p>Put Graphics&#36;M&#36; Convert to &#36;M^&#39;}&#36;   就是给出一个一一对应（不重叠，不产生新的点）的映射 &#36;f:M\to M^{&#39;}&#36;  并且映射&#36;f&#36; 连续（不产生撕裂） 同时&#36;f^{-1}&#36; 也连续（不产生粘连）此时我们称&#36;f&#36; 是一个拓扑变换，并且我们称&#36;M,M^{&#39;It's identical</p>
<p>So-called<strong>Twist nature</strong>It's the nature common to the image of the embryo, and it doesn't distinguish it from that of the embryo, because they have the same character.</p>
<p>As can be seen from the above, mega-modulation is an important means of studying graphics and embryos, and we can judge their homogeneity by construction. On the other hand, we can judge whether or not they're congenial in the same way.</p>
<p>We then focus on general dot-to-do and a little algebra-to-do knowledge as the basis for research into other abstract mathematical structures, which is an effective means of studying high-dimensional space.</p>
<h2>Space and continuity</h2>
<p>The basic definitions of the transformational and proprio motu nature have been presented in the introductory part of this paper, but we are still engaged in discussions based on European space. But the broader mathematics has long been removed from European space, and in order to study them we need to draw out a broader space structure than normal European space and measurement space. We generally call it the architecture of space.</p>
<h3>Totospace.</h3>
<p>The continuity of mapping is an important concept for painting transformations, so that the new space structure we are looking for can paint this.</p>
<p>In analyticals, the main way we study continuity is by</p>
<ul>
<li>&#36;\epsilon - \delta&#36; Languages</li>
<li>Sequence-deep language</li>
<li>Opening language: if&#36;V&#36;Yes.&#36;f(x_0)&#36; , which contains&#36;x_0&#36; Opening&#36;U&#36; Make&#36;f(U)\subset V&#36;</li>
</ul>
<p>Where are we?<a href="/en/blog/2023/03/16/mathematical-analysis-limits-continuity-notes/">Mathematic analysis 1. The theory of limits and continuity</a>It's mostly the first language.<a href="/en/blog/2023/09/11/functional-analysis-notes/">Analysis of general communications</a>The second language is used mainly. They all ask us to define the distance in space, while the third circumvents the limit of distance. It is only necessary to define the openings in space to define continuity, more simply than the first two, which is a good choice to define the expansion of space.</p>
<h4>Definition of space</h4>
<p>Set&#36;X&#36;A non-empty collection, remember?&#36;2^{X}&#36;Yes.&#36;X&#36;♪ A collection of ♪&#36;X&#36;All subsets (including empty collections)&#36;\varnothing&#36;and &#36;X&#36; To be a team of members. &#36;2^X&#36;It's called&#36;X&#36; The sub-clan.</p>
<p>Definitions&#36;X&#36;It's an empty collection. &#36;X&#36;A sub-clan.&#36;\tau&#36;Called&#36;X&#36;A pouncer if it meets</p>
<ol>
<li>&#36;\textit{ X}, \varnothing&#36;It's all in. &#36;\tau&#36; Medium;</li>
<li>&#36;\tau&#36; The sum of any number of members still exists&#36;\tau&#36;Medium;</li>
<li>&#36;\tau&#36;The convergence of a limited number of members is still ongoing.&#36;\tau&#36;Medium.</li>
</ol>
<p>Gather!&#36;X&#36;And one of its pouncers.&#36;\tau&#36;It's called a poking space. &#36;( X, \tau )&#36;
Claims&#36;\tau \textbf{  }.&#36;This is the beginning of this space.</p>
<p><strong>The three conditions in the definition are called trombone.</strong>．</p>
<p><strong>By definition, one of the extensions given to the assembly is to specify which subsets of it are the beginning.</strong>... this provision is not arbitrary,<strong>There must be three rules of justice.</strong>．</p>
<p>Generally speaking, a pool can provide for many different leaps, so when it comes to a popping space, it should be accompanied by an indication of the rally and the required leaps. In the future, without misunderstanding, it is often only called a poking space. &#36;X&#36; It's open space. &#36;Y&#36; Wait.</p>
<p>Obviously, when&#36;X&#36; It's a time to meet. &#36;2^X&#36;  Make one.&#36;X&#36; The one on top, we call it.<strong>I'm not going anywhere.</strong>;at the same time &#36;{X,\phi}&#36; It's also a puff. We call it<strong>It's normal.</strong>I don't know. When the X contains more than one point, the two are different. As for more questions about what constitutes a poking, they can be verified directly by definition.</p>
<p>Set&#36;\tau_1,\tau_2&#36;It's a rally.&#36;X&#36;♪ Up two pounces if ♪&#36;\tau_1\subset\tau_2&#36;And then...&#36;\tau_2&#36;Bigger.&#36;\tau_2&#36;That's right.&#36;\tau _1\textbf{  }&#36;♪ Discretion poking is bigger than any other ♪ and ordinary puffing is smaller than any other ♪</p>
<p>In particular, we've given us a few of the usual pouncers.</p>
<ul>
<li>Set&#36;X&#36;and will gather endlessly,&#36;\tau _f= { A^c\mid A\textbf{ 是 }X\textbf{ 的有限子集 }\cup \left{\varnothing\right}&#36;, which is not difficult to verify&#36;\tau_f&#36;Yes.&#36;X&#36;♪ A pouncer called ♪&#36;X\textbf{  }.&#36;I've got more than enough.</li>
<li>Set&#36;X&#36;There is no limit to it.&#36;\tau _c= { A^c\mid A\textbf{ 是 }X\textbf{ 的 可 数 子 集 } \cup }\left{\varnothing\right}&#36;, then&#36;\tau_c&#36; Yeah. &#36;X&#36; It's called "the last one."</li>
<li>Set&#36;R\textbf{  }&#36;It's a collection of all the actual numbers. <em>e = {U|U\textbf}&#36;,这里“若干”可以是无穷，有限，也可以是零，因此&#36;\varnothing\in&#36; &#36;\tau</em>{e}.&#36;则&#36;\tau  e\textbf{R}&#36; 上的拓扑，称为 &#36;R&#36; 上的欧氏拓扑. 记 &#36;E^1=(R,\tau_e).&#36;</li>
</ul>
<h4>Metrics expand.</h4>
<p>Conceptual reference on measuring space <a href="/en/blog/2023/09/11/functional-analysis-notes/">Definition of measurement space in general correspondence analysis</a> We're studying measuring space in this section.</p>
<p>Set&#36;(X,d)&#36; It's a measuring space. Let's define a scale.</p>
<p>Because of the distance structure of the measuring space, we can easily give the concept of a neighbourhood as follows:
&#36;B(x 0,\varepsilon): =x\inX|d(x 0,x)&lt;\varepsilon}&#36;&#36;</p>
<p>Provisions&#36;X&#36; The sub-clan. &#36;\tau_d&#36;  Yes.&#36;\tau_d={U|U\text{ 是若干个球形邻域的并集&#125;&#125;&#36; We can give without false proof.&#36;\tau_{d}&#36; Yes.&#36;X&#36; The one up there.</p>
<p>We call this space from measurements.<strong>Metrics expand.</strong> Each measurement space is naturally seen as a poking space with scale-up. That's how it works.&#36;E&#36; It's also a space for poking.</p>
<p>In this sense,<strong>Totospace is an extension of the Eurospace and measuring space.</strong>And the three principles of exculpatory justice are abstract from the most basic nature of the beginning of measuring space.</p>
<h4>A few basic concepts in space.</h4>
<p>The basic concepts that we're going to talk about here have appeared in the Oxygen space and in the measurement space, but now they are defined by the open concept.</p>
<p>This section can be combined.<a href="/en/blog/2023/09/11/functional-analysis-notes/">The sprawl nature of measuring space in general correspondence analysis</a>Think together.</p>
<h5>Closed</h5>
<p>Definition (closed): Popping Space&#36;X&#36;A subset&#36;A&#36; It's called closed, if&#36;A^c&#36;It's the beginning.</p>
<p><strong>In other words, closed collections are the rest of the openings, which in turn must be the rest of the closings.</strong></p>
<p>The closed set of spaces meets:</p>
<ul>
<li>Empty and full space are closed.</li>
<li>Any kind of closed encounter is closed.</li>
<li>The limited closed collections are closed.</li>
</ul>
<p>It's also the nature of measuring space closure, as we mentioned earlier.<strong>Totospace is an extension of the Eurospace and measuring space.</strong> Some of their nature is abstract.</p>
<h5>Neighbourhood, Inner Point and Internal</h5>
<p>Definitions (neighborship, interior and interior)
Set &#36;A&#36; It's open space. &#36;X&#36; A subset, point. &#36;x\in A&#36;...if an opening exists &#36;U&#36; ♪ That makes ♪ &#36;x\in U\subset A&#36; , or &#36;x&#36; Yes. &#36;A&#36; A point inside. &#36;A&#36;Yes. &#36;x&#36; a neighbourhood;&#36;A&#36; It's called the collection of all the inner spots. &#36;A&#36; Internal, recorded &#36;A^{\circ}&#36;.</p>
<h5>Gathering and Closed</h5>
<p>Set&#36;A&#36;It's open space.&#36;X&#36;Subset,&#36;x\in X.&#36;If&#36;x&#36;Every neighborhood contains&#36;A\backslash{x}&#36;midpoint, then called&#36;x&#36;Yes&#36;A&#36;Other Organiser &#36;A\textbf{  }&#36; And all the gatherings are called &#36;A\textbf{ }&#36;The guide book.&#36;A^{\prime }.&#36; Summon &#36;\overline A:=A\bigcup A^{\prime}&#36;Yes &#36;A&#36; Closed</p>
<h4>Subspace</h4>
<p>Set &#36;A&#36; It's open space.&#36;(X,\tau)&#36; A non-empty collection.</p>
<p>Provisions&#36;A&#36;The sub-clan.
&#36;&#36;\tau_A:={U\cap A|U\in\tau}.&#36;&#36;
Easy to verify &#36;\tau_A&#36; Yes.&#36;A&#36; The one up there called&#36;\tau&#36;Export &#36;A&#36; # Up above the subspace #&#36;(A,\tau_A)&#36;Yes&#36;(X,\tau)&#36;Subspace</p>
<p>From now on, the subsets of the pinging space will be considered as pinging space, the subspace.</p>
<p>We're in the subspace of space.&#36;A&#36; Go up and export subspace&#36;B&#36; And directly from&#36;X&#36; The export subspace gets the same.</p>
<p>Similarly, if we have a scale expansion, whether to export the subspace from the scale, or to export the measure space from the scale space, then to expand the space, they all have the same space.</p>
<h3>Continuous mapping and congener mapping</h3>
<p>The continuous mapping is another basic concept and subject of research in telegraphy.</p>
<h4>Definition of continuous mapping</h4>
<p>Definitions: Establishment&#36;X&#36;and&#36;Y&#36;It's all space.&#36;f: X{\rightarrow }Y\textbf{ 是 一 个 映 射 , }x&#36; &#36;\in X.&#36;What if...&#36;Y&#36; Medium &#36;f(x)&#36;Any neighbourhood&#36;V, f^- 1( V)&#36; Always &#36;x&#36; And the neighborhood.&#36;f\textbf{  }&#36;Yes.&#36;x\textbf{ }.&#36;Continuous</p>
<p><strong>Obviously, what we're responding to here is a series of maps satisfying the openings, like the openings, which are the parts that are drawn from European space and measurement.</strong></p>
<p>Title&#36;f:X\boldsymbol{\rightarrow}Y&#36;A map,&#36;A\textbf{ 是  }X&#36;Subsets&#36;,x\boldsymbol{\in}A.&#36;Remember&#36;f_A=f|A:A{\rightarrow}Y&#36;Yes.&#36;f&#36; Yes. &#36;A&#36;On the limit, yes.</p>
<ul>
<li>If&#36;f&#36; Yes. &#36;x&#36; It's continuous. &#36;f_A&#36; Yeah.&#36;x&#36; Continuous</li>
<li>If&#36;A&#36; Yes. &#36;x&#36; The neighborhood.&#36;f_A&#36; Yes. &#36;x&#36; In succession. &#36;f&#36; It's the same thing.
The first part of the proposition is very natural, and the second part tells us. <strong>Consistency is a local nature, related only to one neighbourhood.</strong></li>
</ul>
<p>Definition: If Map&#36;f: X\to Y&#36; At either point. &#36;x\in X&#36; It's a series of places, and we call it a series of maps. He's a whole set of images.</p>
<p>Inference: Map&#36;f: X\to Y&#36;  The following propositions are equivalent.</p>
<ul>
<li>&#36;f&#36; It's a continuous map.</li>
<li>It's like an opening.</li>
<li>It's like a closed collection.</li>
</ul>
<p>Here we point out that, although there is also the concept of serial condensation in the space, it cannot be used to paint continuity. It's guaranteed. &#36;x_{n}\to x&#36; Time &#36;f(x_{n})\to f(x)&#36;  But that's not true.</p>
<h4>Nature of continuous mapping</h4>
<p>First, point out a few simple and common continuous maps.</p>
<ul>
<li>Constant mapping is continuous mapping, i.e.&#36;f:X\to X(id(x)\to x)&#36;</li>
<li>Set &#36;A&#36; Yes. &#36;X&#36; The subspace contains maps &#36;f:A\to X(id(x)\to x)&#36; It's a continuous map.</li>
<li>Constant map is continuous map</li>
<li>If&#36;f:X\to Y&#36; of which &#36;X&#36; It's a free space, or... &#36;Y&#36; It's normal space, but the mapping is continuous.</li>
<li>Two consecutive maps are complex; two continuous maps at a point are complex maps</li>
</ul>
<p>Then let's introduce some simple theory.</p>
<p>Set&#36;\mathscr{B}\subset2^X&#36;It's open space.&#36;X&#36;Sub-clans, if&#36;\bigcup_{C\in\mathscr{B&#125;&#125;C=X&#36;(i)&#36;\forall x\in X&#36; At least in&#36;\mathscr{B}&#36; of a member)&#36;\mathscr{B}&#36;Yes.&#36;X&#36;One overlay,
If Overwrite &#36;\mathscr{B}&#36; Each member is an open (closed) collection.&#36;\mathscr{B} \textbf{为 开 ( 闭 ) 覆 盖 ; 覆 盖 }\mathscr{B}&#36; For limited membership, it is called limited coverage.</p>
<p>Theorem (Accumulation Introduction)&#36;\left{A_1,A_2,\cdotp\cdotp\cdotp,A_n\right}&#36;Yes.&#36;X&#36;A limited closed cover.&#36;f:X\boldsymbol{\rightarrow}Y&#36;Every one.&#36;A_i&#36;The limits are continuous, then.&#36;f&#36;It's a continuous map.</p>
<p>He allowed the continuity of our partial mapping.</p>
<h4>Same embryo map</h4>
<p>Here we can respond to the questions in the Introduction.</p>
<p>Definitions&#36;f: X{\to }Y\textbf{  }&#36;It's a match, and... &#36;f,f^{- 1}&#36;It's all continuous.&#36;f\textbf{ }&#36;It's a congener map, or an amphibious transformation, or a congener.&#36;X&#36;Present.&#36;Y&#36;When the same embryo is mapped,&#36;X&#36;and&#36;Y\textbf{  }&#36;Congenital, written&#36;X\cong Y.&#36;</p>
<p>It's easy to give some nature.</p>
<ul>
<li>It's an equal value for a congener in a cosmopolitan collection.</li>
<li>Consistency map is congenial map.</li>
<li>If&#36;f&#36; It's a congenial map. &#36;f^{-1}&#36; It's the same embryo map.</li>
</ul>
<p>With the concept of homogeneity, the concept of the nature of a pomegranate can be defined, as in the introduction.</p>
<p>The concept of expanding the space in the same embryo map is called<strong>The concept of poking</strong>It's called the unaltered nature of the same embryo.<strong>Twist nature</strong></p>
<p>It can be seen that the opening is the concept of enlargement; the concepts of closure, containment, neighbourhood, interior, etc., are all concepts of enlargement. The nature of the painting with the idea of the initialization or its derivatives is one of enlargement; for example, severability is expansion.</p>
<p>The question of the classification of homogeneity in the study of poking space is an essential one of purging. It's important to it.</p>
<h3>Multiplication Space & Toppell</h3>
<h4>Multiplication Space</h4>
<p>Set&#36;\mathscr{B}&#36;Yes.&#36;X&#36;One of the sub-clans.
&#36;\overline{\mathscr{B&#125;&#125;:=\left.\left{U\subset X|U\text{ 是 }\mathscr{B}\text{ 中若干成员的并集}\right}\right.&#36;
&#36;=\left{U\subset X\mid\forall x\in U,\text{存在 }B\in\mathscr{B},\text{使得 }x\in B\subset U\right}.&#36;
Claims&#36;\overline{B}&#36;Yes&#36;\mathscr{B}&#36;It's obvious.&#36;\mathscr{B}\subset\overline{\mathscr{B&#125;&#125;,\varnothing\in\overline{\mathscr{B&#125;&#125;.&#36;</p>
<p>Set&#36;X_1&#36;and&#36;X_2\textbf{ 是 两 个 集 合 , 记 }X_1\times X_2&#36;For their Diccal stock:
&#36;&#36;X_1\times X_2={(x_1,x_2)|x_i\in X_i}.&#36;&#36;
Provisions&#36;j_i:X_1\times X_2{\operatorname*{\to&#125;&#125;X_i&#36;Yes&#36;j_i(x_1,x_2){\operatorname*{=&#125;&#125;x_i(i{\operatorname*{=&#125;&#125;1,2)&#36;,&#36;j_i&#36;Yes&#36;X_1{\operatorname*{\times&#125;&#125;X_2&#36; Present.&#36;X_i&#36;projection</p>
<p>Set &#36;(X_1,\tau_1),(X_2,\tau_2)&#36; It's two pedestals. Now we're gonna be in Cartesian. &#36;X_{1}\times X_2&#36; It provides for a pedestal that is closely linked to known pedestals.&#36;\tau&#36;  Make&#36;j_i&#36; It's the smallest one that's ever done so.</p>
<p>Theorem: Construct&#36;X_{1}\times X_2&#36; Sub-clan&#36;\mathscr{B}={U_{1}\times U_{2}|U_{i}\in\tau_{i&#125;&#125;&#36; We call it &#36;\overline{\mathscr{B&#125;&#125;&#36; Yes.&#36;X_{1}\times X_2&#36;♪ Up on the product scale ♪ &#36;(X_{1}\times X_2,\overline{\mathscr{B&#125;&#125;)&#36; It's a product space. &#36;X_{1}\times X_2&#36;</p>
<p>A similar approach could provide for a limited amount of excretion space.</p>
<p>The multiplication of space is combined, that is,
&#36;&#36;X_1\times X_2\times X_3=(X_1\times X_2)\times X_3=X_1\times(X_2\times X_3).&#36;&#36;</p>
<h4>Nature of the product space</h4>
<p>The definition of multiplication is directly projected.&#36;j_i:X_1\times X_{2} \to X_i&#36;Continuity.&#36;j_i&#36;It's still on the map.</p>
<p>Set&#36;Y&#36;It's any kind of popping space.&#36;f:Y\overset{\cdot}{\operatorname*{\operatorname*{\to&#125;&#125;}X_1\overset{\cdot}{\operatorname*{\operatorname*{\times&#125;&#125;}X_2&#36;It's a map.&#36;f_i=j_i\circ f:Y{\to}X_i(i{=}1,2)&#36;Yes&#36;f&#36;Two parts.</p>
<p>And...&#36;f&#36;It's two points to decide.</p>
<p>Theoretically: For any poking space&#36;Y&#36;and Map&#36;f:Y{\to}X_1{\times}X_2,f&#36;Continuous&#36;\Longleftrightarrow f&#36; The weights are continuous.</p>
<h4>Totoki.</h4>
<p>The multiplication is generated by a particular sub-clan. This rule is extended to measure space already used.<strong>The spherical neighborhoods of measuring space generate scale-up.</strong> <em>It's the source of measuring space from the point of view of scaling up.</em>It's a general concept abstracted from the above approach.</p>
<p>Summon&#36;X&#36;The sub-clan. &#36;\mathscr{B} \textbf{ 为 集 合 }X\textbf{ 的 拓 扑 基 , 如 果 }&#36; &#36;\overline{\mathscr{B&#125;&#125;&#36; is a puff for X; called puff space (&#36;X, \tau ) \textbf{的 子 集 族 }\mathscr{B}&#36;For this pedestal, if&#36;\overline B=\tau.&#36;</p>
<p><strong>Takuba is thinking about what to use to create.</strong></p>
<p>Theoretically:&#36;\mathscr{B} \textbf{ 是 集 合  }X\textbf{ 的 拓 扑 基 的 充 分 必 要 条 件 是 : }&#36;</p>
<ul>
<li>&#36;\bigcup_{B\in\mathscr{B&#125;&#125;B=X&#36; ;</li>
<li>If &#36;B_1,B_2\in\mathscr{B}&#36;, then &#36;B_1\bigcap B_2\in\overline{\mathscr{B&#125;&#125;(&#36;Which means...&#36;\forall x\in B_1\bigcap B_2&#36;, exists&#36;B\in\mathscr{B}&#36;♪ That makes ♪ &#36;x\in B\subset B_1\bigcap B_2).&#36;</li>
</ul>
<p>Theoretically: &#36;\mathscr{B}&#36; It's open space.&#36;(X,\tau)&#36;It's an essential condition for the foundation.</p>
<ul>
<li>&#36;\mathscr{B}\subset\tau(\text{即 }\mathscr{B}\text{ 的成员是开集)}&#36;</li>
<li>&#36;\tau\subset\overline{\mathscr{B&#125;&#125;(\text{即每个开集都是 }\mathscr{B}\text{ 中一些成员的并集}).&#36;</li>
</ul>
<h2>Twist nature</h2>
<h3>Separating Justice and Countable Justice</h3>
<p>Some of the well-known properties of the Oxygen and Metrics are likely to be lost in general expansion, and separability and numericity are often used as additional properties to compensate for the lack of expanse. So they themselves are called justice. And here's two of these. Two of these. And four of them.</p>
<h4>&#36;T_1,T_2&#36; Justice</h4>
<p>T1 Justice: any two differences &#36;x&#36; and &#36;y&#36;,&#36;x&#36; There's no neighborhood. &#36;y&#36;,&#36;y&#36; There's no neighborhood. &#36;x&#36;I'm not sure.
T2 Justice: any two differences have a border that doesn't intersect</p>
<p>It's very obvious, just satisfy.&#36;T_2&#36; Justice, that's the space to be satisfied.&#36;T_1&#36; Justice. In fact, it's not true.</p>
<p>Totospace.&#36;(R,\tau_f)&#36;  When?&#36;x\neq y&#36; When?&#36;R-{x}&#36; Yeah. &#36;y&#36; And he doesn't.&#36;x&#36; And the other way around, so this space is satisfied.&#36;T_1&#36; Justice, and their neighborhoods must be intertwined, because they're all residuals of a limited set, so they're not satisfied. &#36;T_2&#36; Justice</p>
<p>Actually... &#36;T_1&#36; The more important meaning of justice is:
&#36;&#36;X\text{ 满足 }T_1\text{ 公理}\Longleftrightarrow X\text{ 的有限子集是闭集}.&#36;&#36;
In that sense, we're going into space.&#36;(R,\tau_f)&#36; To understand his meaning better.</p>
<p>&#36;T_2&#36;Justice is the most important reason for separation. Satisfaction.&#36;T_2&#36;The pedestal space of justice is known as Hausdorf space; or the equivalent is defined as:&#36;(X,\tau)&#36;For a pouncer.&#36;X&#36;Any of the two different points in it have an unconnected neighbourhood, which is called Hausdorf space.</p>
<h4>&#36;T_3,T_4&#36; Justice</h4>
<p>&#36;T_3&#36; Justice: any aspect of a neighbourhood that does not contain it.
&#36;T_4&#36; Justice: There is no interconnection between any of the closed collections.</p>
<p>If&#36;X&#36; Satisfied &#36;T_1&#36; Justice, its single-point set is closed. So we can start &#36;T_4&#36; Justice rollout. &#36;T_3&#36; Justice, from&#36;T_3&#36; Justice rollout. &#36;T_4&#36; Justice. Not yet.&#36;T_1&#36; This argument is not valid on the condition of justice. Inverse space. &#36;(R,\tau)(\tau={(-\infty,a)|-\infty\leqslant a\leqslant+\infty})&#36; Satisfied &#36;T_4&#36; But not the first three.</p>
<p><strong>Theorem: measure space&#36;(X,d)&#36; To satisfy the four rules of separation.</strong></p>
<p>All he needs to prove is that he's satisfied.&#36;T_1,T_4&#36; Justice is enough.</p>
<h4>Justice.</h4>
<p>We're here to introduce two principles of justice:&#36;C_1,C_2&#36; Justice. Satisfied&#36;C_i&#36; Justice.&#36;C_i&#36; Space, where&#36;C_2&#36; Space is also called full space.</p>
<p>Neighbourhood-based concept: establishment&#36;x\in X&#36;I'll take it.&#36;x&#36;All the neighborhoods are called&#36;x&#36;The neighborhood system.&#36;\mathcal{N}(x).\mathcal{N}(x)&#36;A subset (i.e.&#36;x&#36;♪ A neighborhood ♪&#36;\mathscr{C}&#36;Called&#36;x&#36;A neighborhood base, if&#36;x&#36;Every neighborhood contains at least&#36;\mathscr{U}&#36;One of the members.<strong>Neighbourhood-based study of the production base of the neighbourhood</strong></p>
<p>&#36;C_1&#36;Justice: There's a lot of neighborhoods in space.
&#36;C_2&#36;Justice: a space for poking&#36;X&#36; There's more to go.</p>
<p>&#36;C_2&#36; Justice is a very strong nature, and partly measured space is not enough.&#36;C_2&#36; Righteous, like the classic dispersive measure space.</p>
<p>Satisfied&#36;C_2&#36;Justice must be satisfied.&#36;C_1&#36;Justice, this can be derived from the definition of nature, and we have a lot of pedestals that we can naturally use to launch the Neighbourhood Base.</p>
<p>&#36;C_2&#36; Space's a fractional space, when there's a few of them.&#36;{B_n}&#36; From now on, just...&#36;{B_n}&#36;Middle Pick Point&#36;x_n&#36; You can get a dense number of subsets, which in turn doesn't necessarily divide into space.&#36;C_2&#36; Space</p>
<p>The measure space is... &#36;C_2&#36;  Space. Reference<a href="/en/blog/2023/09/11/functional-analysis-notes/">Separability of measure space in general correspondence analysis</a></p>
<h4>Genetic and multiplierial expansion</h4>
<p>A pedestal is called genetic, and if a pedestal has it, the subspace must have it; a pedestal is called multiplier, and if both spaces have it, their product space also has it.</p>
<ul>
<li>Separability is acceptable, but not genetic.</li>
<li>In separability,&#36;T_1,T_2,T_3&#36;It's just that there is geneticity and complication.&#36;T_4&#36; Neither.</li>
<li>There's geneticity and complication on both counts.</li>
</ul>
<h3>Tightness</h3>
<p>Here we respond to the measurements. <a href="/en/blog/2023/09/11/functional-analysis-notes/">Tightness of measure space in general correspondence analysis</a> <a href="/en/blog/2023/09/11/functional-analysis-notes/">Open-covered measure space in general correspondence analysis</a></p>
<h4>Tightness and Tightness Measurement Space</h4>
<p>We follow the strict definition we once gave:<strong>The pedestal space is called tight if each of its sequences has a condensed subsequence (i.e., a limit point).</strong></p>
<p>In fact, studying sequences in measuring space is not a good idea, so we define tightness based on an open-cover approach:<strong>It's called a tight space, if each of its open covers has a limited subcover.</strong></p>
<p>On the surface, there does not seem to be a direct relationship between the tightness and the tightness, which is, in essence, closely related ... For measuring space, the two characteristics are equivalent (proven in a general analysis). They're not equal in general space, we're talking about tight concepts.</p>
<p><strong>Theoretically:&#36;\text{若 }X\text{ 是度量空间,则 }X\text{ 列紧}\Longleftrightarrow X\text{ 紧致}&#36;</strong></p>
<h4>Nature of compact space</h4>
<p>A poking space.&#36;X&#36;Subsets&#36;A&#36;If the subspace is tight, it's called &#36;X&#36; Yes.<strong>Tight subset</strong></p>
<p>Theoretically:&#36;A\text{是 }X\text{ 的紧致子集 }\Longleftrightarrow A\text{ 在}X\text{ 中的任意开覆盖有有限子覆盖}&#36;</p>
<p>We've seen similar patterns in general analysis.</p>
<ul>
<li>The closes of space are tight.</li>
<li>It's also very tight in space.</li>
<li>Defines a continuous function in a tight space, with a maximum and minimum value.</li>
</ul>
<h4>Tightness of the product space</h4>
<p>It's easy to see that a tight space does not have genetic properties, such as closed compartments. &#36;[a,b]&#36; Tight, it's a subset. &#36;(a,b)&#36;It doesn't matter. But...<strong>Tightness is multiplierable if &#36;X&#36; and&#36;Y&#36; All tight, then. &#36;X\times Y&#36; And tight.</strong></p>
<h3>Connectivity</h3>
<h4>Connectivity Introduction</h4>
<p>The normal geometry of "connectivity" is a very intuitive concept that hardly requires a mathematical definition. For example, everyone knows that in cone curves, ellipses and parabolic lines are connected, and the hyperbolic curves are not. However, for thousands of complex graphics, intuitiveness alone, for example.</p>
<p>Set &#36;E^2&#36;It's a sub-set. &#36;X&#36; By &#36;A&#36; and &#36;B&#36; Two parts.
&#36;&#36;A={(x,\sin\frac1x)\left|x\in(0,1)},\right.\B={(0,y)|-1\leqslant y\leqslant1}.&#36;&#36;
It's hard to judge by the concept of visual. &#36;X&#36; Is it connected? So we need a more abstract definition of connectivity.</p>
<p>Intuitive connectivity can mean two things:</p>
<ul>
<li>Graphics can't be divided into two parts of each other's "adhesive".</li>
<li>It's any two points on the graphic that can be connected to the graphic.</li>
</ul>
<p>In Topography, the two concepts are abstracted as “connectivity” and “road connectivity”, respectively. In fact, the examples given above are connected, but not road connectivity.</p>
<h4>Definition of connectivity</h4>
<p>Definition: Popping Space &#36;X&#36; It's called the connecting space, if it can't be broken down into two non-empty interlocking and</p>
<p>Apparently, the following description can be defined as its equivalent.</p>
<ul>
<li>&#36;X&#36; Can't be broken down into two non-empty interlocking and</li>
<li>&#36;X&#36; There is no open but closed collection of non-reals.</li>
<li>&#36;X&#36; There's an open and closed subset. &#36;X,\phi&#36;</li>
</ul>
<p>&#36;(R,\tau_f)&#36;It is connected because any two of its non-empty openings necessarily intersect;&#36;(R,\tau_c)&#36; And it's connected. The hyperbolic curve is unconnected. It's two separate, non-closed sets.</p>
<p>However, many intuitively connected spaces, judged by the definition above, are not immediately able to conclude, for example, that of parabolic lines, elliptical connectivity. We often argue for connectivity from some known connected spaces.  &#36;E^1&#36; Connectivity is our starting point.</p>
<h4>Nature of connectivity space</h4>
<p>Theoretically: Connecting space is also connected to continuous mapping</p>
<p><strong>We use this theory to study space connectivity.</strong> Like respace.&#36;S^1&#36; Yes.&#36;f(x)=\mathrm{e}^{\mathrm{i}2\pi x}&#36;From&#36;E^1&#36; Got it.</p>
<p>Definitions:&#36;E^1&#36; It's called a sub-set.</p>
<p>Theoretically:&#36;设A\subset E^1,则A连通\Longleftrightarrow A是区间&#36;</p>
<p>Theoretically: Continuous functions in connected spaces<em>Function Map to&#36;E^1&#36;</em>Take all the medians.</p>
<p>Theorem (connective coverage): if&#36;X&#36;There's a connection.&#36;\mathscr{U}(\mathscr{U}&#36;Each member is connected, and&#36;X&#36;There's a link.&#36;A&#36;And it's with&#36;\mathscr{U}&#36;Each of the members shall meet, then&#36;X&#36;Connect.
<strong>It's a common theory of connectivity.</strong></p>
<p><strong>Theorem: Connectivity is acceptable</strong></p>
<h4>Connect branch</h4>
<p>The connectivity branch is a concept that emerges from the study of unconnected spaces.</p>
<p>Definition: Popup Space &#36;X&#36; A subset called &#36;X&#36; A branch of connectivity if it's connected and not &#36;X&#36; The rest of the links are real.
<strong>The branch of connectivity is a huge subset of connectivity.</strong></p>
<p>When? &#36;X&#36; When it's connected, it has only one branch. &#36;X&#36; Self.</p>
<p>Theorem:&#36;X&#36; Each non-empty link subset is contained in the only link branch. Medium</p>
<h3>Road connectivity</h3>
<p>Road connectivity is another leapfrogging concept based on visual connectivity. For it, the road is the key concept.</p>
<h4>Road</h4>
<p>The road concept is the abstraction of the intuitive concept of curves... curves can be seen as the trajectory of point movements. If you count the beginning and end of the movement as 0 and 1, the movement is a closed space. &#36;[0,1]&#36; It's a continuous map to space, and the curve is the image of it.</p>
<p>Definitions: Establishment &#36;X&#36; It's a pouncer. From the unit compartment. &#36;I=[0,1]&#36; Present. &#36;X&#36; A continuous map &#36;a : I\to X&#36; Called &#36;X&#36; Up the road. &#36;a(0)&#36; and &#36;a(1)&#36;Separately known as &#36;a&#36; The beginning and the end, collectively known as the end.</p>
<p>Roads are the mapping itself, not its collections. In fact, there may be many different paths, and they are identical. It's hard to get a map out, to represent it with a collection of images, and to draw an arrow to show the direction of movement.</p>
<p>♪ If the road ♪ &#36;a:I\to X&#36;is a constant map, i.e. &#36;a(I)&#36;One thing, it's called a road point. The road is completely like a spot. &#36;x&#36; Decision</p>
<p>The road from the beginning and the end is called closed.</p>
<p>Definition: A road&#36;a: I\to X&#36;And vice versa.&#36;X&#36;♪ Up the road, recorded ♪&#36;\bar{a}&#36;, provided that &#36;\bar{a}(t)=a(1-t),\forall t\in I&#36;  The reverse of the road is the study of the path to the beginning and end.</p>
<p>Definitions:&#36;X&#36;Two roads up.&#36;a&#36;and&#36;b&#36;If satisfied&#36;a(1)=b(0)&#36;, they can be specified
Product&#36;ab&#36;So is he.&#36;X&#36;The way up, it's called
US&#36;ab(t)=\begin{cases}a(t),&amp;0\leqslant t\leqslant1/2,\b(2t-1),&amp;1/2\leqslant t\leqslant1.&amp;\end{cases}&#36;&#36;</p>
<p>The following is a list of the characteristics of the inverse and product.</p>
<ul>
<li>&#36;\bar{e}_x=e_x&#36;</li>
<li>&#36;\overline{(\bar{a})}=a&#36;</li>
<li>&#36;\text{当 }ab\text{ 有意义时,}\overline{b}\bar{a}\text{ 有意义,且 }\overline{b}\bar{a}\boldsymbol{=}\overline{ab}.&#36;</li>
</ul>
<p><strong>The road concept is useful not only in defining road connectivity, but also as an important basic concept in algebra development and as a basis for building basic clusters.</strong></p>
<h4>Road connectivity space</h4>
<p>Definition: Popup Space &#36;X&#36; It's called a road connection, if&#36;\forall x,y\in X&#36;, exists&#36;X&#36;by&#36;x&#36;and &#36;y&#36; The road for starting and ending.</p>
<p>This definition is very natural.</p>
<p><strong>Theoretically: Road connectivity space must be connected</strong></p>
<p>We need to come up with a few simple proof lines: road connectivity ensures that the entire satellite subset of space is on one branch of connectivity. Here. &#36;X&#36; There's only one connected branch, which is... &#36;X&#36; Connect</p>
<p>Road connectivity space also has some nature of connectivity space</p>
<ul>
<li>The continuum of road connectivity is like road connectivity.</li>
<li>Road connectivity is also viable.</li>
</ul>
<h4>Road connectivity branch</h4>
<p>It's in the open space. &#36;X&#36; It's a relationship between its points.&#36;\sim&#36;Other Organiser &#36;x&#36; and&#36;y&#36; Available &#36;X&#36; And the road is connected. &#36;x&#36; and &#36;y&#36; Related, recorded &#36;x\sim y&#36;It's an equal value.</p>
<p>Definition: Pocketspace in Equivalent Relationship&#36;\sim&#36;The lower class of equivalent is called&#36;X&#36;Road connectivity branch, abbreviated as road branch</p>
<p>As with the non-connected space of the connective branch of the road, the unconnected space of the road, we can naturally give the following proposition:</p>
<ul>
<li>Any&#36;x\in X&#36;belong&#36;X&#36;The only branch of the road.</li>
<li>&#36;X&#36; A subset of each road connected is contained in a branch of the road. Medium</li>
<li>&#36;X&#36;The necessary condition for road connectivity is that it has only one branch.</li>
<li>The branch of the road that expands space is its great link.</li>
</ul>
<p>&#36;X&#36;Every branch of the road is connected, so it has to be included in one of the branches, so... <strong>&#36;X&#36; Each branch of the connection is a combination of some of the roads.</strong></p>
<p><strong>This nature allows us to combine the concepts of road connectivity and connectivity.</strong></p>
<h2>Commercial space and closed face</h2>
<p>In this chapter, we're going to discuss a special kind of poking space: the closed face, which is one of the most important subjects of research in puffing (especially algebra and low-dimensional puffing).</p>
<p>The concept of commercial space offers a way to construct a new space from the existing space, which is very useful in algebracology and will also be the main method used in this chapter in its study of the closed face, which is also the core of the contours of telemetry and algebracology. Section</p>
<h3>A few common sides.</h3>
<p>In the curve, except for the plane. &#36;E^2&#36; And the ball. &#36;S^2&#36;In addition, the most common are the ring, the Mobius belt, the ring, the Klein bottle and the film plane. They can all be adhesived with rectangular pieces.</p>
<h4>Searing and Mobius Belt</h4>
<p>Bend the rectangular face and bind the sides to a column. It's a ring of two concentric circles on the same plane, so it's called<strong>Halo</strong>...a generic description of the space in the price-equisition category, whether the space is indeed a ring, or a cylindrical or other shape, as long as it belongs to the price-equisition category.</p>
<p>When you make a flat ring, just bend the rectangle and don't twist it, so that the point on both sides of the rectangle joins the height. If you twist the rectangular 180 degrees and then bind the sides, the space is famous.<strong>Mobius Belt</strong></p>
<p>Intuitively, there are many differences between the Mobius belt and the ring.</p>
<ul>
<li>The boundary of the ring is two closed curves, and the boundary of the Mobius belt is a closed curve.</li>
<li>The ring is on both sides, Mobius. The belt is on one side.</li>
<li>The middle line of the ring splits it into two rings, while the middle line along the Mobius is either a tangled curve, which is actually a flat ring after cutting.</li>
</ul>
<h4>Ring and Klein</h4>
<p>The rings and the Klein bottles can all be held together by a cylinder.</p>
<p>If the two ends of each end of the end are glued, what you get is...<strong>Ring</strong>At this point, the two intercepts are connected in the same direction. Like peace rings, the ring and other curves are in the same range.
General</p>
<p>If you have two intercepts in the opposite direction, you get them. <strong>Klein</strong>I don't know. To achieve this bond, the cylindrical face must be bent and one end crossed through the wall into the other. This is not possible in 3 dimensions, because there's bound to be an intersection inside the tube. But it's possible in 4 dimensions.</p>
<p>The Klein bottle is also one-sided, and he's different from this side of the circle.</p>
<h4>Insight plane</h4>
<p>Insight plane&#36;P^2&#36;It's a concept in geometry.</p>
<p>We're using a disk at the Topup.&#36;D^2&#36;To construct him. We take the border.&#36;S^1&#36; Each pair of contrast points (two end points on the same diameter) adhesives and gets a projection plane.</p>
<p>It's hard to understand intuitively. The elephant.</p>
<p>It's a common way to create a new poking space with glue. It's hard to understand intuitively that we need a whole new mathematical tool to help us deal with this problem, whether intuitively or not, we need a language to describe the space that binds and gets.</p>
<h3>Business space</h3>
<h4>Business space</h4>
<p>Set up space. &#36;X&#36; Some kind of adhesive to get new space. If you call the point that you're going to stick together as the point that you're going to have, &#36;X&#36; There is an equal value relationship, each of which is bound to a point in the new space.</p>
<p>So the collection of new spaces is the collection of equal value categories, in general, a collection. &#36;X&#36; If there's an equal value&#36;\sim&#36;The corresponding groups of equivalents are recorded &#36;X/\sim&#36;Called &#36;X&#36; About &#36;\sim&#36; Yes.<strong>Business</strong></p>
<p>Put &#36;X&#36; The point above corresponds to its equivalent and is mapped&#36;p:X\to X/\sim&#36; Called<strong>Sticky Map</strong> Set &#36;X&#36; We've already done it. We'll set it up. &#36;X/\sim&#36; The one up there.</p>
<p>Definitions: Yes &#36;( X, \tau )&#36;It's popping space.&#36;\sim&#36;It's a rally. &#36;X&#36; An Equivalent Relationship Above. &#36;X/\sim&#36; The upper sub-clan.
&#36;&#36;\widetilde{\tau}:=\left{V \subset X / \sim \mid p^{-1}(V) \in \tau\right}&#36;&#36;
We can prove it's a business. &#36;X/\sim&#36; The one up there called&#36;\tau&#36; In Equivalent Relationship &#36;\sim&#36; Down<strong>Taku Po.</strong>And put&#36;(X/\sim,\widetilde{\tau})&#36; It's called the original piston space. &#36;\sim&#36;Down<strong>Business space</strong></p>
<p>Theorem: Set&#36;X,Y&#36; It's two popping spaces.&#36;\sim&#36; It's one.&#36;X&#36; It's an equal value.&#36;g:X / \sim \to Y&#36;  It's a map of a commercial space to a general popping space. then &#36;g \text{ 连续}\Longleftrightarrow gp\text{连续}&#36;
<strong>It's natural that we've got two continuous maps combined.&#36;p&#36;It's a continuous map.</strong></p>
<p><strong>The concept of commercial space can bind abstract research.</strong></p>
<p>We give an example of a common space.</p>
<p>Set &#36;A&#36; It's open space. &#36;X&#36; A subset (usually closed subset), &#36;A&#36; A little. &#36;A&#36; As an equivalent category, the other points are classified as equivalent) and are recorded in commercial space &#36;X/A&#36;</p>
<p>♪ On any pouncer ♪ &#36;X&#36; Remember &#36;CX = X\times I / X\times {1}&#36;Called &#36;X&#36; Top<strong>Tink.</strong></p>
<h4>Business Map</h4>
<p>Business mapping and business space are closely related concepts. They look at the same thing differently. The commercial map is viewed from a mapping perspective, which is more conducive to understanding.</p>
<p>Definitions: Establishment &#36;X&#36; and &#36;Y&#36; It's pouncer space, map. &#36;f:X\to Y&#36; Called commercial map, if</p>
<ol>
<li>&#36;f&#36; Continuous</li>
<li>&#36;f&#36; It's full.</li>
<li>&#36;\text{设 }B\subset\text{Y,如果 }f^{-1}(B)\text{是 }X\text{ 的开集,则 }B\text{ 是}Y\text{ 的开集}&#36;
<strong>One, three.</strong></li>
</ol>
<p>At this point, we can rewrite the theory given in the section along the lines of a commercial map.
Theoretically: If&#36;f:X{\to}X^\prime&#36;It's a commercial map.&#36;g:X^\prime{\to}Y&#36;It's a map, then.&#36;g&#36; Continuous&#36;\Longleftrightarrow g&#36;&#36;f&#36; Continuous</p>
<p><strong>From this theory, we can see that commercial mapping is actually a sticky map.&#36;p&#36;They're doing the same thing.</strong></p>
<p>There's a test for commercial mapping.</p>
<ul>
<li>A continuous full map &#36;f:X\to Y&#36; If it's an open or closed map, it's a commercial map.</li>
<li>If &#36;X&#36; Tight. &#36;Y&#36;It's Hausdorf space, full of maps. &#36;f:X\to Y&#36; It's a commercial map.</li>
<li>Commercial map compound is also commercial map</li>
</ul>
<h3>Twist and Curve</h3>
<h4>Fluid</h4>
<p>The sphere, the ring and the other curves we know are much more complex than the plane in general, but...<strong>On the local level, each of them has an area near the plane.</strong>It makes it possible to use analytical tools to study it on a local scale.</p>
<p>It's not that bad.<strong>It's called "floating."</strong>I don't know. Flow formation is a more complex concept, and in different fields of study it is also required to have a variety of special structures.</p>
<p><strong>Definition: A Hausdorf space &#36;X&#36; Called &#36;n&#36; Vitto-flow, if &#36;X&#36; There's one in every one of them. &#36;E^n&#36; or &#36;E^n_+&#36; It's the open neighborhood.</strong></p>
<p>Here.&#36;E_+^n&#36;Half.&#36;n&#36;Violin Space.
&#36;&#36;E_+^n:={(x_1,x_2,\cdotp\cdotp\cdotp,x_n)\in E^n|x_n\geqslant0}.&#36;&#36;</p>
<p>Based on the above definition, it's easy to know.&#36;E^n,S^n,T^n&#36; Both.&#36;n&#36; It's a velvet, but it's not a cone, because it's not possible to find an open border in the same embryonic space.</p>
<p>Set&#36;M&#36;Yes.&#36;n&#36;Flow point &#36;x\in M&#36;If there's a congener&#36;E^n&#36;It's a neighborhood.&#36;x&#36;Yes.&#36;M&#36;Yes.<strong>Inside point</strong>(Note that this concept is distinct from the concept given in chapter I of the sub-set)<strong>Boundary Point</strong>The assembly of all the inner points&#36;M&#36;Inside, it is.&#36;M&#36; An opening episode.</p>
<p>To be clear, we must also acknowledge some facts.&#36;n\neq m&#36;Time&#36;E^n\not\cong E^m(&#36;If you don't, the flow of dimensions will be meaningless.&#36;E_+^n\not\cong E^n(&#36;Otherwise there will be no distinction between the inner and the boundary points)</p>
<p>If &#36;n&#36; The dimensional flow has a boundary point. &#36;\partial M&#36; It's the assembly of its boundary points. &#36;\partial M&#36;It's a non-boundary point.&#36;(n — 1)&#36;Toggle</p>
<h4>Closed face</h4>
<p>It's called a warp. &#36;E^n,S^n,T^n&#36; The rings and the Mobius bands are curved. The first three have no borders. Points</p>
<p>Definitions: Closely connected curves without boundary points are called closed Noodles.</p>
<p>There's only five sides of the line.&#36;S^n,T^n&#36; It's closed. It's insulated.&#36;P^2&#36;It's a closed face, Klein. The bottle is closed. Noodles.</p>
