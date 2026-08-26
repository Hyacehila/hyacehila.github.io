---
title: 'Affine Geometry: Affine Spaces, Coordinate Transformations, and Geometric Quantities'
title_zh: 仿射空间几何学：仿射空间、坐标变换与几何量
date: 2025-02-12 18:22:33 +0800
categories:
- Mathematics
- Geometry & Topology
tags:
- Affine Geometry
author: Hyacehila
mathjax: true
hidden: true
excerpt: Covers affine spaces, coordinate transformations, geometric quantities, vector fields, derivatives, curves, and surfaces.
description: Covers affine spaces, coordinate transformations, geometric quantities, vector fields, derivatives, curves, and
  surfaces.
excerpt_zh: 整理仿射空间、坐标变换、几何量、向量场、导数、曲线与曲面等内容。
permalink: /blog/2025/02/12/affine-geometry-notes/
lang: en
translation_key: 2025-02-12-affine-geometry-notes
translation_status: machine
translation_source_hash: ecaee330f022c8564e647f7a6bffea6df8781c7252070de3e21738080b6abf03
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<p>This note will be taken as<a href="/en/blog/2025/02/04/differential-geometry-notes/">Micro geometry</a>Part of this is taken out separately from the study because of its relative independence and the European space discussions that we are primarily exploring.</p>
<h2>Simulating the geometry base of space</h2>
<p>At the beginning of this chapter, we will move away from the European-style concept of space dependence and length and focus on a concept that relies solely on linear spatial structures, for which we will introduce a new language that will be closely linked to the current introduction.</p>
<h3>Lines, plane, coordinates and coordinates of the mimic space</h3>
<h4>The imitation space and the line and plane above it.</h4>
<p>Geometry is the discipline of the research point set, and we need to first construct a point of space; ordinary vector space cannot be directly a geometrically sub-point because he has a special zero, and geometrically-intended points have the same status at each point.</p>
<p>So we need a better space. His idea is...<strong>Remove vector space centralization</strong></p>
<p>Vectors are derived from the location vector in physics, which is defined as a quantity with both size and direction, but his size and direction are considered to have a specified zero-point effect, which is natural.<strong>The difference in position vector is not related to the choice of zero points</strong> This idea has led us to the idea of a simulation space.</p>
<p>Definitions: Establishment&#36;\mathscr{A}^{n}&#36;It's a non-empty collection, if there's a map.
&#36;&#36;A^{n}\times A^{n}\mapsto\mathbb{R}^{n},\left(A,B\right)\mapsto\overrightarrow{AB}&#36;&#36;
Satisfied</p>
<ul>
<li>&#36;\forall\nu\in\mathbb{R}^{n},A\in\mathscr{A}^{n},\text{存在唯一的}B\in\mathscr{A}^{n}\text{使得}\overrightarrow{AB}=\nu&#36;</li>
<li>&#36;\overrightarrow{AB}+\overrightarrow{BC}=\overrightarrow{AC}.&#36;
Name&#36;\mathscr{A}^{n}&#36;Yes.&#36;n&#36;Simulation Space</li>
</ul>
<p><strong>This definition is based entirely on the definition of the position vector given above, which guarantees certainty and leaves aside the zero option, but has not been coordinated.</strong></p>
<p>Definition: 3D analogue space &#36;\mathscr{A} ^{3}&#36;Non-empty subsets&#36;l&#36;, with points&#36;{A} \in l&#36;, if satisfied &#36;\exists \nu \in \mathbb{R} ^{3}, \nu \neq 0&#36;♪ That makes ♪
&#36;&#36;l=\left{B\in A^{3}|\overrightarrow{AB}=k\nu,k\in\mathbb{R}\right},&#36;&#36;
Name&#36;l&#36;For the past&#36;A&#36; Here.&#36;\nu&#36; To the straight line in the direction,&#36;\nu&#36;Called&#36;l&#36; - The directional amount.</p>
<p>Definition: 3D analogue space &#36;\mathscr{A}^3&#36;Non-empty subsets &#36;\pi&#36;,In &#36;A\in\pi&#36;, meet:&#36;\exists\nu,w\in\mathbb{R}^3&#36;It's not linear.&#36;&#36;\pi=\left{B\in A^{3}|\overrightarrow{AB}=kv+pw,\left(k,p\right)\in\mathbb{R}^{2}\right},&#36;&#36;Name&#36;\pi&#36;For one&#36;\nu&#36; 、&#36;w&#36;Chang Sung passed.&#36;A&#36;Plane, (&#36;\nu,w)&#36;Called&#36;\pi&#36;The direction is right.</p>
<p><strong>With a basic definition of geometry, we can easily prove that the former use of a reasonable geometry system in geometry on the plane proves the proposition that the concept of angle and length is not involved, that is, the nature of the imitation is legitimable.</strong></p>
<h4>Coordinates of the simulation space</h4>
<p>We need to introduce reference points so that the actual numbers can be used to describe the nature of the simulation space.</p>
<p>Definition (simulation coordinates): in &#36;x^n&#36;Select a point on top &#36;O&#36;(known as original)&#36;R^n&#36;The selected base group on the top is &#36;\left{e i}\right} <em>{( i= 1, 2, \cdots , n) },&#36;  我们记&#36;A= \left { O, e</em>{i}\right }&#36;为 &#36;x^n}&#36; up a analogue coordinate system with the following double-jet:
&#36;&#36;\varphi_{A}=A^{n}\mapsto\mathbb{R}^{n}&#36;&#36;
&#36;&#36;A\mapsto\left(x^{1},x^{2},\cdots,x^{n}\right)&#36;&#36;Here.&#36;\overrightarrow{OA}=\sum_{i=1}^{n}x^{i}e_{i}.&#36;</p>
<p>Consider two analogue coordinates.  &#36;\left { O, e_{i}\right },\left{O^{\prime},f_{i}\right}&#36;  They're satisfied.
&#36;&#36;\overrightarrow{OO^{\prime&#125;&#125;=\sum_{i=1}^{n}a^{i}e_{i},e_{i}=\sum_{i=1}^{n}T_{i}^{i}f_{i}&#36;&#36;
That's right.
&#36;&#36; (e) = (f) \begin{matrix}T 1^1&amp;T_2^1&amp;\cdots&amp;T_n^1\T_1^2&amp;T_2^2&amp;\cdots&amp;T_n^2\\vdots&amp;\vdots&amp;&amp;\vdots\T_1^n&amp;T_2^n&amp;\cdots&amp;T_n^n\end{pmatrix}.&#36;&#36;
则有仿射坐标变换（原本的坐标为&#36;x_i&#36;，则新坐标系下的坐标&#36;y_i&#36;的形式）为
&#36;&#36;y^{i}=\sum_{j=1}^{n}T_{j}^{i}\left(x^{j}-a^{j}\right).&#36;&#36;</p>
<p>Definition (simulation coordinates are directed): &#36;det (T) matrix when the imitation coordinates are converted&gt;0&#36;时，称为两个坐标系同向，当&#36;det(T)&lt;&#36; 0, referred to as the two coordinates are contrary to orientation.</p>
<h4>Coordinates of Lines and Flats</h4>
<p>To reduce the use of the sum of the symbols, we agreed to the symbols below.
&#36;&#36;a^{i}e_{i}=\sum_{i=1}^{n}a^{i}e_{i}&#36;&#36;</p>
<p>For the coordinates of the line, we can easily know the line after the coordinates are set. Points&#36;A&#36;Satisfied
&#36;&#36;\overrightarrow{OA}=a^{1}e_{1}+a^{2}e_{2}+a^{3}e_{3}=a^{i}e_{i}&#36;&#36;
And the straight line itself is easily decomposed to
&#36;&#36;v=v^{1}e_{1}+v^{2}e_{2}+v^{3}e_{3}=v^{i}e_{i}&#36;&#36;
So for any point on the line, that's the whole line.
&#36;&#36;\overrightarrow{OX}=\overrightarrow{OA}+\overrightarrow{AX}&#36;&#36;
So there is.
&#36;&#36;x^{i}e_{i}=a^{i}e_{i}+kv^{i}e_{i}\Rightarrow x^{i}=a^{i}+kv^{i},k\in\mathbb{R}&#36;&#36;
Same thing. For the plane.
&#36;&#36;\overrightarrow{OX}=\overrightarrow{OA}+\overrightarrow{AX}&#36;&#36;
That's...
&#36;&#36;x^{i}e_{i}=a^{i}e_{i}+kv^{i}e_{i}+pwe_{i}\Rightarrow x^{i}=a^{i}+kv^{i}+pw^{i},k,p\in\mathbb{R}&#36;&#36;</p>
<h3>Geometric</h3>
<p>Before we go any further, let's just discuss the concept of geometry.</p>
<p>Geometric amounts (geometric amounts) should be simply points of dependence or geometry objects, otherwise they cannot be the subject of geometry research. However, when defining or calculating these amounts, we inevitably do so by the coordinates of the point in a specific coordinate system ... The question arises as to whether a given number of coordinates is defined and how to determine whether it is not a geometric amount.</p>
<p>Consider a 3D analogue space. &#36;\mathscr{A}^3&#36;It's got a coordinate system. &#36;\mathscr{A}=\left{O,e_1,e_2,e_3\right}.&#36;And we're introducing a straight line, an equation.
&#36;&#36;\left(x^{1},x^{2},x^{3}\right)=\left(x_{0}^{1},x_{0}^{2},x_{0}^{3}\right)+t\left(v^{1},v^{2},v^{3}\right),&#36;&#36;
Here.&#36;(v^1,v^2,v^3)&#36;It's a given number.
&#36;&#36;(x_{1}^{1},x_{1}^{2},x_{1}^{3})=(x_{0}^{1},x_{0}^{2},x_{0}^{3})+t_{1}(v^{1},v^{2},v^{3}),&#36;&#36;
&#36;&#36;(x_{2}^{1},x_{2}^{2},x_{2}^{3})=(x_{0}^{1},x_{0}^{2},x_{0}^{3})+t_{2}(v^{1},v^{2},v^{3})&#36;&#36;
Consider two real numbers defined by coordinates
&#36;&#36;\eta:=\sqrt{\sum_{i=1}^{3}\mid x_{1}^{i}-x_{0}^{i}\mid^{2&#125;&#125;=t_{1}\parallel v\parallel=t_{1}\sqrt{\sum_{i=1}^{3}\mid v^{i}\mid^{2&#125;&#125;&#36;&#36;
&#36;&#36;\lambda:=\sqrt{\frac{\sum_{i=1}^{3}\mid x_{2}^{i}-x_{1}^{i}\mid^{2&#125;&#125;{\sum_{i=1}^{3}\mid x_{1}^{i}-x_{0}^{i}\mid^{2&#125;&#125;}=\frac{t_{2}-t_{1&#125;&#125;{t_{1&#125;&#125;.&#36;&#36;
We can prove that.&#36;\eta&#36;Not to imitate geometry in geometry, but...&#36;\lambda&#36;Yes, because the former changes under the analogue coordinates. The blogger adds: <strong>The geometrics of the imitation coordinates remain unchanged for the transformation of the imitation coordinates, i.e., not dependent on the selection of the imitation coordinates.</strong></p>
<h3>The simulation of the space pouncer and the actual amount of the target Field</h3>
<p>In this section we further give a common geometric/physical object called the actual range ... Such quantities have a rich physical background: temperature field, electric force field, density field... Their characteristics are,<strong>Each point in space is given a real value. This "point to number" map is called a real measure field or function.</strong></p>
<p>We will then look at the more complex mapping of "point to vector space" "point to dimension space" that corresponds to the more complex vector field section of "Reflective function of the mirror space and vector field" and the "reflective field" section of the stretch field.</p>
<p>To accurately describe the regularity of the actual field (continuous, micro and smooth), we will first discuss the expansion of the space of the mimic. We will see that the expansion in the analogue space, although defined by the analogue coordinate system, is identical in the sense that it is not dependent on the choice of the specific imitation coordinate system and is thus a mimic of geometric objects.</p>
<p>In order to clear up a few easily confused concepts,<strong>We don't plan for it to be the usual. &#36;A^n&#36;Same&#36;\mathbb{R}^n&#36;Equal</strong>It could cause some writing trouble. But we do it to emphasize that the concept of geometry must be independent of the coordinates. But many concepts are defined by the coordinates.<strong>Distinction &#36;A^n&#36;and&#36;\mathbb{R}^n&#36;It is to distinguish between the concepts of “geometric” and “formula of calculation of geometrics under the coordinates”.</strong></p>
<h4>Imitation of space booming</h4>
<p>We're introducing the concept of poking into the simulation space in this section.<a href="/en/blog/2024/10/16/point-set-topology-notes/">I'll do some tweezing.</a>Foundation</p>
<p>Definitions (): &#36;\mathcal{A}=\left{O,e_i\right}&#36;Yes &#36;n&#36; Simulation Space &#36;\mathcal{A}^n&#36;A simulation coordinates system. A collection.&#36;U\subset\mathcal{A}^n&#36;Call it the opening, if it's at the coordinates. &#36;\varphi_{A}&#36;It's like...&#36;\mathbb{R}^n&#36;Up the beginning.</p>
<p><strong>This is the basic structure of research into pomposity, the opening of space; we have the idea of starting from the definition of the analog system of analogue coordinates.</strong></p>
<p>This definition is not reasonable, because the definition of the opening is based on the selection of the imitation coordinates, which we gave without proof.<strong>The opening definition keeps the capobus and the capophones are the only answer to the question.</strong>So the simulation space is the pouncer space.</p>
<p>We'll have a lot of more to discuss on the coordinates, and then we'll expand to understand that he's still unchanged from the coordinates.</p>
<h4>Function and Scale on Simulating Space Field</h4>
<p>Definitions: &#36;n&#36; Simulation Space &#36;\mathcal{A}^n&#36; The connection is open to the area.</p>
<p>Definitions: Establishment&#36;U&#36;Yes. &#36;\mathcal{A}^n&#36; Up the open area, one from&#36;U&#36;Present.&#36;R&#36;Maps are called functions or actual quantities Field</p>
<p>Definition (functions from imitation coordinates): Set&#36;U&#36;Yes. &#36;\mathcal{A}^n&#36;A region of thought.&#36;U&#36; Functions on &#36;f&#36;, "Defined"&#36;\mathbb{R}&#36;Open the area. &#36;n&#36; Dollars \\c \varphi <em>{\lambda }^{- 1}&#36; 为 从 坐 标 系 &#36;\mathcal{A} = { O,e_i}&#36;中 读 取 &#36;f.&#36;
&#36;&#36;V\xrightarrow{\varphi</em>{A}^{-1&#125;&#125;U\xrightarrow{f}\mathbb{R};~~~~(x^{1},x^{2},\cdots,x^{n})\longmapsto A\longmapsto f\left(A\right).&#36;&#36;
<strong>So we'll use the coordinates in the coordinate space to reverse the map to the pounce space, and then we'll get the amount of the function map from the piston space, which is the range, from the piston space to the map of a scale, and we'll assign a scale to each point of space.</strong></p>
<p>Theorem (decision on continuous functions).&#36;U&#36; Yes. &#36;\mathscr{A}^n&#36;The area above is open.&#36;f&#36; Yes&#36;U&#36;function,
The following three points are equivalent:</p>
<ul>
<li>&#36;f&#36;(a) Continuous;</li>
<li>In a model system. &#36;\mathcal{A}=\left{O,e_i\right}&#36;Medium Read &#36;f&#36; Get one.&#36;n&#36; (a) A continuous function of a meta;</li>
<li>Read in any type of analogue coordinate system&#36;f&#36;I got one.&#36;n&#36;Meta-Consequence Functions</li>
</ul>
<p>Definitions: Establishment&#36;U&#36; Yes. &#36;\mathscr{A}^n&#36;The area above is open.&#36;f&#36; Yes&#36;U&#36; function.&#36;f&#36; It's called micro-synthetic, if there's a analogue system. &#36;\mathcal{A}=\left{O,\boldsymbol{e}_i\right}&#36;to read from this coordinate system &#36;f&#36; It's one.&#36;n&#36; Moto-micro functions.</p>
<p>A fine definition can naturally extend to the change in coordinates, and it's easy to prove.</p>
<p><strong>And finally, we come to conclusions,&#36;n&#36;In the space of the mimic, if there's a mimic coordinate, Yes&#36;A&#36;The following discussion clarifies the nature of the function, so that under any other coordinate system, the nature is the same.</strong></p>
<h3>Broad Coordinate System</h3>
<p>The coordinates are the process of naming points in space, and the two-shot (twice-by-point) requirement of our defined analog system of analogue coordinates, and certain similar and coordinate concepts do not meet this requirement, such as the usual polar-coordinate system, so we here expand the concept of the extended coordinate system and have a simple discussion.</p>
<p>Definitions: Establishment&#36;U&#36; Yes. &#36;\mathscr{A}^n&#36;Up the open area.<em>i=1,2,...,n&#36;是定义在&#36;U&#36; 上的一族&#36;- I'm sorry.
function, satisfy:
&#36; \varphi</em>{U\mapsto\mathbb{r}, \mapsto\left(((A\right), y}lft(A\right), \cdots, y}(A\right\right). &#36;
Set&#36;\mathcal{A}^n&#36;It's got a mimic coordinate system on it.&#36;\left{O,e_i\right}&#36;, the variable is \\left{x^i\right}<em>i=1,2,...n&#36;, coordinates map
&#36; \varphi</em>{A}A\mapsto\mathbb{n}, A\mapsto\left(1}left}, \left(\A\right), \cdots, }left(}A\right\right). &#36;
If</p>
<ul>
<li>&#36;\varphi_{U}:U\mapsto\varphi_{U}\left(U\right)为双射&#36;</li>
<li>&#36;\varphi_{U}\circ\varphi_{A}^{-1}:\varphi_{A}\left(U\right)\mapsto\varphi_{U}\left(U\right)与\varphi_{A}\circ\varphi_{U}^{-1}:\varphi_{U}\left(U\right)\mapsto\varphi_{A}\left(U\right)都是C^{\infty}映射&#36;
So, what do you say?&#36;{U,\phi_U}&#36;It's defined as&#36;U&#36;A wide-scale system on it.</li>
</ul>
<p>We give the following characteristics, and we create a link between the imitation and the broader range of coordinates.</p>
<p>Nature: Establishment&#36;U&#36;Yes.&#36;\mathcal{A}^n&#36;The last open area with a broad coordinates. Yes&#36;{U,\phi_U}&#36; then</p>
<ul>
<li>&#36;U&#36;The neutrons are gathered in&#36;\varphi_{U}&#36;It's like...&#36;R^n&#36;The neutron collection is set up anyway.</li>
<li>Set&#36;f&#36;Yes.&#36;U&#36;defined as the standard field,&#36;f&#36;Continuous equivalent to &#36;f&#36; &#36;\varphi _{U}^{- 1}&#36; Yes. &#36;\varphi _{U}( U)&#36;Up-Sequencing Functions</li>
</ul>
<p>Nature: Establishment&#36;U&#36;Yes.&#36;\mathcal{A}^n&#36;The last open area,&#36;f&#36;Yes&#36;U&#36;, and the following is the same as the term.</p>
<ul>
<li>&#36;f&#36;(a) Is micro-synthetic (in a analog system of analogue coordinates);</li>
<li>&#36;f&#36;(a) Readable micro-functions under a broad coordinates;</li>
<li>&#36;f&#36;Read to micro functions under any broad coordinates</li>
</ul>
<h2>Vector-value functions and vectors in the simulation space Field</h2>
<h3>Point Up Vector Space</h3>
<p>This subsection begins with a discussion of vector-value functions and vector fields in the simulation space, and first, we discuss the vector as a vector in a directional segment, compared to the vector as a directional conductor in the study later.</p>
<p>Definitions: Set-up.&#36;\mathscr{A}^n&#36;For one.&#36;n&#36; Simulation Space &#36;,A\in\mathscr{A}^n&#36;Define its last point.
&#36;&#36;T_{A}={(A,B)\mid B\in A^{n&#125;&#125;&#36;&#36;
Yes.&#36;T_A&#36;Add:&#36;\left(A,B\right)+\left(A,C\right)=\left(A,D\right)&#36;♪ That makes ♪&#36;&#36;\overrightarrow{AD}=\overrightarrow{AB}+\overrightarrow{AC}&#36;&#36;Defined multiplier:&#36;\lambda \left ( A, B\right ) = \left ( A, C\right )&#36;♪ And make &#36;&#36;\overrightarrow {AC}= \lambda \overrightarrow {AB}.&#36;&#36;This will be as defined above in linear space (&#36;T_{A},+,\cdot&#36;)&#36;A&#36;Vector space for points</p>
<p>Intuitively, the definition above describes the&#36;A&#36;The assembly of the directional segment from which the starting point is taken and the addition (parallel quadrilateral law) defined above (extension of the same direction several times)</p>
<p>From a linear space perspective, the definition defines one at each point of the analogue space.&#36;n&#36;Vector space, with different starting points in vector space at different points, varies from vector to vector in each vector space.</p>
<p>Of course, in the simulation space, a relationship called a smooth shift can be defined between vector space at different points due to the existence of a global simulation coordinate system.</p>
<p>Set &#36;A,B\in\mathscr{A}^{n},A\neq B.&#36;  Set&#36;u=\left(A,C\right),v=\left(B,D\right)&#36;Both&#36;T_{A},T_{B}&#36;. If
&#36;&#36;\overrightarrow{AC}=\overrightarrow{BD}&#36;&#36;
We call it&#36;u,v&#36;Parallel movement of each other, abbreviated.</p>
<p>After the given analogue coordinate system, we can imitate each point of the space in a mass form.&#36;A=\left{O,e_{i}\right}&#36;Yes.&#36;\mathcal{A}^n&#36;The above-impressive coordinate system, if&#36;v=\left(A,B\right)\in T_{A}&#36;and
&#36;&#36;\overrightarrow{AB}=v^{i}e_{i}&#36;&#36;
We call it&#36;\left(v^{1},v^{2},\cdots,v^{n}\right)&#36;Yes.&#36;v&#36;In the simulation coordinates system.&#36;A&#36;The next mass form</p>
<h3>The vector below the range coordinates. Field</h3>
<p>This section introduces the concept of vector field, similar to the section “Functions and ranges in the imitation space”, where we assign a vector to each point in space.</p>
<p>Definitions: Establishment&#36;U\subset\mathscr{A}^n&#36;To imitate the area in space,&#36;U&#36; The defined vector field is a map:
&#36;&#36;\begin{matrix}
X:U\mapsto\mathcal{A}^n \
A\mapsto B
\end{matrix}&#36;&#36;
The intuitive explanation is that we are&#36;U&#36;Every point up.&#36;A&#36;All assigned one.&#36;T_A&#36;Up the vector, that's one by&#36;A&#36;Additional destination designation for starting point&#36;B&#36;；<strong>The vector field is to assign each point of the area to one.&#36;\mathscr{A}^n&#36;The middle point is the end. So each point corresponds to a vector.</strong></p>
<p>After the given imitation coordinates, vector field&#36;X&#36;And each point of vector can be written in a mass form, and we remember the vector field as
&#36;&#36;X\left(A\right):=\overrightarrow{AB}\in\mathbb{R}^{n}&#36;&#36;
If there is.&#36;\overrightarrow{AB}=X^i\left(A\right)e_i&#36; Name&#36;\left(X^{1},X^{2},\cdots,X^{n}\right)&#36;is the mass of the vector field under the coordinates, that is, the map of the area to the vector below.
&#36;&#36;00\&amp;U\mapsto\mathbb{R}^{n},\&amp;A\mapsto\left(X^{i}\left(A\right)\right)_{i=1,2,\cdots,n}.\end{aligned}&#36;&#36;</p>
<p>It is easy to imagine that, under different analogue coordinates, the mass form of a vector field is altered.&#36;\phi_A&#36;, and the mapping from space to real number is not related to the coordinates selected) and we need to create a formula that is in the form of a vector field that is changed by the coordinates</p>
<p>Theorem (relationship of coordinates in the form of vector field weights): assumed to be&#36;\mathcal{A}^n&#36;There are two analogue coordinates on the top.  &#36;\left { O, e_{i}\right },\left{O^{\prime},e_{i}^{\prime}\right}&#36;  They're satisfied.
&#36;&#36;\overrightarrow{OO^{\prime&#125;&#125;=a^{i}e_{i},e_{i}=T_{i}^{j}e_{j}^{\prime}&#36;&#36;
Set Vector Field &#36;X&#36; Yes. &#36;\mathcal{A}&#36;The mass below is &#36;\left (X^i(A)\right)<em>i=1,...,n&#36;,在 &#36;\mathcal{A}^\prime&#36;下的分量形式为&#36;\left(X^{\prime i}\left(A\right)\right)</em>\cdots, &#36;n
&#36;&#36;X^{\prime i}\left(A\right)e_{i}^{\prime}=X=X^{j}\left(A\right)e_{j}=X^{j}\left(A\right)T_{j}^{k}e_{k}^{\prime}.&#36;&#36;
So there is.
&#36;&#36;X^{\prime i}=T_{j}^{i}X^{j}\left(A\right).&#36;&#36;</p>
<p>The regularity of the vector field (continuous, micro, smooth) is based on a line of a imitation coordinates from&#36;A\to X^i(A)&#36;The normal decision, which is the standard field.&#36;f&#36; We can easily prove:<strong>There is a regularity under a analogue coordinate system, which is a regularity under all analogue coordinates.</strong></p>
<h3>Vector under Broad Coordinates Field</h3>
<h4>Natural boundary</h4>
<p>We can study speeds under two different coordinates, such as polar and standard straight-angled, which creates two different speeds, and this section discusses the differences and linkages between these two speeds.</p>
<p>Not only is it for polar coordinates set up, but it is also possible to study the broad range of coordinates defined in the general open area, and more generally we have the following calculations:</p>
<p>Assumptions&#36;U\subset\mathscr{A}^n&#36;For an opening, it defines a broad coordinate. Yes&#36;\left{U,\varphi_{U}\right}&#36;, the variable is recorded as&#36;\left{y^i\right}_i=1,...,n.&#36;There's a pouncer coordinate system.&#36;x^i&#36;(Indicated)</p>
<p>So for one...&#36;U&#36;Physics for moving up&#36;P&#36;♪ It's the equation of its movement ♪ &#36;x^i&#36; Lower Writing &#36;&#36; (x^i(t))<em>i=1,...,n&#36;,在 &#36;y^i&#36; 下写作&#36;\left(y^{i}\left(t\right)\right)</em>I'm not sure.
&#36;&#36;\frac{dr}{dt}\left(t\right)=\dot{x} ^{i}\left(t\right)e_{i}=\frac{\partial x^{i&#125;&#125;{\partial y^{j&#125;&#125;\left(y\left(t\right)\right)\frac{dy^{j&#125;&#125;{dt}\left(t\right)e_{i}=\dot{y} ^{i}\left(t\right)\cdot\frac{\partial x^{j&#125;&#125;{\partial y^{i&#125;&#125;\left(y\left(t\right)\right)e_{j}&#36;&#36;
So, yes,&#36;P&#36;A point on the track.&#36;A&#36;We define
&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;=<em>{\varphi</em>You're not gonna get a chance to get a job.
Call it a broad coordinate to&#36;{y^i}_{i=1,\cdots,n}&#36; Natural frame</p>
<p><strong>Under a broad range of coordinates, the natural frames of each point are generally different, so they form the mapping of the point to the frame, which we call the natural frames. Field</strong></p>
<p><strong>We have specific imitation coordinates in natural frames, which is actually valid in all imitation coordinates.</strong></p>
<p>Theorem (platform conversion): set&#36;\left{U,\varphi_{U}\right}和\left{V,\varphi_{V}\right}&#36;It's a simulation space.&#36;\mathscr{A}^n&#36;two broad coordinates of the system, the coordinates of which are recorded as &#36;\left{y^i\right}<em>{i=1,...,n}&#36;,&#36;\left{z^i\right}</em>{i=1,...,n}&#36;,记&#36;\left{\boldsymbol{\sigma}<em>{i}\right}</em>{i=i,\cdots,n},\left{\boldsymbol{\tau}<em>{i}\right}</em>{i=i,\cdots,n}&#36;为其对应的自然标架.设&#36;A \cap V, then
&#36;&#36;\t\right=frac(partial y^)\partial z^}<em>{\varphi</em>{V}\left(A\right)}\sigma_{k}\left(A\right).&#36;&#36;</p>
<h4>The amount of vector field under the natural frame</h4>
<p>We naturally want to write the vector field on the natural frame. Down</p>
<p>Definitions: Establishment&#36;X:U\mapsto\mathscr{A}&#36;Regional&#36;U&#36;Defined vector field, set&#36;\langle U,\varphi_{\upsilon}\rangle&#36;Yes&#36;U&#36;A broad system of coordinates defined above, with a coordinate variable of &#36;{y}^i}<em>{i=1,\ldots,n}&#36; ,其自然标架场记为 &#36;\sigma</em>I'm sorry, but if...
&#36;&#36;X\left(A\right)=X^{i}\left(A\right)\sigma_{i},&#36;&#36;
We call it &#36;\left.<em>{i=1,\cdots,n}&#36;为&#36;X&#36;在&#36;\left{U,\varphi</em>The weight form under &#36;</p>
<p><strong>Like the normal analogue coordinate system, if a vector field meets a regularity under a broad coordinate system, he meets that regularity under other imitation coordinates and other broad coordinate systems.</strong></p>
<p>Theorem: Set&#36;{U,\varphi_{U&#125;&#125;&#36;and &#36;{V,\varphi_{V&#125;&#125;&#36;It's a simulation space. &#36;\mathscr{A}^n&#36;the two broad coordinates defined in the list of the variables as &#36;\left{y^i\right}<em>{i=1,...,n},\left{z^i\right}</em>{i=1,...,n}&#36;  记&#36;\left{\sigma_i\right}<em>i=1,...,n,\left{\tau_i\right}</em>{i=1,...,n}&#36;为其对应的自然标架.设&#36;X&#36;为定义在&#36;U\cap V&#36;上的向量场，设 &#36;A\in U\cap V.&#36;如果记&#36;\left(Y^{i}\right)<em>{i=1,\ldots,n}&#36;为&#36;X&#36; 在&#36;\left{U,\varphi</em>{U}\right}&#36;下的分量形式&#36;,\left(Z^{i}\right)<em>{i=1,\ldots,n}&#36;为&#36;X&#36; 在&#36;\left(V,\varphi</em>The weight form below &#36;&#36;, then
&#36;Y^(A\right)=\frac(partial y^)\fsc(}{\)}fsc(})}fsc(})}fsc(})}fsc)}Fscene(})}Fsc)}Fscene(})}Fsc)<em>{\varphi</em>{v}\left(A\right)}Z^{j}\left(A\right).&#36;&#36;</p>
<h3>Vector as a directional guide</h3>
<h4>Introduction of directional guides</h4>
<p>We can look at the concept of vector from another angle.</p>
<p>Now, let's say,&#36;n&#36;The exemplation of the space open.&#36;U&#36;A fixed point &#36;A&#36;We defined a vector on it. &#36;v=(A,B),\overrightarrow{AB}=v^ie_i&#36; (under a given imitation coordinate system)</p>
<p>We'll think about it.&#36;U&#36;Micro functions on &#36;f&#36;(read from the simulation coordinates system)&#36;\mathbb{R}^{n}&#36;Functions on the opening set, as &#36;f_A=f\circ\varphi_A^{-1}).&#36;We can normally be right. &#36;f&#36; Directional guide numbers:
&#36;&#36;\partial r}f\left (A\right): =frac(partial f A}partial x}i}\left|<em>{\varphi</em>{A}\left(A\right)}v^{i}=\frac{\partial\left(f\circ\varphi_{A}^{-1}\right)}{\partial x^{i&#125;&#125;\right|<em>{\varphi</em>\left (A\right)}v^i}.&#36;
We want to extend this to the broad system, to take the broad range. Yes&#36;{U,\varphi_{U&#125;&#125;&#36; The coordinates are recorded as &#36;y^i}<em>I'm sorry, I'm sorry, but I'm sorry.
&#36;v=w}sigma</em>{i},\sigma_{j}=\frac{\partial x^{i&#125;&#125;{\partial y^{j&#125;&#125;e_{i}.&#36;&#36;
从而
&#36;&#36;v^{i}=\frac{\partial x^{i&#125;&#125;{\partial y^{j&#125;&#125;w^{j}.&#36;&#36;
考虑&#36;f&#36;在广义坐标系下的读取&#36;f_U&#36;则
&#36;&#36;\begin{aligned}&amp;f_{U}:\varphi_{U}\left(U\right)\mapsto\mathbb{R},\&amp;\left(y^{i}\right)<em>{i=1,\cdots,n}\mapsto f\circ\varphi</em>{U}^{-1}\left(y^{i}\right)=\left(f\circ\varphi_{A}^{-1}\right)\circ\left(\varphi_{A}\circ\varphi_{U}^{-1}\right)\left(y^{i}\right).\end{aligned}&#36;&#36;
我们有下面的观察
&#36;&#36;w^{i}\frac{\partial f_{U&#125;&#125;{\partial y^{i&#125;&#125;|<em>{\varphi</em>{U}\left(A\right)}=\frac{\partial f_{A&#125;&#125;{\partial x^{j&#125;&#125;|<em>{\varphi</em>{A}\left(A\right)}\frac{\partial x^{j&#125;&#125;{\partial y^{i&#125;&#125;|<em>{\varphi</em>{U}\left(A\right)}w^{i}=\frac{\partial f_{A&#125;&#125;{\partial x^{i&#125;&#125;|<em>{\varphi</em>{A}\left(A\right)}v^{i}=\partial_{v}f\left(A\right).&#36;&#36;</p>
<p><strong>The blogger adds:&#36;f&#36;Read under any broad coordinate&#36;f_U&#36;The directional guide is the same result, which means that the directional guide is the nature of the vector on the point, not the coordinates.</strong></p>
<h4>Controller</h4>
<p>The directional numbers defined earlier have some bright properties, and we have the following in abstraction (as studied earlier).&#36;\partial_{v}f\left(A\right)&#36;It's a smooth function to the real number, so we'll start talking about it.</p>
<p>Definitions: Establishment&#36;A\in\mathscr{A}^n&#36;Sets &#36;matcal{F} for emulation of a point in space<em>{\mathrm{A&#125;&#125;&#36; 为在&#36;A&#36;点附近有定义的光滑函数集合.称 &#36;D&#36;为 &#36;A&#36; 点的一个导算子，若 &#36;D: \mathcal{F}</em>I'm not gonna let you go.</p>
<ul>
<li>Locality: Yes&#36;f&#36; 、&#36;g&#36;Define in&#36;O&#36;Two smooth functions on the adjacent area, satisfying: if A's neighbourhood exists&#36;U&#36;Yes.&#36;U&#36;Go, go, go!&#36;f=g&#36;, then&#36;Df=Dg.&#36;</li>
<li>Linear:&#36;\forall\alpha,\beta\in\mathbb{R}&#36;, &#36;D\left(\alpha f+\beta g\right)=\alpha Df+\beta Dg.&#36;</li>
<li>The Lebniz formula:&#36;D\left(fg\right)=f\left(A\right)Dg+g\left(A\right)Df.&#36;</li>
</ul>
<p><strong>Especially if&#36;f&#36;Yes.&#36;A&#36;One of the points is within the open area, and then&#36;Df=0&#36;</strong></p>
<p>We can link the conductor to the vector according to the following conclusions.</p>
<p>Definitions: Establishment&#36;\mathscr{A}^n&#36;There's a simulation system on it.&#36;\mathcal{A}=\left{O,e_i\right}.&#36;Set&#36;A\in\mathscr{A}^n,\mathscr{D}&#36;Yes&#36;A&#36;Click a wizard to exist &#36;T_{\mathrm{A&#125;&#125;&#36; Only vector on (&#36;A,B)&#36;♪ That makes ♪&#36;\overrightarrow{AB}=v^ie_i&#36;and
&#36;&#36;Df=\partial_{\nu}f\left(A\right),\forall f\in\mathcal{F}_{A}.&#36;&#36;</p>
<p><strong>Note that for the first time we introduced the concept of directional guide from the vector in the “induction of directional guide” section of this paper, and independently defined the concept of conductor in the “guider” part of this paper, we now find that the definition of a point is the same as the definition of the guidance counter and the definition of the point in vector, except that the latter no longer relies on a specific coordinate system. So we call the sum of all the algorithms of a point the vector space of that point.</strong></p>
<h4>Natural Palette and Guide Calculator</h4>
<p>Although the algorithm seems abstract, he can explain in a visual way many of the computational rules, such as the natural frame that this section wants to discuss again. Field</p>
<p>From the above point of view, we can draw the following marks. Set&#36;\left(U,\varphi_{\upsilon}\right)&#36;is a broad system, &#36;\left (y^i}right)<em>i=1,...,n&#36;为其自变量.我们记&#36;\left{\partial</em>{i}\right}<em>{i=1,\cdots,}for &#36;
&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;...&#36;&#36;&#36;&#36;&#36;...</em>{i}f:=\frac{\partial\left(f\circ\varphi_{U}^{-1}\right)}{\partial y^{i&#125;&#125;.&#36;&#36;</p>
<p>Could be verified in&#36;U&#36; And at every point, it's a conductor, and we can verify it. &#36;A=\left{O,\right.&#36; &#36;e_i}&#36;A analogue coordinate system with a variable of &#36; (x^i)<em>i=1,...,n&#36;.将&#36;f&#36;在&#36;\mathcal{A}&#36;上的读取记为&#36;f</em>\mathcal{A}&#36;, then
&#36;&#36;f\circ\varphi_{U}^{-1}=f\circ\varphi_{A}^{-1}\circ\left(\varphi_{A}\circ\varphi_{U}^{-1}\right)&#36;&#36;</p>
<p>And so, \\left.\text}f\left=freft=frec{\ft=frec\varph ft=fres \resc\ft=ft=frest=ft=ft=ft=frest=ft=ft=frest=ft=ft=ft=rc\vappi ft=ft=ft=t\t\t\t\t\t\t\t\t\wt\t\t\wt\ft=t\t\t}t}t}ft=t\t\t\t\t\t}{\t}{\t}{\t}t\t\t}t\t\t\t\t\t\ft}t\t\t}t\t\t\t\t\t\t\t\t\t\t\t\t\t\frt\t\rt\frt\fr\r \rt\t\frt\t<em>{\varphi</em>{U}\left(A\right)}=\frac{\partial f_{A&#125;&#125;{\partial x^{j&#125;&#125;\left|<em>{\varphi</em>{A}\left(A\right)}\frac{\partial x^{j&#125;&#125;{\partial y^{i&#125;&#125;\right|<em>{\varphi</em>{U}\left(A\right)}.&#36;</p>
<p>Compare the definition of natural frames:
&#36;&#36;\sigma_{i}=\frac{\partial x^{j&#125;&#125;{\partial y^{i&#125;&#125;e_{j}&#36;&#36;</p>
<p>Well...
&#36;&#36;\partial (((((()}f\left(A\right)=\frac(()}partialxj}\left|)<em>{\varphi</em>{A}\left(A\right)}\frac{\partial x^{j&#125;&#125;{\partial y^{i&#125;&#125;\right|<em>{\varphi</em>\left (A\right)}. &#36;
That means, in the same sense.&#36;{\partial_{i&#125;&#125;&#36;The natural marker we're talking about.</p>
<h2>Complementing the geometry of the imitation space</h2>
<h3>Regular curve in the simulation space</h3>
<p>3D imitation space can be imitated.<a href="/en/blog/2025/02/04/differential-geometry-notes/">Vector and curve theory in geometry in three-dimensional European space</a>The concept of the positive curve, because the imitation space does not have the length of the vector, makes the curve long, the curve rate, the scratch rate impossible, but we can still discuss the smooth, positive, the same direction.</p>
<p>Definitions: Establishment &#36;\mathcal{A}^{3}&#36;The #savesaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:&#36;\gamma_:(-\varepsilon,\varepsilon)\mapsto\mathcal{A}^{3}&#36;Set a simulation coordinate system for a map.&#36;\mathcal{A}=\left{O,e_{i}\right}&#36;, the coordinates are marked as&#36;\varphi_{\mathcal{A&#125;&#125;.&#36;If
&#36;&#36;\left(-\varepsilon,\varepsilon\right)\xrightarrow{\gamma}\rightarrow\mathcal{A}^{3}\xrightarrow{\varphi_{i&#125;&#125; R&#36;&#36;
It's continuous/micro/slow map, then called &#36;\gamma&#36; Yes &#36;x^3&#36;. further if&#36;\varphi_{A}\circ \gamma&#36;Yes.&#36;\mathbb{R}^3&#36;, the normal curve above which is called &#36;\gamma&#36;Yes &#36;x^3&#36;- Up the regular curve.</p>
<p>Theorem: Set &#36;\gamma:\left(-\varepsilon ,\varepsilon \right)\mapsto \mathcal{A}^{3}&#36;It is a continuous/micro-sliding/regular curve, &#36;A\in \gamma\left(\left(-\varepsilon,\varepsilon\right)\right),A=\gamma\left(t_0\right).t_0\in\left(-\varepsilon,\varepsilon\right),A&#36;..of the border.&#36;U&#36;There's a wide range of coordinates on it.
&#36;{U,\varphi_{U&#125;&#125;&#36;, the coordinate variable is recorded as&#36;y^{i}&#36;Then...
&#36;&#36;\left(t_{0}-\varepsilon^{\prime},t_{0}+\varepsilon^{\prime}\right)\xrightarrow{\gamma}U\xrightarrow{\varphi_{U&#125;&#125;\varphi_{U}\left(U\right)\subset\mathbb{R}^{3}&#36;&#36;
A continuous/micro/slid/regular curve segment</p>
<h3>Residual Quantities Field and Function Numeration</h3>
<h4>Left</h4>
<p>In this section of the "Guide Calculator", we are a collection of smooth functions that are used near a point.&#36;\mathcal{F}_{\mathrm{A&#125;&#125;&#36; We're looking at this gathering.</p>
<p>Obviously, we found&#36;\mathcal{F}_{\mathcal{A&#125;&#125;&#36;There's a linear structure down there.
&#36;&#36;\left(\alpha f+\beta g\right)\left(x\right):=\alpha f\left(x\right)+\beta g\left(x\right).&#36;&#36;
But this space dimension is too high to be studied. We need to put it all in the right place.&#36;v\in T_A&#36;The function of which the same value is obtained under the function of the (becoming the conductor) is to be considered the same object, and the following definition is introduced for this purpose:</p>
<p>Definitions: Establishment&#36;A&#36;Yes&#36;\mathcal{A}^{n}&#36;Middle point, remember.&#36;\mathcal{F}_{\mathrm{A&#125;&#125;&#36;Define in &#36;A&#36; The nearby smooth function is all... if any definition is made&#36;O&#36;..the calculator&#36;D&#36;Both.
&#36;&#36;Df=Dg&#36;&#36;
We call it&#36;f\sim g.&#36; It's an equal value relationship.</p>
<p>Definitions: Note&#36;\mathscr{F}_A=\mathscr{F}_A/\sim&#36;"...and called &#36;\mathscr{A}&#36; Left vector space of points (<a href="/en/blog/2024/10/16/point-set-topology-notes/">I'm just trying to get a little more commercial space in the tungsten.</a>) function &#36;f&#36; The equivalent of the value is given as&#36;\overline f&#36;, and linear operations:
&#36;&#36;\alpha\overline{f}+\beta\overline{g}:=\overline{\alpha f+\beta g}.&#36;&#36;
We gave a deduction to that definition.&#36;\overline{fg}=g\left(A\right)\overline{f}+f\left(A\right)\overline{g}&#36;</p>
<p>To study the structure of commercial space, we need to first look at who has the zero equivalent, and give the following theorem.</p>
<p>Theorem: Set  &#36;A\in \mathscr{A}, f&#36;Define in &#36;A&#36;Near smooth function. Set &#36;{U, \\varphi <em>U}&#36;为任意一个满足  &#36;A\in U&#36; 的广义坐标系 , 自变量记为 &#36;{ y^i} <em>{i= 1, \ldots , n}&#36;。 记  &#36;f_U= f\circ\varphi <em>Reads the U^-supplies
♪ The world is so full of shit ♪</em>{U&#125;&#125;{\partial y^{i&#125;&#125;|</em>{\varphi</em>{U}\left(A\right)}=0.&#36;&#36;</p>
<h4>Micro-diphor</h4>
<p>Now, how do you calculate the differentials of the function?</p>
<p>Theorem: Assumptions&#36;{ U, \varphi _U}&#36;For any satisfaction  &#36;A\in U&#36; The broad coordinate system, &#36;y^ <em>{i= 1, \ldots , n}&#36;是其自然余标架场，那么对于&#36;U&#36;上任意光滑函数&#36;f&#36;
&#36;&#36;df=\frac{\partial\left(f\circ\varphi</em>{U}^{-1}\right)}{\partial y^{i&#125;&#125;dy^{i}.&#36;&#36;</p>
<h3>Zoom Field</h3>
<p>The concept of the field is to assign an object to each point in the space area, so we can assign a dual linear function to each point, introducing the definition below</p>
<p>Definitions: Establishment&#36;U&#36; Yes &#36;\mathscr{A}^n&#36;The last open area, for any point. &#36;A\in U&#36;, we specify a definition in&#36;A&#36;Double linear function in vector space of points&#36;b(A).&#36;Here's the thing.&#36;b&#36;It's called a two-linear functional field.</p>
<p>Definition: A non-degradable symmetrical double-linear field defined in space areas, called a pseudo-Riemann (Liman) measure. If this two-linear model is still positive at each point, the measure is called the Riemann measure. Further, if the weight is continuous/micro/slid under a natural residual frame of a wide-scaled system, the pseudo-Riemann measure is continuous/micro/slid:</p>
<h2>Simulate the curve of space</h2>
<h3>A face-to-face painting of the curve</h3>
<h4>Parametric painting of the curve</h4>
<p>The mathematical description of the curve is more complex than the curve, and for curves we have used the argument equation below to paint it.
&#36;&#36;r:\left(-\varepsilon,\varepsilon\right)\mapsto\mathscr{A}^3,r^{\prime}\left(\tau\right)\neq0.&#36;&#36;
He can easily understand the equation of movement as a prime point.</p>
<p>In this vein, we still think about the equation of parameters to paint the face of the curve, so that it reflects the specifics of the two-dimensional object.
&#36;&#36;r:\left(-\varepsilon,\varepsilon\right)\times\left(-\varepsilon,\varepsilon\right)\mapsto\mathscr{A}^3,\left(s,t\right)\mapsto r\left(s,t\right).&#36;&#36;
It's actually understood as a single-parametric curve, by changing.&#36;s&#36;The way in which the curve is woven into a curve, and in order to ensure that the curve is properly woven and that the curve is able to be woven smoothly, rather than still being a line, the conditions below need to be met.
&#36;&#36;\begin{matrix}
\partial_{t}r\neq0 \
 \partial_{s}r\neq0.\
\partial_{t}r同\partial_{s}r不共线
\end{matrix}&#36;&#36;</p>
<p>In conclusion, we can define it more fairly.</p>
<p>Definition (parallel drawings): set
&#36;&#36;r:\left(-\varepsilon,\varepsilon\right)\times\left(-\varepsilon,\varepsilon\right)\mapsto\mathscr{A}^3,&#36;&#36;
&#36;&#36;\left(s,t\right)\mapsto r\left(s,t\right)&#36;&#36;</p>
<p># As micro/ #&#36;C^k/C^\infty&#36;Map, satisfy&#36;\partial_sr(s,t)&#36;and&#36;\partial_tr(s,t)&#36;Not conjunctive, so called map.&#36;r&#36;♪ To the tiniest
&#36;C^{k}&#36; / Glossy Float&#36;S=r\left(\left(-\varepsilon,\varepsilon\right)\times\left(-\varepsilon,\varepsilon\right)\right)&#36;- The parametric pattern.</p>
<h4>Curve as Function Image</h4>
<p>Let's discuss a way of biased analysis, after all, is the curve a one-dollar function, and the curve is the same?</p>
<p>Definition (curvature as function image): Set &#36;\mathscr{A}^3&#36;It's built a simulation system. &#36;A={O,e_1 ,e_2,e_3}&#36;, the coordinate variable is recorded as&#36;x^1,x^2,x^3.&#36;Set&#36;f&#36;Define in&#36;\left(-\varepsilon,\varepsilon\right)\times\left(-\varepsilon,\varepsilon\right)\to R&#36;The real value on is insignificant/&#36;C^k&#36;/Slower function. If S\subset \mathscr{A}3 content
&#36;S=\left{varphi^,\left},\right\varepsilon|&lt;x^{1},x^{2}&lt;{\fnH00FFFF}&#36;&#36;US&#36;&#36;US&#36;&#36;US&#36;&#36;US&#36;US&#36;&#36;US&#36;US&#36;US&#36;US&#36;&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;&#36;&#36;US&#36;US&#36;US&#36;US&#36;&#36;US&#36;&#36;&#36;&#36;US&#36;&#36;&#36;US&#36;&#36;US&#36;&#36;&#36;&#36;US&#36;&#36;&#36;&#36;&#36;US&#36;
Claims &#36;S&#36; # As micro/ #&#36;C^k&#36; /Slushy face</p>
<h4>Curve as Equivalent</h4>
<p>We're thinking of the equivalent, the collection of the zero points of the scale function, to represent the curve.<a href="/en/blog/2024/12/12/analytic-geometry-notes/">Parsing Geometry</a>Common thought</p>
<p>Definition (critic as equivalent): set&#36;S\subset \mathscr{A}^3&#36;  &#36;f&#36;Yes&#36;\mathscr{A}^3&#36;Central region&#36;D&#36;Minor/Definable&#36;C^k/&#36;smooth cursor field (real function), and satisfied
&#36;&#36;\forall A\in S,f\left(A\right)=0.&#36;&#36;
If it's in &#36;S&#36;There's more up there. &#36;df\neq0&#36;Then we call it &#36;S&#36; # As micro/ #&#36;C^{k}&#36;/Slushy face,</p>
<h4>Use a wide range of coordinates</h4>
<p>Definition: Curvature can be seen as&#36;\mathscr{A}^3&#36;, and then a non-empty subset of the &#36;S&#36;) it has the following character:&#36;\forall A\in S,\exists U\ni A,U&#36; Yes &#36;\mathscr{A}^3&#36;Open the area, and there's a broad coordinates. Yes&#36;{U,\varphi_{U&#125;&#125;&#36;
&#36;&#36;\varphi:U\mapsto V\subset\mathbb{R}^{3},\varphi_{U}\left(S\cap U\right)\subset\mathbb{R}^{2}\times\left{0\right}\cap V.&#36;&#36;
That means that, in part, the curve can be "slided" with a broad, sufficiently smooth coordinate.</p>
<p><strong>These are intuitive and reasonable drawings, and we'd like to prove that these expressions are price-free.</strong></p>
<h3>Simulate the curves in space</h3>
<h4>Theories of the Invisible Functions</h4>
<p>We'll think about it.&#36;\mathbb{R}^n&#36;Micro/slow map between mid-opening:
&#36;&#36;F:\mathbb{R}^m\supset U\mapsto V\subset\mathbb{R}^n.&#36;&#36;
If &#36;a\in U&#36;We call it a linear map.<em>aF:\mathbb{R}^m\mapsto\mathbb{R}^n&#36;为&#36;F&#36; 在&#36;a.&#36;.00 point guide map if
&#36;&#36;\frac(left|F\left(a+h\right)-F\left(a\right)-D</em>{a}F\left(h\right)\right|<em>{R^{n&#125;&#125;}{\left|h\right|</em>{\cHFFE7C5}Rapsto0,\levft\right
We focus on... &#36;\mathcal{D}_aF&#36; It's full of shit. &#36;min{m,n}&#36;The definition is given below.</p>
<p>Definitions: Establishment&#36;F&#36;Yes&#36;U\mapsto V&#36;A micromap of the&#36;\forall a\in U,D_aF&#36;Full of tarts.</p>
<ul>
<li>&#36;若m\leqslant n,称F在U内是浸入\left(immersion\right)&#36;</li>
<li>&#36;若m\geqslant n,称F在U内是浸没\left(submersion\right)&#36;</li>
</ul>
<p>Theorem: Set &#36;F&#36; Yes.&#36;\mathbb{R}^m&#36;Opening of the session &#36;U&#36; Present.&#36;\mathbb{R}^n&#36;Opening of the session &#36;V&#36; Yes. &#36;C^1&#36; Map</p>
<p>If&#36;F&#36;Yes.&#36;U&#36;Into, for any&#36;a\in U&#36;, exists&#36;a&#36;..of the border.&#36;W&#36;and&#36;F(a)&#36;..of the border.&#36;W^{\prime}&#36;And the microsynthesis of the embryo.
&#36;&#36;\varphi:W^{\prime}\mapsto\varphi\left(W^{\prime}\right)\subset\mathbb{R}^{n}&#36;&#36;
Make
&#36;&#36;\varphi\circ F:W\mapsto\varphi\left(W^{\prime}\right),\\left(x^{1},x^{2},\cdots,x^{m}\right)\mapsto\left(x^{1},x^{2},\cdots,x^{m},0,\cdots,0\right).&#36;&#36;
If&#36;F&#36;Yes.&#36;U&#36;It's immersed, it's present.&#36;\psi ^{-1}(W)\subset R^m&#36;Present.&#36;W&#36;The micro-synthesis of the embryo.&#36;\psi&#36; Make
&#36;&#36;F\circ\psi:R^{m}\supset\psi^{-1}\left(W\right)\mapsto F\left(W\right),\left(x^{1},x^{2},\cdots,x^{m}\right)\mapsto\left(x^{1},x^{2},\cdots,x^{n}\right).&#36;&#36;</p>
<h4>Localisation of Curves</h4>
<p>We can start with the definition of the curve, which is also the definition of the local parameter of the curve.</p>
<p>Set &#36;S\subset \mathscr{A}^3&#36; A collection of non-empty, with a collection of &#36;\mathscr{A}^3&#36;The heir to the throne.&#36;{O,e_1,e_2,e_3}&#36; A analogue coordinate system, which is mapd to &#36;\varphi_{\mathcal{A&#125;&#125;&#36;, the coordinates are recorded as&#36;\left{x^1,x^2,x^3\right}.&#36;If&#36;\forall A\in S&#36;There's one.&#36;A&#36; Yes. &#36;\mathscr{A}^3&#36;Open Border in the Center&#36;U&#36; And a map. &#36;\varphi&#36;♪ That makes ♪
&#36;&#36;\varphi:\left(-\varepsilon,\varepsilon\right)\times\left(-\varepsilon,\varepsilon\right)\mapsto U\cap S&#36;&#36;
Satisfaction:</p>
<p> &#36;\varphi&#36;Yes.&#36;\left(-\varepsilon,\varepsilon\right)\times\left(-\varepsilon,\varepsilon\right)&#36;Present.&#36;U\cap S&#36;- Double-shot;</p>
<p>Map:&#36;&#36;\varphi_{A}\circ\varphi:\left(-\varepsilon,\varepsilon\right)\times\left(-\varepsilon,\varepsilon\right)\mapsto\varphi_{A}\left(U\right),\\left(u,v\right)\mapsto\left(x^{1}\left(u,v\right),x^{2}\left(u,v\right),x^{3}\left(u,v\right)\right)&#36;&#36;(a) To be smooth map;</p>
<p>Remember&#36;x^{i}\left(u,v\right)=\left[\varphi_{A}\circ\varphi\left(u,v\right)\right]^{i}.&#36; vector
&#36;&#36;\left(\partial_{u}x^{1},\partial_{u}x^{2},\partial_{u}x^{3}\right),\left(\partial_{v}x^{1},\partial_{v}x^{2},\partial_{v}x^{3}\right)&#36;&#36;
Yes.&#36;\left(-\varepsilon,\varepsilon\right)\times\left(-\varepsilon,\varepsilon\right)逐点线性无关.&#36;</p>
<p>Well...&#36;\varphi&#36;Called&#36;S&#36;Yes.&#36;A&#36;A local smoother parameter near the point if&#36;S&#36;At least one local smooth parameter at any point, or&#36;S&#36;It's smooth.</p>
<p><strong>With the knowledge of this section, we can prove that the four curvature definitions proposed in the previous section are identical and are price equivalent, and that the present section describes the definition of hidden functions and the definition of local parameters to address this problem.</strong></p>
<h3>The cutting of the curve and the cutting field</h3>
<h4>Vector and cut space</h4>
<p>After the curved surface, we started to study the tangent of the curved plane, which he could see as the contours of the curved vector at the given point, so we started with the vector.</p>
<p>Definitions: Establishment&#36;S\subset \mathscr{A}^3&#36; The blogger says that the government is not a party to the law.&#36;A\in S&#36;  Set
&#36;&#36;\varphi:\left(-\varepsilon,\varepsilon\right)\times\left(-\varepsilon,\varepsilon\right)\rightarrow S,\left(u,v\right)\mapsto\varphi\left(u,v\right)&#36;&#36;
For the point&#36;A&#36;Localisation of nearby parameters with &#36;f\inmathcal{F}<em>{\mathrm{A&#125;&#125;&#36; 是点&#36;A&#36;-defined smooth function, definition
&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;</em>{u}\left(A\right)f:=\frac{\partial\left(f\circ\varphi\right)}{\partial u}\right|<em>- I'm sorry.
Called \\partial</em>{u}\left(A\right)&#36; 为曲面&#36;S&#36;在&#36;A vector of A&#36; points</p>
<p>We can prove it.&#36;\partial_{u}\left(A\right)&#36;It's a conductor. It's vector space.&#36;T_A&#36;in the elements. And so...&#36;T_A&#36;All of them can be seen as defined. <strong>The sum of the algorithms that guide the first parameter under a local parameterisation</strong>  It's called the Curve.&#36;S&#36;Yes.&#36;A&#36;- I'll cut the space.&#36;T_{A}\left(S\right)&#36;</p>
<p>All you need is the vector that any local parameter can get.&#36;\partial_{u},\partial_{v}&#36;And it's a linear combination of all the vectors, which is...
&#36;&#36;T_{A}\left(S\right)=Span\left{\partial_{u},\partial_{v}\right}&#36;&#36;</p>
<h4>Vector Field</h4>
<p>The vector field is also a vector field, where each point on the curve specifies a vector field, which forms; a vector field meets a regularity at a point, only with its weight in a local parameter frame (simulation frame)</p>
<h3>Cotangent Space and Micro-Species of Curve</h3>
<p>The subspace of cut-off vector space, this section we want to do the same thing about vector space for the metrospace — the residual vector space.</p>
<p>Since the left space itself does not have a vector-based visualization, the mathematical theory of this section is omitted and only understood.</p>
<h3>Maps between curves and cut maps</h3>
<p>In this section we discuss the local nature of the mapping between the curves in the 3D analogue space.</p>
<h4>Map between Curves</h4>
<p>Definitions: Establishment&#36;S,S^\prime\in\mathscr{A}^3&#36;For two smooth faces.&#36;,A\in S,A^\prime\in S^{\prime}.&#36; Set &#36;F:S\mapsto S^\prime&#36;For a map, satisfy&#36;F(A)=A^\prime&#36;If you're in &#36;A,A^\prime&#36;There's a difference.&#36;S,S^{\prime}&#36;Localisation of Parameters&#36;{U,\phi_{U} }\left{U^{\prime},\varphi_{U^{\prime&#125;&#125;\right}&#36;  Satisfied&#36;\varphi_{U}\left(0,0\right)=A,\varphi_{U^{\prime&#125;&#125;\left(0,0\right)=A^{\prime}.&#36; If
&#36;&#36;\varphi_{U}^{-1}\circ F\circ\varphi_{U}:\left(-\varepsilon,\varepsilon\right)\times\left(-\varepsilon,\varepsilon\right)\longmapsto U\cap S\xrightarrow{F}U^{\prime}\cap S^{\prime}\xrightarrow{\varphi_{U}^{-1&#125;&#125;\left(-\varepsilon,\varepsilon\right)\times\left(-\varepsilon,\varepsilon\right)&#36;&#36;
Yes.&#36;\mathbb{R}^2&#36;Upset to&#36;\mathbb{R}^2&#36;Continuous/micro-lightly slurable, otherwise called &#36;F&#36;Yes&#36;A&#36; Continuous/micro-slowly-slowly map near point.</p>
<p><strong>This definition relies on the specific local parameterization that actually holds true for any single local parameterization.</strong></p>
<p>It's natural to study the idea under the imitation coordinates.</p>
<p>Definitions: Establishment&#36;A=\left{O,e_i\right}&#36;Yes &#36;\mathscr{A}^3&#36;and the analogue coordinates in it,&#36;S^\prime\subset \mathscr{A}^3&#36;Set for a smooth face &#36;S&#36;Yes &#36;\mathscr{A}^3&#36;The middle side.&#36;A\in S,F:S\mapsto S^\prime&#36;A map between the two sides&#36;,f(A)=A^\prime\in S^{\prime}.&#36; &#36;\left{U,\varphi_U\right}&#36;Yes&#36;A&#36;Near the dot.&#36;S&#36;Localisation of parameters, as the argument map is
&#36;&#36;\varphi_{U}:\left(-\varepsilon,\varepsilon\right)\times\left(-\varepsilon,\varepsilon\right)\mapsto U\cap S,并且\varphi_{U}\left(0,0\right)=A&#36;&#36;
Well...&#36;f&#36;Yes.&#36;A&#36;Continuous/slipper/micro and only &#36;\exist0&lt;\varepsilon
&#36;&#36;\varphi_{A}\circ F\circ\varphi_{U}:\left(-\varepsilon^{\prime},\varepsilon^{\prime}\right)\times\left(-\varepsilon^{\prime},\varepsilon^{\prime}\right)\mapsto R^{3}&#36;&#36;
Yes.&#36;(0,0)&#36;Continuous/slid/sweet</p>
<h4>A cut-in-the-flip map</h4>
<p>Now we're going to study it.&#36;F&#36;♪ Guide map, it'll be a&#36;T_A(S)&#36;Present.&#36;T_{A^{\prime&#125;&#125;(S^{\prime})&#36;The linear map.</p>
<p>First of all, &#36;g\in\mathcal{F}_{\mathcal{A}^{\prime&#125;&#125;(S^{\prime})&#36;That's... &#36;g&#36; Yes. &#36;S^{\prime}&#36;Go, go, go! &#36;A^\prime&#36;There's a defined smooth function nearby, and we can always use the following method to "reciprocate."&#36;S&#36;Go, go, go!&#36;A&#36;A defined smooth function nearby:
&#36;&#36;F^{*}\left(g\right):=g\circ F.&#36;&#36;
This operation is called "will"&#36;g&#36;Pass.&#36;F&#36;Pull back.&#36;S&#36;”.</p>
<p>Then, set &#36;D\in T_A(S)&#36;For a conductor, then.&#36;D&#36;It can be pulled back to the operation.&#36;g&#36; Go, go, go!
&#36;F <em>}\left(D\right)\left(g\right):=D\left(F^{</em>♪ Let's go, let's go, let's go, let's go, let's go, let's go, let's go, let's go, let's go, let's go, let's go, let's go, let's go, let's go, let's go, let's go, let's go, let's go, let's go, let's go, let's go, let's go, let's go, let's go, let's go, let's go, let's go,
This operation is called "will"&#36;D&#36;Launch to&#36;T_{A^{\prime&#125;&#125;S^{\prime}.&#36;  We can prove it was so launched.&#36;F_*(D)&#36;Yes.&#36;F_{\mathcal{A}^{\prime&#125;&#125;(S^{\prime})&#36; A guide to the</p>
<p>That means, "by&#36;F&#36;This operation actually created one.&#36;T_A(S)&#36;Present.&#36;T_{A^{\prime&#125;&#125;(S^{\prime})&#36;We call it a linear map.&#36;F&#36;Yes.&#36;A&#36;Point cut map. As&#36;TF_A&#36;or&#36;dF_A&#36;  <strong>The linear nature of the map is not supported here</strong></p>
<p><strong>Especially if we're looking at the curves to the map of the curves, if they define the vector field (in which case, the curves are defined).&#36;X&#36;Like the vector field, the map.&#36;F&#36;And naturally, a vector field is created, which is the same normal and the same as the original vector field.</strong></p>
