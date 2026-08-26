---
title: 'Analytic Geometry: Vectors, Coordinates, and Planes'
title_zh: 解析几何：向量、坐标与平面与直线
date: 2024-12-12 15:47:21 +0800
categories:
- Mathematics
- Geometry & Topology
tags:
- Analytic Geometry
author: Hyacehila
mathjax: true
hidden: true
excerpt: Covers vectors, coordinates, planes and lines, surfaces, quadrics, and foundations of spatial analytic geometry.
description: Covers vectors, coordinates, planes and lines, surfaces, quadrics, and foundations of spatial analytic geometry.
excerpt_zh: 整理向量、坐标、平面与直线、曲面、二次曲面和空间解析几何基础。
permalink: /blog/2024/12/12/analytic-geometry-notes/
lang: en
translation_key: 2024-12-12-analytic-geometry-notes
translation_status: machine
translation_source_hash: bf425f5b38e372247b9a75e3deecde37a4ab16f24179cad83e4edc853b25c95b
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<p>The geometry is the geometry of our research again, and the first geometry in the higher mathematical system, so here we will briefly describe what geometry really contains.</p>
<p>The questions in this article can also be addressed<a href="/en/blog/2024/10/16/point-set-topology-notes/">• Point-and-take: space, continuity and subspace</a>、<a href="/en/blog/2025/02/04/differential-geometry-notes/">Micro-specific geometry: geometric classification, curve theory and curve Comment</a>How the concept of a relatively close read together is developed in different contexts.</p>
<p>In fact, traditional geometry is broadly divided into European geometry (specified geometry) and non-European geometry; most people study only the former, and their differences are based on the difference in justice.</p>
<p>The first lesson in European geometry is geometry in primary and lower secondary school content, where we first get a preliminary view of the graphics and then extrapolate it on the basis of European geometry, and we come up with a number of important conclusions.</p>
<p>The second lesson in European geometry is in high school mathematics, where we study quantitative geometry, that is, the analysis of geometry, and geometry in the system under the coordinates. This is also the subject of discussion that we wish to continue in this part.</p>
<p>The third lesson in European geometry will be more abstract, and we will discuss the issue of micro-score flow and establish a more general geometry theory, which will be left for later discussion.</p>
<p>The three parts are discussing geometry under the same set of principles, but we have used a number of different research methods, so they have different uses.</p>
<h2>Vector & Coordinates</h2>
<h3>Vector concept</h3>
<p>We repeat in this section a few definitions that have been used for a long time as a definition of accepted geometry.</p>
<p>Definitions: Volumes of both size and direction are called vectors, or vectors, short of target.</p>
<p>The size of the vector is called a vector model, also known as the length of the vector; the vector&#36;\overrightarrow{AB}&#36;and&#36;a&#36; ♪ The model is written separately ♪&#36;\mid\overrightarrow{AB}\mid&#36;and&#36;\mid a\mid.&#36;</p>
<p>The vector of the Moot equals 1 is called the unit vector, and the vector&#36;a&#36;The vector with the same direction is called the vector.&#36;a&#36; , commonly used&#36;a^0&#36; Come on.</p>
<p> The vector of the model equals 0 is called zero. &#36;\mathbf{0}&#36;It is a vector that overlaps the starting and end points, and the direction of the zero-way is variable.</p>
<p>Definition: If the two vectors are identical and have the same direction, then it is called the same vector, all zero-directional
The same amount, vector.&#36;a&#36;and&#36;b&#36;Equal, remember.&#36;a=b.&#36;</p>
<p>Definitions: If two vector models are identical but in the opposite direction, they are referred to as reverses. &#36;a=-a&#36;</p>
<p>Definition: A set of vectors parallel to the same line is called the conlinear vector. Zero vectors are co-lined with any conlinee vector.</p>
<p>Definitions: A set of vectors parallel to the same plane is called a co-direction. Zero vectors are sidelined with any one of the concoction vector groups.</p>
<h3>Add and Multiply</h3>
<p>Definitions: Set known vectors &#36;a,b&#36;Take any space you want. &#36;o&#36; Make a vector for start-up points and then again.&#36;\overrightarrow OA=\boldsymbol{a},\overrightarrow{AB}=\boldsymbol{b}&#36; We'll need a line.&#36;OAB&#36;from the end of the line&#36;O&#36;To another end&#36;B&#36;Vector&#36;\overrightarrow OB=c&#36;It's called a two-way.&#36;a&#36;and&#36;b&#36;And, remember&#36;c=a+b&#36;...for two vectors&#36;a&#36;and&#36;b&#36;and&#36;a+b&#36;And it's called a vector addition, and it's a triangle.</p>
<p>Theoretically: If two vectors&#36;\overrightarrow{OA},\overrightarrow{OB}&#36;A parallel quadrilateral for the next side. &#36;OACB&#36; , the diagonal vector&#36;\overrightarrow OC=\overrightarrow{OA}+\overrightarrow{OB}.&#36; Called the parallel quadrilateral law.</p>
<p>Theorem: vector plus meets the following operating rates</p>
<ul>
<li>&#36;a+b=b+a&#36;</li>
<li>&#36;(\begin{array}{c}a+b\end{array})+c=a+(\begin{array}{c}b+c\end{array})&#36;</li>
<li>&#36;a+0=a&#36;</li>
<li>&#36;a+(-a)=0&#36;</li>
</ul>
<p>Definitions: Vector equivalent&#36;b&#36;& With Vector&#36;c&#36;& The value of the vector&#36;a&#36;that is&#36;b+c=a&#36;♪ We put the vector ♪&#36;c&#36;It's called vector.&#36;a&#36;and&#36;b&#36;The bad, and remember to do&#36;c=a-b.&#36;From 2-way&#36;a&#36;and&#36;b&#36;I beg their failure.&#36;a-b&#36;The calculations are called vector reduction.</p>
<p>Finally, the vector addition can give the following important variations:
&#36;&#36;\mid a_1+a_2+\cdots+a_n\mid\leqslant\mid a_1\mid+\mid a_2\mid+\cdots+\mid a_n\mid&#36;&#36;</p>
<p>Definitions: actual&#36;\lambda&#36;& With Vector&#36;a&#36;The product is a vector, remember&#36;\lambda a&#36;♪ It's a model ♪&#36;|\lambda a|=&#36; &#36;|\lambda||a|;\lambda a&#36;, when &#36;lambda&gt;0&#36;时与&#36;a&#36;相同，当&#36;\lambda&lt;0&#36;时与&#36;A.A.A.A. We call this a multiplication of numbers and vectors, short of numbers.</p>
<p>Known Vector&#36;a&#36; And its vectors in units. &#36;a^{0}&#36; The following equations are clearly in place:
&#36;&#36;a=\mid a\mid a^0,\quad\text{或}\quad a^0=\frac a{\mid a\mid}.&#36;&#36;</p>
<p>Theorem: vector multipliers to meet the following operating rates</p>
<ul>
<li>&#36;1\cdot a=a&#36;</li>
<li>&#36;\lambda\left(\mu a\right)=\left(\lambda\mu\right)a&#36;</li>
<li>&#36;\left(\lambda+\mu\right)a=\lambda a+\mu a&#36;</li>
<li>&#36;\lambda\left(a+b\right)=\lambda a+\lambda b&#36;</li>
</ul>
<h3>Linear relation of vectors decomposition of vectors</h3>
<p>The weighting and multiplying of vectors and vectors are collectively called linear calculations of vectors. We know that a limited vector is run linearly, and its result is still a vector.</p>
<p>Definitions: By &#36;a {<em>1&#125;&#125;,a</em>&#123;&#123;<em>2&#125;&#125;,\cdots,a</em>&#123;&#123;<em>n&#125;&#125;&#36;与实数&#36;\lambda_1,\lambda</em>&#123;&#123;<em>2&#125;&#125;,\cdots,\lambda</em>Vectors made up of &#36;
&#36;&#36;a=\lambda_1a_1+\lambda_2a_2+\cdots+\lambda_na_n&#36;&#36;
It's called vector. &#36;a_1,a_2,\cdots,a_n&#36; The linear combination.</p>
<p>Theorem: If vector&#36;e\neq0&#36;, then vector&#36;r&#36;& With Vector&#36;e&#36;The confederate requirement&#36;r&#36;Use vectors&#36;e&#36; Linear representation, or &#36;r&#36; Yes.&#36;e&#36; , which is the linear combination&#36;r= x\boldsymbol{e}&#36; and coefficient&#36;x&#36;By &#36;e,r&#36; Only sure; at this point we call&#36;e&#36;It's Kee.</p>
<p>Theorem: If vector&#36;e_1,e_2&#36;Not conjunctive. So vector.&#36;r&#36;and&#36;e_1,e_2&#36;The full condition of the joint is&#36;r&#36;Yeah.
By Vector&#36;e_1,e_2&#36;Linear, or vector&#36;r&#36;It's decomposed.&#36;e_1,e_2&#36;, which is the linear combination
&#36;&#36;r=x\boldsymbol{e}_1+y\boldsymbol{e}_2,&#36;&#36;
And coefficient.&#36;x,y&#36;By&#36;e_1,e_2,r&#36;Only sure. And then...&#36;e_1,e_2&#36;It's called a base of the plane vector.</p>
<p>Theorem: If vector&#36;e_1,e_2,e_3&#36;No one else. So, space is free.&#36;r&#36;It can be by vector&#36;e_1,e_2,e_3&#36;
Linear, or space at any vector.&#36;r&#36;It's decomposed into vectors.&#36;e_1,e_2,e_3&#36;Linear combination, i. e.
&#36;&#36;r=xe_1+ye_2+ze_3,&#36;&#36;
And the coefficient.&#36;x,y,z&#36;By&#36;e_1,e_2,e_3,r&#36;Only for sure. &#36;e_1,e_2,e_3&#36; It's called a space vector base.</p>
<p>Definitions: Yes&#36;n&#36; (&#36;n\geqslant1)&#36;Vectors &#36;a 1,a 2,\cdotp\cdotp,\bardsymbol{a}<em>n&#36;,如果存在不全为零的&#36;n&#36;个数&#36;\lambda_1&#36;,
&#36;\lambda_2,\cdots,\lambda_n&#36;, makes &#36;lambda</em>{1}a_{1}+\lambda_{2}a_{2}+\cdots+\lambda_{n}a_{n}=0&#36;,那么&#36;n&#36;个向量&#36;a 1, a 2, \\cdotp\cdotp\cdotp, a n&#36; is called linear, not linearly related vectors are called linear and irrelevant.</p>
<p>Inference: a vector&#36;a&#36;Linear Related Filling Conditions&#36;a=0.&#36;</p>
<p>Theoretically:&#36;n\geqslant2&#36;, vector&#36;a_1,a_2,\cdots,a_n&#36;Linear related filling condition is one of them.
The amount is the linear combination of the rest of the vector.</p>
<p>Theorem: Partial vector linear relevance in a set of vectors, which is linearly related</p>
<p>Inference: vector group linearally relevant with zero vectors</p>
<p>Theoretically: The two vectors co-lines are fully charged to their linear nature, the three vectors are linear to their linear nature and the four or more vectors in space are always linear to their linear nature.</p>
<h3>Tabs and Coordinates</h3>
<p>Definition: a fixed point in space&#36;O&#36; with three different orderly vectors&#36;e_1,e_2,e_3&#36;The whole thing, called a frame in space, remember&#36;{O;e_1,e_2,e_3}&#36;,</p>
<p>If&#36;e_1,e_2,e_3&#36;It's all a unit vector, then.&#36;{O;e_1,e_2,e_3}&#36;It's called the Decalar frame.   &#36;e_1,e_2,e_3&#36;Two or two vertical cartesian frames are called cartesian straight-angle frames, short of straight-angled frames;&#36;{O;e_1,e_2,e_3}&#36; It's called a simulation frame.</p>
<p>Once a frame is built in space, any vector in space can be broken down as follows:
&#36;&#36;r=xe_1+ye_2+ze_3&#36;&#36;</p>
<p>Definitions: in the form&#36;x,y,z&#36;Called a vector.&#36;r&#36;The coordinates, take them. &#36;r\left{x,y,z\right}&#36; or&#36;\left{x,y,z\right}.&#36;</p>
<p>Definitions: For the extraction of frames &#36;{O;e_1,e_2,e_3}&#36; Any point in space &#36;P&#36;vector&#36;\overrightarrow{OP}&#36;It's called a dot. &#36;P&#36; , or points&#36;P&#36;Position vector. Path&#36;\overrightarrow OP&#36;About the frame&#36;{O;\boldsymbol{e}_1,\boldsymbol{e}_2,\boldsymbol{e}_3}&#36;Coordinates&#36;x,y,z&#36;It's called a dot.&#36;P&#36;The coordinates of the frame</p>
<p>We can use coordinates to run vectors. Here, re-write the additions and multipliers and some of the the theorems.</p>
<p>Theorem: the coordinates of the vector are the same as the coordinates of the endpoint of its coordinates minus the starting point; the coordinates of the two vectors and the coordinates of the coordinates are the same as the sum of the coordinates; the coordinates of the number of vectors are the same as the sum of the corresponding coordinates of the number and the vector.</p>
<p>Theorem: two non-zero vectors &#36;a{X_1,Y_1,Z_1},{b}{X_2,Y_2,Z_2}&#36;The co-line is required to be proportional to the coordinates, i.e.
&#36;&#36;\frac{X_1}{X_2}=\frac{Y_1}{Y_2}=\frac{Z_1}{Z_2}.&#36;&#36;</p>
<p>Theorem: three non-zero vectors&#36;a\left{X_1,Y_1,Z_1\right},\boldsymbol{b}\left{X_2,Y_2,Z_2\right}&#36;and&#36;c{X_3,Y_3,Z_3}&#36;Co-Performance
- You're gonna have to.
&#36; \begin{vmatrix}X 1&amp;Y_1&amp;Z_1\X_2&amp;Y_2&amp;Z_2\X_3&amp;Y_3&amp;Z_3\end{vmatrix}=0.&#36;&#36;</p>
<p>Theorem: with a branch of direction&#36;\overrightarrow P_{1}\overrightarrow{P_{2&#125;&#125;&#36;The beginning of the event is&#36;P_1(x_1,y_1,z_1)&#36;♪ The end is ♪&#36;P_2(x_2,y_2,z_2)(&#36;Figure 1-25), then split in a directional segment &#36;P_1P_2&#36; Ratio&#36;\lambda(\lambda\neq-1)&#36;Points &#36;P&#36; The coordinates are...
&#36;&#36;x=\frac{x_{1}+\lambda x_{2&#125;&#125;{1+\lambda},\quad y=\frac{y_{1}+\lambda y_{2&#125;&#125;{1+\lambda},\quad z=\frac{z_{1}+\lambda z_{2&#125;&#125;{1+\lambda}.&#36;&#36;
So the midpoint coordinates are
&#36;&#36;x=\frac{x_{1}+x_{2&#125;&#125;{2},\quad y=\frac{y_{1}+y_{2&#125;&#125;{2},\quad z=\frac{z_{1}+z_{2&#125;&#125;{2}.&#36;&#36;</p>
<h3>Axis projection of vectors</h3>
<p>Set up a little bit of the known space.&#36;A&#36;And an axis&#36;l&#36;, pass&#36;A&#36;Do Vertically on Axes&#36;l&#36;, and then click the&#36;\alpha&#36;♪ We'll put this plane on ♪
Axis&#36;l&#36;, and then click the & Node&#36;A^\prime&#36;It's called a dot.&#36;A&#36;On the axis.&#36;l&#36;The reflection on the top.</p>
<p>Definitions: vectors&#36;\overrightarrow AB&#36;The beginning of the day.&#36;A&#36;And the end.&#36;B&#36;On the axis.&#36;l&#36;The reflections are on the dots.&#36;A^\prime&#36;and&#36;B^{\prime}&#36;Well, then...
Vector&#36;\overrightarrow{A^{\prime}B^{\prime&#125;&#125;&#36;It's called vector.&#36;\overrightarrow{AB}&#36;On the axis.&#36;l&#36;The insulated vector on it.&#36;\overrightarrow{AB}.&#36; And call it a shadow.</p>
<p>Theorem: Vector&#36;\overrightarrow{AB}&#36;On the axis. &#36;l&#36; The above-image is equal to the cosine of the cone of the cone of the cone of the vector and the angle of the condensed angle of the vector:
&#36;&#36;\text{射影}_l\overrightarrow{AB}=\left|\overrightarrow{AB}\right|\cos\theta,\quad\theta=\angle\left(l,\overrightarrow{AB}\right)&#36;&#36;
Theorem: For any vector &#36;a,b&#36;It's a reflection.&#36;_l(a+b)=\text{射影}_l a+&#36;Insight&#36;_l b&#36;</p>
<p>Theorem: For any vector&#36;a&#36;and any actual number&#36;\lambda&#36;I'm in.<em>I (\lambda a) =\lambda\text{</em>{l}a&#36;</p>
<h3>Volume of vectors</h3>
<p>Definitions: Two vectors&#36;a&#36;and&#36;b&#36;The model and the product of their cosine are called vectors.&#36;a&#36;and&#36;b&#36;Quantities (also known as in-house)&#36;a\cdot b&#36; or &#36;ab,&#36;That's...
&#36;&#36;a\cdot b=\mid a\mid\mid b\mid\cos\angle(a,b)&#36;&#36;
If &#36;b=a&#36;Then there is. &#36;a\cdot a=|a|^{2}.&#36; We're building up the numbers.&#36;a\cdot a&#36;It's called&#36;a&#36;Quantities squared and recorded&#36;a^2.&#36;</p>
<p>Theorem: Two-way&#36;a&#36;and&#36;b&#36;The requirement of vertically&#36;a\cdot b=0.&#36;</p>
<p>Theorem: Volume of vectors meets the following pattern of operation</p>
<ul>
<li>&#36;a\cdot b=b\cdot a&#36;</li>
<li>&#36;(\lambda a)\cdot b=\lambda(a\cdot b)=a\cdot(\lambda b)&#36;</li>
<li>&#36;(\begin{array}{c}a+b\end{array})\cdot c=a\cdot c+b\cdot c&#36;</li>
<li>&#36;a\cdot a=a^2&gt;0\quad(a\neq0)&#36;</li>
<li>&#36;(\lambda\boldsymbol{a}+\mu\boldsymbol{b})\cdot\boldsymbol{c}=\lambda(\boldsymbol{a}\cdot\boldsymbol{c})+\mu(\boldsymbol{b}\cdot\boldsymbol{c})&#36;</li>
</ul>
<p>We have the following quantitative formula in the coordinates system.</p>
<p>Theorem: Set&#36;\text{}a=X_{1}\boldsymbol{i}+Y_{1}\boldsymbol{j}+Z_{1}\boldsymbol{k},\boldsymbol{b}=X_{2}\boldsymbol{i}+Y_{2}\boldsymbol{j}+Z_{2}\boldsymbol{k}&#36; then
&#36;&#36;a\cdot b=X_1X_2+Y_1Y_2+Z_1Z_2&#36;&#36;
And there is.
&#36;&#36;a\cdot i=X_1,\quad a\cdot j=Y_1,\quad a\cdot k=Z_1&#36;&#36;</p>
<p>Theorem: Set &#36;\text{}a=X_{}\boldsymbol{i}+Y_{}\boldsymbol{j}+Z_{}\boldsymbol{k}&#36; Well...
&#36;&#36;\mid a\mid=\sqrt{a^{2&#125;&#125;=\sqrt{X^{2}+Y^{2}+Z^{2&#125;&#125;.&#36;&#36;</p>
<h2>Theorem: two points in space &#36;P_1(x_1,y_1,z_1),P_2(x_2,y_2,z_2)&#36; The distance between is...
&#36;&#36;\sqrt{\left(x_{2}-x_{1}\right)^{2}+\left(y_{2}-y_{1}\right)^{2}+\left(z_{2}-z_{1}\right)^{2&#125;&#125;.&#36;&#36;</h2>
<p>An angle of the vector and the axis of the coordinates (or coordinates) is called an angle of the vector, and the cosine of the direction is called the direction cosine of the direction. The direction of a vector can be determined by its directional angle. The direction cosine of the vector can also be expressed by the coordinates of the vector.
Theorem: Non-zero vector&#36;a=Xi+Yj+Zk&#36;The chord of direction is
&#36;&#36;\begin{matrix}
 \cos\alpha=\frac{X}{\mid a\mid}=\frac{X}{\sqrt{X^{2}+Y^{2}+Z^{2&#125;&#125;}\
 \cos\beta=\frac{Y}{\mid a\mid}=\frac{Y}{\sqrt{X^{2}+Y^{2}+Z^{2&#125;&#125;}\
\cos\gamma=\frac{Z}{\mid a\mid}=\frac{Z}{\sqrt{X^{2}+Y^{2}+Z^{2&#125;&#125;}
\end{matrix}&#36;&#36;
And there is.
&#36;&#36;\cos\alpha + \cos\beta + \cos\gamma = 1&#36;&#36; in the&#36;\alpha,\beta,\gamma&#36;Vectors, vectors, vectors, vectors, vectors, vectors, vectors, vectors, vectors, vectors, vectors, vectors, vectors, vectors, vectors, vectors, vectors, vectors, vectors, vectors, vectors, vectors, vectors, vectors, vectors, vectors, vectors, vectors, vectors, vectors, vectors, vectors, vectors, vectors, vectors, vectors, vectors, vectors, vectors, vectors, vectors, vectors, vectors, vectors, vectors, vectors, vectors, vectors, vectors, vectors, vectors, vectors, vectors, vectors, vectors, vectors, vectors, vectors, vectors, vectors, vectors, vectors, vectors, vectors, vectors, vectors, vectors, vectors, vectors, vectors, vectors, vectors, vectors, vectors, vectors, vectors, vectors, vectors, vectors, vectors, vectors, vectors, vectors, vectors, vectors,&#36;a&#36;and&#36;x&#36;Axis&#36;,\gamma&#36;Axis&#36;,z&#36;Angle of axes, or vector&#36;a&#36;Three angles.</p>
<p>Theorem: two non-zero-receivables in space are &#36;\boldsymbol{a}\left{X_1,Y_1,Z_1\right}&#36;and&#36;\boldsymbol{b}\left{X_2,Y_2,Z_2\right}&#36;♪ Then they ♪
&#36;&#36;\cos\angle(a,b)=\frac{a\cdot b}{\mid a\mid\mid b\mid}=\frac{X_1X_2+Y_1Y_2+Z_1Z_2}{\sqrt{X_1^2+Y_1^2+Z_1^2}\cdot\sqrt{X_2^2+Y_2^2+Z_2^2&#125;&#125;.&#36;&#36;</p>
<p>Inference: The vector is highly vertically dependent
&#36;&#36;X_1X_2+Y_1Y_2+Z_1Z_2=0.&#36;&#36;</p>
<h3>Vector Volume</h3>
<p>Definitions: two-way&#36;a&#36;and&#36;b&#36;And the vector volume (also known as the external volume) is a vector, remembering&#36;a\times b&#36; It's a model.
&#36;&#36;\mid a\times b\mid=\mid a\mid\mid b\mid\sin\angle(a,b),&#36;&#36;
It's direction with&#36;a&#36;and&#36;b&#36;Both vertically, and press&#36;a,b,a\times b&#36;This sequence forms the right hand frame.</p>
<p>Theorem: vector of the non-coalition&#36;a,b&#36; a vector volume equivalent to the area of the parallel quadrilaterals made up of its sides.</p>
<p>Theorem: the two vectors of the conglomerate are charged to&#36;a\times b=0&#36;</p>
<p>Theorem: Inverse exchange of vector volume &#36;a\times b=-\left(b\times a\right)&#36;</p>
<p>Theorem: Vector-cum-Gree Fulfill factor combination rate
&#36;&#36;\lambda(a\times b)=(\lambda a)\times b=a\times(\lambda b)&#36;&#36;
Inference: The factor combination rate is as follows:
&#36;&#36;(\lambda\boldsymbol{a})\times(\mu\boldsymbol{b})=(\lambda\mu)(\boldsymbol{a}\times\boldsymbol{b})&#36;&#36;</p>
<p>Theorem: vector volume satisfaction distribution rate
&#36;&#36;(a+b)\times c=a\times c+b\times c&#36;&#36;
The exchange sequence is inference.
&#36;&#36;c\times(a+b)=c\times a+c\times b&#36;&#36;</p>
<h3>Vector Mixed</h3>
<p>If we take the vector first,&#36;a&#36; and&#36;b&#36; Make vector volume &#36;a\times b&#36; So this vector is still a third vector. &#36;c&#36; A further quantitative or vector accumulation, in the former case&#36;(a\times b)\cdot c&#36;  And the latter is the case. &#36;(a\times b)\times c&#36;  We'll discuss this in the last two sections of this chapter.</p>
<p>Definition: Three vectors of given space&#36;a,b,c&#36;, if first two vectors&#36;a&#36;and&#36;b&#36;..the vector volume, the vector to be taken, and the third vector to be taken.&#36;c&#36; The amount of quantity that you get is called a three-way.&#36;a,b,c&#36; "The mix, remember&#36;(a\times b)\cdot c&#36; or&#36;(a,b,c)&#36;or&#36;(abc).&#36;</p>
<hr>
<p>Now, let's start with the nature of the mix.</p>
<p>Theorem: three different degrees of orientation&#36;a,b,c&#36;The absolute value of the mixture equals the value of the&#36;a,b,c&#36;The volume of the parallel hexadecahedron that is elastic&#36;V&#36;♪ And when ♪&#36;a,b,c&#36;(a) The compound is positive when forming the right hand line;&#36;a,b,c&#36;When it's made of the left hand, the mix is negative.
&#36;&#36;(abc)=\varepsilon V&#36;&#36;When?&#36;a,b,c&#36; It's the right hand. &#36;\varepsilon=1;&#36;When?&#36;a,b,c&#36; It's when you're tied up on your left hand. &#36;\varepsilon=-1.&#36;</p>
<p>Theoretically:&#36;\text{三向量 }a,b,c\text{ 共面的充要条件是}(abc)=0.&#36;</p>
<p>Theoretically: Rotation of the three factors in the mix does not change its value, and any two factors are to be changed
+Cype, i.e.
&#36;&#36;(\begin{array}{c}abc\end{array})=(\begin{array}{c}bca\end{array})=(\begin{array}{c}cab\end{array})=-(\begin{array}{c}bac\end{array})=-(\begin{array}{c}cba\end{array})=-(\begin{array}{c}acb\end{array}).&#36;&#36;
Inference:
&#36;&#36;(a\times b)\cdot c=a\cdot(b\times c).&#36;&#36;</p>
<p>For coordinates to indicate yes</p>
<p>Theorem: If there is&#36;&#36;\boldsymbol{a}=X_{1} \boldsymbol{i}+Y_{1} \boldsymbol{j}+Z_{1} \boldsymbol{k}, \boldsymbol{b}=X_{2} \boldsymbol{i}+Y_{2} \boldsymbol{j}+Z_{2} \boldsymbol{k}, \boldsymbol{c}=X_{3} \boldsymbol{i}+Y_{3} \boldsymbol{j}+Z_{3} \boldsymbol{k}&#36;&#36; Well...
&#36;&#36;US&#36;00\rray}
\text{, \text{
== sync, corrected by elderman ==
I'm sorry. &amp; Y_{1} &amp; Z_{1} \
X_{2} &amp; Y_{2} &amp; Z_{2} \
X_{3} &amp; Y_{3} &amp; Z_{3}
\end{array}\right|
\end{array}&#36;&#36;
因此，共面的充要条件为
&#36;&#36;\begin{vmatrix}X_1&amp;Y_1&amp;Z_1\\X_2&amp;Y_2&amp;Z_2\\X_3&amp;Y_3&amp;Z_3\end{vmatrix}=0&#36;&#36;</p>
<h3>Double vector volume of vector</h3>
<p>Definition: Three vectors of space, first the vector of two of which, then the vector of the proceeds and the number of the
Three vector volumes, and the final result remains the constant, called double vector volumes given three vectors; for example,&#36;(a\times b)\times c&#36;It's a three-way.&#36;a,b,c&#36;A double-variant volume.</p>
<p>The geometry of double vector accumulation can be summarized as follows:
&#36;&#36;(a\times b)\times c=(a\cdot c)b-(b\cdot c)a&#36;&#36;
It's not very important to study the nature of double vector volumes alone, so it's sufficient to use the theorem conversion.</p>
<h2>Tracks and Equations</h2>
<h3>Normal equations for flat curves and argument equations</h3>
<p>Here, the curves on the plane (including the line) are seen as a collection of points of a certain character.</p>
<p>On the plane where the coordinates are set, they are reflected in two coordinates above the curve.&#36;x&#36;and&#36;y&#36;Mutual constraints to be met, general equation
&#36;&#36;F\left(x,y\right)=0&#36;&#36;
To express</p>
<p>We can do this in other ways.
&#36;&#36;y=f(x)&#36;&#36;</p>
<p>Definitions: When a equation is relevant to a curve after the coordinates are taken on the plane
1 for equations&#36;(x,y)&#36;Coordinates of a point on the curve; 2 Coordinates of any point on the curve&#36;(x,y)&#36;Meet this equation, and it's called the equation of this curve, which is called the graphics of this equation.</p>
<p><strong>This definition is the core of the equation of the plane curve, and by definition we can convert the graphics to the equation.</strong></p>
<p>Definitions: if available&#36;t(a\leqslant t\leqslant b)&#36;♪ All possible values, paths ♪&#36;r(t)&#36;The end point is always on a curve; in turn, any point on this curve corresponds to the path where it ends, and this path can be&#36;t&#36;value&#36;t_0(a\leqslant t_0\leqslant b)&#36;It's a decision. Then put the expression.&#36;r(t)&#36; The vector equation of the parameters called the curve, where &#36;t&#36; For parameters.</p>
<p>Because the diameter can be measured by the projection of the axis, the more common parameter equation is
&#36;&#36;US&#36;&#36;\begin{cases}x=xbegin{matrix}t\end{matrix},\y\begin{matrix}t\end{matrix}&amp;\end{cases}\begin{pmatrix}a\leqslant t\leqslant b\end{pmatrix}.&#36;&#36;</p>
<p><strong>The parameters of the argument equation can be removed from the normal equation, and the normal equation can be rewritten by the right parameter.</strong></p>
<h3>Curve</h3>
<h4>Curved Accelerator Basis</h4>
<p>The meaning of the space curve is the same as the equation of the peace curve, which is the character of the point on the curve (as the trajectory of the point) after the space has set the coordinates.&#36;x,y&#36;and&#36;z&#36;The relationship formula, usually an equation.
&#36;&#36;F(x,y,z)=0&#36;&#36;
To express</p>
<p>We can also use the following forms.
&#36;&#36;z=f(x,y)&#36;&#36;</p>
<p>The basic definition of the curve equation is fully consistent with the curve equation, and this is not a separate description.</p>
<p><strong>Especially when there's no real number to satisfy the curve equation, we call it a fiction. Noodles.</strong></p>
<h4>Curve Arguments Aquar</h4>
<p>Very naturally, we can rephrase the basic formula of the following curve parameters, which is also achieved by projection of the diameter (very naturally, two parameters are required to form the equation of the parameters).
&#36;&#36;r(u,v)=x(u,v)e_{1}+y(u,v)e_{2}+z(u,v)e_{3}&#36;&#36;
Specific definitions continue to refer to the curve section, which is also generally written as
&#36; \begin{cases}x=x(u,v)\y(u,v)\z=z(u,v).&amp;\end{cases}&#36;&#36;</p>
<h4>Ball and column coordinates</h4>
<p>We're doing a lot of triangulation in high school, but the flat exchange format is simple and can easily be comprehensive. And here we need to study the exchange of space, and we have some of the more common techniques for exchanging dollars, and here we are.</p>
<p>If we look at the point on the curve as a point on a space ball, on the point.&#36;M&#36;And there is.
&#36;&#36;&#36;&#36;&#36;begin{gathered}\left\overrightarrow=right=(\rho\geqslant0\right),\gle QOP=\varphi(-\pi)&lt;\varphi\leq\pi),\\angle POM=\theta\left(-\frac{\pi}{2}\leq\theta\leq\frac{\pi}{2}\right)\end{gathered}&#36;&#36;
&#36;\theta&#36;The projection is a projection.&#36;\varphi&#36;It's a plane projection.</p>
<p>So we can project the equation of the parameters below.
&#36;&#36;00\&amp;x=\rho\cos\theta\cos\varphi,\&amp;y=\rho\cos\theta\sin\varphi,\&amp;z=\rho\sin\theta,\end{aligned}&#36;&#36;
<strong>The equation of parameters in this special parameter equation format called the ball (polar) coordinate system</strong> The reverse calculation formula is
&#36; \begin{cases}\sqrt{x^2+y^z^2,\cos\varpi=x\sqrt{x^2^y^,\sin\varpi=\sqrt{x^2+z^}.\theta=\scin\frac{z}{\sqrt{x^2+z}2}.}&amp;\end{cases}&#36;&#36;</p>
<p>We consider the pillars in space.&#36;z&#36;The axis is not projected, it's just a plane projection and a circle position, and we can easily give the equation of the parameters below.
&#36;&#36;00\begin{cases}x=ro\cos\varphi=y=r\\=&amp;\end{cases},&#36;&#36;
<strong>The parameter equations of this special parameter equation format called the column coordinate system</strong> The reverse calculation formula is
{\cHFFE7C5}&#36;begin}cases}rho=sqrt{x^y^,\cos\varpi=\sqrt=x^x^y^},\uz=.=sqrt{&#125;&#125; } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } = = = } } } = } } } } } } } = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = } } } } &amp;\end{cases}&#36;&#36;</p>
<h3>Space curve equation</h3>
<p>The space curve can be seen as the intersection of two space curves, that is, the conjunctort.
&#36; \begin{cases}F 1(x,y,z)=1(x,y,z)=2(x,y,z)=0&amp;You're not gonna get away with this?
His solution is the coordinates of the points on the space curve, given the equation's equivalence.<strong>The equation of the space curve is certainly not unique, and it can be expressed in different equation groups.</strong></p>
<p>We can also create the equation of parameters, which, because the equation limits one parameter, give the projection form
&#36;&#36;r(t)=x(t)e_1+y(t)e_2+z(t)e_3&#36;&#36;
<strong>The number of parameters is one less than the equation of the space curve.</strong> We usually recast the equation of the space curve parameters as:
&#36;&#36;\begin{cases}x=x\left(t\right),\y=y\left(t\right),\taft(a\leq \leq \b\right)\z=z\left(t\right)&amp;You're not gonna get away with this?
In fact, the equation of the parameters of the space curve is more user-friendly, and we are less connected.</p>
<h2>Line and Space</h2>
<h3>Square Square</h3>
<h4>Point-to-Square of Plane</h4>
<p>I've got a little space.&#36;M_0&#36;Vector with two uncoherent lines&#36;a,b&#36;Then pass it.&#36;M_0&#36;and with Vector&#36;a,b&#36;Parallel plane&#36;\pi&#36;Only one is identified, vector.&#36;a,b&#36;It's called a plane.&#36;\pi&#36;The vector of any of the two of them, obviously, is the same as the plane.&#36;\pi&#36;Parallel non-cooline vectors can be used as plane.&#36;\pi&#36;. Set&#36;M_0&#36;The path is...&#36;r_0&#36; Take any bit of the plane.&#36;M&#36;Path&#36;r&#36; We're easy to give.
&#36;&#36;r=r_0+ua+vb&#36;&#36;
It's because the difference in the direction is a one-way vector on the plane, and he must be a bit of a parallel in the peace side, which is called<strong>Vector parameter forms for the point-to-square equation of a plane</strong></p>
<p>Set
&#36;&#36;\bardsymbol{r}<em>{0}=\left{x</em>{0},y_{0},z_{0}\right},\boldsymbol{r}=\left{x,y,z\right};\boldsymbol{a}={X_{1},Y_{1},Z_{1&#125;&#125;,\boldsymbol{b}={X_{2},Y_{2},Z_{2&#125;&#125;&#36;&#36;
容易给出等价的点向式有
&#36;&#36;\begin{cases}x=x_0+X_1u+X_2v,\y=y_0+Y_1u+Y_2v,\z=z_0+Z_1u+Z_2v.&amp;You're not gonna get away with this?
It's called<strong>Coordinate parameter form for the point-to-square equation of a plane</strong></p>
<p>Depending on the conditions of the coming and the properties of vector volume, you can give them
&#36;&#36;(\begin{array}{c}r-r_0,a,b\end{array})=0&#36;&#36;
That's right.
&#36;&#36;\begin{vmatrix}x-x 0&amp;y-y_0&amp;z-z_0\\X_1&amp;Y_1&amp;Z_1\\X_2&amp;Y_2&amp;Z 2\end{vmatrix}=&#36;0
The two forms of the numeric variable are more general and we will be introducing it right back.</p>
<p>Specifically, we give the plane the cut-off and the three-point cut, which is a point-to-point presentation, as follows:</p>
<p>Three points.
&#36;&#36;\begin{vmatrix}x-x 1&amp;y-y_1&amp;z-z_1\\x_2-x_1&amp;y_2-y_1&amp;z_2-z_1\\x_3-x_1&amp;y_3-y_1&amp;z_3-z_1\end{vmatrix}=0&#36;&#36;
截距形式
&#36;&#36;\frac{x}{a}+\frac{y}{b}+\frac{z}{c}=1&#36;&#36;</p>
<h4>Normal equation for flat</h4>
<p>We introduced you to it.
&#36;&#36;\begin{vmatrix}x-x 0&amp;y-y_0&amp;z-z_0\\X_1&amp;Y_1&amp;Z_1\\X_2&amp;Y_2&amp;Z_2\end{vmatrix}=0&#36;&#36;
展开后整体就可以得到
&#36;&#36;Ax+By+Cz+D = &#36;0
We call it<strong>Normal equation for flat</strong> And the parameters can be calculated from point to point in this way.
&#36;&#36;\left. A=\left\begin\array}Y 1&amp;Z_1\\Y_2&amp;Z_2\end{array}\right.\right|,B=\begin{vmatrix}Z_1&amp;X_1\\Z_2&amp;X_2\end{vmatrix},C=\begin{vmatrix}X_1&amp;Y_1\\X_2&amp;Y_2\end{vmatrix},D=-\begin{vmatrix}x_0&amp;y_0&amp;z_0\\X_1&amp;Y_1&amp;Z_1\\X_2&amp;Y_2&amp;Z_2\end{vmatrix}.&#36;&#36;</p>
<p>Special, when the general pattern has the following special characteristics, the equation has some special characteristics.</p>
<ul>
<li>&#36;D=0&#36; Equivalent to plane through original</li>
<li>&#36;A,B,C\text{ 中有一为零}&#36;<ul>
<li>When?&#36;D\ne0&#36; plane parallels axle with coefficient 0</li>
<li>When?&#36;D=0&#36; Axes of zero for plane pass factor</li>
</ul>
</li>
<li>&#36;A,B,C\text{ 中有两个为零的情况}&#36;<ul>
<li>When?&#36;D\ne0&#36; plane parallels zero</li>
<li>When?&#36;D=0&#36; The plane is the plane with a zero coefficient.</li>
</ul>
</li>
</ul>
<h4>Point-Full equation of a plane</h4>
<p>If you give me a little space,&#36;M_0&#36;And a non-zero vector.&#36;n&#36;Then pass it.&#36;M_0&#36;and with Vector&#36;n&#36;Vertical flat
The surface is the only one that's been determined. We're putting a non-zero- vector vertically to the plane.&#36;n&#36;It's called the law vector of the plane.</p>
<p>Depending on the vertical nature, we can give the equation.
&#36;&#36;n\cdot(r-r_0)=0.&#36;&#36;
If you give the coordinates, there is.
&#36;&#36;A\left(x-x_0\right)+B\left(y-y_0\right)+C\left(z-z_0\right)=0&#36;&#36;
They all call it<strong>Point-Full equation of a plane</strong>  This form is not parameter-based, but equation-based.</p>
<p>It's easy to see if it's true.&#36;D=-(Ax_0+By_0+Cz_0)&#36; So there is.
&#36;&#36;Ax+By+Cz+D=0.&#36;&#36;
The blogger adds:<strong>Factor of normal equation &#36;A,B,C&#36; It's a number of coordinates of a pattern.</strong>It's the most important link between the two.</p>
<p>If using unit method vector, it is easy to give<strong>Fragmentation of plane</strong> Yes.
&#36;&#36;x\cos\alpha+y\cos\beta+z\cos\gamma-p=0.&#36;&#36;
To convert a normal equation into this form, you just need a normal equation multiplied by
&#36;&#36;\lambda=\frac{1}{\pm|n|}=\frac{1}{\pm\sqrt{A^2+B^2+C^2&#125;&#125;&#36;&#36;
We call it the French version of the factor.</p>
<h3>Area to Point Position</h3>
<p>There are only two relationships between the plane and the point, the point of the plane and the point of the plane, which meets the equation of the plane, and here we are primarily looking at the latter.</p>
<h4>Distance between point and plane</h4>
<p>When used<strong>Fragmentation of plane</strong> , give the distance formula as
&#36;&#36;d=|x_0\cos\alpha+y_0\cos\beta+z_0\cos\gamma-p|.&#36;&#36;
It's the difference between the absolute and the absolute.</p>
<p>When used<strong>Normal equation for flat</strong>, give the distance formula as
&#36;&#36;d=\frac{\mid Ax_0+By_0+Cz_0+D\mid}{\sqrt{A^2+B^2+C^2&#125;&#125;&#36;&#36;
It's the difference between the absolute and the absolute.</p>
<h4>Area</h4>
<p>Yeah.<strong>Normal equation for flat</strong> There's a difference.
&#36;&#36;\delta=\lambda\left(Ax+By+Cz+D\right)&#36;&#36;
Which means...
&#36;&#36;Ax+By+Cz+D=\frac{1}{\lambda}\delta.&#36;&#36;
For the point on the side of the plane, the difference is the same symbol, the opposite sign, which means that part of the point is &#36;Ax+By+Cz+D.&gt;0&#36; 另一部分点&#36;Ax+By+Cz+D&lt;The plane divides the space into two parts.</p>
<h3>Location Relationship between Plane</h3>
<p>There are three scenarios of the position of the two space levels in question, namely, intersection, parallel and overlap.
&#36;&#36;\pi_1:A_1x+B_1y+C_1z+D_1=0:,\\pi_2:A_2x+B_2y+C_2z+D_2=0:,&#36;&#36;
So, two planes.&#36;\pi_1&#36;and&#36;\pi_2&#36;Whether it is a cross or parallel or a overlap depends on whether the equations consist of a solvency or an insoluble equation, or whether they differ only by a non-zero factor, so we have the following theory:</p>
<p>The condition for the plane to be intersected is
&#36;&#36;A_1:B_1:C_1\neq A_2:B_2:C_2&#36;&#36;
Parallel Qualifications
&#36;&#36;\frac{A_1}{A_2}=\frac{B_1}{B_2}=\frac{C_1}{C_2}\neq\frac{D_1}{D_2}&#36;&#36;
Reconciling Qualifications
&#36;&#36;\frac{A_1}{A_2}=\frac{B_1}{B_2}=\frac{C_1}{C_2}=\frac{D_1}{D_2}&#36;&#36;</p>
<p>Especially, we can discuss the nature of the following.</p>
<p>Theoretically: two cross-platforms with a number of angles
&#36;&#36;begin{aligned}\cos\angle\left (1},\right)&amp;=\pm\cos\theta=\pm\frac{n_1\cdot n_2}{\mid n_1\mid\mid n_2\mid}\&amp;=\pm\frac{A_{1}A_{2}+B_{1}B_{2}+C_{1}C_{2&#125;&#125;{\sqrt{A_{1}^{2}+B_{1}^{2}+C_{1}^{2&#125;&#125;\sqrt{A_{2}^{2}+B_{2}^{2}+C_{2}^{2&#125;&#125;}.\end{aligned}&#36;&#36;
其中 &#36;n_1,n_2&#36; 是两个平面的法向量，特别的，两个平面垂直的充要条件为
&#36;&#36;A_1A_2+B_1B_2+C_1C_2=0.&#36;&#36;</p>
<h3>Space Line Square</h3>
<h4>Point-to-Package</h4>
<p>I've got a little space.&#36;M_0&#36;With a non-zero vector&#36;v&#36; Then pass it.&#36;M_0&#36;and with Vector&#36;v&#36;Parallel Lines
&#36;l&#36;Only one is identified, vector.&#36;v&#36;It's called a straight line.&#36;l&#36;Obviously, any line with a straight line&#36;l&#36;Parallel non-zero vectors can be used as straight lines&#36;l&#36;- The directional amount.</p>
<p>We'll give it straight to you.
&#36;&#36;r=r_0+t\boldsymbol{v}.&#36;&#36;
Called<strong>The vector equation of the straight line</strong>
&#36;&#36;\begin{cases}x=x_0+Xt,\\y=y_0+Yt,\\z=z_0+Zt.&amp;You're not gonna get away with this?
Called<strong>A coordinate equation of the line</strong></p>
<p>Remove Parameters&#36;t&#36; then
&#36;&#36;\frac{x-x_0}{X}=\frac{y-y_0}{Y}=\frac{z-z_0}{Z}&#36;&#36;
Called<strong>Standard equation for a straight line</strong></p>
<p>Special, which allows the two-point equation of a straight line to be launched
&#36;&#36;\frac{x-x_1}{x_2-x_1}=\frac{y-y_1}{y_2-y_1}=\frac{z-z_1}{z_2-z_1}.&#36;&#36;</p>
<h4>Normal equation for a straight line</h4>
<p>Line can be the line of the plane, the equation group
&#36;&#36;\pi_{1}:A_{1}x+B_{1}y+C_{1}z+D_{1}=0;\pi_{2}:A_{2}x+B_{2}y+C_{2}z+D_{2}=0&#36;&#36;
The equation group called<strong>Normal equation for a straight line</strong></p>
<p>We can calculate the standard equations based on the normal equations, as follows:
&#36;&#36;\frac{x-x 0}begin{vmatrix}B 1&amp;C_1\\B_2&amp;C_2\end{vmatrix&#125;&#125;=\frac{y-y_0}{\begin{vmatrix}C_1&amp;A_1\\C_2&amp;A_2\end{vmatrix&#125;&#125;=\frac{z-z_0}{\begin{vmatrix}A_1&amp;B_1\\A_2&amp;B_2\end{vmatrix&#125;&#125;.&#36;&#36;
其中
&#36;&#36;x_0=\frac{\begin{vmatrix}B_1&amp;D_1\\B_2&amp;D_2\end{vmatrix&#125;&#125;{\begin{vmatrix}A_1&amp;B_1\\A_2&amp;B_2\end{vmatrix&#125;&#125;,\quad y_0=\frac{\begin{vmatrix}D_1&amp;A_1\\D_2&amp;A_2\end{vmatrix&#125;&#125;{\begin{vmatrix}A_1&amp;B_1\\A_2&amp;B 2\end{vmatrix}, \z 0=&#36;0.00
So, we can just relax.<strong>Normal equation for a straight line</strong>Calculates a point on a straight line and its direction vector, and thus the standard equation of the straight line</p>
<h3>Lines to plane position</h3>
<p>There are three situations where the space line intersects with the plane, and the line parallels the plane and the line is on the plane. Now we're going to ask for conditions for the line to be set up at the plane's position....&#36;l&#36;& With Plane&#36;\pi&#36;The equations are the same.
&#36;&#36;l:\frac {x- x_0}X= \frac {y- y_0}Y= \frac {z- z_0}Z, \pi :Ax+ By+ Cz+ D= 0&#36;&#36;</p>
<p>Theorem: For intersectional
&#36;&#36;AX+BY+CZ\neq0&#36;&#36;
For parallels.
&#36;&#36;AX+BY+CZ=0&#36;&#36;
And...
&#36;&#36;Ax_0+By_0+Cz_0+D\neq0&#36;&#36;
For the line on the plane
&#36;&#36;AX+BY+CZ=0&#36;&#36;
And...
&#36;&#36;Ax_0+By_0+Cz_0+D=0&#36;&#36;
<strong>They're all making some calculations of the relationship between the normal vectors, including volume and vector. Jack.</strong></p>
<p>Theorem: In the case of intersections, the angle of the straight line and the plane is calculated as
&#36;&#36;\sin\varphi=\mid\cos\theta\mid=\frac{\mid n\cdot v\mid}{\mid n\mid\cdot\mid v\mid}=\frac{\mid AX+BY+CZ\mid}{\sqrt{A^2+B^2+C^2}\cdot\sqrt{X^2+Y^2+Z^2&#125;&#125;.&#36;&#36;</p>
<h3>Line to Point Location</h3>
<p>There are two situations in which the space line is related to the point, i.e. the point is on the line and the point is not on the line, and the point is on the line provided that the point coordinates satisfy the equation of the line. When the point is not on the line, let's get to the line.</p>
<p>We give the formula without proof.
&#36;d=\frac {sqrt{vmatrix}y 0-y 1&amp;z_0-z_1\Y&amp;Z\end{vmatrix}^2+\begin{vmatrix}z_0-z_1&amp;x_0-x_1\Z&amp;X\end{vmatrix}^2+\begin{vmatrix}x_0-x_1&amp;y_0-y_1\X&amp;Y\end{vmatrix}^2&#125;&#125;{\sqrt{X^2+Y^2+Z^2&#125;&#125;&#36;&#36;</p>
<h3>Location relation between lines</h3>
<h4>Line position relation</h4>
<p>The position of the two straight lines of space is different and common, and there are three situations in which there are intersections, parallels and overlaps. Now we're going to export the conditions for these positions to be established.</p>
<p>Theorem: the condition of the alien is
&#36;&#36;\Delta=\begin{vmatrix}x 2-x 1&amp;y_2-y_1&amp;z_2-z_1\\X_1&amp;Y_1&amp;Z_1\\X_2&amp;Y_2&amp;Z_2\end{vmatrix}\neq0&#36;&#36;
相交的充要条件为
&#36;&#36;\Delta=0,\quad X_1:Y_1:Z_1\neq X_2:Y_2:Z_2&#36;&#36;
平行的充要条件为
&#36;&#36;X_{1}:Y_{1}:Z_{1}=X_{2}:Y_{2}:Z_{2}\neq(x_{2}-x_{1}):(y_{2}-y_{1}):(z_{2}-z_{1})&#36;&#36;
重合的充要条件为
&#36;&#36;X_1:Y_1:Z_1=X_2:Y_2:Z_2=(x_2-x_1):(y_2-y_1):(z_2-z_1)&#36;&#36;</p>
<h4>A straight-lineed angle.</h4>
<p>In the arctic coordinate system, the cosine of the angle of the straight line meets
&#36;&#36;\cos\angle(l_{1},l_{2})=\pm\frac{X_{1}X_{2}+Y_{1}Y_{2}+Z_{1}Z_{2&#125;&#125;{\sqrt{X_{1}^{2}+Y_{1}^{2}+Z_{1}^{2&#125;&#125;\cdot\sqrt{X_{2}^{2}+Y_{2}^{2}+Z_{2}^{2&#125;&#125;}.&#36;&#36;
So you can push it out, and the condition is straight vertical.
&#36;&#36;X_1X_2+Y_1Y_2+Z_1Z_2=0.&#36;&#36;</p>
<h4>Distance of the line from the surface to the line from the public</h4>
<p>The distance of the straight line on the opposite side is the shortest distance above the point, equal to the length of their pyrophone line, and can be calculated using the following formula:
&#36;&#36;d=\frac{\mid(\overrightarrow{M_1M_2},\boldsymbol{v}_1,\boldsymbol{v}_2)\mid}{\mid\boldsymbol{v}_1\times\boldsymbol{v}_2\mid}&#36;&#36;
of which &#36;M_1,M_2&#36; It's a point up the line. &#36;\boldsymbol{v}_1,\boldsymbol{v}_2&#36; is the direction vector of the line; coordinates are as
&#36;d=\frac{vmatrix}x 2-x 1&amp;y_2-y_1&amp;z_2-z_1\X_1&amp;Y_1&amp;Z_1\X_2&amp;Y_2&amp;Z_2\end{vmatrix&#125;&#125;{\sqrt{\begin{vmatrix}Y_1&amp;Z_1\Y_2&amp;Z_2\end{vmatrix}^2+\begin{vmatrix}Z_1&amp;X_1\Z_2&amp;X_2\end{vmatrix}^2+\begin{vmatrix}X_1&amp;Y_1\X_2&amp;Y_2\end{vmatrix}^2&#125;&#125;.&#36;&#36;</p>
<p>Finally, we discuss the equation of the pyropathic line, and he's satisfied.
&#36;&#36;\begin{vmatrix}x-x 1&amp;y-y_1&amp;z-z_1\\X_1&amp;Y_1&amp;Z_1\\X&amp;Y&amp;Z\end{vmatrix}=0&#36;&#36;
以及
&#36;&#36;\begin{vmatrix}x-x_2&amp;y-y_2&amp;z-z_2\\X_2&amp;Y_2&amp;Z_2\\X&amp;Y&amp;Z\end{vmatrix}=0&#36;&#36;
是这两个平面的交线；其中
&#36;&#36;X=\begin{vmatrix}Y_1&amp;Z_1\Y_2&amp;Z_2\end{vmatrix},Y=\begin{vmatrix}Z_1&amp;X_1\Z_2&amp;X_2\end{vmatrix},Z=\begin{vmatrix}X_1&amp;Y_1\X_2&amp;Y 2\end{vmatrix}
Yes.&#36;v_1\times v_2&#36;   Which means... &#36;l_0&#36; Directions</p>
<h3>Plane</h3>
<p>Definition: Called by all planes in a straight line <strong>Axis plane beams</strong> The axis of the line called the plane beam.</p>
<p>Definitions: Summoning all planes in parallel with a straight line <strong>Parallel plane beams</strong></p>
<p>Theoretically: If two planes
&#36;&#36;\begin{array}{c}
\pi_{1}: A_{1} x+B_{1} y+C_{1} z+D_{1}=0 \
\pi_{2}: A_{2} x+B_{2} y+C_{2} z+D_{2}=0
\end{array}&#36;&#36;
Intersect with Line&#36;L&#36; Then pass.&#36;L&#36; ♪ with a axis beam and a axis of the equation ♪
&#36;&#36;l(A_1x+B_1y+C_1z+D_1)+m(A_2x+B_2y+C_2z+D_2)=0&#36;&#36;
of which &#36;l,m&#36; It's not all zero.</p>
<p>Theoretically: by plane&#36;\pi:Ax+By+Cz+D=0&#36; The equation for the parallel plane beams decided upon is
&#36;&#36;Ax+By+Cz+\lambda=0&#36;&#36;
of which&#36;\lambda&#36;It's a real number.</p>
<h2>Conic</h2>
<h3>Column</h3>
<h4>Normal pillar</h4>
<p>Definition: In space, the curves generated by parallel parallel lines that are in the same direction and intersect with a fixed curve are called columns, or columns, or lines that are straight lines, or lines in the same line, or lines in the same line, or lines in the same line, or lines in the same line, or lines in the same line, or lines in the same column.</p>
<p>Set the direct equation as
&#36; \begin{cases}F 1(x,y,z)=1(x,y,z)=2(x,y,z)=0&amp;\end{cases}&#36;&#36;
母线方向数为 &#36;(X,Y,Z)&#36;   设准线上任一点 &#36;M_1(x_1,y_1,z_1)&#36; 那么过点&#36;M_1&#36;的母线方程为
&#36;&#36;\frac{x-x_1}{X}=\frac{y-y_1}{Y}=\frac{z-z_1}{Z}&#36;&#36;
并且满足
&#36;&#36;F_1(x_1,y_1,z_1)=0,\quad F_2(x_1,y_1,z_1)=0.&#36;&#36;
根据四个方程 消去 &#36;M_1(x_1,y_1,z_1)&#36;  就可以得到
&#36;&#36;F(x,y,z) = &#36;0
The equation called the matrix and the line-set column</p>
<p>Theorem (column styl): In the space arctic system, the three-dollar equation with only two dollars (coordinates) represents a column with a parent line parallel to the same number of missing elements (coordinates) Axes.</p>
<p>So, the equations are all column-level.
&#36;&#36;\begin{gathered}\frac{x^{2&#125;&#125;{a^{2&#125;&#125;+\frac{y^{2&#125;&#125;{b^{2&#125;&#125;=1,\\frac{x^{2&#125;&#125;{a^{2&#125;&#125;-\frac{y^{2&#125;&#125;{b^{2&#125;&#125;=1,\y^{2}=2px.\end{gathered}&#36;&#36;
Because they're... &#36;xOy&#36;The projection is the ellipse, the hyperbolic parabolic line, so they also call it the elliptical column, the hyperbolic column, the parabolic column. It's called the quintessential.</p>
<h4>The reflection column of the space curve</h4>
<p>Set a Space Curve
&#36;L:\begin{cases}F(x,y,z)=0,\G(x,y,z)=0.&amp;\end{cases}&#36;&#36;
任意从中消去一个元，则有
&#36;&#36;F (x,y)=0, F (x,z)=0, F (y,z)=0&#36;0
According to the column-based theorem, they are all the same-named pillars of the axis of the missing object (coordinate), which is called <strong>The mirror side of the curve</strong> Curve
That's a good idea.&amp;You're not gonna get away with this?
It's called the Shadow Curve.</p>
<h3>Cone</h3>
<p>Definition: The curve created in space by a line of a clan that has a certain point and intersects with a fixed curve is called a cone
Face, these straight lines are called cone, the fixed point is called the cone's vertebrae, the curve is called the cone of the cone.</p>
<p>The line to which the cone is set is
&#36; \begin{cases}F 1(x,y,z)=1F 2(x,y,z)=0&amp;\end{cases}&#36;&#36;
顶点&#36;A\left(x_0,y_0,z_0\right)&#36;  设&#36;M_1(x_1,y_1,z_1)&#36;是准线上任意一点 则锥面过该点的母线为
&#36;&#36;\frac{x-x_0}{x_1-x_0}=\frac{y-y_0}{y_1-y_0}=\frac{z-z_0}{z_1-z_0}&#36;&#36;
并且
&#36;&#36;F_1(x_1,y_1,z_1)=0,\quad F_2(x_1,y_1,z_1)=0.&#36;&#36;
根据四个方程 消去 &#36;M_1(x_1,y_1,z_1)&#36;  就可以得到
&#36;&#36;F(x,y,z) = &#36;0
An equation called the cone determined by the vertex and the line</p>
<p>Theoretically (concentrated): one&#36;x,y,z&#36;The square equation always indicates the cone of the point of the coordinates at the origin Noodles.</p>
<p>Inference: one&#36;(x-x_0),(y-y_0),(z-z_0)&#36;The equations always indicate the top point&#36;x_0,y_0,z_0&#36;..of the cone</p>
<h3>Rotate Curve</h3>
<p>Definition: In space, a curve&#36;\Gamma&#36;♪ Round the straight line ♪&#36;l&#36;The curve generated by the rotation week is called rotation. Song
A face, or a curved curve.&#36;\Gamma&#36;It's called a conic, fixed line.&#36;l&#36;It's called the rotation axis of the conic, short for the axis.</p>
<p>The equation for setting the rotational curved master is
&#36;&#36;0.00: \begin{cases}F 1(x,y,z)=0, \F 2(x,y,z)=0,&amp;\end{cases}&#36;&#36;
旋转轴为
&#36;&#36;♪ I'm gonna be a little bit more than a little bit more than a little bit more than a little bit of a little bit of a ♪
Set&#36;M_1(x_1,y_1,z_1)&#36;It's the main line.&#36;\Gamma&#36;Up any point, then over the M-min.<em>1}&#36;的纬圆总可以看成是过&#36;M</em>{<em>1}&#36;且垂直于旋转轴&#36;l&#36;的平面与以&#36;P_0(x_0,y_0,z_0)&#36;为球心，&#36;\left|\overrightarrow{P_0M_1}\right|&#36;为半径的球面的交线，所以过&#36;The equation for the M 1 (x 1, y 1, z 1) is
&#36;&#36;00\&amp;X(x-x</em>{1})+Y(y-y_{1})+Z(z-z_{1})=0,\&amp;\left(x-x_{0}\right)^{2}+\left(y-y_{0}\right)^{2}+\left(z-z_{0}\right)^{2}=\left(x_{1}-x_{0}\right)^{2}+\left(y_{1}-y_{0}\right)^{2}+\left(z_{1}-z_{0}\right)^{2}\end{aligned}&#36;&#36;
又因为点&#36;M_1(x_1,y_1,z_1)&#36;是母线&#36;\Gamma&#36;上的任意点 则
&#36;&#36;F_1(x_1,y_1,z_1)=0,\quad F_2(x_1,y_1,z_1)=0.&#36;&#36;
根据四个方程 消去 &#36;M_1(x_1,y_1,z_1)&#36;  就可以得到
&#36;&#36;F(x,y,z) = &#36;0
The equation called the rotational curves determined by the rotation axis and the matrix</p>
<p>In particular, for the rotation curve using the axis as the rotation axis, only the coordinates of the parent line with the same name as the rotation axis are to be kept, replacing the other seat with the square root of the other two axes. Mark</p>
<p>Example: set the main line to &#36;&#36;00/Gamma{\begin{cases}F(y,z)=1x=0.&amp;\end{cases}&#36;&#36;
以&#36;y&#36;轴为旋转轴旋转，则旋转曲面为
&#36;&#36;F(y,\pm\sqrt{x^{2}+z^{2&#125;&#125;)=0&#36;&#36;</p>
<p>This rotation is the way to rotate ellipses, hyperbolic sides, parabolic lines, circles, and you can get them separately.</p>
<ul>
<li>Long, flat, spin-off ellipse.</li>
<li>Single & Double Page Rotate Hyperbolic</li>
<li>Rotate the parabolic surface</li>
<li>Ring
We'll continue to work on it back there.</li>
</ul>
<h3>Ellipse</h3>
<p>Definitions: The equation below is called Ellipse Face
&#36;&#36;\frac{x^2}{a^2}+\frac{y^2}{b^2}+\frac{z^2}{c^2}=1&#36;&#36;
He's got the basic nature of this.</p>
<ul>
<li>About three coordinates, coordinates, coordinates, coordinates, coordinates, coordinates, coordinates, and, uh, mains, the center of the main axis.</li>
<li>The intersection with the three axiss is called the top.</li>
<li>The length of the vertex is called the length of the axis, which is generally called the half-axis, which is long, short, long, short. Axis</li>
<li>The three axes are called the sphere, the two axes are called the long, flat, rotating ellipse, or the three-axis ellipse.</li>
</ul>
<p>To understand the shape of the curve, the cut-off of parallel cutting is required, and coordinates are generally used to cut the shape, as follows:
&#36;&#36;\begin{cases}\frac{2+b^2}=1,\z=0;&amp;\end{cases}&#36;&#36;
&#36;&#36;\begin{cases}\frac{x^2}{a^2}+\frac{z^2}{c^2}=1,\y=0;&amp;\end{cases}&#36;&#36;
&#36;&#36;\begin{cases}\frac{y^2}{b^2}+\frac{z^2}{c^2}=1\x=0.&amp;You're not gonna get away with this?
They're all ellipses.</p>
<p>The plane using parallel coordinates is as follows: &#36;z=h&#36; Cuts.
&#36;&#36;\begin=case}+++=====================================================================================================================================================================================================================================================&amp;\end{cases}&#36;&#36;
<strong>This kind of cutting will make us better able to study its nature.</strong> The circumstances of this type depend on&#36;z&#36;The value is determined by the location of the cutting.</p>
<p>Sometimes he's also studying ellipses with the equation of the parameters below, not a common coordinate system, but a swap experience that works well for ellipse.
&#36;&#36;00begin{cases}x=a\cos\ta\cos\varpha\y=b\cos\theta\sin\varpi\z=c\sin\theta=s&amp;\end{cases},&#36;&#36;</p>
<h3>Hyperbolic</h3>
<h4>Double side of single page</h4>
<p>Definitions: The equation below is called a single page hyperbolic Noodles.
&#36;&#36;\frac{x^2}{a^2}+\frac{y^2}{b^2}-\frac{z^2}{c^2}=1&#36;&#36;
He's got the basic nature of this.</p>
<ul>
<li>About three coordinates, coordinates, axes, coordinates, original points symmetry.</li>
<li>The hyperbolic side with&#36;z&#36;The axis is not interconnected. The intersection with the other two axes is called the vertebrae.</li>
</ul>
<p>We're cutting it with coordinates.
&#36;&#36;\begin{cases}\frac{2+b^2}=1,\z=0;&amp;\end{cases}&#36;&#36;
&#36;&#36;\begin{cases}\frac{x^2}{a^2}-\frac{z^2}{c^2}=1\y=0;&amp;\end{cases}&#36;&#36;
&#36;&#36;\begin{cases}\frac{y^2}{b^2}-\frac{z^2}{c^2}=1\x=0.&amp;\end{cases}&#36;&#36;</p>
<p>Ellipse, hyperbolic, hyperbolic</p>
<p>Use&#36;z=h&#36; Cuts.
&#36;&#36;\begin}cc\\\=====================================================================================================================================================================================================================================================&amp;You're not gonna get away with this?
It's an ellipse, which means... <strong>A single-page hyperbolic ellipse that changes and slides along a hyperbolic curve</strong></p>
<p>Use&#36;y=h&#36; Cuts.
&#36;&#36;US&#36;\begin=c\-==============================================================================================================================================================================================.======================================================&amp;You're not gonna get away with this?
When? &#36;|h|\ne b&#36;And when he was a hyperbolic, but the solid axis was paralleled by different coordinates. Axis
When? &#36;|h|=b&#36;And then he was a two-way line.</p>
<h4>Double Page Hyperbolic</h4>
<p>Definitions: The equation below is called the hyperbolic side of the double page
&#36;&#36;\frac{x^2}{a^2}+\frac{y^2}{b^2}-\frac{z^2}{c^2}=-1&#36;&#36;
He's got the basic nature of this.</p>
<ul>
<li>About three coordinates, coordinates, axes, coordinates, original points symmetry.</li>
<li>♪ That hyperbolic side only with ♪&#36;z&#36;Axis intersection, called the top point.</li>
</ul>
<p>- We're using coordinates to intercept.</p>
<ul>
<li>and&#36;z=0&#36;No intersection</li>
<li>and&#36;x=0,y=0&#36;Hand over two hyperbolics</li>
</ul>
<p>Use &#36;z=h&#36; Delivery
The #savesa #savesa #savesave #savesave #save #savesave #savesave #saves #savesave #savesave #savesaves #savesaves #a.savesavesa #a.savesa. #a.savesa. #a.savesa. #a. #a. #aaaaaaaaaaaa #aaaaaaaaaaa #aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa&amp;You're not gonna get away with this?
Based on&#36;h&#36;Values, single points, empty, ellipse.</p>
<p>That means... <strong>Double-spreaded ellipse changes and moves along two hyperbolic curves</strong></p>
<p>The hyperbolic side of a single page and the hyperbolic side of a double page are commonly called hyperbolic. Noodles.</p>
<h3>The parabolic surface.</h3>
<h4>Ellipse Shack</h4>
<p>Definition: The equation below is called elliptical parabolic Noodles.
&#36;&#36;\frac{x^2}{a^2}+\frac{y^2}{b^2}=2z&#36;&#36;
Apparently, the elliptical parabolic surface is about xOz and yOz coordinates, and about&#36;z&#36;Axes symmetry, but it didn't.
There's symmetry center, it's at point with symmetric axis&#36;(0,0,0)&#36; It's called the top of the elliptical parabolic surface.</p>
<p>Use &#36;x=0,y=0&#36; Cuts.
&#36; \begin{cases}x^2=2a^2z,\y=0&amp;\end{cases}&#36;&#36;
&#36;&#36;\begin{cases}y^2=2b^2z,\x=0,&amp;You're not gonna get away with this?
It's called the main parabolic line.</p>
<p>Use &#36;z=h&#36; Cut
That's a lot of money.&amp;You're not gonna get away with this?
It's always an ellipse, so... <strong>The ellipse is the ellipse that moves along the parabolic line.</strong></p>
<h4>Hyperbolic parabolic</h4>
<p>Definition: The equation below is called hyperbolic parabolic Noodles.
&#36;&#36;\frac{x^2}{a^2}-\frac{y^2}{b^2}=2z&#36;&#36;
Apparently, the hyperbolic parabolic is about xOz and yOz coordinates, and about&#36;z&#36;Axes symmetry, but it didn't.
Symmetry Center</p>
<p>Use&#36;z=0&#36;Intercept
That's right.&amp;You're not gonna get away with this?
It's a line that compares to the original.</p>
<p>Use &#36;x=0,y=0&#36; Cutting two parabolic lines, two main parabolic lines.</p>
<p>Use &#36;z=h&#36; Cut
That's a lot of money.&amp;You're not gonna get away with this?
This is a shift hyperbolic; when &#36;h&gt;0&#36; 时，双曲线的实轴与 &#36;x&#36; 轴平行，虚轴与 &#36;y&#36; 轴平行，顶点( &#36;\pm a\sqrt2h,0,h)&#36;在主抛物线上；当&#36;h&lt;0&#36;时，双曲线的实轴与&#36;y&#36;轴平行，虚轴与&#36;x&#36;轴平行，顶点 &#36;(0,\pm b\sqrt{-2h}, h&#36;) on the main parabolic line</p>
<p>So the curve is split into two parts, top and bottom, the upper half.&#36;x&#36;Two directions on the axis, the lower half.&#36;y&#36;The two directions of the axis are falling, the curve is broadly shaped like a saddle, so the hyperbolic parabolic is also called a saddle. Noodles.</p>
<h3>Direct parent line</h3>
<p>As we have seen before, the poles and cones can be created by a single line, a curve created by a one-clan line called the straight line, and the line created by that family is called the straight line of the same line. The pole and cone are both straight and straight.</p>
<p>And we see in this "blank" part of this article, "blank" and "blank" in this part, there's a straight line on both sides. And we're going to prove that these two sides are not only straight lines, but can be created by one line, so they're all straight lines.</p>
<h4>For a single page hyperbolic</h4>
<p>For a single page hyperbolic
&#36;&#36;\frac{x^2}{a^2}+\frac{y^2}{b^2}-\frac{z^2}{c^2}=1&#36;&#36;
We can give it to you. &#36;u&#36; Home-Current HomeMax
&#36;US&#36;\begin{cases}\frac{xa}+\frac{z}=c}=u\left(+\frac{y&#125;&#125;b}right),\frac{x&#125;&#125;-frac{z&#125;&#125;frac},=frac{1}u}\left(1\frac{y}{b}right),=&amp;\end{cases}&#36;&#36;
取 &#36;u \to 0,u\to \infty&#36; 有
&#36;&#36;\begin{cases}\frac{x}{a}+\frac{z}{c}=0\1-\frac{y}{b}=0&amp;\end{cases}&#36;&#36;
&#36;&#36;\begin{cases}\frac{x}{a}-\frac{z}{c}=0\1+\frac{y}{b}=0.&amp;You're not gonna get away with this?
They all call it &#36;u&#36; Family Straight Master</p>
<p>It's a match. It's a match.&#36;v&#36;Family Straight Master
&#36;&#36;\begin{cases}\frac{x}a}+frac{z&#125;&#125;c}=v\left(1-\frac{y&#125;&#125;b}right),\frac{x}{frac{z&#125;&#125;frac1}v}\left(+\frac{y&#125;&#125;b}\right)&amp;\end{cases}&#36;&#36;</p>
<p><strong>For a single-page hyperbolic point, each of the two straight lines passes through the point, so any one-family straight line generates the entire hyperbolic curve, only in two ways.</strong></p>
<h4>Hyperbolic parabolic</h4>
<p>For hyperbolic parabolic surfaces
&#36;&#36;\frac{x^2}{a^2}-\frac{y^2}{b^2}=2z&#36;&#36;
We can give the same direct line to the two of us.
&#36;US&#36;\begin{cases}\frac{x}a}+\frac{y&#125;&#125;}2u,\ft(\frac{x}a}-\frac{y}{b}right=z&amp;\end{cases}&#36;&#36;
以及
&#36;&#36;\begin{cases}\frac{x}{a}-\frac{y}{b}=2v,\\v\left(\frac{x}{a}+\frac{y}{b}\right)=z.&amp;\end{cases}&#36;&#36;
<strong>For a point on the hyperbolic side, each of the two straight lines passes through the point, so any one straight line can generate the entire hyperbolic curve, only in two ways.</strong></p>
<h4>Relevant nature</h4>
<p>The single-page hyperbolic and hyperbolic parabolic surface is frequently used in architecture, and its straight-back line can form the framework of construction, while the whole remains able to retain a beautiful arc.</p>
<p>Theoretically: any two straight lines of the intergalactic line on the dichotomy of the leaf must be one side, and any one of the intersecticular lines on the dichotomy of the parabolic line must be the interlocking of the intersect.</p>
<p>Theorem: Any two straight lines of the same kind on the one-leaf hyperbolic or hyperbolic parabolic surface are always straight-on-side, and the hyperbolic line of the same family is parallel to the same plane</p>
