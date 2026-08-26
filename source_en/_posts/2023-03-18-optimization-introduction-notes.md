---
title: 'Optimization: Problems, Vector Norms, and Convex Sets'
title_zh: 最优化导论：最优化问题、向量范数与凸集
date: 2023-03-18 21:27:45 +0800
categories:
- Mathematics
- Optimization
tags:
- Optimization
- Gradient Descent
author: Hyacehila
mathjax: true
hidden: true
excerpt: Covers optimization problems, vector norms, convex sets, convex functions, linear programming, simplex methods, and
  gradient descent.
description: Covers optimization problems, vector norms, convex sets, convex functions, linear programming, simplex methods,
  and gradient descent.
excerpt_zh: 整理最优化问题、向量范数、凸集、凸函数、线性规划、单纯形法和梯度下降等内容。
permalink: /blog/2023/03/18/optimization-introduction-notes/
lang: en
translation_key: 2023-03-18-optimization-introduction-notes
translation_status: machine
translation_source_hash: f99666c9c1385edf6cc81ed6890419a31733bb4afc356fb263a5bd16161e1d04
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<h2>Introduction</h2>
<p>Presentation of optimization issues and addition of some of the most basic knowledge reserves</p>
<p>The questions in this article can also be addressed<a href="/en/blog/2023/03/16/mathematical-analysis-limits-continuity-notes/">Mathematical analysis: the theory of limits and continuity</a>、<a href="/en/blog/2023/03/17/advanced-algebra-foundations-notes/">Higher algebra: the basis of the meta-mathematics</a>How the concept of a relatively close read together is developed in different contexts.</p>
<h3>On the issue of optimization</h3>
<p>The most important aspect of mathematical modelling is that it is often combined with mathematical and computer numerical methods.
The general optimization problem is reflected in <strong>The polar question of the target function under binding conditions</strong> And the tools in math analysis are often unmanageable.
At the very beginning of the optimal problem, we often see some extremes of the indifferent constraints, and then, of course, we'll have more.</p>
<h3>Basic concepts</h3>
<h4>Common concepts</h4>
<p>The equation and the array of the different kinds of constraints is the problem. <strong>Available Set</strong>
<strong>The best in the world.</strong> It's the largest or the smallest of the target functions, and if this is the only point, then it's called <strong>Strictly global and optimal</strong>
<strong>Local best solution</strong> The maximum or minimum value under a particular neighbourhood, not the whole set of possibilities.
<em>The best solution is often difficult to study in the whole world, and many of the methods behind are only local best solutions.</em>
&#36;z=min{f(x_{1},x_{2})}&#36; This two-dimensional optimization problem is often a polar issue of curves.
Theoretically: The best solution to any global event is closed. - Yeah.
Theoretically: If the target function, the equation, the differential constraint, are all continuous functions, the feasible field is closed. - Yeah.</p>
<h3>Knowledge supplement</h3>
<h4>Vector Paradigm</h4>
<p>Definition: A measurement structure is the extension of the concept of modeling. &#36;||x||&#36;</p>
<p>Nature of the model:</p>
<ol>
<li>- It's a good time.&gt;0&#36;</li>
<li>- That's right. &#36;||cx||=c||x||&#36;</li>
<li>Triangular Instinct &#36;||x+y|| \ge ||x||+||y||&#36;</li>
</ol>
<p>Different vector-based definitions <em>He's just a measurement structure, not the only one.</em></p>
<ol>
<li>EuroPerformance
&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;&#36;&#36;US&#36;&#36;US&#36;US&#36;&#36;&#36;&#36;US&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;US&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;<em>2 = \sqrt{\sum</em>{i=1}^{n} x_i^2}&#36;&#36;</li>
<li>1 standard
&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;&#36;&#36;US&#36;&#36;US&#36;US&#36;&#36;&#36;&#36;US&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;US&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;<em>1 = \sum</em>{i=1}^{n} |x_i|&#36;&#36;</li>
<li>Infinity
&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;&#36;&#36;US&#36;&#36;US&#36;US&#36;&#36;&#36;&#36;US&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;US&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;<em>{\infty} = \max</em>I'm not gonna get a chance to get a chance to get a chance to get a chance to get a chance to get a chance to get a chance to get a chance to work.
The European paradigm that we use most often in analyticals, he's called the 2th.</li>
</ol>
<p>Vector sequence condensation</p>
<ul>
<li>Concealed by standard &#36;||x^{b}-x^{k}||&#36; Limit to 0</li>
<li>Concentrate by coordinates.</li>
<li>Two contractions are essentially equal.</li>
</ul>
<h4>Hessian Matrix</h4>
<p>The gradient of the multiple function is a vector, and the weights of the vector are also a multifunctional function.
In Optimizing Theory&#36;\nabla&#36; It usually means the gradient count. He's linear.
A few special examples.
&#36;&#36;\bigtriangledown (b^{T}x)=b&#36;&#36;
&#36;&#36;\bigtriangledown (x^{T}x)=2x&#36;&#36;
&#36;&#36;\bigtriangledown (x^{T}Ax)=2Ax&#36;&#36;</p>
<h2>Crum, Crum, Crum, Crum, Crum, Crum, Crum, Crum, Crum, Crum, Crum, Crum, Crum, Crum, Crum, Crum, Crum, Crum, Crum, Crum, Crum, Crum, Crum, Crum, Crum, Crum, Crum, Crum, Crum, Crum, Crum, Crum, Crum, Crum, Crum, Crum, Crum, Crum, Crum, Crum, Crum, Crum, Crum, Crum, Crum, Crum, Crum, Crum, Crum, Crum, Crum, Crum,</h2>
<p>It's a fundamental concept that is important to both linear and non-linear planning, and it's a very necessary intellectual complement, but it's not very deep.</p>
<h3>Crumb</h3>
<h4>Basic concepts</h4>
<p>Definitions:&#36;C&#36;It's a geometry.<del>x,y\in C</del> if<del>\lambda x+(1-\lambda)y\in C</del>\lambda\in[0,1] &#36;C is a concussion
<em>The corresponding concepts are condensed.</em>
For the crumb verification, use definition directly to complete
It's easy to verify the following propositions, which involve a collection of condensed figures, which are real numbers.
&#36;&#36;\beta S_{1}={\beta x|x\in S_{1} }是凸集&#36;&#36;
&#36;&#36;S_{1}\cap S_{2}是凸集&#36;&#36;
&#36;&#36;S_{1}+S_{2}={x^{1}+x^{2}|x^{1}\in S_{1}~~~x^{2\in}S_{2&#125;&#125;&#36;&#36;
&#36;&#36;S_{1}-S_{2}={x^{1}-x^{2}|x^{1}\in S_{1}~~~x^{2\in}S_{2&#125;&#125;&#36;&#36;</p>
<h4>Cream and Multi-Face</h4>
<h4>Polars and polars</h4>
<p>Definition: If S is a non-empty condensation,&#36;x\in S&#36;   If x cannot be the condensed combination of two different points in S, called x the polarity of the condensed S.
The polygons have polar points at their top, and every one of them in the circle is polar.
Inference: For a climax, any single one of these can be a polar condensed condensation, and no one can be able to be able to be able to assemble.
Definitions: Establishment&#36;S&#36;Yes.&#36;R^{n}&#36;Up the closed cam &#36;d&#36;It's not zero.&#36;S&#36;♪ Every one of them ♪&#36;x&#36; There's all the rays.
&#36;&#36;&#123;x+\lambda d|~\lambda \ge0}\in S&#36;&#36;
Name&#36;d&#36;Yes.&#36;S&#36;If one direction cannot be the sum of the other two, then this direction.&#36;d&#36;Yes.&#36;S&#36;♪ The polar direction ♪
<em>It's obvious that only the assembly of the unbounded can have the concept of direction, so only the unbounded can have the extreme direction.</em>
Inference: Any direction can be a positive linear combination of extreme directions</p>
<h4>Crumb Separation Theorem</h4>
<p>The intuitive meaning of the amplification of the separation theorem is that under very weak conditions, two interminglings can always be separated by a super-platform, i.e., for super-platform. &#36;p^{T}x=a&#36; Both assembly points are satisfied. &#36;p^{T}x_{1}\ge a&#36; and&#36;p^{T}x_{2}\le a&#36;
This is what it's called.&#36;H&#36;  Separate. Two sets.</p>
<h4>Combling</h4>
<p>Definitions: For definitions in the condensation <strong>C</strong> Functions on &#36;f(x)&#36; If for &#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36; x, y\in<del>C</del>\forall \lambda \in[0,1]&#36;有&#36;f(\lambda x+(1-\lambda)y)\le \lambda f(x)+(1-\lambda)f(y)&#36; 则称这个函数是凸函数 如果&#36;x\ne y&#36; is called strict condensation (so-called condensed) when you do not take the equivalent
And the following reasoning and theorem are also defined to prove it.
Definition: If the numeric number above is in reverse, it is called the dent. <em>(This corresponds to the one dollar permutation definition)</em>
Definitions:&#36;f(x)&#36;is a convex function&#36;-f(x)&#36;It's a dent.  <em>Two definitions of equal value</em>
Definitions: taken&#36;\lambda=1/2&#36; It's called a mid-point.
Inference: Linear functions are both condensed and dented
Theoretically: Two amphibious functions are combined with a condensed function.
Inferences:&#36;f(x)&#36;It's a convex.&#36;\Longrightarrow&#36; &#36;\Omega_{c}={x|x\in \Omega,f(x)&lt;It's a condensed condensation
Theorem: Consequence of cones on the condensation
Theoretically: Definition in the climax&#36;C&#36;Micro functions on&#36;f(x)&#36;It's a convex.&#36;\Leftrightarrow&#36; &#36;\forall x,y\in C,f(y)\ge f(x)+\nabla f(x)^{T}(y-x)&#36;
If you're demanding strict, the equals are deleted.
Theoretically: from a one-dollar camcorder extension definition is opening Set&#36;C&#36;Micro functions on&#36;f(x)&#36;It's a convex. &#36;\Leftrightarrow&#36; &#36;f(x)&#36;Hessian Matrix is semi-positive.&#36;\nabla^{2}f(x)\ge 0&#36;It's just a matter of changing from a condition of being a mere necessity to a condition of being a sufficient one.</p>
<h4>Cam Planning</h4>
<p>Definitions: both target and binding functions are the planning questions of the condensed function called the condensed planning
Inference: The linear planning problem is Cam.
Inference: The viable set of cam is the cam, the best is cam, the best part of the place is the best in the world.
Theoretically: For cam planning, if the target function is strict and the best solver exists, the best solver exists and the only one is
<em>The only solution is that multiple points achieve the same optimal function, otherwise the best concept cannot be discussed.</em>
Theorem: set to &#36;x{<em>}&#36;是凸规划&#36;(P)&#36;的可行解 则其是最优解的充要条件是 &#36;x^{</em>}&#36; 是规划&#36;min_{x\in S }\nabla f(x^{*})^{T}x&#36;的最优解 其中S是&#36;(P)Approbable Fields of &#36;&#36;</p>
<h2>Basic nature of linear planning</h2>
<p>Linear Programing, his binding and target functions are linear, which is a simpler, more basic type of optimisation, and we're going to study the general solution to linear planning, and it's important to give some basic elements of linear planning and special solutions before we do it.</p>
<h3>Standard form of linear planning</h3>
<h4>Standard form</h4>
<p>Linear planning has so-called standard forms, binding functions can be a mixture of equations and variations, target functions can be maximized, but linear planning has standard forms, which are useful for the description of the solutions that follow.
Theoretically, all linear planning can be translated into the following forms, known as standard forms of linear planning.
<strong>The standard form of linear planning may be in the form of an equation:</strong>&#36;&#36;\begin{aligned} \min_{x_1,x_2,\cdots,x_n} \quad &amp; c_1 x_1+c_2 x_2+\cdots+c_n x_n \ \text{s.t.} \quad &amp; a_{11}x_1+a_{12}x_2+\cdots+a_{1n}x_n= b_1 \ &amp; a_{21}x_1+a_{22}x_2+\cdots+a_{2n}x_n= b_2 \ &amp; \cdots \ &amp; a_{m1}x_1+a_{m2}x_2+\cdots+a_{mn}x_n= b_m \ &amp; x 1, x 2, \cdots, x n\geq0 \end{aligned} &#36;
I'm not sure.&#36;x_i&#36; For the first time &#36;i&#36; A decision variable,&#36;c_i&#36; For the first time &#36;i&#36; The coefficient of the individual decision variable,&#36;a_{ij}&#36; For the first time &#36;i&#36; of the &#36;j&#36; The coefficient of the individual decision variable,&#36;b_i&#36; For the first time &#36;i&#36; right-end constant of a condition.
<strong>If expressed in matrix</strong>
&#36;&#36;\begin{aligned} \min_{x} \quad &amp; c^T x \ \text{s.t.} \quad &amp; A x = b \ &amp; You're not gonna get a chance to get a job.
I'm not sure.&#36;x&#36; Yes. &#36;n&#36; - The vector.&#36;c&#36; Yes. &#36;n&#36; - The vector.&#36;A&#36; Yes. &#36;m\times n&#36; The matrix,&#36;b&#36; Yes. &#36;m&#36; . Vector. Here. &#36;\max&#36; Means maximize target function &#36;c^T x&#36;，&#36;\text{s.t.}&#36; Expressing binding conditions. First line is the target function, second line is the binding condition</p>
<p>Number of variables&#36;n&#36;Called&#36;LP&#36;The dimensions of the problem. The number of equations.&#36;m&#36;Called&#36;LP&#36;Step of the problem&gt;I'm gonna need a little help.
Some of the corresponding matrix of unrelated constraints are called the foundation of linear planning.
The equations that the matrix corresponds to are decomposing as the fundamentals of the matrix. Break
The conditions of restraint permit what is called feasible Break
The best way to meet our target function is to be called the best solution.
Obviously, yes.&#36;m&#36;Step&#36;n&#36;V-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D--D----&#36;LP&#36;Questions, at most.&#36;C_{n}^{m}&#36;Basic viable solutions, with each matrix format corresponding to one basic viable solution</p>
<h4>To standard form</h4>
<p>First of all, we need to deal with the difference between the min max of the target function, which is actually, when the max of the target function is the opposite of the min, which is very easy to simplify.
For the question of the equation below, the equation needs to be introduced into the equation, as follows:
&#36;&#36;a_{11}x_1+a_{12}x_2+\cdots+a_{1n}x_n \ge b_1&#36;&#36;
Introduce loose variables into
&#36;&#36;a_{11}x_1+a_{12}x_2+\cdots+a_{1n}x_n -x_{n+1}= b_1~~~~x_{n+1}\ge 0&#36;&#36;
Same thing.
&#36;&#36;a_{11}x_1+a_{12}x_2+\cdots+a_{1n}x_n\le b_1&#36;&#36;
Other Organiser
&#36;&#36;a_{11}x_1+a_{12}x_2+\cdots+a_{1n}x_n+x_{n+1}= b_1~~~~~x_{n+1}\ge 0&#36;&#36;
Please note that every new variant has to introduce new relaxing variables, not repeat, and the relaxing variable must be larger than zero, adjust its own positive and negative numbers.</p>
<h4>Treatment of Free Variables</h4>
<p>Our standard form is to require that all variables are positive and that free variables are not allowed to appear. Here is an example of how to process them, the core of which is to eliminate them by deciphering an equation.
&#36;&#36;00begin{aligned}\min x 1,x 2,\cdots,x 3}\quad &amp;  x_1+3 x_2+4 x_3 \ \text{s.t.} \quad &amp; x_1+2x_2+x_3= 5 \ &amp; 2x_1+3x_2+x_3= 6  \ &amp; x_2,x_3\geq 0 \end{aligned}&#36;&#36;
目前这个问题中&#36;x_{1}&#36;是自由变量，明显这不符合我们的需求，所以我们解第一个等式方程得到
&#36;&#36;x_{1}=5-2x_{2}-x{3}&#36;&#36;
代入前面的表达式得到
&#36;&#36;\begin{aligned} \min_{x_1,x_2,\cdots,x_3} \quad &amp;  5-2 x_2- x_3 \ \text{s.t.} \quad &amp; x_2+x_3= 4   \ &amp; x 2, x 3\geq 0 \end{aligned} &#36;
And then it's all done. It's all over.</p>
<h3>Figure</h3>
<p>It's easier to figure out the simplest linear method.
We draw the graphics of the binding conditions and the linear functions to be sought, and by moving the line, we can find the maximum value.
The obvious pattern is that it can only deal with two dimensions.
Inference: For a two-dimensional linear planning problem, the best solution must be found at the convex of the convex.
And if the two vertexes are the best, then it means that the best is not the only one, but a line.
And when the assembly is infinite, the best solution may not exist, and it needs to be taken into account.</p>
<h3>Theories of linear planning</h3>
<p>The basic rationale for linear planning is two core theorems, which are the basis for the solution to the problem of linear planning behind us.
&#36;&#36;定理：矢量x是凸集Ax=b的极点的充要条件是x是Ax=b的一个基本可行解&#36;&#36;
&#36;&#36;定理：对于一个标准的LP问题。如果存在可行解，就一定存在基本可行解，
如果存在最优可行解，就一定存在最优的基本可行解&#36;&#36;
Two of theorems can tell us that the best solution to the LP problem is to study the basic viable solution, the polar point of the feasible collection, which gives us the most basic and central elements of the solution, and a review of some of the elements of the diagram.
&#36;&#36;推论：只要可行集非空，至少有一个极点&#36;&#36;
&#36;&#36;推论：只要有限的最优解存在，那一定在一个极点上&#36;&#36;
&#36;&#36;推论：极点的数量至多为有限个&#36;&#36;</p>
<h2>Basic simple methods</h2>
<p>Here we'll present the solution to one of the core LP questions.
According to the basic theorem, we can find the best solution when we look at all the poles, but it is very unrealistic to have a super-large LP problem, so we need to give a simple approach;
He is a conversion method, from one basic viable to another, and ensures that the target function is reduced, and that, over time, the optimal basic viable solution is found with fewer than a few times.
We'll introduce the three inverted lines of thought, and then we'll combine them for a simple table exercise.</p>
<h3>Basic decomposition</h3>
<p>For one.&#36;LP&#36;The problem is as follows:
&#36;&#36;00begin{aligned}\quad &amp; c^T x \ \text{s.t.} \quad &amp; A x = b \ &amp; x \geq 0 \end{aligned}&#36;&#36;
由于我们的原始问题是通过引入松弛变量实现的标准化，所以我们认为系数矩阵中一定存在一个标准阵&#36;E&#36; 即如下的形式
&#36;&#36;\left{\begin{array}{ll}
x_{1}+ &amp; +y_{1, m+1} x_{m+1}+\cdots+y_{1 n} x_{11}=y_{10} \
x_{2}+\cdots \
&amp; Other Organiser
I'm sorry, I'm sorry.
It's easy to find a basic solution at this point.
How do we find another basic solution?
Suppose we want to remove the basic variable xp and introduce the new basic variable xq, and our operation is to...
&#36;p&#36;Okay. &#36;x_{q}&#36;coefficient to 1 (multiplication constant)
Other rows &#36;x_{q}&#36;coefficients to 0 (plus minus)&#36;p&#36;Lines)</p>
<h3>Assurance of feasibility</h3>
<p>Study the feasibility of ensuring that the process of conversion is resolved.</p>
<h3>Decrease in the number of determinations and target functions</h3>
<p>Study the reduction of the target function that is guaranteed to be solved during conversion</p>
<h3>Use of simple tables</h3>
<p>Let's give you an example of that.</p>
<h2>Simple methods for improvement</h2>
<h3>Big M.</h3>
<p>To deal with the absence of a matrix, we introduced the Big M method, which is the following.
We're introducing artificial variables into the original linear planning problem.&#36;y=(y_{1},y_{1}...y_{m})&#36;
tectonic linear planning issues
&#36;&#36;00begin{aligned}\quad &amp; c^T x+ME^{T}y\ \text{s.t.} \quad &amp; A x+y = b \ &amp; You're not gonna get a chance to get a job.
At this point, there is a unit matrix that can be solved using a simple method.
<strong>Theorem</strong>: set (x^)<em>},y^{</em>})&#36; 是修正问题的最优解 那么如果&#36;y^{<em>}&#36;为0 &#36;x^{</em>It's the best solution to the original problem
Otherwise, there's no solution to the problem. Ry.
It's pointless to make M a certain number.</p>
<h3>Two-stage approach</h3>
<p>Or is it the original problem that we introduce artificial variables?&#36;y=(y_{1},y_{1}...y_{m})&#36; The question of the tectonic amendment is as follows:
&#36;&#36;00begin{aligned}\min  &amp; \sum y_{i}\ \text{s.t.} \quad &amp; A x+y = b \ &amp; x \geq 0 ~y \geq 0\end{aligned}&#36;&#36;
<strong>Theorem</strong>: set (x^)<em>},y^{</em>})&#36; 是修正问题的最优解 那么如果&#36;y^{<em>}&#36;为0 &#36;x^{</em>It's the best solution to the original problem
Otherwise, there's no solution to the problem. Ry.</p>
<h3>Degradation and recycling</h3>
<p>If a min=0 appears in the calculation of the entry base variable, the new resolution corresponds to the target function of the old resolution, resulting in an iterative cycle that goes beyond
We don't give you the way to deal with degradation and cycling, just to avoid problems like this as much as possible.</p>
<ul>
<li>There are more than one dollar.&lt;We pick the smallest one as the base variable.</li>
<li>If you have multiple off-base variables, choose&#36;min~r_{k}&#36;That one.</li>
</ul>
<h2>One-dimensional search.</h2>
<h3>The construction rationale for optimizing the problem</h3>
<p>Here is a description of the overall rationale for optimizing the problem.
Take the unbounded question of miniaturization&#36;min(f(x))&#36; For a given initial value&#36;x^{0}&#36; We have an iterative process that's as follows:</p>
<ol>
<li>I'm gonna get it by certain rules.&#36;x_{k}&#36;Falling direction&#36;d_{k}&#36;</li>
<li>Make sure you have a long walk by certain rules. <em>{k}&#36;  一般是&#36;min[f(x^{k}+\lambda</em>{k}d_{k})]&#36;</li>
<li>You!&#36;x^{k+1}=x^{k}+\lambda_{k}d_{k}&#36;</li>
<li>To determine, according to certain rules, whether or not it is necessary to end the iterative, circular or output results
You can get a sequence according to the above method.&#36;x^{k}&#36; His limit is the tiny problem of miniaturization. Points
If there are very small values common to multiple initial points, it's a global contraction or local.</li>
</ol>
<h3>Condensity analysis</h3>
<p>The compulsive analysis of algorithms is a complex exercise, and many algorithms are not sure whether they are constricting, but they are not used in a way that will delay us. Considering the efficiency of implementation is a more important issue in the design of algorithms, and some discussion of conservativity follows.</p>
<p>For a sequence that is condensed by a standard number&#36;x^{k}&#36; If real number exists&#36;\alpha&#36;  and constants &#36;k&#36; Satisfied
&#36;lim  k\rightarrow\infty}\flex()}x{<em>}\right|}{\left|x^{k}-x^{</em>}\right|^{\alpha&#125;&#125;=q&#36;&#36;</p>
<ul>
<li>&#36;\alpha=1~q&gt;It's called linear compression speed.</li>
<li>&#36;1&lt;\alpha&lt;2<del>q&gt;0</del>or<del>\alpha=1</del>q = zero dollars called ultralinear contraction</li>
<li>&#36;\alpha=2&#36; It's called a second-stage containment speed.</li>
</ul>
<p>Based on the condensity analysis, they give the usual iterative termination conditions, which are used to optimize the above-mentioned problems.</p>
<ul>
<li>&#36;||x^{k+1}-x^{k}||&lt;\varepsilon&#36;</li>
<li>&#36;||\bigtriangledown f(x^{k})||&lt;\varepsilon&#36;</li>
</ul>
<h3>Accurate one-dimensional search.</h3>
<p>In the optimal question construction in the front, step long&#36;\lambda&#36;It is based on a one-dimensional, tiny-value problem, which is essentially the optimization of the one-dimensional basis, and we are here to study the calculation of this problem, i.e., a one-dimensional search; we will only introduce some of the methods, because they are not exhaustive;</p>
<h4>It's an analytical solution.</h4>
<p>For a one-dimensional problem, we can study it directly by looking for guidance.
&#36;f&#39;♪ (x) = zero is the point of the extreme, just solve the corresponding value. ♪</p>
<h4>Success - Failure Law</h4>
<p>A one-dimensional unbounded problem.&#36;min(f(x))&#36; For a given initial value&#36;x^{0}&#36; The calculation is as follows:</p>
<ol>
<li>Give initial points&#36;x^{0}&#36; Search step £h&gt;0&#36; 精度&#36;\varepsilon&#36;</li>
<li>Calculate&#36;x^{1}=x^{0}+h;f_{1}=f(x^{1})&#36;</li>
<li>If &#36;f s&lt;Other Organiser <em>&#36;x^{0}=x^{1},f_{0}=f_{1},h=2h&#36;</em></li>
<li>On the contrary, the search failed if &#36;US&#36;|&lt;\varepsilon&#36; 找到极小，搜索结束 否则缩小步长后退搜索&#36;Other Organiser
This is the whole iterative process of success.
No title to search for spaces&#36;\varepsilon&#36; Our goal is to find the best possible possible compartment, not the exact value, and the theory of it is perfectly consistent.</li>
</ol>
<h4>Act No. 0.618</h4>
<p>A means of integrating through the principle of separation of gold.
Consider the following one-dimensional miniaturization &#36;min(f(x))<del>st</del>I'm not sure I'm gonna be able to do this.
We need to know in advance. &#36;\varepsilon,\alpha=0.618&#36;</p>
<ol>
<li>Calculating &#36;lambda 1}a1}(1-\alpha)(b a  )<del>\mu_{1}=a_{1}+\alpha (b_{1}-a_{1})&#36; &#36;f(\lambda_{1})</del>f(\mu_{1})&#36;</li>
<li>If &#36;b k}-a k}&lt;\varepsilon&#36; 迭代结束 最优解为&#36;(b_{k}+a_{k})/2&#36; 如果&#36;f(\lambda_{1})&gt;Turn 3 or turn 4</li>
<li>&#36;a_{k+1}=\lambda_{k},b_{k+1}=b_{k}&#36; And then, in this range, we start again from 1st.</li>
<li>&#36;a_{k+1}=a_{k},b_{k+1}=\mu_{k}&#36; And then, in this range, we start again from 1st.</li>
<li>Angular Cursor
The number of overlaps required by the 0.618 law is often higher because of his very slow pace of contraction.</li>
</ol>
<h4>Diphthesis</h4>
<p>The dichotomy here is perfectly consistent with the dichotomy of rooting, and we use the mediaorical and guidanceal reasoning to determine the exact location of root; we simply need to stop repeating it, and we need to look at the cut-off conditions, which are usually based on &#36;US&#36;xx1}1}{&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#36;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&#125;&lt;\varepsilon
This is a very similar core idea to the Newton traverse, which is to turn the smallest value problem of the original function into a zero point problem of the conductive function and then solve it using a rooting method.</p>
<h4>Newton Theory</h4>
<p>His core idea is to use the second-order Taylor expansion to approximate the original function, to study the position of the tiny values with the analytical nature of the approximate function, which is the following infra.
For a given initial value&#36;x^{0}&#36; We're going to do the iterative process.
&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;&#36;US&#36;US&#36;US&#36;US&#36;&#36;&#36;&#36;US&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;US&#36;&#36;&#36;&#36;&#36;US&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;US&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;US&#36;&#36;&#36;&#36;&#36;&#36;&#39;}(x^{k})/f^{&#39;&#39;(x^k) &#36;
Until then,&#39;}(x^{k}|&lt;\varepsilon
The advantage of this method is that it's very fast to absorb.</p>
<h4>Blanket method</h4>
<p>Using a second-stage function other than Taylor to achieve a similarity, i.e., a multi-value-plug-in approach, we choose three-plug-in to construct a second-order function.
We need initial plug-in point &#36;x .<del>x_{1}</del>x_{2}<del>&#36; 一般默认&#36;The factor of the second-order function can be obtained by demanaging the interpolation equation that we construct in the middle of the equation.
&#36;&#36;\bar{x}=1/2[x_{1}+x_{0}-b_{1}/a_{2}]&#36;&#36;
of which
&#36;&#36;(f  f)/(x x  )=b (f {2}-f  )/ (x 2}=b 2}</del>(b 2(2)-b (x ) / (x}2}) =a (2} &#36;
Just understand. Better remember the formula.</p>
<h3>Unprecision 1-D search</h3>
<p>In some cases, an imprecision one-dimensional search can act as a better acceleration of the contraction, and an accurate one-dimensional search can be a little too difficult to achieve, so an imprecision one-dimensional search is now used very widely.
It's a way out of the way.</p>
<h2>Unbound optimisation method</h2>
<h3>Extreme condition for unbound optimization</h3>
<p>In general, unbounded optimization is achieved through a series of 1-dimensional searches, and the choice of this series of 1-dimensional searches is a matter that we need to consider;
Theorem: The curve drops fastest in the direction of the negative gradient, and the gradient at the very small point is zero.
Theorem: The best local conditions for filling need to be added to the Hessian matrix.
Theorem: For the amphibious function, the best solution in the whole area is a zero gradient.
<em>With these theories, we know that the gradient is at the heart of the study of the problem of the non-binding polarity.</em></p>
<h3>Maximum Drop</h3>
<p>The core of the most rapid drop is the principle of the fastest drop in the negative gradient; then the one-dimensional search determines the length of each step, and the end is the end of the cut-off, followed by a detailed calculation step.</p>
<ol>
<li>Give initial points&#36;x^{0}&#36;  Precision&#36;\varepsilon&#36; You!&#36;k=0&#36;</li>
<li>Calculate&#36;d^{k}=-\bigtriangledown f(x^{k})&#36; If you want to go to the hospital,&lt;\varepsilon&#36;搜索停止，返回现在的&#36;x^{k}&#36;作为&#36;x^{*}&#36;</li>
<li>From&#36;x^{k}&#36; Let's go. Let's do a one-dimensional search.&#36;min{f(x^{k}+\lambda d^{k&#125;&#125;&#36;</li>
<li>Found it.&#36;\lambda&#36;Find a new one.&#36;x^{k+1}&#36; Next iterative
<em>In manual computing, a one-dimensional search often uses the simplest analytical properties to achieve search, not those complex search algorithms, which are more suitable for computer use.</em>
<strong>The algorithm is excellent.</strong></li>
</ol>
<ul>
<li>There's no need for a start point. The front is fast.</li>
<li>The iterative speed is slow at the point of most excellent resolution, unstable for disturbances, the rate of containment is influenced by the scale of the variable, leading to a problem of initial point selection or not, and bad initial point selection leads to complex computation processes</li>
<li>For determining the best in the global context, there are often multiple points for the best in the local context, and if consistent, the best in the global context is considered.</li>
<li>To avoid a decline in the serene state, we're using modified search directions.&#36;d^{k}=x^{k}-x^{k-2}&#36; It's probably understandable. He can avoid a fall in the serene serene.</li>
</ul>
<h3>Newton Act</h3>
<p>Newton uses the double approximation to achieve the search orientation, as follows:</p>
<ol>
<li>Give initial points&#36;x^{0}&#36;  Precision&#36;\varepsilon&#36; You!&#36;k=0&#36;</li>
<li>Calculate&#36;g^{k}=\bigtriangledown f(x^{k})&#36;If &#36;g^k}&lt;\varepsilon&#36;搜索停止,返回现在的&#36;x^{k}&#36;作为&#36;x^{*}&#36;</li>
<li>Calculate&#36;d^{k}=-[\bigtriangledown ^{2} f(x^{k})]^{-1}g^{k}=-H_{k}^{-1}g^{k}&#36; As a direction of decline</li>
<li>Make a one-dimensional search.&#36;min{f(x^{k}+\lambda d^{k&#125;&#125;&#36;</li>
<li>Found it.&#36;\lambda&#36;Find a new one.&#36;x^{k+1}&#36; Next iterative
&#36;H^{-1}&#36; It's a countervailing matrix.
<em>To ensure that the matrix is in place.&#36;Hk&#36;and&#36;gk&#36;We can multiply the matrix. We're gonna make a column vector for the gradient.</em>
<strong>The algorithm is excellent.</strong></li>
</ol>
<ul>
<li>Local deflation speed is excellent, secondary termination, optimal for double contour.</li>
<li>Newton's direction is not necessarily down, is limited to the hessian matrix's heaeness, and requires a non-generic matrix, otherwise the directional calculation is difficult</li>
</ul>
<h3>Co-graduation</h3>
<p>It's not easy to calculate the second-order Hessian matrix and his reverse matrix; the lateer-stage effects of the most rapid decline are not good, and we want to combine the two advantages, the co-graduation approach we're introducing here.
Definitions: vectors&#36;d_{1}~d_{2}&#36; About a matrix co-exist means&#36;d_{1}^{T}Ad_{2}=0&#36; This matrix is a unit when it's down.
&#36;orord: A n*&gt;0,d^d^n}is a set of A-Cyber vectors; f(x)=1/2x^T}Ax+b^T}+c needs to start from any initial point, and a precise one-dimensional search from each co-location direction can find the best solution, at most &#36;n&#36;
With the aberrations, the problem becomes the generation of a group of co-directions, which in fact is the best combination of functions, and we give the most classic co-radical algorithms directly below.</p>
<ol>
<li>Give initial points&#36;x^{1}&#36;  You!&#36;k=1&#36;</li>
<li>Calculate&#36;d^{k}=-g^{k}=-\bigtriangledown f(x^{k})&#36;   <em>Gradients small enough to terminate the iterative</em></li>
<li>One-dimensional search according to the direction of the negative gradient, giving a supporting formula&#36;\lambda _{k}=\frac{(g^{k})^{T}g^{k&#125;&#125;{(d^{k})^{T}Ad^{k&#125;&#125;&#36; (analytical calculations also)</li>
<li>You! &#36;x^{k+1}=x^{k}+\lambda_{k}d^{k};g^{k+1}=\bigtriangledown f(x^{k+1});\alpha_{k}=\frac{(g^{k+1})^{T}g^{k+1&#125;&#125;{(g^{k})^{T}g^{k&#125;&#125;;d^{k+1}=-g^{k+1}+\alpha_{k}d^{k}&#36;</li>
<li>The number of iteratives is the number of variables, counting the number of times from the initial point Subaru
<em>Please note that we need to ensure that the first negative gradient is used to calculate the direction of the construction.</em>
&#36;A&#36;It's the initial Hessian matrix.</li>
</ol>
<h3>The DFP method of variation (Newton Act)</h3>
<h4>Newton Equation</h4>
<p>The newton method is an improvement on the newton method, and it's hoped to find an iterative way to replace the Hessian matrix.
&#36;&#36;H_{k+1}g_{k}=s_{k}&#36;&#36;
It's called the Matrix.&#36;\times&#36;Current gradient = direction of decline.</p>
<h4>Newton Act.</h4>
<p>The idea of a newton law is to replace the Hessian matrix in the Newton approach with other means, which is a huge flaw in the newton law.</p>
<ol>
<li>Give initial points&#36;x^{0}&#36;  Precision&#36;\varepsilon&#36;</li>
<li>You!&#36;H_{1}=E&#36; Calculate&#36;g^{1}=\bigtriangledown f(x^{1})&#36; If you're not gonna get it,&lt;\varepsilon&#36;搜索停止,返回现在的&#36;x^{1}&#36;作为&#36;x^{*}&#36;</li>
<li>You!&#36;d_{k}=-H_{k}g^{k}&#36;</li>
<li>One-dimensional search and find out.&#36;\lambda_{k}&#36; Calculate&#36;x^{k+1},g^{k+1}&#36;</li>
<li>Repeat the operation and use the DFP to fix the formula to get new&#36;H_{k}&#36; Until we find the best solution.
&#36;&#36;DFP fixation formula: {H}<em>{k+1}={H}</em>{k}+\frac{\Delta x^{k}\left(\Delta x^{k}\right)^{T&#125;&#125;{\left(\Delta x^{k}\right)^{T} \Delta g^{k&#125;&#125;-\frac&#123;&#123;H}<em>{k} \Delta g^{k}\left({H}</em>{\kn\k}The difference between \delta g^ and k is &#36;&#36;&#36;&#36;&#36;&#36;</li>
</ol>
<h2>Banning optimization methods</h2>
<h3>Limitation polar conditions</h3>
<p>A binding optimization should be expressed in the following form: the whole is divided into target functions, with varying degrees of binding, with equations bound by three parts.
&#36;&#36;\left{array}
\min <em>{x \in \mathbb{R}^{n&#125;&#125; f(x) \
\text { s.t. } g</em>{i}(x) \geq 0, i=1, \cdots, m, \
h_{j}(x)=0, j=1, \cdots, n .
\end{array}\right.</p>
<p>Gather!</p>
<p>S=\left{\begin{array}{l|l}
x \in \mathbb{R}^{n} &amp; \begin{array}{l}
g_{i}(x) \geq 0, i=1, \cdots, m, \
h_{j}(x)=0, j=1, \cdots, n .
\end{array}
\end{array}\right}</p>
<p>Available set for question (1). &#36;</p>
<p>For the optimization of the existence of binding, it is likely that the location of the target function without binding is not feasible Set&#36;S&#36;It's not possible to do research directly using unbound methods. We need to adapt.
Definition: Falling direction, target function falling as it moves in this direction
Definition: a feasible direction, and as we move in this direction, we can guarantee that only a certain length will remain viable. Internal
Definitions: Linearly feasible direction set of a point
&#36;US&#36;(\,=),=left(=,{\)=
The \neq 0 \matbb{r &amp; \begin{array}
The \nabra g  (\), \geq0, i \i, \i, \i \
=0,j=1,\cdots, n.
I'm sorry.
I'm sorry, I'm sorry.
Definitions: For a workable set&#36;S&#36;One of these, the differential binding conditions are divided into two states, and are met.&#36;g(x_{i})=0&#36;♪ When called positive binding&gt;0&#36; 称为非积极约束，记&#36;I = i = 0, i = 1, 2,... m}As a positive constraint indicator for a point Set
&#36;Kuhn-Tucker polar condition: Sets target and binding functions can be micro, vector collections of CQ {\bigtriangledown g  (\) (\), \\bigtriangledown h {j} (\bar{x&#125;&#125; linearly irrelevant; there is a number w i} if \\bar{x} is the best solution in part<del>♪ V j} Made</del>\\bigtriangredownf (\)=sum w sigtriang  (+bar)+sumv sum \bigtriangledownh} (\barx}); where w i}\g0~ <del>i \in I</del>I is a binding set of indicators
Point to make KT polar conditions valid&#36;\bar{x}&#36;Called&#36;K-T&#36;Points
Theorem: the problem of condensed planning&#36;K-T&#36;The best part of the world.</p>
<h3>&#36;K-T&#36;Validation and calculation of points</h3>
<p>In fact, we are doing this with full definition, and we need to practice and understand in depth the process we need to calculate, and we need to handle a linear equation group that can get K-T point positions; likewise, we can do validation.
It's still a bit difficult to understand the calculation.
The only issue with equations is the LAGL's method of multiplying mathematically, so the KT method is to deal with the two constraints of the equation and combine them.
<strong>First, there's only one kind of indifferent constraint.</strong></p>
<ol>
<li>Construct the LaGrand function &#36;L(x,\lambda)=f(x)+\lambda g(x)&#36;</li>
<li>Scrambling gradient equation &#36;\nabla f (x^)<em>})+\lambda \nabla g(x^{</em>})=0&#36;</li>
<li>Separate discussion&#36;\lambda=0~and\nabla g(x^{*})=0&#36;</li>
<li>Verify if the resolution obtained meets the gradient equation and &#36;\lambda\ge 0~g(x^{*})\le0&#36;
So we can know exactly which KT points we need to know.
<strong>Consolidated</strong></li>
<li>Construct the LaGrand function &#36;L(x,\lambda)=f(x)+\sum\limits\lambda_{i} g_{i}(x)+\sum\limits\mu_{j}h_{j}(x)&#36;</li>
<li>Scrambling for gradients &#36;\nabla f(x)+\sum\limits\lambda_{i} \nabla g_{i}(x)+\sum\limits\mu_{j} \nabla h_{j}(x)=0&#36;</li>
<li>Separate discussion&#36;\lambda_{i}=0~and\nabla g_{i}(x^{*})=0&#36;</li>
<li><em>We discuss each and every one of them on our own.&#36;\lambda_{i}&#36; When it's zero, it's the same.&#36;g_{i}(x)&#36;Not zero, or it'll work.&#36;g_{i}(x)&#36;- It's not.</em></li>
<li>Verify if the resolution obtained meets the gradient equation and &#36;\lambda i}\ge0<del>g(x^{*})\le0</del>h_{i}(x^{*})=0&#36;
<em>It's not easy to be wrong to be careful about the standard form of binding.</em></li>
</ol>
<h3>SUMT Extralegal</h3>
<p>SUMT is the unbounded method of miniaturization.
The idea is to construct a punitive function that allows our own variables to deviate from the feasible domain and then expand rapidly, and then change to unbounded optimization.
<strong>Step one.</strong>
Construct penalty function&#36;p(x)&#36; Okay.&#36;F(x,M)=f(x)+Mp(x)&#36;
&#36;p(x)&#36;♪ Need to meet the continuum, constant, just&#36;x\in S,p(x)=0&#36;
<strong>Step two.</strong>
Construct this&#36;p(x)&#36;It's easy to think about. We have the following idea.
Equivalent Constraints&#36;p(x)=h^{2}(x)&#36;
Impansive constraints&#36;p(x)=0 if x\in S else~ ~p(x)=g^{2}(x)&#36;
For multiple constraints, all of them.&#36;p(x)&#36;It's good to add up.
<strong>Step three.</strong>
Solution Unbound Optimization
If you need it, it's a different story.&#36;M&#36; Use&#36;M_{k+1}=M_{k}c;c\in[4,10]&#36;
Initial&#36;M&#36;We need to specify that, in the computation of analysis, we need to remember that M is very big.</p>
<h3>SUMT Intra-Mechanism</h3>
<p>So the idea is to construct the wall function that allows us to rapidly increase the function after the variable is near the viable boundary, and to change it to an unbounded optimization.
<strong>Step one.</strong>
Construct penalty function&#36;B(x)&#36; Okay.&#36;F(x,r)=f(x)+rB(x)&#36;
&#36;B(x)&#36;It's a constant, constant, and it's a time when B(x) is going to be endless.
<strong>Step two.</strong>
Construct this&#36;B(x)&#36;The item is easy to think of, which one is good, and which one is good, and the standard form is more than zero.
&#36;B(x)=\sum\limits g_{i}^{+}(x);g_{i}^{+}(x)=-\frac{1}{g_{i}(x)}or-ln(-g_{i}(x))&#36;
<strong>Step three.</strong>
Solution Unbound Optimization
If you need it, it's a different story.&#36;r&#36; Use&#36;r_{k+1}=\frac{r_{k&#125;&#125;{c};c\in[4,10]&#36;
Initial&#36;r&#36;We need to specify that, in the computation of analysis, we should remember that r is close to zero.</p>
<h3>SUMT Mixing Method</h3>
<p>Combined punishment and bumping walls can accelerate the iterative process.</p>
<h2>Multi-purpose planning</h2>
<p>The definition of multi-purpose planning is very clear and clear; we have a number of&#36;min~max&#36;function;</p>
<p>It is clear that if all target functions achieve optimal interpretation (absolutely optimal) at the same point, it is certainly good, but it is too ideal to achieve;</p>
<p>In fact, we tend to give the concept of effective solvency here; it means that there is no better solution than he (achieving one goal function would necessarily be another one that deviates from the best), the core point is that the size of the vector is not comparable here, and the final multi-target planning issue is called the question of choosing in a valid solve according to one's preferences, how to choose, without distinction of superiority or inferiority.</p>
<p>Our multi-purpose planning is a process of finding effective solutions, not the best solution, and the best solution is determined by preference;</p>
<p>Some of the classic solutions for effective solutions are described below, with the idea of reducing multiple goals to single targets;</p>
<h3>Very small, very large.</h3>
<p>&#36;&#36;h(x)=min/max~{h_{1}(x),h_{2}(x)...h_{n}(x)}&#36;&#36;
The disadvantage, only one effective solution, the resulting function lost its microlasticity</p>
<h3>Linear weighting</h3>
<p>&#36;&#36;h(x)=\sum \theta_{i}h_{i}(x)&#36;&#36;</p>
<h3>Square weight</h3>
<p>&#36;&#36;h(x)=\sum \theta_{i}(h_{i}(x)-h_{min}(x))~~~h_{min}(x)=min~{h_{1}(x),h_{2}(x)...h_{n}(x)}&#36;&#36;</p>
<h3>Multiply</h3>
<p>Another way to construct, to study later.</p>
<h3>Law of constraint</h3>
<p>Only one target function is retained, and the remaining target functions are converted into a solution to binding conditions</p>
