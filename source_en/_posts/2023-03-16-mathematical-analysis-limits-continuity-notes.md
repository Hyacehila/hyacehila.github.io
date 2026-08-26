---
title: 'Mathematical Analysis: Limits and Continuity'
title_zh: 数学分析：极限与连续理论
date: 2023-03-16 00:10:49 +0800
categories:
- Mathematics
- Mathematical Analysis
tags:
- Mathematical Analysis
author: Hyacehila
mathjax: true
hidden: true
excerpt: Covers sets, functions, sequence limits, function limits, continuity, and related foundations.
description: Covers sets, functions, sequence limits, function limits, continuity, and related foundations.
excerpt_zh: 整理集合、函数、数列极限、函数极限、连续性和相关基础概念。
permalink: /blog/2023/03/16/mathematical-analysis-limits-continuity-notes/
lang: en
translation_key: 2023-03-16-mathematical-analysis-limits-continuity-notes
translation_status: machine
translation_source_hash: ef3e1b6215cfb0b2cb691295ab287fc9b3d0425e507db925dd0c5b44061ccfe2
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<h2>Before all</h2>
<p>Math analysis is one of the basic subjects, like higher mathematics, but math analysis is narrower than higher mathematics, and it contains only the core of calculus knowledge, the geometry, the theory of differential equations, which is taught as a separate subject; and here, as the core of basic lessons, we learn calculus, unlike the first mathematics that we have studied, we will study the amount of change from now on, and it will be very large.
The creation of the calculus theory is a demand-driven result, and we can see the meaning of knowledge at the application level; the creation of the calculus theory consists of two core components: the strict definition of the calculus and the calculus, but the connection between the two.
In math analysis, we need to not only learn to calculate, but understand the idea of calculus is also important.</p>
<h2>Preparatory knowledge</h2>
<p>Learning the necessary preparatory knowledge is the beginning of almost every math lesson book.</p>
<h3>Gather!</h3>
<h4>Some basic learning review of the collection</h4>
<p>And basically, every book will mention it.</p>
<p>The questions in this article can also be addressed<a href="/en/blog/2023/03/18/real-analysis-notes/">Realistic functions: collection and point set, measure and detectability</a>、<a href="/en/blog/2023/09/11/functional-analysis-notes/">General analysis of communications: measuring space, tightness and severability</a>How the concept of a relatively close read together is developed in different contexts.</p>
<p><strong>Definition of a collectivity: a collective of specific or abstract objects of a particular nature</strong>  These objects are called the elements of the assembly.
&#36;&#36;x \in S ~~~~ x \notin S  &#36;&#36;
Common collections
&#36;&#36;N^{*} ~~ N ~~Z ~~ Q ~~ R&#36;&#36;
A collection can be used to indicate a number of numbers and descriptions (high school)</p>
<p>The elements in the collection are not locational, the duplicate elements are meaningless, and the collections that contain no elements are empty.
&#36;&#36;\phi&#36;&#36;
The subset is the one where all elements in one set belong to another.
&#36;&#36; S\subset T&#36;&#36;
The fields are a subset of real numbers.</p>
<h4>Calculator of the collection</h4>
<p>and
&#36;&#36;S \cup T = {x\in S ~~ or~~ x\in T }&#36;&#36;
Hand it over.
&#36;&#36;S \cap  T = {x\in S ~~ and~~ x\in T }&#36;&#36;
Bad
&#36;&#36;S \setminus   T = {x\in S ~~ and~~ x\notin T } ~~和S-T一致&#36;&#36;
Patch
&#36;&#36;S_{x} ^{C} = x\setminus S  ~~~~明显的 S\cup S_{x} ^{C} = x~~~~S\cap  S_{x} ^{C} = \phi &#36;&#36;
Operations <em>Exchange, integration, distribution</em>
&#36;&#36;\begin{align*}
&amp;A\cap B=B\cap A \
&amp;A\cap B\cap C = (A\cap B)\cap C =A\cap (B\cap C) \
&amp;A\cap (B\cup D) = (A\cap B)\cup (A\cap D) \
\end{align*}&#36;&#36;
De Morgan 公式
&#36;&#36;(A\cup B)^{C} = A^{C}\cap B^{C} ~~~~ (A\cap B)^{C} = A^{C}\cup B^{C}&#36;&#36;</p>
<h4>A limited set and an unlimited set</h4>
<p>If the combination of n elements is made up of n is a non-negative integer, it becomes a limited set, and instead it is called infinity. Set</p>
<p>If an infinity-based element can be sequenced according to a pattern, it becomes a columnable one. Set</p>
<p>There must be a column set in the infinite collection, but not necessarily a column set.</p>
<p>Can be rowed and can be columnized  <em>The core is finding a pattern.</em></p>
<p>Rational Numeric Q is a columnable collection</p>
<h4>Descartes Multiplier Geometry</h4>
<p>&#36;&#36;A\times B = {(x,y)\mid x\in A~~ and~~y \in B}&#36;&#36;
Decartes, the product, can represent information with higher dimensions, which has profound implications for the geometric theory.</p>
<h3>Maps & Functions</h3>
<h4>Map</h4>
<p>Map definition: some correspondence between the two pools; X Y is two pools; if the relationship f allows for the only identified Y to match each element x in X, then rule f is a map of X to Y, as recorded
&#36;f: X\toY <del>or</del> y=f(x) &#36;
We call y like x is original X is defined as a Y subset is a value domain  <em>Because some of the Ys are not available, they don't count.</em></p>
<p>It's like being unique, but it's not.</p>
<p>Single: The original is unique
Full shot: Y is the range
Double: single and full.
Reverse map: For single-spectrum f, the map of the construction of Y to X is exactly the opposite of the map of the f, called reverse map, which obviously needs to be double-spectrum to construct reverse map, and we remember as,
&#36;&#36;f^{-1}&#36;&#36;
Composite Map: The following is called the composite Map, and the result is a composite Map
&#36; \begin{align*)
&amp;For ~g: X\to  name<del>f: U \to Y~ if it's a sub-unit of U { Set</del>\&amp;X\toY is called a specific map of f and g ~ as f\c g
{\fnH003F4}
The existence of composite maps depends on whether the previous map is a sub-area of the latter map definition Set</p>
<p>The composite map is sequenced.</p>
<p>CompositeMap Constant Equivalent
&#36;&#36;f\circ f^{-1}(y)=y~~~~f^{-1}\circ f(x) = x &#36;&#36;</p>
<h4>Functions</h4>
<p>For the map in front, we take all X Y and R, and we call it a one-dollar real function.
&#36;&#36;y = f(x)&#36;&#36;
x is called "self-variant" y is called "cause variable f" is called "functional relationship"</p>
<p>The discipline of the so-called mathematical analysis of variables is actually studying a few special functions, such as a multiple function, and the analytical classes of a particular type of function.</p>
<h5>Primary</h5>
<p>Basic primary functions include only the following six categories
&#36; \begin{align*)
&amp; Constant Functions<del>y=c \
&amp; \y=x(alpha}\
&amp; Index Functions</del>y=a^{x}\
&amp; Logarithmic Functions<del>y=log_{a}^{x}\
&amp; Triangular Functions</del>y=sin(x) etc. \
&amp;Inverse Trigonometric ~y=arcsin(x) etc.
{\fnH003F4}
The result of the complex operation is called the basic primary function.</p>
<h5>Division Functions</h5>
<p>Like
&#36;&#36;f(x)=\left{\begin{align}
  \varphi(x) ,x\in A\
  \phi(x) ,x\in B
\end{align}\right.&#36;&#36;
Here are some examples.
Symbol Functions
&#36;f(x)=\left(begin{align}
One, x&gt;0\
  0,x=0\
  -1,x&lt;0
\end{align}\right.&#36;&#36;
整数部分函数
&#36;&#36;y=[x]&#36;&#36;
非负小数部分函数
&#36;&#36;(x)=x-[x]&#36;&#36;</p>
<h5>Invisible Functions</h5>
<p>The function that does not clearly identify the relationship between the variable and the variable is usually an equation, and the biggest question here is whether this equation is a function or not.
Here's the Kepper equation, he's a function.
&#36;&#36;y = x+\epsilon siny&#36;&#36;</p>
<h5>Function expressed in the argument equation</h5>
<p>The third variable, indirect function determination, is often built to facilitate the prevention of hidden expressions that do not constitute a function that is important, and then it is explained that
&#36;&#36;\left{\begin{align}
  x=x(t)\
  y=y(t)
\end{align}\right.&#36;&#36;</p>
<h4>Nature of function</h4>
<p>It's a borderline.
Set&#36;y=f(x)&#36;Define Fields As&#36;D&#36;Yes.&#36;D&#36;Inside, if a positive number exists&#36;M&#36;, make it any way&#36;D&#36;Medium&#36;x&#36;Always.&#36;\left|f(x)\right|\leq M&#36;..., otherwise called functions&#36;y=f(x)&#36;Yes.&#36;D&#36;It's a border, also known as&#36;f(x)&#36;Yes.&#36;D&#36;There are lines of function on it. If there is no positive number,&#36;M&#36;, the function is called&#36;y=f(x)&#36;Yes.&#36;D&#36;Up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up, up,&#36;f(x)&#36;Yes.&#36;D&#36;Up is the no-boundary function
Attention, the world is not unique.</p>
<p>Mono-Telephone
Set Functions&#36;f&#36;In the compartment.&#36;I&#36;There's a definition if&#36;I&#36;Any two unequals&#36;x_1&#36;and&#36;x_2&#36;, when &#36;x 1&lt;x_2&#36;时，恒有&#36;f(x_1)&lt;f(x_2)&#36;，则称函数&#36;f&#36;在区间&#36;I&#36;上是增函数；如果恒有&#36;f(x_1)&gt;f(x_2)&#36;，则称函数&#36;f&#36;在区间&#36;I&#36;上是减函数。如果恒有&#36;f(x_1)\leq f(x_2)&#36;或者恒有&#36;f(x_1)\geq f(x_2)&#36;，则称函数&#36;f&#36;在区间&#36;I'm a one-way or one-way.
Note that constant functions are also monotonous, unlike strict monotonous, which is local and not related to continuity and conductivity.</p>
<p>Fancy.
Set Functions&#36;f(x)&#36;Define Fields As&#36;I&#36;, if&#36;I&#36;Up any one&#36;x&#36;Both.&#36;-x\in I&#36;and&#36;f(-x)=-f(x)&#36;, then function&#36;f(x)&#36;It's called a strange function; if it's&#36;I&#36;Up any one&#36;x&#36;Both.&#36;-x\in I&#36;and&#36;f(-x)=f(x)&#36;, then function&#36;f(x)&#36;It's called a do-off function.
Attention, he's a global character.</p>
<p>Periodicity
If there is a non-zero constant&#36;T&#36;,for any of the defined fields&#36;x&#36;Both.&#36;f(x)=f(x+T)&#36;Always, then.&#36;f(x)&#36;It is called a periodic function.&#36;T&#36;A cycle called this function
It's also global.
Life cycles don't necessarily have a minimum positive cycle, for example.
&#36;f(x)=c&#36;, of which&#36;c&#36;As Any Constant
&#36;D(x)=\begin{cases}1 &amp; x\in\mathbb{Q} \ 0 &amp; x\in\mathbb{R}\setminus\mathbb{Q} \end{cases}&#36;、</p>
<p>Two common insularities.
Triangular Instinct
&#36;\vert a+b\vert \leq \vert a\vert + \vert b\vert&#36;</p>
<p>&#36;\vert a-b\vert \geq \vert a\vert - \vert b\vert&#36;</p>
<p>&#36;\vert a-c\vert \leq \vert a-b\vert + \vert b-c\vert&#36;
Mean inconsistent
&#36;\frac{a+b+c}{3}\geq \sqrt[3]{abc}\geq \frac{3}{\frac{1}{a}+\frac{1}{b}+\frac{1}{c&#125;&#125;&#36;
Two dollars or more, of course.</p>
<h4>Actual continuity basis</h4>
<p>Math analysis is mainly based on the reals, so we need to have some basic knowledge of the reals, and continuity is a more important nature of the reals, and we'll be back with more of the theorems about the reals.</p>
<p>Natural numbers are the starting point of the human system, his and the accumulations are closed, but the difference between the two natural numbers is not necessarily natural, so we expand the range of integers and we get a rational number because the division of integers is not closed. Set
It's easy to find in geometry that the whole number is discrete, but the number of rational coordinates of any length is unlimited, and there is no space without reason, which we call denseness.
It doesn't mean that the logical number is perfect, like the square of a square with a square of one, and the length of the angle line is not a rational number.
I'm sorry.
\begin{align*)
\sqrt{2} &amp;= \frac{a}{b} &amp;&amp; \text{Assuming root number 2 is a logical number and a and b are integers of interchanges}
Two. &amp;= \frac{a^2}{b^2} &amp;&amp; \text{two square} \
A couple of minutes. &amp;= 2 b^2 &amp;&amp; \text{Equivalent Deformation} \\
a &amp;= 2 k &amp;&amp; \text{it follows that a is a double, set to 2k, of which k is an integer}
(2k)^2 &amp;= 2 b^2 &amp;&amp; \text{uptick} \\text
- You're not gonna get me out of here. &amp;= 2 k^2 &amp;&amp; \text{Equivalent Deformation} \\
b &amp;= 2 m &amp;&amp; \text{it follows that b is also a 2-m multiple, set at 2 m, of which m is an integer}
} * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
I'm sorry.
So we proved that the hypothesis is wrong, that the root number 2 is an unreasonable number.</p>
<p>These accounts tell us that, although the rational numbers are dense, he's not distributed across the line, but there are gaps, naturally, and we choose to add infinity uncycled decimals to our domains, to create real-digit domains, and in the later learning we will find that the real-digit domains are continuous, and now we need to add some more knowledge.</p>
<h4>Additional knowledge</h4>
<h5>Maximum and minimum</h5>
<p>If it's any way to go&#36;b\in A&#36;Both.&#36;a\geq b&#36;, then&#36;a&#36;Yes.&#36;A&#36;(max)&#36;\max A=a&#36;
The same reason gives a definition of the minimum number.
Attention, there must be a maximum and a minimum number of limited series, and an infinite set is not necessarily required.</p>
<h5>High and low</h5>
<p>The concept of boundaries has been mentioned earlier, and the assembly of the upper and next generation is called the boundary.
Set Numeric Set&#36;S&#36;There's a high level of memory.&#36;U&#36;It's a collection of all the top levels. We'll give you the results. &#36;U&#36;There must be a minimum number.&#36;S&#36;The best of the rules of the same standard, the best of the same rules, and usually the same of the same rules.&#36;sup和inf&#36;
The Supreme Court of Justice has the following character:
&#36;\beta&#36;It's a series.&#36;S&#36;The upper bounds of the \forall x SS~x&lt;=\beta&#36;
&#36;\forall \epsilon &gt;0 ~~\exists x\in S ~~x&gt;\beta-\epsilon&#36;
That's the principle of certainty, and we'll prove to him in the theory of the integrity of the facts--</p>
<h2>Column limit</h2>
<h3>Definition of the numbering line limits</h3>
<h3>Nature of column limits</h3>
<h3>A few exercises at the limit of the array.</h3>
<h3>Infinity.</h3>
<h2>Theories of Physical Completeness</h2>
<p>The theorem of physical completeness is the theory of continuity of the actual numbers, and so-called completeness is the theory of continuity, but it focuses on different aspects, and many of the the theorems reflect the completeness of the actual numbers, and we will give them all the equivalents in the final certificate.
Two core questions: 1. When the series of bounds will be reduced</p>
<h3>The principle of certainty</h3>
<p>The principle of certainty is the one in which we have not been able to prove it in the upper part of the border, and he describes it intuitively:<strong>There must be a clear line of action for the upper level of the non-empty.
Proof of:</strong>
 &#36;&#36;&#36;\forall a\in R~~ x=[x]+(x) <del>There must be (x)=0.a.  a.  ...  n.   &#36;
Set&#36;S&#36;It's a collection of non-empty upper bounds. &#36;S&#36;All the elements in it are.&#36;[x]+(x)&#36;Composition&#36;[x]=a_{0}&#36; Because&#36;S&#36;There's a top line.&#36;a_{0}&#36; As&#36;\alpha_{0}&#36;  And now we're gonna do this.&#36;S_{0}&#36; Other Organiser</del> and~~[x]=\alpha_{0} }&#36;重复这个过程 给出&#36;Wait
At this point, we think it's clear.&#36;\beta&#36;Yeah.&#36;\alpha_{0}.\alpha_{1}\alpha_{2}....\alpha_{n}....&#36;
We can actually be sure that he's the upper bounds and the smallest.
Intuitively, understand the relationship between the aberration and the continuity of the real numbers.&#36;x_{0}&#36; The whole assembly of real numbers on the right here is a bottom line, but he's not the bottom line that he should have.&#36;x_{0}&#36; The left is the same.</p>
<h3>Single-to-concentrate principle</h3>
<h2>Function limit</h2>
