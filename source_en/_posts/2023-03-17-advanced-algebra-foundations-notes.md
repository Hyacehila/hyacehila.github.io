---
title: 'Advanced Algebra: Algebra Foundations'
title_zh: 高等代数：代数学基础
date: 2023-03-17 11:02:24 +0800
categories:
- Mathematics
- Algebra & Matrix Theory
tags:
- Algebra
author: Hyacehila
mathjax: true
hidden: true
excerpt: Covers polynomials, determinants, Gaussian elimination, systems of linear equations, and related algebraic basics.
description: Covers polynomials, determinants, Gaussian elimination, systems of linear equations, and related algebraic basics.
excerpt_zh: 整理多项式、行列式、高斯消元、线性方程组和相关代数基础。
permalink: /blog/2023/03/17/advanced-algebra-foundations-notes/
lang: en
translation_key: 2023-03-17-advanced-algebra-foundations-notes
translation_status: machine
translation_source_hash: 94d9c47f1cade26e4bd6a53863bb33c25b3c4044d068a1acc61699737bacc183
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<h2>Multiple</h2>
<h3>Domains and Multiples</h3>
<p>Definition (digital domain): An algebra structure in which the number of elements and the amount of differentials within the pool are still referred to as digital domains.</p>
<p>Definitions (unististic): as follows:&#36;a_{n}x^{n}+a_{n-1}x^{n-1}+\cdots+a_{0}&#36; of which&#36;a_i&#36;In Range&#36;P&#36;Go, we call it the digital domain.&#36;P&#36;Up on the one dollar multiform.</p>
<p>Zero-multiple: all coefficients are 0; zero-multiple: maximum zero, or constant.</p>
<h3>Multi-dimensional</h3>
<p>Multi-formation, subtraction, multiplication can be easily handled with high school knowledge, and here we explore the less developed multi-formation. The government has been able to provide the necessary information to the public.</p>
<p>& Use a polygon  &#36;f(x)&#36;  Divide by another non-zero multiplication  &#36;g(x)&#36; ♪ Get me a deal ♪  &#36;q(x)&#36;  Balance  &#36;r(x)&#36; I'm sorry. The following relationships are satisfied:
&#36;f(x)=g(x)q(x)+r(x),\quad\dec(r)&lt;\deg(g(x)). &#36;
Naturally, divided.&#36;f(x)&#36;There should be a higher number of times, otherwise it cannot be carried out.</p>
<p>Example:&#36;\text{算 }f(x)=x^3-4x^2+6x-8\text{ 除以 }g(x)=x-2&#36;
I'm sorry.
\begin{array}ll}
\\text{step 1: first of the calculator} &amp; \frac{x^3}{x} = x^2, \
 &amp; \cdot(x-2) =x^3 -2x2. \
\text{updated decomposed:} &amp; (x^3 - 4x^2 + 6x - 8) - (x^3 - 2x^2) = -2x^2 + 6x - 8. \[10pt]</p>
<p>\text{step 2: the second step of the calculator} &amp; \frac{-2x^2}{x} = -2x, \
 &amp; = -2x^2+4x. \
\text{updated decomposed:} &amp; (-2x^2 + 6x - 8) - (-2x^2 + 4x) = 2x - 8. \[10pt]</p>
<p>\\text{step 3: Calculator's third} &amp; \frac{2x}{x} = 2, \
 &amp; 2\cdot(x-2)=2x-4 \
\text{updated decomposed:} &amp; (2x - 8) - (2x - 4) = -4. \[10pt]</p>
<p>\\text{final result:} &amp; f(x) = (x^2 - 2x + 2)(x - 2) - 4.
\end{array}
&#36;&#36;</p>
<p>When the rest is zero, we write.&#36;f(x)=g(x)h(x)&#36;, whichever is&#36;f(x)&#36;The factor, called the whole relationship, and it's written down.&#36;g(x)|f(x)&#36;</p>
<p>We have the following characteristics for this relationship:</p>
<ul>
<li>&#36;f(x)|g(x)~~~g(x)|f(x)\Rightarrow f(x)=cg(x)&#36;</li>
<li>Passivity:&#36;f(x)|g(x)~~~g(x)|h(x)\Rightarrow f(x)|h(x)&#36;</li>
<li>Grouping:&#36;f(x)|g_{i}(x)\Rightarrow f(x)|g(x)\text{ 的任意线性组合}&#36;</li>
</ul>
<h3>Consolidated division</h3>
<p><strong>Consolidated division</strong>(Synthetic Division) is an efficient algorithm for after-effects of multiple formats. When detached  &#36;f(x)&#36;  The divided is the form  &#36;x - c&#36;  When a combination of multiples is taken, the composite division can quickly yield the balance. It is simpler and more efficient than the traditional multiple-dimensional division.</p>
<p>For a composite division
&#36;&#36;a_nx^n+a_{n-1}x^{n-1}+\cdots+a_0=(x_0-c)(b_{n-1}x^{n-1}+b_{n-2}x^{n-2}+\cdots+b_0)&#36;&#36;
We have a pushover down there.
&#36;&#36;b_{n-1}=a_{n}\quad b_{n-2}=a_{n-1}+cb_{n-1}(a_{n-1}=b_{n-2}-cb_{n-1})&#36;&#36;</p>
<p>About squareding, that's the following multiform expression.
&#36;&#36;a_{n}(x_{0}+c)^{n}+a_{n-1}(x_{0}+c)^{n-1}+\cdots+a_{0}&#36;&#36;
We just have to keep getting rid of them.&#36;x-c&#36;Use a composite division.</p>
<p>If the original is divided,&#36;ax+b&#36; And it should be:&#36;a(x+\frac{b}{a})&#36;  And then we'll take it out.&#36;a&#36;</p>
<h3>Max. Cause</h3>
<p>Definitions: For polygons, the largest public factor (Greate Common Divisor, GCD) refers to the highest number of factors in two or more combinations. The maximum public factor is the two multi-formulture all-factor factor.</p>
<p>Introduction:
&#36;&#36;f(x)=q(x)g(x)+r(x)\Rightarrow f(x),g(x)\text{ 与 }g(x),r(x)\text{ 有相同公因式}&#36;&#36;</p>
<p>According to this reasoning, the approach to the most public-caused approach is clear.</p>
<p>We'll use it first.&#36;f(x),g(x)&#36;, and then remove the rest.&#36;r(x)&#36;, divided&#36;g(x)&#36; , divided by the medium of the two by the low of the remaining zero. The last residual is the biggest public factor.</p>
<p>If you can't spare zero, the maximum public factor is zero, which is&#36;(f(x),g(x))=1&#36; It's called polymix.</p>
<p>The following conclusions can be made about the hyphene.</p>
<ul>
<li>&#36;(f(x),g(x))=1\Longleftrightarrow u(x)f(x)+v(x)g(x)=1&#36;</li>
<li>&#36;(f(x),g(x))=1\quad f(x)|g(x)h(x)\quad\Rightarrow f(x)|h(x)&#36;</li>
<li>&#36;f_1(x)|g(x),f_2(x)|g(x)\text{且}(f_1(x),f_2(x)=1\Rightarrow f_1(x)f_2(x)|g(x)&#36;</li>
</ul>
<p>Theoretically:&#36;\text{若 }d(x)=(f(x),g(x))\text{ 则 }\exists v(x) u(x) \text{ 使得 } d(x)=v(x)f(x)+u(x)g(x)&#36;</p>
<p><strong>The theorem's proof needs to be transected until calculated&#36;d(x)&#36;And then you can keep bringing the rest from top to bottom.</strong></p>
<h3>Interpretative Theorem</h3>
<p>Watch the pattern below.
&#36;&#36;00begin{matrix}x{4}-2=(x^2}(x^2}+2)
=(x+\sqrt{2} (x-\sqrt{2})
=sqrt{2}=(x-\sqrt{2}(x+sqrt{2}i)</p>
<p>\end{matrix}&#36;&#36;</p>
<p>It is easy to see whether the causal breakdown can continue depends on multiple digital selections.<strong>Only a clear study of the numerical dimensions will cause a causal breakdown.</strong></p>
<p>Definitions: in digital domains&#36;P&#36;Go on, if&#36;p(x)&#36;Not decompose to two lower polygons, otherwise called&#36;p(x)&#36;It's digital.&#36;P&#36;On the improbable polygon.</p>
<p>So you can give theorem: no multiform.&#36;p(x)&#36;..the only factor is&#36;c,cp(x)&#36;The reverse theorem is also established.</p>
<p>Same thing. For non-prescribed polygons.&#36;p(x)&#36;Yes.&#36;(p(x),f(x))=1&#36; or &#36;(p(x),f(x))=p(x)&#36;;if&#36;p(x)|f(x)g(x)&#36; then &#36;p(x)|f(x)&#36;or&#36;p(x)|g(x)&#36;</p>
<p>Theorem (caused breakdown of theorem): any polygonal&#36;f(x)=p_{1}(x)p_{2}(x)\cdots=q_{1}(x)q_{2}(x)\cdots&#36;, must have experienced several changes&#36;p_{i}(x)=kq_{i}(x)&#36;, of which&#36;p(x),q(x)&#36;Neither. That is, the breakdown is unique.</p>
<h3>Factorial</h3>
<p>Definitions: If&#36;p^{k}(x)\mid f(x)&#36; and &#36;p^{k+1}(x)\nmid f(x)&#36; Name&#36;p(x)&#36;Yes.&#36;f(x)&#36;Yes.&#36;k&#36;Retroactivity, he'll be in our standard decomposition.</p>
<p>It's not hard to find out if&#36;p(x)&#36;Yes.&#36;f(x)&#36;Yes.&#36;k&#36;Retroactive, then.&#36;p(x)&#36;It's &#36;f}&#39;(x)&#36;的&#36;k-1&#36;重因式，当求&#36;k&#36;阶导的时候，&#36;P(x)&#36; is no longer a factor</p>
<p>If&#36;p(x)&#36;It's &#36;f}&#39;(x)&#36;的&#36;k-1&#36;重因式，不能保证&#36;p(x)&#36;是&#36;f(x)&#36;的&#36;k&#36;重因式，因为可能差一个常数，但是如果是因式的话，那一定是&#36;k&#36; heavy.</p>
<p>No-factor-based-filled condition &#36; (f(x),{f}&#39;(x))=1&#36;，&#36;f(x)&#36;的重因式为&#36;(f(x),{f}&#39;(x) That's the most common way to study the regenerative approach.</p>
<p>Common treatment:</p>
<ul>
<li>Find factor, calculate &#36; (f(x), {f}&#39;(x))&#36;</li>
<li>Go to re-factor, calculate &#36; (f(x), {f}&#39;(x))=p^k(x)&#36; 则重因式为&#36;p^{k+1}(x)&#36; 则&#36;The \frac{f(x)}p^1}(x)} &#36; achieves de-factory.</li>
</ul>
<h3>Multiple Functions</h3>
<p>Multiple functions create a link between this algebra problem and the analytical problem of functions, which is used in many of the problems.</p>
<p>Definitions: term&#36;{f(x)=a_{n}x^{n}+\cdots+a_{0&#125;&#125;&#36; As Multi-Functions</p>
<p>Theorem (residual): used&#36;x-\alpha&#36;Divide by Multiple&#36;f(x)&#36;We can get it.&#36;f(x)=(x-\alpha)h(x)+c&#36;  There are two excellent things about this division.</p>
<ul>
<li>Multi-Acquisition&#36;x=\alpha&#36; And get the rest of it. &#36;c&#36;</li>
<li>&#36;(x-\alpha)|f(x)\Longleftrightarrow\alpha 是f(x)=0的一个解&#36;</li>
</ul>
<p>Definitions: If&#36;x-\alpha&#36;Yes.&#36;f(x)&#36;Yes.&#36;k&#36;heavyfactor, otherwise called&#36;\alpha&#36;Yes.&#36;f(x)&#36;Yes.&#36;k&#36;Root</p>
<p>Theoretically:&#36;n&#36;Multiples Up to&#36;n&#36;Roots. Weights.</p>
<p>Theorem: Different multidimensionals cannot define the same function</p>
<p>Theorem: For two polygons&#36;f(x),g(x)&#36; If there is.&#36;n+1&#36;individual&#36;\alpha_i&#36;Make&#36;f(\alpha_i)=g(\alpha_i)&#36; There is.&#36;f(x)=g(x)&#36;</p>
<p>Theorem (Vedda Theorem): Yes&#36;n&#36;Multi-form&#36;n&#36;- Yeah.</p>
<ul>
<li>&#36;\sum_{i=1}^{n}x_{i}=-\frac{a_{n-1&#125;&#125;{a_{n&#125;&#125;&#36;</li>
<li>&#36;\prod_{i=1}^{n}x_{i}=(-1)^{n}\frac{a_{0&#125;&#125;{a_{n&#125;&#125;&#36;</li>
</ul>
<h3>Multiples of Different Coefficient Fields</h3>
<h4>Multiple Multiple Coefficients</h4>
<p>Multiform fundamentals based on Gauss: multiple coefficients can be broken down into multiple, multiple-formation volumes, i.e., the following decompositions:
&#36;&#36;f(x)=a_{n}(x-\alpha_{1})^{l_1}(x-\alpha_{2})^{l_{2&#125;&#125;\cdots&#36;&#36;</p>
<p>And that's why.&#36;n&#36;There must be a multiplication factor.&#36;n&#36;Roots. Weights.</p>
<h4>Real coefficients</h4>
<p>Theorem: For the actual coefficient formula, if&#36;\alpha&#36; Yes. &#36;f(x)&#36;A replica, then.&#36;\bar{\alpha}&#36;It must be.&#36;f(x)&#36;Another root.</p>
<p>That is, multiple multiple root of the real coefficient, separate root, one multi-factor, and two multi-factor.</p>
<p>Decomposition to
&#36;&#36;f(x)=a_{n}(x-c_{1})^{l_{1&#125;&#125;(\ldots)(x^{2}+p_{1}x+q_{1})^{k_{1&#125;&#125;(\ldots)\cdots&#36;&#36;</p>
<h4>Modular Multiformation</h4>
<p>The existence of an improbable factor of random frequency in the logarithmic multidimensionality has led to the complexity of the question of the coefficient plethora, and we can only discuss some of the typical problems.</p>
<p>For the GSR polygon&#36;f(x)&#36;He can actually turn into a multi-factor by multiplying it by a factor.&#36;g(x)&#36;</p>
<p>Definition: coefficient intersyncs&#36;g(x)&#36;Called the original polygon</p>
<p>The coefficient polygons then have at most two corresponding original polygons, and a negative sign is missing from the two. The study of the logarithmic polygraphs can be translated into the original multi-formulation.</p>
<h3>Multiform Important Theorem</h3>
<h4>This original polygon</h4>
<p>Theorem (Gorse Introduction): Two original polygraphs are either the size of one original polygraph or the size of one original polygraph</p>
<p>Theorem (decomposition theorem): When the integer multiformation can be broken down into two inter-format accumulations, he can certainly be decomposed into two original polygons.</p>
<p>Theorem (decomposition of theorem inverse)&#36;f(x),g(x)&#36;It's the integer multiform.&#36;g(x)&#36;It's the original pluriform.&#36;f(x)=g(x)h(x)&#36; and &#36;h(x)&#36;There is a coefficient polyformula. then&#36;h(x)&#36;It's the integer multiplication.</p>
<h4>r s Theorem</h4>
<p>&#36;\frac{r}{s}&#36;Theorem is the rationale for studying the whole coefficient.</p>
<p>Theorem: For the integer polygon&#36;f(x)=a_{n}x^{n}+\cdots+a_{0}&#36;  If&#36;\frac{r}{s}&#36;It's one of his roots. There must be one.&#36;r|a_0,s|a_n&#36;  If&#36;a_n=1&#36; Well...&#36;f(x)&#36;All the root is the integer and is&#36;a_0&#36;The factor.</p>
<p>This is the theorem that tells us to find it.&#36;a_0,a_n&#36;All the factors that combine to produce are all the possible root, and all the root that can be found in a surrogate study.</p>
<p>It's a guess.&#36;+\frac{r}{s},-\frac{r}{s}&#36;</p>
<h4>Eisenstein's sentence is different.</h4>
<p>This way we can help us judge that an integer multiformation is not possible under a rational numerical domain. About</p>
<p>Theorem: For the integer polygon&#36;f(x)=a_{n}x^{n}+\cdots+a_{0}&#36;  , if there is a prime number&#36;p&#36; Satisfied &#36;p\nmid a_{n},p\mid a_{n-1}...a_{0},p^{2}\nmid a_0&#36; then&#36;f(x)&#36;Not in reasonable numerical terms.</p>
<p>Note:</p>
<ul>
<li>It's only a condition, not a necessity.</li>
<li>This method requires a significant number of known multi-form coefficients.</li>
<li>When a lot of coefficients are not clear, they can be considered.&#36;x+1,x-1&#36;Replace&#36;x&#36;And then we'll judge because...&#36;f(x)&#36;There's no root and no...&#36;f(ax+b)&#36;No root is the same price.</li>
</ul>
<h2>Row</h2>
<h3>Introduction</h3>
<p>The solution of equation groups is a very important issue in mathematics, and both the ode pde and the normal algebra equations are very worthy of our study.</p>
<p>For the binary One-Equation Group
I'm sorry.
\begin{align}
a {11}x 1+a {12}x 2&amp;=b_1\
a_{21}x_1+a_{22}x_2&amp;=b 2
\end{align}
I'm sorry.
If &#36;a_{11}a_{22}-a_{12}a_{21}\ne 0&#36; So the coefficient is not equal to zero, and the only solution to the equation is that the one that we're going to be working on is what we're going to be working on.</p>
<h3>Arrange</h3>
<p>Definitions: An orderly array of 1, 2, 3, 4 is called a&#36;n&#36;Order</p>
<p>The core point is an orderly array and a limited number of numbers.</p>
<p>The large-sized ranking is called natural rankings, as 1,2,3 4; the situation where the large numbers are ahead of the small numbers is referred to as reverse orders, as in 1,3 2, and whenever a group of such situations occurs, we reverse the order +1 as&#36;T(j_{1} j_{2}  j_{n})&#36;
Like &#36;T(1234)=0&#36; &#36;T(2134)=1&#36;</p>
<p>Maximum value of the inverse number is&#36;C_{n}^{2}&#36; That means that every element is in reverse order.</p>
<p>The reverse sequence is called even.</p>
<p>Exchange: Any two numbers in the exchange order will change the oddity of the order.</p>
<p>Inference: All&#36;n&#36;In the order of the order, the number of odd and even is equal to the number of odd and even.&#36;n!\div 2&#36;  All of it.&#36;n&#36;It's all in the order.&#36;n&#36;The number of shifts in the hierarchy is the number of shifts in the order of the number of dolls.</p>
<h3>N-steps</h3>
<p>&#36;&#36;
\begin{vmatrix}
a_{11} &amp; a_{12} \
a_{21} &amp; a {22}
} \ } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } }
♪ I'm gonna be a little bit more like a little girl ♪
I'm sorry.
This is the minimum calculation equation of the rows of the rows, which are naturally arranged by the negative number of the columns, i.e., the negative number of the columns.&#36;(-1)^{T(列)}&#36;and each row label should not be equal and do not appear in the same row or column Marriage Multiply</p>
<p>Here are some of the common basic forms.</p>
<p>&#36;&#36;
\begin{vmatrix}
a_{11} &amp; a_{12} &amp; \cdots &amp; a_{1n} \
a_{21} &amp; a_{22} &amp; \cdots &amp; a_{2n} \
\vdots &amp; \vdots &amp; \ddots &amp; \vdots \
a_{n1} &amp; a_{n2} &amp; \cdots &amp; It's a...
} \ } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } }
== sync, corrected by elderman == @elder man
I'm sorry.
It's the core form of the one-form.</p>
<p>&#36;&#36;
\begin{vmatrix}
a_{1} &amp; 0 &amp; 0 &amp; 0 \
0 &amp; a_{2} &amp; 0 &amp; 0 \
0 &amp; 0 &amp; a_{3} &amp; 0 \
0 &amp; 0 &amp; 0 &amp; a {4}
} \ } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } }
=a b2a 3 a 4}
I'm sorry.
It's a diagonal.</p>
<p>&#36;&#36;
\begin{vmatrix}
0 &amp; 0 &amp; 0 &amp; a_{1} \
0 &amp; 0 &amp; a_{2} &amp; 0 \
0 &amp; a_{3} &amp; 0 &amp; 0 \
a_{4} &amp; 0 &amp; 0 &amp; Photo by Flickr user @un.org
} \ } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } }
== sync, corrected by elderman == @elder man
I'm sorry.
This is the opposite side of the curve, and the last one to multiply is the reverse sequence.&#36;C_{n}^{2}&#36;</p>
<p>&#36;&#36;
\begin{vmatrix}
a_{11} &amp; a_{12} &amp; \cdots &amp; a_{1n} \
0 &amp; a_{22} &amp; \cdots &amp; a_{2n} \
\vdots &amp; \vdots &amp; \ddots &amp; \vdots \
0 &amp; 0 &amp; \cdots &amp; It's a...
} \ } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } }
=a 11}a 22}a 33}a 44}a n}a n}
I'm sorry.
It's a triangle. It's a reverse triangle. Just add the reverse sequence.</p>
<p>And finally, we're going to come up with a general conclusion.
Insumption: We can study the reverse numbers not only in the lines when they're natural, but also in the lines when they're natural, or in the random order and expression of the lines and the columns.</p>
<p>This means that the line in the rows and the column in the rows are different in the sense that they are given, and in the theory of mathematics, there should be no distinction, and we will explain it further in the next section.</p>
<h3>Type of column</h3>
<h4>Convert</h4>
<p>The value of the rows is the same.</p>
<h4>The G.I.D.</h4>
<p>Here's a definition that can be used in a row.
I'm sorry.
\begin{vmatrix}a&amp;a_{12}&amp;\cdots&amp;a_{1n}\a_{21}&amp;a_{22}&amp;\cdots&amp;a_{2n}\\vdots&amp;\vdots&amp;\ddots&amp;\vdots\ka_{i1}&amp;ka_{i2}&amp;\cdots&amp;ka_{in}\\vdots&amp;\vdots&amp;\ddots&amp;\vdots\a_{n1}&amp;a_{n2}&amp;\cdots&amp;a_{nn}\end{vmatrix}
=k<br>\begin{vmatrix}a_{11}&amp;a_{12}&amp;\cdots&amp;a_{1n}\a_{21}&amp;a_{22}&amp;\cdots&amp;a_{2n}\\vdots&amp;\vdots&amp;\ddots&amp;\vdots\a_{i1}&amp;a_{i2}&amp;\cdots&amp;a_{in}\\vdots&amp;\vdots&amp;\ddots&amp;\vdots\a_{n1}&amp;a_{n2}&amp;\cdots&amp;a_{nn}\end{vmatrix}
&#36;&#36;</p>
<h4>Value is 0</h4>
<p>Depending on the previous properties, a single row of zero is a zero because they can propose a factor zero.</p>
<h4>Split</h4>
<h1>&#36;&#36;
\begin{vmatrix}a_{11}&amp;a_{12}&amp;\cdots&amp;a_{1n}\a_{21}&amp;a_{22}&amp;\cdots&amp;a_{2n}\\vdots&amp;\vdots&amp;\ddots&amp;\vdots\a_{i1}+b_{i1}&amp;a_{i2}+b_{i2}&amp;\cdots&amp;a_{in}+b_{in}\\vdots&amp;\vdots&amp;\ddots&amp;\vdots\a_{n1}&amp;a_{n2}&amp;\cdots&amp;a_{nn}\end{vmatrix}</h1>
<p>\begin{vmatrix}a_{11}&amp;a_{12}&amp;\cdots&amp;a_{1n}\a_{21}&amp;a_{22}&amp;\cdots&amp;a_{2n}\\vdots&amp;\vdots&amp;\ddots&amp;\vdots\a_{i1}&amp;a_{i2}&amp;\cdots&amp;a_{in}\\vdots&amp;\vdots&amp;\ddots&amp;\vdots\a_{n1}&amp;a_{n2}&amp;\cdots&amp;a_{nn}\end{vmatrix}+\begin{vmatrix}a_{11}&amp;a_{12}&amp;\cdots&amp;a_{1n}\a_{21}&amp;a_{22}&amp;\cdots&amp;a_{2n}\\vdots&amp;\vdots&amp;\ddots&amp;\vdots\b_{i1}&amp;b_{i2}&amp;\cdots&amp;b_{in}\\vdots&amp;\vdots&amp;\ddots&amp;\vdots\a_{n1}&amp;a_{n2}&amp;\cdots&amp;It's a...
I'm sorry.
It's a conclusion that it's a whole line of splits at a time, so it might take a little bit of a construction to satisfy the need.</p>
<h4>Inference 1</h4>
<p>The rows have two rows and one row of one. The one-in-a-row value is 0.
Proof: There must be two reverse sequences that are different but of the same absolute value.</p>
<h4>Inference 2</h4>
<p>One in one in two rows.
Proof: Combined inference 1 and nature of the cause of the communication</p>
<h4>Inference 3</h4>
<p>Add the k times of one line to the other.
Proof: Evidence of the nature of the citation factor and the value of 0 in a column</p>
<h4>Inference 4</h4>
<p>Exchange two rows of the same value in the same row as the original.
Proof: The exchange of two rows is achieved by adding a row of k to 3</p>
<h3>Normal spread of rows</h3>
<p>The normal extension is for talking to a row or column.
I'm sorry.
The \begin{aligned}
\left\begin{array}
a {11} &amp; a_{12} &amp; \cdots &amp; a_{1n} \
a_{21} &amp; a_{22} &amp; \cdots &amp; a_{2n} \
\vdots &amp; \vdots &amp; \ddots &amp; \vdots \
a_{n1} &amp; a_{n2} &amp; \cdots &amp; a_{nn}
\end{array}\right|
&amp;=a_{11}\left|\begin{array}{ccc}
a_{22} &amp; \cdots &amp; a_{2n} \
\vdots &amp; \ddots &amp; \vdots \
a_{n2} &amp; \cdots &amp; a_{nn}
\end{array}\right| -a_{12}\left|\begin{array}{ccc}
a_{21} &amp; \cdots &amp; a_{2n} \
\vdots &amp; \ddots &amp; \vdots \
a_{n1} &amp; \cdots &amp; a_{nn}
\end{array}\right| +\cdots+(-1)^{1+n}a_{1n}\left|\begin{array}{ccc}
a_{21} &amp; \cdots &amp; a_{n-1,2} \
\vdots &amp; \ddots &amp; \vdots \
a_{n1} &amp; \cdots &amp; a_{n-1,n}
\end{array}\right|\
&amp;
@aligned BAR
I'm sorry.
In fact, we can see that the rows that follow the spread are the lower rows that are drawn from one row and one row.
<strong>The point of the wave is to reduce the number of steps in the rows and simplify our calculations.</strong></p>
<p>Definitions:&#36;n&#36;In the row, cross it off.&#36;a_{ij}&#36;Where it is&#36;i&#36;Line and&#36;j&#36;Row, get one.&#36;n-1&#36;The ranks, we call them the extras.&#36;M_{ij}&#36;</p>
<p>Definitions: term&#36;A_{ij}=(-1)^{i+j}~~M_{ij}&#36;As algebra residual</p>
<p><strong>Expands the form that can easily use algebra and residual expressions</strong></p>
<p>Inference: In a row, the algebra of one line element and another line elements is zero.</p>
<p>Obviously, only a large 0-o-row expansion of a row or column can help to reduce the difficulty of running a line or column.</p>
<h3>The Rapras is on.</h3>
<p>In fact, the expansion can be directed at multiple rows or multiple rows, and that's the La Plass expansion, and the general extension is a special form of La Plas.</p>
<p>Promote residual and algebra residuals:&#36;n&#36;Liner Call&#36;k&#36;Okay.&#36;k&#36;Column, nodal point&#36;k^2&#36;One element, let them make one in the original order.&#36;k&#36;The stairwell, the rest of it.&#36;n-k&#36;Step stub. In fact, they're equal after the shift. They're sub-synthetic.</p>
<p>Definition: Algebra residual is &#36;{A}  &#39; =  {M}  &#39; -1^{i_1+i_2+...+j_1+j_2+...}&#36; 其中 &#36;i,j&#36;是子式取的行列 &#36;M&#36; is the residual.</p>
<p>Introduction: Any sub-form&#36;M&#36;And his algebra residual.  &#39;The accumulation is one of the things that's going on.</p>
<p>Theoretically (Rapras): Yes&#36;k&#36;Rows (columns) need to be expanded&#36;k&#36;Line (column) Find All&#36;k&#36;Step (set as&#36;t&#36;) and the Rapras began as&#36;D=M_1A_1+\cdots+M_tA_t&#36;Get the original one&#36;D&#36;Value</p>
<p>Note:</p>
<ul>
<li>Make sure you go through all the sub-forms, that is,&#36;t=C_{n}^{k}&#36;</li>
<li>When expanding, the order of rows and columns cannot change</li>
</ul>
<h3>Vantermond's horde.</h3>
<p>We call the next form of the column the Vantermond.
&#36;&#36;00\mmatrix}
One.&amp;1  &amp;1  &amp;1  &amp;\cdots  &amp;1 \
  a_1&amp;a_2  &amp;a_3  &amp;a_4  &amp;\cdots  &amp;a_n \
  a_1^2&amp;a_2^2  &amp;a_3^2  &amp;a_4^2  &amp;\cdots  &amp;a_n^2 \
  \cdots&amp;\cdots  &amp;\cdots  &amp;\cdots  &amp;\cdots  &amp;\cdots \
  a_1^{n-1}&amp;a_2^{n-1}  &amp;a_3^{n-1}&amp;a_4^{n-1}  &amp;\cdots  &amp;a_n^{n-1}
\end{vmatrix}&#36;&#36;</p>
<p>This one's worth is all.&#36;a_i-a_j&#36;♪ The size of which ♪&#36;1\le i \le j\le n&#36; So he had a condition of zero.&#36;a_i=a_j&#36;</p>
<p>To prove the nature of the vantermund type of value, which requires a generalization, we need to know here that the value of such a particular form of type is very easy to calculate.</p>
<h3>Classic exercise in a row</h3>
<h4>Aside method and arrow type</h4>
<p><strong>The plurilateral approach is a reverse method of using the expansion, with the idea of adding a dimension of a row of 10...0 (column) which is freely chosen. This addition can create a column that is easy to simplify and has no change in the value of the column.</strong></p>
<p>Request the value of the next row
&#36;&#36;00\mmatrix}
a&amp;  b&amp;  b&amp;b \
  b&amp;  a&amp;  a&amp;a \
  ...&amp;  ...&amp; ... &amp;a
\end{vmatrix}&#36;&#36;</p>
<p>We'll start with the addition.&#36;b&#36; Easy to simplify, as follows:
&#36;&#36;00\mmatrix}
One.&amp;  0&amp;  ...&amp;0   \
  b&amp;  a&amp;b  &amp;b   \
  b&amp;  b&amp;  a&amp;b   \
  b&amp;  ...&amp; ... &amp; a
\end{vmatrix}&#36;&#36;</p>
<p>Add the first line to the back.
&#36;&#36;00begin{vmatrix}1&amp;-1&amp;-1&amp;\cdots&amp;-1\b&amp;a-b&amp;&amp;\b&amp;a-b&amp;&amp;\b&amp;&amp;\ddots&amp;a-b\end{vmatrix}&#36;&#36;</p>
<p>We got an arrow line.</p>
<p><strong>For arrows, we usually put the last row.&#36;-\frac{b}{a-b}&#36; Add to the first column, get zero and repeat it, and then turn the first column into zero, except for the first row, and that's a triangle.</strong></p>
<p>The topic is one of the most important aspects of the topic:<strong>The use of rows and equivalents will be based on&#36;2&#36;Present.&#36;n&#36;Add all rows to first row</strong>
&#36;&#36;\begin{vmatrix}a+(n-1)b&amp;b&amp;\cdots&amp;b\a+(n-1)b&amp;a&amp;b&amp;b\a+(n-1)b&amp;b&amp;\cdots&amp;A \end{vmatrix}
At this point, we can use the one column to simplify the following column.</p>
<h4>Inverse Row</h4>
<p>When?&#36;a_{ij}=a_{ji}&#36;and when a row is a symmetrical one;&#36;a_{ij}=-a_{ji}&#36;and when the row is a cross-bar;</p>
<p>Proof: The value of the odd-numbered inverse inverted row is 0</p>
<p>We need to use a slightly more sophisticated structure to prove it.</p>
<p>It's easy to know.&#36;a_{ii}=-a_{ii}&#36; then &#36;a_{ii}=0&#36;  So there is.
&#36;d=begin{vmatrix}
Photo by Flickr user @un.org&amp;\text{d}
\text{negative}&amp;0
\end{vmatrix}=\begin{vmatrix}
  0&amp;\text{negative} \
\text {Correct}&amp;0
\end{vmatrix}=(-1)^n\begin{vmatrix}
  0&amp;\text{d}
\text{negative}&amp;0
\end{vmatrix}&#36;&#36;
所以
&#36;&#36;d =(-1)^nd&#36;
And...&#36;d&#36;It's odd, so...&#36;d=0&#36;</p>
<h4>Partition</h4>
<p>&#36;&#36;\begin{vmatrix}
  a_1&amp;  a_2&amp;  0&amp;0 \
  a_3&amp;  a_4&amp;  0&amp;0 \
  0&amp;0  &amp;b_1  &amp;b_2 \
  0&amp;  0&amp;b_3  &amp;b_4
\end{vmatrix}=\begin{vmatrix}
  a_1&amp;a_2 \
  a_3&amp;a_4
\end{vmatrix}\times \begin{vmatrix}
  b_1&amp;b_2 \
  b_3&amp;b 4
You're not gonna get me out of here.
That is, the original ceremonial unicorn conclusion is valid for the division, and the rest of it is no longer probative.</p>
<h4>Incisive and summarized</h4>
<p>Request the value of the next row
&#36;&#36;00\mmatrix}
x&amp;  0&amp;  0&amp;  \cdots &amp;a_0 \
  -1&amp;  x&amp;  \vdots&amp;  \vdots&amp;a_1 \
  0&amp;  -1&amp;  x&amp;  \vdots&amp;a_2 \
  \cdots &amp;  0&amp;  -1&amp;  x&amp;\vdots  \
  0&amp;  \cdots &amp; 0 &amp;  -1&amp;x+a_{n-1}
\end{vmatrix}&#36;&#36;</p>
<p>First, we'll use the calculations to eliminate it.&#36;x&#36;  At this point, on the diagonal line,&#36;x&#36;It's all eliminated.</p>
<p>Then we'll open the first line, and the rest of the row is the angle.&#36;-1&#36;The triangle by which
&#36;&#36;(-1)^{n-1}\times(\cdots x(x(x(x+a_{n-1})+a_{n-2})+a_{n-3})\cdots)&#36;&#36;</p>
<p>This is a...&#36;n&#36;Sub-multiple approach, with the following extrapolation, with the help of the similarity after expansion
&#36; \begin{matrix}
== sync, corrected by elderman ==
♪ I'm not gonna let you go ♪
... =xD 2} +a n-3}</p>
<p>\end{matrix}
&#36;&#36;</p>
<p>Finally.
&#36;&#36;D_{n}=x^{n}+a_{n-1}x^{n-1}+\cdots+a_{1}x+a_{0}&#36;&#36;</p>
<h2>Linear Equation Group</h2>
<p>We have described in the column #Induction section of this paper that has emerged to help us solve problems associated with linear equation groups, starting with this chapter, where we will begin to study equation groups.</p>
<h3>The Cramer Law.</h3>
<p>This section examines the equations of equal numbers to the unknown, which is the form below.
&#36;&#36;\left{\begin{matrix} a_{11}x_1+a_{12}x_2+\cdots +a_{1n}x_n=b_1\ \cdots\cdots\cdots\cdots\cdots\cdots\cdots\a_{n1}x_1+a_{n2}x_2+\cdots +a_{nn}x_n=b_n\end{matrix}\right.&#36;&#36;
He's worth the same.&#36;\sum_{j=1}^{n} a_{ij}x_j=b_i&#36;</p>
<p>♪ When all&#36;b_i&#36;When equal to 0, we call the Zero-Line Equation Group, and the other term is the Non-Zone Sub-Line Equation Group. That's our core subject.</p>
<p>The Krammer law studies the situation, divided into three core propositions.</p>
<ul>
<li>Is there a problem?</li>
<li>Unsolved the Unique</li>
<li>What is it?</li>
</ul>
<p>There are coefficients in the column of our abstract equation group.<br>&#36;&#36;
A=
\begin{vmatrix}
  a_{11}&amp;\cdots   &amp;a_{1n} \
  \vdots &amp;\ddots  &amp;\vdots \
  a_{n1}&amp; \cdots &amp;a_{nn}
\end{vmatrix}&#36;&#36;</p>
<p>For a parallel sublinear equation group</p>
<ul>
<li>&#36;A\ne0&#36; The equation is the only solution -- zero, that's...&#36;x_i=0&#36;</li>
<li>&#36;A=0&#36; There's no zero-sum solution. There's no one.</li>
</ul>
<p>For non-linear sub-equipment groups</p>
<ul>
<li>&#36;A\ne0&#36; Only the equation will be solved.&#36;x=\frac{d_{i&#125;&#125;{d}&#36; of which &#36;d=A,d_i&#36;It's the first&#36;i&#36;Columns to Columns&#36;b&#36;Value of the lateral</li>
<li>&#36;A=0&#36;  It's not the only thing that can solve it.</li>
</ul>
<p> The nature of the equation is counter-problem: if the number of equations equals the number of unknown amounts and the equation is unsolved or unsolved, then&#36;A=0&#36;</p>
<h3>Goss is negative.</h3>
<p>The Kramers study the same number of equations and the same number of unknowns.&#36;m=n&#36; Study the problem of solving the problem of a broader array of equations.</p>
<p>For ease of reference, we use the matrix to represent the equation group from this section, and the detailed discussion of the matrix is left to the<a href="/en/blog/2023/06/17/advanced-algebra-matrices-linear-spaces-notes/">High algebra 2 Matrix and linear space</a></p>
<p>A generic linear equation group is expressed as the following form
&#36; \begin{matrix}
a {11}&amp;  a_{12}&amp;  \cdots &amp;  a_{1n}&amp;b_1 \
  \cdots&amp;  \cdots&amp;  \cdots&amp;  \cdots&amp; \cdots\
  a_{m1}&amp;  a_{m2}&amp;  \cdots &amp;  a_{mn}&amp;b_m
\end{pmatrix}&#36;&#36;</p>
<p>How do we solve an equation like this? Let's just say, study the equation.
&#36; \begin{matrix}
Two.&amp;-1  &amp;3  &amp;1 \
  4&amp;  2&amp;  5&amp;4 \
  2&amp;  1&amp;  3&amp;5
\end{pmatrix}&#36;&#36;
我们使用最后一行把第一列化为0
&#36;&#36;\begin{pmatrix}
  2&amp;-1  &amp;3  &amp;1 \
  0&amp;  4&amp;  -1&amp;2 \
  0&amp;  2&amp;  -1&amp;4
\end{pmatrix}&#36;&#36;
在将最后一行的第二列化为0（利用第二行）
&#36;&#36;\begin{pmatrix}
  2&amp;-1  &amp;3  &amp;1 \
  0&amp;  4&amp;  -1&amp;2 \
  0&amp;  0&amp;  1&amp;6
\end{pmatrix}&#36;&#36;
现在方程组被我们化为了一个很简单的形式，靠一些简单的计算就能解出
&#36;&#36;x_1=9,x_2=-1,x_3=-6&#36;&#36;</p>
<p>There are three ways we've changed.</p>
<ul>
<li>Change two row positions</li>
<li>The line is extended to the original&#36;k&#36;Double</li>
<li>Add one line to the other line
<strong>We can prove that the equations that this operation keeps are the same.</strong></li>
</ul>
<p>In fact, the Gossian divide is to remove the equation of the original matrix (the equation group) by maintaining the same-synthetic shift, and by changing it to the form of a stairwell equation that can be easily deciphered.</p>
<p>After the ladder.</p>
<ul>
<li>The constant is 0, which means the equation is unsolved.</li>
<li>If the number of equations is simplified to the same number as the number of unknown, there is only one group. Break</li>
<li>If the number of equations is less than the number of unknowns, it means numerous groups, because there is a free unknown, and a randomly selected set of free unknowns can be obtained.</li>
</ul>
<h3>Vector knowledge supplement</h3>
<p>To study the linear equations more, we need to supplement the knowledge associated with the research vector. The knowledge of vectors that need to be supplemented in the mathematics base is more limited, and the knowledge of the vectors is more limited.<a href="/en/blog/2024/12/12/analytic-geometry-notes/">Parsing Geometry</a>There's more to it.</p>
<h4>Vector and Space</h4>
<p>One.&#36;n&#36;The standard form of a one-time equation for a equation is as follows:
&#36;&#36;a_1x_1+\cdots+a_nx_n-b_1&#36;&#36;
He can use one.&#36;n+1&#36;Vectors are expressed in vectors
&#36;&#36;(a_1,a_2,\cdots,a_n,b_1)&#36;&#36;</p>
<p>Vectors are very diverse in terms of the abstract, so we add knowledge of vectors.</p>
<p>Definition: a digital domain&#36;p&#36;Top&#36;n&#36;V<strong>Ordered arrays</strong>The vectors are called a vector, with a clear map of the coordinates in the 12,3-dimensional vector, and the more high-dimensional vectors are less visible.</p>
<p>Definitions: Equal vectors mean equal weights</p>
<p>Definitions: The vector is added by the corresponding sum of the individual weights</p>
<p>Definitions: 0 points are referred to as zero vectors</p>
<p>Definitions: vector multipliers&#36;k&#36;It means multiplying the weights.&#36;k&#36;</p>
<p>Definition: Negative vector is a multiplier&#36;-1&#36;</p>
<p>Definition: Reduction of vectors by adding negative vectors</p>
<p>According to the above definition, you can give some of the algorithms in which&#36;\alpha,\beta&#36;It's a vector. &#36;k,l&#36;Yes. &#36;1,0&#36; It's all vectors.</p>
<ul>
<li>&#36;k(\alpha+\beta)=k\alpha+k\beta&#36;</li>
<li>&#36;(k+l)\alpha=k\alpha+l\alpha&#36;</li>
<li>&#36;{k}(l\alpha)=(kl)\alpha&#36;</li>
<li>&#36;1\alpha=\alpha&#36;</li>
<li>&#36;0\alpha=0&#36;</li>
<li>&#36;k0=0&#36;</li>
</ul>
<p>Definition ( vector space): All digital domains&#36;P&#36;Top&#36;n&#36;The entire shape of the vector is called the digital area&#36;P&#36;Top&#36;n&#36;Vector space, whether horizontal or vertical</p>
<h4>Linear combination is linear.</h4>
<p>Definitions (linear combinations): if&#36;\alpha=k_1\beta_{1}+k_{2}\beta_{2}+\cdots+k_{n}\beta_{n}&#36;</p>
<ul>
<li>Name&#36;\alpha&#36;Yes.&#36;\beta_i&#36;a linear combination, or&#36;\alpha&#36;It's possible.&#36;\beta_i&#36;Linear Table Out</li>
<li>If the vector group&#36;\alpha&#36;All vectors in it can be vectored Group&#36;\beta_i&#36;Table, which states that vectors can be shown</li>
<li>If two vectors can be shown to each other, they're called their equivalent.</li>
</ul>
<p>The vectors are transmitted, that's...&#36;\alpha&#36;By&#36;\beta&#36;The blogger says:&#36;\beta&#36;By&#36;v&#36;Table of rules&#36;\alpha&#36;By&#36;v&#36;Table</p>
<p>The vector group has the equivalent</p>
<ul>
<li>Self-reverse</li>
<li>Symmetry</li>
<li>Passivity
We call this relationship <strong>Equivalent relationship</strong> The value of the vector group is an equal value.</li>
</ul>
<p>Definition (linear related group): If a vector in a vector group can be shown by other linear tables of vectors within the vector group, the vector group is described as a linearly relevant vector group.</p>
<p>It's natural to give a theory about that.</p>
<ul>
<li>Vectors with zero vectors are linearly relevant vectors Group</li>
<li>If the vectors of both vectors are linear, then&#36;\alpha_1=k\alpha_2&#36;</li>
</ul>
<p>Definitions (linear related group equivalent): if not all 0&#36;k_i&#36;Make&#36;k_1\alpha_1+\cdots k_s\alpha_s=0&#36; The vector is called linear, it's linear or it's linear. Group</p>
<p>Inference:</p>
<ul>
<li>Linear unrelateds means only all.&#36;k&#36;Take zero to make the equation work.</li>
<li>Vector Group&#36;\alpha&#36;..and the vector is linear. Group</li>
<li>If the vector group&#36;\alpha&#36;It's not about the relationship, it's about his non-empty group.</li>
<li>&#36;n&#36;The dimensional vector group is linearly irrelevant</li>
</ul>
<p>Theorem: To determine that a vector group is linearly related, just list the equations&#36;x_1\alpha_1+x_2\alpha_2+x_3\alpha_3=0&#36; The non-zero-sum test of its existence, if it exists, is the linear group, which is the inference to the definition.</p>
<p>Inference: If the equation is only zero solver (linear unrelated group), no matter how many extra equations (relevance plus a few points), he is also linear unrelated (only zero solver)</p>
<p>Conclusions</p>
<ul>
<li>If Vector Group&#36;\alpha&#36;May be used as vector group&#36;\beta&#36;Tables and&#36;\alpha&#36;More in the medium vector than in the medium vector&#36;\beta&#36;The number of medium vectors has a vector group&#36;\alpha&#36;Linear Related</li>
<li>If&#36;\alpha&#36;by&#36;\beta&#36;The blogger says:&#36;\alpha&#36;It is not related; it is&#36;\alpha&#36;Other Organiser&#36;\beta&#36;Number of medium vectors</li>
<li>&#36;n+1&#36;individual&#36;n&#36;The wiring of the vector is relevant</li>
<li>Two linear groups of equal prices, with the same vectors</li>
</ul>
<h4>It's not a team.</h4>
<p>Definition (linear unrelated group and non-substantiate group): For a vector Group&#36;\alpha_{j}&#36; Take a few vectors from it and form a new vector group, which if this new vector class is the antenna, it's called a vector. Group&#36;\alpha_{j}&#36; Yes.<strong>Linearly unrelated group</strong>If you add any vector, this vector is associated with linear, which you call it vector. Group&#36;\alpha_{j}&#36; Yes.<strong>Large Linear Independent Group</strong> It's a big, unconnected group.</p>
<p>There are two ways to find a very unrelated group of vectors.</p>
<ul>
<li>Add each vector: First find two vectors and then add each vector to the vector, to determine whether it is also linearly unrelated, or if so, accept the vector, or if not discard it, then find the whole vector and then find the big difference. Group</li>
<li>Primary variant: The form of a line matrix is a primary transformation of the matrix (without line change) to eliminate the equivalent vector, and the last remaining line is a very irrelevant group for the amount to be given.</li>
</ul>
<p>Theoretically:<strong>The very unrelated group and the initial vector group are equivalent, and the nature of the original vector group can be studied by the very unrelated group.</strong></p>
<p>Definition: The number of vectors contained in the very unrelated group is fixed, and we call it the runk.</p>
<p>Nature:</p>
<ul>
<li>The minimum condition for linear unrelated groups is the same number of vectors as the thallium</li>
<li>The vector is the same as the equivalent.</li>
<li>There must be a very irrelevant vector group with a non-zero vector.</li>
<li>All of them are vector groups of zero vectors, not very unrelated groups, zero.</li>
</ul>
<h3>Decision on linear equation grouping</h3>
<p>After a long period of deference, we came back to study the linear equation, and although the Goss negatives part of this paper can already solve the equation, we still want to find common ways and means to simplify our calculations and our thinking.</p>
<p>Renumber linear equation group as vector form (column vector)
&#36;&#36;\alpha_1=(a_{11},a_{21},\cdots,a_{n1}),\alpha_2=(a_{12},a_{22},\cdots,a_{n2}),\beta=(b_{1},b_{2},\cdots,b_{n})&#36;&#36;</p>
<p>Original equation equals
&#36;&#36;x_1\alpha_1+\cdots+x_n\alpha_n=\beta&#36;&#36;</p>
<p>Theoretically:&#36;x_{1} \alpha_{1} + \cdots + x_{n} \alpha_{n} = \beta \quad \text{有解} \Leftrightarrow \beta \text{可被}\text{线性表出}&#36;</p>
<p>Equivalent Theorem: &#36;\begin{vmatrix}a {11} &amp; \cdots &amp; a_{1n} \ \vdots &amp; \ddots &amp; \vdots \ a_{s1} &amp; \cdots &amp; a \\end{vmatrix} \\text{and its magnification matrix} \\quad\vmatrix}a {11} &amp; \cdots &amp; a_{1n} &amp; b_{1} \ \vdots &amp; \ddots &amp; \vdots &amp; \vdots \ a_{s1} &amp; \cdots &amp; a_{sn} &amp; \d{vmatrix}\d\text{s the same</p>
<p>The extension of the Kramer Code: for an equation group, assuming the coefficient matrix is&#36;r&#36; It means it just needs one of them.&#36;r&#36;All equations are equal.&#36;r&#36;An equation.&#36;n&#36;Unknown</p>
<ul>
<li>If&#36;r=n&#36;The law of the Krammers is applicable to the law.&#36;A\ne0&#36;</li>
<li>If &#36;r&lt;n&#36; 则方程多解，自由未知量的数目为&#36;n-r&#36;</li>
<li>If &#36;r&gt;&#36; n.00 will solve the equation.</li>
</ul>
<h3>Structure of linear equation grouping</h3>
<p>This section begins with the structure of the equational equation, which is expected to be a small number of solutions to indicate the solution of the linear equation.</p>
<p>We give the basic nature of the solution of linear equations:<strong>The linear combination of solvers is the solvers.</strong></p>
<p>For the whole sublinear equation, we give the following description.</p>
<p>Definitions: grouping&#36;\eta_1,\eta_2,\cdots,\eta_r&#36; Called a base system if</p>
<ul>
<li>All the solutions are their linear combination.</li>
<li>They're not linear.</li>
</ul>
<p><strong>The process of proving the existence of the basic decomposition and finding the basic decomposition is the process of decomposing the equation group by matrix transformation, with free unknowns choosing to be a unit vector.</strong></p>
<p>Inference: Any vector that is linearly independent of itself and equal to the base decomposition system is the basic decomposition system, and all basic decomposition systems contain the same number.</p>
<p>For the non-kids linear equation group, we promote the previous narrative.</p>
<p>Definition: We call his export group the same linear equation group as the non-linear sub-equivalent matrix.</p>
<p>We give nature directly:<strong>The difference between the two groups of the non-linear equation group is the solution of the sub-mix group.</strong></p>
<p>Theorem: Set&#36;v&#36;It's a solution for the Fitch Equation Group.&#36;\eta&#36;It's the solution of the export team, then.&#36;v+\eta&#36;It's a solution to the Fixed Equation Group, when&#36;\eta&#36;The blogger says that the government is not going to be able to take the basics.&#36;v+\eta&#36;Overwrite all the solutions, which means that the solution of the non-sub-equacy group is in the following form:
&#36;&#36;v+k_1\eta_1+\cdots+k_{n-\gamma}\eta_{n-\gamma}&#36;&#36;</p>
<p><strong>We just need to study the solutions of the specials and the export groups, and we can give the solutions of the non-syncs.</strong></p>
<p>Inference:&#36;非齐次线性方程组只有一解\Longleftrightarrow 其导出组只有0解&#36;</p>
