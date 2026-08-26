---
title: 'Matrix Theory: Matrix Norms, Spectral Radius, and Hermitian Matrices'
title_zh: 矩阵论：矩阵范数、谱半径与Hermite 矩阵
date: 2024-10-17 15:24:42 +0800
categories:
- Mathematics
- Algebra & Matrix Theory
tags:
- Matrix Theory
- Linear Algebra
author: Hyacehila
mathjax: true
hidden: true
excerpt: Covers matrix norms, spectral radius, Hermitian matrices, matrix decompositions, matrix functions, and related matrix
  theory.
description: Covers matrix norms, spectral radius, Hermitian matrices, matrix decompositions, matrix functions, and related
  matrix theory.
excerpt_zh: 整理矩阵范数、谱半径、Hermite 矩阵、矩阵分解、矩阵函数和相关矩阵理论。
permalink: /blog/2024/10/17/matrix-theory-notes/
lang: en
translation_key: 2024-10-17-matrix-theory-notes
translation_status: machine
translation_source_hash: 02376afd57f8a5062c723d85ca628de3f322c04c405d9353067ac20c18e9bdb1
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<p>With the rapid development of science and technology, classical linear algebra knowledge no longer meets the needs of modern science and technology, and the theory and methodology of the matrix has become an essential tool in modern science and technology.</p>
<p>Fields such as numerical analysis, optimization theory, differential equations, probability statistics, control theory, mechanics, electronics, networking, etc. are closely linked to matrix theory, and even in the areas of economic management, finance, insurance, and social sciences, matrix theory and methodology have important applications.</p>
<p>We're here to discuss the core knowledge of the matrix, based on the basic matrix materials of graduate students, as far as those are concerned.<a href="/en/blog/2023/03/17/advanced-algebra-foundations-notes/">Advanced Algebra 1</a> <a href="/en/blog/2023/06/17/advanced-algebra-matrices-linear-spaces-notes/">High Algebra 2 Matrix and Linear Space</a>   <a href="/en/blog/2024/12/07/advanced-algebra-linear-transformations-euclidean-spaces-notes/">High algebra 3 linear transformation and European space</a>  The knowledge described therein is not repeated.</p>
<p><strong>This is still a basic narrative.</strong> It's a complement to advanced mathematics. Not all of the matrix analysis theory, but we have a lot of deep matrix analysis theory to learn from. Here, we focus only on the basic knowledge of matrix analysis, rather than on its different applications and emerging theories, and much more difficult and less useful matrix knowledge is left to matrix analysis studies.</p>
<h2>Matrix theory supplement</h2>
<h3>Matrix</h3>
<p>This is our first-class knowledge of the matrix that we didn't introduce in advanced generational mathematics.<a href="/en/blog/2023/09/11/functional-analysis-notes/">Analysis of general communications</a>The linkages are very close and are therefore chosen to be added to the matrix.</p>
<p>The matrix's prototype pre-knowledge is the vector's paradigm, and we're already in<a href="/en/blog/2023/09/11/functional-analysis-notes/">Linear enabling and built-in spaces in general communications analysis</a>It's been introduced.</p>
<h4>Definition and nature of matrix models</h4>
<p>Now we're going to extend the concept of the vector size to the matrix, and we're going to pull the matrix straight, and we're going to be able to either a long one-dimensional vector, and we're going to extend the concept of the vector range. And because the matrix is multiplication, not just multiplication and multiplication, there is a need for extra justice to limit him.</p>
<p>Definitions: Establishment&#36;A\in C^{m\times n}&#36;  By some law.&#36;A&#36;an actual function of the above, as&#36;||A||&#36;  If he meets the following four conditions,</p>
<ul>
<li>Non-negative: &#36;:\text{if 0}neq\mathbf,\text{require}\parallel A\parallel&gt;;\text{if A=mathbf},\text}\parallel A\parallel=&#36;0.</li>
<li>Recidivism:&#36;\text{对任意的 }k\in\mathbb{C},\parallel kA\parallel=\mid k\mid\parallel A\parallel&#36;</li>
<li>Triangular variant:&#36;\text{对任意 }A,B\in\mathbb{C}^{m\times n},\parallel A+B\parallel\leqslant\parallel A\parallel+\parallel B\parallel&#36;</li>
<li>Multiplication: when matrix product&#36;AB&#36;When it makes sense, yes.&#36;\parallel AB\parallel\leqslant\parallel A\parallel\parallel B\parallel&#36;
Name&#36;||A||&#36;  It's a matrix.&#36;A&#36;of the</li>
</ul>
<p>The multiplication guarantees a zero matrix.&#36;A^2=0&#36; Non-negative conflict, and the reasonable intensity of matrix numbers, are very reasonable requirements.</p>
<p>When we put one&#36;m\times n&#36;When the WV matrix looks like a straight-up vector, we can naturally give some matrix parameters, which are limited to square formation. Down</p>
<ul>
<li>&#36;\parallel A\parallel_{m_1}=\sum_{i=1}^n\sum_{j=1}^n\mid a_{ij}\mid&#36;</li>
<li>&#36;\parallel A\parallel_{m_{\infty&#125;&#125;=n\bullet\max_{i,j}\mid a_{ij}\mid&#36;</li>
<li>&#36;\parallel A\parallel_{m_2}=\left(\sum_{i=1}^n\sum_{j=1}^n\mid a_{ij}\mid^2\right)^{\frac{1}{2&#125;&#125;&#36;
<strong>Especially, the square 2-model is very common. We call it the Frobenius model.&#36;||A||_F&#36;</strong></li>
</ul>
<p>We can give unproven key theoretics of the matrix: like the vector model, all arrays are equal.</p>
<h4>Calculator</h4>
<p>We're here for the concept of "The Sonoko".<a href="/en/blog/2023/09/11/functional-analysis-notes/">Linear algorithms in general correspondence analysis</a>He was described as a spatial map, and in the matrix we continue to study algorithms, but only the matrix is enough.</p>
<p>Set&#36;A\in C^{m\times n}&#36;  ，&#36;x\in C^n&#36;  There is.&#36;Ax\in C^m&#36;  It's really a count.&#36;A&#36;It's mapped in two different dimensions of vector space if we put&#36;x&#36;It's a matrix.
&#36;&#36;\parallel Ax\parallel\leqslant\parallel A\parallel\parallel x\parallel&#36;&#36;
Which means...
&#36;&#36;\parallel A\parallel\geqslant\frac{\parallel Ax\parallel}{\parallel x\parallel}&#36;&#36;
The right of the difference is the ratio of the vector model, and the left is the matrix model that has not yet been defined, so the matrix model is defined from the vector model.
&#36;&#36;\parallel A\parallel=\sup_{\parallel x\parallel\neq0}\frac{\parallel Ax\parallel}{\parallel x\parallel}&#36;&#36;
When?&#36;||x||=1&#36;, because the unit surface must be closed, the right side is a continuous function.</p>
<p>If we define both the matrix and the vector model, we should define them in such a way as to ensure that the above-mentioned variations are established, which we call the compatibility of the two paradigms.</p>
<p>We'll have a matrix if we already have a vector.
&#36;&#36;\parallel A\parallel=\sup_{\parallel x\parallel\neq0}\frac{\parallel Ax\parallel}{\parallel x\parallel}=\max_{\parallel x\parallel=1}\parallel Ax\parallel&#36;&#36;
Called a matrix model derived from a vector model, or a algorithm submodel</p>
<p>Theorem: Set&#36;A\in C^{m\times n}&#36; &#36;x\in C^n&#36; Let's take the vector.&#36;x&#36;The 1st and the 2nd and the Infinite Numerical Paradigms are available separately.</p>
<ul>
<li>&#36;\parallel A\parallel_1=\max_{i=1}^m\mid a_{ij}\mid\text{(称为列范数)}&#36;</li>
<li>&#36;\parallel A\parallel_2=\sqrt{\lambda_{\max}(A^\mathrm{H}A)}(\text{称为谱范数)}&#36; Of which count&#36;\lambda_{max}&#36;is the maximum feature value of the matrix</li>
<li>&#36;\parallel A\parallel_\infty=\max_i\sum_{i=1}^n\mid a_{ij}\mid\text{(称为行范数)}&#36;</li>
</ul>
<h4>Spectrum and spectral radius</h4>
<p>We know. The matrix's algorithm.&#36;|A|_2&#36;Called&#36;A&#36; The spectra, it's the value through the matrix. &#36;A^\mathrm{H}A&#36; The maximal feature values are calculated, although they are difficult to use, but they are of a very good nature, so they are often used in matrix analysis and system theory.</p>
<p>Theorem: Set&#36;A\in C^{m\times n}&#36;  then</p>
<ul>
<li>&#36;\parallel A\parallel_2=\max_{\parallel x\parallel_2=\parallel y\parallel_2=1}\mid y^\mathrm{H}Ax\mid,x\in\mathbb{C}^n,y\in\mathbb{C}^m&#36;</li>
<li>&#36;\parallel A^\mathrm{H}\parallel_2=\parallel A\parallel_2&#36;</li>
<li>&#36;\parallel A^\mathrm{H}A\parallel_2=\parallel A\parallel_2^2&#36;</li>
</ul>
<p>Theorem: Set&#36;A\in C^{m\times n},U\in C^{m\times m},V\in C^{n\times n},U^{H}U=I_{m} V^HV=I_n&#36; then
&#36;&#36;\parallel UAV\parallel_2=\parallel A\parallel_2&#36;&#36;</p>
<p>Theoretically: There is a model for any algorithm.&#36;A\in C^{n\times n}&#36; If it's true&lt;1&#36; 则 &#36;I-A&#36;为非奇异矩阵，并且&#36;\parallel(I-A)^{-1}\parallel\leqslant(1-\parallel A\parallel)^{-1}&#36;</p>
<p>Definitions: Establishment&#36;A\in C^{n\times n}&#36; If&#36;\lambda_1,\lambda_2,...,\lambda_n&#36;It's his signature value, we call it.
&#36;&#36;\rho(A)=\max_i|\lambda_i|&#36;&#36;
It's a matrix.&#36;A&#36;spectral radius</p>
<p>Theorem (characteristically defined):&#36;A\in C^{n\times n}&#36; There's always. &#36;\rho(A)\leqslant\parallel A\parallel&#36; That is, the spectrum radius is smaller than any of the prototypes.</p>
<p>Theorem: Set&#36;A\in C^{n\times n}&#36; And...&#36;A&#36;It's a positive matrix, and there is.&#36;\rho(A)=\parallel A\parallel_2&#36;</p>
<p>Theoretically: for any non-generic matrix&#36;A\in C^{n\times n}&#36; And...&#36;A&#36;It's a positive matrix. &#36;A&#36;The spectra is
&#36;&#36;\rho(A)=\parallel A\parallel_2=\sqrt{\rho(A^\mathrm{H}A)}=\sqrt{\rho(AA^\mathrm{H})}&#36;&#36;</p>
<h3>Elmitt Matrix and Transformation</h3>
<p>Yes.<a href="/en/blog/2024/12/07/advanced-algebra-linear-transformations-euclidean-spaces-notes/">High algebra 3 linear transformation and European space in European space</a>In the middle, we're introducing a special kind of transformation that keeps the European space structure unchanged, that is,<a href="/en/blog/2024/12/07/advanced-algebra-linear-transformations-euclidean-spaces-notes/">High algebra 3 Linear transformation with positive transformation in European space</a>I don't know. In fact, there's another kind of interesting change and his matrix, which we didn't introduce, is Elmitt.</p>
<p>We'll start with symmetrical shifts, then discuss Elmitt shifts, and the matrix, and finally we'll talk.<a href="/en/blog/2023/06/17/advanced-algebra-matrices-linear-spaces-notes/">Upper Algebra 2 Seconds in Matrix and Linear Space</a>The expansion, Elmitt's positive matrix.</p>
<h4>Symmetrical Change and Symmetry Matrix</h4>
<p>Definitions: Establishment&#36;A&#36;It's European space.&#36;V&#36;A linear shift on it, yes.&#36;V&#36;Any element in&#36;x,y&#36; Both.
&#36;&#36;(A(x),y)=(x,A(y))&#36;&#36;
Name&#36;A&#36;It's European space.&#36;V&#36;A previous symmetrical transformation</p>
<p>Inference: By definition, we can easily prove that symmetrical shifts&#36;A&#36;It's a symmetrical matrix under the standard logarithmic.&#36;A^T=A&#36;  It is equally true.</p>
<p>Depending on the nature of the symmetric matrix, we can give you two operational properties.</p>
<ul>
<li>If&#36;A&#36;Symmetrical matrix, then.&#36;(A(x),y)=(x,A(y))&#36;</li>
<li>If&#36;A&#36;Not a symmetric matrix, then.&#36;(A(x),y)=(x,A^T(y))&#36;</li>
</ul>
<h4>Elmitt Changed and Elmitt Matrix</h4>
<p>Definitions: Establishment&#36;A&#36;It's the space.<a href="/en/blog/2024/12/07/advanced-algebra-linear-transformations-euclidean-spaces-notes/">High algebra 3 linear transformation and European-style space</a>）&#36;V&#36;A linear shift on it, yes.&#36;V&#36;Any element in&#36;x,y&#36; Both.
&#36;&#36;(A(x),y)=(x,A(y))&#36;&#36;
Name&#36;A&#36;It's space.&#36;V&#36;One of the last Elmitts changed.</p>
<p>We'll take the matrix.&#36;A&#36;And then the whole matrix is called&#36;A^H&#36;Which means...&#36;\bar{A^{T&#125;&#125;=A^H&#36;It's to simplify the symbols that will appear later.</p>
<p>Inference: If Elmitt changes&#36;A&#36;The matrix under the standard logarithmic is:&#36;A&#36;Then there is.&#36;A^H=A&#36; The matrix that we will meet this condition is called the Elmet Matrix. If satisfied&#36;A^H=-A&#36; It's called the Anti-Elmet Matrix.</p>
<p>Theorem (Schur Theorem): any&#36;n&#36;The stylactic matrix is similar to a previous triangulation, i.e. for whatever.&#36;n&#36;Step Matrix&#36;A&#36;♪ There's one ♪&#36;n&#36;Classical Matrix&#36;U&#36;And an upper triangle matrix.&#36;T&#36; Satisfied
&#36;&#36;U^HAU=T&#36;&#36;
&#36;T&#36;It's a diagonal element.&#36;A&#36;, the order is determined according to the circumstances</p>
<p>Inference: If&#36;A&#36;For the Elmitt Matrix, then&#36;A&#36;Must be similar to the diagonal.&#36;A&#36;)</p>
<h4>Elmitt's set. Half-correct.</h4>
<p>Definitions: Establishment&#36;A&#36;Yes&#36;n&#36;The Elmitt Matrix, if any&#36;n&#36;Widget&#36;x&#36;Both.
&#36;&#36;x^\text{н}Ax\geqslant0,&#36;&#36;
Name&#36;A&#36;Non-negative (semi-positive) matrix for Elmitt, recorded as&#36;A\geqslant0.&#36;If you're right about anything,&#36;n&#36;D-fi Zero Vector&#36;x&#36;Both.
&#36;&#36;US&#36;US&#36;&#36;&#36;&#36;US&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;...&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;...&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;...&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;...&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;...&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;...&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;,&#36;&#36;&#36;&#36;&#36;&#36;...&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;,&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&gt;0,&#36;
Name&#36;A&#36;Regularize the Elmitt matrix as &#36;A&gt;0.&#36;</p>
<p>We're repeating the concept because the symmetrical matrix in European space looks at the problem of the secondary type.<a href="/en/blog/2023/06/17/advanced-algebra-matrices-linear-spaces-notes/">Upper Algebra 2 Seconds in Matrix and Linear Space</a> And the Elmitt Matrix is the symmetrical matrix in the space of the magma, so we're looking at the Elmitt secondary.</p>
<p>According to this definition, it's easy to say</p>
<ul>
<li>Unit Matrix &#36;I&gt;0&#36;</li>
<li>&#36;A&gt;0,k&gt;0&#36; 则 &#36;kA&gt;0&#36;</li>
<li>&#36;A\ge0,B\ge0&#36; then &#36;A+B\ge 0&#36;</li>
</ul>
<p>Theorem: Matrix&#36;A&#36;A sufficient requirement for a positive matrix is&#36;A&#36;All feature values are positive.</p>
<p>Theorem: Matrix&#36;A&#36;Sufficient requirement for a positive matrix to exist&#36;n&#36;Classical Matrix&#36;P&#36;♪ That makes ♪&#36;A=P^HP&#36; ♪ Half-positive removes non-chilling ♪</p>
<p>Theoretically: Normal Matrix&#36;A&#36;The order of the steps is a positive matrix.</p>
<p>Theorem: Matrix&#36;A&#36;A sufficient requirement for a positive matrix is&#36;A&#36;The sequences are positive or all of them are larger than zero.</p>
<p>Theorem: Set&#36;A,B&#36;Yes.&#36;n&#36;Elmitt Matrix, and &#36;B&gt;0&#36; 则存在非奇异矩阵&#36;Q&#36; made
&#36;&#36;Q^\mathrm{H}BQ=I,\quad Q^\mathrm{H}AQ=\mathrm{diag}(\lambda_1,\lambda_2,\cdotp\cdotp\cdotp,\lambda_n).&#36;&#36;</p>
<h4>Characteristic values of the Elmet Matrix</h4>
<p>The values of the Elmitt matrix are of some very interesting nature, and here we have some brief descriptions.</p>
<p>The matrix used in this section is greater and smaller than, for example, &#36;A&gt;B, A\B, all mean that the difference is positive or semi-positive.</p>
<p>Theorem: Set&#36;A&#36;Yes.&#36;n&#36;Elmitt Matrix, then
&#36;&#36;\lambda_{\min}(A)\boldsymbol{I}\leqslant A\leqslant\lambda_{\max}(A)\boldsymbol{I}&#36;&#36;
of which&#36;\lambda_{\min}(A),\lambda_{\max}(A)&#36; is the minimum and maximum feature value of the matrix</p>
<p>Definitions: Establishment&#36;A&#36;Yes&#36;n&#36;Elmitt Matrix, yes.&#36;\forall x\in\mathbb{C}^n&#36;and&#36;x\neq\mathbf{0}&#36;,&#36;&#36;R(x)=\frac{x^\mathrm{H}Ax}{x^\mathrm{H}x},\quad x\neq0&#36;&#36;To the Elmitt Matrix. &#36;A&#36; Reilly.</p>
<p>Theorem (Basic Nature of the Raleigh Business): Establishment&#36;A&#36;Yes.&#36;n&#36;Cascade Elmitt Matrix, the characteristic value is&#36;\lambda_1\geqslant\lambda_2\geqslant\cdotp\cdotp\cdotp\geqslant\lambda_n&#36;, then</p>
<ul>
<li>&#36;R(k\boldsymbol{x})=R(\boldsymbol{x}),k\in\mathbb{C},k\neq0;&#36;</li>
<li>&#36;\lambda _n\leqslant R( x) \leqslant \lambda _1&#36; &#36;, x\neq 0&#36; ;</li>
<li>&#36;\lambda_{1}=\operatorname*{max}<em>{x\neq0}R\left(x\right),\lambda</em>{n}=\operatorname*{min}_{x\neq0}R\left(x\right).&#36;</li>
</ul>
<p>Theoretically (extremely small): Set&#36;A&#36;Yes.&#36;n&#36;Cascade Elmitt Matrix, the characteristic value is&#36;\lambda_1\geqslant\lambda_2\geqslant\cdotp\cdotp\cdotp\geqslant\lambda_n&#36;, &#36;V_i&#36;Yes.&#36;C^n&#36;Yes.&#36;i&#36;Viko, then
&#36;&#36;\lambda_i=\max_{V_i}\underset{x\neq0,x\in V_i}{\operatorname*{\operatorname*{min&#125;&#125;}R(x),\lambda_i=\min_{V_{n-i+1&#125;&#125;\max_{\begin{array}{c}x\in V_{n-i+1},x\neq0\end{array&#125;&#125;R(x)&#36;&#36;</p>
<p>Using a very small theorem, we can study the range of variations in the relative matrix feature values when the elements of the Elmitt matrix change slightly.</p>
<p>Theorem: Set&#36;A,E&#36;Both.&#36;n&#36;Elmet Matrix&#36;,B=A+E&#36;and&#36;A,B&#36;and&#36;E&#36;Characteristic values for each
&#36;\lambda_1\geqslant\cdots\geqslant\lambda_n,\mu_1\geqslant\cdots\geqslant\mu_n&#36; and&#36;\varepsilon_1\geqslant\cdots\geqslant\varepsilon_n&#36;, then
&#36;&#36;\lambda_i+\varepsilon_n\leqslant\mu_i\leqslant\lambda_i+\varepsilon_1,\quad i=1,2,\cdotp\cdotp\cdotp,n.&#36;&#36;</p>
<h3>Camera analysis</h3>
<p>In numerical calculations, two types of error affect the accuracy of the calculation, i.e. cut-off errors caused by the method of calculation and leave-over errors caused by the environment. In order to analyse the impact of these errors on the resolution of mathematical problems, they are attributed to the effect of the disturbance (or ingestion) of the raw data ... We will examine separately how much the resolution of problems caused by the motion of the raw data, i.e. the stability of the resolution of the problem, has changed during the search for online equations and matrix characterizations.</p>
<h4>Pathological equation group and pathological matrix</h4>
<p>Consider this simple binary equation. Group
&#36; \begin{bmatrix}1&amp;0.99\0.99&amp;It's not like I'm gonna be able to do this.
Exactly decoded as &#36;x_1=100,x_2=-100&#36;</p>
<p>Micro-calibration of the equation group, which is very common due to factors such as error in real experiments.
&#36; \begin{bmatrix}1&amp;0.99\0.99&amp;0.99\end{bmatrix}\begin{pmatrix}x_1+\delta x_1\\x_2+\delta x_2\end{pmatrix}=\begin{bmatrix}1\1.001\end{bmatrix}&#36;&#36;</p>
<p>Then the precise solver of the equation group becomes&#36;x_1+\delta x_1=-0.1,x_2+\delta x_2=\frac{10}{9}&#36;</p>
<p>It can be seen that, although we have a very small reaction to the primary coefficient, the equation group has changed dramatically, and this is a pathological phenomenon.</p>
<p>Definitions: If coefficient matrix&#36;A&#36;Or constant items&#36;b&#36;Small changes that cause equation groups&#36;Ax=b&#36;The large variation by which the equation group is called a pathological equation group with a coefficient matrix&#36;A&#36;It's called a pathological matrix that corresponds to (or reverses) the equation group; on the other hand, the equation group is called the benign equation group.&#36;A&#36;Called a good matrix.</p>
<p>It should be noted that, when it comes to the concept of a “pathological matrix”, it is important to make clear what it is for ... because it is a pathological matrix for the equation group (or reverse), and not necessarily a pathological matrix for characterization values, and vice versa ... So we cannot say in general terms that a matrix is “pathological”.</p>
<h4>Number of conditions in the matrix</h4>
<p>Having learned the concept of pathology, we began to study the criteria for measuring a matrix pathology. As to why the criteria were so closed.</p>
<p>Definitions: Set A as a non-chilling matrix, termed lid&#36; (\bardsymbol{A}) =\left|boldsymbol{A}1}right|<em>{\rho}\left|\boldsymbol{A}\right|</em>{\rho}(p=1,2&#36; 或&#36;\infty)&#36;为矩阵 &#36;A&#36; condition.</p>
<p>This shows that the matrix's conditions are related to the model, and it depicts a possible magnification of the relative error of the equation grouping, which is generally considered to be far more than one condition of a bad nature and less than one condition. But more precise criteria are lacking.</p>
<p>The most common number of conditions is spectrometer.
&#36; \begin{aligned}\oporatorname{cond}<em>2&amp;=\parallel A\parallel_2\parallel A^{-1}\parallel_2\&amp;=\sqrt{\frac{\lambda</em>I don't know.
It's a promotion of the criteria in the linear regression base.</p>
<h4>Aggression analysis of matrix feature values</h4>
<p>We're looking at the solution of linear equations, and we're looking at the kinetic analysis of matrix feature values. The precise solvency of the high-level matrix feature values is difficult to solve, so we look at the approximation.</p>
<p>Definitions: Establishment&#36;A=(a_{ij}&#36; ) is either&#36;n&#36;. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .&#36;n&#36;A disk
&#36;&#36;G_i( \boldsymbol{A}) : \mid z- a_{ii}\mid \leqslant R_i&#36;, &#36;i= 1, 2, \cdots , n&#36;&#36;
Here.&#36;R_{i}=\sum_{j=1}\mid a_{ij}\mid&#36;is the circle of the radius (i.e. the boundary of the disc) called the matrix &#36;A&#36; Gerschgorin Circle, short Gael Circle.</p>
<p>Theoretically (Geal's theorem also known as the disc's theorem): Set &#36;A=(a_{\mathrm{i}j})\in\mathbb{C}^{n\times n}&#36;, then</p>
<ul>
<li>&#36;A&#36; The signature values are in place. &#36;n&#36; A disk&#36;G_i(\mathbf{A})&#36;In other words, it's not like you're in the same group.&#36;\mathbf{A}&#36; And each of these features is down.&#36;A&#36; within a disk)
&#36;&#36;\lambda(A)\subseteq\bigcup_{i=1}^nG_i(A):;&#36;&#36;</li>
<li>Matrix&#36;A&#36;One of them.&#36;m&#36;There are and there are only a few links to a disk&#36;A&#36;Yes.&#36;m&#36;Characteristic value (when)&#36;A&#36;When the same element is present in the main diagonal line, it is calculated by the number of repetitions and, when the same feature values are present, by the number of repetitions).</li>
</ul>
<p>The theorem of the circle helps us estimate the approximate range of the feature values, and the point on the plane is a complex number, possibly a feature of the matrix. Value</p>
<p>Definitions: Establishment &#36;A\in\mathbb{C}^{n\times n}&#36;and has reversible matrix &#36;P&#36; Make &#36;P^{-1}AP=\operatorname{diag}(\lambda_1,\ldots,\lambda_n)&#36;, or&#36;\parallel\boldsymbol{P}^{-1}\parallel\parallel\boldsymbol{P}\parallel&#36;As Matrix&#36;A&#36;“Conditions” for feature values, abbreviations for feature conditions, as follows:&#36;\zeta(\boldsymbol{P}).&#36; If &#36;\zeta(\boldsymbol{P})=\parallel\boldsymbol{P}^{-1}\parallel\parallel\boldsymbol{P}\parallel&#36;Not so big, then. &#36;\boldsymbol{A}&#36; It's a good question of character values.</p>
<p>For the Elmitt Matrix, we have...
Theorem: Set&#36;A,E&#36;Both.&#36;n&#36;Elmet Matrix&#36;,B=A+E&#36;and&#36;A,B&#36;and&#36;E&#36;Characteristic values for each
&#36;\lambda_1\geq\lambda_2\geq\cdots\geq\lambda_n,\mu_1\geqslant\mu_2\geqslant\cdots\geq\mu_n&#36; and&#36;\varepsilon_1\geq\varepsilon_2\geq\cdots\geqslant\varepsilon_n&#36;, then
&#36;&#36;\mid\lambda_i-\mu_i\mid\leqslant\parallel E\parallel_2,\quad i=1,2,\cdotp\cdotp\cdotp,n.&#36;&#36;</p>
<p>The number of characteristic conditions for the Elmitt Matrix can be seen to be 1, which means that the Elmitt Matrix is healthy with regard to the characteristic values.</p>
<h2>Matrix Decomposition</h2>
<p>Matrix decomposition plays a key role in the development of matrix theory and modern computing mathematics. So-called matrix decomposition is the multiplication of a matrix that is more structured or more familiar in nature.</p>
<p>We've studied diagonalization in high algebras, Joran's standard model, etc., and they're all part of the matrix decompose. However, earlier theories were less focused on their application and it was difficult to achieve simplified calculations and deeper theories.</p>
<p>So this chapter provides an aggregate study of common matrix decomposition, either based on previously learned theories or entirely alien, but they all play an important role in modern computing mathematics.</p>
<h3>Triangulation</h3>
<h4>Gaussian</h4>
<p>We're here.<a href="/en/blog/2023/03/17/advanced-algebra-foundations-notes/">High algebra 1 Goss stag in the foundation of the algebra.</a>The matrix balances that were introduced to the solution equation group are, in fact, the transformation of the initial matrix into an up-triangular matrix, and the use of line variations only, so that understanding the essence of the Gossian divide gives us a complete understanding of the triangle breakdown.</p>
<p>For one.&#36;n&#36;Linear equation group of elements, which we represent directly in matrix form
&#36;&#36;Ax=b&#36;&#36;
The Goss negative is using the matrix's primary transformation.&#36;A&#36;Turning to the triangle matrix, we assume that the whole course is not in the row order (which is a very normal option), each line shift is left by a corresponding primary matrix. Since we only consider square formations in the area of matrix decomposition, the final result of the Gossian collapse is:
&#36;L^(n-1)}\cdotsL^(2)}L^A^1}=begin{bmatrix}a 11^1}&amp;a_{12}^{(1)}&amp;\cdots&amp;a_{1n}^{(1)}\&amp;a_{22}^{(2)}&amp;\cdots&amp;a_{2n}^{(2)}\&amp;&amp;\ddots&amp;\vdots\&amp;&amp;&amp;a_{nn}^{(n)}\end{bmatrix}=A^{(n)}&#36;&#36;</p>
<p>In fact, if you want to go into this form, we need to make sure that the diagonal element is not zero, so we can give the following theory.</p>
<p>Theoretically:&#36;n&#36;Step Matrix&#36;A&#36;Before&#36;n-1&#36;The diagonal element is not zero, and the Goss passable can go through.</p>
<h4>Triangular breakdown of matrices</h4>
<p>When the breakdown described in the current section is normal, remember&#36;U=A^{n}&#36; then
&#36;&#36;L^{(n-1)}\cdot\cdot\cdot L^{(2)}L^{(1)}A=U&#36;&#36;
Which means...
&#36;&#36;A=(L^{(1)})^{-1}(L^{(2)})^{-1}\cdots(L^{(n-1)})^{-1}U&#36;&#36;</p>
<p>We know according to the definition of the reverse matrix. &#36;(L^{(1)})^{-1}&#36; It's the lower trigonometric matrix, so its product is the lower trigonometric matrix, which means
&#36;L=(L^)^(L^(2)}^cdotp\cdotp^cdotp}^=begin{bmatrix}1&amp;&amp;&amp;&amp;&amp;&amp;\l_{21}&amp;1&amp;&amp;&amp;&amp;&amp;\l_{31}&amp;l_{32}&amp;1&amp;&amp;&amp;&amp;\l_{41}&amp;l_{42}&amp;l_{43}&amp;\ddots&amp;&amp;&amp;\\vdots&amp;\vdots&amp;\vdots&amp;\ddots&amp;1&amp;&amp;\l_{ln}&amp;l_{n2}&amp;l_{n3}&amp;\cdots&amp;l_{n,n-1}&amp;What's wrong with you?
It's the lower triangle matrix of all the angle elements.</p>
<p>In conclusion, we can start with an initial matrix.&#36;A&#36;Split
&#36;&#36;A=LU&#36;&#36;
The initial matrix is broken down into a product of an upper matrix and a lower matrix</p>
<p>Definition: If FLA&#36;A&#36;Can be broken down into a lower triangle matrix&#36;L&#36;And an upper triangle matrix.&#36;U&#36;, or&#36;A&#36;Could be triangulated or&#36;LU&#36;Decomposition. If&#36;L&#36;It's the sub-unit triangle.&#36;U&#36;For the Triangular Matrix, triangulation at this time is called Doolittle decomposition; if&#36;L&#36; It's the lower triangle matrix, and...&#36;U&#36; It is a unit-on-triangular matrix, which is referred to as the Crout decomposition.</p>
<p>As can be seen by definition, the matrix's triangulation must not be the only one, but at least Doolittle and Crout.
&#36;&#36;A=LU=LDD^{-1}U=(LD)(D^{-1}U)=\widetilde{L}\widetilde{U}&#36;&#36;
So we can find countless decompositions in one decomposition.&#36;D&#36;All you have to do is form a non-0-lined arbitrary diagonal matrix. To that end, we want to find a triangulation that is unique.</p>
<p>Theorem (LDU Basic Theorem)&#36;A&#36; Yes&#36;n&#36; Array, then &#36;A&#36; The only way to break it down is to...
&#36;&#36;A=LDU&#36;&#36;
The only necessary condition&#36;A&#36;Before&#36;n-1&#36;Order Master&#36;\Delta_k\neq0(k=1,2,\cdotp\cdotp\cdotp,n-1).&#36;of which&#36;L,U&#36;It's an under-unit, up-triangular matrix.&#36;D&#36;It's an angle matrix.
&#36;&#36;D=\operatorname{diag}(d_1,d_2,\cdotp\cdotp\cdotp,d_n),&#36;&#36;
&#36;&#36;d_k=\frac{\Delta_k}{\Delta_{k-1&#125;&#125;,\quad k=1,2,\cdots,n,\quad\Delta_0=1.&#36;&#36;</p>
<p>With the LDU theorem, we find the existence of triangulation, which can easily give the existence of the existence of Doolittle and Crout.</p>
<p>Inferences:&#36;A&#36; Yes.&#36;n&#36; Array, then &#36;A&#36; The only necessary condition for Doolittle to break up is before A.&#36;n-1&#36;Order Master
&#36; \Delta k}begin{vmatrix}a 11}&amp;\cdots&amp;a_{1k}\\vdots&amp;&amp;\vdots\a_{k1}&amp;\cdots&amp;a \neq0, \quadk=1,2, \cdots, n-1, &#36;1
of which&#36;L&#36; It is a triangulation of units.&#36;\tilde{U}&#36; It's the up-triangular matrix.</p>
<p>&#36;&#36;\mathbf{A}=\begin{bmatrix}1&amp;&amp;&amp;&amp;&amp;\l_{21}&amp;1&amp;&amp;&amp;&amp;\l_{31}&amp;l_{32}&amp;\ddots&amp;&amp;&amp;\\vdots&amp;\vdots&amp;\ddots&amp;1&amp;&amp;\l_{n1}&amp;l_{n2}&amp;\cdots&amp;l_{n,n-1}&amp;1\end{bmatrix}\begin{pmatrix}u_{11}&amp;u_{12}&amp;\cdots&amp;u_{1n}\&amp;u_{22}&amp;\cdots&amp;u_{2n}\&amp;&amp;\ddots&amp;\vdots\&amp;&amp;&amp;u_{nn}\end{pmatrix},&#36;&#36;
并且若&#36;A&#36;为奇异矩阵，则&#36;u_{nn}=0;&#36;若&#36;A&#36;为非奇异矩阵，则充要条件可换为：&#36;A&#36;的各阶顺序主子式全不为零，即：
&#36;&#36;\Delta_k\neq0,\quad k=1,2,\cdotp\cdotp\cdotp,n.&#36;&#36;</p>
<p>Inference 2 &#36;n&#36;Array&#36;A&#36;But the only way to decompose is to decomposition Cloot.
&#36;A=\tilde(bardsymbol{L}\boldsymbol{U=begin{bmatrix}l 11}&amp;&amp;&amp;\l_{21}&amp;l_{22}&amp;&amp;\\vdots&amp;\vdots&amp;\ddots&amp;\l_{n1}&amp;l_{n2}&amp;\cdots&amp;l_{nn}\end{bmatrix}\begin{pmatrix}1&amp;u_{12}&amp;\cdots&amp;u_{1n}\&amp;1&amp;\cdots&amp;u_{2n}\&amp;&amp;\ddots&amp;\vdots\&amp;&amp;&amp;1\end{pmatrix}&#36;&#36;</p>
<p>The essentials remain pre-inferences. Medium&#36;n-1&#36;If...&#36;A&#36;For a strange matrix, then&#36;l_{nm}=0;&#36;If&#36;A&#36;For non-generic matrices, the conditions can also be replaced by the situation at the same level.</p>
<h4>Common Triangulations</h4>
<p>In practical application if matrix&#36;A&#36;Number of steps&#36;n&#36;It's very high, so we'll take the cut.&#36;A&#36;We're going to introduce two commonly used direct triangulation formulas based on asymmetric and symmetrical A.</p>
<h5>Crout decomposition.</h5>
<p>Set&#36;A&#36;Yes&#36;n&#36;Array (but not necessarily symmetry) and decomposition
&#36;&#36;A=LU,&#36;&#36;
That's...
&#36; \begin{matrix}a 11}&amp;\cdots&amp;a_{1j}&amp;\cdots&amp;a_{1n}\\vdots&amp;&amp;\vdots&amp;&amp;\vdots\a_{i1}&amp;\cdots&amp;a_{ij}&amp;\cdots&amp;a_{in}\\vdots&amp;&amp;\vdots&amp;&amp;\vdots\a_{n1}&amp;\cdots&amp;a_{nj}&amp;\cdots&amp;a_{nn}\end{pmatrix}=\begin{pmatrix}l_{11}&amp;&amp;&amp;&amp;&amp;\\vdots&amp;\ddots&amp;&amp;&amp;&amp;\l_{i1}&amp;\cdots&amp;l_{ii}&amp;&amp;&amp;\\vdots&amp;&amp;&amp;\ddots&amp;&amp;\l_{n1}&amp;\cdots&amp;\cdots&amp;\cdots&amp;l_{nn}\end{pmatrix}\begin{pmatrix}1&amp;u_{12}&amp;\cdots&amp;u_{1j}&amp;\cdots&amp;u_{1n}\&amp;\ddots&amp;&amp;\vdots&amp;&amp;\vdots\&amp;&amp;1&amp;u_{j-1,j}&amp;\cdots&amp;u_{j-1,n}\&amp;&amp;&amp;\ddots&amp;&amp;\vdots\&amp;&amp;&amp;&amp;1&amp;1\end{pmatrix}&#36;&#36;</p>
<p>Here's how the matrix elements are calculated.</p>
<p>When?&#36;i\ge j&#36;  time (means calculating triangle position)
&#36;&#36;l_{ij}=a_{ij}-\sum_{k=1}^{j-1}l_{ik}u_{kj},\quad i=1,\cdots,n,\quad j=1,\cdots,i;&#36;&#36;
When &#36;i&lt; j&#36; (means counting triangle positions)
&#36;&#36;u_{ij}=\left(a_{ij}-\sum_{k=1}^{i}l_{ik}u_{kj}\right)/l_{ii},\quad i=1,\cdots,n-1,\quad j=i+1,\cdots,n.&#36;&#36;
We need an iterative solution using these two formulas.</p>
<h5>Doolittle decompose.</h5>
<p>Similarly, we can give Doolittle the solution below. Pattern
That's right.&amp;i=1,\cdots,n,&amp;j=i,\cdots,n,\\l_{ij}=\left(a_{ij}-\sum_{k=1}^{j-1}l_{ik}u_{kj}\right)/u_{jj},&amp;i=2,\cdots,n,&amp;j=1,\cdots,i-1.&amp;\end{cases}&#36;&#36;</p>
<h5>Cholesky decomposes.</h5>
<p>If&#36;A&#36;The symmetrically positive matrix would significantly reduce the calculated amount of triangulation, which is about half of the workload of the aforementioned Claute or Doolittle.</p>
<p>Theorem: Set&#36;A&#36;Yes&#36;n&#36;There is a real, non-eccentric matrix in the symmetrical positive matrix.&#36;L&#36;♪ That makes
&#36;&#36;A=LL^{\mathrm{T&#125;&#125;.&#36;&#36;
If the limit&#36;L&#36; The diagonal elements are positive, and this breakdown is the only one.</p>
<p>We can easily give the solver formula, because of its symmetry, and we'll only have to count in half.
&#36;&#36;l_{ij}=\left(a_{ij}-\sum_{k=1}^{j-1}l_{ik}l_{jk}\right)/l_{jj},\quad i\geqslant j.&#36;&#36;
Special, when&#36;i=j&#36;Time
&#36;&#36;l_{ii}=\sqrt{a_{ii}-\sum_{k=1}^{i-1}l_{ik}^2}&#36;&#36;
He's also called square root decomposition because of the need for a lot of square root changes.</p>
<h3>QR decomposition</h3>
<p>Since the LU triangle does not solve the problem of some pathological equations, and since some reversible matrices do not exist, we need to propose a better method of decomposition, which is QR, which exists for all reversible matrices.</p>
<h4>Concept of QR decomposition</h4>
<p>Definitions: if true (and) non-chilling matrix&#36;A&#36;It'll turn into a positive (West) matrix.&#36;Q&#36;Up-triangular to (re)activity
Matrix&#36;R&#36;product, or
&#36;&#36;A=QR:,&#36;&#36;
I call it&#36;A&#36;QR decomposition</p>
<p>More commonly used QR decomposition is in real terms, so the main discussion that follows is the decomposition of the decomposition matrix, with partial decomposition.</p>
<p>Theorem: any real, non-eccentric&#36;n&#36;Class A can be broken into a regular matrix.&#36;Q&#36;The product of the upper triangle matrix R and the absolute value of the diagonal element is equal to the 1 diagonal matrix factor&#36;D&#36;Outside, decomposition is the only one.</p>
<p>So-called diagonal matrix factor.&#36;D&#36; It means... &#36;A=QD^{-1}DR&#36;  When a rule&#36;R&#36;All the diagonal elements are positive.&#36;D=I&#36; It's the only time to break up.</p>
<p>The basic QR breakup is being carried out using Schmit, as follows:</p>
<ul>
<li>&#36;A&#36;The column vector is due&#36;A&#36;It's not weird, so it's not linear, it's a column vector.&#36;\alpha_i&#36; Getting Schmit on the line.&#36;(\beta_1,\beta_2,\cdots,\beta_n)=(\alpha_1,\alpha_2,\cdots,\alpha_n)B&#36;</li>
<li>You can give it to me now. &#36;Q=AB&#36;</li>
<li>&#36;B^{-1}=R&#36;It's the Triangular Matrix. &#36;A=QB^{-1}=QR&#36;</li>
</ul>
<h4>Calculation of QR breakdown</h4>
<p>Conversation to solve the QR is more complex, so we are here to present some other methods of calculation that have better effects in more complex issues.</p>
<h5>Givens Method</h5>
<p>The Givens method is based on the primary rotation of the matrix, through constant left Multiplication&#36;R&#36; Take it off.&#36;A&#36;Non-zero elements, finally simplified to the upper triangle matrix</p>
<p>Theoretically: Any non-fiscal matrix can be transformed into a triangle through a left-to-left first-class rotation.</p>
<p>This is proof of the process of finding the QR decompose.</p>
<p>Against Real Reversible Matrix&#36;A=(a_i&#36;) Left multiplied by primary rotation&#36;R_{ij}&#36;From now on, only change.&#36;A&#36;No. No.&#36;i&#36;Line and&#36;j&#36; Line elements. Set&#36;&#36;A^{\prime}=R_{ij}A&#36;&#36;
The effect of the change is
&#36;&#36;a_{ig}^{\prime}=ca_{ig}+sa_{jg},\quad a_{jg}^{\prime}=-sa_{ig}+ca_{jg},\quad a_{jg}^{\prime}=a_{jg},\quad p\neq i,j;g=1,2,\cdots,n.&#36;&#36;
If you want to. &#36;a_{jg_0}^{\prime}=0&#36;  Then just...&#36;a_{ig_0}\text{和 }a_{jg_0}&#36; One is not zero and takes
&#36;&#36;s=\frac{a_{jg_0&#125;&#125;{\sqrt{a_{ig_0}^2+a_{jg_0}^2&#125;&#125;,\quad c=\frac{a_{ig_0&#125;&#125;{\sqrt{a_{ig_0}^2+a_{jg_0}^2&#125;&#125;&#36;&#36;
At this point,
That's right.&gt;&#36;0.00
Which means that the effect of the change is &#36;g_0&#36;Columns &#36;j&#36;I don't know.&#36;g_0&#36;Columns&#36;i&#36;Lines are positive. Other elements are the same.</p>
<p>With this change, we can change the matrix to the upper triangle, which is,
&#36; \begin{aligned}A^(n-1)}&amp;=\boldsymbol{R}<em>{n-1,n}\cdots\boldsymbol{R}</em>{12}\boldsymbol{A}\&amp;=\begin{bmatrix}a_{11}^{(1)}&amp;a_{12}^{(1)}&amp;\cdots&amp;a_{1n}^{(1)}\0&amp;a_{22}^{(2)}&amp;\cdots&amp;a_{2n}^{(2)}\\vdots&amp;\vdots&amp;\ddots&amp;\vdots\0&amp;0&amp;\cdots&amp;a_{nn}^{(n-1)}\end{bmatrix}\end{aligned}&#36;&#36;</p>
<p>It actually implies a QR breakdown.
&#36;&#36;{R}={A}^{(n-1)}&#36;&#36;
&#36;&#36;Q=(R_{n-1,n}\cdotp\cdotp\cdotp R_{12})^{-1}&#36;&#36;
There's a QR decomposition because the primary rotation matrix is positive.
&#36;&#36;A=QR.&#36;&#36;</p>
<p>The Givens method needs to be calculated. &#36;\frac{n(n-1)}{2}&#36; The size of a primary rotation matrix, so it's not practical on a high-dimensional matrix.</p>
<h5>Housholder Method</h5>
<p>Theorem: Any real&#36;n&#36;Step Matrix&#36;A&#36;Available primary reflection matrix&#36;H=I-2\omega\omega^\mathrm{T}&#36;Turn it into a triangle.</p>
<p>The methodology is concrete proof that we are not going to discuss it, and that its idea is also to transform it into an upper trigonometric matrix through constant left-to-left multiplication of the primary reflection matrix, and then reverse it with a method similar to those of the Givens, with the QR decomposition of the original matrix, with a linear increase in the volume and array dimensions, which is faster to deal with the high-dimensional non-sorting matrix than the Givens method.</p>
<h3>Max decomposition</h3>
<p>These two sections are mainly about&#36;n&#36;Several decompositions of the stair array, starting with this section, will introduce the decomposition of several commonly used long arrays.</p>
<p>Definitions: Establishment&#36;m\times n&#36;Matrix</p>
<p>&#36;&#36;\mathbf{A}=\begin{bmatrix}a_{11}&amp;a_{12}&amp;\cdots&amp;a_{1n}\a_{21}&amp;a_{22}&amp;\cdots&amp;a_{2n}\\vdots&amp;\vdots&amp;&amp;\vdots\a_{m1}&amp;a_{m2}&amp;\cdots&amp;a_{mn}\end{bmatrix},&#36;&#36;</p>
<p>If it was&#36;m\leqslant n&#36;Time, there is rank&#36;\boldsymbol{A}=m;&#36;Or when?&#36;m\geqslant n&#36;Time, there is rank&#36;\boldsymbol{A}=n&#36;, the two rectangular arrays are referred to as the maximum square array, which is also referred to as the largest row matrix (the full array or the short array) and the largest column array (the full array or the high array)</p>
<p><strong>The largest rectangular array refers to this rectangular matrix that has the largest of its kind.</strong></p>
<p>It's clear that the largest contours have the following characteristics.
&#36;&#36;\topername{rank}(AA^mathrm{T})=m,\quad\matbf{A}=left(a ij}\right)<em>{m\times n},m\leqslant n&#36;&#36;
或
&#36;&#36;\mathrm{rank}(A^\mathrm{T}A)=n:,\quad A=(a</em>{ij})_{m\times n},m\geqslant n.&#36;&#36;</p>
<p>Definitions: Establishment&#36;A&#36;Yes&#36;m\times n&#36;and is &#36;r&gt;0&#36;的复矩阵，且记为&#36;A\mathbb{C}r^m\times}&#36;, if matrix exists
&#36;B\in\mathbb{C}_r^{m\times r}&#36;and&#36;C\in\mathbb{C}_r^{r\times n}&#36;♪ That makes
&#36;&#36;A=BC:,&#36;&#36;
It's called decomposition as the largest decomposition of matrix A.</p>
<p>Apparently, when&#36;A&#36;When a column has its largest array, or a row has its largest array,&#36;A&#36;One factor is the unit matrix, the other factor is&#36;A&#36;In itself, it's called the greatest division.</p>
<p>Theorem: Set &#36;A\in\mathbb{C}_r^{m\times n}&#36;, it must exist. &#36;\boldsymbol{B}\in\mathbb{C}_r^{m\times r}&#36; and&#36;C\in\mathbb{C}_r^{r\times n}&#36;Make
&#36;&#36;A=BC.&#36;&#36;
The theorem's proof process is to find the matrix.&#36;B,C&#36;The process, the idea is to start with the matrix.&#36;A&#36;the standard shape of the line, then standardized&#36;A&#36;Before&#36;r&#36;Column as&#36;B&#36;Take it.&#36;A&#36;Before&#36;r&#36;Non-zero line as matrix&#36;C&#36;</p>
<p>If we put&#36;A&#36;Standardize columns before retake&#36;r&#36;Column as&#36;B&#36;Front&#36;r&#36;Non-zero line as matrix&#36;C&#36; We can get another maximum breakdown. This means,<strong>Maximum decomposition is not unique</strong>But the possibilities are limited.
&#36;&#36;C^{\mathrm{H&#125;&#125;(CC^{\mathrm{H&#125;&#125;)^{-1}(B^{\mathrm{H&#125;&#125;B)^{-1}B^{\mathrm{H&#125;&#125;&#36;&#36;
It's the same. It's the Mor-Penrose that we're looking at in the matrix analysis.</p>
<h3>Odd altruistic split between SVD and extreme split</h3>
<p>The importance of a matrix's oddly differentiating values in the matrix theory is self-evident, such as the frequency method of classical control, which has been developed thanks to the help of the matrix's strangely differentiating values. Here, only the nature of the oddly different values is given and the matrix is broken down by oddly different values.</p>
<p>First of all, we need to give some preparatory knowledge about the matrix's characterizations and oddities.</p>
<p>Title: Set &#36;A\in\mathbb{C}^m\times n&#36;, there is</p>
<ul>
<li>&#36;A^\mathrm{H}A&#36; and&#36;AA^\mathrm{H}&#36; (a) The characteristic values are non-negative;</li>
<li>&#36;A^\mathrm{H}A&#36; and&#36;AA^\mathrm{H}&#36;The non-zero feature value is the same.</li>
</ul>
<p>Definitions: with &#36;A\mathbb{C}<em>r^m\times n}, A^
&#36;US&#36;US&#36;US&#36;US&#36;&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;&gt;\lambda</em>== sync, corrected by elderman == @elder man
Name &#36;\sigma_i=\sqrt{\lambda_i}\left(i=1,2,\cdots,r\right)&#36;Yes<strong>Matrix &#36;A&#36; Positively odd</strong>Abbreviations<strong>Strange.</strong>
It's a definition and a proposition. &#36;,A&#36; and&#36;A^\mathrm{H}&#36; It's the same thing.</p>
<p>Definitions: Establishment&#36;A,B\in\mathbb{C}^{m\times n}&#36;if&#36;m&#36;Classical Matrix&#36;U&#36;and&#36;n&#36;Classical Matrix&#36;V&#36;♪ That makes ♪
&#36;&#36;B=UAV,&#36;&#36;
Name&#36;A&#36;and&#36;B&#36;Equivalent or low.</p>
<p>Theoretically: If&#36;A&#36;and &#36;B&#36; equal price, then&#36;A&#36;and&#36;B&#36;It's the same thing.</p>
<p>Theorem: with &#36;A\mathbb{C}<em>r^{m\times n}&#36;,则存在&#36;m&#36;阶酉矩阵&#36;U&#36;和&#36;n&#36;阶西矩阵&#36;V, make
&#36;&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;&#36;US&#36;&#36;US&#36;US&#36;US&#36;US&#36;&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;&amp;0\0&amp;0\end{bmatrix}&#36;&#36;
或
&#36;&#36;A=U{\begin{bmatrix}\Delta&amp;0\0&amp;♪ I'll be back ♪
Of which &#36;\Delta=\oporatorname{diag} (\sigma)</em>{1},\sigma_{2},\cdots,\sigma_{r}),\lambda_{i}&#36; 为&#36;AA^\mathrm{H}&#36;的非零特征值，且&#36;\sigma_i=\sqrt{\lambda_{i&#125;&#125;\left(i=1,2,\cdots,r\right)&#36;,而&#36;\sigma_i&#36; 是&#36;All A.A.s. <strong>This is called the Matrix.&#36;A&#36;It's essentially research.&#36;A&#36;Equivalent price with a rectangular diagonal matrix</strong></p>
<p><strong>The odd altruistic decomposition of the calculation matrix is simple, and it is only necessary to calculate the odd amount directly, in order. Matrix&#36;U,V&#36;All you have to do is calculate.&#36;A^HA,AA^H&#36;column matrix of characteristics vectors</strong></p>
<p>Set Reversible Matrix&#36;A&#36;The oddly decomposed value to&#36;A=UDV^\mathrm{H}&#36;, and its reverse odd value is broken down to&#36;A^{-1}=VD^{-1}U^{\mathrm{H&#125;&#125;.&#36;So, if &#36;A&#36; The odd value is &#36;\sigma.&gt;0&#36;,则 &#36;A^{-1}&#36; 的奇异值为 &#36;1/\sigma_n\geqslant1/\sigma_n&#36; &#36;\geqslant\cdotp\cdotp\cdotp\geqslant1/\sigma_1&gt;&#36;0.00
Set&#36;A=U_1DV^\mathrm{H}&#36;Yes.&#36;A&#36;The oddly differentiating,
&#36;&#36;P=U_1DU_1^H,\quad U=U_1V^H,&#36;&#36;
You can get another interesting decomposition of the matrix -- extreme decomposition.</p>
<p>Theorem: Set &#36;A\in\mathbb{C}^{n\times n}&#36;, there's a matrix. &#36;U&#36; And the only semi-positive matrix. &#36;P&#36;♪ That makes ♪
&#36;&#36;A=PU,&#36;&#36;
It's called the Matrix.&#36;A&#36;The extreme breakdown of the matrix.&#36;P&#36;and&#36;U&#36;Separately known as&#36;A&#36;Elmitt and Westin.</p>
<p>In particular, let's just look at some of the properties of the characteristic values and the odd values.</p>
<p>Theorem (genius and feature values): Set&#36;\lambda&#36;Yes.&#36;n&#36;Step Matrix&#36;A&#36;a characteristic value that will&#36;A&#36;The maximum odd value and the smallest odd value are as follows:&#36;\sigma_\mathrm{max}(\boldsymbol{A})&#36;and&#36;\sigma_\mathrm{min}(\boldsymbol{A})&#36;, then&#36;\sigma_\mathrm{max}(\boldsymbol{A})\geqslant|\lambda|\geqslant\sigma_\mathrm{min}(\boldsymbol{A}).&#36;In other words, the maximum odd value of the matrix and the smallest odd value are the upper and lower bounds of its signature value.</p>
<p>Theorem (surprising and matrix tracks): set &#36;A\in\mathbb{C}^{m\times n}&#36;, tr&#36;(A^\mathrm{H}A)=\sum_{i=1}\sigma_i^2.&#36;</p>
<p>Theoretically (generic and weird array): matrix&#36;A&#36;Fill it up.&#36;\Leftrightarrow A&#36;It's not very special, square.&#36;A&#36;It's not weird.&#36;\Leftrightarrow A&#36; None of the odd values. In fact, the matrix is equal to the number of non-zero-eccentric values.</p>
<h3>Spectrolysis</h3>
<p>All the matrix decomposition is designed to simplify the problem, so this chapter presents a very good matrix: one that can be polarized.</p>
<h4>Formal Matrix</h4>
<p>Definitions: Establishment&#36;A&#36;It's a square formation in the plural, if there is one.
&#36;&#36;AA^\text{н}=A^\text{н}A,&#36;&#36;
Name &#36;A&#36; To the formal matrix.</p>
<p>If&#36;A&#36;It's real.&#36;n&#36;Step-by-step, and yes.
&#36;&#36;AA^{\mathrm{T&#125;&#125;=A^{\mathrm{T&#125;&#125;A,&#36;&#36;
Name &#36;A&#36; It's a formal matrix.</p>
<p>We're easy to verify, symmetrical, objectional.&#36;A=-A^{\mathrm{T&#125;&#125;&#36;) The positive matrix is a real matrix; and the quartz, the Elmitt matrix, the anti-Elmitt matrix (i.e.&#36;A=-A^{\mathrm{H&#125;&#125;&#36;It's all part of the formal matrix.</p>
<p><strong>Theorem: Set &#36;A\in\mathbb{C}^{n\times n}&#36;, then &#36;A&#36; It's a necessary condition for a cyborg similar to the diagonal matrix.&#36;A&#36; As Formal Matrix</strong></p>
<p>In fact, the symmetrical matrix is similar to a diagonal matrix, which is the narrowness of the on-line algebra of this theorem, which is the final answer to the question of polarization.</p>
<p>Theorem: For the formal matrix, we can easily give the following inferences:</p>
<ul>
<li>The proper triangle is the angle matrix.</li>
<li>The formal matrix has...&#36;n&#36;A two-to-two unit characteristic vector</li>
<li>The formal matrix has...&#36;n&#36;A different feature value</li>
<li>Regular matrix character values with positive vectors</li>
<li>For a formal matrix, its characteristic values and elements satisfy&#36;\sum_{i=1}^n\mid\lambda_i\mid^2=\sum_{i,j=1}^n\mid a_{ij}\mid^2&#36;</li>
</ul>
<h4>Regular matrix spectrolysis</h4>
<p>We know that the formal matrix is similar to a diagonal matrix.</p>
<p>Set&#36;A&#36;It's a formal matrix, so there's a tatter matrix.&#36;U&#36;Make&#36;U^\mathrm{H}AU=\operatorname{diag}\left(\lambda_1,\lambda_2,\cdots,\lambda_n\right)&#36; Which means...
&#36;&#36;\mathbf{A}=\boldsymbol{U}\mathrm{diag}(\lambda_1,\lambda_2,\cdots,\lambda_n)\boldsymbol{U}^\mathrm{H}.&#36;&#36;
You!&#36;\boldsymbol{U}=(\boldsymbol{\alpha}_1,\boldsymbol{\alpha}_2,\cdots,\boldsymbol{\alpha}_n)&#36; There is.
&#36;A=(\bardsymbol{\alpha}1,\bardsymbol{\alpha}2,\cdots,\bardsymbol{\alpha}\begin{bmatrix}\lambda 1&amp;&amp;&amp;\&amp;\lambda_2&amp;&amp;\&amp;&amp;\ddots&amp;\&amp;&amp;&amp;\lambda_n\end{bmatrix}\begin{bmatrix}\boldsymbol{\alpha}_1^\mathrm{H}\\\boldsymbol{\alpha}_2^\mathrm{H}\\vdots\\boldsymbol{\alpha}_n^\mathrm{H}\end{bmatrix}&#36;&#36;
&#36;&#36;== sync, corrected by elderman ==
Because&#36;\lambda_i&#36;is the characteristic value of the matrix, and&#36;\alpha_i&#36;It's a positive unit characterization vector of the characteristic value, so we call it<strong>Formal Matrix&#36;A&#36;Spectrolysis or characterization</strong></p>
<p>Simplify the integration of items with the same feature values
&#36;&#36;A=\lambda_1P_1+\lambda_2P_2+\cdots+\lambda_sP_s&#36;&#36;</p>
<h4>A simple matrix spectrolysis</h4>
<p>We already know.&#36;,n&#36; A stair array, called a mere matrix when the contemporary number repetition is equal to geometry, can be agular, but not necessarily a Western diagonal (i.e. not a formal matrix).</p>
<p>However, a mere matrix can be similar to a formal matrix definition.&#36;A&#36;It's a spectrolysis.&#36;\lambda_1,\lambda_2,\cdotp\cdotp\cdotp,\lambda_n&#36;Yes.&#36;A&#36;Yes.&#36;n&#36;Characteristic values;&#36;x_1,x_2,\cdotp\cdotp\cdotp,x_n&#36;Yes.&#36;A&#36;Yes.&#36;n&#36; A non-linear character vector, with
&#36;&#36;Ax_i=\lambda_ix_i,\quad i=1,2,\cdotp\cdotp\cdotp,n&#36;&#36;
You!
&#36;&#36;P=(x_1,x_2,\cdots,x_n),&#36;&#36;
&#36;&#36;\boldsymbol{\Lambda}=\begin{bmatrix}\lambda_1\&amp;\lambda_2\&amp;&amp;\ddots\&amp;&amp;&amp;\lambda_n\end{bmatrix}&#36;&#36;
则
&#36;&#36;A=P\Lambda P^{-1}.&#36;&#36;
两边转置有
&#36;&#36;A^{\mathrm{T&#125;&#125;=(P^{\mathrm{T&#125;&#125;)^{-1}{\Lambda}{P}^{\mathrm{T&#125;&#125;.&#36;&#36;
这表明&#36;A^\mathrm{T}&#36;也与对角矩阵相似.因此，设&#36;y_1,y_{2},\cdotp\cdotp\cdotp,y_{n}&#36;是&#36;A^\mathrm{T}&#36;的&#36;n&#36;个线性无关的特征向量，即
&#36;&#36;\mathbf{A}^\mathrm{T}\mathbf{y}_i=\lambda_i\mathbf{y}_i:,\quad i=1,2,\cdots,n,&#36;&#36;
把上式两端取转置得
&#36;&#36;\mathbf \mathbf \mathbf \mathbf \mathbf \y \mathbf}: \mathbf \y:, \mada i=1, 2, \cdotp\cdotp, n, &#36;
So, we say,&#36;y_i^\mathrm{T}&#36;Yes. &#36;A&#36; Left feature vector, term&#36;x_i&#36;Yes.&#36;A&#36;Right Feature Vector</p>
<p>And so...
&#36;&#36;(\mathbf{y}_1,\mathbf{y}_2,\cdots,\mathbf{y}_n)=(\mathbf{P}^\mathrm{T})^{-1}=(\mathbf{P}^{-1})^\mathrm{T},&#36;&#36;
Converted
&#36;&#36;\mathbf{P}^{-1}=\begin{bmatrix}\mathbf{y}_1^\mathrm{T}\\vdots\\mathbf{y}_n^\mathrm{T}\end{bmatrix}&#36;&#36;
Substitute&#36;PP^{-1}=P^{-1}P=I&#36;Okay.
&#36; &#36; (\bardsymbol{x}1,\bardsymbol{x}2,\cdots,\bardsymbol{x}\betin{bmatrix}\boldsymbol{y&#125;&#125;mathrm{y}\vdots\boldsymbol}ymbol} mathrm{x}(\matmary x{,\bedsymbol{x} , \boldsymbol{x} ,\codesymboldsxx}<em>n)=\boldsymbol{I},&#36;&#36;
此即
&#36;&#36;x_1y_1^\mathrm{T}+x_2y_2^\mathrm{T}+\cdots+x_ny_n^\mathrm{T}=I.&#36;&#36;
比较两端即有
&#36;&#36;y_i^\mathrm{T}x_j=\delta</em>{ij}:,\quad i,j=1,2,\cdotp\cdotp\cdotp,n,&#36;&#36;</p>
<p>We can get it.
&#36;A=(x,x 2,\cdots,\begin{bmatrix}\lambda )&amp;\ddots\&amp;&amp;\lambda_{n}\end{bmatrix}\begin{bmatrix}\mathbf{y}<em>{1}^{\mathrm{T&#125;&#125;\\vdots\\mathbf{y}</em>{n}^{\mathrm{T&#125;&#125;\end{bmatrix}\=\lambda_{1}\boldsymbol{x}<em>{1}\boldsymbol{y}</em>{1}^{\mathrm{T&#125;&#125;+\lambda_{2}\boldsymbol{x}<em>{2}\boldsymbol{y}</em>{2}^{\mathrm{T&#125;&#125;+\cdots+\lambda_{n}\boldsymbol{x}<em>{n}\boldsymbol{y}</em>{n}^{\mathrm{T&#125;&#125;.&#36;&#36;
令
&#36;&#36;G_i=x_iy_i^\mathrm{T}&#36;&#36;
即得到
&#36;&#36;A \sum i \mathbf
It's called a mere matrix.&#36;A&#36;♪ Spectrolysis, decomposition ♪&#36;n&#36;individual&#36;G_i&#36;and the coefficient of the linear combination is&#36;A&#36;Characteristic value</p>
<h2>Matrix calculus</h2>
<p>In online algebras, only add-(decrease) to the matrix, multiplication and reverse-centric algebra calculations are discussed, while they do not address at all the same level as the limits, cascades, calculus, etc. in mathematical analysis, but these calculations are also necessary when studying the issues of operational preparation and control of linear systems.</p>
<p>A classic mathematical model is the equivalent of a matrix.&#36;U&#36; Function as a variable&#36;&#36;J( \boldsymbol{U}) = | \boldsymbol{U}\boldsymbol{\alpha }- \boldsymbol{\beta }|&#36;&#36;of which&#36;\boldsymbol{U}\in\mathbb{R}^m\times n,\boldsymbol{\alpha}\in\mathbb{R}^n,\boldsymbol{\beta}\in\mathbb{R}^m&#36; In the context of binding conditions&#36;U^\mathrm{T}U=I&#36; or&#36;UU^\mathrm{T}=I&#36; the lowest point of value (matrix), a viable solution to such optimization is to seek the matrix function &#36;J(U)&#36;About Unknown Matrix&#36;U&#36; , which requires a study of the calculus of the matrix.</p>
<p>The definition of the paradigm gives us the basis for studying distance, and gives us a linear space to export measures, so we study the calculus of the matrix in this chapter, which is the collision of the most elementary analyticals with metrology.</p>
<h3>Limits of vector and matrix series</h3>
<p>Before we study calculus, we'll study limits.</p>
<h4>Limits of vector series</h4>
<p>Definition (concentrated by model): established &#36;x^{(k)},x\in\mathbb{C}^{n}(k=1,2,\cdots)&#36;If
&#36;&#36;\parallel x^{(k)}-x\parallel\to0,\quad k\to+\infty,&#36;&#36;
Name of vector sequence&#36;\langle x^(k)\rangle&#36;Condense to vector &#36;x&#36;or vector &#36;x&#36; It's a vector sequence.&#36;\langle x^(k)\rangle&#36;When?&#36;k\to+\infty&#36;The limit of time, can be recorded as
&#36;&#36;\lim_{k\to+\infty}x^{(k)}=x&#36;&#36;
or
&#36;&#36;x^{(k)}\to x,\quad k\to+\infty.&#36;&#36;</p>
<p>Depending on the value of the vector model, we can tell:<strong>Concentrate under one vector and under others</strong></p>
<p>Theoretically: In Banach space, we know that Cosy's guidelines are in place, so at this point,<strong>Vector and weight are equal.</strong> That's why it's in math analysis.</p>
<h4>Limits of the matrix series</h4>
<p>The matrix can be seen as a high-dimensional vector, so we can define the matrix limits in a similar way, and here we start with a definition of the size.</p>
<p>Definition: A matrix sequence exists&#36;\left{A^{(k)}\right}&#36;, where &#36;A^{(k)}=(a_{ij})^{(k)})\in\mathbb{C}^{n\times n}&#36;and when &#36;k\to+\infty&#36;Time&#36;,a_{ij}^{(k)}\to a_{ij}&#36;, or&#36;\left{A^{(k)}\right}&#36;Keep it together and take the matrix. &#36;\boldsymbol A=(a_{ij})&#36;Called&#36;\left{\boldsymbol A^{(k)}\right}&#36;limit, or term&#36;\left{\boldsymbol A^{(k)}\right}&#36;Concentrating&#36;A&#36;Remember
&#36;&#36;\lim_{k\to-\infty}A^{(k)}=A\quad\text{或}\quad A^{(k)}\to A.&#36;&#36;
It's called dispersing arrays.</p>
<p>Theorem: The aforementioned matrix compression definition is equivalent to that of the matrix model definition, i.e., the equivalent value is equal to
&#36;&#36;\parallel A^{(k)}-A\parallel\to0,\quad k\to+\infty&#36;&#36;</p>
<p>Theoretically: Same as vector contraction, which is equivalent to the model definition of the matrix</p>
<p>For the extreme operation of the matrix, we can give the following characteristics:</p>
<ul>
<li>Linear: with &#36;\pepratorname*{lim}<em>{k\to+\infty}A^{(k)}=A,\operatorname*{lim}</em>{k\to+\infty}B^{(k)}=B&#36; 则&#36;\lim_{k\to+\infty}(a\boldsymbol{A}^{(k)}+b\boldsymbol{B}^{(k)})=a\boldsymbol{A}+b\boldsymbol{B},\quad a,b\in\mathbb{C}&#36;</li>
<li>Multiplication: with &#36;\oporatorname*lim}<em>{k\to+\infty}A^{(k)}=A,\operatorname*{lim}</em>{k\to+\infty}B^{(k)}=B&#36;   则 &#36;\lim_{k\to+\infty}A^{(k)}B^{(k)}=AB&#36;</li>
<li>Retroactive: with &#36;\pepratorname*lim}<em>{k\to+\infty}A^{(k)}=A&#36; 且 &#36;A^{k}&#36;均可逆，则&#36;{(A^{(k)})^{-1&#125;&#125;&#36; 也收敛，并且&#36;\lim</em>{k\to+\infty}(A^{(k)})^{-1}=A^{-1}&#36;</li>
</ul>
<p>Theorem: with matrix sequences&#36;\left{\boldsymbol{A}^{(k)}\right}:\boldsymbol{A},\boldsymbol{A}^2,\cdotp\cdotp\cdotp A^k,\cdotp\cdotp\cdotp&#36;, then&#36;\lim\boldsymbol{A}^k=\boldsymbol{0}&#36; The only necessary condition is the matrix.&#36;A&#36;All feature values are modelled less than 1, i.e.&#36;A&#36;spectral radius less than 1.
&#36;rho&lt;1.&#36;&#36;</p>
<p>Theorem: If for Matrix &#36;A&#36; There's a model for a \\parallel A\parallel&lt;&#36;1, then
&#36;&#36;\lim_{k\to+\infty}A^k=\mathbf{0}.&#36;&#36;</p>
<h3>Matrix Numbers and Matrix Functions</h3>
<h4>Definition and Denunciation of Matrix Levels</h4>
<p>In order to better study matrix functions, we need to first study the matrix-level theory, whose definition and that of several tiers are very similar in nature.</p>
<p>Definition: A matrix sequence exists
&#36;&#36;A^{(1)},A^{(2)},\cdots,A^{(k)},\cdots,&#36;&#36;
of which&#36;A^{(k)}=(a_{ij}^{(k)})\in\mathbb{C}^{n\times n}&#36;It's called Infinity.
&#36;&#36;A^{(0)}+A^{(1)}+A^{(2)}+\cdots+A^{(k)}+\cdots &#36;&#36;
As the number of arrays,&#36;\sum_{k=0}^{\infty}\boldsymbol{A}^{(k)},\boldsymbol{A}^{(k)}&#36;General items called array numbers, i.e., yes
&#36;&#36;\sum_{k=0}^{+\infty}A^{(k)}=A^{(0)}+A^{(1)}+A^{(2)}+\cdots+A^{(k)}+\cdots.&#36;&#36;</p>
<p>Definitions: Before the hierarchy&#36;k+1&#36;and
&#36;&#36;S^{(k)}=A^{(0)}+A^{(1)}+A^{(2)}+\cdots+A^{(k)}&#36;&#36;
Part of what is called a hierarchy, if the matrix sequence&#36;{S}^{(k)}&#36;Take it easy. There are limits. &#36;S&#36;Yes.
&#36;&#36;\lim_{k\to+\infty}\mathbf{S}^{(k)}=S,&#36;&#36;
This post is part of our special coverage Global Voices 2011.&#36;S&#36; Called the sum of the grades.
&#36;&#36;S=\sum_{k=0}^{+\infty}A^{(k)}.&#36;&#36;
It's called dispersing arrays.</p>
<p>Depending on the nature of the matrix, it is easy to know:<strong>The key requirements of the matrix count are corresponding.&#36;n^2&#36;Declining of levels</strong></p>
<p>By definition, we can easily give the following characteristics.</p>
<ul>
<li>&#36;\text{若 }\sum_{k=0}^{+\infty}A^{(k)}\text{ 收敛,则}\lim_{k\to+\infty}A^{(k)}=\mathbf{0};&#36;</li>
<li>&#36;\text{若 }\sum_{k=0}^{+\infty}\mathbf{A}^{(k)}=\mathbf{S},\sum_{k=0}^{+\infty}\mathbf{B}^{(k)}=\mathbf{S}^{\prime},\text{则}&#36; &#36;\sum_{k=0}^{+\infty}(A^{(k)}\pm B^{(k)})=S\pm S^{^{\prime&#125;&#125;;&#36;</li>
<li>&#36;\text{若 }\sum_{k=0}^{+\infty}A^{(k)}=S,\text{则}\sum_{k=0}^{+\infty}\mu A^{(k)}=\mu S,\mu\in\mathbb{C}.&#36;</li>
</ul>
<p>Definitions: number of arrays &#36;\sum_k=0A^{(k)}=A^{(0)}+A^{(1)}+A^{(2)}+\cdots+A^{(k)}+\cdots&#36;, where &#36;A^{(k)}=(a_{ij}^{(k)})\in\mathbb{C}^{n\times n}.&#36;If &#36;n^2&#36; Number of levels
&#36;&#36;a_{ij}^{(0)}+a_{ij}^{(1)}+a_{ij}^{(2)}+\cdots+a_{ij}^{(k)}+\cdots,\quad i,j=1,2,\cdots,n&#36;&#36;
It's absolute, it's called the matrix numbers.</p>
<p>Theorem: number of arrays &#36;\sum_{k=0}^{+\infty}A^{(k)}&#36; It's a condition of absolute restraint.&#36;\sum_{k=0}^{+\infty}\parallel A^{(k)}\parallel=\parallel A^0\parallel+\parallel A^{(1)}\parallel+\parallel A^{(2)}\parallel+\cdotp\cdotp\cdotp+\parallel A^{(k)}\parallel+\cdotp\cdotp\cdotp&#36; Take it down.&#36;|\boldsymbol A^(k)|&#36; Yes&#36;A^(k)&#36; Any kind of standard.</p>
<p>Theorem: two arrays
&#36;&#36;A^{(1)}+A^{(2)}+\cdots+A^{(k)}+\cdots,\quad A^{(k)}\in\mathbb{C}^{n\times n},&#36;&#36;
&#36;&#36;B^{(1)}+B^{(2)}+\cdotp\cdotp\cdotp+B^{(k)}+\cdotp\cdotp\cdotp,\quad B^{(k)}\in\mathbb{C}^{n\times n}&#36;&#36;
It's a total retreat. &#36;A,B&#36;, multiply them by the number of arrays
&#36;&#36;A^{(1)}B^{(1)}+(A^{(1)}B^{(2)}+A^{(2)}B^{(1)})+\cdots+&#36;&#36;
&#36;&#36;(A^{(1)}B^{(k)}+A^{(2)}B^{(k-1)}+\cdots+A^{(k)}B^{(1)})+\cdots &#36;&#36;
It's absolutely condensed and has a peace.&#36;AB.&#36;</p>
<h4>Nature of arrays</h4>
<p>Based on the definition of the array number and some analytical knowledge, we can conduct studies on the nature of the array number.</p>
<p>Theorem: set arrays&#36;\sum_{k=0}^{+\infty}A^{(k)}&#36;Absolute Depression</p>
<ul>
<li>&#36;\text{级数 }\sum_{k=0}^{+\infty}A^{(k)}\text{ 收敛;}&#36;</li>
<li>&#36;\text{级数 }\sum_{k=0}^{+\infty}A^{(k)}\text{ 在任意改变各项的次序后仍然收敛,且其和不变}.&#36;</li>
</ul>
<p>Theorem: Set &#36;P,Q&#36;Yes&#36;n&#36;Classes non-surreal arrays, if class numbers&#36;\sum_{t=0}A^{(k)}&#36;Denunciation (or absolute curtailment), arrays&#36;\sum_{k=0}^{+\infty}\boldsymbol{PA}^{(k)}Q&#36;It's also lulling.</p>
<p>Definitions:
&#36;&#36;c_0I+c_1A+c_2A^2+\cdots+c_kA^k+\cdots &#36;&#36;
The number of arrays is called the number of arrays, where&#36;c_i\in\mathbb{C},A\in\mathbb{C}^n\times n.&#36;</p>
<p>Theoretically: if true&#36;|c_0||\boldsymbol{I}|+\sum_{k=1}|c_k||\boldsymbol{A}|^k&#36;Consistency, array class &#36;c_0\boldsymbol{I}+&#36;
&#36;c_{1}\boldsymbol{A}+c_{2}\boldsymbol{A}^{2}+\cdots+c_{k}\boldsymbol{A}^{k}+\cdots&#36;Absolutely.&#36;\parallel\boldsymbol{A}\parallel&#36;As Matrix &#36;A&#36; Some kind of model.</p>
<p>Inference: if the matrix &#36;A&#36; ♪ Some kind of standard&#36;\parallel A\parallel&#36;In Level
&#36;&#36;\sum_{k=0}^{+\infty}c_0z^k=c_0+c_1z+c_2z^2+\cdots+c_kz^k+\cdots &#36;&#36;
, the number of arrays &#36;\sum_{k=0}^{+\infty}c_k\mathbf{A}^k&#36; Absolutely.</p>
<p>Theorem: Set&#36;A\in\mathbb{C}^{n\times n}&#36;if&#36;A&#36;spectral radius&#36;\rho(\boldsymbol{A})&#36;The value is in pure amount&#36;z&#36;Number of grades&#36;\sum_{k=0}c_kz^k&#36;. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .&#36;\sum_{k=0}^{+\infty}c_kA^k&#36; Absolute restraint; if &#36;A&#36; The characteristic value has a number in the register&#36;\sum_{k=0}^{+\infty}c_kz^k&#36; , the number of arrays&#36;\sum_{k=0}^{+\infty}c_kA^k&#36;Spread out.</p>
<p>Theorem: number of arrays&#36;I+A+A^2+\cdots+A^k+\cdots&#36; The condition of absolute restraint is&#36;A&#36;& spectral radius&lt;1&#36; 且该级数的和为&#36;\left(I-A\right)^{-1}&#36;</p>
<h4>Definition of Matrix Functions</h4>
<p>We know the compound variable.&#36;z&#36; Level
&#36; \begin{aligned}&amp;\mathrm{e}^{z}=1+\frac{z}{1!}+\frac{z^{2&#125;&#125;{2!}+\frac{z^{3&#125;&#125;{3!}+\cdots+\frac{z^{k&#125;&#125;{k!}+\cdots,\&amp;\mathrm{sin}z=z-\frac{z^{3&#125;&#125;{3!}+\frac{z^{5&#125;&#125;{5!}-\cdots+(-1)^{k}\frac{z^{2k+1&#125;&#125;{(2k+1)!}+\cdots,\&amp;\cos z=1-\frac{z^2}+\frac{z^4&#125;&#125;}cdos+(-1)^k}z{2k&#125;&#125;cdos\end{aligned} &#36;
It's all flat.</p>
<p>So for any matrix,&#36;A&#36; Matrix class
&#36; \begin{aligned}&amp;I+\frac{A}{1!}+\frac{A^{2&#125;&#125;{2!}+\frac{A^{3&#125;&#125;{3!}+\cdots+\frac{A^{k&#125;&#125;{k!}+\cdots,\&amp;A-\frac{A^{3&#125;&#125;{3!}+\frac{A^{5&#125;&#125;{5!}-\cdots+(-1)^{k}\frac{A^{2k+1&#125;&#125;{(2k+1)!}+\cdots,\&amp;+cdots+(1)^k\frac{2k)&#125;&#125;cdots\end{aligned} &#36;
It's a total retreat. &#36;e^A,sinA,cosA&#36;</p>
<p>Definitions:&#36;\text{设实函数 }y=f(x),A,B\in\mathbb{C}^{n\times n},\text{称 }B=f(A)\text{为矩阵 }A\text{ 的函数}.&#36;</p>
<p>For matrix functions, we can naturally give the following inferences.</p>
<ul>
<li>If there is. &#36;AB=BA&#36;  then &#36;\mathrm{e}^A\bullet\mathrm{e}^B=\mathrm{e}^B\bullet\mathrm{e}^A=\mathrm{e}^{A+B}.&#36;</li>
<li>&#36;\text{对任意矩阵 }A\in\mathbb{C}^{n\times n},\mathrm{e}^A\text{ 总是可逆的(非奇异的)且(e}^A)^{-1}=\mathrm{e}^{-A}.&#36;</li>
<li>&#36;(\mathrm{~e}^A)^m=\mathrm{e}^{mA}(m\text{为整数)}.&#36;</li>
</ul>
<h4>Arguments for matrix functions</h4>
<p>It's very difficult to define the value of a matrix function directly. We need to calculate a very complex array multiplication. We use examples here to describe if this operation is simplified.</p>
<p>Known 4-step matrix&#36;A&#36;Characteristic values for each &#36;\pi,-\pi,0,0&#36; Please. &#36;e^A,sinA,cosA&#36;</p>
<p>Because... &#36;A&#36; The feature equation is
&#36;&#36;\det(\lambda\boldsymbol{I}-\boldsymbol{A})=(\lambda-\pi)(\lambda+\pi)\lambda^2=\lambda^4-\pi^2\lambda^2=0&#36;&#36;
According to Hamilton-Cally's theorem,
&#36;&#36;A^4=\pi^2A^2&#36;&#36;
Therefore, all items larger than four can use the theorem lower step, and the matrix of the highest step within the entire scale is 3 steps, which can be easily calculated as follows:
&#36; \begin{aligned}sinA=&amp;\mathbf{A}-\frac{1}{3!}\mathbf{A}^{3}+\frac{1}{5!}\mathbf{A}^{5}-\frac{1}{7!}\mathbf{A}^{7}+\frac{1}{9!}\mathbf{A}^{9}-\cdots\=&amp;\mathbf{A}-\frac{1}{3!}\mathbf{A}^{3}+\frac{1}{5!}\pi^{2}\mathbf{A}^{3}-\frac{1}{7!}\pi^{4}\mathbf{A}^{3}+\frac{1}{9!}\pi^{6}\mathbf{A}^{3}-\cdots\=&amp;\mathbf{A}+\left(-\frac{1}{3!}+\frac{1}{5!}\pi^{2}-\frac{1}{7!}\pi^{4}+\frac{1}{9!}\pi^{6}-\cdots\right)\mathbf{A}^{3}\=&amp;\mathbf{A} \mathbf} \mathbf{A} \mathbf{A}, \mathbf{A}, \end{aligned}
The rest of the problem can be solved in a similar way.<strong>The core is about a simple number of steps that can be derived from multiple features.</strong> There are, of course, some array multiplications that need to be calculated.</p>
<p>Another way is to use a special theorem, along the following lines:</p>
<p>Assumptions Matrix&#36;A&#36;Similar to a diagonal matrix, you can find it.
&#36;&#36;C^{-1}AC=\operatorname{diag}(\lambda_1,\lambda_2,\cdots,\lambda_n)&#36;&#36;
Alternative formula
&#36; \begin{aligned}&amp;e^A=C\cdot\mathrm{diag}(\mathrm{e}^{\lambda_1},\mathrm{e}^{\lambda_2},\cdots,\mathrm{e}^{\lambda_n})\cdot C^{-1},\&amp;sinA=C\cdot\mathrm{diag}(\sin\lambda_1,\sin\lambda_2,\cdots,\sin\lambda_n)\cdot C^{-1},\&amp;cosA=C\cdot\mathrm{diag}(\cos\lambda 1,\cos\lambda 2,\cdos,\cos\lambda n)\cdotC^.\end{aligned} &#36;
As for the more complex Jordan and non-typical functions, this is not discussed here.</p>
<h3>Matrix Scores and Scores</h3>
<h4>Numerical Matrix Wizard to Real Variables</h4>
<p>Definitions: if matrix&#36;A=(a_{ij})&#36;The Elements&#36;a_{ij}&#36;Both variables&#36;t&#36;function, i.e.
&#36; \\matbf{A}(t)=\begin{bmatrix}a 11}(t)&amp;a_{12}(t)&amp;\cdots&amp;a_{1n}(t)\\a_{21}(t)&amp;a_{22}(t)&amp;\cdots&amp;a_{2n}(t)\\vdots&amp;\vdots&amp;&amp;\vdots\\a_{m1}(t)&amp;a_{m2}(t)&amp;\cdots&amp;a mn}(t)end{bmatrix}, &#36;
Name&#36;\boldsymbol{A}(t)&#36;Yes<strong>Function Matrix</strong>...by extension, variable&#36;t&#36;It could be a vector or a matrix.</p>
<p>Definition: If all the elements &#36;a_{ij}(t)&#36;Yes. &#36;t=t_0&#36; , there are limits, i.e. &#36;\pepratorname* {lim} <em>{t\to t_0}a</em>{ij}\left ( t\right ) = a_{ij}&#36;, &#36;a {ij} is a constant or<strong>Matrix &#36;A(t)&#36;There are limits.</strong>, and limit value is &#36;A&#36;(Content matrix), i.e.
&#36;lim\lits t(t)=A=begin{bmatrix}a 11}&amp;a_{12}&amp;\cdots&amp;a_{1n}\\a_{21}&amp;a_{22}&amp;\cdots&amp;a_{2n}\\vdots&amp;\vdots&amp;&amp;\vdots\\a_{m1}&amp;a_{m2}&amp;\cdots&amp;a_{mn}\end{bmatrix}&#36;&#36;</p>
<p>The limits of a matrix of functions have similar properties as the normal limits of functions. &#36;t\to t_0&#36; time, function matrix&#36;\mathbf{A}(t)&#36;and &#36;B(t)&#36;There are limits. &#36;A&#36; and &#36;B&#36;, there is
&#36; \begin{aligned}&amp;\operatorname*{lim}<em>{t\to t</em>{0&#125;&#125;[\boldsymbol{A}(t)+\boldsymbol{B}(t)]=\boldsymbol{A}+\boldsymbol{B},\&amp;\operatorname*{lim}<em>{t\to t</em>{0&#125;&#125;[\boldsymbol{A}(t)\boldsymbol{B}(t)]=\boldsymbol{A}\boldsymbol{B}:,\&amp;\operatorname*{lim}<em>{t\to t</em>Other Organiser
of which&#36;A,B&#36;Both constant matrices&#36;,k&#36;As constant.</p>
<p>Definitions: If all functions&#36;a_{ij}(t)&#36;is continuous at a point or area, referred to as<strong>The function matrix is also continuous at this point or in this compartment</strong>.</p>
<p>For multivariant function matrices, there may be similar provisions to those mentioned above, which are not repeated here.</p>
<p>Definitions: set \\bardsymbol{A}(t)=\left(a ij}\left(t\right)\right)<em>{m\times n}&#36;,若 &#36;a</em>{ij}\left(t\right)\left(i=1,2,\cdots,m;j=1,2,\cdots,n\right)&#36;在 &#36;t=t_0&#36; 处(或&#36;[a,b]&#36;上)可导，则称 &#36;\boldsymbol A(t)&#36;在点 &#36;t=t_0&#36; 处(或在&#36;[a,b] US&#36;) May be directed and recorded as
&#36; \matbf&#39;}(t_0)=\frac{\mathrm{d}\mathbf{A}(t)}{\mathrm{d}t}\mid_{t=t_0}=\lim_{\Delta t\to0}\frac{\mathbf{A}(t_0+\Delta t)-\mathbf{A}(t_0)}{\Delta t}&#36;&#36;
&#36;&#36;=\begin{bmatrix}a&#39;<em>{11}(t_0)&amp;a&#39;</em>{12}(t_0)&amp;\cdots&amp;a&#39;<em>{1n}(t_0)\a&#39;</em>{21}(t_0)&amp;a&#39;<em>{22}(t_0)&amp;\cdots&amp;a&#39;</em>{2n}(t_0)\\vdots&amp;\vdots&amp;&amp;\vdots\a&#39;<em>{m1}(t_0)&amp;a&#39;</em>{m2}(t_0)&amp;\cdots&amp;a&#39;<em>{mn}(t_0)\end{bmatrix}</em>{m\times n}.&#36;&#36;</p>
<p>It's not hard to prove it.</p>
<ul>
<li>&#36;\mathbf{A}(t)\text{为常数矩阵的充分必要条件是 }\mathbf{A}^{\prime}(t)=\mathbf{0}&#36;</li>
<li>&#36;\text{}\matbf{A}(t)=\left(a ij}\lt(t\right)\right)<em>\text {t\light=b</em>\left(t\right)\right  m\times}text{guided, then}&#36;  &#36;&#36;\frac{\mathrm{d&#125;&#125;{\mathrm{d}t}(A(t)\pm B(t))=A^{\prime}(t)\pm B^{\prime}(t)&#36;&#36;</li>
<li>&#36;\text{若 }k(t)\text{是可导的实函数},A(t)\text{可导},\text{则}&#36;  &#36;&#36;\frac{\mathrm{d&#125;&#125;{\mathrm{d}t}(k(t)\mathbf{A}(t))=k^{\prime}(t)\mathbf{A}(t)+k(t)\mathbf{A}^{\prime}(t)&#36;&#36;</li>
<li>&#36;\text{设 }A(t)\text{与 }B(t)\text{都可导,则}&#36; &#36;\frac{\mathrm{d&#125;&#125;{\mathrm{d}t}(\boldsymbol{A}(t)\boldsymbol{B}(t))=\boldsymbol{A}^{\prime}(t)\boldsymbol{B}(t)+\boldsymbol{A}(t)\boldsymbol{B}^{\prime}(t)&#36;</li>
<li>&#36;\text{若 }\mathbf{A}(t)\text{与 }\mathbf{A}^{-1}(t)\text{都有导数,则}&#36; &#36;\frac{\mathrm{d}\boldsymbol{A}^{-1}(t)}{\mathrm{d}t}=-\boldsymbol{A}^{-1}(t)\boldsymbol{A}^{\prime}(t)\boldsymbol{A}^{-1}(t)&#36;</li>
<li>Set Function Matrix &#36;\boldsymbol A(t)&#36;Yes. &#36;t&#36; , and &#36;t=f(x)&#36;Yes. &#36;x&#36; and &#36;\boldsymbol A(t)&#36;and &#36;f(x)&#36;, and there's &#36; \frac{\mathrm{d}\mathbf{A}(t)}{\mathrm{d}=mathbf}(t)}{\f&#39;(x)=f&#39;(x):\frac{\mathrm{d}\mathbf{A}(t)}{\mathrm{d}t}.&#36;&#36;
函数矩阵的导数本身也是一个函数矩阵，还可以再进行导数运算，故可以定义函数矩阵对实变量的高阶导数：
&#36;&#36;\frac{\mathrm{d}^k\mathbf{A}\left(t\right)}{\mathrm{d}t^k}=\frac{\mathrm{d&#125;&#125;{\mathrm{d}t}\Big(\frac{\mathrm{d}^{k-1}\mathbf{A}\left(t\right)}{\mathrm{d}t^{k-1&#125;&#125;\Big),\quad k=1,2,\cdots,n.&#36;&#36;</li>
</ul>
<p>We give a simple nature here but do not continue to expand. He is a matrix expression of the analytical guidance formula, and in fact it is easier to reduce to normal functions in a matrix function. For any constant array&#36;A&#36; Yes.</p>
<ul>
<li>&#36;\frac{\mathrm{d&#125;&#125;{\mathrm{d}t}\mathrm{e}^{\mathbf{A}t}=\mathbf{A}\mathrm{e}^{\mathbf{A}t}=\mathrm{e}^{\mathbf{A}t}\mathbf{A}&#36;</li>
<li>&#36;\frac{\mathrm{d&#125;&#125;{\mathrm{d}t}\mathrm{cos}\boldsymbol{A}t=-\boldsymbol{A}(\sin\boldsymbol{A}t)=-(\sin\boldsymbol{A}t)\boldsymbol{A}&#36;</li>
<li>&#36;\frac{\mathrm{d&#125;&#125;{\mathrm{d}t}\mathrm{sin}\boldsymbol{A}t=\boldsymbol{A}(\mathrm{cos}\boldsymbol{A}t)=(\mathrm{cos}\boldsymbol{A}t)\boldsymbol{A}.&#36;</li>
</ul>
<h4>Catalogue function of the matrix</h4>
<p>We start by promoting mathematical analysis4 the concept of a multiple number function as a guide to the vector, and give the matrix function as a definition of a matrix.</p>
<p>Definitions: Establishment&#36;A\in\mathbb{R}^{m\times n},f(A)&#36;As Matrix&#36;A&#36;, that's what it looks like.&#36;m\times n&#36;meta-functions, with numerical functions&#36;f(\mathbf{A})&#36;For Matrix&#36;\mathbf{A}&#36;The wizard is
&#36; \frac{\mathrm{d}fmathrm{A=left=<em>{m\times n}=\begin{bmatrix}\frac{\partial f}{\partial a</em>{11&#125;&#125;&amp;\cdots&amp;\frac{\partial f}{\partial a_{1n&#125;&#125;\\vdots&amp;&amp;\vdots\\frac{\partial f}{\partial a_{m1&#125;&#125;&amp;\cdots&amp;\frac{\partial f}{\partial a_{mn&#125;&#125;\end{bmatrix}.&#36;&#36;
<strong>What we're looking at here is the numerical function of the matrix.&#36;f(A)&#36;, he's not a matrix function, he's a multiple number function&#36;f(A)&#36;It's understandable with the following example.</strong></p>
<p>Set &#36;matbf{x=begin{bmatrix}a&amp;b&amp;c\\d&amp;e&amp;f\end{bmatrix}&#36;  &#36;F(X)=a^2+b^2+c^2+d^2-2e+15F
&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;&#36;&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36; =US&#36;US&#36;US&#36;US&#36;US&#36;US&#36; =&#36;US&#36;US&#36;US&#36; =US&#36;US&#36;US&#36;&#36;&#36;US&#36;US&#36;&#36;US&#36;US&#36; }US = = =US&#36;US&#36;US&#36; = = =US = = =US&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;US&#36;&#36;US&#36;&#36;US&#36;&#36;&#36;&#36;US \&#36; \ \&#36;&#36;&#36;&#36;&#36;US&#36;&#36;&#36; \&#36;&#36; \&#36;&#36;&#36; }&#36;&#36; }&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&amp;\frac{\partial F}{\partial b}&amp;\frac{\partial F}{\partial c}\\frac{\partial F}{\partial d}&amp;\frac{\partial F}{\partial e}&amp;\frac{\partial F}{\partial f}\end{bmatrix}=\begin{bmatrix}2a&amp;&amp;2b&amp;&amp;2c\2d&amp;&amp;-2&amp;&amp;15\end{bmatrix}&#36;&#36;
<strong>No matter how it's calculated,&#36;f(A)&#36;Must be a number function, otherwise it's not natural to get this from the score.</strong></p>
<p>Definitions: Matrixing&#36;F&#36;Yes.&#36;A\in\mathbb{C}^{m\times n}&#36;For Self Variables&#36;p\times q&#36;Matrix, or</p>
<p>&#36;&#36;\boldsymbol{F}(\boldsymbol{A})=\begin{vmatrix}f_{11}\left(\boldsymbol{A}\right)&amp;f_{12}\left(\boldsymbol{A}\right)&amp;\cdots&amp;f_{1q}\left(\boldsymbol{A}\right)\f_{21}\left(\boldsymbol{A}\right)&amp;f_{22}\left(\boldsymbol{A}\right)&amp;\cdots&amp;f_{2q}\left(\boldsymbol{A}\right)\\vdots&amp;\vdots&amp;&amp;\vdots\f_{p1}\left(\boldsymbol{A}\right)&amp;f_{p2}\left(\boldsymbol{A}\right)&amp;\cdots&amp;f_{pq}\left(\boldsymbol{A}\right)\end{vmatrix}<em>}, &#36;
its elements &#36;f_k(\boldsymbol{A})&#36;=a</em>{ij})<em>{m\times n}&#36;的元素为自变量的 &#36;mn&#36; 元函数，则规定矩阵 &#36;F(\boldsymbol{A})&#36;对于矩阵&#36;The guide number for A&#36; is
{\cHFFFFFF}{\cH00FF00} {\cHFFFFFF}{\cH00FF00} {\cHFFFFFF}{\cH00FF00} {\cHFFFFFF}{\cH00FF} {\cHFFFFFF}{\cH00FF00} {\cHFFFFFF}{\cH00FF00} {\cH00FF} {\cHFFFFFF}{\cH00FF00} {\cH00FF00} {\cH00FF00} {\cH00FF00} {\cH00FF00} {\cH00FF00} {\cH00FF} {\cH00FF00} {\cH00FF} {\fH303000cH00h } {\fH00c\boldsymbol} {\fH00H3000} } {\fH30303030303000} } {\cH30303030303000 }</em>{ij&#125;&#125;\Big)<em>{pm\times qn}:=:\begin{bmatrix}\frac{\partial\boldsymbol{F&#125;&#125;{\partial a</em>{11&#125;&#125;&amp;\frac{\partial\boldsymbol{F&#125;&#125;{\partial:a_{12&#125;&#125;&amp;\cdots&amp;\frac{\partial\boldsymbol{F&#125;&#125;{\partial:a_{1n&#125;&#125;\\\frac{\partial\boldsymbol{F&#125;&#125;{\partial a_{21&#125;&#125;&amp;\frac{\partial\boldsymbol{F&#125;&#125;{\partial:a_{22&#125;&#125;&amp;\cdots&amp;\frac{\partial\boldsymbol{F&#125;&#125;{\partial:a_{2n&#125;&#125;\\vdots&amp;\vdots&amp;&amp;\vdots\\\frac{\partial\boldsymbol{F&#125;&#125;{\partial a_{m1&#125;&#125;&amp;\frac{\partial\boldsymbol{F&#125;&#125;{\partial a_{m2&#125;&#125;&amp;\cdots&amp;\frac{\partial\boldsymbol{F&#125;&#125;{\partial:a_{mn&#125;&#125;\end{bmatrix},&#36;&#36;</p>
<p>of which
{\cHFFFFFF}{\cH00FF00} That's a good idea.&amp;\frac{\partial f_{12&#125;&#125;{\partial a_{ij&#125;&#125;&amp;\cdots&amp;\frac{\partial f_{1q&#125;&#125;{\partial a_{ij&#125;&#125;\\frac{\partial f_{21&#125;&#125;{\partial a_{ij&#125;&#125;&amp;\frac{\partial f_{22&#125;&#125;{\partial a_{ij&#125;&#125;&amp;\cdots&amp;\frac{\partial f_{2q&#125;&#125;{\partial a_{ij&#125;&#125;\\vdots&amp;\vdots&amp;\vdots\\frac{\partial f_{p1&#125;&#125;{\partial a_{ij&#125;&#125;&amp;\frac{\partial f_{p2&#125;&#125;{\partial a_{ij&#125;&#125;&amp;\cdots&amp;\frac{\partial f_{pq&#125;&#125;{\partial a_{ij&#125;&#125;&amp;\end{bmatrix},\begin{aligned}i&amp;=1,2,\cdots,m,\j&amp;=1,2,\cdots, n.end{aligned}
This definition is very easy to understand.<strong>It's a multi-layered package, but it's also covering all the previous studies.</strong></p>
<h4>All-minus of the matrix</h4>
<p>Definitions: Arrays &#36;F=(f {ij})<em>{m\times n}&#36;,则规定矩阵&#36;Full split of &#36; F.
&#36; \\mathrm{d}\bardsymbol{f</em>{ij})_{m\times n}.&#36;&#36;</p>
<p>The full calibration of the matrix does not involve a question of guidance to the matrix, which is very natural to calculate.</p>
<p>The entire calibration of the matrix has the following operating properties.</p>
<ul>
<li>&#36;\operatorname{d}(\boldsymbol{F}\pm\boldsymbol{G})=\operatorname{d}\boldsymbol{F}\pm\operatorname{d}\boldsymbol{G};&#36;</li>
<li>&#36;\operatorname{d}(k\boldsymbol{F})=k\operatorname{d}\boldsymbol{F};&#36;</li>
<li>&#36;\text{当 }A\text{ 是常量矩阵时 },\mathrm{d}A=0&#36;</li>
<li>&#36;\operatorname{d}(\boldsymbol{X}^{\mathrm{T&#125;&#125;)=(\operatorname{d}\boldsymbol{X})^{\mathrm{T&#125;&#125;;&#36;</li>
<li>&#36;\operatorname{d}(\operatorname{tr}\boldsymbol{X})=\operatorname{tr}(\operatorname{d}\boldsymbol{X})&#36;</li>
</ul>
<p>Theorem: Set &#36;x=(x_{1},x_{2},\cdots,x_{n})^{\mathrm{T&#125;&#125;&#36;,format &#36;F=(f ij})<em>{s\times m}&#36;,其中 &#36;f</em>{ij}&#36; 都是&#36;x i&#36; 's real function, then there are full differentials for matrix functions
&#36;&#36;\mathrm{d}\boldsymbol{F}=\sum_{i=1}^n\frac{\partial\boldsymbol{F&#125;&#125;{\partial x_i}\mathrm{d}x_i.&#36;&#36;
We can give you further details of the nature of the matrix.</p>
<ul>
<li>Set &#36;A=BC&#36;  then &#36;dA=(dB)C=BdC&#36;</li>
<li>Set &#36;A=A_1A_{2}...A_n&#36; \mathrm{d}\bardsymbol{A}\mathrm{d}\bardsymbol{A}<em>{1}\right) \boldsymbol{A}</em>{2} \cdots \boldsymbol{A}<em>{r}+\boldsymbol{A}</em>{1}\left(\mathrm{<del>d} \boldsymbol{A}<em>{2}\right) \boldsymbol{A}</em>{3} \cdots \boldsymbol{A}<em>{r}+\boldsymbol{A}</em>{1} \cdots \boldsymbol{A}_{r-1}\left(\mathrm{</del>d} \boldsymbol{A}_{r}\right) .&#36;</li>
<li>&#36;d(\alpha^Tx)=\alpha^Tdx=(dx)^T\alpha&#36;</li>
<li>&#36;d(Ax)=Adx&#36;</li>
<li>&#36;d(xA^Tx)=x^T(A^T+A)dx&#36;</li>
</ul>
<h4>Array points</h4>
<p>Definition: Set a function matrix
&#36;&#36;\bardsymbol{A}(t)=\left(\begin{array}{cc}
a {11}(t) &amp; a_{12}(t) &amp; \cdots &amp; a_{1 n}(t) \
a_{21}(t) &amp; a_{22}(t) &amp; \cdots &amp; a_{2 n}(t) \
\vdots &amp; \vdots &amp; &amp; \vdots \
a_{n 1}(t) &amp; a_{n 2}(t) &amp; \cdots &amp; a_{n n}(t)
\end{array}\right)&#36;&#36;
我们定义
&#36;&#36;\begin{array}{c}
\int \boldsymbol{A}(t) \mathrm{d} t=\left(\begin{array}{cccc}
\int a_{11}(t) \mathrm{d} t &amp; \int a_{12}(t) \mathrm{d} t &amp; \cdots &amp; \int a_{1 n}(t) \mathrm{d} t \
\vdots &amp; \vdots &amp; &amp; \vdots \
\int a_{n 1}(t) \mathrm{d} t &amp; \int a_{n 2}(t) \mathrm{d} t &amp; \cdots &amp; \int a_{n n}(t) \mathrm{d} t
\end{array}\right), \
\int_{a}^{b} \boldsymbol{A}(t) \mathrm{d} t=\left(\begin{array}{cccc}
\int_{a}^{b} a_{11}(t) \mathrm{d} t &amp; \int_{a}^{b} a_{12}(t) \mathrm{d} t &amp; \cdots &amp; \int_{a}^{b} a_{1 n}(t) \mathrm{d} t \
\vdots &amp; \vdots &amp; &amp; \vdots \
\int_{a}^{b} a_{n 1}(t) \mathrm{d} t &amp; \int_{a}^{b} a_{n 2}(t) \mathrm{d} t &amp; \cdots &amp; \int_{a}^{b} a_{n n}(t) \mathrm{d} t
\end{array}\right),
\end{array}&#36;&#36;</p>
<p>This is obviously a hypothetical score.  &#36;\int a_{i j}(t) \mathrm{d} t(i, j=1,2, \cdots, n)&#36;  It exists.</p>
