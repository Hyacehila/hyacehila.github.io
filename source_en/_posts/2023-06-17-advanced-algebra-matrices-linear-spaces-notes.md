---
title: 'Advanced Algebra: Matrices and Linear Spaces'
title_zh: 高等代数：矩阵和线性空间
date: 2023-06-17 23:01:45 +0800
categories:
- Mathematics
- Algebra & Matrix Theory
tags:
- Linear Algebra
author: Hyacehila
mathjax: true
hidden: true
excerpt: Covers matrix operations, elementary transformations, rank, inverse matrices, linear spaces, and quadratic forms.
description: Covers matrix operations, elementary transformations, rank, inverse matrices, linear spaces, and quadratic forms.
excerpt_zh: 整理矩阵运算、初等变换、矩阵的秩、逆矩阵、线性空间和二次型等内容。
permalink: /blog/2023/06/17/advanced-algebra-matrices-linear-spaces-notes/
lang: en
translation_key: 2023-06-17-advanced-algebra-matrices-linear-spaces-notes
translation_status: machine
translation_source_hash: 941f4e93d2ad50f2222ea559cfc284e61e504d383fdb8a97affd096766e55649
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<h2>Matrix Foundation</h2>
<h3>Query Out</h3>
<p>We've actually been using a lot of matrix knowledge, and they're all leading us to the matrix.</p>
<ul>
<li>The primary shift of the Goss negative is the primary shift of the matrix.</li>
<li>Multiple vectors naturally form a matrix.</li>
<li>The vector group introduced the concept of thorium, which is the matrix.</li>
<li>The condition for the equation group to solve is that the equation group match the matrix with the magnifying matrix.</li>
</ul>
<p>In fact, the matrix still works in many areas.</p>
<p><strong>Linear transformation</strong>
&#36;&#36;\left{\begin{matrix}
 x=x^{\prime}\cos\theta-y\sin\theta\
 y=x^{\prime}\sin\theta+y^{\prime}\cos\theta
\end{matrix}\right.&#36;&#36;
The transformation can be expressed in the matrix below
&#36; \begin{matrix}\cos\theta&amp;-\sin\theta\\sin\theta&amp;\cos\theta\end{pmatrix}&#36;&#36;</p>
<p><strong>Conic Conic</strong>
&#36;&#36;ax^2+2bx+cy^2+2dx+2ey+f=0&#36;&#36;
You can use the matrix below.
&#36; \begin{matrix}a&amp;b&amp;d\b&amp;c&amp;e\d&amp;e&amp;f\end{pmatrix}&#36;&#36;</p>
<p><strong>Multiple Counterparts</strong>
&#36;s\times n&#36;A correspondence can be expressed in the matrix below.
&#36; \begin{matrix}
a {11}&amp;a_{12}  &amp;\cdots  &amp;a_{1n} \
  \vdots &amp;  &amp;  &amp;\vdots \
  a_{s1}&amp;a_{s2}  &amp;\cdots   &amp;a_{sn}
\end{pmatrix}&#36;&#36;</p>
<h3>Matrix Operations</h3>
<h4>Matrix Equal</h4>
<p>Definitions: Matrixs with equal numbers of rows and columns are referred to as matrices of the same type</p>
<p>Definition: The equivalence of the matrix means that all the elements at the corresponding position are the same</p>
<p>Definition: A matrix that, if the number of rows and columns is equal, is called a square array</p>
<h4>Matrix Plus</h4>
<p>Definitions: Only a single matrix can be added to it; the result is the quantum of the corresponding position Add</p>
<p>Definitions: A matrix with zero elements is referred to as a matrix with zero.</p>
<p>Definition: All elements are preceded by negative numbers, known as the negative matrix of the original matrix, as follows:&#36;-A&#36;</p>
<p>Nature:</p>
<ul>
<li>Union rate:&#36;A+B+C=(A+B)+C=A+(B+C)&#36;</li>
<li>Swap:&#36;A+B=B+A&#36;</li>
<li>&#36;A+0=A&#36;</li>
<li>&#36;A+(-A)=0&#36;</li>
<li>&#36;rank(A+B)\leq rank(A)+rank(B)&#36;</li>
</ul>
<h4>Matrix Multiplication</h4>
<p>Definition: One array multiplied by one number&#36;k&#36; It's every element.&#36;k&#36; As&#36;kA&#36;</p>
<p>Nature</p>
<ul>
<li>&#36;(k+l)A=kA+lA&#36;</li>
<li>&#36;klA=k(lA)&#36;</li>
<li>&#36;k(A+B)=kA+kB&#36;</li>
<li>&#36;k(AB)=(kA)B=A(kB)&#36;</li>
</ul>
<h4>Matrix Multiplication</h4>
<p>Only&#36;A_{s\times n}&#36; and &#36;B_{n\times m}&#36; Type arrays can be multiplied. &#36;C_{s\times m}&#36; Other forms of matrix do not define multiplication.</p>
<p>Let's remember.&#36;C_{s\times m}&#36;The element in any position is&#36;c_{ij}&#36;  So there is.
&#36;&#36;c_{ij}=\sum_{l=1,k=1}^{l=n,k=n}a_{il}b_{kj}&#36;&#36;
Which means...&#36;A&#36;No. No.&#36;i&#36;Line and&#36;B&#36;No. No.&#36;j&#36;Summize the elements of the column 's corresponding position</p>
<p>The matrix multiplication provides a new expression for the linear equation group.&#36;A&#36;As coefficient matrix &#36;x&#36;As a variable column vector &#36;B&#36;is the equation constant column vector
&#36;&#36;Ax=B&#36;&#36;</p>
<p>The matrix multiplication has a pattern below.</p>
<ul>
<li>It doesn't satisfy the rules of exchange, because it doesn't always work.</li>
<li>Unsatisfactory pass rate</li>
<li>Satisfaction factor &#36;ABC=(AB)C=A(BC)&#36;</li>
</ul>
<h4>Matrix</h4>
<p>Definition: The matrix with a main diagonal of 1 and a full zero of the remaining position elements is referred to as the unit matrix as&#36;E_{n}&#36;Or...&#36;I_n&#36;  &#36;n&#36;It's the order of the square.
&#36; \begin{matrix}1&amp;0&amp;0&amp;0\0&amp;1&amp;0&amp;0\0&amp;0&amp;1&amp;0\0&amp;0&amp;0&amp;1\end{pmatrix}&#36;&#36;</p>
<p>For unit matrices, it's easy to know.</p>
<ul>
<li>&#36;A_{s\times n} E_n=A_{s\times n}&#36;</li>
<li>&#36;E_s A_{s\times n}=A_{s\times n}&#36;</li>
</ul>
<p>Definitions: For the square array&#36;A&#36; Let's put&#36;k&#36;individual&#36;A&#36;Multiplication results are called squares.&#36;A^k&#36;</p>
<p>For Fang, it's easy to know</p>
<ul>
<li>&#36;A^kA^l=A^{k+l}&#36;</li>
<li>&#36;(A^k)^l=A^{kl}&#36;</li>
</ul>
<p>Definition: We multiply the unit matrix by one number&#36;k&#36;The matrix is called the quantitative matrix.</p>
<p>It's easy to know, unit matrices, quantity matrices, square arrays. So if&#36;AB=BA&#36;  and&#36;B&#36; It's any matrix, then.&#36;A&#36;It's a quantitative matrix.</p>
<p>With the concept of a matrix, we can link the matrix to multiple forms.</p>
<p>Definitions: Multiple formulas in the form of square arrays, where&#36;A&#36;It's Fong.
&#36;&#36;a_nA^n+\cdots+aA+E=f(A)&#36;&#36;
When?&#36;f(A)=0&#36;Sometimes, it's called a square formation.&#36;A&#36;Zero Multiples</p>
<p>Definitions: The following forms are arrays.&#36;B_i&#36;Yes.&#36;n\times n&#36;Square &#36;\lambda&#36;Yes.
&#36;&#36;\lambda^mB_0+\lambda^{m-1}B_1+\cdots+B_n&#36;&#36;
&#36;n&#36;It's called the steps.&#36;m&#36;Number</p>
<h4>Matrix Transfer</h4>
<p>Definition: The change of the matrix is a matrix line exchange process,&#36;k&#36;Line transformation&#36;k&#36;Row, turn&#36;n\times s&#36;The matrix is converted.&#36;s\times n&#36;Matrix with Symbol&#36;A^T&#36;Or...&#39;Other Organiser</p>
<p>The conversion has the following properties:</p>
<ul>
<li>&#36;(A^T)^{T}=A&#36;</li>
<li>&#36;(A+B)^T=A^T+B^T&#36;</li>
<li>&#36;(kA)^T=kA^T&#36;</li>
<li>&#36;(AB)^T=B^TA^T&#36;</li>
<li>&#36;|A|=|A^T|&#36;</li>
</ul>
<p>The reset matrix also has the concept of conversion, which requires the convergence of elements based on the actual matrix conversion.</p>
<h3>The Back of the Matrix</h3>
<p>The counter-argument of the matrix is limited to&#36;n&#36;The formation of the steps goes on.</p>
<p>We know:&#36;AE=EA&#36;  Unit Matrix&#36;E&#36;It's actually a concept similar to 1.</p>
<p>In elementary math theory, we still have the concept of the penultimate, which is...&#36;a\times \frac{1}{a}=1&#36;  Is there a concept in that matrix, that's the reverse of the matrix?</p>
<p>Definitions: Yes&#36;n&#36;Array&#36;A&#36; Existing Matrix&#36;B&#36;Make &#36;AB=E&#36; of which &#36;E&#36;Yes.&#36;n&#36;Step formation, we call it.&#36;B&#36;Yes.&#36;A&#36;The counter-argument, recorded as&#36;A^{-1}&#36;  And whatever.&#36;A&#36;Corresponding&#36;B&#36;The only one.</p>
<p>Then we have two important issues in the back of the matrix.</p>
<ul>
<li>When does the matrix exist?</li>
<li>What is the common method of calculating the reverse of the matrix?</li>
</ul>
<p>Definitions:&#36;A_{ij}&#36;It's a matrix.&#36;A&#36;Elements&#36;a_{ij}&#36;algebra residual, same<a href="/en/blog/2023/03/17/advanced-algebra-foundations-notes/">The normal extension of the # type in the line# in the upper algebra 1 base</a></p>
<p>Definitions:&#36;A^{\star}&#36;It's a matrix.&#36;A&#36;the accompanying matrix, if any
&#36;A^&amp;\cdots&amp;A_{1n}\A_{n1}&amp;\cdots&amp;A_{nn}\end{pmatrix}&#36;&#36;</p>
<p>It's easy to get a definition. &#36;AA^{\star}=dE&#36;  of which &#36;d=|A|&#36;  So Matrix&#36;A&#36;The reverse matrix is&#36;\frac{1}{d}A^{\star}&#36;  Just...&#36;|A|\ne0&#36;(A matrix)&#36;A&#36;Non-degradable)</p>
<p>With regard to the reverse and transposition of the matrix, the following conclusions
&#36;&#36;AB可逆\rightarrow AB\quad A^T可逆且(A^T)^{-1}=(A^{-1})^T\quad(AB)^{-1}=B^{-1}A^{-1}&#36;&#36;
<strong>A definition-based calculation of the backsliding of the matrix is still very cumbersome, and a better method of calculation is provided in the section of this paper entitled "The Matrix Foundation #Primary Transformation and the Initial Matrix"</strong></p>
<h3>Block Matrix</h3>
<p>We're here.<a href="/en/blog/2023/03/17/advanced-algebra-foundations-notes/">Algebra 1 byr # block byr in the algebra base</a>It gives us the full application of the normal one. Now let's extend the division concept to the matrix.</p>
<h4>Split and Basic Operations</h4>
<p>In dealing with the operation of a high-level matrix, we split it into small low-level matrices to facilitate the operation. There are no rules in the split itself, but we want to make sure that the original calculations are still in place, as illustrated by one example.</p>
<p>&#36;&#36;B=\begin{pmatrix}
  1&amp;  0&amp;3  &amp;2 \
  -1&amp;2  &amp;0  &amp;1 \
  1&amp;0  &amp;4  &amp;1 \
  -1&amp;  -1&amp;  2&amp;0
\end{pmatrix}
=\begin{pmatrix}
  B_{11}&amp;B_{12} \
  B_{21}&amp;B_{22}
\end{pmatrix}&#36;&#36;
&#36;&#36;A=\begin{pmatrix}
  1&amp;  0&amp;0  &amp;0 \
  0&amp;1  &amp;0  &amp;0 \
  -1&amp;2  &amp;1  &amp;0 \
  1&amp;  1&amp;  0&amp;1
\end{pmatrix}
=\begin{pmatrix}
  E_{2}&amp;A_{0} \
  A_{1}&amp;E_{2}
\end{pmatrix}&#36;&#36;
那么有
&#36;&#36;AB=\begin{pmatrix}
 B_{11} &amp;B_{12} \
  A_{1}B_{11}+B_{21}&amp;I'm not sure if you're going to be able to do this.
Can't you see?
As long as we keep our luck, the small matrix division is random.</p>
<h4>Block Matrix Conversion</h4>
<p>Here's how the partition matrix is transferred.
&#36; \begin{matrix}A 1&amp;A_2\A_3&amp;A_4\end{pmatrix}^T=\begin{pmatrix}A_1^T&amp;A_3^T\A_2^T&amp;A 4 ^end{matrix}
We just need to...</p>
<ul>
<li>Switch as a normal matrix</li>
<li>Then we'll switch every part of the matrix.</li>
</ul>
<h4>Block Matrix Reverse</h4>
<p><strong>Opposite Matrix</strong>
&#36;&#36;\begin{pmatrix}A &amp; 0\ 0 &amp; B\end{pmatrix}^{-1}=\begin{pmatrix}A^{-1} &amp; 0\ 0 &amp; B^{-1}\end{pmatrix}&#36;&#36;</p>
<p><strong>Triangular Matrix</strong>
&#36;&#36;\begin{pmatrix}A &amp; 0\ C &amp; B\end{pmatrix}\begin{pmatrix}x_1 &amp; x_2\ x_3 &amp; x_4\end{pmatrix}=\begin{pmatrix}E_1 &amp; 0\ 0 &amp; E_2\end{pmatrix}&#36;&#36;
解方程可以得到
&#36;&#36;\begin{pmatrix}A^{-1}&amp;0\-B^{-1}CA^{-1}&amp;B^{-1}\end{pmatrix}&#36;&#36;</p>
<p>If it's a Triangular Matrix, then it should be converted to the next Triangular Matrix and then apply the formula, which is,
&#36;&#36;D^{-1}=((D^T)^{-1})^T&#36;&#36;
One of which turns his countdown.</p>
<h3>Primary transformation and primary matrix</h3>
<p>Definitions:&#36;E&#36;The matrix generated by a primary transformation is the primary matrix, which undergoes three primary transformations (plus, swap, multiplication)&#36;E&#36;You can generate all primary matrices.</p>
<p>Introduction: In the case of Africa&#36;E&#36;Yes.&#36;s\times n&#36;The matrix changes the first line, with an equal value of one to the left&#36;s\times s&#36;It's the same thing that changes.&#36;E&#36;Generated primary matrix; for non-&#36;E&#36;Yes.&#36;s\times n&#36;The matrix converts the primary column at an equal value of one to the right&#36;n\times n&#36;It's the same thing that changes.&#36;E&#36;Generates a primary matrix.</p>
<p>Definitions: If&#36;A&#36;It's possible.&#36;B&#36;They're created by primary change.<strong>Equities</strong>, this matrix has an equal price.<strong>Important relation of the matrix - Equivalence</strong>）</p>
<ul>
<li>Self-reverse</li>
<li>Symmetry</li>
<li>Passivity</li>
</ul>
<p>Theorem: Any one&#36;s\times n&#36;Matrix&#36;A&#36;Both and&amp;0\0&amp;== sync, corrected by elderman == @elder man</p>
<p>We can naturally make two inferences based on the definition.</p>
<ul>
<li>If&#36;A,B&#36;It's the same price. There must be.&#36;A=P_1P_2\cdots P_nBQ_1Q_2\cdots Q_m&#36; Of which&#36;P,Q&#36;It's a column shift matrix.</li>
<li>If the matrix&#36;A&#36;It's reversible. It's equal to a matrix of the same unit.&#36;E&#36;Which means...&#36;A=P_1P_2\cdots P_nEQ_1Q_2\cdots Q_m&#36; So the reversible matrix can change to the size of several primary matrices.</li>
</ul>
<p>Depending on the relevant nature of the primary transformation, it is easy to give a method for seeking the "reverse" part of the matrix, with only a series of primary (column) changes</p>
<p>Other Organiser&#36;A&#36;It's written below.&#36;E&#36;Yes and&#36;A&#36;Same Square
&#36;&#36;(A\mid E)&#36;&#36;
Use primary line changes only. Put the one on the left of the matrix.&#36;A&#36;Convert&#36;E&#36; At this point, the result of the transformation of the matrix to the right is&#36;A^{-1}&#36; This method can be extended to a wide range of reverse uses.<a href="/en/blog/2024/12/15/matrix-analysis-notes/">Derogation in matrix analysis &#36;A -&#36;</a></p>
<h3>Primary transformation and partition multiplication</h3>
<p>Primary transformation of sub-unit matrix, equivalent to primary transformation of logarithm</p>
<p>The primary transformation of the three unit matrix blocks has
&#36;begin{matrix}Em&amp;0\0&amp;En\end{pmatrix}\rightarrow\begin{pmatrix}0&amp;En\Em&amp;0\end{pmatrix}\begin{pmatrix}D&amp;0\0&amp;En\end{pmatrix}\begin{pmatrix}Em&amp;0\D&amp;E_n\end{pmatrix}&#36;&#36;</p>
<p>The primary transformation of the non-unit matrix blocks is still satisfied: the line transformation is equal to the left multiplied by the primary matrix, and the column variant is equal to the right multiplied by the primary matrix.</p>
<p>♪ For like
&#36; \begin{matrix}
A&amp;B \
  C&amp;D
\end{pmatrix}&#36;&#36;
的倍加行初等变换有
&#36;&#36;\begin{pmatrix}Em&amp;O\P&amp;En\end{pmatrix}\times\begin{pmatrix}A&amp;B\C&amp;D\end{pmatrix}=\begin{pmatrix}A&amp;B\C+PA&amp;D+PB\end{matrix}
This multiply calculation will allow&#36;C+PA=0&#36; Which means...&#36;PA=-C&#36; When?&#36;A&#36;In retroverted times, yes.&#36;P=-CA^{-1}&#36; So you can build an empty angle and get a triangle that's easier to deduce, and that's the technique that's used in the segment matrix.<strong>Hole.</strong></p>
<p>Let's give a simple example of what's going on.&amp;0\ C&amp;D\end{pmatrix}&#36;  其中 &#36;A,D&#36;可逆 求 &#36;T^{-1}&#36;</p>
<p>We put holes in the lower left corner.&#36;C&#36;Turned into zero.
&#36; \begin{matrix}E m&amp;O\-A^{-1}C&amp;E_n\end{pmatrix}\begin{pmatrix}A&amp;O\C&amp;D\end{pmatrix}=\begin{pmatrix}A&amp;O\O&amp;D\end{pmatrix}&#36;&#36;
同时对两边求逆有
&#36;&#36;\begin{pmatrix}
 A^{-1} &amp;O \
 O &amp;D^{-1}
\end{pmatrix}=T^{-1}\times B^{-1}&#36;&#36;
所以
&#36;&#36;\begin{pmatrix}
 A^{-1} &amp;O \
 O &amp;D^{-1}
\end{pmatrix}\times B=T^{-1} &#36;&#36;</p>
<h3>Matrix nature supplement</h3>
<h4>Operational nature</h4>
<p>We can give the following nature for the matrix's operation and calibration.</p>
<ul>
<li><p>&#36;r(AB)\geq r(A)+r(B)-n&#36; of which &#36;A_{m\times n},A_{n\times s}&#36;</p>
</li>
<li><p>&#36;r(AB)\leq\min(r(A),r(B))&#36;</p>
</li>
<li><p>&#36;r(A)=r(A^{T})=r(AA^{T})&#36;</p>
</li>
<li><p>&#36;r(ABC)\geq r(AB)+r(BC)-r(B)&#36;</p>
</li>
<li><p>&#36;r(A)=r(PA)=r(AQ)=r(PAQ)&#36; If &#36;P,Q&#36;Reversible</p>
</li>
<li></li>
</ul>
<p>&#36;&#36;r(\begin{pmatrix}
 M &amp;O \
  K&amp;N
\end{pmatrix})\geq r(\begin{pmatrix}
 M &amp;O \
  O&amp;N
\end{pmatrix})=r(M)+r(N)&#36;&#36;
*
&#36;&#36;r(A^{\star})=\left{\begin{matrix}
  n&amp; r(A)=n\
  1&amp; r(A)=n-1\
  0&amp;r(A)&lt;n-1
\end{matrix}\right.&#36;&#36;</p>
<h4>Quantified.</h4>
<p>For the Quest Array&#36;A^{k}&#36; We have a few common ways of doing this.</p>
<ul>
<li>Calculating 3 to 5 steps, summing up results</li>
<li>The unit matrix square meets &#36;\begin{matrix}\lambda&amp;0&amp;0\0&amp;\lambda&amp;0\0&amp;0&amp;\lambda\end{pmatrix}^n=\begin{pmatrix}\lambda^n&amp;0&amp;0\0&amp;\lambda^n&amp;0\0&amp;0&amp;\lambda^n\end{pmatrix}&#36;</li>
<li>If the matrix happens to be the primary matrix, then it's possible that the constant search is to make some kind of transformation and to use a transformational approach to solve it.</li>
<li>Double-discrete the original matrix arc.&#36;(A+B)^n&#36; , which applies to&#36;A,B&#36;One of them is zero, thus simplifying the operation.</li>
</ul>
<p>We'll have another way of looking for a matrix in the next presentation.</p>
<h4>Swapable Matrix</h4>
<p>Definition: For two squares&#36;A,B&#36;  If there is. &#36;AB=BA&#36; It says the two are interchangeable.</p>
<p>Theoretically: the tradable matrix of the diagonal matrix is the angle matrix</p>
<p><em>The method of proof is to be determined by force of the matrix coefficient</em></p>
<h2>Secondary</h2>
<h3>Introduction of Secondary</h3>
<p>This is our chapter.<strong>The basic research object is&#36;n&#36;An equation of binary</strong>, the core principle is that everything can be converted from a deliberate conic to a standard pattern in the form of coordinates.</p>
<p>Definition: We call the equation that meets the following form a secondary and an equation expression
&#36;f (x 1,x 2...x n)&lt;j\le n}a_{ij}x_ix_j&#36;&#36;</p>
<p>Definitions: We have rearranged the previous secondary form into a matrix called the matrix of the secondary.
&#36; \begin{matrix}
x 1&amp;x_2  &amp;\cdots   &amp;x_n
\end{pmatrix} \begin{pmatrix}
  a_{11}&amp;a_{12}  &amp;\cdots   &amp;a_{1n} \
  a_{21}&amp;a_{22}  &amp;\cdots  &amp;a_{2n} \
  \vdots &amp; \vdots &amp;\cdots   &amp;\vdots \
  a_{n1}&amp;a_{n2}  &amp;\cdots  &amp;I don't know.
\begin{matrix}
What?
x 2\
\vdots
x n
Can't you see?
The middle matrix is called the coefficient matrix. It's easy to see, he's satisfied.&#36;a_{ij}=a_{ji}&#36; A symmetric matrix</p>
<p>We're considering the vector.&#36;X&#36; Coefficient Matrix as&#36;A&#36;  A matrix of a secondary type simply indicates &#36;X^TAX&#36;</p>
<p>Definitions: In practice, the first multi-dimensional definition could still be simplified to
&#36;&#36;\sum_{i=1}^{n}\sum_{j=1}^{n}a_{ij}x_{i}x_{j}&#36;&#36;</p>
<h3>Linear replacement</h3>
<p>Definition: When a secondary cross-item coefficient is 0, we call it the standard model of the secondary and the corresponding coefficient matrix is a diagonal matrix</p>
<p>Definitions: For a secondary type&#36;f(x_1,x_2...x_{n})&#36; Define Replace Below
&#36;&#36;00begin{cases}x=a=a a cdots+a n}&amp;I'm sorry.
It's a linear set of factors that will be used to replace&#36;c_{ij}&#36; As&#36;C&#36;  If&#36;|C|\ne0&#36; They are called non-degradable linear substitutions.</p>
<p>Linear replacement can also be expressed as a matrix
&#36; \begin{pmatrix}x&amp;\&amp;I'm sorry.
Which means...&#36;X=CY&#36;  of which &#36;Y&#36; It's a variable coefficient matrix.</p>
<p>Theorem: Use of definitions to prove that the original coefficient matrix is&#36;A&#36;Secondary type&#36;X^TAX&#36; Linear Replace &#36;X=CY&#36; The result is still a secondary type. &#36;Y^T(C^TAC)Y&#36; Its coefficient matrix is &#36;C^TAC&#36;</p>
<p>Definitions: If Matrix&#36;A,B,C&#36; of which&#36;C&#36;Reversible satisfaction relationship &#36;B=C^TAC&#36;  Call Matrix&#36;A,B&#36; The contract,<strong>The contractual relationship is also an important matrix.</strong>He's satisfied.</p>
<ul>
<li>Self-reverse</li>
<li>Symmetry</li>
<li>Passivity</li>
</ul>
<h3>Standardized</h3>
<p><strong>Theorem: Any sub-type can be converted to a standard by non-degradable linear substitution</strong> The process of proving the theorem is the search for the standard.</p>
<h4>Swap</h4>
<p><strong>When Secondary contains square</strong></p>
<p>For example:&#36;f=x_{1}^{2}-3x_{2}^{2}-2x_{1}x_{2}+2x_{1}x_{3}-6x_{2}x_{3}&#36;</p>
<p>Let's start with everything.&#36;x_1&#36; Get together, get square.&#36;x_1&#36;There's something else you can add.&#36;x_1^2-2x_1x_2+2x_1x_3&#36;  We can do this.
&#36;&#36;(x_1-(x_2-x_3))^2&#36;&#36;
As for missing&#36;(x_2-x_3)^2&#36; We can lose it again.</p>
<p>And all of it.&#36;x_2&#36;And then you get together, and you repeat the process ahead until you put all the items together into squares, and you get a result that's all square, like,
&#36;&#36;y_1^2+y_2^2+\cdots+y_n^2&#36;&#36;
of which&#36;y_i&#36;It's Ham.&#36;x_i&#36;Multiform. The final exchange will be replaced.</p>
<p><strong>When without squares</strong></p>
<p>For example:&#36;f=2x_1x_2+2x_1x_3-6x_2x_3&#36;</p>
<p>At this point, we need an extra step.
&#36;&#36;\left{\begin{matrix}
 x_1=y_1+y_2\
 x_2=y_1-y_2\
x_{3}=y_{3}
\end{matrix}\right.&#36;&#36;
It's now the result of a double line.</p>
<h4>Contract conversion method</h4>
<p>We can transform the original matrix into a standard model by changing contracts and recording our variations.</p>
<p><strong>Contract changes require that all primary changes to rows be repeated in a row, which is at the heart of such changes</strong></p>
<p>Write original matrix as
&#36;&#36;\begin{pmatrix}
 A\
E
\end{pmatrix}&#36;&#36;
Will&#36;A&#36;The matrix is converted to a diagonal matrix.&#36;E&#36;It's ours.&#36;C&#36;  At this point,&#36;C^TAC&#36; That's what we're looking for.</p>
<p>There are ways to do it.</p>
<ol>
<li>When?&#36;a_{11}\ne0&#36;  Chemical &#36;a_{i1},a_{1i}&#36;For 0 to lower array processing</li>
<li>When?&#36;a_{11}=0&#36; And now... &#36;a_{ii}\ne0&#36; Then use the exchange.&#36;a_{11}\ne0&#36; Turn back one.</li>
<li>When?&#36;a_{11}=0&#36; And now... &#36;a_{ii}=0&#36;  Take advantage of it.&#36;a_{11}\ne0&#36; Turn back one.</li>
<li>Once you have finished the downgrade, repeat this step to deal with the lower array until the diagonal.</li>
</ol>
<p><strong>The standard type is not the only one, so the answer is only a reference and requires self-validation of the accuracy of the coefficient matrix</strong></p>
<h3>Unique and normative</h3>
<p>We're not hard to find in the front. The same double standard is not the only one, but they contract each other, so they're the same. Therefore, the same number of square items in a standard type where the median coefficient is not zero is not related to the non-degradable linear replacement. So that's the core measure of second-class uniqueness.</p>
<p><strong>On a complex field</strong> A standard model is the form below
&#36;&#36;d_1y_1^2+d_2y_2^2+\cdots+d_ry_r^2\quad d_i\neq0&#36;&#36;
We'll take it.&#36;y_{r}=\frac{1}{\sqrt{d_r&#125;&#125;z_{r}&#36;, with the permission of negative numbers in root numbers, which can be reduced to a simple secondary type
&#36;&#36;z_1^2+z_2^2+\cdots+z_r^2&#36;&#36;</p>
<p>That means:<strong>The two matrix contracts have the same price, and the plural domain has the same form of determination.</strong></p>
<p>We call it the standard model of a single coefficient. <strong>Normative</strong> The norm in the plural domain is unique</p>
<p><strong>On Real Area</strong> We can do the same thing, but we can't turn the negatives into positives, so the secondarys are simplified.
&#36;&#36;z_1^2+z_2^2+\cdots+z_p^2-z_{p+1}^2\cdots-z_r^2&#36;&#36;
The norm on the real-digit field is being&#36;r&#36;and number of regular items&#36;p&#36;It's also decided that it's unique.</p>
<p>We refer to normative singularity as a secondary inertia.</p>
<h3>Symbol Difference and Positive Stereotyping</h3>
<h4>Sign difference and inertia index</h4>
<p>In this section, we only discuss the sub-species on the real range, and the further derivatives of their normative shape.</p>
<p>Definitions: For normative types&#36;f(x_1.x_2,\cdots,x_n)&#36; square count&#36;p&#36;Called positive inertia index, negative square count&#36;r-p&#36;It's called a negative inertial index. They're bad.&#36;2p-r&#36;It's called a symbol difference.</p>
<p>If we're going to look at the norm, then...</p>
<ul>
<li>Yeah.&#36;n&#36;It's a second class.&#36;n+1&#36;The seed, it's a troupe.&#36;0&#36;Present.&#36;n&#36;</li>
<li>Yeah.&#36;n&#36;Second class. We have.&#36;n+1&#36;It's a specter.&#36;rank-1&#36;A positive or negative inertia.&#36;\frac{n(n+1)}{2}&#36;Contract structure</li>
</ul>
<h4>Resize Second</h4>
<p>Definitions: For secondary types&#36;f(x_1.x_2,\cdots,x_n)&#36; If not zero for any group&#36;(c_1\cdots c_n)&#36; All of them.&gt;&#36;0.00 calls the subtype positive.</p>
<p>Obviously.&#36;f(x_{1}\cdots x_{n})=x_{1}^{2}+x_{2}^{2}+\cdots+x_{n}^{2}&#36;It's settled.</p>
<p>It's not hard to verify.&#36;f(x_1\cdots x_n)=d_1x_1^2+\cdots+d_nx_n^2&#36; Equivalent to &#36;d i&gt;0&#36; 对所有&#36;Set up</p>
<p>Here we give the important theorem:<strong>Linear replacement of secondary non-degradable matrices, or contractual transformation of their coefficient matrix, without altering their positive characterization</strong>  There must be a positive matrix.&#36;p=n&#36; The positive inertia index is the same as the matrix dimensions.</p>
<p>So, there are three ways to judge whether a matrix is positive.</p>
<ul>
<li>Definitions</li>
<li>Standardized</li>
<li>Calculate positive inertia index</li>
</ul>
<h4>A positive matrix</h4>
<p>Definitions: If secondary &#36;XA^TX&#36; It's positive, it's called the coefficient matrix.&#36;A&#36;It's a positive matrix.</p>
<p>Theorem: The matrix of contracts in the unit matrix is fixed</p>
<p>Theoretically: The correct matrix is more than zero in a row, and the corresponding counterproblem is not valid</p>
<p>Now we're proposing a new approach that looks directly at the characterization of the matrix itself.</p>
<p>Definitions:</p>
<ul>
<li>Subform: Formed from several rows of the matrix</li>
<li>Algebra residual and residual: same<a href="/en/blog/2023/03/17/advanced-algebra-foundations-notes/">The normal extension of the # type in the line# in the upper algebra 1 base</a> and <a href="/en/blog/2023/03/17/advanced-algebra-foundations-notes/">High algebra 1 #rapras in the ranks of the algebra base</a>definition</li>
<li>Main sub-form: sub-form with the same row and column numbers &#36;i&#36; The master is not the only one.</li>
<li>Order master:&#36;i&#36;Sequence Masters First&#36;i&#36;Line and Front&#36;i&#36;It's the only one.</li>
</ul>
<p>Theorem: Matrix&#36;A&#36;（&#36;n&#36;The symmetrical array) is the full condition of the positive matrix, with all sequenced masters greater than zero, that is, the sequences are positive.</p>
<p>Inference: Matrix&#36;A&#36;It's settled, and it's accompanied by a matrix.&#36;A^{\star}&#36;Deal.</p>
<h4>Positive parallel concepts</h4>
<p>Definitions: For secondary types&#36;f(x_1.x_2,\cdots,x_n)&#36; If not zero for any group&#36;(c_1\cdots c_n)&#36; All of them.&lt;&#36; 0.00 for this subtype is negative and the coefficient matrix is negative</p>
<p>Definitions: If</p>
<ul>
<li>&#36;f(c_1\cdots c_n)\ge0&#36; It's called semi-positive.</li>
<li>&#36;f(c_1\cdots c_n)\le0&#36; It's called half-negative.</li>
</ul>
<p>Theorem: If a secondary&#36;f(x_1.x_2,\cdots,x_n)&#36;Tunnel, second type&#36;-f(x_1.x_2,\cdots,x_n)&#36; Negative</p>
<p>For a semi-positive matrix, the next issue is equal.</p>
<ul>
<li>&#36;f(x_1.x_2,\cdots,x_n)&#36;Half positive.</li>
<li>Positive inertial index&#36;p&#36; ♪ And Zoo ♪&#36;r&#36; Equal, not equal&#36;n&#36;</li>
<li>Contract model &#36;d_i\ge0&#36;</li>
<li>All mains are non-negative.</li>
<li>Main sequence semi-correction</li>
</ul>
<p><strong>Our core idea in the second chapter is to study the forms of norms hidden behind them, from special to general.</strong></p>
<h2>Linear Space</h2>
<h3>Gather and Map</h3>
<h4>Gather!</h4>
<p>The description of the assembly and mapping<a href="/en/blog/2023/03/16/mathematical-analysis-limits-continuity-notes/">Mathematic analysis 1. The theory of limits and continuity</a>The necessary knowledge is reviewed here as a basis for the subsequent discussions on linear space.</p>
<p>Definitions: Gathering, looking at things as a whole, these things are called elements. &#36;a\in S&#36;</p>
<p>Definitions: A collection that does not contain any elements is called an empty collection, remember&#36;\phi&#36; But we have built a collection.&#36;{\phi}&#36; It's not empty.</p>
<p>Definitions:&#36;当a\in M\text{ 当且仅当 }a\in N\text{时 则称两个集合相等}&#36;</p>
<p>Definitions:&#36;当a\in M\Rightarrow a\in N则称M是子集M\subset N&#36;  It's the same thing.</p>
<p>Definitions:&#36;若M\subset N且N\subset M 则 N=M&#36;</p>
<p>About a collection of handouts and some sort of assembly calculations.</p>
<h4>Map</h4>
<p>Definition: set &#36;M~M&#39;&#36;是两个集合， &#36;M&#36;到&#36;M&#39;&#36;的映射 指一个法则 它使 &#36;Every one of M&#36;
Elements&#36;a&#36; Both.&#36;M^{\prime}&#36;Another element&#36;a^{\prime}&#36;Corresponding, recorded&#36;\sigma(a)=a^{\prime}&#36; or&#36;\sigma: M\rightarrow M^{\prime }&#36; &#36;a^{\prime}&#36;It's made of images. &#36;a&#36;Call it original.</p>
<p>Definitions: Maps from M to M are called mutations</p>
<p>Map,&#36;M&#36;All the elements need to be mapped, but...&#36;M^{\prime}&#36; Not all elements can find the original.</p>
<p>&#36;\sigma,\sigma^{\prime}&#36; The equivalent condition is that the corresponding collection is the same and&#36;\sigma(a)=\sigma^{\prime}(a)&#36;</p>
<p>Definitions: will&#36;a&#36;Show&#36;a&#36;Map&#36;(\sigma(a)=a)&#36;It's called a unit map or a constant equivalent map.&#36;1_{m}&#36;</p>
<p>Definition: the composite of the map is also known as the product of the map, recorded as &#36;\sigma t&#36; I don't know. Operations need to be combined to the right and cannot be exchanged in order of operation</p>
<p>Definition: When&#36;\sigma(M)=M^{\prime}&#36;  That's...&#36;M^{\prime}&#36;All the images can be found. It's called a full shot or a map.</p>
<p>Definition: When&#36;M^{\prime}&#36;Every one of them. The corresponding image is different. So there's no situation that corresponds to two originals called single-shot (1-1).</p>
<p>Definition: a map that is both single and full is double-shot (1-1 corresponding)</p>
<p>Definitions: For a double shot&#36;\sigma: M\to M^{\prime}&#36;  Reverse Map As &#36;\sigma^{-1}:M^{\prime}\to M&#36;</p>
<p>We can easily give the nature of the reverse map.</p>
<ul>
<li>&#36;\sigma\sigma^{-1}=1&#36;</li>
<li>&#36;\sigma^{-1}\sigma=1&#36;</li>
<li>&#36;(\sigma^{-1})^{-1}=\sigma&#36;</li>
</ul>
<p>Theoretically: two double-fired combinations, or double-fired.</p>
<h3>Definition of linear space</h3>
<p>In many previous studies, we have found a continuous function,&#36;n&#36; Many concepts, such as sequenced arrays, matrices, etc. are a collection of multiplications and multiplications. So we try to abstract a model to solve this kind of problem.</p>
<p>Definitions: Yes&#36;V&#36;It's not empty.&#36;P&#36;It's a digital domain, defined&#36;V&#36;An add-and-number multiplication of elements (defined freely), if the sum and multiplication of the addition is still present&#36;V&#36;Medium (composition and number of times closed), and according to the following rules, referred to as linear space</p>
<ul>
<li>&#36;\alpha+\beta=\beta+\alpha&#36;</li>
<li>&#36;(\alpha+\beta)+\gamma=\alpha+(\beta+\gamma)&#36;</li>
<li>Element 0 exists, allowing&#36;\alpha=0+\alpha&#36;</li>
<li>There are negative elements that make&#36;\alpha+\beta=0&#36;</li>
<li>Element 1 exists so that&#36;1 \alpha =\alpha&#36;</li>
<li>&#36;k(l\alpha)=(kl)\alpha&#36;</li>
<li>&#36;(k+l)\alpha=k\alpha+l\alpha&#36;</li>
<li>&#36;k(\alpha+\beta)=k\alpha+k\beta&#36;</li>
</ul>
<p><strong>It's worth emphasizing that the addition and multiplications we have here are self-defined, and elements 0 and 1 are not necessarily traditional natural numbers 0 and 1.</strong></p>
<p>Linear space also becomes vector space, where vectors are broad vectors, as long as elements of linear space are called vectors.</p>
<p>We can naturally give some examples of linear space.</p>
<ul>
<li>Domain&#36;P&#36;The one-dollar polyring. &#36;P[x]&#36;</li>
<li>Domain&#36;P&#36;The one-dollar polyring. &#36;P[x]&#36; But only less than&#36;n&#36;Part</li>
<li>Closed&#36;[a,b]&#36;Continuous function on &#36;C[a,b]&#36;</li>
<li>Domain&#36;P&#36; Top&#36;n&#36;Order array</li>
<li>Domain&#36;P&#36; Some matrix on &#36;P^{m\times n}&#36;</li>
<li>Domain&#36;P&#36;Self</li>
</ul>
<p>For linear space, we can still give theorems and introduce other calculations.</p>
<p>Theorem: Zero elements in a linear space&#36;0&#36;And an element.&#36;\alpha&#36;Negative element&#36;-\alpha&#36;The only one.</p>
<p>Definition: Reductions in linear space can be considered as adding his negative elements</p>
<p>Three conclusions in linear space</p>
<ul>
<li>&#36;0\alpha=0&#36;</li>
<li>&#36;k0=0&#36;</li>
<li>&#36;(-1)\alpha=-\alpha&#36;</li>
</ul>
<p><strong>The same collection, which defines different forms of operation or depends on different digits, creates different linear spaces.</strong>The type of linear space is so varied that we are completely unable to list and understand what kind of space we are studying in the light of the context of the moment.</p>
<h3>Dimensions, Bases and Coordinates</h3>
<p>First, we need to look back at the knowledge points that are learned in the front vector space and to promote them naturally, including linear, linear, linear, etc.</p>
<p>Definitions: If&#36;V&#36;Yes.&#36;n&#36;It's not a linear vector, but there's no more unrelated vectors. &#36;n&#36;If you can find infinity, it's called infinity. Unlimited vectors are not the focus of our research.</p>
<p>Definitions:&#36;n&#36;Found in wiring space&#36;n&#36;An unrelated vector&#36;\varepsilon_1\cdots\varepsilon_n&#36;  They can be linear.&#36;V&#36;Everything in it is called a base.<strong>Base and dimension, but not unique</strong></p>
<p>Definitions:&#36;n&#36;Vector in Wystem&#36;\alpha&#36;It's a base.&#36;\varepsilon_1\cdots\varepsilon_n&#36;  Table&#36;\alpha=a_{1}\varepsilon_{1}+a_{2}\varepsilon_{2}+\cdots+a_{n}\varepsilon_{n}&#36;  So this number?&#36;(a_1,a_2\cdots a_n)&#36; It's the vector.&#36;\alpha&#36;Coordinates under this base  <strong>The coordinates are influenced by both the base and the vector.</strong></p>
<p>As can be seen from the three successive definitions, the dimension and base issues should be addressed together and not separately.</p>
<p>Theoretically: if linear space&#36;V&#36;Yes.&#36;n&#36;A linearly unrelated vector &#36;\alpha_1,\cdots,\alpha_n&#36; and&#36;V&#36;Either vector can be shown by them.&#36;V&#36;Yes.&#36;n&#36;Vi. &#36;\alpha_1,\cdots,\alpha_n&#36;It's a set of foundations.</p>
<p>Theoretically: Bases are not unique, and vector groups of base prices are also fundamental, so there is the concept of a standard base.</p>
<p>It's very important that the foundation reflects the idea of limitless vectors.</p>
<p>Yes. &#36;n&#36; Area&#36;P^n&#36;In the middle, we usually call the base set below standard.
&#36;&#36;\left{\begin{matrix}
 \epsilon_{1}=(1,0,0\cdots,0)\
 \epsilon_{2}=(0,1,0,\cdots0)\
 \vdots\
\epsilon_{n}=(0,0,\cdots,1)
\end{matrix}\right.&#36;&#36;
Any other base vectors, etc. Price</p>
<h3>Base and Coordinate Change</h3>
<p>Now let's look at the system, how to change the base and how the coordinates will change at this point.</p>
<p>We gave the following basic variations.
&#36;&#36;\left{\begin{matrix}
\varepsilon_{1}^{\prime}=a_{11}\varepsilon_{1}+a_{12}\varepsilon_{2}+\cdots+a_{1n}\varepsilon_{n} \
\varepsilon_{2}^{\prime}=a_{21}\varepsilon_{1}+a_{22}\varepsilon_{2}+\cdots+a_{2n}\varepsilon_{n} \
\cdots \
\varepsilon_{n}^{\prime}=a_{n1}\varepsilon_{1}+a_{n2}\varepsilon_{2}+\cdots+a_{nn}\varepsilon_{n}
\end{matrix}\right.&#36;&#36;</p>
<p>Use matrix multiplication to indicate yes
^, \ \ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^
a {11}&amp;a_{12}  &amp; \cdots &amp;a_{1n} \
  a_{21}&amp;a_{22}  &amp; \cdots &amp;a_{2n} \
  \vdots&amp;\vdots  &amp;\ddots  &amp;\vdots \
  a_{n1}&amp;a_{n2}  &amp; \cdots &amp;I don't know.
I'm sorry.
We call it the transition matrix.&#36;A&#36;</p>
<p>And there is.
&#36;&#36;\varepsilon=\varepsilon^{\prime}A^{-1},\varepsilon^{\prime}=\varepsilon A&#36;&#36;
That's the formula for the base shift.</p>
<p>It's easy to give a vector to his coordinates.
&#36;&#36;x=(\varepsilon_{1},\varepsilon_{2},\cdots\varepsilon_{n})\begin{pmatrix}
 x_1\
 x_2\
 \vdots\
x_n
\end{pmatrix}=(\varepsilon_{1}^{\prime},\varepsilon_{2}^{\prime},\cdots\varepsilon_{n}^{\prime})\begin{pmatrix}
 x_1^{\prime}\
 x_2^{\prime}\
 \vdots\
x_n^{\prime}
\end{pmatrix}&#36;&#36;</p>
<p>There's a formula for the base shift that you get in front.
&#36;&#36;A^{-1}x=x^{\prime}&#36;&#36;
Which means...
&#36;&#36;x=Ax^{\prime}&#36;&#36;
of which&#36;x,x^{\prime}&#36;A column vector &#36;A&#36;It's the matrix we gave you earlier.</p>
<p>And it's not hard to see just by feeling that it's simple to look for a transition matrix between the standard bases, and two non-standard bases looking for a transition matrix are very complex, so let's give an example.</p>
<p>Two bases.
&#36;&#36;\left{\begin{matrix}
 \epsilon_{1}=(1,0,0\cdots,0)\
 \epsilon_{2}=(0,1,0,\cdots0)\
 \vdots\
\epsilon_{n}=(0,0,\cdots,1)
\end{matrix}\right.\quad\left{\begin{matrix}
 \epsilon_{1}^{\prime}=(1,1,1\cdots,1)\
 \epsilon_{2}^{\prime}=(0,1,1,\cdots1)\
 \vdots\
\epsilon_{n}^{\prime}=(0,0,\cdots,1)
\end{matrix}\right.&#36;&#36;
So it's easy to know from the standard base to the transition matrix of the latter.
&#36;&#36;\left.\left.\left(\begin{array}{cc}1&amp;1&amp;1&amp;\cdots&amp;1\0&amp;1&amp;1&amp;\cdots&amp;1\0&amp;0&amp;1&amp;\cdots&amp;1\0&amp;0&amp;0&amp;\cdots&amp;It's not like you're going to have to do it.
All we have to do is observe.</p>
<p>So, for the transition matrix between the non-standard bases, we give the intermediary method, for the base.&#36;A(a_1,a_2,\cdots,a_n)&#36; Key.&#36;B(b_1,b_2,\cdots,b_n)&#36;
There's a foundation.&#36;A&#36;  Equals the natural base multiplied by the matrix &#36;A&#36;  Key.&#36;B&#36;  Equals the natural base multiplied by the matrix &#36;B&#36;</p>
<p>Base for proxy calculation&#36;B&#36; equals the base A times the matrix&#36;A^{-1}&#36; Multiplication Matrix&#36;B&#36;  The transition matrix is&#36;A^{-1}B&#36;</p>
<h3>Linear Subspace</h3>
<p>It's easy to see that some linear space is part of a linear space.</p>
<p>Definitions: Fields&#36;P&#36;Linear Space Up&#36;V&#36;A non-empty collection.&#36;W&#36; Called&#36;V&#36;A linear subspace. If&#36;W&#36;Yeah.&#36;V&#36;Two calculations also make up linear space.</p>
<p>Theoretically: if linear space&#36;V&#36;Non-empty collections&#36;W&#36;Yeah.&#36;V&#36;The two calculations are closed, then.&#36;W&#36;It's just a subspace, so don't try to prove eight more.</p>
<p>Linear subspace is also a linear space, with the concept of dimensions and foundations. Because they can't have more irrelevant vectors than the entire space, the dimensions can be smaller than the original dimensions.</p>
<p>E.G.</p>
<ul>
<li>A subset of a single zero vector is a linear space called 0 subspace with a dimension of 0</li>
<li>Linear Space&#36;V&#36;Yeah.&#36;V&#36;A subspace equals two dimensions.</li>
<li>There are these two spaces in any linear space, they're called ordinary spaces, other linear spaces are called extraordinary spaces.</li>
</ul>
<p>Definitions: Establishment&#36;\alpha_1\alpha_2\cdots\alpha_n&#36; It's linear space.&#36;V&#36;It's not hard to see all linear combinations of this vector.&#36;k_{1}\alpha_{1}+k_{2}\alpha_{2}+\cdots+k_{n}\alpha_{n}&#36;It's a non-empty, closed collection of two algorithms, so it's...&#36;V&#36;One of the subspaces is recorded by vector. Group &#36;\alpha_1\alpha_2\cdots\alpha_n&#36; Subspace generated&#36;L(\alpha_1,\alpha_2\cdots \alpha_n)&#36;  of which&#36;\alpha_1\alpha_2\cdots\alpha_n&#36;Called its Generating Meta vectors</p>
<p>In limited dimensions of linear space, any subspace is thus available. Just use it as a base to generate a metre vector set.</p>
<p>Generating a meta vector set can be relevant (i.e. not necessarily a base) and it must be a foundation if the generation of a meta vector group is not linear. On the other hand, it would be nice to study it in highly unrelated groups. The dimensional study vector is fine.</p>
<p>Theories of this section:</p>
<ul>
<li>Two vector groups generate the same subspace, two vector groups, etc. Price</li>
<li>Generating subspace is the youngest space to include the generation of meta vectors.</li>
<li>Set&#36;W&#36;Yes. &#36;n&#36;Area&#36;V&#36;It's a sub-space. &#36;w_1,\cdots,w_m&#36;(&#36;m&#36;We'll find it. &#36;n-m&#36; A vector.&#36;V&#36;A set of foundations.</li>
</ul>
<h3>Concludement of subspace</h3>
<p>Theoretically: If&#36;V_1,V_2&#36;Yes.&#36;V&#36;Two linear subspaces, and so are their contact.&#36;V&#36;The linear subspace with a symbol. &#36;V_{1}\cap V_{2}&#36; Organisation</p>
<p>For space sex, we have</p>
<ul>
<li>&#36;V_1\cap V_2=V_2\cap V_1&#36;</li>
<li>&#36;(V_1\cap V_2)\cap V_3=V_1\cap(V_2\cap V_3)&#36;</li>
</ul>
<p>Definitions: If&#36;V_1,V_2&#36;Yes.&#36;V&#36;Two linear subspaces, the so-called subspaces, and&#36;V_1+V_2&#36;It means all that can be expressed as&#36;\alpha+\beta&#36; of which &#36;\alpha\in V_{1},\beta \in V_2&#36; The vectors make up the subassemblies.</p>
<p>Theoretically: If&#36;V_1,V_2&#36;Yes.&#36;V&#36;Two linear subspaces, then. &#36;V_1+V_2&#36; Yeah.&#36;V&#36;Linear subspace</p>
<p>For space and we have</p>
<ul>
<li>&#36;V_1+V_2=V_2+V_1&#36;</li>
<li>&#36;(V_1+V_{2})+V_{3}=V_{1}+(V_{2}+V_{3})&#36;</li>
</ul>
<p>With regard to the transfer of space, the following conclusions are clearly valid:</p>
<ul>
<li>&#36;W\subset V_1\quad W\subset V_2\quad\Rightarrow W\subset V_{1}\cap V_2&#36;</li>
<li>&#36;V_1\subset W\quad V_2\subset W\Rightarrow V_1+V_2\subset W&#36;</li>
<li>&#36;V_{1}\subset V_2\Longleftrightarrow V_1\cap V_2=V_1\Longleftrightarrow V_1+V_2=V_2&#36;</li>
</ul>
<h3>A dimension formula for subspace</h3>
<p>Introduction: For Zhang Seong-ja, we have&#36;L(\alpha_1,\alpha_2\ldots\alpha_s)+L(\beta_1,\beta_2\ldots\beta_r)=L(\alpha_1\ldots\alpha_s,\beta_1\ldots\beta_r)&#36;</p>
<p>Theorem (dimensional formula): We use&#36;dim(A)&#36;It means space.&#36;A&#36;The dimensions, so there is.&#36;dim(V_1)+dim(V_2)=dim(V_1+V_2)+dim(V_{1}\cap V_2)&#36;</p>
<p>Inference: From the dimension formula, it is not difficult to find that the subspace and the dimensions are generally less than the sum of the subspace dimensions, only&#36;dim(V_{1}\cap V_2)=0&#36;The time is equal. Well, if&#36;n&#36;The sum of the two subspace dimensions of the dimension of the dimension of the dimension of the dimension of the dimension of the dimension is greater than&#36;n&#36;There must be a non-zero public vector for both subspaces.</p>
<p>Here's a brief description of how to study the dimensions and bases of space using a dimension formula.</p>
<p>For space: by using the reasoning given above, the formation of a meta vector is very unrelated, so that the dimension and base can be obtained simultaneously.</p>
<p>For traffic: the dimension formula can help us get the dimension of delivery. We can give you space equations. &#36;V=x_{1}\alpha_{1}+\cdots+x_{s}\alpha_{s}=y_{1}\beta_{1}+\cdots+y_{r}\beta_r&#36;  of which &#36;\alpha,\beta&#36;It's the foundation of the subspace.&#36;x,y&#36;The equation is unknown.</p>
<p>We solve this equation and get a coefficient.&#36;x&#36;Or a coefficient.&#36;y&#36;It is easy to calculate the base and the dimensions of the space in which the trade is made.</p>
<h3>Straightness of subspace</h3>
<p>Definitions: Establishment&#36;V_1,V_2&#36;Yes.&#36;V&#36;Two linear subspaces if and &#36;V_1+V_2&#36; of each vector&#36;\alpha&#36; The decomposition is the only (&#36;\alpha=\alpha_{1}+\alpha_{2}\quad\alpha_{1}\in V_{1}\quad\alpha_{2}\in V_{2}&#36;) and it's called straight and straight.&#36;V_1\oplus V_2&#36;</p>
<p><strong>The straightness of linear space is just a special form of operation, straightness and means that&#36;\alpha=\alpha_{1}+\alpha_{2}\quad\alpha_{1}\in V_{1}\quad\alpha_{2}\in V_{2}&#36; We found it.&#36;\alpha_1,\alpha_2&#36;Is certain and only</strong></p>
<p>We're going to look at straight and straight theoretics to understand what it means.</p>
<p>Theoretically:&#36;V_1+V_2&#36;It's the only decomposition of straight, when and only as zero; that is,&#36;\alpha_1+\alpha_2=0\quad\alpha_1\in V_1\quad\alpha_2\in V_2\quad\Rightarrow\alpha_1=\alpha_2=0&#36;</p>
<p>Theoretically:&#36;V_1+V_2&#36;It's straight and simple.&#36;V_{1}\cap V_{2}={0}&#36;;i.e. 0 dimensions of the delivery space</p>
<p>Theorem: Set&#36;V_1,V_2&#36;Yes.&#36;V&#36;Two linear subspaces, then&#36;V_1+V_2&#36;It's a direct condition.&#36;dim(V_1)+dim(V_2)=dim(V_1+V_2)&#36;</p>
<p>Theorem (residual space): set&#36;U&#36;Yes.&#36;V&#36;There must be a subspace.&#36;V&#36;Subspace&#36;W&#36;Make&#36;V=U\oplus W&#36; At this point, we call&#36;W&#36;Yes.&#36;U&#36;Yeah.&#36;V&#36;The spare space.</p>
<p><strong>Only&#36;U&#36;When it's normal space, the spare space is unique.</strong></p>
<p>Theorem: Set&#36;(\varepsilon_{1},\varepsilon_{2},\cdots\varepsilon_{n})&#36; and&#36;(\eta_{1},\eta_{2},\cdots \eta_{n})&#36; Yes.&#36;V_1,V_2&#36;A set of foundations, then.&#36;V_1+V_2&#36;It's straight and simple.&#36;(\varepsilon_{1},\varepsilon_{2},\cdots\varepsilon_{n},\eta_{1},\eta_{2},\cdots \eta_{n})&#36; Linear is irrelevant.</p>
<h3>Symptoms of linear space</h3>
<p>The linear spatial homogeneity is similar to the decomposition structure, the unit matrix of the matrix, and the sub-type standard. It's an intergenerational, simple zero. It's dedicated to finding what's most important in linear space, that's research dimensions and foundations.</p>
<p>Set &#36;\varepsilon_{1},\varepsilon_{2},\cdots\varepsilon_{n}&#36; Yes. &#36;V_n^P&#36;On a set of foundations, so whatever. &#36;V&#36; There must be a coordinate on the vector. &#36;P^n&#36;. Which means the vector is related to the coordinates.&#36;V\to P^n&#36;The map on that must be a double. And it's the same thing with this mapping relationship in linear calculations.</p>
<p><strong>This means that the original study of abstract space can reach the space we know best.&#36;P^n&#36;Medium</strong></p>
<p>Definitions: Fields&#36;P&#36;It's two parts of space, and it's only one.&#36;V&#36;Present.&#36;V^{\prime}&#36;Double-shot&#36;\sigma&#36;Make
&#36;\sigma(\alpha+\beta)=\sigma(\alpha)+\sigma(\beta)&#36;   &#36;\sigma(k\alpha)=k\sigma(\alpha)&#36; of which &#36;\alpha,\beta \in V.&#36; &#36;k\in P&#36;
It's called a synonym.&#36;V\cong V^{\prime}&#36;</p>
<p><strong>Consistency is a relationship between two spaces, not necessarily between&#36;P^n&#36;Association. But any space.&#36;V&#36;I'm sure we'll find something like him.&#36;P^n&#36;</strong>  Use cell space&#36;P^n&#36;It's a good way to handle more abstract linear space.</p>
<p>For the same unit, we can easily give the following characteristics:</p>
<ul>
<li>&#36;\sigma(0)=0\quad\sigma(-\alpha)=-\sigma(\alpha)&#36;</li>
<li>&#36;\sigma(k_{1}\alpha_{1}+k_{2}\alpha_{2}+\cdots+k_{r}\alpha_{r})=k_{1}\sigma(\alpha_{1})+k_{2}\sigma(\alpha_{2})+\cdots+k_{r}\sigma(\alpha_{r})&#36;</li>
<li>&#36;V&#36;vector group in  &#36;\alpha_{1},\alpha_{2}\cdots \alpha_{r}&#36;   Linear is irrelevant. When and only when &#36;\sigma(\alpha_{1}),\sigma(\alpha_{2})\cdots \sigma(\alpha_{r})&#36;  Linear is irrelevant.</li>
<li>Same dimensions in the same space.</li>
<li>If&#36;V_1&#36;Yes.&#36;V&#36;Subspace, then.&#36;\sigma(V_1)&#36;Yes.&#36;\sigma(V)&#36;The subspace, and&#36;V_1&#36;and&#36;\sigma(V_1)&#36;There are the same dimensions. Map&#36;\sigma&#36;It's the same map.</li>
<li>The reverse map of the same map, the size of the same map, is still the same map.</li>
</ul>
<p>Theorem (same theorem):<strong>The two linear spaces are constructed on the same dimensional basis.</strong>And this theorem tells us that the dimensions are the essence of linear space.</p>
