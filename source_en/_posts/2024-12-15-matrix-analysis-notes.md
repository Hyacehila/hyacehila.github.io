---
title: 'Matrix Analysis: Generalized Inverses, Special Matrices, and Matrix Products'
title_zh: 矩阵分析：广义逆、特殊矩阵与特殊积
date: 2024-12-15 00:42:23 +0800
categories:
- Mathematics
- Algebra & Matrix Theory
tags:
- Matrix Analysis
- Linear Algebra
author: Hyacehila
mathjax: true
hidden: true
excerpt: Covers generalized inverses, special matrices, special products, Kronecker products, and common results in matrix
  analysis.
description: Covers generalized inverses, special matrices, special products, Kronecker products, and common results in matrix
  analysis.
excerpt_zh: 整理广义逆、特殊矩阵、特殊积、Kronecker 积和矩阵分析中的常用结论。
permalink: /blog/2024/12/15/matrix-analysis-notes/
lang: en
translation_key: 2024-12-15-matrix-analysis-notes
translation_status: machine
translation_source_hash: 6392888801a5c550d37700018bcec8c56fc0eded7ccf657f7f2f22c0d0c8789d
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<p>Matrix analysis and promising studies for its application<a href="/en/blog/2024/10/17/matrix-theory-notes/">Matrix</a>In the absence of a complete description of the matrix branch of the theory of near-modern algebras, we first consider studying the "extensive reverse" part of this paper and the "special build" part of this paper and the "special matrix" part of this paper.</p>
<p>As for further matrix analysis, what else are we looking at?</p>
<h2>The broad reverse of the matrix</h2>
<h3>Basic concepts of broad reversal</h3>
<p>The broad inverse matrix is a general extension of the reverse matrix, the need for which is the practical need for a linear equation to solve the problem, with a linear equation. Group
&#36;&#36;Ax=b,&#36;&#36;
When?&#36;A&#36; Yes.&#36;n&#36; Step formation, and &#36;detA\neq0&#36; , the solution of the equation group exists and is unique and can be written in
&#36;&#36;x=A^{-1}b.&#36;&#36;
However, the matrix encountered in many practical problems&#36;A&#36;It's often a guillotine or random.&#36;m\times n&#36;Matrix (General)&#36;m\neq n&#36;It's clear there's no usual counter-argument.&#36;A^{-1}&#36;And this makes one wonder if it's possible to promote the concept of a reverse matrix and introduce some kind of matrix of a similar nature. &#36;G&#36;So that it can still be interpreted as
&#36;&#36;x=Gb.&#36;&#36;</p>
<p>Pennos points out: for any complex number matrix &#36;A_{m\times n}&#36;, if a repeat matrix exists &#36;G_{n\times m}&#36;meet the following conditions:
&#36;begin{aligned}AGA=&amp;A,\GAG=&amp;G,\(GA)^{\mathrm{H&#125;&#125;=&amp;GA:,\(AG)^{\mathrm{H&#125;&#125;=&amp;AG:, \end{aligned}
Name&#36;G&#36;Yes&#36;A&#36;One of the Mor-Penos is broadly reversible, and the four equations above are called the Mor-Penos equation, short of the M-P equation.</p>
<p>Because these four equations are of a certain quality, and so is the contenting part.</p>
<p>Definitions: Establishment &#36;A\in\mathbb{C}^{m\times n}&#36;If there's one &#36;G\in\mathbb{C}^n\times m&#36;, meet all or part of the M-P equation, or &#36;G&#36; Yes&#36;A&#36; The broad counter-argument, abbreviated by the broad reverse.</p>
<p>We know that the reverse matrix can satisfy only a partial nature, so we can actually give 15 broad reverses.&#36;\mathrm{C}_4^1+\mathrm{C}_4^2+\mathrm{C}_4^3+\mathrm{C}_4^4=15&#36;But only part of it is more common.</p>
<ul>
<li>♪ Meet the first equation ♪&#36;A{1}&#36;  It's called the minus sign. &#36;A^-&#36;</li>
<li>Meet the equations one, two.&#36;A{1,2}&#36; It's called self-inversion. &#36;A^-_r&#36;</li>
<li>Meet the equation one, three do it.&#36;A{1,3}&#36;  It's called the minimum standard. &#36;A^-_m&#36;</li>
<li>Meet equations one, four.&#36;A{1,4}&#36; It's called the minimum two times broad reverse. &#36;A^-_l&#36;</li>
<li>Meet the equation one, two, three, four.&#36;A{1,2,3,4}&#36; It's called Qatar, Hypocrisy. &#36;A^+&#36;</li>
</ul>
<p>It's just a little counterproductive.&#36;A{1,2,3,4}&#36; Yes, and the rest of the broad reverses are not the only ones, as we will state later in our narrative.</p>
<h3>Deductive. &#36;A^-&#36;</h3>
<p>Definitions: Existing&#36;m\times n&#36;Real Matrix&#36;A(m\leqslant n&#36;When&gt;n&#36;时，可讨论&#36;A^{\mathrm{T&#125;&#125;).&#36;若有一个&#36;n\times m&#36;实矩阵(记为&#36;A^-&#36;)存在，使下式成立，则称&#36;A^-&#36;为&#36;A&#36;的减号逆或&#36;gReverse:
&#36;&#36;AA^-A=A.&#36;&#36;
When?&#36;A^{-1}&#36;When it existed, obviously.&#36;A^-1&#36;Satisfied top, visible minus sign reverse &#36;A^-&#36; It's an ordinary reverse matrix. &#36;A^-1&#36; promotion; in addition, by&#36;AA^-A=A&#36;Okay.
&#36;&#36;(AA^-A)^\mathrm{T}=A^\mathrm{T},\quad\text{即}\quad A^\mathrm{T}(A^-)^\mathrm{T}A^\mathrm{T}=A^\mathrm{T}.&#36;&#36;
Visible, when&#36;A^-&#36;Yes&#36;A&#36; And when a sign is reversed,&#36;(A^-)^\mathrm{T}&#36;Yeah.&#36;A^\mathrm{T}&#36; A derogatory sign.</p>
<p>Note: Subtracts are not unique, e. g.
&#36; \bardsymbol =begin{bmatrix}1&amp;0\1&amp;0\1&amp;0\end{bmatrix},\boldsymbol{B}=\begin{bmatrix}1&amp;0&amp;0\0&amp;1&amp;0\end{bmatrix},\boldsymbol{C}=\begin{bmatrix}1&amp;0&amp;0\0&amp;0&amp;What's wrong with you?
At this point, &#36;B,C&#36; Both. &#36;A&#36; ♪ The minuscule reverse ♪</p>
<p>Here, we're talking about proving the existence of the minus sign, that is, looking for the minus sign of any matrix.</p>
<p>Theoretically:&#36;\text{任给 }m\times n\text{ 矩阵 }A,\text{那么减号逆 }A^--\text{定存在 },\text{但不惟一}.&#36;</p>
<p>If &#36;rankA=0&#36; Then there must be, whatever.&#36;X\in R^{n\times m}&#36;  Both. &#36;0X0=0&#36; So the minus sign is negative and not unique.</p>
<p>If &#36;rankA\neq0&#36;  And there must be plenty of them.&#36;m&#36; Step Matrix&#36;P&#36; And full.&#36;n&#36; Step Matrix&#36;Q&#36;  Make
&#36;PAQ=begin{bmatrix}I r&amp;0\0&amp;0\end{bmatrix}=B\in\mathbb{R}^{m\times n}&#36;&#36;
<strong>It's actually a primary shift to a unit matrix.&#36;P,Q&#36;It's all a matrix for the transformation of the primary.</strong></p>
<p>According to the nature not given here, yes.
&#36; \bardsymbol^&amp;&amp;\star\\star&amp;&amp;\star\end{bmatrix}\quad(\star\text{optional}). &#36;</p>
<p>And by the nature of one of the things not given here, yes.
&#36;A^=Q{\begin{bmatrix}I r&amp;&amp;\star\\\star&amp;&amp;\star\end{bmatrix}P&#36;
Because&#36;\star&#36;It's arbitrary, it's not unique.</p>
<p>We'll give you a calculation.&#36;P,Q&#36;This is a sign.&#36;A&#36;It's a 2-line, 3-line matrix.&#36;2\times3&#36;Matrix for Top Left&#36;I_2&#36;The rest is zero.
&#36; \begin{bmatrix}
A&amp; I_2\
  I_3&amp;0
\end{bmatrix}=\begin{bmatrix}1&amp;-1&amp;2&amp;1&amp;0\2&amp;2&amp;3&amp;0&amp;1\1&amp;0&amp;0&amp;\0&amp;1&amp;0&amp;\0&amp;0&amp;1&amp;What do you mean?
Theoretically:&#36;\quad\mathrm{rank}A^-\geqslant\mathrm{rank}A.&#36;</p>
<h3>Self-deductive. &#36;A^-_r&#36;</h3>
<p>An ordinary counter-argument is self-reversible, that is,&#36;(A^{-1})^{-1}=A&#36; But the average minus sign is not enough, for example.
&#36; \mathbf{1=begin{bmatrix}&amp;0\1&amp;0\1&amp;0\end{bmatrix},\quad\mathbf{A}^-=\begin{bmatrix}1&amp;0&amp;0\0&amp;1&amp;0\end{bmatrix}&#36;&#36;
容易验证 &#36;AA^-A=A.&#36; 一侧的减号逆成立，但是
&#36;&#36;A^-AA^-=\begin{bmatrix}1&amp;0&amp;0\1&amp;0&amp;I'm sorry.
Which means... &#36;(A^{-1})^{-1}=A&#36; No, so we need to limit the concept of retrogression to a self-retrogressive one, and then we'll calculate retrogression.</p>
<p>Definition: For one&#36;m\times n&#36;Real Matrix&#36;A&#36;♪ That makes
&#36;&#36;AGA= A\text{及}GAG= G&#36;&#36;
Created at the same time.&#36;n\times m&#36;Real Matrix&#36;G&#36;Call it.&#36;A&#36;A self-retrogression.</p>
<p>Here's how we're going to do this. First, we need to introduce the concept of reverse.</p>
<p>Definitions: Establishment &#36;A\in\mathbb{R}^m\times n&#36;If there is &#36;G\in\mathbb{R}^n\times m&#36;♪ That makes ♪
&#36;&#36;AG=I\quad\text{或}\quad GA=I,&#36;&#36;
Name&#36;G&#36; Yes&#36;A&#36; and the right reverse (or the left reverse), as &#36;A_{\mathbb{R&#125;&#125;^{-1}(&#36;or &#36;A_{\mathbb{L&#125;&#125;^{-1})&#36;i.e.&#36;AA_{\mathbb{R} }^{- 1}= I&#36; or &#36;A_{\mathbb{L} }^{- 1}A= I.&#36;</p>
<p>In general&#36;,A_{\mathbb{R&#125;&#125;^{-1}\neq A_{\mathbb{L&#125;&#125;^{-1}&#36;If &#36;A_{\mathbb{R&#125;&#125;^{-1}=A_{\mathbb{L&#125;&#125;^{-1}&#36;, then &#36;A^{-1}&#36;exist, and &#36;A^{-1}=A_{\mathbb{R&#125;&#125;^{-1}=A_{\mathbb{L&#125;&#125;^{-1}.&#36;</p>
<p>Theoretically: Set A is the largest in the row.&#36;m\times n&#36;Real Matrix&#36;m\leqslant n&#36;. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .&#36;A&#36;On the right,
&#36;&#36;A_{\mathrm{R&#125;&#125;^{-1}=A^{\mathrm{T&#125;&#125;(AA^{\mathrm{T&#125;&#125;)^{-1}:;&#36;&#36;
Common: Establishment&#36;\boldsymbol{A}&#36;It's the one in the column.&#36;n\times m&#36;Real Matrix&#36;m\geqslant n)&#36;. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .&#36;{A}&#36;On the left,
&#36;&#36;A_{\mathrm{L&#125;&#125;^{-1}=(A^{\mathrm{T&#125;&#125;A)^{-1}A^{\mathrm{T&#125;&#125;.&#36;&#36;</p>
<p><strong>From the theory, there's only...&#36;m=n&#36;And...&#36;A&#36;When full, both the left and the right are present and equal, equal to their counter-argument.&#36;A^{-1}&#36;</strong></p>
<p>Theorem: The nature of the left and right reverses calculated on the basis of the former theorem</p>
<ul>
<li>Satisfactory 1  &#36;AA_{\mathrm{R&#125;&#125;^{-1}A=A\quad(AA_{\mathrm{L&#125;&#125;^{-1}A=A)&#36;</li>
<li>Meet equation 2 &#36;A_\mathrm{R}^{-1}AA_\mathrm{R}^{-1}=A_\mathrm{R}^{-1}\quad(A_\mathrm{L}^{-1}AA_\mathrm{L}^{-1}=A_\mathrm{L}^{-1})&#36;</li>
<li>Meet equation 3 &#36;(A_{\mathrm{R&#125;&#125;^{-1}A)^{\mathrm{T&#125;&#125;=A_{\mathrm{R&#125;&#125;^{-1}A\quad(A_{\mathrm{L&#125;&#125;^{-1}A)^{\mathrm{T&#125;&#125;=A_{\mathrm{L&#125;&#125;^{-1}A&#36;</li>
<li>Meet equation 4 &#36;(AA_{\mathrm{R&#125;&#125;^{-1})^{\mathrm{T&#125;&#125;=AA_{\mathrm{R&#125;&#125;^{-1}\quad(AA_{\mathrm{L&#125;&#125;^{-1})^{\mathrm{T&#125;&#125;=AA_{\mathrm{L&#125;&#125;^{-1}&#36;</li>
</ul>
<p>In other words, for rows or arrays full of arrays, the left reverse or right reverse given according to the preceding calculation is not only a reverse minus, but also a reverse minus, the smallest is a broad reverse, the smallest two times a broad reverse, plus a negative plus.</p>
<p>Here's the calculation of the reverse of the inverse, which is universal and inherited from this paper. &#36;A -&#36;Part of it, and it would be very good to expand to the latter.</p>
<p>When?&#36;A&#36;When you're in rows or arrays, you can use right and right, and you can only talk about the situation, and you can find it.
&#36; \mathbf{PAQ}=begin{bmatrix}\mathbf{I}<em>r&amp;\mathbf{0}\\mathbf{0}&amp;\mathbf{0}\end{bmatrix}&#36;&#36;
这等价于
&#36;&#36;\mathbf{A}=\mathbf{P}^{-1}\begin{bmatrix}\mathbf{I}<em>r&amp;\mathbf{0}\\mathbf{0}&amp;\mathbf{0}\end{bmatrix}\mathbf{Q}^{-1}=\mathbf{P}^{-1}\begin{bmatrix}\mathbf{I}<em>r\\mathbf{0}\end{bmatrix}(\mathbf{I}<em>r\mathbf{0})\mathbf{Q}^{-1}&#36;&#36;
令
&#36;&#36;I'm sorry.
So there is.<strong>It's actually...<a href="/en/blog/2024/10/17/matrix-theory-notes/">The biggest breakdown in the matrix theory</a>But the narrative has changed.</strong>）
&#36;&#36;A=BC&#36;&#36;
Calculate
&#36;B</em>\mathrm{L}^{-1}=(B^\mathrm{T}B)^{-1}B^\mathrm{T},\quad C</em>\mathrm{R}^{-1}=C^\mathrm{T}(CC^\mathrm{T})^{-1}&#36;&#36;
因此有
&#36;&#36;A</em>\mathrm{r}^-=C</em>{\cHFFFFFF}{\cH00FFFF}
It's verified that it meets one of the M-P equations, and that's self-reverse.</p>
<h3>The minimum standard is broadly reversed. &#36;A^-_m&#36;</h3>
<p>Definitions: Establishment &#36;A\in\mathbb{R}^{m\times n}\left(m\leqslant n\right)&#36;, if there's one &#36;n\times m&#36; Step Matrix &#36;G&#36;Satisfied
&#36;&#36;AGA=A\quad\text{及}\quad(GA)^\mathrm{T}=GA:,&#36;&#36;
Name&#36;G&#36;Yes&#36;A&#36;And one of the smallest of them is a broad reverse.&#36;A_\mathrm{m}^-.&#36;</p>
<p>The minimum standard is broadly reversed. &#36;A^-_m&#36;There's a method of calculation below.</p>
<ul>
<li>As Matrix&#36;A&#36;When rows or columns are full, use right and left reverses. (Proved earlier)</li>
<li>As Matrix&#36;A&#36;If you do not meet the line or line full, you can use the maximum dissectation&#36;A=BC&#36; There is.&#36;&#36;A_{\mathrm{m&#125;&#125;^{-}=C_{\mathrm{R&#125;&#125;^{-1}B_{\mathrm{L&#125;&#125;^{-}&#36;&#36;
Which means the method of calculation and this paper, "Reverse-minus-inverse." &#36;A -_r&#36;Partly identical</li>
</ul>
<p>Theoretically: We can also give a simpler formula to calculate the minimum standard broadly against &#36;A^--<em>Yeah.
&#36;A</em>{\mathrm{m&#125;&#125;^{-}=A^{\mathrm{T&#125;&#125;(AA^{\mathrm{T&#125;&#125;)&#36;&#36;</p>
<h3>Minimum 2 times broad reverse &#36;A^-_l&#36;</h3>
<p>Definitions: Establishment&#36;A\in\mathbb{R}^{m\times n}\left(m\leqslant n\right)&#36;♪ If there's one ♪&#36;n\times m&#36;Step Matrix&#36;G&#36;Satisfied&#36;AGA= A&#36; and &#36;( AG) ^{\mathrm{T} }= AG&#36; Name&#36;G&#36; Yes&#36;A&#36; A minimum of two times the broad reverse, as follows: &#36;A_{\mathrm{l&#125;&#125;^{-}.&#36;</p>
<p>The minimum standard is broadly reversed. &#36;A^-_m&#36;There's a method of calculation below.</p>
<ul>
<li>As Matrix&#36;A&#36;When rows or columns are full, use right and left reverses. (Proved earlier)</li>
<li>As Matrix&#36;A&#36;If you do not meet the line or line full, you can use the maximum dissectation&#36;A=BC&#36; There is.&#36;&#36;A_{\mathrm{m&#125;&#125;^{-}=C_{\mathrm{R&#125;&#125;^{-1}B_{\mathrm{L&#125;&#125;^{-}&#36;&#36;
Which means the method of calculation and this paper, "Reverse-minus-inverse." &#36;A -_r&#36;Partly identical</li>
</ul>
<p>Theorem: We can also give a simpler formula to calculate a minimum of two times the broad reverse. &#36;A^-_l&#36;Yes.
&#36;&#36;A_1^-=(A^\mathrm{T}A)^-A^\mathrm{T}.&#36;&#36;</p>
<h3>Add reverse.&#36;A^+&#36;</h3>
<p>We're in front of you.&#36;A^{-}&#36;Different restrictions are imposed to produce a retrograde reverse of a different nature, such as self-retrogression in broad terms&#36;A_\mathrm{r}^-&#36;Minimum standard is broadly reversed&#36;A_\mathrm{m}^-&#36;a minimum of two times the broad reverse&#36;A_\mathrm{l}^-&#36;Wait.</p>
<p>In fact, there's another category that's more special and more important in the broad sense of the word.&#36;A^+&#36;The essence of it is the condition of reduction. &#36;AGA=A&#36; on the basis of all the above-mentioned conditions.&#36;A^+&#36;Not only is it particularly important in terms of application, it's also a lot of fun.</p>
<p>Definitions: Establishment&#36;A\in\mathbb{R}^{m\times n}\left(m\leqslant n\right)&#36;♪ If there's one ♪&#36;n\times m&#36;Step Matrix&#36;G&#36;Satisfied simultaneously
&#36;begin{aligned}AGA=&amp;A,\GAG=&amp;G,\(GA)^{\mathrm{H&#125;&#125;=&amp;GA:,\(AG)^{\mathrm{H&#125;&#125;=&amp;AG:, \end{aligned}
Name&#36;G&#36;Yes&#36;A&#36;A Mur-Penos who is in the broadest sense of the word, or is called the "Mur-Penos."&#36;A^+&#36;</p>
<p>As can be seen from the definition, the two matrices are in exactly the same position, that is,
&#36;&#36;(A^+)^+=A&#36;&#36;</p>
<p>Theorem: We can calculate the additional countervailing if&#36;A=BC&#36;It's the biggest disassembly, and there is.
&#36;&#36;X=C^\mathrm{T}(CC^\mathrm{T})^{-1}(B^\mathrm{T}B)^{-1}B^\mathrm{T}&#36;&#36;
Yes.&#36;A&#36;; of course, the right and the left are still available in the case of rows or columns; and for this paper, "The self-reversibility is still available. &#36;A -_r&#36;The disaggregation of grievances given in the section is still available; the results are consistent.</p>
<p>Theorem: for any person&#36;A\in R^{m\times n}&#36; And the extra-protest &#36;A^+&#36; Existence and Unique</p>
<p>Inference: when&#36;A&#36;Yes.&#36;n&#36;When the formation is full, it's...&#36;A^{-1}&#36;Normal reverse presence, so there is.
&#36;&#36;A^+=A^{-1}=A^-&#36;&#36;</p>
<p>Theorem: Add reverse&#36;A^+&#36; It's of a nature.</p>
<ul>
<li>&#36;(A^{\mathrm{T&#125;&#125;)^{+}=(A^{+})^{\mathrm{T&#125;&#125;&#36;</li>
<li>&#36;{A}^+=({A}^{T}{A})^+{A}^{T}={A}^{T}({A}{A}^{T})^+&#36;</li>
<li>&#36;(A^\mathrm{T}A)^+=A^+(A^\mathrm{T})^+&#36;</li>
<li>&#36;\mathrm{rank}A=\mathrm{rank}A^+=\mathrm{rank}A^+A=\mathrm{rank}AA^+&#36;</li>
</ul>
<h2>Special Matrix</h2>
<p>The special matrix would like to study such forms as the diagonal matrix, the triangle matrix, symmetric matrix, which are of a special nature. Those that are already in the upper algebra with<a href="/en/blog/2024/10/17/matrix-theory-notes/">Matrix</a>The research will not be repeated here. The matrix that we're looking at here consists of non-negative matrix, random matrix, M and H matrix, etc.</p>
<h3>Non-negative matrix</h3>
<p>In a very large number of applications, elements are often presented as non-negative matrices. We call it a non-negative matrix in mathematics, and his basic features are already an essential part of the matrix. This section discusses the nature of the non-negative matrix and its derivatives.</p>
<h4>Non-negative and positive matrices</h4>
<p>Definitions: Establishment &#36;A=(a_{ij})\in\mathbb{R}^{m\times n}&#36;if
&#36;&#36;a_{ij}\geqslant0:,\quad i=1:,\cdots,m:;:j=1:,\cdots,n:,&#36;&#36;
That's...&#36;A&#36;All elements are non-negative, but call&#36;A&#36;As non-negative matrix, recorded&#36;A\geqslant0;&#36;If strict range is established, &#36;a {&gt;0&#36; (&#36;i=1,\cdots,m;j=1,\cdots,n&#36;),则称 &#36;A&#36; 为正矩阵，记为&#36;\boldsymbol{A}&gt;0.&#36;</p>
<p>Set &#36;A,{B}\in{R}^{m\times n}&#36;, if established &#36;A-{B\geqslant}0&#36;as at &#36;A\geqslant{B};&#36;   If established &#36;A-{B]&gt;}0&#36;,则记作 &#36;A&gt;{B}.&#36;</p>
<p>For random &#36;{A}=(a_{ij})\in\mathbb{C}^{m\times n}&#36;, import marks
&#36;&#36;\mid A\mid=(\mid a_{ij}\mid),&#36;&#36;
..is to be used as a &#36;a_{ij}&#36;Oh, Jin-mo.&#36;|a_{ij}|&#36;(a) Non-negative matrix of elements;</p>
<p>Especially when &#36;x=(x_1,\cdots,x_n)^{\mathrm{T&#125;&#125;\in\mathbb{C}^n&#36;I don't know.&#36;| \boldsymbol{x}| = (&#36; | &#36;x_1&#36; | &#36;, \cdots&#36; , | &#36;x_n&#36; | &#36;) ^\mathrm{T}&#36; It means a non-negative vector.</p>
<p>Theorem: The non-negative matrix can easily give the following characteristics: &#36;A,B,C,D\in\mathbb{C}^{m\times n}&#36; then</p>
<ul>
<li>&#36;|A|\geqslant0,\text{并且}|A|=0\text{ 当且仅当 }A=0&#36;</li>
<li>&#36;\text{对任意复数 }\alpha,\text{有}\mid\alpha A\mid=\mid\alpha\mid\mid A\mid&#36;</li>
<li>&#36;|A+B|\leqslant|A|+|B|&#36;</li>
<li>&#36;\text{若 }A\geqslant0,B\geqslant0,a,b\text{ 是非负实数,则 }aA+bB\geqslant0&#36;</li>
<li>&#36;\text{若 }A\geqslant B,\text{且 }C\geqslant D,\text{则 }A+C\geqslant B+D&#36;</li>
<li>&#36;\text{若 }A\geqslant B,\text{且 }B\geqslant C,\text{则 }A\geqslant C&#36;</li>
<li>General,&#36;A\ge0&#36; and &#36;A\ne0&#36; Can not open &#36;A&gt;0&#36;</li>
</ul>
<p>Theorem: The non-negative matrix can easily give the following characteristics: &#36;A,B,C,D\in\mathbb{C}^{m\times n},x\in C^n&#36;  then</p>
<ul>
<li>&#36;|Ax|\leqslant|A|\mid x|&#36;</li>
<li>&#36;|AB|\leqslant|A|\mid B|&#36;</li>
<li>&#36;\text{对任意正整数 }m,\text{有}\mid A^m\mid\leqslant\mid A\mid^m&#36;</li>
<li>&#36;\text{若 }0\leqslant A\leqslant B,0\leqslant C\leqslant D,\text{则 }0\leqslant AC\leqslant BD&#36;</li>
<li>&#36;\text{若 }0\leqslant A\leqslant B,\text{对任意正整数 }m,\text{有 }0\leqslant A^m\leqslant B^m&#36;</li>
<li>&#36;\text{if] A\geqslant0(A)&gt;),\text{for any positive integer}m, A^m\geqslant0 (A^m)&gt;0)&#36;</li>
<li>If A-A-A-A-A-A-A-A-A-A-A-A-A-A-A-A-A-A-A-A-A-A-A-A-A-A-A-A-A-A-A-A-A-A-A-A-A-A-A-A-A-A-A-A-A-A-A-A-A-A-A-A-A-A-A-A-A-A-A-A-A-A-A-A-A-A-A-A-A-A-A-A-A-A-A-A-A-A-A-A-A-A-A-A-A----------&gt;0, x\geqslant0 and x\neq0, x x&gt;0&#36;</li>
<li>&#36;\text{若}|A|\leqslant B,\text{则}\parallel A\parallel_2\leqslant\parallel\mid A\mid\parallel_2\leqslant\parallel B\parallel_2&#36;</li>
</ul>
<p>Theorem (uniformity of spectrum radius): set &#36;A,B\in\mathbb{C}^{n\times n}&#36;If&#36;|A|\leqslant\boldsymbol{B}&#36;, then
&#36;&#36;\rho(A)\leqslant\rho(\mid A\mid)\leqslant\rho(B).&#36;&#36;
We can easily give two inferences about the spectrum radius.</p>
<ul>
<li>&#36;\text{设 }A,B\in\mathbb{R}^{n\times n},\text{若 }0\leqslant A\leqslant B,\text{则 }\rho(A)\leqslant\rho(B).&#36;</li>
<li>&#36;\text{设 }A\in\mathbb{R}^{n\times n},\text{若 }A\geqslant0,A^{(k)}\text{是 }A\text{ 的任一主子矩阵},\text{则 }\rho(A^{(k)})\leqslant\rho(A).&#36;  &#36;\text{ 特别地 },\max_{1\leq i\leq n}\langle a_{ii}\rangle\leqslant\rho(A).&#36;</li>
</ul>
<p> Theorem (Perron Theorem, The Nature of the Positive Matrix Characteristics and the Characteristic Vectors established by Perron): Set &#36;A\in\mathbb{R}^{n\times n}&#36;and &#36;\rho(A)&#36;& Radius of the spectrum if &#36;A&gt;0 (&#36; positive matrix), then</p>
<ul>
<li>&#36;\rho(\boldsymbol{A})为\boldsymbol{A}&#36;the positive characteristic value, the corresponding characteristic vector &#36;y\in\mathbb{R}^n&#36;and shall be a positive vector;</li>
<li>Yeah. &#36;A&#36; any other feature value &#36;\lambda&#36;It's all in the twilight.&lt;\rho(A);&#36;</li>
<li>&#36;\rho(A)&#36;Yes. &#36;A&#36; . The single characteristic value.</li>
</ul>
<p>Inferences:&#36;\text{正矩阵 }\mathbf{A}\text{ 的“模等于 }\rho(\mathbf{A})\text{”的特征值是惟一的}&#36;</p>
<p>We can give two more important theories, and they have a good application.</p>
<p>Theorem: set &#36;A=(a {j})<em>{n\times n},\boldsymbol{B}=(b</em>{ij})<em>{n\times n}\in\mathbb{R}^{n\times n}&#36;为非负矩阵&#36;|a_i,|\leqslant b</em>♪ I, j = 1, 2,... n,
&#36;&#36;\lambda(A)\subset\bigcup\limits_{i=1}^n\langle z\in\mathbb{C}\big|\mid z-a_{ii}:|\leqslant\rho(B)-b_{ii}:\rangle &#36;&#36;</p>
<p>Theorem: Set &#36;A\in\mathbb{R}^n\times n&#36;, if &#36;A&gt;0,x&#36; 是&#36;A&#36; 的对应于特征值&#36;\rho(A)&#36;的正特征向量，又 &#36;y&#36;是&#36;A^{\mathrm{T&#125;&#125;&#36;的对应于特征值 &#36;\rho(A)&#36;, any positive feature vector, if
&#36;&#36;\lim_{m\to\infty}[\rho(A)^{-1}A]^m=(y^\mathrm{T}x)^{-1}xy^\mathrm{T}.&#36;&#36;</p>
<h4>Negative Matrix Not subject to engagement</h4>
<p>We'll continue.<strong>Promote Penno Theorem to a broader matrix.</strong>He's not a negative matrix, but there's no positive matrix.</p>
<p>In the online algebra, we know we need to reconcile the matrix. &#36;A&#36; No. No.&#36;i,j&#36;Two rows (column), equivalent to one&#36;A&#36;Left (right) multiplied by the corresponding calibration matrix&#36;I_{i,j}&#36;
&#36;&#36;\boldsymbol{I}_{i,j}=\begin{pmatrix}1\&amp;\ddots\&amp;&amp;0&amp;\cdots&amp;1\&amp;&amp;\vdots&amp;\ddots&amp;\vdots\&amp;&amp;1&amp;\cdots&amp;0\&amp;&amp;&amp;&amp;&amp;\ddots\&amp;&amp;&amp;&amp;&amp;&amp;I'm sorry, I'm sorry.
We're putting together a series of calibration matrices.&#36;P&#36;Called the replacement matrix (or the array) which is clearly available&#36;P^{-1}=P^T&#36;</p>
<p>Definition (A matrix of concretisation and non-concreteability): set &#36;A\in\mathbb{R}^{n\times n}(n\geqslant2)&#36;, if it exists &#36;n&#36; The styl array &#36;P&#36;♪ And so on ♪
&#36; \bardsymbol^mathrm^T}: \left=<em>{11}&amp;\boldsymbol{A}</em>{12}\\boldsymbol{0}&amp;\boldsymbol{A}<em>{\cHFFFFFF}{\cH00FF00} {\cHFFFFFF}{\cH00FF00}
Of which &#36;A</em>{11}&#36;为&#36;r&#36;阶方阵&#36;,A_{22}&#36;为&#36;n-r&#36;阶方阵( l&#36;\leqslant r&lt;n&#36;),则称&#36;A&#36;为可约(可分)矩阵，否则称&#36;A.A. is the non-arrangeable matrix.  <strong>It's actually a triangle based on multiple alignments.</strong></p>
<p>Obviously, if all elements are not zero, then they must be impossible.</p>
<p>The concept of the negotiability is derived from the solver of the linear equation group. A linear equation is a calibration matrix that is possible, indicating that the equation group can be solved by adjusting the order of the equation and the unknown number to two lower steps of the equation group. Group
&#36;&#36;Ax=b&#36;&#36;
. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .&#36;A&#36;When you can, you can find a replacement matrix.&#36;P&#36;♪ And I'm gonna be ♪&#36;A&#36;Present
&#36; \bardsymbol^mathrm^T}: \left(\begin{matrix}\bardsymbol{A}<em>{11}&amp;\boldsymbol{A}</em>{12}\\boldsymbol{0}&amp;\boldsymbol{A}_{22}\end{matrix}\right).&#36;&#36;
于是原方程组可化为
&#36;&#36;PAP =(Px)=Pb.&#36;
In turn&#36;y=px=(\mathbf{y}_1^\mathrm{T},\mathbf{y}_2^\mathrm{T})^\mathrm{T}&#36;And \hat(boldsymbol{b=b=<em>♪ I'm not sure I'm gonna be able to do this ♪
&#36; \begin{cases}\bardsymbol{A}</em>{11}\boldsymbol{y}<em>1+\boldsymbol{A}</em>{12}\boldsymbol{y}_2=\boldsymbol{\hat{b&#125;&#125;<em>1,\\boldsymbol{A}</em>\bardsymbol{bardsymbol{\boldsymbol{\b}b
So the equations are organized into two separate low-scale equation groups, easier and simpler than the direct solution to the original equation.
I am not sure. &#36;A&#36; The characterization polyformula is also converted into a multiplication of the two low-level matrices.</p>
<p>Theorem (judgment of availability): Set&#36;A\in R^{n\times n}&#36; then</p>
<ul>
<li><p>&#36;A为不可约矩阵的充分必要条件是A^T为不可约矩阵&#36;</p>
</li>
<li><p>&#36;如果A是不可约非负矩阵,B是n阶非负矩阵,则A+B是不可约非负矩阵&#36;</p>
</li>
<li><p>&#36;n(\geqslant2)\text{阶非负矩阵 A 不可约的充分必要条件是存在正整数 }s\leqslant n-1&#36; Make &#36;&#36;&#36;(I+A)^s&gt;&#36;0.00
Theorem (Perón-Frobenius Theorem): &#36;A\in\mathbb{R}^{n\times n}&#36;It's a non-negative matrix, then.</p>
</li>
<li><p>&#36;A&#36;A positive signature is exactly the spectral radius. &#36;\rho(A)&#36;, and there is a positive amount &#36;x\in\mathbb{R}^n&#36;♪ That makes ♪&#36;&#36;Ax= \rho ( A) x&#36;&#36;</p>
</li>
<li><p>&#36;\rho(A)&#36;Yes. &#36;A&#36; ;</p>
</li>
<li><p>When?&#36;A&#36;Any element(s) added&#36;,\rho(A)&#36;Increase.</p>
</li>
</ul>
<p>It's worth mentioning that the matrix is generally non-negotiable.&#36;A&#36;Péron-Frobenius theorem is not a guarantee.&#36;A&#36;"The model equals &#36;\rho(A)”&#36;The only characteristic value is the one that Penno theorem does not promote.</p>
<p>We still give some valuable theorem to the end of this section.</p>
<hr>
<h2>Theorem: Set up \\bardsymbol{A}(a ijj})<em>{n\times n} &#36; is a non-negotiable matrix, or
&#36;US&#36;US&#36;&#36;USM</em>{j=1}^na_{ij}:=:\rho(A:):,\quad i:=:1:,2:,\cdots,n:,&#36;&#36;
或者
&#36;&#36;\min_{1\leqslant i\leqslant n}\sum_{j:=:1}^na_{ij}:&lt;\rho(A)&lt;\max_{1\leqslant i\leqslant n}\sum_{j:=:1}^na_{ij}:.&#36;&#36;</h2>
<h2>Inferences:&#36;A&#36;For non-negligible matrix, the positive amount is given to any given vector&#36;x=(x_1,x_2,\cdots,x_n)^{\mathrm{T&#125;&#125;&#36;Or there is.
&#36;&#36;\frac{1}{x_i}\sum_{j=1}^na_{ij}x_j=\rho(A):,\quad i=1,2,\cdots,n&#36;&#36;
Or there is.
&#36;US&#36;1qslant i\legl}&lt;\rho(A)&lt;\max_{1\leqslant i\leqslant n}\biggl(\frac{1}{x_{i&#125;&#125;\sum_{j=1}^{n}a_{ij}x_{j}\biggr).&#36;&#36;</h2>
<h4>Sutangular and Cycle Matrix</h4>
<p>To this end, a concept of a matrix between a non-negative matrix and a positive matrix - a veritable matrix and a circulation matrix - has been introduced, with a variety of definitions, defined by the weight of the spectrum radius and as a different form of nature.</p>
<p>Definitions: Establishment&#36;A&#36;Yes.&#36;n&#36;Step is not negative matrix, and there is&#36;m&#36;The signature value is equally spectra-ranged.&#36;\rho(\boldsymbol{A})&#36;, and when&#36;m=1&#36; When you're in, you're in.&#36;A&#36;is the vegan matrix (or the original matrix); when &#36;m&gt;1&#36;时，就称&#36;A&#36;是循环矩阵(或非素矩阵).&#36;m&#36; is referred to collectively as a non-informative indicator.</p>
<p>Theorem: Set&#36;A,B&#36;Both.&#36;n&#36;step is not negative matrix, and&#36;A&#36;It's a vegan matrix, then.</p>
<ul>
<li>&#36;A^{\mathrm{T&#125;&#125;&#36; It's also a matrix;</li>
<li>for any positive integer&#36;k,A^k&#36;It's also a matrix;</li>
<li>&#36;A+ B&#36; It's also a vegan matrix.</li>
</ul>
<p>Theorem: Non-negative matrix&#36;A&#36;It is a sufficient requirement for the vegan matrix (this original matrix), it is the existence of a positive integer k, making &#36;A^k&gt;0.&#36;</p>
<p><strong>Penno Theorem and its reasoning are still in place on the matrix, which is in fact a special matrix.</strong>  As for the Péron-Frobenius Theorem, we can only give the following.</p>
<p>Theorem: Set &#36;A\in\mathbb{R}^{n\times n}&#36;As a non-negative matrix, there are conclusions:</p>
<ul>
<li>&#36;\rho(\boldsymbol{A})&#36;Yes.&#36;\boldsymbol{A}&#36;, and belongs to &#36;\rho(\boldsymbol{A})&#36;..that is, there is a non-negative vector that is not zero. &#36;x&#36;♪ That makes ♪ &#36;Ax=\rho(A)x(&#36;Attention, here. &#36;\rho(A)&#36;and &#36;x&#36; Not necessarily positive;</li>
<li>&#36;\boldsymbol{A}&#36;. The characteristic values can be divided into groups, each of which has an equal signature model and are " evenly" distributed around a circle with its origin centre (note, here).&#36;A.&#36;All feature values are not less than or equal to&#36;\rho(A)&#36; )</li>
</ul>
<h3>Random Matrix</h3>
<p>Here is another very important matrix -- the random matrix -- which we're looking at in terms of its nature and the context of some applications.</p>
<p>Definitions: Establishment&#36;A=(a_{i,j})\in\mathbb{R}^{n\times n}&#36;Is a non-negative matrix if&#36;A&#36;The sum of the elements on each line equals one, i.e.
&#36;&#36;\sum_{j:=:1}^na_{ij}:=:1:,\quad i:=:1:,2:,\cdots,n:,&#36;&#36;
Name&#36;A&#36;is a random matrix; if&#36;A&#36;Still satisfied.
&#36;&#36;\sum_{i:=:1}^na_{ij}:=:1:,\quad j:=:1:,2:,\cdots,n:,&#36;&#36;
Name &#36;A&#36; For a double random matrix.</p>
<p><strong>&#36;A&#36;The reason why you call it a random matrix is because&#36;A&#36;Every line of work can be seen as having&#36;n&#36;The distribution of discrete concepts in the sample space of each point. Such matrices often appear in a variety of mathematical modelling issues in the areas of inter-urban migration models, Markov chain research and economics and operational preparation.</strong></p>
<p>Theorem: We are simply discussing the unique nature of some random matrices.</p>
<ul>
<li>Set &#36;A\in\mathbb{R}^{n\times n}&#36;It's a random matrix, and there is.&#36;\rho(A)=1.&#36;</li>
<li>&#36;n&#36;The step-by-step array A is a sufficient condition for the random matrix.&#36;x=(1,\cdots,1)^{\mathrm{T&#125;&#125;\in\mathbb{R}^{n}&#36;Yes&#36;A&#36; Characteristic vector corresponding to feature value 1, i.e.&#36;Ax=x.&#36;</li>
<li>The accumulation of the same random matrix or the random matrix</li>
<li>Set &#36;n&#36; Step Non-negative Matrix &#36;A&#36; & Radius of \\rho&gt;0&#36;,且有 &#36;x=(x_1,\cdots,x_n)^{\mathrm{T&#125;&#125;&gt;0&#36;,则矩阵&#36;A&#36; 能相似于数  &#36;\rho ( A)&#36;与某个随机矩阵 &#36;P&#36; 的乘积，即 &#36;A=D(\rho(A)P)D^{-1}&#36; 其中 &#36;D=\operatorname{diag}(x_1,\cdots,x_n).&#36;即&#36;(D^-1AD) / \\rho(A) is a random matrix.</li>
</ul>
<p>Theorem (resistence of random array sequences): set&#36;A&#36;For an unacceptable random matrix, the limit&#36;\lim_m\to\infty A^m&#36;The condition of existence is sufficient. A is the original matrix.</p>
<p>The two-random matrix is a special type of random matrix, and it therefore has all the characteristics of a random matrix, and the following results.</p>
<p>Theorem: Set &#36;A\in\mathbb{R}^{n\times n}&#36;It's a double random matrix, then.</p>
<ul>
<li>&#36;\rho(\boldsymbol{A})=1&#36;and &#36;\boldsymbol x=(1,\cdots,1)^\mathrm{T}&#36; Yes.&#36;\boldsymbol{A}&#36; and &#36;A^\mathrm{T}&#36; (a) Special vectors corresponding to characteristic value 1;</li>
<li>&#36;\parallel A\parallel_2\geqslant1.&#36;</li>
</ul>
<h3>Mono-Class</h3>
<p>This section briefly describes a matrix of this type.&#36;A&#36;It's a counter-argument.&#36;A^{-1}&#36;It's a non-negative matrix -- a single-tone matrix, and it's a reference to the solution linear equation group.</p>
<p>Definitions: Establishment &#36;A\in\mathbb{R}^{n\times n}&#36;, if it's against matrix &#36;\mathbf{A}^-1\geqslant0&#36;, or &#36;\mathbf{A}&#36; For the single-tone matrix.</p>
<p>Theorem (distinguishment):&#36;A\in\mathbb{R}^{n\times n}&#36;, then&#36;A&#36;The necessary conditions for a single-tangular matrix are sufficient:&#36;Ax\geqslant0&#36;Launch&#36;x\geqslant0&#36;Here. &#36;x&#36; It's a column vector.</p>
<p>Theorem: Set&#36;A&#36;For a single-tangular matrix, if vectors are found&#36;x^\prime=(x_1^{\prime},\cdots,x_n^{\prime})^{\mathrm{T&#125;&#125;&#36;and&#36;x^{\prime\prime}=(x^{\prime\prime}_1,\cdots,x^{\prime\prime})^{\mathrm{T&#125;&#125;&#36;Separated &#36;Ax^{\prime}\leqslant b,Ax^{\prime\prime}\geqslant b&#36;, and then there's an estimate.
&#36;&#36;x^{\prime}\leqslant\bar{x}\leqslant x^{\prime\prime}&#36;&#36;
or
&#36;&#36;x_i^{\prime}\leqslant\tilde{x}_i\leqslant x_i^{\prime\prime},\quad i=1,\cdots,n.&#36;&#36;</p>
<p><strong>The meaning of the theorem is to help us estimate the upper and lower horizons of the equation.</strong></p>
<h3>M and H matrix</h3>
<p>Definitions: Establishment &#36;A\in\mathbb{R}^{n\times n}&#36;and can be expressed as
&#36;A=si-B,\quad s&gt;0, \d B\geqslant0.&#36;0.00
If &#36;s\geqslant\rho(B),则称A&#36;as M matrix; if s&gt;&#36;\rho(B),则称A&#36;As Non-Tarctic M Matrix</p>
<p>For a better discussion.&#36;M&#36;The nature of the matrix, we introduce it.&#36;Z&#36;The models are:</p>
<p>Set &#36;A=(a {j})<em>You're not gonna get a chance to get a job.
&#36;a</em>\leqslant0:, \erad i\neq j:, i:,j=1:,2: \cdots, n: &#36;,&#36;
And then, "Could"&#36;A&#36; Yes&#36;{Z}&#36;Type matrix, all&#36;\textbf{ }n&#36; Step&#36;Z&#36; The assembly marks of the model matrix &#36;Z^{n\times n}&#36;It's clear that M's matrix is the special case of the Z-type matrix.</p>
<p>Theorem: Set &#36;A\in Z^{n\times n}&#36;is non-generic M matrix, and &#36;D\in Z^{n\times n}&#36;Satisfied &#36;D\geqslant A&#36;, then</p>
<ul>
<li>&#36;\boldsymbol{A}^{-1}&#36;and &#36;\boldsymbol D^{-1}&#36;exist, and &#36;\boldsymbol{A}^- 1\geqslant \boldsymbol{D}^{- 1}\geqslant 0&#36; ;</li>
<li>&#36;D&#36;is the positive value for each of the physical characteristics;</li>
<li>&#36;det D\geqslant\det A&gt;0.&#36;</li>
</ul>
<p>Theorem: non-genocidious&#36;M&#36;The matrix has many conditions of parity. &#36;A\in Z^{n\times n}&#36; The next issue is the same price.</p>
<ul>
<li>&#36;\text{A 为非奇异 M 矩阵}&#36;</li>
<li>&#36;\text{若 }B\in Z^{n\times n}\text{且 }B\geqslant A,\text{则 }B\text{ 非奇异}&#36;</li>
<li>&#36;A\text{ 的任意主子矩阵的每一个实特征值为正数}&#36;</li>
<li>&#36;\text{A 的所有主子式为正数}&#36;</li>
<li>&#36;\text{对每个 }k(1\leqslant k\leqslant n),A\text{ 的所有 }k\text{ 阶主子式之和为正数}&#36;</li>
<li>&#36;\text{A 的每一个实特征值为正数}&#36;</li>
<li>&#36;\text{Existing}A split of A\text{P-Q,\text{P}&lt;1&#36;</li>
<li>&#36;A\text{ 非奇异,且 }A^{-1}\geqslant0.&#36;</li>
</ul>
<p>Theorem: Set&#36;A\in Z^{n\times n}&#36;Right, then.&#36;A&#36;The necessary condition for the non-unique M matrix is&#36;A&#36;Sets the positive matrix</p>
<p>Theorem: Set&#36;A,B\in\mathbb{R}^{n\times n}&#36;is non-unique M matrix, then&#36;AB&#36;The necessary condition for the non-unique M matrix is&#36;AB\in Z^n\times n.&#36;</p>
<hr>
<p>Let's talk about something next.&#36;M&#36;The Matrix Problem</p>
<p>Theorem: Set &#36;A\in Z^{n\times n}&#36; The next issue is the same price.</p>
<ul>
<li>&#36;A是M矩阵&#36;</li>
<li>*Varepsilon for each&gt;0, A+\varepsilon I'm a non-genocious M matrix</li>
<li>&#36;\text{A 的任意主子矩阵的每个实特征值非负}&#36;</li>
<li>&#36;\text{A的所有主子式非负}&#36;</li>
<li>&#36;\text{对每个 }k=1,2,\cdotp\cdotp\cdotp,n,A\text{ 的所有 }k\text{ 阶主子式之和为非负实数}&#36;</li>
<li>&#36;\text{A 的每个实特征值非负}&#36;</li>
</ul>
<p>Theorem: Set&#36;A&#36;It's a strange thing to be expected.&#36;M&#36;Matrix</p>
<ul>
<li>&#36;\mathrm{rank}A=n-1&#36;</li>
<li>&#36; positive vector existsx&gt;0, make Ax=0&#36;</li>
<li>All the Gods' Queues are non-surreal M {, \text {specially }a n&gt;0\mathrm{(1}\leqslant i\leqslant n)&#36;</li>
<li>&#36;\text{对任意 }x\in\mathbb{R}^n,\text{若 }Ax\geqslant0,\text{则 }Ax=0&#36;</li>
</ul>
<hr>
<p>Next&#36;n&#36;Array&#36;A&#36;Extension to Retrace Matrix, and use&#36;A&#36;..in the cell of the cell, you can see that the elemental model is a new comparison matrix. &#36;{H}(\boldsymbol{A})&#36;if &#36;H(\boldsymbol{A}&#36;It's not weird. &#36;M&#36; Matrix, define &#36;\boldsymbol{A}&#36; Yes &#36;H&#36; Matrix.</p>
<p>Definitions: Establishment &#36;A=(a_{ij})\in\mathbb{C}^{n\times n}&#36;and set
&#36;&#36;\mathrm{H}(\mathbf{A})=(m_{ij})\in\mathbb{R}^{n\times n},&#36;&#36;
of which</p>
<p>&#36;&#36;m_{ij}=\begin{cases}\quad\mid a_{ij}\mid,\quad j=i,\-\mid a_{ij}\mid,\quad j\neq i,\end{cases}i,j=1,\cdotp\cdotp\cdotp,n,&#36;&#36;
&#36;H(A)&#36;Called &#36;A&#36; - The comparative matrix.</p>
<p>Definitions: Establishment&#36;A\in\mathbb{C}^{n\times n}&#36;if&#36;A&#36;. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .&#36;H(A)&#36;It's a non-genocious M matrix, which is called&#36;A&#36;For the Fictator. &#36;H&#36; Matrix, abbreviations &#36;H&#36; Matrix</p>
<p>Here's a brief description. &#36;H&#36; Some of the nature of the matrix.</p>
<p>Theorem: Set &#36;A,\boldsymbol{B}\in\mathbb{C}^{n\times n},A&#36; It's a strange thing. &#36;M&#36; Matrix,&#36;H(B)\geqslant\mathcal{A}&#36;, then</p>
<ul>
<li>&#36;B&#36;Yes. &#36;H&#36; Matrix;</li>
<li>&#36;B&#36;It's not weird, and...&#36;A^-1\geqslant|B^{-1}|\geqslant0;&#36;</li>
<li>&#36;\mid\det B\mid\geqslant det A&gt;0&#36;</li>
</ul>
<p>Theorem: Set &#36;A\in C^{n\times n}&#36; It's not like that.</p>
<ul>
<li>&#36;\mathcal{H}(A)\in\mathbb{Z}^{n\times n}&#36;</li>
<li>&#36;\mathcal{H}(A)=A\text{ 的充分必要条件是 }A\in\mathbb{Z}^{n\times n}&#36;</li>
<li>&#36;A\text{ 为 M 矩阵的充分必要条件是 H}(A)=A,\text{且 A 为 H 矩阵}&#36;</li>
<li>H(A) can be expressed as the difference between a non-negative diagonal matrix and a non-negative matrix with zero-reverse angle:&#36;&#36;H( \boldsymbol A) = \mid diag(a_{11},\cdots,a_{nn})\mid-[\mid\boldsymbol{A}\mid-\mid diag(a_{11},\cdots,a_{nn})\mid]&#36;&#36;Here.&#36;|\boldsymbol{X}|\equiv[|x_{ij}|]&#36;Express Matrix&#36;\boldsymbol{X}=(x_ij)\in\mathbb{C}^{n\times n}&#36;, which is the basis for the total value of the matrix;</li>
<li>If&#36;A&#36;It's the M matrix.&#36;&#36;\mathbf{A}=\operatorname{diag}(a_{11},\cdotp\cdotp\cdotp,a_{nn})-\begin{bmatrix}\operatorname{diag}(a_{11},\cdotp\cdotp\cdotp,a_{nn})-\mathbf{A}\end{bmatrix}&#36;&#36;</li>
</ul>
<h3>T and Hinkler.</h3>
<p>We'll meet the following type of matrix in many areas.
&#36; \matbf{&amp;a_{-1}&amp;a_{-2}&amp;\cdots&amp;a_{-n+1}\\a_1&amp;a_0&amp;a_{-1}&amp;\cdots&amp;a_{-n+2}\\a_2&amp;a_1&amp;a_0&amp;\cdots&amp;a_{-n+3}\\vdots&amp;\vdots&amp;\vdots&amp;&amp;\vdots\\a_{n-2}&amp;a_{n-3}&amp;a_{n-4}&amp;\cdots&amp;a_{-1}\\a_{n-1}&amp;a_{n-2}&amp;a_{n-3}&amp;\cdots&amp;I'm sorry, I'm sorry, but I'm sorry, but I'm sorry.
Any straight line parallel to the main diagonal line is identical, and we call it the T-format.</p>
<p>The nature of the T-format is not well studied, so the focus is moving towards the next form of the matrix.
&#36;&#36;\bardsymbol{H}<em>{n+1}=\begin{bmatrix}a_0&amp;a_1&amp;a_2&amp;\cdots&amp;a_n\\a_1&amp;a_2&amp;a_3&amp;\cdots&amp;a</em>{n+1}\\a_2&amp;a_3&amp;a_4&amp;\cdots&amp;a_{n+2}\\vdots&amp;\vdots&amp;\vdots&amp;&amp;\vdots\\a_n&amp;a_{n+1}&amp;a_{n+2}&amp;\cdots&amp;A \end{bmatrix} &#36;
Any straight line parallel to the side-drive is identical. We call this the Hankel Matrix, which is a non-unusual matrix.</p>
<p>It's possible to verify directly that the T-format and the Henkel-format can be converted. In fact, the T-format is a very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very &#36;A&#36;The Hankel Matrix is&#36;H_{n+1}&#36;, and then the matrix.
&#36;&#36;\bardsymbol=begin{bmatrix}&amp;&amp;&amp;1\&amp;1&amp;&amp;\1&amp;&amp;&amp;\end{bmatrix}&#36;&#36;
乘矩阵 &#36;H_{n+1}&#36;,其结果 &#36;JH_{n+1}&#36;或 &#36;H_{n+1}J&#36; 都是 T 矩阵，且有
&#36;&#36;== sync, corrected by elderman ==
On the contrary, use&#36;J&#36;Multiply T Matrix&#36;A&#36;, then&#36;JA&#36;or&#36;AJ&#36;It's all the Hankel Matrix.</p>
<h2>Special volume</h2>
<p>The special size of the matrix, or the study matrix? &#36;AB&#36; But no more requests at this time. &#36;A&#36; Number of columns equal to &#36;B&#36; . This special amount, which is not bound by the matrix, has a lot of simplicity in many places.</p>
<h3>Cronekkeke.</h3>
<p>Two matrices were defined earlier&#36;A&#36;and&#36;B&#36;Product&#36;AB&#36; ♪ It's asking ♪&#36;A&#36;. The number of columns must equal&#36;B&#36;the number of lines and columns of the matrix.</p>
<p>Definitions: Establishment &#36;A=(a_{ij})\in\mathbb{C}^{m\times n},\boldsymbol{B}=(b_{ij})\in\mathbb{C}^{p\times q}&#36;, the following block matrix is called
&#36;A\otimes B=\begin{vmatrix}a&amp;a_{12}B&amp;\cdots&amp;a_{1n}B\a_{21}B&amp;a_{22}B&amp;\cdots&amp;a_{2n}B\\vdots&amp;\vdots&amp;&amp;\vdots\a_{m1}B&amp;a_{m2}B&amp;\cdots&amp;A B\m}vmatrix}In the middle of the night, you're gonna have to go to the hospital.
Yes&#36;A&#36;Kronecker, or&#36;A&#36;and &#36;B&#36; The volume of the volume, or the volume, which is as follows:&#36;A\otimes\boldsymbol{B}=(a_{ij}\boldsymbol{B})_{mp\times nq}.&#36; That's...&#36;A\otimes B&#36;It's one.&#36;m\times n&#36;The block matrix, the last one.&#36;mp\times nq&#36;Matrix.</p>
<p>Apparently, Cronekkeh didn't meet the exchange, but the finals were the same.</p>
<p>Easy to verify. Cronek has a down-to-down calculation.</p>
<ul>
<li>&#36;k(A\otimes B)=kA\otimes B=A\otimes kB,k\in\mathbb{C}&#36;</li>
<li>&#36;(A+B)\otimes C=A\otimes C+B\otimes C&#36;</li>
<li>&#36;(A\otimes B)\otimes C=A\otimes(B\otimes C)&#36;</li>
</ul>
<hr>
<p>Theorem: set &#36;A=(a {j})<em>{m\times n},\boldsymbol{B}=(b</em>{ij})<em>{s\times r},\boldsymbol{C}=(c</em>{ij})<em>{n\times p},\boldsymbol{D}=(d</em>r\times l}&#36;, then
&#36;&#36;(A\otimes B)(C\otimes D)=AC\otimes BD.&#36;&#36;</p>
<p>Inference: set &#36;A=(a {j})<em>{m\times n},\boldsymbol{B}=(b</em>s\times r}&#36;, then
&#36;&#36;A\otimes B=(A\otimes I_n)(I_m\otimes B)=(I_m\otimes B)(A\otimes I_n)&#36;&#36;</p>
<p>Theorem: Set &#36;A=(a_{ij})_m\times n&#36;, then &#36;rank(\boldsymbol{A})\leqslant1\Leftrightarrow\boldsymbol{A}&#36; It's a Cronek, a row and a column.</p>
<p>Theorem: Set up \\bardsymbol{A}(a ijj})<em>{m\times n},\boldsymbol{B}=(b</em>p\times q}&#36;, then
&#36;&#36;00begin{aligned} (A\omiesB)^mathrm^T}&amp;=A^\mathrm{T}\otimes B^\mathrm{T}:,\(A\otimes B)^\mathrm{H}&amp;♪ The world is so full of shit ♪
The Cronecker or the Symmetric Matrix (Ermet Matrix) is easily launched from this point on.</p>
<p>Theorem: Set&#36;A,B&#36;Both&#36;m&#36;Them of steps&#36;n&#36;, and then&#36;A\otimes B&#36;And it's a reversible matrix.
&#36;&#36;(A\otimes B)^{-1}=A^{-1}\otimes B^{-1}.&#36;&#36;</p>
<p>Theorem: Set up \\bardsymbol{A}(a ijj})<em>{m\times n},\boldsymbol{B}=(b</em>p\times q}&#36;, then
&#36;&#36;\operatorname{rank}(\boldsymbol{A}\otimes\boldsymbol{B})=\operatorname{rank}(\boldsymbol{A})\operatorname{rank}(\boldsymbol{B}).&#36;&#36;
&#36;&#36;\mathrm{tr}(A\otimes B)=\mathrm{tr}A\mathrm{tr}B&#36;&#36;</p>
<p>Theorem: Set&#36;x_1,x_2,\cdots,x_n&#36;Yes.&#36;n&#36;It's not about linear.&#36;m&#36;Vixel vector&#36;,y_1,y_2,\cdots,y_q&#36;Yes.&#36;q&#36;It's not about linear.&#36;p&#36;a wi-View vector, then&#36;nq&#36;individual &#36;m p&#36; Vixel vector&#36;\mathbf{x}_i\otimes\mathbf{y}_j(i=1,\cdotp,n;j=1,\cdotp\cdotp,q)&#36;It's not linear, it's not linear.</p>
<p>Theorem: Set&#36;A,B&#36;Both&#36;m,p&#36;The stairwell, and there is.
&#36;&#36;A\otimes B=&amp;|B|^m.\end{array}&#36;&#36;</p>
<p>Theorem: set &#36;A=(a {j})<em>{m\times p},\boldsymbol{B}=(b</em>p\times n}&#36;, there is
&#36;&#36;(AB)^{[k]}=A^{[k]}B^{[k]}&#36;&#36;</p>
<p>Theorem: Set &#36;\lambda_1,\lambda_2,\cdots,\lambda_m&#36; Yes.&#36;A_m\times m&#36;Yes. &#36;m&#36; Characteristic Values &#36;,\mu_1,\mu_2,\cdots,\mu_p&#36; Yes.&#36;B_{p\times p}&#36;Yes. &#36;p&#36; A characteristic value, then.&#36;A\otimes B&#36;Yes. &#36;m p&#36; The character value is&#36;&#36;\lambda_i\mu_j(i=1,2,\cdotp\cdotp\cdotp,m;j=1,2,\cdotp\cdotp\cdotp,p).&#36;&#36;</p>
<p>Theorem: Set&#36;A&#36;Yes&#36;m&#36;Step Matrix, B is&#36;n&#36;, and there is&#36;A\otimes B&#36;Similar to&#36;B\otimes A&#36;</p>
<p>Theorem: Set&#36;f(x,y)=\sum_{i,j=0}^{r}a_{ij}x^{i}y^{j}&#36;It's a variable.&#36;x,y&#36;, for&#36;A\in\mathbb{C}^m\times m&#36;, &#36;B\in\mathbb{C}^{n\times n}&#36;Definitions&#36;mn&#36;Structometer:
&#36;&#36;f(A,B)=\sum_{i,j=0}^p\alpha_{ij}A^i\otimes B^j.&#36;&#36;
If&#36;A&#36;and&#36;B&#36;The feature value is&#36;\lambda_1,...,\lambda_m&#36;and&#36;\mu_1,...,\mu_n&#36;, they're responding to the character vectors.&#36;x_{1},\cdots,x_{m}&#36;and&#36;y_1,\cdots,y_n&#36;,the matrix &#36;f(A,B)&#36;The characteristic value is &#36;f(\lambda_r,\mu_s)&#36;, and the corresponding &#36;f(\lambda_r,\mu_s)&#36;The characteristic vector is&#36;x_r\otimes y_s( r= 1, . . . , m;&#36; &#36;s= 1, . . . , n) .&#36;</p>
<p>Based on the theory, we can easily infer from the following.</p>
<p>Insumption 1: We take&#36;f(x,y)=xy&#36;  &#36;\boldsymbol{A}\otimes\boldsymbol{B}&#36; The characteristic value is &#36;mn&#36; Number &#36;\lambda_r\mu_s&#36;  The corresponding characteristic vector is &#36;x_r\otimes y_s&#36;</p>
<p>Inference 2:&#36;f(x,y)=x+y&#36; Which means... &#36;f(x,y)=xy^0+x^0y&#36;  There is. &#36;A\otimes I_n+I_m\otimes B&#36;The feature value is&#36;\lambda_r+\mu_s&#36; Characteristic vector &#36;x_r\otimes y_s&#36;</p>
<p>We call it the Matrix.
&#36;&#36;A\otimes I_n+I_m\otimes B&#36;&#36;
Yes&#36;A&#36;and&#36;B&#36;Cronek and</p>
<p>Finally, we'll introduce a special matrix.</p>
<p>Definition: Square array with 1 or 1 element &#36;H\in\mathbb{R}^{n\times m}&#36;If there is
&#36;&#36;HH^\mathrm{T}=nI_n,&#36;&#36;
Name&#36;H&#36;Yes&#36;n&#36;The Hadamard Matrix.</p>
<p>Theorem: Set&#36;H_m&#36; and&#36;H_n&#36; The Adama Matrix, the Matrix. &#36;H_m\otimes H_n&#36; Yes &#36;mn&#36; The Hadamard Matrix.</p>
<h3>Adama Kei.</h3>
<p>The Adama multiplication is far simpler than the usual matrix multiplication, but it is not widely understood ... It appears in many issues, so we are here to discuss him.</p>
<p>Definitions: Establishment&#36;A=(a_{ij}),{B}=(b_{ij})\in\mathbb{C}^{m\times n}.&#36;Use&#36;A^\circ B&#36;Organisation&#36;A&#36;The equivalent of B is multiplied by the corresponding elements&#36;m\times n&#36;Matrix:
&#36;A\cB=\begin{bmatrix}a&amp;a_{12}b_{12}&amp;\cdots&amp;a_{1n}b_{1n}\a_{21}b_{21}&amp;a_{22}b_{22}&amp;\cdots&amp;a_{2n}b_{2n}\\vdots&amp;\vdots&amp;&amp;\vdots\a_{m1}b_{m1}&amp;a_{m2}b_{m2}&amp;\cdots&amp;a mn}b mn}end{bmatrix}, &#36;&#36;
Adamaje, also known as Shuljee.</p>
<p><strong>Apparently, Adamaja needs two matrixes, and he's a swap. &#36;A\circ B =B\circ A&#36;</strong></p>
<p>Theorem: Set &#36;A,{B},C\in\mathbb{C}^{m\times n}.&#36; The following is the nature of the operation of the Adamar stock.</p>
<ul>
<li>&#36;A\circ(B+C)=A\circ B+A\circ C&#36;</li>
<li>&#36;A\circ(B\circ C)=(A\circ B)\circ C&#36;</li>
<li>&#36;(A\circ B)^\mathrm{T}=A^\mathrm{T}\circ B^\mathrm{T}&#36;</li>
<li>&#36;(A\circ B)^{\mathrm{H&#125;&#125;=A^{\mathrm{H&#125;&#125;\circ B^{\mathrm{H&#125;&#125;&#36;</li>
<li>&#36;\text{如果 }A\text{ 和 }B\text{ 是自伴矩阵(即埃尔米特矩阵)},\text{那么 }A\circ B\text{ 也是自伴矩阵}&#36;</li>
<li>&#36;\text{如果 }A\text{ 和 }B\text{ 是斜自伴(即反埃尔米特)矩阵},\text{那么 }A\circ B\text{ 是自伴矩阵}&#36;</li>
<li>&#36;\text{如果 }A\text{ 是自伴矩阵 },B\text{ 是斜自伴矩阵 },\text{那么 }A\circ B\text{ 是斜自伴矩阵}&#36;</li>
<li>&#36;\mathrm{rank}(A\circ B)\leqslant(\mathrm{rank}A)(\mathrm{rank}B)&#36;</li>
<li>&#36;\text{若 }A,B\text{ 是半正定矩阵},\text{则 }A\circ B\text{ 也是半正定矩阵}&#36;</li>
<li>&#36;\text{若 }B\text{ 是正定矩阵 },A\text{ 是半正定矩阵且无零对角元素 },\text{则 }A\circ B\text{ 是正定矩阵}&#36;</li>
<li>&#36;若A和B都是正定矩阵,则A\circ B也是正定矩阵&#36;</li>
</ul>
<p>Theorem: Set &#36;A,B\in\mathbb{C}^{n\times n}&#36;It's a semi-positive matrix, and it's set up.
&#36;&#36;\lambda_{\min}(A\circ B)\geqslant\lambda_{\min}(A)\lambda_{\min}(B)&#36;&#36;
and
&#36;&#36;\lambda_{\max}(A\circ B)\leqslant\lambda_{\max}(A)\lambda_{\max}(B):,&#36;&#36;
of which &#36;\lambda_{\min}(\boldsymbol{A})&#36;and &#36;\lambda_{\max}(\boldsymbol{A})&#36;Separately &#36;\boldsymbol{A}&#36; . The minimum feature value and the maximum feature value.</p>
<p>and</p>
<p>Theorem: Set &#36;A,B\in\mathbb{C}^{n\times n}&#36;It's a semi-positive matrix, and it's set up.
&#36;&#36;\lambda_{\min}(A\circ B)\geqslant\lambda_{\min}(AB^{\mathrm{T&#125;&#125;)&#36;&#36;
&#36;&#36;\lambda_{\min}(A\circ B)\geqslant\lambda_{\min}(AB).&#36;&#36;</p>
<h3>Inflat</h3>
<p>Definitions: Establishment &#36;\boldsymbol{A}=(a_{ij}),\boldsymbol{B}=(b_{ij})\in\mathbb{C}^{m\times n}.&#36;You're the one who's gonna get you.
&#36;c ij}=begin{cases}\quad a}b}ii}:,\quad&amp;j=i:,\[2ex]-a_{ij}b_{ij}:,\quad&amp;j\neq i:,\quad&amp;i =1: \cdots, m: \qad j=1: \cdots, n. \end{cases}
Remember&#36;A\star B=(c_{ij})\in\mathbb{C}^{m\times n}&#36;and called it&#36;A&#36;and&#36;B&#36;Fan accumulation.</p>
<p>Easy to see:<strong>Inverses are a variation of Adam's stock.</strong></p>
<p>Theoretically: Adama stock on both the accumulation and the non-negative matrix has the following basic properties.</p>
<ul>
<li>If&#36;A,B\in\mathbb{R}^{n\times n}&#36;is the M matrix, then&#36;A\star B&#36;It is also the M Matrix;</li>
<li>If &#36;A,\boldsymbol{B}\in\mathbb{C}^{n\times n}&#36;is the H matrix, then &#36;A\star\boldsymbol{B}&#36; It is also a H matrix.&#36;A\circ\boldsymbol{B}&#36; It's not weird.</li>
</ul>
<p>Theorem: Set&#36;A,B\in\mathbb{R}^{n\times n},A\geqslant0,B\geqslant0,则&#36;</p>
<ul>
<li>&#36;A^{\circ}B\geqslant0&#36;In other words, the non-negative matrix category is closed under Adama ' s stock;</li>
<li>&#36;\rho ( A^{\circ }B) \leqslant \rho ( A) \beta ( B) .&#36;</li>
</ul>
<p>Theorem: Set &#36;A,\boldsymbol{B}\in\mathbb{R}^{n\times n}&#36;is the M matrix, then &#36;A\circ\boldsymbol{B}^{-1}&#36;And M matrix.</p>
<h3>Cronekkekekeke applications</h3>
<p>Using the matrix 's Cronek-Judge nature, we can easily study the linear matrix equation.
&#36;&#36;A_1XB_1+A_2XB_2+\cdots+A_pXB_p=C&#36;&#36;
In fact, he can convert to a normal linear equation.
&#36;&#36;Gx=c&#36;&#36;
That's the question this section wants to discuss.</p>
<h4>& Aligning the matrix</h4>
<p>Definitions: set up at \bardsymbol{A}=(a ijj})<em>{m\times n}&#36;,将&#36;\bardsymbol<strong>Lines by Row</strong>Got it.&#36;mn&#36;A villary vector, which is called&#36;A&#36;♪ Straighten up, remember ♪&#36;\vec{A}&#36;that is
&#36;\vec{A}=(a)</em>{11}, a cdotp\cdotp\cdotp}, a 2}, a cdotp\cdotp\cdotp, a 2}, \cdotp\cdotp\cdotp, a m m2}, \cdotp\cdotp\cdotp, a\cdotp\cdotp\cdotp, a^mn} ^matthrm^T}.&#36;&#36;
It's easy to know. The straight line is linear. &#36;\overrightarrow{A+B}=\vec{A}+\vec{B},\quad\vec{kA}=k\vec{A}&#36;</p>
<p>Theoretically: With regard to straight counter-counts, we can give the following continuous proofs:</p>
<ol>
<li>&#36;xy^\mathrm{T}=x\otimes y,\text{其中 }x,y\text{ 为 }n\text{ 维列向量}&#36;</li>
<li>&#36;\boldsymbol{E}_{ij}=\boldsymbol{e}_i\boldsymbol{e}<em>j^\mathrm{T}&#36;,其中&#36;\boldsymbol E</em>{ij}&#36;表示(&#36;i,j)&#36;元素为 l,其余元素为 0 的 &#36;m\times n&#36; 阶矩阵，&#36;\boldsymbol e_i&#36; 表示第 &#36;i. &#36;1 for elements and 0 for the rest of elements;</li>
<li>&#36;Ae_i=\begin{bmatrix}a_{1i}\\a_{2i}\\vdots\\a_{mi}\end{bmatrix}&#36;</li>
<li>&#36;e_j^\mathrm{T}A=(a_{j1},a_{j2},\cdots,a_{jn})&#36;</li>
<li>&#36;\vec{E}_{ij}=e_i\otimes e_j&#36;</li>
</ol>
<p>Theorem: set-up &#36;A=(\mu ij})<em>{m\times n},\boldsymbol{B}=(b</em>{ij})<em>{n\times p},\boldsymbol{C}=(c</em>{ij})<em>&#36;, then
&#36;&#36;\overrightarrow{ABC}=(A\otimes C^{\mathrm{T&#125;&#125;)\vec{B}.&#36;&#36;
Inference: set-A=(\mu)</em>{ij})<em>{m\times n},\boldsymbol{B}=(b</em>{ij})<em>{n\times p},\boldsymbol{X}=(x</em>p\times q}&#36;, then</p>
<ul>
<li>&#36;\overrightarrow{AX}=(A\otimes I_{n})\vec{X}&#36;</li>
<li>&#36;\overrightarrow{XB}=(I_m\otimes B^\mathrm{T})\vec{X}&#36;</li>
<li>&#36;\overrightarrow{AX+XB}=(A\otimes I_n+I_m\otimes B^\mathrm{T})\vec{X}.&#36;</li>
</ul>
<h4>The solution of the linear matrix equation</h4>
<p>Theorem: Matrix&#36;X\in\mathbb{C}^{m\times n}&#36;It's the matrix equation.&#36;A_1XB_1+A_2XB_2+\cdots+A_pXB_p=C&#36;The only necessary condition for his release is&#36;x=\vec{X}&#36;As a normal linear equation group
&#36;&#36;Gx=c&#36;&#36;
♪ The way it's gonna be ♪&#36;G=\sum_{i=1}^pA_i\otimes B_i^\mathrm{T},c=\vec{C}.&#36;</p>
<hr>
<p>We'll discuss a special case.
&#36;&#36;AX+XB=C&#36;&#36;
Theorem: The only solution to the equation of the matrix described above&#36;X\in\mathbb{C}^{m\times n}&#36;♪ And the only thing that matters ♪&#36;A&#36;and &#36;-B&#36; No same feature value, i.e.
&#36;&#36;\lambda_i+\mu_j\neq0,\quad i=1,\cdotp\cdotp\cdotp,m,\quad j=1,\cdotp\cdotp\cdotp,n.&#36;&#36;</p>
<hr>
<p>Study the equation
&#36;&#36;X+AXB=C&#36;&#36;
Theorem: The only solution to the equation of the matrix described above &#36;x\in\mathbb{C}^{m\times n}&#36;♪ And the only thing that matters ♪&#36;\lambda_i\mu_j\neq-1(i=1,\cdots&#36;,
&#36;m;j=1,\cdots,n&#36;)&#36;,\lambda_i&#36; and&#36;\mu_j&#36; Both&#36;A&#36; and&#36;B&#36; . The characteristic value of the</p>
