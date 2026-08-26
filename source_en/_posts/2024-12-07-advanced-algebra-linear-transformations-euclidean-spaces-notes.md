---
title: 'Advanced Algebra: Linear Transformations and Euclidean Spaces'
title_zh: 高等代数：线性变换与欧式空间
date: 2024-12-07 16:19:23 +0800
permalink: /blog/2024/12/07/advanced-algebra-linear-transformations-euclidean-spaces-notes/
categories:
- Mathematics
- Algebra & Matrix Theory
tags:
- Linear Algebra
excerpt: Covers linear transformations, eigenvalues, Jordan normal form, Euclidean spaces, unitary spaces, and orthogonal
  transformations.
description: Covers linear transformations, eigenvalues, Jordan normal form, Euclidean spaces, unitary spaces, and orthogonal
  transformations.
lang: en
translation_key: 2024-12-07-advanced-algebra-linear-transformations-euclidean-spaces-notes
translation_status: machine
translation_source_hash: c2c805014f03d100d5f80b40eb0017489f8f20922c2eddaf1ca1745db9794369
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<h2>Linear transformation</h2>
<p>vector group</p>
<h3>Definition of linear transformation</h3>
<p>And at the end of the last chapter, we explained the essence of a linear space with the same construction. But it is also important to study the link between linear space. This is reflected in a map between linear spaces.</p>
<p>Maps from linear to linear space are called mutations. The simplest linear variation in the change is what this chapter is about.</p>
<p><strong>The same is the construction of linear space.&#36;V&#36;To the classic linear space&#36;P^n&#36;We're focusing in the same section on studying the nature of space rather than changing itself.</strong></p>
<p>Definitions: For several domains&#36;P&#36; Linear Space Up &#36;V&#36; If for anything&#36;\alpha,\beta \in V,p\in P&#36; Both.&#36;A(a+\beta)=A(\alpha)+A(\beta)&#36;    &#36;A(k\alpha)=kA(\alpha)&#36;Constantly known as transformation&#36;A&#36; It's a linear shift, which means...<strong>Changes in the additions and multipliers</strong></p>
<p>Here are some examples of linear variations.</p>
<ul>
<li>Constant transformation &#36;A(\alpha)=\alpha&#36;</li>
<li>Zero Change &#36;A(\alpha)=0&#36;</li>
<li>Multiplication &#36;A(\alpha)=kA(\alpha)&#36;</li>
<li>Micro-modification (division to original function)</li>
<li>Numerical transformation (inflexible points for original functions)</li>
<li>Vector transformation &#36; (\begin{array}x^({\prime}\end{array})=(\begin{array}{cos\theta}&amp;-\sin\theta\\sin\theta&amp;\cos\theta\end{array})(\begin{array}{c}x\y^{\prime}\end{array})&#36;</li>
</ul>
<p>Linear transformations have the following properties:</p>
<ul>
<li>&#36;A(0)=0&#36;</li>
<li>&#36;A(-\alpha)=-A(\alpha)&#36;</li>
<li>Conservative group &#36;\beta=k_{1}\alpha_{1}+\cdots+k_{1}\alpha_{n}\Rightarrow A(\beta)=kA(\alpha_{1})+\cdots+k_{n}A(\alpha_{n})&#36;</li>
<li>Linear transformation to map linearly relevant vectors to linearly relevant vectors Group</li>
</ul>
<p>Theoretically: In limited-dimensional linear space, single, full and double are equal; they all map linearly related vectors into linearly relevant vectors, and it is irrelevant to map linearly unrelated vectors.</p>
<h3>The operation of linear transformations</h3>
<p>Now we know how to operate a linear transformation of this linear space.</p>
<p>Definitions (multiplication):&#36;AB(\alpha)=A(B(\alpha))&#36; It's a linear multiplication.&#36;A,B&#36;It's a linear shift.</p>
<p>There's a multiplication for linear variations.</p>
<ul>
<li>The product or linear transformation of two phenomena</li>
<li>Applicable to integration rate but not to exchange Lew</li>
<li>Define Unit Change&#36;\varepsilon&#36;  If there is. &#36;\varepsilon A=A\varepsilon=A&#36;</li>
</ul>
<p>Definitions (additional):&#36;(A+B)(\alpha)=A(\alpha)+B(\alpha)&#36;It's a linear shift.</p>
<p>There are additions to linear variations.</p>
<ul>
<li>Linear change and linear change.</li>
<li>Apply to integration rates, apply exchange laws</li>
<li>For zero-change zero. &#36;A+0=A&#36;</li>
<li>You can define negative changes on this basis.&#36;(-A)(\alpha)=-A(\alpha)&#36;  He's also a linear shift.</li>
<li>Multiplication and addition&#36;A(B+C)=AB+AC&#36;</li>
</ul>
<p>Definition (number multiplier):&#36;(kA)(\alpha)=kA(\alpha)&#36; It's a linear mutation.</p>
<p>There's a number multiplication for linear transformations.</p>
<ul>
<li>Linear mutation or linear multiplication</li>
<li>&#36;(kl)A=k(lA)&#36;</li>
<li>&#36;(k+l)A=kA+lA&#36;</li>
<li>&#36;k(A+B)=kA+kB&#36;</li>
<li>&#36;1A=A&#36;</li>
</ul>
<p>Definition (reverse variant):&#36;\sigma&#36; Yes. &#36;V&#36; A linear shift on if&#36;\sigma\tau=\tau\sigma=\text{单位变换}&#36;  Name&#36;\sigma,\tau&#36; It's reversible.</p>
<p>For reverse transformations, we can give the following characteristics:</p>
<ul>
<li>The reverse transformation is also a linear transformation.</li>
<li>&#36;\sigma\in L(V) 可逆 \Longleftrightarrow \sigma是双射\Longleftrightarrow\sigma是 V对V的同构映射&#36;</li>
<li>&#36;\sigma&#36;Reversible rules&#36;\sigma&#36;Base map to base, unconnected group to irrelevant group</li>
</ul>
<p>After the definition of linear variations and the definition of operations, it's easy to find out that,<strong>Space&#36;V&#36;In Range&#36;P&#36;The entire linear transformation on the top also forms a dichotomy.&#36;P&#36;The linear space.&#36;L(V)&#36;</strong></p>
<h3>Linear Change Multiples</h3>
<p>We can define linear transformations of the gills.
&#36;&#36;\sigma^{n}=\sigma \sigma {\cdots}\sigma&#36;&#36;
It's a linear shift.&#36;n&#36;I don't know.</p>
<p>Define negative integer as
&#36;&#36;(\sigma^{-1})^n=\sigma^{-n}&#36;&#36;</p>
<p>When we've got the thorium, we can define the linear variations, for the pluriforms.
&#36;&#36;f(x)=a_mx^{m}+a_{m-1}x^{m-1}+\cdots+a_{0}&#36;&#36;
His linear shift.&#36;A&#36;Multiples are
&#36;&#36;f(A)=C_mA^{m}+C_{m-1}A^{m-1}+\cdots+C_0\varepsilon.&#36;&#36;</p>
<p>Linear transformation multiform bond plus and related operating rates</p>
<h3>Linear Change Matrix</h3>
<h4>Definition of linear transformation matrix</h4>
<p>We can easily see if&#36;\varepsilon_i&#36;It's a set of radicals.&#36;\varepsilon&#36;Yes.&#36;\varepsilon=a_1\varepsilon_1+a_2\varepsilon_2+\cdots+a_n\varepsilon_n&#36;  There is.&#36;A\varepsilon=a_1A\varepsilon_1+a_2A\varepsilon_2+\cdots+a_nA\varepsilon_n&#36;</p>
<p>The nature of this means that we just need to know all the underlying images, so we can know all the information about this linear transformation, and we can look at a linear transformation.</p>
<p>Definitions: Establishment&#36;\varepsilon_i&#36;It's linear space.&#36;V&#36;A set of foundations.&#36;A&#36;It's a linear shift above him, so the transformation of the base vector can be derived from the original base vector linear table, which is,
&#36;&#36;\left{\begin{matrix}
 A\varepsilon_{1}=a_{11}\varepsilon_{1}+a_{12}\varepsilon_{2}+\cdots+a_{1n}\varepsilon_{n}\
 A\varepsilon_{2}=a_{21}\varepsilon_{1}+a_{22}\varepsilon_{2}+\cdots+a_{2n}\varepsilon_{n}\
 \cdots\
A\varepsilon_{n}=a_{n1}\varepsilon_{1}+a_{n2}\varepsilon_{2}+\cdots+a_{nn}\varepsilon_{n}
\end{matrix}\right.&#36;&#36;
Let's remember.
&#36; \begin{matrix}
a {11}&amp;a_{12}  &amp; \cdots &amp;a_{1n} \
  a_{21}&amp;a_{22}  &amp; \cdots &amp;a_{2n} \
  \vdots&amp;\vdots  &amp;\ddots  &amp;\vdots \
  a_{n1}&amp;a_{n2}  &amp; \cdots &amp;I don't know.
=A&#36;
There is an original change as &#36;A(\varepsilon_1,\varepsilon_2,\cdots,\varepsilon_n)=(\varepsilon_1,\varepsilon_2,\cdots,\varepsilon_n)A^T&#36;</p>
<p><strong>Let's take the matrix. &#36;A^T&#36; It's called a linear transformation of the matrix under this matrix.</strong> Add this conversion symbol</p>
<h4>Projection and Linear Transformation</h4>
<p>The projection of vectors in space is also an important concept of generational mathematics.&#36;M&#36; Two subspaces. &#36;N_1,N_2&#36; And there is. &#36;N_1+N_2=M,N_{1}\cap N_{2}= \phi&#36;  Consider&#36;M&#36; Medium vector &#36;z&#36; The only breakdown. &#36;z=x+y&#36; And... &#36;x\in N_{1},y\in N_{2}&#36; Linear transformation
&#36;&#36;P_{N_1|N_2}z=x&#36;&#36;
Call it &#36;z&#36; Follow &#36;N_2&#36; Yes. &#36;N_1&#36; It's called the projection. &#36;x&#36; The corresponding matrix is called the projection matrix.</p>
<h4>The operation of linear transformation matrices</h4>
<p>We're giving the following information about the operation of linear transformations and the connection to his matrix.</p>
<ul>
<li>Linear transformations and equals the sum of the corresponding matrix</li>
<li>Linear transformation equals the volume of the corresponding matrix.</li>
<li>Linear transformation multiplication equals the number of arrays</li>
<li>Reversible linear transformations correspond to reversible matrices, and reverse transformations are the same as matrix reversals.</li>
</ul>
<p>Theorem: With a linear transformation matrix, we can easily calculate the coordinates before and after the transformation, which is the reverse definition.
&#36;&#36;\begin{pmatrix}
 y_1\
  y_2\
 \vdots \
 y_n
\end{pmatrix}=A\begin{pmatrix}
 x_1\
  x_2\
 \vdots \
 x_n
\end{pmatrix}&#36;&#36;
of which &#36;y&#36; is the changed coordinates;&#36;x&#36; is the pre-change coordinates;&#36;A&#36;It's a linear transformation matrix.</p>
<h4>Base transformation of linear transformation matrix (similar)</h4>
<p>It is very encouraging to see so many conclusions before us, but we will also face a very real problem: the current linear transformation matrix and the choice of base binding, rather than the transformation itself, and how the linear transformation matrix can be changed in the case of base transformation, which is the subject of this section.</p>
<p>Theorem: Sets a linear change of presence. He has two bases. &#36;\varepsilon_{1},\varepsilon_{2},\cdots\varepsilon_{n},\eta_{1},\eta_{2},\cdots\eta_{n}&#36;  Change the matrix under two bases. &#36;A,B&#36;  Transition matrix between the two bases&#36;\varepsilon_\to\eta&#36; Yes &#36;X&#36; So there is. &#36;B=X^{-1}AX&#36;</p>
<p>We can abstract this matrix into a separate study.</p>
<p>Definitions<strong>Matrix Similar</strong>: Set&#36;A,B&#36; It's digital.&#36;P&#36; Two arrays, if you can find them. &#36;n&#36; A reversible matrix.&#36;X&#36;Make&#36;B=X^{-1}AX&#36;  Call Matrix&#36;A,B&#36; Similar. <strong>It's the third nature that comes after the equivalent.</strong></p>
<p>According to the previous definition, it is easy to give: linear transformation matrices are similar under different bases, and similar matrices can be seen as the same linear transformation matrix under different bases.</p>
<p>We'll give you an important example.<a href="/en/blog/2023/06/17/advanced-algebra-matrices-linear-spaces-notes/">High Algebra 2 Matrix & Linear Space</a> Other Organiser</p>
<p>Calculate
&#36; \begin{matrix}
Two.&amp;1 \
  -1&amp;0
\end{pmatrix}^k&#36;&#36; 并且这个矩阵是某个线性变换在基&#36;\varepsilon_{1},\varepsilon_{2}&#36; 下的矩阵，同时有
&#36;&#36;\begin{pmatrix}
  \eta_1 &amp;  \eta_2
\end{pmatrix}=\begin{pmatrix}
  \varepsilon_{1}&amp;\varepsilon_{2}
\end{pmatrix}\begin{pmatrix}
  1&amp;1 \
  -1&amp;2
\end{pmatrix}&#36;&#36;</p>
<p>So we can calculate the linear transformation at the base.&#36;\eta&#36; Matrix Below&#36;B&#36;Yes.
&#36; \begin{matrix}
One.&amp;1 \
  -1&amp;2
\end{pmatrix}^{-1}\begin{pmatrix}
  2&amp;1 \
  -1&amp;0
\end{pmatrix}\begin{pmatrix}
  1&amp;1 \
  -1&amp;2
\end{pmatrix}=\begin{pmatrix}
  1&amp;1 \
  1&amp;0
\end{pmatrix}&#36;&#36;
那么我们就有
&#36;&#36;X^{-1}AX=B\to A=XBX^{-1}&#36;&#36;
所以
&#36;&#36;It's not a good idea.
And...&#36;B&#36;It's a triangle matrix. It's easy to calculate. We're down a lot.<strong>The approximation achieved by similar matrices</strong></p>
<h3>Feature values and feature vectors</h3>
<h4>Definition of feature values and feature vectors</h4>
<p>Base will influence the linear transformation matrix, and how to select the appropriate matrix so that it can take its simplest form.
That is what we are going to discuss from this section.</p>
<p>Definitions:&#36;\sigma&#36; Yes. &#36;V&#36; Linear change on top if&#36;P&#36; Medium &#36;\lambda&#36;  Existing Vector &#36;\xi&#36; Make &#36;\sigma(\xi)=\lambda\xi&#36;  Name &#36;\lambda&#36; It is a characteristic value of this linear transformation. &#36;\xi&#36;It's the corresponding characteristic vector.</p>
<p>Explanation of feature values and feature values</p>
<ul>
<li>If&#36;\xi&#36;It's a characteristic vector, then.&#36;k\xi&#36; It's also the characteristic vector of this characteristic value.</li>
<li>The same characteristic vector corresponds to only one feature Value</li>
<li>The matrix's characteristic values and characteristic vectors are the linear transformations of the matrix's characteristic values and characteristic vectors.</li>
</ul>
<p>Here's what we're going to do.</p>
<p>Based on the definition of feature value and feature vector, there must be
&#36;&#36;\lambda \begin{pmatrix}
x_1 \
 x_2\
 \vdots\
x_n
\end{pmatrix}=A\begin{pmatrix}
x_1 \
 x_2\
 \vdots\
x_n
\end{pmatrix}&#36;&#36;
Which means...
&#36;&#36;(\lambda E-A )\begin{pmatrix}
x_1 \
 x_2\
 \vdots\
x_n
\end{pmatrix}=0&#36;&#36;
It's essentially a linear equation, so he'll solve it.&#36;|\lambda E-A|=0&#36;</p>
<p>Definitions:&#36;|\lambda E-A|&#36; Post-expanded&#36;\lambda&#36; It's called a feature polygraph. In fact, the root of the feature polygraph is a feature value.</p>
<p>The vector to which the root belt from which the desolation feature is obtained is the vector to which the characteristic value corresponds in the equation group</p>
<h4>Characteristic values and feature vector calculations and weights</h4>
<p>For the quantity matrix &#36;kE&#36;  Characteristic multiples have
&#36;&#36;|\lambda E-kE|=0\Rightarrow(\lambda-k)^{n}=0&#36;&#36;
So there is.&#36;n&#36;Characteristic Values &#36;k&#36;  We call it a feature value.&#36;k&#36; The algebra weight is &#36;n&#36;</p>
<p>For the diagonal matrix, the characteristics are multiple.
&#36;&#36;(\lambda-a_{11})(\lambda-a_{22})\cdots(\lambda-a_{nn})&#36;&#36;
It's a characteristic value. &#36;a_{ii}&#36;  There's no uncertainty about the weight.</p>
<p>For the Triangular Matrix, the features are multiple.
&#36;&#36;(\lambda-a_{11})(\lambda-a_{22})\cdots(\lambda-a_{nn})&#36;&#36;
It's a characteristic value. &#36;a_{ii}&#36;  There's no uncertainty about the weight.</p>
<p>Definition: For a feature value&#36;\lambda&#36; All characterization vectors plus zero vectors constitute a subspace. The dimension is that the characteristic vector group is highly unrelated. This dimension is also called geometry.</p>
<p>Admittedly, algebraic weight greater than or equal to geometric weight for any feature value</p>
<h4>Important theorems associated with the characteristic vector</h4>
<p><strong>Two important characteristics</strong></p>
<ul>
<li>&#36;A&#36;to the sum of the total feature values &#36;a_{11}+a_{22}+\cdots+a_{nn}&#36;</li>
<li>&#36;A&#36;the value of the whole feature is &#36;|A|&#36;</li>
</ul>
<p>Theoretically:<strong>Similar matrices have the same characteristics in multiple forms.</strong>  Theorem finally allows us to study linear transformations and to move away from their foundation, which is the simplest. Hold hands.</p>
<p><strong>Spectrum Map</strong>  Here's the correct conclusion.</p>
<ul>
<li>&#36;\lambda&#36; Yes. &#36;A&#36; Characteristic value</li>
<li>&#36;\lambda^{-1}&#36; Yes. &#36;A^{-1}&#36; Characteristic value</li>
<li>&#36;\frac{|A|}{\lambda}&#36; Yes. &#36;A^{\star}&#36; Characteristic value</li>
</ul>
<p>Theorem(s)<strong>Hamidon Claire theorem.</strong>) : Note &#36;A&#36; It's a matrix. &#36;f(\lambda)&#36; Yes.&#36;A&#36;There are multiple features
&#36;&#36;f(A)=A^n+a_1A^{n-1}+\cdots+a_{n-1}A+a_nE=0&#36;&#36;</p>
<p>Theorem(s)<strong>Spectrum Theorem</strong>: Matrix&#36;A&#36; The characteristic value is &#36;\lambda_i&#36;  &#36;f(x)\in P[x]&#36; Well... &#36;f(A)&#36; The characteristic value is &#36;f(\lambda_i)&#36;</p>
<h4>Problems with known feature values and feature vectors in retrospect</h4>
<p>For questions about known feature values and characteristic vectors, we divide them into categories.</p>
<p><strong>All known feature values and feature vectors</strong>
We know.
&#36;&#36;X^{-1}AX=B&#36;&#36;
of which&#36;X&#36;It's a feature vector matrix. &#36;A&#36; It's the original matrix. &#36;B&#36; It's a signature diagonal matrix.</p>
<p><strong>Only part of the characterization vector, but the original matrix.&#36;A&#36;Symmetric Matrix</strong>
For symmetrical matrices, we know that his characteristic vectors are positive, so that equations or missing characteristic vectors can be constructed.</p>
<p><strong>Only part of the characteristic vector, no other nature.</strong>
The only way to simplify it is by relying on the fact that the characterization vector between the different characteristic values is irrelevant.</p>
<h3>Diagonal Matrix</h3>
<p>This section is intended to simplify the linear transformation matrix, which should be one of the most efficient. What kind of matrix is similar to the diagonal matrix, and how to turn it into a diagonal matrix that this section is going to discuss.</p>
<p>Definitions: &#36;\sigma&#36;Yes.&#36;V&#36;A linear shift on it if it exists&#36;V&#36;And one of the bases that makes the linear transformation of the matrix under it is the angular matrix, which says it can be polarized.</p>
<p>Theoretically: For a diagonal matrix, the element on the angular line is the characteristic value of the matrix, which is fully defined except in order.</p>
<p>Theorem: linear transformation&#36;\sigma&#36;The matrix can be polarized. &#36;n&#36; An unrelated character vector</p>
<p>Theorem: linear transformation&#36;\sigma&#36;Characteristic vectors belonging to different feature values are irrelevant</p>
<p>Inference: If&#36;n&#36;Linear changes in the dimension space&#36;A&#36;There are multiple features.&#36;n&#36;Different root.&#36;n&#36;A different feature value)&#36;A&#36;It can be diagonized.</p>
<p>Inference: Due to the plural field&#36;n&#36;There's got to be a polygon.&#36;n&#36;Roots, so without root, they mean they can be polarized.</p>
<p>Inference: None&#36;n&#36;It's not necessarily an angle. It's just...&#36;n&#36;It's good that a characteristic vector is irrelevant.</p>
<p>Hypothesis: Angularization means that the characteristic subspace dimension is&#36;n&#36;</p>
<p>Inferences: the sum of geometric weights between different characteristic values and the sum of &#36;n&#36;</p>
<p>Inference: the directness of each characteristic subspace is &#36;V&#36;  If there's only one feature space, it is. &#36;V&#36;</p>
<h3>Linear transformation ranges and cores</h3>
<p>Definitions: Establishment&#36;A&#36;It's a linear transformation.&#36;A&#36;It's called a collection of people.&#36;A&#36;♪ And the world ♪&#36;AV&#36;
All by&#36;A&#36;Transformation&#36;0&#36;Vector Composition&#36;A&#36;of the United Nations&#36;A^{-1}(0)&#36;   Or we could use symbols.&#36;ker(\sigma) ~~N(\sigma)&#36;</p>
<p>Theorem: linear transformation&#36;A&#36;Yeah.&#36;V&#36;The values and cores.&#36;V&#36;Subspace</p>
<p>Definition: We will&#36;AV&#36; The dimensions are called linear mutations.&#36;A^{-1}(0)&#36;The dimensions are called zero degrees.</p>
<p>Theoretically: Set linear variations&#36;A&#36;The corresponding matrix is:&#36;A&#36; There is.</p>
<ul>
<li>&#36;A&#36;The value is the subspace generated by the vector group after the original base vector has been converted</li>
<li>Linear transformation&#36;A&#36;It's a matrix.&#36;A&#36;It's a tummy.</li>
</ul>
<p>All we have to do is calculate the need for nuclear space. &#36;AX=0&#36; The equation is fine.</p>
<p>Theoretically:&#36;A&#36;♪ And the twilight ♪&#36;A&#36;Zero degrees and then&#36;n&#36;</p>
<p>Theorem: If&#36;A^2=A&#36; then&#36;A&#36; Similar to Diagonal Matrix</p>
<h3>No change subspace</h3>
<p>Definitions: Establishment&#36;W&#36;Yes.&#36;V&#36;Subspace, if any&#36;\xi\in W&#36; Both. &#36;A\xi\in W&#36; Name &#36;W&#36; Yes.&#36;A-&#36;Subspace, or...&#36;A&#36;No change subspace</p>
<p>Theorem: Visible</p>
<ul>
<li>&#36;V&#36;and&#36;0&#36; These two ordinary spaces are static.</li>
<li>&#36;A&#36;Other Organiser &#36;AV&#36; and &#36;A^{-1}(0)&#36;   It's the constant subspace.</li>
</ul>
<p>Theorem: We can prove by definition.</p>
<ul>
<li>If&#36;A,B&#36;It's a linear shift that can be exchanged.&#36;B&#36;The range and core are&#36;A-&#36;No change subspace</li>
<li>&#36;f(A)&#36;and&#36;A&#36;It's fungible, so they're in constant space.</li>
</ul>
<p>Theorem: By definition, we can understand.</p>
<ul>
<li>Any subspace is a constant subspace of multiplication.</li>
<li>The characterization vector itself forms a one-dimensional constant subspace.</li>
<li>&#36;A-&#36;The exchange of the same subspace and the same subspace</li>
</ul>
<p>Theoretically:&#36;A-&#36;Subspace is subject to base vector Group&#36;A-&#36;No change.</p>
<p>Theoretically: If&#36;V&#36;It can be broken down into several.&#36;A-&#36;Direct sum of subspaces
&#36;&#36;V=W_1\oplus W_2\oplus\cdots\oplus W_s&#36;&#36;
There is.&#36;V&#36;Midlinear Changes&#36;A&#36;The matrix is
&#36; \begin{matrix}
A 1&amp;  0&amp;\cdots   &amp;0 \
  0&amp;  A_2&amp;0  &amp; \vdots\
  \vdots&amp;  0&amp;  \ddots &amp;\vdots \
  0&amp; \cdots  &amp;  0&amp;A_s
\end{pmatrix}&#36;&#36;</p>
<p>Theoretically, a change of character is essentially a change of character.&#36;A&#36;It's decomposed into the straightness of the characteristic subspace of several characteristic vectors, which we call the characteristic subspace of the characteristic vector.<strong>Root Space</strong></p>
<h3>Introduction of Jordan Standard</h3>
<p>It has already been mentioned that the diagonal matrix is the simplest form of simplicity, but meets the need for&#36;n&#36;Only a non-specific vector. What should be the simplest form of other forms? The issue in this section will be discussed in the plural.</p>
<p>Definition: We call the matrix Jordan block as shown in the figure
&#36;J
\lambda 0&amp;  0&amp;  0&amp;0 \
  1&amp;  \lambda_0&amp;  0&amp;0 \
  0&amp;  1&amp; \ddots  &amp;0 \
  0&amp;  0&amp;  1&amp;\lambda_0
\end{pmatrix}_{k\times k}&#36;&#36;</p>
<p>Definitions: The Jordan-type matrix, as shown in the figure below
&#36;A=begin{matrix}
J (\lambda 1, k 1)&amp;  0&amp;0 \
  0&amp;  J(\lambda_i,k_i)&amp;0 \
  0&amp; 0 &amp;(\lambda s, k s)
Can't you see?
of which&#36;k_i&#36;It can be a complex number, or one.</p>
<p>Theoretically: If&#36;A&#36;It's a complex range.&#36;V&#36;Previous linear transformation, then&#36;V&#36;There must be a base in it.&#36;A&#36;The matrix under this group is the Jordan-type matrix, which we call the Jordan standard for linear transformation.</p>
<p>Equivalent description: Any&#36;n&#36;Stage Matrix&#36;A&#36; It's all similar to one Jordan standard, except for the order of the Jordan block, and he's...&#36;A&#36;Absolutely.</p>
<p>As for how to solve the Jordan standard, we will use this text."&#36;lambda&#36;The matrix is part of a chapter. Not here.</p>
<h3>Minimum Multiform</h3>
<p>Definition: According to Hamidon-Claire theorem, any given matrix&#36;A&#36; There's always multiples that can be found.&#36;f(A)=0&#36;I don't know. Obviously.&#36;f(x)&#36; There is no uniqueness (the coefficient can be multiplied, the smallest polygon may be characterized by a multi-factor), the smallest number of which we call the first one is the corresponding matrix.<strong>Minimum Multiform</strong>。</p>
<p>Introduction: The minimum polygon is unique</p>
<p>Introduction:&#36;g(x)&#36;Yes.&#36;A&#36;, and then&#36;f(x)&#36;Satisfied &#36;f(A)=0&#36; Equivalent &#36;g(x)|f(x)&#36;</p>
<p>Theoretically:<strong>The smallest polygon of the matrix must be the factor of the feature polygon</strong></p>
<p>Inference: The matrix has the same minimum pluriformity, and the proposition is not valid.</p>
<p>Introduction: set &#36;A = \\begin{matrix}A 1&amp;0 \0&amp;A_2\end{pmatrix}&#36; 则&#36;A&#36;的最小多项式为&#36;A 1, A 2&#36;, 2 minimum polygons</p>
<p>Introduction:&#36;k&#36; The minimum polygon of the Jordan block is &#36;(x-a)^k&#36;</p>
<p>Theorem: A matrix can be diagonal, equal to the minimum number of formulas.&#36;P&#36;A one-time factor product of the supermix</p>
<p>Inference: A compound matrix can be polarized at the same price as the smallest multi-dimensional weightless Root</p>
<p><strong>Studying the smallest polygons and the simplest of the diagonals, Jordan's simplest is the basis for our next chapter to look at Jordan's solution.</strong></p>
<h2>&#36;\lambda&#36;Matrix</h2>
<p>The whole of this chapter has been prepared for the final section of this paper, "Solver of Jordan's Standard", and the question of how to use Jordan's Standard, which was inherited from the previous chapter, is at the heart of this chapter.</p>
<h3>&#36;\lambda&#36;Definition of Matrix</h3>
<p>Definitions: Establishment&#36;P&#36;It's a digital area.&#36;\lambda&#36;It's a text, a matrix of multiple rings, if his element is&#36;P[\lambda]&#36;, which is called the Matrix &#36;\lambda&#36;Matrix. When?&#36;\lambda&#36; Yes.&#36;P&#36; And when one of them is a number, he's the digital matrix that we studied.</p>
<p>Definitions:&#36;\lambda&#36;All the calculations and nature of the matrix can be derived from<a href="/en/blog/2023/06/17/advanced-algebra-matrices-linear-spaces-notes/">High algebra 2 Matrix and Matrix Operations in Linear Space</a>Inherited, lined from<a href="/en/blog/2023/03/17/advanced-algebra-foundations-notes/">Algebra 1 in a row in the foundation of the algebra</a>Inheritance, he needs a sub-style definition.</p>
<p>Definitions:&#36;\lambda&#36;Matrix Reversible &#36;A(\lambda)B(\lambda)=E=B(\lambda)A(\lambda)&#36;    <a href="/en/blog/2023/06/17/advanced-algebra-matrices-linear-spaces-notes/">High algebra 2 Reverse of matrices and arrays in linear space</a></p>
<p>Theoretically:&#36;\lambda&#36;The reversible condition of the matrix is &#36;|A(\lambda)|&#36;  It's a non-zero number, not a number that contains&#36;\lambda&#36; Multiple</p>
<p>Definitions:&#36;\lambda&#36;Primary transformation of matrix  <a href="/en/blog/2023/06/17/advanced-algebra-matrices-linear-spaces-notes/">High algebra 2 Primary transformation and primary matrix in matrix and linear space</a></p>
<ul>
<li>Line (column) Swap</li>
<li>Line (column) times non-zero constant &#36;c&#36;</li>
<li>Row (column) with another row (column) &#36;\phi(\lambda)&#36; of which&#36;\phi(\lambda)&#36;Minimum Zero Multiples
The primary transformation still meets the corresponding primary matrix with a left (right) multiplier and the primary matrix must be reversible</li>
</ul>
<p>Definition: We call two.&#36;\lambda&#36;The matrix equals if two arrays can be obtained through a series of primary transformations.</p>
<h3>&#36;\lambda&#36;Standard Type of Matrix</h3>
<p>The content of this section is intended to prove that:&#36;\lambda&#36;The equivalence standard type of the matrix is a diagonal matrix and the equivalent is found.</p>
<p><strong>Introduction (lower value)</strong>: Set&#36;\lambda&#36;The upper left corner element of the matrix is &#36;a_{11}(\lambda)\ne0&#36; and &#36;A(\lambda)&#36; If at least one of the elements can't be eliminated, then one can be found.&#36;A(\lambda)&#36; Equivalent. &#36;B(\lambda)&#36; Satisfied &#36;b_{11}(\lambda)\ne0&#36; less than &#36;a_{11}(\lambda)&#36;  Here's the way.</p>
<p><strong>If&#36;A(\lambda)&#36;, the first row (column) has one element that cannot be&#36;a_{11}(\lambda)&#36; Extinct</strong></p>
<p>So there is.</p>
<p>It's a no.&#36;i&#36;Line (column) minus first row (column)&#36;q(\lambda)&#36;Multiply and then will&#36;i&#36;Line (column) and first row (column) in the order in which the remainder will be exchanged&#36;r(\lambda)&#36;Put it on.</p>
<p><strong>If&#36;A(\lambda)&#36;The first row (column) does not contain an element that cannot be&#36;a_{11}(\lambda)&#36; Exploding, but elements.&#36;a_{ij}&#36;  Can't be.&#36;a_{11}(\lambda)&#36; Extinct</strong></p>
<p>So let's make the following changes.</p>
<ul>
<li>Use first row&#36;a_{11}&#36;Position&#36;a_{i1}&#36;Turn to 0  <em>The nature of the multiple use</em></li>
<li>Put the no.&#36;i&#36;Double the line to line 1</li>
<li>This is when we turn the problem into a situation where we can continue looking for the equivalent.</li>
</ul>
<p>Arbitrary non-zero theorem (alienability of the equivalent standard)&#36;\lambda&#36;Matrix Equivalent
&#36; \begin{matrix}
d 1 (\lambda) &amp;0  &amp; \cdots  &amp;0 \
  0&amp;  d_2(\lambda )&amp;0  &amp; \vdots \
  \vdots &amp;  0&amp;  \ddots &amp; 0\
  0&amp; \cdots  &amp;  0&amp;d n(\lambda)
Can't you see?
of which &#36;d_i(\lambda)|d_{i+1}(\lambda)&#36;  That's the higher the down.</p>
<p>This is proof of the fact that the upper left corner has been transformed into one of the elements using the reasoning of lower sub-equivalent prices.&#36;A(\lambda)&#36;All elements of the factor, then all non-nationals in the first row&#36;a_{11}&#36;Place zero, then study it.&#36;n-1&#36;Step line, repeat the steps ahead.</p>
<h3>No change factor</h3>
<p>No change factor study&#36;\lambda&#36;Matrix Equivalent Standard Unique</p>
<p>Definitions: Establishment&#36;\lambda&#36;Matrix&#36;A(\lambda)&#36;- What?&#36;r&#36; Yeah. &#36;1\le k\le r&#36;  &#36;A(\lambda)&#36;There must be something non-zero.&#36;k&#36;Step pattern. All non-zeros.&#36;k&#36;Maximum public factor of 1 for the first coefficient of the step&#36;D_k&#36;Called&#36;A(\lambda)&#36;Yes.&#36;k&#36;.</p>
<p>Theorem: Obvious, for example&#36;r&#36;Yes.&#36;\lambda&#36;Matrix&#36;A(\lambda)&#36;, exists&#36;r&#36;Row factors</p>
<p>Theorem: for equal value&#36;\lambda&#36;The matrix. They have the same type of hormonal factor.</p>
<p>Theoretically:&#36;\lambda&#36;The matrix standard is the only one.</p>
<p>Proof: Easy to know. &#36;D_{k}(\lambda)=d_{1}(\lambda)d_{2}(\lambda)\cdots d_{k}(\lambda)&#36;  It's the sum of all the diagonal elements. It's easy to know.
&#36;&#36;d_{1}(\lambda)=D_1(\lambda)\quad d_{2}(\lambda)=\frac{D_{2}(\lambda)}{D_1(\lambda)}\quad d_{r}(\lambda)=\frac{D_{r}(\lambda)}{D_{r-1}(\lambda)}&#36;&#36;
Definition: We say that the previous standard of proof of equivalence is defined only at the time&#36;d_{i}(\lambda)&#36;Yes.&#36;\lambda&#36;Matrix no change factor.</p>
<p>Theoretically: No change factor, round factor, standard equivalent model mutually determined</p>
<p>Theorem: The smallest multiformation of the matrix is the last of all constants.</p>
<h3>Matrix Similar Conditions</h3>
<p>This section is a transitional chapter, setting up a digital matrix and &#36;\lambda&#36;The link between the matrix</p>
<p>Theorem: Numerical Matrix &#36;A,B&#36; It's similar. &#36;\Leftrightarrow&#36; Its Feature Matrix &#36;\lambda E -A,\lambda E -A&#36;As&#36;\lambda&#36;The matrix is equal.</p>
<p>Definitions: numerical matrix&#36;A&#36;Corresponding characterization matrix&#36;\lambda E-A&#36; The constant factor is the constant factor of the digital matrix.</p>
<p>Inference:&#36;A&#36;and&#36;A^T&#36;It's equal.</p>
<h3>Primary factor</h3>
<p>This is the last step of our mating. This section's research is in the plural.</p>
<p>Definitions: Putting Matrix&#36;A&#36; All constant factors greater than zero are broken down into a product of a single factor of 1 that differs from each other. All the factor squares are called matrices.&#36;A&#36;Primary factors</p>
<p>Let's give you an example.</p>
<p>All constant factors are: &#36;9&#36; individual &#36;1&#36;   &#36;(\lambda-1)^2&#36;   &#36;(\lambda-1)^2(\lambda+1)&#36;  &#36;(\lambda-1)^2(\lambda+1)(\lambda^2+1)^2&#36;</p>
<p>So the primary factor is
&#36; (\lambda-1)^<del>(\lambda-1)^{2}</del>(\lambda-1)^{2}~~~(\lambda+1)<del>(\lambda+1)</del>(\lambda-i)^{2}~~(\lambda+i)^{2}&#36;&#36;</p>
<p>The primary factor can be changed back to the constant.</p>
<p>Obviously, there's an integral relationship between the constant factors, so it's possible to take into account the same number of constant factors (not enough to add 1) for each of the same single factors.</p>
<p>It's the same as before.
&#36; \begin{matrix}
(llambda-1)&amp; (\lambda+1) &amp;(\lambda-i)^{2}  &amp; (\lambda+i)^{2}\
  (\lambda-1)^{2}&amp; (\lambda+1) &amp; \vdots  &amp; \vdots \
  (\lambda-1)^{2}&amp;  \vdots  &amp; \vdots  &amp;\vdots  \
  \vdots &amp;   \vdots &amp;\vdots   &amp;\vdots
Can't you see?
The primary factor is the sum of every line.</p>
<p>Theoretically: The two matrices are similar, equal to the same primary factors.</p>
<p>Theorem (a primary factor approach): Trying to require a primary factor&#36;A(\lambda)&#36;  It's a direct one.&#36;A(\lambda)&#36; Diagonalize it and decompose it into a factor product.&#36;A&#36;All primary factors.</p>
<h3>Jordan standard solver</h3>
<p>This is the heart of this chapter.</p>
<p>Think about a Jordan block.
&#36;J
\lambda 0&amp;  0&amp;  0&amp;0 \
  1&amp;  \lambda_0&amp;  0&amp;0 \
  0&amp;  1&amp; \ddots  &amp;0 \
  0&amp;  0&amp;  1&amp;\lambda 0
\end{matrix}
Primary factors &#36;(\lambda-\lambda_0)^k&#36;</p>
<p>It's not hard to get a primary factor for a Jordan-type matrix.
&#36;&#36;(\lambda-\lambda_{1})^{k_{1&#125;&#125;~~(\lambda-\lambda_{2})^{k_{2&#125;&#125;\ldots&#36;&#36;</p>
<p>Theoretically: Jordan's matrix is fully determined by his primary factor, except the order of the Jordan block</p>
<p>Theoretically:&#36;A&#36;It's a linear shift in the complex range.&#36;V&#36;There must be a base in it.&#36;A&#36;The corresponding matrix is a Jordan-type matrix, with the exception of the order of the Jordan block, which is fully determined</p>
<p>Theoretically: The matrix can be parded to equalize all primary factors once.</p>
<p>Theoretically: the matrix can be equated to all constant factors without rooting</p>
<p>As to what kind of similarity can get this Jordan-type matrix, we will not study it.</p>
<h3>Rational Standard</h3>
<p>This section looks at a definition of a standard type similar to the Jordan standard type, which exists in terms of uniqueness, opinio juris, i.e. reasonable standard type.</p>
<p>Definitions:&#36;d(\lambda)=\lambda^{n}+a_{1}\lambda^{n-1}+\cdots+a_{n}&#36; Call Matrix&#36;A&#36;It's a friend matrix if it's satisfied.
&#36;A=begin{matrix}
CC BY-NC-ND 2.0&amp; 0 &amp; 0 &amp;-a_n \
  1&amp;  0&amp; 0 &amp;\vdots  \
  0&amp;  1&amp;  \ddots &amp;-a_2 \
  0&amp;  0&amp;  1&amp;-a_1
\end{pmatrix}&#36;&#36;</p>
<p>Definitions: Block-to-angle matrix consisting of friends is called Rational Standard</p>
<p>Theoretically: the Tomomo Matrix&#36;A&#36;No change factor is a lot of 1 and &#36;d(\lambda)&#36;</p>
<p>Theoretically: the constant factor for a reasonable standard matrix is the constant factor for each friend matrix and 1</p>
<p>Theoretically:&#36;P&#36;Top&#36;n&#36;Array&#36;A&#36;It's similar to the only logical standard, which is being&#36;A&#36;No change factor is fully established, including order (based on the number of no change factor, lower above)</p>
<h2>European Space</h2>
<p>In the online space, the distance of the vector is only multiplied and multiplied by the pressure that leads to angles, which are not available at any distance, so this chapter will add in-house calculations to the linear space and get European space.</p>
<h3>Definition and basic nature of European space</h3>
<p>Definitions: online space&#36;V&#36;in which the binary is defined as internality if the conditions below it meet</p>
<ul>
<li>Symmetry&#36;(\alpha,\beta)=(\beta,\alpha)&#36;</li>
<li>Linear&#36;(k\alpha,\beta)=k(\alpha,\beta)&#36;  &#36;(\alpha+\beta,v)=(\alpha+v)+(\beta,v)&#36;</li>
<li>Positive &#36;(\alpha,\alpha)\geq0\quad\text{使当}\alpha=0\text{ 时 等号成立}&#36;</li>
</ul>
<p><strong>The European space is essentially online with a new operation that does not change the original linear space itself.</strong></p>
<p>We can give you two common European spaces.</p>
<ul>
<li>&#36;R^n&#36;Space for&#36;\alpha=(a_1,a_2,...,a_n),\beta=(b_1,b_2,...,b_n)&#36; Definitions &#36;(\alpha,\beta)=a_1b_1+a_2b_2+\cdots+a_nb_n&#36;</li>
<li>&#36;[a,b]&#36;Upline Function Space&#36;c&#36;  Yeah. &#36;f(x),g(x)&#36; Definitions &#36;(f(x),g(x))=\int_{a}^{b} f(x)g(x) dx&#36;</li>
</ul>
<p>Definitions: non-negative actuals &#36;\sqrt{(\alpha,\alpha)}&#36;  Called &#36;\alpha&#36; The length of writing &#36;|\alpha|&#36;   Obviously.  &#36;|k\alpha|=k|\alpha|&#36;</p>
<p>Definitions:&#36;|\alpha-\beta|&#36; The distance between two vectors.</p>
<p>Definitions:&#36;\frac{\alpha}{|\alpha|}&#36; We call this vector a unit vector.</p>
<p>Definitions: &#36;\cos&lt;\alpha,\beta&gt; == sync, corrected by elderman == @elder man</p>
<p>Theoretically:&#36;|(\alpha,\beta)|\le|\alpha||\beta|&#36; It's called Cauchy.</p>
<ul>
<li>&#36;|a_{1}b_{1}+\cdots+a_{n}b_{n}|\leq\sqrt{a_{1}^{2}+\cdots+a_{n}^{2&#125;&#125;\sqrt{b_{1}^{2}+\cdots+b_{n}^{2&#125;&#125;&#36;</li>
<li>&#36;|\int_{a}^{b}f(x)g(x)dx|\leq\sqrt{\int_{a}^{b}f(x)^2dx}\sqrt{\int_{a}^{b}g(x)^2dx}&#36;</li>
</ul>
<p>We can also give a triangulation. &#36;|\alpha+\beta|=|\alpha|+|\beta|&#36;</p>
<p>Definitions: If &#36;(\alpha,\beta)=0&#36;  We call these two vectors positive/vertical.</p>
<p>Theorem (extension of the numeric theorem):&#36;|\alpha_{1}+\ldots+\alpha_{m}|^{2}\le |\alpha_{1}|^{2}+|\alpha_{2}|^{2}+\ldots+|\alpha_{m}|^{2}&#36;  When and only when all the vectors are right, the equals are set.</p>
<p>A matrix of these linear transformations, we can matrix the internalization operations.</p>
<p>Definitions: Taking a base set&#36;\varepsilon_i&#36;   Two vectors&#36;X,Y&#36; Yes. &#36;X=x_1\varepsilon_1+x_2\varepsilon_2+\cdots+x_n\varepsilon_n&#36;  &#36;Y=y_1\varepsilon_1+y_2\varepsilon_2+\cdots+y_n\varepsilon_n&#36;  So there is. &#36;(\alpha,\beta)=X^TAY&#36;  of which &#36;X,Y&#36;It's a coordinate column vector.&#36;A&#36;It's a matrix he meets.
&#36;A=begin{matrix}
a {11}&amp;a_{12}  &amp; \cdots &amp;a_{1n} \
  a_{21}&amp;a_{22}  &amp; \cdots &amp;a_{2n} \
  \vdots&amp;\vdots  &amp;\ddots  &amp;\vdots \
  a_{n1}&amp;a_{n2}  &amp; \cdots &amp;I don't know.
Can't you see?
And there is. &#36;a_{ij}=(\varepsilon_i,\varepsilon_j)&#36;  So easy to know, he's a symmetric matrix.<strong>An internal matrix can be calculated from the base's internal volume.</strong></p>
<p> It's easy to calculate when the base changes occur.&#36;(\eta_1,\eta_2\ldots,\eta_n)=(\varepsilon_1\ldots\varepsilon_n)C&#36;And then, the matrix of the interior became... &#36;C^TAC&#36;  Which means we're here.<a href="/en/blog/2023/06/17/advanced-algebra-matrices-linear-spaces-notes/">Upper Algebra 2 Seconds in Matrix and Linear Space</a>Change of contract described therein.</p>
<p>It's easy to give theorems in combination with these two chapters: the measurement matrix is positive, and the positive matrix can be the measurement matrix.</p>
<p>Theoretically: the subspace of the European space still constitutes the European space for the original internal operation.</p>
<h3>Standard Logic</h3>
<h4>Definition of standard positive basis</h4>
<p>We're here.<a href="/en/blog/2023/06/17/advanced-algebra-matrices-linear-spaces-notes/">High algebra 2 Linear space in matrix and linear space</a>It presents the standard base, and after the definition of the built-in operation, we decided to study how to turn a common base into a standard.</p>
<p>Definitions: A set of non-zero, two-way vectors called positive vectors Group</p>
<p>Theoretically: The positive vector is a linearly unrelated vector, and he has the potential to become a matrix.</p>
<p>Definitions:&#36;n&#36;In the U-European space,&#36;n&#36;Two or two positive vectors are referred to as positive price bases, square bases of unit vector composition are referred to as standard logarithmics and standard logarithmics are satisfied
&#36; (\varepsilon i,\varepsilon i) = left{begin{matrix}
CC BY-NC-ND 2.0 &amp;i\ne j \
  1&amp;i=j
\right.
That is to say, the internal measurement matrix using the standard positive price base is the unit matrix, and the standard positive price base must exist because the arbitrary symmetric matrix has a certain contract in the unit matrix.</p>
<p>When we used a standard log, we did.</p>
<ul>
<li>Vector Coordinate Satisfactory &#36;x_i=(x,\varepsilon_i)&#36;</li>
<li>Internal &#36;(\alpha,\beta)=x_1y_1,x_2y_2,\cdots,x_ny_n&#36;</li>
</ul>
<h4>Schmidt's in contact.</h4>
<p>Theoretically: any positive vector group can be expanded to a set of standard squares</p>
<p>Theorem (Schmidt circulator): Any group of foundations&#36;\varepsilon_i&#36; We can all find a standard log. &#36;\eta_i&#36; Just have to follow the way down to a positive exchange.</p>
<p>First we get the real deal.&#36;\xi_i&#36;</p>
<ul>
<li>&#36;\xi_1=\varepsilon_i&#36;</li>
<li>&#36;\xi_{2}=\varepsilon_{2}-\frac{(\varepsilon_{2},\xi_{1})}{(\xi_{1},\xi_{1})}\xi_1&#36;</li>
<li>&#36;\xi_{m+1}=\varepsilon_{m+1}-\frac{(\varepsilon_{m+1},\xi_{1})}{(\xi_{1},\xi_{1})}\xi_{1}-\frac{(\varepsilon_{m+1}\xi_{2})}{(\xi_{2},\xi_{2})}\xi_{2}-\cdots-\frac{(\varepsilon_{m+1}\xi_{n})}{(\xi_{n},\xi_{n})}\xi_{n}&#36;</li>
</ul>
<p>And then we'll turn it back on.&#36;\xi_i&#36;  Standardization allows for a standard logarithmic. &#36;\eta_i&#36;</p>
<h4>An active matrix</h4>
<p>Transition between the two groups is satisfactory&#36;&#36;(\eta_{1} \cdots \eta_n)=(\xi_1\cdots \xi_n)A&#36;&#36;
Which means... &#36;A&#36; Satisfied
&#36;a {1i}a}+\cdots+a a nj}=begin{cases}1&amp;i=j\0&amp;i\neq j&amp;I'm sorry.
That means... &#36;AA^T=E&#36;  Or... &#36;A^{-1}=A^T&#36;   We call it a positive matrix.</p>
<h3>The same building of European space</h3>
<p>This section is about us.<a href="/en/blog/2023/06/17/advanced-algebra-matrices-linear-spaces-notes/">High algebra 2 Linear space in matrix and linear space</a>It's going to the end of the inner space.</p>
<p>In European space, the contours need to be satisfied.</p>
<ul>
<li>&#36;\sigma(\alpha+\beta)=\sigma(\alpha)+\sigma(\beta)&#36;</li>
<li>&#36;\sigma(kq)=k\sigma(q)&#36;</li>
<li>&#36;(\sigma(\alpha),\sigma(\beta))=(\alpha,\beta)&#36;</li>
</ul>
<p><strong>The contours of European space must be contours of linear space.</strong></p>
<p>Theoretically:&#36;n&#36;Vioux Space and&#36;R^n&#36;Compositing</p>
<p>Theoretically: the essence of the European spatial equation is its dimensions, and the same dimensions of the European spatial equation must be identical.</p>
<h3>Switching</h3>
<p>From this section, we promote linear transformation to European space. Medium</p>
<p>Definition: If in a European space &#36;V&#36; Medium, he's responding to linear changes in linear space. &#36;A&#36; Satisfied with the constantity of internality, it's called a positive transformation in European space, that is,&#36;(A\alpha,A\beta)=(\alpha,\beta)&#36;</p>
<p>We can also paint European space in other spaces.</p>
<ul>
<li>Keep the length of the vector unchanged:&#36;|A\alpha|=|\alpha|&#36;</li>
<li>Keep vector distance unchanged:&#36;d(A\alpha,A\beta)=d(\alpha,\beta)&#36;</li>
<li>Maintain standard logarithmic:&#36;\xi&#36; It's standard, then.&#36;A\xi&#36; It's standard.</li>
<li>&#36;A&#36;The matrix that has been converted to the standard logarithmic is the positive matrix.</li>
</ul>
<p>Because the positive matrix is reversible, there is.</p>
<ul>
<li>It's reversible.</li>
<li>Is there a trade-off between reverse and accumulation?</li>
<li>Turning into a European-style co-constructor.</li>
</ul>
<h3>Subspace in European space</h3>
<p>This is the extension of the subspace section to European space.</p>
<p>Definition: Vector&#36;\alpha&#36; Handover to Space &#36;V_1&#36; ♪ When and only when he's in space ♪ &#36;V_1&#36;All vectors recorded&#36;\alpha\bot V_1&#36;</p>
<p>Definition: Space&#36;V_1&#36;It's in space.&#36;V_2&#36;  When and only when &#36;V_1&#36;All vectors in the middle are in space.&#36;V_2&#36;  Recorded&#36;V_{1}\bot V_2&#36;</p>
<p>Theoretically: If &#36;V_1,V_2,\cdots,V_n&#36; Two and two. &#36;V_1+V_2+\cdots+V_n&#36; It's straight.</p>
<p>Definitions: If&#36;V_{1}\bot V_2&#36;  &#36;V_1+V_2=V&#36;  Name&#36;V_1,V_2&#36; We'll remember.&#36;V_1&#36;Other Organiser &#36;V_1^{\bot}&#36;</p>
<p>Theorem: the following conclusions are valid:</p>
<ul>
<li>Rectification is unique</li>
<li>&#36;rank(V)+rank(V^{\perp})=n&#36;</li>
<li>&#36;(W^{\bot})^{\bot}=W&#36;</li>
<li>&#36;V_1^{\bot}&#36;  ♪ Just by all ♪ &#36;V_1&#36; Composition of the positive vector</li>
</ul>
<h3>Standard type of actual symmetric matrix</h3>
<p>This section is an extension of our similar standard, that is, thinking about the shape of the inner space of the corner matrix and the Jordan matrix. The so-called symmetric matrix is the result of the fact that all the internal matrices are a symmetric matrix and that there is no contradiction in the study of the internal matrix.</p>
<p>Yes. <a href="/en/blog/2023/06/17/advanced-algebra-matrices-linear-spaces-notes/">Upper Algebra 2 Seconds in Matrix and Linear Space</a> We already know in one section:<strong>All symmetric matrix contracts in a diagonal matrix</strong>  This is the standard contract for the matrix. &#36;C^TAC&#36;</p>
<p>Theorem: An arbitrary symmetry matrix can be obtained by a positive symmetrical array, i.e. a symmetric matrix&#36;B&#36; Angular form &#36;A&#36;  Yes.    &#36;B=C^TAC&#36;  Or...  &#36;B=C^{-1}AC&#36;</p>
<p>Introduction: all real symmetric matrices have real values</p>
<p>Introduction: satisfaction &#36;( A\alpha , \beta ) = (\alpha, A\beta )&#36; Or... &#36;P^TA\alpha = \alpha\beta&#36;The transformation is called symmetrical transformation, and on the actual symmetric matrix the linear transformation is symmetrical transformation.</p>
<p>Theorem (converting standard type):&#36;A&#36;It's a symmetric matrix, then&#36;A&#36;Characteristic vectors of different characteristic values are converse and we can find the diagonal matrix from which they are transversed as follows:&#36;B&#36;and transform matrix&#36;T&#36;</p>
<ol>
<li>Found Real Symmetry Matrix&#36;A&#36;Characteristic value</li>
<li>Solve the corresponding characterization vector, standardize each characterization vector</li>
<li>Putting all the characterization vectors in a matrix is a matrix that's being converted.&#36;T&#36; The diagonal matrix of characteristic values is&#36;B&#36;  I'm writing.&#36;T&#36; and &#36;B&#36; Make sure you follow the same order.</li>
</ol>
<p>This theorem is expressed as a secondary: the actual substrate can be replaced by a positive liner &#36;\lambda_1y_1^2+\cdots+\lambda_ny_n^2&#36;  of which &#36;\lambda_i&#36; It's the root of multiple patterns.</p>
<p><strong>The standard model that is being converted is the only one, which is the most important change he has made to the contract, which can be multiple, but the standard model that is being converted is the only other than sequence.</strong></p>
<p><strong>Numbers of non-zero feature values corresponding to a symmetric matrix. Positive or negative inertial index is the number of positive or negative characteristics</strong></p>
<h3>Space.</h3>
<p>We're here to study the European space in the plural.</p>
<p>Definition: If the internal operation of an built-in space is in a complex range and meets the following characteristics, it is referred to as quail space</p>
<ul>
<li>Cosymmetry &#36;(\alpha,\beta)=\overline{(\beta,\alpha)}&#36;</li>
<li>Linear &#36;(k\alpha,\beta)=k(\alpha,\beta)&#36; &#36;(\alpha+\beta,v)=(\alpha,\gamma)+(\beta,v)&#36;  of which &#36;k&#36; is an arbitrary plural</li>
<li>Positive characterization &#36;(\alpha,\alpha)=0\quad\text{当且仅当 }\alpha=0&#36;</li>
</ul>
<p>There's a computational character down there.</p>
<ul>
<li>&#36;(\alpha,k\beta)=-\bar{k}(\alpha,\beta)&#36;</li>
<li>&#36;(\alpha+\beta,v)=(\alpha,\gamma)+(\beta,v)&#36;</li>
<li>Vector&#36;\alpha&#36; Length is &#36;\sqrt{(\alpha,\alpha)}&#36;</li>
<li>Do not define an angle</li>
<li>Cosy's variant:&#36;|(\alpha,\beta)|\le|\alpha||\beta|&#36;</li>
<li>Triangular heterogeneity  &#36;|\alpha+\beta|=|\alpha|+|\beta|&#36;</li>
</ul>
<p>Definition: Satisfaction &#36;(A\alpha,A\beta)=(\alpha,\beta)&#36; It's called the twilight shift.</p>
<p>Definition: The matrix under the standard logarithmic is the matrix, set&#36;A&#36;It's the matrix, and it's satisfied.&#36;A\bar{A^{T&#125;&#125;=\bar{A^{T&#125;&#125;A=E&#36;</p>
