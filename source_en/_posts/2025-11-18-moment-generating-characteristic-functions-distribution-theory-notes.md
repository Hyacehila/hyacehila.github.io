---
title: Moment Generating Functions, Characteristic Functions, and Distribution Theory
title_zh: 矩母函数、特征函数与分布理论：生成函数视角下的概率分布
date: 2025-11-18 21:52:54 +0800
categories:
- Data Science
- Probability & Statistical Foundations
tags:
- Statistics
- Probability
- Distribution Theory
author: Hyacehila
mathjax: true
hidden: true
excerpt: A overview on generating functions, moment generating functions, characteristic functions, and distribution theory
  from a probabilistic distribution perspective.
description: A overview on generating functions, moment generating functions, characteristic functions, and distribution theory
  from a probabilistic distribution perspective.
excerpt_zh: 整理生成函数视角下的概率分布研究方法。
permalink: /blog/2025/11/18/moment-generating-characteristic-functions-distribution-theory-notes/
lang: en
translation_key: 2025-11-18-moment-generating-characteristic-functions-distribution-theory-notes
translation_status: machine
translation_source_hash: 7787e6efbf7cc3c2cc277abcce41bbdffac8f19f4dcc98b54945e6edad95c0e1
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<h2>Mother and Rectangular Functions</h2>
<p>Before the introduction of the high probability theory, we looked back at some of the more complex elements of the primary probability theory.<strong>The theory of probability-generated functions allows us to turn the probability problem into an analytical one.</strong> Proof of certain issues would be a good simplification.</p>
<p>The questions in this article can also be addressed<a href="/en/blog/2023/03/18/elementary-probability-notes/">Primary probability theory: random events, probability models and random variables</a>、<a href="/en/blog/2024/09/24/probability-and-statistics-exercises-notes/">Probability and statistical caselines: sample distribution, parameter estimates and statistical nature</a>How the concept of a relatively close read together is developed in different contexts.</p>
<h3>Definition and nature of parent function</h3>
<p>The parent function is designed to simplify some studies by introducing another expression of probability distribution.</p>
<p>For those random variables that take only non-negative values (e.g., the two distributions, the Benuli distribution, the bellow distribution, etc.), we call whole-value random variables, the parent function can be used as a support for this type of random variable.</p>
<p>The parent function is defined as
&#36;&#36;G(t)=\sum_{k=0}^\infty p_kt^k=p_0+p_1t+\ldots+p_nt^n+\ldots &#36;&#36;
It's easy to see. &#36;G(t)=E(t^{p_k})&#36;  </p>
<p>And by the nature of the grade, we know that the parent function has the following characteristics.</p>
<ul>
<li>The parent function and the distribution column are only certain to each other</li>
<li>&#36;G(1)=1&#36;</li>
<li>&#36;\mathrm{E}(X)=G^{\prime}(1)&#36;</li>
<li>&#36;\mathrm{Var}(X)=G^{\prime\prime}(1)+G^{\prime}(1)\left(1-G^{\prime}(1)\right)&#36;</li>
<li>Set their parent function as the&#36;G_{X_1}(t),G_{X_2}(t)&#36; According to the discrete volume formula, they meet the parent function.&#36;G_X(t)=G_{X_1+X_2}(t)=G_{X_1}(t)\times G_{X_2}(t)&#36;</li>
</ul>
<h3>Definition and Nature of Rectangular Mother Functions</h3>
<p>The parent function can only be used for discrete random variables, and the feature function involves the inconvenient operation of a compound fraction (his advantage is to be present forever).</p>
<p>Here we introduce rectangular functions.<strong>Moment Generating Function</strong>He has a lot of merit in dealing with a lot of random variables.</p>
<p>Random Variables&#36;X&#36;. The rectangular function is defined as follows:
&#36;&#36;M_X(t)=E[\mathrm{e}^{tX}]=\begin{cases}\int_{-\infty}^{+\infty}\mathrm{e}^{tx}f(x)\mathrm{d}x,X\text{ 具有密度函数 }f(x);\\\sum_{i=0}^{\infty}\mathrm{e}^{tx_i}\rho\left(x_i\right),X\text{ 具有分布律 }p(x).\end{cases}&#36;&#36;
We give you some common nature.</p>
<ul>
<li>Rectangular Mother Functions&#36;M_X(t)&#36;and random variables&#36;X&#36; Only sure</li>
<li>If Random Variables&#36;X,Y&#36;- I'm independent. - So...&#36;M_{X+Y}(t)=M_{X}(t)M_{Y}(t)&#36;</li>
<li>If&#36;Y=aX+b&#36; Well...&#36;M_Y(t)=e^{bt}M_{X}(at)&#36;</li>
<li>Rectangular mother functions can be used to get the originals.  &#36;E(X^{n})=M_{X}^{(n)}(0)&#36; </li>
<li>&#36;M(1)=1&#36; It's the same as the parent function.</li>
</ul>
<h3>Simple application</h3>
<p>Here we will present some of their applications and their probative skills.</p>
<h4>Rectangular function of normal distribution</h4>
<p>&#36;&#36;\begin{aligned}
M_{X}\left(t\right) =E\left[e^{tX}\right]  \
&amp;=\int_{-\infty}^{\infty}e^{tx}\frac{1}{\sqrt{2\pi\sigma&#125;&#125;e^{-\frac{\left(x-\mu\right)^{2&#125;&#125;{2\sigma^{2&#125;&#125;}dx \
&amp;=e^{\left(\mu t+\frac{1}{2}\sigma^{2}t^{2}\right)}\int_{-\infty}^{\infty}\frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{\left[x-\left(\mu+\sigma^{2}t\right)\right]^{2&#125;&#125;{2\sigma^{2&#125;&#125;}dx \
&amp;The #Sighma is a very important tool for the development of the country.
I'm sorry, I'm sorry.
The crediting technique involved in this is to construct a probability density function, which is one, which simplifys the problem of the fraction, which is very common and will be used again.
According to the four characteristics that we've given you, we know that.&#36;E(X)=\mu ~~~~E(X^{2})=\mu^{2}+\sigma^{2},Var(X)=\sigma^{2}&#36;</p>
<h4>Rectangular function of Gamma distribution</h4>
<p>&#36;&#36;\begin{gathered}f(x)=\frac{\beta^\alpha}{\Gamma(\alpha)}x^{\alpha-1}e^{-\beta x},x&gt;~, ~, =, =, =, =, =, =, =, =, =, =, =, =, =, =, =, =, =, =, =, =, =, =, =, =, =, =, =, =, =, =, =, =, =, =, =, =, =, =, =, =, =, =, =, =, =, =, =, =, =, =, =, =, =, =, =, =, =, =, =, =, =, =, =, =, =, =, =, =, =, =, =, =, =, =, !, }, !, !, !, !, !, !, !, &#36;&#36;
这是Gamma分布的形式表示 还有一些比较重要的性质叙述
&#36;&#36;♪ I'm not gonna let you go ♪
Or the idea of a probabilistic density function in the front?
With him, we're pretty sure.&#36;E[X]=M^{^{\prime&#125;&#125;(0)=\frac\alpha\beta,\quad\text{进而}Var(X)=\frac\alpha{\beta^2}&#36;
<em>Attention.&#36;\alpha=1&#36;And he was the index distribution.&#36;f(x)=ke^{-kx}&#36;</em> </p>
<h4>Rectangular and parent functions for distribution of two items</h4>
<p>&#36;&#36;\begin{aligned}
M(t)&amp; =E[e^{tX}]=\sum_{k=0}^ne^{tk}C_n^kp^k(1-p)^{n-k}  \
&amp;=\sum_{k=0}^nC_n^k(pe^t)^k(1-p)^{n-k} \
&amp;=(pe^t+1-p)^
I'm sorry, I'm sorry.
The idea is the same, or the classic two-part extension used here is easy to calculate.
&#36;E[X]=np,\quad E[X^{2}]=M^{^{\prime\prime&#125;&#125;(0)=n(n-1)p^{2}+np,\quad Var(X)=np(1-p)&#36;</p>
<p>The parent function should be
&#36;&#36;M_{X}(t)=E\left(e^{t X}\right)=\left(1-p+p e^{t}\right)^{n}&#36;&#36;</p>
<h4>Rectangular and parent functions of the Porpine distribution</h4>
<p>&#36;&#36;\begin{aligned}
M(t)&amp; =E[e^{tX}]=\sum_{k=0}^\infty\frac{e^{tk}e^{-\lambda}\lambda^k}{k!}  \
&amp;=e^{-\lambda}\sum_{k=0}^\infty\frac{(\lambda e^t)^k}{k!}=e^{-\lambda}e^{\lambda e^t} \
&amp;== sync, corrected by elderman ==
I'm sorry, I'm sorry.
Or the classic numbers we use are extended.
It's easy to calculate.
&#36;E[X]=\lambda,\quad E[X^2]=M^{^{\prime\prime&#125;&#125;(0)=\lambda+\lambda^2,\quad Var(X)=\lambda&#36;</p>
<p>The parent function should be
&#36;&#36;M_{X}(t)=E\left(e^{t X}\right)=e^{\lambda\left(e^{t}-1\right)}&#36;&#36;</p>
<h4>Carcone Distribution Rectangles</h4>
<p>&#36;\begin{aligned}&amp;\\text{Order}Z 1, Z 2, \\cdots, Zn\text{and}Z\text{is the standard normal distribution random variable of independent and distributed,}&amp;\text and =1^2Z2^cdots+\n^text{, we call X subject to the C.A. distribution}end{aligned}
It's easy to know.
&#36;&#36;M_X(t)=[M_{Z^2}(t)]^n=(E[e^{tZ^2}])^n&#36;&#36;
Make the following calculations
&#36;&#36;00\
E[e]&amp; =\frac1{\sqrt{2\pi&#125;&#125;\int_{-\infty}^\infty e^{tz^2}e^{-\frac{z^2}2}dz  \
&amp;=\frac1{\sqrt{2\pi&#125;&#125;\int_{-\infty}^\infty e^{-\frac{z^2}{2\sigma^2&#125;&#125;dz\quad(\sigma^2=(1-2t)^{-1}) \
&amp;== sync, corrected by elderman == @elder man
I'm sorry, I'm sorry.
Or the classic way of building it. We got a rectangular function. Please.&#36;n&#36;The next is the rectangular function that you want.
&#36;E[X]=n,\quad E[X^2]=n(n+2),\quad Var(X)=2n&#36;
<em>We can do this part of the calculation by bringing the density function directly into the normal distribution.</em></p>
<h3>One example</h3>
<p>Calculate the points
&#36;&#36;00\&amp;(1)\int_{-\infty}^\infty(2x^2+2x+3)e^{-(x^2+2x+3)}dx\text{,}\&amp;(2)\int_0^\infty(4x^2+5x+6)e^{-(2x+1)}dx.\end{aligned}&#36;&#36;
两个问题的思路是一样的，都是需要构造分布 借助矩表示原有的积分，然后用简单的方法得到矩 来进行表示 其核心还是在于构造分布
(1) &#36;&#36;=e^{-2}\int_{-\infty}^{+\infty}\left(2x^{2}+2x+3\right)e^{-\frac{(x+1)^{2&#125;&#125;{2\left(\frac{1}{\sqrt{2&#125;&#125;\right)^{2&#125;&#125;}dx&#36;&#36;
现构造出我们需要的密度函数
&#36;&#36;=sqrt{pi}e^-2}\left(2E(x^2)+2E(x)+3\right) &#36;
It's based on the original definition of the exact same thing.
Bringing the original to the original is the last answer.</p>
<p>The second question is exactly the same idea, but instead of constructing the Gamma distribution function or the index distribution function.</p>
<h3>Multi-RectMatter Functions</h3>
<p><strong>Multi-Rect parent function definition</strong>Assumptions &#36;X = (X 1, \ldots, X n)&#39;&#36; 是一个具有 &#36;n&#36; 维密度 &#36;f(x_1, \ldots, x_n)&#36; 的 &#36;n \times 1&#36; 随机向量。&#36;X&#36; 's multi-fold rectangular parent functions are defined as:
I'm sorry.
The \begin{aligned}
\psi X 1,\ldots,x n} (t 1,\ldots, t n) &amp;= E(e^{t_1 X_1 + \ldots + t_n X_n}) \
&amp;= \int_{-\infty}^{\infty} \cdots \int_{-\infty}^{\infty} e^{t_1 x_1 + \ldots + t_n x_n} f(x_1, \ldots, x_n) , dx_1 \ldots dx_n \
&amp;= E(e^{t&#39; I'm sorry.
@aligned BAR
I'm sorry.
of which &#36;t= (t 1, \ldots, t n)&#39;&#36; 且 &#36;X = (X_1, \ldots, X_n)&#39;&#36;</p>
<p><strong>Nature of multiple rectangular parent functions</strong></p>
<p>Set &#36;X = (X 1, \ldots, X n)&#39;&#36; 且 &#36;t = (t_1, \ldots, t_n)&#39;&#36;，多元矩母函数 &#36;\\psi X(t)&#36; has the following characteristics:</p>
<ol>
<li>&#36;\psi_X(0) = 1&#36;</li>
<li>If &#36;X_1, \ldots, X_n&#36; is independent, and:&#36;&#36;
\psi_X(t) = \prod_{i=1}^{n} \psi_{X_i}(t_i)
&#36;&#36;of which &#36;\psi_{X_i}(t_i)&#36; Yes. &#36;X_i&#36; . The one-dollar rectangular parent function.</li>
<li>You can get the rectangular from multiple rectangles: &#36;&#36;
\loft. \fc(partial 1 \cdots +k})\cdots \cdots \ft. \psi x(t 1, \ldots, \n)\right|<em>{t_1 = \cdots = t_n = 0} = E(X_1^{k_1} \cdots X_n^{k_n})
&#36;&#36;例如，当 &#36;n = 2&#36; 时，&#36;X = (X_1, X_2)&#39;&#36;，我们有：&#36;&#36;
\left. \frac{\partial^5}{\partial t_1^2 \partial t_2^3} \psi_X(t_1, t_2) \right|</em>{t_1 = t_2 = 0} = E(X_1^2 X_2^3)
&#36;&#36;</li>
<li>By not being in marginal distribution &#36;X_j&#36; Corresponding &#36;t_j&#36; Set 0 to be available &#36;X&#36; . Any rectangular function of the marginal distribution. For example, when &#36;n = 4&#36; ,&#36;X = (X 1, X 2, X 3, X 4)&#39;&#36;，&#36;\psi_X(t_1, t_2, t_3, t_4)&#36; 是 &#36;X&#36; 的多元矩母函数。则 &#36;\psi_{X_1,X_3}(t_1, t_3) = \psi_X(t_1, 0, t_3, 0)&#36;，&#36;\psi (t 1) = \\psi X (t 1, 0, 0) &#36; etc.</li>
<li>Multipurpose feature function defined as &#36;\phi X(t) = E(e^it)&#39;X})&#36;，其中 &#36;i = \sqrt{-1}&#36;。特征函数总是对任何随机变量或向量存在，但矩母函数对某些随机变量可能不存在。因此，特征函数在证明某些结果时比矩母函数更有用。特征函数与矩母函数的关系为：&#36;&#36;
\phi_X(t_1, \ldots, t_n) = \psi_X(it_1, \ldots, it_n)
&#36;&#36;</li>
</ol>
<h2>Feature Functions</h2>
<p>As previously studied, we need to find a function that exists for any random variable or vector to paint a random variable and a random vector.</p>
<h3>Introduction</h3>
<p>We've been able to introduce the function of the Fouriers in mathematical analysis.
&#36;&#36;\varphi\left(t\right)=\int_{-\infty}^{\infty}e^{itx}p\left(x\right)dx&#36;&#36;
If&#36;p(x)&#36;It's a density function. &#36;\varphi(t)=E\left(e^{itX}\right)&#36; </p>
<p>And that's the problem of the feature function that we're looking at here, which is a very good tool for dealing with a lot of probabilities, which can simplify calculations and thinking.</p>
<h3>Definition of feature functions</h3>
<p>The definition of a duplicate random variable is as &#36;Z=Z\left(w\right)=X\left(w\right)+iY\left(w\right)&#36;  He has the nature to be close to normal, real-value random variables, but all the operational ranges are complex.</p>
<p><strong>Definitions</strong>: Set&#36;X&#36;A random variable called
&#36;&#36;
\phi(t) = E(e^{itX}) = \int_{-\infty}^{\infty} e^{itx} , dF(x)
&#36;&#36;
Yes.&#36;X&#36;, we know that. &#36;X&#36;. The distribution function is determined only by its feature function.</p>
<p>The feature function of the discrete random variable is
&#36;&#36;\varphi\left(t\right)=\sum_{k=1}^{\infty}\mathrm{e}^{\mathrm{i}tx_{k&#125;&#125;p_{k}&#36;&#36;
The feature function of a continuous random variable is
&#36;&#36;\varphi\left(t\right)=\int_{-\infty}^{\infty}e^{itx}p\left(x\right)dx&#36;&#36;
The feature function is also the only dependent and distributed function that we can also call the feature function of the distribution.</p>
<p><strong>The feature function is essentially a Fourier transformation of probability, i.e., a probability measure.</strong>
&#36;&#36;\Phi_{\mu}(\theta):=\int_{\mathbb{R}^{n&#125;&#125; e^{\mathrm{i} \theta x} \mu(\mathrm{~d} x)=\int_{\mathbb{R&#125;&#125; \cos (\theta x) \mu(\mathrm{d} x)+\mathrm{i} \int_{\mathbb{R}^{n&#125;&#125; \sin (\theta x) \mu(\mathrm{d} x)&#36;&#36;
As we have described earlier, the basis for expectations and points shows, this is not in contradiction with our primary definition.</p>
<h3>Feature Functions of Common Distribution</h3>
<h4>Single Point Distribution</h4>
<p>&#36;&#36;\varphi\left(t\right)=e^{ita}&#36;&#36;
Of which&#36;a&#36;is the density function to be taken&#36;1&#36;Point</p>
<h4>Distribution of 0-1</h4>
<p>&#36;&#36;\varphi\left(t\right)=pe^{it}+q&#36;&#36;</p>
<h4>Two distributions</h4>
<p>&#36;&#36;\varphi\left(t\right)=\left(pe^{jt}+q\right)^{n}&#36;&#36;</p>
<h4>Porcelain distribution</h4>
<p>&#36;&#36;\varphi\left(t\right)=\sum_{k=0}^{\infty}e^{ikt}\frac{\lambda^{k&#125;&#125;{k!}e^{-\lambda}=e^{-\lambda}e^{\lambda e^{it&#125;&#125;=e^{\lambda\left(e^{it}-1\right)}.&#36;&#36;</p>
<h4>even distribution</h4>
<p>&#36;&#36;\varphi\left(t\right)=\int_{a}^{b}\frac{e^{itx&#125;&#125;{b-a}dx=\frac{e^{ibt}-e^{iat&#125;&#125;{it\left(b-a\right)}.&#36;&#36;</p>
<h4>Standard normal distribution</h4>
<p>&#36;&#36;\varphi\left(t\right)=e^{-\frac{t^2}{2&#125;&#125;&#36;&#36;</p>
<h4>Index distribution</h4>
<p>&#36;&#36;\varphi\left(t\right)=\left(1-\frac{it}{\lambda}\right)^{-1}&#36;&#36;</p>
<h4>Normal distribution</h4>
<p>&#36;&#36;\varphi\left(t\right)=e^{i\mu t-\frac{\sigma^{2}t^{2&#125;&#125;2}&#36;&#36;</p>
<h4>Gamma distribution</h4>
<p>&#36;&#36;\varphi\left(t\right)=\left(\frac\lambda{\lambda-jt}\right)^r&#36;&#36;</p>
<h3>Nature of feature function</h3>
<ul>
<li>&#36;\varphi(0)=1,|\varphi(t)|\leq\varphi(0)&#36;</li>
<li>&#36;\varphi(-t)=\bar{\varphi}(t)&#36;</li>
<li>&#36;Y=aX+b&#36; then&#36;\varphi_Y(t)=e^{ibt}\varphi_X(at)&#36; </li>
<li>&#36;Z=X+Y&#36; And independent of each other &#36;\varphi_Z(t)=\varphi_X(t)\varphi_Y(t)&#36;</li>
<li>&#36;\varphi^{(k)}(0)=i^kE[X^k]&#36; If the rectangular exists and the characteristic function is able to guide</li>
</ul>
<p><strong>Theoretically.</strong>(Reversive formula):&#36;F(x)&#36; It's a distribution function. &#36;\varphi(t)&#36; It's a feature function, and there is.
&#36;&#36;F\left(x_{2}\right)-F\left(x_{1}\right)=\lim_{r\to\infty}\frac{1}{2\pi}\int_{-r}^{r}\frac{e^{-ix_{1&#125;&#125;-e^{-ix_{2&#125;&#125;}{it}\varphi\left(t\right)dt.&#36;&#36;
By reversing the formula, we can export the only distribution function from the feature function.
Which means... <strong>The only thing that makes sense between the feature and distribution functions</strong></p>
<h3>Multipurpose Feature Functions</h3>
<p><strong>Multi-Purpose Function Definition</strong> : If Random Vector &#36;(\xi_1, \ldots, \xi_n)&#36; The distribution function is &#36;F(x_1, \ldots, x_n)&#36;, whose feature function is defined as:
&#36;&#36;
\phi(t_1, \ldots, t_n) = E(e^{i(t_1 \xi_1 + \ldots + t_n \xi_n)}) = \int_{-\infty}^{\infty} \cdots \int_{-\infty}^{\infty} e^{i(t_1 x_1 + \ldots + t_n x_n)} , dF(x_1, \ldots, x_n)
&#36;&#36;</p>
<p><strong>Theories of Monopolies</strong>: Distribution Functions &#36;F(x_1, \ldots, x_n)&#36; is the only function of its feature.</p>
<p><strong>Nature of multiple feature functions:</strong></p>
<ol>
<li>&#36;\phi(t_1, \ldots, t_n)&#36; Yes. &#36;\mathbb{R}^n&#36; consistent with continuous, and &#36;|\phi(t_1, \ldots, t_n)| \leq \phi(0, \ldots, 0) = 1&#36;</li>
<li>If &#36;(\xi_1, \ldots, \xi_n)&#36; The feature function is &#36;\phi(t_1, \ldots, t_n)&#36;, &#36;\eta = a 1 \lts + a \n = a&#39; \xi&#36; 的特征函数为：&#36;&#36;
\phi_\eta(t) = \phi(a_1 t, \ldots, a_n t) = \phi(ta)
&#36;&#36;</li>
<li>If &#36;E(\xi_1^{k_1} \cdots \xi_n^{k_n})&#36; exists, and:&#36;&#36;
\left. \frac{\partial^{k_1 + \cdots + k_n&#125;&#125;{\partial t_1^{k_1} \cdots \partial t_n^{k_n&#125;&#125; \phi(t_1, \ldots, t_n) \right|_{t_1 = \ldots = t_n = 0} = i^{k_1 + \cdots + k_n} E(\xi_1^{k_1} \cdots \xi_n^{k_n})
&#36;&#36;</li>
<li>If &#36;\xi_j&#36; The feature function is &#36;\phi_{\xi_j}(t_j)&#36;，&#36;j = 1, 2, \ldots, n&#36;, and then random variable &#36;\xi_1, \ldots, \xi_n&#36; The following are the essential conditions for independence:&#36;&#36;
\phi(t_1, \ldots, t_n) = \phi_{\xi_1}(t_1) \cdots \phi_{\xi_n}(t_n)
&#36;&#36;</li>
<li>If you're not... &#36;\phi(t_1, \ldots, t_n, u_1, \ldots, u_m)&#36;，&#36;\phi(t_1, \ldots, t_n)&#36; and &#36;\phi(u_1, \ldots, u_m)&#36; Separate Random Vector &#36;(\xi_1, \ldots, \xi_n)&#36;，&#36;(\eta_1, \ldots, \eta_m)&#36; and &#36;(\xi_1, \ldots, \xi_n, \eta_1, \ldots, \eta_m)&#36; , and then &#36;(\xi_1, \ldots, \xi_n)&#36; and &#36;(\eta_1, \ldots, \eta_m)&#36; Independence is fully contingent on: the actual number &#36;t_1, \ldots, t_n&#36; and &#36;u_1, \ldots, u_m&#36; Yes:
&#36;&#36;
\phi(t_1, \ldots, t_n, u_1, \ldots, u_m) = \phi(t_1, \ldots, t_n) \phi(u_1, \ldots, u_m)
&#36;&#36;</li>
<li>If &#36;Y = AX + b&#36;, random vector &#36;Y&#36; , and the feature function is:
I'm sorry.
\phi Y(t)=E(e^it)&#39;(AX + b)}) = e^{it&#39;b} \phi_X(A&#39;t)
&#36;&#36;</li>
</ol>
<h2>Multiple normal distribution in the perspective of the feature function</h2>
<h3>Multiple Normal Distribution Definition</h3>
<p><strong>Definitions</strong>Assumptions &#36;X = (X 1, \ldots, X n)&#39;&#36;。若 &#36;X&#36; with density function:
I'm sorry.
f(x)=(2\pi)^n/2}|Sigma^1/2}\exp\left{-\frac}(x-\mu)&#39; I'm sorry, I'm sorry.
I'm sorry.
And then, "Could" &#36;X&#36; With Average &#36;\mu&#36;. The matrix of the agreement is &#36;\Sigma&#36; Yes. &#36;n&#36; We have a multi-dimensional normal distribution.</p>
<p><strong>Definition of Equivalence</strong>Assumptions &#36;X = (X 1, \ldots, X n)&#39;&#36;。若 &#36;X&#36; with rectangular function:
I'm sorry.
\psi X(t)=\exp\left{t&#39;\mu + \frac{1}{2} t&#39; \Sigma t \right}
&#36;&#36;
或者特征函数：
&#36;&#36;
\phi_X(t) = \exp\left{ it&#39;\mu - \frac{1}{2} t&#39; ♪ The way you're gonna be ♪
I'm sorry.
And then, "Could" &#36;X&#36; With Average &#36;\mu&#36;. The matrix of the agreement is &#36;\Sigma&#36; Yes. &#36;n&#36; Multiplastic normal distribution, recorded &#36;X \sim N_n(\mu, \Sigma)&#36;I'm sorry. To prove it, yes. From&#36;Y \sim N_n(0, I)&#36; ..the feature function starts, uses a transformation&#36;X = \Sigma^{1/2} Y + \mu&#36; The shape-forming function allows you to get the form above.</p>
<h3>Nature of multiple normal distribution</h3>
<ol>
<li><p><strong>Linear transformation</strong>: If &#36;X \sim N_n(\mu, \Sigma)&#36;, &#36;Y = AX + b \sim n (A\mu + b, A\Sigma A)&#39;)&#36;
<strong>Prove it.</strong>：&#36;Y&#36; , and the feature function is:
I'm sorry.
The \begin{aligned}
\phi Y(t) &amp;= E(e^{it&#39;(AX + b)}) = e^{it&#39;b} \phi_X(A&#39;t) \
&amp;= e^{it&#39;b} \exp\left{ i(A&#39;t)&#39;\mu - \frac{1}{2} (A&#39;t)&#39; \Sigma (A&#39;t) \right} \
&amp;= \exp\left{ it&#39;(A\mu + b) - \frac{1}{2} t&#39; (A\Sigma A&#39;@t\right}
@aligned BAR
I'm sorry.
So, &#36;Y \sim n (A\mu+b, A\Sigma A)&#39;)&#36;。</p>
</li>
<li><p><strong>Linear combination of independent normal variables</strong>: Assumptions &#36;X_1, \ldots, X_k&#36; It's independent and each one &#36;X_i \sim N_n(\mu_i, \Sigma_i)&#36;，&#36;i = 1, \ldots, k&#36;I don't know. Assumptions &#36;a_1, \ldots, a_k&#36; It's a mark, definition:
&#36;&#36;
Y = a_1 X_1 + \ldots + a_k X_k
&#36;&#36;
then &#36;Y\sim n (mu^)<em>, \Sigma^</em>)&#36;，其中 &#36;\mu^* = \sum_{i=1}^{k} a_i \mu_i&#36; 且 &#36;♪ I'm not gonna let you go ♪ This can be demonstrated by using rectangular parent functions.</p>
</li>
<li><p><strong>Marginal distribution</strong>: If &#36;X \sim N_n(\mu, \Sigma)&#36;, will &#36;X&#36; Division to &#36;X = \begin{pmatrix} X_1 \ X_2 \end{pmatrix}&#36;, of which &#36;X_1&#36; Yes. &#36;r \times 1&#36;，&#36;X_2&#36; Yes. &#36;(n-r) \times 1&#36;I'm sorry. Will &#36;\mu&#36; Division to &#36;\mu = \begin{pmatrix} \mu_1 \ \mu_2 \end{pmatrix}&#36;, of which &#36;\mu_1&#36; Yes. &#36;r \times 1&#36;，&#36;\mu_2&#36; Yes. &#36;(n-r) \times 1&#36;I'm sorry. Similar to the one you're going to have. &#36;\Sigma&#36; The segments are:
I'm sorry.
\Sigma = \begin{matrix} \Sigma 11} &amp; \Sigma_{12} \ \Sigma_{21} &amp; The PHP has a new name.
I'm sorry.
of which &#36;\Sigma_{11}&#36; Yes. &#36;r \times r&#36;，&#36;\Sigma_{12}&#36; Yes. &#36;r \times (n-r)&#36;，&#36;\Sigma_{21} = \Sigma_{12}&#39;&#36; 是 &#36;(n-r) \times r&#36;，&#36;\Sigma_{22}&#36; 是 &#36;(n-r) \times (n-r)&#36;。&#36;X_1&#36; 的边际分布为 &#36;X_1 \sim N_r(\mu_1, \Sigma_{11})&#36;。这可以通过在 &#36;X_2&#36; 对应的 &#36;t j&#36; places zero to prove using rectangular parent functions.</p>
</li>
<li><p><strong>Conditional distribution</strong>: If &#36;X \sim N_n(\mu, \Sigma)&#36;, use the division in 3) if:
&#36;&#36;
X_1 | X_2 = x_2 \sim N_r\left( \mu_1 + \Sigma_{12} \Sigma_{22}^{-1} (x_2 - \mu_2), \Sigma_{11.2} \right)
&#36;&#36;
of which &#36;\Sigma_{11.2} = \Sigma_{11} - \Sigma_{12} \Sigma_{22}^{-1} \Sigma_{21}&#36;。</p>
</li>
<li><p><strong>Conditions of independence</strong>: If &#36;X \sim N_n(\mu, \Sigma)&#36;, will &#36;X&#36; Division to &#36;X = \begin{pmatrix} X_1 \ X_2 \end{pmatrix}&#36;, of which &#36;X_1&#36; Yes. &#36;r \times 1&#36;，&#36;X_2&#36; Yes. &#36;(n-r) \times 1&#36;I'm sorry. Will &#36;\mu&#36; Division to &#36;\mu = \begin{pmatrix} \mu_1 \ \mu_2 \end{pmatrix}&#36;, of which &#36;\mu_1&#36; Yes. &#36;r \times 1&#36;，&#36;\mu_2&#36; Yes. &#36;(n-r) \times 1&#36;I'm sorry. Similar to the one you're going to have. &#36;\Sigma&#36; The segments are:
I'm sorry.
\Sigma = \begin{matrix} \Sigma 11} &amp; \Sigma_{12} \ \Sigma_{21} &amp; The PHP has a new name.
I'm sorry.
of which &#36;\Sigma_{11}&#36; Yes. &#36;r \times r&#36;，&#36;\Sigma_{12}&#36; Yes. &#36;r \times (n-r)&#36;，&#36;\Sigma_{21} = \Sigma_{12}&#39;&#36; 是 &#36;(n-r) \times r&#36;，&#36;\Sigma_{22}&#36; 是 &#36;(n-r) \times (n-r)&#36;。则 &#36;X_1&#36; 和 &#36;X_2&#36; 独立当且仅当 &#36;\Sigma_{12} = 0&#36;。</p>
</li>
<li><p><strong>Complete Character</strong>: If &#36;X \sim N_n(\mu, \Sigma)&#36;, and &#36;X&#36; All marginal distributions, distributions of conditions and &#36;X&#36; The linear combination of the mass is a multi-normal distribution.</p>
<p><strong>Attention.</strong>The counter-arguments of the aberrations are not valid. For example, if all marginal distributions are multiple normal distributions, this does not mean that joint distribution is multiple normal distributions.</p>
</li>
<li><p><strong>Unique</strong>: Multiple normal distributions are expressed entirely in their average vector and their coordinated arrays. This means that once the mean vector and the allergic matrix is specified, the density and rectangular function of the MVN is fully established.</p>
</li>
<li><p><strong>Symmetrical</strong>: If &#36;X \sim N_n(\mu_x, \Sigma_x)&#36; and &#36;Y \sim N_m(\mu_y, \Sigma_y)&#36;，&#36;X&#36; and &#36;Y&#36; Independent, in the case of:
&#36;&#36;
\begin{pmatrix} X \ Y \end{pmatrix} \sim N_{n+m}(\mu, \Sigma)
&#36;&#36;
of which &#36;\mu = \begin{pmatrix} \mu_x \ \mu_y \end{pmatrix}&#36; And... &amp; 0 \ 0&#39; &amp; \Sigma_y \end{pmatrix}&#36;。</p>
</li>
<li><p><strong>Centre</strong>: If &#36;X \sim N_n(\mu, \Sigma)&#36;, and &#36;E(X) = \text{mode}(X) = \text{median}(X) = \mu&#36;。</p>
</li>
</ol>
<h3>Recognize polynormal distributions</h3>
<p><strong>Theoretically.</strong>Assumptions &#36;X = (X 1, \ldots, X n)&#39;in the form of:
&#36;&#36;
f(x) = c \exp{-Q/2}
&#36;&#36;
density of \\exp{-Q/2} \\propto\exp\left{-\frac{2} (x-mu)&#39; \Sigma^{-1} (x - \mu) \right}&#36; 且 &#36;c&#36; 是归一化常数。则 &#36;X \sim N_n(\mu, \Sigma)&#36;。</p>
<hr>
<p><strong>Example:</strong>: Assumed &#36;X = (X 1, X 2)&#39;&#36; 具有形式为 &#36;f(x) = c \exp{-Q/2}&#36; 的密度，其中 &#36;Q = x_1^2 + 2x_1 x_2 + 4x_2^2 + 2x_1&#36;。&#36;What's the distribution of X-&#36;?</p>
<p>We know. &#36;X&#36; It must be a plural normal distribution. We're gonna find it. &#36;E(X)&#36;We have:
I'm sorry.
The \begin{aligned}
I'm sorry, I'm sorry. &amp;= 2x_1 + 2x_2 + 2 = 0 \
\frac{\partial Q}{\partial x_2} &amp;= 2x 1 + 8x 2 = 0
@aligned BAR
I'm sorry.
We'll solve these equations. &#36;x_1 = -4/3&#36; and &#36;x_2 = 1/3&#36;I'm sorry. So &#36;m= (4/3, 1/3)&#39;&#36;。</p>
<p>We're gonna find it. &#36;\Sigma^{-1}&#36; The elements. Let's check. &#36;Q&#36; Two orders.
I'm sorry.
== sync, corrected by elderman == &amp; \sigma^{(12)} \ \sigma^{(12)} &amp; \sigma^{(22)} \end{pmatrix}
&#36;&#36;
因此：
&#36;&#36;
x&#39; \Sigma^{-1} x = (x_1, x_2) \begin{pmatrix} \sigma^{(11)} &amp; \sigma^{(12)} \ \sigma^{(12)} &amp; \sigma^{(22)} \end{pmatrix} (x_1, x_2)&#39; = \sigma^{(11)} x_1^2 + 2\sigma^{(12)} x_1 x_2 + \sigma^{(22)} x_2^2
&#36;&#36;
对于我们的问题，&#36;\sigma^{(11)} = 1&#36;，&#36;2\sigma^{(12)} = 2&#36; 且 &#36;\sigma^{(22)} = 4&#36;。因此：
&#36;&#36;
\Sigma^{-1} = \begin{pmatrix} 1 &amp; 1 \ 1 &amp; 4 \end{pmatrix}
&#36;&#36;
对 &#36;\Sigma^{-1}&#36; 求逆得到：
&#36;&#36;
\Sigma = \begin{pmatrix} 4/3 &amp; -1/3 \ -1/3 &amp; 1/3 \end{pmatrix}
&#36;&#36;</p>
<p>Check the linear items. We have:
I'm sorry.
- I'm sorry, I'm sorry. &amp; 1 \ 1 &amp; 4 \begin{matrix} -4/3 \end{matrix} = -2 \begin{matrix} -1 \end{matrix} \begin{matrix} 2 \0 \end{matrix}
I'm sorry.
So, &#36;-2x&#39;\Sigma^{-1}\mu = 2x_1&#36;，这是 &#36;Linear items in Q&#36;. Therefore:
I'm sorry.
X\sim 2\left (\begin{matrix} -4/3 \end{matrix},\begin{matrix} 4/3 &amp; -1/3 \ -1/3 &amp; 1/3 \end{pmatrix} \right)
&#36;&#36;</p>
<h2>Three distributions in the perspective of the feature function</h2>
<h3>Carside distribution</h3>
<p><strong>Definitions</strong>: Random variable &#36;X&#36; It's called having &#36;n&#36; Central Cube Distribution of Free Degrees (Writing) &#36;X \sim \chi^2(n)&#36;) if &#36;X&#36; has density:
&#36;&#36;
f(x) = \frac{1}{\Gamma(n/2)} \left(\frac{1}{2}\right)^{n/2} x^{n/2 - 1} e^{-x/2}
&#36;&#36;
of which &#36;\Gamma(\alpha) = \int_0^\infty y^{\alpha - 1} e^{-y} dy&#36;。</p>
<p>Random Variables &#36;X&#36; Rectangular Mother Function (MGF) &#36;\psi_X(t)&#36; Defined as:
&#36;&#36;
\psi_X(t) = E(e^{tX}) = \int_{-\infty}^{\infty} e^{tx} f(x) dx
&#36;&#36;
If &#36;X&#36; Separated, then the fractions are taken and replaced.</p>
<p><strong>Theoretically.</strong>: If &#36;X \sim \chi^2(n)&#36;, and &#36;\psi_X(t) = (1 - 2t)^{-n/2}&#36;。</p>
<p><strong>Theoretically.</strong>: Assumptions &#36;Z_1, \ldots, Z_n&#36; It's separate and distributed. &#36;N(0, 1)&#36; Random variable. Definitions:
&#36;&#36;
X = \sum_{i=1}^{n} Z_i^2
&#36;&#36;
then &#36;X \sim \chi^2(n)&#36;。</p>
<hr>
<h3>Non-centre-based distribution of cards</h3>
<p><strong>Definitions</strong>: Set &#36;Z_1, \ldots, Z_n&#36; It's independent.&#36;Z_i \sim N(\mu_i, 1)&#36;I'm sorry. then &#36;W = \sum_{i=1}^{n} Z_i^2&#36; Yes &#36;n&#36; Freedom, non-centre parameters &#36;\gamma = \frac{1}{2} \sum_{i=1}^{n} \mu_i^2&#36; * Non-centre-based distribution of cards. We write. &#36;W \sim \chi^2(n, \gamma)&#36;。</p>
<p>Non-centre-carat distribution appears in the hypothetical tests, especially when there is interest in testing the statistical distribution under alternative scenarios in linear models.</p>
<p><strong>Theoretically.</strong>: Assumptions &#36;Y_1, \ldots, Y_n&#36; It's independent.&#36;Y_i \sim N(\mu_i, \sigma^2)&#36;，&#36;i = 1, \ldots, n&#36;I'm sorry. Definitions:
&#36;&#36;
X = \frac{1}{\sigma^2} \sum_{i=1}^{n} Y_i^2
&#36;&#36;
then &#36;X \sim \chi^2(n, \gamma)&#36;, of which &#36;\gamma = \frac{1}{2\sigma^2} \sum_{i=1}^{n} \mu_i^2&#36;。</p>
<p><strong>Nature of non-centre-based distribution of the card</strong>：</p>
<ol>
<li><p>If &#36;X \sim \chi^2(n, \gamma)&#36;, and:
&#36;&#36;
\psi_X(t) = (1 - 2t)^{-n/2} \exp\left{ \frac{2\gamma t}{1 - 2t} \right}
&#36;&#36;
This can be demonstrated by using the definition of non-centre card square density and by exchanging points and the order of the sum.</p>
</li>
<li><p>If &#36;X \sim \chi^2(n, \gamma)&#36;, and &#36;E(X) = n + 2\gamma&#36; and &#36;\text{Var}(X) = 2n + 8\gamma&#36;I'm sorry. This can be demonstrated by using the rectangular function in 1)</p>
</li>
<li><p>If &#36;X \sim \chi^2(n, \gamma)&#36; and &#36;\gamma = 0&#36;, which corresponds to &#36;n&#36; A central, free-scale, card-based random variable. That is,&#36;X \sim \chi^2(n, 0) = \chi^2(n)&#36;。</p>
</li>
</ol>
<hr>
<h3>t Distribution</h3>
<p><strong>Definitions</strong>: Assumptions &#36;X \sim N(0, 1)&#36;，&#36;Y \sim \chi^2(n)&#36;and &#36;X&#36; and &#36;Y&#36; Independence. Defines random variables:
&#36;&#36;
T = \frac{X}{\sqrt{Y/n&#125;&#125;
&#36;&#36;
then &#36;T&#36; It's called having &#36;n&#36; A t distribution of freedom. We write. &#36;T \sim t(n)&#36;。</p>
<p><strong>Non-centre t distribution</strong>: Assumptions &#36;X \sim N(\mu, 1)&#36; and &#36;Y \sim \chi^2(n)&#36;and &#36;X&#36; and &#36;Y&#36; Independence. Defines random variables:
&#36;&#36;
W = \frac{X}{\sqrt{Y/n&#125;&#125;
&#36;&#36;
then &#36;W&#36; It's called having &#36;n&#36; Freedom, non-centre parameters &#36;\mu&#36; . The non-centre t distribution. We write. &#36;W \sim t(n, \mu)&#36;I'm sorry. If &#36;\mu = 0&#36;, and &#36;W&#36; Simplified as with &#36;n&#36; A free center t distribution.</p>
<hr>
<h3>F distribution</h3>
<p><strong>Definitions</strong>: Assumptions &#36;X_1 \sim \chi^2(n_1, \gamma_1)&#36; and &#36;X_2 \sim \chi^2(n_2, \gamma_2)&#36;and &#36;X_1&#36; and &#36;X_2&#36; Independence. Defines random variables:
&#36;&#36;
F = \frac{X_1 / n_1}{X_2 / n_2}
&#36;&#36;
then &#36;F&#36; It's called having &#36;(n_1, n_2)&#36; Freedom, non-centre parameters &#36;(\gamma_1, \gamma_2)&#36; is the dual uncentre F distribution. We write. &#36;F \sim F(n_1, n_2, \gamma_1, \gamma_2)&#36;。</p>
<p>(a) If &#36;\gamma_2 = 0&#36;, and &#36;F&#36; Called the non-centre F distribution. We say, "I'm not a man."&#36;F \sim F(n_1, n_2, \gamma_1)&#36;</p>
<p>(b) If &#36;\gamma_1 = 0&#36; and &#36;\gamma_2 = 0&#36;, and &#36;F&#36; Called the center F distribution. We say, "I'm not a man." &#36;F \sim F(n_1, n_2)&#36;。</p>
<p>Centre F is distributed in hypothetical tests of embedded linear models. In this case, the distribution of statistical data is usually measured in the original scenario with a central F distribution. Non-centre F distribution is derived from the distribution of statistical tests under alternative scenarios. Testing the distribution of statistical data under alternative scenarios is important for the calculation of efficacy.</p>
<p><strong>F Nature of distribution</strong>: If &#36;F \sim F(n_1, n_2, \gamma)&#36;, and:</p>
<p>a) &#36;E(F) = \frac{n_2(n_1 + 2\gamma)}{n_1(n_2 - 2)}&#36;，&#36;n_2 &gt; 2&#36;</p>
<p>b) &#36;\text{Var}(F) = 2 \left(\frac{n_2}{n_1}\right)^2 \frac{(n_1 + 2\gamma)^2 + (n_1 + 4\gamma)(n_2 - 2)}{(n_2 - 2)^2 (n_2 - 4)}&#36;，&#36;n_2 &gt; 4&#36;</p>
<p>Set by setting in a) and b) &#36;\gamma = 0&#36;, you can get the average and the variance formula of the centre F.</p>
