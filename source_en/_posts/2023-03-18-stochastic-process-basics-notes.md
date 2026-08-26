---
title: 'Stochastic Processes: Definitions, Stationarity, and Markov Foundations'
title_zh: 随机过程基础：随机过程定义、数字特征与平稳过程
date: 2023-03-18 21:28:17 +0800
categories:
- Data Science
- Time Series & Spatial Data
tags:
- Statistics
- Stochastic Processes
author: Hyacehila
mathjax: true
hidden: true
excerpt: Covers definitions, numerical characteristics, stationary processes, Poisson processes, Brownian motion, and Markov
  process foundations.
description: Covers definitions, numerical characteristics, stationary processes, Poisson processes, Brownian motion, and
  Markov process foundations.
excerpt_zh: 整理随机过程定义、数字特征、平稳过程、泊松过程、布朗运动和马尔可夫过程基础。
permalink: /blog/2023/03/18/stochastic-process-basics-notes/
lang: en
translation_key: 2023-03-18-stochastic-process-basics-notes
translation_status: machine
translation_source_hash: 020c0e58fe28c28b7b1db1ef9556aafe78a001adf0ce9c3f9f99521d409160ec
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<h2>Basic knowledge of random processes</h2>
<h3>Definition of random processes</h3>
<p><strong>How to quantify and fully reflect a random phenomenon</strong>
In probability theory and mathematical statistics, we introduce random variables and random vectors to paint a random process.
But can he really paint a complete random process?
Here's an example.
Look at some sort of oscillator output wave shape.
&#36;&#36;X_t=A\cos(\omega t+\Phi)&#36;&#36;
Of which &#36;A&gt;0&#36;为常数 &#36;w&#36;为常数 随机变量&#36;\Phi&#36; 服从&#36;[0,2\pi]&#36; 上的均匀分布 &#36;t\in[0, \infty]
It's a random process. Every single one.&#36;t&#36;This random variable is a function of another random variable.</p>
<h4>Definition giving</h4>
<p>If you give it to me,&#36;t_{0}&#36;, &#36;(0,t0]&#36;The number of visits to the website during the time is as follows:&#36;X_{t0}&#36;A random variable
In fact, this random variable is not a complete picture of this random phenomenon.&#36;t_{0}&#36; The random variable that changes and changes is the perfect way to paint this random phenomenon.
Remember the random variable of the family as &#36;&#123;X_t, t∈[0,\infty)}&#36;
So we've been promoting the following random process definitions:
&#36;&#36;00\
&amp;\\text{} (\Omega,\mathsf{F}, \mathsf{P}\text{is a probability space}, \mathsf{T}\text{is a parameter set, \mathsf{R}, \\mathsf}
&amp;If each t\in T is defined on (\Omega, F, P) \
&amp;\text{random variable}\masf{X} (\omega, \t), (\omega(}Omega)\text{correspond}, \
&amp;\\text{x}(\mega, \mathsf}(\mega, \mathsf{F}, \mathsf{P}\text{a random process on} (\mathsf{S}.\mathsf{P}.)
I'm sorry, I'm sorry.
We usually put it in brief.&#36;&#123;X_t, t∈T}&#36;<br>This.&#36;w&#36;It's a parameter that's painted for random variables, and it reflects the randomness of random variables.
Parameter Set&#36;T&#36; The following classification criteria are used for the random process, which generally indicates time or space.
&#36;\mathbf{T={0,1,2,...&#125;&#125;\text{或 T={-2,-1,0,1,2,...&#125;&#125;&#36; , which is called random process of discrete parameters (random time series)
&#36;\mathbf{T=[a,b]}&#36; , which is called a continuous parameter random process.</p>
<h4>Further explanation</h4>
<ul>
<li>&#36;\color{}{\text{X(ω,t)&#125;&#125;&#36; It's both random and functional. It's a piece on&#36;t&#36;function in&#36;t&#36;He was a random variable at the time of the determination, and because random variables are essentially a function, the nature of random processes is defined as&#36;T\times Q&#36;Dinosaur Monument</li>
<li>For every fixed&#36;t&#36; &#36;\color{}{\text{X(ω,t)&#125;&#125;&#36;A random variable &#36;X_{t} ~t\in T&#36;The collection of all possible values is called the state space of the random process.&#36;S&#36; The elements of the state space are called state.&#36;S&#36;</li>
<li>For a certain&#36;\boldsymbol{\omega_0}\in\boldsymbol{\Omega}&#36; &#36;\color{}&#123;&#123;X(w_0,t)&#125;&#125;&#36;   It's one.&#36;T&#36;And the normal function on it is called a sample function of a random process, his graphics are called sample curves.</li>
</ul>
<h4>Sample orbital continuity</h4>
<p>There's a random process.&#36;&#123;X_t, t∈T}&#36;   If for any of them,&#36;t\in T&#36; Yes.
&#36;&#36;\mathbb{P}(\lim_{s\to t}|X_s-X_t|=0)=1&#36;&#36;
Name of random process&#36;X&#36;Yes.&#36;T&#36;The probability is continuous, and at this point the random process has a continuous sample orbit.
<strong>Probability one is the most important.</strong>
Of course, the definition of continuity is not unique.
If it's any of your business,&#36;t\in T&#36; And constant &#36;\epsilon&gt;- Yes, I do.
&#36;&#36;\lim_{s\to t}\mathrm{P}(\left|X_s-X_t\right|\geq\varepsilon)=0&#36;&#36;
Called random process X on T-based probability continuum (random continuous)
If for any reason&#36;t\in T&#36; and&#36;P\ge 1&#36; Yes.<br>&#36;&#36;\mathbb{E}|{X}_t|^p&lt;You're not gonna get away with this.
Called&#36;L^p&#36;Continuous &#36;p=2&#36;It's called the average continuity.
&#36;L^p&#36;It's always random.</p>
<h4>Classification of random processes</h4>
<p>Based on<strong>Parameter Set&#36;T&#36;</strong> and <strong>Status Space</strong> It's not hard to understand how to classify successive and discrete.</p>
<h5>Dispersion parameters</h5>
<p>Studying the growth of biological groups
You're the one who's gonna get you.&#36;X_n&#36; For the first time&#36;n&#36;Generations of biological groups
We need random variables.&#36;X_{n},n=1,2,...&#36;</p>
<h5>Dispersive parameters, continuous status</h5>
<p>Look at the highest temperature in a certain area.
You're the one who's gonna get you.&#36;X_t&#36; For the first time&#36;t&#36;Highest temperature observed
We need random variables.&#36;X_{t},t=1,2,...&#36;</p>
<h5>Continuous parameters Dispersed</h5>
<p>Visits to the website
For given&#36;t_{0}&#36; From 0 to&#36;t_{0}&#36; Visits to the website&#36;X_{t_{0&#125;&#125;&#36; A random variable
We need random variables.&#36;X_{t_{0&#125;&#125;,t\in[0,\infty)&#36;</p>
<h5>Continuous</h5>
<p>Look at some sort of oscillator output wave shape.
&#36;&#36;X_t=A\cos(\omega t+\Phi)&#36;&#36;
Of which &#36;A&gt;0&#36;为常数 &#36;w&#36;为常数 随机变量&#36;\Phi&#36; 服从&#36;[0,2\pi]&#36; 上的均匀分布 &#36;t\in[0, \infty]
It's a random process. Every single one.&#36;t&#36;It all corresponds to a random variable.
This random variable is a function of another random variable.</p>
<h4>Some random processes that are common.</h4>
<h5>The Bernouli process</h5>
<p>Set Random Process&#36;X_{n},n=1,2,...&#36;
If the random variable is&#36;X_i&#36; They're separate and distributed, and they're subject to Bernouli's distribution.
And this random process is called the Bernoulian process.
The Bernoulian process describes a series of random experiments that are independent and distributed.
<strong>Dispersion parameters</strong></p>
<h5>Two Processes</h5>
<p>If you order &#36;S_n=\sum_{k=1}^nX_k,\quad S_0=0&#36;
Obviously. &#36;S_{n},n=1,2,...&#36;  It's a random process.
We call it a two-process.
Just as two distributions are the supersing of the one-one distribution, so is two processes.
<strong>Dispersion parameters</strong></p>
<h5>Jinghus White Noise Process</h5>
<p>Set Random Process&#36;X_{n},n=1,2,...&#36;
If the random variable is&#36;X_i&#36; <strong>Independence and distribution</strong>
And follow Gauss distribution. &#36;N(0,\sigma^2)&#36;
Name&#36;X&#36;For the White Noise Process of Jinghus
<strong>Dispersive parameters, continuous status</strong></p>
<h5>Counting process</h5>
<p>If&#36;N_t&#36;Means to the moment&#36;t&#36;The total number of random events that have occurred to date is referred to as the random process&#36;&#123;N_t,t≥0}&#36;For Count Process
- I usually do. &#36;N_t&#36; is a non-negative integer and&#36;N_0=0&#36;
<strong>Continuous parameters Dispersed</strong></p>
<h5>Normal process</h5>
<p>It's also called the Gauss Process.
Set&#36;X= {X_{t},t\in T}&#36;It's a random process of real value, if any.&#36;n≥1&#36;and
&#36;t_1,t_2,…,t_n∈T&#36;, &#36;n&#36;Wy random variable (%2)&#36;X_{t_1}, X_{t_2}, …, X_{t_n})&#36;
Obey.&#36;n&#36;Wyss distribution, called X is normal (Gose) process
Parameter Set&#36;T&#36;The situation determines whether it's a continuous random process, but the state must be continuous.</p>
<h5>Prove a random process is a normal process.</h5>
<p>Set&#36;Z_t=X+Yt&#36;,&#36;-\infty&lt;t&lt;+\infty&#36;,其中随机变量&#36;X&#36;,&#36;Y&#36;相互独立，且都服从&#36;N(0,\sigma^2)&#36;分布。证明随机过程&#36;Z={Z_t,-\infty&lt;t&lt;+infty}&#36; is a normal process
It's easy to know.
&#36;&#36;(X,Y)\sim N_{}(\mu,\sum)&#36;&#36;
is a two-dimensional normal vector (because of independence, two normal variables make a normal vector)
Yes.
&#36;&#36;\left.C {2, n}=\left[\begin{array}{cccc}1&amp;1&amp;1&amp;\cdots&amp;1\t_1&amp;t_2&amp;t_3&amp;\cdots&amp;t_n\end{array}\right.\right].&#36;&#36;
有
&#36;&#36;♪ ♪ I'm not gonna let you go ♪
So we know that the new high-dimensional vector is also a normal vector, which proves that the process is a normal one.</p>
<p>The questions in this article can also be addressed<a href="/en/blog/2024/10/17/financial-stochastic-analysis-notes/">Financial random analysis: financial derivatives, fork tree pricing and the theory of arbitrage</a>、<a href="/en/blog/2025/09/11/markov-chain-notes/">Markov Chain: Transfer probability, status classification and smooth distribution</a>How the concept of a relatively close read together is developed in different contexts.</p>
<p>Set Random Variables&#36;R&#36;and&#36;\Theta&#36;I'm independent of one another.&#36;R&#36;Following Riley's distribution, the density is...
&#36;f(r)=\begin{cases}\frac{r}{\sigma^2},&amp;\quad r\ge0\0,&amp;\quad r&lt;0\end{cases}&#36;&#36;
&#36;\Theta&#36;服从&#36;[0,2\pi]&#36;上的均匀分布
定义
&#36;&#36;X t=R\cos(\Theta+at),\quadt\inR,a\text{consistent} &#36;
Proves he's a normal process.
<strong>We can transform it into the same problem as before.</strong>
&#36;&#36;X_{t_{&#125;&#125;=R\cos(\theta+at)=R\cos\theta\cos a t_{}-R\sin\theta\sin\alpha t_{}&#36;&#36;
We can easily see the vectors we want to prove.
&#36;&#36;(R\cos\theta,R\sin\theta)\binom{\cos at_1,\cdots,\cos at_n}{\sin at_1,\cdots,\sin at_n}&#36;&#36;
Just prove that this is a normal vector in front.
He's a function of two random variables, and the probability theory gives us a way to prove his density function.
Check it out, it's the density function of the normal vector.</p>
<h3>Further classification of random processes</h3>
<h4>Incremental nature</h4>
<h5>Incremental process</h5>
<p>Set&#36;X={X_t,t∈T}&#36;is a real-value random process if for any &#36;t      &lt;}t_{2}\leq t_{3}&lt;t {4}\mathbb{T}&#36; both
&#36; \\mathrm{E}\left[\textbf{X}<em>{t_2}-\textbf{X}</em>{t_1}\right)\left(\textbf{X}<em>{t_4}-\textbf{X}</em>♪ The world is so bright ♪
Call it a positive increment process
Here's the amount of random vectors considered to be internal.</p>
<h5>Independent incremental process</h5>
<p>Set&#36;X={X_t,t∈T}&#36;is a real-value random process if for any &#36;t      &lt;}t_{2}\leq t_{3}&lt;...&lt;t n}in\matbb{T}&#36; both
&#36;&#36;X_{t_2}-X_{t_1},X_{t_3}-X_{t_2},\cdots,X_{t_n}-X_{t_{n-1&#125;&#125;&#36;&#36;
It's a random variable that is separate from each other.&#36;X&#36;It's an independent incremental process.</p>
<h5>Steady incremental process</h5>
<p>Set&#36;X={X_t,t∈T}&#36;It's a real-value random process if for any &#36;s&lt;t\in T&#36;
&#36;&#36;X_{t}-X_{s}&#36;&#36;
Only depend on&#36;t-s&#36; Name&#36;X&#36;It's a smooth incremental process.
We'll be back with a random process called smooth, and a smooth incremental process.
<strong>The independent incremental process and the smooth incremental process are important points of view, and they tend to come back together.</strong></p>
<h4>Sample orbital continuity</h4>
<h5>Definitions</h5>
<p>Set &#36;\xi(\omega)&#36; ,&#36;\eta{(\omega)}&#36; are two random variables defined in the same probability space. Define Random Process&#36;\mathrm{X= }{ X_t{: }\quad t\geq 0}&#36;Is:
&#36;&#36;
X_t=\xi(\omega)+\operatorname{t}\eta(\omega)
&#36;&#36;
And X is a continuous orbital random process.
We're often only introducing a few of the core random cars. Pass.</p>
<h5>Continuous orbital Random Process</h5>
<p>The Standard Brown movement is a continuous orbital random process.
If the random process W is&#36;&#123;\mathbf{W}_{t},\mathbf{t}\geq0}&#36;Satisfy:
&#36;(1)\quad W_0=0&#36;
&#36;(2)\quad W={W_t,t\geq0}&#36; It's a smooth, independent incremental process.
(3) For any of the flaileq s&lt;t&#36;,有&#36;W t-W s\sim N(0,(t-s))
The random process W is standard Brown movement.
He's a continuous orbit random process, and we can see that in his sample curve.</p>
<h5>Random process for jumping orbit</h5>
<p>Call random process N=&#36;&#123;N_t,t\geq0}&#36;is the parameter as&#36;\lambda&#36; The porcelain process, if it meets the following three conditions:
&#36;\begin{pmatrix}1\end{pmatrix}\quad N_0=0&#36;
(2) For any of the freckles&lt;t&#36;,增量&#36;N_t&#36; -&#36;N_s&#36;服从参数为&#36;Porcelain distribution of \lambda(t-s)&#36;, i.e.
&#36;&#36;
P(N_t:-N_s=k)=\frac{\left(\lambda(t-s)\right)^ke^{-\lambda(t-s)&#125;&#125;{k!},k=0,1,2,\cdots
&#36;&#36;
(3) Arbitrary&#36;n\geq2&#36;And freckt 0&lt;t_1&lt;\cdots&lt;t_n&lt;\cdots, n&#36; increments
&#36;&#36;
N_{t_n}-N_{t_{n-1&#125;&#125;,\cdots,N_{t_2}-N_{t_1},N_{t_1}-N_{t_0}
&#36;&#36;
It's a random variable that's separate from each other.
He's a random process of orbits that are not continuous.
<strong>These two most basic random processes share common features, with an initial value of 0, which is a smooth incremental process and an independent incremental process, with any margin distribution subject to a specific distribution.</strong></p>
<h4>Extension of random processes</h4>
<h5>Multi-dimensional Random Process</h5>
<p>Set&#36;X_{t}~t\in T ~Y_{t} ~t\in T&#36;Is a definition in the same probability space (&#36;\Omega&#36;F, P) two random processes.&#36;&#123; X_t, Y_t, \mathrm{t\in T&#125;&#125;&#36;It's a 2-dimensional random process.</p>
<h5>Re-random Process</h5>
<p>Set&#36;&#123; \mathrm{X_t, t\in T&#125;&#125;&#36; and &#36;&#123; \mathrm{Y_t, t\in T&#125;&#125;&#36; It's defined as the same probability space. &#36;(\Omega,\mathbb{F},\mathbb{P})&#36;Two random processes on the top.
&#36;&#36;\color{}{\mathrm{Z_t\color{}{=}X_t\color{}{+}jY_t}\quad\color{}{t\in T&#125;&#125;&#36;&#36;
Name&#36;&#123; \mathbb{Z} _t, \mathrm{t\in T&#125;&#125;&#36;It's a random process.</p>
<h3>Limited-Destroyal Functions</h3>
<h4>Definitions</h4>
<p>We've already introduced the concept of distributional functions in probabilities, and now we have to extend it to random processes, so to simplify the problem, we're only here to study discrete and limited parameters. Group&#36;T&#36;
Set &#36;X={X_t,t\in T}&#36;is defined in probability space (&#36;\Omega,F,P)&#36;Random process to take up real value.&#36;n≥1&#36;and arbitrary &#36;t_1,t_2,\cdots,t_n\in T&#36; and actual &#36;x_1,x_2,\cdots,x_n\in R&#36;, called n-dimensional random variable &#36;(X_{t_1},X_{t_2},\cdots,X_{t_n})&#36; Joint Distribution Functions
&#36;&#36;
F_{t_1,\cdots,t_n}\left(x_1,\cdots,x_n\right.)=P(X_{t_l}\leq x_1,\cdots,X_{t_n}\leq x_n),
&#36;&#36;
N-Deasing function for random process X
Remember as a whole all limited-dimensional distribution functions of random process X
&#36;&#36;
\mathbf{F}={F_{t_1,\cdots,t_n}(x_1,\cdots,x_n)::\forall n\in\mathbf{N},:t_1,t_2,\cdots,t_n\in T\text{,}x_1,x_2,\cdots,x_n\in R}
&#36;&#36;
Call F the function set as the n-dimensional distribution of random process X
The family is formed because of time.&#36;t&#36;And the uncertainty is more because of the company.&#36;n&#36;I can't even get a value.
Because even&#36;t&#36;We're not sure we can also run as constants, but...&#36;n&#36;And the change in the function shapes the whole function, so we'll see it later.&#36;n&#36;Calculates the distribution function of the dimension</p>
<h4>Explanation</h4>
<p>Yes (&#36;1,2,\cdots,n&#36;) One by one (&#36;k_l,\cdots,k_n&#36;) There is.
&#36;&#36;
F_{t_1,\cdots,t_n}\left(x_1,\cdots,x_n\right)=F_{t_{k_1},\cdots,t_{k_n&#125;&#125;\left(x_{k_1},\cdots,x_{k_n}\right)
&#36;&#36;
If &#36;m&lt;I'm not sure I'm gonna be able to do that.
&#36;&#36;F_{t_1,\cdots,t_m}(x_1,\cdots,x_m)=F_{t_1,\cdots,t_m,t_{m+1},\cdots,t_n}(x_1,\cdots,x_m,+\infty,\cdots,+\infty)&#36;&#36;</p>
<h4>Examples</h4>
<h5>One.</h5>
<p>&#36;X{=}{X_t=Vcos\omega t,t\in R}&#36;of which &#36;\omega&#36; For constant, random variable V obeys&#36;[0,1]&#36;Equal distribution on the scale.
&#36;&#36;
t=\frac{3\pi}{4\omega}\text{和}t=\frac\pi{2\omega}
&#36;&#36;
 , the one-dimensional distribution function of random process X
When?&#36;t=\frac{3\pi}{4\omega}&#36;  &#36;X_{<em>♪ The world is so full of shit ♪
We need to study the distribution of the known distributions, and we have also described the probability theory, so...
&#36;X</em>{t}&#36;的密度函数为&#36;&#36;f(x)=\begin{cases}\sqrt{2},&amp;-\frac{\sqrt{2&#125;&#125;{2}\leq x\leq0\0,&amp;\text{Other}\end{cases}&#36;&#36;一维分布函数就是积分计算问题了
&#36;&#36;&#123;F}<em>{\frac{3\pi}{4\omega&#125;&#125;(x)=\int</em>{-infty}xf x X  &#123;&#123;\frac{3\pi&#125;&#125;(t)dt&#36;
The results don't have to go on.
&#36;t=\frac{\pi}{2\omega},X_{\frac{\pi}{2\omega&#125;&#125;=V\cos\omega\times\frac{\pi}{2\omega}=0&#36;
Do density functions always have 0 in the defined domain and the distribution functions in the natural dimension always have 0?</p>
<h5>Two.</h5>
<p>Isolated can do it in the same way.
Set Random Process&#36;X={X_t=Acost,t\geq0}&#36;, where random variable A has a distribution law
&#36;&#36;
P(A=i)=\frac13,\quad i=1,2,3.
&#36;&#36;
(1) One-dimensional distribution function for random process X&#36;&#123;F_{\frac\pi2&#125;&#125;(x)&#36;
(2) A 2-D distribution function for random process X&#36;F_{0,\frac\pi3}(x_1,x_2)&#36;
Calculate using the formula in front
&#36;&#36;X_{\frac\pi4}=A\cos\frac\pi4=\frac{\sqrt2}2A,&#36;&#36;
And because it's discrete, the distribution is better after the function.
&#36; \\mathrm{F} begin{cases}0,&amp;\quad x&lt;\frac{\sqrt{2&#125;&#125;{2}\\frac{1}{3},&amp;\quad\frac{\sqrt{2&#125;&#125;{2}\leq x&lt;\sqrt{2}\\frac{2}{3},&amp;\quad\sqrt{2}\leq x&lt;\frac{3}{2}\sqrt{2}\1,&amp;\facing \sq\s}
The calculation here does not need to use formula A, which is discrete, in a sequential manner.&#36;A&#36;The value will be taken; only the defined domain will change; his value will be stable; or we will use &#36;\frac }sqrt2}2A directly.&lt;x&#36; 然后变形为&#36;A&lt;\sqrt2x&#36;  类比标准的分布函数定义&#36;A&lt;X&#36;, that's all. </p>
<p>Likewise, let's look at the two-dimensional situation, and stick to the definition.
&#36;&#36;00begin{aligned}F 0,\fra\pi3}(x 1,x 2)&amp;=P(X_0\leq x_1,X_{\left.\frac\pi3\leq x_2\right)}\&amp;=P(A\leq x_1,\frac A2\leq x_2)\&amp;=P(A\leq x_1,A\leq2x_2)\end{aligned}&#36;&#36;
我们能用两者独立然后单独计算然后乘起来吗？
不能 明显&#36;A&#36;不可能和自己独立
分类讨论
&#36;&#36;=\begin{cases}P(A\le x_1)\quad x_1\le2x_2\P(A\le2x_2)\quad x_1&gt;2x_2\end{cases}&#36;&#36;
这时候明显左边的分布列是已知的 非常自然的计算就好了
分别给出分布函数结果
&#36;&#36;\begin{aligned}P(A\leq x_1)=\begin{cases}0,&amp;\quad x_1&lt;1\\frac13,&amp;\quad1\leq x_1&lt;2\\frac23,&amp;\quad2\leq x_1&lt;3\1,&amp;\quad x_1\geq3&amp;\end{cases}\quad P(A\leq2x_2)=\begin{cases}0,&amp;\quad2x_2&lt;1\\frac13,&amp;\quad1\leq2x_2&lt;2\\frac23,&amp;\quad2\leq2x_2&lt;3\1,&amp;\quad2x_2\geq3&amp;\end{cases}\end{aligned}&#36;&#36;</p>
<h3>A limited-dimensional feature function</h3>
<h4>Definitions</h4>
<p>We've already introduced the concept of characteristic functions in probabilistic theory. Now we have to extend it to random processes. Medium
<strong>It is recommended to review the probabilistic feature function part and to understand as fully as possible, and to review the characteristics of the distributions</strong>
Here's the random process.&#36;n&#36;Definition of a V-Female Function
&#36;&#36;
\varphi_{t_1,t_2,...,t_n}(u_1,u_2,...,u_n)\quad=\operatorname{E}[e^{j(u_1X_{t_1}+\cdots+u_nX_{t_n})}]\quad\forall ^{}u_{1},u_{2},...,u_{n}\in R
&#36;&#36;
The uncertainty of the dimensions of the study is the limited-dimensional distribution function.
&#36;&#36;
\Phi={\varphi_{t_1,t_2,...,t_n}(u_1,u_2,...,u_n),t_{i}\in {T},u_{i}\in {R},i=1,2,\cdots,n}
&#36;&#36;</p>
<h4>Examples</h4>
<p><strong>We'll study this example and try to understand its ideas.</strong>
If the random process W=&#36; {\mathbf{W}<em>Other Organiser
&#36;(1)\quad W_0=0&#36;
&#36;(2)\quad W={W_t,t\geq0}&#36; It's a smooth, independent incremental process.
(3) For any of the flaileq s&lt;t&#36;,有&#36;W t-W s\sim N(0,(t-s))
The random process W is standard Brown movement.
We'll calculate his limited-dimensional feature function.
He knows by definition that his signature function is
&#36; \varphi</em>{t_1,\cdots,t_n}(u_1,\cdots,u_n)=\mathrm{E}[e^{j(u_1W_{t_1}+\cdots+u_nW_{t_n})}]&#36;&#36;
令
&#36;&#36;\begin{aligned}Y_1=W_{t_1},Y_2=W_{t_2}-W_{t_1},\cdots,Y_n=W_{t_n}-W_{t_{n-1&#125;&#125;\end{aligned}&#36;&#36;
明显的 我们的&#36;Y&#36;是过程增量 容易知道增量具有独立性 也就是所有的&#36;Y&#36;独立
反解&#36;W_t&#36;带入特征函数的计算
&#36;&#36;\varphi_{t_1,\cdots,t_n}(u_1,\cdots,u_n)=\mathrm{E}[e^{j[u_1Y_1+u_2(Y_1+Y_2)+\cdots+u_n(Y_1+\cdots+Y_n)]}]&#36;&#36;
整理成每一个增量的形式
&#36;&#36;\operatorname{E}[e^{j[(u_{1}+u_{2}+\cdots+u_{n})Y_{1}+(u_{2}+\cdots+u_{n})Y_{2}+\cdots+u_{n}Y_{n}]}]&#36;&#36;
根据增量的独立性有
&#36;&#36;=\operatorname{E}[e^{j[(u_1+u_2+\cdots+u_n)Y_1}]\operatorname{E}[e^{(u_2+\cdots+u_n)Y_2}]\cdotp\cdotp\cdotp\operatorname{E}[e^{u_nY_n}]&#36;&#36;
能看出 每一个均值部分都是一个特征函数
&#36;&#36;=\varphi_{Y_1}(u_1+u_2+\cdotp\cdotp\cdotp+u_n)\varphi_{Y_2}(u_2+u_3+\cdotp\cdotp\cdotp+u_n)\cdotp\cdotp\cdotp\cdotp\varphi_{Y_n}(u_n)&#36;&#36;
又因为我们知道 事实上每个增量的分布都是正态分布 他的特征函数是容易计算的
&#36;&#36;\begin{aligned}&amp;\varphi_{Y_1}(u_1+\cdots+u_n)=e^{-\frac12(u_1+\cdots+u_n)^2t_1}\\&amp;\varphi_{Y_k}(u_k+\cdots+u_n)=e^{-\frac12(u_k+\cdots+u_n)^2(t_k-t_{k-1})}\end{aligned}&#36;&#36;
把它带回就是我们想计算的特征函数
&#36;&#36;e^{-\frac12(u_1+\cdots+u_n)^2t_1}\cdot e^{-\frac12(u_2+\cdots+u_n)^2(t_2-t_1)}\cdot\cdots\cdot e^{-\frac12(u_n)^2(t_n-t_{n-1})}&#36;&#36;
<strong>Based on this example, we can add a general conclusion: the limited dimensions distribution function of the independent incremental process is its one-dimensional distribution function (see also para.&#36;Y_1&#36;) and incremental distribution functions ( )&#36;Y_n,n\ne1&#36;)</strong>
<strong>We can calculate the characterizations in the same way as the above, if we know it's an independent incremental process.</strong></p>
<h3>Numeric Characteristics</h3>
<h4>Digital features of random processes</h4>
<h5>Mean Functions</h5>
<p>Set &#36;X={X_t,t\in\mathbb{T&#125;&#125;&#36; It's a random process of real value, if any.&#36;t{\in }&#36;T, yes.
&#36;&#36;\operatorname{E}[X_t]\text{存在}&#36;&#36;
Name &#36;&#123;E}[X_t]&#36;is the average of random process X, as&#36;m_x(t).&#36; </p>
<h5>Difference Functions</h5>
<p>Set &#36;X={X_t,t\in\mathbb{T&#125;&#125;&#36; It's a random process of real value, if any.&#36;t{\in }&#36;T, yes.
&#36;&#36;\mathbb{E}[X_t-m_X(t)]^2\text{ 存在}&#36;&#36;
Name &#36;&#123;E}[X_t]&#36;is the variance function for random process X, as&#36;D_x(t).&#36; </p>
<h5>Accompanimental difference function</h5>
<p>Set &#36;X={X_t,t\in\mathbb{T&#125;&#125;&#36; It's a random process of real value, if any.&#36;t,s{\in }&#36;T, yes.
&#36;&#36;Co\nu(X_s,X_t)=\operatorname{E}\left[(X_s-m_X(s))(X_t-m_X(t))\right]\text{存在}&#36;&#36;
Called the random process X's synergetic function.&#36;C_X(s,t).&#36;</p>
<h5>Related Functions</h5>
<p>Set &#36;X={X_t,t\in\mathbb{T&#125;&#125;&#36; It's a random process of real value, if any.&#36;t,s{\in }&#36;T, yes.
&#36;&#36;
\operatorname{E}[X_sX_t]\text{存在}
&#36;&#36;
is called a (self) related function of a random process X.&#36;R_X(s,t).&#36;</p>
<h5>Equivalent Functions</h5>
<p>Set &#36;X={X_t,t\in\mathbb{T&#125;&#125;&#36; It's a random process of real value, if any.&#36;t,s{\in }&#36;T, yes.
&#36;&#36;\mathbb{E}[X_t]^2\text{ 存在}&#36;&#36;
Called the average square function of random process X.&#36;\Phi_X(t).&#36;</p>
<h5>Relationship of random process numerical characteristics</h5>
<p>&#36;&#36;\begin{aligned}
&amp;C(s,t)=R_X(s,t)-m_X(s)m_X(t) \
&amp;D_{X}(t)=C_{X}(t,t) \
&amp;\Phi_{X}(t)=R_{X}(t,t)
\end{aligned}&#36;&#36;
<strong>Our calculation habits are to calculate the mean function.&#36;m(x)&#36;, and associated functions&#36;R(x,t)&#36; Export the altruistic difference function from this basis and then the variance function</strong></p>
<h4>Digital features of two random processes</h4>
<p>Set &#36; & &#36; {\<em>t,\mathcal{Y}<em>t,t\in\mathcal{T&#125;&#125;&#36;为二维随机过程，对任意&#36;I'm sorry, but I'm sorry.
If &#36;E[X]</em>{s}Y</em>Existence
It's called the two-dimensional random process's interconnective function, remember? &#36;R_{X Y}(s,t).&#36;
If &#36;Co\nu(X_s,Y_t)={E}[(X_s-m_x(s))(Y_t-m_y(t))]&#36; Existence
The two-dimensional random process, called the inter-coordinate difference, remembers&#36;C_{XY}(s,t).&#36;
The intercoordination differential functions define the relevance of two processes when they are 0 and the two random processes are not relevant</p>
<h4>Digital features of the duplicate random process</h4>
<p>Average Difference Function definition remains unchanged
Equalise function definition amended to &#36;\Phi_Z(t)=\mathbb{E}\left|Z_t\right|^2&#36;Add a layer of compound conversion
The respective functions and the agreed differences are amended to &#36;R z}(s, t)=\mathrm{E}[\bar{Z}<em>{s}Z</em>{t}]&#36;&#36;&#36;&#36;\begin{aligned}C_Z(s,t)&amp;=\mathbb{E}[(\overline{Z_s-m_Z(s)})(\mathbb{Z}<em>t-m_Z(t))]\end{aligned}&#36;&#36;因此我们修正原本的性质为
&#36;&#36;\begin{aligned}
&amp;m</em>{Z}(t)=m_{X}(t)+jm_{Y}(t),t\in T \
&amp;D_{Z}(t)=D_{X}(t)+D_{Y}(t),t\in T \
&amp;C_{Z}(s,t)=R_{Z}(s,t)-\overline{m_{Z}(s)}m_{Z}(t),t\in T
\end{aligned}&#36;&#36;</p>
<h2>Brown Movement</h2>
<p>In fact, random processes are a very complex thing, and we can't really characterize much of it, so we'll be able to describe four more simple and sufficiently random processes in the next three chapters, and we'll start with the Brown movement.</p>
<h3>Standard Brown Movement</h3>
<h4>Definitions</h4>
<p>If the random process W is&#36;&#123;\mathbf{W}_{t},\mathbf{t}\geq0}&#36;Satisfy:
&#36;(1)\quad W_0=0&#36;
&#36;(2)\quad W={W_t,t\geq0}&#36; It's a smooth, independent incremental process.
(3) For any of the flaileq s&lt;t&#36;,有&#36;W t-W s\sim N(0,(t-s))
The random process W is standard Brown movement.
Remove the first 0-point, which is called Brown Movement.</p>
<h4>A limited-dimensional distribution function</h4>
<p>We're counting the character functions of the standard Brown movement in the section before us.
It's a collection of normal distribution feature functions.
&#36;&#36;\varphi_{\iota_1,\cdots,\ell_n}\left(u_1,\cdots,u_n\right.)=\prod_{k=1}^ne^{-\frac12(u_k+\cdots+u_n)^2(t_k-t_{k-1})}=e^{-\frac12\sum_{k=1}^n(u_k+\cdots+u_n)^2(t_k-t_{k-1})}&#36;&#36;
We can use a completely similar approach to the distribution function of standard Brown movement.</p>
<h5>One-dimensional.</h5>
<p>I can see it.&#36;W_{t}=W_{t}-W_{0}\sim N(0,t)&#36;
His distribution function is
&#36;&#36;F_{t}(x)=\frac{1}{\sqrt{2\pi t_{1&#125;&#125;}\int_{-\infty}^{x}e^{-\frac{x^{2&#125;&#125;{2t_{1&#125;&#125;}\mathrm{d}x,x\in\mathbb{R}&#36;&#36;
It's the distribution function of the normal distribution.</p>
<h5>Two-dimensional.</h5>
<p>We know by definition.
&#36;&#36;00\
&amp;F_{t_{1},t_{2&#125;&#125;(x_{1},x_{2})=P(W_{t_{1&#125;&#125;\leq x_{1},W_{t_{2&#125;&#125;\leq x_{2}) \
&amp;=P(W_{t_1}\leq x_1,W_{t_1}+(W_{t_2}-W_{t_1})\leq x_2),\&amp;\text\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\\t\t\t
&amp;=P(\xi\leq x_{1},\xi+\eta\leq x_{2})
\end{aligned}&#36;&#36;
还是使用前面的思路构造增量函数
我们知道两个增量是独立并且服从正态分布
&#36;&#36;\begin{aligned}&amp;\eta=W_{t_2}-W_{t_1}\sim N(0,t_2-t_1)\&amp;\xi=W_{t_1}-W_0\sim N(0,t_1),\end{aligned}&#36;&#36;
但是此时我们的联合分布并不是独立的 还是需要使用基础的定义来进行求解 研究联合分布的密度函数
&#36;&#36;\begin{aligned}
&amp;=\int_{-\infty}^{x_1}P(\eta\leq x_2-\xi|\xi\in(y,y+\mathrm{d}y)P(\xi\in(y,y+\mathrm{d}y)) \
&amp;=\int_{-\infty}^{x_{1&#125;&#125;\int_{-\infty}^{x_{2}-y}[f_{\eta}(z)\mathrm{d}z]g_{\xi}(y)\mathrm{d}y
\end{aligned}&#36;&#36;</p>
<h5>&#36;n&#36;V</h5>
<p>We chose to solve it by using a normal process.
It's easy to know that incremental is a separate normal variable, which is...
&#36;&#36;(\begin{array}{c}W_{t_1},W_{t_2}-W_{t_1},\cdots,W_{t_n}-W_{t_n-1}\\end{array})&#36;&#36;
It's a normal vector.
&#36;&#36;(W_{t_{1&#125;&#125;,W_{t_{2&#125;&#125;,\cdots,W_{t_{n&#125;&#125;)&#36;&#36;
It's some kind of combination of normal vectors, so he's a normal vector.
So we can see that Standard Brown is a normal process; a direct definition can count him.&#36;n&#36;Weight density function<br>&#36;&#36;\frac1{(2\pi)^{\frac n2}\Bigg(\prod_{k=1}^n(t_k-t_{k-1})\Bigg)^{\frac12&#125;&#125;e^{-\frac12\sum_{k=1}^n\frac{(w_k-w_{k-1})^2}{(t_k-t_{k-1})&#125;&#125;&#36;&#36;
It's possible to do this by determining the high-dimensional distribution by imitating the two-dimensional calculation.</p>
<h4>Numeric Characteristics</h4>
<h5>Mean</h5>
<p>Because Standard Brown is a normal process and it's easy to know.
&#36;&#36;W_{t}=W_{t}-W_{0}\sim N(0,t)&#36;&#36;
So there is.
&#36;&#36;m_{_{W&#125;&#125;(t)=0&#36;&#36;</p>
<h5>Difference</h5>
<p>I'm sorry.
&#36;&#36;D_W(t)=t&#36;&#36;</p>
<h5>Related coefficient</h5>
<p>Or is it by definition?
&#36;&#36;00\
R W}(s,t)&amp; =\operatorname{E}[W_{s}W_{t}]  \
&amp;=\mathrm{E}[(W_{s}-W_{0})(W_{t}-W_{s}+W_{s})] \
&amp;=\mathrm{E}[(W_s-W_0)(W_t-W_s)]+\mathrm{E}[W_s]^2 \
&amp;=0+\operatorname{E}[W_s]^2 \
&amp;=D[W_{s}]+\left(\operatorname{E}[W_{s}]\right)^{2} \
&amp;=s
\end{aligned}&#36;&#36;
因此
&#36;&#36;&#123;R}_{\mathrm{w&#125;&#125;(s,t)=\min(s,t)&#36;&#36;</p>
<h5>Agreement</h5>
<p>Study using assistive formulae
&#36;&#36;C_{W}(s,t)=R_{W}(s,t)-m_{W}(s)m_{W}(\mathrm{t})=\min(s,t)&#36;&#36;</p>
<h3>Nature of the Brown Movement</h3>
<p>When?&#36;\mathbf{W}={\mathbf{W}_{t},t\geq0}&#36; Standard Brown movement.</p>
<h4>Symmetry</h4>
<p>&#36;&#36;\mathbf{-W}\mathbf{={-W_{t},t\geq0&#125;&#125;&#36;&#36;
It's standard Brown.</p>
<h4>Self-relationship</h4>
<p>For any &#36;a&gt;- Yes, I do.
&#36;&#36;\mathbf{W_{at&#125;&#125;\doteq\mathbf{a^{1/2&#125;&#125;\mathbf{W_{t&#125;&#125;&#36;&#36;
It's standard Brown.</p>
<h4>Time reversal</h4>
<p>For Fixed &#36;T&gt;- Yes, I do.
&#36;&#36;\mathbf{B_{t&#125;&#125;=\mathbf{W_{T&#125;&#125;-\mathbf{W_{T-t&#125;&#125;\quad\mathbf{0\leq t\leq T}&#36;&#36;
It's standard Brown.</p>
<h4>Sample orbital properties</h4>
<p>The sample track for Standard Brown's movement is...<strong>Continuous</strong>And yes.<strong>It's so hard to find.</strong>Yes.</p>
<h2>Porcelain Process</h2>
<p>We should have been talking about the randomity of jumping, just as the previous chapter was about the continuous randomity, but in the last chapter we focused on the standard Brown movement of continuous randomity, we chose to study the porcelain process, thereby reducing the difficulty.</p>
<h3>Counting process</h3>
<p>If&#36;N_t&#36;Means to the moment&#36;t&#36;The number of random incidents that have occurred so far is still high.
Call it a random process&#36;&#123;N_t,t≥0}&#36;For Count Process
Obviously, the counting process is very extensive in the real world, like the number of vehicles passing through, the transmission of data packages, etc., which is part of the counting process.
It's easy to know that the counting process should have the following characteristics.</p>
<ul>
<li>&#36;\forall t,N_t\geq0\text{,}N_0=0&#36;</li>
<li>&#36;N_{t}\text{取非负整数}&#36;</li>
<li>&#36;\forall0\leq s&lt;t,N_t\geq N_s&#36;</li>
<li>&#36;\forall0\leq s&lt;t,N t-N s&#36; for total number of times occurring within time period
If there is one... &#36;N={N_{t},t\geq0}&#36; It's a counting process.
And there is.
&#36;&#36;\mathrm{T_n=inf}{t:N_t{=}\boldsymbol{n&#125;&#125;&#36;&#36;
Name of random sequence &#36;T_1,T_2,\cdots,T_n\cdots&#36;For the time series of arrival of the count process
Yes.
&#36;&#36;\tau_n=T_n-T_{n-1},n=1,2,...&#36;&#36;
The interval sequence called the time of arrival of the count process
Obviously.&#36;T_{n}=\sum_{k=1}^{n}\tau_{k},n=1,2,\cdots&#36;
Obviously, any count process corresponds to three random sets of variables. &#36;N_t,T_n,\tau_n&#36;
They're all a random process, it's not easy to study their distribution, and then we'll have a chance to think about it later.</li>
</ul>
<h3>Basic definition and nature of the porcelain process</h3>
<h4>Definition of the Percein Process</h4>
<p>In fact, the porcelain process is a special type of counting process.
If the counting process&#36;&#123;N_t,t≥0}&#36; Satisfied</p>
<ul>
<li>&#36;\quad N_{0}=0&#36; </li>
<li>- Any way.&#36;n\geq2&#36;And that's a good one.&lt;t_1&lt;\cdots&lt;t_n&#36;, 增量&#36;t n t n t n n n ,\cdts, n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n n   n    n n n n</li>
<li>For any of the flaileq s&lt;t&#36;,增量&#36;N_t-N_s&#36;服从参数为&#36;\lambda(t-s)&#36;的泊松分布：&#36;P(N t-N s=k)=\frac{ke^-\lambda(t-s)}k!}, k=0,1,2,\cdots&#36;
Call this counting process an argument&#36;\lambda&#36;♪ And the porridge process ♪
Or the definition we mentioned earlier?</li>
</ul>
<h4>One-dimensional and multi-dimensional distribution of the porcelain process</h4>
<p>The one-dimensional distribution is very well studied, and it can be obtained by the nature of the incrementality.
&#36;&#36;00begin{aligned}\mathrm{P} (N t=k)&amp;=\mathrm{P}(N_t-N_0=k)\&amp;== sync, corrected by elderman == @elder man
This is the density function, of course, cumulatively.&#36;F(x)&#36;It's okay.</p>
<h4>Digital features of the porcelain process</h4>
<h5>Mean Functions</h5>
<p>The one-dimensional distribution that was given in the front of the study is
&#36;&#36;m_N(t){=}\lambda t&#36;&#36;</p>
<h5>Difference Functions</h5>
<p>The difference should be the same as the average.
&#36;&#36;D_{N}(t){=}\lambda t&#36;&#36;</p>
<h5>Related Functions</h5>
<p>I'm still going to have to deformate on the basis of incremental independence.
&#36;&#36;R_N(\mathrm{s,t}){=}\mathrm{E}[N_sN_t]&#36;&#36;
Incrementally deformed&#36;&#36;=\mathrm{E}[(N_s-N_0)(N_t-N_s+N_s)]&#36;&#36;Simplified split
&#36;&#36;=\mathrm{E}[(N_s-N_0)(N_t-N_s)]+\mathrm{E}[N_s^2]&#36;&#36;
Because the increment is independent, the second-order rectangular can be the sum of the difference and the first-step rectangular square.
&#36;=\pepratorname{E}[N s}\pectorname{E}[N t}-N s}+\pepratorname{D}<em>{N}(s)+(m</em>{N}(s))^{2}&#36;&#36;
带入化简有
&#36;&#36;\begin{aligned}&amp;=\lambda^2st+\lambda s\&amp;=\lambda^2st+\lambda\min(s,t)\end{aligned}&#36;&#36;</p>
<h4>Sample tracks of the porcelain process</h4>
<p>The sample tracks of the porcelain process are jumping, right-right continuous.
We still don't have proof here.
<em>It's a confirmation of the bellow distribution.&#36;\lambda&#36; The whole process was determined, only one more parameter than the Weiner process.</em> </p>
<h4>Theorem of the decision.</h4>
<p>We know. <strong>Porcelain processes are a special kind of counting process.</strong> Now, let's look at how, apart from definition, a counting process is a porcelain process.
<strong>Theorem</strong>
If the counting process&#36;N={N_t,\quad t\geq0}&#36;. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .&#36;&#123;\tau_n,n=1,2,\cdots}&#36;It's a standalone, conciliating parameter of &#36; \lambda&gt;Index distribution of &#36;0.00
The count must be the parameter.&#36;\lambda&#36; ♪ And the porridge process ♪
<strong>Auxiliary Theorem</strong>
If the random variable sequence&#36;&#123;\tau_n,n=1,2,\cdots}&#36;Independence and obedience parameters are &#36;\lambda&gt;Index distribution of &#36;0.00
I'm sorry.
T-shirts<em>n=\sum</em>The hyena is a very small area of the world.
I'm sorry.
T&#36;<em>{n}&#36;服从参数为(n,&#36;\lambda&#36;)的伽玛分布&#36;\\Gamma(n,\lambda) &#36;, density function is
&#36;f</em>{T_{n&#125;&#125;(x)=\begin{cases}\dfrac{\lambda^{n&#125;&#125;{(n-1)!}x^{n-1}e^{-\lambda x},&amp;\quad x\geq0\0,&amp;\quad x&lt;I'm sorry, I'm sorry.
We use this aided theorem to calculate the arrival time series from the arrival interval sequence to include the arrival time sequence in the hand study.&#36;N_t&#36;
<strong>Theorem</strong>
This is the inverse theory of the judgment.
Set&#36;&#123;N}={N_t,t{\geqslant}0}&#36;is the parameter as&#36;\lambda&#36;, and N arrive time interval sequence &#36;\tau_1,\tau_2,\cdots,\tau_n\cdots&#36;Independence and obedience parameters&#36;\lambda&#36;Index distribution</p>
<h4>Supplement</h4>
<p>The sum of the two separate Porpine processes is still the Porcerines process</p>
<h3>Definition of Porcelain process</h3>
<h4>Definition of Equivalence</h4>
<p><strong>Definitions</strong>
If the conditions are met</p>
<ul>
<li>&#36;N_0=0&#36; </li>
<li>&#36;N&#36;It's an independent incremental process and a peaceful incremental process.</li>
<li>&#36;&#123;P}{N_{t+h}-N_{t}=0}=1-\lambda h+\circ(h)&#36;</li>
<li>&#36;&#123;P}{N_{t+h}-N_t=1}=\lambda h+\circ(h)&#36;
Counting process&#36;N={Nt,t≥0}&#36;is the parameter as&#36;\lambda&#36; ♪ And the porridge process ♪
<strong>Theorem</strong>
Set&#36;N={Nt,t≥0}&#36;is the parameter as&#36;\lambda&#36; There must be a porcelain process.
&#36;begin{aligned}\quad&amp;\mathbf{P}{N_{t+h}-N_t=0}=1-\lambda h+\circ(h)\2)\quad&amp;\matbf{t=t=t=mbda h+c(c(end{aligned}
The 0-1 rule called the Porcelain Process
<strong>Intuitive interpretation: within a sufficient small time, random events either occur once or do not occur</strong>
It proves that the definition of equal value is indeed reasonable.
<strong>Theorem</strong>
If the counting process&#36;N_t&#36;A smooth, independent increment that meets 0-1 Law must be the parameter&#36;\lambda&#36;♪ And the porridge process ♪</li>
</ul>
<h4>Distribution of arrival times during bellows</h4>
<p>Set &#36;&#123;N_t,t\geq0}&#36; is the parameter as&#36;\lambda&#36; The porcelain process is known to be&#36;[0, t)&#36;The only random point that arrived was within.&#36;T_1&#36;It's the time of arrival, the time of arrival at random.&#36;T_{1}&#36;What is the probability distribution?
<strong>Conclusions  &#36;N_t=1&#36;, the first random point of arrival&#36;T_1&#36;Obey.&#36;[0,t]&#36;even distribution on</strong>
&#36;&#36;P(T_1\leq s\big|N_t=1)=\frac{P(T_1\leq s,N_t=1)}{P(N_t=1)}&#36;&#36;
&#36;&#36;\begin{aligned}&amp;=P{N_s=1,N_t-N_s=0}/P(N_t=1)\\&amp;== sync, corrected by elderman == @elder man I'm sorry.
Let's think of something more general.
Set &#36;&#123;N_t,t\geq0}&#36; is the porcelain process with the parameter poignant, if known&#36;[0, t)&#36; Inside only&#36;n&#36;A random spot arrives, a random spot.&#36;n&#36;A time of arrival&lt;{T}_2&lt;...&lt;What is the probability distribution?
&#36;&#36; (u 1, u 2, \cdots, u n) =begin{cases}\frac{n!}&amp;0&lt;u_1&lt;u_2&lt;\cdots&lt;u_n\leq t\0,&amp;\text{Other}&amp;\end{cases}&#36;&#36;</p>
<h4>Two examples.</h4>
<h5>One.</h5>
<p>The number of customers arriving at a station is a perpone process, reaching 5 customers per 10 minutes, and trying to calculate the probability of at least 10 customers arriving at a station within 20 minutes.
Resolve: Order&#36;N_t&#36;Organisation&#36;[0,t)&#36;Number of customers arriving at the station, then&#36;&#123;N_{t},t\geq0}&#36; It's the bellow process.
Parameters are: &#36;\lambda=\frac{5}{10}=0.5&#36;
At least 10 customers will be at the station in 20 minutes.
&#36;&#36;P(N_{20}\geq10)=1-\sum_{k=0}^9\frac{(10)^ke^{-10&#125;&#125;{k!}=0.5402&#36;&#36;</p>
<h5>Two.</h5>
<p>A mechanical device in&#36;[0,t)&#36;N, which is a pine process of 5 times strength and which malfunctions when the 100th vibration occurs</p>
<ul>
<li>Probability density function for life</li>
<li>Average lifetime of the device</li>
<li>Probability density function for two vibration intervals</li>
<li>Average time interval between two vibrating adjacent to each other
We know life expectancy is related to the 100th Porcelain process, so there is.</li>
<li>Function with lifetime of 100 to arrival &#36;T_{100}&#36;</li>
<li>The life expectancy is there. &#36;E[T_{100}]=20&#36;</li>
<li>Time-spacing sequences, and the distribution of bellows.</li>
<li>Mean value of the spacing sequence</li>
</ul>
<h3>Random processes associated with the porcelain process</h3>
<h4>Cobalt Porsche Process</h4>
<p>Set N&#36;={N_t,t\geq0}&#36; is the parameter as&#36;\lambda&#36; The porridge process,&#36;&#123;Y_{k}.k=1,2,...}&#36;is a column of random variables that are separate and distributed, and with&#36;N&#36;I'm going to get you a new one.<em>{k=1}&#36; &#36;Y</em>Other Organiser, t\geq0,  BAR x
Claims &#36;X=&#36; &#36;&#123;X_t,t\geq0}&#36;For the compound porridge process</p>
<p>Set Random Variables &#36;&#123;Y}<em>n(n{=}1,2,...)&#36; 数学期望&#36;&#123;E}Y_n=\mu&#36; 方差&#36;DY n=sigma^2 for compound pine process
&#36;X</em>The hype is a very good idea.
, the difference function and the associated function </p>
<h2>Steady process</h2>
<p>It's a kind of special random process, which can be assumed by name, that their statistical characteristics are not changing over time, and that is something that deserves our research.</p>
<h3>Definition of smooth process</h3>
<h4>Steady process.</h4>
<p>If you want to get &#36;t<del>\tau</del>I'm fine.
&#36;&#36;F_{t_{1},\cdots,t_{n&#125;&#125;(x_{1},\cdots,x_{n})=F_{t_{1}+\tau,\cdots,t_{n}+\tau}(x_{1},\cdots,x_{n})&#36;&#36;
So the limited dimensions of distribution function is not subject to change at any time.
It's a more demanding requirement.</p>
<p>Here's an example of a smooth process.
Set &#36;N={N,:t\geqslant0}&#36;is the parameter &#36;\lambda&gt;0&#36; 的泊松过程，对任意固定的常数 &#36;a&gt;0&#36;,
&#36;&#36;
X_{t}=N_{t+a}-N_{t},\quad\quad\text{其中 }t\geqslant0
&#36;&#36;
It's a smooth process.</p>
<p>This is because the flat and independent increment of the porcelain distribution can be seen as random variables.&#36;X_t&#36;It's all in accordance with the independent Porpine distribution. Steady.</p>
<p>Briefly describe the nature of some smooth processes.
Set &#36;X={X_t:t\in\mathbb{T&#125;&#125;&#36;It's a smooth process, and the second-order rectangular X exists, and it's arbitrary.&#36;t,t_1,t_2\in\mathbb{T}&#36;, the mean function of X&#36;m_x(t)&#36;Is constant, relevant function&#36;R_x(t_1,t_2)&#36;Reliance on time-based indicators only&#36;t_2-t_1&#36;</p>
<h4>Broad and smooth process</h4>
<p>So we've relaxed the definition and given the smooth and broad process.</p>
<p>Set &#36;X={X_t:t\in\mathbb{T&#125;&#125;&#36;is a second-stage rectangular process that may take a compound value. We call it &#36;X&#36; is a broad and smooth process if for any time indicator s, &#36;t\in\mathbb{T}&#36;</p>
<ol>
<li>Mean function for X &#36;m_{x}(t)\equiv C&#36;(of which) &#36;C\in\mathbb{C}&#36; (A constant)</li>
<li>&#36;X&#36; , and then click the&#36;R_x(s,t)=R_x(t-s)&#36;, which is the function of the function&#36;R_x(s,t)&#36;Values depend on time indicators only Bad &#36;t-s&#36; Or as an expression&#36;R_X(t,t+\tau)=R_X(\tau)&#36;
The broad and smooth process is defined by only numerical characteristics, which are easier to verify.</li>
</ol>
<p>In fact, the broad-scale smooth process is more widely used than the smooth process, and the concept of the undefined application of smooth processes is all about smooth processes.</p>
<h4>Examples</h4>
<p>Set &#36;mathrm{X}<em>{\mathrm{t&#125;&#125;=A\cos(\omega t+\Phi),A,\omega&#36;是常数，&#36;\Phi&#36;为随机变量,服从&#36;[0,2\pi]&#36;均匀分布，则称&#36;\mathrm{X=x t,\quadt\geq) \&#36;A random initial believer;
&#36;m</em>{x}( t) = \mathbb{E} [ X_{\mathrm{t} }]=\int_{0}^{2\pi}\frac{1}{2\pi}A\cos(\omega t+\varphi)d\varphi{=}0&#36;
 &#36;&#36;\begin{aligned}
R_{X}(s,t)&amp; ={E}[{X}<em>{s}{X}</em>{\mathrm{t&#125;&#125;]  \
&amp;=\int_{0}^{2\pi}\frac{1}{2\pi}A^{2}\cos(\omega s+\varphi)\cos(\omega t+\varphi)d\varphi  \
&amp;== sync, corrected by elderman == @elder man
I'm sorry, I'm sorry.
So it's a smooth process.</p>
<p>Set&#36;X={X_t,t\geq0}&#36;It's just a pick.&#36;+{-}1&#36; A random process of two values with the number of symbol changes as one parameter&#36;\lambda&#36;The Porcelain Process<em>{t}&#36;, 且对任意的&#36;t&#36;
&#36;&#36;P(X</em>{t}=-1)=P(X_{t}=1)=1/2&#36;&#36;
则称&#36;X&#36;为随机电报信号过程.验证&#36;X&#36;是平稳过程
&#36;&#36;m_X(t)=0&#36;&#36;
&#36;&#36;R_x(t,t+\tau)=\mathbb{E}[X_tX_{t+\tau}]&#36;&#36;
&#36;&#36;=\sum_{k=0}^{\infty}\frac{(\lambda|\tau|)^{2k&#125;&#125;{(2k)!}e^{-\lambda|\tau|}-\sum_{k=0}^{\infty}\frac{(\lambda|\tau|)^{2k+1&#125;&#125;{(2k+1)!}e^{-\lambda|\tau|}&#36;&#36;
&#36;&#36;=e^{-2\lambda|\tau|}&#36;&#36;</p>
<h4>Linkage between the smooth and smooth processes</h4>
<p>As a definition, the smooth process is not necessarily a smooth process, but rather a smooth one.
But we've been able to tell you the nature of the process in a smooth and stable way.
<strong>The second-level rectangular is defined as having a smooth and smooth process to satisfy a wide and smooth process.</strong>
Let's go back and see when the broad and smooth process is smooth.
<strong>The smooth and smooth normal process must be very smooth.</strong>
We've already described the normal process, and here we'll use examples to illustrate the theorem.
Set&#36;Y={Y_t,t\geq0}&#36;It's a normal process. And...&#36;m_{Y}(t)=\alpha+\beta t,\quad C_{Y}(t,t+\tau)=e^{-a|\tau|}&#36;, where &#36;alpha, \beta, \bada a&gt;&#36; 0,000, please.
I'm sorry.
X t=Y t+b}-Y t, t\geq0, \\text{b)&gt;0,
&#36;&#36;
试证明&#36;\mathbf{X}={X_t,t\geq0}&#36;是一严平稳过程
均值函数
&#36;&#36;m_{X}(t)=\operatorname{E}[X_{t}]=\operatorname{E}[\color{}{Y_{t+b}-Y_{t&#125;&#125;]=\beta{b}&#36;&#36;
协方差函数
&#36;&#36;\begin{aligned}C_X(t,t+\tau)&amp;=\mathrm{cov}(X_t,X_{t+\tau})\&amp;=\mathrm{cov}(\color{}{Y_{t+b}-Y_{t},Y_{t+\tau+b}-Y_{t+\tau&#125;&#125;)\end{aligned}&#36;&#36;
&#36;&#36;=2e^{-a|\tau|}-e^{-a|\tau-b|}-e^{-a|\tau+b|}&#36;&#36;
协方差函数
&#36;&#36;R (t, + \tau)=2e^-e}-e}-e}-e}-e\-e\-e\-e\-e\-e}-e}-e}-e}-e}-e}-e}-e}
It's a smooth process, and I'm thinking if it's a normal process.
We can easily use a matrix to transform the normal vector from the current vector.
So it's a normal process.
In conclusion, we have proven that the other problems are the same.</p>
<h3>Related Functions</h3>
<p>The average function and the related function reflect the statistical characteristics of the random process, and because the constant is the constant of the flat process, his main characteristic is to be painted from the relevant function, which is the question to be considered in this section.</p>
<h4>Nature of the function</h4>
<p>Set&#36;X={X_t:t\in\mathbb{T&#125;&#125;&#36;To smooth the process, &#36;X&#36; , and then click the &#36;R_X(\tau)&#36;Be of the following nature</p>
<ul>
<li>&#36;R_{X}(0)=\mathbb{E}[\mid X_{\iota}\mid^{2}]\geqslant0,\quad t\in\mathbb{T}&#36;</li>
<li>&#36;\mid R_{X}(\tau)\mid\leqslant R_{X}(0)&#36;</li>
<li>&#36;\overline{R_{X}(\tau)}=R_{X}(-\tau)&#36;</li>
<li>&#36;\sum_{k=1}^n\sum_{l=1}^n\bar{\alpha}_k\alpha_lR_X(t_k-t_l)\geqslant0&#36;</li>
</ul>
<p>Easy to get smooth process alignments&#36;C_X(\tau)&#36; Satisfied</p>
<ul>
<li>&#36;C_X(0)=\mathcal{D}_X(t)\ge0;&#36;</li>
<li>&#36;|C_X(\tau)|\leq C_X(0)&#36;</li>
</ul>
<p>For a smooth life cycle&#36;X&#36; It's a existence.&#36;T_0&#36;  &#36;X_{t+T_0}=X_t&#36; At this point,
&#36;&#36;00\
R X(\tau+T )&amp; =\mathrm{E}\left[\overline{X}<em>{t}X</em>{t+\tau+T_{0&#125;&#125;\right]  \
&amp;=\mathbf{E}\left[\overline{X}<em>{t}X</em>{t+\tau}\right] \
&amp;=R X}(\tau)
I'm sorry, I'm sorry.
The function is also pro-cyclical.</p>
<p>You can see the function&#36;R_X(\tau)&#36; It reflects the smooth process.&#36;X&#36;The size of the linear relevance of the two random variables we can use to study some properties of the smooth process.
In engineering practice, we generally think that there is no cycle of smoothness.
&#36;&#36;00\
\pepratorname*lim}<em>{\tau\rightarrow\infty}R</em>{X}\left(\tau\right)&amp; =\lim_{r\rightarrow\infty}E\left[X_{t}X_{t+r}\right]  \
&amp;=\lim_{r\to\infty}\langle\mathbf{E}\left[X,\right]\mathbf{E}\left[X_{t+\tau}\right]\rangle  \
&amp;== sync, corrected by elderman == @elder man
I'm sorry, I'm sorry.
That's not what it's all about.</p>
<p>To eliminate the effects of the smooth process itself on the function, we standardize it.
&#36;&#36;r_{X}(\tau):=\frac{R_{X}(\tau)-m_{X}^{2&#125;&#125;{C_{X}(0)}&#36;&#36;
Called the relevant coefficient
He's still used to paint random process time intervals.&#36;\tau&#36;The magnitude of the linear relevance of two random variables is known by definition.&#36;\lim_{r\to\infty}r_X(\tau)=0&#36; </p>
<p>In engineering, we think that when the relevant coefficient is less than&#36;0.05&#36;And then they're not connected.
We can start with&#36;\mid r_{X}(\tau_{0})\mid\leqslant0.05&#36; To calculate the relevant time
It's also working.&#36;\tau_{0}=\int_{0}^{+\infty}r_{X}(\tau)\mathrm{d}\tau&#36; It's a calculation.
<strong>The size of the time involved reflects the speed at which the process is moving, the short time at which it is going, the small influence it has had, the faster the rise and the reverse. Yeah.</strong></p>
<p>Here's a simple example.
Set a steady signal.&#36;\mathrm{X= { X_t: t\geq 0} }&#36;and&#36;\mathrm{Y= { Y_t: t\geq 0} }&#36; The synoptics are:
&#36;&#36;C_X(\tau)\color{}{=}\frac14e^{-2\lambda|\tau|},\quad\quad\quad\quad C_Y(\tau)\color{}{=}\frac{\sin\lambda\tau}{\lambda\tau}&#36;&#36;
Calculates and explains the respective functions and associated time
&#36;r X}(\tau)=\frac{R{X}(\tau)-m XX^2&#125;&#125;C X&#125;&#125;=frac{\matcal{C}<em>{X}(\tau)}{\mathcal{C}</em>{X}(0)}=e^{-2\lambda|\tau|}&#36;&#36;
&#36;&#36;r_{<em>Y}(\tau)=\frac{\mathsf{C}</em>{<em>Y}(\tau)}{\mathsf{C}</em>{<em>Y}(0)}=\frac{\sin\lambda\tau}{\lambda\tau}&#36;&#36;
积分计算相关时间
&#36;&#36;\tau</em>{0}^{\chi}=\int_{0}^{\infty}\mathbf{r}<em>{X}(\tau)d\tau=\int</em>{0}^{\infty}e^{-2\lambda|\tau|}\mathbf{d}\tau=\frac{1}{2\lambda}&#36;&#36;
&#36;&#36;\tau_{0}^{Y}=\int_{0}^{\infty}\mathbf{r}<em>{Y}(\tau)d\tau=\int</em>Other Organiser
That means the X is more up and down than Y.</p>
<h4>Discuss smooth process continuity with relevant functions</h4>
<p>Steady process &#36;X={X_t:t\in\mathbb{T&#125;&#125;&#36; Equal continuing full condition: X-related functions &#36;R_X(\tau)&#36;Yes.&#36;\tau=0&#36;Continuous
Steady process &#36;X={X_t:t\in\mathcal{T&#125;&#125;&#36;. The function is&#36;R_x(\tau)&#36;, and &#36;R_X(\tau)&#36;At any point.&#36;\tau\in\mathbb{R}&#36; The requirement of continuous service &#36;R_x(\tau)&#36; Yes. &#36;\tau=0&#36; Continuous</p>
<h4>Two smooth processes.</h4>
<p>Set &#36;X={X_t:t\in\mathbb{T&#125;&#125;&#36; and &#36;Y={Y_t,t\in\mathbb{T&#125;&#125;&#36;To smooth the process, to be free. &#36;s,t\in\mathcal{T}&#36;- What?
&#36;R XY}(s,:t)=\matbf{E}\left[overline{X}<em>{s}Y</em>{t}\right] &#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;&#36;&#36;US&#36;US&#36;US&#36;US&#36;&#36;US&#36;&#36;US&#36;US&#36;&#36;US&#36;&#36;US&#36;US&#36;US&#36;US&#36;&#36;US&#36;US&#36;US&#36;US&#36;US&#36;&#36;&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US {
If
&#36;R (t, t+\tau)=\matbb{E}\left[\overline{X}<em>{t}Y</em>{t+\tau}\right]=R_{XY}(\tau)&#36;&#36;
则称这两个平稳过程是联合平稳的
互相关系数定义为
&#36;&#36;=xy}(\tau): =frac{R{XY&#125;&#125;m}x}Y}{[C X}(0)]^[C Y}(0)]^
If the number of relationships is zero, it's called two random processes that are irrelevant.
If the interconnective function is 0, then two random processes are active.
We can give the union a smooth and stable character.
If the two smooth processes are combined and smooth, the interrelated functions satisfy</p>
<ul>
<li>&#36;R_{XY}(\tau)=\overline&#123;&#123;R_{YX}\left(-\tau\right)&#125;&#125;&#36;</li>
<li>&#36;|R_{XY}(\tau)|^{2}{\leqslant}R_{X}(0)R_{Y}(0),|R_{YX}(\tau)|^{2}{\leqslant}R_{X}(0)R_{Y}(0)&#36;</li>
<li>&#36;Z_{t}=\alpha X_{t}+\beta Y_{t}&#36; It's a smooth process.</li>
</ul>
<h3>Various patterns</h3>
<p>The use of sample information to study the digital characteristics of smooth processes requires several sample functions to measure smooth processes (classical statistics based on large-digit laws) but multiple observations over time are very difficult to achieve;</p>
<p>Random processes are dual in nature, and at the same time,&#36;w~t&#36;Related, random.&#36;w&#36;It's hard for us to study, but...&#36;t&#36;The function that makes us wonder if we can estimate digital characteristics using a long-detected information? That's the problem with the history of the different forms.</p>
<h4>Introduction</h4>
<p>For smooth process Mean function and time indicator. The function is a time indicator differential function. We consider only one sample function to estimate the average function and the associated function. </p>
<p>Definitions: For a smooth process&#36;X_t&#36; If the average limit is
&#36;&#36;US&#36;&#36;&#36;&#36;&#36;&#36; Xlanx peratorname*l.i.m}<em>{T\to\infty}\frac{1}{2T}\int</em>{\t&#36;US&#36;20.00}
Existing&lt;X_{t}&gt;It's the average time of the smooth process
If for any fixed&#36;\tau&#36; Average limit
I'm sorry.&lt;\overline{X}<em>{t}X_{t+\tau}&gt;=l.i.m\frac{1}{2T}\int</em>{-T}^{T}\overline{X}<em>{t}X</em>Dt
Existing&lt;\overline{X}_{t}X_{t+\tau}&gt;&#36; is the time-related function of a smooth process
For a smooth process with a parameter set greater than zero, we can make the following corrections.
I'm sorry.<x_t>=\underset{T\to\infty}{\operatorname*{l.i.m&#125;&#125;\frac1T\color{}{\int_0^TX_tdt}&#36;&#36;
&#36;&#36;&lt;\overline{X}_tX_{t+\tau}&gt;=li.m\frac1T\int_0^T\overline{X}<em>tX</em>{t+\tau}dt&#36;&#36;</p>
<h4>Definitions</h4>
<p>Set&#36;X={X_t:t\in(-\infty,+\infty)}&#36;It's a smooth process. If it's probability 1, yes.
&#36;&#36;
\langle X_{t}\rangle=m_{_{X&#125;&#125;
&#36;&#36;
The mean function of the equation X, which is called smooth process, has a different history.</p>
<p>If you're looking at any real number&#36;\tau&#36;, with probability 1, yes
&#36; \langle\overline{x}<em>{\iota}X</em>I'm sorry, I'm sorry.
It's called smoothing. &#36;X&#36; . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .</p>
<p><strong>If it's smooth, &#36;X&#36; ..the average function and the related function are of a different historical nature, which is referred to as the smooth process. &#36;X&#36; It's all about history.</strong></p>
<p>The significance of the history is that any sample of the smooth process has experienced the various possible conditions of the process for a sufficient period of time, so that the average of the smooth process and the associated functions can be estimated by the information of the long-observed samples for the smooth process.</p>
<h4>Decision</h4>
<h5>Mean</h5>
<p>Set&#36;X={X_t:{t\in(-\infty,+\infty)&#125;&#125;&#36;It's a smooth process.&#36;C_x(\tau)&#36;Yes.&#36;X&#36; A syntax function, the average X function has a relativity requirement
&#36;&#36;
\lim_{\tau\to\infty}\frac{1}{2:T}\int_{-2:T}^{2:\tau}\Big(1-\frac{|:\tau:|}{2:T}\Big)C_{X}(\tau):\mathrm{d}\tau=0
&#36;&#36;
If&#36;X&#36;It's a smooth process, so the conditions can change.
&#36;&#36;\lim_{T\to+\infty}\frac1T\int_0^{2T}(1-\frac\tau{2T})C_X(\tau)d\tau=0&#36;&#36;
It's because the synonym is a doll.
Yes.&#36;t\ge0&#36;The conditions above are now the same as the conditions.
&#36;&#36;\lim_{T\to+\infty}\frac1T\int_{-T}^T(1-\frac{|\tau|}T)C_X(\tau)d\tau=0&#36;&#36;
And if it's still smooth, it can continue to be transformed.
&#36;&#36;\lim_{T\to+\infty}\frac2T\int_{0}^{T}(1-\frac\tau T)C_{X}(\tau)d\tau=0&#36;&#36;
We can also offer a condition that is sufficiently unnecessary.
Steady process X=&#36;&#36;x t,-\infty&lt;t&lt;+infty}&#36; +infty}Accompanying difference
&#36;&#36;\lim_{\tau\to\infty}C_X(\tau)=0&#36;&#36;
The average of X is of a different kind. <strong>This is a better theorem than before, and of course it's not necessary.</strong></p>
<h5>Related Functions</h5>
<p>Set &#36;X={X_t:t\in(-\infty,+\infty)}&#36;It's a smooth process and a fixed real number. &#36;\tau&#36;You're...
I'm sorry.
Y t}\overline{X}<em>{t}X</em>{t+\tau}
&#36;&#36;
若&#36;Y=\langle Y_t:t\in(-\infty,+\infty)\rangle&#36;是平稳过程，则&#36;X&#36;的相关函数具有的各态历经性是新的平稳过程的均值的各态历经性 把协方差替换了就好
所以充要条件为
&#36;&#36;
\lim\limits_{\tau\to\infty}\frac{1}{2T}\int_{-2T}^{2\tau}\left(1-\frac{\mid\boldsymbol{u}\mid}{2T}\right)(R_Y(u)-\mid R_X(\tau)\mid^2):\mathrm{d}u=0
&#36;&#36;
<strong>The principle of the averages in the front is used here again.</strong></p>
<h3>Power spectral density</h3>
<p>In the radio, communications, there are a number of issues that usually require analysis of the frequency-area structure of the smooth process, and therefore the power-spectral density of the smooth process.</p>
<h4>The concept of power spectrum density</h4>
<p>&#36;X={X_t:t\in\mathbb{T&#125;&#125;&#36; It's a smooth process. Remember.
&#36;&#36;S_{\chi}(\omega)=\lim_{T\to+\infty}\frac1{2T}\mathrm{E}\left|\int_{-T}^{T}e^{-j\omega t}X_tdt\right|^2&#36;&#36;
Claims &#36;S_x(\omega)&#36;To smooth the process. &#36;X&#36; Power spectrum density, short spectrum density
Also known as
&#36;&#36;\lim_{T\to\infty}\frac{1}{2T}\mathrm{E}\Big[\int_{-T}^{T}|X_{t}|^{2}\mathrm{d}t\Big]&#36;&#36;
Average power for smoothing the process</p>
<p>Theorem, smoothing the process. &#36;X={X_t:t\in\mathbb{T&#125;&#125;&#36; , and then click the &#36;R_x(\tau)&#36;It's absolutely sizable, and there is.
&#36;&#36;S_X(\omega)=\int_{-\infty}^{+\infty}e^{-j\omega t}R_X(\tau)dt&#36;&#36;
So we know. <strong>The function and spectrodenity are a pair of Fouriers.</strong> We call it the Sinchinvina formula.
&#36;US&#36;S X(\mega)=int=-\infty^e^-\omega\tau}R tau,-\info&lt;\omega&lt;+\infty &#36;&#36;
&#36;&#36;R_X(\tau)=\frac1{2\pi}\int_{-\infty}^{+\infty}e^{j\omega\tau}S_X(\omega)d\omega,-\infty&lt;\tau&lt;+\infty &#36;&#36;</p>
<h4>Nature of spectral density</h4>
<p>Set&#36;S_X(w)&#36; It's a spectrum density.</p>
<ul>
<li>&#36;\text{谱密度 }S_X(\omega)\text{为实值非负函数}&#36;</li>
<li>&#36;\text{如果 }X\text{ 为实平稳过程,则谱密度 }S_X(\omega)\text{为偶函数。}&#36;</li>
<li>&#36;\begin{cases}S_X(0)=\int_{-\infty}^{+\infty}R_X(\tau)d\tau\\R_X(0)=\frac{1}{2\pi}\int_{-\infty}^{+\infty}S_X(\omega)d\omega\end{cases}&#36;
The equation function is &#36;R_X(t,t)&#36;  The basis of the digital signature is already mentioned.
The former can directly export the values of spectrodenity at zero from the function in question, avoiding the points in the plural.
The latter can export the value of the function at 0 from the spectrum density, which is the average power.</li>
</ul>
<h4>Calculation of power spectral density</h4>
<p>The exact exact exact exact exact exact exact exact exact exact exact exact exact exact exact exact exact exact exact exact exact exact exact exact exact exact exact exact exact exact exact exact exact exact exact exact exact exact exact exact exact exact exact exact exact exact exact exact exact exact exact exact exact exact exact exact exact exact exact exact exact exact exact exact exact exact exact exact exact exact exact exact exact exact exact exact exact exact exact exact exact exact exact exact exact exact exact exact exact exact exact exact meaning of the exact object of the object.
The nature of the changes that complement the Fouriers are as follows:</p>
<ul>
<li>&#36;\mathcal{F}[\alpha f_{1}(t)+\beta f_{2}(t)]=\alpha\mathcal{F}[f_{1}(t)]+\beta\mathcal{F}[f_{2}(t)]&#36;</li>
<li>&#36;\mathcal{F}[f(t\pm t_0)]=e^{\pm jat_0}\mathcal{F}[f(t)]&#36;</li>
<li>&#36;\mathcal{F}[f^{(n)}(t)]=(j\omega)^{n}\mathcal{F}[f(t)]&#36;
It's important to see the example. Yeah.</li>
</ul>
<h3>Spectrolysis</h3>
<h4>Spectrolysis of related functions</h4>
<p>Set&#36;X={X_i:t\in\mathbb{T&#125;&#125;&#36;is a smooth, continuous process of the average, and the function of the function is relevant &#36;R_x(\tau)&#36; Other Organiser
&#36;&#36;
R_{X}\left(\tau\right)=\frac{1}{2\pi}\int_{-\infty}^{+\infty}\mathrm{e}^{i\omega\tau}:\mathrm{d}F_{X}\left(\omega\right),\quad\tau\in\left(-\infty,:\infty\right)
&#36;&#36;
This is the basic theorem for the function spectrolysis. &#36;F_X(w)&#36;It's called a spectro-function.</p>
<p>We can easily verify that if the function is absolutely staggered, we can tell by using the Sinchinvina formula.
&#36;&#36;F_{X}\left(\omega\right)=\int_{-\infty}^{\omega}S_{X}\left(\omega\right)\mathrm{d}\omega &#36;&#36;
It's a more relaxed function of the calculator. </p>
<p>One example
X.Y. is two randomly independent variables.&#36;E(X)=0~D(X)=1&#36; &#36;Y&#36;The distribution function is&#36;F(x)&#36;You're...
&#36;Z <em>t}=Xe^{jtY},-\infty&lt;t&lt;+\infty,&#36;&#36;
计算&#36;Z&#36;的谱函数
容易计算
&#36;&#36;m</em>{z}(t)=0&#36;&#36;
&#36;&#36;\begin{gathered}
R_{z}(t,t+\tau)=\int_{-\infty}^{+\infty}e^{j\tau\omega}dF(\omega) \
=\frac{1}{2\pi}\int_{-\infty}^{+\infty}e^{j\tau\omega}d(2\pi F(\omega))
\end{gathered}&#36;&#36;
再代入进行傅立叶变化计算谱密度函数 代入前面的公式计算谱函数有
&#36;&#36;F_{z}(\omega)=2\pi F(\omega)&#36;&#36;</p>
<h4>Spectrolysis of smooth processes</h4>
<p>We're just here to explain.
<strong>The continuous smoothing of the mean zero average can be spectrolysis.</strong></p>
<p>The real meaning of the the theorem is that
<strong>The smooth process can be seen as the limited supersing of the condensed wave of an amplitude, the agular frequency, and the average limit.</strong></p>
