---
title: 'Markov Chains: Transition Probabilities, State Classification, and Stationary Distributions'
title_zh: Markov Chain：转移概率、状态分类与平稳分布
date: 2025-09-11 22:53:52 +0800
categories:
- Data Science
- Time Series & Spatial Data
tags:
- Statistics
- Stochastic Processes
- Markov Chains
author: Hyacehila
mathjax: true
hidden: true
excerpt: Covers discrete-time Markov chains, transition probabilities, state classification, stationary distributions, and
  core properties.
description: Covers discrete-time Markov chains, transition probabilities, state classification, stationary distributions,
  and core properties.
excerpt_zh: 整理离散时间马尔可夫链、转移概率、状态分类、平稳分布和相关基础性质。
permalink: /blog/2025/09/11/markov-chain-notes/
lang: en
translation_key: 2025-09-11-markov-chain-notes
translation_status: machine
translation_source_hash: d6f8cc5f095b8e181f23ec5fddd186e00827b74dab7456d8117fe562b7d7c185
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<h2>Markov Chain at Breaktime</h2>
<p>And then we'll introduce a special series of random variables, which is characterized by the fact that the results are only now affecting the next results, and that the results before them have no impact on the next results; that's the Marcov chain.
He's both mathematically easy to calculate and more in keeping with the randomness of reality, and is now widely used.
Recent research has allowed the Malkov chain to play an important role in the Monte Carlo approach.
We started working on it from the break-up.</p>
<h3>Definition of the Markov chain at discrete time</h3>
<h4>Basic definitions</h4>
<p>Definitions&#36;X={X_n:n\geqslant0}&#36;is defined in probability space (in the case of a "fault")&#36;\Omega,\widehat{F},\mathbb{P})&#36;random process on which the state space is a numerical assembly S if it is for any non-negative integer &#36;n\geq0&#36; and&#36;i_0,i_1,...,i_n,i_{n+1}\in S&#36; Yes.
&#36;&#36;00\
&amp;P(X_{n+1}=i_{n+1}|X_{0}=i_{0},X_{1}=i_{1},\cdots,\color{}{X_{n}=i_{n&#125;&#125;) \
&amp;♪ I'm a little girl ♪
I'm sorry, I'm sorry.
This random process is called a separate Markov chain of time.</p>
<p>The questions in this article can also be addressed<a href="/en/blog/2023/03/18/stochastic-process-basics-notes/">Random process basis: random process definition, digital characteristics and smooth process</a>、<a href="/en/blog/2024/10/17/financial-stochastic-analysis-notes/">Financial random analysis: financial derivatives, fork tree pricing and the theory of arbitrage</a>How the concept of a relatively close read together is developed in different contexts.</p>
<p>The definition of Markov Chain is the most essential feature we can call<strong>Marquisity.</strong>Or amnesia. <strong>Only now will the results affect the next results, and the previous results will not affect the next results.</strong>
Conditional probability in expression
&#36;&#36;P(X_{n+1}=i_{n+1}|X_{n}=i_{n})&#36;&#36;
Called from the state&#36;i_{n}&#36;To Status&#36;i_{n+1}&#36;And the probability of a step of diversion is recorded as&#36;p_{i_{n}i_{n+1&#125;&#125;(n)&#36;
Additional definitions:
&#36;&#36;p_{ij}^{(k)}(n)\stackrel&#123;&#123;def&#125;&#125;{=}{P}(X_{n+k}=j\mid X_{n}=i)&#36;&#36;
From a state&#36;i&#36;Moving out. Experience.&#36;k&#36;- It's a transfer to Ta.&#36;j&#36;The probability of a&#36;n&#36;The moment.&#36;k&#36;Sub-transfer probability
It's clear that the probability of a single transfer is...&#36;k&#36;Special case of the probability of a secondary transfer
We're in a deal.&#36;k&#36;The matrix of secondary probability is presented below.
&#36;&#36;\left.\matbf{P&#125;&#125;loft(n\right)=\left(\begin{matrix&#125;&#125;p}<em>{ij}^{(k)}\left(n\right)\\end{matrix}\right.\right)&#36;&#36;
&#36;k=1&#36;的时候 一步转移概率矩阵为
&#36;&#36;\mathbf{P}^{(1)}(n)=(p</em>{ij}^{(1)}(n))&#36;&#36;
简记为 &#36;\color{}{\mathbf{P}(n)=(p_{ij}(n))}&#36;
对于&#36;k=0&#36;的情况 我们约定
&#36;&#36;\left.p_{ij}^{(0)}(n)=\delta_{ij}=\left{\begin{matrix}{1,}&amp;{i=j}\{0,}&amp;{i\neq j}\\end{matrix}\right.\right.\quad(i,j\in S)&#36;&#36;
此时转移概率矩阵是单位矩阵
不难验证 转移概率矩阵是随机矩阵 即
&#36;&#36;p_{ij}^{(k)}\left(n\right)\geqslant0,\quad\sum_{j\in S}p_{ij}^{\left(k\right)}\left(n\right)=1&#36;&#36;</p>
<h4>Zhiji Markov chain.</h4>
<p>In practical applications, we'll see a more specific nature of the Maast chain:<strong>One step to transfer probability and time&#36;n&#36;It's not about that.</strong>&#36;&#36;p_{ij}\left(n\right)=p_{ij}\left(n+1\right)=p_{ij}\left(n+2\right)=\cdots &#36;&#36;At this point, we call the chain "Cyber" "Cyber" or "Cyber" "Cyber" Chain
<strong>We're looking at the chain after that, which is often a series of times, and it's a little complicated if we use different transfer probability matrices at every time.</strong>
<strong>One intuitive way of expressing a probability of a single shift: state chart; it's just a matter of writing all the states, and then drawing the probability of a single move from each to the other.</strong></p>
<h4>Examples</h4>
<p>We've given some examples of the Marcov chain and researched how to verify that a random process was Markov. Chain</p>
<h5>Unlimited randomly swimming</h5>
<p>Unlimited random motion on the pivot if&#36;p&#36;And the probability of one. &#36;q&#36;One less chance can shift the probability from definition to definition.
&#36;&#36;p j==begin{cases}p,&amp;\text{i=0,±1,±2,...,}j=i+1\q,&amp;\text{i=0,±1,±2,...,}j=i-1\0,&amp;|i-j|&gt;1\end{cases}&#36;&#36;
简单理解一下我们是怎么写出来的
从&#36;i&#36;状态一步转移到&#36;j&#36;状态的条件是两者得相邻 一侧是&#36;p&#36; 另一侧是&#36;q&#36; 如果不相邻 那概率就是0
这当然满足马氏性（从直观的理解上） 我们用定义也可以做到验证
我们尝试用定义验证他的马氏性
还是从变换的机理出发 把&#36;X_n&#36;变为一系列随机变量序列的和（里面的随机变量是广义的伯努利分布）
&#36;&#36;\mathbf{X}<em>{n}=\xi</em>{0}+\xi_{1}+\cdots+\xi_{n}&#36;&#36;
所以有
&#36;&#36;\begin{gathered}
P(X_{n+1}=i_{n+1}\big|X_{0}=i_{0},X_{1}=i_{1},\cdots,\color{}{X_{n}=i_{n&#125;&#125;\big) \
=\frac{P(X_{0}=i_{0},X_{1}=i_{1},\cdots,X_{n}=i_{n},X_{n+1}=i_{n+1})}{P(X_{0}=i_{0},X_{1}=i_{1},\cdots,X_{n}=i_{n})} \
=\frac{P(\xi_{0}=i_{0},\xi_{1}=i_{1}-i_{0},\cdots,\xi_{n+1}=i_{n+1}-i_{n})}{P(\xi_{0}=i_{0},\xi_{1}=i_{1}-i_{0},\cdots,\xi_{n}=i_{n}-i_{n-1})}
\end{gathered}&#36;&#36;
根据独立性化简有
&#36;&#36;=P(\xi_{n+1}=i_{n+1}-i_{n})&#36;&#36;
也就是
&#36;&#36;\begin{aligned}=\begin{cases}p,&amp;i_{n+1}-i_n=1\1-p,&amp;i_{n+1}-i_n=-1\0,&amp;\text{Other}\end{cases}\end{aligned}&#36;&#36;
&#36;&#36;=P (x+n+n) &#36;
So the definition also validates marzipanity, and the other questions of proof of marzipanity start from a similar point of view here.</p>
<h5>There's a limit to randomly moving.</h5>
<p>Set Particle Point on Line&#36;{0，1，···，a}&#36;I'm gonna do random jiggling on every dot. I'm gonna do the following.</p>
<ul>
<li>Except...&#36;0&#36;and&#36;a&#36;Up the twilight &#36;p&#36;Shift right &#36;q&#36;Shift left &#36;r&#36;No change.</li>
<li>Yes.&#36;0&#36;Move around&#36;r_{0}&#36; No change. &#36;p_0&#36; Shift right</li>
<li>Yes.&#36;1&#36;Move around&#36;r_{a}&#36; No change. &#36;p_a&#36; Shift left
Our rules are right about time.&#36;n&#36;No restrictions. All of them are Zidmar. Chain
One step to transfer the probability matrix to
&#36; \matbf{&amp;p_0&amp;0&amp;0&amp;\cdots&amp;0&amp;0&amp;0\q&amp;r&amp;p&amp;0&amp;\cdots&amp;0&amp;0&amp;0\0&amp;q&amp;r&amp;p&amp;\cdots&amp;0&amp;0&amp;0\\vdots&amp;\vdots&amp;\vdots&amp;\vdots&amp;\cdots&amp;\vdots&amp;\vdots&amp;\vdots\0&amp;0&amp;0&amp;0&amp;\cdots&amp;q&amp;r&amp;p\0&amp;0&amp;0&amp;0&amp;\cdots&amp;0&amp;q_a&amp;r_a\end{bmatrix}&#36;&#36;</li>
</ul>
<h5>Group growth</h5>
<p>Each individual with a biological group produces a future by its own means during its lifetime, assuming that each individual is based on probability.&#36;p_k&#36;Generate&#36;k&#36;A generation, and there are
&#36;&#36;p_k\geq0,\sum_kp_k=1&#36;&#36;
Use&#36;X_n&#36;Other Organiser&#36;n&#36;He's a Marx. Chain
To study the probability of a single move requires that we know the number of offspring of each creature, so...
Definitions&#36;i&#36;The number of offspring of an individual is random.
&#36;&#36;P(\xi_{l}=k)=p_{k}&#36;&#36;
Natural one step from definition to probability
&#36; \begin{aligned}p j}&amp;=P(X_{n+1}=j\big|X_{n}=i\big)\&amp;=P (1}22}cdots+\i}j\big)\end{aligned} &#36;
The exact nature of the sub-resistence is natural.
<strong>We're both starting with the most natural sense, starting with the mechanisms of change, and finally getting the probability matrix.</strong></p>
<h3>The probability distribution of the Marcov chain</h3>
<h4>Chapman-kolmogorov equation (C-K equation)</h4>
<p>Set&#36;X={X_{n},n\ge 0}&#36;It's a horse chain on a state space S, and there is.
&#36;&#36;p_{ij}^{(k+m)}(n)=\sum_{l\in S}p_{il}^{(k)}(n)p_{lj}^{(m)}(n+k),n,m,k\geq0,i,j\in S&#36;&#36;
Or in matrix form.
&#36;&#36;\mathbf{P}^{(k+m)}(n)=\mathbf{P}^{(k)}(n)\mathbf{P}^{(m)}(n+k)&#36;&#36;
The conclusion of this equation tells us
<strong>The Markov chain.&#36;k&#36;The probability of a step transfer is fully determined by the probability of a step transfer.</strong>
If he was a Zilong Ma's chain,
We just need to get it.&#36;m=1&#36; We can move and move.&#36;k&#36;Step Transfer Export&#36;k+1&#36;Step shift
If you take it again,&#36;k=1&#36;  You can export all kinds of transfer probability distributions.
<strong>Intuitive interpretation</strong>
System&#36;n&#36;from Status&#36;i&#36;Let's go, bye.&#36;k+m&#36;Step shift,&#36;n+k+m&#36;Time reached state&#36;j&#36;It's a good idea.
System&#36;n&#36;from Status&#36;i&#36;Let's go. First.&#36;k&#36;Step shift,&#36;n+k&#36;When it reaches a certain point
Intermediate state&#36;l&#36;Again.&#36;n+k&#36;from this midstate&#36;l&#36;Let's go.&#36;m&#36;Step shift,&#36;n+k+m&#36;Time reached state&#36;j&#36;, and the middle statel needs to be taken through the entire state space&#36;S&#36;
<strong>Note that the matrix indicates that it is possible to calculate multiple-step transfer probability matrices directly using the matrix product, and that it is better to do it in a series of times. Fine.</strong></p>
<h4>Initial distribution versus absolute distribution</h4>
<h5>Initial Distribution</h5>
<p>The Maple Chain.&#36;X={X_n}&#36;. The status space is&#36;S&#36; Remember &#36;q_j^{(0)}=P(X_0=j)&#36;, &#36;j\in S&#36;
Name of probability distribution &#36;{q_j^{(0)}:j\in S}&#36; It's the initial distribution of the Maze Chain X.
Vector
&#36;&#36;\mathbf{q}^{(0)}=(q_1^{(0)},q_2^{(0)},\cdots q_j^{(0)},\cdots)&#36;&#36;
The initial distribution vector for the Marcov chain
The initial distribution study is that we don't do any transfer, the initial time, the probability of each state.</p>
<h5>Absolute distribution</h5>
<p>The Maple Chain.&#36;X={X_n}&#36; . The status space is&#36;S&#36; Remember &#36;q_{j}^{(n)}=\mathbf{P}(X_{n}=j)&#36;I'm not sure.
Name of probability distribution &#36;{q_j^{(n)}:j\in S}&#36; It's the absolute distribution of the Maze Chain X.
Vector
&#36;&#36;\mathbf{q}^{(n)}=(q_1^{(n)},q_2^{(n)},\cdots q_j^{(n)},\cdots)&#36;&#36;
It's the absolute distribution of the Marcov chain.
The absolute distribution study is the probability of a state of last resort at a time after constant transfer.</p>
<h5>The link between the two</h5>
<p>Zhiji Markov chain.&#36;X&#36;The absolute distribution is fully determined by its initial distribution and the probability of a step transfer.
Probability form is
&#36;&#36;q_j^{(n)}=\sum_{i\in S}q_i^{(0)}p_{ij}^{(n)}(0)&#36;&#36;
The matrix is as
&#36;&#36;{q}^{(n)}={q}^{(0)}{P}^{(n)}(0)&#36;&#36;
The Zilong Ma's chain can be removed.
<strong>We use the CK equation to move from one step to another; this is the the theorem that can now export the absolute distribution of any time from the initial distribution.</strong></p>
<h4>Limited dimensions distribution</h4>
<p>The Markov chain.&#36;X&#36;The limited dimensions of distribution are fully determined by their initial distribution and the probability of a step transfer
We gave the formula there.
&#36;&#36;P{X_{t_1}=i_1,X_{t_2}=i_2,\cdots,X_{t_n}=i_n}&#36;&#36;
&#36;&#36;=\sum_{i\in S}q_i^{(0)}\cdot p_{ii_1}^{(t_1)}(0)\cdot p_{ii_2}^{(t_2-t_1)}(t_1)\cdots p_{i_{n-1}i_n}^{(t_n-t_{n-1})}(t_{n-1}).&#36;&#36;
<strong>All the examples in this section are treated in the same way; whatever the problem, it turns into a demand.&#36;k&#36;The problem of moving probability matrices and initial distribution for absolute distribution; just figure out the problem.</strong></p>
<h3>Classification of the Marcov chain state</h3>
<h4>Status Type Definition</h4>
<p><strong>Study yourself.</strong>
Definitions:
&#36;&#36;f_{ij}^{(n)}=P{X_n=j,X_k\neq j,k=1,2,\cdots,n-1|X_0=i}&#36;&#36;
Yes&#36;0&#36;Time from Status&#36;i&#36;Let's go. &#36;n&#36;First time after step shift&#36;j&#36;The probability of a single-time probability.
&#36;&#36;f_{ij}^{(+\infty)}=P{X_{n}\neq j,n=1,2,\cdots|X_{0}=i}&#36;&#36;
Yes&#36;0&#36;Time from Status&#36;i&#36;Let's go. Never get there.&#36;j&#36;Probability
&#36;&#36;f_{ij}=\sum_{n=1}^{\infty}f_{ij}^{(n)}&#36;&#36;
Yes&#36;0&#36;Time from Status&#36;i&#36;We're moving out, and we're going to get to the state after a limited movement.&#36;j&#36;The probability of a time-out is called the probability of a time-out.
Special &#36;i=j&#36; Time
&#36;f_{ii}&#36; Yes&#36;0&#36;Time from Status&#36;i&#36;We're going to go back after a limited move.&#36;i&#36;Probability
Definitions:</p>
<ul>
<li>If&#36;f_{ii}=1&#36; And I'm a permanent return.</li>
<li>If &#36;f {ii}&lt;&#36;1 million, which is called state i is very retrograde, skimmed.
If a state is always back, then there is.&#36;\begin{aligned}f_{ii}=\sum_{n=1}^{\infty}f_{ii}^{(n)}=1\end{aligned}&#36;  Which means it's a probability distribution.
&#36;&#36;\mu_{ii}=\sum_{n=1}^{\infty}n\cdot f_{ii}^{(n)}&#36;&#36;
Average time called return
If a state is always returned, if</li>
<li>&#36;\mu_{ii}&lt;\infty&#36; says it's normal.</li>
<li>&#36;\mu_{ii}=\infty&#36;  It's called a zero return.
<strong>You can't just start with definitions.</strong></li>
</ul>
<p>The maximum number of conventions to be assembled by GCD functions
&#36;d i=mathrm{GCD}n\geq1, p ii}n}n}n}&gt;I'm sorry
If &#36;d i&gt;1&#36; 称为周期状态 周期为&#36;d i&#36;
If &#36;d_i=1&#36; Called a non-cyclical state
<strong>If Status&#36;i&#36;Normal return to non-cycle is called "performing " ; if normal return to cycle, call it " cyclical "</strong>
<strong>Cycle is a concept of permanent return, very often not a study cycle</strong></p>
<h4>Status Type Adjudication</h4>
<p>Theorem: For the Marx chain</p>
<ul>
<li>&#36;i&#36;It's always back.&#36;(f_{ii}=1)&#36;Other Organiser &#36;\sum_{n=1}^{\infty}p_{ii}^{(n)}=+\infty&#36;</li>
<li>&#36;i&#36;Very good.&lt;1)&#36;充要条件为 &#36;\sum_{n=1}^{\infty}p_{ii}^{(n)}&lt;+infty
This is the theory that allows us to...&#36;n&#36;The probability of a step shift begins to determine normality.</li>
</ul>
<p>Theorem: Set-up status&#36;i&#36;It's always coming back, then. </p>
<ul>
<li>&#36;i&#36;It's a zero-sum condition.&#36;\lim_{n\to\infty}p_{ii}^n=0&#36;</li>
<li>&#36;i&#36;It's a perfect condition for the waltz.&gt;0&#36; </li>
<li>&#36;i&#36;It's a condition of normal return cycle.&#36;\lim_{n\to\infty}p_{ii}^n&#36;  Cannot initialise Evolution's mail component.</li>
</ul>
<p>Inference: if&#36;j&#36;Yes, very or very often. Any situation is arbitrary.&#36;i&#36; Yes.
&#36;&#36;\lim_{n\to\infty}p_{ij}^{(n)}=0&#36;&#36;
Inference: for non-cyclical issues</p>
<ul>
<li>If it exists&#36;n&#36; Let &#36;p ii}n}n}&gt;0,p_{ii}^{(n+1)}&gt;0&#36; 则&#36;i&#36;/&#36;/non-cycle</li>
<li>If positive number exists&#36;m&#36; Status&#36;j&#36; Let &#36;p ({\mmathrm}&gt;0&#36;则&#36;j&#36; Non-cycle</li>
</ul>
<h4>Relationship between status</h4>
<p><strong>Studying two states together.</strong>
Definitions:
If it exists&#36;n\ge 1&#36; Let &#36;p j}n}&gt;0&#36;  则称状态i可达状态j 记为 &#36;I'm not a good guy.
If two states are possible, it's called two states.
Easy to verify</p>
<ul>
<li>It's transmissible.</li>
<li>Interconnectivity is transmissible.</li>
<li>Interconnectivity is symmetrical.
Theoretically:</li>
<li>&#36;i\rightarrow j\Leftrightarrow f_{ij}&gt;0&#36;</li>
<li>If I ever come back and... &#36;i\rightarrow j&#36; There is. &#36;f_{ji}=1&#36; So the two are connected.
Theoretically:
Two interoperable states:
Either it's the same or it's the same or it's the same or it's the same or it's the same or it's the same.
<strong>So we can extrapolate from one state to another.</strong></li>
</ul>
<h4>Disaggregation of state space</h4>
<p>It's easy to verify, to satisfy each other with self-reversible symmetry, transmission, that's an equal value relationship.
Then we can divide the price classes.
Because the state of interconnectivity is the same, and that's the basis for decomposition.</p>
<p>Theorem: contains equivalents of normal return Category&#36;S_n&#36;It's not closed. Set
Theorem: state space of the chain&#36;S&#36;But the only way to break down is to divide into a limited or unlimited array of non-interconnected sub-sets, which is,
&#36;&#36;S=D\cup C_1\cup C_2\cup\cdots &#36;&#36;
Of which&#36;D&#36;It's a very retrogressive subset of state. &#36;C&#36;It's all about the non-closure of a state of normality.</p>
<p>Theorem: X is a limited-state unicorn chain, and</p>
<ul>
<li>The X's very back-state set D can't be closed.</li>
<li>X doesn't have a zero-return state.</li>
<li>If X is not available, all X's state is returned.
<strong>These are theorems that allow us to decompose the space, to decompose or to transfer the probability. Figure</strong></li>
</ul>
<h4>Analysis of status from status map</h4>
<p>The CK equation is too complicated. The state map is the easiest.</p>
<ul>
<li>All the people who have gone out of nowhere are very, very happy.</li>
<li>It's always the same when you go out and you can go back to yourself.</li>
<li>There's no one in the chain with a single one.</li>
<li>The study cycle is being conducted by manual search for more than zero pii for maximum number of conventions</li>
<li>Interconnective status is the same type of state</li>
</ul>
<h3>Maximum probability of diversion</h3>
<p>The limit of the probability of a transfer is,&#36;&#36;\lim_{n\to\infty}P_{ij}^{(n)}&#36;&#36;including the existence and&#36;i&#36;Is it irrelevant?</p>
<h4>Maximum distribution</h4>
<p>Set&#36;X={X_n,n=0,1,...}&#36;For the second-martial chain, if any, &#36;i,j&#36;
&#36;\lim_{n\to\infty}p_{ij}^{(n)}=\pi_j&#36;, and \pi j&gt;0,\sum_{j\in S}\pi_j=1&#36; &#36;I'm sorry.
then&#36;{\pi_j,j\in S}&#36;It's a probability distribution, called the maximum distribution of the Mall chain.
And the rest of the research is about this.</p>
<h4>Distribution from state-of-the-art space research limit</h4>
<p>When &#36;i<del>J.A.A.T.A.T.A., right?
&#36;&#36;\lim_{n\to\infty}p_{ij}^{(n)}=0&#36;&#36;
When &#36;i</del>j.&#36;0.00 for normal return-cycle subsets Time
&#36;&#36;\lim_{n\to\infty}p_{ij}^{(n)}\text{不存在}&#36;&#36;
Now let's start with the calculation.
&#36;&#36;\lim_{n\to\infty}\color{}{p_{ij}^{(nd_j+r)&#125;&#125;&#36;&#36;
Give theorem:&#36;j&#36;It's normal to return, but...
&#36;&#36;\lim_{n\to\infty}p_{ij}^{(nd_j+r)}=f_{ij}(r)\frac{d_j}{\mu_{jj&#125;&#125;&#36;&#36;
And to calculate this limit probability, you need to use it.&#36;n&#36;Stepover Period Average return time
Inference: When the chain is an unattainable chain, for any dollar I<del>I've got it.
&#36;US&#36;milm n\infory}p =(n)}\frac{1}{\mu}j}&gt;&#36;0.00
We just need to calculate the average return time, but we can keep it simple.
Theorem: When the chain is an unattainable chain, for any dollar I</del>I've got it.
&#36;&#36;\lim_{n\to\infty}p_{ij}^{(n)}=\frac{1}{\mu_{jj&#125;&#125;\stackrel{def}{=}\pi_{j}&#36;&#36;
And there is. &#36;\pi_j&#36;It's a linear equation group. &#36;\begin{aligned}x_j=\sum_{i\in S}x_ip_{ij}\end{aligned}&#36; Conditions met&#36;x_{j}\geq0,\sum_{j\in S}x_{j}=1&#36; The only solution.
We'll calculate the maximum distribution later on, just the equation group, and then we'll use the penultimate to get back to the average time.
The equation group, which we introduced in the algebra, can be solved slowly even with hand count.</p>
<h3>A smooth distribution of transfer probability</h3>
<h4>Smooth distribution</h4>
<p>Probability distribution&#36;\pi_j&#36; It's a smooth distribution of the Zilong Ma's chain.
&#36;&#36;\pi_j=\sum_{i\in S}\pi_ip_{ij},\quad j\in S&#36;&#36;
Or the matrix is.
&#36;&#36;\pi=\pi P&#36;&#36;
of which&#36;\pi&#36;It's the limit distribution vector. &#36;P&#36;It's a transfer probability matrix.
<strong>The smooth distribution is defined on the basis of the maximum distribution.</strong>
Because a single shift is constant, there must be one.
&#36;&#36;\pi_j=\sum_{i\in S}\pi_ip_{ij}^{(n)}&#36;&#36;
Theoretically: If&#36;\pi&#36;It's a smooth distribution of a unicorn chain.&#36;\pi&#36;For initial distribution yes
One:
&#36;&#36;P(X_n=i)=\sum_{k\in S}P(X_0=k)P(X_n=i\begin{vmatrix}X_0=k\end{vmatrix}&#36;&#36;
&#36;&#36;=\sum_{k\in S}\pi_{k}p_{ki}^{(n)}=\pi_{i}&#36;&#36;
Which means absolute distribution is constant.
II: Making a decision&#36;t,n,m,i&#36; Yes.
&#36;&#36;P(X_{t_1+m}=i_1,\cdots,X_{t_n+m}=i_n)=P(X_{t_1}=i_1,\cdots,X_{t_n}=i_n)&#36;&#36;
The Matheric is a steady time series.</p>
<h4>Studying the existence and calculation of smooth distribution</h4>
<h5>Unarranged Ma's chain - all the way through the chain.</h5>
<p>Only limit distribution&#36;&#36;{\pi_{j}=\frac{1}{\mu_{jj&#125;&#125;,j\in S}&#36;&#36;The equations that calculate this flat distribution can be obtained from the first section's understanding equation.</p>
<h5>Unaccepted Marx chain - cycle chain</h5>
<p>Assumptions&#36;X&#36;It's a chain of maxs, and every state in the space is returned in a normal way.&#36;d&#36;There is.&#36;X&#36;There's only one smooth distribution.&#36;{\pi_{j}=\frac{1}{\mu_{ij&#125;&#125;,j\in S}&#36;
He can also solve the equation. Group
&#36;&#36;00\
&amp;\pi_{j}=\sum_{i\in S}\pi_{i}p_{ij} \
&amp;== sync, corrected by elderman == @elder man
I'm sorry, I'm sorry.
Got it.</p>
<h5>Usually a Zhiqikov chain.</h5>
<p>Set Status Space to&#36;S=D\cup C_{0}\cup C_{1}\cup\cdots&#36; of which&#36;D&#36;It's a very retrograde state. Set &#36;C_0&#36;It's a zero-sum situation. Set&#36;C_m&#36;It's normal to close the collection.&#36;H=\bigcup_{k\geq1}C_{k}&#36;  then</p>
<ul>
<li>&#36;X&#36;There is no requirement for a smooth distribution. &#36;H=\Phi&#36; </li>
<li>&#36;X&#36;The only condition for a smooth distribution is a normal return to the closed-door. Set</li>
<li>&#36;X&#36;There are numerous and smooth distributions, which require at least two normal return closures. Set
Even if it's not the only one, we can still calculate the flat distribution, as follows: Example
For seven states of Zithalkov, the one-step transfer probability matrix is
&#36;P=\begin{bmatrix}0.5&amp;0.5&amp;0&amp;0&amp;0&amp;0\0&amp;2/3&amp;1/3&amp;0&amp;0&amp;0&amp;0\1/3&amp;0&amp;2/3&amp;0&amp;0&amp;0&amp;0\0&amp;0&amp;0&amp;0.5&amp;0.5&amp;0&amp;0\0&amp;0&amp;0&amp;0.5&amp;0.5&amp;0&amp;0\0&amp;0&amp;0&amp;0&amp;0&amp;1&amp;0\\frac{1}{7}&amp;\frac{1}{7}&amp;\frac{1}{7}&amp;\frac{1}{7}&amp;\frac{1}{7}&amp;\frac{1}{7}&amp;\frac{1}{7}\end{bmatrix}&#36;&#36;
分解状态空间有
&#36;&#36;\begin{aligned}S&amp;=D\cup C_1^+\cup C_2^+\cup C_3^+\\&amp;={6}\cup{0,1,2}\cup{3,4}\cup{5}\end{aligned}&#36;&#36;
所以有无穷多个平稳分布  使用分块矩阵
&#36;&#36;P_1=\begin{pmatrix}1/2&amp;1/2&amp;0\0&amp;2/3&amp;1\1/3&amp;0&amp;2/3\end{pmatrix}\quad P_2=\begin{pmatrix}1/2&amp;1/2\1/2&amp;1/2\end{pmatrix}\quad P_3=1&#36;&#36;
解方程组有
&#36;&#36;\pi^{(1)}={\frac28,\frac38,\frac38}\quad\pi^{(2)}={\frac12,\frac12}\quad\pi^{(3)}={1}&#36;&#36;
所以平稳分布为
&#36;&#36;{\fnH00FFFF}\fradsymbol{\fnH00FF}\fscH00FF}\fscH00FF}\fscH00FFFF}\fscH00FF}\fscH00FF}\fscH00FF}\fscH00FF}\fscH00FFFF}\fscH00FF}\fsc(fscH00}\fsc(fscH00FF}\fsc(fsc)\fsc(fsc)\fsc(fsc)\fsc(fsc)\fsc(fscH00(fsc)\fsc(sc)\fsc(sc)\fsc(fsc(fsc)\sc(sc)\fsc(sc)\fsc(fsc(fsc)\fsc(fscsc)\fscsc(fscscsc)\f
- There's one.&#36;\lambda_1+\lambda_2+\lambda_3=1&#36;
<strong>And it's easy to calculate on a smooth distribution basis, and we have a lot of good things to use.</strong></li>
</ul>
