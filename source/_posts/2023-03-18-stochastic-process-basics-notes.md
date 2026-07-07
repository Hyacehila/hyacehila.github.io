---
title: "随机过程基础：随机过程定义、数字特征与平稳过程"
title_en: "Stochastic Processes: Definitions, Stationarity, and Markov Foundations"
date: 2023-03-18 21:28:17 +0800
categories: ["Data Science", "Time Series & Spatial Data"]
tags: ["Learning Notes", "Statistics", "Stochastic Processes"]
author: Hyacehila
excerpt: "整理随机过程定义、数字特征、平稳过程、泊松过程、布朗运动和马尔可夫过程基础。"
excerpt_en: "Covers definitions, numerical characteristics, stationary processes, Poisson processes, Brownian motion, and Markov process foundations."
mathjax: true
hidden: true
permalink: '/blog/2023/03/18/stochastic-process-basics-notes/'
---
## 随机过程基本知识
### 随机过程的定义
**如何量化和全面反映一个随机现象**
在概率论和数理统计中 我们引入随机变量和随机向量来刻画一个随机过程
但是他真的能全面的刻画随机过程吗？
如下是一个例子
考察某种振荡器输出波形
$$X_t=A\cos(\omega t+\Phi)$$
其中$A>0$为常数 $w$为常数 随机变量$\Phi$ 服从$[0,2\pi]$ 上的均匀分布 $t\in[0,\infty)$ 
这就是一个随机过程 每一个$t$都对应了一个随机变量 这个随机变量是另一个随机变量的函数
#### 定义的给出
如果给定$t_{0}$, $(0,t0]$时间内该网站的访问次数记为$X_{t0}$是一个随机变量
事实上这个随机变量并不能完整的刻画这个随机现象 我们需要一族随机变量 根据时间$t_{0}$ 的变化而改变的一族随机变量才能完整的刻画这个随机现象
记一族随机变量为 $\{X_t, t∈[0,\infty)\}$ 
因此我们推广得到以下的随机过程定义
$$\begin{aligned}
&\text{设}(\Omega,\mathsf{F},\mathsf{P})\text{为一概率空间},\mathsf{T}\text{为一参数集,Т}\mathsf{R}, \\
&若对每一t\in T,均有定义在(\Omega,F,P)上的一个 \\
&\text{随机变量}\mathsf{X}(\omega,t),(\omega{\in}\Omega)\text{与之对应}, \\
&\text{则称}\mathsf{X}(\omega,t)\text{为}(\Omega,\mathsf{F},\mathsf{P})\text{上的一个随机过程}(\mathsf{S}.\mathsf{P}.)
\end{aligned}$$
我们一般简写为$\{X_t, t∈T\}$  
这个$w$是对随机变量刻画的一个参数 体现了随机变量身上的随机性
参数集$T$ 一般表示时间或者空间 据此随机过程有以下的分类标准
$\mathbf{T=\{0,1,2,...\}}\text{或 T=\{-2,-1,0,1,2,...\}}$ 的时候称为离散参数随机过程（随机时间序列）
$\mathbf{T=[a,b]}$ 的时候 称为连续参数随机过程
#### 进一步解释
* $\color{}{\text{X(ω,t)}}$ 同时具有随机性和函数性  它是一个关于$t$的函数 在$t$确定的时候他是一个随机变量 由于随机变量本质上也是一个函数 所以随机过程的本质是定义在$T\times Q$上的二元单值函数
* 对于每一个固定的$t$ $\color{}{\text{X(ω,t)}}$是一个随机变量 $X_{t} ~t\in T$的所有可能的取值的集合称为随机过程的状态空间$S$ 状态空间的元素称为状态$S$
* 对于一个确定的$\boldsymbol{\omega_0}\in\boldsymbol{\Omega}$ $\color{}{{X(w_0,t)}}$   是一个$T$上的普通函数 称为随机过程的一个样本函数（样本轨道） 他的图形称为样本曲线
#### 样本轨道连续性
有一随机过程$\{X_t, t∈T\}$   如果对于任何$t\in T$ 有
$$\mathbb{P}(\lim_{s\to t}|X_s-X_t|=0)=1$$
则称随机过程$X$在$T$上以概率1连续 此时称随机过程具有连续的样本轨道
**概率1连续是最重要的**
当然连续性的定义并不唯一
如果对于任意$t\in T$ 和常数$\epsilon>0$ 有
$$\lim_{s\to t}\mathrm{P}(\left|X_s-X_t\right|\geq\varepsilon)=0$$
则称为随机过程X在T上依概率连续（随机连续）
若对于任意$t\in T$ 和$P\ge 1$ 有  
$$\mathbb{E}|{X}_t|^p<\infty$$
则称为$L^p$连续 $p=2$的时候称为均方连续
$L^p$连续 则一定随机连续
#### 随机过程的分类
根据**参数集$T$** 和 **状态空间** 的连续与离散进行分类 不难理解
##### 离散参数 离散状态
考察生物群体的增长情况
令$X_n$ 表示第$n$代的生物群体个数
我们需要一族随机变量$X_{n},n=1,2,...$
##### 离散参数 连续状态
考察某地的最高气温
令$X_t$ 表示第$t$次观测的最高气温
我们需要一族随机变量$X_{t},t=1,2,...$
##### 连续参数 离散状态
考察网站的访问次数
对于给定的$t_{0}$ 从0到$t_{0}$ 网站的访问次数$X_{t_{0}}$ 是一个随机变量
我们需要一族随机变量$X_{t_{0}},t\in[0,\infty)$
##### 连续参数 连续状态
考察某种振荡器输出波形
$$X_t=A\cos(\omega t+\Phi)$$
其中$A>0$为常数 $w$为常数 随机变量$\Phi$ 服从$[0,2\pi]$ 上的均匀分布 $t\in[0,\infty)$ 
这就是一个随机过程 每一个$t$都对应了一个随机变量 
这个随机变量是另一个随机变量的函数
#### 一些常用的随机过程
##### 伯努利过程
设随机过程$X_{n},n=1,2,...$ 
如果其中的随机变量$X_i$ 相互独立同分布 并且服从伯努利分布
则称这个随机过程为伯努利过程
伯努利过程描述了一系列独立同分布的随机试验.
**离散参数 离散状态**
##### 二项过程
如果令 $S_n=\sum_{k=1}^nX_k,\quad S_0=0$
明显的 $S_{n},n=1,2,...$  也是一个随机过程
我们称其为二项过程
正如二项分布是01分布的叠加 二项过程也是如此
**离散参数 离散状态**
##### 严高斯白噪声过程
设随机过程$X_{n},n=1,2,...$ 
如果其中的随机变量$X_i$ **相互独立同分布** 
并且服从Gauss分布（正态分布） $N(0,\sigma^2)$
则称$X$为严高斯白噪声过程
**离散参数 连续状态**
##### 计数过程
如果$N_t$表示直到时刻$t$为止发生的某随机事件总数, 则称实随机过程${N_t,t≥0}$为计数过程
一般有 $N_t$ 是非负整数 并且$N_0=0$
**连续参数 离散状态**
##### 正态过程
也称为Gauss过程
设$X= {X_{t},t\in T}$是一实值随机过程,若对任意$n≥1$及
$t_1,t_2,…,t_n∈T$, $n$维随机变量($X_{t_1}, X_{t_2}, …, X_{t_n})$
服从$n$维正态分布,则称X是正态(高斯)过程
参数集$T$的情况决定它是否是连续参数随机过程 不过状态一定是连续的
##### 证明一个随机过程是正态过程
设$Z_t=X+Yt$,$-\infty<t<+\infty$,其中随机变量$X$,$Y$相互独立，且都服从$N(0,\sigma^2)$分布。证明随机过程$Z=\{Z_t,-\infty<t<+\infty\}$是一正态过程
容易知道
$$(X,Y)\sim N_{}(\mu,\sum)$$
是一个二维的正态向量（因为独立性，让两个正态变量构成正态向量）
对于
$$\left.C_{2,n}=\left[\begin{array}{ccccc}1&1&1&\cdots&1\\t_1&t_2&t_3&\cdots&t_n\end{array}\right.\right].$$
有
$$(Z_{t_1},Z_{t_2},\ldots,Z_{t_n})=(X,Y)C_{2,n}.$$
因此我们知道 新的高维向量也是正态向量 证明了原过程是正态过程

设随机变量$R$和$\Theta$相互独立 其中$R$服从瑞利分布 密度为
$$f(r)=\begin{cases}\frac{r}{\sigma^{2}}e^{-\frac{r^{2}}{2\sigma^{2}}},&\quad r\ge0\\0,&\quad r<0\end{cases}$$
$\Theta$服从$[0,2\pi]$上的均匀分布
定义 
$$X_t=R\cos(\Theta+at),\quad t\in R,a\text{为常数}$$
证明他是一个正态过程
**我们可以用一步变形来变成和前面一样的问题**
$$X_{t_{}}=R\cos(\theta+at)=R\cos\theta\cos a t_{}-R\sin\theta\sin\alpha t_{}$$
我们容易看出来 想要证明的向量就是
$$(R\cos\theta,R\sin\theta)\binom{\cos at_1,\cdots,\cos at_n}{\sin at_1,\cdots,\sin at_n}$$
只用证明前面这个向量是正态向量就好了
他是两个随机变量的函数 概率论也给出了证明他的密度函数的方法
验证一下 果然就是正态向量的密度函数 证明完成
### 随机过程的进一步分类
#### 增量性质
##### 正交增量过程
设$X={X_t,t∈T}$是实值随机过程 如果对于任意的$t_{1}{<}t_{2}\leq t_{3}<t_{4}\in\mathbb{T}$ 都有
$$\mathrm{E}\left[\left(\textbf{X}_{t_2}-\textbf{X}_{t_1}\right)\left(\textbf{X}_{t_4}-\textbf{X}_{t_3}\right)\right]=0$$
则称其为一个正交增量过程
这里把随机向量的积视作内积（后面再解释）
##### 独立增量过程
设$X={X_t,t∈T}$是实值随机过程 如果对于任意的$t_{1}{<}t_{2}\leq t_{3}<...<t_{n}\in\mathbb{T}$ 都有
$$X_{t_2}-X_{t_1},X_{t_3}-X_{t_2},\cdots,X_{t_n}-X_{t_{n-1}}$$
是相互独立的随机变量 则称$X$是独立增量过程
##### 平稳增量过程
设$X={X_t,t∈T}$是实值随机过程 如果对于任意的$s<t\in T$ 有
$$X_{t}-X_{s}$$
仅依赖于$t-s$ 则称$X$是平稳增量过程
我们在后面还会介绍一种叫平稳过程的随机过程 和平稳增量过程不一样
**独立增量过程和平稳增量过程都是很重要的观点 他们在后面往往会同时出现**
#### 样本轨道连续性
##### 定义
设 $\xi(\omega)$ ,$\eta{(\omega)}$ 是定义在同一概率空间上的两个随机变量。定义随机过程$\mathrm{X= }\{ X_t{: }\quad t\geq 0\}$为：
$$
X_t=\xi(\omega)+\operatorname{t}\eta(\omega)
$$
则X是一个连续轨道随机过程
按照定义验证反而不多 我们往往只介绍几个最核心的随机过车过
##### 连续轨道随机过程
标准布朗运动（维纳过程）是连续轨道随机过程
若实随机过程W=$\{\mathbf{W}_{t},\mathbf{t}\geq0\}$满足： 
$(1)\quad W_0=0$ 
$(2)\quad W=\{W_t,t\geq0\}$ 是平稳的独立增量过程.
(3) 对任意的$0\leq s<t$,有$W_t-W_s\sim N(0,(t-s))$
则称随机过程W是标准布朗运动(维纳过程)
他是一个连续轨道随机过程 我们可以从他的样本曲线中看出这一点
##### 跳跃轨道随机过程
称随机过程N=$\{N_t,t\geq0\}$是参数为$\lambda$ 的泊松过程，如果它满足以下三条件：
$\begin{pmatrix}1\end{pmatrix}\quad N_0=0$
(2) 对任意的$0\leq s<t$,增量$N_t$ -$N_s$服从参数为$\lambda(t-s)$的泊松分布，即
$$
P(N_t\:-N_s=k)=\frac{\left(\lambda(t-s)\right)^ke^{-\lambda(t-s)}}{k!},k=0,1,2,\cdots 
$$
(3)对任意的$n\geq2$,及$0\leq t_0<t_1<\cdots<t_n<\cdots,n$个增量
$$
N_{t_n}-N_{t_{n-1}},\cdots,N_{t_2}-N_{t_1},N_{t_1}-N_{t_0}
$$
是相互独立的随机变量
他是轨道不连续的随机过程 我们可以从他的样本曲线中看出这一点
**这两个最基础的随机过程有着共同的特点，初始的值为0，是平稳增量过程兼独立增量过程，任意差值分布服从特定的分布**
#### 随机过程的推广
##### 多维随机过程
设$X_{t}~t\in T ~Y_{t} ~t\in T$是定义 在同一概率空间($\Omega$,F,P)上的两个实随机过程. 则称$\{ X_t, Y_t, \mathrm{t\in T}\}$是二维随机过程
##### 复随机过程
设$\{ \mathrm{X_t, t\in T}\}$ 和 $\{ \mathrm{Y_t, t\in T}\}$ 是定义在同一概率空间 $(\Omega,\mathbb{F},\mathbb{P})$上的两个实随机过程.令
$$\color{}{\mathrm{Z_t\color{}{=}X_t\color{}{+}jY_t}\quad\color{}{t\in T}}$$
则称$\{ \mathbb{Z} _t, \mathrm{t\in T}\}$是复随机过程
### 有限维分布函数族
#### 定义
我们在概率论中就已经介绍过分布函数的概念了，现在我们必须将其推广到随机过程中 为了简化问题 我们这里只研究离散并且有限的参数族$T$
设 $X=\{X_t,t\in T\}$是定义在概率空间 ($\Omega,F,P)$上取实值的随机过程.对任意的自然数$n≥1$,及任意的 $t_1,t_2,\cdots,t_n\in T$ 和实数 $x_1,x_2,\cdots,x_n\in R$,称n维随机变量 $(X_{t_1},X_{t_2},\cdots,X_{t_n})$ 的联合分布函数
$$
F_{t_1,\cdots,t_n}\left(x_1,\cdots,x_n\right.)=P(X_{t_l}\leq x_1,\cdots,X_{t_n}\leq x_n),
$$
为随机过程X的n维分布函数
将随机过程X的所有有限维分布函数的全体记为
$$
\mathbf{F}=\{F_{t_1,\cdots,t_n}(x_1,\cdots,x_n):\:\forall n\in\mathbf{N},\:t_1,t_2,\cdots,t_n\in T\text{,}x_1,x_2,\cdots,x_n\in R\}
$$
则称函数集F为随机过程X的n维分布函数族
形成族是因为时间$t$的不确定性 更是因为连$n$的取值都不能确定
因为哪怕$t$不确定我们也可以视为常数进行运算 但是$n$的变化就影响了整个函数的形式 因此我们后面会见到$n$维的分布函数的计算
#### 解释
对 ($1,2,\cdots,n$)的任一个排列 ($k_l,\cdots,k_n$),有 
$$
F_{t_1,\cdots,t_n}\left(x_1,\cdots,x_n\right)=F_{t_{k_1},\cdots,t_{k_n}}\left(x_{k_1},\cdots,x_{k_n}\right)
$$
若$m<n$ 有
$$F_{t_1,\cdots,t_m}(x_1,\cdots,x_m)=F_{t_1,\cdots,t_m,t_{m+1},\cdots,t_n}(x_1,\cdots,x_m,+\infty,\cdots,+\infty)$$
#### 例子
##### 一
$X{=}\{X_t=Vcos\omega t,t\in R\}$其中 $\omega$ 为常数， 随机变量V服从$[0,1]$上的均匀分布.分别计算当
$$
t=\frac{3\pi}{4\omega}\text{和}t=\frac\pi{2\omega}
$$
 时，随机过程X的一维分布函数
 当$t=\frac{3\pi}{4\omega}$  $X_{_t}=V\cos\omega\frac{3\pi}{4\omega}=-\frac{\sqrt{2}}{2}V$ 
 我们需要研究的分布是已知分布的函数 我们在概率论中也介绍过了 故
 $X_{t}$的密度函数为$$f(x)=\begin{cases}\sqrt{2},&-\frac{\sqrt{2}}{2}\leq x\leq0\\0,&\text{其它}\end{cases}$$一维分布函数就是积分计算问题了
$${F}_{\frac{3\pi}{4\omega}}(x)=\int_{-\infty}^xf_{X_{\frac{3\pi}{4\omega}}}(t)dt$$
具体结果没必要继续写了 非常的基础
$t=\frac{\pi}{2\omega},X_{\frac{\pi}{2\omega}}=V\cos\omega\times\frac{\pi}{2\omega}=0$
密度函数应该就是定义域内一直为0 自然其一维的分布函数也是一直为0 吗？
 
##### 二
离散型也可以用一样的思路做
设随机过程$X=\{X_t=Acost,t\geq0\}$,其中随机变量A有分布律
$$
P(A=i)=\frac13,\quad i=1,2,3.
$$
(1)随机过程X的一维分布函数${F_{\frac\pi2}}(x)$
(2)随机过程X的二维分布函数$F_{0,\frac\pi3}(x_1,x_2)$
使用前面的公式计算 
$$X_{\frac\pi4}=A\cos\frac\pi4=\frac{\sqrt2}2A,$$
由于是离散型 函数后的分布反而更好求了 我们直接计算一维分布函数有
$$\mathrm{F}_{\frac{\pi}{4}}(x)=\begin{cases}0,&\quad x<\frac{\sqrt{2}}{2}\\\frac{1}{3},&\quad\frac{\sqrt{2}}{2}\leq x<\sqrt{2}\\\frac{2}{3},&\quad\sqrt{2}\leq x<\frac{3}{2}\sqrt{2}\\1,&\quad x\geq\frac{3}{2}\sqrt{2}\end{cases}$$
这里的计算不需要使用公式 A是离散型的 依次代入$A$的取值情况就可以了；只有定义域会发生变化 他的取值情况会是稳定的 或者我们直接用$\frac{\sqrt2}2A<x$ 然后变形为$A<\sqrt2x$  类比标准的分布函数定义$A<x$ 就可以了 

同理 我们来研究一下二维的情况 要坚持落实定义
$$\begin{aligned}F_{0,\frac\pi3}(x_1,x_2)&=P(X_0\leq x_1,X_{\left.\frac\pi3\leq x_2\right)}\\&=P(A\leq x_1,\frac A2\leq x_2)\\&=P(A\leq x_1,A\leq2x_2)\end{aligned}$$
我们能用两者独立然后单独计算然后乘起来吗？
不能 明显$A$不可能和自己独立 
分类讨论
$$=\begin{cases}P(A\le x_1)\quad x_1\le2x_2\\P(A\le2x_2)\quad x_1>2x_2\end{cases}$$
这时候明显左边的分布列是已知的 非常自然的计算就好了
分别给出分布函数结果
$$\begin{aligned}P(A\leq x_1)=\begin{cases}0,&\quad x_1<1\\\frac13,&\quad1\leq x_1<2\\\frac23,&\quad2\leq x_1<3\\1,&\quad x_1\geq3&\end{cases}\quad P(A\leq2x_2)=\begin{cases}0,&\quad2x_2<1\\\frac13,&\quad1\leq2x_2<2\\\frac23,&\quad2\leq2x_2<3\\1,&\quad2x_2\geq3&\end{cases}\end{aligned}$$

### 有限维特征函数
#### 定义
我们在概率论中就已经介绍过特征函数的概念了，现在我们必须将其推广到随机过程中
**建议复习概率论特征函数部分，并且尽量完全理解，复习各个分布的特征函数**
下面是随机过程$n$维特征函数的定义
$$
\varphi_{t_1,t_2,...,t_n}(u_1,u_2,...,u_n)\quad=\operatorname{E}[e^{j(u_1X_{t_1}+\cdots+u_nX_{t_n})}]\quad\forall ^{}u_{1},u_{2},...,u_{n}\in R 
$$
研究存在的维数不确定性 就有有限维分布函数族
$$
\Phi=\{\varphi_{t_1,t_2,...,t_n}(u_1,u_2,...,u_n),t_{i}\in {T},u_{i}\in {R},i=1,2,\cdots,n\}
$$
#### 例子
**我们只研究这一个例子并尝试理解其思想**
若实随机过程W=$\{\mathbf{W}_{t},\mathbf{t}\geq0\}$满足： 
$(1)\quad W_0=0$ 
$(2)\quad W=\{W_t,t\geq0\}$ 是平稳的独立增量过程.
(3) 对任意的$0\leq s<t$,有$W_t-W_s\sim N(0,(t-s))$
则称随机过程W是标准布朗运动(维纳过程) 
我们来计算他的有限维特征函数
根据定义知道 他的特征函数是
$$\varphi_{t_1,\cdots,t_n}(u_1,\cdots,u_n)=\mathrm{E}[e^{j(u_1W_{t_1}+\cdots+u_nW_{t_n})}]$$
令
$$\begin{aligned}Y_1=W_{t_1},Y_2=W_{t_2}-W_{t_1},\cdots,Y_n=W_{t_n}-W_{t_{n-1}}\end{aligned}$$
明显的 我们的$Y$是过程增量 容易知道增量具有独立性 也就是所有的$Y$独立
反解$W_t$带入特征函数的计算
$$\varphi_{t_1,\cdots,t_n}(u_1,\cdots,u_n)=\mathrm{E}[e^{j[u_1Y_1+u_2(Y_1+Y_2)+\cdots+u_n(Y_1+\cdots+Y_n)]}]$$
整理成每一个增量的形式
$$\operatorname{E}[e^{j[(u_{1}+u_{2}+\cdots+u_{n})Y_{1}+(u_{2}+\cdots+u_{n})Y_{2}+\cdots+u_{n}Y_{n}]}]$$
根据增量的独立性有
$$=\operatorname{E}[e^{j[(u_1+u_2+\cdots+u_n)Y_1}]\operatorname{E}[e^{(u_2+\cdots+u_n)Y_2}]\cdotp\cdotp\cdotp\operatorname{E}[e^{u_nY_n}]$$
能看出 每一个均值部分都是一个特征函数
$$=\varphi_{Y_1}(u_1+u_2+\cdotp\cdotp\cdotp+u_n)\varphi_{Y_2}(u_2+u_3+\cdotp\cdotp\cdotp+u_n)\cdotp\cdotp\cdotp\cdotp\varphi_{Y_n}(u_n)$$
又因为我们知道 事实上每个增量的分布都是正态分布 他的特征函数是容易计算的
$$\begin{aligned}&\varphi_{Y_1}(u_1+\cdots+u_n)=e^{-\frac12(u_1+\cdots+u_n)^2t_1}\\\\&\varphi_{Y_k}(u_k+\cdots+u_n)=e^{-\frac12(u_k+\cdots+u_n)^2(t_k-t_{k-1})}\end{aligned}$$
把它带回就是我们想计算的特征函数
$$e^{-\frac12(u_1+\cdots+u_n)^2t_1}\cdot e^{-\frac12(u_2+\cdots+u_n)^2(t_2-t_1)}\cdot\cdots\cdot e^{-\frac12(u_n)^2(t_n-t_{n-1})}$$
**根据这个例子我们可以外推出一般性的结论：独立增量过程的有限维分布函数由其一维分布函数（$Y_1$）和增量分布函数($Y_n,n\ne1$)确定**
**我们只要知道了这是独立增量过程，就可以模仿上面的思路计算特征函数**
### 数字特征
#### 随机过程的数字特征
##### 均值函数
设 $X=\{X_t,t\in\mathbb{T}\}$ 是一实值随机过程，若对任意$t{\in }$T,有
$$\operatorname{E}[X_t]\text{存在}$$
则称 ${E}[X_t]$为随机过程X的均值函数，记为$m_x(t).$ 
##### 方差函数
设 $X=\{X_t,t\in\mathbb{T}\}$ 是一实值随机过程，若对任意$t{\in }$T,有
$$\mathbb{E}[X_t-m_X(t)]^2\text{ 存在}$$
则称 ${E}[X_t]$为随机过程X的方差函数，记为$D_x(t).$ 
##### 协方差函数
设 $X=\{X_t,t\in\mathbb{T}\}$ 是一实值随机过程，若对任意$t,s{\in }$T,有
$$Co\nu(X_s,X_t)=\operatorname{E}\left[(X_s-m_X(s))(X_t-m_X(t))\right]\text{存在}$$
则称之为随机过程X的协方差函数.记为$C_X(s,t).$
##### 相关函数
设 $X=\{X_t,t\in\mathbb{T}\}$ 是一实值随机过程，若对任意$t,s{\in }$T,有
$$
\operatorname{E}[X_sX_t]\text{存在}
$$
则称之为随机过程X的(自)相关函数.记为$R_X(s,t).$
##### 均方值函数
设 $X=\{X_t,t\in\mathbb{T}\}$ 是一实值随机过程，若对任意$t,s{\in }$T,有
$$\mathbb{E}[X_t]^2\text{ 存在}$$
则称之为随机过程X的均方值函数.记为$\Phi_X(t).$
##### 随机过程数字特征的关系
$$\begin{aligned}
&C(s,t)=R_X(s,t)-m_X(s)m_X(t) \\
&D_{X}(t)=C_{X}(t,t) \\
&\Phi_{X}(t)=R_{X}(t,t)
\end{aligned}$$
**我们的计算习惯是计算均值函数$m(x)$和相关函数$R(x,t)$ 从这个为基础导出协方差函数再导出方差函数**
#### 两个随机过程的数字特征
设$\{\mathcal{X}_t,\mathcal{Y}_t,t\in\mathcal{T}\}$为二维随机过程，对任意$s,t\in T$,
若${E[X_{s}Y_{t}]}$ 存在
则称之为该二维随机过程的互相关函数，记 $R_{X Y}(s,t).$ 
若 $Co\nu(X_s,Y_t)={E}[(X_s-m_x(s))(Y_t-m_y(t))]$ 存在
则称之为该二维随机过程的互协方差函数，记$C_{XY}(s,t).$
互协方差函数可以定义两个过程的相关性 当其为0的时候认为两个随机过程不相关
#### 复随机过程的数字特征
均值 方差 函数定义不变
均方值函数定义修正为 $\Phi_Z(t)=\mathbb{E}\left|Z_t\right|^2$增加了一层复实转换
相关函数和协方差函数分别修正为$$R_{z}(s,t)=\mathrm{E}[\bar{Z}_{s}Z_{t}]$$$$\begin{aligned}C_Z(s,t)&=\mathbb{E}[(\overline{Z_s-m_Z(s)})(\mathbb{Z}_t-m_Z(t))]\end{aligned}$$因此我们修正原本的性质为
$$\begin{aligned}
&m_{Z}(t)=m_{X}(t)+jm_{Y}(t),t\in T \\
&D_{Z}(t)=D_{X}(t)+D_{Y}(t),t\in T \\
&C_{Z}(s,t)=R_{Z}(s,t)-\overline{m_{Z}(s)}m_{Z}(t),t\in T
\end{aligned}$$
## 布朗运动
事实上随机过程是一个非常复杂的东西 很多内容我们都难以定性研究 因此在随后的三章里面我们会分别介绍四种比较简单并且研究的比较充分的随机过程 本章我们先来研究布朗运动
### 标准布朗运动
#### 定义
若实随机过程W=$\{\mathbf{W}_{t},\mathbf{t}\geq0\}$满足： 
$(1)\quad W_0=0$ 
$(2)\quad W=\{W_t,t\geq0\}$ 是平稳的独立增量过程.
(3) 对任意的$0\leq s<t$,有$W_t-W_s\sim N(0,(t-s))$
则称随机过程W是标准布朗运动(维纳过程) 
去除掉第一条初值为0 则称为布朗运动
#### 有限维分布函数
我们在前面的一节里面计算标准布朗运动的特征函数
它是一堆正态分布特征函数的积 有
$$\varphi_{\iota_1,\cdots,\ell_n}\left(u_1,\cdots,u_n\right.)=\prod_{k=1}^ne^{-\frac12(u_k+\cdots+u_n)^2(t_k-t_{k-1})}=e^{-\frac12\sum_{k=1}^n(u_k+\cdots+u_n)^2(t_k-t_{k-1})}$$
我们可以使用完全类似的思路研究标准布朗运动的分布函数（特征函数前面已经研究过了，并且给出了方法）
##### 一维
能注意到$W_{t}=W_{t}-W_{0}\sim N(0,t)$ 
他的分布函数是
$$F_{t}(x)=\frac{1}{\sqrt{2\pi t_{1}}}\int_{-\infty}^{x}e^{-\frac{x^{2}}{2t_{1}}}\mathrm{d}x,x\in\mathbb{R}$$
就是正态分布的分布函数
##### 二维
根据定义我们知道
$$\begin{aligned}
&F_{t_{1},t_{2}}(x_{1},x_{2})=P(W_{t_{1}}\leq x_{1},W_{t_{2}}\leq x_{2}) \\
&=P(W_{t_1}\leq x_1,W_{t_1}+(W_{t_2}-W_{t_1})\leq x_2),\\&\quad\text{记}\xi=W_{t_1},\quad\eta=W_{t_2}-W_{t_1} \\
&=P(\xi\leq x_{1},\xi+\eta\leq x_{2})
\end{aligned}$$
还是使用前面的思路构造增量函数
我们知道两个增量是独立并且服从正态分布
$$\begin{aligned}&\eta=W_{t_2}-W_{t_1}\sim N(0,t_2-t_1)\\&\xi=W_{t_1}-W_0\sim N(0,t_1),\end{aligned}$$
但是此时我们的联合分布并不是独立的 还是需要使用基础的定义来进行求解 研究联合分布的密度函数
$$\begin{aligned}
&=\int_{-\infty}^{x_1}P(\eta\leq x_2-\xi|\xi\in(y,y+\mathrm{d}y)P(\xi\in(y,y+\mathrm{d}y)) \\
&=\int_{-\infty}^{x_{1}}\int_{-\infty}^{x_{2}-y}[f_{\eta}(z)\mathrm{d}z]g_{\xi}(y)\mathrm{d}y
\end{aligned}$$
##### $n$维
我们选择借助正态过程来进行求解
容易知道 增量是独立的正态变量 也就是
$$(\begin{array}{c}W_{t_1},W_{t_2}-W_{t_1},\cdots,W_{t_n}-W_{t_n-1}\\\end{array})$$
是一个正态向量
$$(W_{t_{1}},W_{t_{2}},\cdots,W_{t_{n}})$$
就是正态向量的某种组合  所以他也是正态向量
那么我们能看出 标准布朗运动是正态过程；直接使用定义就可以计算他的$n$维密度函数  
$$\frac1{(2\pi)^{\frac n2}\Bigg(\prod_{k=1}^n(t_k-t_{k-1})\Bigg)^{\frac12}}e^{-\frac12\sum_{k=1}^n\frac{(w_k-w_{k-1})^2}{(t_k-t_{k-1})}}$$
模仿二维的计算方式通过确定高维分布的思路也是可以完成的
#### 数字特征
##### 均值
由于标准布朗运动是正态过程 并且容易知道
$$W_{t}=W_{t}-W_{0}\sim N(0,t)$$
所以有
$$m_{_{W}}(t)=0$$
##### 方差
同上有
$$D_W(t)=t$$
##### 相关系数
还是根据定义研究有
$$\begin{aligned}
R_{W}(s,t)& =\operatorname{E}[W_{s}W_{t}]  \\
&=\mathrm{E}[(W_{s}-W_{0})(W_{t}-W_{s}+W_{s})] \\
&=\mathrm{E}[(W_s-W_0)(W_t-W_s)]+\mathrm{E}[W_s]^2 \\
&=0+\operatorname{E}[W_s]^2 \\
&=D[W_{s}]+\left(\operatorname{E}[W_{s}]\right)^{2} \\
&=s
\end{aligned}$$
因此
$${R}_{\mathrm{w}}(s,t)=\min(s,t)$$
##### 协方差
使用辅助公式研究
$$C_{W}(s,t)=R_{W}(s,t)-m_{W}(s)m_{W}(\mathrm{t})=\min(s,t)$$

### 布朗运动的性质
当$\mathbf{W}=\{\mathbf{W}_{t},t\geq0\}$ 是标准布朗运动 则
#### 对称性
$$\mathbf{-W}\mathbf{=\{-W_{t},t\geq0\}}$$
是标准布朗运动
#### 自相似性
对于任意的$a>0$ 有
$$\mathbf{W_{at}}\doteq\mathbf{a^{1/2}}\mathbf{W_{t}}$$
是标准布朗运动
#### 时间逆转性
对于固定的$T>0$ 有
$$\mathbf{B_{t}}=\mathbf{W_{T}}-\mathbf{W_{T-t}}\quad\mathbf{0\leq t\leq T}$$
是标准布朗运动
#### 样本轨道性质
标准布朗运动的样本轨道是**连续的**并且是**处处不可微**的
## 泊松过程
原本我们这一章应该主讲跳跃随机过程（样本轨道不连续），正如前一章主讲连续随机过程一样，但是在上一章我们着重研究了连续随机过程中的标准布朗运动（维纳过程） 这一章我们选择主要研究泊松过程 从而降低难度
### 计数过程
如果$N_t$表示直到时刻$t$为止发生的某随机事件总数, 
则称实随机过程${N_t,t≥0}$为计数过程
很明显的 计数过程在现实世界中非常的广泛 比如通过的车辆数 数据包的传输等等过程都属于计数过程的范畴
容易知道计数过程应该有以下的性质
* $\forall t,N_t\geq0\text{,}N_0=0$
* $N_{t}\text{取非负整数}$
* $\forall0\leq s<t,N_t\geq N_s$
* $\forall0\leq s<t,N_t-N_s$ 表示时间段内发生的次数总数
如果有 $N=\{N_{t},t\geq0\}$ 是计数过程 
则有
$$\mathrm{T_n=inf}\{t:N_t{=}\boldsymbol{n}\}$$
则称随机序列 $T_1,T_2,\cdots,T_n\cdots$为计数过程的到达时间序列
有
$$\tau_n=T_n-T_{n-1},n=1,2,...$$
称为计数过程到达时间的间隔序列 
明显的$T_{n}=\sum_{k=1}^{n}\tau_{k},n=1,2,\cdots$
明显的 任何一个计数过程都对应了三组随机变量 $N_t,T_n,\tau_n$
他们都是一个随机过程 研究他们的分布并不是一个容易的事情 以后有机会再考虑
### 泊松过程基本定义与性质
#### 泊松过程的定义
事实上 泊松过程是一类特殊的计数过程 
如果计数过程${N_t,t≥0}$ 满足
* $\quad N_{0}=0$ 
* 对任意的$n\geq2$及$0\leq t_0<t_1<\cdots<t_n$, 增量$N_{t_n}-N_{t_{n-1}},\cdots,N_{t_1}-N_{t_0}$相互独立
* 对任意的$0\leq s<t$,增量$N_t-N_s$服从参数为$\lambda(t-s)$的泊松分布：$P(N_t-N_s=k)=\frac{(\lambda(t-s))^ke^{-\lambda(t-s)}}{k!},k=0,1,2,\cdots$
则称这个计数过程为参数$\lambda$的泊松过程
还是我们前面提到过的那个定义
#### 泊松过程的一维分布与多维分布
一维分布是很好研究的 根据增量的性质能得到
$$\begin{aligned}\mathrm{P}(N_t=k)&=\mathrm{P}(N_t-N_0=k)\\&=\frac{(\lambda t)^ke^{-\lambda t}}{k!}\end{aligned}$$
这是密度函数（分布列）  当然累加计算$F(x)$也是可以的
#### 泊松过程的数字特征
##### 均值函数
研究前面给出的一维分布有
$$m_N(t){=}\lambda t$$
##### 方差函数
根据一维分布就是泊松分布也能得到 方差应该和均值一样
$$D_{N}(t){=}\lambda t$$
##### 相关函数
还是得根据增量的独立性进行变形
$$R_N(\mathrm{s,t}){=}\mathrm{E}[N_sN_t]$$
借助增量独立性变形$$=\mathrm{E}[(N_s-N_0)(N_t-N_s+N_s)]$$拆分化简
$$=\mathrm{E}[(N_s-N_0)(N_t-N_s)]+\mathrm{E}[N_s^2]$$
由于增量是独立的  二阶矩可以表示为方差和一阶矩平方的和
$$=\operatorname{E}[N_{s}]\operatorname{E}[N_{t}-N_{s}]+\operatorname{D}_{N}(s)+(m_{N}(s))^{2}$$
带入化简有
$$\begin{aligned}&=\lambda^2st+\lambda s\\&=\lambda^2st+\lambda\min(s,t)\end{aligned}$$
#### 泊松过程的样本轨道
泊松过程的样本轨道是跳跃的 右连续的 
我们依旧不在这里给出证明了
*确定了泊松分布中的$\lambda$ 整个过程就确定了，只比维纳过程多了一个参数* 
#### 判定定理
我们知道 **泊松过程是一类特殊的计数过程** 下面我们来研究 除了定义以外 如何证明一个计数过程是泊松过程
**定理**
如果计数过程$N=\{N_t,\quad t\geq0\}$的到达时间间隔序列$\{\tau_n,n=1,2,\cdots\}$是独立的、同服从参数为$\lambda>0$ 的指数分布
则该计数过程一定是参数为$\lambda$ 的泊松过程
**辅助定理**
如果随机变量序列$\{\tau_n,n=1,2,\cdots\}$独立同服从参数为$\lambda>0$ 的指数分布
$$
{T}_n=\sum_{k=1}^n\tau_k,\:n=1,2,\cdots 
$$
则T$_{n}$服从参数为(n,$\lambda$)的伽玛分布$\Gamma(n,\lambda)$,密度函数为
$$f_{T_{n}}(x)=\begin{cases}\dfrac{\lambda^{n}}{(n-1)!}x^{n-1}e^{-\lambda x},&\quad x\geq0\\0,&\quad x<0\end{cases}$$
我们就是借助这个辅助定理 从到达时间间隔序列计算到达时间序列 从到达时间序列入手研究$N_t$
**定理**
这是前面判定定理的逆定理
设${N}=\{N_t,t{\geqslant}0\}$是参数为$\lambda$的泊松过程， 则N的到达时间间隔序列 $\tau_1,\tau_2,\cdots,\tau_n\cdots$相互独立同服从参数为$\lambda$的指数分布
#### 补例
两个独立的泊松过程之和仍然是泊松过程

### 泊松过程的等价定义
#### 等价定义
**定义**
如果满足条件
* $N_0=0$ 
* $N$是独立增量过程和平稳增量过程
* ${P}\{N_{t+h}-N_{t}=0\}=1-\lambda h+\circ(h)$
* ${P}\{N_{t+h}-N_t=1\}=\lambda h+\circ(h)$
称计数过程$N={Nt,t≥0}$是参数为$\lambda$ 的泊松过程
**定理**
设$N={Nt,t≥0}$是参数为$\lambda$ 的泊松过程，则一定有
$$\begin{aligned}1)\quad&\mathbf{P}\{N_{t+h}-N_t=0\}=1-\lambda h+\circ(h)\\2)\quad&\mathbf{P}\{N_{t+h}-N_t=1\}=\lambda h+\circ(h)\end{aligned}$$
称为泊松过程的 0-1律
**直观解释：在充分小的时间内，随机事件要么出现1次，要么不出现**
证明了这个等价定义确实有等价的合理性
**定理**
如果计数过程$N_t$具有平稳独立增量性,且满足0-1律  则该计数过程一定是参数为$\lambda$的泊松过程
#### 泊松过程中到达时间的条件分布
设 $\{N_t,t\geq0\}$ 是参数为$\lambda$ 的泊松过程，已知在$[0, t)$内仅有一个随机点到达，$T_1$是其到达时间，则该随机点的到达时间$T_{1}$服从怎样的概率分布？
**结论  $N_t=1$的条件下，第一个随机点的到达时间$T_1$，服从$[0,t]$上的均匀分布**
$$P(T_1\leq s\big|N_t=1)=\frac{P(T_1\leq s,N_t=1)}{P(N_t=1)}$$
$$\begin{aligned}&=P\{N_s=1,N_t-N_s=0\}/P(N_t=1)\\\\&=\lambda se^{-\lambda s}\cdot e^{-\lambda(t-s)}/\lambda te^{-\lambda t}=\frac{s}{t}\end{aligned}$$
我们来考虑更一般的问题
设 $\{N_t,t\geq0\}$ 是参数为λ 的泊松过程，若已知在$[0, t)$ 内仅有$n$个随机点到达，则随机点的$n$个到达时刻${T}_1<{T}_2<...<{T}_n$服从怎样的概率分布？
$$p(u_1,u_2,\cdots,u_n)=\begin{cases}\frac{n!}{t^n},&0<u_1<u_2<\cdots<u_n\leq t\\0,&\text{其它}&\end{cases}$$
#### 两个例子
##### 一
到达某车站的顾客数是一泊松过程，平均每10分钟到达5位顾客，试计算在20分钟内至少有10位顾客到达车站的概率
解：令$N_t$表示$[0,t)$内到达车站的顾客数，则$\{N_{t},t\geq0\}$ 是泊松过程
参数为 $\lambda=\frac{5}{10}=0.5$
则20分钟内至少有10位顾客到达车站的概率
$$P(N_{20}\geq10)=1-\sum_{k=0}^9\frac{(10)^ke^{-10}}{k!}=0.5402$$
##### 二
某机械装置在$[0,t)$内发生的震动次数N，是强度为5次/小时的泊松过程，且当第100次震动发生时，此机械装置发生故障 计算
* 寿命的概率密度函数
* 该装置的平均寿命
* 两次震动时间间隔的概率密度函数
* 相邻两次震动的平均时间间隔
我们知道寿命和泊松过程到达100有关  所以有
* 寿命为到达时间到100的函数 $T_{100}$
* 平均寿命求寿命的期望就有 $E[T_{100}]=20$
* 时间间隔序列 也是泊松分布
* 求时间间隔序列的均值

### 与泊松过程相关的随机过程
#### 复合泊松过程
设N$=\{N_t,t\geq0\}$ 是参数为$\lambda$ 的泊松过程，$\{Y_{k}.k=1,2,...\}$是一列独立同分布的随机变量， 且与$N$独立$令X_t=\sum^{N_t}_{k=1}$ $Y_{k},t\geq0,\quad X_{0}=0$ 
称 $X=$ $\{X_t,t\geq0\}$为复合泊松过程

设随机变量${Y}_n(n{=}1,2,...)$ 数学期望${E}Y_n=\mu$ 方差$DY_n=\sigma^2$,求复合泊松过程
$$X_{t}=\sum_{k=1}^{N_{t}}Y_{k},t\geq0$$
的均值函数、方差函数和相关函数 
## 平稳过程
这又是一类特殊的随机过程 从名字上就可以猜测到 他们的统计特性不随时间的推移而改变 如此特殊的性质值得我们进行研究
### 平稳过程的定义
#### 严平稳过程
如果任取的$t~\tau~x$ 有
$$F_{t_{1},\cdots,t_{n}}(x_{1},\cdots,x_{n})=F_{t_{1}+\tau,\cdots,t_{n}+\tau}(x_{1},\cdots,x_{n})$$
也就是有限维分布函数随时间的推移不发生任何变化 
这是一个比较苛刻的要求

下面是一个严平稳过程的例子
设 $N=\{N,:t\geqslant0\}$是参数为 $\lambda>0$ 的泊松过程，对任意固定的常数 $a>0$,
$$
X_{t}=N_{t+a}-N_{t},\quad\quad\text{其中 }t\geqslant0
$$
这就是一个严平稳的过程

这是因为泊松分布的平稳增量性和独立增量性 可以知道随机变量$X_t$都是服从独立的泊松分布 从定义入手证明了严平稳

简单介绍一些严平稳过程的性质
设 $X=\{X_t:t\in\mathbb{T}\}$是一个严平稳过程，且 X 的二阶矩存在，则对任意的$t,t_1,t_2\in\mathbb{T}$,X的均值函数$m_x(t)$是常数，相关函数$R_x(t_1,t_2)$仅依赖于时间指标差$t_2-t_1$
#### 宽平稳过程
严平稳的要求还是太苛刻了 因此我们放宽了定义 给出了宽平稳过程

设 $X=\{X_t:t\in\mathbb{T}\}$是一个可能取复值的二阶矩过程。我们称 $X$ 是一个宽平稳过程，如果对任意的时间指标 s, $t\in\mathbb{T}$
1. X 的均值函数 $m_{x}(t)\equiv C$(其中 $C\in\mathbb{C}$ 是某个常数)
2. $X$ 的相关函数$R_x(s,t)=R_x(t-s)$,也就是相关函数$R_x(s,t)$的值仅依赖于时间指标差 $t-s$ 或者表示为$R_X(t,t+\tau)=R_X(\tau)$
宽平稳过程仅仅使用数字特征来定义 更容易验证 

事实上 宽平稳过程比严平稳过程的使用广泛的多 后面再不加限定的适用平稳过程的概念 都是指宽平稳过程
#### 例子
设$\mathrm{X}_{\mathrm{t}}=A\cos(\omega t+\Phi),A,\omega$是常数，$\Phi$为随机变量,服从$[0,2\pi]$均匀分布，则称$\mathrm{X= \{ X_t, \quad t\geq 0\} }$为随机初相信号;
 $m_{x}( t) = \mathbb{E} [ X_{\mathrm{t} }]=\int_{0}^{2\pi}\frac{1}{2\pi}A\cos(\omega t+\varphi)d\varphi{=}0$
 $$\begin{aligned}
R_{X}(s,t)& ={E}[{X}_{s}{X}_{\mathrm{t}}]  \\
&=\int_{0}^{2\pi}\frac{1}{2\pi}A^{2}\cos(\omega s+\varphi)\cos(\omega t+\varphi)d\varphi  \\
&=\frac{A^{2}}{2}\cos\omega(t-s).s,t\geq0
\end{aligned}$$
因此这是一个平稳过程

设$X=\{X_t,t\geq0\}$是只取$+{-}1$ 两个值的随机过程，其符号的改变次数是一参数为$\lambda$的泊松过程$N={N}_{t}$, 且对任意的$t$ 
$$P(X_{t}=-1)=P(X_{t}=1)=1/2$$
则称$X$为随机电报信号过程.验证$X$是平稳过程
$$m_X(t)=0$$
$$R_x(t,t+\tau)=\mathbb{E}[X_tX_{t+\tau}]$$
$$=\sum_{k=0}^{\infty}\frac{(\lambda|\tau|)^{2k}}{(2k)!}e^{-\lambda|\tau|}-\sum_{k=0}^{\infty}\frac{(\lambda|\tau|)^{2k+1}}{(2k+1)!}e^{-\lambda|\tau|}$$
$$=e^{-2\lambda|\tau|}$$


#### 宽平稳过程和严平稳过程的联系
从定义能够看出 严平稳过程不一定直接就是宽平稳过程  反之也是这样的
但是我们在严平稳过程里面介绍过性质
**二阶矩存在严平稳过程满足宽平稳过程的定义**
下面我们来反过来研究一下 看看宽平稳过程什么时候是严平稳过程
**宽平稳的正态过程一定是严平稳过程**
关于正态过程前面介绍过了 这里我们用例子来说明这个定理
设$Y=\{Y_t,t\geq0\}$是正态过程.且$m_{Y}(t)=\alpha+\beta t,\quad C_{Y}(t,t+\tau)=e^{-a|\tau|}$,其中$\alpha,\quad\beta,\quad a>0$, 令
$$
X_t=Y_{t+b}-Y_t,t\geq0,\text{其中}b>0,
$$
试证明$\mathbf{X}=\{X_t,t\geq0\}$是一严平稳过程
均值函数
$$m_{X}(t)=\operatorname{E}[X_{t}]=\operatorname{E}[\color{}{Y_{t+b}-Y_{t}}]=\beta{b}$$
协方差函数
$$\begin{aligned}C_X(t,t+\tau)&=\mathrm{cov}(X_t,X_{t+\tau})\\&=\mathrm{cov}(\color{}{Y_{t+b}-Y_{t},Y_{t+\tau+b}-Y_{t+\tau}})\end{aligned}$$
$$=2e^{-a|\tau|}-e^{-a|\tau-b|}-e^{-a|\tau+b|}$$
协方差函数
$$R_{\chi}(t,t+\tau)=2e^{-a|\tau|}-e^{-a|\tau-b|}-e^{-a|\tau+b|}+\beta^{2}b^{2}$$
保证了是严平稳过程 下面考虑是不是正态过程
我们可以非常容易的用一个矩阵从原本的正态向量变形得到现在的向量
所以是正态过程 
综上 我们证明了是严平稳过程 其他问题的证明思路是一样的
### 相关函数
均值函数和相关函数反映了随机过程的统计特性 由于平稳过程的均值函数是常数 因此他的主要特性要从相关函数来刻画 这是这节要考虑的问题
#### 相关函数的性质
设$X=\{X_t:t\in\mathbb{T}\}$为复平稳过程，则 $X$ 的相关函数 $R_X(\tau)$具有以下性质
* $R_{X}(0)=\mathbb{E}[\mid X_{\iota}\mid^{2}]\geqslant0,\quad t\in\mathbb{T}$
* $\mid R_{X}(\tau)\mid\leqslant R_{X}(0)$
* $\overline{R_{X}(\tau)}=R_{X}(-\tau)$
* $\sum_{k=1}^n\sum_{l=1}^n\bar{\alpha}_k\alpha_lR_X(t_k-t_l)\geqslant0$

容易得到 平稳过程协方差函数$C_X(\tau)$ 满足
* $C_X(0)=\mathcal{D}_X(t)\ge0;$
* $|C_X(\tau)|\leq C_X(0)$

对于存在周期的平稳过程$X$ 也就是存在$T_0$  $X_{t+T_0}=X_t$ 此时
$$\begin{aligned}
R_{X}(\tau+T_{0})& =\mathrm{E}\left[\overline{X}_{t}X_{t+\tau+T_{0}}\right]  \\
&=\mathbf{E}\left[\overline{X}_{t}X_{t+\tau}\right] \\
&=R_{X}(\tau)
\end{aligned}$$
相关函数也是周期性的

能看出 相关函数$R_X(\tau)$ 反应了平稳过程$X$的两个随机变量线性相关程度的大小 我们可以用相关函数研究平稳过程的一些特性
在工程实践上 对于无周期的平稳过程 我们一般认为
$$\begin{aligned}
\operatorname*{lim}_{\tau\rightarrow\infty}R_{X}\left(\tau\right)& =\lim_{r\rightarrow\infty}E\left[X_{t}X_{t+r}\right]  \\
&=\lim_{r\to\infty}\langle\mathbf{E}\left[X,\right]\mathbf{E}\left[X_{t+\tau}\right]\rangle  \\
&=m_{X}^{2}\geqslant0
\end{aligned}$$
也就是趋于无关

为了消除平稳过程自身对相关函数值的影响 我们进行标准化
$$r_{X}(\tau):=\frac{R_{X}(\tau)-m_{X}^{2}}{C_{X}(0)}$$
称为相关系数
他还是用来刻画随机过程时间间隔为$\tau$的两个随机变量的线性相关程度的大小 根据定义知道$\lim_{r\to\infty}r_X(\tau)=0$ 

在工程上 我们认为当相关系数的值小于$0.05$的时候 就认为他们不相关 这个时间称为相关时间
我们可以从$\mid r_{X}(\tau_{0})\mid\leqslant0.05$ 来计算相关时间 
也可以用$\tau_{0}=\int_{0}^{+\infty}r_{X}(\tau)\mathrm{d}\tau$ 来计算
**相关时间的大小体现了过程起伏速度的快慢，相关时间短，意味着受曾经的影响小，起伏变化更快 反之亦然**

下面给出一个简单的例子
设平稳信号$\mathrm{X= \{ X_t: t\geq 0\} }$和$\mathrm{Y= \{ Y_t: t\geq 0\} }$ 的协方差函数分别为
$$C_X(\tau)\color{}{=}\frac14e^{-2\lambda|\tau|},\quad\quad\quad\quad C_Y(\tau)\color{}{=}\frac{\sin\lambda\tau}{\lambda\tau}$$
计算两者的相关函数和相关时间并且做解释
$$r_{X}(\tau)=\frac{R_{X}(\tau)-m_{X}^{2}}{C_{X}(0)}=\frac{\mathcal{C}_{X}(\tau)}{\mathcal{C}_{X}(0)}=e^{-2\lambda|\tau|}$$
$$r_{_Y}(\tau)=\frac{\mathsf{C}_{_Y}(\tau)}{\mathsf{C}_{_Y}(0)}=\frac{\sin\lambda\tau}{\lambda\tau}$$
积分计算相关时间
$$\tau_{0}^{\chi}=\int_{0}^{\infty}\mathbf{r}_{X}(\tau)d\tau=\int_{0}^{\infty}e^{-2\lambda|\tau|}\mathbf{d}\tau=\frac{1}{2\lambda}$$
$$\tau_{0}^{Y}=\int_{0}^{\infty}\mathbf{r}_{Y}(\tau)d\tau=\int_{0}^{\infty}\frac{\sin{\lambda\tau}}{\lambda\tau}\mathbf{d}\tau=\frac{1}{\lambda}$$
也就是X的起伏程度比Y更大

#### 用相关函数讨论平稳过程连续
平稳过程 $X=\{X_t:t\in\mathbb{T}\}$ 均方连续的充要条件是：X 的相关函数 $R_X(\tau)$在$\tau=0$处连续
设平稳过程 $X=\{X_t:t\in\mathcal{T}\}$的相关函数为$R_x(\tau)$,则 $R_X(\tau)$在任意一点$\tau\in\mathbb{R}$ 处连续的充要条件是 $R_x(\tau)$ 在 $\tau=0$ 处连续
#### 两个平稳过程的相关
设 $X=\{X_t:t\in\mathbb{T}\}$ 和 $Y=\{Y_t,t\in\mathbb{T}\}$为平稳过程，对任意的 $s,t\in\mathcal{T}$,称
$R_{XY}(s,\:t)=\mathbf{E}\left[\overline{X}_{s}Y_{t}\right]$为平稳过程 X 和 Y 的互相关函数
如果
$$R_{XY}(t,t+\tau)=\mathbb{E}\left[\overline{X}_{t}Y_{t+\tau}\right]=R_{XY}(\tau)$$
则称这两个平稳过程是联合平稳的
互相关系数定义为
$$r_{XY}(\tau):=\frac{R_{XY}(\tau)-m_{X}m_{Y}}{[C_{X}(0)]^{\frac{1}{2}}[C_{Y}(0)]^{\frac{1}{2}}}$$
如果互相关系数为0 就称为两个随机过程不相关
如果互相关函数为0 就称两个随机过程正交
我们可以给出联合平稳的性质
如果两个平稳过程是联合平稳的 互相关函数满足
* $R_{XY}(\tau)=\overline{{R_{YX}\left(-\tau\right)}}$
* $|R_{XY}(\tau)|^{2}{\leqslant}R_{X}(0)R_{Y}(0),|R_{YX}(\tau)|^{2}{\leqslant}R_{X}(0)R_{Y}(0)$
* $Z_{t}=\alpha X_{t}+\beta Y_{t}$ 也是一个平稳过程
### 各态历经性
使用样本信息研究平稳过程的数字特征 需要测量平稳过程的多个样本函数（经典统计学基于大数定律进行这方面的研究）但是同时间的多次观测是非常难实现的；

随机过程具有二重性 同时和$w~t$相关 随机的$w$的让我们难以研究 但是$t$的函数性让我们才想 能否使用一个长时间观测到的信息去估计数字特征？ 这就是各态历经性要探讨的问题
#### 引入
对于平稳过程 均值函数和时间指标无关 相关函数是时间指标差的函数 ，我们考虑只用一个样本函数来估计均值函数和相关函数 

定义：对于一个平稳过程$X_t$ 如果均方极限
$$\langle X_{\iota}\rangle\triangleq\operatorname*{l.i.m}_{T\to\infty}\frac{1}{2T}\int_{-T}^{\tau}X_{\iota}\mathrm{d}t$$
存在 则称$<X_{t}>$是平稳过程的时间平均
如果对于任意固定的$\tau$ 均方极限
$$<\overline{X}_{t}X_{t+\tau}>=l.i.m\frac{1}{2T}\int_{-T}^{T}\overline{X}_{t}X_{t+\tau}dt$$
存在 则称$<\overline{X}_{t}X_{t+\tau}>$为平稳过程的时间相关函数
对于参数集大于0的平稳过程  我们可以做下面的修正
$$<X_t>=\underset{T\to\infty}{\operatorname*{l.i.m}}\frac1T\color{}{\int_0^TX_tdt}$$
$$<\overline{X}_tX_{t+\tau}>=li.m\frac1T\int_0^T\overline{X}_tX_{t+\tau}dt$$
#### 定义
设$X=\{X_t:t\in(-\infty,+\infty)\}$是平稳过程，如果以概率1，有
$$
\langle X_{t}\rangle=m_{_{X}}
$$
则称平稳过程 X 的均值函数具有各态历经性。

如果对任意的实数$\tau$,以概率 1,有
$$\langle\overline{\mathbf{X}}_{\iota}X_{\star\tau}\rangle=R_{x}(\tau)$$
则称平稳过程 $X$ 的相关函数具有各态历经性

**如果平稳过程 $X$ 的均值函数和相关函数都具有各态历经性，则称平稳过程 $X$ 具有各态历经性**

各态历经性的意义是平稳过程的任一样本在足够长的时间经历了这个过程的各种可能的状态 因此对于具有各态历经性的平稳过程 可以用长时间的观测得到的样本信息来估计平稳过程的均值函数和相关函数
#### 判定
##### 均值
设$X=\{X_t:{t\in(-\infty,+\infty)}\}$是平稳过程，$C_x(\tau)$是$X$ 的协方差函数， 则 X 的均值函数具有各态历经性的充要条件是
$$
\lim_{\tau\to\infty}\frac{1}{2\:T}\int_{-2\:T}^{2\:\tau}\Big(1-\frac{|\:\tau\:|}{2\:T}\Big)C_{X}(\tau)\:\mathrm{d}\tau=0
$$
如果$X$是一个实平稳过程 那么条件可以变化为
$$\lim_{T\to+\infty}\frac1T\int_0^{2T}(1-\frac\tau{2T})C_X(\tau)d\tau=0$$
这是因为协方差函数是偶函数
对于$t\ge0$的平稳过程 则上面的条件变为
$$\lim_{T\to+\infty}\frac1T\int_{-T}^T(1-\frac{|\tau|}T)C_X(\tau)d\tau=0$$
此时如果还是实平稳过程可以继续变形为
$$\lim_{T\to+\infty}\frac2T\int_{0}^{T}(1-\frac\tau T)C_{X}(\tau)d\tau=0$$
我们还能给出一个充分非必要的条件
设平稳过程X=$\{X_t,-\infty<t<+\infty\}$的协方差
$$\lim_{\tau\to\infty}C_X(\tau)=0$$
则X的均值具有各态历经性 **这个定理比前面更好用了，当然也变得非必要**
##### 相关函数
设 $X=\{X_t:t\in(-\infty,+\infty)\}$是平稳过程，且对固定的实数 $\tau$,令
$$
Y_{t}=\overline{X}_{t}X_{t+\tau}
$$
若$Y=\langle Y_t:t\in(-\infty,+\infty)\rangle$是平稳过程，则$X$的相关函数具有的各态历经性是新的平稳过程的均值的各态历经性 把协方差替换了就好
所以充要条件为
$$
\lim\limits_{\tau\to\infty}\frac{1}{2T}\int_{-2T}^{2\tau}\left(1-\frac{\mid\boldsymbol{u}\mid}{2T}\right)(R_Y(u)-\mid R_X(\tau)\mid^2)\:\mathrm{d}u=0
$$
**前面的均值的定理都可以再用到这里了**
### 功率谱密度
在无线电、通信技术等领域的一些问题中, 通常需要分析平稳过程(信号)的频域结构 为此引入平稳过程的功率谱密度
#### 功率谱密度的概念
$X=\{X_t:t\in\mathbb{T}\}$ 是平稳过程，记
$$S_{\chi}(\omega)=\lim_{T\to+\infty}\frac1{2T}\mathrm{E}\left|\int_{-T}^{T}e^{-j\omega t}X_tdt\right|^2$$
称 $S_x(\omega)$为平稳过程 $X$ 的功率谱密度，简称谱密度
又称
$$\lim_{T\to\infty}\frac{1}{2T}\mathrm{E}\Big[\int_{-T}^{T}|X_{t}|^{2}\mathrm{d}t\Big]$$
为平稳过程的平均功率

定理 设平稳过程 $X=\{X_t:t\in\mathbb{T}\}$ 的相关函数 $R_x(\tau)$绝对可积，则有
$$S_X(\omega)=\int_{-\infty}^{+\infty}e^{-j\omega t}R_X(\tau)dt$$
因此我们知道 **相关函数和谱密度是一对傅立叶变换** 我们称为辛钦维纳公式
$$S_{X}(\omega)=\int_{-\infty}^{+\infty}e^{-j\omega\tau}R_{X}(\tau)d\tau,-\infty<\omega<+\infty $$
$$R_X(\tau)=\frac1{2\pi}\int_{-\infty}^{+\infty}e^{j\omega\tau}S_X(\omega)d\omega,-\infty<\tau<+\infty $$
#### 谱密度的性质
设$S_X(w)$ 是谱密度 则
* $\text{谱密度 }S_X(\omega)\text{为实值非负函数}$
* $\text{如果 }X\text{ 为实平稳过程,则谱密度 }S_X(\omega)\text{为偶函数。}$
* $\begin{cases}S_X(0)=\int_{-\infty}^{+\infty}R_X(\tau)d\tau\\\\R_X(0)=\frac{1}{2\pi}\int_{-\infty}^{+\infty}S_X(\omega)d\omega\end{cases}$
均方值函数就是 $R_X(t,t)$  数字特征的基础就讲过了
前者能从相关函数直接导出谱密度在0处的取值 躲避了复数域上的积分
后者能从谱密度中导出相关函数在0处的取值 也就是平均功率
#### 功率谱密度的计算
使用前面给出的辛钦维纳公式进行计算 结合傅立叶变换的性质
补充傅立叶变化的性质如下
* $\mathcal{F}[\alpha f_{1}(t)+\beta f_{2}(t)]=\alpha\mathcal{F}[f_{1}(t)]+\beta\mathcal{F}[f_{2}(t)]$
* $\mathcal{F}[f(t\pm t_0)]=e^{\pm jat_0}\mathcal{F}[f(t)]$
* $\mathcal{F}[f^{(n)}(t)]=(j\omega)^{n}\mathcal{F}[f(t)]$
这个还是很重要的 看看例子理解一下吧
### 谱分解
#### 相关函数的谱分解
设$X=\{X_i:t\in\mathbb{T}\}$是均方连续的平稳过程，则其相关函数 $R_x(\tau)$ 可表示为
$$
R_{X}\left(\tau\right)=\frac{1}{2\pi}\int_{-\infty}^{+\infty}\mathrm{e}^{i\omega\tau}\:\mathrm{d}F_{X}\left(\omega\right),\quad\tau\in\left(-\infty,\:\infty\right)
$$
这就是相关函数谱分解的基本定理 我们称这个分解式为谱分解式 $F_X(w)$称为谱函数
 
我们容易验证 如果相关函数绝对可积 使用辛钦维纳公式可以知道 
$$F_{X}\left(\omega\right)=\int_{-\infty}^{\omega}S_{X}\left(\omega\right)\mathrm{d}\omega $$
可以用来比较轻松的计算谱函数 

一个例子
设X，Y是两个相互独立的实随机变量，$E(X)=0~D(X)=1$ $Y$的分布函数为$F(x)$,令
$$Z_{_t}=Xe^{jtY},-\infty<t<+\infty,$$
计算$Z$的谱函数
容易计算
$$m_{z}(t)=0$$
$$\begin{gathered}
R_{z}(t,t+\tau)=\int_{-\infty}^{+\infty}e^{j\tau\omega}dF(\omega) \\
=\frac{1}{2\pi}\int_{-\infty}^{+\infty}e^{j\tau\omega}d(2\pi F(\omega)) 
\end{gathered}$$
再代入进行傅立叶变化计算谱密度函数 代入前面的公式计算谱函数有
$$F_{z}(\omega)=2\pi F(\omega)$$
#### 平稳过程的谱分解
我们只在这里说明
**零均值均方连续的复平稳过程可以进行谱分解**

定理的实际意义是
**平稳过程可以看作一定振幅 角频率 的谐波的有限叠加和的均方极限**
