---
title: "多元统计引论：随机向量、协方差矩阵与多元正态分布"
title_en: "Multivariate Statistics Introduction: Random Vectors, Covariance Matrices, and Multivariate Normal Distributions"
date: 2024-09-11 18:36:49 +0800
categories: ["Data Science", "Statistical Modeling & Inference"]
tags: ["Learning Notes", "Statistics", "Multivariate Statistics"]
author: Hyacehila
excerpt: "整理随机向量、协方差矩阵、多元正态分布、广义方差、距离和相关基础概念。"
excerpt_en: "Introduces multivariate statistics through random vectors, covariance matrices, multivariate normal distributions, generalized variance, distances, and core concepts."
mathjax: true
hidden: true
permalink: '/blog/2024/09/11/multivariate-statistics-introduction-notes/'
---
## 随机向量的补充知识
在研究线性统计模型的时候，随机向量广泛的出现在各个地方，这里对概率论中未能详细叙述的随机向量给出一些补充的内容
### 随机向量数字特征
对于随机向量 我们定义他的均值为
$$E(X)=(EX_1,\cdots,EX_n)^{\prime}$$
如果 $Y=AX+b$ 那么
$$E(Y)=AE(X)+E(b).$$
并且有
$${E}(АXB)=\mathrm{A}{E}(X)B$$

对于随机向量 我们定义他的协方差矩阵为
$$\mathrm{Cov}(X)=E[(X-EX)(X-EX)^{\prime}].$$
容易看出 矩阵的是对称的，各个元是概率论中经常见到的协方差，对角线上就是方差$Var(X_{i})$ 协方差可以用来判断相关
**（只能判断线性相关性，方差与相关系数为0都不意味着相互独立）**
研究协方差矩阵的迹有
$$\mathrm{trCov}(X)=\sum_{i=1}^{n}\mathrm{Var}(X_{i})$$
研究矩阵二次型的正定性质有
$$协方差矩阵是对称的半正定矩阵$$
如果$Y=AX$
$$\operatorname{Cov}(\boldsymbol{Y})=A\operatorname{Cov}(\boldsymbol{X})\boldsymbol{A}^{\prime}.$$
推广到两个随机向量之间的协方差矩阵有
$$\mathrm{Cov}(X,Y)=E[(X-EX)(Y-EY)^{\prime}].$$

补充常见的计算性质有
$$\mathrm{Cov}(AX,BY)=A\mathrm{Cov}(X,Y)B^{\prime}.$$

### 随机向量二次型
定义 随机向量的二次型为
$$X^{\prime}AX=\sum^{n}\sum^{n}a_{ij}X_{i}X_{j}$$
我们的随机向量取代了原本的$X$的位置，而不是用协方差矩阵取代原本的对称矩阵$A$
请注意 随机向量的二次型本质上是一个随机变量

对于这个随机变量 我们能给出期望的计算公式
$$\text{设 }\operatorname{E}(X)=\boldsymbol{\mu},\operatorname{Cov}(X)=\boldsymbol{\Sigma},\text{则}E(X^{\prime}AX)=\boldsymbol{\mu}^{\prime}A\boldsymbol{\mu}+\mathrm{tr}(A\boldsymbol{\Sigma}).$$
$tr$ 运算符是求矩阵的迹使用的 也就是对角线元素的和
根据这个公式 我们能给出一些推论
$$\text{若 }\mu=0,\text{则}E(X'AX)=\mathrm{tr}(A\Sigma);$$
$$\textrm{若}\boldsymbol{\Sigma}=\sigma^{2}\boldsymbol{I},\text{则}E(X^{\prime}AX)=\mu^{\prime}A\mu+\sigma^{2}(\sum_{i=1}^{n}a_{ii})=\mu^{\prime}A\mu+\sigma^{2}\mathrm{tr}A;$$
$$\text{若 }\mathbf{\mu}=\mathbf{0},\mathbf{\Sigma}=\mathbf{I},\text{则}E(\boldsymbol{X'AX})=\mathrm{tr}\boldsymbol{A}.$$
### 正态随机向量
#### 基础知识
我们在前面学习过正态随机变量 表示服从正态分布的随机变量，他的形式如下 记作$N(\mu,\sigma^{2})$
$$f(x)=\frac{1}{\sqrt{2\pi}\sigma}\mathrm{e}^{-\frac{1}{2\sigma^{2}}(x-\mu)^{2}},-\infty<x<+\infty $$
他的二维推广应该是
$$f(x_1,x_2)=\frac1{2\pi\sigma_1\sigma_2\sqrt{1-\rho^2}}e^{-\frac1{2(1-\rho^2)}\left(\frac{(x_1-\mu_1)^2}{\sigma_1^2}-2\rho\cdot\frac{x_1-\mu_1}{\sigma_1}\cdot\frac{x_2-\mu_2}{\sigma_2}+\frac{(x_2-\mu_2)^2}{\sigma_2^2}\right)}$$
现在，我们自然的推广这个一维的形式到随机向量上 
定义：$\text{设 }n\text{ 维随机向量 }{X}=(X_1,\cdotp\cdotp\cdotp,X_n)`\text{具有密度函数}$
$$f(x)=\frac{1}{(2\pi)^{\frac{n}{2}}(\det\boldsymbol{\Sigma})^{\frac{1}{2}}}\mathrm{e}^{-\frac{1}{2}(x-\boldsymbol{\mu})/\boldsymbol{\Sigma}^{-1}(x-\boldsymbol{\mu})},$$
我们一般将其记作$N(\boldsymbol{\mu},\boldsymbol{\Sigma})$ 这两个量是他们的分布参数
定理 ： 如上分布的正态向量满足 $E(X)=\mu,\quad\mathrm{Cov}(X)=\boldsymbol{\Sigma}.$

从上面的最核心的叙述上能够看出，多元正态分布被他的均值向量$\mu$与协方差矩阵$\Sigma$ 完全确定
另外的 当$\mathbf{\mu}=\mathbf{0},\boldsymbol{\Sigma}=\mathbf{I}$ 的时候 我们称其为多元标准正态分布

他的特征函数为
$$\Phi_{X}\left(t\right)=\exp\left[\mathrm{i}t^{\prime}\mu-\frac{1}{2}t^{\prime}\Sigma t\right]$$
特征函数可以和分布函数（密度函数相互确定）
#### 判定定理
我们知道 正态向量的每一个分量是正态变量 但是两个正态变量组成的联合分布不一定是一个正态向量 因此我们给出了如下的判定定理

随机向量 $X=(X_1,\quad X_2,\cdots,\quad X_n)^T$ 服从 $n$ 维正态分布 $N(a,B)$ 的充分必要条件是它的任何一个线性组合$Y=\sum_{i=1}^nl_iX_i$服从 一维正态分布$$N(\sum_{i=1}^nl_ia_i,\sum_{i=1}^n\sum_{k=1}^nl_il_k\operatorname{cov}(X_i,X_k))$$
根据这个定理很容易看出以前证明的结论 两个独立的正态变量的联合分布一定是正态向量
#### 分解定理
我们在研究二元正态分布（概率论）的时候提到过二元正态分布的分解，我们现在进一步将它推广
设多元正态分布的协方差矩阵有$\Sigma$满足分块对角矩阵形式
$$\left.\boldsymbol{\Sigma}=\left[\begin{matrix}\boldsymbol{\Sigma}_{11}&\boldsymbol{0}\\\boldsymbol{0}&\boldsymbol{\Sigma}_{22}\end{matrix}\right.\right],$$
那么我们可以进行对应的分解
$$\boldsymbol{X}=\begin{pmatrix}\boldsymbol{X}_1\\\boldsymbol{X}_2\end{pmatrix},\quad\boldsymbol{\mu}=\begin{pmatrix}\boldsymbol{\mu}_1\\\boldsymbol{\mu}_2\end{pmatrix},$$
这时候我们能验证到
$$f(x)=f_{1}(x_{1})f_{2}(x_{2}),$$
当
$$\begin{aligned}f_1(x_1)&=\frac{1}{(2\pi)^{\frac{m}{2}}\det\Sigma_{11}}\mathrm{e}^{-\frac{1}{2}(x_1-\mu_1)/\Sigma_{11}^{-1}(x_1-\mu_1)},\\f_2(x_2)&=\frac{1}{(2\pi)^{\frac{n-m}{2}}\det\Sigma_{22}}\mathrm{e}^{-\frac{1}{2}(x_2-\mu_2)\Sigma_{22}^{-1}(x_2-\mu_2)}.\end{aligned}$$
其实我们在二元正态分布里面提到的相关系数$\rho$    就是决定协方差矩阵副对角元是否为0的关键
根据上面的推论我们能够给出一个重要的正态分布分解定理
$$\begin{aligned}\text{(a) 设 X}\sim N(\mu,\Sigma),\text{且 X 和}\mu\text{ 分别有分块形式},\text{而}\boldsymbol{\Sigma}\text{ 具有分块对角形式},\text{则 }\boldsymbol{X}_i\sim N(\boldsymbol{\mu}_i,\boldsymbol{\Sigma}_{ii}),i=1,2\text{ 且相互独立}.\end{aligned}$$
$$\begin{aligned}(\mathrm{b})\text{若 }\boldsymbol{\Sigma}=\sigma^2\boldsymbol{I},\text{且记 }\boldsymbol{X}=(X_1,\cdots,X_n)^{\prime},\boldsymbol{\mu}=(\mu_1,\cdots,\mu_n)^{\prime},\text{则 }X,N(\mu_i,\sigma^2),i&=1,\cdots,n\text{ 且相互独立}.\end{aligned}$$
*这个定理告诉我们对于某个正态向量的分量而言 独立和不相关是等价的，我们只需要验证协方差为0就能保证独立 当然 我们的前提是他们构成一个正态向量，后面的定理就会帮助我们解决这个问题*
#### 累加的正态向量
我们要求各个正态变量相互独立
$$\sum_{r=1}^na_rX_r\sim N{\left(\sum_{r=1}^na_r\mu_r,\sum_{r=1}^na_r^2\sigma_r^2\right)}$$
我们要求各个正态向量相互独立
$$\sum_{r=1}^nX_rA_r\sim N_m\left(\sum_{r=1}^n\mu_rA_r,\sum_{r=1}^n(A_r^{\prime}\sum_rA_r)\right)$$

#### 维数不变正态向量的变换
我们先给出正态向量变换的最核心的一个定理
$$\begin{gathered}
\text{ 设 n 维随机向量}\mathbf{X}\sim N(\boldsymbol{\mu},\boldsymbol{\Sigma}),\boldsymbol{A}\text{ 为}n\times n\text{ 非随机可逆阵},\boldsymbol{b} \\
\text{为 }n\times1\text{ 向量,记 }Y=AX+b\text{ ,则} \\
\boldsymbol{Y}\sim N(\boldsymbol{A}\boldsymbol{\mu}+\boldsymbol{b},\boldsymbol{A}\boldsymbol{\Sigma}\boldsymbol{A}^{\prime}). 
\end{gathered}$$
非常明显的，我们可以取一些比较特殊的变换用随机可逆矩阵实现一些特殊的效果
**修正协方差矩阵**
$$\text{设 }X\sim N(\boldsymbol{\mu},\boldsymbol{\Sigma}),\text{则 }Y=\boldsymbol{\Sigma}^{-\frac{1}{2}}\boldsymbol{X}\sim N(\boldsymbol{\Sigma}^{-\frac{1}{2}}\boldsymbol{\mu},\boldsymbol{I}).$$
变换实现了将原本可能相关的各个正态分量变为无关的 并且方差均为1
**正交变换**
$$\text{设 }\boldsymbol{X}\sim N_n(\boldsymbol{\mu},\sigma^2\boldsymbol{I}),\boldsymbol{Q}\text{ 为 }n\times n\text{ 正交阵,则 }\boldsymbol{Q}\boldsymbol{X}\sim N_n(\boldsymbol{Q}\boldsymbol{\mu},\sigma^2I)$$
正交变换保证了原本的独立且等方差性质
**再生性**
$$\begin{aligned}\text{设 }\boldsymbol{X}-\boldsymbol{N}_n(\boldsymbol{\mu},\boldsymbol{\Sigma}),\boldsymbol{\chi}\boldsymbol{X},\boldsymbol{\mu},\boldsymbol{\Sigma}\text{ 分块为}\\\boldsymbol{X}&=\begin{bmatrix}\boldsymbol{X}_1\\\boldsymbol{X}_2\end{bmatrix},\boldsymbol{\mu}=\begin{bmatrix}\boldsymbol{\mu}_1\\\boldsymbol{\mu}_2\end{bmatrix},\boldsymbol{\Sigma}:=\begin{bmatrix}\boldsymbol{\Sigma}_{11}&\boldsymbol{\Sigma}_{12}\\\boldsymbol{\Sigma}_{21}&\boldsymbol{\Sigma}_{22}\end{bmatrix},\text{其中 }\boldsymbol{X}_1\text{ 和 }\boldsymbol{\mu}_1\text{ 为 }m\times1\text{ 向盘,而 }\boldsymbol{\Sigma}_{11}\text{为 }m\times\boldsymbol{m}\text{ 矩阵,则 }\boldsymbol{X}_1-\boldsymbol{N}_m(\boldsymbol{\mu}_1,\boldsymbol{\Sigma}_{11}).\end{aligned}$$
这个定理告诉我们正态向量的任意维数子向量也是正态向量 这就是再生性

#### 维数变化的正态向量的变换
核心定理 只是前面的小改变
$$\begin{array}{c}\text{设 }X\sim N_n\left(\boldsymbol{\mu},\boldsymbol{\Sigma}\right),\boldsymbol{A}\text{ 为}m\times n\text{ 矩阵,且秩为 }m\left(<n\right),\text{则}\\\boldsymbol{Y}=\boldsymbol{A}\boldsymbol{X}\sim\boldsymbol{N}_m\left(\boldsymbol{A}\boldsymbol{\mu},\boldsymbol{A}\boldsymbol{\Sigma}\boldsymbol{A}^{\prime}\right).\end{array}$$
这个定理对维数增加的情况也适用 也就是 $m>0$ 的情形 
这让他在随机过程中研究正态过程用处很多
**维数降低到1**
$$\begin{array}{cc}&\text{设 }X\sim{~N_n(\boldsymbol{\mu},\boldsymbol{\Sigma}),\boldsymbol{c}\text{ 为}n\times1\text{ 非零向量,则}} \\ & c ^ { \prime }\boldsymbol{X}\sim N(\boldsymbol{c}^{\prime}\boldsymbol{\mu},\boldsymbol{c}^{\prime}\boldsymbol{\Sigma}\boldsymbol{c}).\end{array}$$
正态向量的线性组合是正态变量
**再把抽取的分量数变成1**
$$\begin{aligned}\quad&\text{设 }X\sim N_n\left(\boldsymbol{\mu},\boldsymbol{\Sigma}\right),\boldsymbol{\mu}=\left(\boldsymbol{\mu}_1,\cdots,\boldsymbol{\mu}_n\right)^{\prime},\boldsymbol{\Sigma}=\left(\boldsymbol{\sigma}_{ij}\right),\text{则 }X_i{\sim}&N\left(\boldsymbol{\mu}_i\right.,\\\sigma_{ii}\left.\right),&i=1,\cdots,n.\end{aligned}$$
这个定理告诉我们正态向量的每一个分量都是正态向量
但是请注意 这个定理的逆定理不成立 我们在概率论中给出过例子和证明
#### 条件分布与条件期望
$$\begin{aligned}\text{定理 2.3.2 }&\text{设 }X=\begin{bmatrix}X^{(1)}\\\\X^{(2)}\end{bmatrix}_{p-r}\sim N_{\rho}(\mu,\Sigma)(\Sigma>0),\text{则当 }X^{(2)}\text{给}\\\text{定时},X^{(1)}\text{的条件分布为}\\(X^{(1)}|X^{(2)})&\sim N_{r}(\mu_{1,2},\Sigma_{11,2}),\end{aligned}$$
其中
$$\\{\mu_{1\cdot2}=\mu^{(1)}+\Sigma_{12}\Sigma_{22}^{-1}(x^{(2)}-\mu^{(2)})}\\~~~{\Sigma_{11\cdot2}=\Sigma_{11}-\Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}.}$$

对应的 对于前例的条件分布有
$$(X^{(1)}|X^{(2)})\sim N_{r}(\mu_{1,2},\Sigma_{11,2})$$
则称
$$\mu_{1\cdot2}=\mu^{\left(1\right)}+\sum_{12}\sum_{n2}^{-1}\left(x^{\left(2\right)}-\mu^{\left(2\right)}\right)$$
是条件期望 记为$E\left(X^{\left(1\right)}|X^{\left(2\right)}\right)$

### 卡方分布
$\begin{aligned}&\text{令}Z_1,Z_2,\cdots,Z_n\text{ 以及}Z\text{ 是独立同分布的标准正态分布随机变量,}\\&\text{且}X=Z_1^2+Z_2^2+\cdots+Z_n^2\text{,则我们称X服从自由度n的卡方分布}\end{aligned}$
容易知道 卡方分布的密度函数是
$$g(x)=\begin{cases}\dfrac{1}{\Gamma\Big(\dfrac{n}{2}\Big)2^{\frac{n}{2}}}\mathrm{e}^{-\frac{x}{2}}x^{\frac{n}{2}-1},&\text{若}x>0,\\\\0,&\text{若}x\leqslant0.\end{cases}$$
定理 
$$\text{设 }\boldsymbol{X}\sim\mathrm{N}_n(\boldsymbol{0},\boldsymbol{\Sigma}),\boldsymbol{\Sigma}\text{ 为正定阵},\text{则 }\boldsymbol{X}^{\prime}\boldsymbol{\Sigma}^{-1}\boldsymbol{X}\sim\chi_n^2.$$
定理 
$$\begin{aligned}X\sim N_n(0,I_n),A\text{ 为 }n\times n\text{ 秩}r\text{对称阵},\text{}\text{ }\mathbf{A}^2=\mathbf{A}\text{时}\text{二次型}X^{'}AX\sim\chi_{r}^{2}\end{aligned}$$
定理
$$\begin{aligned}&\text{设 X}\sim N_n(\mathbf{0},\mathbf{I}_n),A\text{ 为 }n\times n\text{ 对称阵},B\text{ 为 }m\times n\text{ 阵}.\text{芳 }BA\\&=\mathbf{0},\text{则 BX 与}X^{\prime}AX\text{ 相互独立}.\end{aligned}$$
定理
$$\begin{gathered}
\quad\text{设 }X\sim N_n(\mathbf{0},\mathbf{I}),A\text{ 和B 皆为 }n\text{ 阶对称阵},\text{且 }A\boldsymbol{B}=\mathbf{0},\text{则二次} \\
\text{型 X'AX与X'BX 相互独立。} 
\end{gathered}$$



## 多元统计基础
### 多元正态分布的参数估计
#### 补充定义
多元正态总体的常见数字特征我们在概率论中已经介绍过了 [初等概率论](/blog/2023/03/18/elementary-probability-notes/) 的“随机向量的补充知识”一节 了解下面几个量就足够了
* 均值向量
* 协方差矩阵
* 相关系数阵

补充定义**样本离差矩阵**为：
$$\begin{aligned}A&=\sum_{a=1}^n\left(X_{(a)}-\overline{X}\right)\left(X_{(a)}-\overline{X}\right)^{\prime}=X^{\prime}X-n\overline{X}\overline{X^{\prime}}\\&=X^{\prime}\left[I_n-\frac{1}{n}\mathbf{1}_n\mathbf{1}_n^{'}\right]X\xrightarrow{\mathrm{def}}\left(a_{ij}\right)_{p\times p}\end{aligned}$$
其中
$$a_{ij}=\sum_{a=1}^{n}\left(x_{ai}-\overline{x}_{i}\right)\left(x_{aj}-\overline{x}_{j}\right)$$
这就是协方差矩阵没有除 $n~or~ n-1$  
#### 均值和协方差矩阵的极大似然估计
现在 我们来了解如何估计多元正态分析中的两个核心参数 $\mu,\Sigma$ 

在$\Sigma$给定的时候 有
$$\begin{aligned}
\ln L\left(\mu,\Sigma\right)& =-\frac{np}{2}\ln2\pi-\frac{n}{2}\ln\left|\Sigma\right|  \\
&-\frac{1}{2}\mathrm{tr}\left[\Sigma^{-1}\sum_{i=1}^{n}\left(x_{\left(i\right)}-\mu\right)\left(x_{\left(i\right)}-\mu\right)^{\prime}\right] \\
&=C-\frac{1}{2}\mathrm{tr}\left[\Sigma^{-1}A+n\Sigma^{-1}\left(\overline{X}-\mu\right)\left(\overline{X}-\mu\right)^{\prime}\right] \\
&=C-\frac{1}{2}\mathrm{tr}\left(\Sigma^{-1}A\right)-\frac{n}{2}\left[\left(\overline{X}-\mu\right)^{\prime}\Sigma^{-1}\left(\overline{X}-\mu\right)\right] \\
&\leqslant C-\frac{1}{2}\mathrm{tr}\left(\Sigma^{-1}A\right).
\end{aligned}$$
等号只在$\mu=\overline{X}$ 的时候取到 也就是
$$\ln L\left(\overline{X},\Sigma\right)=\max_{\mu}\ln L\left(\mu,\Sigma\right).$$

使用类似的思路我们可以证明
当$$\hat{\Sigma}=\frac{1}{n}A$$的时候 有
$$\mathrm{ln}L\left(\overline{X},\hat{\Sigma}\right.)=\max_{\bar{X},\Sigma>0}\ln L\left(\overline{X},\Sigma\right):$$
这就是参数极大似然估计过程 

#### 其他极大似然估计量的导出与相关系数的极大似然估计
我们在学习前面的数理统计的过程中就介绍过如何从 $\mu,\Sigma$ 的极大似然估计导出$\phi(\mu,\Sigma)$ 的极大似然估计量 整体的思想就是自然的函数加工

那么 我们可以自然的导出相关系数的MLE
协方差矩阵元素的极大似然估计有
$$\hat{\sigma}_{ij}=\frac{1}{n}\sum_{t=1}^{n}\left(x_{ti}-\overline{x}_{i}\right)\left(x_{tj}-\overline{x}_{j}\right)=\frac{1}{n}a_{ij}.$$
根据相关系数定义式可以给出
$$r_{ij}=\frac{\hat{\sigma}_{ij}}{\sqrt{\hat{\sigma}_{ii}\cdot\hat{\sigma}_{jj}}}=\frac{a_{ij}}{\sqrt{a_{ii}\cdot a_{jj}}}.$$
#### 估计量的性质
极大似然估计量有下面的性质
* $\overline{X}\sim N_{p}\left(\mu,\frac{1}{n}\sum\right)$
* $\overline{X}和A相互独立$
* $A\xrightarrow{d}\sum_{i=1}^{n-1}Z_{i}Z_{i}^{\prime},\text{其中}Z_{1},\cdots,Z_{n-1}\text{独立同 }N_{p}(0,\Sigma)\text{分布}$
* $P\left\{A>0\right\}=1\Leftrightarrow n>p.$
* 均值的极大似然估计 无偏 有效 相合 具有渐进正态性 是充分统计量
* 协方差矩阵的极大似然估计 修正后无偏 有效 相合 具有渐进正态性 是充分统计量
* 相关系数的估计渐进无偏
* 单纯的均值与协反差矩阵的估计不要求正态总体


### 多元正态总体抽样分布
在一元正态总体中 假设检验涉及一个总体 两个总体 推广到多总体均值的方差分析问题；我们设计了很多精妙的统计量来帮助我们解决这些问题；这里我们将那些抽样统计量推广到多元正态总体 包括常用的$\chi^2,t,F$三种统计量的多元正态总体形式

#### 威沙特（Wishart）$W$分布
我们知道 在给出的两个重要统计量中 
$$\overline{X}\sim N_{p}\left(\mu,\frac{\sum}{n}\right).$$
那么协方差矩阵的估计 $S=\frac{1}{n-1}A$ 有着什么样的分布？

定义 设$X(a)\sim ~N_p\left(0,\Sigma\right)\left(\alpha=1,\cdots,n\right)$相互独立，记$X=$
$\left(X_{\left(1\right)},\cdots,X_{\left(n\right)}\right)^{\prime}为n\times p矩阵，则称随机阵$
$$W=\sum_{a=1}^{n}X_{\left(a\right)}X_{\left(a\right)}^{\prime}=X^{\prime}X$$
的分布为威沙特分布，记为W$\sim W_{p}\left(n,\Sigma\right).$

当$p=1$的时候 则有
$$W=\sum_{a=1}^{n}X_{\left(a\right)}^{2}\sim\sigma^{2}\chi^{2}\left(n\right),$$
也就是说 这就是卡方分布在多元正态总体中的推广
#### 霍特林（Hotelling）$T^2$分布
我们可以直接给出结论，它是原本的$t$分布的推广

设 $X\sim N_{p}\left ( 0, \Sigma \right )$,随机阵$W\sim W_{p}\left(n,\Sigma\right)\left(\Sigma>0,\right.$ $n\geqslant p),且X与W相互独立，则称统计量T^{2}=nX^{\prime}W^{-1}X$为霍特林$T^{2}$统计量，其分布称为服从$n$个自由度的$T^2$分布，记为$T^{2}\sim T^{2}\left(p,n\right).$
更一般地，若$X\sim N_{\rho}\left(\mu,\Sigma\right)\left(\mu\neq0\right)$,则称$T^2$的分布为非中心霍特林$T^2$分布，记为$T^2\sim T^2(p,n,\mu).$
#### 威尔克斯（Wilks）$A$分布
我们在参数估计的时候，使用 $A$ 作为协方差矩阵的估计；我们在多元统计中定义了广义方差 是协方差矩阵的行列式 [多元统计分析](/blog/2024/01/30/multivariate-statistical-analysis-notes/) 的“广义方差”一节

设$A_1\sim W_{p}\left(n_{1},\Sigma\right),A_{2}\sim W_{p}\left(n_{2},\Sigma\right)\left(\Sigma>0,n_{1}\geq\right.$$p),且A_{1}与A_{2}独立，则称广义方差之比$
$$\Lambda=\frac{\left|A_{1}\right|}{\left|A_{1}+A_{2}\right|}\\\text{为威尔克斯统计量或 }\Delta\text{统计量,其分布称为威尔克斯分布,记为}\\\Lambda\sim\Lambda\left(p,n_{1},n_{2}\right).\\当p=1时,\Lambda 统计量的分布正是一元统计中的参数为n_{1}/2\\n_{2}/2\text{ 的 }\beta\text{ 分布(记为 }\beta(n_{1}/2,n_{2}/2)).$$
#### 特别结论
$$\Delta\left(p,n,1\right)\frac{d}{1+\frac{1}{n}T^{2}\left(p,n\right)}$$

$$T^{2}\left(p,n\right)=n\cdot\frac{1-\Lambda\left(p,n,1\right)}{\Lambda\left(p,n,1\right)}$$

$$\frac{n-p+1}{np}T^{2}=\frac{n-p+1}{p}\frac{1-A}{\Lambda}=F\left(p,n-p+1\right).$$

## 多元统计推断
### 单个总体均值向量的推断
想要检验
$$H_0:\boldsymbol{\mu}=\boldsymbol{\mu}_0,\quad H_1:\boldsymbol{\mu}\neq\boldsymbol{\mu}_0$$
#### 协方差矩阵已知
构造统计量
$$T_0^2=(\overline{x}-\boldsymbol{\mu}_0)^{\prime}\left(\frac1n\boldsymbol{\Sigma}\right)^{-1}(\overline{\boldsymbol{x}}-\boldsymbol{\mu}_0)=n(\overline{\boldsymbol{x}}-\boldsymbol{\mu}_0)^{\prime}\boldsymbol{\Sigma}^{-1}(\overline{\boldsymbol{x}}-\boldsymbol{\mu}_0)$$
原假设为真时则有
$$T_{0}^{2}\sim\chi^{2}\left(p\right)$$
使用单侧检验有
$$\text{若 }T_0^2\geqslant\chi_\alpha^2(p),\text{ 则拒绝 }H_0$$
#### 协方差矩阵未知
构造统计量（霍特林$T^2$统计量）
$$T^{2}=n\left(\bar{x}-\mu_{0}\right)^{\prime}S^{-1}\left(\bar{x}-\mu_{0}\right)$$
原假设为真时则有
$$\frac{n-p}{p(n-1)}T^{2}\sim F(p,n-p)$$
使用单侧检验有
$$\text{若}\frac{n-p}{p\left(n-1\right)}T^2\geqslant F_a(p,n-p),\text{则拒绝 }H_0$$
#### 大样本推断
前面的证明都是基于多元正态假设的，但是有时候多元正态假设并不满足；但是当样本容量足够大的时候 多元中心极限定理可以为我们解决问题
当样本容量足够大的时候 有以下的近似服从关系
$$T_0^2=(\overline{x}-\boldsymbol{\mu}_0)^{\prime}\left(\frac1n\boldsymbol{\Sigma}\right)^{-1}(\overline{\boldsymbol{x}}-\boldsymbol{\mu}_0)=n(\overline{\boldsymbol{x}}-\boldsymbol{\mu}_0)^{\prime}\boldsymbol{\Sigma}^{-1}(\overline{\boldsymbol{x}}-\boldsymbol{\mu}_0)$$
$$T_{0}^2=n(\overline{x}-\mu)^{\prime}S^{-1}(\overline{x}-\mu)$$
$$T_{0}^{2}\sim\chi^{2}\left(p\right)$$
所有的假设检验和区间估计工作都沿用上面的公式

$$S^{-1}=A^{-1}(n-1)$$
### 似然比统计量
多元统计中非常多的重要检验统计量都是用最大似然比原理导出的，而不是延用数理统计中的最大似然原理。这里我们介绍似然比统计量和最大似然比原理

设 $p$ 元总体的密度函数为$f\left(x,\theta\right)$,其中$\theta$是未知参数，且$\theta\in\Theta$
$\left(参数空间\right),又设\Theta_{0}是\Theta 的子集，我们希望对下列假设：$
$$H_{0}:\theta\in\Theta_{0},H_{1}:\theta\in\Theta_{0}$$
作出判断，这就是假设检验问题.

从总体$X$抽取容量为$n$的样本$X_{(t)}(t=1,\cdots,n).$把样本的联合密度函数
$$L\left(x_{\left(1\right)},\cdots,x_{
\left(n\right)};\theta\right)=\prod_{t=1}^{n}f\left(x_{\left(t\right)};\theta\right)$$
记为$L\left(X;\theta\right)$,并称它为样本的似然函数

引入统计量
$$\lambda=\max_{\theta\in\theta_{0}}L\left(X;\theta\right)/\max_{\theta\in\theta}L\left(X;\theta\right),$$
它是样本$X_{(t)}\left(t=1,\cdots,n\right)$的函数，常称$\lambda$为似然比统计量.

由最大似然比原理知，如果$\lambda$取值太小，说明$H_0$为真时观测到此样本$X_{(\omega)}(t=1,...,n)$的概率比 $H_0$ 为不真时观测到此样本$X_{(\omega)}$ 的概率要小得多.故有理由认为假设$H_0$不成立

根据传统方法，我们需要计算似然比统计量的精确抽样分布才能给出一个假设检验结果，但是多元统计太复杂了，我们给出一个大样本近似定理有
当样本容量 n 很大时
$$-2ln\lambda=-2ln\left[\max_{\theta\in\Theta_{0}}L\left(X;\theta\right)/\max_{\theta\in\Theta}L\left(X;\theta\right)\right]$$
近似服从自由度为$f$的$\chi^2$分布 ,其中$f=\Theta$的维数$-\Theta_0$的维数（也就是自由度被限制的差值）
### 两个总体均值的推断
#### 协方差矩阵相等
设两个总体 $N_{p}(\mu_1,\Sigma),N_{p}(\mu_2,\Sigma)$ 各自独立的抽取两组样本$x_{n_1},y_{n_2}$ 
我们希望检验
$$H_0:\boldsymbol{\mu}_1=\boldsymbol{\mu}_2,\quad H_1:\boldsymbol{\mu}_1\neq\boldsymbol{\mu}_2$$
我们可以自然的从一元统计的情形中得到霍特林$T$统计量
$$\begin{aligned}T^2=&\left(\frac{1}{n_1}+\frac{1}{n_2}\right)^{-1}\left(\overline{x}-\overline{y}\right)^{\prime}S^{-1}\left(\overline{x}-\overline{y}\right)\\=&\frac{n_1n_2}{n_1+n_2}\left(\overline{x}-\overline{y}\right)^{\prime}S^{-1}\left(\overline{x}-\overline{y}\right)\end{aligned}$$
当原假设成立的时候
$$\frac{n_{1}+n_{2}-p-1}{p\left(n_{1}+n_{2}-2\right)}T^{2}\sim F(p,n_{1}+n_{2}-p-1)$$
可以自然的进行单侧检验量 方向和我们在前面介绍的一样
其中
$$S^{-1}=(\frac{A_1+A_2}{n_1+n_2-2})^{-1}$$
**两个均值向量之间存在显著差异，不意味着他们一定存在有着显著差异的分量**；也就是均值向量相等被拒绝，不意味着我们对每个分量单独检验的时候一定会检验出显著性差异； 

但是这种差异仍旧是均值向量差异的主要原因，在习惯上，我们在检验出整个向量的显著性差异后，会再单独研究分量之间的显著性差异
#### 成对实验情形
假定两个样本独立在某些情况下不成立；在不少的实验中，两个样本可能成对存在但是并不独立；成对出现的数据往往能带来更好的统计推断结果
令
$$d_i=x_i-y_i,\quad i=1,2,\cdots,n$$
则有$d_{i}$服从新的分布 
$$N_{p}\left(\delta,\Sigma\right)$$
其中
$$\delta=\mu_{1}-\mu_{2}$$
那么原假设
$$H_0:\boldsymbol{\mu}_1=\boldsymbol{\mu}_2,\quad H_1:\boldsymbol{\mu}_1\neq\boldsymbol{\mu}_2$$
变为
$$H_0:\boldsymbol{\delta}=\boldsymbol{0},\quad H_1:\boldsymbol{\delta}\neq0$$
问题转化为了单总体的情形 我们使用前面介绍过的思路就可以了

### 多个总体均值的比较检验（多元方差分析）
假设为
$$H_0:\boldsymbol{\mu}_1=\boldsymbol{\mu}_2=\cdot\cdot\cdot=\boldsymbol{\mu}_k,\quad H_1:\boldsymbol{\mu}_i\neq\boldsymbol{\mu}_j,\text{至少存在一对 }i\neq j$$
其中我们的$\mu$都是向量 而非方差分析中介绍的一元变量

记
$$T=SST=\sum_{i=1}^k\sum_{j=1}^{n_i}(x_{ij}-\overline{x})(x_{ij}-\overline{x})^{\prime}$$
$$E=SSE=\sum_{i=1}^k\sum_{j=1}^{n_i}(x_{ij}-\overline{x_i})(x_{ij}-\overline{x_i})^{\prime}$$
$$H=SSTR=\sum_{i=1}^kn_i\left(x_i-\overline{x}\right)\left(x_i-\overline{x}\right)^{\prime}$$
则有
$$T=E+H$$
利用似然比检验的方法可以得到威尔克斯（Wilks）统计量
$$\Lambda=\frac{|E|}{|E+H|}$$
当原假设为真的时候 统计量服从参数为$(p,k-1,n-k)$的威尔克斯分布

拒绝规则为
$$\text{若}\Lambda\leqslant\Lambda_{1-a}(p,k-1,n-k),\text{则拒绝 }H_0$$
多元检验出无显著差异，并不意味着他们的分量没有显著差异；反过来也是一样的；在习惯性上，如果多元检测出显著性差异，我们还是要进行一元的方差分析，看看差异大致来自于哪里
### 协方差矩阵的推断 
我们不考虑单总体的协方差矩阵的假设检验，因为其检验方法较复杂且并不唯一

假设为
$$H_0:\boldsymbol{\Sigma}_1=\boldsymbol{\Sigma}_2=\cdots=\boldsymbol{\Sigma}_k,\quad H_1:\boldsymbol{\Sigma}_i\neq\boldsymbol{\Sigma}_j,\text{至少存在一对 }i\neq j$$
修正的似然比统计量为
$$\lambda=\frac{\prod_{i=1}^k|S_i|^{(n_i-1)/2}}{|S_p|^{(n-k)/2}}$$
其中
$$S_i=\frac1{n_i-1}\sum_{j=1}^{n_i}{(x_{ij}-\bar{x_i})\left(x_{ij}-\bar{x_i}\right)}^{\prime}$$
$$S_p=\frac1{n-k}\sum_{i=1}^k{(n_i-1)S_i}=\frac1{n-k}E$$
构造$M$统计量有
$$M=-2\mathrm{ln}\lambda=\left.(n-k)\ln|S_p|-\sum_{i=1}^k{(n_i-1)}\ln|S_i\right. $$
当原假设为真的时候
$$u=(1-c)M$$
近似服从自由度为$\frac{1}{2}(k-1)p(p-1)$ 的卡方分布
其中
$$c=\Big(\sum_{i=1}^{k}\frac{1}{n_{i}-1}-\frac{1}{n-k}\Big)\frac{2p^{2}+3p-1}{6\left(p+1\right)\left(k-1\right)}$$
拒绝规则为
$$\text{若 }u\geqslant\chi_{a}^{2}\left[\frac{1}{2}\left(k-1\right)p\left(p+1\right)\right],\text{则拒绝 }H_{0}$$


## 多元统计预备知识
### 方差与广义方差
在一元概率论中 方差用来衡量一个随机变量的离散程度（变异程度）我们在数理统计中对样本的方差也进行了解释；概率论的多元部分 我们用协方差解释了两个随机变量之前的方差问题，虽然在数理统计中没有解释，但是不难外推出样本协方差（也需要修正为$n-1$）的概念，数学软件也可以为我们代劳；
协方差矩阵是一个矩阵；如何用一个数衡量随机向量的总变异性，这就是我们要回答的问题；
#### 总方差
总方差的定义为
$$\mathrm{tr}\left(\boldsymbol{\Sigma}\right)=\sum_{i=1}^{p}\sigma_{ii}$$
他没有考虑样本之间相关性的影响；这是他的一个缺陷，不过我们仍然会在后面某些地方使用它
$p=1$的时候退化为方差
#### 广义方差
广义方差最常用的定义式为
$$\left|\Sigma\right|$$
广义方差考虑了变量之间的相关性；但是可能被误导，也就是两个完全不同的协方差矩阵导出了相同的广义方差
$p=1$的时候退化为方差
### 欧氏距离和马氏距离
#### 欧氏距离
$p$维欧氏空间上两个点的欧氏距离为
$$d(x,y)=\sqrt{(x_{1}-y_{1})^{2}+(x_{2}-y_{2})^{2}+\cdots+(x_{p}-y_{p})^{2}}$$
为了避免根号表达 我们习惯使用平方欧氏距离
$$d^2(x,y)=(x-y)^{\prime}(x-y)=(x_1-y_1)^2+(x_2-y_2)^2+\cdotp\cdotp\cdotp+(x_p-y_p)^2$$
在各个分量单位不同的情况下，欧氏距离没有什么实际价值；
哪怕是单位相同的情况下，我们还是需要先对数据进行标准化（一般是z-scores）的处理；否则计算欧氏距离仍然没有实际意义；标准化也是数据预处理工作的基础步骤 ^3f42ea
#### 马氏距离
当随机变量之间存在某种线性关系的时候；欧氏距离会失去它原本的作用，无法对离群情况和接近情况作出正确的判断；

马氏距离的提出就是为了解决这个问题；他的本质是旋转坐标轴，让相关性消失后再使用欧氏距离的结果

两个向量的马氏距离定义如下
$$d^{2}\left(x,y\right)=\left(x-y\right)^{\prime}\boldsymbol{\Sigma}^{-1}\left(x-y\right)$$
向量到总体的马氏距离定义如下
$$d^{2}\left(x,\pi\right)=\left(x-\mu\right)^{\prime}\Sigma^{-1}\left(x-\mu\right)$$
其中$x,y$是随机向量 $\mu,\Sigma$ 分别是均值向量和协方差矩阵

下面我们简单介绍一下马氏距离的性质
* 关于变换的不变性$y=Cx+b$
比例尺度的变化可以表示为$y=Cx$  某种加减定量值的变换为$y=x+b$ 

马氏距离对他们具有不变性意味着进行这种变换前后不会影响我们对马氏距离的判断

标准化变换也是前面变换的特殊形式，进行标准化前后马氏距离不变
**当各个分量不相关的时候，马氏距离就是标准化后的欧氏距离**
