---
title: "概率与统计例题：抽样分布、参数估计与统计量性质"
title_en: "Probability and Statistics Exercises: Sampling Distributions, Estimation, and Statistics"
date: 2024-09-24 00:14:54 +0800
categories: ["Data Science & Statistics", "Probability & Statistical Foundations"]
tags: ["Learning Notes", "Statistics", "Probability", "Exercises"]
author: Hyacehila
excerpt: "整理古典概型、抽样分布、矩估计、极大似然估计和统计量性质等例题。"
excerpt_en: "A overview of probability and statistics exercises, covering classical probability, sampling distributions, moment estimation, maximum likelihood estimation, and properties of statistics."
mathjax: true
hidden: true
permalink: '/blog/2024/09/24/probability-and-statistics-exercises-notes/'
---
## 古典概型的经典例子
### 抛硬币
甲有n+1个硬币 乙有n个硬币 求甲的正面比乙多的概率？
*从样本点的角度研究绝对是非常复杂的，需要进行复杂的概率运算，这个题目给了我们一个有趣的思路*
记$A$是我们的目标事件的概率 那么$\bar{A}$ 就是甲的正面比乙少的概率 明显的 他也是甲反面比乙多的概率 借助对称性 $P(A)=P(\bar{A})且P(A)+P(\bar{A})=1$ 得到结论 答案为0.5
### 投球入格（生日问题，Maxwell-Boltzman统计的抽象形式）
$n个求 N个格子 N>n 各个球落入各个格子的概率均等，求$
* 指定的n个格子各有一个球 $P=n!/N^{n}$
* 任意的n个格子各有一个球 $P=N!/N^{n}(N-n)!$
*本质上的运算并不复杂，这里只是给 出一些叙述*
### 抽签与顺序无关的概率解释
$a个黑球 b个白球，求第k次摸出黑球的概率$
**认定球之前是不同的情况** 需要分别讨论**认定球之前是相同的情况**
$总数m=C_{a+b}^{k} 样本点数量n=C_{a+b-1}^{k-1}$
*这两个结果是一样的*
*（这是研究古典概型很重要的一点，你要保证在前后的认定一致）*


## 概率论中的数理统计例子
在实际问题的研究中，我们不应该知道次品的数量或者次品率，我们需要从样的情况来研究他，这是一个例子；

从池子中捕捞1200条鱼，标记后放回，然后再次进行1000个捕捞，发现有100个有标记，推算总体的数目；

明显的，这属于一个超几何分布问题，因为他是不放回的抽样，设出总数为N；我们能知道以上事件发生的概率应该是
$$P=\left(\begin{array}{l}
n_{1} \\
k
\end{array}\right)*\left(\begin{array}{l}
n-n_{1} \\
r-k
\end{array}\right)/\left(\begin{array}{l}
n \\
r
\end{array}\right)$$
其中$n_{1}$特殊的总数，$n$是总数，$k$是本次捕捞特殊的数目，$r$是本次捕捞的数目

根据**极大似然估计**的理论 我们研究P达到max的情况 得到 $n=n_{1}*r/k$ 是我们的答案


## 数理统计抽样分布例子
### 1
设母体$\xi\sim b(1,p)$(二点分布),$(\xi_1,\xi_2,\cdots,\xi_n)$为取自此母体的一个子样，$\bar{\xi}$为子样均值. 如果$p=0.2$ 那么样本容量$n$ 需要取多大才能满足
$$P(|\bar{\xi}-p|\leq0.1)\geq0.75;$$

如果我们做正态相关的问题还能把分布求出来 但是这样的就不行了 只能用基础的变形来做 然后查表来做
能看出$\sum\limits X_{i}$ 是二项分布 这是我们后面查表的基础
$$P(|\bar{\xi}-p|\leq0.1)=P(0.1\leq\bar{\xi}\leq0.3)\\=P(0.1n\leq\sum_{i=1}^{n}X_{i}\leq0.3n)\geq0.75$$
直接查表能得到$n=10$ 
### 2
子样($\xi_1,\xi_2,\xi_3)$来自正态母体$N(0,1)$,又$\eta_1=0.8\xi_1+0.{6}\xi_2$, $\eta_{2}=\sqrt{2}\left(0.3\xi_{1}-0.4\xi_{2}-0.5\xi_{3}\right),\eta_{3}=\sqrt{2}\left(0.3\xi_{1}-0.4\xi_{2}+0.5\xi_{3}\right)$,求($\eta_1,\eta_2$, $\eta_{3}$)的联合分布密度及$\eta_1,\eta_2,\eta_3$ 的边际分布

这是一个需要眼力的题目

首先我们能看出来  $\eta_1,\eta_2,\eta_3$ 还是服从$N(0,1)$ 的正态分布
子样($\xi_1,\xi_2,\xi_3)$来自正态母体 确保了他们是一个正态向量并且相互独立
新的$\eta_1,\eta_2,\eta_3$ 是一个原本的线性组合

我们知道如果原本是独立正态向量 那么乘上正交矩阵还是独立的正态向量
我们确实可以验证 这个变换的系数矩阵是正交矩阵

因此$\eta_1,\eta_2,\eta_3$ 相互独立 联合密度是各个密度函数的积 （各个密度函数都是变形后的正态 但是还是服从$N(0,1)$ ）

### 3

设母体$\xi$的分布函数为$F(x),(\xi_1,\xi_2,...,\xi_n)$是取自此母体的一个子样.若$F(x)$的二阶矩存在，$\overline{\xi}$为子样均值，试证($\xi_i-\bar{\xi})$ 与($\xi_j-\bar{\xi})$ 的相关系数为
$$\rho=-\frac{1}{n-1}$$

这个题目还是需要我们用基础的定义来求解 

相关系数的计算公式有 
$$Corr\left(X,Y\right)=\frac{Cov\left(X,Y\right)}{\sqrt{Var\left(X\right)}\sqrt{Var\left(Y\right)}}=\frac{Cov\left(X,Y\right)}{\sigma_{X}\sigma_{Y}}$$
而 $Cov(X,Y)=E(XY)-E(X)E(Y)$   
因此我们的问题还是转化为了期望的求解问题 
由于抽取和样本和样本均值一定不是独立的 我们需要拆开均值 对小项求期望 然后化简就可以得到题目中的相关系数结果

## 数理统计矩估计例子
### 1
设总体 $x$ 的均值和方差分别为 $\mu$ 与 $\sigma^{2}$,$X_1,X_2,\cdotp\cdotp,X_n$是总体$X$ 的样本，若总体的一、二阶原点矩都存在， 求 $\mu$与 $\sigma^2$ 的矩估计量.

根据矩估计法我们知道 需要两个方程
$$\mu=EX,\quad E(X^{2})=DX+E^{2}X=\mu^{2}+\sigma^{2}$$
令总体矩等于样本矩
$$\begin{cases}\mu=\frac1n\sum_{i=1}^nX_i,\\\\\mu^2+\sigma^2=\frac1n\sum_{i=1}^nX_i^2,\end{cases}$$
经典处理手法化简方差的表达式能得到（方差等于二阶矩减去期望的平方）
$$\left.\left\{\begin{matrix}{\mu=\overline{X},}\\{\sigma^{2}=\frac{1}{n}\sum_{i=1}^{k}(X_{i}-\overline{X})^{2}}\\\end{matrix}\right.\right.$$
这个例子告诉我们  无论分布情况如下
**总体均值的矩估计量是样本均值，总体方差的矩估计量是样本方差**
这是会是以后非常常用的定理

*注意，矩估计这里对方差的估计是有偏的，这点我们在后面才会详细介绍*
### 2
设总体的概率密度为$f(x)=\frac1{2\sigma}e^{-\frac{|x|}\sigma}$ 求解参数的矩估计

一个参数只需要一个矩法方程  计算总体的均值
$$E(X)=\int_{-\infty}^{+\infty}x\cdot\frac1{2\sigma}e^{-\frac{|x|}\sigma}dx=0$$
涉及奇函数的积分 不用计算就能得到结果
明显这种情况下 我们靠这个方程得不到矩估计的结果 所以计算二阶矩
$$\begin{gathered}
E(X^2)=\int_{-\infty}^{+\infty}x^2\cdot\frac1{2\sigma}e^{-\frac{|x|}\sigma}dx=\int_{0}^{+\infty}x^2\cdot\frac1\sigma e^{-\frac x\sigma}dx \\
=\int_{0}^{+\infty}(-x^{2})de^{-\frac{x}{\sigma}}=(-x^{2})e^{-\frac{x}{\sigma}}|_{0}^{+\infty}+\int_{0}^{+\infty}2xe^{-\frac{x}{\sigma}}dx \\
=2\int_0^{+\infty}(-x)d\sigma e^{-\frac x\sigma}=2\sigma\int_0^{+\infty}e^{-\frac x\sigma}dx=2\sigma^2 
\end{gathered}$$
还是涉及了一些对称性的简化和分部积分 后面建立参数估计方程就很简单了

## 数理统计极大似然估计例子
### 1
总体$X$的密度是 
$$\left.f(x;\lambda)=\left\{\begin{matrix}\lambda e^{-\lambda x},&x>0\\0,&\text{其它,}\end{matrix}\right.\right.;$$
其中 $\lambda>0$为未知参数，$X_1,X_2,...,X_n$ 是取自总体$X$的一组样本， 求$\lambda$的极大似然估计量与矩估计量

计算似然函数
$$\left.L(\lambda)=\prod_{i=1}^{n}f(x_{i};\lambda)=\left\{\begin{array}{c}\lambda^{n}\prod_{i=1}^{n}e^{-\lambda x_{i}},&x_{i}>0,i=1,2,\cdots,n;\\0,&\text{其他}.\end{array}\right.\right.$$
明显如果取等于0的时候没法计算估计量 所以找全部大于0 的样本 计算对数似然函数
$$\ln L(\lambda)=n\ln\lambda-\lambda\sum_{i=1}^{n}x_{i}$$
求导计算得到似然方程
$$\frac{d\ln L(\lambda)}{d\lambda}=\frac{n}{\lambda}-\sum_{i=1}^{n}x_{i}=0$$
得到极大似然估计量
$$\hat{\lambda}=\frac{1}{\overline{X}}$$
使用上一节的手段的计算矩估计量 
$$EX=\int_{-\infty}^{+\infty}xf(x;\lambda)dx=1/\lambda $$
$$\hat{\lambda}=\frac{1}{\overline{X}}$$
能看出这个问题中矩估计量和极大似然估计量结果一致

### 2
设总体$x\sim N(\mu,\sigma^2)$, 其中 $\mu,\sigma^2$ 均未知，设$X_1,X_2,...X_n$ 是取自$X$的一个样本. 求 $\mu$与 $\sigma^2$ 的极大似然估计量

多参数的问题的处理思路是完全一样的 还是数学分析中介绍的极大化方法
计算似然函数
$$L(\mu,\sigma^{2})=\prod_{i=1}^{n}f(x_{i};\mu,\sigma^{2})=\prod_{i=1}^{n}\frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{(x_{i}-\mu)^{2}}{2\sigma^{2}}}$$
对数似然函数
$$\ln L(\mu,\sigma^{2})=-\frac{n}{2}\ln2\pi\sigma^{2}-\frac{1}{2\sigma^{2}}\sum_{i=1}^{n}(x_{i}-\mu)^{2}$$
两个参数 得到似然方程组 分别做两个偏导就好了
$$\begin{aligned}&\frac{\partial\ln L(\mu,\sigma^2)}{\partial\mu}=\frac1{\sigma^2}\sum_{i=1}^{n}(x_i-\mu)=0,\\&\frac{\partial\ln L(\mu,\sigma^2)}{\partial\sigma}=-\frac n{\sigma}+\frac1{\sigma^3}\sum_{i=1}^{n}(x_i-\mu)^2=0,\end{aligned}$$
解方程得到 
$$\begin{cases}\hat{\mu}=\frac{1}{n}\sum_{i=1}^{n}X_{i}=\overline{X},\\\hat{\sigma}^{2}=\frac{1}{n}\sum_{i=1}^{n}(X_{i}-\mu)^{2}\end{cases}$$
也就是正态分布的极大似然估计量 也等于矩估计量

### 3
设总体$x$服从均匀分布$U[0,\theta]$,为$\theta$未知参数， $X_1,X_2,...,X_n$ 是总体$X$的一组样本，求$\theta$的极大似然估计量

样本似然函数为
$$\left.L(\theta)=\prod_{i=1}^{n}f(x_{i};\theta)=\left\{\begin{array}{ll}\frac{1}{\theta^{n}}&0\leq x_{i}\leq\theta,i=1,2,\cdots,n;\\0\text{,其他}.\end{array}\right.\right.$$

明显的 无论是直接对似然函数求导还是建立对数似然函数求导都不能做到极大化这个函数 这是数学分析中介绍的这个方法的缺点 他并不是万能的 我们只能直接使用极大似然原则来进行求解

明显的 似然函数是单调减少的 我们需要尽可能让$\theta$ 变小 
不过我们一定需要满足 $$0\leq x_{i}\leq\theta({i=1,2,\cdots,n})$$这个的话 我们可以取 极大似然估计量为
$$\hat{\theta}=\max_{1\leq i\leq n}\{X_{i}\}$$
### 4
设$\xi_1,\xi_2,\cdots,\xi_n$ 是取自对数正态分布母体$\xi$ 的一个子样，即 In $\xi$ - $N(\mu,\sigma^2),-\infty<\mu<+\infty$ ,$0<\sigma<+\infty$.试求 $\xi$ 的期望值 $E\xi$ 和方差 $D\xi$ 的极大似然估计

这里不是在问我们参数$\mu,\sigma^2$的极大似然估计而是分布特征数的极大似然估计，事实上，分布的特征数就是一个含有分布参数的表达式，我们需要求出这个表达式，然后带入参数的极大似然估计，就是特征数的极大似然估计

明显的 需要研究的分布$\xi$ 是一个正态分布的函数 需要用这个思路来简化期望和方差的计算
记$\eta=ln\xi$ 则$\eta\sim N(\mu,\sigma^2)$  $\xi=e^{\eta}$  使用期望和方差的函数公式进行计算有
$$E\xi=\exp\left\{\frac{1}{2}(2\mu+\sigma^{2})\right\}$$
$$D\xi=e^{2\mu+\sigma^{2}}[e^{\sigma^{2}}-1].$$
使用概率论中定理计算$\xi$的分布有
$$\left.p_{\xi}\left(y\right)=\left\{\begin{array}{cc}\frac{1}{\sqrt{2\pi\sigma}}\exp\{-\frac{1}{2\sigma^2}(\ln x-\mu)^2\}\frac{1}{y},&y>0,\\0,&\text{else}\end{array}\right.\right.$$
他的极大似然估计前面已经计算过了
$$\begin{cases}\hat{\mu}=\frac{1}{n}\sum_{i=1}^{n}X_{i}=\overline{X},\\\hat{\sigma}^{2}=\frac{1}{n}\sum_{i=1}^{n}(X_{i}-\mu)^{2}\end{cases}$$
将$X_{i}$ 变化为 $ln\xi_{i}$  把他们的结果带入到 
$$E\xi=\exp\left\{\frac{1}{2}(2\mu+\sigma^{2})\right\}$$
$$D\xi=e^{2\mu+\sigma^{2}}[e^{\sigma^{2}}-1].$$
就是特征数的极大似然估计量

## 数理统计估计量的评估准则
### 无偏性
#### 1
设总体$x\sim N(\mu,\sigma^2)$, 其中 $\mu,\sigma^2$ 均未知，设$X_1,X_2,...X_n$ 是取自$X$的一个样本. 求 $\mu$与 $\sigma^2$ 的极大似然估计量和矩估计量的无偏性
对于矩估计 
$$\hat{\theta}_{M}=2\bar{X}$$
则有
$$E(\hat{\theta}_{M})=2E(\bar{X})=2E(X)=2\cdot\frac{\theta}{2}=\theta $$
对于极大似然估计量
$$\hat{\theta}=\max_{1\leq i\leq n}\{X_{i}\}$$
如何研究$E[\hat\theta]$ 
能看出来 实际上$\hat\theta$ 是一个顺序统计量 因此我们是能够给出他的密度函数的 所以我们可以从此计算其期望
密度函数为
$$\left.f_{\max}(z)=\left\{\begin{matrix}n\frac{1}{\theta}\bigg(\int_0^z\frac{1}{\theta}\mathrm{d}z\bigg)^{n-1},0<z<\theta\\0,&\textbf{其它}\end{matrix}\right.\right.=\left\{\begin{matrix}n\frac{1}{\theta}\bigg(\frac{z}{\theta}\bigg)^{n-1},0<z<\theta\\0,&\textbf{其它}\end{matrix}\right.$$
计算期望有
$$\begin{array}{c}{E(\hat{\theta}_{L})=\int_{-\infty}^{\infty}z\cdot f_{\max}(z)\mathrm{d}z}\\{=\int_{0}^{\theta}z\cdot n\frac{1}{\theta}\biggl(\frac{z}{\theta}\biggr)^{n-1}\mathrm{d}z=\frac{n}{n+1}\theta<\theta}\\\end{array}$$
实际上这是一个有偏估计量 可以通过修正系数的方法转换为无偏估计量

#### 2
前面都只解释了如何判断一个量是否是无偏估计量 现在解释如何构造一个无偏估计量

设随机变量$\xi$服从二项分布
$$
\left.P\left(\begin{array}{c}{\xi=x}\\\end{array}\right.\right)=\left(\begin{array}{c}{n}\\{x}\\\end{array}\right)\theta^{x}\left(\begin{array}{c}{1-\theta}\\\end{array}\right)^{n-x},x=0\:,1\:,\cdots 
$$
试求 $\theta^2$ 的无偏估计量

我们不可能凭空想到一个无偏的估计量，所以所有类似的问题都是从一些我们知道的估计量出发 使用一些计算性质去凑出无偏估计的估计量
我们知道
$$E\bar{\xi}=E\xi=n\theta,ES^{*2}=D\xi=n\theta(1-\theta)=n\theta-n\theta^{2}$$
这是最基础的估计量了 如何去凑$\theta^2$的形式呢？ 靠观察
$$E\frac{\vec{\xi}-S^{*2}}{n}=\frac{E\vec{\xi}-ES^{*2}}{n}=\frac{n\theta-(n\theta-n\theta^{2})}{n}=\theta^{2}$$
因此$\theta^2$的无偏估计量为
$$\frac{\vec{\xi}-S^{*2}}{n}$$
其他类似的问题我们也是使用这个思路进行求解 有时候题目里会给出一些估计量供我们参考

### 均方误差
求正态总体$N(\mu,\sigma^2)$均值$\mu$和方差$\sigma^{2}$的MLE的均方误差研究均值 明显的我们知道MLE估计均值是无偏的 所以$bias=0$ $MSE=var(\hat\mu)=\frac{\sigma^2}{n}$ 
研究方差的MLE估计
$$\begin{gathered}
b(\theta,\hat{\sigma}^{2})=E(\hat{\sigma}^{2})-\sigma^{2}=-\frac{\sigma^{2}}{n}, \\
Var(\hat{\sigma}^{2})=Var\biggl(\frac{(n-1)S_{n}^{*2}}{n}\biggr)=Var\biggl(\frac{(n-1)S_{n}^{*2}}{\sigma^{2}}\frac{\sigma^{2}}{n}\biggr) \\
=\frac{\sigma^{4}}{n^{2}}Var\Bigg(\frac{(n-1)S_{n}^{*2}}{\sigma^{2}}\Bigg)=\frac{\sigma^{4}}{n^{2}}2(n-1) 
\end{gathered}$$
所以
$$MSE=Var(\hat{\sigma}^2)+b^2(\theta,\hat{\sigma}^2)=\frac{\sigma^4(2n-1)}{n^2}.$$
研究无偏的修正估计
$$\begin{aligned}
&MSE=Var\Bigg(\frac{(n-1)S_{n}^{*2}}{\sigma^{2}}\frac{\sigma^{2}}{n-1}\Bigg)-0=\frac{\sigma^{4}2(n-1)}{\left(n-1\right)^{2}}
\end{aligned}$$
能看出 有偏的MLE估计有着更小的MSE

## 数理统计Fisher信息量例题
计算指数分布的Fisher信息量
$$p(x;\theta)=\frac{1}{\theta}\exp\left\{-\frac{x}{\theta}\right\},\quad x>0,\theta>0$$
没说就按照只抽取一个样本进行计算
$$\frac{\partial}{\partial\theta}\ln p(x;\theta)=-\frac{1}{\theta}+\frac{x}{\theta^{2}}=\frac{x-\theta}{\theta^{2}}$$
则有
$$I(\theta)=E\Bigg(\frac{x-\theta}{\theta^2}\Bigg)^2=\frac{\mathrm{Var}(x)}{\theta^4}=\frac{1}{\theta^2}$$
## 数理统计CR不等式例题
设$\xi_1,\xi_2,\cdots,\xi_n$为取自正态母体$N(\mu,\sigma^2)$的一个子样. 试证 
$(1)\hat{\mu}=\overline{\xi}$是$\mu$的一个有效估计；
(2)若$\mu$已知，则$S_1^2=\frac1n\sum_{i=1}^n(\xi_i-\mu)^2$是$\sigma^{2}$的有效估计
若$\mu$未知，则$S_{2}^{2}=\frac{1}{n-1}\sum_{i=1}^{n}(\xi_{i}-\overline{\xi})^{2}$不是$\sigma^{2}$的有效估计


对于均值估计 以前学习过的推论知
$$D\overline{\xi}=\frac{\sigma^2}n$$
使用推论计算Fisher信息量有
$$I(\mu)=\frac{1}{\sigma^{2}}$$
因此有效性容易证明

对于方差的估计
若$\mu$已知 研究无偏性
$$S_1^2=\frac1n\sum_{i=1}^n(\xi_i-\mu)^2$$
$$ES_{1}^{2}=\frac{1}{n}E(\xi_{i}-E\xi_{i})^{2}=\frac{1}{n}\sum_{i=1}^{n}D\xi_{i}=\sigma^{2}$$
确实无偏 有讨论有效估计的前提
根据推广的结论计算Fisher信息量最后计算CR下界得到
我们计算的是关于$\sigma^2$ 的信息量 求导要把$\sigma^2$ 看作整体
$$I(\mu)=\frac{1}{2\sigma^{4}}$$ 容易知道
$$\frac{1}{\sigma^{2}}\sum_{i=1}^{n}(\xi_{i}-\mu)^{2}\sim\chi^{2}(n)$$
其方差为$2n$ 
则有 
$$D(S_1^2)=\frac{2\sigma^4}n=\frac1{nI(\sigma^2)}$$
是有效估计

若$\mu$未知 我们知道是无偏的
Fisher信息量沿用上问的不变  但是$S^2$ 的分布发生了变化
$$\frac1{\sigma^2}\sum_{i=1}^n(\xi_i-\overline{\xi})^2=\frac{(n-1)S_2^2}{\sigma^2}\sim\chi^2(n-1)$$
所以有
$$D(S_2^2)=\frac{2\sigma^4}{n-1}\neq\frac1{nI(\sigma^2)}$$
不是有效估计 但是是渐进有效估计

## 数理统计充分统计量例题
设总体 $x$ 服从两点分布$B(1,p)$,即$P\left(\mathrm{X=x}\right)=p^{x}\left(1-p\right)^{1-x},x=0,1$,其中 $0<p<1$ $\quad(X_1,X_2,...,X_n)$ 为来自总体$X$一个样本， 研究统计量$\overline{X}=\frac1n\sum_{i=1}^nX_i$ 的充分性

这里采用的最基础的定义进行验证 
先着手研究一下统计量分布 容易知道
$$n\overline{X}=\sum_{i=1}^{n}X_{i}-B(n,p),$$
所以条件可以取为 $\overline{X}=\frac{k}{n}$   这就是统计量取某个特定值
研究样本的条件分布
$$\begin{gathered}
P\left(X_{1}=x_{1},X_{2}=x_{2},\cdots,X_{n}=x_{n}\left|\overline{X}=\frac{k}{n}\right)\right.  \\
=\frac{P\left(X_{1}=x_{1},X_{2}=x_{2},\cdots,X_{n}=x_{n},\overline{X}=\frac{k}{n}\right)}{P\left(\overline{X}=\frac{k}{n}\right)} 
\end{gathered}$$
$$=\begin{cases}\dfrac{P(X_1=x_1,X_2=x_2,\cdots,X_n=x_n)}{P(n\overline{X}=k)},\text{如果}\sum_{i=1}^nx_i=k,\\0,\text{如果}\sum_{i=1}^nx_i\neq k,\end{cases}$$
带入分布的定义进行化简得到 
$$\left.=\left\{\begin{aligned}&\frac1{C_n^k},\text{如果}\sum_{i=1}^nx_i=k,\\&\textbf{0,如果}\sum_{i=1}^nx_i\neq k,\end{aligned}\right.\right.$$
明显的 我们最后的结果和分布中的参数$p$ 无关 所以$\overline{X}$ 是参数$p$ 的充分统计量

## 数理统计完备统计量例题
设$X_1,X_2,...,X_n$是来自两点分布$B(1,p)$的样本。由前面的例题知$\overline{X}=\frac1n\sum_{i=1}^nX_i$ 是的$p$ 充分统计量。下面验证$\overline{X}$ 也是完备统计量
容易给出$\overline{X}$ 的分布律 
$$P\left\{\overline{X}=\frac kn\right\}=C_n^kp^k\left(1-p\right)^{n-k}$$
假设存在$g(X)$  满足前面的要求（使用了期望的函数结论）
$$E_p[g(X)]=\sum_{k=0}^ng{\left(\frac kn\right)}C_n^kp^k(1-p)^{n-k}=0$$
等价于
$$(1-p)^n\sum_{k=0}^ng{\left(\frac kn\right)}C_n^k{\left(\frac p{1-p}\right)}^k=0$$
等价于
$$\sum_{k=0}^ng(\frac kn)C_n^k\left(\frac p{1-p}\right)^k=0$$
能看出 想要满足这个等式成立 就需要
$$g\left(\frac kn\right)=0$$
这就是满足了前面的要求

## 数理统计UMVUE
### 1
设两点分布总体为$p(x;\theta)=p^{\mathrm{x}}(1-p)^{1-x},x=0,1$求$p$的UMVUE
容易知道
$$
\overline{X}=\frac{1}{n}\sum_{i=1}^{n}X_{i}\text{}.
$$
是参数$p$的充分完备统计量
$$\overline{X}=\frac{1}{n}\sum_{i=1}^{n}X_{i},$$
是参数$p$ 的一个无偏估计
因此这个估计是唯一的UMVUE
### 2
求泊松分布总体参数$\lambda$的UMVUE
$$p(k,\lambda)=\frac{\lambda^{k}}{k!}e^{-\lambda}$$
容易知道 
$$T(X_{1},X_{2},\cdots,X_{n})=\sum_{i=1}^{n}X_{i}.$$
是参数$\lambda$的**充分完备统计量**
并且我们知道
$$\bar{X}=\frac{1}{n}\sum_{i=1}^{n}X_{i}$$
是参数$\lambda$的无偏估计量
能看出这个无偏估计量是充分完备统计量的函数 所以他是UMVUE
### 3
总体$X$ 的密度函数为
$$p(x)=\left\{\begin{array}{ll}
\lambda e^{-\lambda x}, & x\ge0, \\
0, & \text { 其他. }
\end{array}\right.$$
求$\frac{1}{\lambda}$的UMVUE
根据指数分布族的理论我们知道 $\sum\limits X_i$ 是一个充分完备统计量
又因为 $\overline{X}$ 是一个待求参数的无偏估计 因此他就是一个UMVUE
### 4
设二项分布总体为$X\sim B(n,p),$ 求${p}(1-p)$ 的UMVUE
根据指数分布族的相关理论 我们知道$X=\sum\limits X_i$ 是一个充分完备统计量 
记$\overline{X}=\frac1n\sum_{i=1}^nX_i=\frac1nX$, 则$X$服从二项分布$B(n,p).$ 并且
**为什么要这么想？我们知道$\overline{X}$是$p$的无偏估计量，所以希望从这里构造出新的估计量**
$$E(\frac Xn(1-\frac Xn))\color{}{=\frac{n-1}np(1-p)}$$
所以
$$\varphi(\bar{X})=\frac{n}{n-1}\frac{X}{n}(1-\frac{X}{n})=\frac{n}{n-1}\bar{X}(1-\bar{X})$$
是一个$p(1-p)$的无偏估计量
能看出它是充分统计量$\overline{X}$ （两点分布的均值）的函数 因此是UMVUE
### 5
设总体$X$在$[0,\theta]$上服从均匀分布，其中$\theta$是未知参数，$X_1,X_2,\cdots,X_n$是来自总体的样本， 求参数$\theta$的UMVUE
$$\begin{aligned}p(x_1,x_2,\cdots,x_n;\theta)&=\begin{cases}\frac1{\theta^n},&\mathbf{0}\leq x_{(1)}\leq x_{(n)}\leq\theta,\\0,&\mathrm{otherwise}.&\end{cases}\\&=\frac1{\theta^n}I_{_{(X_{(n)}\leq\theta)}}I_{_{(X_{(1)}\geq0)}}\end{aligned}$$
根据因子分解定理知
$$X{_{(n)}}=\max\{x_{1},x_{2},\cdots,x_{n}\}$$
是一个参数的充分统计量 能证明它也是完备的，证明方法是用定义；从函数期望为0去推导$g(X)$ 为0 就可以了 需要一步对积分上限函数求导
$$\int_{0}^{\theta}[g(X)]{\cdot}n\frac{x^{n-1}}{\theta^{n}}dx=0$$
所以
$$[g(X)]\theta^{n-1}=0$$
因此
$$g(X)=0$$
是完备统计量 证毕
能得到它的无偏估计
$$\hat{\theta}=\frac{(n+1)}nX_{(n)}$$
是充分完备统计量的函数 所以是UMVUE
### 6
设总体$X$服从正态分布$N(\mu,\sigma^2)$, $\theta=(\mu,\sigma^2)$未知，$X_1,X_2,\cdots,X_n$是来自总体的样本. 求参数$\mu$和$\sigma^{2}$的$UMVUE.$ 并且验证参数的UMVUE的方差是否达到了C-R下界 
实际上没有达到CR下界
