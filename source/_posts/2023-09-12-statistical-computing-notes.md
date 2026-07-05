---
title: "统计计算学习笔记"
title_en: "Statistical Computing Notes"
date: 2023-09-12 18:43:25 +0800
categories: ["Data Science & Statistics", "Statistical Modeling & Inference"]
tags: ["Learning Notes", "Statistics", "Statistical Computing", "Simulation"]
author: Hyacehila
excerpt: "一篇统计计算学习笔记，整理随机数生成、随机变量模拟、蒙特卡洛方法、抽样算法和数值计算相关内容。"
excerpt_en: "A study note on statistical computing, covering random number generation, random variable simulation, Monte Carlo methods, sampling algorithms, and numerical computation."
mathjax: true
hidden: true
permalink: '/blog/2023/09/12/statistical-computing-notes/'
---
## 模拟随机变量
### 01区间均匀分布随机数的生成
更大的区间 更不同的分布的所有随机数生成都建立在01区间均匀分布随机数上
产生均匀分布随机数的方法很多  我们一般要求他们具有以下的性质
* 具有分布的均匀性
* 周期长 不会很快出现循环
* 计算简单 生成随机数不需要耗费过高的运算资源
生成随机数有三个基本方法 分别是使用随机数表（基本已经淘汰） 研究随机的物理过程（仅在一些小众场合使用） 数学递归的随机数生成手段
数学的递归手段是一种伪随机 他在经历过一定的运算后 一定会出现退化为0或者循坏 
#### 自然取中法
也叫做平方取中法 核心在于将基础的种子平方后取出中间的N位 然后除以$10^{N}$ 实现到01的区间的随机数

#### 倍积取中法
自然取中法的衍生，将原本的平方取中更改为了成固定数后取中 如下
取定$位数N=4;倍积K=5678;w_{0}=1234$
$Kw_{0}$为自然生成数，取其中的四位然后除以$10^{N}$ 实现到01的区间的随机数；
将取出的N位作为迭代的下一个种子 重复倍积取中
#### 一阶线性同余法
它的迭代公式为
$$w_{n}=k_{1}*w_{n-1}~~(modm)$$
其中$k_{1},m$ 是迭代给定的数字 $w_{0}$ 是给出的种子 迭代产生的序列之需要除以$m$就可以回归目标的区间
自然可以推广出更加高阶的线性同余法
$$w_{n}=k_{1}*w_{n-1}+k_{2}*w_{n-2}~~~~(modm)$$
### 逆变换法（The Inverse Transform Method）
  分布函数一定是单调的 我们定义分布函数的广义逆如下
$$F_{X}^{-1}(u)=\inf\{x;F_{X}(x)\geqslant u\},0\leqslant u\leqslant1.$$
我们很容易给出一个定理 他是我们逆变换法生成随机变量的基础
$$\text{定理:设 }U{\sim}U(0\text{,1),则}Y{=}F_X^{-1}(U)\text{的分布函数为}F_X(x).$$

离散型可能还复杂一下 我们这里给一点介绍
按照以下方式抽样 就可以满足我们的要求
$$X=\begin{cases}x_0,U<p_0,\\x_1,p_0\leqslant U<p_0+p_1,\\\vdots\\x_j,\sum_{i=0}^{j-1}p_i\leqslant U<\sum_{i=0}^{j}p_i,\\\vdots\end{cases}$$
#### 普通的离散型随机变量
我们先来介绍分布列很容易写出的离散型随机变量的模拟 
对于这种只有有限个可能的离散型随机变量，每一个值都有对应的概率 我们可以直接使用$sample$函数进行抽样

#### 无穷个取值可能的离散型随机分布
几何分布 二项分布都是这样的 
对于这种情况 我们可以借助无限循环来处理 不断的累加概率 直到满足条件

#### 连续性随机变量
直接按照我们给出的定理就可以处理
核心在于怎么求出反函数
### 舍选法（The Acceptance-Rejection Method）
逆变换需要研究广义的反函数 在很多的情况下不好处理（不如无法找到解析的分布函数，分布函数不好找广义逆） 这就是舍选法使用的目的
如果需要模拟的随机变量$X$的密度函数$f(x)$比较复杂 那么我们可以先找到一个和X同取值的随机变量$Y$ 他有着密度函数$g(x)$ 以$f(Y)/g(Y)$成比例的接受模拟到的值 从而实现对$X$的模拟
这个算法的直接原理并不难以理解 后面的问题是如何操作 

#### 算法原理
非常明显 我们接受的概率并不可能大于1 这意味着我们需要保证分子$f(Y)$更小 
因此我们首先要求 存在常数$C$满足 也就是放大辅助概率密度函数
$$\frac{f\left(t\right)}{g\left(t\right)}\leqslant C,\forall t\left(\text{使得 }f\left(t\right)>0\right)$$
算法过程
1. 从$g(x)$中产生随机数$Y$
2. 从$(0,1)$上产生随机数$U$
3. 如果$U<\frac{f(Y)}{Cg(Y)}$ 则接受$Y$ 否则拒绝 然后返回第一步继续生成随机数

算法原理
某个抽选到的值被接受的概率为
$$\begin{aligned}
p\left(\text{accept}\mid Y=y\right)& =p\Big(U<\frac{f(Y)}{Cg(Y)}|Y=y\Big)  \\
&=p\Big(U<\frac{f(y)}{Cg\left(y\right)}\Big)=\frac{f(y)}{Cg\left(y\right)}.
\end{aligned}$$
那么被接受的概率可以使用全概率公式计算
$$p(\mathrm{accept})=\sum_{y}p\left(\mathrm{accept}\mid y\right)p(Y=y)=\sum_{y}\frac{f\left(y\right)}{\mathrm{Cg}\left(y\right)}g(y)=\frac{1}{\mathrm{C}}.$$
最后我们可以知道每个被接受的值对应的概率就是符合密度函数的
$$\begin{aligned}
p(k\mid\mathrm{accepted})& =\frac{p(\operatorname{accepted}\mid k)\operatorname{g}(k)}{p(\operatorname{accepted})}  \\
&=\frac{\left[f(k)/(\operatorname{Cg}(k))\right]g(k)}{1/C} \\
&=f(k).
\end{aligned}$$
这就是整个舍选法的工作原理
连续型的推导略去

#### 注意事项
对于舍选法 选取我们的辅助函数$g(x)$是非常重要的
如果想要我们的抽样更加有效 也就是被抽出的值更加容易被接受 我们需要尽量控制$C$的大小 
这意味着在选取辅助函数的时候 我们不仅需要结合前面学过的各种常见分布 找到取值范围一致的分布 并根据下面的方法计算$C$
$$C=\max\left\{\frac{f(x)}{g(x)}\colon x>0\right\}$$
这个式子的原理并不难以理解 注意 我们有时候可以适当放大C 如果不好算
对于辅助函数的选取 可以首先找到一个目标区间可积连续函数$t(x)$ 然后对这个函数进行归一化 也就是除以其在目标函数上的积分

### 合成法（The Composition Approach）
#### 条件概率合成
合成法的思想在于 目标函数的概率分布（包括分布列形式和函数形式）$p(x)$不好进行抽样 但是条件分布 $p(x|y)$ 和 条件$g(y)$ 容易确定  那么我们可以借助条件进行抽样工作
$$p(x)=p(x|y)g(y)$$
事实上 条件抽样工作的进行和前面并没有区别 采用什么方法都可以
目标的抽样工作需要在条件抽样完成后 根据条件抽样结果的不同而产生结果的改变 但是本质还是一次普通的抽样
#### 混合分布
事实上对于混合正态模拟我们往往采用合成法进行 他是我们下面要介绍的内容的一种
对于符合以下形式的分布
$$p\{X=j\}=\alpha p_j^{(1)}+(1-\alpha)p_j^{(2)}.$$
$$f(x)=\alpha f_1(x)+(1-\alpha)f_2(x).$$
这样的随机变量可以用下面的形式得到
$$\left.X=\left\{\begin{array}{ll}X_1,&\quad\text{以概率}\alpha\\X_2,&\quad\text{以概率}1-\alpha\end{array}\right.\right.$$
这种问题可以直接进行推广到更多初选可能的情况
可以理解为每一个函数对最终函数的影响份额
$$F(x)=\sum\limits P_{i}F_{i}(x).\text{其中}P_{i}\ge0 \sum\limits P_{i}=1$$
分布函数也是一样的
是我们在其他书籍中见过的混合分布 也是很重要的一类分布
#### 随机变量的函数合成
有时候我们需要研究的随机变量是另一个随机变量的函数  比如
$$e^{X};X+Y,UV$$
此时我们可以先从原始随机变量里面进行抽取 
然后通过函数运算得到我们需要的随机数

### 变换法
非常多的分布可以使用其他的分布形式变换得到 这也可以用于我们的抽样 
下面给出几个例子
$$\begin{aligned}&1.\text{ 若}Z\sim N(0,1),\text{则}V=Z^2\sim\chi^2(1).\\&2.\text{ 若}U\sim\chi^2(m),V\sim\chi^2(n),\text{ 则}F=\frac{U/m}{V/n}\sim F(m,n).\\&\text{3. 若}Z\sim N(0,1),V\sim\chi^2(n),\text{ 则}T=\frac Z{\sqrt{V/n}}\sim t(n).\end{aligned}$$
除此以外 用来得到正态分布本身也可以通过Box Muller变换来实现
有的分布可以通过绝对值变换后进行抽样 然后再恢复绝对值效果 对于某些分布区间在$(-\infty,\infty)$ 的分布比较常用
#### Box Muller变换
一个很常用的变换手段 可以生成正态分布随机数 这里介绍一下

## 条件期望与条件方差
条件期望与方差是为了解决随机向量产生后我们需要研究的一些问题，除去那些独立的量以外，具有相关性的多个随机变量在某种条件下的期望与方差也值得我们的研究；在这里我们补充一些概率论中未能叙述的内容
条件期望实际上还是一个随机变量，是关于条件变动的随机变量，通过不断的变动条件，我们就能分治原本的空间
我们在这里需要理解的是，全期望公式（双重期望公式）是在条件期望被提出之后的非常重要的一环 ，就和我们以前研究过的全概率公式一样 我们现在可以用一个量对原有的样本空间进行划分 之后再去进行求期望 这种分治的思想是这个公式的核心

为了进行分治 我们肯定需要一种先后或者划分条件，下面的例子是用来帮助我们理解这种思想的
理论研究在概率论中已经有足够的介绍了
### 条件期望

#### 矿工问题
$$\begin{gathered}
\text{一名矿工被困在有三个门的矿井里,第一个门通往一个坑道,} \\
\text{沿此坑道走3个小时可以到达安全地点;第二个门会使他走5个小时后又} \\
\text{回到原处;第三个门会使他走7个小时后也回到原处。假定该矿工在任何} \\
\text{时刻等可能地选定其中一个门. 求他到达安全地点平均需要多长时间}? 
\end{gathered}$$
设X为矿工到达安全地点所需时间Y 表示所选的门,则
$$\begin{aligned}
E[X]& =E[E[X|Y]]  \\
&=E[X|Y=1]P\{Y=1\}+E[X|Y=2]P\{Y=2\} \\
&+E[X|Y=3]P\{Y=3\} \\
\text{又}E[X|Y=1]& \begin{aligned}=3,E[X|Y=2]=5+E[X],E[X|Y=3]=7+E[X]\end{aligned} 
\end{aligned}$$
带入进行求解得到 平均时间为15小时

#### 几何分布的均值与方差
$\begin{aligned}\text{若第一次成功,则令}Y&=1\text{,否则令}Y=0..\end{aligned}$
${可知}E(X|Y=1)=1\text{,}E(X|Y=0)=E(1+X)=1+E(X)$
双重期望公式得
$$\begin{aligned}
E[X]& =E[E[X|Y]]  \\
&=E[X|Y=1]P\{Y=1\}+E[X|Y=0]P\{Y=0\} \\
&=p+E(X+1)(1-p) \\
&=p+[E(X)+1](1-p)
\end{aligned}$$
解方程就可以得到均值 $E[X]=\frac{1}{p}$
对于处理方差的问题 我们通过研究$E[X^{2}]$来研究它
$$\begin{aligned}
E[X^{2}]& =E[E[X^2|Y]]  \\
&=E[X^2|Y=1]P\{Y=1\}+E[X^2|Y=0]P\{Y=0\} \\
&=p+E(X+1)^2(1-p) \\
&=p+[E(X^2)+2E(X)+1](1-p) \\
&=p+(1-p)E(X^2)+2\frac{1-p}p+(1-p)
\end{aligned}$$
解方程就可以得到$E[X^{2}]=\frac{2-p}{p^{2}}$ 套公式运算就能得到方差
使用 $Var(X)=E[X^{2}]-E[X]^{2}$ 

#### 求解二元正态分布的相关系数
$$\begin{aligned}f(x,y)&=\frac{1}{2\pi\sigma_x\sigma_y\sqrt{1-\rho^2}}\exp\Big(-\frac{1}{2(1-\rho^2)}\Big[\Big(\frac{x-\mu_x}{\sigma_x}\Big)^2\\&-2\rho\frac{(x-\mu_x)(y-\mu_y)}{\sigma_x\sigma_y}+\Big(\frac{y-\mu_y}{\sigma_y}\Big)^2\Big]\Big).\end{aligned}$$
我们需要证明他的相关系数是$\rho$
$$\mathrm{Corr}(X,Y)=\frac{\mathrm{Cov}(X,Y)}{\sigma_x\sigma_y}=\frac{\mathrm{E}(XY)-\mu_x\mu_y}{\sigma_x\sigma_y}$$
求解相关系数的核心在于${E}(XY)$
我们还是通过全期望公式的思想 先确定变量中的一个来简化原始问题
$$\begin{aligned}
E(XY\mid Y=y)& =E(Xy\mid Y=y)  \\
&=yE(X\mid Y=y) \\
&=y\left[\mu_x+\rho\frac{\sigma_x}{\sigma_y}(y-\mu_y)\right] \\
&=\left.y\mu_x+\rho\frac{\sigma_x}{\sigma_y}(y^2-\mu_yy)\right.
\end{aligned}$$
条件密度函数也是正态分布 所以条件均值容易求解
借助全期望公式得
$$\begin{aligned}
E(XY)& =E[E(XY\mid Y)]  \\
&=E[Y_{\mu_{x}}+\rho\frac{\sigma_{x}}{\sigma_{y}}(Y^{2}-\mu_{y}Y)] \\
&=\mu_{x}E(Y)+\rho\frac{\sigma_{x}}{\sigma_{y}}E(Y^{2}-\mu_{y}Y) \\
&=\mu_x\mu_y+\rho\frac{\sigma_x}{\sigma_y}\mathrm{Var}(Y) \\
&=\mu_x\mu_y+\rho\sigma_x\sigma_y.
\end{aligned}$$
这就是前面问题的处理思路 要理解我们这里使用的处理方法

### 全期望公式反推全概率公式
通过引入示性随机变量X 满足
$$\left.X=\left\{\begin{array}{ll}1&\quad\text{若}A\text{ 发生}\\0&\quad\text{若}A\text{ 不发生}\end{array}\right.\right.$$
我们容易知道 $E(X)=P(A)$
再进行全期望公式的处理有$P(A)=E(X)=E(E(X|Y))$
因此有
$$\left.P(A)=\left\{\begin{array}{l}\sum_yE(X|Y=y)P(Y=y)=\sum_yP(A|Y=y)P(Y=y)\\\int E(X|Y=y)f_Y(y)dy=\int P(A|Y=y)f_Y(y)dy\end{array}\right.\right.$$
如果我们在前面随机变量的基础上定义事件，那么这就是全概率公式，前面提到过的分治思想的体现

#### 一个经典的例子
$X~Y$ 是独立的随机变量
$$\begin{aligned}
P\{X<Y\}& =\int P\{X<Y|Y=y\}f_Y(y)dy  \\
&=\int P\{X<y|Y=y\}f_Y(y)dy \\
&=\int P\{X<y\}f_{Y}(y)dy \\
&=\int F_X(y)f_Y(y)dy
\end{aligned}$$
这样我们就可以取代那些繁杂的积分运算 从而快速的得到结论；
卷积公式等很多类似的问题都可以这样进行处理
在使用这种方式的时候 要搞明白符合的意义
前者是$X$自己单独的分布函数带入$y$ 后者是$y$的密度函数

#### 另一个有趣的例子
令$U_{1}~U_{2}...$是一个独立均匀分布序列 他们均服从$(0,1)$上的均匀分布 令$N = min\{n:\sum\limits_{k=0}^{k=n}U_{k}>1\}$  求$E(N)$
考虑更加一般的情况 我们给出$N(x)$表示随机数和超过$x$的个数
令$m(x)=E(N(x))$ 
容易得到 根据全期望公式
$$m(x)=E\big[N(x)\big]=\int_0^1E\big[N(x)\big|U_1=y\big]\mathrm{d}y.$$
$$E[N(x)\mid U_1=y]=\begin{cases}1,~~~~~~~~~~~~~~~~~~~~y>x;\\1+m(x-y),y\leqslant x.\end{cases}$$
所以
$$\begin{aligned}
m\left(x\right)& =\int_0^x[1+m(x-y)]\mathrm{d}y+\int_x^1\mathrm{d}y  \\
&=1+\int_0^xm(x-y)\operatorname{d}y \\
&=1+\int_0^xm(u)\mathrm{d}u.\quad(\text{令 }u=x-y)
\end{aligned}$$
两边求导得到
$$m'(x)=m(x),\text{即}\quad\frac{m'(x)}{m(x)}=1.$$
处理微分方程并且确定C后得到
$$m(x)=e^{x}$$

### 条件方差
条件方差的定义 $Var(Y|X)=E\Big[(Y-E[Y|X])^2|X\Big]$
统计方差公式 $Var[Y]=E[Var(Y|X)]+Var[E(Y|X)]$

**在整个条件概率，条件期望，条件方差这一部分中，离散型的最好处理方法还是把分布列写出来，根据要求进行书写，而不是使用前面给出的这些公式，前面的这么多方法更多的被用在连续型随机变量和那些分布列不容易写出的随机变量中，毕竟分布列是离散型的核心，是研究一切离散型随机变量的根基**

对于条件概率的问题 需要知道根据条件概率公式只是一般针对连续性的处理方法 对于离散型分布而言 我们在已知分布列的情况下 只需要计算条件发生的概率 再计算事件发生的概率 最后作比就可以了 对于条件方差问题 我们需要处理的问题并不发生本质变化 还是对公式的套用（使用最基础的方差公式运算 而不是条件方差公式） 耐心才是处理这一类问题中最需要的东西 离散型的问题通常繁琐但是不困难

## 有效抽样次数
在这里我们简单的研究一下 如果想要达到预设的精度 我们需要进行多少抽样的问题 我们在这仅仅研究对数学期望 也就是一阶原点矩的估计问题 更复杂的问题留到以后再处理
本质上有效抽样次数这一小章是对数理统计中有效性的分析 后面的方差减小技术也是如此

### 有效抽样次数的确定
假设$X_1,X_2,\cdots,X_n$ 是一组独立同分布的样本 他有均值$\mu$和方差$\sigma^{2}$ 
我们可以估计得到
$$E(\overline{X})=\mu,\quad\text{并且}Var(\overline{X})=E[(\overline{X}-E(\overline{X}))^2]=\frac{\sigma^2}n.$$
非常明显的 我们知道估计$\overline{X}$是无偏的 当$\frac{\sigma^2}n.$ 越小的时候 估计越有效 这意味着我们的估计均方误差较小 均方误差较小已经成为了现代数据分析中对估计要求的最重要一点
对于样本方差$\sigma^{2}$ 我们使用无偏估计形式
$$S^2=\frac{1}{n-1}\sum_{i=1}^{n}(X_i-\overline{X})^2$$
有了衡量偏差大小的手段 我们进行有效抽样次数的研究就很容易了
1. 确定适当的小正数$d$作为估计量的标准误差
2. 至少抽取100个样本
3. 连续抽取新的样本 直到$\frac{S}{\sqrt{n}}\leq d$
4. 得到估计量$\overline{X}$

整个算法的设计非常自然并且容易进行

### 改进的有效抽样次数确定
每次计算方差$S^{2}$来验算是否满足不等式是计算量非常巨大的 尤其是抽样数量上升后 这里给出迭代公式来减少计算量的需求
$$\left.\overline{X_{j+1}}=\overline{X_j}+\frac{X_{j+1}-\overline{X_j}}{j+1},S_{j+1}^2=(1-\frac1j)S_j^2+(j+1)(\overline{X_{j+1}}-\overline{X_j})^2.\right.$$
使用迭代公式可以避免每次重新计算样本方差
可以减少计算资源的消耗

对于样本方差是样本期望的函数的情况（比如01分布）
我们应该在迭代过程中计算样本均值用函数得到样本方差
直接计算方差对计算资源的消耗是比较大的



## 方差缩减技术
Monte Carlo 方法是一大类方法  他不仅仅包括Monte Carlo积分中介绍的方法，整个基于随机模拟实现的概率问题解决方法都属于Monte Carlo方法的一种
我们可以大致把Monte Carlo方法处理的问题分为两大类
* 需要解决的问题本身存在随机性，比如核物理和分子运动模拟
* 需要解决的问题可以转化为某种分布的特征数 比如Monte Carlo积分
在第二类问题中 我们实际上在构造统计量（估计量）进行问题的研究，确定估计的诸如无偏性 有效性是比较重要的 他们可以评估我们的估计设计是否合理
在无偏的基础上 均方误差等价于方差 方差减少技术这一章节就是在研究方法希望减少最后估计量的方差可以试用的方法 提升估计的有效性

我们可以综合使用后面介绍的方法 同时使用多种 这样能起到更好的降低方差的作用

定义 方差的缩减率
$$\frac{\mathrm{Var}(\hat{\theta})-\mathrm{Var}(\hat{\theta}_{\epsilon}\cdot)}{\mathrm{Var}(\hat{\theta})}$$
### 随机投点和样本均值法的有效性比较
对于一个非常标准的定积分问题
$$\theta=\int_{a}^{b}f\left(x\right)\mathrm{d}x.$$
随机投点法给出的估计是
$$\hat{\theta}_1=\hat{\theta}=M(b-a)\frac{n_0}n.$$
$n_{0}$符合两点分布 所以计算方差得到
$$\mathrm{Var}(\hat{\theta}_{1})=\frac{M^{2}(b-a)^{2}}{n^{2}}\mathrm{Var}(n_{0})=\frac{\theta}{n}\bigl[M(b-a)-\theta\bigr].$$
类似的原理能得到样本均值法的估计和方差
$$\hat{\theta}_{2}=\tilde{\theta}=\frac{1}{n}\sum_{i=1}^{n}\frac{f(X_{i})}{g(X_{i})}=\frac{b-a}{n}\sum_{i=1}^{n}f(X_{i}).$$
$$\begin{aligned}
\operatorname{Var}(\hat{\theta}_{2})& =\text{Var}\biggl[\frac1n(b-a)\sum_{i=1}^nf(x_i)\biggr]  \\
&=\frac1n\bigg[\left.(b-a)^2\int_a^bf^2\left(x\right)\frac1{b-a}\mathrm{d}x-\theta^2\right] \\
&=\frac1n\bigg[\left.(b-a)\int_a^bf^2\left(x\right)\mathrm{d}x-\theta^2\right].
\end{aligned}$$
作差比较能得到 样本均值法的方差更小 这意味着估计的有效性更好
是否能通过某些手段来继续提高有效性 这就是我们这后面要展开研究的问题

### 重要抽样法
#### 引入重要抽样法
对于样本均值法
$$\theta=\int_{a}^{b}\frac{f(x)}{g(x)}g(x)\mathrm{d}x=E_{\kappa}\bigg[\frac{f(X)}{g(X)}\bigg].$$
抽样并估计有
$$\tilde{\theta}=\frac1n\sum_{i=1}^n\frac{f(X_i)}{g(X_i)}.$$
直接计算方差有
$$\mathrm{Var}(\hat{\theta})=\frac{1}{n}\bigg[E_{\kappa}\Big(\frac{f(X)}{g(X)}\Big)^{2}-\theta^{2}\Big].$$
这个计算结果告诉我们 辅助抽样用分布$g(X)$和原始分布$f(X)$满足$g(x)=\frac{f(x)}{\theta}$ 时我们有最小的方差 当然$\theta$是未知的  这当然不可能 不过 我们可以让他们两者尽可能接近
这就是重要抽样法的思想 把原本使用的均匀抽样进行改变 让对估计更加有效的样本更多的出现 从而增加估计的有效性
现在我们给出一个简单的例子来说明我们猜测确实是有道理的
考虑积分$\theta=\int_{0}^{1}\mathrm{e}^{x}\mathrm{d}x.$ 采用两种MC方差进行尝试
对于样本均值法
$$\begin{aligned}
\mathrm{Var}(\hat{\theta})& \left.=\left.\frac1n\right[\int_0^1\mathrm{e}^{2x}\mathrm{d}x-(\mathrm{e}-1)^2\right]  \\
&=\frac1n\biggl[\frac12(\mathrm{~e}^2-1)-(\mathrm{~e}-1)^2\biggr] \\
&=\frac{0.242}n.
\end{aligned}$$
对于重要抽样法的改进 我们使用
原始函数的Taylor展开并且规范化的$\frac{2}{3}(1+x)=g(x)$
$$\begin{aligned}
Var(\bar{\theta})& =\frac{1}{n}\biggl[\int_{0}^{1}\frac{f^{2}(x)}{g(x)}\mathrm{d}x-(\mathrm{e}-1)^{2}\biggr]   \\
&=\frac1n\biggl[\int_0^1\frac32\frac{\mathrm{e}^{2x}}{1+x}\mathrm{d}x-(\mathrm{e}-1)^2\biggr] \\
&=\frac{0.0269}n\quad
\end{aligned}$$
重要抽样法确实起到了降低方差的作用
#### 重要抽样法和倾斜密度函数
##### 重要抽样法
在一般意义下 对于符合密度函数$f(x)$的随机变量$X$  对于估计
$$\theta=E_{f}\bigl[h(X)\bigr]=\int_{a}^{b}h\left(x\right)f(x)\mathrm{d}x.$$
重要抽样法便是从样本均值法中进行衍生 找到新的$X$的分布函数$g(x)$ 
$$\theta=\int_{a}^{b}\frac{h\left(x\right)f\left(x\right)}{g\left(x\right)}g\left(x\right)\mathrm{d}x=E_{k}\bigg[\frac{h\left(X\right)f\left(X\right)}{g\left(X\right)}\bigg].$$
在实际的模拟中
$$\hat{\theta}=\frac{1}{n}\sum_{i=1}^{n}\frac{h(X_{i})f(X_{i})}{g(X_{i})}$$
只要选取合适的$g(x)$就能实现减少方差的作用
##### 倾斜密度函数
对于原始分布的矩母函数
$$M(t)=E_{f}\bigl[\mathrm{e}^{tX}\bigr]=\int\mathrm{e}^{tx}f\left(x\right)\mathrm{d}x$$
我们构造如下形式称为原始分布的倾斜密度函数
$$f_t(x)=\frac{\mathrm{e}^{tx}f(x)}{M(t)}$$
要知道 倾斜密度函数本质上是一族函数 我们要选取合适的$t$ 才能确定需要的函数 怎么确定这个$t$呢 让它接近我们需要侧重抽样的位置

下面介绍几个比较特殊的分布的倾斜密度函数
如果$f(x)$是参数为$\lambda$的指数分布 则$f_{t}$是参数为$\lambda-t$的指数分布
如果$f(x)$是参数为$p$的二项分布 则$f_{t}$是参数为$\frac{pe^{t}}{pe^{t}+1-p}$的二项分布
如果$f(x)$是参数为$(\mu,\sigma^{2})$的正态分布 则$f_{t}$是参数为$(\mu+\sigma^{2}t,\sigma^{2})$的正态分布
只要会求矩母函数这里都很好计算
#### 重要抽样法与小概率事件模拟
我们这里给出一个例子 以后如果遇到类似的问题也可以进行类似的处理
$$X{\sim}N(0,1),\text{ 欲通过模拟方法估计 }\theta{=}P(X{\geqslant}20).$$
直接进行模拟  我们只能得到概率为0的奇怪的答案  这其实是因为这个概率实在是太低了 大概为$e^{-89}$这个数量级 所以我们直接模拟根本得不到需要的答案
事实上 原始问题等价于
$$\theta=E(I_{\{X\geqslant20\}})$$
没错 事实上也是一个研究期望的问题，只是从正态分布中抽样 然后大于20的当作1 剩下的作为0 然后对向量进行求期望的操作 
现在我们来使用重要抽样法修正这个抽样操作
原始的正态分布是$f(x)$ $h(x)=I_{\{X\geqslant20\}}$ $g(x)$是计算得到的倾斜密度函数
现在有
$$\begin{aligned}
&\theta =E_{f}[h(X)]=\int_{-\infty}^{\infty}h(x)f(x)\mathrm{d}x  \\
&=\int_{-\infty}^{\infty}\frac{h(x)f(x)}{g(x)}g(x)\mathrm{d}x \\
&=\int_{-\infty}^{\infty}I_{(x\geqslant20)}\left\{\frac{\dfrac{1}{\sqrt{2\pi}}\mathrm{e}^{-\frac{x^2}{2}}}{\dfrac{1}{\sqrt{2\pi}}\mathrm{e}^{-\frac{(x-\mu)^2}{2}}}\right\}\sqrt{2\pi}\mathrm{e}^{-\frac{(x-\mu)^2}{2}}\mathrm{d}x \\
&=\int_{-\infty}^{\infty}I_{(x\geqslant20)}\mathrm{e}^{-\mu x+\mu^{2}/2}\frac1{\sqrt{2\pi}}\mathrm{e}^{-\frac{(x-\mu)^{2}}2}\mathrm{d}x \\
&=E_{_{\mathcal{g}}}[I_{(X\geqslant20)}\mathrm{~e}^{-\mu X+{\mu}^{2}/2}],
\end{aligned}$$
我们只需要从新的倾斜密度函数中抽样 然后模拟计算均值就可以了
上机运行可以发现 使用重要抽样法以后 我们只需要使用少量的模拟次数就可以实现原本非常高的模拟次数就可以达到的效果

### 分层抽样法
我们这里着重介绍积分的MC方法的分层抽样 其他问题形式转化为这个问题形式进行解决
分层抽样法也是利用贡献率来降低估计方差的方法，我们一般先把整个区间分成若干部分 计算他的权重 然后分配抽样次数 借助这样的方法提高抽样的效率
#### 核心思路介绍
对于积分$\theta=\int_{0}^{1}f(x)\mathrm{d}x$ 我们把积分区间分成$m$个小区间 端点记作$a_{i}$ 则
$$\theta=\int_0^1f(x)\mathrm{d}x=\sum_{i=1}^m\int_{a_{i+1}}^{a_i}f(x)\mathrm{d}x=\sum_{i=1}^mI_i.$$
再记$l_{i}=a_{i}-a_{i-1}$ 使用样本均值方法就可以计算$I_{i}$ 然后就能计算我们需要的估计$\theta$ 并且有 $\tilde{\theta}_{3}=\sum_{i=1}^{in}\hat{I}_{i}.$
非常明显的 我们知道
$$\hat{E}I_i=I_i$$
$$\mathrm{Var}(\hat{\theta}_{3})=\mathrm{Var}\Big\{\sum_{i=1}^{m}\frac{l_{i}}{n_{i}}\sum_{j=1}^{n_{i}}f(X_{ij})\Big>=\sum_{i=1}^{m}\frac{l_{i}^{2}}{n_{i}}\sigma_{i}^{2}.$$
其中
$$\sigma_i^2=\int_{a_{i-1}}^{a_i}\frac{f^2\left(x\right)}{l_i}\mathrm{d}x-\left(\frac{I_i}{l_i}\right)^2.$$
#### 核心定理
在$\sigma^{2},l_{i},n$都已知的情况下 当进行如下的分配方式时候
$$\frac{n_i}n=\frac{l_i\sigma_i}{\sum_{j=1}^ml_j\sigma_j}$$
估计的方差最小 为$\frac{1}{n}\left[\sum_{j=1}^{m}l_{j}\sigma_{j}\right]^{2}.$
一些补充的证明过程
根据前文给出的结论
$$\mathrm{Var}(\hat{\theta}_{3})=\sum_{i=1}^{m}\frac{l_{i}^{2}}{n_{i}}\sigma_{i}^{2}=\frac{l_{1}^{2}}{n_{1}}\sigma_{1}^{2}+\frac{l_{2}^{2}}{n_{2}}\sigma_{2}^{2}+\cdots+\frac{l_{m-1}^{2}}{n_{m-1}}\sigma_{m-1}^{2}+\frac{l_{m}^{2}}{n-n_{1}-n_{2}-\cdots-n_{m-1}}\sigma_{m}^{2}.$$
针对每一个$n_{i}$ 求偏导研究极小化问题都有
$$\frac{l_1\sigma_1}{n_1}=\frac{l_m\sigma_m}{n-n_1-n_2-\cdots-n_{m-1}}.$$
根据等比性质有
$$\frac{l_1\sigma_1}{n_1}=\frac{l_2\sigma_2}{n_2}=\frac{l_3\sigma_3}{n_3}=\cdots=\frac{l_{m-1}\sigma_{m-1}}{n_{m-1}}=\frac{l_m\sigma_m}{n+n_1-n_2-\cdots-n_{m-1}}=\frac{\sum_{i=1}^ml_i\sigma_i}{n}$$
也就是此时取得方差最小化
带回各个量便能计算出最后的极小化方差

对于抽样区间划分的问题 执行区间等分是最简单并且常用的
关于抽样次数的分配问题 哪怕是根据区间长度进行等比例的分配 依然不会产生更大的方差

#### 总结
我们需要首先进行一次预抽样 根据预抽样的结果计算方差等数据 然后按照要求在预抽样的结果上进行抽样比例的分配 然后用新的比例继续使用均值法计算我们需要的估计就可以了
这里我们不继续推广方法的使用范围 如果希望使用分层抽样法 那么先把问题转化为Monte-Carlo积分问题就可以了
*连续函数的期望就是一个积分问题*

### 对偶变量法
#### 思路引入
如果我们对$\theta=E[X]$感兴趣 并且有$X_{1},X_{2}$同分布并且都有均值$\theta$ 那么我们能得到
$$\mathrm{Var}\Big(\frac{X_1+X_2}{2}\Big)=\frac{1}{4}\Big[\mathrm{Var}(X_1)+\mathrm{Var}(X_2)+2\mathrm{Cov}(X_1,X_2)\Big].$$
只要$X_{1},X_{2}$负相关 就可以得到更低的方差 但是 他仍然是无偏估计
$$E\Big(\frac{X_{1}+X_{2}}{2}\Big)=\theta.$$
这就是我们对偶变量法的核心操作思路 现在需要解决的就是构造一个这样的随机变量

#### 处理方法
我们先介绍两个需要用到的定理
定理：对于独立随机变量$X_{1},X_{2}...X_{n}$相互独立，对于一元增函数$f,g$ 则对于任意的$x\ge y$ 有
$$E[f(X)g(X)]\geqslant E[f(X)]E[g(X)].$$
推论：如果$h(x_{1},x_{2},\cdots,x_{n})$ 是他每个自变量的单调函数 则对随机数集合$U_{1},U_{2}...U_{n}$ 有
$$\mathrm{Cov}[h(U_1,U_2,\cdots,U_n),h(1-U_1,1-U_2,\cdots,1-U_n)]\leqslant0.$$

现在就可以用这样的定理尝试构造对偶变量了
假设$X_{1}=h(U_{1},U_{2}...U_{n})$ 其中$U_{i}$是独立随机数 那么我们构造$X_{2}=h(1-U_{1},1-U_{2}...1-U_{n})$ 容易知道$U,1-U$**同分布** 并且$\operatorname{Cov}(U,1-U)=-\frac{1}{12}.$
如果$h$是每个坐标的单调函数，那么$X_{1},X_{2}$ 同分布并且负相关 则称两者互为对偶变量
对偶变量法能够减少估计量的方差 并且可以减少前文分层抽样两次抽取随机数的性能开销
本质上就是样本均值法 只是在抽样上进行了改进
当然这个改进也是有代价的 我们不得不控制自己只能在 $[0,1]$区间上进行抽样
#### 举例说明
我们用一个例子说明对偶变量法的使用 以后来模仿这个例子的方法
用对偶变量法估计积分$\theta=\int_0^1\mathrm{e}^x\mathrm{d}x=\mathrm{e}-1.$
非常明显的 $h(x)=e^{x}$ 是一个一元单调函数 这个区间也符合我们进行对偶变量求解的要求
那么对于估计量$X_{1}$ 我们设计为$e^{U_{}}$  估计量$X_{2}$ 设计为 $e^{1-U_{}}$   所以这样构造符合我们使用对偶变量的要求
因此最后的估计量为
$$(\frac{e^{U}+e^{1-U}}{2})$$
我们人工抽取$U$ 通过对偶的方式得到另一部分随机数 最后组成我们的估计量 
对于类似的在这样的$(0,1)$区间 进行积分的问题都可以进行这样的处理
对偶变量使用的条件只有两个 积分区间$(0,1)$ 被积函数单调

对于一些别的情况 我们可以转化一下 满足使用对偶变量法求解的要求
计算以下积分
$$\phi\left(x\right)=\int_{-\infty}^{x}\frac{1}{\sqrt{2\pi}}\mathrm{e}^{-t^{2}/2}\mathrm{d}t.$$
这是一个无穷区间的积分问题 我们在蒙特卡洛方法刚开始就说了要考虑转换
非常明显的 如果 $x<0$ 我们可以考虑求 $[0,-x]$ 上的积分 然后和$\frac{1}{2}$ 作差
对于 $x>0$ 我们可以考虑求 $[0,x]$ 上的积分 然后加上$\frac{1}{2}$ 
也就是问题归结为
$$\Phi\left(x\right)=\int_{0}^{x}\frac{1}{\sqrt{2\pi}}\mathrm{e}^{-t^{2}/2}\mathrm{d}t.$$
然后我们作$y=\frac{t}{x}$ 的代换 这样就把积分区间缩放回到$[0,1]$ 
问题现在是求
$$\theta=\int_{0}^{1}xe^{-\left(xy\right)^{2}/2}dy.$$
非常明显的 这里满足对$y$单调和$[0,1]$区间的条件 可以使用对偶变量法
### 控制变量法
#### 理论介绍
这里我们直接使用期望的思路来处理，MC积分问题直接转化为期望进行研究 样本均值法就行

在前面 $\theta=E[h(X)]$ 的估计是$\frac{1}{n}\sum\limits h(X_{i})$  现在我们构造一个新的分布 $f(X)$ 并且$\mu=E[f(X)]$ 
现在我们构造$Y=h(X)+c[f(X)-\mu],$ 容易知道 $E(Y)=\theta$ 研究方差得到
$$\begin{aligned}
\text{Var(Y)}& =\operatorname{Var}\langle h(X)+c[f(X)-\mu]\rangle   \\
&=\operatorname{Var}[h\left(X\right)]+c^2\operatorname{Var}[f\left(X\right)-\mu]+2c\operatorname{Cov}\langle h\left(X\right),f(X)-\mu\rangle  \\
&=\operatorname{Var}[h(X)\operatorname{}+c^2\operatorname{Var}[f(X)\operatorname{}+2c\operatorname{Cov}[h(X)\operatorname{}f(X)\operatorname{}].
\end{aligned}$$
为了减少方差 我们继续化简
$$\begin{aligned}
\text{Var(Y)}& =c^2\operatorname{Var}[f(X)]+2c\operatorname{Cov}[h(X),f(X)]+\operatorname{Var}[h(X)]  \\
&=\operatorname{Var}[f(X)]\Big|c+\frac{\operatorname{Cov}[h(X),f(X)\Big]}{\operatorname{Var}[f(X)\Big]}\Big|^2+\operatorname{Var}[h(X)\Big] \\
&-\frac{\mathrm{Cov}^{2}\bigl[h\left(X\right),f\left(X\right)\bigr]}{\mathrm{Var}\bigl[f\left(X\right)\bigr]}.
\end{aligned}$$
因此我们已经找到了最后可以缩减方差的方法
$$c=c^*=-\frac{\operatorname{Cov}\left[h(X),f(X)\right]}{\operatorname{Var}\left[f(X)\right]}$$
现在问题已经转化为了这个具体的$c^{*}$ 该怎么计算；非常明显的，我们还是得先进行一个模拟 计算这里的方差和协方差 公式如下
$${\operatorname*{Cov}}\big[h(X),f(X)\big]=\frac{\sum\limits_{i=1}^n\{\big[h(x_i)-\bar{h}\big][f(x_i)-\mu\big]\}}{n-1},\\\\{\operatorname*{Var}}\big[f(X)\big]=\frac{\sum\limits_{i=1}^n\big[f(x_i)-u\big]^2}{n-1}.$$

这样的叙述基本已经够清楚了 能够处理的问题的范围也都已经介绍详细 
#### 例子
用控制变量法给出$\theta=E[e^{(U+V)^{2}}]$ 的估计 其中$U,V$是独立的$U(0,1)$ 控制变量选取为 $f=(U+V)^{2}$
这里我们一般会采用逐渐增加随机数的办法来实现模拟
我们需要在新的随机数生成的时候来重新计算 均值$\mu$ 和 控制数 $c$ 
下面我们用代码来叙述一下

### 条件期望法
#### 理论
条件期望公式知道
$$\operatorname{Var}(X)=E[\operatorname{Var}(X\mid Y)]+\operatorname{Var}[E(X\mid Y)],$$
那么由于非负性能确定
$$\mathrm{Var}(X)\geqslant\mathrm{Var}[E(X\mid Y)].$$
这么来看条件抽样比直接抽样更加有效 因为我们知道
$$E[E(X\mid Y)]=EX=\theta.$$
双重期望公式保证了哪怕我们增加了一层条件抽样（同时需要套一层期望）还是一个无偏估计

这个结论适用范围就比较大了 只要构造一种条件 然后通过两层模拟的方式处理两层期望就可以了

请注意 条件期望法比较特殊 他需要我们进行一定的理论推导才能使用；如果本身题目里有关于条件的内容 直接顺着他的条件进行抽样模拟那和我们前面直接模拟是没什么区别的

#### 例子
假定$Y$是均值为$1$的指数分布随机变量 在$Y=y$条件下 $X\sim N(y,4)$ 模拟$\theta=p\{X>1\}$
最直接的模拟方法就是
先生成随机数$y$ 根据生成的$y$模拟随机变量$X$ 然后看满足条件的个数标准$I$ 最后求$E(I)$ 
这不是我们的条件期望法 下面才是！
作变形构造正态变量方便后面分析
$$Z=\frac{X-y}2,$$
对条件期望作理论分析有
$$\begin{aligned}
E[I\mid Y=y]& =P\langle X>1\mid Y=y\rangle   \\
&=P\left\{Z>\frac{1-y}2\right\} \\
&=\overline{\Phi}\Big(\frac{1-y}2\Big),
\end{aligned}$$
理解我们的第一步等式 这个期望就是和这个概率相等的 
对$y$ 进行抽样 然后计算期望
这才是条件期望法能减少方差的操作步骤
我们知道条件期望那两层分布的情况 否则条件期望那一步理论推导推不出来

另一个例子 
用条件期望求解 $\theta=E[e^{XY}]$  其中随机变量 $X,Y$ 都服从 $b(n,p)$
我们还是需要先确定条件期望 $E[e^{XY}|X=x]$
$$\begin{aligned}
&E(e^{XY}|X=x)=E(e^{xY}) \\
&=\sum_{y=0}^{\infty}e^{xy}\cdot C_{n}^{y}p^{y}\left(r-p\right)^{n-y} \\
&=\sum^{}\left(e^{x}p\right)^{y}\left(1-p\right)^{n-y} \\
&=(e^{xp}+1-p)^{n}
\end{aligned}$$
然后在对$y$求期望 还原会我们需要求的期望
