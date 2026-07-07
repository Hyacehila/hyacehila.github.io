---
title: "数理统计：总体与样本、统计量与抽样分布"
title_en: "Mathematical Statistics: Populations, Samples, and Sampling Distributions"
date: 2023-03-18 21:28:08 +0800
categories: ["Data Science", "Probability & Statistical Foundations"]
tags: ["Learning Notes", "Statistics", "Mathematical Statistics", "Statistical Inference"]
author: Hyacehila
excerpt: "整理总体与样本、统计量、抽样分布、参数估计、假设检验、方差分析和常见检验方法。"
excerpt_en: "Covers populations and samples, statistics, sampling distributions, parameter estimation, hypothesis testing, ANOVA, and common tests."
mathjax: true
hidden: true
permalink: '/blog/2023/03/18/mathematical-statistics-notes/'
---
## 基本概念
统计学是概率论的逆问题 在概率论中我们有事情发生的本质 去研究事情的结果 在统计学中我们通过对现实世界的观测 反推其原理
数理统计是整个统计学的基础知识 后面的非常多内容都从这里展开
数理统计：使用概率论和数学的方法，研究如何收集带有随机误差的数据，并在设定的模型（统计模型）下对收集到数据进行分析（统计分析） 以对所研究的问题作出推断（统计推断）
下面我们会从统计学最基础的概念开始 搭建经典统计学的基础 也就是频率学派；至于对应的Bayes学派 则会在以后再进行单独进行介绍

在展开具体的统计学内容叙述之前 我们需要分清楚统计学的另一种派别划分方式 描述统计学派和统计推断学派 前者致力于研究观测值本身 后者侧重于从观测值入手分析研究对象本身 两者不可分离 描述统计学虽然难度并不高 但是依旧是不可缺少的一部分
### 总体与样本
#### 总体和个体
研究对象的全体称为总体；组成整体的元素称为个体
在统计研究中往往关注的指标有限 因此要研究的指标的全体此时就是总体，与之对应的仍有个体
非常明显的 我们可以把这些数量指标看作一个随机变量 
此时 总体是一个随机变量（或随机向量） 他的分布和数字特征叫做整体分布于数字特征
核心点在于，总体是一个概率分布
由于参数未知 我们也可以说总体是一个概率分布族
有时候总体并不能用一个含参数分布表示，我们称为非参数总体，此时适用于非参数统计的处理方法
#### 抽样和样本
为推断总体分布及各种特征，按一定规则从总体中抽取若干个体进行观察试验，以获得有关总体的信息，这一抽取过程称为 “抽样”；
所抽取的部分的个体称为样本.   样本中所包含的个体数目称为样本容量
由于样本的抽取是随机的，每个个体是一个随机变量 容量为$n$的样本可以看作$n$维随机变量
一旦取定一组样本，得到的是$n$个具体的数，称其为样本的一个观察值，简称样本值 
那些直接给出精确值的样本称为完全样本
只给出样本观测值范围的称为分组样本
#### 简单随机样本
由于抽样的目的是为了对总体进行统计推断，为了使抽取的样本能很好地反映总体的信息，必须考虑抽样方法 最常用的抽样方法是**简单随机抽样** 他满足
* 抽取的所有样本和总体同分布
* 样本之间相互独立
使用简单随机抽样得到的样本叫做简单随机样本
在后面的研究中 不额外叙述那么所有的样本都视作简单随机样本

最后
事实上我们抽样后得到的资料都是具体的、确定的值；它们是样本值而不是样本；
统计学的任务就是从已有的资料（样本值）去研究总体的特征 其中样本是我们的桥梁；我们之所以能进行统计学 是因为样本值是被总体分布确定的

### 统计量及其分布
#### 统计量
样本是进行统计推断的依据。但在实际应用时，一般不是直接使用样本本身，而是针对具体问题构造适当的函数，也就是统计量,利用这些函数 来进行研究

定义：$X_{i}$是来自总体的样本 $x_{i}$ 是样本值 那么我们称连续函数$g(X_{1},...X_{n})$为统计量 对应的$g(x_{1}...x_{n})$为对应的观察值
注意 统计量一定不能含有任何总体分布中的未知参数 
比如 $\frac1{\sigma^2}\sum_{i=1}^n(X_i-\mu)^2$ 是否为统计量取决于参数$\sigma,\mu$是否已知
样本的均值 方差 标准差 各阶原点矩 中心矩都是重要的统计量 他们的表达式就不再赘述 

#### 修正样本方差
在概率论中我们使用的方差是
$$S^{2}=\frac{1}{n}\sum_{i=1}^{n}(X_{i}-\mu)^{2}$$
但是在统计学中 为了满足我们后面会指出的关于无偏估计的问题 我们除了要用样本均值代替总体均值以外 还需要进行以下的修正
$$S^{2}=\frac{1}{n-1}\sum_{i=1}^{n}(X_{i}-\overline{X})^{2}=\frac1{n-1}[\sum_{i=1}^nX_i^2-n\overline{X}^2]$$



#### 三大抽样分布
统计量是完全由样板确定的函数 他也是一个随机变量 他的分布称为抽样分布 下面我们来介绍三个非常重要的抽样分布 他们都来自正态总体这一最经典的总体情况
##### 卡方分布（$\chi ^{2}$分布）
定义 ：设$X_{1},X_{2}...X_{n}$ 是来自标准正态总体$N(0,1)$的样本 则称统计量
$$\chi^2=X_1^2+X_2^2+\cdots+X_n^2$$
称为自由度为$n$的卡方分布 记作$\chi^2\sim\chi^2(n).$
我们直接给出卡方分布的概率密度函数 具体的证明可以用矩母函数等特征函数手段
概率密度函数为
$$\left.f(x)=\left\{\begin{array}{cc}\frac{1}{2^{n/2}\Gamma(n/2)}x^{\frac{n}{2}-1}e^{-\frac{x}{2}},&x>0,\\0,&\text{其它}.\end{array}\right.\right.$$
同时给出一些性质
卡方分布具有可加性（特征函数理论可证明）$$X\sim\chi^2(n_1),Y\sim\chi^2(n_2),\text{且}X,Y\text{独立}\Rightarrow X+Y\thicksim\chi^2(n_1+n_2)$$
卡方分布的期望和方差为（中心矩，原点矩理论可证明）
$$E(\chi^2)=n,D(\chi^2)=2n.$$
定义 某点$x$是$\chi^2\sim\chi^2(n).$ 的上$\alpha$ 分位点 当且仅当
$$P\{\chi^{2}>\chi_{\alpha}^{2}(n)\}=\alpha(0<\alpha<1).$$
(非常基础的关于分位点的定义)
##### T分布
设$X\sim N(0,1),Y\sim\chi^2(n)$ 两者相互独立 则称随机变量
$$t=\frac X{\sqrt{Y/n}}$$
满足自由度为$n$的T分布 记作$t(n)$
给出T分布的概率密度函数
$$f(x)=\frac{\Gamma[(n+1)/2]}{\sqrt{\pi n}\Gamma(n/2)}{\left(1+\frac{x^2}n\right)^{-\frac{n+1}2}}(-\infty<x<+\infty)$$
容易看出来T分布的概率密度函数是偶函数 并且标准正态概率密度是T分布在$n$ 逼近 $\infty$ 时候的极限

作为偶函数的T分布 关于分位点有以下性质
$$t_{1-\alpha/2}(n)=-t_{\alpha/2}(n)$$
##### F分布
设$X\sim\chi^{2}(n_{1}),Y\sim\chi^{2}(n_{2})$ 两者相互独立 则称随机变量 
$$F=\frac{X/n_1}{Y/n_2}$$
服从自由度为$n_{1},n_{2}$的F分布 记作$F(n_{1},n_{2})$
给出F分布的概率密度函数
$$f(x)=\begin{cases}\Gamma[(n_1+n_2)/2](n_1/n_2)^{\frac{n_1}2}x^{\frac{n_1}2-1}\\\hline\Gamma(n_1/2)\Gamma(n_2/2)[1+(n_1x/n_2)]^{\frac{n_1+n_2}2}\\0,&\text{其它}.\end{cases}$$
容易知道
F分布有关于交换构成部分分子和分母的性质
$$F\sim F(n_1,n_2)\Rightarrow\frac1F\sim F(n_2,n_1)$$
关于F分布的分位点有这样的性质
$$F_{1-\alpha}(n_1,n_2)=\frac1{F_\alpha(n_2,n_1)}$$
##### 关于自由度
自由度 原文 degree of freedom  在最初引入的时候是在卡方分布的时候
他表示了我们这个统计量（抽样分布）中蕴含的独立随机变量的个数 
在计算上 往往是 样本容量减去约束方程的数量 我们在后面的很多地方都会继续见到关于自由度的很多表述
#### 样本均值与样本方差的分布
##### 定理一
假设总体有均值和方差 
$$E(X)=\mu,D(X)=\sigma^2,$$
那么无论总体符合什么分布 对于来自这个总体的样本 
对于样本均值一定有
$$E(\overline{X})=\mu,D(\overline{X})=\frac{\sigma^{2}}{n}.$$
样本（修正）方差的期望和方差可以用定理二给出的分布情况进行处理
结合$\chi^2$ 分布的均值和方差进行计算
##### 定理二
对于单正态总体$\mathrm{N(~\mu~,~\sigma~2)}$ 其样本均值$\overline{X}$和样本修正方差$S^2$满足
* $\overline{X}\sim N(\mu,\frac{\sigma^{2}}{n}).$
* $\frac{(n-1)S^2}{\sigma^2}\sim\chi^2(n-1);$ 
* 如果是不修正的以$\frac{1}{n}$为系数的方差则是$\frac{nS^2}{\sigma^2}\sim\chi^2(n-1);$ 
* 如果方差的计算改用真实均值而不是样本均值 $\chi^2$分布的自由度变为$n$ ；修正方差不涉及使用真实均值的情况，修正是为了解决无偏问题，有真实均值不涉及这个
* $\frac{\overline{X}-\mu}{S^{\color{red}}/\sqrt{n}}\sim t(n-1);$
* 样本均值$\overline{X}$和样本修正方差$S^2$相互独立
##### 定理三
对于同方差的双正态总体 $\mathrm{N(~\mu_{1}~,~\sigma^2)}$  $\mathrm{N(~\mu_{2}~,~\sigma^2)}$ 的均值差有
$$\frac{(\overline{X}-\overline{Y})-(\mu_1-\mu_2)}{S_w\cdot\sqrt{\frac1{n_1}+\frac1{n_2}}}\sim t(n_1+n_2-2)$$
其中
$$S_{w}^{2}=\frac{(n_{1}-1)S_{X}^{2}+(n_{2}-1)S_{Y}^{2}}{n_{1}+n_{2}-2}.$$
如果不修正则是
$$S_{w}^{2}=\frac{(n_{1})S_{X}^{2}+(n_{2})S_{Y}^{2}}{n_{1}+n_{2}-2}.$$
其中$\overline{X}~~ ~~\overline{Y}~~ ~~S_{X}^{2} ~~~~S_{Y}^{2}$ 表示样本均值和方差
并且有 
$$\frac{S_X^2}{S_Y^2}\thicksim F(n_1-1,n_2-1)$$
如果不修正则是
$$\frac{n_{1}(n_{2}-1)S_X^2}{n_2(n_1-1)S_Y^2}\thicksim F(n_1-1,n_2-1)$$
*当两者方差不同，但是都采用修正方差形式的时候有*
$$\frac{\frac{S_X^2}{\sigma_{x}^2}}{\frac{S_Y^2}{\sigma_{y}^2}}\thicksim F(n_1-1,n_2-1)$$
本节给出的重要公式和给出的三大分布都非常的重要 后面会经常用到来处理一些假设检验问题

里面涉及的推导大部分都是一些基础的变形和对前面的定理的运用
注意严格区分样本方差和样本修正方差 这个系数的差距对很多问题都有影响
#### 例题
[本部分例题](/blog/2024/09/24/probability-and-statistics-exercises-notes/) 的“数理统计抽样分布例子”一节
### 次序统计量及其分布
#### 次序统计量
假设$X_{1}...X_{n}$是取自总体分布函数$F(x)$的样本 如果我们把这些样本观测值从下到大进行排序 得到有序样本 $X_{(1)}...X_{(n)}$  
我们称第$i$个次序统计量为从小到大排列的第$i$个量 $X_{i}$ 
明显的 $X_{1}$ 称为最小次序统计量 $X_{n}$ 称为最大次序统计量

非常明显的 次序统计量应该是离散的分布 因为样本数量一定是有限的若干个点 而且我们可以研究次序统计量的分布 不过明显的是 次序统计量分布之间应该并不是独立的 
对于原本就离散的分布 次序统计量非常好求 只需要列出所有的情况就可以
下面我们研究怎么求连续分布的次序统计量
#### 次序统计量的分布
设总体$X$的密度函数为$p(x)$ 分布函数为$F(x)$  $X_{1}...X_{n}$是取自总体分布函数$F(x)$的样本 则第$k$个次序统计量$X_{k}$的分布为 
$$p_{k}\left(x\right)=\frac{n!}{\left(k-1\right)!\left(n-k\right)!}\left(F\left(x\right)\right)^{k-1}\left(1-F\left(x\right)\right)^{n-k}p(x)$$

对于多个次序统计量的联合分布 我们直接给出二元情况的公式为 
$$\begin{aligned}p_{ij}(y,z)&=\frac{n!}{\left(i-1\right)!\left(j-i-1\right)!\left(n-j\right)!}\left[F(y)\right]^{i-1}\left[F(z)-F(y)\right]^{j-i-1}\\\\&\cdot\left[1-F(z)\right]^{n-j}p(y)p(z),\quad y\leq z\end{aligned}$$
对于更广泛的情况我们就不给出公式了
### 经验分布函数
假设$x_{1}...x_{n}$是取自总体分布函数$F(x)$的样本 如果我们把这些样本观测值从下到大进行排序 得到有序样本 $x_{(1)}...x_{(n)}$ 用有序样本定义以下函数 
$$F_n(x)=\begin{cases}0,&x<x_{(1)}\\k/n,&x_{(k)}\le x<x_{(k+1)},\\1,&x_{(n)}\le x\end{cases}\quad k=1,2,...,n-1$$
明显的 这个函数满足是一个分布函数的所有条件 我们称其为经验分布函数$F_{n}(x)$     他是一个普通的跳跃函数
根据伯努利大数定律 $F_{n(x)}$依概率收敛于分布函数$F(x)$ 

经验分布函数是从样本得到的 哪怕相同的样本容量不同的样本会生成不同的经验分布函数
对于确定的$x$ 经验分布函数是一个随机变量。是事件$\{X<x\}$ 发生的频率
我们给出一个关于经验分布函数的定理 他说明了只要样本足够大 经验分布函数是分布函数的良好近似 从样本推断整体是可行的
设$x_1,x_2,\cdots,x_n$是总体分布函数为$F(x)$的样本， $F_n(x)$为其经验分布函数，当$n\to+\infty$时，有
$$P\{ \lim sup\mid F_n( x) - F( x) \mid = 0\} =1$$

### 样本数据的整理和展示
参考 [描述性统计与可视化](/blog/2023/11/05/descriptive-statistics-and-visualization-notes/)


## 参数估计

### 参数估计问题
#### 定义
数理统计的基本问题是根据样本提供的信息，对总体的分布以及分布的某些数字特征作出推断。这个问题中的一类是总体分布的类型为已知，而它的某些参数为未知，根据所得样本对这些参数作出推断，这类问题称为参数估计
#### 未知参数的估计量和估计值
假设我们有一个正态分布总体$X$ 服从$N(\mu,\sigma^{2})$ 参数均未知 
有随机抽查得到的样本数据一百个 我们如何估计未知参数呢？
对于$\mu$ 可以用样本均值 样本中位数等 对于$\sigma$ 可以用方差进行估计
能看出  我们需要构造一个样本函数 $\hat{\theta}(X_1,X_2,\cdotp\cdotp\cdotp X_n)$ 明显的这是一个统计量 用于参数估计的统计量我们称为估计量 带入样本值就是估计值
这种直接给出未知参数的值的方法称为**点估计**
估计$(0,1)$ 覆盖$\mu$  的概率为百分之九十五
给出未知参数取值范围的方法称为**区间估计**
#### 常用的估计方法
* 矩估计法
* 极大似然估计法
* 最小二乘估计法
* 贝叶斯方法
* 尺度不变方法
* 最大最小估计法
常用的估计基本是这些 我们会在不同的课程中学习到这些内容
数理统计中 我们会介绍矩估计法和极大似然方法 他们都属于点估计方法
### 矩估计
#### 理论依据
矩估计的思想是一种简单的替换思想 由统计学家Pearson提出 其基本思想是使用样本矩去估计总体矩 
理论依据 辛钦大数定律 
$$\lim_{n\to\infty}P(|\frac1n\sum_{i=1}^nX_i-\mu|<\varepsilon)=1.$$
样本矩依概率收敛于总体矩
由于 $X_{i}^{k}$ 仍然保证了独立同分布 因此高阶原点矩也可以使用辛钦大数定律
$$\lim_{n\to\infty}P(\frac{1}{n}\sum_{i=1}^{n}X_{i}^{k}-E(X^{k})|<\varepsilon)=1$$
这种用相应的样本矩去估计总体矩从而确定待定参数的估计值的估计方法就称为**矩估计法**  
#### 方法
设总体$X$的概率函数$f(x;\theta_1,\theta_2,...,\theta_l)$中含有$l$个未知参数$\theta_{1}...\theta_{l}$ 
$X_1,X_2,\cdotp\cdotp,X_n$是总体$X$ 的样本，且 且总体的前$l$阶原点矩$E(X^k)\left(_{k=1,2,..,l}\right)$存在，则它们应是这$\iota$个参数的函数：
$$E(X^{k})=g_{k}(\theta_{1},\cdots,\theta_{l}),\quad k=1,2,\cdots,l$$
又 样本的$k$ 阶原点矩为$A_{k}=\frac{1}{n}\sum_{i=1}^{n}X_{i}^{k}$
因此我们可以建立并解以下的方程来确定参数的矩估计值
$$\begin{cases}g_1(\theta_1,\cdots,\theta_l)=\frac1n\sum_{i=1}^nX_i,\\g_2(\theta_1,\cdots,\theta_l)=\frac1n\sum_{i=1}^nX_i^2,\\\cdots\cdots\\g_l(\theta_1,\cdots,\theta_l)=\frac1n\sum_{i=1}^nX_i^l,\end{cases}$$
非常明显 我们应该根据参数的数量选取建立的方程的数量 选取原点矩的最高阶数
#### 例子
[本部分例题](/blog/2024/09/24/probability-and-statistics-exercises-notes/) 的“数理统计矩估计例子”一节

矩法的优点是简单易行,  并不需要事先知道总体是什么分布 

缺点是，当总体类型已知时，**没有充分利用分布提供的信息** . 

同时 一般场合下,   矩估计量不具有唯一性 ，其主要原因在于建立矩法方程时，选取那些总体矩用相应样本矩代替带有一定的随意性 
### 极大似然估计
极大似然估计（MLE）是一种在总体分布类型已知条件下使用的一种参数估计方法 .
其最初提出是数学家Gauss 但是真正的发扬光大归功于统计学家Fisher
极大似然估计的思想非常简单
在所有的参数选择中选取能使样本观测值出现的概率为最大的那一个来作为它的估计值
这种估计思想是非常符合人们的直觉的
#### 极大似然原理
一般说，若事件A发生的概率与参数$\theta\in\Theta$有关，$\theta$ 取值不同，P(A)也不同。则应记事件A发生的概率为$P(A|\theta).$若一次试验，事件A发生了，可认为此时的$\theta$值应是在Θ中使P(A|$\theta$)达到最大的那一个。这就是极大似然原理
他的优点在于在整体分布已知的时候运用的整体分布给予的信息 质量更好 但是计算难度明显上升了
极大似然估计的前提是具有参数形式的已知总体分布 也是一种参数方法
#### 方法
##### 离散总体的极大似然估计
如果总体$X$是离散型 那么其分布律应该为 $P\{X=x\}=p(x;\theta)$ 形式已知但是参数未知  设$X_1,\cdots,X_n$是来自$X$的样本；则$X_1,\cdots,X_n$的联合分布律
$$\prod_{i=1}^np(x_i;\theta)$$
又设$x_1,\cdots,x_n$是$X_1,\cdots,X_n$的一个样本值： 易知样本$X_1,...,X_n$取$x_1,...,x_n$的概率，亦即 事件$\{X_1=x_1,\cdots,X_n=x_n\}$发生的概率为：
$$L(\theta)=L(x_1,\cdots,x_n;\theta)=\prod_{i=1}^np(x_i;\theta),\theta\in\Theta.$$
其中$L(\theta)$ 称为似然函数 根据极大似然估计的思路 我们只需要找到在确定的样本值下 挑选合适的参数让似然函数极大化 作为参数的估计值 即
$$L(x_1,\cdots,x_n;\hat{\theta})=\max_{\theta\in\Theta}L(x_1,\cdots,x_n;\theta)$$
$\hat{\theta}(x_1,\cdotp\cdotp,x_n)$ 称为极大似然估计值 
$\hat{\theta}(X_1,\cdotp\cdotp\cdotp,X_n)$ 称为参数的极大似然估计量
##### 连续总体的极大似然估计
使用完全一样的原理可以得到 似然函数 
$$L(\theta)=L(x_{1},\cdots,x_{n};\theta)=\prod_{i=1}^{n}f(x_{i};\theta)$$
还是构造让似然函数极大化
$$L(x_1,\cdots,x_n;\hat{\theta})=\max_{\theta\in\Theta}L(x_1,\cdots,x_n;\theta)$$
称$\hat{\theta}(x_1,\cdots,x_n)$为$\theta$的极大似然估计值
称$\hat{\theta}(X_1,\cdots,X_n)$为$\theta$的极大似然估计量
##### 极大化方法
如果密度函数$f(x;\theta),p(x;\theta)$ 关于$\theta$ 可微 那么我们可以使用分析中的微分手段进行极大化研究 
$$\frac{dL(\theta)}{d\theta}=0$$
求解的$\theta$ 就是我们最后参数的估计结果

由于似然函数$L(\theta)$ 是多个函数的积 所以研究对数似然函数$\ln(L(\theta))$可以简化求导的操作 简化运算 最后求出的结果是不变的 因为$\ln(x)$ 单调

这点源于定理：
若$\hat\theta$为未知参数$\theta$的极大似然估计量，而$g(\theta)$为$\theta$的单调函数 则${g}(\hat\theta)$ 也是$g(\theta)$的极大似然估计量 


#### 例子
[本部分例题](/blog/2024/09/24/probability-and-statistics-exercises-notes/) 的“数理统计极大似然估计例子”一节
### 中位数估计
我们来尝试估计一个Cauchy分布的参数$\theta$ 密度函数为
$$f\left(x,\theta\right)=\frac{1}{\pi\left[1+\left(x-\theta\right)^{2}\right]}$$
我们知道 Cauchy分布的各阶矩不存在 所以矩估计无法使用 尝试极大似然估计
$$\sum_{i=1}^{n}\frac{X_{i}-\theta}{1+\left(X_{i}-\theta\right)^{2}}=0$$
这个方程有很多根 并且并不好求根 极大似然估计此时并不实用
但是我们有一个较为简单的可行的方法

$\theta$是Cauchy分布的中位数 我们可以使用样本的中位数来估计它 这和矩估计的思想有一些接近 我们称为中位数估计
### 估计量的优良性准则
前面介绍了三种估计方法 他们的估计结果可能相同 也可以能不同 矩估计量本身也不唯一 所以在有多个估计量的时候 哪一个估计量更好 这就是我们现在要研究的问题 如果判断估计量的好坏 如果得到更好的估计量
#### 无偏性
##### 无偏性介绍
我们知道 点估计量其实是一个随机变量  那么作为一个波动的估计量 其众多波动的结果如果在参数的真值附近波动 他或许可以被视为一个好的估计量
也就是无偏性研究估计的系统误差

设总体$X\sim F(x;\theta)(\theta\in\Theta)$, Θ为参数空间. 设$X_1,X_2,...,X_n$为总体$X$的样本，$\hat{\theta}=\hat{\theta}(X_1,X_2,...,X_n)$为未知参数 $\theta$ 的点估计
如果估计量 $\hat\theta$ 的数学期望存在 并且有 
$$E_{\theta}(\hat{\theta})=\theta $$
则称$\hat\theta$ 是$\theta$ 的无偏估计 否则称为有偏估计
称
$$b_{n}(\hat{\theta},\theta)=E_{\theta}(\hat{\theta})-\theta $$
为估计量的偏差 

如果 $b_{n}(\hat{\theta},\theta)\ne0$ 则称$\hat\theta$ 是$\theta$ 的有偏估计
如果$\operatorname*{lim}_{n\to\infty}b_{n}({\hat{\theta}})=0$   则称$\hat\theta$ 是$\theta$ 的渐进无偏估计

无偏性要求不存在系统误差，这在理论上当然是好的，但是在实际应用上，无偏是否有价值还需要根据具体的事件来决定
##### 定理
无论总体$X$服从什么分布，若
$$
\mu\overset{\Delta}{\operatorname*{=}}E(X)\:,\:\sigma^{2}\overset{\Delta}{\operatorname*{=}}D(X)
$$
都存在，则$\hat{\mu}=\overline{X},\hat{\sigma}^2=S^2$分别是 $\mu,\sigma^2$ 的无偏估计.

这里的$S^2$ 是样本方差 是修正后的 $\frac{1}{n-1}$为系数的方差

在数理统计的开头我们介绍过
$$E(\overline{X})=\mu,D(\overline{X})=\frac{\sigma^{2}}{n}.$$
前者就是是均值估计无偏性的证明 关于方差估计无偏性的证明如下
$$\begin{aligned}
E\left(S^{2}\right)& =\frac{1}{n-1}E\left[\sum_{i=1}^{n}\left(X_{i}-\overline{X}\right)^{2}\right]  \\
&=\frac{1}{n-1}E\left[\sum_{i=1}^{n}X_{i}^{2}-n\overline{X}^{2}\right] \\
&=\frac{1}{n-1}\left[\sum_{i=1}^{n}E\left(X_{i}^{2}\right)-nE\left(\overline{X}^{2}\right)\right] \\
& =\frac{1}{n-1}\left[\sum_{i=1}^{n}\left(\sigma^{2}+\mu^{2}\right)-n\left[\operatorname{Var}(\bar{X})+\left(E(\bar{X})\right)^{2}\right]\right]  \\
&=\frac{1}{n-1}\left[n\left(\sigma^{2}+\mu^{2}\right)-n\left[\frac{\sigma^{2}}{n}+\mu^{2}\right]\right] \\
&=\sigma^{2}.
\end{aligned}$$

注意 对于没有修正的样本方差(以$\frac{1}{n}$ 为系数 ) 只是有渐进无偏性
$$\begin{gathered}
{S_{n}}^{2}=\frac{1}{n}\sum_{i=1}^{n}(X_{i}-\bar{X})^{2}=\frac{n-1}{n}S_{n}^{*2} \\
E({S_{n}}^{2})=\frac{n-1}{n}E(S_{n}^{*2}) \\
=\frac{n-1}{n}\sigma^{2}\rightarrow\sigma^{2}~(n\rightarrow\infty) 
\end{gathered}$$
这意味着 无论是矩估计还是MLE 他们对总体方差的估计直接采用样本中心矩的方式是没有满足无偏性的
##### 例子
[本部分例题](/blog/2024/09/24/probability-and-statistics-exercises-notes/) 的“数理统计估计量的评估准则 / 无偏性”一节
#### 均方误差
假设用$T(x)$作为参数$q(\theta)$的估计量，评价估计优劣的一个自然准则可定义如下：
$$
MSE_\theta(T)=R(\theta,T)=E(T(x)-q(\theta))^2
$$
称上式为均方误差，简记为MSE(Mean Squared Error)

这是一个非常自然的用来估计误差大小的准则 在统计学的非常多地方 只要涉及点估计就很常用（与之对应的形式的方差一般用于评估数据离散程度）
均方误差一般可以进行如下形式的分解
$$\begin{gathered}
MSE=E_\theta\left[(\hat{\theta}-\theta)^2\right]=E_\theta\left[\hat{\theta}^2+\theta^2-2\hat{\theta}\theta\right] \\
=E_\theta\left[\hat{\theta}^2\right]-E_\theta\left[\hat{\theta}\right]^2+E_\theta\left[\hat{\theta}\right]^2+\theta^2-2\theta E_\theta\left[\hat{\theta}\right] \\
=V_\theta\left[\hat{\theta}\right]+(\theta-E_\theta\left[\hat{\theta}\right])^2 
\end{gathered}$$
如果我们记 偏差 bias 为
$$bias=E_\theta[\hat{\theta}]-\theta $$
则有
$$MSE=V_\theta[\hat{\theta}]+bias^2$$
也就是均方误差为 估计的方差 和 偏差的平方 的和

能看出如果估计无偏 偏差应该为0 MSE就等于估计的方法
实际上 理论的均方误差仍然含有关于真实值的式子 在实际的计算中我们需要用估计值来代替才能得到MSE 真实值是我们永远不知道的

定义 如果对于所有的$\theta\in\Theta$ 总有$R(\theta,T)\leq R(\theta,S)$ 那么则称$S$是不容许的估计 或者称$T$比$S$更好 我们一般不选择不容许的估计

很明显的 我们希望能找到一个对于各种参数取值情况都有着最小MSE的估计 但是这是不可能的；均方误差一致达到最小的最优估计并不存在；
一般我们往往选择对估计提出一些合理性的要求 在满足合理性要求的估计中选择优良的估计 比如选择一致最小方差无偏估计 

[本部分例题](/blog/2024/09/24/probability-and-statistics-exercises-notes/) 的“均方误差”一节
#### 有效性
定义 设$X_1,X_2,\cdot\cdot,X$,是总体$X\sim F(x,\theta);\theta\in\Theta$ 的样本，
$\hat{\theta}_{1}=\hat{\theta}_{1}(X_{1},X_{2},\cdots,X_{n}),\:\hat{\theta}_{2}=\hat{\theta}_{2}(X_{1},X_{2},\cdots X_{n})$ 
都是$\theta$ 的无偏估计，即$E(\hat{\theta}_1)=E(\hat{\theta}_2)=\theta\quad$.若$\forall\theta\in\Theta$ 有
$$
D(\hat{\theta}_{1})\leq D(\hat{\theta}_{2})
$$
则称$\hat\theta_{1}$ 比$\hat\theta_{2}$ 更有效 
能看出来 有效性只是一个更狭义的MSE 
我们在C-R不等式介绍了有效估计的概念，事实上整个C-R不等式和UMVUE都是从有效性这里衍生的
#### 相合性（一致性）
##### 定义
这是一个研究随着样本容量增加的时候 估计量应该怎么变化而给出的估计量优良性准则 
一个非常自然的思想是 随着样本容量的增加 估计应该更加精确
定义 设$\hat{\theta}_n=\hat{\theta}(X_1,X_2,...,X_n)$是未知参数 $\theta$ 的点估计， 若 $\forall\theta\in\Theta$ 满足： $\forall\varepsilon>0$ 有 $\lim P\{|\hat{\theta}_{n}-\theta|\geq\varepsilon\}=0$ 
则称 $\hat{\theta}_n$ 是$\theta$ 的相合估计
##### 定理
无论总体$_{X}$服从什么分布，若 $\mu\triangleq E(X),\:\sigma^2\triangleq D(X)$
都存在，则$\hat{\mu}=\overline{X},\hat{\sigma}^2=S_n^{*2}$分别是 $\mu,\sigma^2$ 的相合估计
研究相合性的核心是辛钦大数定律 也就是样本矩依概率收敛于总体矩 这是研究相合性的核心
直接根据辛钦大数定律知
$$\bar{X}=\frac{1}{n}\sum_{i=1}^{n}X_{i}\xrightarrow{P}\mu(n\rightarrow\infty)$$
研究方差$$\begin{gathered}
S_{n}^{*2}=\frac{1}{n-1}\sum_{i=1}^{n}(X_{i}-\bar{X})^{2} \\
=\frac{1}{n-1}\big(\sum_{i=1}^{n}X_{i}^{2}-n(\bar{X})^{2}\big) \\
=\frac{n}{n-1}\cdot\frac{1}{n}\sum_{i=1}^{n}X_{i}^{2}-\frac{n}{n-1}(\bar{X})^{2} 
\end{gathered}$$
明显的 其结果依概率收敛于方差 因为方差等于二阶原点矩和一阶原点矩平方的差
##### 结论
* 矩估计是相合估计
* 极大似然估计一般是相合估计
* 相合估计不一定是无偏估计 无偏估计也不一定相合
* 如果$\hat\theta$ 是（渐进）无偏估计 那么根据切比雪夫不等式知道 当$\lim_{n\to\infty}D(\hat{\theta})=0$时它是相合估计（**这个结论也非常的好用**）
根据切比雪夫不等式这里的推论 能给出相合估计的充分条件：**（渐进）无偏估计并且方差逼近0**
证明：切比雪夫不等式为
$$P\left(\left|\xi- E(\varepsilon)\right|\geqslant\varepsilon\right)\leqslant\frac{D\left(\xi\right)}{\varepsilon^{2}}$$
如果有$\lim_{n\to\infty}D(\hat{\theta})=0$ 则 不等式左端根据夹逼定理也逼近到零 就是相合估计的定义了 证毕

#### 渐进正态性
许多形状复杂的统计量 在$n$接近$\infty$ 的时候 他们都渐进于正态分布 
渐进正态性是中心极限定理的外推结论；
哪些统计量具有渐进正态性 如何判断 不是我们这里研究的重点
渐进正态性和相合性一样 都属于大样本性质
### C-R不等式
一个参数往往会有多个无偏估计 我们认为方差小的无偏估计更加有效
我们自然希望方差越小越好 但是这个方差有无下界 什么条件下存在下界
C-R不等式 （Cramer-Rao 不等式）解释了这一点 
证明了在某些条件下 无偏估计量的方差$\hat\theta$存在一个正的下界 
#### Fisher信息量
##### 定义
对于对数似然函数$ln(\xi;\theta)$  定义Fisher信息量如下
$$I(\theta)=E(\frac{\partial\ln f(\xi;\theta)}{\partial\theta})^{2}>0$$
这里的期望是 视参数为确定的值 对$X$ （样本）求期望 也就是$E_{X|\theta}$ 
种种性质表明 Fisher信息量越大 可以视为抽取的样本包含的关于未知参数的信息越到
##### 例子
[本部分例题](/blog/2024/09/24/probability-and-statistics-exercises-notes/) 的“数理统计Fisher信息量例题”一节
##### 结论
如果（一般大部分分布都满足）
$$\frac\partial{\partial\theta}\int\frac{\partial f(x;\theta)}{\partial\theta}dx=\int\frac{\partial^2f(x;\theta)}{\partial\theta^2}dx,$$
则
$$I(\theta)=-E[\frac{\partial^2\ln f(\xi;\theta)}{\partial\theta^2}]$$

#### C-R不等式
设$\xi_1,\xi_2,\cdots,\xi_n$为取自具有概率函数$f(x;\theta),\theta\in\Theta$ $=\{\theta:a<\theta<b\}$的母体的一个子样 ,其中$a,b$为已知常数， 且可设$a=-\infty,b=+\infty.$ 又$\eta=u(\xi_1,\xi_2,\cdots,\xi_n)$是$g(\theta)$的一个无偏估计 并且满足正则条件
$$\text{集合}\{x:f(x;\theta)>0\}\text{与}\theta\text{无关};$$
$$\begin{aligned}g^{\prime}(\theta)&\text{与}\frac{\partial f(x;\theta)}{\partial\theta}\text{存在,且对一切}\theta\in\Theta,\\\frac\partial{\partial\theta}&\int f(x;\theta)dx=\int\frac{\partial f(x;\theta)}{\partial\theta}dx\end{aligned}$$
$$\begin{aligned}\frac\partial{\partial\theta}&{\int\cdots\int u(x_1,x_2,\cdots,x_n)f(x_1;\theta)\cdots f(x_n;\theta)dx_1\cdots dx_n}\\&=\int\cdots\int u(x_1,x_2,\cdots,x_n)\frac\partial{\partial\theta}[\prod_{i=1}^nf(x_i;\theta)]dx_1\cdots dx_n\end{aligned}$$
则对于Fisher信息量
$$I(\theta)=E(\frac{\partial\ln f(\xi;\theta)}{\partial\theta})^{2}>0$$
有
$$D_\theta\eta\geq{\frac{[g^{\prime}(\theta)]^2}{nI(\theta)}}$$
对于$g(\theta)=\theta$ 的情形 
$$D_\theta\eta\geq\frac1{nI(\theta)}$$
如此 我们给出了估计量的CR下界 CR不等式也称为信息量不等式

我们定义 满足正则条件的估计量称为正规估计量
容易看出 CR下界是针对正规无偏估计的方差下界
对于其他并不正规或者并不无偏的估计 不能用CR不等式给出方差下界
#### 应用CR不等式
定义 如果$\theta$的一个无偏估计$\hat\theta$ 使得CR不等式中
$$D(\hat{\theta})=\frac1{nE[(\frac{\partial\ln f(\xi;\theta)}{\partial\theta})^2]}=\frac1{nI(\theta)}$$
成立 则称其为有效估计（这里的信息量，不考虑多抽样的问题，只抽取一个样本）
定义 称$e=\frac{1}{nI(\theta)}/D(\hat\theta)$ 称为无偏估计的效率 能看出 $e=1$ 称为有效估计
定义 对于$e\ne1$ 的估计 如果$lim(e)=1$ 则称其为渐进有效估计

[本部分例题](/blog/2024/09/24/probability-and-statistics-exercises-notes/) 的“数理统计CR不等式例题”一节

### 充分统计量
#### 充分统计量
##### 引入
样本来自总体 包含着总体的信息 但是我们往往是采用构造样本的函数——统计量来进行统计推断 如何将样本中包含的总体信息提取出来 我们是否提取完全来样本中的总体信息 这就是这里想要解决的
*这里我们一般是只考虑样本中包含的关于总体分布参数的信息，不考虑总体分布类型的信息*
我们使用一个例子来引出充分统计量的概念
例 为研究某个运动员的打靶命中率，我们对该运动员进行测试,观测其10次,发现除第三、六次未命中外，其余8次都命中。
现在想要研究这个运动员命中率这个参数
非常明显的  构造统计量$T=x_{1}+x_{2}+...+x_{n}$ 这种情况下构造的统计量完全不会丢失任何关于命中率$\theta$ 的信息 也就是统计量将样本中关于未知参数的信息完全提取出来了 这就是统计学家Fisher提出的概念 **充分统计量**
有了充分统计量 我们对于这个参数的统计推断就可以转为从统计量出发 不再需要样本数据
##### 定义
样本$X_{1}...X_{n}$ 有着一个样本分布$F_{\theta}(x)$ 里面包含了所有关于参数$\theta$ 的信息 统计量$T$ 也有抽样分布 $T_{\theta}(t)$ 自然的充分统计量的意思就是$T_{\theta}(t)$ 包含了所有$F_{\theta}(x)$中关于参数$\theta$ 的信息 也就是样本的条件分布$F_\theta(x|T=t)$ 不含任何关于参数$\theta$ 的信息
根据这个意思我们就可以给出充分统计量的定义

定义 设$X_1,X_2,...,X_n$为来自总体$X$ 的样本，$X$ 的分布函数为$F(x;\theta),\quad T{=}T(X_1,X_2,...,X_n)$为一个统计量， 当给定 T=t 时，如果样本($X_1,X_2,...,X_n)$ 的条件分布(离散总体时为条件概率，连续总体时为条件密度) 与参数$\theta$无关，则称$T$为参数$\theta$的充分统计量 

等价定义 设$X_1,X_2,..,X_n$为来自总体$x$ 的样本，$X$ 的概率函数为$f(x,\theta),\quad T{=}T(X_1,X_2,...,X_n)$为一个统计量， 其概率函数$g(t,\theta)$,若$\frac{f(x_1,\theta)f(x_2,\theta)\text{L }f(x_n,\theta)}{g(T(x_1,x_2,\text{L },x_n),\theta)}=h(x_1,x_2,\text{L },x_n)$成立；
且当$t=T(x_1,x_2,\mathcal{L},x_n)$取一固定值时，$T=t$发生条件下的条件概率函数$h(x_1,x_2,\mathcal{L},x_n)$不依赖于$\theta$, 则称 $T$ 为参数$\theta$ 的充分统计量
##### 例子
[本部分例题](/blog/2024/09/24/probability-and-statistics-exercises-notes/) 的“数理统计充分统计量例题”一节
#### 因子分解定理
从定义判断一个统计量是否是充分统计量是繁琐的 因此我们给出了因子分解定理 它可以大大的简化寻找一个充分统计量的问题

因子分解定理
$T$ 是$\theta$ 的充分统计量的充要条件为 样本的联合分布可以分解为以下的形式 其中$h$ 非负并且和$\theta$ 无关 $g$ 仅通过$T$ 和样本联系起来
$$L(\theta)=\prod_{i=1}^nf(x_i;\theta)=h(x_1,x_2,\cdots,x_n)g(T(x_1,x_2,\cdots,x_n);\theta)$$
可以非常简单的使用
我们需要做的只是把联合密度函数进行适当的变形和分解 
找到它是哪两个函数的积   核心就是分解

定理 如果$T$ 是$\theta$ 的一个充分统计量 $f(t)$ 是单值可逆函数 那么$f(T)$ 也是$\theta$ 的充分统计量 
#### 完备统计量
首先我们引入完备分布函数族的概念
定义 设总体$X$的分布函数族为$\{F(x;\theta),\theta\in\Theta\}$ 对任意一个满足$E_{\theta}[g(X)]=0$, 对一切$\theta\in\Theta$的随机变量$g(X)$,总有
$$
P_{\theta}\{g(X)=0\}=1,\:\text{对一切}\theta\in\Theta,
$$
则称$\{F(x;\theta),\theta\in\Theta\}$为完备的分布函数族
定义  设$(X_1,X_2,...,X_n)$为来自总体$F(x;\theta)(\theta\in\Theta)$的一个样本，若统计量$T=T(X_1,X_2,...,X_n)$的分布函数族$\{F_{\tau}(x;\theta),\theta\in\Theta\}$ 是完备的分布函数族，则称$T=T(X_1,X_2,\cdots,X_n)\text{}$ 为完备统计量
能看出完备统计量有如下的特征
$$\begin{aligned}P_{\theta}\big\{g_{1}(T)=g_{2}(T)\big\}&=1,\quad\forall\theta\in\Theta\\\Leftrightarrow E_{\theta}\big[g_{1}(T)\big]&=E_{\theta}\big[g_{2}(T)\big],\forall\theta\in\Theta\text{。}\end{aligned}$$
[本部分例题](/blog/2024/09/24/probability-and-statistics-exercises-notes/) 的“数理统计完备统计量例题”一节
#### 指数分布族
这是一类广泛应用的分布族
##### 单参数指数分布族
定义：
设总体 $X$ 或 $X|\theta$ 的分布密度 $p(x|\theta)$ 为：
$$
p(x|\theta)=g(x)h(\theta)\exp\{t(x)\phi(\theta)\}
$$
其中的函数都是一般的已知函数  则称$p(x|\theta)$ 属于单参数指数分布族
例子
研究正态分布$N(\mu,\sigma^{2})$ 当$\sigma^2$已知时 
$$\begin{aligned}
p(x|\mu)& =\left(2\pi\sigma^2\right)^{-\frac12}\exp\left\{-\frac1{2\sigma^2}(x-\mu)^2\right\}  \\
&=\left[\left(2\pi\sigma^2\right)^{-\frac12}\exp\left\{-\frac{x^2}{2\sigma^2}\right\}\right]\exp\left\{-\frac{\mu^2}{2\sigma^2}\right\}\exp\left\{\frac{x\mu}{\sigma^2}\right\}
\end{aligned}$$
符合单参数指数分布族的定义
类似的 如果认为均值已知 方差未知 也是属于单参数指数分布族
类似的 还有泊松分布 二项分布 Gamma分布 Beta分布 都属于指数分布族 
实际上指数分布族非常的广泛
但是均匀分布这一形式非常基础的分布并不是指数分布族 因为他的定义集合与参数有关 就无法归纳为指数的形式
**我们给出的指数分布的定义只是其中一种形式，实际上可以指数分布族有很多种不同的定义形式，不过他们都是等价的**
##### 两参数指数分布族
定义
设总体 $X$ 或 $X|\theta,\varphi$ 的分布密度为 $p(x|\theta,\varphi),\theta,\varphi$ 为未知参数，若
$$
p(x|\theta,\varphi)=g(x)h(\theta,\varphi)\exp\{t(x)\phi(\theta,\varphi)+u(x)\chi(\theta,\varphi)\}
$$
则称 $X$ 的分布属于两参数指数分布族
##### 一个定理
设随机变量$x$ 具有单参数指数分布，$X_1,X_2$,L ,$X_n$ 是取自总体$x$的一组子样，则统计量$\sum_{i=1}^nt(X_i)$是参数$\theta$的充分统计量
这里的$t(X_i)$ 是定义指数分布族的时候里面的函数形式
如果$\Theta^*$作为$\mathbb{R}^k$的子集有内点 此时这里的充分统计量也是完备的；
也就是对于指数分布族而言 充分统计量和完备统计量非常的接近
**这个定理用来给出充分完备统计量效果很好，单参数指数分布族非常广泛**
### 一致最小方差无偏估计
C-R不等式告诉了我们
* 统计量的方差存在下界
* 事实上不是每个参数都有有效估计 因为不是任何无偏估计都能达到C-R下界（不给出例子了）
因此我们提出了两个问题
* 已知一个无偏估计 能不能构造一个新的无偏估计 其方差比原来的小
* 一个无偏估计哪怕不有效 但是他的方差已经达到了最小
#### Rao-Blackwell 定理
##### Rao-Blackwell 定理
设$X$和$Y$是两个随机变量 $\boldsymbol{EX}=\boldsymbol{\mu},\mathbf{Var}(X)>\boldsymbol{0}$  定义$$\varphi(y)=E(X\mid Y=y)$$
则有$E\varphi(Y)=\mu,\mathrm{Var}(\varphi(Y))\leq\mathrm{Var}(X)$
等号成立的条件是${:}X\text{和 }\phi(Y)\text{几乎处处相等}.$
他提供了缩减方差的办法
给出证明
设$X,Y$ 的联合密度为$p(x,y)$ $X$ 的条件密度为$h(x|y)$ 则有
$$\varphi(y)=E\left(X\mid Y=y\right)=\int xh(x\mid y)dx=\int x\frac{p(x,y)}{p_{Y}(y)}dx$$
因此可以给出
$$\begin{aligned}E\phi(Y)&=\int\varphi(y)p_Y(y)dy\\&=\int\int xp(x,y)dxdy=EX=\mu\end{aligned}$$
$$\begin{aligned}\operatorname{Var}(X)&=E\left[(X-\varphi(Y))+(\varphi(Y)-\mu)\right]^2\\&=E\left(X-\varphi(Y)\right)^2+E\left(\varphi(Y)-\mu\right)^2\\&\color{}{+}2E[(X-\varphi(Y))(\varphi(Y)-\mu)]\end{aligned}$$
针对后半部分单独计算有
$$\begin{aligned} 
&E[(X-\varphi(Y))(\varphi(Y)-\mu)] \\
&=\int\int[x-\varphi(y)][\varphi(y)-\mu]p(x,y)dxdy \\
&=\int\int[x-\varphi(y)][\varphi(y)-\mu]p_Y(y)h(x\mid y)dxdy \\
&=\int[\phi(y)-\mu]\{\int[x-\phi(y)]h(x|y)dx\}p_Y(y)dy=0
\end{aligned}$$
因此
$$\begin{aligned}\operatorname{Var}(X)&=E\left(X-\varphi(Y)\right)^2+\operatorname{Var}(\varphi(Y))\\\operatorname{Var}(X)&=\operatorname{Var}(\varphi(Y))\quad\Leftrightarrow P\left(X-\varphi(Y)=0\right)=1\end{aligned}$$

##### 利用充分统计量
设总体概率密度函数是$p(x;\theta),X_1,X_2$, ,$X_{n}$  是其样本，$T=T(X_{1}...X_{n})$ 是$\theta$的充分统计量 $S(X_{1}...X_{n})$ 是参数$g(\theta)$的一个无偏估计 则
$$\varphi(T)=E(S(X)|T(X))$$
是$g(\theta)$ 的一个无偏估计  并且
$$Var_{\theta}(\varphi(T))\leq Var_{\theta}(S(X)),$$
当且仅当$P(\varphi(T)=S(X))=1$ 时等号成立
这个定理告诉我们 条件期望法进行方差缩减的时候 利用充分统计量时非常好的选择 这告诉了我们选取条件$Y$的方式
更一般的：
如果无偏估计不是充分统计量的函数，则将之对充分统计量求条件期望可以得到一个新的无偏估计，该估计的方差比原来的估计的方差要小，从而降低了无偏估计的方差。换言之，考虑$\theta$的估计问题只需要在基于充分统计量的函数中进行即可，该说法对所有的统计推断问题都是正确的，这便是所谓的**充分性原则**
##### 一个例子
设$X_{1}...X_{n}$是来自$b(1,p)$ 的样本 则$\overline{X}$ 是$p$ 的充分统计量 求$\theta=p^2$的无偏估计
我们知道样本均值 和 样本方差都是总体均值和方差的无偏估计 则有
$$E\overline{X}=E(X)=p;ES^{*^2}=D(X)=p(1-p)$$
两者做差有 期望为$p^2$ 这就是我们构造出来的无偏估计
在后面研究UMVUE的过程中 用类似的手法构造无偏估计非常的常用
#### 最小方差无偏估计
现在我们来回答另一个问题 能否在无偏估计类的全体中找到一个达到最小方差的估计量
##### UMVUE定义
对于参数估计问题，设$\hat{\theta}$是$\theta$的一个无偏估计，如果对另外任意一个$\theta$的无偏估计$\tilde{\theta}$, 在参数空间$\Theta$上都有
$$Var(\hat{\theta})\leq Var(\tilde{\theta})$$
则称其为最小方差无偏估计（Uniformly Minimum Variance Unbiased Estimate） （UMVUE）
* 如果UMVUE存在 则一定是充分统计量的函数
* 对于无偏估计 如果方差能达到C-R下界 则它一定是UMVUE
* UMVUE 的方差不一定能达到C-R下界
##### 判断UMVUE
设$X=(X_1,X_2,\cdots,X_n)$是来自某总体的样本一个$\hat{\theta}=\hat{\theta}(X)$是$\theta$的一个无偏估计，Var$( \hat{\theta} ) < + \infty.$如果对任意满足$E(\phi(X))=0$ 的$\phi(X)$ 都有
$$\color{}{\mathrm{Cov}_\theta(\widehat{\theta},\varphi)=0},\quad\forall\theta\in\Theta,$$
则$\hat\theta$是$\theta$的UMVUE
##### 构造UMVUE
设$T(X)$是充分完备统计量，$S(X)$是$g(\theta)$的无偏估计，则$\varphi(T)=E_{\theta}(S(X)|T(X))$是$g(\theta)$的UMVUE  
进一步的
如果对所有 $\theta\in\Theta$,$Var_{\theta}(\varphi(T))<\infty$, 则$\varphi(T)$是$g(\theta)$唯一的$UMVUE$
这个定理告诉我们 可以利用充分完备统计量构造对应的UMVUE
并且UMVUE在概率意义下唯一
事实上 这个定理告诉了我们两种寻找UMVUE的方法 不过都需要先找到充分完备统计量$T(X)$
* 统计量函数法 :若$\varphi(T(X))$是$g(\theta)$无偏统计量，则$\varphi(T(X))$ 也是$g(\theta)$的UMVUE. 即寻找充分完备统计量的函数使之成为$g(\theta)$的无偏估计.
* 条件期望法 :若能获得$g(\theta)$的一个无偏估计量$\varphi(X)$, 则$E(\varphi(X)|T(X))$就是$g(\theta)$的UMVUE.
##### UMVUE与C-R下界
有些UMVUE能达到C-R下界 但是也有的UMVUE达不到
一般来说 达到C-R下界的无偏估计量就是UMVUE 前提是C-R下界存在
不能片面的因为没有达到C-R下界就否定其是UMVUE
能达到C-R下界的无偏估计值得我们的重视 其被称为有效估计
##### 例子
[本部分例题](/blog/2024/09/24/probability-and-statistics-exercises-notes/) 的“数理统计UMVUE”一节

### 区间估计
区间估计这一节会对假设检验的内容进行一些引入，也会对后面才会讲解的内容提前运用一些，不过还是属于参数估计的范畴
点估计总是有偏差的 我们使用均方误差衡量偏差程度；但是仍旧缺乏可靠性的概念，均方误差多小算准确，并没有固定的标准；区间估计则是按一定的可靠性程度对待估参数给出一个区间范围，这便是整个区间估计一节要讲述的内容
#### 区间估计的基本概念
##### 置信区间的定义
设总体$X$的分布函数$F(x;\theta)$含有一个未知参数$\theta$,对于给定值$\alpha\left(0<\alpha<1\right)$,若由样本$X_1,X_2,\cdots$, $X_n$ 确定的两个统计量
$\underline{\theta}=\underline{\theta}(X_1,X_2,\cdots,X_n)$和$\text{}\bar{\theta}=\overline{\theta}(X_1,X_2,\cdots,X_n)\text{ }$满足
$$
P\{\underline{\theta}(X_1,X_2,\cdots,X_n)<\theta<\overline{\theta}(X_1,X_2,\cdots,X_n)\}=1-\alpha,
$$
 则称随机区间($\underline{\theta},\overline{\theta})$是$\theta$的置信度为1-$\alpha$的置信区间，$\underline{\theta}$ 和 $\overline{\theta}$分别称为置信度为1$-\alpha$ 的双侧置信区间 分别称为置信上限和置信下限
 * 待定参数是确定的 但是未知（这点和贝叶斯学派的观点不同，他们认为参数是随机变量）区间是随机的
 * 因此我们不能说参数有$1-\alpha$的概率落入随机区间 因为这暗含了参数有随机性的意思 我们应该说随机区间有$1-\alpha$的概率包含参数
 * 如果反复抽样得到大量的区间 那么包含参数的区间大约占比为$1-\alpha$这是伯努利大数定律决定的
##### 置信区间的求解步骤
###### 一
寻求一个样本 $X_1,X_2,...,X_n$的函数 :
$$
Z=Z(X_1,X_2,\cdotp\cdotp,X_n;\theta)
$$
 仅包含待估参数 $\theta$,并且 $Z$ 的分布已知且不依赖任何未知参数（包括$\theta$）
 明显的 这不符合统计量的定义了  我们称为枢轴量 
 ###### 二
 对于给定的置信度 1-$\alpha$,定出两个常数$a,b$, 使 $$P\{a<Z(X_1,X_2,\cdots,X_n;\theta)<b\}=1-\alpha.$$
 $a,b$可以任意找 只需要满足我们的区间要求
###### 三
若能从$a<Z(X_1,X_2,\cdots,X_n;\theta)<b$得到等价的不等式 $\underline{\theta}<\theta<\overline{\theta}$, 其中 $\theta=\theta(X_1,X_2,\cdots,X_n)$, $\overline{\theta}=\overline{\theta}(X_1,X_2,...,X_n)$都是统计量，那么 $(\underline{\theta},\overline{\theta})$ 就
是 $\theta$的一个置信度为 1- $\alpha$ 的置信区间
就是一个解不等式的过程
###### 几个说明
* 不同的置信水平$\alpha$ 参数$\theta$对应的置信区间不同
* 置信区间越小 估计越精确 对应的置信水平就会降低 反之亦然
* 想要不降低置信水平的缩小置信区间 只能靠增大样本容量解决
* 同一置信度，置信区间也不唯一 我们选择长度最小（精度最高的）
* 枢轴量一般是通过统计量变形得到的，这比较考验思考的水平，因此如何构造枢轴量并不是我们需要掌握的重点
#### 正态总体均值的区间估计
##### 总体方差已知
如果总体$X$服从$N(\mu,\sigma^2)$ 其中$\sigma_{2}$已知 则取$U$统计量 
$$U=\frac{\bar{X}-\mu}{\sigma/\sqrt{n}}$$
对参数$\mu$做区间估计
对于给定的置信水平$1-\alpha$ 有
$$P\{|U|<u_{1-\frac{\alpha}{2}}\}=1-\alpha $$
*为什么是这样的的结果，我们有时候需要理论说明*
则得到置信区间有
$$\color{}{\left(\bar{X}-u_{\alpha/2}\frac\sigma{\sqrt{n}},\bar{X}+u_{\alpha/2}\frac\sigma{\sqrt{n}}\right)}$$
*置信区间的得到就是不断重复处理解不等式的问题*
理解$u$值的意思就知道后面应该如何计算了
置信区间只有这一种吗？ 并不是 容易看出下面也是一种置信区间
$$\left(\overline{X}-\frac\sigma{\sqrt{n}}u_{0.01},\overline{X}+\frac\sigma{\sqrt{n}}u_{0.04}\right)$$
我们之所以不这样选择 是因为这样的置信区间长度更大 所以精度不够高
事实上置信区间的选择有无数种 但是我们后面只会选择长度最短的那一种
##### 总体方差未知
这时候需要选取$t$统计量用于检验 容易知道 
$$\frac{\overline{X}-\mu}{S^{\color{red}}/\sqrt{n}}\sim t(n-1);$$
因此给出置信度为$1-\alpha$的置信区间
$$\color{}{\left(\bar{X}-\frac{S_n^*}{\sqrt{n}}\cdot t_{\alpha/2}(n-1),\quad\bar{X}+\frac{S_n^*}{\sqrt{n}}\cdot t_{\alpha/2}(n-1)\right)}$$
理解$t$值的意思就知道应该如何计算了 计算过程的推导就省略了

#### 两个正态总体均值差的区间估计
##### 两个方差均已知的情况
$$\overline{X}\sim N(\mu_1,\frac{\sigma_1^2}n),\quad\overline{Y}\sim N(\mu_2,\frac{\sigma_2^2}m)$$
很容易构造枢轴量有
$$\frac{(\bar{X}-\bar{Y})-(\mu_1-\mu_2)}{\sqrt{\frac{\sigma_1^2}n+\frac{\sigma_2^2}m}}\sim N(0,1)$$
因此可以得到$\mu_1-\mu_2$的置信区间有
$$\left((\overline{X}-\overline{Y})-u_{1+\frac a2}\sqrt{\frac{\sigma_1^2}n+\frac{\sigma_2^2}m},\quad(\overline{X}-\overline{Y})+u_{1+\frac a2}\sqrt{\frac{\sigma_1^2}n+\frac{\sigma_2^2}m}\right)$$
##### 方差未知但是两个分布的方差相等
$$\overline{X}\sim N(\mu_1,\frac{\sigma^2}n),\quad\overline{Y}\sim N(\mu_2,\frac{\sigma^2}m)$$
给出枢轴量
$$\frac{(\overline{X}-\overline{Y})-(\mu_1-\mu_2)}{\sqrt{\frac1n+\frac1m}\sqrt{\frac{(n-1)S_1^2+(m-1)S_2^2}{n+m-2}}}\sim t(n+m-2)$$
理论推理得到
$$P\left(\left|\left.\frac{(\overline{X}-\overline{Y})-(\mu_1-\mu_2)}{\sqrt{\frac1n+\frac1m}\sqrt{\frac{(n-1)S_1^2+(m-1)S_2^2}{n+m-2}}}\right|<t_{1-\frac\alpha2}\right)=1-\alpha\right. $$
给出置信区间
$$\left((\overline{X}-\overline{Y})\pm t_{1+\frac\alpha2}\sqrt{\frac1n+\frac1m}\sqrt{\frac{(n-1)S_1^2+(m-1)S_2^2}{n+m-2}}\right)$$
##### 方差未知但是样本量均足够大
当样本量足够大（一般认为均大于50的时候）我们可以用样本修正方差取代真实方差 回归到我们方差已知的情况 直接给出参数估计区间有
$$\left((\overline{X}-\overline{Y})\pm u_{1-\frac\alpha2}\sqrt{\frac{S_1^2}n+\frac{S_2^2}m}\right)$$
##### 方差未知但是抽样数相等
$$\overline{X}\sim N(\mu_1,\frac{\sigma_1^2}n),\quad\overline{Y}\sim N(\mu_2,\frac{\sigma_2^2}m)$$
并且有
$$n=m$$
令$Z_{i}=X_{i}-Y_{i}$
我们可以认为现在的样本$Z_{i}$都来自于
$$Z\sim N(\mu_{1}-\mu_{2},\sigma_{1}^{2}+\sigma_{2}^{2})$$
我们可以认为此时我们正在进行 样本方差未知的的正态分布的均值估计
选取$t$统计量
$$\frac{\overline{Z}-\mu}{S^{\color{red}}/\sqrt{n}}\sim t(n-1);$$
变形前的公式为
$$\color{}{\left(\overline{Z}-\frac{S_z^*}{\sqrt{n}}\cdot t_{\alpha/2}(n-1),\quad\overline{Z}+\frac{S_
z^*}{\sqrt{n}}\cdot t_{\alpha/2}(n-1)\right)}$$
换入我们需要的量得到估计区间
$$\left((\overline{X}-\overline{Y})\pm t_{1+\frac\alpha2}(n-1)\frac{S_Z}{\sqrt{n}}\right)$$
#### 正态总体方差的区间估计
总体$X$服从$N(\mu,\sigma^2)$ 
我们只需要介绍$\mu$未知的情况 
*对于均值已经知道的情况 在统计量一节给出了一个类似的形式可以用于构造枢轴量 自由度和系数会变化*
前面我们介绍过的一个卡方分布作为枢轴量
$$\frac{(n-1)S^2}{\sigma^2}\sim\chi^2(n-1)$$
 因此有
 $$P\left\{\chi_{\alpha/2}^2(n-1)<\frac{(n-1)S^2}{\sigma^2}<\chi_{1-\alpha/2}^2(n-1)\right\}=1-\alpha $$
 计算得到置信区间为
 $$\left(\frac{(n-1)S^2}{\chi_{1-\alpha/2}^2(n-1)},\frac{(n-1)S^2}{\chi_{\alpha/2}^2(n-1)}\right)$$
 开方就可以得到标准差的置信区间
 $$\left(\frac{\sqrt{n-1}S}{\sqrt{\chi_{1-\alpha/2}^2(n-1)}},\frac{\sqrt{n-1}S}{\sqrt{\chi_{\alpha/2}^2(n-1)}}\right).$$
 这是我们介绍的第一个非对称的枢轴量 后面的$F$枢轴量也一样的处理
 我们还是选取了双侧的对称分位点来用于确定置信区间 
 这只是一种选取习惯的问题
#### 两个正态总体方差比的区间估计
我们还是只讨论总体均值都是未知的情况 
*虽然我们还是可以仿照前面的均值已知的方差估计研究构造出新的枢轴量 很容易猜到 均值已经知道的时候自由度会增加*
还是根据我们在统计量一节中介绍的结论给出枢轴量
$$\frac{\frac{S_X^2}{\sigma_{x}^2}}{\frac{S_Y^2}{\sigma_{y}^2}}\thicksim F(n_1-1,n_2-1)$$
因此 给出不等式
$$P\left\{F_{\alpha/2}(n_1-1,n_2-1)<\frac{S_1{}^2/{\sigma_1}^2}{S_2{}^2/{\sigma_2}^2}<F_{1-\alpha/2}(n_1-1,n_2-1)\right\}=1-\alpha $$
置信区间为
$$\color{}{\left(\frac{S_1^2}{S_2^2}\frac1{F_{1-\alpha/2}(n_1-1,n_2-1)},\frac{S_1^2}{S_2^2}\frac1{F_{\alpha/2}(n_1-1,n_2-1)}\right)}.$$
#### 单侧置信区间
此时我们确定置信区间的核心不等式变为
$$\begin{aligned}P(\underline{\theta}<\theta)&=1-\alpha\quad(\text{ 或 }P(\theta<\overline{\theta})=1-\alpha)\end{aligned}$$
还是很好理解的
此时$\underline{\theta}$ 称为单侧置信下限 $\overline{\theta}$称为单侧置信上限
构造枢轴量的方式对于单侧置信区间不会发生变化 只是不等式的确定改变了
单侧置信区间仍旧符合我们对置信区间定义的要求
他实际上和双侧置信区间具有同等的价值 我们后面在假设检验处就会介绍了
#### 比例的置信区间（大样本容量）
当总体$X$的分布未知 但是样本容量很大（一般认为是50以上） 根据中心极限定理我们知道
$$\overline{X}\sim N(\mu,\frac{\sigma^{2}}{n})$$
实际上我们又回归了前面正态总体的问题 
使用估计量$\overline{X}$来构造需要的枢轴量 最后确定参数的估计区间

给出一个例子 研究某个比率（往往被归结为两点分布的参数$p$）是在统计学中一个很常用的问题

设分布$X$服从参数为$p$的两点分布 样本为$X_{1},...,X_{n}$ $n>50$  求参数$p$置信度为$1-\alpha$的置信区间

这并不是一个正态分布的问题 但是可以使用中心极限定理辅助我们研究，容易知道
$$\overline{X}\sim N(p,\frac{p(1-p)}{n})$$
*注意，我们要带入的是两点分布的期望和方差，不是二项分布的，要理解*
现在要处理的问题就是 方差未知 正态分布均值的区间估计 吗？

虽然方差未知 但是他里面只含我们想要估计的参数 事实上这种情况应该当作方差已知进行处理 构造$U$统计量作为枢轴量
$$P\left(-u_{1-\frac\alpha2}\right.<\frac{\sqrt{n}(\overline{X}-p)}{\sqrt{p(1-p)}}<u_{1-\frac\alpha2})\approx1-\alpha $$
得到关于$p$的方程（和前面的思路还是有点区别的）
$$\begin{aligned}0\leq\frac{n(\overline{X}-p)^2}{p(1-p)}<u_{1-\frac{\alpha}2}^2\end{aligned}$$
化简
$$(n+u_{1-\frac\alpha2}^2)p^2-(2n\overline{X}+u_{1-\frac\alpha2}^2)p+n\overline{X}^2<0$$
解方程就能得到想要的区间估计 这里就不解这样含有很多未知数的方程了
## 假设检验

### 假设检验介绍和假设的介绍
当我们对参数毫无了解的时候 一般会采用上一章的参数估计方法来处理 
但是当参数估计完成后 对参数有了基本的了解 我们想要知道我们的估计是否正确 这就是这一章假设检验需要解决的问题 
#### 什么是假设？
对总体参数的具体数值所作的陈述 
比如总体均值大于某个数 总体方差小于多少等等 需要是一个可以证明或者证伪的命题
#### 什么是假设检验?(Hypothesis Test)
先对总体的参数(或分布形式)提出某种假设，然后利用样本信息判断假设是否成立的过程

分为了参数检验和非参数检验两种 其区别和非参数统计与参数统计的区别一样

**逻辑上运用反证法，统计上依据小概率原理**
#### 原假设和备择假设
null hypothesis     alternative hypothesis

**原假设是我们收集证据想要反对的假设，拒绝可信而接受不可信**

在我们后面的假设检验问题中 
认为一般是含有等号的式子等式 如$\mu=10,\mu\ge10$ 

备择假设是我们想要支持的假设 
在我们后面的假设检验问题中 
认为一般是不含有等号的式子等式 如$\mu\ne10,\mu>0$ 

原假设和备择假设一定是相互对立的  他们是一个完备事件组


### 假设检验一个例子
我们先用一个例子来叙述假设检验的各个过程

某厂生产的螺钉,按标准强度为68, 而实际生产的强度$X$ 服$N(m,3.6^2 )$.   若$E(X)=\mu=68$,则认为这批螺钉符合要求,否则认为不符合要求.现从整批螺钉中取容量为36的子样,若均值分别为为: 69.5和67.5 请问这批螺钉符合要求吗？

我们提出两个假设 原假设 $\mu=68$ 备择假设 $\mu\ne68$ 

现在我们从两个假设中选择其中一个 进行假设检验 ，我们假设原假设 $\mu=68$ 正确  （选择原假设）

那么此时有
$$\overline{X}\sim N(68,3.6^2/36)$$
现在构造一个矛盾 也就是某个小概率事件发生了 也就是 
其中按照小概率要求 这里的概率$\alpha=0.05$ 
$$P\left(\left|\frac{\overline{X}-68}{3.6/6}\right|>u_{\frac\alpha2}\right)=\alpha $$
解这个概率不等式就可以得到 接受域与拒绝域 
### 补充解释一些名词
#### 两类错误
非常明显的 我们的检验结果（是否接受假设）这件事情是完全被子样的情况确定的 那么就会出现两种情况

* 原假设是真的 但是由于抽样原因被拒绝了（第一类错误 也是就弃真）
* 原假设是假的 但是由于抽样原因被接受了（第二类错误 也是就取伪）

我们一般记犯第一类错误的概率是$\alpha$ 犯第二类错误的概率是$\beta$
#### 小概率原理
在一次试验中，一个几乎不可能发生的事件发生的概率称为小概率
在一次试验中小概率事件一旦发生，我们就有理由拒绝原假设 
小概率是我们自己决定的 就是下面的显著性水平
#### 显著性水平
就是我们在前面认定为小概率事件的那个小概率$\alpha$ 
这是由我们提前确定的 一般取$0.01~ ~0.05~ ~0.1$
就是犯第一类错误的概率 这也是为什么我们这里复用符号 因为值一样

#### 检验统计量
检验统计量(test statistic)是根据样本观测结果得到的 
用来对原假设和备择假设作出决策的样本统计量 它还是一个统计量 但是我们使用它有着特别的目的
他是原假设$H_0$为真的时候 某个构造出来的已知的统计量
选择合适的统计量时经典统计学假设检验问题核心

#### 拒绝域
拒绝域是能够拒绝原假设的检验统计量的所有可能的取值的集合，拒绝域的边界称为临界值

### 关于两类错误的概率
#### 定性分析
第二类错误概率 
* 随着假设的总体参数的减少而增大
* 随第一类错误减少而增大
* 随总体标准差增大而增大
* 随样本容量减少而增大

我们要知道 

**确定的子样容量下，不可能同时减少犯两类错误的概率**
**通过增大子样的大小 可以减少第二类误差**

下面的计算过程可以说明
#### 计算两种错误的概率
我们直接用一个例子来说明就好了
例子：$X_1,X_2...X_{n}\sim N(\mu,\sigma^2)$ 其中$\sigma^2$ 已知
$$H_0:\mu=\mu_0~H_1:\mu>\mu_0$$
拒绝域取为 $\bar{x}\ge c_{0}$  
* 求犯两类错误的概率 
* 在$\mu_0=0.5,\sigma=0.2,\alpha=0.05,n=9,\mu=0.65$时计算不犯第二类错误的概率

我们只需要从定义入手进行分析 使用检验统计量进行一些辅助化简

第一类错误时弃真 也就是原假设成立 但是拒绝 
第二类错误是取伪 也就是原假设不成立 但是接受
$$\begin{aligned}
&\alpha=P\left(\bar{X}\ge c_0|H_0真\right) \\
&=P_{\mu_{0}} \left(\overline{x}\geq c_{0}\right) \\
&=P\left(\frac{\overline{x}-\mu_{0}}{\sqrt{\frac{\sigma^2}{n}}}\geq\frac{c_0-\mu_{0}}{\sqrt{\frac{\sigma^{2}}{n}}}\right)=1-\phi\left(\frac{c_{0}-\mu_{0}}{\sqrt{\frac{\sigma^{2}}{n}}}\right)
\end{aligned}$$
$$\begin{aligned}
&\beta=P\left(\bar{X}\le c_0|H_1真\right) \\
&=P_{\mu} \left(\overline{x}\leq c_{0}\right) \\
&=P\left(\frac{\overline{x}-\mu}{\sqrt{\frac{\sigma^2}{n}}}\leq\frac{c_0-\mu}{\sqrt{\frac{\sigma^{2}}{n}}}\right)=1-\phi\left(\frac{c_{0}-\mu}{\sqrt{\frac{\sigma^{2}}{n}}}\right)
\end{aligned}$$

**从此能看出第一类错误和第二类错误此消彼长的规律；因此第一类错误的概率也不能无限压低，对应了第二类错误概率会爆炸**

我们就是用最基本的定义计算两种错误的概率
然后通过检验统计量向某种标准分布化简 最后得到两类错误的概率

**能看出第一类错误和第二类错误中的$c_0$是完全未知的，因为拒绝域要通过选定第一类错误的概率来确定，在实际的假设检验问题中我们正是先控制第一类错误的概率来得到拒绝域的**

**想要计算两类错误的具体值，对于第一类错误，我们不需要计算，这个表达式实际上是一个方程，第一类错误的概率是人为指定的；对于第二类错误给定的$\alpha$代入解出$c_0$然后代入$\beta$的表达式就可以得到结果**

**如果我们直接给出拒绝域（或者前面的问题中计算出了结果），那么就可以计算第一二类错误的概率（重新按照定义推导），这种情况下往往是看第一类错误概率是否得到了适当的控制**
### 总体均值的假设检验
#### 方差已知 均值取值的假设检验
给出两个假设
$$H_0\colon\mu=\mu_0;\quad H_1\colon\mu\neq\mu_0$$
构造检验统计量
$$U=\frac{\overline{X}-\mu_0}{\sigma/\sqrt{n}}\sim N(0,1)$$
在接受原假设的时候 检验统计量$\overline{X}$ 的情况已知 修正后的分布是$N(0,1)$ 

推导拒绝域 也就是小概率事件发生的区间
$$P_{H_{0}}(\left|\frac{\overline{X}-\mu_{0}}{\sigma/\sqrt{n}}\right|\geq u_{\frac{\alpha}{2}})=\alpha $$
通过变形就可以得到统计量$\overline{X}$的拒绝域

称为$U$检验 因为检验统计量是$U$统计量
#### 方差未知 均值取值的假设检验
构造检验统计量
$$T=\frac{\overline{X}-\mu_{0}}{S^{*}/\sqrt{n}}  \sim t(n-1)$$
根据小概率原理很容易给出拒绝域
检验这个统计量是否落在拒绝域就可以了
称为$t$检验
#### 双正态母体均值是否相等的假设检验
##### 当方差未知但是相等的时候
检验统计量为
$$\frac{\overline{X}-\overline{Y}}{\sqrt{\frac1n+\frac1m}\sqrt{\frac{(n-1)S_1^{*2}+(m-1)S_2^{*2}}{n+m-2}}}\sim t(n+m-2)$$
##### 大子样的情况下
根据中心极限定理知道 两个均值都是服从正态分布的 因此可以构造$U$统计量
$$U=\frac{\bar{X}-\bar{Y}}{\sqrt{\frac{S_1^{*2}}n+\frac{S_2^{*2}}m}} \sim N(0,1)$$
### 总体方差的假设检验
#### 均值已知的情况下 方差取值的假设检验
检验统计量
$$\chi^2=\frac{\sum_{i=1}^n(X_i-\mu)^2}{\sigma_0^2}\sim\chi^2(n)$$
这里把方差里面的$n$和辅助构造的$n$约分掉了
#### 均值未知的情况下 方差取值的假设检验
检验统计量
$$\chi^{2}=\frac{(n-1)S^{*2}}{\sigma_{0}^{2}}\sim\chi^{2}(n-1)$$
#### 大样本情况下
对于已知我们构造的检验统计量服从$\chi^2(n)$的情况下 
当抽样样本足够大的时候 根据中心极限定理有
$$\frac{\chi^2-n}{\sqrt{2n}}\overset{\text{}}{ \operatorname* { \sim }}N(0,1)$$
因此把上面计算的检验统计量带入到这里就有新的检验统计量了
所有的卡方检验在大样本的情况下都可以这样构造转换为$U$检验
#### 双正态母体方差比的假设检验
检验统计量
$$F=\frac{S_1^{*2}/\sigma_1^2}{S_2^{*2}/\sigma_2^2}=\frac{S_1^{*2}}{S_2^{*2}}\sim F(n-1,m-1)$$
使用这里检验方差是否相等就可以返回去检验均值是否相等了；
可以理解为方差比是的检验是研究均值相等检验不可缺少的一环
### 配对数据的$t$检验
配对数据的$t$检验是统计学中应用最广泛的假设检验方式之一，重要的我们在这里单独耗费笔墨进行介绍
#### 如何配对
我们往往希望把条件接近的两个受试对象
其目的是尽量避免除了实验处理以外的因素影响实验结果
#### 配对设计资料的三种情况
* 配成对子的同对受试对象分别给予两种不同的处理，其目的是推断两种处理的效果有无差别
* 同一受试对象分别接受两种不同处理，其目的是推断两种处理的效果有无差别
* 同一受试对象处理前后的比较，其目的是推断某种处理有无作用
**可以看出，配对资料的$t$检验应该非常广泛的用于各种实验数据的处理**
#### $t$检验
配对设计的$t$检验研究的是两组数据之间的均值的差异
如果我们对两组数据作差（和区间估计的做法一样）就变成了一组整体
我们想要研究的是这组差值总体的均值是否为0的假设检验 因此有
$$t=\frac{\overline{d}-\mu_{d}}{s_{\overline{d}}}=\frac{\overline{d}-0}{s_{d}/\sqrt{n}}=\frac{\overline{d}}{s_{d}/\sqrt{n}}\quad\sim{{t(n-1)}}$$
就是未知方差总体均值的假设检验
具体的例子在这里就不叙述了 理解这种检验的思想就足够了

### 单侧假设检验
#### 双侧和单侧假设检验
备择假设没有方向性 采用符号$\ne$ 的称为双侧检验或者双尾检验(two-tailed test)
备择假设具有特定的方向性，并含有符号$>~<$ 称为单侧检验或单尾检验(one-tailed test)

其中使用符号 $<$ 的称为左侧检验 使用$>$ 的称为右侧检验
#### 实现单侧检验
单侧假设检验的方法是非常自然的 在检验统计量不变的情况下我们要转换拒绝域的构造

是否使用单侧检验需要看假设检验的具体研究对象是否有这样使用的需求；

规定某种导线其电阻的标准差不得超过0.2 现需通过子样值检验这批导线的标准差是否显著偏大
明显的 我们应该取备择假设为 $H_1:\sigma^2>0.2^2$ 
我们的目的是否认偏大 所以应该把偏小取为原假设

食盐自动包装机包装的每袋标准重量不少于500 在机器调整后取一个子样,  检验平均每袋重量是否显著偏低？
我们的目的是否认偏小 也就是 $H_1{:}\mu{<}500.$

这就是需要使用单侧假设检验的情况
### 非正态总体的假设检验
#### 比例的假设检验（大样本的情形）
还是使用中心极限定理再归为正态总体假设检验的范畴，我们只介绍一种情况 也就是01分布参数的假设检验
设总体为
$$P\bigl\{X=x\bigr\}=p^{x}\left(1-p\right)^{1-x},\quad x=0,1$$
两个假设选择为
$$H_{0}:p=p_{0},\quad H_{1}:p\neq p_{0}$$
根据中心极限定理我们知道 当原假设成立且样本容量充分大的时候
$$U=\frac{\overline{X}-p_{0}}{\sqrt{p_{0}(1-p_{0})/n}}.$$
近似服从标准正态分布$N(0,1)$ 

可以使用$U$检验导出假设检验的拒绝域 
变换就可以得到样本均值对应的拒绝域
#### 指数分布参数值的假设检验
假设检验的核心是构造检验统计量 他的来源并不是靠感觉实现的 首先我们需要保证检验统计量的分布情况已知，其次我们需要大概至少 检验统计量怎么偏移的时候是不合理的 下面是一个例子
$$p(x)=\left\{\begin{array}{ll}
\lambda e^{-\lambda x}, & x\ge0, \\
0, & \text { 其他. }
\end{array}\right.$$
由于我们已知$E(X)=\frac{1}{\lambda}$
所以非常自然的思想是 根据均值的偏移方向和均值变形后可能的分布来构造检验统计量量
不妨设原假设为 $H_0:\lambda=\lambda_1$
那么 均值过大或者过小的时候 拒绝原假设  又
$$2\lambda(x_{1}+\cdots+x_{n})=2n\lambda\bar{x}\sim \chi_{2n^{2}}$$
后面拒绝域的选取就不是什么难事了
构造检验统计量非常依赖人们的的经验和对各种分布的掌握


### 假设检验的实际应用
#### 假设检验的p值
在我们前面进行的假设检验中 我们是先给出允许的第一类错误，然后确定拒绝域来进行的；

在目前使用的统计学软件中，我们往往是给出$p$值来进行假设检验的

$p$的计算方法为，根据我们观测到的值，反推此时拒绝域的理论边界位置，然后根据样本值恰好位于拒绝域边缘的假设计算我们控制的第一类错误概率$\alpha$  此时计算得到的值就是$p$值，因此$p$小于规定的显著性就可以判断显著

在原假设为真的条件下，P值是服从$[0,1]$区间上的均匀分布的，这点需要根据检验统计量来进行推导
#### 关于样本容量
在$p$值越来越小的时候，检验的结果越来越显著

非常明显的，$p$值随着样本容量的增大会变小 同等的$p$值样本容量不同我们的理解上是不一样的，进行假设检验的时候一定要给出样本容量的大小，实际上在样本容量的选取上，我们还需要讨论统计功效的问题, 一个统计的功效我们一般使用第二类错误的概率$\beta$ 来计算 使用$1-\beta$ 来作为一个假设检验的统计功效.

### 检验的效应与最小样本量
在假设检验中，**最小可检测效应（Minimum Detectable Effect, MDE）** 是指在给定样本量、显著性水平($\alpha$)、统计功效($1-\beta$)的条件下。检验能够显著检测到的最小效应大小。

最小可检测效应大小为：
$$\Delta_{\min}=\sqrt{\frac{2\sigma^2}{n}}\cdot(z_{\alpha/2}+z_\beta)$$
这个公式是根据统计功效相关内容构造对应的分布得到的，讨论的内容是最普通的双正态总体方差已知，均值差的Z test，研究其余检验情况基本类似。其中
* $z_{\alpha/2}$是显著性水平对应的正态分布分位数。如果单边研究则是$z_\alpha$ 如果考虑其他检验则对应的分布也需要更换，$\alpha$是显著性水平即第一类错误概率，由于这里研究的$z$分布的对称性所以将$z_{1-\alpha/2}$写作$z_{\alpha/2}$
* $z_\beta$是统计功效对应的正态分布分位数，统计功效是$1-\beta$ 即第二类错误概率与1的差，由于这里研究的$z$分布的对称性所以将$z_{1-\beta}$写作$z_{\beta}$
* $\sigma ^{2}$是总体方差，$n$是单组样本量。

如果考虑研究T检验，那么其需要更换为T分布，由于其对称性依旧可以简写，但是需要补充T分布的自由度，不同自由度的T分布并不一样。

卡方检验主要用于分类数据的独立性检验或拟合优度检验等，通常不直接用类似于 Z 检验和 T 检验那样的公式来定义最小可检测效应。一般采用Cramer's V 等来衡量效应大小。 如[描述性统计与可视化](/blog/2023/11/05/descriptive-statistics-and-visualization-notes/) 的“相关分析 / 列联表相关系数 / $V$相关系数”一节

最小可检测效应大小由显著性水平、统计功效、样本量和样本标准差决定。
* 样本量增加时或标准差减小的时候，$\Delta_{\min}$随之下降，表示实验更敏感
* 当显著性水平或统计功效要求更高时，$\Delta_{\min}$随之上升，表示需要更大效应才能被检测。
* MDE 帮助研究者评估实验设计的能力，以确保合理的样本量分配和资源优化。


根据最小可检测效应大小公式，提前给定我们希望检测到的最小预期效应$\Delta$ ，则上述公式可变为用于计算所需最小样本量的公式即
$$n=\frac{2\sigma^2}{\Delta_{\min}^2}\cdot(z_{\alpha/2}+z_\beta)^2$$
### 非参数假设检验
#### 简介
非参数假设检验是非参数统计中的一个部分
而非参数统计的关键点在于 **总体分布的未知性** 我们在这里只会介绍很少的一点非参数假设检验的内容 作为**非参数统计的一个引子** 
概率图纸法是一种经典的非参数检验方法 概率图纸使得来自正态总体的点在图纸上呈现为一条直线 他就是正态QQ（Quantile-Quantile）图检验 使用分位数研究样本是否来自正态总体
除此以外 我们再介绍Peason卡方拟合优度检验 这一比较基础的非参数检验方法
**这个方式是用来检验某个样本是否来自我们选定的分布的**
也是非参数统计中的一个经典问题
#### Peason卡方拟合优度检验的思想
我们划分随机试验的结果为一个完备事件组 $A_{1},A_2,...,A_k$
也就是 $\mathbf{\Omega}=\mathbf{\bigcup}_{i=1}^{\kappa}\mathbf{A}_{i},\quad A_{i}A_{j}=\mathbf{\Phi},i,j=1,2,\mathbf{L},k.$
我们假设在假设$H_0$下有
$$p_i=P(A_i)$$
那么分析实际频数和理论频数有
$$\chi^{2}=\sum_{i=1}^{k}\frac{(f_{i}-np_{i})^{2}}{np_{i}}$$
当他足够大的时候 偏离很大 那么原假设为假 反之原假设为真
#### Peason卡方拟合优度检验的定理
当$H_0$为真并且$n$充分大的时候 统计量
$$\chi^{2}=\sum_{i=1}^{k}\frac{(f_{i}-np_{i})^{2}}{np_{i}}=\sum_{i=1}^{k}\frac{f_{i}^{2}}{np_{i}}-n$$
近似服从$\chi^{2}(k-1)$
从这个检验统计量的分布就可以导出我们需要使用的拒绝域 
由于我们前面解释了 当偏离大的时候原假设为假 因此需要选择单侧假设检验 拒绝域为$\chi^{2}\geq\chi_{\alpha}^{2}(k-1)$

如果我们在选取假设$H_0$的时候 如果$X$的分布函数含有未知参数 
首先在假设下利用样本求出未知参数的最大似然估计, 以估计值作为参数值
我们给出结论当$H_0$为真并且$n$充分大的时候 统计量
$$\chi^{2}=\sum_{i=1}^{k}\frac{\left(f_{i}-n\hat{p}_{i}\right)^{2}}{n\hat{p}_{i}}=\sum_{i=1}^{k}\frac{f_{i}^{2}}{n\hat{p}_{i}}-n$$
近似服从$\chi^{2}(k-1-r)$
其中$r$是分布函数中含有的未知参数的个数 
拒绝域为$\chi^{2}\geq\chi_{\alpha}^{2}(k-1-r)$

#### 操作习惯
使用Peason卡方拟合优度检验一般需要保证下面的几个要求来进行样本的分类
* 大样本 一般认为$n>50$
* 要求各组的理论频数 $np_i>5$
* 一般数据分为7-14组 为了满足第二条可以小于七组
事实上将全部的样本数据进行合适的分组是Peason卡方拟合优度检验的核心

#### 补充内容
我们目前对卡方拟合优度的检验都只适用于理论分布只含有有限个值的情况

如果想要处理连续型变量，我们需要对区间进行分化，修正为有限个区间并计算各个区间的概率大小

Person卡方拟合优度检验的思想在列联表中应用很大，我们在列联表的相关及其他分析中再作介绍，他们都有着一个分类，频率结构
