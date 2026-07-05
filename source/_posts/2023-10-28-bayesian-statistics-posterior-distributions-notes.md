---
title: "贝叶斯统计：后验分布"
title_en: "Bayesian Statistics: Posterior Distributions"
date: 2023-10-28 17:36:48 +0800
categories: ["Data Science & Statistics", "Probability & Statistical Foundations"]
tags: ["Learning Notes", "Statistics", "Bayesian Statistics"]
author: Hyacehila
excerpt: "整理 Bayes 公式、先验分布、后验分布、边缘分布、共轭先验和贝叶斯计算基础。"
excerpt_en: "Covers Bayes formula, priors, posteriors, marginal distributions, conjugate priors, and Bayesian computation basics."
mathjax: true
hidden: true
permalink: '/blog/2023/10/28/bayesian-statistics-posterior-distributions-notes/'
---
## 绪论
### 引言    
#### Bayes公式
我们在概率论中学习过全概率公式
$$P(A)=P\bigg(\sum_{i=1}^{n}AB_{i}\bigg)=\sum_{i=1}^{n}P(A|B_{i})P(B_{i}).$$

并且以此为基础推出了概率论中的Bayes公式
$$P(B_i|A)=\frac{P(A|B_i)P(B_i)}{P(A)}=\frac{P(A|B_i)P(B_i)}{\sum_{j=1}^nP(A|B_j)P(B_j)}.$$

这个公式本来只是一个普通的推论 但是Bayes的研究赋予了这个公式更加深刻的思想

我们只看$P(B_{1}),\cdots,P(B_{n}),$ 他只是没有进一步信息（$A$的信息）下人们对时间$B$的认识 但是当我们拥有了关于事件$A$的信息时 我们对事件$B$有了更新的认识$P(B_1|A),\cdots,P(B_n|A)$

如果我们把A看作事件的结果 那么全概率公式是从事件的原因来研究事件的结果；Bayes公式则恰好和他相反 他的作用是从试验得到的结果$A$ 来推测某州原因发生的概率 

事实上这是非常常见的现代统计中需要研究的问题 我们用一个例子来介绍一下
	一种诊断某癌症的试剂，经临床试验有如下记录： 癌症病人试验结果是阳性的概率为 95%, 非癌症病人试验结果是阴性的概率为 95%. 现用这种试剂某社区进行癌症普查，设该社区癌症发病率为 0.5%, 问某人反应为阳性时，该如何判断他是否患有癌症？
这时候 患病与否是原因 指标阳性是结果 我们用Bayes公式计算一下知道
$$P(B_1|A)=0.087$$
对应了概率就是$0.913$ 根据概率的大小知道 这个病人应该没有患病

#### 三种信息
对于我们抽样进行的总体 他拥有自己的分布和分布参数 关于它的信息称为总体信息

对于从总体中抽取得到的样本 我们能得到样本信息

总体信息和样本信息在一起 得到抽样信息（sampling  information）

基于总体信息和样本信息进行统计推断的理论上方法称为经典统计学或者频率统计学 他从样本信息入手 研究总体的情况

另一种信息是先验信息 （prior information） 也就是在我们抽样之前 就已经对想要了解的统计推断问题有了一定的了解 他往往来自经验和历史资料 也可以被我们用于统计推断中

用抽样信息修正我们的先验信息就得到了后验信息 Bayes学派就是着手后验信息进行统计推断

这种利用先验信息的统计学派就是Bayes统计学

#### 历史
现在我们回顾一下Bayes统计的历史

在形式上 Bayes只是一个全概率公式的推论 但是Bayes 发现其中蕴含的归纳推理的思想 并且予以发表  后世的学者逐渐把它发展为来一种关于统计推断的系统理论和方法 称为Bayes方法 

这些方法就构成了Bayes统计 支持Bayes统计学的学者就形成了数理统计学中的Bayes学派 （Bayes School）

与Bayes学派相对的思想是 频率学派(Classical School) 两者的本质差异在于 
**Bayes 学派认为待估计参数是一个随机变量 而频率学派则认为其是一个确定的数 后面的种种差异都源于这一点**

#### 两个统计学派的争论
频率学派（古典学派）和贝叶斯学派是目前统计学中最大的两个学派 
坚持概率的频率解释 坚持通过大量重复来研究的学者都属于频率学派 
坚持先验信息的意义的学者都属于贝叶斯学派
两者的争论至今没有结果 我们着手介绍几个比较有意思的点
##### 频率学派对贝叶斯学派的批评
他们认为先验信息的确定 尤其是通过主观概率的方法来确定是问题非常大的  此时概率因人而异 和概率的频率解释相悖 没有客观性的科学缺乏客观的科学价值

此外 Bayes 学派也从样本分布为出发点 这就是概率的频率解释

Bayes学派的回应有下面几点
* 主观概率是人们非常常用的 符合习惯的 可以被理解的
* 统计推断与决策本身就是需要行动者承担后果 人们认知水平不同自然权衡得到的结果不同，此时强调客观性没有意义
* 非常多的频率学派的推断结果是其中特殊的Bayes解

##### 贝叶斯学派对的频率学派批评
* 大量的试验不可重复 研究频率没有意义
* 假设检验中精度是提前确定的 和样本无关 这不合理

### 贝叶斯统计推断的基本概念
贝叶斯统计与经典统计的不同就在于对参数先验信息的使用
从贝叶斯统计学派的观点来看，**一切统计推断必须从后验分布出发**
#### 先验分布和后验分布
参数空间上的任何概率分布都是先验分布 （prior distribution）

我们一般使用$\pi(\theta)$ 表示先验分布 密度函数还是分布列都这样表示 

如何确定先验分布会在后面进行介绍 

随着抽样进行 我们得到了样本$X$ 根据样本调整我们对$\theta$ 的认知 就可以得到后验分布（posterior distribution） 他是总体信息 样本信息 先验信息的综合

定义后验分布为
$$\pi(\theta|x)=\frac{h(x,\theta)}{m(x)}=\frac{f(x|\theta)\pi(\theta)}{\int_{\Theta}f(x|\theta)\pi(\theta)\mathrm{d}\theta}$$
其中的$m(x)$ 称为边缘分布 形式为
$$m(x)=\int_\Theta h(x,\theta)\mathrm{d}\theta=\int_\Theta f(x|\theta)\pi(\theta)\mathrm{d}\theta $$
对于离散的情况 可以使用分布列表示
$$\pi(\theta_i|x)=\frac{f(x|\theta_i)\pi(\theta_i)}{\sum_if(x|\theta_i)\pi(\theta_i)}\quad(i=1,2,\cdots).$$

事实上 离散情况的后验分布就是Bayes公式 也就是说我们的后验分布本质上是Bayes公式的一种推广

在这里面 $\pi(\theta)$ 是先验信息 $f(x|\theta)$ 是抽样信息 包含了样本信息和总体信息
$\pi(\theta|x)$ 就是我们所说的后验分布 整个公式就是对最基础的Bayes公式的一种推广

前面介绍的是单个样本的公式情况 ，当我们抽取了多个样本的时候 我们会在后面介绍统计推断的时候再进行详细的解释
#### 点估计
从贝叶斯统计学派的观点来看，**一切统计推断必须从后验分布出发**我们的贝叶斯点估计也是如此

定义：使后验密度 $\pi(\theta|x)$ 达到最大的值 $\hat{\theta}_{MD}$ 称为$\theta$ 的后验众数(Mode)估计； 后验分布的中位数 $\hat{\theta}_{_{Me}}$称为 $\theta$的后验中位数(Median)估计； 后验分布的期望(Expectation)值 $\hat{\theta}_E$ 称为 $\theta$ 的后验期望值估计，这三个估计都称为贝叶斯估计,记为$\hat\theta_{B}$

* 后验众数估计也称为最大后验估计
* 这三个估计一般是不同的 但是当后验密度是对称的时候 三种估计重合

使用这三种贝叶斯点估计来估计未知参数就是贝叶斯点估计的思想
#### 区间估计
从贝叶斯统计学派的观点来看，**一切统计推断必须从后验分布出发**我们的贝叶斯区间估计也是如此 

定义：可信区间
参数$\theta$的后验分布为$\pi(\theta|x)$,对给定的样本 $x$ 和概率 $1-\alpha(0<\alpha<1)$,若存在这样的两个统计量 $\hat{\theta}_L=\hat{\theta}_L(x)$ 与$\hat{\theta}_U=\hat{\theta}_U(x)$ 使得
$$
P(\hat{\theta}_L\leq\theta\leq\hat{\theta}_U\mid x)\geq1-\alpha 
$$
则称区间$\hat{\theta}_L,\hat{\theta}_U$为参数$\theta$的可信水平为 1- $\alpha$ 贝叶斯可信区间估计，或简称为$\theta$的 $1-\alpha$ 可信区间

满足$P(\theta\geq\hat{\theta}_L\mid x)\geq1-\alpha$ 的$\hat{\theta}_L$称为$\theta$的 $1-\alpha$ (单侧)可信下限
满足$P(\theta\leq\hat{\theta}_U\mid x)\geq1-\alpha$ 的$\hat{\theta}_U$称为$\theta$的$1-\alpha$ (单侧)可信上限

这就是贝叶斯区间估计的思想 后面为在这个基础上介绍不同的贝叶斯区间估计方法
#### 假设检验
1. 获得后验概率$\pi(\theta|x)$ 后 分别计算假设$H_0$ $H_1$ 的后验概率 $\alpha_i=P(\theta_i|x)$ 
2. 当后验概率比（机会比）$\frac{\alpha_0}{\alpha_{1}}>1$ 时不拒绝$H_0$  $\frac{\alpha_0}{\alpha_{1}}<1$  时不拒绝$H_{1}$ 接近$1$的时候不做判断 无法给出结论
这就是贝叶斯假设检验的核心思想

后面我们再介绍贝叶斯因子 辅助我们研究贝叶斯统计中的假设检验问题
#### 预测推断
**这里我们回应了边缘分布的意义**

事实上 我们在预测推断一节就是使用边缘分布来进行我们的预测工作的 在这一节里面我们称边缘分布为预测分布 使用边缘分布$m(x)$的期望 中位数 众数（还是极大）来作为我们的预测值

其思路可以按照如下解释
由于 $\pi(\theta|\boldsymbol{x})$ 为 $\theta$ 的后验分布，所以 $g(z|\theta)\pi(\theta|\boldsymbol{x})$ 为给定 $x$ 条件下$(Z, θ)$的联合分布. 把它对 $\theta$ 积分，得到给定 $x$ 时随机变量 $Z$ 的条件边缘分布，或称为后验预测密度
### 贝叶斯统计决策的基本概念
#### 统计判决三要素
**样本空间** $\chi$ 是样本可能取值的集合 **样本分布族**$f(x|\theta)$ 是样本的密度函数族
对于样本空间的一个事件$A$
$$\left.P(A|\theta)=\left\{\begin{array}{ll}\int_Af(x|\theta)\mathrm{d}x,&x\text{为连续随机变量},\\\\\sum_{x\in A}f(x|\theta),&x\text{为离散随机变量}.\end{array}\right.\right.$$

**行动空间**：我们对于某个统计决策问题可以采取的行动构成的非空集合
对于参数估计问题：行动空间是估计量的集合 对于假设检验问题：行动空间只有两个行动 接受与拒绝原假设

**损失函数**：定义在参数空间和行动空间$\Theta\times D$ 上的二元可测函数；评估了某参数取值下采取某个行动的损失 

损失函数的类型有很多我们在这里区分收益矩阵（函数）和损失矩阵（函数）收益可正可负 往往采用货币单位 正意味着收益 负意味着亏损

损失函数只有正值 我们用它描述 我们本该得到但是没有得到的收益 即
$$L(\theta,a)=\max Q(\theta,a)-Q(\theta,a)$$
其中的$Q$是收益函数 $L$是损失函数
#### 风险函数和一致最优决策函数
**决策函数** ：定义于样本空间而取值于决策空间的函数 $\delta=\delta(\boldsymbol{x})$ 

**风险函数** ：用来衡量某种决策的平均损失 用决策函数代入前面的损失函数并对样本分布求期望
$$R(\theta,\delta)=E[L(\theta,\delta(\boldsymbol{X}))]=\int_{\mathcal{E}}L(\theta,\delta(\boldsymbol{x}))\mathrm{d}F(\boldsymbol{x}|\theta)$$
$$\left.=\left\{\begin{array}{l}\int_{\mathcal{X}}L(\theta,\delta(\boldsymbol{x}))f(\boldsymbol{x}|\theta)\mathrm{d}\boldsymbol{x},\\\sum_{\boldsymbol{x}\in\mathscr{X}}L(\theta,\delta(\boldsymbol{x}))f(\boldsymbol{x}|\theta),\end{array}\right.\right.$$

根据Wald提出的统计决策理论 评价决策函数的唯一标准就是他的风险函数
风险函数当然是越小越好

如果一个风险函数$R(\theta,\delta)$ 在所有的$\theta$取值上都小于等于另一个风险函数
我们就称他的决策函数为更优的

如果能找到最小风险的决策函数 就称为**一致最优决策函数**
#### 贝叶斯期望损失和贝叶斯风险
定义：设 $\delta(\boldsymbol{x})$ 为 $\theta$ 的决策函数，$L(\theta,\delta(\boldsymbol{x}))$ 为损失函数，$F^{\pi}(\theta)$ 为$\theta$ 的先验分布函数 我们称下面的形式为贝叶斯期望损失
$$
R(\pi,\delta(\boldsymbol{x}))=\int_{\Theta}L(\theta,\delta(\boldsymbol{x}))\mathrm{d}F^{\pi}(\theta)
$$
$$\left.=\left\{\begin{array}{l}\int_{\Theta}L(\theta,\delta(\boldsymbol{x}))\pi(\theta)\mathrm{d}\theta,\\\sum_iL(\theta_i,\delta(\boldsymbol{x}))\pi(\theta_i),\end{array}\right.\right.$$

它和风险函数的概念并不相同 因为均值针对于$\theta$的先验进行计算 前面针对样本分布求期望

定义：风险函数$R(\theta,\delta)$  $F^{\pi}(\theta)$ 为$\theta$ 的先验分布函数  称
$$\begin{gathered}
R_{\pi}(\delta(\boldsymbol{x})) =\int_\Theta R(\theta,\delta(\boldsymbol{x}))\mathrm{d}F^\pi(\theta)=E^\pi[R(\theta,\delta(\boldsymbol{X}))] \\
=\int_\Theta\int_{\mathscr{X}}L(\theta,\delta(\boldsymbol{x}))\boldsymbol{f}(\boldsymbol{x}|\theta)\mathrm{d}\boldsymbol{x}\mathrm{d}F^\pi(\theta) 
\end{gathered}$$
为贝叶斯风险

他是把风险函数针对先验密度函数再进行了一次期望，和贝叶斯期望损失并不一样
#### 贝叶斯解
如果有一个决策函数让贝叶斯风险最小
我们就称这个决策函数为统计决策问题的贝叶斯解
若先验分布为广义先验分布，对应求得的贝叶斯解称为广义贝叶斯解

### 贝叶斯统计计算
这里我们需要贝叶斯统计衍生出的一些统计计算方法 

介绍的统计算法的目的都是为了解决贝叶斯统计中的一些计算问题；而贝叶斯统计中最核心的问题就是关于后验分布的计算以及后验分布数字特征，我们主要介绍的算法为 EM算法 与 MCMC

### 似然比检验
我们在数理统计中学习过各种检验方法 比如针对正态总体的假设检验，对$p$值的检验， 还有一些非参数检验方法；[数理统计](/blog/2023/03/18/mathematical-statistics-notes/) 的“假设检验”一节 
不过 我们在介绍了极大似然估计以后确实没有介绍一种非常重要的检验方法：**似然比检验** 我们在贝叶斯统计中补充他
#### 似然比统计量
对于最基本的假设检验问题：
$$H_{0}:\theta\in\Theta_{0}\leftrightarrow H_{1}:\theta\in\Theta_{1},$$

我们考虑类似极大似然估计中的思想 [数理统计](/blog/2023/03/18/mathematical-statistics-notes/) 的“极大似然估计”一节 考虑在样本下的两种假设的似然函数
$$L_{\Theta_{0}}\left(x\right)=\sup_{\theta\in\Theta_{0}}f\left(x,\theta\right),\\L_{\Theta_{1}}\left(x\right)=\sup_{\theta\in\Theta_{1}}f\left(x,\theta\right).$$
构造似然比统计量有
$$\lambda\left(X\right)=\frac{\sup_{\theta\in\Theta}f\left(X,\theta\right)}{\sup_{\theta\in\Theta_{0}}f\left(X,\theta_{0}\right)}$$
当似然比统计量比较大的时候 我们自然有着拒绝原假设的倾向 因为其对应的似然更小；
我们可以自然的给出检验函数为
$$\left.\varphi\left(x\right)=\begin{cases}1,&\lambda\left(x\right)>c,\\r,&\lambda\left(x\right)=c,\\0,&\lambda\left(x\right)<c\end{cases}\right.$$

问题的核心现在是**研究似然比统计量的分布，或者其等价形式的分布，据此确定我们的拒绝域**
#### 似然比检验
一般的 $\lambda(X)$ 表达式比较复杂，计算他的分布是非常困难的；因此我们给出结论：**若$\lambda(X)=g(T(X))$ 为$T(X)$的单调递增（递减）函数 那么检验函数可以自然变形为**
$$\left.\varphi\left(x\right)=\begin{cases}1,&T\left(x\right)>c,\\r,&T\left(x\right)=c,\\0,&T\left(x\right)<c.\end{cases}\right.$$
递减的时候将$\phi(x)$中不等号反向

如果分布无法具体确定 使用极限分布也是可以接受的 后面会介绍这一点本文“似然比的极限分布”一节

#### 似然比统计量的一个例子
设$X=\left(X_{1},X_{2},\cdots,X_{n}\right)$ 是从正态分布族 $\{N\left(\mu,\sigma^{2}\right),$ $-\infty< \mu<+\infty,\sigma^{2}>0\}$ 中轴取的 i.i.d. 样本，求下列检验问题
$$
H_{0}:\mu=\mu_{0}\leftrightarrow H_{1}:\mu\neq\mu_{0}
$$
的似然比检验

似然函数为
$$f\left(x,\theta\right)=\left(2\pi\sigma^{2}\right)^{-\frac{n}{2}}\exp\left\{-\frac{1}{2\sigma^{2}}\sum_{i=1}^{n}\left(x_{i}-\mu\right)^{2}\right\},$$

两个假设下的极大似然估计分别为
$$\widehat{\mu}=\overline{X},\widehat{\sigma}^{2}=\frac{1}{n}\sum_{i=1}^{n}(x_{i}-\overline{X})^{2};$$
$$\tilde{\sigma}^{2}=\frac{1}{n}\sum_{i=1}^{n}\left(x_{i}-\mu_{0}\right)^{2}.$$

因此给出两个假设下的似然有
$$\sup_{\theta\in\Theta}f\left(x,\theta\right)=f\left(x,\widehat{\mu},\widehat{\sigma}^{2}\right)=\left(\frac{2\pi e}{n}\right)^{-\frac{n}{2}}\left(\sum_{i=1}^{n}\left(x_{i}-\bar{x}\right)^{2}\right)^{-\frac{n}{2}},
\\\sup_{\theta\in\Theta_{0}}f\left(x,\theta\right)=f\left(x,\mu_{0},\tilde{\sigma}^{2}\right)=\left(\frac{2\pi e}{n}\right)^{-\frac{n}{2}}\left(\sum_{i=1}^{n}\left(x_{i}-\mu_{0}\right)^{2}\right)^{-\frac{n}{2}}$$
因此有似然比统计量为
$$\lambda\left(X\right)=\left(1+\frac{1}{n-1}T^{2}\right)^{\frac{n}{2}}$$
其中的$T$为$$T=\sqrt{n}\left(\overline{x}-\mu_{0}\right)/\sqrt{\frac{1}{n-1}\sum_{i=1}^{n}\left(x_{i}-\overline{x}\right)^{2}}.$$
因此 我们可以使用统计量$T$来进行似然比检验 要求
$$P\left(\left|T\right|>c\left|H_{0}\right)=\alpha.\right.$$
当原假设成立的时候 $T\sim t_{n-1}$  因此
$$\varphi\left(X\right)=\left\{\begin{matrix}1,&\left|T\right|\geqslant t_{n-1}\left(\alpha/2\right)\\0,&\left|T\right|<t_{n-1}\left(\alpha/2\right)\end{matrix}\right.$$

#### 似然比的极限分布
似然比在原假设成立的时候的分布并不一定容易计算 其精确分布有时候可能无法求解 这也是假设检验中经常遇到的情况；

但是 如果 我们的样本是 i.i.d 的 则可以用似然比的极限分布来研究

定理: 设 $\Theta$ 的维数为 $k,\Theta_0$ 的维数为 $s$ ,若 $k-s=t>0$, 且样本分在满足一定的正则条件，则对似然比检验问题,在原假设 $H_{0}$ 成立之下，当样本 $n\to\infty$的时候
$$
2\ln\lambda\xrightarrow{}\chi_{t}^{2}.
$$


## 先验分布的选取
### 主观概率
#### 介绍
**主观概率是人们根据经验对事件发生概率的机会的推测** 比如一场打赌一场球赛的输赢 股票的涨跌 这些随机现象都是不可重复的 无法使用频率来研究概率

此时我们实际上抛弃了经典统计学中概率的频率定义，但是这算得上是对传统概率定义的补充（频率无法观测）而且符合我们的直观感受（实际上主观概率经常被大家自然的采用）

只有在什么先验相关的资料都没有得情况下，我们才需要使用主观概率 
#### 利用相对似然性
概率具有公理化定义 

如果我们知道一个事件只有两面 $\overline{A} ~and~A$ 并且前者的概率是后者的两倍
那么我们根据公理化定义解方程就可以得到概率分别是 $\frac{1}{3}~ \frac{2}{3}$  这就是利用相对似然性

*似然 英文likelihood 也就是可能性 统计中的专用词汇 不能和概率混淆 概率是我们在研究概率论中的内在存在的特征，似然是从样本统计的角度得到的特征*

这种方法一般只在理论上存在价值
#### 利用专家建议
这是确定主管概率的最核心方法
评估多个相关领域专家的建议并且综合他们
专家的主观概率一般并普通人更加准确
#### 利用历史资料
研究类似问题的历史情况进行评估

这是使用概率的频率解释吗？  并不是
我们这里利用的是类似事件的历史资料 而不是对目前研究的事件进行多次观测研究频率

事实上我们目前研究的事件无法重复试验才会产生了贝叶斯统计学

### 利用先验信息
利用先验信息确定确定先验分布需要我们对先验有一定的了解 

直方图和核密度曲线需要使用主观概率（专家的）或者历史信息来辅助研究
相对似然法是对前面利用相对似然这种主观信息的拓展

定分度和变分度法都需要专家给出多次判断

通过超参数确定先验分布还是需要一些历史信息

如果我们把历史信息替换成真正的关于先验的抽样信息那么才是真正的利用先验信息了
**综合主观概率一节和利用先验信息一节来理解他们的思想，他的想法是接近的**
#### 直方图与核密度曲线
适用于参数空间为有限子区间并且先验信息足够多，根据主观概率或者历史信息获得先验分布在某个区间时发生的概率

根据这些先验信息绘制直方图或者核密度曲线 他们都是一种非参数的估计 估计原始分布的情况

此时我们可以根据推测的原始分布情况研究需要的先验概率
#### 相对似然法
这种方法一般适用于先验分布落在有限区间的情况 是对前面利用相对似然性的扩展 目标是得到先验分布的草图

比如 我们知道先验分布落在区间$[0,1]$ 上 那么可以根据可能性绘制似然图 一般我们会用最小似然来作为1 相对似然图如下 我们需要进行规范化才可用
![贝叶斯统计](/assets/images/probability-statistics-notes/bayesian-statistics-posterior-distributions-notes-01.png)

#### 定分度法和变分度法
定分度法和变分度法都是依据专家咨询得到各种主观概率 然后加工成概率曲线的方法 其实还是对主观概率思想的继续扩展 他们非常的相似 

定分度法是将参数可能取到的区间分成等长的区间，在每一个小区间上邀请专家给出主观概率

变分度法是把所有的区间逐渐二分为机会均等的两个小区间（区间不需要等长） 此时专家还需要给出分点

一般我们更多采用变分度法 这样能更好的驱动专家思考给出更优秀的回答

在拥有了专家给出的划分后 我们能很容易根据这些信息完成概率直方图的构造
#### 先确定先验分布再确定超参数
这是一种使用非常广泛的方法 当然我们需要对先验分布有一定的了解
##### 超参数
我们称先验分布中的参数为超参数（Hyperparamater）

**在机器学习中，超参数的概念就来自贝叶斯统计学，其中的参数指的是在模型学习的过程中会自动确定的数，超参数指的是那些模型中需要提前确定的数，设计超参数自动调节管道就可以自动确定超参**

这一节我们的思路是 首先假定（有一定依据的假定）$\theta$的先验密度是 $\pi(\theta)$ 其中有待定的参数 $\mu$我们后面只需要去定这个超参数$\mu$ 就知道先验分布了

当然 容易看出来 这个方法的核心是选定先验分布 $\pi(\theta)$  如果这里选定失误 那么后面的估计将会偏差巨大
##### 确定先验分布形式
根据参数空间的特点确定
参数空间为 $(-\infty,\infty)$ 的分布： 正态分布，柯西分布 (均值和方差不存在),学生 $t$ 分布等；
参数空间为 (0,\infty) 的分布： 指数分布，威布尔分布，伽马分布等；
参数空间为 $0,1,\ldots$ 的分布： 泊松分布，几何分布等；
##### 确定超参数
###### 矩估计超参数
从先验信息中加工得到前几阶先验分布的样本矩 使用样本矩等于总体矩的方法解方程确定超参数 这就是矩估计 只是应用的贝叶斯统计中了
###### 分位数估计超参数
这是另一种对超参数的估计思想

我们知道总体分位数一定是超参数的函数 而样本分位数可以使用样本信息加工得到 我们令两者相等就可以解方程确定超参数

**分位数估计实际上是经典统计学中一个还算重要的分支，只是在大多数数理统计教材中没有进行介绍**

**确定超参数的方法当然不止这两种 我们可以结合两者进行估计或者使用各种不同的数理统计中的思想 比如极大似然估计**

#### 使用分位数确定先验CDF
如果有较多的先验分布中位数 可以近似拟合CDF曲线 拟合后的CDF曲线就反映了先验分布的特征 和本文“直方图与核密度曲线”一节 思想接近
### 利用边缘分布
**边缘分布缺少实际含义，对边缘分布抽样只在理论上存在可能性，实际上这一小节的知识在实际应用过程中使用的比较少**
#### 边缘分布
##### 定义
我们前面给出了边缘分布的定义 现在来复习一下 本文“先验分布和后验分布”一节

当随机变量$X$有概率密度函数$f(x|\theta)$ 先验分布密度函数为$\pi(\theta)$  则可以定义随机变量的边缘分布为
$$\begin{aligned}
m\left(x\right)& =\int_{\Theta}f(x|\theta)\mathrm{d}F^{\pi}(\theta)  \\
&\left.=\left\{\begin{array}{ll}\int_\Theta f(x|\theta)\pi(\theta)\mathrm{d}\theta,&\theta\text{ 为连续随机变量},\\\sum_if(x|\theta_i)\pi(\theta_i),&\theta\text{ 为离散随机变量}.\end{array}\right.\right.
\end{aligned}$$
边缘分布是结合抽样信息和先验信息得到的 
##### 混合分布
当随机变量$X$以概率$p$在$F({x|\theta_{1}})=F_{1}$中取值 $1-p$的概率在$F({x|\theta_{1}})=F_{2}$中取值 我们根据混合分布的要求知道 $X$的混合分布函数为$$F(x)=pF(x|\theta_1)+(1-p)F(x|\theta_2)$$
能看出 边缘分布其实就是混合分布的一种推广形式 限制$\theta$为离散型变量 边缘分布就是可列个概率密度函数混合的结果

当$\theta$是连续性变量时 $m(x)$是由无限个不可数个密度函数混合而成
##### 边缘分布计算的一个例子
设给定$\theta$的时候样本$X$ 服从正态分布$N(\theta,\sigma^{2})$ 其中$\sigma$已知（这就是抽样信息） 如果$\theta$的先验分布为$N(\mu_\pi,\sigma_\pi^2)$ 计算边缘分布$m(x)$ 
$$\begin{aligned}
m(x)& =\int_{-\infty}^{\infty}f(x|\theta)\pi(\theta)\mathrm{d}\theta   \\
&=\frac{1}{2\pi\sigma\sigma_{\pi}}\int_{-\infty}^{\infty}\exp\left\{\left.-\frac{1}{2}\left[\frac{(x-\theta)^{2}}{\sigma^{2}}+\frac{(\mu_{\pi}-\theta)^{2}}{\sigma_{\pi}^{2}}\right]\right\}\mathrm{d}\theta\right.  \\
&=\frac{1}{2\pi\sigma\sigma_{\pi}}\int_{-\infty}^{\infty}\exp\left\{-\frac{A}{2}\left(\theta-\frac{B}{A}\right)^{2}\right\}\cdot\exp\left\{-\frac{1}{2}\left(C-\frac{B^{2}}{A}\right)\right\}\mathrm{d}\theta  \\
&=\frac1{\sqrt{2\pi(\sigma^2+\sigma_\pi^2)}}\exp\left\{-\frac{(x-\mu_\pi)^2}{2(\sigma^2+\sigma_\pi^2)}\right\}
\end{aligned}$$
其中
$$A=\frac{1}{\sigma^2}+\frac{1}{\sigma_{\pi}^2},\quad B=\frac{x}{\sigma^2}+\frac{\mu_{\pi}}{\sigma_{\pi}^2},\quad C=\frac{x^2}{\sigma^2}+\frac{\mu_{\pi}^2}{\sigma_{\pi}^2}$$
也就是
$$N(\mu_\pi,\sigma^2+\sigma_\pi^2)$$

还是构造一个密度函数的积分最后值为1 这是概率与统计里面处理积分的经典方法
##### 边缘概率
设某电子元件的失效时间 $X$ 服从指数分布 $Exp(1/\theta)$, 其密度函数为 $f(x|\theta)=\theta^{-1}\mathrm{e}^{-x/\theta}(x>0)$, 若未知参数 $\theta$ 的先验分布为逆伽马分布 $\Gamma^{-1}(1,100)$计算该元件在时间 200 之前失效的边缘概率。
**边缘概率是边缘分布在对应区间的积分**
也就是这里实际上计算的结果应该是
$$\int_{0}^{200}m(x)dx =\frac{2}{3}$$

#### 选择先验分布的ML-II方法
回归一下我们本节的正题 是研究先验分布选取 那么现在我们来介绍如何使用边缘分布来确定先验分布

我们的核心思想还是大似然更可能在抽样中发生（极大似然法）

也就是 我们要选取让抽样分布出现目前情况概率更大的先验分布 
##### 定义与方法
定义：
设$\Gamma$ 是我们考虑的先验类 对来自边缘分布的样本$x=(x_{1},x_{2}...x_{n})$ 若存在 $\hat{\pi}\in\Gamma$ 使得
$$m(\boldsymbol{x}|\hat{\pi})=\sup_{\pi\in\Gamma}\prod_{i=1}^nm(x_i|\pi)$$
则称$\hat{\pi}$是类型II 的最大似然先验 简称ML-II先验 

（实际上是把$m(x)$看作似然函数的一种极大似然先验）
如果先验密度函数形式已知 只是其中含未知的超参数 那么我们可以把上面问题简化为如下形式 其中$\Lambda$ 是超参数取值集合
$$m(\boldsymbol{x}|\hat{\lambda})=\sup_{\lambda\in\Lambda}m(\boldsymbol{x}|\lambda)=\sup_{\lambda\in\Lambda}\prod_{i=1}^nm(x_i|\lambda)$$
这就是极大似然的问题研究我们的超参数$\Lambda$的取值就可以了


为什么是一堆连乘？
根据基础边缘分布计算公式计算的边缘分布是对于一个边缘分布样本而言的
（当然我们前面的那个后验分布基础定义也是这样的），我们当然不可能只有一个边缘分布样本
**我们可以用极大似然估计的思想来理解，这是就是在用样本构造似然函数**

而简单随机样本自然独立(i.i.d)，大似然方法本身就是一堆概率的连乘然后极大化
对于多样本的连乘结果我们一般称为 **联合边缘密度函数**

##### 例子
设随机变量 $X\sim N(\theta,\sigma^2)$,其中 $\sigma^2$ 已知，又设 $\theta\sim N(\mu_\pi,\sigma_\pi^2)$.如果 $X=(X_1,\cdots,X_n)$ 为从边缘分布 $m(x|\lambda)$ 中抽取的 i.i.d. 样本，试确定 $\theta$ 的先验分布

首先我们研究边缘分布  $X$的边缘分布是$N(\mu_\pi,\sigma^2+\sigma_\pi^2)$ 本文“边缘分布计算的一个例子”一节

根据前面的方法给出我们需要进行极大化的函数，其中$\bar{x}$ 和$S^{2}$是抽出的样本的均值和方差 则
$$\begin{aligned}
&L(\mu_\pi,\sigma_\pi^2|\boldsymbol{x}) =m(\boldsymbol{x}|\boldsymbol{\lambda})  \\
& =\left[2\pi(\sigma^2+\sigma_\pi^2)\right]^{-n/2}\exp\left\{-\frac{1}{2(\sigma^2+\sigma_\pi^2)}\cdot\sum_{i=1}^{n}(x_i-\mu_\pi)^2\right\}  \\
&=\left[2\pi(\sigma^2+\sigma_\pi^2)\right]^{-n/2}\exp\left\{\frac{-nS^2}{2(\sigma^2+\sigma_\pi^2)}\right\}\cdot\exp\left\{-\frac{n(\bar{x}-\mu_\pi)^2}{2(\sigma^2+\sigma_\pi^2)}\right\},
\end{aligned}$$

很容易能看出来
如果$\sigma_{\pi}^{2}$固定 那么$\mu_{\pi}=\bar{x}$的时候最大化 不需要进行求偏导研究

带入$\mu_{\pi}=\bar{x}$
$$\phi(\sigma_{\pi}^{2})=\big[2\pi(\sigma^{2}+\sigma_{\pi}^{2})\big]^{-n/2}\exp\bigg\{\frac{-nS^{2}}{2(\sigma^{2}+\sigma_{\pi}^{2})}\bigg\}.$$
取对数求导数研究极大得到
$$\hat{\sigma}_\pi^2=S^2-\sigma^2$$
明显不可能取负 因此当$S^{2}$过小的时候取$0$即可

#### 选择先验分布的矩方法
这里的核心是研究边缘分布和先验分布矩之间的关系 从而通过对现实的抽样实现对先验分布参数的矩估计 和前面直接研究先验矩估计并不一样 此时我们缺乏直接对先验的直接了解 

**核心思想在于边缘分布矩估计 要把边缘分布矩表示位超参数的函数**
##### 理论推导
计算样本分布$f(x|\theta)$的期望和方差（我们视$\theta$是常数，对$x$求期望与方差）
$$\mu(\theta)=E^{X|\theta}(X),\quad\sigma^{2}(\theta)=E^{X|\theta}[X-\mu(\theta)]^{2},$$
计算边缘分布$m(x)=m(x|\lambda)$的期望和方差 (我们视$\lambda$（超参数）为常数对$x$求期望与方差)

$$\begin{aligned}
&\mu_{m}(\lambda) =E^{X|\lambda}(X)=\int_{\mathcal{C}}xm(x|\lambda)\mathrm{d}x=\int_{\mathcal{C}}\int_{\Theta}xf(x|\theta)\pi(\theta|\lambda)\mathrm{d}\theta\mathrm{d}x  \\
&=\int_\Theta\left[\int_{\mathscr{E}}xf(x|\theta)\mathrm{d}x\right]\pi(\theta|\lambda)\mathrm{d}\theta=\int_\Theta\mu(\theta)\pi(\theta|\lambda)\mathrm{d}\theta  \\
&=E^{\theta|\lambda}[\mu(\theta)], \\
&\sigma_{m}^{2}(\lambda) =E^{X|\lambda}\left\{[X-\mu_{m}(\lambda)]^{2}\right\}=\int_{\mathcal{E}}[x-\mu_{m}(\lambda)]^{2}m(x|\lambda)\mathrm{d}x  \\
&=\int_{\mathscr{X}}\int_{\Theta}[x-\mu_m(\lambda)]^2f(x|\theta)\pi(\theta|\lambda)\mathrm{d}\theta\mathrm{d}x \\
&=\int_{\Theta}\left\{\int_{\mathscr{X}}[x-\mu_{m}(\lambda)]^{2}f(x|\theta)\mathrm{d}x\right\}\pi(\theta|\lambda)\mathrm{d}\theta  \\
&=\int_{\Theta}E^{X|\theta}\left[x-\mu_{m}(x)\right]^{2}\pi(\theta|\lambda)\mathrm{d}\theta,
\end{aligned}$$
其中
$$\begin{aligned}
&E^{X|\theta}\left\{\left[x-\mu_m(\lambda)\right]^2\right\} =E^{X|\theta}\left(\left\{\left[x-\mu(\theta)\right]+\left[\mu(\theta)-\mu_{m}(\lambda)\right]\right\}^{2}\right)  \\
&=E^{X|\theta}\left\{\left[x-\mu(\theta)\right]^2\right\}+E^{X|\theta}\left\{\left[\mu(\theta)-\mu_m(\lambda)\right]^2\right\} \\
& =\sigma^2(\theta)+\left[\mu(\theta)-\mu_m(\lambda)\right]^2. 
\end{aligned}$$
因此方差实际上的表示为
$$\begin{gathered}
\sigma_{m}^{2}(\lambda) =\int_{\Theta}\sigma^{2}(\theta)\pi(\theta|\lambda)\mathrm{d}\theta+\int[\mu(\theta)-\mu_{m}(\lambda)]^{2}\pi(\theta|\lambda)\mathrm{d}\theta  \\
=E^{\theta|\lambda}\left[\sigma^2(\theta)\right]+E^{\theta|\lambda}\left\{\left[\mu(\theta)-\mu_m(\lambda)\right]^2\right\}. 
\end{gathered}$$
这里的参数$\lambda$是对先验分布中参数的统称 而不只是一个参数

如果我们的先验分布只有两个超参数 $\lambda_{1},\lambda_{2}$ 那么矩估计就只需要两个量，容易计算两个边缘分布样本矩（最基础的期望和方差）为
$$\hat{\mu}_{m}=\overline{X}=\frac{1}{n}\sum_{i=1}^{n}X_{i}\quad\text{和}\quad\hat{\sigma}_{m}^{2}=S^{2}=\frac{1}{n-1}\sum_{i=1}^{n}(X_{i}-\overline{X})^{2}$$

解边缘分布样本矩的方程就有我们需要知道的关于超参数的结果
$$\left.\left\{\begin{array}{l}\hat{\mu}_m=E^{\theta|\boldsymbol{\lambda}}\big[\mu(\theta)\big],\\\hat{\sigma}_m^2=E^{\boldsymbol{\theta}|\boldsymbol{\lambda}}\big[\sigma^2(\theta)\big]+E^{\boldsymbol{\theta}|\boldsymbol{\lambda}}\big[\mu(\boldsymbol{\theta})-\mu_m(\boldsymbol{\lambda})\big]^2\bigg\}.\end{array}\right.\right.$$

最后的公式的语言表述：
* 对于边缘分布均值的估计：我们计算抽样分布均值关于参数$\lambda$ 的期望
* 对于边缘分布方差的估计 ：抽样分布方差关于参数$\lambda$ 的期望  与 抽样分布均值关于参数$\lambda$ 的方差 的和

**抽样分布的均值和方差 都有自己的分布 他们和参数$\lambda$有关 这就是计算的原理**

##### 例子
设 $X|\theta\sim N(\theta,1)$,参数 $\theta$ 的先验分布取为 $N(\mu_\pi,\sigma_\pi^2)$,其中 $\lambda=(\mu_\pi,\sigma_\pi^2)$ 未知. 设 $X=(X_1,\cdots,X_n)$ 为从边缘分布 $m(x|\lambda)$ 中抽取的 i.i.d. 样本，由样本算出样本均值 $\bar{X}=10,S^{2}=3.$ 试确定 $\theta$ 的先验分布

采用矩估计的手段研究

样本分布的期望和方差分别为 $\theta$ 与 $1$

计算边缘分布的期望和方差有
$$\left.\left\{\begin{array}{l}\mu_m(\lambda)=E^{\theta|\lambda}(\theta)=\mu_\pi,\\\sigma_m^2(\lambda)=E^{\theta|\lambda}(\sigma^2(\theta))+E^{\theta|\lambda}[\theta-\mu_\pi]^2=1+\sigma_\pi^2.\end{array}\right.\right.$$

带入抽样的均值和方差有
$$\left.\left\{\begin{array}{l}10=\overline{X}=\mu_\pi,\\3=S^2=1+\sigma_\pi^2.\end{array}\right.\right.$$
因此$\theta$的先验分布是$N(10,2)$

### 无信息先验分布
贝叶斯统计的特点是在统计推断的过程中使用先验信息

但是有时候先验信息非常少或者没有 我们依然想使用贝叶斯统计的思想 这就是**无信息先验(noninformation prior) 即对参数空间的任何一点都不存在任何偏爱**
#### 贝叶斯假设和广义先验分布
无信息先验分布指我们的先验分布不包含任何$\theta$的信息 对任何值都不存在偏爱 
##### 定义
非常自然的 我们可以把$\theta$作为在取值范围上的 均匀分布 看作先验分布 ，这种看法就是贝叶斯假设（Bayes assumption） 一般有以下的情形
* 离散均匀分布 如果$\Theta$ 是有限集 那么离散均匀分布为 $P(\theta=\theta_i)=1/n$
* 有限区间均匀分布 如果$\Theta$ 是有限区间$[a,b]$ 那么有限区间均匀分布为$U(a,b)$
* 广义先验分布 如果$\Theta$ 无界 那么取$\pi(\theta)\equiv1$ 他不满足概率规范性所以是广义

定义：
如果$\theta$的先验分布$\pi(\theta)$ 满足
* $\pi(\theta)\equiv1$  并且$\int_{\Theta}\pi(\theta)\mathrm{d}\theta=\infty;$
* 后验密度$\pi(\theta|x)$是正常的密度函数
则称$\pi(\theta)$ 是$\theta$ 的广义先验密度 (improper prior density)

**容易知道 广义先验密度乘上任意一个常数还是广义先验密度**
##### 贝叶斯假设的不足
贝叶斯假设是存在缺点的 最大的缺点是其不确定性

如果我们对$p$同等无知 那么我们应该也对$p^2,p^3$同等无知 那么理论上我们取贝叶斯假设$U(0,1)$作为他们三者的分布 结果应该是不会发生变化的；很显然，在很多情况下并不是这样 这就是贝叶斯假设存在的不确定性

**贝叶斯假设不满足变换下的不变性**
例如： 考虑正态标准差 $\sigma\in(0,\infty)$,定义一个变换
$$
\eta=\sigma^2\in(0,\infty)
$$
则 $\eta$ 为正态方差
设 $\sigma$ 的先验密度函数为 $\pi(\sigma)$, $\eta$ 的先验密度函数为 $\pi^*(\eta)$, 那么$\eta$的密度函数可表示为
$$\pi^*(\eta)=\pi(\sqrt{\eta})\left|\frac{d\sigma}{d\eta}\right|$$

可以看出 不能随意设定一个常数为某参数的先验分布 也就是贝叶斯假设不能随便使用
#### 位置参数族的无信息先验
设总体 $X$ 的密度函数的形式为 $f(x-\theta)$,其样本空间 $\mathscr{X}$ 和参数空间 $\Theta$ 皆为实轴 R, 则此类密度函数构成的分布族称为位置参数族 (location parameter family), $\theta\in\Theta$ 称为位置参数.

下面就是两个位置参数族的例子
正态分布
$$\frac{1}{\sqrt{2\pi}\sigma}\exp\Big\{\frac{1}{2\sigma^2}(x-\theta)^2\Big\}=f(x-\theta)$$
Cauchy分布
$$\frac1\pi\cdot\frac\lambda{\lambda^2+(x-\mu)^2}=f(x-\mu)$$

容易看出 位置参数族在具有平移变换群下的不变性

对$X$做平移变换得到 $Y=X+c$  同时对参数$\theta$做平移变换得到 $\mu=\theta+c$ 容易知道 $Y$的密度函数形式为 $f(y-\mu)$ 还是位置参数族的成员并且样本空间和参数空间不变 

所以两者研究的统计问题结构相同 我们应当认为他们有着相同的无信息先验 下面我们来证明无信息先验密度应该取为$\pi(\theta)\equiv1$ 

令 $\pi$ 和$\pi^{*}$  分别表示 $\theta$ 与 $\eta$ 的无信息先验密度，以上论点说明 $\pi$ 和  $\pi^{*}$ 应有着相同的先验密度 也就是
$$\pi(\tau)=\pi^*(\tau)$$
由于前面的线性关系我们知道
$$\pi(\eta)=\pi^*(\eta)=\pi(\eta-c)$$
特别的 我们取$\eta=c$ 
$$\pi(c)=\pi(0)=\text{常数}.$$
因此我们取$\pi(\theta)\equiv1$ 是合理的
综上所述 当$\theta$ 是位置参数时 无信息先验取为常数或者1

#### 尺度参数族的无信息先验
设总体 $X$ 的密度函数的形式为 $\sigma^{-1}\varphi(x/\sigma)$, 其中 $\sigma>0$ 为刻度参数，参数空间为 $\mathbb{R}_+=(0,\infty)$, 则此类密度函数构成的分布族称为刻度参数族(scale parameter family)

下面是几个例子
均值为0的正态分布
$$\begin{aligned}
f(x|\sigma)& =\frac1{\sqrt{2\pi}\sigma}\exp\left\{-\frac{x^2}{2\sigma^2}\right\}  \\
&=\sigma^{-1}\left[\frac1{\sqrt{2\pi}}\exp\Big\{\left.-\frac12\left(\frac x\sigma\right)^2\right\}\right]=\sigma^{-1}\varphi\Big(\frac x\sigma\Big),
\end{aligned}$$
伽马分布
$$f(x|\lambda)=\frac{\lambda^{-r}}{\Gamma(r)}x^{r-1}\mathrm{e}^{-x/\lambda}=\lambda^{-1}\Big[\frac{1}{\Gamma(r)}\Big(\frac{x}{\lambda}\Big)^{r-1}\mathrm{e}^{-x/\lambda}\Big]=\lambda^{-1}\varphi\Big(\frac{x}{\lambda}\Big),$$

还是一样的原理 能看出刻度参数族在刻度变换群下的不变性

对$X$做变换 $Y=cX$ 对$\theta$作相应的变换 $\eta=c\sigma$  得到的$Y$ 还是刻度参数族的成员，因此两者选择一样的无信息先验是合理的

根据前文类似的证明手段 ，我们取
$$\pi(\sigma)=\frac{1}{\sigma}\quad(\sigma>0)$$
作为刻度参数族的无信息先验分布

#### 位置-尺度参数族
我们结合前面的内容接受一下位置-尺度参数族

设密度函数中有两个参数 $\mu$ 与 $\sigma$, 且密度具有下述形式：
$$
p(x;\mu,\sigma)=\frac1\sigma f\left(\frac{x-\mu}\sigma\right),\mu\in(-\infty,\infty),\sigma\in(0,\infty)
$$
其中 $f( x)$ 是一个完全确定的函数， $\mu$ 称为位置参数，$\sigma$ 称为尺度参数，这类分布族称为位置-尺度参数族

正态分布 Cauchy分布 指数分布 均匀分布都属于这一类

**特别地，$\sigma=1$ 时称为位置参数族，而$\mu=0$ 时称为尺度参数族，位置-尺度参数族是前面两者的综合情况**

他对应的无信息先验为 $\pi(\theta,\sigma)=\frac{1}{\sigma^{2}}$ 

#### 一般情形下的无信息先验
对于一般情形的无信息先验 Jeffreys的方法是最常用的 由于其推导涉及了很多关于抽象代数变换群和Harr测度的内容 我们下面只介绍方法
##### Jeffreys无信息先验
假定样本分布族 $\{f(x|\theta),\theta\in\Theta\}$ 满足 CR 正则条件 这里 $\theta=(\theta_1,\cdots,\theta_p)$ 为 $p$ 维参数向量. 设 $\boldsymbol{X}=(X_1,\cdots,X_n)$ 是从总体 $f(x|\theta)$ 中抽取的简单样本.
当$\theta$ 无先验信息可用时，Jeffreys 用Fisher信息阵行列式的平方根作为 $\theta$ 的无信息先验，这样的无信息先验称为

 Jeffreys 无信息先验求解步骤如下.
1. 写出参数$\theta$的对数似然函数
$$l(\boldsymbol{\theta}|\boldsymbol{x})=\ln\left[\prod_{i=1}^nf(x_i|\boldsymbol{\theta})\right]=\sum_{i=1}^n\ln f(x_i|\boldsymbol{\theta}).$$
2. 计算Fisher信息阵
$$I(\boldsymbol{\theta})=\left(I_{ij}(\boldsymbol{\theta})\right)_{p\times p},\quad I_{ij}(\boldsymbol{\theta})=E_{\boldsymbol{X}\mid\boldsymbol{\theta}}\Big\{-\frac{\partial^2l}{\partial\theta_i\partial\theta_j}\Big\}\quad(i,j=1,\cdots,p).$$
*对于单参数的情形 Fisher信息阵为$1\times1$矩阵*
$$I(\boldsymbol{\theta})=E_{\boldsymbol{X}|\boldsymbol{\theta}}\Big\{-\frac{\partial^{2}l}{\partial\boldsymbol{\theta}^{2}}\Big\}.$$
**拆这个$E$进去对$X$作用就好了 里面肯定有样本**
3. 计算信息阵的行列式的平方根作为无信息先验
$$\pi(\theta)=\left[\det I(\theta)\right]^{1/2}$$
*对于单参数情形* 
$$\pi(\theta)=[I(\theta)]^{1/2}.$$

Fisher信息量的定义式为
$$I(\theta)=E_\theta\left[\frac\partial{\partial\theta}\ln p(x;\theta)\right]^2$$
我们上面使用的形式是一个满足题意的等价形式 对Fisher信息量展开详细的研究的时候就会详细介绍定义式及推导 这里直接使用结论就好了
**没有强调样本数量的时候不妨只抽一个样本，如果多样本不好算**

##### 例子
设 $X=(X_1,\cdots,X_n)$ 是从总体 $N(\mu,\sigma^2)$ 中抽取的简单样本。记$\theta=(\mu,\sigma)$, 求 $(\mu,\sigma)$ 的联合无信息先验

计算对数似然函数有
$$l(\boldsymbol{\theta}|\boldsymbol{x})=-\frac n2\ln2\pi-n\mathrm{ln}\sigma-\frac1{2\sigma^2}\sum_{i=1}^n(x_i-\mu)^2.$$
求偏导再求期望求Fisher信息阵的各个元素
$$\begin{aligned}I_{11}(\boldsymbol{\theta})&=E_{\boldsymbol{X}|\boldsymbol{\theta}}\Big\{-\frac{\partial^2l(\boldsymbol{\theta}|\boldsymbol{x})}{\partial\mu^2}\Big\}=\frac{n}{\sigma^2},\\I_{22}(\boldsymbol{\theta})&=E_{\boldsymbol{X}|\boldsymbol{\theta}}\Big\{-\frac{\partial^2l(\boldsymbol{\theta}|\boldsymbol{x})}{\partial\sigma^2}\Big\}=-\frac{n}{\sigma^2}+\frac{3}{\sigma^4}E\Big\{\sum_{i=1}^n(X_i-\mu)^2\Big\}=\frac{2n}{\sigma^2},\\I_{12}(\boldsymbol{\theta})&=I_{21}(\boldsymbol{\theta})=E_{\boldsymbol{X}|\boldsymbol{\theta}}\Big\{-\frac{\partial^2l(\boldsymbol{\theta}|\boldsymbol{x})}{\partial\mu\partial\sigma}\Big\}=E\Big\{\frac{2}{\sigma^3}\sum_{i=1}^n(X_i-\mu)\Big\}=0,\end{aligned}$$
计算Fisher信息阵和行列式的平方根有
$$\left.I(\boldsymbol{\theta})=\left(\begin{array}{cc}\frac n{\sigma^2}&0\\0&\frac{2n}{\sigma^2}\end{array}\right.\right),\quad[\det I(\boldsymbol{\theta})]^{1/2}=\frac{\sqrt{2}n}{\sigma^2}.$$
由于广义先验可以自由调整常数 联合无信息先验为
$$\pi(\mu,\sigma)=1/\sigma^2,$$

还是这个例子  我们能看出

当$\sigma$已知的时候 属于位置参数族 无信息先验应为$\pi(\theta)\equiv1$ 实际上使用使用费希尔信息阵计算Jeffreys无信息先验也是这个结果

当$\mu$已知的的时候 属于尺度参数族 无信息先验应为$\pi(\sigma)=\frac{1}{\sigma}\quad(\sigma>0)$实际上使用使用Fisher信息阵计算Jeffreys无信息先验也是这个结果

因此当两者独立的时候 无信息联合先验密度为$\pi(\sigma)=\frac{1}{\sigma}\quad(\sigma>0)$
当两者不独立的时候 无信息联合先验密度为$\pi(\mu,\sigma)=1/\sigma^2,$

能看出 **无信息先验密度并不唯一**

**事实上 不同无信息先验对贝叶斯推断的影响很小 任何无信息先验都可接受**

**无信息先验是贝叶斯统计中最成功的部分之一**

**很多经典统计学中的估计方式可以被看作取某种无信息先验的贝叶斯估计**

### 共轭先验分布
这是一种从理论出发的确定先验的方式 在已知样本分布的情况下为了理论研究的需要 确定的先验分布

事实上高维情况下的边缘分布很不好计算
#### 定义
定义 设$\mathscr{F}$ 表示由 $\theta$ 的先验分布 $\pi(\theta)$ 构成的分布族。如果对任取的$\pi\in\mathscr{F}$ 及样本值 $x$, 后验分布$\pi( \theta|x)$  仍属于 $\mathscr{F}$ ,那么称 $\mathscr{F}$ 是一个共轭先验分布 (conjugate prior distribution family)

明显的 共轭先验分布族和样本分布族的取值情况有关 脱离样本分布族谈论先验分布族是否共轭是没有意义的

同时 共轭先验分布针对分布中的某一个参数而言的 对于含有多个参数的情况要注意理解

#### 共轭先验分布的例子
设$X\sim B(n,\theta)$ 
* 若 $\theta$ 服从均匀分布 $U(0,1)$, 证明： $\theta$ 的后验分布为贝塔分布；
* 若取 $\theta$ 的先验分布为贝塔分布 $Beta(a,b)$, 其中 $a,b$ 已知，证明： $\theta$ 的后验分布仍为贝塔分布，即 $\theta$ 的共轭先验分布为贝塔分布.

连续两问其实就是求后验分布的问题
1. 如下
样本分布是二项分布形式如下
$$f(x|\theta)=\binom nx\theta^x(1-\theta)^{n-x}\quad(x=0,1,\cdots,n)$$
先验分布是均匀分布 $\pi(\theta)=1$ 计算后验分布如下
$$\pi(\theta|x)=\frac{\theta^x(1-\theta)^{n-x}}{\int_0^1\theta^x(1-\theta)^{n-x}\mathrm{d}\theta}$$
计算积分有 数学分析中关于Gamma函数积分介绍过
$$\int_0^1\theta^x(1-\theta)^{n-x}\mathrm{d}\theta=\frac{\Gamma(x+1)\Gamma(n-x+1)}{\Gamma(n+2)}.$$
因此得到后验分布有
$$\pi(\theta|x)=\frac{\Gamma(n+2)}{\Gamma(x+1)\Gamma(n-x+1)}\theta^{(x+1)-1}(1-\theta)^{(n-x+1)-1}$$

2. 如下
样本分布不变 先验分布是Beta分布 还是一样的原理计算后验分布有
$$\pi(\theta|x)=\frac{\theta^{x+a-1}(1-\theta)^{n-x+b-1}}{\int_0^1\theta^{x+a-1}(1-\theta)^{n-x+b-1}\mathrm{d}\theta}$$
计算积分并且带入我们的结果得到后验密度有
$$\pi(\theta|x)=\frac{\Gamma(n+a+b)}{\Gamma(x+a)\Gamma(n-x+b)}\theta^{(x+a)-1}(1-\theta)^{(n-x+b)-1}$$
还是一个Beta分布 因此我们确定这是共轭先验分布族

#### 使用共轭先验确定后验分布
我们可以认为 样本$X$的边缘密度$f_{m}(x)$ 和$\theta$无关 也就是认为他是一个常数 故
$$\pi(\theta|x)=\frac{f(x|\theta)\pi(\theta)}{f_m(x)}\propto f(x|\theta)\pi(\theta)$$

定义 ：概率函数的核（Kernel）是这个概率函数只和参数有关的部分
*Eg.  样本概率函数$f(x|\theta)$的核 是$f(x|\theta)$中只和$\theta$有关的部分

对于共轭先验分布的情形 求后验密度可以采用以下的步骤

1. 写出样本概率函数$f(x|\theta)$的核 和 先验密度函数 $\pi(\theta)$ 的核
2. 使用前面结尾的公式给出后验密度的核 他是样本核和先验和的积
3. 添加一个正则化常数因子就得到了后验密度 （常数因子可能和$x$有关）

怎么添加正则化常数因子：共轭先验分布的后验密度的分布形式是已知的，我们去凑对应的后验密度函数就可以了

这种方法一般只适用于共轭先验分布的情况 

因为共轭先验时的 我们知道后验和先验的密度函数形式一样 我们知道后验的密度的分布 就容易给出常数因子

当非共轭先验 但是后验的核我们知道是某常用分布的核 也可以给出；

其他情况下 我们无法确定常数因子 需要按照后验的基本公式进行计算

具体的例子可以参考本文“后验分布的计算”一节

### 多层先验（分阶段先验）
#### 基本思路
当给定的先验分布的超参数难以确定的时候 可以针对超参数再给出一个先验 第二个先验称为超先验 如果超先验的参数还是难以确定我们可以继续套娃 根据先验和超先验一起决定的新先验称为多层先验（hierarchical prior）

多层先验的思路如下：
第一层先验
$F_1=\{\pi_1(\theta|\lambda):\pi_1$的函数形式已知，$\lambda\in\Lambda\}$,
其中 $\Lambda$ 为超参数 $\lambda$ 的取值范围，且 λ 未知

第二层先验
$\lambda$的先验分布是$\pi(\lambda)$ 不含任何未知参数

计算规范的二层先验有
$$\pi(\theta)=\int_{\Lambda}\pi_1(\theta|\lambda)\pi_2(\lambda)\mathrm{d}\lambda=\int_{\Lambda}\pi(\theta,\lambda)\mathrm{d}\lambda$$

核心就是通过多层的复合先验得到我们的规范先验分布 然后使用这个规范先验分布用于贝叶斯统计推断
#### 分层先验的例子
为了研究不合格率的先验分布 ，我们首先认为不合格率的先验是$U(0,1)$ 但是不合格率是很低的 所以这个先验不是很合理 因此选择使用多层先验来研究

我们认为 不合格率是 $U(0,\lambda)$ 其中超参数$\lambda$的先验为$U(0.1,0.5)$ 

计算规范化的先验。
$$\pi(\theta)=\int_{\Lambda}\pi_{1}(\theta|\lambda)\pi_{2}(\lambda)\mathrm{d}\lambda=\frac{1}{0.5-0.1}\int_{0.1}^{0.5}\lambda^{-1}I_{[0,\lambda]}(\theta)\mathrm{d}\lambda$$
其中$I$是一个示性函数 分几种情况计算先验的结果

$$\begin{aligned}&(a)\text{ 当 }0<\theta<0.1\text{ 时},\\&\pi\left(\theta\right)=\frac{1}{0.4}\int_{0.1}^{0.5}\lambda^{-1}d\lambda=2.5\ln5\approx4.0236;\end{aligned}$$
$$\begin{aligned}&\text{(b)当0.1}\leqslant\theta<0.5\text{时},\\&\pi\left(\theta\right)=\frac{1}{0.4}\int_{\theta}^{0.5}\lambda^{-1}d\lambda=2.5\left(\ln\left(0.5-\ln\theta\right)\approx-1.7329-2.5\ln\theta\right);\end{aligned}$$
$$(c)当\theta\geqslant0.5时,\pi(\theta)=0.$$

综上
$$\left.\pi(\theta)=\left\{\begin{array}{ll}4.0236,&0<\theta<0.1,\\-1.7329-2.5\ln\theta,&0.1\leqslant\theta<0.5,\\0,&0.5\leqslant\theta<1.\end{array}\right.\right.$$
恰好满足概率密度函数规范性的要求

#### 分层先验的思想特点
分层先验模型允许在建模时将相对复杂的情形转化为一系列筒单的情形
我们在前面的例子中看到了，虽然我们还是在把一个分层先验规范为普通的先验 但是分层贝叶斯模型允许我们在建模时，把相对复杂的情况分解为一系列简单的情形 让建模的难度降低 **有时候我们的规范先验甚至会复杂到没有显式表达，但是分层贝叶斯仍然能够帮助我们给出模型**

分层先验模型的另一个特点是便于计算
有时候后验密度会过于复杂（一些积分表示） 导致我们很难计算他和他的一些数字特征 导致贝叶斯统计推断 统计决策难以进行 
但是如果我们使用多层结构的后验来表示后验 即便外层积分没有显示表达也可以使用MCMC之类的方法进行计算
### 指数分布族的共轭先验

指数分布族是什么数理统计中已经进行了必要的介绍[数理统计](/blog/2023/03/18/mathematical-statistics-notes/) 的“指数分布族”一节
##### 单参数指数分布族的共轭先验分布
若 $X|\theta$ 的分布属于指数分布族（样本分布）： $x=(x_1,\cdots,x_n)$ 为 i.i.d. 的样本 似然函数可以表示为
$$
l(\theta|x)\propto[h(\theta)]^n\exp\left\{\sum_{i=1}^nt(x_i)\phi(\theta)\right\}
$$
 即指数分布族参数 $\theta$ 的共轭先验族 $\Pi$ 为 $\pi(\theta)$ 为：
 （使用共轭先验密度函数核与样本的密度函数核具有类似形式给出）
$$
\pi(\theta)\propto[h(\theta)]^{\gamma}\exp{\{\tau\phi(\theta)\}}
$$
其中超参数 $\gamma,\tau$ 已知。
##### 两参数指数分布族的共轭先验分布
若 $x=(x_1,\cdots,x_n)$ 为 i.i.d. 的样本，$X$ 的分布为两参数的指数分布族，则似然函数 $l(\theta,\varphi|x)$ 可写为：
$$
l(\theta,\varphi|x)\propto[h(\theta,\varphi)]^n\exp\left\{\sum[t(x_i)\phi(\theta,\varphi)]+\sum[u(x_i)\chi(\theta,\varphi)]\right\}
$$
 参数 $(\theta,\varphi$ 共轭先验族 $\Pi$ 为以下形式：
$$
\pi(\theta,\varphi)\propto[h(\theta,\varphi)]^\gamma\exp\left\{\alpha\phi(\theta,\varphi)+\beta\chi(\theta,\varphi)\right\}
$$
其中超参数 $\gamma,\alpha,\beta$ 已知

### 多参数模型
#### 多参数模型的思想
求解某一个参数的后验分布的思想我们前面都给出过介绍了

基本思想是先给出先验 然后使用公式进行后验的计算 存在共轭先验的话可以使用对应的手段简化计算

在大量的实际问题中 含有多个未知参数是普遍的情况 我们可以使用和单参数类似的方法给出后验分布

有时候我们只关注所有参数中的一部分 此时为了得到我们感兴趣的参数的边际后验密度 我们需要对整体的后验密度对讨厌参数求积分

#### 多参数模型的例子
设总体分布为正态分布，$\underline{x}=(x_1,\cdots,x_n)$ 为样本容量为$n$ 的 i.i.d 样本，未知参数 $(\theta,\sigma^2)$ 为一个 2 维随机变量，若取$(\theta,\sigma^2)$ 的先验密度为 $\pi(\theta,\sigma^2)\propto\frac1{\sigma^2}$,

$(1)\left(\theta,\sigma^{2}\right)|x$ 的联合后验密度函数 $\pi(\theta,\sigma^2|x)$ 为：
$$
\pi(\theta,\sigma^2|x)\propto(\sigma^2)^{-\frac{\gamma+1}2-1}\exp\left\{-\frac1{2\sigma^2}\left(S+n(\bar{x}-\theta)^2\right)\right\}
$$
其中 $\gamma=n-1,\:S=\sum_{i=1}^{n}\left(x_{i}-\bar{x}\right)^{2},\:\bar{x}=\frac{1}{n}\sum_{i=1}^{n}x_{i}$ 

(2) $\theta$ 的边际后验分布有：

$$
t=\frac{\theta-\bar{x}}{s/\sqrt{n}}\sim t(\gamma)
$$
其中 $s^2=\frac1{n-1}S$, $t(\gamma)$ 是自由度为 $\gamma$ 的 $t$ 分布

(3) 方差 $\sigma^2$ 的后验边际分布：
$$\begin{aligned}
\pi(\sigma^{2}|x)& =\int\pi(\theta,\sigma^{2}|x)d\theta   \\
&=\int_{-\infty}^{+\infty}\left(\sigma^2\right)^{-\frac{\gamma+1}2-1}\exp\left\{-\frac1{2\sigma^2}\left(S+n(\theta-\bar{x})^2\right)\right\}d\theta  \\
&\propto\left(\sigma^{2}\right)^{-\frac{\gamma}{2}-1}\exp\left\{-\frac{S}{2\sigma^{2}}\right\} \\
&\times\int_{-\infty}^{+\infty}\left(\frac{2\pi\sigma^2}n\right)^{-\frac12}\exp\left\{-\frac1{2\sigma^2}\cdot n(\theta-\bar{x})^2\right\}d\theta  \\
&=\left(\sigma^2\right)^{-\frac{\gamma}{2}-1}\exp\left\{-\frac{S}{2\sigma^2}\right\}
\end{aligned}$$
等式右边的式子为倒 Gamma 分布的核，故方差 $\sigma^2$ 的后验边际分布为$IGamma(\frac{\gamma}{2},\frac{S}{2})$
## 后验分布的计算

### 后验分布的计算
#### 理论介绍
**后验分布的计算是一切贝叶斯统计推断的基础 这里我们进行详细的研究**

后验分布的计算公式为
$$\pi(\theta|x)=\frac{h(x,\theta)}{m(x)}=\frac{f(x|\theta)\pi(\theta)}{\int_{\Theta}f(x|\theta)\pi(\theta)\mathrm{d}\theta}$$
也就是我们需要做一个积分来求边缘分布

这一步积分确实不一定好计算（基本上一定很难计算） 因此我们需要研究如何简化后验分布的计算
#### 常见的后验分布计算方法
我们有三种方法来计算后验分布
* 基于贝叶斯公式的一般计算方法：后验分布的定义
* 基于分布核的简化计算方法
* 基于充分统计量的计算方法
比较常见的先验选取有两种
* 无信息先验分布
* 共轭先验分布

### 后验分布的计算的例子

#### 例子1
我们还是计算前面的 设$X\sim B(n,\theta)$   $\theta$ 的先验分布为贝塔分布 $Beta(a,b)$
样本密度的核$\theta^x(1-\theta)^{n-x}$  先验密度的核是 $\theta^{a-1}(1-\theta)^{b-1}$
因此后验密度满足
$$\pi(\theta|x)\propto f(x|\theta)\pi(\theta)\propto\theta^{x+a-1}(1-\theta)^{n-x+b-1}.$$
明显的 他也是一个Beta分布的核 添加正则化因子补充为Beta分布
$$\pi(\theta|x)=\frac{\Gamma(n+a+b)}{\Gamma(x+a)\Gamma(n-x+b)}\theta^{(x+a)-1}(1-\theta)^{(n-x+b)-1}$$
#### 例子2
设$X$服从正态分布$N(\theta,\sigma^{2})$ 其中方差已知但是均值未知 如果$\theta$的先验是$N(\mu,\tau^2)$ 参数都已知 求后验分布
$$\pi(\theta|x)\propto f(x|\theta)\pi(\theta)\propto\exp\left\{-\frac{1}{2}\Big[\frac{(x-\theta)^2}{\sigma^2}+\frac{(\theta-\mu)^2}{\tau^2}\Big]\right\}.$$
令
$$\rho=\frac{1}{\tau^{2}}+\frac{1}{\sigma^{2}}=\frac{\sigma^{2}+\tau^{2}}{\sigma^{2}\tau^{2}}.$$
进行构造的平方化简
$$\pi(\theta|x)\propto\exp\Big\{-\frac\rho2\Big[\theta-\frac1\rho\Big(\frac\mu{\tau^2}+\frac x{\sigma^2}\Big)\Big]^2-\frac{(x-\mu)^2}{2(\sigma^2+\tau^2)}\Big\}\propto\exp\Big\{-\frac{\rho}{2}\Big[\theta-\frac{1}{\rho}\Big(\frac{\mu}{\tau^{2}}+\frac{x}{\sigma^{2}}\Big)\Big]^{2}\Big\}$$
能看出 这是 $N(\mu(x),\eta^{2})$ 的核 添加正则化因子 得到后验密度
$$\pi(\theta|x)=\frac{1}{\sqrt{2\pi}\eta}\exp\Big\{-\frac{1}{2\eta^2}[\theta-\mu(x)]^2\Big\}.$$
其中
$$\begin{aligned}\mu(x)&=\frac{1}{\rho}\left(\frac{\mu}{\tau^2}+\frac{x}{\sigma^2}\right)=\frac{\sigma^2}{\sigma^2+\tau^2}\mu+\frac{\tau^2}{\sigma^2+\tau^2}x,\\\eta^2&=\rho^{-1}=\frac{\sigma^2\tau^2}{\sigma^2+\tau^2}.\end{aligned}$$
样本分布为方差已知的正态分布时，均值参数为$\theta$的共轭先验分布族为正态分布族   （参数先验是正态的时候就共轭了）
#### 例子3
下面我们给出一些关于共轭先验分布的例子 不计算而直接给出一些例子
* 样本分布为 泊松分布$P(\theta)$的时候 如果先验密度服从伽马分布 那么后验分布也是伽马分布 也就是共轭先验分布族是伽马分布族
* 样本分布是为伽马分布 $\Gamma(r,\lambda)$ 其中$r$已知 则$\lambda$的共轭先验分布族是伽马分布族
* 指数分布是伽马分布的特例 所以样本分布为$Exp(\lambda)$的时候  $\lambda$的共轭先验分布族是伽马分布族
* 样本分布是均值已知的正态分布的时候 则方差$\sigma^{2}$的共轭先验分布族是逆伽马分布族

### 简单总结
求共轭先验分布族是有固定的思路的 
1. 写出样本概率函数的核
2. 选取核样本概率函数有同类核（类似形式）的先验分布作为共轭先验分布 从而得到共轭先验分布族

**样本概率密度的核，是含有$\theta$的函数，但是此时我们的分布是把$X$作为自变量给出的，如果我们把$\theta$看作自变量，那他应该是另一个分布的核**

能看出
* 共轭先验分布计算后验分布非常的方便
*  后验分布的很多参数可以有很好的解释 

比如前面正态分布例子中 后验均值是根据先验和抽样一起决定的 随着样本信息量的增大  先验信息的作用被自然的弱化了

对于那些无法使用共轭先验和基本定义简单解决的问题 我们后面一方面介绍一种利用充分统计量的方法本文“后验分布与充分性”一节 以及数值计算方法本文“贝叶斯统计计算”一节

### 后验分布与充分性
#### 数理统计中的充分性
统计量的充分性是数理统计中最重要的概念之一  其直观定义为 ：不损失信息的统计量

理论定义 给定确定的$T(x)=t$ 的条件时 $X$的条件分布和参数$\theta$ 无关

最好用的判断方法是 使用因子分解定理
#### 贝叶斯统计中的充分性
实际上 我们进行定义的手段是完全一致的
直观定义：使用样本分布和统计量$T(x)$的分布计算出来的后验分布一致

理论定义：设 $\mathbf{x}=(x_1,\cdots,x_n)$ 是来自密度函数$p(x|\theta)$的一个样本，$T=T(x)$是统计量，它的密度函数为$q(t|θ)$,又设$\mathbf{H}=\{\pi(\theta)\}$是$\theta$的某个先验分布族，则$\mathrm{T( x) }$为$\theta$的充分统计量的充要条件是对任一先验分布$\pi(\theta)\in H$ 有
$$\pi(\theta\mid\mathrm{T}(\mathbf{x}))=\pi(\theta\mid\mathbf{x})$$

这个定理有什么用？——简化后验分布的计算

如果我们确定某个统计量是充分统计量 ；可以用充分统计量来计算后验分布 而不是用样本分布来计算后验分布

判断这个统计量是否是充分统计量的方法还是一模一样的因子分解定理 
$$L(\theta)=\prod_{i=1}^nf(x_i;\theta)=h(x_1,x_2,\cdots,x_n)g(T(x_1,x_2,\cdots,x_n);\theta)$$
统计量的含义并没有发生变化

#### 贝叶斯统计中的充分性的应用
任取先验分布$\pi(\mu)$ 利用充分统计量计算正态分布$\mathbb{N}(\mu,1)$中参数$\mu$ 的后验分布
我们知道统计量$\overline{x}$ 是参数的充分统计量 并且
$$\overline{x}\sim N(\mu,\sigma^2/n)$$
利用后验分布的计算公式计算后验分布
$$\pi(\theta\mid\overline{x})=\frac{\exp\left\{-\frac{n(\overline{x}-\theta)^2}{2\sigma^2}\right\}\pi(\theta)}{\int_{-\infty}^{\infty}\exp\left\{-\frac{n(\overline{x}-\theta)^2}{2\sigma^2}\right\}\pi(\theta)d\theta}$$
能看出 其实我们就是把原本的$f(x|\theta)$ 变成了 $\bar{x}$ 的密度函数
其他任何地方都没有变化 他的作用是 规避了$\prod_{i=1}^nf(x_i;\theta)$ 带来的复杂运算

当然我们可以扩展到双参数的情况
设总体分布为正态分布 $N(\mu,\sigma^2)$, 样本$\mathbf{x}=(x_1,...,x_n)$ 为i.i.d样本，均值 $\mu$ 和方差$\sigma^2$ 未知，利用充分统计量计算后验分布
给出
$$\overline{x}=\frac1n\sum_{i=1}^nx_i\quad\quad Q=\sum_{i=1}^n\left(x_i-\overline{x}\right)^2$$
容易知道 二维统计量$(\overline{x},Q)$ 是$(\mu,\sigma^2)$ 的充分统计量 并且
$$\bar{x}\sim N(\mu,\sigma^2/n),Q/\sigma^2\sim\chi^2(n-1)$$
密度函数如下
$$\begin{aligned}
&p(\bar{x}\mid\mu,\sigma^2) =\frac{\sqrt{n}}{\sqrt{2\pi\sigma}}\exp\left\{-\frac{n(\overline{x}-\mu)^2}{2\sigma^2}\right\}  \\
&p(Q|\mu,\sigma^2) =\frac1{\Gamma(\frac{n-1}2)(2\sigma^2)^{\frac{n-1}2}}Q^{\frac{n-3}2}\exp\{-\frac Q{2\sigma^2}\} 
\end{aligned}$$
两者本身独立 得到联合分布有
$$\begin{aligned}
p(\bar{x},Q\mid\mu,\sigma^2)& =\frac{\sqrt{n}}{\sqrt{2\pi\sigma}}\frac1{\Gamma(\frac{n-1}2)(2\sigma^2)^\frac{n-1}2}Q^\frac{n-3}2  \\
&\times\exp\{-\frac1{2\sigma^2}[Q+n(\overline{x}-\mu)^2]\}
\end{aligned}$$
计算后验分布
$$\pi(\mu,\sigma^2\mid\overline{x},\underline{Q})=\frac{(\sigma^2)^{\frac n2}\exp\left\{-\frac1{2\sigma^2}\left(Q+n(\overline{x}-\mu)^2\right)\right\}\pi(\mu,\sigma^2)}{\int_{-\infty}^{\infty}\int_{0}^{\infty}(\sigma^2)^{\frac n2}\exp\left\{-\frac1{2\sigma^2}\left(Q+n(\overline{x}-\mu)^2\right)\right\}\pi(\mu,\sigma^2)d\sigma^2d\mu}$$
和我们使用样本计算的结果其实是一样的


### 多样本的后验分布
我们前面所有的例子都是针对样本抽样数为$1$进行讲解的，包括在序章中给出的公式；

但是实际上多样本的后验分布才是实际应用中最常用的情况；我们前面一直淡化这一点是为了降低思考的难度，因此我们在已经熟练的掌握各种技法之后，解释在多样本的情况下后验分布的计算方法，解释前面的方法会如何的变化
#### 多样本后验计算方法
原始的后验分布计算公式为
$$\pi(\theta|x)=\frac{h(x,\theta)}{m(x)}=\frac{f(x|\theta)\pi(\theta)}{\int_{\Theta}f(x|\theta)\pi(\theta)\mathrm{d}\theta}$$

在抽取来多个i.i.d的样本后 我们给出多个样本的联合概率密度（单个概率密度的积，但是要区分样本的变化）
$$f\left(\boldsymbol{x}|\theta\right)=\prod_{i=1}^{n} f(x_{i}|\theta)$$

用$f\left(\boldsymbol{x}|\theta\right)=\prod_{i=1}^{n} f(x_{i}|\theta)$代替原本的$f(x|\theta)$ 使用贝叶斯后验公式进行计算 就是多样本的后验分布

非常明显的 直接积分方法可以完全沿用 基于分布核的方法只需要重新研究一下核就可以了，不需要记忆结论 

至于充分统计量方法 这个方法的存在就是为了解决$\prod_{i=1}^nf(x_i;\theta)$ 带来的复杂运算 用于多样本是最合适的了

#### 核方法与多样本
若进一步假定 $X_1,\cdots,X_n$ i.i.d. $\sim N(\theta,\sigma^2),\sigma^2$ 已知，$\theta\sim N(\mu,\tau^2)$, 试求 $\theta$ 的后验密度. 

由于样本 $X=(X_1,\cdots,X_n)$ 的联合密度是
$$\begin{aligned}
f\left(\boldsymbol{x}|\theta\right)& =(2\pi\sigma^2)^{-n/2}\exp\left\{-\frac1{2\sigma^2}\sum_{i=1}^n(x_i-\theta)^2\right\}  \\
&\propto\exp\left\{-\frac1{2\sigma^2}\Big[\sum_{i=1}^n(x_i-\bar{x})^2+n(\bar{x}-\theta)^2\Big]\right\} \\
&\propto\exp\left\{-\frac{n(\bar{x}-\theta)^2}{2\sigma^2}\right\},
\end{aligned}$$

忽略一些细节上的变动 很明显这还是一个正态分布的核 我们可以使用核的方法给出正态先验情况下的后验密度函数为
$$\pi(\theta|\boldsymbol{x})=\frac1{\sqrt{2\pi}\eta_n}\exp\left\{-\frac1{2\eta_n^2}[\theta-\mu_n(\boldsymbol{x})]^2\right\},$$
其中
$$\mu_n(\boldsymbol{x})=\frac{\sigma^2/n}{\sigma^2/n+\tau^2}\mu+\frac{\tau^2}{\sigma^2/n+\tau^2}\bar{x},$$
$$\eta_n^2=\frac{\tau^2\cdot\sigma^2/n}{\sigma^2/n+\tau^2}=\frac{\sigma^2\tau^2}{n\tau^2+\sigma^2}.$$
其实也可以把前面的计算结果进行替换$\sigma^2=\frac{\sigma^2}n,x=\overline{x}$ 也可以得到这个结果

#### 充分统计量与多样本
事实上 充分统计量计算后验的方法可以和核计算的方法叠加进行 再简化一点运算 

对正态总体 $N(\theta,1)$ 做三次观察，获得样本的具体观测值为 2, 4, 3. 若 θ 的先验分布为正态分布为 $N(3,1)$,求 $\theta$ 的后验密度

如果我们用普通的核的方法做 那么会由于有三个样本 最后的核会是一个非常长 不好计算具体值的核 因此我们可以考虑充分统计量方法

我们知道 $\bar{x}$ 是正态分布的一个充分统计量 容易看出
$$\bar{x}\sim N(\theta,1/3)$$
他的观测值为3

所以变形的样本核为
$$e^{-\frac{(3-\theta)^{2}}{\frac{2}{3}}}$$
先验的核是
$$e^{-\frac{(3-\theta)^{2}}{2}}$$
所以后验的核是 
$$e^{-\frac{(3-\theta)^{2}}{\frac{6}{7}}}$$
后验还是一个正态分布

## 贝叶斯公式，全概率公式，条件概率公式之间的联系
首先，我们需要研究一下这些公式是怎么进行推导出来的。

最为基础的肯定是全概率公式，他就是分类讨论思想的一个应用，描述为
$$P(A)=\sum_nP(A\mid B_n)P(B_n).$$
完全不需要考虑他的证明，非常的自然

现在我们再考虑条件概率公式
$$P(A|B)=\frac{P(AB)}{P(B)}=\frac{P(B|A)P(A)}{P(B)}$$
其中前一步是最基础的条件概率的思想，后一步则是自然的概率等式的运用，其完整形式如下，
$$P(A B)=P(A|B)P(B) , P(AB)=P(B|A)P(A) ,P(B|A)P(A)=P(A|B)P(B)$$

那么我们就可以自然的推导得到贝叶斯公式了
$$P(A\mid B)=\frac{P(A)P(B\mid A)}{P(B)}$$
得到完整的贝叶斯公式。（一般需要把分母用全概率公式化简）

仅仅从公式的形式上来看，**贝叶斯公式不是从条件概率中推导得到的，而是条件概率公式本身** 这自然也没有单独命名一个公式的需求，作为一个基础推论就可以了，真正让贝叶斯公式成为单独的一个重要的公式还是其数学思想决定的

全概率公式的基本思想是分类讨论的分治思想，其中自然蕴含了事件发生的先后顺序，$B$ 先发生 $A$ 后发生 所以才有了 $P(A|B)$

条件概率公式的思想则是最基本的由因及果，我们从先发生$B$中推导出结果$A$ 的概率。

贝叶斯公式则把原本的因果思想转换，我们观测的$B$ 是事件的结果，$A$ 是未知的原因，他可以使用正常的因果关系$P(A)P(B\mid A)$  来进行反推。

$A$作为原因，在初始研究的时候就有先验$P(A)$ 现在有了观测 $P(B|A)$ 以后，更新出了原因的后验，这就是贝叶斯的思想，从先验到后验，我们不断修正初始的概率，得到更加真实的，符合目前实际情况的新的原因的概率，也就是$P(A|B)$
