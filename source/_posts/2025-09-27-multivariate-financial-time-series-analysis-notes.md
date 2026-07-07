---
title: "多元金融时间序列分析：向量自回归、协整与状态空间模型"
title_en: "Multivariate Financial Time Series Analysis: VAR, cointegration, and state space models"
date: 2025-09-27 18:50:29 +0800
categories: ["Data Science", "Time Series & Spatial Data"]
tags: ["Statistics", "Time Series", "Financial Time Series"]
author: Hyacehila
excerpt: "整理向量自回归、协整、状态空间模型、卡尔曼滤波和多元波动建模。"
excerpt_en: "Covers VAR, cointegration, state space models, Kalman filtering, and multivariate volatility modeling."
mathjax: true
hidden: true
permalink: '/blog/2025/09/27/multivariate-financial-time-series-analysis-notes/'
---
## 向量自回归模型
经济的全球一体化和信息传播的发展使得各国的金融市场相互关联， 一个市场的价格变动可以很快地扩散到另一个市场。 持有多个资产的投资者也希望了解多个资产的收益率之间的关系。 这些问题属于多元时间序列分析的范畴。我们从这一章开始研究多元时间序列分析，而不是将他们看作一个个个体进行分析。
### 多元时间序列基本概念
#### 弱平稳列
当一个多元时间序列 $r_{t}= \{r_{1t}...r_{nt}\}$ 满足如下条件时，称其为一个多元的弱平稳列
$$\begin{cases}E\boldsymbol{r}_t=\boldsymbol{\mu}\text{ 与}t\text{无关}\\\mathrm{~Var}(\boldsymbol{r}_t)=\Gamma_0\text{ 与}t\text{无关}\\\mathrm{~Cov}(\boldsymbol{r}_t,\boldsymbol{r}_{t-l})=\Gamma_l,l=0,1,2,\ldots\text{ 与}t\text{无关}&\end{cases}$$
能看出，多元弱平稳列的概念是从一元宽平稳的概念中自然变形过来的[随机过程基础](/blog/2023/03/18/stochastic-process-basics-notes/) 的“宽平稳过程”一节

#### 互相关矩阵
一元时间序列只需要研究方差和滞后的相关系数就足够了，但是多元需要考虑更多的问题。

我们记
$$\rho_{ij}(0)=\mathrm{corr}(r_{it},r_{jt})=\frac{\mathrm{Cov}(r_{it},r_{jt})}{\sqrt{\mathrm{Var}(r_{it})\mathrm{Var}(r_{jt})}}=\frac{\Gamma_{ij}(0)}{\sqrt{\Gamma_{ii}(0)\Gamma_{jj}(0)}}$$
为滞后0（同步的）多元时间序列互相关矩阵，他是一个对角线元素全为1的对称矩阵，研究多元时间序列的各个子序列的相关性，他是从普通的协方差矩阵中修正得到的

为了研究滞后关系，我们定义$k$ 元弱平稳序列 $r_t$ 的滞后$l$ 的互协方差矩阵为
$$\Gamma_l=(\Gamma_{ij}(l))_{k\times k}=E[(\boldsymbol{r}_t-\boldsymbol{\mu})(\boldsymbol{r}_{t-l}-\boldsymbol{\mu})^T]$$
他也是从一元的自协方差函数的自然推广，仅仅依赖于滞后而无关时间

从其中修正得到滞后$l$ 的互相关矩阵为 
$$\rho_{ij}(l)=\mathrm{corr}(r_{it},r_{j,t-l})=\frac{\Gamma_{ij}(l)}{\sqrt{\Gamma_{ii}(0)\Gamma_{jj}(0)}}$$
他一般情况下不是对称矩阵，当滞后互相关矩阵分量不为0的时候，我们一般称为具有先导作用

样本互相关矩阵的计算方法很容易想到有 互协方差矩阵的估计为 
$$\hat{\Gamma}_l=\frac1T\sum_{t=l+1}^T(\boldsymbol{r}_t-\bar{\boldsymbol{r}})(\boldsymbol{r}_{t-l}-\bar{\boldsymbol{r}})^T$$
样本互相关矩阵可以从互协方差矩阵计算

#### 时间序列之间的线性相依性的分类
多元时间序列的互相关阵反应了时间序列的线性相依性问题，这里是一个简单的总结章节。

我们将多元时间序列的互相关阵记为 $p_l$ 其各个元素为 $r_{ij}(l)$ 那么我们可以给出
* 对角线元素 $r_{ii}(l)$  是一元时间序列 $r_{it}$ 的ACF
* $p_{ij}(0)$ 是两个分量 $r_{it},r_{jt}$ 的同步线性关系
* $p_{ij}(l)$  反应了 $r_{it}$ 对$r_{jt}$  的过去值的依赖 0意味着不线性相关

根据不同的 $p_{ij}(l)$ 的情况，我们可以把多元时间序列之间的联系划分为
* $p_{ij}(l)=p_{ji}(l)=0$  对于任意$l$ 成立 两个序列没有任何相关性
*  $p_{ij}(l)=p_{ji}(l)=0$  对于任意$l>0$ 成立 解释如上 称为分离现象
* 有一个不为0 称为单向引导和滞后
* 两个均不为0 称为互相引导和滞后
#### 多元混成检验
一元的Ljung-Box白噪声检验推广到了多元的情形。 对一个多元序列，检验零假设
$$H_0:\boldsymbol{\rho}_1=\cdots=\boldsymbol{\rho}_m=\boldsymbol{0}$$
对立假设是不全为零矩阵

使用检验统计量
$$Q_k(m)=T^2\sum_{l=1}^m\frac1{T-l}\mathrm{tr}(\hat{\Gamma}_l^T\hat{\Gamma}_0^{-1}\hat{\Gamma}_l\hat{\Gamma}_0^{-1})$$
就可以实现类似于Ljung-Box白噪声检验，判断各序列是否为白噪声

### VAR模型基础
#### VAR模型结构
多个资产收益率的联合模型中最常用的是向量自回归 (Vector Autoregression, VAR)模型 我们给出$k$元的$VAR(1)$ 模型结构有
$$\boldsymbol{r}_t=\boldsymbol{\phi}_0+\boldsymbol{\Phi}\boldsymbol{r}_{t-1}+\boldsymbol{a}_t$$
其中$\phi_0$ 是$k$维常数 $\Phi$ 是$k$ 阶方阵 $a_t$ 是误差列 一般假定其服从均值为0的$k$元正态分布

考虑 $k=2$ 的情况 模型结构变为
$$\left\{\begin{array}{l}r_{1t}=\phi_{10}+\phi_{11}r_{1,t-1}+\phi_{12}r_{2,t-1}+a_{1t}\\r_{2t}=\phi_{20}+\phi_{21}r_{1,t-1}+\phi_{22}r_{2,t-1}+a_{2t}\end{array}\right.$$
如果 $\phi_{12}=\phi_{21}=0$ 则称两个序列是分离的 如果分离的序列的残差项$a_t$ 也不相关 则我们称其为非耦合的
反之，如果决定分离现象的系数不为0 则称为两个序列有相互反馈的关系

统计学对于这种相互反馈的关系有自己的解释方法，当两个相互反馈的序列的$a_{1t}$ 和 $a_{2t}$ 也不相关，则称为他们有传递函数关系，我们可以通过调整$r_1$ 来调整$r_2$ 在计量经济学中，称为格兰杰因果关系

Granger 对此有更加详细的解释：考虑一个二元序列的超前$l$步预测问题，分别使用VAR模型和一元模型来预测 如果$r_{2t}$ 的二元预测比他的一元预测更准 则称$r_{1t}$ 是其格兰格原因。当然，也可以互为格兰格原因

我们这里不详细解释预测误差的推导，本质上就是最简单的MSE，回归前面的例子。

当$\phi_{12}=0$的时候 预测$r_2$ 需要用$r_1$ 的信息，所以$r_1$ 是$r_2$ 的格兰格原因，反之亦然。当序列的新息项$a_t$ 的协方差矩阵不是对角阵的时候，两个序列存在同步相关性，也就是瞬时格兰格因果关系。

前面的形式都针对$VAR(1)$ 研究，在实际应用中不能从这么简单的系数关系中发现格兰格因果性。不过对于理解模型本身，这些已经足够了

#### VAR的简化结构
在前面使用的模型结构中 $\Phi$ 体现了动态的相依性，同步相依性则使用$a_t$ 的协方差矩阵 $\Sigma$ 的非对角线元素来体现。这种形式一般被我们称为VAR模型的 简化形式（reduced form），因为该模型没有清楚地表现出分量序列之间的同步线性相依性。

我们可以使用矩阵变换来把同步相依性进行显式表达，记$a_t$ 的协方差矩阵 $\Sigma$  存在 Cholesky 分解有 $\Sigma = LGL^T$  其中$G$是$k$ 阶对角方阵，$L$是对角线为1的下三角矩阵 定义 
$$\boldsymbol{b}_t=\boldsymbol{L}^{-1}\boldsymbol{a}_t=(b_{1t},\ldots,b_{kt})^T$$
则有
$$\begin{aligned}E\boldsymbol{b}_t=&0\\\mathrm{Var}(\boldsymbol{b}_t)=&\boldsymbol{L}^{-1}\mathrm{Var}(\boldsymbol{a}_t)L^{-T}=\boldsymbol{G}\end{aligned}$$
因此我们可以对原本的VAR模型进行同时左乘$L^{-1}$ 得到
$$\begin{aligned}\boldsymbol{L}^{-1}\boldsymbol{r}_{t}=&\boldsymbol{L}^{-1}\boldsymbol{\phi}_{0}+\boldsymbol{L}^{-1}\boldsymbol{\Phi}\boldsymbol{r}_{t-1}+\boldsymbol{L}^{-1}\boldsymbol{a}_{t}\\=&\boldsymbol{\phi}_{0}^{*}+\boldsymbol{\Phi}^{*}\boldsymbol{r}_{t-1}+\boldsymbol{b}_{t}\end{aligned}$$
他的最后一个子方程为
$$r_{kt}+\sum_{i=1}^{k-1}w_{ki}r_{it}=\phi_{k0}^*+\sum_{i=1}^k\phi_{ki}^*r_{i,t-1}+b_{kt}$$
由于$b_{kt}$ 一定是和$b_{ki}$ 不相关的 因此这个方程直接体现了同步依赖性 我们称为结构方程(structural form)

**在时间序列分析中通常使用简化形式**，因为
- 简化形式更容易估计；
- 在预测时，同步形式无法使用；
#### 平稳条件和矩
在线性时间序列分析中我们分析过关于AR模型平稳性的问题 [线性时间序列分析](/blog/2024/01/30/linear-time-series-analysis-notes/) 的“平稳随机时间序列ARMA / 自回归过程（AR）”一节 并且使用特征多项式的思想研究其平稳性。

在VAR模型中存在类似的问题，但是依旧较为复杂，这里不进行介绍

#### VAR（p）模型
这里我们拓展VAR（1）到VAR（p）模型，我们称$k$元时间序列服从$VAR(p)$ 当
$$\boldsymbol{r}_t=\boldsymbol{\phi}_0+\boldsymbol{\Phi}_1\boldsymbol{r}_{t-1}+\cdots+\boldsymbol{\Phi}_p\boldsymbol{r}_{t-p}+\boldsymbol{a}_t$$
其中各种系数的规定不发生变换

在$VAR(p)$ 中，模型的系数$\Phi$ 也是体现各个分量之间的先导关系，只是较为复杂 
### VAR模型使用
#### 估计与定阶
VAR模型建模也基本遵循定阶、模型估计和模型检验这样的反复尝试过程。 一元的PACF可以推广到多元情形用以辅助定阶。

对于一个真实的数据，我们分别考虑下面阶数递进的VAR模型
$$\begin{aligned}
&\boldsymbol{r}_t= \boldsymbol{\phi}_0+\boldsymbol{\Phi}_1\boldsymbol{r}_{t-1}+\boldsymbol{a}_t  \\
&\boldsymbol{r}_t= \boldsymbol{\phi}_0+\boldsymbol{\Phi}_1\boldsymbol{r}_{t-1}+\boldsymbol{\Phi}_2\boldsymbol{r}_{t-2}+\boldsymbol{a}_t  \\
&\text{:} \\
&\boldsymbol{r}_t= \boldsymbol{\phi}_0+\boldsymbol{\Phi}_1\boldsymbol{r}_{t-1}+\cdots+\boldsymbol{\Phi}_p\boldsymbol{r}_{t-p}+\boldsymbol{a}_t 
\end{aligned}$$
模型参数可以对每个方程分别用OLS（最小二乘）方法估计，也就是多元线性回归问题

我们对其中的第$i$个方程进行估计  得到残差项估计有
$$\hat{\boldsymbol{a}}_{t}^{(i)}=\boldsymbol{r}_{t}-\hat{\boldsymbol{\Phi}}_{1}^{(i)}\boldsymbol{r}_{t-1}-\cdots-\hat{\boldsymbol{\Phi}}_{i}^{(i)}\boldsymbol{r}_{t-i}$$
他的协方差矩阵为
$$\hat{\boldsymbol{\Sigma}}_i=\frac1{T-(k+1)i-1}\sum_{t=i+1}^T\hat{\boldsymbol{a}}_t^{(i)}[\hat{\boldsymbol{a}}_t^{(i)}]^T$$
据此我们可以逐一对$l$进行假设检验 $H_0:\boldsymbol{\Phi}_l=\mathbf{0}\leftrightarrow H_a:\boldsymbol{\Phi}_l\neq\mathbf{0}$
检验统计量为 $$M(1)=-(T-k-\frac{5}{2})\ln\frac{|\hat{\boldsymbol{\Sigma}}_{1}|}{|\hat{\boldsymbol{\Sigma}}_{0}|}$$他在原假设成立时服从卡方分布

或者我们可以使用AIC等信息准则确定，他需要利用极大似然估计的残差项的协方差矩阵，形式为
$$\tilde{\boldsymbol{\Sigma}}_{i}=\frac{1}{T}\sum_{t=i+1}^{T}\hat{\boldsymbol{a}}_{t}^{(i)}[\hat{\boldsymbol{a}}_{t}^{(i)}]^{T}$$
信息量的定义形式这里不介绍，这些准则的选择结果不受量纲的影响
#### 模型检验
可以计算模型残差，对残差进行多元白噪声检验(多元混成检验)。残差的多元混成检验因为使用了估计的参数，所以统计量的自由度会减少$k^2p$, 这是系数矩阵 $\Phi _j, j= 1, 2, \ldots , p$中的参数个数。

如果系数矩阵中某些参数固定为0，应按无约束的参数个数计算要扣除的自由度。
#### 模型简化
当VAR中分量个数$k$较大时，模型有许多参数，系数矩阵中参数个数为$k^2p$个。如果没有先验知识要求参数非零，可以将不显著的参数约束为零再估计。

这和我们在实际应用中简化一元的时间序列模型原理一致，根据$t=test$ 和直观来把某些系数固定为0
#### 格兰格因果性检验
如果模型可以简化为某些代表格兰杰因果性的系数等于零，则可以据此进行格兰杰因果性的检验。在二元的VAR(1)模型中，如果约束$\phi_{12}(1)=0$后的模型与无约束模型没有显著差异，则$r_{2t}$不是$r_{1t}$的格兰杰原因。$p$阶以及$k$元的情形类似。

为了比较无约束与约束的模型，使用对数似然比检验，得到的统计量在约束参数等于零的零假设下渐近服从卡方分布。

用基于VAR的方法检验格兰杰因果性， 局限是各分量也必须平稳， 不支持协整模型。也就是协整模型不能使用这里介绍的函数
#### 预测
若VAR($p$)模型已知，满足平稳性条件，设$\{a_t\}$是独立的弱平稳时间序列。用$F_t$表示截止到$t$时刻为止的$r_s,s\leq t$所包含的信息，则$E(\boldsymbol{a}_t|F_{t-1})=0$。基于$t$时刻的信息进行超前$l$步预测，预测为
$$\boldsymbol{r}_t(l)=E(\boldsymbol{r}_{t+l}|F_t)$$
当$l=1$时
$$\boldsymbol{r}_t(1)=\boldsymbol{\phi}_0+\boldsymbol{\Phi}_1\boldsymbol{r}_t+\cdots+\boldsymbol{\Phi}_p\boldsymbol{r}_{t+1-p}$$
当$l=2$时
$$\begin{aligned}\boldsymbol{r}_t(2)=&E(\boldsymbol{r}_{t+2}|F_t)\\=&\boldsymbol{\phi}_0+\boldsymbol{\Phi}_1E(\boldsymbol{r}_{t+1}|F_t)+\boldsymbol{\Phi}_2\boldsymbol{r}_t+\cdots+\boldsymbol{\Phi}_p\boldsymbol{r}_{t+2-p}\end{aligned}$$
若记
$$\left.\boldsymbol{r}_t(l)=\left\{\begin{array}{ll}E(\boldsymbol{r}_{t+l}|F_t),&l>0\\\boldsymbol{r}_{t+l},&l\leq0\end{array}\right.\right.$$
则超前$l$步预报可以写成
$$r_t(l)=E(r_{t+l}|F_t)=\boldsymbol{\phi}_0+\sum_{j=1}^p\boldsymbol{\Phi}_j\boldsymbol{r}_t(l-j)$$
可见超前多步预测可以递推地计算。

对于满足平稳性条件的VAR$(p)$模型，可以证明
$$\lim_{l\to\infty}\boldsymbol{r}_t(l)=\boldsymbol{\mu}=E\boldsymbol{r}_t$$也就是预测具有均值回归性

预测误差容易写为
$$\boldsymbol{e}_t(l)=\boldsymbol{r}_{t+l}-\boldsymbol{r}_t(l)=\boldsymbol{r}_{t+l}-E(\boldsymbol{r}_{t+l}|F_t)$$

## 协整分析与向量误差修正模型
### 虚假回归问题
线性回归分析是统计学的最常用的模型之一， 但是， 如果回归的自变量和因变量都是时间序列， **回归就不满足回归分析的基本假定： 模型误差项独立同分布。**

当出现这种虚假回归问题的时候，回归可能不相合， 或者估计相合但是回归结果中的标准误差估计和假设检验有错误。我们在[线性时间序列分析](/blog/2024/01/30/linear-time-series-analysis-notes/) 的“含有时间序列误差的回归模型”一节中介绍了虚假回归问题中的一种较为常见的情况和处理方式

这里我们将继续更加严谨的讨论设计虚假回归的问题，并且完善相关的理论

[线性时间序列分析](/blog/2024/01/30/linear-time-series-analysis-notes/) 的“含有时间序列误差的回归模型”一节一节中对原始序列进行了足够的差分，保证了其平稳性，这样最后的误差序列一定是时间序列的形式，本章的后半部分 **协整分析** 会更加复杂一点
### 协整分析
#### 协整分析的概念
对于二元时间序列$\boldsymbol{x}_t=(x_{1t},x_{2t})^T$,如果$x_{1t}$和$x_{2t}$都是一元单位根过程，但存在非零线性组合$\beta=(\beta_1,\beta_2)$使得 $z_t=\beta_1x_{1t}+\beta_2x_{2t}$弱平稳，则称两个分量$x_{1t}$和$x_{2t}$存在协整关系(cointegration) , $(\beta_1,\beta_2)^T$称为$x_t$的协整向量。 

多个分量的多元时间序列可以类似地定义协整关系，多元时可以有多个协整向量。

#### Engle和Granger两阶段法
想要考察多元时间序列$r_t$的协整性，首先需要使用一元的单位根检验确认两个分量都是单位根过程， 并且差分之后就没有单位根，这样的单位根过程称为“单整”的

其次，将$x_{1t}$当作因变量，$x_{2t}$当作自变量，作一元线性回归，得到残差$e_t$序列，和回归系数$\beta_1$,方程为
$$x_{1t}=\beta_0+\beta_1x_{2t}+e_t$$

根据Engle和Granger的研究， 回归在协整关系成立时参数估计相合， 但是系数的估计非正态， 所以用线性最小二乘估计得到的**点估计可用， 但是结果中的t检验和F检验结果无效**。

想要验证协整关系是否成立，只需要对回归的残差项进行单位根检验，当他不存在单位根的时候，我们称两个分量是协整的。不过由于$e_t$ 是回归残差，因此我们需要使用Phillips-Ouliaris协整检验

Engle和Granger两阶段法的第二阶段指的是在多元情况下需要找出所有的协整向量， 这需要利用向量误差修正模型(VECM)，我们在后面进行介绍

#### VARMA模型
仿照一元的ARMA模型， VAR模型可以推广成VARMA模型，其形式为
$$P(B)\boldsymbol{r}_t=Q(B)\boldsymbol{a}_t$$
其中
$$\begin{aligned}P(z)=&\boldsymbol{I}-\boldsymbol{\Phi}_{1}z-\cdots-\boldsymbol{\Phi}_{p}z^{p}\\Q(z)=&\boldsymbol{I}+\boldsymbol{\Theta}_{1}z+\cdots+\boldsymbol{\Theta}_{q}z^{q}\end{aligned}$$
VARMA存在同一个模型能够表示为不同参数形式的问题， 所以尽可能使用VAR，避免使用VARMA。

### 误差修正模型与协整
#### 误差修正模型
因为在协整系统中， 单位根非平稳分量的个数多于单位根的个数 （通过线性组合可以使得单位根非平稳的分量减少）， 所以如果对每个单位根非平稳分量计算差分， 虽然使得分量都平稳了， 但是会造成过度差分。

这种过度差分是我们在一元模型中不会见到的，专属于协整模型的国度差分情形

为了修正这种过度差分，我们提出向量误差修正模型(VECM, vector error correction model)

对于一个VARMA模型，如果含有$m$个协整因子 单位根个数大于$m$ 则存在下面形式的误差修正形式(VECM)
$$\Delta\boldsymbol{x}_t=\boldsymbol{\alpha}\boldsymbol{\beta}^T\boldsymbol{x}_{t-1}+\sum_{j=1}^{p-1}\boldsymbol{\Phi}_j^*\Delta\boldsymbol{x}_{t-j}+\boldsymbol{a}_t+\sum_{j=1}^q\boldsymbol{\Theta}_j\boldsymbol{a}_{t-j}$$
其中$\boldsymbol{\alpha}$和$\beta$都是$k\times m$列满秩矩阵，MA部分没有单位根，$m$维时间序列$\boldsymbol{y}_t=\boldsymbol{\beta}^T\boldsymbol{x}_t$是平稳列(没有单位根),$\boldsymbol{\beta}$的每一列都是$\boldsymbol{x}_t$的一个协整系数。$\boldsymbol{\Phi}_j^*$和$\boldsymbol{\alpha},\boldsymbol{\beta}$都依赖于原来的AR部分的系数矩阵 $\boldsymbol{\Phi}_j$,关系为：
$$\begin{aligned}\boldsymbol{\Phi}_{j}^{*}=&-\sum_{i=j+1}^{p}\boldsymbol{\Phi}_{i},j=1,2,\ldots,p-1\\\boldsymbol{\alpha}\boldsymbol{\beta}^{T}=&\boldsymbol{\Phi}_{p}+\cdots+\boldsymbol{\Phi}_{1}-\boldsymbol{I}=-P(1)\end{aligned}$$
系数$\alpha,\beta$ 并不唯一

#### 相关使用
VECM模型的系数使用最大似然估计确定

对于VECM模型的检验需要使用Johansen协整检验，检验的本质是对$\mathbf{\Pi}=\boldsymbol{\alpha}\boldsymbol{\beta}^T$ 的$rank(\Pi)$ 的检验，也就是协整关系的数量的选取

估计的VECM模型可以用于预测。首先可以从模型得到$\Delta\boldsymbol{x}_t$序列的预测值，然后可以从$\Delta\boldsymbol{x}_t$反解得到$\boldsymbol{x}_t$ 的预测值。VECM预测与VAR预测的区别在于VECM允许有单位根和协整关系，VAR预测不允许有单位根。

**VECM模型是目前为止我们学习的唯一一个允许单位根存在的时间序列模型，这也是他独特的地方**

## 状态空间模型
### 简单介绍
状态空间模型是时间序列分析领域中一类强大、灵活、多样的模型， 配合卡尔曼滤波技术，可以涵盖ARIMA模型、许多非平稳的、带有外生变量的模型， **比前面（包括[线性时间序列分析](/blog/2024/01/30/linear-time-series-analysis-notes/)）所述的线性时间序列模型更为灵活**。 

R扩展包**statespacer**实现了许多基于线性高斯状态空间模型的模型， 并且可以自定义模型。状态空间模型是一个相对独立的知识，不过性能相当强大，比[金融时间序列分析(一元)](/blog/2024/05/06/univariate-financial-time-series-analysis-notes/) 的“向量自回归模型”一节 与 [金融时间序列分析(一元)](/blog/2024/05/06/univariate-financial-time-series-analysis-notes/) 的“协整分析与向量误差修正模型”一节的使用要多的多

作为入门， 先介绍一个局部水平模型。 这个模型很简单， 所以可以用来演示状态空间模型的表示和估计。后面我们在整体探究状态空间模型。

### 局部水平模型
设$\{y_t,t=1,2,\ldots,T\}$为时间序列，满足如下模型
$$\begin{aligned}y_{t}=&\mu_t+e_t,\:\{e_t\}\sim\mathrm{iid~N}(0,\sigma_e^2),\:t=1,2,\ldots,n,\\\mu_{t+1}=&\mu_t+\eta_t,\:\{\eta_t\}\sim\mathrm{iid~N}(0,\sigma_\eta^2),\end{aligned}$$
其中$\{e_t\}$与$\{\eta_t\}$相互独立，初始值$\mu_1$为给定值或者是服从正态分布的随机变量，且与$\{e_t,\eta_t,t>0\}$相互独立。称$\{\mu_t\}$为$\{y_t\}$的水平，模型中$\{y_t\}$可观测而$\{\mu_t\}$不可观测。

这个方程结构就是线性高斯状态空间模型的一个特例。我们可以从这个模型结构中看到和前面研究的各种时间序列结构上类似的地方。

其中的$\{\mu_t\}$称为 **状态方程** $\{y_t\}$ 称为 **观测方程** $\{e_t\}$是观测误差，是瞬态的误差或者噪声

这个模型称为局部水平模型， 也是“结构时间序列模型”的一个特例。

我们可以注意到 
$$y_t-y_{t-1}=\eta_{t-1}+e_t-e_{t-1},$$
也就是一阶的差分服是一个均值项和一阶滞后的随机误差的和，这意味着原始的 $y_t$ 服从 $ARIMA(0,1,1)$ 

局部水平模型可以处理多元时间序列，不过其只是把他们分开成一元的序列进行处理，并没有什么特殊的地方
### 滤波、平滑和预报
我们继续以局部水平模型为例，研究状态空间模型的各种分析与建模手法，滤波、平滑和预报是我们在状态空间模型中最常考虑的问题。如下
* **滤波**：从$\{y_1,\ldots,y_t\}$ 估计 $\mu_t$
* **平滑**：从$\{y_1,\ldots,y_n\}$ 估计 $\{\mu_1,...\mu_n\}$
* **预报**：从$\{y_1,\ldots,y_t\}$ 估计 $\mu_{t+h}$
因此我们可以给出
* 滤波解为$E(\mu_t|y_{1:t});$
* 平滑解为$E(\mu_t|y_{1:n});$
* 预报解为$E(\mu_{t+h}|y_{1:t})$或$E(y_{t+h}|y_{1:t})$。

对于局部水平模型，设$\mu_1\sim\mathbb{N}(a_1,P_1)$,且与扰动序列独立。由正态分布的性质，局部水平模型是高斯过程， 其条件分布仍为高斯分布，所以$\mu_t|y_{1:s}$和$y_t|y_{1:s}$仍服从高斯分布(多元正态分布),其条件期望为为最小均方误差估计，也是线性无偏估计。

$\mu_t$在$y_1:s$下的条件分布完全由条件期望和条件方差决定。记$\mu_t|s=E(\mu_t|y_{1:s})$,记$\Sigma_t|s=Var(\mu_t|y_{1:s})$。

记$y_{t|s}=E(y_t|y_{1:s})$。特别地，记$a_t=E(\mu_t|y_{1:t-1}),P_t=$Var$(\mu_t|y_{1:t-1})$。 由高斯分布性质，条件方差都是非随机的。记
$$v_t=y_t-E(y_t|y_{1:t-1}),$$
这是对$y_t$做最优一步预报时的误差，显然$Ev_t=0$,令
$$F_t=Ev_t^2=\mathrm{Var}(v_t),$$
由多元正态分布性质，$v_t$与$y_{1:t-1}$独立，所以也有
$$\begin{aligned}F_{t}=&\mathrm{Var}(v_t)=E(v_t^2)=E(v_t^2|y_{1:t-1})\\=&E[(y_t-E(y_t|y_{1:t-1}))^2|y_{1:t-1}]\\=&\mathrm{Var}(y_t|y_{1:t-1}).\end{aligned}$$
### 卡尔曼滤波
卡尔曼滤波是一种递推算法，对$t=1,2,\ldots$, 基于$\mu_t|y_{1:t-1}$的条件分布和新得到的观测值$y_t$,求$\mu_t|y_{1:t}$条件分布，这等于$\mu_t|(y_{1:t-1},v_t)$条件分布，只要求条件高斯分布的期望和方差。

由前一节，$\mu_t|y_{1:t-1}\sim\mathbb{N}(a_t,P_t),v_t\sim\mathbb{N}(0,F_t)$与$y_{1:t-1}$独立。注意$\{e_t\}$与$\{\eta_t\}$独立所以$\{e_t\}$与$\{\mu_t\}$独立，可以给出滤波的条件期望为
$$\begin{aligned}
&\mu_{t|t}=E(\mu_t|y_{1:t}) \\
&= E(\mu_t|y_{1:t-1},v_t)  \\
&= E(\mu_t|y_{1:t-1})+E(\mu_t-E\mu_t|v_t)  \\
&= a_t+\frac{\mathrm{Cov}(\mu_t-E\mu_t,v_t)}{\mathrm{Var}(v_t)}v_t  \\
&= a_t+\frac{\mathrm{Cov}(\mu_t,v_t)}{F_t}v_t. 
\end{aligned}$$
其中
$$\begin{aligned}
&\mathrm{Cov}(\mu_t,v_t) \\
&= E(\mu_tv_t)\quad(\text{注意}Ev_t=0)  \\
&= E[\mu_t(y_t-a_t)]  \\
&= E[\mu_t(\mu_t+e_t-a_t)]  \\
&= E[\mu_t(\mu_t-a_t)]+E[\mu_te_t]  \\
&= E[\mu_t(\mu_t-a_t)]+0  \\
&= E\left\{E\left[(\mu_t-a_t)^2|y_{1:t-1}\right]\right\}  \\
&= E\{P_t\}=P_t. 
\end{aligned}$$
化简有
$$E(\mu_t|y_{1:t-1},v_t)=a_t+\frac{P_t}{F_t}v_t,$$
记
$$K_{t}=\frac{P_{t}}{F_{t}}=\frac{P_{t}}{P_{t}+\sigma_{e}^{2}},$$
所以有滤波的条件期望为
$$E(\mu_t|y_{1:t-1},v_t)=a_t+K_tv_t,$$
也就是说，我们把 $y_1,...,y_t$ 对 $\mu_t$ 的最优预报（也就是滤波）公式分解为两部分；第一部分是$y_1,...,y_{t-1}$ 对 $\mu_t$ 的最优预报，第二部分是新息对$y_t$的最优预报，后者的系数是卡尔曼增益$K_t$，最优预报是线性的。

在实际使用中，卡尔曼滤波的操作是一轮一轮进行的，每一轮的结构如下，在一轮轮的循环结构中，得到了整个滤波序列
$$\left\{\begin{aligned}v_t=&y_t-a_t,\\F_t=&P_t+\sigma_e^2,\\K_t=&P_t/F_t,\\a_{t+1}=&\mu_{t+1|t}=a_t+K_tv_t,\\P_{t+1}=&\Sigma_{t+1|t}=P_t(1-K_t)+\sigma_\eta^2,\mathrm{~}t=1,2,\ldots,n.\end{aligned}\right.$$
算法的初始分布参数 $a_{1}$ 和 $P_1$ 的选取对整个Kalman滤波都有着重要的影响，我们后面会单独介绍

### 一步预报误差
在我们前面进行Kalman滤波的时候，就进行了一步预报并且研究一步预报误差，当前前面的理论研究并不足够，递推计算一步预报误差
$$\begin{aligned}
&v_{1}= y_1-a_1,  \\
&v_2= y_2-a_2=y_2-a_1-K_1(y_1-a_1),  \\
&v_3= y_3-a_3=y_3-a_1-K_2(y_2-a_1)-K_1(1-K_2)(y_1-a_1), 
\end{aligned}$$

我們可以把它写作矩阵形式，如
$$\boldsymbol{v}=\boldsymbol{K}(Y_n-a_1\mathbf{1}_n)$$
其中
$$K=\begin{pmatrix}1&0&0&\cdots&0\\k_{21}&1&0&\cdots&0\\k_{31}&k_{32}&1&\cdots&0\\\vdots&\vdots&\vdots&\ddots&\vdots\\k_{n1}&k_{n2}&k_{n3}&\cdots&1\end{pmatrix}$$
我们后面还会用到这个形式
### 状态与扰动的平滑
#### 状态的平滑
在滤波中，我们希望用已有的观测去预测 $\mu_t|y_{1:t}$  当我们在获得所有观测 $\{y_1,...y_n\}$  利用所有观测来估计 $\mu_t$  也就是获得 $\mu_t|Y_n$ 这个问题称为平滑问题

我们直接给出局部水平模型的状态平滑计算方法而不给出证明有

为了求得$\hat{\mu}_t=\mu_{t|n}$,需要先进行卡尔曼滤波求出$a_t,P_t,v_t,F_t,K_t,L_t$,然后令$r_n=0$,用反向递推计算：
$$\begin{aligned}r_{t-1}=&\frac{v_t}{F_t}+L_tr_t,\\\hat{\mu}_{t}=&\mu_{t|n}=a_t+P_tr_{t-1},\:t=n,n-1,\ldots,2,1.\end{aligned}$$
同理，可以反向递推计算状态平滑方差有
$$\begin{aligned}
N_{t-1}=& \frac1{F_t}+L_t^2N_t,  \\
V_{t}=& \Sigma_{t|n}=P_t-P_t^2N_{t-1},t=n,n-1,\ldots,2,1. 
\end{aligned}$$
#### 扰动的平滑
在得到平滑状态与平滑方差后，我们还可以估计 $e_{t},\eta_t$ 的条件分布， 这个问题称为扰动的平滑。他可以用来进行模型诊断， 查找状态的突变点（对局部水平模型相当于水平的跳跃点或变点）， 查找观测误差的异常值。

记
$$\hat{e}_t=E(e_t|y_{1:n}),\quad\hat{\eta}_t=E(\eta_t|y_{1:n}),\mathrm{~}t=1,2,\ldots,n.$$
因为 $e_t=y_t-\mu_t$ 所以有
$$e_t\left|y_{1:n}\right.\sim\mathrm{N}(y_t-\mu_{t|n},\Sigma_{t|n})=\mathrm{N}(y_t-\hat{\mu}_t,V_t).$$
$$\hat{\eta}_t=E(\mu_{t+1}|y_{1:n})-E(\mu_t|y_{1:n})=\mu_{t+1|n}-\mu_{t|n}=\hat{\mu}_{t+1}-\hat{\mu}_t,$$

我们可以直接给出计算公式有
$$\begin{gathered}
E(e_t|y_{1:n})= \sigma_e^2\left(F_t^{-1}v_t-K_tr_t\right), \\
\mathrm{Var}(e_t|y_{1:n})= \sigma_e^2-\sigma_e^4\left(\frac1{F_t}+K_t^2N_t\right), 
\end{gathered}$$
对于状态方程扰动有
$$\begin{aligned}
E(\eta_t|y_{1:n})=& \sigma_\eta^2r_t,  \\
\mathrm{Var}(\eta_t|y_{1:n})=& \sigma_\eta^2-\sigma_\eta^4N_t,\mathrm{~}t=n,n-1,\ldots,2,1. 
\end{aligned}$$


### 缺失值的处理与预测
一般的时间序列模型都很难处理出现在时间区间内部的缺失值。状态空间模型的一大优势就是可以比较容易的允许观测有缺失值。

在局部水平模型中，设$\{y_t\}_{t=\ell+1}^{\ell+h}$缺失。状态空间模型可以用多种方法解决缺失值问题，这里使用不改变时间步数和模型形式的方法。

对于 $t\in\{\ell+1,\ldots,\ell+h\}$ 根据局部水平模型的公式我们可以给出
$$\mu_t=\mu_{t-1}+\eta_{t-1}=\cdots=\mu_{\ell+1}+\sum_{j=\ell+1}^{t-1}\eta_j,$$

我们可以给出滤波结构为
$$\begin{aligned}
E(\mu_t|Y_{t-1})=& E(\mu_t|Y_\ell)=a_{\ell+1},  \\
\mathrm{Var}(\mu_t|Y_{t-1})=& \mathrm{Var}(\mu_t|Y_\ell)=P_{\ell+1}+(t-\ell-1)\sigma_\eta^2, 
\end{aligned}$$
于是有递推式
$$\begin{aligned}a_t=&\mu_{t|t-1}=\mu_{t-1|t-2}=a_{t-1},\\P_t=&\Sigma_{t|t-1}=P_{t-1}+\sigma_\eta^2,\mathrm{~}t=\ell+2,\ldots,\ell+h.\end{aligned}$$
这意味着我们之前进行的Kalman滤波依旧可以正常运行，对于缺失的$y_t$ 我们应该取相应的$v_{t}= 0$ 于此同时对应的$K_{t}= 0$ 也就是没有Kalman增益

实际上，我们进行的预测本质上就是Kalman滤波，结果和设未来的值为缺失直接进行滤波给出的结果是一样的
### 初值分布参数的选取与模型的参数估计
#### 初值分布参数的选取
Kalman滤波需要假定知道 $\mu_1\sim\operatorname{N}(a_1,P_1)$  实际上其中的$a_1,P_1$ 都是未知的

利用滤波公式得
$$\begin{aligned}
v_{1}=& y_1-a_1,\quad F_1=P_1+\sigma_e^2,  \\
a_{2}=& a_1+\frac{P_1}{F_1}v_1=a_1+\frac{P_1}{F_1}(y_1-a_1)  \\
\rightarrow & y_1\quad(P_1\to\infty),  \\
P_{2}=& P_1\left(1-\frac{P_1}{P_1+\sigma_e^2}\right)+\sigma_\eta^2  \\
=& \frac{P_1}{P_1+\sigma_e^2}\sigma_e^2+\sigma_\eta^2  \\
\rightarrow & \sigma_e^2+\sigma_\eta^2\quad(P_1\to\infty), 
\end{aligned}$$
因此$P_1\to\infty$时相当于认为$y_1$是非随机的确定值，而$\mu_1\sim\mathbb{N}(y_1,\sigma_e^2)$。这种初始化方法称为扩散(diffuse)初始化或者扩散先验。 扩散先验相当于对初始状态分布没有任何知识。
#### 模型的参数估计
滤波和平滑算法都是假定模型参数$\sigma_e^2$和$\sigma_\eta^2$已知的。 为了估计参数可以使用最大似然估计法，计算似然函数时可以利用滤波算法进行计算。

### 状态空间模型
我们在前面全部的介绍都是局部水平模型的相关知识，局部水平模型是线性高斯状态空间模型的一个简单特例。 本节给出状态空间模型， 举例说明这种模型能够表示的其它模型 并给出滤波、平滑、预报公式和参数估计方法。

注意参考 [R TSA](/blog/2024/05/04/r-time-series-analysis-learning-notes/) 的“状态空间模型”一节 本节的模型记号方式地我们后面使用状态空间模型进行建模非常重要。

**很多模型都可以被表示为状态空间模型的形式，不过研究这种表示在应用中意义不是很大**
#### 线性高斯状态空间模型
状态空间模型有许多不同的表达形式，按照(Durbin and Koopman 2012)的公式，线性高斯模型为： ^b0460f
$$\begin{gathered}
\boldsymbol{y}_t= Z_t\boldsymbol{\alpha}_t+\boldsymbol{\varepsilon}_t,\boldsymbol{\varepsilon}_t\sim\mathrm{N}(0,H_t), \\
\boldsymbol{\alpha}_{t+1}= T_t\boldsymbol{\alpha}_t+R_t\boldsymbol{\eta}_t,\boldsymbol{\eta}_t\sim\mathrm{N}(0,Q_t), 
\end{gathered}$$
其中
$$\alpha_1\sim\mathrm{N}(a_1,P_1).$$

其中$y_t$是$t$时刻的观测值，为$p\times1$向量；$\boldsymbol{\alpha}_t$是$t$时刻系统的状态，是不可观测的$m\times1$随机向量，第一个方程称为观测方程，第二个方程称为状态方程。

$\{\boldsymbol{\varepsilon}_t\}$和$\{\boldsymbol{\eta}_t\}$相互独立，都是独立同分布向量白噪声列，$\boldsymbol{\varepsilon}_t$为$p\times1$随机向量，$\boldsymbol{\eta}_t$为$r\times1$随机向量，$r\leq m$。

设各矩阵$Z_t,T_t,R_t,H_t,Q_t$已知，$Z_t$和$T_{t-1}$ 允许依赖于$\boldsymbol{y}_1,\ldots,\boldsymbol{y}_{t-1}$,初始状态$\boldsymbol{\alpha}_1$服从$N(\boldsymbol{a}_1,P_1)$,设$\boldsymbol{a}_1,P_1$已知，$\boldsymbol{\alpha}_1$与$\{\boldsymbol{\varepsilon}_t\}$和$\{\boldsymbol{\eta}_t\}$独立。

当参数未知时，设$\psi$为未知参数，矩阵$Z_t,T_t,R_t,H_t,Q_t$可以依赖于未知参数$\psi$。

模型中的$R_t$常常是单位阵，$r=m$, 有些教材的模型就没有$R_t$这一项。包含$R_t$的好处是，$R_t$常常是单位阵$I_m$的某些列组成的一个$m\times r$矩阵，称为选择矩阵，这允许某些状态分量对应的方程误差为0，同时$\boldsymbol{\eta}_t$的方差阵$Q_t$还可以是满秩的$r\times r$正定阵，如果没有$R_t$矩阵$Q_t$就可能不满秩。如果$R_t$是一般的$m\times r$矩阵，关于状态空间模型的大部分结论仍成立。

#### 推广的状态空间模型
这是对上一节的继承，可以将线性高斯的状态空间模型， 推广到状态方程仍为线性高斯形式， 而观测方程的分布为非高斯分布， 或者观测方程中观测变量与状态变量的关系非线性， 更进一步可以推广到状态方程的关系也非线性， 分布为非高斯分布。

一般化的非线性、非高斯状态空间模型形式为
$$\begin{aligned}\boldsymbol{y}_{t}\sim&f_{t}(\boldsymbol{\alpha}_{t};\boldsymbol{\beta}),\\\boldsymbol{\alpha}_{t+1}\sim&g_{t}(\boldsymbol{\alpha}_{t};\boldsymbol{\theta}),\end{aligned}$$
这样的模型一般需要使用MCMC、序贯重要抽样等随机模拟方法进行滤波、平滑和估计。

#### MARSS包的模型
**MARSS**是一个较为常用的R状态空间模型软件包，他对模型的形式有一定的约定，我们需要了解这种形式来方便我们对软件包的使用。MARSS是多元自回归状态空间模型的缩写， 实际上就是线性高斯状态空间模型。

基本的模型公式为：
$$\begin{aligned}&\boldsymbol{x}_{t}=B\boldsymbol{x}_{t-1}+\boldsymbol{u}+\boldsymbol{w}_{t},\quad&\boldsymbol{w}_{t}\sim\mathrm{N}(0,Q),\\&\boldsymbol{y}_{t}=Z\boldsymbol{x}_{t}+\boldsymbol{a}+\boldsymbol{v}_{t},\quad&\boldsymbol{v}_{t}\sim\mathrm{N}(0,R),\\&\boldsymbol{x}_{0}\sim\mathrm{N}(\boldsymbol{\pi},\Lambda).\end{aligned}$$
这里和[金融时间序列分析(一元)](/blog/2024/05/06/univariate-financial-time-series-analysis-notes/) 的“线性高斯状态空间模型”一节的结构基本相同，只是更改了记号。较为特色的是，矩阵 $B,Z,u,a$ 都允许时变。

 更复杂的模型还可以在两个方程中增加关于外生变量影响的部分。MARSS扩展包的参数化方法和估计方法与其它状态空间模型扩展包有比较大的区别。

包含外生变量的回归部分并且各个矩阵允许时变的模型可以写成
$$\begin{aligned}&\boldsymbol{x}_{t}=B_{t}\boldsymbol{x}_{t-1}+\boldsymbol{u}_{t}+C_{t}\boldsymbol{c}_{t}+\boldsymbol{w}_{t},\quad&\boldsymbol{w}_{t}\sim\mathrm{N}(0,Q_{t}),\\&\boldsymbol{y}_{t}=Z_{t}\boldsymbol{x}_{t}+\boldsymbol{a}_{t}+D_{t}\boldsymbol{d}_{t}+\boldsymbol{v}_{t},\quad&\boldsymbol{v}_{t}\sim\mathrm{N}(0,R_{t}),\\&\boldsymbol{x}_{0}\sim\mathrm{N}(\boldsymbol{\pi},\Lambda).\end{aligned}$$
其中$c_t$是系统方程中的$p$维外生变量数据，可以输入为一个$p\times T$矩阵；

$C_t$是相应的回归载荷矩阵，可以包含未知量，如果是非时变的，只要输入为$m\times p$矩阵；如果是时变的，则需要输入为$m\times p\times T$的三维数组，用最后一个下标表示时间$t$。

$\boldsymbol d_t$则是观测方程中的$q$维外生变量数据，$D_t$是相应的载荷矩阵。

这样的MARSS模型不仅可以表示回归模型，也可以表示有内生状态变量$x_t$的带有外生变量(回归自变量)的时间序列模型。

## 隐马氏模型HMM
隐马氏模型(HMM)类似于状态空间模型， 但是其状态遵从一个马氏链， 一般是离散状态的。 此模型也有广泛应用， 比如生物研究、模式识别、金融建模等。

- ([Zucchini, MacDonald, and Langrock 2016](https://www.math.pku.edu.cn/teachers/lidf/course/fts/ftsnotes/html/_ftsnotes/hmm.html#ref-Zucchini2016:HMM-TS-R)): Hidden Markov Models for Time Series - An Introduction Using R. 2nd ed., 2016, CRC Press.
### HMM基本介绍
#### 预备知识
隐马氏模型的观测值变量在简单情况下边缘分布服从独立混合分布。 设$\delta_1,\ldots,\delta_m$是加权平均系数，$p_j(x)$,$j=1,2,\ldots,m$是$m$个密度(或概率质量函数),令
$$p(x)=\sum_{j=1}^m\delta_jp_j(x),$$
则$p(x)$是一个密度 (或概率质量函数),其分布称为独立混合分布或者简称为混合分布。 设$X_j\sim p_j$,$X\sim p$,则
$$E(X)=\sum_{j=1}^m\delta_jE(X_j).$$
且
$$E(X^k)=\sum_{j=1}^m\delta_jE(X_j^k).$$

关于马氏链的介绍我们可以参考 [随机过程基础](/blog/2023/03/18/stochastic-process-basics-notes/) 的“离散时间Markov链”一节 

#### 隐马氏链定义
设$\{C_t\}$为马氏链，$\{X_t\}$为随机过程，$X_t$ 在$X_1,\ldots,X_{t-1},C_1,\ldots,C_t$下的条件分布等于$X_t$在$C_t$下的条件分布，则称$\{X_t\}$服从隐马氏模型。实际上状态空间模型也是这样的隐马氏模型，但状态空间模型中的状态方程一般不是离散状态马氏链。

若马氏链$\{C_t\}$的状态空间仅有$m$个值，则称模型为$m$状态HMM。隐马氏过程的其它一些名称，我们看到名字就能自然的联想到

设$p_i(x)$表示$X_t$在$C_t=i$条件下的分布，离散分布时为概率质量函数，连续分布时为概率密度函数。

#### 隐马氏链简单性质
对于一元的分布，我们可以直接给出为：
$$P(X_t=x)=[\boldsymbol{u}(1)]^T\Gamma^{t-1}P(x)\mathbf{1}.$$

二元的分布为
$$\begin{aligned}
&P(X_t=v,X_{t+k}=w) \\
&= \sum_{i=1}^m\sum_{j=1}^mu_i(t)p_i(v)\gamma_{ij}(k)p_j(w)  \\
&= \boldsymbol{u}(t)^TP(v)\Gamma^kP(w)\mathbf{1}. 
\end{aligned}$$

对于矩的性质，我们可以给出
$$\begin{aligned}
E(X_t)& =\sum_{i=1}^mu_i(t)E(X_t|C_t=i)  \\
&=\sum_{i=1}^m\delta_iE(X_t|C_t=i)\quad(\text{平稳时}).
\end{aligned}$$
#### 隐马氏链似然函数
设隐马氏模型的观测值序列有$T$个，记$\boldsymbol{X}^{( t) }= ( X_1, \ldots , X_t) ^T$, $\boldsymbol{x}^{( t) }= ( x_1, \ldots , x_t) ^T。( x_1, \ldots , x_T)$ 的似然函数，即$P(\boldsymbol{X}^{(T)}=\boldsymbol{x}^{(T)})$,需要将$P(X_1=x_1,\ldots,X_T=x_T,C_1=c_1,\ldots,C_T=c_T)$中的每个$C_t$项关于$c_t$求和，共$T$重求和，求和的每一项都是$2T$项的乘积，所以表面上看似然函数的计算量达到$O(Tm^T)$,$T$较大时计算不可行；但实际上一般有计算量$O(Tm^2)$的算法。

我们直接给出似然函数值用矩阵表示为
$$L_T=\boldsymbol{\delta}^TP(x_1)\Gamma P(x_2)\Gamma P(x_3)\cdots\Gamma P(x_T)\mathbf{1}.$$
其中$P(x)=\operatorname{diag}((p_1(x),\ldots,p_m(x))),p_j(x)=P(X_t=x|C_t=j)$,不依赖于$t$的值。

最大似然估计方法略

**这一小节的内容的理论简单了解就足够了**

### 观测值预测、状态估计
在给定观测值后进行最大似然估计， 然后可以对缺失观测值进行估计， 预测观测值， 估计马氏链状态，等等。 这都是基于条件分布的计算。

我们不要求马氏链是平稳的 $\delta$ 是 $t=1$ 状态$C_1$ 的分布

#### 观测值的条件分布
记$\boldsymbol{x}^{(-t)}$表示在$\boldsymbol{x}^{(t)}=(x_1,\ldots,x_T)^T$中删去$x_t$后的向量。$\boldsymbol{X}^{(-t)}$含义类似。 考虑$X_t$在$\boldsymbol{x}^{(-t)}$条件下的条件分布，这可以用来填补缺失值。

要计算
$$P(X_t=x|\boldsymbol{X}^{(-t)}=\boldsymbol{x}^{(-t)})=\frac{P(X_t=x,\boldsymbol{X}^{(-t)}=\boldsymbol{x}^{(-t)})}{P(\boldsymbol{X}^{(-t)}=\boldsymbol{x}^{(-t)})},$$

最后其结构可以写作
$$P(X_t=x|\boldsymbol{X}^{(-t)}=\boldsymbol{x}^{(-t)})=\sum_{j=1}^mw_j(t)p_j(x),\mathrm{~}t=1,2,\ldots,T.$$
其中
$$w_j(t)=\frac{d_j(t)}{\sum_{k=1}^md_k(t)}.$$

#### 观测值的预测分布
$$\text{预测分布指条件概率}P(X_{T+h}=x|\boldsymbol{X}^{(T)}=\boldsymbol{x}^{(T)}),\quad\text{可以看成是}X_{T+1},\ldots,X_{T+h}\text{缺失情况下的计算。}$$
此时
$$P(X_{T+h}=x|\boldsymbol{X}^{(T)}=\boldsymbol{x}^{(T)})=\frac{P(\boldsymbol{X}^{(T)}=\boldsymbol{x}^{(T)},X_{T+h}=x)}{P(\boldsymbol{X}^{(T)}=\boldsymbol{x}^{(T)})},$$
最后预测分布可以化简为下面观测条件分布的混合分布的形式

$$P(X_{T+h}=x|\boldsymbol{X}^{(T)}=\boldsymbol{x}^{(T)})=\sum_{j=1}^m\xi_j(h)p_j(x),$$
#### 解码
解码是根据观测值进行还原 状态$C_t$ 的条件分布的流程，从而使用条件分布的众数预测 $C_t$

解码的实现方法我们这里略去

#### 状态的预测
可以证明，状态的预测可以被等价为解码问题

### 模型的选择与诊断
增加状态个数$m$能改变拟合，但是会以$m^{2}$速度增加参数个数，有过度拟合风险。 某些特殊的模型可能会精简状态转移矩阵或者条件分布，使其仅依赖于少量的参数。

可以使用AIC、BIC准则比较不同的模型。对于模型拟合的充分性，可以计算伪残差，进行残差诊断。

#### 用AIC、BIC进行模型选择
使用信息量准则选择模型，其思想不必要继续重复

#### 使用伪残差进行模型诊断
拟合了模型以后， 需要评估拟合是否充分， 找出拟合效果特别差的异常点。 在正态的线性回归模型建模时， 可以用残差来进行模型诊断； 在HMM等更一般的情形下， 可以定义“伪残差”， 或称分位数残差， 用来进行模型诊断

设$X$服从连续分布，分布函数为$F(\cdot)$,则$U=F(X)$服从U(0,1)分布。随机变量$X_t$若观测值为$x_t$,在假定的模型下计算其分布函数值
$$u_t=P(X_t\leq x_t)=F_{X_t}(x_t),$$
则模型正确时$u_t$应该服从U(0,1)分布，取值异常的情况是$u_t$靠近0或者1。因为将不同的分布都转换到了0和1之间，使得不同分布的观测值都可以比较。

设数据为$x_1,\ldots,x_T$,模型为$X_t\sim F_t$,$x_t$是$X_t$的观测值，因为分布不同，这些$x_t$是不可比的。计算$u_t=F_t(x_t)$,称$u_1,\ldots,u_T$为均匀伪残差，这些伪残差是可比的。可以作$u_1,\ldots,u_T$的直方图和相对于均匀分布的QQ图，如果与均匀分布表现有明显差异，就说明模型设定有误。

均匀残差用于识别异常值则不太方便，0.01分位数和0.05分位数也仅相差0.04,如果在正态分布中就已经相差很大了。因为我们熟悉正态分布，所以定义正态伪残差
$$z_t=\Phi^{-1}(u_t)=\Phi^{-1}(F_t(x_t)),$$
更容易识别异常值。模型正确时正态伪残差应该表现为标准正态分布样本。正态伪残差的值反映了$x_t$偏离其分布中位数(不是均值)的偏离程度。可以作直方图、正态QQ图、正态性检验验证模型是否正确。

伪残差最重要的性质是其分布近似标准均匀分布（或者标准正态分布）， 而不能假定其相互独立， 伪残差之间是不独立的。

### 协变量以及其它相依性
如时间趋势、季节项之类的影响， 可以作为非随机的协变量引入模型中。 还可以考虑隐藏状态模型为二阶或高阶马氏链的情形。 还可以放松对条件独立性的假定。

#### 含有协变量的HMM
可以运行观测值条件分布参数或者马氏链转移概率依赖于协变量。 这样仍可以进行最大似然估计。 协变量的值看做是已知的。

#### 基于二阶马氏链的HMM


### 连续状态的HMM
状态个数$m$有时很难客观选择，当$m$很大时未知参数个数会过多。 所以有时连续状态的隐马氏模型可能更有优势。 这就与状态空间模型很接近了。

### 半隐马氏模型
状态变量用一阶马氏链表示有时是不够准确的。 改用高阶马氏链会增大参数个数。 另一种推广是状态过程取为半马氏链。

设$Y_t$是状态空间为$\{1,\ldots,m\}$的时齐马氏链，其状态转移矩阵$\Omega$对角线元素等于0。 这样在状态序列中相邻两个时间点状态必然不同。

设$d_i$是一个正整数集上的概率分布，对$i=1,2,\ldots,m$有$m$个这样的分布，称为停留时间分布。 从$Y_t$与$\{d_i\}$构造过程$\{C_t\}$如下。每个$Y_t$值代表连续的若干个状态不变的$C_s$的值，而连续不变的个数，当$Y_t=i$时服从停留时间分布$d_i$。

这样得到的$\{C_t\}$一般不是马氏链，称为半马氏链(SMC)。若$\{C_t\}$所有$d_i$都是几何分布，则仍是马氏链。

将隐马氏模型中的状态$C_t$替换成半马氏链，就称为隐半马氏模型(HSMM)。隐半马氏模型计算比HMM要复杂得多。引入协变量也很困难。

可以通过扩充HMM的状态空间用HMM近似任意的HSMM。
### 纵向数据的HMM
设有$K$个个体，每个个体持续观测了$T$个时间点，观测为$\{x_{tk},t=1,\ldots,T,i=1,\ldots,K\}$。这称为纵向数据，经济学中称为面板(pane)数据。要注意到同一个个体的多次观测之间有相关性。设每个个体的时间序列使用相同模型，但参数可以不同。

某些情况下可以假定$K$个序列都依赖于一个共同的潜在状态序列$C_t$,给定状态序列后各个序列之间条件独立，则可以考虑HMM。一个例子是考虑多只股票的收益率，设其共同受到同一个潜在市场状态的影响。这可以看成是一个观测值为多维的HMM。

有些情况下不能认为各个序列有共同的状态序列，比如，不同病人在不同时间点上的多个测量值。如果能假定各个序列、以及相应状态独立，则似然函数为各个序列的似然函数的乘积。如果假定各个观测序列模型中的一部分参数是相同的，就可以利用所有观测序列的数据共同来估计模型，可以增加估计精度，在数据长度不足时这种联合起来的做法能够对单个建模无法估计的模型进行建模。

可以用协变量值区分个体之间的变化。
