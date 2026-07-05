---
title: "一元金融时间序列分析学习笔记"
title_en: "Univariate Financial Time Series Analysis Notes"
date: 2024-05-06 23:07:47 +0800
categories: ["Data Science & Statistics", "Time Series & Spatial Data"]
tags: ["Learning Notes", "Statistics", "Time Series", "Financial Time Series"]
author: Hyacehila
excerpt: "一篇一元金融时间序列分析学习笔记，整理资产收益率、ARCH/GARCH 效应、波动率建模和相关模型检验方法。"
excerpt_en: "A study note on univariate financial time series analysis, covering asset returns, ARCH/GARCH effects, volatility modeling, and related model diagnostics."
mathjax: true
hidden: true
permalink: '/blog/2024/05/06/univariate-financial-time-series-analysis-notes/'
---
##  金融数据及其特征
金融时间序列分析将是原本的[线性时间序列分析](/blog/2024/01/30/linear-time-series-analysis-notes/)的扩展 我们将补充金融时间序列的背景知识 扩展在原本的时间序列分析中学习的方法；并且在[金融时间序列分析(多元)](/blog/2025/09/27/multivariate-financial-time-series-analysis-notes/)引入多元时间序列分析的问题

### 资产收益率
我们计$P_t$为某时刻资产的净值 时间单位不限制
#### 简单收益率
单期简单毛收益率：
$$1+R_t=\frac{P_t}{P_{t-1}}$$
单期简单净收益率，简单收益率：
$$R_t=\frac{P_t}{P_{t-1}}-1=\frac{P_t-P_{t-1}}{P_{t-1}}$$
$k$期简单毛收益率：
$$1+R_t[k]=\frac{P_t}{P_{t-k}}=\prod_{j=0}^{k-1}(1+R_{t-j}).$$
$k$期净收益率：
$$R_t[k]=\frac{P_t}{P_{t-k}}-1=\frac{P_t-P_{t-k}}{P_{t-k}}$$
#### 连续复利收益率
设某资产的的初始值为$C$,名义上的年利率为$r$,但是在一年内分成$m$次付息，理论上每次付息$C\frac rm$,最终的资产净值应为$C+C\frac{r}{m}\times m=C(1+r);$ 

但是，因为提前付息，所以提前支付的利息也进入账户增值，从第二次付息开始，支付的利息就超过了$C\frac rm$,使得一年后的净值要高于$C(1+r)$。一年后的净值为
$$C\Big(1+\frac rm\Big)^m$$
当$m\to\infty$时，由极限$\lim_x\to+\infty(1+\frac1x)^x=e$,可知
$$\lim_{m\to\infty}C\Big(1+\frac{r}{m}\Big)^m=\lim_{m\to\infty}C\Big[\Big(1+\frac{r}{m}\Big)^{\frac{m}{r}}\Big]^r=Ce^r.$$
这时$r$称为连续复利，它也对应某个时间单位(一般是年)

$R=e^r-1$是连续复利$r$对应的实际利率，$r$与$R$的关系为
$$R=e^r-1,\quad r=\ln(1+R)$$
#### 资产组合收益率
设有$N$项资产，在$t-1$时刻组合净值为
$$A_{p,t-1}=\sum_{j=1}^NA_{i,t-1}=A_{p,t-1}\sum_{j=1}^Nw_i$$
其中$w_i=A_{i,t-1}/A_{p,t-1}$是第$i$项资产的权重

于是
$$\begin{aligned}A_{p,t}&=\sum_{j=1}^NA_{i,t-1}(1+R_{i,t})=A_{p,t-1}\sum_{j=1}^nw_i(1+R_{i,t})\\&=A_{p,t-1}\left(1+\sum_{j=1}^nw_iR_{i,t}\right)\end{aligned}$$
所以资产组合的简单收益率为
$$R_{p,t}=\sum_{j=1}^nw_iR_{i,t}$$
注意其中$w_i$是第$t-1$时刻的权重。如果继续计算$R_{p,t+1}$, 权重应该使用$t$时刻的权重。 
当然，如果资产比例变化不大，使用不变的$\{w_i\}$近似也是可以的。

对于对数收益率没有如此简单的公式。 当收益很小时近似有
$$r_{p,t}\approx\sum_{j=1}^nw_ir_{i,t}$$

#### 红利支付与收益率
对价格$P_{t-1}$的某资产，如果在$t-1$到$t$之间每单位还支付$D_t$红利，则到$t$时刻时，收益为
$P_t-P_{t-1}+D_t$,所以这时收益率应计算为
$$R_t=\frac{P_t-P_{t-1}+D_t}{P_{t-1}},\quad r_t=\ln(P_t+D_t)-\ln P_{t-1}$$

#### 超额收益率
$$Z_t=R_t-R_{0t},\quad z_t=r_t-r_{0t}$$
其中$R_{0t}$和$r_{0t}$是某项参考资产的收益率，如美国短期国债收益率。

超额收益率为负数 实际收益率为正数 依旧一般被认为是亏损

### 债劵收益率
#### 债劵类型
投资者以市场价格买入债券，在到期日收回票面价格的现金。买入价格低于票面价格。从而赚取收益

有些债券还在持有期间定期派发利息，利息按照票面利率(coupon payment)和面额计算，比如，面值为100元，票面利率为6%, 如果每半年派发一次利息，则每次派息$100\times0.06/2=3$元。有些债券不在中间派息，这样的债券称为零息债券。

#### 当期收益率
当期收益率仅计算每年的表面收益，不考虑资金的时间成本。

$$\text{当期收益率}=\frac{\text{每年派息额}}{\text{买入价格}}\times100\%$$

#### 到期收益率
对于零息债券，持有期间没有任何利息收入。 如果购入价格为$P$, 面值为$F$,持有$k$年到期，则收益率为
$$\left(\dfrac{F}{P}\right)^{1/k}-1$$
这称为到期收益率(Yield To Maturity, YTM)。

如果持有期间有利息 那么计算将会比较复杂 分别计算每一个阶段的派息才能给出准确的结果 不过直接考虑简单收益率也未尝不可

### 对数收益率
对数收益率（Logarithmic Return），是金融学中常用的一种计算资产收益率的方法。它通过取自然对数的方式来衡量资产价格变化的百分比。对数收益率通常用于股票、债券、外汇等金融资产的收益计算。
定义为
$$r_t=\ln\left(\frac{P_t}{P_{t-1}}\right)$$
我们之所以使用对数收益率，是因为
* **时间可加性**：对数收益率在时间上是可加的，即多个连续时间段的对数收益率之和等于这些时间段的总对数收益率。这使得对数收益率非常适合用于长期投资收益的计算
* 对数收益率反映了资产价格的连续复合增长，这与金融资产的复利效应相符合
* 对数收益率对于价格的极端变化不敏感，这有助于减少异常值对收益率计算的影响。
* 我们一般假设**对数收益率正态分布** 但是它有着厚尾特性 这样普通的收益率服从对数正态分布


### 资产波动率
金融数据中最关心的除了资产价格、收益率， 就是资产波动率。 资产波动率度量某项资产的风险，波动率是期权定价和资产分配的关键因素。 波动率对计算风险管理中的VaR（风险值）有重要作用。 一些波动率指数已经成为金融工具本身

#### 波动率的特征
波动率(volatility)指的是资产价格的波动强弱程度， 类似于概率论中随机变量标准差的概念。 **波动率不能直接观测**， 可以从资产收益率中看出波动率的一些特征 如
- 存在波动率聚集(volatility clustering)
- 波动率随时间连续变化，一般不出现波动率的跳跃式变动
- 波动率一般在固定的范围内变化，意味着动态的波动率是平稳的
- 在资产价格大幅上扬和大幅下跌两种情形下， 波动率的反映不同， 大幅下跌时波动率一般也更大， 这种现象称为杠杆效应（leverage effect）
这些性质对波动率模型的提出、改进有重要意义， 许多新的波动率模型都是诊断原有模型不能反映上面的某型特征而提出的

不同的波动率计算方法使用不同的数据源 股票的三种数据源有
- 每个交易日的日收益率；
- 伴随IBM股票的期权数据
- 盘中交易和报价的分笔数据；
分别可以计算如下三种不同的波动率：
- **作为日收益率的条件标准差（或条件方差）** 本章的模型是针对这种波动率定义。
- 隐含波动率：根据期权的理论公式如BS公式， 从股票价格和期权价格数据反解出模型中的波动率， 这样的得到的波动率称为隐含波动率。 隐含波动率倾向于比用日收益率建模得到的波动率数值要大。 CBOE的VIX指数就是隐含波动率。
- 实际波动率：利用一天内所有的收益率数据， 如每5分钟的收益率，估计一天收益率的方差（波动率平方）。 已实现波动率（realized volatility, RV）是这样的波动率

类似于利率， 度量波动率的时间区间一般也取为一年， 波动率一般是年化波动率。 如果有了日波动率， 可以将其乘以转换$\sqrt{252}$成年化的波动率

#### 波动率模型的结构
我们从一元波动率模型开始我们的介绍；**一元波动率模型就是试图刻画收益率本身不相关或低阶自相关， 但是不独立的性质** *比如我们发现一个序列通过了白噪声检验 ACF也体现了基本没有的相关性 但是如果作绝对值（或平方等运算）以后就无法通过白噪声检验并且ACF体现了自相关性*

用$F_{t-1}$表示截止到$t-1$时刻的收益率信息，尤其是包括这些收益率的线性组合，考察$r_t$在$F_{t-1}$条件下的条件均值和条件方差：
$$\mu_t=E(r_t|F_{t-1}),\quad\sigma_t^2=\mathrm{Var}(r_t|F_{t-1})=E[(r_t-\mu_t)^2|F_{t-1}]$$
通过分析实例的经验可知，、$\{r_t\}$通常比较简单，如平稳ARMA$(p,q)$序列。

对一般的对数收益率$\{r_t\}$,设其服从ARMA$(p,q)$模型：
$$r_t=\phi_0+\sum_{j=1}^p\phi_jr_{t-j}+a_t+\sum_{j=1}^q\theta_ja_{t-j}$$
其中$\{a_t\}$为不相关的白噪声列 于是
$$\mu_t=E(r_t|F_{t-1})=\phi_0+\sum_{j=1}^p\phi_jr_{t-j}+\sum_{j=1}^q\theta_ja_{t-j}=r_t-a_t$$
这里我们对白噪声列假定$a_t=r_t-E(r_t|F_{t-1})$,称这样的白噪声列$\{a_t\}$为平稳列$\{r_t\}$的新息列(innovation)或扰动列

$r_t$可分解为（移项）
$$r_t=\mu_t+a_t.$$
如果可以获得其他的解释变量(外生变量),可以建立模型$r_t=\mu_t+a_t$,其中
$$\mu_t=\phi_0+\sum_{i=1}^k\beta_ix_{i,t-1}+\sum_{j=1}^p\phi_jy_{t-j}+\sum_{j=1}^q\theta_ja_{t-j}$$
其中$x_{i,t-1}$是第$i$个解释变量在$t-1$时刻的值，$y_{t-j}$是剔除解释变量影响后的$r_{t-j}$的值。（这就是回归引入了时间序列误差[线性时间序列分析](/blog/2024/01/30/linear-time-series-analysis-notes/) 的“含有时间序列误差的回归模型”一节）

$\mu_t$服从的ARMA$(p,q)$的阶与数据的采样频率有关，股票指数的日频数据往往有较小的前后相关性，月度数据则可能没有任何显著的前后相关。

**现在我们建立了一个结构，整体思想试：原本的基本时间序列可以分解为均值列和新息列的和，均值列可以用含有时间序列误差的回归模型估计，新息列不相关

**我们曾经的时间序列建模结构[线性时间序列分析](/blog/2024/01/30/linear-time-series-analysis-notes/)都是在对均值列建模，新息列视为随机误差，现在这里要有变化了** 

综合前面的结构有
$$\sigma_t^2=\mathrm{Var}(r_t|F_{t-1})=\mathrm{Var}(a_t|F_{t-1})=E(a_t^2|F_{t-1}).$$
**这里的$\sigma_t$就是波动率，是收益率的条件标准差**。如果假设模型中的白噪声$\{a_t\}$是独立序列，则$\sigma_t^2\equiv\sigma^2$ ,波动率就没有建模的可能。这里假定$\{a_t\}$是零均值不相关平稳列，满足$E(a_t|F_{t-1})=0$,但不是独立序列。

本章的问题就是对$\sigma_t^2$建模，这种模型叫做条件异方差模型。条件异方差模型分为两类：
* 用确定函数来刻画$\sigma_t^2$的变化，ARCH和GARCH模型属于这一类；
* 用随机方程描述$\sigma_t^2$的变化，随机波动率(SV)模型属于这一类。

$\mu_t$ 的模型称为$r_t$的均值方程，$\sigma_t^2$的模型称为$r_t$的波动率方程。条件异方差模型就是在原来对$r_t$的均值$\mu_t$建模的基础上，再增加一个描述资产收益率的条件方差随时间变化的模型。这可以更精确刻画给定$r_1,r_2,\ldots,r_t$后$r_t+1$所服从的条件分布。

#### 建立波动率模型的步骤
对资产收益率序列建立波动率模型需要如下四个步骤：

1. 通过检验序列的自相关性建立均值项的方程， 必要时还可以引入适当的解释变量；
2. 对均值方程的残差作白噪声检验， **通过后，对残差检验ARCH效应**；
3. **如果ARCH效应检验结果显著， 则指定一个波动率模型**， 对均值方程和波动率方程进行联合估计；
4. 对得到的模型进行验证， 需要时做改进。


## ARCH模型
### ARCH效应检验
我们自然的知道 想要知道是否需要ARCH模型 首先需要对ARCH效应作检验

为了检验ARCH效应，先建立均值模型，拟合$\mu_t$,计算残差$a_t=r_t-\mu_t$。用残差序列的平方$\{a_t^2\}$作ARCH效应检验。

有两种检验方法。一种是对$\{a_t^2\}$作Ljung-Box白噪声检验，检验不显著时没有ARCH效应，检验显著时有ARCH效应。

另一种检验方法是R.F.Engle提出的。 考虑如下的最小二乘问题：
$$a_t^2=\alpha_0+\alpha_1a_{t-1}^2+\cdots+\alpha_ma_{t-m}^2+e_t,\:t=m+1,\ldots,T$$
其中$T$为样本量，$m$是适当的AR阶数，$e_t$为回归残差。零假设为
$$H_0:\:\alpha_1=\cdots=\alpha_m=0$$
拒绝$H_{0}$时有ARCH效应。 这称为Engle的拉格朗日乘子法检验。
这就是OLS的方程显著性检验 [线性回归基础](/blog/2023/09/04/linear-regression-basics-notes/) 的“回归方程的显著性检验”一节

### ARCH公式
ARCH模型（自回归条件异方差模型）是对波动率的建模的最基础模型 我们的基本思想是
* 资产收益率的扰动序列$a_t=r_t-E(r_t|F_{t-1})$是前后不相关的，但是前后不独立。
* $a_t$的不独立性，描述为Var$( r_t| F_{t- 1}) =$Var$( a_t| F_{t- 1})$ 可以用$a_t^2$的滞后值的线性组合表示。
其中$F_t=\sigma(\{r_t,r_{t-1},\ldots\})$

具体地，ARCH($m)$模型为
$$\begin{aligned}&a_{t}=\sigma_{t}\varepsilon_{t},\\&\sigma_{t}^{2}=\alpha_{0}+\alpha_{1}a_{t-1}^{2}+\cdots+\alpha_{m}a_{t-m}^{2}.\end{aligned}$$
其中$\{\varepsilon_t\}$是零均值单位方差的独立同分布白噪声，$\alpha _0> 0$,$\alpha _j\geq 0, j= 1, 2, \ldots , m$,另外$\{\alpha_j\}$还需要满足一些条件使得Var$( a_t)$有限，类似于AR$(p)$序列的平稳性的特征根条件。
本质上这里就是一个$AR(p)$ 序列来估计方差
**第一行体现了随机性，第二行体现了波动率隐含的确定性** ^03b43c

在波动率方程的右侧，仅出现了截止到$t-1$时刻的 $a_{t-1},\ldots,a_{t-m}$的确定性函数而没有新增的随机扰动，所以称ARCH模型为确定性的波动率模型，这意味着$\sigma_{t}^2$关于$F_{t-1}$可测，即在$t-1$时刻可以确定条件方差$\sigma_t^2$的值。

$\varepsilon_{t}$的分布常取为标准正态分布，标准化的t分布，广义误差分布(Generalized Error Distribution), 有些情况下还取为有偏的分布。

如果$\varepsilon_t\sim N(0,1)$, 记$\mu_t=E(r_t|F_{t-1})$, 考虑$p=1$情形，则
$$r_t|F_{t-1}\sim\mathrm{N}(\mu_t,\sigma_t^2)=\mathrm{N}(\mu_t,\alpha_0+\alpha_1a_{t-1}^2).$$
所以在ARCH模型中$\varepsilon_{t}$的分布称为“条件分布”, 即在$F_{t-1}$条件下$r_t$的条件分布的类型。

因为$a_t=r_t-E(r_t|F_{t-1})$所以$Ea_t= 0$, $E( a_t| F_{t- 1}) = 0$。由$\{\varepsilon_t\}$独立可知 $\varepsilon_t$与$F_{t-1}$独立，从而与$\sigma_t^2$独立，于是
$$\begin{aligned}\mathrm{Var}(r_{t}|F_{t-1})=&E[(r_{t}-E(r_{t}|F_{t-1}))^{2}|F_{t-1}]=E(a_{t}^{2}|F_{t-1})\\=&E(\varepsilon_{t}^{2}\sigma_{t}^{2}|F_{t-1})=\sigma_{t}^{2}E(\varepsilon_{t}^{2}|F_{t-1})=\sigma_{t}^{2}\\=&\alpha_{0}+\alpha_{1}a_{t-1}^{2}+\cdots+\alpha_{m}a_{t-m}^{2}.\end{aligned}$$
即前一节给出了$r_t$的条件方差方程。

因为系数$\alpha_j$都是非负数，所以历史值$a_{t-j}^2$较大意味着$a_t$的条件方差较大，于是在ARCH模型框架下，大的扰动后面倾向于会出现较大的扰动。

这里“倾向于”不是指一定会出现大的扰动，因为$a_{t-j}^2$较大使得条件方差$\sigma_t^2$较大，而方差大只能说出现较大的$a_t$的概率变大，而不是一定会出现大的扰动$a_t$。这种现象能够解释资产收益率的波动率聚集现象。

**现在我们有多种序列和研究的对象，**
* **我们研究原始的收益率序列$r_t$ 以及他的方差$\sigma^2_t$** 
* **我们研究$r_t$减除了趋势项以后的波动项序列$a_t$  计算他需要使用随机波动$\epsilon_t$和收益率序列的方差 $\sigma^2_t$** 
* **想要估计收益率序列的方差$\sigma^2_t$则需要使用$a_t^2$滞后的线性组合**

注意：有些作者用$h_t=\sigma_t^2$作为条件方差的记号，这时扰动$a_t=\varepsilon_t\sqrt{h_t}$。

### ARCH模型的性质
我们以最简单的ARCH模型为例
$$a_t=\sigma_t\varepsilon_t,\:\sigma_t^2=\alpha_0+\alpha_1a_{t-1}^2.$$
其中$\alpha_0>0,0<\alpha_1<1$。
$\alpha_1>0$是因为如果等于零就不能算ARCH(1),$\alpha_1<1$是为了$a_t$方差有限。
#### 新息性质
令$F_t= \sigma ( \{ r_t, r_{t- 1}, \ldots \} )$ 按$a_t$定义，
$$a_t=r_t-E(r_t|F_{t-1}),$$
$\{a_t\}$称为$\{r_t\}$的新息序列(innovation series)。

他满足
$$\begin{aligned}E(a_{t}|F_{t-1})=&E[r_t-E(r_t|F_{t-1})|F_{t-1}]\\=&E(r_t|F_{t-1})-E(r_t|F_{t-1})=0,\end{aligned}$$

实际上
$$E(a_t)=E[E(a_t|F_{t-1})]=0.$$
计算方差有
$$\begin{aligned}
&&\mathrm{Var}(a_{t})=& E(a_t^2)=E[E(a_t^2|F_{t-1})]  \\
&&&E[E(\sigma_t^2\varepsilon_t^2|F_{t-1})]=E[\sigma_t^2E(\varepsilon_t^2|F_{t-1})] \\
&&&E[\sigma_{t}^{2}E(\varepsilon_{t}^{2})]=E(\sigma_{t}^{2}) \\
&&&= \alpha_0+\alpha_1E(a_{t-1}^2).  \\
&\text{因为}\operatorname{Var}(a_t)\text{为常数,所以} \\
&&&\mathrm{Var}(a_{t})=E(a_{t}^{2})=\frac{\alpha_{0}}{1-\alpha_{1}}. \\
&\text{这要求}0<\alpha_{1}<1。
\end{aligned}$$
#### AR模型
所谓的ARCH效应，在数据上就是$\{r_t\}$本身表现为白噪声，但是$\{r_t^2\}$具有明显的相关性。以前面一节的例子为例，
若$r_t= a_t$, 可以证明$r_t^2$服从一个AR(1)模型

#### 一般化的ARCH模型的性质
其性质和我们在前面讨论的简单例子类似 最核心的就是新息性质
### ARCH模型优缺点
优点：
* 可以产生波动率聚集（广泛存在于金融时间序列的特点）
* 扰动$a_t$具有厚尾分布（广泛存在于金融时间序列的特点）

缺点
* 因为假定$a_{t-j}$通过$a_{t-j}^2$影响波动率$\sigma_t$,所以正的扰动和负的扰动对波动率影响相同，但是实际的资产收益率中正负扰动对波动率影响不同，较大的负扰动比正扰动引起的波动更大。
* ARCH模型对模型参数有较严格的约束条件，即使是ARCH(1),为了能计算峰度，也需要$\alpha_1\in(0,\frac{\sqrt{3}}3)$,高阶的$ARCH(m)$的约束条件更为复杂。这对带高斯新息的ARCH模型通过超额峰度表达厚尾性是一个限制
* 只能描述条件方差的变化，但是不能解释变化的原因。
* 由模型做的波动率预测会偏高。
* 可能需要较大的$m$。
### ARCH模型建模方法
#### 定阶
在ARCH效应检验显著后本文“ARCH效应检验”一节  可以考察$\{a_t^2\}$ 的PCAF来进行定阶

首先，模型为
$$\sigma_t^2=\alpha_0+\alpha_1a_{t-1}^2+\cdots+\alpha_ma_{t-m}^2$$
因为$E(a_t^2|F_{t-1})=\sigma_t^2$,所以认为近似有
$$a_t^2\approx\alpha_0+\alpha_1a_{t-1}^2+\cdots+\alpha_ma_{t-m}^2$$
这样可以用$\{a_t^2\}$序列的PACF的截尾性来估计ARCH阶$m$。

如果定阶发现需要过高阶的AR模型 那么可能ARCH模型不太适用于此序列

#### 模型估计
我们在公式出介绍过 实际上ARCH模型的估计应该分为两个步骤
参见本文“相关段落”一节。
模型的似然函数与假定的$\epsilon_t$的分布有关， 存在多种似然函数形式。

##### 正态分布
当假定$\varepsilon_t$为独立同标准正态分布分布随机变量列时 条件对数似然函数为
$$\ell(a_{m+1},\ldots,a_T|\boldsymbol{\alpha})=-\frac{1}{2}\sum_{t=m+1}^T\left[\ln\sigma_t^2+\frac{a_t^2}{\sigma_t^2}\right]+\text{常数项}$$
##### t分布
因为收益率分布厚尾， 有些应用中假设服从标准化t分布 先验的设计t分布的自由度以后
$$\begin{aligned}
&\ell(a_{m+1},\ldots,a_T|v,\boldsymbol{\alpha},a_1,\ldots,a_m) \\
\text{=}& (T-m)\left[\ln\Gamma\left(\frac{v+1}{2}\right)-\ln\Gamma\left(\frac{v}{2}\right)-\frac{1}{2}\ln((v-2)\pi)\right]  \\
&+\ell(a_{m+1},\ldots,a_T|\boldsymbol{\alpha},a_1,\ldots,a_m).
\end{aligned}$$
##### 有偏t分布
资产收益率分布除了厚尾之外还常常有偏。可以修改t分布使其变成标准化的有偏的单峰密度。有多种方法可以做这种修改，这里使用(Fernandez and Steel)的做法。该方法可以在任何连续单峰且关于0对称的一元分布中引入有偏性。将t($v$)分布进行有偏化后密度为
$$\left.g(\varepsilon|v,\xi)=\left\{\begin{array}{ll}\frac{2c_2}{\xi+\frac{1}{\xi}}f(\xi(c_2\varepsilon_t+c_1)\mid v),&\varepsilon_t<-\frac{c_1}{c_2},\\\frac{2c_2}{\xi+\frac{1}{\xi}}f((c_2\varepsilon_t+c_1)/\xi\mid v),&\varepsilon_t\geq-\frac{c_1}{c_2}.\end{array}\right.\right.$$
##### 广义误差分布假设
$\varepsilon_t$的另一种可取分布是广义误差分布(GED),密度为
$$f(x|v)=\frac{v}{\lambda2^{1+\frac{1}{v}}\Gamma(\frac{1}{v})}e^{-\frac{1}{2}\left|\frac{w}{\lambda}\right|^{v}},\:x\in(-\infty,\infty)\:(0<v\leq\infty).$$
其中$v=2$时即标准正态分布，$0< v< 2$时为厚尾分布。
$$\lambda=\left[2^{-\frac2v}\frac{\Gamma(\frac1v)}{\Gamma(\frac3v)}\right]^{\frac12}.$$
#### 模型验证
对一个建立好的ARCH模型，可计算标准化残差
$$\tilde{a}_t=\frac{a_t}{\sigma_t},$$
其中$a_t$是均值方程的残差，$\sigma_t$是波动率方程拟合的值。$\{\tilde{a}_t\}$应表现为零均值、单位标准差的独立同分布序列。

* 对$\{\tilde{a}_t\}$作Ljung-Box白噪声检验，可以考察均值方程的充分性。
* 对$\{\tilde{a}_t^2\}$作Ljung-Box白噪声检验，可以考察波动率方程的充分性。
* $\{\tilde{a}_t\}$的偏度、峰度、QQ图可以用来与$\varepsilon_t$的假定分布比较，以检验模型假定的正确性。
#### 预测
ARCH模型的预测类似AR模型的预测。从预测原点$h$出发，对$\sigma_t^2$序列作超前一步预测，即预测$\sigma_{h+1}^2$,有
$$\sigma_h^2(1)=\sigma_{h+1}^2=\alpha_0+\alpha_1a_h^2+\cdots+\alpha_ma_{h+1-m}^2.$$
要做超前2步预测时，因为$a_{h+1}$未知，有 $E(a_{h+1}^2|F_h)=\sigma_h^2(1)$,所以
$$\sigma_h^2(2)=\alpha_0+\alpha_1\sigma_h^2(1)+\alpha_2a_h^2+\cdots+\alpha_ma_{h+2-m}^2.$$
一般地，$\sigma_h^2(\ell)$可以滚动计算，
$$\sigma_h^2(\ell)=\alpha_0+\sum_{j=1}^m\alpha_j\sigma_h^2(\ell-j).$$
## GARCH模型
ARCH模型用来描述波动率能得到很好的效果， 但实际建模时可能需要较高的阶数， 考虑类似从AR推广到ARMA的模型变化。

### GARCH模型方程
(Bollerslev 1986)提出了ARCH模型的一种重要推广模型，称为GARCH模型。 对于一个对数收益率序列$r_t$,令$a_t=r_t-\mu_t=r_t-E(r_t|F_{t-1})$为其新息序列，称$\{a_t\}$服从GARCH$(m,s)$模型，如果$a_t$满足
$$a_t=\sigma_t\varepsilon_t,\quad\sigma_t^2=\alpha_0+\sum_{i=1}^m\alpha_ia_{t-i}^2+\sum_{j=1}^s\beta_j\sigma_{t-j}^2$$
其中$\{\varepsilon_t\}$为零均值单位方差的独立同分布白噪声列，$\alpha _0> 0, \alpha _i\geq 0, \beta _j\geq 0$,
$0<\sum_{i=1}^m\alpha_i+\sum_{j=1}^s\beta_j<1$,这最后一个条件用来保证满足模型的$a_t$的无条件方差有限且不变，而条件方差$\sigma_t^2$可以随时间$t$而变。

### GARCH模型性质
下面以最简单的$GARCH(1,1)$为例研究GARCH模型的性质。 令$F_{t-1}$表示截止到$t-1$时刻的$a_{t-i}$和$\sigma_{t-j}$所包含的信息。模型为
$$\begin{aligned}&a_{t}=\sigma_{t}\varepsilon_{t},\quad\varepsilon_{t}\mathrm{~i.i.d.~WN}(0,1),\\&\sigma_{t}^{2}=\alpha_{0}+\alpha_{1}a_{t-1}^{2}+\beta_{1}\sigma_{t-1}^{2}.\end{aligned}$$
为了计算无条件均值$Ea_t$,先计算条件期望
$$E(a_t|F_{t-1})=E(\sigma_t\varepsilon_t|F_{t-1})=\sigma_tE(\varepsilon_t|F_{t-1})=0.$$
这里用了$\sigma_t\in F_{t-1}$而$\varepsilon_t$与$F_{t-1}$独立。于是
$$Ea_t=E[E(a_t|F_{t-1})]=0.$$
即GARCH模型的新息$a_t$的无条件期望为零。

来计算$a_t$的无条件方差。 设模型(18.1)的$\{a_t\}$序列存在严平稳解，则
$$\begin{aligned}\mathrm{Var}(a_{t})=&E(a_{t}^{2})=E[E(a_{t}^{2}|F_{t-1})]=E[E(\sigma_{t}^{2}\varepsilon_{t}^{2}|F_{t-1})]\\=&E[\sigma_{t}^{2}E(\varepsilon_{t}^{2}|F_{t-1})]=E[\sigma_{t}^{2}E(\varepsilon_{t}^{2})]\\=&E[\sigma_{t}^{2}]=E[\alpha_{0}+\alpha_{1}\alpha_{t-1}^{2}+\beta_{1}\sigma_{t-1}^{2}]\\=&\alpha_{0}+\alpha_{1}E(a_{t-1}^{2})+\beta_{1}E[E(a_{t-1}^{2}|F_{t-2})]\\=&\alpha_{0}+(\alpha_{1}+\beta_{1})E(a_{t-1}^{2}).\end{aligned}$$
令$Ea_t^2=Ea_{t-1}^2$,解得
$$\mathrm{Var}(a_t)=Ea_t^2=\frac{\alpha_0}{1-\alpha_1-\beta_1}.$$
**这些研究和本文“ARCH模型的性质”一节没有区别**

总结有：
第一，像ARCH模型一样，$a_t$存在波动率聚集，一个较大的$a_{t-1}$或$\sigma_{t-1}$使得1步以后的条件方差变大，从而倾向于出现较大的对数收益率。

第二，当$\varepsilon_t$为标准正态分布时，在如下条件下$a_t$有无条件四阶矩：
$$1-2\alpha_1^2-(\alpha_1+\beta_1)^2>0.$$
这时超额峰度为
$$\frac{Ea_t^4}{(Ea_t^2)^2}-3=\frac{2\left[1-(\alpha_1+\beta_1)^2+\alpha_1^2\right]}{1-(\alpha_1+\beta_1)^2-2\alpha_1^2}>0.$$
即$a_t$分布厚尾。但是，对实际数据建模时即使使用条件t分布，对数据的厚尾性的拟合仍可能不足。

第三，GARCH模型给出了一个比较简单的波动率模型。

第四，因为$\sigma_t^2$对$a_{t-i}$的依赖是通过$a_{t-i}^2$,所以一个取正值的扰动$a_{t-i}$和一个取负值的$a_{t-i}$,只要绝对值相等，对后续波动率的影响就是相等的，不能体现杠杆效应。

### 预测
可以用类似ARMA预测的方法预测波动率。 仍以$GARCH(1,1)$为例 进行超前一步预测有
$$\sigma_{h+1}^2=\alpha_0+\alpha_1a_h^2+\beta_1\sigma_h^2\in F_h.$$
迭代生成$a_{h+1}$ 计算两步预测有
$$\begin{aligned}
\sigma_{h+2}^{2}=& \alpha_0+\alpha_1a_{h+1}^2+\beta_1\sigma_{h+1}^2  \\
\text{=}& \alpha_0+\alpha_1\sigma_{h+1}^2\varepsilon_{h+1}^2+\beta_1\sigma_{h+1}^2  \\
=& \alpha_0+(\alpha_1\varepsilon_{h+1}^2+\beta_1)\sigma_{h+1}^2 
\end{aligned}$$
借此实现循环 进行超前多步预测有
$$\sigma_h^2(\ell)\to\frac{\alpha_0}{1-\alpha_1-\beta_1}=\mathrm{Var}(a_t).$$
也就是超前多步条件方差预测趋于$a_{t}$的无条件方差

### 模型估计
ARCH模型的建模步骤也适用于GARCH模型的建模。 GARCH模型的定阶方法研究不多 使用类似于ARMA模型研究方法就基本已经足够 一般我们不会使用过高阶数的GARCH模型 也就是最高不超过$GARCH(2,2)$ 此时使用试错法也是可以接受的

模型的检验方法完全参考 本文“模型验证”一节 通过考察标准化残差来考察模型的估计效果

### 两步估计法
在传统的GARCH建模中 我们使用的估计方法是本文“ARCH模型建模方法”一节 直接对整个模型进行极大似然估计 

在很多的研究中我们提出如下的两步估计方法来估计GARCH模型
* 忽略ARCH效应， 用线性时间序列建模方法（如最大似然估计）对收益率序列建立均值方程。 残差序列用$a_{t}$表示
* 将$\{a_t^2\}$作为观测序列，可以用最大似然估计方法估计参数。用$\hat{\phi}_i$和$\hat{\theta}_i$分别表示AR和MA部分的系数估计值。则GARCH模型参数估计为$\hat{\beta}_i=-\hat{\theta}_i,\hat{\alpha}_i=\hat{\phi}_i+\hat{\theta}_i$。**这种估计方法只是一种近似，没有理论结果证明其合理性，但是一些经验显示这样的估计往往能够给出不错的近似结果，尤其是大中样本情形**

### IGARCH
我们考虑向ARIMA模型的扩展

GARCH模型可以写成(18.2)这样的关于$a_t^2$的ARMA形式，其中$\eta_t=a_t^2-\sigma_t^2$是模型的扰动，是鞅差白噪声：
$$a_t^2=\alpha_0+\sum_{i=1}^{\max(m,s)}(\alpha_i+\beta_i)a_{t-i}^2+\eta_t-\sum_{j=1}^s\beta_j\eta_{t-j},$$
如果这个模型中的AR部分有单位根 (有特征根等于1), 则对应的模型不再满足GARCH模型条件，称为IGARCH模型，或单位根GARCH模型。类似于ARIMA模型，IGARCH模型的扰动$\eta_t=a_t^2-\sigma_t^2$对$a_t^2$的影响是持久的、不衰减的。

IGARCH(1,1)模型可以写成
$$a_t=\sigma_t\varepsilon_t,\quad\sigma_t^2=\alpha_0+\alpha_1a_{t-1}^2+(1-\alpha_1)\sigma_{t-1}^2.$$

### GARCH-M模型
有些金融资产的收益率的条件均值受到其波动率的影响，称为风险溢价。GARCH-M模型就是用来描述这样的现象，M表示条件均值依赖于GARCH模型。 一种简单的GARCH-M(1,1)模型为
$$\begin{aligned}&r_{t}=\mu+c\sigma_{t}^{2}+a_{t},\quad a_{t}=\sigma_{t}\varepsilon_{t},\\&\sigma_{t}^{2}=\alpha_0+\alpha_1a_{t-1}^2+\beta_1\sigma_{t-1}^2\end{aligned}$$
模型中的收益率条件均值为 $E(r_t|F_{t-1})=\mu+c\sigma_t^2$ 需要用条件方差$\sigma_t^2=\operatorname{Var}(r_t|F_{t-1})$描述。参数$c$称为风险溢价参数，如果$c$为正值则收益率与波动率正相关。

文献中还有其他的风险溢价模型形式，如
$$\begin{aligned}&r_t=\mu+c\sigma_t+a_t\\&r_t=\mu+c\ln\sigma_t^2+a_t\end{aligned}$$
收益率$r_t$不再是不相关列，而是序列相关的，相关性来自$\sigma_{t}^{2}$的序列相关性。 风险溢价的存在是股票收益率具有序列相关性的原因之一。

## 扩展GARCH
本章讲GARCH模型的一些有针对性的改进 暂时没有细致的整理
### EGARCH
(Nelson 1991)提出的指数GARCH(EGARCH)模型允许正负资产收益率对波动率有不对称的影响。 考虑如下变换
$$g(\varepsilon_t)=\alpha\varepsilon_t+\gamma\left[|\varepsilon_t|-E|\varepsilon_t|\right],$$
其中$\alpha$和$\gamma$是实常数。$\{\varepsilon_t\}$和$\{|\varepsilon_t|-E|\varepsilon_t|\}$都分别是零均值独立同分布白噪声，分布为连续分布。易见
$Eg(\varepsilon_t)=0$。
由下式可见$g(\varepsilon_t)$的分布是非对称的：
$$g(\varepsilon_t)=\left\{\begin{matrix}(\alpha+\gamma)\varepsilon_t-\gamma E|\varepsilon_t|,&\varepsilon_t\geq0,\\(\alpha-\gamma)\varepsilon_t-\gamma E|\varepsilon_t|,&\varepsilon_t<0.\end{matrix}\right.$$
当$\varepsilon_t\sim$N(0,1)时$E|\varepsilon_t|=\sqrt{\frac2\pi}$。 对式(17.6)中的标准t分布，有
$$E|\varepsilon_t|=\frac{2\sqrt{v-2}\Gamma(\frac{v+1}2)}{(v-1)\Gamma(\frac v2)\sqrt{\pi}}.$$



EGARCH$(m,s)$模型可以用滞后算子的形式写成
$$a_t=\sigma_t\varepsilon_t,\quad\ln\sigma_t^2=\alpha_0'+\frac{1+\alpha_2B+\cdots+\alpha_mB^{m-1}}{1-\beta_1B-\cdots-\beta_sB^s}g(\varepsilon_{t-1})$$


$\alpha_0^{\prime}$为常数，其中$B$是滞后算子，多项式$1+\alpha_2z+\cdots+\alpha_mz^{m-1}$和 $1-\beta_1z-\cdots-\beta_mz^m$的根都在

单位圆外且两个多项式没有公因子。

注意这里的模型阶相当于GARCH(m,s)

记$\xi_t=\ln\sigma_t^2$,则(19.3)给出的$\xi_{t}$为一个平稳线性ARMA$(s,m-1)$序列，以零均值独立同分布白噪声$g(\varepsilon_{t-1})$为新息；但是，$\ln\sigma_t^2$通过$\varepsilon_{t-j}=a_{t-j}/\sigma_{t-j}$对$\{a_t\}$序列依赖。原始的GARCH模型的$\sigma_t^2$的方程是直接依赖于$a_t-j^2$的，$\pm a_{t-j}$对$\sigma_t^2$影响相同。

易见$E\ln\sigma_t^2=\alpha_0^{\prime}$。

更一般地，可以令$g(\cdot)$中的$\gamma$随滞后$j$变化，模型变成：
$$\begin{aligned}a_{t}=&\sigma_{t}\varepsilon_{t},\\\ln\sigma_{t}^{2}=&\alpha_{0}+\sum_{j=1}^{m}\left[\alpha_{j}\varepsilon_{t-j}+\gamma_{j}(|\varepsilon_{t-j}|-E|\varepsilon_{t-j}|)\right]+\sum_{i=1}^{s}\beta_{i}\ln\sigma_{t-i}^{2}.\end{aligned}$$
(19.4)

在(19.4)中，$\alpha_j$代表了对数收益率的正负扰动对波动率的不同影响，如果$\alpha_j=0$,则正负扰动对波动率

影响就相同了。

EGARCH与GARCH模型的区别还有：

1. 使用条件方差的对数建模，因为对数值可正可负，这就取消了GARCH模型对系数必须非负的限制。2. $g(\varepsilon_{t-j})=g(a_{t-j}/\sigma_{t-j})$的使用使得波动率对$a_{t-j}$的依赖关系与$a_{t-j}$的正负号有关，可以用来描述正
负收益率对波动率的不同的影响，即杠杆效应。

### GJR-GARCH
GJR-GARCH模型是另一个能够反映杠杆效应的波动率模型，参见(Glosten,Jagannathan,and Runkle
1993)和(Zakoian 1994)。 或称为TGARCH。 GJR-GARCH($m,s)$模型的形式为
$$\sigma_t^2=\alpha_0+\sum_{i=1}^m(\alpha_i+\gamma_iN_{t-i})a_{t-i}^2+\sum_{j=1}^s\beta_j\sigma_{t-j}^2$$
(19.9)

其中$N_{t-i}$是表示$a_{t-i}<0$的示性函数，即
$$N_{t-i}=\left\{\begin{array}{ll}1,&a_{t-i}<0\\0,&a_{t-i}\geq0\end{array}\right.$$
$\alpha_i,\gamma_i,\beta_j$是非负参数，满足与GARCH模型类似的参数条件。
正的$a_{t-i}$对$\sigma_t^2$的影响为$\alpha_ia_{t-i}^2$,负的$a_{t-i}$对$\sigma_t^2$的影响为$(\alpha_i+\gamma_i)a_{t-i}^2$,当$\gamma_i>0$时负的$a_t-i$影响较大。
模型用0作为$a_{t-i}$的门限，还可以用其它实数值作为门限。关于门限模型参见(Tsay 2010)第4章。

### ARARCH模型
Ding,Granger.and Engle提出了非对称幂(asymmetric power) ARCH模型(APARCH模型), 模型形式为
$$\begin{aligned}&r_{t}=\mu_{t}+a_{t},\quad a_{t}=\sigma_{t}\varepsilon_{t},\quad\varepsilon_{t}\sim D(0,1)\\&\sigma_{t}^{\delta}=\omega+\sum_{i=1}^m\alpha_i(|a_{t-i}|-\gamma_ia_{t-i})^\delta+\sum_{j=1}^s\beta_j\sigma_{t-j}^\delta\end{aligned}$$
其中$\mu_t$是条件均值，$D(0,1)$表示某个零均值单位方差分布，$\delta$为正实数，系数$\omega,\alpha_i,\gamma_i,\beta_j$满足某些正则性条件使得波动率为正。

最常用的是最简单的APARCH(1,1)模型。这个模型中包含了许多其它模型。
当$\delta=2$且$\gamma_j=0$时即普通的GARCH模型。
当$\delta=2$时即TGARCH模型(形式略有不同)。
当$\delta=1$时波动率方程直接使用波动率$\sigma_t$和新息$a_t$而非其平方。APARCH中的幂变换旨在提高拟合程度，但幂次$\delta$没有很好解释。
### 普通随机波动率SV模型
前面的波动率方程中$\sigma_t^2=\operatorname{Var}(a_t|F_{t-1})$都是被$\sigma_{t-1},\ldots$和 $a_{t-1},\ldots$完全决定。另一种方法是假定$\sigma_t^2$的模型本身有新息，这样的模型称为随机波动率(Stochastic Volatility, SV)模型。模型写成
$$a_t=\sigma_t\varepsilon_t,\quad(1-\alpha_1B-\cdots-\alpha_mB^m)\ln\sigma_t^2=\alpha_0+v_t.$$
其中$\sigma_t^2$取对数是为了取消系数必须为非负的限制。$\{\varepsilon_t\}$独立同标准正态分布，$\{v_t\}$独立同N$(0,\sigma_v^2)$分布， $\{\varepsilon_t\}$和$\{v_t\}$相互独立。$\alpha_i$为常数，特征多项式$1-\alpha_1z-\cdots-\alpha_mz^m$根都在单位圆外。记$\xi_t=\ln\sigma_t^2$ ,则$\{\xi_t\}$是一个严平稳AR$(m)$序列。

加入$v_t$新息后，收益率$r_t$的一个新息$a_t$就包含了$\varepsilon_t$和$v_t$两个新息，这增加了模型的自由度，但是使得从$r_t$数据估计模型参数变得更加困难，需要使用Kalman滤波或者随机模拟方法计算拟似然估计。

SV模型经常在拟合上有所改善， 但是波动率的样本外预测时好时坏

### 长记忆随机波动率模型
对资产收益率的实证分析发现，收益率本身没有长记忆性，但是其平方序列或者绝对值序列的ACF往往衰减很慢。 前面GARCH类模型的建模中$\sigma_{t-1}^2$的系数很接近于1，也提示有长记忆。

简单的长记忆随机波动率(LMSV)模型可以写成
$$a_t=\sigma_t\varepsilon_t,\quad\sigma_t=\sigma e^{\frac{1}{2}u_t},\quad(1-B)^du_t=\eta_t.$$
其中$\sigma > 0$, $\{ \varepsilon _{t}\}$和$\{\eta_{t}\}$是两个相互独立的独立同分布高斯白噪声列，$\varepsilon_t\sim$N$(0,1),\eta_{t}\sim$N$(0,\sigma_{\eta}^{2})$,$0<d<0.5$。长记忆来源于分数差分$(1-B)^d$,这使得$u_{t}$的ACF以负幂速度衰减而非负指数速度衰减。

对LMSV有
$$\begin{aligned}\ln a_{t}^{2}=&\ln(\sigma_t^2\varepsilon_t^2)=\ln\sigma^2+u_t+\ln\varepsilon_t^2\\=&(\ln\sigma^2+E\ln\varepsilon_t^2)+u_t+(\ln\varepsilon_t^2-E\ln\varepsilon_t^2)\\=&\mu+u_t+e_t.\end{aligned}$$
其中$u_t$是一个长记忆的平稳高斯时间序列，$e_t$是一个非高斯的独立同分布白噪声列。

LMSV估计比较复杂，分数参数$d$可以用拟最大似然估计法或者回归方法估计。
