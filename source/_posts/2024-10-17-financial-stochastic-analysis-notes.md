---
title: "金融随机分析：金融衍生品、二叉树定价与无套利理论"
title_en: "Financial Stochastic Analysis: Derivatives, Binomial Pricing, and No-Arbitrage Theory"
date: 2024-10-17 15:19:25 +0800
categories: ["Data Science", "Time Series & Spatial Data"]
tags: ["Learning Notes", "Statistics", "Stochastic Processes", "Financial Mathematics"]
author: Hyacehila
excerpt: "整理金融衍生品、二叉树定价、无套利理论、风险中性测度、鞅和连续时间金融模型。"
excerpt_en: "Covers derivatives, binomial pricing, no-arbitrage theory, risk-neutral measures, martingales, and continuous-time financial models."
mathjax: true
hidden: true
permalink: '/blog/2024/10/17/financial-stochastic-analysis-notes/'
---
## 引言与导论
### 引言
金融随机分析研究金融学的数学理论，我们主要关注金融机构开发的金融衍生品，也就是股票和现货市场以外的，由初级金融产品衍生与组合的投资产品。金融随机分析希望能对这些金融衍生品进行合理的定价，保证金融机构与市场上购买金融衍生品的企业都能从中获取合理的利润。

整个金融随机分析基于CMU硕士的教材进行，总共分两卷，分别研究离散时间的定价理论与连续时间上的随机分析理论。我们选择性的学习其中的内容，只关注金融数学中的基本理论与方法。

### 一些经典的金融衍生品
金融衍生品自身的种类繁多，这里我们需要介绍一些常见的金融衍生品才容易继续推进我们的内容

#### 期权与欧式期权
最为常见的两种金融衍生品分别是 **期权（Options）与 期货（Futures）** 我们在基础的定价模型中只考虑前者。他是基于股票（Stock）设计出的一种金融衍生品。所有的期权都是一种权利，而不是某种切实存在的商品。

首先我们来介绍最基本的**欧式看涨（Call）期权** 与之对应的还有美式，看跌等不同种类的期权，我们在后面介绍。

欧式看涨期权的定义为：甲方向乙方购买了一份合约，约定甲方可以在一个未来的时刻$T$以确定的价格$K$向乙方购买一定数量的股票。

对应的可以给出欧式看跌期权（Put）的定义为：甲方向乙方购买了一份合约，约定甲方可以在一个未来的时刻$T$以确定的价格$K$向乙方卖出一定数量的股票。

我们不妨假设未来$T$时刻股票的价格为$S_T$ 那么可以自然的给出两种期权的收益如下
* 欧式看涨期权 $W = max(S_{T}-K,0)$ 
* 欧式看跌期权 $W = max(K-S_{T},0)$ 

当然我们也可以给出一些其他的期权，比如
* 二元期权，同时具有涨价行权和跌价行权的资格
* 在初始的欧式期权的基础上增加障碍价格，当触及障碍价格后期权生效或无效

事实上，很多经典的投资组合都只需要基础的欧式期权就可以组合出，因此我们主要介绍的内容只包括欧式看涨/看跌期权。

#### 期权的作用
作为最基础的金融衍生品，期权可以提供很大的作用，如下
##### 杠杆 Leverage
假设我们有初始资金100元，并且认为股票价格会上涨，且欧式看涨期权售价2元。

如果股票上涨到110元，那么投资股票能带来百分之十的收益，而购买50份欧式看涨期权则可以带来500元的收益，是原始收益的50倍，这就是期权的杠杆作用。

当然，如果股票价格下跌，那么购买的期权将全部成为废纸而没有行权的价值，我们也会亏损百分百。
##### 对冲 Hedge
我们还是认为股票价格会上涨，但是我们选择在购买股票的同时购买一份欧式看跌期权。此时我们的收益变为
$$W = S_{T} + (K-S_{T})^{+}$$
也就是无论股价如何下跌，我们都会有一个亏损下限，收益最低为 $K$  这就是对冲风险
##### 套利 Arbitrage
套利指的是我们发现了期权定价的缺陷，可以通过按照一定比例购买期权与股票后，无论股票价格怎么变化，我们都可以无风险获取利润的行为。

研究如何套利和如何无风险套利是金融机构与市场上的投资者互相进攻的行为。

#### 一些欧式期权衍生品
只需要最简单的欧式期权与股票，我们就能构造出各种组合
##### 股票+看跌期权
收益满足
$$W = S_{T} + (K-S_{T})^{+} = max(S_T,K)$$
对冲风险的组合

##### 备兑认购期权（买入股票的同时卖出看涨期权）
收益满足
$$W = S_{T} - (S_{T}- K)^{+} = min(S_T,K)$$

依靠卖出期权的收益获得收益，我们认为股票会涨价，但是不会涨到卖出的看涨期权被行权

##### 跨式期权
买$K$价格的欧式看涨和$K$价格的欧式看跌，收益满足
$$W = (S_{T}- K)^{+} + (K-S_T)^{+}$$
我们希望股票的价格剧烈波动来获利，用于我们认为会波动但是不知道波动方向。

##### 宽跨式期权
买$K_1$价格的欧式看涨和$K_2$价格的欧式看跌，收益满足
$$W = (S_{T}- K_1)^{+} + (K_2-S_T)^{+}$$
在$S_T$位于$K_1$到$K_2$之间的时候无收益，这是比跨式期权更激进的方案。

##### 蝶式差价期权
买入 $K_{1}<K_{3}$两个价格的看涨期权 卖出两份 $K_{2}=\frac{K_{1}+K_{3}}{2}$ 的看涨期权 ，收益满足
$$W = (S_{T}- K_1)^{+} + (S_{T}- K_{3})^{+} - 2 (S_{T}- K_2)^{+}$$
一种风险对冲策略，只有在股票价格恰好为$K_2$的时候我们才能收益最大化，相比于跨式期权的剧烈波动，蝶式差价期权是预测股票价格稳定的时候的方案，同时使用两份不同价格的看涨对冲部分风险
##### 倒置风险期权
买入$K_2$看涨 卖出$K_1$看跌 其中 $K_{1}<K_2$ 收益满足
$$W = (S_{T}-K_{2})^{+} - (K_{1} - S_{T})^{+}$$
纯粹的看涨行为 同时靠卖出看跌期权的收益和价格上涨行权看涨期权的收益获取利润，平衡了部分看涨期权未能行权的亏损

### 连续复利债券
很多资产的价值都和利率有关，对于这种资产我们一般称为固定收益资产。

最经典的固定收益资产是零息债卷 Bond 。他在指定的时期（到期日）会为你支付指定的现金（称为面额）。只要利率大于0，那零息债卷的价格在到期前一定是低于面额的。

不妨设目前时刻$t$ 到期时刻$T$ 到期时零息债卷价格为$1$ 每个周期利率为$r$  那么可以给出目前时刻零息债卷的价格$x$ 满足
$$ x e^{r(T-t)} = 1$$
也就是
$$B_{t}(T)= e^{-r(T-t)}$$

至于为什么是 $e$ 那是因为 正常的连续复利满足
$$(1 + \frac{r}{n})^{n}$$
随着 $n \to \infty$ 也就是高频计算利息 则有其极限为 $e^{rt}$ 也就是连续复利

## 二叉树无套利定价模型
### 单时段二叉树模型
#### 单时段二叉树定价模型的定义
二叉树资产定价模型将会为我们理解套利定价理论提供知识基础，我们分别考虑单时段的二叉树定价模型，更为广泛使用的多时段二叉树定价模型，以及模型的相关计算。值得注意的是： **二叉树定价模型针对离散时间的问题研究** 对连续时间不适用。

单时段的二叉树模型有两个时刻，时刻0 和 时刻1 在时刻0的时候我们持有一份股票，其价格为$S_0$ 他默认为正数。在时刻1，股票有两个可能的价格 $S_1(H)$ 与 $S_1(T)$  其中$H,T$ 分别代表一次硬币投掷实验的正面与反面。也就是说时刻1 的股票价格取决于一次投硬币实验。我们假设这次实验出现正面的概率为$p$ 那对应的反面的概率就是 $q = 1-p$ 

据此我们可以定义两个新的正数
$$u=\frac{S_1(H)}{S_0},\quad d=\frac{S_1(T)}{S_0}$$
我们保证$d < u$ 如果违背在调换硬币正反面的定义 我们称$u$ 是上升因子 $d$ 是下降因子 仅凭直觉，我们可以说$u >1, d<1$  通常我们假定$d = \frac{1}{u}$不过这和我们的推导无关。

为了让模型更能贴合市场，我们引入利率 $r$  意味着 时刻0 的 $1$美元在 时刻1 经由货币市场变为$1+r$ 美元  一般市场 $r \ge 0$ 我们的推导只要求    $r \ge -1$

有效市场的本质在于：**如果一个交易可以不劳而获，那他一定存在风险，否则我们就称为存在套利机会** 所谓的套利就是亏损的概率为0 而赚钱的概率为正 我们的模型不允许套利的存在，在真实市场中， 只要存在套利机会就会有套利的交易发生让机会消失。

在单时段二叉树模型里，为了避免套利，我们要求
$$0<d<1+r<u$$
我们已假设股票价格恒为正值，故必有$d>0$。

其他两个不等式是根据无套利得来的，我们对此解释如下：如果 $d\geqslant1+r$,投资者可以零财富开始，在时刻 0 从银行借款买股票。即使遇到最差的情况，时刻 1股票的价值也足以偿付银行的贷款，此外仍有余钱的概率为正，这就提供了一个套利机会。

另一方面，如果 $u\leqslant1+r$, 投资者可卖空股票，将所得投资货币市场。即使在股票的最好情况下，时刻1股票的价值也不会超过在货币市场投资的价值，又时刻 1股票的价值严格小于货币市场投资价值的概率为正，这也提供了一个套利机会。

真实的股票变动远比二叉树模型复杂，但是
* 二叉树模型可以清楚的解释套利与风险中性定价
* 多时段的二叉树模型可以有效的处理连续时间问题
* 他将引出后面我们一些重要的数学概率

#### 考虑欧式看涨期权的定价
考虑一个欧式看涨期权，它赋予持有者在时刻1以敲定价格 $K$购买一份股票的权利(但非义务)。假定 ${S}_1(T)<K<{S}_1(H)$,如果时刻 1 的股票价格低于敲定价格 $K$,则期权无价值；如果时刻 1 的股票价格高于敲定价格$K$,则期权被实施，由此获利为 $S_{1}(H)-K$。因此，期权在时刻1 的价值为$(S_1-K)^+$ 期权的定价问题就在于，已知时刻1的结果前 期权在时刻0的价值是多少。

下面我们用一个例子来说明期权的套利定价方法

考虑单时段模型，设 $S(0)=4,u=2,d=\frac{1}{2}$, $r=\frac{1}{4}$,则 $S_1(H)=8$ 且 $S_1(T)=2$。设欧式看涨期权的敲定价格 $K=5$.

进一步假设我们的初始财富为 $X_0=1.20$,同时在时刻 0 买进 $\Delta_0=\frac12$份股票。因为在时刻 0,每份股票价格为 4,我们必须利用初始财富 $X_0=1.20$,并且借入额外的 0.80。这样我们的现金头寸为 $-0.8$ 在时刻 1, 我们的现金头寸为$(1+r)(X_0-\Delta_0S_0)=-1$(即我们在货币市场的负债为 1)。

另一方面，在时刻 1,我们将有价值为$\frac12S_1(H)=4$ 或者$\frac12S_1(T)=1$ 的股票。具体地说，如果抛掷硬币的结果为正面，在时刻1,我们(股票和货币市场账户)的资产组合价值将为：
$$X_1(H)=\frac12S_1(H)+(1+r)(X_0-\Delta_0S_0)=3$$
对应的，如果是反面
$$X_1(T)=\frac12S_1(T)+(1+r)(X_0-\Delta_0S_0)=0$$

任何情况下，在时刻 1,资产组合与期权价值相等，即$(S_1(H)-5)^+=3$ 或
$(S_1(T)-5)^+=0$。我们通过在股票与货币市场的交易复制了期权。

为了复制上述资产组合所需要的初始财富1.2 就是期权在时刻0的无套利价格，如果期权价格高于1.2 卖出期权者就可以复制该投资组合并把多余的钱投入货币市场，实现无风险的套利行为，并且无须投入任何本金。

如果期权价格低于1.2 那投资者应该买入期权，建立反向的投资策略：也就是卖空半份股票，将所得2元 分别购买期权（小于1.2的价格）和投入货币市场。实现无须投入本金的无风险套利。

如果期权价格恰好是1.2 那么无论是买入期权还是卖出期权两种策略都不存在套利空间，这就是无套利原理。

我们上面的论述基于以下假设
* 股票可以无限细分买卖且允许做空
* 投资与借贷利率相同且非负
* 无交易费用与借贷风险
* 股票下时刻只有两个可能价格（二叉树模型需求）
* 无套利原理（推证需求）

他们在现实世界中近似成立

实际上，根据看涨看跌期权平价原理，对看涨期权的定价完成后看跌期权的定价也容易计算出。公式为
$$C+K=P+S_T$$
其中$C$是看涨期权价格 $K$是行权价格（我们要买入这么多计息资产，一般是债卷）$P$是看跌期权价格 $S$是股票标的当前价格

也就是构建两份在$T$时刻收益相同的资产，分别是
* 看涨期权+$e^{-rT}K$的现金（现金进行了折现）
* 看跌期权+一份标的股票
公式基于无套利原理推出
#### 衍生证卷及其定价模型
在一般的单时段模型中，我们定义衍生证券为一种证券，如果抛掷硬币的结果为正面，在时刻 1 的支付则为$V_1(H);$如果抛掷硬币的结果为背面， 在时刻 1 的支付则为$V_1(T)$。

欧式看涨期权是一种特别的衍生证券（支付为$(S_{1}-K)^{+}$）。另一种是欧式看跌期权，在时刻 1 的支付为($K-S_1)^+$,其中 $K$ 为常量。第三种是远期合约，这种衍生证券在时刻 1 的价值为$S_1-K$

为了确定衍生证卷的价格，我们仍采取复制的策略

假设我们的初始财富为$X_0$,在时刻 0 买入 $\Delta_0$份股票，现金头寸为$X_0-\Delta_0S_0$
那么时刻1的组合价值为
$$X_1=\Delta_0S_1+(1+r)(X_0-\Delta_0S_0)=(1+r)X_0+\Delta_0[S_1-(1+r)S_0]$$

我们希望选择$X_0$和$\Delta_0$使得$X_1(H)=V_1(H)$和$X_1(T)=V_1(T)[$这里$V_{1}(H)$和$V_1(T)$为已知值，是期权约定的，取决于抛掷硬币的结果；在时刻 0,我们知道$V_1(H)$和$V_1(T)$的发生概率，但不知道这两个可能性中哪个会成为现实]

因此，为复制，必须有：
$$X_0+\Delta_0\left(\frac1{1+r}S_1(H)-S_0\right)=\frac1{1+r}V_1(H)$$
$$X_0+\Delta_0\left(\frac1{1+r}\mathrm{S}_1(T)-\mathrm{S}_0\right)=\frac1{1+r}V_1(T)$$
这是一个二元方程组 处理他可以得到 **delta 对冲公式**
$$\Delta_0=\frac{V_1(H)-V_1(T)}{S_1(H)-S_1(T)}$$
$X_0$ 也可给出
$$X_0=\frac1{1+r}[\tilde{p}V_1(H)+\tilde{q}V_1(T)]$$

按照计算出的方案就可以复制出本节开篇给出的衍生证卷，其定价就应该是复制证卷所需的资金$X_0$ 这个公式也被我们称为**风险中性定价公式**。

概率解出的结果为
$$\tilde{p}=\frac{1+r-d}{u-d},\quad\tilde{q}=\frac{u-1-r}{u-d}$$
**他们是风险中性概率而不是真实概率**，只是帮助我们解方程的工具
### 多时段二叉树模型
我们现在将上节的研究推广到多时段。设想反复抛掷硬币，只要抛掷结果是正面，股票价格乘以上升因子 $u;$只要抛掷结果是背面，股票价格乘以下降因子$d$。市场上除了股票外，还存在常利率 $r$ 的货币市场资产。关于这些参数的唯一假设是无套利条件
$$0<d<1+r<u$$

有了多阶段模型的定义，我们可以容易的给出两阶段模型的股票价格
$$S_2(HH)=uS_1(H)=u^2S_0,\quad S_2(HT)=dS_1(H)=duS_0,$$
$$S_2(TH)=uS_1(T)=udS_0,\quad S_2(TT)=dS_1(T)=d^2S_0$$

据此我们也可以容易的给出更多阶段的模型。

我们还是研究欧式看涨期权，他在时刻 $T$ 敲定 并且敲定价格为 $K$ 此时期权的定价需要考虑前面若干次硬币投掷的结果了。

定理 (**多时段二叉树模型中的复制**) 考虑一个 N时段二叉树资产定价模型，其中 $0<d<1+r<u$,并且；
$$\tilde{p}=\frac{1+r-d}{u-d},\quad\tilde{q}=\frac{u-1-r}{u-d}$$
设$V_{N}$ 为一个随机变量(衍生证券在时刻 N 的支付),它取决于前 N 次抛掷
硬币过程$\omega_1\omega_2{\cdots}\omega_N$。由
$$V_n(\omega_1\omega_2\cdots\omega_n)=\frac{1}{1+r}[\widetilde{p}V_{n+1}(\omega_1\omega_2\cdots\omega_nH)+\widetilde{q}V_{n+1}(\omega_1\omega_2\cdots\omega_nT)]$$
可以按照时间反向递归得到 $V_{n-1}...V_0$  他们就是衍生证卷的价格。

通过以下公式就可以知道模拟期权所需要买卖的股票数量
$$\Delta_n(\omega_1\cdots\omega_n)=\frac{V_{n+1}(\omega_1\cdots\omega_nH)-V_{n+1}(\omega_1\cdots\omega_nT)}{S_{n+1}(\omega_1\cdots\omega_nH)-S_{n+1}(\omega_1\cdots\omega_nT)}$$

**该定理也适用于路径依赖期权，只需要按照定理要求递推就可以得到定价。**
### 模型的计算
#### 考虑一个回望期权
例 $S_0=4,u=2,d=\frac12$,又假设利率 $r=\frac{1}{4}$,则 $\widetilde p=\tilde{q}=\frac12$。考虑这样一个回望期权，其在时刻 3 的支付为：
$$V_3=\max_{0\leqslant n\leqslant3}S_n-S_3$$
则我们可以进行如下的定价推导

$V_3$定价有
$$\begin{aligned}
&V_{3}(HHH)=&& S_3(HHH)-S_3(HHH)=32-32=0 \\
&V_{3}(HHT)=&& S_{2}(HH)-S_{3}(HHT)=16-8=8 \\
&V_{3}(HTH)=&& S_1\left(H\right)-S_3\left(HTH\right)=8-8=0 \\
&V_{3}(HTT)=&& S_1(H)-S_3(HTT)=8-2=6 \\
&V_{3}(THH)=&& S_3(THH)-S_3(THH)=8-8=0 \\
&V_{3}(THT)=&& S_{2}(TH)-S_{3}(THT)=4-2=2 \\
&V_{3}(TTH)=&& S_{0}-S_{3}\left(TTH\right)=4-2=2 \\
&V_{3}(TTT)=&& S_{0}-S_{3}\left(TTT\right)=4-0.50=3.50 
\end{aligned}$$
反推$V_2$定价有
$$\begin{gathered}
V_{2}(HH)= \frac{4}{5}\biggl[\frac{1}{2}V_{3}(HHH)+\frac{1}{2}V_{3}(HHT)\biggr]=3.20 \text{.} \\
V_{2}(HT)= \frac{4}{5}\biggl[\frac{1}{2}V_{3}(HTH)+\frac{1}{2}V_{3}(HTT)\biggr]=2.40 \\
V_{2}(TH)= \frac{4}{5}\bigg[\frac{1}{2}V_{3}(THH)+\frac{1}{2}V_{3}(THT)\bigg]=0.8 \text{-} \\
V_{2}(TT)= \frac{4}{5}\bigg[\frac{1}{2}V_{3}(TTH)+\frac{1}{2}V_{3}(TTT)\bigg]=2.20 
\end{gathered}$$
反推$V_1$有
$$\begin{gathered}
V_{1}(H)= \frac{4}{5}\bigg[\frac{1}{2}V_{2}(HH)+\frac{1}{2}V_{2}(HT)\bigg]=2.24 \\
V_{1}(T)= \frac{4}{5}\bigg[\frac{1}{2}V_{2}(TH)+\frac{1}{2}V_{2}(TT)\bigg]=1.20 \\
\end{gathered}$$
得到期权的初始定价为
$$V_{0}= \frac{4}{5}\bigg[\frac{1}{2}V_{1}(H)+\frac{1}{2}V_{1}(T)\bigg]=1.376 $$

当然，我们可以通过购买下面数量的股票就可以复制出第一阶段的投资组合
$$\Delta_0=\frac{V_1(H)-V_1(T)}{S_1(H)-S_1(T)}=\frac{2.24-1.20}{8-2}=0.1733$$
#### 简化运算的思路
如果用初始的$V(TT..HH)$思路来描述空间，那么二叉树模型的结果将会有$2^n$种，对于过多阶段的模型，我们迭代计算起来是苦难的，指数的膨胀太可怕了。

不过如果只考虑股票价格与对应的衍生证卷支付，那问题就简化多了，3阶段的模型只有4种价格，实际上对于二叉树模型，$n$阶段会带来 $n+1$中股票价格。我们可以把原本的$V(TT..HH)$变化为$v(s)$其中$s$是目前的股票价格，映射后的结果依旧是衍生证卷的价格。

那么对于3阶段模型
$$V_{2}\left(\omega_{1} \omega_{2}\right)=\frac{2}{5}\left[V_{3}\left(\omega_{1} \omega_{2} H\right)+V_{3}\left(\omega_{1} \omega_{2} T\right)\right]$$
变化为
$$v_{2}(s)=\frac{2}{5}\left[v_{3}(2 s)+v_{3}\left(\frac{1}{2} s\right)\right]$$

事实上，期权在任意时刻$n$的价格只和股票价格$S_n$有关，而和投掷硬币的情况无关，我们可以用$v(S_n)$代替原本的$V_(TT...HH)$来进行计算而不影响结果

在回望期权等路径依赖期权中，类似的思路依旧可用，但是不能简单的直接套用。我们需要根据题意将原本的依赖价格计算的公式增加回望期权的特性
## 随机分析的数学基础
### 离散时间鞅
#### 股票价格贴现过程
我们前面给出了股票目前的价格是未来价格的平均贴现值
$$S_n(\omega_1\cdots\omega_n)=\frac1{1+r}[\widetilde{p}S_{n+1}(\omega_1\cdots\omega_nH)+\widetilde{q}S_{n+1}(\omega_1\cdots\omega_nT)]$$
有了条件期望的符号后改写为
$$S_n=\frac1{1+r}\mathbb{E}_n[S_{n+1}]$$
两边除以$(1+r)^n$有
$$\frac{S_{n}}{(1+r)^{n}}=\widetilde{\mathbb{E}}_{n}\left[\frac{S_{n+1}}{(1+r)^{n+1}}\right]$$
经济学含义是：对于不支付红利的股票，基于$n$时刻信息对$n+1$时刻的股票贴现价格的最好估计就是$n$时刻的贴现价格。

#### 适应与鞅
不过我们来这里不是为了讨论经济学，实际上，这样的一个过程就是一个鞅

定义：对于随机变量序列$\{M_{i}\},i = 0,1,2,...$  每个$M_n$只依赖前$n$次的硬币投掷结果，我们称这样的随机过程为**适应随机过程**

定义：如果一个适应随机过程$M_n$满足
$$M_{n}= E_n[M_{n+1}]$$
则称这个过程是一个鞅（过程）**意味着股票无溢价**

对应的，如果满足$M_{n}\le E_n[M_{n+1}]$ 则称这个有递增趋势的序列是一个**下鞅**；如果满足$M_{n}\ge E_n[M_{n+1}]$ 则称这个有递减趋势的序列是一个**上鞅**

鞅的定义中，是一步超前的性质，但是实际上我们可以证明
$$M_{n}= E_n[M_{m}]$$
其中$m \ge n$ 

并且我们知道，鞅的直接期望（非条件期望是常数）
$$M_{0}= E[M_{n}]$$

鞅没有递增或者递减的趋势，但是股票有，因此我们一般认为股票是一个上鞅（如果有递减的趋势）或是一个下鞅（如果有递增的趋势）

但是，如果我们从真正的概率变为风险中性概率，那与之对应的股票贴现价格就是一个鞅了。本质上，这还是因为风险中性概率和真实概率之间的不同，**风险中性概率用$u,d,r$计算得到，是保证风险中性原则的概率**

#### 财富过程与衍生证卷贴现价格
我们继续回归考虑在二叉树定价模型中研究的财富过程
$$X_{n+1}=\Delta_nS_{n+1}+(1+r)(X_n-\Delta_nS_n)$$
他也是适应过程，我们直接给出结论，对于贴现的财富过程$\frac{X_{n+1}}{(1+r)^{n+1}}$  是风险中性概率测度下的鞅 也就是
$$\frac{X_n}{(1+r)^n}=\tilde{\mathbb{E}}_n\bigg[\frac{X_{n+1}}{(1+r)^{n+1}}\bigg],\quad n=0,1,\cdots,N-1$$
并且可以自然的给出推论
$$\tilde{\mathbb{E}}\frac{X_n}{(1+r)^n}=X_0,\quad n=0,1,\cdotp\cdotp\cdotp,N$$

实际上，我们可以继续给出推论，衍生证卷的贴现价格在风险中性概率测度下也是一个鞅，即
$$\frac{V_n}{(1+r)^n}=\mathbb{E}_n\bigg[\frac{V_{n+1}}{(1+r)^{n+1}}\bigg],\quad n=0,1,\cdots,N-1$$


### 连续时间鞅
这里本应该是随机过程的内容，但是在其中未能叙述，随机分析中又要使用到，因此进行补充。

#### 信息和域流
定义：设$T$是固定的正常数，且$\{\mathscr{F}_t\}_{t\in[0,T]}$是一族$\sigma$-代数，若对$\forall0\leqslant s\leqslant t\leqslant T$,都有$\mathscr{F}_s\subset\mathscr{F}_t$,则称该族$\sigma-$代数 $\left\{\mathscr{F}_t\right\}_{t\in[0,T]}$ 是域流/过滤(filtration)

**也就是说，域流/过滤(filtration)是一族非减的$\sigma$代数**

大多数情况下，我们会取$\mathscr{F}_{0} = \{ \phi, \Omega\}$  也就是平凡$\sigma$代数

定义：我们称$\left(\Omega,\mathcal{F},\mathbb{F}=\left\{\mathcal{F}_{t};t\geqslant0\right\},\mathbb{P}\right)$为一个过滤概率空间，也就是一个有过滤的概率空间

定义：设$(\Omega,\mathscr{F},\{\mathscr{F}_t\}_{t\in[0,T]},\mathbb{P})$ 是一过滤概率空间，并且$\{X_t\}_{t\in[0,T]}$是其上的一族随机变量。若对$\forall t\in[0,T]$,都有$X_t\in\mathscr{F}_t$,则称随机变量关于该域流是适应的。
#### 鞅与马氏过程
定义： 设$(\Omega,\mathscr{F},\{\mathscr{F}_t\}_{t\in[0,T]},\mathbb{P})$是一过滤概率空间，并且$X_t$是其上的随机过程如果
* $X$是适应的
* 对任意 $s\leq t$,都有$\mathbf{E}[X_t|\mathscr{F}_s]\geq(\leq)X_s$,
则称$X$是下 ( 上 ) 鞅 。如果随机过程 $X$ 既是下鞅又是上鞅 , 则称 $X$ 是鞅

定义：设$(\Omega,\mathscr{F},\{\mathscr{F}_t\}_{t\in[0,T]},\mathbb{P})$是一过滤概率空间，并且$X=\{X_t\}_{t\in[0,T]}$是其上的随机过程。如果
* $X$是适应的 (即对$\forall t\in[0,T]$,有$X_t\in\mathscr{F}_t)$ 。
* 对有界函数$f$,存在函数$g$,使得$\mathbf{E}[f(X_t)|\mathscr{F}_s]=g(X_s)$ 其中$s<t$ 
则称$X$是马氏过程。 **这里比马氏链更加广义了，不过基本思路不变**

定义：对于定义在概率空间上的实值随机过程$\{X(t),t\geq0\}$,其二次变差也是一个随机过程，记作$[X]_t$,定义为：
$$[X]_t=\lim_{\|P\|\to0}\sum_{i=0}^{n-1}(X(t_{i+1})-X(t_i))^2$$
其中，$P$取遍区间$0,t$所有的划分，$\|P\|$等于$P$中最长的子区间的长度，极限使用依概率收敛来定义。

一般，我们使用符号$Q_{\Pi }$ 表示 划分 $\Pi$ 的$\sum_{i=0}^{n-1}(X(t_{i+1})-X(t_i))^2$ 也就是
$$Q_{\Pi }=\sum_{i=0}^{n-1}(X(t_{i+1})-X(t_i))^2$$
最后借助极限与求和交换顺序的方法证明二次变差的取值。

#### 布朗运动及其性质
##### 布朗运动的定义
在随机过程中，布朗运动定义如下：若实随机过程W=$\{\mathbf{W}_{t},\mathbf{t}\geq0\}$满足： 
* $\quad W_0=0$ 
* $\quad W=\{W_t,t\geq0\}$ 是平稳的独立增量过程.
*   **对任意的$0\leq s<t$,有$W_t-W_s\sim N(0,(t-s))$**
则称随机过程W是标准布朗运动(维纳过程) 

去除掉第一条初值为0 则称为布朗运动

此时我们将第二条修改为：**$W_t$具有独立增量性，对于任意$0\le s \le t$，$W_t-W_s$与$\mathscr{F}_s$相互独立**

修改后的性质实际上和原性质等价，但更加有利于我们后面的讨论，这里对其等价性不在证明

##### 布朗运动的均值与相关函数
参见[随机过程基础](/blog/2023/03/18/stochastic-process-basics-notes/) 的“布朗运动 / 数字特征”一节。

##### 布朗运动是鞅
适应性无需证明，只用证明鞅性
$$\begin{aligned}
\mathbb{E}[W_t|\mathscr{F}_s]& =\mathbb{E}[W_t-W_s+W_s|\mathscr{F}_s] \\
&=\mathbb{E}[W_t-W_s|\mathscr{F}_s]+\mathbb{E}[W_s|\mathscr{F}_s] \\
&=\mathbb{E}[W_t-W_s]+W_s \\
&=W_{s}
\end{aligned}$$

##### 布朗运动是马氏过程
适应性无需证明，只用证明马氏性，即对有界函数$f$,存在函数$g$,使得$\mathbf{E}[f(X_t)|\mathscr{F}_s]=g(X_s)$ 

根据布朗运动相关证明基本的构造思想，有
$$\mathbb{E}[f(W_t)|\mathscr{F}_s]=\mathbb{E}[f(W_t-W_s+W_s)|\mathscr{F}_s]$$
不妨就让后面的期望作为我们的$g(X_s)$  因为$W_t-W_s\bot\mathscr{F}_s$和$W_s\in\mathscr{F}_s$  有
$$g(x)=\mathbb{E}[f(W_t-W_s+x)]=\int_{-\infty}^{+\infty}f(y+x)e^{-\frac{y^2}{2(t-s)}}dy$$
就是满足题意的$g(x)$ 因此布朗运动是马氏过程

##### 布朗运动的二次变差
布朗运动在$[0,T]$上的二次变差就是$T$ 我们直接就此开始证明 记
$$Q_\Pi=\sum_{j=0}^{n-1}(W_{t_{j+1}}-W_{t_{j}})^2$$
根据布朗运动的定义，我们知道 $W_{t_{j+1}}-W_{t_{j}}$ 就是一个均值为0正态分布

容易计算期望与方差
$$\mathbb{E}[(W_{t_{j+1}} -W_{t_{j}})^2]=\mathrm{Var}[W_{t_{j+1}}-W_{t_{j}}]=t_{j+1}-t_j$$

$$\begin{aligned}&\mathrm{Var}[(W_{t_{j+1}}-W_{t_{j}})^2]=\mathbf{E}[\left|(W_{t_{j+1}} -W_{t_{j}})^2-(t_{j+1}-t_j)\right|^2]\\&=\mathbb{E}[(W_{t_{j+1}}-W_{t_{j}})^4]-2(t_{j+1}-t_j)\mathbb{E}[(W_{t_{j+1}}-W_{t_{j}})^2]+(t_{j+1}-t_j)^2\\&=3(t_{j+1}-t_j)^2-2(t_{j+1}-t_j)^2+(t_{j+1}-t_j)^2=2(t_{j+1}-t_j)^2\end{aligned}$$

综上，我们有
$$\begin{gathered}
\mathbf{E}[\left(Q_\Pi-T\right)^2]=\mathbf{E}\left[\left(\sum_{j=0}^{n-1}\left(W_{t_{j+1}} -W_{t_{j}}\right)^2-T\right)^2\right] \\
=\sum_{j=0}^{n-1}\mathbb{E}\left[\left|\left(W_{t_{j+1}}-W_{t_{j}}\right)^2-\left(t_{j+1}-t_j\right)\right|^2\right] \\
=\sum_{j=0}^{n-1}\mathrm{Var}[\left(W_{t_{j+1}}-W_{t_{j}}\right)^2]=\sum_{j=0}^{n-1}2(t_{j+1}-t_j)^2 \\
\leq\sum_{j=0}^{n-1}2\parallel\Pi\parallel(t_{j+1}-t_j)=2\parallel\Pi\parallel T\to0.\quad(\parallel\Pi\parallel\to0) 
\end{gathered}$$

我们一般把布朗运动二次变差记为
$$dW(t)dW(t)=dt$$
## 随机分析基础
### Riemann - Stieltjes 积分
Riemann - Stieltjes 积分是Riemann积分的推广形式，允许我们把原本的对$x$积分转换为对函数$g(x)$积分，这对于我们理解随机积分很有帮助。

Riemann - Stieltjes 积分的基本形式如下
$$\int_a^bf\left(x\right)dg\left(x\right)$$
很自然的，如果$g(x)$处处可导，我们可以直接转换为Riemann积分研究，不过使用基本的分割求和思想研究本质才是本节想要讨论的。

分割求和有
$$\begin{aligned}S_\Pi(f)&=\sum_{k=1}^nf(c_k)[g\left(x_{k+1}\right)-g\left(x_k\right)]\\&c_k\in[x_k,x_k+1]\end{aligned}$$
如果存在一个值$I$,使得对于任意的$\varepsilon>0$,都存在一个$\delta>0$,使得对于任意满足$\|P\|<\delta$的分割$P$,都有$|I-S(f,g,P)|<\varepsilon$,则称$I$为$f$关于$g$在$a,b$上的黎曼-斯蒂尔杰斯积分，并表示为：
$$\int_a^bf\left(x\right)dg\left(x\right)$$

这就是RS积分的基本定义，本质上还是分割求和思想的一个变种。

### Ito积分的定义

对于正数$T$ 我们来研究下面积分
$$\int_0^T\Delta(t)dW(t)$$
里面的基本要素包括布朗运动$W(t)$  对应的域流$\mathscr{F}_t$  被积函数$\Delta(t)$需要是适应随机过程。我们遇到的最大的问题是 **布朗运动本身不可关于时间微分**

因此，我们希望先从最简单的随机过程入手，也就是简单过程
$$\Delta_t=\sum_{k\operatorname{=}0}^{n\operatorname{-}1}\Delta_{t_k}\mathbf{1}_{[t_k,t_{k\operatorname{+}1})}(t\operatorname{)}$$
这意味着，如果积分在某个局部进行就是常数被积，我们可以给出下面的推论
$$\begin{aligned}&I\left(t\right)=\Delta{_0}(W_t-W_{t_0})=\Delta_0W_t,\quad0\leq t\leq t_1,\\&I\left(t\right)=\Delta_0W_{t_1}+\Delta_{t_1}(W_t-W_{t_1}),\quad t_1\leq t\leq t_2,\\&I\left(t\right)=\Delta_0W_{t_1}+\Delta_{t_1}(W_{t_2}-W_{t_1})+\Delta_{t_2}(W_t-W_{t_2}),\quad t_2\leq t\leq t_3.\end{aligned}$$
一般的，对于某个$k$ 使得 $t_{k}\le t \le t_{k+1}$ 
$$I\left(t\right)=\sum_{j=0}^{k-1}\Delta_{t_j}\left(W_{t_{j+1}}-W_{t_j}\right)+\Delta_{t_k}(W_t-W_{t_k}).$$

对于一般的$\Delta(t)$ 我们可以选择一系列的简单过程$\Delta_n(t)$ 逼近他，这一列简单过程随机积分的极限就是原本函数的Ito积分。

Ito积分也是一个随机过程，指标$t$就是他的时间，是积分的上限。
### Ito积分的性质
#### Ito积分的六个性质
设$T$是正数，$\Delta(t)$ 是满足前述性质的适应随机过程，则Ito积分$\int_0^T\Delta(t)dW(t)$
有下面的性质：
* **连续性**：作为积分上限$t$的函数，$I(t)$的路径连续
* **适应性**：对每个$t,I(t)$为 $\mathcal{F}(t)$-可测
* **线性性**：$\int_0^t(\alpha\Delta_s+\beta\Gamma_s)dW_s=\alpha\int_0^t\Delta_sdW_s+\beta\int_0^t\Gamma_sdW_s.$
* **鞅性**：$\{I\left(t\right)\}_{t\in[0,T]}\text{ 关于域流 }\{\mathscr{F}_t\}_{t\in[0,T]}\text{ 是鞅}$
* **Ito等距**：$\mathbb{E}[I^2(t)]=\mathbb{E}\bigg[\int_0^t\Delta_u^2du\bigg]$
* **二次变差**：$dI\left(t\right)\cdot dI\left(t\right)=\Delta_t^2dt$

我们对鞅性，Ito等距，二次变差进行单独的证明研究，核心点是结合定义式中的
$$I\left(t\right)=\sum_{j=0}^{k-1}\Delta_{t_j}\left(W_{t_{j+1}}-W_{t_j}\right)+\Delta_{t_k}(W_t-W_{t_k}).$$
展开基于定义的证明。 至于剩下三个性质，非常自然，无须额外证明。

#### Ito积分是鞅
从定义来说，我们需要证明
$$\text{对任意 }0\leq s\leq t\leq T,\text{ 有 }\mathbb{E}[I(t)|\mathscr{F}_s]=I(s).$$
如果$s,t$属于同一个划分（简单随机过程的划分）则有$t_k\leq s\leq t<t_{k+1}$ 因此使用我们研究布朗运动的做差的思想
$$I\left(t\right)-I\left(s\right)=\Delta_{t_k}\left(W_t-W_s\right).$$
还是使用前面研究布朗运动二次变差一样的收敛方法
$$\begin{aligned}\mathbb{E}[I(t)-I(s)|\mathscr{F}_s]&=\mathbb{E}[\Delta_{t_k}(W_t-W_s)|\mathscr{F}_s]\\&=\Delta_{t_k}\mathbb{E}[(W_t-W_s)|\mathscr{F}_s]=\Delta_{t_k}\mathbb{E}[W_t-W_s]=0\end{aligned}$$
因此有鞅性得证。

若$s$和$t$不属于同一划分，即存在$t_\ell$和$t_k$,满足$t_\ell<t_k$,且$s\in[t_\ell,t_{\ell+1})$和$t\in[t_k,t_{k+1})$,则对$I(t)$变形有
$$\begin{aligned}I\left(t\right)&=\sum_{j=0}^{\ell-1}\Delta_{t_j}\left(W_{t_{j+1}}-W_{t_j}\right)+\Delta_{t_\ell}\left(W_{t_{\ell+1}}-W_{t_\ell}\right)\\&+\sum_{j=\ell+1}^{k-1}\Delta_{t_j}\left(W_{t_{j+1}}-W_{t_j}\right)+\Delta_{t_k}\left(W_t-W_{t_k}\right)\end{aligned}$$
继续对此拆分
$$\begin{aligned}I\left(t\right)&=\sum_{j=0}^{\ell-1}\Delta_{t_j}\left(W_{t_{j+1}}-W_{t_j}\right)+\Delta_{t_\ell}\left(W_s-W_{t_\ell}\right)+\Delta_{t_\ell}\left(W_{t_{\ell+1}}-W_s\right)\\&+\sum_{j=\ell+1}^{k-1}\Delta_{t_j}\left(W_{t_{j+1}}-W_{t_j}\right)+\Delta_{t_k}\left(W_t-W_{t_k}\right)\end{aligned}$$
继续做差有
$$I\left(t\right)-I\left(s\right)=\Delta_{t_\ell}\left(W_{t_{\ell+1}}-W_s\right)+\sum_{j=\ell+1}^{k-1}\Delta_{t_j}\left(W_{t_{j+1}}-W_{t_j}\right)+\Delta_{t_k}\left(W_t-W_{t_k}\right)$$
还是使用条件期望为0证明鞅性

#### Ito等距
我们想要证明
$$\mathbb{E}[I^2(t)]=\mathbb{E}\bigg[\int_0^t\Delta_u^2du\bigg]$$
将前面的积分式进行改写$D_j=W_{t_{j+1}}-W_{t_j}$ 且 $D_k=W_t-W_{t_k}$ 则有$I\left(t\right)=\sum_{j=0}^{k-1}\Delta_{t_{j}}D_{j}$ 进而，原期望可以改写为
$$\begin{aligned}\mathbb{E}[I^2(t)]&=\mathbb{E}\left[\left(\sum_{j=0}^{k-1}\Delta_{t_j}D_j\right)^2\right]\\&=\sum_{j=0}^{k-1}\mathbb{E}\left[\Delta_{t_j}^2D_j^2\right]+2\sum_{0\leq i<j\leq k}\mathbb{E}\left[\Delta_{t_i}D_i\Delta_{t_j}D_j\right]\end{aligned}$$
由于独立性和布朗运动性质，交叉项期望一定为0 则
$$\begin{aligned}\mathbb{E}[I^2(t)]&=\sum_{j=0}^{k-1}\mathbb{E}\left[\Delta_{t_j}^2D_j^2\right]=\sum_{j=0}^{k-1}\mathbb{E}\left[\Delta_{t_j}^2\right]\mathbb{E}\left[D_j^2\right]\\&=\sum_{j=0}^{k-1}\mathbb{E}[\Delta_{t_j}^2(t_{j+1}-t_j)]+\mathbb{E}[\Delta_{t_k}^2(t-t_k)]\\&=\mathbb{E}\bigg[\int_0^t\Delta_u^2du\bigg].\end{aligned}$$
#### 二次变差
想要证明
 $$dI\left(t\right)\cdot dI\left(t\right)=\Delta_t^2dt$$
形式上: $I(t)=\int_{0}^{t} \Delta_{s} d W_{s}$ , 于是  $d I(t)=\Delta_{t} d W_{t}$ .

从而  $$d I(t) \cdot d I(t)=\Delta_{t} d W_{t} \cdot \Delta_{t}  W_{t}=\Delta_{t}^{2} d W_{t} \cdot d W_{t}=\Delta_{t}^{2} d t$$
证明完毕

### Ito过程与Ito公式
#### Ito过程（Ito Process）
Ito过程是一种连续的随机过程，它可以通过随机微分方程来描述。具体来说，一个Ito过程可以表示为：
$$dX_t=\Theta_tdt+\Delta_tdW_t$$
也有积分形式为
$$X_t=X_0+\int_0^t\Theta_sds+\int_0^t\Delta_sdW_s.$$
其中$W_t$是布朗运动，其他部分目前不用考虑，一般称$dt$前是漂移项，$dW_t$前是扩散项，分别称为漂移系数和扩散系数。

其二次变差为：
$$dX_tdX_t=\Delta_t^2dt$$
#### Ito公式（Ito's Formula）
Ito公式是随机微积分中的核心工具，它允许我们计算一个随机过程的函数的微分，设$X_t$是一个Ito过程，$f(x)$是二阶可微函数，那么$f(X_t)$也是Ito过程，考虑二元函数有$f(T,X_T)$ 

那么我们可以给出Ito公式有
$$\begin{aligned}
&\begin{aligned}&f\left(T,X_T\right)=f\left(0,X_0\right)+\int_0^Tf_t(t,X_t)dt+\int_0^Tf_x(x,X_t)dX_t\end{aligned} \\
&+\frac12\int_0^Tf_{xx}(x,X_t)dX_tdX_t \\
&=f\left(0,X_0\right)+\int_0^Tf_t(t,X_t)dt+\int_0^Tf_x(x,X_t)\Delta_tdW \\
&+\int_0^Tf_x(x,X_t)\Theta_tdt+\frac12\int_0^Tf_{xx}(t,X_t)\Delta_t^2dt.
\end{aligned}$$
这实际上就是Newton - L公式在随机分析中的形式，我们求偏导然后积分，最后补充一个二阶导与二次变差项，再使用Ito过程的二次变差性质与Ito过程的定义化简。

可以给出Ito公式的微分形式有
$$\begin{aligned}
df\left(t,X_{t}\right)& =f_t(t,X_t)dt+f_x(x,X_t)dX_t+\frac12f_{xx}(x,X_t)dX_tdX_t \\
&=f_t(t,X_t)dt+f_x(x,X_t)\Delta_tdW_t+f_x(x,X_t)\Theta_tdt \\
&+\frac12f_{xx}(t,X_t)\Delta_t^2dt.
\end{aligned}$$

形式上，Ito公式就是传统的全微分公式增加了二次变差项
#### 一个例子
我们有
$$X_t=\int_0^t\sigma_sdW_s+\int_0^t\biggl(\mu_s-\frac12\sigma_s^2\biggr)ds$$
并且$S_t=e^{X_t}$ 求$S_t$满足的方程

很自然，$X_t$是Ito过程，我们考试使用Ito公式，并按照惯例使用微分形式有
$$\begin{aligned}dS_t&=df\left(X_t\right)=f'(X_t)dX_t+\frac12f''(X_t)dX_tdX_t\\&=\left(\mu_t-\frac12\sigma_t^2\right)S_tdt+\sigma_tS_tdW_t+\frac12\sigma_t^2S_tdt\\&=\mu_tS_tdt+\sigma_tS_tdW_t.\end{aligned}$$

### Black-Scholes-Merton公式

#### BS公式定义
有了前面的讨论基础以后，我们考虑本章的收尾**Black-Scholes-Merton方程** 也就是金融领域非常重要的一个方程。

股票价格$S_t$ 是几何布朗运动
$$dS_t\boldsymbol{=}\mu_tS_tdt\boldsymbol{+}\sigma_tS_tdW_t.$$
投资策略设为$\Delta_t$  结合本文“单时段二叉树模型”一节内容有财富过程为
$$X_t=\Delta_tS_t+(X_t-\Delta_tS_t)$$
将前面的部分记为股票账户，后面的记为银行账户$D_t$ ；对于前者，微分只需要对$S_t$微分，对于后者 $dD_{t}= rD_{t}$ 即连续复利过程。则财富过程微分有
$$dX_t\boldsymbol{=}\Delta_tdS_t\boldsymbol{+}r\left(X_t\boldsymbol{-}\Delta_tS_t\right)dt$$
研究贴现财富过程
$$e^{-rt}X_t$$
欧式看涨期权满足
$$C_t(K,T;S)=c\left(t,S_t\right),\text{ 满足 }C_T(K,T;S)=(S_T-K)^+$$
根据无套利原理复制有
$$e^{-rt}X_t=e^{-rt}c\left(t,S_t\right)$$
也就是
$$d\left[e^{-rt}X_t\right]=d\left[e^{-rt}c\left(t,S_t\right)\right]$$
现在我们只需要根据Ito公式将两边的微分进行拆解得到BS方程
$$\begin{aligned}&c_t(t,x)+rxc_x(t,x)+\frac12\sigma^2x^2c_{xx}(t,x)=rc\left(t,x\right)\\&c\left(T,x\right)=(x-K)^+\end{aligned}$$
#### BS公式计算与性质
所有的计算都基于Ito公式进行，这里的证明实际上只是一些计算

**考虑贴现股价的微分** 即$d(e^{- rt}S(t))$ 他并不是必要的，但是可以帮助我们理解其中的一些计算。

先套公式
$$\begin{aligned}
d(e^{-rt}S(t))& =df(t,S(t)) \\
&=f_{t}(t,S(t))dt+f_{x}(t,S(t))dS(t)+\frac{1}{2}f_{x}(t,S(t))dS(t)dS(t)
\end{aligned}$$
代入计算，结合前一节的定义
$$\begin{aligned}&=-re^{-rt}S(t)dt+e^{-rt}dS(t)\\&=(\mu-r)e^{-rt}S(t)dt+\sigma e^{-rt}S(t)dW(t)\end{aligned}$$


**考虑贴现财富过程微分**
$$\begin{aligned}
d(e^{-rt}X(t))& =df(t,X(t)) \\
&=f_{t}(t,X(t))dt+f_{x}(t,X(t))dX(t)+\frac{1}{2}f_{xx}(t,X(t))dX(t)dX(t) \\
&=-re^{-rt}X(t)dt+e^{-rt}dX(t) \\
&={\Delta(t)}({\mu}-r)e^{-rt}S(t)dt+\Delta(t)\sigma e^{-rt}S(t)dW(t) \\
&& \\
&=\Delta(t)d(e^{-rt}S(t))
\end{aligned}$$
贴现财富过程价格的改变被贴现股票价格的改变完全决定

**考虑贴现期权价格微分**
$$\begin{aligned}&=e^{-rt}\left[-rc(t,S(t))+c_{t}(t,S(t))+\mu S(t)c_{x}(t,S(t))+\frac{1}{2}\sigma^{2}S^{2}(t)c_{xx}(t,S(t))\right]dt\\&+e^{-rt}\sigma S(t)c_{x}(t,S(t))dW(t)&\end{aligned}$$


重新代入
$$d\left[e^{-rt}X_t\right]=d\left[e^{-rt}c\left(t,S_t\right)\right]$$
根据$dW_t$相等可以得到
$$\Delta(t)=c_{x}(t,S(t))$$
这称为delta对冲法则

再令$dt$系数相等就可以得到前一节给出的结论
$$\begin{aligned}&c_t(t,x)+rxc_x(t,x)+\frac12\sigma^2x^2c_{xx}(t,x)=rc\left(t,x\right)\\&c\left(T,x\right)=(x-K)^+\end{aligned}$$
#### 三个符号
我们把函数$c(t,x)$关于各个变量的导数起了一些特殊的名字，以方便在某些真实的金融场景里使用

**delta**： $c_x(t,x)$

**theta**：$c_t(x,t)$

**gamma**：$c_{xx}(x,t)$

在其他金融书籍中会见到这个名字。


##  状态价格
在本文“二叉树无套利定价模型”一节我们同时研究了风险中性概率和真实概率两个概率。后者是真实存在的，是现实世界中的概率。前者是虚构的，是根据利率等资产变动的指标构造的。但对于很多结果的表达都有好处

这本质上是同一个概率空间上的两个概率测度，这里我们对金融中的这个问题进行更多讨论 其根源来自[高等概率论](/blog/2024/10/09/advanced-probability-notes/) 的“Radon-Nikodym 定理”一节

### 测度变换
考虑一般的有限样本空间 $\Omega$ 上的两个概率测度$\mathbb{P}$ 和$\widetilde{\mathbb{P}}$。假定$\mathbb{P}$ 和$\widetilde{\mathbb{P}}$ 对 $\Omega$
中的每个元素都给出正概率，因此我们可以写出下面的商：
$$Z(\omega)=\frac{\widetilde{\mathbb{P}}(\omega)}{\mathbb{P}(\omega)}$$
**$Z$是随机变量**，因为它依赖于随机试验的结果$\omega$。$Z$被称作$\hat{\mathbb{P}}$关于$\mathbb{P}$的拉东一尼柯迪姆( Radon- Nikodim)导数。这个定义和[高等概率论](/blog/2024/10/09/advanced-probability-notes/) 的“Radon-Nikodym 定理”一节是一样的，是一种更加狭义的表达。

尽管在有限样本空间$\Omega$的情形，它其实是商而不是导数。随机变量$Z$具有三个重要性质，我们将其表述为下面的定理。

定理：设$\mathbb{P}$ 和$\tilde{P}$ 是有限样本空间 $\Omega$ 上的两个概率测度，对任意的
$\omega\in\Omega,\mathbb{P}\left(\omega\right)>0,\tilde{\mathbb{P}}\left(\omega\right)>0$,定义随机变量$Z$如上。我们有：
* $\mathbb{P}(Z>0)=1;$
* $EZ=1$,
* 对任意的随机变量$Y$,有：$\tilde{\mathbb{E}}Y=\mathbb{E}\begin{bmatrix}ZY\end{bmatrix}$

在回想我们的问题出在哪？ **全程使用风险中性概率测度进行计算的定价公式完全没有考虑真实的事件发生的概率，这一定是有问题的。** 想要解决这个问题，就要用两个测度之间的RN导数来帮助我们加权，这就是测度变换这一节的意义。

定义：考虑 $N$ 时段二叉树模型，设真实概率测度为 $P$，风险中性概率测度为$\hat{\mathbb{P}}$。$Z$ 表示$\hat{\mathbb{P}}$关于 $P$ 的拉东—尼柯迪姆导数，即：
$$Z(\omega_1\cdots\omega_N)=\frac{\tilde{\mathbb{P}}(\omega_1\cdots\omega_N)}{\mathbb{P}(\omega_1\cdots\omega_N)}=\left(\frac{\tilde{p}}p\right)^{\#H(\omega_1\cdots\omega_N)}\left(\frac{\tilde{q}}q\right)^{\#T(\omega_1\cdots\omega_N)}$$
序列$\omega_1\cdots\omega_N$中出现背面的次数。定义**状态价格密度随机变量**为：
$$\zeta(\omega)=\frac{Z(\omega)}{(1+r)^N}$$
并且称$\zeta(\omega)\mathbb{P}(\omega)$为相应于$\omega$的**状态价格**。

回顾第二章中的风险中性定价公式在时刻 $N$ 支付为$V_{\mathrm{N}}$ 的任何衍生证券在时刻 0 的价格为$V_0=\widetilde{\mathbb{E}}\frac{V_N}{(1+r)^N}$。利用状态价格密度，可以简单地将其重写为：
$$V_0=\mathbb{E}\begin{bmatrix}\zeta V_N\end{bmatrix}=\sum_{\omega\in\Omega}V_N(\omega)\zeta(\omega)\mathbb{P}(\omega)$$

### RN导数过程
在上一节，我们考虑了$N$时段二叉树模型中风险中性概率测度关于真实概率测度的拉东一尼柯迪姆导数。这个随机变量$Z$依赖于模型中的 $N$ 次抛掷硬币结果。

为了得到依赖于较少次抛掷硬币结果的相应的随机变量， 我们可以基于时刻$n<\mathbb{N}$的信息，对$Z$进行估计（就是条件期望）。这种估计在其他场合也会出现，因此以下我们给出一个一般的结论，其中并不要求$Z$是拉东—尼柯迪姆导数。

定理：设 $Z$ 是 $N$ 时段二叉树模型中的一个随机变量，定义：
$$Z_n=\mathbb{E}_nZ,\quad n=0,1,\cdotp\cdotp\cdotp,N$$
则$Z_n,n=0,1,\cdotp\cdotp\cdotp,N$在真实概率测度$\mathbb{P}$下是一个鞅。（风险中性概率测度也如此）

仍旧坚持原本符号，定义RN导数过程
$$Z_n=\mathbb{E}_n[Z],\quad n=0,1,\cdotp\cdotp\cdotp,N$$
自然的有$Z_{N}=z,z_0=1$

那么前面的衍生证卷在$n$价格修正为
$$V_n=\tilde{\mathbb{E}}_n\frac{V_N}{(1+r)^{N-n}}=\frac{(1+r)^n}{Z_n}\mathbb{E}_n\frac{Z_NV_N}{(1+r)^N}=\frac1{\zeta_n}\mathbb{E}_n\begin{bmatrix}\zeta_NV_N\end{bmatrix}$$
其中状态密度价格为过程为
$$\zeta_n=\frac{Z_n}{(1+r)^n},\quad n=0,1,\cdotp\cdotp\cdotp,N$$

## 计价单位变换
资产的价格一般使用某国的货币进行衡量，处于方便金融投资或者简化模型的目的，我们经常需要变换计价单位，这会导致一些问题的产生，值得被讨论。

事实上，当计价单位发生变换的时候，**风险中性概率测度必须变换，否则大量过程的鞅性将失去**  RN导数公式为
$$\frac{D(t)N(t)}{N(0)}=\widetilde{\mathbb{E}}\left[\frac{D(T)N(T)}{N(0)}|\mathcal{F}(t)\right],0\leqslant t\leqslant T$$

其中$D(t)$是贴现过程  $N(t)$是某一资产的价格过程

## 依赖利率的资产
本章，我们研究价值依赖利率的资产，我们称为固定收益资产，其中最为典型的固定收益资产是零息债卷。

我们定义零息债卷的**收益率**为
$$\text{零息债券价格}=\text{面值}\times e^{-\text{收益单}\times\text{存续期}}$$
对于每一个到期日，我们都有零息债卷的存在。将不同到期日的零息债卷的收益率绘制称曲线，我们称为**收益率曲线**

这意味着大量的可交易资产存在，避免其中的套利就是我们要研究的。研究这一类问题的模型称为**期限结构模型**

经典的依赖利率的模型包括 **利率二叉树模型** **远期合约与期货模型** **多因子的仿射收益模型** **HJM模型** **远期LIMBOR模型** 这里省略其介绍。
