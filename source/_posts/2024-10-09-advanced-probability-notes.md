---
title: "高等概率论：概率空间、随机变量与测度基础"
title_en: "Advanced Probability: Probability Spaces, Random Variables, and Measure-Theoretic Foundations"
date: 2024-10-09 21:29:48 +0800
categories: ["Data Science & Statistics", "Probability & Statistical Foundations"]
tags: ["Learning Notes", "Statistics", "Probability", "Measure Theory"]
author: Hyacehila
excerpt: "整理概率空间、随机变量、测度基础、收敛性、独立性、条件期望和相关概率理论。"
excerpt_en: "Covers probability spaces, random variables, measure-theoretic foundations, convergence, independence, conditional expectation, and related theory."
mathjax: true
hidden: true
permalink: '/blog/2024/10/09/advanced-probability-notes/'
---
## 概率空间与随机变量
高等概率论研究公理化的概率体系，这也是概率走向现代的重要一步，概率体系的基本发展情况如下

1. 概率的古典定义（两大概型）
2. 集合与测度理论的引入
3. 公理化结构的建立

这一章补充完整概率理论中的基础数学知识，以及建立我们基本的概率体系，将会包括一些基础的数学知识，以及对概率论中最基础的概念，概率空间和随机变量的精确数学定义

### $\sigma$代数
为了引入公理化的概率理论，我们首先补充关于代数的知识

定义：集类也称集合类，是集合构成的集合

定义：在空间全$\Omega$上对交运算封闭的集合类称为 $\pi$ 类

定义：如果集类$\mathcal{A}$ 满足
* $\Omega \in \mathcal{A}$ ($\phi \in \mathcal{A}$)
* $\text{若}A\in \mathcal{A}\text{则}A^{c}\in \mathcal{A}$
* $\text{若}A_1,A_2,\ldots,A_n\in \mathcal{A},\text{则}\sum_{i=1}^nA_i\in \mathcal{A}$
则称 集类$\mathcal{A}$  是一个代数

从定义能看出 ： **代数是 $\pi$ 类，而 $\pi$ 类不一定是代数**

定义：设$F$是集类 且满足
* $\Omega \in F$ ($\phi \in F$)
* $\text{若}A\in F\text{则}A^{c}\in F$
* $\text{若}A_1,A_2,\ldots\in F,\text{则}\sum_{i=1}^{\infty} A_i\in \mathcal{A}$
则称$F$ 是一个$\sigma$代数

**我们把定义从有限并封闭改为了可列并封闭，因此$\sigma$代数一定是一个代数**

我们可以给出$\sigma$代数的两个性质，他们可以结合定义进行证明
* $\sigma$代数的交是$\sigma$代数
* $\sigma$代数的并不一定是$\sigma$代数代数

定义：设$\mathcal{A}$是$\Omega$的一个子集，$F$是$\sigma$代数 如果
* $\mathcal{A}\in\mathcal{F}$
* $任意包含A的\sigma 代数F^{\prime},均有F\subset F^{\prime}$
则称$F$是$A$生成的$\sigma$代数 记作$F=\sigma(A)$ 

对于理解生成$\sigma$代数的概念，我们可以给出两条性质
* $\sigma(A)$是包含 $A$ 的最小$\sigma$代数
* $\sigma(A)$是所有包含 $A$ 的$\sigma$代数的交

由全体闭区间生成的$\sigma$代数称为Borel代数，他将是我们后面研究随机变量的时候最常用的$\sigma$代数
### $\pi - \lambda$ 类定理
定义：称集类$F$是$\lambda$类 如果有
* $\Omega\in F$
* $对任意A,B\in F 且A\subset B,则 B/A\in F$
* $A_{1}A_{1},A_{2},\ldots\in F A_{n}\uparrow A=\cup_{i=1}^{\infty}A_{i},则A\in F$ 


明显的
* $\sigma$代数是$\lambda$类
* $\lambda$类对补运算封闭

定理：我们容易给出定理
$$集类F是\sigma代数\Longleftrightarrow F是\pi类且是\lambda类$$

给出本节最重要的 $\pi - \lambda$ 类定理

如果有两个集合类$P$和$L$,其中$P$是一个$\pi$系统(即在有限交运算下封闭的集合族),$L$是一个$\lambda$-系统(包含空集，对补集和可数不相交并运算封闭的集合族),并且$P$ 是$L$ 的子集，那么由$P$ 生成的$\sigma$-代数$\sigma(P)$是$L$的子集。

### 概率测度
从本节开始我们开始研究公理化的概率理论，首先我们来研究概率测度和概率空间

定义：集函数是从集合到实数的映射

$P$  是定义在  $\sigma$代数 $\mathcal{F}$  上的概率测度, 如果它是一个集函数  $P: \mathcal{F} \rightarrow[0,1]$ , 并满足以下三个公理：
1. 非负性：对于任意事件  $A \in \mathcal{F}$  ，有  $P(A) \geq 0$  
2. 单位性：整个样本空间的概率为 1 , 即  $P(\Omega)=1$  
3. 可列加性：对于任意可数个两两不相交的事件  $A_{1}, A_{2}, \ldots \in \mathcal{F} , 有  P\left(\bigcup_{i=1}^{\infty} A_{i}\right)=   \sum_{i=1}^{\infty} P\left(A_{i}\right)$

概率测度是一种特殊的测度，是我们在概率论中主要研究的对象，测度这个概念最早在[实变函数](/blog/2023/03/18/real-analysis-notes/)中进行了介绍

自然的，我们可以给出一些概率测度的性质
* 单调性：$A \subset B$ 则 $P(A) \le P(B)$
* 补集规则 ： $P(A^{c}) = 1-P(A)$
* 有限可加性 ：任意事件$A_{1}, A_{2}, \ldots \in \mathcal{F}$  有 $P\left(\bigcup_{i=1}^{\infty} A_{i}\right) \le\sum_{i=1}^{\infty} P\left(A_{i}\right)$
* 连续性：$如果  A_{1} \subseteq A_{2} \subseteq \ldots  是一个递增的事件序列, 且  \bigcup_{i=1}^{\infty} A_{i}=A , 则  \lim _{n \rightarrow \infty} P\left(A_{n}\right)=P(A)$递减情况同理
* 容斥原理：$P(A\cup B) = P(A)+P(B)-P(AB)$

在明确了概率测度的概念之后，我们可以给出概率空间的准确定义：

一个概率空间由三元组 $(\Omega,F,P)$ 组成，其中：
* 样本空间$\Omega$ 非空并且包含了所有的实验结果（样本点）
* 事件域$F$是样本空间$\Omega$ 上的一个$\sigma$代数
* 概率测度$P$满足我们刚才给出的定义，是一个$\sigma$代数的集函数

它们为随机事件的分析和概率的计算提供了必要的结构和规则，也就是样本全体，可测的事件域以及事件的概率

现在我们来介绍概率空间的完备化理论，他是后面的一些问题的基础

定义：在概率空间 $(\Omega, \mathcal{F}, P)$ 中，如果有$A \in \mathcal{F}$ 并且 $P(A) = 0$ 则称$A$为一个零事件，我们一般用$\mathcal{N}$表示概率空间上零事件的全体

特别的，并不是所有的概率空间上，**零事件的子集都是零事件**，但是我们把满足这一条性质的概率空间称为**完备的概率空间**（和泛函分析中的完备性不是一个概念）

尽管任意概率空间不一定完备，但是一定可以把他们延拓成完备的概率空间，思路如下

定理：设存在概率空间 $(\Omega, \mathcal{F}, P)$ ，则一定存在一个完备的概率空间 $(\Omega, \overline{\mathcal{F}}, \overline{P})$ 满足 $\mathcal{F} \subset \overline{\mathcal{F}},P = \overline{P}$ 


### 随机变量
有了前面这么多的讨论基础，我们可以开始讨论随机变量这一个概率论中最重要的概念了，这里我们给出他的公理化解释

定义 设$\left(\Omega,\mathcal{F},\mathbb{P}\right)$为一个概率空间和$(S,\mathcal{S})$是任意可测空间(即$\mathcal{S}$为$S$ 中某些子集所形成的 $\sigma$-代数),称定义在样本空间上的函数 $X(\omega)$: $\Omega\to S$是一个$(\Omega,\mathcal{F},\mathbb{P})$上的$S$-值随机变量 (r.v.),如果对任意 $B\in S$ 有
$$X^{-1}\left(B\right):=\left\{\omega\in\Omega;X\left(\omega\right)\in B\right\}\in\mathcal{F}.$$

从定义可以自然的看出 **事件域的大小决定了可测函数是否是随机变量** 

我们在[初等概率论](/blog/2023/03/18/elementary-probability-notes/)中一般研究实值的随机变量，此时就是将$(S,\mathcal{S})$ 换成 $(R^d,B_{R^d})$  $d = 1$是随机变量 否则为随机向量

我们可以将原本的问题反过来，给出下面的定理
$$\begin{aligned}&\text{设}X:\Omega\to S\quad\text{且}(S,\mathcal{S})\quad\text{是可测空间}.\quad\text{则}J=\{X^{-1}(B)|B\in S\}\text{是}\sigma \text{代数}.\end{aligned}$$
因此，我们记$\sigma(X) = \{X^{-1}(B)|B\in S\}$ 为映射$X$生成的$\sigma$代数，他也是让样本空间上的函数$X$构成随机变量的最小事件域

下面我们简单刻画一下多维随机变量所生成的$\sigma$代数
设$X=\left(X_1,\cdots,X_n\right):\left(\Omega,\mathcal{F}\right)\to\left(\mathbb{R}^n,B_{\mathbb{R}^n}\right)$是一个$\mathbb{R}^d$值随机变量，则有
$$\sigma\left(X\right)=\sigma\left(\bigcup_{i=1}^{n}\sigma\left(X_{i}\right)\right).$$
也就是说，多维随机变量所生成的$\sigma$代数 是他各个分量的生成的$\sigma$代数的并所生成的$\sigma$代数

由于可测函数复合后可测，我们可以容易的给出下面定理 
设$f:\left(\mathbb{R}^n,B_{\mathbb{R}^n}\right)\to (R,B_{R})$可测  $X_i$是$\left(\Omega,\mathcal{F},\mathbb{P}\right)$的随机变量  则有$f(X_1,X_2...X_n)$ 是 $\left(\Omega,\mathcal{F},\mathbb{P}\right) \to (R,B_R)$ 上的随机变量

其证明思路非常简单，仅仅是因为$f$是两层可测函数的复合，因此他可测，所以构成随机变量


## 分布与积分
### 随机变量的分布
随机变量是从概率空间这个**三元对**到可测空间这个**二元对**上的可测映射，从三元映射到二元是一件很奇怪的事情，实际上这是因为我们在前面少介绍了一个概念，他就是**随机变量的分布**，他将把二元对补充为三元对

定义  设  $X:(\Omega, \mathcal{F}) \rightarrow(S, \mathcal{S})$ 为概率空间  $(\Omega, \mathcal{F}, \mathbb{P})$ 上的一个随机变量. 对任意  $B \in \mathcal{S}$  ，定义：
$$\mathcal{P}_{X}(B):=\mathbb{P}\left(X^{-1}(B)\right)$$
则称  $\mathcal{P}_{X}$ 为随机变量 $X$ 的分布

**这里定义的是随机变量的分布而不是分布函数**

我们很容易验证 $(S, \mathcal{S},\mathbb{P}_X)$ 也是一个概率空间，也就是**我们从原本的事件概率空间，来到了更加抽象的分布概率空间**，从而简化一些研究

如果  $(S, \mathcal{S})=\left(\mathbb{R}, \mathcal{B}_{\mathbb{R}}\right)$ , 则称如下函数:
$$F_{X}(x):=\mathcal{P}_{X}((-\infty, x]) \quad x \in \mathbb{R}$$
为实值随机变量  $X$  的分布函数 ^ac85f9

现在我们可以明确了，所谓的分布函数只是使用随机变量的分布构造出的一种函数形式，也就是分布函数是从随机变量的分布诱导的。

这里我们终于对分布做出了一个准确定义，在初等概率论中我们只研究了分布函数，对分布仅有一个只进行了感觉上的定义。

下面我们来给出随机变量同分布的定义

定义：设$X,Y$是取值于同一个空间$S$的$S$值随机变量，并且有$\mathcal{P}_{X}=\mathcal{P}_{Y}$ 也就是任意$B\in S$ 有 $\mathcal{P}_{X}(B)=\mathcal{P}_{Y}(B)$  则称 $X$ 与 $Y$ 同分布 写作 $X \overset{d}{=} Y$

特别的 **同分布可以不定义在同一概率空间上**，分布是比概率空间更加抽象的概念，只研究随机变量的取值空间而不研究原本的概率空间

下面我们再补充一个定理，研究随机变量相等的证明

定理：设$X,Y$是取值于同一个空间$S$的随机变量，满足$\mathcal{S} = \sigma(\mathcal{A})$ 并且  $\mathcal{A}$ 是一个$\pi$ 类 如果对任意$B\in \mathcal{A}$ 有 $\mathcal{P}_{X}(B)=\mathcal{P}_{Y}(B)$  则  $X \overset{d}{=} Y$

推论：取$(S, \mathcal{S})=\left(\mathbb{R}, \mathcal{B}_{\mathbb{R}}\right)$   $\mathcal{A} = \{(-\infty,x]|x\in R\}$ 并且$\sigma(A) = B_R$  则根据上面的定理有 任意$x\in R$ 有 $F_X(x)= F_Y(x)$    则  $X \overset{d}{=} Y$

继续推论：几乎处处相等的分布函数也导出相同的分布

**随机变量的这样的抽象过程是可能导致信息压缩的，与此同时这也意味着问题的简化，合适的设置随机变量是重要的**

### 分布函数的性质
前面我们给出了随机变量分布函数的定义

参见本文“相关段落”一节。

实际上，结合在初等概率论中对随机变量的介绍，我们容易给出他的三条性质
* $F(-\infty) = 0,F(\infty) = 1$
* $F(x)$ 是单调增函数
* $F(x)$ 在每个点都右连续

事实上，我们也可以给出这个问题的另一个角度：所有满足前述三个性质的函数都是分布函数，都可以反向诱导出分布

### 分布函数的分类与分解

#### 离散部分与连续部分
前面给予了足够的证明，保证了一个分布函数的间断点至多可列，我们不妨记为$\{a_n\}$ 据此，我们可以定义跃度 对于任意的$n$ 有

$$b_{n}=\Delta F\left(a_{n}\right)=F\left(a_{n}\right)-F\left(a_{n}-\right)$$
其中$b_n$称为 跃度 其中 $F\left(a_{n}-\right)$ 是因为函数的右连续性质而给出的

进一步的，我们可以定义
$$F_{d}\left(x\right):=\sum_{n\in\mathbb{Z}}b_{n}l_{\left[a_{n},\infty\right)}\left(x\right),x\in\mathbb{R}$$
其中$l$是示性部分，函数的含义是跃度的和，我们将$F_d(x)$称为分布函数的离散部分

容易验证，$F_d(x)$满足如下的性质
* $F_d(-\infty) = 0,F_d(\infty) \le  1$
* $F_d(x)$ 是单调增函数
* $F_d(x)$ 在每个点都右连续

如果 $\sum_{n\in\mathbb{Z}}b_{n} = 1$  则$F_d(x)$也是一个分布函数($F_d(\infty) =  1$)，此时我们称为离散型分布函数 

特别的，我们可以给出分布函数的连续部分的的定义有
$$F_{c(x)}= F(x) - F_d(x)$$
容易验证，他也满足
* $F_c(-\infty) = 0,F_c(\infty) \le  1$
* $F_c(x)$ 是单调增函数
* $F_c(x)$ 在每个点都右连续

同样的，如果他也构成分布函数，我们称为连续性分布函数
#### Jordan分解
定理 (**分布函数的 Jordan 分解**) 设$F(x),x\in\mathbb{R}$是任意一个分布函数，则存在且唯一存在$\alpha\in[0,1]$使得
$$F\left(x\right)=\alpha F_{1}\left(x\right)+\left(1-\alpha\right)F_{2}\left(x\right),x\in\mathbb{R},$$
其中$F_1$为离散型分布函数和$F_2$为连续型分布函数
#### 绝对连续与奇异连续
我们已经有连续和离散的概念了，这足够了吗？ 这里我们需要结合初等概率论的思想来思考

我们前面介绍的 离散部分对应了初等概率论中的离散型分布，那连续部分呢？ 

如果我们想让他有对应的连续型分布，那需要找到对应的密度函数，而仅仅满足前面的性质是不能保证这一点的，因此我们定义绝对连续型分布函数

定义(绝对连续型分布函数) 设$F(x),x\in\mathbb{R}$为一个分布函数。如果 $F$还是绝对连续的$(AC)$,即对任意$-\infty<x_1<y_1<x_2<y_2<\cdots<x_m<y_m<+\infty$ 和任意$\varepsilon>0,存在\delta>0使得$
$$\sum_{i=1}^{m}\left|y_{i}-x_{i}\right|<\delta\Longrightarrow\sum_{i=1}^{m}\left|F\left(y_{i}\right)-F\left(x_{i}\right)\right|<\varepsilon,$$
则称$F$是一个绝对连续型分布函数

根据绝对连续函数的性质，我们还有：如果$F$是一个绝对连续分布函数，则
存在一个非负函数$f\in L^1(\mathbb{R})$使得，对任意$x_1<x_2$满足
$$F\left(x_{2}\right)-F\left(x_{1}\right)=\int_{x_{1}}^{x_{2}}p\left(x\right)dx,$$
其中$\|f\|_{L^1}:=\int_{\mathbb{P}}|p(x)|$d$x=1.$于是$F^\prime=p\geqslant0$, a.e. (因为分布函数$F$是单增的),以后我们称$x\to p(x)$为绝对连续型分布函数$F$的概率密度函数.

我们可以提出对等的概念，也就是奇异型分布函数

定义：如果$F(x)$是一个分布函数 并且$F^{'}=0$ a.e. 则称$F$是奇异型分布函数，进一步的，如果$F$是连续的，则称为连续奇异型分布函数

任何离散型分布函数都是奇异型分布函数，连续奇异型分布函数需要使用Contor三分集构造
#### Lebesgue分解
设$F$是任意一个分布函数，那么存在$\alpha,\beta \in [0,1]$ 使得 
$$F=\alpha F_{1}+\beta(1-\alpha)F_{2}+(1-\alpha)(1-\beta)F_{3}$$
其中$F_1$是离散部分 也可以写作$F_d$ $F_2$是绝对连续部分 也写作$F_{a.c}$  $F_3$是奇异连续部分 也写作$F_{c.s.}$ 

这样的分解一定存在且唯一
### 积分（数学期望）
这一节我们来讨论随机变量$X$的积分，也就是随机变量的数学期望，我们研究期望的定义，性质，以及一些推广定理

#### 积分的定义
我们在随机变量上引入了概率测度的概念，非常自然，我们在公理化体系中应该考虑Lebesgue积分，而所谓的**随机变量恰好就是一个可测函数**，因此他的Lebesgue积分的定义也就非常自然了，我们分下面四步来介绍

关于随机变量与可测函数：随机变量的每个取值（或者是$B_{R}$） 根据相关的定理，最能找到对应的一个$\Omega$上的子集$A$与他对应，对应的概率测度$P(A)$就是随机变量的可测函数的取值对应的测度

有了这些叙述，我们就能模仿经典的四步定义给出随机变量的Lebesgue积分的定义了

STEP1
对于示性随机变量 
$$I_{A}(w)=\begin{cases}1,w\in A\\0,w \notin A\end{cases}$$
他的期望有
$$E(I_{A})= 1 \times P(A)+0 \times P(A^c)$$

STEP2
对于非负简单随机变量 存在$\Omega$的不相交划分$A_i$ 和对应权重 $b_i$ 随机变量满足
$$X\left(\omega\right)=\sum_{i=1}^{n}b_{i}I_{A_{i}}\left(\omega\right)$$
我们可以自然定义$X$关于$P$的积分
$$\mathbb{E}\left[X\right]=\int_{\Omega}X\left(\omega\right)\mathbb{P}\left(d\omega\right):=\sum_{i=1}^{n}b_{i}\mathbb{P}\left(A_{i}\right).$$

STEP3
对于任何非负随机变量$X$，我们能找到一列 单增 非负 简单 随机变量列 $X^{m}$ 满足
$$0\leqslant\left|X\left(\omega\right)-X^{\left(m\right)}\left(\omega\right)\right|\leqslant2^{-m},\forall\omega\in\Omega.$$
这样我们就可以用极限来定义积分 有
$$\mathbb{E}[X]:=\sup\left\{\mathbb{E}[\xi];\xi\text{ 是非负简单随机变量且 }\xi\leqslant X\right\}\in[0,+\infty].$$
如果$E[X] \to \infty$ 则称为积分不存在

STEP4
对于任何实值随机变量 拆分为 $X = X^{+}+X^{-}$ 定义积分有
$$\mathbb{E}[X]=\int_{\Omega}X\left(\omega\right)\mathbb{P}\left(d\omega\right):=\mathbb{E}\left[X^{+}\right]-\mathbb{E}\left[X^{-}\right].$$

综上，我们的积分，或者说数学期望的定义就结束了，一般情况下我们会用下面的符号来表示
$$\begin{aligned}
E[X]& =\int_{\Omega}X(\omega)P(d\omega)=\int_{\Omega}X(\omega)dP(\omega) \\
&=\int_{x}XdP
\end{aligned}$$

#### 期望的性质
1. 线性可加性$\mathbb{E}[aX+bY]=a\mathbb{E}[X]+b\mathbb{E}[Y]$
2. 设$X$为可积随机变量且$X\geqslant0$,a.e.(即$\mathbb{P}(X\geqslant0)=1)$,则$\mathbb{E}[X]\geqslant0.$特别地，如果$X=0$, a.e.,则$\mathbb{E}[X]=0.$
3. 设$X,Y$为两个可积随机变量和$X\leqslant Y$, a.e.,则$\mathbb{E}[X]\leqslant\mathbb{E}[Y]$
4. $\text{设 }X,Y\text{为两个可积随机变量，那么}|X+Y|\text{也是可积的，以及}$ $\mathbb{E}[|X+Y|]\leqslant\mathbb{E}[|X|]+\mathbb{E}[|Y|].$
5. 设随机变量$X$可积 则$|\mathbb{E}[X]|\leq\mathbb{E}[|X|]$ 
6. 设随机变量$X$可积 和 $A\in F$ 如果存在常数 $a \le X(w) \le b$ 任意点都成立 那么$a\mathbb{P}\left(A\right)\leqslant\mathbb{E}\left[X11_{A}\right]\leqslant b\mathbb{P}\left(A\right).$
#### 期望的重要定理
本节我们研究随机变量序列的收敛问题，本节我们讨论的收敛属于 [初等概率论](/blog/2023/03/18/elementary-probability-notes/) 的“收敛性 / 几乎处处收敛，概率1收敛”一节 

**(单调收敛定理)** 设$\{X_n\}_{n\geqslant1}$为概率空间$(\Omega,\mathcal{F},\mathbb{P})$非负、单增和可积随机变量，那么有
$$E\left[\lim_{n\to\infty}X_{n}\right]=\lim_{n\to\infty}E[X_{n}].$$
 **(Fatou 引理)** 设$\{X_n\}_{n\geqslant1}$是概率空间$(\Omega,\mathcal{F},\mathbb{P})$上的一列非负可积随机变量，则
$$\mathbb{E}\left[\lim_{n\to\infty}X_{n}\right]\leqslant\lim_{n\to\infty}\mathbb{E}\left[X_{n}\right].$$

**(控制收敛定理)** 设$\{X_n\}_{n\geqslant1}$是概率空间$(\Omega,\mathcal{F},\mathbb{P})$上的一列随机变量且满足
$$\left|X_{n}\right|\leqslant Y,\forall n\geqslant1,$$
其中$Y$是一个独立于$n$ 的非负可积随机变量.进一步，如果
$$\mathbb{P}\left(\lim_{n\to\infty}X_{n}=X\right)=1\left(\text{我们记为 }X_{n}\xrightarrow{\mathrm{a.e.}}X,n\to\infty\right),$$
那么有
$$\mathbb{E}\left[\lim_{n\to\infty}X_{n}\right]=\lim_{n\to\infty}\mathbb{E}[X_{n}].$$

 **(有界收敛定理)** 设  $\left\{X_{n}\right\}_{n \geqslant 1}$  和  $X$  是概率空间  $(\Omega, \mathcal{F}, \mathbb{P})$  上的一列有界随机变量且满足  $X_{n} \xrightarrow{\text { a.e. }} X, n \rightarrow \infty$, 则
$$\lim _{n \rightarrow \infty} \mathbb{E}\left[X_{n}\right]=\mathbb{E}\left[\lim _{n \rightarrow \infty} X_{n}\right] .$$


我们给出两个非常自然的推论
* 设$X$是非负可积实值随机变量，则$P(A)=0$ 推出 $E[X;A]=0$
* 设$X$是处处正值可积实值随机变量，则$E[X;A]=0$ 推出 $P(A)=0$ 
他们都是非常自然的性质

#### 概率不等式
**Hölder不等式** 设$1<p,q<+\infty$ 和 $p^{-1}+q^{-1}=1$,则有
$$|\mathbb{E}[XY]|\leqslant\mathbb{E}[|XY|]\leqslant\{\mathbb{E}[|X|^{p}]\}^{p^{-1}}\{\mathbb{E}[|Y|^{q}]\}^{q^{-1}}.$$

**Minkovski 不等式** 对任意 $p>0$,则有
$$\left\{\mathbb{E}[|X+Y|^{p}]\right\}^{p^{-1}}\leqslant\left\{\mathbb{E}[|X|^{p}]\right\}^{p^{-1}}+\left\{\mathbb{E}[|Y|^{p}]\right\}^{p^{-1}}.$$

 **Lyapunov 不等式** 对任意  $1<p<q<+\infty$ , 则有
$$\left\{\mathbb{E}\left[|X|^{p}\right]\right\}^{p^{-1}} \leqslant\left\{\mathbb{E}\left[|X|^{q}\right]\right\}^{q^{-1}} .$$


**Jensen 不等式** 设  $\varphi(x): \mathbb{R}^{d} \rightarrow \mathbb{R}$  为一个凸函数, 则有
$$\varphi(\mathbb{E}[X]) \leqslant \mathbb{E}[\varphi(X)]$$

**Cr 不等式** 对于任意$p > 0$ 有
$$|X_{1}+X_{2}+...X_{n}|^{p}\le Cr(|X_{1}|^{p}+|X_{2}|^{p}+...|X_{n}|^{p})$$
其中 $Cr = 1 ~if~ p\le 1~else~Cr = n^{p-1}$   

#### 变量变换公式（积分的计算）
欧式空间值的随机变量关于概率测度的积分不容易计算，这里介绍定理将这样的概率测度积分转换为Riemann - Stieltjes 积分来方便计算，变量变换公式有其更加广义的形式，但这里我们仅仅介绍关于概率测度的类型

**定理  (变量变换公式)** 设  $X:(\Omega, \mathcal{F}) \rightarrow(S, \mathcal{S})$  为概率空间  $(\Omega, \mathcal{F}, \mathbb{P})$  上的一个随机变量和  $h$  为定义在  $S$  上的一个实值可测函数, 使得  $h(X)$  是可积的,那么
$$\mathbb{E}[h(X)]=\int_{\Omega} h(X(\omega)) \mathbb{P}(\mathrm{d} \omega)=\int_{S} h(x) \mathcal{P}_{X}(\mathrm{~d} x)$$
如果  $S=\mathbb{R}^{d}$  ，则有
$$\mathbb{E}[h(X)]=\int_{\Omega} h(X(\omega)) \mathbb{P}({d} \omega)=\int_{S} h(x) \mathcal{P}_{X}({~d} x)=\int_{\mathbb{R}^{d}} h(x) F({d} x),$$
其中  $\mathcal{P}_{X}$  表示随机变量  $X$  的分布，而当  $S=\mathbb{R}^{d}$  时，  $F(x), x \in \mathbb{R}$  则表示其分布函数.

经由这个公式，我们把可以把测度积分转换为RS积分计算

如果$F$是绝对连续的，那么最后的$F(dx)$就是$d(F(x))$  原本的问题可以转换为Riemann积分进行计算

如果他是离散的，则
$$F\left(x\right):=\sum_{n\in\mathbb{Z}}b_{n}l_{\left[a_{n},\infty\right)}\left(x\right),x\in\mathbb{R}$$
原始的问题就转换为一个离散型分布的函数的期望问题，我们在[初等概率论](/blog/2023/03/18/elementary-probability-notes/) 的“随机变量的函数的数学期望”一节 积分变求和，仍旧可以计算。

如果存在奇异连续的情况，不属于目前可以解决的范畴，放弃计算。

如果原始分布函数情况过于复杂，使用本文“Lebesgue分解”一节来分解分布函数，分别进行计算。

关于Riemann - Stieltjes 积分的定义，参考[金融随机分析](/blog/2024/10/17/financial-stochastic-analysis-notes/) 的“Riemann - Stieltjes 积分”一节

### 独立性
我们在[初等概率论](/blog/2023/03/18/elementary-probability-notes/)中研究了事件的独立性和随机变量的独立性，但是他们的定义总是让我们感觉少了数学的严谨，这里就是为了弥合这个缺点。
#### 两个事件的独立性
我们参考初等概率论的介绍 参见[初等概率论](/blog/2023/03/18/elementary-probability-notes/) 的“两个事件的独立性”一节。
#### 两个$\sigma$ 代数的独立性
定义（平凡$\sigma$代数 ）：以后称$\mathcal{H}$是一个平凡 $\sigma$代数，如果
$$\mathbb{P}\left(H\right)=0或1,\forall H\in H.$$

定义（$\sigma$ 代数的独立性）：设$(Ω,\mathcal{F},\mathbb{P})$是一个概率空间.假设$\mathcal{H},\mathcal{G}\subset\mathcal{F}$ 为两个$\sigma$代数 .如果对任意$H\in\mathcal{H}$和$G\in\mathcal{G}$,我们有
$$\mathbb{P}(G\cap H)=\mathbb{P}(G)\mathbb{P}(H),$$
则称$\sigma$ 代数$\mathcal{H},\mathcal{G}$是相互独立的

容易看出：平凡$\sigma$代数和其他这个空间上的$\sigma$代数都是独立的，我们在后面的证明里会用到这一点
#### 两个随机变量的独立性
定义 (随机变量的独立性)设$X,Y$为概率空间$(\Omega,\mathcal{F},\mathbb{P})$上的两个随机变量。如果$\sigma\left(X\right)$与$\sigma\left(Y\right)$是相互独立的，那么称$X$与$Y$是相互独立的。

这个定义是非常自然的，我们用随机变量生成的$\sigma$代数的独立性来研究随机变量的独立性。
#### 多个独立性
类比两个随机变量的独立性的研究，我们可以自然的给出下面的推理
参见[初等概率论](/blog/2023/03/18/elementary-probability-notes/) 的“多个事件的独立性”一节。

$\sigma$代数是一个集类，因此我们可以先根据事件独立性的定义引出集类独立性的定义有：

定义(一族集类的独立性)：设$(\Omega,\mathcal{F},\mathbb{P})$是一个概率空间和$\{\mathcal{A}_\alpha\}_{\alpha\in I}\subset\mathcal{F}$为一族集类 (其中$I$可以不可数).如果对任意的正整数$L>0$和互不相同的$\alpha_{1},\cdots,\alpha_{L}\in I$,
$$\mathbb{P}\left(\bigcap_{k=1}^{L}A_{k}\right)=\prod_{k=1}^{L}\mathbb{P}\left(A_{k}\right),\forall A_{k}\in\mathcal{A}_{\alpha_{k}},k=1,\cdots,L,$$
则称$\{\mathcal{A}_\alpha\}_{\alpha\in I}$是相互独立的。

定义($\sigma$代数的独立性)：我们允许将$n$个$\sigma$代数独立性的定义优化为
$$\mathbb{P}\left(\bigcap_{k=1}^{n}A_{k}\right)=\prod_{k=1}^{n}\mathbb{P}\left(A_{k}\right)$$
也就是不再要求任意组合独立，整体独立就是相互独立。

定义：一族随机变量$\{X_\alpha\}_{\alpha\in I}$是相互独立的如果$\left\{\sigma\left(X_{\alpha}\right)\right\}_{\alpha\in I}$是相互独立的
#### 初等概率论与高等概率论中独立性的差异
我们在初等概率论中研究随机变量的独立性的内容为
参见[初等概率论](/blog/2023/03/18/elementary-probability-notes/) 的“随机变量独立性”一节。

这和高等概率论中独立性的差别在哪，等价吗？

高等概率论允许非实值的随机变量，分布函数只允许实值随机变量。从这点看定义是不等价的。

哪怕我们将高等概率论中的随机变量限制到实数范围，定义在表面上还不是等价的
* 初等概率中研究分布函数，只要求$[-\infty,x]$范围上原像的独立
* 高等概率论在限制到实数取值后，研究$B_R$ 范围明显比$[-\infty,x]$大
**但是，我们不假证明的给出结论，这两个定义本质上是等价了，可以使用$\pi-\lambda$定义实现证明**
### 乘积测度
定义 (乘积$\sigma$-代数)：设$F_1和F_2$是两个$\sigma$-代数。定义如下矩阵集
$$R:=\left\{A\times B;A\in\mathcal{F}_{1},B\in\mathcal{F}_{2}\right\},$$
那 么 称 $\mathcal{F} _1\otimes \mathcal{F} _2: = \sigma \left ( \mathcal{R} \right )$为$\sigma$-代数$\mathcal F_1,\mathcal{F}_2$的乘积$\sigma$-代数.进一步，我们称$R$中
的集合为可测矩形。

定义（乘积可测空间与截口）：设$(\Omega_k,\mathcal{F}_k),k=1,2$是两个可测空间，那么称
$$\left(\Omega,\mathcal{F}\right):=\left(\Omega_{1}\times\Omega_{2},\mathcal{F}_{1}\otimes\mathcal{F}_{2}\right)$$
为乘积可测空间.对任意$E\subset\Omega_1\times\Omega_2$和$w_i\in\Omega_i,i=1,2$,我们定义：
$$\begin{cases}E_{\omega_{1}}:=\left\{w\in\Omega_{2};\left(\omega_{1},\omega\right)\in E\right\}\subset\Omega_{2},\\E_{\omega_{2}}:=\left\{w\in\Omega_{1};\left(\omega,\omega_{2}\right)\in E\right\}\subset\Omega_{1}.\end{cases}$$
进一步，分别称$E_{\omega_1}$为$E$的$\omega_1$-截口$\left(\omega_1\text{-section)和 }E_{\omega_2}\text{ 为 }E\text{ 的 }\omega_2\text{-截口}\right.$

定理：任何$\Omega_1\times\Omega_2$中的可测集的截口都是可测的

定理：设$f:\Omega_1\times\Omega_2\to\overline{\mathbb{R}}$是一个可测函数.对于$\omega_i\in\Omega_i,i=1,2$,定义截口函数：
$$f_{\omega_{1}}(\omega_{2}):=f(\omega_{1},\omega_{2}),\quad f_{\omega_{2}}(\omega_{1}):=f(\omega_{1},\omega_{2}).$$
那么有
$\left(\mathrm{i}\right)对任意\omega_1\in\Omega_1,f_{\omega_1}:\Omega_2\to\overline{\mathbb{R}}$是可测的；
$\left(\mathrm{ii}\right)对任意\omega_{2}\in\Omega_{2},f_{\omega_{2}}:\Omega_{1}\rightarrow\overline{\mathbb{R}}$是可测的

定理：对于  $i= 1, 2, 设 \nu _{i}$为可测空间$\left(\Omega_i,\mathcal{F}_i\right)$上的$\sigma$-有限测度，那么
存在唯一的$\left(\Omega,\mathcal{F}\right):=\left(\Omega_{1}\times\Omega_{2},\mathcal{F}_{1}\otimes\mathcal{F}_{2}\right)上的\sigma-有限测度\nu,使得$
$$\nu\left(\biguplus_{k=1}^{m}A_{k}\times B_{k}\right)=\sum_{k=1}^{m}\nu_{1}\left(A_{k}\right)\nu_{2}\left(B_{k}\right),$$
其中$A_k\in\mathcal{F}_1,B_k\in\mathcal{F}_2$使得$\{A_k\times B_k\}_{k=1}^m$是互不相交的。对于这样的$v$我们称为乘积测度，$v_1,v_2$称为边际测度。

最后，我们将乘积测度和联合分布建立联系有

定理 随机变量$X_1,\cdots,X_n$是相互独立的当且仅当如下等式成立：
$$P_{\left(X_{1},\cdots,X_{n}\right)}=\prod_{i=1}^{n}P_{X_{i}},\text{在}S^{\otimes n}上.$$

## 条件期望
### 离散时间条件期望
我们在[初等概率论](/blog/2023/03/18/elementary-probability-notes/)中研究的条件期望理论实际上是残缺的，只关注计算而确实对其本质的理解，这里需要补充一定的关于条件期望的内容以方便[金融随机分析](/blog/2024/10/17/financial-stochastic-analysis-notes/)展开关于鞅的讨论。

在每个时刻$n$ 对于每个抛硬币结果序列 我们都可以对股票进行二叉树的定价，这和前面介绍的期权定价模型本质一样，形式如下
$$S_n(\omega_1\cdots\omega_n)=\frac1{1+r}[\widetilde{p}S_{n+1}(\omega_1\cdots\omega_nH)+\widetilde{q}S_{n+1}(\omega_1\cdots\omega_nT)]$$

为了简化记号，我们定义
$$\tilde{\mathbb{E}}_n[S_{n+1}](\omega_1\cdotp\cdotp\cdotp\omega_n)=\tilde{p}S_{n+1}(\omega_1\cdotp\cdotp\cdotp\omega_nH)+\tilde{q}S_{n+1}(\omega_1\cdotp\cdotp\cdotp\omega_nT)$$
这样就可以简化原本的式子为
$$S_n=\frac1{1+r}\mathbb{E}_n[S_{n+1}]$$

这里的$E[S_{n+1}]$ 称为 基于时刻$n$信息的$S_{n+1}$的条件期望。

据此我们能够给出进一步推广

定义 设$n$满足 1$\leqslant n\leqslant N$,对于给定的序列 $\omega_1\cdots\omega_n$,存在 2$^{N-n}$种可能的后续 $\omega_n+1\cdots\omega_N$。用 $\sharp H(\omega_{n+1}\cdots\omega_N)$表示在后续 $\omega_{n+1}\cdots\omega_N$ 中出现正面的次数，$\sharp T(\omega_n+1\cdots\omega_N)$表示出现背面的次数。我们定义：
$$\tilde{\mathbb{E}}_n[X](\omega_1\cdots\omega_n)=\sum_{\omega_{n+1}\cdots\omega_N}\tilde{p}^{\#H(\omega_{n+1}\cdots\omega_N)}\tilde{q}^{\#T(\omega_{n+1}\cdots\omega_N)}X(\omega_1\cdots\omega_n\omega_{n+1}\cdots\omega_N)$$
为基于时刻$n$信息的$X$的条件期望

在只有时刻0的信息的时候，条件期望也是一个随机变量，依赖于$n$次实验的结果。我们在本节条件期望符号的下角标$n$标示了条件是前几次实验的结果，至于是否已知基于问题的情况。

### 离散时间条件期望的性质
我们不假证明的给出下面的条件期望性质

设 N 为正整数，$X$ 和$Y$ 为依赖于前$N$次抛掷硬币结果的随机变量。对于给定的 0$\leqslant n\leqslant N$ ,以下性质成立

**条件期望的线性性**：对于所有常数$c_1$ 和$c_2$有
$$\mathbb{E}_n\begin{bmatrix}c_1X+c_2Y\end{bmatrix}=c_1\mathbb{E}_n\begin{bmatrix}X\end{bmatrix}+c_2\mathbb{E}_n\begin{bmatrix}Y\end{bmatrix}$$

**提取已知量**：如果$X$实际上只依赖于前$n$次硬币抛掷，那么：
$$\mathbb{E}_n[XY]=X\cdot\mathbb{E}_n[Y]$$

**累次条件期望**：（其实就是全期望公式）如果 $0\leqslant n\leqslant m\leqslant N$,那么：
$$\mathbb{E}_n[\mathbb{E}_m[X]]=\mathbb{E}_n[X]$$

**独立性**：如果$X$只依赖从第$n+1$次至第 $N$ 次抛掷硬币的结果，那么：
$$\mathbb{E}_n[X]=\mathbb{E}X$$

**条件詹森不等式**：如果$\varphi(x)$为哑变量$x$的凸函数，那么：
$$\mathbb{E}_n[\varphi(X)]\geqslant\varphi(\mathbb{E}_n[X])$$

这些性质将有利于我们后面的一些证明的展开，至于广义的条件期望的性质，参考本文“条件期望 / 条件期望的性质”一节
### 符号测度
符号测度是为了补充高等概率论中的测度基础而进行的，对于证明本文“独立性”一节中相关性质以及研究本文“条件期望的存在唯一性”一节有作用
#### 符号测度的定义
数学分析中，我们研究了不定积分和导数的问题，对于连续的$f$
$$F(x) = \int_{a}^{x}f(y)dy $$
分别称为不定积分与导数

相应的，对于测度空间上积分存在的可测函数$f$ 我们可以定义集函数$\varphi$ 
$$\varphi(A) = \int_{A}fd\mu~~~A\in F$$
分别为不定积分与关于测度$\mu$的导数 

对于集函数的导数是否存在与关于测度的求导的问题，就是本节研究的对象
定义：对于测度空间上积分存在的可测函数$f$的不定积分
$$\varphi(A) = \int_{A}fd\mu~~~A\in F$$
根据积分的定义我们知道，$\varphi$满足非负性以外的所有测度的基本条件，我们称其为**符号测度**  并且所有满足这样的条件的测度均称为符号测度。

#### Hann与Jordan分解
符号测度不具备非负性这件事情让人非常讨厌，我们有没有什么方法让他再有类似非负性的性质呢？ 这就是分解所研究的。

考虑前面的不定积分，对可测函数的定义域按照下面的规则分割
$$X^{+}=\langle f\geqslant0\rangle,\quad X^{-}=\langle f<0\rangle $$
则可以把原始空间$X$分成下面的两部分
$$A\in\mathscr{F},A\subset X^{+}\Rightarrow\varphi(A)\geqslant0;\\A\in\mathscr{F},A\subset X^{-}\Longrightarrow\varphi(A)\leqslant0.$$
我们将这样的对原始空间$X$的分解称为Hann分解。

再令
$$\varphi^{\pm} (A)=\int_{A}f^{\pm} \mathrm{d}\mu $$
则可以得到测度$\varphi$被分解为两个测度，并有分解式
$$\varphi=\varphi^+-\varphi^-$$
这称为Jordan分解，也成为符号测度的全变差。

**对于一般的符号测度，有存在Hann与Jordan分解**

#### Radon-Nikodym 定理
设 $\varphi$是测度空间($X,\mathscr{F},\mu)$上的符号测度，我们将着手定义 $\varphi$ 的导数. 基本的想法其实很简单：如果这个符号测度能惟一地表成不定积分形式
$$\varphi(A) = \int_{A}fd\mu~~~A\in F$$
那么就可以认为他们互相具有导数和不定积分。

定义：设 $\varphi$是测度空间(X,$\mathscr{F},\mu)$上的符号测度.如果存在a.e.意义下惟一的可测函数$f$前述式成立，则称$f$为$\varphi$对于$\mu$的R-N(Radon-Nikodym)导数(或简称导数),记为$\frac{\mathrm{d}\varphi}{\mathrm{d}\mu}\overset{\mathrm{def}}{\operatorname*{=}}f.$

正如微积分中并非所有的函数都可以求导一样，也不是每一个符号测度都有 R-N 导数.什么样的符号测度才有 R-N 导数呢？**只有当 $\varphi$对 $\mu$ 绝对连续时才有可能** 

定义 设 $\varphi$和 $\mu$ 分别是可测空间($X,\mathscr{F}$)上的符号测度和测度.如果对任何$A\in\mathscr{F}$均有
$$\mu(A)=0\Rightarrow\varphi(A)=0\:$$
则称 $\varphi$对 $\mu \textbf{}$绝对连续,记作$\varphi \ll \mu .$ 

#### Lebegue分解
本节的主题是 Lebesgue 分解，其目的是要证明，任何σ有限符测度$\varphi$对于任意一个 $\sigma$有限测度 $\mu$,可以分解成两部分：一部分对$\mu$绝对连续；另一部分则与$\mu$是相互奇异的
### 条件期望的存在唯一性
原始的初等概率论中的条件期望也很难让我们满意，因此这里我们研究公理化的条件期望的问题。

定义 设  $X, Y$  是定义在概率空间 $(\Omega, \mathscr{F}, \mathbb{P})$  上的可积变量，  $\mathscr{G}$  是  $\mathscr{F}$ 的子  $\sigma$ -代数。如果
* $Y \in \mathscr{G}$  （即  $Y$  是  $\mathscr{G}$  可测的，即 $\sigma(Y)\subset \mathscr{G}$）；
* 对任意  $A \in \mathscr{G}$ ,$$\int_{A} Y(\omega) d \mathbb{P}(\omega)=\int_{A} X(\omega) d \mathbb{P}(\omega),$$
则称  $Y$  是  $X$  在  $\mathscr{\mathscr { G }}$  的条件期望, 记为  $Y = \mathbb{E}[X \mid \mathscr{G}]$ 

好像和以前所学习的条件期望有一点不太一样？ 所有有下面附注

**附注1** ：若$\mathscr{G}$是由某个随机变量$Z$生成的$\sigma$代数，则
$$\mathbb{E}[X|\mathscr{G}]=\mathbb{E}[X|\sigma(Z)]=\mathbb{E}[X|Z].$$

**附注2** ：对于概率空间上$(\Omega, \mathscr{F}, \mathbb{P})$  的任意可积实值变量$X$ 和任意$\sigma$ -代数$\mathscr{G}  \subset \mathscr{F}$ 一定存在可测随机变量$Y$满足前述条件，并且该随机变量唯一。

**附注3** ：随机变量的条件期望还是一个随机变量，还是原本的概率空间$(\Omega, \mathscr{F}, \mathbb{P})$ 上的可测函数，我们可以记为$f(Z)$因为其生成的$\sigma$代数变小了

**附注4** ：一个事件$A$关于一个$\sigma$代数的条件概率就定义为$E[I_{A}\mid \mathscr{G}]$
### 条件期望的性质
设$X,Y$是定义在概率空间$(\Omega,\mathscr{F},\mathbb{P})$上的随机变量，且$\mathcal{H}\subset\mathscr{G}$ 是$\mathscr{F}$ 的子 $\sigma$代数
则有下面的性质成立
1. 对任意$a,b\in\mathbb{R}$,都有$\mathbb{E}[aX+bY|\mathscr{G}]=a\mathbb{E}[X|\mathscr{G}]+b\mathbb{E}[Y|\mathscr{G}].$
2. 若 $X\in \mathscr{G}$, 则  $\mathbb{E} [ XY| \mathscr{G} ] = X\mathbb{E} [ Y| \mathscr{G} ] .$ 
3. 若 $X\bot \mathscr{G}$, 则  $\mathbb{E} [ X| \mathscr{G} ] \Longrightarrow \mathbb{E} [ X] .$
4. $\mathbb{E} [ \mathbb{E} [ X| \mathscr{G} ] \mid \mathscr{H} ] = \mathbb{E} [ X| \mathscr{H} ] .$ 全期望公式
5. 若$\varphi$是凸函数，则$\varphi(\mathbb{E}[X|\mathscr{G}])\leq\mathbb{E}[\varphi(X)|\mathscr{G}].$

他们分别对应
1. 线性性
2. 把知道的拿出去（提取已知量）
3. 独立性
4. 累次条件期望（即全期望公式的广义形式）
5. 条件詹森不等式
