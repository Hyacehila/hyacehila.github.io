---
title: "泛函分析：度量空间、紧性与可分性"
title_en: "Functional Analysis: Metric Spaces, Compactness, and Banach Spaces"
date: 2023-09-11 21:36:59 +0800
categories: ["Mathematics", "Mathematical Analysis"]
tags: ["Functional Analysis", "Metric Spaces"]
author: Hyacehila
excerpt: "整理度量空间、紧性、可分性、完备性、线性赋范空间、Banach 空间和线性算子等内容。"
excerpt_en: "Covers metric spaces, compactness, separability, completeness, normed linear spaces, Banach spaces, and linear operators."
mathjax: true
hidden: true
permalink: '/blog/2023/09/11/functional-analysis-notes/'
---
在数学分析中，我们研究了整个高等数学理论的核心微积分；随后的复变函数拓展相关理论到复数域，实变函数引入了更为广义的勒贝格积分；泛函分析研究更加多的函数，研究他们和微积分一样的性质，他是后续非常多的课程的基础知识，是重要的基础课之一

## 度量空间
微积分理论中研究的对象是函数，我们是从极限和连续函数开始非常多的研究的；刻画极限需要使用距离作为基础；这就是度量空间中需要研究的内容；我们希望研究更为广义的距离
### 度量空间的定义
#### 度量空间的定义
设非空集合$X$，存在二元映射$d(x,y)\to R$ 使得任意的$x,y$属于$X$ 满足以下的条件
1. 非负性（Positivity):$d(x,y)\ge 0 当且仅有x=y时取等号$
2. 对称性（symmetry）:$d(x,y)=d(y,x)$
3. 三角不等式（triangle inequality）:$d(x,y)\le d(x,z)+d(y,z)$

则称$d$时$x$上的一个距离函数 称$(x,d)$是距离空间 称函数$d$的结果是距离
#### 例子
##### 欧式空间
取空间$R^{n}={(x_{1},x_{2},x_{3}...x_{n}|x_{i}\in R)}$
取距离函数$d(x,y)=\sqrt{\sum\limits(x_{i}-y_{i})^{2}}$
证明他是一个度量空间

使用定义就可以验证；
$$\left[\sum_{i=1}^{n}\left(x_{i}-y_{i}\right)^{2}\right]^{\frac{1}{2}}=\left[\sum_{i=1}^{n}\left(x_{i}-z_{i}+z_{i}-y_{i}\right)^{2}\right]^{1/2}\leqslant\left[\sum_{i=1}^{n}\left(x_{i}-z_{i}\right)^{2}\right]^{1/2}+\left[\sum_{i=1}^{n}\left(z_{i}-y_{i}\right)^{2}\right]^{1/2}$$
非负性和对称性是明显满足的 我们只需要证明三角不等式性质成立
需要使用Minkowski不等式 我们就可以证明之

公式如下
$$\begin{aligned}&&(\sum_{i=1}^n|\left.a_i+b_i\right|^k)^{\frac1k}\leqslant(\sum_{i=1}^n|\left.a_i\right|^k)^{\frac1k}+(\sum_{i=1}^n\left|\left.b_i\right|^k\right)^{\frac1k}.\end{aligned}$$
k偶数的时候就可以去除绝对值符号了
这个公式还有一个积分形式 使用勒贝格积分 也非常常用 如下
$$\left(\int_{x}\left|f\left(x\right)+g\left(x\right)\right|^{s}\mathrm{d}x\right)^{\frac{1}{s}}\leq\left(\int_{x}\left|f\left(x\right)\right|^{s}\mathrm{d}x\right)^{\frac{1}{s}}+\left(\int_{E}\left|g\left(x\right)\right|^{s}\mathrm{d}x\right)^{\frac{1}{s}}$$
可以看出这个不等式应该是非常好用的

我们一般称上面定义的距离函数为标准欧式距离；还存在一些其他的定义方式比如 $d(x,y)=max\{|x_{k}-y_{k}|\}$   $d(x,y)=\sum\limits{|x_{k}-y_{k}|}$   等 ；

**同一个空间也有很多种定义度量的方法 形成多种度量空间 度量空间是空间和度量函数的整体**

##### 离散度量空间
对任意非空集合X定义距离函数如下
$$d_{0\left(x,y\right)}=\left|\begin{matrix}0,x=y,\\1,x\neq y.\end{matrix}\right.$$
容易验证 它满足度量空间的三个条件 我们称这样的空间是离散度量空间；
任意的非空集合总能这样定义度量空间 他也是一个很特殊的度量空间 我们在后面举反例会用到很多

##### 连续函数空间
设$C[a,b]=\{f:[a,b]\to R|f连续\}$ 定义距离函数为
$$d\left(f,g\right)=\max_{t\in\left[a,b\right]}\left|f\left(t\right)-g\left(t\right)\right|$$
非常明显满足对称性和非负性，想要证明三角不等式只需要借助**经典的绝对值三角不等式进行展开**；
我们称这样的空间为连续函数空间
连续函数空间也有一种关于距离的定义方式
$$d\left(f,g\right)=\int_{a}^{b}\left|f(x)-g(x)\right|\mathrm{d}x$$
##### 有界数列空间
记 $l^{\infty}=\left\{x=\left(x_{1},x_{2},\cdots,x_{n},\cdots\right)=\left(x_{i}\right)|\sup_{i\geqslant1}\left\{\left|x_{i}\right|\right\}<\infty\right.$  对于$x=(x_{i}),y=(y_{i})\in l^{\infty}$ 定义
$$d\left(x,y\right)=\sup\left|x_{i}-y_{i}\right|,$$
容易验证（借助我们这里使用的模的性质） 这是一个度量空间  记为$l^{\infty}$

##### p次幂可和的数列空间
这是对原本欧式空间的一个推广
$$\begin{aligned}\text{记 }l^p=(x=(x_1,x_2,\cdots,x_n,\cdots)=(x_n,\cdots)\sum_{i=1}^n|x_i|^{\prime}<\alpha|\text{,其中 }1\leqslant p<+\alpha,\text{对于}\forall x=(x_i),y=(y_i)\in l^p,\text{定义}\end{aligned}$$
$$d_p(x,y)=(\sum_{i=1}^n|x_i-y_i|^p)^{\frac1p}$$
这构成了一个度量空间 一般记作$l^{p}$
##### p次幂可积的函数空间
这是对连续函数空间的积分形式的一种推广
$$\begin{aligned}&\text{记}L^{p}[a,b]=\{f(t)|\int_{[a,b]}\mid f(t)\mid^p\mathrm{d}t<+\infty\}\\&\text{其中 }1\leqslant p<+\infty,\int_{[a,b]}\mid f(t)\mid^p\mathrm{d}t\text{ 表示}|f(t)|^p\text{ 在}[a,b]\text{ 上的勒贝格积分,即 L积为}\\&L^p[a,b]\text{中,几乎处处相等的函数规为同一函数. 对于 }f,g\in L^s[a,b]\text{,定义距离}\\&d(f,g)=(\int_{[a,b]}|f(t)-g(t)\mid^s\mathrm{d}t)^{\frac1p},\end{aligned}$$
能够验证这满足度量空间的定义


至此所有重要的度量空间和他们涉及到的一些证明方法已经全部给出 后面我们将着手研究这个空间的各种性质


### 度量空间的拓扑性质
本节需要对照参考[点集拓扑学中的拓扑空间](/blog/2024/10/16/point-set-topology-notes/)
#### 基本拓扑性质
定义（邻域）：$O(x_{0},\sigma)=\{x|d(x,x_{0})<\sigma\}$ 称为邻域

定义（内点）：集合内某点存在某个邻域在这个集合内 则称这个点是内点

定义（开集）：一个集合的所有点都是他的内点的集合 也就是$intG=G$

定义（闭集）：如果$F^{C}=X\setminus F$ 是开集 那么$F$是闭集

开集的性质
* 空集和全空间都是开集
* 任意多开集的并是开集
* 有限个开集的交是开集

闭集满足：
* 空集和全空间是闭集
* 任意多闭集的交是闭集
* 有限个闭集的并都是闭集

#### 更多的拓扑性质
定义（聚点和闭包）
设 $A$ 是拓扑空间 $X$ 的子集，$xin X$。如果 $x$ 的每个邻域都含有 $Aackslash{x}$ 中的点，则称 $x$ 为 $A$ 的聚点；$A$ 的所有聚点的集合称为 $A$ 的导集，记作 $A^{prime}$。称集合 $overline A:=Aigcup A^{prime}$ 为 $A$ 的闭包。
能看出 聚点不一定在集合中

定理：以下命题等价
* $x_{0}\in A^{'}$当且仅当存在${x_{n}}\in A,s.t~ ~\lim_{x \to \infty} x_{n}=x_{0}$
* $\bar{A}$是闭集
* $A$是闭集当且仅当$A=\bar{A}$
* 如果存在$F\subset X$使得$A\subset F$ 则$A\subset\bar{A}\subset F$

**这部分定理和推论告诉我们，集合的闭包是包含他的最小闭集，也是比他大的所有闭集的交**


### 度量空间的极限和连续
#### 极限
极限理论是微积分理论的基础 将他拓展到更大范围的度量空间是有必要的

定义（极限）：$\lim_{x \to \infty}d(x_{n},x_{0})=0$ 就是收敛 称$x_{0}$是$x_{n}$的极限 反之发散

**数列是否收敛和度量空间的选取有关，和$X$与$d$都有关**

定义（子空间）：对于度量空间$(X,d)$ 明显的对于子集$A$也有度量空间$(A,d)$ 称为子空间

定义（距离）：点到集合的距离为$d(x,A)=inf_{y\in A}\{d(x,y)\}$

定义（直径）：集合$A$的直径为$diaA=sup_{x,y\in A}\{d(x,y)\}$

**直径有限 则是有界集 反之是无界集**

定理（极限的性质）：
* 点列收敛则极限唯一
* 点列收敛 则其子列也收敛并且有相同极限
* 收敛点列看作集合 他也是一个有界集


#### 连续和一致连续
这个问题要针对映射才能研究

定义：
设$(X,d),(Y,d)$是两个度量空间 $f$是他们上面的一个映射
$$\begin{aligned}&\text{ 对于 }x_0\in X,\text{如果 }\forall\varepsilon>0,\exists\delta>0,\text{当}x\in X\text{且}d(x,x_0)<\delta\\&\text{则称}f\text{ 在点 }x_0\text{ 处连续.若 }f\text{在X 的每一点处都连续,则称映射 }f\text{ 在 }X\text{ 上连续}.\\&\text{ 如果 }\forall\epsilon>0,\exists\delta>0,{\forall x,y\in X,\text{当}d(x,y)\leq\delta}\text{时,有}\rho(f(x),f(y))<\varepsilon\text{,则称}\\&f\text{在}X\text{上一致连续}\end{aligned}$$
**和数学分析一样的是，连续概念针对局部，一致连续概念针对整体**
函数在闭区间上连续和一致连续等价

定理（连续的等价条件）
以下命题等价
* 映射$f$在$x_{0}$点连续
* 存在$f[O(x_0,\delta)]\subset O(f(x_0),\varepsilon).$
* $\lim_{n\to\infty}f\left(x_{n}\right)=f\left(x_{0}\right)~if~ x_{0}是x_{n}$
*把直观的思想叙述出来就是证明了*
定理（连续的充要条件）
$$\text{对}Y\text{中的任一}\text{开集}G,\text{其原像}f^{-1}\left(G\right)=\{x|x\in X,f\left(x\right)\in G\}\text{是开集}$$
这就是连续的充要条件

本定理还有一个等价形式是 闭集的原像是闭集也是连续的充要条件

### 度量空间的可分性

#### 稠密的定义
定义（稠密）
设$X$是度量空间$A,B\subset X$ 如果$B$中任意点的任何邻域都含有$A$中的点 则称$A$在$B$中稠密 如果$A$恰好是$B$的子集 那么称$A$是$B$的稠密子集
非常容易理解 有理数在无理数和实数中稠密 无理数在有理数和实数中稠密
稠密所体现的意思和他字面的意思非常接近
定理：$(X,d)$是度量空间$A,B\subset X$ 以下命题等价
* $A$在$B$中稠密
* $\forall x\in B,\exists\left\{x_{n}\right\}\subset A,\text{使得}\lim_{n\to\infty}d\left(x_{n},x\right)=0$
* $B\subset\overline{A}$
* $\text{任取}\delta>0,\text{有}B\subset\cup_{x\in A} O\left(x,\delta\right).$

定理（稠密集的传递性）
设$X$是度量空间$A,B,C\subset X$ 如果 $A$在$B$中稠密 $B$在$C$中稠密 则$A$在$C$中稠密


#### 可分的定义
定义（可分）
设$X$是度量空间$A\subset X$ 如果$A$存在**可列的稠密子集** 则称$A$是可分点集 如果$X$本身是可分点集 则称$X$是可分的度量空间

非常容易验证 实数集是一个可分的空间 ，他有有理数集这样的稠密子集 而有理数集是可列的
#### 可分空间举例
* 欧式空间$R^{n}$可分
* 连续函数空间$C[a,b]$可分
* $p$次幂可积函数空间$L^{p}[a,b]$ 可分
* $p$次幂可和的数列空间$l^{p}$可分
* 有界数列空间$l^{\infty}$不可分
* 设$X=[0,1]$ 离散度量空间$(X,d)$不可分
* $X$可列时 离散度量空间$(X,d)$可分

#### 推论
定理：
$$设X是可分的度量空间Y是X的子空间 则Y是可分的子空间$$
推论：
$$\begin{aligned}&\text{ 设 }X\text{是度量空间, }Y\subset X\text{ 是不可列子空间,且存在 }\delta>0\text{,}\forall x\text{,}y\in Y,\text{满足}d(x,y)\ge\sigma \text{则}X\text{不是可分空间}\end{aligned}$$

### 度量空间的完备性
这里我们推广实数中的Cauchy收敛原理到度量空间中
#### 定义与反例
定义（基本列）
$$\begin{aligned}\text{设}\{ x_n\}\text{是度量空间}X\text{ 中的一个点列},\text{若对任意}\epsilon>0,\text{存在}n、N\in N,\text{当 }m,n>N\text{时,有}\\&d(x_n,x_n)<\epsilon,\text{则} \{x_{n}\},\text{是}X\text{ 中的一个基本列}\end{aligned}$$

定理：基本列的性质 在度量空间$(X,d)$中
* 收敛列是基本列
* 基本列构成的集合是有界集合
* 如果基本列含有收敛子列 则他是收敛列并且和子列收敛于同一点
**这个定理表示收敛列一定是基本列 事实上反过来他不一定在度量空间上成立，仅仅在欧式空间成立**


#### 完备的度量空间
定义
对于一个度量空间$X$ 如果她的任何基本列都是收敛列 则我们称$X$是完备的度量空间

* $n$维欧氏空间是完备的
* 连续函数空间是完备的（当采用第一种距离定义形式）
* 连续函数空间是不完备的（当采用第二种距离定义形式）

#### 一些推论
定理（闭球套）
$$\begin{aligned}&(X,d)\text{是完备的度量空间,}B_n=\overline{\mathcal{O}}(x_n,\delta_n)\text{是一套闭球，后者是前者的子集}\\&\text{如果球的半径 }\delta_n\to0,n\to\infty,\text{那么在在唯一的点 }x\in\bigcap_{n=1}^\infty B_n.\end{aligned}$$
闭球套定理也是度量空间的另一种刻画形式
正如我们在数学分析中介绍了的实数完备性定理一样 大量的等价定理构成了实数的完备性

定理
设$(X,d)$是完备的度量空间 则$M\subset X$是完备集当且仅当$M$是闭集

对常用度量空间的可分与完备性总结
![常用度量空间可分性与完备性总结](/assets/images/mathematics-notes/functional-analysis-01.png)
这张图我们需要记住；
前面的很多证明题目其实就是在证明上面的结论


### 度量空间的紧性
从直观的角度理解 紧意味着每个元素在集合中排列的足够密（致密性）
在数学分析中  闭区间连续函数有着最值 有一致连续性 这也源于紧性
实际上 **紧性的数学语言描述是 有界点列必有收敛子列 也就是致密性定理**

定义：
设$X$是度量空间$A\subset X$ 如果$A$中任何点列都有收敛于$X$的子列 那么则称$A$是列紧集 如果$A$同时也是闭集 则称$A$是紧集 如果$X$本身是列紧集（它必是闭集） 则称$X$是紧空间
*全空间和空集都是既是开集也是闭集，前面介绍过*
**列紧只控制一种情况，就是收敛点列收敛到了集合外，理解这一点**

性质
* 设$X$是度量空间$A\subset X$ 则$A$是紧集当且仅当任意$A$中点列均存在收敛子列收敛于$A$中的点
* 任意有限集是紧集
* 列紧集的子集是列紧的
* 任意多列紧集的交是列紧集
* 有限多列紧集的并是列紧的
* $A$是$X$的列紧集当且仅当$\bar{A}$是紧集

推论 设$X$是紧空间$A\subset X$
* 紧空间是有界空间
* 紧空间是完备的度量空间
* $A$是紧集当$A$是闭集

定理：设$A$是$n$维欧氏空间的一个子集
* $A$是列紧集当且仅当$A$是有界集
* $A$是紧集当且仅当$A$是有闭界集

定理：
连续映射将紧集映射为紧集
定义
映射到实数的连续映射（实值连续映射）也称为泛函
定理：
设$X$是紧空间，紧集$A\subset X$ $f$是定义在X上的泛函  那么$f$在$A$上能取到最大值和最小值


### 度量空间的全有界集
这个概念还是用来刻画列紧性的；
在完备的度量空间中 列紧性和全有界性等价；
在一般的度量空间中 列紧集一定是全有界集

定义（$\epsilon$网）
设$X$是度量空间 $A,B\subset X$ 对于给定的$\epsilon$ 如果对于$B$中任何点$x$ 一定存在$A$中点$x^{'}$使得$d(x,x^{'})<\epsilon$则称A是B的一个 $\epsilon$ 网 也就是$B\subset \cup_{x\in A} O(x,\epsilon)$
**明显的 稠密可以保证$\epsilon$网的存在，但是反过来是不可以的**
因为稠密要求的是任意距离 但是$\epsilon$网是要求有限距离（这是弱化的稠密）
定义（全有界集）
设$X$是度量空间$A\subset X$ 如果对于任意给定的$\epsilon>0$ $A$总存在有限的$\epsilon$网 则称A是$X$中的全有界集
核心点有两个** 总是存在和有限的网**
引理
$A$是度量空间的全有界集 当且仅当$\forall \epsilon > 0 ~\exists\{x_{1},x_{2}...x_{n}\} \subset A$  使得 $A\subset \cup_{i=1}^{n}O(x_{i},\epsilon)$
引理的证明仅仅是定义的运用 非常的自然

定理
设$X$是度量空间$A\subset X$ 若A是$X$中的全有界集 则
* A是有界集
* A是可分集

定理
A是全有界集的充要条件是 A中的任何点列都有基本子列

Hausdorff 定理
设$X$是度量空间$A\subset X$ 则
* A是列紧集，则A是全有界集
* 若X是完备度量空间 则A是列紧集当且仅当A是全有界集

总结
$$\begin{aligned}\\\text{紧集}\Rightarrow\text{列紧集}\Rightarrow\text{全有界集}\Rightarrow\text{有界集}+\text{可分集}\\\text{紧集}\Leftarrow_{\text{闭}}\text{列紧集}\Leftarrow_{\text{完备}}\text{全有界集}\end{aligned}$$

### 度量空间中的开覆盖
定义：
$$\begin{aligned}\text{设 }X\text{ 是度量空间},&\Lambda\text{ 为一指标集,}A\subset X\text{,}\forall\lambda\in\Lambda,G_\lambda\text{ 是}X\text{ 的开子集,如果 A}\subset\bigcup_{\lambda\in A}G_\lambda,\\\text{则称}&\{G_\lambda|\lambda\in\Lambda\}\text{是 A 的开覆盖}\end{aligned}$$
这里对指标集合的可列性没有要求
引理：
设$X$是度量空间 A是X的紧子集； $\{G_\lambda|\lambda\in\Lambda\}$  是A的一个开覆盖 则存在$\epsilon>0$ 使得任意$x\in A$ 存在$G_{x}\in \{G_{\lambda}\}$ 满足$O(x,\epsilon)\subset G_{x}$
定理：
设$X$是度量空间$A\subset X$ 则 A是紧集当且仅当A的任意开覆盖存在有限开覆盖
**这个定理是非常重要的，也是这一小节唯一的核心**
性质：
紧空间上的连续映射是一致连续映射
性质：
$$\quad\text{设}\left(X,d\right)\text{为度量空间},\text{则 }X\text{ 为紧空间的充要条件是:对 }X\text{ 中的任意闭集}\\\text{族}F_{\lambda},\left(\lambda\in\Lambda\right)\text{,若其中任意有限个闭集 }F_{\lambda}\text{的交集都为非空集,则}\bigcap_{\lambda\in\Lambda}F_{\lambda}\text{也必为非空集}$$


## 线性赋范空间和内积空间
### 线性赋范空间的定义和性质
#### 定义
设$X$是数域$F$上的线性空间（实数R复数C都可以） 如果对于每一个$x\in X$ 有一个确定的是数与之对应 记为 $||x||$  并且有
* 非负性 $||x||\ge0$ 并且只有$x=0$ 有  $||x||=0$
* 齐次性 $||ax||=a||x||$
* 三角不等式 $||x+y||\le||x||+||y||$
则称$||x||$ 是x的范数 定义了范数的线性空间称为线性赋范空间($B^{*}$空间 )
如果我们定义 $$d\left(x,y\right)=\left\|x-y\right\|$$
容易验证这个距离一定满足非负性 对称性 三角不等式 所以可以构成了一个度量空间 称为由线性赋范空间诱导的度量空间
此时我们前面在度量空间一章中研究的全部问题都可以继续运用了
#### 依范数收敛
设$X$为线性赋范空间，${x_n}$是$X$ 中的点列，$x\in X$,如果$lim||x_{n}-x||=0$ 依范数收敛于$x$(简称${x_n}$收敛于x),记为$\lim x_n=x$或$x_n\to x,n\to\infty$
能看出 依范数收敛和依范数导出的距离收敛是一样的
给出几个性质
(1)范数的连续性：范数$||x||$是从$X$到$R$上的连续映射。
(2)有界性：若$(x_n)$收敛于$x$,则{H}$||x_{n}||$有界
(3)线性运算的连续性：若$x_n\to x,y_n\to y,n\to\infty$,则$x_n+y_n\to x+y,ax_n\to ax,n\to\infty$, 其中$a$为常数。
从这些性质来看 范数一定是一个连续泛函（从空间到实数域上的映射）
#### Banach空间
设$X$是一个线性赋范空间 如果他根据自己范数导出的距离$d\left(x,y\right)=\left\|x-y\right\|$ 是完备的 （基本列一定收敛） 那么我们称其为Banach空间 简称为$B$ 空间
这个理论对我们后面引入非常多重要的定理都很有用
我们可以从上一章中关于某些空间的完备性来反着研究什么样的线性赋范空间是Banach空间
![度量空间与线性赋范空间关系总结](/assets/images/mathematics-notes/functional-analysis-02.png)
注意：并不是所有度量空间都能找到对应的线性赋范空间 也就是单向对应
比如离散度量空间就找不到对应的线性赋范空间 因为违法了他的三条公理
$$\left\|x\right\|=d_{0}\left(x,\theta\right)=1,\left\|2x\right\|=d_{0}\left(2x,\theta\right)=1$$
下面是度量空间构成线性赋范空间的条件
$$d(x-y,\theta)=d(x,y),d(ax,\theta)=\big|\alpha\big|d\left(x,\theta\right).$$
#### 级数
设$X$为线性赋范空间，点列$\{x_n\}\subset X$,称表达式 $x_1+x_2+\cdots+x_n+\cdots=\sum x_n$ 为$X$中的级数。若部分和点列$S_n=x_1+x_2+\cdotp\cdotp+x_n$依范数收敛于$s\in X$,则称级数$\sum x_n$收敛于$s$,称$s$为级数的和，记为$s=\sum_{n=1}^{\infty}x_n$.如果数项级数$\sum_{n=1}^{\infty}\|x_n\|$收敛，则称级数$\sum_{n=1}^{\infty}x_n$绝对收敛
定理：设$X$是线性赋范空间，则$X$ 是 Banach 空间当且仅当$X$ 中任何级数的绝对收敛总蕴含级数收敛
定理：设 X 是 Banach 空间，$\left\{x_{n}\right\},\left\{y_{n}\right\}\subset X$ ,且存在 $N\in\mathbb{N},当 n>\mathbb{N}$ 时，
$||x_{n}\parallel=c\parallel y_{n}\parallel$,其中$c$为常数，那么若$\sum^{\infty}y_{n}$绝对收敛，则$\sum_{n=1}^{\infty}x_{n}$也绝对收敛.

### 线性赋范空间的子集和商空间
#### 凸集
设$X$为数域F上的线性空间，C为X 的子集，若$\forall x,y\in C$,有
$$\left\{\alpha x+\left(1-\alpha\right)y|0\leqslant\alpha\leqslant1\right\}\subset C$$
则C为X的凸集
证明线性赋范空间上的闭单位球$B(0,1)=\{x|~||x||\le1\}$
#### 子空间
子空间(Subspace) 设($X,\|\cdot\|)$为线性赋范空间，V是X 的线性子空间，并且V 中元素 $x$ 的范数依物是其在$X$ 中的范数$\|x\|$,则称$(V,\|\cdot\|)$或者$V$是线性赋范空间$X$ 的子空间
性质：
设$X$是线性赋范空间  则子空间的闭包$\bar{V}$ 是线性子空间
闭包是为了保证线性
设$X$是Banach $M$是$X$的线性子空间 则$M$是Banach的子空间当且仅当是闭集
我们只需要知道他的完备性 前面在完备性一节的定理可以保证这一点

*我们知道，线性赋范空间R”的子空间R是闭集，那么一般情况下，线性赋范空间的子空间是否一定是闭集呢？如果线性赋范空间$X$作为线性空间时它的维数为$n$,则称$X$为n维线性赋范空间、若$X$ 是有限维线性赋范空间 $X$ 的任何子空间都是闭集.然而，对于无穷维线性赋范空间而言，子空间未必是闭集
这个内容我们会在下一节详细解释 这里不用管*

#### 线性张
设$E$是数域F上线性空间$X$的非空子集，E中所有有限集的所有线性组合形成的集合就是E的线性张(Linear Span),记为
$$\left.\mathrm{span}E=\left<a_{1}x_{1}+a_{3}x_{2}+\cdots+a_{n}x_{n}\right|x_{1}+x_{2},\cdots,x_{n}\in E,a_{1},a_{2},\cdots,a_{n}\in\mathbb{R}\right>.$$
等价地有，spanE 是包含 E 的所有线性子空间的交集. 事实上，spanE 是包含 E 的最小线性子空间

设$X$为数域F上的线性赋范空间，E是X 的非空子集，则称包含 E 的所有闭线性子空间的交集为E的闭线性张，记为$\overline{\mathrm{span}}E$

设 X 为数城F上的线性赋范空间，E 是 X 的非空子集，则
$\left(1\right)\overline{span}E\in X\text{的闭线性子空间}.$
$\left(2\right)\overline{span}E=\overline{spanE},$
#### 商空间
设 X 为数城 F上的线性赋范空间，V 是 X 的闭子空间，若 $x-y\in V$,则称 $x$ 和 y 属于同一等价类，记为$[x]$或者$\widetilde{x}$,这些等价类的全体记为 $X/V=\langle[x]|[x]=x+V\rangle$ ,称 X/V 是X 关于V 的商空间、商空间 X/V 的加法、数乘以及范数的定义如下
$\forall\left[x\right],\left[y\right]\in X/V,\alpha\in\mathbb{F},\text{有}$
$\left[x\right]+\left[y\right]=\left(x+V\right)+\left(y+V\right)=x+y+V=\left[x+y\right]$
$a\left[x\right]=a\left(x+V\right)=ax+V=\left[ax\right]$
$\left\|\left[x\right]\right\|=\left\|x+V\right\|=\inf\left\{d\left(x,v\right)|v\in V\right\}.$
理解商空间  商空间中每一个元素$[x]$都代表了原始线性赋范空间中的一个子集
$$\left.\left[x\right]=x+V=\left<x+v\right|v\in V\right>\subset X.$$
也就是$x$是$[x]$中的一个代表 我们可以在商空间定义范数
$$\begin{aligned}\|\begin{bmatrix}x\end{bmatrix}\|&=\inf\langle d(x,v)\mid v{\in}V\rangle=\inf\{d(x,x-y+v)\mid x-y+v{\in}V\rangle\\&=\inf\{d(y,v)\mid v{\in}V\rangle=\|[y]\|.\end{aligned}$$
能验证他也是一个线性赋范空间

一些性质
设 X 为线性赋范空间，V 是 X 的闭子空间
$\left(1\right)设Q；X\to X/V$为自然映射(Natural Map), $Q\left(x\right)=\left[x\right]=x+V$, 则$\forall x\in X$,$\left\|Q\left(x\right)\right\|\leqslant\left\|x\right\|,Q$为连续映射
(2)如果X为 Banach 空间，则商空间 X/V 也为 Banach 空间，
(3)W是$X/V$的开集当且仅当$Q^{-1}\left(W\right)=\left\{x\left|Q\left(x\right)=\left[x\right]\in W\right\}\in X\right.$的开集，
(4)如果$U$是$X$的开集，则$Q\left(U\right)$是商空间$X/V$的开集

### 线性赋范空间的同构和范数等价
在线性代数中 同构指的是两个有限维线性空间上存在保持加法和数乘的一一映射 在同构的意义上 他们是有着完全一样的性质的空间
#### 线性等距同构
设$(X,\|\cdot\|_x),(Y,\|\cdot\|_Y)$是同一数域F上的两个线性赋范空间，如果存在一一映射$T:X\to Y,$满足：
(1)线性： $\forall x_{1},x_{2}\in X,\alpha,\beta\in\mathbb{F},T(\alpha x_{1}+\beta x_{2})=\alpha T(x_{1})+\beta T(x_{2});$ $\left(2\right)等距：\forall x\in X,\parallel Tx\parallel_{Y}=\parallel x\parallel_{X}$,
则称又和$Y$线性等距同构，并称映射$T$是线性等距同构映射
我们在线性代数中给出了结论 同构等价于他们有着相同的维数 事实上在线性赋范空间中还是能给出类似的结论
$$\text{定理2.3.1 设 X 是实数域R上的 n 维线性赋范空间,则 X 与R"线性等距同构.}$$
根据此容易给出推论：有限维线性赋范空间的子空间一定是闭集（因为$R^n$是这样的）
#### 等价的范数
有了线性赋范空间的同构 我们肯定想知道同构的时候范数的关系
定义
设$\|\cdot\|_1和\|\cdot\|_2$是定义在同一线性空间$X$上的两个范数，点列$\left\{x_n\right\}\subset X$,如果由$x_n\parallel_1\to0$ 可得$\|x_n\|_2\to0$,则称$\|\cdot\|_1比\|\cdot\|_2$强，
如果$\|\cdot\|_1比\|\cdot\|_2$强，且$\|\cdot\|_2比\|\cdot\|_1$强，则称范数$\|\cdot\|_1和\|\cdot\|_2$等价
定理
线性赋范空间$X$上的两个范数$\|\cdot\|_1$和$\|\cdot\|_2$ 等价当且仅当存在正实数a和$b$,使得$\forall x\in X$,有
$$a\left\|x\right\|_{2}\leqslant\left\|x\right\|_{1}\leqslant b\left\|x\right\|_{2}.$$
定理
有限维线性赋范空间上任意的范数等价
这个定理告诉我们 可以选取不同的范围或者说最简单的范数来思考问题
### 线性赋范空间的维数和紧性
#### 线性赋范空间的维数
如果线性赋范空间$X$作为线性空间时它的维数为$n$,则称$X$为$n$维线性赋范空间
如果他的维数是无穷的 那么我们称其为无穷维的
#### 线性赋范空间的紧性和维数的关系
在度量空间中我们介绍过 $R^{n}$子空间紧等价于其有界性
定理：
设$X$是线性赋范空间 那么$X$的维数有限当且仅当$X$中的每一个有界集都是列紧集
等价命题：
设$X$是线性赋范空间 那么$X$是无穷维的，那么至少$X$中的一个有界集不是列紧集

Riesz引理：设$A$是线性赋范空间$X$ 的闭子空间，且$A\neq X,0<\alpha<1$,则存在$x_a\in X$,使得$||x_{a}\parallel=1,且d\left(x_{a},A\right)>\alpha.$


### 内积空间的定义
还是在线性代数中 我们在介绍完线性空间后便去研究了内积空间，尝试在原本的线性空间中增加几何结构的度量 现在我们要在线性赋范空间上复现这些研究
#### 定义
设$X$是数域$F$上的线性空间，若存在映射(・,・),$X×X→F$,使得$\forall x,y,z\in\mathbb{X},\alpha~\beta\in F$ 满足
* 非负性：$\left(x,x\right)\geqslant0,\left(x,x\right)=0\text{ 当且仅当 }x=0$
* 共轭对称性（实数上的就是对称性）$\left(x,y\right)=\overline{\left(y,x\right)}$
* 第一变元线性性（第二变元是共轭线性的，也就是实数的时候也线性）$\left(\alpha x+\beta x,y\right)=\alpha\left(x,y\right)+\beta\left(z,y\right)$
我们称$F=R$的时候为实内积空间 $F=C$的时候为复内积空间
有限维的实内积空间称为欧氏空间 有限维的复内积空间称为酉空间
容易证明 对于连续函数空间$C[a,b]$ 定义内积为
$$\left(f,g\right)=\int_{a}^{a}f\left(x\right)g\left(x\right)dx$$
容易验证他的确实是一个实内积空间
#### 导出线性赋范空间
对于内积空间 定义范数为$\left\|x\right\|=\left(x,x\right)^{\frac{1}{2}}$
给出另一种Cauchy不等式作为引理
$$\left.\text{设}X\text{为内积空间,证明}\forall x,y\in X,\text{有}|\left(x,y\right)\right|\leqslant\left\|x\right\|\cdot\left\|y\right\|$$
**他建立了内积和范数的关系** 这个公式非常重要
使用Cauchy不等式可以证明
使用内积导出的范围满足正定性和齐次性 下面证明了三角不等式也被满足
$$\begin{aligned}&\|x+y\|^2=|(x+y,x+y)|=|(x,x+y)+(y,x+y)|\\&\leqslant|(x,x+y)|+|(y,x+y)|\leqslant\|x\|\cdot\|x+y\|+\|y\|\cdot\|x+y\\&\leqslant(\|x\|+\|y\|)\|x+y\|,\\&\text{故}\|x+y\|\leqslant\|x\|+\|y\|.\end{aligned}$$
核心点就是使用第一变元线性性进行拆分 结合新的Cauchy不等式进行计算，这在内积空间后面的研究中会非常的常用
现在我们可以知道
$$\text{内积空间}\longrightarrow \text{线性赋范空间}\longrightarrow \text{度量空间.}$$
三种空间可以依次导出
#### Hilbert空间
设$X$是数域$F$上的内积空间 按照内积导出的范围为$\left\|x\right\|=\left(x,x\right)^{\frac{1}{2}}$   如果按照这个范围$X$是Banach空间 我们则称内积空间$X$是Hilbert空间 称为H空间
**实际上无论是H空间还是B空间 他们都是研究导出的距离是否保证完备性，两者的研究是基本一致的**
容易给出下面的定理
$H$空间的子空间也是$H$空间当且仅当他是闭集
还是沿用研究B空间的思路 我们反找那些满足条件的内积有
实内积空间$R^n$ 定义内积为$(x,y)=x_{1}y_{1}+x_{2}y_{2}+\cdots+x_{4}y_{n}$
复内积空间$C^n$ 定义内积为$\left(x,y\right)=x_{1}\overline{y_{1}}+x_{2}\overline{y_{2}}+\cdots+x_{s}\overline{y_{n}}$
复内积空间$l^2$ 定义内积为$\left(x,y\right)=x_{1}\overline{y_{1}}+x_{2}\overline{y_{2}}+\cdots+x_{s}\overline{y_{n}}+...$
复内积空间$L^2[a,b]$  定义内积为 $(x,y)=(L)\int_{[0,\delta]}x(t)\overline{y(t)}dt.$
理解我们是怎么证明的就好了实际上就是借助已知的完备空间在反找，我们在例子中证明了其他的非常多原本是Banach的空间都无法继续诱导维Hilbert空间
### 内积空间和线性赋范空间
内积空间能导出线性赋范空间 前面的结尾我们还进行一些反找 但是这是一定可以找到的吗？ 其实并不一定能
#### 极化恒等式
对于实内积空间和它导出的范数
$$\left(x,y\right)=\frac{1}{4}\left(\parallel x+y\parallel^{2}-\parallel x-y\parallel^{2}\right).$$
对于复内积空间和它导出的范数
$$(x,y)=\frac{1}{4}(\parallel x+y\parallel^2-\parallel x-y\parallel^2+\mathrm{i}\parallel x+\mathrm{i}y\parallel^2-\mathrm{i}\parallel x-\mathrm{i}y\parallel^2).$$
#### 平行四边形公式
线性赋范空间$X$成为内积空间当且仅当$\forall x,y\in$ X,范数满足平行四边形公式
$$\left\|x+y\right\|^{2}+\left\|x-y\right\|^{2}=2\left\|x\right\|^{2}+2\left\|y\right\|^{2}.$$


### 内积空间的正交分解
在三维空间中 一切向量都可以分解成两个垂直方向向量的和 我们就是靠正交分解才建立了基的体系简化问题 这个内容是否可以在内积空间中进行推广
#### 正交
如果$(x,y)=0$ 称两个向量正交
如果两个集合$A,B$中的向量都两两正交 则称两个集合正交
如果对于内积空间$X$  ，$E$和$X$正交 则称$E$是$X$的一个正交集

根据正交的性质我们可以自然的推广出勾股定理
$$\text{设}X\text{是内积空间},x,y\in X,\text{ 若 }x\perp y,\text{ 则 }\parallel x+y\parallel^2=\parallel x\parallel^2+\parallel y\parallel^2.$$
他的逆定理有
在实内积空间上 满足勾股定理则知道两个向量正交（复内积空间则不一定）
推广这个定理有
$$\parallel a_{1}x_{1}+a_{2}x_{2}+\cdots+a_{n}x_{n}\parallel^{2}=\left|a_{1}\right|^{2}\parallel x_{1}\parallel^{2}+\left|a_{2}\right|^{2}\parallel x_{2}\parallel^{2}+\cdots+\left|a_{n}\right|^{2}\parallel x_{n}\parallel^{2} $$
**这个定理还是非常重要的，极化恒等式，平行四边形公式，勾股定理，是我们现在知道的仅有的抽象范数计算化简公式**
#### 正交补
正交补是对补集概念的一个拓展
设$X$是内积空间，$M\subset X$,记$M^{\perp}=\left\{x\mid x\perp M,x\in X\right\}$,则称$M^{\perp}$为子集 M 的正交补. 显然有 $X^{\perp}=\left\{0\right\},\left\{0\right\}^{\perp}=X,以及 M^{\perp}\bigcap M=\left\{0\right\}.$
由正交补的定义可得下列性质
* $\text{若}M\bot N,则M\subset N^{\perp}$
* $若M\subset N,则M^{\perp}\supset N^{\perp}$
* $M\subset\left(M^{\perp}\right)^{\perp}$
定理：
 $$\text{设 }X\text{ 是内积空间},{M\subset X,\text{则}M^\perp\text{是 }X\text{ 的闭线性子空间}} .$$
由于我们知道 完备度量空间中的子空间完备的要求是子空间闭
所以这个定理可以用于研究子空间 完备 Banach Hilbert属性的研究
Hilbert空间子空间的正交补一定是Hilbert空间
#### 正交分解
设$M$是内积空间$X$ 的子空间，$x\in X$,如果存在 $x_0\in M,z\in M^\perp$,使得 $x=x_0+z$,则称$x_{0}$为$x$在$M$上的正交投影或正交分解
**其实就是我们开头说的问题，分解为两个正交方向的向量**
引理
设 X 是内积空间，M 是 X 的线性子空间，$x\in\mathbb{X}$, 若存在 $y\in M$,使得
$||x-y||=d(x,M),那么 x-y\perp M.$
*引理只是为了用于下面投影定理的证明*
投影定理(Projection Theorem)
设$M$是 Hilbert 空间$H$上的闭线性子空间，则$H$ 中的元素 $x$ 在M 中存在唯一的正交投影 也就是 $\forall x\in H$  有$x=x_{0}+z$ 其中$x_{0}\in M,z\in M^\perp$
投影定理告诉我们 Hilbert空间加上闭线性子空间 就可以实现我们想要的分解 并且分解唯一
#### 子空间的直和
设 M 和 N 是线性空间 U 的两个子空间，称 $M+N=\{m+n\mid m\in M,n\in\mathbb{N}\}$ 为$M$ 与 N 的和(Sum)。如果 $M\bigcap N=\left\{\theta\right\}$,则称$\{m+n\mid m\in M,n\in N\}$ 为$M$ 与 N 的直和，此时记为
$$M\oplus N=\left\{m+n|m\in M,n\in N\right\},M\bigcap N=\left\{\theta\right\}.$$
根据投影定理知，若 M 是 Hilbert 空间 $H$ 上的闭线性子空间，则 $$H=M\oplus M$$
给出推论
$$\text{设 H 是 Hilbert空间, }M\subset H,\text{那么}M\text{ 是闭子空间当且仅当 }M=(M^\perp)^\perp $$
$$\text{设 }H\text{是 Hilbert空间},MH,\text{那么}M\text{是 }H\text{ 的稠密子集当且仅当 }M^\perp=\langle\theta\rangle.$$


### 内积空间的正交系
正交概念就是为了得到基
#### 标准正交基的定义
设 X 是内积空间，$E=\left\{e_{\lambda}|\lambda\in\Lambda\right\}$ 是 $X$ 的正交集(或正交系),其中 Λ 为指标集. 若$\forall e_{i},e_{i}\in E$满足
$$\left.\left(e_{i},e_{j}\right)=\left\{\begin{matrix}1,&i=j,\\0,&i\neq j,\end{matrix}\right.\right.$$
则称E为X中的标准正交基或标准正交系
**注意，标准正交基有两个要点，标准和正交，但是从未有完全**
性质：
* 内积空间任意标准正交基之间距离为$\sqrt{2}$
* 设$H$是可分的Hilbert空间 则他的任意标准正交基是可列集


#### 标准正交基的导出
我们在线性代数中介绍过了 任意的线性独立系都可以被正交化为标准正交基；反过来 标准正交基自然是一个线性独立系
**定理**： 设$X$是一个内积空间 $E$是他的标准正交基 $\left\{e_{n_{1}},e_{n_{2}},\cdots,e_{n_{k}}\right|\subset E$ 记
$$M=span\left\{e_{n_{1}},e_{n_{3}},\cdots,e_{n_{k}}\right\},$$

$\forall x\in X,x_{k}=\sum\left(x,e_{n_{k}}\right)e_{n_{k}}$是$x$在$M$上的正交投影 也就是$x_{k}\in M~x=x_k+z$  $\left(x-x_{k}\right)\perp M.$
这个定理告诉我们如何计算向量在这组基上的系数
#### Schmidt正交化
定理
若$\left\{x_n\right\}$为内积空间 X 中的任意一组线性独立系，则可将${x_n}$用格拉姆施密特(Gram-Schmidt)方法化为标准正交基$\{e_{n}\}$,且对任何自然数$n$,存在$\alpha_i^{(n)},\beta_k^{(n)}\in\mathbb{F}$,使得
$$
x_{n}=\sum_{k=1}^{n}\alpha_{k}^{\left(n\right)}e_{k},e_{n}=\sum_{k=1}^{n}\beta_{k}^{\left(n\right)}x_{k},
$$
同时 $span\{e_{1},e_{2},\cdots,e_{n}\}=span\left\{x_{1},x_{2},\cdots,x_{n}\right\}.$

正交化定理并不是最重要的；实际上重要的是我们如何进行正交化（从线性独立系得到正交基），下面既是这个正交化定理的证明 也是我们如何进行正交化的过程 非常重要
令$e_1=\frac{x_{1}}{||x_{1}||}$ 则有$M_1=span\{e_1\}$ 是第一步迭代的标准正交基
我们知道$x_2$一定可以针对$M_1$进行正交分解 得到$x_2=(x_2,e_1)e_1+v_2$
此时得到的$v_2$一定和目前迭代的标准正交基是正交的 因此可以给出
$e_2=\frac{v_{2}}{||v_{2}||}$  $M_2=span\{e_1,e_2\}$
重复这个过程 继续正交分解得到$v_{3}....v_n$ 然后做标准化 就可以得到一组
$$e_1,e_{2}.....e_n$$ 这就是我们得到的标准正价基
如果把他公式化就是以下的结果
$$\begin{aligned}e_{1}&=\frac{x_{1}}{\parallel x_{1}\parallel},e_{2}=\frac{x_{2}-(x_{2},e_{1})e_{1}}{\parallel x_{2}-(x_{2},e_{1})e_{1}\parallel},e_{0}=\frac{x_{3}-(x_{3},e_{1})e_{1}-(x_{3},e_{2})e_{2}}{\parallel x_{3}-(x_{3},e_{1})e_{1}-(x_{3},e_{2})e_{2}\parallel},\cdots,\\e_{n}&=\frac{x_{n}-(x_{n},e_{1})e_{1}-(x_{n},e_{2})e_{1}-\cdots-(x_{n},e_{n-1})e_{n-1}}{\parallel x_{n}-(x_{n},e_{1})e_{1}-(x_{n},e_{2})e_{2}-\cdots-(x_{n},e_{n-1})e_{n-1}\parallel},\cdots.\end{aligned}$$

### 傅立叶级数和收敛性
在数学分析中 我们就介绍过傅立叶级数的展开了 他广泛的用于各种应用领域，现在我们将傅立叶级数的工具推广的我们内积空间 特殊一点的就是Hilbert空间
#### 傅立叶级数的定义
设$e_n$为内积空间$X$的标准正交基，$x\in X$,则称级数
$$\sum\left(x,e_{k}\right)e_{k}=\sum c_{k}e_{k}$$

为$x$关于$e_n$的傅立叶级数，$c_{i}=\left(x,e_{i}\right)$为$x$关于$e_i$的傅立叶系数
#### 最佳逼近定理和贝塞尔不等式
设为$e_n$内积空间$X$的标准正交基$x\in X,c_k=\left(x,e_k\right),k=1,2,...,则对任何数组$ $\left\{\alpha_{1},\alpha_{2},\cdots,\alpha_{n}\right\}\subset\mathbb{F}$有
$$\left\|x-\sum_{k=1}^{n}c_{k}e_k\right\|\leq\left\|x-\sum_{k=1}^{n}\alpha_{k}e_{k}\right\|$$
最佳逼近定理告诉我们傅立叶级数展开的误差的特性

$$\text{设}e_n\text{为内积空间 }X\text{ 的标准正交基,则 }\forall x\in X,\text{有}\sum_{k=1}|(x,e_k)|^2\leqslant\|x\|^2.$$
贝塞尔不等式的几何意义是投影长度的平方和不会大于原始长度的平方和
他研究的是傅立叶级数系数平方和的特征
#### 收敛的充要条件
设${e_n}$为内积空间$X$ 的标准正交基 $x\in{X}$,则$x$ 关于${e_n}$的傅立叶级数$\sum\left(x,e_k\right)e_k$收敛于$x$ 的充要条件为
$${\parallel x\parallel^{2}=\sum_{k=1}^{\infty}\left|c_{k}\right|^{2},}$$
能看出 这就是贝塞尔公式取等的情况 我们称为帕塞瓦尔公式

事实上 这个取等只被一个因素影响 那就是我们的标准正交基取的数量够不够 只要取和空间维数相同的标准正交基 那么帕塞瓦尔公式就成立
#### 完全标准正交基
**定义** ：设$E=\{e_{\lambda}|\lambda\in\Lambda\}$ 是Hilbert空间H的一个完全标准正交基，则有$\forall x\perp e_{\lambda}$  则$x=0$ （解释什么时候正交基完全了）
引理：设$E=\{e_{\lambda}|\lambda\in\Lambda\}$ 是Hilbert空间H的一个完全标准正交基 $M=spanE$ 则有 $H=\overline{M}$
定理：设$H$是一个Hilbert空间 记$c_k=(x,e_k)$ 则以下命题等价
* $\{e_k\}$是$H$的完全标准正交基
* $\forall x\in H$ $x$关于$\{e_n\}$ 的傅立叶级数收敛
* $\forall x\in H$ $||x||^{2}=\sum\limits |c_k|^2$
性质：设$E=\{e_{\lambda}|\lambda\in\Lambda\}$ 是Hilbert空间H的一个标准正交基 则他是标准正交基当且仅当$E^{\perp}=0$
定理：任何非零的内积空间都有完全标准正交基

### Hilbert空间的同构
在线性赋范空间上 我们有线性等距同构 保证了线性性和范数不变 非常自然的我们可以引出内积空间上的线性等距同构
设$X_1,X_2$ 是同一个数域上$F$ 上的内积空间 如果存在一个一一映射$\phi$ 保证了
$$\begin{aligned}
&\varphi\left(\alpha x+\beta y\right)=\alpha\varphi\left(x\right)+\beta\varphi\left(y\right), \\
&\left(\varphi\left(x\right),\varphi\left(y\right)\right)=\left(x,y\right),
\end{aligned}$$
则称$X_1,X_2$ 线性等距同构
定理：设$H$是$n$维的Hilbert空间 则$H$和复内积空间$C^n$ 同构
定理：无限维的Hilbert空间$H$可分 当且仅当他有完全标准正交基
定理：若无限维的Hilbert空间$H$可分 则他和$l^2$同构
对于第一个定理：做正交化就可以了
对于第三个定理：我们已经知道了他有完全标准正交基 验证映射是否保证我们需要的线性和内积就好了

## 线性算子
算子 这个概念虽然我们我们以前基本没有接触 但是实际上应用的非常多；
任何一个线性赋范空间到线性赋范空间上的映射都被称为算子

所以我们前面学习的微分 积分 都是算子的一种 在这一章我们希望研究抽象的算子概念 但是为了让问题足够简单 我们先只研究线性算子
### 线性算子的定义和性质
#### 线性算子的定义
**定义（算子）**：
设 X 和 Y 是两个线性赋范空间，若 T 是 X 的某个子集 D 到 Y 中的一个映射，则称 $T$ 为子集 D 到 Y 中的算子，称 D 为算子 T 的定义城，记为 D(T); 并称 Y 的子集$R(T)=\{y|y=T(x),x\in D\}$为算子$T$的值域。对于$x\in D$,通常记$x$ 的像$T(x)$为$Tx$
特别的 如果$X=Y=R$则称算子为函数 如果$Y$是数域 则称算子为泛函 根据$Y$的实or复决定他们是实泛函还是复泛函
定义（连续算子）：
为D到Y中的算子， 设X和Y是两个线性赋范空间，$x_{0}\in D$ $\forall \varepsilon~ ~\delta >0$
对于任意的$x\in D$,当$\|x-x_0\|<\delta$时，有$\|Tx-Tx_0\|\leq\varepsilon$,则称算子$T$在点$x$ 处连续.若算子 T 在 D 中每一点都连续，则称 T 为 D 上的连续算子.
$f\left(x\right)$在$x_{0}$点连续等价于$\forall\left\{x_{n}\right\}\subset D$,若$x_{n}\to x_{0}$,则有$f\left(x_{n}\right)\to f\left(x_{0}\right)$
定义（线性算子）：
设X和Y是两个线性赋范空间，D$\subset X,T$为D 到Y 中的算子，如果$\forall x,y\in D$,
$$T\left(\alpha x+\beta y\right)=\alpha T\left(x\right)+\beta T\left(y\right)$$
定义（线性有界算子）：
设X和Y是两个线性赋范空间，D$\subset X,T:D\to Y$为线性算子，如果存在M>0,
$\forall x\in D,有\parallel Tx\parallel\leqslant M\parallel x\parallel$则称 T为 D 上的线性有界算子，
注意 算子有界的概念和函数有界并不一致 比如微积分中的函数$f(x)=x$ 明显在实数域$R$上是无界的 但是有
$$||f\left(x\right)||=||x||\leqslant M\left\|x\right\|,M=1$$
也就是无界的函数可能是有界的泛函

常用的算子举例
* 恒等算子$I$
* 零算子$0$
* 微分算子$T$和积分算子$T$
* 矩阵转置算子 $T$


#### 线性算子的性质
定理：线性算子在$D$上连续当且仅当在某一点连续
定理：线性算子是线性有界的当且仅当它把$D$中有界集映成有界集
定理：线性算子连续当且进行它是线性有界算子
定理：当$X$是有限维线性线性赋范空间的时候 线性算子一定是线性有界算子
**这意味着，有限维线性赋范空间定义域，线性算子，线性有界算子，线性单点连续算子，线性连续算子等价**

### 线性算子的零空间
#### 零空间的定义
设$X$和$Y$是两个线性赋范空间，称集合$\ker(T)=\{x\mid Tx=0,x\in X\}$为算子$T,X\to Y$ 的零空间或者算子T 的核(kernel)
容易证明 算子$T$的零空间一定是$X$的一个线性子空间

下面我们研究一下零空间非常常用的性质 他们基本上的核心只有一点 就是零空间闭
#### 零空间的性质
定理：
设$T$是线性赋范空间$X$上的线性有界算子，则零空间 ker(T)是 X 的闭线性子空间
**线性有界算子推出零空间闭**
请注意 这个命题的逆命题不成立 我们做了一个例子练习
定理：
设 X 是数域 F上的线性赋范空间，$f:X\to\mathbb{F}$为线性泛函，则映射 定理3.2.1
$G:X/\ker(f)\to\mathbb{F}$为线性连续泛函，其中$G([x])=G(x+\ker(f))=f(x)$,同时$G$ 是从商空间 $X/\ker\left(f\right)$ 到 $f$ 的值域 $R\left(f\right)\subset\mathbb{F}$上的线性同构映射.
*这个定理只是为了辅助下面的证明*
定理：
设 X 是数城F上的线性赋范空间，$f:X\to\mathbb{F}$为线性泛函，则 $f$ 为线性连续泛函**当且仅当**零空间 ker(f)是闭集
**线性连续泛函和零空间闭等价；**
定理：
设 X 是数域F上的线性赋范空间，$f:X\to\mathbb{F}$为非零线性泛函，则 $f$ 为连续泛函当且仅当零空间 ker(f)在 X 中非稠密.
也就是零空间如果稠密  那就不是线性连续泛函了
**最后和零空间的稠密联系起来了**
### 线性有界算子空间
算子是定义在空间上的，但是他未尝不可构成一个新的空间 从这一节开始我们研究 算子构成的空间
#### 定义
设$X$和$Y$是两个线性赋范空间，令$L(X\to Y)$表示从$X$ 到$Y$上所有线性算子的集合 也就是
$L\left ( X\to Y\right ) = \left \{ T\right | T$ 是 X → Y 的线 性 算 子 \}
我们可以验证 按照下面的方法定义加法和数乘 我们能确定线性算子构成一个线性算子空间
$$\begin{aligned}
&\left(T_{1}+T_{2}\right)\left(x\right)=T_{1}\left(x\right)+T_{2}\left(x\right), \\
&\left(\alpha T_{1}\right)\left(x\right)=\alpha T_{1}\left(x\right).
\end{aligned}$$
非常自然的 我们可以同样的定义线性有界算子空间
$$B\left(X\to Y\right)=\left\{T|T\text{ 是 }X\to Y\text{的线性有界算子}\right\}$$

这是一个线性空间了 非常自然的思想 能不能给他赋予范数得到线性赋范空间？ 下面我们来解释这一点
定义：
设$T\in B\left(X\to Y\right),T$的范数定义为
$$\parallel T\parallel\triangle\sup_{x\neq0}\{\frac{\parallel Tx\parallel}{\parallel x\parallel}\}$$
线性赋范空间 $B\left(X\rightarrow Y\right)$称为线性有界算子空间，特别记$B\left(X\right)=B\left(X\to X\right)$
#### 性质
下面给出关于上面定义的线性有界算子空间（一个线性赋范空间）的一些解释 这里的内容非常的重要 最后给出几个例子作为练习
当$X$和$Y$是两个线性赋范空间
* $T\in B\left(X\to Y\right)\text{当且仅当}\sup_{x\neq0}\{\frac{\parallel Tx\parallel}{\parallel x\parallel}\}\text{是有限值}.$
* $\text{通过}\parallel T\parallel=\sup_{x\neq0}\{\frac{\parallel Tx\parallel}{\parallel x\parallel}\}\text{定义的范数满足“范数”三条公理}.$
* $当x\in X时,\text{有}\left\|T\left(x\right)\right\|\leqslant\left\|T\right\|\cdot\left\|x\right\|.$
* $\left\|T\right\|=\sup_{x\neq0}\left\{\frac{\left\|Tx\right\|}{\left\|x\right\|}\right\}=\sup_{\left\|x\right\|\to1}\left\{\left\|Tx\right\|\right\}=\sup_{\left\|x\right\|\leq1}\left\{\left\|Tx\right\|\right\}.$


定理：
$$\text{设 }X\text{ 是有限维线性藏范空间,}Y\text{是任意的线性赋范空间,则}L\left(X\rightarrow Y\right)=B\left(X\rightarrow Y\right).$$
$$设 X 是线性服范空间,Y 是 Banach 空间,那么 B(X\to Y)是 Banach空间$$
这两个定理的含义还是很好理解的  前者是因为有限维线性赋范空间上的算子一定连续并且有界 后者是确实可以证明完备性
#### 投影算子
定义：
设M是Hilbert 空间 H 上的闭子空间，映射 $P:H\to M$ 定义为$\forall x\in H,P\left(x\right)=x_{0},x-Px=x-x_{0}\in M^{\perp}$
其中$x_{0}$是$x$在$M$上的正交投影，称P为M上的投影算子或正交投影算子(Orthographic Projection Operator),也记为 $P_M$
理解投影算子还是非常容易的 我们需要区分 这并不是在完全标准正交基上进行投影 而是一个普通的子空间

一些关于投影算子的性质
设M是Hilbert空间$H$上的非零闭子空间，P为${M}$上的投影算子，则
$\left(1\right)P$的零空间 $\ker\left(P\right)=M^{\perp}$, 值域 $R\left(P\right)=M.$
(2)P为H上的线性算子。
$\left(3\right)\parallel P\parallel=1.$

设M是Hilbert空间$H$上的非零闭子空间，P为M上的投影算子，则
$H=\ker\left(P\right)\bigoplus R\left(P\right)$
这个定理告诉我们 原空间是投影核核投影值域的直和
其思想是非常容易理解的

设 M是 Hilbert 空间 $H$ 上的非零闭子空间，P 为 M 上的投影算子，则
P为幂等算子
这个定理告诉我们投影算子幂等

设H是Hilbert 空间。$P\in B(H)~ker(P)\perp R(P)~ P=P^2$ 则$P$是投影算子
这就是把前面的两个定理反过来了
### 对偶空间和Risez表示定理
刚才我们研究了所有线性有界算子构成线性赋范空间 现在我们再来特殊一点
研究线性有界泛函$B(X\to F)$构成的线性赋范空间
#### 对偶空间
设$X$为一线性赋范空间，X上的全体线性有界泛函组成的集合$B(X-(E))$记为$X^{*}$,即
$$X^{*}=\left\{f|f:X\to\mathbb{F},f\text{为线性有界泛函}\right\}$$
称线性赋范空间 $X^*$为 X 的对偶空间或共轭空间(Conjugate Space)
为了方面后面的研究 规定一类函数
$$\left.\delta_{ij}=\left\{\begin{matrix}1,i=j,\\0,i\neq j.\end{matrix}\right.\right.$$
非常明显 对偶空间很复杂不适合我们研究 后面的工作就是简化对偶空间方便我们的研究
$$ 设 X 为n维线性赋范空间, \langle e _ 1 , e _ 2 , \cdots , e _ n \rangle 是 X\text{的基},\text{则存在其对偶空间的基}\{f_1,...f_n\}$$ 使得
$$f_{i}\left(e_{j}\right)=\delta_{ij}$$
我们知道 原本的初始空间是普通的线性赋范空间 这个定理告诉我们对偶空间的基（一堆线性有界泛函）映射原本空间基的结果

另一个定理
$$对偶空间X^*是Banach空间$$
并且给出
再线性等距同构的意义下
$$(R^n)^*=R^n~(C^n)^*=C^n$$

#### Risez 表示定理
设 H 为 Hilbert 空间，$f$是$H$ 上的线性连续泛函，则存在唯一的 $z\in H$,满足： $\forall x\in H$,有
$f\left(x\right)=\left(x,z\right),\left\|f\right\|=\left\|z\right\|.$

Risez表示定理的意义是
对于所有 Hilbert 空间  线性有界泛函总是可以转化为一个 Hilbert 空间  空间上的内积 并且有这个线性有界泛函的范数 就是 内积选中的那个点的范数
大大简化了我们研究 Hilbert 空间  上的线性有界泛函
### 算子乘法和逆算子
这里我们继续研究线性有界算子构成的线性赋范空间上的结构
#### 算子乘积
**定义 算子乘积**
设$X,Y,Z$是同一数域上的线性赋范空间
$T_{1}\in B(X\to Y),T_{2}\in B(Y\to Z),\forall x\in X$, 定义$(T_{2}T_{1})x\bigtriangleup T_{2}(T_{1}x)$,则称$T_{2}T_{1}$为$T_{1}$右乘以$T_{2}$,或者$T_{2}$左乘以$T_{1}$
我们很容易就能得到算子乘积的范数的性质
$$\begin{aligned}\quad\text{设 }X,Y,Z\text{ 是同一数域上的线性赋范空间,若 }T_1\in\mathbb{B}(X\to Y),T_2\in\mathbb{Q}(Y\to Z),\\T_2T_1\in\mathbb{B}(X\to Z),\parallel T_2T_1\parallel\leqslant\parallel T_2\parallel\parallel T_1\parallel.\end{aligned}$$
也就是算子乘积的范数小于范数的乘积

**定义** 可交换代数
设$X$ 是数域$F$上的一个线性空间，若对任意的元素$x,y,z\in X$ 及$\lambda\in\mathcal{F}$,存在的“乘法”满足$xy\in X,x\left(yz\right)=\left(xy\right)z,x\left(y+z\right)=xy+xz,\left(x+y\right)z=xz+yz,\lambda\left(xy\right)=x\left(\lambda y\right)$, 则称$X$为一个代数(Algebra).若存在一个非零元素 $e\in X,\forall x\in X$ 有 $ex=xe=x$,则称 e 为代数X 的单位元(Identity element).若$\forall x,y\in X$,有$xy=yx$,则称$X$为可交换代数。如果在线性赋范空间$X$的元素之间通过定义乘法使其成为一个代数，且$\forall x,y\in X$有$\|xy\|\leqslant\|x\|\|y\|$,则称$X$为赋范代数. 完备的赋范代数称为 Banach 代数(Banach Algebra).
我们规定赋范代数中的算子的一些性质
$$\left(T_{1}T_{2}\right)T_{3}=T_{1}\left(T_{2}T_{3}\right)$$
$$I:X\to X\text{ 为, I}x=x.\parallel I\parallel=1,\forall T\in B(X),有IT=TI=T.$$
$$P\left(T\right)x=a_{0}x+a_{1}Tx+a_{2}T^{2}x+\cdots+a_{n}T^{n}x.$$

分别是算子乘法结合 单位算子 算子多项式
#### 逆算子
设$X,Y$是同一数域上的线性赋范空间，且$T\in B(X\to Y)$,如果存在$S\in B(Y\to X)$,便得$ST=I_X,TS=I_Y$,则称 T 是可逆算子且 S 与 T 互为逆算子，记为 $T^{-1}=S$ ,其中，$I_{X}$ $I_{Y}$分别是$X,Y$上的恒等算子

容易证明，T 的可逆算子 S 唯一存在且($T^{-1})^{-1}=T$ 以及(一$T)^{-1}=-T^{-1}$,可逆算子的乘积也可逆
容易给出以下结论
* $\left(T^{-1}\right)^{-1}=T$
* $\left(ST\right)^{-1}=T^{-1}S^{-1}$
定理：
$$\begin{aligned}&\quad\text{设 }X\text{ 是 Banach 空间,若 }T\boldsymbol{\in}\boldsymbol{B}(X),\underline{\|T\|<1},\text{可逆}{(I-T),(T-I)}\\(I-T)^{-1}&=\sum_{i=0}^{\infty}T^i,(T-I)^{-i}=-\sum_{i=0}^{\infty}T.\end{aligned}$$
定理：
如果$X$是线性赋范空间，Y 是 Banach 空间，那么${B}(X\to Y)$中的所有可逆算子组成它的一个开集
### Baire纲定理
下面的连续两节 都是针对逆算子研究得到的定理
定理：
设$X,Y$是线性赋范空间，以及$T\in L(X\to Y)$,那么$T\in B(X\to Y)$当且仅当$\left\{x\in X\right|\parallel Tx\parallel\leqslant1 \}的内部为非空集$。

定义：
设 X 是度量空间，A드X,如果 A 的闭包的内部（内部：内点的集合）是空集，即$(\overline{A})^{\circ}=\phi$,则称 A 为稀疏集或疏朗集或无处稠密集(Nowhere Dense), 如果 A 可以表示成至多可数个稀疏集的并，即$A=\bigcup_{n}^{\infty}A_{n},A_{n}$是稀疏集，则称 A 为第一纲集； 不是第一纲集的集合，称之为第二纲集，
稀疏集的含义就是字面意思 它在任何开球中都不稠密
关于稀疏集的判断给出以下的性质
* $A$为稀疏集
* $\bar{A}$不含有任何点的邻域
* $\bar{A}^C$稠密
注意 虽然原集稀疏可以得到补集稠密 但是稠密集的补集不一定稀疏 不能尝试用这点判断一个集合是稀疏集
因此我们给出下面的常用稀疏集

欧几里得空间R"中的任一有限集是稀疏集，特别地，单点集{x}是稀疏集，所以旧中任一可列集是第一纲集。由于($\overline{\mathbb{Q}})^{\circ}=\mathbb{R}^{\circ}=\mathbb{R}$,所以R中的有理数集Q不是稀疏集。
由于$\{x\}=O(x,0.5)$是离散度量空间(X,$d_0$)中的单点集，所以{x}是第二纲集。
设 X 为度量空间，$x_0\in X$,如果存在邻域$O(x_0,\delta)$,使得$O(x_0,\delta)\bigcap X=\{x_0\}$,则称$x_{0}$是度量空间 X 的孤立点(Isolated Point).
再给出 一些性质
* 稀疏集的子集和闭包稀疏
* 有限个稀疏集的并稀疏
* 不含孤立点的度量空间 有限集都稀疏

$$\begin{aligned}\text{ 设 }X\text{ 是度量空间,}A\subseteq X,\text{那么 }A\text{ 是稀疏集的充要条件是对于任意开球}\\O(x,\epsilon),\text{ 存在 }O(y,r)\subset O(x,\epsilon),\text{使得 A}\bigcap Q(y,r)=\phi.\end{aligned}$$
$$\text{完备的度量空间(X,d)是第二纲集.}$$

### 开映射定理和逆算子定理
#### 开映射
设X，Y是线性赋范空间，若算子$T:X\to Y$能把$X$ 中的任何一个开集映射成Y中的开集，则称算子T为开映射
容易知道 离散度量空间上的所有集合都是开集 因此$Y$离散的度量空间 的时候 所有映射都是开映射；容易知道开映射的复合也是开映射
推论：
$$\begin{aligned}\text{设 }X&\text{ 是线性赋范空间, }A,B\subseteq X.\text{ 若 }x\in A^{\circ}\text{及 }y\in B^{\circ},\text{则}\\x+y&\in\left(A+B\right)^{\circ}.\end{aligned}$$

开映射定理：
$$\text{设 }X,Y\text{ 是 Banach 空间, 算子 }T\in B(X\rightarrow Y),R(T)=Y,\text{则 }T\text{ 为开映射}.$$
只要线性连续（有界就是连续）算子满射 则一定是开映射
#### 逆算子
逆算子定理：
设X，Y是Banach空间，算子$T\in\mathbb{B}(X\to Y)$,若算子 T 是满射（既单射又满射),则 $T^{-1}\in B\left(Y\to X\right).$
核心点在于Banach空间  一定要注意验证这一点
推论：
设线性赋范空间$X_{2}$上有两个范数$\|\cdot\|_{1}和\|\cdot\|_{2}$  这两个范数定义下均是Banach空间，而且$\|\cdot\|_{2}$比$\|\cdot\|_{1}$强)那么范数$\|\cdot\|_{1}和\|\cdot\|_{2}$ 等价
推论
(1)存在常数 $M>0$,使得 $\forall x\in D\left(T\right)$ 有$\parallel Tx\parallel\geq M\parallel x\parallel$,则 $T$ 可逆$T^{-1}\in B\left(R\left(T\right)\rightarrow D\left(T\right)\right),且\forall y\in R\left(T\right)有$
$\left\|T^{-1}y\right\|\leq\frac{1}{M}\left\|y\right\|.$
$$\begin{aligned}\text{如果}T^{-1}\text{存在且}T^{-1}\in B\left(R\left(T\right)\to D\left(T\right)\right),\text{则存在常数}M>0,\text{使得}\forall x\in D\left(T\right)\text{有}\\\left\|Tx\right\|\geqslant M\left\|x\right\|\end{aligned}$$
### 线性泛函延拓定理
设$M$为线性赋范空间$X$ 的子空间，对于定义在子空间$M$上的线性泛函$f\in M^{\circ}$,若存在空间$X$上的线性泛函$F\in X^*$,使得当$x\in M$时，$F(x)=f(x)$,则称F是$f$ 在X上的延拓，f是F在M上的限制，即记为$F|_{M}=f$
下面我们讨论这种延拓的存在性

**Hahn-Bananch延拓定理**
设M为线性赋范空间 X 的子空间，$f\in M^{*}$,则存在 $F\in X^{*}$,使得$F|_{M}=f$ 及
$||F||=||f||$
这就是我们最终给出的**Hahn-Bananch延拓定理** 只要是线性赋范空间 就可以进行延拓 并且延拓出来的算子范数和原本的算子的范数一致

推论：
设$X$为一线性赋范空间，对任何$x_0\in X$,$x_0\neq0$,则必存在$x$上的线性连续泛函$f$,满足$f(x_0)=\|x_0\|$以及$\|f\|=1.$

最后 任何线性赋范空间一定有完备空间
### 闭图像定理
我们在微积分里面就介绍过 函数的图像是一条曲线 如果函数是连续的 图像中对应的点的集合是一个闭集合 我们希望在线性赋范空间中研究类似的性质
#### 线性赋范空间中乘积空间
设$X$ 和$Y$是同一数域F上的线性赋范空间，记$X\times Y=\left\{\left(x,y\right)\left|x\in X,y\in Y\right\}\right.$,在$X\times$ Y上定义加法和数乘，具体如下：$\forall(x_1,y_1),(x_2,y_2)\in X\times Y$ 以及$\forall_a\in\mathbb{F}$,有
$(x_{1},y_{1})+(x_{2},y_{2})=\left(x_{1}+x_{2},y_{1}+y_{2}\right),\alpha\left(x_{1},y_{1}\right)=\left(\alpha x_{1},\alpha y_{1}\right)$,
那么$X\times Y$构成线性空间
设$x\in X,y\in Y$,其范数分别为$\|x\|,\|y\|$,于是在$X\times Y$上可定义范数
$$\begin{aligned}\|\left(x,y\right)\|_{p}&=\left(\|x\|^{p}+\|y\|^{p}\right)^{\frac{1}{p}},1\leq p<+\infty,\\\|\left(x,y\right)\|_{\infty}&=\max(\|x\|,\|y\|).\end{aligned}$$
最常用的是$p=1,p=2$和 无穷范数
明显的 可以验证乘积空间也是一个线性赋范空间 所有的相关性质都可以研究了
#### 闭线性算子
设$X$ 和$Y$是同一数域F上的线性赋范空间，若 T 的图像(Graph)
$G\left(T\right)=\left\{\left(x,y\right)\mid y=Tx,x\in D\left(T\right)\right\}$
是乘积空间$X\times Y$的闭子集，则称$T$为闭线性算子，简称闭算子

$T:D(T)(\subset X)\to Y$是线性有界第子，如果 D(T)是 X 的闭线性子空间、那么 T 为闭线性算子
这个定理 让$D(T)=X$ 那意思就是：
**线性有界算子是闭算子**
#### 闭图像定理
设 X 和 Y 都是 Banach 空间，$T:D(T)(\subset X)\to Y$ 是闭线性算子，D(T)是 X 的闭线性子空间，那么 T 为线性有界算子
这个定理 让$D(T)=X$ 那意思就是：
**闭算子是线性有界算子** 如果他是Banach空间上的算子

推论：
$$\begin{aligned}\textbf{}&\text{设 }X\text{ 和 Y 都是 Banach 空间, }T\in L(X\to Y),\text{那么 }T\text{ 为线性有界算子当且}\\\text{仅当 T 为闭算子. }\Box\end{aligned}$$
这个定理可太好用了 我们前面研究了大量的关于线性有界算子的问题，现在只要他是Banach空间 立刻就能得到他是闭算子
在别的情况下 闭和有界互不蕴含
### 一致有界定理
这里研究一簇算子有界的问题
设 X 和 Y 是同一数域F上的线性赋范空间，$F\subset B(X\to Y)$,如果$||T||~T\in F$是有界集，则称算子族F为一致有界

设 X 是 Banach 空间，Y 是线性赋范空间，算子族 $F\subset B(X\to Y)$ ,那么算子族 F 一致有界当且仅当$\forall x\in X,\left\{\parallel Tx\parallel\parallel T\in F\right\}$为有界集
**算子族一致有界的问题转化为映射后的结果有界**
这就是一致有界定理 也叫共鸣定理


### Banach不动点定理

#### 基本概念
##### 不动点
设$x$ 是一个非空集合，如果对于映射$A{:}X\to X$ ,存在$x^*\in X$满足$A(x^*)=x^*$,则称$x^*$为映射$A$的不动点（Fixed points）


如果已知映射，找不动点就是解方程，但是此时我们希望不解方程，研究不动点的存在性
##### 压缩映射
设$x\text{为度量空间，如果映射}_{A:X\to X}$,存在常数$\alpha\in(0,1)$, $\forall x,y\in X$,有$d(Ax,Ay)\leq\alpha d(x,y)$, 则称$A$为$X$上的压缩映射. 称常数$\alpha$为压缩系数(contraction coefficient).
明显的 它有以下的性质
* 压缩映射是连续映射
* 压缩映射的复合还是压缩映射
现在我们想知道一个问题 反复使用压缩映射作用一个点 会发生什么？
也就是$\text{记}x_n=A^n(x_0)\text{,那么点列}\{x_n\}\text{有什么特点}?$
#### Banach不动点定理
**Banach不动点定理**
设$X$ 是完备的度量空间，$A{:}X\to X$ 是压缩映射， 则$A$在$X$中具有唯一的不动点$x$’,使得$x^*=A(x^*).$
也就是说 对于完备的度量空间 压缩映射的不动点存在并且唯一
这个结论只是充分条件 并不是必要的
下面我们来研究这个定理的证明 其实就是寻找不动点的过程
*证明的总体思路就是利用压缩映射构造柯西列，利用完备性得到收敛点，再讨论收敛点是否为不动点*
压缩映射构造基本列：
$\text{任取}_{x_0\in X}\text{,构造点列}\left\{x_n\right\},\text{ 其中 }x_n=A(x_{n-1}).$
容易知道
$$\begin{aligned}
d(x_{n},x_{n+k})& \leq d(x_{n},x_{n+1})+d(x_{n+1},x_{n+2})+\cdots+d(x_{n+k-1},x_{n+k})  \\
&\leq(\alpha^{n}+\alpha^{n+1}+\cdots+\alpha^{n+k-1})c_{0}=\frac{\alpha^{n}(1-\alpha^{k})}{1-\alpha}c_{0}\leq\frac{\alpha^{n}}{1-\alpha}c_{0}\rightarrow0
\end{aligned}$$
所以压缩映射确实能得到基本列
给出收敛点：
因为 X 是完备的度量空间，所以基本列收敛 不妨设$x_{n}\rightarrow x^{*}(n\rightarrow\infty)$  易得
$$x^{*}=\lim_{n\to\infty}x_{n}=\lim_{n\to\infty}A(x_{n-1})=A(\lim_{n\to\infty}x_{n-1})=Ax^{*}.$$
证明唯一性
假设存在第二个不动点 则
$$d(x_{1}^{*},x^{*})=d(Ax_{1}^{*},Ax^{*})\leq\alpha d(x_{1}^{*},x^{*})$$
因此
$$(1-\alpha)d(x_{i}^{*},x^{*})\leq0,\text{从而}d(x_{i}^{*},x^{*})\leq0,\text{即}x_{i}^{*}=x^{*}$$
按照这个思路迭代就可以得到不动点了
