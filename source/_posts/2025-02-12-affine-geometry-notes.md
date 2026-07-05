---
title: "仿射空间几何学：仿射空间、坐标变换与几何量"
title_en: "Affine Geometry: Affine Spaces, Coordinate Transformations, and Geometric Quantities"
date: 2025-02-12 18:22:33 +0800
categories: ["Mathematics", "Geometry & Topology"]
tags: ["Learning Notes", "Mathematics", "Affine Geometry"]
author: Hyacehila
excerpt: "整理仿射空间、坐标变换、几何量、向量场、导数、曲线与曲面等内容。"
excerpt_en: "Covers affine spaces, coordinate transformations, geometric quantities, vector fields, derivatives, curves, and surfaces."
mathjax: true
hidden: true
permalink: '/blog/2025/02/12/affine-geometry-notes/'
---
本则笔记将作为[微分几何](/blog/2025/02/04/differential-geometry-notes/)中的一部分，由于其相对独立与我们主要探讨的欧式空间讨论，因此单独取出研究。

## 仿射空间几何学基础
本章开始，我们抛开欧式空间依赖的角度与长度的概念，专注与一套只依赖线性空间结构的概念，为此我们将引入一套新的语言，他和流形导论将会紧密的联系起来。
### 仿射空间的直线,平面,坐标与坐标变换
#### 仿射空间与其上的直线与平面
几何学是研究点集的学科，我们首先需要构造一个 点 构成的空间；普通的向量空间不能够直接作为几何意义下的点集，因为他存在特殊的零元，而几何中的点集每个点都有着相同的地位。

因此，我们需要一个更好的空间，他的思路是**去除向量空间的中心化**

向量源自物理学中的位置矢量，规定为同时具有大小和方向的量，但是他的大小与方向都被认为规定的0点影响，而很自然的，**位置向量的差与零点的选择无关** 这个思路让我们提出了仿射空间的概念

定义：设$\mathscr{A}^{n}$是一个非空集合，若存在映射
$$A^{n}\times A^{n}\mapsto\mathbb{R}^{n},\left(A,B\right)\mapsto\overrightarrow{AB}$$
满足
* $\forall\nu\in\mathbb{R}^{n},A\in\mathscr{A}^{n},\text{存在唯一的}B\in\mathscr{A}^{n}\text{使得}\overrightarrow{AB}=\nu$
* $\overrightarrow{AB}+\overrightarrow{BC}=\overrightarrow{AC}.$
则称$\mathscr{A}^{n}$是$n$维仿射空间

**这个定义完全根据前面给出的位置向量的思路来给出，保证差的确定性而抛弃零点的选择，只是还没有坐标化**

定 义: 三维仿射空间 $\mathscr{A} ^{3}$的非空子集$l$,含有点${A} \in l$,若满足 $\exists \nu \in \mathbb{R} ^{3}, \nu \neq 0$,使得
$$l=\left\{B\in A^{3}|\overrightarrow{AB}=k\nu,k\in\mathbb{R}\right\},$$
则称$l$为过$A$ 以$\nu$ 为方向的直线，$\nu$称为$l$ 的方向向量.

定义:三维仿射空间 $\mathscr{A}^3$的非空子集 $\pi$,含有 $A\in\pi$,满足：$\exists\nu,w\in\mathbb{R}^3$,线性无关$$\pi=\left\{B\in A^{3}|\overrightarrow{AB}=kv+pw,\left(k,p\right)\in\mathbb{R}^{2}\right\},$$则称$\pi$为一个由$\nu$ 、$w$张成的过$A$的平面 ,($\nu,w)$称为$\pi$的方向向量对.

**有了仿射几何的基本定义以后，我们可以轻松的证明以前在平面几何中使用公理化几何体系证明那些不涉及角度与长度概念的命题，即仿射性质可证**

#### 仿射空间的坐标
我们需要引入参考点，从而使用实数来描述仿射空间上的性质

定义(仿射坐标系)：在 $x^n$上选定一个点 $O$(称为原点),在$R^n$上选定一组基 $\left \{ e_{i}\right \} _{( i= 1, 2, \cdots , n) },$  我们记$A= \left \{ O, e_{i}\right \}$为 $x^{n}$上的一个仿射坐标系,并有以下双射 :
$$\varphi_{A}=A^{n}\mapsto\mathbb{R}^{n}$$
$$A\mapsto\left(x^{1},x^{2},\cdots,x^{n}\right)$$这里$\overrightarrow{OA}=\sum_{i=1}^{n}x^{i}e_{i}.$

考虑两个仿射坐标系  $\left \{ O, e_{i}\right \},\left\{O^{\prime},f_{i}\right\}$  他们满足
$$\overrightarrow{OO^{\prime}}=\sum_{i=1}^{n}a^{i}e_{i},e_{i}=\sum_{i=1}^{n}T_{i}^{i}f_{i}$$
也就是
$$(e_1\quad e_2\quad\cdots\quad e_n)=(f_1\quad f_2\quad\cdots\quad f_n)\begin{pmatrix}T_1^1&T_2^1&\cdots&T_n^1\\T_1^2&T_2^2&\cdots&T_n^2\\\vdots&\vdots&&\vdots\\T_1^n&T_2^n&\cdots&T_n^n\end{pmatrix}.$$
则有仿射坐标变换（原本的坐标为$x_i$，则新坐标系下的坐标$y_i$的形式）为
$$y^{i}=\sum_{j=1}^{n}T_{j}^{i}\left(x^{j}-a^{j}\right).$$

定义（仿射坐标系定向）：当仿射坐标变换的矩阵$det(T)>0$时，称为两个坐标系同向，当$det(T)<0$，称为两个坐标系具有相反的定向。

#### 直线与平面的坐标
为了减少求和符号的使用，我们约定下面的符号
$$a^{i}e_{i}=\sum_{i=1}^{n}a^{i}e_{i}$$

对于直线的坐标，在建立坐标系后，我们容易知道直线上点$A$满足
$$\overrightarrow{OA}=a^{1}e_{1}+a^{2}e_{2}+a^{3}e_{3}=a^{i}e_{i}$$
而直线自身的方向向量容易分解为
$$v=v^{1}e_{1}+v^{2}e_{2}+v^{3}e_{3}=v^{i}e_{i}$$
那么对于直线上的任意一点（也就是整条直线）有
$$\overrightarrow{OX}=\overrightarrow{OA}+\overrightarrow{AX}$$
那么有
$$x^{i}e_{i}=a^{i}e_{i}+kv^{i}e_{i}\Rightarrow x^{i}=a^{i}+kv^{i},k\in\mathbb{R}$$
同理，对于平面有
$$\overrightarrow{OX}=\overrightarrow{OA}+\overrightarrow{AX}$$
即
$$x^{i}e_{i}=a^{i}e_{i}+kv^{i}e_{i}+pwe_{i}\Rightarrow x^{i}=a^{i}+kv^{i}+pw^{i},k,p\in\mathbb{R}$$
### 几何量
在进一步的讨论前，我们简单讨论一下几何量的概念

几何学中的量(几何量)都应该是仅仅依赖点或者几何对象，否则这些量就不能成为几何学研究的对象.然而在具体定义或者计算这些量的时候，我们又不可避免地通过点在某个具体坐标系内的坐标值来进行.那么就产生了一个问题：具体给定一个由坐标定义的量，如何确定它是不是一个几何量。

考虑一个三维仿射空间 $\mathscr{A}^3$,带有坐标系 $\mathscr{A}=\left\{O,e_1,e_2,e_3\right\}.$在其中我们引入一条直线，方程写作
$$\left(x^{1},x^{2},x^{3}\right)=\left(x_{0}^{1},x_{0}^{2},x_{0}^{3}\right)+t\left(v^{1},v^{2},v^{3}\right),$$
这里$(v^1,v^2,v^3)$是一组给定的数(对应一个给定向量).同时记
$$(x_{1}^{1},x_{1}^{2},x_{1}^{3})=(x_{0}^{1},x_{0}^{2},x_{0}^{3})+t_{1}(v^{1},v^{2},v^{3}),$$
$$(x_{2}^{1},x_{2}^{2},x_{2}^{3})=(x_{0}^{1},x_{0}^{2},x_{0}^{3})+t_{2}(v^{1},v^{2},v^{3})$$
考虑两个根据坐标定义的实数
$$\eta:=\sqrt{\sum_{i=1}^{3}\mid x_{1}^{i}-x_{0}^{i}\mid^{2}}=t_{1}\parallel v\parallel=t_{1}\sqrt{\sum_{i=1}^{3}\mid v^{i}\mid^{2}}$$
$$\lambda:=\sqrt{\frac{\sum_{i=1}^{3}\mid x_{2}^{i}-x_{1}^{i}\mid^{2}}{\sum_{i=1}^{3}\mid x_{1}^{i}-x_{0}^{i}\mid^{2}}}=\frac{t_{2}-t_{1}}{t_{1}}.$$
我们可以证明，$\eta$不是仿射几何意义下的几何量，而$\lambda$是，因为前者在仿射坐标变换下会发生改变。也就是说， **仿射坐标系下的几何量，要对仿射坐标变换保持不变性，即不依赖仿射坐标系的选取**

### 仿射空间的拓扑与实标量场
在本节中我们进一步给出一个常见几何/物理对象，称为实标量场.这类量有丰富的物理背景：温度场、电势场、密度场......它们的特点是，**空间中的每一个点都给定一个实数值.这种“点到实数”的映射称为实标量场或者函数**

以后我们将会研究更加复杂的“点到向量空间”“点到张量空间”的映射， 它们对应更加复杂的向量场本文“仿射空间的向量值函数与向量场”部分、张量场本文“余向量场”部分。

为了精确地描述实标量场的正则性(连续、可微和光滑),我们将首先讨论仿射空间中的拓扑.我们将看到仿射空间上的拓扑虽然是通过仿射坐标系定义的，但是各个坐标系导出的拓扑是相同的，所以这个拓扑并不依赖于具体仿射坐标系的选择，因而是仿射几何对象。

为了理清几个容易混淆的概念，**我们不计划如通常那般将 $A^n$同$\mathbb{R}^n$等同**. 这可能会带来书写上的一点麻烦。但是我们这么做是为了强调；几何学中研究的概念都必须是不依赖于坐标系选取的.但是很多概念却是通过坐标系来定义的.**区分 $A^n$和$\mathbb{R}^n$是为了区分“几何量”和“几何量在坐标系下的计算公式” 这两个概念。**

#### 仿射空间拓扑
我们在本节将拓扑概念引入仿射空间，这需要[点集拓扑学](/blog/2024/10/16/point-set-topology-notes/)基础

定义()：设 $\mathcal{A}=\left\{O,e_i\right\}$为 $n$ 维仿射空间 $\mathcal{A}^n$的一个仿射坐标系.一个集合$U\subset\mathcal{A}^n$称为开集，如果它在坐标映射 $\varphi_{A}$下的像是$\mathbb{R}^n$上的开集。

**这就是研究拓扑学的最基本结构，空间上的开集；我们从仿射坐标系的定义中获取了开集的定义灵感**

这个定义不够合理，因为开集的定义依赖于仿射坐标的选取，我们不加证明的给出，这个**开集定义保持拓扑变换不变性同时满足拓扑公理**，因此仿射空间是拓扑空间。

后面我们还会有很多地方基于坐标讨论，最后再拓展了解到，他对坐标变换保持不变性

#### 仿射空间上的函数与标量场
定义： $n$ 维仿射空间 $\mathcal{A}^n$ 上的连通开集为开区域

定义：设$U$是 $\mathcal{A}^n$ 上的开区域，则一个从$U$到$R$的映射称为函数或实标量场

定义 (从仿射坐标系看函数)：设$U$是 $\mathcal{A}^n$的一个开区域.考虑$U$ 上的函数 $f$,称 下面定义在$\mathbb{R}$开区域上的 $n$ 元函 数 $f \circ \varphi _{\lambda }^{- 1}$ 为 从 坐 标 系 $\mathcal{A} = \{ O,e_i\}$中 读 取 $f.$
$$V\xrightarrow{\varphi_{A}^{-1}}U\xrightarrow{f}\mathbb{R};~~~~(x^{1},x^{2},\cdots,x^{n})\longmapsto A\longmapsto f\left(A\right).$$
**也就是说，我们先从坐标空间利用坐标逆映射到拓扑空间，再同拓扑空间得到函数映射出的标量，这就是标量场，从拓扑空间到一个标量的映射，为空间的每一个点指定一个标量**

定理 （连续函数的判定).设$U$ 是 $\mathscr{A}^n$上的开区域，$f$ 为$U$上的函数，
则以下三点相互等价：
* $f$连续；
* 在某一个仿射坐标系 $\mathcal{A}=\left\{O,e_i\right\}$中读取 $f$ 得到一个$n$ 元连续函数；
* 在任意一个仿射坐标系中读取$f$都得到一个$n$元连续函数

定义：设$U$ 是 $\mathscr{A}^n$上的开区域，$f$ 为$U$ 上的函数.$f$ 称为可微的，若存在某个仿射坐标系 $\mathcal{A}=\left\{O,\boldsymbol{e}_i\right\}$,使得从该坐标系中读取 $f$ 是一个$n$ 元可微函数。

可微的定义可以自然拓展到坐标变换不变，不难证明

**最后，我们给出结论，在$n$维仿射空间中，只要存在一个仿射坐标系$A$下讨论清楚了函数（标量场）的性质，那么在其他任何坐标系下，性质都是一样**

### 广义坐标系
坐标就是给空间上的点起名字的过程，我们定义的仿射坐标系的双射要求较高（逐点双射），某些类似与坐标的概念不满足这个要求，如常用的极坐标系，因此我们这里拓展拓扑坐标系的概念，并进行简单的讨论

定义：设$U$ 是 $\mathscr{A}^n$上的开区域.{y}$^i\}_i=1,2,...,n$是定义在$U$ 上的一族$C^\infty$
函数，满足：
$$\varphi_{U}:U\mapsto\mathbb{R}^{n},\\A\mapsto\left(y^{1}\left(A\right),y^{2}\left(A\right),\cdots,y^{n}\left(A\right)\right).$$
设$\mathcal{A}^n$上带有仿射坐标系$\left\{O,e_i\right\}$,自变量记为$\left\{x^i\right\}_i=1,2,...,n$,坐标映射
$$\varphi_{A}:A^{n}\mapsto\mathbb{R}^{n},\\A\mapsto\left(x^{1}\left(A\right),x^{2}\left(A\right),\cdots,x^{n}\left(A\right)\right).$$
若
* $\varphi_{U}:U\mapsto\varphi_{U}\left(U\right)为双射$
* $\varphi_{U}\circ\varphi_{A}^{-1}:\varphi_{A}\left(U\right)\mapsto\varphi_{U}\left(U\right)与\varphi_{A}\circ\varphi_{U}^{-1}:\varphi_{U}\left(U\right)\mapsto\varphi_{A}\left(U\right)都是C^{\infty}映射$
那么称$\{U,\phi_U\}$是定义在$U$上的一个广义坐标系

我们给出下面的性质，建立仿射坐标系和广义坐标系的联系

性质：设$U$是$\mathcal{A}^n$上一个开区域，带有广义坐标系$\{U,\phi_U\}$ 则
* $U$中开子集在$\varphi_{U}$下的像是$R^n$中开子集，反正也成立
* 设$f$是$U$上定义的标量场，则$f$连续等价于 $f$ $\varphi _{U}^{- 1}$ 是 $\varphi _{U}( U)$上的连续函数

性质：设$U$是$\mathcal{A}^n$上一个开区域，$f$为$U$上的函数，则下列命题等价，
* $f$是可微的(在某个仿射坐标系中);
* $f$在某个广义坐标下读取得到可微函数；
* $f$在任意广义坐标下读取得到可微函数

## 仿射空间的向量值函数与向量场
### 点上向量空间
本小节开始我们讨论仿射空间的向量值函数与向量场，首先，我们讨论作为有向线段的向量，与后面研究作为方向导数的向量形成对比

定义：设.$\mathscr{A}^n$为一个$n$ 维仿射空间 $,A\in\mathscr{A}^n$为其上一个点.定义
$$T_{A}=\{(A,B)\mid B\in A^{n}\}$$
在$T_A$上定义加法：$\left(A,B\right)+\left(A,C\right)=\left(A,D\right)$,使得$$\overrightarrow{AD}=\overrightarrow{AB}+\overrightarrow{AC}$$定义数乘：$\lambda \left ( A, B\right ) = \left ( A, C\right )$, 使得 $$\overrightarrow {AC}= \lambda \overrightarrow {AB}.$$将如上定义的线性空间($T_{A},+,\cdot$)称为$A$点的向量空间

直观来看，上面的定义描述了以$A$为起点的有向线段的集合以及他们上面定义的加法（平行四边形法则）与数乘（同方向延长若干倍）

从线性空间的角度看，上述定义在仿射空间的每一个点上定义了一个$n$维向量空间，不同点的向量空间上的向量有着不同的起点，因此每个向量空间中的向量都不尽相同。

当然，在仿射空间中，由于全局仿射坐标系的存在，不同点的向量空间之间可以定义一个叫做平移的关系

设 $A,B\in\mathscr{A}^{n},A\neq B.$  设$u=\left(A,C\right),v=\left(B,D\right)$分别为$T_{A},T_{B}$中的向量.若
$$\overrightarrow{AC}=\overrightarrow{BD}$$
我们称$u,v$互为对方的平行移动，简称平移。

在给定仿射坐标系后，我们可以仿射空间中的每个点都写成分量形式，设$A=\left\{O,e_{i}\right\}$是$\mathcal{A}^n$上的仿射坐标系，若$v=\left(A,B\right)\in T_{A}$且
$$\overrightarrow{AB}=v^{i}e_{i}$$
我们称$\left(v^{1},v^{2},\cdots,v^{n}\right)$是$v$在仿射坐标系$A$下的分量形式
### 仿射坐标系下的向量场
本节引入向量场的概念，和本文“仿射空间上的函数与标量场”部分类似，本节我们给空间上的每个点指定指定一个向量。


定义：设$U\subset\mathscr{A}^n$为仿射空间上的区域，$U$ 上定义的向量场是一个映射：
$$\begin{matrix}
X:U\mapsto\mathcal{A}^n \\
A\mapsto B
\end{matrix}$$
直观的解释是：我们在$U$上的每一个点$A$都指定了一个$T_A$上的向量，也就是一个以$A$为起点，额外指定终点$B$；**向量场就是把区域中的每一个点都指定一个$\mathscr{A}^n$中的点作为终点。从而每个点都对应了一个向量。**

在给定仿射坐标系后，向量场$X$的每个点的向量都可以有分量形式写出，我们记向量场为
$$X\left(A\right):=\overrightarrow{AB}\in\mathbb{R}^{n}$$
如果有$\overrightarrow{AB}=X^i\left(A\right)e_i$ 则称$\left(X^{1},X^{2},\cdots,X^{n}\right)$是向量场在坐标系下的分量形式，也就是如下的区域到向量的映射
$$\begin{aligned}&U\mapsto\mathbb{R}^{n},\\&A\mapsto\left(X^{i}\left(A\right)\right)_{i=1,2,\cdots,n}.\end{aligned}$$

很容易想象到，在不同的仿射坐标系下，一个向量场的分量形式会发生变换（标量场不会发生这个问题，在更换坐标系的时候，我们只变换函数$\phi_A$，而从空间到实数的映射与坐标的选取无关）因此我们需要建立向量场的分量形式在坐标变换下的公式

定理（向量场分量形式的坐标变换）：假设在$\mathcal{A}^n$上有两个仿射坐标系  $\left \{ O, e_{i}\right \},\left\{O^{\prime},e_{i}^{\prime}\right\}$  他们满足
$$\overrightarrow{OO^{\prime}}=a^{i}e_{i},e_{i}=T_{i}^{j}e_{j}^{\prime}$$
设向量场 $X$ 在 $\mathcal{A}$之下的分量形式为$\left(X^i(A)\right)_i=1,...,n$,在 $\mathcal{A}^\prime$下的分量形式为$\left(X^{\prime i}\left(A\right)\right)_{i=1},\cdots,n$ 则
$$X^{\prime i}\left(A\right)e_{i}^{\prime}=X=X^{j}\left(A\right)e_{j}=X^{j}\left(A\right)T_{j}^{k}e_{k}^{\prime}.$$
因此有
$$X^{\prime i}=T_{j}^{i}X^{j}\left(A\right).$$

向量场的正则性（连续，可微，光滑）根据在某个仿射坐标系下，从$A\to X^i(A)$的正则性决定，也就是标量场中的$f$ 我们容易证明：**在某个仿射坐标系下有某正则性，则在所有仿射坐标系下都具备该正则性**


### 广义坐标下的向量场
#### 自然标架场
我们可以在两个不同的坐标系下研究速度，如极坐标系与标准直角坐标系，这就产生了两个不同的速度，本节讨论这两个速度之间的不同之处与联系。

不仅仅针对极坐标系成立，对一般开区域上定义的广义坐标系也可以这样研究， 更一般地，我们做下列计算：

假设$U\subset\mathscr{A}^n$为一个开集，上面定义了广义坐标系$\left\{U,\varphi_{U}\right\}$,自变量记为$\left\{y^i\right\}_i=1,...,n.$（也有一个拓扑坐标系，用$x^i$表示）

那么对于一个$U$上运动的质点$P$,它的运动方程在 $x^i$ 下写作$(x^i(t))_i=1,...,n$,在 $y^i$ 下写作$\left(y^{i}\left(t\right)\right)_{i=1,\cdots,n}$那么：
$$\frac{dr}{dt}\left(t\right)=\dot{x} ^{i}\left(t\right)e_{i}=\frac{\partial x^{i}}{\partial y^{j}}\left(y\left(t\right)\right)\frac{dy^{j}}{dt}\left(t\right)e_{i}=\dot{y} ^{i}\left(t\right)\cdot\frac{\partial x^{j}}{\partial y^{i}}\left(y\left(t\right)\right)e_{j}$$
因此对于$P$轨迹上的一个点$A$我们定义
$$\sigma_{i}\left(A\right)=\frac{\partial x^{j}}{\partial y^{i}}|_{\varphi_{U}\left(A\right)}e_{j}$$
称为广义坐标到${y^i}_{i=1,\cdots,n}$ 的自然标架

**在广义坐标系下，每个点的自然标架一般是不同的，因此形成了点到标架的映射，我们称为自然标架场**

**在自然标架场中我们易来了具体的仿射坐标系，实际上在所有仿射坐标系中，这个定义都是有效的**

定理（标架变换公式）：设$\left\{U,\varphi_{U}\right\}和\left\{V,\varphi_{V}\right\}$是仿射空间$\mathscr{A}^n$中的两个广义坐标系，其坐标自变量分别记为$\left\{y^i\right\}_{i=1,...,n}$,$\left\{z^i\right\}_{i=1,...,n}$,记$\left\{\boldsymbol{\sigma}_{i}\right\}_{i=i,\cdots,n},\left\{\boldsymbol{\tau}_{i}\right\}_{i=i,\cdots,n}$为其对应的自然标架.设$A\in U\cap V$,那么
$$\tau_{i}\left(A\right)=\frac{\partial y^{k}}{\partial z^{i}}|_{\varphi_{V}\left(A\right)}\sigma_{k}\left(A\right).$$
#### 向量场在自然标架下的分量
我们自然的想把向量场写在自然标架下

定义：设$X:U\mapsto\mathscr{A}$为区域$U$上定义的向量场，设$\langle U,\varphi_{\upsilon}\rangle$为$U$上定义的一个广义坐标系，其坐标变量记为${y}^{i}_{i=1,\ldots,n}$ ,其自然标架场记为 $\sigma_{i}$.那么如果
$$X\left(A\right)=X^{i}\left(A\right)\sigma_{i},$$
我们称$\left(X^{i}\left(A\right)\right)_{i=1,\cdots,n}$为$X$在$\left\{U,\varphi_{\upsilon}\right\}$之下的分量形式

**和普通仿射坐标系一样，如果一个向量场在某个广义坐标系下满足某正则性，他在其他仿射坐标系与其他广义坐标系下也满足该正则性质**

定理：设$\{U,\varphi_{U}\}$和 $\{V,\varphi_{V}\}$是仿射空间 $\mathscr{A}^n$中定义的两个广义坐标系，其坐标自变量分别记为$\left\{y^i\right\}_{i=1,...,n},\left\{z^i\right\}_{i=1,...,n}$  记$\left\{\sigma_i\right\}_i=1,...,n,\left\{\tau_i\right\}_{i=1,...,n}$为其对应的自然标架.设$X$为定义在$U\cap V$上的向量场，设 $A\in U\cap V.$如果记$\left(Y^{i}\right)_{i=1,\ldots,n}$为$X$ 在$\left\{U,\varphi_{U}\right\}$下的分量形式$,\left(Z^{i}\right)_{i=1,\ldots,n}$为$X$ 在$\left(V,\varphi_{V}\right)$下的分量形式，那么
$$Y^{i}\left(A\right)=\frac{\partial y^{i}}{\partial z^{j}}|_{\varphi_{v}\left(A\right)}Z^{j}\left(A\right).$$

### 作为方向导数的向量
#### 方向导数的引入
我们可以从另一个观点来审视向量这个概念.

现在假设在$n$维仿射空间开区域$U$的一个固定点 $A$上我们定义了一个向量 $v=(A,B),\overrightarrow{AB}=v^ie_i$ (在一个给定的仿射坐标系下).

我们考虑$U$上的可微函数 $f$(从仿射坐标系中读取，看成$\mathbb{R}^{n}$开集上的函数，记为 $f_A=f\circ\varphi_A^{-1}).$一般情况下我们可以对 $f$ 求方向导数：
$$\partial_{r}f\left(A\right):=\frac{\partial f_{A}}{\partial x^{i}}\left|_{\varphi_{A}\left(A\right)}v^{i}=\frac{\partial\left(f\circ\varphi_{A}^{-1}\right)}{\partial x^{i}}\right|_{\varphi_{A}\left(A\right)}v^{i}.$$
我们希望把这个操作扩展到广义坐标系，取广义坐标系$\{U,\varphi_{U}\}$ 其坐标变量记为${y}^{i}_{i=1,\ldots,n}$ 那么
$$v=w^{i}\sigma_{i},\sigma_{j}=\frac{\partial x^{i}}{\partial y^{j}}e_{i}.$$
从而
$$v^{i}=\frac{\partial x^{i}}{\partial y^{j}}w^{j}.$$
考虑$f$在广义坐标系下的读取$f_U$则
$$\begin{aligned}&f_{U}:\varphi_{U}\left(U\right)\mapsto\mathbb{R},\\&\left(y^{i}\right)_{i=1,\cdots,n}\mapsto f\circ\varphi_{U}^{-1}\left(y^{i}\right)=\left(f\circ\varphi_{A}^{-1}\right)\circ\left(\varphi_{A}\circ\varphi_{U}^{-1}\right)\left(y^{i}\right).\end{aligned}$$
我们有下面的观察
$$w^{i}\frac{\partial f_{U}}{\partial y^{i}}|_{\varphi_{U}\left(A\right)}=\frac{\partial f_{A}}{\partial x^{j}}|_{\varphi_{A}\left(A\right)}\frac{\partial x^{j}}{\partial y^{i}}|_{\varphi_{U}\left(A\right)}w^{i}=\frac{\partial f_{A}}{\partial x^{i}}|_{\varphi_{A}\left(A\right)}v^{i}=\partial_{v}f\left(A\right).$$

**也就是说，$f$在任何一个广义坐标系下读取$f_U$求取方向导数，得到的结果是一样的，也就是说方向导数是点上的向量自身的性质，与坐标系无关**
#### 导算子
前面定义的方向导数有着一些很亮好的性质，我们抽象得到下面的定义（前面研究的$\partial_{v}f\left(A\right)$是光滑函数到实数的映射，我们就从此开始讨论）

定义：设$A\in\mathscr{A}^n$为仿射空间中的一个点，设 $\mathcal{F}_{\mathrm{A}}$ 为在$A$点附近有定义的光滑函数集合.称 $D$为 $A$ 点的一个导算子，若 $D: \mathcal{F}_{\Lambda}\to\mathbb{R}$满足：
* 局部性：对于$f$ 、$g$定义在$O$邻域上的两个光滑函数，满足：若存在 A的邻域$U$,在$U$上$f=g$,则$Df=Dg.$
* 线性性：$\forall\alpha,\beta\in\mathbb{R}$, $D\left(\alpha f+\beta g\right)=\alpha Df+\beta Dg.$
* 莱布尼兹公式：$D\left(fg\right)=f\left(A\right)Dg+g\left(A\right)Df.$

**特别的，如果$f$在$A$点的一个开邻域内是常数，则$Df=0$**

我们可以根据下面的结论把导算子和向量联系起来

定义：设$\mathscr{A}^n$上有一个仿射坐标系$\mathcal{A}=\left\{O,e_i\right\}.$设$A\in\mathscr{A}^n,\mathscr{D}$为$A$点一个导算子，则存在 $T_{\mathrm{A}}$ 上的唯一一个向量($A,B)$,使得$\overrightarrow{AB}=v^ie_i$,并且
$$Df=\partial_{\nu}f\left(A\right),\forall f\in\mathcal{F}_{A}.$$

**注意，我们在本文“方向导数的引入”部分第一次从向量引入了方向导数的概念，又在本文“导算子”部分独立的定义了导算子的概念，现在我们发现，定义一个点上的导算子和定义这个点的向量是等同的，只是后者不再依赖于具体的坐标系。因此我们也把某点所有导算子构成的集合称为该点的向量空间**
#### 自然标架场与导算子
虽然导算子看起来非常抽象，但是他可以直观的解释很多计算法则，比如本节希望再次讨论的自然标架场

根据以上从方向导数看向量的观点，我们可以引人以下记号.设$\left(U,\varphi_{\upsilon}\right)$是一个广义坐标系，$\left(y^{i}\right)_i=1,...,n$为其自变量.我们记$\left\{\partial_{i}\right\}_{i=1,\cdots,n}为$
$$\partial_{i}f:=\frac{\partial\left(f\circ\varphi_{U}^{-1}\right)}{\partial y^{i}}.$$

可以验证在$U$ 的每个点上，这是一个导算子，同时我们可以验证，设 $A=\left\{O,\right.$ $e_i\}$为一个仿射坐标系，自变量为$(x^i)_i=1,...,n$.将$f$在$\mathcal{A}$上的读取记为$f_\mathcal{A}$,则
$$f\circ\varphi_{U}^{-1}=f\circ\varphi_{A}^{-1}\circ\left(\varphi_{A}\circ\varphi_{U}^{-1}\right)$$

于是$\left.\text{}\partial_{i}f\left(A\right)=\frac{\partial\left(f\circ\varphi_{U}^{-1}\right)}{\partial y^{i}}\right|_{\varphi_{U}\left(A\right)}=\frac{\partial f_{A}}{\partial x^{j}}\left|_{\varphi_{A}\left(A\right)}\frac{\partial x^{j}}{\partial y^{i}}\right|_{\varphi_{U}\left(A\right)}.$

对照之前自然标架的定义：
$$\sigma_{i}=\frac{\partial x^{j}}{\partial y^{i}}e_{j}$$

那么
$$\partial_{\sigma_{i}}f\left(A\right)=\frac{\partial f_{A}}{\partial x^{j}}\left|_{\varphi_{A}\left(A\right)}\frac{\partial x^{j}}{\partial y^{i}}\right|_{\varphi_{U}\left(A\right)}.$$
也就是说，在同构意义下$\{\partial_{i}\}$就是我们讨论的自然标架
## 仿射空间几何学补充问题
### 仿射空间中的正则曲线
三维仿射空间也可以模仿[微分几何中的三维欧式空间中的向量与曲线论](/blog/2025/02/04/differential-geometry-notes/)中正则曲线的概念，由于仿射空间没有向量的长度，那么曲线的弧长，曲率，挠率都无法建立，不过我们依旧可以讨论光滑，正则，同向的问题

定义：设 $\mathcal{A}^{3}$为一个仿射空间，$\gamma_:(-\varepsilon,\varepsilon)\mapsto\mathcal{A}^{3}$为一个映射，设一个仿射坐标系$\mathcal{A}=\left\{O,e_{i}\right\}$,其坐标映射记为$\varphi_{\mathcal{A}}.$如果
$$\left(-\varepsilon,\varepsilon\right)\xrightarrow{\gamma}\rightarrow\mathcal{A}^{3}\xrightarrow{\varphi_{i}} R$$
是连续/可微/光滑映射，那么称 $\gamma$ 为 $x^3$上的连续/可微/光滑曲线.进一步如果$\varphi_{A}\circ \gamma$是$\mathbb{R}^3$上的正则曲线，则称 $\gamma$为 $x^3$上的正则曲线.

定理：设 $\gamma:\left(-\varepsilon ,\varepsilon \right)\mapsto \mathcal{A}^{3}$是一条连续/可微/光滑/正则曲线， $A\in \gamma\left(\left(-\varepsilon,\varepsilon\right)\right),A=\gamma\left(t_0\right).t_0\in\left(-\varepsilon,\varepsilon\right),A$的邻域$U$上有一个广义坐标系
$\{U,\varphi_{U}\}$,坐标变量记为$y^{i}$,那么
$$\left(t_{0}-\varepsilon^{\prime},t_{0}+\varepsilon^{\prime}\right)\xrightarrow{\gamma}U\xrightarrow{\varphi_{U}}\varphi_{U}\left(U\right)\subset\mathbb{R}^{3}$$
是一个连续/可微/光滑/正则曲线段


### 余向量场与函数微分
#### 余向量
在本文“导算子”部分中，我们是用来一个点附近的光滑函数构成的集合$\mathcal{F}_{\mathrm{A}}$ 本节我们关注这个集合

明显的，我们发现$\mathcal{F}_{\mathcal{A}}$有下面的线性结构
$$\left(\alpha f+\beta g\right)\left(x\right):=\alpha f\left(x\right)+\beta g\left(x\right).$$
但是这个空间维数太高难以研究，我们需要把在所有$v\in T_A$（看作导算子）作用下都得到一样值的函数看成同一个对象，为此引入下面的定义

定义：设$A$为$\mathcal{A}^{n}$中一个点，记$\mathcal{F}_{\mathrm{A}}$为定义在 $A$ 附近的光滑函数全体.若对于任意的定义在$O$的导算子$D$,都有
$$Df=Dg$$
我们称$f\sim g.$ 这是一个等价关系

定义：记$\mathscr{F}_A=\mathscr{F}_A/\sim$,称为 $\mathscr{A}$ 点的余向量空间（[点集拓扑学中的商空间](/blog/2024/10/16/point-set-topology-notes/)），函数 $f$ 所在的等价类记为$\overline f$,则有线性运算：
$$\alpha\overline{f}+\beta\overline{g}:=\overline{\alpha f+\beta g}.$$
我们给出该定义的一个推论（莱布尼茨公式）$\overline{fg}=g\left(A\right)\overline{f}+f\left(A\right)\overline{g}$

为了研究出商空间的结构，我们首先需要研究清楚0等价类都有谁，为此给出下面的定理

定理： 设  $A\in \mathscr{A}, f$为定义在 $A$附近的光滑函数 .设 $\{ U, \varphi _U\}$为任意一个满足  $A\in U$ 的广义坐标系 , 自变量记为 $\{ y^i\} _{i= 1, \ldots , n}$。 记  $f_U= f\circ\varphi _U^{- 1}$ 的读取,则
$$\overline{f}=0\Leftrightarrow\frac{\partial f_{U}}{\partial y^{i}}|_{\varphi_{U}\left(A\right)}=0.$$

#### 微分
现在讨论如何计算函数的微分

定理：假定$\{ U, \varphi _U\}$为任意一个满足  $A\in U$ 的广义坐标系 ,  $\{ dy^i\} _{i= 1, \ldots , n}$是其自然余标架场，那么对于$U$上任意光滑函数$f$
$$df=\frac{\partial\left(f\circ\varphi_{U}^{-1}\right)}{\partial y^{i}}dy^{i}.$$
### 张量场
场的概念是为空间区域的每个点指定一个对象，因此我们可以在每个点上指定一个双线性函数，引入下面的定义

定义：设$U$ 为 $\mathscr{A}^n$上的一个开区域，对于任意的点 $A\in U$,我们指定一个定义在$A$点的向量空间上的双线性函数$b(A).$这样的$b$称为一个双线性函数场

定义：空间区域上定义的一个非退化对称的双线性型场，称为一个伪Riemann(黎曼)度量.如果这个双线性型在每个点还是正定的，则这个度量称为 Riemann 度量.进一步，如果在一个广义坐标系的自然余标架场下其分量都连续/可微/光滑，称该伪 Riemann 度量为连续/可微/光滑：


## 仿射空间的曲面
### 曲面的直观刻画
#### 曲面的参数式刻画
在数学上描述曲面比曲线更加复杂，对于曲线问题，我们前文使用下面的参数方程进行刻画
$$r:\left(-\varepsilon,\varepsilon\right)\mapsto\mathscr{A}^3,r^{\prime}\left(\tau\right)\neq0.$$
他可以容易的理解为质点的运动方程

仿照这个思路，我们仍旧考虑参数方程来刻画曲面，为了体现曲面是二维对象的特定，我们用
$$r:\left(-\varepsilon,\varepsilon\right)\times\left(-\varepsilon,\varepsilon\right)\mapsto\mathscr{A}^3,\left(s,t\right)\mapsto r\left(s,t\right).$$
这实际上可以理解为单参数曲线族，通过变动$s$的方法使得曲线被编织成曲面，为了保证曲线段正则以及曲线能够顺利编织成面而不是仍旧是线，需要满足下面的条件
$$\begin{matrix}
\partial_{t}r\neq0 \\
 \partial_{s}r\neq0.\\
\partial_{t}r同\partial_{s}r不共线
\end{matrix}$$

综上，我们可以给出更为公理化的定义有

定义(曲面的参数式刻画)：设
$$r:\left(-\varepsilon,\varepsilon\right)\times\left(-\varepsilon,\varepsilon\right)\mapsto\mathscr{A}^3,$$
$$\left(s,t\right)\mapsto r\left(s,t\right)$$

为可微/$C^k/C^\infty$映射，满足$\partial_sr(s,t)$与$\partial_tr(s,t)$不共线，那么称映射$r$为可微
$C^{k}$ / 光滑曲面片$S=r\left(\left(-\varepsilon,\varepsilon\right)\times\left(-\varepsilon,\varepsilon\right)\right)$的参数式.
#### 曲面作为函数图像
我们再讨论一个偏向分析的方法，毕竟曲线是一元函数的图像，曲面会不会也是呢？

定义(曲面作为函数图像)：设 $\mathscr{A}^3$上建立了仿射坐标系 $A=\{O,e_1 ,e_2,e_3\}$,其坐标变量记为$x^1,x^2,x^3.$设$f$为定义在$\left(-\varepsilon,\varepsilon\right)\times\left(-\varepsilon,\varepsilon\right)\to R$上的实值可微 /$C^k$/光滑函数.若S\subset \mathscr{A}^3满足
$$S=\left\{\varphi_{A}^{-1}\left(x^{1},x^{2},f\left(x^{1},x^{2}\right)\right)|-\varepsilon<x^{1},x^{2}<\varepsilon\right\}$$
称 $S$ 为可微/$C^k$ /光滑曲面片

#### 曲面作为等值面
我们在考虑等值面，也就是标量函数零点的集合来表示曲面，这是[解析几何](/blog/2024/12/12/analytic-geometry-notes/)中常用的思路

定义(曲面作为等值面)：设$S\subset \mathscr{A}^3$  $f$为$\mathscr{A}^3$中区域$D$上定义的可微/$C^k/$光滑标量场(实值函数),并且满足
$$\forall A\in S,f\left(A\right)=0.$$
如果在 $S$上还有 $df\neq0$,那么我们称 $S$ 为可微/$C^{k}$/光滑曲面片，

#### 使用广义坐标系
定义：曲面片可以看成是$\mathscr{A}^3$中的一个非空子集(记为 $S$),它具有以下性质：$\forall A\in S,\exists U\ni A,U$ 为 $\mathscr{A}^3$开区域，使得存在一个广义坐标系$\{U,\varphi_{U}\}$
$$\varphi:U\mapsto V\subset\mathbb{R}^{3},\varphi_{U}\left(S\cap U\right)\subset\mathbb{R}^{2}\times\left\{0\right\}\cap V.$$
也就是说在局部，曲面可以用一个充分光滑的广义坐标“展平”.

**这些都是很直观很合理的刻画，下面我们希望证明这几种表达方式其实等等价的**
### 仿射空间中的曲面
#### 隐函数定理
我们考虑$\mathbb{R}^n$中开集之间的可微/光滑映射：
$$F:\mathbb{R}^m\supset U\mapsto V\subset\mathbb{R}^n.$$
若 $a\in U$,我们称线性映射 $\mathcal{D}_aF:\mathbb{R}^m\mapsto\mathbb{R}^n$为$F$ 在$a$ 点的导映射，若
$$\frac{\left\|F\left(a+h\right)-F\left(a\right)-D_{a}F\left(h\right)\right\|_{R^{n}}}{\left\|h\right\|_{R^{m}}}\mapsto0,\left\|h\right\|_{R^{m}}\mapsto0.$$
我们重点关注 $\mathcal{D}_aF$ 是满秩(秩为 $min\{m,n\}$的情况),并引人以下定义。

定义：设$F$为$U\mapsto V$的可微映射.若$\forall a\in U,D_aF$满秩
* $若m\leqslant n,称F在U内是浸入\left(immersion\right)$
* $若m\geqslant n,称F在U内是浸没\left(submersion\right)$

定理：设 $F$ 是$\mathbb{R}^m$中的开集 $U$ 到$\mathbb{R}^n$中的开集 $V$ 的 $C^1$ 映射

若$F$在$U$内是浸入，则对于任意$a\in U$,存在$a$的邻域$W$以及$F(a)$的邻域$W^{\prime}$以及微分同胚(可逆并且每个点的导映射可逆)
$$\varphi:W^{\prime}\mapsto\varphi\left(W^{\prime}\right)\subset\mathbb{R}^{n}$$
使得
$$\varphi\circ F:W\mapsto\varphi\left(W^{\prime}\right),\\\left(x^{1},x^{2},\cdots,x^{m}\right)\mapsto\left(x^{1},x^{2},\cdots,x^{m},0,\cdots,0\right).$$
若$F$在$U$内是浸没，则存在$\psi ^{-1}(W)\subset R^m$到$W$的微分同胚$\psi$ 使得
$$F\circ\psi:R^{m}\supset\psi^{-1}\left(W\right)\mapsto F\left(W\right),\left(x^{1},x^{2},\cdots,x^{m}\right)\mapsto\left(x^{1},x^{2},\cdots,x^{n}\right).$$
#### 曲面的局部参数化
下面我们可以开始引入曲面的定义，也是曲面的局部参数化定义

设 $S\subset \mathscr{A}^3$ 为一个非空集合，带有从 $\mathscr{A}^3$上继承的子拓扑.设$\{O,e_1,e_2,e_3\}$ 为一个 仿射坐标系，其坐标映射记为 $\varphi_{\mathcal{A}}$,坐标自变量记为$\left\{x^1,x^2,x^3\right\}.$如果$\forall A\in S$,都存在一个$A$ 在 $\mathscr{A}^3$中的开邻域$U$ 以及一个映射 $\varphi$,使得
$$\varphi:\left(-\varepsilon,\varepsilon\right)\times\left(-\varepsilon,\varepsilon\right)\mapsto U\cap S$$
满足：

 $\varphi$是$\left(-\varepsilon,\varepsilon\right)\times\left(-\varepsilon,\varepsilon\right)$到$U\cap S$的双射；

映射：$$\varphi_{A}\circ\varphi:\left(-\varepsilon,\varepsilon\right)\times\left(-\varepsilon,\varepsilon\right)\mapsto\varphi_{A}\left(U\right),\\\left(u,v\right)\mapsto\left(x^{1}\left(u,v\right),x^{2}\left(u,v\right),x^{3}\left(u,v\right)\right)$$为光滑映射；

记$x^{i}\left(u,v\right)=\left[\varphi_{A}\circ\varphi\left(u,v\right)\right]^{i}.$ 则向量
$$\left(\partial_{u}x^{1},\partial_{u}x^{2},\partial_{u}x^{3}\right),\left(\partial_{v}x^{1},\partial_{v}x^{2},\partial_{v}x^{3}\right)$$
在$\left(-\varepsilon,\varepsilon\right)\times\left(-\varepsilon,\varepsilon\right)逐点线性无关.$

那么$\varphi$称为$S$在$A$点附近的一个局部光滑参数化.如果$S$在任何一点都至少有一个局部光滑参数化，则称$S$是光滑曲面

**有了本节的知识，我们可以证明上一节提出的四个曲面定义殊途同归，是等价的，本节介绍隐函数定义与局部参数化定义都是为了解决这个问题**
### 曲面的切空间与切向量场
#### 切向量与切空间
给定曲面之后，我们着手研究曲面的切平面，他可以看作是曲面在给定点的切向量的集合，因此我们先研究切向量

定义：设$S\subset \mathscr{A}^3$ 为一个光滑曲面，$A\in S$  设
$$\varphi:\left(-\varepsilon,\varepsilon\right)\times\left(-\varepsilon,\varepsilon\right)\rightarrow S,\left(u,v\right)\mapsto\varphi\left(u,v\right)$$
为点$A$附近的局部参数化，设$f\in\mathcal{F}_{\mathrm{A}}$ 是点$A$附近定义的光滑函数，定义
$$\left.\partial_{u}\left(A\right)f:=\frac{\partial\left(f\circ\varphi\right)}{\partial u}\right|_{\varphi^{-1}\left(A\right)}$$
称$\partial_{u}\left(A\right)$ 为曲面$S$在$A$点的一个切向量

我们可以证明$\partial_{u}\left(A\right)$是一个导算子，是向量空间$T_A$中的元素。因此$T_A$中所有可以被看成是如上定义的 **某个局部参数化下对第一个参数求导的算子构成的集合**  称为曲面$S$在$A$点的切空间，记作$T_{A}\left(S\right)$

只需要任何一个局部参数化得到的切向量$\partial_{u},\partial_{v}$，就可以线性组合出所有的切向量，也就是
$$T_{A}\left(S\right)=Span\left\{\partial_{u},\partial_{v}\right\}$$

#### 切向量场
切向量场也是一个向量场，就是在曲面上的每个点都指定一个切向量，形成的向量场；一个切向量场在某点满足某正则性，只需要他在某个局部参数标架（仿射标架）中的分量满足该正则性
### 曲面的余切空间与微分形式
切空间死向量空间的子空间，本节我们希望对向量空间的对偶空间——余向量空间做同样的事情。

由于余空间本身没有向量空间直观，因此本节的数学理论省去，仅了解即可
### 曲面之间的映射与切映射
在本节中我们讨论三维仿射空间中的曲面之间的映射的局部性质，
#### 曲面间的映射
定义：设$S,S^\prime\in\mathscr{A}^3$为两片光滑曲面$,A\in S,A^\prime\in S^{\prime}.$ 设 $F:S\mapsto S^\prime$为一个映射，满足$F(A)=A^\prime$.如果在 $A,A^\prime$附近分别有$S,S^{\prime}$的局部参数化$\{U,\phi_{U} \}\left\{U^{\prime},\varphi_{U^{\prime}}\right\}$  满足$\varphi_{U}\left(0,0\right)=A,\varphi_{U^{\prime}}\left(0,0\right)=A^{\prime}.$ 如果
$$\varphi_{U}^{-1}\circ F\circ\varphi_{U}:\left(-\varepsilon,\varepsilon\right)\times\left(-\varepsilon,\varepsilon\right)\longmapsto U\cap S\xrightarrow{F}U^{\prime}\cap S^{\prime}\xrightarrow{\varphi_{U}^{-1}}\left(-\varepsilon,\varepsilon\right)\times\left(-\varepsilon,\varepsilon\right)$$
是$\mathbb{R}^2$上开集到$\mathbb{R}^2$上开集的连续/可微/光滑映射，则称 $F$为$A$ 点附近的连续/可微/光滑映射。

**这个定义依赖于具体的局部参数化选取，实际上对任意一对局部参数化都成立**

这个思路可以自然的在仿射坐标系下研究

定义：设$A=\left\{O,e_i\right\}$为 $\mathscr{A}^3$中的仿射坐标系，$S^\prime\subset \mathscr{A}^3$为一个光滑曲面设 $S$为 $\mathscr{A}^3$中曲面，$A\in S,F:S\mapsto S^\prime$为两曲面之间的映射$,f(A)=A^\prime\in S^{\prime}.$ $\left\{U,\varphi_U\right\}$为$A$点附近$S$的局部参数化，参数映射记为
$$\varphi_{U}:\left(-\varepsilon,\varepsilon\right)\times\left(-\varepsilon,\varepsilon\right)\mapsto U\cap S,并且\varphi_{U}\left(0,0\right)=A$$
那么$f$在$A$连续/光滑/可微 当且仅当 $\exists0<\varepsilon^{\prime}\leq\varepsilon$ 使得
$$\varphi_{A}\circ F\circ\varphi_{U}:\left(-\varepsilon^{\prime},\varepsilon^{\prime}\right)\times\left(-\varepsilon^{\prime},\varepsilon^{\prime}\right)\mapsto R^{3}$$
在$(0,0)$连续/光滑/可微
#### 曲面间的切映射
现在我们研究$F$的导映射，这将是一个$T_A(S)$到$T_{A^{\prime}}(S^{\prime})$的线性映射.

先对于 $g\in\mathcal{F}_{\mathcal{A}^{\prime}}(S^{\prime})$,也就是 $g$ 是 $S^{\prime}$上 $A^\prime$附近有定义的光滑函数，我们总可以用下面的办法把它“对应为”$S$上$A$附近有定义的光滑函数：
$$F^{*}\left(g\right):=g\circ F.$$
这个操作称为“将$g$通过$F$拉回到$S$”.

然后，设 $D\in T_A(S)$为一个导算子，那么$D$可以通过上述拉回操作作用在$g$ 上
$$F_{*}\left(D\right)\left(g\right):=D\left(F^{*}\left(g\right)\right)=D\left(g\circ F\right).$$
这个操作称为“将$D$推出到$T_{A^{\prime}}S^{\prime}.$  我们可以证明如此被推出的$F_*(D)$是$F_{\mathcal{A}^{\prime}}(S^{\prime})$ 的一个导算子

也就是说，“通过$F$推出” 这个操作实际上产生了一个$T_A(S)$到$T_{A^{\prime}}(S^{\prime})$的线性映射. 我们将其称为$F$在$A$点的切映射。记为$TF_A$或$dF_A$  **映射的线性性此处不额外证明**

**特别的，我们在此处研究的曲面到曲面的映射，如果曲面上定义了向量场（$X$）如切向量场，那么映射$F$自然也产生一个向量场，这个向量场的正则性和原始的向量场一样**
