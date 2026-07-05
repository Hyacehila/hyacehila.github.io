---
title: "微分几何：几何学分类、曲线论与曲面论"
title_en: "Differential Geometry: Curves, Surfaces, and Metrics"
date: 2025-02-04 13:35:34 +0800
categories: ["Mathematics", "Geometry & Topology"]
tags: ["Learning Notes", "Mathematics", "Differential Geometry"]
author: Hyacehila
excerpt: "整理几何学分类、曲线论、曲面论、曲面度量和相关微分几何基础。"
excerpt_en: "Covers geometry classifications, curve theory, surface theory, surface metrics, and related foundations."
mathjax: true
hidden: true
permalink: '/blog/2025/02/04/differential-geometry-notes/'
---
## 绪论
### 什么是几何学
我们可以认为，几何学是研究三维欧几里得空间的点集性质的学问，在18世纪以前都如此，目前这被我们称为古典几何学，大致有下面分支
* 初等几何：我们在通识教育中学习的初等几何学，使用公理体系研究
* 初等解析几何：我们在[解析几何](/blog/2024/12/12/analytic-geometry-notes/)中学习的使用坐标等代数学方法研究
* 古典微分几何：使用坐标方法，综合微积分与线性代数，研究更一般的性质

随着数学的发展，几何学突破古典几何学的范畴，产生了射影几何，仿射几何，球面与双曲几何，拓扑学等分支。人们希望找到统一的理论研究几何学。Klein提出了使用变换的方法研究几何学，认为几何学研究一个点集。在某种变换下保持不变的性质。

进一步的，我们可以认为几何学研究集合在某个变换群作用下，仍旧保持不变的性质的那些知识。

### 几何学分类
我们假设已经有了一个具备标准直角坐标系的平面，研究点集在某个特定变换群下的保持不变的性质，就是几何学

#### 等距变换群
对于下面的变换
$$\begin{aligned}(\xi,A):&\mathbb{R}^2\mapsto\mathbb{R}^2,\\&x\mapsto Ax+\xi.\end{aligned}$$
其中$A\in O(2),\xi\in R^2$；我们能看出，这是一个等距线性变换与平移，他保持图形的尺寸（点与点之间的距离）和形状（任意向量之间的夹角不变）这就是平面几何的研究范围，也就是这个变换群代表 **平面欧式几何** 包括整个古典几何学

#### 仿射变换群
对于下面的变换
$$\left(\xi,A\right):\mathbb{R}^{2}\to\mathbb{R}^{2},x\mapsto Ax+\xi.$$
其中$A\in GL(2),\xi\in R^2$；我们能看出，这是一个非退化线性变换与平移，此时一条直线上两个线段的长度的比保持不变，这个变换群代表 **仿射几何学** 特别的，凡是仿射几何研究的性质都被欧式几何概括
#### 射影变换群
考虑矩阵
$$M=\begin{pmatrix}0&1&0\\0&0&1\\1&0&0\end{pmatrix}.$$
我们进行划分
$$\boldsymbol{M}=\left(\begin{array}{ll|l}
\boldsymbol{A} & \boldsymbol{B} \\
\hline c \quad d & e
\end{array}\right)$$
也就是一个 $2\times 2$矩阵 一个  $2\times 1$矩阵 三个数

我们考虑变换
$$\left.\left[\begin{array}{c}x^{1}\\x^{2}\end{array}\right.\right]\rightarrow\left.\left[A\left(\begin{array}{c}x^{1}\\x^{2}\end{array}\right.\right)+B\right]/\left(cx^{1}+dx^{2}+e\right)=\left.\left[\begin{array}{c}x^{2}/x^{1}\\1/x^{1}\end{array}\right.\right]$$
这样的变换保持若干点共线等性质，代表着 **射影几何学**

#### 连续变换群
考虑所有连续变换构成的群，保持连通性，紧性等性质不变，代表着 **拓扑学**

### 后续内容
在本节之后，我们开始讨论更为一般的几何对象，还是继承我们在之前的几何研究中， **寻找标准形式，寻求分类** 的基本思路。主要讨论三维欧式空间的几何学以及一些简单的仿射几何学。主要包括
* 微分几何基础与曲线论
* 仿射几何基础与仿射曲线论
* 仿射曲面
* 微分几何曲面论


## 三维欧式空间中的向量与曲线论
### 三维欧式空间中的曲线
那些三维欧式空间的基本性质我们已经在[解析几何](/blog/2024/12/12/analytic-geometry-notes/) 数学分析4 多元的微分与积分理论中进行了详尽的介绍，这里仅仅简单的复习一些比较重要且以前并不是非常熟练的内容，作为我们开始学习微分几何的引子

定义：设$r$是$(a,b)$到$R^3$的连续映射 即
$$r:(a,b)\mapsto\mathbb{R}^{3}$$
并且该映射满足 $r^{\prime}\left(t\right)\neq0$  令 $C=\left\{r\left(t\right)\in\mathbb{R}^{3}|t\in\left(a,b\right)\right\}$ 则称$C$是正则曲线， $r$是曲线$C$的一个正则参数化

明显的，**一条曲线可以有多个正则参数化**

对于曲线的弧长，我们自然的可以给出下面的定义：设$u(t)$是一个正则参数化，那么从点$a\to b$的弧长为
$$S=\int_{t_{a}}^{t_{b}}\left\|u^{\prime}\left(t\right)\right\|_{2}dt=\int_{t_{a}}^{t_{b}}\sqrt{\left(u^{\prime}\left(t\right),u^{\prime}\left(t\right)\right)_{E}}dt$$

这存在一个问题，弧长是曲线的性质，但是和一个不确定的量：正则参数化联系起来了，如果换正则参数化，弧长还是确定的吗？

我们不加额外说明的给出下面的补充定义与命题
* 定义：使用映射$\mu$ 作用于一个正则参数化，保持曲线不变，得到一个新的正则参数化我们则称映射$\mu$是一个正则参数变换，新的正则参数化称为正则再参数化
* 命题：**正则参数化不改变弧长**
* 命题：曲线的等距变换不影响弧长

### 曲线的形状
#### 曲率
从本节开始，我们专注学找一个几何量，让我们刻画曲线的形状，这些几何量应该在等距变换群与正则变换群保持不变。由于正则变换群不是我们研究得重点，因此我们主要考虑等距变换群。

在拥有弧长之后，研究点在弧上的运动就显得很重要。很明显，之前对弧的刻画不包括点运动速度的问题，因此我们不妨认为速度的大小一直为1,据此我们可以考虑给出下面的定义：

定义：假设 $r:(a,b)\mapsto\mathbb{R}^3$为一条弧长参数 $C^3$ 曲线，也就是说$\|\dot{r}(s)\|_2\equiv1$,定义下面的两个向量
$$\alpha\left(s\right):=\dot{r}\left(s\right),\beta\left(s\right):=\ddot{r}\left(s\right)/\left\|\ddot{r}\left(s\right)\right\|_{2}.$$

因为$\parallel\boldsymbol{\alpha}(s)\parallel_2\equiv1,\boldsymbol{\alpha}(s)\perp\boldsymbol{\beta}(s).$记$\kappa(s):=\parallel\ddot{r}(s)\parallel_2$,称为曲线在$r(s)$点的**曲率**

显然，上述 $\alpha(s)$是曲线的切线方向.将 $\beta(s)$称为曲线的**主法向**.注意，如
果$\ddot{r}\left(s\right)=0$，则在这个点主法向无定义.进一步：
$$\gamma\left(s\right):=\alpha\left(s\right)\times\beta\left(s\right)$$
称为曲线在该点的**次法向**

在上面的定义中，$\alpha$是速度，因此有其大小一直为1,$\beta$是单位化加速度，他的大小$\kappa(s)$反映了曲线转换的缓急。这与物理学中研究加速度是殊途同归的，也是区分了切向与法向的加速度，并将法向加速度被原始速度的影响进行了扣除，得到了运动曲线本身形状的描述。

定理：**曲率度量曲线与直线的差距，直线的曲率为0**

#### 密切平面
定义：将过 $r(s)$点、以 $\alpha(s)$为法向的平面称为曲线在该点的**法平面**；称过$r(s)$点以 ${\beta}(s)$为法向的平面为曲线在该点的从**切平面**；称过$r(s)$点以$\gamma(s)$为法向的平面为曲线的**密切平面**

**注意，此处的定义都要求以设定的向量为法向量，因此切线对应法平面，主法对应从切平面，次法对应密切平面**

定理（密切平面的几何意义）：设
$$r:\left(a,b\right)\mapsto R^{3},s\mapsto r\left(s\right)$$
是$C^3$ 正则曲线$C$的弧长参数化 设曲线在点$s_0$的曲率$\kappa(s_0)\ne0$ 那么有
$$\gamma\left(s_{0}\right)//\lim_{\Delta s\to0^{+}}\frac{\left(r\left(s_{0}\right)-r\left(s_{0}-\Delta s\right)\right)\times\left(r\left(s_{0}+\Delta s\right)-r\left(s_{0}\right)\right)}{\left(\Delta s\right)^{3}}.$$
也就是说， **在曲率不为0的点前后找两个点，三个点构成三角形并张成一个平面，当前后两个点不断接近，他们张成的平面的极限位置就是其密切平面**
#### 挠率
现在我们可以给出一些自然的观察，如果运动曲线是平面曲线，那么速度方向和主法向均在次运动面上，其密切平面是固定的，就是运动的平面；如果曲线不在一个平面上，那么**密切平面的法向量$\gamma$会变换，因此这个向量的变换率就反应了曲线偏离平面曲线的程度。** 据此我们可以给出下面的叙述

定义：设
$$r:\left(a,b\right)\mapsto R^{3},s\mapsto r\left(s\right)$$
是$C^3$ 正则曲线$C$的弧长参数化 则定义$\gamma\left(s\right)=r\left(s\right)\times\frac{\ddot{r}\left(s\right)}{\left\|\ddot{r}\left(s\right)\right\|_{2}}$  记
$$\dot{\gamma}\left(s\right)=-\tau\left(s\right)\beta\left(s\right).$$
这个$\tau(s)$称为曲线在$r(s)$点的**挠率**

定理：**挠率度量曲线与平面曲线的差距，平面曲线的挠率为0**

### Frenet标架和其中的曲线
#### Frenet标架与Frenet公式
定义：设 $r:\left(a,b\right)\mapsto R^{3}$ 为一条正则$C^3$正则曲线$C$的弧长参数化，$$\alpha\left(s\right):=\dot{r}\left(s\right),\beta\left(s\right):=\ddot{r}\left(s\right)/\left\|\ddot{r}\left(s\right)\right\|_{2},\gamma\left(s\right):=\alpha\left(s\right)\times\beta\left(s\right)$$称以$r\left(s\right)$为原点以$\alpha\left(s\right)$、 $\beta(s)、\gamma(s)$为基的坐标系为曲线在 $r(s)$的 Frenet 标架(frame)；自然，当$\dot{\alpha}(s)=0$ 时 ,$\boldsymbol\beta(s)$无定义.所以 Frenet 标架仅仅在曲线曲率不为零的点有定义.

Frenet 标架的意义是由曲线自身在一个点决定的三个右手正交方向，而不是我们人为选取的方向.所以 Frenet 标架的性质在一定程度上体现了曲线本身的性质(实际上经过本节的学习，大家会看到 Frenet 标架刻画了曲线的所有性质：两个具有相同 Frenet 标架的曲线事实上是重合的)

本节我们将研究 Frenet 标架沿着曲线弧长参数增加的方向怎样变化.更确切地说，我们将研究$\dot{\alpha}\left(s\right),\dot{\beta}\left(s\right),\dot{\gamma}\left(s\right).$

已知的结果是：$\dot{a}\left(s\right)=\kappa\left(s\right)\beta\left(s\right),\dot{\gamma}\left(s\right)=-\tau\left(s\right)\beta\left(s\right)$,现在仅仅需要计算$\beta(s)$ .所谓“计算”指的是将$\dot{\beta}(s)$表示成$\alpha、\beta、\gamma$ 的线性组合.已经知道β是单位向量，所以$\dot{\beta}\left(s\right)\perp\beta\left(s\right).$于是
$$\dot{\beta}\left(s\right)=b_{1}\left(s\right)\alpha\left(s\right)+b_{3}\left(s\right)\gamma\left(s\right)$$
计算系数有
$$\begin{aligned}b_{1}\left(s\right)&=\left(\alpha\left(s\right),\dot{\beta}\left(s\right)\right)_{E}=\frac{d}{ds}\left(\alpha,\beta\right)_{E}\left(s\right)-\left(\dot{\alpha}\left(s\right),\beta\left(s\right)\right)_{E}\\&=-\left(\dot{\alpha}\left(s\right),\beta\left(s\right)\right)_{E}=-\kappa\left(s\right);\\b_{3}\left(s\right)&=\left(\gamma\left(s\right),\dot{\beta}\left(s\right)\right)_{E}=-\left(\dot{\gamma}\left(s\right),\beta\left(s\right)\right)_{E}=\tau\left(s\right).\end{aligned}$$
因此我们可以得到$\dot{\alpha}\left(s\right),\dot{\beta}\left(s\right),\dot{\gamma}\left(s\right).$满足
$$\frac{\mathrm{d}}{\mathrm{d}s}\begin{bmatrix}\alpha\left(s\right)\\\beta\left(s\right)\\\gamma\left(s\right)\end{bmatrix}=\begin{bmatrix}0&\kappa\left(s\right)&0\\-\kappa\left(s\right)&0&\tau\left(s\right)\\0&-\tau\left(s\right)&0\end{bmatrix}\begin{bmatrix}\alpha\left(s\right)\\\beta\left(s\right)\\\gamma\left(s\right)\end{bmatrix}$$
这称为Frenet公式或者标架运动公式，他说明了，曲线上Frenet标架的运动方式完全被逐点的曲率和挠度完全确定。
#### 曲线的唯一性
定理(曲线的唯一性定理)：设$r:\left(a,b\right)\to\mathbb{R}^{3}$和$\tilde{r}:\left(a,b\right)\to\mathbb{R}^{3}$分别是
$C^{3}$正则曲线$C$和$\tilde{C}$的弧长参数化.若$\forall s\in\left(a,b\right),\kappa\left(s\right)\neq0$,并且
$$\kappa\left(s\right)=\kappa\left(s\right)\neq0,\tau\left(s\right)=\tau\left(s\right).$$
那么存在一个等距变换$(A,\xi)$使得
$$Ar\left(s\right)+\xi=\widetilde{r}\left(s\right),\forall s\in\left(a,b\right).$$

**曲线的唯一性定理说明相同曲率和挠度下，曲线在等距变换群作用下唯一，也就是欧式几何下的唯一性**
#### 曲线的存在性
现在我们研究唯一性的反问题，给出曲率挠度，寻找曲线

定理(曲线的局部存在性)：设$(a,b)\subset \mathbb{R},\kappa$、$\tau$为($a,b)$上定义的 $C^1$实函数，并且 $\kappa>0.$则存在着$\mathbb{R}^3$中的弧长参数化曲线 $r:(a,b)\mapsto\mathbb{R}^3$使得 $\kappa$、$\tau$为该曲线的曲率和挠率。

**局部存在性只保证一个点的存在性**


## 仿射空间几何学
由于仿射空间几何学的相对独立性，我们在[仿射空间几何学](/blog/2025/02/12/affine-geometry-notes/)中进行讨论，他作为微分几何理论的补充内容呈现，对于一些重要问题的讨论有着铺垫
## 曲面局部内蕴几何
从本章开始我们系统的讨论一个问题，给定一个曲面片，能否在不进行"拉伸"和"挤压"的条件下，把他变成一个平面片。

在本章中，我们研究在内蕴几何的意义下，哪些曲面局部可以看作平面；事实上，我们会发现圆柱面，圆锥面与球面，双曲面存在巨大不同，并且引入Riemann曲率来区分他们。

**从本章开始，我们恢复对于欧式空间的讨论，而不是前一章研究的仿射空间几何学**
### 被束缚在曲面的质点
#### 欧氏空间
我们研究一个欧氏空间$E^3$ 他是一个仿射空间$\mathscr{A}^3$加上度量，也就是内积
$$(\overrightarrow{AB},\overrightarrow{CD})$$
也可以理解为在空间上定义了一个双线性函数，
$$(\bullet,\bullet):\mathbb{R}^3\times\mathbb{R}^3\mapsto\mathbb{R}$$
他满足度量所具备的性质，如非负性与交换性；这样的带上内积的仿射空间称为 **欧几里得空间**

现在，我们可以建立一个特殊的仿射坐标系—— **标准正交坐标系** 也就是基向量的内积为0或者1，取决于是否同编号，即$(e_{i},e_{j})=\delta_{ij}$

#### 欧氏空间的曲面
我们已经在仿射空间中定义了曲面 [仿射空间几何学中的仿射空间的曲面](/blog/2025/02/12/affine-geometry-notes/)，现在可以自然的将定义拓展到欧氏空间中，毕竟他就是一个增加了度量的仿射空间，并建立一个特殊的坐标系

假设$A\in S$ 在$A$的邻域$U$有$S$的局部参数化
$$\varphi:\left(-\varepsilon,\varepsilon\right)\times\left(-\varepsilon,\varepsilon\right)\mapsto S\cap U.$$
那么我们可以容易的给出质点$P$的运动方程为
$$\overrightarrow{OP}\left(t\right)=x^{i}\left(u\left(t\right),v\left(t\right)\right)e_{i}.$$
**也就是参数与基的线性组合**

因此可以给出质点的速度满足（对参数求导）
$$V\left(t\right)=\dot{u}\left(t\right)\partial_{u}x^{i}e_{i}+\dot{v}\left(t\right)\partial_{v}x^{i}e_{i}=\dot{u}\left(t\right)\partial_{u}+\dot{v}\left(t\right)\partial_{v}.$$

进一步求导给出加速度方程为
$$a(t)=\ddot{u}\left(t\right)\partial_{u}+\ddot{v}\left(t\right)\partial_{v}+\left(\dot{u}\left(t\right)\dot{u}\left(t\right)\partial_{uu}^{2}x^{i}+2\dot{u}\left(t\right)\dot{v}\left(t\right)\partial_{uv}^{2}x^{i}+\dot{v}\left(t\right)\dot{v}\left(t\right)\partial_{vv}^{2}x^i\right)e_i$$
**这里只涉及传统多元分析学的求导，只是符号比较特殊**

为了方便后面我们对加速度的讨论，我们记二阶导部分为$b$ 一阶导部分为$p$ 后者可以进一步分为和曲面相切与和曲面垂直的两部分 也就是
$$a=b+p=b+p_{\perp}+p_{\parallel}$$

进一步约定下面的符号有
$$\begin{matrix}
 y^{1}=u,y^{2}=v,\partial_{1}=\partial_{{y}^{1}}=\partial_{u},\partial_{2}=\partial_{{y}^{2}}=\partial_{v}\\
 g_{ab}=\left(\partial_{a},\partial_{b}\right)\\
\Gamma_{ab}^{l}=\frac{1}{2}g^{lc}\left(\partial_{a}g_{bc}+\partial_{b}g_{ac}-\partial_{c}g_{ab}\right)
\end{matrix}$$
那么有
$$p_{\parallel}=\dot{y}^{a}\dot{y}^{b}\Gamma_{ab}^{c}\partial_{c}$$
以及
$$a_{\parallel}=b+p_{\parallel}=\left(\ddot{y}^{a}+\dot{y}^{b}\dot{y}^{c}\Gamma_{bc}^{a}\right)\partial_{a}$$
#### 曲面上的自由运动与测地线
我们考虑质点在曲面上自由运动，但是只受束缚里的作用，那么加速度一定在曲面的法向，也就是$a_{\parallel}=0$ 这时的特殊的运动轨迹称为曲面的测地线，满足方程
$$\left(\ddot{y}^{a}+\dot{y}^{b}\dot{y}^{c}\Gamma_{bc}^{a}\right)\partial_{a}=0$$
#### 度量
在上面的计算中我们讨论了
$$g_{ab}=\left(\partial_{a},\partial_{b}\right)$$
其中$\partial_{a},\partial_{b}$ 是曲面的切向量，也就是说我们在计算切向量的内积；这种定义在某点的切空间上的度量称为曲面的Riemann度量，我们会在后面进一步研究

### 度量与第一基本形式
现在我们考虑 $S\subset \mathbb{E}^3\text{以及E}^3$上的标准正交坐标系$\{O,e_i\}.$考虑 $A\in\mathcal{S}$ 开集$U$上的局部参数化
$$\left(-\varepsilon,\varepsilon\right)\times\left(-\varepsilon,\varepsilon\right)\mapsto S\cap U\mapsto\mathbb{R}^{3},$$
$$\left(y^{1},y^{2}\right)\mapsto P\mapsto\left(x^{1}\left(y^{1},y^{2}\right),x^{2}\left(y^{1},y^{2}\right),x^{3}\left(y^{1},y^{2}\right)\right).$$

回顾上一节的定义
$$g_{ab}=\left(\partial_{a},\partial_{b}\right)=\sum_{i=1}^{3}\partial_{a}x^{i}\cdot\partial_{b}x^{i}$$
是局部参数标架的向量的内积.同样地对于(曲面上某个点 B)任意切空间上的两个向量，我们都可以计算向量的内积

如果两个切向量$(X,Y)$ 在 $\left(\partial_{u},\partial_{v}\right)$这组基下写出来是
$$X=X^{a}\partial_{a},Y=Y^{a}\partial_{a}.$$
那么则有内积为
$$\left(X,Y\right)=\left(X^{a}\partial_{a},Y^{b}\partial_{b}\right)=X^{a}Y^{b}\left(\partial_{a},\partial_{b}\right)=X^{a}Y^{b}g_{ab}$$
这里$\partial_{1}=\partial_{y^{1}},\partial_{2}=\partial_{y^{2}}.$

根据我们在本文“度量”部分给出的定义，这里定义了一个**Riemann度量**，对于曲面，上述度量的形式称为**曲面的第一基本形式**，第一基本形式的几何量称为**内蕴几何量**，研究内蕴几何量的几何学称为**内蕴几何学**

### 联络
我们在本文“被束缚在曲面的质点”部分的计算中已经发现，求质点的加速度其实就是对质点的速度这个向量求导数。我们已经看到加速度的法向分量一般不是零(这个法向分量是速度的二次型，一般被称为**曲面的第二基本形式**，下一章将予以讨论)而我们考虑的加速度的切向分量仅仅是速度导数的一个分量.但是这么做的好处是，这个分量依然是切向量

根据这个思路的启发，我们给出下面的定义

定义(曲面上的 Riemann 联络导数)：设$S\subset \mathbb{E}^3\text{以及E}^3$是一个充分光滑曲面.设$\{O,e_{i}\}$为一标准正交坐标系(自变量记为 $x^{i}$).设 $A\in S,U 为 A 的开邻域$ .

若在$U\cap S$上定义了一个光滑向量场$X=X^ie_i$(对$U\cap S$ 的每个点 $B$,指定一个向量 $X(B)\in T_B(S)).$ 设$Y\in T_A\left(S\right),\gamma_{:}\left(-\varepsilon,\epsilon\right)\to S$ 为$S$ 上一段正则曲线，满足$\gamma(0)=A,\dot{\gamma}(0)=Y$,则定义向量场$X$在$A$点沿$Y$方向的联络导数$\nabla_YX$为向量

$$\frac{d\left(X^{i}\circ\gamma\right)}{dt}|_{t=0}e_{i}$$
在$T_{A}\left(S\right).$上的正交投影

### 测地线
本节我们给出测地线的定义，也是平面上的直线在曲面上的推广

定义：设
$$\gamma:\left(-\varepsilon,\varepsilon\right)\mapsto S$$
为曲面 $S$上的一段光滑正则曲线.如果该曲线是自平行的，也就是该曲切向量沿着曲线自身是平行的，那么这样的曲线称为测地线.

**这个思路是非常自然的，很容易理解我们为什么把他称为曲面上的直线**

定理：测地线的切向量的模式是常数

即自由质点的速度不变，也就是约束力垂直运动轨迹不做工

典型的例子是圆锥，他的所有母线都是测地线，另一个测地线的例子是地球上的大圆

### 弧长变分
测地线是直线在曲面上的推广，直线段的一个重要性质是他是两个点之间的最短线，测地线也有类似的结果

定理：设 $A,B\in S$ 为光滑曲面上两个点.记 $\Gamma_\mathrm{AB}$为自$A$ 出发到 $B$ 的
正则曲线集合.若$\gamma\in\Gamma_\mathrm{AB}$为其中弧长最短者，则$\gamma$为一条测地线

**证明这个定理就需要使用弧长变分的知识**

### 局部平坦曲面与Riemann曲率
本节作为这一章的收尾，我们回答开篇给出的问题，如何判定一个曲面在某点附近可以展开为平面
#### 局部平坦曲面
定义：一个曲面 $S$在某个点 $A$ 附近称为是平坦的，如果存在开集$A\in U,S\cap U$上存在一个局部参数化(称为局部标准正交参数化)
$$\varphi:\left(-\varepsilon,\varepsilon\right)\times\left(-\varepsilon,\varepsilon\right)\mapsto S\cap U,\\(y^{1},y^{2})\mapsto P\in S\cap U,$$
这里 $\varphi(0,0)=A$,并且$\left\{\partial_{1},\partial_{2}\right\}$构成一个局部单位正交标架，也就是说在$S\cap U$上
$$(\partial_{i},\partial_{j})=\delta_{ij}$$

以上定义告诉我们，如果在曲面局部（$A$附近）可以找到一个直角坐标系，$A$附近的曲面就变成了平面。

现在，我们希望判断一个曲面是否平坦，这就需要我们判断是否能够建立一个这样的坐标系
#### Riemann曲率
我们本节引入Riemann曲率的概念，并且证明曲面在某点附近近似平坦等价于曲率在某点附近恒为0

我们定义
$$\begin{matrix}
 \Phi:\left(-\varepsilon,\varepsilon\right)\times\left(-\varepsilon,\varepsilon\right)\mapsto S\cap U\\
\left(t,u\right)\mapsto\exp_{\gamma\left(t\right)}\left(uX\left(t\right)\right)
\end{matrix}$$
其中$A$是$S$上一点 $\gamma$是通过$A$的一条测地线，$X$是测地线上的向量场，$\exp_{\gamma\left(t\right)}\left(uX\left(t\right)\right)$是指数映射，含义是以$\gamma(t)$为起点$X(t)$为初速度的测地线，考查其在$u$时刻的值（是曲面的一个点）

我们说明，如果Riemann曲率为0,那么$\Phi$就是$A$附近的一个局部标准正交参数化。也就是局部平坦

定义：**Riemann曲率**定义为
$$\left(\nabla_{\partial_{a}}\nabla_{\partial_{b}}-\nabla_{\partial_{b}}\nabla_{\partial_{a}}\right)\partial_{c}=R_{abc}^{d}\partial_{d}$$

特别的，上述Riemann曲率在给定的局部参数化$\Phi$计算，实际上，**在任何一个局部参数化下计算，Riemann曲率都是同一个结果** 因此我们只需要找到任意的一个局部参数化，就可以计算Riemann曲率来判定曲面局部是否平坦，而不需要研究更为复杂的局部标准正交参数化


## 曲面局部外蕴几何
### 外蕴与内蕴
所谓外蕴，指从现在起我们关心的**几何量不仅仅依赖于第一基本形式**，为了描述外蕴和内蕴的区别，依旧考虑上一章提到的柱面的例子.圆柱面沿母线剪开之后可以展开成平面，我们知道这是因为其 Riemann 曲率为零.然而圆柱面毕竟不是平面，它依然是“弯曲的”.这就需要外蕴几何进行刻画.

为了阐述两种弯曲的区别，我们考虑两种定义曲面上两个点的距离的方
式.设$S$为一片曲面，$A,B\in S.$一种方式是
$$D\left(A,B\right)=\inf\left\{S\text{上从}A\text{ 到 }B\text{ 的正则曲线的弧长}\right\}.$$
另一种方式是直接定义为三维欧氏空间中两个点$A、B$的欧氏距离，记为$d\left(A,B\right).$永远有 $d\left(A,B\right)\leqslant D\left(A,B\right).$

注意$,D\left(A,B\right)$ 的计算完全依赖于第一基本形式.所以凡是研究只同 $D(A,B)$有关的几何学都是仅仅由第一基本形式决定.

如果我们要研究曲面关于$d(A,B)$的性质，这就不能仅仅由第一基本形式决定
### 第二基本形式
#### 第二基本形式的物理意义
我们仍旧讨论曲面上的加速度，在第一基本形式中，我们研究了切向分量，并且明确了法向分量一定在加速度表达式的后三项中。而后三项都是速度的二次型，这就意味着加速度的法向分量也是速度的二次型，这个二次型就是我们要引入的第二基本形式。

如果设$\gamma:(-\varepsilon,\varepsilon)\mapsto\mathcal{S}$为质点的运动方程(同时也是一条曲线),$\dot{\gamma}^ie_i$,为其速度，那么加速度：
$$a=\partial_{\dot{\gamma}}\dot{\gamma}^{i}e_{i}.$$
上面第二基本形式的定义可以写成
$$\Pi\left(\dot{\gamma},\dot{\gamma}\right)_{i}=\left(\partial_{i}\dot{\gamma}^{i}e_{i}\right)_{\perp},$$
垂直符号意味着取和曲面正交的分量
#### 第二基本形式的定义
定义：设$\mathbb{E}^3$中有曲面 $S$ 和一个标准正交坐标系${O,e_{i}}$(记坐标为$x^{i}).$ 设$A\in S$为曲面上的一个点，$U$ 为$A$ 的一个邻域. 设 $X,Y$ 为两个定义在$U\cap S$上的切向量场.定义
$$\Pi\left(X,Y\right)=\left(\partial_{X}Y^{i}e_{i}\right)_\perp$$
为曲面的第二基本形式.这里$\partial_x$ 为$X$ 的方向导数，$\bot$的意思是取这个向量在曲面法方向的分量。

我们容易证明下面的性质

定理(第二基本形式的基本性质)：设$\Pi$为曲面$S$的第二基本形式$X,Y$为曲面在$A$点的两个切向量，$f$为定义在 S 上的光滑函数.则：
* $\Pi\left(X,Y\right)=\Pi\left(Y,X\right)$,
* $\Pi\left(fX,Y\right)=\Pi\left(X,fY\right)=f\Pi\left(X,Y\right).$


### Gauss绝妙定理
我们用Gauss绝妙定理来作为本章和整个微分几何入门的收尾；从某种程度上可以说它是微分几何发展史早期最重要的局部定理之一；正是由于这个定理人们开始认识内蕴几何与外蕴集合的区别，人们发现第一基本形式也包含弯曲的信息

我们先介绍Gauss曲率；设$S.$为$E^3$中和的一片光滑曲面 $A\in S$，在$A$点建立适当的标准正交坐标系$\left\{A,e_i\right\}$,总可以使得 $S$ 在$A$ 附近成为一个二元实值函数$f$的图像也就是
$$P\in S\cap U\Leftrightarrow x^{3}\left(P\right)=f\left(x^{1}\left(P\right),x^{2}\left(P\right)\right),x^{1}\left(A\right)=x^{2}\left(A\right)=x^{3}\left(A\right)=0.$$
这里 $x^i(P)$指 $P$ 点的第$i$ 个坐标.并且 $\partial_af(0,0)=0,a=1,2.f$ 在$(0,0)$点的
Hessian 矩阵的行列式值，称为曲面 $S$ 在$A$点的 Gauss 曲率。

这个古怪的定义是因为省略了很多必要的叙述，现在进行补充
* 建立这个古怪的坐标系而非使用原本的原点是为了方斌叙述
* Hessian矩阵对应的二次型是曲面在原点的二阶展开，他的形状直接影响曲面在原点的性质
* Hessian矩阵此时是$2\times2$的，因此一定可以对角化，行列式值就是对角化后对角元（特征值）的积
* 接上一条，这两个特征值的平均值也有含义，称为平均曲率

定理（Gauss绝妙定理）：Gauss曲率被曲面度量（也就是第一基本形式）完全确定

这个定理的绝妙在于，定义Gauss曲率的过程是相当外蕴的，我们依赖一个外部坐标系，计算Hessian矩阵；然而他又是一个内蕴的几何量，与曲面如何在外部坐标系中存在并不直接相关。

Gauss绝妙定理揭示了，仅仅依赖第一基本形式，就携带了相当多的曲面弯曲的信息。
