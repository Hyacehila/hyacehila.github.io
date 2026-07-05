---
title: "解析几何：向量、坐标与平面与直线"
title_en: "Analytic Geometry: Vectors, Coordinates, and Planes"
date: 2024-12-12 15:47:21 +0800
categories: ["Mathematics", "Geometry & Topology"]
tags: ["Learning Notes", "Mathematics", "Analytic Geometry"]
author: Hyacehila
excerpt: "整理向量、坐标、平面与直线、曲面、二次曲面和空间解析几何基础。"
excerpt_en: "Covers vectors, coordinates, planes and lines, surfaces, quadrics, and foundations of spatial analytic geometry."
mathjax: true
hidden: true
permalink: '/blog/2024/12/12/analytic-geometry-notes/'
---
解析几何是我们再一次系统的研究几何学，也是高等数学体系上的第一次几何学内容，因此这里我们将简单介绍几何学究竟包含什么内容

事实上，传统的几何学大致分为欧式几何（平面几何）与非欧几何；而大部分人只学习其中的前者，他们的区别在于公理的不同。

欧式几何的第一课在于小学与初中内容中的几何学，我们先初步的认识了图形，然后基于欧式几何的公理进行推证，导出了许多重要的结论。

欧式几何的第二课则在高中数学之中，我们研究了量化的几何理论，也就是解析几何，在坐标系下系统的研究了几何学。这也是我们本部分希望继续讨论的问题。

欧式几何的第三课则将更为抽象，我们将讨论微分流形的问题，建立更有概括性的几何理论，留给以后讨论。

三个部分都在讨论同一套公理下的几何理论，但是我们使用了一些截然不同的研究方法，因此他们有着各异的用途。
## 向量与坐标
### 向量的概念
本节我们复述几个已经使用了很久的定义，作为解析几何中接受的定义

定义：既有大小又有方向的量叫做向量，或称矢量，简称矢。

向量的大小叫做向量的模，也称向量的长度；向量$\overrightarrow{AB}$与$a$ 的模分别记做$\mid\overrightarrow{AB}\mid$与$\mid a\mid.$

模等于 1 的向量叫做单位向量，与向量$a$具有同一方向的单位向量叫做向量$a$ 的单位向量，常用$a^0$ 来表示。

 模等于 0 的向量叫做零向量，记做 $\mathbf{0}$,它是始点与终点重合的向量，零向量的方向不定。

定义：如果两个向量的模相等且方向相同，那么叫做相等向量，所有的零向
量都相等，向量$a$与$b$相等，记做$a=b.$

定义：如果两个向量模相等但是方向相反，则称互为反向量 记作 $a=-a$

定义：平行于同一直线的一组向量叫做共线向量.零向量与任何共线的向量组共线.

定义：平行于同一平面的一组向量叫做共面向量.零向量与任何共面的向量组共面.

### 加法与数乘
定义：设已知向量 $a,b$,以空间任意一点 $o$ 为始点接连作向量$\overrightarrow OA=\boldsymbol{a},\overrightarrow{AB}=\boldsymbol{b}$ 得一折线$OAB$,从折线的端点$O$到另一端点$B$的向量$\overrightarrow OB=c$,叫做两向量$a$与$b$的和， 记做$c=a+b$.求两向量$a$与$b$的和$a+b$的运算叫做向量加法，这也成为三角形法则

定理：如果以两个向量$\overrightarrow{OA},\overrightarrow{OB}$为邻边组成一个平行四边形 $OACB$ ,那么对角线向量$\overrightarrow OC=\overrightarrow{OA}+\overrightarrow{OB}.$ 称为平行四边形法则

定理：向量加法满足下面的运算率
* $a+b=b+a$
* $(\begin{array}{c}a+b\end{array})+c=a+(\begin{array}{c}b+c\end{array})$
* $a+0=a$
* $a+(-a)=0$

定义：当向量$b$与向量$c$的和等于向量$a$,即$b+c=a$时 ,我们把向量$c$叫做向量$a$与$b$的差，并记做$c=a-b.$由两向量$a$与$b$求它们的差$a-b$的运算叫做向量减法.

最后，向量的加法还可以给出下面的重要不等式
$$\mid a_1+a_2+\cdots+a_n\mid\leqslant\mid a_1\mid+\mid a_2\mid+\cdots+\mid a_n\mid$$

定义：实数$\lambda$与向量$a$的乘积是一个向量，记做$\lambda a$,它的模是$|\lambda a|=$ $|\lambda||a|;\lambda a$的方向，当$\lambda>0$时与$a$相同，当$\lambda<0$时与$a$相反.我们把这种运算叫做数量与向量的乘法，简称为数乘。

已知向量$a$ 和它的单位向量 $a^{0}$ ,下面的等式显然成立：
$$a=\mid a\mid a^0,\quad\text{或}\quad a^0=\frac a{\mid a\mid}.$$

定理：向量数乘满足下面的运算率
* $1\cdot a=a$
* $\lambda\left(\mu a\right)=\left(\lambda\mu\right)a$
* $\left(\lambda+\mu\right)a=\lambda a+\mu a$
* $\lambda\left(a+b\right)=\lambda a+\lambda b$

### 向量的线性关系与向量分解
向量的加法和数量与向量的乘法统称为向量的线性运算.我们知道有限个向量通过线性运算，它的结果仍然是一个向量.

定义：由向量$a_{{_1}},a_{{_2}},\cdots,a_{{_n}}$与实数$\lambda_1,\lambda_{{_2}},\cdots,\lambda_{{_n}}$所组成的向量
$$a=\lambda_1a_1+\lambda_2a_2+\cdots+\lambda_na_n$$
叫做向量 $a_1,a_2,\cdots,a_n$ 的线性组合.

定理：如果向量$e\neq0$,那么向量$r$与向量$e$共线的充要条件是$r$可以用向量$e$ 线性表示 ,或者说 $r$ 是$e$ 的线性组合，即$r= x\boldsymbol{e}$ ,并且系数$x$被 $e,r$ 惟一确定;此时我们称$e$是基

定理：如果向量$e_1,e_2$不共线，那么向量$r$与$e_1,e_2$共面的充要条件是$r$可
以用向量$e_1,e_2$线性表示，或者说向量$r$可以分解成$e_1,e_2$的线性组合，即
$$r=x\boldsymbol{e}_1+y\boldsymbol{e}_2,$$
并且系数$x,y$被$e_1,e_2,r$惟一确定。这时$e_1,e_2$叫做平面上向量的基.

定理：如果向量$e_1,e_2,e_3$不共面 ,那么空间任意向量$r$可以由向量$e_1,e_2,e_3$
线性表示，或者说空间任意向量$r$可以分解成向量$e_1,e_2,e_3$的线性组合 ,即
$$r=xe_1+ye_2+ze_3,$$
并且其中系数$x,y,z$被$e_1,e_2,e_3,r$惟一确定. 这时 $e_1,e_2,e_3$ 叫做空间向量的基.

定义：对于$n$ ($n\geqslant1)$个向量$a_1,a_2,\cdotp\cdotp\cdotp,\boldsymbol{a}_n$,如果存在不全为零的$n$个数$\lambda_1$,
$\lambda_2,\cdots,\lambda_n$,使得$\lambda_{1}a_{1}+\lambda_{2}a_{2}+\cdots+\lambda_{n}a_{n}=0$,那么$n$个向量$a_1,a_2,\cdotp\cdotp\cdotp,a_n$称为线性相关的，不是线性相关的向量称为线性无关的

推论 ：一个向量$a$线性相关的充要条件为$a=0.$

定理：当$n\geqslant2$时，向量$a_1,a_2,\cdots,a_n$线性相关的充要条件是其中有一个向
量是其余向量的线性组合

定理：一组向量中的部分向量线性相关，则这一组向量就线性相关

推论：含有0向量的向量组线性相关

定理：两个向量共线的充要条件是他们线性相关，三个向量共面的条件是他们线性相关，空间中四个及以上的向量总是线性相关。

### 标架与坐标
定义：空间中的一个定点$O$ ,连同三个不共面的有序向量$e_1,e_2,e_3$的全体， 叫做空间中的一个标架，记做$\{O;e_1,e_2,e_3\}$,

如果$e_1,e_2,e_3$都是单位向量，那么$\{O;e_1,e_2,e_3\}$叫做笛卡儿标架 ；   $e_1,e_2,e_3$两两相互垂直的笛卡儿标架叫做笛卡儿直角标架，简称直角标架；    在一般的情况下，$\{O;e_1,e_2,e_3\}$ 叫作仿射标架

在空间中建立一个标架后，空间中任何向量都可以进行如下分解
$$r=xe_1+ye_2+ze_3$$

定义：式中的$x,y,z$称作向量$r$的坐标分量，记做 $r\left\{x,y,z\right\}$ 或$\left\{x,y,z\right\}.$

定义：对于取定了标架 $\{O;e_1,e_2,e_3\}$ 的空间中任意点 $P$,向量$\overrightarrow{OP}$叫做点 $P$ 的向径，或称点$P$的位置向量.向径$\overrightarrow OP$关于标架$\{O;\boldsymbol{e}_1,\boldsymbol{e}_2,\boldsymbol{e}_3\}$的坐标$x,y,z$叫做点$P$关于该标架的坐标

我们可以用坐标来进行向量的运算，这里重写前面的加法与数乘以及一些定理如下

定理：向量的坐标等于其终点的坐标减起点的坐标；两个向量和的坐标等于坐标的和；数乘向量的坐标等于这个数与向量的对应坐标的积。


定理：两个非零向量 $a\{X_1,Y_1,Z_1\},{b}\{X_2,Y_2,Z_2\}$共线的充要条件是对应坐标成比例，即
$$\frac{X_1}{X_2}=\frac{Y_1}{Y_2}=\frac{Z_1}{Z_2}.$$

定理：三个非零向量$a\left\{X_1,Y_1,Z_1\right\},\boldsymbol{b}\left\{X_2,Y_2,Z_2\right\}$和$c\{X_3,Y_3,Z_3\}$共面的充
要条件是
$$\begin{vmatrix}X_1&Y_1&Z_1\\X_2&Y_2&Z_2\\X_3&Y_3&Z_3\end{vmatrix}=0.$$

定理：设有向线段$\overrightarrow P_{1}\overrightarrow{P_{2}}$的始点为$P_1(x_1,y_1,z_1)$,终点为$P_2(x_2,y_2,z_2)($图1-25),那么分有向线段 $P_1P_2$ 成定比$\lambda(\lambda\neq-1)$的分点 $P$ 的坐标是
$$x=\frac{x_{1}+\lambda x_{2}}{1+\lambda},\quad y=\frac{y_{1}+\lambda y_{2}}{1+\lambda},\quad z=\frac{z_{1}+\lambda z_{2}}{1+\lambda}.$$
因此有中点坐标为
$$x=\frac{x_{1}+x_{2}}{2},\quad y=\frac{y_{1}+y_{2}}{2},\quad z=\frac{z_{1}+z_{2}}{2}.$$

### 向量的轴投影
设已知空间的一点$A$与一轴$l$,通过$A$作垂直于轴$l$的平面$\alpha$,我们把这个平面与
轴$l$的交点$A^\prime$叫做点$A$在轴$l$上的射影

定义：设向量$\overrightarrow AB$的始点$A$和终点$B$在轴$l$上的射影分别为点$A^\prime$和$B^{\prime}$,那么
向量$\overrightarrow{A^{\prime}B^{\prime}}$叫做向量$\overrightarrow{AB}$在轴$l$上的射影向量,记做射影向量$\overrightarrow{AB}.$ 并将其大小称为射影

定理：向量$\overrightarrow{AB}$在轴 $l$ 上的射影等于向量的模乘轴与该向量的夹角的余弦：
$$\text{射影}_l\overrightarrow{AB}=\left|\overrightarrow{AB}\right|\cos\theta,\quad\theta=\angle\left(l,\overrightarrow{AB}\right)$$
定理：对于任何向量 $a,b$,有射影$_l(a+b)=\text{射影}_l a+$射影$_l b$

定理：对于任何向量$a$与任意实数$\lambda$有射影$_l(\lambda a)=\lambda\text{射影}_{l}a$


### 向量的数量积
定义：两个向量$a$和$b$的模和它们夹角的余弦的乘积叫做向量$a$和$b$的数量积(也称内积),记做$a\cdot b$ 或 $ab,$即
$$a\cdot b=\mid a\mid\mid b\mid\cos\angle(a,b)$$
如果 $b=a$,那么有 $a\cdot a=|a|^{2}.$ 们把数量积$a\cdot a$叫做$a$的数量平方 ,并记做$a^2.$

定理：两向量$a$与$b$相互垂直的充要条件是$a\cdot b=0.$

定理：向量的数量积满足下面运算规律
* $a\cdot b=b\cdot a$
* $(\lambda a)\cdot b=\lambda(a\cdot b)=a\cdot(\lambda b)$
* $(\begin{array}{c}a+b\end{array})\cdot c=a\cdot c+b\cdot c$
* $a\cdot a=a^2>0\quad(a\neq0)$
* $(\lambda\boldsymbol{a}+\mu\boldsymbol{b})\cdot\boldsymbol{c}=\lambda(\boldsymbol{a}\cdot\boldsymbol{c})+\mu(\boldsymbol{b}\cdot\boldsymbol{c})$

在坐标系，我们有下面的数量积表示方法

定理：设$\text{}a=X_{1}\boldsymbol{i}+Y_{1}\boldsymbol{j}+Z_{1}\boldsymbol{k},\boldsymbol{b}=X_{2}\boldsymbol{i}+Y_{2}\boldsymbol{j}+Z_{2}\boldsymbol{k}$ 则
$$a\cdot b=X_1X_2+Y_1Y_2+Z_1Z_2$$
并且有
$$a\cdot i=X_1,\quad a\cdot j=Y_1,\quad a\cdot k=Z_1$$

定理：设 $\text{}a=X_{}\boldsymbol{i}+Y_{}\boldsymbol{j}+Z_{}\boldsymbol{k}$ 那么
$$\mid a\mid=\sqrt{a^{2}}=\sqrt{X^{2}+Y^{2}+Z^{2}}.$$

定理：空间两点 $P_1(x_1,y_1,z_1),P_2(x_2,y_2,z_2)$ 间的距离是
$$\sqrt{\left(x_{2}-x_{1}\right)^{2}+\left(y_{2}-y_{1}\right)^{2}+\left(z_{2}-z_{1}\right)^{2}}.$$
---

向量与坐标轴(或坐标向量)所成的角叫做向量的方向角，方向角的余弦叫做向量的方向余弦.一个向量的方向完全可由它的方向角来决定.向量的方向余弦也可用向量的坐标来表示
定理：非零向量$a=Xi+Yj+Zk$的方向余弦是
$$\begin{matrix}
 \cos\alpha=\frac{X}{\mid a\mid}=\frac{X}{\sqrt{X^{2}+Y^{2}+Z^{2}}}\\
 \cos\beta=\frac{Y}{\mid a\mid}=\frac{Y}{\sqrt{X^{2}+Y^{2}+Z^{2}}}\\
\cos\gamma=\frac{Z}{\mid a\mid}=\frac{Z}{\sqrt{X^{2}+Y^{2}+Z^{2}}}
\end{matrix}$$
并且有
$$\cos\alpha + \cos\beta + \cos\gamma = 1$$ 式中的$\alpha,\beta,\gamma$分别为向量$a$与$x$轴$,\gamma$轴$,z$轴的交角，即向量$a$的三个方向角

定理：设空间中两个非零向量为 $\boldsymbol{a}\left\{X_1,Y_1,Z_1\right\}$和$\boldsymbol{b}\left\{X_2,Y_2,Z_2\right\}$,那么它们
$$\cos\angle(a,b)=\frac{a\cdot b}{\mid a\mid\mid b\mid}=\frac{X_1X_2+Y_1Y_2+Z_1Z_2}{\sqrt{X_1^2+Y_1^2+Z_1^2}\cdot\sqrt{X_2^2+Y_2^2+Z_2^2}}.$$

推论：向量相互垂直的充要条件是
$$X_1X_2+Y_1Y_2+Z_1Z_2=0.$$
### 向量的向量积
定义：两向量$a$与$b$的向量积(也称外积)是一个向量，记做$a\times b$ 它的模是
$$\mid a\times b\mid=\mid a\mid\mid b\mid\sin\angle(a,b),$$
它的方向与$a$和$b$都垂直，并且按$a,b,a\times b$这个顺序构成右手标架

定理：不共线的向量$a,b$ 的向量积的模，等于以其为边构成的平行四边形的面积

定理：两个向量共线的充要条件为$a\times b=0$

定理：向量积的反交换的 $a\times b=-\left(b\times a\right)$

定理：向量积满足数因子结合率有
$$\lambda(a\times b)=(\lambda a)\times b=a\times(\lambda b)$$
推论：数因子结合率有下面的推论
$$(\lambda\boldsymbol{a})\times(\mu\boldsymbol{b})=(\lambda\mu)(\boldsymbol{a}\times\boldsymbol{b})$$

定理：向量积满足分配率有
$$(a+b)\times c=a\times c+b\times c$$
交换顺序有推论
$$c\times(a+b)=c\times a+c\times b$$
### 向量的混合积
如果我们先把向量$a$ 和$b$ 作出向量积 $a\times b$ ,那么这个向量还可以与第三个向量 $c$ 再作数量积或向量积，在前一种情形有$(a\times b)\cdot c$  后一种情况则有 $(a\times b)\times c$  本章的最后两节我们就讨论这个问题

定义：给定空间的三个向量$a,b,c$,如果先作前两个向量$a$与$b$的向量积， 再作所得的向量与第三个向量$c$ 的数量积，最后得到的这个数叫做三向量$a,b,c$ 的混合积，记做$(a\times b)\cdot c$ 或$(a,b,c)$或$(abc).$

---

下面我们开始讨论混合积的性质

定理：三个不共面向量$a,b,c$的混合积的绝对值等于以$a,b,c$为棱的平行六面体的体积$V$,并且当$a,b,c$构成右手系时混合积是正数；当$a,b,c$构成左手系时，混合积是负数.也就是有
$$(abc)=\varepsilon V$$当$a,b,c$ 是右手系时 $\varepsilon=1;$当$a,b,c$ 是左手系时 $\varepsilon=-1.$

定理：$\text{三向量 }a,b,c\text{ 共面的充要条件是}(abc)=0.$

定理：轮换混合积的三个因子，并不改变它的值，对调任何两个因子要改
变乘积符号，即
$$(\begin{array}{c}abc\end{array})=(\begin{array}{c}bca\end{array})=(\begin{array}{c}cab\end{array})=-(\begin{array}{c}bac\end{array})=-(\begin{array}{c}cba\end{array})=-(\begin{array}{c}acb\end{array}).$$
推论：
$$(a\times b)\cdot c=a\cdot(b\times c).$$

对于坐标表示有

定理：如果有$$\boldsymbol{a}=X_{1} \boldsymbol{i}+Y_{1} \boldsymbol{j}+Z_{1} \boldsymbol{k}, \boldsymbol{b}=X_{2} \boldsymbol{i}+Y_{2} \boldsymbol{j}+Z_{2} \boldsymbol{k}, \boldsymbol{c}=X_{3} \boldsymbol{i}+Y_{3} \boldsymbol{j}+Z_{3} \boldsymbol{k}$$ 那么
$$\begin{array}{l}
\text {  }  \text { ， }\\
(\boldsymbol{a b c})=\left|\begin{array}{lll}
X_{1} & Y_{1} & Z_{1} \\
X_{2} & Y_{2} & Z_{2} \\
X_{3} & Y_{3} & Z_{3}
\end{array}\right|
\end{array}$$
因此，共面的充要条件为
$$\begin{vmatrix}X_1&Y_1&Z_1\\\\X_2&Y_2&Z_2\\\\X_3&Y_3&Z_3\end{vmatrix}=0$$


### 向量的双重向量积
定义：给定空间三向量，先作其中两个向量的向量积，再作所得向量与第
三个向量的向量积，那么最后的结果仍然是一向量，叫做所给三向量的双重向量积；例如$(a\times b)\times c$就是三向量$a,b,c$的一个双重向量积.

双重向量积的上述几何关系可以概括为下面一个定理：
$$(a\times b)\times c=(a\cdot c)b-(b\cdot c)a$$
单独研究双重向量积的性质意义不大，使用该定理转化计算即可


## 轨迹与方程
### 平面曲线普通方程与参数方程
在这里，平面上的曲线(包括直线)都看成具有某种特征性质的点的集合.

在建立了坐标系的平面上，反映为曲线上点的两个坐标$x$与$y$所应满足的相互制约条件，一般用方程
$$F\left(x,y\right)=0$$
来表达

我们也可以换为此形式
$$y=f(x)$$


定义：当平面上取定了坐标系之后，如果一个方程与一条曲线有看关系
1 满足方程的$(x,y)$必是曲线上某一点的坐标；2 曲线上任何一点的坐标$(x,y)$满足这个方程，那么这个方程就叫做这条曲线的方程，而这条曲线叫做这个方程的图形

**这个定义便是平面曲线方程的核心，根据定义我们就可以进行图形与方程的转换**

定义：若取$t(a\leqslant t\leqslant b)$的一切可能取的值，向径$r(t)$的终点总在一条曲线上；反过来，在这条曲线上的任意点，总对应着以它为终点的向径，而这向径可由$t$的某一值$t_0(a\leqslant t_0\leqslant b)$完全决定，那么就把表达式$r(t)$ 叫做曲线的向量式参数方程，其中的 $t$ 为参数.

由于向径都可以向坐标轴投影得到分量，因此更为常用的参数方程形式为
$$\begin{cases}x=x\begin{pmatrix}t\end{pmatrix},\\y=y\begin{pmatrix}t\end{pmatrix}&\end{cases}\begin{pmatrix}a\leqslant t\leqslant b\end{pmatrix}.$$

**消去参数方程的参数就可以得到普通方程，反之找到合适的参数改写普通方程就可以得到参数方程了，通识阶段已经见过很多转化的技巧了**


### 曲面方程
#### 曲面方程基础
空间曲面方程的意义和平面曲线的方程是一样的，那就是在空间建立坐标系之后，把曲面(作为点的轨迹)上的点的特征性质，用点的坐标$x,y$与$z$之间的关系式来表达，一般是用方程
$$F(x,y,z)=0$$
来表达

我们也可以用下面的形式
$$z=f(x,y)$$

曲面方程的基本定义和曲线方程完全一致，这里不另外叙述

**特别的，当没有实数满足曲面方程的时候，我们称之为虚曲面**

#### 曲面参数方程
非常自然的，我们可以改写出下面的曲面参数方程基本形式，也是根据向径的投影来实现有（非常自然的，需要使用两个参数构成参数方程）
$$r(u,v)=x(u,v)e_{1}+y(u,v)e_{2}+z(u,v)e_{3}$$
具体的定义继续参考曲线部分，一般也写为
$$\begin{cases}x=x(u,v)\\\\y=y(u,v)\\\\z=z(u,v).&\end{cases}$$
#### 球与柱坐标系
我们在高中阶段研究解析几何就不少使用三角换元，不过平面的换元格式整体比较简单，很容易就可以研究全面。这里则需要研究空间中的换元情况，我们有一些比较常用的换元技巧，在这里介绍

如果我们把曲面上的点都看做一个空间球上的一点，对于点$M$则有
$$\begin{gathered}\left|\overrightarrow{OM}\right|=\rho\left(\rho\geqslant0\right),\\\angle QOP=\varphi(-\pi<\varphi\leq\pi),\\\angle POM=\theta\left(-\frac{\pi}{2}\leq\theta\leq\frac{\pi}{2}\right)\end{gathered}$$
$\theta$是纵投影角，$\varphi$是平面投影角

据此，我们可以进行下面的参数方程投影
$$\begin{aligned}&x=\rho\cos\theta\cos\varphi,\\&y=\rho\cos\theta\sin\varphi,\\&z=\rho\sin\theta,\end{aligned}$$
**这种特殊的参数方程格式称为球（极）坐标系的参数方程** 反向计算公式为
$$\begin{cases}\rho=\sqrt{x^2+y^2+z^2},\\\\\cos\varphi=\frac{x}{\sqrt{x^2+y^2}},\sin\varphi=\frac{y}{\sqrt{x^2+y^2}}\\\\\theta=\arcsin\frac{z}{\sqrt{x^2+y^2+z^2}}.&\end{cases}$$

我们考虑空间中的柱体，$z$轴无须投影，只需要平面投影以及平面圆位置，我们容易给出下面的参数方程
$$\begin{cases}x=\rho\cos\varphi\\y=\rho\sin\varphi\\z=u&\end{cases},$$
**这种特殊的参数方程格式称为柱坐标系的参数方程** 反向计算公式为
$$\begin{cases}\rho=\sqrt{x^2+y^2},\\\\\cos\varphi=\frac{x}{\sqrt{x^2+y^2}},\quad\sin\varphi=\frac{y}{\sqrt{x^2+y^2}},\\\\u=z.&\end{cases}$$

### 空间曲线方程
空间曲线可以看作两个空间曲面的交线，也就是联立方程
$$\begin{cases}F_1(x,y,z)=0\\\\F_2(x,y,z)=0&\end{cases}$$
他的解就是空间曲线上的点的坐标，由于方程的等价性，**空间曲线的方程当然不具备唯一性，可以适用不同的方程组来表示**

我们也可以建立参数方程，由于方程解限制了一个参数，给出投影形式有
$$r(t)=x(t)e_1+y(t)e_2+z(t)e_3$$
**和空间曲面参数方程比，参数的数量少了一个** 一般我们改写空间曲线参数方程为
$$\begin{cases}x=x\left(t\right),\\y=y\left(t\right),\quad\left(a\leq t\leq b\right)\\z=z\left(t\right)&\end{cases}$$
实际上，空间曲线的参数方程更加方便使用，我们用联立的形式反而更少
## 平面直线与空间
### 平面的方程
#### 平面的点向式方程
在空间给定了一点$M_0$与两个不共线的向量$a,b$,那么通过点$M_0$且与向量$a,b$平行的平面$\pi$就惟一地被确定，向量$a,b$叫做平面$\pi$的方位向量，显然任何一对与平面$\pi$平行的不共线向量都可以作为平面$\pi$的方位向量. 设$M_0$的向径是$r_0$ 取平面上任意一点$M$的向径$r$ 我们容易给出
$$r=r_0+ua+vb$$
这是因为 向径的差是平面上一向量，他一定和平面的方位向量共面，这称为**平面的点向式方程的向量参数形式**

设
$$\boldsymbol{r}_{0}=\left\{x_{0},y_{0},z_{0}\right\},\boldsymbol{r}=\left\{x,y,z\right\};\boldsymbol{a}=\{X_{1},Y_{1},Z_{1}\},\boldsymbol{b}=\{X_{2},Y_{2},Z_{2}\}$$
容易给出等价的点向式有
$$\begin{cases}x=x_0+X_1u+X_2v,\\y=y_0+Y_1u+Y_2v,\\z=z_0+Z_1u+Z_2v.&\end{cases}$$
这称为**平面的点向式方程的坐标参数形式**

根据共面的条件与向量积的特性，可以给出
$$(\begin{array}{c}r-r_0,a,b\end{array})=0$$
也就是
$$\begin{vmatrix}x-x_0&y-y_0&z-z_0\\\\X_1&Y_1&Z_1\\\\X_2&Y_2&Z_2\end{vmatrix}=0$$
这两种形式消去的参变量，是更为一般的形式，我们马上在后面介绍

特别的，我们给出平面的截距形式与三点形式，他是从点向式推出的表示方法，如下

三点形式
$$\begin{vmatrix}x-x_1&y-y_1&z-z_1\\\\x_2-x_1&y_2-y_1&z_2-z_1\\\\x_3-x_1&y_3-y_1&z_3-z_1\end{vmatrix}=0$$
截距形式
$$\frac{x}{a}+\frac{y}{b}+\frac{z}{c}=1$$
#### 平面的一般式方程
我们把前面介绍过的
$$\begin{vmatrix}x-x_0&y-y_0&z-z_0\\\\X_1&Y_1&Z_1\\\\X_2&Y_2&Z_2\end{vmatrix}=0$$
展开后整体就可以得到
$$Ax+By+Cz+D=0$$
我们称为**平面的一般式方程** 其中参数可以这样从点向式计算
$$\left.A=\left|\begin{array}{cc}Y_1&Z_1\\\\Y_2&Z_2\end{array}\right.\right|,B=\begin{vmatrix}Z_1&X_1\\\\Z_2&X_2\end{vmatrix},C=\begin{vmatrix}X_1&Y_1\\\\X_2&Y_2\end{vmatrix},D=-\begin{vmatrix}x_0&y_0&z_0\\\\X_1&Y_1&Z_1\\\\X_2&Y_2&Z_2\end{vmatrix}.$$

特别的，当一般式有下面的特殊性质的时候，方程也有一些特殊性质
* $D=0$ 等价于平面通过原点
* $A,B,C\text{ 中有一为零}$
	* 当$D\ne0$ 平面与系数为0的轴平行
	* 当$D=0$ 平面通过系数为0的轴
* $A,B,C\text{ 中有两个为零的情况}$
	* 当$D\ne0$ 平面与系数为0的平面平行
	* 当$D=0$ 平面就是系数为0的平面
#### 平面的点法式方程
如果在空间给定一点$M_0$和一个非零向量$n$,那么通过点$M_0$且与向量$n$垂直的平
面也惟一地被确定.我们把与平面垂直的非零向量$n$叫做平面的法向量。

根据垂直的性质，我们可以给出方程
$$n\cdot(r-r_0)=0.$$
如果给出坐标，则有
$$A\left(x-x_0\right)+B\left(y-y_0\right)+C\left(z-z_0\right)=0$$
他们都称为**平面的点法式方程**  这种形式不是含参数的，而是方程形式

容易看出，如果记$D=-(Ax_0+By_0+Cz_0)$ 那么有
$$Ax+By+Cz+D=0.$$
也就是说，**一般式方程的系数 $A,B,C$ 就是平面的一个法向量的坐标**，这是二者之间最为重要的联系

如果使用单位法向量，容易给出**平面的法式方程** 有
$$x\cos\alpha+y\cos\beta+z\cos\gamma-p=0.$$
想要把一般式方程转化为此形式，只需要一般式方程乘以
$$\lambda=\frac{1}{\pm|n|}=\frac{1}{\pm\sqrt{A^2+B^2+C^2}}$$
我们称为法式化因子
### 平面与点的位置关系
平面与点只有两种关系，点在平面上与点不在平面上，前者点满足平面方程，这里我们主要研究后者。

#### 点与平面的距离
当使用**平面的法式方程** 的时候，给出距离公式为
$$d=|x_0\cos\alpha+y_0\cos\beta+z_0\cos\gamma-p|.$$
去掉绝对值则为离差

当使用**平面的一般式方程**的时候，给出距离公式为
$$d=\frac{\mid Ax_0+By_0+Cz_0+D\mid}{\sqrt{A^2+B^2+C^2}}$$
去掉绝对值则为离差
#### 平面划分空间
对于**平面的一般式方程** 离差有
$$\delta=\lambda\left(Ax+By+Cz+D\right)$$
也就是
$$Ax+By+Cz+D=\frac{1}{\lambda}\delta.$$
对于平面同侧的点 则离差的符号一致，反之离差的符号相反，也就是说，部分点有$Ax+By+Cz+D>0$ 另一部分点$Ax+By+Cz+D<0$ 平面把空间划分为了两部分
### 平面之间的位置关系
空间两个平面的相关位置有三种情形，即相交、平行和重合
$$\pi_1:A_1x+B_1y+C_1z+D_1=0\:,\\\pi_2:A_2x+B_2y+C_2z+D_2=0\:,$$
那么两平面$\pi_1$与$\pi_2$是相交还是平行或是重合，就取决于由方程构成的方程组是有解还是无解，或是方程仅相差一个不为零的数因子，因此我们就得到了下面的定理：

平面相交的充要条件为
$$A_1:B_1:C_1\neq A_2:B_2:C_2$$
平行的充要条件为
$$\frac{A_1}{A_2}=\frac{B_1}{B_2}=\frac{C_1}{C_2}\neq\frac{D_1}{D_2}$$
重合的充要条件为
$$\frac{A_1}{A_2}=\frac{B_1}{B_2}=\frac{C_1}{C_2}=\frac{D_1}{D_2}$$


特别的，我们还可以讨论下面的性质

定理：两个相交的平面的夹角有
$$\begin{aligned}\cos\angle\left(\pi_{1},\pi_{2}\right)&=\pm\cos\theta=\pm\frac{n_1\cdot n_2}{\mid n_1\mid\mid n_2\mid}\\&=\pm\frac{A_{1}A_{2}+B_{1}B_{2}+C_{1}C_{2}}{\sqrt{A_{1}^{2}+B_{1}^{2}+C_{1}^{2}}\sqrt{A_{2}^{2}+B_{2}^{2}+C_{2}^{2}}}.\end{aligned}$$
其中 $n_1,n_2$ 是两个平面的法向量，特别的，两个平面垂直的充要条件为
$$A_1A_2+B_1B_2+C_1C_2=0.$$
### 空间直线方程
#### 点向式方程
在空间给定了一点$M_0$与一个非零向量$v$ ,那么通过点$M_0$且与向量$v$平行的直线
$l$就惟一地被确定，向量$v$叫做直线$l$的方向向量.  显然， 任何一个与直线$l$平行的非零向量都可以作为直线$l$的方向向量.

我们直接给出
$$r=r_0+t\boldsymbol{v}.$$
称为**直线的向量式参数方程**
$$\begin{cases}x=x_0+Xt,\\\\y=y_0+Yt,\\\\z=z_0+Zt.&\end{cases}$$
称为**直线的坐标式参数方程**

消掉参数$t$ 则
$$\frac{x-x_0}{X}=\frac{y-y_0}{Y}=\frac{z-z_0}{Z}$$
称为**直线的标准式方程**


特别的，据此可以推出直线的两点式方程
$$\frac{x-x_1}{x_2-x_1}=\frac{y-y_1}{y_2-y_1}=\frac{z-z_1}{z_2-z_1}.$$

#### 直线的一般式方程
直线可以表示为平面的交线，也就是方程组
$$\pi_{1}:A_{1}x+B_{1}y+C_{1}z+D_{1}=0;\pi_{2}:A_{2}x+B_{2}y+C_{2}z+D_{2}=0$$
这样的方程组称为**直线的一般式方程**

我们可以根据一般式方程计算标准方程 如下
$$\frac{x-x_0}{\begin{vmatrix}B_1&C_1\\\\B_2&C_2\end{vmatrix}}=\frac{y-y_0}{\begin{vmatrix}C_1&A_1\\\\C_2&A_2\end{vmatrix}}=\frac{z-z_0}{\begin{vmatrix}A_1&B_1\\\\A_2&B_2\end{vmatrix}}.$$
其中
$$x_0=\frac{\begin{vmatrix}B_1&D_1\\\\B_2&D_2\end{vmatrix}}{\begin{vmatrix}A_1&B_1\\\\A_2&B_2\end{vmatrix}},\quad y_0=\frac{\begin{vmatrix}D_1&A_1\\\\D_2&A_2\end{vmatrix}}{\begin{vmatrix}A_1&B_1\\\\A_2&B_2\end{vmatrix}},\quad z_0=0.$$
也就是，我们可以轻松的从**直线的一般式方程**中计算直线上的一点和他的方向向量，从而化为直线的标准方程
### 直线与平面的位置关系
空间直线与平面的相关位置有直线与平面相交，直线与平面平行和直线在平面上的三种情况，现在我们来求直线与平面相关位置成立的条件.设直线$l$与平面$\pi$的方程分别为
$$l:\frac {x- x_0}X= \frac {y- y_0}Y= \frac {z- z_0}Z, \pi :Ax+ By+ Cz+ D= 0$$

定理：对于相交有
$$AX+BY+CZ\neq0$$
对于平行有
$$AX+BY+CZ=0$$
并且
$$Ax_0+By_0+Cz_0+D\neq0$$
对于直线在平面上有
$$AX+BY+CZ=0$$
并且
$$Ax_0+By_0+Cz_0+D=0$$
**他们都是对法向量的关系进行一些计算，包括数量积和向量积**

定理：对于相交的情况，直线与平面的交角的计算公式为
$$\sin\varphi=\mid\cos\theta\mid=\frac{\mid n\cdot v\mid}{\mid n\mid\cdot\mid v\mid}=\frac{\mid AX+BY+CZ\mid}{\sqrt{A^2+B^2+C^2}\cdot\sqrt{X^2+Y^2+Z^2}}.$$
### 直线与点的位置关系
空间直线与点的相关位置有两种情况，即点在直线上与点不在直线上，点在直线上的条件是点的坐标满足直线的方程.当点不在直线上时，我们来求点到直线的距离.

我们不加证明的直接给出公式有
$$d=\frac{\sqrt{\begin{vmatrix}y_0-y_1&z_0-z_1\\Y&Z\end{vmatrix}^2+\begin{vmatrix}z_0-z_1&x_0-x_1\\Z&X\end{vmatrix}^2+\begin{vmatrix}x_0-x_1&y_0-y_1\\X&Y\end{vmatrix}^2}}{\sqrt{X^2+Y^2+Z^2}}$$

### 直线之间的位置关系
#### 直线的位置关系
空间两直线的相关位置有异面与共面，在共面中又有相交、平行与重合的三种情况.现在我们来导出这些相关位置成立的条件. 他们都是从直线的方向向量进行研究的

定理：异面的充要条件为
$$\Delta=\begin{vmatrix}x_2-x_1&y_2-y_1&z_2-z_1\\\\X_1&Y_1&Z_1\\\\X_2&Y_2&Z_2\end{vmatrix}\neq0$$
相交的充要条件为
$$\Delta=0,\quad X_1:Y_1:Z_1\neq X_2:Y_2:Z_2$$
平行的充要条件为
$$X_{1}:Y_{1}:Z_{1}=X_{2}:Y_{2}:Z_{2}\neq(x_{2}-x_{1}):(y_{2}-y_{1}):(z_{2}-z_{1})$$
重合的充要条件为
$$X_1:Y_1:Z_1=X_2:Y_2:Z_2=(x_2-x_1):(y_2-y_1):(z_2-z_1)$$
#### 直线的夹角
在直角坐标系中，直线的夹角的余弦满足
$$\cos\angle(l_{1},l_{2})=\pm\frac{X_{1}X_{2}+Y_{1}Y_{2}+Z_{1}Z_{2}}{\sqrt{X_{1}^{2}+Y_{1}^{2}+Z_{1}^{2}}\cdot\sqrt{X_{2}^{2}+Y_{2}^{2}+Z_{2}^{2}}}.$$
因此可以推出，直线垂直的充要条件为
$$X_1X_2+Y_1Y_2+Z_1Z_2=0.$$
#### 异面直线的距离与公垂线
异面直线的距离是其上的点的最短距离，等于他们的公垂线的长，可以使用下面的公式计算有：
$$d=\frac{\mid(\overrightarrow{M_1M_2},\boldsymbol{v}_1,\boldsymbol{v}_2)\mid}{\mid\boldsymbol{v}_1\times\boldsymbol{v}_2\mid}$$
其中 $M_1,M_2$ 是直线上的点 $\boldsymbol{v}_1,\boldsymbol{v}_2$ 是直线的方向向量；坐标表示为
$$d=\frac{\begin{vmatrix}x_2-x_1&y_2-y_1&z_2-z_1\\X_1&Y_1&Z_1\\X_2&Y_2&Z_2\end{vmatrix}}{\sqrt{\begin{vmatrix}Y_1&Z_1\\Y_2&Z_2\end{vmatrix}^2+\begin{vmatrix}Z_1&X_1\\Z_2&X_2\end{vmatrix}^2+\begin{vmatrix}X_1&Y_1\\X_2&Y_2\end{vmatrix}^2}}.$$

最后，我们讨论公垂线的方程，他满足
$$\begin{vmatrix}x-x_1&y-y_1&z-z_1\\\\X_1&Y_1&Z_1\\\\X&Y&Z\end{vmatrix}=0$$
以及
$$\begin{vmatrix}x-x_2&y-y_2&z-z_2\\\\X_2&Y_2&Z_2\\\\X&Y&Z\end{vmatrix}=0$$
是这两个平面的交线；其中
$$X=\begin{vmatrix}Y_1&Z_1\\Y_2&Z_2\end{vmatrix},Y=\begin{vmatrix}Z_1&X_1\\Z_2&X_2\end{vmatrix},Z=\begin{vmatrix}X_1&Y_1\\X_2&Y_2\end{vmatrix}$$
是$v_1\times v_2$   也就是 $l_0$ 的方向数


### 平面束
定义：通过一条直线的所有平面的集合叫作 **有轴平面束** 那个直线称为平面束的轴

定义：平行与一条直线的所有平面的集合叫作 **平行平面束**

定理：如果两个平面
$$\begin{array}{c}
\pi_{1}: A_{1} x+B_{1} y+C_{1} z+D_{1}=0 \\
\pi_{2}: A_{2} x+B_{2} y+C_{2} z+D_{2}=0
\end{array}$$
相交于直线$L$ 那么通过$L$ 的有轴平面束并以其为轴的方程为
$$l(A_1x+B_1y+C_1z+D_1)+m(A_2x+B_2y+C_2z+D_2)=0$$
其中 $l,m$ 是不全为0的实数

定理：由平面$\pi:Ax+By+Cz+D=0$ 决定的平行平面束的方程为
$$Ax+By+Cz+\lambda=0$$
其中$\lambda$是任意实数
## 二次曲面
### 柱面
#### 一般的柱面
定义：在空间，由平行于定方向且与一条定曲线相交的一族平行直线所生成的曲面叫做柱面，定方向叫做柱面的方向，定曲线叫做柱面的准线，那族平行直线中的每一条直线，都叫做柱面的母线。

设准线方程为
$$\begin{cases}F_1(x,y,z)=0\\\\F_2(x,y,z)=0&\end{cases}$$
母线方向数为 $(X,Y,Z)$   设准线上任一点 $M_1(x_1,y_1,z_1)$ 那么过点$M_1$的母线方程为
$$\frac{x-x_1}{X}=\frac{y-y_1}{Y}=\frac{z-z_1}{Z}$$
并且满足
$$F_1(x_1,y_1,z_1)=0,\quad F_2(x_1,y_1,z_1)=0.$$
根据四个方程 消去 $M_1(x_1,y_1,z_1)$  就可以得到
$$F(x,y,z)=0$$
称为该母线和准线确定的柱面的方程

定理（柱面判别）：在空间直角坐标系中,只含两个元(坐标)的三元方程所表示的曲面是一个柱面，它的母线平行于所缺元(坐标)的同名坐标轴.

因此 如下方程都是柱面
$$\begin{gathered}\frac{x^{2}}{a^{2}}+\frac{y^{2}}{b^{2}}=1,\\\frac{x^{2}}{a^{2}}-\frac{y^{2}}{b^{2}}=1,\\y^{2}=2px.\end{gathered}$$
由于他们在 $xOy$的投影分别是椭圆 双曲线 抛物线 因此他们也称为椭圆柱面 双曲柱面 抛物柱面。通称为二次柱面
#### 空间曲线的射影柱面
设空间曲线
$$L:\begin{cases}F(x,y,z)=0,\\G(x,y,z)=0.&\end{cases}$$
任意从中消去一个元，则有
$$F_{1}(x,y)=0,F_{2}(x,z)=0,F_{3}(y,z)=0$$
根据柱面判别定理，他们都是母线平行于所缺元(坐标)的同名坐标轴的柱面，这称为 **曲线的射影柱面** 曲线
$$\begin{cases}F_1(x,y)=0\\z=0&\end{cases}$$
称为射影曲线
### 锥面
定义： 在空间通过一定点且与定曲线相交的一族直线所生成的曲面叫做锥
面，这些直线都叫做锥面的母线，那个定点叫做锥面的顶点，定曲线叫做锥面的准线.

设锥面的准线为
$$\begin{cases}F_1(x,y,z)=0\\F_2(x,y,z)=0&\end{cases}$$
顶点$A\left(x_0,y_0,z_0\right)$  设$M_1(x_1,y_1,z_1)$是准线上任意一点 则锥面过该点的母线为
$$\frac{x-x_0}{x_1-x_0}=\frac{y-y_0}{y_1-y_0}=\frac{z-z_0}{z_1-z_0}$$
并且
$$F_1(x_1,y_1,z_1)=0,\quad F_2(x_1,y_1,z_1)=0.$$
根据四个方程 消去 $M_1(x_1,y_1,z_1)$  就可以得到
$$F(x,y,z)=0$$
称为该顶点和准线确定的锥面的方程

定理（锥面判别定理）：一个关于$x,y,z$的齐次方程总是表示顶点在坐标原点的锥面

推论：一个关于$(x-x_0),(y-y_0),(z-z_0)$的齐次方程总是表示顶点在$x_0,y_0,z_0$的锥面
### 旋转曲面
定义：在空间，一条曲线$\Gamma$绕着定直线$l$旋转一周所生成的曲面叫做旋转曲
面，或称回转曲面.曲线$\Gamma$叫做旋转曲面的母线，定直线$l$叫做旋转曲面的旋转轴，简称为轴.

设旋转曲线母线的方程为
$$\Gamma:\begin{cases}F_1(x,y,z)=0,\\F_2(x,y,z)=0,&\end{cases}$$
旋转轴为
$$l:\frac{x-x_0}{X}=\frac{y-y_0}{Y}=\frac{z-z_0}{Z}$$
设$M_1(x_1,y_1,z_1)$是母线$\Gamma$上的任意点，那么过$M_{_1}$的纬圆总可以看成是过$M_{_1}$且垂直于旋转轴$l$的平面与以$P_0(x_0,y_0,z_0)$为球心，$\left|\overrightarrow{P_0M_1}\right|$为半径的球面的交线，所以过$M_1(x_1,y_1,z_1)$的纬圆的方程为
$$\begin{aligned}&X(x-x_{1})+Y(y-y_{1})+Z(z-z_{1})=0,\\&\left(x-x_{0}\right)^{2}+\left(y-y_{0}\right)^{2}+\left(z-z_{0}\right)^{2}=\left(x_{1}-x_{0}\right)^{2}+\left(y_{1}-y_{0}\right)^{2}+\left(z_{1}-z_{0}\right)^{2}\end{aligned}$$
又因为点$M_1(x_1,y_1,z_1)$是母线$\Gamma$上的任意点 则
$$F_1(x_1,y_1,z_1)=0,\quad F_2(x_1,y_1,z_1)=0.$$
根据四个方程 消去 $M_1(x_1,y_1,z_1)$  就可以得到
$$F(x,y,z)=0$$
称为该旋转轴和母线确定的旋转曲面的方程

特别的，对于使用坐标轴当作旋转轴的旋转曲线，只需要保留母线中和旋转轴同名的坐标，用其他两个坐标轴的平方和的平方根来替换另一坐标

例子，设母线为 $$\Gamma{:}\begin{cases}F(y,z)=0\\x=0.&\end{cases}$$
以$y$轴为旋转轴旋转，则旋转曲面为
$$F(y,\pm\sqrt{x^{2}+z^{2}})=0$$

根据这种旋转方法旋转椭圆，双曲面，抛物线，圆，就可以分别得到
* 长 扁形旋转椭球面
* 单 双页旋转双曲面
* 旋转抛物面
* 环面
我们在后面继续研究
### 椭球面
定义：下面的方程称为椭球面
$$\frac{x^2}{a^2}+\frac{y^2}{b^2}+\frac{z^2}{c^2}=1$$
他有着下面的基本性质
* 关于三个坐标面 坐标轴 坐标原点对称 分别称为主平面 主轴 中心
* 和三个坐标轴的交点称为顶点
* 顶点间的长度称为轴长 其一般称为半轴长 分别有长中短轴
* 三轴相等称为球面 两轴相等称为长 扁形旋转椭球面 否则称为三轴椭球面

为了了解曲面的形状，需要使用平行切割的截口来判断，一般使用坐标平面切割，如下
$$\begin{cases}\frac{x^2}{a^2}+\frac{y^2}{b^2}=1,\\z=0;&\end{cases}$$
$$\begin{cases}\frac{x^2}{a^2}+\frac{z^2}{c^2}=1,\\y=0;&\end{cases}$$
$$\begin{cases}\frac{y^2}{b^2}+\frac{z^2}{c^2}=1\\x=0.&\end{cases}$$
称为主截线，他们都是椭圆

使用平行坐标平面的平面 如 $z=h$ 切割有
$$\begin{cases}\frac{x^2}{a^2}+\frac{y^2}{b^2}=1-\frac{h^2}{c^2},\\z=h.&\end{cases}$$
**这样的切割可以让我们更好的研究其性质** 该式的具体情形取决于$z$的取值，也就是取决于切割的位置

有时候也用下面的参数方程研究椭球，他不属于常见坐标系，而是一种换元的经验，对于椭球情形很好用
$$\begin{cases}x=a\cos\theta\cos\varphi\\y=b\cos\theta\sin\varphi\\z=c\sin\theta&\end{cases},$$
### 双曲面
#### 单页双曲面
定义：下面的方程称为单页双曲面
$$\frac{x^2}{a^2}+\frac{y^2}{b^2}-\frac{z^2}{c^2}=1$$
他有着下面的基本性质
* 关于三个坐标面 坐标轴 坐标原点对称
* 该双曲面与$z$轴不相交 与其余两轴的交点称为顶点

使用坐标平面切割有
$$\begin{cases}\frac{x^2}{a^2}+\frac{y^2}{b^2}=1,\\z=0;&\end{cases}$$
$$\begin{cases}\frac{x^2}{a^2}-\frac{z^2}{c^2}=1\\y=0;&\end{cases}$$
$$\begin{cases}\frac{y^2}{b^2}-\frac{z^2}{c^2}=1\\x=0.&\end{cases}$$

分别是椭圆，双曲线，双曲线

使用$z=h$ 切割有
$$\begin{cases}\frac{x^2}{a^2}+\frac{y^2}{b^2}=1+\frac{h^2}{c^2}\\z=h,&\end{cases}$$
是一组椭圆，也就是说 **单页双曲面椭圆变动并且沿着双曲线滑行得到的**

使用$y=h$ 切割有
$$\begin{cases}\frac{x^2}{a^2}-\frac{z^2}{c^2}=1-\frac{h^2}{b^2}\\y=h.&\end{cases}$$
当 $|h|\ne b$的时候 他是双曲线，但是实轴平行于不同的坐标轴
当 $|h|=b$的时候 他是两条相交的直线

#### 双页双曲面
定义：下面的方程称为双页双曲面
$$\frac{x^2}{a^2}+\frac{y^2}{b^2}-\frac{z^2}{c^2}=-1$$
他有着下面的基本性质
* 关于三个坐标面 坐标轴 坐标原点对称
* 该双曲面只与$z$轴相交，称为顶点

使用坐标平面截取有
* 与$z=0$不相交
* 与$x=0,y=0$交出两条双曲线

使用 $z=h$ 交 得到
$$\begin{cases}\frac{x^2}{a^2}+\frac{y^2}{b^2}=\frac{h^2}{c^2}-1\\z=h.&\end{cases}$$
根据$h$取值情况，分别是单点 空 椭圆

也就是说 **双页双曲面椭圆变动并且沿着两个双曲线滑行得到的**

单页双曲面和双页双曲面 通称为双曲面
### 抛物面
#### 椭圆抛物面
定义：如下方程称为椭圆抛物面
$$\frac{x^2}{a^2}+\frac{y^2}{b^2}=2z$$
显然椭圆抛物面关于xOz与yOz坐标面对称，也关于$z$轴对称，但是它没
有对称中心 , 它与对称轴交于点$(0,0,0)$ , 这点叫做椭圆抛物面的顶点

使用 $x=0,y=0$ 切割有
$$\begin{cases}x^2=2a^2z,\\y=0&\end{cases}$$
$$\begin{cases}y^2=2b^2z,\\x=0,&\end{cases}$$
称为主抛物线

使用 $z=h$ 切割
$$\begin{cases}\frac{x^2}{2a^2h}+\frac{y^2}{2b^2h}=1\\z=h.&\end{cases}$$
总是一个椭圆，因此 **椭圆抛物面是椭圆沿着抛物线运动得到的**

#### 双曲抛物面
定义：如下方程称为双曲抛物面
$$\frac{x^2}{a^2}-\frac{y^2}{b^2}=2z$$
显然双曲抛物面关于xOz与yOz坐标面对称，也关于$z$轴对称，但是它没
有对称中心

使用$z=0$截取
$$\begin{cases}\frac{x^2}{a^2}-\frac{y^2}{b^2}=0\\z=0.&\end{cases}$$
这是一组相较于原点的直线

使用 $x=0,y=0$ 切割得到两条抛物线，是两个主抛物线

使用 $z=h$ 切割
$$\begin{cases}\frac{x^2}{2a^2h}-\frac{y^2}{2b^2h}=1\\z=h.&\end{cases}$$
这是一个变动的双曲线；当$h>0$ 时，双曲线的实轴与 $x$ 轴平行，虚轴与 $y$ 轴平行，顶点( $\pm a\sqrt2h,0,h)$在主抛物线上；当$h<0$时，双曲线的实轴与$y$轴平行，虚轴与$x$轴平行，顶点 $( 0,\pm b\sqrt{-2h},h$)在主抛物线上

因此，曲面被xOy平面分割成上下两部分，上半部沿$x$轴的两个方向上升，下半部沿$y$轴的两个方向下降，曲面的大体形状像一只马鞍子，所以双曲抛物面也叫做马鞍曲面

### 直母线
我们在前面已经看到，柱面与锥面都可以由一族直线所生成，这种由一族直线所生成的曲面叫做直纹曲面，而生成曲面的那族直线叫做这曲面的一族直母线.柱面与锥面都是直纹曲面.

我们又在本文“双曲面”部分 本文“抛物面”部分中看到单叶双曲面与双曲抛物面上都包含有直线.下面我们来证明，这两曲面不仅含有直线，而且可以由一族直线所生成，因而它们都是直纹曲面.

#### 对于单页双曲面
对于单页双曲面
$$\frac{x^2}{a^2}+\frac{y^2}{b^2}-\frac{z^2}{c^2}=1$$
我们可以给出 $u$ 族直母线方程
$$\begin{cases}\frac{x}{a}+\frac{z}{c}=u\left(1+\frac{y}{b}\right),\\\frac{x}{a}-\frac{z}{c}=\frac{1}{u}\left(1-\frac{y}{b}\right),&\end{cases}$$
取 $u \to 0,u\to \infty$ 有
$$\begin{cases}\frac{x}{a}+\frac{z}{c}=0\\1-\frac{y}{b}=0&\end{cases}$$
$$\begin{cases}\frac{x}{a}-\frac{z}{c}=0\\1+\frac{y}{b}=0.&\end{cases}$$
他们统称为 $u$ 族直母线

对应的，还可以给出$v$族直母线
$$\begin{cases}\frac{x}{a}+\frac{z}{c}=v\left(1-\frac{y}{b}\right),\\\frac{x}{a}-\frac{z}{c}=\frac{1}{v}\left(1+\frac{y}{b}\right)&\end{cases}$$

**对于单页双曲线上的点，两族直母线各有一点通过该点，任意一族直母线就可以生成整个双曲线，只是有两种生成方式**

#### 双曲抛物面
对于双曲抛物面
$$\frac{x^2}{a^2}-\frac{y^2}{b^2}=2z$$
我们可以同样的给出两族直母线为
$$\begin{cases}\frac{x}{a}+\frac{y}{b}=2u,\\\\u\left(\frac{x}{a}-\frac{y}{b}\right)=z&\end{cases}$$
以及
$$\begin{cases}\frac{x}{a}-\frac{y}{b}=2v,\\\\v\left(\frac{x}{a}+\frac{y}{b}\right)=z.&\end{cases}$$
**对于双曲抛物面上的点，两族直母线各有一点通过该点，任意一族直母线就可以生成整个双曲线，只是有两种生成方式**

#### 相关性质
单页双曲面和双曲抛物面在建筑学中频繁被使用，其直母线可以构成建筑的框架，而整体仍旧能够保留优美的弧形

定理：单叶双曲面上异族的任意两直母线必共面，而双曲抛物面上异族的任意两直母线必相交.

定理：单叶双曲面或双曲抛物面上同族的任意两直母线总是异面直线，而双曲抛物面同族的全体直母线平行于同一平面
