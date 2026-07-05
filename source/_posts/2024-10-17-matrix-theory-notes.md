---
title: "矩阵论：矩阵范数、谱半径与Hermite 矩阵"
title_en: "Matrix Theory: Matrix Norms, Spectral Radius, and Hermitian Matrices"
date: 2024-10-17 15:24:42 +0800
categories: ["Mathematics", "Algebra & Matrix Theory"]
tags: ["Learning Notes", "Mathematics", "Matrix Theory", "Linear Algebra"]
author: Hyacehila
excerpt: "整理矩阵范数、谱半径、Hermite 矩阵、矩阵分解、矩阵函数和相关矩阵理论。"
excerpt_en: "Covers matrix norms, spectral radius, Hermitian matrices, matrix decompositions, matrix functions, and related matrix theory."
mathjax: true
hidden: true
permalink: '/blog/2024/10/17/matrix-theory-notes/'
---
随着科学技术的迅速发展，古典的线性代数知识已不能满足现代科技的需要，矩阵的理论和方法业已成为现代科技领域必不可少的工具。

诸如数值分析、 优化理论、微分方程、概率统计、控制论、力学、电子学、网络等学科领域都与矩阵理论有着密切的联系，甚至在经济管理、金融、保险、社会科学等领域，矩阵理论和方法也有着十分重要的应用。

这里我们基于研究生最基本的矩阵论教材，讨论矩阵论中最核心的知识，至于那些已经在[高等代数1 代数学基础](/blog/2023/03/17/advanced-algebra-foundations-notes/) [高等代数2 矩阵和线性空间](/blog/2023/06/17/advanced-algebra-matrices-linear-spaces-notes/)   [高等代数3 线性变换与欧式空间](/blog/2024/12/07/advanced-algebra-linear-transformations-euclidean-spaces-notes/)  中叙述过的知识，不再重复的解释。

**本文前文仍旧是一个偏向于基础内容的叙述** 是对高等代数学的补充。并不是矩阵分析理论的全部，我们还有很多深奥的矩阵分析理论可以学习。这里我们只强调矩阵分析的基础知识，而不侧重其再不同领域的应用与那些新兴理论，更难并且更少被用到的矩阵知识留给矩阵分析研究。
## 矩阵理论补充
### 矩阵范数
这是我们在高等代数学中没有介绍的关于矩阵的初等知识，他也和[泛函分析](/blog/2023/09/11/functional-analysis-notes/)联系的非常紧密，因此选择在矩阵论中补充。

矩阵的范数的前置知识是向量的范数，我们已经在[泛函分析中的线性赋范空间和内积空间](/blog/2023/09/11/functional-analysis-notes/)中介绍过了

#### 矩阵范数的定义与性质
现在我们将范数这个衡量向量大小的概念推广到矩阵之上，将矩阵拉直，我们就能或者一个很长的一维向量，就可以把向量范数的概念推广过来。又因为矩阵存在乘法，而非仅仅有加法和数乘，因此需要而外补充公理来限制他

定义：设$A\in C^{m\times n}$  按照某种法则规定$A$上的一个实值函数，记作$||A||$  如果他满足下面四个条件
* 非负性：$:\text{ 如果 }A\neq\mathbf{0},\text{则}\parallel A\parallel>0;\text{ 如果 }A=\mathbf{0},\text{则}\parallel A\parallel=0.$
* 齐次性：$\text{对任意的 }k\in\mathbb{C},\parallel kA\parallel=\mid k\mid\parallel A\parallel$
* 三角不等式：$\text{对任意 }A,B\in\mathbb{C}^{m\times n},\parallel A+B\parallel\leqslant\parallel A\parallel+\parallel B\parallel$
* 次乘性：当矩阵乘积$AB$有意义时，有$\parallel AB\parallel\leqslant\parallel A\parallel\parallel B\parallel$
则称$||A||$  是矩阵$A$的范数

次乘性保证了幂零矩阵$A^2=0$ 不与非负性冲突，以及矩阵级数的合理收敛性，其是非常合理的要求。

当我们把一个$m\times n$维矩阵看成一个拉直后的向量的时候，我们可以很自然的给出一些矩阵范数，他们都限制在方阵了方阵前提下
* $\parallel A\parallel_{m_1}=\sum_{i=1}^n\sum_{j=1}^n\mid a_{ij}\mid$
* $\parallel A\parallel_{m_{\infty}}=n\bullet\max_{i,j}\mid a_{ij}\mid$
* $\parallel A\parallel_{m_2}=\left(\sum_{i=1}^n\sum_{j=1}^n\mid a_{ij}\mid^2\right)^{\frac{1}{2}}$
**特别的，方阵的2-范数非常的常用，我们称为Frobenius范数 写做$||A||_F$**

我们可以不加证明的给出矩阵范数的重要定理：和向量范数一样，所有的矩阵范数都是等价的
#### 算子范数
算子的概念我们在[泛函分析中的线性算子](/blog/2023/09/11/functional-analysis-notes/)中介绍过，他只是一种空间上的映射，在矩阵论中我们继续研究算子，不过只需要研究矩阵就足够了。

设$A\in C^{m\times n}$  ，$x\in C^n$  则有$Ax\in C^m$  这其实就是算子$A$在两个不同维数的向量空间上进行了映射，如果我们把$x$看作矩阵 矩阵范数的次乘性有
$$\parallel Ax\parallel\leqslant\parallel A\parallel\parallel x\parallel$$
也就是说
$$\parallel A\parallel\geqslant\frac{\parallel Ax\parallel}{\parallel x\parallel}$$
不等式右侧是向量范数的比，左侧是尚未定义的矩阵范数，因此从向量范数定义矩阵范数有
$$\parallel A\parallel=\sup_{\parallel x\parallel\neq0}\frac{\parallel Ax\parallel}{\parallel x\parallel}$$
当$||x||=1$的时候，由于单位球面一定是有界闭集合，因此右侧是连续函数 sup可以换作max

如果我们同时定义了矩阵范数与向量范数，我们应该在定义时保证前述的不等式成立，这被我们称为两个范数相容。

对于已经规定了向量范数的情况，我们将矩阵范数
$$\parallel A\parallel=\sup_{\parallel x\parallel\neq0}\frac{\parallel Ax\parallel}{\parallel x\parallel}=\max_{\parallel x\parallel=1}\parallel Ax\parallel$$
称为从向量范数诱导得到的矩阵范数，或称为算子范数

定理：设$A\in C^{m\times n}$ $x\in C^n$ 我们把向量$x$的1范数 2范数以及无穷范数均向算子范数诱导可以分别得到
* $\parallel A\parallel_1=\max_{i=1}^m\mid a_{ij}\mid\text{(称为列范数)}$
* $\parallel A\parallel_2=\sqrt{\lambda_{\max}(A^\mathrm{H}A)}(\text{称为谱范数)}$ 其中算子$\lambda_{max}$是算矩阵的最大特征值
* $\parallel A\parallel_\infty=\max_i\sum_{i=1}^n\mid a_{ij}\mid\text{(称为行范数)}$

#### 谱范数与谱半径
我们知道，矩阵的算子范数$\|A\|_2$称为$A$ 的谱范数，它的值是通过矩阵 $A^\mathrm{H}A$ 的最大特征值来计算的，尽管求特征值比较麻烦，但这种范数有非常好的性质，所以在矩阵分析和系统理论中常常使用.下面专门讨论谱范数的性质

定理：设$A\in C^{m\times n}$  则
* $\parallel A\parallel_2=\max_{\parallel x\parallel_2=\parallel y\parallel_2=1}\mid y^\mathrm{H}Ax\mid,x\in\mathbb{C}^n,y\in\mathbb{C}^m$
* $\parallel A^\mathrm{H}\parallel_2=\parallel A\parallel_2$
* $\parallel A^\mathrm{H}A\parallel_2=\parallel A\parallel_2^2$

定理：设$A\in C^{m\times n},U\in C^{m\times m},V\in C^{n\times n},U^{H}U=I_{m} V^HV=I_n$ 则
$$\parallel UAV\parallel_2=\parallel A\parallel_2$$

定理：对于任意的算子范数，都有设$A\in C^{n\times n}$ 若$||A||<1$ 则 $I-A$为非奇异矩阵，并且$\parallel(I-A)^{-1}\parallel\leqslant(1-\parallel A\parallel)^{-1}$

定义：设$A\in C^{n\times n}$ 若$\lambda_1,\lambda_2,...,\lambda_n$是他的特征值，我们称
$$\rho(A)=\max_i|\lambda_i|$$
是矩阵$A$的谱半径

定理（特征值上界定理）：对于任意$A\in C^{n\times n}$ 总有 $\rho(A)\leqslant\parallel A\parallel$ 也就是谱半径小于任何一种范数

定理：设$A\in C^{n\times n}$ 并且$A$是正交矩阵，则有$\rho(A)=\parallel A\parallel_2$

定理：对于任意非奇异矩阵$A\in C^{n\times n}$ 并且$A$是正交矩阵 $A$的谱范数为
$$\rho(A)=\parallel A\parallel_2=\sqrt{\rho(A^\mathrm{H}A)}=\sqrt{\rho(AA^\mathrm{H})}$$


### 埃尔米特矩阵与变换
在[高等代数3 线性变换与欧式空间中的欧式空间](/blog/2024/12/07/advanced-algebra-linear-transformations-euclidean-spaces-notes/)中，我们介绍一类特殊的变换，保持欧式空间结构的不变性，也就是[高等代数3 线性变换与欧式空间中的正交变换](/blog/2024/12/07/advanced-algebra-linear-transformations-euclidean-spaces-notes/)。事实上，还有一类有趣的变换以及他的矩阵我们没有介绍，就是埃尔米特变换。

我们从对称变换开始讨论，然后讨论埃尔米特变换以及矩阵，最后讨论我们在[高等代数2 矩阵和线性空间中的二次型](/blog/2023/06/17/advanced-algebra-matrices-linear-spaces-notes/)的拓展，埃尔米特正定矩阵。

#### 对称变换与对称矩阵
定义：设$A$是欧式空间$V$上的一个线性变换，对于$V$中的任意元素$x,y$ 都有
$$(A(x),y)=(x,A(y))$$
则称$A$是欧式空间$V$上的一个对称变换

推论：根据定义我们就能容易的证明，对称变换$A$在标准正交基下的矩阵是对称矩阵，也就是$A^T=A$  其逆命题同样成立。

根据对称矩阵的性质，我们可以给出两个运算性质
* 如果$A$是对称矩阵，那么$(A(x),y)=(x,A(y))$
* 如果$A$不是对称矩阵，那么$(A(x),y)=(x,A^T(y))$

#### 埃尔米特变换与埃尔米特矩阵
定义：设$A$是酉空间（[高等代数3 线性变换与欧式空间中的酉空间](/blog/2024/12/07/advanced-algebra-linear-transformations-euclidean-spaces-notes/)）$V$上的一个线性变换，对于$V$中的任意元素$x,y$ 都有
$$(A(x),y)=(x,A(y))$$
则称$A$是酉空间$V$上的一个埃尔米特变换

我们将矩阵$A$的元素先共轭再转置整个矩阵称为$A^H$，也就是$\bar{A^{T}}=A^H$，这是为了简化后面将要出现的符号

推论：如果埃尔米特变换$A$在标准正交基下的矩阵为$A$，那么则有$A^H=A$ ，我们将满足这样条件的矩阵称为埃尔米特矩阵。如果满足$A^H=-A$ 则称为反埃尔米特矩阵

定理（Schur定理）：任何$n$阶矩阵都酉相似于一个上三角矩阵，也就是对于任意$n$阶矩阵$A$，存在一个$n$阶酉矩阵$U$和一个上三角矩阵$T$ ，满足
$$U^HAU=T$$
$T$的对角线的元素就是$A$的特征值，顺序根据情况确定

推论：如果$A$为埃尔米特矩阵，则$A$一定酉相似于对角阵，其对角元（$A$的特征值均为实数）

#### 埃尔米特正定，半正定矩阵
定义：设$A$为$n$阶埃尔米特矩阵，如果对任意$n$维复向量$x$都有
$$x^\text{н}Ax\geqslant0,$$
则称$A$为埃尔米特非负定(半正定)矩阵，记作$A\geqslant0.$如果对任意$n$维非零复向量$x$都有
$$x^\text{н}Ax>0,$$
则称$A$为埃尔米特正定矩阵，记作$A>0.$

我们之所以重复使用了正定的概念，是因为欧式空间上的对称矩阵研究了二次型有关的问题[高等代数2 矩阵和线性空间中的二次型](/blog/2023/06/17/advanced-algebra-matrices-linear-spaces-notes/) 而埃尔米特矩阵就是酉空间上的对称矩阵，因此我们研究埃尔米特二次型。

根据这个定义，我们容易给出
* 单位矩阵$I>0$
* $A>0,k>0$ 则 $kA>0$
* $A\ge0,B\ge0$ 则 $A+B\ge 0$

定理：矩阵$A$为正定矩阵的充分必要条件为$A$的所有特征值都是正数（半正定则改为非负数）

定理：矩阵$A$为正定矩阵的充分必要条件为存在$n$阶非奇异矩阵$P$，使得$A=P^HP$ （半正定则把非奇异去除）

定理：正定矩阵$A$的各阶顺序主子阵都是正定矩阵

定理：矩阵$A$为正定矩阵的充分必要条件为$A$的各阶顺序主子式都是正数或所有主子式都大于0

定理：设$A,B$是$n$阶埃尔米特矩阵，且$B>0$ 则存在非奇异矩阵$Q$使得
$$Q^\mathrm{H}BQ=I,\quad Q^\mathrm{H}AQ=\mathrm{diag}(\lambda_1,\lambda_2,\cdotp\cdotp\cdotp,\lambda_n).$$

#### 埃尔米特矩阵的特征值
埃尔米特矩阵的特征值有一些非常值得研究的性质，我们这里简单介绍一些

本节所使用的矩阵大于和小于 如$A>B,A\ge B$ 均意味着 其差是正定或者半正定的。

定理：设$A$是$n$阶埃尔米特矩阵，则
$$\lambda_{\min}(A)\boldsymbol{I}\leqslant A\leqslant\lambda_{\max}(A)\boldsymbol{I}$$
其中$\lambda_{\min}(A),\lambda_{\max}(A)$ 是矩阵的最小和最大特征值

定义：设$A$为$n$阶埃尔米特矩阵，对$\forall x\in\mathbb{C}^n$且$x\neq\mathbf{0}$,称$$R(x)=\frac{x^\mathrm{H}Ax}{x^\mathrm{H}x},\quad x\neq0$$为埃尔米特矩阵 $A$ 的瑞利商

定理（瑞利商的基本性质）：设$A$是$n$阶埃尔米特矩阵，其特征值为$\lambda_1\geqslant\lambda_2\geqslant\cdotp\cdotp\cdotp\geqslant\lambda_n$,则
* $R(k\boldsymbol{x})=R(\boldsymbol{x}),k\in\mathbb{C},k\neq0;$
* $\lambda _n\leqslant R( x) \leqslant \lambda _1$ $, x\neq 0$ ;
* $\lambda_{1}=\operatorname*{max}_{x\neq0}R\left(x\right),\lambda_{n}=\operatorname*{min}_{x\neq0}R\left(x\right).$

定理（极大极小定理）：设$A$是$n$阶埃尔米特矩阵，其特征值为$\lambda_1\geqslant\lambda_2\geqslant\cdotp\cdotp\cdotp\geqslant\lambda_n$, $V_i$是$C^n$的$i$维子空间，则
$$\lambda_i=\max_{V_i}\underset{x\neq0,x\in V_i}{\operatorname*{\operatorname*{min}}}R(x),\lambda_i=\min_{V_{n-i+1}}\max_{\begin{array}{c}x\in V_{n-i+1},x\neq0\end{array}}R(x)$$

使用极大极小定理，我们可以研究埃尔米特矩阵的元素发生微小变化的时候，相应的矩阵特征值的变换范围

定理： 设$A,E$均为$n$阶埃尔米特矩阵$,B=A+E$,且$A,B$和$E$的特征值分别为
$\lambda_1\geqslant\cdots\geqslant\lambda_n,\mu_1\geqslant\cdots\geqslant\mu_n$ 和$\varepsilon_1\geqslant\cdots\geqslant\varepsilon_n$,则
$$\lambda_i+\varepsilon_n\leqslant\mu_i\leqslant\lambda_i+\varepsilon_1,\quad i=1,2,\cdotp\cdotp\cdotp,n.$$

### 摄动分析
在数值计算中，通常存在两类误差影响计算结果的精度，即计算方法引起的截断误差和计算环境引起的舍人误差.为了分析这些误差对数学问题解的影响，人们将其归结为原始数据的扰动(或摄动)对解的影响.下面我们将分别研究在线性方程组求解和矩阵特征值求解过程中，因原始数据的摄动而引起问题的解有多大的变化，即研究问题解的稳定性.

#### 病态方程组与病态矩阵
考虑下面这个简单的二元方程组
$$\begin{bmatrix}1&0.99\\0.99&0.98\end{bmatrix}\begin{bmatrix}x_1\\x_2\end{bmatrix}=\begin{bmatrix}1\\1\end{bmatrix}$$
其精确解为 $x_1=100,x_2=-100$

将方程组进行微小的摄动，由于真实实验的误差等因素，这种摄动非常常见
$$\begin{bmatrix}1&0.99\\0.99&0.99\end{bmatrix}\begin{pmatrix}x_1+\delta x_1\\\\x_2+\delta x_2\end{pmatrix}=\begin{bmatrix}1\\1.001\end{bmatrix}$$

则方程组的精确解变为$x_1+\delta x_1=-0.1,x_2+\delta x_2=\frac{10}{9}$

能看出，虽然我们对原始系数的摄动很微小，但是方程组的解变化巨大，这种现象就是病态。

定义：如果系数矩阵$A$或常数项$b$的微小变化，引起方程组$Ax=b$解的巨大变化，则称方程组为病态方程组，其系数矩阵$A$就叫做对应于解方程组(或求逆)的病态矩阵； 反之，方程组就称为良态方程组，$A$称良态矩阵。

应该指出，谈到“病态矩阵”概念时，必须明确它是对什么而言的.因为对于解方程组(或求逆)来说是病态矩阵，对于求特征值来说并不一定是病态的，反之亦然.所以我们不能笼统地说某个矩阵是“病态”的。

#### 矩阵的条件数
了解了病态的概念之后，我们开始研究衡量一个矩阵病态的标准。至于标准为什么是这样不再讨论。

定义：设 A 为非奇异矩阵，称数 cond$(\boldsymbol{A})=\left\|\boldsymbol{A}^{-1}\right\|_{\rho}\left\|\boldsymbol{A}\right\|_{\rho}(p=1,2$ 或$\infty)$为矩阵 $A$ 的条件数.

由此看出矩阵的条件数与范数有关，它刻画了方程组解的相对误差可能的放大率，我们一般认为远大于1的条件数是不良好的性质，对应小于1的条件数是比较好的。但是缺少更为确切的标准。

最常用的条件数是谱条件数
$$\begin{aligned}\operatorname{cond}(A)_2&=\parallel A\parallel_2\parallel A^{-1}\parallel_2\\&=\sqrt{\frac{\lambda_{\max}(A^{\mathrm{H}}A)}{\lambda_{\min}(A^{\mathrm{H}}A)}}.\end{aligned}$$
这是线性回归基础中的条件数判别法的推广

#### 矩阵特征值的摄动分析
前面我们是针对线性方程组的求解问题进行研究，现在考虑求矩阵特征值的摄动分析。高阶矩阵特征值的精确求解是比较困难的，因此我们研究近似。

定义：设$A=(a_{ij}$ )为任一$n$阶复数矩阵，复平面上的$n$个圆盘
$$G_i( \boldsymbol{A}) : \mid z- a_{ii}\mid \leqslant R_i$, $i= 1, 2, \cdots , n$$
这里以$R_{i}=\sum_{j=1}\mid a_{ij}\mid$为半径的圆(即圆盘的边界),称为矩阵 $A$ 的 Gerschgorin 圆，简称盖尔圆.

定理（盖尔定理又称圆盘定理）：设 $A=(a_{\mathrm{i}j})\in\mathbb{C}^{n\times n}$,则
* $A$ 的特征值都在 $n$ 个圆盘$G_i(\mathbf{A})$的并集内(换句话说，$\mathbf{A}$ 的每个特征值都落在$A$ 的某个圆盘之内),即
$$\lambda(A)\subseteq\bigcup_{i=1}^nG_i(A)\:;$$
* 矩阵$A$的任一个由$m$个圆盘组成的连通区域中，有且只有$A$的$m$个特征值(当$A$的主对角线上有相同元素时，则按重复次数计算，有相同特征值时也需按重复次数计算).

圆盘定理可以帮助我们估计特征值的大致所在区间，复平面上的点就是一个复数，可能是复矩阵的特征值

定义：设 $A\in\mathbb{C}^{n\times n}$,并且存在可逆矩阵 $P$ 使得 $P^{-1}AP=\operatorname{diag}(\lambda_1,\ldots,\lambda_n)$,则称$\parallel\boldsymbol{P}^{-1}\parallel\parallel\boldsymbol{P}\parallel$为矩阵$A$关于特征值问题的“条件数”,简称特征条件数，记为$\zeta(\boldsymbol{P}).$ 若 $\zeta(\boldsymbol{P})=\parallel\boldsymbol{P}^{-1}\parallel\parallel\boldsymbol{P}\parallel$不是很大，则 $\boldsymbol{A}$ 的特征值问题是良态的

对于埃尔米特矩阵（酉空间的对称矩阵）我们还有
定理：设$A,E$均为$n$阶埃尔米特矩阵$,B=A+E$,且$A,B$和$E$的特征值分别为
$\lambda_1\geq\lambda_2\geq\cdots\geq\lambda_n,\mu_1\geqslant\mu_2\geqslant\cdots\geq\mu_n$ 和$\varepsilon_1\geq\varepsilon_2\geq\cdots\geqslant\varepsilon_n$,则
$$\mid\lambda_i-\mu_i\mid\leqslant\parallel E\parallel_2,\quad i=1,2,\cdotp\cdotp\cdotp,n.$$

能看出埃尔米特矩阵的特征条件数为1，这意味着埃尔米特矩阵在特征值问题上，都是良态的。
## 矩阵分解
矩阵分解对矩阵理论及近代计算数学的发展起了关键作用.所谓矩阵分解，就是将一个矩阵写成结构比较简单的或性质比较熟悉的另一些矩阵的乘积。

我们在高等代数中研究过对角化，研究过Joran标准型等等问题，他们都属于矩阵分解的一种。不过以前的理论更少的关注应用，难以实现简化计算与深入理论的问题。

因此本章对常见的矩阵分解进行汇总研究，他们有的基于曾经学过的理论，还有一些则完全陌生，但是他们在近代计算数学中都发挥了很重要的作用

### 三角分解
#### 高斯消元
我们在[高等代数1 代数学基础中的高斯消元](/blog/2023/03/17/advanced-algebra-foundations-notes/)中介绍过求解方程组的矩阵消元，实际上就是将初始矩阵变为了一个上三角矩阵，并且只使用了行变换，理解高斯消元的本质就可以让我们对三角分解有完整的理解。

对于一个$n$元的线性方程组，我们直接用矩阵形式表示
$$Ax=b$$
高斯消元就是在利用矩阵的初等行变换将矩阵$A$变换为上三角矩阵，我们假设全程不交换行的顺序（这也是很正常的选择），那么每一次行变换就是左乘一个对应的初等矩阵。由于在矩阵分解领域我们只考虑方阵，那么高斯消元的最终结果为
$$L^{(n-1)}\cdots L^{(2)}L^{(1)}A^{(1)}=\begin{bmatrix}a_{11}^{(1)}&a_{12}^{(1)}&\cdots&a_{1n}^{(1)}\\&a_{22}^{(2)}&\cdots&a_{2n}^{(2)}\\&&\ddots&\vdots\\&&&a_{nn}^{(n)}\end{bmatrix}=A^{(n)}$$

事实上，想要消元到这种形式，我们需要保证对角线元素不为0，因此可以给出下面的定理

定理：当$n$阶矩阵$A$的前$n-1$阶顺序主子式都不为0的时候，则对角线元素不为0，高斯消元可以进行到底。

#### 矩阵的三角分解
当前一节所叙述的分解可以正常进行到底时，记$U=A^{n}$ 则
$$L^{(n-1)}\cdot\cdot\cdot L^{(2)}L^{(1)}A=U$$
也就是
$$A=(L^{(1)})^{-1}(L^{(2)})^{-1}\cdots(L^{(n-1)})^{-1}U$$

根据逆矩阵的定义知道 $(L^{(1)})^{-1}$ 是下三角矩阵，因此其连乘积也是下三角矩阵，也就是说
$$L=(L^{(1)})^{-1}(L^{(2)})^{-1}\cdotp\cdotp\cdotp(L^{(n-1)})^{-1}=\begin{bmatrix}1&&&&&&\\l_{21}&1&&&&&\\l_{31}&l_{32}&1&&&&\\l_{41}&l_{42}&l_{43}&\ddots&&&\\\vdots&\vdots&\vdots&\ddots&1&&\\l_{ln}&l_{n2}&l_{n3}&\cdots&l_{n,n-1}&1\end{bmatrix}$$
是对角线元素均为1的下三角矩阵

综上所述，我们可以将初始矩阵$A$分解
$$A=LU$$
初始矩阵被分解为一个上三角矩阵和一个下三角矩阵的乘积

定义：如果方阵$A$可分解成一个下三角矩阵$L$和一个上三角矩阵$U$的乘积，则称$A$可作三角分解或$LU$分解。如果$L$是单位下三角矩阵，$U$为上三角矩阵，此时的三角分解称为杜利特(Doolittle)分解；若$L$ 是下三角矩阵，而$U$ 是单位上三角矩阵，则称三角分解为克劳特(Crout)分解。

从定义能看出，矩阵的三角分解一定不是唯一的，最起码还有Doolittle分解和Crout分解两种，事实上
$$A=LU=LDD^{-1}U=(LD)(D^{-1}U)=\widetilde{L}\widetilde{U}$$
由此我们就可以从一个分解中找到无数个分解，$D$只需要是行列式不为0的任意对角矩阵即可。为此，我们希望寻找一种具有唯一性的三角分解

定理 (LDU 基本定理) 设$A$ 为$n$ 阶方阵，则 $A$ 可以惟一地分解为
$$A=LDU$$
的充分必要条件是$A$的前$n-1$个顺序主子式$\Delta_k\neq0(k=1,2,\cdotp\cdotp\cdotp,n-1).$其中$L,U$分别是单位下、上三角矩阵，$D$是对角矩阵
$$D=\operatorname{diag}(d_1,d_2,\cdotp\cdotp\cdotp,d_n),$$
$$d_k=\frac{\Delta_k}{\Delta_{k-1}},\quad k=1,2,\cdots,n,\quad\Delta_0=1.$$

有了LDU定理，我们找到三角分解的存在唯一性，据此可以很容易给出Doolittle分解和Crout分解的存在唯一性

推论：设$A$ 是$n$ 阶方阵，则 $A$ 可以惟一地进行杜利特分解的充分必要条件是A 的前$n-1$个顺序主子式
$$\Delta_{k}=\begin{vmatrix}a_{11}&\cdots&a_{1k}\\\vdots&&\vdots\\a_{k1}&\cdots&a_{kk}\end{vmatrix}\neq0,\quad k=1,2,\cdots,n-1,$$
其中$L$ 为单位下三角矩阵，$\tilde{U}$ 是上三角矩阵，即有

$$\mathbf{A}=\begin{bmatrix}1&&&&&\\l_{21}&1&&&&\\l_{31}&l_{32}&\ddots&&&\\\vdots&\vdots&\ddots&1&&\\l_{n1}&l_{n2}&\cdots&l_{n,n-1}&1\end{bmatrix}\begin{pmatrix}u_{11}&u_{12}&\cdots&u_{1n}\\&u_{22}&\cdots&u_{2n}\\&&\ddots&\vdots\\&&&u_{nn}\end{pmatrix},$$
并且若$A$为奇异矩阵，则$u_{nn}=0;$若$A$为非奇异矩阵，则充要条件可换为：$A$的各阶顺序主子式全不为零，即：
$$\Delta_k\neq0,\quad k=1,2,\cdotp\cdotp\cdotp,n.$$

推论 2 $n$阶方阵$A$可惟一地进行克劳特分解
$$A=\tilde{\boldsymbol{L}}\boldsymbol{U}=\begin{bmatrix}l_{11}&&&\\l_{21}&l_{22}&&\\\vdots&\vdots&\ddots&\\l_{n1}&l_{n2}&\cdots&l_{nn}\end{bmatrix}\begin{pmatrix}1&u_{12}&\cdots&u_{1n}\\&1&\cdots&u_{2n}\\&&\ddots&\vdots\\&&&1\end{pmatrix}$$

的充要条件仍为式前推论中$n-1$阶的情况.若$A$为奇异矩阵，则$l_{nm}=0;$若$A$为非奇异矩阵，则充要条件也可换为前述中各阶的情况

#### 常用的三角分解
在实际应用中，如果矩阵$A$的阶数$n$很高 ,那么按消元步骤来得出$A$的三角分解是相当麻烦的.下面我们将分别根据 A 的不对称和对称的情况，介绍两个常用的直接三角分解公式

##### Crout分解
设$A$为$n$阶方阵(但不一定对称),且有分解式
$$A=LU,$$
即
$$\begin{pmatrix}a_{11}&\cdots&a_{1j}&\cdots&a_{1n}\\\vdots&&\vdots&&\vdots\\a_{i1}&\cdots&a_{ij}&\cdots&a_{in}\\\vdots&&\vdots&&\vdots\\a_{n1}&\cdots&a_{nj}&\cdots&a_{nn}\end{pmatrix}=\begin{pmatrix}l_{11}&&&&&\\\vdots&\ddots&&&&\\l_{i1}&\cdots&l_{ii}&&&\\\vdots&&&\ddots&&\\l_{n1}&\cdots&\cdots&\cdots&l_{nn}\end{pmatrix}\begin{pmatrix}1&u_{12}&\cdots&u_{1j}&\cdots&u_{1n}\\&\ddots&&\vdots&&\vdots\\&&1&u_{j-1,j}&\cdots&u_{j-1,n}\\&&&\ddots&&\vdots\\&&&&1&1\end{pmatrix}$$

下面给出矩阵各元素的计算方法

当$i\ge j$  的时候（表示计算下三角位置）
$$l_{ij}=a_{ij}-\sum_{k=1}^{j-1}l_{ik}u_{kj},\quad i=1,\cdots,n,\quad j=1,\cdots,i;$$
当$i< j$  的时候（表示计算下三角位置）
$$u_{ij}=\left(a_{ij}-\sum_{k=1}^{i}l_{ik}u_{kj}\right)/l_{ii},\quad i=1,\cdots,n-1,\quad j=i+1,\cdots,n.$$
我们需要迭代使用这两个公式逐步进行求解

##### Doolittle分解
类似的，我们可以给出Doolittle分解下面的求解式
$$\begin{cases}u_{ij}=a_{ij}-\sum_{k=1}^{i-1}l_{ik}u_{kj},&i=1,\cdots,n,&j=i,\cdots,n,\\\\l_{ij}=\left(a_{ij}-\sum_{k=1}^{j-1}l_{ik}u_{kj}\right)/u_{jj},&i=2,\cdots,n,&j=1,\cdots,i-1.&\end{cases}$$

##### Cholesky分解
如果$A$是对称正定矩阵，则可以使三角分解的计算量大为减少，大约是前述的克劳特分解或杜利特分解工作量的一半.

定理：设$A$为$n$阶对称正定矩阵，则存在一个实的非奇异下三角矩阵$L$,使
$$A=LL^{\mathrm{T}}.$$
如果限定$L$ 的对角元素为正时，这种分解是惟一的.

我们可以轻松的给出求解公式，由于其对称性我们只需要计算一半
$$l_{ij}=\left(a_{ij}-\sum_{k=1}^{j-1}l_{ik}l_{jk}\right)/l_{jj},\quad i\geqslant j.$$
特别的，当$i=j$的时候
$$l_{ii}=\sqrt{a_{ii}-\sum_{k=1}^{i-1}l_{ik}^2}$$
由于需要进行不少的平方根变换，他也被称为平方根分解
### QR（正交三角）分解
由于LU三角分解不能解决一些病态方程组的问题，于此同时有些可逆矩阵不存在LU分解，因此我们需要提出更为优秀的分解方法，这就是QR分解，他对一切可逆矩阵均存在。

#### QR分解的概念
定义：如果实(复)非奇异矩阵$A$能化成正交(西)矩阵$Q$与实(复)非奇异上三角
矩阵$R$的乘积，即
$$A=QR\:,$$
则称其是$A$的QR分解

更为常用的QR分解针对实数而言，因此我们后面主要讨论实矩阵的正交分解，同时给出部分复分解的结论。

定理：任何实的非奇异$n$阶矩阵 A 可以分解成正交矩阵$Q$和上三角矩阵 R 的乘积，且除去相差一个对角线元素之绝对值全等于 1 的对角矩阵因子$D$外，分解式是惟一的.

所谓的相差一个对角矩阵因子$D$ 指的是 $A=QD^{-1}DR$  当规定$R$的对角线元素都是正实数的时候，$D=I$ 此时分解唯一。

最基础的QR分解使用Schmit正交化进行，如下
* $A$的列向量由于$A$是非奇异的，因此一定线性无关，对列向量$\alpha_i$ 进行Schmit正交化有$(\beta_1,\beta_2,\cdots,\beta_n)=(\alpha_1,\alpha_2,\cdots,\alpha_n)B$
* 此时可以给出 $Q=AB$
* $B^{-1}=R$是上三角矩阵 此时有 $A=QB^{-1}=QR$
#### QR分解的计算
正交化求解QR分解比较复杂，因此我们这里介绍一些其他的计算方法，他们在更加复杂的问题中有着更好的效果。

##### Givens方法
Givens方法基于矩阵的初等旋转变换，通过不断的左乘$R$ 消去$A$的非零元素，最后化简到上三角矩阵

定理：任何实非奇异矩阵可通过左连乘初等旋转阵化为上三角阵

这个定理的证明就是寻找QR分解的过程

对实可逆矩阵$A=(a_i$,)左乘以初等旋转阵$R_{ij}$以后，只改变$A$的第$i$行和第$j$ 行元素.设$$A^{\prime}=R_{ij}A$$
则变换的效果为
$$a_{ig}^{\prime}=ca_{ig}+sa_{jg},\quad a_{jg}^{\prime}=-sa_{ig}+ca_{jg},\quad a_{jg}^{\prime}=a_{jg},\quad p\neq i,j;g=1,2,\cdots,n.$$
如果想要 $a_{jg_0}^{\prime}=0$  那么只需要$a_{ig_0}\text{和 }a_{jg_0}$ 之一不为0，并且取
$$s=\frac{a_{jg_0}}{\sqrt{a_{ig_0}^2+a_{jg_0}^2}},\quad c=\frac{a_{ig_0}}{\sqrt{a_{ig_0}^2+a_{jg_0}^2}}$$
此时
$$a_{ig_0}^{\prime}=\sqrt{a_{ig_0}^2+a_{ig_0}^2}>0$$
也就是说，该变换的效果为 $g_0$列 $j$行化0，$g_0$列$i$行变正 其他元素不变

有了这样的变换后，我们就可以通过不断左乘初等旋转矩阵，将矩阵变换为上三角矩阵，也就是
$$\begin{aligned}A^{(n-1)}&=\boldsymbol{R}_{n-1,n}\cdots\boldsymbol{R}_{12}\boldsymbol{A}\\&=\begin{bmatrix}a_{11}^{(1)}&a_{12}^{(1)}&\cdots&a_{1n}^{(1)}\\0&a_{22}^{(2)}&\cdots&a_{2n}^{(2)}\\\vdots&\vdots&\ddots&\vdots\\0&0&\cdots&a_{nn}^{(n-1)}\end{bmatrix}\end{aligned}$$

这实际上就蕴含了一个QR分解
$${R}={A}^{(n-1)}$$
$$Q=(R_{n-1,n}\cdotp\cdotp\cdotp R_{12})^{-1}$$
由于初等旋转矩阵的正交性，所以有QR分解
$$A=QR.$$

Givens方法需要计算 $\frac{n(n-1)}{2}$ 个初等旋转矩阵的积，因此在高维矩阵上并不实用
##### Housholder方法
定理：任何实的$n$阶矩阵$A$可用初等反射矩阵$H=I-2\omega\omega^\mathrm{T}$化为上三角矩阵.

该方法具体证明我们不再讨论，其思想也是通过不断左乘初等反射矩阵化为上三角矩阵，然后用类似于Givens的方法求逆，得到原始矩阵的QR分解，其计算量和矩阵维数线性增加，在处理高维非稀疏矩阵比Givens方法更快。


### 最大秩分解
以上两节主要是介绍了$n$阶方阵的几种分解，从本节开始，将介绍几种常用的长方阵的分解

定义：设$m\times n$矩阵

$$\mathbf{A}=\begin{bmatrix}a_{11}&a_{12}&\cdots&a_{1n}\\a_{21}&a_{22}&\cdots&a_{2n}\\\vdots&\vdots&&\vdots\\a_{m1}&a_{m2}&\cdots&a_{mn}\end{bmatrix},$$

如果当$m\leqslant n$时 ,存在有 rank$\boldsymbol{A}=m;$或者当$m\geqslant n$时 ,存在有 rank$\boldsymbol{A}=n$,则称这两种长方阵为最大秩长方阵(满秩长方阵),前者又称行最大秩矩阵(行满秩矩阵或矮矩阵),后者又称为列最大秩矩阵(列满秩矩阵或高矩阵)

**最大秩长方阵指的是这个长方形矩阵有其拥有的最大的秩**

显然，最大秩长方阵具有如下性质
$$\operatorname{rank}(AA^\mathrm{T})=m,\quad\mathbf{A}=\left(a_{ij}\right)_{m\times n},m\leqslant n$$
或
$$\mathrm{rank}(A^\mathrm{T}A)=n\:,\quad A=(a_{ij})_{m\times n},m\geqslant n.$$

定义：设$A$为$m\times n$且秩为$r>0$的复矩阵，且记为$A\in\mathbb{C}_r^{m\times n}$,如果存在矩阵
$B\in\mathbb{C}_r^{m\times r}$和$C\in\mathbb{C}_r^{r\times n}$,使
$$A=BC\:,$$
则称其的分解为矩阵 A 的最大秩分解(满秩分解).

显然，当$A$是列最大秩(列满秩)或行最大秩(行满秩)矩阵时，$A$的最大秩分解的两个因子中，一个因子是单位矩阵，另一个因子是$A$本身，称这种最大秩分解为平凡分解.

定理：设 $A\in\mathbb{C}_r^{m\times n}$,则一定存在 $\boldsymbol{B}\in\mathbb{C}_r^{m\times r}$ 和$C\in\mathbb{C}_r^{r\times n}$使得
$$A=BC.$$
定理的证明过程就是寻找矩阵$B,C$的过程，思路是先求矩阵$A$的行标准形，然后取标准化后$A$的前$r$列作为$B$，取$A$的前$r$非零行作为矩阵$C$

如果我们把$A$进行列标准化，再取前$r$列作为$B$，前$r$非零行作为矩阵$C$ 我们能得到另一种最大秩分解。这意味着，**最大秩分解不具备唯一性**，但是可能性有限，实际上两种最大秩分解的
$$C^{\mathrm{H}}(CC^{\mathrm{H}})^{-1}(B^{\mathrm{H}}B)^{-1}B^{\mathrm{H}}$$
是相同的，这个形式就是我们在矩阵分析中会研究的穆尔-彭罗斯逆

### 奇异值分解SVD与极分解
矩阵的奇异值分解在矩阵理论中的重要性是不言而喻的，例如古典控制中的频率法，正是由于有了矩阵奇异值分解的帮助而得到了新的发展.这里，只给出奇异值的性质以及矩阵按奇异值的分解。

首先，我们需要给出一些预备知识，关于矩阵的特征值与奇异值

命题：设 $A\in\mathbb{C}^m\times n$,则有
* $A^\mathrm{H}A$ 与$AA^\mathrm{H}$ 的特征值均为非负实数；
* $A^\mathrm{H}A$ 与$AA^\mathrm{H}$的非零特征值相同.

定义：设$A\in\mathbb{C}_r^{m\times n},A^{\mathrm{H}}A$的特征值为
$$\lambda_1\geqslant\lambda_2\geqslant\cdots\lambda_r>\lambda_{r+1}=\lambda_{r+2}=\cdots=\lambda_n=0,$$
则称 $\sigma_i=\sqrt{\lambda_i}\left(i=1,2,\cdots,r\right)$为**矩阵 $A$ 的正奇异值**，简称**奇异值.**
由此定义和命题可知 $,A$ 与$A^\mathrm{H}$ 有相同的奇异值

定义：设$A,B\in\mathbb{C}^{m\times n}$,如果存在$m$阶酉矩阵$U$和$n$阶酉矩阵$V$,使得
$$B=UAV,$$
则称$A$与$B$酉等价或酉相低。

定理:若$A$与 $B$ 酉等价，则$A$与$B$有相同的奇异值.

定理：设$A\in\mathbb{C}_r^{m\times n}$,则存在$m$阶酉矩阵$U$和$n$阶西矩阵$V$,使得
$$U^\text{н}AV=\begin{bmatrix}\Delta&0\\0&0\end{bmatrix}$$
或
$$A=U{\begin{bmatrix}\Delta&0\\0&0\end{bmatrix}}V^{\mathrm{H}}\:,$$
其中$\Delta=\operatorname{diag}(\sigma_{1},\sigma_{2},\cdots,\sigma_{r}),\lambda_{i}$ 为$AA^\mathrm{H}$的非零特征值，且$\sigma_i=\sqrt{\lambda_{i}}\left(i=1,2,\cdots,r\right)$,而$\sigma_i$ 是$A$的全部奇异值. **这就称为矩阵$A$的奇异值分解，本质上是研究$A$与一个长方形对角矩阵的酉等价**

**计算矩阵的奇异值分解很简单，奇异值矩阵只需要直接计算奇异值就可以得到，按顺序排列。矩阵$U,V$只需要计算$A^HA,AA^H$的特征向量构成的列矩阵**

设可逆矩阵$A$的奇异值分解为$A=UDV^\mathrm{H}$,则其逆的奇异值分解为$A^{-1}=VD^{-1}U^{\mathrm{H}}.$因此，若 $A$ 的奇异值为$\sigma_1\geqslant\sigma_2\geqslant\cdotp\cdotp\cdotp\geqslant\sigma_n>0$,则 $A^{-1}$ 的奇异值为 $1/\sigma_n\geqslant1/\sigma_n$ $\geqslant\cdotp\cdotp\cdotp\geqslant1/\sigma_1>0.$
设$A=U_1DV^\mathrm{H}$是$A$的奇异值分解 ,令
$$P=U_1DU_1^H,\quad U=U_1V^H,$$
即可得到矩阵的另一种有趣分解——极分解.

定理：设 $A\in\mathbb{C}^{n\times n}$,则存在酉矩阵 $U$ 和惟一的半正定矩阵 $P$,使得
$$A=PU,$$
上式称为矩阵$A$的极分解.矩阵$P$与$U$分别称为$A$的埃尔米特因子和西因子.

特别的，我们这里简单研究一下特征值和奇异值的一些性质

定理（奇异值与特征值) ：设$\lambda$是$n$阶矩阵$A$的一个特征值，又将$A$的最大奇异值与最小奇异值分别记为$\sigma_\mathrm{max}(\boldsymbol{A})$与$\sigma_\mathrm{min}(\boldsymbol{A})$,则$\sigma_\mathrm{max}(\boldsymbol{A})\geqslant|\lambda|\geqslant\sigma_\mathrm{min}(\boldsymbol{A}).$换言之，矩阵的最大奇异值与最小奇异值是其特征值的模的上下界.

定理（奇异值与矩阵的迹) ：设 $A\in\mathbb{C}^{m\times n}$,则 tr$(A^\mathrm{H}A)=\sum_{i=1}\sigma_i^2.$

定理(奇异值与奇异矩阵) ：矩阵$A$列满秩$\Leftrightarrow A$的奇异值均非 0.特别地，方阵$A$非奇异$\Leftrightarrow A$ 的奇异值均非0。事实上，矩阵非零奇异值的个数和秩相等

### 谱分解
所有的矩阵分解都是为了简化问题，因此本章我们再介绍一种性质优秀的矩阵：可以酉对角化的矩阵
#### 正规矩阵
定义：设$A$是复数域上的方阵，如果有
$$AA^\text{н}=A^\text{н}A,$$
则称 $A$ 为正规矩阵.

如果$A$是实数域上的$n$阶方阵，且有
$$AA^{\mathrm{T}}=A^{\mathrm{T}}A,$$
则称 $A$ 为实正规矩阵.

我们容易验证，对称矩阵、反对称矩阵($A=-A^{\mathrm{T}}$),正交矩阵都是实正规矩阵；而酉矩阵、埃尔米特矩阵、反埃尔米特矩阵(即$A=-A^{\mathrm{H}}$)均属于复正规矩阵.

**定理：设 $A\in\mathbb{C}^{n\times n}$,则 $A$ 酉相似于对角矩阵的充分必要条件是$A$ 为正规矩阵**

事实上，实对称矩阵正交相似与一个对角矩阵，就是本定理在线性代数中的狭义情况，这个定理是研究对角化问题的最终答案。

定理：对于正规矩阵，我们可以轻松的给出下面的推论
* 正规的三角矩阵是对角矩阵
* 正规矩阵有$n$个两两正交的单位特征向量
* 正规矩阵有$n$个不同的特征值
* 正规矩阵不同特征值的特征向量正交
* 对于正规矩阵，其特征值和元素满足$\sum_{i=1}^n\mid\lambda_i\mid^2=\sum_{i,j=1}^n\mid a_{ij}\mid^2$

#### 正规矩阵的谱分解
我们知道正规矩阵酉相似与一个对角矩阵，这就是谱分解希望研究的

设$A$为正规矩阵，因此存在酉矩阵$U$使得$U^\mathrm{H}AU=\operatorname{diag}\left(\lambda_1,\lambda_2,\cdots,\lambda_n\right)$ 也就是说
$$\mathbf{A}=\boldsymbol{U}\mathrm{diag}(\lambda_1,\lambda_2,\cdots,\lambda_n)\boldsymbol{U}^\mathrm{H}.$$
令$\boldsymbol{U}=(\boldsymbol{\alpha}_1,\boldsymbol{\alpha}_2,\cdots,\boldsymbol{\alpha}_n)$ 则有
$$A=(\boldsymbol{\alpha}_1,\boldsymbol{\alpha}_2,\cdots,\boldsymbol{\alpha}_n)\begin{bmatrix}\lambda_1&&&\\&\lambda_2&&\\&&\ddots&\\&&&\lambda_n\end{bmatrix}\begin{bmatrix}\boldsymbol{\alpha}_1^\mathrm{H}\\\\\boldsymbol{\alpha}_2^\mathrm{H}\\\vdots\\\boldsymbol{\alpha}_n^\mathrm{H}\end{bmatrix}$$
$$=\lambda_1\alpha_1\alpha_1^\mathrm{H}+\lambda_2\alpha_2\alpha_2^\mathrm{H}+\cdots+\lambda_n\alpha_n\alpha_n^\mathrm{H}.$$
由于$\lambda_i$是矩阵的特征值，而$\alpha_i$是特征值对应的正交单位特征向量，因此我们称为**正规矩阵$A$的谱分解或者特征值分解**

把相同特征值的项进行整合化简可以得到
$$A=\lambda_1P_1+\lambda_2P_2+\cdots+\lambda_sP_s$$

#### 单纯矩阵的谱分解
我们已知道$,n$ 阶方阵当代数重复度与几何重复度相等时，称之为单纯矩阵，这样的矩阵可对角化，但不一定可以西对角化(即不一定是正规矩阵).

不过，单纯矩阵也可以类似于正规矩阵定义$A$的谱分解.不妨设$\lambda_1,\lambda_2,\cdotp\cdotp\cdotp,\lambda_n$是$A$的$n$个特征值；$x_1,x_2,\cdotp\cdotp\cdotp,x_n$是$A$的$n$ 个线性无关的特征向量，且有
$$Ax_i=\lambda_ix_i,\quad i=1,2,\cdotp\cdotp\cdotp,n$$
令
$$P=(x_1,x_2,\cdots,x_n),$$
$$\boldsymbol{\Lambda}=\begin{bmatrix}\lambda_1\\&\lambda_2\\&&\ddots\\&&&\lambda_n\end{bmatrix}$$
则
$$A=P\Lambda P^{-1}.$$
两边转置有
$$A^{\mathrm{T}}=(P^{\mathrm{T}})^{-1}{\Lambda}{P}^{\mathrm{T}}.$$
这表明$A^\mathrm{T}$也与对角矩阵相似.因此，设$y_1,y_{2},\cdotp\cdotp\cdotp,y_{n}$是$A^\mathrm{T}$的$n$个线性无关的特征向量，即
$$\mathbf{A}^\mathrm{T}\mathbf{y}_i=\lambda_i\mathbf{y}_i\:,\quad i=1,2,\cdots,n,$$
把上式两端取转置得
$$\mathbf{y}_i^\mathrm{T}\mathbf{A}=\lambda_i\mathbf{y}_i^\mathrm{T}\:,\quad i=1,2,\cdotp\cdotp\cdotp,n,$$
据此，我们称$y_i^\mathrm{T}$是 $A$ 的左特征向量，称$x_i$是$A$的右特征向量

由此
$$(\mathbf{y}_1,\mathbf{y}_2,\cdots,\mathbf{y}_n)=(\mathbf{P}^\mathrm{T})^{-1}=(\mathbf{P}^{-1})^\mathrm{T},$$
转置得
$$\mathbf{P}^{-1}=\begin{bmatrix}\mathbf{y}_1^\mathrm{T}\\\vdots\\\mathbf{y}_n^\mathrm{T}\end{bmatrix}$$
代入$PP^{-1}=P^{-1}P=I$得
$$(\boldsymbol{x}_1,\boldsymbol{x}_2,\cdots,\boldsymbol{x}_n)\begin{bmatrix}\boldsymbol{y}_1^\mathrm{T}\\\vdots\\\boldsymbol{y}_n^\mathrm{T}\end{bmatrix}=\begin{bmatrix}\boldsymbol{y}_1^\mathrm{T}\\\vdots\\\boldsymbol{y}_n^\mathrm{T}\end{bmatrix}(\boldsymbol{x}_1,\boldsymbol{x}_2,\cdots,\boldsymbol{x}_n)=\boldsymbol{I},$$
此即
$$x_1y_1^\mathrm{T}+x_2y_2^\mathrm{T}+\cdots+x_ny_n^\mathrm{T}=I.$$
比较两端即有
$$y_i^\mathrm{T}x_j=\delta_{ij}\:,\quad i,j=1,2,\cdotp\cdotp\cdotp,n,$$

综上我们可以得到
$$A=(x_{1},x_{2},\cdots,x_{n})\begin{bmatrix}\lambda_{1}\\&\ddots\\&&\lambda_{n}\end{bmatrix}\begin{bmatrix}\mathbf{y}_{1}^{\mathrm{T}}\\\vdots\\\mathbf{y}_{n}^{\mathrm{T}}\end{bmatrix}\\=\lambda_{1}\boldsymbol{x}_{1}\boldsymbol{y}_{1}^{\mathrm{T}}+\lambda_{2}\boldsymbol{x}_{2}\boldsymbol{y}_{2}^{\mathrm{T}}+\cdots+\lambda_{n}\boldsymbol{x}_{n}\boldsymbol{y}_{n}^{\mathrm{T}}.$$
令
$$G_i=x_iy_i^\mathrm{T}$$
即得到
$$A=\sum_{i=1}^n\lambda_i\mathbf{G}_i$$
这就称为单纯矩阵$A$的谱分解，分解为$n$个$G_i$的和，线性组合的系数是$A$的特征值


## 矩阵微积分
在线性代数中，只讨论矩阵的加(减)法、乘法和求逆为核心的代数运算，而完全没有涉及类似于数学分析中的极限、级数、微积分等运算，可是，在研究运筹学以及线性系统的可控制等方面的问题时，这些运算又是十分必要的

一类经典的的数学模型是：相当于求以矩阵$U$ 为自变量的函数$$J( \boldsymbol{U}) = \| \boldsymbol{U}\boldsymbol{\alpha }- \boldsymbol{\beta }\|$$其中$\boldsymbol{U}\in\mathbb{R}^m\times n,\boldsymbol{\alpha}\in\mathbb{R}^n,\boldsymbol{\beta}\in\mathbb{R}^m$ 在约束条件$U^\mathrm{T}U=I$ 或$UU^\mathrm{T}=I$ 下的最小值点(矩阵),解决此类优化问题的一个可行办法是求矩阵函数 $J(U)$关于未知矩阵$U$ 的导数，这就需要研究矩阵的微积分。

范数的定义给了我们研究距离的基础，也就赋范线性空间导出度量空间，因此本章我们研究矩阵的微积分，他是最为初等的分析学与代数学的碰撞。

### 向量序列与矩阵序列的极限
在研究微积分之前，我们自然还是先研究极限的问题

#### 向量序列的极限
定义(按范数收敛)：设 $x^{(k)},x\in\mathbb{C}^{n}(k=1,2,\cdots)$,若
$$\parallel x^{(k)}-x\parallel\to0,\quad k\to+\infty,$$
则称向量序列$\langle x^(k)\rangle$收敛于向量 $x$,或说向量 $x$ 是向量序列$\langle x^(k)\rangle$当$k\to+\infty$时的极限，可记为
$$\lim_{k\to+\infty}x^{(k)}=x$$
或
$$x^{(k)}\to x,\quad k\to+\infty.$$

根据向量范数的等价性我们可以知道：**在某一向量范数下收敛，在其他范数下也收敛**

定理：在Banach空间中，我们知道柯西收敛准则成立，因此此时**向量收敛和其分量分别收敛是等价的** 因此数学分析中该命题成立，这里我们推广到Banach空间


#### 矩阵序列的极限
矩阵可以看作一个高维向量，因此我们可以类似的给出矩阵极限的定义，这里我们首先从分量的收敛给出定义

定义：设有矩阵序列$\left\{A^{(k)}\right\}$,其中 $A^{(k)}=(a_{ij})^{(k)})\in\mathbb{C}^{n\times n}$,且当 $k\to+\infty$时$,a_{ij}^{(k)}\to a_{ij}$,则称$\left\{A^{(k)}\right\}$收敛，并把矩阵 $\boldsymbol A=(a_{ij})$叫做$\left\{\boldsymbol A^{(k)}\right\}$的极限，或称$\left\{\boldsymbol A^{(k)}\right\}$收敛于$A$,记为
$$\lim_{k\to-\infty}A^{(k)}=A\quad\text{或}\quad A^{(k)}\to A.$$
不收敛的矩阵序列称为发散的.

定理：前述的矩阵收敛定义与使用矩阵范数定义是等价的，也就是等价于
$$\parallel A^{(k)}-A\parallel\to0,\quad k\to+\infty$$

定理：和向量收敛同样的，矩阵的各个范数定义的收敛也等价

对于矩阵的极限运算，我们可以给出下面的性质
* 线性性：设$\operatorname*{lim}_{k\to+\infty}A^{(k)}=A,\operatorname*{lim}_{k\to+\infty}B^{(k)}=B$ 则$\lim_{k\to+\infty}(a\boldsymbol{A}^{(k)}+b\boldsymbol{B}^{(k)})=a\boldsymbol{A}+b\boldsymbol{B},\quad a,b\in\mathbb{C}$
* 乘性：设$\operatorname*{lim}_{k\to+\infty}A^{(k)}=A,\operatorname*{lim}_{k\to+\infty}B^{(k)}=B$   则 $\lim_{k\to+\infty}A^{(k)}B^{(k)}=AB$
* 逆性：设$\operatorname*{lim}_{k\to+\infty}A^{(k)}=A$ 且 $A^{k}$均可逆，则$\{(A^{(k)})^{-1}\}$ 也收敛，并且$\lim_{k\to+\infty}(A^{(k)})^{-1}=A^{-1}$

定理：设有矩阵序列$\left\{\boldsymbol{A}^{(k)}\right\}:\boldsymbol{A},\boldsymbol{A}^2,\cdotp\cdotp\cdotp A^k,\cdotp\cdotp\cdotp$,则$\lim\boldsymbol{A}^k=\boldsymbol{0}$ 的充分必要条件是矩阵$A$的所有特征值的模都小于 1,即$A$的谱半径小于 1.
$$\rho(A)<1.$$

定理 ：若对于矩阵 $A$ 的某一范数有$\parallel A\parallel<1$,则
$$\lim_{k\to+\infty}A^k=\mathbf{0}.$$
### 矩阵级数与矩阵函数
#### 矩阵级数的定义与收敛
为了更好的研究矩阵函数，我们首先需要研究矩阵级数理论，他和数项级数理论的定义与性质非常相似。

定义：设有矩阵序列
$$A^{(1)},A^{(2)},\cdots,A^{(k)},\cdots,$$
其中$A^{(k)}=(a_{ij}^{(k)})\in\mathbb{C}^{n\times n}$,称无穷和
$$A^{(0)}+A^{(1)}+A^{(2)}+\cdots+A^{(k)}+\cdots $$
为矩阵级数，记为$\sum_{k=0}^{\infty}\boldsymbol{A}^{(k)},\boldsymbol{A}^{(k)}$称为矩阵级数的一般项，即有
$$\sum_{k=0}^{+\infty}A^{(k)}=A^{(0)}+A^{(1)}+A^{(2)}+\cdots+A^{(k)}+\cdots.$$

定义：级数前$k+1$项的和
$$S^{(k)}=A^{(0)}+A^{(1)}+A^{(2)}+\cdots+A^{(k)}$$
称为级数的部分和，如果矩阵序列${S}^{(k)}$收 敛 , 且有极限 $S$,即有
$$\lim_{k\to+\infty}\mathbf{S}^{(k)}=S,$$
则称矩阵级数收敛，$S$ 称为级数的和，记作
$$S=\sum_{k=0}^{+\infty}A^{(k)}.$$
不收敛的矩阵级数称为是发散的

根据矩阵收敛的性质我们容易知道：**矩阵级数收敛的充要条件是对应的$n^2$个数项级数收敛**

从定义我们容易给出下面性质
* $\text{若 }\sum_{k=0}^{+\infty}A^{(k)}\text{ 收敛,则}\lim_{k\to+\infty}A^{(k)}=\mathbf{0};$
* $\text{若 }\sum_{k=0}^{+\infty}\mathbf{A}^{(k)}=\mathbf{S},\sum_{k=0}^{+\infty}\mathbf{B}^{(k)}=\mathbf{S}^{\prime},\text{则}$ $\sum_{k=0}^{+\infty}(A^{(k)}\pm B^{(k)})=S\pm S^{^{\prime}};$
* $\text{若 }\sum_{k=0}^{+\infty}A^{(k)}=S,\text{则}\sum_{k=0}^{+\infty}\mu A^{(k)}=\mu S,\mu\in\mathbb{C}.$

定义：设矩阵级数 $\sum_k=0A^{(k)}=A^{(0)}+A^{(1)}+A^{(2)}+\cdots+A^{(k)}+\cdots$,其中 $A^{(k)}=(a_{ij}^{(k)})\in\mathbb{C}^{n\times n}.$如果 $n^2$ 个数项级数
$$a_{ij}^{(0)}+a_{ij}^{(1)}+a_{ij}^{(2)}+\cdots+a_{ij}^{(k)}+\cdots,\quad i,j=1,2,\cdots,n$$
都绝对收敛，则称矩阵级数绝对收敛

定理：矩阵级数 $\sum_{k=0}^{+\infty}A^{(k)}$ 绝对收敛的充分必要条件是$\sum_{k=0}^{+\infty}\parallel A^{(k)}\parallel=\parallel A^0\parallel+\parallel A^{(1)}\parallel+\parallel A^{(2)}\parallel+\cdotp\cdotp\cdotp+\parallel A^{(k)}\parallel+\cdotp\cdotp\cdotp$ 收敛其中$\|\boldsymbol A^(k)\|$ 为$A^(k)$ 的任何一种范数

定理：设两个矩阵级数
$$A^{(1)}+A^{(2)}+\cdots+A^{(k)}+\cdots,\quad A^{(k)}\in\mathbb{C}^{n\times n},$$
$$B^{(1)}+B^{(2)}+\cdotp\cdotp\cdotp+B^{(k)}+\cdotp\cdotp\cdotp,\quad B^{(k)}\in\mathbb{C}^{n\times n}$$
都绝对收敛，其和分别为 $A,B$,则将它们按项相乘后作成的矩阵级数
$$A^{(1)}B^{(1)}+(A^{(1)}B^{(2)}+A^{(2)}B^{(1)})+\cdots+$$
$$(A^{(1)}B^{(k)}+A^{(2)}B^{(k-1)}+\cdots+A^{(k)}B^{(1)})+\cdots $$
绝对收敛，且具有和$AB.$

#### 矩阵级数的性质
根据矩阵级数的定义以及一些分析学中的知识，我们可以展开对矩阵级数性质的研究

定理：设矩阵级数$\sum_{k=0}^{+\infty}A^{(k)}$绝对收敛 则
* $\text{级数 }\sum_{k=0}^{+\infty}A^{(k)}\text{ 收敛;}$
* $\text{级数 }\sum_{k=0}^{+\infty}A^{(k)}\text{ 在任意改变各项的次序后仍然收敛,且其和不变}.$

定理：设 $P,Q$为$n$阶非奇异矩阵，若级数$\sum_{t=0}A^{(k)}$收敛(或绝对收敛),则矩阵级数$\sum_{k=0}^{+\infty}\boldsymbol{PA}^{(k)}Q$也收敛(或绝对收敛).

定义：形如
$$c_0I+c_1A+c_2A^2+\cdots+c_kA^k+\cdots $$
的矩阵级数称为矩阵幂级数，其中$c_i\in\mathbb{C},A\in\mathbb{C}^n\times n.$

定理：若正项级数$|c_0|\|\boldsymbol{I}\|+\sum_{k=1}|c_k|\|\boldsymbol{A}\|^k$收敛，则矩阵幂级数 $c_0\boldsymbol{I}+$
$c_{1}\boldsymbol{A}+c_{2}\boldsymbol{A}^{2}+\cdots+c_{k}\boldsymbol{A}^{k}+\cdots$绝对收敛，其中$\parallel\boldsymbol{A}\parallel$为矩阵 $A$ 的某种范数

推论：若矩阵 $A$ 的某一种范数$\parallel A\parallel$在幂级数
$$\sum_{k=0}^{+\infty}c_0z^k=c_0+c_1z+c_2z^2+\cdots+c_kz^k+\cdots $$
的收敛圆内，则矩阵幂级数 $\sum_{k=0}^{+\infty}c_k\mathbf{A}^k$ 绝对收敛.

定理：设$A\in\mathbb{C}^{n\times n}$,如果$A$的谱半径$\rho(\boldsymbol{A})$的值在纯量$z$的幂级数$\sum_{k=0}c_kz^k$的收敛圆内，那么矩阵幂级数$\sum_{k=0}^{+\infty}c_kA^k$ 绝对收敛；如果 $A$ 的特征值中有一个在幂级数$\sum_{k=0}^{+\infty}c_kz^k$ 的收敛圆外，则矩阵幂级数$\sum_{k=0}^{+\infty}c_kA^k$发散.

定理：矩阵幂级数$I+A+A^2+\cdots+A^k+\cdots$ 绝对收敛的充要条件为$A$的谱半径$(A)<1$ 且该级数的和为$\left(I-A\right)^{-1}$

#### 矩阵函数的定义
我们知道，复变量（纯量）$z$ 的级数
$$\begin{aligned}&\mathrm{e}^{z}=1+\frac{z}{1!}+\frac{z^{2}}{2!}+\frac{z^{3}}{3!}+\cdots+\frac{z^{k}}{k!}+\cdots,\\&\mathrm{sin}z=z-\frac{z^{3}}{3!}+\frac{z^{5}}{5!}-\cdots+(-1)^{k}\frac{z^{2k+1}}{(2k+1)!}+\cdots,\\&\cos z=1-\frac{z^{2}}{2!}+\frac{z^{4}}{4!}-\cdots+(-1)^{k}\frac{z^{2k}}{(2k)!}+\cdots\end{aligned}$$
在整个复平面都是收敛的

因此对于任意矩阵$A$ 矩阵幂级数
$$\begin{aligned}&I+\frac{A}{1!}+\frac{A^{2}}{2!}+\frac{A^{3}}{3!}+\cdots+\frac{A^{k}}{k!}+\cdots,\\&A-\frac{A^{3}}{3!}+\frac{A^{5}}{5!}-\cdots+(-1)^{k}\frac{A^{2k+1}}{(2k+1)!}+\cdots,\\&I-\frac{A^2}{2!}+\frac{A^4}{4!}-\cdots+(-1)^k\frac{A^{2k}}{(2k)!}+\cdots\end{aligned}$$
都是绝对收敛的，我们记为 $e^A,sinA,cosA$

定义：$\text{设实函数 }y=f(x),A,B\in\mathbb{C}^{n\times n},\text{称 }B=f(A)\text{为矩阵 }A\text{ 的函数}.$

对于矩阵函数，我们可以自然的给出下面的推论
* 如果有 $AB=BA$  则 $\mathrm{e}^A\bullet\mathrm{e}^B=\mathrm{e}^B\bullet\mathrm{e}^A=\mathrm{e}^{A+B}.$
* $\text{对任意矩阵 }A\in\mathbb{C}^{n\times n},\mathrm{e}^A\text{ 总是可逆的(非奇异的)且(e}^A)^{-1}=\mathrm{e}^{-A}.$
* $(\mathrm{~e}^A)^m=\mathrm{e}^{mA}(m\text{为整数)}.$

#### 矩阵函数值的求法
直接使用定义计算矩阵函数的值是非常困难的，我们需要计算非常复杂的矩阵乘法，这里我们使用例子来介绍如果简化这种运算

已知4阶矩阵$A$的特征值分别为 $\pi,-\pi,0,0$ 求 $e^A,sinA,cosA$

因为 $A$ 的特征方程为
$$\det(\lambda\boldsymbol{I}-\boldsymbol{A})=(\lambda-\pi)(\lambda+\pi)\lambda^2=\lambda^4-\pi^2\lambda^2=0$$
根据哈密顿-凯莱定理 有
$$A^4=\pi^2A^2$$
因此，所有大于四次的项都可以使用该定理降低阶数，整个级数内最高阶的矩阵幂就是3阶，我们可以容易计算，如下
$$\begin{aligned}sinA=&\mathbf{A}-\frac{1}{3!}\mathbf{A}^{3}+\frac{1}{5!}\mathbf{A}^{5}-\frac{1}{7!}\mathbf{A}^{7}+\frac{1}{9!}\mathbf{A}^{9}-\cdots\\=&\mathbf{A}-\frac{1}{3!}\mathbf{A}^{3}+\frac{1}{5!}\pi^{2}\mathbf{A}^{3}-\frac{1}{7!}\pi^{4}\mathbf{A}^{3}+\frac{1}{9!}\pi^{6}\mathbf{A}^{3}-\cdots\\=&\mathbf{A}+\left(-\frac{1}{3!}+\frac{1}{5!}\pi^{2}-\frac{1}{7!}\pi^{4}+\frac{1}{9!}\pi^{6}-\cdots\right)\mathbf{A}^{3}\\=&\mathbf{A}+\frac{\sin\pi-\pi}{\pi^{3}}\mathbf{A}^{3}=\mathbf{A}-\pi^{-2}\mathbf{A}^{3},\end{aligned}$$
其余问题均可类似求解，**核心是使用特征多项式带来的阶数约简** 当然还有部分矩阵乘法需要计算。

另一种方法是利用一个特殊的定理，思路如下

假定矩阵$A$与一个对角矩阵相似，则可以找到
$$C^{-1}AC=\operatorname{diag}(\lambda_1,\lambda_2,\cdots,\lambda_n)$$
代入公式
$$\begin{aligned}&e^A=C\cdot\mathrm{diag}(\mathrm{e}^{\lambda_1},\mathrm{e}^{\lambda_2},\cdots,\mathrm{e}^{\lambda_n})\cdot C^{-1},\\&sinA=C\cdot\mathrm{diag}(\sin\lambda_1,\sin\lambda_2,\cdots,\sin\lambda_n)\cdot C^{-1},\\&cosA=C\cdot\mathrm{diag}(\cos\lambda_1,\cos\lambda_2,\cdots,\cos\lambda_n)\cdot C^{-1}.\end{aligned}$$
至于更加复杂的Jordan型与非三种典型函数，这里不讨论

### 矩阵微分与积分
#### 函数矩阵对实变量的导数
定义：若矩阵$A=(a_{ij})$的诸元素$a_{ij}$均是变量$t$的函数，即
$$\mathbf{A}(t)=\begin{bmatrix}a_{11}(t)&a_{12}(t)&\cdots&a_{1n}(t)\\\\a_{21}(t)&a_{22}(t)&\cdots&a_{2n}(t)\\\vdots&\vdots&&\vdots\\\\a_{m1}(t)&a_{m2}(t)&\cdots&a_{mn}(t)\end{bmatrix},$$
则称$\boldsymbol{A}(t)$为**函数矩阵**.推而广之，变量$t$还可以是向量，也可以是矩阵

定义：如果所有的元素 $a_{ij}(t)$在 $t=t_0$ 时 ,存在极限，即有$\operatorname* { lim} _{t\to t_0}a_{ij}\left ( t\right ) = a_{ij}$, $a_{ij}$为一常数，则称**矩阵 $A(t)$有极限**，且极限值为 $A$(常量矩阵),即
$$\lim\limits_{t\to t_0}A(t)=A=\begin{bmatrix}a_{11}&a_{12}&\cdots&a_{1n}\\\\a_{21}&a_{22}&\cdots&a_{2n}\\\vdots&\vdots&&\vdots\\\\a_{m1}&a_{m2}&\cdots&a_{mn}\end{bmatrix}$$

一个函数矩阵的极限，具有通常函数极限的相似性质.例如，当 $t\to t_0$ 时，函数矩阵$\mathbf{A}(t)$和 $B(t)$有极限 $A$ 和 $B$,则有
$$\begin{aligned}&\operatorname*{lim}_{t\to t_{0}}[\boldsymbol{A}(t)+\boldsymbol{B}(t)]=\boldsymbol{A}+\boldsymbol{B},\\&\operatorname*{lim}_{t\to t_{0}}[\boldsymbol{A}(t)\boldsymbol{B}(t)]=\boldsymbol{A}\boldsymbol{B}\:,\\&\operatorname*{lim}_{t\to t_{0}}k\boldsymbol{A}(t)\:=\:k\boldsymbol{A}\:,\end{aligned}$$
其中$A,B$均为常量矩阵$,k$为常数.

定义：如果所有函数$a_{ij}(t)$在某一点或某一区间上是连续的，则称此**函数矩阵在此点或在此区间上也是连续的**.

对于多变量的函数矩阵，也可以有与上述类似的规定，这里就不一一重复了.

定义：设 $\boldsymbol{A}(t)=\left(a_{ij}\left(t\right)\right)_{m\times n}$,若 $a_{ij}\left(t\right)\left(i=1,2,\cdots,m;j=1,2,\cdots,n\right)$在 $t=t_0$ 处(或$[a,b]$上)可导，则称 $\boldsymbol A(t)$在点 $t=t_0$ 处(或在$[a,b]$上)可导，且记为
$$\mathbf{A}^{'}(t_0)=\frac{\mathrm{d}\mathbf{A}(t)}{\mathrm{d}t}\mid_{t=t_0}=\lim_{\Delta t\to0}\frac{\mathbf{A}(t_0+\Delta t)-\mathbf{A}(t_0)}{\Delta t}$$
$$=\begin{bmatrix}a'_{11}(t_0)&a'_{12}(t_0)&\cdots&a'_{1n}(t_0)\\a'_{21}(t_0)&a'_{22}(t_0)&\cdots&a'_{2n}(t_0)\\\vdots&\vdots&&\vdots\\a'_{m1}(t_0)&a'_{m2}(t_0)&\cdots&a'_{mn}(t_0)\end{bmatrix}_{m\times n}.$$

下面的性质都是从数学分析中平移得到，不难证明
* $\mathbf{A}(t)\text{为常数矩阵的充分必要条件是 }\mathbf{A}^{\prime}(t)=\mathbf{0}$
* $\text{设 }\mathbf{A}(t)=\left(a_{ij}\left(t\right)\right)_{m\times n}\text{与B}\left(t\right)=\left(b_{ij}\left(t\right)\right)_{m\times n}\text{可导,则}$  $$\frac{\mathrm{d}}{\mathrm{d}t}(A(t)\pm B(t))=A^{\prime}(t)\pm B^{\prime}(t)$$
* $\text{若 }k(t)\text{是可导的实函数},A(t)\text{可导},\text{则}$  $$\frac{\mathrm{d}}{\mathrm{d}t}(k(t)\mathbf{A}(t))=k^{\prime}(t)\mathbf{A}(t)+k(t)\mathbf{A}^{\prime}(t)$$
* $\text{设 }A(t)\text{与 }B(t)\text{都可导,则}$ $\frac{\mathrm{d}}{\mathrm{d}t}(\boldsymbol{A}(t)\boldsymbol{B}(t))=\boldsymbol{A}^{\prime}(t)\boldsymbol{B}(t)+\boldsymbol{A}(t)\boldsymbol{B}^{\prime}(t)$
* $\text{若 }\mathbf{A}(t)\text{与 }\mathbf{A}^{-1}(t)\text{都有导数,则}$ $\frac{\mathrm{d}\boldsymbol{A}^{-1}(t)}{\mathrm{d}t}=-\boldsymbol{A}^{-1}(t)\boldsymbol{A}^{\prime}(t)\boldsymbol{A}^{-1}(t)$
* 设函数矩阵 $\boldsymbol A(t)$是 $t$ 的函数，而 $t=f(x)$是 $x$ 的实值函数.且 $\boldsymbol A(t)$与 $f(x)$均可导，则有$$\frac{\mathrm{d}\mathbf{A}(t)}{\mathrm{d}x}=\frac{\mathrm{d}\mathbf{A}(t)}{\mathrm{d}t}f'(x)=f'(x)\:\frac{\mathrm{d}\mathbf{A}(t)}{\mathrm{d}t}.$$
函数矩阵的导数本身也是一个函数矩阵，还可以再进行导数运算，故可以定义函数矩阵对实变量的高阶导数：
$$\frac{\mathrm{d}^k\mathbf{A}\left(t\right)}{\mathrm{d}t^k}=\frac{\mathrm{d}}{\mathrm{d}t}\Big(\frac{\mathrm{d}^{k-1}\mathbf{A}\left(t\right)}{\mathrm{d}t^{k-1}}\Big),\quad k=1,2,\cdots,n.$$

我们这里给出一个简单的性质但不继续推广，他是分析学中求导公式的矩阵化体现，实际上在矩阵函数的导数问题中，降维成普通函数是更容易地。对于任何常量方阵$A$ 有
* $\frac{\mathrm{d}}{\mathrm{d}t}\mathrm{e}^{\mathbf{A}t}=\mathbf{A}\mathrm{e}^{\mathbf{A}t}=\mathrm{e}^{\mathbf{A}t}\mathbf{A}$
* $\frac{\mathrm{d}}{\mathrm{d}t}\mathrm{cos}\boldsymbol{A}t=-\boldsymbol{A}(\sin\boldsymbol{A}t)=-(\sin\boldsymbol{A}t)\boldsymbol{A}$
* $\frac{\mathrm{d}}{\mathrm{d}t}\mathrm{sin}\boldsymbol{A}t=\boldsymbol{A}(\mathrm{cos}\boldsymbol{A}t)=(\mathrm{cos}\boldsymbol{A}t)\boldsymbol{A}.$

#### 矩阵的标量函数对矩阵的导数
我们首先推广数学分析4 多元的微分与积分理论中研究得 多元数量函数对向量的导数的概念，给出矩阵数量函数对矩阵的导数的定义

定义：设$A\in\mathbb{R}^{m\times n},f(A)$为矩阵$A$的数量函数，即看成是$m\times n$元函数，则规定数量函数$f(\mathbf{A})$对于矩阵$\mathbf{A}$的导数为
$$\frac{\mathrm{d}f}{\mathrm{d}A}=\left(\frac{\partial f}{\partial a_{ij}}\right)_{m\times n}=\begin{bmatrix}\frac{\partial f}{\partial a_{11}}&\cdots&\frac{\partial f}{\partial a_{1n}}\\\vdots&&\vdots\\\frac{\partial f}{\partial a_{m1}}&\cdots&\frac{\partial f}{\partial a_{mn}}\end{bmatrix}.$$
**我们这里研究的是矩阵的数量函数$f(A)$，他不是一个矩阵函数，而是一个多元的数量函数$f(A)$，借助下面的例子就可以理解了**

设 $\mathbf{X}=\begin{bmatrix}a&b&c\\\\d&e&f\end{bmatrix}$  $F(X)=a^2+b^2+c^2+d^2-2e+15f$  则有
$$\frac{\mathrm{d}F}{\mathrm{d}\boldsymbol{X}}=\begin{bmatrix}\frac{\partial F}{\partial a}&\frac{\partial F}{\partial b}&\frac{\partial F}{\partial c}\\\frac{\partial F}{\partial d}&\frac{\partial F}{\partial e}&\frac{\partial F}{\partial f}\end{bmatrix}=\begin{bmatrix}2a&&2b&&2c\\2d&&-2&&15\end{bmatrix}$$
**无论是如何计算的，但是$f(A)$一定是一个数量函数，否则无法从数分中自然的得到此推论**

定义：设矩阵$F$是以$A\in\mathbb{C}^{m\times n}$为自变量的$p\times q$矩阵，即

$$\boldsymbol{F}(\boldsymbol{A})=\begin{vmatrix}f_{11}\left(\boldsymbol{A}\right)&f_{12}\left(\boldsymbol{A}\right)&\cdots&f_{1q}\left(\boldsymbol{A}\right)\\f_{21}\left(\boldsymbol{A}\right)&f_{22}\left(\boldsymbol{A}\right)&\cdots&f_{2q}\left(\boldsymbol{A}\right)\\\vdots&\vdots&&\vdots\\f_{p1}\left(\boldsymbol{A}\right)&f_{p2}\left(\boldsymbol{A}\right)&\cdots&f_{pq}\left(\boldsymbol{A}\right)\end{vmatrix}_{p\times q},$$
其元素 $f_k(\boldsymbol{A})$是以矩阵 $\boldsymbol{A}=(a_{ij})_{m\times n}$的元素为自变量的 $mn$ 元函数，则规定矩阵 $F(\boldsymbol{A})$对于矩阵$A$的导数为
$$\frac{\mathrm{d}\boldsymbol{F}}{\mathrm{d}\boldsymbol{A}}=\:\Big(\frac{\partial\boldsymbol{F}}{\partial\:a_{ij}}\Big)_{pm\times qn}\:=\:\begin{bmatrix}\frac{\partial\boldsymbol{F}}{\partial a_{11}}&\frac{\partial\boldsymbol{F}}{\partial\:a_{12}}&\cdots&\frac{\partial\boldsymbol{F}}{\partial\:a_{1n}}\\\\\frac{\partial\boldsymbol{F}}{\partial a_{21}}&\frac{\partial\boldsymbol{F}}{\partial\:a_{22}}&\cdots&\frac{\partial\boldsymbol{F}}{\partial\:a_{2n}}\\\vdots&\vdots&&\vdots\\\\\frac{\partial\boldsymbol{F}}{\partial a_{m1}}&\frac{\partial\boldsymbol{F}}{\partial a_{m2}}&\cdots&\frac{\partial\boldsymbol{F}}{\partial\:a_{mn}}\end{bmatrix},$$

其中
$$\frac{\partial\boldsymbol{F}}{\partial a_{ij}}=\begin{bmatrix}\frac{\partial f_{11}}{\partial a_{ij}}&\frac{\partial f_{12}}{\partial a_{ij}}&\cdots&\frac{\partial f_{1q}}{\partial a_{ij}}\\\frac{\partial f_{21}}{\partial a_{ij}}&\frac{\partial f_{22}}{\partial a_{ij}}&\cdots&\frac{\partial f_{2q}}{\partial a_{ij}}\\\vdots&\vdots&\vdots\\\frac{\partial f_{p1}}{\partial a_{ij}}&\frac{\partial f_{p2}}{\partial a_{ij}}&\cdots&\frac{\partial f_{pq}}{\partial a_{ij}}&\end{bmatrix},\begin{aligned}i&=1,2,\cdots,m,\\j&=1,2,\cdots,n.\end{aligned}$$
这个定义非常容易理解，**虽然多层套娃但是同时囊括了前面研究的各种情况**

#### 矩阵的全微分
定义：设矩阵$F=(f_{ij})_{m\times n}$,则规定矩阵$F$的全微分为
$$\mathrm{d}\boldsymbol{F}=(\mathrm{d}f_{ij})_{m\times n}.$$

矩阵的全微分不涉及对矩阵求导的问题，计算起来非常的自然

矩阵的全微分有下面的运算性质
* $\operatorname{d}(\boldsymbol{F}\pm\boldsymbol{G})=\operatorname{d}\boldsymbol{F}\pm\operatorname{d}\boldsymbol{G};$
* $\operatorname{d}(k\boldsymbol{F})=k\operatorname{d}\boldsymbol{F};$
* $\text{当 }A\text{ 是常量矩阵时 },\mathrm{d}A=0$
* $\operatorname{d}(\boldsymbol{X}^{\mathrm{T}})=(\operatorname{d}\boldsymbol{X})^{\mathrm{T}};$
* $\operatorname{d}(\operatorname{tr}\boldsymbol{X})=\operatorname{tr}(\operatorname{d}\boldsymbol{X})$

定理:设 $x=(x_{1},x_{2},\cdots,x_{n})^{\mathrm{T}}$,矩阵 $F=(f_{ij})_{s\times m}$,其中 $f_{ij}$ 都是$x_i$ 的实函数，那么对于矩阵函数的全微分有
$$\mathrm{d}\boldsymbol{F}=\sum_{i=1}^n\frac{\partial\boldsymbol{F}}{\partial x_i}\mathrm{d}x_i.$$
对于矩阵的全微分，我们可以进一步给出下面的性质
* 设 $A=BC$  则 $dA=(dB)C=BdC$
* 设 $A=A_1A_{2}...A_n$ 则 $\mathrm{d} \boldsymbol{A}=\left(\mathrm{d} \boldsymbol{A}_{1}\right) \boldsymbol{A}_{2} \cdots \boldsymbol{A}_{r}+\boldsymbol{A}_{1}\left(\mathrm{~d} \boldsymbol{A}_{2}\right) \boldsymbol{A}_{3} \cdots \boldsymbol{A}_{r}+\boldsymbol{A}_{1} \cdots \boldsymbol{A}_{r-1}\left(\mathrm{~d} \boldsymbol{A}_{r}\right) .$
* $d(\alpha^Tx)=\alpha^Tdx=(dx)^T\alpha$
* $d(Ax)=Adx$
* $d(xA^Tx)=x^T(A^T+A)dx$

#### 矩阵的积分
定义：设函数矩阵
$$\boldsymbol{A}(t)=\left(\begin{array}{cccc}
a_{11}(t) & a_{12}(t) & \cdots & a_{1 n}(t) \\
a_{21}(t) & a_{22}(t) & \cdots & a_{2 n}(t) \\
\vdots & \vdots & & \vdots \\
a_{n 1}(t) & a_{n 2}(t) & \cdots & a_{n n}(t)
\end{array}\right)$$
我们定义
$$\begin{array}{c}
\int \boldsymbol{A}(t) \mathrm{d} t=\left(\begin{array}{cccc}
\int a_{11}(t) \mathrm{d} t & \int a_{12}(t) \mathrm{d} t & \cdots & \int a_{1 n}(t) \mathrm{d} t \\
\vdots & \vdots & & \vdots \\
\int a_{n 1}(t) \mathrm{d} t & \int a_{n 2}(t) \mathrm{d} t & \cdots & \int a_{n n}(t) \mathrm{d} t
\end{array}\right), \\
\int_{a}^{b} \boldsymbol{A}(t) \mathrm{d} t=\left(\begin{array}{cccc}
\int_{a}^{b} a_{11}(t) \mathrm{d} t & \int_{a}^{b} a_{12}(t) \mathrm{d} t & \cdots & \int_{a}^{b} a_{1 n}(t) \mathrm{d} t \\
\vdots & \vdots & & \vdots \\
\int_{a}^{b} a_{n 1}(t) \mathrm{d} t & \int_{a}^{b} a_{n 2}(t) \mathrm{d} t & \cdots & \int_{a}^{b} a_{n n}(t) \mathrm{d} t
\end{array}\right),
\end{array}$$

这里显然假设积分  $\int a_{i j}(t) \mathrm{d} t(i, j=1,2, \cdots, n)$  是存在的．
