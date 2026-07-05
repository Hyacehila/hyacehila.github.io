---
title: "高等代数：矩阵和线性空间学习笔记"
title_en: "Advanced Algebra: Matrices and Linear Spaces Notes"
date: 2023-06-17 23:01:45 +0800
categories: ["Mathematics", "Algebra & Matrix Theory"]
tags: ["Learning Notes", "Mathematics", "Linear Algebra"]
author: Hyacehila
excerpt: "一篇高等代数矩阵和线性空间学习笔记，整理矩阵运算、初等变换、矩阵的秩、逆矩阵、线性空间和二次型等内容。"
excerpt_en: "A study note on matrices and linear spaces in advanced algebra, covering matrix operations, elementary transformations, rank, inverse matrices, linear spaces, and quadratic forms."
mathjax: true
hidden: true
permalink: '/blog/2023/06/17/advanced-algebra-matrices-linear-spaces-notes/'
---
## 矩阵基础
### 矩阵的引出
我们前面其实已经用了不少的矩阵知识，他们都是我们研究矩阵的引子
* 高斯消元进行的初等行变换就是矩阵的初等变换
* 多个向量自然的会组成矩阵
* 向量组引入了秩的概念，他就是矩阵的秩，研究有效的向量的个数
* 方程组有解的条件就是方程组对应的矩阵与增广矩阵同秩

事实上，矩阵还在非常多领域发挥作用

**线性变换**
$$\left\{\begin{matrix}
 x=x^{\prime}\cos\theta-y\sin\theta\\
 y=x^{\prime}\sin\theta+y^{\prime}\cos\theta
\end{matrix}\right.$$
变换可以用下面的矩阵表示
$$\begin{pmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{pmatrix}$$

**二次曲线**
$$ax^2+2bx+cy^2+2dx+2ey+f=0$$
可以用下面的矩阵表示
$$\begin{pmatrix}a&b&d\\b&c&e\\d&e&f\end{pmatrix}$$

**多对多对应关系**
$s\times n$个对应关系可以用下面的矩阵表示
$$\begin{pmatrix}
  a_{11}&a_{12}  &\cdots  &a_{1n} \\
  \vdots &  &  &\vdots \\
  a_{s1}&a_{s2}  &\cdots   &a_{sn}
\end{pmatrix}$$
### 矩阵运算
#### 矩阵相等
定义：行数和列数都相等的矩阵称为同型矩阵

定义：矩阵的相等意味着所有的对应位置的元素都相同

定义：某个矩阵，如果他的行数和列数相等，则称为方阵
#### 矩阵加法
定义：只有同型矩阵可相加；结果是对应位置的量相加

定义：元素全为0的矩阵称为0矩阵，记为0

定义：所有元素前面都加负号，称为原本的矩阵的负矩阵 记为$-A$


性质：
* 结合率：$A+B+C=(A+B)+C=A+(B+C)$
* 交换律：$A+B=B+A$
* $A+0=A$
* $A+(-A)=0$
* $rank(A+B)\leq rank(A)+rank(B)$
#### 矩阵数乘
定义：一个矩阵数乘一个数字$k$ 是每个元素都乘$k$ 记为$kA$

性质
* $(k+l)A=kA+lA$
* $klA=k(lA)$
* $k(A+B)=kA+kB$
* $k(AB)=(kA)B=A(kB)$
#### 矩阵乘法
只有$A_{s\times n}$ 和 $B_{n\times m}$ 类型的矩阵才可以相乘，结果为 $C_{s\times m}$ 其他形式的矩阵不定义乘法。

我们记$C_{s\times m}$中的任意位置的元素为$c_{ij}$  那么有
$$c_{ij}=\sum_{l=1,k=1}^{l=n,k=n}a_{il}b_{kj}$$
也就是$A$的第$i$行和$B$的第$j$列处的对应位置的元素相加再求和

矩阵乘法提供了一种新的线性方程组的表示方法，记$A$为系数矩阵 $x$为自变量列向量 $B$是方程常数项列向量 则
$$Ax=B$$

矩阵乘法有下面的运算规律
* 不满足交换律，因为交换后不一定可乘
* 不满足消去率
* 满足结合率 $ABC=(AB)C=A(BC)$
#### 矩阵方幂
定义：主对角线为1，其余位置元素全为0的矩阵称为单位矩阵，记作$E_{n}$或者$I_n$  $n$是方阵的阶数 形式为
$$\begin{pmatrix}1&0&0&0\\0&1&0&0\\0&0&1&0\\0&0&0&1\end{pmatrix}$$

对于单位矩阵，容易知道
* $A_{s\times n} E_n=A_{s\times n}$
* $E_s A_{s\times n}=A_{s\times n}$

定义：对于方阵$A$ 我们把$k$个$A$相乘的结果称为方幂 记为$A^k$

对于方幂，容易知道
* $A^kA^l=A^{k+l}$
* $(A^k)^l=A^{kl}$

定义：我们把单位矩阵数乘一个数$k$的矩阵称为数量矩阵

容易知道，单位矩阵，数量矩阵，方阵的可交换。因此如果$AB=BA$  且$B$ 是任意矩阵，那么$A$是数量矩阵

在有了矩阵方幂的概念后，我们可以将矩阵与多项式联系起来，

定义：如下形式的多项式为方阵多项式，其中$A$是方阵
$$a_nA^n+\cdots+aA+E=f(A)$$
当$f(A)=0$的时候，则称这个多项式是方阵$A$的零化多项式

定义：如下形式的多项式为矩阵多项式，$B_i$是$n\times n$方阵 $\lambda$是数
$$\lambda^mB_0+\lambda^{m-1}B_1+\cdots+B_n$$
$n$称为阶数，$m$称为次数
#### 矩阵转置
定义：矩阵的转置是矩阵行列交换的过程，把$k$行变成$k$列，把$n\times s$矩阵转置成了$s\times n$矩阵 用符号$A^T$或者${A}'$表示

转置有下面的性质
* $(A^T)^{T}=A$
* $(A+B)^T=A^T+B^T$
* $(kA)^T=kA^T$
* $(AB)^T=B^TA^T$
* $|A|=|A^T|$

复矩阵也有转置的概念，需要在实矩阵转置的基础上再将各个元素求共轭
### 矩阵的逆
矩阵的逆的讨论只针对$n$阶的方阵进行

我们知道：$AE=EA$  单位矩阵$E$实际上是类似于1的概念

在初等的数学理论中，我们还存在倒数的概念，也就是$a\times \frac{1}{a}=1$  那矩阵中是否存在这样的概念，这就是矩阵的逆。

定义：对于$n$阶方阵$A$ 存在矩阵$B$使得 $AB=E$ 其中 $E$是$n$阶方阵，我们称$B$是$A$的逆矩阵，记作$A^{-1}$  并且任意的$A$对应的$B$是唯一的。

那么我们在矩阵的逆中就要讨论两个重要问题
* 矩阵的逆什么时候存在
* 矩阵的逆有什么通用的计算方法

定义：$A_{ij}$是矩阵$A$的元素$a_{ij}$的代数余子式，同[高等代数1 代数学基础中的行列式#行列式的普通展开](/blog/2023/03/17/advanced-algebra-foundations-notes/)

定义：$A^{\star}$是矩阵$A$的伴随矩阵，如果有
$$A^{\star}=\begin{pmatrix}A_{11}&\cdots&A_{1n}\\A_{n1}&\cdots&A_{nn}\end{pmatrix}$$

有以上定义容易得到 $AA^{\star}=dE$  其中 $d=|A|$  故矩阵$A$的逆矩阵为$\frac{1}{d}A^{\star}$  仅才$|A|\ne0$（矩阵$A$非退化）的时候存在

关于矩阵的逆和转置，有下面的结论
$$AB可逆\rightarrow AB\quad A^T可逆且(A^T)^{-1}=(A^{-1})^T\quad(AB)^{-1}=B^{-1}A^{-1}$$
**基于定义来计算矩阵的逆还是非常的繁琐，本文“矩阵基础#初等变换与初等矩阵”部分会给出更好的计算方法**
### 分块矩阵
我们在[高等代数1 代数学基础中的行列式#分块行列式](/blog/2023/03/17/advanced-algebra-foundations-notes/)里面直接给出了分块行列式完全适用普通行列式的运算法则，现在我们来把分块的概念推广到矩阵

#### 分块的拆分与基本运算
在处理一个高阶矩阵的运算问题的时候，我们把他拆分成若干个小的低阶矩阵来方便运算。拆分本身没有规则，但是我们要保证原本的运算依旧可进行，我们用一个例子来说明。

$$B=\begin{pmatrix}
  1&  0&3  &2 \\
  -1&2  &0  &1 \\
  1&0  &4  &1 \\
  -1&  -1&  2&0
\end{pmatrix}
=\begin{pmatrix}
  B_{11}&B_{12} \\
  B_{21}&B_{22}
\end{pmatrix}$$
$$A=\begin{pmatrix}
  1&  0&0  &0 \\
  0&1  &0  &0 \\
  -1&2  &1  &0 \\
  1&  1&  0&1
\end{pmatrix}
=\begin{pmatrix}
  E_{2}&A_{0} \\
  A_{1}&E_{2}
\end{pmatrix}$$
那么有
$$AB=\begin{pmatrix}
 B_{11} &B_{12} \\
  A_{1}B_{11}+B_{21}&A_{1}B_{12}+B_{22}
\end{pmatrix}$$
只要我们保持运算是可做的，那小矩阵的划分就是随意的，想要保证基本运算可作只需要保证，加法矩阵同型，乘法前行划分和后列划分一致
#### 分块矩阵转置
分块矩阵转置的方法如下
$$\begin{pmatrix}A_1&A_2\\A_3&A_4\end{pmatrix}^T=\begin{pmatrix}A_1^T&A_3^T\\A_2^T&A_4^T\end{pmatrix}$$
我们只需要
* 先当作普通矩阵进行转置
* 然后将分块矩阵的每个分块都转置
#### 分块矩阵求逆
**准对角矩阵**
$$\begin{pmatrix}A & 0\\ 0 & B\end{pmatrix}^{-1}=\begin{pmatrix}A^{-1} & 0\\ 0 & B^{-1}\end{pmatrix}$$

**准三角矩阵**
$$\begin{pmatrix}A & 0\\ C & B\end{pmatrix}\begin{pmatrix}x_1 & x_2\\ x_3 & x_4\end{pmatrix}=\begin{pmatrix}E_1 & 0\\ 0 & E_2\end{pmatrix}$$
解方程可以得到
$$\begin{pmatrix}A^{-1}&0\\-B^{-1}CA^{-1}&B^{-1}\end{pmatrix}$$

如果是上三角矩阵则应该先转置为下三角矩阵再套用公式，也就是
$$D^{-1}=((D^T)^{-1})^T$$
其中数的转置他的倒数
### 初等变换与初等矩阵
定义：$E$经过一次初等变换生成的矩阵是初等矩阵，经过三种初等变换（加和，交换，扩倍）$E$可以生成全部的初等矩阵。

引理：对于一个非$E$的$s\times n$矩阵进行初等行变换，则等价于左乘一个$s\times s$的同样进行该变换的$E$生成的初等矩阵；对于一个非$E$的$s\times n$矩阵进行初等列变换，则等价于右乘一个$n\times n$的同样进行该变换的$E$生成的初等矩阵。

定义：如果$A$可以由$B$通过初等变化生成，则称他们是**等价**的，这个矩阵的等价满足（**矩阵的重要关系——等价**）
* 自反性
* 对称性
* 传递性

定理：任意一个$s\times n$矩阵$A$都和$(\begin{array}{ll}\mathrm{E_r}&0\\0&0\end{array})\quad r=rankA$ 等价

根据前面的定义我们可以自然的给出两个推论
* 如果$A,B$等价，那么一定有$A=P_1P_2\cdots P_nBQ_1Q_2\cdots Q_m$ 其中的$P,Q$是行列变换矩阵
* 如果矩阵$A$可逆，那么其等价于一个同型单位矩阵$E$，也就是$A=P_1P_2\cdots P_nEQ_1Q_2\cdots Q_m$ 因此可逆矩阵可以变化为若干个初等矩阵的积

根据初等变换相关的性质，我们容易给出一个求本文“矩阵的逆”部分的方法，只需要进行一系列的初等行（列）变换

将矩阵$A$写成下面的形式，$E$是和$A$同型方阵
$$(A\mid E)$$
仅使用初等行变换，把矩阵左边的$A$转化为$E$ 此时矩阵右侧变换后的结果就是$A^{-1}$ 此方法可以推广到广义逆处使用，参考[矩阵分析中的减号逆 $A -$](/blog/2024/12/15/matrix-analysis-notes/)
### 初等变换与分块乘法
对分块单位矩阵的初等变换，等价于对数的初等变换

三种单位矩阵分块矩阵的初等变换有
$$\begin{pmatrix}Em&0\\0&En\end{pmatrix}\rightarrow\begin{pmatrix}0&En\\Em&0\end{pmatrix}\begin{pmatrix}D&0\\0&En\end{pmatrix}\begin{pmatrix}Em&0\\D&E_n\end{pmatrix}$$

对于非单位矩阵的分块矩阵的初等变换，仍旧满足：行变换等于左乘对应初等矩阵，列变换等于右乘对应初等矩阵。

对于形如
$$\begin{pmatrix}
  A&B \\
  C&D
\end{pmatrix}$$
的倍加行初等变换有
$$\begin{pmatrix}Em&O\\P&En\end{pmatrix}\times\begin{pmatrix}A&B\\C&D\end{pmatrix}=\begin{pmatrix}A&B\\C+PA&D+PB\end{pmatrix}$$
这个倍加运算可以让$C+PA=0$ 也就是$PA=-C$ 当$A$可逆的时候 有$P=-CA^{-1}$ 这样就可以构造出一个空的角，得到更容易推理的三角形矩阵，这就是分块矩阵常用的技巧——**打洞法**

我们给出一个简单的例子，对于$T=\begin{pmatrix}A&0\\ C&D\end{pmatrix}$  其中 $A,D$可逆 求 $T^{-1}$

我们打洞使得左下角的$C$变成0有
$$\begin{pmatrix}E_m&O\\-A^{-1}C&E_n\end{pmatrix}\begin{pmatrix}A&O\\C&D\end{pmatrix}=\begin{pmatrix}A&O\\O&D\end{pmatrix}$$
同时对两边求逆有
$$\begin{pmatrix}
 A^{-1} &O \\
 O &D^{-1}
\end{pmatrix}=T^{-1}\times B^{-1}$$
所以
$$\begin{pmatrix}
 A^{-1} &O \\
 O &D^{-1}
\end{pmatrix}\times B=T^{-1} $$
### 矩阵性质补充
#### 运算性质
对于矩阵的运算与秩，我们可以给出下面的性质
* $r(AB)\geq r(A)+r(B)-n$ 其中 $A_{m\times n},A_{n\times s}$
* $r(AB)\leq\min(r(A),r(B))$
* $r(A)=r(A^{T})=r(AA^{T})$
* $r(ABC)\geq r(AB)+r(BC)-r(B)$
* $r(A)=r(PA)=r(AQ)=r(PAQ)$ 如果 $P,Q$可逆

*
$$r(\begin{pmatrix}
 M &O \\
  K&N
\end{pmatrix})\geq r(\begin{pmatrix}
 M &O \\
  O&N
\end{pmatrix})=r(M)+r(N)$$
*
$$r(A^{\star})=\left\{\begin{matrix}
  n& r(A)=n\\
  1& r(A)=n-1\\
  0&r(A)<n-1
\end{matrix}\right.$$

#### 方阵求幂
对于求方阵幂$A^{k}$ 我们目前有下面几种常见的方法
* 计算三阶到五阶，归纳结果
* 单位矩阵的方幂满足$\begin{pmatrix}\lambda&0&0\\0&\lambda&0\\0&0&\lambda\end{pmatrix}^n=\begin{pmatrix}\lambda^n&0&0\\0&\lambda^n&0\\0&0&\lambda^n\end{pmatrix}$
* 如果矩阵恰好为初等矩阵，那不断求幂可能就是不断进行某种初等变换，使用变换的思路求解
* 二次式展开，把原本矩阵方幂拆为形如$(A+B)^n$ 的二次式，然后进行二次式展开，这种方法适用于$A,B$中有一个若干方幂为0，从而简化运算。

在后面的介绍中我们还会有别的求矩阵方幂的方法，如
#### 可交换矩阵
定义：对于两个方阵$A,B$  如果有 $AB=BA$ 则称两者可交换

定理：对角矩阵的可交换矩阵是对角矩阵

*证明方法是待定矩阵系数强行计算*
## 二次型
### 二次型的引入
本章我们的**基本研究对象为$n$元二次的方程**，其核心原理为：一切的有心的二次曲线均可通过坐标变换的形式转化为标准型。

定义：我们将满足下面的形式的方程称为一个二次型，并称为方程表示
$$f(x_1,x_2...x_{n})=\sum_{i=1}^{n}a_{ii}x_i^2+2\sum_{1\le i<j\le n}a_{ij}x_ix_j$$

定义：我们将前面的二次型的形式重新排列为矩阵，称为二次型的矩阵表示
$$\begin{pmatrix}
  x_1&x_2  &\cdots   &x_n
\end{pmatrix} \begin{pmatrix}
  a_{11}&a_{12}  &\cdots   &a_{1n} \\
  a_{21}&a_{22}  &\cdots  &a_{2n} \\
  \vdots & \vdots &\cdots   &\vdots \\
  a_{n1}&a_{n2}  &\cdots  &a_{nn}
\end{pmatrix}\begin{pmatrix}
 x_1\\
 x_2\\
 \vdots \\
x_n
\end{pmatrix}$$
中间的矩阵称为系数矩阵，容易看出，他满足$a_{ij}=a_{ji}$ 是一个对称矩阵

我们把列向量视为$X$ 系数矩阵记为$A$  则有二次型的矩阵简单表示 $X^TAX$

定义：实际上，第一种多项式定义的形式仍可以简化为
$$\sum_{i=1}^{n}\sum_{j=1}^{n}a_{ij}x_{i}x_{j}$$
### 线性替换
定义：当一个二次型的交叉项系数都为0的时候，我们称其为二次型的标准型，对应的系数矩阵是一个对角矩阵

定义：对于一个二次型$f(x_1,x_2...x_{n})$ 定义下面的替换
$$\begin{cases}x_{1}=a_{1}y_{1}+a_{2}y_{2}+\cdots+a_{n}y_{n}\\\vdots\\x_{n}=a_{1}y_{1}+a_{2}y_{2}+\cdots+a_{n}y_{n}&\end{cases}$$
是一组线性替换，将替换使用的系数矩阵$c_{ij}$ 记为$C$  如果$|C|\ne0$ 则称其为非退化的线性替换。

线性替换也可以用矩阵表示为
$$\begin{pmatrix}x_1\\x_2\\\vdots\\x_n\end{pmatrix}=\begin{pmatrix}c_{11}&\\&c_{nn}\end{pmatrix}\begin{pmatrix}y_1\\\vdots\\y_n\end{pmatrix}$$
也就是$X=CY$  其中 $Y$ 是变换的系数矩阵

定理：使用定义能够证明，对原本系数矩阵为$A$的二次型$X^TAX$ 做线性替换 $X=CY$ 的结果仍是一个二次型 $Y^T(C^TAC)Y$ 其系数矩阵为 $C^TAC$

定义：如果矩阵$A,B,C$ 其中$C$可逆 满足关系 $B=C^TAC$  则称矩阵$A,B$ 合同，**合同关系也是一种重要的矩阵关系**，他满足
* 自反性
* 对称性
* 传递性

### 化标准型
**定理：任何一个二次型均可以通过非退化线性替换变为标准型** 证明这个定理的过程就是寻找标准型的过程

#### 换元法
**当二次型含有平方项**

例如：$f=x_{1}^{2}-3x_{2}^{2}-2x_{1}x_{2}+2x_{1}x_{3}-6x_{2}x_{3}$

我们先把所有$x_1$ 凑到一起，得到平方式（需要用完$x_1$，可以额外凑点其他）的项，有$x_1^2-2x_1x_2+2x_1x_3$  我们可以凑成
$$(x_1-(x_2-x_3))^2$$
至于缺少的$(x_2-x_3)^2$ 可以再减掉

再把所有含$x_2$的凑到一起，重复前面的过程，直到把所有的项都凑成平方项，得到一个全是平方式的结果，如
$$y_1^2+y_2^2+\cdots+y_n^2$$
其中$y_i$是含$x_i$多项式，最后换元就可以得到替换的样子了

**当不含平方项的时候**

例如：$f=2x_1x_2+2x_1x_3-6x_2x_3$

此时我们需要一步额外的变换为
$$\left\{\begin{matrix}
 x_1=y_1+y_2\\
 x_2=y_1-y_2\\
x_{3}=y_{3}
\end{matrix}\right.$$
此时就变成了有二次项的结果，继续沿用

#### 合同变换法
我们可以通过多次合同的变换将原始矩阵变为标准型，并且记录我们的变换

**合同变换要求，所有对行进行的初等变换要再对列进行一次，这是这种变换的核心**

将原始矩阵写为
$$\begin{pmatrix}
 A\\
E
\end{pmatrix}$$
将$A$矩阵用合同变换变换为对角矩阵，那么$E$就是我们的$C$  此时$C^TAC$ 就是我们找的标准型。

实操方法有
1. 当$a_{11}\ne0$  化 $a_{i1},a_{1i}$为0 降为低阶矩阵处理
2. 当$a_{11}=0$ 且此时 $a_{ii}\ne0$ 那么利用交换将$a_{11}\ne0$ 转回1
3. 当$a_{11}=0$ 且此时 $a_{ii}=0$  利用行列倍加将$a_{11}\ne0$ 转回1
4. 完成降阶后，重复这个步骤处理低阶矩阵直到对角

**标准型不唯一，因此答案只用做参考，需要自行验证系数矩阵是否正确**
### 唯一性与规范性
我们在前面的不难发现。同一个二次型的标准型是不唯一的，但是它们之间互相合同 ，故同秩。因此在一个标准型中系数不为0的平方项个数一样 ，与所做的非退化线性替换无关。 所以秩才是衡量二次型唯一性的核心标准

**在复数域上** 一个标准型为下面的形式
$$d_1y_1^2+d_2y_2^2+\cdots+d_ry_r^2\quad d_i\neq0$$
我们取$y_{r}=\frac{1}{\sqrt{d_r}}z_{r}$，其中根号内允许负数，可以化简二次型为
$$z_1^2+z_2^2+\cdots+z_r^2$$

也就是说：**两个复矩阵合同等价于他们有着相同的秩，复数域上有着二次型的确定形式**

我们把系数全为1的标准型称为 **规范型** 复数域上的规范型具有唯一性

**在实数域上** 我们可以使用一样的变换思路，但是无法把负号变为正号，因此二次型化简为
$$z_1^2+z_2^2+\cdots+z_p^2-z_{p+1}^2\cdots-z_r^2$$
实数域上的规范型被秩$r$以及正项数$p$同时决定，也具有唯一性

我们把规范性的唯一性称为二次型的惯性定理。
### 符号差与正定型
#### 符号差与惯性指数
在本节，我们只讨论实数域上的二次型，讨论其规范型的进一步衍生

定义：对于规范型$f(x_1.x_2,\cdots,x_n)$ 正平方项个数$p$称为正惯性指数，负平方项个数$r-p$称为负惯性指数，他们的差$2p-r$称为符号差

如果我们想从规范型研究二次型，那么
* 对于$n$阶复二次型，一共有$n+1$种，分别是秩从$0$到$n$
* 对于$n$阶实二次型，我们有$n+1$种秩情况，每种秩情况对应$rank-1$种正负惯性情况，总共有$\frac{n(n+1)}{2}$种合同结构

#### 正定二次型
定义：对于二次型$f(x_1.x_2,\cdots,x_n)$ ，如果对于任意一组不为0的$(c_1\cdots c_n)$ 都有$f(c_1\cdots c_n)>0$ 则称该二次型为正定二次型

显然的$f(x_{1}\cdots x_{n})=x_{1}^{2}+x_{2}^{2}+\cdots+x_{n}^{2}$是正定的

不难验证，$f(x_1\cdots x_n)=d_1x_1^2+\cdots+d_nx_n^2$ 正定等价于 $d_i>0$ 对所有$i$成立

我们这里给出重要定理：**对二次型的非退化的线性替换，或者说对其系数矩阵的合同变换，不改变其正定性**  对于正定矩阵 一定有$p=n$ 也就是正惯性指数和矩阵维数一样

至此，想要判断一个矩阵（二次型）是否是正定的，有三种方式
* 定义
* 化规范型
* 计算正惯性指数

#### 正定矩阵
定义：如果二次型 $XA^TX$ 是正定的，则称其系数矩阵$A$是正定矩阵

定理：合同于单位矩阵的矩阵是正定的

定理：正定矩阵的行列式大于0，对应的逆命题不成立

现在我们提出新的方法，直接从矩阵本身研究其正定性

定义：
* 子式：任取矩阵若干行列交点形成的
* 代数余子式与余子式：同[高等代数1 代数学基础中的行列式#行列式的普通展开](/blog/2023/03/17/advanced-algebra-foundations-notes/) 与 [高等代数1 代数学基础中的行列式#拉普拉斯展开](/blog/2023/03/17/advanced-algebra-foundations-notes/)中定义
* 主子式：取相同的行号与列号的子式 $i$ 阶主子式不唯一
* 顺序主子式：$i$阶顺序主子式取前$i$行与前$i$列，是唯一的

定理：矩阵$A$（$n$阶对称方阵）是正定矩阵的充要条件为，所有顺序主子式均大于0，也就是各阶顺序主子式正定

推论：矩阵$A$正定，则有其伴随矩阵$A^{\star}$正定

#### 正定的平行概念
定义：对于二次型$f(x_1.x_2,\cdots,x_n)$ ，如果对于任意一组不为0的$(c_1\cdots c_n)$ 都有$f(c_1\cdots c_n)<0$ 则称该二次型为负定二次型，系数矩阵称为负定矩阵

定义：如果
* $f(c_1\cdots c_n)\ge0$ 则称为半正定
* $f(c_1\cdots c_n)\le0$ 则称为半负定

定理：如果二次型$f(x_1.x_2,\cdots,x_n)$正定，则二次型$-f(x_1.x_2,\cdots,x_n)$ 负定

对于半正定矩阵，下面命题等价
* $f(x_1.x_2,\cdots,x_n)$半正定
* 正惯性指数$p$ 和秩$r$ 相等，不要求等于$n$
* 合同规范型 $d_i\ge0$
* 所有的主子式均非负（从顺序主子式到主子式，从正到非负）、
* 各阶顺序主子式半正定

**我们的在二次型一章的核心思想在于从特殊到一般，研究隐藏在背后的规范形式**
## 线性空间
### 集合与映射
#### 集合
关于集合和映射的叙述在[数学分析1 极限与连续理论](/blog/2023/03/16/mathematical-analysis-limits-continuity-notes/)已经介绍的很详细了，这里复习其中的必要知识作为后面关于线性空间的讨论的基础。

定义：集合，把一些东西看作整体，至面这些东西称为元素 记作 $a\in S$

定义：不包含任何元素的集合称为空集合，记$\phi$ ，但是我们构造集合的集合$\{\phi\}$ 是非空集合

定义：$当a\in M\text{ 当且仅当 }a\in N\text{时 则称两个集合相等}$

定义：$当a\in M\Rightarrow a\in N则称M是子集M\subset N$  同理可以定义真子集

定义：$若M\subset N且N\subset M 则 N=M$

关于集合的交 并 补之类的集合运算这里略去

#### 映射
定义：设$M~ ~M'$是两个集合， $M$到$M'$的映射 指一个法则 它使 $M$中每一个
元素$a$ 都有$M^{\prime}$中另一个元素$a^{\prime}$与之对应,记作$\sigma(a)=a^{\prime}$ 或$\sigma: M\rightarrow M^{\prime }$ $a^{\prime}$构为像 $a$称为原像

定义：M到M的映射称为变换

映射中，$M$的所有元素都需要被映射，但是$M^{\prime}$ 中并不是所有元素都能找到原像

$\sigma,\sigma^{\prime}$ 相等的条件是 对应的集合相同 且$\sigma(a)=\sigma^{\prime}(a)$

定义：将$a$映到$a$的映射$(\sigma(a)=a)$称作单位映射 或 恒等映射   记作$1_{m}$

定义：映射的复合也叫 映射的乘积 记作 $\sigma t$ 。运算时需要向右结合，不能交换运算顺序

定义： 当$\sigma(M)=M^{\prime}$  即$M^{\prime}$中所有像都可以找到原象。称为满射或者映上

定义：当$M^{\prime}$中每个像。对应的原像都不同 。也就是不存在一个象对应了两个原像的情况，称为单射（1-1的）

定义：若一个 映射同时是单射和满射，则其为双射（1-1对应的）

定义：对于一个双射$\sigma: M\to M^{\prime}$  逆映射为 $\sigma^{-1}:M^{\prime}\to M$

我们很容易的给出逆映射的性质
* $\sigma\sigma^{-1}=1$
* $\sigma^{-1}\sigma=1$
* $(\sigma^{-1})^{-1}=\sigma$

定理：两个双射复合，结果还是双射
### 线性空间的定义
在以前诸多的学习中，我们发现有连续函数、$n$ 元有序数组、矩阵等诸多概念均是具备加法和数乘运算的集合。我们因此尝试抽象出一个模型用来解决这一类问题。

定义：设是$V$非空集合，$P$是一个数域，定义中$V$元素的一种加法运算和数乘运算（可以自由的定义），若加法的和、数乘的积仍在$V$中（对加法和数乘封闭），并且满足以下规则，则称是线性空间
* $\alpha+\beta=\beta+\alpha$
* $(\alpha+\beta)+\gamma=\alpha+(\beta+\gamma)$
* 存在元素0，使得$\alpha=0+\alpha$
* 存在负元素，使得$\alpha+\beta=0$
* 存在元素1，使得$1 \alpha =\alpha$
* $k(l\alpha)=(kl)\alpha$
* $(k+l)\alpha=k\alpha+l\alpha$
* $k(\alpha+\beta)=k\alpha+k\beta$

**值得强调的是，我们这里的加法运算和数乘运算都是自己定义的，元素0和元素1也不一定是传统的自然数0和1，需要根据定义推算**

线性空间也成为向量空间，这里的向量是广义的向量，只要是线性空间的元素就称为向量。

我们可以自然的给出一些线性空间的例子
* 数域$P$上的一元多项式环 $P[x]$
* 数域$P$上的一元多项式环 $P[x]$ 但是只取次数小于$n$的部分
* 闭区间$[a,b]$上的连续函数 $C[a,b]$
* 数域$P$ 上的$n$元有序数组
* 数域$P$ 上的某型矩阵 $P^{m\times n}$
* 数域$P$自身

对于线性空间，我们还是可以给出一些定理以及推出一些其他的运算

定理：在一个线性空间中 零元素$0$和某个元素$\alpha$的负元素$-\alpha$是唯一的

定义：线性空间上的减法可以视为加上他的负元素

三条线性空间上的结论
* $0\alpha=0$
* $k0=0$
* $(-1)\alpha=-\alpha$

**同样的集合，定义不同形式的运算或者依赖不同的数域，就会形成不同的线性空间**，线性空间的种类非常多变，以至于我们完全无法列举，要根据此时的语境理解我们在研究什么样的空间。
### 维数，基与坐标
首先，我们需要回顾一下在前面向量空间学习的知识点，并把他们自然的推广，包括线性相关，线性无关，线性表出等。

定义：若$V$中有$n$个线性无关的向量 但是没有更多的无关向量则称其是 $n$维的 若可以找到无限个无关向量 则称其是无限维的。 无限维向量不是我们研究的重点

定义：在$n$维线性空间中 找到$n$个无关的向量$\varepsilon_1\cdots\varepsilon_n$  它们可以 线性表出$V$中的一切向量 这便称之为一组基。**基和维数关联，但是并不唯一**

定义：在$n$维线性空间中 向量$\alpha$可以被基$\varepsilon_1\cdots\varepsilon_n$  表出$\alpha=a_{1}\varepsilon_{1}+a_{2}\varepsilon_{2}+\cdots+a_{n}\varepsilon_{n}$  那么这组数$(a_1,a_2\cdots a_n)$ 就是向量$\alpha$在这一组基下的坐标  **坐标同时被基和向量影响**

从三个连续的定义可以看出，维数和基的问题应该一起解决，不应该分开单独考虑。

定理：若线性空间$V$中有$n$个线性无关的向量 $\alpha_1,\cdots,\alpha_n$ 且$V$中任一向量都可以由它们表出则$V$是$n$维的 $\alpha_1,\cdots,\alpha_n$便是一组基

定理：基不具有唯一性，和基等价的向量组也是基 ，故存在标准基的概念

基体现了无限向量量有限化的思想 ，这是非常重要的

在 $n$ 维线性空间$P^n$中，我们一般把下面的一组基称为标准基
$$\left\{\begin{matrix}
 \epsilon_{1}=(1,0,0\cdots,0)\\
 \epsilon_{2}=(0,1,0,\cdots0)\\
 \vdots\\
\epsilon_{n}=(0,0,\cdots,1)
\end{matrix}\right.$$
其他的任何基向量组都和标准基等价
### 基变换与坐标变换
现在我们来系统的研究一下，如何进行基的变换，并且此时坐标将如何变换

我们给出基变换如下
$$\left\{\begin{matrix}
\varepsilon_{1}^{\prime}=a_{11}\varepsilon_{1}+a_{12}\varepsilon_{2}+\cdots+a_{1n}\varepsilon_{n} \\
\varepsilon_{2}^{\prime}=a_{21}\varepsilon_{1}+a_{22}\varepsilon_{2}+\cdots+a_{2n}\varepsilon_{n} \\
\cdots \\
\varepsilon_{n}^{\prime}=a_{n1}\varepsilon_{1}+a_{n2}\varepsilon_{2}+\cdots+a_{nn}\varepsilon_{n}
\end{matrix}\right.$$

使用矩阵乘法表示有
$$(\varepsilon_{1}^{\prime},\varepsilon_{2}^{\prime},\cdots\varepsilon_{n}^{\prime})=(\varepsilon_{1},\varepsilon_{2},\cdots\varepsilon_{n})\begin{pmatrix}
  a_{11}&a_{12}  & \cdots &a_{1n} \\
  a_{21}&a_{22}  & \cdots &a_{2n} \\
  \vdots&\vdots  &\ddots  &\vdots \\
  a_{n1}&a_{n2}  & \cdots &a_{nn}
\end{pmatrix}^{T}$$
我们把矩阵的转置称为过渡矩阵$A$

并且有
$$\varepsilon=\varepsilon^{\prime}A^{-1},\varepsilon^{\prime}=\varepsilon A$$
这就是基变换的公式

对于一个向量和他对应的坐标，我们容易给出
$$x=(\varepsilon_{1},\varepsilon_{2},\cdots\varepsilon_{n})\begin{pmatrix}
 x_1\\
 x_2\\
 \vdots\\
x_n
\end{pmatrix}=(\varepsilon_{1}^{\prime},\varepsilon_{2}^{\prime},\cdots\varepsilon_{n}^{\prime})\begin{pmatrix}
 x_1^{\prime}\\
 x_2^{\prime}\\
 \vdots\\
x_n^{\prime}
\end{pmatrix}$$

代入前面得到的基变换的公式有
$$A^{-1}x=x^{\prime}$$
也就是
$$x=Ax^{\prime}$$
其中$x,x^{\prime}$是列向量 $A$是我们前面给出的矩阵

仅凭感觉 我们也不难发现 在标准基之间寻找过渡矩阵是简单的，而两个非标准基寻找过渡矩阵 非常复杂 让我们用例子来解释一下。

两个基分别为
$$\left\{\begin{matrix}
 \epsilon_{1}=(1,0,0\cdots,0)\\
 \epsilon_{2}=(0,1,0,\cdots0)\\
 \vdots\\
\epsilon_{n}=(0,0,\cdots,1)
\end{matrix}\right.\quad\left\{\begin{matrix}
 \epsilon_{1}^{\prime}=(1,1,1\cdots,1)\\
 \epsilon_{2}^{\prime}=(0,1,1,\cdots1)\\
 \vdots\\
\epsilon_{n}^{\prime}=(0,0,\cdots,1)
\end{matrix}\right.$$
那么我们容易知道从标准基到后者的过渡矩阵为
$$\left.\left.\left.\left(\begin{array}{cccc}1&1&1&\cdots&1\\0&1&1&\cdots&1\\0&0&1&\cdots&1\\0&0&0&\cdots&1\end{array}\right.\right.\right.\right)^T$$
我们只需要观察就够了

因此，对于求非标准基间的过渡矩阵，我们给出中介法，对于基$A(a_1,a_2,\cdots,a_n)$ 基$B(b_1,b_2,\cdots,b_n)$
则有基$A$  等于 自然基 乘 矩阵 $A$  基$B$  等于 自然基 乘 矩阵 $B$

代入计算有  基$B$ 等于 基A 乘 矩阵$A^{-1}$ 乘 矩阵$B$  则过渡矩阵为$A^{-1}B$
### 线性子空间
不难看出，有的线性空间是一个线性空间的一部分 这个概念当然也有研究的价值

定义： 数域$P$上的线性空间$V$的一个非空集合$W$ 称为$V$的一个线性子空间。如果$W$对于$V$的两种运算也构成线性空间

定理：若线性空间$V$的非空子集合$W$对$V$的两种运算封闭，那么$W$就是一个子空间，不要再验证八条性质。

线性子空间也是一个线性空间，它也有维数与基的概念。由于其不可能比整个空间有更多的无关向量 故维数只能小于等于原维数

E.G.
* 由单个0向量组成的子集合是一个线性空间，称为0子空间，其维数为0
* 线性空间$V$也是$V$的一个子空间两者维数相等
* 任意的线性空间都有这两个空间，他们被称为平凡子空间，其它的线性子空间叫做非平凡子空间


定义：设$\alpha_1\alpha_2\cdots\alpha_n$ 是线性空间$V$中的一组向量 ，不难看出，这组向量的所有线性组合$k_{1}\alpha_{1}+k_{2}\alpha_{2}+\cdots+k_{n}\alpha_{n}$构成了非空的 对两种运算封闭的集合，因而是$V$的一个子空间 记作由向量组 $\alpha_1\alpha_2\cdots\alpha_n$ 生成的子空间 记作$L(\alpha_1,\alpha_2\cdots \alpha_n)$  其中$\alpha_1\alpha_2\cdots\alpha_n$称为其生成元向量组

在有限维线性空间中，任何一个子空间都可以这样得到。用它的基作为生成元向量组就好了。

生成元向量组可以相关(即不一定为基) 若生成元向量组线性无关 则它一定是基。反之，研究它的极大无关组就好了。维数研究向量组的秩就可以了。

本节的定理：
* 两个向量组生成相同子空间则这两个向量组等价
* 生成子空间是包含生成元向量组的最小子空间
* 设$W$是 $n$维线性空间$V$的子空间 其基为 $w_1,\cdots,w_m$($m$维的)，则必可找到 $n-m$ 个向量 把这组基扩为$V$的一组基 (通议自然基)
### 子空间的交与和
定理：如果$V_1,V_2$是$V$的两个线性子空间，则它们的交也是$V$的线性子空间（空间的交就是集合的交） 用符号 $V_{1}\cap V_{2}$ 表示

对于空间的交，我们有
* $V_1\cap V_2=V_2\cap V_1$
* $(V_1\cap V_2)\cap V_3=V_1\cap(V_2\cap V_3)$

定义：如果$V_1,V_2$是$V$的两个线性子空间，所谓的子空间的和$V_1+V_2$是指所有可以表示为$\alpha+\beta$ 其中 $\alpha\in V_{1},\beta \in V_2$ 的向量构成的子集合

定理：如果$V_1,V_2$是$V$的两个线性子空间，那么 $V_1+V_2$ 也是$V$的线性子空间

对于空间的和，我们有
* $V_1+V_2=V_2+V_1$
* $(V_1+V_{2})+V_{3}=V_{1}+(V_{2}+V_{3})$

关于空间的交与和，下面的结论显然成立
* $W\subset V_1\quad W\subset V_2\quad\Rightarrow W\subset V_{1}\cap V_2$
* $V_1\subset W\quad V_2\subset W\Rightarrow V_1+V_2\subset W$
* $V_{1}\subset V_2\Longleftrightarrow V_1\cap V_2=V_1\Longleftrightarrow V_1+V_2=V_2$
### 子空间的维数公式
引理：对于张成子空间的和，我们有$L(\alpha_1,\alpha_2\ldots\alpha_s)+L(\beta_1,\beta_2\ldots\beta_r)=L(\alpha_1\ldots\alpha_s,\beta_1\ldots\beta_r)$

定理（维数公式）：我们用$dim(A)$表示空间$A$的维数，那么有$dim(V_1)+dim(V_2)=dim(V_1+V_2)+dim(V_{1}\cap V_2)$

推论：从维数公式不难发现，子空间和的维数一般小于子空间维数的和，只有$dim(V_{1}\cap V_2)=0$的时候才相等。那么，如果$n$维线性空间的两个子空间维数的和大于$n$，一定能得到这两个子空间有非零公共向量。

下面我们简单说明如何用维数公式研究空间的维数与基

对于和空间：借助前面给出的引理，计算生成元向量的组极大无关组，就可以同时得到维数与基

对于交空间：维数公式可以帮助我们得到交空间的维数。对于求基，我们可以给出张成空间方程有 $V=x_{1}\alpha_{1}+\cdots+x_{s}\alpha_{s}=y_{1}\beta_{1}+\cdots+y_{r}\beta_r$  其中 $\alpha,\beta$是子空间的基，$x,y$是方程未知数。

我们解这个方程，得到系数$x$或者系数$y$的关系，就可以得到张成交空间的向量，其基和维数就容易计算了。
### 子空间的直和
定义：设$V_1,V_2$是$V$的两个线性子空间，如果和 $V_1+V_2$ 中的每个向量$\alpha$ 的分解式是唯一的（$\alpha=\alpha_{1}+\alpha_{2}\quad\alpha_{1}\in V_{1}\quad\alpha_{2}\in V_{2}$） 则称这个和运算是直和，记作$V_1\oplus V_2$

**线性空间的直和只是和运算的一种特殊形式，直和意味着对于$\alpha=\alpha_{1}+\alpha_{2}\quad\alpha_{1}\in V_{1}\quad\alpha_{2}\in V_{2}$ 我们找到的$\alpha_1,\alpha_2$是确定且唯一的**

下面我们研究直和的判定定理，借此理解直和究竟意味着什么

定理：$V_1+V_2$是直和，当且仅当零向量的分解式唯一；也就是$\alpha_1+\alpha_2=0\quad\alpha_1\in V_1\quad\alpha_2\in V_2\quad\Rightarrow\alpha_1=\alpha_2=0$

定理：$V_1+V_2$是直和当且仅当$V_{1}\cap V_{2}=\{0\}$；也就是交空间维数为0

定理：设$V_1,V_2$是$V$的两个线性子空间，则$V_1+V_2$是直和的充要条件为$dim(V_1)+dim(V_2)=dim(V_1+V_2)$

定理（余子空间）：设$U$是$V$的子空间，那么一定存在$V$的子空间$W$使得$V=U\oplus W$ 此时我们称$W$是$U$对于$V$的余子空间

**只有$U$是平凡子空间的时候，余子空间才具有唯一性**

定理：设$(\varepsilon_{1},\varepsilon_{2},\cdots\varepsilon_{n})$ 和$(\eta_{1},\eta_{2},\cdots \eta_{n})$ 是$V_1,V_2$的一组基，那么$V_1+V_2$是直和当且仅当$(\varepsilon_{1},\varepsilon_{2},\cdots\varepsilon_{n},\eta_{1},\eta_{2},\cdots \eta_{n})$ 线性无关

### 线性空间的同构
线性空间的同构类似于解的结构，矩阵的单位矩阵，二次型的标准型。是一种以代多，以简代零的思路。致力于寻找线性空间最本质的东西，也就是研究维数与基。

设 $\varepsilon_{1},\varepsilon_{2},\cdots\varepsilon_{n}$ 是 $V_n^P$上的一组基 那么任意 $V$ 上的向量必有一个坐标 是 $P^n$中的元素。即向量与坐标的关系是一个$V\to P^n$上的映射 这个映射必是 一个双射。并且这个映射关系反应到线性运算上是同样成立的

**这意味着：原本对抽象空间的研究，都可以到我们最熟悉的空间$P^n$中**

定义：数域$P$上的两个空间同构，当且仅当存在一个$V$到$V^{\prime}$的双射$\sigma$使得
$\sigma(\alpha+\beta)=\sigma(\alpha)+\sigma(\beta)$   $\sigma(k\alpha)=k\sigma(\alpha)$ 其中 $\alpha,\beta \in V.$ $k\in P$
这样的映射称为同构映射，记作计作$V\cong V^{\prime}$

**同构是两个空间之间的关系，不一定和$P^n$关联。但是任意空间$V$一定能找到和他同构的$P^n$**  使用同构的空间$P^n$是处理较为抽象的线性空间的很好方法

对于同构，我们很容易给出下面的性质
* $\sigma(0)=0\quad\sigma(-\alpha)=-\sigma(\alpha)$
* $\sigma(k_{1}\alpha_{1}+k_{2}\alpha_{2}+\cdots+k_{r}\alpha_{r})=k_{1}\sigma(\alpha_{1})+k_{2}\sigma(\alpha_{2})+\cdots+k_{r}\sigma(\alpha_{r})$
* $V$中的向量组  $\alpha_{1},\alpha_{2}\cdots \alpha_{r}$   线性无关。当且仅当 $\sigma(\alpha_{1}),\sigma(\alpha_{2})\cdots \sigma(\alpha_{r})$  线性无关
* 同构空间有相同的维数
* 如果$V_1$是$V$的子空间，那么$\sigma(V_1)$是$\sigma(V)$的子空间，并且$V_1$和$\sigma(V_1)$有相同的维数。映射$\sigma$是同构映射
* 同构映射的逆映射，同构映射的积仍旧是一个同构映射

定理（同构定理）：**两个线性空间同构的充要条件为两者的维数相等**，这个定理告诉了我们，维数是线性空间的最本质特征。
