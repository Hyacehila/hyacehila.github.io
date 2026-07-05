---
title: "高等代数：线性变换与欧式空间学习笔记"
title_en: "Advanced Algebra: Linear Transformations and Euclidean Spaces Notes"
date: 2024-12-07 16:19:23 +0800
categories: ["Mathematics", "Algebra & Matrix Theory"]
tags: ["Learning Notes", "Mathematics", "Linear Algebra"]
author: Hyacehila
excerpt: "一篇高等代数线性变换与欧式空间学习笔记，整理线性变换、特征值、Jordan 标准型、欧式空间、酉空间和正交变换。"
excerpt_en: "A study note on linear transformations and Euclidean spaces, covering linear transformations, eigenvalues, Jordan normal form, Euclidean spaces, unitary spaces, and orthogonal transformations."
mathjax: true
hidden: true
permalink: '/blog/2024/12/07/advanced-algebra-linear-transformations-euclidean-spaces-notes/'
---
## 线性变换
的向量组
### 线性变换的定义
在上一章的结尾 我们用同构解释清楚了一个线性空间的最本质。但是研究线性空间之间的联系也是非常重要的。这体现为线性空间之间的映射。

从线性空间映射到线性空间的映射称为变换。而变换中最简单的线性变换便是本章要讨论的内容。

**同构就是构建了线性空间$V$到最经典的线性空间$P^n$之间的双射，但是我们在同构一节侧重于研究空间的性质而不是变换本身**

定义： 对于数域$P$ 上的线性空间 $V$ 若对于任意$\alpha,\beta \in V,p\in P$ 都有$A(a+\beta)=A(\alpha)+A(\beta)$    $A(k\alpha)=kA(\alpha)$恒成之 则称变换$A$ 是一个线性变换    也就是说**保加法和数乘的变换**

下面是一些线性变换的例子
* 恒等变换 $A(\alpha)=\alpha$
* 零变换 $A(\alpha)=0$
* 数乘变换 $A(\alpha)=kA(\alpha)$
* 微分变换（对原函数求微分）
* 积分变换（对原函数求不定积分）
* 向量的变换 $(\begin{array}{c}x^{\prime}\\y^{\prime}\end{array})=(\begin{array}{cc}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{array})(\begin{array}{c}x\\y^{\prime}\end{array})$

线性变换有下面的性质
* $A(0)=0$
* $A(-\alpha)=-A(\alpha)$
* 保线性组合式 $\beta=k_{1}\alpha_{1}+\cdots+k_{1}\alpha_{n}\Rightarrow A(\beta)=kA(\alpha_{1})+\cdots+k_{n}A(\alpha_{n})$
* 线性变换把线性相关向量组映射为线性相关向量组

定理：在有限维线性空间上，单射，满射，双射是等价的；他们都把线性相关的向量组映射为线性相关的向量组，把线性无关的向量组映射为线性无关
### 线性变换的运算
现在我们了解线性变换这种线性空间上的运算该如何进行运算

定义（乘法）：$AB(\alpha)=A(B(\alpha))$ 是线性变换的乘法，$A,B$是线性变换

对于线性变换的乘法有
* 两个现象变换的乘积还是线性变换
* 适用于结合率，但是不适用交换律
* 定义单位变换$\varepsilon$  如果有 $\varepsilon A=A\varepsilon=A$

定义（加法）：$(A+B)(\alpha)=A(\alpha)+B(\alpha)$是线性变换的加法

对于线性变换的加法有
* 线性变换的和还是线性变换
* 适用于结合率，适用交换律
* 对于零变换0有 $A+0=A$
* 可以据此定义负变换$(-A)(\alpha)=-A(\alpha)$  他也是一个线性变换
* 结合乘法与加法有$A(B+C)=AB+AC$

定义（数量乘法）：$(kA)(\alpha)=kA(\alpha)$ 是线性变换的乘法

对于线性变换的数量乘法有
* 线性变换的数量乘法还是线性变换
* $(kl)A=k(lA)$
* $(k+l)A=kA+lA$
* $k(A+B)=kA+kB$
* $1A=A$

定义（逆变换）：$\sigma$ 是 $V$ 上的一个线性变换，如果$\sigma\tau=\tau\sigma=\text{单位变换}$  则称$\sigma,\tau$ 是可逆变换，他们而这互为对方的逆变换

对于逆变换，我们可以给出下面的性质
* 逆变换也是一个线性变换
* $\sigma\in L(V) 可逆 \Longleftrightarrow \sigma是双射\Longleftrightarrow\sigma是 V对V的同构映射$
* $\sigma$可逆 则$\sigma$把基映射为基，把无关组映射为无关组

在经历了对线性变换的定义与运算的定义后，我们容易发现，对于**空间$V$在数域$P$上的全体线性变换，也构成一个对于数域$P$的线性空间，记作$L(V)$**
### 线性变换的多项式
我们可以定义线性变换的幂，将
$$\sigma^{n}=\sigma \sigma {\cdots}\sigma$$
为线性变换的$n$次幂

定义负整数幂为
$$(\sigma^{-1})^n=\sigma^{-n}$$

在有了幂以后，我们可以定义线性变换的多项式，对于多项式
$$f(x)=a_mx^{m}+a_{m-1}x^{m-1}+\cdots+a_{0}$$
他对应的线性变换$A$的多项式为
$$f(A)=C_mA^{m}+C_{m-1}A^{m-1}+\cdots+C_0\varepsilon.$$

线性变换的多项式保证加法和相关运算率
### 线性变换的矩阵
#### 线性变换的矩阵的定义
我们容易看出，如果$\varepsilon_i$是一组基，那么对于这个空间上的任意向量$\varepsilon$有$\varepsilon=a_1\varepsilon_1+a_2\varepsilon_2+\cdots+a_n\varepsilon_n$  则有$A\varepsilon=a_1A\varepsilon_1+a_2A\varepsilon_2+\cdots+a_nA\varepsilon_n$

这个性质意味着，只需要知道所有的基的像，就可以知道这个线性变换的所有信息，我们可以据此去研究一个线性变换。

定义：设$\varepsilon_i$是线性空间$V$一组基，$A$是他上面的一个线性变换，因此基向量变换后的像可以由原本的基向量线性表出（变换的定义），也就是
$$\left\{\begin{matrix}
 A\varepsilon_{1}=a_{11}\varepsilon_{1}+a_{12}\varepsilon_{2}+\cdots+a_{1n}\varepsilon_{n}\\
 A\varepsilon_{2}=a_{21}\varepsilon_{1}+a_{22}\varepsilon_{2}+\cdots+a_{2n}\varepsilon_{n}\\
 \cdots\\
A\varepsilon_{n}=a_{n1}\varepsilon_{1}+a_{n2}\varepsilon_{2}+\cdots+a_{nn}\varepsilon_{n}
\end{matrix}\right.$$
我们记
$$\begin{pmatrix}
  a_{11}&a_{12}  & \cdots &a_{1n} \\
  a_{21}&a_{22}  & \cdots &a_{2n} \\
  \vdots&\vdots  &\ddots  &\vdots \\
  a_{n1}&a_{n2}  & \cdots &a_{nn}
\end{pmatrix}=A$$
则有原变换表示为 $A(\varepsilon_1,\varepsilon_2,\cdots,\varepsilon_n)=(\varepsilon_1,\varepsilon_2,\cdots,\varepsilon_n)A^T$

**我们把矩阵 $A^T$ 称为线性变换在这一组基下的矩阵** 增加了这个转置符号
#### 投影与线性变换
向量在空间上的投影也是一个重要的代数学概念,设全空间$M$ 有两个子空间 $N_1,N_2$ 并且有 $N_1+N_2=M,N_{1}\cap N_{2}= \phi$  考虑$M$ 中向量 $z$ 的唯一分解 $z=x+y$ 并且 $x\in N_{1},y\in N_{2}$ 则线性变换
$$P_{N_1|N_2}z=x$$
就称为 $z$ 遵循 $N_2$ 在 $N_1$ 上的投影  其中投影的结果称为 $x$ 对应的矩阵称为投影矩阵.
#### 线性变换的矩阵的运算
我们给出下面的关于线性变换的运算和他的矩阵的联系
* 线性变换的和等同于对应矩阵的和
* 线性变换的积等同于对应矩阵的积
* 线性变换的数乘 等同于对应矩阵的数种
* 可逆的线性变换与可逆的矩阵对应，逆变换等同于矩阵求逆

定理：在有了线性变换的矩阵以后，我们可以容易计算变换前后的坐标情况，也就是定义反了过来
$$\begin{pmatrix}
 y_1\\
  y_2\\
 \vdots \\
 y_n
\end{pmatrix}=A\begin{pmatrix}
 x_1\\
  x_2\\
 \vdots \\
 x_n
\end{pmatrix}$$
其中 $y$ 是变换后的坐标 ；$x$ 是变换前的坐标；$A$是线性变换的矩阵
#### 线性变换的矩阵的基变换（相似）
前面得到的如此多的结论非常令人欣喜，不过我们还会面临一个非常现实的问题，目前的线性变换的矩阵和选择的基绑定，而不是和变换本身绑定，在基变换的情况下线性变换的矩阵会怎么变换，这就是本节研究的问题。

定理：设存在线性变换  他有两组基 $\varepsilon_{1},\varepsilon_{2},\cdots\varepsilon_{n},\eta_{1},\eta_{2},\cdots\eta_{n}$  变换在两组基下的矩阵分别为 $A,B$  两组基之间的过渡矩阵$\varepsilon_\to\eta$ 为 $X$ 那么有 $B=X^{-1}AX$

我们可以把这种矩阵关系抽象出来进行单独的研究

定义（**矩阵相似**）：设$A,B$ 是数域$P$ 上的两个矩阵，如果能找到 $n$ 阶可逆矩阵$X$使得$B=X^{-1}AX$  则称矩阵$A,B$ 相似。 **这是在等价，合同后提出的第三个性质**

根据前面的定义，我们容易给出：线性变换的矩阵在不同基下相似，相似矩阵可以看作不同基下面的同一个线性变换的矩阵。

我们给出一个重要例题，关于[高等代数2 矩阵和线性空间中的矩阵方幂](/blog/2023/06/17/advanced-algebra-matrices-linear-spaces-notes/) 的求解

计算
$$\begin{pmatrix}
  2&1 \\
  -1&0
\end{pmatrix}^k$$ 并且这个矩阵是某个线性变换在基$\varepsilon_{1},\varepsilon_{2}$ 下的矩阵，同时有
$$\begin{pmatrix}
  \eta_1 &  \eta_2
\end{pmatrix}=\begin{pmatrix}
  \varepsilon_{1}&\varepsilon_{2}
\end{pmatrix}\begin{pmatrix}
  1&1 \\
  -1&2
\end{pmatrix}$$

据此，我们可以计算线性变换在基$\eta$ 下的矩阵$B$有
$$\begin{pmatrix}
  1&1 \\
  -1&2
\end{pmatrix}^{-1}\begin{pmatrix}
  2&1 \\
  -1&0
\end{pmatrix}\begin{pmatrix}
  1&1 \\
  -1&2
\end{pmatrix}=\begin{pmatrix}
  1&1 \\
  1&0
\end{pmatrix}$$
那么我们就有
$$X^{-1}AX=B\to A=XBX^{-1}$$
所以
$$A^{k}=XB^{k}X^{-1}$$
而$B$的方幂是三角矩阵，很容易计算，我们顺序实现了降低次数，是**依靠相似矩阵实现的约分**
### 特征值和特征向量
#### 特征值与特征向量的定义
基会影响线性变换的矩阵，如何选取合适的基，让矩阵有最简的形式。
就是我们从这一节开始要讨论的内容。

定义：$\sigma$ 是 $V$ 上的线性变换，如果对于$P$ 中的数 $\lambda$  存在向量 $\xi$ 使得 $\sigma(\xi)=\lambda\xi$  则称 $\lambda$ 是这个线性变换的一个特征值， $\xi$是对应的特征向量

关于特征值与特征值的解释
* 如果$\xi$是特征向量，那么$k\xi$ 也是这个特征值对应的特征向量
* 同一个特征向量只对应一个特征值
* 矩阵的特征值与特征向量就是矩阵对应的线性变换的特征值与特征向量

下面我们来研究计算矩阵的特征值和特征向量

根据特征值与特征向量的定义，一定有
$$\lambda \begin{pmatrix}
x_1 \\
 x_2\\
 \vdots\\
x_n
\end{pmatrix}=A\begin{pmatrix}
x_1 \\
 x_2\\
 \vdots\\
x_n
\end{pmatrix}$$
也就是说
$$(\lambda E-A )\begin{pmatrix}
x_1 \\
 x_2\\
 \vdots\\
x_n
\end{pmatrix}=0$$
这本质上是一个齐次线性方程组，那么他有解的条件就是$|\lambda E-A|=0$

定义：$|\lambda E-A|$ 展开后的关于$\lambda$ 的多项式称为特征多项式，实际上，特征多项式的根就是特征值。

将解特征多项式得到的根带回到齐次方程组中解到的向量就是这个特征值对应的特征向量

#### 特征值与特征向量的计算与重数
对于数量矩阵 $kE$  特征多项式有
$$|\lambda E-kE|=0\Rightarrow(\lambda-k)^{n}=0$$
因此有$n$个特征值 $k$  我们称为特征值$k$ 的代数重数是 $n$

对于对角矩阵，特征多项式有
$$(\lambda-a_{11})(\lambda-a_{22})\cdots(\lambda-a_{nn})$$
也就是有特征值 $a_{ii}$  没有有重数不确定

对于三角矩阵，特征多项式有
$$(\lambda-a_{11})(\lambda-a_{22})\cdots(\lambda-a_{nn})$$
也就是有特征值 $a_{ii}$  没有有重数不确定

定义：对于一个给出特征值$\lambda$ 所有的特征向量加上零向量构成了一个子空间。其维数就是特征向量组极大无关组个数。 这个维数也称为几何重数。

不难发现，对于任意特征值，代数重数大于等于几何重数
#### 特征值与特征向量相关的重要定理
**两个重要性质**
* $A$的全体特征值的和为 $a_{11}+a_{22}+\cdots+a_{nn}$
* $A$的全体特征值的积为 $|A|$

定理：**相似的矩阵有相同的特征多项式**  这个定理终于让我们研究线性变换而可以脱离其基存在了，这就是找最简单基的抓手

**类谱映射**  下面是对等的结论
* $\lambda$ 是 $A$ 的特征值
* $\lambda^{-1}$ 是 $A^{-1}$ 的特征值
* $\frac{|A|}{\lambda}$ 是 $A^{\star}$ 的特征值

定理（**哈密顿凯莱定理**） ：记 $A$ 是一个矩阵 $f(\lambda)$ 是$A$的特征多项式 则有
$$f(A)=A^n+a_1A^{n-1}+\cdots+a_{n-1}A+a_nE=0$$

定理（**谱映射定理**）： 矩阵$A$ 的特征值是 $\lambda_i$  $f(x)\in P[x]$ 那么 $f(A)$ 的特征值是 $f(\lambda_i)$

#### 已知特征值与特征向量反求原矩阵的问题
对于已知特征值与特征向量反求原矩阵的问题，我们把问题分为几类

**已知全部的特征值与特征向量**
我们知道
$$X^{-1}AX=B$$
其中$X$是特征向量矩阵（数着排） $A$ 是原矩阵 $B$ 是特征值对角矩阵

**只知道部分特征向量，但是原矩阵$A$是对称矩阵**
对于对称矩阵，我们知道，他的特征向量相互正交，因此可以构造方程或者缺少的特征向量。

**只知道部分特征向量，无其他性质**
只能依靠不同特征值之间的特征向量是无关的来想办法化简了
### 对角矩阵
本节旨在于简化线性变换的矩阵，而对角矩阵应该算是最筒单的一种。什么样的矩阵相似于对角矩阵，怎么把它化为对角矩阵 这便是本节要讨论的。

定义： $\sigma$是$V$上的一个线性变换 若存在$V$中的一组基 使在这组基下的线性变换的矩阵是对角矩阵，则称它是可对角化的。

定理：对于可以对角化的矩阵，其对角线上的元素就是矩阵的特征值，这个对角矩阵除了顺序以外被完全确定。

定理：线性变换$\sigma$的矩阵可以对角化 等价于 他有 $n$ 个无关的特征向量

定理：线性变换$\sigma$属于不同特征值的特征向量是无关的

推论：如果在$n$维线性空间中，线性变换$A$的特征多项式有$n$个不同的根（$n$个不同的特征值）则$A$是可以对角化的

推论：由于复数域上的$n$次多项式一定有$n$个根，因此没有重根就意味着可以对角化

推论：没有$n$特征值也不一定不可以对角化，只需要$n$个特征向量是无关的就好了

推论：可以对角化意味着特征子空间维数是$n$

推论：互异的特征值之间的几何重数的和为 $n$

推论：各个特征子空间的直和是 $V$  如果只有一个特征子空间，那就是 $V$

### 线性变换的值域与核
定义：设$A$是一个线性变换。$A$的全体像组成的集合称为$A$的值域，记作$AV$
所有被$A$变身$0$向量的向量组成$A$的核 记作$A^{-1}(0)$   或者我们也可以使用符号$ker(\sigma) ~~N(\sigma)$

定理：线性变换$A$对于$V$的值域与核都是$V$的子空间

定义：我们将$AV$ 的维数称为线性变换的秩，$A^{-1}(0)$的维数称为零度

定理：设线性变换$A$对应的矩阵为$A$ 则有
* $A$的值域是原本的基向量被变换后的向量组生成的子空间
* 线性变换$A$的秩就是矩阵$A$的秩

对于求核空间的问题，我们只需要计算 $AX=0$ 解方程就可以了

定理：$A$的秩和$A$的零度的和为$n$

定理：如果有$A^2=A$ 则$A$ 相似于对角矩阵

### 不变子空间
定义：设$W$是$V$的子空间，如果对于任意$\xi\in W$ 都有 $A\xi\in W$ 则称 $W$ 是$A-$子空间，或者说$A$的不变子空间

定理：明显的
* $V$和$0$ 这两个平凡子空间是不变子空间
* $A$的值域与核 $AV$ 与 $A^{-1}(0)$   是不变子空间

定理：根据定义我们可以证明
* 如果$A,B$是可以交换的线性变换，那么$B$的值域与核是$A-$的不变子空间
* $f(A)$和$A$是可交换的，因此他们的值域与核互为不变子空间

定理：根据定义我们可以了解到
* 任何子空间都是数乘变换的不变子空间
* 特征向量自己构成了一个一维的不变子空间
* $A-$的不变子空间的交与和仍是不变子空间

定理：$A-$子空间的充要条件是对基向量组$A-$不变

定理：如果$V$可以分解为若干个$A-$子空间的直和 如
$$V=W_1\oplus W_2\oplus\cdots\oplus W_s$$
则有$V$中线性变换$A$的矩阵为
$$\begin{pmatrix}
  A_1&  0&\cdots   &0 \\
  0&  A_2&0  & \vdots\\
  \vdots&  0&  \ddots &\vdots \\
  0& \cdots  &  0&A_s
\end{pmatrix}$$

定理：特征变换本质上就是把$A$分解为了若干个特征向量的特征子空间的直和，我们把特征向量对应的特征子空间称为**根子空间**
### Jordan标准型的引入
前文已经提到了对角矩阵是化简的最简形式，但是要满足有$n$个无关特征向量才可以。 那么其它形式应该有什么样的最简形式呢？本节问题将在复数域上讨论。

定义：我们将如图所示的矩阵称为Jordan块
$$J(\lambda_0,k_0)=\begin{pmatrix}
  \lambda_0&  0&  0&0 \\
  1&  \lambda_0&  0&0 \\
  0&  1& \ddots  &0 \\
  0&  0&  1&\lambda_0
\end{pmatrix}_{k\times k}$$

定义：如下图所示的形式称为Jordan型矩阵
$$A=\begin{pmatrix}
  J(\lambda_1,k_1)&  0&0 \\
  0&  J(\lambda_i,k_i)&0 \\
  0& 0 &J(\lambda_s,k_s)
\end{pmatrix}$$
其中$k_i$可以是复数 也可以为1

定理：如果$A$是复数域$V$上一个线性变换，则$V$中一定存在一组基使得$A$在这一组基下的矩阵为Jordan型矩阵，我们称为线性变换的Jordan标准型。

等价叙述：任意$n$阶复矩阵$A$ 总和一个Jordan标准型相似，除开Jordan块的排列顺序，他由$A$完全确定。

至于如何求解Jordan标准型，我们将用本文“$lambda$矩阵”部分一整章来解决。这里不叙述了
### 最小多项式
定义：据哈密顿-凯莱定理 任意给定矩阵$A$ 总能找到多项式使得$f(A)=0$。显然$f(x)$ 不具唯一性（系数可以扩倍，最小多项式可能可以特征多项式的0因式），我们称次数最小且首一的多项式为对应矩阵的**最小多项式**。

引理：最小多项式具有唯一性

引理：设$g(x)$是$A$的最小多项式，则$f(x)$满足 $f(A)=0$ 等价于 $g(x)|f(x)$

定理：**矩阵的最小多项式一定是特征多项式的因式**

推论：相似矩阵有着一样的最小多项式，该命题逆命题不成立

引理：设 $A = \begin{pmatrix}A_1&0 \\0&A_2\end{pmatrix}$ 则$A$的最小多项式为$A_1,A_2$两个最小多项式的最小公倍式

引理：$k$ 阶Jordan块的最小多项式为 $(x-a)^k$

定理：一个矩阵可以对角化 等价于 最小多项式是$P$上互素的一次因式的乘积

推论：一个复矩阵可以对角化 等价于 最小多项式无重根

**研究最小多项式与对角最简型，Jordan最简型的联系是为了我们下一章研究Jordan标准型的求解的基础**
## $\lambda$矩阵
本章的全部铺垫都是为了最后的本文“Jordan标准型的求解”部分，研究怎么求Jordan标准型这个从上一章继承的问题是本章的核心
### $\lambda$矩阵的定义
定义：设$P$是一个数域，$\lambda$是一个文字，做多项式环的一个矩阵，如果他的元素是$P[\lambda]$的元素，则称矩阵为 $\lambda$矩阵。当$\lambda$ 是$P$ 中的一个数的时候，他就是我们之前研究的数字矩阵。

定义：$\lambda$矩阵的各种运算与性质都可以从[高等代数2 矩阵和线性空间中的矩阵运算](/blog/2023/06/17/advanced-algebra-matrices-linear-spaces-notes/)中继承，行列式从[高等代数1 代数学基础中的行列式](/blog/2023/03/17/advanced-algebra-foundations-notes/)继承，他的秩需要采用子式进行定义。

定义：$\lambda$矩阵可逆为 $A(\lambda)B(\lambda)=E=B(\lambda)A(\lambda)$    [高等代数2 矩阵和线性空间中的矩阵的逆](/blog/2023/06/17/advanced-algebra-matrices-linear-spaces-notes/)

定理：$\lambda$矩阵可逆的充要条件为 $|A(\lambda)|$  是一个非零的数，而不是一个含有$\lambda$ 的多项式

定义：$\lambda$矩阵的初等变换  [高等代数2 矩阵和线性空间中的初等变换与初等矩阵](/blog/2023/06/17/advanced-algebra-matrices-linear-spaces-notes/)
* 行（列）互换
* 行（列）乘非零常数 $c$
* 行（列）加上 另一行（列） 的 $\phi(\lambda)$ 倍 其中$\phi(\lambda)$最小是零次多项式
初等变换仍满足左（右）乘对应的初等矩阵，且初等矩阵一定可逆

定义：我们称两个$\lambda$矩阵等价 如果两个矩阵可以经由一系列初等变换得到
### $\lambda$矩阵的标准型
本节内容是为了证明，$\lambda$矩阵的等价标准型是一个对角矩阵，并找到这个等价标准型。

**引理（降次等价）**：设$\lambda$矩阵左上角元素为 $a_{11}(\lambda)\ne0$ 且 $A(\lambda)$ 中至少有一个元素不能被他除尽，则可以找到一个和$A(\lambda)$ 等价的 $B(\lambda)$ 满足 $b_{11}(\lambda)\ne0$ 且次数小于 $a_{11}(\lambda)$  寻找方法如下

**如果$A(\lambda)$的第一行（列）有一个元素不能被$a_{11}(\lambda)$ 除尽**

那么有

也就是用第$i$行（列）减去第一行（列）的$q(\lambda)$倍 然后将$i$行（列）和第一行（列）交换顺序，将余数$r(\lambda)$换上去

**如果$A(\lambda)$的第一行（列）没有一个元素不能被$a_{11}(\lambda)$ 除尽，而是元素$a_{ij}$  不能被$a_{11}(\lambda)$ 除尽**

那么我们进行下列变换
* 使用第一行$a_{11}$将位置$a_{i1}$化为0  *利用倍加的性质*
* 把第$i$行的1倍加到第1行
* 此时我们就把问题转换为了前面的情况，可以继续找等价形式了

定理（等价标准型存在性）任意的非零$\lambda$矩阵等价于
$$\begin{pmatrix}
 d_1(\lambda ) &0  & \cdots  &0 \\
  0&  d_2(\lambda )&0  & \vdots \\
  \vdots &  0&  \ddots & 0\\
  0& \cdots  &  0&d_n(\lambda )
\end{pmatrix}$$
其中 $d_i(\lambda)|d_{i+1}(\lambda)$  也就是越向下次数越高

这个定理的证明就是利用降次等价的引理，将左上角的元素化为一个是$A(\lambda)$中所有元素的因子的元素，然后将第一行列所有非$a_{11}$的位置化0，然后其研究$n-1$阶子阵，重复前面的步骤。
### 不变因子
不变因子研究$\lambda$矩阵等价标准型唯一性

定义：设$\lambda$矩阵$A(\lambda)$的秩为$r$ 对于 $1\le k\le r$  $A(\lambda)$一定有非零的$k$阶子式。所有的非零的$k$阶子式的首项系数为1的最大公因式$D_k$称为$A(\lambda)$的$k$阶行列式因子。

定理：明显的，对于秩为$r$的$\lambda$矩阵$A(\lambda)$，存在$r$个行列式因子

定理：对于等价的$\lambda$矩阵，他们有相同的行列式因子

定理：$\lambda$矩阵的等价标准型是唯一的

证明：容易知道 $D_{k}(\lambda)=d_{1}(\lambda)d_{2}(\lambda)\cdots d_{k}(\lambda)$  是所有对角线元素的积，那么容易知道
$$d_{1}(\lambda)=D_1(\lambda)\quad d_{2}(\lambda)=\frac{D_{2}(\lambda)}{D_1(\lambda)}\quad d_{r}(\lambda)=\frac{D_{r}(\lambda)}{D_{r-1}(\lambda)}$$
定义：我们称前面证明等价标准型是唯一的时候定义的$d_{i}(\lambda)$是$\lambda$矩阵的不变因子。

定理：不变因子，行列式因子，等价标准型相互确定

定理：矩阵的最小多项式，就是所有的不变因子中最后的那个（次数最高的那个）
### 矩阵相似的条件
本节是过渡章节，建立数字矩阵 和 $\lambda$矩阵 之间的联系

定理：数字矩阵 $A,B$ 是相似的 $\Leftrightarrow$ 其特征矩阵 $\lambda E -A,\lambda E -A$作为$\lambda$矩阵是等价的

定义：数字矩阵$A$对应的特征矩阵$\lambda E-A$ 的不变因子是数字矩阵的不变因子

推论：$A$和$A^T$是等价的
### 初等因子
这是我们铺垫的最后一步了，本节的研究在复数域上进行

定义：把矩阵$A$ 所有次数大于0的不变因子 分解为互不相同的首项为1的一次因式方幂的乘积。所有的一次因式方幂(相同的按出现次数计) 称为矩阵$A$的初等因子

我们用一个例子来说明

所有的不变因子为 $9$ 个 $1$   $(\lambda-1)^2$   $(\lambda-1)^2(\lambda+1)$  $(\lambda-1)^2(\lambda+1)(\lambda^2+1)^2$

那么初等因子为
$$(\lambda-1)^{2}~~(\lambda-1)^{2}~~(\lambda-1)^{2}~~~(\lambda+1)~~(\lambda+1)~~(\lambda-i)^{2}~~(\lambda+i)^{2}$$

初等因子是可以变回不变因子的

显然不变因子之间是有整除关系的，因此可以考虑每个相同的一次因式按照降幂排列为不变因子的个数（不够补1） 就可以的得到不变因子了

还是前面的例子
$$\begin{pmatrix}
  (\lambda-1)^{2}& (\lambda+1) &(\lambda-i)^{2}  & (\lambda+i)^{2}\\
  (\lambda-1)^{2}& (\lambda+1) & \vdots  & \vdots \\
  (\lambda-1)^{2}&  \vdots  & \vdots  &\vdots  \\
  \vdots &   \vdots &\vdots   &\vdots
\end{pmatrix}$$
初等因子就是每一行的积

定理：两个同阶复矩阵相似 等价于 他们有相同的初等因子

定理（一种初等因子求法）：想要求初等因子$A(\lambda)$  可直接把$A(\lambda)$ 化对角型，然后分解为一次因式的乘积，就是$A$全部的初等因子。
### Jordan标准型的求解
这是这章的核心

考虑一个Jordan块
$$J(\lambda_0,k_0)=\begin{pmatrix}
  \lambda_0&  0&  0&0 \\
  1&  \lambda_0&  0&0 \\
  0&  1& \ddots  &0 \\
  0&  0&  1&\lambda_0
\end{pmatrix}_{k\times k}$$
其初等因子 $(\lambda-\lambda_0)^k$

不难得到，一个Jordan型矩阵的初等因子
$$(\lambda-\lambda_{1})^{k_{1}}~~(\lambda-\lambda_{2})^{k_{2}}\ldots$$

定理：Jordan型矩阵被他的初等因子完全确定，除了Jordan块的顺序

定理：$A$是复数域上的一个线性变换，$V$中一定存在一组基使得$A$对应的矩阵为Jordan型矩阵，除了Jordan块的顺序完全确定

定理：矩阵可以对角化等价于所有初等因子全是1次的

定理：矩阵可以对角化等价于所有不变因子没有重根

至于什么样的相似可以得到这个Jordan型矩阵，我们不再研究
### 有理标准型
本节研究一种和Jordan标准型类似的标准型的定义，存在与唯一性，求法，也就是有理标准型。

定义：$d(\lambda)=\lambda^{n}+a_{1}\lambda^{n-1}+\cdots+a_{n}$ 称矩阵$A$是友矩阵如果其满足
$$A=\begin{pmatrix}
  0& 0 & 0 &-a_n \\
  1&  0& 0 &\vdots  \\
  0&  1&  \ddots &-a_2 \\
  0&  0&  1&-a_1
\end{pmatrix}$$

定义：称友矩阵组成的块对角矩阵称为有理标准型

定理：友矩阵$A$的不变因子为 大量的1 和 $d(\lambda)$

定理：有理标准型矩阵的不变因子为各个友矩阵的不变因子和1

定理：数域$P$上的$n$阶方阵$A$相似于唯一一个有理标准型，这个有理标准型被$A$的不变因子完全确定，包括顺序在内（根据不变因子的次数决定，低次在上）
## 欧式空间
在线性空间中，向量的远算只有加法和数乘 压度导致了角度，距离这些度量都无法使用 故本章将为线性空间添加内积运算，得到欧式空间。
### 欧式空间的定义与基本性质
定义：在线性空间$V$中定义二元实函数，称为内积，如果它下面的条件满足
* 对称性$(\alpha,\beta)=(\beta,\alpha)$
* 线性$(k\alpha,\beta)=k(\alpha,\beta)$  $(\alpha+\beta,v)=(\alpha+v)+(\beta,v)$
* 正定型 $(\alpha,\alpha)\geq0\quad\text{使当}\alpha=0\text{ 时 等号成立}$

**欧式空间本质上在线性空间中添加了新的运算，不改变原本的线性空间本身**

我们可以给出两个常用的欧式空间
* $R^n$空间 对于$\alpha=(a_1,a_2,...,a_n),\beta=(b_1,b_2,...,b_n)$ 定义 $(\alpha,\beta)=a_1b_1+a_2b_2+\cdots+a_nb_n$
* $[a,b]$上连续函数空间$c$  对于 $f(x),g(x)$ 定义 $(f(x),g(x))=\int_{a}^{b} f(x)g(x) dx$

定义：非负实数 $\sqrt{(\alpha,\alpha)}$  称为 $\alpha$ 的长度 写作 $|\alpha|$   明显的  $|k\alpha|=k|\alpha|$

定义：$|\alpha-\beta|$ 是两个向量之间的距离

定义：$\frac{\alpha}{|\alpha|}$ 的长度为1 ，我们把这样的向量称为单位向量

定义：$\cos<\alpha,\beta> = \frac{(\alpha,\beta)}{|\alpha||\beta|}$ 称为向量的夹角

定理：$|(\alpha,\beta)|\le|\alpha||\beta|$ 这称为Cauchy不等式  代入到刚才举例的空间有
* $|a_{1}b_{1}+\cdots+a_{n}b_{n}|\leq\sqrt{a_{1}^{2}+\cdots+a_{n}^{2}}\sqrt{b_{1}^{2}+\cdots+b_{n}^{2}}$
* $|\int_{a}^{b}f(x)g(x)dx|\leq\sqrt{\int_{a}^{b}f(x)^2dx}\sqrt{\int_{a}^{b}g(x)^2dx}$

我们还可以给出三角不等式有 $|\alpha+\beta|=|\alpha|+|\beta|$

定义：如果 $(\alpha,\beta)=0$  我们称这两个向量是正交的/垂直的

定理（勾股定理的推广）：$|\alpha_{1}+\ldots+\alpha_{m}|^{2}\le |\alpha_{1}|^{2}+|\alpha_{2}|^{2}+\ldots+|\alpha_{m}|^{2}$  当且仅当所有向量正交的时候等号成立

仿照这对线性变换的矩阵化，我们可以矩阵化内积化运算有

定义：取一组基$\varepsilon_i$   两个向量$X,Y$ 有 $X=x_1\varepsilon_1+x_2\varepsilon_2+\cdots+x_n\varepsilon_n$  $Y=y_1\varepsilon_1+y_2\varepsilon_2+\cdots+y_n\varepsilon_n$  那么有 $(\alpha,\beta)=X^TAY$  其中 $X,Y$分别是坐标列向量，$A$是某一矩阵，他满足
$$A=\begin{pmatrix}
  a_{11}&a_{12}  & \cdots &a_{1n} \\
  a_{21}&a_{22}  & \cdots &a_{2n} \\
  \vdots&\vdots  &\ddots  &\vdots \\
  a_{n1}&a_{n2}  & \cdots &a_{nn}
\end{pmatrix}$$
并且有 $a_{ij}=(\varepsilon_i,\varepsilon_j)$  那么容易知道，他是一个对称矩阵。**可以根据基的内积来计算内积的矩阵**

 容易计算得到，当发生基变换$(\eta_1,\eta_2\ldots,\eta_n)=(\varepsilon_1\ldots\varepsilon_n)C$的时候，内积的矩阵变为了 $C^TAC$  也就是我们在[高等代数2 矩阵和线性空间中的二次型](/blog/2023/06/17/advanced-algebra-matrices-linear-spaces-notes/)中介绍的合同变换。

结合这两章容易给出定理：度量矩阵是正定的，正定矩阵可以是度量矩阵

定理：欧式空间的子空间仍旧对原本的内积运算构成欧式空间
### 标准正交基
#### 标准正价基的定义
我们在[高等代数2 矩阵和线性空间中的线性空间](/blog/2023/06/17/advanced-algebra-matrices-linear-spaces-notes/)里面介绍了标准基，在定义的内积运算后，我们决定研究如何把一组普通的基变成标准的。

定义：一组非0的，两两正交的向量组称为正交向量组

定理：正交向量组是线性无关的向量组，他有成为一组基的潜质

定义：$n$维欧式空间中，$n$个两两正交的向量组称为正价基，单位向量组成的正交基称为标准正交基，标准正交基满足
$$(\varepsilon_i ,\varepsilon_i) = \left\{\begin{matrix}
 0 &i\ne j \\
  1&i=j
\end{matrix}\right.$$
也就是说，使用标准正价基的内积度量矩阵是单位矩阵，由于任意对称矩阵一定合同于单位矩阵，因此标准正价基一定存在

当我们使用标准正交基的时候，有
* 向量的坐标满足 $x_i=(x,\varepsilon_i)$
* 内积 $(\alpha,\beta)=x_1y_1,x_2y_2,\cdots,x_ny_n$

#### Schmidt正交化
定理：任何一个正交向量组均可以扩充为一组标准正交基

定理（Schmidt正交化）：任何一组基$\varepsilon_i$ 都可以找到标准正交基 $\eta_i$ 只需要按照下面的方式 进行正交化。

首先得到正交基$\xi_i$
* $\xi_1=\varepsilon_i$
* $\xi_{2}=\varepsilon_{2}-\frac{(\varepsilon_{2},\xi_{1})}{(\xi_{1},\xi_{1})}\xi_1$
* $\xi_{m+1}=\varepsilon_{m+1}-\frac{(\varepsilon_{m+1},\xi_{1})}{(\xi_{1},\xi_{1})}\xi_{1}-\frac{(\varepsilon_{m+1}\xi_{2})}{(\xi_{2},\xi_{2})}\xi_{2}-\cdots-\frac{(\varepsilon_{m+1}\xi_{n})}{(\xi_{n},\xi_{n})}\xi_{n}$

然后把正交基$\xi_i$  进行标准化就可以得到标准正交基 $\eta_i$
#### 正交矩阵
两组正交基之间的过渡满足$$(\eta_{1} \cdots \eta_n)=(\xi_1\cdots \xi_n)A$$
也就是说 $A$ 满足
$$a_{1i}a_{1j}+\cdots+a_{ni}a_{nj}=\begin{cases}1&i=j\\0&i\neq j&\end{cases}$$
这就是说 $AA^T=E$  或者 $A^{-1}=A^T$   我们把这样的矩阵称为正交矩阵
### 欧式空间的同构
这一节是对我们讨论[高等代数2 矩阵和线性空间中的线性空间](/blog/2023/06/17/advanced-algebra-matrices-linear-spaces-notes/)推广到内积空间上的收尾工作了

在欧式空间上，同构需要满足
* $\sigma(\alpha+\beta)=\sigma(\alpha)+\sigma(\beta)$
* $\sigma(kq)=k\sigma(q)$
* $(\sigma(\alpha),\sigma(\beta))=(\alpha,\beta)$

**欧式空间的同构映射一定是线性空间上的同构映射，相关性质仍旧成立**

定理：$n$维欧式空间与$R^n$同构

定理：欧式空间同构最本质的特征是其维数，等维的欧式空间一定同构
### 正交变换
从这一节开始，我们推广线性变换到欧式空间中

定义：如果在一个欧式空间 $V$ 中，他对应的线性空间上的线性变换 $A$ 满足 内积的不变性，则称其是欧式空间上的一个正交变换，也就是$(A\alpha,A\beta)=(\alpha,\beta)$

我们也可以从其他空间刻画欧式空间有
* 保持向量的长度不变：$|A\alpha|=|\alpha|$
* 保持向量距离不变：$d(A\alpha,A\beta)=d(\alpha,\beta)$
* 保持标准正交基：$\xi$ 是标准正交基，则$A\xi$ 是标准正交基
* $A$变换在标准正交基下的矩阵是正交矩阵

由于正交矩阵是可逆的，则有
* 正交变换是可逆的
* 正交变换的逆与积还是正交变换
* 正交变换是一个欧式空间的自同构

### 欧式空间的子空间
这是对子空间一节的到欧式空间上的扩展

定义：向量$\alpha$ 正交与空间 $V_1$ 当且仅当 他正交与空间 $V_1$的所有向量 记作$\alpha\bot V_1$

定义：空间$V_1$正交于空间$V_2$  当且仅当 $V_1$中的所有向量正交于空间$V_2$  记作$V_{1}\bot V_2$

定理：如果 $V_1,V_2,\cdots,V_n$ 两两正交 则 $V_1+V_2+\cdots+V_n$ 是直和

定义：如果$V_{1}\bot V_2$  $V_1+V_2=V$  则称$V_1,V_2$ 互为正交补空间 我们记$V_1$的正交补为 $V_1^{\bot}$

定理：下面的结论成立
* 正交补具有唯一性
* $rank(V)+rank(V^{\perp})=n$
* $(W^{\bot})^{\bot}=W$
*  $V_1^{\bot}$  恰好由所有与 $V_1$ 正交的向量构成
### 实对称矩阵的标准型
这一节是对我们相似标准型的推广，也就是 对角矩阵 和 Jordan 矩阵的内积空间的形式的思考。  所谓实对称矩阵，是因为所有的内积矩阵都是一个实对称矩阵，和研究内积矩阵不矛盾。

在 [高等代数2 矩阵和线性空间中的二次型](/blog/2023/06/17/advanced-algebra-matrices-linear-spaces-notes/) 一节中我们已经知道了：**所有的实对称矩阵合同于一个对角矩阵**  这就是矩阵的合同标准型 $C^TAC$

定理：任意实对称矩阵可以通过正交变换得到一个对角矩阵，也就是或对于实对称矩阵$B$ 其对角形式 $A$  有    $B=C^TAC$  或者  $B=C^{-1}AC$

引理：所有实对称矩阵的特征值为实数

引理：满足 $( A\alpha , \beta ) = (\alpha, A\beta )$ 或者说 $P^TA\alpha = \alpha\beta$的变换称为对称变换，在实对称矩阵上 线性变换是对称变联

定理（正交变换标准型）：设$A$是实对称矩阵，则$A$属于不同特征值的特征向量正交，并且我们可以按照如下形式找到其正交变换得到的对角型矩阵$B$和变换的矩阵$T$
1. 找到实对称矩阵$A$的特征值
2. 求解对应的特征向量，将每一个特征向量标准化
3. 将所有的特征向量竖着写成矩阵就是正交变换的矩阵$T$ 特征值构成的对角矩阵就是$B$  在写$T$ 和 $B$ 的时候要保证按照同一个顺序

这个定理用二次型表述为：实二次型可以通过正交线性替换为 $\lambda_1y_1^2+\cdots+\lambda_ny_n^2$  其中 $\lambda_i$ 就是特征多项式的根

**正交变换得到的标准型是唯一的，这就是他对合同变换做出的最重要改变，合同变换可以得到多个标准型，但正交变换的标准型除顺序以外唯一**

**实对称矩阵的秩对应的非零特征值的个数。正负惯性指数是正负特征值的个数**

### 酉空间
我们这里研究复数域上的欧式空间

定义：如果一个内积空间的内积运算在复数域上并且满足下面的性质，则称其为酉空间
* 共轭对称性 $(\alpha,\beta)=\overline{(\beta,\alpha)}$
* 线性 $(k\alpha,\beta)=k(\alpha,\beta)$ $(\alpha+\beta,v)=(\alpha,\gamma)+(\beta,v)$  其中 $k$ 是任意复数
* 正定性 $(\alpha,\alpha)=0\quad\text{当且仅当 }\alpha=0$

酉空间有下面的计算性质
* $(\alpha,k\beta)=-\bar{k}(\alpha,\beta)$
* $(\alpha+\beta,v)=(\alpha,\gamma)+(\beta,v)$
* 向量$\alpha$ 的长度为 $\sqrt{(\alpha,\alpha)}$
* 不定义夹角
* 柯西不等式 ：$|(\alpha,\beta)|\le|\alpha||\beta|$
* 三角不等式  $|\alpha+\beta|=|\alpha|+|\beta|$


定义：满足 $(A\alpha,A\beta)=(\alpha,\beta)$ 的酉空间上的变换称为酉变换

定义：酉变换在标准正交基下的矩阵是酉矩阵，设$A$是酉矩阵，则满足$A\bar{A^{T}}=\bar{A^{T}}A=E$
