---
title: "矩阵分析学习笔记"
title_en: "Matrix Analysis Notes"
date: 2024-12-15 00:42:23 +0800
categories: ["Mathematics", "Algebra & Matrix Theory"]
tags: ["Learning Notes", "Mathematics", "Matrix Analysis", "Linear Algebra"]
author: Hyacehila
excerpt: "一篇矩阵分析学习笔记，整理广义逆、特殊矩阵、特殊积、Kronecker 积和矩阵分析中的常用结论。"
excerpt_en: "A study note on matrix analysis, covering generalized inverses, special matrices, special products, Kronecker products, and common results in matrix analysis."
mathjax: true
hidden: true
permalink: '/blog/2024/12/15/matrix-analysis-notes/'
---
矩阵分析及其应用希望研究[矩阵论](/blog/2024/10/17/matrix-theory-notes/)中未能叙述完全的近现代代数理论中的矩阵分支，首先我们考虑研究本文“矩阵的广义逆”部分与本文“特殊积”部分和本文“特殊矩阵”部分

至于进一步的矩阵分析还研究什么内容，我们留给后续的课程研究
## 矩阵的广义逆
### 广义逆的基本概念
广义逆矩阵是通常逆矩阵的推广，这种推广的必要性，是线性方程组的求解问题的实际需要，设有线性方程组
$$Ax=b,$$
当$A$ 是$n$ 阶方阵，且 $detA\neq0$ 时，则方程组的解存在且惟一，并可写成
$$x=A^{-1}b.$$
但是，在许多实际问题中所遇到的矩阵$A$往往是奇异方阵或是任意的$m\times n$矩阵(一般$m\neq n$),显然不存在通常的逆矩阵$A^{-1}$.这就促使人们去想象能否推广逆矩阵的概念，引进某种具有普通逆矩阵类似性质的矩阵 $G$,使得其解仍可以表示为
$$x=Gb.$$

彭诺斯指出：对任意复数矩阵 $A_{m\times n}$,如果存在复矩阵 $G_{n\times m}$,满足如下条件
$$\begin{aligned}AGA=&A,\\GAG=&G,\\(GA)^{\mathrm{H}}=&GA\:,\\(AG)^{\mathrm{H}}=&AG\:,\\\end{aligned}$$
则称$G$为$A$的一个穆尔-彭诺斯广义逆，并把上面 4 个方程叫做穆尔-彭诺斯方程，简称 M-P方程.

由于这四个方程都有一定的优秀性质，满足部分也是很好的，因此我们给出定义有

定义：设 $A\in\mathbb{C}^{m\times n}$,若有某个 $G\in\mathbb{C}^n\times m$,满足 M-P 方程中的全部或一部分，则称 $G$ 为$A$ 的广义逆矩阵，简称为广义逆.

我们知道，逆矩阵可以只满足部分性质，因此实际上我们可以给出15种广义逆（$\mathrm{C}_4^1+\mathrm{C}_4^2+\mathrm{C}_4^3+\mathrm{C}_4^4=15$） 但是只有部分比较常用，分别为
* 满足第一个方程 记做$A\{1\}$  称为减号逆 $A^-$
* 满足方程1，2 记做$A\{1,2\}$ 称为自反减号逆 $A^-_r$
* 满足方程1，3记做$A\{1,3\}$  称为最小范数广义逆 $A^-_m$
* 满足方程1，4 记做$A\{1,4\}$ 称为最小二乘广义逆 $A^-_l$
* 满足方程1，2，3，4 记做$A\{1,2,3,4\}$ 称为加号逆，伪逆 穆尔-彭诺斯广义逆 $A^+$

只有加号逆$A\{1,2,3,4\}$ 是确定的，其余广义逆均不唯一确定，我们在后面的叙述中会说明这一点。
### 减号逆 $A^-$
定义：设有$m\times n$实矩阵$A(m\leqslant n$,当$m>n$时，可讨论$A^{\mathrm{T}}).$若有一个$n\times m$实矩阵(记为$A^-$)存在，使下式成立，则称$A^-$为$A$的减号逆或$g$逆：
$$AA^-A=A.$$
当$A^{-1}$存在时，显然$A^-1$满足上式，可见减号逆 $A^-$ 是普通逆矩阵 $A^-1$ 的推广；另外，由$AA^-A=A$得
$$(AA^-A)^\mathrm{T}=A^\mathrm{T},\quad\text{即}\quad A^\mathrm{T}(A^-)^\mathrm{T}A^\mathrm{T}=A^\mathrm{T}.$$
可见，当$A^-$为$A$ 的一个减号逆时，$(A^-)^\mathrm{T}$就是$A^\mathrm{T}$ 的一个减号逆.

注意：减号逆不唯一确定 ，例如
$$\boldsymbol{A}=\begin{bmatrix}1&0\\1&0\\1&0\end{bmatrix},\boldsymbol{B}=\begin{bmatrix}1&0&0\\0&1&0\end{bmatrix},\boldsymbol{C}=\begin{bmatrix}1&0&0\\0&0&1\end{bmatrix}$$
此时 $B,C$ 均是 $A$ 的减号逆

下面，我们讨论证明减号逆的存在性，也就是去寻找任意矩阵的减号逆

定理：$\text{任给 }m\times n\text{ 矩阵 }A,\text{那么减号逆 }A^--\text{定存在 },\text{但不惟一}.$

如果 $rankA=0$ 那么一定有 任意$X\in R^{n\times m}$  都有 $0X0=0$ 因此减号逆存在且不唯一

如果 $rankA\neq0$  则 一定存在满秩的$m$ 阶矩阵$P$ 和 满秩的$n$ 阶矩阵$Q$  使得
$$PAQ=\begin{bmatrix}I_r&0\\0&0\end{bmatrix}=B\in\mathbb{R}^{m\times n}$$
**实际上就是初等变换为单位矩阵，$P,Q$都是对应初等行列变换的矩阵**

根据一个此处没有给出的性质 有
$$\boldsymbol{B}^-=\begin{bmatrix}\boldsymbol{I}_r&&\star\\\star&&\star\end{bmatrix}\quad(\star\text{可任意选取}).$$

再根据一个此处没有给出的性质 有
$$A^-=Q{\begin{bmatrix}I_r&&\star\\\\\star&&\star\end{bmatrix}}P$$
由于$\star$的任意性，减号逆存在但不唯一，证毕

我们给出一个计算$P,Q$的示意，$A$是2行3列矩阵，通过初等行列变换让左上角的$2\times3$矩阵为左上角的$I_2$，其余全为0即可
$$\begin{bmatrix}
  A& I_2\\
  I_3&0
\end{bmatrix}=\begin{bmatrix}1&-1&2&1&0\\2&2&3&0&1\\1&0&0&\\0&1&0&\\0&0&1&\end{bmatrix}$$
定理：$\quad\mathrm{rank}A^-\geqslant\mathrm{rank}A.$

### 自反减号逆 $A^-_r$
普通的逆矩阵有着自反的性质，也就是$(A^{-1})^{-1}=A$ 但是一般的减号逆不满足这一点，例如
$$\mathbf{A}=\begin{bmatrix}1&0\\1&0\\1&0\end{bmatrix},\quad\mathbf{A}^-=\begin{bmatrix}1&0&0\\0&1&0\end{bmatrix}$$
容易验证 $AA^-A=A.$ 一侧的减号逆成立，但是
$$A^-AA^-=\begin{bmatrix}1&0&0\\1&0&0\end{bmatrix}\neq A^-$$
也就是说 $(A^{-1})^{-1}=A$ 不成立，因此我们需要对减号逆的概念进行限制，使其满足自反的性质，后面我们主要计算的也是自反减号逆

定义：对于一个$m\times n$实矩阵$A$,使
$$AGA= A\text{及}GAG= G$$
同时成立的$n\times m$实矩阵$G$,称为是$A$的一个自反减号逆

下面我们来研究自反减号逆的计算方法，首先我们需要引入左右逆的概念

定义：设 $A\in\mathbb{R}^m\times n$,若有 $G\in\mathbb{R}^n\times m$,使得
$$AG=I\quad\text{或}\quad GA=I,$$
则称$G$ 为$A$ 的右逆(或左逆),记为 $A_{\mathbb{R}}^{-1}($或 $A_{\mathbb{L}}^{-1})$,即$AA_{\mathbb{R} }^{- 1}= I$ 或 $A_{\mathbb{L} }^{- 1}A= I.$

在一般情况下$,A_{\mathbb{R}}^{-1}\neq A_{\mathbb{L}}^{-1}$.若 $A_{\mathbb{R}}^{-1}=A_{\mathbb{L}}^{-1}$,则 $A^{-1}$存在，且 $A^{-1}=A_{\mathbb{R}}^{-1}=A_{\mathbb{L}}^{-1}.$

定理：设 A 是行最大秩(行满秩)的$m\times n$实矩阵($m\leqslant n$),则必存在$A$的右逆，
$$A_{\mathrm{R}}^{-1}=A^{\mathrm{T}}(AA^{\mathrm{T}})^{-1}\:;$$
同理：设$\boldsymbol{A}$是列最大秩(列满秩)的$n\times m$实矩阵($m\geqslant n)$,则必存在${A}$的左逆，
$$A_{\mathrm{L}}^{-1}=(A^{\mathrm{T}}A)^{-1}A^{\mathrm{T}}.$$

**从定理可以看出，只有$m=n$并且$A$满秩的时候，左逆和右逆才同时存在并且相等，等于其逆矩阵$A^{-1}$**

定理：根据前定理计算出的左逆和右逆满足下面的性质
* 满足方程1  $AA_{\mathrm{R}}^{-1}A=A\quad(AA_{\mathrm{L}}^{-1}A=A)$
* 满足方程2 $A_\mathrm{R}^{-1}AA_\mathrm{R}^{-1}=A_\mathrm{R}^{-1}\quad(A_\mathrm{L}^{-1}AA_\mathrm{L}^{-1}=A_\mathrm{L}^{-1})$
* 满足方程3 $(A_{\mathrm{R}}^{-1}A)^{\mathrm{T}}=A_{\mathrm{R}}^{-1}A\quad(A_{\mathrm{L}}^{-1}A)^{\mathrm{T}}=A_{\mathrm{L}}^{-1}A$
* 满足方程4 $(AA_{\mathrm{R}}^{-1})^{\mathrm{T}}=AA_{\mathrm{R}}^{-1}\quad(AA_{\mathrm{L}}^{-1})^{\mathrm{T}}=AA_{\mathrm{L}}^{-1}$

也就是说，对于行或者列满秩的矩阵，按照前面计算方法给出的左逆或者右逆，不仅是减号逆，还是自反减号逆，最小范数广义逆，最小二乘广义逆，加号逆。

下面介绍自反减号逆的计算方法，这种方法具备普适性，继承自本文“减号逆 $A -$”部分，并且将很好的扩展到后面的内容中。

当$A$是行或者列满秩的矩阵的时候，使用左右逆即可，下面只讨论不行列满秩的情况，此时一定能找到
$$\mathbf{PAQ}=\begin{bmatrix}\mathbf{I}_r&\mathbf{0}\\\mathbf{0}&\mathbf{0}\end{bmatrix}$$
这等价于
$$\mathbf{A}=\mathbf{P}^{-1}\begin{bmatrix}\mathbf{I}_r&\mathbf{0}\\\mathbf{0}&\mathbf{0}\end{bmatrix}\mathbf{Q}^{-1}=\mathbf{P}^{-1}\begin{bmatrix}\mathbf{I}_r\\\mathbf{0}\end{bmatrix}(\mathbf{I}_r\mathbf{0})\mathbf{Q}^{-1}$$
令
$$B=P^{-1}\begin{bmatrix}I_r\\0\end{bmatrix},\quad C=(I_r0)Q^{-1}$$
那么有（**这其实就是[矩阵论中的最大秩分解](/blog/2024/10/17/matrix-theory-notes/)，不过叙述语言有改变**）
$$A=BC$$
计算
$$B_\mathrm{L}^{-1}=(B^\mathrm{T}B)^{-1}B^\mathrm{T},\quad C_\mathrm{R}^{-1}=C^\mathrm{T}(CC^\mathrm{T})^{-1}$$
因此有
$$A_\mathrm{r}^-=C_\mathrm{R}^{-1}B_\mathrm{L}^{-1}$$
经过验证，其满足 M-P 方程中的1，2 这就是自反减号逆

### 最小范数广义逆 $A^-_m$
定义：设 $A\in\mathbb{R}^{m\times n}\left(m\leqslant n\right)$,如果有一个 $n\times m$ 阶矩阵 $G$,满足
$$AGA=A\quad\text{及}\quad(GA)^\mathrm{T}=GA\:,$$
则称$G$为$A$的一个最小范数广义逆，记为$A_\mathrm{m}^-.$

最小范数广义逆 $A^-_m$有下面的计算方法
* 当矩阵$A$行或列满秩的时候，使用左右逆即可（前面已经证明）
* 当矩阵$A$不满足行或列满秩的时候，使用最大秩分解得到$A=BC$ 则有$$A_{\mathrm{m}}^{-}=C_{\mathrm{R}}^{-1}B_{\mathrm{L}}^{-}$$
也就是说计算方法和本文“自反减号逆 $A -_r$”部分完全一致

定理：我们还可以给出一个更为简单的公式计算最小范数广义逆 $A^-_m$ 有
$$A_{\mathrm{m}}^{-}=A^{\mathrm{T}}(AA^{\mathrm{T}})$$


### 最小二乘广义逆 $A^-_l$
定义：设$A\in\mathbb{R}^{m\times n}\left(m\leqslant n\right)$,若有一个$n\times m$阶矩阵$G$满足$AGA= A$ 及 $( AG) ^{\mathrm{T} }= AG$ 则称$G$ 为$A$ 的一个最小二乘广义逆，记为 $A_{\mathrm{l}}^{-}.$

最小范数广义逆 $A^-_m$有下面的计算方法
* 当矩阵$A$行或列满秩的时候，使用左右逆即可（前面已经证明）
* 当矩阵$A$不满足行或列满秩的时候，使用最大秩分解得到$A=BC$ 则有$$A_{\mathrm{m}}^{-}=C_{\mathrm{R}}^{-1}B_{\mathrm{L}}^{-}$$
也就是说计算方法和本文“自反减号逆 $A -_r$”部分完全一致

定理：我们还可以给出一个更为简单的公式计算最小二乘广义逆 $A^-_l$有
$$A_1^-=(A^\mathrm{T}A)^-A^\mathrm{T}.$$

### 加号逆$A^+$
前面我们对减号逆$A^{-}$加以不同的限制，得出减号逆的具有不同性质的减号逆，如自反广义逆$A_\mathrm{r}^-$、最小范数广义逆$A_\mathrm{m}^-$、最小二乘广义逆$A_\mathrm{l}^-$等.

其实，还有一类更特殊也更为重要广义逆，这就是将要介绍的加号逆$A^+$.它的实质是在减号逆的条件 $AGA=A$ 的基础上用上述所有条件同时加以限制.用这样的方式得出的$A^+$,不仅在应用上特别重要，而且有很多有趣的性质.

定义：设$A\in\mathbb{R}^{m\times n}\left(m\leqslant n\right)$,若有一个$n\times m$阶矩阵$G$同时满足
$$\begin{aligned}AGA=&A,\\GAG=&G,\\(GA)^{\mathrm{H}}=&GA\:,\\(AG)^{\mathrm{H}}=&AG\:,\\\end{aligned}$$
则称$G$为$A$的一个穆尔-彭诺斯广义逆，或者称为加号逆，伪逆，记为$A^+$

从定义可以看出，加号逆和逆一样，两个矩阵处于完全相同的地位，也就是
$$(A^+)^+=A$$

定理：我们可以用如下方式计算加号逆，如果$A=BC$是最大秩分解，则有
$$X=C^\mathrm{T}(CC^\mathrm{T})^{-1}(B^\mathrm{T}B)^{-1}B^\mathrm{T}$$
是$A$的加号逆；当然对于行或列满秩的情况，左右逆仍旧可用；对于本文“自反减号逆 $A -_r$”部分给出的不满秩的分解方法仍旧可用；各方法给出的结果一致。

定理：对于任意$A\in R^{m\times n}$ 其加号逆 $A^+$ 存在且唯一

推论：当$A$是$n$阶满秩方阵时，也就是$A^{-1}$普通逆存在，那么有
$$A^+=A^{-1}=A^-$$

定理：加号逆$A^+$ 有下面的性质
* $(A^{\mathrm{T}})^{+}=(A^{+})^{\mathrm{T}}$
* ${A}^+=({A}^{T}{A})^+{A}^{T}={A}^{T}({A}{A}^{T})^+$
* $(A^\mathrm{T}A)^+=A^+(A^\mathrm{T})^+$
* $\mathrm{rank}A=\mathrm{rank}A^+=\mathrm{rank}A^+A=\mathrm{rank}AA^+$


## 特殊矩阵
特殊矩阵希望研究如对角矩阵，三角矩阵，对称矩阵，这些具有特殊的性质的矩阵。那些已经在高等代数与[矩阵论](/blog/2024/10/17/matrix-theory-notes/)研究过的内容在这里不会重复叙述。我们这里主要研究的矩阵包括非负矩阵，随机矩阵，M与H矩阵等。

### 非负矩阵
在非常多的应用领域，经常出现元素都是非负实数的矩阵。在数学上我们统称其为非负矩阵，他的基本特征已经是矩阵论中不可缺少的一部分。本节就来讨论非负矩阵的性质以及其衍生。
#### 非负矩阵与正矩阵
定义：设 $A=(a_{ij})\in\mathbb{R}^{m\times n}$,如果
$$a_{ij}\geqslant0\:,\quad i=1\:,\cdots,m\:;\:j=1\:,\cdots,n\:,$$
即$A$的所有元素是非负的，则称$A$为非负矩阵，记作$A\geqslant0;$若式中严格不等号成立，即$a_{ij}>0$ ($i=1,\cdots,m;j=1,\cdots,n$),则称 $A$ 为正矩阵，记为$\boldsymbol{A}>0.$

设 $A,{B}\in{R}^{m\times n}$,如果成立 $A-{B\geqslant}0$,则记作 $A\geqslant{B};$   如果成立 $A-{B>}0$,则记作 $A>{B}.$

对于任意的 ${A}=(a_{ij})\in\mathbb{C}^{m\times n}$,引进记号
$$\mid A\mid=(\mid a_{ij}\mid),$$
即表示以 $a_{ij}$,之模$|a_{ij}|$为元素所得的非负矩阵；

特别地，当 $x=(x_1,\cdots,x_n)^{\mathrm{T}}\in\mathbb{C}^n$时，$| \boldsymbol{x}| = ($ | $x_1$ | $, \cdots$ , | $x_n$ | $) ^\mathrm{T}$ 表示一个非负向量.

定理：非负矩阵容易给出下面的性质 $A,B,C,D\in\mathbb{C}^{m\times n}$ 则
* $|A|\geqslant0,\text{并且}|A|=0\text{ 当且仅当 }A=0$
* $\text{对任意复数 }\alpha,\text{有}\mid\alpha A\mid=\mid\alpha\mid\mid A\mid$
* $|A+B|\leqslant|A|+|B|$
* $\text{若 }A\geqslant0,B\geqslant0,a,b\text{ 是非负实数,则 }aA+bB\geqslant0$
* $\text{若 }A\geqslant B,\text{且 }C\geqslant D,\text{则 }A+C\geqslant B+D$
* $\text{若 }A\geqslant B,\text{且 }B\geqslant C,\text{则 }A\geqslant C$
* 一般的，$A\ge0$ 和 $A\ne0$ 不能推出 $A>0$


定理：非负矩阵容易给出下面的性质 $A,B,C,D\in\mathbb{C}^{m\times n},x\in C^n$  则
* $|Ax|\leqslant|A|\mid x|$
* $|AB|\leqslant|A|\mid B|$
* $\text{对任意正整数 }m,\text{有}\mid A^m\mid\leqslant\mid A\mid^m$
* $\text{若 }0\leqslant A\leqslant B,0\leqslant C\leqslant D,\text{则 }0\leqslant AC\leqslant BD$
* $\text{若 }0\leqslant A\leqslant B,\text{对任意正整数 }m,\text{有 }0\leqslant A^m\leqslant B^m$
* $\text{若 }A\geqslant0(A>0),\text{对任意正整数 }m,A^m\geqslant0(A^m>0)$
* $若A>0,x\geqslant0且x\neq0,则Ax>0$
* $\text{若}|A|\leqslant B,\text{则}\parallel A\parallel_2\leqslant\parallel\mid A\mid\parallel_2\leqslant\parallel B\parallel_2$

定理 (谱半径的单调性)：设 $A,B\in\mathbb{C}^{n\times n}$,若$|A|\leqslant\boldsymbol{B}$,则
$$\rho(A)\leqslant\rho(\mid A\mid)\leqslant\rho(B).$$
我们容易给出两个关于谱半径的推论
* $\text{设 }A,B\in\mathbb{R}^{n\times n},\text{若 }0\leqslant A\leqslant B,\text{则 }\rho(A)\leqslant\rho(B).$
* $\text{设 }A\in\mathbb{R}^{n\times n},\text{若 }A\geqslant0,A^{(k)}\text{是 }A\text{ 的任一主子矩阵},\text{则 }\rho(A^{(k)})\leqslant\rho(A).$  $\text{ 特别地 },\max_{1\leq i\leq n}\langle a_{ii}\rangle\leqslant\rho(A).$

 定理 (佩龙定理，Perron建立的正矩阵特征值与特征向量的性质的定理)： 设 $A\in\mathbb{R}^{n\times n}$,且 $\rho(A)$为其谱半径，若 $A>0($正矩阵),则
* $\rho(\boldsymbol{A})为\boldsymbol{A}$的正特征值，其对应的一个特征向量 $y\in\mathbb{R}^n$必为正向量；
* 对 $A$ 的任何其他特征值 $\lambda$,都有$|\lambda|<\rho(A);$
* $\rho(A)$是 $A$ 的单特征值.

推论：$\text{正矩阵 }\mathbf{A}\text{ 的“模等于 }\rho(\mathbf{A})\text{”的特征值是惟一的}$

我们可以再给出两个重要的定理，他们也有着很好的应用效果

定理：设 $A=(a_{ij})_{n\times n},\boldsymbol{B}=(b_{ij})_{n\times n}\in\mathbb{R}^{n\times n}$为非负矩阵$|a_i,|\leqslant b_{ij},i,j=1,2,...,n$,则
$$\lambda(A)\subset\bigcup\limits_{i=1}^n\langle z\in\mathbb{C}\big|\mid z-a_{ii}\:|\leqslant\rho(B)-b_{ii}\:\rangle $$

定理：设 $A\in\mathbb{R}^n\times n$,如果 $A>0,x$ 是$A$ 的对应于特征值$\rho(A)$的正特征向量，又 $y$是$A^{\mathrm{T}}$的对应于特征值 $\rho(A)$的任一正特征向量，则
$$\lim_{m\to\infty}[\rho(A)^{-1}A]^m=(y^\mathrm{T}x)^{-1}xy^\mathrm{T}.$$
#### 不可约非负矩阵
下面我们继续**推广Penno定理到一类更加广义的矩阵**，他还属于非负矩阵，但是没有正矩阵这么简单了

在线性代数中，我们知道要对调矩阵 $A$ 的第$i,j$两行(列),相当于将$A$左(右)乘对应的对调矩阵$I_{i,j}$
$$\boldsymbol{I}_{i,j}=\begin{pmatrix}1\\&\ddots\\&&0&\cdots&1\\&&\vdots&\ddots&\vdots\\&&1&\cdots&0\\&&&&&\ddots\\&&&&&&1\end{pmatrix}$$
我们把一系列对调矩阵的积$P$称为置换矩阵（或排列矩阵），明显的有$P^{-1}=P^T$

定义 (可约与不可约矩阵) :设 $A\in\mathbb{R}^{n\times n}(n\geqslant2)$,若存在 $n$ 阶置换矩阵 $P$,使
$$\boldsymbol{PAP}^\mathrm{T}\:=\:\left(\begin{array}{cc}\boldsymbol{A}_{11}&\boldsymbol{A}_{12}\\\boldsymbol{0}&\boldsymbol{A}_{22}\end{array}\right),$$
其中$A_{11}$为$r$阶方阵$,A_{22}$为$n-r$阶方阵( l$\leqslant r<n$),则称$A$为可约(可分)矩阵，否则称$A$为不可约矩阵.  **实际上就是根据多次对调能否产生块三角**

明显的，如果所有元素都非0，那么一定不可约

可约的概念来源于线性方程组的求解问题.一个线性方程组的系数矩阵是可约的，表明该方程组可通过适当调整方程和未知数的次序，化为两个低阶的方程组来求解.即如果线性方程组
$$Ax=b$$
的系数矩阵$A$可约时，则可找到置换矩阵$P$使$A$呈
$$\boldsymbol{PAP}^\mathrm{T}\:=\:\left(\begin{matrix}\boldsymbol{A}_{11}&\boldsymbol{A}_{12}\\\boldsymbol{0}&\boldsymbol{A}_{22}\end{matrix}\right).$$
于是原方程组可化为
$$PAP^{\mathrm{T}}(Px)=Pb.$$
依次记$y=px=(\mathbf{y}_1^\mathrm{T},\mathbf{y}_2^\mathrm{T})^\mathrm{T}$和$\hat{\boldsymbol{B}}=Pb=(\hat{\boldsymbol{b}}_1^\mathrm{T},\hat{\boldsymbol{b}}_2^\mathrm{T})^\mathrm{T}$,就有
$$\begin{cases}\boldsymbol{A}_{11}\boldsymbol{y}_1+\boldsymbol{A}_{12}\boldsymbol{y}_2=\boldsymbol{\hat{b}}_1,\\\boldsymbol{A}_{22}\boldsymbol{y}_2=\boldsymbol{\hat{b}}_2.\end{cases}$$
于是方程组化为两个独立的低阶方程组，比直接解原方程组要方便、简单.
同样 , $A$ 的特征多项式也化为两个低阶矩阵的特征多项式的乘积.

定理（判断是否可约）：设$A\in R^{n\times n}$ 则
* $A为不可约矩阵的充分必要条件是A^T为不可约矩阵$
* $如果A是不可约非负矩阵,B是n阶非负矩阵,则A+B是不可约非负矩阵$
* $n(\geqslant2)\text{阶非负矩阵 A 不可约的充分必要条件是存在正整数 }s\leqslant n-1$ 使得$$(I+A)^s>0$$
定理 (佩龙-弗罗贝尼乌斯定理) ：设 $A\in\mathbb{R}^{n\times n}$是不可约非负矩阵，则
* $A$有一正实特征值恰等于它的谱半径 $\rho(A)$,并且存在正向量 $x\in\mathbb{R}^n$,使得$$Ax= \rho ( A) x$$

* $\rho(A)$是 $A$ 的单特征值；
* 当$A$的任意元素(一个或多个)增加时$,\rho(A)$增加.

值得提出的是，对于一般不可约非负矩阵$A$,佩龙-弗罗贝尼乌斯定理并不能保证$A$的“模等于 $\rho(A)”$的特征值是惟一的，Penno定理的这个推论无法推广

我们仍旧给出一些很有价值的定理在本节的结尾

----

定理：设 $\boldsymbol{A}=(a_{ij})_{n\times n}$为不可约非负矩阵，则或者
$$\sum_{j=1}^na_{ij}\:=\:\rho(A\:)\:,\quad i\:=\:1\:,2\:,\cdots,n\:,$$
或者
$$\min_{1\leqslant i\leqslant n}\sum_{j\:=\:1}^na_{ij}\:<\rho(A)<\max_{1\leqslant i\leqslant n}\sum_{j\:=\:1}^na_{ij}\:.$$
-----

推论：$A$为不可约非负矩阵，则对任意给定的正向量$x=(x_1,x_2,\cdots,x_n)^{\mathrm{T}}$,或者有
$$\frac{1}{x_i}\sum_{j=1}^na_{ij}x_j=\rho(A)\:,\quad i=1,2,\cdots,n$$
或者有
$$\min_{1\leqslant i\leqslant n}\biggl(\frac{1}{x_{i}}\sum_{j=1}^{n}a_{ij}x_{j}\biggr)<\rho(A)<\max_{1\leqslant i\leqslant n}\biggl(\frac{1}{x_{i}}\sum_{j=1}^{n}a_{ij}x_{j}\biggr).$$
----


#### 素矩阵与循环矩阵
现转到非负矩阵进一步的分类问题上.为此，引进一类介于不可约非负矩阵与正矩阵之间的矩阵——素矩阵与循环矩阵的概念，素矩阵有多种不同的定义方式，这里采用按谱半径的重数来定义，另外的方式作为性质.

定义：设$A$是$n$阶非负矩阵，且有$m$个特征值的模均等于谱半径$\rho(\boldsymbol{A})$,则当$m=1$ 时，就称方阵$A$为素矩阵(或本原矩阵);当$m>1$时，就称$A$是循环矩阵(或非素矩阵).$m$ 统称为A 的非素性指标。

定理：设$A,B$均为$n$阶非负矩阵，并且$A$是素矩阵，则
* $A^{\mathrm{T}}$ 也是素矩阵；
* 对任一正整数$k,A^k$也是素矩阵；
* $A+ B$ 也是素矩阵.

定理 ：非负矩阵$A$是素矩阵(本原矩阵)的充分必要条件，是存在某个正整数 k, 使得$A^k>0.$

**Penno定理以及其推论在素矩阵上仍旧成立，事实上正矩阵就是一种特殊的素矩阵**  至于佩龙-弗罗贝尼乌斯定理 我们只能给出下面的定理

定理：设 $A\in\mathbb{R}^{n\times n}$为非负矩阵，则有结论：
* $\rho(\boldsymbol{A})$是$\boldsymbol{A}$的特征值，且属于 $\rho(\boldsymbol{A})$的特征向量可取作非负的，即存在不为零的非负向量 $x$,使得 $Ax=\rho(A)x($注意，这里 $\rho(A)$和 $x$ 不一定是正的);
* $\boldsymbol{A}$的特征值可分成若干组，每组中的特征值模都相等，而且“均匀”地分布在以原点为圆心的某一圆周上(注意，这里$A.$的所有特征值的模都不超过即小于等于$\rho(A)$ )
### 随机矩阵
这里介绍另一类非常重要的矩阵——随机矩阵，我们研究其性质以及一些应用的背景

定义：设$A=(a_{i,j})\in\mathbb{R}^{n\times n}$是非负矩阵，如果$A$的每一行上的元素之和都等于 1,即
$$\sum_{j\:=\:1}^na_{ij}\:=\:1\:,\quad i\:=\:1\:,2\:,\cdots,n\:,$$
则称$A$为随机矩阵；如果$A$还满足
$$\sum_{i\:=\:1}^na_{ij}\:=\:1\:,\quad j\:=\:1\:,2\:,\cdots,n\:,$$
则称 $A$ 为双随机矩阵.

**$A$之所以称为随机矩阵，是因为$A$的每一行可以看成有$n$个点的样本空间上的离散概念分布.这样的矩阵常常出现在城市间的人口流动模型、马尔可夫(Markov)链的研究及经济学和运筹学等领域的各种各样的数学模型问题中.**

定理：下面我们简单的讨论一些随机矩阵独有的性质有
* 设 $A\in\mathbb{R}^{n\times n}$是随机矩阵，则有$\rho(A)=1.$
* $n$阶非负矩阵A 是随机矩阵的充分必要条件是$x=(1,\cdots,1)^{\mathrm{T}}\in\mathbb{R}^{n}$为$A$ 对应于特征值 1 的特征向量，即$Ax=x.$
* 同阶随机矩阵的积还是随机矩阵
* 设 $n$ 阶非负矩阵 $A$ 的谱半径 $\rho(A)>0$,且有 $x=(x_1,\cdots,x_n)^{\mathrm{T}}>0$,则矩阵$A$ 能相似于数  $\rho ( A)$与某个随机矩阵 $P$ 的乘积，即 $A=D(\rho(A)P)D^{-1}$ 其中 $D=\operatorname{diag}(x_1,\cdots,x_n).$即$(D^-1AD)/\rho(A)$是随机矩阵.


定理（随机矩阵幂序列的收敛性）：设$A$为不可约随机矩阵，则极限$\lim_m\to\infty A^m$存在的充分必要条件是 A 为本原矩阵.

双随机矩阵是一类特殊的随机矩阵，因而它具有随机矩阵的所有性质，并且还有如下结果.

定理：设 $A\in\mathbb{R}^{n\times n}$是双随机矩阵，则
* $\rho(\boldsymbol{A})=1$,且 $\boldsymbol x=(1,\cdots,1)^\mathrm{T}$ 是$\boldsymbol{A}$ 与 $A^\mathrm{T}$ 对应于特征值 1 的特殊向量；
* $\parallel A\parallel_2\geqslant1.$

### 单调矩阵
本节简要介绍一类矩阵$A$,其特点是它的逆矩阵$A^{-1}$是非负的矩阵——单调矩阵，并说明它在求解线性方程组中的应用.

定义：设 $A\in\mathbb{R}^{n\times n}$,如果它的逆矩阵 $\mathbf{A}^-1\geqslant0$,则称 $\mathbf{A}$ 为单调矩阵.

定理（判别）：设$A\in\mathbb{R}^{n\times n}$,则$A$为单调矩阵的充分必要条件是：可从$Ax\geqslant0$推出$x\geqslant0$,这里 $x$ 是列向量.

定理：设$A$为单调矩阵，若能找到向量$x^\prime=(x_1^{\prime},\cdots,x_n^{\prime})^{\mathrm{T}}$和$x^{\prime\prime}=(x^{\prime\prime}_1,\cdots,x^{\prime\prime})^{\mathrm{T}}$分别使 $Ax^{\prime}\leqslant b,Ax^{\prime\prime}\geqslant b$,则有估计式
$$x^{\prime}\leqslant\bar{x}\leqslant x^{\prime\prime}$$
或
$$x_i^{\prime}\leqslant\tilde{x}_i\leqslant x_i^{\prime\prime},\quad i=1,\cdots,n.$$

**该定理的意义是可以帮助我们估计得到方程解的上下界**
### M与H矩阵
定义：设 $A\in\mathbb{R}^{n\times n}$,且可表示为
$$A=sI-B,\quad s>0,\quad B\geqslant0.$$
若 $s\geqslant\rho(B),则称A$为 M 矩阵；若 s>$\rho(B),则称A$为非奇异 M 矩阵

为了更好的讨论$M$型矩阵的性质，我们引入$Z$型矩阵有：

设 $A=(a_{ij})_{n\times n}$,且
$$a_{ij}\leqslant0\:,\quad i\neq j\:,i\:,j=1\:,2\:,\cdots,n\:,$$
则称$A$ 为${Z}$型矩阵,全体$\textbf{ }n$ 阶$Z$ 型矩阵的集合用记号 $Z^{n\times n}$表示.显然，M 矩阵是 Z 型矩阵的特殊情况.

定理：设 $A\in Z^{n\times n}$为非奇异 M 矩阵，且 $D\in Z^{n\times n}$满足 $D\geqslant A$,则
* $\boldsymbol{A}^{-1}$与 $\boldsymbol D^{-1}$存在，且 $\boldsymbol{A}^- 1\geqslant \boldsymbol{D}^{- 1}\geqslant 0$ ;
* $D$的每个实特征值为正数；
* $det D\geqslant\det A>0.$

定理：非奇异$M$型矩阵有许多等价条件 $A\in Z^{n\times n}$ 下面命题等价
* $\text{A 为非奇异 M 矩阵}$
* $\text{若 }B\in Z^{n\times n}\text{且 }B\geqslant A,\text{则 }B\text{ 非奇异}$
* $A\text{ 的任意主子矩阵的每一个实特征值为正数}$
* $\text{A 的所有主子式为正数}$
* $\text{对每个 }k(1\leqslant k\leqslant n),A\text{ 的所有 }k\text{ 阶主子式之和为正数}$
* $\text{A 的每一个实特征值为正数}$
* $\text{存在 }A\text{ 的一种分裂 }A=P-Q,\text{使得 }P^{-1}\geqslant0,Q\geqslant0\text{ 且 }\rho(P^{-1}Q)<1$
* $A\text{ 非奇异,且 }A^{-1}\geqslant0.$

定理 ：设$A\in Z^{n\times n}$是对称的，则$A$为非奇异 M 矩阵的充分必要条件是$A$为正定矩阵

定理：设$A,B\in\mathbb{R}^{n\times n}$是非奇异 M 矩阵，则$AB$为非奇异 M 矩阵的充分必要条件是$AB\in Z^n\times n.$

---

下面我们来讨论一些$M$矩阵的问题

定理：设 $A\in Z^{n\times n}$ 下面命题等价
* $A是M矩阵$
* $对每个\varepsilon>0,A+\varepsilon I是非奇异M矩阵$
* $\text{A 的任意主子矩阵的每个实特征值非负}$
* $\text{A的所有主子式非负}$
* $\text{对每个 }k=1,2,\cdotp\cdotp\cdotp,n,A\text{ 的所有 }k\text{ 阶主子式之和为非负实数}$
* $\text{A 的每个实特征值非负}$

定理：设$A$是不可约的奇异$M$矩阵 则
* $\mathrm{rank}A=n-1$
* $存在正向量x>0,使得Ax=0$
* $A\text{ 的所有真主子矩阵为非奇异的 M 矩阵},\text{特别有 }a_n>0\mathrm{(1}\leqslant i\leqslant n)$
* $\text{对任意 }x\in\mathbb{R}^n,\text{若 }Ax\geqslant0,\text{则 }Ax=0$

---
下面将$n$阶方阵$A$推广到复矩阵，且利用$A$中的元素取模构造出一个新的比较矩阵，记为 ${H}(\boldsymbol{A})$,如果 $H(\boldsymbol{A}$)是非奇异的 $M$ 矩阵，则定义 $\boldsymbol{A}$ 为 $H$ 矩阵.

定义：设 $A=(a_{ij})\in\mathbb{C}^{n\times n}$,并设
$$\mathrm{H}(\mathbf{A})=(m_{ij})\in\mathbb{R}^{n\times n},$$
其中

$$m_{ij}=\begin{cases}\quad\mid a_{ij}\mid,\quad j=i,\\-\mid a_{ij}\mid,\quad j\neq i,\end{cases}i,j=1,\cdotp\cdotp\cdotp,n,$$
$H(A)$称为 $A$ 的比较矩阵.

定义：设$A\in\mathbb{C}^{n\times n}$,如果$A$的比较矩阵$H(A)$是非奇异的 M 矩阵，则称$A$为非奇异 $H$ 矩阵，简称 $H$ 矩阵

下面简要给出 $H$ 矩阵的一些性质.

定理：设 $A,\boldsymbol{B}\in\mathbb{C}^{n\times n},A$ 是非奇异 $M$ 矩阵，$H(B)\geqslant\mathcal{A}$,则
* $B$是 $H$ 矩阵；
* $B$是非奇异的，且$A^-1\geqslant|B^{-1}|\geqslant0;$
* $\mid\det B\mid\geqslant det A>0$

定理：设 $A\in C^{n\times n}$ 则有下面性质
* $\mathcal{H}(A)\in\mathbb{Z}^{n\times n}$
* $\mathcal{H}(A)=A\text{ 的充分必要条件是 }A\in\mathbb{Z}^{n\times n}$
* $A\text{ 为 M 矩阵的充分必要条件是 H}(A)=A,\text{且 A 为 H 矩阵}$
* H(A)可表示为非负对角矩阵与具有零对角的非负矩阵之差：$$H( \boldsymbol A) = \mid diag(a_{11},\cdots,a_{nn})\mid-[\mid\boldsymbol{A}\mid-\mid diag(a_{11},\cdots,a_{nn})\mid]$$这里$|\boldsymbol{X}|\equiv[|x_{ij}|]$表示矩阵$\boldsymbol{X}=(x_ij)\in\mathbb{C}^{n\times n}$的逐个元素取绝对值后的矩阵；
* 如果$A$是M矩阵，那么前式改写为$$\mathbf{A}=\operatorname{diag}(a_{11},\cdotp\cdotp\cdotp,a_{nn})-\begin{bmatrix}\operatorname{diag}(a_{11},\cdotp\cdotp\cdotp,a_{nn})-\mathbf{A}\end{bmatrix}$$
### T矩阵与汉克尔矩阵
我们在很多领域会经常遇到下面类型的矩阵
$$\mathbf{A}=\begin{bmatrix}a_0&a_{-1}&a_{-2}&\cdots&a_{-n+1}\\\\a_1&a_0&a_{-1}&\cdots&a_{-n+2}\\\\a_2&a_1&a_0&\cdots&a_{-n+3}\\\vdots&\vdots&\vdots&&\vdots\\\\a_{n-2}&a_{n-3}&a_{n-4}&\cdots&a_{-1}\\\\a_{n-1}&a_{n-2}&a_{n-3}&\cdots&a_0\end{bmatrix}$$
任意一条平行于主对角线的直线的元素完全相同，我们将这样的矩阵称为T矩阵

T矩阵的性质不好研究，因此人们的重心逐渐转向下面形式的矩阵
$$\boldsymbol{H}_{n+1}=\begin{bmatrix}a_0&a_1&a_2&\cdots&a_n\\\\a_1&a_2&a_3&\cdots&a_{n+1}\\\\a_2&a_3&a_4&\cdots&a_{n+2}\\\vdots&\vdots&\vdots&&\vdots\\\\a_n&a_{n+1}&a_{n+2}&\cdots&a_{2n}\end{bmatrix}$$
任意一条平行于副对角线的直线的元素完全相同，我们将这样的矩阵称为汉克尔矩阵，他是一个非奇异矩阵

可以直接验证，T 矩阵与汉克尔矩阵是可以互相转化的.事实上，设 T 矩阵为 $A$,汉克尔矩阵为$H_{n+1}$,则用矩阵
$$\boldsymbol{J}=\begin{bmatrix}&&&1\\&1&&\\1&&&\end{bmatrix}$$
乘矩阵 $H_{n+1}$,其结果 $JH_{n+1}$或 $H_{n+1}J$ 都是 T 矩阵，且有
$$(JH_{n+1})^\mathrm{T}=H_{n+1}J.$$
反之，用$J$乘 T 矩阵$A$,则$JA$或$AJ$都是汉克尔矩阵.

## 特殊积
矩阵的特殊积，还是研究矩阵 $AB$ 但是此时不再要求 $A$ 的列数等于 $B$ 的行数。这种不受矩阵行列约束的特殊积，在很多地方有着简洁的效果
### 克罗内克积
前面定义过两个矩阵$A$和$B$的乘积$AB$ ,它要求$A$的列数必须等于$B$的行数.下面引进一种新的乘法运算，它对矩阵的行数和列数没有任何要求.

定义：设 $A=(a_{ij})\in\mathbb{C}^{m\times n},\boldsymbol{B}=(b_{ij})\in\mathbb{C}^{p\times q}$,则称如下的分块矩阵
$$A\otimes B=\begin{vmatrix}a_{11}B&a_{12}B&\cdots&a_{1n}B\\a_{21}B&a_{22}B&\cdots&a_{2n}B\\\vdots&\vdots&&\vdots\\a_{m1}B&a_{m2}B&\cdots&a_{mn}B\end{vmatrix}\in\mathbb{C}^{mp\times nq}$$
为$A$的克罗内克( Kronecker)积，或称$A$与 $B$ 的直积 ,或张量积，简记为$A\otimes\boldsymbol{B}=(a_{ij}\boldsymbol{B})_{mp\times nq}.$ 即$A\otimes B$是一个$m\times n$块的分块矩阵，最后是一个$mp\times nq$矩阵.

明显的，克罗内克积不满足交换律，不过最后结果的阶数是相同的

容易验证，克罗内克积有下面的运算率
* $k(A\otimes B)=kA\otimes B=A\otimes kB,k\in\mathbb{C}$
* $(A+B)\otimes C=A\otimes C+B\otimes C$
* $(A\otimes B)\otimes C=A\otimes(B\otimes C)$

---

定理：设 $A=(a_{ij})_{m\times n},\boldsymbol{B}=(b_{ij})_{s\times r},\boldsymbol{C}=(c_{ij})_{n\times p},\boldsymbol{D}=(d_{ij})_{r\times l}$,则
$$(A\otimes B)(C\otimes D)=AC\otimes BD.$$

推论：设 $A=(a_{ij})_{m\times n},\boldsymbol{B}=(b_{ij})_{s\times r}$,则
$$A\otimes B=(A\otimes I_n)(I_m\otimes B)=(I_m\otimes B)(A\otimes I_n)$$

定理：设 $A=(a_{ij})_m\times n$,则 $rank(\boldsymbol{A})\leqslant1\Leftrightarrow\boldsymbol{A}$ 可以表示成一个行向量和一个列向量的克罗内克积.

定理：设 $\boldsymbol{A}=(a_{ij})_{m\times n},\boldsymbol{B}=(b_{ij})_{p\times q}$,则
$$\begin{aligned}(A\otimes B)^\mathrm{T}&=A^\mathrm{T}\otimes B^\mathrm{T}\:,\\(A\otimes B)^\mathrm{H}&=A^\mathrm{H}\otimes B^\mathrm{H}.\end{aligned}$$
据此容易推出，对称矩阵（埃尔米特矩阵）的克罗内克积还是对称矩阵（埃尔米特矩阵）

定理：设$A,B$分别为$m$阶和$n$阶可逆矩阵，则$A\otimes B$也为可逆矩阵.且
$$(A\otimes B)^{-1}=A^{-1}\otimes B^{-1}.$$

定理：设 $\boldsymbol{A}=(a_{ij})_{m\times n},\boldsymbol{B}=(b_{ij})_{p\times q}$,则
$$\operatorname{rank}(\boldsymbol{A}\otimes\boldsymbol{B})=\operatorname{rank}(\boldsymbol{A})\operatorname{rank}(\boldsymbol{B}).$$
$$\mathrm{tr}(A\otimes B)=\mathrm{tr}A\mathrm{tr}B$$

定理：设$x_1,x_2,\cdots,x_n$是$n$个线性无关的$m$维列向量$,y_1,y_2,\cdots,y_q$是$q$个线性无关的$p$维列向量，则$nq$个 $m p$ 维列向量$\mathbf{x}_i\otimes\mathbf{y}_j(i=1,\cdotp,n;j=1,\cdotp\cdotp,q)$亦线性无关，反之亦然

定理：设$A,B$分别为$m,p$阶方阵，则有
$$|A\otimes B|=\begin{array}{c|cc}|A|^p&|B|^m.\end{array}$$


定理：设 $A=(a_{ij})_{m\times p},\boldsymbol{B}=(b_{ij})_{p\times n}$,则有
$$(AB)^{[k]}=A^{[k]}B^{[k]}$$

定理：设 $\lambda_1,\lambda_2,\cdots,\lambda_m$ 是$A_m\times m$的 $m$ 个特征值 $,\mu_1,\mu_2,\cdots,\mu_p$ 是$B_{p\times p}$的 $p$ 个特征值，那么$A\otimes B$的 $m p$ 个特征值为$$\lambda_i\mu_j(i=1,2,\cdotp\cdotp\cdotp,m;j=1,2,\cdotp\cdotp\cdotp,p).$$

定理：设$A$为$m$阶矩阵，B 为$n$阶矩阵，则有$A\otimes B$相似于$B\otimes A$

定理：设$f(x,y)=\sum_{i,j=0}^{r}a_{ij}x^{i}y^{j}$是变量$x,y$的复系数多项式，对于$A\in\mathbb{C}^m\times m$, $B\in\mathbb{C}^{n\times n}$定义$mn$阶矩阵：
$$f(A,B)=\sum_{i,j=0}^p\alpha_{ij}A^i\otimes B^j.$$
如果$A$和$B$的特征值分别是$\lambda_1,...,\lambda_m$和$\mu_1,...,\mu_n$,它们对应的特征向量分别是$x_{1},\cdots,x_{m}$和$y_1,\cdots,y_n$,则矩阵 $f(A,B)$的特征值是 $f(\lambda_r,\mu_s)$,而对应 $f(\lambda_r,\mu_s)$的特征向量为$x_r\otimes y_s( r= 1, . . . , m;$ $s= 1, . . . , n) .$


基于本定理，我们容易给出下面推论

推论1：我们取$f(x,y)=xy$  $\boldsymbol{A}\otimes\boldsymbol{B}$ 的特征值是 $mn$ 个数 $\lambda_r\mu_s$  对应的特征向量为 $x_r\otimes y_s$

推论2：取$f(x,y)=x+y$ 也就是 $f(x,y)=xy^0+x^0y$  则有 $A\otimes I_n+I_m\otimes B$的特征值为$\lambda_r+\mu_s$ 特征向量 $x_r\otimes y_s$

我们称矩阵
$$A\otimes I_n+I_m\otimes B$$
为$A$与$B$的克罗内克和

最后，我们介绍一类特殊的矩阵

定义：元素为 1 或-1 的方阵 $H\in\mathbb{R}^{n\times m}$,若有
$$HH^\mathrm{T}=nI_n,$$
则称$H$为$n$阶阿达马(Hadamard)矩阵.

定理：设$H_m$ 与$H_n$ 均为阿达马矩阵，则矩阵 $H_m\otimes H_n$ 为 $mn$ 阶的阿达马(Hadamard)矩阵.

### 阿达马积
阿达马乘法远比通常矩阵乘法简单，但未被广泛地了解.它出现在很多问题中，因此我们在此处讨论他。

定义：设$A=(a_{ij}),{B}=(b_{ij})\in\mathbb{C}^{m\times n}.$用$A^\circ B$表示$A$和 B 的对应元素相乘而得到的$m\times n$矩阵：
$$A\circ B=\begin{bmatrix}a_{11}b_{11}&a_{12}b_{12}&\cdots&a_{1n}b_{1n}\\a_{21}b_{21}&a_{22}b_{22}&\cdots&a_{2n}b_{2n}\\\vdots&\vdots&&\vdots\\a_{m1}b_{m1}&a_{m2}b_{m2}&\cdots&a_{mn}b_{mn}\end{bmatrix},$$
称为 A 和 B 的阿达马积 ,也称为舒尔积。

**明显的，阿达马积需要两个矩阵同型，并且他是可交换的 $A\circ B =B\circ A$**

定理：设 $A,{B},C\in\mathbb{C}^{m\times n}.$ 关于阿达马积的运算有下面的性质
* $A\circ(B+C)=A\circ B+A\circ C$
* $A\circ(B\circ C)=(A\circ B)\circ C$
* $(A\circ B)^\mathrm{T}=A^\mathrm{T}\circ B^\mathrm{T}$
* $(A\circ B)^{\mathrm{H}}=A^{\mathrm{H}}\circ B^{\mathrm{H}}$
* $\text{如果 }A\text{ 和 }B\text{ 是自伴矩阵(即埃尔米特矩阵)},\text{那么 }A\circ B\text{ 也是自伴矩阵}$
* $\text{如果 }A\text{ 和 }B\text{ 是斜自伴(即反埃尔米特)矩阵},\text{那么 }A\circ B\text{ 是自伴矩阵}$
* $\text{如果 }A\text{ 是自伴矩阵 },B\text{ 是斜自伴矩阵 },\text{那么 }A\circ B\text{ 是斜自伴矩阵}$
* $\mathrm{rank}(A\circ B)\leqslant(\mathrm{rank}A)(\mathrm{rank}B)$
* $\text{若 }A,B\text{ 是半正定矩阵},\text{则 }A\circ B\text{ 也是半正定矩阵}$
* $\text{若 }B\text{ 是正定矩阵 },A\text{ 是半正定矩阵且无零对角元素 },\text{则 }A\circ B\text{ 是正定矩阵}$
* $若A和B都是正定矩阵,则A\circ B也是正定矩阵$

定理：设 $A,B\in\mathbb{C}^{n\times n}$是半正定矩阵，则成立
$$\lambda_{\min}(A\circ B)\geqslant\lambda_{\min}(A)\lambda_{\min}(B)$$
和
$$\lambda_{\max}(A\circ B)\leqslant\lambda_{\max}(A)\lambda_{\max}(B)\:,$$
其中 $\lambda_{\min}(\boldsymbol{A})$和 $\lambda_{\max}(\boldsymbol{A})$分别表示 $\boldsymbol{A}$ 的最小特征值和最大特征值。

和

定理：设 $A,B\in\mathbb{C}^{n\times n}$是半正定矩阵，则成立
$$\lambda_{\min}(A\circ B)\geqslant\lambda_{\min}(AB^{\mathrm{T}})$$
$$\lambda_{\min}(A\circ B)\geqslant\lambda_{\min}(AB).$$


### 反积
定义：设 $\boldsymbol{A}=(a_{ij}),\boldsymbol{B}=(b_{ij})\in\mathbb{C}^{m\times n}.$令
$$c_{ij}=\begin{cases}\quad a_{ii}b_{ii}\:,\quad&j=i\:,\\[2ex]-a_{ij}b_{ij}\:,\quad&j\neq i\:,\quad&i=1\:,\cdots,m\:,\quad j=1\:,\cdots,n.\end{cases}$$
记$A\star B=(c_{ij})\in\mathbb{C}^{m\times n}$,并称其为$A$和$B$的反积(Fan 积).

容易看出 ：**反积是阿达马积的一种变异.**


定理：关于反积以及非负矩阵的阿达马积有如下的基本性质.
* 若$A,B\in\mathbb{R}^{n\times n}$是 M 矩阵，则$A\star B$也是 M 矩阵；
* 若 $A,\boldsymbol{B}\in\mathbb{C}^{n\times n}$是 H 矩阵，则 $A\star\boldsymbol{B}$ 也是 H 矩阵，$A\circ\boldsymbol{B}$ 是非奇异的.

定理：设$A,B\in\mathbb{R}^{n\times n},A\geqslant0,B\geqslant0,则$
* $A^{\circ}B\geqslant0$,也就是说，非负矩阵类在阿达马积下是封闭的；
* $\rho ( A^{\circ }B) \leqslant \rho ( A) \beta ( B) .$

定理：设 $A,\boldsymbol{B}\in\mathbb{R}^{n\times n}$是 M 矩阵，则 $A\circ\boldsymbol{B}^{-1}$也是 M 矩阵.

### 克罗内克积的应用
利用矩阵克罗内克积的性质，我们可以容易的研究线性矩阵方程
$$A_1XB_1+A_2XB_2+\cdots+A_pXB_p=C$$
事实上，他可以转换为一般线性方程
$$Gx=c$$
这就是本节希望讨论的问题

#### 矩阵的拉直
定义：设$\boldsymbol{A}=(a_{ij})_{m\times n}$,将$\boldsymbol{A}$的**各行依次按列纵排**得到的$mn$维列向量，这种运算称为$A$的拉直，记为$\vec{A}$,即
$$\vec{A}=(a_{11},a_{12},\cdotp\cdotp\cdotp,a_{1n},a_{21},a_{22},\cdotp\cdotp\cdotp,a_{2n},\cdotp\cdotp\cdotp,a_{m1},a_{m2},\cdotp\cdotp\cdotp,a_{mn})^\mathrm{T}.$$
容易知道，拉直算子是线性的 * $\overrightarrow{A+B}=\vec{A}+\vec{B},\quad\vec{kA}=k\vec{A}$

定理：关于拉直算子，我们可以给出下面的连续推证
1. $xy^\mathrm{T}=x\otimes y,\text{其中 }x,y\text{ 为 }n\text{ 维列向量}$
2. $\boldsymbol{E}_{ij}=\boldsymbol{e}_i\boldsymbol{e}_j^\mathrm{T}$,其中$\boldsymbol E_{ij}$表示($i,j)$元素为 l,其余元素为 0 的 $m\times n$ 阶矩阵，$\boldsymbol e_i$ 表示第 $i$ 个元素为 1,其余元素为 0 的列向量；
3. $Ae_i=\begin{bmatrix}a_{1i}\\\\a_{2i}\\\vdots\\\\a_{mi}\end{bmatrix}$
4. $e_j^\mathrm{T}A=(a_{j1},a_{j2},\cdots,a_{jn})$
5. $\vec{E}_{ij}=e_i\otimes e_j$

定理：设 $A=(\mu_{ij})_{m\times n},\boldsymbol{B}=(b_{ij})_{n\times p},\boldsymbol{C}=(c_{ij})_{p\times q}$,则
$$\overrightarrow{ABC}=(A\otimes C^{\mathrm{T}})\vec{B}.$$
推论：设 $A=(\mu_{ij})_{m\times n},\boldsymbol{B}=(b_{ij})_{n\times p},\boldsymbol{X}=(x_{ij})_{p\times q}$,则
* $\overrightarrow{AX}=(A\otimes I_{n})\vec{X}$
* $\overrightarrow{XB}=(I_m\otimes B^\mathrm{T})\vec{X}$
* $\overrightarrow{AX+XB}=(A\otimes I_n+I_m\otimes B^\mathrm{T})\vec{X}.$

#### 线性矩阵方程的解
定理：矩阵$X\in\mathbb{C}^{m\times n}$是矩阵方程$A_1XB_1+A_2XB_2+\cdots+A_pXB_p=C$的解的充分必要条件是$x=\vec{X}$为通常的线性方程组
$$Gx=c$$
的解，其中$G=\sum_{i=1}^pA_i\otimes B_i^\mathrm{T},c=\vec{C}.$

---
下面我们来讨论一个特殊情况，研究方程
$$AX+XB=C$$
定理：前述矩阵方程有惟一解$X\in\mathbb{C}^{m\times n}$的充要条件是$A$和 $-B$ 没有相同的特征值，即
$$\lambda_i+\mu_j\neq0,\quad i=1,\cdotp\cdotp\cdotp,m,\quad j=1,\cdotp\cdotp\cdotp,n.$$

---
研究方程
$$X+AXB=C$$
定理：前述矩阵方程有惟一解 $x\in\mathbb{C}^{m\times n}$的充要条件是$\lambda_i\mu_j\neq-1(i=1,\cdots$,
$m;j=1,\cdots,n$)$,\lambda_i$ 和$\mu_j$ 分别为$A$ 与$B$ 的特征值.
