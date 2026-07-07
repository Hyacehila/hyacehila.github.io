---
title: "矩母函数、特征函数与分布理论：生成函数视角下的概率分布"
title_en: "Moment Generating Functions, Characteristic Functions, and Distribution Theory"
date: 2025-11-18 21:52:54 +0800
categories: ["Data Science", "Probability & Statistical Foundations"]
tags: ["Statistics", "Probability", "Distribution Theory"]
author: Hyacehila
excerpt: "整理生成函数视角下的概率分布研究方法。"
excerpt_en: "A overview on generating functions, moment generating functions, characteristic functions, and distribution theory from a probabilistic distribution perspective."
mathjax: true
hidden: true
permalink: '/blog/2025/11/18/moment-generating-characteristic-functions-distribution-theory-notes/'
---
## 母函数与矩母函数
在高等概率论开始之前，我们将初等概率论中的一些比较复杂的内容进行一些回顾，**事实上概率的生成函数理论允许我们把概率问题转化为分析问题思考** 对某些问题的证明将起到很好的简化作用

### 母函数的定义与性质
母函数是为了引入概率分布的另一种表示方法 来简化一些研究

对于那些只取非负整值的随机变量（比如二项分布，伯努利分布，泊松分布等）我们称为整值随机变量 对于这一类随机变量 母函数可以起到一定的辅助研究的作用（他的核心是一种形式幂级数，每一项的系数体现了序列的全部信息）

母函数的定义为
$$G(t)=\sum_{k=0}^\infty p_kt^k=p_0+p_1t+\ldots+p_nt^n+\ldots $$
容易看出来 $G(t)=E(t^{p_k})$  

根据级数的性质我们知道 母函数有下列性质
* 母函数和分布列相互唯一确定
* $G(1)=1$
* $\mathrm{E}(X)=G^{\prime}(1)$
* $\mathrm{Var}(X)=G^{\prime\prime}(1)+G^{\prime}(1)\left(1-G^{\prime}(1)\right)$
* 对于两个独立随机变量 设他们的母函数为$G_{X_1}(t),G_{X_2}(t)$ 则根据离散卷积公式有 他们的和的母函数满足$G_X(t)=G_{X_1+X_2}(t)=G_{X_1}(t)\times G_{X_2}(t)$

### 矩母函数定义和性质
母函数只能用于离散型随机变量，特征函数涉及复积分不方便操作（他的优点就是永远存在）

这里我们介绍矩母函数（**Moment Generating Function**）理论 他在处理很多和随机变量相关的问题上有不少的优点 所谓的矩母 意味着他可以生成矩

随机变量$X$的矩母函数定义如下
$$M_X(t)=E[\mathrm{e}^{tX}]=\begin{cases}\int_{-\infty}^{+\infty}\mathrm{e}^{tx}f(x)\mathrm{d}x,X\text{ 具有密度函数 }f(x);\\\\\sum_{i=0}^{\infty}\mathrm{e}^{tx_i}\rho\left(x_i\right),X\text{ 具有分布律 }p(x).\end{cases}$$
我们给出一些常用的性质
* 矩母函数$M_X(t)$和随机变量$X$ 唯一确定
* 如果随机变量$X,Y$相互独立  那么$M_{X+Y}(t)=M_{X}(t)M_{Y}(t)$
* 如果$Y=aX+b$ 那么$M_Y(t)=e^{bt}M_{X}(at)$
* 矩母函数可以用来求原点矩  $E(X^{n})=M_{X}^{(n)}(0)$ 
* $M(1)=1$ 这一点和母函数一样
### 简单的应用
这里会介绍一些应用和他们的值得学习的证明技巧
#### 正态分布的矩母函数
$$\begin{aligned}
M_{X}\left(t\right) =E\left[e^{tX}\right]  \\
&=\int_{-\infty}^{\infty}e^{tx}\frac{1}{\sqrt{2\pi\sigma}}e^{-\frac{\left(x-\mu\right)^{2}}{2\sigma^{2}}}dx \\
&=e^{\left(\mu t+\frac{1}{2}\sigma^{2}t^{2}\right)}\int_{-\infty}^{\infty}\frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{\left[x-\left(\mu+\sigma^{2}t\right)\right]^{2}}{2\sigma^{2}}}dx \\
&=e^{\left(\mu t+\frac{1}{2}\sigma^{2}t^{2}\right)},
\end{aligned}$$
这里面涉及的积分技巧是通过构造了一个概率密度函数（无关常数放外面），他的积分是1 这样简化了积分问题 这个思路非常的常见 以后还会再次用到
根据前面给出的性质四 我们知道$E(X)=\mu ~~~~E(X^{2})=\mu^{2}+\sigma^{2},Var(X)=\sigma^{2}$

#### Gamma分布的矩母函数
$$\begin{gathered}f(x)=\frac{\beta^\alpha}{\Gamma(\alpha)}x^{\alpha-1}e^{-\beta x},x>0,~~\Gamma(\alpha)=\int_0^\infty y^{\alpha-1}e^{-y}dy\\\text{易知}\Gamma(\frac12)=\sqrt{\pi},\Gamma(1)=1,\Gamma(\alpha+1)=\alpha\Gamma(\alpha),\Gamma(n+1)=n!\end{gathered}$$
这是Gamma分布的形式表示 还有一些比较重要的性质叙述
$$M(t)=\int_0^\infty\frac{\beta^\alpha}{\Gamma(\alpha)}x^{\alpha-1}e^{-\beta x}e^{tx}dx=(\frac\beta{\beta-t})^\alpha $$
还是前面的构造了概率密度函数的思路
借助他我们很容易确定$E[X]=M^{^{\prime}}(0)=\frac\alpha\beta,\quad\text{进而}Var(X)=\frac\alpha{\beta^2}$
*注意$\alpha=1$的时候，他就是指数分布$f(x)=ke^{-kx}$* 

#### 二项分布的矩母函数与母函数
$$\begin{aligned}
M(t)& =E[e^{tX}]=\sum_{k=0}^ne^{tk}C_n^kp^k(1-p)^{n-k}  \\
&=\sum_{k=0}^nC_n^k(pe^t)^k(1-p)^{n-k} \\
&=(pe^t+1-p)^n
\end{aligned}$$
思路没变,也可以说这里使用的经典的二项展开式 容易计算得到
$E[X]=np,\quad E[X^{2}]=M^{^{\prime\prime}}(0)=n(n-1)p^{2}+np,\quad Var(X)=np(1-p)$

对应有母函数为
$$M_{X}(t)=E\left(e^{t X}\right)=\left(1-p+p e^{t}\right)^{n}$$
#### 泊松分布的矩母函数与母函数
$$\begin{aligned}
M(t)& =E[e^{tX}]=\sum_{k=0}^\infty\frac{e^{tk}e^{-\lambda}\lambda^k}{k!}  \\
&=e^{-\lambda}\sum_{k=0}^\infty\frac{(\lambda e^t)^k}{k!}=e^{-\lambda}e^{\lambda e^t} \\
&=e^{[\lambda(e^{t}-1)]}
\end{aligned}$$
也可以说我们使用的经典的幂级数展开式
容易计算得到
$E[X]=\lambda,\quad E[X^2]=M^{^{\prime\prime}}(0)=\lambda+\lambda^2,\quad Var(X)=\lambda$

对应有母函数为
$$M_{X}(t)=E\left(e^{t X}\right)=e^{\lambda\left(e^{t}-1\right)}$$

#### 卡方分布的矩母函数
$\begin{aligned}&\text{令}Z_1,Z_2,\cdots,Z_n\text{ 以及}Z\text{ 是独立同分布的标准正态分布随机变量,}\\&\text{且}X=Z_1^2+Z_2^2+\cdots+Z_n^2\text{,则我们称X服从卡方分布}\end{aligned}$
容易知道
$$M_X(t)=[M_{Z^2}(t)]^n=(E[e^{tZ^2}])^n$$
进行如下计算
$$\begin{aligned}
E[e^{tZ^2}]& =\frac1{\sqrt{2\pi}}\int_{-\infty}^\infty e^{tz^2}e^{-\frac{z^2}2}dz  \\
&=\frac1{\sqrt{2\pi}}\int_{-\infty}^\infty e^{-\frac{z^2}{2\sigma^2}}dz\quad(\sigma^2=(1-2t)^{-1}) \\
&=(1-2t)^{-\frac12}
\end{aligned}$$
还是经典的构造手段，我们得到了一阶矩母函数 再进行求$n$次方就是需要求的矩母函数
$E[X]=n,\quad E[X^2]=n(n+2),\quad Var(X)=2n$
*我们直接带入正态分布的密度函数就能完成这部分的计算*
### 一个例子
计算积分
$$\begin{aligned}&(1)\int_{-\infty}^\infty(2x^2+2x+3)e^{-(x^2+2x+3)}dx\text{,}\\&(2)\int_0^\infty(4x^2+5x+6)e^{-(2x+1)}dx.\end{aligned}$$
两个问题的思路是一样的，都是需要构造分布 借助矩表示原有的积分，然后用简单的方法得到矩 来进行表示 其核心还是在于构造分布
(1) $$=e^{-2}\int_{-\infty}^{+\infty}\left(2x^{2}+2x+3\right)e^{-\frac{(x+1)^{2}}{2\left(\frac{1}{\sqrt{2}}\right)^{2}}}dx$$
现构造出我们需要的密度函数
$$=\sqrt{\pi}e^{-2}\left(2E(x^2)+2E(x)+3\right)$$
这是根据原点矩的定义得到的
带入原点矩就可以得到最后的答案

第二题是完全一模一样的思路，只是换成构造Gamma分布函数 或者说 指数分布函数

### 多元矩母函数
**多元矩母函数定义**：假设 $X = (X_1, \ldots, X_n)'$ 是一个具有 $n$ 维密度 $f(x_1, \ldots, x_n)$ 的 $n \times 1$ 随机向量。$X$ 的多元矩母函数定义为：
$$
\begin{aligned}
\psi_{X_1,\ldots,X_n}(t_1, \ldots, t_n) &= E(e^{t_1 X_1 + \ldots + t_n X_n}) \\
&= \int_{-\infty}^{\infty} \cdots \int_{-\infty}^{\infty} e^{t_1 x_1 + \ldots + t_n x_n} f(x_1, \ldots, x_n) \, dx_1 \ldots dx_n \\
&= E(e^{t' X})
\end{aligned}
$$
其中 $t = (t_1, \ldots, t_n)'$ 且 $X = (X_1, \ldots, X_n)'$

**多元矩母函数的性质**

设 $X = (X_1, \ldots, X_n)'$ 且 $t = (t_1, \ldots, t_n)'$，多元矩母函数 $\psi_X(t)$ 具有以下性质：

1. $\psi_X(0) = 1$
2. 如果 $X_1, \ldots, X_n$ 是独立的，则：$$
   \psi_X(t) = \prod_{i=1}^{n} \psi_{X_i}(t_i)
   $$其中 $\psi_{X_i}(t_i)$ 是 $X_i$ 的一元矩母函数。
3. 可以通过对多元矩母函数求导获得矩：$$
   \left. \frac{\partial^{k_1 + \cdots + k_n}}{\partial t_1^{k_1} \cdots \partial t_n^{k_n}} \psi_X(t_1, \ldots, t_n) \right|_{t_1 = \cdots = t_n = 0} = E(X_1^{k_1} \cdots X_n^{k_n})
   $$例如，当 $n = 2$ 时，$X = (X_1, X_2)'$，我们有：$$
   \left. \frac{\partial^5}{\partial t_1^2 \partial t_2^3} \psi_X(t_1, t_2) \right|_{t_1 = t_2 = 0} = E(X_1^2 X_2^3)
   $$
4. 通过将不在边际分布中的 $X_j$ 对应的 $t_j$ 设为 0，可获得 $X$ 的任何边际分布的矩母函数。例如，当 $n = 4$ 时，$X = (X_1, X_2, X_3, X_4)'$，$\psi_X(t_1, t_2, t_3, t_4)$ 是 $X$ 的多元矩母函数。则 $\psi_{X_1,X_3}(t_1, t_3) = \psi_X(t_1, 0, t_3, 0)$，$\psi_{X_1}(t_1) = \psi_X(t_1, 0, 0, 0)$ 等。
5. 多元特征函数定义为 $\phi_X(t) = E(e^{it'X})$，其中 $i = \sqrt{-1}$。特征函数总是对任何随机变量或向量存在，但矩母函数对某些随机变量可能不存在。因此，特征函数在证明某些结果时比矩母函数更有用。特征函数与矩母函数的关系为：$$
   \phi_X(t_1, \ldots, t_n) = \psi_X(it_1, \ldots, it_n)
   $$
   


## 特征函数

正如前面研究的,我们需要找一个对任何随机变量或向量存在的函数来刻画随机变量与随机向量.
### 引入
我们在数学分析中介绍过函数傅立叶变化 形式为
$$\varphi\left(t\right)=\int_{-\infty}^{\infty}e^{itx}p\left(x\right)dx$$
如果$p(x)$是一个密度函数 那么我们知道 $\varphi(t)=E\left(e^{itX}\right)$ 

这就是我们这里要研究的特征函数的问题 它是非常多概率问题的优秀处理工具 可以简化运算和思考
### 特征函数的定义
复随机变量的定义形式为 $Z=Z\left(w\right)=X\left(w\right)+iY\left(w\right)$  他有着和普通的实值随机变量接近的性质 只是所有的运算结果范围变成了复数

**定义**：设$X$是一个随机变量 称
$$
\phi(t) = E(e^{itX}) = \int_{-\infty}^{\infty} e^{itx} \, dF(x)
$$
是$X$的特征函数, 我们知道 $X$的特征函数总是存在 无论随机变量的形式如何.并且分布函数由其特征函数唯一确定。

离散型随机变量的特征函数为
$$\varphi\left(t\right)=\sum_{k=1}^{\infty}\mathrm{e}^{\mathrm{i}tx_{k}}p_{k}$$
连续型随机变量的特征函数为
$$\varphi\left(t\right)=\int_{-\infty}^{\infty}e^{itx}p\left(x\right)dx$$
特征函数也唯一的依赖与分布函数 我们也可以称其为分布的特征函数

**特征函数本质上是概率测度的Fourier变换，即**
$$\Phi_{\mu}(\theta):=\int_{\mathbb{R}^{n}} e^{\mathrm{i} \theta x} \mu(\mathrm{~d} x)=\int_{\mathbb{R}} \cos (\theta x) \mu(\mathrm{d} x)+\mathrm{i} \int_{\mathbb{R}^{n}} \sin (\theta x) \mu(\mathrm{d} x)$$
根据我们在前面介绍的关于期望与积分的基础能看出，这和我们的初等定义不矛盾。

### 常用分布的特征函数
#### 单点分布
$$\varphi\left(t\right)=e^{ita}$$
其中的$a$是密度函数取$1$的点
#### 0-1分布
$$\varphi\left(t\right)=pe^{it}+q$$
#### 二项分布
$$\varphi\left(t\right)=\left(pe^{jt}+q\right)^{n}$$
#### 泊松分布
$$\varphi\left(t\right)=\sum_{k=0}^{\infty}e^{ikt}\frac{\lambda^{k}}{k!}e^{-\lambda}=e^{-\lambda}e^{\lambda e^{it}}=e^{\lambda\left(e^{it}-1\right)}.$$
#### 均匀分布
$$\varphi\left(t\right)=\int_{a}^{b}\frac{e^{itx}}{b-a}dx=\frac{e^{ibt}-e^{iat}}{it\left(b-a\right)}.$$
#### 标准正态分布
$$\varphi\left(t\right)=e^{-\frac{t^2}{2}}$$
#### 指数分布
$$\varphi\left(t\right)=\left(1-\frac{it}{\lambda}\right)^{-1}$$
#### 正态分布
$$\varphi\left(t\right)=e^{i\mu t-\frac{\sigma^{2}t^{2}}2}$$
#### Gamma分布
$$\varphi\left(t\right)=\left(\frac\lambda{\lambda-jt}\right)^r$$
### 特征函数的性质
* $\varphi(0)=1,|\varphi(t)|\leq\varphi(0)$
* $\varphi(-t)=\bar{\varphi}(t)$
* $Y=aX+b$ 则$\varphi_Y(t)=e^{ibt}\varphi_X(at)$ 
* $Z=X+Y$ 且相互独立 则 $\varphi_Z(t)=\varphi_X(t)\varphi_Y(t)$
* $\varphi^{(k)}(0)=i^kE[X^k]$ 如果矩存在 并且特征函数可导

**定理**（逆转公式）：$F(x)$ 是分布函数 $\varphi(t)$ 是特征函数 则有
$$F\left(x_{2}\right)-F\left(x_{1}\right)=\lim_{r\to\infty}\frac{1}{2\pi}\int_{-r}^{r}\frac{e^{-ix_{1}}-e^{-ix_{2}}}{it}\varphi\left(t\right)dt.$$
根据逆转公式 我们可以从特征函数唯一的导出分布函数
这意味着 **特征函数和分布函数相互唯一确定**

### 多元特征函数
**多元特征函数定义** : 若随机向量 $(\xi_1, \ldots, \xi_n)$ 的分布函数为 $F(x_1, \ldots, x_n)$，其特征函数定义为：
$$
\phi(t_1, \ldots, t_n) = E(e^{i(t_1 \xi_1 + \ldots + t_n \xi_n)}) = \int_{-\infty}^{\infty} \cdots \int_{-\infty}^{\infty} e^{i(t_1 x_1 + \ldots + t_n x_n)} \, dF(x_1, \ldots, x_n)
$$

**唯一性定理**：分布函数 $F(x_1, \ldots, x_n)$ 由其特征函数唯一决定。

**多元特征函数的性质：**
1. $\phi(t_1, \ldots, t_n)$ 在 $\mathbb{R}^n$ 中一致连续，且 $|\phi(t_1, \ldots, t_n)| \leq \phi(0, \ldots, 0) = 1$
2. 如果 $(\xi_1, \ldots, \xi_n)$ 的特征函数为 $\phi(t_1, \ldots, t_n)$，则 $\eta = a_1 \xi_1 + \ldots + a_n \xi_n = a' \xi$ 的特征函数为：$$
   \phi_\eta(t) = \phi(a_1 t, \ldots, a_n t) = \phi(ta)
   $$
3. 若 $E(\xi_1^{k_1} \cdots \xi_n^{k_n})$ 存在，则：$$
   \left. \frac{\partial^{k_1 + \cdots + k_n}}{\partial t_1^{k_1} \cdots \partial t_n^{k_n}} \phi(t_1, \ldots, t_n) \right|_{t_1 = \ldots = t_n = 0} = i^{k_1 + \cdots + k_n} E(\xi_1^{k_1} \cdots \xi_n^{k_n})
   $$
4. 若 $\xi_j$ 的特征函数为 $\phi_{\xi_j}(t_j)$，$j = 1, 2, \ldots, n$，则随机变量 $\xi_1, \ldots, \xi_n$ 相互独立的充要条件为：$$
   \phi(t_1, \ldots, t_n) = \phi_{\xi_1}(t_1) \cdots \phi_{\xi_n}(t_n)
   $$
5. 若以 $\phi(t_1, \ldots, t_n, u_1, \ldots, u_m)$，$\phi(t_1, \ldots, t_n)$ 及 $\phi(u_1, \ldots, u_m)$ 分别记随机向量 $(\xi_1, \ldots, \xi_n)$，$(\eta_1, \ldots, \eta_m)$ 和 $(\xi_1, \ldots, \xi_n, \eta_1, \ldots, \eta_m)$ 的特征函数，则 $(\xi_1, \ldots, \xi_n)$ 与 $(\eta_1, \ldots, \eta_m)$ 独立的充要条件是：对一切实数 $t_1, \ldots, t_n$ 及 $u_1, \ldots, u_m$ 有：
   $$
   \phi(t_1, \ldots, t_n, u_1, \ldots, u_m) = \phi(t_1, \ldots, t_n) \phi(u_1, \ldots, u_m)
   $$
6. 若 $Y = AX + b$，则随机向量 $Y$ 的特征函数为：
   $$
   \phi_Y(t) = E(e^{it'(AX + b)}) = e^{it'b} \phi_X(A't)
   $$

## 特征函数视角下的多元正态分布
### 多元正态分布定义
**定义**：假设 $X = (X_1, \ldots, X_n)'$。若 $X$ 具有密度函数：
$$
f(x) = (2\pi)^{-n/2} |\Sigma|^{-1/2} \exp\left\{ -\frac{1}{2} (x - \mu)' \Sigma^{-1} (x - \mu) \right\}
$$
则称 $X$ 具有均值为 $\mu$、协方差矩阵为 $\Sigma$ 的 $n$ 维多元正态分布。

**等价定义**：假设 $X = (X_1, \ldots, X_n)'$。若 $X$ 具有矩母函数：
$$
\psi_X(t) = \exp\left\{ t'\mu + \frac{1}{2} t' \Sigma t \right\}
$$
或者特征函数：
$$
\phi_X(t) = \exp\left\{ it'\mu - \frac{1}{2} t' \Sigma t \right\}
$$
则称 $X$ 具有均值为 $\mu$、协方差矩阵为 $\Sigma$ 的 $n$ 维多元正态分布，记作 $X \sim N_n(\mu, \Sigma)$。 想要证明这个问题,需要从$Y \sim N_n(0, I)$ 的特征函数出发,使用变换$X = \Sigma^{1/2} Y + \mu$ 变形特征函数,就可以得到上面的形式.

### 多元正态分布的性质

1. **线性变换**：若 $X \sim N_n(\mu, \Sigma)$，则 $Y = AX + b \sim N_n(A\mu + b, A\Sigma A')$
   **证明**：$Y$ 的特征函数为：
   $$
   \begin{aligned}
   \phi_Y(t) &= E(e^{it'(AX + b)}) = e^{it'b} \phi_X(A't) \\
   &= e^{it'b} \exp\left\{ i(A't)'\mu - \frac{1}{2} (A't)' \Sigma (A't) \right\} \\
   &= \exp\left\{ it'(A\mu + b) - \frac{1}{2} t' (A\Sigma A') t \right\}
   \end{aligned}
   $$
   因此 $Y \sim N_n(A\mu + b, A\Sigma A')$。

2. **独立正态变量的线性组合**：假设 $X_1, \ldots, X_k$ 是独立的，且每个 $X_i \sim N_n(\mu_i, \Sigma_i)$，$i = 1, \ldots, k$。假设 $a_1, \ldots, a_k$ 是标量，定义：
   $$
   Y = a_1 X_1 + \ldots + a_k X_k
   $$
   则 $Y \sim N_n(\mu^*, \Sigma^*)$，其中 $\mu^* = \sum_{i=1}^{k} a_i \mu_i$ 且 $\Sigma^* = \sum_{i=1}^{k} a_i^2 \Sigma_i$。这可以通过使用矩母函数来证明。

3. **边际分布**：若 $X \sim N_n(\mu, \Sigma)$，将 $X$ 分块为 $X = \begin{pmatrix} X_1 \\ X_2 \end{pmatrix}$，其中 $X_1$ 是 $r \times 1$，$X_2$ 是 $(n-r) \times 1$。将 $\mu$ 分块为 $\mu = \begin{pmatrix} \mu_1 \\ \mu_2 \end{pmatrix}$，其中 $\mu_1$ 是 $r \times 1$，$\mu_2$ 是 $(n-r) \times 1$。类似地将 $\Sigma$ 分块为：
   $$
   \Sigma = \begin{pmatrix} \Sigma_{11} & \Sigma_{12} \\ \Sigma_{21} & \Sigma_{22} \end{pmatrix}
   $$
   其中 $\Sigma_{11}$ 是 $r \times r$，$\Sigma_{12}$ 是 $r \times (n-r)$，$\Sigma_{21} = \Sigma_{12}'$ 是 $(n-r) \times r$，$\Sigma_{22}$ 是 $(n-r) \times (n-r)$。$X_1$ 的边际分布为 $X_1 \sim N_r(\mu_1, \Sigma_{11})$。这可以通过在 $X_2$ 对应的 $t_j$ 处放入零来使用矩母函数证明。

4. **条件分布**：若 $X \sim N_n(\mu, \Sigma)$，使用 3) 中的分块，则：
   $$
   X_1 | X_2 = x_2 \sim N_r\left( \mu_1 + \Sigma_{12} \Sigma_{22}^{-1} (x_2 - \mu_2), \Sigma_{11.2} \right)
   $$
   其中 $\Sigma_{11.2} = \Sigma_{11} - \Sigma_{12} \Sigma_{22}^{-1} \Sigma_{21}$。

5. **独立性条件**：若 $X \sim N_n(\mu, \Sigma)$，将 $X$ 分块为 $X = \begin{pmatrix} X_1 \\ X_2 \end{pmatrix}$，其中 $X_1$ 是 $r \times 1$，$X_2$ 是 $(n-r) \times 1$。将 $\mu$ 分块为 $\mu = \begin{pmatrix} \mu_1 \\ \mu_2 \end{pmatrix}$，其中 $\mu_1$ 是 $r \times 1$，$\mu_2$ 是 $(n-r) \times 1$。类似地将 $\Sigma$ 分块为：
   $$
   \Sigma = \begin{pmatrix} \Sigma_{11} & \Sigma_{12} \\ \Sigma_{21} & \Sigma_{22} \end{pmatrix}
   $$
   其中 $\Sigma_{11}$ 是 $r \times r$，$\Sigma_{12}$ 是 $r \times (n-r)$，$\Sigma_{21} = \Sigma_{12}'$ 是 $(n-r) \times r$，$\Sigma_{22}$ 是 $(n-r) \times (n-r)$。则 $X_1$ 和 $X_2$ 独立当且仅当 $\Sigma_{12} = 0$。

6. **完整特征**：如果 $X \sim N_n(\mu, \Sigma)$，则 $X$ 的所有边际分布、条件分布以及 $X$ 的分量的线性组合都是多元正态分布。

   **注意**：上述定理的逆命题不成立。例如，如果所有边际分布都是多元正态分布，这并不意味着联合分布是多元正态分布。

7. **唯一性**：多元正态分布完全由其均值向量和协方差矩阵表征。这意味着一旦指定了均值向量和协方差矩阵，MVN 的密度和矩母函数就完全确定了。

8. **对称性**：如果 $X \sim N_n(\mu_x, \Sigma_x)$ 且 $Y \sim N_m(\mu_y, \Sigma_y)$，$X$ 和 $Y$ 独立，则：
   $$
   \begin{pmatrix} X \\ Y \end{pmatrix} \sim N_{n+m}(\mu, \Sigma)
   $$
   其中 $\mu = \begin{pmatrix} \mu_x \\ \mu_y \end{pmatrix}$ 且 $\Sigma = \begin{pmatrix} \Sigma_x & 0 \\ 0' & \Sigma_y \end{pmatrix}$。

9. **中心性**：如果 $X \sim N_n(\mu, \Sigma)$，则 $E(X) = \text{mode}(X) = \text{median}(X) = \mu$。


### 识别多元正态分布
**定理**：假设 $X = (X_1, \ldots, X_n)'$ 具有形式为：
$$
f(x) = c \exp\{-Q/2\}
$$
的密度，其中 $\exp\{-Q/2\} \propto \exp\left\{ -\frac{1}{2} (x - \mu)' \Sigma^{-1} (x - \mu) \right\}$ 且 $c$ 是归一化常数。则 $X \sim N_n(\mu, \Sigma)$。

---

**示例**：假设 $X = (X_1, X_2)'$ 具有形式为 $f(x) = c \exp\{-Q/2\}$ 的密度，其中 $Q = x_1^2 + 2x_1 x_2 + 4x_2^2 + 2x_1$。$X$ 的分布是什么？

我们知道 $X$ 必须是多元正态分布。要找到 $E(X)$，我们有：
$$
\begin{aligned}
\frac{\partial Q}{\partial x_1} &= 2x_1 + 2x_2 + 2 = 0 \\
\frac{\partial Q}{\partial x_2} &= 2x_1 + 8x_2 = 0
\end{aligned}
$$
解这些方程得到 $x_1 = -4/3$ 和 $x_2 = 1/3$。因此 $\mu = (-4/3, 1/3)'$。

要找到 $\Sigma^{-1}$ 的元素，我们查看 $Q$ 中的二次项。令：
$$
\Sigma^{-1} = \begin{pmatrix} \sigma^{(11)} & \sigma^{(12)} \\ \sigma^{(12)} & \sigma^{(22)} \end{pmatrix}
$$
因此：
$$
x' \Sigma^{-1} x = (x_1, x_2) \begin{pmatrix} \sigma^{(11)} & \sigma^{(12)} \\ \sigma^{(12)} & \sigma^{(22)} \end{pmatrix} (x_1, x_2)' = \sigma^{(11)} x_1^2 + 2\sigma^{(12)} x_1 x_2 + \sigma^{(22)} x_2^2
$$
对于我们的问题，$\sigma^{(11)} = 1$，$2\sigma^{(12)} = 2$ 且 $\sigma^{(22)} = 4$。因此：
$$
\Sigma^{-1} = \begin{pmatrix} 1 & 1 \\ 1 & 4 \end{pmatrix}
$$
对 $\Sigma^{-1}$ 求逆得到：
$$
\Sigma = \begin{pmatrix} 4/3 & -1/3 \\ -1/3 & 1/3 \end{pmatrix}
$$

检查线性项，我们有：
$$
-2\Sigma^{-1}\mu = -2 \begin{pmatrix} 1 & 1 \\ 1 & 4 \end{pmatrix} \begin{pmatrix} -4/3 \\ 1/3 \end{pmatrix} = -2 \begin{pmatrix} -1 \\ 0 \end{pmatrix} = \begin{pmatrix} 2 \\ 0 \end{pmatrix}
$$
所以 $-2x'\Sigma^{-1}\mu = 2x_1$，这是 $Q$ 中的线性项。因此：
$$
X \sim N_2\left( \begin{pmatrix} -4/3 \\ 1/3 \end{pmatrix}, \begin{pmatrix} 4/3 & -1/3 \\ -1/3 & 1/3 \end{pmatrix} \right)
$$
## 特征函数视角下的三大分布
### 卡方分布

**定义**：随机变量 $X$ 被称为具有 $n$ 个自由度的中心卡方分布（写作 $X \sim \chi^2(n)$），如果 $X$ 具有密度：
$$
f(x) = \frac{1}{\Gamma(n/2)} \left(\frac{1}{2}\right)^{n/2} x^{n/2 - 1} e^{-x/2}
$$
其中 $\Gamma(\alpha) = \int_0^\infty y^{\alpha - 1} e^{-y} dy$。

随机变量 $X$ 的矩母函数 (MGF) $\psi_X(t)$ 定义为：
$$
\psi_X(t) = E(e^{tX}) = \int_{-\infty}^{\infty} e^{tx} f(x) dx
$$
如果 $X$ 是离散的，则积分被求和替代。

**定理**：如果 $X \sim \chi^2(n)$，则 $\psi_X(t) = (1 - 2t)^{-n/2}$。

**定理**：假设 $Z_1, \ldots, Z_n$ 是独立同分布的 $N(0, 1)$ 随机变量。定义：
$$
X = \sum_{i=1}^{n} Z_i^2
$$
则 $X \sim \chi^2(n)$。

---

### 非中心卡方分布

**定义**：设 $Z_1, \ldots, Z_n$ 是独立的，$Z_i \sim N(\mu_i, 1)$。则 $W = \sum_{i=1}^{n} Z_i^2$ 具有 $n$ 个自由度、非中心参数 $\gamma = \frac{1}{2} \sum_{i=1}^{n} \mu_i^2$ 的非中心卡方分布。我们写作 $W \sim \chi^2(n, \gamma)$。

非中心卡方分布在假设检验中出现，特别是在对线性模型中备择假设下检验统计量分布感兴趣的情况下。

**定理**：假设 $Y_1, \ldots, Y_n$ 是独立的，$Y_i \sim N(\mu_i, \sigma^2)$，$i = 1, \ldots, n$。定义：
$$
X = \frac{1}{\sigma^2} \sum_{i=1}^{n} Y_i^2
$$
则 $X \sim \chi^2(n, \gamma)$，其中 $\gamma = \frac{1}{2\sigma^2} \sum_{i=1}^{n} \mu_i^2$。

**非中心卡方分布的性质**：

1. 如果 $X \sim \chi^2(n, \gamma)$，则：
   $$
   \psi_X(t) = (1 - 2t)^{-n/2} \exp\left\{ \frac{2\gamma t}{1 - 2t} \right\}
   $$
   这可以通过使用非中心卡方密度的定义并交换积分和求和的顺序来证明。

2. 如果 $X \sim \chi^2(n, \gamma)$，则 $E(X) = n + 2\gamma$ 且 $\text{Var}(X) = 2n + 8\gamma$。这可以使用 1) 中的矩母函数证明。

3. 如果 $X \sim \chi^2(n, \gamma)$ 且 $\gamma = 0$，则这对应于具有 $n$ 个自由度的中心卡方随机变量。即，$X \sim \chi^2(n, 0) = \chi^2(n)$。

---

### t分布

**定义**：假设 $X \sim N(0, 1)$，$Y \sim \chi^2(n)$，且 $X$ 和 $Y$ 独立。定义随机变量：
$$
T = \frac{X}{\sqrt{Y/n}}
$$
则 $T$ 被称为具有 $n$ 个自由度的 t 分布。我们写作 $T \sim t(n)$。

**非中心 t 分布**：假设 $X \sim N(\mu, 1)$ 且 $Y \sim \chi^2(n)$，且 $X$ 和 $Y$ 独立。定义随机变量：
$$
W = \frac{X}{\sqrt{Y/n}}
$$
则 $W$ 被称为具有 $n$ 个自由度、非中心参数 $\mu$ 的非中心 t 分布。我们写作 $W \sim t(n, \mu)$。如果 $\mu = 0$，则 $W$ 简化为具有 $n$ 个自由度的中心 t 分布。

---

### F分布

**定义**：假设 $X_1 \sim \chi^2(n_1, \gamma_1)$ 且 $X_2 \sim \chi^2(n_2, \gamma_2)$，且 $X_1$ 和 $X_2$ 独立。定义随机变量：
$$
F = \frac{X_1 / n_1}{X_2 / n_2}
$$
则 $F$ 被称为具有 $(n_1, n_2)$ 个自由度、非中心参数 $(\gamma_1, \gamma_2)$ 的双重非中心 F 分布。我们写作 $F \sim F(n_1, n_2, \gamma_1, \gamma_2)$。

a) 如果 $\gamma_2 = 0$，则 $F$ 被称为非中心 F 分布。我们表示为$F \sim F(n_1, n_2, \gamma_1)$

b) 如果 $\gamma_1 = 0$ 且 $\gamma_2 = 0$，则 $F$ 被称为中心 F 分布。我们表示为 $F \sim F(n_1, n_2)$。

中心 F 分布在嵌套线性模型的假设检验中出现。在这种情况下，原假设下检验统计量的分布通常具有中心 F 分布。非中心 F 分布来自备择假设下检验统计量的分布。备择假设下检验统计量的分布对于功效计算很重要。

**F 分布的性质**：如果 $F \sim F(n_1, n_2, \gamma)$，则：

a) $E(F) = \frac{n_2(n_1 + 2\gamma)}{n_1(n_2 - 2)}$，$n_2 > 2$

b) $\text{Var}(F) = 2 \left(\frac{n_2}{n_1}\right)^2 \frac{(n_1 + 2\gamma)^2 + (n_1 + 4\gamma)(n_2 - 2)}{(n_2 - 2)^2 (n_2 - 4)}$，$n_2 > 4$

通过在 a) 和 b) 中设置 $\gamma = 0$，可获得中心 F 的均值和方差公式。
