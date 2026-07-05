---
title: "线性回归基础：线性模型、最小二乘估计与回归诊断"
title_en: "Linear Regression Basics: Linear Models, Least Squares, and Diagnostics"
date: 2023-09-04 23:00:57 +0800
categories: ["Data Science & Statistics", "Statistical Modeling & Inference"]
tags: ["Learning Notes", "Statistics", "Regression", "Linear Models"]
author: Hyacehila
excerpt: "整理线性模型、最小二乘估计、回归诊断、模型评价、方差分析和广义最小二乘。"
excerpt_en: "Covers linear models, least squares estimation, regression diagnostics, model evaluation, ANOVA, and generalized least squares."
mathjax: true
hidden: true
permalink: '/blog/2023/09/04/linear-regression-basics-notes/'
---
## 引论
线性统计模型是现代统计学使用最广泛的模型之一 也是非常多的统计学研究的基础；这是因为现实世界存在大量的线性依赖关系和大量可以转换线性的依赖关系；线性的模型非常容易建立和处理，我们在引论部分仅仅介绍一些简单的例子，方便我们后面理解各种处理方法

在引论的开头 我们要知道建立线性统计模型的过程与目的

建立一个线性统计模型 可以让我们可以了解自变量和因变量之间的关系；了解自变量之间存在的潜在的联系，最后借助模型实现我们最需要的预测功能,决策,理解关联等功能；

当然建立一个模型并不是那么的容易，我们需要不断的检验 修正前面使用的各种方法，最后直到我们满意；

**回归分析是一个经典的统计问题，我们不应该缺少统计的作图思维，见[安斯库姆四重奏](/blog/2026/01/14/anscombes-quartet/)**
### 线性回归模型

#### 简单介绍
在现实世界中，存在着大量这样的情况:两个变量例如 X 和Y有一些依赖关系， 由X可以部分地决定Y的值，比如身高和体重 气温和城市耗电量；我们不能准确的确定（和精确的函数并不一样） 这种关系被我们称为**相关关系**
对于相关关系的研究，回归模型就是非常重要的一个部分

在上面的例子中 Y称为响应变量 X称为预报变量 我们知道X部分决定Y 因此可以给出这样的一个模型，他同时包含了X对Y的决定作用和那些没有被考虑的因素和随机因素
$$Y=f(X)+e$$
这里$f(X)$表示那些决定作用 $e$是随机误差 我们有理由要求$E(e)=0$

当$f(X)$取特殊的线性函数$f(X)=\beta_{0}+\beta_{1}X$ 时 我们得到这一部分内容要研究的线性回归模型 我们称$f(X)=\beta_{0}+\beta_{1}X+e$ 是线性回归模型 里面没确定的两个系数时回归系数 他们都需要通过观测来确定

在运用适当的统计方法确定这些回归系数后 我们称得到的结果为线性回归方程

在实际的应用中 自变量往往不止一个 所以很容易给出线性回归的多元形式
$$Y=\beta_0+\beta_1X_1+\cdots+\beta_{p-1}X_{p-1}+e$$
#### 矩阵形式
我们对$n$元的线性回归模型进行$n$次观测 得到观测值
$$x_{i1},\cdotp\cdotp\cdotp,x_{i,p-1},y_i,\quad i=1,\cdotp\cdotp\cdotp,n$$
他们满足
$$y_{i}=\beta_{0}+x_{i1}\beta_{1}+\cdots+x_{i,p-1}\beta_{p-1}+e_{i} $$
引入矩阵记号
$$\mathbf{y}=\left|\begin{array}{c}y_1\\y_2\\\vdots\\y_n\end{array}\right|,\mathbf{X}=\left|\begin{array}{cc}1&x_{11}\cdots x_{1,p-1}\\1&x_{21}\cdots x_{2,p-1}\\\vdots&\vdots&\vdots\\1&x_{n1}\cdots x_{n,p-1}\end{array}\right|,\mathbf{\beta}=\left|\begin{array}{c}\beta_0\\\beta\mathbf{\lambda}\\\vdots\\\beta_{p-1}\end{array}\right|,\mathbf{e}=\left|\begin{array}{c}e_1\\e_2\\\vdots\\\vdots\\e_n\end{array}\right|,$$
得
$$y=X\beta+e$$
在矩阵形式下，$y$称为观测向量 $X$称为模型矩阵 $\beta$是未知参数向量 $e$是随机误差向量 

随机误差一般要求满足以下的条件（Gauss-Markov假设）
* $E(e_{i})=0$ 误差均值为0
* $Var(e_{i})=\sigma^{2}$ 误差项等方差
* $Cov(e_{i},e_{j})=0$.  误差彼此不相关

#### 线性化的回归模型
对于函数
$$Q_{t}=aL_{t}^{b}K_{t}^{c}$$
作对数操作得
$$
\ln\left(Q_{t}\right)=\ln a+b\ln\left(L_{t}\right)+c\ln\left(K_{t}\right) 
$$
换元就得到线性模型
$$y_{t}=\ln(Q_{t}),x_{t1}=\ln(L_{t}),x_{t2}=\ln(K_{t})$$
线性化的核心就在于换元 我们在后面会经常用到

### 方差分析模型
回归分析模型在研究变量之间的依赖关系，研究的多是连续变量 
本节的不同之处在于我们研究示性变量作为自变量 他往往表示某种效应是否存在 只取 0 1两个值 ；这样的模型对我们研究两个或者多个因素效应的大小非常有意义；它一般被称为方差分析模型。
#### 单变量方差分析模型
我们用一个例子来描述我们这一节想要叙述的模型 方便理解

我们想要研究三种小麦的品种的优劣 因此安排了六块完全一样的土地，每两个土地播种一种小麦，进行一样的管理,这是一个简单的实验

我们用$y_{ij}$描述$i$种小麦$j$块土地的产量，很容易得到如下的分解
$$y_{ij}=\mu+\alpha_{i}+e_{ij},$$
其中因变量 均值$\mu$ 影响因素$\alpha_{i}$  随机误差都被包含在模型内
矩阵化得
$$\begin{bmatrix}y_{11}\\y_{12}\\y_{21}\\y_{22}\\y_{22}\\y_{31}\\y_{32}\end{bmatrix}=\begin{vmatrix}1&1&0&0\\1&1&0&0\\1&0&1&0\\1&0&1&0\\1&0&1&0\\1&0&0&1\\1&0&0&1\end{vmatrix}\begin{bmatrix}\mu\\\alpha_{1}\\\alpha_{2}\\\alpha_{3}\\\end{bmatrix}+\begin{vmatrix}e_{11}\\e_{12}\\e_{21}\\e_{21}\\e_{22}\\e_{31}\\e_{31}\end{vmatrix}$$

 非常自然的可以用以下的形式进行表示
 $$y=X\beta+e$$
 我们能看到矩阵形式上，方差分析模型和线性回归模型有类似的形式
#### 多变量方差分析模型
 我们自然的单因素方差模型向多因素方差模型推广，在前面的例子中增加两种不同的化肥，自然的推广得
 $$y_{ij}=\mu+\alpha_{i}+\beta_{i}+e_{ij}$$
 其中$\alpha_{i}$ 有三种取法 $\beta_{i}$有两种取法 我们非常自然的推广设计矩阵$X$ 有
$$\boldsymbol{X}=\left|\begin{matrix}1&1&0&0&1&0\\1&1&0&0&1&0\\1&1&0&0&0&1\\1&1&0&0&0&1\\1&0&1&0&1&0\\1&0&1&0&1&0\\1&0&1&0&0&1\\1&0&1&0&0&1\\1&0&0&1&1&0\\1&0&0&1&1&0\\1&0&0&1&0&1\\1&0&0&1&0&1\end{matrix}\right|.$$
明显的 前面提到的矩阵表示不需要发生变化；多因素的方差模型和多元的线性回归模型也是有相似的形式，这意味着我们的后面可以采用类似的手法来对这两种不同的模型进行处理
#### 协方差分析模型
协方差分析（Analysis of Covariance, ANCOVA）是一种结合了**方差分析（ANOVA）** 与**回归分析（Regression）** 的统计模型，用于在控制一个或多个连续型协变量（covariates）的影响后，比较不同组别（分类自变量）在因变量上的均值差异。

一个典型的协方差分析模型包括均值项,处理效应项,以及回归协变量项和残差. **本质上,ANCOVA 是 ANOVA 的扩展版本**,在 ANOVA 基础上加入了对协变量的线性调整. 他有斜率齐的假设,也就是各组的系数一致且固定.

$$y_{ij} = \mu + \alpha_i + \gamma x_{ij} + \varepsilon_{ij}, \quad i = 1,2,\ j = 1,2,3$$

ANCOVA 本质上是**带分类变量的线性回归模型** 因此也可以使用多元回归的角度来进行解释,只是两者强调的效应不同,一个侧重组之间的比较,一个侧重对变量效应的估计.

**这里引入ANCOVA实际上是无关紧要的,我们在研究多元回归以及One-hot编码的时候自然就可以理解这个问题 [广义线性回归](/blog/2024/05/24/generalized-linear-regression-notes/) 的“分类自变量与虚拟变量问题”一节** 


**我们这里虽然介绍方差分析模型可以用回归分析类似的方法处理，但是实际上它有着自己的处理方法，我们会在试验设计中详细的研究关于方差分析的问题，这里就省略了** [试验设计方法](/blog/2024/02/29/experimental-design-methods-notes/)

## 回归分析概述
### 概述
这里对回归分析进行简要的概述 更加细节的东西单独学习
**回归分析考虑变量之间近似的函数关系**
我们考察的是一种联系 但是并没有到严格决定的程度 
我们使用因变量和自变量作为回归分析中变量的命名，但是实际上不存在因果关系
回归模型的一般形式为
$$Y=f(X_1,\cdotp\cdotp\cdotp,X_p)+e$$
如果我们已知$f$的形式 但是其中的参数未知 则称为参数回归 反之我们称为非参数回归 属于非参数统计的范畴
一般情况下 我们研究线性的回归函数 形式为
$$f(x_1,\cdotp\cdotp\cdotp,x_p)=b_0+b_1x_1+\cdotp\cdotp\cdotp+b_px_p$$
### 应用
回归分析的应用或者说使用的步骤一般有以下几个
* 描述：作散点图等 对原始数据进行描述性的分析，揭示样本的规律
* 估计：估计回归函数 揭示隐藏的关系
* 预测：基于回归函数进行预测
* 控制：基于回归函数 我们希望把因变量控制在一定范围内，如何选定自变量的值比较合适
### 一些解释
* 只有自变量处于合理范围内，回归方程才有价值
* 我们应该尽量减少除自变量以外的因素对因变量的影响再应用回归方程
* 如果自变量和因变量都是观测世界所得，那么他揭示的是一种自然规律，如果人工干涉规律则不存在
* 回归方程应当尽量避免外推使用，如确实需要 则要控制外推幅度并注意外推是否合理
* 在用于预测的情况下 回归方程不得逆转使用，而是要建立新的回归方程（控制则允许）
* 一元线性回归中 系数反应了自变量影响的幅度，但是在多元回归中，由于交互作用的存在，这种分析缺少实际价值

## 回归参数的估计1（最小二乘估计）
我们现在已经知道了线性回归模型的基本形态 现在的问题已经非常明显了 我们应该怎么根据观测的数据来计算回归模型里面的参数 这就是这一章需要处理的事情 当然我们知道 方差分析模型其实就是一种特殊的线性回归模型 所以实际上这一章的内容也可以用来处理方差分析模型 
这章我们实际上介绍了整个线性统计模型中最核心的东西

对于我们前面已经给出的线性统计模型的形式
$$\mathbf{y}=\mathbf{X}\mathbf{\beta}+\mathbf{e}$$
我们认为其满足Gauss-Markov假设 也就是
$$E(e)=0,\quad\mathrm{Cov}(e)=\sigma^2I_n$$
这就是现在我们要研究的最基本的线性统计模型 
现在我们需要来研究参数向量$\beta$ 了
### 最小二乘估计（LSE）
其中普通最小二乘估计为OLS
#### 核心内容
得到待估计的参数向量的最基本也是最常用的方法是最小二乘法
也就是找到合适的参数向量$\beta$ 使得偏差向量$e=y-X\beta$ 的平方和$\parallel y-x\beta\parallel^2$ 最小   这就是最小二乘法执行的思想 这个要求是存在合理性的（统计学本身只要求合理，而非确定的某一个答案）
**还有很多其他的最小化方法，对应着很多其他的估计形式**
展开我们需要最小化的偏差平方和 
$$Q(\beta)=y^{\prime}y-2y^{\prime}X\beta+\beta^{\prime}X^{\prime}X\beta.$$
这就是多元函数极小化的问题 直接使用数学分析中的极值理论 求偏导后要求偏导数为0 得到正则方程（组） *这个偏导操作需要借助矩阵微商的知识*
$$X^{\prime}X\beta=X^{\prime}y.$$
这个方程组有唯一解的要求$rank(X)=p$ 也就是$\beta$的维数 我们总是认为确实满足这个条件 最后得到方程的解为
$$\hat{\beta}=(X^{\prime}X)^{-1}{X}^{\prime}y.$$
我们能够证明此时的$\hat{\beta}$不仅仅是驻点 而且保证了极小化  
现在 我们能得到经验回归方程
$$\hat{Y}=\hat{\beta}_{0}+\hat{\beta}_{1}X_{1}+\cdots+\hat{\beta}_{p-1}X_{p-1},$$
注意 经验回归方程并不是我们认为的真实关系 还需要进行进一步的统计检验 我们会在后面的章节中再介绍这些关于检验的内容 本章更加侧重估计的进行
#### 应用举例
对于一元的线性回归问题 $y_{i}=\alpha+\beta x_{i}+e_{i},i=1,\cdots,n$  我们进行$n$次观测 得到各个相关的矩阵和向量为
$$X=\begin{bmatrix}1&x_1\\1&x_2\\\vdots&\vdots\\1&x_n\end{bmatrix},\beta=\begin{bmatrix}\alpha\\\beta\end{bmatrix},y=\begin{bmatrix}y_1\\y_2\\\vdots\\y_n\end{bmatrix}$$
计算正则方程可以得到
$$\left.\left(\begin{matrix}n&\Sigma x_i\\\Sigma x_i&\Sigma x_i^2\end{matrix}\right.\right)\left(\begin{matrix}\alpha\\\beta\end{matrix}\right)=\left(\begin{matrix}\Sigma y_i\\\Sigma x_iy_i\end{matrix}\right),$$
最后能化简得到答案为
$$\begin{aligned}\hat{\beta}&=\frac{\sum x_i\mathbf{y}_i-\sum y_i\overline{x}}{\sum x_i^2-n\overline{x}^2}\\\hat{\alpha}&=\overline{y}-\hat{\beta}\overline{x}\end{aligned}$$
这就是我们整个一元回归的最小二乘估计 和高中数学中涉及的部分是一样的
#### 中心化与标准化
这两种操作本身都是存在统计学意义的
##### 中心化
修正原本的回归模型为以下的形式
$$y_{i}=\alpha+(x_{i1}-\bar{x}_{1})\beta_{1}+\cdots+(x_{i,p-1}-\bar{x}_{p-1})\beta_{p-1}+e_{i}$$
事实上 完成中心化以后 我们可以改写线性回归模型为以下的形式
$$\mathbf{y}=a\mathbf{1}_{n}+X_{c}\mathbf{\beta}+e$$
此时我们的设计矩阵为
$$\boldsymbol{X}_{c}=\begin{bmatrix}x_{11}-\bar{x}_{1}&x_{12}-\bar{x}_{2}&\cdots&x_{1,p-1}-\bar{x}_{p-1}\\x_{21}-\bar{x}_{1}&x_{22}-\bar{x}_{2}&\cdots&x_{2,p-1}-\bar{x}_{p-1}\\\vdots&\vdots&&\vdots\\x_{n1}-\bar{x}_{1}&x_{n2}-\bar{x}_{2}&\cdots&x_{n,p-1}-\bar{x}_{p-1}\\\end{bmatrix}$$
**我们实现了什么？ 
我们把回归系数和回归常数执行了分离**
还是前面的一套操作 我们可以得到回归的结果为
$$\begin{cases}\hat{\boldsymbol{\alpha}}=\bar{\boldsymbol{y}},\\\hat{\boldsymbol{\beta}}=({\boldsymbol{X}}_c^{\prime}\mathbf{X}_c)^{-1}\mathbf{X}_c^{\prime}\mathbf{y}.\end{cases}$$
实际上中心化起到的作用就是分离了回归系数和回归常数 
在实际的研究中 我们关系回归系数远大于回归常数 这就是我们中心化的意义
##### 标准化
我们这里介绍最主要使用的**z-score标准化** 
$$\begin{aligned}s_j^2&=\sum_{i=1}^n(x_{ij}-\bar{x}_j)^2\\\\z_{ij}&=\frac{x_{ij}-\bar{x}_j}{s_j}.\end{aligned}$$
在减去均值以后 我们再将所有的涉及数据除以了他们的标准差
**标准化我们能得到两个比较有用的结果**
第一
$$R=Z^{\prime}Z=(r_{ij}).$$
$$r_{ij}=\frac{\sum_{k=1}^{n}(x_{ki}-\overline{x}_{i})(x_{kj}-\overline{x}_{j})}{s_{i}s_{j}},$$
这意味着 进行标准化的设计矩阵可以直接运算得到自变量的相关矩阵$R$
第二
标准化后的数据消去了回归自变量单位和取值范围的差异，回归系数的估计值变得更加容易进行统计分析和具备直观意义

#### 估计的期望和方差
对于最小二乘估计 $\hat{\beta}=\left(X^{\prime}X\right)^{-1}X^{\prime}y$  我们有
* $E(\hat{\beta})=\beta;$
* $\mathrm{Cov}(\hat{\beta})=\sigma^2(X^{\prime}X)^{-1}.$ 其中$\sigma^{2}$是偏差$e$的方差 是经常用到的一个量

补充定义：
* 如果估计量是观测值的线性函数 那么称他为线性估计
* 最佳线性无偏估计（BLUE）在对这个参数的全部线性无偏估计中，这个估计是最小方差的（有效性解释了BLUE定义合理）

**对于满足Gauss-Markov假设的线性回归问题 最小二乘估计是BLUE** 
**这为最小二乘法提供了合理性的解释**
#### 更多的性质
前面的性质都是在满足Gauss-Markov假设的线性回归问题 这个基础上进行的 下面我们要求$e\sim N(0,\sigma^2I)$ 给出更多的性质 

当$e\sim N(0,\sigma^2I)$ 时 给出定理
* $\hat{\beta}\sim N(\beta,\sigma^2(X^{\prime}X)^{-1});$
* $\frac{\mathrm{RSS}}{\sigma^2}\sim\chi_{n-p}^2;$
* $\beta\text{ 与 RSS 相互独立.}$

对于中心化后的线性回归模型
* $E\left(\stackrel{\wedge}{\alpha}\right)=\alpha,\quad E\left(\hat{\beta}\right)=\beta,$ $\text{这里 }\hat{\alpha}=\overline{y},\hat{\beta}=(X_{c}^{\prime}X_{c})^{-1}X_{c}^{\prime}\mathbf{y}.$
* $\left.\text{ Cov}\left[\begin{array}{c}{\hat{\alpha}}\\{\hat{\beta}}\\\end{array}\right.\right]=\sigma^{2}\left[\begin{array}{cc}{\frac{1}{n}}&{0}\\{0}&{(X_{c}^{\prime}X_{c})^{-1}}\\\end{array}\right].$
* 如果进一步的要求$e\sim N(0,\sigma^2I)$  有 $\hat{\alpha}\sim N\left(\alpha,\frac{\sigma^{2}}{n}\right),\hat{\beta}\sim N(\beta,\sigma^2(X_c^{\prime}X_c)^{-1}),$  并且两者相互独立
中心化还是把回归常数项和回归系数执行了分离

#### 最小二乘估计下的残差
定义：拟合值向量$\hat{y}=X\hat{\beta}$  那么带入参数向量的最小二乘计算结果又
$$\hat{\mathbf{y}}=\boldsymbol{X}(\boldsymbol{X}^{\prime}\boldsymbol{X})^{-1}\boldsymbol{X}^{\prime}\boldsymbol{y}=\boldsymbol{H}\mathbf{y}~~~H=X(X^{\prime}X)^{-1}X^{\prime}$$
容易看出 矩阵$H$的作用是给观测值带上了帽子 所以他也被称为帽子矩阵

能够验证得到帽子矩阵的性质 对称幂等性
$$H^{'}=H,\quad H^{2}=H$$

那么我们可以把残差向量用这个结果进行表示
$$\hat{e}=y-\hat{y}=(I-H)y=(I-H)e$$

然后我们可以验证得到下面的性质
* $E(\hat{e})=0,{Cov}(\hat{e})=\sigma^2(I-H)$
* $e\sim N(0,\sigma^2I)$ 时$\hat{e}\sim N(0,\sigma^2(I-H))$

事实上我们这里得到的残差的性质还不够好 计算残差的方差得到
$$\mathrm{Var}(\hat{e}_{i}^{})=\sigma^{2}(1-h_{ii}),\text{这里 }h_{ii}\text{为H 的第i个对角元}.$$
因此我们修正这个残差得到学生化残差（T化残差）
$$r_{i}=\frac{\hat{e}^{}_{i}}{\hat{\sigma}\sqrt{1-h_{ii}}}$$

现在我们给出断言 当误差项满足$e\sim N(0,\sigma^2I)$ 学生化残差服从$N(0,1)$
并且拟合值向量$\hat{y}$与残差向量$\hat{e}$相互独立,此时学生化残差将会是一个很好的残差分析工具,我们在后面的分析中会经常使用它.

### 约束最小二乘估计
在前面两个小节的讨论中 我们没有对参数变量$\beta$进行任何的约束 这里我们给出拥有线性约束情况下的约束最小二乘估计
定理：对于满足Gauss-Markov假设的线性回归问题 他在约束$A\beta=b$ 下的约束最小二乘估计为
$$\hat{\beta}_c=\hat{\beta}-(x^{\prime}x)^{-1}A\left(A\left(x^{\prime}x\right)^{-1}A^{\prime}\right)^{-1}(A\hat{\beta}-b),$$
其中 $\hat{\beta}=(X^{\prime}X)^{-1}X^{\prime}y$ 是无约束情况下的最小二乘估计
定理的证明略去 这就是约束最小二乘估计的全部内容了
### 广义最小二乘估计（加权OLS）
在残差分析的部分我们就提到过 很多线性回归模型的误差等方差并且不相关这一条并一定成立 因此我们需要给出对应来的处理方法 除去前面讲解的Box-Cox变换族以外 广义最小二乘法也可以解决这个问题

这个问题也被称为加权最小二乘估计 其中权值就和误差项方差有关
本节需要处理的问题是如下的回归问题 我们假定$\Sigma$已知
$$y=X\beta+e,E(e)=0,\mathrm{Cov}(e)=(\sigma^2\Sigma).$$
事实上本节的实用性不高 更侧重于理论研究的介绍
由于矩阵$\Sigma$已知并且是正定的 所以能得到对角化矩阵$\boldsymbol{\Sigma}=\boldsymbol{P}^{\prime}\boldsymbol{\Lambda P}$ 其中$\Lambda$是特征值矩阵
记$\boldsymbol{\Sigma}^{-\frac{1}{2}}=\boldsymbol{P}^{'}\mathrm{diag}(\lambda_{1}^{-\frac{1}{2}},\cdots,\lambda_{n}^{-\frac{1}{2}})\boldsymbol{P}.$
对原始的线性回归问题左乘$\Sigma^{-\frac{1}{2}}$   得到
$$z=U\beta+\varepsilon,\quad E(\varepsilon)=0,\quad\mathrm{Cov}(\varepsilon)=\sigma^2I,$$
这样问题就转化为了我们前面处理过的最小二乘估计问题 结论也很容易给出
$$\beta^{\star}=(U^{\prime}U)^{-1}U^{\prime}z=(X^{\prime}\Sigma^{-1}X)^{-1}X^{\prime}\Sigma^{-1}y.$$
下面给出广义最小二乘估计的一些基础性质
* $E(\beta^*)=\beta^;$
* $\mathrm{Cov}(\boldsymbol{\beta}^{*})=\sigma^{2}(\boldsymbol{X}^{\prime}\boldsymbol{\Sigma}^{-1}\boldsymbol{X})^{-1}$
* 对于本节开篇提到的问题形式 广义最小二乘估计是BLUE 也就是Guass-Markov定理范围被拓展了

很容易看出来 本节开篇补充的假设其实很难实现 假定$\Sigma$已知并不是一个应用层面好的假设

我们一般还是采用从标准的最小二乘估计入手 从残差分析中得到一些关于误差向量的信息（偏差向量估计误差向量，是可行的） 然后再考虑使用广义最小二乘估计 

当然部分特殊的问题确实让误差向量一些特殊的结构 这时候可以进行特殊问题特殊研究

## 回归参数的估计2（复共线性下的估计）
### 复共线性（Multicollinearity）
最小二乘估计被广泛使用 因为他在线性无偏估计类中有最小方差的特性 但是随着现代计算机技术发展 人们有能力去处理一些超大规模的线性回归问题后 很多情况下最小二乘估计的回归系数严重偏离人们的预先猜测（绝对值过大或者符号和实际意义相违背）

研究表明 这些问题的原因中最核心的部分是自变量之间存在近似线性的关系 也就是复共线性（Multicollinearity）

这一节我们来研究复共线性的存在和他的影响 并且后面开始考虑复共线性的解决
#### 均方误差（MSE）
我们这里介绍一个非常重要的评价估计的标准 事实上均方误差（Mean Squared Errors）已经是现代统计学界用来评估估计的最核心指标 比我们曾经介绍的无偏性有效性都要重要
定义：
$$\begin{aligned}MSE(\tilde{\boldsymbol{\theta}})&=E\parallel\tilde{\boldsymbol{\theta}}-\theta\parallel2\\&=E(\tilde{\boldsymbol{\theta}}-\theta)^{\prime}(\tilde{\boldsymbol{\theta}}-\theta).\end{aligned}$$
定理：
$$M\mathrm{SE}(\bar{\theta})=\mathrm{trCov}(\tilde{\theta})+\parallel E\tilde{\theta}-\theta\parallel^2$$
推论：记$\tilde{\theta}=(\tilde{\theta}_{1},\tilde{\theta}_{2},\cdots,\tilde{\theta}_{p})^{\prime}$ 则
$$\operatorname{trCov}(\tilde{\theta})=\sum_{i=1}^p\operatorname{Var}(\tilde{\theta}_i).$$
$$\parallel E\tilde{\theta}-\theta\parallel^2=\sum_{i=1}^p{(E\tilde{\theta}_i-\theta_i)^2}$$
均方误差是由两个部分组成的 他们分别是偏差平方和（一元时偏差的平方）和分量的方差和（一元时为方差）

这个评估标准是合理 综合考虑了估计量方差和估计偏差的问题 这就是为什么它比无偏和有效性都更为重要 
#### 均方误评估最小二乘法
考虑线性回归模型
$$y=\alpha\mathbf{1}+X\mathbf{\beta}+e,E(e)=\mathbf{0},\mathrm{Cov}(e)=\sigma^2\mathbf{I}.$$
我们前面已经给出了最小二乘估计（分离回归系数和回归常数）
$$\hat{\alpha}^{\Lambda}=\bar{y}=\frac{1}{n}\sum_{i-1}^{n}y_{i},\\\\\hat{\beta}=(x'x)^{-1}X'y.$$
我们这里直接计算MSE能知道（无偏估计后半部分为0，只用研究方差和）
 $$MSE(\hat{\boldsymbol{\beta}})=\Delta_{1}=\sigma^{2}\mathrm{tr}(X^{\prime}X)^{-1}=\sigma^2\sum_{i=1}^p\frac1{\lambda_i}.~~~\text{其中}\lambda_{i}\text{是}(X^{\prime}X)^{-1}\text{的特征值}$$
 这告诉我们 如果$(X^{\prime}X)^{-1}$ 的特征值有一个非常小，那么从均方误差的角度考虑最小二乘法不是好的估计
 此时参数估计向量$\hat{\beta}$ 会有某个绝对值过大的分量
 
 这个我们前面给出的Gauss-Markov定理并不违背 我们只是说最小二乘估计是所有线性无偏估计中方差最低的 但是这个最低也非常大 这其实就意味着 所有的线性无偏估计类在这个情况下都不是好的估计
 
此时设计矩阵的列向量之间存在近似的线性关系 等价于 回归自变量存在线性关系  我们称这样的线性回归模型存在复共线性

**完全的复共线性导致模型无法求解，不完全的强复共线性会导致部分系数的求解不精确（极度不精确），我们希望尽可能降低复共线性再进行OLS**
#### 度量复共线性
##### 特征根判别法
研究矩阵$X^{'}X$ 的特征根情况 
如果有一个或者几个特征根趋近于0 则线性代数理论保证了存在线性组合
此时存在复共线性
##### 条件数判别法
我们一般使用方阵$(X^{\prime}X)$ 的条件数来衡量复共线性的大小 定义其条件数为
$$k=\frac{\lambda_1}{\lambda_p}$$
也就是最大特征值和最小特征值的比 从经验角度考虑 $k<100$可以认为不存在复共线性 $100<k<1000$认为存在较强的复共线性 $k>1000$认为存在非严重的复共线性

那个非常小的特征值对应的特征向量可以体现复共线性关系

假设最小的特征值对应的特征向量为$\phi$ 那么有
$$X\varphi\approx0$$
$X$就是使用列向量表示 或者说 X表示回归自变量（向量）
##### 方差膨胀因子（VIF） 
Variance Inflation Factor
定义为 
$$VIF_{j}=\frac{1}{1-{R_j}^2}$$
其中$R_{j}$ 是以变量$j$ 为因变量 剩下的作为自变量 进行最小二乘回归得到的可决定系数 
一般认为 $VIF>10$ 就意味着存在比较强的复共线性
### 逐步回归
逐步回归的思想是在最小二乘上进行修改
通过移除复共线性的变量来让最小二乘法规避最小二乘法造成的过高MSE
我们会在第五章详细介绍这一点 研究多个回归方程的选择
### PC回归
全称 Principal Component Regression
主成分回归  (PCR)

主成分回归本身非常好理解 我们现在可以直接介绍主成分回归的执行方法
1. 执行PCA 得到主成分
2. 选择主要的主成分 去掉一些贡献比较小的主成分
3. 对剩下的主成分进行最小二乘回归
4. 将主成分变量还原为原始变量

现在我们来介绍一些主成分估计的性质
* 主成分估计是有偏估计
* 当设计矩阵存在复共线性关系时，适当选择主成分可以降低均方误差
事实上主成分估计的难点就是PCA的执行和理解 
本身回归就是非常普通的最小二乘估计

**主成分回归本质上是一种降维消除复共线性后的回归方法，对应的诸如因子分析，结构方程模型的降维手段都可以使用类似主成分回归的思想来进行回归**
### Ridge 回归
岭回归
#### 模型简介
现在我们来开始研究一类线性回归方法 增加正则项来处理一些最小二乘回归中遇到的问题
从最小二乘估计的结果出发
$$\hat{\beta}=(X^\top X)^{-1}X^\top y$$
当收集到的数据存在复共线性的时候 设计矩阵很可能不满秩的 这导致求逆矩阵出现问题 而Ridge就进行了如下的处理
$$\hat{\beta}(k)=(X^{\prime}X+kI)^{-1}X^{\prime}y,$$
这个增加矩阵$kI$就被称为岭 他是一处凸起 其中的超参数$k$是我们需要研究的
同时它对应的最小化函数应该为
$$\begin{aligned}\text{minimize}\|y-X\beta\|_2^2+\lambda\|\beta\|_2^2\end{aligned}$$
没错 它等价于对原本的优化问题部分增加了一个$L2$的正则项作为惩罚 
我们给出一些理论的推导，毕竟$L2$正则项还是可以保证可导的
$$\mathcal{L}(\hat{w})=||X\hat{w}-Y||_2^2+\lambda||\hat{w}||_2^2=(X\hat{w}-Y)^T(X\hat{w}-Y)+\lambda\hat{w}^T\hat{w}$$
对$\omega$求导有
$$\frac{\partial\mathcal{L}(\hat{w})}{\partial\hat{w}}=2X^TX\hat{w}-2X^TY+2\lambda\hat{w}=0$$
解方程得到
$$(X^TX+\lambda I)\hat{w}=X^TY$$
也就是
$$\hat{w}=(X^TX+\lambda I)^{-1}X^TY$$

现在我们着手来研究一些岭估计的性质
岭估计是一种有偏估计
$$\begin{aligned}E\hat{\boldsymbol{\beta}}(k)&=(X^{\prime}\boldsymbol{X}+k\boldsymbol{I})^{-1}\boldsymbol{X}^{\prime}\boldsymbol{E}\boldsymbol{y}\\&=(X^{\prime}\boldsymbol{X}+k\boldsymbol{I})^{-1}\boldsymbol{X}^{\prime}\boldsymbol{X}\boldsymbol{\beta}\\&\neq\boldsymbol{\beta},\end{aligned}$$
岭估计在某种情况下有着更小的均方误差
存在$k>0$使得
$$MSE(\hat{\beta}(k))<MSE(\hat{\beta}).$$
岭估计的特色是均衡 他并不是一种稀疏模型 不会把一些系数降为0（最多接近0）它对共线性问题的处理方案是均衡共线性的变量的回归系数 最后实现降低MSE的作用 也就是Ridge倾向于把权重分散给各个特征
Ridge更适合特征之间相关较大并且特征数量有限并没有压缩特征的计划
#### 超参数确定
对于Ridge估计 我们需要明确一个概念 那就是随着$k$的增大 Ridge和OLE的偏差增大 RSS随之增大 （不影响才是MSE是在减少的）
##### 岭迹法
Ridge存在的核心是为了解决共线性导致部分系数过大 部分系数过小的问题 我们可以很容易的把各个系数随着$k$变化的图画出来 也就是岭迹法
![线性统计模型 6](/assets/images/probability-statistics-notes/linear-regression-basics-notes-01.png)
能看出 随着$k$的增加 系数趋近于平稳状态 我们应该大致选择刚进入平稳状态的点作为我们的估计 因为此时能尽量同时兼顾平稳性和RSS
##### 控制残差平方和
RSS会随着$k$的增大而增大 我们选择一个常数$c$ 控制岭回归系数满足 $RSS(k)<cRSS(LS)$
##### 控制方差膨胀因子
和逐步回归里面提到的一样
我们通过检验是否继续存在复共线性来决定这里的压缩是否充足
##### Hoerl-Kennard 公式
$$\dot{\hat{k}}=\frac{(p-1)\hat{\sigma}^{2}}{\sum_{i=1}^{p}(\hat{\beta}_{i}^{*})^{2}}$$
其中$p$ 是参与回归的自变量个数
剩下的变量都是LS回归得到的估计结果
### LASSO回归
#### 模型简介
全称 Least Absolute Selection and Shrinkage Operator  
最小绝对值收敛和选择算子算法回归
在岭估计为优化函数添加$L2$正则的影响下 LASSO估计应运而生 他为优化函数添加了一个$L1$正则作为惩罚
$$\operatorname*{minimize}_{}\|y-X\beta\|_2^2+\lambda\|\beta\|_1$$
对于LASSO回归 由于添加了$L1$正则项 继续研究解析解就比较复杂了 我们一般会采用数值方法研究 比如梯度下降

LASSO模型的特色是他是一种稀疏模型，会快速的把一些参数压缩为0 让他离开我们的回归模型 这点和Ridge区别很大
事实上 LASSO起到了非常强的特征选择作用 如果有特征之间存在强相关 LASSO也倾向于选择其中的一个 并且把其他的降为0
因此LASSO更适合特征之间相关性小 但是特征数量庞大的问题
#### 超参数确定
LASSO的超参数确定比Ridge稍微复杂一点 因为涉及一个特征数量被压缩的问题 我们一般要结合下面两个图来实现
![线性统计模型 7](/assets/images/probability-statistics-notes/linear-regression-basics-notes-02.png)
这张图揭示了变量选择的情况 纵轴是各个变量的大小 下面的横轴表示$log(\lambda)$ 的变化 上面表示了此时特征剩余的数量
![线性统计模型 8](/assets/images/probability-statistics-notes/linear-regression-basics-notes-03.png)
这张图体现了MSE随着$log(\lambda)$  的变化情况 综合MSE和变量压缩作用是我们需要做到的
我们实际上要通过机器学习领域的交叉验证来确定最后的$\lambda$ 
具体采用几折的交叉验证一般计算机会帮助我们决定
最后我们能得到一个最低MSE的$\lambda$ 同时能得到$\lambda$ 自身的标准误
如果最低MSE不能保证比较好的压缩效果 那么一个标准误的缩放保证更好的压缩也是可行的

### ElasticNet 回归
弹性网络回归
#### 模型简介
我们都可以为优化函数分别添加两种正则了 同时添加两种正则项凭什么不可以 因此 ElasticNet很快就被数学家研究出来了 其对优化函数进行了如下修正 此时我们需要确定两个超参数
$$\min Q(\beta)=|y-X\beta|^2+\lambda\alpha\sum_{j=0}^n|\beta_j|+\lambda(1-\alpha)\sum_{j=0}^n\beta_j^2\color{red}$$
ElasticNet产生的目的就是为了综合LASSO和Ridge两者的优点 
通过结合 L1, L2 范数使得 ElasticNet 既保留了LASSO 容易产生稀疏解的特性，也结合了 L2 正则化岭回归的性质，同时解决了 LASSO 方法在多变量相关性大时产生多个解的问题，在变量选择时可以有效剔除无关变量，保留相关性大的有关变量。 
#### 超参数确定
ElasticNet 模型需要同时确定两个超参数 但是我们的核心思想没有发生变化
对于只需要确定$\lambda$ 的情况 我们前面在LASSO中已经介绍了过了处理方法 也就是K折交叉验证寻找最小MSE
那么增加了变量$\alpha$的情况呢？
首先选取 0.01 到 0.99 以0.01为步长的 $\alpha$ 混合超参数
在确定的$\alpha$ 的情况下 我们还是能使用K折交叉验证寻找到最合适的$\lambda$ （最小MSE） 和对应的标准误 
我们可以比对 每一个$\alpha$下面的MSE 找到MSE最小的那一个
这样就实现了两个超参数的选择
事实上在R语言中 ElasticNet LASSO 都使用glment包进行处理
他们在处理方法上是完全一样的 仅仅增加了一个超参数
事实上 这就是在指定的参数域（$[0,1]$） 上进行网格搜索 只是这里重新叙述了一遍过程
