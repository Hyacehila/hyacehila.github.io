---
title: "R 经典统计：估计、检验、线性模型与 GLM"
title_en: "R Classical Statistics: Estimation, Tests, Linear Models, and GLMs"
date: 2024-09-05 12:20:32 +0800
categories: ["Programming", "R"]
tags: ["Learning Notes", "R", "Classical Statistics"]
author: Hyacehila
excerpt: "整理描述性统计、参数估计、假设检验、线性模型、方差分析、非线性回归和 GLM。"
excerpt_en: "Covers descriptive statistics, estimation, hypothesis testing, linear models, ANOVA, nonlinear regression, and GLMs."
mathjax: true
hidden: true
permalink: '/blog/2024/09/05/r-classical-statistics-learning-notes/'
---

## EDA与描述性统计

这里我们将介绍EDA技术中那些以描述性统计内容为主的部分 也会涉及一些其他的部分，EDA和描述性统计作为整个统计分析中比较小的一个部分，确实没有必要单独设计一本书来描述它 融入整个统计学详解是最好的选择了

## 概率分布图（PDF）

虽然我们一直说PDF和CDF有着类似的作用 但是我们还是使用PDF为主 在作图的时候也是这样进行的，我们基本不会研究CDF图 因为它不够直观

### 常用分布的密度函数图

了解总体分布的形态，有助于把握样本的基本特征.我们先通过具体的例子考查第三章中提到的一些常用分布的概率函数(对于离散分布指分布律，对于连续分布指其密度函数)的图形. 我们使用那些关于PDF的函数来实现我们的图形绘制

下面我们给出一些R代码帮助我们理解常见的PDF图

二项分布

```r
n<-20
p<-0.2
k<-seq(0,n)
plot(k,dbinom(k,n,p),type='h', main='Binomial distribution, n=20, p=0.2',xlab='k')
```

泊松分布

```r
lambda<-4.0
k<-seq(0,20)
plot(k,dpois(k,lambda),type='h', main='Poisson distribution, lambda=5.5',xlab='k')
```

几何分布

```r
p<-0.5
k<-seq(0,10)
plot(k,dgeom(k,p),type='h', main='Geometric distribution, p=0.5',xlab='k')
```

超几何分布

```r
N<-30
M<-10
n<-10
k<-seq(0,10)
plot(k,dhyper(k,N,M,n),type='h', main='Hypergeometric distribution,
     N=30, M=10, n=10',xlab='k')
```

负二项分布

```r
n<-10
p<-0.5
k<-seq(0,40)
plot(k, dnbinom(k,n,p), type='h',
     main='Negative Binomial distribution,
     n=10, p=0.5',xlab='k')
```

正态分布

```r
curve(dnorm(x,0,1), xlim=c(-5,5), ylim=c(0,.8),col='red', lwd=2, lty=3)
curve(dnorm(x,0,2), add=T, col='blue', lwd=2, lty=2)
curve(dnorm(x,0,1/2), add=T, lwd=2, lty=1)
title(main="Gaussian distributions")
legend(par('usr')[2], par('usr')[4], xjust=1, c('sigma=1', 'sigma=2', 'sigma=1/2'),
       lwd=c(2,2,2), lty=c(3,2,1),col=c('red', 'blue', par("fg")))
```

t分布

```r
curve(dt(x,1), xlim=c(-3,3), ylim=c(0,.4),col='red', lwd=2, lty=1)
curve(dt(x,2), add=T, col='green', lwd=2, lty=2)
curve(dt(x,10), add=T, col='orange', lwd=2, lty=3)
title(main="Student T distributions")
legend(par('usr')[2], par('usr')[4], xjust=1, c('df=1', 'df=2', 'df=10', 'Gaussian distribution'),
lwd=c(2,2,2,2), lty=c(1,2,3,4),
col=c('red', 'blue', 'green', par("fg")))
```

卡方分布

```r
curve(dchisq(x,1), xlim=c(0,10), ylim=c(0,.6), col='red', lwd=2)
curve(dchisq(x,2), add=T, col='green', lwd=2)
curve(dchisq(x,3), add=T, col='blue', lwd=2)
title(main='Chi square Distributions')
```

F分布

```r
curve(df(x,1,1), xlim=c(0,2), ylim=c(0,.8), lty=1)
curve(df(x,3,1), add=T, lwd=2,lty=2)
curve(df(x,6,1), add=T, lwd=2, lty=3)
title(main="Fisher's F")
```

### 直方图和密度函数的估计

#### 直方图

直方图是探索性数据分析的基本工具，它给出了数据的频率分布图形，在组距相等场合下常用宽度相等的长条矩形表示，矩形的高低表示频率的大小；在图形上，横坐标表示所关心变量的取值区间，纵坐标表示频率(或频数)的大小, 这样就得到频数(或频数)直方图

```r
hist(x, breaks = "Sturges", freq = NULL, probability = !freq,
     col = NULL, main = paste("Histogram of" , xname),
     xlim = range(breaks), ylim = NULL,
     xlab = xname, ylab, axes = TRUE, nclass = NULL)
```

其中breaks用来指定分割（整数就是区间数） col指明颜色 freq指明是否使用频数直方图

#### 核密度估计

```r
density(x, bw = "nrd0",
        kernel = c("gaussian", "epanechnikov", "rectangular",
                   "triangular", "biweight", "cosine", "optcosine"),
        n = 512, from, to)
```

kernel 决定光滑函数的使用 默认就是正态；n给出等间隔的核密度估计点数量 from与to分别给出需要计算核密度估计的左右端点

#### 例子

```r
N <- 100000
n <- 100
p <- .9
x <- rbinom(N,n,p)
hist(x, xlim=c(min(x),max(x)), probability=T,
     nclass=max(x)-min(x)+1, col='lightblue',
     main='Binomial distribution, n=100, p=.5')
lines(density(x,bw=1), col='red', lwd=3)
```

## 描述性统计分析

描述性统计分析是EDA的一个重要组成部分 我们在这里介绍它 从数据类型的差距来分别阐述

关于图形化的知识可以参考[R Visualization](/blog/2024/03/15/r-visualization-learning-notes/)
### 单组数据的描述性统计分析

#### 图形考察

单组数据的分布可以通过上面介绍的直方图以及核密度曲线和箱线图考查.正态性检验一般用QQ图完成，

```r
library(DAAG)
data(possum)
fpossum <- possum[possum$sex=="f",]
par(mfrow=c(1,2))
attach(fpossum)
hist(totlngth,breaks=72.5+(0:5)*5, ylim=c(0,22),
       xlab="total length", main="A:Breaks at 72.5,77.5…")

stem(fpossum$totlngth)

boxplot(fpossum$totlngth)

qqnorm(fpossum$totlngth, main="Normality Check via QQ Plot")
qqline(fpossum$totlngth, col='red')
```

#### 数据考察

常用的统计量我们在Obsidian中都介绍过了 这里简单介绍一些代码实现就足够了

```r
library(DAAG)
data(possum)
fpossum <- possum[possum$sex=="f",]

## 趋势部分
summary(fpossum$totlngth) #汇总分析
fivenum(fpossum$totlngth) #五数分布
quantile(fpossum$totlngth) #分位数
median(fpossum$totlngth) #中位数
max(fpossum$totlngth)
min(fpossum$totlngth)
mean(fpossum$totlngth) #极大 极小 均值

#离散部分
max(fpossum$totlngth)-min(fpossum$totlngth)
IQR(fpossum$totlngth)
sd(fpossum$totlngth)
var(fpossum$totlngth)
mad(fpossum$totlngth)

#偏度峰度
library(fBasics)
skewness(fpossum$totlngth)
kurtosis(fpossum$totlngth)
```

特别的 统计包**fBasics**中的函数`basicStats( )`可以提供几乎所有的描述性统计计算 ,**pastecs**包中有一个名为`stat.desc()`的函数也有一样的作用

### 多组数据的描述性统计分析

#### 图形考察

对于多组数据的考察 我们比较常用的有 plot matplot boxplot几种方式 （plot函数可以作用多组数据，此时等价于pairs） 我们用一个例子来说明就足够了

```r
n<-10
d<-data.frame(y1 = abs(rnorm(n)),
y2 = abs(rnorm(n)),
y3 = abs(rnorm(n)),
y4 = abs(rnorm(n)),
y5 = abs(rnorm(n)) )
plot(d)
matplot(d, type = 'l', ylab = "", main = "Matplot")
boxplot(d)
```

#### 数据考察

我们需要考察的数据并不比单组数据多多少 就是原本的函数更换了一些使用方法进行推广 再稍微补充一点点内容

特别的，针对数据框而言，我们很可能需要复用那些针对某列的描述性统计分析函数，我们比较建议使用函数 `apply() and sapply() aggregate()` 他们可以增加编程的效率 这三个函数参考 [将函数应用于矩阵和数据框](/blog/2024/09/05/r-basic-learning-notes/)

```r
## 汇总分析和一些针对数据框的操作函数
summary(state.x77)
aggregate(state.x77, list(Region = state.region), mean)
aggregate(state.x77, list(Region = state.region, Cold = state.x77[,"Frost"] > 130),mean)

sd(state.x77)
#var函数不可以继续计算方差了

#相关分析
x<-c(44.4, 45.9, 46.0, 46.5, 46.7, 47, 48.7, 49.2, 60.1)
y<-c(2.6, 10.1, 11.5, 30.0, 32.6, 50.0, 55.2, 85.8, 86.8)
cor(x,y)
cor(x,y,method="spearman")
cor(x,y,method="kendall")
cor.test(x,y, method="spearman")
```

`cor（）`函数可以同时计算三种相关性 Pearson Spearman kendall 我们这里对相关分析的叙述简单代过 后面单独进行研究 也可以对相关性进行假设检验

### 分类数据的描述性统计

#### 列联表

如果数据集中对应的变量都是定性变量, 这样的数据称为分类数据. 这种数据常使用表格来描述, 并为进一步的统计分析服务. 我们主要考虑由二元定性数据所构成的二维列联表数据

使用非常简单的代码就可以建立一个列联表 这是创建数据构建列联表

```r
## 本质上就是构造了一个有行名和列名的矩阵，不能能用数据框替代
Eye.Hair <- matrix(c(68,20,15,5, 119,84,54,29, 26,17,14,14, 7,94,10,16),  nrow=4,byrow=T)
colnames(Eye.Hair) <- c("Brown", "Blue", "Hazel", "Green")
rownames(Eye.Hair) <- c("Black","Brown","Red", "Blond")
Eye.Hair
```

也可以从原始数据构造列联表 使用table函数

```r
## table()函数从因子factor中获取频数的函数
## 当接受两个factor的时候，table函数创建一个二维的列联表，高维也可
table(menarche,tanner)
```

对于高维的列联表 `ftable()`函数可以以一种紧凑而吸引人的方式输出多维列联表，参考函数

```r
ftable(table(factorA,factorB,factorC))
```

列联表的边际 是非常重要的一环 我们有函数可以研究它；除此以外 研究频率列联表也相当重要

```r
## 创建列联表
Eye.Hair <- matrix(c(68,20,15,5, 119,84,54,29, 26,17,14,14, 7,94,10,16), nrow=4,byrow=T)
colnames(Eye.Hair) <- c("Brown", "Blue", "Hazel", "Green")
rownames(Eye.Hair) <- c("Black","Brown","Red", "Blond")

## 1 按照行 2 按照列 计算边缘总和
margin.table(Eye.Hair,1)
margin.table(Eye.Hair,2)

## 1 按照行 2 按照列 计算边缘概率
prop.table(Eye.Hair,1)
prop.table(Eye.Hair,2)

#直接获取概率矩阵
Eye.Hair/sum(Eye.Hair)
```

#### 简单的图形描述

条形图是描述列联表的最基础图形 还有很常用的是马赛克图 我们在图形章节中单独介绍 [马赛克图](/blog/2024/03/15/r-visualization-learning-notes/)

```r
data(HairEyeColor)
a <- as.table(apply(HairEyeColor,c(1,2),sum))
barplot(a, legend.text = attr(a, "dimnames")$Hair)
barplot(a, beside = TRUE, legend.text = attr(a, "dimnames")$Hair)
```

## 数据的中心化和标准化

```r
scale(x, center = TRUE, scale = TRUE)
#中心化or标准化函数 x接受矩阵和数据框 返回中心化or标准化后的结果
#第一个参数控制中心化 第二个参数控制缩放
```

## 统计推断

这里 我们介绍经典统计学中关于统计推断的那一部分，也就是包含了参数估计 参数假设检验，还有一些比较简单的非参数假设检验问题 虽然我们花费了整个一个学期来了解他们的基础知识 但是对于R实现 使用一章基本已经足够了

## 参数估计

### 矩法估计和极大似然估计

#### 矩法估计

由辛钦大数定律和科尔莫哥洛夫强大数定理可知,如果总体X的k阶矩存在, 则样本的k阶矩以概率收敛到总体的k阶矩, 样本矩的连续函数收敛到总体矩的连续函数. 这就启发我们可以用样本矩作为总体矩的估计量

我们没有必要在这里研究任何理论 一个例子就足够了 事实上这里我们根本不接触什么新的R知识 只是对就知识进行了一些应用

设总体为参数为λ 的指数分布,其密度函数为

$$
p(x|\lambda)=\lambda\exp^{-\lambda x},\quad x>0
$$

则$\lambda$的矩法估计为

$$
\hat{\lambda}=\frac{1}{\overline{X}}
$$

```r
X<-c(0.59132754,0.12854935,0.46900228,0.29835980,0.24341462, 0.06566637,0.40085536,2.99687123,0.05278912,0.09898594)
lambda<- 1/mean(X)
lambda
```

这就是矩估计所做的 就是原本的数学计算交给R计算了一部分 还是很简单的一部分

#### 极大似然估计（介绍R最优化函数）

极大似然估计基于似然函数运行 所以我们需要自己先计算似然函数 然后使用软件进行极大化的处理 也就是说 本质是我们在R里实现的是一个优化问题 下面我们先来介绍一下R中的优化函数

```r
#optimize( )的调用格式
optimize(f = , interval = , lower = min(interval),
        upper = max(interval), maximum = TRUE,
        tol = .Machine$double.eps^0.25, ...)
```

其中

-   f是似然函数 需要我们定义一个function 在基础知识里介绍过
-   interval是参数θ的取值范 围；lower是θ的下界 upper是θ的上界 给出前者就可以
-   maximum = TRUE是求极大值, 否则(maximum = FALSE)表示求函数的极小值
-   tol是表示求值的精确度 一般默认就可以

**optimize只适用于单参数的最优化问题 不过同时适用于最大和最小值两种问题**

```r
#nlm( )的调用格式
nlm(f, p, hessian = FALSE, typsize=rep(1, length(p)),     fscale=1,print.level = 0, ndigit=12, gradtol = 1e-6,
stepmax = max(1000 * sqrt(sum((p/typsize)^2)), 1000),
steptol = 1e-6, iterlim = 100, check.analyticals = TRUE, ...)
```

**它使用牛顿-拉夫逊算法求函数的最小值点**

```r
#optim( )的调用格式
optim(par, fn, gr = NULL,
method = c("Nelder-Mead", "BFGS", "CG", "L-BFGS-B", "SANN"),
lower = -Inf, upper = Inf, control = list( ), hessian = FALSE, ...)
```

**提供method选项给出的5种方法中的一种进行优化**

后两个都可以用于多维的问题

下面我们介绍一个极大似然估计的例子 它是一维的 似然函数直接给出

```r
f <- function(P){(P^517)*(1-P)^483}
optimize(f,c(0,1),maximum = TRUE)
```

它的运行结果分了两部分
* maximum 是给出的极大似然估计值
* objective 是此时似然函数的函数值

### 单正态总体参数的区间估计

区间估计 是一类比较特殊的问题 他和假设检验有着非常密切的联系 这是因为枢轴量和检验统计量有着比较接近的形式

事实上 很多的区间估计问题就是用假设检验的函数完成的 区间估计的结果是假设检验函数的结果的一部分

不过有些问题的函数 base R并没有提供 事实上 这些内容确实没有必要设计函数来辅助我们的求解，我们后面只介绍一些R提供了函数的区间估计问题

方差未知 研究均值的区间估计才是最常见 R提供了非常简单的算法 数学形式为

$$
\begin{pmatrix}\overline{X}-\frac{S}{\sqrt{n}}t_{1-\frac{\alpha}{2}}(n-1),\overline{X}+\frac{S}{\sqrt{n}}t_{1-\frac{\alpha}{2}}(n-1)\end{pmatrix}
$$

容易知道 我们进行的是一个t检验 R提供了如下的t检验函数

```r
t.test(x, y = NULL,alternative = c("two.sided", "less", "greater"), mu = 0, paired = FALSE, var.equal = FALSE, conf.level = 0.95, ...)
```

-   x，y是用于检验的数据 都给出就是双样本t检验
-   alternative 决定区间的类型
-   mu是均值 只在假设检验中有作用

下面是一个简单的例子

```r
x<-c(175 , 176 , 173 , 175 ,174 ,173 , 173, 176 , 173,179 )
t.test(x)
t.test(x)$conf.int
#可以用conf.int选择只访问置信区间 本质上就是在列表上选择了一部分分量
```

### 两正态总体参数的区间估计

对于两个总体方差未知但是相等的时候 均值差的区间估计的数学形式为

$$
\left((\overline{X}-\overline{Y})\pm t_{1+\frac\alpha2}\sqrt{\frac1n+\frac1m}\sqrt{\frac{(n-1)S_1^2+(m-1)S_2^2}{n+m-2}}\right)
$$

此时我们知道 需要进行的是一个t检验 使用前面介绍的t检验函数就可以 例子如下

```r
x<-c(628,583,510,554,612,523,530,615)
y<-c(535,433,398,470,567,480,498,560,503,426)
t.test(x,y,var.equal=TRUE)
#var.equal需要设定TRUE 此时认为两个总体方差相等
```

研究方差比也是两正态总体比较常见的情况 函数为var.test

```r
var.test(x, y, ratio = 1,
         alternative = c("two.sided", "less", "greater"),conf.level = 0.95, ...)
```

一个例子如下

```r
x<-c(20.5,19.8,19.7,20.4,20.1,20.0,19.0,19.9)
y<-c(20.7,19.8,19.5,20.8,20.4,19.6,20.2)
var.test(x,y)
```

### 单总体比率p的区间估计

在许多实际问题中, 我们经常要去估计在总体中具有某种特性的个体占总体的比例(率) 这一类问题值得单独的研究 非常的重要 数学理论上一般使用大样本得到正态分布来估计 形式如下

$$
\hat{p}\pm z_{1-\frac{\alpha}{2}}\sqrt{\hat{p}(1-\hat{p})/n}-\frac{1}{2n}.
$$

我们有专门的R函数来实现这个目的 **抽样特性近似服从超几何分布 我们可以选择使用正态分布或者二项分布分布来拟合**（后者数学形式也变了）如下

```r
#调用格式
prop.test(x, n, p = NULL,
          alternative = c("two.sided", "less", "greater"),
          conf.level = 0.95, correct = TRUE)
#正态检验是一种近似检验 需要大样本
binom.test(x, n, p = NULL,
          alternative = c("two.sided", "less", "greater"),
          conf.level = 0.95, correct = TRUE)
#二项分布是一种精确检验 不需要大样本 这里我们本质上是调用了二项分布的估计和检验
```

-   x是样本数量 n是总体数量
-   correct是是否使用连续分布近似
-   p是原假设的概率 在假设检验中才有用

### 两总体比率差的区间估计

在大样本的情况下 他们近似服从正态分布 所以可以给出公式有

$$
(\hat{p}_1-\hat{p}_2)\pm z_{1-\frac{\alpha}{2}}\sqrt{\frac{\hat{p}_1(1-\hat{p}_1)}{n_1}+\frac{\hat{p}_2(1-\hat{p}_2)}{n_2}.}
$$

我们只有一种形式（正态近似）来处理这一类问题 参数设置和前面的思路是一样的

```r
like<-c(478, 246)
people<-c(1000, 750)
prop.test(like, people)
```

### 样本容量的确定

这是一类区间估计的反问题 我们给出参数估计的最大允许误差 反过来计算所需要的样本容量；从数学角度上，计算没有任何的变化 不过确实需要一些新的函数来处理问题了

R base 并没有提供任何合适的函数帮助我们处理这一类问题 建议进行手工的推导 使用R中的函数进行一些辅助的计算工作；当然也可以在需要的时候查询有没有什么packages帮我们处理这一类问题

## 参数的假设检验

统计推断的另一重要内容是假设检验，由抽取的样本提供的信息, 构造合适的统计量, 对所提供的假设进行检验；

假设检验工作分为两个大类 参数的假设检验（经典数理统计学习）此时总体的分布已知 我们是研究参数的情况；

非参数的假设检验 此时总体的分布未知 我们对分布类型进行假设检验 在非参数统计中才进行详细的研究，也需要更多的理论支撑才可以更好的理解

### 假设检验中一些重要的理论知识

我们的核心步骤为

1.  给出原假设 获得对应的备择假设
2.  确定显著性水平$\alpha$
3.  研究检验统计量 确定他的分布
4.  给出拒绝原假设的拒绝域（统计量落在拒绝域的时候拒绝原假设）
5.  计算样本点对应的检验统计量的值
6.  判断是否拒绝原假设

假设检验的$p$值是和我们以前学习的假设检验不一样的一个存在；此时我们不事先确定显著性水平，而是计算$p$ 值 通过比较它和常见几个显著性水平的情况来决定是否要拒绝原假设

$p$值越小 越应该拒绝原假设 一般有常见的几个显著性水平 0.1 0.05 0.01 每小于一个 就愈发有拒绝原假设的显著趋势 在后面的假设检验中 我们最重视的就是$p$值

我们在后面还是只介绍一些比较重要的假设检验问题 他们都提供了base R得函数

### 单正态总体参数的检验

方差未知 单正态总体均值的假设检验 我们容易查证 数学形式是一个t检验

$$
T=\frac{\overline{X}-\mu_{0}}{S^{*}/\sqrt{n}}  \sim t(n-1)
$$

此时我们还是调用前面介绍过的函数

```r
salt<-c(490 , 506, 508, 502, 498, 511, 510, 515 , 512)
t.test(salt, mu=500)
```

其中的 mu 就是原假设的均值 这个函数和均值的区间估计的函数一样 事实上 他们进行的就是一样的工作 如果有需要 我们可以进行单侧检验 调整里面的参数就可以了

### 两正态总体参数的检验

关于两个正态总体均值差是否为0的检验问题 也可以直接进行假设检验 我们前面的函数要求方差相等

```r
x<-c(628,583,510,554,612,523,530,615)
y<-c(535,433,398,470,567,480,498,560,503,426)
t.test(x,y,var.equal=TRUE)
#var.equal需要设定TRUE 此时两个总体方差相等 否则我们需要修改参数为FALSE
```

方差比的问题也可以用前面的函数直接计算 原假设还是比为1

```r
x<-c(20.5,19.8,19.7,20.4,20.1,20.0,19.0,19.9)
y<-c(20.7,19.8,19.5,20.8,20.4,19.6,20.2)
var.test(x,y)
```

### 成对数据的t检验

我们使用前面介绍的t.test函数就可以了 只是需要修改一些参数 此时方差相等无所谓了 是配对需要单独声明

```r
x<-c(20.5, 18.8, 19.8, 20.9, 21.5, 19.5, 21.0, 21.2)
y<-c(17.7, 20.3, 20.0, 18.8, 19.0, 20.1, 20.0, 19.1)
t.test(x, y, paired=TRUE)
```

### 单样本比率的检验

我们还是前面的方法 分为了精确检验（二项分布检验）和近似检验（大样本接近正态检验）两种

```r
binom.test(c(7, 5), p=0.4)
#这是另一种调用格式 输入向量 前面是成功次数 后面是失败次数
prop.test(7, 12, p=0.4, correct=TRUE)
#近似检验可能不准 R会在任何需要的时候给出警告
```

### 两样本比率的检验

事实上 非常的简单 和前面参数估计的形式完全一样 原假设是两总体比率相等

```r
like<-c(478, 246)
people<-c(1000, 750)
prop.test(like, people)
```

## 相关分析
## 分类变量的相关性
首先我们需要检验分类变量的独立性，也就是对列联表进行独立性检验

R提供了多种**检验类别型变量独立性**的方法，我们在这里简单介绍
### 卡方独立性检验
可以使用`chisq.test()`函数对二维表的行变量和列变量进行卡方独立性检验，示例如下

```r
compare<-matrix(c(60,32,3,11), nr = 2, dimnames = list(c("cancer", "normal"),c("smoke", "Not smoke")))

chisq.test(compare, correct=TRUE) #检验函数接受一个二维表
```

### Fisher精确检验
前面的卡方独立性检验只允许二维列联表中有百分之二十的类别期望频数小于5 否则会出现警告 此时我们可以使用Fisher精确检验 而不是近似的卡方检验，可以在任意行列数大于等于2的二维列联表上使用，但是不用于$2\times2$

```r
compare<-matrix(c(60,32,3,11), nr = 2, dimnames = list(c("cancer", "normal"),c("smoke", "Not smoke")))

fisher.test(compare, correct=TRUE)
```

### Cochran-Mantel-Haenszel检验
`mantelhaen.test()`函数可用来进行Cochran-Mantel-Haenszel卡方检验，其原假设是，**两个名义变量在第三个变量的每一层中都是条件独立的**。下列代码可以检验治疗情况和改善情况在性别的每一水平下是否独立。

```r
## 构建列联表
mytable <- xtabs(~Treatment+Improved+Sex, data=Arthritis)
## 检验
mantelhaen.test(mytable)
```

### 相关系数的计算
上一节中的显著性检验评估了是否存在充分的证据以拒绝变量间相互独立的原假设。如果可以拒绝原假设，那么你的兴趣就会自然而然地转向用以衡量相关性强弱的相关性度量。**vcd**包中的`assocstats()`函数可以用来计算二维列联表的phi系数、列联系数和Cramer’s V系数。代码示例如下

```r
library(vcd)
mytable <- xtabs(~Treatment+Improved, data=Arthritis)
assocstats(mytable)
```
## Pearson、Spearman和Kendall相关
Pearson积差相关系数衡量了两个定量变量之间的线性相关程度。Spearman等级相关系数则衡量分级定序变量之间的相关程度。Kendall’s Tau相关系数也是一种非参数的等级相关度量。 他也是我们最基本的三种相关系数。可以使用`cor`函数计算，`methon`参数用于选择计算的方法，示例如下

```r
states<- state.x77[,1:6]
#cor函数可以对一个数据框进行操作
cor(states)
```

计算系数之后，我们也应该考虑显著性的问题，可以使用cor.test()函数对单个的Pearson、Spearman和Kendall相关系数进行检验。简化后的使用格式为：

```r
cor.test(x, y, alternative = , method = )
```

## 偏相关系数
偏相关是指在控制一个或多个定量变量时，另外两个定量变量之间的相互关系。你可以使用**ggm**包中的`pcor()`函数计算偏相关系数 函数调用格式为：

```r
pcor(u, S)
```

其中的`u`是一个数值向量，前两个数值表示要计算相关系数的变量下标，其余的数值为条件变量（即要排除影响的变量）的下标。`S`为变量的协方差阵。示例为

```r
library(ggm)
colnames(states)
pcor(c(1,5,2,3,6), cov(states))
```

对应的，我们有假设检验的函数

```r
pcor.test(r, q, n)
```

其中的`r`是由`pcor()`函数计算得到的偏相关系数，`q`为要控制的变量数（以数值表示位置），`n`为样本大小。

## 方差分析
方差分析(analysis of variance, 简写为ANOVA)是工农业生产和科学研究中分析试验数据的一种有效的统计方法.

引起观测值不同(波动)的原因主要有两类: 一类是试验过程中随机因素的干扰或观测误差所引起不可控制的的波动, 另一类则是由于试验中处理方式不同或试验条件不同引起的可以控制的波动.

方差分析的主要工作就是将观测数据的总变异(波动)按照变异的原因的不同分解为因子效应与试验误差，并对其作出数量分析，比较各种原因在总变异中所占的重要程度，以此作为进一步统计推断的依据.

**ANOVA在各种实验和准实验设计的分析中都有广泛应用，是研究因子是分类变量情况下的另一种回归模型**
## 单因子方差分析

### 基本方法

具体的数学模型我们在试验设计中已经介绍过了 这里主要介绍R中应该怎么使用 基本形式为

```r
aov(formula, data=NULL, projections=FALSE,qr=TRUE, contrasts=NULL, ...)
```

其中formula是方差分析公式 data是要使用的数据框 我们在后面会逐步介绍这个公式应该怎么使用

给出一个单因素 五个水平的 方差分析的例子 用这个代码应该能大概理解含义了

```r
X<-c(25.6, 22.2, 28.0, 29.8, 24.4, 30.0, 29.0, 27.5, 25.0, 27.7,
23.0, 32.2, 28.8, 28.0, 31.5, 25.9, 20.6, 21.2, 22.0, 21.2)
A<-factor(rep(1:5, each=4))
#rep函数 times参数是控制整体重复的次数 each是控制每个元素重复的次数
A
miscellany<-data.frame(X, A)
aov.mis<-aov(X~A, data=miscellany)
#X列标识了数据 A列标识了数据对应的因子水平
summary(aov.mis)

plot(miscellany$X ~ miscellany$A)
#绘制分组图形的箱线图，直观的比较差异
plot(miscellany$A, miscellany$X)
#这个和上面是一样的 自变量为因子的plot函数也会绘制boxplot
```

### 均值的多重比较

进行方差分析后发现各效应的均值之间有显著差异, 此时只能知道有某些均值彼此不同, 但无法知道哪些均值不同, 下面的方法帮助我们**找出在进行方差分析时哪些均值是不同的.**

其实就是针对某一个因子A得两个水平进行比较 使用t检验判断 这种t检验和我们在前面使用的t检验没有本质区别

不过 在进行多重t检验的时候 需要调整p值才能保证正常判断 如下

```r
p.adjust.methods
```

他会告诉我们有哪些调整p值的方法 目前使用比较多的是bonferroni 现在 我们进行多重t检验的方法如下

```r
pairwise.t.test(x, g, p.adjust.method=p.adjust.methods,pool.sd=TRUE, ...)
```

`x`是响应变量构成的向量 `g`是分组向量(因子)  `p.adjust.method`是上面提到的调整p值的方法

返回的矩阵中的量都是`p`值

### 同时置信区间: Tukey法

前面我们拒绝了各个水平之间没有差异的假设 并且判断了到底是谁与众不同 **现在我们希望对效应的差做出置信区间** 这其实是另一种判断哪些效应与众不同的方法 Turky 对这里的理论完成了证明

函数的形式为

```r
TukeyHSD(x, which, ordered=FALSE, conf.level=0.95)
```

`x` 是方差分析的结果 `which` 是需要计算比较区间的因子向量 `ordered` 是逻辑值, 如果为"true", 则因子的水平先递增排序 conf.level是置信水平

我们还是用一个简单的例子来解释用法

```r
sales<-data.frame( X=c(23, 19, 21, 13, 24, 25, 28, 27, 20, 18, 19, 15, 22, 25, 26, 23, 24, 23, 26, 27), A=factor(rep(1:5, c(4, 4, 4, 4, 4))) )
#数据集生成
summary(aov(X~A, sales))
#方差分析
pairwise.t.test(sales$X, sales$A, p.adjust.method="bonferroni")
#单组比较
TukeyHSD(aov(X~A, sales))
#计算所有均值差的计算区间 因为我们没改which
plot(TukeyHSD(fit))
#绘图
```

### 方差齐性检验

想要进行方差分析 需要保证下面三个条件

-   可加性（变异可加）
-   独立正态性（水平之间独立，之内正态）
-   方差齐性（不同水平方差相等）

现在我们研究如何对方差齐性做检验 最常用的方法就是Bartlett检验和Levene检验

**事实上，回归分析是有放松方差齐的算法的，但是方差分析不能这样，因此我们一般只在方差分析中研究方差齐的问题**

残差的正态性可以直接参考回归问题进行分析
#### Bartlett检验

函数格式为 参数的含义没必要解释了 和前面很多地方相同

```r
bartlett.test(x, g, ...)
bartlett.test(formula, data, subset, no.action, ...)
```

#### Levene检验

函数格式为 参数同上

```r
leveneTest(x, group)
```

#### 例子

```r
bartlett.test(X~A, data=sales)
library(car)
leveneTest(sales$X, sales$A)
```

两个p值都很大 也就是没有拒绝原假设 而原假设是等方差（简单的作为了原假设）

### 备注

方差分析模型可视为一种特殊的线性模型, 因此方差分析还可以使用线性模型函数`lm( )`, 并用函数`anova( )`提取其中的方差分析表, 因此`aov(formula)`等价于`anova(lm(formula))
`
单因子方差分析还可使用函数`oneway.test( )`, 若各水平下数据的方差相等(使用选项`var.equal=TRUE`), 它等同于使用函数`aov( )`进行一般的方差分析; 若各水平下数据的方差不相等(使用选项`var.equal=FALSE`), 则它使用Welch(1951)的近似方法进行方差分析;

当各水平下的分布未知时（一般默认正态，就用上面的方法了），则采用Kruskal-Wallis秩和检验等方式进行方差分析.他们属于非参数统计范畴 这里不赘述了

## 双因子方差分析

### 没有交互作用的情况

理论证明略去 我们的代码思路如下 函数没有发生变化

```r
juice<-data.frame(
X = c(0.05, 0.46, 0.12, 0.16, 0.84, 1.30, 0.08, 0.38, 0.4, 0.10, 0.92, 1.57, 0.11, 0.43, 0.05, 0.10, 0.94, 1.10, 0.11, 0.44, 0.08, 0.03, 0.93, 1.15), A = gl(4, 6),B = gl(6, 1, 24)
)
#数据建立
juice.aov<-aov(X~A+B, data=juice)
summary(juice.aov)
#方差分析 把公式部分改了
bartlett.test(X~A, data=juice)
bartlett.test(X~B, data=juice)
#两个方差齐性检验
```

### 有交互作用影响的情况下

理论证明略去 我们的代码思路如下 函数没有发生变化

```r
rats<-data.frame(
Time=c(0.31, 0.45, 0.46, 0.43, 0.82, 1.10, 0.88, 0.72, 0.43, 0.45, 0.63, 0.76, 0.45, 0.71, 0.66, 0.62, 0.38, 0.29, 0.40, 0.23, 0.92, 0.61, 0.49, 1.24, 0.44, 0.35, 0.31, 0.40, 0.56, 1.02, 0.71, 0.38, 0.22, 0.21, 0.18, 0.23, 0.30, 0.37, 0.38, 0.29, 0.23, 0.25, 0.24, 0.22, 0.30, 0.36, 0.31, 0.33),
Toxicant=gl(3, 16, 48, labels = c("I", "II", "III")),
Cure=gl(4, 4, 48, labels = c("A", "B", "C", "D")) )
#数据集构建
op<-par(mfrow=c(1, 2))
#设置绘图参数并存储
plot(Time~Toxicant+Cure, data=rats)
#一类特殊的boxplot绘制方法
with(rats,interaction.plot(Toxicant, Cure, Time, trace.label="Cure"))
with(rats, interaction.plot(Cure, Toxicant, Time, trace.label="Toxicant"))
#绘制交互效应图 如果不出现明显交叉就基本认为没有交互作用
rats.aov<-aov(Time~Toxicant*Cure, data=rats)
summary(rats.aov)
#虽然认为没有交互 但是还是进行了带有交互效应的方法分析 上面的函数等价于
## rats.aov<-aov(Time~Toxicant+Cure+Toxicant:Cure, data=rats)
## 只考虑交互效应则为
## rats.aov<-aov(Time~Toxicant:Cure, data=rats)
```

方差分析的结果表示 两个因素影响显著 但是交互作用影响不显著 我们如果对这个数据集进行方差齐性检验 会得到方差不齐的结果 在真实分析中残差的检验是必须要进行的

## 协方差分析

前面两节介绍的方差分析方法中两组或多组均值间比较的假设检验, 其处理因素一般是可以控制的. 但在实际工作中, 有时有些因素无法加以控制, 如何在比较两组或多组均数间差别的同时扣除或均衡这些不可控因素的影响, 可考虑采用协方差分析的方法.

协方差分析(Analysis of Covariance, 简称ancova)是**将线性回归分析与方差分析结合起来的一种统计分析方法**. 其基本思想就是: 将一些对响应变量Y 有影响的变量(指未知或难以控制的因素)看作协变量(covariate), 建立响应变量Y 随协变量X变化的线性回归关系, 并利用这种回归关系把X值化为相等后再对各处理组Y 的修正均值(adjusted means)间差别进行假设检验, 其实质就是从Y 的总的平方和中扣除X对Y 的回归平方和, 对残差平方和作进一步分解后再进行方差分析, 以更好地评价这种处理的效应.

我们直接用函数来解释问题 需要使用程序包HH中的函数 ancova 形式为

```r
ancova(formula, data.in = sys.parent(),x, groups)
```

formula是协方差分析的公式 data.in 是数据 x是方差分析协变量 groups是因子 更多信息参考其他资料

例子如下

```r
feed<-as.factor(rep(c("A","B","C"),each=8) )
Weight_Initial <- c(15,13,11,12,12,16,14,17,17,16,
                    18,18,21,22,19,18,22,24,20,23, 25,27,30,32)
Weight_Increment <-c(85,83,65,76,80,91,84,90,97,90,
                     100,95,103,106,99,94,89,91,83,
                     95,100,102,105,110)
data_feed<-data.frame(feed,Weight_Initial,Weight_Increment)
#数据集构建 其中Weight_Initial是我们想要考虑的协变量
ancova(Weight_Increment ~ Weight_Initial+feed , data=data_feed)
#不考虑交互
ancova(Weight_Increment ~ Weight_Initial*feed , data=data_feed)
#考虑交互
```

## 正交试验设计与方差分析

### 正交表试验

我们下面用一个例子来说明一下正交表试验的数据是怎么被读入数据框之中的

```r
rate<-data.frame(
A=gl(3,3),
B=gl(3,1,9), C=factor(c(1,2,3,2,3,1,3,1,2)),
Y=c(31, 54, 38, 53, 49, 42, 57, 62, 64)
)
#正交试验数据的建立 存储了每个数据对应的各个因子水平
K<-matrix(0, nrow=3, ncol=3, dimnames=list(1:3, c("A","B","C")))
for (j in 1:3)
  for (i in 1:3)
    K[i,j]<-mean(rate$Y[rate[j]==i])
#计算了每个水平的均值（实际上可以使用tapply函数简化 如下）
#K <- tapply(rate$Y, rate$A, mean)
plot(as.vector(K), axes=F, xlab="Level", ylab="Rate")
xmark<-c(NA,"A1","A2","A3","B1","B2","B3","C1","C2","C3",NA)
axis(1,0:10,labels=xmark)
axis(2,4*10:16)
axis(3,0:10,labels=xmark)
axis(4,4*10:16)
lines(K[,"A"]); lines(4:6, K[,"B"]); lines(7:9,K[,"C"])
#因子各个水平 指标均值情况
```

这就是直观方法分析正交试验 实际上已经可以给出最优的指标了 不过没有方差分析那么的严谨

### 正交试验方差分析

我们在R函数的使用上没有发生变换 例子为

```r
rate.aov<-aov(Y~A+B+C, data=rate)
summary(rate.aov)
```

正交试验也可以分析交互作用 方法和前面使用的一样

### 重复实验问题

所谓重复测量方差分析，即受试者被测量不止一次。本节重点关注含一个组内和一个组间因子的重复测量方差分析（这是一个常见的设计）。

因变量是二氧化碳吸收量（uptake）自变量是植物类型 Type 和七种水平的二氧化碳浓度（conc），Type是组间因子，conc是组内因子，Plant是个体标示符号

对于这种涉及重复试验的问题，我们有单独的处理方法，首先我们要求数据结构为。**仍旧为每次观测一行，这一行需要同步给出组内因子，组间因子，以及个体标示符号** 并且方差分析代码变为

```r
## A组内因子 W组内因子 B组间因子
y ~ A + Error (Subject/A) #单因素组内 ANOVA
y ~ B * W + Error (Subject/W) #含单个组内因子（w）和单个组间因子（B）的重复测量ANOVA
```

## 回归分析
相关分析只能得出两个变量之间是否相关, 但却不能回答在两个变量之间存在相关关系时, 它们之间是如何联系的, 即无法找出刻画它们之间因果关系的函数关系. 回归分析就可以解决这一问题

## OLS及衍生
### R表达式中常用的符号
这里是我们第一次接触R的较复杂的函数体系，有一些符号使我们需要理解的。

我们在这里建立对照表来说明那些比较常用的符号

| 符号    | 表达式含义                                                                                       |
| ----- | ------------------------------------------------------------------------------------------- |
| `~`   | 分隔符号，左边为响应变量，右边为解释变量                                                                        |
| `+`   | 分隔预测变量                                                                                      |
| `:`   | 表示预测变量的交互项                                                                                  |
| `*`   | 表示所有可能交互项的简洁方式 建议少用                                                                         |
| `^`   | 表示交互项达到某个次数 例如 $\text{代码} y \sim(x + z + w)^2 \text{可展开为} y\sim x+ z + w + x:z + x:w + z:w$ |
| `-1`  | 删除截距项                                                                                       |
| `I()` | 从算术的角度来解释括号中的元素，避免符号冲突                                                                      |

### 一元线性回归

数学形式没必要浪费实现介绍了 我们这里来介绍最基本但是重要的关于回归的函数，他们也是我们继续学习其他更多的函数的基础

```r
#回归函数
lm(formula, data, subset, weights, na.action,method="qr",
   model=TRUE, x=FALSE, y=FALSE, qr=TRUE, singular.OK=TRUE, contrasts=NULL, offset)

#返回模型的参数
coefficients(object)

#模型参数的置信区间
confint(object, level=0.95, ...)

#汇总分析函数
summary(object)

#返回预测残差
residuals()
rstandard(model, infl=lm.influence(model, do.coef=FALSE),
          sd=sqrt(deviance(model)/df.residual(model)), ...)
rstudent(model, infl=lm.influence(model), do.coef=FALSE)

#列出拟合模型的预测值
fitted()

#预测
predict(object, newdata, interval = "confidence", level = 0.95)

#绘制回归曲线图 一般和plot联合使用
abline(object)

#手动计算p-value
f_statistic <- summary(X.lm)$fstatistic
f_value <- f_statistic[1]
p_value <- pf(f_value, f_statistic[2], f_statistic[3], lower.tail = FALSE)
```

`lm()`函数是求出回归方程的核心函数 其中
-   formula是回归模型的选择
-   data是数据框
-   subset是样本观察的子集
-   weights是用于拟合的加权向量
-   na.action显示数据是否包含缺失值
-   method是指出用于拟合的方法
-   后面的逻辑值是是否返回其值的意思

`summary（）`用来回复整个模型的信息 就是最经典的summary的函数 内容有
-   模型的参数估计
-   模型的假设检验(不包含可以直接引用的方程p值 但是有f值，可以自己设计函数处理这个问题)

### 多项式回归

我们可以轻易的构建一个多项式回归的形式 如下

```r
fit2 <- lm(weight ~ height + I(height^2), data=women)
## I的含义是里面增加了一个算术项 我们构建是一个多元回归
```

这仍旧属于线性回归的范畴，属于其中的多项式回归

实际上，**无论我们在等式的右侧怎么构造变量，只要参数项是线性的，就不影响其线性回归的属性**
### 多元线性回归与变量选择

函数在这里没有形式上的变换 给出一个例子就明白了

```r
lm.reg<-lm(y~x1+x2+x3+x4, data=blood)
```

如果想要研究交互作用，代码应该修改为

```r
fit <- lm(mpg ~ hp + wt + hp:wt, data=mtcars)
```

我们这里额外介绍一下关于变量选择的问题 R提供了一个step函数如下 使用AIC作为变量选择指标

```r
step(object, scope, scale=0,direction=c("both", "backward", "forward"),trace=1, keep=NULL, steps=1000, k=2)
```

参数解释

-   object是线性模型或广义线性模型分析的结果
-   scope意思是是否需要限制模型选择的范围 一般缺省
-   direction是选择方法 向前 向后 还是逐步回归

**这个函数会直接修改模型**

## 回归诊断

其主要内容有：残差分析、影响分析、共线性诊断；我们还是在线性回归这里研究回归诊断的问题，这是因为此时我们的很多研究内容确实只针对于前面介绍的OLS 当然有的内容可以无缝的外推 在其他模型中也可以很好的使用 诊断 尤其是其中的残差分析 是整个回归中非常重要的一环 值得非常深的研究

#### 标准分析方法

残差分析是一个非常庞大的模块，我们这里仅针对一些比较常用的部分给出介绍

最详尽的残差分析应该基于 `residuals（）`函数展开自定义的分析，我们这里不展开介绍。

最基础的残差分析基于 `plot` 函数 他为我们提供四个最常用的残差分析图
```r
fit <- lm(weight ~ height, data=women)
par(mfrow=c(2,2))
plot(fit)
```

包含了模型诊断最常用的图形化工具
-   残差对y的函数
-   残差的qq图检验
-   标准化残差的平方根的分布
-   Cook距离

据此我们分别研究
* 残差和因变量的独立性，也就是是否遗漏了重要的回归项
* 残差的正态性
* 残差是否同方差
* 影响分析问题
#### 影响分析

我们有很多中研究影响的方法 **但是没有任何一种可以直接给出结论 都是供我们残差 具体问题具体分析** 如下

```r
lm.influence(model, do.coef=TRUE)
```
它给出去掉了某个观测点后的模型回归系数 可以用来判断影响

```r
cooks.distance(model, infl=im.influence(model, do.coef=FALSE),
               res=weighted.residuals(model), sd=sqrt(deviance(model)/df.residual(model)),
               hat=infl$hat, ...)
```
cook统计量也是非常常用的判断影响的统计量

```r
dffits(model, infl=..., res=...)
```
这是DFFITS准则

```r
covratio(model, infl=lm.influence(model, do.coef=FALSE),res=weighted.residuals(model))
```
COVRATIO准则

```r
influence.measures(model)
```
一个针对影响的统计量的概括 包括了上面的几种

```r
library(car)
influencePlot(fit, id.method="identify", main="Influence Plot",
              sub="Circle size is proportional to Cook's distance")
```
**car**包提供的一个函数，可以整合离群点、杠杆值和强影响点的信息在一个可视化的图表中
#### 复共线性诊断

比较常见的复共线性诊断方式有 特征值法 kappa值 方差膨胀因子VIF 函数如下

```r
eigen(x, symmetric, only.values=FALSE, EISPACK=FALSE)
#计算矩阵的特征值 辅助判断复共线性
kappa(x, exact=FALSE, ...)
#计算矩阵的kappa值 也就是条件数 100以上就是强相关 30以上中度相关
vif(lmobj, digits=5)
#计算方差膨胀因子VIF 这个函数来自于DAGG包 10意味着强相关
```

#### 回归的综合检验
`gvlma()`函数由Pena和Slate（2006）编写，能对线性模型假设进行综合验证，同时还能做偏斜度、峰度和异方差性的评价。换句话说，它给模型假设提供了一个单独的综合检验（通过/不通过） 来自包 **gvlma**

使用起来非常方便
```r
library(gvlma)
gvmodel <- gvlma(fit)
summary(gvmodel)
```
## GLM回归

这里我们来讨论GLM回归的问题

### 关于广义线性模型

广义线性模型(Generalized Linear Model)的一种是通常的正态线性模型的推广，它要求响应变量只能通过线性形式依赖于解释变量。这里可以理解为：GLM 保留线性预测子的结构，但用连接函数把响应变量期望和解释变量线性组合联系起来，同时允许响应变量来自指数分布族。

R语言直接提供了拟合和计算广义线性模型的函数`glm( )`, 其调用格式为

```r
log<-glm(formula, family=family.generator,data=data.frame)
```

-   `formula`为拟合公式 其意义与线性模型相同;
-   `family`为分布族, 包括正态分布(gaussian)、二项分布(binomial)、泊淞分布(poission)和伽玛分布(gamma), 分布族还可通过选项link=来指定使用的连接函数，参考下表
-   data为数据框.

GLM 中常见分布族与默认连接函数可以按下表理解：

| 分布类型 | 默认连接函数 | 常见模型 |
| --- | --- | --- |
| `binomial` | `logit` | Logit 回归 |
| `gaussian` | `identity` | 普通线性回归 |
| `gamma` | `inverse` | Gamma GLM |
| `inverse.gaussian` | `1/mu^2` | 逆高斯 GLM |
| `poisson` | `log` | 泊松回归 |
| `quasi` | `identity` 与常数方差 | 准分布 GLM |
| `quasibinomial` | `logit` | 准分布 Logit |
| `quasipoisson` | `log` | 准分布泊松 |

几个分布族调用方式例子如下

```r
#正态分布 恒等连接
fm <- glm(formula, family = gaussian(link = identity), data = data.frame)

#二项分布 logit连接 是logistics回归的形式
log<-glm(formula, family = binominal(link = logit),data = data.frame)

#Possion分布
log<-glm(formula, family = poisson(link = log),data = data.frame)

#Gamma分布
log<-glm(formula, family = gamma(link = inverse),data = data.frame)

```

### 更多的参考函数
R中扩展的Logistic回归和变种如下所示

* 稳健Logistic回归 robust包中的glmRob()函数可用来拟合稳健的广义线性模型，包括稳健Logistic回归。当拟合Logistic回归模型数据出现离群点和强影响点时，稳健Logistic 回归便可派上用场。
* 多项分布回归 若响应变量包含两个以上的无序类别（比如，已婚/寡居/离婚），便可使用mlogit包中的mlogit()函数拟合多项Logistic回归。
* 序数Logistic回归 若响应变量是一组有序的类别（比如，信用风险为差/良/好），便可使用rms包中的lrm()函数拟合序数Logistic回归。

R提供了基本泊松回归模型的一些有用扩展
* 可以拟合允许时间段变化的泊松回归模型，我们的处理习惯是转换为下面的拟合模型形式$\log_\mathrm{e}\left(\frac{\lambda}{time}\right)=\beta_0+\sum_{j=1}^p\beta_jX_j$
* pscl包中的zeroinfl()函数可做零膨胀泊松回归
* robust包中的glmRob()函数可以拟合稳健广义线性模型，含稳健泊松回归
## 非线性回归模型

### 内在线性回归

最为经典的内在线性回归是多项式回归；其中的普通多项式回归进行变形以后用前面介绍的多元线性回归方法就可以求解了 我们下面介绍正交多项式的计算方法 回归方法没有变化

函数形式为

```r
poly(x, ..., degree = 1, coefs = NULL)
#计算正交多项式 degree是阶数
```

### 内在非线性回归

这里涉及最优化相关的问题 相关的函数并不唯一， 一个个介绍，我们先建立我们认为的回归函数，然后优化获取参数

```r
nls(formula, data = parent.frame(), start, control = nls.control(), algorithm = "default", trace = FALSE, subset, weights, na.action, model = FALSE)
#nls函数对于实现内在非线性回归非常的实用

nlm(f, p, hessian = FALSE,
typsize=rep(1, length(p)), fscale=1, print.level = 0, ndigit=12, gradtol = 1e-6, stepmax = max(1000 * sqrt(sum((p/typsize)^2)), 1000), steptol = 1e-6, iterlim = 100, check.analyticals = TRUE, ...)
#nlm函数也可以处理这个问题 当然它本身就是用来处理最优化问题的 所以需要转化问题形式
```

例子有

```r
cl<-data.frame(
X=c(rep(2*4:21, c(2, 4, 4, 3, 3, 2, 3, 3, 3, 3, 2, 3, 2, 1, 2, 2, 1, 1))), Y=c(0.49, 0.49, 0.48, 0.47, 0.48, 0.47, 0.46, 0.46, 0.45, 0.43, 0.45, 0.43, 0.43, 0.44, 0.43, 0.43, 0.46, 0.45, 0.42, 0.42, 0.43, 0.41, 0.41, 0.40, 0.42, 0.40, 0.40, 0.41, 0.40, 0.41, 0.41, 0.40, 0.40, 0.40, 0.38, 0.41, 0.40, 0.40, 0.41, 0.38, 0.40, 0.40, 0.39, 0.39))
nls.sol<-nls(Y~a+(0.49-a)*exp(-b*(X-8)), data=cl, start = list( a= 0.1, b = 0.01 ))
summary(nls.sol)

fn<-function(p, X, Y){
f <- Y-p[1]-(0.49-p[1])*exp(-p[2]*(X-8))
res<-sum(f^2)
f1<- -1+exp(-p[2]*(X-8))
f2<- (0.49-p[1])*exp(-p[2]*(X-8))*(X-8)
J<-cbind(f1,f2)
attr(res, "gradient") <- 2*t(J)%*%f
res
}
#建立最优化函数
out<-nlm(fn, p=c(0.1, 0.01), X=cl$X, Y=cl$Y, hessian=TRUE); out
```

## 多元统计分析
多元统计分析也是统计学的重要组成部分 我们这里介绍多元统计分析中的常见方法应该怎么在R中实现

## 主成分分布与因子分析

作为两种经典的降维手法 我们在这里一起介绍他们的R实现，实际上有非常多共同的地方。因此我们大可以将其联合介绍

### Base R 提供的算法

#### 主成分分析

```r
#PCA的计算
princomp(x, cor = FALSE, scores = TRUE, covmat = NULL,subset = rep(TRUE, nrow(as.matrix(x))), ...)

#提取主成分信息
summary(object, loadings = FALSE, cutoff = 0.1, ...)

#分析载荷矩阵
loadings(x)

#预测新数据主成分的值
predict(object, newdata, ...)

#绘制主成分的碎石图
screeplot(x, npcs = min(10, length(x$sdev)),type = c("barplot", "lines"), main = deparse(substitute(x)), ...)

#绘制数据关于主成分的散点图
biplot(x, choices = 1:2, scale = 1, pc.biplot = FALSE, ...)

```

-   `x`是用于主成分分析的数据 要求是数据框
-   `cor` 的T和F决定了使用样本相关阵主成分分析还是用协方差矩阵作主成分分析 标准化后两者一样

#### 因子分析

因子分析的函数为

```r
factanal(x, factors, data = NULL, covmat = NULL, n.obs = NA,
         subset, na.action, start = NULL, scores = c("none", "regression", "Bartlett"),
         rotation = "varimax", control = NULL, ...)
```

-   `x`是数据 用数据框表示
-   `factors`表示因子个数
-   `scores`表示选用因子得分的方法
-   `rotation = "varimax"`表示用最大方差旋转

因子分析的分析方法和主成分分析基本一样 都有方差贡献率 载荷矩阵这些量供我们分析 所有的输出结果都存储成了一个列表的形式供我们后续分析使用

### psych包提供的函数
相对于基础的R来说，这里的函数有着略微高一点的自由度和翔实的信息，他们将函数体系整合了起来

```r
#含多种可选的方差旋转方法的主成分分析
principal()

#可用主轴、最小残差、加权最小平方或最大似然法估计的因子分析
fa()

#含平行分析的碎石图（做随机数据矩阵相应的平均特征值，辅助选择主成分个数或辅助因子分析的进行，毕竟两者本质接近）
fa.parallel()

#绘制因子分析或主成分分析的结果
factor.plot()

#绘制因子分析或主成分的载荷矩阵
fa.diagram()

#因子分析和主成分分析的碎石图
scree()
```

## 判别分析

最常用的的判别分析方法有距离判别和Fisher判别两种 贝叶斯判别相对而言不是那么的实用

### Fisher判别

它本质是LDA线性判别方法 在多元统计中一般叫做了Fisher判别 函数就是LDA的函数

函数如下 它来自MASS包

```r
lda(formula, data, ... , subset, na.action)
```

-   formula是判别公式 表示称来源对分类变量的回归
-   subset指明训练用样本

用iris数据集作一个例子

```r
data(iris)
attach(iris)
names(iris)
library(MASS)
iris.lda <- lda(Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width)
iris.lda
iris.pred=predict(iris.lda)$class
#用于预测 此时预测的是训练用数据集
table(iris.pred, Species)
detach(iris)
```

最后的预测矩阵展示了判别和原始的情况 versicolor vriginica 都出现了错判

### 距离判别

我们知道 距离判别的核心是计算距离 其他的比较多变 因此R没有设计专门用于距离判别的函数 但是有函数可以帮助我们计算马氏距离

```r
mahalanobis(x, center, cov, inverted=FALSE, ...)
```

它接受一个数据框作为输入 其中center是数据中心 cov是协方差矩阵 输出为矩阵形式体现两两元素之间的距离

## 聚类分析

### 系统聚类

对于系统聚类分析 有着非常简单的函数可以帮助我们进行

```r
#计算距离矩阵使用
dist(x, method = "euclidean", diag = FALSE, upper = FALSE, p = 2)

#计算聚类结果用
hclust(d, method = "complete", members=NULL)

#绘制聚类图 它有着聚类专用形式为
plot(object, hang=-1)

#用来对聚类结果进行切割 给出我们需要的类个数或者高度就可以了
plclust(object, hang=-1)
rect.hclust(tree, k = NULL, which = NULL, x = NULL, h = NULL,border = 2, cluster = NULL)

#聚类结果转化为树状的谱系图 使用plot绘制
as.dendrogram(object, hang = -1, ...)

```

-   其中d是距离结构 method是系统聚类方法的选择 默认最长距离
-   x是数据框
-   method是计算距离的方法
-   diag和upper两个逻辑变量控制了只输出对角线还是输出上三角

### 动态聚类

最为经典的动态聚类方法就是k-means聚类了 R中提供了非常简单的实现方式 函数形式为

```r
kmeans(x, centers, iter.max = 10, nstart = 1,
algorithm = c("Hartigan-Wong", "Lloyd", "Forgy", "MacQueen"))
```

-   x是数据框
-   canters是聚类个数或者初始聚类中心
-   iter.max 是最大迭代次数
-   algorithm是动态聚类的算法

### 聚类的数量
聚类问题最希望解决的一个问题就是我们需要把样本聚为多少类，如果类别过少，可能把严重异质性的数据放到同一组，如果类别过多，则可能分类不完全，真实的聚类场景很可能需要一些对领域知识的了解来选择。我们这里介绍一些稳定的的类别数量选择方法

**NbClust**包提供了众多的指数来确定在一个聚类分析里类的最佳数目。不能保证这些指标得出的结果都一致。事实上，它们可能不一样。但是结果可用来作为选择聚类个数K值的一个参考。`NbClust()`函数的输入包括需要做聚类的矩阵或是数据框，使用的距离测度和聚类方法，并考虑最小和最大聚类的个数来进行聚类。它返回每一个聚类指数，同时输出建议聚类的最佳数目。

一个代码示例为
```r
library(NbClust)
nc <- NbClust(nutrient.scaled, distance="euclidean",
                  min.nc=2, max.nc=15, method="average")
#返回结果包含了各种指标决定的聚类类别个数，以及他们的投票结果
```
## 典型相关分析

典型相关分析就是研究两组变量之间相关关系的一种多元统计方法 也是我们目前学习的唯一一种可以用尽可能简单的形式表示两个组之间的相关关系的方法 他的结果核心是典型相关系数

函数的形式为

```r
cancor(x, y, xcenter = TRUE, ycenter = TRUE)
```

-   其中x y是两组变量的数据矩阵

结果包括

-   典型相关系数
-   为了构造典型相关使用的载荷系数

典型相关也有假设检验的步骤 这里省略

## 对应分析

函数的形式为 来自MASS包

```r
corresp(x, nf = 1, ...)
```

-   x是数据矩阵的形式
-   nf是因子的个数

一个简单的例子为

```r
x.df=data.frame(HighlyFor=c(2, 6, 41, 72, 24), For =c(17, 65, 220, 224, 61), Against=c(17, 79, 327, 503, 300), HighlyAgainst=c(5, 6, 48, 47, 41))
rownames(x.df)<-c("BelowPrimary", "Primary", "Secondary", "HighSchool","College")
biplot(corresp(x.df, nf=2))
#最后这是绘制了对应分析图 怎么分析我们在理论研究的时候证明过了
```
