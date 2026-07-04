---
title: "R 统计可视化学习笔记"
title_en: "R Statistical Visualization Learning Notes"
date: 2024-03-15 16:54:45 +0800
categories: ["Programming", "R"]
tags: ["Learning Notes", "R", "Data Visualization"]
author: Hyacehila
excerpt: "一篇 R 统计可视化学习笔记，按单变量、双变量、多变量和函数型数据整理常见统计图形及 base R、ggplot2 实现。"
excerpt_en: "A study note on statistical visualization in R, organizing common graphics for univariate, bivariate, multivariate, and functional data with base R and ggplot2 examples."
mathjax: true
hidden: true
permalink: '/blog/2024/03/15/r-visualization-learning-notes/'
---

这里的笔记将遵循 《现代统计图形》一书的结构，辅助补充《数据可视化：基于R语言》——贾俊平 中的内容；为了方便和其他的知识结构形成交叉，最终选用在 OBSIDIAN 中一Markdown的形式呈现；

幸运的是， MSG包为我们提供了全部构建笔记需要的图形，我们不需要自行重写所有需要的代码就可以得到需要的图形；

根据现代统计图形中的架构；我们将整个笔记分为了三个大部分，分别介绍统计作图本身 以字典形式介绍各种统计图形 最后介绍R中的一些统计作图模块。

我们选择将图库以外的部分迁移到 [R Graph](/blog/2024/09/16/r-graph-learning-notes/) 在里面研究作图的思想和方法介绍

## 单变量图
从这一章开始 我们介绍统计图形的图库 我们按照变量结构进行大类的分类；对于各种图形 我们希望能详细的介绍其原理 分析方法 以及包括`base R `与 `ggplot2` 的代码实现

其中图形一般只用`base R`的例子 相对而言美观度略差；对于 `ggplot2` 的代码 想要理解他确实需要一些单独的学习 我们这里仅留作参考

就代码实现而言 我们不会详细解释各种参数 而是仅介绍那些这个函数的特色 更多的需要的变换我们可以借助`help`

**单变量图旨在展示单一变量，我们有时候会把多个单变量图放在一起比较，但是仍属于单变量图的范畴（虽然确实使用了多个变量）**

### 条形图
条形图目前是各种统计图形中应用最广泛的，但条形图所能展示的统计量比较贫乏：它只能以矩形条的长度展示原始数值，对数据没有任何概括或推断

#### 基本介绍
R 中条形图的函数为 `barplot()`
* 最核心的参数 `height` 指定长条的长度 可以接受数值向量或者矩阵 对于接受数值向量的情况，他就是绘制最基本的条形图 **如果接受数值矩阵，他对于每一列作为一条进行画图，此时被`beside`参数控制**
* 参数 `beside` 设置为 FALSE，矩阵的每一列占据一条 `beside` 设置为 TRUE 则不进行堆砌
* `horiz` 设置图的方向 是否横向放置
#### 图形示例
![R语言 统计可视化-14](/assets/images/r-learning-notes/r-language-stat-visualization-14.png)
前后两张图就是修改了 `beside` 参数

#### base R
```r
## 基础作图法绘制弗吉尼亚死亡率数据条形图
data(VADeaths)
library(RColorBrewer) # 用分类调色板
par(mfrow = c(2, 1), mar = c(3, 2.5, 0.5, 0.1))
death = t(VADeaths)[, 5:1]
barplot(death, col = brewer.pal(4, "Set1"))
barplot(death, col = brewer.pal(4, "Set1"),
        beside = TRUE, legend.text = TRUE)
```
#### ggplot
```r
## ggplot2 绘制弗吉尼亚死亡率数据条形图
library(ggplot2)
library(patchwork)
data(VADeaths)
reshape_VADeaths = transform(
  expand.grid(sex = colnames(VADeaths), age = rownames(VADeaths)),
  rates = as.vector(t(VADeaths))
)
p = ggplot(data = reshape_VADeaths,
            aes(x = age, y = rates, fill = sex)) +
  labs(x = "年龄", y = "死亡率", fill = "性别") +
  scale_fill_discrete(labels = c("农村男性", "农村女性",
                                 "城市男性", "城市女性"))
p1 = p + geom_col(position = "stack")
p2 = p + geom_col(position = "dodge")
print(p1 / p2)
```

### Cleveland 点图
实上点图和条形图的功能非常类似：条形图通过条的长度表示数值大小，点图通过点的位置表示数值大小，二者几乎可以在任何情况下互换。

#### 基本介绍
R 中点图的函数为 `dotchart()`
* 参数 `x` 和条形图中的 `height` 需求一致
#### 图形示例
![R语言 统计可视化-16](/assets/images/r-learning-notes/r-language-stat-visualization-16.png)
可以看作前面的图形只是更改了方向
**使用的非常少 因为其没有条形图直观，点数量很小的时候才可能使用**

#### base R
```r
## 基础作图法绘制弗吉尼亚死亡率数据的 Cleveland 点图
library(RColorBrewer)
data(VADeaths)
colnames(VADeaths) = c("农村男性", "农村女性", "城市男性", "城市女性")
par(mar = c(2, 6, 0.2, 0.2))
dotchart(t(VADeaths)[, 5:1],
         col = brewer.pal(4, "Set1"), pch = 19, cex = .65)
```

#### ggplot
ggplot 没有提供专门的点图绘制函数 我们只能只用散点图来曲线进行；**点图的使用需求太低了**
```r
## ggplot2 绘制弗吉尼亚死亡率数据的 Cleveland 点图
data(VADeaths)
library(ggplot2)
colnames(VADeaths) = c("农村男性", "农村女性", "城市男性", "城市女性")
tm = rownames(VADeaths)
rownames(VADeaths) = NULL
vd = data.frame(cbind(tm, VADeaths))
vd = reshape(vd, direction = "long", varying = names(vd)[2:5],
              v.name = c("rate"), times = names(vd)[2:5])
vd$rate = as.numeric(vd$rate)
vd$tm = factor(vd$tm)
vd$tm = factor(vd$tm,levels = rev(levels(vd$tm)))
p = ggplot(vd, aes(time, rate, color = time)) + geom_point() +
  facet_grid(tm ~ .) + coord_flip() +
  theme(legend.position = "", axis.title = element_blank())
print(p)
```

### 直方图
直方图（Histogram）是展示连续数据分布最常用的工具，它本质上是对密度函数的一种估计。

直方图作为密度函数估计工具的基本思想：划分区间并计数有多少数据点落入该区间。实际数据不可能无限稠密，因此 h→0 的条件往往是不可能实现的，于是我们退而求其次，只是在某一些区间段里面估计区间上的密度。

关于区间划分，这里我们需要特别指出的是，直方图的理论并非想象中或看起来的那么简单，窗宽也并非可以任意选择，不同的窗宽或区间划分方法会导致不同的估计误差；因此 让用户自己随便设置宽度的直方图往往并不可靠；添加正态密度估计曲线给直方图也没有价值 因为样本不一定正态

直方图实际上是对数据离散化分组，所以它不可避免有一定的随意性，**这里的分组有一定的理论背景 不会像随意分组一样丢失大量信息**

在画直方图（包括移动平均直方图）时，若有可能，则尽量加上密度曲线，或者坐标轴须；因为密度曲线不受分组区间的影响，坐标轴须能反映原始数据的位置 可以避免因为某些分组导致的误判
#### 基本介绍
R 中提供了 `hist()` 函数 用于直方图的绘制
* 参数`x` 为欲估计分布的数值向量
* 参数`breaks` 决定了计算分段区间的方法，它可以是一个向量（依次给出区间端点），或者一个数字（决定拆分为多少段），或者一个字符串（给出计算划分区间的算法名称），或者一个函数（给出划分区间个数的方法） **根据前面的解释我们知道，这个参数非常重要**
* `freq` 和 `probability` 参数均取逻辑值（二者互斥），前者决定是否以频数作图，后者决定是否以概率密度作图（这种情况下矩形面积为 1）
* `labels` 为逻辑值，决定是否将频数的数值添加到矩形条的上方

#### 图形示例
![R语言 统计可视化-17](/assets/images/r-learning-notes/r-language-stat-visualization-17.png)
体现了区间划分 以及频率参数的影响
#### base R
```r
## 基础作图法绘制直方图与密度曲线的结合
par(mar = c(1.8, 3, 0.5, 0.1), mgp = c(2, 0.5, 0), mfrow = c(1, 2))
data(geyser, package = "MASS")

hist(geyser$waiting, freq = FALSE, main = "")
lines(density(geyser$waiting))

hst = hist(geyser$waiting, probability = TRUE,
           main = "", xlab = "waiting")
d = density(geyser$waiting)
polygon(c(min(d$x), d$x, max(d$x)), c(0, d$y, 0),
        col = "lightgray", border = NA)
lines(d)
ht = NULL
brk = seq(40, 110, 5)
for (i in brk) ht = c(ht, d$y[which.min(abs(d$x - i))])
segments(brk, 0, brk, ht, lty = 3)
```
#### ggplot
```r
## ggplot2 绘制直方图与密度曲线的结合
library(ggplot2)
data(geyser, package = "MASS")
p = ggplot(aes(waiting), data = geyser) +
  labs(x = "间隔时间", y = "分布密度") +
  geom_histogram(breaks = seq(40, 110, by = 5), aes(y = ..density..))+
  geom_density(color = "blue", size = 1.2)
print(p)
```

#### 关于核密度估计
今天核密度估计理论已经非常完备 他给出了一种非常优秀的方法来对连续变量的概率分布进行估计 让其他方法完全失去了基本的价值 前面的代码均考虑了在直方图基础上增加核密度估计的问题
* 对于 `base R` 我们使用`density()` 估计核密度再单独绘图
* 对于 `ggplot` 我们提供了原生的方法

**综合核密度与直方图是连续变量概率分布估计如今的唯一解，对于不同因子水平下的核密度，我们也有非常成熟的作图工具**
### 茎叶图
一种比较原始的对连续变量分布进行估计的手段 如今他已经失去了价值 我们这里仅仅看看他的想法

#### 基本介绍
R 中茎叶图的函数为 `stem()`
* 参数 `scale` 控制着 m，即节与节之间的长度（ `scale` 越大则 m 越小）；
* `width` 控制了茎叶图的宽度，若叶子的长度超出了这个设置，则叶子会被截取到长度 `width` ，然后以一个整数表示后面尚有多少片叶子没有被画出来
* `x` 和直方图一致
#### 图形示例
各个陆地块面积的分布  严重的右偏
![R语言 统计可视化-18](/assets/images/r-learning-notes/r-language-stat-visualization-18.png)

### 箱线图
箱线图（Box Plot 或 Box-and-Whisker Plot）主要是从四分位数的角度出发 描述数据的分布;我们可以大致推断出数据的集中或离散趋势（长度越短，说明数据在该区间上越密集，反之则稀疏）

#### 基本介绍
R 中相应的函数为 `boxplot()`

`boxplot()` 是一个泛型函数，所以它可以适应不同的参数类型。目前它支持两种参数类型：公式（ `formula` ）和数据，后者对我们来说可能更容易理解（给一批数据、作相应的箱线图），而前者则是根据类别型变量生成多个并列的箱线图，适合直观的考查分组均值差异

对于前者 其参数可以被如下解释
* 参数 `x` 为一个数值向量或者列表，若为列表则对列表中每一个子对象依次作出箱线图
* `range` 是一个延伸倍数，决定了箱线图的末端（须）延伸到什么位置，这主要是考虑到离群点的原因，只将图形延伸到离箱子两端的$range\times Q_3-Q_1$ 处
* `width` 给定箱子的宽度
* `varwidth` 为逻辑值，若为 `TRUE`，那么箱子的宽度与样本量的平方根成比例，这在多批数据同时画多个箱线图时比较有用，能进一步反映出样本量的大小
* `notch` 也是一个有用的逻辑参数，它决定了是否在箱子上画凹槽，凹槽所表示的实际上是中位数的一个区间估计
其余的问题我们可以参考帮助文档 有些参数 比如 `horizontal` 类似参数设置水平放置问题

#### 图形示例
![R语言 统计可视化-19](/assets/images/r-learning-notes/r-language-stat-visualization-19.png)

#### base R
```r
## 使用公式表示
data(InsectSprays)
boxplot(count ~ spray, data = InsectSprays,
        col = "lightgray", horizontal = TRUE, pch = 4,varwidth = TRUE)

## 使用传统的数据表示
x = rnorm(150)
y = rnorm(50, 0.8)
boxplot(list(x, y),names = c("x", "y"), horizontal = TRUE,
        col = 2:3, notch = TRUE, varwidth = TRUE)
```

#### ggplot
```r
data(InsectSprays)
library(ggplot2)
p = ggplot(aes(y = count, x = spray), data = InsectSprays) +
  geom_boxplot(outlier.shape = 4) +
  labs(x = "杀虫剂", y = "频数") +
  coord_flip()
print(p)
```

### 小提琴图
小提琴图（Violin Plot）是密度曲线图与箱线图的结合，因为它的外观有时候与小提琴的形状比较相像（尤其是展示双峰数据的密度时），所以我们称之为小提琴图
#### 基本介绍与代码
base R不支持小提琴图的绘制 虽然确实有很多包 如 `lattice` 和 `vioplot` 包进行绘制 但是考虑到图形的连贯性 我们这里给出 `ggplot2`的示例代码
```r
## ggplot2 绘制三组双峰数据的小提琴图比较
library(ggplot2)
f = function(mu1, mu2) c(rnorm(300, mu1, 0.5), rnorm(200, mu2, 0.5))
x1 = f(0, 2)
x2 = f(2, 3.5)
x3 = f(0.5, 2)
df = reshape(data.frame(A = x1, B = x2, C = x3),
              direction = "long", varying = c("A", "B", "C"),
              v.name=c("value"), times=c("A", "B", "C"))
p = ggplot(df, aes(value, time)) +
  geom_violin(fill = "bisque") +
  geom_boxplot(width = .1) +
  labs(x = "", y = "")
print(p)
```
#### 图形示例
![R语言 统计可视化-23](/assets/images/r-learning-notes/r-language-stat-visualization-23.png)

### 坐标轴须
坐标轴须（Rug）顾名思义就是往坐标轴上添加短须。短须的作用是标示出相应坐标轴上的变量数值的具体位置，每一根短须都对应着一个数据。这样做的好处在于，我们可以从坐标轴须的分布了解到该变量的分布

**坐标轴须只是一种图形的附加物（属于base R的低级作图函数），但是其确实比较实用，因此这里单独介绍**

#### 基本介绍
R 中坐标轴须的函数为 `rug()`
* `x` 为一个向量，给出短须的位置 （就是正常的数据向量）
* `ticksize` 为短须的长度
* `side` 为欲画短须的坐标轴的位置（我们在后面介绍base R元素的时候可以理解了）

#### 图形示例
![R语言 统计可视化-24](/assets/images/r-learning-notes/r-language-stat-visualization-24.png)

#### base R
```r
## 基础作图法绘制带坐标轴须的喷泉喷发时间密度曲线图
data(faithful)
par(mar = c(3, 4, 0.4, 0.1))
plot(density(faithful$eruptions), main = "")
rug(faithful$eruptions)
```

#### ggplot
```r
## ggplot2 绘制带坐标轴须的喷泉喷发时间密度曲线图
library(ggplot2)
data(faithful)
p = ggplot(faithful, aes(eruptions)) + geom_line(stat = "density") +
  geom_rug() + xlim(c(1, 6)) + labs(x = "喷发时间", y = "分布密度")
print(p)
```
### 带状图
带状图（Strip Chart），又叫一维散点图（1-D Scatter Plot），是针对一维数据的散点图，它本质上是数据与固定值（固定 x 或固定 y）之间的散点图，这样形成的图形外观是带状的，因此称之为带状图

**虽然有着保留原始数据的优点，但是他的使用确实较少，`ggplot2`中没有专门的支持**
#### 基本介绍
R 中带状图的函数为 `stripchart()` 带状图函数为泛型函数，可以直接接受数据参数或者公式参数 对于泛型函数的问题我们以后会进行详细的解释
* `x` 为数据，一般为一个向量
* `method`指定作图方法，取值 `overplot` 意思是将所有的数据点画在一条直线上，不管它们是否有重叠
* `jitter` 意思是将直线上的数据随机打乱，以免数据重叠导致我们不知道在某个位置究竟有多少个点
* `stack` 意思是将重叠的数据堆砌起来，某个位置重叠的数据越多，则堆砌越高

#### 图形示例
![R语言 统计可视化-25](/assets/images/r-learning-notes/r-language-stat-visualization-25.png)

#### base R
```r
## 基础作图法绘制各种杀虫剂下昆虫数目的带状图
data(InsectSprays)
layout(matrix(1:2, 2), height = c(1, 1))
par(mar = c(4, 4, 0.2, 0.2))
boxplot(count ~ spray, data = InsectSprays, horizontal = TRUE,
        border = "red", col = "lightgreen", at = 1:6 - 0.3,
        xlab = "频数", ylab = "杀虫剂")
stripchart(count ~ spray, data = InsectSprays, method = "stack",
           add = TRUE)
stripchart(count ~ spray, data = InsectSprays, method = "jitter",
           xlab = "频数", ylab = "杀虫剂")
```

### 饼图
饼图是目前应用非常广泛的统计图形，然而，根据统计学家（主要是 Cleveland 和 McGill）和一些心理学家的调查结果 ([Cleveland 1985](https://bookdown.org/xiangyun/msg/gallery.html#ref-Cleveland85))，这种以比例展示数据的统计图形实际上是很糟糕的可视化方式，因此，R 关于饼图的帮助文件中清楚地说明了并不推荐使用饼图，而是使用条形图[条形图](/blog/2024/03/15/r-visualization-learning-notes/)或点图[Cleveland 点图](/blog/2024/03/15/r-visualization-learning-notes/)作为替代

虽然我们并不推荐使用饼图 但是依旧提供使用它的方法；`ggplot2` 不提供饼图 原因如上

#### 基本介绍
R 提供了函数 `pie()` 制作饼图
* 参数 `x` 为一个数值向量（保证和为1）
* `labels` 为标签
* 其它参数基本上都是为多边形准备的

特别的 相比平面饼图而言还有使用体验更为糟糕的三维饼图 它使用`plotrix` 包
#### 图形示例
![R语言 统计可视化-26](/assets/images/r-learning-notes/r-language-stat-visualization-26.png)
#### base R
```r
## 基础作图法绘制馅饼销售饼图、点图和条形图
layout(matrix(c(1, 2, 1, 3), 2)) # 拆分作图区域
par(mar = c(4, 4, 0.2, 0.2))
pie.sales = c(0.12, 0.3, 0.26, 0.16, 0.04, 0.12)
names(pie.sales) = c("蓝莓", "樱桃", "苹果",
                     "波士顿奶油", "其它", "香草奶油")
pie.col = c("purple", "violetred1", "green3",
             "cornflowerblue", "cyan", "white")
pie.sales = sort(pie.sales, decreasing = TRUE) # 排序有助于可读性
pie(pie.sales, col = pie.col)
dotchart(pie.sales, xlim = c(0, 0.3))
barplot(pie.sales, col = pie.col, horiz = TRUE,
        names.arg = "", space = 0.5)
```

### QQ图
关于统计分布的检验有很多种，例如 KS 检验、卡方检验等，从图形的角度来说，我们也可以用 QQ 图（Quantile-Quantile Plots）来检查数据是否服从某种分布；它基于理论分布和实际分布的分位数进行

#### 基本介绍
R 中 QQ 图的函数为 `qqplot()` ，由于正态分布是我们经常检验的分布，R 也直接提供了一个画正态分布 QQ 图的函数 `qqnorm()` ，这两个函数都在基础包 **stats** 包中
* `qqplot()` 检验的是两批数据的分布是否相同，所以它需要两个数据参数 x 和 y （理论数据用函数生成，实际分布给出）
* `qqnorm()` 只需要一个数据参数 x （理论数据自动用正态分布生成）
#### 图形示例
![R语言 统计可视化-27](/assets/images/r-learning-notes/r-language-stat-visualization-27.png)
#### base R
```r
## 基础作图法绘制喷泉间隔时间的正态分布 QQ 图
data(geyser, package = "MASS")
geyser$waiting_scaled = scale(geyser$waiting)
qqnorm(geyser$waiting_scaled, cex = 0.7, asp = 1, main = "")
abline(0, 1)
```

#### ggplot
```r
## ggplot2 绘制喷泉间隔时间的正态分布 QQ 图
library(ggplot2)
library(qqplotr)
library(patchwork)
data(geyser, package = "MASS")
geyser$waiting_scaled = scale(geyser$waiting)
qq1 = ggplot(data = geyser, mapping = aes(sample = waiting_scaled)) +
  coord_fixed(ratio = 1, xlim = c(-3, 3), ylim = c(-3,3)) +
  geom_abline(aes(intercept = 0, slope = 1), color = "blue") +
  stat_qq_point() +
  labs(x = "理论分位数", y = "实际分位数")
qq2 = ggplot(aes(waiting_scaled), data = geyser) +
  geom_density() +
  stat_function(mapping = aes(x), data = data.frame(x = c(-3, 3)),
                fun = dnorm, n = 101, args = list(mean = 0, sd = 1),
                linetype = 2) +
  labs(x = "间隔时间（标准化）", y = "分布密度")
print(qq1 | qq2)
```

### 瀑布图
瀑布图（Waterfall Plot）是一种常用于展示数据变化趋势的可视化图表，尤其适用于展示不同时间段或阶段中的累计变化。它通常由一系列相邻的条形或柱状图组成，其中每个条形代表某个特定变量的增减幅度。瀑布图最常见的应用领域包括财务分析、销售数据分析、实验结果展示等。

他是传统条形图的一个变种

```r
library(waterfall)
library(dplyr)

## 创建原始数据
a <- c("Start", "Sales Increase", "Cost Increase", "Tax Decrease", "End")
data <- data.frame(
  Stage = factor(a,ordered = T,levels = a),
  Change = c(100, 50, -30, 10, 0)  # End值设置为0，确保它出现在最后
)

## 使用 waterfallchart 绘制瀑布图
waterfallchart(data  = data,
               Change~Stage,
               main = "Waterfall Chart (Staircase Style)",
               ylab = "Cumulative Value",
               xlab = "Stage")
```

形式如下
![R 统计可视化-18](/assets/images/r-learning-notes/r-stat-visualization-18.png)
## 双变量图
本章介绍反映两个变量之间随机关系的统计图形；

**最为经典的一种是散点图，描述两个连续变量之间的连续**

**如果两个变量中有一个是定性变量，那么符合我们对比多个单变量图形的思想，比如前一章介绍的条形图，箱线图，点图**

至于涉及矩阵图形与大于二的多变量的问题 我们在后面单独研究 [多变量图](/blog/2024/03/15/r-visualization-learning-notes/) [矩阵图形](/blog/2024/03/15/r-visualization-learning-notes/)

### 散点图
散点图通常用来展示两个变量之间的关系，这种关系可能是线性或非线性的，图中每一个点的横纵坐标都分别对应两个变量各自的观测值，因此散点所反映出来的趋势也就是两个变量之间的关系。

散点图太常用了 我们不进行额外的介绍了
#### 图形示例
![R语言 统计可视化-28](/assets/images/r-learning-notes/r-language-stat-visualization-28.png)
右侧的图调整了透明度设计产生了一个高度叠加的圈，对于高密度情况很实用，当然，我们也有一些其他手段处理高密度散点重叠的问题，比如 **graphics** 中 `smoothScatter` 中的核密度平滑。这是一个很值得解决的问题 参考[平滑散点图](/blog/2024/03/15/r-visualization-learning-notes/)
#### base R
```r
## 基础作图法绘制半透明散点图中
data(BinormCircle, package = "MSG")
par(mfrow = c(1, 2), pch = 20, ann = FALSE, mar = rep(.05, 4))
plot(BinormCircle, col = rgb(1, 0, 0), axes = FALSE)
box()
plot(BinormCircle, col = rgb(1, 0, 0, alpha = .01), axes = FALSE)
box()
```

#### ggplot
```r
## ggplot2 绘制半透明散点图中
data(BinormCircle, package = "MSG")
library(ggplot2)
library(patchwork)
p = ggplot(BinormCircle, aes(V1, V2)) +
  theme(axis.ticks = element_blank(), axis.text = element_blank(),
        axis.title = element_blank())
p1 = p + geom_point(color = rgb(1,0,0)) + theme_void()
p2 = p + geom_point(color = rgb(1,0,0), alpha = 0.01) + theme_void()
print(p1 | p2)
```

### 一元函数曲线
我们介绍如何绘制一元函数曲线图 也只有一元函数能在二维平面比较好的展示了

这种曲线图对于数据分析并没有什么太深的连续 因此`ggplot2`不提供绘制的方法
#### 基本介绍
R 中函数曲线图的函数为 `curve()` R 专门提供了一个函数，目的是为了节省我们去使用低层作图函数（如 `lines()`）的精力和时间
* 参数 `expr` 为一个一元函数或者该函数的名称；
* `from` 和 `to` 分别定义了曲线的起点和终点；
* `n` 决定将定义域分成多少个小区间，以便计算函数值并连接曲线， `n` 值越大曲线越光滑
#### 图形示例
![R语言 统计可视化-29](/assets/images/r-learning-notes/r-language-stat-visualization-29.png)
#### base R
```r
## 基础作图法绘制一元函数曲线图
par(par(mar = c(4.5, 4, 0.2, 0.2)), mfrow = c(2, 1))
chippy = function(x) sin(cos(x) * exp(-x / 2))
curve(chippy, -8, 7, n = 2008, xlab = "x", ylab = "chippy(x)")
curve(sin(x) / x, from = -20, to = 20, n = 200,
      xlab = "t", ylab = expression(phi*X(t)))
```

### 向日葵散点图
向日葵散点图（Sunflower Scatter Plot）是用来克服散点图中数据点重叠问题的特殊散点图工具。它采用的办法是在有重叠的地方用一朵”向日葵花”的花瓣数目来表示重叠数据的个数，这样我们就很容易看出来散点图中哪些地方的数据有重叠，而且能知道重叠的具体数目。

向日葵散点图在数据特别密集或者数据类型为分类数据时很有用，因为这两种情况下都容易产生重复的数据点

`ggplot2`不提供绘制的方法  这种散点图在正式的文献中很少出现 也不属于其适配目标

#### 基本介绍
R 中向日葵散点图的函数为 `sunflowerplot()`
* `x` 和 `y` 分别为散点图的两个变量；
* `number` 为人工给定的数据频数，即图中的花瓣数目，若不指定这个参数的话 R 会自动从 `x` 和 `y` 计算；
*  `rotate` 决定是否随机旋转向日葵的角度；
* `pch` 给定散点图的点的类型；
* `cex` 给定散点图的点的缩放倍数；
* `cex.fact`给定向日葵中心点的缩小倍数，真正的缩放倍数为 `cex/cex.fact` ；

#### 图形示例
![R语言 统计可视化-30](/assets/images/r-learning-notes/r-language-stat-visualization-30.png)
#### base R
```r
## 绘制鸢尾花花瓣长和宽的向日葵散点图
data(iris)
par(mar = c(4, 4, 0.2, 0.2))
sunflowerplot(iris[, 3:4], col = "gold", seg.col = "gold",
              xlab = "花瓣长度", ylab = "花瓣宽度")
```

### 平滑散点图
平滑散点图的基础是散点图，但它并不直接将散点画出来，而是基于二维核密度估计 用特定颜色深浅表示某个位置的密度值大小，默认颜色越深说明二维密度值越大，即该处数据点越密集。

由于平滑散点图大致保留了原始数据点的位置，因此两个变量之间的关系仍然可以从图中看出来，这一点和普通的散点图类似。平滑散点图进一步的优势在于它同时还显示了二维变量的密度，从密度中我们也许可以观察到局部的聚类现象（大块的深色）

平滑散点图看起来和[散点图](/blog/2024/03/15/r-visualization-learning-notes/)右图比较相似，但前者蕴含了更多的数理统计背景。不过，我们也不必一味追求数学理论，透明色可叠加这一点性质体现出的原理又何尝不是一种密度估计呢？

#### 基本介绍
R 中平滑散点图的函数为 `smoothScatter()`
*  `x` 和 `y` 是两个数值向量，或者如果不提供 `y` 的话，可以提供一个两列的矩阵/数据框等给 `x` ；
* `nbin` 为横纵坐标方向上划分网格的数目，可以是长度为 1 或 2 的整数向量；
* `bandwidth` 为计算核密度估计时使用的带宽；
* `colramp` 为生成颜色向量的函数，默认生成从白色到蓝色渐变的颜色向量；
* `nrpoints` 为需要画出来的点的数目，因为平滑散点图的目的不是画散点，而是画颜色块，但有时候图形中某些地方的密度估计非常低，因此对应颜色也非常浅，导致读者难以察觉那些地方还有数据点的存在，此时不妨直接将这些”离群点”直接画出来
#### 图形示例
![R语言 统计可视化-31](/assets/images/r-learning-notes/r-language-stat-visualization-31.png)
#### base R
```r
## 基础作图法绘制 BinormCircle 数据的平滑散点图
data("BinormCircle", package = "MSG")
par(mar = c(4, 4, 0.3, 0.1))
smoothScatter(BinormCircle)
```
#### ggpot
```r
## ggplot2 绘制 BinormCircle 数据的平滑散点图
data("BinormCircle", package = "MSG")
library(ggplot2)
library(ggpointdensity)
p = ggplot(data = BinormCircle, aes(x = V1, y = V2)) +
  geom_pointdensity(adjust = 0.1) +
  scale_color_gradient(low="lightblue", high="darkblue") +
  theme(legend.position = "")
print(p)
```

### 风玫瑰图
风玫瑰图是一类非常特殊的图形 包括风向玫瑰图和风速玫瑰图；只是我们一班使用后者 同时展示风速和风向；

**它只用来体现风速和风向，是气象领域专用的图形，只是因为人文方向偶尔有使用才被我们列入**

风玫瑰图本质上就是堆叠的条形图，绘制在极坐标系中，一般分为16个扇区；每个扇区都是这个风向中各个风速的频率 以`beside = F` 的条形图绘制

#### 基本介绍
R中的 `openair` 包提供了函数 `windRose` 进行绘制 用法如下
* `mydata` 记录风向和风速的数据框
* `ws` 风速列名称
* `wd` 风向列名称
* `angel` 风向划分角度
**更多的需求我们可以专门查阅**

#### 图形示例
![R语言 统计可视化-32](/assets/images/r-learning-notes/r-language-stat-visualization-32.png)
#### openair 代码实现
```r
## 绘制风玫瑰图
library(openair)
windRose(mydata)
windRose(mydata = mydata, ws = "ws", wd = "wd",
         key.position = "right", paddle = FALSE, seg = 0.9,
         angle = 22.5, ws.int = 0.5,
         cex = 3, breaks = c(seq(0,5,1), 21))
```

### 生存函数图
在很多医学研究中，我们主要关心的变量是病人的某种事件发生的时间，例如死亡、疾病复发等。事实上，以“生存时间”为研究对象的领域并不仅限于医学，例如在金融领域，我们可能需要了解信用卡持有者的信用风险发生时间。这类数据一般统称为生存数据（survival data），而生存数据通常有一个特征就是删失，即观测对象因为某种原因退出了我们的观察

**这种研究单独被称为生存分析，是一类很重要的问题，有专门的实现**

#### 基本介绍
本节要介绍的图形对象主要是生存函数（Survival Function），其定义是个体存活超过时间 $t$ 的概率
$$S(t)=P(T>t);t\geq0$$
对于存在删失的生存数据 $(t_i,\delta_i),i=1,\cdots,n$ (其中 $t_i$ 为记录时间，$\delta_i=0$ 表示存在删失，1 表示个体没有删失) , 生存函数的 Kaplan-Meier 估计为：

$$
\left.\hat{S}(t)=\left\{\begin{array}{ll}\prod_{i\colon t_{(i)}\leq t}(\frac{n-i}{n-i+1})^{\delta_{(i)}},&\text{对}t\leq t_{(n)};\\0&\text{如果}\delta_{(n)}=1,\\\text{未定义}&\text{如果}\delta_{(n)}=0,\end{array}\right.\right.\text{对}t>t_{(n)}.
$$

**survival** 包 提供了生存函数的计算和估计方法。具体函数为 `survfit()`，它返回一个 survfit 类的对象;而 **survival** 包扩展了泛型函数 `plot()`，使其拥有子函数 `plot.survfit()`，因此在估计完生存函数之后，我们可以直接调用 `plot()` 生成生存函数图
#### 图形示例
![R语言 统计可视化-33](/assets/images/r-learning-notes/r-language-stat-visualization-33.png)
#### survival代码实现
```r
data("leukemia", package = "survival")
library(survival)
leukemia.surv = survfit(Surv(time, status) ~ x, data = aml)
plot(leukemia.surv, lty = 1:2, xlab = "time")
legend("topright", c("Maintenance", "No Maintenance"),
       lty = 1:2, bty = "n")
```

### 条件密度图
条件密度图 (Conditional Density Plot), 顾名思义，展示的是一个变量的条件密度，确切的说是一个分类变量$Y$相对一个连续变量$X$ 的条件密度 $P(Y|X)$。假设 $Y$ 的取值为 $1,2,\cdots,k$, 那么条件密度图将按 照$X$ 的取值从小到大在纵轴方向上依次展示出$Y=i\left(i=1,2,\cdots,k\right)$ 的条件概率分布比例

#### 基本介绍
R 中条件密度图的函数为 `cdplot()`，它主要是基于密度函数 `density()` 完成条件密度的计算
*  `x` 为条件变量 X，它是一个数值向量， `y` 是一个因子向量，即离散变量 Y；他也是泛型函数
* `plot` 为逻辑值，决定了是否作出图形（或者仅仅是计算而不作图）

我们可以用其研究那些原本应该用`logit` 回归研究的问题 作一个主管的参考
#### 图形示例
![R语言 统计可视化-34](/assets/images/r-learning-notes/r-language-stat-visualization-34.png)

#### base R
```r
## 基础作图法绘制航天飞机 O 型环在不同温度下失效的条件密度图
data(orings, package = "DAAG")
orings$Fail = factor(apply(orings[, -1], 1, function(x) all(x == 0)),
                     labels = c("yes", "no"))
cdplot(Fail ~ Temperature, data = orings, col = c("lightblue", "red"))
points(orings$Temperature, c(0.75, 0.25)[as.integer(orings$Fail)],
       col = "blue", bg = "yellow", pch = 21)
```

#### ggplot
```r
## ggplot2 绘制航天飞机 O 型环在不同温度下失效的条件密度图
library(ggplot2)
library(DAAG)
data(orings, package = "DAAG")
orings$Fail = factor(apply(orings[, -1], 1, function(x) all(x == 0)),
                     labels = c("yes", "no"))
p = ggplot(orings,
           aes(Temperature, ..count.., fill = Fail)) +
  geom_density(position = "fill") +
  geom_point(aes(Temperature, c(0.75, 0.25)[as.integer(Fail)])) +
  xlab("温度") +
  scale_y_continuous("失效", breaks = c(0.25, 0.75),
                     labels = c("否", "是")) +
  theme(legend.position = "")
print(p)

```

### 二维箱线图
我们介绍了普通的箱线图[箱线图](/blog/2024/03/15/r-visualization-learning-notes/)，即用箱线表示一维数据的各个分位数，在二维情况下，我们可以用类似的思想画二维箱线图。二维箱线图又名袋图（Bag Plot）

二维箱线图的做法是从数据的中心向外，逐渐用凸包多边形将散点图中的点包起来，直到包到一半的数据点，此时的凸包相当于普通箱线图中的箱子，然后再向外包到所有数据点。二维箱线图的基本构成就是一个中心和两个多边形，它们能粗略描述数据的二维分布情况。

二维箱线图也需要专门的图包进行绘制
#### 基本介绍
R 中 **aplpack** 包提供了一个函数 `bagplot()` 可以用来画二维箱线图
* `x` 和 `y` 分别是横纵坐标轴上的数据向量，也可以直接提供一个 2 列的矩阵或数据框；
* `factor` 类似 `boxplot()` 中的 `range` 参数，用来定义离群点，取值越大，则离群点越少（数据点离中心的距离可以越远）；
* `approx.limit` 界定了大数据的样本量，如果原始数据的样本量超过这个数字，则随机抽取 `approx.limit` 个数据点用作二维箱线图的计算；
* `dkmethod` 取值 1 或 2，决定用哪种方法计算袋子的范围，取值 2 计算更精确
#### 特别的
```r
msg("5.9) #出现报错
```

## 多变量图
从这一章开始介绍三个及以上变量的图形；但是我们将一类特殊的图形：矩阵图形留给下一章介绍 矩阵图形研究一种特殊的三变量图 也就是行数列数+取值的问题

### 散点图矩阵
散点图矩阵（Scatterplot Matrices）是散点图的高维扩展，它的基本构成是普通散点图，只是将多个变量的两两散点图以矩阵的形式排列起来，就构成了所谓的散点图矩阵

散点图矩阵从一定程度上克服了在平面上展示高维数据的困难，对于我们查看变量之间的两两关系非常有用。

#### 基本介绍
R 中散点图矩阵的函数为 `pairs()`
*  `x` 是一个矩阵或数据框，包含了要作散点图的那些变量
* `panel` 参数给定一个画散点图的函数，这个函数将应用在每一格图形中；
* 有时候我们并不需要统一的散点图函数，这时可以用 `lower.panel`  和 `upper.panel` 来分别指定上三角窗格和下三角窗格中的作图函数

**car**包中的`scatterplotMatrix()`函数也可以生成散点图矩阵，自定义程度更高一点
#### 图形示例
![R语言 统计可视化-35](/assets/images/r-learning-notes/r-language-stat-visualization-35.png)

#### base R
```r
## 基础函数作图法绘制鸢尾花数据的散点图矩阵
## 观察如何使用 hist() 做计算并用 rect() 画图
data("iris")
panel.hist = function(x, ...) {
  usr = par("usr")
  on.exit(par(usr))
  par(usr = c(usr[1:2], 0, 1.5))
  h = hist(x, plot = FALSE)
  breaks = h$breaks
  nB = length(breaks)
  y = h$counts / max(h$counts)
  rect(breaks[-nB], 0, breaks[-1], y, col = "beige")
}
idx = as.integer(iris[["Species"]])
names(iris)[1:4] = c("花萼长度", "花萼宽度", "花瓣长度", "花瓣宽度")
pairs(iris[1:4],
      upper.panel = function(x, y, ...)
        points(x, y, pch = c(17, 16, 6)[idx], col = idx),
      pch = 20, oma = c(2, 2, 2, 2),
      lower.panel = panel.smooth, diag.panel = panel.hist
)
```

#### ggplot
我们补充新的包实现更好的效果
```r
## ggplot2 绘制鸢尾花数据的散点图矩阵
library(ggplot2)
library(GGally)
data("iris")
names(iris) = c("花萼长度", "花萼宽度", "花瓣长度", "花瓣宽度", "种类")
p = ggpairs(iris, aes_string(colour="种类", alpha=0.5))
print(p)
```

### 条件分割图
条件分割图（Conditioning Plot）的思想源自于统计学中的条件分布，即：给定某一个（或几个）变量之后看我们所关心的变量的分布情况。在条件分割图中，这种”分布”主要指的是两个变量之间的关系，通常以散点图表示。

条件分割图可以看作是对散点图的进一步深入发掘，它可以以一个或者两个条件变量作为所有数据的划分条件，条件变量在图形的边缘用灰色矩形条标记出变量的取值范围，每个矩形条对应着一幅散点图（严格来说此时应该称作”条件散点图”），这就是条件分割图的基本做法。

#### 基本介绍
R 中条件分割图的函数为 `coplot()`
* 参数 `formula` 为一个公式，形式为 `y ~ x | a`（一个条件变量）或 `y ~ x | a * b`（两个条件变量），“`|`” 后面即为条件变量；
* `data` 为数据，其中包含了 `x` 、 `y` 、 `a` 和 `b` 等变量；
* `given.values` 指定条件变量的取值范围；
* `panel` 参数为该函数的关键参数，它决定了每一幅散点图的画法，默认只是画点
* `number` 和 `overlap` 传给 `co.intervals()` 函数用来计算划分连续变量的区间，前者设定划分段数，后者设定区间之间的重叠比例
**连续变量划分区间指的是对于条件变量我们如何划分，因为散点图是有限幅的，划分区间数要和图数量匹配**

#### 图形例子
![R语言 统计可视化-36](/assets/images/r-learning-notes/r-language-stat-visualization-36.png)
#### base R
```r
## 基础作图法绘制给定震源深图的地震经纬度条件分割图
data(quakes)
library(maps)
par(mar = rep(0, 4), mgp = c(2, .5, 0))
coplot(lat ~ long | depth, data = quakes, number = 4,
       xlab = c("经度", "深度"), ylab = "纬度",
       ylim = c(-45, -10.72), panel = function(x, y, ...) {
         map("world2",
             regions = c("New Zealand", "Fiji"),
             add = TRUE, lwd = 0.1, fill = TRUE, col = "lightgray"
         )
         text(180, -13, "Fiji", adj = 1)
         text(170, -35, "NZ")
         points(x, y, col = rgb(0.2, 0.2, 0.2, .5))
       }
)
```

### 符号图
符号图是用各种符号展示高维数据的图示工具，它的主要思想是将高维数值体现在图形中符号的特征上。

符号图本质上是一种高度自定义化的散点图 比前面设置`panel`参数自定义程度更高

如：以矩形为散点图的基本符号，那么我们可以用其长宽分别代表两个变量，这样一幅图形中至少可以放置四个变量 我们借此实现了高维的呈现

绘制符号图需要我们审慎的思考 没有专门的函数能够帮助我们一蹴而就

#### 基本介绍
R 中的符号图函数为 `symbols()`，它提供了六种基本符号：圆、正方形、长方形、星形、温度计和箱线图，分别由相应的参数指定
* `circles` 圆：一个数值向量，给定圆的半径
* `squares` 正方形：一个数值向量，给定正方形的边长
* `rectangles` 长方形：一个矩阵，列数为 2，这两列分别给定长方形的宽和高
* `stars` 星形：一个矩阵，列数 ≥3，类似雷达图，给定从星星中心向每个方向的射线的长度（严格说是线段（星形在符号图中并不直观，推荐直接使用星状图）
* `thermometers` 温度计：一个矩阵，列数为 3 或 4，前两列分别给定温度计的宽和高；若矩阵为三列，那么第三列为温度计内的”温度”高度，注意这一列的值应该小于 1，否则温度的填充会超过温度计的范围；若矩阵为四列，那么温度将按照第三列与第四列的比率进行填充，同样，这两列的比率需要小于 1
* `boxplots` 箱线图：一个矩阵，列数为 5，前两列分别给定箱子的宽和高，第三、四列分别给定两条线（下线和上线）的长度，第五列与温度计类似，给定箱线图内的中位数标记线在箱子内部的高度比例，因此这一列数据也需要在 $[0,1]$ 范围内；这里只是借用了箱线图的称谓，符号图中的箱线与真正的箱线图之间没有关系

#### 图形示例
![R语言 统计可视化-37](/assets/images/r-learning-notes/r-language-stat-visualization-37.png)

它的基础是一幅等高图  利用人口预期寿命和高学历人数两个变量计算二维密度，画出等高线，便完成了底图的制作；
然后我们通过人口预期寿命和高学历人数两个变量的数值往图中添加温度计符号，温度计宽代表增长率，高代表总人口数，温度代表城镇人口比重；
然后我们用 `text()` 函数将各省市的文本标签添加到图中。

经过这些图形元素的表达，全国 31 省市自治区的五项人口特征便一目了然，例如通过温度计的高度可以观察出三个人口大省广东、山东、河南（相应的人口总量小的地区如西藏、青海、宁夏等也容易看出），由宽度可以看出西藏、青海、宁夏、新疆等省市自治区的人口自然增长率非常高（而北京、上海、天津等直辖市的增长率则很低），从温度指示的情况来看，北京、上海和天津三大直辖市的城镇人口比例要远高于其它地区；

从整幅散点图来看，人口平均预期寿命与高学历者人数呈比较明显的正相关关系。箱线图和坐标轴须分别刻画了人口平均预期寿命与高学历者人数各自的分布特征。这样，我们就完成了在平面上描述五维变量的任务。
#### base R
```r
## 以下是生成图形的代码：
ChinaPop =
structure(c(1.09, 1.43, 6.09, 6.02, 4.62, 0.97, 2.57, 2.67, 0.96,
2.21, 5.02, 6.2, 5.98, 7.83, 5.83, 5.25, 3.05, 5.15, 7.02, 8.16,
8.93, 3, 2.9, 7.38, 7.97, 10.79, 4.01, 6.02, 9.49, 10.98, 11.38,
1538, 1043, 6851, 3355, 2386, 4221, 2716, 3820, 1778, 7475, 4898,
6120, 3535, 4311, 9248, 9380, 5710, 6326, 9194, 4660, 828, 2798,
8212, 3730, 4450, 277, 3720, 2594, 543, 596, 2010, 0.8362, 0.7511,
0.3769, 0.4211, 0.472, 0.587, 0.5252, 0.531, 0.8909, 0.5011,
0.5602, 0.355, 0.473, 0.37, 0.45, 0.3065, 0.432, 0.37, 0.6068,
0.3362, 0.452, 0.452, 0.33, 0.2687, 0.295, 0.2665, 0.3723, 0.3002,
0.3925, 0.4228, 0.3715, 76.1, 74.91, 72.54, 71.65, 69.87, 73.34,
73.1, 72.37, 78.14, 73.91, 74.7, 71.85, 72.55, 68.95, 73.92,
71.54, 71.08, 70.66, 73.27, 71.29, 72.92, 71.73, 71.2, 65.96,
65.49, 64.37, 70.07, 67.47, 66.03, 70.17, 67.41, 48001, 18601,
40036, 23197, 23660, 44404, 22832, 30888, 40549, 63909, 33115,
29007, 21877, 19946, 50909, 48450, 36287, 34917, 66510, 22556,
5524, 16122, 35297, 14897, 18117, 293, 28734, 13637, 4682, 4895,
21340), .Dim = c(31L, 5L), .Dimnames = list(c("北京", "天津",
"河北", "山西", "内蒙古", "辽宁", "吉林", "黑龙江", "上海", "江苏",
"浙江", "安徽", "福建", "江西", "山东", "河南", "湖北", "湖南",
"广东", "广西", "海南", "重庆", "四川", "贵州", "云南", "西藏",
"陕西", "甘肃", "青海", "宁夏", "新疆"), c("增长率", "总人口",
"城镇人口比重", "预期寿命", "高学历人数")), adj = structure(c(-0.5,
-0.5, 0.5, 0.5, 1.3, -0.4, -0.6, -0.5, -0.5, -0.6, -0.6, 0.6,
0.5, 0.5, 0.5, 0.5, 0.5, 1.8, 0.5, 1.7, 0.5, -0.6, -0.6, 0.5,
0.5, 0.5, 1.7, -0.7, 0.5, 0.5, 0.5, 0.5, 0.5, -0.5, 0, -0.5,
0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -0.3, 2.1, -0.3, -0.5, -0.5, -1.6,
0.5, -0.7, 1.3, -0.7, 0.5, 0.5, 2.4, -0.3, -0.6, 0.5, 0.5, -0.8,
-0.7, -0.8), .Dim = c(31L, 2L), .Dimnames = list(c("北京", "天津",
"河北", "山西", "内蒙古", "辽宁", "吉林", "黑龙江", "上海", "江苏",
"浙江", "安徽", "福建", "江西", "山东", "河南", "湖北", "湖南",
"广东", "广西", "海南", "重庆", "四川", "贵州", "云南", "西藏",
"陕西", "甘肃", "青海", "宁夏", "新疆"), c("horizontal", "vertical"
))))
library(KernSmooth)
x = ChinaPop
x[, 1:2] = apply(x[, 1:2], 2, function(z) 20 * (z -
    min(z)) / (max(z) - min(z)) + 5)
symbols(x[, 4], x[, 5],
  thermometers = x[, 1:3], fg = "gray40",
  inches = 0.5, xlab = "\u4EBA\u5747\u9884\u671F\u5BFF\u547D", ylab = "\u9AD8\u5B66\u5386\u8005\u4EBA\u6570"
)
est = bkde2D(x[, 4:5], apply(x[, 4:5], 2, dpik))
contour(est$x1, est$x2, est$fhat, add = TRUE, lty = "12")
for (i in 1:nrow(x)) {
  text(x[i, 4], x[i, 5], rownames(x)[i],
    cex = 0.75, adj = attr(x, "adj")[i, ]
  )
}
rug(x[, 4], 0.02, side = 3, col = "gray40")
rug(x[, 5], 0.02, side = 4, col = "gray40")
boxplot(x[, 4],
  horizontal = TRUE, pars = list(
    boxwex = 7000,
    staplewex = 0.8, outwex = 0.8
  ), at = -6000, add = TRUE, notch = TRUE, col = "skyblue",
  xaxt = "n"
)
boxplot(x[, 5],
  at = 63, pars = list(
    boxwex = 1.4,
    staplewex = 0.8, outwex = 0.8
  ), add = TRUE, notch = TRUE, col = "skyblue",
  yaxt = "n"
)
text(67, 60000, "2005", cex = 3.5, col = "gray")
```

### 星状图
星状图（Star Plot）、蛛网图（Spider Plot）和雷达图（Radar Plot）本质上是一类图形，它们都用线段离中心的长度来表示变量值的大小，这三种图形名称的区别在于星状图用来展示很多个多变量个体，各个个体的图形相互独立，从而整幅图形看起来就像很多星星

蛛网图和雷达图将多个多变量个体放在同一张图形上，看起来就像是蛛网或雷达的形状，这样重叠的图形就称为蛛网图或者雷达图。简单说来，就是星状图有若干个中心，而蛛网图和雷达图只有一个中心。

 **base R已经足够使用**

#### 基本介绍
R 中星状图的函数为 `stars()`
* 参数 `x` 为一个多维数据矩阵或数据框，每一行数据将生成一个星形；
* `full` 为逻辑值，决定了是否使用整圆（或半圆）；
* `scale` 决定是否将数据标准化到区间 $[0,1]$ 内；
* `radius` 决定是否画出半径；
* `labels` 为每个个体的名称，默认为数据的行名；
* `locations` 以一个两列的矩形给出每个星形的放置位置，默认放在一个规则的矩形网格上，若提供给该参数一个长度为 2 的向量，那么所有的星形都将被放在该坐标上，从而形成蛛网图或雷达图；

#### 图形示例
![R语言 统计可视化-38](/assets/images/r-learning-notes/r-language-stat-visualization-38.png)

#### base R
```r
## 绘制汽车数据的星状图
## 预设调色板，stars() 默认用整数来表示颜色
palette(rainbow(12, s = 0.6, v = 0.75))
stars(mtcars[, 1:7], len = 0.8, key.loc = c(14, 1.5), ncol = 7,
      main = "", draw.segments = TRUE)
palette("default") # 恢复默认调色板
```

### 脸谱图
脸谱图由 Chernoff 提出，它以一种非常形象有趣的方式来展示多元数据：人的脸部（确切来说是头部）有很多特征，例如眼睛大小、眉毛弧度、脸宽、鼻高等，由于这些特征都可以用数值大小来测量，因此我们也可以反过来将一批数值对应到这些脸部特征上来

**TeachingDemos** 包提供了两个脸谱图函数 `faces()` 和 `faces2()` ，两个函数能反映的面部特征不尽相同，各有所长，例如 `faces()` 可以画头发和耳朵，但 `faces2()` 可以画更多的变量，这里我们只介绍后者

**在众多统计图形中，脸谱图可算是最有幽默味道的一种，读者不妨在一些轻松的场合或听众精力不集中时尝试使用这种图形，也许能让听众感觉眼前一亮，主动解读图中的数据。**

#### 基本介绍
`faces2()` 函数是我们最需要研究的
* `mat` 是主要参数，它是一个数据矩阵，每一行对应着一张脸谱，脸谱中各个部位的特征对应着矩阵中的列；
* `which` 也是一个重要参数，它用来指定数据矩阵中的每一列分别对应着何种面部特征，它是一个整数向量，向量的每个元素取值在 1 到 18 之间

#### 图形示例
![R语言 统计可视化-39](/assets/images/r-learning-notes/r-language-stat-visualization-39.png)
#### face2
```r
## 绘制部分汽车数据的脸谱图
library(TeachingDemos)
faces2(mtcars[, c("hp", "disp", "mpg", "qsec", "wt")],
       which = c(14, 9, 11, 6, 5))
```

### 三元图
三元图是一类非常特殊的统计图形 它只能处理列数为3 每一行的和为1 或者 100 的数据；他们往往是化学中的成分数据 比如某混合物三种成分的百分比

对于三元图 我们需要包 `vcd` 进行绘制

具体的函数介绍可以查阅帮助

#### 图形示例
![R语言 统计可视化-40](/assets/images/r-learning-notes/r-language-stat-visualization-40.png)
#### vcd
```r
## 绘制土壤样本三元图
data(murcia, package = "MSG")
library(vcd)
ternaryplot(murcia[, 2:4], main = "",
            dimnames = c("砂粒", "粉粒", "黏粒"),
            col = MSG::vec2col(murcia$site), cex = .5)
```

### 马赛克图
马赛克图（Mosaic Plots）是展示多维列联表多元统计分析：列联表数据的工具 它对列联表的维数没有限制 不像关联图和四瓣图仅限于低维的列联表

马赛克图的表现形式为与频数成比例的矩形块，整幅图形看起来就像是若干块马赛克放置在平面上。马赛克图背后的统计理论是对数线性模型（log-linear model）

#### 基本介绍
R 中马赛克图的函数为 `mosaicplot()`
* `x` 为一个列联表数据（可以用函数 `table()` 生成）；
* `main` 、 `sub` 、 `xlab` 和 `ylab` 分别设定主标题、副标题和坐标轴标题；
*  `sort` 指定展示变量的顺序；
* `dir` 指定马赛克图的拆分方向（横向拆分或纵向拆分）；
* `type` 给定残差的类型

#### 图形示例
![R语言 统计可视化-41](/assets/images/r-learning-notes/r-language-stat-visualization-41.png)
#### base R
```r
## 绘制泰坦尼克号生还数据的马赛克图
data(Titanic)
par(mar = c(2, 3.5, .1, .1))
mosaicplot(Titanic, shade = TRUE, main = "")
```

### 因素效应图
方差分析研究多组别的因变量均值有没有显著差异

因素效应图可以被看成一种弱化的且推广的方差分析问题 我们可以比较更多的统计量而不仅仅是均值 但是我们失去了统计学检验方法 只拥有探索性的分析作用 偶尔还是很有作用的 毕竟我们可以选用一些更加稳健的方法

**方差分析就配套类似因素效应的作图问题**
#### 基本介绍
R 中因素效应图的函数为 `plot.design()`
* `x` 为包含自变量（分类变量）的数据框，它也可以包含因变量，这种情况下第二个参数就不必提供了；
* `y` 为因变量；
* `fun` 为计算因变量水平的函数；

#### 图形示例
![R语言 统计可视化-42](/assets/images/r-learning-notes/r-language-stat-visualization-42.png)

#### base R
```r
## 经纱断裂数据的因素效应图
data(warpbreaks)
names(warpbreaks) = c("断裂数目", "羊毛种类", "拉力强度")
par(mfrow = c(2, 1), mar = c(4.5, 4, 0.2, 0.2))
plot.design(warpbreaks, col = "blue",
            xlab = "因素", ylab = "断裂数目均值")
plot.design(warpbreaks, fun = median, col = "blue",
            xlab = "因素", ylab = "断裂数目中值")
```

### 交互效应图
在回归模型或方差分析中，我们常遇到交互效应的概念；交互效应图一般针对分类变量之间的交互

看一个分类变量给定分类水平时，因变量在另一个分类变量各水平下的均值如何变化，这种变化趋势如果在前一个分类变量换一个取值水平后仍然保持相同的话，则说明这两个分类变量没有交互效应。

#### 基本介绍
R 中交互效应图的函数为 `interaction.plot()`
* `x.factor` 是横坐标上的分类变量；
* `trace.factor` 是第二个分类变量，按照这个分类变量的不同取值水平将 `x.factor` 分类下的因变量均值连接起来；
* `response` 是因变量；
* `fun` 是指定的对因变量汇总的函数，默认为均值，当然我们也可以指定其它计算函数如中位数 `median()`
#### 图形示例
![R语言 统计可视化-43](/assets/images/r-learning-notes/r-language-stat-visualization-43.png)
#### base R
```r
## 法国食道癌数据的交互效应图
data(esoph)
par(mar = c(4, 4, 0.2, 0.2))
with(esoph, {
  interaction.plot(agegp, alcgp, ncases / (ncases + ncontrols),
                   trace.label = "饮酒量", fixed = TRUE,
                   xlab = "年龄", ylab = "患癌概率")
})
```

### 分类与回归树（决策树）
分类与回归树（Classification and Regression Tree，CART）是一种递归分割（Recursive Partition）技术，它的目的是寻找自变量的某种分割，使得样本分割之后因变量各组之间的差异最大。这种分割会一直递归进行下去，直到满足停止条件。 他就是我们研究的决策树模型 机器学习导论与监督学习：决策树

#### 基本介绍
**rpart** 包提供了分类与回归树的计算拟合函数 `rpart()` ，该函数包同时也扩充了泛型函数 `plot()` ，凡是 rpart 类型的对象在作图时都会自动调用 `plot.rpart()` 生成树图。

**我们在脱离`mlr3`以后就有着更多的作图自由度，毕竟它只是一个整合，必然对原始功能有所抛弃来精简实现**

* `x` 是一个 rpart 类型的对象，一般由 `rpart()` 函数拟合产生；
* `uniform` 决定是否在从上至下的枝节点之间使用相等的纵向距离以避免树枝在某些局部区域靠得太近使图形难以辨认，
* `branch` 设定树枝的形状，0 为 “V” 字型，1 为垂直的形状，该参数可以取$[0,1]$ 之间的数值以使得数值形状更像 “V” 或更垂直；
* `compress` 设定是否在横向上压缩树枝的间距使得图形更紧凑

#### 图形示例
![R语言 统计可视化-44](/assets/images/r-learning-notes/r-language-stat-visualization-44.png)
#### base R
```r
## 脊椎矫正手术结果的分类树图
library(rpart)
data(kyphosis, package = "rpart")
levels(kyphosis$Kyphosis) = c("不存在", "存在")
names(kyphosis)[c(2, 4)] = c("年龄", "位置")
fit = rpart(Kyphosis ~ `年龄` + Number + `位置`, data = kyphosis)
par(mar = rep(1, 4), xpd = TRUE)
plot(fit, branch = 0.7)
text(fit, use.n = TRUE, digits = 7)
```

### 平行坐标图
平行坐标 是对通常的笛卡尔坐标思维的替代，我们知道，笛卡尔坐标系通常情况下最多只能容纳两个变量（横轴 x 纵轴 y），所以在这样的坐标系下无法直接画出多个变量，当然，前面提到了很多变通方法，使得多元数据可以在笛卡尔坐标系下被表达出来

平行坐标系的基本做法是将相互垂直的坐标轴改成平行的坐标轴，由于平面上可以容纳很多平行线，所以平行坐标系中可以放置多个变量。

而对于一行观测数据，由于它有多列，每一列都相应对应着一根平行线上的点，最终我们把这些点用折线连起来，也就形成了构成平行坐标图的基本元素。

类似地，多行数据就能描绘出多条折线，平行坐标图就是由这些折线加上相应的平行坐标轴构成的。

平行坐标图有非常多实现方式 我们介绍最为推荐的 基于`ggplot2`系统的 **GGally** 包中的 `ggparcoord()` 函数

**平行坐标图也成为轮廓图，不过此时一般样本点的数量较少**

#### 图形示例
![R语言 统计可视化-45](/assets/images/r-learning-notes/r-language-stat-visualization-45.png)

平行坐标图中线段相交则意味着负相关、平行则意味着正相关

由于平行坐标图画出了多个的变量，有时候我们可以借助图中折线的位置来观察聚类现象

平行坐标图中的变量顺序非常重要，它直接影响了图的外观，也限制了我们对数据的观察，尤其是相关关系，因为从平行坐标图中我们只可能观察相邻变量之间的关系。有时候将变量顺序交换一下，则也许可以观察到新的信息。

#### ggplot
```r
## 鸢尾花数据的平行坐标图
data("iris")
library(GGally)
names(iris)[1:4] = c("花萼长度", "花萼宽度", "花瓣长度", "花瓣宽度")
p = ggparcoord(iris, columns = 1:4,
               groupColumn = 5, scale = "uniminmax") +
  geom_line(size = 1.2) +
  labs(x = "变量", y = "数值", color = "种类")
print(p)
```

### 调和曲线图
调和曲线图由 Andrews 提出，它是一种巧妙的展示多元数据的技术
#### 数学原理
对于一个数据矩阵$X_{n\times p}$, 我们把其中每一行$X_i=(X_{i,1},\ldots,X_{i,p})$ 转化为一条曲线：
$$\left.f_{i}(t)=\left\{\begin{array}{cc}\frac{X_{i,1}}{\sqrt{2}}+X_{i,2}\sin(t)+X_{i,3}\cos(t)+\cdots\\+X_{i,p-1}\sin(\frac{p-1}{2}t)+X_{i,p}\cos(\frac{p-1}{2}t)&\text{若}p\text{为奇数}\\\frac{X_{i,1}}{\sqrt{2}}+X_{i,2}\sin(t)+X_{i,3}\cos(t)+\cdots\\+X_{i,p}\sin(\frac{p}{2}t)&\text{若}p\text{为偶数}\end{array}\right.\right.$$

 其中$t\in[-\pi,\pi]$。这样一来，将$t$ 取一系列值，则每一行观测数据都可以画出一条曲线，最终可以得到 $n$ 条曲线，也就形成
 了调和曲线图。这种数学转化表面上看起来很不直观，然而它却有很多好的数学性质及对应的实际意义， 这里仅列举两条：

1. 如果我们用 $L_{2}$ 范数来度量两条曲线之间的距离，那么得到的距离值正好是欧氏距离平方的 $\pi$ 倍，换句话说，两行观测之间的距离恰好可以表现为图中两条曲线之间的差距。这条性质使得我们可以直观地在图中观察聚类现象和离群点，因为聚类和离群点的概念都是基于距离的 (距离的定义有多种，这里用欧氏距离的平方) 。如果读者感兴趣，可以验证一下这个 $L_{2}$ 范数的结果：

$$
\int_{-\pi}^\pi\left(f_i(t)-f_j(t)\right)^2dt=\pi\sum_{k=1}^p\left(X_{i,k}-X_{j,k}\right)^2
$$

2. 这个变换从一定程度上保持了线性性，即：若一个观测 $X_l$ 的所有数值都小于 $X_i$ 而大于 $X_j$, 那么在调和曲线图上 $X_l$ 对应的曲线也位于 $X_i$ 和 $X_j$ 之间。这一点性质是非常明显的。
**这两条性质暂且用于分析调和曲线图形**
#### 基本介绍
可以参考`MSG`包中的 `andrews_curve()` 函数
* x 是数据矩阵
* n 为画曲线时取点的个数

有资料现实有包 **andrews** 提供了可用于绘制该曲线的函数，不过其图形精致程序略显逊色。
#### 图形示例
![R语言 统计可视化-46](/assets/images/r-learning-notes/r-language-stat-visualization-46.png)
#### MSG
```r
## 鸢尾花数据和黑莓树数据的调和曲线图
data(iris)
data(trees)
library(MSG)
iris.col = vec2col(iris$Species)
par(mfrow = c(2, 2))
par(mar = c(4, 4, 0.2, 0.2))
andrews_curve(iris[, 1:4], n = 50, col = iris.col,
              xlab = "t", ylab = "f(t)")
legend("topleft", col = unique(iris.col), lty = 1, bty = "n",
       legend = unique(iris$Species))
andrews_curve(iris[, c(3, 4, 2, 1)], n = 50, col = iris.col,
              xlab = "t", ylab = "f(t)")
andrews_curve(scale(iris[, 1:4]), n = 50, col = iris.col,
              xlab = "t", ylab = "f(t)")
x = andrews_curve(scale(trees), n = 50,
                   xlab = "t", ylab = "f(t)")
```

## 矩阵图形
矩阵图形在直观看来有两个变量 行坐标与列坐标 但是它对应各个坐标还有这一个取值 由于形式上确实特殊 因此这里进行单独的介绍

### 等高图和等高线
这是一种把原始的三维矩阵数据降维的方法 对于纸质文章而言 将高维数据降维表示是永恒的问题 **毕竟三维图形涉及视角的问题 很难找到合适的角度体现全部的信息**

等高图的思想来自地理上的等高线 只是将坐标变成了不连续的矩阵行列
#### 基本介绍
R中绘制等高图和等高线需要 `contour` 函数
* `nlevels` 设定等高线的条数 越多图就会越密
* `levels` 设定等高线的$z$值 在这个值附近的点才会被连接
* `methon` 设定画法 `simple` 在线末端加标签 重叠添加 `edge` 嵌入标签 `flattest` 在线平缓的地方加标签
* `x`和`y`是表示网格点的向量，它们定义了在二维平面上绘制等高线的位置
* `z`是一个矩阵，表示在`(x, y)`网格点上的函数值

#### 图形示例
![R 统计可视化](/assets/images/r-learning-notes/r-stat-visualization.png)
疏密体现了一种聚类特征
#### base R
```r
## 基础作图法绘制中国 31 地区国民预期寿命和高学历人数密度等高图
library(KernSmooth)
data(ChinaLifeEdu, package = "MSG")
par(mar = c(4, 4, 0.2, 0.2))
est = bkde2D(ChinaLifeEdu, apply(ChinaLifeEdu, 2, dpik))
contour(est$x1, est$x2, est$fhat, nlevels = 15, col = "darkgreen",
        vfont = c("sans serif", "plain"),
        xlab = "预期寿命", ylab = "高学历人数")
points(ChinaLifeEdu, pch = 20)
```
#### ggplot
```r
## ggplot2 绘制中国 31 地区国民预期寿命和高学历人数密度等高图
library(KernSmooth)
library(metR)
data(ChinaLifeEdu, package = "MSG")
est = bkde2D(ChinaLifeEdu, apply(ChinaLifeEdu, 2, dpik))
est_tidy = data.frame(
  life = rep(est$x1, length(est$x2)),
  edu = rep(est$x2, each = length(est$x1)),
  z = as.vector(est$fhat)
)
levels = pretty(range(est_tidy$z, finite = TRUE), 15)
p = ggplot(est_tidy, aes(life, edu)) +
  geom_contour(aes(z = z), breaks = levels) +
  geom_text_contour(aes(z = z)) +
  geom_point(aes(Life.Expectancy, High.Edu.NO), data = ChinaLifeEdu) +
  labs(x ="预期寿命", y = "高学历人数")
print(p)
```

### 颜色等高图
在原理上和等高图没区别 只是使用颜色的不同来体现等高特征
#### 基本介绍
R 中的颜色等高图函数为 `filled.contour()`

大多数参数与 `contour()` 函数完全相同，区别在于多了几个定义颜色的参数 颜色调整不属于我们这里需要研究的内容

#### 图形示例
![R 统计可视化-1](/assets/images/r-learning-notes/r-stat-visualization-01.png)
#### base R
```r
## 火山高度数据颜色等高图
par(mar = c(4, 4, 2, 2), cex.main = 1)
x = 10 * 1:nrow(volcano)
y = 10 * 1:ncol(volcano)
filled.contour(x, y, volcano,
               color = terrain.colors,
               plot.title = title(
                 xlab = "北部长度（米）", ylab = "西部长度（米）"
               ),
               plot.axes = {
                 axis(1, seq(100, 800, by = 100))
                 axis(2, seq(100, 600, by = 100))
               },
               key.title = title(main = "高度\n(米)"),
               key.axes = axis(4, seq(90, 190, by = 10))
)
```

### 颜色图
颜色图是一种软化的颜色等高图 我们不再进行平滑处理 而是单纯的用颜色映射出一个矩阵 用颜色方块来表示数值的大小

**颜色图是矩阵数据的一种可视化手段，诸如协方差矩阵 相关系数矩阵都可以用颜色图来可视化，颜色比看数字直观的多**

由于相关系数颜色涂使用的过于广泛，因此我们单独研究[相关系数热力图](/blog/2024/03/15/r-visualization-learning-notes/)
#### 基本介绍
R 中颜色图的函数为 `image()`
* 参数 `x` 、 `y` 、 `z` 与等高线的参数类似
* `col` 设置一个颜色序列以便映射到不同大小的数值
* `breaks` 给定 `z` 分段的区间端点

#### 图形示例
![R 统计可视化-2](/assets/images/r-learning-notes/r-stat-visualization-02.png)

#### base R
```r
## 基础作图法绘制火山高度数据颜色图
data(volcano)
par(mar = rep(0, 4), ann = FALSE)
x = 10 * (1:nrow(volcano))
y = 10 * (1:ncol(volcano))
image(x, y, volcano, col = terrain.colors(100), axes = FALSE)
contour(x, y, volcano, levels = seq(90, 200, by = 5),
        add = TRUE, col = "peru")
box()
```

#### ggplot
```r
## ggplot2 绘制火山高度数据颜色图
data(volcano)
library(ggplot2)
p = ggplot(transform(reshape2::melt(volcano),
                 x = Var1 * 10, y = Var2 * 10),
       aes(x = x, y = y, z = value, fill = value)) +
  geom_tile() +
  geom_contour() +
  scale_fill_distiller(palette="RdYlGn") +
  labs(x = "北部长度（米）", y = "西部长度（米）",
       fill = "高度\n(米)")
print(p)
```

### 三维透视图
这就是等高线图的三维表示
自然的 `ggplot2` 不会提供我们需要的解决方式
#### 基本介绍
R 中透视图的函数为 `persp()`
* 参数 `x` 、 `y` 、 `z` 与等高线的参数类似
* `theta` 和 `phi` 分别设定立体图形左右方向和上下方向旋转的角度
* `r` 设定眼睛离透视图中心的距离

特别的
* **grDevices** 包提供了一个相关的三维透视图转换函数 `trans3d()` ，它可以将一个空间的点的三维坐标根据透视图的特征转换为平面坐标，这样我们就可以很方便地使用一般的底层作图函数向立体图中添加图形元素
* **scatterplot3d** 包作为一个专用的三维作图包 提供了很多作图方法
* **rgl**包基于OpenCV编写 提供了可以交互的三维图形
#### 图形示例
![R 统计可视化-3](/assets/images/r-learning-notes/r-stat-visualization-03.png)
#### base R
```r
## 火山的三维透视图
data("volcano")
z = volcano
x = 4 * (1:nrow(z))
y = 4 * (1:ncol(z))
par(mar = rep(0, 4))
persp(x, y, z, theta = 150, phi = 30, col = "green3", ltheta = -120,
      shade = 0.75, scale = FALSE, border = NA, box = FALSE)
```

### 矩阵图 矩阵点 矩阵线
矩阵图的名称来自于其参数类型，它可以针对一个矩阵将所有列以曲线的形式表达出来，同一元函数曲线图[一元函数曲线](/blog/2024/03/15/r-visualization-learning-notes/)一样，它也没有什么特别之处，仅仅是提供了一个便利的封装，我们可以不必调用 `lines()` 等函数依次对矩阵的所有列画曲线。

#### 基本介绍
R 中矩阵图的函数为 `matplot()`，矩阵点的函数为 `matpoints()`，矩阵线的函数为 `matlines()`
函数 `matplot()` 为高层作图函数（创建新图形），而后两个函数均为低层作图函数（向现有图形上添加元素）
* 参数 `x` 和 `y` 为输入的矩阵，做图的方式是用 `x` 的列为横轴方向的变量， `y` 的列为纵轴方向的变量，然后用这些列依次作散点图（ `x` 的第一列对 y 的第一列， `x` 的第二列对 `y` 的第二列，依次类推）；
* 如果这两个参数有一个缺失，那么 x 将被 `1:nrow(y)` 代替

#### 图形示例
![R 统计可视化-4](/assets/images/r-learning-notes/r-stat-visualization-04.png)
#### base R
```r
## 基础作图法用矩阵图画出的一系列正弦曲线
sines = outer(1:20, 1:4, function(x, y) sin(x / 20 * pi * y))
par(mar = c(2, 4, .1, .1))
matplot(sines, type = "b", pch = 21:24, col = 2:5, bg = 2:5)
## 数据矩阵的前 6 行
round(head(sines), 5)
```

#### ggplot
```r
## ggplot2 画出的一系列正弦曲线
sines = outer(1:20, 1:4, function(x, y) sin(x / 20 * pi * y))
df = expand.grid(x = 1:20, y = factor(1:4))
df$sines = as.vector(sines)
p = ggplot(df, aes(x = x, y = sines, color = y)) +
  geom_point(aes(shape = y)) +
  geom_line()
print(p)
```

### 热图
热图是在颜色图的基础上增加了行与列的层次聚类实现的 同时他会为我们准备谱系图 体现一些聚类特征

**heapmap本身不提供添加数字的功能，也不易添加图例，因此一般不用于相关系数热力图的绘制，而只用于聚类问题**
#### 基本介绍
R 中热图函数为 **stats** 包中的 `heatmap()`
* 其中 `x` 是数据矩阵，它的类型只能是矩阵，不能是数据框或其它类型；
* `Rowv` 和 `Colv` 分别决定了行和列如何计算层次聚类和重新排序，若设置为 `NULL`（默认），则按层次聚类的结果将行和列重新排序并相应画谱系图，若为 `NA` 的话，则不画谱系图；
* `distfun` 决定用哪个函数计算距离以便进一步计算聚类，默认为 `dist()` ；
* `hclustfun` 决定用哪个函数计算层次聚类；
* `...` 参数传递给 `image()` ，所以我们还可以利用 `image()` 的参数来调整图形外观，比如用 `col` 设置单元格的颜色系列

#### 图形示例
![R 统计可视化-5](/assets/images/r-learning-notes/r-stat-visualization-05.png)

#### base R
```r
## 汽车数据的热图
## 用极端化调色板
library(RColorBrewer)
heatmap(as.matrix(mtcars), col = brewer.pal(9, "RdYlBu"),
        scale = "column", margins = c(4, 8))
```

### 关联图
关联图（Cohen-Friendly Association Plot）是展示二维列联表数据的一种工具 它主要是基于列联表的独立性检验理论（Pearson χ2 检验）生成的图形
体现了数据是否符合我们的期望

#### 基本介绍
R 中关联图的函数为 `assocplot()`
* 其中 `x` 为一个列联表数据（或者矩阵）；
* `col` 为朝上和朝下矩形的颜色；
* `space` 用来设置矩形之间的间距。

#### 图形示例
![R 统计可视化-6](/assets/images/r-learning-notes/r-stat-visualization-06.png)
#### base R
```r
## 眼睛颜色与头发颜色的关联图
data(HairEyeColor)
x = margin.table(HairEyeColor, c(1, 2))
rownames(x) = c("黑色", "棕色", "红色", "金色")
colnames(x) = c("棕色", "蓝色", "褐色", "绿色")
assocplot(x, xlab = "头发", ylab = "眼睛")
```

### 四瓣图
四瓣图（Fourfold Plot）是用来查看 2×2×k 列联表中两个二分变量之间关联关系的一种图示工具，它主要是基于二维列联表的检验理论而建立起来的

它从优比（Odds Ratio，OR）的角度出发对列联表进行检验、

它将优比体现在两个相邻的四分之一圆的半径之比上，如果两个扇形半径差异显著，那么说明行列变量不独立，即因素对事件有影响，这便是四瓣图最基本的用法，而背后还有关于优比置信区间的计算，并且这个置信区间也在图中用两道弧线表现了出来，四瓣图最终的读法就是观察两瓣相邻扇形的置信区间弧线是否有重叠，有则说明不能拒绝零假设，反之可以拒绝。这是基于假设检验和区间估计之间的转换关系而得以成立的

#### 基本介绍
R 中四瓣图函数 `fourfoldplot()`
* `x` 是一个 2×2×k 的数组，当 k=1 时，它也可以直接取一个 2×2 的矩阵；
* `color` 设定四分之一圆的填充颜色，处于同一对角线上的扇形颜色相同，颜色填充的顺序也反映出优比与 1 的大小；
* `onf.level` 为置信水平；
* `std` 为列联表的标准化方法，决定了标准化时分母所除的数。
* 当 k≥1 时，该函数会依次生成 k 幅四瓣图

#### 图形示例
![R 统计可视化-7](/assets/images/r-learning-notes/r-stat-visualization-07.png)
#### base R
```r
## 加州伯克利分校录取数据四瓣图
data("UCBAdmissions")
dimnames(UCBAdmissions) <-
  list(`录取情况` = c("录取", "拒绝"),
       `性别`= c("男性", "女性"),
       `院系` = LETTERS[1:6])
fourfoldplot(UCBAdmissions, mfcol = c(2, 3)) # 2 行 3 列排版
```

### 相关系数热力图
相关性热图（相关性图），相信是大家最熟悉的一种数据可视化方法之一，各类文献中的出现频率很高。相关性图是一种表示两个变量之间相关关系的热图，几乎所有表达相关性数值的数据都可以用相关性图进行可视化。

最容易使用的相关系数热力图是 **ggcorrplot** 包提供的 `ggcorrplot`

只需要给他提供数据矩阵 他就是按照我们所期望的目标进行绘图，它本质是调用了 **ggplot2** 进行作图，但是他对函数进行了满足base R规范的重做

如果想要使用 base R 的方法，可以使用 **corrgram** 包的 `corrgram` 函数 以及 **corrplot** 的 `corrplot` 函数
