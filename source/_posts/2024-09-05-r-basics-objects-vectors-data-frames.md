---
title: "R 基础：对象、向量、数据框与函数"
title_en: "R Basics: Objects, Vectors, Data Frames, and Functions"
date: 2024-09-05 10:29:12 +0800
categories: ["Programming", "R"]
tags: ["R"]
author: Hyacehila
excerpt: "整理 R 基本原理、对象、向量、矩阵、数据框、列表、时间序列、函数、循环和数据读写。"
excerpt_en: "Covers core concepts, objects, vectors, matrices, data frames, lists, time series, functions, loops, and data I/O."
mathjax: true
hidden: true
permalink: '/blog/2024/09/05/r-basic-learning-notes/'
---

## R的基本原理

R是一门非常简单的脚本化编程语言 它的Python非常的接近；和Python一样 我们对R的学习更多的是侧重于各种函数 各种Package中的内容，当然 在这之前我们需要对R 本身有一些了解 并且了解它基本的语法规则；

R和Python⼀样 不提供⼀体化的编辑器解决⽅案，建议直接使⽤RStutio，他已经能解决R语⾔的⼏乎所有需求 正如Pycharm和Python的关系

R中最重要的部分是R函数 所有的稍微有一点复杂的工作都是靠R函数实现的，**所有的合法的R函数都要表示为一个带有圆括号的形式 直接输入函数名儿没有括号 则是显示一些关于函数本身的内容 是否含有圆括号就是我们区分R函数 和 R对象的方法** 函数中可以没有参数

```r
ls()
ls
```

所有可以被使用的R函数存储在library中 如果需要使用 我们会把函数从packages中加载到library中 然后就可以使用了，我们可以在Rstudio的右侧辅助操作界面帮助我们加载R包

研究对象是最基本的操作，输入对象的名字来显示内容（也可以使用R Studio 中的Environment中检查）以及给对象进行赋值了（赋值的方式不止一种，如下）

```r
n <- 10
10 -> n
assign("n", 10)
n=10
n
```

对象的名字必须是以一个字母开头(A-Z 或a-z), 中间可以包含字母、数字(0--9)、点(.)及下划线 R对对象的名字区分大小写，x和X就可以代表两个完全不同的对象.

有时候我们也会只进行运算不进行赋值 此时结果不会保存到内存中 只会在Console中显示

R使用#作为注释符号 一个良好的代码 注释和文档都是不可缺少的

```r
## 这是一句注释
; #这是用来分割代码的符号
ctrl/command + shift + c #批量注释和批量取消注释
```

## R的数据结构

我们在前面介绍了 R基于对象和函数来运行 R函数是什么前面已经介绍了 但是R对象的种类并不唯一 所以我们需要逐一的进行介绍

### R对象的属性与对象信息
#### R对象属性

任何一个R对象都有两个内在属性：类型和长度.

类型是对象元素的基本种类，共有四种：

-   数值型
-   字符型
-   逻辑型
-   复数型（统计分析中少见，不介绍）
-   因子型 （一种比较特殊的类型，我们在对象类型里都会看到它，放在哪里都是正确的）

**R中需要要class属性支持面向对象的操作 class属性的不同可以让不同的对象类进行不同的操作 其实我们在很多地方已经使用这种对象类了 只是没有介绍**

长度是对象中元素的个数 我们可以用下面的函数研究对象的属性

```r
x<-1
mode(x)  #返回类型
length(x) #返回长度

is.numeric() #这类函数可以进⾏类型判断，返回布尔值
digits <- as.character(z) #这⼀类函数可以进⾏强制的类型转换
```

R中的对象往往有一些特殊的情况 他们使用一些特殊的符号表示 其中

-   NA表示缺失
-   Inf表示无穷
-   NaN表示非数字

字符型的值输入时须加上双引号"，如果需要引用双引号的话，可以让它跟在反斜杠"\\"后面

```r
x <- "Double quotes \" delimitate R's strings."
```

R中的对象基本有下面几种

| 对象   | 类型                      | 是否允许对象混合类型 |
| ---- | ----------------------- | ---------- |
| 向量   | 数值型，字符型，复数型，逻辑型         | 否          |
| 因子   | 数值型, 字符型                | 否          |
| 数组   | 数值型，字符型，复数型，逻辑型         | 否          |
| 矩阵   | 数值型，字符型，复数型，逻辑型         | 否          |
| 数据框  | 数值型，字符型，复数型，逻辑型         | 是          |
| 时间序列 | 数值型，字符型，复数型，逻辑型         | 否          |
| 列表   | 数值型，字符型，复数型，逻辑型 ，函数，表达式 | 是          |

至于R中的运算符 和Python中基本运算符没有什么区别 介绍一些发生了轻微变化和不怎么常用的

| 运算含义 | 运算符号 |
|----------|----------|
| 乘方     | \^       |
| 模       | %%       |
| 整除     | %/%      |

#### 检查对象信息

检查内存中全部对象（不如使用可视化的R studio） 不过它可以检索特定关键词的对象

```r
ls()
ls(pat = "m")
```

想要删除对象也是可以的 下面是用于删除对象的函数

```r
rm(x)
rm(list=ls(pat="^m"))
```

### 向量

#### 数值型向量

常用的建立方法都在下面了 还有什么其他的疑问可以使用help

```r
1:10 # 等差1序列
seq(1,5,by=0.5) #等差序列
rep(2:5,2) #重复某个向量化
c(42,7,64,9) #直接建立
scan()  #获取输入
```

#### 字符型向量

如下

```r
c("green","blue sky","-99")
paste(c("X","Y"), 1:10, sep="")
#把两个字符黏合成⼀个 得到一个新的字符
```

#### 逻辑向量

我们一般不会直接建立逻辑型向量 一般选择使用运算给出 结果有 T F NA 三种

```r
x <- c(10.4, 5.6, 3.1, 6.4, 21.7)
x > 13 # 逐个元素比较
7!=6
all(c(1, 2, 3, 4, 5, 6, 7) > 3)
#判断⼀个逻辑向量是否全为真
any(c(1, 2, 3, 4, 5, 6, 7) > 3)
#判断⼀个逻辑向量是否有真
```

#### 因子型向量

因子是一种单独的格式 但是我们还是把它纳入向量中研究了 这样是很自然的 对于那些取离散值的量 我们往往使⽤因⼦factor表示，他的存在可以帮助我们更好的研究分类问题

一个因子包括分类变量本身以及它的各种水平 建立因子往往是从其他类型中转化的 函数的使用格式如下

```r
factor(x, levels = sort(unique(x), na.last = TRUE),
       labels = levels, exclude = NA,ordered = is.ordered(x))

```

其中 x 表示我们引入的向量 levels 指定因子的水平（如果有序的话，这个部分可以帮助我们手工指定水平的顺序） lables指定水平的名字 exclude剔除一些水平 order指定因子是否有着次序

部分的情况下我们可以使用`gl（）`函数生成需要使用的因子
```r
gl(n, k, length = n*k, labels = 1:n, ordered = FALSE)
```

n是水平数 k是水平的重复次数 length是总观测数

下面给出一些基本的转化例子

```r
## 生成因子
a <- c("green", "blue", "green", "yellow")
factor(a)
b <- c(1,2,3,1)
factor(b)

## 重新指定因子水平
levels(b) <- c("low", "middle", "high")
b
```

除了建立 我们也可以对因子进行一些其他形式的检验

```r
sex <- c("M","F","M","M", "F")
sexf <- factor(sex)

is.factor(sexf) #检验是不是因⼦变量
as.factor(sexf) #转换向量为因⼦变量 没有上⾯的函数实⽤

levels(sexf) #得到因⼦的⽔平
table(sexf) #得到因⼦的频数

height <- c(174, 165, 180, 171, 160)
tapply(height, sex, mean)
#区分类型的元素 研究在某种因子水平下的数字特征 很实用，不过也可以用其他函数曲线救国实现
```

#### 向量的运算

数值型的向量存在运算的概念，作为涉及长度的一个对象 我们需要一套基本的语法规则来处理长度不一样的向量的运算的问题；

如果它们的长度不同,表达式的结果是一个与表达式中最长向量有相同长度的向量, 表达式中较短的向量会根据它的长度被重复使用若干次(不一定是整数次)，直到与长度最长的向量相匹配, 而常数将被不断重复 --- 这一规则称为循环法则(recycling rule)

```r
x <- c(10.4, 5.6, 3.1, 6.4, 21.7)
y <- c(x,0,x)
2*x + y + 1
```

对于向量和常数的运算 有一些比较自然的规则

-   向量与一个常数的加、减、乘、除为向量的每一个元素与此常数进行加、减、乘、除;
-   向 量 的 乘 方( ˆ )与 开 方(sqrt)为 每 一 个 元 素 的 乘 方 与 开 方, 这 对像`log(), exp(),sin(), cos(),tan()` 等普通的运算函数同样适用
-   同样长度向量的加、减、乘、除等运算为对应元素进行加、减、乘、除等
-   不同长度向量的加、减、乘、除遵从循环法则(recycling rule), 但要注意这种场合通常要求向量的长度为倍数关系,

最后 我们简单介绍一些那些非常常用的需要记忆的关于向量的函数 他们的复杂语法规则和那些不常用的函数在需要的时候单独再查询就可以了

| 函数                          | 作用              |
| --------------------------- | --------------- |
| `max(x) min(x)`             | 返回向量x中最大（最小）的元素 |
| `which.max(x) which.min(x)` | 返回最大最小元素的下标     |
| `mean(x)`                   | 均值              |
| `median(x)`                 | 中位数             |
| `var(x) sd(x)`              | 方差和标准差          |
| `quantile(x)`               | 四分位数            |
| `summary(x)`                | 常用的描述性统计量联合返回   |
| `sort(x)`                   | 排序              |
| `sum(x)`                    | 求和              |
| `cov(x,y)`                  | 协方差             |
| `cor(x,y)`                  | 相关系数（Pearson）   |

其中的部分函数也不仅仅针对向量进行操作 我们后面会再提到

#### 向量元素的提取

我们可以根据索引index来提取向量中的元素 使用方括号来进行 返回的向量长度和用于索引的向量的长度一致；比较特殊的情况是使用逻辑向量提取 此时一般要求两者长度一致 我们只提取逻辑中标记为TRUE的量；我们也可以使用提取这个功能对向量进行修改

修改可以和提取同步进行 负数的提取就是删除对应元素

**R语言的下标逻辑是从1开始的**

```r
x <- c(42,7,64,9,10,8)
x[1:5]
x[c(1,4)]
x[x>10]
x[x>10] <- 10
v[-(1:5)]
```

#### 关于特殊符号

```r
is.na() #检测向量是否有缺失 T意味着这是缺失数据
is.nan() #检测数据是否是不确定的
is.finite()  #检测数据是否有限
is.infinite() #检测数据是否⽆穷
```

### 数组和矩阵

和其他语言一样的是 我们这里把**数组视为一种多维的向量形式** 因此 我们前面介绍的向量就是一维的数组，而矩阵是二维的数组 是一种比较常用的特殊形式 **对于数组和矩阵 除了我们前面提到的长度和类型以外 还需要维数向量dim来描述**

#### 数组的建立

建立数组往往是使用向量来实现的 向量中含有我们在数组中要使用的元素

**一般使用`c()`建立向量 `array()`建立数组 `matrix()`建立矩阵**

```r
array(data, dim, dimnames)
```

其中data是向量 含有我们需要的元素 dim是维数向量 它的元素表示了我们的划分层级 它的长度多一 划分的层级就多了一级 最后的描述各个维的名字 如下

```r
A <- array(1:8, dim = c(2, 2, 2))
A
## 建立了一个三维的数组
```

#### 矩阵的建立

矩阵是数组的特例 是二维的数组 不过我们还是建议使用单独的函数来实现建立

```r
X <- matrix(1, nr = 2, nc = 2) #指定行数和列数
X <- matrix(1:4, 2, 4, byrow=TRUE) # 是否要从按列排转为按行排
v <- c(10, 20, 30)
diag(v) # 建立对角矩阵
```

#### 矩阵和数组的下标

和向量的一样 我们可以使用数和逻辑向量来提取内容 不过会稍微复杂一点点 整体上还是符合数学的逻辑；使用索引进行修改也是自然的

```r
x <- matrix(1:6, 2, 3) #2行3列
x[2,2] #第2行第2列
x[2,]  #只要第二行
x[,3] <- NA #对第三列操作
x[is.na(x)] <- 1 #生成矩阵布尔索引，很自然
```

想要去除某些行列可以使用负数下标

```r
x <- matrix(1:6, 2, 3)
x[-1,]
x[,-2]
```

#### 矩阵的运算

转置 对角线元素提取 按照行或者列合并

```r
X <- matrix(1:6, 2, 3)
t(X) #转置

X <- matrix(1:4, 2, 2)
diag(X)  #对角线提取

X<-matrix(1:4, 2,2)
det(X) #行列式

```

对应的元素进行运算方式如下 如果缺失则会按照向量的类似原则进行 行列式

```r
m2 <- matrix(2, nr = 2, nc = 2)
m2*m2 #直接乘法不使用矩阵乘法 而是对应元素乘法
```

### 数据框

data frame 的存在是统计分析中最核心的需要操作的对象类型了；它约定了每一个行是一次观测 每一列是一种变量 显示数据框是告诉我们行和列的编号 列一般还会有名字，他是符合Codd代数第三范式的数据结构 探索性数据分析

#### 数据框的建立

如果我们想在R中建立数据框可以采用下面的方式

```r
x=c(42,7,64,9)
y=1:4
z.df=data.frame(INDEX = y, VALUE = x)
z.df
```

在更多的情况下 **我们依赖从外部导入数据作为统计分析的数据框** 我们后面会单独研究怎么从外部导入数据的问题，所有从外部导入的都使用data frame存储 我们后面单独介绍数据存储的问题 在数据结构之后

#### 数据框的简单处理

我们用一个例子来说明这个问题

```r
attach(Puromycin)
#attach函数可以把数据框中的数据直接加载 就不需要用$调用了

summary(Puromycin)
## 数据框概述
head(Puromycin)
## 看看前几行

cor(Puromycin$conc,Puromycin$rate)
## 计算两个列的相关系数

pairs(Puromycin, panel = panel.smooth)
## 散点图pairs函数

xtabs(~state + conc, data = Puromycin)
## 交叉表函数 以后用到再详细研究

detach(Puromycin)
#取消attach的加载
```

#### 数据框元素的提取

数据框的提取基本和矩阵是相同的 因为他们有着近似的结构 但是由于它有名字 所以有其他的访问方法，我们还有限制条件的访问思路 也就是subset函数

```r
attach(Puromycin)
Puromycin[1, 1]
Puromycin[c(1, 3, 5), c(1, 3)]
## 类似矩阵访问方法

Puromycin$conc
Puromycin$state
## 按照名字访问

subset(Puromycin, state == "treated" & rate > 160)
## subset是数据框子集提取函数 适用于复杂的子集提取
```

#### 在数据框中添加变量

方法并不唯一 下面逐个介绍

```r
#最常用的方法 建立新的列
Puromycin$iconc <- 1/Puromycin$conc

#with可以从数据框中提取信息 用的不多
Puromycin$iconc <- with(Puromycin, 1/conc)

#transform是修正数据框用的函数
Puromycin <- transform(Puromycin, iconc = 1/conc, sqrtconc = sqrt(conc))
```

#### 数据框的实例标识符
在病例数据中，病人编号（patientID）用于区分数据集中不同的个体。在R中，实例标识符（case identifier）可通过数据框操作函数中的rowname选项指定，如下

```r
patientdata <- data.frame(patientID, age, diabetes,
                          status, row.names=patientID)
```

这种的实例标示操作可以帮助很多绘图函数工作，建议进行。
### 列表

列表是一类特殊的对象 它被用于比较复杂的数据分析工作 可以包含任何类型的对象 也就是把对象套娃成了一组对象；

列表可以用函 数list( )创 建；列表的下标与子集的提取也与数据框没有本质区别. 数据分析时通常是在提取部分对象后按上面讲述的向量、矩阵或数据框等运算进行,下面仅举一例进行说明.

```r
L2 <- list(x = 1:6, y = matrix(1:4, nrow = 2))
L2
L2$x
L2[1]
```

非常多函数都是返回列表 里面包含着运算的各种结果 因为他兼容并包 可以同时含有很多信息，同时他的语法规则也较为宽松

### 时间序列

时间序列是一种特殊类型的数据 我们可以创建一元或者多元的时间序列 它的特殊之处是元素顺序被赋予了价值；分析时间序列数据是统计学中一个重要的分支，我们会展开单独的研究

关于TS的一点基本介绍
```r
ts(data = NA, start = 1, end = numeric(0), frequency = 1,
    deltat = 1, ts.eps = getOption("ts.eps"), class, names)
```

-   data是被纳入时间序列的数据 可以是向量或者矩阵 体现了一元或者多元的时间序列
-   start是第一个观察值的时间 可以是一个数字 也可以是两个数字组成的向量
-   end 最后一个观察值的时间 规则同上
-   frequency 单位时间观测的频数 （一般取1，4，12意味着年季度月）
-   deltat 两个观测值的间隔 和frequency只能给出一个 是分数
-   class 对象的类型 一般缺省就好
-   names 多元序列中每个一元序列的名字

```r
ts(1:10, start = 1959)
ts(1:47, frequency = 12, start = c(1959, 2))
ts(1:10, frequency = 4, start = c(1959, 2))
ts(matrix(rpois(36,5),12,3), start=c(1961,1),frequency=12)
```

## 日期型变量
### 存储逻辑
日期型是一种特殊的类型，我们并没有一种专门的用于导入日期的数据结构，但是他确实非常的常用，因此我们这里进行了单独的补充介绍。

日期值通常以字符串的形式输入到R中，然后转化为以数值形式存储的日期变量。函数`as.Date()`用于执行这种转化。其语法为`as.Date(x, "input_format")`，其中`x`是字符型数据，`input_format`则给出了用于读入日期的适当格式.

### 格式表

| 符  号 | 含  义    | 示  例    |
| ---- | ------- | ------- |
| %d   | 数字表示的日期 | 01~31   |
| %a   | 缩写的星期名  | Mon     |
| %A   | 非缩写的星期名 | Monday  |
| %m   | 月份      | 01-12   |
| %b   | 缩写的月份   | Jan     |
| %B   | 非缩写的月份  | January |
| %y   | 两位数的年份  | 07      |
| %Y   | 四位数的年份  | 2007    |
### 示例与日期函数
一个帮助我们理解的例子有
```r
strDates <- c("01/05/1965", "08/16/1975")
dates <- as.Date(strDates, "%m/%d/%Y")
```

特别的，日期有两个专用的函数有
```r
Sys.Date() #返回今天日期
date() #返回现在的日期和时间

#我们可以实现反向的输出
today <- Sys.Date()
format(today, format="%B %d %Y")

#计算两个时间的间隔
dob   <- as.Date("1956-10-12")
difftime(today, dob, units="weeks")
```
## 数据存储和读取

如果我们只需要在R中进行研究 那么数据就用R的格式就好了 但是我们需要分析的数据一定会来自其他地方 导出的结果也需要提交给其他部分进行分析 所以介绍数据的存储和读取就是必要的了

原则上 所有的数据存储和读取工作都应该在WorkSpace进行 保存至Project 文件移动到Project再读取 不过我们也可以用指定目录的方式来进行

Rstudio为我们提供可视化的数据读取与读出方法
### 数据存储

常用的存储格式有
* 简单的文本文件txt
* 逗号分隔文本文件csv
* R数据文件Rdata

```r
d <- data.frame(obs = c(1, 2, 3), treat = c("A", "B", "A"), weight = c(2.3, NA, 9))

## 写txt
write.table(d, file = "c:/data/foo.txt", row.names = F, quote = F)

## 写csv
write.csv(d, file = "c:/data/foo.csv", row.names = F, quote = F)

## 写Rdata
save(object1, object2, file = "my_data.RData")
```

### 基本数据读取

经常被读取的文件格式有
* txt
* 从Excel中复制的剪贴板数据
* R中的数据集datasets 其中datasets数据集有加载和外挂两种形式 区别在于是否可以直接访问其中的变量
* Rdata

```r
read.table(file="houses.txt")
read.delim("clipboard")
data(mtcars)
attach(mtcars)
load("my_data.RData")
#我们可以自由的把读取的数据赋给需要的变量
```

### 高级数据读取

最常用的包是foreign包 它提供了多种读取其他软件数据的方式 首先介绍几大统计学软件

```r
library(foreign)
rs <- read.spss("educ_scores.sav") #spss的读取
rx <- read.xport("educ_scores.xpt") #sas的读取
rs <- read.S("educ_scores") #s的读取
rd <- read.dta("educ_scores.dta") #stata的读取
#除了spss以外 都是使⽤数据框格式

rs<-read.spss("educ_scores.sav", to.data.frame=TRUE) #这样spss读取进来的就是数据框了
```

Excel是比较特殊的统计学软件 它的xls不能被foreign包读取 我们可以在excel中转换为csv文件 或者使用另外的包

```r
rc <- read.csv("educ_scores.csv") #csv是最常用的读取方法
library(readxl)
so2_df <- read_excel('/Users/sylnne/Desktop/DAMS/Datasets/SO2.xlsx')
```

### 在数据的存储与读取最后
实际上，我们只推荐将数据保存为两种形式
* R自己的Rdata文件
* 导出为csv文件，使用 `write.csv()` 函数

我们只推荐一种数据导入的方法，基于Rstudio的可视化界面进行数据的导入，其中 base 类型对应 txt 与 csv 文件。其他类型按照字面对应。

## R编程

### 循环和控制

控制结构是编程语言的必备品 有下面是常用的控制结构的例子 整体的语法结构偏向C 没有python的高度灵活

```r
#分⽀语句
if (cond_1)
  {statement_1}
else if (cond_2)
  {statement_2}
else if (cond_3)
  {statement_3}
else
  {statement_4}
#还是⽼规矩的if else 结构
switch (statement, list)
#根据statement返回的值 决定返回列表的那个值 statemnt只有⼀个 列表可能很⻓ 别忘了他的定义
#如果找不到这个值 返回NULL  明显的 statement不⼀定是数
#cond语句和C语⾔⼀致 就跟括号⼀样

#循环语句 非常自然的循环写法
for (i in 1:5) {
  print(i)
}

i <- 1
while (i <= 5) {
  print(i)
  i <- i + 1
}
```

### 向量化

在R在, 很多情况下循环和控制结构可以通过向量化避免(简化): 向量化使得循环隐含在表达式中；

在实际设计中 尽量使用向量化语句而不是循环与分支 原因如下

-   代码更简洁
-   在R中使用向量化，R会立即调用C进行运算，并且会在可行的地方进行并行，因而大大提高了计算的效率

```r
y[x == b] <- 0
y[x != b] <- 1
```

向量化就是使用逻辑型变量来控制原本需要循环和分支才能实现的功能，在Python的`numpy` 中，我们依旧大量是使用编写好的向量化语句代替粗糙的循环控制，从而正确的执行多核的运算。在更加复杂的深度学习问题，如`Pytorch`框架，也需要通过类似的操作来调用GPU，相关内容会在Python的相关位置介绍。
`
### 程序与函数

大多数R的工作是通过函数来实现的, 而且这些函数的输入参数都放在一个括弧里面. 用户可以编写自己的函数, 并且这些函数和R里面的其它函数有一样的特性 一个函数的例子如下

```r
myfun <- function(S, F) {
  data <- read.table(F)
  plot(data$V1, data$V2, type="l")
  title(S)
}
```

调用函数可以使用顺序参数也可以用名字

```r
foo1(u, v, w)
foo1(arg3=w, arg2=v, arg1=u)
```

我们可以对自行定义的函数给出默认设置 这非常的实用

```r
foo2 <- function(arg1, arg2 = 5, arg3 = FALSE) {...}
```

r函数可以递归 这为很多算法的设计带来的便利 也是比较新的语言的标配

## 内置函数
### 数学函数
```r
abs(x) # 绝对值
acos(x) # 反余弦函数
asin(x) # 反正弦函数
atan(x) # 反正切函数
atan2(y, x) # 两个参数的反正切函数
ceil(x) # 向上取整
cos(x) # 余弦函数
cosh(x) # 双曲余弦函数
exp(x) # 指数函数
floor(x) # 向下取整
log(x) # 自然对数
log10(x) # 以10为底的对数
logb(x, base) # 以base为底的对数
sin(x) # 正弦函数
sinh(x) # 双曲正弦函数
sqrt(x) # 平方根
tan(x) # 正切函数
tanh(x) # 双曲正切函数
```
### 统计函数
```r
mean(x) # 计算平均值
median(x) # 计算中位数
sum(x) # 计算总和
min(x) # 计算最小值
max(x) # 计算最大值
range(x) # 计算范围（最大值与最小值的差）
diff(x) # 计算差分（相邻元素的差）
prod(x) # 计算乘积
var(x) # 计算方差
sd(x) # 计算标准差
cor(x, y) # 计算相关系数
cov(x, y) # 计算协方差
quantile(x, probs) # 计算分位数
t.test(x, y) # 进行t检验
chisq.test(x) # 进行卡方检验
cor.test(x, y) # 进行相关性检验
```
### 概率函数
研究密度函数和分布率是概率论中的重要部分 对应的我们还会有CDF函数 程序设计中，我们不区分连续或者离散型不过PDF和CDF都是重要的需要研究的量

R语言对每一种分布提供四种常用的函数 他们分别是 密度函数（PDF）分布函数（CDF）分位数函数 随机数函数 他们和分布的R名称相对应 用修饰词来实现四种功能

我们先给出各种分布的R名称在下表中

| 分布名称             | R名称    | 参数控制            |
|----------------------|----------|---------------------|
| beta                 | beta     | shape1, shape2      |
| binomial             | binom    | size, prob          |
| Cauchy               | cauchy   | location=0, scale=1 |
| exponential          | exp      | rate                |
| chi-sqaured (χ2)     | chisq    | df, ncp             |
| Fisher--Snedecor (F) | f        | df1, df2, ncp       |
| gamma                | gamma    | shape, scale=1      |
| geometric            | geom     | prob                |
| hypergeometric       | hyper    | m, n, k             |
| lognormal            | lnorm    | meanlog=0, sdlog=1  |
| logistic             | logis    | location=0, scale=1 |
| multinomial          | multinom | size, prob          |
| normal               | norm     | mean=0, sd=1        |
| negative binomial    | nbinom   | size, prob          |
| Poisson              | pois     | lambda              |
| Student's (t)        | t        | df                  |

对于给定的分布名称 我们使用下面的参数控制

-   d 表示 density 指密度函数 PDF
-   p 表示分布函数 CDF
-   q quantile 表示分位数函数
-   r random 表示随机模拟

这四类函数有自身的语法规则

-   密度函数 第一个参数是数值向量 返回向量对应点的概率值
-   分布函数 第一个参数是数值向量 返回向量对应点的概率值
-   分位数函数 第一个参数是概率向量 返回CDF对应点的分位数
-   随机模拟 第一个向量是数值向量 返回对应数量的随机数

`p` 和 `q` 两种函数都有`lower.tail`参数控制是从小到大还是从大到小计算 默认为TRUE 从左到右 `log.p`控制是否为对数化 默认为FALSE

### 抽样模拟函数
概率论在早期是为了研究游戏和赌博中存在的随机现象 了解他们的软件实现当然也是非常必要的了，只需要一个非常简单的函数并且控制一些他们的参数就可以解决关于抽样的各种问题

等可能不放回的抽样 其中x是抽取的向量 n是抽取的个数

```r
sample(x, n)
```

等可能的有放回的随机抽样:

```r
sample(x, n, replace=TRUE)
```

不等可能的随机抽样 最后的参数用来指定各个元素中出现的概率

```r
sample(x, n, replace=TRUE, prob=y)
```

随机抽样中有一些结果需要排列组合表示 R也提供了对应的计算方法

```r
#计算阶乘prod还是最方便的 它是向量所有元素的积
prod(52:49)
#组合数提供了专门的函数
choose(52,4)
```

### 字符处理函数
```r
nchar(x) # 计算字符串的长度
substring(x, start, stop) # 提取子字符串
strsplit(x, split, fixed = FALSE) # 按给定的分隔符分割字符串
paste(..., sep = " ", collapse = NULL) # 连接字符串
paste0(..., collapse = NULL) # 连接字符串，不添加分隔符
sprintf(format, ...) # 格式化字符串
toupper(x) # 将字符串转换为大写
tolower(x) # 将字符串转换为小写
trimws(x) # 去除字符串两端的空白字符
cat(..., sep = " ", fill = FALSE, labels = NULL, file = "") # 连接并打印字符串
gsub(pattern, replacement, x) # 替换字符串中的模式
sub(pattern, replacement, x) # 替换字符串中的第一个匹配模式
chartr(old, new, x) # 替换字符串中的字符
strtrim(x, side = "both") # 修剪字符串中的空白
strsplit(x, split, fixed = FALSE, useBytes = FALSE) # 按给定的分隔符分割字符串
grep(pattern, x, value = FALSE, fixed = FALSE, perl = FALSE, ...) # 搜索匹配的字符串
regexpr(pattern, text) # 返回匹配的正则表达式的位置
gregexpr(pattern, text) # 返回匹配的正则表达式的位置和匹配的字符串
```
### 将函数应用于矩阵和数据框
R中提供了一个`apply()`函数，可将一个任意函数“应用”到矩阵、数组、数据框的任何维度上

```r
apply(x, MARGIN, FUN, ...)
```

其中，`x`为数据对象，`MARGIN`是维度的下标，`FUN`是由你指定的函数，而...则包括了任何想传递给`FUN`的参数。在矩阵或数据框中，`MARGIN=1`表示行，`MARGIN=2`表示列

在R中使用一个或多个by变量和一个预先定义好的函数来折叠（collapse）数据是比较容易的。调用格式为：

```r
aggregate(x, by, FUN)
```

其中`x`是待折叠的数据对象，`by`是一个变量名组成的列表，这些变量将被去掉以形成新的观测，而`FUN`则是用来计算描述性统计量的标量函数，它将被用来计算新观测中的值。`by`中的变量必须在一个列表中

这是一个比较神奇的函数，还不太明白他的用处
