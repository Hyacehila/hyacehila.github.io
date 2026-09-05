---
title: "R 基础：对象、向量、数据框、函数与数据分析"
title_en: "R Basics: Objects, Vectors, Data Frames, Functions, and Data Analysis"
date: 2024-09-05 10:29:12 +0800
categories: ["Programming", "Programming Languages"]
tags: ["R", "Data Management", "Exploratory Data Analysis"]
author: Hyacehila
excerpt: "系统整理 R 的对象、向量、矩阵、数据框、列表、时间序列、函数与常用内置函数，并覆盖数据导入、数据管理、缺失值处理和探索性数据分析。"
excerpt_en: "A practical guide to R objects, vectors, matrices, data frames, lists, time series, functions, data import, data management, missing-data handling, and exploratory data analysis."
mathjax: true
hidden: true
permalink: '/blog/2024/09/05/r-basic-learning-notes/'
---

## R 的基本原理

R 是一门面向数据分析的编程语言。它的核心并不复杂：对象保存数据，函数处理对象，包则提供更多函数和数据集。这篇笔记从这些基础概念出发，再进入数据导入、整理与探索。

R 本身包含解释器和标准库，RStudio 是常用的集成开发环境。二者需要分别安装；RStudio 负责编辑代码、管理项目和展示对象，实际计算仍由 R 完成。

函数调用使用圆括号。直接输入函数对象的名字，R 会打印它的定义；加上圆括号才会调用函数。

```r
ls       # 查看函数对象
ls()     # 调用函数
```

函数也是对象，因此不能靠“是否带圆括号”判断一个对象是不是函数。需要检查时可以使用 `is.function()`。

```r
is.function(ls)
```

R 包安装在本地库（library）中，使用前通过 `library()` 将包加载到当前会话。包只需安装一次，但每次启动新的 R 会话都要重新加载。

```r
install.packages("readxl")  # 只需安装一次
library(readxl)             # 每个新会话都要加载
```

### 对象与赋值

R 常用 `<-` 赋值。`=`、`assign()` 和从右向左的 `->` 也能赋值，但混用会降低可读性。

```r
n <- 10
n = 10
assign("n", 10)
10 -> n
n
```

对象名区分大小写。常规名称以字母或不跟数字的点开头，后面可以包含字母、数字、点和下划线。`x` 与 `X` 是两个不同对象。

表达式没有赋值时，结果只会打印到控制台，不会自动保存。注释从 `#` 开始，一直延续到行末。

```r
2 + 3  # 打印 5，但不保存结果

result <- 2 + 3
result
```

## R 的数据结构

R 对象既有底层存储类型，也可能带有 `class`、`dim`、`names` 等属性。理解这两层信息，比把所有对象硬分成互斥的“类型”更准确。

### 对象属性与信息

原子向量常见的底层类型包括逻辑型、整数型、双精度数值型、复数型、字符型和原始字节型。因子不是独立的底层类型，而是带有 `levels` 和 `class = "factor"` 属性的整数向量。

```r
x <- 1:5

typeof(x)       # 底层存储类型
class(x)        # 对象类别
length(x)       # 元素个数
attributes(x)   # 对象属性
```

`is.*()` 函数用于判断对象，`as.*()` 函数用于显式转换。调用时要传入实际对象。

```r
is.numeric(x)
digits <- as.character(x)
```

R 使用几个特殊值描述无法按普通数值处理的情况：

- `NA` 表示缺失值。
- `NaN` 表示未定义的数值结果，例如 `0 / 0`。
- `Inf` 和 `-Inf` 表示正负无穷。
- `NULL` 通常表示对象或结果不存在，和长度为一的缺失值 `NA` 不同。

字符值放在单引号或双引号中。字符串内需要同类引号时，可以用反斜杠转义。

```r
message <- "Double quotes \" delimit R strings."
```

常见数据结构如下：

| 数据结构 | 主要特点 | 是否允许列或元素具有不同类型 |
| --- | --- | --- |
| 向量 | 一维、同质 | 否 |
| 因子 | 用整数编码分类水平 | 否 |
| 数组 | 多维、同质 | 否 |
| 矩阵 | 二维数组 | 否 |
| 数据框 | 二维表格，每列是一个向量 | 是 |
| 时间序列 | 带时间索引属性的向量或矩阵 | 否 |
| 列表 | 可以容纳任意对象 | 是 |

部分常用运算符：

| 运算 | 运算符 |
| --- | --- |
| 乘方 | `^` |
| 取模 | `%%` |
| 整除 | `%/%` |
| 矩阵乘法 | `%*%` |

`ls()` 可以列出当前环境中的对象，也可以按名称筛选。`rm()` 用于删除对象。

```r
ls()
ls(pattern = "^m")

rm(x)
rm(list = ls(pattern = "^m"))
```

### 向量

向量是一组同类型元素。`c()`、`:`、`seq()` 和 `rep()` 是最常用的创建方式。

#### 数值型向量

```r
1:10
seq(1, 5, by = 0.5)
rep(2:5, times = 2)
c(42, 7, 64, 9)
```

`scan()` 可以从控制台或文本连接读取数据，但脚本中通常更适合使用明确的数据导入函数。

#### 字符型向量

```r
colors <- c("green", "blue sky", "-99")
paste(c("X", "Y"), 1:2, sep = "")
```

`paste()` 会连接字符，`paste0()` 等价于 `paste(..., sep = "")`。

#### 逻辑型向量

逻辑值包括 `TRUE`、`FALSE` 和 `NA`。比较运算通常会生成逻辑向量。

```r
x <- c(10.4, 5.6, 3.1, 6.4, 21.7)
x > 13
7 != 6

all(1:7 > 3)
any(1:7 > 3)
```

`T` 和 `F` 可以被重新赋值，不适合在正式代码中代替 `TRUE` 和 `FALSE`。

#### 因子

因子用于表示分类变量。它保存观测对应的整数编码，并通过 `levels` 记录类别名称。类别有顺序时，可以设置 `ordered = TRUE`。

```r
colors <- c("green", "blue", "green", "yellow")
color_factor <- factor(colors)

scores <- factor(
  c(1, 2, 3, 1),
  levels = c(1, 2, 3),
  labels = c("low", "middle", "high"),
  ordered = TRUE
)
```

`gl()` 可以按规则生成因子。`n` 是水平数，`k` 是每个水平连续重复的次数，`length` 是结果长度。

```r
gl(n = 3, k = 2, length = 6, labels = c("A", "B", "C"))
```

检查和汇总因子时常用以下函数：

```r
sex <- factor(c("M", "F", "M", "M", "F"))
height <- c(174, 165, 180, 171, 160)

is.factor(sex)
levels(sex)
table(sex)
tapply(height, sex, mean)
```

#### 向量运算与循环补齐

向量运算按位置逐元素进行。两个向量长度不同时，较短向量会被重复使用，这称为循环补齐规则（recycling rule）。如果较长向量的长度不是较短向量长度的整数倍，R 通常会给出警告；依赖这种不完整循环补齐容易隐藏错误。

```r
x <- c(10.4, 5.6, 3.1, 6.4, 21.7)
x * 2 + 1

c(1, 2, 3, 4) + c(10, 20)
```

常用汇总函数：

| 函数 | 作用 |
| --- | --- |
| `min(x)`、`max(x)` | 最小值与最大值 |
| `which.min(x)`、`which.max(x)` | 最小值与最大值的位置 |
| `mean(x)`、`median(x)` | 均值与中位数 |
| `var(x)`、`sd(x)` | 方差与标准差 |
| `quantile(x)` | 分位数 |
| `summary(x)` | 根据对象类别返回常用摘要 |
| `sort(x)` | 排序 |
| `sum(x)`、`prod(x)` | 求和与连乘 |
| `cov(x, y)`、`cor(x, y)` | 协方差与相关系数 |

许多汇总函数都有 `na.rm` 参数。数据含有缺失值时，需要明确决定是否排除缺失值，而不是机械地加上 `na.rm = TRUE`。

```r
mean(c(1, 2, NA), na.rm = TRUE)
```

#### 向量元素的提取

R 的索引从 1 开始。方括号内可以使用正整数、负整数、逻辑向量或名称；正负索引不能混用，`0` 除外。

```r
x <- c(42, 7, 64, 9, 10, 8)

x[1:5]
x[c(1, 4)]
x[x > 10]
x[-(1:5)]

x[x > 10] <- 10
```

检查特殊数值时，要把对象传给判断函数：

```r
values <- c(1, NA, NaN, Inf)

is.na(values)
is.nan(values)
is.finite(values)
is.infinite(values)
```

### 数组和矩阵

数组是带 `dim` 属性的同质向量，矩阵是二维数组。它们只能保存一种底层类型；如果混入字符值，数值通常会被转换为字符。

#### 创建数组和矩阵

```r
A <- array(1:8, dim = c(2, 2, 2))
A

X <- matrix(1:8, nrow = 2, ncol = 4)
X_by_row <- matrix(1:8, nrow = 2, ncol = 4, byrow = TRUE)

diagonal <- diag(c(10, 20, 30))
```

R 默认按列填充矩阵。`byrow = TRUE` 会改为按行填充。

#### 矩阵和数组索引

矩阵使用 `[行, 列]` 索引。省略一个维度表示选择该维度的全部元素。

```r
x <- matrix(1:6, nrow = 2, ncol = 3)

x[2, 2]
x[2, ]
x[, 3] <- NA
x[is.na(x)] <- 1

x[-1, ]
x[, -2]
```

单行或单列结果默认可能降维成向量。需要保留矩阵结构时使用 `drop = FALSE`。

```r
x[1, , drop = FALSE]
```

#### 矩阵运算

```r
X <- matrix(1:4, nrow = 2)

t(X)       # 转置
diag(X)    # 对角线元素
det(X)     # 行列式
X * X      # 对应元素相乘
X %*% X    # 矩阵乘法
```

### 数据框

数据框是 R 中最常见的表格结构。每一行通常对应一次观测，每一列对应一个变量。各列长度必须一致，但可以分别保存数值、字符、逻辑值或因子。

#### 创建与检查数据框

```r
measurements <- data.frame(
  id = 1:4,
  value = c(42, 7, 64, 9),
  group = c("A", "A", "B", "B")
)

str(measurements)
summary(measurements)
head(measurements)
```

实际分析中，数据框更多来自外部文件。导入方法放在后面的“数据导入、存储与分析准备”一节。

直接使用 `data$column` 或函数的 `data` 参数，比 `attach()` 更清楚，也不会因为搜索路径中存在同名对象而取错数据。

```r
cor(Puromycin$conc, Puromycin$rate)
pairs(Puromycin, panel = panel.smooth)
xtabs(~ state, data = Puromycin)
```

#### 选择行和列

数据框既支持矩阵式索引，也支持按列名访问。

```r
Puromycin[1, 1]
Puromycin[c(1, 3, 5), c("conc", "rate")]
Puromycin$conc

subset(Puromycin, state == "treated" & rate > 160)
```

`subset()` 适合交互式分析。编写需要严格控制求值环境的函数时，使用显式的方括号索引更稳妥。

```r
selected <- leadership[
  leadership$age >= 35 | leadership$age < 24,
  c("q1", "q2", "q3", "q4")
]
```

#### 创建和重命名变量

```r
Puromycin$inverse_conc <- 1 / Puromycin$conc

Puromycin <- transform(
  Puromycin,
  inverse_conc = 1 / conc,
  sqrt_conc = sqrt(conc)
)

names(Puromycin)[names(Puromycin) == "rate"] <- "reaction_rate"
```

`with()` 适合读取数据框中的列，但在其中赋值不会自动写回原数据框。`fix()` 会打开交互式编辑器，不利于复现，因此不作为常规的数据管理方法。

#### 合并与追加数据

有共同键的数据框可以用 `merge()` 联结。默认结果是内联结，只保留两边都匹配的键；`all.x = TRUE` 可得到左联结。

```r
total <- merge(dataframe_a, dataframe_b, by = "ID")
left_total <- merge(dataframe_a, dataframe_b, by = "ID", all.x = TRUE)
```

`cbind()` 按当前行顺序横向拼接对象，不会根据键对齐，因此应先确认行数和行顺序一致。`rbind()` 纵向追加数据框，要求列名和列类型能够对应。

```r
wide <- cbind(dataframe_a, extra_columns)
long <- rbind(dataframe_a, dataframe_b)
```

#### 行标识符

行名可以保存实例标识符，但通常更适合把标识符保留为普通列，这样导出、联结和检查重复值都更直接。

```r
patient_data <- data.frame(
  patient_id = patient_id,
  age = age,
  diabetes = diabetes,
  status = status
)

anyDuplicated(patient_data$patient_id)
```

### 列表

列表可以容纳不同类型、不同长度的对象，包括向量、矩阵、数据框、函数和其他列表。许多建模函数会返回列表，因为拟合结果通常同时包含系数、残差和诊断信息。

```r
results <- list(
  values = 1:6,
  matrix = matrix(1:4, nrow = 2)
)

results$values
results[[1]]  # 提取第一个元素本身
results[1]    # 返回只含第一个元素的子列表
```

### 时间序列

`ts()` 为向量或矩阵添加规则时间索引。它适合等间隔序列；日期不规则或需要时区信息的数据通常要用其他时间对象。

```r
ts(
  data = NA,
  start = 1,
  end = numeric(0),
  frequency = 1,
  deltat = 1,
  names = NULL
)
```

- `data` 是一元向量或多元矩阵。
- `start` 和 `end` 指定首尾观测的位置。
- `frequency` 表示每个时间单位中的观测数，例如季度数据取 4，月度数据取 12。
- `deltat` 表示相邻观测的时间间隔，与 `frequency` 二选一。
- `names` 用于设置多元序列的列名。

```r
annual <- ts(1:10, start = 1959)
monthly <- ts(1:47, frequency = 12, start = c(1959, 2))
quarterly <- ts(1:10, frequency = 4, start = c(1959, 2))

multivariate <- ts(
  matrix(rpois(36, lambda = 5), nrow = 12, ncol = 3),
  start = c(1961, 1),
  frequency = 12
)
```

## 日期与时间

`Date` 对象在底层以数值保存，表示相对 1970-01-01 的天数。字符日期需要用 `as.Date()` 按实际格式转换。

```r
date_strings <- c("01/05/1965", "08/16/1975")
dates <- as.Date(date_strings, format = "%m/%d/%Y")
```

常用格式符：

| 符号 | 含义 | 示例 |
| --- | --- | --- |
| `%d` | 两位日期 | `01` 至 `31` |
| `%a` | 星期缩写 | `Mon` |
| `%A` | 完整星期名 | `Monday` |
| `%m` | 两位月份 | `01` 至 `12` |
| `%b` | 月份缩写 | `Jan` |
| `%B` | 完整月份名 | `January` |
| `%y` | 两位年份 | `07` |
| `%Y` | 四位年份 | `2007` |

获取、格式化和比较日期：

```r
today <- Sys.Date()
now <- Sys.time()

format(today, format = "%B %d %Y")

dob <- as.Date("1956-10-12")
difftime(today, dob, units = "weeks")
```

`date()` 返回当前日期和时间的字符表示；需要继续计算时应使用 `Sys.Date()` 或 `Sys.time()`。

## 数据导入、存储与分析准备

可复现的分析应尽量使用项目目录和相对路径。`getwd()` 可以查看当前工作目录，`file.path()` 能以跨平台方式拼接路径。不要把个人电脑上的绝对路径写进共享脚本。

```r
getwd()
data_path <- file.path("data", "measurements.csv")
```

### 存储数据

文本、CSV 和 RData 是常见的存储格式。CSV 便于与其他软件交换，RData 可以在一个文件中保存多个 R 对象。

```r
d <- data.frame(
  observation = c(1, 2, 3),
  treatment = c("A", "B", "A"),
  weight = c(2.3, NA, 9)
)

write.table(
  d,
  file = file.path("data", "observations.txt"),
  row.names = FALSE,
  quote = FALSE
)

write.csv(
  d,
  file = file.path("data", "observations.csv"),
  row.names = FALSE
)

save(d, file = file.path("data", "objects.RData"))
```

### 读取数据

基础 R 可以直接读取文本和 CSV 文件，也可以加载随包提供的数据集及 RData 文件。

```r
houses <- read.table("houses.txt", header = TRUE)
scores <- read.csv("educ_scores.csv")

data("mtcars")
load(file.path("data", "objects.RData"))
```

`read.delim("clipboard")` 在部分桌面环境中可以读取剪贴板，但它依赖操作系统，不适合需要稳定复现的脚本。

读取 SPSS、SAS Transport 和 Stata 等格式时，可以使用 `foreign` 包。它是基础工作流中的传统方案，具体函数对不同格式的支持程度并不完全一致。

```r
library(foreign)

spss_data <- read.spss("educ_scores.sav", to.data.frame = TRUE)
sas_data <- read.xport("educ_scores.xpt")
stata_data <- read.dta("educ_scores.dta")
```

Excel 文件不能由 `foreign` 读取，可以先另存为 CSV，或使用 `readxl`。

```r
library(readxl)
so2_data <- read_excel(file.path("data", "SO2.xlsx"))
```

导入后先检查结构、行列数和关键变量，再进入数据管理。文件能够读入，并不意味着列类型、缺失值编码和标识符已经正确。

```r
str(so2_data)
dim(so2_data)
summary(so2_data)
```

### 缺失数据

真实数据中经常出现缺失值。处理前要先判断缺失发生在哪里、为什么缺失，以及分析方法怎样使用这些观测。`is.na()` 返回逐元素结果，`complete.cases()` 标记完整行。

```r
is.na(d)
colSums(is.na(d))
complete.cases(d)
```

#### 尝试恢复

如果原始问卷、日志或其他字段能够确定缺失值，可以回到数据来源恢复。例如，总分与分项之间存在确定关系时，缺失分项有时可以据此核对。恢复必须有明确依据，不能把猜测写回原始数据。

#### 完整案例分析

`na.omit()` 会删除包含缺失值的整行。许多统计函数也会通过 `na.action` 采用类似处理。这样做简单，却可能减少样本量；当缺失并非完全随机时，还可能引入偏差。

```r
complete_data <- na.omit(d)
```

是否删除行应由缺失机制、缺失比例和后续分析共同决定，而不是作为默认清洗步骤。

#### 多重插补

多重插补会生成多个合理的完整数据集，分别拟合模型，再合并估计值和不确定性。`mice` 包提供了常用实现。

```r
library(mice)

imp <- mice(data, m = 5, seed = 123)

fits <- with(imp, lm(y ~ x1 + x2))
pooled <- pool(fits)
summary(pooled)

completed_data <- complete(imp, action = 1)
```

其中：

- `data` 是含缺失值的数据框或矩阵。
- `imp` 保存多个插补数据集和插补过程信息。
- `with()` 在每个插补数据集上执行同一个分析表达式。
- `pool()` 按多重插补规则合并模型结果。
- `complete()` 可以提取某一个完整数据集。

只取一个插补数据集继续做推断，会丢失插补之间的不确定性。如果后续方法无法直接配合 `with()` 和 `pool()`，应明确记录这种限制及采用的合并策略。

### 探索性数据分析

探索性数据分析（EDA）发生在正式建模之前。目标是了解变量分布、异常值、缺失模式和变量之间的关系，并检查数据是否符合研究设计。

```r
str(data)
summary(data)
table(data$group, useNA = "ifany")

numeric_data <- data[vapply(data, is.numeric, logical(1))]
cor(numeric_data, use = "pairwise.complete.obs")
```

数值摘要不能替代图形。直方图、箱线图、散点图和分组图经常能暴露均值与相关系数看不到的结构。更完整的统计方法见 [EDA 与描述性统计](/blog/2024/09/05/r-classical-statistics-learning-notes/)，绘图方法见 [R 统计可视化](/blog/2024/03/15/r-visualization-learning-notes/) 和 [R 统计图形](/blog/2024/09/16/r-graph-learning-notes/)。

## R 编程

### 条件与循环

R 提供 `if`、`else`、`switch`、`for`、`while` 和 `repeat` 等控制结构。大括号可以避免多行分支产生歧义。

```r
if (condition_1) {
  statement_1
} else if (condition_2) {
  statement_2
} else {
  statement_3
}

for (i in 1:5) {
  print(i)
}

i <- 1
while (i <= 5) {
  print(i)
  i <- i + 1
}
```

`switch()` 根据字符或位置选择一个分支，适合处理少量固定选项。

```r
operation <- "mean"

switch(
  operation,
  mean = mean(1:5),
  sum = sum(1:5),
  stop("Unknown operation")
)
```

### 向量化

向量化是把逐元素操作交给向量函数或逻辑索引，而不是显式编写循环。它通常更简洁，一些函数也会调用经过优化的底层实现，但向量化本身不等于自动并行。

```r
y <- numeric(length(x))
y[x == b] <- 0
y[x != b] <- 1

y <- ifelse(x == b, 0, 1)
```

循环并不是错误。操作存在前后依赖、每次迭代返回复杂对象，或没有合适的向量函数时，清楚的循环往往更合适。

### 自定义函数

函数由参数列表和函数体组成。它可以返回任何 R 对象；没有显式调用 `return()` 时，最后一个表达式的值就是返回值。

```r
plot_file <- function(title_text, file_path) {
  data <- read.table(file_path, header = TRUE)
  plot(data[[1]], data[[2]], type = "l")
  title(title_text)

  invisible(data)
}
```

调用函数时可以按位置或名称传参。命名参数不必遵循定义顺序，但应使用完整名称，避免部分匹配带来的歧义。

```r
foo1(u, v, w)
foo1(arg3 = w, arg2 = v, arg1 = u)
```

参数可以有默认值，也可以通过 `...` 接收额外参数。

```r
foo2 <- function(arg1, arg2 = 5, arg3 = FALSE, ...) {
  list(arg1 = arg1, arg2 = arg2, arg3 = arg3, extra = list(...))
}
```

R 支持递归函数，但深层递归可能受调用栈限制。许多数据处理任务使用循环或向量函数更直接。

### 帮助系统

R 自带帮助系统。`help.start()` 会打开帮助首页，`help()` 和 `?` 用于查询函数或特殊语法。

```r
help.start()

help("lm")
?lm

help("bs", try.all.packages = TRUE)
help("bs", package = "splines")
```

部分包还提供介绍设计思路或完整工作流的 vignette 文档。

```r
vignette()
vignette(package = "survival")
```

### 普通函数、泛型与方法

普通函数直接执行固定实现，泛型函数则根据对象类别选择方法。S3 泛型常用 `UseMethod()` 分派。例如，`mean()` 会依据 `class(x)` 选择 `mean.default`、`mean.Date` 等实现，调用形式仍然是 `mean(x)`。

```r
mean
methods("mean")
```

典型输出会显示泛型定义及已经注册的方法：

```text
function (x, ...)
UseMethod("mean")

[1] mean.Date     mean.default  mean.difftime mean.POSIXct  mean.POSIXlt
```

### 查看函数源码

对于由 R 代码实现且当前可见的函数，直接输入函数名可以打印定义。

```r
lm
```

查看 S3 方法时，可以使用 `getS3method()`。`methods()` 输出中的星号表示方法不可见，此时 `getAnywhere()` 可以查找对象及其定义位置。

```r
getS3method("mean", "default")

methods("predict")
getAnywhere("predict.Arima")
```

如果函数的核心由 C 或 Fortran 实现，打印 R 函数通常只能看到调用编译代码的封装层。继续追踪时需要查看 R 或对应包的源代码。

## 常用内置函数

### 数学函数

```r
abs(x)             # 绝对值
acos(x)            # 反余弦
asin(x)            # 反正弦
atan(x)            # 反正切
atan2(y, x)        # 根据 x、y 坐标计算反正切
ceiling(x)         # 向上取整
floor(x)           # 向下取整
round(x, digits)   # 四舍五入
cos(x)             # 余弦
cosh(x)            # 双曲余弦
exp(x)             # 指数函数
log(x)             # 自然对数
log10(x)           # 以 10 为底的对数
logb(x, base)      # 指定底数的对数
sin(x)             # 正弦
sinh(x)            # 双曲正弦
sqrt(x)            # 平方根
tan(x)             # 正切
tanh(x)            # 双曲正切
```

### 统计函数

```r
mean(x)             # 均值
median(x)           # 中位数
sum(x)              # 总和
min(x)              # 最小值
max(x)              # 最大值
range(x)            # 返回最小值和最大值
diff(range(x))      # 极差
diff(x)             # 相邻元素之差
prod(x)             # 连乘
var(x)              # 样本方差
sd(x)               # 样本标准差
cor(x, y)           # 相关系数
cov(x, y)           # 协方差
quantile(x, probs)  # 分位数

t.test(x, y)
chisq.test(x)
cor.test(x, y)
```

### 概率分布函数

R 对常见分布使用统一的命名方式。分布缩写前加 `d`、`p`、`q` 或 `r`，分别得到概率密度或概率质量、累积分布、分位数和随机数函数。少数分布只实现其中一部分，例如多项分布提供 `dmultinom()` 和 `rmultinom()`，但没有对应的 `p*()` 和 `q*()` 函数。

| 分布 | R 缩写 | 常用参数 |
| --- | --- | --- |
| Beta | `beta` | `shape1`, `shape2` |
| 二项 | `binom` | `size`, `prob` |
| Cauchy | `cauchy` | `location`, `scale` |
| 指数 | `exp` | `rate` |
| 卡方 | `chisq` | `df`, `ncp` |
| F | `f` | `df1`, `df2`, `ncp` |
| Gamma | `gamma` | `shape`, `rate` 或 `scale` |
| 几何 | `geom` | `prob` |
| 超几何 | `hyper` | `m`, `n`, `k` |
| 对数正态 | `lnorm` | `meanlog`, `sdlog` |
| Logistic | `logis` | `location`, `scale` |
| 多项 | `multinom` | `size`, `prob` |
| 正态 | `norm` | `mean`, `sd` |
| 负二项 | `nbinom` | `size`, `prob` 或 `mu` |
| Poisson | `pois` | `lambda` |
| Student t | `t` | `df` |

以正态分布为例：

```r
dnorm(0)                    # x = 0 处的密度
pnorm(1.96)                 # P(X <= 1.96)
qnorm(0.975)                # 97.5% 分位数
rnorm(100, mean = 0, sd = 1)  # 生成 100 个随机数
```

连续分布的 `d*()` 返回密度，离散分布的 `d*()` 返回概率质量。`r*()` 的第一个参数通常是要生成的随机数个数。`p*()` 和 `q*()` 常用 `lower.tail` 控制计算左尾还是右尾，用 `log.p` 控制是否使用对数概率。

```r
pnorm(1.96, lower.tail = FALSE)
qnorm(log(0.025), log.p = TRUE)
```

### 抽样与组合

`sample()` 从向量中抽样。`size` 是抽取数量，`replace` 控制是否放回，`prob` 可以指定与元素一一对应的抽样权重。

```r
sample(x, size = 5)
sample(x, size = 5, replace = TRUE)
sample(x, size = 5, replace = TRUE, prob = weights)
```

排列组合计算可以使用 `factorial()` 和 `choose()`。

```r
factorial(5)
choose(52, 4)

# 从 52 个对象中依次取 4 个且不放回的排列数
prod(52:49)
```

### 字符处理函数

```r
nchar(x)                          # 字符数
substr(x, start, stop)            # 提取或替换子串
substring(x, first, last)         # 向量化的子串操作
paste(..., sep = " ", collapse = NULL)
paste0(..., collapse = NULL)
sprintf(format, ...)              # 格式化字符串
toupper(x)
tolower(x)
trimws(x)                         # 去除两端空白
strtrim(x, width)                 # 截断到指定显示宽度
cat(..., sep = "", file = "")    # 连接并输出
gsub(pattern, replacement, x)     # 替换全部匹配
sub(pattern, replacement, x)      # 替换首个匹配
chartr(old, new, x)               # 逐字符替换
strsplit(x, split, fixed = FALSE)
grep(pattern, x, value = FALSE)
regexpr(pattern, text)
gregexpr(pattern, text)
```

### 将函数应用到矩阵和数据框

`apply()` 沿矩阵或数组的指定维度调用函数。`MARGIN = 1` 表示行，`MARGIN = 2` 表示列。

```r
apply(x, MARGIN, FUN, ...)

apply(matrix_data, 1, mean)
apply(matrix_data, 2, sd)
```

对混合类型的数据框使用 `apply()` 时，数据框可能先被转换成矩阵，导致列类型被统一。按列处理数据框时，`lapply()` 或 `vapply()` 通常更合适。

```r
lapply(dataframe, class)
vapply(dataframe, is.numeric, logical(1))
```

`aggregate()` 按一个或多个分组变量汇总数值列。

```r
aggregate(x, by, FUN, ...)

aggregate(
  reaction_rate ~ state,
  data = Puromycin,
  FUN = mean
)
```

公式接口会按 `state` 分组，计算每组 `reaction_rate` 的均值。返回值仍是数据框，便于继续联结或绘图。
