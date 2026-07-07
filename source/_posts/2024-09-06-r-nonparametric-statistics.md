---
title: "R 非参数统计：符号检验、Wilcoxon 与多总体比较"
title_en: "R Nonparametric Statistics: Sign Tests, Wilcoxon, and Multi-Sample Comparison"
date: 2024-09-06 22:23:58 +0800
categories: ["Programming", "R"]
tags: ["R", "Nonparametric Statistics"]
author: Hyacehila
excerpt: "整理符号检验、Wilcoxon 检验、分布一致性检验、两总体比较和多总体非参数检验。"
excerpt_en: "Covers sign tests, Wilcoxon tests, goodness-of-fit checks, two-sample comparisons, and multi-sample tests."
mathjax: true
hidden: true
permalink: '/blog/2024/09/06/r-nonparametric-statistics-learning-notes/'
---

## 非参数的假设检验

这里我们介绍一些简单的 比较常用的非参数检验方法 他们的理论知识可能在数理统计中介绍了一些，更多的部分应该在非参数统计中详细介绍

### 单总体位置参数的检验

这里 我们研究的问题是分布的中心在哪里 虽然我们并不知道分布的形状如何 但是我们依旧可以研究他们的中心 正如在描述性统计中学习均值 中位数 众数等概念一样 但是现在加入了假设检验的成分

#### 中位数的符号检验

并没有现成的函数可以被调用

#### Wilcoxon符号秩检验

函数的形式如下

```r
wilcox.test(x, y=NULL, alternative=c("two.sided","less","greater"),
            mu=0, paired = FALSE, exact = NULL, correct =TRUE,
            conf.int = FALSE, conf.level = 0.95, ...)
```

exact表示是否算出准确的p值; correct表示大样本时是否做连续性修正 例子如下

```r
insure<-c(4632, 4728, 5052, 5064, 5484, 6972, 7696, 9048, 14760, 15013, 18730, 21240, 22836, 52788, 67200)
wilcox.test(insure,mu=6064,conf.int = TRUE)
```

### 分布一致性检验

这就是我们以前学习的$\chi^2$检验 函数形式如下

```r
chisq.test(x, y = NULL, correct =TRUE,p=rep(1/length(x),length(x)),
           rescale.p = FALSE, simulate.p.value = FALSE, B = 2000)
```

我们给出一个简单的例子 更加复杂的情况以后再学习

```r
v<-c(35,16,15,17,17,19,11,16,30,24)
chisq.test(v)
#检验几类型是否等概率出现了 当然我们自己设定p来检验某些特定的分布
```

### 两总体的比较与检验

我们有好几种值得在两总体的情况下研究的问题 逐个进行介绍

#### 独立性检验

也就是卡方独立性检验 使用的函数还是在分布一致性检验介绍的那个 例子如下

```r
compare<-matrix(c(60,32,3,11), nr = 2, dimnames = list(c("cancer", "normal"),
                                                       c("smoke", "Not smoke")))
chisq.test(compare, correct=TRUE)
```

#### 精确检验

 函数如下

```r
fisher.test(x, y=NULL, workspace=200000, hybrid=FALSE, control=list( ),
            or = 1, alternative = "two.sided", conf.int = TRUE,
            conf.level = 0.95, simulate.p.value = FALSE, B = 2000)
```

一个例子为

```r
compare<-matrix(c(60,32,3,11),nr = 2, dimnames = list(c("cancer", "normal"),
                                                      c("smoke", "Not smoke")))
fisher.test(compare, alternative = "greater")
```

#### Wilcoxon秩和检验法和Mann-Whitney U检验

函数前面介绍过了 直接给出一个简单的例子

```r
diabetes<-c(42,44,38,52,48,46,34,44,38)
normal<-c(34,43,35,33,34,26,30,31,31,27,28,27,30,37,32)
wilcox.test(diabetes,normal,exact = FALSE, correct=FALSE)
```

#### Mood检验

函数为

```r
mood.test(x, y, alternative =c("two.sided", "less", "greater"),...)
```

例子如下

```r
A<-c(321, 266, 256, 388, 330, 329, 303, 334, 299, 221, 365, 250, 258, 342, 343, 298, 238, 317, 354)
B<-c(488, 598, 507, 428, 807, 342, 512, 350, 672, 589, 665, 549, 451, 481, 514, 391, 366, 468)
diff<-median(B)-median(A)
A<-A+diff
mood.test(A,B)
```

### 多总体检验

我们在学习非参数统计后再来研究这个问题 缺少理论支撑的情况下没有必要进行系统性的学习 在需要使用的时候查阅资料就可以了

```r
shapiro.test(x)
#正态性的W检验
ks.test (x,y,alternative = c("two.sided", "less", "greater"),exact = NULL)
#KS检验 检验任意分布
```

## 组间差异的非参数检验
如果数据无法满足t检验或ANOVA的参数假设，可以转而使用非参数方法。我们在这里还是希望研究组之间的差异。

#### Wilcoxon秩和检验法和Mann-Whitney U检验

若两组数据独立，可以使用Wilcoxon秩和检验（更广为人知的名字是Mann-Whitney U检验）

调用格式为：
```r
## y是数值型变量 对应的x是二分类变量
## 可选参数data为一个包含了这些变量的数据框
wilcox.test(y ~ x, data)

## y1 y2 是两组结果变量
wilcox.test(y1, y2)
```

#### 超过两组的情况
如果各组独立，则Kruskal-Wallis检验将是一种实用的方法。如果各组不独立（如重复测量设计或随机区组设计），那么Friedman检验会更合适。

格式分别为
```r
kruskal.test(y ~ A, data)

friedman.test(y ~ A | B, data)
```

其中的`y`是数值型结果变量，`A`是一个分组变量，而`B`是一个用以认定匹配观测的区组变量（blocking variable）。也就是说，`B`用于对于单个个体重复实验的时候，标明我们的个体
