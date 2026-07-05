---
title: "R in Action：数据导入、数据管理与 EDA"
title_en: "R in Action: Data Import, Data Management, and EDA"
date: 2024-09-08 22:33:29 +0800
categories: ["Programming", "R"]
tags: ["Learning Notes", "R", "Data Management"]
author: Hyacehila
excerpt: "围绕数据导入、数据管理、缺失值处理和探索性数据分析整理实践要点。"
excerpt_en: "A overview inspired by R in Action, organizing practical material on data import, data management, missing data handling, and exploratory data analysis."
mathjax: false
hidden: true
permalink: '/blog/2024/09/08/r-in-action-learning-notes/'
---

整个R语言笔记体系都围绕着各种知识来进行，包括一些基本的操作知识，图形化的知识，各种比较新颖的Packages等等。

但是R始终是一门数据科学的专门语言，所以他的应用广泛在数据科学本身而非Python一样专注于各个不同的领域，因此我们在这里组织R in Action，希望抽离出知识以外的，一条专注于应用本身的知识线。

我们这里从原始的R笔记中频繁的引用，仅仅重写少量的知识，那些相对独立的知识板块仅引用，不重写。

R的学习曲线较为陡峭。因为它的功能非常丰富，所以文档和帮助文件也相当多。另外，由于许多功能都是由独立贡献者编写的可选模块提供的，这些文档可能比较零散而且很难找到。事实上，要掌握R的所有功能，可以说是一项挑战。 **根据我们所进行的研究来针对性的进行学习是非常必要的。**

当然，在这里我们并不准备介绍 [R Basic](/blog/2024/09/05/r-basic-learning-notes/) 中的基本内容，作为R语言的基本知识合集，每一个使用者都需要实现了解，并且不断地应用中强化。

## 聊聊数据本身
### 数据的导入与存储
作为数据科学的语言，导入与存储数据是不可能被忽视的内容，也是一切的开始。

这里我们参考 [数据存储和读取](/blog/2024/09/05/r-basic-learning-notes/) 中的内容，实际上我们只需要了解下面的部分
这里主要记住三件事：数据文件最好放在项目工作目录中；常见输出用 `write.table()`、`write.csv()` 和 `save()`；读取文本、CSV 或 RData 时优先使用清晰的路径和对应的 `read.table()`、`read.csv()`、`load()`。

### 数据管理
现实数据极少以直接可用的格式出现，这里我们将关注如何将数据转换或修改为更有助于分析的形式。 处理数据是数据科学家的主要工作。

这里我们对应的理论知识包含在 探索性数据分析 描述性统计与可视化 特征工程

非常自然的，数据管理中不可避免地需要创建新的变量。常用写法包括 `df$new_col <- ...`、`with()` 以及 `transform()`。
更加常见的，我们知道 探索性数据分析：数据转换（Data Transformation）是非常常见的，我们在数据科学中一般称之为重编码，我们一般使用各种逻辑运算符进行向量化编程实现，在AI技术发展的现在这种内容一般可以交给他完成。

变量的重命名，最基本的操作是使用`names()` 函数进行，不过我们建议直接使用GUI来实现，使用`fix()`函数就可以了
```r
fix(dataframe_name)
```

最基本的缺失值处理方法是删除缺失行，我们直接用以下代码实现
```r
na.omit(dataframe_name)
```

当数据来自不同的来源的时候，合并数据就非常的自然

要**横向合并两个数据框（数据集）**，请使用`merge()`函数。在多数情况下，两个数据框是通过一个或多个共有变量进行联结的（即一种内联结，inner join）

基本的实例为
```r
total <- merge(dataframeA, dataframeB, by="ID")
```

如果我们仅仅需要合并，不需要某种对齐，可以考虑`cbind()`函数，他需要行数是一样的
```r
total <- cbind(dataframeA, dataframeB)
```

要纵向合并两个数据框（数据集），需要使用`rbind()` 两个数据框必须拥有相同的变量，不过它们的顺序不必一定相同
```r
total <- rbind(dataframeA, dataframeB)
```

提取数据本质就是一种高级的索引，我们一般使用`subset()` 实现，给出例子就可以了
```r
## 选择所有age值大于等于35或age值小于24的行，保留了变量q1到q4
newdata <- subset(leadership, age >= 35 | age < 24,                   select=c(q1, q2, q3, q4))

## 选择所有25岁以上的男性，并保留了变量gender 到q4（gender、q4和其间所有列）
newdata <- subset(leadership, gender=="M" & age > 25,                   select=gender:q4)
```

### 缺失数据的处理
在真实世界中，缺失数据的现象是极其普遍的。但是大部分统计方法都假定处理的是完整矩阵、向量和数据框。因此处理缺失数据是展开具体的研究的第一步。这里的理论知识可以参考 探索性数据分析：缺失值处理

#### 尝试恢复数据
这里最混杂的一种缺失数据处理形式，有时候缺失的数据可以根据其他问题回答的情况进行反向的推导，例如
* 某些选项之间存在数学运算关系
* 可以使用确定的逻辑关系来大致推导选项之间的关联情况

恢复数据需要对数据本身进行思考，高度依赖于经验，因此这里不额外进行介绍了，在处理真实数据的时候考虑恢复就好了

#### 行删除
大部分流行的统计软件包都默认采用行删除法来处理缺失值。如果我们没有预先处理数据，那么各种统计函数很可能直接选择使用行删除来进行处理，甚至用户完全都没有关注到。

```r
na.omit(dataframe_name)
```
#### 多重插补
多重插补（MI）是一种基于重复模拟的处理缺失值的方法。

基本流程应该为
```r
#得到插补结果
library(mice)
imp <- mice(data, m)

#对插补得到的结果进行分析，这可以帮助我们同时对多个插补数据集进行分析
fit <- with(imp, analysis)
pooled <- pool(fit)
summary(pooled)

#选定一个插补数据集 x是1，2，3的编号
complete(imp, action=x)
```

其中

* data是一个包含缺失值的矩阵或数据框。
* imp是一个包含m个插补数据集的列表对象，同时还含有完成插补过程的信息。默认m为5。
* analysis是一个表达式对象，用来设定应用于m个插补数据集的统计分析方法。方法包括做线性回归模型的lm()函数、做广义线性模型的glm()函数、做广义可加模型的gam()，以及做负二项模型的nbrm()函数。
* fit是一个包含m个单独统计分析结果的列表对象。
* pooled是一个包含这m个统计分析平均结果的列表对象。

**在一般的数据分析问题中，我们使用的模型不是那么确定的，所以需要单独导出多个插补数据集来用于算法实践*
## EDA
在一切的建模开始之前，了解数据的内在模式是非常重要的，此处我们参考 [EDA与描述性统计](/blog/2024/09/05/r-classical-statistics-learning-notes/)

当然 图形化也是EDA的很好模式，因此此时引入足够的可视化手段非常的必要 参考[R Visualization](/blog/2024/03/15/r-visualization-learning-notes/) [R Graph](/blog/2024/09/16/r-graph-learning-notes/)
