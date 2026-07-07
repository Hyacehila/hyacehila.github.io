---
title: "R mlr3verse：任务、学习器、评估与调参"
title_en: "R mlr3verse: Tasks, Learners, Evaluation, and Tuning"
date: 2024-04-06 20:28:48 +0800
categories: ["Programming", "R"]
tags: ["R", "Machine Learning"]
author: Hyacehila
excerpt: "整理 mlr3 的任务、学习器、预测、评估、重抽样、基准实验、调参与分类评估等内容。"
excerpt_en: "Covers tasks, learners, prediction, evaluation, resampling, benchmarking, tuning, and classification metrics."
mathjax: true
hidden: true
permalink: '/blog/2024/04/06/r-mlr3verse-learning-notes/'
---

## 预备知识与概述
### mlr3
`mlr3`  包和更广泛的 `mlr3verse` 为 R 语言的回归、分类和其他机器学习任务提供了一个通用的、面向对象的和可扩展的框架。

在最基本的层面上，统一接口提供了训练、测试和评估许多机器学习算法的功能。还可以通过超参数优化、计算管道、模型解释等更进一步

`mlr3` 与 `scikit-learn`  `caret`  `tidymodels` 具有相似的总体目标

`mlr3` 旨在提供比其他 ML 框架更大的灵活性， `mlr3` 同时仍提供使用高级功能的简单方法。虽然 `tidymodels` 特别使执行简单的 ML 任务变得非常容易， `mlr3` 但更适合高级 ML

**`mlr3` 不能接受中文的列名 导入的数据框应当注意安全**
### 例子
现在我们用两个例子来介绍后面都会研究一些什么 我们会在后面一点点解释我们此时举出的两个例子

一个简单的决策树
```r
## 训练
library(mlr3)
task = tsk("penguins")
split = partition(task)
learner = lrn("classif.rpart")

learner$train(task, row_ids = split$train)
learner$model

## 预测
prediction = learner$predict(task, row_ids = split$test)
prediction

## 评估
prediction$score(msr("classif.acc"))
```

更复杂的例子
```r
library(mlr3verse)

tasks = tsks(c("breast_cancer", "sonar"))

glrn_rf_tuned = as_learner(ppl("robustify") %>>% auto_tuner(
    tnr("grid_search", resolution = 5),
    lrn("classif.ranger", num.trees = to_tune(200, 500)),
    rsmp("holdout")
))
glrn_rf_tuned$id = "RF"

glrn_stack = as_learner(ppl("robustify") %>>% ppl("stacking",
    lrns(c("classif.rpart", "classif.kknn")),
    lrn("classif.log_reg")
))
glrn_stack$id = "Stack"

learners = c(glrn_rf_tuned, glrn_stack)
bmr = benchmark(benchmark_grid(tasks, learners, rsmp("cv", folds = 3)))

bmr$aggregate(msr("classif.acc"))
```

### 两个必要的软件包
为了更加轻松的实现面向对象来处理ML 我们需要在R中增加软件包`R6`
同时 为了更加高效的处理大量数据 我们引入`data.frame`的改进`data.table`
#### R6
`R6` 是 R 面向对象编程的最新范例之一 它和其他面向对象的语言的思想是完全相同的 在语法规则上可能略有差异

`$new()` 是创建R6类对象的初始化方法 比如
```r
foo = Foo$new(bar = 1)
```
我们用`Foo`类创建了一个`foo`对象  并且设置了参数
在`mlr3 `中 我们实现了很多  sugar functionality 让创建需要的对象变得很容易

对于那些有可变状态(fields)的对象 我们也提供了方法来进行修改可变状态 使用`$`
```r
foo$bar = 2
```
我们访问了对象中的可变状态并进行了修改

除去可变状态(fields) 我们当然还有方法(methods) 方法允许用户检查对象的状态、检索信息或执行更改对象内部状态的操作
例如 对于学习器(learner) `$train()` 方法可以修改学习器的内部状态 这种方法通常会返回一个同类型的 `R6`对象 当然也有方法给出其他的东西 方法的返回结果是相当自由的  我们当然也可以 一次性输入多个方法 依次调用 如
```r
Foo$bar()$hello_world()
```
先对对象`Foo `进行了方法 `bar` 然后对他们返回的结果运行了方法 `hello_world`

最后 `R6`的对象是`environments` 直接赋值是引用 而不是创建新的对象 如下
```r
foo2 = foo
foo2 = foo$clone(deep = TRUE)
```
前者不创建新的对象 而是引用`foo` 对`foo2`的修改等同于直接对`foo`修改
想要复制对象 需要使用`clone`方法 见第二条

#### data.table
它是对`R`中 的`data.frame` 的拓展 它速度极快，并且可以很好地扩展到更大的数据 这也是我们为什么在`mlr3`中使用它

他在基础语法规则上和`data.frame`没有任何的区别 虽然属于扩展包但是完全基于R的语法规则构建

`data.table` 还使用引用语义，因此需要使用 `copy()` 克隆 `data.table`

### 实用应用程序
`mlr3` 包括一些重要的实用程序，这些实用程序对于简化mlr3生态系统中的代码至关重要
#### Sugar Functions 糖函数

`mlr3` 的大多数对象都可以通过称为辅助函数或糖函数的便利函数创建。它们为常见的代码习惯用语提供了快捷方式，减少了用户必须编写的代码量。例如 `lrn("regr.rpart")` ，返回学习器，而不必显式创建新的 R6 对象

sugar Functions 旨在涵盖大多数用户的大多数用例，只有在要构建自定义对象或扩展时才需要有关完整 `R6` 后端的知识

`mlr3 `中的对象根据约定标准化  `mlr_<type>_<key>`

#### Dictionaries 字典
`mlr3` 使用字典来存储 R6 类 字典中的值通常通过从相关字典中检索对象的 sugar 函数来访问
例如 `lrn("regr.rpart")` ，它是  `mlr_learners$get("regr.rpart")`包装器，因此是从 `mlr_learners` 加载决策树学习器的更简单方法

字典对相关对象的大量集合进行分组，以便可以轻松列出和检索它们  例如
`as.data.table(mlr_learners)  lrn() `都可以查看已经加载的包中的可用的学习器
#### mlr3viz
`mlr3viz` 包括mlr3的所有绘图功能 以及一些被使用的ggplot2中的功能
我们使用 `theme_minimal()` 来统一我们的审美，但与所有 `ggplot` 输出一样，用户可以完全自定义它

`mlr3viz` 扩展 `fortify` 和 `autoplot` 他们被用于用于常见 `mlr3` 输出，包括 `Prediction` 、 `Learner` 和 `BenchmarkResult` 对象

了解的 `mlr3viz` 最好方法是通过实验; 加载包，看看在 `mlr3` 对象上运行 `autoplot` 时会发生什么
绘图类型记录在相应的手册页面中，可以通过以下方式访问 `?autoplot.<class>` ，例如，可以通过运行 `?autoplot.TaskRegr` 来查找回归任务的不同类型的绘图

### 设计原则
下面是开发者对于`mlr3`的设计原则的介绍 了解他们可以帮助我们形成更好的代码习惯  并且理解`mlr3`的基本构成逻辑

* 面向对象编程：我们拥抱 `R6` 干净的、面向对象的设计、对象状态更改和引用语义
* 表格数据：拥抱 `data.table` 其一流的计算性能以及表格数据作为可以轻松进一步处理的结构
* 统一的表格输入和输出数据格式：大大简化了 API
* 防御性编程和类型安全：所有用户输入都使用 `checkmate`  进行检查
* 减少依赖关系：主要 `mlr` 的维护负担之一是跟上不断变化的学习器界面和它所依赖的许多软件包的行为。我们需要的 `mlr3` 软件包要少得多，这使得安装和维护更容易。我们仍然提供相同的功能，但它被拆分为更多单独具有更少依赖项的包
* 计算和表示的分离。 `mlr3` 生态系统的大多数软件包都专注于处理和转换数据、应用 ML 算法和计算结果，数据和结果的可视化在 `mlr3viz` 中提供

## 数据与基础建模 Data and Basic Modeling

在本章中，我们将介绍实现机器学习基本构建块的 `mlr3` 对象和相应的 `R6` 类。这些构建块包括数据（以及创建训练集和测试集的方法）、机器学习算法（及其训练和预测过程）、通过超参数配置机器学习算法以及评估预测质量的评估措施

这一章节将会是我们整个`mlr3verse`的核心 它介绍了构建一个模型所需要的全部内容 是对ML流程的完整介绍 本书的其余部分将建立在本章中看到的基本元素的基础上

我们的介绍将从最基础的回归开始 然后逐步扩展到分类问题 也是监督学习中最核心的两个问题都会被我们本章的内容覆盖
### 任务 Tasks
任务是包含定义机器学习问题的数据（通常是表格）和附加元数据(metadata)的对象；附加元数据包含监督机器学习问题的目标特征的名称
该信息在需要时自动提取，因此用户不必在每次训练模型时指定预测目标
#### 构建任务
`mlr3` 在 `mlr_tasks`  `Dictionary` 中包含一些预定义的机器学习任务
要从字典中获取任务，使用 `tsk()` 函数并将返回值赋给新变量
不带任何参数运行 `tsk()` 将列出字典中的所有任务，这也适用于其他sugar函数
```r
#查看字典中存储的预先定义的任务
mlr_tasks

#建立一个任务 我们是从字典中提取的
tsk_mtcars = tsk("mtcars")
#打印任务简报
tsk_mtcars

#查看字典中存储的预先定义的任务
tsk()
```

要创建自己的回归任务，您需要构造一个新的 `TaskRegr` 实例。最简单的方法是使用函数 `as_task_regr()`将 `data.frame` 类型的对象转换为回归任务，通过将其传递给 `target` 参数来指定目标特性 下面是一个例子 我们还是用`mtcars`数据集

```r
#引入数据 提取其中的一部分作为我们需要的数据集 然后简单展示
data("mtcars", package = "datasets")
mtcars_subset = subset(mtcars, select = c("mpg", "cyl", "disp"))
str(mtcars_subset)

## 用数据集建立一个回归任务 其中预测目标是 'mpg' id 是任务的简述 后面绘图会用它称呼我们的任务
tsk_mtcars = as_task_regr(mtcars_subset, target = "mpg", id = "cars")
```

`as_task_regr()` 可以接受各种类型的数据框 包括`data.frame data.table tibble` 兼容性非常的强

特别的 `as_task_regr()` 在很多情况下不接受UTF-8编码的名称 注意检验我们的数据框实用ASCII编码输入 如果不是我们可以更改名称再继续使用

对于任务 我们可以用`mlr3viz`绘制他的简要报告 它直接对mlr3对象进行了适配 全部自动就可以满足基本需求

```r
library(mlr3viz)
autoplot(tsk_mtcars, type = "pairs")
```

#### 检索数据
我们已经了解了如何创建任务来存储数据和元数据，现在我们将了解如何检索存储的数据

可以使用各种字段来检索关于任务的元数据 如
* 功能列和目标列的名称分别存储在 `$feature_names` 和 `$target_names` 里面
* 可以使用 `$nrow` 和 `$ncol` 检索维度
* 任务的列具有唯一的 `character` 值名称，行由唯一的自然数（称为行ID）标识。可以通过 `$row_ids` 字段访问
* 任务中包含的数据可以通过 `$data()` 访问，它返回一个 `data.table` 对象。这个方法有可选的 `rows` 和 `cols` 参数来指定要检索的数据子集
```r
## 回报维数 也就是行数（观测数）列数（特征数）
c(tsk_mtcars$nrow, tsk_mtcars$ncol)

#回报功能列和目标列名称
c(Features = tsk_mtcars$feature_names,
  Target = tsk_mtcars$target_names)

#回报行ID head用于限定返回前几个元素（默认6个）
head(tsk_mtcars$row_ids)
```
特别的  行ID与行号不同 我们过滤掉一些行的时候 行号发生变换 但是行ID不变 这是便于数据库操作的设计  例如
```r
task = as_task_regr(data.frame(x = runif(5), y = runif(5)),
  target = "y")
task$row_ids
## 返回 1 2 3 4 5
task$filter(c(4, 1, 3))
task$row_ids
## 返回 1 3 4
```

任务中包含的数据可以通过 `$data()` 访问，它返回一个 `data.table` 对象。这个方法有可选的 `rows` 和 `cols` 参数来指定要检索的数据子集
**默认行是用行ID检索的**
```r
## 返回全部数据子集
tsk_mtcars$data()

## 根据行ID以及列名称返回数据子集
tsk_mtcars$data(rows = c(1, 5, 10), cols = tsk_mtcars$feature_names)

## 这就起到了用行号检索数据子集的效果
tsk_mtcars$data(rows = task$row_ids[2])
```

#### 任务变更 Task mutators
它起到的作用是 在创建任务之后 对任务进行修改 我们能修改的只有关于数据和元数据的一些内容了 它和前面介绍的`$data()`的区别就是 **它直接修改了任务**

使用 `$select()` 可以通过特征（列）进行子集化，所需的特征名称作为字符向量传递，使用 `$filter()`可以通过将行ID作为数字向量传递来执行观察（行）子集化
**这些方法直接修改了任务**
由于R6的引用语意 如果想要同时保留原本的任务 需要 `$clone()` [R6 引用语义示例](/blog/2024/04/06/r-mlr3verse-learning-notes/)
```r
## 创建任务
tsk_mtcars_small = tsk("mtcars")
## 只选取一个列（特征） 它不可以移除target
tsk_mtcars_small$select("cyl")
## 选取下面的行
tsk_mtcars_small$filter(2:3)
```

要向任务添加额外的行和列，您可以分别使用 `$rbind()` 和 `$cbind()`
```r
tsk_mtcars_small$cbind(
  data.frame(disp = c(150, 160))
)

tsk_mtcars_small$rbind(
  data.frame(mpg = 23, cyl = 5, disp = 170)
)
```

### 学习器 Learners
#### 学习器介绍
类 `Learner` 的对象为R中许多流行的机器学习算法提供了一个统一的接口。 `mlr_learners` 字典包含 `mlr3`中可用的所有学习器。我们将在后面讨论可用的学习器;现在，我们将使用回归树学习器作为例子来讨论 `Learner`接口。与任务一样，可以使用单个sugar函数从字典中访问学习者，在本例中为 `lrn()`
```r
## 查看可用的学习器
mlr_learners
lrn()

#从字典中建立一个学习器 作学习器的简要报告（因为没存储）
lrn("regr.rpart")
```
学习器是算法的核心 和任务不同的是 学习器与训练数据无关 我们在大多数情况下 都只会调用已经存在的学习器 只有在自行从底层设计新算法的时候才需要自定义新学习器 所以从字典中调用学习器已经足够我们使用了

所有 `Learner` 对象都包含以下元数据，可以在学习器简要报告的输出中看到
* `$feature_types` ：学习器可以处理的特征类型
* `$packages` ：使用学习者需要安装的软件包
* `$properties` ：学习者的属性  例如，“missings”属性意味着模型可以处理丢失的数据，而“importance”意味着它可以计算每个特征的相对重要性
* `$predict_types` ：模型可以进行的预测类型
* `$param_set` ：可用超参数集

一个完整的运行机器学习实验，学习者需要经历两个阶段：
* 训练：训练 `Task` 被传递到学习者的 `$train()` 函数，该函数训练并存储模型，即特征与目标的学习关系
* 预测：新的数据，可能是原始数据集的不同划分，被传递给训练好的学习器的 `$predict()` 方法来预测目标值

**训练和预测的方法都属于学习器 而和任务无关 任务的设计和学习器的学习是两个完全独立的过程**
#### 训练
通过使用 `$train()` 方法将任务传递给学习器来训练模型  训练后的模型存储在字段`$model`中
```r
## load mtcars task
tsk_mtcars = tsk("mtcars")
## load a regression tree
lrn_rpart = lrn("regr.rpart")
## 训练学习器 任务和学习器均已经给出
lrn_rpart$train(tsk_mtcars)
## 查看训练出的模型 具体怎么理解可以参考模型的help 我们的帮助是可以对对象使用的
lrn_rpart$model
```

在很多时候 我们希望不使用全部原始数据来训练模型 这里我们介绍一种最简单的划分方法  对应我们在机器学习理论中介绍的留出法 机器学习导论与监督学习：留出法

`partition()` 函数创建索引集，将给定任务随机分为两个不相交的集合：训练集（默认情况下占总数据的67%）和测试集（剩余的数据）
```r
## 划分数据 返回的是两个集合
splits = partition(tsk_mtcars)
splits

## 借助前面的划分来实现 row_ids让我们可以选择一部分行号进行训练
lrn_rpart$train(tsk_mtcars, row_ids = splits$train)
```

#### 预测
从训练好的模型进行预测就像将数据作为 `Task` 传递给训练好的 `Learner`的 `$predict()` 方法一样简单
##### 基础的预测
由于预测也需要数据 而且按照机器学习的习惯 训练和测试的数据会被一起整合到`tasks`中 所以我们还是传入了`tasks` 指定预测的行号
```r
prediction = lrn_rpart$predict(tsk_mtcars, row_ids = splits$test)
```

`$predict()` 方法返回一个从 `Prediction` 继承的对象 根据我们的任务不同对象会有差异 一般有三列；`row_ids` 列对应于预测观测的行ID。 `truth`列包含对象从任务中提取的真实数据（如果可用）`response` 列包含模型预测的值

使用 `as.data.table()` 等函数可以很容易地将 `Prediction` 对象分别转换为 `data.table` 或 `data.frame` 对象

##### 预测对象的特殊处理方法
作为`mlr3verse`中的对象 它也有着自己特殊的一些操作方法
```r
## 直接访问 如果访问的不是很多没必要调用数据框相关函数 语法也非常的自然
prediction$response[1:2]

#`mlr3viz` 为 所有继承`Prediction`类的对象提供了一个 `autoplot()` 方法
library(mlr3viz)
prediction = lrn_rpart$predict(tsk_mtcars, splits$test)
autoplot(prediction)
```

##### 预测新数据
虽然机器学习习惯性把所有的数据一并传入tasks中 不过我们有时候确实有预测一些新数据的需求 此时也没有必要重建任务 mlr3 考虑到了这种情况 使用 `$predict_newdata()` 就可以了
```r
mtcars_new = data.table(cyl = c(5, 6), disp = c(100, 120),
  hp = c(100, 150), drat = c(4, 3.9), wt = c(3.8, 4.1),
  qsec = c(18, 19.5), vs = c(1, 0), am = c(1, 1),
  gear = c(6, 4), carb = c(3, 5))
prediction = lrn_rpart$predict_newdata(mtcars_new)
prediction
```
此时` truth `列全部是空的` row_ids `沿用新的数据框
**注意 预测新数据的时候务必保证列名称和`tasks`中是一致的**

##### 更改预测类型
虽然预测单个数值量是回归中最常见的预测类型，但它不是唯一的预测类型。一些回归模型也可以在预测的同时给出预测标准误差 正如我们在传统的线性回归中进行的那样

为了预测这一点，在训练之前，必须将 `LearnerRegr` 的 `$predict_type` 字段从“response”（默认值）更改为 `"se"`

我们上面使用的 `"rpart"` 学习器不支持预测标准误差，因此在下面的示例中，我们将使用线性回归模型`lrn("regr.lm")`
```r
## 导入需要的包并且建立新的学习器
library(mlr3learners)
lrn_lm = lrn("regr.lm", predict_type = "se")
## 训练与预测
lrn_lm$train(tsk_mtcars, splits$train)
lrn_lm$predict(tsk_mtcars, splits$test)
```

#### 超参数 Hyperparameters
`Learners` 封装了一个机器学习算法及其超参数，这些超参数会影响算法的运行方式，并且可以由用户设置。

超参数可能会影响模型的训练方式或预测方式，决定如何设置超参数可能需要专业知识

超参数可以自动优化 我们在后面会介绍如何自动优化超参数的问题，但在本章中，我们将专注于如何手动设置它们 这是我们后面设置自动优化的基础 而且在实际的机器学习工作中 超参数经常是手工设定的

##### 参数和参数集
我们前面介绍过了 学习器的超参数应该使用`$param_set`访问 [学习器超参数说明](/blog/2024/04/06/r-mlr3verse-learning-notes/)  如下
```r
lrn_rpart$param_set
```
输出是一个 `ParamSet` 对象，由 `paradox` 包提供。这些对象提供有关超参数的信息，包括它们的名称（ `id` ）、数据类型（ `class` ）、超参数值的技术有效范围（ `lower` 、 `upper` ）、如果数据类型是分类的则可能的级别数（ `nlevels` ）、来自底层包的默认值（ `default` ）以及最后的设置值（ `value` ）

`class` 继承的是在 `paradox` 中定义的类，这些类确定参数的类和它可以采用的可能值 常见的超参数类如下表

| Hyperparameter Class                                              | Hyperparameter Type |
| ----------------------------------------------------------------- | ------------------- |
| [`ParamDbl`](https://paradox.mlr-org.com/reference/ParamDbl.html) | 实值（数值）              |
| [`ParamInt`](https://paradox.mlr-org.com/reference/ParamInt.html) | 整数                  |
| [`ParamFct`](https://paradox.mlr-org.com/reference/ParamFct.html) | 因子                  |
| [`ParamLgl`](https://paradox.mlr-org.com/reference/ParamLgl.html) | 布尔类型 （T or F）       |
| [`ParamUty`](https://paradox.mlr-org.com/reference/ParamUty.html) | 无类型                 |

在大多数情况下 超参数都会被正确的初始化为应该有的默认情况 但是在部分情况下 它可能被错误的初始化 也就是在建立学习器的时候超参数并没有在它应该在的情况 此时一般会给出提示 这种`bug`是开发的时候在尽量避免但是未能完全避免的

##### 获取和设置超参数值
现在我们已经了解了超参数集是如何存储的，我们可以考虑获取和设置它们。回到我们的决策树，假设我们有兴趣生长一棵深度为 `1` 的树，也称为“决策树桩”，其中数据只被分成两个终端节点

有几种不同的方法可以改变这个超参数。最简单的方法是在构造学习器的过程中，将超参数名称和新值传递给 `lrn()` 如
```r
## 建立学习器的时候设置超参数
lrn_rpart = lrn("regr.rpart", maxdepth = 1)

## 返回那些非默认超参数的列表 本质上就是在超参数集上多了一层访问
lrn_rpart$param_set$values
```
**直接在构造学习器的时候手动设置超参数是最为实用的方法**

就在刚刚 我们介绍了另一种访问超参数集合的方法 我们也可以从 `$value`上着手修改超参数
```r
lrn_rpart$param_set$values$maxdepth = 2
lrn_rpart$param_set$values
```

刚才的方法一次只能修改一个超参数值 当然我们有更多的方法
```r
lrn_rpart$param_set$set_values(xval = 2, cp = 0.5)
lrn_rpart$param_set$values
```

`lrn_rpart$param_set$values` 返回一个 `list` 但是不要使用传入新的 `list`的方法来修改超参数 他会导致有的超参数被抹消 因为我们总是在建立 `list`的时候只包含自己想修改的超参数

**所有的修改超参数的方法 都会进行相关越界的检查 保证类型合规**

##### 超参数依赖
更复杂的超参数空间可能包含依赖关系，这发生在设置一个超参数是以另一个超参数的值为条件时;
一个这样的示例是支持向量机（ `lrn("regr.svm")` ）。字段 `$deps` 返回一个 `data.table` ，它列出了 `Learner` 中的超参数依赖关系
```r
lrn("regr.svm")$param_set$deps
```
其中 `id` 列指出来谁依赖其他超参数 `on` 列告诉我们谁被依赖 `cond` 列告诉我们条件是什么
```r
#访问cond列内容
lrn("regr.svm")$param_set$deps[[1, "cond"]]
lrn("regr.svm")$param_set$deps[[3, "cond"]]
```
`CondAnyOf` 意味着`on`是集合中的量中的某一个
`CondEqual` 意味着`on` 等于某一个值
它就意味着我们的条件

如果在不满足相关超参数的条件时设置了相关超参数，则 `Learner` 将出错

#### 基准学习器
在我们继续学习器评估之前，我们将重点介绍一类重要的学习器。这些是非常简单或“弱”的学习者，称为基准；

对于回归，我们已经实现了基准 `lrn("regr.featureless")` ，它总是预测新值是训练数据中目标的平均值（或中位数，如果 `robust` 超参数设置为 `TRUE` ）

基准学习器将会是我们用来测试新的模型的好方法 如果一个模型的效果比基准学习器还要差 那它将会是一个很差的模型  下面是一些简单的代码示例 可以作为对学习器一节的回顾
```r
df = as_task_regr(data.frame(x = runif(1000), y = rnorm(1000, 2, 1)),
  target = "y")
lrn("regr.featureless")$train(df, 1:995)$predict(df, 996:1000)
```

### 评价 Evaluation
也许应用机器学习工作流程中最重要的一步是评估模型性能。如果没有这一点，我们将无法知道我们的训练模型是否能做出非常准确的预测，是否比随机猜测更糟糕，或者介于两者之间; 下面是我们的代码示例 也可以视为是对前面的一些代码内容的复习

```r
lrn_rpart = lrn("regr.rpart")
tsk_mtcars = tsk("mtcars")
splits = partition(tsk_mtcars)
lrn_rpart$train(tsk_mtcars, splits$train)
prediction = lrn_rpart$predict(tsk_mtcars, splits$test)
```

#### 评价器
预测的质量是使用将它们与监督学习任务的真实数据进行比较的措施来评估的
与 `Tasks` 和 `Learners`类似， `mlr3` 中的可用度量存储在名为 `mlr_measures` 的字典中，并且可以使用 `msr()` 访问 如
```r
## 访问评价器的字典
mlr_measures
msr()
```

由于评价一个模型的思路往往是固定的 因此我们的评价器也很少需要建立新的 继承字典中有的评价器是最经常的

`mlr3` 中实现的所有度量主要由三个组件定义
* 度量的函数
* 更低或更高的值是否被认为是“好”
* 度量可以采用的可能值的范围
除此以外 一个评价器还有一些元数据 如
* 度量是否具有任何特殊属性
* 度量可以评估的预测类型
* 度量是否具有任何“控制参数”

直接查看一个评价器就可以得到全部的这些需要的数据 如下
```r
measure = msr("regr.mae")
measure
```

#### 预测打分
为了计算模型性能，我们只需调用 `Prediction` 对象的 `$score()` 方法，并将我们想要计算的度量作为单个参数传递 事实上 `Prediction` 存储了我们用来评价一个模型所需要的全部数据 包括真实值和预测值

```r
prediction$score(measure)
```

所有任务类型都有默认的评价器 比如 回归模型使用均方误差MSE作为默认评价器 如果我们不传入`$score()` 参数 那么就使用默认评价器

**评价器对只测试数据进行评价，我们关注泛化性能而不是拟合效果，后面采用其他的测试方式也是如此**

通过将多个评级器传递给 `$score()` ，可以同时计算多个评价

```r
## 同时把多个评价器给了这个变量 然后一起传入
measures = msrs(c("regr.mse", "regr.mae"))
prediction$score(measures)
```

#### 另一些评价
`mlr3` 还提供了不量化模型预测质量的度量，而是提供关于模型的“元”信息。其中包括：
* `msr("time_train")` -训练模型所需的时间
* `msr("time_predict")` -模型进行预测所用的时间
* `msr("time_both")` -训练模型然后进行预测所花费的总时间
* `msr("selected_features")` -模型选择的特征数量，仅当模型具有“selected_features”属性时才能使用

一个简单的例子为
```r
measures = msrs(c("time_train", "time_predict", "time_both"))
prediction$score(measures, learner = lrn_rpart)
```
我们把学习器一起传入了`$score()` 这是评价器所拥有的特殊属性

对于模型选择特征的数量 有
```r
## 查看评估器的元数据
msr_sf = msr("selected_features")
msr_sf
```
特别的 这个评估器的元数据中有这两条条
* Parameters: normalize=FALSE
* Properties: requires_task, requires_learner, requires_model
也就是 这个评估器有参数可以设置 关于评估器参数的设置可以直接参考 [超参数 Hyperparameters](/blog/2024/04/06/r-mlr3verse-learning-notes/) 所有的方法都是一样的
* `normalize` 超参数指定返回的选定特征数是否应按特征总数进行规范化
* `Properties` 告诉了我们这个评估器需要 任务 学习器 要一起传入

展示使用这个评估器的代码为
```r
## 设置了评估器参数
msr_sf$param_set$values$normalize = TRUE
## 调用了评估器 它需要任务和学习器
prediction$score(msr_sf, task = tsk_mtcars, learner = lrn_rpart)
```

### 回归实验 Exercise
在研究 `mlr3` 的使用如何扩展到分类之前，我们将暂停一下，在一个简短的实验中将上述所有内容放在一起，以评估我们预测的质量

独立的审阅下面的代码 理解用法 并且学会扩展

```r
library(mlr3)
set.seed(349)
## 任务构建和划分 并没有自建任务
tsk_mtcars = tsk("mtcars")
splits = partition(tsk_mtcars)
## 加载学习器 这是基准学习器
lrn_featureless = lrn("regr.featureless")
## 加载学习器 这是决策树
lrn_rpart = lrn("regr.rpart", cp = 0.2, maxdepth = 5)
## 加载评估器 两种评估方法
measures = msrs(c("regr.mse", "regr.mae"))
## 对两个学习器训练 使用训练数据
lrn_featureless$train(tsk_mtcars, splits$train)
lrn_rpart$train(tsk_mtcars, splits$train)
## 对两个学习器预测 使用预测数据 同时对预测的结果进行评价
lrn_featureless$predict(tsk_mtcars, splits$test)$score(measures)
lrn_rpart$predict(tsk_mtcars, splits$test)$score(measures)
```

会注意到我们的学习器和度量都有 `"regr."` 前缀，这是一种方便的方式，提醒我们正在处理回归任务，因此必须使用为回归构建的学习器和度量

在下一节中，我们将用 `mlr3` 来考虑分类任务 仅仅是一点微小的改变 很好理解的

### 分类 Classif
分类问题是一个模型预测一个离散的、分类的目标，而不是一个连续的、数字的量。例如，从企鹅的物理特征预测其物种将是一个分类问题，因为存在一组定义的物种

`mlr3` 确保所有任务的接口尽可能相似（如果不是完全相同的话） 因此只关注使分类成为独特机器学习问题的差异

我们将首先通过执行一个与[回归实验](/blog/2024/04/06/r-mlr3verse-learning-notes/)中非常相似的实验来演示回归和分类之间的相似性，使用的代码现在已经熟悉了

然后，我们将讨论任务、学习者和预测的差异，然后再讨论阈值，这是一种特定于分类的方法

#### 分类试验
代码如下 结合我们前面已有的了解来理解这些代码
```r
library(mlr3)
set.seed(349)
## 构建任务 这里我们的任务还是直接用已有的 建立一个新的预测任务的代码与回归存在的差异后面来解释 划分了集合
tsk_penguins = tsk("penguins")
splits = partition(tsk_penguins)
## 加载一个分类学习器 它是基准学习器
lrn_featureless = lrn("classif.featureless")
## 加载一个分类学习器 还是决策树 但是是分类专用
lrn_rpart = lrn("classif.rpart", cp = 0.2, maxdepth = 5)
## 加载分类评价器
measure = msr("classif.acc")
## 训练
lrn_featureless$train(tsk_penguins, splits$train)
lrn_rpart$train(tsk_penguins, splits$train)
## 预测并评价
lrn_featureless$predict(tsk_penguins, splits$test)$score(measure)
lrn_rpart$predict(tsk_penguins, splits$test)$score(measure)
```

#### 分类任务
分类任务是从 `TaskClassif` 继承的对象，除了目标变量是因子类型，非常类似于回归任务

可以通过过滤 `mlr_tasks` 字典查看 `mlr3` 中预定义的分类任务
```r
as.data.table(mlr_tasks)[task_type == "classif"]
```

可以使用 `as_task_classif` 创建自己的分类任务
```r
as_task_classif(palmerpenguins::penguins, target = "species")
```

`mlr3` 中支持两种类型的分类任务：二元分类，其中结果可以是两个类别中的一个，以及多类分类，其中结果可以是三个或更多类别中的一个

我们可以在任务的简要报告中看到它的各种相关属性 并且用最自然的习惯来访问

这些任务之间的一个重要区别是，二进制分类任务有一个名为 `$positive` 的额外字段，它定义了“正”类。在二元分类中，由于只有两种可能的类别类型，按照惯例，其中一种被称为“正”类，另一种被称为“负”类
```r
## 加载数据
data(Sonar, package = "mlbench")
## 建立tasks
tsk_classif = as_task_classif(Sonar, target = "Class", positive = "R")
## 查看正类
tsk_classif$positive
## 修改正类
tsk_classif$positive = "M"
```

虽然类别的选择是任意的，但它们对于确保模型和性能指标的结果按预期解释至关重要-这在我们讨论阈值和ROC指标时得到了证明

最后，可以使用 `autoplot.TaskClassif` 进行绘图
```r
library(ggplot2)
autoplot(tsk("penguins"), type = "duo") +
  theme(strip.text.y = element_text(angle = -45, size = 8))
```
#### 分类学习器
继承自 `LearnerClassif` 的分类学习器与回归学习器具有几乎相同的接口；

但是 分类中可能的预测是并不唯一的 它可能是`"response"` 也就是预测观察的类 也可能是`"prob"` 预测属于每个类的观察的概率向量（或者称为后验概率）
`response` 默认情况下是具有最高预测概率的类
#### 分类评价器
分类度量（类别 `MeasureClassif` ）的接口与回归度量相同

但是 我们发现： 分类的任务类型被划分为了二分类和多分类，分类的预测类型被划分为了概率预测和类预测 他们都会涉及到评价器的差距 我们需要在前面查阅所有评价器的基础上再进行选择 代码如下
```r
as.data.table(msr())[
    task_type == "classif" & predict_type == "prob" ]
```
第一个部分限制了评价器的任务类型是`classif`还是`regr`
第二个部分限制了预测类型是`prob`还是`response`

代码的例子为 整体的接口形式是一样的
```r
measures = msrs(c("classif.mbrier", "classif.logloss", "classif.acc"))
prediction$score(measures)
```
#### 分类中的预测
`PredictionClassif` 对象与它们的回归模拟有两个重要的区别。
* 首先是添加的字段 `$confusion`
* 其次是添加的方法 `$set_threshold()`

他们都不会被
```r
prediction
```
代码直接访问到，他们都属于分类问题预测带来的特殊量
##### 混淆矩阵 Confusion matrix
混淆矩阵是一种流行的方式，通过查看模型是否擅长（错误）将观察结果分类在特定类别中，以更详细的方式显示分类（响应）预测的质量

对于二进制和多类分类，混淆矩阵存储在 `PredictionClassif` 对象的 `$confusion` 字段中 访问代码为
```r
prediction$confusion
```
关于其理论解释可以查看 机器学习导论与监督学习：查准率、查全率与F1

特别的 我们可以直接对混淆矩阵带来的图形进行可视化
```r
autoplot(prediction)
```

##### Thresholding 阈值
和回归问题相比 分类带来的另一个问题是阈值问题；

默认的 `response` 预测类型是预测概率最高的类，如果最大概率不是唯一的，即多个类别被预测为具有最高概率，然后从这些类别中随机选择；

在二进制分类中，这意味着如果预测的类别大于50%，则将选择阳性类别，否则将选择阴性类别

**这个50%的值被称为阈值，如果存在类不平衡（当一个类在数据集中过度或不足时），或者如果存在与类相关联的不同成本，或者仅仅是如果偏好“过度”预测一个类，则更改此阈值可能很有用**

在二分类问题中 设置阈值是非常容易的
```r
prediction$set_threshold(0.7)
```
此时只有`prob>0.7`的时候  才会被预测为正类

在多类分类中，阈值处理的工作原理是首先为每个 `n` 类分配一个阈值，将每个类的预测概率除以这些阈值以返回 `n` 比率，然后选择具有最高比率的类 此时阈值依旧体现了我们选择的偏好 阈值越大我们越偏离它

在 `mlr3` 中，这是通过向 `$set_threshold()` 传递命名列表来实现
```r
library(ggplot2)
library(patchwork)

tsk_zoo = tsk("zoo")
splits = partition(tsk_zoo)
lrn_rpart = lrn("classif.rpart", predict_type = "prob")
lrn_rpart$train(tsk_zoo, splits$train)
prediction = lrn_rpart$predict(tsk_zoo, splits$test)
before = autoplot(prediction) + ggtitle("Default thresholds")
new_thresh = proportions(table(tsk_zoo$truth(splits$train)))
new_thresh
prediction$set_threshold(new_thresh)
after = autoplot(prediction) + ggtitle("Inverse weighting thresholds")
before + after + plot_layout(guides = "collect")
```
这种操作一般被称为反向加权  我们在以后的学习中还会遇到类似的手法

### 任务列角色
现在我们已经介绍了回归和分类，我们将简要地返回到任务；列角色是最重要的元数据，学习者和其他对象可以使用这些元数据与任务进行交互；有七个列角色：
* `"feature"` ：用于预测的功能
* `"target"` ：要预测的目标变量
* `"name"` ：行名称/观察标签，例如，对于 `mtcars` ，这是 `"model"` 列
* `"order"` ：用于对 `$data()` 返回的数据进行排序的变量; 使用 `order()`
* `"group"` ：用于在重新分配期间将观察结果保持在一起的变量
* `"stratum"` ：在重新采样期间分层的变量
* `"weight"` ：观察权重。只有一个数值列可以具有此角色

`feature` 和 `target` 我们已经在前面介绍过了[任务 Tasks](/blog/2024/04/06/r-mlr3verse-learning-notes/)
`stratum` 和 `group` 我们会在后面的章节中再进行介绍
我们不会详细介绍 `name` ，它主要用于绘图，并且几乎总是底层数据的 `rownames()`

**使用 `$set_col_roles()` 更新列角色** 列角色被更新以后 就不会被当作其他列角色使用了 也就是每一个列只有一个列角色
#### order
对于`"order"` 角色 数据根据该列进行排序
当我们运行 `$data()` 时，它不再被用作一个特征，而是用于根据其值对观测进行排序。此元数据不会传递给学习者
```r
df = data.frame(mtcars[1:2, ], idx = 2:1)
tsk_mtcars_order = as_task_regr(df, target = "mpg")
## 初始排序
tsk_mtcars_order$data(ordered = TRUE)

## 根据列 idx 进行排序
tsk_mtcars_order$set_col_roles("idx", roles = "order")
tsk_mtcars_order$data(ordered = TRUE)
```

#### weight
`weights` 列角色用于对数据点进行不同的加权；在具有严重类别不平衡的分类任务中，其中更重地加权少数类别可能会提高模型对该类别的预测性能

代码示例为
```r
cancer_unweighted = tsk("breast_cancer")
summary(cancer_unweighted$data()$class)

## add column where weight is 2 if class "malignant", and 1 otherwise
df = cancer_unweighted$data()
df$weights = ifelse(df$class == "malignant", 2, 1)

## create new task and role
cancer_weighted = as_task_classif(df, target = "class")
cancer_weighted$set_col_roles("weights", roles = "weight")

## compare weighted and unweighted predictions
split = partition(cancer_unweighted)
lrn_rf = lrn("classif.ranger")
lrn_rf$train(cancer_unweighted, split$train)$
  predict(cancer_unweighted, split$test)$score()

lrn_rf$train(cancer_weighted, split$train)$
  predict(cancer_weighted, split$test)$score()
```
在本例中，加权提高了模型的整体性能;并非所有模型都可以处理任务中的权重，因此请检查学习者的属性，以确保按预期使用此列角色

### 支持的学习算法
`mlr3`支持许多学习算法；这些主要由 `mlr3` 、 `mlr3learners` 和 `mlr3extralearners` 包提供；当然，那些比较新的软件包一般都会包含一些比较新的算法

#### mlr3
`mlr3` 中包含的学习器列表故意很小，借此减少对其他包的依赖； 有
* Featureless learners (`"regr.featureless"`/`"classif.featureless"`) 作为基准学习器使用
* Debug learners (`"regr.debug"`/`"classif.debug"`) 用于代码调试
* Classification and regression trees (also known as CART: `"regr.rpart"`/`"classif.rpart"`)  分类和回归树 也称作CRAT

#### mlr3learners
`mlr3learners` 包包含了mlr团队选择的一系列算法（和选择的实现） 它应该是我们整个机器学习算法流程的基础
* 线性（ `"regr.lm"` ）和逻辑（ `"classif.log_reg"` ）回归
* 惩罚广义线性模型，其中惩罚要么作为超参数（ `"regr.glmnet"` / `"classif.glmnet"` ），要么自动优化（ `"regr.cv_glmnet"` / `"classif.cv_glmnet"` ）
* 加权的$k$近邻（ `"regr.kknn"` / `"classif.kknn"` ）
* Kriging / Gaussian process regression（ `"regr.km"` ）
* 线性（ `"classif.lda"` ）和二次（ `"classif.qda"` ）判别分析
* 朴素贝叶斯分类（ `"classif.naive_bayes"` ）
* 支持向量机（ `"regr.svm"` / `"classif.svm"` ）
* 梯度增强（ `"regr.xgboost"` / `"classif.xgboost"` ）
* 回归和分类的随机森林（ `"regr.ranger"` / `"classif.ranger"` ）

#### 查看学习算法
一般我们会先把所有可用的学习器提取 转换成一个数据框 同时简要审阅其格式
```r
learners_dt = as.data.table(mlr_learners)
learners_dt
```
生成的 `data.table` 包含大量元数据，这些元数据对于识别具有特定属性的学习者非常有用

列出所有支持分类问题的学习器：
```r
learners_dt[task_type == "classif"]
```

多个条件进行过滤，列出所有可以预测标准误差的回归学习器：
```r
learners_dt[task_type == "regr" &
  sapply(predict_types, function(x) "se" %in% x)]
```

## 评价与基准 Evaluation and Benchmarking
监督机器学习模型只有在具有良好的泛化性能时才能在实践中部署，因此，泛化性能的准确估计对于机器学习应用和研究的许多方面都是至关重要的，它将是我们在多个模型中进行选择，以及超参数调整中的重要基础；

我们知道，使用相同的数据来训练和测试模型是一个糟糕的策略，它完全无法解决过拟合的问题。在前面的章节中我们介绍了 `partition()` ，它将数据集分为训练数据（用于训练模型的数据）和测试数据（用于测试模型和估计泛化性能的数据）[partition() 划分说明](/blog/2024/04/06/r-mlr3verse-learning-notes/) 这被称为 holdout strategy, 它将是我们这一章内容的开始。随后我们将考虑评估泛化性能的更高级策略。

一个常见的误解是，holdout 和其他更高级的 resampling strategies 可以防止模型过拟合，事实上这些方法只是让过拟合变得可见，因为我们可以单独评估训练/测试性能。他允许我们对泛化误差进行几乎无偏的估计

### Holdout 策略
ML的一个重要目标是学习一个模型，然后可以用来预测新数据。为了使这个模型尽可能准确，我们理想地使用尽可能多的数据来训练它。然而，数据是有限的，正如我们所讨论的，我们不能在相同的数据上训练和测试模型。

在实践中，人们通常会创建一个中间模型，该模型在可用数据的子集上进行训练，然后在其余数据上进行测试。通过将模型预测与真实数据进行比较而获得的该中间模型的性能是对最终模型的泛化性能的估计。最后拿到中间模型信息和超参数信息在全部的数据上训练模型，就是我们最后输出的结果

holdout策略是一种简单的方法，在训练和测试数据集之间创建分割。理想情况下，训练数据集应该尽可能大，以便中间模型尽可能代表最终模型，另一方面，测试数据集应该尽可能大，以便准确的估计泛化误差。

根据经验，通常使用2/3的数据进行训练，1/3用于测试，因为这在泛化性能估计的偏差和方差之间提供了合理的权衡

在前面我们已经介绍过他需要使用的代码了 现在再简要复习一下
```r
tsk_penguins = tsk("penguins")
splits = partition(tsk_penguins)
lrn_rpart = lrn("classif.rpart")
## 在训练集上训练 测试集上预测
lrn_rpart$train(tsk_penguins, splits$train)
prediction = lrn_rpart$predict(tsk_penguins, splits$test)
## 对测试集的预测结果进行评分
prediction$score(msr("classif.acc"))
```

在分割数据时，必须先对观测值进行打乱顺序，以删除数据排序中编码的任何信息。 这是因为 `tasks` 建立所使用的数据很可能使用一些含有规律的数据建立，这也是我们在收集数据中常见的操作。但是它会影响我们的测试与训练集划分

`partition()` 和下面讨论的所有 resampling 策略都会自动随机分割数据，以防止任何偏倚，保证我们对模型的训练 预测 泛化误差的估计都是有效的

许多性能指标都是基于“可分解”的损失，这意味着它们首先在观察水平上计算预测值和真值之间的差异，然后将测试集上的各个损失值汇总为单个数值分数

事实上 我们还有更复杂的评估策略 也就是不可分解的性能度量 后面如果继续讨论到再说.

### Resampling 策略
Resampling 策略重复地将所有可用数据分成多个训练和测试集。其中一个重复对应于 `mlr3` 中的 resampling iteration 或者称为重采样迭代

**泛化性能最终通过聚合多个重采样迭代的性能得分来估计**

通过重复数据拆分过程，数据点可重复用于训练和测试，从而更有效地使用所有可用数据进行性能估计。此外，大量的重新排序迭代可以减少分数的方差，从而得到更可靠的性能估计。这意味着性能估计不太可能受到“不幸的”拆分的影响

**我们一般可以认为Resampling 策略策略比前面介绍的 Holdout策略能给出更好的泛化误差估计。不过他会同时带来更多的性能开销，因为我们训练并测试了多个模型**

#### Resampling 策略理论解释
##### CV
一个非常常见的策略是k折交叉验证（CV）（ k-fold cross-validation）
它将数据随机划分为$k$个不重叠的子集，称为折; $k$模型总是在$k-1$折叠上训练，剩余的折叠用作测试数据;重复这个过程，直到每个折叠作为测试集精确地执行一次。最后，通常通过求平均值来汇总每个折叠的性能估计.CV保证每个观测值在测试集中只使用一次，从而有效地利用可用数据进行性能估计

$k$的常见值是5和10，这意味着每个训练集将分别由原始数据的4/5或9/10组成

CV存在几种变体，包括重复k折交叉验证（其中k折过程重复多次）和留一交叉验证（LOO-CV），其中折数等于观察数，导致每个折中的测试集仅由一个观察组成

理论解释可以参考 机器学习导论与监督学习：交叉验证法

##### Subsampling and Bootstrapping
子采样Subsampling 随机选择给定比例（常见的是4/5和9/10）的数据用于训练数据集，其中数据集中的每个观察结果都是从原始数据集中提取的，而不需要替换。模型在此数据上进行训练，然后在剩余数据上进行测试，此过程重复$k$次

Bootstrapping 策略就是自助法 可以参考 机器学习导论与监督学习：自助法

##### 策略选择
Resampling策略选择通常取决于手头的具体任务和绩效评估的目标，不过确实也有一些经验法则

如果可用数据相当小（$N<500$ ），可以使用大量重复的重复交叉验证来保持性能估计的方差较低（10次和10次折是一个很好的起点）

LOO-CV也被推荐用于这些小样本量，但这种估计方案计算开销非常大（除了存在计算捷径的特殊情况），并且违反直觉的具有相当高的方差。同时，他对于不平衡的二元分类任务中也存在问题

对于$500<N<5000$范围，通常建议5至10折CV

Bootstrapping已经变得不那么常见了 因为重复采样会导致不少机器学习算法出现问题

后面 我们将详细介绍怎么在R中施行这些Resampling策略

#### 建立 Resampling 策略
所有实现的 Resampling 策略都存储在 `mlr_resamplings` 字典中
```r
as.data.table(mlr_resamplings)
```
`params` 列示出了每个 Resampling 策略的参数 可以在后面构造`Resampling` 对象的时候修改参数
`iters`列显示了默认 Resampling 迭代的次数 我们一般不需要调整

`Resampling` 对象可以通过将策略“key”传递给sugar函数 `rsmp()` 来构造 比如
```r
rsmp("holdout", ratio = 0.8)
```
构建了 `holdout` 策略 我们修改了默认的划分比例 从三分之二训练集 到 五分之四训练集

从 `Resampling` 继承的对象的参数的调整和评价器 学习器完全一样 事实上
`Resampling` 构造的语法规则只是把Sugar Function 更换为了 `rsmp()`
```r
## three-fold CV
cv3 = rsmp("cv", folds = 3)
## Subsampling with 3 repeats and 9/10 ratio
ss390 = rsmp("subsampling", repeats = 3, ratio = 0.9)
## 2-repeats 5-fold CV
rcv25 = rsmp("repeated_cv", repeats = 2, folds = 5)
```

我们完全可以手动的拆分数据来训练 但是这种操作的繁琐 而且效果往往不好 建立 Resampling 策略就是为了让我们轻松的对准确的泛化误差进行估计

#### 将Resampling对象用于实际学习
`partition()` 函数接受任务自动为我们划分训练和测试集，返回行号；当然 Resampling对象也应该有这样的功能；

`resample()` 函数接受给定的 `Task` 、 `Learner` 和 `Resampling` 对象来运行给定的 Resampling 策略。 `resample()` 在训练集上重复拟合模型，在相应的测试集上进行预测，并将其存储在 `ResampleResult` 对象中，该对象包含估计泛化性能所需的所有信息
```r
rr = resample(tsk_penguins, lrn_rpart, cv3)
rr
```
**我们在Resampling的时候更改了前面学习器学习的流程，同时更改了训练和预测两个步骤**

当然 我们还需要评价器来评价 此时语法变换不大
```r
## 返回在每次迭代中的性能
acc = rr$score(msr("classif.ce"))
acc[, .(iteration, classif.ce)]

## 聚合多次迭代 给出更加常用的结果
rr$aggregate(msr("classif.ce"))
```

默认情况下，大多数度量将使用宏平均值（直接对各个测试集的得分平均）来聚合分数 但是我们可以指定使用微平均值（他考虑了各个测试集的大小不同的问题）来聚合分数
```r
## 这就是采用了微平均值 需要直接修改我们的评价器
rr$aggregate(msr("classif.ce", average = "micro"))
```
通过查询 `Measure` 对象的 `$average` 字段可以找到聚合方法的默认类型

可视化重新排序结果，可以使用 `autoplot.ResampleResult()` 函数将分数绘制为箱线图或直方图。直方图可以用来直观地衡量重新排序迭代中性能结果的方差，而箱线图通常用于并排比较多个学习器
```r
## 训练模型 返回ResampleResult对象
rr = resample(tsk_penguins, lrn_rpart, rsmp("cv", folds = 10))
## 使用 autoplot 函数 和前面一样 他为各种 mlr3 对象绘图 此时可以选择绘图的类型
autoplot(rr, measure = msr("classif.acc"), type = "boxplot")
autoplot(rr, measure = msr("classif.acc"), type = "histogram")
```

#### ResampleResult对象
我们前面为了使用Resampling策略 更改了学习器学习用的函数，因此 学习后产生的结果变为了ResampleResult对象

我们前面介绍了如何使用ResampleResult对象来计算泛化误差，但是ResampleResult对象不可能只能用于这一个目的 这里对它的其他方法进行介绍

我们可以使用 `$predictions()`方法来获得与每个resource迭代的预测相对应的 `Prediction` 对象的列表  **里面的对象是`Prediction`**
```r
## 返回结果是一个列表 里面含有迭代次数个元素
rrp = rr$predictions()
```

默认情况下，在预测步骤之后，在每个Resampling策略迭代中产生的中间模型被丢弃，以减少 `ResampleResult`对象的内存消耗（它最大的作用就是性能度量）

但是 我们可以通过设置 `store_models = TRUE` 来配置 `resample()` 功能以保持拟合的中间模型。然后，可以通过 `$learnersi$model` 访问在特定的恢复迭代中训练的每个模型，其中 `i` 是指第 `i`次恢复迭代
```r
rr = resample(tsk_penguins, lrn_rpart, cv3, store_models = TRUE)
## 得到各个学习器 后面的$model查看学习器的模型
rr$learners[[1]]$model
```
此时我们就可以查看关于模型的各个信息了 如果我们需要的话

#### 自定义Resampling
自定义Resampling确实是可能需要的 比如我们要复现其他研究者的结果 `mlr3` 提供了相应的方式

自定义 `holdout` 的代码 供参考
```r
rsmp_custom = rsmp("custom")

## resampling strategy with two iterations
train_sets = c(1:5, 153:158, 277:280)
rsmp_custom$instantiate(tsk_penguins,
  train = list(train_sets, train_sets + 5),
  test = list(train_sets + 15, train_sets + 25)
)
resample(tsk_penguins, lrn_rpart, rsmp_custom)$prediction()
```

自定义`cv`  的代码 供参考
```r
tsk_small = tsk("penguins")$filter(c(1, 100, 200, 300))
rsmp_customcv = rsmp("custom_cv")
folds = as.factor(c(1, 2, 1, 2))
rsmp_customcv$instantiate(tsk_small, f = folds)
resample(tsk_small, lrn_rpart, rsmp_customcv)$predictions()
```

#### 分层和分层
使用任务列角色，可以根据数据中的特定列对观察结果进行分组或分层
##### Grouped Resampling
在纵向研究中，在多个时间点从同一个体进行测量。如果我们不对这些数据进行分组，我们可能会高估模型对未知个体的泛化能力，因为对相同个体的观察可能同时存在于训练集和测试集中。

`"group"` 列角色允许我们在数据中指定定义观察的组结构的列。此时在构造 Resampling 的时候 原本对每一个观测的折叠会变成对各个组的折叠
```r
rsmp_loo = rsmp("loo")
tsk_grp = tsk("penguins")
tsk_grp$set_col_roles("year", "group") rsmp_loo$instantiate(tsk_grp)
```
##### Stratified Sampling
分层抽样确保训练集和测试集中的一个或多个离散特征将具有与包含所有观测的原始任务相似的分布；这可以保证在交叉验证中 多个迭代的估计都是准确的；

与分组不同，可以使用 `"stratum"` 列角色按多个离散特征进行分层。在这种情况下，层将由分层特征的每个组合形成 如下
```r
tsk_str = tsk("penguins")
## 设定 species 同时作为分层用的`"stratum"` 列 和"target"列
tsk_str$set_col_roles("species", c("target", "stratum"))
rsmp_cv10$instantiate(tsk_str)
```

### 基准测试 Benchmarking
监督机器学习中的基准测试是指在一个或多个任务上比较不同的学习器。

当在单个任务或由多个类似任务上比较多个学习器时，主要目的通常是根据预定义的性能度量对学习器进行排名，并确定所考虑的任务的最佳学习器。

当在多个任务上比较多个学习器时，主要目的往往没有前面那么简单。
例如，深入了解不同学习器在不同数据情况下的表现，或者是否存在严重影响某些学习器（或学习器的某些超参数）表现的某些数据属性。

由于基准测试通常由许多可以相互独立运行的评估组成， `mlr3` 因此提供了自动并行化它们的可能性。本节我们介绍使用的最广泛的基准测试，以后再讨论更复杂的基准测试问题

#### 基准
`mlr3` 基准实验是用 `benchmark()` 进行的，它只是分别对每个任务和学习者运行 `resample()` ，然后收集结果。提供的重采样策略会在每个任务上自动实例化，以确保将所有学习者与相同的训练和测试数据进行比较

非常明显的，对于基准测试问题，我们需要引入多个任务 多个学习器 可能多种Resample 方法 因此代码有
```r
## 建立两个任务
tasks = tsks(c("german_credit", "sonar"))
## 建立三个学习器
learners = lrns(c("classif.rpart", "classif.ranger",
  "classif.featureless"), predict_type = "prob")
## 建立一种 Resample 方法
rsmp_cv5 = rsmp("cv", folds = 5)

## 构造`benchmark()`方案并审阅
design = benchmark_grid(tasks, learners, rsmp_cv5)
head(design)
```

生成的设计本质上只是一个 `data.table` ，如果您想删除特定的组合，可以对其进行修改，甚至可以在没有该 `benchmark_grid()` 功能的情况下从头开始创建

然后，可以将构造的基准设计传递给 `benchmark()` 运行实验，结果是一个 `BenchmarkResult` 对象：
```r
bmr = benchmark(design)
bmr
```

由于 `benchmark()` 这只是 `resample()` 的扩展，我们可以再次使用 `$score()` ，或者 `$aggregate()` 借此审阅我们得到的结果
```r
bmr$score()[c(1, 7, 13), .(iteration, task_id, learner_id, classif.ce)]

bmr$aggregate()[, .(task_id, learner_id, classif.ce)]
```

在这里 我们没有进行严格的统计假设检验，因此要谨慎的给出具体哪个模型更优的结论

#### BenchmarkResult 对象
对象 `BenchmarkResult` 是多个 `ResampleResult` 对象的集合

我们可以提取出BenchmarkResult 对象中的`ResampleResult` 对象 有
```r
rr1 = bmr$resample_result(1)
rr1
```
之后如果需要更详细的访问 都可以通过[ResampleResult对象](/blog/2024/04/06/r-mlr3verse-learning-notes/) 中的代码来实现访问

此外， `as_benchmark_result()` 还可用于将对象从 `ResampleResult` 转换为 `BenchmarkResult` 。`c()` 可用于组合多个 `BenchmarkResult` 对象
```r
bmr1 = as_benchmark_result(rr1)
bmr2 = as_benchmark_result(rr2)

c(bmr1, bmr2)
```

BenchmarkResult 对象 也有专属的可视化方法 它给出boxplot来比较多个算法的效果
```r
autoplot(bmr, measure = msr("classif.acc"))
```

### 二元分类器的评估
我们在[分类评价器](/blog/2024/04/06/r-mlr3verse-learning-notes/) [分类中的预测](/blog/2024/04/06/r-mlr3verse-learning-notes/) 里面介绍过关于分类评价的问题了；现在 我们把这个问题再深入一步研究

我们在机器学习的理论部分介绍了一些相关知识 机器学习导论与监督学习：性能度量 这里简单复习并且进行代码实现

`mlr3measures` 软件包允许您使用以下 `confusion_matrix()` 函数计算几个常见的基于混淆矩阵的度量
```r
mlr3measures::confusion_matrix(truth = prediction$truth,
  response = prediction$response, positive = tsk_german$positive)
```

绘制 ROC 曲线则需要
```r
autoplot(prediction, type = "roc")
```

给出AUC的计算值则需要 它并不属于混淆矩阵衍生出的常见度量
```r
prediction$score(msr("classif.auc"))
```

绘制 PRC 曲线(精密度-召回率曲线)则需要
```r
autoplot(prediction, type = "prc")
```

最后我们考虑阈值和指标之间的关系有
```r
autoplot(prediction, type = "threshold", measure = msr("classif.fpr"))
autoplot(prediction, type = "threshold", measure = msr("classif.acc"))
```

这些可视化的方法完全可以用于 `ResampleResult BenchmarkResult
对象 替换原本的` prediction` 就可以了

## 超参数优化 Hyperparameter Optimization
从本章开始 我们连续三章研究如何提升学习器性能；包括最基本的自动超参数调节；再研究进一步的调优方法与特征工程；这将是我们在研究完基础的`mlr3`之后的进阶流程 我们开始学习如何构建一个更好的模型
