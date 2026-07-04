---
title: "R 进阶学习笔记"
title_en: "R Advanced Learning Notes"
date: 2024-09-16 19:09:37 +0800
categories: ["Programming", "R"]
tags: ["Learning Notes", "R"]
author: Hyacehila
excerpt: "一篇 R 进阶学习笔记，整理帮助系统、函数源代码查看、类函数与非类函数以及命名空间访问方式。"
excerpt_en: "An advanced R study note covering the help system, source inspection, functions versus methods, and namespace access patterns."
mathjax: false
hidden: true
permalink: '/blog/2024/09/16/r-advanced-learning-notes/'
---

## R 帮助

关于R的基本知识 我们可以用下面的代码查阅，我们可在其中查看入门和高级的帮助手册

```r
help.start()
```

关于函数或者那些有着特殊语法含义的字符 我们还是用help函数；这个函数只会直接检验library中的函数 如果想要检验全部安装的包 要使用第二种方式

```r
help(fun)
help("bs",try.all.packages=TRUE)
help("bs",package = "splines")
```

R 也有专门的介绍性PDF文档，但是不是所有Packages都提供，下面的函数可以查看
```r
vignette()
```

一种更为简单的获取帮助的方式是 ? 后面跟package或者function都可以

```r
?lm()
```

## R 查看函数源代码
一个扩展包中定义的函数有区分公开和不公开的 对于公开源代码的函数 可以用我们后面介绍的各种方法来找到源代码；对于不公开源代码的函数，我们只能联系扩展包的制作者；

哪怕是一个公开源代码的函数 根据函数的类型 我们会有不同的获取源代码的方法 这是因为 哪怕是 `base R` 他也是一门面向对象设计的语言 针对不同的对象会有这不同的函数对应

面向对象这一点在 `mlr3 `中理解比较深 但是实际上 哪怕只是最简单的函数 `mean() `他针对不同的输入`object` 比如 向量 矩阵 数据框 都会有不同的结果

事实上 哪怕同样是面向对象 R中依旧提出了类函数（`methods`）和非类函数（`functions`）的区别  这点也导致了函数调用方法的不同

### 类函数与非类函数
在R中，函数可以分为类函数（methods）和非类函数（functions） 这两种函数之间的主要区别在于它们对待对象的方式和调用方式
#### 非类函数（functions）
- 非类函数是R中最基本的函数形式，它们接受参数并返回结果，类似于其他编程语言中的函数
- 非类函数不与特定的对象或类相关联，因此在调用时不需要通过对象来访问

示例：`mean()`、`sum()`、`print()`等函数都是非类函数，它们可以直接调用，例如 `mean(x)`、`sum(numbers)`、`print("Hello")` 他们自适应输入来输出

#### 类函数（methods）
- 类函数是与特定类或对象相关联的函数，它们对对象执行操作或提供特定的功能
- 类函数可以在不同的类之间具有相同的名称，但根据对象的类别，调用不同的实现
- 使用类函数时，需要通过对象的类别来调用，以便选择正确的实现

 示例：在面向对象编程中，一个对象可以有其自己的方法（例如 `print()`）这些方法是特定于该对象的，而其他对象的相同方法可能会执行不同的操作

总之，在R中，非类函数是通用的函数，可以直接调用，而类函数与特定类或对象相关联，需要通过对象的类别来调用
对于非类函数，可以简单地使用函数名和参数调用，而对于类函数，通常需要首先创建对象，然后通过对象的方法来调用函数

### 查看函数的源代码
#### 基本形式
最简单的方式是直接输入函数的名称 不添加括号声明我们要调用一个函数 如下 我们查看回归分析中`lm()`函数的源代码
```r
lm
```

#### 多对象形式
这种方法并不是万能的 **对于计算方法不同的函数，要用`methods()`来定义具体的查看对象** 如
```r
mean
```

它的返回结果是
```text
function (x, ...)
UseMethod("mean")
```

意味着 这个函数有着不同的计算方法（针对不同类型的对象进行） 提示我们查看不同的计算方法分别查看 于是我们有
```r
methods(mean)
```

返回结果为
```text
[1] mean.Date     mean.default  mean.difftime mean.POSIXct  mean.POSIXlt
```

想要查看具体的函数源代码我们需要针对上面的不同函数分别查看 有
```r
mean.default
```

#### 类函数形式
对于前面中`methods()`得出的类函数中带星号标注的源代码，用函数`getAnywhere()`  如

代码
```r
methods(predict)
```

返回结果
```text
[1] predict.ar* predict.Arima* ...
```

此时如果输入函数名字是无法查看到函数源代码的  会提示
```text
Error: object 'xxx' not found
```

此时我们需要
```r
getAnywhere(predict.Arima)
```

### 写在最后
直接在CRAN 下载函数源代码包 下载解压后就可以查看函数源代码

特别的 部分的R packages 使用 C 或者 Fortran 写成 这一般是为了提高计算的效率 此时我们只能去下载函数源代码包 使用 C 与 Fortran 相关的IDE查看相关代码的解释
