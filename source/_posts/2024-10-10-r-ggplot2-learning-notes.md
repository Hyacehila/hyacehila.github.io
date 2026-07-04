---
title: "R ggplot2 学习笔记"
title_en: "R ggplot2 Learning Notes"
date: 2024-10-10 17:21:49 +0800
categories: ["Programming", "R"]
tags: ["Learning Notes", "R", "ggplot2"]
author: Hyacehila
excerpt: "一篇 R ggplot2 学习笔记，整理几何对象、统计量、标度、坐标系、分面、主题、保存以及常见外观调整。"
excerpt_en: "A study note on ggplot2 in R, covering geoms, statistics, scales, coordinate systems, facets, themes, saving plots, and common appearance adjustments."
mathjax: false
hidden: true
permalink: '/blog/2024/10/10/r-ggplot2-learning-notes/'
---

关于 ggplot2 系统的基本介绍我们已经在 [ggplot2作图系统简介](/blog/2024/09/16/r-graph-learning-notes/)中完整的介绍过了，这里我们介绍其中各个子模块中都有什么样的函数可以使用，以及我们应该如何组合这些模块得到我们想要的图形

### 基本介绍
在 **ggplot2** 中，画图只需要将若干个图层简单相加即可，语法非常精炼
```r
library(ggplot2)

p <- ggplot(aes(x = wt, y = mpg), data = mtcars) +
  geom_point() +
  labs(title="Automobile Data", x="Weight", y="Miles Per Gallon")
#ggplot()初始化图形并且指顶要用到的数据来源（mtcars）和变量（hp、mpg）
#aes()则是解释了我们如何使用数据，这里把x轴给了hp y轴给了mpg
#geom_point()函数在图形中画点，创建了一个散点图
#labs()添加了注释 包括了标题和轴注释

p + geom_smooth(method = "loess")
#加上平滑层

print(p)
#输出图形到默认显示器
```

使用 ggplot2 时通常我们不必担心细节问题，例如图形的边距会自动调整，不会留出大片空白，元素颜色会自动根据变量取值从调色板中选取，图例会自动添加，等等。**这些自动化的设计可以为我们节省大量的调整细节的时间**

函数 `ggplot()` 是 ggplot2 中的核心函数之一，它能让我们快速画出灵活的图形; 我们只需要提供一个数据框 `data`

整个 **ggplot2** 系统大致由几何形状（geom）、统计量（statistic）、标度（scale）、坐标系（coordinate system）和切片（facet）构成

由于其高度的自动化设计 围绕着**ggplot2**  已经发展出了一套新的可视化系统，大量的包扩展了**ggplot2** 的功能 他们之间也互相吸收 形成完整的 简单易用且美观的R图形系统
### 几何形状（geom）
在 ggplot2 中几何形状简称 geom （Geometric objects），这些形状包括：点、条、线、箱线图和文本等。实际上它们就是前面介绍的基础图形元素，

但是 ggplot2 在这些元素上做了更多工作，例如箱线图并非基础图形元素，但它在 ggplot2 中的地位也是基础形状，还有平滑曲线和平滑带，背后都涉及到大量的统计计算，而 ggplot2 对它打包之后，用户用起来就简单多了

![R 统计可视化-13](/assets/images/r-learning-notes/r-stat-visualization-13.png)
```r
## 汽车马力与每加仑汽油行驶里程的关系
library(ggplot2)
p = ggplot(aes(x = hp, y = mpg), data = mtcars) +
  geom_point() +
  geom_smooth(method = "loess") +
  labs(x = "马力", y = "每加仑汽油行驶里程")
print(p)
```
### 统计量（statistic）
统计量指定了对原始数据做何种变换，进而用几何形状表达出来。ggplot2 中除了划分直方图区间求频数、求分位数、计算密度值这些普通的变换功能之外，还有一些新颖的统计量，例如根据二维数据用网格划分区间求每个格子内的数据频数（实际上就是二维直方图），或者用蜂巢形状将平面划分为一系列的六边形区间再求数据频数。

![R 统计可视化-14](/assets/images/r-learning-notes/r-stat-visualization-14.png)
```r
## 钻石重量与价格的蜂巢图
library(ggplot2)
p = ggplot(aes(x = carat, y = price), data = diamonds) +
  geom_hex() + labs(x = "重量", y = "价格", fill = "频数")
print(p)
```

### 标度（scale）
标度通常指定如何从数据映射到几何形状的颜色、符号和大小等属性，这也是 ggplot2 系统的一个非常吸引人的特征。大多数情况下，我们只需要指定用来做标度的变量即可，剩下的映射工作 ggplot2 会自动完成。

![R 统计可视化-15](/assets/images/r-learning-notes/r-stat-visualization-15.png)
```r
data("iris")
library(ggplot2)
p = ggplot(aes(x = Petal.Length, y = Petal.Width), data = iris) +
  geom_point(aes(color = Species, shape = Species)) +
  labs(x = "花瓣长度", y = "花瓣宽度", color = "种类", shape = "种类")
print(p)
```

### 坐标系（coordinate system）
我们平时用到的坐标系大多数都是笛卡尔坐标系，ggplot2 也提供了极坐标系和地图坐标系，并支持笛卡尔坐标系的翻转，即交换 x 轴和 y 轴。

函数 `coord_flip()` 可以用来翻转几乎任何图形，而且由于 ggplot2 的“图层分解”概念，我们可以先画一幅图保存在一个变量中，如果想翻转就加上 `coord_flip()`再打印即可

![R 统计可视化-16](/assets/images/r-learning-notes/r-stat-visualization-16.png)
```r
## 钻石雕琢水平和对数价格的关系
data("diamonds")
library(ggplot2)
library(patchwork)
levels(diamonds$cut) = c("一般", "良好", "优质", "珍贵", "完美")
p = ggplot(aes(x = cut, y = log(price)), data = diamonds) +
  geom_boxplot() + labs(x = "切工", y = "log(价格)")
print(p / (p + coord_flip()))
```

### 切片（facet）
切片的思想来自于 Trellis 图形：将整批数据按照某一个或两个分类变量切成一个个子集，然后对这些子集画图。在 ggplot2 中实现切片也很简单，指定 `facets` 参数通常就可以了，这个参数取值为一个公式，公式左侧决定在行上摆放的子集图形，右侧决定列上的图形。

![R 统计可视化-17](/assets/images/r-learning-notes/r-stat-visualization-17.png)
```r
## 按雕琢水平切片后的钻石重量密度曲线
data("diamonds")
library(ggplot2)
levels(diamonds$cut) = c("一般", "良好", "优质", "珍贵", "完美")
p = ggplot(aes(x = carat), data = diamonds) +
  geom_density() +
  labs(x = "重量", y = "分布密度") +
  facet_grid(cut ~ .)
print(p)
```

### 主题
ggplot2 图形有一套自己的独特风格，它与别的图形系统在外观上的典型区别就是它通常会画一个灰色的背景，背景中有网格线。

首先，网格线是为了辅助阅读图形而画的，这是非常重要的图形组成部分；其次，灰色的背景也有其原因：因为一篇文章的文字通常是黑色，所以灰底的图形会和黑色文字能融合得更好，这是一点美学上的考虑。

有的用户可能喜欢这样的设置，有的用户则可能很不习惯这种默认设置。ggplot2 可以自定义主题 一些附加包也为我们提供了额外的主题可以选择

### 保存
`ggsave()`函数能方便地保存它。它的选项包括保存哪幅图形，保存在哪里和以什么形式保存

```r
myplot <- ggplot(data=mtcars, aes(x=mpg)) + geom_histogram()
ggsave(file="mygraph.png", plot=myplot, width=5, height=4)
```

设定文件的扩展名可以设置导出的格式

## 几何函数 (geom)
ggplot()函数指定要绘制的数据源和变量，几何函数则指定这些变量如何在视觉上进行表示（使用点、条、线和阴影区）有大量的函数可供使用 他们统一使用 `geom_` 作为函数前缀 下面是一些常用的函数，更多的需求可以查阅帮助或者其他搜索工具 这些函数的使用可以查阅帮助

| 函数                  | 几何对象 |
| ------------------- | ---- |
| `geom_bar()`        | 条形图  |
| `geom_boxplot()`    | 箱线图  |
| `geom_density()`    | 密度图  |
| `geom_histogram() ` | 直方图  |
| `geom_hline()`      | 水平线  |
| `geom_jitter() `    | 抖动点  |
| `geom_line() `      | 线图   |
| `geom_point() `     | 散点图  |
| `geom_rug()`        | 坐标轴须 |
| `geom_smooth() `    | 拟合曲线 |
| `geom_text() `      | 文字注解 |
| `geom_violin()`     | 小提琴图 |
| `geom_vline() `     | 垂线   |

每个函数都有一些参数可以选择和设置，这里简单介绍其中的一部分

| 参数       | 参数的含义                                                               |
| -------- | ------------------------------------------------------------------- |
| color    | 对象的着色，也包括图形的边界（着色方法参考[标度（scale）](/blog/2024/09/16/r-graph-learning-notes/) 也可以直接设置颜色） |
| fill     | 对象填充区域的着色                                                           |
| alpha    | 透明度设置（从0到1）                                                         |
| linetype | 线条的样式（如果有线条）（1=实线，2=虚线，3=点，4=点破折号，5=长破折号，6=双破折号）                    |
| size     | 点的尺寸和线的宽度                                                           |
| shape    | 点的形状 参考参数pch [R Graph 中的点形参数说明](/blog/2024/09/16/r-graph-learning-notes/)                                    |
| position | 对象的位置设置 条形图的排列方式设置和点的减少重叠                                           |
| sides    | 坐标轴须的位置                                                             |
| width    | 宽度设置                                                                |

## 切片（facet）
切片让我们来建立并列而非重叠的图形，他与我们在[分组](/blog/2024/10/10/r-ggplot2-learning-notes/)中介绍的堆叠方法一样 基于因子变量改变图形的绘制逻辑

我们可以使用`facet_wrap()`函数和`facet_grid()`函数创建网格图形，其中`var rowvar colvar`是因子

| 函数                           | 切片排列方式                  |
| ---------------------------- | ----------------------- |
| `facet_wrap(~var, ncol=n)`   | 根据var切片成n列              |
| `facet_wrap(~var, nrow=n)`   | 根据var切片成n行              |
| `facet_grid(rowvar~colvar) ` | 根据rowvar划分行，根据colvar划分列 |
| `facet_grid(rowvar~.) `      | 一行一个 成一列图形              |
| `facet_grid(.~colvar) `      | 一列一个 成一行图形              |

## 其他情况
### 分组
分组的核心意义在于堆叠比较，在一个图中画出两个或更多组的观察值，然后比较他们之间的区别，当然，我们可以用绘制多个图然后 `+` 他们的方法来实现堆叠，但是他们自然没有原生提供的分组功能更好用

`ggplot()`声明中的`aes()`函数负责分配变量（图形的视觉特征），所以这是一个分配分组变量的自然的地方 使用`fill` 参数实现

```r
data(Salaries, package="car")
library(ggplot2)
ggplot(data=Salaries, aes(x=salary, fill=rank)) +
       geom_density(alpha=.3)
#基于rank因子变量进行分组绘制，我们的实现方法是根据rank设置填充颜色
```

**通常来说，变量应该设在`aes()`函数内，分配常数应该在`aes()`函数外在后面的绘图函数中我们也可以继续设置`aes()`函数，如果我们相对某个图层进行某种美学的映射**

### 外观修改
`par()` 对图形绘制的基本外观的修改不起作用，这里需要使用 **ggplot2** 提供特定的函数来进行外观的修改

#### 坐标轴
如下表

| 函数                                             | 选项                                                                 |
| ---------------------------------------------- | ------------------------------------------------------------------ |
| scale_x_continuous()和 <br>scale_y_continuous() | breaks=指定刻度标记，labels=指定刻度标记标签，limits=控制要展示的值的范围（仅对连续的坐标轴成立）        |
| scale_x_discrete()和 <br>scale_y_discrete()     | breaks=对因子的水平进行放置和排序，labels=指定这些水平的标签，limits=表示哪些水平应该展示（仅对离散坐标轴成立） |
| coord_flip()                                   | 翻转坐标轴                                                              |

#### 图例
ggplot2包能自动生成图例，而且在很多时候能够满足我们的需求，但是了解如何手工定义图例也是重要的

图例的标题和位置是最常用的定制特征，示例代码如下
```r
data(Salaries,package="car")
library(ggplot2)
ggplot(data=Salaries, aes(x=rank, y=salary, fill=sex)) +
       geom_boxplot() +
       labs(title="Faculty Salary by Rank and Gender",                   x="", y="", fill="Gender") +
       theme(legend.position=c(.1,.8))
```

此时我们所有的图例为填充所使用的 `fill` 参数，想要修改图例的标题，我们需要在`labs`中设置 `fill = 'mytitle'`

图例的位置需要在主题中设置，可以设置为（）"left"、"top"、"right"（默认值）和"bottom" 也可以使用一个二元的向量 这里标示左侧边缘10%和底部边缘80%的部分

删除图例，可以使用`legend.position="none"`
