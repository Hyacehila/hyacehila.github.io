---
title: 'R Statistical Visualization: Univariate, Multivariate, and Functional Data Graphics'
title_zh: R 统计可视化：单变量、多变量与函数型数据图形
date: 2024-03-15 16:54:45 +0800
categories:
- Programming
- R
tags:
- R
- Data Visualization
author: Hyacehila
mathjax: true
hidden: true
excerpt: A overview on statistical visualization in R, organizing common graphics for univariate, bivariate, multivariate,
  and functional data with base R and ggplot2 examples.
description: A overview on statistical visualization in R, organizing common graphics for univariate, bivariate, multivariate,
  and functional data with base R and ggplot2 examples.
excerpt_zh: 按单变量、双变量、多变量和函数型数据整理常见统计图形及 base R、ggplot2 实现。
permalink: /blog/2024/03/15/r-visualization-learning-notes/
lang: en
translation_key: 2024-03-15-r-statistical-visualization
translation_status: machine
translation_source_hash: ab53c7235e9bec4adba1f3084f3b5fa690124bb72544fa967c551f9719be1fc9
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<p>The notes will follow the structure of the book Modern Statistical Graphics, supplementing the content of Data Visualization: Based on the R Language - Joa Junping; for convenience and cross-cutting with other knowledge structures, they will eventually be presented in the form of a Markdown in OBSIDIAN;</p>
<p>Fortunately, the MSG package provides us with all the graphics we need to build the notes, and we do not need to rewrite all the codes we need to get the graphics we need;</p>
<p>Based on the structure in modern statistical graphics, we divided the entire note into three main parts, each presenting the statistical drawing itself, presenting the various statistical graphics in a dictionary, and finally some of the statistical drawing modules in R.</p>
<p>We choose to move the rest of the library to <a href="/en/blog/2024/09/16/r-graph-learning-notes/">R Graph</a> Studying ideas and methods in it.</p>
<h2>Single Variable Chart</h2>
<p>Starting with this chapter, we're going to present the diagram of statistical graphics, and we're going to make a broad classification by variable structure; we're going to want to be able to give a detailed description of all the graphics, the analysis methods, and including...<code>base R </code>and <code>ggplot2</code> Code Achieved</p>
<p>Of which graphics are usually only used<code>base R</code>It's an example of a slightly different view; <code>ggplot2</code> We'll just leave it here for reference.</p>
<p>In terms of code realization, we don't explain parameters in detail, but we just describe the features of the function, and we can use more of the changes we need.<code>help</code></p>
<p><strong>The single variable diagram is designed to show a single variable, and we sometimes compare multiple single variables, but still falls within the single variable diagram (although multiple variables are used).</strong></p>
<h3>Bar Chart</h3>
<p>The barchart is currently the most widely used of all statistical graphics, but the barchart shows relatively poor statistics: It shows the original values only by the length of the rectangular bar, without any summary or extrapolation of the data.</p>
<h4>Basic introduction</h4>
<p>The function of the barchart in R is <code>barplot()</code></p>
<ul>
<li>Core parameters <code>height</code> Specifies the length of a long bar to accept a numerical vector or matrix, which is the most basic bar map in the case of an accepted numerical vector. <strong>If he accepts the numeric matrix, he's drawing each column as one, and at this point he's going to be<code>beside</code>Parameter Control</strong></li>
<li>Parameters <code>beside</code> Set to FALSE, each column of the matrix takes one <code>beside</code> Set to TRUE without stacking</li>
<li><code>horiz</code> Sets the direction of the chart.</li>
</ul>
<h4>Graphics Example</h4>
<p><img src="/assets/images/r-learning-notes/r-language-stat-visualization-14.png" alt="Statistical visualization 14">
It's been modified. <code>beside</code> Parameters</p>
<h4>base R</h4>
<pre><code class="language-r">## 基础作图法绘制弗吉尼亚死亡率数据条形图
data(VADeaths)
library(RColorBrewer) # 用分类调色板
par(mfrow = c(2, 1), mar = c(3, 2.5, 0.5, 0.1))
death = t(VADeaths)[, 5:1]
barplot(death, col = brewer.pal(4, &quot;Set1&quot;))
barplot(death, col = brewer.pal(4, &quot;Set1&quot;),
        beside = TRUE, legend.text = TRUE)
</code></pre>
<h4>ggplot</h4>
<pre><code class="language-r">## ggplot2 绘制弗吉尼亚死亡率数据条形图
library(ggplot2)
library(patchwork)
data(VADeaths)
reshape_VADeaths = transform(
  expand.grid(sex = colnames(VADeaths), age = rownames(VADeaths)),
  rates = as.vector(t(VADeaths))
)
p = ggplot(data = reshape_VADeaths,
            aes(x = age, y = rates, fill = sex)) +
  labs(x = &quot;年龄&quot;, y = &quot;死亡率&quot;, fill = &quot;性别&quot;) +
  scale_fill_discrete(labels = c(&quot;农村男性&quot;, &quot;农村女性&quot;,
                                 &quot;城市男性&quot;, &quot;城市女性&quot;))
p1 = p + geom_col(position = &quot;stack&quot;)
p2 = p + geom_col(position = &quot;dodge&quot;)
print(p1 / p2)
</code></pre>
<h3>Cleveland Point</h3>
<p>The functions of the dots and bars are very similar: the length of the bar represents the size of the value and the position of the point of the dot indicates the size of the value, which can be exchanged almost in any case.</p>
<h4>Basic introduction</h4>
<p>The function of the mid-point chart for R is <code>dotchart()</code></p>
<ul>
<li>Parameters <code>x</code> and in the bar chart <code>height</code> Consistent needs</li>
</ul>
<h4>Graphics Example</h4>
<p><img src="/assets/images/r-learning-notes/r-language-stat-visualization-16.png" alt="Language R Statistical visualization 16">
It can be seen as just changing direction.
<strong>Very little is used because it doesn't have a bar chart intuitively, and it's only possible when the number of points is small.</strong></p>
<h4>base R</h4>
<pre><code class="language-r">## 基础作图法绘制弗吉尼亚死亡率数据的 Cleveland 点图
library(RColorBrewer)
data(VADeaths)
colnames(VADeaths) = c(&quot;农村男性&quot;, &quot;农村女性&quot;, &quot;城市男性&quot;, &quot;城市女性&quot;)
par(mar = c(2, 6, 0.2, 0.2))
dotchart(t(VADeaths)[, 5:1],
         col = brewer.pal(4, &quot;Set1&quot;), pch = 19, cex = .65)
</code></pre>
<h4>ggplot</h4>
<p>ggplot does not provide a specific dot-drawing function, which we can only use as a sprawl curve;<strong>The use of the dots is too low.</strong></p>
<pre><code class="language-r">## ggplot2 绘制弗吉尼亚死亡率数据的 Cleveland 点图
data(VADeaths)
library(ggplot2)
colnames(VADeaths) = c(&quot;农村男性&quot;, &quot;农村女性&quot;, &quot;城市男性&quot;, &quot;城市女性&quot;)
tm = rownames(VADeaths)
rownames(VADeaths) = NULL
vd = data.frame(cbind(tm, VADeaths))
vd = reshape(vd, direction = &quot;long&quot;, varying = names(vd)[2:5],
              v.name = c(&quot;rate&quot;), times = names(vd)[2:5])
vd&#36;rate = as.numeric(vd&#36;rate)
vd&#36;tm = factor(vd&#36;tm)
vd&#36;tm = factor(vd&#36;tm,levels = rev(levels(vd&#36;tm)))
p = ggplot(vd, aes(time, rate, color = time)) + geom_point() +
  facet_grid(tm ~ .) + coord_flip() +
  theme(legend.position = &quot;&quot;, axis.title = element_blank())
print(p)
</code></pre>
<h3>Histogram</h3>
<p>Histogram (Histogram) is the most commonly used tool to demonstrate continuous data distribution and is essentially an estimate of density functions.</p>
<p>Histograms serve as the basic idea for the density function estimation tool: dividing the compartment and counting how many data points fall into it. The actual data cannot be infinity, so the h-0 conditions are often impossible to achieve, so we go back to the other end of it, just to estimate the density of the zone in some segments.</p>
<p>With regard to compartmentalization, we need to point out, in particular, that the theory of the histogram is not as simple as might have been imagined or apparent, that the window width is not optional and that different window widths or partitioning methods lead to different estimates of error; that, therefore, the histograms that allow users to set the width at random are often unreliable; and that adding a normal density estimate curve is not valuable to the histogram either because the sample is not necessarily normal</p>
<p>The histogram is actually a discrete grouping of data, so it is inevitable that it is random.<strong>There's a certain theoretical background in this group that doesn't lose as much information as any random grouping.</strong></p>
<p>When drawing histograms (including moving average hetograms), add density curves, if possible, or coordinate axes; as density curves are not influenced by clusters, the axis must reflect the location of the original data and avoid errors caused by certain clusters</p>
<h4>Basic introduction</h4>
<p>R provided <code>hist()</code> Function for drawing histograms</p>
<ul>
<li>Parameters<code>x</code> To estimate the numerical vector of the distribution</li>
<li>Parameters<code>breaks</code> The method of calculating the partitions was determined, which could be a vector (in turn, an inter-zone endpoint), a number (to decide how many sections to split), a string (to give the name of the algorithm to calculate the partitions), or a function (to give the number of partitions) <strong>As explained earlier, we know this parameter is very important.</strong></li>
<li><code>freq</code> and <code>probability</code> Parameters are based on logical values (which are mutually exclusive), the former on the number of frequencies and the latter on the probability density (in which case rectangular area is 1)</li>
<li><code>labels</code> Whether to add the value of the frequency to the upper of the rectangle bar for logical values</li>
</ul>
<h4>Graphics Example</h4>
<p><img src="/assets/images/r-learning-notes/r-language-stat-visualization-17.png" alt="Statistical visualization-17">
It reflects the division and the effect of frequency parameters.</p>
<h4>base R</h4>
<pre><code class="language-r">## 基础作图法绘制直方图与密度曲线的结合
par(mar = c(1.8, 3, 0.5, 0.1), mgp = c(2, 0.5, 0), mfrow = c(1, 2))
data(geyser, package = &quot;MASS&quot;)

hist(geyser&#36;waiting, freq = FALSE, main = &quot;&quot;)
lines(density(geyser&#36;waiting))

hst = hist(geyser&#36;waiting, probability = TRUE,
           main = &quot;&quot;, xlab = &quot;waiting&quot;)
d = density(geyser&#36;waiting)
polygon(c(min(d&#36;x), d&#36;x, max(d&#36;x)), c(0, d&#36;y, 0),
        col = &quot;lightgray&quot;, border = NA)
lines(d)
ht = NULL
brk = seq(40, 110, 5)
for (i in brk) ht = c(ht, d&#36;y[which.min(abs(d&#36;x - i))])
segments(brk, 0, brk, ht, lty = 3)
</code></pre>
<h4>ggplot</h4>
<pre><code class="language-r">## ggplot2 绘制直方图与密度曲线的结合
library(ggplot2)
data(geyser, package = &quot;MASS&quot;)
p = ggplot(aes(waiting), data = geyser) +
  labs(x = &quot;间隔时间&quot;, y = &quot;分布密度&quot;) +
  geom_histogram(breaks = seq(40, 110, by = 5), aes(y = ..density..))+
  geom_density(color = &quot;blue&quot;, size = 1.2)
print(p)
</code></pre>
<h4>On nuclear density estimates</h4>
<p>The theory of nuclear density estimates is well developed today, and he's given a very good way of estimating the probability distribution of successive variables, and the other methods have completely lost their value.</p>
<ul>
<li>Yeah. <code>base R</code> We use<code>density()</code> Estimating nuclear density and mapping separately</li>
<li>Yeah. <code>ggplot</code> We offer the original method.</li>
</ul>
<p><strong>A composite nuclear density and histogram is the only solution to the probability distribution of successive variables today, and we also have a very sophisticated mapping tool for nuclear density at different factor levels.</strong></p>
<h3>Tube Chart</h3>
<p>Now he's out of value.</p>
<h4>Basic introduction</h4>
<p>The function of the mid-R leaf map is <code>stem()</code></p>
<ul>
<li>Parameters <code>scale</code> Controls m, i.e. the length between sections ( <code>scale</code> The larger the m, the smaller;</li>
<li><code>width</code> Controlled the width of the sprite, and if the length of the leaves exceeds that, the leaves will be intercepted to the length. <code>width</code> , and then a whole number of leaves left behind.</li>
<li><code>x</code> Align with Histogram</li>
</ul>
<h4>Graphics Example</h4>
<p>The distribution of land mass is severe right. Offset
<img src="/assets/images/r-learning-notes/r-language-stat-visualization-18.png" alt="Statistical visualization - 18"></p>
<h3>Line Chart</h3>
<p>Box diagrams (Box Plot or Box-and-Whisker Plot) describe the distribution of data mainly from the angle of a four-point number; We can generally extrapolate the trend of data concentration or fragmentation (the shorter the length, the more dense the data are in the zone and the more thin the data are in the opposite).</p>
<h4>Basic introduction</h4>
<p>The corresponding function in R is <code>boxplot()</code></p>
<p><code>boxplot()</code> is a broad function, so it adapts to different parameter types. It currently supports two types of parameters: formulae. <code>formula</code> ) and data, which may be easier for us to understand (to give a set of data, to make the corresponding box diagrams), while the former generate multiple parallel box charts based on type variants, suitable for intuitive examination of the average values of groups</p>
<p>For the former, the parameters can be explained as follows:</p>
<ul>
<li>Parameters <code>x</code> A numerical vector or a list, and if a list, make a box line in order for each sub-object in the list Figure</li>
<li><code>range</code> It's an extension multiple that determines where the end (must) of the box chart extends, mainly for reasons of isolation, only from both ends of the box.&#36;range\times Q_3-Q_1&#36; Location</li>
<li><code>width</code> The width of the given box</li>
<li><code>varwidth</code> is the logical value, if <code>TRUE</code>, the width of the box is proportional to the square root of the sample, which is more useful when multiple batches of data are drawn together with multiple box lines, which further reflects the size of the sample.</li>
<li><code>notch</code> It's also a useful logical parameter, which determines whether to draw a dent on the box, which is actually an estimate of a medium number.
The rest of the questions we can refer to the help document, some parameters, for example. <code>horizontal</code> Similar Parameter Set Horizontal Placement Problem</li>
</ul>
<h4>Graphics Example</h4>
<p><img src="/assets/images/r-learning-notes/r-language-stat-visualization-19.png" alt="Language R Statistical visualization 19"></p>
<h4>base R</h4>
<pre><code class="language-r">## 使用公式表示
data(InsectSprays)
boxplot(count ~ spray, data = InsectSprays,
        col = &quot;lightgray&quot;, horizontal = TRUE, pch = 4,varwidth = TRUE)

## 使用传统的数据表示
x = rnorm(150)
y = rnorm(50, 0.8)
boxplot(list(x, y),names = c(&quot;x&quot;, &quot;y&quot;), horizontal = TRUE,
        col = 2:3, notch = TRUE, varwidth = TRUE)
</code></pre>
<h4>ggplot</h4>
<pre><code class="language-r">data(InsectSprays)
library(ggplot2)
p = ggplot(aes(y = count, x = spray), data = InsectSprays) +
  geom_boxplot(outlier.shape = 4) +
  labs(x = &quot;杀虫剂&quot;, y = &quot;频数&quot;) +
  coord_flip()
print(p)
</code></pre>
<h3>Violin Chart</h3>
<p>Violin Plot is a combination of density curves and box lines because its appearance is sometimes similar to the shape of the violin (especially when showing the density of double-peak data), so we call it the violin. Figure</p>
<h4>Basic Introduction and Code</h4>
<p>Case R doesn't support the drawing of violins, although there are lots of bags like <code>lattice</code> and <code>vioplot</code> The package is drawn, but given the consistency of the graphics, we give it here. <code>ggplot2</code>Example code</p>
<pre><code class="language-r">## ggplot2 绘制三组双峰数据的小提琴图比较
library(ggplot2)
f = function(mu1, mu2) c(rnorm(300, mu1, 0.5), rnorm(200, mu2, 0.5))
x1 = f(0, 2)
x2 = f(2, 3.5)
x3 = f(0.5, 2)
df = reshape(data.frame(A = x1, B = x2, C = x3),
              direction = &quot;long&quot;, varying = c(&quot;A&quot;, &quot;B&quot;, &quot;C&quot;),
              v.name=c(&quot;value&quot;), times=c(&quot;A&quot;, &quot;B&quot;, &quot;C&quot;))
p = ggplot(df, aes(value, time)) +
  geom_violin(fill = &quot;bisque&quot;) +
  geom_boxplot(width = .1) +
  labs(x = &quot;&quot;, y = &quot;&quot;)
print(p)
</code></pre>
<h4>Graphics Example</h4>
<p><img src="/assets/images/r-learning-notes/r-language-stat-visualization-23.png" alt="Statistical Visualization-23"></p>
<h3>Axis of coordinates</h3>
<p>The coordinates must (Rug) is by definition the addition of short-shaves to the coordinates. The function of a truncheon is to indicate the exact location of the variable values on the respective axis, each of which corresponds to one data. The advantage of this is that we can see the distribution of the variable from the distribution of the axis.</p>
<p><strong>The axis of the coordinates must be an attachment to a graphic (low-level graph function of Base R), but it is practical, so it is presented here separately.</strong></p>
<h4>Basic introduction</h4>
<p>The function of the axis in the R is <code>rug()</code></p>
<ul>
<li><code>x</code> For a vector, give the short-shave position.</li>
<li><code>ticksize</code> is the length of the short mustache</li>
<li><code>side</code> The location of the coordinates for the short-shu.</li>
</ul>
<h4>Graphics Example</h4>
<p><img src="/assets/images/r-learning-notes/r-language-stat-visualization-24.png" alt="Statistical visualization-24"></p>
<h4>base R</h4>
<pre><code class="language-r">## 基础作图法绘制带坐标轴须的喷泉喷发时间密度曲线图
data(faithful)
par(mar = c(3, 4, 0.4, 0.1))
plot(density(faithful&#36;eruptions), main = &quot;&quot;)
rug(faithful&#36;eruptions)
</code></pre>
<h4>ggplot</h4>
<pre><code class="language-r">## ggplot2 绘制带坐标轴须的喷泉喷发时间密度曲线图
library(ggplot2)
data(faithful)
p = ggplot(faithful, aes(eruptions)) + geom_line(stat = &quot;density&quot;) +
  geom_rug() + xlim(c(1, 6)) + labs(x = &quot;喷发时间&quot;, y = &quot;分布密度&quot;)
print(p)
</code></pre>
<h3>Belt Chart</h3>
<p>Strip Chart, also called 1-D Scatter Plot, is a scatterchart for one-dimensional data, which is essentially a scatterchart between the data and fixed values (fixed x or fixed y), resulting in a graphic appearance of a band, which is called a band Figure</p>
<p><strong>Although he had the advantage of retaining raw data, he did use less.<code>ggplot2</code>No specific support</strong></p>
<h4>Basic introduction</h4>
<p>The function of the band chart in R is <code>stripchart()</code> The belt chart function is a broad function, which directly accepts data parameters or formulae parameters.</p>
<ul>
<li><code>x</code> For data, usually as a vector</li>
<li><code>method</code>Specify the drawing method, take values <code>overplot</code> It means drawing all the data points on a straight line, whether or not they overlap.</li>
<li><code>jitter</code> It means randomly shattering data on a straight line, so we don't know how many points we have at a particular location.</li>
<li><code>stack</code> It means stacking up overlapping data, and the more data there is, the higher the pile.</li>
</ul>
<h4>Graphics Example</h4>
<p><img src="/assets/images/r-learning-notes/r-language-stat-visualization-25.png" alt="Statistical visualization - 25"></p>
<h4>base R</h4>
<pre><code class="language-r">## 基础作图法绘制各种杀虫剂下昆虫数目的带状图
data(InsectSprays)
layout(matrix(1:2, 2), height = c(1, 1))
par(mar = c(4, 4, 0.2, 0.2))
boxplot(count ~ spray, data = InsectSprays, horizontal = TRUE,
        border = &quot;red&quot;, col = &quot;lightgreen&quot;, at = 1:6 - 0.3,
        xlab = &quot;频数&quot;, ylab = &quot;杀虫剂&quot;)
stripchart(count ~ spray, data = InsectSprays, method = &quot;stack&quot;,
           add = TRUE)
stripchart(count ~ spray, data = InsectSprays, method = &quot;jitter&quot;,
           xlab = &quot;频数&quot;, ylab = &quot;杀虫剂&quot;)
</code></pre>
<h3>Pie Chart</h3>
<p>The pie map is currently used very widely, but according to the findings of statisticians (mainly Cleveland and McGill) and a number of psychologists (see below).<a href="https://bookdown.org/xiangyun/msg/gallery.html#ref-Cleveland85">Cleveland 1985</a>) This statistical display of data in proportion is in fact a bad visualization, so it is clear from the Help Paper on Pie Charts that it is not recommended to use pie charts, but rather to use bar charts<a href="/en/blog/2024/03/15/r-visualization-learning-notes/">Bar Chart</a>Or a dot.<a href="/en/blog/2024/03/15/r-visualization-learning-notes/">Cleveland Point</a>Alternative</p>
<p>Although we do not recommend the use of pie charts, we still offer the means to use them.<code>ggplot2</code> I don't want to offer a pie.</p>
<h4>Basic introduction</h4>
<p>R provides functions <code>pie()</code> Make pies.</p>
<ul>
<li>Parameters <code>x</code> as a numerical vector (assured and equal to 1)</li>
<li><code>labels</code> As Tab</li>
<li>The other parameters are basically for polygons.</li>
</ul>
<p>Specially, there's a three-dimensional picture of the experience that's worse than the flat pie.<code>plotrix</code> Package</p>
<h4>Graphics Example</h4>
<p><img src="/assets/images/r-learning-notes/r-language-stat-visualization-26.png" alt="Statistical visualization-26"></p>
<h4>base R</h4>
<pre><code class="language-r">## 基础作图法绘制馅饼销售饼图、点图和条形图
layout(matrix(c(1, 2, 1, 3), 2)) # 拆分作图区域
par(mar = c(4, 4, 0.2, 0.2))
pie.sales = c(0.12, 0.3, 0.26, 0.16, 0.04, 0.12)
names(pie.sales) = c(&quot;蓝莓&quot;, &quot;樱桃&quot;, &quot;苹果&quot;,
                     &quot;波士顿奶油&quot;, &quot;其它&quot;, &quot;香草奶油&quot;)
pie.col = c(&quot;purple&quot;, &quot;violetred1&quot;, &quot;green3&quot;,
             &quot;cornflowerblue&quot;, &quot;cyan&quot;, &quot;white&quot;)
pie.sales = sort(pie.sales, decreasing = TRUE) # 排序有助于可读性
pie(pie.sales, col = pie.col)
dotchart(pie.sales, xlim = c(0, 0.3))
barplot(pie.sales, col = pie.col, horiz = TRUE,
        names.arg = &quot;&quot;, space = 0.5)
</code></pre>
<h3>QQ Chart</h3>
<p>There are many kinds of tests for statistical distribution, such as KS tests, calibration tests, etc., and from a graphic point of view, we can also use QQQ diagrams (Quantile-Quantile Plots) to check whether data are subject to a certain distribution; it is based on the theoretical distribution and the fraction of the actual distribution.</p>
<h4>Basic introduction</h4>
<p>The function of the QQQ figure in R is <code>qqplot()</code> , since normal distribution is the distribution that we've been checking so often, R also provides a function of drawing normal distribution of QQ Q diagrams <code>qqnorm()</code> , both functions are in the base pack <strong>stats</strong> Package</p>
<ul>
<li><code>qqplot()</code> The test is whether the distribution of the two batches is the same, so it takes two data parameters x and y (theoretical data are generated in functions, actually distributed given)</li>
<li><code>qqnorm()</code> Only one data parameter x (theoretically generated by normal distribution)</li>
</ul>
<h4>Graphics Example</h4>
<p><img src="/assets/images/r-learning-notes/r-language-stat-visualization-27.png" alt="Statistical visualization-27"></p>
<h4>base R</h4>
<pre><code class="language-r">## 基础作图法绘制喷泉间隔时间的正态分布 QQ 图
data(geyser, package = &quot;MASS&quot;)
geyser&#36;waiting_scaled = scale(geyser&#36;waiting)
qqnorm(geyser&#36;waiting_scaled, cex = 0.7, asp = 1, main = &quot;&quot;)
abline(0, 1)
</code></pre>
<h4>ggplot</h4>
<pre><code class="language-r">## ggplot2 绘制喷泉间隔时间的正态分布 QQ 图
library(ggplot2)
library(qqplotr)
library(patchwork)
data(geyser, package = &quot;MASS&quot;)
geyser&#36;waiting_scaled = scale(geyser&#36;waiting)
qq1 = ggplot(data = geyser, mapping = aes(sample = waiting_scaled)) +
  coord_fixed(ratio = 1, xlim = c(-3, 3), ylim = c(-3,3)) +
  geom_abline(aes(intercept = 0, slope = 1), color = &quot;blue&quot;) +
  stat_qq_point() +
  labs(x = &quot;理论分位数&quot;, y = &quot;实际分位数&quot;)
qq2 = ggplot(aes(waiting_scaled), data = geyser) +
  geom_density() +
  stat_function(mapping = aes(x), data = data.frame(x = c(-3, 3)),
                fun = dnorm, n = 101, args = list(mean = 0, sd = 1),
                linetype = 2) +
  labs(x = &quot;间隔时间（标准化）&quot;, y = &quot;分布密度&quot;)
print(qq1 | qq2)
</code></pre>
<h3>Waterfall Chart</h3>
<p>Waterfall Plot is a visualized chart often used to demonstrate trends in data, especially for cumulative changes over different time periods or stages. It usually consists of a series of adjacent bar or column charts, each of which represents the increase or decrease in a given variable. The most common areas of application for waterfall maps include financial analysis, analysis of sales data, demonstration of experimental results, etc.</p>
<p>He's a variant of the traditional strip.</p>
<pre><code class="language-r">library(waterfall)
library(dplyr)

## 创建原始数据
a &lt;- c(&quot;Start&quot;, &quot;Sales Increase&quot;, &quot;Cost Increase&quot;, &quot;Tax Decrease&quot;, &quot;End&quot;)
data &lt;- data.frame(
  Stage = factor(a,ordered = T,levels = a),
  Change = c(100, 50, -30, 10, 0)  # End值设置为0，确保它出现在最后
)

## 使用 waterfallchart 绘制瀑布图
waterfallchart(data  = data,
               Change~Stage,
               main = &quot;Waterfall Chart (Staircase Style)&quot;,
               ylab = &quot;Cumulative Value&quot;,
               xlab = &quot;Stage&quot;)
</code></pre>
<p>The format is as follows:
<img src="/assets/images/r-learning-notes/r-stat-visualization-18.png" alt="R Statistical visualization-18"></p>
<h2>Double Variable Chart</h2>
<p>This chapter presents statistical graphics reflecting the random relationship between the two variables;</p>
<p><strong>One of the most classic is a scatterchart describing the continuity between two consecutive variables.</strong></p>
<p><strong>If one of the two variables is a qualitative one, it is consistent with our thinking of comparing the graphics of multiple single variables, such as the bar chart, the box line, the point figure in the previous chapter.</strong></p>
<p>As for the issue of matrix graphics and multiple variables larger than two, we'll look at it separately later. <a href="/en/blog/2024/03/15/r-visualization-learning-notes/">Multivariate Chart</a> <a href="/en/blog/2024/03/15/r-visualization-learning-notes/">Matrix Graphics</a></p>
<h3>Scatter Chart</h3>
<p>The scatterchart is usually used to show the relationship between the two variables, which may be linear or non-linear, and the vertical and vertical coordinates of each point in the map correspond to the observations of each of the two variables, so that the trend reflected by the breakpoint is the relationship between the two variables.</p>
<p>We don't have any extra information.</p>
<h4>Graphics Example</h4>
<p><img src="/assets/images/r-learning-notes/r-language-stat-visualization-28.png" alt="Statistical visualization - 28">
The right-hand map adjusts the transparency design to create a highly superheavy circle that is useful for high density situations, and, of course, we also have some other means of dealing with the problem of high density fragmentation overlaps, for example. <strong>graphics</strong> Medium <code>smoothScatter</code> The nuclear density in it is smooth. It's a very worthwhile solution, reference.<a href="/en/blog/2024/03/15/r-visualization-learning-notes/">Smooth Scatter Point Chart</a></p>
<h4>base R</h4>
<pre><code class="language-r">## 基础作图法绘制半透明散点图中
data(BinormCircle, package = &quot;MSG&quot;)
par(mfrow = c(1, 2), pch = 20, ann = FALSE, mar = rep(.05, 4))
plot(BinormCircle, col = rgb(1, 0, 0), axes = FALSE)
box()
plot(BinormCircle, col = rgb(1, 0, 0, alpha = .01), axes = FALSE)
box()
</code></pre>
<h4>ggplot</h4>
<pre><code class="language-r">## ggplot2 绘制半透明散点图中
data(BinormCircle, package = &quot;MSG&quot;)
library(ggplot2)
library(patchwork)
p = ggplot(BinormCircle, aes(V1, V2)) +
  theme(axis.ticks = element_blank(), axis.text = element_blank(),
        axis.title = element_blank())
p1 = p + geom_point(color = rgb(1,0,0)) + theme_void()
p2 = p + geom_point(color = rgb(1,0,0), alpha = 0.01) + theme_void()
print(p1 | p2)
</code></pre>
<h3>A one-digit function curve</h3>
<p>We're going to tell you how to draw a one-dimensional curve, and only one-dimensional functions can be shown better on the two-dimensional plane.</p>
<p>This curve is not too deep for data analysis, so...<code>ggplot2</code>No method of drawing available</p>
<h4>Basic introduction</h4>
<p>The function of the function curve in R is <code>curve()</code> R provides a function specifically designed to save us from the use of lower layers for mapping functions (e.g. <code>lines()</code>The energy and the time.</p>
<ul>
<li>Parameters <code>expr</code> (a) for a single function or the name of that function;</li>
<li><code>from</code> and <code>to</code> The starting point and end point of the curve are defined separately;</li>
<li><code>n</code> Determines how many sub-areas the defined domain is divided to calculate the function and connect the curve, <code>n</code> The larger the value, the smoother the curve.</li>
</ul>
<h4>Graphics Example</h4>
<p><img src="/assets/images/r-learning-notes/r-language-stat-visualization-29.png" alt="Statistical visualization - 29"></p>
<h4>base R</h4>
<pre><code class="language-r">## 基础作图法绘制一元函数曲线图
par(par(mar = c(4.5, 4, 0.2, 0.2)), mfrow = c(2, 1))
chippy = function(x) sin(cos(x) * exp(-x / 2))
curve(chippy, -8, 7, n = 2008, xlab = &quot;x&quot;, ylab = &quot;chippy(x)&quot;)
curve(sin(x) / x, from = -20, to = 20, n = 200,
      xlab = &quot;t&quot;, ylab = expression(phi*X(t)))
</code></pre>
<h3>Sunflower Scatter</h3>
<p>The SunFlower Scatter Plot is a special scattering tool for overcoming the overlap of data points in the scatter. It uses the method of using the number of petals of a `sunflower' to indicate the number of overlapping data where there are overlaps, so that we can easily see where the data in the scattered maps overlap and know the exact number of overlaps.</p>
<p>Scattered sunflower maps are useful when data are particularly dense or the type of data is classified, since in both cases it is easy to generate duplicate data points</p>
<p><code>ggplot2</code>It doesn't provide the method of drawing.</p>
<h4>Basic introduction</h4>
<p>The function of the Sunflower graph in R is <code>sunflowerplot()</code></p>
<ul>
<li><code>x</code> and <code>y</code> Two variables, respectively, for the dispersion map;</li>
<li><code>number</code> Number of data frequencies to be given manually, i.e., the number of petals in the chart, if this parameter is not specified, R automatically from <code>x</code> and <code>y</code> Calculation;</li>
<li> <code>rotate</code> Whether to rotate the angle of sunflower at random;</li>
<li><code>pch</code> Type of point given in the dispersed map;</li>
<li><code>cex</code> (a) The number of times the point of a scattered map is scaled;</li>
<li><code>cex.fact</code>Multiplier reduction of the center point of the directed sunflower. The real scaling factor is <code>cex/cex.fact</code> ；</li>
</ul>
<h4>Graphics Example</h4>
<p><img src="/assets/images/r-learning-notes/r-language-stat-visualization-30.png" alt="Language R Statistical visualization - 30"></p>
<h4>base R</h4>
<pre><code class="language-r">## 绘制鸢尾花花瓣长和宽的向日葵散点图
data(iris)
par(mar = c(4, 4, 0.2, 0.2))
sunflowerplot(iris[, 3:4], col = &quot;gold&quot;, seg.col = &quot;gold&quot;,
              xlab = &quot;花瓣长度&quot;, ylab = &quot;花瓣宽度&quot;)
</code></pre>
<h3>Smooth Scatter Point Chart</h3>
<p>The smooth-scatter map is based on a disassembly map, but it is not based directly on a 2-D nuclear density estimate, but rather on a specific colour indicating the density value of a position at a shallow level, the deeper the default colour, the greater the 2-D density value, the more dense the data point.</p>
<p>The relationship between the two variables can still be seen in the map, as the smooth break-point map roughly retains the location of the original data point, which is similar to the normal break-point map. The further advantage of smooth-spreading is that it also shows the density of the two-dimensional variable, from which we may be able to observe local cluster phenomena (dark mass).</p>
<p>Smooth dispersing dots look like peace.<a href="/en/blog/2024/03/15/r-visualization-learning-notes/">Scatter Chart</a>The right figure is similar, but the former contains more mathematical statistical background. However, we do not have to pursue mathematical theory at all, and why is it not a density estimate that is reflected in the nature of transparency?</p>
<h4>Basic introduction</h4>
<p>The function of a smooth dispersing pointchart in R is <code>smoothScatter()</code></p>
<ul>
<li> <code>x</code> and <code>y</code> is a two-value vector or if not provided <code>y</code> can provide a two-column matrix/data box, etc. Here. <code>x</code> ；</li>
<li><code>nbin</code> (a) the number of grids assigned to the vertical and vertical coordinates, which may be an integer vector of 1 or 2 length;</li>
<li><code>bandwidth</code> The bandwidth used to calculate nuclear density estimates;</li>
<li><code>colramp</code> The default generation of colour vectors from white to blue gradients for the function of generating colour vectors;</li>
<li><code>nrpoints</code> For the number of points that need to be drawn, because the smooth dispersed dots are not meant to draw the dots but the colours, but sometimes the density estimates are very low in some parts of the picture, so the corresponding colours are very shallow, making it difficult for the reader to detect the presence of data points in those places, and it would be useful to draw them directly.</li>
</ul>
<h4>Graphics Example</h4>
<p><img src="/assets/images/r-learning-notes/r-language-stat-visualization-31.png" alt="Statistical visualization - 31"></p>
<h4>base R</h4>
<pre><code class="language-r">## 基础作图法绘制 BinormCircle 数据的平滑散点图
data(&quot;BinormCircle&quot;, package = &quot;MSG&quot;)
par(mar = c(4, 4, 0.3, 0.1))
smoothScatter(BinormCircle)
</code></pre>
<h4>ggpot</h4>
<pre><code class="language-r">## ggplot2 绘制 BinormCircle 数据的平滑散点图
data(&quot;BinormCircle&quot;, package = &quot;MSG&quot;)
library(ggplot2)
library(ggpointdensity)
p = ggplot(data = BinormCircle, aes(x = V1, y = V2)) +
  geom_pointdensity(adjust = 0.1) +
  scale_color_gradient(low=&quot;lightblue&quot;, high=&quot;darkblue&quot;) +
  theme(legend.position = &quot;&quot;)
print(p)
</code></pre>
<h3>Wind Rose Map</h3>
<p>The wind roses are a very special kind of graphics, including wind to roses and wind speed roses; it's just that we use the latter to show wind speed and direction.</p>
<p><strong>It's only for wind speed and direction, it's for the weather, and it's only because it's used occasionally.</strong></p>
<p>The wind roses are essentially stacked strips drawn in a polar system, usually divided into 16 sectors; each sector is the frequency of wind speeds in the direction of the wind.<code>beside = F</code> Bar Drawing</p>
<h4>Basic introduction</h4>
<p>In R <code>openair</code> Package provides function <code>windRose</code> The drawing is done using the following:</p>
<ul>
<li><code>mydata</code> Data frame for recording wind direction and speed</li>
<li><code>ws</code> Wind Quick Row Name</li>
<li><code>wd</code> Windward column name</li>
<li><code>angel</code> Wind-direction angle
<strong>More needs we can look at.</strong></li>
</ul>
<h4>Graphics Example</h4>
<p><img src="/assets/images/r-learning-notes/r-language-stat-visualization-32.png" alt="Statistical Visualization-32"></p>
<h4>Openair Code Achieved</h4>
<pre><code class="language-r">## 绘制风玫瑰图
library(openair)
windRose(mydata)
windRose(mydata = mydata, ws = &quot;ws&quot;, wd = &quot;wd&quot;,
         key.position = &quot;right&quot;, paddle = FALSE, seg = 0.9,
         angle = 22.5, ws.int = 0.5,
         cex = 3, breaks = c(seq(0,5,1), 21))
</code></pre>
<h3>Survival Function Chart</h3>
<p>In many medical studies, our main concern is the timing of a patient's event, such as death, relapse. In fact, the area of “lifetime” as the object of research is not limited to medicine, for example, in the area of finance, where we may need to know when credit risk occurs for credit card holders. This type of data is generally referred to as survival data, which is usually characterized by the omission of the object from our observations for some reason.</p>
<p><strong>Such research, referred to separately as survival analysis, is a very important category of issues, with specific realizations.</strong></p>
<h4>Basic introduction</h4>
<p>The graphic object to be presented in this section is primarily a survival function, defined as the individual ' s survival beyond time. &#36;t&#36; Probability
&#36;S(t)=P(T)&gt;t); t\geq&#36;0
For the existence of missing survival data &#36;(t_i,\delta_i),i=1,\cdots,n&#36; of which &#36;t_i&#36; For recording time,&#36;\delta_i=0&#36; An estimated Kaplan-Meier for the survival function is:</p>
<p>&#36;&#36;
\left.\hat{S}(t)=\left{\begin{array}{ll}\prod_{i\colon t_{(i)}\leq t}(\frac{n-i}{n-i+1})^{\delta_{(i)&#125;&#125;,&amp;\\text{t\leqt (n)};\0&amp;\\text{if}\delta (n)}=1,\text{undefined}&amp;\\text{(n)=,\end{array}\right.\right.\text{t){&gt;t_{(n)}.
&#36;&#36;</p>
<p><strong>survival</strong> The package provides the method of calculating and estimating the survival function. The function is <code>survfit()</code>, it returns a survfit class object; And... <strong>survival</strong> The package expands the generic function <code>plot()</code>to have sub-functions <code>plot.survfit()</code>, so after estimating the survival function, we can call directly. <code>plot()</code> Generate survival function maps</p>
<h4>Graphics Example</h4>
<p><img src="/assets/images/r-learning-notes/r-language-stat-visualization-33.png" alt="Statistical visualization-33"></p>
<h4>Survival code achieved</h4>
<pre><code class="language-r">data(&quot;leukemia&quot;, package = &quot;survival&quot;)
library(survival)
leukemia.surv = survfit(Surv(time, status) ~ x, data = aml)
plot(leukemia.surv, lty = 1:2, xlab = &quot;time&quot;)
legend(&quot;topright&quot;, c(&quot;Maintenance&quot;, &quot;No Maintenance&quot;),
       lty = 1:2, bty = &quot;n&quot;)
</code></pre>
<h3>Conditional Density Chart</h3>
<p>Conditional Density Chart (Conditions Density Plot), which by definition shows the condition density of a variable, a classification variable&#36;Y&#36;Relative to a continuous variable&#36;X&#36; Conditional density &#36;P(Y|X)&#36;I don't know. Assumptions &#36;Y&#36; The value to be taken is &#36;1,2,\cdots,k&#36;, then the condition density map will be by&#36;X&#36; The extraction values are displayed in descending directions from small to large. Out&#36;Y=i\left(i=1,2,\cdots,k\right)&#36; Rate of probability distribution of conditions</p>
<h4>Basic introduction</h4>
<p>The function of the condition density diagram in R is <code>cdplot()</code>, it's based mainly on density functions <code>density()</code> Completion of condition density calculation</p>
<ul>
<li> <code>x</code> For the conditional variable X, it's a numerical vector, <code>y</code> is a factor vector, the discrete variable Y; he is also a generic function</li>
<li><code>plot</code> Determines whether graphics are made for logical values (or only calculated without drawing)</li>
</ul>
<p>We could use it to study what was supposed to be used.<code>logit</code> The question of re-entry is a reference for a supervisor.</p>
<h4>Graphics Example</h4>
<p><img src="/assets/images/r-learning-notes/r-language-stat-visualization-34.png" alt="Language R Statistical visualization 34"></p>
<h4>base R</h4>
<pre><code class="language-r">## 基础作图法绘制航天飞机 O 型环在不同温度下失效的条件密度图
data(orings, package = &quot;DAAG&quot;)
orings&#36;Fail = factor(apply(orings[, -1], 1, function(x) all(x == 0)),
                     labels = c(&quot;yes&quot;, &quot;no&quot;))
cdplot(Fail ~ Temperature, data = orings, col = c(&quot;lightblue&quot;, &quot;red&quot;))
points(orings&#36;Temperature, c(0.75, 0.25)[as.integer(orings&#36;Fail)],
       col = &quot;blue&quot;, bg = &quot;yellow&quot;, pch = 21)
</code></pre>
<h4>ggplot</h4>
<pre><code class="language-r">## ggplot2 绘制航天飞机 O 型环在不同温度下失效的条件密度图
library(ggplot2)
library(DAAG)
data(orings, package = &quot;DAAG&quot;)
orings&#36;Fail = factor(apply(orings[, -1], 1, function(x) all(x == 0)),
                     labels = c(&quot;yes&quot;, &quot;no&quot;))
p = ggplot(orings,
           aes(Temperature, ..count.., fill = Fail)) +
  geom_density(position = &quot;fill&quot;) +
  geom_point(aes(Temperature, c(0.75, 0.25)[as.integer(Fail)])) +
  xlab(&quot;温度&quot;) +
  scale_y_continuous(&quot;失效&quot;, breaks = c(0.25, 0.75),
                     labels = c(&quot;否&quot;, &quot;是&quot;)) +
  theme(legend.position = &quot;&quot;)
print(p)
</code></pre>
<h3>2-D box line</h3>
<p>We've got a regular box chart.<a href="/en/blog/2024/03/15/r-visualization-learning-notes/">Line Chart</a>, i.e., the fractions of one-dimensional data are expressed in a box line, and in a two-dimensional scenario we can draw a two-dimensional chart of the box with similar ideas. Two-dimensional graphs, also known as Bag Plot</p>
<p>The approach of the 2-dimensional graph is to move out of the centre of the data and gradually wrap up points in the scattered maps in a condensed polygon until they reach half the data points, when the condensation is equivalent to boxes in the ordinary graphs and then outsourced to all data points. The basic composition of the two-dimensional diagram is a centre and two polygons, which provide a rough description of the two-dimensional distribution of data.</p>
<p>A 2-D box line also needs a special kit to do it.</p>
<h4>Basic introduction</h4>
<p>Center R <strong>aplpack</strong> The package provides a function <code>bagplot()</code> It can be used to draw two-dimensional graphs.</p>
<ul>
<li><code>x</code> and <code>y</code> A data vector on a vertical axis, or a matrix or data frame for two columns, directly;</li>
<li><code>factor</code> Similar <code>boxplot()</code> Medium <code>range</code> Parameters, which define the distance from the group point and the larger the value, the smaller the number of points away (the distance from the centre of the data point);</li>
<li><code>approx.limit</code> Sample quantities of large data are defined and randomly extracted if sample quantities of original data exceed this number <code>approx.limit</code> A data point is used for the calculation of a two-dimensional chart;</li>
<li><code>dkmethod</code> Take values 1 or 2, determine which method to calculate the scope of the bag, take values 2 more precisely</li>
</ul>
<h4>Special</h4>
<pre><code class="language-r">msg(&quot;5.9) #出现报错
</code></pre>
<h2>Multivariate Chart</h2>
<p>Start with this chapter with three and more variables; but we leave a special graphic: the matrix graphic to the next chapter.</p>
<h3>Spreadchart Matrix</h3>
<p>Scatterplot Matrices is a high-dimensional extension of the scatterchart, which basically consists of a normal breakchart, and which simply sets out the two-diggles of multiple variables in a matrix form, constituting the so-called breakchart matrix.</p>
<p>The scatterchart matrix has, to some extent, overcome the difficulty of displaying high-dimensional data on the plane, which is very useful in looking at the two relationships between variables.</p>
<h4>Basic introduction</h4>
<p>The function of the stand-down matrix for R is <code>pairs()</code></p>
<ul>
<li> <code>x</code> It's a matrix or data box that contains the variables that are to be used to make a scatter map.</li>
<li><code>panel</code> Parameters give a function to draw a scatterchart, which is applied to each cell;</li>
<li>Sometimes we don't need a uniform scattering function. <code>lower.panel</code>  and <code>upper.panel</code> to specify graph functions in the upper and lower triangles, respectively</li>
</ul>
<p><strong>car</strong>Package<code>scatterplotMatrix()</code>function can also generate a scatterchart matrix with a higher degree of customisation</p>
<h4>Graphics Example</h4>
<p><img src="/assets/images/r-learning-notes/r-language-stat-visualization-35.png" alt="Language R Statistical visualization-35"></p>
<h4>base R</h4>
<pre><code class="language-r">## 基础函数作图法绘制鸢尾花数据的散点图矩阵
## 观察如何使用 hist() 做计算并用 rect() 画图
data(&quot;iris&quot;)
panel.hist = function(x, ...) {
  usr = par(&quot;usr&quot;)
  on.exit(par(usr))
  par(usr = c(usr[1:2], 0, 1.5))
  h = hist(x, plot = FALSE)
  breaks = h&#36;breaks
  nB = length(breaks)
  y = h&#36;counts / max(h&#36;counts)
  rect(breaks[-nB], 0, breaks[-1], y, col = &quot;beige&quot;)
}
idx = as.integer(iris[[&quot;Species&quot;]])
names(iris)[1:4] = c(&quot;花萼长度&quot;, &quot;花萼宽度&quot;, &quot;花瓣长度&quot;, &quot;花瓣宽度&quot;)
pairs(iris[1:4],
      upper.panel = function(x, y, ...)
        points(x, y, pch = c(17, 16, 6)[idx], col = idx),
      pch = 20, oma = c(2, 2, 2, 2),
      lower.panel = panel.smooth, diag.panel = panel.hist
)
</code></pre>
<h4>ggplot</h4>
<p>We add new packages to achieve better results.</p>
<pre><code class="language-r">## ggplot2 绘制鸢尾花数据的散点图矩阵
library(ggplot2)
library(GGally)
data(&quot;iris&quot;)
names(iris) = c(&quot;花萼长度&quot;, &quot;花萼宽度&quot;, &quot;花瓣长度&quot;, &quot;花瓣宽度&quot;, &quot;种类&quot;)
p = ggpairs(iris, aes_string(colour=&quot;种类&quot;, alpha=0.5))
print(p)
</code></pre>
<h3>Conditional Partition Chart</h3>
<p>The idea of the partitioning of conditions (Conditioning Plot) stems from the distribution of conditions in statistics, i.e. the distribution of variables of interest to us after a given variable (or variables) has been given. This “distribution” refers mainly to the relationship between the two variables in the condition partition figure, which is usually expressed as a scattered figure.</p>
<p>The partitioning of conditions can be seen as a further in-depth discovery of the dispersion map, which can be divided into all data with one or two condition variables, which, on the edge of the graphic, mark the range of values of the variable in a grey rectangular bar, each of which corresponds to a scattering chart (which should, strictly speaking, be referred to as the “conditional break-up chart”), which is the basic approach.</p>
<h4>Basic introduction</h4>
<p>The function of the condition partition in R is <code>coplot()</code></p>
<ul>
<li>Parameters <code>formula</code> A formula in the form of <code>y ~ x | a</code>(a condition variable) or <code>y ~ x | a * b</code>(two condition variables), "<code>|</code>is followed by the condition variable;</li>
<li><code>data</code> For data, it contains <code>x</code> 、 <code>y</code> 、 <code>a</code> and <code>b</code> Variables;</li>
<li><code>given.values</code> Specify the range of values for the condition variable;</li>
<li><code>panel</code> Parameters are the key parameters of this function, which determines the pattern of each scattered map, and the default is only a dot</li>
<li><code>number</code> and <code>overlap</code> Pass. <code>co.intervals()</code> The function is used to calculate the number of partitions that divide the continuous variable, which sets the overlap ratio between the compartments.
<strong>The continuous variable divide refers to how we divide the condition variable, because the scatterchart is limited and the number of compartments needs to be compared to the number of maps Match</strong></li>
</ul>
<h4>Graphic Examples</h4>
<p><img src="/assets/images/r-learning-notes/r-language-stat-visualization-36.png" alt="Language R Statistical visualization - 36"></p>
<h4>base R</h4>
<pre><code class="language-r">## 基础作图法绘制给定震源深图的地震经纬度条件分割图
data(quakes)
library(maps)
par(mar = rep(0, 4), mgp = c(2, .5, 0))
coplot(lat ~ long | depth, data = quakes, number = 4,
       xlab = c(&quot;经度&quot;, &quot;深度&quot;), ylab = &quot;纬度&quot;,
       ylim = c(-45, -10.72), panel = function(x, y, ...) {
         map(&quot;world2&quot;,
             regions = c(&quot;New Zealand&quot;, &quot;Fiji&quot;),
             add = TRUE, lwd = 0.1, fill = TRUE, col = &quot;lightgray&quot;
         )
         text(180, -13, &quot;Fiji&quot;, adj = 1)
         text(170, -35, &quot;NZ&quot;)
         points(x, y, col = rgb(0.2, 0.2, 0.2, .5))
       }
)
</code></pre>
<h3>Symbolic Chart</h3>
<p>The symbol chart is a graphic tool for displaying high-dimensional data with symbols, and its main idea is to give high-dimensional values to the character of the symbol in the graphic.</p>
<p>The symbol chart is in essence a highly customised scattered map more than the previous settings<code>panel</code>More defined parameters</p>
<p>For example, using a rectangle as the basic symbol of the scattering map, we can use its width to represent two variables, so that at least four variables can be placed in a graphic, and we can achieve a high-dimensional presentation.</p>
<p>It takes us to think carefully.</p>
<h4>Basic introduction</h4>
<p>The symbol chart function in R is <code>symbols()</code>, it provides six basic symbols: round, square, rectangular, star, thermometer and box line charts, specified by respective parameters</p>
<ul>
<li><code>circles</code> Circle: a numerical vector, radius of given circle</li>
<li><code>squares</code> Square: a numerical vector, to the side of a square Long</li>
<li><code>rectangles</code> rectangular: a matrix, with two columns, each giving the width of the rectangular and High</li>
<li><code>stars</code> Star: a matrix, columns 3 and a radar-like map of the length of the ray from the center of the star to each direction (stellar segment strictly speaking) (stellar shape is not intuitive in the symbol chart, direct astrograph is recommended)</li>
<li><code>thermometers</code> Thermometers: a matrix with 3 or 4 columns, the width and height of the thermometers for each of the first two columns; if the matrix is 3 columns, then the “temperature” height of the third column included in the thermometer should be smaller than 1, otherwise the temperature filling would exceed the thermometer range; if the matrix was 4 columns, the temperature would be filled at the ratio of the third column to the fourth column; similarly, the ratio of the two columns would need to be less than 1</li>
<li><code>boxplots</code> A matrix with a number of 5 columns, the width and height of each of the first two columns, the length of the third and fourth columns, respectively, of two lines (downline and upline), similar to the thermometer in the fifth column, and the proportion of the median marking line within the given box chart to the height of the inner section of the box, so the data for this column is also required &#36;[0,1]&#36; Range; here's just the name of the box chart, which has nothing to do with the actual box chart.</li>
</ul>
<h4>Graphics Example</h4>
<p><img src="/assets/images/r-learning-notes/r-language-stat-visualization-37.png" alt="Statistical Visualization - 37"></p>
<p>It is based on a map of equal heights, using the variables of life expectancy and the number of highly educated people to calculate the two-dimensional density and draw the same line, thus completing the bottom map;
We then add the thermometer symbol to the map using the values of both the life expectancy of the population and the number of highly educated persons. The thermometer width represents the growth rate, the number of the population is high and the temperature represents the proportion of the urban population;
Then we'll use it. <code>text()</code> function.</p>
<p>The five demographic characteristics of the autonomous regions of the provinces and municipalities of the country, as expressed by these graphic elements, are evident, for example, in the height of thermometers, which show that the three population departments, Guangdong, Shandong and Henan (the corresponding small areas of the population, such as Tibet, Qinghai and Ningxia, etc.), show a very high rate of natural population growth in the autonomous regions of Tibet, Qinghai, Ningxia and Xinjiang (while the rate of growth in directly administered municipalities such as Beijing, Shanghai and Tianjin is very low), and that, according to temperature indicators, the proportion of the population living in the cities directly under the Beijing, Shanghai and Tianjin municipalities is much higher than in other regions;</p>
<p>Based on the overall scattered figure, the average life expectancy of the population is relatively positive in relation to the number of persons with higher education. An average life expectancy of the population and the distribution of the number of highly educated persons respectively are to be depicted in the diagrams and coordinates. So we've done the task of describing five-dimensional variables on the plane.</p>
<h4>base R</h4>
<pre><code class="language-r">## 以下是生成图形的代码：
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
21340), .Dim = c(31L, 5L), .Dimnames = list(c(&quot;北京&quot;, &quot;天津&quot;,
&quot;河北&quot;, &quot;山西&quot;, &quot;内蒙古&quot;, &quot;辽宁&quot;, &quot;吉林&quot;, &quot;黑龙江&quot;, &quot;上海&quot;, &quot;江苏&quot;,
&quot;浙江&quot;, &quot;安徽&quot;, &quot;福建&quot;, &quot;江西&quot;, &quot;山东&quot;, &quot;河南&quot;, &quot;湖北&quot;, &quot;湖南&quot;,
&quot;广东&quot;, &quot;广西&quot;, &quot;海南&quot;, &quot;重庆&quot;, &quot;四川&quot;, &quot;贵州&quot;, &quot;云南&quot;, &quot;西藏&quot;,
&quot;陕西&quot;, &quot;甘肃&quot;, &quot;青海&quot;, &quot;宁夏&quot;, &quot;新疆&quot;), c(&quot;增长率&quot;, &quot;总人口&quot;,
&quot;城镇人口比重&quot;, &quot;预期寿命&quot;, &quot;高学历人数&quot;)), adj = structure(c(-0.5,
-0.5, 0.5, 0.5, 1.3, -0.4, -0.6, -0.5, -0.5, -0.6, -0.6, 0.6,
0.5, 0.5, 0.5, 0.5, 0.5, 1.8, 0.5, 1.7, 0.5, -0.6, -0.6, 0.5,
0.5, 0.5, 1.7, -0.7, 0.5, 0.5, 0.5, 0.5, 0.5, -0.5, 0, -0.5,
0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -0.3, 2.1, -0.3, -0.5, -0.5, -1.6,
0.5, -0.7, 1.3, -0.7, 0.5, 0.5, 2.4, -0.3, -0.6, 0.5, 0.5, -0.8,
-0.7, -0.8), .Dim = c(31L, 2L), .Dimnames = list(c(&quot;北京&quot;, &quot;天津&quot;,
&quot;河北&quot;, &quot;山西&quot;, &quot;内蒙古&quot;, &quot;辽宁&quot;, &quot;吉林&quot;, &quot;黑龙江&quot;, &quot;上海&quot;, &quot;江苏&quot;,
&quot;浙江&quot;, &quot;安徽&quot;, &quot;福建&quot;, &quot;江西&quot;, &quot;山东&quot;, &quot;河南&quot;, &quot;湖北&quot;, &quot;湖南&quot;,
&quot;广东&quot;, &quot;广西&quot;, &quot;海南&quot;, &quot;重庆&quot;, &quot;四川&quot;, &quot;贵州&quot;, &quot;云南&quot;, &quot;西藏&quot;,
&quot;陕西&quot;, &quot;甘肃&quot;, &quot;青海&quot;, &quot;宁夏&quot;, &quot;新疆&quot;), c(&quot;horizontal&quot;, &quot;vertical&quot;
))))
library(KernSmooth)
x = ChinaPop
x[, 1:2] = apply(x[, 1:2], 2, function(z) 20 * (z -
    min(z)) / (max(z) - min(z)) + 5)
symbols(x[, 4], x[, 5],
  thermometers = x[, 1:3], fg = &quot;gray40&quot;,
  inches = 0.5, xlab = &quot;\u4EBA\u5747\u9884\u671F\u5BFF\u547D&quot;, ylab = &quot;\u9AD8\u5B66\u5386\u8005\u4EBA\u6570&quot;
)
est = bkde2D(x[, 4:5], apply(x[, 4:5], 2, dpik))
contour(est&#36;x1, est&#36;x2, est&#36;fhat, add = TRUE, lty = &quot;12&quot;)
for (i in 1:nrow(x)) {
  text(x[i, 4], x[i, 5], rownames(x)[i],
    cex = 0.75, adj = attr(x, &quot;adj&quot;)[i, ]
  )
}
rug(x[, 4], 0.02, side = 3, col = &quot;gray40&quot;)
rug(x[, 5], 0.02, side = 4, col = &quot;gray40&quot;)
boxplot(x[, 4],
  horizontal = TRUE, pars = list(
    boxwex = 7000,
    staplewex = 0.8, outwex = 0.8
  ), at = -6000, add = TRUE, notch = TRUE, col = &quot;skyblue&quot;,
  xaxt = &quot;n&quot;
)
boxplot(x[, 5],
  at = 63, pars = list(
    boxwex = 1.4,
    staplewex = 0.8, outwex = 0.8
  ), add = TRUE, notch = TRUE, col = &quot;skyblue&quot;,
  yaxt = &quot;n&quot;
)
text(67, 60000, &quot;2005&quot;, cex = 3.5, col = &quot;gray&quot;)
</code></pre>
<h3>Star Chart</h3>
<p>Star Plot, Spider Plot and Radar Plot are essentially graphics, all of which represent the size of the variable by the length of the line from the center, and the difference between these three graphic names is that the star chart is used to show many multivariant individuals, each individual is independent of each other, so the whole picture looks like a lot of stars. Star</p>
<p>Cobweb and radar maps place multiple variable individuals on the same graphic, which looks like the shape of a spider web or radar, so the overlapping graphic is called a spider map or radar map. In short, there are several centres in the asterisk map, and only one centre in the spider and radar maps.</p>
<p> <strong>Base R is enough.</strong></p>
<h4>Basic introduction</h4>
<p>The function of the star chart in R is <code>stars()</code></p>
<ul>
<li>Parameters <code>x</code> For a multi-dimensional data matrix or data frame, each line will generate a star shape;</li>
<li><code>full</code> The use of round (or half circle) is determined for logical values;</li>
<li><code>scale</code> Whether or not to standardize the data into a zone &#36;[0,1]&#36; Inside;</li>
<li><code>radius</code> Whether or not to draw a radius;</li>
<li><code>labels</code> For the name of each individual, the default is the line name of the data;</li>
<li><code>locations</code> In a two-column rectangle, give the position of each star, and by default place it on a rule rectangle grid. If the parameter is provided with a vector of 2 length, all stars will be placed on the coordinates, thus forming a spider map or radar map;</li>
</ul>
<h4>Graphics Example</h4>
<p><img src="/assets/images/r-learning-notes/r-language-stat-visualization-38.png" alt="Language R Statistical visualization - 38"></p>
<h4>base R</h4>
<pre><code class="language-r">## 绘制汽车数据的星状图
## 预设调色板，stars() 默认用整数来表示颜色
palette(rainbow(12, s = 0.6, v = 0.75))
stars(mtcars[, 1:7], len = 0.8, key.loc = c(14, 1.5), ncol = 7,
      main = &quot;&quot;, draw.segments = TRUE)
palette(&quot;default&quot;) # 恢复默认调色板
</code></pre>
<h3>Face map</h3>
<p>The Facebook map, presented by Chernoff, presents multiple data in a very interesting way: There are many features to a person's face, such as eye size, eyebrow radians, face width and nostrils, which can be measured in numerical sizes, so we can also reciprocate a set of values to these faces.</p>
<p><strong>TeachingDemos</strong> The package provides two Facebook functions <code>faces()</code> and <code>faces2()</code> , the two functions reflect different facial features, each of which has merit, e.g. <code>faces()</code> You can draw hair and ears, but... <code>faces2()</code> You can draw more variables. We only introduce the latter.</p>
<p><strong>Among the many statistical graphics, Facebook can be one of the most humorous, and readers may try to use it at some easy times or when the audience is not focused, perhaps to give the audience a sense of light and a proactive reading of the data in the map.</strong></p>
<h4>Basic introduction</h4>
<p><code>faces2()</code> The function is what we need to study most.</p>
<ul>
<li><code>mat</code> It is the main parameter, it is a data matrix, each line corresponds to a face and the characteristics of each part of the face correspond to the column in the matrix;</li>
<li><code>which</code> It is also an important parameter used to specify which facial characteristics each column in the data matrix corresponds to, and it is an integer vector with values ranging from 1 to 18 for each element of the vector</li>
</ul>
<h4>Graphics Example</h4>
<p><img src="/assets/images/r-learning-notes/r-language-stat-visualization-39.png" alt="Statistical Visualization - 39"></p>
<h4>face2</h4>
<pre><code class="language-r">## 绘制部分汽车数据的脸谱图
library(TeachingDemos)
faces2(mtcars[, c(&quot;hp&quot;, &quot;disp&quot;, &quot;mpg&quot;, &quot;qsec&quot;, &quot;wt&quot;)],
       which = c(14, 9, 11, 6, 5))
</code></pre>
<h3>Three-dollar figure</h3>
<p>The three-dollar map is a very special statistical graphic that can only process data of 3 columns and 1 or 100 for each row; they are often chemical composition data, such as the percentage of three components of a mixture. That's right.</p>
<p>For the three-dollar map, we need a package. <code>vcd</code> Perform drawing</p>
<p>Specific Function Introduction Available Help</p>
<h4>Graphics Example</h4>
<p><img src="/assets/images/r-learning-notes/r-language-stat-visualization-40.png" alt="Statistical visualization - 40"></p>
<h4>vcd</h4>
<pre><code class="language-r">## 绘制土壤样本三元图
data(murcia, package = &quot;MSG&quot;)
library(vcd)
ternaryplot(murcia[, 2:4], main = &quot;&quot;,
            dimnames = c(&quot;砂粒&quot;, &quot;粉粒&quot;, &quot;黏粒&quot;),
            col = MSG::vec2col(murcia&#36;site), cex = .5)
</code></pre>
<h3>Marcello.</h3>
<p>Mosaic Plots is a tool for displaying multiple statistical analysis of the multi-dimensional tables: the data of the arrays that do not limit the dimensions of the arrays, unlike the association and the quadratic charts, which are limited to low-dimensional arrays.</p>
<p>Masektu is expressed as a rectangle in proportion to the frequency, and the whole picture looks like a few Marseks on the plane. The statistical theory behind Masektu is a linear model (log-linear model)</p>
<h4>Basic introduction</h4>
<p>The function of R-Massektu is <code>mosaicplot()</code></p>
<ul>
<li><code>x</code> Data for a column (may be used for functions) <code>table()</code> Generating);</li>
<li><code>main</code> 、 <code>sub</code> 、 <code>xlab</code> and <code>ylab</code> Sets the main title, the subheading and the coordinates heading, respectively;</li>
<li> <code>sort</code> Specifies the order in which the variables are displayed;</li>
<li><code>dir</code> (b) Specifying the direction of splits (horizontal splits or vertical splits) in Masaketu;</li>
<li><code>type</code> Type of disability given</li>
</ul>
<h4>Graphics Example</h4>
<p><img src="/assets/images/r-learning-notes/r-language-stat-visualization-41.png" alt="Language R Statistical visualization 41"></p>
<h4>base R</h4>
<pre><code class="language-r">## 绘制泰坦尼克号生还数据的马赛克图
data(Titanic)
par(mar = c(2, 3.5, .1, .1))
mosaicplot(Titanic, shade = TRUE, main = &quot;&quot;)
</code></pre>
<h3>Effort of factors</h3>
<p>Whether there are significant differences in the average variable values of the multi-group differential analysis study</p>
<p>The factor effect map can be seen as a weak and widespread problem of differential analysis, and we can make more statistics than just averages, but we've lost the statistical test, and we've only had exploratory analytical effects, and occasionally it's working, because we can choose some more robust methods.</p>
<p><strong>The problem of differential analysis in mapping the effects of similar factors</strong></p>
<h4>Basic introduction</h4>
<p>The function of the factor effect chart in R is <code>plot.design()</code></p>
<ul>
<li><code>x</code> For the data box containing the self-variant (classification variable), it may also contain the cause variable, in which case the second parameter need not be provided;</li>
<li><code>y</code> For variables;</li>
<li><code>fun</code> The function for which the variable level is calculated;</li>
</ul>
<h4>Graphics Example</h4>
<p><img src="/assets/images/r-learning-notes/r-language-stat-visualization-42.png" alt="Language R Statistical visualization 42"></p>
<h4>base R</h4>
<pre><code class="language-r">## 经纱断裂数据的因素效应图
data(warpbreaks)
names(warpbreaks) = c(&quot;断裂数目&quot;, &quot;羊毛种类&quot;, &quot;拉力强度&quot;)
par(mfrow = c(2, 1), mar = c(4.5, 4, 0.2, 0.2))
plot.design(warpbreaks, col = &quot;blue&quot;,
            xlab = &quot;因素&quot;, ylab = &quot;断裂数目均值&quot;)
plot.design(warpbreaks, fun = median, col = &quot;blue&quot;,
            xlab = &quot;因素&quot;, ylab = &quot;断裂数目中值&quot;)
</code></pre>
<h3>Forest maps</h3>
<p>The name Forest Plot is a little strange, but in fact we have used it in many places without a specific introduction.</p>
<h4>Basic introduction</h4>
<p>Forest maps are used to visualize the effects of multiple studies and to compare the confidence zones, and are usually used to aggregate and compare the results of different studies, especially in meta-Anallysis. Of course, the Cox scale risk regression model that we've built before can also show results using forest maps.</p>
<p>As can be seen from the definition, forest maps are available in all places where a comparison of effects is needed. Each line in the figure represents a study or model variable: a point expression of impact estimates, a horizontal segment of confidence interval; all results are organized vertically to produce a visual effect like forest trees. For such values as HR, OR and RR, the reference lines are usually located <code>1</code>;for the mean margin index, the reference line is usually located <code>0</code>。</p>
<p>In the condensation analysis, each line usually corresponds to a study and the overall effect can also be expressed in a diamond form; in the regression model, each line corresponds to a self-variant. The single-factor forest map is now less used and is largely based on multi-factor forest maps: It also shows the direction of the variables in the model, the amount of effects and the estimated uncertainty. Specific methods of drawing can be used to continue searching, and the rationale is not complex.</p>
<h4>Graphics Example</h4>
<p><img src="/assets/images/r-learning-notes/r-language-stat-visualization-47.png" alt="Multifactors Cox Scale Forest Map Returning Risk"></p>
<p>The above figure simulates a set of multiple factors Cox returns. The blue dot is the estimated risk ratio (Hazard Ratio, HR) and the line is <code>95% CI</code>, the dotted line indicates no effect. <code>HR = 1</code>I don't know. The direction of the effect is clearer when the confidence interval falls entirely on the side of the reference line; if the range passes through the reference line, there remains greater uncertainty about the effect of this variable.</p>
<h4>base R</h4>
<pre><code class="language-r">## 基础作图法绘制多因素 Cox 回归结果的森林图
forest_data = data.frame(
  variable = c(&quot;年龄（每增加 10 岁）&quot;, &quot;男性（相对女性）&quot;,
               &quot;III–IV 期（相对 I–II 期）&quot;, &quot;接受治疗（相对未治疗）&quot;,
               &quot;肿瘤大小 ≥ 5 cm&quot;, &quot;吸烟史&quot;),
  hr = c(1.32, 1.18, 2.41, 0.63, 1.47, 1.09),
  lower = c(1.08, 0.86, 1.71, 0.45, 1.03, 0.78),
  upper = c(1.61, 1.62, 3.39, 0.88, 2.09, 1.52)
)
forest_data = forest_data[nrow(forest_data):1, ]
positions = seq_len(nrow(forest_data))

par(mar = c(4, 11, 0.5, 0.5))
plot(forest_data&#36;hr, positions, type = &quot;n&quot;, log = &quot;x&quot;,
     xlim = c(0.4, 4), ylim = c(0.5, nrow(forest_data) + 0.5),
     xaxt = &quot;n&quot;, yaxt = &quot;n&quot;,
     xlab = &quot;风险比（HR，95% CI）&quot;, ylab = &quot;&quot;)
axis(1, at = c(0.5, 1, 2, 4), labels = c(&quot;0.5&quot;, &quot;1&quot;, &quot;2&quot;, &quot;4&quot;))
axis(2, at = positions, labels = forest_data&#36;variable, las = 1)
abline(v = 1, lty = 2, col = &quot;grey50&quot;)
arrows(forest_data&#36;lower, positions, forest_data&#36;upper, positions,
       angle = 90, code = 3, length = 0.05, col = &quot;#1f77b4&quot;)
points(forest_data&#36;hr, positions, pch = 19, col = &quot;#1f77b4&quot;)
</code></pre>
<h3>Interactive Impact Chart</h3>
<p>In regression models or differential analyses, we often encounter the concept of interaction; interactive effect maps are generally for interaction between classification variables</p>
<p>When looking at the level of a classification variable given, the change in the average of the variables under the levels of the other classification variable indicates that there is no interaction between the two classification variables if the trend remains the same after the change of the previous classification variable to an extraction level.</p>
<h4>Basic introduction</h4>
<p>The function of the interactive effect chart in R is <code>interaction.plot()</code></p>
<ul>
<li><code>x.factor</code> is the classification variable on the cross-coordinate;</li>
<li><code>trace.factor</code> It's the second classification variable, depending on the level of extraction of this classification variable. <code>x.factor</code> Interconnecting with the mean of the variable under the classification;</li>
<li><code>response</code> is due to variables;</li>
<li><code>fun</code> is the function specified for the attribute aggregation, default is the average, and of course we can also specify other calculation functions such as the median <code>median()</code></li>
</ul>
<h4>Graphics Example</h4>
<p><img src="/assets/images/r-learning-notes/r-language-stat-visualization-43.png" alt="Language R Statistical visualization 43"></p>
<h4>base R</h4>
<pre><code class="language-r">## 法国食道癌数据的交互效应图
data(esoph)
par(mar = c(4, 4, 0.2, 0.2))
with(esoph, {
  interaction.plot(agegp, alcgp, ncases / (ncases + ncontrols),
                   trace.label = &quot;饮酒量&quot;, fixed = TRUE,
                   xlab = &quot;年龄&quot;, ylab = &quot;患癌概率&quot;)
})
</code></pre>
<h3>Classification and regression (decision tree)</h3>
<p>Categorization and regression Tree, CART, is a recursive separation (Recursive Partition) technique that seeks to separate some of the variables, leaving the sample split with the greatest variation between groups of variables. This division will continue until the conditions for cessation are met. He's the model of the decision tree we're studying.</p>
<h4>Basic introduction</h4>
<p><strong>rpart</strong> The package provides a calculation function for classification and regression tree <code>rpart()</code> , the function package also expands the generic function <code>plot()</code> , any part-type object is automatically called when drawing <code>plot.rpart()</code> Generates tree maps.</p>
<p><strong>We're breaking away.<code>mlr3</code>After all, it's just an integration, and it's bound to leave the original function behind to streamline it.</strong></p>
<ul>
<li><code>x</code> An object of a rpart type, usually by <code>rpart()</code> (a) Functions to be produced together;</li>
<li><code>uniform</code> Whether or not to use the same vertical distance between top-down branch nodes in order to prevent the branches from being too close to certain local areas to be easily identifiable.</li>
<li><code>branch</code> Sets the shape of the branch, 0 as the "V " font, 1 as a vertical shape, which you can take&#36;[0,1]&#36; between values to make the value shape more like " V " or more vertical;</li>
<li><code>compress</code> Sets whether the spacing of branches is reduced horizontally to make the graphic more compact</li>
</ul>
<h4>Graphics Example</h4>
<p><img src="/assets/images/r-learning-notes/r-language-stat-visualization-44.png" alt="Statistical visualization - 44"></p>
<h4>base R</h4>
<pre><code class="language-r">## 脊椎矫正手术结果的分类树图
library(rpart)
data(kyphosis, package = &quot;rpart&quot;)
levels(kyphosis&#36;Kyphosis) = c(&quot;不存在&quot;, &quot;存在&quot;)
names(kyphosis)[c(2, 4)] = c(&quot;年龄&quot;, &quot;位置&quot;)
fit = rpart(Kyphosis ~ `年龄` + Number + `位置`, data = kyphosis)
par(mar = rep(1, 4), xpd = TRUE)
plot(fit, branch = 0.7)
text(fit, use.n = TRUE, digits = 7)
</code></pre>
<h3>Parallel Coordinate Chart</h3>
<p>Parallel coordinates are an alternative to normal Cartesian coordinate thinking, and we know that the Cartesian coordinate system normally accommodates only two variables at most (cross-axis x axis y), so it is not possible to draw multiple variables directly under such coordinates, and, of course, there are a number of alternatives mentioned earlier that allow multiple data to be expressed under Cartesian coordinates.</p>
<p>The basic approach for parallel coordinate systems is to convert the vertical axis of each other into a parallel axis, where multiple variables can be placed, as the plane can accommodate many parallel lines.</p>
<p>For one line of observations, because of the number of columns, each column corresponds to a point on a parallel line, which, in the end, we link together and form the basic elements that form the parallel map.</p>
<p>Similarly, multiple rows of data can draw multiple lines, and parallel coordinates are made up of these lines with the corresponding parallel axis.</p>
<p>The parallel map has a lot of ways to achieve it.<code>ggplot2</code>System <strong>GGally</strong> Package <code>ggparcoord()</code> Functions</p>
<p><strong>Parallel coordinate maps are also used as contours, but at this point the average sample points are smaller</strong></p>
<h4>Graphics Example</h4>
<p><img src="/assets/images/r-learning-notes/r-language-stat-visualization-45.png" alt="Statistical Visualization-45"></p>
<p>The intersection of the middle segment of the parallel map means negative, parallel, positive.</p>
<p>Because the parallel map draws a number of variables, sometimes we can use the position of the middle line to observe the concentration phenomenon.</p>
<p>The order of variables in the parallel map is very important, as it directly affects the appearance of the map and limits our observation of the data, especially its relevance, since it is only possible to observe the relationship between adjacent variables from the parallel map. Sometimes the order of variables is exchanged, and perhaps new information is observed.</p>
<h4>ggplot</h4>
<pre><code class="language-r">## 鸢尾花数据的平行坐标图
data(&quot;iris&quot;)
library(GGally)
names(iris)[1:4] = c(&quot;花萼长度&quot;, &quot;花萼宽度&quot;, &quot;花瓣长度&quot;, &quot;花瓣宽度&quot;)
p = ggparcoord(iris, columns = 1:4,
               groupColumn = 5, scale = &quot;uniminmax&quot;) +
  geom_line(size = 1.2) +
  labs(x = &quot;变量&quot;, y = &quot;数值&quot;, color = &quot;种类&quot;)
print(p)
</code></pre>
<h3>Combine Curve</h3>
<p>The concoction curve is presented by Andrews, which is a sophisticated technique for displaying multiple data.</p>
<h4>Math principles</h4>
<p>For a Data Matrix&#36;X_{n\times p}&#36;♪ We put every line of it ♪&#36;X_i=(X_{i,1},\ldots,X_{i,p})&#36; To a curve:
&#36;&#36;\left.f=(t)=\left}begin{array&#125;&#125;frac{X i,}{\sqrt{2}+X i,}cos(t)+cdots\X i,p\\\sin+t)+xXx+ \frac{t)+xXi,p}x i,p},p}x i,p}\cos(\t){&amp;\\text{p\text{x{i,&#125;&#125;{\sqrt{2&#125;&#125;x{i,2}\sin(t)+X i,3}\cos(t)+cdots\X i,p}\sin(\frac{p}2}t)&amp;\text{p\text{even}\end{right.\right.&#36;</p>
<p> of which&#36;t\in[-\pi,\pi]&#36;I don't know. This way, will you?&#36;t&#36; If you take a series of values, you can draw a curve for each line of observations, and eventually you can. &#36;n&#36; A bar curve is formed.
A concoction curve. This mathematical transformation appears to be intuitive, yet it has a lot of good mathematical properties and practical implications, and here are just two examples:</p>
<ol>
<li>If we use it... &#36;L_{2}&#36; It's a model to measure the distance between the two curves, so the distance is exactly the same as the distance from the Oxygen squared. &#36;\pi&#36; Multiply, in other words, the distance between the two lines of observation can happen to be the difference between the two curves in the picture. This nature allows us to observe in a visual way the phenomenon of clustering and detached points, since the concepts of clustering and detached points are based on distance (there are many definitions of distance, here using the squares of the occupant distance). If the reader is interested, you can verify this. &#36;L_{2}&#36; Results of the model:</li>
</ol>
<p>&#36;&#36;
\int_{-\pi}^\pi\left(f_i(t)-f_j(t)\right)^2dt=\pi\sum_{k=1}^p\left(X_{i,k}-X_{j,k}\right)^2
&#36;&#36;</p>
<ol start="2">
<li>This shift is somewhat linear: if one observation &#36;X_l&#36; All values are less than &#36;X_i&#36; More than &#36;X_j&#36;, on the concoction curve &#36;X_l&#36; The corresponding curve is also located in &#36;X_i&#36; and &#36;X_j&#36; Between. This is of a very obvious nature.
<strong>Both properties are used temporarily to analyse concoction graphics</strong></li>
</ol>
<h4>Basic introduction</h4>
<p>Reference<code>MSG</code>Package <code>andrews_curve()</code> Functions</p>
<ul>
<li>x is the data matrix</li>
<li>n Number of points for drawing curves</li>
</ul>
<p>There's information, there's a package. <strong>andrews</strong> A function that can be used to draw the curve is provided, although the graphic precision program is slightly less.</p>
<h4>Graphics Example</h4>
<p><img src="/assets/images/r-learning-notes/r-language-stat-visualization-46.png" alt="Statistical Visualization - 46"></p>
<h4>MSG</h4>
<pre><code class="language-r">## 鸢尾花数据和黑莓树数据的调和曲线图
data(iris)
data(trees)
library(MSG)
iris.col = vec2col(iris&#36;Species)
par(mfrow = c(2, 2))
par(mar = c(4, 4, 0.2, 0.2))
andrews_curve(iris[, 1:4], n = 50, col = iris.col,
              xlab = &quot;t&quot;, ylab = &quot;f(t)&quot;)
legend(&quot;topleft&quot;, col = unique(iris.col), lty = 1, bty = &quot;n&quot;,
       legend = unique(iris&#36;Species))
andrews_curve(iris[, c(3, 4, 2, 1)], n = 50, col = iris.col,
              xlab = &quot;t&quot;, ylab = &quot;f(t)&quot;)
andrews_curve(scale(iris[, 1:4]), n = 50, col = iris.col,
              xlab = &quot;t&quot;, ylab = &quot;f(t)&quot;)
x = andrews_curve(scale(trees), n = 50,
                   xlab = &quot;t&quot;, ylab = &quot;f(t)&quot;)
</code></pre>
<h2>Matrix Graphics</h2>
<p>The matrix graphics have two variables in direct view, the line coordinates and the column coordinates, but it's also a value that should be taken from each of the coordinates because it's really special in form, so here's a separate introduction.</p>
<h3>Waiting for high maps and contours</h3>
<p>It's a way of lowering the original three-dimensional matrix data. <strong>After all, it's hard to find the right angle for all the information.</strong></p>
<p>The idea of a high map comes from a geographical contours, but it turns the coordinates into a non-continuous matrix.</p>
<h4>Basic introduction</h4>
<p>R to draw e.g. high maps and contours <code>contour</code> Functions</p>
<ul>
<li><code>nlevels</code> Set the number of lines at the same height, and the more it gets, the more it gets.</li>
<li><code>levels</code> Set an equal high line&#36;z&#36;Value to be connected at point near this value</li>
<li><code>methon</code> Set Drawing Method <code>simple</code> End of online tag <code>edge</code> Embedded Tabs <code>flattest</code> Places on the online level</li>
<li><code>x</code>and<code>y</code>is the vector of the grid point, which defines the position of the peg on the 2D plane</li>
<li><code>z</code>It's a matrix. It's a matrix.<code>(x, y)</code>Function on Grid Point</li>
</ul>
<h4>Graphics Example</h4>
<p><img src="/assets/images/r-learning-notes/r-stat-visualization.png" alt="R Statistical visualization">
Consorption is a group feature.</p>
<h4>base R</h4>
<pre><code class="language-r">## 基础作图法绘制中国 31 地区国民预期寿命和高学历人数密度等高图
library(KernSmooth)
data(ChinaLifeEdu, package = &quot;MSG&quot;)
par(mar = c(4, 4, 0.2, 0.2))
est = bkde2D(ChinaLifeEdu, apply(ChinaLifeEdu, 2, dpik))
contour(est&#36;x1, est&#36;x2, est&#36;fhat, nlevels = 15, col = &quot;darkgreen&quot;,
        vfont = c(&quot;sans serif&quot;, &quot;plain&quot;),
        xlab = &quot;预期寿命&quot;, ylab = &quot;高学历人数&quot;)
points(ChinaLifeEdu, pch = 20)
</code></pre>
<h4>ggplot</h4>
<pre><code class="language-r">## ggplot2 绘制中国 31 地区国民预期寿命和高学历人数密度等高图
library(KernSmooth)
library(metR)
data(ChinaLifeEdu, package = &quot;MSG&quot;)
est = bkde2D(ChinaLifeEdu, apply(ChinaLifeEdu, 2, dpik))
est_tidy = data.frame(
  life = rep(est&#36;x1, length(est&#36;x2)),
  edu = rep(est&#36;x2, each = length(est&#36;x1)),
  z = as.vector(est&#36;fhat)
)
levels = pretty(range(est_tidy&#36;z, finite = TRUE), 15)
p = ggplot(est_tidy, aes(life, edu)) +
  geom_contour(aes(z = z), breaks = levels) +
  geom_text_contour(aes(z = z)) +
  geom_point(aes(Life.Expectancy, High.Edu.NO), data = ChinaLifeEdu) +
  labs(x =&quot;预期寿命&quot;, y = &quot;高学历人数&quot;)
print(p)
</code></pre>
<h3>Colours high</h3>
<p>It doesn't make any difference in principle to the height of the grade.</p>
<h4>Basic introduction</h4>
<p>The color equal high graph function in R is <code>filled.contour()</code></p>
<p>Most parameters and <code>contour()</code> The function is exactly the same, and the difference is that there are several more parameters that define colours.</p>
<h4>Graphics Example</h4>
<p><img src="/assets/images/r-learning-notes/r-stat-visualization-01.png" alt="R Statistical visualization-1"></p>
<h4>base R</h4>
<pre><code class="language-r">## 火山高度数据颜色等高图
par(mar = c(4, 4, 2, 2), cex.main = 1)
x = 10 * 1:nrow(volcano)
y = 10 * 1:ncol(volcano)
filled.contour(x, y, volcano,
               color = terrain.colors,
               plot.title = title(
                 xlab = &quot;北部长度（米）&quot;, ylab = &quot;西部长度（米）&quot;
               ),
               plot.axes = {
                 axis(1, seq(100, 800, by = 100))
                 axis(2, seq(100, 600, by = 100))
               },
               key.title = title(main = &quot;高度\n(米)&quot;),
               key.axes = axis(4, seq(90, 190, by = 10))
)
</code></pre>
<h3>Colour Chart</h3>
<p>The colour map is a softened image of heights, so we don't do smooth processing, but we do simple colour mapping of a matrix, and color squares are the size of a number.</p>
<p><strong>Colour maps are a visualization tool for matrix data, such as the Aligning Matrix.</strong></p>
<p>Because the relevant coefficient colours are used too widely, we study them separately.<a href="/en/blog/2024/03/15/r-visualization-learning-notes/">Related coefficient heat</a></p>
<h4>Basic introduction</h4>
<p>The function of the colour chart in R is <code>image()</code></p>
<ul>
<li>Parameters <code>x</code> 、 <code>y</code> 、 <code>z</code> Similar to the parameters of the contours</li>
<li><code>col</code> Set a colour sequence to map values of different sizes</li>
<li><code>breaks</code> Organisation <code>z</code> Endpoint of the segment</li>
</ul>
<h4>Graphics Example</h4>
<p><img src="/assets/images/r-learning-notes/r-stat-visualization-02.png" alt="R Statistical visualization-2"></p>
<h4>base R</h4>
<pre><code class="language-r">## 基础作图法绘制火山高度数据颜色图
data(volcano)
par(mar = rep(0, 4), ann = FALSE)
x = 10 * (1:nrow(volcano))
y = 10 * (1:ncol(volcano))
image(x, y, volcano, col = terrain.colors(100), axes = FALSE)
contour(x, y, volcano, levels = seq(90, 200, by = 5),
        add = TRUE, col = &quot;peru&quot;)
box()
</code></pre>
<h4>ggplot</h4>
<pre><code class="language-r">## ggplot2 绘制火山高度数据颜色图
data(volcano)
library(ggplot2)
p = ggplot(transform(reshape2::melt(volcano),
                 x = Var1 * 10, y = Var2 * 10),
       aes(x = x, y = y, z = value, fill = value)) +
  geom_tile() +
  geom_contour() +
  scale_fill_distiller(palette=&quot;RdYlGn&quot;) +
  labs(x = &quot;北部长度（米）&quot;, y = &quot;西部长度（米）&quot;,
       fill = &quot;高度\n(米)&quot;)
print(p)
</code></pre>
<h3>3-D view</h3>
<p>That's the 3D of the contours.
Naturally. <code>ggplot2</code> It won't provide the solution we need.</p>
<h4>Basic introduction</h4>
<p>The function of the medium-through view for R is <code>persp()</code></p>
<ul>
<li>Parameters <code>x</code> 、 <code>y</code> 、 <code>z</code> Similar to the parameters of the contours</li>
<li><code>theta</code> and <code>phi</code> Set the angles for the rotation of the 3D graphics in the right, right and down directions, respectively</li>
<li><code>r</code> Set the distance between the eyes and the center of the lens view</li>
</ul>
<p>Special</p>
<ul>
<li><strong>grDevices</strong> The package provides a related 3-D lens view conversion function <code>trans3d()</code> , it converts the three-dimensional coordinates of a space point to a flat coordinate according to the characteristics of the perceiving view, so that we can easily use the general bottom-forming function to add graphic elements to the stereo map</li>
<li><strong>scatterplot3d</strong> As a dedicated three-dimensional drawing package, the package offers a lot of graphics.</li>
<li><strong>rgl</strong>A package based on OpenCV provides interactive 3D graphics</li>
</ul>
<h4>Graphics Example</h4>
<p><img src="/assets/images/r-learning-notes/r-stat-visualization-03.png" alt="R Statistical visualization-3"></p>
<h4>base R</h4>
<pre><code class="language-r">## 火山的三维透视图
data(&quot;volcano&quot;)
z = volcano
x = 4 * (1:nrow(z))
y = 4 * (1:ncol(z))
par(mar = rep(0, 4))
persp(x, y, z, theta = 150, phi = 30, col = &quot;green3&quot;, ltheta = -120,
      shade = 0.75, scale = FALSE, border = NA, box = FALSE)
</code></pre>
<h3>Matrix, matrix points, matrix lines</h3>
<p>The name of the matrix is derived from its parameter type, which can express all columns in a curve, the same meta-function curve, for a matrix Figure<a href="/en/blog/2024/03/15/r-visualization-learning-notes/">A one-digit function curve</a>Again, there's nothing special about it. It's just a convenient cover. We don't have to call. <code>lines()</code> equals draw curves of all columns of the matrix in turn.</p>
<h4>Basic introduction</h4>
<p>The function of the matrix in R is <code>matplot()</code>, the function of the matrix point is <code>matpoints()</code>, the function of the matrix is <code>matlines()</code>
Functions <code>matplot()</code> High-level graphic functions (creation of new graphics), the latter two functions being lower-level graphic functions (adding elements to existing graphics)</p>
<ul>
<li>Parameters <code>x</code> and <code>y</code> To enter the matrix, the pattern is made using <code>x</code> and the variables listed in the cross-axis direction, <code>y</code> , and then use these columns to make a dispersed map (in turn). <code>x</code> The first row against the first row of y, <code>x</code> 2nd row pair <code>y</code> 2nd column, in descending order);</li>
<li>If one of these two parameters is missing, then the x will be <code>1:nrow(y)</code> Replace</li>
</ul>
<h4>Graphics Example</h4>
<p><img src="/assets/images/r-learning-notes/r-stat-visualization-04.png" alt="R Statistical visualization 4"></p>
<h4>base R</h4>
<pre><code class="language-r">## 基础作图法用矩阵图画出的一系列正弦曲线
sines = outer(1:20, 1:4, function(x, y) sin(x / 20 * pi * y))
par(mar = c(2, 4, .1, .1))
matplot(sines, type = &quot;b&quot;, pch = 21:24, col = 2:5, bg = 2:5)
## 数据矩阵的前 6 行
round(head(sines), 5)
</code></pre>
<h4>ggplot</h4>
<pre><code class="language-r">## ggplot2 画出的一系列正弦曲线
sines = outer(1:20, 1:4, function(x, y) sin(x / 20 * pi * y))
df = expand.grid(x = 1:20, y = factor(1:4))
df&#36;sines = as.vector(sines)
p = ggplot(df, aes(x = x, y = sines, color = y)) +
  geom_point(aes(shape = y)) +
  geom_line()
print(p)
</code></pre>
<h3>Hot Chart</h3>
<p>The heat map is achieved by adding rows and columns to the colour map, and he'll prepare us a spectrograph that will reflect some of the group characteristics.</p>
<p><strong>Heapmap itself does not provide the ability to add numbers, nor is it easy to add legends, so it is generally not used for the drawing of relevant coefficients, but only for cluster problems.</strong></p>
<h4>Basic introduction</h4>
<p>Thermal chart function for R <strong>stats</strong> Package <code>heatmap()</code></p>
<ul>
<li>of which <code>x</code> It is a data matrix, which can only be a matrix, not a data frame or other type;</li>
<li><code>Rowv</code> and <code>Colv</code> determines how rows and columns are calculated and reordered as <code>NULL</code>(Default) reorder rows and columns by hierarchical grouping and draw spectrographs accordingly if <code>NA</code> If so, no spectrograph;</li>
<li><code>distfun</code> Determines which function to calculate the distance to further calculate the grouping, default to <code>dist()</code> ；</li>
<li><code>hclustfun</code> (a) the function to be used to calculate the hierarchy;</li>
<li><code>...</code> Parameters passed to <code>image()</code> So we can still use it. <code>image()</code> , for example <code>col</code> Sets the colour series of cells</li>
</ul>
<h4>Graphics Example</h4>
<p><img src="/assets/images/r-learning-notes/r-stat-visualization-05.png" alt="R Statistical visualization 5"></p>
<h4>base R</h4>
<pre><code class="language-r">## 汽车数据的热图
## 用极端化调色板
library(RColorBrewer)
heatmap(as.matrix(mtcars), col = brewer.pal(9, &quot;RdYlBu&quot;),
        scale = &quot;column&quot;, margins = c(4, 8))
</code></pre>
<h3>Link Chart</h3>
<p>Associated Chart (Cohen-Friendly Association Plot) is a tool for displaying data from the 2-D Tables. It is based mainly on the Pearson χ2 test of the Tables' independence theory.
It shows that the data are in line with our expectations.</p>
<h4>Basic introduction</h4>
<p>The function of the correlation chart in R is <code>assocplot()</code></p>
<ul>
<li>of which <code>x</code> Data for a column (or matrix);</li>
<li><code>col</code> (a) The colour of the upper and lower rectangle;</li>
<li><code>space</code> to set the spacing between rectangles.</li>
</ul>
<h4>Graphics Example</h4>
<p><img src="/assets/images/r-learning-notes/r-stat-visualization-06.png" alt="R Statistical visualization-6"></p>
<h4>base R</h4>
<pre><code class="language-r">## 眼睛颜色与头发颜色的关联图
data(HairEyeColor)
x = margin.table(HairEyeColor, c(1, 2))
rownames(x) = c(&quot;黑色&quot;, &quot;棕色&quot;, &quot;红色&quot;, &quot;金色&quot;)
colnames(x) = c(&quot;棕色&quot;, &quot;蓝色&quot;, &quot;褐色&quot;, &quot;绿色&quot;)
assocplot(x, xlab = &quot;头发&quot;, ylab = &quot;眼睛&quot;)
</code></pre>
<h3>Four National Maps</h3>
<p>FourFold Plot is a graphic tool for looking at the correlation between two dichotomy variables in the 2x2xk column table, which is based mainly on the test theory of the 2-D list.</p>
<p>It tests the list from the perspective of Odds Ratio, OR,</p>
<p>It's a good comparison between two four-percent radiuss, and if there's a significant difference in the two fan radiuss, the column variable is not independent, i.e. the factor has an impact on the event, which is the most basic use of the Quadrilateral, and there's a calculation of the Quadrilateral, which is also shown in the Quadrilateral with two arcs. There is no overlap between confidence-building arcs, which means that the zero hypothesis cannot be rejected, and vice versa. This is based on the relationship between the hypothetical tests and the estimates.</p>
<h4>Basic introduction</h4>
<p>Four-Purpose Functions for R <code>fourfoldplot()</code></p>
<ul>
<li><code>x</code> is a 2x2xk array that can also take a 2x2 matrix directly when k=1;</li>
<li><code>color</code> Sets a one-quarter circle colour for filling, with the same sector colour on the same diagonal line, and the order of colour filling also reflects the size of the ratio to 1;</li>
<li><code>onf.level</code> As a level of confidence;</li>
<li><code>std</code> The division of the denominator at the time of standardization was determined for the standardized method of the list.</li>
<li>When k≥1, this function will generate k four-polvee in order</li>
</ul>
<h4>Graphics Example</h4>
<p><img src="/assets/images/r-learning-notes/r-stat-visualization-07.png" alt="R Statistical visualization 7"></p>
<h4>base R</h4>
<pre><code class="language-r">## 加州伯克利分校录取数据四瓣图
data(&quot;UCBAdmissions&quot;)
dimnames(UCBAdmissions) &lt;-
  list(`录取情况` = c(&quot;录取&quot;, &quot;拒绝&quot;),
       `性别`= c(&quot;男性&quot;, &quot;女性&quot;),
       `院系` = LETTERS[1:6])
fourfoldplot(UCBAdmissions, mfcol = c(2, 3)) # 2 行 3 列排版
</code></pre>
<h3>Related coefficient heat</h3>
<p>Thermal maps of relevance (relevance maps) are believed to be one of the most familiar data visualization methods, with a high frequency of occurrence in various literatures. A correlation map is a hot map indicating the correlation between the two variables, and almost all data expressing the correlation value can be visualized using the correlation map.</p>
<p>The most easily used relevant coefficients of heat are: <strong>ggcorrplot</strong> Packaged <code>ggcorrplot</code></p>
<p>All he needs to do is provide him with a data matrix, and he's drawing what we're looking for. <strong>ggplot2</strong> Make a drawing, but he redos the function to satisfy the base R norm</p>
<p>If you want to use base R method, you can use <strong>corrgram</strong> The bag. <code>corrgram</code> Functions and <strong>corrplot</strong> Yes. <code>corrplot</code> Functions</p>
