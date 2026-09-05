---
title: 'R Classical Statistics: Estimation, Tests, Linear Models, and GLMs'
title_zh: R 经典统计：估计、检验、线性模型与 GLM
date: 2024-09-05 12:20:32 +0800
categories:
- Programming
- Programming Languages
tags:
- R
- Classical Statistics
author: Hyacehila
mathjax: true
hidden: true
excerpt: Covers descriptive statistics, estimation, hypothesis testing, linear models, ANOVA, nonlinear regression, and GLMs.
description: Covers descriptive statistics, estimation, hypothesis testing, linear models, ANOVA, nonlinear regression, and
  GLMs.
excerpt_zh: 整理描述性统计、参数估计、假设检验、线性模型、方差分析、非线性回归和 GLM。
permalink: /blog/2024/09/05/r-classical-statistics-learning-notes/
lang: en
translation_key: 2024-09-05-r-classical-statistics-linear-models-glm
translation_status: machine
translation_source_hash: 108a443b14ab5bc9c8c896d9e99c038d6c5da206bcb5749980b6c441c7b42ba7
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<h2>EDA and descriptive statistics</h2>
<p>And here we'll introduce the parts of the EDA technology that are mainly descriptive, and there are other parts that are also relevant.</p>
<h2>Probability Map (PDF)</h2>
<p>Although we keep saying that PDF and the CDF have a similar role to play, we use PDF as the main thing, and we do it when we make the drawings, and we don't really study the CDF because it's not intuitive enough.</p>
<h3>Density function graphs of the usual distribution</h3>
<p>Understanding the pattern of the overall distribution helps to capture the basic characteristics of the sample. We first look at some of the probabilistic functions of the commonly used distribution mentioned in chapter III through specific examples (see below). For the discrete distribution point method, for the continuous distribution means its density function. We use the functions of the PDF to make our graphics work.</p>
<p>Here we have some R code to help us understand the usual PDF maps.</p>
<p>Two distributions</p>
<pre><code class="language-r">n&lt;-20
p&lt;-0.2
k&lt;-seq(0,n)
plot(k,dbinom(k,n,p),type=&#39;h&#39;, main=&#39;Binomial distribution, n=20, p=0.2&#39;,xlab=&#39;k&#39;)
</code></pre>
<p>Porcelain distribution</p>
<pre><code class="language-r">lambda&lt;-4.0
k&lt;-seq(0,20)
plot(k,dpois(k,lambda),type=&#39;h&#39;, main=&#39;Poisson distribution, lambda=5.5&#39;,xlab=&#39;k&#39;)
</code></pre>
<p>Geometric distribution</p>
<pre><code class="language-r">p&lt;-0.5
k&lt;-seq(0,10)
plot(k,dgeom(k,p),type=&#39;h&#39;, main=&#39;Geometric distribution, p=0.5&#39;,xlab=&#39;k&#39;)
</code></pre>
<p>Supergeometric distribution</p>
<pre><code class="language-r">N&lt;-30
M&lt;-10
n&lt;-10
k&lt;-seq(0,10)
plot(k,dhyper(k,N,M,n),type=&#39;h&#39;, main=&#39;Hypergeometric distribution,
     N=30, M=10, n=10&#39;,xlab=&#39;k&#39;)
</code></pre>
<p>Negative binary distribution</p>
<pre><code class="language-r">n&lt;-10
p&lt;-0.5
k&lt;-seq(0,40)
plot(k, dnbinom(k,n,p), type=&#39;h&#39;,
     main=&#39;Negative Binomial distribution,
     n=10, p=0.5&#39;,xlab=&#39;k&#39;)
</code></pre>
<p>Normal distribution</p>
<pre><code class="language-r">curve(dnorm(x,0,1), xlim=c(-5,5), ylim=c(0,.8),col=&#39;red&#39;, lwd=2, lty=3)
curve(dnorm(x,0,2), add=T, col=&#39;blue&#39;, lwd=2, lty=2)
curve(dnorm(x,0,1/2), add=T, lwd=2, lty=1)
title(main=&quot;Gaussian distributions&quot;)
legend(par(&#39;usr&#39;)[2], par(&#39;usr&#39;)[4], xjust=1, c(&#39;sigma=1&#39;, &#39;sigma=2&#39;, &#39;sigma=1/2&#39;),
       lwd=c(2,2,2), lty=c(3,2,1),col=c(&#39;red&#39;, &#39;blue&#39;, par(&quot;fg&quot;)))
</code></pre>
<p>t Distribution</p>
<pre><code class="language-r">curve(dt(x,1), xlim=c(-3,3), ylim=c(0,.4),col=&#39;red&#39;, lwd=2, lty=1)
curve(dt(x,2), add=T, col=&#39;green&#39;, lwd=2, lty=2)
curve(dt(x,10), add=T, col=&#39;orange&#39;, lwd=2, lty=3)
title(main=&quot;Student T distributions&quot;)
legend(par(&#39;usr&#39;)[2], par(&#39;usr&#39;)[4], xjust=1, c(&#39;df=1&#39;, &#39;df=2&#39;, &#39;df=10&#39;, &#39;Gaussian distribution&#39;),
lwd=c(2,2,2,2), lty=c(1,2,3,4),
col=c(&#39;red&#39;, &#39;blue&#39;, &#39;green&#39;, par(&quot;fg&quot;)))
</code></pre>
<p>Carside distribution</p>
<pre><code class="language-r">curve(dchisq(x,1), xlim=c(0,10), ylim=c(0,.6), col=&#39;red&#39;, lwd=2)
curve(dchisq(x,2), add=T, col=&#39;green&#39;, lwd=2)
curve(dchisq(x,3), add=T, col=&#39;blue&#39;, lwd=2)
title(main=&#39;Chi square Distributions&#39;)
</code></pre>
<p>F distribution</p>
<pre><code class="language-r">curve(df(x,1,1), xlim=c(0,2), ylim=c(0,.8), lty=1)
curve(df(x,3,1), add=T, lwd=2,lty=2)
curve(df(x,6,1), add=T, lwd=2, lty=3)
title(main=&quot;Fisher&#39;s F&quot;)
</code></pre>
<h3>Histogram and density function estimation</h3>
<h4>Histogram</h4>
<p>Histogram is the basic tool for exploratory data analysis, giving a frequency distribution diagram of data, with a long rectangle of equal widths commonly used in group distance situations, where the rectangle represents the size of the frequency; on the graphic, the cross-references represent the range of values to be taken from the variable of interest, and the coordinates indicate the frequency (or frequency) size, so that the frequency (or frequency) is the right direction. Figure</p>
<pre><code class="language-r">hist(x, breaks = &quot;Sturges&quot;, freq = NULL, probability = !freq,
     col = NULL, main = paste(&quot;Histogram of&quot; , xname),
     xlim = range(breaks), ylim = NULL,
     xlab = xname, ylab, axes = TRUE, nclass = NULL)
</code></pre>
<p>Where the breaks are used to specify partitions (integer numbers are number of spaces) col indicates colours freq indicates whether to use frequency-numbers in the square Figure</p>
<h4>Nuclear density estimates</h4>
<pre><code class="language-r">density(x, bw = &quot;nrd0&quot;,
        kernel = c(&quot;gaussian&quot;, &quot;epanechnikov&quot;, &quot;rectangular&quot;,
                   &quot;triangular&quot;, &quot;biweight&quot;, &quot;cosine&quot;, &quot;optcosine&quot;),
        n = 512, from, to)
</code></pre>
<p>kernel decides that the smooth function is used by default is normal; n gives the number of nuclear density estimates at equal intervals from the left and right ends of the nuclear density estimates to be calculated separately from to Points</p>
<h4>Example</h4>
<pre><code class="language-r">N &lt;- 100000
n &lt;- 100
p &lt;- .9
x &lt;- rbinom(N,n,p)
hist(x, xlim=c(min(x),max(x)), probability=T,
     nclass=max(x)-min(x)+1, col=&#39;lightblue&#39;,
     main=&#39;Binomial distribution, n=100, p=.5&#39;)
lines(density(x,bw=1), col=&#39;red&#39;, lwd=3)
</code></pre>
<h2>Descriptive statistical analysis</h2>
<p>Descriptive statistical analysis is an important part of the EDA, and we're here to present it, in terms of the data-type gap.</p>
<p>Knowledge on graphics is available for reference<a href="/en/blog/2024/03/15/r-visualization-learning-notes/">R Visualization</a></p>
<h3>Descriptive statistical analysis of individual data sets</h3>
<h4>Graphical missions</h4>
<p>The distribution of the single group data can be done by the histograms described above, as well as by the nucleodensity curve and box line. Normality tests are generally done using QQ graphs.</p>
<pre><code class="language-r">library(DAAG)
data(possum)
fpossum &lt;- possum[possum&#36;sex==&quot;f&quot;,]
par(mfrow=c(1,2))
attach(fpossum)
hist(totlngth,breaks=72.5+(0:5)*5, ylim=c(0,22),
       xlab=&quot;total length&quot;, main=&quot;A:Breaks at 72.5,77.5…&quot;)

stem(fpossum&#36;totlngth)

boxplot(fpossum&#36;totlngth)

qqnorm(fpossum&#36;totlngth, main=&quot;Normality Check via QQ Plot&quot;)
qqline(fpossum&#36;totlngth, col=&#39;red&#39;)
</code></pre>
<h4>Data missions</h4>
<p>We've been introduced to the Obsidian, and it's enough to give a simple description of some code.</p>
<pre><code class="language-r">library(DAAG)
data(possum)
fpossum &lt;- possum[possum&#36;sex==&quot;f&quot;,]

## 趋势部分
summary(fpossum&#36;totlngth) #汇总分析
fivenum(fpossum&#36;totlngth) #五数分布
quantile(fpossum&#36;totlngth) #分位数
median(fpossum&#36;totlngth) #中位数
max(fpossum&#36;totlngth)
min(fpossum&#36;totlngth)
mean(fpossum&#36;totlngth) #极大 极小 均值

#离散部分
max(fpossum&#36;totlngth)-min(fpossum&#36;totlngth)
IQR(fpossum&#36;totlngth)
sd(fpossum&#36;totlngth)
var(fpossum&#36;totlngth)
mad(fpossum&#36;totlngth)

#偏度峰度
library(fBasics)
skewness(fpossum&#36;totlngth)
kurtosis(fpossum&#36;totlngth)
</code></pre>
<p>Special statistical kit<strong>fBasics</strong>Functions in<code>basicStats( )</code>It provides almost all descriptive statistics,<strong>pastecs</strong>There's a name in the bag.<code>stat.desc()</code>And the function of the function has the same effect.</p>
<h3>Descriptive statistical analysis of multiple sets of data</h3>
<h4>Graphical missions</h4>
<p>We're more likely to have a few ways of looking at multiple data.</p>
<pre><code class="language-r">n&lt;-10
d&lt;-data.frame(y1 = abs(rnorm(n)),
y2 = abs(rnorm(n)),
y3 = abs(rnorm(n)),
y4 = abs(rnorm(n)),
y5 = abs(rnorm(n)) )
plot(d)
matplot(d, type = &#39;l&#39;, ylab = &quot;&quot;, main = &quot;Matplot&quot;)
boxplot(d)
</code></pre>
<h4>Data missions</h4>
<p>The data we need to look at is not much more than the single set of data, but the original function changes some of the methods used to promote it, and then adds a little bit of content.</p>
<p>In particular, we may need to revert to descriptive statistical analysis functions for a column for the data box, and we prefer to use functions. <code>apply() and sapply() aggregate()</code> They can increase programming efficiency. <a href="/en/blog/2024/09/05/r-basic-learning-notes/">Apply function to matrix and data box</a></p>
<pre><code class="language-r">## 汇总分析和一些针对数据框的操作函数
summary(state.x77)
aggregate(state.x77, list(Region = state.region), mean)
aggregate(state.x77, list(Region = state.region, Cold = state.x77[,&quot;Frost&quot;] &gt; 130),mean)

sd(state.x77)
#var函数不可以继续计算方差了

#相关分析
x&lt;-c(44.4, 45.9, 46.0, 46.5, 46.7, 47, 48.7, 49.2, 60.1)
y&lt;-c(2.6, 10.1, 11.5, 30.0, 32.6, 50.0, 55.2, 85.8, 86.8)
cor(x,y)
cor(x,y,method=&quot;spearman&quot;)
cor(x,y,method=&quot;kendall&quot;)
cor.test(x,y, method=&quot;spearman&quot;)
</code></pre>
<p><code>cor（）</code>The function calculates three correlations at the same time, as Pearson Spearman Kendall, we simply do the description here, and then we study the correlation separately, and we also use the correlation as a hypothetical test.</p>
<h3>Descriptive statistics for disaggregated data</h3>
<h4>List of rows</h4>
<p>If the variables to which the data are focused are qualitative, the data are referred to as disaggregated data, which are often described in tables and serve further statistical analysis, and we consider mainly the data in the D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D</p>
<p>A very simple code allows for a column table to be created.</p>
<pre><code class="language-r">## 本质上就是构造了一个有行名和列名的矩阵，不能能用数据框替代
Eye.Hair &lt;- matrix(c(68,20,15,5, 119,84,54,29, 26,17,14,14, 7,94,10,16),  nrow=4,byrow=T)
colnames(Eye.Hair) &lt;- c(&quot;Brown&quot;, &quot;Blue&quot;, &quot;Hazel&quot;, &quot;Green&quot;)
rownames(Eye.Hair) &lt;- c(&quot;Black&quot;,&quot;Brown&quot;,&quot;Red&quot;, &quot;Blond&quot;)
Eye.Hair
</code></pre>
<p>You can also construct a column combination from the original data using the table function</p>
<pre><code class="language-r">## table()函数从因子factor中获取频数的函数
## 当接受两个factor的时候，table函数创建一个二维的列联表，高维也可
table(menarche,tanner)
</code></pre>
<p>A joint list for the Govi <code>ftable()</code>function can output multi-dimensional arrays in a compact and attractive way, reference functions</p>
<pre><code class="language-r">ftable(table(factorA,factorB,factorC))
</code></pre>
<p>The margins of the list are very important, and we have a function to study it; besides that, it's also important to study the frequency list.</p>
<pre><code class="language-r">## 创建列联表
Eye.Hair &lt;- matrix(c(68,20,15,5, 119,84,54,29, 26,17,14,14, 7,94,10,16), nrow=4,byrow=T)
colnames(Eye.Hair) &lt;- c(&quot;Brown&quot;, &quot;Blue&quot;, &quot;Hazel&quot;, &quot;Green&quot;)
rownames(Eye.Hair) &lt;- c(&quot;Black&quot;,&quot;Brown&quot;,&quot;Red&quot;, &quot;Blond&quot;)

## 1 按照行 2 按照列 计算边缘总和
margin.table(Eye.Hair,1)
margin.table(Eye.Hair,2)

## 1 按照行 2 按照列 计算边缘概率
prop.table(Eye.Hair,1)
prop.table(Eye.Hair,2)

#直接获取概率矩阵
Eye.Hair/sum(Eye.Hair)
</code></pre>
<h4>Simple graphical description</h4>
<p>The bar chart is the most basic graphic depiction of the matrix, and it's very common to have Marseic charts, which we'll present in a separate graphic section. <a href="/en/blog/2024/03/15/r-visualization-learning-notes/">Marcello.</a></p>
<pre><code class="language-r">data(HairEyeColor)
a &lt;- as.table(apply(HairEyeColor,c(1,2),sum))
barplot(a, legend.text = attr(a, &quot;dimnames&quot;)&#36;Hair)
barplot(a, beside = TRUE, legend.text = attr(a, &quot;dimnames&quot;)&#36;Hair)
</code></pre>
<h2>Centralization and standardization of data</h2>
<pre><code class="language-r">scale(x, center = TRUE, scale = TRUE)
#中心化or标准化函数 x接受矩阵和数据框 返回中心化or标准化后的结果
#第一个参数控制中心化 第二个参数控制缩放
</code></pre>
<h2>Statistical extrapolations</h2>
<p>Here we're going to introduce the part of the classic statistics that we're talking about statistical inferences, which is to include parameter estimates, parameter hypothesis tests, and a few more simple non-parametric hypothesis tests, which, while we spent a whole semester on basic knowledge, we're basically good enough to use a chapter on R to achieve it.</p>
<h2>Parameter estimation</h2>
<h3>Rectangular and very similar estimates</h3>
<h4>Rectangular estimate</h4>
<p>By the law of Sinchin and the strong Como Gorov theorem, if the overall X-K-Rect exists, the k-Rect of the sample is reduced by probability to the overall k-Rect, and the continuous function of the sample-Rect is reduced to the continuous function of the general rect.</p>
<p>We don't have to study any theory here, and one example is enough.</p>
<p>An index distribution with a total parameter of λ, the density function is</p>
<p>&#36;&#36;
p(x|\lambda)=\lambda\exp^{-\lambda x},\quad x&gt;0
&#36;&#36;</p>
<p>then&#36;\lambda&#36;The rule is estimated.</p>
<p>&#36;&#36;
\hat{\lambda}=\frac{1}{\overline{X&#125;&#125;
&#36;&#36;</p>
<pre><code class="language-r">X&lt;-c(0.59132754,0.12854935,0.46900228,0.29835980,0.24341462, 0.06566637,0.40085536,2.99687123,0.05278912,0.09898594)
lambda&lt;- 1/mean(X)
lambda
</code></pre>
<p>So this is what the rectangular estimate does, is the original mathematical calculations are part of R's calculations, or is it a simple part of R's calculations?</p>
<h4>Very seemingly estimated (presentation of R optimized function)</h4>
<p>So we need to calculate the apparent function and then use software to do a hugely enhanced processing, which means that what we're actually doing in R is an optimisation problem.</p>
<pre><code class="language-r">#optimize( )的调用格式
optimize(f = , interval = , lower = min(interval),
        upper = max(interval), maximum = TRUE,
        tol = .Machine&#36;double.eps^0.25, ...)
</code></pre>
<p>of which</p>
<ul>
<li>f is an apparent function that requires us to define a form of basic knowledge. Pass.</li>
<li>Interval is the value of the parametric gill; the lower is the bottom of the gill, upper is the upper of the gill.</li>
<li>maxim = TRUE is for a large value, otherwise (maximm = FALSE) indicates the very small value of the function.</li>
<li>Tol is the exact value requested, and it's normal to default.</li>
</ul>
<p><strong>Optimize only applies to the optimization of single parameters, but it applies to both the maximum and the minimum.</strong></p>
<pre><code class="language-r">#nlm( )的调用格式
nlm(f, p, hessian = FALSE, typsize=rep(1, length(p)),     fscale=1,print.level = 0, ndigit=12, gradtol = 1e-6,
stepmax = max(1000 * sqrt(sum((p/typsize)^2)), 1000),
steptol = 1e-6, iterlim = 100, check.analyticals = TRUE, ...)
</code></pre>
<p><strong>It uses the Newton-Lafson algorithm to find the minimum point of the function.</strong></p>
<pre><code class="language-r">#optim( )的调用格式
optim(par, fn, gr = NULL,
method = c(&quot;Nelder-Mead&quot;, &quot;BFGS&quot;, &quot;CG&quot;, &quot;L-BFGS-B&quot;, &quot;SANN&quot;),
lower = -Inf, upper = Inf, control = list( ), hessian = FALSE, ...)
</code></pre>
<p><strong>Optimize one of the five methods given by the method option</strong></p>
<p>The last two can be used for multidimensional issues.</p>
<p>And here we're going to give an example of a very similar estimate, which is a one-dimensional function that is given directly.</p>
<pre><code class="language-r">f &lt;- function(P){(P^517)*(1-P)^483}
optimize(f,c(0,1),maximum = TRUE)
</code></pre>
<p>It's operating in two parts.</p>
<ul>
<li>maximm is a very similar estimate.</li>
<li>objective is the function of the function at this time</li>
</ul>
<h3>Inter-sectional estimates of mono-normal overall parameters</h3>
<p>Inter-sectional estimates are a relatively special kind of problem, and he has a very close connection to the hypothetical tests, because the core and the statistical data are in a relatively close form.</p>
<p>In fact, many of the problem with inter-sectional estimates is that the results of the inter-sectional estimates, which are performed by the hypothetical function, are part of the results of the hypothetical test function.</p>
<p>But the function of some questions is not provided by R, which in fact does not have to design functions to assist our solution, and we'll just go back to the problem of R providing an estimate of the function.</p>
<p>The difference is unknown, and the most common estimate of the average is the average value of the study.</p>
<p>&#36;&#36;
\begin{pmatrix}\overline{X}-\frac{S}{\sqrt{n&#125;&#125;t_{1-\frac{\alpha}{2&#125;&#125;(n-1),\overline{X}+\frac{S}{\sqrt{n&#125;&#125;t_{1-\frac{\alpha}{2&#125;&#125;(n-1)\end{pmatrix}
&#36;&#36;</p>
<p>It's easy to know that we're doing a t-test.</p>
<pre><code class="language-r">t.test(x, y = NULL,alternative = c(&quot;two.sided&quot;, &quot;less&quot;, &quot;greater&quot;), mu = 0, paired = FALSE, var.equal = FALSE, conf.level = 0.95, ...)
</code></pre>
<ul>
<li>X, y is the data used for the tests, and all give is a double samplet.</li>
<li>Alternative decision-type</li>
<li>mu is average, only works with hypothetical tests.</li>
</ul>
<p>Here's a simple example.</p>
<pre><code class="language-r">x&lt;-c(175 , 176 , 173 , 175 ,174 ,173 , 173, 176 , 173,179 )
t.test(x)
t.test(x)&#36;conf.int
#可以用conf.int选择只访问置信区间 本质上就是在列表上选择了一部分分量
</code></pre>
<h3>Inter-sectional estimates of two normal overall parameters</h3>
<p>The mathematical form of the range estimate for the difference between the two aggregates, which is unknown but equal, is the average difference.</p>
<p>&#36;&#36;
\left((\overline{X}-\overline{Y})\pm t_{1+\frac\alpha2}\sqrt{\frac1n+\frac1m}\sqrt{\frac{(n-1)S_1^2+(m-1)S_2^2}{n+m-2&#125;&#125;\right)
&#36;&#36;</p>
<p>And at this point, we know that what is needed is a test t. Using the t test function described above, we can use the following examples:</p>
<pre><code class="language-r">x&lt;-c(628,583,510,554,612,523,530,615)
y&lt;-c(535,433,398,470,567,480,498,560,503,426)
t.test(x,y,var.equal=TRUE)
#var.equal需要设定TRUE 此时认为两个总体方差相等
</code></pre>
<p>The study equation is also a common feature of the two normals in general.</p>
<pre><code class="language-r">var.test(x, y, ratio = 1,
         alternative = c(&quot;two.sided&quot;, &quot;less&quot;, &quot;greater&quot;),conf.level = 0.95, ...)
</code></pre>
<p>One example is as follows:</p>
<pre><code class="language-r">x&lt;-c(20.5,19.8,19.7,20.4,20.1,20.0,19.0,19.9)
y&lt;-c(20.7,19.8,19.5,20.8,20.4,19.6,20.2)
var.test(x,y)
</code></pre>
<h3>Inter-area estimates for single-total ratio p</h3>
<p>In many practical questions, we often have to estimate the proportion of individuals with certain characteristics in the overall population, which is a category that deserves separate research, which is very important, and the mathematical theory generally uses large samples to get a normal distribution in the form of the following:</p>
<p>&#36;&#36;
\hat{p}\pm z_{1-\frac{\alpha}{2&#125;&#125;\sqrt{\hat{p}(1-\hat{p})/n}-\frac{1}{2n}.
&#36;&#36;</p>
<p>We have a special R function to do this. <strong>The sample properties are almost subject to hypergeometric distributions, and we can choose to use either normal or two distributions to match them.</strong>The latter is a different mathematical form.</p>
<pre><code class="language-r">#调用格式
prop.test(x, n, p = NULL,
          alternative = c(&quot;two.sided&quot;, &quot;less&quot;, &quot;greater&quot;),
          conf.level = 0.95, correct = TRUE)
#正态检验是一种近似检验 需要大样本
binom.test(x, n, p = NULL,
          alternative = c(&quot;two.sided&quot;, &quot;less&quot;, &quot;greater&quot;),
          conf.level = 0.95, correct = TRUE)
#二项分布是一种精确检验 不需要大样本 这里我们本质上是调用了二项分布的估计和检验
</code></pre>
<ul>
<li>x is the number of samples n is the total number</li>
<li>Correct is whether to use a continuous distribution approximation</li>
<li>P is the probability of the original hypothesis, which is useful in the hypothetical test.</li>
</ul>
<h3>Estimated inter-area differences in the two overall ratios</h3>
<p>In the case of large samples, they are almost subject to normal distribution, so they can give the formula that has been used to make a case for the distribution of the normal.</p>
<p>&#36;&#36;
(\hat{p}_1-\hat{p}<em>2)\pm z</em>{1-\frac{\alpha}{2&#125;&#125;\sqrt{\frac{\hat{p}_1(1-\hat{p}_1)}{n_1}+\frac{\hat{p}_2(1-\hat{p}_2)}{n_2}.}
&#36;&#36;</p>
<p>We have only one form to deal with this type of problem.</p>
<pre><code class="language-r">like&lt;-c(478, 246)
people&lt;-c(1000, 750)
prop.test(like, people)
</code></pre>
<h3>Sample capacity determination</h3>
<p>This is a counter-question of a range of estimates, and we give the maximum permissible error of the parameter estimates, and we calculate the sample capacity required; mathematically, there's no change in the calculation, but there are some new functions that are needed to deal with the problem.</p>
<p>R base does not provide any function that would help us deal with this type of problem, suggesting manual extrapolation, using R-based functions to do some ancillary computation; of course, it can also ask if there are any packages that help us with this type of problem when needed.</p>
<h2>Parameter hypothetical tests</h2>
<p>Another important element of statistical extrapolation is the hypothetical tests, the information provided by the sample, the construction of the appropriate statistical quantity, and the testing of the assumptions provided;</p>
<p>Assuming that the tests are divided into two broad categories of parameters, the overall distribution at this time (classical mathematical studies) is known as the case of the parameters we study;</p>
<p>We're testing the distribution type for a detailed study in non-parametric statistics, and we need more theoretical support to understand it better.</p>
<h3>Some important theoretical knowledge in the hypothetical test</h3>
<p>Our core step is</p>
<ol>
<li>Give the original assumptions, get the corresponding alternative assumptions.</li>
<li>Determination of the level of visibility&#36;\alpha&#36;</li>
<li>Study the numbers, determine his distribution.</li>
<li>Gives a field of rejection of the original rejection (statistical down to the original rejection)</li>
<li>Calculates the value of the test statistics to which the sample points correspond</li>
<li>Disclaimer of the original hypothesis</li>
</ol>
<p>- Assuming it's a test.&#36;p&#36;Value is a different kind of hypothetical test from our previous studies; at this point we do not determine the level of prominence in advance, but we calculate it.&#36;p&#36; Value: To decide whether to reject the original hypothesis by comparing it with the common situation of several levels of prominence</p>
<p>&#36;p&#36;The smaller the value, the more it should be rejected, the more common it is to have a few significant levels of 0.1 0.05 0.01, and every less, the more significant trend is to reject the original assumptions, and the more important that we are in the later hypothesis test is that we are looking at the following:&#36;p&#36;Value</p>
<p>We're just going to go back and introduce some of the more important hypothetical questions.</p>
<h3>Test for single-normal overall parameters</h3>
<p>The hypothetical test of the single normal overall average is easy to verify.</p>
<p>&#36;&#36;
T=\frac{\overline{X}-\mu_{0&#125;&#125;{S^{*}/\sqrt{n&#125;&#125;  \sim t(n-1)
&#36;&#36;</p>
<p>And we're still calling the functions we've described.</p>
<pre><code class="language-r">salt&lt;-c(490 , 506, 508, 502, 498, 511, 510, 515 , 512)
t.test(salt, mu=500)
</code></pre>
<p>And the mu is the average of the original assumption, which is the same function as the estimated spatially of the average, and in fact, they do the same thing, and if they need to, we can do a single-sided examination, and adjust the parameters inside.</p>
<h3>Test of two normal overall parameters</h3>
<p>The test of whether the two normal general averages are zero can also be conducted directly as a hypothetical test.</p>
<pre><code class="language-r">x&lt;-c(628,583,510,554,612,523,530,615)
y&lt;-c(535,433,398,470,567,480,498,560,503,426)
t.test(x,y,var.equal=TRUE)
#var.equal需要设定TRUE 此时两个总体方差相等 否则我们需要修改参数为FALSE
</code></pre>
<p>The problem with the difference can also be calculated directly from the function in front, the assumption is still one.</p>
<pre><code class="language-r">x&lt;-c(20.5,19.8,19.7,20.4,20.1,20.0,19.0,19.9)
y&lt;-c(20.7,19.8,19.5,20.8,20.4,19.6,20.2)
var.test(x,y)
</code></pre>
<h3>Test of data in pairs t</h3>
<p>We can use the t.test function that we've already described, but we just need to modify some parameters, and the difference doesn't matter at this point, but the pairs need to be declared individually.</p>
<pre><code class="language-r">x&lt;-c(20.5, 18.8, 19.8, 20.9, 21.5, 19.5, 21.0, 21.2)
y&lt;-c(17.7, 20.3, 20.0, 18.8, 19.0, 20.1, 20.0, 19.1)
t.test(x, y, paired=TRUE)
</code></pre>
<h3>Test of single sample ratio</h3>
<p>We're still ahead of us, with two kinds of precision tests and a proximate test.</p>
<pre><code class="language-r">binom.test(c(7, 5), p=0.4)
#这是另一种调用格式 输入向量 前面是成功次数 后面是失败次数
prop.test(7, 12, p=0.4, correct=TRUE)
#近似检验可能不准 R会在任何需要的时候给出警告
</code></pre>
<h3>Test of two sample ratios</h3>
<p>In fact, it's very simple, exactly the same pattern as the previous parameter estimates.</p>
<pre><code class="language-r">like&lt;-c(478, 246)
people&lt;-c(1000, 750)
prop.test(like, people)
</code></pre>
<h2>Relevant analysis</h2>
<h2>Relevance of classification variables</h2>
<p>First we need to test the independence of the classification variables, that is, the independence test of the matrix.</p>
<p>R offers a variety of things.<strong>Test for type variable independence</strong>The way we're going to be here to introduce you.</p>
<h3>The Carabineros Independence Test</h3>
<p>Available<code>chisq.test()</code>function to test the column and column variables of the 2-dimensional table for the card independence of the column, as shown below</p>
<pre><code class="language-r">compare&lt;-matrix(c(60,32,3,11), nr = 2, dimnames = list(c(&quot;cancer&quot;, &quot;normal&quot;),c(&quot;smoke&quot;, &quot;Not smoke&quot;)))

chisq.test(compare, correct=TRUE) #检验函数接受一个二维表
</code></pre>
<h3>Fisher's exact inspection.</h3>
<p>The previous C.I.S. test allows only 20% of the two-dimensional list to have an expected frequency of less than 5 or a warning that we can use the Fisher test, instead of a similar calibration, to be used on a two-dimensional list of any number of columns greater than 2 but not for a warning.&#36;2\times2&#36;</p>
<pre><code class="language-r">compare&lt;-matrix(c(60,32,3,11), nr = 2, dimnames = list(c(&quot;cancer&quot;, &quot;normal&quot;),c(&quot;smoke&quot;, &quot;Not smoke&quot;)))

fisher.test(compare, correct=TRUE)
</code></pre>
<h3>Cochran-Mantel-Haenszel Test</h3>
<p><code>mantelhaen.test()</code>The function is used to perform the Cochran-Mantel-Haenszel card-check, the assumption being,<strong>Two nominal variables are independent in each of the third variables.</strong>I'm sorry. The following codes test the independence of treatment and improvement at each level of gender.</p>
<pre><code class="language-r">## 构建列联表
mytable &lt;- xtabs(~Treatment+Improved+Sex, data=Arthritis)
## 检验
mantelhaen.test(mytable)
</code></pre>
<h3>Calculation of relevant coefficients</h3>
<p>The distinguishing test in the previous section assessed whether there was sufficient evidence to reject the original assumption that the variables were independent of each other. If you can reject the assumption, then your interest will naturally shift to the measure of relevance that measures the strength and weakness of relevance.<strong>vcd</strong>Package<code>assocstats()</code>function to calculate the phi coefficient, the column link and the Cramer's V coefficient of the 2-dimensional array table. Example code</p>
<pre><code class="language-r">library(vcd)
mytable &lt;- xtabs(~Treatment+Improved, data=Arthritis)
assocstats(mytable)
</code></pre>
<h2>Pearson, Spearman and Kendall are related.</h2>
<p>Pearson ' s coefficients measure linear correlation between two quantitative variables. The Specarman grade-related coefficient measures the degree of correlation between the hierarchical sequence variables. Kendall's Tau-related coefficient is also a grade-related measure of non-parameters. He's also our most basic three related coefficients. Available<code>cor</code>function calculates,<code>methon</code>Parameters are used to select the method of calculation, as follows:</p>
<pre><code class="language-r">states&lt;- state.x77[,1:6]
#cor函数可以对一个数据框进行操作
cor(states)
</code></pre>
<p>After calculating the coefficient, we should also consider the issue of visibility, which can be tested using the Cor.test() function for individual Pearson, Spearman and Kendall-related coefficients. The simplified format for use is:</p>
<pre><code class="language-r">cor.test(x, y, alternative = , method = )
</code></pre>
<h2>Offset factor</h2>
<p>Relevance refers to the interrelationship between the other two quantitative variables when controlling one or more quantitative variables. You can use it.<strong>ggm</strong>Package<code>pcor()</code>function calculates the relative coefficient The function calls in the format:</p>
<pre><code class="language-r">pcor(u, S)
</code></pre>
<p>Of which<code>u</code>is a numerical vector, the first two values represent the subscript of the variable to which the relevant coefficient is to be calculated, and the remaining values are the subscript of the conditional variable (i.e. the variable to which the impact is to be excluded).<code>S</code>is the synapse array of variables. Example:</p>
<pre><code class="language-r">library(ggm)
colnames(states)
pcor(c(1,5,2,3,6), cov(states))
</code></pre>
<p>Corresponds, we have a hypothetical test function.</p>
<pre><code class="language-r">pcor.test(r, q, n)
</code></pre>
<p>Of which<code>r</code>By<code>pcor()</code>function calculates the coefficient of bias,<code>q</code>For the number of variables (in place of value),<code>n</code>It's the size of the sample.</p>
<h2>Difference Analysis</h2>
<p>The equation analysis (analysis of analysis, short of ANOVA) is an effective statistical method for analysing experimental data in industrial and agricultural production and scientific research.</p>
<p>There are two main causes of differences in observations (variability): uncontrolled fluctuations caused by random factors or observations errors during the testing, and manageable fluctuations caused by different treatments in the testing or by different conditions.</p>
<p>The main work of the variance analysis is to decompose the total variation (variant) of the observation data into factor effects and test errors according to the causes of the variation and to provide quantitative analysis of the factors, comparing the importance of the various causes in the total variation as a basis for further statistical inferences.</p>
<p><strong>ANOVA is widely used in various experimental and quasi-experimental designs, and is another regression model in the case of a classification variable.</strong></p>
<h2>Single factor variance analysis</h2>
<h3>Basic methodology</h3>
<p>We've already introduced specific mathematical models in our pilot design, and here's what R should do.</p>
<pre><code class="language-r">aov(formula, data=NULL, projections=FALSE,qr=TRUE, contrasts=NULL, ...)
</code></pre>
<p>And we'll be back to tell you how this formula should be used.</p>
<p>Gives a single factor, five-level example of a differential analysis, and this code should be able to understand the meaning.</p>
<pre><code class="language-r">X&lt;-c(25.6, 22.2, 28.0, 29.8, 24.4, 30.0, 29.0, 27.5, 25.0, 27.7,
23.0, 32.2, 28.8, 28.0, 31.5, 25.9, 20.6, 21.2, 22.0, 21.2)
A&lt;-factor(rep(1:5, each=4))
#rep函数 times参数是控制整体重复的次数 each是控制每个元素重复的次数
A
miscellany&lt;-data.frame(X, A)
aov.mis&lt;-aov(X~A, data=miscellany)
#X列标识了数据 A列标识了数据对应的因子水平
summary(aov.mis)

plot(miscellany&#36;X ~ miscellany&#36;A)
#绘制分组图形的箱线图，直观的比较差异
plot(miscellany&#36;A, miscellany&#36;X)
#这个和上面是一样的 自变量为因子的plot函数也会绘制boxplot
</code></pre>
<h3>Multiple comparisons of average values</h3>
<p>When we do the differential analysis, we find that there are significant differences in the mean values of the effects, and we can only know that there are certain values that differ, but we can't tell which is different, and the following approach helps us.<strong>Find out which values are different when the variance is analysed.</strong></p>
<p>It's actually a comparison between two levels of a factor A, using a test test that's not fundamentally different from the test we're using in front of us.</p>
<p>However, in the course of multiple t tests, the p-values need to be adjusted to ensure normal judgment, as follows:</p>
<pre><code class="language-r">p.adjust.methods
</code></pre>
<p>He'll tell us what the p-value adjustment is, and there's more than a Bonferroni, and now we're doing multiple t tests, and here's the way we do it.</p>
<pre><code class="language-r">pairwise.t.test(x, g, p.adjust.method=p.adjust.methods,pool.sd=TRUE, ...)
</code></pre>
<p><code>x</code>It's the vector that responds to the variable. <code>g</code>is the group vector (factor)  <code>p.adjust.method</code>It's the way to adjust the p value mentioned above.</p>
<p>The amount of the matrix returned is all<code>p</code>Value</p>
<h3>The Tukey Act</h3>
<p>We rejected the assumption that there was no difference between levels and determined who was different. <strong>Now we want to make a trust gap in the effects.</strong> It's actually another way to determine what effects are different.</p>
<p>function in the</p>
<pre><code class="language-r">TukeyHSD(x, which, ordered=FALSE, conf.level=0.95)
</code></pre>
<p><code>x</code> It's the result of the variance analysis. <code>which</code> is the factor vector that needs to be calculated between the comparative zones <code>ordered</code> is the logical value, if&quot;true&quot;, the factor level is a confidence level</p>
<p>We'll use a simple example to explain the use.</p>
<pre><code class="language-r">sales&lt;-data.frame( X=c(23, 19, 21, 13, 24, 25, 28, 27, 20, 18, 19, 15, 22, 25, 26, 23, 24, 23, 26, 27), A=factor(rep(1:5, c(4, 4, 4, 4, 4))) )
#数据集生成
summary(aov(X~A, sales))
#方差分析
pairwise.t.test(sales&#36;X, sales&#36;A, p.adjust.method=&quot;bonferroni&quot;)
#单组比较
TukeyHSD(aov(X~A, sales))
#计算所有均值差的计算区间 因为我们没改which
plot(TukeyHSD(fit))
#绘图
</code></pre>
<h3>Sqincy test</h3>
<p>To do the differential analysis, we need to make sure that the following three conditions are met.</p>
<ul>
<li>Addability (variability plus)</li>
<li>Independent normality (level of independence, internal normality)</li>
<li>Sq. Alignness (equals of different horizontal differences)</li>
</ul>
<p>Now we're studying how to test each other for compatibility.</p>
<p><strong>In fact, regression analysis is a relaxed equation, but it's not the same. So we usually only study the differences in the equation analysis.</strong></p>
<p>The normality of the disability can be analysed directly by reference to the issue of return</p>
<h4>Bartlett Test</h4>
<p>Function format is that the meaning of the parameters need not be explained, as many places in the front line.</p>
<pre><code class="language-r">bartlett.test(x, g, ...)
bartlett.test(formula, data, subset, no.action, ...)
</code></pre>
<h4>Levene's testing.</h4>
<p>Function format is the same as the argument</p>
<pre><code class="language-r">leveneTest(x, group)
</code></pre>
<h4>Example</h4>
<pre><code class="language-r">bartlett.test(X~A, data=sales)
library(car)
leveneTest(sales&#36;X, sales&#36;A)
</code></pre>
<p>Both p-values are very large, which means they didn't reject the original hypothesis, which is the equation.</p>
<h3>Remarks</h3>
<p>The variance analysis model can be considered a special linear model, so the differential analysis can also use linear model functions.<code>lm( )</code>, and also function<code>anova( )</code>Extract the variance analysis table, therefore<code>aov(formula)</code>Equivalent to<code>anova(lm(formula)) </code>
A single factor differential analysis can also be used for functions<code>oneway.test( )</code>, if the difference between the data below each level is equal (use options)<code>var.equal=TRUE</code>) , it is equivalent to using a function<code>aov( )</code>perform normal variance analysis; if the differences in the data below each level are not equal (use options)<code>var.equal=FALSE</code>(iii) Welch (1951), which uses the approximation method;</p>
<p>When the distribution below each level is unknown (generally default normal, the above method is used), the variance analysis is performed using Kruskal-Wallis, etc. They're non-parametric statistics.</p>
<h2>Double Factor Difference Analysis</h2>
<h3>No interaction</h3>
<p>Theoretically, we've omitted the following code lines.</p>
<pre><code class="language-r">juice&lt;-data.frame(
X = c(0.05, 0.46, 0.12, 0.16, 0.84, 1.30, 0.08, 0.38, 0.4, 0.10, 0.92, 1.57, 0.11, 0.43, 0.05, 0.10, 0.94, 1.10, 0.11, 0.44, 0.08, 0.03, 0.93, 1.15), A = gl(4, 6),B = gl(6, 1, 24)
)
#数据建立
juice.aov&lt;-aov(X~A+B, data=juice)
summary(juice.aov)
#方差分析 把公式部分改了
bartlett.test(X~A, data=juice)
bartlett.test(X~B, data=juice)
#两个方差齐性检验
</code></pre>
<h3>Situations with interactive effects Down</h3>
<p>Theoretically, we've omitted the following code lines.</p>
<pre><code class="language-r">rats&lt;-data.frame(
Time=c(0.31, 0.45, 0.46, 0.43, 0.82, 1.10, 0.88, 0.72, 0.43, 0.45, 0.63, 0.76, 0.45, 0.71, 0.66, 0.62, 0.38, 0.29, 0.40, 0.23, 0.92, 0.61, 0.49, 1.24, 0.44, 0.35, 0.31, 0.40, 0.56, 1.02, 0.71, 0.38, 0.22, 0.21, 0.18, 0.23, 0.30, 0.37, 0.38, 0.29, 0.23, 0.25, 0.24, 0.22, 0.30, 0.36, 0.31, 0.33),
Toxicant=gl(3, 16, 48, labels = c(&quot;I&quot;, &quot;II&quot;, &quot;III&quot;)),
Cure=gl(4, 4, 48, labels = c(&quot;A&quot;, &quot;B&quot;, &quot;C&quot;, &quot;D&quot;)) )
#数据集构建
op&lt;-par(mfrow=c(1, 2))
#设置绘图参数并存储
plot(Time~Toxicant+Cure, data=rats)
#一类特殊的boxplot绘制方法
with(rats,interaction.plot(Toxicant, Cure, Time, trace.label=&quot;Cure&quot;))
with(rats, interaction.plot(Cure, Toxicant, Time, trace.label=&quot;Toxicant&quot;))
#绘制交互效应图 如果不出现明显交叉就基本认为没有交互作用
rats.aov&lt;-aov(Time~Toxicant*Cure, data=rats)
summary(rats.aov)
#虽然认为没有交互 但是还是进行了带有交互效应的方法分析 上面的函数等价于
## rats.aov&lt;-aov(Time~Toxicant+Cure+Toxicant:Cure, data=rats)
## 只考虑交互效应则为
## rats.aov&lt;-aov(Time~Toxicant:Cure, data=rats)
</code></pre>
<p>The results of the variance analysis suggest that two factors are significant, but the interaction is not significant.</p>
<h2>Coordinated differential analysis</h2>
<p>The hypothetical tests of comparison of two or more groups of average values in the range analysis methods described in the preceding two sections, which are generally manageable, are sometimes handled by factors that are not controlled in practice, how to deduct or balance the effects of these uncontrollable factors while comparing the differences between two or more groups of equal values, and the method of co-ordinated differential analysis may be considered.</p>
<p>Analysis of Covariance, ancova<strong>A statistical analysis methodology combining linear regression analysis with differential analysis</strong>The underlying idea is to consider some variables (i.e. unknown or uncontrollable) that have an impact on the response variable Y as covariate, to create linear regression relationships that respond to Y and the change in X, and to use this regression to equalize X values before hypothetically testing the difference between the values of the revised Y for each processing group, which is the substance of the reduction of X-Y squared from the overall equation of Y, and to analyse the difference squared and further decomposed before the squared analysis of the equation is made to better evaluate the effects of this treatment.</p>
<p>We use functions to explain the problem. Use the function in the package HH ancova as</p>
<pre><code class="language-r">ancova(formula, data.in = sys.parent(),x, groups)
</code></pre>
<p>Formula is the formula for the coordinated differential analysis, Data.in is the data, x is the variance analysis, groups is the factor.</p>
<p>Here are some examples.</p>
<pre><code class="language-r">feed&lt;-as.factor(rep(c(&quot;A&quot;,&quot;B&quot;,&quot;C&quot;),each=8) )
Weight_Initial &lt;- c(15,13,11,12,12,16,14,17,17,16,
                    18,18,21,22,19,18,22,24,20,23, 25,27,30,32)
Weight_Increment &lt;-c(85,83,65,76,80,91,84,90,97,90,
                     100,95,103,106,99,94,89,91,83,
                     95,100,102,105,110)
data_feed&lt;-data.frame(feed,Weight_Initial,Weight_Increment)
#数据集构建 其中Weight_Initial是我们想要考虑的协变量
ancova(Weight_Increment ~ Weight_Initial+feed , data=data_feed)
#不考虑交互
ancova(Weight_Increment ~ Weight_Initial*feed , data=data_feed)
#考虑交互
</code></pre>
<h2>Test design and variance analysis in progress</h2>
<h3>Checklist Test</h3>
<p>Let's give an example of how data from the active test was read into the data box.</p>
<pre><code class="language-r">rate&lt;-data.frame(
A=gl(3,3),
B=gl(3,1,9), C=factor(c(1,2,3,2,3,1,3,1,2)),
Y=c(31, 54, 38, 53, 49, 42, 57, 62, 64)
)
#正交试验数据的建立 存储了每个数据对应的各个因子水平
K&lt;-matrix(0, nrow=3, ncol=3, dimnames=list(1:3, c(&quot;A&quot;,&quot;B&quot;,&quot;C&quot;)))
for (j in 1:3)
  for (i in 1:3)
    K[i,j]&lt;-mean(rate&#36;Y[rate[j]==i])
#计算了每个水平的均值（实际上可以使用tapply函数简化 如下）
#K &lt;- tapply(rate&#36;Y, rate&#36;A, mean)
plot(as.vector(K), axes=F, xlab=&quot;Level&quot;, ylab=&quot;Rate&quot;)
xmark&lt;-c(NA,&quot;A1&quot;,&quot;A2&quot;,&quot;A3&quot;,&quot;B1&quot;,&quot;B2&quot;,&quot;B3&quot;,&quot;C1&quot;,&quot;C2&quot;,&quot;C3&quot;,NA)
axis(1,0:10,labels=xmark)
axis(2,4*10:16)
axis(3,0:10,labels=xmark)
axis(4,4*10:16)
lines(K[,&quot;A&quot;]); lines(4:6, K[,&quot;B&quot;]); lines(7:9,K[,&quot;C&quot;])
#因子各个水平 指标均值情况
</code></pre>
<p>This is the visual method of analyzing the direct test, which is actually the best indicator available, but not as rigorous as the differential analysis.</p>
<h3>I'm just trying to get a difference.</h3>
<p>We didn't change the R function.</p>
<pre><code class="language-r">rate.aov&lt;-aov(Y~A+B+C, data=rate)
summary(rate.aov)
</code></pre>
<p>The test is also an analysis of the interaction.</p>
<h3>Repeat the experiment.</h3>
<p>The so-called duplicate measurement differential analysis, i.e. the testee was measured more than once. This section focuses on the analysis of the difference in measurements (a common design) with a group and an inter-group factor.</p>
<p>Because the variable is the carbon dioxide absorption (uptake) variable is the plant type Type and the CO2 concentration (conc) at seven levels, Type is the inter-group factor, conc is the intra-group factor, Plant is the individual symbol</p>
<p>We have a separate approach to this problem involving repeated tests, and we first ask for data structure.<strong>Still one line per observation, which requires simultaneous intra-group factors, inter-cluster factors and individual symbols</strong> And the variance analysis code becomes</p>
<pre><code class="language-r">## A组内因子 W组内因子 B组间因子
y ~ A + Error (Subject/A) #单因素组内 ANOVA
y ~ B * W + Error (Subject/W) #含单个组内因子（w）和单个组间因子（B）的重复测量ANOVA
</code></pre>
<h2>Regressive analysis</h2>
<p>The analysis only leads to a correlation between two variables, but it does not answer how they relate to each other, i.e., they cannot identify the function of a causal relationship between them.</p>
<h2>OLS and derivatives</h2>
<h3>Symbol used in R expression</h3>
<p>Here is the more complex system of functions that we first came into contact with R, and there are symbols that we need to understand.</p>
<p>We're here to create a control sheet to show the more common symbols.</p>
<table>
<thead>
<tr>
<th>Symbol</th>
<th>Expression Meaning</th>
</tr>
</thead>
<tbody><tr>
<td><code>~</code></td>
<td>Separator, left to respond to variable, right to interpret variable</td>
</tr>
<tr>
<td><code>+</code></td>
<td>Separating projection variables</td>
</tr>
<tr>
<td><code>:</code></td>
<td>Intersections that represent the projection variables</td>
</tr>
<tr>
<td><code>*</code></td>
<td>It's not useful to suggest a simple way to all possible interactive items.</td>
</tr>
<tr>
<td><code>^</code></td>
<td>Means that the interactive item reaches a certain number, for example &#36;\text{代码} y \sim(x + z + w)^2 \text{可展开为} y\sim x+ z + w + x:z + x:w + z:w&#36;</td>
</tr>
<tr>
<td><code>-1</code></td>
<td>Remove Intersection</td>
</tr>
<tr>
<td><code>I()</code></td>
<td>Explain the bracketed elements from the arithmetical point of view, avoiding symbol conflicts</td>
</tr>
</tbody></table>
<h3>One-linear regression</h3>
<p>Math forms need not be wasted on the presentation of the basic but important function of regression, which is the basis for our continuing learning of more functions.</p>
<pre><code class="language-r">#回归函数
lm(formula, data, subset, weights, na.action,method=&quot;qr&quot;,
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
predict(object, newdata, interval = &quot;confidence&quot;, level = 0.95)

#绘制回归曲线图 一般和plot联合使用
abline(object)

#手动计算p-value
f_statistic &lt;- summary(X.lm)&#36;fstatistic
f_value &lt;- f_statistic[1]
p_value &lt;- pf(f_value, f_statistic[2], f_statistic[3], lower.tail = FALSE)
</code></pre>
<p><code>lm()</code>Function is the core function of the equation to which the regression equation is based</p>
<ul>
<li>Formula is the choice of regression models.</li>
<li>Data is a data frame</li>
<li>Subset is a subset of sample observations.</li>
<li>Weights are weighted vectors for the assembly</li>
<li>na.action shows whether the data contains missing values</li>
<li>Method is pointing out the method used to make the match.</li>
<li>The logical value behind is whether to return the value</li>
</ul>
<p><code>summary（）</code>The information that is used to answer the entire model is the classic summy function of the system.</p>
<ul>
<li>Model parameter estimates</li>
<li>Model hypothetical test (without including the equation p value that can be directly quoted, but with the f value that can be used to design a function to address this)</li>
</ul>
<h3>Multiple Returns</h3>
<p>We can easily construct a multi-form return form, as follows:</p>
<pre><code class="language-r">fit2 &lt;- lm(weight ~ height + I(height^2), data=women)
## I的含义是里面增加了一个算术项 我们构建是一个多元回归
</code></pre>
<p>It's still a linear regression, a multi-form regression.</p>
<p>In fact,<strong>Whatever we construct on the right side of the equation, as long as the parameter item is linear, it does not affect the properties of linear regression.</strong></p>
<h3>Multi-linear regression and variable selection</h3>
<p>The function has no formal change here.</p>
<pre><code class="language-r">lm.reg&lt;-lm(y~x1+x2+x3+x4, data=blood)
</code></pre>
<p>If you want to study interaction, the code should be changed to</p>
<pre><code class="language-r">fit &lt;- lm(mpg ~ hp + wt + hp:wt, data=mtcars)
</code></pre>
<p>And here we have a little extra question about the selection of variables.</p>
<pre><code class="language-r">step(object, scope, scale=0,direction=c(&quot;both&quot;, &quot;backward&quot;, &quot;forward&quot;),trace=1, keep=NULL, steps=1000, k=2)
</code></pre>
<p>Parameter Interpretation</p>
<ul>
<li>object is the result of a linear model or a broad linear model analysis</li>
<li>The scope means whether or not to limit the scope of model selection</li>
<li>Direction is the choice of the method, forward, backward or back.</li>
</ul>
<p><strong>This function changes the model directly.</strong></p>
<h2>Re-entry diagnosis</h2>
<p>The main elements are: disability analysis, impact analysis, colinear diagnosis; we're still here to study regression diagnosis, because at this point in time, many of our research is really focused on the previous presentation of the OLS, and some of the elements can certainly be extrapolated seamlessly, and can be used well in other models, and diagnosis, especially in the case of disability analysis, is a very important link in the overall return, and it deserves very deep study.</p>
<h4>Standard analytical methodology</h4>
<p>The disability analysis is a very large module, and we're here to present only some of the more common parts.</p>
<p>The most detailed disability analysis should be based on <code>residuals（）</code>function to perform a custom analysis, we do not present here.</p>
<p>The most basic disability analysis is based on <code>plot</code> He's been providing us with four of the most common residual analysis. Figure</p>
<pre><code class="language-r">fit &lt;- lm(weight ~ height, data=women)
par(mfrow=c(2,2))
plot(fit)
</code></pre>
<p>It contains the most popular graphical tool for model diagnosis.</p>
<ul>
<li>Function of the difference pair of y</li>
<li>Disability qq graph testing</li>
<li>Distribution of square root of standardized residuals</li>
<li>Cook Distance</li>
</ul>
<p>So we'll study it separately.</p>
<ul>
<li>Disability and the independence of variables, i.e. whether important regression items are missing</li>
<li>Normality of the disability</li>
<li>Whether the disability is equal to the difference</li>
<li>Impact analysis issues</li>
</ul>
<h4>Impact analysis</h4>
<p>We have a lot of ways to study impacts. <strong>But none of these can be directly concluded by the fact that we're not doing anything to make a specific analysis of the problem.</strong> The following is the text:</p>
<pre><code class="language-r">lm.influence(model, do.coef=TRUE)
</code></pre>
<p>It gives a model regression factor after a certain point of observation, which can be used to determine the impact.</p>
<pre><code class="language-r">cooks.distance(model, infl=im.influence(model, do.coef=FALSE),
               res=weighted.residuals(model), sd=sqrt(deviance(model)/df.residual(model)),
               hat=infl&#36;hat, ...)
</code></pre>
<p>Cook statistics are also very common statistics on the impact of judgement</p>
<pre><code class="language-r">dffits(model, infl=..., res=...)
</code></pre>
<p>This is the DFFITS Code.</p>
<pre><code class="language-r">covratio(model, infl=lm.influence(model, do.coef=FALSE),res=weighted.residuals(model))
</code></pre>
<p>COVRATIO Guidelines</p>
<pre><code class="language-r">influence.measures(model)
</code></pre>
<p>A summary of the statistics on impact, including the above.</p>
<pre><code class="language-r">library(car)
influencePlot(fit, id.method=&quot;identify&quot;, main=&quot;Influence Plot&quot;,
              sub=&quot;Circle size is proportional to Cook&#39;s distance&quot;)
</code></pre>
<p><strong>car</strong>A function provided by the package that integrates information on the discrete points, leverage values and powerful impact points in a visualized chart Medium</p>
<h4>Symmetrical linear diagnosis</h4>
<p>The more common combination of linear diagnostics is the characteristic value kappa, the differential expansion factor VIF, which functions as follows:</p>
<pre><code class="language-r">eigen(x, symmetric, only.values=FALSE, EISPACK=FALSE)
#计算矩阵的特征值 辅助判断复共线性
kappa(x, exact=FALSE, ...)
#计算矩阵的kappa值 也就是条件数 100以上就是强相关 30以上中度相关
vif(lmobj, digits=5)
#计算方差膨胀因子VIF 这个函数来自于DAGG包 10意味着强相关
</code></pre>
<h4>Comprehensive test for regression</h4>
<p><code>gvlma()</code>The functions, which were prepared by Pena and Slate (2006), allow for a comprehensive validation of linear model assumptions, together with an evaluation of slopes, peaks and heterogeneity. In other words, it provides a separate comprehensive test (passed/not passed) for model assumptions from the package <strong>gvlma</strong></p>
<p>It's very convenient to use.</p>
<pre><code class="language-r">library(gvlma)
gvmodel &lt;- gvlma(fit)
summary(gvmodel)
</code></pre>
<h2>GLM Return</h2>
<p>Here's where we're going to discuss the return of the GLM.</p>
<h3>On broad linear models</h3>
<p>One of the broad linear models (Generalized Linear Model) is the promotion of normal linear models, which require that the response variable is only dependent on interpretation of the variable in linear form. It is understandable here that GLM retains the structure of the linear projection sub-program, but links the response variable expectations to the explanation variable linear combination with a connecting function, while allowing the response variable to come from the index distribution group.</p>
<p>R directly provides the function of matching and calculating broad linear models<code>glm( )</code>, it is called in the format</p>
<pre><code class="language-r">log&lt;-glm(formula, family=family.generator,data=data.frame)
</code></pre>
<ul>
<li><code>formula</code>For the purpose of formulating formulae, the meaning is the same as that of linear models;</li>
<li><code>family</code>For the distribution group, including normal distribution, bi-biNMial, porpoise distribution and gamma distribution, the distribution group can also specify the connecting functions to be used by using option link=, refer to the table below</li>
<li>Data is the data box.</li>
</ul>
<p>The common distribution family and default connection functions in GLM can be understood by following the following table:</p>
<table>
<thead>
<tr>
<th>Distribution Type</th>
<th>Default Connect Functions</th>
<th>Common model</th>
</tr>
</thead>
<tbody><tr>
<td><code>binomial</code></td>
<td><code>logit</code></td>
<td>Logit Return</td>
</tr>
<tr>
<td><code>gaussian</code></td>
<td><code>identity</code></td>
<td>Normal linear regression</td>
</tr>
<tr>
<td><code>gamma</code></td>
<td><code>inverse</code></td>
<td>Gamma GLM</td>
</tr>
<tr>
<td><code>inverse.gaussian</code></td>
<td><code>1/mu^2</code></td>
<td>Anti-Gorgos GLM</td>
</tr>
<tr>
<td><code>poisson</code></td>
<td><code>log</code></td>
<td>Porcelain returns.</td>
</tr>
<tr>
<td><code>quasi</code></td>
<td><code>identity</code> Difference with constant</td>
<td>Aquarified GLM</td>
</tr>
<tr>
<td><code>quasibinomial</code></td>
<td><code>logit</code></td>
<td>Paradispersion Logit</td>
</tr>
<tr>
<td><code>quasipoisson</code></td>
<td><code>log</code></td>
<td>Accurate Porcelain</td>
</tr>
</tbody></table>
<p>A few examples of the way in which the distributional community is called are given below.</p>
<pre><code class="language-r">#正态分布 恒等连接
fm &lt;- glm(formula, family = gaussian(link = identity), data = data.frame)

#二项分布 logit连接 是logistics回归的形式
log&lt;-glm(formula, family = binominal(link = logit),data = data.frame)

#Possion分布
log&lt;-glm(formula, family = poisson(link = log),data = data.frame)

#Gamma分布
log&lt;-glm(formula, family = gamma(link = inverse),data = data.frame)
</code></pre>
<h3>More Reference Functions</h3>
<p>The extended Logistic returns and variants in R are as follows: Icon</p>
<ul>
<li>The glmRob() function in the robust Logistic returns robust package can be used to develop broadly linear models that are robust, including robust Logistic returns. When the proposed Logistic regression model data are isolated and strongly influenced, a robust Logistic return can be useful.</li>
<li>Multiple distribution regressions can be combined with multiple Logistic returns using the mlogit() function in the mlogit package if the response variable contains more than two disorderly categories (e.g. married/widow/divorce).</li>
<li>The Logistic returns if the response variable is an orderly group (e.g., credit risk is differential/good/good), the lrm() function in the rms package is used to combine the Logistic returns.</li>
</ul>
<p>R provides some useful extensions to the basic porcelain regression model</p>
<ul>
<li>We have a processing habit of converting to the following formulation model.&#36;\log_\mathrm{e}\left(\frac{\lambda}{time}\right)=\beta_0+\sum_{j=1}^p\beta_jX_j&#36;</li>
<li>Zeroinfol() function in pscl package allows zero-inflating borose return</li>
<li>The glm Rob() function in the robust package can be designed to match a robust broad linear model with a robust permersal pine regression</li>
</ul>
<h2>Non-linear regression model</h2>
<h3>Internal online sex return</h3>
<p>The most classic internal online regression is multi-return; the normal multi-returnal deformation that can be solved by the multi-linear regression method described above, which we will now present in the next section is the multi-dimensional calculation method.</p>
<p>Function form is</p>
<pre><code class="language-r">poly(x, ..., degree = 1, coefs = NULL)
#计算正交多项式 degree是阶数
</code></pre>
<h3>Inlinear nonlinear regression</h3>
<p>This is about the most optimized questions. The functions are not unique, one introduction, we create what we think is a regression function, then optimize the acquisition of parameters.</p>
<pre><code class="language-r">nls(formula, data = parent.frame(), start, control = nls.control(), algorithm = &quot;default&quot;, trace = FALSE, subset, weights, na.action, model = FALSE)
#nls函数对于实现内在非线性回归非常的实用

nlm(f, p, hessian = FALSE,
typsize=rep(1, length(p)), fscale=1, print.level = 0, ndigit=12, gradtol = 1e-6, stepmax = max(1000 * sqrt(sum((p/typsize)^2)), 1000), steptol = 1e-6, iterlim = 100, check.analyticals = TRUE, ...)
#nlm函数也可以处理这个问题 当然它本身就是用来处理最优化问题的 所以需要转化问题形式
</code></pre>
<p>There are examples.</p>
<pre><code class="language-r">cl&lt;-data.frame(
X=c(rep(2*4:21, c(2, 4, 4, 3, 3, 2, 3, 3, 3, 3, 2, 3, 2, 1, 2, 2, 1, 1))), Y=c(0.49, 0.49, 0.48, 0.47, 0.48, 0.47, 0.46, 0.46, 0.45, 0.43, 0.45, 0.43, 0.43, 0.44, 0.43, 0.43, 0.46, 0.45, 0.42, 0.42, 0.43, 0.41, 0.41, 0.40, 0.42, 0.40, 0.40, 0.41, 0.40, 0.41, 0.41, 0.40, 0.40, 0.40, 0.38, 0.41, 0.40, 0.40, 0.41, 0.38, 0.40, 0.40, 0.39, 0.39))
nls.sol&lt;-nls(Y~a+(0.49-a)*exp(-b*(X-8)), data=cl, start = list( a= 0.1, b = 0.01 ))
summary(nls.sol)

fn&lt;-function(p, X, Y){
f &lt;- Y-p[1]-(0.49-p[1])*exp(-p[2]*(X-8))
res&lt;-sum(f^2)
f1&lt;- -1+exp(-p[2]*(X-8))
f2&lt;- (0.49-p[1])*exp(-p[2]*(X-8))*(X-8)
J&lt;-cbind(f1,f2)
attr(res, &quot;gradient&quot;) &lt;- 2*t(J)%*%f
res
}
#建立最优化函数
out&lt;-nlm(fn, p=c(0.1, 0.01), X=cl&#36;X, Y=cl&#36;Y, hessian=TRUE); out
</code></pre>
<h2>Multi-statistical analysis</h2>
<p>We're here to show how common methods in multiple statistical analysis can be achieved in R.</p>
<h2>Main ingredient distribution and factor analysis</h2>
<p>As two classic techniques of relief, we are here to present their R-realization, actually, in a very common way. So we could have put it together.</p>
<h3>Base R-provided algorithm</h3>
<h4>Main ingredient analysis</h4>
<pre><code class="language-r">#PCA的计算
princomp(x, cor = FALSE, scores = TRUE, covmat = NULL,subset = rep(TRUE, nrow(as.matrix(x))), ...)

#提取主成分信息
summary(object, loadings = FALSE, cutoff = 0.1, ...)

#分析载荷矩阵
loadings(x)

#预测新数据主成分的值
predict(object, newdata, ...)

#绘制主成分的碎石图
screeplot(x, npcs = min(10, length(x&#36;sdev)),type = c(&quot;barplot&quot;, &quot;lines&quot;), main = deparse(substitute(x)), ...)

#绘制数据关于主成分的散点图
biplot(x, choices = 1:2, scale = 1, pc.biplot = FALSE, ...)
</code></pre>
<ul>
<li><code>x</code>It's data for the main ingredient analysis.</li>
<li><code>cor</code> T and F determine whether to use the sample for the main component analysis or the matrix for the main ingredient analysis.</li>
</ul>
<h4>Factor analysis</h4>
<p>The function of the factor analysis is</p>
<pre><code class="language-r">factanal(x, factors, data = NULL, covmat = NULL, n.obs = NA,
         subset, na.action, start = NULL, scores = c(&quot;none&quot;, &quot;regression&quot;, &quot;Bartlett&quot;),
         rotation = &quot;varimax&quot;, control = NULL, ...)
</code></pre>
<ul>
<li><code>x</code>is the data, as expressed in the data box</li>
<li><code>factors</code>Meaning factor</li>
<li><code>scores</code>This means that you have to use the factor score.</li>
<li><code>rotation = &quot;varimax&quot;</code>This means rotate with the maximum variance</li>
</ul>
<p>The analysis of the factor analysis is basically the same as that of the main ingredient, with differential contribution rates, and the load matrix, which is available for analysis, has all been stored in a list form for subsequent analysis.</p>
<h3>Functions provided by psych packages</h3>
<p>The function here has a slightly higher degree of freedom and a slightly more detailed information than the underlying R, and they combine the function system.</p>
<pre><code class="language-r">#含多种可选的方差旋转方法的主成分分析
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
</code></pre>
<h2>Disaggregation</h2>
<p>The most common method of diagnosing is distance and Fisher.</p>
<h3>Fisher says goodbye.</h3>
<p>It's basically a linear LDA method of classification, which is generally called Fisher's method in multiple statistics, and it's a function of LDA.</p>
<p>The function is as follows:</p>
<pre><code class="language-r">lda(formula, data, ... , subset, na.action)
</code></pre>
<ul>
<li>Formula is the formula for the sorting of the variable that is described as the return of the source to the classification.</li>
<li>Subset indicates training samples</li>
</ul>
<p>Use iris data sets as an example</p>
<pre><code class="language-r">data(iris)
attach(iris)
names(iris)
library(MASS)
iris.lda &lt;- lda(Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width)
iris.lda
iris.pred=predict(iris.lda)&#36;class
#用于预测 此时预测的是训练用数据集
table(iris.pred, Species)
detach(iris)
</code></pre>
<p>The final predictive matrix shows the difference and the original situation.</p>
<h3>Distance-segment</h3>
<p>We know that the core of distance determination is the calculation of distance, and so R does not design a function for distance separation, but there are functions that help us calculate the distance of the horse.</p>
<pre><code class="language-r">mahalanobis(x, center, cov, inverted=FALSE, ...)
</code></pre>
<p>It accepts a data box as input, and center is the data centre cov is the matrix of the co-ordinated array, which is output in the form of a matrix that reflects the distance between the two elements.</p>
<h2>Cluster analysis</h2>
<h3>System Cluster</h3>
<p>There are very simple functions for system cluster analysis that can help us do this.</p>
<pre><code class="language-r">#计算距离矩阵使用
dist(x, method = &quot;euclidean&quot;, diag = FALSE, upper = FALSE, p = 2)

#计算聚类结果用
hclust(d, method = &quot;complete&quot;, members=NULL)

#绘制聚类图 它有着聚类专用形式为
plot(object, hang=-1)

#用来对聚类结果进行切割 给出我们需要的类个数或者高度就可以了
plclust(object, hang=-1)
rect.hclust(tree, k = NULL, which = NULL, x = NULL, h = NULL,border = 2, cluster = NULL)

#聚类结果转化为树状的谱系图 使用plot绘制
as.dendrogram(object, hang = -1, ...)
</code></pre>
<ul>
<li>The D is the distance structure, the method is the choice of the system cluster method, the default maximum distance.</li>
<li>x is the data box</li>
<li>Method is the method of calculating distance.</li>
<li>The diag andupper logical variables control whether the output is only diagonal or the output is up triangle.</li>
</ul>
<h3>Dynamic Cluster</h3>
<p>The classic dynamic cluster method is k-means.</p>
<pre><code class="language-r">kmeans(x, centers, iter.max = 10, nstart = 1,
algorithm = c(&quot;Hartigan-Wong&quot;, &quot;Lloyd&quot;, &quot;Forgy&quot;, &quot;MacQueen&quot;))
</code></pre>
<ul>
<li>x is the data box</li>
<li>Canters are a group number or an initial cluster centre.</li>
<li>It's the maximum number of words.</li>
<li>Algorithm is an algorithm for dynamic clustering.</li>
</ul>
<h3>Number of clusters</h3>
<p>One of the most desirable types of clustering is the number of types we need to group samples, which, if too few, may place data of a serious heterogeneity in the same group, or if too many, may not be fully classified, and the real cluster scenario may well require some knowledge of the field. We're here to present some stable methods of selecting the number of categories.</p>
<p><strong>NbClust</strong>The package provides a large number of indicators to determine the optimal number of categories in a cluster analysis. There is no guarantee that the results of these indicators will be consistent. In fact, they may be different. However, the results can be used as a reference for selecting the group of K-digit values.<code>NbClust()</code>The function input includes a matrix or data frame that needs to be used for a cluster, the distance measure and the group method used, and the number of the smallest and largest grouping to be used for a cluster. It returns each cluster index, and it also outputs the optimal number of recommended clusters.</p>
<p>An example of a code is</p>
<pre><code class="language-r">library(NbClust)
nc &lt;- NbClust(nutrient.scaled, distance=&quot;euclidean&quot;,
                  min.nc=2, max.nc=15, method=&quot;average&quot;)
#返回结果包含了各种指标决定的聚类类别个数，以及他们的投票结果
</code></pre>
<h2>Typical relevant analysis</h2>
<p>The typical relevant analysis is a multi-dimensional statistical approach to the relationship between the two sets of variables, and the only way we can learn to express the correlation between the two groups in as simple a form as possible.</p>
<p>function in the</p>
<pre><code class="language-r">cancor(x, y, xcenter = TRUE, ycenter = TRUE)
</code></pre>
<ul>
<li>Where x y is a data matrix of two variables</li>
</ul>
<p>Results include:</p>
<ul>
<li>Typical correlation factor</li>
<li>The payload factor used to construct typical correlations</li>
</ul>
<p>The typical relevance of the procedure is also the hypothetical test.</p>
<h2>Corresponding analysis</h2>
<p>Function is in the form of a MASS package</p>
<pre><code class="language-r">corresp(x, nf = 1, ...)
</code></pre>
<ul>
<li>x is the form of the data matrix.</li>
<li>The nf is the factor.</li>
</ul>
<p>One simple example is</p>
<pre><code class="language-r">x.df=data.frame(HighlyFor=c(2, 6, 41, 72, 24), For =c(17, 65, 220, 224, 61), Against=c(17, 79, 327, 503, 300), HighlyAgainst=c(5, 6, 48, 47, 41))
rownames(x.df)&lt;-c(&quot;BelowPrimary&quot;, &quot;Primary&quot;, &quot;Secondary&quot;, &quot;HighSchool&quot;,&quot;College&quot;)
biplot(corresp(x.df, nf=2))
#最后这是绘制了对应分析图 怎么分析我们在理论研究的时候证明过了
</code></pre>
