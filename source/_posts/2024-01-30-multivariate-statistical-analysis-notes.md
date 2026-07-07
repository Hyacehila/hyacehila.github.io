---
title: "多元统计分析：判别分析、主成分分析与因子分析"
title_en: "Multivariate Statistical Analysis: Discriminant Analysis, PCA, and Factor Analysis"
date: 2024-01-30 23:46:10 +0800
categories: ["Data Science", "Statistical Modeling & Inference"]
tags: ["Statistics", "Multivariate Statistics"]
author: Hyacehila
excerpt: "整理判别分析、主成分分析、因子分析、聚类分析、对应分析和列联表分析。"
excerpt_en: "Covers discriminant analysis, PCA, factor analysis, clustering, correspondence analysis, and contingency table analysis."
mathjax: true
hidden: true
permalink: '/blog/2024/01/30/multivariate-statistical-analysis-notes/'
---
多元统计分析是对数理统计中一元统计的继承和发展

继承在与我们基于多元正态分布 研究抽样分布和统计推断 [多元统计引论](/blog/2024/09/11/multivariate-statistics-introduction-notes/)

发展在多元统计分析由于其多元属性 有着自己单独的研究内容 诸如主成分分析和因子分析代表的降维思路 判别分析和聚类分析代表的分类问题等等 本文

## 判别分析
### 概述
判别分析属于多元统计分类问题中的一部分；在判别分析中我们只解决一类问题：**根据对象的某些特征指标判断他属于已有的哪一个类别**
我们需要先给出一部分对象的特征指标和它所属于的类别，然后判断一些有特征指标但是没有类别指标的对象的类别
判别分析只能处理完全定量的变量，所有的判别分析方法都如此，如果想解决某些定类的问题 我们需要一些其他的处理方法 如Logic回归等 我们这里不做介绍
判别分析除了分类以外 还会有一些其他作用 我们在后面会介绍到；Fisher判别可以体现组间的差异性
### 距离判别法
我们使用马氏距离进行距离判别（单位因此并不影响距离判别分析的进行）
核心思想很简单 **距离越小的变量越应该被分入同一个类中**
#### 只有两类的情况
##### 基础推导
对于组1 我们知道均值为 $\mu_1$ 协方差矩阵为 $\Sigma_1$ 
对于组2 我们知道均值为 $\mu_2$ 协方差矩阵为 $\Sigma_2$ 
此时我们的判别规则为 **$x$到哪一个总体的平方马氏距离更小 就属于哪一类**
也就是
$$\begin{cases}x\in\pi_1,&\text{若 }d^2(x,\pi_1)\leqslant d^2(x,\pi_2)\\x\in\pi_2,&\text{若 }d^2(x,\pi_1)>d^2(x,\pi_2)&\end{cases}$$
当两个总体的协方差矩阵相等的时候 $\Sigma_1=\Sigma_2$ 我们有下面的简化规律 
$$d^{2}\left(x,\pi_{1}\right)-d^{2}\left(x,\pi_{2}\right)= -2\boldsymbol{a}^{}\left(x-\overline{\boldsymbol{\mu}}\right)=-W(x)$$
其中 $\overline{\mu}=(\mu_1+\mu_2)/2$  $a=\sum^{-1}\left(\mu_{1}-\mu_{2}\right)$
判别规则自然修改为
$$\begin{cases}x\in\pi_1,&\text{若}W(x)\geqslant0\\x\in\pi_2,&\text{若}W(x)<0&\end{cases}$$
当协方差矩阵不想等的时候 我们还是用判别函数来表示
$$\begin{aligned}
W(x)& =d^{2}(x,\pi_{1})-d^{2}\left(x,\pi_{2}\right)  \\
&=\left(x-\mu_{1}\right)^{\prime}\Sigma_{1}^{-1}\left(x-\mu_{1}\right)-\left(x-\mu_{2}\right)^{\prime}\Sigma_{2}^{-1}\left(x-\mu_{2}\right)
\end{aligned}$$
判别规则为
$$\begin{cases}x\in\pi_1,&\text{若}W(x)\leqslant0\\x\in\pi_2,&\text{若}W(x)>0&\end{cases}$$
在实际的应用中，我们不知道具体的均值和协方差，这就需要利用到我们再前面预备知识的部分介绍的统计推断知识了 用样本的均值和协方差来代替，并且对他们是否相等进行假设检验


##### 正态组误判概率
只要是判别就是出错的可能 我们希望研究发生误判的概率 在基于正态假设和前面协方差矩阵相等的假设下 我们给出
记误判概率表示为
$$\begin{aligned}P(2\mid1)=&P(W(x)<0\mid x\in\pi_1)\\P(1\mid2)=&P(W(x)\geqslant0\mid x\in\pi_2)\end{aligned}$$
然后令
$$\Delta^2=\left(\boldsymbol{\mu}_1-\boldsymbol{\mu}_2\right)^{\prime}\boldsymbol{\Sigma}^{-1}\left(\boldsymbol{\mu}_1-\boldsymbol{\mu}_2\right)$$
则有
$$P(2\mid1)=P(1\mid2)=\Phi\left(-\frac\Delta2\right)$$
容易看出：两个组的分开越大 误判概率就越小 我们的$\Delta$ 刻画了这一个指标
但是我们该如何度量他们的分开程度？
一个非常自然的思想是对$\mu_1=\mu_2$ 进行假设检验 但是事实上 哪怕我们拒绝相等假设也不意味着可以进行这样的判别
真正可靠的方法是结合具体的情况 根据误判概率的大小来决定是否使用我们的判别分析结果
**对于那些不满足正态分布的数据，统计中有让他们正态化的手法，然后再进行分析，关于正态的判断和分析是非常重要的**
##### 非正态组误判概率
此时对应的就是一些非常一般的分析误判概率的方法了，他们的思想在其他地方都会用到 而不局限于判别分析这里 要深刻理解
###### 回代法
我们代回那些用来进行训练的样本 看看有多少的样本被错误的划分了
令$n(2|1)$为样本中来自$\pi_1$而误判为$\pi_2$的个数，$n(1|2)$为样本中来自$\pi_{2}$而误判为$\pi_{1}$的个数，则$P(2|1)$和$P(1|2)$可估计为 
$$\hat{P}(2\mid1)=\frac{n(2\mid1)}{n_{1}},\quad\hat{P}(1\mid2)=\frac{n(1\mid2)}{n_{2}}$$
这种方法的效果一般并不好 而且是偏低的 但是在训练用样本量非常大的时候 一般就比较可用了
###### 划分训练样本
参考[机器学习补充知识](/blog/2024/09/23/machine-learning-supplementary-topics/) 的“留出法”一节
###### 交叉验证
机器学习领域的基本方法 是对划分训练样本的一种改进 [机器学习补充知识](/blog/2024/09/23/machine-learning-supplementary-topics/) 的“交叉验证法”一节
在判别分析领域 我们一般使用$N$折交叉验证 
#### 多分类的判别分析
我们还是非常自然的 寻找那个马氏距离最小的总体 
平方马氏距离为 
$$d^2\left(x,\pi_i\right)=\left(x-\mu_i\right)^{\prime}\boldsymbol{\Sigma}_i^{-1}\left(x-\boldsymbol{\mu}_i\right).$$
判别规则为
$$x\in\pi_l,\quad\text{若 }d^2(x,\pi_l)=\min_{1\leqslant i\leqslant k}d^2(x,\pi_i)$$
我们还是可以提出一个协方差矩阵是否都相等的问题 它确实会影响我们此时的分析结果 这是因为：如果我们使用协方差矩阵都相等 则会使用全部的样本用来估计一个协方差矩阵 反之我们需要估计每一个类的协方差矩阵 下面我们来阐述他们的差异
当我们认为协方差矩阵都相等的时候
$$d^2(x,\pi_i)=(x-\mu_i)^{\prime}\boldsymbol{\Sigma}^{-1}(x-\boldsymbol{\mu}_i)=x^{\prime}\boldsymbol{\Sigma}^{-1}x-2\left(\boldsymbol{I}_i^{\prime}\boldsymbol{x}+c_i\right)$$
判别规则可以修改为
$$x\in\pi_l\text{,若}I_l^{\prime}x+c_l=\max_{1\leqslant i\leqslant k}(I_i^{\prime}x+c_i)$$
我们称为线性判别函数的判别
当我们认为协方差矩阵不想等的时候
$$x\in\pi_l,\quad\text{若 }d^2(x,\pi_l)=\min_{1\leqslant i\leqslant k}d^2(x,\pi_i)$$
这称为二次判别函数
*记得用样本的协方差，均值来进行估计*

在是否能使用判别分析的方面：我们希望使用多元方差分析来看均值是否充分的分离 当然正如我们前面所说的 研究误判概率才是最安全的 多元方差分析不能向我们证明均值分离充分 只能向我们证明均值的分离还不够充分

在实际分析中 是否认为协方差矩阵相等也是一个问题
在处理现实的问题的时候 协方差矩阵不可能是完全相等的 我们需要考虑的是 哪一种判别函数对我们更有利 我们有下面的习惯
* 样本量较小的情况下 认为协方差矩阵相等
* 在更多的情况下 使用误判概率来分析问题（小样本使用误判概率不可靠）
### 贝叶斯判别
贝叶斯判别基于下面的思想
如果A组比B组容量大很多 那么下一个样本来自A组的主观概率自然上升 也就是利用的先验信息
贝叶斯判别都基于贝叶斯的关于先验与后验的思想 会用到很多关于贝叶斯统计的内容
贝叶斯判别基于后验分布进行研究 对分布的情况要求较多
#### 最大后验概率
设有$k$组$\pi_1,\pi_2,...,\pi_k$,且组 $\pi_i$ 的概率密度为 $f_i(x)$,样品 $x$ 来自组 $\pi_i$ 的先验概率为$p_i,i=1,2,...,k$,满足$p_1+p_2+...+p_k=1$。利用贝叶斯理论，x属于$\pi_i$ 的后验概率
$$P(\pi_i\mid x)=\frac{p_if_i(x)}{\sum_{j=1}^kp_jf_j(x)},$$
这是离散的后验概率计算 贝叶斯统计里面也介绍过了
最大后验概率判别法的判别规则为
$$x\in\pi_l,\quad\text{若 }P(\pi_l|x)=\max_{1\leqslant i\leqslant k}P(\pi_i|x)$$
这个方法不仅给出了所属的组别 还给出了我们置信的概率 这是他的优点

如果所有的组别都是正态的 并且我们对所属的组别都同等的无知 也就是取先验为均匀分布 那么最大后验概率判别等价于我们前面介绍的距离判别
#### 最小期望误判代价法
在贝叶斯统计中我们引入过损失与损失函数的概念 用来衡量决策；事实上，误判概率并不决定一切，部分误判的后果非常的严重 所以我们极力避免其发生 也就是误判的代价
##### 两组的情形
首先我们需要引入代价矩阵的概念 他衡量了我们对这个误判的厌恶程度
![多元统计分析](/assets/images/probability-statistics-notes/multivariate-statistical-analysis-notes-01.png)
我们假设对先验概率 各个组的概率密度都已知  那么有
$$\begin{cases}x{\in}\pi_1\text{,若}\frac{f_1\left(\boldsymbol{x}\right)}{f_2\left(\boldsymbol{x}\right)}\geqslant\frac{c\left(1\mid2\right)\boldsymbol{p}_2}{c\left(2\mid1\right)\boldsymbol{p}_1}\\x{\in}\pi_2\text{,若}\frac{f_1\left(\boldsymbol{x}\right)}{f_2\left(\boldsymbol{x}\right)}<\frac{c\left(1\mid2\right)\boldsymbol{p}_2}{c\left(2\mid1\right)\boldsymbol{p}_1}&\end{cases}$$
这就是我们的最小ECM判别规则 让平均误判代价最低
##### 多组的情况
我们外推的要求对先验概率 各个组的概率密度都已知 
判别规则为$$x\in\pi_{i},\quad\underset{j\neq l}{\text{若}\operatorname*{\sum}_{j=1}^{k}p_{j}c}(l\mid j)f_{j}(x)=\underset{1\leqslant i\leqslant k}{\operatorname*{\min}}\sum_{j\neq i;j=1}^{k}p_{j}c(i\mid j)f_{j}(x)$$
我们不对贝叶斯判别给出更多的介绍  了解即可 它也不是我们应用的重点
### Fisher判别
#### Fisher判别的思想
Fisher判别法是一种基于线性组合的判别方法
他的思路是 $p$维随机向量的少数几种线性组合可以将他们映射为$r$维的数据 我们只需要基于降维后的数据进行判别就可以了 
如果是二维或者三维的结果 我们可以考虑绘制直观的图形 来让我们从直观的感觉上对类进行判别
#### Fisher判别函数
首先我们来研究什么样的线性组合可以最大化的体现出他们的差异 也就是什么样的投影方式可以尽可能的体现组之间的差异
我们的讨论基于各个组协方差相同的情况进行 并且需要数据服从正态分布；
如果假设并不满足 Fisher判别的效果可能会不是很好
下面的记号为
$\pi_i$ 组别    $x_{ij}$ 来自组$i$的某个观测向量   $a$ 常数向量，也就是线性组合系数
以下为新的均值记号
$$\begin{aligned}\bar{y_i}&=\frac{1}{n_i}\sum_{j=1}^{n_i}y_{ij}=a^{\prime}\overline{x}_i\\\bar{y}&=\frac{1}{n}\sum_{i=1}^{k}\sum_{j=1}^{n_i}y_{ij}=\frac{1}{n}\sum_{i=1}^{k}n_i\overline{y}_i=a^{\prime}\overline{x}\end{aligned}$$
推导新的变量有
$y_{ij}$组间平方和 
$$\begin{aligned}\mathrm{SSTR}&=\sum_{i=1}^kn_i(\bar{y}_i-\bar{y})^2=\sum_{i=1}^kn_i(a'\overline{x}_i-a'\overline{x})^2=a'&\textbf{H}a\end{aligned}$$
其中$H=\sum_{i=1}^{k}n_{i}\left(\bar{x_{i}}-\overline{x}\right)\left(\overline{x_{i}}-\overline{x}\right)^{\prime}$
组内平方和有
$$\mathrm{SSE}=\sum_{i=1}^k\sum_{j=1}^{n_i}(y_{ij}-\bar{y_i})^2=\sum_{i=1}^k\sum_{j=1}^{n_i}(a^{\prime}x_{ij}-a^{\prime}\overline{x_i})^2=a^{\prime}\mathrm{E}a$$
其中$E=\sum_{i=1}^{k}\left(n_{i}-1\right)S_{i}=\sum_{i=1}^{k}\sum_{j=1}^{n_{i}}\left(x_{ij}-\overline{x_{i}}\right)\left(x_{ij}-\overline{x_{i}}\right)^{\prime}$、
现在我们可以给出度量组间分离程度的量有
$$\Delta(\boldsymbol{a})=\frac{\mathrm{SST}R}{\mathrm{SS}E}=\frac{\boldsymbol{a}^{\prime}\boldsymbol{Ha}}{\boldsymbol{a}^{^{\prime}}\boldsymbol{Ea}}$$
现在 我们需要做的就是寻找让上式最大的向量$a$ 由于用$ca$代替$a$对上式无影响 因此我们希望我们的$a$拥有单位方差
上式中未知的量需要用样本来进行估计
设$E^{-1}H$的全部非零特征值依次为$\lambda_1\geqslant\lambda_2\geqslant\cdotp\cdotp\cdotp\geqslant\lambda_i>0$
对应的特征向量记为 $t_1,t_{2}...t_s$ 
我们直接给出结论 $a_1=t_1$时 给出的线性函数就是Fisher第一判别函数 就是降维中应该使用的函数
当样本量较大的时候 仅仅使用第一判别函数并不够  后续的判别函数为$a_i=t_i$  我们应该使用更多的判别函数 直到分类效果较好
前$r$个判别函数的方差贡献率为
$$\sum_{i=1}^{r}\lambda_{i}/\sum_{i=1}^{s}\lambda_{i}$$
我们习惯性在贡献率达到百分之75的时候认为效果较好
基于降维的思想 方差贡献率达到的时候 要尽量减少变量的添加
#### 判别函数得分图
前面我们已经介绍了如何确定判别函数实现了降维
利用目测法分析降维后生成的图形（一般是二维或者三维）是Fisher判别最重要的内容 我们可以发现数据的结构 数据的异常情况等等 
给出一个判别函数得分图的例子如下![多元统计分析](/assets/images/probability-statistics-notes/multivariate-statistical-analysis-notes-02.jpg)
从图中我们就能看出 类别1的分类效果很好 类别23之间不容易区分
#### 判别规则
在使用了$r$个判别函数对$s$维数据进行降维后 我们的判别规则实际上非常的简单 就是对$r$维的降维后的多维数据进行距离判别 距离近的就划归那一类中
判别规则就是
$$x\in\pi_l,\quad\text{若}\sum_{j=1}^r(y_j-\bar{y_j})^2=\min_{1\leqslant i\leqslant k}\sum_{j=1}^r(y_j-\bar{y_{ij}})^2$$
这样我们实际上可以看出
距离判别 贝叶斯判别 Fisher判别之间实际上是有内在联系的
* 认为缺少先验信息的贝叶斯判别就是普通的距离判别
* 不进行降维处理的Fisher判别就是普通的距离判别
**距离判别才是判别分析中最核心的部分**
### 逐步判别
逐步判别的思想和逐步回归是一样的 有价值的变量是有限的 让没有价值的变量纳入我们的研究可能对结果有反作用 还是浪费大量的计算资源
我们希望寻找一个可能最优的变量子集
#### 附加信息检验
附加信息检验研究 添加新的变量以后 对我们的判别效果有没有提升 
我们想要检验的是
$$H_0:\text{各组的 }E(\mathbf{x}_2|\mathbf{x}_1)\text{相等},\quad H_1:\text{各组的 }E(\mathbf{x}_2|\mathbf{x}_1)\text{不全相等}$$
也就是在$x_1$已经选用的情况下 添加$x_2$有无效果
检验统计量有
$$\Lambda(x_2|x_1)=\frac{\Lambda(x_1,x_2)}{\Lambda(x_1)}\\\\\Lambda(x_1,x_2)=\frac{|E|}{|E+H|},\quad\Lambda(x_1)=\frac{|E_{11}|}{|E_{11}+H_{11}|}$$
当原假设为真的时候有
$$\Lambda\left(x_{2}\mid x_{1}\right)\text{服从}\Lambda\left(p-r,k-1,n-k-r\right)$$
当然我们并不关系这种情况 我们更关系$p$个变量下的增减 此时有
$$\Lambda=\Lambda(x_p|x_1,x_2,\cdotp\cdotp\cdotp,x_{p-1})=\frac{\Lambda(x_1,x_2,\cdotp\cdotp\cdotp,x_p)}{\Lambda(x_1,x_2,\cdotp\cdotp\cdotp,x_{p-1})}$$
服从$\Lambda\left(1,k-1,n-k-p+1\right)$
直接根据检验统计量进行判断就可以了
当然我们也可以把它转换成$F$统计量进行检验 
#### 变量选择的方法
这里和逐步回归的思路是完全一样的
无论是向前法 向后法 还是逐步回归法 我们需要做的就是不断的判断 剔除or引入变量 
这里不作更多的解释了
## 聚类分析
在聚类分析里 我们还是希望研究关于分类的问题；但是和判别分析不同的是 这时候的分类不是提前确定的 而是根据样本或者变量的特征来确定的

我们不是在判别类别 而是在寻找是否存在内在的类别；Fisher判别的得分图 就是一种直观的聚类手法

聚类分析主要有两种 
* 按照变量对所观察的样本进行分类称为Q型聚类
* 按照样本对多个变量进行分类，则称为R型聚类
两种聚类没有本质区别 我们一般更加关注 **Q型聚类**

### 系统聚类法
我们不可能给出绝对最优的聚类结果，这种方法是列出所有的可能聚类，计算量过于巨大了；我们在后面介绍的聚类方法（动态聚类法和系统聚类法）都是在寻找一种好的 但是未必是最好的聚类方法
#### 系统聚类法的思想
系统聚类法 或者说 层次聚类法，hierarchical clustering method 是通过一系列的合并或者分割来确定最后的分类的 分为聚集的（agglomerative）和分割的（divisive）两种，适用于样品数目，不是非常大的情形

聚集方法的思路是 开始时将 $n$个样品各自作为一类，并规定样品之间的距离和类与类之间的距离，然后将距离最近的两类合并成一个新类，循环这个方法直到结束

分割的方法是 由$n$个样品组成一类开始，按某种最优准则将它分割成两个尽可能远离的子类，再用同样准则将每一子类进一步地分割成两类 循环这个方法直到结束

聚集的方法比分割的方法更为常用 后面我们只介绍它 所有的方法本质上只是距离的定义不同 其余都是对这个思想的一个具象

下面 我们用$d_{ij}$表示两个样品之间的距离 $G_i$表示类 $D_{KL}$表示两个类之间的距离 在最初 每个样品各成一类 没有特殊声明时类与类之间的距离与样品之间的距离相同
#### 最短聚类法
定义类与类之间的距离为两类最近样品间的距离 
$$D_{K,}=\min_{i\in G_{K},j\in G_{L}}d_{ij}$$
称这种聚集系统法为最短距离法或单连接法（single linkage method）
具体的计算细节这里就省略了 只是按照前面的思想循环计算
可以用于变量的聚类
#### 最长距离法
类与类之间的距离定义为两类最远样品间的距离
$$D_{KL}=\max_{i\in G_{K},j\in G_{L}}d_{ij}$$
称这种系统聚类法为最长距离法或完全连接法（complete linkage method）
最长距离法非常容易被一个极端的值影响 它会被一个极端值影响
可以用于变量的聚类
#### 类平均法
类平均法或称平均连接法（average linkage method）有两种定义，一种定义方法是把类与类之间的距离定义为所有样品对之间的平均距离
$$D_{K L}=\frac{1}{n_{K}n_{L}}\sum_{i\in G_{K},j\in G_{L}}d_{ij}$$
另一种定义方法是定义类与类之间的平方距离样品对之间平方距离的平均值$$D_{\mathrm{KL}}^{2}=\frac{1}{n_{\mathrm{K}}n_{L}}\sum_{i\in G_{K},j\in G_{L}}d_{\psi}^{2}$$
由于类平均法对信息的利用更加的充分 所以一般是效果比较好 
可以用于变量的聚类
#### 重心法
类与类之间的距离定义为它们的重心（均值）之间的欧氏距离$$D_{KL}^{2}=d_{x_{K}^{2}x_{L}}^{2}=(\overline{x}_{K}-\overline{x}_{L})^{\prime}(\overline{x}_{K}-\overline{x}_{L})$$
这种系统聚类法称 重心法（centroid method）
重心法在处理异常值方面更稳健 大样本量很好的抵消了一个异常值的影响（除非它异常的太离谱了）
#### 中间距离法
在重心法中，如果两个类大小差异较大，合并后的新类的重心将明显靠近于原先较大类的重心
有时候 我们想避免这种加权 所以我们对重心做下面的修改
$$\boldsymbol{m}_{M}=\frac{1}{2}\left(\overline{\boldsymbol{x}}_{K}+\overline{\boldsymbol{x}}_{L}\right)$$
也就是 新的类的中间值（重心的替代品）是原本类的中间值的均值
后面考虑类之间的距离就是考虑类的中间值之间的距离有
$$D_{KL}^{2}=(\overline{m}_{K}-\overline{m}_{L})^{\prime}(\overline{m}_{K}-\overline{m}_{L})$$
#### 离差平方和法
类中各样品到类重心的平方欧氏距离之和称为（类内）离差平方和
设类$G_K$和$G_L$合并成新类$G_W$，则它们的离差平方和为
$$\begin{gathered}
W_{K}=\sum_{i\in G_{K}}(x_{i}-\overline{x}_{K})^{\prime}(x_{i}-\overline{x}_{K}) \\
W_{L}=\sum_{i\in G_{L}}\left(x_{i}-\overline{x}_{L}\right)^{\prime}\left(x_{i}-\overline{x}_{L}\right) \\
W_{M}=\sum_{i\in G_{M}}\left(x_{i}-\overline{x}_{M}\right)^{\prime}\left(x_{i}-\overline{x}_{M}\right) 
\end{gathered}$$
对固定的类内样品数，它们反映了各自类内样品的分散程度 
如果合并后增加的离差平方和较小 则认为它们较为接近
于是我们定义两个类之间的距离为
$$D_{\bar{k}L}^{2}=W_{M}-W_{K}-W_{L}$$
这种系统聚类法称为离差平方和法或 Ward 方法（Ward's method）
#### 系统聚类方法的统一
我们统一的聚类方法的公式为
$$D_{MJ}^{2}=\alpha_{K}D_{\bar{K}J}^{2}+\alpha_{L}D_{\bar{L}J}^{2}+\beta D_{\bar{K}L}^{2}+\gamma|\left.D_{\bar{K}J}^{2}-D_{\bar{L}J}^{2}\right|$$
不同的聚类方法就是对其中的四个参数进行单独的确定
#### 聚类与图形
如果数据是二维或者三维的 那么我们可以用散点图进行直观的聚类

如果维度更高 我们前面也介绍了 可以使用一些降维手法 如Fisher判别中降维方法 后面介绍的主成分分析 因子分析 把数据的维度降低到二维或者三维后进行主观的聚类

#### 类的个数
我们到底应该分多少类 这是聚类问题中一定要考虑的问题
一般有下面几种方法
* 观测树形图 主观给出
* 观测散点图 主观给出
* 基于离差平方和构造统计量 基本思路就是类内的离差平方和要尽可能的控制小，合并的类如果离差平方和增加的太多就不是好的合并
### 动态聚类法
动态聚类法的特点是 某个样品所属的类别可以改变 

基本思想是：选择一批凝聚点或给出一个初始的分类，让样品按某种原则向凝聚点凝聚，对凝聚点进行不断的修改或迭代，直至分类比较合理或迭代稳定为止。类的个数$k$需先指定一个

我们只介绍一种比较流行的动态聚类法 $k-means$聚类方法 步骤为
1. 选择$k$个样品作初始凝聚点
2. 对所有的样品逐个归类，将每个样品归入凝聚点离它最近的那个类该类的凝聚点更新为这一类目前的均值，直至所有样品都归了类
3. 重复步骤（2），直至所有的样品都不能再分配为止

由于初始点选取的主观性存在 需要一定的经验 或者最好多选取几次初始点看看结果 如果不同初始凝聚点的选择产生明显不同的最终聚类结果，或者迭代的收敛是极缓慢的，那么可能表明没有自然的类可以形成

有时候我们会结合两种聚类方法 先用系统聚类法确定几个类 然后用它们的类的重心来作为聚类点 有时会对结果产生改善

经验表明，聚类过程中的绝大多数重要变化均发生在第一次再分配中

## 主成分分析
Principal Component Analysis （PCA）是一种经典的对数据进行降维处理的方法，是经典的多元统计分析手段 是经典的无监督机器学习的一种

我们会在这里非常详细的介绍PCA这一重要的数据降维算法，整体的叙述分为总体主成分分析和样本主成分分析两个部分 前者侧重于核心的理论研究，后者侧重于算法设计；最后我们介绍一些PCA中的注意事项
### 总体主成分分析
#### 基本思想
统计分析中 样本存在相关性是大样本情况下的几乎必然 这样会导致分析的难度大大增加 因此使用少数的，不相关的变量来代替大量的存在复共线性的变量是统计分析中非常常见的想法

非常自然的 在对**数据完成规范化**后，原本线性相关的数据，进行一次正交变换后就会变成若干个线性无关的新变量表示的数据，他们就被我们称为主成分

主成分可以近似的表示原始数据 选取一些信息保存最多的主成分可以让我们实现降维

主成分分析需要基于线性组合实现降维度，为了保证系数和结果的实际意义，我们需要进行标准化再进行主成分分析

本文“因子分析”一节作为线性组合降维的方法 也是需要先对数据进行标准化的
#### 定义与导出
对于$m$维随机变量$x$ 其协方差矩阵为$\Sigma$ 对于一个随机变量$x$的线性变换
$$y_i=\alpha_i^\mathrm{T}\boldsymbol{x}=\alpha_{1i}x_1+\alpha_{2i}x_2+\cdots+\alpha_{mi}x_m$$
非常容易得到三个基本的性质
$$\begin{aligned}
&E\left(y_{i}\right)=\alpha_{i}^{T} E(x),i=1,2,\cdots,m \\
&var\left(y_{i}\right)=\alpha_{i}^{T}\Sigma\alpha_{i},i=1,2,\cdots,m \\
&cov\left(y_{i},y_{j}\right)=\alpha_{i}^{T}\Sigma\alpha_{j},i=1,2,\cdots,m;
\end{aligned}$$
现在我们给出总体主成分的定义

定义：对于前文给出的线性变换 如果满足下列条件
1. 系数向量$\alpha_i^\mathrm{T}$是单位向量 即$\alpha_i^\mathrm{T}\alpha_i=1$ 
2. $cov(y_{i},y_{j})=0 ～if ～i\ne j$
3. $y_{1}$是$x$的所有线性变换中方差最大的 $y_{2}$是所有和$y_{1}$不相关的$x$的线性变换中方差最大的 以此类推

直接根据这个定义就可以找到各个主成分 只是可能搜索起来有一定的难度
#### 主要性质
##### 核心定理
设$x$是$m$维随机变量 $\Sigma$是他的协方差矩阵 $\lambda_{i}$是协方差矩阵的$m$个特征值并且有$\lambda_{1}\geqslant\lambda_{2}\geqslant\cdots\geqslant\lambda_{m}\geqslant0,$ $\alpha_{i}$是它对应的$m$个单位特征向量 则$x$的第$k$主成分为
$$y_{k}=\alpha_{k}^{T}x=\alpha_{1k}x_{1}+\alpha_{2k}x_{2}+\cdots+\alpha_{mk}x_{m}$$
方差满足
$$\mathrm{var}\left(y_{k}\right)=\alpha_{k}^{T}\Sigma\alpha_{k}=\lambda_{k}$$
这个定理能够让我们快速的实现主成分的运算

**推论**
$m$维随机变量$y$的各个分量分别是$x$的第一 第二 第$m$主成分的充要条件为
* $y=A^{T}x~A$是正交矩阵
* $y$的协方差矩阵是对角矩阵 并且其对角线元素就是$\Sigma$的特征值并且从大到小排列
##### 因子负荷量和载荷
总体主成分的协方差矩阵满足
$$\mathrm{cov}(\boldsymbol{y})=\Lambda=\mathrm{diag}(\lambda_1,\lambda_2,\cdots,\lambda_m)$$
总体主成分$y$的方差之和满足 其中$\sigma_{ii}$是$x_{i}$的方差
$$\sum_{i=1}^m\lambda_i=\sum_{i=1}^m\sigma_{ii}$$
第$k$个主成分$y_{k}$和变量$x_{i}$的相关系数$p(y_{k},x_{i})$称为因子负荷量（factor loading） 体现了某个主成分和某个原始变量的相关关系
$$\rho(y_{k},x_{i})=\frac{\sqrt{\lambda_{k}}\alpha_{ik}}{\sqrt{\sigma_{ii}}},$$
我们把系数$\alpha_{ik}$称为载荷 能看出因子负荷量和载荷基本起到了接近的描述作用
第$k$个主成分$y_{k}$和对应的$m$个原始变量的因子负荷量满足
$$\sum_{i=1}^m\sigma_{ii}\rho^2(y_k,x_i)=\lambda_k$$
$$\sum_{k=1}^{m}\rho^{2}\left(y_{k},x_{i}\right)=1$$
##### 主成分得分
我们计算得到的每一个主成分都是把原本的$n$维数据构建了一个线性组合变为了一个$p$维数据

代入某个观测在$n$个变量上的取值 进入某个主成分的系数的方程；能得到每一个观测在每一个主成分上的值，我们称为其在这个主成分上的得分

使用得分 可以把每个观测排序 或者绘制散点图进行观察

#### 主成分的个数
选择多少个主成分来实现降维的作用是PCA能发挥作用的核心

而选择多少个主成分的核心就是我们在开篇就提到过的问题 用尽可能少的变量保留尽可能多的信息

越靠前的主成分保留了更大的方差 而方差就是信息的体现

现在我们给出每个主成分的方差贡献率的定义 它体现了主成分对总体信息的保留情况
某个主成分对总体的方差贡献率为
$$\begin{aligned}\eta_k=\frac{\lambda_k}{\sum_{i=1}^m\lambda_i}\end{aligned}$$
前k个主成分对总体的方差贡献率为
$$\sum_{i=1}^k\eta_i=\frac{\sum_{i=1}^k\lambda_i}{\sum_{i=1}^m\lambda_i}$$
类似的原理能体现前k个主成分对某个变量的解释度 从因子负荷量入手很好理解
$$\nu_i=\rho^2(x_i,(y_1,y_2,\cdots,y_k))=\sum_{j=1}^k\rho^2(x_i,y_j)=\sum_{j=1}^k\frac{\lambda_j\alpha_{ij}^2}{\sigma_{ii}}$$
### 样本主成分
样本主成分和总体主成分的定义和性质都是完全一样的，只是将前文使用的总体协方差矩阵改写为了样本协方差矩阵

样本主成分有两种基本的求法，一种是沿用前面总体主成分的求法，从特征向量入手计算主成分，另一种则基于SVD奇异值分解进行
#### 基于特征值分解的PCA
所有的计算方法都和前面一样
参见本文“核心定理”一节。

事实上，我们各个主成分的方向满足下面的叙述，设样本协方差矩阵为
$$C=\frac{1}{n-1}X^TX$$
对协方差矩阵进行特征分解有
$$C=V\Lambda V^T$$
$V$中的各个列向量就是主成分的方向，即特征向量
#### 基于SVD奇异值分解的PCA
SVD分解：对数据矩阵$X$直接进行SVD，得到$X=U\Sigma V^T$,其中：
* $V$的列是主成分方向 (与协方差矩阵$X^TX$的特征向量相同)。
* $\Sigma$对角线元素为奇异值，与协方差矩阵的特征值相关($\sigma_i^2=(n-1)\lambda_i$)

优势：SVD无需显式计算协方差矩阵，数值稳定性更高，尤其适用于高维数据。目前更为常用的就是使用SVD进行奇异值分解

SVD的理论叙述参考 [矩阵论](/blog/2024/10/17/matrix-theory-notes/) 的“奇异值分解SVD与极分解”一节
#### 数据重构与低秩近似视角
最佳低秩近似：PCA等价于用SVD的前$k$个奇异值/向量对$X$进行低秩近似：
$$X\approx U_k\Sigma_kV_k^T$$
其中$U_k$、$\Sigma_k$、$V_k$仅保留前$k$个成分。

物理意义：保留方差最大的方向，舍弃噪声或次要成分，实现降维。

### PCA的一些注意事项
这里阐述的注意事项基本只适用于主成分分析 部分思想在多元统计的其他部分也有一定的价值
#### 主成分保留的个数
* 从主观的角度看，我们需要贡献率达到一定的程度，这个值根据主观确定，这是我们最支持的方法
* 只保留贡献率大于平均贡献率的主成分
* 如果我们需要解释主成分的含义，那么保留那些比较适合解释的主成分就好
* 单个主成分可以进行排序，两个或者三个主成分可以作散点图
#### 异常值和样本容量
* 在理论研究上 主成分对观测没有要求；但是为了避免某些极端情况的产生，观测数最好不要小于50 小于五倍的变量数，取较大的一个
* 异常值应该在数据处理的前段被清洗掉
#### 关于时间序列数据
时间序列数据一般不得用于主成分分析，同样的也不用于后面介绍的因子分析；其原因是时间序列数据之间存在相关性的可能性极大，这会导致我们的样本协方差矩阵对总体协方差矩阵的估计是无效的；因此尽量不要在时间序列数据上使用降维手段，我们有专门的时间序列分析可以研究
#### 主成分分析和聚类分析
* 我们可以根据2-3维的主成分的得分图进行主观的聚类
* 使用主成分之间的距离来聚类效果反而不如使用原始变量
* Fisher判别函数比主成分得分图更适合判断聚类的结果是否合适
#### 不同时期的主成分分析
对于同样的一些变量，不同时期的观测结果在主成分分析上的表现往往并不会一样；需要每个时期单独进行研究，不过有时候适当的外推是可行的
#### 定性数据于主成分分析
如果定性数据是有序的定性数据；那么可以转换成定量的来进行主成分分析
完全定类的数据不可以进行主成分分析
#### 主成分综合得分法是不可行的
在部分研究中，有下面一种主成分综合得分法，来给出所有样品（观测）的综合排名；
对 $p$ 个原始变量 $x_1,x_2,....,x_p$,通过主成分分析，取累计贡献率已达较高水平的前$m$ 个主成分$y_1,y_2,...,y_m$,其方差分别为$\lambda_1,\lambda_2,...,\lambda_m$,以每个主成分
$y_i$的贡献率$\alpha_i=\lambda_i/\sum\lambda_i$ 作为权数，构造综合评价函数 
$$F=\alpha_1y_1+\alpha_2y_2+\cdotp\cdotp\cdotp+\alpha_my_m$$
通过计算得到的综合平方给出排名
这个方法看似很有吸引力，但是实际上是完全错误的 有如下的解释
* 主成分可以取相反符号，并不影响他们还是主成分，这样可以产生大量的综合评价结果，我们使用其中的某一个并不合理
* 主成分的线性组合并不像原始变量的线性组合一样有价值；主成分是根据最大变差选择的，往往能给出有意义的解释，但是他们线性组合并不能被解释
* 主成分往往是第一主成分方差最大，这意味着线性组合就是在重复$y_1$的信息，那为什么不直接用$y_1$呢 他还有解释
* 主成分的线性组合究竟含有什么样的信息，我们不得而知，那为什么要使用它

**下一章我们会介绍因子分析，也是一种降维手法，部分教材也给出了因子分析综合评价法，它和主成分分析综合评价法一样的不合理**

## 因子分析
### 引言
#### 基本介绍
因子分析是一种降维手法 和主成分分析的目的相同但是有一定的禅意；
* 主成分分析的主成分可解释度度往往不高，但是因子分析中的因子一定要求可解释
* 因子分析模型的构建需要一些关键的假设没，主成分分析不用
* 主成分是变量的线性组合，变量是因子的线性组合，因子不能表示为变量的线性组合
* 主成分的解是唯一的 而因子的解可以有很多 主要是因为因子旋转
* 主成分不会因其提取个数的改变而变化，但因子往往会随模型中因子个数的不同而变化
#### 因子分析的基本概念
假设目前我们有十个变量的得分，他们可以归结为四个方向上的能力，就可以建立得分和能力之间的因子模型
$$x_{i}=\mu_{i}+a_{i1}f_{1}+a_{i2}f_{2}+a_{i3}f_{3}+a_{i4}f_{4}+\varepsilon_{i},\quad i=1,2,\cdots,10$$
其中的$\mu$是方向得分的均值 $f_i$代表这四个因子 称为公共因子；原始变量都可以表示为因子的不同的线性组合 系数$\alpha_{ij}$称为变量在因子上的载荷 最后的$\varepsilon_{i}$   是不能被四个公共因子解释的部分 称为特殊因子
上面的模型和线性回归与主成分分析有一点接近，他和主成分分析的区别我们已经介绍过了；他和线性回归的区别是：因子是潜变量，它不能被观测
### 正交因子模型
#### 数学模型
因子分析模型的一般形式为
$$\begin{cases}x_1=\mu_1+a_{11}f_1+a_{12}f_2+\cdotp\cdotp\cdotp+a_{1m}f_m+\varepsilon_1\\x_2=\mu_2+a_{21}f_1+a_{22}f_2+\cdotp\cdotp\cdotp+a_{2m}f_m+\varepsilon_2\\\vdots\\x_p=\mu_p+a_{p1}f_1+a_{p2}f_2+\cdotp\cdotp\cdotp+a_{pn}f_m+\varepsilon_p\end{cases}$$
写成矩阵的形式有
$$x=\boldsymbol{\mu}+\boldsymbol{A}f+\boldsymbol{\varepsilon}$$
其中的$f$称为公共因子向量 $\varepsilon$  称为特殊因子向量 $A$称为因子载荷矩阵
当满足假设
$$\begin{aligned}
&E\left(f\right)=0 \\
&E\left(\boldsymbol{\varepsilon}\right)=0 \\
&V\left(f\right)=I \\
&V\left(\boldsymbol{\varepsilon}\right)=\boldsymbol{D}=\mathrm{diag}\left(\sigma_{1}^{2},\sigma_{2}^{2},\cdots,\sigma_{f}^{2})\right. \\
&Cov(f,\boldsymbol{\varepsilon})=E(f\boldsymbol{\varepsilon}^{\prime})=\boldsymbol{0}
\end{aligned}$$
称为正交因子模型
这些假设的要求都是非常自然且合理的
**在因子分析中，特殊因子缺少研究的价值，我们一般只研究公共因子，所以后面公共因子称为因子**
#### 正交因子模型的性质
##### 分解
原始变量的协方差矩阵有下面的分解形式
$$\boldsymbol{\Sigma}=V(Af+\boldsymbol{\varepsilon})=V(Af)+V(\boldsymbol{\varepsilon})=AV(f)\boldsymbol{A}^{\prime}+V(\boldsymbol{\varepsilon})=A\boldsymbol{A}^{\prime}+\boldsymbol{D}$$
这意味着因子载荷矩阵被原始变量之间的协方差决定
如果原始变量被标准化 也就是协方差矩阵写作相关矩阵的性质
$$R=AA^{\prime}+D$$
##### 单位变化
正交因子模型不受单位变化的影响
##### 旋转
因子载荷矩阵经历任何旋转都还是因子载荷矩阵，它不具有唯一性
#### 因子载荷矩阵的统计意义
##### A的元素
不妨假设我们已经进行过了标准化 则有
$$\rho(x_{i},f_{j})=\frac{\mathrm{Cov}(x_{i},f_{j})}{\sqrt{V(x_{i})V(f_{j})}}=\mathrm{Cov}(x_{i},f_{j})=a_{ij}$$
也就是$A$的元素（因子载荷）就是因子和原始变量之间的相关系数
##### A的元素平方和
元素平方和为
$$\mathrm{tr}(AA^{'})=\sum_{i=1}^{p}\sum_{j=1}^{m}a_{ij}^{2}=\sum_{i=1}^{p}h_{i}^{2}$$
它体现了因子最总方差的贡献
对于已经标准化的变量有 因子对方差的贡献率为
$$\sum_{i=1}^ph_i^2/p$$
在正交因子模型中 方差贡献率这个指标可能失灵 但是在应用中我们依然用这个指标研究因子数量的问题 因为理论上的失灵情况是很少发生的
**因子分析本质上还是需要我们在贡献率代表的拟合度和降维的目标中进行权衡**
### 参数估计
对于$x_1...x_n$的$p$维样本 我们对它进行因子分析
能给出均值和协方差矩阵的估计有
$$\overline{x}=\frac1n\sum_{i=1}^nx_i\quad\text{和}\quad\boldsymbol{S}=\frac1{n-1}\sum_{i=1}^n\left(x_i-\overline{\boldsymbol{x}}\right)\left(x_i-\overline{\boldsymbol{x}}\right)^{\prime}$$
如果已经进行了标准化 那么协方差矩阵的估计就是样本相关矩阵$R$
我们知道
$$R=AA^{\prime}+D$$
那么为了建立因子分析模型 我们还需要估计 因子载荷矩阵$A$和特殊方差矩阵$D$

在给出参数估计以后，我们需要对因子进行合理的解释，这种解释是基于我们对要研究的对象的基本机理进行了解，根据因子载荷矩阵给出主观的因子解释，并没有什么可以客观进行的方法
#### 主成分法
不妨设样本协方差矩阵$S$ 特征值$\lambda_i$ 对应的特征向量$t_{i}$ 则有
$$\begin{aligned}\mathbf{S}=&\hat{\lambda}_1\hat{\boldsymbol{t}}_1\hat{\boldsymbol{t}}_1^{\prime}+\cdot\cdot\cdot+\hat{\lambda}_m\hat{\boldsymbol{t}}_m\hat{\boldsymbol{t}}_m^{\prime}+\hat{\lambda}_{m+1}\hat{\boldsymbol{t}}_{m+1}\hat{\boldsymbol{t}}_{m+1}^{\prime}+\cdot\cdot\cdot+\hat{\lambda}_p\hat{\boldsymbol{t}}_p\hat{\boldsymbol{t}}_p^{\prime}\\\approx&\hat{\lambda}_1\hat{\boldsymbol{t}}_1\hat{\boldsymbol{t}}_1^{\prime}+\cdot\cdot\cdot+\hat{\lambda}_m\hat{\boldsymbol{t}}_m\hat{\boldsymbol{t}}_m^{\prime}+\hat{\boldsymbol{D}}=\hat{\boldsymbol{A}}\hat{\boldsymbol{A}}^{\prime}+\hat{\boldsymbol{D}}\end{aligned}$$
因此我们可以给出
因子载荷矩阵的$j$列就是从$S$出发的计算的第$j$个主成分的系数向量再差一个系数$\sqrt{\lambda_i}$  这就可以计算因子载荷矩阵了
我们称其为主成分解
虽然主成分法和主成分分析看起来非常的接近 非常容易混淆
但是他们本质上原理完全不同 主成分法不计算任何主成分 是因子分析中的一种参数估计方法
我们非常自然的思想是针对标准化后的相关矩阵$R$计算因子载荷矩阵
它实际上就是$p$个原始变量和$m$个主成分之间的样本相关矩阵
当然计算原理还是上面的样子
这个结果可以作为主成分分析的输出结果 也可以作为因子分析的输出结果
这也是为什么我们在PCA一处称相关系数为因子负荷量 就是源于因子分析
#### 主因子法
这是对主成分法的修正 我们假设已经进行了标准化 则
$$R=AA^{\prime}+D$$
则有
$$R^{\prime}=R-D=AA^{\prime}$$
我们称为约相关矩阵（此时要求我们先给出一个特殊方差的合适的初始估计）
则计算$R^*$的特征值和特征向量可以给出主因子解
$$\hat{A}=(\begin{array}{c}\sqrt{\lambda_1^*}\hat{t}_1^*,&\sqrt{\lambda_2^*}\hat{t}_2^*,\cdots,&\sqrt{\lambda_m^*}\hat{t}_m^*\end{array})$$
此时可以重新估计特殊方差矩阵 然后循环进入这个迭代流程

对于特殊方差矩阵的初始估计 常用的有
$$\hat{\sigma}_i^2=1/r^{ii}$$
其中的$r^{ii}$ 是相关矩阵逆矩阵的对角线上的元素
### 因子旋转
前面我们提到了参数估计后需要进行主观的因子解释 它过于依赖于经验 有时候很难给出；因子旋转就是为了辅助我们解释因子
因子是否容易解释和因子载荷矩阵的结构密切相关 如果每一行总有元素接近正负1 而其他元素接近0 那么因子就很容易解释
因子旋转就是为了让因子模型变成前面所述的容易解释的形式 
**因子旋转方法有正交旋转和斜交旋转两类，我们这里只讨论正交旋转**
**因子旋转并不是万能的，有时候他能让我们的因子更容易解释，有时候会起到反作用，不能滥用**

对公共因子作正交旋转 
$$f^{*}=T^{\prime}f$$
因子载荷矩阵自然的发生变化有
$$\mathbf{A}^{*}=\mathbf{A}\mathbf{T}$$
非常容易证明
$$\mathrm{tr}(A^{*}A^{*^{\prime}})=\mathrm{tr}(AA^{\prime})$$
也就是我们的旋转对因子贡献率并没有影响（事实上对残差矩阵也没有影响）
现在我们知道因子旋转时没有什么负面影响的，但是如何选取合适的旋转让我们能得到前述的简单结构 或者说接近这种结构 我们介绍一种方法：**最大方差旋转法**
令
$$d_{ij}=\frac{a_{ij}^*}{h_i},\quad\overline{d}_j=\frac1p\sum_{i=1}^pd_{ij}^2$$
则某列的相对方差可以表示为
$$V_j=\frac1p\sum_{i=1}^p(d_{ij}^2-\overline{d}_j)^2$$
所谓的最大方差旋转 就是找到合适的正交旋转矩阵 让
$$V=V_1+V_2+\cdotp\cdotp\cdotp+V_m$$
达到最大化
### 因子得分
因子得分是一种反过来思考的方式
我们知道公共因子作为隐变量 是不可以进行观测的；但是我们想知道变量$x_{i}$在公共因子$f_{i}$上取值
这就是因子得分；
#### 加权最小二乘法
我们可以使用求解线性回归模型的思路来求解因子的近似解
因为$p$个特殊方差并不相等 所以我们不能使用OLS 
把因子得分看作未知系数 把变量的值和我们前面计算得到因子载荷矩阵认为是已知的 对
$$\sum_{i=1}^p\left[x_i-(\mu_i+a_i\hat{f}_1+a_{i2}\hat{f}_2+\cdots+a_m\hat{f}_m)\right]^2/\sigma_i^2$$
进行极小化
最后计算得到的因子的取值就是巴特莱特（Bartlett）因子得分
矩阵形式表示为（我们把因子作为了因变量考虑）
$$\hat{f}=(A^{\prime}D^{-1}A)^{-1}A^{\prime}D^{-1}(x-\mu)$$
需要计算因子得分的时候代入$A,D,\mu,x$就可以了
#### 回归法
假设$\binom f\varepsilon\text{服从(}m+p)\text{元正态分布}$
那么可以根据回归法计算有
$$\hat{f}=A^{\prime}\left(AA^{\prime}+D\right)^{-1}\left(x-\mu\right)$$
$$\hat{f}=(I+A^{\prime}D^{-1}A)^{-1}A^{\prime}D^{-1}(x-\mu)$$
我们可以证明这两个形式是等价的
称为Thompson因子得分
#### 两种因子得分法的效果比较
从无偏性角度考虑 加权最小二乘法效果更好，从有效性的角度考虑，回归法更好；
在实际的应用中我们有时候可以牺牲一定的无偏性，所以回归法的使用更加广泛
**虽然因子得分被写作了变量线性组合的形式，但是这不意味着在理论上因子可以用变量的线性组合表示，相反的，变量是因子的线性组合**
因子得分可以用来评判某个变量在某个因子上的大小，并根据此给出一些判断，正如我们在PCA中比较主成分大小一样
## 对应分析
对应分析是一种研究行列之间关联的一种低维度的图形表示法；它足够的简单，但是足够的直观；


我们这里只介绍二重列联表的情况，多重对应分析研究多重列联表我们以后有机会再研究

**对应分析研究的是列联表，是对列联表相关分析后的一种更加高级的分析，比列联表相关分析能给出更多信息，自然的它也不再属于描述性统计范畴**
### 行轮廓和列轮廓
#### 列联表
列联表是一种研究分类频数的数据展示方式；
我们主要研究的二重列联表会有两个分类方式，每个方式对应若干种类别
一个正常的列联表如下图所示![多元统计分析 1](/assets/images/probability-statistics-notes/multivariate-statistical-analysis-notes-03.png)
其中$p q$是列联表的行列维数 $n_{ij}$是第$i$行$j$列组别的频数 $n_i$ $n_j$是某行或者某列的频数的和 $n$是总频数
#### 对应矩阵
非常自然的 频数除以$n$就是频率 如下所示
![多元统计分析 2](/assets/images/probability-statistics-notes/multivariate-statistical-analysis-notes-04.png)
我们称频率矩阵
$$p=(p_{ij})=(n_{ij}/n)$$
为对应矩阵
每一行或者列的密度和称为行列的密度
#### 轮廓
称
$$r_i^{\prime}=\left(\frac{p_{i1}}{p_i.},\frac{p_{i2}}{p_i.},\cdots,\frac{p_{iq}}{p_i.}\right)=\left(\frac{n_{i1}}{n_i.},\frac{n_{i2}}{n_i.},\cdots,\frac{n_{iq}}{n_i.}\right)$$
是第$i$行的轮廓
它体现了这一行中各个组别占本行的比例
根据行列对称的思想 有第$j$列的轮廓
$$c_j=\left(\frac{p_{1j}}{p_{.j}},\frac{p_{2j}}{p_{.j}},\cdots,\frac{p_{pi}}{p_{.j}}\right)^{\prime}=\left(\frac{n_{1j}}{n_{.j}},\frac{n_{2j}}{n_{.j}},\cdots,\frac{n_{pi}}{n_{.j}}\right)^{\prime}$$
我们可以把行列轮廓组成对应的轮廓矩阵
#### 马赛克图
mosaic plot
一种直观的研究行列轮廓的图 如下所示
![多元统计分析 3](/assets/images/probability-statistics-notes/multivariate-statistical-analysis-notes-05.png)
![多元统计分析 4](/assets/images/probability-statistics-notes/multivariate-statistical-analysis-notes-06.png)
我们能从图中清楚的看到 当我们进入某个类别后 另一个分类标准中各个类别的占比情况
他可以直观的体现出两个分类标准之间存在的某中联系
### 独立性检验和总惯量
#### 独立性检验
对于研究列联表而言 我们一个非常自然的思想是研究行和列之间是否独立
对列联表中行变量和列变量之间的独立性进行检验，其检验统计量为
$$\chi^2=n\sum_{i=1}^p\sum_{j=1}^q\frac{(p_{ij}-p_i.p._j)^2}{p_i.p._j}$$
当独立性的原假设为真，且样本容量n充分大 期望频数（无论什么分类标准，单独看这一个标准的时候，每一个类的变量数量称为期望频数）大于5
检验统计量近似服从自由度为$(p-1)(q-1)$的卡方分布 拒绝规则为
$$\text{若 }\chi^2\geqslant\chi_e^{2\left[\left(p-1\right)\left(q-1\right)\right]}\text{,则拒绝独立性的原假设}$$
越不符合独立，越偏离独立条件 检验统计量的值自然增大 所以采用单侧检验
#### 总惯量
检验表达式中的
$$\sum_{i=1}^p\sum_{j=1}^q\frac{(p_{ij}-p_i.p._j)^2}{p_i.p._j}$$
可作为行、列变量之间关联性的度量，称为总惯量（total inertia）
总惯量还体现了行轮廓或列轮廓的总变差；行和列之间的关联性越强，行（列）轮廓之间的差异性就越大；反之亦然
总惯量为零是一种极端的情形，在实际中几乎不会出现（因为是根据样本数据算得的）
它意味着所有行轮廓相等 所有列轮廓相等 行列之间相互独立，与此同时，某中组合的频率将会是其边缘频率的积（这一句并不严谨，但是理解就好）
如果行变量与列变量相互独立，则（作样本值的）总惯量将接近于零
### 行列轮廓的坐标
我们前面已经介绍的轮廓的概念
参见本文“轮廓”一节。
**它体现了这一行（列）中各个组别占本行的比例**
但是现在的轮廓中分量都是正的 我们尝试把它进行中心化并且分解有
$i$行的轮廓为
$$r_{i}^{\prime}-c^{\prime}=x_{i1}\boldsymbol{b}_{1}^{\prime}+x_{i2}\boldsymbol{b}_{2}^{\prime}+\cdots+x_{ik}\boldsymbol{b}_{k}^{\prime}$$
同理 $j$列为
$$c_j-r=y_{j1}\boldsymbol{a}_1+y_{j2}\boldsymbol{a}_2+\cdotp\cdotp\cdotp+y_{jk}\boldsymbol{a}_k$$
对应的$x~y$是 在某中坐标表示下的坐标；
各行点和列点在第$i$坐标轴上的坐标平方的加权平均都等于$\lambda_{i}^2$，称之第$i$主惯量
它体现的是第$i$坐标轴上的变差；它反映了列联表数据在第$i$维上的信息量，其在对应分析中的角色相当于主成分分析中的第之主成分的方差
这意味着我们可以尝试类似主成分分析中的降维思路 用较少维度的信息体现原本的列联表的全部信息
又因为我们进行了中心化，所以无论是行轮廓对应的点和列轮廓对应的点都有着相同的尺度 所以可以画在同一张坐标图上分析
### 对应分析图
我们前面已经提到了根据总惯量是否达到目标来实现降维然后叠加做图来分析 降维如下图所示
$$\begin{aligned}&r_i^{\prime}-c^{\prime}\approx x_{i1}b_1^{\prime}+\cdots+x_{im}b_m^{\prime},\quad i=1,2,\cdots,p\\&c_j-r\approx y_{j1}a_1+\cdots+y_{jm}a_m,\quad j=1,2,\cdots,q\end{aligned}$$
原则上 $m$只取$1,2,3$ 后面的讨论以$m=2$为核心 其他的同理
#### 对应分析图的构建
第i个中心化的行轮廓$r_i^{\prime}-c^{\prime}$在由$b_1$和$b_2$构成的平面坐标系中的坐标是($x_{i1},x_{i2})$,第 $j$ 个中心化的列轮廓$c_j$一$r$ 在由$a_1$和$a_2$构成的平面坐标系中的坐标是($y_{j1},y_{j2})$ 现将这两个坐标系重叠在一个平面坐标系中，$b_{1}$和$a_{1}$重叠在第一维坐标轴上，具有同一主惯量$\lambda_{1}^{2}$,其对总惯量的贡献率为$\lambda_1^2/\sum_{i=1}^n\lambda_i^2$。$b_2$和$a_2$重叠在第二维上，皆有主惯量$\lambda_i^2$,其贡献率为$\lambda_i^2/\sum_{i=1}^n\lambda_i^2$。前 
$\sum\lambda_i^2$,该值如很大，则说明所作的对应分析图几乎 二维对总惯量的累计贡献率为($\lambda_1^2+\lambda_2^2)$解释了列联表数据的所有变差 
原则上我们要求坐标轴相互垂直来做图 取三维的时候也同理
#### 行（列）点之间的距离
为了看清楚对应分析图中行（列）点之间的欧氏距离到底说明了什么，我们以下给出有关的近似等式 它成立的前提是我们的前两维总惯量贡献率足够大
两个行点之间的距离为
$$\begin{aligned}d_{ij}^{2}\left(r\right)&=\left(x_{i1}-x_{j1}\right)^{2}+\left(x_{i2}-x_{j2}\right)^{2}\\&\approx\left(r_{i}-r_{j}\right)^{\prime}\mathbf{D}_{c}^{-1}\left(r_{i}-r_{j}\right)=\widetilde{d}_{ij}^{2}\left(r\right)\end{aligned}$$
也就是行点之间的欧氏距离体现了两个轮廓之间的卡方距离；根据前面的叙述 这体现了两个行轮廓之间是否相似；也就是选择列的这两个不同的选项的时候，他们对行的影响是一致的
列点的距离同理 体现了两个列轮廓是否相似
至于行点和列点的欧氏距离 缺乏实际的含义
#### 行点和列点相近的含义
如果第$i$个行点和第$j$个列点接近 也就是
$$(x_{i1},x_{i2}){\approx}(y_{j1},y_{j2})$$
如果目前总惯量贡献率足够 则有
$$\frac{n_{ij}-n_i,n,_j/n}{n_i,n,_j/n}\approx\frac{x_{i1}^2}{\lambda_1}+\frac{x_{i2}^2}{\lambda_2}\geqslant0\\\\\frac{n_{ij}-n_i,n,_j/n}{n_i,n,_j/n}\approx\frac{y_{j1}^2}{\lambda_1}+\frac{y_{j2}^2}{\lambda_2}\geqslant0$$
可见，如果一个行点和一个列点相近 ；则表明行、列两个变量的相应类别组合发生的实际频数一般会高于这两个变量相互独立情形下的期望频数，也就意味着该行类别与该列类别相关联；
应用中我们基本上是判断行、列点离原点的远近，从而大致了解其关联性的强弱；
综上所述，一般地，对于相近的行点和列点，它们离原点越远，其关联性就越强，也就是其类别组合的实际频数越是明显高于两变量独立情形下的期望频数。如果它们都在原点附近，则其关联性一般较弱、甚至可能几乎无关联性；
下面是一个对应分析图的例子 我们可以看图了解一下
![多元统计分析 5](/assets/images/probability-statistics-notes/multivariate-statistical-analysis-notes-07.png)
其中用数字标准的是行点 用字母标注的是列点
根据行（列）点之间的距离可以分析他们的轮廓是否接近
行点和列点接近并且远离原点则意味着行列之间关联性比较强



## 轮廓分析
对同一个单元进行$p$中处理（或者在$p$个时间段内分别测量）依次可以得到$p$组测量数据，也就对应着$p$个均值  
如下图 我们用之间把均值依次连接起来 称为总体的轮廓图
![多元统计分析](/assets/images/probability-statistics-notes/multivariate-statistical-analysis-notes-08.jpg)
轮廓分析研究一个轮廓自己的分析和多个轮廓的比较；

下面我们介绍单总体的轮廓分析和两个总体的轮廓分析 更多总体的轮廓分析这里不做介绍
### 单总体轮廓分析
对于单总体的轮廓分析 我们总是想研究轮廓是否是水平的 也就是假设
$${H_0:\mu_1=\mu_2=\cdot\cdot\cdot=\mu_p,\quad H_1:\mu_i\neq\mu_j,\text{至少存在一对 }i\neq j}{}$$
令
$$\left.C=\left(\begin{array}{rrrrrr}1&-1&0&\cdots&0\\1&0&-1&\cdots&0\\\vdots&\vdots&\vdots&&\vdots\\1&0&0&\cdots&-1\end{array}\right.\right)$$
原问题表示为
$H_{0}:C\mu=0$, $H_{1}:C\mu\neq0$
原假设成立的时候 构造检验统计量为
$$T^2=n\bar{\mathbf{x}}^{\prime}\mathbf{C}^{\prime}\left(\mathbf{CSC}^{\prime}\right)^{-1}\mathbf{C}\overline{\mathbf{x}}$$
拒绝规则为
$$\text{若 }T^2\geqslant T_a^2(p-1,n-1),\text{则拒绝 }H_0$$
其中
$$T_a^2(p-1,n-1)=\frac{(p-1)(n-1)}{n-p+1}F_a(p-1,n-p+1)$$
### 两总体轮廓分析；
我们希望研究三类问题：
* 两个总体的轮廓是否有着接近的形状（平行）
* 如果平行 他们是否重合
* 如果重合 他们是否水平
#### 对于第一个问题有
假设可以被写成
$$\left.H_{01}:\left[\begin{matrix}\mu_{12}-\mu_{11}\\\mu_{13}-\mu_{12}\\\vdots\\\mu_{1p}-\mu_{1,p-1}\end{matrix}\right.\right]=\left[\begin{matrix}\mu_{22}-\mu_{21}\\\mu_{23}-\mu_{22}\\\vdots\\\mu_{2p}-\mu_{2,p-1}\end{matrix}\right]$$
令
$$\left.\boldsymbol{C}=\left[\begin{array}{ccccc}-1&1&0&\cdots&0\\0&-1&1&\cdots&0\\\vdots&\vdots&\vdots&&\vdots\\0&0&0&\cdots&1\end{array}\right.\right]$$
则假设为
$$H_{01}:\boldsymbol{C}\boldsymbol{\mu}_1=\boldsymbol{C}\boldsymbol{\mu}_2,\quad H_{11}:\boldsymbol{C}\boldsymbol{\mu}_1\neq\boldsymbol{C}\boldsymbol{\mu}_2$$
构造检验统计量
$$\begin{aligned}T^2=\frac{n_1n_2}{n_1+n_2}(\overline{x}-\overline{y})^{\prime}C^{\prime}\left(CS_pC^{\prime}\right)^{-1}C(\overline{x}-\overline{y})\end{aligned}$$
拒绝规则为
$$\text{若 }T^2\geqslant T_a^2(p-1,n_1+n_2-2),\text{则拒绝 }H_{01}$$
#### 对于第二个问题有
假设为
$$H_{02}:1^{'}\boldsymbol{\mu}_{1}=1^{'}\boldsymbol{\mu}_{2},\quad H_{12}:\boldsymbol{1}^{'}\boldsymbol{\mu}_{1}\neq1^{'}\boldsymbol{\mu}_{2}$$
构造检验统计量
$$\begin{aligned}T^2=&\frac{n_1n_2}{n_1+n_2}(\bar{x}-\bar{y})^{\prime}\mathbf{1}(1^{\prime}\mathbf{S}_p\mathbf{1})^{-1}\mathbf{1}^{\prime}(\bar{x}-\bar{y})\\=&\frac{n_1n_2}{n_1+n_2}\frac{\mathbf{1}^{\prime}(\bar{x}-\bar{y})}{\mathbf{1}^{\prime}\mathbf{S}_p\mathbf{1}}&(4.\end{aligned}$$
拒绝规则为
$$\text{若 }T^2\geqslant F_a(1,n_1+n_2-2),\text{则拒绝 }H_{02}$$
#### 对于第三个问题有
原假设为
$$H_{03}:\boldsymbol{C\mu}=0,\quad H_{13}:\boldsymbol{C\mu}\neq0$$
其中
$$\left.\boldsymbol{C}=\left[\begin{array}{rrrrrr}1&-1&0&\cdots&0\\1&0&-1&\cdots&0\\\vdots&\vdots&\vdots&&\vdots\\1&0&0&\cdots&-1\end{array}\right.\right]$$
检验统计量为
$$T^2=(n_1+n_2)\overline{\boldsymbol{z}}^{\prime}\boldsymbol{C}^{\prime}(\boldsymbol{C}\boldsymbol{S}\boldsymbol{C}^{\prime})^{-1}\boldsymbol{C}\overline{\boldsymbol{z}}$$
拒绝规则为
$$\text{若 }T^2\geqslant T_a^2(p-1,n_1+n_2-1\text{),则拒绝 }H_{03}$$
