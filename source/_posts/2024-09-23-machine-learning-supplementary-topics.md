---
title: "机器学习补充主题：多分类、类别不平衡与聚类评估"
title_en: "Machine Learning Supplementary Topics: Multiclass Learning, Class Imbalance, and Clustering Evaluation"
date: 2024-09-23 22:22:06 +0800
categories: ["Data Science & Statistics", "Applied Machine Learning & AutoML"]
tags: ["Learning Notes", "Machine Learning", "Clustering", "Class Imbalance"]
author: Hyacehila
excerpt: "整理多分类学习、类别不平衡问题，以及聚类模型的外部、内部和相对性能度量。"
excerpt_en: "Covers multiclass learning, class imbalance, and external, internal, and relative evaluation metrics for clustering."
mathjax: true
hidden: true
permalink: '/blog/2024/09/23/machine-learning-supplementary-topics/'
---
我们讨论一些相对独立的机器学习知识，他们很重要，值得被单独挑出来进行一些研究。
## 多分类学习
对于多分类学习 我们之前没有进行过系统性的介绍 在Logit回归中介绍了多项式Logit回归简单介绍了一点 但是是很不全面的 这里我们详细的解释如何从最常用的二分类模型构建多分类问题模型

我们的核心思想是 **把多分类问题拆解成多个二分类问题来处理**
#### One vs. One(O v O)
把$N$分类数据集进行二分类配对 产生$N(N-1)/2$ 个分类器 分别把数据放入分类器中进行训练 最后对于我们要进行的分类任务：新样本将同时提交给所有分类器，于是我们将得到$N(N-1)/2$个分类结果,最终结果通过投票产生
#### One vs. Rest(OvR)
每次将一个类的样例作为正例、所有其他类的样例作为反例来

训练$N$个分类器.在测试时若仅有一个分类器预测为正类，则对应的类别标记作为最终分类结果，所示.若有多个分类器预测为正类，则通常考虑各分类器的预测置信度，选择置信度最大的类别标记作为分类结果
#### Many vs. Many (MvM)
MvM是绛次将若干个类作为正类,若干个其他类作为反类 容易看出 前面两个方法都是它的特例 容易看出 我们需要一种方法来构造正反类了 最常用的技术为 纠错输出码（Error Correcting Output Codes,简称 ECOC）

它有着一定的纠错能力

ECOC的过程主要分类两步
* 编码：对 N 个类别做河次划分，每次划分将一部分类别划为正类，一部分划为反类,从而形成一个二分类训练集；这样一共产生M 个训练集,可训练出M 个分类器
* 解码：M 个分类器分别对测试样本进行预测，这些预测标记组成一个编码.将这个预测编码与每个类别各自的编码进行比较,返回其中距离最小的类别作为最终预测结果

一种常用的编码矩阵 (coding matrix)形式为
![多分类编码矩阵示意](/assets/images/machine-learning-notes/ml-supplementary-coding-matrix.png)
## 类别不平衡问题
类别不平衡(class-imbalance)就是指分类任务中不同类别的训练样例数目差别很大的情况

在现实的分类学习任务中，我们经常会遇到类别不平衡，例如在通过拆分法解决多分类问题时，即使原始问题中不同类别的训练样例数目相当，在使
用OvR、MvM策略后产生的二分类任务仍可能出现类别不平衡现象

一种最基本的类别不平衡处理方法是再缩放 他通过按照正反例比例来再缩放我们的分类阈值实现  遗憾的是 再缩放的前提是 **训练集是真实样本总体的无偏采样** 这往往并不是真实的（尤其是我们进行了多分类学习调整） 因此我们还有更加一般的处理手段

现有技术大体上有三类做法：第一类是直接对训练集里的反类样例进行“欠采样”(undersampling),即去除一些反例使得正、反例数目接近，然后再进行学习；第二类是对训练集里的正类样例进行“过采样" (oversampling),即增加一些正例使得正、反例数目接近，然后再进行学习

注意 过采样手法不能简单地对初始正例样本进行重复采样，否则会招致严重的过拟合 一般要采用插值手法来得到额外的正例

欠采样法的代表性算法EasyEnsemble 则是利用集成学习机制，将反例划分为若干个集合供不同学习器使用，这样对每个学习器来看都进行了欠采样，但在全局来看却不会丢失重要信息





## 聚类模型的性能度量
聚类是一种相对特殊的机器学习任务类型 我们也需要给出一些略显不同的有效性指标(validity index) 

聚类的目标是什么？ 直观上看，我们希望 “物以类聚”，即同一簇的样本尽可能彼此相似，不同簇的样本尽可能不同.换言之，聚类结果的“簇内相似度”(intra-cluster similarity)高 且 “簇间相似度”(inter-cluster similarity)低.
### 外部指标
顾名思义，外部验证度量假设事先知道准确的或真实的聚类。真实的分簇标签(即外部信息)用于评估一个给定的聚类。通常我们是不知道准确的聚类的；但外部度量可以用于测试和验证不同的聚类方法。

所有外部度量都需要一个 $r\times k$ 的列联表$N$,该表是根据某个聚类$\mathcal{C}$和真实值分划$T$生成的，定义如下：
$$N(i,j)=n_{ij}=|C_i\cap T_j|$$
换句话说，计数值$n_{ij}$代表分簇$C_i$和真实值划分$T_j$所共有的点的数目。

此外，为明确起见， 令$n_i=|C_i|$代表分簇$C_i$中点的数目，$m_j=|T_j|$代表划分$T_j$中点的数目。列联表可以从$T$ 和$\mathcal{C}$在$O(n)$时间内计算出来.

#### 基于匹配的度量
##### 纯度
纯度(purity)量化了一个分簇$C_{i}$中只包含一个划分的实体的程度。换句话说，它度量了每个分簇有多“纯净”。分簇$C_i$的纯度定义为：
$$\mathrm{purity}_i=\frac1{n_i}\max_{j=1}^k\{n_{ij}\}$$
聚类$C$的纯度定义为所有分簇纯度的带权和：
$$\mathrm{purity}=\sum_{i=1}^r\frac{n_i}n\text{purity}_i=\frac1n\sum_{i=1}^r\max_{j=1}^k\{n_{ij}\}$$
其中比例$\frac{n_i}n$表示分簇$C_i$中的点所占的比例。

$C$的纯度越大，说明它与真实值的吻合度越高。纯度的最大值为 1,指每个簇都是仅由一个划分中的点构成的。若$r=k$,则纯度值为1表示一个完美聚类，即分簇与划分一一对应。不过，即使$r>k$,纯度也可能为 1(当每个分簇都是一个标准划分的子集时)。若$r<k$,则纯度不可能为 1,因为至少有一个分簇包含来自多于一个分划的点。

##### 最大匹配
最大匹配(maximum matching)度量选择分簇和划分之间的某个映射， 使得公共点的数目之和最大化(假设给定一个划分，只有一个分簇可以与之匹配)。这与纯度的情况不同。

形式层面来讲，我们将列联表看作一个完全带权二部图$G=(V,E)$,其中每个划分和每个分簇都是一个节点，即$V=\mathcal{C}\cup\mathcal{T}$,且存在一条边 $(C_i,T_j)\in E$,以及权值 $w(C_i,T_j)=n_{ij}$, 对于所有$C_i\in\mathcal{C}$和$T_j\in\mathcal{T}$。

图中的一个匹配(matching)$M$是$E$的一个子集，使得$M$中的边两两不相邻(即没有共同的顶点)。最大匹配度量定义为$G$中的最大权匹配：
$$\text{match}=\arg\max_M\left\{\frac{w(M)}n\right\}$$
其中一个匹配$M$的权值为$M$中所有边的权值之和，即$w(M)=\sum_e\in Mw(e)$

##### F Measure
给定分簇$C_i$,令$j_i$代表包含$C_i$中最多点的划分，即$j_i=\max_j=1^k\{n_{ij}\}$。一个分簇$C_i$的精度(precision)与其纯度相同：
$$\mathrm{prec}_i=\frac{1}{n_i}\max_{j=1}^k\{n_{ij}\}=\frac{n_{ij_i}}{n_i}$$

分簇$C_i$的召回(recall)定义为：

$$\mathrm{recall}_i=\frac{n_{ij_i}}{|T_{j_i}|}=\frac{n_{ij_i}}{m_{j_i}}$$

其中$m_{j_i}=|T_{j_i}|$。它衡量了划分$T_{j_i}$与分簇$C_i$共同拥有的点的比例。

F-measure 是每一个分簇的精度值和召回值的调和平均数。分簇$C_i$的 F-measure 为：
$$F_i=\frac{2}{\frac{1}{\mathrm{prec}_i}+\frac{1}{\mathrm{recall}_i}}=\frac{2\cdot\mathrm{prec}_i\cdot\mathrm{recall}_i}{\mathrm{prec}_i+\mathrm{recall}_i}=\frac{2n_{ij_i}}{n_i+m_{j_i}}$$
聚类$\mathcal{C}$的 F-measure 为各分簇的 F-measure 的均值：
$$F=\frac1r\sum_{i=1}^rF_i$$

他希望在精度和召回之间取得平衡

#### 基于熵的度量
##### 条件熵
一个聚类$C$的熵定义为：
$$H(\mathcal{C})=-\sum_{i=1}^rp_{C_i}\log p_{C_i}$$
其中$p_{C_i}=\frac{n_i}n$是分簇$C_i$的概率。

同样，分划$T$的熵定义为：
$$H(\mathcal{T})=-\sum_{j=1}^kp_{T_j}\log p_{T_j}$$其中$p_{T_j}=\frac{m_j}n$是划分$T_j$的概率。

$T$的分簇熵，即$T$关于分簇$C_i$的相对熵，定义为：
$$H(\mathcal{T}|C_i)=-\sum_{j=1}^k\left(\frac{n_{ij}}{n_i}\right)\log\left(\frac{n_{ij}}{n_i}\right)$$

给定聚类$C$ 分划$T$ 的条件熵定义为
$$\begin{aligned}H\left(T|\mathcal{C}\right)&=\sum_{i=1}^r\frac{n_i}{n}H(\mathcal{T}|C_i)=-\sum_{i=1}^r\sum_{j=1}^k\frac{n_{ij}}{n}\log\left(\frac{n_{ij}}{n_i}\right)\\&=-\sum_{i=1}^r\sum_{j=1}^kp_{ij}\log\left(\frac{p_{ij}}{p_{C_i}}\right)\end{aligned}$$
其中$p_{ij}=\frac{n_{ij}}n$是分簇$i$中的一个点同时也属于划分$j$的概率。

一个分簇中的点越是分散到不同的划分中，条件熵就越大。对于一个完美聚类，条件熵的值为 0,而在最坏情况下条件熵的值为$\log k$。
##### 归一化互信息
互信息(mutual information)研究聚类$C$和分划$T$之间共享的信息量，定义为：
$$I(\mathcal{C},\mathcal{T})=\sum_{i=1}^r\sum_{j=1}^kp_{ij}\log\left(\frac{p_{ij}}{p_{C_i}\cdot p_{T_j}}\right)$$
互信息度量了$\mathcal{C}$和$\mathcal{T}$的联合概率$p_{ij}$和期望联合概率$p_{C_i}\cdot p_{T_j}$ (在独立假设下)之间的相关性。

若$C$和$T$是彼此独立的，则$p_{ij}=p_{C_i}\cdot p_{T_i}$,因此$I(\mathcal{C},T)=0$。不过，互信息没有上界。

展开互信息我们可以得到
$$I(\mathcal{C},\mathcal{T})=H(\mathcal{T})-H(\mathcal{T}|\mathcal{C})I(\mathcal{C})$$
据此我们可以给出归一化互信息（NMI）
$$\mathrm{NMI}(\mathcal{C},\mathcal{T})=\sqrt{\frac{I(\mathcal{C},\mathcal{T})}{H(\mathcal{C})}\cdot\frac{I(\mathcal{C},\mathcal{T})}{H(\mathcal{T})}}=\frac{I(\mathcal{C},\mathcal{T})}{\sqrt{H(\mathcal{C})\cdot H(\mathcal{T})}}$$
他的取值范围在 $[0,1]$ 之间 接近1意味着好的聚类

##### 信息差异
这一指标是基于聚类$C$和真实值分划$T$的互信息及它们的熵，定义如下：
$$\begin{aligned}\mathrm{VI}(\mathcal{C},\mathcal{T})&=(H(\mathcal{T})-I(\mathcal{C},\mathcal{T})+(H(\mathcal{C})-I(\mathcal{C},\mathcal{T}))\\&=H(\mathcal{T})+H(\mathcal{C})-2I(\mathcal{C},\mathcal{T})\end{aligned}$$
信息差异(VI)值为0，当且仅当$C$与$T$相同。因此，VI 值越小，聚类$\mathcal{C}$就越好。

#### 成对度量
对数据集 $D=\{\boldsymbol{x}_1,\boldsymbol{x}_2,\ldots,\boldsymbol{x}_m\}$, 假定通过聚类给出的簇划分为 $\mathcal{C}=\{C_1$, $C_2,\ldots,C_k\}$, 参考模型给出的簇划分为$C^*=\{C_1^*,C_2^*,\ldots,C_s^*\}$.相应地，令$\lambda$ 与$\lambda^*$ 分别表示与$C$ 和$C^*$ 对应的簇标记向量. 我们将样本两两配对考虑，定义
$$\begin{gathered}
a= |SS|,SS=\{(\boldsymbol{x}_{i},\boldsymbol{x}_{j})\mid\lambda_{i}=\lambda_{j},\lambda_{i}^{*}=\lambda_{j}^{*},i<j)\}, \\
b= |SD|,SD=\{(\boldsymbol{x}_{i},\boldsymbol{x}_{j})\mid\lambda_{i}=\lambda_{j},\lambda_{i}^{*}\neq\lambda_{j}^{*},i<j)\}, \\
c= |DS|,DS=\{(\boldsymbol{x}_{i},\boldsymbol{x}_{j})\mid\lambda_{i}\neq\lambda_{j},\lambda_{i}^{*}=\lambda_{j}^{*},i<j)\}, \\
d= |DD|,~DD=\{(\boldsymbol{x}_{i},\boldsymbol{x}_{j})\mid\lambda_{i}\neq\lambda_{j},\lambda_{i}^{*}\neq\lambda_{j}^{*},i<j)\}, 
\end{gathered}$$
其中 SS 表示两模型都在相同簇中的样本对 SD表示前者相同簇 后者不同簇的样本 ，DS与DD也同理解释。


据此 我们可以定义

##### Jaccard
Jaccard 系数(Jaccard Coefficient,简称 JC)
$$\mathrm{JC}=\frac{a}{a+b+c}.$$
完美划分的Jaccard 系数为1
##### Rand 指数
Rand 指数(Rand Index,简称 RI) 
$$\mathrm{RI}=\frac{2(a+d)}{m(m-1)}.$$
其中$m$是总点数，完美划分Rand 指数为1

##### FM 指数
FM 指数(Fowlkes and Mallows Index,简称 FMI) 
$$\mathrm{FMI}=\sqrt{\frac{a}{a+b}\cdot\frac{a}{a+c}}.$$
完美划分FM 指数为1

#### 关联度量 
##### Hubert 统计量的定义
令$X$和$Y$为两个对称$n\times n$矩阵，且$N=\binom n2$。令$x,y\in\mathbb{R}^N$分别代表对$X$和 Y 的上三角元素(不包括主对角线元素)通过线性化得到的向量。令$\mu_X$代表$x$的逐元素均值， 定义为：
$$\mu_X=\frac1N\sum_{i=1}^{n-1}\sum_{j=i+1}^nX(i,j)=\frac1Nx^\mathrm{T}x$$
令$z_x$代表居中的$x$向量，定义为：
$$z_x=x-1\cdot\mu_X$$
其中$1\in R^N$是全 1 向量。同样，令$\mu_Y$代表$y$的逐元素均值，$z_y$为居中的$y$向量。

Hubert 统计量定义为$X$和$Y$的平均逐元素乘积：
$$\Gamma=\frac1N\sum_{i=1}^{n-1}\sum_{j=i+1}^nX(i,j)\cdot\boldsymbol{Y}(i,j)=\frac1N\boldsymbol{x}^\mathrm{T}\boldsymbol{y}$$

归一化 Hubert 统计量定义为$X$和$Y$的逐元素相关度：
$$\Gamma_n=\frac{\sum_{i=1}^{n-1}\sum_{j=i+1}^n(\boldsymbol{X}(i,j)-\mu_X)(\boldsymbol{Y}(i,j)-\mu_Y)}{\sqrt{\sum_{i=1}^{n-1}\sum_{j=i+1}^n(\boldsymbol{X}(i,j)-\mu_X)^2\quad\sum_{i=1}^{n-1}\sum_{j=i+1}^n(\boldsymbol{Y}[i]-\mu_Y)^2}}=\frac{\sigma_{XY}}{\sqrt{\sigma_X^2\sigma_Y^2}}$$

##### 离散 Hubert 统计量
令$T$和$C$为$n\times n$的矩阵，定义如下：
$$\left.\boldsymbol{T}(i,j)=\left\{\begin{array}{ll}1&y_i=y_j,\:i\neq j\\0&\text{其他情况}\end{array}\right.\right.\quad\boldsymbol{C}(i,j)=\left\{\begin{array}{ll}1&\hat{y}_i=\hat{y}_j,\:i\neq j\\0&\text{其他情况}\end{array}\right.$$
同时，令$t,c\in\mathbb{R}^N$分别表示由$T$和$C$的上三角元素(不包括对角线元素)构成的$N$维向量，其中$N=\binom n2$代表不同的点对的数目。最后，令$z_t$和$z_c$代表居中的$t$向量和$c$向量。

离散 Hubert 统计量可以利用公式 (17.14)(令$x=t,y=c$)计算得到：
$$\Gamma=\frac1Nt^\mathrm{T}c=\frac{\mathrm{TP}}N$$

##### 归一化离散 Hubert 统计量
离散 Hubert 统计量的归一化版本即$t$和$c$之间的相关度
$$\Gamma_n=\frac{z_t^\mathrm{T}z_c}{\|z_t\|\|z_c\|}=\cos\theta $$
注意$\mu_T=\frac1Nt^\mathrm{T}t$是属于同一划分($y_i=y_j$)的点对的比例，不论$\hat{y}_i$与$\hat{y}_j$是否匹配。因此，可得：
$$\mu_T=\frac{t^\mathrm{T}t}N=\frac{\mathrm{TP}+\mathrm{FN}}N$$
### 内部指标
非常明显的 外部指标在大多数情况下都没有价值 因为我们没有参考模型可以使用 除非我们是已知真实分类，只是想研究一下聚类算法的性能。内部指标往往依赖于样本间的距离与近似度，因此和[机器学习进阶与无监督学习：谱聚类与图聚类](/blog/2024/04/06/advanced-machine-learning-unsupervised-learning/)联系密切 其中的归一割与模块度可以直接用于性能度量。

考虑样本之间的距离给出下面的定义
$$\begin{aligned}
\mathrm{avg}(C)& =\frac{2}{|C|(|C|-1)}\sum_{1\leqslant i<j\leqslant|C|}\operatorname{dist}(\boldsymbol{x}_{i},\boldsymbol{x}_{j}),  \\
\operatorname{diam}(C)& =\max_{1\leqslant i<j\leqslant|C|}\mathrm{dist}(\boldsymbol{x}_{i},\boldsymbol{x}_{j}),  \\
d_{\min}(C_{i},C_{j})& =\min_{\boldsymbol{x}_{i}\in C_{i},\boldsymbol{x}_{j}\in C_{j}}\mathrm{dist}(\boldsymbol{x}_{i},\boldsymbol{x}_{j}),  \\
d_{\mathrm{cen}}(C_{i},C_{j})& =\mathrm{dist}(\boldsymbol{\mu}_{i},\boldsymbol{\mu}_{j}), 
\end{aligned}$$
四种样本间距离 如下 分别是 簇内样本间中心距离 簇内样本间最远距离 簇间最近距离 簇间中心距离 

#### DB 指数
DB 指数(Davies-Bouldin Index,简称 DBI)  
$$\mathrm{DBI}={\frac{1}{k}}\sum_{i=1}^{k}\max_{j\neq i}\left({\frac{\mathrm{avg}(C_{i})+\mathrm{avg}(C_{j})}{d_{\mathrm{cen}}(\mu_{i},\mu_{j})}}\right)$$
DBI 的值越小越好
#### Dunn 指数
Dunn 指数(Dunn Index,简称 DI) 
$$\mathrm{DI}=\min\limits_{1\leqslant i\leqslant k}\left\{\min\limits_{j\neq i}\left(\frac{d_{\min}(C_i,C_j)}{\max_{1\leqslant l\leqslant k}\operatorname{diam}(C_l)}\right)\right\}.$$
而DI值越大越好.

#### BetaCV
BetaCV 度量是簇内距离均值与簇间距离均值的比值：
$$\mathrm{BetaCV}=\frac{avg(C)}{d_{avg}}$$
BetaCV 值越小，聚类的效果就越好，因为它表示簇内距离平均要小于簇间距离。
### 相对度量
相对度量比较同一个聚类算法的不同参数的聚类性能
#### Calinski-Harabasz(CH)
给定数据集$D=\{x_i\}_{i=1}^n,D$的散度矩阵(scatter matrix)为：
$$S=n\boldsymbol{\Sigma}=\sum_{j=1}^n(\boldsymbol{x}_j-\boldsymbol{\mu})(\boldsymbol{x}_j-\boldsymbol{\mu})^\mathrm{T}$$
其中$\mu=\frac1n\sum_{j=1}^nx_j$是均值，$\Sigma$是协方差矩阵。散度矩阵可以分解为两个矩阵$S=S_W+S_B$,其中$S_W$是簇内散度矩阵，$S_B$是簇间散度矩阵，分别表示为：
$$\begin{aligned}&S_{W}=\sum_{i=1}^k\sum_{x_j\in C_i}(x_j-\mu_i)(x_j-\mu_i)^\mathrm{T}\\&S_{B}=\sum_{i=1}^kn_i(\mu_i-\mu)(\mu_i-\mu)^\mathrm{T}\end{aligned}$$
其中$\mu_i=\frac1{n_i}\sum_{x_j\in C_i}x_j$是分簇$C_i$的均值。

对于一个给定的$k$值，Calinski-Harabasz(CH)方差比定义为：
$$\begin{aligned}CH(k)&=\frac{\mathrm{tr}(S_B)/(k-1)}{\mathrm{tr}(S_W)/(n-k)}\\&=\frac{n-k}{k-1}\cdot\frac{\mathrm{tr}(S_B)}{\mathrm{tr}(S_W)}\end{aligned}$$

其中 tr$(S_W)$和 tr$(S_B)$是簇内散度矩阵和簇间散度矩阵的迹(即对角线元素之和)。

对于一个较好的$k$值，可以预测簇内的散度要相对小于簇间的散度，因此会得到一个较高的 $CH(k)$ 值。另一方面，我们不想要一个很大的$k$值；

因此可以将 CH 值作图，并找到一个较大的增长处  (且其后没有或只有很小的增长)。

#### 分簇稳定性
分簇稳定性背后的主要思想是：从与$D$相同的分布抽样得到的数据集生成的聚类应当是相似或“稳定”的。

分簇稳定性的方法可用于找出一个给定的聚类算法的合适参数值； 本书主要考虑合适的$k$值，即分簇的正确数目。

$D$的联合概率分布通常是未知的。因此，为以相同的分布抽样数据集，我们可以使用一系列方法，包括随机扰动(random perturbation)、子抽样(subsampling)或自助抽样(bootstrap resampling)。我们先考虑自助法(bootstrapping):

通过从$D$抽样(带放回，即允许同一个数据点被选择多次，每个样本$D_i$因此是不同的)生成$t$个大小为$n$的样本。接下来，对每一个样本$D_i$,分别用不同的$k$ 值 ( 从 2 到 $k^\mathrm{max}$)运行相同的聚类算法。

令$C_k(D_i)$表示给定$k$时从样本$D_i$获得的聚类。接下来，该方法用某个聚类函数比较所有聚类对$C_k(D_i)$和$C_k(D_j)$之间的距离。某些外部聚类评估度量可以用作距离度量，例如，令$C=C_k(D_i),T=C_k(D_j)$,反之亦然。根据这些值，我们计算每个$k$值的期望成对距离。最后，使得从再抽样数据集获得的不同聚类的偏差最小的值$k^*$是$k$的最佳选择，因为它对应的稳定性最高。

#### 聚类趋向性
聚类趋向性或可聚类性(clusterability)旨在判断数据集$D$是否存在有意义的分组。这样做通常很难，因为首先很难定义什么是一个分簇，例如分区的、层次式的、基于密度的、 基于图的，等等。

即便确定了分簇的类型，对于一个给定的数据集$D$,依然很难定义一个合适的零模型(null model,即没有任何聚类结构的模型)。此外，即便判定数据是可聚类的，我们依然要面临判断分簇数目的问题。

Hopkins 统计量是一种对空间随机性的稀疏抽样检验。给定一个包含$n$个点的数据集$D$,我们生成$t$个随机子样本$R_i$ (每个子样本包含$m$个点，其中$m\ll n$)。这些样本的数据空间与$D$相同，在每个维度上随机均匀地生成。

此外，我们还直接从$D$中生成$t$个子样本(每个含$m$个点),使用无放回的抽样。令$D_i$代表第$i$个直接子样本。接下来，计算每个$x_j\in D_i$和$D$中每个点之间的最小距离：
$$\delta_{\min}(\boldsymbol{x}_j)=\min_{\boldsymbol{x}_i\in D,\boldsymbol{x}_i\neq\boldsymbol{x}_j}\{\delta(\boldsymbol{x}_j,\boldsymbol{x}_i)\}$$

第$i$对样本$R_i$和$D_i$ Hopkins 统计量(在$d$个维度上)定义：

$$\mathrm{HS}_i=\frac{\sum_{y_j\in\mathbf{R}_i}(\delta_{\min}(\boldsymbol{y}_j))^d}{\sum_{y_j\in\mathbf{R}_i}(\delta_{\min}(\boldsymbol{y}_j))^d+\sum_{\boldsymbol{x}_j\in\boldsymbol{D}_i}(\delta_{\min}(\boldsymbol{x}_j))^d}$$

这一统计量将随机生成的数据点的最近邻分布和$D$中数据点的随机子集的最近邻分布进行比较。若数据具有良好的聚类性，我们期望$\delta_\min(x_j)$要小于$\delta_\min(y_j)\text{,且在这种情况下，HS}_i$ 趋向于 1。

若两个最近邻距离相似，则 HS$_i$取值接近于 0.5,这意味着数据近乎随机且没有明显的聚类性。

最后，若$\delta_\min(x_j)$的值要大于$\delta_\min(y_j)$,则 HS$_i$倾向于 0,这意味着点排斥， 且无聚类。

根据$t$个不同的 HS$_i$值，可以通过计算该统计量的均值和方差来判断$D$是否可聚类。
