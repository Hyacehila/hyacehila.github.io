---
title: 'Clustering Model Evaluation: External, Internal, and Relative Metrics'
title_zh: 聚类模型的性能度量：外部、内部与相对指标
date: 2026-07-31 20:00:00 +0800
categories:
- Machine Learning
- Classical Machine Learning
tags:
- Machine Learning
- Clustering
- Evaluation
author: Hyacehila
mathjax: true
hidden: true
excerpt: An overview of external, internal, and relative validity measures for clustering models, covering purity, NMI, VI,
  RI, FMI, Hubert statistics, DBI, Dunn, BetaCV, CH, cluster stability, and the Hopkins statistic.
description: An overview of external, internal, and relative validity measures for clustering models, covering purity, NMI,
  VI, RI, FMI, Hubert statistics, DBI, Dunn, BetaCV, CH, cluster stability, and the Hopkins statistic.
excerpt_zh: 整理聚类模型的外部、内部和相对性能度量，包括纯度、NMI、VI、RI、FMI、Hubert 统计量、DBI、Dunn、BetaCV、CH、聚类稳定性与 Hopkins 统计量。
permalink: /blog/2026/07/31/clustering-model-evaluation/
lang: en
translation_key: 2026-07-31-clustering-model-evaluation
translation_status: machine
translation_source_hash: d5ddfa28be71b33e09cde02ad94123a56da965948edc0428bde848c0e4951523
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<h2>Performance measures for a polygon model</h2>
<p>Clusters are a relatively special type of machine-learning task, and we need to give some slightly different indicators of effectiveness. </p>
<p>What are the goals of clustering? Intuitively, we want to see "integrity" in which the same body sample is as similar as possible, and different groups of samples as possible. In other words, the concentration result is high intra-cluster similarity and low inter-cluster similarity.</p>
<p>See you at the presentation of the cluster algorithm itself.<a href="/en/blog/2024/04/06/advanced-machine-learning-unsupervised-learning/">Machine learning progress and unsupervised learning: spectral and graphic Category</a>。</p>
<h3>External indicators</h3>
<p>By definition, external validation measures assume that the precise or real cluster is known in advance. Real cluster labels (i.e. external information) are used to assess a given cluster category. Usually we don't know the exact grouping; but external measurements can be used to test and validate different grouping methods.</p>
<p>All external measurements need one. &#36;r\times k&#36; List of rows&#36;N&#36;, the table is based on a grouping&#36;\mathcal{C}&#36;Split with Real Value&#36;T&#36;is defined as follows:
&#36;&#36;N(i,j)=n_{ij}=|C_i\cap T_j|&#36;&#36;
In other words, count.&#36;n_{ij}&#36;Representative Division&#36;C_i&#36;and real value split&#36;T_j&#36;Number of points common to all.</p>
<p>Besides, for the sake of clarity,&#36;n_i=|C_i|&#36;Representative Division&#36;C_i&#36;Number of midpoints,&#36;m_j=|T_j|&#36;Representational&#36;T_j&#36;Middle point number. The list can be used to read&#36;T&#36; and&#36;\mathcal{C}&#36;Yes.&#36;O(n)&#36;Count it in.</p>
<h4>Based on a matching measure</h4>
<h5>purity</h5>
<p>The purity quantifys a fraction.&#36;C_{i}&#36;The extent to which only one divided entity is included. In other words, it measures how "purity" each sub-column is. Split&#36;C_i&#36;purity is defined as:
&#36; \\mathrm{purity}<em>i=\frac1{n_i}\max</em>{j=1}^k{n_{ij&#125;&#125;&#36;&#36;
聚类&#36;C&#36;的纯度定义为所有分簇纯度的带权和：
&#36;&#36;\mathrm{purity}=\sum_{i=1}^r\frac{n_i}n\text{purity}<em>i=\frac1n\sum</em>I'm not gonna let you go.
Percentage of&#36;\frac{n_i}n&#36;For a breakup&#36;C_i&#36;the percentage of points in the middle.</p>
<p>&#36;C&#36;The greater the purity, the higher the degree of conformity with the true value. The maximum purity value is 1 and means that each cluster is made up of only one point in the division. If&#36;r=k&#36;, the purity value is 1 to indicate a perfect grouping, i.e. the cluster corresponds to the division. But even if it's &#36;r.&gt;k&#36;,纯度也可能为 1(当每个分簇都是一个标准划分的子集时)。若&#36;r&lt;k&#36;, which cannot be pure, because at least one subset contains more than one split point.</p>
<h5>Maximum Match</h5>
<p>The maximm watching is a map of the selection of the fractions and the dividing, maximizing the number of public points (assuming that one division is given, only one can match it). This is not the case with purity.</p>
<p>Formally, we see the list as a fully-owned part two. Figure&#36;G=(V,E)&#36;Each division and each sub-column is a node, i.e.&#36;V=\mathcal{C}\cup\mathcal{T}&#36;, and there's a side &#36;(C_i,T_j)\in E&#36;and power &#36;w(C_i,T_j)=n_{ij}&#36;,for all&#36;C_i\in\mathcal{C}&#36;and&#36;T_j\in\mathcal{T}&#36;。</p>
<p>Matching in a figure (matching)&#36;M&#36;Yes.&#36;E&#36;A subset that makes&#36;M&#36;The two sides of the equation are not adjacent (i.e. there is no common vertex). Maximum Match Measure is defined as&#36;G&#36;, and then the right match:
&#36;&#36;\text{match}=\arg\max_M\left{\frac{w(M)}n\right}&#36;&#36;
One of them matches.&#36;M&#36;The right value is&#36;M&#36;The sum of the weights of all sides, i.e.&#36;w(M)=\sum_e\in Mw(e)&#36;</p>
<h5>F Measure</h5>
<p>Give a scorer.&#36;C_i&#36;You're...&#36;j_i&#36;Organisation&#36;C_i&#36;The division of the maximum points of the midpoint, i.e.&#36;j_i=\max_j=1^k{n_{ij&#125;&#125;&#36;I'm sorry. A partition.&#36;C_i&#36;The precision (precision) is the same as its purity:
&#36; \\mathrm{prec}<em>i=\frac{1}{n_i}\max</em>{j=1}^k{n_{ij&#125;&#125;=\frac{n_{ij_i&#125;&#125;{n_i}&#36;&#36;</p>
<p>Split&#36;C_i&#36;The recall is defined as:</p>
<p>&#36;&#36;\mathrm{recall}<em>i=\frac{n</em>{ij_i&#125;&#125;{|T_{j_i}|}=\frac{n_{ij_i&#125;&#125;{m_{j_i&#125;&#125;&#36;&#36;</p>
<p>of which&#36;m_{j_i}=|T_{j_i}|&#36;I'm sorry. It measures the division.&#36;T_{j_i}&#36;And the partition.&#36;C_i&#36;Proportion of shared sites.</p>
<p>F-measure is the sum average of the precision and recall values of each fraction. Split&#36;C_i&#36;The F-measure is:
&#36;F i=\frac{1mathrm{prec}+\frac{1mathrm{recall}=cdot\mathrm{prec} cdot\mathrm{recall}<em>i}{\mathrm{prec}<em>i+\mathrm{recall}<em>i}=\frac{2n</em>{ij_i&#125;&#125;{n_i+m</em>{j_i&#125;&#125;&#36;&#36;
聚类&#36;\mathcal{C}&#36;的 F-measure 为各分簇的 F-measure 的均值：
&#36;&#36;F=\frac1r\sum</em>{i=1}^rF_i&#36;&#36;</p>
<p>He wants to balance precision with recall.</p>
<h4>Measure based on entropy</h4>
<h5>Conditional entropy</h5>
<p>A cluster&#36;C&#36;The term entropy is defined as:
&#36;&#36;H(\mathcal{C})=-\sum_{i=1}^rp_{C_i}\log p_{C_i}&#36;&#36;
of which&#36;p_{C_i}=\frac{n_i}n&#36;It's a partition.&#36;C_i&#36;- The probability.</p>
<p>Again, split.&#36;T&#36;The term entropy is defined as:
&#36;&#36;H(\mathcal{T})=-\sum_{j=1}^kp_{T_j}\log p_{T_j}&#36;&#36;of which&#36;p_{T_j}=\frac{m_j}n&#36;It's division.&#36;T_j&#36;- The probability.</p>
<p>&#36;T&#36;The split, which is&#36;T&#36;About the partition&#36;C_i&#36;is defined as:
&#36;&#36;H(\mathcal{T}|C_i)=-\sum_{j=1}^k\left(\frac{n_{ij&#125;&#125;{n_i}\right)\log\left(\frac{n_{ij&#125;&#125;{n_i}\right)&#36;&#36;</p>
<p>Grouping&#36;C&#36; Division&#36;T&#36; The condition is defined as
&#36;&#36;00begin{aligned}H\left (T\mathcal{C}right)&amp;=\sum_{i=1}^r\frac{n_i}{n}H(\mathcal{T}|C_i)=-\sum_{i=1}^r\sum_{j=1}^k\frac{n_{ij&#125;&#125;{n}\log\left(\frac{n_{ij&#125;&#125;{n_i}\right)\&amp;♪ I'm not gonna let you go ♪
of which&#36;p_{ij}=\frac{n_{ij&#125;&#125;n&#36;It's a partition.&#36;i&#36;One of the points is also divided.&#36;j&#36;- The probability.</p>
<p>The more the points in a partition spread into different divisions, the larger the conditions. For a perfect group, the value of the condition entropy is 0, while the value of the conditional entropy in the worst case is 0.&#36;\log k&#36;。</p>
<h5>Normalize mutual information</h5>
<p>Mutual information research grouping&#36;C&#36;and division&#36;T&#36;The amount of information shared between them is defined as:
&#36;&#36;I(\mathcal{C},\mathcal{T})=\sum_{i=1}^r\sum_{j=1}^kp_{ij}\log\left(\frac{p_{ij&#125;&#125;{p_{C_i}\cdot p_{T_j&#125;&#125;\right)&#36;&#36;
I've got information.&#36;\mathcal{C}&#36;and&#36;\mathcal{T}&#36;Joint probability&#36;p_{ij}&#36;And the expectation of a combination of probability.&#36;p_{C_i}\cdot p_{T_j}&#36; Relevance (under independent assumptions).</p>
<p>If&#36;C&#36;and&#36;T&#36;It's independent of each other, then.&#36;p_{ij}=p_{C_i}\cdot p_{T_i}&#36;♪ And so ♪&#36;I(\mathcal{C},T)=0&#36;I'm sorry. However, there is no upper bounds for information.</p>
<p>We can get information from each other.
&#36;&#36;I(\mathcal{C},\mathcal{T})=H(\mathcal{T})-H(\mathcal{T}|\mathcal{C})I(\mathcal{C})&#36;&#36;
So we can give a generic information.
&#36;&#36;\mathrm{NMI}(\mathcal{C},\mathcal{T})=\sqrt{\frac{I(\mathcal{C},\mathcal{T})}{H(\mathcal{C})}\cdot\frac{I(\mathcal{C},\mathcal{T})}{H(\mathcal{T})&#125;&#125;=\frac{I(\mathcal{C},\mathcal{T})}{\sqrt{H(\mathcal{C})\cdot H(\mathcal{T})&#125;&#125;&#36;&#36;
His range is in. &#36;[0,1]&#36; In between, close to one means good cluster.</p>
<h5>Information discrepancies</h5>
<p>This indicator is based on clustering&#36;C&#36;Split with Real Value&#36;T&#36;The information and entropy of these are defined as:
&#36;&#36;00begin{aligned}\mathrm{VI} (\mathcal{C},\mathcal{T}&amp;=(H(\mathcal{T})-I(\mathcal{C},\mathcal{T})+(H(\mathcal{C})-I(\mathcal{C},\mathcal{T}))\&amp;=H(\mathcal{T}+H(\mathcal{C}2I(\mathcal{C},\mathcal{T}\end{aligned} &#36;&#36;
Information difference (VI) value 0, current and only&#36;C&#36;and&#36;T&#36;Same. So, the smaller the VI value, the more the grouping&#36;\mathcal{C}&#36;The better.</p>
<h4>Pair</h4>
<p>Default of &#36;D= {bardsymbol{x}<em>1,\boldsymbol{x}<em>2,\ldots,\boldsymbol{x}<em>m}&#36;, 假定通过聚类给出的簇划分为 &#36;\mathcal{C}={C_1&#36;, &#36;C_2,\ldots,C_k}&#36;, 参考模型给出的簇划分为&#36;C^<em>={C_1^</em>,C_2^<em>,\ldots,C_s^</em>}&#36;.相应地，令&#36;\lambda&#36; 与&#36;\lambda^<em>&#36; 分别表示与&#36;C&#36; 和&#36;C^</em>We'll have the sample paired, define it.
&#36;&#36;00begin{gathered}
A = \SS, =SS = (\bardsymbol{)</em>{i},\boldsymbol{x}</em>{j})\mid\lambda</em>{i}=\lambda_{j},\lambda_{i}^{<em>}=\lambda_{j}^{</em>},i&lt;j)}, \
b= |SD|,SD={(\boldsymbol{x}<em>{i},\boldsymbol{x}</em>{j})\mid\lambda_{i}=\lambda_{j},\lambda_{i}^{<em>}\neq\lambda_{j}^{</em>},i&lt;j)}, \
c= |DS|,DS={(\boldsymbol{x}<em>{i},\boldsymbol{x}</em>{j})\mid\lambda_{i}\neq\lambda_{j},\lambda_{i}^{<em>}=\lambda_{j}^{</em>},i&lt;j)}, \
d= |DD|,~DD={(\boldsymbol{x}<em>{i},\boldsymbol{x}</em>{j})\mid\lambda_{i}\neq\lambda_{j},\lambda_{i}^{<em>}\neq\lambda_{j}^{</em>},i&lt;The blogger says:
I'm sorry, I'm sorry.
In which SS indicates that both models are sampled in the same clusters to SD, the former in the same clusters, and the latter in different clusters, the DS and DD in the same way.</p>
<p>So we can define it.</p>
<h5>Jaccard</h5>
<p>Jaccard coefficient (Jaccard Coefficent, JC)
&#36;&#36;\mathrm{JC}=\frac{a}{a+b+c}.&#36;&#36;
Perfectly divided Jaccard coefficient is 1.</p>
<h5>Rand Index</h5>
<p>Rand Index (Rand Index, RI)
&#36;&#36;\mathrm{RI}=\frac{2(a+d)}{m(m-1)}.&#36;&#36;
of which&#36;m&#36;It's the total point. Perfectly divided Rand index to 1.</p>
<h5>FM Index</h5>
<p>FM Index (Fowlkes and Mallows Index, short FMI)
&#36;&#36;\mathrm{FMI}=\sqrt{\frac{a}{a+b}\cdot\frac{a}{a+c&#125;&#125;.&#36;&#36;
Perfectly divide FM index to 1.</p>
<h4>Link Measures</h4>
<h5>Definition of Hubert statistics</h5>
<p>You!&#36;X&#36;and&#36;Y&#36;Two symmetrys&#36;n\times n&#36;matrix, and&#36;N=\binom n2&#36;I'm sorry. You're the one who's gonna get you.&#36;x,y\in\mathbb{R}^N&#36;- For each other.&#36;X&#36;and Y's upper triangle elements (excluding main diagonal elements) are the vectors obtained by linearization. You're the one who's gonna get you.&#36;\mu_X&#36;Representative&#36;x&#36;, defined as:
&#36;&#36;\mu_X=\frac1N\sum_{i=1}^{n-1}\sum_{j=i+1}^nX(i,j)=\frac1Nx^\mathrm{T}x&#36;&#36;
You!&#36;z_x&#36;Centred&#36;x&#36;Vector, defined as:
&#36;&#36;z_x=x-1\cdot\mu_X&#36;&#36;
of which&#36;1\in R^N&#36;is the full 1 vector. Again, the order.&#36;\mu_Y&#36;Representative&#36;y&#36;The average of the elements by element,&#36;z_y&#36;Centred&#36;y&#36;vector.</p>
<p>Hubert Statistically defined&#36;X&#36;and&#36;Y&#36;Average element-by-component product:
&#36;&#36;\Gamma=\frac1N\sum_{i=1}^{n-1}\sum_{j=i+1}^nX(i,j)\cdot\boldsymbol{Y}(i,j)=\frac1N\boldsymbol{x}^\mathrm{T}\boldsymbol{y}&#36;&#36;</p>
<p>Normalization Hubert Statistically defined&#36;X&#36;and&#36;Y&#36;, and then the following:
&#36;&#36;\Gamma_n=\frac{\sum_{i=1}^{n-1}\sum_{j=i+1}^n(\boldsymbol{X}(i,j)-\mu_X)(\boldsymbol{Y}(i,j)-\mu_Y)}{\sqrt{\sum_{i=1}^{n-1}\sum_{j=i+1}^n(\boldsymbol{X}(i,j)-\mu_X)^2\quad\sum_{i=1}^{n-1}\sum_{j=i+1}^n(\boldsymbol{Y}[i]-\mu_Y)^2&#125;&#125;=\frac{\sigma_{XY&#125;&#125;{\sqrt{\sigma_X^2\sigma_Y^2&#125;&#125;&#36;&#36;</p>
<h5>Disconnected Hubert Statistics</h5>
<p>You!&#36;T&#36;and&#36;C&#36;Yes&#36;n\times n&#36;and the matrix, defined as:
&#36;&#36;\left.\bardsymbol{T}(i,j)=\left{begin{array}ll}1&amp;y_i=y_j,:i\neq j\0&amp;\text{others}\right.\right.\quadd\bardsymbol{C}(i,j)=\left{begin{array}{1&amp;\hat{y}_i=\hat{y}_j,:i\neq j\0&amp;\text{Others}\right.&#36;
Meanwhile,&#36;t,c\in\mathbb{R}^N&#36;Other Organiser&#36;T&#36;and&#36;C&#36;and the upper triangulation elements (excluding diagonal elements)&#36;N&#36;& Vector, where&#36;N=\binom n2&#36;Numbers representing different points. Finally, your orders&#36;z_t&#36;and&#36;z_c&#36;Centred&#36;t&#36;Vector and&#36;c&#36;vector.</p>
<p>Dispersed Hubert statistics can use formula (17.14) You're the one who's gonna get you.&#36;x=t,y=c&#36;) Calculated:
&#36;&#36;\Gamma=\frac1Nt^\mathrm{T}c=\frac{\mathrm{TP&#125;&#125;N&#36;&#36;</p>
<h5>Normalized discrete Hubert statistics</h5>
<p>Dispersed Hubert, the uniform version of statistics is&#36;t&#36;and&#36;c&#36;Relevance between
&#36;&#36;\Gamma_n=\frac{z_t^\mathrm{T}z_c}{|z_t||z_c|}=\cos\theta &#36;&#36;
Attention.&#36;\mu_T=\frac1Nt^\mathrm{T}t&#36;is the same division ((s)&#36;y_i=y_j&#36;) Point-to-point ratio, regardless of&#36;\hat{y}_i&#36;and&#36;\hat{y}_j&#36;Whether it matches. Thus, it is possible to:
&#36;&#36;\mu_T=\frac{t^\mathrm{T}t}N=\frac{\mathrm{TP}+\mathrm{FN&#125;&#125;N&#36;&#36;</p>
<h3>Internal indicators</h3>
<p>And it's obvious that external indicators are in most cases of no value because we don't have reference models that we can use unless we're known to be real classifications, just to study the performance of the cluster algorithm. Internal indicators often depend on the distance between samples and the approximation, and therefore<a href="/en/blog/2024/04/06/advanced-machine-learning-unsupervised-learning/">Machine learning progress and unsupervised learning: spectral and graphic Category</a>Close links, where the integration and modularity can be directly used for performance measurement.</p>
<p>Considering the distance between samples, give the following definition
&#36;&#36;00\
\mathrm{avg}&amp; =\frac{2}{|C|(|C|-1)}\sum_{1\leqslant i&lt;j\leqslant|C|}\operatorname{dist}(\boldsymbol{x}<em>{i},\boldsymbol{x}</em>{j}),  \
\operatorname{diam}(C)&amp; =\max_{1\leqslant i&lt;j\leqslant|C|}\mathrm{dist}(\boldsymbol{x}<em>{i},\boldsymbol{x}</em>{j}),  \
d_{\min}(C_{i},C_{j})&amp; =\min_{\boldsymbol{x}<em>{i}\in C</em>{i},\boldsymbol{x}<em>{j}\in C</em>{j&#125;&#125;\mathrm{dist}(\boldsymbol{x}<em>{i},\boldsymbol{x}</em>{j}),  \
d_{\mathrm{cen&#125;&#125;(C_{i},C_{j})&amp; =\mathrm{dist}(\boldsymbol{\mu}<em>{i},\boldsymbol{\mu}</em>The blogger says:
I'm sorry, I'm sorry.
The four samples are the following: the central distance between the inner samples, the longest distance between the inner samples, the nearest distance between the clusters, the central distance between the clusters. </p>
<h4>DB Index</h4>
<p>DB Index (Davis-Bouldin Index, short DBI)<br>&#36;&#36;\mathrm{DBI}={\frac{1}{k&#125;&#125;\sum_{i=1}^{k}\max_{j\neq i}\left({\frac{\mathrm{avg}(C_{i})+\mathrm{avg}(C_{j})}{d_{\mathrm{cen&#125;&#125;(\mu_{i},\mu_{j})&#125;&#125;\right)&#36;&#36;
The smaller the DBI, the better.</p>
<h4>Dunn Index</h4>
<p>Dunn Index (Dunn Index, DI)
&#36;&#36;\mathrm{DI}=\min\limits_{1\leqslant i\leqslant k}\left{\min\limits_{j\neq i}\left(\frac{d_{\min}(C_i,C_j)}{\max_{1\leqslant l\leqslant k}\operatorname{diam}(C_l)}\right)\right}.&#36;&#36;
And the bigger the D, the better.</p>
<h4>BetaCV</h4>
<p>BetaCV measures the ratio between the intra-clan distance average and the interclan distance average:
&#36;&#36;\mathrm{BetaCV}=\frac{avg(C)}{d_{avg&#125;&#125;&#36;&#36;
The smaller the BetaCV value, the better the effect of the cluster because it means that the inner distance is on average less than the interclause distance.</p>
<h3>Relative measures</h3>
<p>Relative measures compare the group nature of different parameters of the same conglomeration algorithm Yes.</p>
<h4>Calinski-Harabasz(CH)</h4>
<p>The given data set is &#36;D=x i}<em>The D.R. is a scatter matrix:
&#36;S=n\bardsymbol}sum}</em>{j=1}^n(\boldsymbol{x}<em>j-\boldsymbol{\mu})(\boldsymbol{x}<em>^mathrm{
Where \mu=\frac1\sum</em>{j=1}^nx_j&#36;是均值，&#36;\Sigma&#36;是协方差矩阵。散度矩阵可以分解为两个矩阵&#36;S=S_W+S_B&#36;,其中&#36;S_W&#36;是簇内散度矩阵，&#36;S B&#36; is a cluster-wide dispersion matrix, which is indicated as:
&#36;&#36;00\&amp;S</em>{W}=\sum_{i=1}^k\sum_{x_j\in C_i}(x_j-\mu_i)(x_j-\mu_i)^\mathrm{T}\&amp;I'm sorry, I'm sorry.
of which&#36;\mu_i=\frac1{n_i}\sum_{x_j\in C_i}x_j&#36;It's a partition.&#36;C_i&#36;average.</p>
<p>For a given&#36;k&#36;Value, Calinski-Harabasz (CH) variance is defined as:
&#36; \begin{aligned}CH(k)&amp;=\frac{\mathrm{tr}(S_B)/(k-1)}{\mathrm{tr}(S_W)/(n-k)}\&amp;=\frac{n-k}{k-1}\cdot\frac{\mathrm{tr}(S_B)}{\mathrm{tr}(S_W)}\end{aligned}&#36;&#36;</p>
<p>of which&#36;(S_W)&#36;and tr&#36;(S_B)&#36;is the trace of the inner and inter-clave-dispersible arrays (i.e. the sum of the diagonal elements).</p>
<p>For a better one.&#36;k&#36;Value, can predict a relatively small dispersion in the cluster, and therefore a higher dispersion is obtained. &#36;CH(k)&#36; value. On the other hand, we don't want a big one.&#36;k&#36;Value;</p>
<p>Thus, CH values can be mapped and a larger growth area found (and no or only small growth thereafter).</p>
<h4>Division stability</h4>
<p>The main idea behind the stability of the divide is to be able to&#36;D&#36;The clustering of data sets from the same distributed sample should be similar or “stable”.</p>
<p>The method of partition stability can be used to find the appropriate parameter values for a given cluster algorithm; the book is mainly appropriate for consideration&#36;k&#36;value, the correct number of the fractional clusters.</p>
<p>&#36;D&#36;The joint probability distribution is usually unknown. Thus, for the same distribution of sample data sets, we can use a range of methods, including random disturbances (random perturbation), subsampling or self-help sampling (bootstrap resampling). We'll start with the self-help method:</p>
<p>By From&#36;D&#36;Sampling (replaced, i.e. allowing the same data point to be selected several times, each sample)&#36;D_i&#36;So it's different to generate it.&#36;t&#36;Size&#36;n&#36;The sample. Next, for each sample,&#36;D_i&#36;, with different&#36;k&#36; Value (from 2 to) &#36;k^\mathrm{max}&#36;) Runs the same group algorithm.</p>
<p>You!&#36;C_k(D_i)&#36;Organisation&#36;k&#36;From Sample&#36;D_i&#36;Get a cluster. Next, the method compares all clusters with a certain group function&#36;C_k(D_i)&#36;and&#36;C_k(D_j)&#36;Distance between. Some external concentration assessment measures can be used as distance measures, e.g., by&#36;C=C_k(D_i),T=C_k(D_j)&#36;And vice versa. Based on these values, we calculate each.&#36;k&#36;The expectations of values are in pairs. Finally, the lowest deviation from the different clusters obtained from the re-sampling data sets&#36;k^*&#36;Yes.&#36;k&#36;The best choice is because it has the highest degree of stability.</p>
<h4>Cluster trend</h4>
<p>Cluster tendency or clusterability (clusterability) is designed to judge data sets&#36;D&#36;There are meaningful clusters. This is often difficult because it is difficult to define what is a subset in the first place, such as partitioning, hierarchy, density-based, map-based, etc.</p>
<p>Even if you have a sort of cluster, for a given data, Set&#36;D&#36;It remains difficult to define a suitable zero model (null model, i.e., model without any cluster structure). Moreover, even if data are judged to be conglomerate, we still face the problem of determining the number of judgement clusters.</p>
<p>Hopkins statistics are a thin sample test of space randomity. Give a Organisation&#36;n&#36;Data set for points&#36;D&#36;We create&#36;t&#36;A random sample.&#36;R_i&#36; (Each subsampling contains&#36;m&#36;Point, of which&#36;m\ll n&#36;I'm not sure. Dataspaces of these samples and&#36;D&#36;Same, randomly and evenly generated at each dimension.</p>
<p>Besides, we're going to go straight to...&#36;D&#36;Generating&#36;t&#36;Samples (each inclusive)&#36;m&#36;(Place) (Placed), use unreleased samples. You're the one who's gonna get you.&#36;D_i&#36;Representative's first&#36;i&#36;A direct subsampling. Next, calculate each one.&#36;x_j\in D_i&#36;and&#36;D&#36;Minimum distance between points:
&#36;&#36;\delta (\bardsymbol{x}<em>j)=\min</em>{\boldsymbol{x}_i\in D,\boldsymbol{x}_i\neq\boldsymbol{x}_j}{\delta(\boldsymbol{x}_j,\boldsymbol{x}_i)}&#36;&#36;</p>
<p>I'm sorry.&#36;i&#36;- Yes, it's a sample.&#36;R_i&#36;and&#36;D_i&#36; Hopkins Statistics&#36;d&#36;Definition:</p>
<p>&#36;&#36;\mathrm{HS}<em>i=\frac{\sum</em>{y_j\in\mathbf{R}<em>i}(\delta</em>{\min}(\boldsymbol{y}<em>j))^d}{\sum</em>{y_j\in\mathbf{R}<em>i}(\delta</em>{\min}(\boldsymbol{y}<em>j))^d+\sum</em>{\boldsymbol{x}_j\in\boldsymbol{D}<em>i}(\delta</em>{\min}(\boldsymbol{x}_j))^d}&#36;&#36;</p>
<p>This statistical volume will provide a recent neighbourhood distribution of the data points generated at random and will be distributed over the next few years.&#36;D&#36;Compares the latest neighbourhood distribution of random subsets of the medium data points. If the data are of good fusion, we expect&#36;\delta_{\min}(x_j)&#36;Less than&#36;\delta_{\min}(y_j)\text{,且在这种情况下，HS}_i&#36; Trends to 1.</p>
<p>If the two closest neighbors are similar, HS&#36;_i&#36;The value is close to 0.5, which means that the data are almost random and not clearly clustered.</p>
<p>And finally, if...&#36;\delta_{\min}(x_j)&#36;Value greater than&#36;\delta_{\min}(y_j)&#36;, HS&#36;_i&#36;A zero, which means a little exclusion, and no cluster.</p>
<p>Based on&#36;t&#36;A different HS.&#36;_i&#36;Value, as judged by the average and variance of the statistical volume&#36;D&#36;Can cluster.</p>
