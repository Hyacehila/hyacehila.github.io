---
title: 'Descriptive Statistics and Visualization: Measurement, Distances, and Statistical Graphics'
title_zh: 描述性统计与可视化：数据测度、距离与相关分析
date: 2023-11-05 00:21:21 +0800
categories:
- Data Science
- Data Practice
tags:
- Statistics
- Data Visualization
- Descriptive Statistics
author: Hyacehila
mathjax: true
hidden: true
excerpt: Covers measurement, distances, association analysis, distribution summaries, robust statistics, and common statistical
  graphics.
description: Covers measurement, distances, association analysis, distribution summaries, robust statistics, and common statistical
  graphics.
excerpt_zh: 整理数据测度、距离、相关分析、分布描述、稳健统计量和常见统计图形。
permalink: /blog/2023/11/05/descriptive-statistics-and-visualization-notes/
lang: en
translation_key: 2023-11-05-descriptive-statistics-and-visualization-notes
translation_status: machine
translation_source_hash: 39339764a3f6442b421c1503633540528993e2872a74dc8bf04a92461efd7fdf
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<h2>Summary</h2>
<h3>Statistical analysis and descriptive statistical analysis</h3>
<p>Statistical analysis: the process by which the regularity of the data is derived from the performance data</p>
<p>Statistical analysis is the core of the entire statistical research, and is more important and complex than data collection, and most of our course content falls within the scope of statistical analysis.</p>
<p>In the course of statistical analysis, statisticians have been divided about how to organize data.
So, two main branches of descriptive statistical analysis and extrapolational statistical analysis were created.</p>
<ul>
<li>Qualitative statistical analysis studies on how to organize and describe data (e.g., common discrete concentration trends, some visualization)</li>
<li>Instimulating statistical analysis uses samples to reverse the overall situation (e.g., parameter estimates and hypothetical tests)</li>
</ul>
<p>They're both successful. They can't be lost.</p>
<p>Here we focus on descriptive statistical analysis; as a branch of the study of data shape, descriptive statistical analysis and data analysis<a href="/en/blog/2024/02/29/exploratory-data-analysis-learning-notes/">Explored data analysis</a>Close. </p>
<h3>Some common data classifications</h3>
<h4>Measure</h4>
<p>Depending on the units of the data, we give the following classification.</p>
<ul>
<li>Interval (quantitative) variables: variables that can change continuously</li>
<li>Order variable: No clear quantitative changes, but grade changes</li>
<li>Classified variables: Variables are expressed in some categories as being equal between classes</li>
</ul>
<h4>Source of data</h4>
<p>Depending on the source of the data, there are the following classifications:</p>
<ul>
<li>Observation data: data collected for observations of the real world</li>
<li>Test data: data collected in experimental human control variables</li>
</ul>
<h4>Time-related</h4>
<ul>
<li>Cross-section data: data collected by objects at the same time (column is characteristic, row is object)</li>
<li>Time series data: data collected from observations of the same object at different times (column is characteristic, row is different times)</li>
<li>Panel data: multiple observations of multiple time (lists of objects and characteristics, rows of different time) with characteristics of the first two<strong>The regression analysis of panel data will certainly involve highly relevant observations, i.e., non-I.I.D samples, which will seriously affect the analysis.</strong></li>
</ul>
<h2>Distance and Similarity Coefficient</h2>
<h3>Distance</h3>
<h4>Definitions</h4>
<p>As we have studied in the general analysis, the following definition of distance is required:
See<a href="/en/blog/2023/09/11/functional-analysis-notes/">General analysis of communications</a> and the “Definition of measuring space” section.</p>
<p>In part, we may be able to use this basic definition to determine the distance subjectively if we really need to.
We have some of the more common distances down there in the cluster analysis.</p>
<h4>Minkowski distance</h4>
<p>&#36;&#36;d\left(x,y\right)=\left[\sum_{i=1}^{p}|x_{i}-y_{i}|^{q}\right]^{1/q}&#36;&#36;
Ming's distance is the widest distance used.
When?&#36;q=1&#36;And then he was the absolute distance we learned.
When?&#36;q=2&#36;And then, he was the O'Shea distance.
When?&#36;q=\infty&#36; ♪ When ♪ &#36;d\left(x,y\right)=\max_{1\leqslant i\leqslant\rho}\mid x_{i}-y_{i}\mid&#36; It's called Chelby Shev's distance.</p>
<p>♪ With&#36;q&#36;Ming distance is becoming more sensitive to anomalies.
Just like we did when we introduced the O'Shea distance.<a href="/en/blog/2024/09/11/multivariate-statistics-introduction-notes/">Multi-statistical inferences</a> The "O'S" section.</p>
<p>Ming's distance also requires standardization before operation.
<strong>The Ming-Zhi distance is applied to the calculation of the distance between orderly variables</strong></p>
<h4>Lance and Williams distance</h4>
<p>When all the data is right, define the range of the Langham.
&#36;&#36;d(x,y)=\sum_{i=1}^{p}\frac{|x_{i}-y_{i}|}{x_{i}+y_{i&#125;&#125;&#36;&#36;
It works well when it's dealing with data that are more skewed and contain abnormalities, and it's not about units.</p>
<h4>Ma'am's distance.</h4>
<p><a href="/en/blog/2024/09/11/multivariate-statistics-introduction-notes/">Multi-statistical inferences</a> The "Marzie Distance" section.
In cluster analysis, we don't usually use marts, because the class is dynamically changing, and it's hard to give a definite matrix of the differences.</p>
<h4>Hamming Distance</h4>
<p><strong>Hamming Distant</strong> It's a measure of two.<strong>Long String</strong>(or sequence) the measure of the number of elements at the same location. The idea is very simple, with the corresponding character (category) at the same location, and Hamming is one more distance, and it is sufficient to calculate each position of the whole string (or sequence)</p>
<h4>Leveshtein Distance</h4>
<p>Leveshtein distance, also known as editing distance, is one way to measure the difference between the two strings. It quantifys this difference by calculating the minimum number of operations required to convert a string into another string. He's a Hamming distance extension, which can be used for multiple categories.</p>
<p>Common operations are:</p>
<ol>
<li><strong>Insert</strong>: Inserts a character in a string.</li>
<li><strong>Delete</strong>: Remove a character from a string.</li>
<li><strong>Replace</strong>: Replace one character in a string with another.</li>
</ol>
<p><strong>We determine the distance of Leventein based on the number of times the string is changed, and he needs to use dynamic planning to solve it.</strong></p>
<h4>VDM Distance (Value Difference Metric)</h4>
<p><strong>Value Difference Metric (VDM, value difference measure)</strong> It is a similarity measure for classification tasks, which is particularly applicable to processing<strong>Nominal Properties</strong>(i.e., type characteristics) data.</p>
<p>The heart of VDM is through<strong>Probability difference</strong>Measuring the similarities of two samples in a particular characteristic </p>
<ul>
<li>For each type characteristic, VDM calculates the probability of conditions under each target category for which the characteristic is different.</li>
<li>The difference in this characteristic between the two samples is determined by the probability differences in the conditions for which they each take the values.</li>
</ul>
<p>VDM distance measures<strong>Cumulative extent of the difference in the distribution of target categories between the two samples with their values obtained in each characteristic</strong>I'm sorry. The greater the distance, the more significant the differences in the contribution of the characteristic values of the two samples to the classification results, the less likely they would fall into the same category.</p>
<p>To calculate VDM distance, the next steps are required, and the features we study are discrete, that is, a classification.</p>
<p>calculating the probability of conditions;for features&#36;A&#36;..of the value of the value&#36;a&#36;in the target category&#36;c&#36;The probability of the condition is:
&#36;&#36;P(c\mid A=a)=\frac{\text{类别为}c,\text{特征取值为}a\text{的样本数&#125;&#125;{\text{特征取值为}a\text{的样本数&#125;&#125;&#36;&#36;</p>
<p>calculating single feature VDM distance; sample is&#36;x,y&#36;   Only study characteristics &#36;A&#36; Two samples in the signature.&#36;A&#36;The values are given in the&#36;a_x,a_y&#36;  &#36;C&#36;Total is the target category
&#36; \\mathrm{VDM}<em>A(x,y)=\sum</em>{c=1}^C\left[P(c\mid A=a_x)-P(c\mid A=a_y)\right]^2&#36;&#36;
计算多特征的两样本 VDM 距离；
&#36;&#36;\mathrm{VDM}(x,y)=\sum_{i=1}^m\mathrm{VDM}_{A_i}(x,y)&#36;&#36;</p>
<p><strong>The distance measure of the mixture properties can be achieved by combining VDM distance with OSD distance, commonly known as HVDM; in some cases weighting of VDM characteristics is required to distinguish more important features</strong>  </p>
<p><strong>VDM is based on the probability of conditions, which may be quite inaccurate when the number of samples is small, and even a problem of zero probability exists.</strong></p>
<p><strong>VDM design targets are classification issues and are not available for return or task. By using the existing cluster centre as a pseudo-label, updating the new cluster results using dynamic cluster methods, recalculating VDM distances, and obtaining the final results over time, the cluster issues can be indirectly used</strong></p>
<h3>Similarity factor</h3>
<h4>Definitions</h4>
<p>Unlike distance, the larger the similarity coefficient, the closer the two samples are, the smaller the distance means the closer; we can export the similarity coefficients by distance, but we can also redefine some similar coefficients. The similar coefficients are in general the same.&#36;[0,1]&#36;There are, of course, a few possible breaches.</p>
<h4>Clindrum Cosine</h4>
<p>Defines the similarity factor as
&#36;&#36;\mathrm{cos}\theta_{y}=\frac{\sum_{k=1}^{n}x_{ki}x_{kj&#125;&#125;{\left[\left(\sum_{k=1}^{n}x_{kj}^{2}\right)\left(\sum_{k=1}^{n}x_{kj}^{2}\right)\right]^{1/2&#125;&#125;&#36;&#36;
It's the cosine of the cross between two vectors.<strong>At this point, we focus on the vector angle, not on the size of the value, which is very effective for the existence of thin high-dimensional data.</strong></p>
<h4>Related coefficient</h4>
<p>Draws similarity using two vector-related coefficients
&#36;&#36;cor=\frac{\sum_{i=1}^n(a_i-\bar{A})(b_i-\bar{B})}{\sqrt{\sum_{i=1}^n(a_i-\bar{A})^2\sum_{i=1}^n(b_i-\bar{B})^2&#125;&#125;&#36;&#36;
of which&#36;a_i,b_i&#36;It's a sample.&#36;A,B&#36;We've already regulated it.&#36;\bar{A},\bar{B}&#36; It's vector.&#36;A,B&#36;The average of the sample is the average of the sample rather than the average of the characteristic.</p>
<p><strong>In particular, we are very reluctant to use two vector-related coefficients to study sample similarities except for the following application scene.</strong></p>
<ul>
<li>Sample characteristics are time series, and at this point we can assume that the distribution is similar within the same sample and therefore the relevant coefficients can be used.</li>
<li>If the characteristics of the sample represent a certain distribution (e.g., the distribution of interest of a user, the pixel histogram of an image) e.g., user A, the interest in five commodity categories min &#36;[5,3,1,0,2]&#36; User B rating interest in 5 commodity groups &#36;[3,1,2,0,3]&#36; The correlation coefficient at this time implies similarity in preference.</li>
</ul>
<h4>Jaccard coefficient</h4>
<p><strong>Jaccard Index</strong>(Yakar Index) is an indicator used to measure the similarities between the two clusters. It's used to calculate the two pools. <strong>Similarity</strong>and especially for <strong>Diagonal characteristics</strong> or <strong>Pool type of data</strong></p>
<p>For dual feature, we can calculate as follows, and the formula is
&#36;&#36;J(A, B)=\frac{|A \cap B|}{|A \cup B|}&#36;&#36;</p>
<ul>
<li>Intersection: indicates the position of both samples at the same level of feature as 1.</li>
<li>: indicates the position of the characteristic 1 in at least one sample of two samples</li>
</ul>
<p>For the group characteristics, each sample is a series of random collections, calculating the number of elements that are combined and intersect, comparing, and the formula remains the same.
&#36;&#36;J(A, B)=\frac{|A \cap B|}{|A \cup B|}&#36;&#36;</p>
<h2>Relevant analysis</h2>
<h3>Brief description of the analysis</h3>
<p>The simple purpose here is to analyze data to see if there's a relationship before several data sets and measure it.
<strong>Relationship: When one or more variables change, the other variable that corresponds to it, although not certain, changes according to a certain pattern.</strong></p>
<p><strong>The relationship is not causal.</strong> The two variables he studied were equally equal. </p>
<p>The correlation between variables is divided into two types of functional and non-confirmity relationships, which obviously are more statistically studied than is the case with the latter, which requires a great deal of knowledge about the intrinsic characteristics of the event.</p>
<p>The vague definition of the relationship shows that he must be in the real world and in our statistics.</p>
<p>From the structure of the data studied, the analysis is divided into numerical data-related analysis, sequence data-related analysis, category data-related analysis, and the analysis of the data-related data of the classification data.</p>
<h3>Three relevant coefficients</h3>
<h4>Related coefficient</h4>
<p>The correlation coefficients for two random variables are a linear correlation between them.</p>
<p>The correlation coefficients are linearly relevant, even if there is a function, but non-linear; the correlation coefficient between the two random variables is 0, as if
&#36;X\sim N(0,1),Y=cosX,Z=X^2&#36; then &#36;Corr(X,Y)=0,Corr(X,Z)=0&#36;
We can verify that.
When the correlation coefficient is&#36;(0,1)&#36;And sometimes, it proves that there is some degree of linear correlation, but not entirely.</p>
<p>Although the coefficients are only linear, he is still the most widely used indicator, and we have suggested that many other indicators do not replace him, except that they are too complex; and because they are not very complex. <strong>The incoherent and independent relevance of the 2D normal distribution is equal.</strong></p>
<p>So, the correlation coefficient is the measure of correlation, not just the measure of linear correlation, when the whole is normal.</p>
<h4>Recontribution factor</h4>
<p>We want to be able to study the correlation between a random variable and a random vector and to paint it as a value.
According to the foregoing, a very simple idea is to use a linear combination of random vectors to describe all his information and then study the relevance of linear combinations and random variables; we call the maximum correlation coefficient between linear combinations and random variables as a compound correlation factor.
It's easy.
&#36;&#36;00\
\l{\rho^l^rft}&amp; =\frac{Cov^{2}\left(y,l^{\prime}x\right)}{V\left(y\right)\cdot V\left(l^{\prime}x\right)}=\frac{\left(\sigma_{xy}^{\prime}l\right)^{2&#125;&#125;{\sigma_{yy}\cdot l^{\prime}\Sigma_{xx}l}  \
&amp;\leqslant\frac{\left(\boldsymbol{\sigma}<em>{xy}^{\prime}\boldsymbol{\Sigma}</em>{xx}^{-1}\boldsymbol{\sigma}<em>{xy}\right)\left(\boldsymbol{l}^{\prime}\boldsymbol{\Sigma}</em>{xx}\boldsymbol{l}\right)}{\boldsymbol{\sigma}<em>{xy}\cdot\boldsymbol{l}^{\prime}\boldsymbol{\Sigma}</em>{xx}\boldsymbol{l&#125;&#125;=\frac{\left(\boldsymbol{\sigma}<em>{xy}^{\prime}\boldsymbol{\Sigma}</em>{xx}^{-1}\boldsymbol{\sigma}<em>{xy}\right)}{\boldsymbol{\sigma}</em>{xy&#125;&#125;
\end{aligned}&#36;&#36;
当时&#36;l=\Sigma_{xx}^{-1}\sigma_{xy}&#36; 等号成立 则有
&#36;&#36;\begin{gathered}
\rho_{y}\cdot x =\max_{l\neq0}\rho\left(y,l^{\prime}x\right)=\rho\left(y,\sigma^{\prime}<em>{xy}\boldsymbol{\Sigma}</em>\right)
== sync, corrected by elderman == @elder man
I'm sorry, I'm sorry.
The synthesis can give the conclusion.</p>
<ul>
<li>When?&#36;p=1&#36;..times when the compound correlation coefficient is natural degradation as normal correlation coefficients.</li>
<li>The compound coefficient is 0, which means it's not relevant.</li>
<li>The compound correlation factor is constant for unit changes</li>
<li>The sum of the individual relative coefficients when the amount of the random vector is independent of each other and the squares of the compound correlation coefficients are equal
Under the assumption of multiple normality, the compound coefficient for the sample is (the compound coefficient is much more likely to be estimated)
&#36;r y.<em>x=\sqrt{\frac{s</em>{xy}^{\prime}\boldsymbol{S}<em>{xx}^{-1}\boldsymbol{s}</em>{xy&#125;&#125;{s_{yy&#125;&#125;}=\sqrt{\boldsymbol{r}<em>{xy}^{\prime}\hat{\boldsymbol{R&#125;&#125;</em>{xx}^{-1}\boldsymbol{r}_{xy&#125;&#125;.&#36;&#36;</li>
</ul>
<h4>Offset factor</h4>
<p>When we look at the most common correlation coefficients, we tend to calculate the Pearson correlation coefficients; in fact, he's affected by indirect correlation, so we usually call it the total correlation coefficients, and now we want to eliminate this indirect influence, and we study the most common correlations.
Definitions
&#36;&#36;\bardsymbol(Sigma)}<em>{11}\boldsymbol{.}<em>2\boldsymbol{=}\boldsymbol{\Sigma}</em>{11}-\boldsymbol{\Sigma}</em>{12}\boldsymbol{\Sigma}<em>{22}^{-1}\boldsymbol{\Sigma}</em>{21}=(\boldsymbol{\sigma}<em>{i\boldsymbol{j&#125;&#125;\boldsymbol{\cdot}</em>{k+1,\cdots,p})&#36;&#36;
为&#36;x_2&#36;给定的时候 &#36;x_1&#36;的偏协方差矩阵 他的对角线元素称为偏协方差 对角线元素称为偏方差
定义
&#36;&#36;\rho_{ij+k+1,\cdots,p}=\frac{\sigma_{ij}\cdot_{k+1,\cdots,p&#125;&#125;{\sqrt{\sigma_{ii}\cdot k+1,\cdots,p\sigma_{jj}\cdot_{k+1,\cdots,p&#125;&#125;},\quad1\leqslant i,j\leqslant k&#36;&#36;
为&#36;x_2&#36;给定的时候 &#36;x_i,x_j&#36;的&#36;(p-k)&#36;阶偏相关系数 他剔除了&#36;x_{k+1}&#36;到&#36;x_{p}&#36;的线性影响
在多元正态的假定下 偏相关系数的极大似然估计（样本偏相关系数为）
&#36;&#36;r_{\hat{y}\cdot k+1,\cdots,p}=\frac{s_{\vec{y&#125;&#125;\cdot k+1,\cdots,p}{\sqrt{S_{\hat{p&#125;&#125;\cdot k+1,\cdots,pS_{jj}\cdot k+1,\cdots,p&#125;&#125;&#36;&#36;
其中
&#36;&#36;S_{11}.<em>{2}=S</em>{11}-S_{12}S_{22}^{-1}S_{21}=\left(s_{ij}._{k+1},\cdots,p\right)&#36;&#36;</p>
<h3>Analysis of quantitative data relevance (Pearson correlation factor)</h3>
<h4>Description and measurement of the relationship</h4>
<h5>Description of the relationship</h5>
<p>Before we do the relationship analysis, we have a very central assumption that we only study linear relevance.
<strong>We only study linear correlations between variables.</strong>
We've already emphasized this in many courses in probabilistic theory, mathematical statistics.
A quantitative relationship description for only two variables
Draw a scattering map and draw a general relationship
But we need a more precise quantitative approach.</p>
<h5>Relevant relational measures</h5>
<p>Here we study and give the probability coefficients directly.&#36;r&#36;
&#36;&#36;r=\frac{\sum\left(x-\overline{x}\right)\left(y-\overline{y}\right)}{\sqrt{\sum\left(x-\overline{x}\right)^2}\cdot\sqrt{\sum\left(y-\overline{y}\right)^2&#125;&#125;,&#36;&#36;
And then you can go over the nature of the probabilistic theory that's given you.</p>
<ul>
<li>The range of values to be taken is&#36;[-1,1]&#36; </li>
<li>Symmetrical</li>
<li>Relevant coefficient size and&#36;x,y&#36;The origin of the event is not a matter of scale.</li>
<li>Only linear relationships (but non-linear transformation of raw data)</li>
<li>There's no guarantee of a causal relationship between the two.</li>
</ul>
<p>A relevant factor of this kind&#36;r&#36; Now we call it the Pearson correlation coefficient, which is only a fraction of what counts, and there are plenty of statisticians behind it who contribute to the relevance study. </p>
<p>The general Pearson correlation coefficient is used as a precondition &#36;x,y&#36; The data of the sample of the variable should be paired and each group of data should be independent of each other, and the sample number should be greater than 30, and the number of samples should be more than 30, and the number of samples should be more than 30, and the number of data in the data should be more than 30, and the number of data in the data should be more than 30, and the number of data in the data should be more than 30, and the number of data in the data should be more than 30, and the number of data in the data should be more than 30, and the data in the data should be more than 30, and the data in the data should be more than 30, and the data in the data should be used in the data form.
These conditions are usually assumed to be valid.</p>
<h5>The size and relevance of the relevant coefficient</h5>
<p>In the traditional sense,
The correlation factor is generally judged in accordance with the following rules:</p>
<ul>
<li>More than 0.8</li>
<li>Medium relevance above 0.5</li>
<li>Weaknesses above 0.3</li>
<li>Not relevant</li>
</ul>
<h4>Assumptions of relevance</h4>
<p>The test of the relevancy of the coefficient is generally used.&#36;t&#36; Test
Establishment of assumptions
&#36;&#36;\begin{aligned}H_{0}:\rho=0;\H_{1}:\rho\neq0。\end{aligned}&#36;&#36;
Introduce testing statistics
&#36;&#36;t=|r|\sqrt{\frac{n-2}{1-r^{2&#125;&#125;}\sim t\left(n-2\right)&#36;&#36;
Execute the two-sided&#36;t&#36;Just test it.</p>
<p><strong>In the case of large samples, basically all the relevant coefficient tests are considered significant, so sometimes the practicalities are omitted, and we don't mention them anymore.</strong></p>
<h3>Relevance analysis of sequenced data (Spearman & Kendall & C)</h3>
<p>First, we need to be clear about what sort of sequenced data is;</p>
<h4>Source: Spearman.</h4>
<p>This is essentially a supplement to the Pearson coefficient that requires a logical equidistance of sample data, just the size of the relationship.</p>
<p>That's right here.<strong>Changed the analysis of numerical data to qualitative data</strong></p>
<p>Note: The variables involved in the analysis are being qualityd here, even if one of them would have met the Pearson relevance test.</p>
<p>For raw data&#36;x_{i},y_{i}&#36; Sort them separately in order of small to large, and get the respective serial number of each data set.&#39;},y_{i}^{&#39;}&#36; 称为原始数据的秩次 记秩次的差为&#36;d i}&#36;..so the formula that gives Spearman's coefficient is
&#36;&#36;\rho_{s}=1-\frac{6\sum d_{i}^{2&#125;&#125;{n\left(n^{2}-1\right)},&#36;&#36;
The analysis of the relative coefficients and the weak relevance is consistent with Spearman and Pearson.</p>
<h4>Generate Kendall's correlation coefficient</h4>
<p>The Kendall-related coefficient, also known as the Kendall-related coefficient, is also a thorium-related coefficient, although it targets orderly class variables such as size, age, obese grade (high, moderate, mild, non-obese). It measures the strength of the single-tangular relationship between two orderly variables.<strong>The Kendall coefficient uses the concept of pairing to determine the strength and weakness of the coefficient.</strong></p>
<p>Two pairs can be divided into one pair (Concordant) and one pair (Discordant). Consistency refers to the relative relationship between the values obtained by the two variables, which can be understood as having the same symbol as Y2-Y1; differences refer to their relative relationship, with the opposite symbol of X2-X1 and Y2-Y1.<strong>Each pair of requirements involves two variables and four elements.</strong></p>
<p>The Kendall coefficient has two formulas, one for Tau-a and the other for Tau-b. The difference is that Tau-b can deal with situations with the same value, i.e., a parallel, which we need to describe separately.</p>
<p>For Tau-a
&#36;&#36;\text{Tau-a}=\frac{c-d}{\frac{1}{2}n(n-1)}&#36;&#36;
of which&#36;n&#36;is the number of samples, the denominator measures the total number of possible combinations, which is the number of possible combinations, if not repeated&#36;c+d&#36; &#36;c&#36;It's a consistent logarithm.&#36;d&#36;It's a divide.</p>
<p>Kendall is related to the nature of the difference between the former, which is a relative improvement of the Pearson, and Spearman, which is based on a study of differences and unanimity, which is more sensitive to issues of their own sort, is not sensitive to abnormal and non-linear relationships and is not covered by linear correlation coefficients.</p>
<p>Where the two data are identical (i.e. duplicate), they are neither considered synergetic nor inconsistent. However, the double value affects the calculation of the denominator (the total logarithm is subject to the deduction of the duplicate logarithm). That's Ta-b.
&#36;&#36;\text{Tau-b}=\frac{c-d}{\sqrt{(c+d+t_x)(c+d+t_y)&#125;&#125;&#36;&#36;
of which&#36;t_x,t_y&#36;Yes. &#36;X&#36; and &#36;Y&#36; .</p>
<p>Tau-a above, Tau-b only apply to square tables, i.e. <strong>The two variables need the same number.</strong> For rectangular tables, the maximum value may be less than 1 as calculated by the formula above, for which the improved Tau-c by Tu-1 is introduced
&#36;&#36;\text{Tau-c}=\frac{2m(c-d)}{n^2(m-1)}&#36;&#36;
of which&#36;m&#36;is the smaller of the rows and columns,&#36;n&#36;It's a sample volume.</p>
<h3>Analysis of the relevance of qualitative data (column analysis)</h3>
<p>The matrix (contingency table) is the basis for the analysis of the defined data, which is the frequency tables that are presented when observations are classified according to two or more data.</p>
<h4>Definition of the matrix</h4>
<p>Assume that the individual in a general is based on two attributes&#36;A,B&#36;I'm sorting out.&#36;r&#36;Zero and&#36;c&#36;We'll take a sample. &#36;f_{ij}&#36; It's the type of observation frequency that we're looking at so we can build a two-dimensional one.&#36;rc&#36;List of rows
It's natural to expand more attribute dimensions, but it's more difficult to visualize and use them more limited.
We're just here to study the most basic of the matrix, and we've got more analysis of the matrix than we've ever seen.
<a href="/en/blog/2024/01/30/multivariate-statistical-analysis-notes/">Multi-statistical analysis</a> and “Responsive analysis” section</p>
<h4>Independentness test of the list</h4>
<p>We have to introduce the independent test of the list to examine whether the two variables in the matrix are relevant or independent.
We've been introduced to this in our analysis.
<a href="/en/blog/2024/01/30/multivariate-statistical-analysis-notes/">Multi-statistical analysis</a> "Independentness test" section</p>
<h4>Correlation factors in the joint table</h4>
<p>Adopt&#36;\chi^2&#36;The relevant coefficients (size of measurement of relevance) of the value by means of study of the array are at the heart of this section.
We'll introduce three methods and give them the right information.
<strong>The list is classified and has no concept of size, so the coefficients should be positive and there is no positive or negative correlation, and they are relevant.</strong></p>
<h5>&#36;\varphi&#36;Related coefficient</h5>
<p>To measure&#36;2\times 2&#36;Level of relevance of the matrix
Calculate formulae as
&#36;&#36;\varphi=\sqrt{\frac{\chi^{2&#125;&#125;{n&#125;&#125;,&#36;&#36;
of which
&#36;&#36;\chi^{2}=\sum_{i=1}^{2}\sum_{j=1}^{2}\frac{\left(f_{ij}-e_{ij}\right)^{2&#125;&#125;{e_{ij&#125;&#125;&#36;&#36;
Finally, a simplified result.
&#36;&#36;\varphi=\sqrt{\frac{\chi^{2&#125;&#125;{n&#125;&#125;=\frac{ad-bc}{\sqrt{\left(a+b\right)\left(c+d\right)\left(a+c\right)\left(b+d\right)&#125;&#125;&#36;&#36;
&#36;\varphi&#36;The correlation coefficient analysis is meaningless.
The relative weight and coefficients are determined by the Pearson coefficient</p>
<h5>&#36;C&#36;Related coefficient</h5>
<p>- It's for the larger list. - Yes.&#36;\varphi&#36;Theoretical expansion of the relevant coefficients
Calculate formulae as
&#36;&#36;C=\sqrt{\frac{\chi^{2&#125;&#125;{\chi^{2}+n&#125;&#125;,&#36;&#36;
of which
&#36;&#36;\chi^{2}=\sum_{i=1}^{1}\sum_{j=1}^{c}\frac{\left(f_{ij}-e_{ij}\right)^{2&#125;&#125;{e_{ij&#125;&#125;&#36;&#36;
The analysis factor means the same size as the Pearson coefficient.
I can see it. &#36;C&#36;Linking the relevant coefficients to the rows and columns in the column tables
So do not compare the coefficients of multiple arrays.
At this point, analysis is meaningless.</p>
<h5>&#36;V&#36;Related coefficient</h5>
<p>&#36;&#36;V=\sqrt{\frac{\chi^{2&#125;&#125;{n\cdot\min\left[\left(r-1\right),\left(c-1\right)\right]&#125;&#125;=\sqrt{\frac{\chi^{2&#125;&#125;{n\left(m-1\right)&#125;&#125;&#36;&#36;
It is noted that the V-related coefficients derived from the two rows or columns of the rows are not suitable for comparison;
When?&#36;min(r,c)=2&#36;And he was.&#36;\varphi&#36;Related coefficient (a form of extension)</p>
<h4>Summary of attention</h4>
<ul>
<li>A hyphenation analysis of the amount of causality from the column should be made at the row, because the variable is in the column</li>
<li>When data is divided into two categories, the theoretical frequency should not be less than five.</li>
<li>When data are divided into more categories, the theoretical frequency group of data less than 5 should not exceed 20% of the total data split
<strong>The matrix analysis should not be inconsistent with these concerns</strong></li>
</ul>
<h3>Typical relevant analysis</h3>
<p>Typical relevant analysis is one way to study the correlation between the two sets of variables, which reveals linear correlations between the two sets of variables, and he is very extensive in practical application, and we are promoting the coefficients after they have been extended.</p>
<p>We introduced typical relevant analyses because of two sets of variables (both of which are:&#36;p&#36;and&#36;q&#36;The correlation coefficient between the two is:&#36;pq&#36;The matrix that we often use is not good enough;</p>
<p>Naturally, we will introduce a downscaling of the main ingredients analysis, with as few as possible numbers to map the correlation between the two sets of variables.</p>
<h4>Generally relevant</h4>
<h5>Typical Related Export</h5>
<p>Set&#36;x=(x_1,x_2,...,x_p)^{\prime}&#36;and&#36;y=(y_1,y_2,...,y_q)^{\prime}&#36;It's two random sets of variables and &#36;V(x)=\bardsymbol(Sigma}<em>{11}(&gt;0),V(y)&#36; &#36;= \Sigma</em>{22}\left ( &gt; ♪ Right, Cov\left (x, y\right) ♪
&#36;V(binom xy)=begin{bmatrix}\boldsymbol{\<em>{11}&amp;\boldsymbol{\Sigma}</em>{12}\\boldsymbol{\Sigma}<em>{21}&amp;\boldsymbol{\Sigma}</em>You're not gonna get away with this?
We want to use an indicator to flip the line functions that maximize the correlation between the two sets of variables, which is very natural, using two vectors.&#36;u=a^{\prime}x\text{ 和 }\upsilon=b^{\prime}y&#36; Compress them into single variables. Recalculate&#36;uv&#36;The correlation coefficient between them makes them the largest.</p>
<p>Naturally.
&#36;&#36;00\
\oporatorname{Cov}(u,v)&amp; =Cov(a^{\prime}x,b^{\prime}y)=a^{\prime}Cov(x,y)\boldsymbol{b}=\boldsymbol{a}^{\prime}\boldsymbol{\Sigma}<em>{12}\boldsymbol{b}  \
V(u)&amp; =V(a^{\prime}x)=a^{\prime}V(x)a=a^{\prime}\boldsymbol{\Sigma}</em>{11}\boldsymbol{a}  \
V(v)&amp; =V(\boldsymbol{b}^{\prime}\mathbf{y})=\boldsymbol{b}^{\prime}\boldsymbol{V}(\mathbf{y})\boldsymbol{b}=\boldsymbol{b}^{\prime}\boldsymbol{\Sigma}<em>{22}\boldsymbol{b}
\end{aligned}&#36;&#36;
所以相关系数为
&#36;&#36;\rho(u,v)=\frac{a^{^{\prime&#125;&#125;\boldsymbol{\Sigma}</em>{12}\boldsymbol{b&#125;&#125;{\sqrt{\boldsymbol{a}^{^{\prime&#125;&#125;\boldsymbol{\Sigma}<em>{11}\boldsymbol{a&#125;&#125;\sqrt{\boldsymbol{b}^{^{\prime&#125;&#125;\boldsymbol{\Sigma}</em>{22}\boldsymbol{b&#125;&#125;}&#36;&#36;
为了避免一些毫无意义的结果重复 我们一般要求&#36;uv&#36;都是标准化的变量 也就是
&#36;&#36;a^{\prime}\boldsymbol{\Sigma}<em>{11}\boldsymbol{a}=1\mathrm{<del>,</del>}\quad\boldsymbol{b}^{\prime}\boldsymbol{\Sigma}</em>{22}\boldsymbol{b}=1&#36;&#36;
因此我们希望极大化的相关系数为
&#36;&#36;\rho(u,v)=a^{\prime}\boldsymbol{\Sigma}<em>\bardsymbol{b}&#36;&#36;
We're omitting some unnecessary proofs, giving a way to calculate the coefficient to be so large.
Easy to know</em>{11}^{-1}\Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21},\Sigma_{21}^{-1}\Sigma_{21}^{-1}\Sigma_{12},\Sigma_{11}^{-1/2}\Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}\Sigma_{11}^{-1/2}(\geqslant0)&#36;和&#36;\Sigma_{22}^{-1/2}&#36; &#36;\boldsymbol{\Sigma}<em>{211}^{-1}\boldsymbol{\Sigma}</em>{12}\boldsymbol{\Sigma}<em>{22}^{1/2}(\geqslant0)&#36;都有着相同的非零特征值，可记为 &#36;\rho_1^2\geq\rho_2^2\geq\cdots\geq\rho_n^2&gt;0&#36;,这里 &#36;m&#36; 为 &#36;\boldsymbol{\Sigma}</em>The tidbit of &#36;12.00
&#36;a_1,a_2,...,a_m&#36; Yes.&#36;\Sigma_{11}^{-1}\Sigma_{12}^{-1}\Sigma_{22}^{-1}\Sigma_{21}&#36;Corresponds to&#36;\rho_1^2,\rho_2^2,...,\rho_m^2&#36;. The character vector
&#36;b_1,b_2,...,b_m&#36;  Yes&#36;\Sigma_{22}^{-1}\Sigma_{21}\Sigma_{11}^{-1}\Sigma_{12}^{-1}&#36;Corresponds to &#36;\rho_1^2,\rho_2^2,...,\rho_m^2&#36; . The character vector
We'll take it. &#36;a=a_1,b=b_1&#36;When the correlation coefficients are so great, we call it
&#36;&#36;u_1=\boldsymbol{a}_1^{\prime}\boldsymbol{x},\quad v_1=\boldsymbol{b}_1^{\prime}\boldsymbol{y}&#36;&#36;
It's the first of the typical variables.&#36;a_1,b_1&#36;It's the first pair of typical correlation coefficients.&#36;p_1&#36;It's the first typical correlation factor.
If the amount of information extracted from the first typical variable is not sufficient, we can give the first&#36;i&#36;It's typically relevant.&#36;a_i,b_i,p_i&#36;</p>
<h5>Nature of typical variables</h5>
<h6>The typical variables of the same group are not relevant.</h6>
<p>&#36;&#36;u_i=\boldsymbol{a}_i^{\prime}\boldsymbol{x},\quad v_i=\boldsymbol{b}<em>i^{\prime}\boldsymbol{y}&#36;&#36;
则
&#36;&#36;\begin{aligned}\rho(u_i,u_j)=&amp;\mathrm{Cov}(u_i,u_j)=\boldsymbol{a}^{\prime}\boldsymbol{\Sigma}</em>{11}\boldsymbol{a}<em>j=0,\quad1\leqslant i\neq j\leqslant m\\rho(v_i,v_j)=&amp;\mathrm{Cov}(v_i,v_j)=\boldsymbol{b}^{\prime}\boldsymbol{\Sigma}</em>{22}\boldsymbol{b}_j=0,\quad1\leqslant i\neq j\leqslant m\end{aligned}&#36;&#36;</p>
<h6>Relevance between typical vectors</h6>
<p>&#36;&#36;\rho(u_{i},v_{i})=\rho_{i},\quad i=1,2,\cdots,m&#36;&#36;
&#36;&#36;\begin{aligned}
\rho\left(u_{i},v_{j}\right)&amp; =Cov(u_{i},v_{j})=Cov(a_{i}^{\prime}x,b_{j}^{\prime}y)=a_{i}^{\prime}Cov(x,y)b_{j}  \
&amp;=\boldsymbol{\alpha}<em>i^{\prime}\boldsymbol{\Sigma}</em>{11}^{-1/2}\boldsymbol{\Sigma}<em>{12}\boldsymbol{\Sigma}</em>{22}^{-1/2}\boldsymbol{\beta}_j=\rho_j\boldsymbol{\alpha}_i^{\prime}\boldsymbol{\alpha}_j=0,\quad1\leqslant i\neq j\leqslant m
\end{aligned}&#36;&#36;</p>
<h6>Correlation coefficient between original and typical variables</h6>
<p>&#36;&#36;\text{记}A=(a_1,a_2,\cdots,a_m),B=(b_1,b_2,\cdots,b_m),\text{则}\u=A^{\prime}x,\quad v=B^{\prime}y&#36;&#36;
&#36;&#36;\begin{gathered}
\operatorname{Cov}(x,u) =Cov(x,A^{\prime}x)=\boldsymbol{\Sigma}<em>{11}A \
\operatorname{Cov}\left(x,\nu\right) =\mathrm{Cov}(x,B^{\prime}y)=\boldsymbol{\Sigma}</em>{12}\boldsymbol{B} \
Cov\left(y,u\right) =\mathrm{Cov}(y,A^{\prime}x)=\boldsymbol{\Sigma}<em>{21}A \
\operatorname{Cov}\left(y,v\right) =\mathrm{Cov}(y,B^{\prime}y)=\boldsymbol{\Sigma}</em>{22}\boldsymbol{B}
\end{gathered}&#36;&#36;</p>
<h6>Typical and general correlation factors</h6>
<p>It's clear from the definition that when&#36;p=q=1&#36;When it happens, it's typically the normal correlation coefficient.
When &#36;p=1<del>or</del>q = &#36;1 he's a compound coefficient.
So, re-relevance is a typical case, and simple re-relevance is a re-relevance.
Their size is linked.
The first typical correlation factor is at least Same&#36;x&#36;(or)&#36;y&#36;) any of the weights and&#36;y&#36;(or)&#36;x&#36;The compound correlation coefficient is as large (as can be seen from the first typical relevant definition)
Even if all these compound coefficients are small, the first typical correlation coefficient may be significant.
The compound coefficient is not less than the relevant coefficient between any of the fractions (as can be seen from the compound definition)
Even if all these relevant factors are small, the compound coefficient may be significant.</p>
<h6>Typical relevant coefficients after standardization</h6>
<p>Sometimes we'll standardize the weights before we'll calculate the typical correlation coefficient.
<strong>The calculations we've been working on are not standardized, and this is not the same as the factor analysis, the primary component analysis, the classic two techniques of reduction.</strong>
The standardized matrix is the matrix, and we can calculate the typical correlation coefficients based on the matrix, and calculate them in exactly the same way.
The notion of typical correlation coefficients is not static for standardized transformations, but does not mean that our linear combination coefficients are not, which is the result of the promotion of normal correlation and reconnectivity coefficients.</p>
<h4>Samples typically relevant</h4>
<p>In practical applications, we use sample-related matrices to estimate the overall matrix; using fully consistent computation techniques, we can calculate typical correlation coefficients, typical variables, and the coefficients of their linear combinations.
In practice, we usually use standardized and then calculated (using the matrix) so that the coefficient is also analytical.
It's typically a purely numerical research method, but it's not worth using it, although it does have a concept of score.
The point of understanding is that...
<strong>How to calculate the coefficients of typical relevant variables, the relevant coefficients, and try to justify them by using the combination coefficients, which is the usual statistical analysis technique after a linear combination</strong></p>
<h4>Test of typical correlation coefficients</h4>
<h5>All the typical overall correlation coefficients are tested as zero.</h5>
<p>Consideration of hypothetical tests
&#36;&#36;H_0:\rho_1=\rho_2=\cdotp\cdotp\cdotp=\rho_m=0,\quad H_1:\rho_1,\rho_2,\cdotp\cdotp\cdotp,\rho_m\text{ 至少有一个不为零}&#36;&#36;
Create seemingly statistically comparable
&#36;&#36;\Lambda_1=\prod_{i=1}^m\text{ (1-}r_i^2)&#36;&#36;
For a full-sized&#36;n&#36; When the original hypothesis is established, the statistics are available.
&#36;&#36;Q_1=-\left[n-\frac12(p+q+3)\right]\text{ln}\Lambda_1&#36;&#36;
Obey freedom.&#36;pq&#36;Yes.&#36;\chi^2&#36;Distribution
When the statistical data are too large, the one-sided test rejects the original hypothesis that the correlation between typical variables is significant or otherwise not.</p>
<h5>Test of zero for some typical overall correlation coefficient</h5>
<p>Our natural hope is to use as few logarithms of typical relevant variables as possible, so we need to test the hypothetical zeros of some of the smaller typical relevant coefficients.
Consideration of hypothetical tests
&#36;&#36;H_0:\rho_2=\cdotp\cdotp\cdotp=\rho_m=0,\quad H_1:\rho_2,\cdotp\cdotp\cdotp,\rho_m\text{ 至少有一个不为零}&#36;&#36;
If the original hypothesis is accepted, then only the first pair of typical variables is significant, or we think the second is also significant, and we continue to perform the test.
&#36;&#36;H_0:\rho_3=\cdotp\cdotp\cdotp=\rho_m=0,\quad H_1:\rho_3,\cdotp\cdotp\cdotp,\rho_m\text{ 至少有一个不为零}&#36;&#36;
So cycle, do sequence check.
Test the number of statistics to be
&#36;&#36;\Lambda_{k+1}=\prod_{i=k+1}^m\text{ (1-}r_i^2)&#36;&#36;
Of which&#36;k&#36;It's the order of the test we're looking at at at this moment.&#36;k=0&#36;When it's all over the whole thing, then it's before we remove it.&#36;k&#36;A follow-up check.
For the big enough.&#36;n&#36; When the original hypothesis was established,
&#36;&#36;Q_{k+1}=-\left[n-k-\frac12(p+q+3)+\sum_{i=1}^kr_i^{-2}\right]\text{ln}\Lambda_{k+1}&#36;&#36;
Obey freedom.&#36;(p-k)(q-k)&#36;Yes.&#36;\chi^2&#36;Distribution. Refusal principle as above</p>
<h2>Link analysis</h2>
<h3>Basic concepts</h3>
<h4>Basic concepts of linkages</h4>
<p>The connection that happens when something happens in nature is called a connection.</p>
<p>The connection is...<strong>Two or more variables</strong>It's a kind of important thing that exists between the values.<strong>Some kind of pattern that can be found.</strong></p>
<p>Links can be divided into simple linkages, time series linkages, causal linkages</p>
<ul>
<li>Simple linkages mean exploring whether there is some statistical link between two or more variables without taking into account time sequences. This association is usually based on the co-existence frequency of variables, i.e. the number of times they occur simultaneously with data centralization.</li>
<li>The time series is about the relationship between variables over time.</li>
<li>Causal linkages are a more in-depth linkage analysis that not only explores the linkages between variables but also attempts to determine whether a variable leads to a change in another variable, i.e. whether there is a causal link, which is part of the causal inference.</li>
</ul>
<h4>Basic concepts of linkage analysis</h4>
<p>Linking analysis aims to find hidden linkages between the data entry centralized for the given data record, describing the closeness of the data degrees</p>
<p>There are two types of correlation analysis: the rules of association and the sequence patterns.</p>
<ul>
<li>The rules of association are used to find relevance to different items that appear in the same event</li>
<li>The sequence pattern is similar, but it seeks temporal correlation between events.</li>
</ul>
<h4>Associated rules</h4>
<p>The main target of the linkage rule is transactional data. Library</p>
<p>The rules of association are the knowledge model of the pattern that appears simultaneously between the items in a transaction, and more precisely the rules of association are the impact on the presence of item Y by quantifying the number of items X.</p>
<p>He was created to do a shopping basket analysis, which is to study which customers of goods may buy at the same time as they do at a shopping mall, to help us sell our goods, and to sell them better.</p>
<p><strong>Now, the rules of association can also be used for research into other data, and his core is to study the correlation of a number of qualitative and self-variant variables.</strong></p>
<h4>Formalization of rules on association</h4>
<p>Only a strict mathematical definition will facilitate further modelling, and here we will use the trade database as the basis for the definition.</p>
<p>The transaction data set that the connection rule excavates is recorded as&#36;D&#36; Where \\mathrm{D=<del>{T_{1},</del>T_{2},<del>\ldots,</del>T_{k},~\ldots,T_{n&#125;&#125;}&#36; 其中的 &#36;T k&#36; is called a transaction, and each transaction has a separate number called TID;</p>
<p>All the goods that can be purchased are called "sections." &#36;i_m&#36; The graphs are not used to describe the text.<del>i_{2},</del>\ldots,~i_{m&#125;&#125;}&#36; 是&#36;D&#36;中全体元素的集合 所有的&#36;T_k&#36;都 &#36;Subset of I&#36;</p>
<p>Set two items&#36;X,Y&#36; They all are.&#36;I&#36;=&gt;Y&#36; expression is called the association rule</p>
<h4>Measurement of the associated rules</h4>
<p>All the rules of association have expressions.
&#36;&#36;X\Rightarrow Y[s,c]&#36;&#36;
That's why the connection has confidence.&#36;c&#36; Support &#36;s&#36; He's the measure of our greatest concern for a connection rule.</p>
<ul>
<li>&#36;s&#36; It means the probability of both.</li>
<li>&#36;c&#36; Organisation &#36;X&#36; In the event of a situation &#36;Y&#36;The probability of a situation.</li>
</ul>
<p>The method of calculation is by definition easy to give.</p>
<p>It is not enough to use only the rule of supporting confidence to evaluate linkages, because they do not consider the widespread problem of imbalance, so define it.</p>
<ul>
<li>Expectation of credibility: Description of the associated rule &#36;X = =&gt; Y&#36;在没有任何条件影响时，&#36;Y&#36;在所有交易中出现的频率有多大。即没有&#36;X&#36;的作用下，&#36;Y.A.'s own level of support</li>
<li>Improvement: Description&#36;X&#36;It's right.&#36;Y&#36;And what's the impact of that?</li>
<li>Interest:&#36;\frac{\text{置信度}－\text{支持度&#125;&#125; {Max{\text{置信度},\text{支持度&#125;&#125;}&#36;The greater the interest in a rule is greater than zero, the greater the actual value of utilization; the smaller the actual value of utilization is less than zero.</li>
</ul>
<h4>Linkage rule dig</h4>
<p>The linkage rule that meets the minimum confidence threshold and the minimum support threshold is strong and meaningful.</p>
<p>The problem with excavating linkage rules is the creation of linkages that are more supportive and more credible than the minimum support threshold and the minimum confidence threshold given by the user, respectively.</p>
<h3>Linkage rule dig</h3>
<h4>Basic concepts</h4>
<ul>
<li>&#36;k&#36;Set of items: Include&#36;k&#36;The collection of items</li>
<li>The frequency of the items is the number of services that contain the items</li>
<li>If the frequency of the collection is greater &#36;\text{最小支持度}\times D\text{中的事务总数}&#36;, the collection is called the Frequent Encyclopedia</li>
</ul>
<p>So, the question of digging all the connection rules in the transaction database D can be divided into two sub-issues.</p>
<ul>
<li>Find all frequent items with minimum support</li>
<li>Set) Use frequent set of items to generate desired correlation rules</li>
</ul>
<h4>Apriori algorithm</h4>
<p>The Apriori algorithm uses pre-qualification of the mass of the complex, and is being searched in an iterative manner,&#36;k&#36;- The collection is for inspection.&#36;(k+1)&#36;He's gonna need to scan the data once in a while. Library</p>
<p>He's using nature:<strong>The non-empty subsets of frequent items are also frequent.</strong> Search</p>
<p>His basic implementation model is:</p>
<ul>
<li>Input: Data set&#36;D&#36;, Support threshold&#36;\alpha&#36;</li>
<li>Output: Maximum frequency&#36;k&#36;Set
Steps</li>
</ul>
<ol>
<li>Scan the entire data set and get all the data that have emerged as a candidate for a series of frequent entries (see table 2).&#36;k=1&#36;, with zero frequency sets empty)</li>
<li>I've been digging a lot.&#36;k&#36;Set<ol>
<li>Scan data calculation candidate frequency&#36;k&#36;Support for the set</li>
<li>Remove the frequency of the candidate&#36;k&#36;The data set of the items with a concentration of support below the threshold is frequently k-set. If you get it more often,&#36;k&#36;Items are empty and return frequently&#36;k-1&#36;The collection of the items of the collection is the result of the algorithm, which ends. If you get it more often,&#36;k&#36;Only one set is available, and direct returns are frequent&#36;k&#36;The collection of the items of the collection is the result of the algorithm, which ends.</li>
<li>Based on frequency&#36;k&#36;Entries, connects to generate frequent candidates&#36;k+1&#36;Entries.</li>
</ol>
</li>
<li>You!&#36;k=k+1,&#36;Step 2</li>
</ol>
<p>It's the rules of strong association that satisfy minimum support and minimum confidence, and the rules that come from frequent clusters meet the requirements of support.</p>
<p><strong>The Apriori algorithms are very inefficient, but the underlying underlying of the various associated rules algorithms behind them, most of them have made significant improvements in the Apriori algorithms in terms of efficiency.</strong></p>
<h3>Multiple rules of association</h3>
<ul>
<li>Simple rules of association: like basketball =&gt;Basketball suits, only items.</li>
<li>Quantification of the rules of association: we changed the rules of association that were initially used only for non-variant types.</li>
<li>Multi-dimensional rules: Sex = “Men” =&gt; Purchase = “basketball” involving two dimensions </li>
<li>Cross-layer rules:<ul>
<li>The same level of connection rule: Adidas basketball =&gt; Nike's basketball suit.</li>
<li>Rules of inter-layered association: basketball =&gt; Nike's basketball suit.</li>
</ul>
</li>
</ul>
<p>For the rules on quantitative linkages involving numerical fields, we need to separate the original values to be used in the generation of the linkage rules; this fragmentation can be predefined or produced while the linkage rules are established, which is often more effective, after all, unjustified fragmentation can have negative effects. <a href="/en/blog/2024/03/15/r-visualization-learning-notes/">R Visualization</a> “Prudently processing” section</p>
<p>Digging cross-layer rules requires more advanced algorithms and pre-set settings.</p>
<h3>From Association to Analysis</h3>
<p>• We need an indicator to measure inter-incident correlation or dependence
&#36;A&#36;and&#36;B&#36;Relevance:&#36;corr_{A,B}=\frac P(A\cup B){\mathrm{P(A)P(B)&#125;&#125;=P(B\mid A)/P(B)&#36;
&#36;\cdot&#36;When Set&#36;A&#36;The B-based project is a very important tool for the development of the B-based project.&#36;P(A\cup B)=P(A)P(B)&#36; That's the corrr.<em>{A,B}=1&#36;,表明&#36;A\text{not B, cur}</em>{A,B}&gt;1&#36; 表明&#36;A&#36;与B正相关，corr&#36;_{A,B}&lt;1&#36; 表明&#36;A&#36; is related to B.
Using the relevance indicator for the preceding example, it can be concluded that the relevance of the video and the game will be:
&#36;\mathsf{P}({game,video})/(\mathsf{P}({game})\times\mathsf{P}({video}))=0.4/(0.75\times0.6)=0.89&#36;
Conclusion: negative correlation between video and game</p>
<h2>Measurement of concentration trends and positions</h2>
<h3>Classic concepts in mathematical statistics</h3>
<h4>Mean</h4>
<p>Normal mean is defined as
&#36;&#36;\overline{x}=\frac{\sum_{i=1}^{n}x_{i&#125;&#125;{n}&#36;&#36;
In part, we use weighted averages, which is the difference in importance between data we have.
&#36;&#36;\overline{x}=\frac{\sum_{i=1}^{k}w_{i}x,}{\sum_{i=1}^{k}w_{i&#125;&#125;&#36;&#36;
We can also give the sum of averages.
&#36;&#36;&#123;\sqrt{x_{1}\cdot x_{2}\cdots x_{n&#125;&#125;}&#36;&#36;
But he rarely appears in traditional statistics.</p>
<h4>Medium</h4>
<p>To avoid the effects of some extreme anomalies, to increase robustness, we introduced the concept of median ^dd161e.</p>
<p>The fractional study is the relative location of the data, determined by the location of each sample.
We can use the median as a measure of concentration, and of course we can use the very poor. Medium Points
&#36;&#36;\frac{x_{1}+x_{n&#125;&#125;{2}&#36;&#36;
The data in order can only be studied by the number of numbers and the fractions.</p>
<h4>Number</h4>
<p>The largest number of data available was only studied in some exceptional cases</p>
<p>The concentration of disaggregated data can only be studied in a numerical manner</p>
<h3>Measurement of concentration trends in EDA</h3>
<p>The only central purpose of the measurement method in EDA, both here and in the back, is to increase the robustness of the estimate, which is to make the most of the data available to the population.</p>
<ul>
<li>Statistics are not sensitive to a small amount of large deviation data</li>
<li>Statistics are not sensitive to a large amount of small deviation data.</li>
</ul>
<p>So we've revised the statistics.</p>
<h4>&#36;L&#36;Statistics</h4>
<p>We'll take it. &#36;X_{(i)}&#36; It's the first.&#36;i&#36;Order statistics are intended to increase the robustness of estimates by sequencing statistics, as we do in the median, and this is the relevant paragraph of this paper. Section</p>
<p>&#36;L&#36;The estimated amount is in the form of
&#36;&#36;T=\sum_{i=1}^{r}a_{i}X_{(i)}&#36;&#36;
We'll tell you all about the estimates.&#36;L&#36;The estimates, including the averages, the medians, the numbers, the weighting averages, of course.</p>
<p>I'm introducing you.&#36;L&#36;When we do, we can produce a large range of estimates. <strong>Ending average</strong> That's the calculation of the average after the end of the head.</p>
<ul>
<li>Percent&#36;X&#36; Cut the tail, take it off by 1%.&#36;X&#36;</li>
<li>Median average: take 50 per cent of the median</li>
<li>Median: one to two in the middle</li>
<li>Tri-average: three values in quartiles</li>
</ul>
<h4>Assessment of the effects of concentration trend estimates</h4>
<p>After the next study, we'll give you the following conclusions.&#36;n&#36;- It's a sample.</p>
<ul>
<li>&#36;n&lt;&#36;6.00 using median</li>
<li>&#36;n=7&#36; Two of them are removed from each side's tail.</li>
<li>&#36;n&gt;&#36;8.0 million, 25% on both sides.</li>
</ul>
<h2>Measurement of the discrete trend</h2>
<p>Now let's consider the degree of fragmentation of data in a quantitative study.</p>
<h3>The classic concept of classical mathematical statistics</h3>
<h4>Offset</h4>
<p>We can think of the worst.
&#36;&#36;x_n-x_1&#36;&#36;
Or consider average deviations.
&#36;&#36;\sum_{i=1}^{n}(x_{i}-\overline{x})/n&#36;&#36;</p>
<h4>Difference and Standard</h4>
<p>We've studied this amount more than once in mathematical statistics.
&#36;&#36;\sigma^{2}=\frac{\sum_{i=1}^{N}(x_{i}-\mu)^{2&#125;&#125;{N}&#36;&#36;
It's the most common amount of data dissegregation.</p>
<p>It's a standard deviation, and it has the same units as the original data. Root</p>
<p>The sample variance was amended to address the non-selectivity of the sample (and the standard deviation was subsequently amended)
&#36;&#36;s^{2}=\frac{\sum_{i=1}^{\pi}(x_{i}-\bar{x})^{2&#125;&#125;{n-1}&#36;&#36;</p>
<h4>Measure of relative deviation (variability factor)</h4>
<p>The amount given below is ununited.</p>
<p>Defines the variable factor as
&#36;&#36;V=\frac{\sigma}{\mu}&#36;&#36;</p>
<h3>Isolation measures in EDA</h3>
<p>Here we remove the measure method from the front, some of the measures that are of a robust nature.</p>
<h4>Medium-digit difference in sample</h4>
<p>&#36;&#36;&#123;AD}={\frac{1}{n&#125;&#125;\sum_{i=1}^{n}|x_{i}-{M}|&#36;&#36;
of which&#36;M&#36; is the median of the sample </p>
<h4>The median is absolutely different from the median.</h4>
<p>&#36;&#36;&#123;MAD}=\mathrm{median}<em>{i}{\left|x</em>- \\boldsymbol
We calculated all the differences and then we calculated the median.
The median absolute difference between the R and the R is about 1.4, so the aim is to estimate the sample variance, which we can basically ignore in the actual calculations.</p>
<h4>Four-point difference.</h4>
<p>We'll consider the margin of the quarter.
&#36;&#36;IQR = d_{F}=F_{U}-F_{L}&#36;&#36;
And what we're doing is we're using it very often in the study of dissegregation and abnormality.<strong>Box chart and five-digit summary</strong></p>
<h4>Assessment of the effects of several discrete measures</h4>
<p>We can get a very simple but important conclusion from looking at the extent to which these statistics are dissegregated in different distributions.</p>
<p><strong>The quartet is the best possible sum of the size of the sample dissipation.</strong></p>
<h2>Measurement of the distribution shape</h2>
<p>They focus on two characteristics of the sample, whether their centres are concentrated (measures of concentration trends) and whether they are uneven (measures of dispersing trends) are both the shape of the distribution.
Their characteristic is that they compare to the normal distribution.</p>
<h3>Skewer factor (skewness)</h3>
<p>&#36;&#36;g_1=\frac{n}{(n-1)(n-2)s^3}\sum_{i=1}^{n}(x_i-\overline{x})^3=\frac{n^2\mu_3}{(n-1)(n-2)s^3},&#36;&#36;
The symmetrical symmetry factor is 0</p>
<h3>Peak coefficient (kurtosis)</h3>
<p>&#36;&#36;\begin{array}{rcl}g_2&amp;=&amp;\frac{n(n+1)}{(n-1)(n-2)(n-3)s^4}\sum_{i=1}^n(x_i-\overline{x})^4-3\frac{(n-1)^2}{(n-2)(n-3)}\&amp;=&amp;\\frac{n^2(n+1)\mu}4}(n-1)(n-2)(n-3)s^4}-3\frac{(n-1)^2}(n-2)(n-3)},\end{array} I'm sorry.
And then, when you're looking at the peak of distribution, you're going to have a peak greater than zero when you're going to be going to the normal distribution, or you're going to have a small peak of zero.</p>
<h2>Descriptive statistical visualization</h2>
<p>We don't need to discuss too many visual tools in basic descriptive statistics, and the basic descriptive statistical techniques presented here only involve a part of Excel, which includes</p>
<ul>
<li>Studying numerical fundamentals:<a href="/en/blog/2024/03/15/r-visualization-learning-notes/">R Visualization</a> . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .  <a href="/en/blog/2024/03/15/r-visualization-learning-notes/">R Visualization</a> The "Cleveland Plot" section</li>
<li>Basic forms of distribution of research:<a href="/en/blog/2024/03/15/r-visualization-learning-notes/">R Visualization</a> . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .   <a href="/en/blog/2024/03/15/r-visualization-learning-notes/">R Visualization</a> and the “Regular Nuclear Density Estimates” section of the </li>
<li>Study of basic ratios:<a href="/en/blog/2024/03/15/r-visualization-learning-notes/">R Visualization</a> "The pie-tarts section."</li>
<li>Study the existence of extreme values:<a href="/en/blog/2024/03/15/r-visualization-learning-notes/">R Visualization</a> I'm not sure I'm gonna be able to get a picture of this.</li>
</ul>
<p>Especially, we're here to add a more non-usual but important graphic to statistics itself.<a href="/en/blog/2023/03/18/mathematical-statistics-notes/">Mathematical statistics</a> Section on the " Empirical Distribution Function "</p>
