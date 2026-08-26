---
title: 'Advanced Machine Learning: Unsupervised and Semi-Supervised Learning'
title_zh: 机器学习进阶：无监督学习与半监督学习
date: 2024-04-06 01:38:58 +0800
categories:
- Machine Learning
- Classical Machine Learning
tags:
- Machine Learning
- Unsupervised Learning
- Semi-Supervised Learning
author: Hyacehila
mathjax: true
hidden: true
excerpt: Covers clustering, kernel density estimation, spectral and graph clustering, anomaly detection, and semi-supervised
  learning.
description: Covers clustering, kernel density estimation, spectral and graph clustering, anomaly detection, and semi-supervised
  learning.
excerpt_zh: 整理聚类、核密度估计、谱聚类、图聚类、异常检测和半监督学习。
permalink: /blog/2024/04/06/advanced-machine-learning-unsupervised-learning/
lang: en
translation_key: 2024-04-06-advanced-machine-learning-unsupervised-learning
translation_status: machine
translation_source_hash: 822bd1b89c54e3decdd8fd3f7ca7418329a2338aa8bdbce619a82a8676eaae71
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<h2>Cluster</h2>
<p>Starting with this chapter, we're introducing some of the classic unsupervised learning tools that are used extensively in data mining, especially in the EDA direction. </p>
<h3>Cluster tasks</h3>
<p>Clusters attempt to divide samples from data sets into subsets that are usually non-interconnected, each of which is referred to as a "cluster." Through this division, each cluster may correspond to some potential concepts.</p>
<p>The clustering process can only automatically form a cluster structure, and the syntax of the cluster is subject to user control.</p>
<p>Clusters can be used as a separate process to find the inner distribution structure of the data or as subdivisions
Other learning tasks.</p>
<h3>Cluster performance measures</h3>
<p>References <a href="/en/blog/2026/07/31/clustering-model-evaluation/">Performance measures of the cluster model</a></p>
<h3>Level Cluster</h3>
<p>We've already described it in multiple statistics, where we don't repeat the narrative, but we've changed the perspective of the idea of clustering.</p>
<p>System cluster is a cluster structure that attempts to divide data sets at different levels, resulting in tree formation. Datasets can be divided using a “bottom-up” aggregation strategy, or a “top-down” fragmentation strategy.</p>
<h3>Prototype Cluster</h3>
<p>Prototype-based clustering  <strong>"Prototype" means a representative point in the sample space.</strong>  Very common in real-life cluster assignments</p>
<p>Using different prototypes, different solvency, different algorithms are created that give different group results.</p>
<h4>k Average value algorithm</h4>
<p>The original k-mean algorithm is a GP problem that requires minimizing square error.
&#36;E=sum sum sum sum symbol{\<em>{i}||</em>2}&#36;&#36;
It's not easy to calculate, so we're using an iterative approach to approximate solvency.
This algorithm has the character of automatic cessation, which is not the same as a system cluster.</p>
<p><strong>k average algorithms also have the means to nuclei, and we map the data to higher dimensions to deal with non-linear clusters, depending on our actual data.</strong></p>
<h4>Quantification of learning vectors</h4>
<p>Similar to k average algorithm,&quot;Learning Vector Quantification, Jane
LVQ) is also an attempt to find a prototype vector to paint a cluster structure  </p>
<p>But... <strong>LVQ assumes that samples of data have category tags to support grouping, which is actually a supervisory learning algorithm</strong> The goal of training is to find a team.&#36;n&#36;Wedge vectors. Each prototype represents a cluster.</p>
<p>What's our algorithm thinking?</p>
<ol>
<li>Initializing prototype vectors</li>
<li>Each sample found the nearest prototype vector.</li>
<li>Update prototype vector</li>
<li>Repeat 2-3 until cessation conditions are met.
The core is how to update the prototype vector.</li>
</ol>
<p>Intuitively, for the sample.<em>j,&#36;若最近的原型向量&#36;p_i^<em>&#36; 与&#36;x_j&#36; 的类别标记相同，则令&#36;p_i&#36;</em> Center&#36;x_j&#36; The new prototype vector is
&#36;US&#36;P}(prime}=p</em>{i^{<em>&#125;&#125;+\eta\cdot(\boldsymbol{x}<em>{j}-\boldsymbol{p}</em>{i^{</em>What's wrong?
If the category labels are different, then the principle of distance change is that&#36;|(1+\eta)\cdot||p_{i^{*&#125;&#125;-x_{j}||_{2}&#36;
Ours.&#36;\eta&#36;Still learning.</p>
<h4>Gaussian Mixed Cluster</h4>
<p>What's different from before is that we use probabilistic models to express the probabilities of a cluster prototype, and we use models to study the probability of each sample falling into each category. Instead of giving a definite classification.</p>
<h3>Density Cluster</h3>
<p>Such algorithms assume that the cluster structure can be determined by the closeness of the sample distribution; the density cluster algorithm examines the connectivity between samples from the point of view of sample density and is based on the continuous expansion of the cluster to obtain final cluster results.</p>
<h4>DBSCAN</h4>
<p>DBSCAN is a famous density cluster algorithm based on the closeness of a group of neighbourhood parameters for given&#36;n&#36;The DDS defines the concepts below</p>
<ul>
<li>&#36;\epsilon&#36;- Neighborhood: Yeah. &#36;x_j\in D&#36;Other &#36;\epsilon&#36;- Neighborhood contains sample collections. &#36;D&#36; Center with &#36;\dot{x}<em>j&#36; 的距离不大于 &#36;\epsilon&#36; 的样&#36;\text{i}</em>{\epsilon}(\boldsymbol{x}<em>{j})={\boldsymbol{x}</em>{i}\in D\mid\mathrm{dist}(\boldsymbol{x}<em>{i},\boldsymbol{x}</em>{j})\leqslant\epsilon};&#36;</li>
<li>Directly decreasing-reachable:&#36;x_j&#36; in&#36;x_i&#36; Yes. &#36;\epsilon&#36;- In the neighborhood, and...&#36;x_i&#36; is the core object, then &#36;x_j&#36; by &#36;x_i&#36; Direct density;</li>
<li>Density-reachable: &#36;x_i&#36; and &#36;x_j&#36;, if sample sequence exists &#36;p_1,p_2,\ldots,p_n&#36; Where \\boldsymbol{p} 1=\boldsymbol{i,\dot{p} n\doteq\boldsymbol{x}<em>j&#36; 且&#36;p</em>{i+1}&#36; 由&#36;p_i&#36;密度直达，则称 &#36;x_j&#36; 由 &#36;x i&#36; Density to reach;</li>
<li>Density-connected: &#36;x_i&#36; and &#36;x_j&#36;, if existing &#36;x_k&#36; Make &#36;x_i&#36; and &#36;x_j&#36; Both by&#36;x_k&#36; Density to be achieved, in other words &#36;x_i&#36; and &#36;x_j&#36; Density connections ... (a new sample was added as a connection)</li>
</ul>
<p>Based on these concepts, the DNSCAN definition cluster is: the largest-density-connected collection of samples from a density-accessible relationship <strong>Which means...&#36;x&#36;For the core object, the collection of all the densityable samples is our cluster.</strong></p>
<p>The DBSCAN algorithm only needs to select a core object, find its cluster, then find a new core object in the rest of the sample, repeat the search, and know that we think it's over.</p>
<p>Finally, there was no group sample called noise.</p>
<h4>Nuclear density estimates</h4>
<p>Nuclear density estimates are not a cluster approach, but are closely linked to the cluster. Density estimates want to find a point-intensive area to determine an unknown probability density function, which can be used for a cluster.</p>
<p>As a method that did not require parameters, he did not assume the existence of a probability model, but infer the probability density at the bottom.</p>
<h5>One dollar nuclear density</h5>
<p>The cumulative distribution function is very easy to give.
&#36;&#36;\hat{F}(x)=\frac{1}{n}\sum_{i=1}^nI(x_i\leqslant x)&#36;&#36;
We can use his guide to estimate density and consider a small window.
&#36;&#36;\hat{f}(x)=\frac{\hat{F}(x+\frac{h}{2})-\hat{F}(x-\frac{h}{2})}{h}=\frac{k/n}{h}=\frac{k}{nh}&#36;&#36;
&#36;h&#36;The choice is very important. It's very big.&#36;h&#36;You can smooth out the density estimates, too small.&#36;h&#36;It would result in too few points being included and seriously inaccurate estimates.</p>
<p>The nuclear density estimate relies on a non-negative, symmetrical nuclear function that builds up to 1&#36;K&#36;that is:&#36;K(x)\geqslant0,K(-x)=&#36; &#36;K(x)&#36; For all&#36;x&#36;and&#36;\int K( x)&#36;d&#36;x= 1&#36;I don't know. So,&#36;K&#36;It's actually a probability density function.</p>
<p><strong>Dispersed Nuclear</strong> We can rewrite the density estimates in front of the discrete core.
&#36;&#36;\hat{f}(x)=\frac{1}{nh}\sum_{i=1}^nK\left(\frac{x-x_i}{h}\right)&#36;&#36;
of which&#36;K&#36;Yes
&#36;&#36;\left.K(z)=\left(begin{array}ll}1&amp;|z|\leqslant\frac{1}{2}\0&amp;\text{Others}\right.\right.&#36;</p>
<p>In order to be more smooth, we can think about it. <strong>Gaussian nuclear</strong> Define
&#36;&#36;K(z)=\frac{1}{\sqrt{2\pi&#125;&#125;\exp\left{-\frac{z^2}{2}\right}&#36;&#36;
It's the beginning of this era.&#36;\hat{f}(x)&#36;Yes.
&#36;&#36;K\left(\frac{x-x_i}{h}\right)=\frac{1}{\sqrt{2\pi&#125;&#125;\exp\left{-\frac{(x-x_i)^2}{2h^2}\right}&#36;&#36;
<strong>At this point in the larger zone, the points are factored into local density estimates with different probabilistic weights.</strong></p>
<h5>Multiple nuclear density</h5>
<p>For the estimation of one&#36;d&#36;Dots&#36;x=(x_1,x_2,\cdots,x_d)^\mathrm{T}&#36;Probability density, definition&#36;d&#36;The "Window" for V is&#36;d&#36;One of the supercubes in space, that's the one that's going to...&#36;x&#36;It's central and it's long.&#36;h&#36;the supercube. Like this.&#36;d&#36;The size of the dimension supercube is:
&#36;&#36;\mathrm{vol}(H_d(h))=h^d&#36;&#36;
Like a dollar, we can write a nuclear density estimate.
&#36;&#36;\hat{f}(x)=\frac{1}{nh^d}\sum_{i=1}^nK\left(\frac{x-x_i}{h}\right)&#36;&#36;
of which&#36;K&#36;It's a multiple nuclear function. It's a multiple probabilistic density function.<a href="/en/blog/2024/03/28/machine-learning-introduction-supervised-learning/">Introduction to Machine Learning and Supervision Learning: Nuclear Methods</a>The nuclear function that's in it.</p>
<p>We can put<strong>Dispersed Nuclear</strong>Define
&#36;&#36;\left.K(z)=\left(begin{array}ll}1&amp;|z_j|\leqslant\frac{1}{2}\0&amp;\text Other Organiser&#36;&#36;
高斯核定义为
&#36;&#36;K(z)=\frac{1}{(2\pi)^{d/2&#125;&#125;\exp\left{-\frac{z^\mathrm{T}z}{2}\right}&#36;&#36;
代入 &#36;z=\frac&#123;&#123;x-x_{i&#125;&#125;}{h}&#36; 有
&#36;&#36;K(\boldsymbol{z})=\frac{1}{(2\pi)^{d/2&#125;&#125;\exp\left{-\frac{(\boldsymbol{x}-\boldsymbol{x}_i)^\mathrm{T}(\boldsymbol{x}-\boldsymbol{x}_i)}{2h^2}\right}&#36;&#36;</p>
<h5>Nearest nuclear density</h5>
<p>The way forward is to look for points within a fixed volume to estimate nuclear density; the other is the number of fixed points&#36;k&#36; Allow volume changes; commonly referred to as density estimates&#36;k&#36;The approach to proximity is also a non-parametric approach.</p>
<p>Number of given neighbours&#36;k&#36;estimate&#36;x&#36;The density is as follows:
&#36;&#36;\hat{f}(\boldsymbol{x})=\frac k{n\mathrm{vol}(S_d(h_{\boldsymbol{x&#125;&#125;))}&#36;&#36;
of which&#36;h_x&#36;Yes.&#36;x&#36;To the first of it.&#36;k&#36;A recent neighbor's distance.&#36;vol(S_d(h_x))&#36;Yes.&#36;x&#36;For the center,&#36;h_x&#36;Radius&#36;d&#36;Vi superspherical&#36;S_d(h_x)&#36;The volume. In other words, width (radio)&#36;h_x&#36;Now it's a dependency.&#36;x&#36;and&#36;k&#36;.</p>
<h4>DENCLUE</h4>
<p>On the basis of nuclear density, we can give a generic cluster method based on density; find peaks in density through gradient optimization; and then find areas with density above a certain threshold.</p>
<p>Definitions: &#36;x^<em>&#36;是概率密度函数&#36;f. A local large point, referred to as*<em>Density attraction</em></em>(density attractor) A density attraction from&#36;x&#36;Start with the gradient down. is the maximum gradient of the probability density function. By extrapolating, we can give the formula below.
&#36;US&#36;x sum sum i=nK(\frac{\boldsymbol{t-\boldsymbol{x}\boldsymbol{x}<em>i}{\sum</em>{i=1}^nK(\frac{\boldsymbol{x}_t-\boldsymbol{x}_i}{h})}&#36;&#36;</p>
<p>Definition: Give a cluster&#36;C\subseteq D&#36;, if all points &#36;x\in C&#36; They're attracted to a single density attraction. &#36;x^<em>&#36;,使得&#36;\hat{f}(x^</em>)\geqslant\xi&#36;,其中&#36;\xi&#36; is a user-defined minimum density threshold called<strong>Center defined cluster (center-defined cluster)</strong>that is:
That's a lot of money.<em>)=\frac1{nh^d}\sum_{i=1}^nK\left(\frac{x^</em>-x_i}h\right)\geqslant\xi &#36;&#36;</p>
<p>Definition: A cluster of random shapes&#36;C\subseteq D&#36;It's one.<strong>Density-based cluster</strong>, if there is a group of density attractions &#36;x 1^<em>,x_2^</em>,\cdots, x m*, making:</p>
<ul>
<li>Every point.&#36;x\in C&#36;They're attracted to something.&#36;x_i^*;&#36;</li>
<li>Every density attraction is more dense.&#36;\xi&#36;i.e.&#36;\hat{f}(x_i^*)\geqslant\xi;&#36;</li>
<li>Any two density attractions &#36;x i^<em>&#36;和&#36;x_j^</em>&#36;都是密度可达的，即存在一条从&#36;x_i^<em>&#36;到&#36;x_j^</em>&#36;的路径，使得所有在该路径上的点&#36;y&#36;都有&#36;\hat{f}(y)\geqslant\xi&#36;。</li>
</ul>
<p>Here's the idea of the DENCLUE algorithm.</p>
<ol>
<li>Calculating the density attraction of each point &#36;x i^<em>&#36;，如果大于阈值&#36;\xi&#36;，则将其加入吸引子集合&#36;A&#36;，对应点加入被点信息的集合&#36;R(x_i^{</em>})&#36; </li>
<li>Find the largest subset of all attractions &#36;C&#36; Make sure any of them attracts a child density.</li>
<li>The biggest subset of these attractions. &#36;C&#36;It forms a seed based on density, and it attracts points into the cluster to form a cluster result.</li>
</ol>
<p>I can prove it.<strong>DBSCAN is a special case of DENCLUE based on a general nuclear density estimation grouping method</strong>I don't know. If you order&#36;h=\epsilon&#36;and&#36;\xi=&#36;Minpts, with a discrete nucleus, DENCLUE will get the same results as DBSCAN. Each density attraction corresponds to a core point, and the aggregation of connecting core points defines a cluster of attractions based on density.</p>
<p>It can also prove that it's the right choice.&#36;h&#36;and&#36;\xi&#36;I don't know.<strong>K-means is also a special case of concentration based on density.</strong>and the density attraction corresponds to the cluster centre. In addition, notably,<strong>The density-based approach can change.&#36;\xi&#36;Thresholds, generate layers</strong>I don't know. For example, reduction&#36;\xi&#36;Value makes several clusters merge together. At the same time, if the peak density is greater than the decrease,&#36;\xi&#36;value, a new cluster may be generated.</p>
<h3>Spectrums and graphs</h3>
<p>This section looks at clustering on the map, he and the hierarchical clustering, the spectrolysis of the matrix, and the nuclear-based clustering, which we will then explain clearly.</p>
<h4>Figure and matrix</h4>
<p>Assignance&#36;\mathbb{R}^d&#36;Medium&#36;n&#36;D=x i}<em>i=1^n&#36;,令&#36;A&#36;代表这些点之间的&#36;A pair of likeness matrices:
&#36;A = \begin{matrix}a</em>{11}&amp;a_{12}&amp;\cdots&amp;a_{1n}\a_{21}&amp;a_{22}&amp;\cdots&amp;a_{2n}\\vdots&amp;\vdots&amp;\cdots&amp;\vdots\a_{n1}&amp;a_{n2}&amp;\cdots&amp;A \end{matrix} &#36;
of which&#36;A(i,j)=a_{ij}&#36;Expression&#36;x_i&#36;and&#36;x_j&#36;Similarity descriptive statistics and similarity factors in visualization. We're asking for symmetry and non-negativeness.&#36;a_{ij}=a_{ji}&#36;and&#36;a_{ij}\geqslant0&#36;。</p>
<p>Matrix&#36;A&#36;It can be seen as a rights-based map.&#36;G=(V,E)&#36;It has a rights-based neighbourhood matrix, which represents the neighbourhood, thus transforming samples into graphic data for analysis.</p>
<p>For each vertex&#36;x_i&#36;We can calculate his degree.&#36;d_i&#36;
&#36;&#36;d_i=\sum_{j=1}^na_{ij}&#36;&#36;
So we can export the array of degrees to
&#36;&#36;\left.\Delta=\left(\begin{array}{cc}d 1&amp;0&amp;\cdots&amp;0\0&amp;d_2&amp;\cdots&amp;0\\vdots&amp;\vdots&amp;\ddots&amp;\vdots\0&amp;0&amp;\cdots&amp;d_n\end{array}\right.\right)=\begin{pmatrix}\sum_{j=1}^na_{1j}&amp;0&amp;\cdots&amp;0\0&amp;\sum_{j=1}^na_{2j}&amp;\cdots&amp;0\\vdots&amp;\vdots&amp;\ddots&amp;\vdots\0&amp;0&amp;\cdots&amp;\sum_{j=1}^na_{nj}\end{pmatrix}&#36;&#36;</p>
<p>By dividing each line of the adjacent matrix by the degree of the corresponding node, we can get <strong>Normalization of the Neighbourhood Matrix</strong> As follows:
That's right.&amp;\frac{a_{12&#125;&#125;{d_1}&amp;\cdots&amp;\frac{a_{1n&#125;&#125;{d_1}\\frac{a_{21&#125;&#125;{d_2}&amp;\frac{a_{22&#125;&#125;{d_2}&amp;\cdots&amp;\frac{a_{2n&#125;&#125;{d_2}\\vdots&amp;\vdots&amp;\ddots&amp;\vdots\\frac{a_{n1&#125;&#125;{d_n}&amp;\frac{a_{n2&#125;&#125;{d_n}&amp;\cdots&amp;\frac{a_{nn&#125;&#125;{d_n}\end{pmatrix}\end{gathered}&#36;&#36;</p>
<p>Then we can define <strong>The Lapras Matrix of the Fig.</strong> As follows:
&#36;&#36;L=\Delta-A&#36;&#36;
That's...
That's right.&amp;-a_{12}&amp;\cdots&amp;-a_{1n}\-a_{21}&amp;\sum_{j\neq2}a_{2j}&amp;\cdots&amp;-a_{2n}\\vdots&amp;\vdots&amp;\ddots&amp;\vdots\-a_{n1}&amp;-a_{n2}&amp;\cdots&amp;\sum j\neq}a\end{matrix} &#36;
He's a semi-positive symmetrical matrix; one of them.&#36;n&#36;A non-negative real number feature and its characteristic vector is positive; the La Plas matrix of the figure can also be calibrated and then calibrated to vary and obtained <strong>A unified Tulapras matrix</strong></p>
<h4>Figure cutting</h4>
<p>A picture.&#36;k&#36;Passepartout wants a good division.&#36;C&#36;This makes the same cluster more similar and the different clusters less similar, which is very intuitive. Let us begin with some basic knowledge of graphic cutting.</p>
<p>For a given rights chart and its similarity matrix, whatever&#36;S,T\subset V&#36;  We define&#36;W(S,T)&#36; Other Organiser&#36;S&#36;The other node is&#36;V&#36;and
&#36;&#36;W(S,T)=\sum_{v_i\in S}\sum_{v_j\in T}a_{ij}&#36;&#36;</p>
<p>Organisation&#36;S\subseteq V&#36;♪ With ♪&#36;\bar{S}&#36;This represents a contours of complementarities, i.e.&#36;\bar{S}=V-S&#36;I don't know. One of the figures <strong>(vertext cut)</strong> Defined&#36;V&#36;Division to&#36;S\subset V&#36;and&#36;\bar{S}&#36;I don't know. Cut weight is defined as&#36;S&#36;and&#36;\bar{S}&#36;the sum of the weights of the edges formed by the vertex, i.e.&#36;W(S,\bar{S})&#36;</p>
<p>Give one contains&#36;k&#36;A cluster.&#36;C={C_1,\cdots,C_k}&#36;One. <strong>Cluster&#36;C_i&#36;Size (size)</strong> Defined as the number of nodes in a cluster, i.e.&#36;|C_i|&#36;I don't know. A cluster. <strong>&#36;C_i&#36;Volume (volume)</strong> Defines the sum of all power values containing the edges of the vertex in the cluster:
&#36;&#36;\mathrm{vol}(C_i)=\sum_{v_j\in C_i}d_j=\sum_{v_j\in C_i}\sum_{v_r\in V}a_{jr}=W(C_i,V)&#36;&#36;</p>
<p>You!&#36;c_i&#36;He's satisfied.
&#36;&#36;\left.c =left{array}ll}1&amp;v_j\in C_i\0&amp;v_j\notin C_i\end{array}\right.\right.&#36;&#36;</p>
<p>Here's the weight of the cut in matrix mode:
&#36;&#36;00begin{aligned}W (C I},\overline{C I})&amp;=\sum_{v_r\in C_i}\sum_{v_s\in V-C_i}a_{rs}=W(C_i,V)-W(C_i,C_i)\&amp;== sync, corrected by elderman == @elder man
He's connected to the Lapras matrix of similarity.</p>
<p>So, we're almost there.</p>
<h4>Target function for grouping (minimised)</h4>
<p>The cluster target function can be just one.&#36;k&#36;With regard to the optimization of roads, we would like to look for some good optimization objectives, one of two common ones.</p>
<p><strong>Scale cut</strong></p>
<p>&#36;k&#36;The ratio cut on the road is defined as follows:
&#36;US&#36;US&#36;\m m m m m m m m m m m=m=m=m=m=m=m=m=m=m=m=m=m=m=m=m=m=m=m=m=m=m=m=m=m=m=m=m=m=m=m=m=m=m=m=m=m=m=m=m=m=m m m=m m m=m=m m m m m m m m m m m m m m m m m m m m m m m m m m m m ,\m m m m m m m m m m ,\m m m m m m m m m m m m m ,\m m\m m m\m\m\m\m\m\m\m\m\<em>i)}{|C_i|}=\sum</em>I'm sorry.
Proportional cut attempts to minimize from the cluster&#36;C_i&#36;To the others.&#36;\overline{C}_i&#36;to take into account the size of each cluster. It can be observed that the target function is smaller when the right value of the cut is minimized and larger.</p>
<p>Unfortunately, for the binary range,&#36;c_i&#36;The rationing target is hard for NPs. An obvious way to relax is to allow&#36;c_i&#36;Takes any real value.</p>
<p><strong>Separability</strong></p>
<p>The combined cut is similar to the ratio, except that it divides the weight of each cluster by the size of the cluster, not its size. The target function gives:
&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;USK K\LE{I, \overline}&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US  K\K\N{, \K{L{, \E, \overline{C}<em>i)}{\mathrm{vol}(C_i)}=\sum</em>I don't know.
There is also the same optimisation problem as proportional circumcision, which needs to be solved with relaxation.</p>
<h4>Spectrometry</h4>
<p>Based on the optimised algorithms given earlier (although we do not extrapolate the specific solvency process and results of the algorithms), here we can point out that we actually need only to study the given matrix, such as the La Plas matrix).&#36;L&#36;La Plass Matrix.&#36;L^s&#36;;and then calculates its feature value and feature vector, and selects the maximum feature value or the smallest feature value&#36;k&#36;as an indicator vector)</p>
<p>We finally got a series of real-value clusters.&#36;u_i&#36;  Both &#36;u_n...u_{n-k+1}&#36;  As these indicator vectors are defunct, they are not binary and need to be further addressed. We see him as a new matrix:
&#36;&#36;\left.U=\left&amp;|&amp;&amp;|\u_n&amp;u_{n-1}&amp;\cdots&amp;u_{n-k+1}\|&amp;|&amp;&amp;|\end{array}\right.\right)=\begin{pmatrix}u_{n,1}&amp;u_{n-1,1}&amp;\cdots&amp;u_{n-k+1,1}\u_{n,2}&amp;u_{n-1,2}&amp;\cdots&amp;u_{n-k+1,2}\|&amp;|&amp;\cdots&amp;|\u_{n,n}&amp;u_{n-1,n}&amp;\cdots&amp;u_{n-k+1,n}\end{pmatrix}&#36;&#36;
每一行进行归一化处理：
&#36;&#36;y_i=\frac{1}{\sqrt{\sum_{j=1}^ku_{n-j+1,i}^2&#125;&#125;(u_{n,i},u_{n-1,i},\cdots,u_{n-k+1,i})^\mathrm{T}&#36;&#36;
目前每一行都是一个单位向量，如下
&#36;&#36;\left.Y=\left(\begin{array}{ccc}-&amp;y_1^\mathrm{T}&amp;-\-&amp;y_2^\mathrm{T}&amp;-\&amp;\vdots&amp;\-&amp;y_n^\mathrm{T}&amp;- I'm sorry.
Now we can use fast concentration algorithms like K-means to make the current&#36;n&#36;Line Vector Considers&#36;n&#36;Point grouping&#36;k&#36;A cluster is the final cluster result.<strong>This spectral method is only appropriate for dealing with the similarity matrix and the La Plass matrix after integration.</strong></p>
<h4>Target function for grouping (maximized)</h4>
<p>Let's discuss two more cluster target functions.</p>
<p><strong>Average weight</strong> Objectives are defined as follows:
&#36;&#36;\max_{\mathcal{C&#125;&#125;J_{aw}(\mathcal{C})=\sum_{i=1}^k\frac{W(C_i,C_i)}{|C_i|}=\sum_{i=1}^k\frac{c_i^\mathrm{T}Ac_i}{c_i^\mathrm{T}c_i}&#36;&#36;
We still need to find a solution.</p>
<p><strong>Average weight with K-mes</strong></p>
<p>We're here to discuss an interesting connection, if it's a powered neighbour matrix.&#36;A&#36;represents the nuclear value of a pair of points, and has&#36;a_{ij}=K(x_i,x_j)&#36;, the nuclear K-means square error and target can be used for grouping. SSE targets are:
&#36;&#36;00begin{aligned}\min &amp;=\sum_{j=1}^nK(\boldsymbol{x}<em>i,\boldsymbol{x}<em>j)-\sum</em>{i=1}^k\frac{1}{|C_i|}\sum</em>{\boldsymbol{x}<em>r\in C_i}\sum</em>{\boldsymbol{x}<em>s\in C_i}K(\boldsymbol{x}<em>r,\boldsymbol{x}<em>s)\&amp;=\sum</em>{j=1}^na</em>{jj}-\sum</em>{i=1}^k\frac{1}{|C_i|}\sum_{v_r\in C_i}\sum_{v_s\in C_i}a_{rs}\&amp;=\sum_{j=1}^na_{jj}-\sum_{i=1}^k\frac{c_i^\mathrm{T}Ac_i}{c_i^\mathrm{T}c_i}\&amp;== sync, corrected by elderman ==
I can see that.&#36;\sum_{j=1}^na_{jj}&#36;Not related to cluster, minimizing SSE is maximizing the average weight of AW, and the two issues are the same as the final equivalent; for this NP problem, the nuclear K-means use a greedy hiatus to solve the problem, and the average power study its laxity.</p>
<p><strong>Modularity</strong> The module wants to discuss the extent to which the same cluster is connected.
&#36; \begin{aligned}\max {mathcal{C}&amp;=\sum_{i=1}^k\left(\frac{c_i^\mathrm{T}Ac_i}{\mathrm{tr}(\boldsymbol{\Delta})}-\frac{(\boldsymbol{d}^\mathrm{T}c_i)^2}{\mathrm{tr}(\boldsymbol{\Delta})^2}\right)\&amp;=\sum_{i=1}^k\left(c_i^\mathrm{T}\left(\frac{A}{\mathrm{tr}(\boldsymbol{\Delta})}\right)c_i-c_i^\mathrm{T}\left(\frac{d\cdot d_i^\mathrm{T&#125;&#125;{\mathrm{tr}(\boldsymbol{\Delta})^2}\right)c_i\right)\&amp;=\sum_{i=1}^kc_i^\mathrm{T}Qc_i\end{aligned}&#36;&#36;
其中&#36;Q&#36;是模块度矩阵为
&#36;&#36;Q=\mathrm{tr}(\bardsymbol{\D\cdot d i^mathrm{tr} (\bardsymbol{\Delta}
There's still a need for a loose solution.</p>
<p><strong>Normalized modularity equals average weights, and thus, in some cases, equals nuclear K-means</strong></p>
<h3>Markov Cluster</h3>
<p>The Marcov cluster uses the original transfer matrix as the transfer matrix for the Malkov chain, hoping to obtain the final cluster results by simulating the transfer of the Mars chain, and we will briefly describe its thinking below.</p>
<p>Give a chart&#36;G&#36;. . . . . . .&#36;A&#36;, corresponding to the normalized adjacent matrix is&#36;M=\Delta^{-1}A&#36;I don't know. Matrix&#36;M&#36;It can be seen as&#36;n\times n&#36;transfer matrix (transaction matrix), with each array item&#36;m_{ij}=\frac{a_{ij&#125;&#125;{d_i}&#36;It could be seen as a node.&#36;i&#36;Go to Node&#36;j&#36;The probability. He's meeting the conditions of the Ma's chain transfer matrix.</p>
<p>Assuming that this is a chain of horseback, that is, the transfer probability matrix has nothing to do with the current position, then we can calculate his position.&#36;n&#36;Step transfer probability matrix. Last time.&#36;n&#36;We'll stop the calculations.</p>
<p><strong>The last available transfer probability matrix can draw a transfer probability map based on which we can naturally discover clusters, such as the lower matrix.</strong>
&#36;&#36;\boldsymbol{M}=\begin{pmatrix}&amp;1&amp;2&amp;3&amp;4&amp;5&amp;6&amp;7\1&amp;0&amp;0&amp;0&amp;1&amp;0&amp;0&amp;0\2&amp;0&amp;0&amp;0&amp;1&amp;0&amp;0&amp;0\3&amp;0&amp;0&amp;0&amp;1&amp;0&amp;0&amp;0\4&amp;0&amp;0&amp;0&amp;1&amp;0&amp;0&amp;0\5&amp;0&amp;0&amp;0&amp;0&amp;0&amp;0.5&amp;0.5\6&amp;0&amp;0&amp;0&amp;0&amp;0&amp;0.5&amp;0.5\7&amp;0&amp;0&amp;0&amp;0&amp;0&amp;0.5&amp;0.5\end{matrix}
The numbering of the first column and the first line of the expression point is not the transfer probability.</p>
<h2>Unusual detection</h2>
<p>The most dominant abnormality algorithm is similar to ours. <a href="/en/blog/2024/02/29/exploratory-data-analysis-learning-notes/">Explored data analysis: processing of anomalies</a>The information presented in this section shows that, in some cases, anomalies are information and therefore need to be identified. The most dominant method is probabilities-based treatment, which can be judged when the probability of a sample appears below a certain threshold. As for the acquisition of this probability, it may be based on the use of generation models or on density. In any case, the overall idea of monitoring anomalies is very simple.</p>
<p>A lot of unusual detection methods need to be fine-tuned by a sample of complete information, and we should discuss when to choose the anomaly algorithm or the supervisory learning algorithm. When we have only a very small sample of labels, such as in the area of financial fraud detection, we have fewer cases of fraud and a relatively small number of unmarked samples, and we know that the number of fraud cases is rather small, the unusual detection may be a more appropriate method.</p>
<p>In fact, monitoring learning and abnormality detection uses two completely different approaches to understanding data, which model normal data and then reveal anomalies. The latter is the identification of irregular samples by means of supervision and learning, and when such samples are too few, it is difficult for him to learn all abnormal patterns, which can be avoided by abnormal testing. In other words, it is difficult to generalize what is not seen, which is limited by the algorithm itself.</p>
<p>Anomalous detection is more dependent on the choice of characteristics, requiring more rigorous judgement as to the value of each characteristic and whether its distribution meets the needs of the model, more than monitoring learning requirements. There is a need to construct more sophisticated features based on the experience of individuals in this field and to remove unnecessary ones.</p>
<h2>Semi-supervisory learning</h2>
<h3>Unmarked sample</h3>
<p>It is very common in the world to show that a large number of samples are not marked (lack of information on variables), that if traditional monitoring learning techniques are used directly, a significant amount of unmarked sample information is wasted, and that perhaps because the data set of marked samples is too small to produce good results in training.<strong>Semi-supervised learning: machine learning methods using unmarked samples</strong></p>
<p>How can unmarked samples be incorporated into models? The most natural way is to mark unmarked samples, of course, but the resources that are being consumed may be too large, so we try to find another way.</p>
<p>We can train a model with a marked sample, then we can use the model to find the most useful sample for the model's progress, and then we can mark it; we can use a smaller sample to achieve a better result, which is called "active learning."&quot;(active learning), the goal is to achieve the best possible performance by using as few "Query".</p>
<p>If no additional markers (expert knowledge) are introduced, can unmarked samples improve modelability? It's actually working.</p>
<p>In fact, unmarked samples do not directly contain tagging information, but if they are taken independently from the same data source and in the same distribution, they contain information on data distribution.
A model would be very useful.<strong>Using characteristic distribution information from unmarked samples to help us upgrade modelability.</strong></p>
<p>The use of unmarked samples necessitates some data distribution information from unmarked samples
Assumptions relating to category tags</p>
<p>The most common is the Cluster Assumptions, which assume that the data are clustered and that the same cluster of samples belong to the same category.
Another common hypothesis is "Flow-shaped." &quot;(manifold assumption), i.e., assuming data are distributed on a flow structure, the adjacent sample has similar output values.</p>
<p>The popular hypothesis has no limit on output values, and it is more widely applied, and it's the same scenario that we're using now.</p>
<p>Semi-supervisory learning can be further divided into pure (pure) semi-supervisory learning, which assumes that unmarked samples from training data are not to be predicted, while the latter is false
Unmarked samples considered in the course of the learning process are just to be predicted and the purpose of the learning is to obtain the best generalization on these unmarked samples Yes.</p>
<h3>Generating Method</h3>
<p>Generating methods (generative methods) are methods based directly on the generation model.
Assuming that all data (whether marked or not) are generated by the same potential model.</p>
<p>This assumption allows us to link unmarked data to learning objectives through parameters of a potential model, while tags of unmarked data can be considered missing parameters of the model, which can usually be based on EM algorithms to provide a very similar estimate of solvency.</p>
<p>The difference in such methods is mainly in the assumptions of the generation model, which will produce different methods.</p>
<p>given sample&#36;x&#36;, whose real category is marked&#36;y\in\mathcal{Y}&#36;, where&#36;\mathcal{Y}={1,2,\ldots,N}&#36;Assuming that samples are generated by a Gaussian hybrid model and that each category corresponds to a Gaussian mixture. In other words, data samples are generated on the basis of the following probability density:
&#36;&#36; (\bardsymbol{x} =sum i\cdot p} (\bardsymbol{mbol{\i,\bardsymbol{\symbol{\Sigma}<em>(i): &#36;
of which, blending factor &#36; \\alpha i\geqslant0,\sum</em>{i=1}^N\alpha_i=1;p(\boldsymbol{x}\mid\boldsymbol{\mu}_i,\boldsymbol{\Sigma}_i)&#36;是样本&#36;x&#36;属于第&#36;I'm a Gossian.
Probability of combinations;&#36;\boldsymbol{\mu}_i&#36;and&#36;\boldsymbol{\Sigma}_i&#36;It's a parameter for the Goss mix.</p>
<p>Using MLE to estimate EM solver, you can judge the type of information.</p>
<h3>Semi-supervised SVM</h3>
<p>Semi-Supervised Support Vector Machine (S3VM) supports the promotion of vectors in semi-supervised learning.</p>
<p>Without considering unmarked samples, the support vector tries to find the maximum spaced hyperplatform, while after considering unmarked samples, the S3VM tries to find an overplatform that separates two marked samples and passes through the low-density area of the data,</p>
<p>The basic assumption here is "low-density security," which is obviously the extension of the cluster scenario after consideration of linear super-levels.</p>
<p>As for how to solve it, it's not here.</p>
<h3>Figure 1. Semi-oversighted learning</h3>
<p>A data set is given, and we can map it as a map, and each sample is assembled to match a node in the map, and if the two samples are very similar (or relevant), there is a side between the corresponding node, with the "strength" of the side being more similar (or relevant) to the sample.
It's the idea of drawing machines learning. </p>
<p>We can imagine the nodes of the marked samples as dyed, while the unmarked samples of the nodes have not yet dyed. So, semi-supervisory learning corresponds to the process of spreading or spreading colours on the map.</p>
<h3>Disagreement-based approach</h3>
<p>Unlike the use of unmarked data in single-learning devices, such as the method of generation, semi-supervised SVM, and graphics-supervised learning, the use of multi-learning devices is essential for the use of unmarked data.</p>
<p>Co-training is an important representative of this approach, which was originally designed for multi-view data and is therefore also seen as a representative of multi-view learning. Before we introduce teamwork, let's see what multi-view data is.</p>
<p>In a number of practical applications, a data object tends to have multiple "attribute set " , and each property set constitutes a "view".</p>
<p>For example, for a film, it has several properties: the attribute set for image images, the attribute set for sound messages, the attribute set for subtitle messages, and even the attribute set for online advocacy discussions. Each property set can be seen as a view. </p>
<p>So, a film is a sample.&#36;(\langle\boldsymbol{x}^1,\boldsymbol{x}^2\rangle,y)&#36;, where&#36;x^i&#36;It's the sample in the view.&#36;i&#36;, i.e. attribute vector based on the description of the view properties, you may want to fake Freeze.&#36;x^{1}&#36;is the attribute vector in the image view,&#36;x^2&#36;is the attribute vector in the sound view;&#36;y&#36;It's a label. It's assumed to be the type of movie, like "action film," "love film."&#36;(\langle x^1,x^2\rangle,y)&#36;This kind of data is multi-view data.</p>
<p>Assume that different views have "compatibility" (compatibility), i.e. that they contain about output space&#36;\gamma&#36;The message is identical:&#36;\gamma^{1}&#36;This means that the marking space is distinguished from image information.&#36;\gamma^{2}&#36;indicates that the tag space is distinguished from sound information, and there is&#36;\mathcal{Y}=\mathcal{Y}^1=\mathcal{Y}^2&#36;It is clear that, on the basis of compatibility, the “complementarity” of different views of information would make it easier to build a learning device.</p>
<p>The synergetic training makes good use of the multi-view "compatible complementarities", assuming that the data have two full (sufficient) and independent viewing of conditions, meaning that each view contains information sufficient to produce the best learners, and "conditional independence" means that two views are independent under the given category tag. In this case, a simple method can be used to use unmarked data:</p>
<p>First, a taxonomy is trained on each view based on a marked sample, then each classifier is given a false mark to select its own “most sure” unmarked sample, and a pseudomark sample is provided to another classifier as an additional marked sample for training updating ... This process of “learning from each other and making progress together” continues in succession until both categories do not change or reach the predefined number of iterative wheels.</p>
<p>While the collaborative training process is simple, it is surprising that theoretical evidence shows that unmarked samples can be used to raise the generalization performance of the weak taxonomy to an arbitrary high if the two views are adequate and independent. However, the independence of the viewing is often difficult to satisfy in a realistic mission, and therefore the performance increase is not so large, but research suggests that, even under weaker conditions, teamwork can effectively enhance the performance of the weak taxonomy. </p>
<p>Co-training algorithms are themselves designed for multi-view data, but since then a number of variable algorithms have emerged that can be used in single-view data, either using different learning algorithms or using different data sampling, or even using different parameter settings to produce different learning devices that can effectively use unmarked data to enhance performance. Subsequent theoretical studies have found that such algorithms in fact require, inter alia, multiple viewing of data, and only significant differences (or differences) between learners, which can be enhanced by providing each other with false tag samples. <strong>It's not important to have multiple viewing designs.</strong></p>
