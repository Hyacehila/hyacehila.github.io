---
title: 'Machine Learning Introduction: Supervised Learning and Bayesian Methods'
title_zh: 机器学习导论：监督学习与贝叶斯方法
date: 2024-03-28 18:22:50 +0800
categories:
- Machine Learning
- Classical Machine Learning
tags:
- Machine Learning
- Supervised Learning
author: Hyacehila
mathjax: true
hidden: true
excerpt: Covers core concepts, decision trees, SVMs, kernel methods, Bayesian classifiers, and Bayesian networks.
description: Covers core concepts, decision trees, SVMs, kernel methods, Bayesian classifiers, and Bayesian networks.
excerpt_zh: 整理机器学习基本概念、决策树、支持向量机、核方法、贝叶斯分类器和贝叶斯网。
permalink: /blog/2024/03/28/machine-learning-introduction-supervised-learning/
lang: en
translation_key: 2024-03-28-machine-learning-introduction-supervised-learning
translation_status: machine
translation_source_hash: aa135314b46a319a289fa63206f6ddbd528482e67110a0a7e76b8c0fdeca081a
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<h2>Overview of Machine Learning</h2>
<h3>What is machine learning?</h3>
<p>In traditional research, we mainly use model recognition (the regular expression is a model recognition) as a way of dealing with problems as a machine type; in fact, model recognition has a variety of problems, and it is not ideal to rely on subjective human extraction features and then leave them to the machine to judge their correctness and to do nothing about more complex issues;</p>
<p>That's why we thought of it spontaneously.<strong>Let the machine improve itself.</strong>This is precisely the subject of machine learning, which is dedicated to studying how to improve the system ' s own performance through computing techniques and experience;</p>
<p>At the stage of development in this area, symbolic learning, represented by rule learning, is the first machine learning technology to be studied, but because of performance problems, the rule learning technology has been largely replaced by, or developed as a form of integration.</p>
<h3>Basic terminology</h3>
<p>Thus, the main element studied by the Machine Learning Institute is the algorithm that produces "mode" from data on computers, i.e. "Learning algorithm".</p>
<p>With learning algorithms, we can provide it with empirical data that will make models based on them; in the face of new circumstances, models will give us the judgement that, if computer science is a study of algorithms, similarly, machine learning is a study of algorithms.</p>
<p>The process of learning models from data is referred to as “learning” or “training” and is accomplished by implementing a learning algorithm. The data used in training is called “training data”, each of which is called a “training sample” and a collection of training samples is called “training set”.</p>
<p>After learning the model, the process of predicting it is called testing.</p>
<p>Depending on whether training data contain tagging information, learning assignments can be broadly classified into two broad categories: “supervised learning” and “unsupervised learning”, with classification and regression representing the former, while clustering and regression represent the latter.</p>
<p>It needs to be noted that the goal of machine learning is to make the model learned well applicable to the "new sample" rather than simply working well on the training sample; that is, generalization.</p>
<h3>Development experience</h3>
<p>The reasoning phase: the mode recognition phase, the artificial characterisation, the machine's logical reasoning.</p>
<p>Summarizing the learning phase: the most important way to enable machines to learn themselves, and we want machines to have their own learning skills, to summarize commonality from samples and then to reason.</p>
<p><strong>The entire field of machine learning is now in the general learning phase, followed by its own development.</strong></p>
<p>The symbol learning phase. The decision tree is an important representation.</p>
<p>The main difference between connectionism and symbol learning is whether or not it's a black box.</p>
<p>A statistical learning phase, supporting vector algorithms is an important representation.</p>
<p>The depth learning phase, with the growth of computing, the connectionism has re-emerged into multi-layer neural learning as a new attraction in machine learning, continuing the low interpretability of connectivityism, requiring a lot of training data and very high calculus, and we don't dwell much on in-depth learning in machine learning.</p>
<h3>Rear</h3>
<p>In the current process of machine learning, we have two more important stages of research transformation:</p>
<ul>
<li>From symbolism to statistical learning</li>
<li>The LLM phase from statistical learning to purely industry-led learning</li>
</ul>
<p>Only for the first time was a major theoretical breakthrough; statistical methods were introduced in the field of machine learning;</p>
<p>Continued integration of ideas in mathematics will be at the heart of the next theoretical breakthrough in machine learning.</p>
<h3>Classification of machine learning issues</h3>
<p>Machine learning is essentially a question of making the machine look for our decision-making functions, and depending on the decision-making functions, we can classify machine learning issues.</p>
<p>Assuming that the output of the function to be found is a value, a metric, the task of this machine learning is called<strong>Return</strong>。</p>
<p>Besides returning, another common task.<strong>Classification</strong>I don't know. The sorting task is for machines to choose. Humans first have some options that are referred to as classes, and the output of the function now sought is the selection of one as an output from the set option, the task being referred to as a classification.</p>
<p>In the field of machine learning, besides regression and classification,<strong>Structured learning</strong>(structured learning) The machine is not just going to make a choice or output a number, but rather to produce a structured object, such as a drawing and an article. This problem of machines producing structural things is called structured learning.</p>
<p>Of course. Unsupervised.<strong>Cluster & Declination</strong>It is also an important machine learning technique, which is regularly and supervised for collaborative use.</p>
<p>For model assessment and selection, see<a href="/en/blog/2025/10/02/supervised-learning-model-evaluation/">Monitoring of learning performance assessment</a>。</p>
<h2>Decision Tree</h2>
<h3>Basic definitions</h3>
<p>The decision tree is a common method of machine learning. By definition, the decision tree is based on tree structures, which is a natural mechanism for humans to deal with when they are faced with decision-making issues.</p>
<p>In general, a decision tree contains a root node, several internal nodes and several leaf nodes; leaves correspond to the decision result, while each other corresponds to a attribute test; the sample collection contained in each node is divided into subnodes based on the results of the attribute test; root node contains the whole sample set. The path from the root node to each leaf node corresponds to a decision test series;</p>
<p>The decision tree learning aims to produce a broad-based decision tree;</p>
<p>The decision tree is a sort of a regression process; we judge the sample that is currently available, if it's empty or all in a category, then it should be a leaf node; if not, we need to select the best attribute and then divide it and then create a new node.</p>
<h3>Split Selection</h3>
<p>It is natural that the selection of the most important attribute in the entire decision tree learning process is expected to be as high as possible, as the division continues, and the sample contained in the decision tree branch nodes, the purity of the node, is as high as possible.</p>
<h4>Information Gains</h4>
<p>"Info entropy &quot;(information entropy) is the most commonly used indicator for measuring the purity of the sample collection.</p>
<p>Assuming current sample collection&#36;D&#36;Medium&#36;k&#36; The percentage of samples in the category is&#36;p_k&#36;  So you can define his entropy as
&#36;&#36;\operatorname{Ent}(D)=-\sum\limits_{k=1}^{|\mathcal{Y}|}p_k\log_2p_k.&#36;&#36;
The smaller the message entropy, the higher the purity. </p>
<p>Suppose we have a attribute for division.&#36;a&#36; He's on the same page.&#36;V&#36;A possible value, then, to divide it, you can get it.&#36;V&#36;We can calculate the information entropy of each branch, and then we can give weight to the different samples of each branch, and then we can calculate the information gain from this attribute division on the sample.
&#36;&#36;\mathrm{Gain}(D,a)=\mathrm{Ent}(D)-\sum_{v=1}^{V}\frac{|D^{v}|}{|D|}\mathrm{Ent}(D^{v}).&#36;&#36;
The greater the information gain, the better the purity increase, we can use the information gain to select the attribute for the criterion.</p>
<h4>Gain rate</h4>
<p>The information gain code is not without its disadvantages, and if we separate each sample into a separate category, then the information gain is the greatest; in fact, the information gain guideline is biased towards properties that may have a higher value, so that we can reduce the negative impact of this preference, we introduce the gain rate.
&#36;&#36;\operatorname{Gain}\text{ratio}(D,a)=\frac{\operatorname{Gain}(D,a)}{\operatorname{IV}(a)}&#36;&#36;
of which
&#36;&#36;\mathrm{IV}(a)=-\sum_{v=1}^{V}\frac{|D^{v}|}{|D|}\log_{2}\frac{|D^{v}|}{|D|}\quad.&#36;&#36;
The gain rate can suppress the preference for information gain, which unfortunately has a preference for a smaller number of classifications, and we better take the two ways ahead of weight.</p>
<h4>Gini index</h4>
<p>We introduced another way to measure the purity of the data.
&#36;&#36;00\
& Dopectorname{Gini}&amp; =\sum_{k=1}^{|J|}\sum_{k^{\prime}\neq k}p_{k}p_{k^{\prime&#125;&#125;  \
&amp;=1-\sum_{k=1}^{|\mathcal{Y}|}p_{k}^{2}.
\end{aligned}&#36;&#36;
因此 属性a划分带来的纯度提升可以用基尼指数来衡量
&#36;&#36;\mathrm{Gini}(D,a)=sum=v=
The specific method remains the same.</p>
<h3>Cut.</h3>
<p>Pruning is the main means of learning algorithms for decision-making trees against "over-sizing."
In practice, in order to classify training samples as accurately as possible, nodes will be drawn up repeatedly, sometimes resulting in too many branches of decision-making, which may lead to a convergence of training samples as a result of “too good” learning to treat some of their own features as general in nature for all data. Therefore, the risk of a possible merger can be reduced by actively removing some branches.</p>
<p>The basic strategies for deciding on the cut are pre-cuting and "backcuting." Twig &quot; (post-
pruning)</p>
<p>Precuts are the estimation of each node before dividing it during the decision tree generation process, and if the current node is not divided to bring about an increase in the decision tree ' s pancreasity, stop dividing and mark the current node as a leaf node;</p>
<p>The latter cut is to generate a full decision tree from the training set, then to examine the non-leaf nodes from the bottom up, and if replacing the subtree with the node would lead to increased decision tree panification, the subtree would be replaced with the node.</p>
<p>As for how to judge the improvement of the generalized performance, we could consider the performance assessment component of this paper.</p>
<h3>Continuous and Missing Values</h3>
<h4>Continuous value processing</h4>
<p>Since the number of desirable values for continuous properties is no longer limited, no nodes can be divided directly on the basis of the value of the continuous properties ... When the dagger is stopped, the continuous attribute discrete technology is useful;</p>
<p>All we have to do is set the continuous properties a certain step and set the range of small areas as a classification of the continuous properties.</p>
<p>Unlike the discrete properties, if the current node is divided into continuous properties, it can also be the attribute of the lateral node.</p>
<p><strong>In most decision tree algorithms, continuous attributes automatically generate only two branches in each decision-making, selecting the best points according to an indicator, rather than using the division step method we have described.</strong></p>
<h4>Missing value processing</h4>
<p>In reality, there are often incomplete samples, some of which are missing; sometimes we can discard missing samples, but sometimes this leads to too few remaining samples.</p>
<p>We need to address two issues:</p>
<ul>
<li>How can you select attributes in the absence of attribute values?</li>
<li>How can a given division attribute be divided if the value of the sample on that attribute is missing?</li>
</ul>
<p>For the first question, we're training the model using only samples that are not missing on this attribute, and we need to amend the ratio in the calculation of the entropy to ensure that the ratio is equal to one.</p>
<p>For the second question, we'll have to x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-</p>
<h3>Multivariate Decision Tree</h3>
<p>When the true classification boundaries of the learning mission are complex, the decision tree algorithms described above must be used in many differentiating to achieve a better approximation; the decision tree at this time will be complex and, because of the extensive attribute tests, the projected time costs will be substantial</p>
<p>The idea of the multivariant decision tree is that the non-leaf node is no longer just a certain attribute, but rather a linear combination of the properties; that is, each non-leaf node is a linear sorter (the design of which we will learn in the classification algorithm) and each non-leaf node is determined at the same time as the appropriate linear node; this method is effective in reducing the complexity of the decision tree.</p>
<p>The idea of a multivariant decision tree is actually a great idea for improvement, embedding different algorithms in the decision tree, combining their strengths, and achieving classification.</p>
<h2>Support vector SVM</h2>
<h3>Interval with Support Vector</h3>
<p>Supporting vectors is another way of monitoring learning; it's about dealing with classification; his thinking is very simple, based on training data. Set&#36;D&#36; A hyper plane was found in the sample space to separate the different types of samples;
So there are two questions about how to find the super plane and how to separate the sample, and how to find the most suitable one in the multiple super plane that we're looking for, and give him the best generalization.</p>
<p>In the sample space, the split of the hyperlevel can be described by the linear equation below.
&#36;&#36;w^\mathrm{T}x+b=0&#36;&#36;
The entire split super-platform is being trans-variated&#36;w&#36;Scroll Off&#36;b&#36;Decision</p>
<p>So, any distance from the sample space to the hyper plane can be written as
&#36;&#36;r=\frac{|\boldsymbol{w^\mathrm{T&#125;&#125;\boldsymbol{x}+b|}{||\boldsymbol{w}||}.&#36;&#36;
If the super plane can classify the training samples correctly, assuming we separate them.&#36;y_i&#36;Take positive and negative 1 and there is
&#36;&#36;\left.\left{bärray}l^mathrm{x i+b\geqslant+1,&amp;y_i=+1;\w^\mathrm{T}x_i+b\leqslant-1,&amp;y_i=-1.\end{array}\right.\right.&#36;&#36;
我们知道 距离超平面最近的这几个训练样本一定可以让前式中的等号成立 他们被称为支持向量(support vector) 两个异类支持向量到超平面的距离为
&#36;&#36;{\cHFFFFFF}{\cH00FFFF} That's a good idea.
We call it the interval. </p>
<p>To find the maximum interval between the super-levels, the most optimal problem is the one that needs to be optimized below.
&#36;&#36;00begin{aligned}\min {\boldsymbol{w},b}&amp;\frac{1}{2}|\boldsymbol{w}|^2\\mathrm{s.t.}&amp;^mathrm^bardsymbol{i+gqslant1,i=1,\ldots,\boldsymbol{m}. ^end{aligned} I'm sorry.
This is the basic type of support vector.</p>
<p><em>The binding condition means, category&#36;y_i&#36;The volume of the flat-sided calculation is greater than one, meaning that all samples are classified correctly.</em></p>
<h3>Nuclear Functions and Methods</h3>
<p>In the discussion that followed, we assumed that the training sample was linear, that there was a super-level division that would correctly classify the training sample. But in reality, there might not be a super-level in the original sample space that would correctly divide the two types of sample, that is, the original sample was spatially linear.</p>
<p>If linear, it means that a simple linear classification can be used, such as logit return and svm</p>
<p>If linear, either complex non-linear classification or a kernel method to map to high-dimensional space Go, go, go!</p>
<p>For such a problem, the sample can be mapped from the original space to a higher dimension of the characteristic space, which allows the sample to be divided in the inner dimension of the characteristic space; fortunately, if the original space is limited in dimension, i.e., the number of properties is limited, then there must be a high dimension feature space to allow the sample to be divided.</p>
<p>And when we're in high space, our best question changes:
&#36;&#36;00begin{aligned}\min {\boldsymbol{w},b}&amp;\frac{1}{2}|\boldsymbol{w}|^2\\text{s.t.}&amp;^mathrm^ (\bardsymbol{)\geqslant1,\quad i=1,2,\ldots,m. ^end{aligned} I'm sorry.
of which&#36;\phi(x)&#36; It's a characteristic vector after mapping.</p>
<p>The solver at this time involves calculating &#36;\phi (\bardsymbol{x}<em>{i})^{\mathrm{T&#125;&#125;\phi(\boldsymbol{x}</em>It's usually difficult to calculate directly because the dimension of the characteristic space may be high or even infinity. To avoid this barrier, it is conceivable.
&#36;&#36;500 (\bardsymbol{x}i,\bardsymbol{x}=langle\phi(\bardsymbol{x}<em>i),\phi(\boldsymbol{x}<em>j)\rangle=\phi(\boldsymbol{x}<em>i)^\mathrm{T}\phi(\boldsymbol{x}<em>j)&#36;&#36;
也就是特征空间的内积等于它们在原始样本空间中通过函数&#36;\kappa&#36; 计算的结果 这样我们就可以规避前面的内积计算 原始的最优化问题转变为
&#36;&#36;\begin{aligned}\max</em>{\boldsymbol{\alpha&#125;&#125;&amp;\sum</em>{i=1}^m\alpha_i-\frac{1}{2}\sum</em>{i=1}^m\sum</em>{j=1}^m\alpha_i\alpha_jy_iy_j\kappa(\boldsymbol{x}_i,\boldsymbol{x}<em>j)\\text{s.t.}&amp;\sum</em>{i=1}^m\alpha_iy_i=0,\&amp;\alpha_i\geqslant0,\quad i=1,2,\ldots,m.\end{aligned}&#36;&#36;</p>
<p>If we know the form of non-linear mapping, we can give the form of a nuclear function.<strong>We need to choose our own nuclear function, which is the biggest variable of the SVM algorithm.</strong> If the nuclear function is not selected properly, it means that the sample is mapped to an inappropriate feature space, which may result in poor performance</p>
<p>We give theoretics: if a nuclear matrix that corresponds to a symmetric function is semi-corrected, It can be used as a nuclear function.
&#36;&#36;\matbf{ \begin{bmatrix}\kappa (\bardsymbol{1,\bardsymbol{x} 1)&amp;\cdots&amp;\kappa(\boldsymbol{x}_1,\boldsymbol{x}_j)&amp;\cdots&amp;\kappa(\boldsymbol{x}_1,\boldsymbol{x}_m)\\vdots&amp;\ddots&amp;\vdots&amp;\ddots&amp;\vdots\\kappa(\boldsymbol{x}_i,\boldsymbol{x}_1)&amp;\cdots&amp;\kappa(\boldsymbol{x}_i,\boldsymbol{x}_j)&amp;\cdots&amp;\kappa(\boldsymbol{x}_i,\boldsymbol{x}_m)\\vdots&amp;\ddots&amp;\vdots&amp;\ddots&amp;\vdots\\kappa(\boldsymbol{x}_m,\boldsymbol{x}_1)&amp;\cdots&amp;\kappa(\boldsymbol{x}_m,\boldsymbol{x}_j)&amp;\cdots&amp;\kappa(\boldsymbol{x}_m,\boldsymbol{x}_m)\end{bmatrix}.&#36;&#36;</p>
<p>Several commonly used nuclear functions</p>
<table>
<thead>
<tr>
<th>Name</th>
<th>Expression</th>
</tr>
</thead>
<tbody><tr>
<td>Linear core</td>
<td>&#36;\kappa(\boldsymbol{x}_i,\boldsymbol{x}_j)=\boldsymbol{x}_i^\mathrm{T}\boldsymbol{x}_j&#36;</td>
</tr>
<tr>
<td>Multiple nuclear</td>
<td>&#36;\kappa(\boldsymbol{x_{i&#125;&#125;,\boldsymbol{x_{j&#125;&#125;)=(\boldsymbol{x_{i}^{\mathrm{T&#125;&#125;x_{j&#125;&#125;)^{d}&#36;</td>
</tr>
<tr>
<td>Goss core.</td>
<td>&#36;\kappa(\boldsymbol{x}<em>{i},\boldsymbol{x}</em>{j})=\exp\big(-\frac{|\boldsymbol{x}<em>{i}-\boldsymbol{x}</em>{j}|^{2&#125;&#125;{2\sigma^{2&#125;&#125;\big)&#36;</td>
</tr>
<tr>
<td>La Plass nuclear</td>
<td>&#36;\kappa(\boldsymbol{x}_i,\boldsymbol{x}_j)=\exp\left(-\frac{|\boldsymbol{x}_i-\boldsymbol{x}_j|}{\sigma}\right)&#36;</td>
</tr>
<tr>
<td>Sigmoid Nuclear</td>
<td>&#36;\kappa(\boldsymbol{x}<em>{i},\boldsymbol{x}</em>{j})=\tanh(\beta\boldsymbol{x}<em>{i}^{\mathrm{T&#125;&#125;\boldsymbol{x}</em>{j}+\theta)&#36;</td>
</tr>
</tbody></table>
<p>Some of the original nuclear functions are also combined as nuclear functions, as in the case of
&#36;&#36;\gamma_{1}\kappa_{1}+\gamma_{2}\kappa_{2}&#36;&#36;
&#36;&#36;\kappa_1\otimes\kappa_2(x,z)=\kappa_1(x,z)\kappa_2(x,z)&#36;&#36;
&#36;&#36;\kappa(x,z)=g(x)\kappa_1(x,z)g(z)&#36;&#36;</p>
<h3>Soft interval and regularization</h3>
<p>In the preceding discussion, we have always assumed that training samples are linear in the sample space or in the characteristic space, i.e., that there is a super plane that can fully divide the different types of samples;</p>
<p>It is often difficult to determine the appropriate nuclear function in a realistic mission to distinguish the training sample from linear in the characteristic space; to back off, even if it happens to find a nuclear function that allows training to be divided in the characteristic space, it is difficult to conclude that this apparent linearity is not the result of over-coding. </p>
<p>One way to alleviate this problem is to allow the support vector to come out of some samples, that is, introduce soft spacing.</p>
<p>The AVRs described earlier require that all samples meet our constraints, which is called "hard margin" and soft space allows some samples to be unbound.
&#36;&#36;y_{i}(\boldsymbol{w^{\mathrm{T&#125;&#125;x_{i&#125;&#125;+b)\geqslant1.&#36;&#36;
And of course, we have a sample of the least-fulfilment constraint that introduces the loss to make the best of the problem.
&#36;US&#36; \min (boldsymbol{w}, b}\bardsymbol{w}c\sum x}<em>{i}+b\right)-1\right)&#36;&#36;
其中损失函数的选取比较自由 我们在这里只进行简单的介绍
最基础的01损失为
&#36;&#36;\ell</em>{0/1}(z)=\begin{cases}1,&amp;\text{if}z&lt;0;\0,&amp;\text{otherwise}. \end{cases}
Because loss 1 is not easy to optimize, we have an alternative to this.</p>
<ul>
<li>hinge Loss&#36;:\ell_{hinge}(z)=\max(0,1-z):;&#36; </li>
<li>Index loss: &#36;\ell_{exp}(z)=\exp(-\dot{z}):;&#36;</li>
<li>Logical loss&#36;) {: }\ell_{log}( z) = \log ( 1+ \exp ( - z) ) .&#36;</li>
</ul>
<p>In essence, we can get other learning models by replacing the best target; just as we introduce regularity in online retrogression, introduce other penalties at minimal intervals in the base, get our goal.</p>
<h3>Support vector regression</h3>
<p>Now we're thinking about the return problem, and the variables that we're training have become a real number, and we want to learn a regression model that will allow&#36;f(x)&#36;and&#36;y&#36;As close as possible.</p>
<p>The traditional regression model is usually based directly on the difference between model output and real output, and only when their difference is zero, unlike the one that supports vector return, assuming that we can tolerate one between them.&#36;\epsilon&#36; Only when the deviation is greater than this one do we calculate the loss.</p>
<p>So, SVR can be written in
&#36;US&#36;US&#36; \min (boldsymbol{w}, b}\boldsymbol{w} +c\sum i=m}\mellll {\epsilon}\left(\boldsymbol{x}<em>{i})-y</em>{i}\right),&#36;&#36;
其中&#36;C&#36;是正则化常数 损失函数的形式应该为
&#36;&#36;\left.\ell_\epsilon(z)=\left{\begin{array}{ll}0,&amp;\text{if}|z|\leqslant\epsilon;\|z|-\epsilon,&amp;\text{otherwise}.\end{array}\right.\right.&#36;&#36;</p>
<p>We can see it as a linear regression, allowing deviations and punishing coefficient complexity.</p>
<h3>Nuclear methods</h3>
<h4>Introduction to nuclear methods</h4>
<p>We've studied SVM and SVR, and the models they've learned in the end are all linear combinations of nuclear functions, and it's actually a general conclusion.</p>
<p>(expressing the theorem)&#36;H&#36; is a nuclear function &#36;\kappa&#36; The corresponding regenerative pelvic, the Zero, the Zero.<em>{\mathrm{H&#125;&#125;&#36; 表示 &#36;H&#36; 空间中关于 &#36;h&#36; 的范数，对于任意单调递增函数 &#36;\Omega:[0,\infty]\mapsto\mathbb{R}&#36; 和任意非负损失函数 &#36;\ell: \\mathbb{R}m\mapsto[0, \infty], optimisation problem
&#36;min\lits</em>{h\in\mathbb{H&#125;&#125;F(h)=\Omega(|h|_{\mathbb{H&#125;&#125;)+\ell\big(h(\boldsymbol{x}_1),h(\boldsymbol{x}_2),\ldots,h(\boldsymbol{x}<em>m)\big)&#36;&#36;
的解总可以写作
&#36;&#36;h^*(\boldsymbol{x})=\sum</em>^m\alpha i\kappa(\bardsymbol{,\bardsymbol{x} i). &#36;
So there's no limit to the loss function, and there's a requirement for a single increment to the regularization item. </p>
<p>So we've developed a lot of learning methods based on nuclear functions, collectively called "kernel methods."</p>
<p><strong>In a rough way, in any algorithm with a point operation, replacing a point with a nuclear function can be called a nuclear method, not just for SVM.</strong></p>
<p><strong>Nuclear methods allow us to enjoy the benefits of high-dimensional space, but at the same time do not need to suffer the disadvantages, the greatest of which is that low-dimensional non-linear issues are very easy to solve in high-dimensional ways, i.e., delinearizing models.</strong></p>
<h4>Nuclear Linear Analysis KLDA</h4>
<p>We'll just introduce the "nuclear linear analysis."&quot;It's a...<strong>Introduce nuclear functions to expand linear learning devices to non-linear learning Device</strong> The following is the text:</p>
<p>We assume that some kind of mapping is going to reach a characteristic space.&#36;F&#36;Yes.&#36;F&#36;. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
&#36;&#36;h(\boldsymbol{x})=\boldsymbol{w}^{\mathrm{T&#125;&#125;\phi(\boldsymbol{x}).&#36;&#36;
It's easy to give us a goal for learning.
{\fnH00FFFF}&#36;US&#36;US&#36;US&#36; mx {\boldsymbol{w}=holdsymbol=<em>{b}^{\phi}\boldsymbol{w&#125;&#125;{\boldsymbol{w}^{\mathrm{T&#125;&#125;\mathbf{S}</em>\bardsymbol{w, &#36;
of which&#36;S&#36;It's a dispersive matrix. It's just a map.&#36;F&#36;Up the sprawl matrix</p>
<p>Select a nuclear function&#36;\Omega = 0&#36; By the expression of theorem,
&#36;h (\bardsymbol{x})<em>{\cdot}=\sum</em>\kappa (\bardsymbol{,\bardsymbol{xi}), &#36;&#36;
So, we can calculate.&#36;w&#36; and &#36;\alpha&#36; And finally, we're given it.&#36;h(x)&#36; </p>
<h4>Nuclear methods are widely used</h4>
<p>The basic idea of the nuclear approach is to map the high-dimensional space only and increase the non-linear division capability of the subsequent algorithm. So there are a lot of algorithms that can add nuclear components, like,</p>
<ul>
<li>NUCLEARLY PCA, NON-LINELINE DEVICE</li>
<li>denuclearization of LDA, non-linear classification</li>
<li>denuclearization SVM, classification</li>
<li>nucleination K means grouping</li>
</ul>
<h2>Bayesian Catalogue</h2>
<h3>The Bayesian Decision theory and our goals</h3>
<p>Bayesian decision theory is the basic method for implementing decision-making within a probabilistic framework. For classification tasks, in ideal cases where all the relevant probabilities are known, the Bayesian decision theory considers how to select the best category labels based on these probabilities and miscalculation losses.</p>
<p>The expectation of the loss function for a later distribution is called the later risk function, and the decision to minimize the later risk is the option we should choose.</p>
<p>The loss function is better given, and we can select the appropriate loss function depending on our goals, but the problem of the later probability is a more troublesome one, and we're calculating it using theory in Bayes, but it's not realistic in machine learning, so... <strong>What machines learn is to estimate the probability of a posteriori with the greatest possible accuracy based on a limited sample collection of training.</strong></p>
<p>In general, there are two main strategies: </p>
<ul>
<li>Discriminative model&#36;x&#36;Give it directly.&#36;P(c|x)&#36; </li>
<li>Generating models calculate the a priori and sample distributions according to the Bayesian theorem
The decision tree, the BP network, the support vector, etc., which we've described earlier, can be classified as a type model, while the Beyers cataloguer that we've described later is a form model.</li>
</ul>
<h3>PARK Soo Bayesian Catalogue</h3>
<p>First, we're limiting this Bayesian decision to the classification of the problem, which is a multi-variant, a variable that is a variable that is a variable that is a variable that is a variable that is a variable that is a variable that is a variable that is a variable that is a variable that is a variable that is a variable that is a variable that is a variable that is a variable that is a variable that is a variable that is a variable that is a variable that is a variable that is a variable that is a variable that is a variable that is a variable that is a variable that is a variable that is a variable that is a variable that is a variable that is a variable that is a variable that is a variable that is a variable that is a variable that is a variable that is a variable that is a variable that is a variable that is a variable that is a variable that is a variable that is a variable that is a variable that is a variable that is a variable that is a variable that is a variable that is a variable that is a variable that is a variable that is a variable that is a variable that is a variable that is a variable that is a given a variable that is a variable that is a given a variable</p>
<p>Select the loss function as
&#36;&#36;\left.\lambda=left{\&amp;\text{if}i=j;\1,&amp;\text{otherwise},\end{array}\right.\right.&#36;&#36;
此时我们可以给出分类器为
&#36;&#36;*(\bardsymbol{x} *arg\max c\matcal{Y}P(c\mid\bardsymbol{x}), &#36;
That's every sample.&#36;x&#36; Select the most probabilities mark for the post-check</p>
<p>The biggest problem we're dealing with the Bayesian taxonomy is that there are many variables, and they have different distributions, which make it difficult to solve the post-probability problem in order to avoid such obstacles. <strong>The "Instructive Condition Independence " assumption is used by the PARK Bayes classifier &quot; (attribute conditional independence assumption):</strong> For known categories, assuming that all attributes are independent of each other ... in other words, assuming that each attribute independently influences the classification results</p>
<p>So we can fix the probability of a posteriori.
&#36;&#36;P(c\mid\boldsymbol{x})=\frac{P(c)P(\boldsymbol{x}\mid\boldsymbol{c})}{P(\boldsymbol{x})}=\frac{P(\boldsymbol{c})}{P(\boldsymbol{x})}\prod_{i=1}^dP(x_i\mid c)&#36;&#36;
The accumulation behind this is the probability of different variables.
&#36;&#36;h_{nb}(\boldsymbol{x})=\arg\max_{c\in\mathcal{Y&#125;&#125;P(c)\prod_{i=1}^{d}P(x_{i}\mid c)&#36;&#36;</p>
<p>Now, we can estimate probabilities for the training data set.
&#36;&#36;P(c)=\frac{|D_c|}{|D|}&#36;&#36;
For discrete properties, the probability of a condition is estimated.
&#36;&#36;P(x_i\mid c)=\frac{|D_{c,x_i}|}{|D_c|}&#36;&#36;
For continuous properties, the probability density function is assumed to match the normal distribution, using MLE estimates of the mean and the difference in the class, and then there is
&#36;&#36;p(x_{i}\mid c)=\frac{1}{\sqrt{2\pi}\sigma_{c,i&#125;&#125;\exp\left(-\frac{(x_{i}-\mu_{c,i})^{2&#125;&#125;{2\sigma_{c,i}^{2&#125;&#125;\right)&#36;&#36;
With the above foundation, you can use the PARK Soo-Bayers taxonomyr to estimate the classification of new samples.
The training process is the one that just happened to be calculated for the new samples; the probability is calculated separately based on the criteria given above, and see what conditions are being significantly enhanced, and give a classification conclusion.</p>
<h3>semi-Puerhaps Cataloguer</h3>
<p>The PARK Soo Bayes taxonomy uses the assumption of the independence of the attribute, but in reality it is often difficult to establish this assumption in the mission. So, people tried to relax a certain degree of the attribute independence assumption, and this led to a learning method called "semi-naive Bayes crasciferers".</p>
<p>The basic idea of the semi-Puerius Beyers classification is to give due consideration to the interdependence of some properties.
The most common idea is to assume that each attribute depends on only one other attribute outside the category, i.e., one-dependent Estimator.
&#36;&#36;P(c\mid\boldsymbol{x})\propto P(c)\prod_{i=1}^{d}P(x_{i}\mid c,pa_{i}),&#36;&#36;
of which attribute&#36;pa_i&#36;Called&#36;x_i&#36;Father's property in this case, we can be right&#36;\prod_{i=1}^{d}P(x_{i}\mid c,pa_{i})&#36; Conducting estimates</p>
<p>The heart of the semi-supplex Beyce taxonomy is how to set up paternity. Device</p>
<p>The most immediate approach is to assume that all properties depend on the same attribute, called "super-father", and then determine the super-paternity by means of model selection methods such as cross-validation, called SPODE (Super-Parent ODE)</p>
<p>There are, of course, many other ways to determine paternity, like the TAN, which has the effect of retaining dependency on strong and relevant properties, and the AODE, which is a hyper-father approach based on integrated learning, considering all attributes as hyper-fathers, as shown in the following figure.
<img src="/assets/images/machine-learning-notes/ml-naive-bayes-dependencies.png" alt="Half-Puerius relies on structural indications."></p>
<p>Very naturally, could the relaxation of the presumption of the independence of attribution be continued in exchange for a broader performance? Here we go back to our original problem, where the introduction of the attribute independent hypothesis is intended to address the problem of inadequate training samples due to the probability of a combination of higher levels, and the continued relaxation of this problem will fall back into the same situation again.</p>
<h3>Bayesian Network</h3>
<h4>The introduction of the Bayesian web</h4>
<p>Bayesian network, also known as faith network, is a tool for the development of the Internet.<strong>Directed Acyclic Graph, short DAG</strong> To define the dependency relationship between properties and to describe the joint probability distribution of attributes using the Conditional Probability Table, CPT (our example will be described as a variable-by-variant, which is virtually unlimited)</p>
<p>The Bayesnet is a classic probability map model.<a href="/en/blog/2024/04/06/advanced-machine-learning-unsupervised-learning/">Machine learning progression and unsupervised learning: Probability mapping model</a></p>
<p>Specifically, a Beyers web.&#36;B&#36; By Structure&#36;G&#36; and parameters&#36;\Theta&#36; Two parts. Network structure&#36;G&#36; A chart with a no-ring map, each of which corresponds to one attribute, which is linked by one side if the two attributes are directly dependent. Parameters&#36;\Theta&#36; Quantitatively describes this dependency if the attribute&#36;x_i&#36;The parent point is&#36;\pi_i&#36; Parameters&#36;\Theta&#36;It's all there is.&#36;P(x_i|\pi_i)&#36;  </p>
<p>One simple example is
<img src="/assets/images/machine-learning-notes/ml-bayesian-network-example.png" alt="Bates Network."></p>
<h4>Structure of the Bayesian network</h4>
<p>The Bayesnet has modified the ODE idea that we introduced in Park Soo Bayes, but it can be described as a broad-based PARK Soo Bayes. His structure is an effective expression of the conditions of independence between the attributes.<strong>The parent node set, the Bayesnet assumes that each property is independent of its non-descendants.</strong> Here we have to understand this sentence with <strong>There's a loopless map.</strong> Contact </p>
<p>So that's the definition of a joint probability distribution to
&#36;&#36;P_B(x_1,x_2,\dots,x_d)=\prod\limits_{i=1}^dP_B(x_i\mid\pi_i)=\prod\limits_{i=1}^d\theta_{x_i|\pi_i}.&#36;&#36;
For the example of the habit we've been using, his combined probability is that
&#36;&#36;P(x_1,x_2,x_3,x_4,x_5)=P(x_1)P(x_2)P(x_3\mid x_1)P(x_4\mid x_1,x_2)P(x_5\mid x_2)&#36;&#36;</p>
<p>In fact, the Bayesian network has three basic dependency structures, as follows:
<img src="/assets/images/machine-learning-notes/ml-bayesian-network-dependencies.png" alt="The Beyers web relies on structural indications."></p>
<p>We can find a problem in the V-type structure.&#36;x_4&#36; It affects his father's independence, and when the child is unknown, the two father's independence is not.</p>
<p>It's just a simple question of independence, and we understand the structure of the Bayesian network as sufficient.</p>
<h4>The Beyers Network.</h4>
<p>If the network structure is known, i.e. the dependency between attributes is known, the learning process of the Beyers network is relatively simple, and it is sufficient to estimate the probability of conditions for each node by “counting” the training sample.</p>
<p>In fact, we often do not know the network structure in practical applications, as in the semi-spure Beyers ODE, who is the parent node. The first task of Beyers' learning is to find the most structured "appropriate" Beyers network based on training data sets.</p>
<p>Query search is a common way of solving this problem. Specifically, we define a score function to assess the compatibility of the Beyeth network with training data, and then we base our search on this rating function on the best structured Beyth network. The selection of the rating functions affected the results of our last Beyers web.</p>
<p>Common scoring functions are usually based on informational guidelines, and we usually select the length of the code (including a description of the network).
The shortest beyets, the Minimal Description Length, is the code.</p>
<p>Unfortunately, the best Beyers network structure is a NH from all possible cyberstructures.
Hard to solve, hard to solve quickly. Two common strategies can be found to get close to it within a limited time: The first is the law of greed, the other is the method of defining structure, such as limiting the network structure to tree form.</p>
<h4>Inference</h4>
<p>The Beyers network is trained to answer the Query, which is through some attribute variables.
The observations are used to speculate on the extraction of other attribute variables. So the process of speculating for the search of variables is called "inferment" by the observation of known variables. &quot; (inference)</p>
<p>Ideally, the precise calculation of post-probability is based directly on the joint probability distribution as defined by the Beyers network, which unfortunately is difficult to extrapolate when the network has more nodes and is densely connected. We need to think about some of the more recent methods, and we're doing it now, usually using Gibbs Sampling.</p>
<p>You!&#36;\mathbf{Q}={Q_1,Q_2,\ldots,Q_n}&#36; This means that the variables are to be asked.&#36;\mathbf{E}={E_1,E_2,\ldots,E_k}&#36; For the evidence variable, the value is known to be&#36;\mathbf{e}={e_1,e_2,\ldots,e_k}.&#36; Target is for post-calculation.&#36;P(\mathbf{Q}=\mathbf{q}\mid\mathbf{E}=\mathbf{e})&#36;of which&#36;\mathbf{q}={q_1,q_2,\ldots,q_n}&#36;is a group of values to be asked for variables</p>
<p>Gibbs sampled algorithms that produced a random piece of evidence.&#36;\mathbf{E}=\mathbf{e}&#36;A consistent sample.&#36;\mathbf{q}^{0}&#36;As an initial point, and then each step from the current sample, the next sample is produced.&#36;t&#36; In subsampling, algorithms assume first&#36;\mathbf{q}^t=\mathbf{q}^{t-1}&#36;, then sample the non-evidence variable one by one to change the value to be taken, and the probability of sampling is based on the Beyers Network&#36;B&#36;, and the current value of other variables (i.e.,&#36;\mathbf{Z}=\mathbf{z})&#36;Calculation obtained. Assumed experience&#36;T&#36;Subsampling of the &#36;\mathbf{q}&#36; A consistent sample is shared. &#36;n_q&#36; One, which can estimate the probability of a posteriori.
&#36;&#36;P(\mathbf{Q}=\mathbf{q}\mid\mathbf{E}=\mathbf{e})\simeq\frac{n_q}{T}.&#36;&#36;</p>
<p>In fact, this is the MCMC method of Bayesian statistics. Chain</p>
<p>If you want to continue to understand how the Bayesian network is being expanded into a dynamic model and how it differs from the no-turn map model, readable<a href="/en/blog/2026/02/09/belief-network-learning/">Probability mapping model basis: Bayesian network, the Cain Markov model and Markov airport with them</a>。</p>
