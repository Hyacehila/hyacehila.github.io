---
title: 'Tree-Based Models Are Still SOTA for Tabular Data: XGBoost, LightGBM, and CatBoost'
title_zh: 表格数据上仍旧是SOTA：XGBoost、LightGBM 与 CatBoost
date: 2026-01-22 10:00:00 +0800
categories:
- Machine Learning
- Classical Machine Learning
tags:
- XGBoost
- Tabular Data
- Baselines
author: Hyacehila
mathjax: true
excerpt: Why are boosted trees still such strong baselines for structured tabular data? This post compares XGBoost, LightGBM,
  and CatBoost by technical focus and best-fit use cases.
description: Why are boosted trees still such strong baselines for structured tabular data? This post compares XGBoost, LightGBM,
  and CatBoost by technical focus and best-fit use cases.
excerpt_zh: 为什么结构化表格数据场景里，提升树至今仍常是最强 baseline？本文聚焦 XGBoost、LightGBM 与 CatBoost，讨论它们各自的技术重心、工程取舍与最佳战场。
permalink: /blog/2026/01/22/tree-based-models-tabular-data/
lang: en
translation_key: 2026-01-22-tree-based-models-tabular-data
translation_status: machine
translation_source_hash: 1af52bffa1677ea637dd7e78955cca1354e50948bc2765048547796d4cc42ac2
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<p>If the main battlefield of image, voice and natural language processing is already studied in depth today, then table data always resemble another completely different rhythm. There is no unified spatial structure, nor is there a natural continuous local model, which is replaced by hybrid type fields, missing values, abnormal values, long tail categories, operating rules, limited sample volumes, and unenviable engineering costs.</p>
<p>The questions in this article can also be addressed<a href="/en/blog/2026/04/24/autogluon-baseline-automl/">AutoGluon: Simplify machine learning Baseline to several lines of code</a>、<a href="/en/blog/2026/03/11/from-bagging-to-stacking-ensemble-learning/">From Bagging to Stacking: an integrated learning methodology map</a>How the concept of a relatively close read together is developed in different contexts.</p>
<p>And so, for the first time in the industrial machinery, many people feel that they are “era-fault”: Transformer, proliferation models and multi-modular systems on the one hand, and a large number of projects in wind control, advertising, referral, search, marketing, medical and industrial surveillance on the other, are still carefully compared. <code>XGBoost</code>、<code>LightGBM</code> and <code>CatBoost</code>。</p>
<p>It's not nostalgia, it's not a historical remnant of tree models. More precisely:<strong>In a large number of structured table tasks, tree upgrading remains the most common strong baseline and, in many scenarios, the best option or near best option.</strong> They have not been absent for a long time, for the simple reason that they remain valid.</p>
<p>This paper will not expand the whole ICF or discuss in detail the pedagogical extrapolation of random forests, Bagging and original GDDTs. I would like to focus on three of the most representative modern tree-upgrading systems:<strong>XGBoost, LightGBM and CatBoost</strong>I'm sorry. They all come from the GDDT spectrum, but the focus of the problem is completely different:</p>
<ul>
<li>XGBoost is a common, robust, manageable baseline selection.</li>
<li>LightGBM is the industrial system that pushes mass training and memory efficiency to the top.</li>
<li>CatBoost is a re-programming of the system around the type characteristics and target leaks.</li>
</ul>
<p>Understanding the differences between them is far more important than remembering “who is faster, who is more accurate and who supports the class characteristics”.</p>
<blockquote>
<p>This term “still SOTA” is not to say that they are always first in all data sets, all tasks and all input modes. More precisely, they remain the strongest baseline, common best solution or at least the closest type of method in the large number of structured table tasks.</p>
</blockquote>
<h2>Trey, why aren't you out yet?</h2>
<p>Before entering into three models, answer one root question: Why is it that in 2026, Tree-based Boosting is still strong on the table task for a long time?</p>
<h3>Isomer field: tables are never evenly spread</h3>
<p>The image is the rule grid, the text is sequenced, the voice is continuous; the table is much more confusing. Each column in Table Gerry may have a completely different semantic meaning: some are continuous values, some are high base category, some are Boolean fields, some are timetamps, times, percentages, and even string IDs and a small number of text. It is not a natural type of input space, but rather a isomer space that is composing business syntax.</p>
<p>The tree model is exactly what fits this imbalance. It does not require all features to share the same geometric structure or a uniform smooth transformation, but rather it cuts the sample space directly by “whether a field is larger than the threshold” and “whether a category falls into a pool”.<strong>For tables, Tree tends to be more inversely biased than a neuronet to the shape of the original problem.</strong></p>
<h3>Small and medium samples: data from the real world may not be large enough to reach End</h3>
<p>The strengths of in-depth learning are largely based on big data, pre-training resources and end-to-end expressions of learning ability. However, many real operations do not have such conditions. The sample of tabular tasks in the enterprise may be tens to hundreds of thousands, with a large number of features but of variable quality and distribution with marked long tails and operational deviations.</p>
<p>In such cases, a lifting tree with strong summary bias, clear parameters control and a more friendly approach to small and medium samples tends to give more stable results.<strong>Many tasks are not as large as they have to rely on heavy loads to indicate learning.</strong></p>
<h3>Missing and thin: default status of industrial data</h3>
<p>In the real table data, the " Unfilled " " " Not occurs " " System " , " System " is valid only for certain categories of users " , is almost a default status, not an exception. The missing values themselves often carry business information; the rare features are also daily in the tables.</p>
<p>The Tre-based model is natural here. A number of Boosting frames can directly incorporate missing values into the divide logic without a single round of rough fillings. For high-dimensional thinness, tree models are also often closer to the luxuries of information only.</p>
<h3>Feature interaction: first meet the conditions before the next step is discussed</h3>
<p>A number of table tasks are not derived from a single field, but from a conditional interaction between fields. For example, “overdue and recent trades have been abruptly reduced” “Equipment types are of certain types and geographically in some cities” “high number of hits but low cost per passenger unit”. Such models are neither linear nor simple non-linear, but more like a set of conditioned f-else rules.</p>
<p>The tree model is structured precisely in this language: it cuts out a local area and continues to subdivided it. And that's why Trey is particularly sensitive to such issues as “determinating the next step when conditions are established”.</p>
<h3>Project constraints: Explanatory, training costs and iterative efficiency</h3>
<p>Industrial machines never learn more than their limit precision. Training time, parameter searches, effectiveness stability, feature importance, SHAP access, line delay and deployment price will all alter model selection.</p>
<p>The trees are strong for the long term, also because they form mature ecology in these dimensions. Teams often need not the SOTA in the sense of the paper, but rather the stable handling of the task today.</p>
<h2>From Random Forest to GDDT: Before the official content begins</h2>
<p>The main character of this paper is not random forest, nor is it original GDDT, but a short mattress is still needed to keep the discussions from going.</p>
<h3>Random Forest: The classic strong baseline on Bagging's route</h3>
<p>Random Forest's idea is to train multiple trees in parallel, to vote or to average, along the Bagging route, with a technical focus on reducing the difference. It remains a strong baseline and suitable for use as an introductory model and control group.</p>
<p>But it is not the focus of this paper, because I will then write a separate article on integrated learning, which will make it clear in a single framework that Bagging, Boosting, Random Forests, and the Difference-Equal Balance.</p>
<h3>GLDT: Common parent of three modern lifting trees</h3>
<p>If the random forest is Bagging, then the GDDT is Boosting:</p>
<p>&#36;&#36;
F_t(x) = F_{t-1}(x) + \eta f_t(x)
&#36;&#36;</p>
<p>The tree here. &#36;f_t&#36; Instead of independent training, the current model has not been completed. Intuitively, it is constantly correcting errors; more strictly, it is moving in the direction of loss in the functional space.</p>
<p>Original GDDT is important because XGBoost, LightGBM and CatBoost are all on this main line. But in today ' s engineering practice, what really affects experience and performance is how these modern achievements deal with target functions, split search, missing values, class characteristics, memory layouts and parallel efficiency.</p>
<h2>XGBoost: Move GGBDT from "good experience" to "methodology."</h2>
<p><a href="https://arxiv.org/abs/1603.02754">XGBoost: A Scalable Tree Boosting System</a> The impact is not just from speed. It's a practical way of moving GDDT forward.<strong>Target functions are clear, regularized, approximate algorithms are complete, thin process mature, work solid</strong>The tree system.</p>
<h3>Why is it the first stop of many people's defaults?</h3>
<p>The observation form is a learning exercise and shows a steady pattern: When the task is not fully tacted, people are often willing to try an XGBoost first. The reason is not mysterious, but rather the sense of balance that is present in the whole picture, rather than the occasional speciality of an indicator.</p>
<p>It is strong and stable, it is flexible in target functions, it is relatively clear in the logic of hyperparameters and it is natural to process missing values. XGBoost often offers<strong>A reliable Baseline.</strong>It is used to clarify the structure of the problem and then to determine whether to pursue speed, class characteristics or some specific optimization more aggressively.</p>
<h3>Math Identity: Structure Regular is written directly into the target function</h3>
<p>One of the most significant contributions of XGBoost is to clearly define the optimization of each new tree round as a target with a complex punishment:</p>
<p>&#36;&#36;
\mathcal{L}^{(t)} = \sum_i l\big(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)\big) + \Omega(f_t)
&#36;&#36;</p>
<p>The tree's regular item reads:</p>
<p>&#36;&#36;
\Omega(f) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2
&#36;&#36;</p>
<p>Here. &#36;T&#36; The leaves count,&#36;w_j&#36; It's the first. &#36;j&#36; A leaf's weight. The idea behind this is:<strong>The trees are not as complex as they are, and each leaf must prove worth it.</strong> &#36;\gamma&#36; The complexity of the punishment structure,&#36;\lambda&#36; The leaves are not too heavy to weigh.</p>
<h3>Training core: second stage roll-out, leaf weight closed and split main</h3>
<p>Another key point for XGBoost is that it does not look at the first step gradient, but rather at the second step of the loss approximation:</p>
<p>&#36;&#36;
\tilde{\mathcal{L&#125;&#125;^{(t)} \approx \sum_i \left[g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i)\right] + \Omega(f_t)
&#36;&#36;</p>
<p>of which &#36;g_i&#36; It's a step gradient.&#36;h_i&#36; It's a second-stage gradient. This second-tier perspective brings two direct benefits.</p>
<p>First, the optimal weight of the leaves is closed to the structure of the given tree:</p>
<p>&#36;&#36;
w_j^* = -\frac{G_j}{H_j + \lambda}
&#36;&#36;</p>
<p>Second, the benefits of the candidate division can also be written in a prominent way:</p>
<p>&#36;&#36;
\text{Gain} = \frac{1}{2}\left(\frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda}\right) - \gamma
&#36;&#36;</p>
<p>This thing makes the "separation" in the XGBoost a clear optimisation.<strong>Gains must be substantial enough to cover structural complexity penalties.</strong></p>
<h3>Project core:<code>exact</code>、<code>approx</code>、<code>hist</code> With the right-winged quantile sketch</h3>
<p>Many people still have the impression that XGBoost is in precise greddy search, but this is not the most common view in modern practice. According to the official <a href="https://xgboost.readthedocs.io/en/stable/treemethod.html">Tree Methods</a> Documents, XGBoost, have three main split paths:</p>
<ul>
<li><code>exact</code>: accurate greeny search, accurate but slow, least scalable;</li>
<li><code>approx</code>: Construct a candidate point using a weighted bitsketch;</li>
<li><code>hist</code>: First, do a global partition, then build on histogram training, which is usually the most common route today.</li>
</ul>
<p>It's particularly worth naming here. <strong>weighted quantile sketch</strong>I'm sorry. It does not disperse the continuous feature by hand, but is designed to be a similar structure specifically designed for the weighted fraction, aligning the structure of the candidate cut points with the optimisation logic of the second tier.</p>
<p>This is also a typical feature of XGBoost: it is not necessarily the most radical, but it is often complete enough.</p>
<h3>Scatter and Missing: To treat absence as part of the building</h3>
<p>XGBoost <strong>sparsity-aware split finding</strong> The default direction for each node to learn the missing values. That is, when a sample is missing on the current split, left or right, not pre-coded, but in training.</p>
<p>This brings two practical advantages: first, you do not need to fill all missing values in a rough round to get the model running; second, for high-dimensional thin input, it is closer to the partially utilised amount of information only.</p>
<p>Therefore, the "XGBoost Supported Missing Value" is stated as a simple functional introduction. More precisely:<strong>It would see the missing as part of the need to optimize together in a divisive structure.</strong></p>
<h3>What do you think of today?</h3>
<p>It was often summed up by a phrase: XGBoost supports non-ingenital type characteristics. That statement is outdated today. According to the official <a href="https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html">Categorical Data</a> The program, which supports the class data from 1.5. <a href="https://xgboost.readthedocs.io/en/stable/changes/v3.1.0.html">3.1 Release Notes</a>,3.1 Version started this capability by removing the executive label.</p>
<p>But this needs to be more detailed:</p>
<ul>
<li>Categorical spit dependency <code>hist</code> or <code>approx</code>, do not support <code>exact</code>；</li>
<li>Models should normally be stored using JSON / UBJSON, otherwise information on categories may be lost;</li>
<li>It supports one-hot and part-based split, but the "class feature processing" is still not its core design centre.</li>
</ul>
<p>Today, XGBoost cannot handle the categorical; but it would be too much to understand if it were to be understood as a CatBoost alternative.</p>
<h3>It's the strongest scene.</h3>
<p>The most natural battlegrounds in XGBoost are the following types of tasks:</p>
<ul>
<li>Numerical characteristics are the main table tasks;</li>
<li>A robust baseline is needed to determine whether to pursue further projects that are either insinuated or treated by category;</li>
<li>Special task(s) requiring special objective, such as sorting, survival analysis, counting modelling, fractional regression;</li>
<li>(b) The scenario where the missing value is high and the model is expected to absorb missing structural information;</li>
<li>Industrial systems that require fine particle control training processes.</li>
</ul>
<h3>Misunderstanding and borders: robust, not the equivalent of almighty</h3>
<p>With respect to XGBoost, these are often the most common types of misunderstandings:</p>
<ul>
<li><strong>"XGBoost is just faster than GDDT."</strong> That's way underestimating it. It's worth making a structure, a goal-full, a mature system.</li>
<li><strong>"XGBoost does not support class characteristics."</strong> This is outdated today, but its methodological focus on the characteristics of the category is still less central than CatBoost.</li>
<li><strong>"XGBoost's strength is still the exact greeny."</strong> In modern practice,<code>hist</code> It is often the mainstream.</li>
<li><strong>"Leave value is 0."</strong> For trees, the two concepts are not equal, and the missing direction is to learn in training.</li>
</ul>
<p>Its borders are also clear: on a very large, high-dimensional and thinly broad scale, its ingestion and memory efficiency is often less radical than that of LightGBM; nor is it the most natural first choice in the task of the high base category.</p>
<h3>Summary of the sentence</h3>
<p>The advantage of XGBoost is not always the fastest, but...<strong>Steady, universal, manageable.</strong>I'm sorry. If you don't know which tree you should start with, XGBoost is often the most natural starting point; if you don't know what model you're going to use, it's always the first stop.</p>
<h2>LightGBM: When bottlenecks turn into swallowing, memory and width size</h2>
<p>If XGBoost is complete and robust, then <a href="https://papers.nips.cc/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf">LightGBM: A Highly Efficient Gradient Boosting Decision Tree</a> It's more like a distinct engineering system from the beginning: it's focused on design.<strong>Large-scale training speed, memory efficiency and wide-ranging vomiting</strong>Go on.</p>
<h3>Why is it more engineering from the beginning?</h3>
<p>LightGBM is so common in advertising, recommendation, search sequencing, CTR estimates, etc., not because it simply takes a little speed on XGBoost, but because it has reorganised the training package. According to the official <a href="https://lightgbm.readthedocs.io/en/v4.5.0/Features.html">Features</a> Document, whose advantages are derived from a set of synergistic designs: histogram, histogram sub-section, leaf-wise growth, GOOSS, EFB, thin optimization and distributed friendly.</p>
<p>LightGBM closer<strong>A re-engineered Boosting system for training on large-scale industrial forms</strong>, not just another GDDT library.</p>
<h3>Speed source: histogram and histogram subtration</h3>
<p>LightGBM will first disperse the continuous feature to a limited number of bins, then find the best cut points on histogram. The benefits of this are straightforward: the cost of building histogram remains related to the sample number, but once hetogram is built, only the bin is scanned, without having to scan all original values repeatedly.</p>
<p>The bigger speed point is from <strong>histogram subtraction</strong>I'm sorry. The parent's node, histogram, is quickly available through the Parent's node minus bronode, as long as hetogram is clearly constructed for one of the subnodes. That sounds like the details to be achieved, but it is one of the important sources of the speed of training.</p>
<h3>Structural philosophy: Leaf-wise why it's stronger and why it's more dangerous</h3>
<p>LightGBM's most famous and easily misunderstood design is its default adoption. <strong>leaf-wise / best-first</strong> Growing, not more conservatively by layer. Each step of the way it picks the leaves that currently bring the largest loss of the losaic, and continues to split, rather than extending the whole layer down.</p>
<p>The result is straightforward: in the same leaf budget, leaf-wise tends to lower training losses more quickly and thus often provides a greater alignment capacity within a fixed time frame. But the problem is also from here:<strong>More radical, and usually more easily conciliated, is also implied.</strong></p>
<p>So in LightGBM, controls are not always just seen. <code>max_depth</code>, and control it simultaneously. <code>num_leaves</code>、<code>min_data_in_leaf</code>The government has also been working on a project to improve the quality of the media. A lot of starters think it's a mistake. <code>max_depth</code> It's like turning LightGBM back into a level-wise, which is not true, because its growth philosophy itself is not changed.</p>
<h3>Sample dimensions optimization: GOSS is not just subsample</h3>
<p>It's from the LightGBM paper. <strong>GOSS（Gradient-based One-Side Sampling）</strong>, often misunderstood as another random sample. But it's different from the ordinary subsample's problem consciousness.</p>
<p>During the Boosting process, the absolute size of the gradients usually means that the models have not learned them well. So the GOSS strategy is not to delete samples evenly and randomly, but to:</p>
<ol>
<li>(a) To retain a portion of the large gradient sample;</li>
<li>(a) Randomly extract a portion from a small gradient sample;</li>
<li>The latter is compensated for weighting to keep the estimates of gains as high as possible from being heavily distorted.</li>
</ol>
<p>The instinct behind it is:<strong>What can't be lost easily is those samples that are not yet understood.</strong> This is more abosting than simple subsample training logic.</p>
<h3>Characteristic dimensions optimization: why EFC is particularly suitable for width</h3>
<p>Besides the sample dimensions, LightGBM has structural compression of the characteristic dimensions, which is <strong>EFB（Exclusive Feature Bundling）</strong>。</p>
<p>It observes that many of the rare features are mutually exclusive, such as a column after one-hot has been expanded, and that the same sample usually has only one column activated. Since these features are almost non-zero at the same time, they can be packaged into the same bookle, thereby significantly reducing histogram construction costs.</p>
<p>This is why LightGBM is so natural about high-dimensional slush tables such as recommendation, advertising, and searching for sorting. It is best handled, often not by dozens of neat numerical features, but by hundreds of, if not more, coded and spelled industrial broad-table fields.</p>
<h3>Category support: it can handle the categoric, but not the CatBoost type of treatment</h3>
<p>LightGBM original support for the categoric world, of course, it's worth it. But it's a category of characterization route and CatBoost is completely different. According to the official <a href="https://lightgbm.readthedocs.io/en/v4.5.0/Advanced-Topics.html">Advanced Topics</a> document, which is essentially the best two-pointer in the order of the statistically ordered category, instead of ordered statistics.</p>
<p>It addresses issues more in favour of:<strong>How to efficiently divide categories</strong>; instead of treating category code leakage and Boosting deviations under the same principle as CatBoost.</p>
<p>At the same time, several borders must be remembered:</p>
<ul>
<li>Category characteristics should normally be coded as non-negative integers;</li>
<li>Negative values are considered missing;</li>
<li>High base category does not necessarily naturally fit natural, and official documents themselves caution this matter.</li>
</ul>
<h3>Missing value:<code>zero_as_missing</code></h3>
<p>LightGBM default supports missing values, but <code>zero_as_missing</code> Changes the semantic boundary between numeric 0 and missing values.</p>
<p>This is particularly dangerous in the rare matrix, the LibSVM format or the industrial characteristic of "0 without presence". If 0 is a state of clarity in business, and you treat it as a missing item, then the model sees it not as the watch you think it is.</p>
<p>With regard to LightGBM, it is not “it is fast” that deserves to be emphasized, but:<strong>It's quick, but it requires you to be sufficiently clear about the semantics of data.</strong></p>
<h3>It's the strongest scene.</h3>
<p>LightGBM's most natural advantage scenario, which is basically these:</p>
<ul>
<li>(a) The number of samples is large, reaching hundreds of thousands, millions or even higher;</li>
<li>The features are numerous and thin.</li>
<li>Training in ingestion and memory occupation are major contradictions;</li>
<li>The tasks are typical industrial issues such as CTR, recommendation, search sequencing, advertising, wind-control bandwidth;</li>
<li>A large number of experimental iteratives need to be completed quickly.</li>
</ul>
<p>In many cases, the value of LightGBM is not just in the final indicator, but in the way that you can run the experiment faster, adjust the parameters and sift the features.</p>
<h3>Misunderstanding and boundaries: fast, not automatically saving hearts</h3>
<p>With regard to LightGBM, the most common misconceptions include:</p>
<ul>
<li><strong>"A primary supported category feature is equal to any string that can be eaten directly. "</strong> It still requires a reasonable integer code.</li>
<li><strong>"leaf-wise must be more comprehensive than level-wise."</strong> More precisely, it usually lowers the loss, but it's easier to match.</li>
<li><strong>"GOS is an alias of subsample."</strong> No, it is essentially "retain the hard sample + weight".</li>
<li><strong>"the high base category is naturally suitable for the LightGBM natural."</strong> This must be done with caution.</li>
<li><strong>“<code>zero_as_missing</code> It's just a little switch."</strong> It may actually change the semantic of data directly.</li>
</ul>
<p>It's also clear: it's not always the least cost-effective in small data, especially when it's a validation protocol,<code>num_leaves</code> When the formal constraints are not controlled, the joint meeting is faster.</p>
<h3>Summary of the sentence</h3>
<p>LightGBM is the most like "High-intensity Boostling System for Large-scale Industrial Forms Learning". The main advantage of this is that<strong>Speed, memory and width matching</strong>- Not more stable.</p>
<h2>CatBoost: Combining the category characteristics with the target leaking problem Rewrite</h2>
<p>Put three together, CatBoost's highest-resolution. It's the most valuable place, not just to support class characteristics, but to be like, <a href="https://arxiv.org/abs/1706.09516">CatBoost: unbiased boosting with categorical features</a> The paper highlighted:<strong>It deals with the deviations from the category characteristics and the Boosting itself, and it is reworked under the same design principle.</strong></p>
<h3>Why does it have a different sense of consciousness than the other two?</h3>
<p>When people first met CatBoost, they remembered that it was good at dealing with categoric feature. Of course it is, but it is not enough. CatBoost, the real difference is that it's not about how to cut categories more efficiently, but about how to start:<strong>Will the statistics of the training phase leak back into the target information?</strong></p>
<p>This makes it natural to focus on two other families. XGBoost places greater emphasis on target integrity and controlability, LightGBM places greater emphasis on insulation, memory and width efficiency, while CatBoost has been dealing with predation Shift and target leakage from the beginning.</p>
<h3>First level: ordered handling predation shift</h3>
<p>Traditional GDDTs are used for each round of training using the current model for the training set's disability or gradient. The problem is that these disabilities are themselves affected by the same training sample labels, and the statistical structure observed at the training and testing stages is not entirely consistent. CatBoost calls this deviation <strong>prediction shift</strong>。</p>
<p>The solution is... <strong>ordered boosting</strong>: randomly sorted training samples first; for a sample, in the &#36;t&#36; The training statistics are constructed using only sample information before the array. Intuitively, it's to try to make the training sample more like a real new one.</p>
<p>This requires a boundary to be filled: the distinction between CatBoost's official parameters <code>Ordered</code> and <code>Plain</code> Two types of boosting, the default behaviour is affected by CPU/GPU, sample size and task type. Therefore, the more precise statement should be:<strong>Ordered Boosting is CatBoost's representative mechanism, but it is not the same as any configuration is defaulted on.</strong></p>
<h3>Second floor: avoid surface space</h3>
<p>The most easily pedaled pit in the type characterization is the target leakage. Normal tranged encoding if the category target is calculated directly on the whole set, then the current sample's own label will leak back to its own signature expression.</p>
<p>CatBoost <strong>ordered target statistics</strong> This problem is mitigated by “using data only before the current sample”. One way to visualize is:</p>
<p>&#36;&#36;
\text{CTR}<em>i = \frac{\text{countInClass}</em>{&lt;i} + \text{prior&#125;&#125;{\text{totalCount}_{&lt;i} + 1}
&#36;&#36;</p>
<p>Here&lt;i. &#36; indicates only the records that appear before the current sample in the array. It is very clear:<strong>The current sample does not leak its own label into its own code.</strong></p>
<p>Further, CatBoost also constructs group features. It makes it right. <code>user_id × item_id</code>、<code>city × device</code>、<code>ad_slot × hour</code> Such interaction is particularly sensitive and explains why it often behaves naturally in high-base category assignments.</p>
<h3>Level 3: SymmetricTree gives it a high degree of visibility.</h3>
<p>CatBoost's classic default tree structure is <strong>SymmetricTree</strong>(also commonly known as obblivious tree) It is characterized by the fact that all nodes on the same level use the same partition condition and therefore the depth is as follows: &#36;d&#36; ♪ The trees will be ♪ &#36;2^d&#36; A leaf.</p>
<p>This structure brings three things:</p>
<ul>
<li>(b) The reasoning path is well structured and online forecasts are usually fast;</li>
<li>(a) A more structured and self-contained regularity;</li>
<li>Expressional ability is also constrained, not only by profit.</li>
</ul>
<p>CatBoost has been supported in the current official parameter document <code>grow_policy = SymmetricTree / Depthwise / Lossguide</code>I'm sorry. The exact statement should therefore be:<strong>CatBoost's classic default form is derived from symmetric trees, but it does not support symmetric trees alone.</strong></p>
<h3>Time and text:<code>has_time</code> And the true meaning of the built-in text lines</h3>
<p>CatBoost has two other points that deserve the attention of the industrial reader.</p>
<p>The first one is... <code>has_time</code>I'm sorry. If the data are in a natural chronological order, then you do not want to do anything to disrupt the sample during the training phase and then construct the type statistics. Otherwise, much of the future information may be re-routed into the training process through a code sequence or cross-check process.<code>has_time</code> The point is to remind you:<strong>Order itself is part of the data structure.</strong></p>
<p>The second is its text feature pipe:<code>tokenizers -&gt; dictionaries -&gt; feature calcers -&gt; numerical features -&gt; boosting</code>I'm sorry. The features calculators it supports include BoW, NaiveBayes, BM25, etc. This is practical, as many table tasks do mix short text fields, search terms, titles, label descriptions, etc.</p>
<p>But the border must also be clear:<strong>CatBoost's text support is not a tree model, Bert, but a classic text statistical feature project embedded in the Boosting system.</strong></p>
<h3>It's the strongest scene.</h3>
<p>CatBoost's most important priority scenario, which is basically the following:</p>
<ul>
<li>(a) The characteristics of the category are numerous and many are high-base category;</li>
<li>(a) The discrete fields of user ID, commodity ID, equipment, geography, channels, advertising posts are very strong;</li>
<li>Not to write a large number of projects by hand;</li>
<li>Data type heterogeneity, values, categories and small amounts of text;</li>
<li>Hope to get a strong baseline quick without pre-treatment.</li>
</ul>
<h3>Misunderstanding and boundaries: good at categories, not the best for all tables</h3>
<p>With regard to CatBoost, the most common misconceptions include:</p>
<ul>
<li><strong>"CatBoost is just automatic target encoding."</strong> Actually, ordered Prince works both in code and boosting two layers.</li>
<li><strong>"CatBoost must be defaulted for ordered booping."</strong> This depends on the current configuration and training backend.</li>
<li><strong>"CatBoost only trains symmetric trees."</strong> This is not the case with official documents at the moment.</li>
<li><strong>"With CatBoost, there's no time-spacing."</strong> Still, the right time is to be split.</li>
<li><strong>"Standing support for CatBoost is equivalent to deep semantic modelling."</strong> No, it is still a text feature project in the tree model context.</li>
</ul>
<p>Its boundaries are also clear: in pure numbers, super-sized, trained-to-smuggling priorities, LightGBM is often more attractive; and CatBoost’s core advantages are not necessarily released when the characteristics of the categories are not too numerous.</p>
<h3>Summary of the sentence</h3>
<p>CatBoost's real value is not "can also eat class features," but...<strong>The goal of avoiding leakage has been elevated into a uniform design principle and the matter has been incorporated into the category code and the Boostling training process.</strong></p>
<h2>Three routes: what are they optimizing?</h2>
<p>If only three core differences are observed, they can be summarized in the table below.</p>
<table>
<thead>
<tr>
<th>Model</th>
<th>Design Centre</th>
<th>Tree Growth Policy</th>
<th>Category characterization processing</th>
<th>Speed/RAM</th>
<th>Default robustness</th>
<th>The best job.</th>
<th>Main costs</th>
</tr>
</thead>
<tbody><tr>
<td>XGBoost</td>
<td>Universal Boosting, Visible Regulars, Flexible Targets</td>
<td>More conservative, frequently depth-wise</td>
<td>One-hot and partSpit support, but not design center</td>
<td>Medium Stable</td>
<td>High</td>
<td>Table with main values, missing values, needing robust baseline or special objective</td>
<td>A large-scale broad-band stale may not be a good idea.</td>
</tr>
<tr>
<td>LightGBM</td>
<td>Industrial ingestion, memory efficiency, broad-scale optimization</td>
<td>Defaultleaf-wise / best-first</td>
<td>Native class splits, but does not resolve</td>
<td>Usually the fastest, most economical memory</td>
<td>Medium</td>
<td>Million-degree samples, high-dimensional swirl list, CTR, recommendation, sorting</td>
<td>Small data is easier to match, and more semantic pits are available.</td>
</tr>
<tr>
<td>CatBoost</td>
<td>Category characterization modelling, leakage control, reduced manual code</td>
<td>Classic default is SymmetricTree</td>
<td>Ordered target statuses / CTR is the core advantage.</td>
<td>Training is usually slower than LightGBM. The reasoning is very well-defined.</td>
<td>High</td>
<td>High base category, ID feature intensive, isomeric tables, small text mixing</td>
<td>Training costs are higher. Pure values are too large to be optimal.</td>
</tr>
</tbody></table>
<p>So the three really represent not “who is higher than whom”, but three completely different technical positions:</p>
<ul>
<li><strong>XGBoost</strong>(a) robust, universal and manageable;</li>
<li><strong>LightGBM</strong>High-intensity, broad-mindedness, industrial efficiency priority;</li>
<li><strong>CatBoost</strong>: Class characterization modelling and leak control priority.</li>
</ul>
<h2>What do you choose when you actually land?</h2>
<p>In engineering practice, the most important is not back-model history, but rather the development of enforceable selection principles.</p>
<ol>
<li><strong>If the features are numerical, it is hoped that a robust, manageable, ecologically mature baseline will be obtained first, with priority for the XGBoost test.</strong></li>
<li><strong>If it is faced with a large, high-dimensional, thin-wielding scale, training for ingestion and memory efficiency is the first contradiction, giving priority to the LightGBM.</strong></li>
<li><strong>If the class characteristics, ID characteristics, high base numbers are so numerous, and you don't want to write a bunch of code logic yourself, try CatBoost first.</strong></li>
<li><strong>If the task is clearly sequenced, no model can substitute for the correct time-slicing validation.</strong> CatBoost's ordered idea is helpful, but assessing protocols is more important than model names.</li>
<li><strong>If it is not clear who is better, all three are considered as a line candidate, with fair comparison under the same identity, cross-checking and evaluation indicators.</strong> The real credit is the out-of-fold result of your mission.</li>
</ol>
<p>In other words, the key to the selection is not to ask who is the most advanced, but to ask:<strong>What kind of watch is my watch? Are my bottlenecks generalized, insinuated, class-specific, or are they the cost of characteristic engineering?</strong></p>
<h2>Concluding remarks</h2>
<p>The competition in table data has never been a pure model complex. In many cases, the decision is not about who has more parameters, who is closer to the data structure, who is more fit for engineering constraints, who can more steadily bring the signal out of the table.</p>
<p>XGBoost, LightGBM and CatBoost are still important not because they are only three GGBDT libraries, but because they represent three very clear technical positions:</p>
<ul>
<li>XGBoost Representative <strong>Steady, universal, manageable.</strong>；</li>
<li>LightGBM Representative <strong>High-intensity, broad-mindedness, industrial efficiency priority</strong>；</li>
<li>CatBoost, on behalf of <strong>Category characterization modelling and leakage control priority</strong>。</li>
</ul>
<p>A truly mature engineering judgement is never the dogma who says it's always the best, but rather the knowledge that:<strong>Better definitions are not always the same in the different tabular forms and data distributions.</strong></p>
<h2>References</h2>
<h3>XGBoost</h3>
<ul>
<li><a href="https://arxiv.org/abs/1603.02754">XGBoost: A Scalable Tree Boosting System</a></li>
<li><a href="https://xgboost.readthedocs.io/en/stable/treemethod.html">XGBoost Tree Methods</a></li>
<li><a href="https://xgboost.readthedocs.io/en/stable/parameter.html">XGBoost Parameters</a></li>
<li><a href="https://xgboost.readthedocs.io/en/stable/faq.html">XGBoost FAQ</a></li>
<li><a href="https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html">XGBoost Categorical Data</a></li>
<li><a href="https://xgboost.readthedocs.io/en/stable/changes/v3.1.0.html">XGBoost 3.1 Release Notes</a></li>
</ul>
<h3>LightGBM</h3>
<ul>
<li><a href="https://papers.nips.cc/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf">LightGBM: A Highly Efficient Gradient Boosting Decision Tree</a></li>
<li><a href="https://lightgbm.readthedocs.io/en/v4.5.0/Features.html">LightGBM Features (v4.5)</a></li>
<li><a href="https://lightgbm.readthedocs.io/en/v4.5.0/Advanced-Topics.html">LightGBM Advanced Topics (v4.5)</a></li>
</ul>
<h3>CatBoost</h3>
<ul>
<li><a href="https://arxiv.org/abs/1706.09516">CatBoost: unbiased boosting with categorical features</a></li>
<li><a href="https://catboost.ai/docs/en/concepts/algorithm-main-stages_fighting-biases">CatBoost Unbiased Boosting</a></li>
<li><a href="https://catboost.ai/docs/en/concepts/algorithm-main-stages_cat-to-numberic">CatBoost: Transforming categorical features to numerical features</a></li>
<li><a href="https://catboost.ai/docs/en/references/training-parameters/common">CatBoost Common Training Parameters</a></li>
<li><a href="https://catboost.ai/docs/en/concepts/parameter-tuning.html">CatBoost Parameter Tuning</a></li>
<li><a href="https://catboost.ai/docs/en/concepts/algorithm-main-stages_text-to-numeric">CatBoost: Transforming text features to numerical features</a></li>
</ul>
