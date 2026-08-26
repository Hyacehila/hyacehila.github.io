---
title: 'From Euclidean Space to Manifold Topology: Dimensionality Reduction for High-Dimensional Data'
title_zh: 从欧氏空间到流形拓扑：高维数据的降维之旅
date: 2026-01-16 10:00:00 +0800
categories:
- Data Science
- Statistical Modeling & Inference
tags:
- Dimensionality Reduction
- Statistics
- Statistical Inference
author: Hyacehila
excerpt: Starting from high-dimensional, small-sample settings and the curse of dimensionality, this post frames dimensionality
  reduction alongside EDA, regularization, and sparsity before surveying PCA/MDS, t-SNE, UMAP, and autoencoders.
description: Starting from high-dimensional, small-sample settings and the curse of dimensionality, this post frames dimensionality
  reduction alongside EDA, regularization, and sparsity before surveying PCA/MDS, t-SNE, UMAP, and autoencoders.
excerpt_zh: 从高维小样本与维数灾难出发，梳理 EDA、正则化与稀疏性等高维统计思路，并重点介绍 PCA/MDS、t-SNE、UMAP、Autoencoder 等降维方法的数学直觉与适用场景。
permalink: /blog/2026/01/16/dimensionality-reduction-high-dimensional-data/
lang: en
translation_key: 2026-01-16-dimensionality-reduction-high-dimensional-data
translation_status: machine
translation_source_hash: 88c337b3830c93b550550158c1de7bae65716e9b0f92910e7a7a3ce6e8ff1aaa
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<h2>Motives and context of the decline</h2>
<p>The questions in this article can also be addressed<a href="/en/blog/2023/09/04/linear-regression-basics-notes/">Linear regression base: linear model, minimum 2x2 estimate and regression diagnosis</a>、<a href="/en/blog/2023/09/12/statistical-computing-notes/">Statistical calculations: random number generation, random variable simulation and Monte Carlo method</a>How the concept of a relatively close read together is developed in different contexts.</p>
<h3>The challenge of high-dimensional data</h3>
<p>Modern data science and machine learning often involves processing hundreds of thousands of, if not more, dimensions. The dimensions (i.e. the number of characteristics) are common once they are close to or above the sample amount <strong>"Gavy Small Sample"</strong> Scenario: The validity, stability and interpretability of traditional reliance on statistical methods that “sample numbers are much greater than their dimensions” are tested.</p>
<p>High-dimensional is not a simple extension of low-dimensional data. It also presents challenges in computing, perception and statistics: rapid increases in the cost of revolving sub-units or complex estimates; the difficulty of humans taking direct control of high-dimensional geometry structures; and the increasing scarcity of limited samples in larger spaces, which destabilizes many classical methods that rely on local neighbourhood or distribution estimates.</p>
<h3>Dimensions disaster: from kNN to &#36;p \gg n&#36;</h3>
<p>The most representative difficulty is that <strong>"Curse of Demension"</strong>I'm sorry. As the dimensions grow, the number of samples required to maintain the same sampling density increases dramatically; the distance between the nearest and the most distant neighbours in the high-dimensional zone tends to shrink, leading to a gradual loss of distinction in distance itself.</p>
<p>In the case of k Neighbor (kNN), it selects the closest to the test sample based on distance &#36;k&#36; A training sample, which is projected by voting or weighted averages; its effectiveness relies on sufficiently intensive samples and distance measures that distinguish the proximity. But when space becomes thinner, nuclear density estimates, and the like are robust in low dimensions, either a near explosive increase in sample volumes or an increase in the range makes it unreliable.</p>
<p>Empirically, using “at least 5 observations per dimension” as a rough threshold for reliable modelling; if the number of features is &#36;p&#36;, corresponding empirical baseline is about &#36;5p&#36; One observation. But in genetic expression, text mining, etc., it's common. &#36;p \gg n&#36; In the circumstances, this requirement is often not met.</p>
<h3>Response framework for high-level statistics</h3>
<p>Rather than simply abandoning the traditional framework, GVS is reorganizing the analysis process through structural assumptions and new theoretical tools. Explored data analysis (EDA) can first explore potential models with relevance analysis, cluster and anomaly testing to provide empirical evidence for subsequent assumptions; and retroversion can offer a moderate deviation in exchange for lower deviations, which will stabilize projections under limited samples.</p>
<p>Structural assumptions such as slurability and low stylity further re-enable variable selection, co-scaling estimates and high-dimensional extrapolation to be identifiable and statistically assured. Under these conditions, the theory of convergence and non-accuracy no longer requires “fixed dimensions, sample sizes are endless”, but allows dimensions to grow in step with sample volumes, providing the theoretical basis for the rare regression, modeling and so on.</p>
<p>And that means that Gaul is not just a curse. It forces us to focus on the balance between computing feasibility, statistical efficiency and interpretability, while also potentially bringing about better linear symmetry and providing space for such ideas as nuclear approaches. The key is not to increase or reduce the dimensions mechanically, but to identify the structures that are actually available in the data.</p>
<h3>Why is the decline the key?</h3>
<p>In these responses,<strong>Demutation</strong> This is particularly important: it places high-dimensional data into more compact representations, retaining meaningful geometry, distance or local neighbourhood structures under manageable information losses, thus serving visualization, exploration, computing and subsequent modelling. It can find effective low-dimensional spaces and reveal potential non-linear currents.</p>
<p>Instead of replicating the entire high-dimensional statistical spectrum, the following is a step-by-step discussion of the more common downscaling route and its mathematical intuition, starting with the traditional approach of maintaining global structures. The review and comparison of the PCA, AE etc. that have already been developed in other notes is still dominated by application levels, without replicating the theory.</p>
<h2>Traditional approach to maintaining global structure</h2>
<p>Early downscaling techniques focus mainly on how to keep data points between data points<strong>Global Geometry</strong>(like the distance of the o'clock)</p>
<h3>MDS: Faithful reduction of distance</h3>
<p>MDS (Multiple Dimensional Scaling) is a classic algorithm for keeping distance. It's a simple hunch:<strong>If two dots are far away in high space, they should also be far away in low space.</strong></p>
<p>Assumptions &#36;m&#36; The distance matrix of a sample in the original space is &#36;D&#36;, the weight is &#36;dist_{ij}&#36;I'm sorry. Our goal is to find low-dimensional mapping. &#36;Z \in \mathbb{R}^{d^* \times m}&#36;♪ And make ♪ &#36;|z_{i}-z_{j}| \approx dist_{ij}&#36;。</p>
<p>Mathically, by building an internal matrix. &#36;B = Z^{\mathrm{T&#125;&#125; Z&#36;, using cosine theorem to extrapolate:
&#36;&#36;b_{ij}=-\frac{1}{2}(dist_{ij}^{2}-dist_{i.}^{2}-dist_{.j}^{2}+dist_{..}^{2})&#36;&#36;
of which &#36;dist_{i.}^2&#36; equal to the average of a row or column. ♪ Against the Matrix ♪ &#36;B&#36; Decomposition of feature values:
&#36;Z = \Lambda <em>}^{1/2} V_{</em>You're not gonna get me out of here.
The coordinates are available taking the number of characteristics that corresponds to the number of characteristics.</p>
<p>The main limitation of MDS is that it attempts to keep the distance between all points, and this stringent requirement for European distance (or possibly other distance) can easily be mistaken. When processing non-linear data (e.g., a curly Swiss volume), the O'Schille distance itself may be a wrong measure: two points of space proximity on the curly side may actually be very far in flow.</p>
<h3>PA and KPCA: From linear to nuclear techniques</h3>
<p><strong>PCA (Main component analysis)</strong> There has been a detailed presentation, which is discussed briefly in the macro-level perspective of the backsliding. It needs to be reiterated that the PCA is equal to<strong>Using the MidSpecies</strong>I'm sorry. It's looking for the biggest difference. The data is kept.<strong>Global linear structure</strong>。</p>
<p>When the linear projection fails,<strong>nucleination linear downscaling (KPCA)<strong>It's introduced.&quot;Nuclear techniques&quot;(Kernel Trick). Other</strong>Intuition.</strong>Yes: The indissociable data of low dimensions is mapped into the high (even infinity) space, making it linear.</p>
<p>Yes.<strong>Form</strong>Go, by solve. &#36;\left(\sum_{i=1}^m z_i z_i^\mathrm{T}\right) W = \lambda W&#36; And use nuclear functions in calculations &#36;\kappa(x_i, x_j)&#36; Replaces the direct internalization calculation. Although mapping the data to higher levels seems anti-intuitive, it is introduced for the PAA<strong>Non-linear processing capacity</strong>This is a bridge between traditional statistical methods and fluent learning.</p>
<h2>Fluid learning and probability mapping model (modern downscaling core)</h2>
<p>When data is distributed on a curve, yes.<strong>Manifold Learning</strong>I'm sorry. It is concerned that:<strong>Maintain local neighbourhood structures and weaken remote global distances.</strong> As to what flow is per se, it is discussed in greater detail in the troplasm.</p>
<h3>t-SNE: from distance to probability distribution</h3>
<p><strong>t-SNE (t-Distributed Stochastic Neighbor Embedding)</strong> It is a data visualization technique that has a high impact on the night before the age of deep learning. It is no longer committed to the rigid constraint of “staying distance”, but instead to “maintaining probability distribution”.</p>
<h4>Technical intuition and mathematical forms</h4>
<p><strong>Probability of neighbours in the Gave space (Gose distribution)</strong>: In the high space, instead of using direct distance, ask: &#36;x_j&#36; Yes, it is. &#36;x_i&#36; What is the probability of a neighbor? We use<strong>Goss distribution</strong>To define the probability of this condition. &#36;p_{j|i}&#36;
&#36;&#36;p_{j|i} = \frac{\exp(-|x_i - x_j|^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-|x_i - x_k|^2 / 2\sigma_i^2)}&#36;&#36;
Watch this one. &#36;\sigma_i&#36; Calculated separately for each point, by argument <strong>Perplexity</strong> Decision.</p>
<p><strong>Quarter probability (t-distribution) for low-dimensional space</strong>: In low space, we're looking for a spot &#36;y_i, y_j&#36;I'm sorry. To solve it.<strong>Crowding Problems</strong>,t-SNE uses a freedom of 1 <strong>t-distribution (Cossi distribution)</strong>
&#36;&#36;q_{ij} = \frac{(1 + |y_i - y_j|^2)^{-1&#125;&#125;{\sum_{k \neq l} (1 + |y_k - y_l|^2)^{-1&#125;&#125;&#36;&#36;</p>
<p><strong>Why the t-distribution?</strong> Because the t-distribution is...&quot;Rewind&quot;Yeah. The same probability value (similarity) is obtained for the distribution of Goss, t-situation requirement point/point distance<strong>Farr</strong>I'm sorry. This forced the data cluster that was originally crowded together in the high space, and was forced to be held in the low space.&quot;Blow it up!&quot;And form a clear separation cluster.</p>
<p><strong>Loss function: KL diffusion</strong>
&#36;&#36;C = KL(P||Q) = \sum_i \sum_j p_{ij} \log \frac{p_{ij&#125;&#125;{q_{ij&#125;&#125;&#36;&#36;
From<strong>Gradient Analysis</strong>See: KL is asymmetrical. If &#36;p_{ij}&#36; It's big, but... &#36;q_{ij}&#36; Small (low-dimensional separation) and heavy punishment. On the contrary, if &#36;p_{ij}&#36; It's small. &#36;q_{ij}&#36; Very, very small punishment. Eventually.<strong>Result</strong>Yes: t-SNE is extremely good at it.<strong>Retain local structures</strong>(Consisting neighbours) but little of the overall structure (the distance between the cluster and the cluster is usually meaningless).</p>
<p>It needs to be explained in advance that the efficiency of t-SNE is not high; the speed bottlenecks in large-scale data are also one of the reasons for the subsequent introduction of UMAP.</p>
<p><strong>Perplexity (disturbation)</strong>: This is the main parameter of t-SNE, which is commonly taken between 5 and 50, which can be understood as&quot;Estimated number of neighbours&quot;I'm sorry. If set too small, the data will break into millions of small groups to make noise; if set too large, local details will be ignored and the result will become more and more like the PCA.</p>
<h3>UMAP: Success in data analysis</h3>
<p><strong>UMAP (Uniform Manifold Approximation and Projection)</strong> is the current SOTA down-the-dimensional algorithm. It introduces a rigorous Lehman geometry and algebraic purge based on t-SNE, which solves the problem of t-SNE slow and loss of global structures.</p>
<h4>Technical intuition: even distribution in flow shape</h4>
<p>UMAP is based on one assumption:<strong>Data are evenly distributed in a Lehman-like shape.</strong> If the data appear to be unevenly distributed in real space (some are secret, others are alien), it's because we use it to observe.&quot;Rulers&quot;(Oxley distance) It's wrong.</p>
<p><strong>Measurement of self-adaptation (Liman measure)</strong>: UMAP for each point &#36;x_i&#36; Defines a partial Lehman measure. In the data-slipped area, UMAP would&quot;Long&quot;Distance; in a dense area, will&quot;Shorten&quot;Distance. This is achieved through k-neighborhood and a weighted kNN figure is constructed.</p>
<p><strong>Fuzzy Simplifical Complex</strong>: By the above measure, the UMAP converts the data into a polo structure (simply double).</p>
<p><strong>Optimizing target: binary Cross-Entropy</strong>:t-SNE uses KL only, only focuses on&quot;Close the neighbors.&quot;I'm sorry. UMAP uses cross-breathing:
&#36;&#36;CE = \sum p_{ij} \log(\frac{p_{ij&#125;&#125;{q_{ij&#125;&#125;) + \sum (1-p_{ij}) \log(\frac{1-p_{ij&#125;&#125;{1-q_{ij&#125;&#125;)&#36;&#36;
of which<strong>First</strong>Like t-SNE, generating gravity, pulling the neighbors. The second one concerns &#36;(1-p_{ij})&#36;It's punishment.&quot;Non-neighbors are being drawn in.&quot;And the situation, it's a situation.<strong>Global repulsion.</strong>I'm sorry. Eventually.<strong>Result</strong>Yes: UMAP keeps the cluster tight and forces<strong>Keep the balance between the clusters right.</strong>, thus retaining more global structures.</p>
<p>UMAP has more parameters than t-SNE. In addition to controlling local size, it is necessary to control the degree of compactness embedded; it seeks to retain a part of the global structure and can also demonstrate a certain degree of spatial expansion.</p>
<p><strong><code>n_neighbors</code></strong>: Control the size of local maps. Small values (e.g. 5) capture high frequency details; large values (e.g. 200) capture global profiles (like PCA).</p>
<p><strong><code>min_dist</code></strong>: Control the compactness of low-dimensional embedded. Small values (0.1) allow overlap and are suitable for cluster analysis; large values (0.8) force points are separated and suitable for displaying a pedestal structure.</p>
<h2>Neural network and modern extension</h2>
<h3>Autoencoder (AE): Non-linear compression</h3>
<p>AE and its derivative model are discussed in detail in the chapter on the self-codifier. Put it in the light of the downside, AE is<strong>Parameterisation</strong>The downside. It's the main one.<strong>Difference</strong>It's: the flow method is non-parametric, which gives only coordinates and cannot process new data directly; whereas AE learns a function &#36;f(x)&#36;, can process new samples at any time.</p>
<p>Normal AEs usually have less of a de-dimensional effect than a specialized visualization method. For the t-SNE and UMAP scenarios, which often compress space to 2-3D, AE is more suitable to be reduced to several dozen or hundreds of dimensions as a feature pre-treatment first. (b) The features after extraction are then given downstream tasks; for example, <strong>VAE (Variational AE)</strong> And its variants are still commonly used in generating models (Stable Disfusion) and character decoupling.</p>
<h3>Other important models in modern science (notable)</h3>
<p>In bioinformatics (especially single-cell sequencing) and computer visual studies, in addition to UMAP, two models are of interest:</p>
<p><strong>PHATE (Potential of Heat-diffusion for Affinity-based Transition Embedding)</strong>: of which<strong>Intuition.</strong>Use&quot;The heat spreads.&quot;Process to simulate the probability of a transfer between data points. Main<strong>Advantages</strong>It is: it is good at retaining data in ** trajectory** and branch structures (e.g. stem cell separation processes) and is better suited than UMAP to demonstrate the continuous evolution of data.</p>
<p><strong>PacMAP (Pairwise Controlled Manifold Approximation)</strong>: of which<strong>Intuition.</strong>It's by designing special features.&quot;Medium Distance&quot;Point-to-point, visible balance of local gravity and global repulsive force. Current<strong>Status</strong>Yes: It is considered a strong competitor for t-SNE and UMAP, usually more robust in retaining the global structure than UMAP and less sensitive to parameters.</p>
<h2>Metric Learning</h2>
<p>Finally, back to the original purpose of the decline. We are often reduced because the distance of the O'Delk is not working in the high space.<strong>Measuring learning</strong>It's a reverse thought:<strong>Instead of mapping data into low-dimensional space to adapt to the Oxygen distance, you should learn a new distance measurement function. &#36;d(x_i, x_j)&#36;。</strong></p>
<p>Yes.<strong>Ma's distance from study</strong>In this context: Learning a matrix &#36;M&#36;♪ Make the distance ♪ &#36;d(x, y) = \sqrt{(x-y)^T M (x-y)}&#36; The similarities in data are best reflected.</p>
<p>Yes.<strong>Syamese Networks</strong> On the other hand, this is the mainstream of modern depth measurement. The mapping of two inputs into the characteristic space through the nervous network directly optimizes the distance between the characteristic vectors (e.g. Triplet Los), bringing the same type of sample closer and different types of sample far away.</p>
<h2>A summary of other common dimensions</h2>
<p>Beyond the main lines ahead, there are a number of downscaling methods that are common in the project, many of which are not commonly used. Only a brief summary is provided here, highlighting their relationship to the front-line approach.</p>
<p><strong>Linear approach</strong>: LDA (linear determination analysis) is the supervised version of the PCA - to find a source when category labels are available&quot;Maximum inter-group dispersion, minimum intra-group dispersion&quot;The projection direction is often used for the downscaling before classification; the factor analysis (FA) assumes that observation variables are generated by a few public factors plus special factors, with greater emphasis on the interpretation of the structure rather than on compression.</p>
<p><strong>Non-linear approach to retaining local structures</strong>Local linear insulation (LLE) assumes that each point can be reformed by a point linearity within the adjacent area and that the reconstructive weights remain unchanged, while the La Plas characterization (LE) allows the nearest point to be as close as possible through the t-SNE, UMAP, which is the same as the t-SNE, which is described above.&quot;Retain local structures&quot;One family, only earlier and less frequent. SNE is the precursor of t-SNE, which also converts similarity to a condition probability, and t-SNE is just using a rear end t distribution to alleviate congestion.</p>
<p><strong>Matrix decomposition perspective</strong>LSA, NMF and low-level approximation see the down dimension as a matrix decompose. Potential semantic analysis (LSA) is a low approximation of SVD for word-document matrix, often used in text subject space; the non-negative matrix decomposition (NMF) requires both factors to be non-negative, and therefore output is often interpreted as&quot;Partial additions and additions&quot;(e.g. theme, combination of components); low approximation is itself the search for the low matrix closest to the original matrix, with the exception of the PCA/SVD. The objectives of ICA (independent component analysis) are different - not to remove relevance but to find statistically independent elements, often used in blind source separation missions such as signal separation.</p>
<p><strong>Nuclearization and migration learning</strong>: nucleination linear downscaling (KPCA, KLDA) introduces nuclear techniques into linear methods, and KPCA has been described earlier, where KLDA is the corresponding monitoring version; and migration learning downscaling (TCA) is oriented towards cross-domain scenarios, mapping source and target domains into the same common subspace, minimizing differences in area distribution.</p>
<p><strong>A reminder.</strong>: Degraded, while providing computing and visualization facilities, often<strong>Damage error analysis and model interpretability</strong>The characteristics of the projection are no longer original, the meaning of the business and subsequent interpretation are discounted and the loss is irretrievable in the vast majority of cases. The choice of a model is worth confirming whether these two elements are acceptable.</p>
<h2>Summary: How?</h2>
<table>
<thead>
<tr>
<th align="left">Methodology</th>
<th align="left">Core Math Thought</th>
<th align="left">Reservations to global structures</th>
<th align="left">Reservations to local clusters</th>
<th align="left">Suggest scene</th>
</tr>
</thead>
<tbody><tr>
<td align="left"><strong>PCA</strong></td>
<td align="left">Disaggregation of the synonyms</td>
<td align="left">[Extraordinary] (Excellently)</td>
<td align="left">[Low] (distant)</td>
<td align="left">Data preprocessing, baseline testing, linear data</td>
</tr>
<tr>
<td align="left"><strong>MDS</strong></td>
<td align="left">Distance Matrix Reconstruct</td>
<td align="left">[Extraordinary]</td>
<td align="left">[Low]</td>
<td align="left">When strictly keeping distance matrix</td>
</tr>
<tr>
<td align="left"><strong>t-SNE</strong></td>
<td align="left">Probability distribution (KL diffusion)</td>
<td align="left">[Low] (near all)</td>
<td align="left">[Extraordinary] (Excellently)</td>
<td align="left">Explore data analysis, emphasis on cluster separation degrees</td>
</tr>
<tr>
<td align="left"><strong>UMAP</strong></td>
<td align="left">Popup Pure Double + Cross-Cross</td>
<td align="left">[High] (Good)</td>
<td align="left">[Extraordinary] (Excellently)</td>
<td align="left"><strong>Current Preferred</strong>, take into account global and local, large data sets</td>
</tr>
<tr>
<td align="left"><strong>PHATE</strong></td>
<td align="left">Hot-spreading information distance</td>
<td align="left">[Extraordinary]</td>
<td align="left">[High]</td>
<td align="left">Data with continuous evolutionary trajectory (e.g. time series, biodevelopment)</td>
</tr>
<tr>
<td align="left"><strong>AE</strong></td>
<td align="left">Neural network re-construction error.</td>
<td align="left">[High]</td>
<td align="left">[Low]</td>
<td align="left">When character extractor is required for downstream tasks</td>
</tr>
</tbody></table>
