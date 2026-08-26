---
title: 'Probabilistic Graphical Models: From Bayesian Networks to LDA'
title_zh: 概率图模型：从贝叶斯网络到 LDA
date: 2026-02-09 12:00:00 +0800
categories:
- Machine Learning
- Probabilistic Graphical Models
tags:
- Graphical Models
author: Hyacehila
mathjax: true
hidden: true
excerpt: Connects Bayesian networks, hidden Markov models, and Markov random fields to LDA, showing how graphical structure,
  latent variables, conjugate priors, and approximate inference form a unified probabilistic modeling language.
description: Connects Bayesian networks, hidden Markov models, and Markov random fields to LDA, showing how graphical structure,
  latent variables, conjugate priors, and approximate inference form a unified probabilistic modeling language.
excerpt_zh: 从贝叶斯网络、隐马尔可夫模型与马尔可夫随机场出发，理解有向、动态和无向图如何分解联合分布，并以 LDA 展示潜变量、共轭先验与近似推断如何共同生成文本主题。
permalink: /blog/2026/02/09/belief-network-learning/
lang: en
translation_key: 2026-02-09-belief-network-learning
translation_status: machine
translation_source_hash: 0c327c412bae114ed0d9ee0916b0ef71dede5d42135dc337d88f546f1ccb9031
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<h2>Introduction: Probability structure expressed in graphics</h2>
<p>When hundreds of variables are reached, the direct modelling of the combined probability distribution &#36;P(X_1, X_2, \dots, X_n)&#36; Often not feasible: the number of parameters increases exponentially with the number of variables.<strong>Probability chart model</strong>(Probabilistic Graphial Models, PGM) uses graphics to express the structure of the variables, quantify the intensity of dependence with probabilities, and thus disassembly the joint distribution of the high dimensions into a local, interpretable part.</p>
<p>If you want to move from a condition-based assumption in a classification mission to the subject, you can recall that<a href="/en/blog/2024/03/28/machine-learning-introduction-supervised-learning/">Machine Learning Introduction and Monitoring Learning: The Bayesian Catalogue</a>I'm sorry. The PARK Soo-Bayes and PARK Soo-Bayes to the Bayesian network are the same path that gradually relaxes the assumption of identity independence.</p>
<p>This paper places four representative models in the same framework:</p>
<table>
<thead>
<tr>
<th>Model</th>
<th>Figure structure</th>
<th>Main Process Objects</th>
<th>Expression of condition independence</th>
</tr>
</thead>
<tbody><tr>
<td>Bayesian Network (BN)</td>
<td>A-DAG-DAG-DAG-DIG-DIG-DIG-DIG-DIG-DIG-DIG-DIG-DIG-DIG-DIG-DIG-DIG-DIG-DIG-DIG-DIG-DIG-DIG-DIG-DIG-DIG-DIG-DIG-DIG-DIG-DIG-DIG-DIG-DIG-DIG-DIG-DIG-DIG-DIG-DIG-DIG-DIG-DIG-DIG-D-D-DIG-D-D-DIG-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D-D</td>
<td>Static variable depends on</td>
<td>Local independence after the given parent node</td>
</tr>
<tr>
<td>The Cain Markov Model (HMM)</td>
<td>A directional map over time</td>
<td>Sequence and Hiding</td>
<td>Current status depends only on the previous status</td>
</tr>
<tr>
<td>Markov Airport with MRF</td>
<td>No Flow</td>
<td>Symmetrical spatial or neighbourhood relationships</td>
<td>Conditional independence of the figure after separation</td>
</tr>
<tr>
<td>Potential Dilekre Allocation (LDA)</td>
<td>Prospective Generate with Plate Figure</td>
<td>Document and potential theme</td>
<td>When assigned to the given theme, the word is only based on the corresponding theme</td>
</tr>
</tbody></table>
<p>The four share the idea of “the structure of the map determines how to break the probability distribution”, but the edge of the map is not the same semantic. In particular,<strong>One of the DAGs is to indicate to the side first the direction of reliance and fragmentation in modelling; it can only be interpreted as causality if structural cause-effect models, intervention syntax and sufficient area assumptions are included.</strong></p>
<h2>There's a static map: Bayesian network.</h2>
<h3>Structures and Factor Decompose</h3>
<p>Bayesian Network, also known as the Belif Network, is a qualitative graphic structure. &#36;\mathcal{G}&#36; and quantitative parameters &#36;\Theta&#36; Composition. Here's a picture.<strong>There's a loopless map.</strong>(Directed Acycric Graph, DAG): Each node &#36;X_i&#36; It's random. &#36;X_j \to X_i&#36; Organisation &#36;X_j&#36; Yes. &#36;X_i&#36; The parent node.</p>
<p>At the heart of it is the assumption.<strong>Local Markov Nature</strong>: given parent node &#36;Pa(X_i)&#36; After, Node &#36;X_i&#36; Independent of all non-descendant node conditions.</p>
<p>Thus, joint distribution can be written as a product of a local condition distribution:</p>
<p>&#36;&#36;
P(X_1, \dots, X_n) = \prod_{i=1}^{n} P(X_i \mid Pa(X_i)).
&#36;&#36;</p>
<p>This factor breakdown translates the unspecified high-dimensional joint distribution into a probability table for several conditions (Conditional Production Table, CPT) or a condition density function.</p>
<p><img src="/assets/images/machine-learning-notes/ml-bayesian-network-example.png" alt="Bates Network."></p>
<h3>Example of a burglar alarm</h3>
<p>The classic alarm network can be described in five variables: earthquakes. &#36;E&#36;Theft &#36;B&#36;The alarm. &#36;A&#36;John called. &#36;J&#36; Call Mary. &#36;M&#36;。</p>
<pre><code class="language-mermaid">graph TD
    E[Earthquake] --&gt; A[Alarm]
    B[Burglary] --&gt; A
    A --&gt; J[JohnCalls]
    A --&gt; M[MaryCalls]
</code></pre>
<p>The network is distributed as follows:</p>
<p>&#36;&#36;
P(E, B, A, J, M) = P(E)P(B)P(A \mid E, B)P(J \mid A)P(M \mid A).
&#36;&#36;</p>
<p>If you specify five sets of local probability, you can paint the entire system. More importantly, the figure also tells us which variables can be ignored after the given evidence.</p>
<h3>D-Definition and conditions independence</h3>
<p><strong>D-Segment</strong>(D-Separation) is the graphical rule for the independence of the conditions judged in DAG. It is structured around three basic structures:</p>
<p><img src="/assets/images/machine-learning-notes/ml-bayesian-network-dependencies.png" alt="The Beyers web relies on structural indications."></p>
<h4>It's okay.</h4>
<p>&#36;&#36;X \to Y \to Z&#36;&#36;</p>
<p>Not given &#36;Y&#36; , the information can be passed along the path; given &#36;Y&#36; Then the path was blocked, so... &#36;X \perp Z \mid Y&#36;。</p>
<h4>Branch</h4>
<p>&#36;&#36;X \leftarrow Y \to Z&#36;&#36;</p>
<p>&#36;Y&#36; Yes. &#36;X&#36; and &#36;Z&#36; Common cause. Unobserved &#36;Y&#36; , which are usually relevant; control &#36;Y&#36; The blogger says:&#36;X \perp Z \mid Y&#36;。</p>
<h4>Convergence</h4>
<p>&#36;&#36;X \to Y \leftarrow Z&#36;&#36;</p>
<p>No observations. &#36;Y&#36; The first is the "Standards of the Earth":&#36;X&#36; and &#36;Z&#36; Marginal independence; once observed &#36;Y&#36;, the path is activated, and the two reasons are relevant. This phenomenon is called<strong>Empirical</strong>(Explaining Away): For example, if an earthquake is found to have occurred, the need to alert the police is reduced by theft.</p>
<h3>Learning and extrapolation</h3>
<p>The Bayesian network has two levels of learning:</p>
<ol>
<li><p><strong>Parameter Learning</strong>: An estimate of the probability of conditions at each node when the chart is known to be structured. The discrete variable can be estimated to be the largest possible number:</p>
<p>&#36;&#36;
\theta_{ijk}^{MLE}=\frac{N_{ijk&#125;&#125;{\sum_k N_{ijk&#125;&#125;.
&#36;&#36;</p>
<p>Dirichlet is used to smooth first when data is thin:</p>
<p>&#36;&#36;
\theta_{ijk}^{Bayes}=\frac{N_{ijk}+\alpha_{ijk&#125;&#125;{\sum_k(N_{ijk}+\alpha_{ijk})}.
&#36;&#36;</p>
</li>
<li><p><strong>Structural learning</strong>: Finds a suitable DAG from the data when the structure of the chart is unknown. This is a NH-Hard problem, and the common routes include:</p>
<ul>
<li>(a) A restraint-based approach: the use of independent testing of the conditions for the construction of the skeleton, such as the PC algorithm;</li>
<li>(a) A rating-based method: use BIC, BDeu and other rating functions to support mountain climbing or Tabu Search searches;</li>
<li>Mixing method: The pool of candidates is reduced by binding method, followed by statutory scoring, e.g. MMHC.</li>
</ul>
</li>
</ol>
<p>At the inference stage, the target is usually to calculate the probability of searching under evidentiary conditions, for example, &#36;P(\mathbf{Q}=\mathbf{q}\mid\mathbf{E}=\mathbf{e})&#36;I'm sorry. Accurate solvers can be found when the network is smaller; similar methods such as Gibbs sampling are commonly used when the network is dense or larger.</p>
<h2>There's a dynamic map: The Invisible Markov model</h2>
<p>The static Bayesian network has a set of variables discussed at the same time, and many data are naturally time-series.<strong>The Cain Markov model.</strong>Hidden Markov Model, HMM is the most classic and restricted form of the dynamic Bayesian network: it produces observation sequences using a hidden state chain.</p>
<h3>Model definition</h3>
<p>HMM contains two-layer random processes:</p>
<ol>
<li><strong>Invisible status sequence</strong> &#36;Q={q_1,q_2,\dots,q_T}&#36;: the true state of the system, which is not normally directly observed;</li>
<li><strong>Observation sequences</strong> &#36;O={o_1,o_2,\dots,o_T}&#36;: Data available at every moment.</li>
</ol>
<p>A discrete HMM usually says &#36;\lambda=(N,M,A,B,\pi)&#36;：</p>
<ul>
<li><p>Status set &#36;S={s_1,\dots,s_N}&#36;；</p>
</li>
<li><p>Observation cluster &#36;V={v_1,\dots,v_M}&#36;；</p>
</li>
<li><p>State Transfer Matrix &#36;A=[a_{ij}]&#36;, of which</p>
<p>&#36;&#36;a_{ij}=P(q_{t+1}=s_j\mid q_t=s_i);&#36;&#36;</p>
</li>
<li><p>Launch probability matrix &#36;B=[b_j(k)]&#36;, of which</p>
<p>&#36;&#36;b_j(k)=P(o_t=v_k\mid q_t=s_j);&#36;&#36;</p>
</li>
<li><p>Distribution of Initial Status &#36;\pi=[\pi_i]&#36;, of which &#36;\pi_i=P(q_1=s_i)&#36;。</p>
</li>
</ul>
<p>It relies on two key assumptions:</p>
<ol>
<li><p><strong>Zhith Markov's hypothesis</strong>The current state of secrecy depends only on the state of secrecy of the previous moment.</p>
<p>&#36;&#36;
P(q_t\mid q_{t-1},o_{t-1},\dots,q_1,o_1)=P(q_t\mid q_{t-1});
&#36;&#36;</p>
</li>
<li><p><strong>Observation independence assumptions</strong>: Current observations rely only on the current state of concealment,</p>
<p>&#36;&#36;
P(o_t\mid q_T,o_T,\dots,q_1,o_1)=P(o_t\mid q_t).
&#36;&#36;</p>
</li>
</ol>
<h3>Three core issues</h3>
<h4>Probability calculation: Forward algorithm</h4>
<p>Give model and observation sequences, calculation &#36;P(O\mid\lambda)&#36;I'm sorry. The complexity of all hidden sequences directly enumerated is &#36;O(N^T\cdot T)&#36;, the forward algorithm reduces it to &#36;O(N^2\cdot T)&#36;。</p>
<p>Define forward probability:</p>
<p>&#36;&#36;
\alpha_t(i)=P(o_1,\dots,o_t,q_t=s_i\mid\lambda).
&#36;&#36;</p>
<p>Gradually:</p>
<p>&#36;&#36;
\alpha_1(i)=\pi_i b_i(o_1),
&#36;&#36;</p>
<p>&#36;&#36;
\alpha_{t+1}(j)=\left[\sum_{i=1}^N\alpha_t(i)a_{ij}\right]b_j(o_{t+1}),
&#36;&#36;</p>
<p>&#36;&#36;
P(O\mid\lambda)=\sum_{i=1}^N\alpha_T(i).
&#36;&#36;</p>
<h4>Decoding: Witterby algorithm</h4>
<p>Give a view sequence to find the most likely hidden path:</p>
<p>&#36;&#36;
Q^*=\arg\max_Q P(Q\mid O,\lambda).
&#36;&#36;</p>
<p>The Witby algorithm retains the best path to each state. You're the one who's gonna get you. &#36;\delta_t(i)&#36; Means at the moment &#36;t&#36; in &#36;s_i&#36; , the maximum path probability is:</p>
<p>&#36;&#36;
\delta_1(i)=\pi_i b_i(o_1),
&#36;&#36;</p>
<p>&#36;&#36;
\delta_t(j)=\max_{1\le i\le N}[\delta_{t-1}(i)a_{ij}]b_j(o_t).
&#36;&#36;</p>
<p>And record the best forwards of every step. &#36;\psi_t(j)&#36;, the whole state sequence is restored by retroactive return after the maximum value of the endpoint is obtained.</p>
<h4>Learning: Baum-Welch Algorithm</h4>
<p>If only the observation sequence is not indicated in the hidden state, it needs to be estimated &#36;A&#36;、&#36;B&#36; and &#36;\pi&#36;I'm sorry. Because the model contains hidden variables, it is not possible to complete a normal, very similar estimate directly; special cases of HMM using the EM algorithms -<strong>Baum-Welch algorithm</strong>。</p>
<p>E step calculation:</p>
<ul>
<li>&#36;\xi_t(i,j)&#36;: Time &#36;t&#36; As Status &#36;i&#36; and &#36;t+1&#36; As Status &#36;j&#36; (b) the probability of a post-test;</li>
<li>&#36;\gamma_t(i)=\sum_j\xi_t(i,j)&#36;: Time &#36;t&#36; Status &#36;i&#36; - The probability of a posteriori.</li>
</ul>
<p>M step by step to update transfer probability and launch probability in the expected number:</p>
<p>&#36;&#36;
\hat{a}<em>{ij}=\frac{\sum</em>{t=1}^{T-1}\xi_t(i,j)}{\sum_{t=1}^{T-1}\gamma_t(i)},
&#36;&#36;</p>
<p>&#36;&#36;
\hat{b}<em>j(k)=\frac{\sum</em>{t=1,o_t=v_k}^{T}\gamma_t(j)}{\sum_{t=1}^{T}\gamma_t(j)}.
&#36;&#36;</p>
<h3>Example: word type</h3>
<p>In word type labels, word series are observations and word type labels are hidden. The probability of a shift between word-types and word-types is estimated at the time of training; the most likely word-type sequences are recovered from the sentence using the Witby algorithm at the time of projection. While many modern missions have used RNN, LSTM or Transformer, HMM is an important starting point for understanding serial probabilities modelling, dynamic planning and learning about hidden variables.</p>
<h2>No direction: Markov follows the airport</h2>
<p>When the relationship is symmetrical and does not have a clear causal direction, the no-go map is more natural.<strong>Markov follows the airport.</strong>(Markov Random Field, MRF) is particularly appropriate for expressing local correlations between image pixels, space units or network neighbours.</p>
<h3>Figure separation and global marcroft</h3>
<p>MRF uses no-go map &#36;G=(V,E)&#36;I'm sorry. Gathering Nodes &#36;A&#36;、&#36;B&#36;、&#36;C&#36;, if &#36;C&#36; Blocked the drawing from &#36;A&#36; Present. &#36;B&#36; All the paths, then:</p>
<p>&#36;&#36;
A\perp B\mid C.
&#36;&#36;</p>
<p>It's called<strong>The Global Marcov Nature</strong>I'm sorry. Unlike D-division of DAG, there is no need to address the observation of V-type structures in the unwinding map; the separation of the diagram itself gives conditional independence.</p>
<h3>Group, Fist and Gibbs Distribution</h3>
<p>No Flow by<strong>Corps</strong>(Clike) Expresss a partial interaction. If a strictly correct distribution meets the Marcov properties of the diagram, Hammersley-Clifford ensures that it can be written as a product of a large grouping function:</p>
<p>&#36;&#36;
P(X)=\frac{1}{Z}\prod_{C\in\mathcal{C&#125;&#125;\psi_C(x_C).
&#36;&#36;</p>
<p>Of which:</p>
<ul>
<li><p>&#36;\mathcal{C}&#36; It's a huge gathering.</p>
</li>
<li><p>&#36;\psi_C(x_C)\ge0&#36; is a dynamic function that measures the compatibility of the state of the variables within the group;</p>
</li>
<li><p>&#36;Z&#36; It's a sub-function, which is the function of the sub-function:</p>
<p>&#36;&#36;
Z=\sum_x\prod_{C\in\mathcal{C&#125;&#125;\psi_C(x_C).
&#36;&#36;</p>
</li>
</ul>
<p>If you order &#36;\psi_C(x_C)=\exp(-E_C(x_C))&#36;, and get energy form Gibbs distribution:</p>
<p>&#36;&#36;
P(X)=\frac{1}{Z}\exp(-E(x)).
&#36;&#36;</p>
<p>The lower energy state is more likely, which also connects probability-mapping models to statistical physics.</p>
<h3>Ising Models and Images Go Noise</h3>
<p>Ising Model is the simplest pair of MRFs. Make every node &#36;x_i\in{-1,+1}&#36;, whose energy function can be written as:</p>
<p>&#36;&#36;
E(x)=-\sum_{(i,j)\in E}J_{ij}x_ix_j-\sum_{i\in V}h_ix_i.
&#36;&#36;</p>
<p>The first encourages the alignment of values at adjacent nodes, the second expresses that individual nodes are affected by external information. When using a binary image to go to noise, you can order &#36;y_i&#36; For the noise pixels,&#36;x_i&#36; For recovery pixels:</p>
<p>&#36;&#36;
E(x,y)=-\beta\sum_{(i,j)\in E}x_ix_j-\eta\sum_{i\in V}x_iy_i.
&#36;&#36;</p>
<p>Minimize energy equivalence for the search for maximum back-probability (MAP) resolution: preserves the local smoothness of the image without detaching from the pixels observed.</p>
<h3>Insumption and Gibbs Sample</h3>
<p>The difficulty of MRF is the split function &#36;Z&#36; Usually you need to list all the variables. If there is one, &#36;N&#36; A binary variable, sum size is &#36;2^N&#36;, precise extrapolations quickly become unfeasible.</p>
<p>MCMC can avoid direct calculations. &#36;Z&#36;I'm sorry. In particular, in Gibbs samples, the full probability of a variable depends only on its neighbour:</p>
<p>&#36;&#36;
P(x_i\mid x_{-i})=P(x_i\mid x_{\text{neighbors&#125;&#125;)=\frac{\exp(-E(x_i,x_{\text{neighbors&#125;&#125;))}{\sum_{x_i&#39;\in Val(x_i)}\exp(-E(x_i&#39;,x_{\text{neighbors&#125;&#125;))}.
&#36;&#36;</p>
<p>The attribute is offset by the molecule and the denominator. When the actual sample is taken, the individual node is updated repeatedly from the random starting value, with a conditional distribution; after the chain has constricted, the sample can approximate the target distribution.</p>
<h2>Text Generation Diagram: LDA Theme Model</h2>
<p>The first three models show static dependence, time dependence and local inactivity respectively. They also lay down the tools needed to understand more complex models: to break into joint distributions by graphics, to express invisible structures by hidden variables, and to re-establish them by approximate inferences.<strong>Potential Delicré distribution</strong>(Latent Dirichlet Allocation, LDA) bringing these ideas to the text would naturally constitute the last stop of this probability map model.</p>
<h3>From wordbags to generating models</h3>
<p>The simplest way to process text data<strong>Wordbag model</strong>(Bag-of-Words, BoW). It ignores the words and syntax, and records only the words that appear in the document and the frequency of each. BoW can convert text to vectors without being able to explain directly the semantic structure behind the document. Intuitively, an article is organized around several themes, each of which tends to use a specific set of terms.</p>
<p><strong>Theme Model</strong>(Topic Mode) Writes this instinct into a generation process: the document selects the subject, the theme then the word. IDA is the classic model of it. It uses a flow map to indicate the generation relationship between variables, submersible variables to carry the theme structure in the document that cannot be directly observed, and a double plate compression to indicate the location of the entire document and the word.</p>
<h3>Beyes Perspective: Dirichlet Co-examining</h3>
<p>Before entering the LDA structure, the probability component used by it needs to be understood. When a single subject or word is sampled in a single location, a classification distribution (Categoric Distribution) is used; multi-scale distribution (Multilingual Distribution) is used to read the number of multiple samples together. The probability vector behind both can be used. <strong>Dirichlet Distribution</strong>As a priori.</p>
<p>In PLSA (Probabiliistic Late Security Analysis), the subject proportion of the document is usually used as an estimation parameter. LDA uses Beyers Modelling: Documentation - Theme Distribution &#36;\theta_d&#36; and theme-word distribution &#36;\phi_k&#36; It's a random variable, and it's subject to Dirichlet's first test.</p>
<ul>
<li><strong>Classification/multiple distribution</strong>Describe the selections made in a discrete group, and the number of counts obtained after multiple selections;</li>
<li><strong>Dirichlet Distribution</strong>Describe the probabilistic vectors corresponding to these discrete categories;</li>
<li>Dirichlet is classified/multiple distribution<strong>Co-protest</strong>The posterioris therefore remain within the Dirichlet distribution group.</li>
</ul>
<p>Co-benefits do not automatically make all late counting simple, but it allows us to decipher parts of the continuum, which makes it possible to fold Gibbs later.</p>
<h3>Generate Process and Plate Notation</h3>
<p>LDA describes the language library as a layered generation process from the subject scale to the specific vocabulary.</p>
<h4>Core Variables</h4>
<ul>
<li><strong>Document</strong>: Language library contains &#36;D&#36; Part of the document, No. &#36;d&#36; - Yes, I do. &#36;N_d&#36; (a) Words;</li>
<li><strong>Theme</strong>Other Organiser &#36;K&#36; Theme, word distribution for each theme &#36;\phi_k&#36; Define in Size &#36;V&#36; on the glossary;</li>
<li><strong>Document Theme Scale</strong>：&#36;\theta_d&#36; Representing documents &#36;d&#36; Yeah. &#36;K&#36; (a) The mix of the themes;</li>
<li><strong>Theme Assign</strong>：&#36;z_{d,n}&#36; Representing documents &#36;d&#36; Medium &#36;n&#36; (a) The theme of the choice of the word location;</li>
<li><strong>Observation term</strong>：&#36;w_{d,n}&#36; is the word actually observed at that location.</li>
</ul>
<h4>Figure structure</h4>
<pre><code class="language-mermaid">graph TD
    subgraph Plate_K [K Topics]
        beta((beta)) --&gt; phi((phi))
    end

    subgraph Plate_D [D Documents]
        alpha((alpha)) --&gt; theta((theta))
        subgraph Plate_N [N_d Words]
            theta --&gt; z((z))
            z --&gt; w((w))
            phi --&gt; w
        end
    end

    style w fill:#ddd,stroke:#333,stroke-width:2px
</code></pre>
<ul>
<li>&#36;w&#36; (a) is the actual word in the observation variable, the counterpart library;</li>
<li>&#36;z&#36;、&#36;\theta&#36; and &#36;\phi&#36; is the subvariant or unknown random amount to be extrapolated;</li>
<li>&#36;\alpha&#36; and &#36;\beta&#36; It's a pre-checked hyper-parameter to control Dirichlet;</li>
<li>Two layers of Plate indicate that the same generation steps are repeated for " for each document " and " for each word position in the document " , respectively.</li>
</ul>
<h4>Generate stories</h4>
<ol>
<li>For each theme &#36;k \in {1, \dots, K}&#36;From a priori &#36;\operatorname{Dir}(\boldsymbol{\beta})&#36; in sample theme - word distribution &#36;\phi_k&#36;。</li>
<li>For each document &#36;d \in {1, \dots, D}&#36;From a priori &#36;\operatorname{Dir}(\boldsymbol{\alpha})&#36; Medium Sample Document - Theme Distribution &#36;\theta_d&#36;。</li>
<li>Against Documents &#36;d&#36; , and then click the &#36;n&#36;：<ul>
<li>From &#36;\theta_d&#36; Medium Sample Theme Assign &#36;z_{d,n}&#36;；</li>
<li>Distribution of words from the subject &#36;\phi_{z_{d,n&#125;&#125;&#36; Medium sampled observation term &#36;w_{d,n}&#36;。</li>
</ul>
</li>
</ol>
<p>From the dependency of the conditions in the figure, the complete joint distribution can be broken down into:</p>
<p>&#36;&#36;
P(\mathbf{w}, \mathbf{z}, \boldsymbol{\theta}, \boldsymbol{\phi} \mid \boldsymbol{\alpha}, \boldsymbol{\beta})
= \prod_{k=1}^K P(\phi_k \mid \boldsymbol{\beta})
  \prod_{d=1}^D \left[
    P(\theta_d \mid \boldsymbol{\alpha})
    \prod_{n=1}^{N_d}
      P(z_{d,n} \mid \theta_d)
      P(w_{d,n} \mid \phi_{z_{d,n&#125;&#125;)
  \right].
&#36;&#36;</p>
<p>It is not necessary to write each of these conditions in the natural language description: the process of generation is already given together with the breakdown of the figure and factor.</p>
<h3>Insumption: folding Gibbs sample</h3>
<p>When training LDA, only words are actually observed &#36;\mathbf{w}&#36;I'm sorry. Target is to assign the theme by the language library back. &#36;\mathbf{z}&#36;, the document theme ratio &#36;\boldsymbol{\theta}&#36; and theme word distribution &#36;\boldsymbol{\phi}&#36;, which means a posteriori:</p>
<p>&#36;&#36;
P(\mathbf{z}, \boldsymbol{\theta}, \boldsymbol{\phi} \mid \mathbf{w}, \boldsymbol{\alpha}, \boldsymbol{\beta}).
&#36;&#36;</p>
<p>This posteriori sub-consolidation constant requires the sum of a large number of hidden variables or points, which cannot be calculated directly. Similar to the previous MRF, LDA usually relies on approximation. Using Dirichlet co-relationship, you can use continuous variables &#36;\boldsymbol{\theta}&#36; and &#36;\boldsymbol{\phi}&#36; Parsing fractions, assigning only to discrete themes &#36;\mathbf{z}&#36; Sample, therefore called<strong>Collapse Gibbs Sample</strong>（Collapsed Gibbs Sampling）。</p>
<h4>Sample formula</h4>
<p>To simplify marking, assuming &#36;\alpha&#36; and &#36;\beta&#36; is the symmetric parameter for the pre-test of Dirichlet. Assign theme to all other word positions, current word &#36;w_{d,n}&#36; Allocation to themes &#36;k&#36; The probability of a condition is:</p>
<p>&#36;&#36;
P(z_{d,n}=k \mid \mathbf{z}<em>{\neg(d,n)}, \mathbf{w}, \alpha, \beta)
\propto
\left(n</em>{d,k}^{\neg(d,n)}+\alpha\right)
\frac{n_{k,w_{d,n&#125;&#125;^{\neg(d,n)}+\beta}
{n_k^{\neg(d,n)}+V\beta}.
&#36;&#36;</p>
<p>Of which:</p>
<ul>
<li>&#36;n_{d,k}^{\neg(d,n)}&#36; is the document after excluding the current position &#36;d&#36; to the theme &#36;k&#36; the number of words;</li>
<li>&#36;n_{k,w_{d,n&#125;&#125;^{\neg(d,n)}&#36; It's the word after the current position is excluded. &#36;w_{d,n}&#36; Allocation to themes &#36;k&#36; Number of times;</li>
<li>&#36;n_k^{\neg(d,n)}&#36; is all allocated to the subject after excluding the current position &#36;k&#36; the number of words;</li>
<li>The first bias is a topic already common in the document, the second tends to generate the subject of the current word.</li>
</ul>
<p>The complete probability of the side of the document also contains the denominator &#36;N_d-1+K\alpha&#36;But it's all the subjects that are candidates &#36;k&#36; It's all the same, so it's in this positive pattern.</p>
<h4>Algorithms process and parameter restoration</h4>
<ol>
<li>A random initial theme is assigned to each word position in the language library.</li>
<li>Repeatedly through all word locations:<ul>
<li>Remove the theme assignment from the count of the current position;</li>
<li>(a) Calculate its weight as a matter of subject matter based on the probability of the above conditions;</li>
<li>Sample new themes with a centralized weight and update the count.</li>
</ul>
</li>
<li>The preheating samples that have not yet been pre-dumped are discarded and the following samples are used to estimate the thematic structure.</li>
</ol>
<p>When the sample is stabilized, the document - the theme distribution and the theme - the word distribution - can be restored by smooth count:</p>
<p>&#36;&#36;
\hat{\theta}<em>{d,k}=\frac{n</em>{d,k}+\alpha}{N_d+K\alpha},
\qquad
\hat{\phi}<em>{k,v}=\frac{n</em>{k,v}+\beta}{n_k+V\beta}.
&#36;&#36;</p>
<p>Thus, the result of LDA is not just a theme label for each document, but rather a mix of documents on multiple themes and a probability distribution of the vocabulary for each theme.</p>
<h2>Summary: from chart structure to generation model</h2>
<p>The unified idea of the probability map model is:<strong>Quantification of uncertainty by probability, using graphics</strong>。</p>
<ul>
<li><strong>Bayesian Network</strong>(a) The use of DAG as a static condition depends on the direction of decomposition with joint distribution and changes independent relationships through D-division of research evidence;</li>
<li><strong>HMM</strong>(a) The structure of the direction is extended over time, describing the generation and evolution of the sequence in a hidden state;</li>
<li><strong>MRF</strong>(a) a local interaction of symmetrical symmetry with no directional charts, groups and force functions;</li>
<li><strong>LDA</strong>A replicated orientation generation structure with Plate combines submersible variables, co-prospect and approximate extrapolations into text theme modelling.</li>
</ul>
<p>The four models are not an upgraded version of each other, but are extended on different data structures in the same model language: static variables, time series, spatial domains and text collections, where conditions can be determined independently, and the joint distribution can be broken down and extrapolated algorithms selected.</p>
<p>And the route to LDA is also a complete closed ring. After reading the correspondence between maps, joint distribution and post-pregnation, and in the face of a new probability model, it is not necessary to remember only the algorithm name, but to ask three more stable questions: What variables can be observed, what structures are hidden, and what dependencies are generated by data. LDA is fit to end it because it is not just another subject algorithm, but rather centralizes the core components of the Probability Map model into the same example.</p>
<h3>Extending reading</h3>
<ul>
<li><a href="/en/blog/2024/03/28/machine-learning-introduction-supervised-learning/">Machine Learning Introduction and Monitoring Learning: The Bayesian Catalogue</a>: understand why the chart model expresses the dependency of attributes from plain/semi-supra.</li>
<li><a href="/en/blog/2026/02/19/kalman-filter/">Karman filter family: KF, EKF, UKF and EnKF</a>: Recursive estimation and filtering routes in continuous status space.</li>
</ul>
