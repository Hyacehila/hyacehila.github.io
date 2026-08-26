---
title: 'From LSH to K-Center Greedy: Semantic Embeddings for Deduplication, Cleaning, and Sample Selection'
title_zh: 从 LSH 到 K-center-greedy：语义嵌入如何做数据去重、清洗与样本筛选
date: 2026-03-19 23:20:00 +0800
categories:
- Foundation Models
- Model Mechanics
tags:
- Embeddings
- Data Curation
author: Hyacehila
mathjax: true
excerpt: Semantic embeddings are not only for retrieval. LSH filtering, Faiss deduplication, and K-center greedy sampling
  all use the same representation space for redundancy and coverage.
description: Semantic embeddings are not only for retrieval. LSH filtering, Faiss deduplication, and K-center greedy sampling
  all use the same representation space for redundancy and coverage.
excerpt_zh: 语义嵌入不只是用来做检索。LSH 初筛、Faiss 语义去重和 K-center-greedy 样本筛选，都在利用同一个表示空间处理冗余、覆盖与召回问题。
permalink: /blog/2026/03/19/from-lsh-to-kcenter-greedy-semantic-embedding-deduplication/
lang: en
translation_key: 2026-03-19-from-lsh-to-kcenter-greedy-semantic-embedding-deduplication
translation_status: machine
translation_source_hash: a0b5e1a8d46c39ebac4c81df149561d646ee45a18bfd47f81fb9a7fef08ddff3
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<p>If there's only a bunch of embedding vectors, without any other a priori information, many downstream tasks can be rewritten as the same problem:<strong>How to compare points with points in high space.</strong> The focus is on finding “too close”, searching is looking for “nearest point to search”, clustering is looking for “nature formation” and sample screening is looking for “best point to cover the whole spectrum”, and recommending “the best point to match the user vector”. Instead of simply listing tools, the article is based on the most basic mathematical objects, explaining what these methods do in vector space.</p>
<p>The questions in this article can also be addressed<a href="/en/blog/2024/08/15/llm-lifecycle-overview/">LLM Life Cycle Overview: From Data, Pre-Training to Decoding and Deployment</a>、<a href="/en/blog/2024/09/20/prompt-engineering-and-in-context-learning/">Indications for engineering and context learning: from basic design to technical mapping and scenario practice</a>How the concept of a relatively close read together is developed in different contexts.</p>
<h2>Start with a vector: likeness, distance and fusion</h2>
<p>Sets a text that gets vectors after embeding</p>
<p>&#36;&#36;
x_i \in \mathbb{R}^d
&#36;&#36;</p>
<p>The first step is to do it. <code>L2</code> Normalization:</p>
<p>&#36;&#36;
\hat{x}_i = \frac{x_i}{|x_i|_2}
&#36;&#36;</p>
<p>After integration, the cosine similarity of the two texts can be written directly into a fraction:</p>
<p>&#36;&#36;
s(i,j)=\cos(\theta_{ij})=\hat{x}_i^\top \hat{x}_j
&#36;&#36;</p>
<p>If you prefer distance, you can use it directly. For a vector, the two are almost identical:</p>
<p>&#36;&#36;
|\hat{x}_i-\hat{x}_j|_2^2 = 2 - 2\hat{x}_i^\top \hat{x}_j
&#36;&#36;</p>
<p>This formula is useful: as long as vectors are unified, then you make the same kind of comparison, whether you use the "maximum similarity" or "minimum distance". So the text data went from " string collection " to " cloud spot " , and many data cleansing and retrieval problems went into searching, grouping, sideline and covering the cloud at this point.</p>
<h2>LSH: First, it's impossible. <code>O(n^2)</code> It's more like a candidate for recall.</h2>
<p>Suppose you do. <code>n</code> Text of the bar, if you compare all samples directly, the complexity is close</p>
<p>&#36;&#36;
O(n^2)
&#36;&#36;</p>
<p>When? <code>n</code> By the time hundreds of thousands or even millions of people are in the process of being used, this is almost non-existent.<code>LSH</code> The idea is simple: not to be precise compared to all points, but to send the "possibly close" vector into the same barrel with a very cheap Hashi function.</p>
<blockquote>
<p>Many engineering algorithms are dealing with square complexity from pairwise. LSH processing vectors are similar, linear treatment processes, Tournament processes reward comparisons. The targets are different, but the pressure comes from a two-to-two comparison.</p>
</blockquote>
<p>Overlay with the most common random plane <code>LSH</code> For example, a vector, or super-platform in a high-dimensional space, is randomly sampled.</p>
<p>&#36;&#36;
r \sim \mathcal{N}(0, I)
&#36;&#36;</p>
<p>Then define a 1 bit Hash function:</p>
<p>&#36;&#36;
h_r(x)=
\begin{cases}
1, &amp; r^\top x \ge 0 \
0, &amp; r^\top x &lt; 0
\end{cases}
&#36;&#36;</p>
<p>Its instincts are simple: to see the vector. <code>x</code> Which side of the super plane? If two vectors are close, they are more likely to fall on the same side. There is a good conclusion to be made about the angle distance:</p>
<p>&#36;&#36;
\Pr[h_r(x)=h_r(y)] = 1 - \frac{\theta(x,y)}{\pi}
&#36;&#36;</p>
<p>The smaller the angle, the higher the probability of collision. The actual project will not be a single super-platform, but will be a combination of multiple bits and more Hashi watches. The effects of this are:<strong>Then Hashi will recall his candidacy, then make precise parallel calculations in the pool.</strong></p>
<p>The problem with this step is not “who will eventually repeat whom”, but “who deserves to be examined”. It is particularly appropriate for initial cleansing web pages to capture text, forum posts, and template descriptions, as it can quickly narrow the near-repeated sample to a small number of candidates. The price is it's gonna be called back wrongly and by default, so... <code>LSH</code> It is appropriate to do the first layer of filter and not to make a final decision independently.</p>
<h2>Fais + Neighbourhood Map: Rewriting semantics to Image Problem</h2>
<p>When candidates are gathered, the next step is a more precise semantic comparison. The most common move here is top-k nearest neighbor for each sample vector. Samping <code>i</code> ♪ The neighbors gather ♪</p>
<p>&#36;&#36;
N_k(i)=\operatorname{TopK}_j \ s(i,j)
&#36;&#36;</p>
<p>If you're on every one of them, <code>i</code> I'm counting on it. <code>N_k(i)</code>And the whole data is going to be seen as one. <code>kNN</code> Figure.<code>Faiss</code> The effect is to do this efficiently. In small cases, you can do a precise search directly; in larger cases, you can do a precise search.<code>Faiss</code> Yes, it will. <code>IVF</code>、<code>PQ</code>、<code>HNSW</code> The like proximity structures reduce the cost of searching by turning the "pool-by-pool scan" into "searching in a small number of candidate pools".</p>
<p>The weight is not usually “recovered the most like sample”, but define a side:</p>
<p>&#36;&#36;
E={(i,j)\mid s(i,j)\ge \tau}
&#36;&#36;</p>
<p>Here. &#36;\tau&#36; is the similarity threshold. So the whole data was written in a picture.</p>
<p>&#36;&#36;
G=(V,E)
&#36;&#36;</p>
<p>The nodes attached to the figure are a set of highly similar texts. There are two things that can be done:</p>
<p>The first one is...<strong>Connect blocks to heavy</strong>I'm sorry. If a group of samples are close to each other, they are considered a cluster and only one representative sample is retained. The second one is...<strong>Edge Filter + Manual Script</strong>I'm sorry. That is, the view is only that the very similar edge is a “hard repetition” and that the border samples are manually reviewed. This process is much more stable than simply looking at a pair of texts, as it uses local geometry rather than just isolated analogue values.</p>
<p>Here's another detail: it's best to split the barrels by category or source, then build them in the barrels. <code>Faiss</code> Index. Because embedding is a synonym, but business is often concerned with "synthesis of the same semantics." If commodity reviews and commodity titles are combined to map, it is easy to have synonyms that are close but not heavy.</p>
<h2>K-center-greedy: not the most like, but the most representative</h2>
<p>The solution is redundancy, but there is another problem with the data set: Even without repeat samples, data may be highly concentrated in several dense regions. For example, 10,000 samples are almost “a certain type of loophole”, and after weighting, training or budget labelling is still being spent in the same area. Semantic embedding is semantic repetition; if RAG resets the large areas of content that are close to the subject, it also needs a cover sample to lower the label costs. K-center-gredy can be understood as a training sample driven by embedded vectors themselves; correspondingly, the classification and re-sampling techniques in manual tagging are also trying to reduce the disadvantages of random sampling.</p>
<p><code>K-center</code> The objective of the problem is to: <code>k</code> A centre, covering as much as possible the entire sample, so that the distance from the nearest centre is as small as possible:</p>
<p>&#36;&#36;
\min_{S:|S|=k} \max_{x \in X} \min_{c \in S} |x-c|_2
&#36;&#36;</p>
<p>It's hard to get the best of the best of the best, so it's common in engineering. <code>K-center-greedy</code>I'm sorry. Its rules are intuitive: assuming that a pool has been selected at this time. &#36;S_t&#36;, always pick the point farthest from the current assembly:</p>
<p>&#36;&#36;
x_{t+1} = \arg\max_{x \in X \setminus S_t} \min_{c \in S_t} |x-c|_2
&#36;&#36;</p>
<p>This formula can be translated directly into human language:<strong>The most unrepresentative sample of the current one is always found.</strong> The selected subset would not be solely focused on high frequency areas, but would automatically cover border areas, long tail areas and isolated areas.</p>
<p>So... <code>K-center-greedy</code> The solution is not “reduced repetition”, but “protected coverage”. If your downstream tasks are active learning, manual labelling, training compression, it is often more effective than random sampling, because random sampling can be repeated in dense areas, and <code>K-center-greedy</code> The budget will be invested proactively in regions that are currently under-represented.</p>
<h2>Searches, clusters, RAGs, recommendations are essentially re-engineered in the same vector operation.</h2>
<p>Once it is understood that semantic vectors are the point in space, many of the seemingly different tasks simply change the target function. The first step, whether it be for queries, documents, users, commodities or sentence clips, is usually to encode the original object into embedding; the latter is repeated, mainly by several types of basic operations:<strong>Counting similarities, finding immediate neighbors, collating, controlling coverage.</strong>I'm sorry. The tasks are different, often using these calculations to optimize different indicators, and the operations performed do not differ significantly in mathematics.</p>
<p>The most common analogy is either internal.</p>
<p>&#36;&#36;
s(x,y)=x^\top y
&#36;&#36;</p>
<p>Or cosine symmetry.</p>
<p>&#36;&#36;
\cos(x,y)=\frac{x^\top y}{|x|_2|y|_2}
&#36;&#36;</p>
<p>If vectors are previously normalized, then they are almost identical. So, a lot of systems are on the surface of different tracks, and the bottom is all done around the same score. <code>Top-K</code>, recent neighbourhood search, cluster distribution, centre updating and coverage optimization. And that's why you're re-engineered the semantic reset of embedding + ANN, with a slightly different target function, and you're often able to continue with the search, the RG, and even the recommendation for recall.</p>
<p><strong>Search</strong> It's the most direct. Give a query vector &#36;q&#36;, in document vectors &#36;&#123;d_i}&#36; Finds the highest score:</p>
<p>&#36;&#36;
\operatorname{TopK}_i \ q^\top d_i
&#36;&#36;</p>
<p>That's the basic semantic search. It does not stare at the reverbs, but rather looks for the nearest document in the vector space. Inverted indexes are good at visible keywords; vector search complements synonyms, upper and lower concepts and rewritings. Engineering processes are also fixed: documents are indexed by offline encoding, and search encoded online Done. &#36;q&#36;Again. <code>Faiss</code>、<code>HNSW</code> The approximation of the next-door algorithms that have been used to get candidates for the top-k. Reorder the cross-encoder or rule layer when more precision is required. So-called semantic retrieval systems, the core is "Sort to Quantification + Similarity + Near-Neighbour Search."</p>
<p><strong>Cluster</strong> The question is not who is best at asking, but what is natural and should be part of a cluster. The classic is... <code>k-means</code>：</p>
<p>&#36;&#36;
\min_&#123;&#123;c_i},{\mu_j&#125;&#125; \sum_i |x_i-\mu_{c_i}|_2^2
&#36;&#36;</p>
<p>of which &#36;\mu_j&#36; It's the first. &#36;j&#36; The city is a small city.&#36;c_i&#36; Which group is the sample. To untangle the formula, it is still a few steps old: first, each point is assigned to the nearest centre, then each centre is updated to the average of the samples in the cluster. The cluster is not calculated for similarity, but rather for one-on-one comparisons to centre comparisons, which are repeated over time. It is useful in the analysis of the language: a particular size and complexity of a given group, which often indicates many examples of templateized expressions, duplicate samples or rewritings of syntax proximity; and a large radius, which suggests that the subject matter is very diverse in itself and cannot simply be retained in one article. For weighting, the greatest value of the cluster is to reduce the search space first: to judge by cluster organization, then by more detailed similarities within clusters.</p>
<p><strong>RAG</strong> Structurally, it is "Retrieving + Context assembly + Generating." Cut the document to chunk, get the vector. &#36;&#123;c_i}&#36;;after query comes in, first take</p>
<p>&#36;&#36;
\mathcal{C}(q)=\operatorname{TopK}_i \ q^\top c_i
&#36;&#36;</p>
<p>And then you can spell these chunks into the hints and give them to the generation model. What is really new here is that the last step is to feed the search results to LLM; the recall part that is ahead is still standard vector search. Therefore, the quality cap of the RAG depends not only on whether the model will say it, but also on whether you can pick out the context of “relevant, non-duplicative, complementary”. If there are a large number of close repeats in the chunk library, Top-k can easily return five paragraphs that are almost the same thing, wasting context windows; if the pieces are too small, the system may recover a few partially relevant pieces without complete factual chains. So in RAG, it's all about re-engineering, clustering, and coverage optimization. To reset the apparent redundancy, grouping the close chunk together, and the overlay strategy avoids top-k-only looking at the same high-density area. Many RAG-specific problems are followed up to the end, and the same old question is: how well is vector space organized, and whether the results of the search in the immediate neighbourhood are overcrowded and repetitive.</p>
<p><strong>Recommendations</strong> It's the same thing. If the user has a vector &#36;u&#36;The object has vectors. &#36;v_i&#36;..the basic scoring function is</p>
<p>&#36;&#36;
\operatorname{score}(u,i)=u^\top v_i
&#36;&#36;</p>
<p>In the content recommendations,&#36;u&#36; This can be the average, weighted or real time generated by the user tower of the historical click on the content vector;&#36;v_i&#36; The article is from the object tower or content encoder. The core issues at the recall stage, however complex the model is, remain:<strong>The current user is also considered a query vector and then the nearest neighbour is located in the object vector.</strong> This is almost the same mathematically as the search vector for the most similar documents, except that the source of the query is different: the retrieval system is from text entered by the user and the recommended system is from the user behavior sequence. In many industrial systems, search recall and recommend recall even share the same vector index with ANN infrastructure. The recommended diversity control, redispersion, long tail exploration, and the operation ahead: Limit flow within the same content cluster to avoid a row of similar short videos on the first page; add a limit on coverage, so that the result is a little exploration space.</p>
<p>So if you look at these tasks in the same system, you find that they share a very simple set of geometrical words:<strong>Similarity determines who is closer, the immediate neighbour determines who is first, the aggregate determines what the “representative point” looks like, and the overlaying determines whether the result is composted in local areas.</strong> Retrieving the "relevance", cluster attention "structures", RAG focus on "relevance + information density", recommending "preference matching + diversity control"; these differences occur more in target layers than in bottom algorithms. Once the underlying modules of embedding space, near-neighbor index, similarity thresholds, and deweighting strategies are firmly established, many upper layers of applications will have growth space.</p>
<h2>Concluding remarks</h2>
<p>From a formula perspective, these methods are not as dispersed as they appear.<code>LSH</code> It is in the Gavin space that the approximation of the drums is first made to address the recall of the candidates;<code>Faiss</code> (a) The efficient search of a close neighbour in vector space, and the creation of high-level, similar samples into maps and semantic solutions;<code>K-center-greedy</code> The coverage is optimized in the same space and sample selection is resolved. As for retrieval, clustering, RAG, recommendations, they can also be seen as different mission variants around “similarity, distance, proximity and coverage”.</p>
<p>So when you have an embeding vector, you can't do anything. A complete data cleansing and retrieval system will be supported by a clear definition of similarities and distances, combined with search, join, cluster and coverage of these basic operations.</p>
