---
title: 'Embedding Atlas: Visualizing Embedding Spaces for Better RAG'
title_zh: Embedding Atlas：用可视化理解 RAG 的嵌入空间
date: 2026-05-29 20:00:00 +0800
categories:
- Agent Systems
- Agent Infrastructure
tags:
- RAG
- Retrieval
- Embeddings
- Data Curation
- Dimensionality Reduction
author: Hyacehila
mathjax: true
excerpt: Embedding visualization is not just a pretty UMAP plot; it is a diagnostic layer for RAG retrieval, coverage, chunking,
  confusion, and data quality.
description: Embedding visualization is not just a pretty UMAP plot; it is a diagnostic layer for RAG retrieval, coverage,
  chunking, confusion, and data quality.
excerpt_zh: Embedding 可视化的用处不在于画一张好看的 UMAP 图，而是把 RAG 的召回、覆盖、混淆、chunk 和数据质量问题放到同一个几何空间里看。
permalink: /blog/2026/05/29/embedding-atlas-rag-embedding-visualization/
lang: en
translation_key: 2026-05-29-embedding-atlas-rag-embedding-visualization
translation_status: machine
translation_source_hash: e1aea612d6259ef7e366acbf0db67ede1d495e92eb9fe44ced70d36d6eb8d1f2
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<p>When the RAG system is wrong, it's easy to throw the pot to the model: is the model not strong enough, is the hint not good enough, is there a more complex agent loop? It's a natural reaction, because the last thing that's wrong is a model. But in many real questions, the answer is broken earlier. The context of the model is not right, and how it is generated is simply a linguistic organization on the wrong material.</p>
<p>The questions in this article can also be addressed<a href="/en/blog/2026/06/10/ai-agent-retrieval-tools/">AI Agent How to Access Information from the Internet: The Evolution of Search, Capture and Structured Cleaning Tool</a>、<a href="/en/blog/2026/08/18/how-i-build-rag/">RG: From project practice to system approach</a>How the concept of a relatively close read together is developed in different contexts.</p>
<p>Early <a href="https://arxiv.org/abs/2005.11401">RAG</a> and <a href="https://arxiv.org/abs/2004.04906">DPR</a> The study has made this division of labour clear: the generation of external evidence that relies on the quality of the search phase. In engineering practice, this phrase is going to get into a lot of more specific troubles. Often, search failures are not a single-point failure, but a set of spatial problems: queries are not close to the correct document, correct documents are cut too thin, semantics are composing different elements, long-tailed knowledge becomes an island, data from some sources contaminate the entire neighbourhood, or the embedding model treats surface similarities as mission-like.</p>
<p>I'm concerned. <a href="https://github.com/apple/embedding-atlas">Embedding Atlas</a>And it is precisely this issue that it deals with. It is not another little tool to "reduce the vector to two dimensions and draw a scattering map." It places embedding, metadata, search, filtering, proximity and clustering in the same interface, allowing for a genuine reversion of text, pictures or other objects in the way they are organized in space. It's paper. <a href="https://arxiv.org/abs/2505.06386">Embedding Atlas: Low-Friction, Interactive Embedding Visualization</a> Emphasis was placed on “low friction” and “interactive analysis”. It's important for RAG developers: we usually need not a nice cut-off, but a diagnostic portal that can be asked over and over again.</p>
<p>For RAG, the question I really want to ask is simple:</p>
<p><strong>What did this vector space put close to and what did it put far away?</strong></p>
<h2>The search for RAG is a matter of space organization.</h2>
<p>The simplest RAG process can be written as:</p>
<pre><code class="language-mermaid">flowchart LR
    DOC[&quot;原始文档&quot;] --&gt; CHUNK[&quot;切分为 chunk&quot;]
    CHUNK --&gt; EMB[&quot;计算 embedding&quot;]
    EMB --&gt; INDEX[&quot;向量索引&quot;]
    Q[&quot;用户问题&quot;] --&gt; QEMB[&quot;问题 embedding&quot;]
    QEMB --&gt; RET[&quot;近邻检索&quot;]
    INDEX --&gt; RET
    RET --&gt; CTX[&quot;上下文拼装&quot;]
    CTX --&gt; LLM[&quot;生成答案&quot;]
</code></pre>
<p>In engineering terms, there are many steps in this: dissecting, cleaning, cutting, quantifying, indexing, recall, rearranging, spelling the context, and generating. But looking at the floor of embeding, the move is simple: Turns document blocks and queries into points in high space, and then determines who should enter the context with distance or similarity.</p>
<p>Set Document Block Vector to</p>
<p>&#36;&#36;
d_i \in \mathbb{R}^m
&#36;&#36;</p>
<p>Query vector is</p>
<p>&#36;&#36;
q \in \mathbb{R}^m
&#36;&#36;</p>
<p>If I do, <code>L2</code> Normalize, so cosine analogies can be written directly into point size:</p>
<p>&#36;&#36;
s(q,d_i)=q^\top d_i
&#36;&#36;</p>
<p>The thing that vector search does is find the highest degree of similarity before. &#36;k&#36; Document blocks:</p>
<p>&#36;&#36;
\operatorname{TopK}_{d_i} \ s(q,d_i)
&#36;&#36;</p>
<p>The formula is clean, but the real language never cleans. The document contains repeat paragraphs, template text, version differences, advertising noise, short title, long table, code segment, cross-language content, outdated content and permission boundaries. The query does not always look like a document. It may be a question of speech, a misnomer, an implicit need, or even an incomplete mandate directive.</p>
<p>The difficulty of RAG is not "to be retrieved by a vector database." The real hard thing is:<strong>Is space really organized out of the semantic relationship required for the mission?</strong></p>
<p>Nor should the search infrastructure stop at "Throw Query to the vector bank and return to Top-K." Developers need to know what is being recalled, what is missing, whether the noise comes from the slice or from the data source, whether the wrong sample is clustered, and whether the operational boundary has access to vector space. Only when these issues are visible can the recall process be diagnostic and interventionist.</p>
<p>Visualization serves as a window of observation for this issue.</p>
<h2>Embeding Atlas is better suited for diagnosis.</h2>
<p><a href="https://apple.github.io/embedding-atlas/">Embedding Atlas</a> It's the Apple Open Source mpedding visualization tool. It can be used as a command line tool, or embedded in Notebook or Streadlit; the current PyPI version is as <code>0.20.0</code>, request Python <code>&gt;=3.11</code>。</p>
<p>The minimum start-up is direct:</p>
<pre><code class="language-bash">pip install embedding-atlas
embedding-atlas path_to_dataset.parquet
</code></pre>
<p>If you've already calculated the Gaeves embeding, you can specify the vector column:</p>
<pre><code class="language-bash">embedding-atlas path_to_dataset.parquet --vector embedding_vectors
</code></pre>
<p>If you have already calculated the 2D projection in advance, you can also specify the coordinates:</p>
<pre><code class="language-bash">embedding-atlas path_to_dataset.parquet --x projection_x --y projection_y
</code></pre>
<p>It also supports reading data from Hugging Face dataset and provides projections. The latter is particularly useful for RAG diagnostics, because we don't just want to see the position on the two-dimensional map, but we also want to know who the real neighbor is in the raw vector space at each point.</p>
<p>This is the difference between Embeding Atlas and the normal static scattering. The static map tells you at best "The dots are probably in groups"; the interface also allows you to continue asking:</p>
<ul>
<li>What exactly is the text of this regiment?</li>
<li>Is the structure still in place when the results are coloured by source, label, time, length of chunk, and recall?</li>
<li>Who's the closest neighbor to a wrong sample?</li>
<li>After searching for a keyword, do the hits spread or are they concentrated in a particular area?</li>
<li>Is some kind of metadata contaminating a neighbourhood?</li>
<li>Is query and it should be in the same local space?</li>
</ul>
<p>These questions cannot be answered by a single chart. They need to tie together the EMbeding space, the original text and the metada.</p>
<h2>Research and tool links</h2>
<p>It's not Embeding Atlas that thought to project an embedding image into a browsable map.<a href="https://projector.tensorflow.org/">TensorFlow Embedding Projector</a> The #Para, t-SNE, UMAP and metadata filters were used as interactive interfaces early on.<a href="https://home.nomic.ai/blog/posts/data-mapping">Nomic Atlas data making series</a> It has also been said that unstructured data maps are not just decorations, but are themselves an interface for data exploration and governance.</p>
<p>Embeding Atlas is an increase more like pulling this route back to the developers' daily. It is concerned not only with projection algorithms, but also with large-scale cloud dotting, density clustering, automatic labels and low-cost access to existing data workflows. In other words, it took epedding visualizes from "sometimes making an analysis" to "opening a set of vectors with your hand."</p>
<p>The RAG direction has a similar tool vein.<a href="https://arize.com/docs/phoenix/retrieval/overview-retrieval">Arize Phoenix</a> Links retrieving, tracking, eval and UMAP views to analyse the relevance of bad response grouping, unneighbored queries and retrieval.<a href="https://github.com/Renumics/renumics-rag">Renumics RAG</a> and articles <a href="https://medium.com/data-science/visualize-your-rag-data-evaluate-your-retrieval-augmented-generation-system-with-ragas-fc2486308557">Visualize your RAG Data</a>, and then inserts the problem, the document clip and the RAGAS fraction in the same UMAP view. And look inside the RAG system.<a href="https://arxiv.org/abs/2411.01751">RAGViz</a> The focus on the token and document levels is visualized, and the focus is on the following:<a href="https://arxiv.org/abs/2601.12991">RAGExplorer</a> You can use different RAG configurations to make comparative diagnostics.</p>
<p>So, visualization of RAG is not one thing, but at least three layers. The first level is eptedding space visualization, looking at the distribution of query, document, metadata and wrong samples; the second level is evaluability of the search chain, looking at the trace, the search results and the context correlation; and the third level is visualization of the generation process, looking at which documents the model was looking at when it was generated. This paper is mainly devoted to the first layer, but it is preferable that it not be separated from the second two.</p>
<h2>Vector to figure: layers behind visualization</h2>
<p>Embeding is usually subject to four layers of conversion: high-dimensional vector, near-neighbor chart, 2-D projection and interpretable labels.</p>
<p>The first level is high-dimensional vector. Embedding model maps text, pictures or other objects as vectors. The distance here is not physical distance in ordinary space, but the representational relationship that models learn. It may capture themes, entities, tone, format, language, sources, or may magnify some surface features that are less important to the operation.</p>
<p>The second level is the Neighbourhood Map. For each point top-k near the neighbor, you can get a picture:</p>
<p>&#36;&#36;
G=(V,E)
&#36;&#36;</p>
<p>Each of these samples is nodal and is sufficiently contiguous between the closely related samples. Many of the RAG phenomena are better understood in this picture: repeat content becomes a secret group, isolated knowledge becomes a marginal point, cross-thematic pollution becomes a faulty bridge, and semantic confusion is a entanglement between two business categories in the immediate neighbourhood.</p>
<p>The third floor is a 2D projection. UMAP, t-SNE, PCA, etc. have put high-dimensional points to 2D, which allows for eye-witnesses. Embeding Atlas documents refer to two-dimensional coordinates that can be calculated on the basis of an available vector or read directly.<a href="https://arxiv.org/abs/1802.03426">UMAP</a> The instinct is to keep as close a point as possible in the high space as possible in the 2D. It is suitable for observing local mass, bridge and dislodge sites.</p>
<p>The fourth floor is label with metadata. The embedding diagram without metada can easily become a "looks structurally" illusion. Only by folding the fields of source, type, time, type of document, length of chunk, number of recalls, manual marking, type of error, will the spatial structure begin to be a diagnostic clue.</p>
<p>I prefer to draw the RAG observation process this way:</p>
<pre><code class="language-mermaid">flowchart LR
    A[&quot;文档与查询样本&quot;] --&gt; B[&quot;chunk 与清洗&quot;]
    B --&gt; C[&quot;embedding 向量&quot;]
    C --&gt; D[&quot;近邻图 / UMAP 投影&quot;]
    D --&gt; E[&quot;Embedding Atlas 交互分析&quot;]
    M[&quot;metadata&lt;br/&gt;来源 · 类型 · 时间 · 标签 · 错误样本&quot;] --&gt; E
    E --&gt; F[&quot;RAG 诊断&lt;br/&gt;召回 · 覆盖 · 混淆 · 数据质量&quot;]
</code></pre>
<p>Visualization is not the last image to be presented here, but rather the diagnostic layer between data construction and system assessment.</p>
<h2>First diagnosis: query and the correct documents are not close.</h2>
<p>The most direct RAG diagnosis is to put query in the evaluation and into the same EMbeding space.</p>
<p>If a problem has a correctly manually marked document or if there is a weak monitoring signal such as clicks, adoptions, quotations, etc., you can visualize both query and the possible document. Ideally, query should be near it, at least into the same local neighborhood.</p>
<p>Once they are not close, it is worth stopping to read the original language.</p>
<p>Sometimes problems come in query-document mismatch. The user is in the task language, and the document is in the description language. For example, the user asks " Why is the time is running out here " , and the document says "the default request is 30 seconds. The two are semantics relevant, but not in a close word. This may require query rewriting, HyDE, field fine-tuning, or adding FAQ titles to the index.</p>
<p>Sometimes the problem is in chunk. The correct answer cuts through several paragraphs, but after the cut, each chunk keeps only local information, resulting in neither single chunk nor query being close enough. This is not about vector banks, but about the chunk strategy. When visualize, you see multiple chunks of the same document scattered in different areas, or query falls between several relevant chunks, none of which is close enough.</p>
<p>And there are times when the embedding model itself is not fully integrated. Generic embedding may be better at subject-like, and your RAG mission needs to distinguish between cause and effect, step, API constraint, error code, permission boundaries or version differences. The figure appears to be “thematically similar”, but the operational document that really should be recalled is not prominent.</p>
<p>This diagnosis turns abstract recalm fair into a question of observation: is query a fault, a fall on the border of the regiment, or is the mission itself a mess?</p>
<h2>Diagnosis II: Whether the wrong recall is a conundrum</h2>
<p>Individual cases of error are often not sufficiently illustrative. A query that did not recall the correct document may be a coincidence; but if many mistakes were recalled in a single cluster, it would mean that the system was making the same mistake in a stable manner.</p>
<p>For example, there is an internal knowledge base in an enterprise, along with a “reimbursement process” “contract approval” “procurement request” “payment request”. These documents contain the words " application, approval, amount, invoice, person responsible " . General embeding can easily put them in the same group. RG may be recalled to the claims process or procurement approvals once such questions as “how to apply for payment” have been encountered.</p>
<p>If you look at online logs alone, these errors will spread under different querys, and it is difficult to create an instinct. Put them in the embedding diagram, they are presented as a mixed area: chunk of different business types interpenetrates each other and wrongly recalls a large number of cross-tapes.</p>
<p>Several modifications could be considered:</p>
<ul>
<li>Adding business domain headings before chunk to allow vectors to contain clearer context;</li>
<li>(a) Distinct the search into two sections: domain tracking, then in-area vector search;</li>
<li>Introduction of metadata formulae, such as sectors, document type, product line, version;</li>
<li>hard classes, fine-tuning embedding or reranker;</li>
<li>Reranker supplements the indistinguishable fine particle size limits.</li>
</ul>
<p>But don't be too quick to "get mixed up and change the model." Some blends are similar to the subject matter and not necessarily wrong; some mix suggest that operational boundaries are being broken. Visualization provides only clues and ultimately returns to the assessment sample and business semantics.</p>
<h2>Third diagnosis: whether longtail knowledge has become an island</h2>
<p>A common goal of RAG is to enable models to take advantage of long-tailed knowledge: cold door functionality, old version descriptions, few error codes, low-frequency customer issues, internal process details. The trouble is, longtail knowledge is often seen as an island in the embeding space.</p>
<p>It's not necessarily bad for an island. Some elements are inherently special and should be kept away from the parent community. But if the content of an island is often asked but rarely recalled, it means that the system is not good for its accessibility.</p>
<p>In Embeding Atlas, points can be coloured by frequency of access, frequency of recall, manual problem coverage or online failure rate. This can be seen in several different forms:</p>
<ul>
<li>HF access areas: should be maintained with a focus and error has a high impact;</li>
<li>Low frequency but high value islands: need assessment and assessment of specialized coverage and cannot be flooded by average indicators;</li>
<li>(a) Insolent islands: it may be cold knowledge, it may be outdated, repetitive or noise;</li>
<li>Bridge points at the edge of the main group: Often cross-thematic documents, which may provide connections or create false recall.</li>
</ul>
<p>This will change the way we understand RAG assessments. Average recall@k only tells you the overall recall rate, and space perspective reminds you that errors are not evenly distributed. The system may be doing well on mainstream issues, while systematically neglecting marginal regions.</p>
<h2>Diagnosis number four: chunk is too broken, too thick, or is it the wrong context?</h2>
<p>The most likely underestimation in RAG is chunk. The splitting strategy is not just a text pre-processing process, it directly determines the basic unit in the embeding space.</p>
<p>When chunk is broken, many points become semantic. They may be half a step, one sentence definition, one table row or one code segment. Such a point is easily sucked away in space by the surface, close to something that does not really answer the question.</p>
<p>When chunk is too thick, another problem arises: there are multiple themes within one point. It could be a bridge between several groups on the 2D chart. Such bridges sometimes help to cross-thematic recall, but they may also link unrelated neighbourhoods, resulting in wrongful recall.</p>
<p>There is also a more subtle situation: the chunk itself is not short, but lacks context such as title, path, product name, version number, etc. For example, the phrase “select advanced options after clicking on settings” does not in itself indicate which product, which page, which version. When looking only at chunk content, embedding is likely to mix similar steps in different products.</p>
<p>The chunk analysis should not be limited to length distributions, but should also be accompanied by several fields in visualization:</p>
<ul>
<li><code>chunk_length</code>: whether the short spots are concentrated in noise areas;</li>
<li><code>document_id</code>: whether the chunk of the same document is overdispersed;</li>
<li><code>section_title</code>: whether the title helps the chunk to get back to the right theme;</li>
<li><code>source</code>: whether certain sources form an abnormal mass;</li>
<li><code>version</code>: whether the old and new versions are mixed in the same neighbourhood.</li>
</ul>
<p>If the chunk of the same document is completely dispersed in space, it may not be wrong, because the document may be multi-topic. But if the answers to certain key documents are broken to the point where query cannot be hit, then the cut, the title is inserted or the parent chunk search is reconsidered.</p>
<h2>The fifth diagnosis: data quality issues will be visible in space.</h2>
<p>Embeding Space also exposes data cleansing.</p>
<p>Repeating texts would create a very dense grouping. Templateed pages are approaching by sharing headers, footers, navigational text files. Machine translation or OCR noise may create strange marginal areas. The obsolete document may be similar to the height of the new document, but it is in a different version on the metadata. The error pages, login pages, table of contents pages that the reptiles have captured may also form a visible mass.</p>
<p>The effect of these problems on RAG is real. Repeating content is wasteful top-k slots; template text can contaminate similarities; old versions of documents will be recalled in competition with new versions; table of contents pages will be wrongly recalled because of the wide coverage of keywords; and the wrong pages, once they enter the context, may result in wrong answers.</p>
<p>Many cleaning problems are exposed earlier than in the table if data sources, crawling times, document type and version are added to the embedding diagram. You might find:</p>
<ul>
<li>A source forms a large, but low-value, community;</li>
<li>(b) There is almost overlap between different versions of the same document;</li>
<li>(a) The blank pages, error pages and navigation pages are grouped in small groups;</li>
<li>Some high-recall chunk is actually just a template text;</li>
<li>A large number of short chunks are dominated by titles or fixed speech.</li>
</ul>
<p>And at this point, Embeding Atlas is more like a data check-up tool than just a RG reference tool. It helps you decide what to weigh, drop, filter, retort or supplement metada.</p>
<h2>Visualization into a serious RAG assessment process</h2>
<p>The greatest risk of visualization is to over-trust images. The 2D projection is convincing, but it is not the Gaelic Space itself. Both UMAP and t-SNE sacrifice other structures to preserve some structures. The distance between the two clusters in the figure does not necessarily mean that they are also far away in the raw vector space; nor does it seem separate from each other when retrieved.</p>
<p>So visualization of embedding should not replace indicators, but rather should be a cycle of indicators.<a href="https://arxiv.org/abs/2104.08663">BEIR</a> and <a href="https://arxiv.org/abs/2210.07316">MTEB</a> Reminds us that searching and embeding require a cross-mission assessment;<a href="https://arxiv.org/abs/2309.15217">RAGAS</a> Further decompose RAG into dimensions such as context, faithfulness, power quality;<a href="https://arxiv.org/abs/2405.07437">RAG Overview of Assessments</a> and Chinese base <a href="https://arxiv.org/abs/2401.17043">CRUD-RAG</a> It is clear that retriever, context length, knowledge base control and LLM can influence the outcome.</p>
<pre><code class="language-mermaid">flowchart LR
    A[&quot;离线评测集&quot;] --&gt; B[&quot;检索指标&lt;br/&gt;recall@k · MRR · nDCG&quot;]
    B --&gt; C[&quot;错误样本集合&quot;]
    C --&gt; D[&quot;Embedding Atlas 可视化&quot;]
    D --&gt; E[&quot;形成诊断假设&quot;]
    E --&gt; F[&quot;修改数据 / chunk / embedding / reranker&quot;]
    F --&gt; B
    G[&quot;人工抽检与线上反馈&quot;] --&gt; C
</code></pre>
<p>In this cycle, the indicator answers “Is it better or not” and visualizes “Why is it possible to be better or worse”. Neither of these issues can be spared.</p>
<p>A practical process could be:</p>
<ol>
<li>Prepare a batch of query, proposive document, negative document and online error samples;</li>
<li>Query and document chunk are placed in the same data table;</li>
<li>Add fields:<code>type</code>、<code>source</code>、<code>doc_id</code>、<code>section</code>、<code>chunk_length</code>、<code>label</code>、<code>retrieval_result</code>、<code>error_type</code>；</li>
<li>Calculates embeding and maintains vector columns;</li>
<li>Open data with Embeding Atlas, colouring and filtering different metadatas;</li>
<li>Read the original text of the wrong sample and develop specific assumptions;</li>
<li>Modify the chunk, metadata filler, reranker or embedding policy;</li>
<li>Back to offline indicators and manual assessment validation.</li>
</ol>
<p>The most important step here is step 6. Visualization can only lead you to "a place worth reading" and not to reason for it. The real diagnosis is still to be found in the original language, the mission and the user's intentions.</p>
<h2>Changes in understanding resulting from embedded visualization</h2>
<p>I think the most interesting place for visualization is that it will change our instincts about RAG.</p>
<p>First, it transforms the “knowledge base” from a collection of documents into a geometric object. In the past, we have talked about the quality of the knowledge base, often thinking about the incompleteness of the document, the unsophisticated format, and whether it has been updated. After visualization, you start asking: Are these knowledges well covered in space? Which areas are too dense and which are empty? Which borders are blurred and which islands are not accessible?</p>
<p>Secondly, it allows “recall errors” to change from case to case. An error query may be just a coincidence, and a misdirection suggests a stable deviation in the system. RAG optimization should not be pursued only in individual bad case, but should find the wrong space model.</p>
<p>And third, it makes the "embeding model OK" more specific. Not abstractally speaking, a model has higher scores, but it's a matter of whether it expresses the boundaries of your mission. For the client service RAG, product line boundaries may be important; for the paper RAG, the relationship between methodology, tasks and data sets may be more important; for the code RAG, API name, call constraints and version differences may be more important than similar themes.</p>
<p>And fourthly, it reminds us that RAG is not just a search algorithm, it's also a data design. Many of the search problems were not solved by changing the ANN index, but by training in data cleansing, chunk re-engineering, headline injection, metadata filter, stratification routers and hard negative training.</p>
<p>So I'll focus on Embeding Atlas as a tool. It turns embedding from an intermediate product hidden in a vector database into an object of work that can be observed, discussed and modified.</p>
<h2>But don't take the map as an answer.</h2>
<p>Finally, it is important to make the borders clear: do not take the map as an answer.</p>
<p>A 2D projection is not a primitive high-dimensional space. Visualized clusters do not automatically equal real categories, discrete points do not automatically equal bad data and mixing is not automatically equal to errors. The UMAP parameters, sampling methods, aggregation, vector models, metadata selections will influence the picture you see.</p>
<p>Visualization is better for three things:</p>
<ul>
<li>Discrepancies are detected;</li>
<li>(b) Development of diagnostic assumptions;</li>
<li>Select the sample of manual sampling.</li>
</ul>
<p>It is not appropriate to draw final conclusions alone. Finally, it is still up to the recall@k, MRR, nDCG, answerr policy, context regulation, Faithfulness, manual preferences, online clicks, user feedback and business costs. There are similar reminders in Chinese practice articles: <a href="https://jimmysong.io/book/ai-handbook/rag/observability/">RG Observable</a> It's a two-dimensional map that puts search indicators, semantic quality, logs and tracks together.</p>
<p>Embeding Atlas provides a kind of observationability. It doesn't automatically make RAG better, it doesn't make a decision about changing models, crunk, or reranker. But it will allow a part of the RAG to fail to emerge from the black box: what is the model problem, which is the chunk problem, which is the data problem, which is the business boundary without entering vector space.</p>
<p>When we say "better RAG," we should not think about bigger generation models, stronger reranker, or more complex anent. And sometimes, better RAG comes from a few things that are more simple: knowing what your language looks like, where your queries are, how your mistakes are made, whether your long-tail knowledge is covered.</p>
<p>Embeding the visual meaning is right here. It does not make decisions for the system, but it finally allows us to observe vector space itself.</p>
<h2>References</h2>
<p><strong>Project and official documents</strong></p>
<ul>
<li><a href="https://github.com/apple/embedding-atlas">apple/embedding-atlas</a></li>
<li><a href="https://apple.github.io/embedding-atlas/">Embedding Atlas Documentation</a></li>
<li><a href="https://pypi.org/project/embedding-atlas/">Embedding Atlas PyPI</a></li>
<li><a href="https://projector.tensorflow.org/">TensorFlow Embedding Projector</a></li>
<li><a href="https://home.nomic.ai/blog/posts/data-mapping">Nomic Atlas Data Mapping</a></li>
</ul>
<p><strong>Articles and benchmarks</strong></p>
<ul>
<li><a href="https://arxiv.org/abs/2505.06386">Embedding Atlas: Low-Friction, Interactive Embedding Visualization</a></li>
<li><a href="https://arxiv.org/abs/1802.03426">UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction</a></li>
<li><a href="https://arxiv.org/abs/2005.11401">Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks</a></li>
<li><a href="https://arxiv.org/abs/2004.04906">Dense Passage Retrieval for Open-Domain Question Answering</a></li>
<li><a href="https://arxiv.org/abs/2104.08663">BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models</a></li>
<li><a href="https://arxiv.org/abs/2210.07316">MTEB: Massive Text Embedding Benchmark</a></li>
<li><a href="https://arxiv.org/abs/2309.15217">Ragas: Automated Evaluation of Retrieval Augmented Generation</a></li>
<li><a href="https://arxiv.org/abs/2405.07437">Evaluation of Retrieval-Augmented Generation: A Survey</a></li>
<li><a href="https://arxiv.org/abs/2401.17043">CRUD-RAG: A Comprehensive Chinese Benchmark for Retrieval-Augmented Generation of Large Language Models</a></li>
<li><a href="https://arxiv.org/abs/2411.01751">RAGViz: Diagnose and Visualize Retrieval-Augmented Generation</a></li>
<li><a href="https://arxiv.org/abs/2601.12991">RAGExplorer: A Visual Analytics System for the Comparative Diagnosis of RAG Systems</a></li>
</ul>
<p><strong>Tool practice</strong></p>
<ul>
<li><a href="https://arize.com/docs/phoenix/retrieval/overview-retrieval">Arize Phoenix Retrieval Overview</a></li>
<li><a href="https://github.com/Renumics/renumics-rag">Renumics RAG</a></li>
<li><a href="https://medium.com/data-science/visualize-your-rag-data-evaluate-your-retrieval-augmented-generation-system-with-ragas-fc2486308557">Visualize your RAG Data: Evaluate your Retrieval-Augmented Generation System with Ragas</a></li>
</ul>
<p><strong>Chinese practice and translation</strong></p>
<ul>
<li><a href="https://blog.csdn.net/deephub/article/details/136094642">Use UMAP to down-vision RG embedded</a></li>
<li><a href="https://llamaindex.org.cn/blog/evaluating-the-ideal-chunk-size-for-a-rag-system-using-llamaindex-6207e5d3fec5">RAG Division Size Guide: Find Best Settings</a></li>
<li><a href="https://jimmysong.io/book/ai-handbook/rag/observability/">RG Observability: how to monitor the search for enhanced generation systems</a></li>
</ul>
