---
title: 'Text Embedding: From Bag-of-Words to Qwen3 Embedding'
title_zh: 文本嵌入：从词袋模型到 Qwen3 Embedding
date: 2024-09-25 20:00:00 +0800
categories:
- Machine Learning
- Classical Machine Learning
tags:
- NLP
- Embeddings
- Word2Vec
- BERT
- Retrieval
author: Hyacehila
mathjax: true
hidden: true
excerpt: 'Traces text embedding evolution: Bag-of-Words, TF-IDF, Word2Vec, BERT dynamic embeddings, GPT autoregressive embeddings,
  to Qwen3 Embedding.'
description: 'Traces text embedding evolution: Bag-of-Words, TF-IDF, Word2Vec, BERT dynamic embeddings, GPT autoregressive
  embeddings, to Qwen3 Embedding.'
excerpt_zh: 梳理文本嵌入技术的发展脉络：词袋模型、TF-IDF、Word2Vec、BERT 动态嵌入、GPT 自回归嵌入到 Qwen3 Embedding。
permalink: /blog/2024/09/25/text-embedding-from-bow-to-qwen3/
lang: en
translation_key: 2024-09-25-text-embedding-from-bow-to-qwen3
translation_status: machine
translation_source_hash: c780f7953d8b55bc3532be043d39e5c5740d66559eb9a583f208985cd5cece5b
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<p>From Bag of Words to Qwen3 Embedding describes the development and application of embedded text models.</p>
<p>The purpose of this paper is to provide a comprehensive overview of the core development nexus of text retrieval and expression techniques, and to map a clear evolutionary path from classic statistical models to modern large-scale neuronet models.</p>
<p>We will understand its narrow search paradigm based on word-frequency statistics, starting with the word bag and BM25 algorithms, the cornerstone of information retrieval. It then entered the age of deep learning, reading Word2Vec, which created static words embedded in the river.</p>
<p>Then focus on the two revolutions brought about by Transformer: the deep two-way understanding model represented by BERT and the powerful self-regression generation model represented by GPT, exploring how they can generate revolutionary dynamic contextual embedding.</p>
<p>Finally, taking the example of the recently released Qwen3 Embedding, an in-depth analysis of how the new generation of SOTA (State-of-the-Art) models, based on the ideas of the heirs, is moving the text embedded and reordering technology to a new peak through advanced structures and training strategies.</p>
<h2>Text embedding with the beginning of a statistical language model</h2>
<p>In order to use the text data for various machine learning model analyses, it needs to be converted into a standardized format. The first question is how to convert text information into flat vectors, which is also the first step of the text data signature project.</p>
<h3>Word bag model</h3>
<h4>One word bag model</h4>
<p>In the Bag of Words model,<strong>Each text document (a sentence or sentences) is converted into a numerical vector corresponding to a point in high-dimensional space.</strong></p>
<p>This number vector contains all the words that may appear in the vocabulary (or all the words that we need to study) and determines the structure of the vector according to the number of times the word appears in this document. If not, the vector is zero.</p>
<p>The wordbag model does not represent any word level, so it is called a flat model. It's a natural problem. &quot;not bad&quot; Meaning &quot;good&quot;But in the wordbag model, it'll be understood as &quot;bad&quot; Meaning. Distinction of the phrase structure entails inevitable loss of meaning.</p>
<h4>n Phrase Bag</h4>
<p>&#36;n&#36; The wordbag (bag of n-grams) is a natural extension of the wordbag model, and we would like to retain the meaning of the sequence structure in some texts.</p>
<p>After some verbology,&#36;n&#36; The meta-bag converts the entire document into a number of vectors, each of which represents a group of words that can be repeated.</p>
<p>&#36;n&#36; The meta-bag model. &#36;n&#36; indicates that the maximum number of words allowed contains the gram.&#36;n&#36; The larger the content, the more informative the language is, but also the higher vector dimension.</p>
<h3>Filter problem for BOW</h3>
<p>BOW is the core instrument of the research embedded, but it also has natural flaws. The question of how to separate meaningful information and noise from the text in an appropriate way is a question that we need to study and, after filtering, models will be more effective.</p>
<h4>Disable Word</h4>
<p>Discontinuing words such as the,a,on, which have no practical meaning, are largely devoid of real meaning and are generally used only in emotional analysis. For the purpose of understanding, the discontinuation could be completely removed.</p>
<p>The NLTK package for Python contains a list of inactive words constructed by linguists, all of which are lowercase and contain a set number, and uses this package to remove the disabled word.</p>
<h4>HF words</h4>
<p>A high-efficiency technique for handling high-frequency words: HF words appearing in a language library are likely to be discontinued. After the word has been deleted, it is likely to be a very common term in a particular language library and its relevance needs to be considered separately.</p>
<h4>Rare words</h4>
<p>Very rare words in a language library are also worth considering; they may be some kind of spelling error or a real rare word.</p>
<p>The presence of very rare words leads to excessive vector dimensions of the word bag model, which often cannot be used as a basis for prediction, but can lead to huge computing costs. Removing the rare words in the wordbag model is a very common NLP feature engineering technique.</p>
<h4>Word dry extraction problem</h4>
<p>In the NLP problem, the variations of a word are very common data. They are counted separately in word bags as different words, but they are very effective in combining their counts because they do have the same meaning.</p>
<p>The Python NLTK package provides an interface for word dry extraction, but it is not almighty. Sometimes a term that is close in form but has different meanings leads to a bad effect after a dry extraction.</p>
<h3>TF-IDF</h3>
<p>The word frequency count in the wordbag model does not automatically remove the disabled word. Although it can be removed at a later stage, the model itself is not as appropriate.</p>
<p>The core idea of TF-IDF is:<strong>The value of a word to a document is proportional both to the frequency it appears in the document and to the extent to which it is prevalent throughout the language library.</strong> Pass.&quot;Counter-document Frequency&quot;To punish words that are too common.</p>
<hr>
<p>TF-IDF is the product of two components:</p>
<p><strong>Term Frequence, TF</strong> Expression &#36;t&#36; In Document &#36;d&#36; The frequency of occurrence is commonly defined as:</p>
<p>Original Frequency
&#36;&#36;\text{TF}(t, d) = \text{count}(t, d)&#36;&#36;
Normalization frequency (most commonly used)
&#36;&#36;\text{TF}(t, d) = \frac{\text{count}(t, d)}{\text{总词数 in } d}&#36;&#36;
logarithmic scaling frequency (for excessively long documents)
&#36;\text{TF}(t, d) = 1 + \\log(\text{count}(t, d))\quad (\text{if}\text{count} &gt; 0)&#36;&#36;</p>
<p><strong>Counter-document frequency (Inverse Document Frequency, IDF)</strong> Rareness of measured terms, definitions
&#36;&#36;\text{IDF}(t, D) = \log\left( \frac{N}{\text{df}(t)} \right)&#36;&#36;
of which &#36;N&#36; is the total number of documents,&#36;dt(t)&#36; It's a word. &#36;t&#36; , the number of documents is obtained according to custom.</p>
<p>In order to avoid a zero error, smoothing items are often added in practice:
&#36;&#36;\text{IDF}(t, D) = \log\left( \frac{N + 1}{\text{df}(t) + 1} \right) + 1&#36;&#36;</p>
<p><strong>The final TF-IDF value is:</strong>
&#36;&#36;\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)&#36;&#36;
Every document &#36;d&#36; It is a vector whose value is the TF-IDF weight of a word in the vocabulary of each dimension.</p>
<hr>
<p>TF-IDF (inverted document frequency) as<strong>Simple and efficient</strong>The NLP feature extraction technique is the most recommended option, more reliable and easier to use than BOW, if a simplest and comprehensible model is needed to study machine learning of text data.</p>
<h2>Rare search technology peak: BM25</h2>
<h3>Okapi-TREC</h3>
<p>Before discussing embedded models, it is important to understand the powerful Baseline that it needs to transcend. BM25 is not one.&quot;Embedded Model&quot;, instead a lexicon search (Lexical Retrieval) algorithm. BM25 is the core technology in modern information retrieval, and almost all modern search engines use BM25 algorithms or their improved form.</p>
<p>BM25 is not a cross-cutting development, it's...<strong>Classic probability model in the field of information retrieval</strong>(Probabilistic Model) Natural evolution by <strong>Okapi project</strong>The BM series is entitled Okapi-TrEC, where BM means Best Match.</p>
<p>BM25 and TF-IDF are classic text weights used to measure the importance of words in documents. Although they're similar in form.&quot;Word Frequency&quot;and&quot;Counter-document Frequency&quot;Elements) but the actual design thinking is not the same.</p>
<p>TF-IDF Hope<strong>Quantify the document</strong>BM25 estimates that one document is correct<strong>Rating of the relevance of the given query</strong>For sorting. The former is used to create usable features for the document, while the latter is used to retrieve existing documents based on queries. Therefore, the application of the scenario was inconsistent.</p>
<h3>BM1</h3>
<p>Okapi in TREC-1 is original <strong>Robertson-Sparck Jones Weight</strong>to give a weight to a word under the existing document library:</p>
<p>&#36;&#36;w(1) = \log \frac{(r + 0.5)/(R - r + 0.5)}{(n - r + 0.5)/(N - n - R + r + 0.5)}&#36;&#36;
of which &#36;N&#36; is the total number of documents,&#36;n&#36; It's a word. &#36;t&#36; .&#36;R&#36; is the number of documents known to be clearly associated with the word,&#36;r&#36; Is the words contained in these relevant documents &#36;t&#36; Number. If there is no pre-reference, the two are degraded to 0, and the pattern is similar <strong>IDF</strong> Format:
&#36;&#36;w(1) \approx \log \frac{N - n + 0.5}{n + 0.5} \propto \text{IDF}&#36;&#36;</p>
<p>BM1 is given a score algorithm based on the weight of words and documents that can be calculated earlier. For one. &#36;m&#36; Word Query &#36;Q={q_1,q_2,...q_m}&#36; And a document. &#36;D&#36;, BM1 score formula is:
&#36;\text{Score}<em>{\text{BM1&#125;&#125;(D, Q) = \sum</em>{i\capD}w(1) q i}&#36;&#36;
Peace only<strong>Words that appear in both Query and Document D</strong>Conducted (i.e., intersecting).</p>
<hr>
<p>BM1 is one<strong>Binary model</strong>(binary model), only&quot;Query words appearing in document&quot;Peace. If word &#36;q_i&#36; Not in Document &#36;D&#36; , it &#36;D&#36; Score<strong>No contribution</strong>..and its contribution to &#36;w(1)_{q_i}&#36;BM1 doesn't consider word frequency TF.</p>
<p><strong>General in physical search &#36;R = r = 0&#36;</strong>, because the document cannot be confirmed when the search occurs. At this point, the score calculation portion of the BM1 algorithm automatically degrades to the smooth form of the IDF.</p>
<h3>BM11 and BM15</h3>
<p>In order to address the limitations of BM1, two variants were proposed in TREC-2, addressing three issues: (1) not processing word frequency - one and 10 scores; (2) not considering the impact of document length on the probability of word occurrence - - Longer than possible.&quot;By chance.&quot;Include query words even if their content is not relevant; (3)<strong>Unable to distinguish the importance of query words</strong>— No matter how many times a query word appears, no weight is added.</p>
<p><strong>BM15</strong>Just right.<strong>Word Frequency</strong>Saturation,<strong>Do not consider document length</strong>：
&#36;&#36;\text{score} \propto \frac{\text{tf&#125;&#125;{k_1 + \text{tf&#125;&#125; \cdot w(1)&#36;&#36;
of which &#36;tf&#36; It's a word frequency.&#36;k_1&#36; As the control of word saturation as an ultra-parameter,&#36;w(1)&#36; Calculates the weight of a word to the document according to the previous method. BM15 still uses the preceding request and method in calculating the total points - only&quot;Query words appearing in document&quot;Peace, but consider &#36;tf&#36; The weight.</p>
<p><strong>BM11</strong>On the basis of BM15,<strong>Include document length harmonization</strong>(based on&quot;Redundancy assumptions&quot;）：
&#36;&#36;\text{score} \propto \frac{\text{tf&#125;&#125;{k_1 \left( (1 - b) + b \cdot \frac{\text{dl&#125;&#125;{\text{avdl&#125;&#125; \right) + \text{tf&#125;&#125; \cdot w(1)&#36;&#36;
of which &#36;dl&#36; The length of the current document,&#36;avdl&#36; The average length of all documents,&#36;b&#36; is the super-parameter for the control length to the unified strength. BM11 still uses the preceding requirements and methods in calculating the total points - only&quot;Query words appearing in document&quot;Peace, but consider &#36;tf&#36;、&#36;dl&#36; The weight.</p>
<p>The core idea of the redundancy scenario is that long documents contain more words because they contain a large number of words.&quot;Redundancy&quot;or&quot;Repeat&quot;Content, not more.</p>
<h3>BM25</h3>
<p>In TREC-3, the author found BM11 punishment for document length<strong>Too heavy.</strong>（&quot;Redundancy assumptions&quot;May exaggerate the length effect, original BM11 is taken in part &#36;b=1&#36;) and then one.<strong>General form</strong>, unite BM11 with BM15:</p>
<p>BM25 Core Formula (for single query words) &#36;q_i&#36; In Document &#36;D&#36; :
&#36;&#36;\text{score} (D, )=sum (n)\text{IDF}(q i)\cdot\frac{text{tf}<em>{q_i,D} \cdot (k_1 + 1)}{ \text{tf}</em>{i,D} \k 1\left1 - b \cdot\frac{dl}{\text{avgdl&#125;&#125;&#36;&#36;
Of which IDF is similar to the front &#36;w(1)&#36; No form of feedback,&#36;tf&#36; This means the frequency at which the word appears in the document,&#36;dl&#36; The length of the current document,&#36;avdl&#36; is the average length of all documents.</p>
<p>&#36;k_1&#36; Control of word saturation as ultra-parameters: degradation of the model to binary values at close to 0, closer to infinite proximity &#36;tf&#36; The linear growth is usually between 1 and 2.&#36;b&#36; It's the hyper-parameter for the control length to the unified strength:&#36;b=0&#36; It's not like you're going to be able to do it with the same length.&#36;b=1&#36; Full integration (equivalent to BM11).</p>
<p>Supplementary note: qtf in BM25, although standard BM25 achieves constant omission of qtf, the original formula consists of:
&#36;&#36;\text{score} (D, )=sum (n)\text{IDF}(q i)\cdot\frac{text{tf}<em>{q_i,D} \cdot (k_1 + 1)}{ \text{tf}</em>{q_i,D} + k_1 \left(1 - b + b \cdot \frac{|D|}{\text{avgdl&#125;&#125; \right) } \cdot \underbrace{ \frac{\text{qtf}<em>{q_i} \cdot (k_3 + 1)}{ \text{qtf}</em>{q_i} + k_3 } }_{\text{query term frequency component&#125;&#125;&#36;&#36;</p>
<ul>
<li>If the user repeats a word &quot;AI AI AI&quot;) increases its weight.</li>
<li>Exactly.<strong>Distinguishing the importance of query words</strong>A mechanism.</li>
</ul>
<p>However, in modern physical search, the query usually comes from the topic CONCEPTS field (without repetition), so qtf is often 1 and the item is ignored.</p>
<h3>Summary</h3>
<p>Design ideas for BM25:</p>
<ul>
<li><strong>Word Saturation</strong>(TF Saturation): 10 or 20 times a word appears in a document and the difference of importance should be less than 1 change and 2 times. A logarithmic growth structure was therefore introduced to curb linear expansion of word frequency.</li>
<li><strong>Harmonization of document length</strong>Long files naturally contain more words and require punishment. Use &#36;b&#36; Parameters are flexible in controlling the degree of integration and adaptation to different data sets.</li>
<li><strong>Keep Probability Model Foundation</strong>: IDF items still originate from Robertson-Sparck Jones<strong>Probability estimates for relevance</strong>It has a theoretical basis.</li>
</ul>
<p>BM25<strong>BM11/BM15</strong>with<strong>Query Extension</strong>(pseudo-relevance feedback) and<strong>Paragraph Search</strong>(passage retrieval) The effect is further enhanced to become the core algorithm of the Okapi system in TREC-3. Still.<strong>One of the most widely used search sequence algorithms in industry and academia</strong>。</p>
<h2>Embedding Technology Beginning: Word2Vec</h2>
<h3>Thought</h3>
<p>Word2Vec and &quot;Efficient Estimation of Word Representations in Vector Space&quot; It's one of the core creations in the modern NLP field. It proved, for the first time, that there was a great deal of semantic relations between words that could be learned through a simple neural network and that it was truly universal.&quot;Word Embedding&quot;This concept.</p>
<p>Word2Vec's core idea is:<strong>The meaning of a word depends on the words around it.</strong>I don't know. Word2Vec automatically learns the mathematical expression -- a dense vector -- of each word in the context of a large volume of text.</p>
<p>Word2Vec contains two main models:</p>
<ol>
<li>CBOW: Conjecture the central word from the context</li>
<li>Skip-gram: Conjecture context information based on the central word</li>
</ol>
<p>Word2Vec wants to learn more complex language meaning information from big data than the low dimensions (50-100-D) that were previously studied. I'm also concerned that word vectors can be captured.<strong>Multilevel Similarity</strong>(if &quot;big : bigger :: small : smaller&quot;And even...<strong>Semantic analogy</strong>(if &quot;King – Man + Woman ≈ Queen&quot;) and learning efficiency of models on mega-data sets.</p>
<h3>Model structure</h3>
<h4>Structure overview</h4>
<p>Overall objectives:
<strong>Input</strong>: a large language library (e.g. Google News, 6 billion words)
<strong>Output</strong>: Each word corresponds to one <strong>D Weighted Real Vector</strong>(e. g. 300-D) allows a syntax analogy where a semantic/syllable similarity is close in vector space and supports vector operation.</p>
<p>At the pre-processing stage, the removal of particularly low-frequency words (which are still maintained on a million-scale vocabulary) is intended to avoid data contamination by spelling errors. Each remaining word is assigned a single integer ID, which is used in subsequent narratives &#36;V&#36; is the number of high frequency words.</p>
<p>Word2Vec does not directly model language models, but will<strong>Context forecast tasks</strong>Convert<strong>Classification issues</strong>。</p>
<p><strong>For CBOW:</strong></p>
<ul>
<li><strong>Objective</strong>: Centre Word &#36;w_t&#36;</li>
<li><strong>Context</strong>: Other words in the window, e.g. &#36;w_{t-1},w_{t-2},w_{t+1},w_{t+2}&#36;(window size) &#36;C&#36;=2）</li>
<li><strong>Training sample</strong>：(context → target)</li>
</ul>
<p><strong>For Skip-gram:</strong></p>
<ul>
<li><strong>Objective</strong>: a word in the context, if &#36;w_{t+1}&#36;</li>
<li><strong>Input</strong>: Centre Word &#36;w_t&#36;</li>
<li><strong>Training sample</strong>：(center → context_word)</li>
<li>Skip-gram produces multiple samples for each central word (one for each context)</li>
</ul>
<p>Shared component of model structure: term vector matrix (Embeding Matrix)</p>
<ul>
<li>Define One <strong>&#36;V \times D&#36; Other Organiser &#36;W&#36;</strong>of which &#36;i&#36; Words. &#36;i&#36; vector means &#36;v_i&#36;。</li>
<li>This matrix is the ultimate learning.<strong>Word embeddings</strong>。</li>
<li>In CBOW and Skip-gram,&#36;W&#36; Both.<strong>Enter layer weight</strong>, or as an output layer weight.</li>
</ul>
<h4>CBOW</h4>
<p>Structure process:</p>
<ol>
<li><strong>Input Layer</strong>: context words &#36;w_{t-1},w_{t-2},w_{t+1},w_{t+2}&#36; one-hot vector</li>
<li><strong>Projection Layer</strong>：<ol>
<li>Each one-hot vector multiplied by &#36;W&#36;♪ Got a match ♪ &#36;D&#36; dimension vector.</li>
<li>These vectors<strong>Average</strong>(or peace) &#36;h&#36;, the structure of a wordbag discards the context order.</li>
<li>Note: Matrix used here &#36;W&#36; Finally, the words learned are embedded in themselves, because the input is actually one-hot encoded, and the product only needs to be used. &#36;W&#36; One line in.</li>
</ol>
</li>
<li><strong>Output Layer</strong>：<ol>
<li>Will &#36;h&#36; Enter to One <strong>softmax Catalog</strong>And predict the central word.</li>
<li>Use the matrix &#36;W^ before entering the Softmax classifier&#39;}&#36; 将 &#36;h&#36; 映射回 &#36;V.D.</li>
<li>Calculated projected losses for reverse dissemination.</li>
</ol>
</li>
</ol>
<p>Training objective: to maximize the logarithmic of the correct central words:
&#36;&#36;\mathcal{L} = \log P(w_t \mid w_{t-C}, \dots, w_{t-1}, w_{t+1}, \dots, w_{t+C})&#36;&#36;</p>
<h4>Skip-gram</h4>
<p>Structure process:</p>
<ol>
<li><strong>Input Layer</strong>: Centre Word &#36;w_t&#36; one-hot vector</li>
<li><strong>Projection Layer</strong>: multiplied by &#36;W&#36;♪ Got a match ♪ &#36;D&#36; Dimension vector &#36;v_t&#36;。</li>
<li><strong>Output Layer</strong>: For each context position &#36;w_{t+k}&#36; Use &#36;v_t&#36; Go predict the word, use the matrix.&#39;}&#36; 将 &#36;v_t&#36; 映射回 &#36;V.D. The losses projected in each context are combined for reverse dissemination as projected losses.</li>
</ol>
<p>Training objective: Maximum logarithmic
&#36;&#36;\mathcal{L} = \sum_{k=-C,, k \neq 0}^{C} \log P(w_{t+k} \mid w_t)&#36;&#36;</p>
<h4>On computing efficiency</h4>
<p>This is a neural network structure involving large-scale data. To ensure their calculability at the required scale, many adjustments to model structures and computational strategies are required.</p>
<p>In traditional neuronet language models (e.g. NNLM), the complexity of training is mainly derived from:</p>
<ul>
<li><strong>Non-linear hidden layer</strong></li>
<li><strong>Super-high Softmax Header</strong></li>
</ul>
<p>Even with the use of parallel training, it is difficult to extend to very large language material.</p>
<p>To that end, Word2Vec <strong>Moves unless a linear hidden layer becomes a log-linear Model</strong>I don't know. CBOW and Skip-gram <strong>Remove the hidden layer completely.</strong>, the projection layer (the average word vector or a single word vector) directly connects the output layer, i.e. transforms itself into a fully linear sorter, thus reducing the computational burden.</p>
<p>Word2Vec to deal with Softmax <strong>Use Hierarchical Softmax (horizontal Softmax)</strong>Give up the whole size of each sample. &#36;V&#36; The list of words (millions of degrees). It organizes the glossary as <strong>Huffman, fork tree.</strong>Change the projection mission to a series of root-to-leaf paths<strong>Class II (sigmoid)</strong>, the ultimate probability is the product of all sigmoids on the path, thereby reducing linear complexity to logarithmic complexity.</p>
<p>The very simple design framework runs throughout: whether CBOW or Skip-gram, the model structure is simple and the layers are extremely low, all to ease the counting efficiency bottlenecks. While benefiting from team strengths, Word2Vec uses Google internal <strong>DistBelief Distribution Framework</strong>, use multi-machine small-volume walk-through training and central server parameters to accelerate computing efficiency.</p>
<h3>Summary</h3>
<p>This article found:</p>
<ul>
<li>Increased amount or dimension of data enhances performance, but exists<strong>Marginal benefits diminishing</strong>In order to continuously upgrade model capabilities, both need to be upgraded. This finding also guides the current training of large language models.</li>
<li>Skip-gram significantly outperformed other models in semantic tasks; CBOW was slightly superior in terms of terms, predicting other words from the central word to make models learn better semantic information.</li>
<li>The dense vectors obtained by embedding these instruments allow semantic analogy, excavating information through simple calculations between vectors, but there are certain errors.</li>
<li><strong>Simple model (CBOW/Skip-gram) to efficiently train high quality word vectors</strong>, and could continue to expand almost indefinitely to any task requiring text embedding.</li>
</ul>
<p>Word2Vec<strong>It's an era of pre-training words vectors.</strong>, guided the development of many NLPs, their Open Source Toolkit <code>word2vec</code> It has become an essential tool for NLP missions. The great thing about Word2Vec is:<strong>With a very simple structure + big data, it reveals the deep structure of the language.</strong>I don't know. It proves.&quot;Simple Model + Big Data&quot;It can go beyond that.&quot;Complex Model + Small Data&quot;This idea profoundly affects the whole AI field.</p>
<h2>Transformer</h2>
<h3>Attention, Bert and GPT</h3>
<p>The Transformer structure presented in the 2017 paper “Attention Is All You Need” led to two different pre-training paradigms. They address the context in different ways and collectively redefine modern NLPs. For a detailed discussion of the self-directional mechanism <a href="/en/blog/2024/11/14/self-attention-and-transformer-architecture/">Self-care mechanisms and Transformer</a>。</p>
<hr>
<p>BERT (Pre-training of Deep Bidirectional Transports for Language University) is the beginning of a deep two-way understanding of language models, where a real two-way context is embedded. BERT is a radical paradigm revolution that solves the dilemma of static embedding that cannot deal with the multidimensional meaning of the word.</p>
<p>Its core self-care mechanism allows the model to handle a word while obtaining context information across the board. The key innovation is the masked language model (MLM): random in the sentence&quot;Query&quot;It also allows the model to project according to the two-way context, forcing the model to integrate the context in depth and generate dynamic, context-sensitive word vectors. BERT's representative.&quot;Pre-training -- fine-tuning&quot;The paradigm still dominates the entire NLP field.</p>
<hr>
<p>GPT is another technical route with Bert. It uses &quot;Decoder-only&quot; The Transformer structure with the goal of generating a coherent text. In order to predict the next word accurately, the GPT model forms a deep internal understanding and efficient expression of the above (left context).</p>
<p>The success of GPT has also revealed the Scaling Laws — the larger the models, the larger the data and the larger the resources, the greater the capabilities of the models, including their representational capabilities. Its strong capacity for generation, in turn, has generated a huge demand for high-quality retrieval systems (i.e. RAG).</p>
<h3>BERT Embedding</h3>
<p><strong>BERT type model</strong>Embeding through <code>[CLS]</code> Mark or average pool acquisition.<code>[CLS]</code> The tag itself does not contain any semantics, and when the sequence flows through multiple layers Transformer Encoder, the self-care mechanism at each level allows <code>[CLS]</code> Mark it.&quot;Attention&quot;all other words in the sequence. Through this two-way exchange of information,<code>[CLS]</code> Marked vectors are trained to be able to<strong>Absorption and summary of semantic information for the entire input sequence</strong>。</p>
<p>In the last hidden layer of the model output (lower layer is the restoration to high-dimensional execution Softmax), directly removed<strong>With input <code>[CLS]</code> The final hidden state vector corresponding to the mark</strong>, a 768-dimensional vector is considered to be Embeding the entire input text. Or get the last output of the model by average pool.<strong>All Characters</strong>to hide the state vector, to average the Embedding vector for all elements.</p>
<h3>GPT Embedding</h3>
<p>Use with BERT in generating embedded <code>[CLS]</code> The most natural and efficient way to create a GPT series model is to embed it.<strong>The hidden status of the last mark in the input series</strong>I don't know. The logic behind it is closely linked to its goal of self-return training. In embedded tasks of GPT models, except for the beginning <code>&lt;|begin_of_text|&gt;</code>It usually adds at the end. <code>EOS</code> That's... <code>&lt;|end_of_text|&gt;</code> Mark.</p>
<p>The hidden status of each tag is updated when the sequence passes the Transformer layer of GPT. But the last word is the only one that can.&quot;Yeah.&quot;The whole input sequence (from <code>&lt;s&gt;</code> Present. <code>token_n</code>. The last character is the end point of the entire information stream and is the final carrier of all context information.</p>
<p>The embedded vector of the sentence is from the model.<strong>Last Hidden Layer</strong>In the hidden layer before returning to the high-dimensional Softmax, extract<strong>Last Input Tag</strong>Whether it's the original last word, or... <code>EOS</code> Marks the corresponding hidden state vector. For GPT-2, it's also 768D.</p>
<p>Embedded models based on GPT structures allow commands to control embedded generation and adapt it to different downstream tasks, that is,<strong>Command fine-tuning</strong>I don't know. It's based on the Bert structure that can't do it. Allowing instructions to fine-tune means allowing<strong>A model is used for a variety of different types of downstream tasks.</strong>It is no longer necessary to train a dedicated model for each mission. Finally.<strong>The GPT scale method to improve performance</strong>。</p>
<h2>Qwen3 Embedding</h2>
<p>This section summarizes the technical report that was uploaded by the Qwen team in June 2025 Qwen3 Embeding: Advancing Text Embeding and Reraning Through Foundation Models. The Qwen3 Embeding model (8B, 4B, 0.6B) is ranked 2-4 on the MTEB Integrated List, after the unknown parameter Gemini Embeding 001. The whole series of models has followed the Apace 2.0 license open source.</p>
<h3>Introduction</h3>
<p>Text embedding and reordering is an important part of the NLP task, and high-quality semantic embedding is the basis for many important tasks, such as RAG and Agens. Despite notable progress, embedded and re-sorting models where training is well performed in terms of scalability, context understanding and alignment with specific downstream tasks remain challenging.</p>
<p>This paper introduces the Qwen3 Embedding series of models based on the Qwen3 Foundation model, making full use of their powerful multilingual text understanding and generation capacity to unleash their potential in training embedded models and reordering models.</p>
<p>In order to train in embedded models, a multi-stage training process was implemented: large-scale unsupervised pre-training, and monitored fine-tuning of high-quality data sets. Model Merge was used to integrate different checkpoints in order to enhance modelity and generalization. The Qwen3 directive model was used to synthesize training data, with a high-quality component for the second phase of supervisory training.</p>
<p>A similar two-stage training programme was used for the reordering model. Three embedded and reordered models (8B, 4B, 0.6B) were eventually released based on the Qwen3 model of different scales. To facilitate applications in downstream missions, the Qwen3 Embeding series model supports elastic dimension expression and customisation instructions.</p>
<h3>Model structure</h3>
<p>The core idea of embedded and reordered models is to assess relevance in a mission perception manner. For queries &#36;q&#36; and documents &#36;d&#36;, the model needs to follow instructions &#36;I&#36; To assess the correlation between the two. To this end, the model uses the following forms of data for training:
&#36;&#36;{ I_i, q_i, d_i^+, d_{i,1}^-, \dots, d_{i,n}^- }&#36;&#36;
Separately indicates the command, the query, the relevant unrelated document. Training to cover various types of similar data pairs will expand their performance in downstream tasks in different fields.</p>
<p><strong>Structure</strong>: Qwen3 Embedded and Reordered Models are based on a dense version of the Qwen3 Foundation Model, using Dense version and containing three parameter sizes, and initializing with pre-trained parameters to use the capabilities they have acquired in large-scale pre-training.</p>
<p><strong>Embedded Model</strong>: Add one at the end of the input series using a large language model with causal attention mechanisms <code>EOS</code> Mark. The final embedded vector corresponds to the last layer. <code>EOS</code> Marks the hidden status generation.</p>
<p>To ensure that embedded vectors follow instructions in downstream missions, will<strong>Command & Query Spell</strong>as a single input context, and<strong>Document part remains unchanged until sent to LLM processing</strong>I don't know. The input sequence for command query is: <code>{Instruction} {Query} &lt;|endoftext|&gt;</code>, where <code>&lt;|endoftext|&gt;</code> Qwen series is always used <code>EOS</code> Mark. The document is <code>{Doc} &lt;|endoftext|&gt;</code>。</p>
<p><strong>Reorder Model</strong>: For a more accurate assessment of text similarities, the large-language model (LLM) is used in a single context <strong>point-wise</strong> is reordered. Similar to embedded models, enter the instructions in the context to enable compliance. Use standard dialogue templates and model similar assessment missions as a question of classification. LLM input follows the following template:</p>
<pre><code class="language-text">&lt;|im_start|&gt;system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be &quot;yes&quot; or &quot;no&quot;.&lt;|im_end|&gt;
&lt;|im_start|&gt;user
&lt;Instruct&gt;: {Instruction}
&lt;Query&gt;: {Query}
&lt;Document&gt;: {Document}&lt;|im_end|&gt;
&lt;|im_start|&gt;assistant
&lt;think&gt;\n\n&lt;/think&gt;\n\n
</code></pre>
<p>The next word is assessed to be &quot;yes&quot; or &quot;no&quot; Possible. The mathematical expression is as follows:
&#36;&#36;\text{score}(q, d) = \frac{e^{P(\text{yes} \mid I, q, d)&#125;&#125;{e^{P(\text{yes} \mid I, q, d)} + e^{P(\text{no} \mid I, q, d)&#125;&#125;&#36;&#36;
Reordering model does not use the embedded vector of the last layer, but allows the model to run the next word projection of the last layer, generating a dimension equal to that of the vocabulary <strong>logits vector</strong>, then give this vector to the Softmax layer for a probability distribution. The probability of both words is what we need.</p>
<p>The Qwen3 Text Embedding model supports MRL (Matryoshka RepresentationLearning) and allows adjustments to the dimensions of the embedded vector. The Text Relanging model does not address this function because it does not involve an output embedded vector. They all support the implementation of an integrated Aware that is embedded or reordered on the basis of command dynamics. Models of all sizes support 32K context, with slightly different layers.</p>
<h3>Training and evaluation</h3>
<p>This section describes the multi-stage training process used and describes the key elements of the training programme, including training objectives, training data synthesis and the screening of high-quality training data.</p>
<h4>Training objectives</h4>
<p>For embedded models, an improved version based on the InfoNCE comparative learning loss framework is used. For a size of &#36;N&#36; , whose loss function is:
&#36;&#36;L_{\text{embedding&#125;&#125; = -\frac{1}{N} \sum_{i}^{N} \log \frac{e^{s(q_i, d_i^+) / \tau&#125;&#125;{Z_i}&#36;&#36;
of which &#36;s()&#36; is a function used to calculate similarities, using cosine similarities;&#36;\tau&#36; is the super-parameter for temperature control;&#36;Z_i&#36; is a normalized factor that combines the similarity scores of positive and negative samples:
&#36;&#36;Z_i = e^{s(q_i, d_i^+) / \tau} + \sum_{k}^{K} m_{ik} e^{s(q_i, d_{i,k}^-) / \tau} + \sum_{j \neq i} m_{ij} e^{s(q_i, q_j) / \tau} + \sum_{j \neq i} m_{ij} e^{s(d_i^+, d_j) / \tau} + \sum_{j \neq i} m_{ij} e^{s(q_i, d_j) / \tau}&#36;&#36;</p>
<p>The parameters are as follows, in order from left to right:</p>
<ul>
<li>&#36;d_i^{+}&#36; indicates that the correct document is paired and the positive document corresponding to this query is measured;</li>
<li>&#36;K&#36; and &#36;d_{i,k}^-&#36; represents the number of negative documents and the number of negative documents to which this query corresponds;</li>
<li>&#36;q_j&#36; represent other queries in the bat, measure this query and other queries in the bat;</li>
<li>&#36;d_j&#36; represent other documents in the bat, measure positive documents and other documents;</li>
<li>The last item does not involve new parameters, but only measures the relationship between this query and other documents.</li>
</ul>
<p>Mask Factor &#36;m_{ij}&#36; With a view to mitigating the effects of false negatives, the definition is as follows:
&#36;m=
\begin{cases}
CC BY-NC-ND 2.0 &amp; \text{if } s_{ij} &gt; s(q_i, d_i^+) + 0.1 \text{ or } d_j == d_i^+, \
1 &amp; \text{otherwise}
I'm sorry.
of which &#36;s_{ij}&#36; is the correlation factor between the query and the document.</p>
<p>For reordering models, optimized oversight fine-tuning (SFT) loss functions are defined as follows:
&#36;&#36;L_{\text{reranking&#125;&#125; = - \log p(l \mid \mathcal{P}(q, d)),&#36;&#36;
of which &#36;p(\cdot \mid *)&#36; This indicates the probability of being allocated by the Large Language Model (LLM). Label &#36;l&#36; For regular documents as &quot;yes&quot;, for the negative document &quot;no&quot;I don't know. The loss function encourages the model to assign a higher probability to the correct label, thereby increasing the ranking performance.</p>
<h4>Multi-stage training</h4>
<p>Multi-stage training is a common practice for training text embedding models. This strategy usually begins with initial training on large-scale semi-oversight data containing noise, followed by fine-tuning using smaller but high-quality monitoring data sets.</p>
<p>The large-scale weak monitoring training data contribute significantly to the model ' s ability to extend, while the fine-tuning of subsequent phases using high-quality data further enhances model performance. A combination of two-step processes will result in embedded models that have a higher capacity for generalization and performance.<strong>Note that the training process of the reordering model does not include weak monitoring training.</strong></p>
<p>Building on the existing multi-stage training framework, the Qwen3 Embeding series introduced the following innovations:</p>
<ul>
<li><strong>Weak surveillance training driven by large-scale synthetic data</strong>: Directly synthesized into data pairs using the underlying model's strong text understanding and generation capability. This method allows flexibility in defining the multiple dimensions of the data required in a synthetic reminder, such as task type, language, length and difficulty. Data synthesis driven by basic models is more manageable than data collection from open-area sources.</li>
<li><strong>Application of quality synthetic data in monitoring fine-tuning</strong>: The excellent performance of the Qwen3 base model has resulted in significantly higher quality of synthetic data, further enhancing the overall performance and generalization of the model.</li>
<li><strong>Model integration</strong>: Following completion of the supervisory fine-tuning, model integration techniques based on meta-line interpolation (SLERP) are used. The technology combines multiple models kept during fine-tuning with checkpoints, which are designed to enhance the robustness and generalization of models in different data distributions.</li>
</ul>
<h4>Synthetic Data Set</h4>
<p>In order to build a robust synthetic data set for training models to perform various similar tasks, we have produced diversified text pairs covering search, bilingual excavation, classification and semantic text similarities (STS). The quality of these synthetic data is ensured by data synthesis using the Qwen3-32B model as the base model.</p>
<p>We have designed a variety of warning strategies to enhance the richness and authenticity of the data generated: assigning specific roles, and simulated scenarios of potential users searching the document. This way of infusing a user perspective increases the diversity and relevance of synthetic queries. The hint template also incorporates dimensions such as type of query (keyword type, fact type, summary type, judgement type), length of query, difficulty and language. This multi-dimensional design ensures the quality and diversity of synthetic data.</p>
<p>Ultimately, approximately 150 million groups of weak monitoring training data were created. Experimental results show that embedded models trained in the use of these synthetic data perform very well in downstream assessments, significantly exceeding many previous monitoring models.</p>
<p>For the second stage of training, a simple cosine similarity calculation method is used to screen data pairs: preserve a cosine similarity greater than 0.7 from random sampling data. Some 12 million high-quality monitoring training data were eventually selected for use in follow-up training.</p>
<h4>Evaluation</h4>
<p>We made a comprehensive and fair assessment of the Qwen3 Embeding model on multiple benchmark tests. For text embedded models, use<strong>Large-scale Multilingual Text Embedding Benchmark Test (MMTEB)</strong>, covering over 500 quality control assessment tasks in more than 250 languages.</p>
<p>In addition to the traditional text tasks (e.g., similarity of various types of search, classification and semantic text), MMTEB also contains a series of challenging and novel tasks, such as command compliance, long-document search and code retrieval, which are currently the largest and most widely covered assessment tasks in the model field.</p>
<p>of which <strong>Qwen3-Reranker-8B best performed in most missions</strong>, the overall performance is only slightly behind the latest version of Gemini-Embeding.</p>
<h3>Conclusions</h3>
<p>The digestion experiment showed that<strong>Large-scale and weak pre-training stages of oversight are critical to achieving excellence</strong>，<strong>Model integration is also a key element in building strong models.</strong>。</p>
<p>This technical report officially publishes the Qwen3-Embeding series, a comprehensive text embedded and reordered model based on the Qwen3 base model. These models are designed to perform well in the embedded and reordered tasks of various text types and cover multiple scenarios such as multilingual retrieval, code retrieval and complex command compliance.</p>
<p>The Qwen3-Embeding model is based on a robust multi-stage training process that combines large-scale weak monitoring pre-training on synthetic data with monitored fine-tuning models on high-quality data sets. In doing so, the Qwen3 Large Language Model has played a key role in synthesizing diversified training data across multiple languages and missions, thus effectively enhancing the capabilities of the Model.</p>
<p>A comprehensive assessment showed that the Qwen3-Embeding model achieved SOTA performance in MTEB, CMTEB, MMTEB and several other search baseline tests. The model follows the Apache 2.0 license open source in GitHub.</p>
<h2>Appendix</h2>
<p>The appendix contains additional relevant information on the Attention Mechanism, Transformer, BERT, GPT, citing notes from other parts as appendices.</p>
<h3>Appendix A: Focusing Mechanisms</h3>
<p>See below for details of the self-care mechanism. <a href="/en/blog/2024/11/14/self-attention-and-transformer-architecture/#%E8%87%AA%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6">Self-care mechanisms</a>。</p>
<h3>Appendix B: Transformer Architecture</h3>
<p>For more details about the Transformer structure, see <a href="/en/blog/2024/11/14/self-attention-and-transformer-architecture/#Transformer%E6%9E%B6%E6%9E%84">Transformer Structure</a>。</p>
<h3>Appendix C: Self-supervised Learning</h3>
<p>For details on self-monitoring learning see <a href="/en/blog/2024/11/14/self-attention-and-transformer-architecture/#%E8%87%AA%E7%9B%91%E7%9D%A3%E5%AD%A6%E4%B9%A0">Self-supervised learning</a>。</p>
