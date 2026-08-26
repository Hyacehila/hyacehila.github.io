---
title: 'From LLM to VLM: How Language Models Learn Visual Understanding'
title_zh: 从 LLM 到 VLM,语言模型如何实现视觉理解
date: 2026-01-20 00:00:00 +0800
categories:
- Foundation Models
- Model Mechanics
tags:
- Vision-Language Models
- Multimodality
- Model Architecture
- Survey
author: Hyacehila
excerpt: A technical overview of the path from language-only models to vision-language models, covering CLIP, VLMs, diffusion
  guidance, and native multimodal tokenization.
description: A technical overview of the path from language-only models to vision-language models, covering CLIP, VLMs, diffusion
  guidance, and native multimodal tokenization.
excerpt_zh: 梳理从纯语言模型到视觉-语言模型的技术路线，说明 CLIP、VLM、扩散模型与原生多模态在输入表示、训练目标和推理方式上的差异。
permalink: /blog/2026/01/20/from-llm-to-vlm-visual-understanding/
lang: en
translation_key: 2026-01-20-from-llm-to-vlm-visual-understanding
translation_status: machine
translation_source_hash: c49ed770e55617ae4bf707c80e14b5f778c5c337d9ba5165d021ad9f143e6524
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<p>The development of large polymodular models is broadly along a technical path: Starts with a pure language LLM, and then CLIP achieves visual-linguistic alignment, then produces VLM, and a more original multi-modular modelling. This paper is a synthesis of this technology and analyses the underlying principles and architecture of the various models.</p>
<p>The questions in this article can also be addressed<a href="/en/blog/2026/06/23/joyai-vl-interaction/">JoyAI-VL-Interaction: Return from Chat to a continuous interactive visual language model</a>、<a href="/en/blog/2026/04/30/deepseek-ngram-poor-is-good/">The difference is good: starting with DeepSeek not using n-gram structures</a>How the concept of a relatively close read together is developed in different contexts.</p>
<p>The paper contains three main parts:<strong>From plain text to multiple mosaics</strong>(LLM CLAIP PLAYING TECHNOLOGY PLAYING)<strong>Progress topic</strong>(scatter models and primary polymorphs), and<strong>Appendix</strong>(Basic conceptual supplements, including Cross Attention, Cross-Cyber, Softmax, etc.).</p>
<h2>From plain text to multiple mosaics</h2>
<p>Multi-modular AI development is not a one-off exercise. More commonly, the route is to train text models and then introduce visual encoders, cross-modular alignment layers and generation interfaces.<strong>Each phase is re-energized by a part of the previous phase ' s capacity, while at the same time exposing new interface problems.</strong></p>
<p><strong>Evolution Path</strong>:</p>
<pre><code class="language-mermaid">graph LR
    A[纯语言LLM&lt;br/&gt;基础] --&gt;|扩展视觉编码能力| B[CLIP&lt;br/&gt;视觉-语言对齐]
    B --&gt;|结合LLM的生成能力| C[生成式VLM&lt;br/&gt;多模态理解与生成]
</code></pre>
<p>This section will dismantle the technical rationale of these three phases, highlighting the inheritance relationship and evolutionary logic between them.</p>
<h3>Phase 1: Pure language model LLM - a single-modular intellectual engine</h3>
<p><strong>LLM is the starting point for understanding the subsequent multi-modular architecture.</strong>I'm sorry. See how the language model handles token, how to generate text, and then see how CLIP and VLM can access visually.</p>
<p>LLM is<strong>Generating models that process text molluscs only</strong>, using Transformer Decoder-only structures, learn language distribution through extensive pre-text training. The common training goal is to improve mission performance by adjusting to human alignment by following instructions for the next Token (Next Token Protection) from the regression prediction.</p>
<p>But the input interface limits for LLM are clear:<strong>It can only process text directly and cannot receive multiple mosaics of images, audio, etc.</strong>I'm sorry. So multi-modular extension starts with a engineering question:<strong>How to get LLM &quot;Yeah.&quot;Images?</strong> Two technical routes emerged as a result:</p>
<ol>
<li><strong>CLIP route</strong>Build visual-linguistic alignment to embed images and text into the same vector space</li>
<li><strong>VLM route</strong>: Load LLM directly&quot;Eyes.&quot;Let it see and speak.</li>
</ol>
<h3>Phase 2: CLIP - Building a bridge between visual and linguistic</h3>
<p>CLIP (Contrastive Language-Image Pre-training)<strong>Aligning visual and language</strong>It's a reusable embedded problem. It is not a model that is generated, but rather embedded: images and text are mapped into the same vector space and compared by similarity.</p>
<p><strong>Relationship with LLM</strong>:</p>
<ul>
<li>CLIP.<strong>Text Encoder</strong>The Transformer structure that directly inherited LLM.</li>
<li>CLIP reuses LLM text understanding skills but migrates them to visual-linguistic alignment tasks</li>
<li>CLIP's training target is from&quot;Forecast the next Token.&quot;Change to&quot;Zitru Dock&quot;</li>
</ul>
<h4>Core positioning: visual-linguistic alignment model</h4>
<p>CLIP can be understood as<strong>Embedding Model</strong>I'm sorry. It maps images and text into the same low-dimensional vector space (insulated space) and allows for a close text to be compared to the space.</p>
<p><strong>Problems addressed</strong>:</p>
<ul>
<li>Pure visual models (e.g. ResNet) can only output fixed category labels and cannot understand natural languages</li>
<li>Pure Language Model Not Available&quot;Yeah.&quot;Image</li>
<li>CLIP provides embedded interfaces between visual and language, and achieves a symmetric alignment across the mosaics</li>
</ul>
<h4>Structure design: Double-Encoder</h4>
<p>CLIP uses&quot;Double Tower&quot;The design is straightforward - the visual encoder and the text encoder work independently and do not interfere with each other, but only interact at the end through similarity calculations.</p>
<h5>Visual Encoder</h5>
<p><strong>Foundation</strong>: usually used <strong>ViT (Vision Transformer)</strong> Series (e.g. ViT-L/14 or ViT-g)</p>
<p><strong>Workflow</strong>:</p>
<ol>
<li><strong>Image Segment</strong>: Split input images &#36;N&#36; Small squares, for example &#36;16 \times 16&#36; Pixels</li>
<li><strong>Spreading and Maping</strong>: Each Patch is set to a vector by linear mapping</li>
<li><strong>Add Location Encoding</strong>: Add location information for each Patch</li>
<li><strong>Transformer Process</strong>: feature extraction and interaction through multilayer Transformer</li>
<li><strong>Output Sampling</strong>: Take special <strong>[CLS] Token</strong> The output vector is expressed as a global visual expression</li>
</ol>
<p><strong>Output</strong>: A person who represents the semantics of the image<strong>Global Characteristic Vector</strong>, the dimensions are as follows: &#36;d=512&#36;</p>
<h5>Text Encoder</h5>
<p><strong>Foundation</strong>: Standard Transformer Encoder()<strong>Inheritance from LLM Structure</strong>)</p>
<p><strong>Workflow</strong>:</p>
<ol>
<li><strong>Interword</strong>: text is divided into Token sequences by Tokenizer, if <code>[SOS, A, dog, is, running, EOS, PAD...]</code></li>
<li><strong>Embedded</strong>: Convert each Token to a vector</li>
<li><strong>Transformer Process</strong>: Multilayer Self-Attention allows every Token&quot;Yeah.&quot;Context</li>
<li><strong>Output Sampling</strong>: Take <strong>[EOS] Token</strong>(End of Security)</li>
</ol>
<p><strong>Output</strong>: a word for the whole sentence<strong>Global Text Vector (Text Embeding)</strong></p>
<h5>Alignment mechanism</h5>
<p>The two towers work independently.<strong>There is no complex cross-mode interaction layer.</strong>I'm sorry. They only calculate two vectors at the end.<strong>Cosine Similarity</strong>Interactive.</p>
<p>&#36;&#36; \text{Similarity} = \cos(\theta) = \frac{v \cdot t}{|v| \cdot |t|} &#36;&#36;</p>
<p>of which &#36;v&#36; The video shows the visual embedding of the vector, which is a very important example of the visualization of the vector.&#36;t&#36; is the text embedded vector (Text Embedding).</p>
<h4>Training target: Comparative Learning</h4>
<p>The main change in the CLIP is the way in which it is trained:<strong>Comparative learning</strong>I'm sorry. Its objectives are clear:<strong>Let the matching text close to the distance, and the mismatch text push away.</strong>。</p>
<h5>InfoNCE Los (standard practice of the CLIP)</h5>
<p><strong>Basic thinking</strong>: In a Batch, the text matching is considered a multi-classification problem.</p>
<p><strong>scene settings</strong>: Suppose there's a battling &#36;N&#36; Yes, it is. &#36;(I_1, T_1), (I_2, T_2), ..., (I_N, T_N)&#36;I'm sorry. of which<strong>Positive sample</strong>It's a pair of horn lines. &#36;(I_i, T_i)&#36;, i.e. a matching text pair; and<strong>Negative sample</strong>It's for pictures. &#36;I_i&#36;The rest of the same Batch. &#36;N-1&#36; Text &#36;(T_j, j \neq i)&#36;。</p>
<p><strong>Calculating Process</strong>: first calculate the two-cosine similarity of all the text vectors, the composition Construction &#36;N \times N&#36; is the likeness matrix. For the first &#36;i&#36; An image that is matched to the correct text by subdivisioning its similarity to all text into a probability distribution through the Softmax function &#36;T_i&#36; and predict the probability. The loss function uses the classic multi-category cross-brenade loss:</p>
<p>&#36;&#36; L_i = -\log \frac{\exp(sim(I_i, T_i) / \tau)}{\sum_{j=1}^{N} \exp(sim(I_i, T_j) / \tau)} &#36;&#36;</p>
<p><strong>Symbolic Interpretation</strong>:</p>
<ul>
<li>&#36;sim(I_i, T_i)&#36;: No. &#36;i&#36; Chart and &#36;i&#36; Similarity of paragraph text</li>
<li>&#36;\tau&#36; (Tau):<strong>Temperature coefficient (Temperature)</strong>Controlling distribution&quot;Pointy&quot;<ul>
<li>&#36;\tau&#36; Small: fine differences in similarity are magnified and models focus on the most difficult negative samples to distinguish</li>
<li>&#36;\tau&#36; Large: distributions become smoother</li>
</ul>
</li>
<li>&#36;\exp(\cdot) / \sum \exp(\cdot)&#36;: Softmax function, converts a similar value to probability</li>
</ul>
<p><strong>Basic logic</strong>:
InfoNCE forced the model to &#36;N&#36; options&quot;Pick out&quot;Right about that. For positive samples, the gradient brings the two in close proximity; for negative samples, the gradient keeps the two away.</p>
<p><strong>Limitations</strong>:
InfoNCE relies on Batch Size. The bigger the Batch, the more negative the sample, the harder the mission, the better the characteristics the model learned. If the bat is small, the model is easy to guess randomly.</p>
<h5>SigLIP (Sigmaid Los improved)</h5>
<p>SigLIP is a Google-driven improvement that is based on the idea of translating the issue of multi-classification into one of the most important ways to improve the situation of the population. &#36;N \times N&#36; (b) A separate issue of two classifications. Unlike InfoNCE, SigLIP no longer relies on Softmax for global integration, thus avoiding the calculation of the denominator and the resulting communication costs in distributed training.</p>
<p><strong>Algorithmic principles</strong>: For each element in the likeness matrix &#36;(i, j)&#36;, separate target labels depending on whether they are positive samples. When? &#36;i=j&#36; ♪ When the positive sample is right, the label ♪ &#36;y_{ij}=1&#36;, expect Sigmoid output close to 1; &#36;i \neq j&#36; ♪ When negative samples are right, label ♪ &#36;y_{ij}=0&#36;Sigmoid output is expected to be close to zero. Loss function defined as:</p>
<p>&#36;&#36; L = - \frac{1}{N} \sum_{i}\sum_{j} \left[ y_{ij} \log \sigma(z_{ij}) + (1-y_{ij}) \log (1-\sigma(z_{ij})) \right] &#36;&#36;</p>
<p><strong>Symbolic Interpretation</strong>：</p>
<ul>
<li>&#36;N&#36;: Number of graphics in the Batch</li>
<li>&#36;y_{ij}&#36;: Second classification label,&#36;y_{ij}=1&#36; Which means positive samples are correct. &#36;(i=j)&#36;，&#36;y_{ij}=0&#36; The negative sample is right. &#36;(i \neq j)&#36;</li>
<li>&#36;z_{ij}&#36;: Section &#36;i&#36; Chart and No. &#36;j&#36; Similarity fractions of paragraph text</li>
<li>&#36;\sigma(\cdot)&#36;: Sigmoid function, replacing multi-classical Softmax</li>
</ul>
<p>This approach has two direct advantages: first, because the total sum of the synchronized denominator between GPUs is not required to support super-large-scale Batch training (e.g. 32k scale); and second, because in the same size of the model, SigLIP is generally better performing than InfoNCE in the zero sample classification task.</p>
<h4>CLIP embedded space alignment and capability boundaries</h4>
<p>CLIP has acquired two types of common-purpose capability by building a unified visual-linguistic embedded space:<strong>Zero sample classification</strong>and<strong>Cross-modular Search</strong>。</p>
<p>In the case of zero sample classification tasks, CLIP does not need to be trained for specific categories, but simply converts the category name into a natural language description (see also the table below). Like&quot;A photo of the {class}&quot;) to calculate the similarity of the image to the description of the text in the various categories, and to complete the classification. It's a very low cost of cross-model access, either.&quot;Search the text.&quot;Or is it?&quot;Search the map in writing.&quot;, the CLIP can complete vector point count at milliseconds.</p>
<p>However, this embedded space also has a cost. CLIP encoders tend to remain under a comparative learning drive<strong>Global Semantics</strong>, and loses a portion of the details (e.g. OCR text, number of objects, precise space location). Training objectives are already met by a broadly matched text.</p>
<p>Another limitation is that the CLIP is only encoder, not coder, so it's not possible to execute it.&quot;Forecast the next word&quot;. It can judge.&quot;Do you think so?&quot;、&quot;Isn't that right?&quot;But not directly.&quot;Say it.&quot;Image content and lack of expertise in generating complex scene descriptions. VLM addresses this interface gap: after the visual features are aligned, they are sent to the language model, which generates text.</p>
<h3>Phase 3: Generating VLM - Putting eyes on LLM</h3>
<p>CLIP's settled.&quot;Visual-Language Alignment&quot;Problem, but it can't.&quot;Talk.&quot;I'm sorry. The generation interface is supplemented by the generation VLM (Visual Language Model):<strong>Give LLM access visual features so that it can generate text answers around images</strong>。</p>
<p>Representative models include LLAVA, InstractBLIP, Qwen-VL, MiniGPT-4, which usually extract features from CLIP visual capabilities and then align them through layers&quot;Translation&quot;LLM is given the final opportunity to produce text using LLM ' s expression skills, thus achieving detailed description, reasoning and question and answer questions on the content of the image.</p>
<p><strong>Succession to LLM and CLIP</strong>:</p>
<ul>
<li>VLM's.<strong>LLM directly to the bottom of the language</strong>(e.g., Vicuna, Qwen-7B) inherits its powerful linguistic expression skills</li>
<li>VLM's.<strong>Visual encoder directly reuses CLIP</strong>(e.g. CLIP VIT-L/14) to inherit the visual manifestations they have learned</li>
<li>VLM's innovation is that<strong>Align Layer</strong>, the visual features of the CLIP&quot;Translation&quot;To understand the LLM.</li>
</ul>
<h4>Structure design: three-part structure</h4>
<p>Generating VLM&quot;Visual encoder - Language Model - Modular Aligning Layer&quot;The three-part structure with clear division of labour and synergies between components.</p>
<p><strong>Visual Encoder<strong>The design philosophy of the image characterization is:</strong>Reuse pre-trained CLIP/SigLIP</strong>(Freezing parameters). CLIP has mastered excellent visual manifestations through comparative learning, and direct re-use can save the process of starting training while lowering training costs and preventing catastrophic oblivion. Early on, Qwen-VL adopted CLIPVIT-based structures, and the InstractBLIP used ViT-g/14, VLM directly inherited CLIP's visual encoder without having to retrain visual representation.</p>
<p><strong>Language Model Bottom (LLM Backbone)<strong>The philosophy of design is the same.</strong>Reuse Plain Language LLM</strong>(e.g., Vicuna, Qwen-7B). Pure LLM has acquired powerful linguistic expressions, logic and world knowledge, and direct reuse can quickly acquire multimodular capabilities and maintain original pure text capability. LLM receives visual features (as special&quot;Visual Token&quot;) and text Prompt, self-regression generation to predict the next Token. The use of Vicuna (a fine-tuned Llama), the use of Qwen-7B by Qwen-VL, and the use of Qwen-7B by LaVA are also used by Vicuna, where VLM directly re-references LLM pre-training weights, which inherits its linguistic understanding and earning capacity.</p>
<p><strong>Model Aligning Layer (Adapter/Projector)</strong> is an important interface in the VLM architecture, as&quot;Translator&quot;It's for visual features.&quot;Translation&quot;The language that you can understand is the language of the LLM. Two mainstream design concepts currently exist, representing direct mapping and fine extraction of technical routes.</p>
<p><strong>Option I: LLAVA (Linear/MLP Production) — Simple and efficient</strong></p>
<p>LLAVA uses the most direct two-tier MLP structure (Linear →Gelu →Linear), the underlying assumption being that visual features and language are embedded in a linear space to be matched.</p>
<ul>
<li><p><strong>Architecture</strong>:</p>
<ul>
<li>Visual encoder: CLIP Vit-L/14 (freezing parameters)</li>
<li>Align: two floors MLP</li>
<li>LLM: Vicuna/Llama (blip, Full Fine-tanning or LoRA)</li>
</ul>
</li>
<li><p><strong>In the process of deduction</strong>:</p>
<ol>
<li>Picture input ViT, output all Patch feature sequences&#36;H_v&#36;(if&#36;576 \times 1024&#36;)</li>
<li>&#36;H_v&#36;After alignment, the dimensions are mapd from 1024 to 4096 in LLM.</li>
<li>Output vector &#36;H&#39;v&#36; considered&quot;Visual Token&quot;</li>
<li>Text Prompt was converted to Embeedding by Tokenize&#36;H_t&#36;</li>
<li>Will &#36;H&#39;<em>v&#36;和&#36;H_t&#36;拼接：&#36;[\text{Token}</em>{\text{Visual&#125;&#125;, \text{Token}_{\text{Text&#125;&#125;]&#36;</li>
<li>Throw it to LLM for Next Token Prevention</li>
</ol>
</li>
<li><p><strong>Reverse dissemination mechanisms</strong>:</p>
<ol>
<li>Calculating Cross-Entropy Loses</li>
<li>Gradient from Los →LLM →ScopeMLP →VIT</li>
<li>Gradient to output end stopped due to the ViT freeze</li>
<li>Only update MLP and LLM parameters</li>
</ol>
</li>
</ul>
<p><strong>Option II: InstractBLIP (Q-Former) - fine extraction pie</strong></p>
<p>InstractBLIP considers that throwing all the Patches directly to LLM is too redundant and slow to calculate, and therefore uses Q-Former (Querying Transformer) structures to achieve compression and dynamic extraction of information.</p>
<ul>
<li><p><strong>Architecture</strong>:</p>
<ul>
<li>Visual encoder: ViT-g/14 (freezing)</li>
<li>Aligning layer: Q-Former (lightweight BERT structure)</li>
<li>LLM: Vicuna/Flan-T5 (freezing or LoRA)</li>
</ul>
</li>
<li><p><strong>Working mechanisms of Q-Former</strong>:</p>
<p>Q-Former's core is<strong>32 Query Vectors for Learning</strong>, the characterization of the command sensor is obtained through two layers of Attention:</p>
<p><strong>Step A: Self-Attention (mixed text with Query)</strong></p>
<ul>
<li>Enter: [\\text{Query}<em>{\text{Learned&#125;&#125;, \text{Token}</em>TTt =&#36;Close</li>
<li>Learned Queries interacts with Text Tokens, adjusts themselves to the text</li>
<li>For example: the text is&quot;Find the dog.&quot;, the Query vector becomes&quot;Looking for dog-like features.&quot;shape</li>
</ul>
<p><strong>Step B: Cross-Attention (extracting information from images)</strong></p>
<ul>
<li><strong>Q(Query)</strong>: Queries that integrate text command information</li>
<li><strong>K(Key)</strong>: frozen ViT image feature, map by linear layer</li>
<li><strong>V(Value)</strong>: frozen ViT image feature, map by linear layer</li>
</ul>
<p>The mathematical process of Cross-Attention is:
&#36;&#36; \text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q \cdot K^T}{\sqrt{d_k&#125;&#125;\right) \cdot V &#36;&#36;</p>
<p>32 Query vectors are calculated to be similar to 257 Image Patches, as Q has been integrated&quot;Find the dog.&quot;, on behalf of&quot;Dog.&quot;Image Patches received high marks, and eventually Query vectors&quot;Suck it away.&quot;The most relevant visual information is given, and irrelevant background is ignored.</p>
</li>
</ul>
<h4>Training strategy: multi-phase training</h4>
<p>The training of the VLM is multistaged and gradually upgraded.<strong>Pre-training alignment phase</strong>It's about letting the LLM&quot;Read it.&quot;Visual features, which at this stage freeze visual encoders and LLM, are trained only in the alignment layer, using large-scale graphic matching (e.g. Concaptual Captions, COCO) to match the task with the graphic, so that the alignment layer learns to map visual features into embedded spaces that LLM understands.</p>
<p><strong>Command fine-tuning phase</strong>The modeling community answers the questions by drawing pictures. At this point, LLM (or part of the layer) and alignment layers were unfrozen using mission data such as VQA, Caption, Dialogue and so forth, for training in cross-blanc losses (Next Token Protection). Through multitasking data, modeling institutions combine visual understanding and language generation to develop a full VQA capability.</p>
<p><strong>Balance mechanism</strong>In order to prevent LLM from forgetting language knowledge or hallucinogenic, it is common to mix pure text data in training data to ensure that models retain their original language understanding while acquiring multi-modular capabilities.</p>
<h4>Capacity and constraints</h4>
<p>The generating VLM has a strong multimodular understanding: it supports visual questions and answers (VQA), image descriptions, OCR text recognition, and can be reasoned on the basis of images, such as writing codes, creating poetry, explaining scientific concepts, etc.</p>
<p>However, VLM also has significant limitations:</p>
<ul>
<li><strong>High cost of reasoning</strong>: needs to run both ViT and LLM, with a visible demand usually above 16GB</li>
<li><strong>The problem of hallucinations is very strong.</strong>: The model may describe what does not exist in the image because the pattern is not fully aligned - visual encoder features may be missing and the alignment transmission signal is weakened, resulting in LLM cause&quot;I can't see.&quot;And rely on language a priori to guess, for example, to describe a white cat as&quot;White cat in red hat.&quot;</li>
<li><strong>Limited capacity to understand details</strong>: The loss of information from the CLIP encoder makes it difficult to identify fine words, count and understand small details</li>
</ul>
<h3>Evolving Summary: Succession and Innovation of the Three Level Models</h3>
<p>And we can put the contents together, and the relationship between LLM, CLIP, VLM can be organized into this table below.</p>
<p><strong>Comparative table</strong>:</p>
<table>
<thead>
<tr>
<th align="left">Contrast Dimensions</th>
<th align="left">Plain Language LLM</th>
<th align="left">CLIP type model</th>
<th align="left">Generating VLM</th>
</tr>
</thead>
<tbody><tr>
<td align="left"><strong>Enter a mollusc</strong></td>
<td align="left">Text only</td>
<td align="left">Image + text (independent two-channel)</td>
<td align="left">Image/Video+Text (Mixed Series)</td>
</tr>
<tr>
<td align="left"><strong>Basic structure</strong></td>
<td align="left">Single Tower Transformer Decoder</td>
<td align="left">Double Tower Encoder (ViT + Text Encoder)</td>
<td align="left">Dota Integration (View Encoder+ Alignment Layer+LLM Decoder)</td>
</tr>
<tr>
<td align="left"><strong>Training objectives</strong></td>
<td align="left">Forecast the next Token.</td>
<td align="left">Textbooks Compare Learning</td>
<td align="left">Forecast the next Token (input visual features)</td>
</tr>
<tr>
<td align="left"><strong>Scope of competence</strong></td>
<td align="left">Text Task</td>
<td align="left">Match, classify, retrieve</td>
<td align="left">Text understanding, description, question and answer</td>
</tr>
<tr>
<td align="left"><strong>The reasoning.</strong></td>
<td align="left">Generate Word by Word</td>
<td align="left">One-time calculation vector point, extremely fast</td>
<td align="left">Visual encoding (slow) + Word by Word generation, highest delay</td>
</tr>
<tr>
<td align="left"><strong>Typical application</strong></td>
<td align="left">Translation, summary, logical reasoning</td>
<td align="left">Zero sample classification, graphic search</td>
<td align="left">VQA, image description, OCR</td>
</tr>
</tbody></table>
<p>These three are not substitutes for each other, but...<strong>Reuse and Completion</strong>Relationship:</p>
<p><strong>CLIP inherits the text capability of LLM</strong>:</p>
<ul>
<li>CLIP.<strong>Text Encoder</strong>The same Transformer structure as LLM.</li>
<li>CLIP has reverted to the LLM's text understanding and presentation capability.</li>
<li>CLIP is innovative in extending text capability to visual-linguistic alignment tasks</li>
</ul>
<p><strong>2. VLM succession to both LLM and CLIP</strong>:</p>
<ul>
<li><strong>From LLM</strong>VLM<strong>Directly re-enact LLM from the bottom of the language.</strong>It inherited its language expression, its logic and its world knowledge.</li>
<li><strong>From CLIP</strong>VLM<strong>Visual encoders go straight to pre-trained CLIP.</strong>And it inherited the visual expression that it had learned.</li>
<li><strong>VLM Innovation</strong>: In design<strong>Align Layer</strong>, the visual features of the CLIP&quot;Translation&quot;To understand the LLM.</li>
</ul>
<p><strong>3. Commonality of the three</strong>:</p>
<ul>
<li><strong>Structure Same Source</strong>: Main structures are from Transformer</li>
<li><strong>The expression is similar.</strong>: Consisting discrete information (pixels, words) into high-dimensional and dense semantic vectors (Embeding)</li>
<li><strong>Data driver</strong>: Self-monitoring/semi-monitoring pre-training, which relies on large-scale Internet data</li>
</ul>
<h2>The topic of progress - proliferation models and original multimodules</h2>
<p>Based on an understanding of CLIP and VLM, this section discusses two progressive topics: how CLIP in the proliferation model leads to image generation, and from&quot;Suture.&quot;The structure goes to the original multi-modular technological evolution.</p>
<h3>CLIP in the proliferation model: How does the text guide the image generation?</h3>
<p>The core of text understanding of image generation models such as Stable Diffusion remains <strong>CLIP Text Encoder</strong>。</p>
<h4>Text Encoding</h4>
<p>User inputtips like&quot;A cyberpunk city&quot;, CLIP Text Encoder processes text and output features. Unlike VLM,<strong>This isn't just the last EOS vector.</strong>I'm sorry. Stable Diffusion uses CLIP text code Device<strong>Full Token Sequence Output on the Last Level</strong>, output shape is&#36;77 \times 768&#36;(Assuming maximum length 77, dimension 768) this retains a separate semantic information for each word.</p>
<h4>Injection U-Net (Injection)</h4>
<p>The heart of Stable Diffusion is one.<strong>U-Net</strong>It's responsible for predicting noise and denocating. U-Net's full.<strong>Cross-Attention Layer</strong>。</p>
<ul>
<li><strong>Q (Query)</strong>From<strong>Image feature for the U-Net current layer</strong>(Noise Chart Generating)</li>
<li><strong>K (Key) &amp; V (Value)</strong>From<strong>CLIP text feature sequence</strong>(&#36;77 \times 768&#36;)</li>
</ul>
<h4>Physical meaning of the generation process</h4>
<p>When U-Net processes a pixel area of the image, it asks as Query:&quot;What should I draw here?&quot;It compares it with 77 tokens of CLIP, if there is one in the text.&quot;city&quot;This word, and the corresponding Key matches the current Query area, then&quot;city&quot;The corresponding Value will be weighted. Eventually, U-Net, based on semantic maps provided by CLIP, took random noises one by one&quot;Sculpt&quot;becomes an image that matches the description of the text.</p>
<p><strong>Role</strong>: CLIP for generating models&quot;Navigation maps&quot;, U-Net gradually adjusts noise to the semantic direction of CLIP Embeding.</p>
<h3>From&quot;Suture.&quot;To original multiform</h3>
<p>The LLAVA, InstractBLIP that was discussed earlier is owned by&quot;Suture.&quot;(Glue approach): Take ready visual and linguistic models and stick them in a paired layer. There are two obvious limitations to such a programme.</p>
<p><strong>&quot;Suture.&quot;Issues</strong>:</p>
<p>- Yes.<strong>Information is compromised.</strong>I'm sorry. CLIP, which is designed for comparative learning, tends to retain the global semantic and discard details (e.g. OCR text, number of objects, spatial location). For example, VLM has difficulty in seeing small words because ViT has been able to compress this information as early as the coding phase.</p>
<p>Two.<strong>Modular Gap</strong>I'm sorry. LLM is not direct.&quot;Yeah.&quot;Images, which are translated mathematical vectors, have natural modulation boundaries.</p>
<p><strong>Native multiple.</strong>(e.g. GPT-4o, Gemini, Chameleon)<strong>End-to-End Early Fusion</strong>The concept of (end-to-end early integration) addresses these issues. The core approach is no longer using CLIP, but training<strong>Visual Tokenizer</strong>(e.g. VQ-VAE), slice images and convert them into<strong>Separated Token ID</strong>(e.g. Token #482 for a texture) then it will be &#36;[\text{Token}<em>{\text{Text&#125;&#125;, \text{Token}</em>As a hybrid sequence, train a huge Transformer.</p>
<p>The advantages of this structure are that the model can output images token and generate images directly (no external connection to Stable Diffusion); that understanding is more detailed and no longer subject to pre-training objectives of the CLIP; and that support<strong>Intersect Input Output</strong>(Texts in mix)</p>
<h3>VQ-VAE: Paranormal multimodular visual Tokenization</h3>
<p>The original multimodulars seek to turn images into similar text.<strong>Discrete Tokens (dispersed Token)</strong>。</p>
<h4>Core objectives</h4>
<p>One.&#36;256 \times 256&#36;And the image becomes a series of integers:<code>[382, 10, 998, ...]</code>I'm sorry. So that the LLM can predict the next word, like the next word.&quot;Image Block&quot;。</p>
<h4>VQ-VAE Structure</h4>
<p>VQ-VAE (Vector Quantized - Varial AutoEncoder) has three parts:</p>
<p><strong>Encoder (coding)</strong>:</p>
<ul>
<li>CNN compresses images into low-resolution feature maps grids (e.g.&#36;32 \times 32&#36;vector)</li>
</ul>
<p><strong>Codebook -- key</strong>:</p>
<ul>
<li>I saved it.&#36;K&#36;Learning vectors (e.g. 8192) as&#36;e_1, e_2, ..., e_K&#36;</li>
<li>This is a...&quot;Dictionary&quot;</li>
</ul>
<p><strong>Quantification (Quantification-Check Dictionary)</strong>:</p>
<ul>
<li>For each vector on the feature map, found in Codebook<strong>Most like</strong>The vector.&#36;e_k&#36;</li>
<li><strong>Core Operations</strong>: Directly&#36;e_k&#36;Replace Encoder Output</li>
<li>Index to Record&#36;k&#36;- That's it.<strong>Visual Token</strong></li>
</ul>
<p><strong>Decoder (Code decoder)</strong>:</p>
<ul>
<li>Reassemble features using the vectors in Codebook Figure</li>
<li>Reverting to pixel images by inverse volume</li>
</ul>
<h4>The problem of reverse transmission:</h4>
<p><strong>Problem</strong>:VAE 's quantitative operation (recent neighboring argmin) is<strong>Not transposable</strong>- It's impossible to calculate.&quot;Take Index&quot;The gradient of this operation.</p>
<p><strong>Solutions</strong>: Same way VAE solved this problem <strong>Straight-Through Estimator (STE)</strong></p>
<ul>
<li><strong>When forward transmission</strong>: quantify, replace encoder with a Codebook vector</li>
<li><strong>When it's being transmitted backwards</strong>:<strong>Fraud Gradient</strong>- The gradient that sent Deoder back jumped over the quantitative layer and copied it to the encoder output unmoved</li>
<li><strong>Logical</strong>: Although the middle is broken, the gradient passes directly through the codebook vector and encoder output, assuming it's close enough.</li>
</ul>
<h4>Combined with LLM</h4>
<p>Once you've trained VQ-VAE:</p>
<ol>
<li>Image past Encoder → Quantification → Get Token Sequence</li>
<li>Image Tokens and Text Tokens.</li>
<li>Train Transformer predictor sequences</li>
<li><strong>When Generating</strong>LLM predicts Image Token ID → to check the vector in Codebook → to throw to Decoder → to generate pixels Figure</li>
</ol>
<p>This process allows for multi-modular understanding and generation into the same end-to-end training process.</p>
<h2>Concluding remarks</h2>
<p>From LLM to CLIP, to the generated VLM, the change occurs mainly in two locations: how input is expressed and how the different modes are aligned.<strong>LLM</strong>The text generation base is provided.<strong>CLIP</strong>The video is also available in the following video:<strong>VLM</strong>The visual feature access language generation; the diffusion model uses a text encoder to inject language tips into the image generation process, and the original multimodular route further attempts to convert images into a Token sequence that can be processed by Transformer.</p>
<p>These routes are not entirely for who they are. CLIP is suitable for retrieval and matching, generating VLM is suitable for generating answers around images, spreading models are suitable for text to image generation, and original multimodulars attempt to place understanding and generation in the same end-to-end training process. When doing the system, the choice of route depends on whether the mission requires a search, question and answer, generation or a more finer typologies interaction.</p>
<p>Understanding the rationale, structure differences and applicable boundaries of these models will help to judge whether a new paper is changing the visual coding, aligning interfaces, generating methods, or adjusting training objectives.</p>
<h2>Appendix: Detailed Core Foundation Concepts</h2>
<p>The core underlying concepts in the contents of this appendix are included for further learning.</p>
<h3>Cross Attention</h3>
<h4>Core definitions</h4>
<p>Cross Attention is a focus mechanism that allows models to refer to and integrate information from another sequence when dealing with one sequence. It's mathematically expressed as:</p>
<p>&#36;&#36; \text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q \cdot K^T}{\sqrt{d_k&#125;&#125;\right) \cdot V &#36;&#36;</p>
<p>Of which Query comes from one source, Key and Value.</p>
<h4>Distinction from Self-Attention</h4>
<p><strong>Self-Attention</strong>: Q, K, V are all from the same input to capture the dependency relationships within the sequence.</p>
<p><strong>Cross-Attention</strong>: Q comes from one sequence, K and V from another sequence, which is used to integrate information across the sequence.</p>
<h4>Apply scene</h4>
<ul>
<li><strong>Machine translation</strong>: Decode query Encoder's hidden status through Cross Attention</li>
<li><strong>Multimodular Generation</strong>: Image features of U-Net in the Sttable Diffusion merge CLIP text features through Cross Attention</li>
<li><strong>Visual questions and answers</strong>: Question Question Question Query image features to get the answer thread</li>
</ul>
<h3>Cross Entropy</h3>
<h4>From information theory to loss function</h4>
<p>Cross-breathing is derived from the theory of information and is used to measure the difference in the two probability distributions. Understanding its application in depth learning requires retroactivity of its sources of informatics.</p>
<p><strong>Information Content</strong>:
For probability of occurrence &#36;p(x)&#36; Events &#36;x&#36;, the amount of information is defined as:
&#36;&#36; I(x) = -\log(p(x)) &#36;&#36;</p>
<p>Intuitive understanding: when the probability of an event is lower, we feel the more.&quot;Surprise.&quot;, the greater the amount of information obtained.</p>
<p><strong>Entropy</strong>:
Random Variables &#36;X&#36; The entropy is the expectation of its amount of information:
&#36;H(P)=\matbb{E}<em>{P}[I(X)] = - \sum</em>{x} p(x) \log(p(x)) &#36;&#36;</p>
<p>Entropy means the distribution using the best code &#36;P&#36; The average bits used to encode are the lower bounds of the amount of information.</p>
<p><strong>Cross Entropy</strong>:
Use Based Distribution &#36;Q&#36; Encoding to encode from distribution &#36;P&#36; Data:
&#36;&#36; H(P, Q) = - \sum_{x} p(x) \log(q(x)) &#36;&#36;</p>
<h4>Category II scenario: dual cross-cracker</h4>
<p>For the issue of the sub-classification, labels &#36;y \in {0, 1}&#36;, the probability of the model predicts is &#36;\hat{y} = \sigma(z)&#36;, of which &#36;\sigma&#36; It's the Sigmoid function.</p>
<p><strong>Probability distribution means</strong>:</p>
<ul>
<li>Real Tab &#36;y=1&#36;: &#36;P = [1-p, p]&#36; of which &#36;p&#36; It's a positive probability.</li>
<li>Real Tab &#36;y=0&#36;: &#36;P = [1, 0]&#36;</li>
<li>Model prediction: &#36;Q = [1-\hat{y}, \hat{y}]&#36;</li>
</ul>
<p><strong>Diutsil cross-cracker losses</strong>:
&#36;&#36; L = - [y \log(\hat{y}) + (1-y) \log(1-\hat{y})] &#36;&#36;</p>
<p>This formula allows for two situations to be dealt with in a uniform manner:</p>
<ul>
<li>When? &#36;y=1&#36;: &#36;L = -\log(\hat{y})&#36; (only for probabilities of positive classes)</li>
<li>When? &#36;y=0&#36;: &#36;L = -\log(1-\hat{y})&#36; (Concerning only the probability of predicting negative classes)</li>
</ul>
<p><strong>Cooperation with Sigmoid</strong>:</p>
<p>The Sigmoid function output models &#36;z&#36; Map to &#36;(0, 1)&#36;, meets the probabilities of entropy.
&#36;&#36; \hat{y} = \sigma(z) = \frac{1}{1 + e^{-z&#125;&#125; &#36;&#36;</p>
<h4>Multi-Category scene: classification cross-crear</h4>
<p>Yeah. &#36;K&#36; Classification issues, labels coded with One-hot &#36;y \in {0, 1}^K&#36;, the model output gets the probability distribution through Softmax &#36;\hat{y} \in [0, 1]^K&#36;。</p>
<p><strong>Categorized cross-brenade losses</strong>:
&#36;&#36; L = - \sum_{k=1}^{K} y_k \log(\hat{y}_k) &#36;&#36;</p>
<p>Because &#36;y&#36; One-hot code, just... &#36;y_{target} = 1&#36;The remainder is 0, so the simplification is:
&#36;&#36; L = - \log(\hat{y}_{target}) &#36;&#36;</p>
<p><strong>The Softmax.</strong>:</p>
<p>Softmax will logits &#36;z = [z_1, ..., z_K]&#36; Convert to probability distribution:
&#36;&#36;&#36;2 million<em>i = \frac{e^{z_i&#125;&#125;{\sum</em>{j=1}^{K} e^{z_j&#125;&#125; &#36;&#36;</p>
<h4>Comparison with average error (MSE)</h4>
<p><strong>MSE Loss Functions</strong>:
&#36;&#36; L_{MSE} = \frac{1}{2}(y - \hat{y})^2 &#36;&#36;</p>
<p>For forecast &#36;\hat{y}&#36; Guide:
&#36;&#36; \frac{\partial L_{MSE&#125;&#125;{\partial \hat{y&#125;&#125; = \hat{y} - y &#36;&#36;</p>
<p><strong>Key issues</strong>: When used in conjunction with Sigmoid/Softmax, the active function conductor in the gradient chain law will cause the gradient to disappear.</p>
<p>In the case of Sigmoid, the full gradient is:
&#36;&#36; \frac{\partial L_{MSE&#125;&#125;{\partial z} = \frac{\partial L_{MSE&#125;&#125;{\partial \hat{y&#125;&#125; \cdot \frac{\partial \hat{y&#125;&#125;{\partial z} = (\hat{y} - y) \cdot \hat{y}(1 - \hat{y}) &#36;&#36;</p>
<p>When the prediction is almost completely wrong (&#36;\hat{y} \approx 0&#36;, &#36;y=1&#36;)</p>
<ul>
<li>Error &#36;(\hat{y} - y) \approx -1&#36;</li>
<li>Sigmoid Wizard &#36;\hat{y}(1-\hat{y}) \approx 0&#36;</li>
<li><strong>Total gradient &#36;\approx 0&#36;</strong>, leading to the disappearance of the gradient, slowness in the re-engineering of parameters</li>
</ul>
<p><strong>The elegant design of cross-breathing entropy.</strong>:</p>
<p>As noted earlier, the cross-bridge, in conjunction with Sigmoid/Softmax, is perfectly offset by the number of lines that activate the function, which eventually translates into:
&#36;&#36; \frac{\partial L_{CE&#125;&#125;{\partial z} = \hat{y} - y &#36;&#36;</p>
<p>Even when the projections are completely incorrect, the gradient remains maximum to ensure rapid contraction.</p>
<h3>Softmax (incorporated index function)</h3>
<h4>Core definitions</h4>
<p>Softmax converts any real vector to a probabilistic distribution, which is a standard activated function in a multi-classic question. For Vector &#36;z = [z_1, z_2, ..., z_K]&#36;:</p>
<p>&#36;&#36; S_i = \frac{e^{z_i&#125;&#125;{\sum_{j=1}^{K} e^{z_j&#125;&#125; &#36;&#36;</p>
<h4>Mathistics</h4>
<ul>
<li><strong>Non-negative</strong>: &#36;e^z&#36; Always be right. Make sure it's not negative.</li>
<li><strong>Integrative</strong>: All outputs combined are 1 and constitute a legal probability distribution</li>
<li><strong>Mono-Telephone</strong>: Keep the relative size of the input fraction</li>
<li><strong>Zoom in.</strong>: The index function magnifies the difference between fractions, making models more predictive&quot;Decisive.&quot;</li>
</ul>
<h4>Why? &quot;Soft&quot; max?</h4>
<ul>
<li><strong>Hard Max</strong>: Direct Output <code>[1, 0, 0]</code>It's not a guide.</li>
<li><strong>Softmax</strong>: Output smooth probability distribution as <code>[0.66, 0.24, 0.10]</code>, which allows the relative size information to be enabled and kept</li>
</ul>
<h4>Compared to Sigmoid</h4>
<ul>
<li><strong>Sigmoid</strong>: For a second or multiple-label classification, each category is independent and the probability does not necessarily sum up to 1</li>
<li><strong>Softmax</strong>: For cross-classification, all categories compete, and the probability and probability is mandatory 1</li>
</ul>
<h3>Transformer's attention to Q, K, V, O.</h3>
<h4>Symbolic description</h4>
<ul>
<li>&#36;d_{model}&#36;: total embedded dimensions of the model (e.g. BERT-base 768)</li>
<li>&#36;h&#36;: Number of heads (e.g. 12)</li>
<li>&#36;d_k&#36;: the dimension of each head (&#36;d_k = d_{model} / h&#36;)</li>
<li>&#36;X&#36;: Enter the matrix, shape &#36;[\text{Size}<em>{\text{Batch&#125;&#125;, \text{Length}</em>{\text{Sequence&#125;&#125;, d_{\text{model&#125;&#125;]&#36;</li>
</ul>
<h4>Q, K, V Matrix</h4>
<p>In the multi-direction, each head. &#36;i&#36; There are separate weight matrices. &#36;W_i^Q, W_i^K, W_i^V&#36;:</p>
<p><strong>(1) Q (Query) Matrix</strong>:</p>
<ul>
<li>Enter projection to&quot;Query Subspace&quot;</li>
<li>Effect: Generate query vectors for matching other vectors</li>
<li>Multiple meaning: Each head is concerned with different characteristics of input Dimensions</li>
</ul>
<p><strong>(2) K (Key) Matrix</strong>:</p>
<ul>
<li>Enter projection to&quot;Keyspace&quot;</li>
<li>Role: Generate a Query Matched Feature Vector</li>
</ul>
<p><strong>(3) V (Value) Matrix</strong>:</p>
<ul>
<li>Enter projection to&quot;Value Space&quot;</li>
<li>Effect: Store actual content information, extract the corresponding V after matching Q and K</li>
</ul>
<h4>O (Output) Matrix</h4>
<ul>
<li><strong>Role</strong>: Information integration and integration</li>
<li><strong>Process</strong>: close all head outputs, pass &#36;W^O&#36; Interact at full dimensions</li>
<li><strong>Meaning</strong>: Integration of the diverse features of different head extractions</li>
</ul>
<h4>Design rationale</h4>
<p>The multihead mechanism allows the model to follow information in different subspaces in parallel with different locations, each learning different patterns of attention. The ultimate integration of these diverse features through the O matrix indicates that the complex dependency relationships in the sequence are more abundant than the single-headed focus.</p>
