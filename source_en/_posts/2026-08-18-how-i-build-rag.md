---
title: 'How I Build RAG: From Project Experience to a Systematic Approach'
title_zh: 我如何做 RAG：从项目实践到系统方法
date: 2026-08-18 18:00:00 +0800
categories:
- Agent Systems
- Agent Infrastructure
tags:
- RAG
- Retrieval
- Document Parsing
- Agentic RAG
- AI Engineering
- Evaluation
author: Hyacehila
mathjax: false
hidden: false
excerpt: Drawing on real-world RAG projects, this article examines the roles of the fact, augmentation, and query layers,
  the tradeoff between fixed retrieval and agentic RAG, and why the hard part is not getting a demo running but continuously
  optimizing, evaluating, and iterating on the system.
description: Drawing on real-world RAG projects, this article examines the roles of the fact, augmentation, and query layers,
  the tradeoff between fixed retrieval and agentic RAG, and why the hard part is not getting a demo running but continuously
  optimizing, evaluating, and iterating on the system.
excerpt_zh: 从真实 RAG 项目的工程实践出发，讨论事实层、增强层与查询层如何分工，固定检索与 Agentic RAG 如何取舍，以及为什么真正困难的不是跑通 Demo，而是持续优化、评测和迭代系统。
permalink: /blog/2026/08/18/how-i-build-rag/
lang: en
translation_key: 2026-08-18-how-i-build-rag
translation_status: machine
translation_source_hash: 39ea157a0993ceba2a21dcc79986765246628ab2e4d0405df9c1e36f118b0532
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<p>In this blog, I would like to talk about how I understand and how I can complete a RAG system and consider optimizing it.</p>
<p>The questions in this article can also be addressed<a href="/en/blog/2026/05/29/embedding-atlas-rag-embedding-visualization/">Embedding Atlas: understand the embedded space of RAG with visualization</a>How the concept of a relatively close read together is developed in different contexts.</p>
<p>When I first started studying AI Agent, I was not very interested in the RAG technique; the tips for a few samples were initially valuable, but when the knowledge base was larger and humans were trained in its more adequate models, they could shift from a point to a point where the model was designed to solve the problem to one that limits its ability to act. Building an AI Agent requires a balance between restraint and freedom, and I always thought that RAG was a compromise technique, and we should focus on building the system itself, and RAG is just one component of it.</p>
<p>But there have been some changes in thinking, and I thought about it in July, when I re-taked the AIAgent technology of the last six months. Perhaps RAG has a future, where training can be knowledgeable and well-intended, but not the reality of the world's multiplicity of habits. At this point, to collect and clean data and to think about how knowledge should be stored and iterative, we may be doing a QA Bot for the time being, but our goal is to build a systematic knowledge system that will allow AI to understand the complex reality of the world and real engineering. Maybe that's what it means to study RAG now.</p>
<p>This Blog will be more based on some practical approaches to how I construct a RAG system on different issues; how to separate the factual layer, enhance it, and search it; how to understand Agentic RAG and the most basic information queries and aggregations; and how to find suitable solutions in the many and costed RAG methods available on the market. We will bring technology from the problems themselves, rather than giving the reader a large comparative table that seems useful.</p>
<p>Like I did a lot of Agent projects, I hope I can go to more answers about why. The final result of Agent is not complicated, but the choice of this technology, the choice of iteration, is the more worthwhile question to answer when doing the AI Agent and RAG systems.</p>
<p>LlamaIndex is actually a good project to do RAG, but learning too many seal structures in an AI Coding era is not comfortable, starting at the bottom of the system to assemble better systems for engineering, and not to introduce many historically useless modules.</p>
<h2>POPO QA Bot</h2>
<p>The POPO QA Bot is a problem I've been doing when I was on the Internet, and it's a simple one, but the simpler the problem is, the better it is to start with, the more we can talk about it.</p>
<p>Our data is from the POPO file library, which you can interpret as a fly book document within the web-efficiencies. Throughout the long-term development of the project, we have evolved a document database based on the documentation system. According to the categories in game development for planning, programming, etc., there are basically a dozen broad categories, some of which are embedded in subcategories 2 or 3 and others have less content, and only large categories. Data source quality is general, including <code>.pptx</code>  <code>.docx</code>   <code>.pdf</code>   <code>.md</code>   <code>.xlsx</code>  <code>.html</code> Multiple formats, the ultimate goal is to build a QA robot that can answer questions based on the document library and that is better able to continue to evolve.</p>
<p>In fact, when we come to this point, our final question is clear, and we need a means of dealing with data from this mixed source in a relatively uniform manner and in a way that is appropriately tailored to our understanding of the data themselves. And then we need to build a simple search layer to answer the questions and finally think about how the system should evolve further.</p>
<p>We'll go one by one.</p>
<h3>Build Factual Layer</h3>
<p>This is the first RAG project, so I need to talk about one of my basic personal tendencies.</p>
<p>I don't like to convert all documents into standard markdown, and set a slice size, and then make some overlap. This idea is very easy to achieve and it can be a good way to aggregate data from a variety of sources. But it loses the structure of the document itself, and humans don't read a document without structure, so why do we have to let AI read a document without structure? DeepRead is a paper that I like very much, and PageIndex is an open-source project that we have to talk about building a knowledge base, so let's keep the structure at this point, not destroy it.</p>
<p>Now that we need to keep the document structure, Docling is a good place to put it on the floor of the reality layer. It is an IBM-initiated open-source document resolution framework that supports multiple Office, PDF, HTML, Markdown formats and provides uniform formats <code>DoclingDocument</code> - Show. It also has OCR, layout analysis and photo understanding capabilities. For most internal files, this is enough; if you meet a particularly complex version of PDF, I'll consider MinerU, which is a reference.<a href="/en/blog/2026/07/04/make-ocr-great-again/">"Let the OCR be great again."</a>。</p>
<p>The six input formats for this project are: <code>.pptx</code>、<code>.docx</code>、<code>.pdf</code>、<code>.md</code>、<code>.xlsx</code> and <code>.html</code>I'm sorry. They were eventually consolidated into three types of structures: document tree, PPT and tables. Three types of structure are in. <code>DoclingDocument</code>However, the same conversion method is not required. We'll be back in the back.</p>
<h4>Basic form of DoclingDocument</h4>
<p><code>DoclingDocument</code> Not a re-formatted Markdown, but a document object with structured and traceable information. It usually contains the document ' s name, source, version, page information and a collection of the following:</p>
<ul>
<li><code>body</code>: root nodes of the body, the order and hierarchy of the document to be preserved;</li>
<li><code>groups</code>: Chapters, lists, etc., can also indicate a slide or a Sheet in our appliance logic;</li>
<li><code>texts</code>: text nodes such as headings, paragraphs, list items, codes and formulae;</li>
<li><code>tables</code>Table nodes, where cell grids can continue to be saved;</li>
<li><code>pictures</code>: Photo nodes, with photo locations, annotated and original page information.</li>
</ul>
<p><code>body</code> And the nodes do not necessarily fit all the contents directly together. In a tree. <code>children</code> Usually save JSON Pointer, for example <code>#/texts/0</code>;quoted nodes are re-approved <code>parent</code> Point back to your container,<code>self_ref</code> It's its own stable address. For PDF or PPTX, node can also be passed <code>prov</code> Save Page Numbers and <code>bbox</code>, this will result in a search result that returns to the original page, instead of leaving a non-located text.</p>
<p>It is worth noting that for a Docling Document, the core content and the structure of the body tree are actually separated, and the content will be placed in the list, and we will only quote in the tree. This would avoid the excessive reading burden of storing structures and text simultaneously in trees. To minimize this, a document is probably the structure below.</p>
<pre><code class="language-text">DoclingDocument
│
├── body: GroupItem
│   └── children
│       ├── Ref(&quot;#/texts/0&quot;)
│       ├── Ref(&quot;#/groups/0&quot;)
│       └── Ref(&quot;#/tables/0&quot;)
│
├── groups[]
│   └── GroupItem(...)
│
├── texts[]
│   ├── TitleItem(...)
│   ├── SectionHeaderItem(...)
│   └── TextItem(...)
│
├── tables[]
│   └── TableItem(...)
│
├── pictures[]
│   └── PictureItem(...)
│
├── key_value_items[]
├── form_items[]
├── field_regions[]
├── field_items[]
│
└── pages{}
</code></pre>
<pre><code class="language-json">{
  &quot;schema_name&quot;: &quot;DoclingDocument&quot;,
  &quot;version&quot;: &quot;1.0.0&quot;,
  &quot;name&quot;: &quot;example.docx&quot;,
  &quot;origin&quot;: {
    &quot;filename&quot;: &quot;example.docx&quot;,
    &quot;mimetype&quot;: &quot;application/vnd.openxmlformats-officedocument.wordprocessingml.document&quot;
  },
  &quot;body&quot;: {
    &quot;self_ref&quot;: &quot;#/body&quot;,
    &quot;children&quot;: [
      { &quot;&#36;ref&quot;: &quot;#/texts/0&quot; },
      { &quot;&#36;ref&quot;: &quot;#/groups/0&quot; }
    ]
  },
  &quot;groups&quot;: [
    {
      &quot;self_ref&quot;: &quot;#/groups/0&quot;,
      &quot;label&quot;: &quot;section&quot;,
      &quot;name&quot;: &quot;第一章&quot;,
      &quot;parent&quot;: { &quot;&#36;ref&quot;: &quot;#/body&quot; },
      &quot;children&quot;: [
        { &quot;&#36;ref&quot;: &quot;#/texts/1&quot; },
        { &quot;&#36;ref&quot;: &quot;#/tables/0&quot; }
      ]
    }
  ],
  &quot;texts&quot;: [
    {
      &quot;self_ref&quot;: &quot;#/texts/0&quot;,
      &quot;label&quot;: &quot;title&quot;,
      &quot;text&quot;: &quot;文档标题&quot;,
      &quot;parent&quot;: { &quot;&#36;ref&quot;: &quot;#/body&quot; }
    },
    {
      &quot;self_ref&quot;: &quot;#/texts/1&quot;,
      &quot;label&quot;: &quot;paragraph&quot;,
      &quot;text&quot;: &quot;第一章中的一段正文&quot;,
      &quot;parent&quot;: { &quot;&#36;ref&quot;: &quot;#/groups/0&quot; }
    }
  ],
  &quot;tables&quot;: [
    {
      &quot;self_ref&quot;: &quot;#/tables/0&quot;,
      &quot;label&quot;: &quot;table&quot;,
      &quot;parent&quot;: { &quot;&#36;ref&quot;: &quot;#/groups/0&quot; },
      &quot;data&quot;: { &quot;table_cells&quot;: [] }
    }
  ],
  &quot;pictures&quot;: []
}
</code></pre>
<p>In the standard Docling process, the solver converts the source file to this object, after which you can continue to export to Markdown, HTML or JSON. Recover the document object and then do the splits, the abstract and the search on it. Press all the inputs into a single section of Markdown, which is equivalent to losing the structure and letting the system guess where it is.</p>
<h4>Document tree: DOCX, PDF, Markdown and HTML</h4>
<p>There is a certain document level in each of these four formats. Below the heading you can have paragraphs, lists, tables and pictures, and there is a parent-child relationship between the chapters. So we prioritize Docling standard resolution process, keeping the document tree that it recovered:</p>
<pre><code class="language-text">DoclingDocument
└── body
    ├── text(label=&quot;section_header&quot;)
    ├── group(label=&quot;section&quot;)
    │   ├── text(label=&quot;paragraph&quot;)
    │   ├── group(label=&quot;list&quot;)
    │   ├── table
    │   └── picture
    └── group(label=&quot;section&quot;)
        └── text(label=&quot;paragraph&quot;)
</code></pre>
<p>PDF usually also provides page numbers and coordinates, while DOCX, HTML and Markdown are closer to a continuous flow of documents. No need to force unity of location information: save the page coordinates when they exist <code>page_no</code> and <code>bbox</code>, at least keep chapter path, node reference and original file information.</p>
<h4>PPT: PPTX by Slide Group</h4>
<p>Text, pictures and tables in PPTX are usually the elements displayed by page, and there is not necessarily a chapter hierarchy as reliable as DOCX. This project is based on a defined and unified approach: each slide corresponds to one <code>group</code>。</p>
<pre><code class="language-text">DoclingDocument
└── body
    ├── group(label=&quot;slide&quot;, name=&quot;slide-0&quot;)
    │   ├── text
    │   ├── picture
    │   └── table
    ├── group(label=&quot;slide&quot;, name=&quot;slide-1&quot;)
    │   └── picture
    └── group(label=&quot;slide&quot;, name=&quot;slide-2&quot;)
        ├── text
        └── table
</code></pre>
<p>This is just an organizational strategy on the side of the project, not to say that all PPTX has a natural chapter structure like this. Slide titles, page numbers, element coordinates and source file information are saved as facts. Some of the PPPXs are some of the chapters that we can use, but I'm not having much of, and the way we're dealing with it changes with the data itself, and we're going to have this problem later.</p>
<h4>Table: XLSX press Sheet grouping</h4>
<p>The main information for XLSX is usually in sheets and cells. For internal scheduling, scheduling and field lists, we'll let one Sheet match one <code>group</code>, saving one or more in group <code>table</code>：</p>
<pre><code class="language-text">DoclingDocument
└── body
    ├── group(label=&quot;sheet&quot;, name=&quot;目录&quot;)
    │   └── table
    ├── group(label=&quot;sheet&quot;, name=&quot;排班表&quot;)
    │   └── table
    └── group(label=&quot;sheet&quot;, name=&quot;字段说明&quot;)
        └── table
</code></pre>
<p>The table node inside can continue to be as <code>tables[].data.table_cells[]</code>I'm sorry. Cells need to keep at least text, position of rows, range of merges and whether they are the top of the row or list:</p>
<pre><code class="language-json">{
  &quot;row_span&quot;: 1,
  &quot;col_span&quot;: 1,
  &quot;start_row_offset_idx&quot;: 4,
  &quot;end_row_offset_idx&quot;: 5,
  &quot;start_col_offset_idx&quot;: 2,
  &quot;end_col_offset_idx&quot;: 3,
  &quot;text&quot;: &quot;夜班&quot;,
  &quot;column_header&quot;: false,
  &quot;row_header&quot;: false,
  &quot;row_section&quot;: false
}
</code></pre>
<h4>The boundary of the de facto layer</h4>
<p>After the above processing, all inputs eventually have a single structured drop point, but this drop point is responsible for recording only what actually exists in the source document. DoclingDocument is the only structured fact layer; the format solvers are responsible for faithfully mapping and cannot add model guesses here.</p>
<p>Severing, photo descriptions, table summaries, Embeding, Rerank, Agent and generation responses are all derivatives of the facts. They can either return to the original language by reference to the node or expand the context during the query, but they cannot modify the factual layer in reverse. This has the practical advantage of not re-deprove the original file when changing the Embeding model, and not changing the document itself when changing the graphic summary hint.</p>
<p>We need to reduce the degree of convergence of the system. That's why the factual layer is separated from the system behind it.</p>
<h3>Enhanced layers and retrieved information</h3>
<p>The original data is enhanced by doing something that RAG would never avoid. We will consider here how to deal with the factual layer, to make some enhancements and to get the last retrieved information, and it needs to be noted that the enhancement and retrieval layer itself is a completely separate layer, and that we will not modify the factual layer, but rather create a separate database to avoid confusion between facts and the illusions of enhanced models.</p>
<p>The most basic enhancements are on the charts and tables. We don't have a multi-modular embedded device, and we can't just turn all the contents of the table into an embedded vector. The charts and tables therefore need to be summarized into a text that is then retrieved and eventually returned to the facts. That is the most basic element to be done.</p>
<p>If we want to build a structured index system similar to DeepRead or PageIndex, then a certain Summary is necessary for every natural structure, and also for queries.</p>
<p>I'll just talk about how I do it, as a reference.</p>
<p>In fact, the enhanced layer and the search layer were constructed together in this project. For vectors and BM25, we need to build the Chink as a search layer, so all I'm doing here is slightly adjusting the logic of Docling 's HybridChunker.</p>
<p>I still use HybridChunker in most places to maintain the whole context of HybridChunker: it will default on each minimum node as a chunk, then consider a consolidation at the same level until it is close to the target we set for chunk; for those who have exceeded the set size, there are some verse-spectrums (no Overlap). Generated Chunk <code>meta.doc_items</code> Still contains a pointer pointing to the original document TableItem or PictureItem. No merger will cross the Chapter protection chapter logic.</p>
<p>I have here some separate logic to better process project pictures and tables. The picture is not enhanced by default, it is placed directly with the caption, and I added VLM understanding of the picture to the Chunker and merge it with Captain as the text to be retrieved. The default table processing for HybridChunker will split the table when it is larger and repeats the table head, and I completely abandon this logic, because most of the tables in the project are more scheduling and schedule, which is not easy to split, and will eventually produce only a single table of summary as the text to be retrieved. No pictures with no information will be analysed, such as the LOGO decorations, and a separate request will be used to filter them.</p>
<p>About chunk size (token count)</p>
<ul>
<li>Knowledge base QA: General 50-200, depending on the length of the question and the answer</li>
<li>General technical documentation: around 400, plus-negative 200</li>
<li>Papers, long reports, long documents: around 600 positive and negative 200
Experimental support is generally required for validation only as an empirical reference; 400 were selected after the experiment and given on the basis of 50 test samples.</li>
</ul>
<p>The Chunker is followed by a summary of the possible queries to the Chunk and a second step in the enhancement and retrieval layer. This is for the Chunk level, and tables and pictures are not considered for direct reuse of the secondary summary. Each chunk will let AI carry out a simple Summary. Pre-treatment is done in advance, and the pure knowledge is obtained by consuming computing during the build-up period (Build-time), in exchange for a very high recall rate and a noise ratio for the retrieval period (Run-time).</p>
<p>There are three lines of thought as to how they're going to get into the library.</p>
<ul>
<li>Search only for Summary</li>
<li>Summary and Content Layers</li>
<li>Sumary and Content are independent
The second was used, but no distinction was found in the actual project tests.</li>
</ul>
<p>The fact layer from which Chunk has gradually left has become part of the enhanced information layer. Each Chunk will eventually contain the text after the slice, some metadata indexes and an index to the DoclingDocument Factual Layer.</p>
<p>Note: We use the CTextualize (chunk) automatic collage header structure to each chunk to facilitate Dense and BM25 retrieval; this is my personal preference and is not validated by reliable experiments.</p>
<p>For the structured indexing Agencia RAG, we have returned to PageIndex, without investing too much effort in optimising it. A simple document index system is constructed to facilitate access to the corresponding document by some Summary and folder names, and search within the document is simply a re-used PageIndex system, pre-building index trees and Summary.</p>
<p>While I like the idea of structured indexing for Deepread and PageIndex, their consumption of token is too large and time-consuming, and the benefits are minimal, so structured indexing in default systems is actually closed.</p>
<h3>Query Layers and Query Realizations</h3>
<p>As with my personal habits, we used SQLite mainly in the query database of this project, which is already sufficient for the current data scale of the system. Vector and text databases are independent; vector data are quantified by each Chunk using Gentra Embedding 3-2048 and are also retrieved by the same search method, using SQLite-vev for vector recall; text databases are directly deposited and FTS5 is used for query.</p>
<p>Rewrites each query is an issue that needs to be considered at any level of the query. In this system, rewrite of the query layer is a default act that will not normally be closed. We'll rewrite each Query to achieve better recall, not to rewrite it into many pieces, in combination with some of the concepts, knowledge and some of the oral habits of the game itself.</p>
<p>Throughout the system, we support both the Agenic RAG and Direct models. In any case, we do not accept the use of information obtained from the Retrieval Level to answer. Once the search layer has been given a specific node number, we will look at the factual layer and make a final response based on the information on the factual layer.</p>
<p>Under Direct, each recall is automatically appropriate to expand the Windows at and below the node with a default value of 2 (total of 5 articles) to achieve what is called Small to Big. The Agenic RAG model is where Agent will decide for itself which nodes to look for and how large the window to use.</p>
<p>Considering that the system involves aggregation of different layers, we default to RRF aggregation; that the general single query provides 30 Chink and aggregates 12 Chunk before the selection; that PageIndex does not involve aggregation, he will provide the final result (PapageIndex default); that we do not design the Reranker model, which is requested through a separate LLM if it is needed; and that the Agenic RAG mode does not involve aggregation and Reranker, which obtains original queries and then decides on its own and resumes, but is limited to the pre-set Budget rounds.</p>
<h3>Evolution</h3>
<p>In fact, our file warehouse is not of high quality. There are many problems that may be more dependent on some lessons learned than on being recorded in warehouses. There are also warehouses that are obsolete and result in some erroneous returns.</p>
<p>That is the meaning of the existence of a self-evolutionary system, and when a user finds that his questions are not answered correctly, he can provide a feedback to the system as a whole. Feedback itself will only describe what this is supposed to look like, and it will not be the user's task to clearly ask where you should modify the data warehouse.</p>
<p>Each feedback is systematically recorded, and the entire log system is used to determine which fast, LLM is involved in the process to confirm which factual layers were wrong and what the correct form should be.</p>
<p>Of course, we do not allow AI to modify the system directly. The entire system of evolution is operated under manual supervision, and any modification of the factual layer requires manual intervention and confirmation and the regeneration of the retrieval layer at a time when the modification is made.</p>
<p>Outdated is an issue that any large enterprise database has to consider, and in fact, nothing in our system has been used to deal with it, and the script is not capable of judging the problem.</p>
<p>Metadata will help us solve this problem at the last level of answer, because all our information will go back to the final level of fact, where the creation of the document can be recorded. If conflicting data are recalled at the same time, then the system will prioritize the document that is updated according to the time of creation, perhaps more true, and of course he will need to warn users at this time that we have a file conflict problem and will need manual decision-making.</p>
<h3>Summary</h3>
<p>This is all that RAG Demo offers, which basically defines the basic operating logic of me doing RAG, but it's not really doing much engineering optimization, for example, with a lot of technical options involving branches and some parameters that are based more on experience than on detailed experiments, and there are really too many variables.</p>
<h2>Game knowledge mapping</h2>
<p>This is a separate RAG project and also from the experience of the Easy Internship, although with some different focus and a greater number of participants.</p>
<p>Our ultimate goal is to build a knowledge base within the game and provide a QA BOT for use by planners or players. How this thing gets, how this level goes through, and so on.</p>
<p>Like previous projects, we need to focus on how data should be obtained, how ultimately they should be asked and answered, and how the system should be updated.</p>
<h3>Build Game Knowledge</h3>
<p>The most critical issue in the discussion of the database remains data. How to access data inside the game, what kind of play, what kind of objects, what kind of description.</p>
<p>We were actually the first to want to use the planning documents, but this was a very unsuccessful attempt, all of which were seriously outdated and were heavily self-restrainted and difficult to match.</p>
<p>We have used the method of obtaining the truth from the code warehouse, which is relatively more reliable for a game like this. We start with UI engineering files, see how the planner table is loaded, extract from the code warehouse a variety of entities, and then form an approximate classification by its own class properties. Such as things, formulas, raw materials, regions, games, monsters, etc.</p>
<p><strong>Note that it is unrealistic for an Agent to read the entire codebase and extract everything. Game source can run into tens of millions of lines, beyond current context windows. We need a Map and Reduce approach, and discovery should be rule-based from the code rather than LLM-based; start from data files and trace through UI files, because not every UI file is useful.</strong></p>
<p>The specific classification here is based on AST, with AI and designers working together. The result is a local sub-classification with a secondary classification, while most areas have only a broad category. We got about 2w nodes, with some nodes connected by edges; 10,000 nodes have edges, with about 30,000 edges in total. All edges are one-way and have no attributes. The general information for individual nodes is the node's name, its description in the planning table, classification metadata, and information about its connected edges. The edges themselves are obtained from the program, not from the model, deterministic code, or manual labels.</p>
<p>These sides are still taken from codes, and the edges themselves may be nodes, but we do not deal with this in the system, and not all the edges of the codes are modelled, and triggers like those in the level are not handled. This is certainly a relatively compromise mapping system, but we do need to control the number of final total variables to ensure that search maps are more usable.</p>
<p>It can be seen from the above structure that he is essentially a LLM Wiki-like map, but at this point the mapping is created by a code-to-program relationship in order to reduce costs and maintain credibility of the summary. We also moved from using Markdown Wiki to using JSON Wiki, and each node still retains a tree-like hierarchy. This is the whole system of fact, a map of the information.</p>
<p>An Agent updates newly introduced data based on SVN commit information, updating classifications and nodes according to the architecture documents established earlier, while cross-checking to ensure that new SVN commits do not alter the existing classification system.</p>
<p>For information that may not be intended for player-facing display (non-public information), we identify it through loghub hits. This is effectively a whitelist mechanism; anything not on the whitelist is handed to designers for manual processing.</p>
<h3>Query on the map</h3>
<p>The graphics are no longer relevant for any of the enhancements, and we need only consider how to look at the graphics.</p>
<p>Because of the limited capacity of its own classification system, some subcategories may have hundreds or hundreds of entities, and it is not realistic to attempt to construct a structured index system similar to PageIndex (not really trying Tree Seach). We still need to use some cold start-up on the entire system using a line similar to the LLM Wiki.</p>
<p>We have provided a basic wiki-search tool to find nodes by coding BM25 and retrieving volume.</p>
<pre><code class="language-json">{&quot;action&quot;:&quot;tool&quot;,&quot;tool&quot;:&quot;wiki.search&quot;,&quot;query&quot;:&quot;检索词&quot;}
</code></pre>
<p>There is no excessive engineering optimization here, but the basic thinking is the same as before us.</p>
<p>Since there are tools to retrieve the segments, there are also tools to read the segments.</p>
<pre><code class="language-json">{&quot;action&quot;:&quot;tool&quot;,&quot;tool&quot;:&quot;wiki.read_page&quot;,&quot;path&quot;:&quot;wiki/example.md&quot;}
</code></pre>
<p>When you find some nodes, you can go over them and check them.</p>
<pre><code class="language-json">{&quot;action&quot;:&quot;tool&quot;,&quot;tool&quot;:&quot;graph.search&quot;,&quot;path&quot;:&quot;node&quot;}
</code></pre>
<p>Of course, the model needs time to answer the question.</p>
<pre><code class="language-json">{&quot;action&quot;:&quot;final&quot;,&quot;answer&quot;:&quot;最终回答&quot;}
</code></pre>
<p>The answer requires evidence.</p>
<p>This system is completely AG-based, and we do not discuss the normal QA, which is the same thing as the LLM Wiki approach, which is to analyze actions, execute tools, add the results to the next round of context as observation; then he can continue to rewrite the query, expand the chart search or answer questions when he thinks that knowledge is sufficient. Allows multiple jumps in the spectra, some restraint by hints, and sets the windowt for query, with default set to 6 jump.</p>
<p>The final response level, which sets aside shares for all routes, is the result of the RRF aggregation of the two routes, and the one-way top-k can also return. Graph here is the Agent who decides which returns. All of them are all llm as reranker if conditions permit. </p>
<p>The side actually has attributes, but in the previous system they were modelled for one-way purposes, and if they were, they could bring better benefits and provide the Agenic RAG with more adequate decision-making information.</p>
<p>The improved version would use tree-based requery: Agentic retrieval over the tree supplies supplementary information for requerying, after which traditional BM25 and dense-vector retrieval obtains nodes, expands over the graph, and gathers information for the answer.</p>
<h2>AI-PM: Managing Director of Refined Products</h2>
<p>AI-PM was originally meant to give me a little bit of a product thinking, after all, the AI Coding era, and not just about technology, or a little bit about the product, to see if it could become a Builder.</p>
<p>But now my formal work and some of the things that I need to learn are basically exhausting me. I can't really be a professional product manager, so I'd like to try refining Lenny Rachitsky, who has a subscription database for you, and he has a lot of experience in the product, and that's what AI-PM wants to do.</p>
<h3>Factual and Retrieving Layers</h3>
<p>The content documents that entered the knowledge base are 349 Newsletters and 289 Podcasts, 638 of which are available. But before we give it to Docling for resolution, we need some basic cleaning and reinforcement.</p>
<p>All data are in the form of a markdown document, but the title level is not clear and contains an undefeated YaML field. Newsletter is mainly a definitive title that is consolidated and gives the necessary content to the title. Podcast inserts the title of the chapter and subsection at the boundary of the speaker turn. Chapter titles are derived from the model, but the text order is retained. They're all trying to keep the entire document system structured, which we've mentioned at the beginning of the entire Blog.</p>
<p>Next, the Markdown of Docling <code>SimplePipeline</code> Convert enhanced documents to <code>DoclingDocument</code>I'm sorry. The project closes the remote and local photo grabs, so the fact layer saves only the URL, Caption and Docling references. Because all the pictures actually contain Caption, the images here are not actually enhanced.</p>
<p>We still use the separation of the factual and retrieval layers, and the whole of the factual layer is sliced with the Hybridtrunker. The maintenance of the CTextualize (chunk) automatically collating header structure to each Chunk is still enhanced by Summary.</p>
<h3>Query Layer</h3>
<p>The project will not generate a search index until the fact layer is completed. Keyword index is used in SQLite FTS5, which separates titles, chapters, metadata and body text and gives them different weights; semantic index is used in 2048D GM-Embeding-3. The two databases are independent and the results are constructed to answer questions by returning to the factual level.</p>
<p>In Direct, the problem goes through Query Rewrite, which is a language library. Rewrite results to include both natural language queries suitable for semantic search and keywords and phrases suitable for FTS5. The top 60 candidates are given for both keywords and vectors, and the RRF is used to merge and 12 loctors are retained. RRF here is just a merger, not a determination of fact, but only evidence.</p>
<p>Each selected Chunk reads the commanding document of the selected DoclingDocument object itself and two adjacent objects on both sides of the object, and weighs it in the same document. If there is no searchable, then the probability is that the factual layer has changed but that the search layer has not been updated, and the re-establishment of the search layer needs to be considered. The search index is a specter, and only the factual level can answer whether the content is still the one we recognize.</p>
<p>The difference between the Agenic model is mainly about who decides the next search. It first gives Agent a small number of candidates for both keywords and semantics, not an automatic RRF in Direct mode, and not a hidden Query Rewrite. After that, Agent can re-select keyword search, semantic search, read a window for a localtor, or observe the content of evidence that has been read.</p>
<p>And then, then, after seeing a search, it's not like that, either. Only <code>read_source_window</code> The resulting Evidence is allowed to enter the context of the answer, the number of times read, the number of tool wheels and the number of tokens are limited by the button, and the final reference in the answer also checks whether the evidence actually corresponds to the evidence that has been read in the current round.</p>
<p>Agenic RAG did not bypass the factual layer, but simply handed out the "what, how, how many windows" from the fixed program to Agent. Fixed and Agenic searches use different strategies, but both must ultimately cross the same evidentiary boundary.</p>
<p>It seems we didn't do anything new in this project, but it was interesting and it worked.</p>
<p>The project completely abandoned the structured indexing system similar to PageIndex, because we do not have a clear and credible index structure. I tried to turn all the documents into a LLM Wiki knowledge base, which did look like it was, but it didn't feel like it was the quality of the answers. This is the construction map, which is costly, but the benefits are not clear.</p>
