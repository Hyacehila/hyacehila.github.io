---
title: Make OCR Great Again
title_zh: 让 OCR 再次伟大
date: 2026-07-04 23:30:00 +0800
categories:
- Agent Systems
- Agent Infrastructure
tags:
- OCR
- Document Parsing
- AI Engineering
author: Hyacehila
mathjax: false
excerpt: 'A retrieval-oriented view of production OCR: what a parser should deliver for text, sections, formulas, images,
  and tables; how to route documents across Docling and MinerU; why DoclingDocument makes a good unified representation; and
  why the representation layer is not the query layer.'
description: 'A retrieval-oriented view of production OCR: what a parser should deliver for text, sections, formulas, images,
  and tables; how to route documents across Docling and MinerU; why DoclingDocument makes a good unified representation; and
  why the representation layer is not the query layer.'
excerpt_zh: 从面向检索的交付标准出发，讨论文字、章节、公式、图片和表格应该如何被结构化与增强，再比较市面上常见的 OCR 解决方案，顺便聊聊我喜欢怎么做。
permalink: /blog/2026/07/04/make-ocr-great-again/
lang: en
translation_key: 2026-07-04-make-ocr-great-again
translation_status: machine
translation_source_hash: 9b2e280f2c4b26882009ea5723ea79a098452011c43b08d5fb2c143315c221c3
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<p>I don't study OCR. I used to talk about it, but I thought about it in my head, but I just wrote it from the picture.&quot;Boom!&quot;It's pretty good to come out, like the extraction of words from the big phone albums.</p>
<p>The questions in this article can also be addressed<a href="/en/blog/2026/08/18/how-i-build-rag/">RG: From project practice to system approach</a>、<a href="/en/blog/2026/02/16/mcp-model-context-protocol/">MCP (Model Context Protocol)</a>How the concept of a relatively close read together is developed in different contexts.</p>
<p>And then I started studying Agent, and I realized that OCR sometimes gets stuck in the front of the entire system. The real world is not so clean.<code>.png</code>、<code>.pdf</code>、<code>.docx</code> And all the strange formats are inputs that the system has to process. Scientific research can directly use the cleaned data sets, not engineering, and we have to get these things in.</p>
<p>The document is usually troublesome more than words. Where the table is, how the formula is kept, what the title and the text are, whether the reading order is not broken, whether the diagram and the annotated note are to be placed together, whether the picture on the page is to be cut, described or left with a reference. These issues, which were more like later, are now becoming part of OCR itself. Now OCR is not the same as it used to be, it's a fixed line used to clean up raw data and facilitate the back RAG.</p>
<h2>OCR, not recognition, document resolution.</h2>
<p>If it's just a text, a lot of numbers PDF should never have gone OCR. The text is drawn directly from PyMuPDF, pdfPlumber, cheap, fast, and does not re-identify the original clean text and introduce errors.</p>
<p>OCR is becoming interesting again because we start looking at documents as a structured object. The page is not a string of characters, but a mixture of text, title, formulae, tables, pictures, footnotes, header feet and reading order. If the model is just a piece of text, it's going to take a lot of effort to guess where it comes from.</p>
<p>So now the more accurate name of this thing is document parsing. It's going to tear the page and spell back what a machine can do and can be read. Markdown is a good exit, but not the only exit, and more valuable is the structural information it retains.</p>
<p>So the question becomes: what should this structure look like?</p>
<h2>Beyond the model: what OCR at the level of delivery should produce</h2>
<p>If downstream is search, RG or Agent, OCR's delivery cannot be just a Markdown that seems right. Markdown is more like a preview layer, and what really should be delivered is a document object that can be tracked, split, enhanced and indexed.</p>
<p>The big factory has given the reference answer.<a href="https://cloud.google.com/document-ai/docs/layout-parse-chunk">Google Document AI Layout Parser</a> You keep titles, tables, formulae, lists and hierarchical relationships, and then you generate contact-awarechunds with ancestral titles.<a href="https://learn.microsoft.com/en-us/azure/ai-services/content-understanding/document/markdown">Markdown of Azure Content Understanding</a> Saves the chapter, table, formulae, pictures, page numbers and directories in a visible way. The common judgement behind them is that the pure OCR text flattens the reading order and context, and that the retrieval system needs to know which section, which page, which chart or which table a section is related to.</p>
<p>There are several design lines that can be learned from it: elements need to be stable, position information, content generated separately from the original, structure cannot be flattened, complex elements need searchable explanations, and each result can go back to the original page. It's the complete hosting service that's abandoned because I need to be able to run locally and to customize the logic of the process. In an era when Data is all you need, it is not a good idea to outsource the data-cleaning pipe.</p>
<h3>Different elements require different delivery criteria</h3>
<table>
<thead>
<tr>
<th align="left">Object</th>
<th align="left">Minimum deliverables</th>
<th align="left">Retrieval-oriented enhancements</th>
</tr>
</thead>
<tbody><tr>
<td align="left">Text</td>
<td align="left">Paragraphs, lists, reading order, language and page numbers, dealing with hyphens, hyphens and header footers</td>
<td align="left">Keep chapter path, paragraph roles, key entities and context, without all text being crushed into a paragraph</td>
</tr>
<tr>
<td align="left">Chapter Structure</td>
<td align="left">Title level, paternity, directory anchor and chapter range</td>
<td align="left">Write the ancestral title to chunk mettadata, let&quot;Methodology&quot;&quot;Result&quot;There's still a whole spectrum of such duplicate titles.</td>
</tr>
<tr>
<td align="left">Formula</td>
<td align="left">LaTeX or MathML, line/line type, formula number, original map and coordinates</td>
<td align="left">Add a neighbouring definition, variable interpretation and searchable natural language interpretation and mark whether the interpretation is generated by models</td>
</tr>
<tr>
<td align="left">Pictures and Charts</td>
<td align="left">Original cropping, original drawings, OCCR, page numbers, bbox, and references to text</td>
<td align="left">Generate descriptive text describing objects, coordinates, legends and main visible relationships; while retaining visual access to embedding</td>
</tr>
<tr>
<td align="left">Table</td>
<td align="left">Cell grid, row title, merge cell, unit, title, footnote and original coordinates</td>
<td align="left">Also provide structured data and retrieval summaries, and write key words into natural languages, without taking into account the conclusions under the original table</td>
</tr>
</tbody></table>
<p>Formulas are particularly susceptible to error. Only one formula picture is left, and text search is almost impossible to find; only LaTeX is kept and may be missing&quot;What's this formula for?&quot;semantics. It would be more prudent to tie together formulae, numbers, ex-ante interpretations and variables. User Search&quot;How to calculate long-term incentive discounts&quot;, you should be able to hit the section where the formula is, not ask for a query to appear. <code>\gamma</code>。</p>
<p>Pictures and forms should not be left alone. <code>![](images/xxx.png)</code>I'm sorry. At least keep the original drawing notes, the text in the diagram, the chapter to which you belong and the quotations in the body. For pictures without a note, a description can be generated using VLM.<a href="https://docling-project.github.io/docling/usage/enrichments/#picture-description">Docling's picture of the situation</a> That's what it is. But keep the source tag because&quot;What is clearly written in the figure?&quot;and&quot;The model thinks the map is a sign of something.&quot;Not the same kind of evidence.</p>
<p>This leads to one of the most important items on the entire list that I believe is: faithful restitution of the original language and search-oriented enhanced text, which must be two fields. Photo descriptions, table summaries, formulae interpretations can expand recall, but they cannot be disguised as original facts. You can never tell whether a search result is a document or a model illusion.</p>
<h3>The page number and coordinates are not all in the files.</h3>
<p><code>page_idx</code> and <code>bbox</code> For PDF is mandatory field. Word documents do not have a reliable set of final pages that are not relevant to the rendering environment. A visible page break can be found in the file, but the page numbers that you see in Word will still change with font, paper size and printer drive. Word and HTML, EPA, Markdown are closer to a continuous stream document.</p>
<p>So positioning information must be downgraded: page breaks to page numbers and bbox, flow formats to structure paths and characters. And the downstream search is written accordingly.&quot;There's bbox, frame, no structure.&quot;, the different positioning branches are maintained in format.</p>
<h3>Don't cut the pieces, then try to find the structure.</h3>
<p>The cut-off for retrieval should occur after structural recovery. A cut of every 500 tokens can easily separate the title from the body, separate the tablehead from the data line, or lose the picture description.</p>
<p>More rational order is:</p>
<ol>
<li>Recover the page elements and read order first</li>
<li>Creates chapter tree, element and page-crossing relationships</li>
<li>chunk by chapter, paragraph and element boundary</li>
<li>Attach each chunk ancestor title, location information, document version and element type</li>
<li>Generate separate chunk for pictures, tables and formulae while retaining their relationship with the body father cunk</li>
</ol>
<p>Character accuracy remains important at this time, but it is not the final acceptance standard. A system that delivers should also answer these questions:</p>
<ul>
<li>Can you hit the right paragraph when searching for the subject of the chapter and bring back the full title path?</li>
<li>Can you retrieve the corresponding rows, column headings, units and footnotes at the same time when searching for a condition in the table?</li>
<li>When searching for a phenomenon expressed in a chart, can you find a picture by drawing or by generating a description and return to the original page?</li>
<li>When searching for formulae, can you return formulae, variable definitions and adjacent interpretations, rather than isolated LaTeX?</li>
<li>Can each result be traced to the source document, location and distinguishing between the original language and the content of the generation?</li>
<li>When a page or element is not deciphered, can the system clearly report failure rather than producing a seemingly complete Markdown?</li>
</ul>
<p>The last one I felt was the easiest to ignore. The failure of the analysis is acceptable, and silence fails because you don't even know what's wrong.</p>
<h2>How to select the right tool</h2>
<h3>A simple comparison.</h3>
<p>With this list above, I have chosen three comparable options. They're all in OCR, but the scenes are different.</p>
<table>
<thead>
<tr>
<th align="left"></th>
<th align="left">Docling</th>
<th align="left">MinerU</th>
<th align="left">Unlimited OCR</th>
</tr>
</thead>
<tbody><tr>
<td align="left">It's a product.</td>
<td align="left">IBM Research Zurich / LF AI &amp; Data</td>
<td align="left">Shanghai AI Lab / OpenDataLab</td>
<td align="left">100 degrees</td>
</tr>
<tr>
<td align="left">Positioning</td>
<td align="left">Multiformer Parser + Unique</td>
<td align="left">Document Parsing Frame, Three Core Modes</td>
<td align="left">End to Long Document VLM</td>
</tr>
<tr>
<td align="left">Input</td>
<td align="left">PDF, Office, HTML, EPA, mail, audio, video, ODF, XBRL, etc. 20+</td>
<td align="left">PDF, Pictures, DOCX, PPTX, XLSX</td>
<td align="left">Pictures and PDFs only</td>
</tr>
<tr>
<td align="left">Internal structure</td>
<td align="left">Trees,<code>body</code> Root + JSON pointer</td>
<td align="left">Legacy output is a reading-ordered list; 3.0 from a single structure with a separate page by page</td>
<td align="left">Model straight out of marked text</td>
</tr>
<tr>
<td align="left">Long Document</td>
<td align="left">Page-based Processing Unit</td>
<td align="left">Page/Sliding Window Organization</td>
<td align="left">R-SWA, dozens of pages available for joint input</td>
</tr>
<tr>
<td align="left">Permission</td>
<td align="left">MIT</td>
<td align="left">3.1.0 Commencement&quot;Customise permission based on Apache 2.0&quot;</td>
<td align="left">MIT</td>
</tr>
<tr>
<td align="left">Mature</td>
<td align="left">High, Langchain / LlamaIndex / Haystack Official Integration</td>
<td align="left">High, big community.</td>
<td align="left">Weak, access to Vllm, and so on.</td>
</tr>
</tbody></table>
<p>MinerU is more specialized in PDF and complex layouts, and the original format supported by Docling is more extensive, and provides a unified expression. Unlimmed OCR is trying to solve the problem of continuity of long documents.</p>
<p>Many document resolution models still use pages as basic inputs, with cross-page paragraphs, cross-page tables and serial numbers to be added from the outside; Unlimed OCR R-SWA allows the model to see visual input at all times when it is generated, but only the nearest window is left for the generated text, so KV Cache is pressed near a fixed limit and dozens of pages can be read out at once. It has been able to deploy through Transformers, vLLM and SSGLang, but the project itself has been unable to enter production directly.</p>
<p>DOCX, XLSX, PPTX are essentially ZIP containers, containing XML, media documents and the relationship between them. Most of the structure and content are already in the document, and the focus is on reading the original structure, rather than re-producing each page. Docling's fit in in that respect is more complete. MinernU has been more focused on PDF, pictures, formulae and complex layouts that require OCR or VLM involvement.</p>
<p>Excel has two uses, sometimes with very few tables, a watch schedule or board, and sometimes a data sheet. The former could be part of the knowledge base, while the latter should be placed in the database for use by the data centre department. For small-scale scheduling forms, I usually do the Docling forms. <code>Chunker</code>- Let a Sheet match a Table. I'll give you the big watch. <code>HybridChunker</code> Press token cut; it will customise the header when the table crosses the chunk.</p>
<h3>First judge the document, then select the solver.</h3>
<p>In this way, the logic of route is simple:</p>
<table>
<thead>
<tr>
<th align="left">Input</th>
<th align="left">Default Selection</th>
<th align="left">When will you upgrade?</th>
</tr>
</thead>
<tbody><tr>
<td align="left">Office / HTML / EPUB / markdown</td>
<td align="left">Docling format backend; straight out <code>DoclingDocument</code> Time to go. <code>SimplePipeline</code></td>
<td align="left">If the original structure is not read or format is not supported, turn PDF and go to the branch below</td>
</tr>
<tr>
<td align="left">Digital Native PDF (text layer complete, layout general)</td>
<td align="left">Docling <code>standard</code> Or MinerU. <code>pipeline</code></td>
<td align="left">Try VLM when there is a clear error in reading order, crossbar or complex table</td>
</tr>
<tr>
<td align="left">Scan / Spectrum PDF</td>
<td align="left">Open first OCR; available MinerU <code>pipeline</code> Or Docling. <code>standard</code> Sample</td>
<td align="left">Try again when OCR is still unable to restore complex layouts <code>vlm</code>、Docling <code>vlm</code></td>
</tr>
<tr>
<td align="left">Re-formatted, complex tables, formulae intensive PDF</td>
<td align="left">MinerU <code>vlm</code>, or Docling <code>vlm</code></td>
<td align="left">No upgrade. Watch the effect.</td>
</tr>
</tbody></table>
<p>PDF has no available text layer is the first dividing line, but having a text layer does not mean that OCR is not necessarily used. I will extract the number of characters per page, printable character proportions and photo coverage, and then randomly render page contrasts; some PDF hidden text layers are not coded, wrongly positioned, and should be processed as a scanned copy. Nor are thresholds suitable for death, and the distribution of Chinese slides, double-bar papers and invoices varies considerably.</p>
<h3>Docling: Processing XML and PDF should not be on the same pipe</h3>
<p><a href="https://docling-project.github.io/docling/usage/api_server/managed/">Docling Official Hostage Service</a>Please. <strong>Docling for IBM watsonx</strong>Bottom of the line. <code>docling-serve</code> RET API. Given that there is no free access, I am generally deployed locally.</p>
<p>Docling selects the backend and Pipeline in the input format and adjusts the parameters inside Pipeline:</p>
<table>
<thead>
<tr>
<th align="left">Layer</th>
<th align="left">Options</th>
<th align="left">What do you actually do?</th>
</tr>
</thead>
<tbody><tr>
<td align="left">Native Structure Format</td>
<td align="left">Format backend + <code>SimplePipeline</code></td>
<td align="left">DOCX, PPTX, HTML, Markdown, etc. directly read paragraphs, titles, tables and photographic relationships in the file, output <code>DoclingDocument</code>；<code>SimplePipeline</code> Receive this result and execute uniform delivery without running PDF layout analysis, OCR and TableFormer</td>
</tr>
<tr>
<td align="left">PDF / Photo Processing Pipeline</td>
<td align="left"><code>standard</code> / <code>vlm</code></td>
<td align="left"><code>standard</code> Correspond <code>StandardPdfPipeline</code>, by linking specialized models such as layout, OCR, table structure, etc.;<code>vlm</code> Correspond <code>VlmPipeline</code>, convert visual language models to page-to-end versions suitable for scanning and unconventional layouts, but with higher speed, cost and generation error</td>
</tr>
<tr>
<td align="left"><code>standard</code> Table Mode in</td>
<td align="left"><code>fast</code> / <code>accurate</code></td>
<td align="left"><code>fast</code> Fits to simple forms or previews;<code>accurate</code> is the default mode for TableFormer, which is used when complex headers, cells are merged and error columns are larger</td>
</tr>
</tbody></table>
<p>For direct generation <code>DoclingDocument</code> The form of the original structure.<code>DocumentConverter</code> Usually, it's automatic. <code>SimplePipeline</code>: it takes the result of the backend, and no longer runs the PDF set of page-level visual structures.<code>SimplePipeline</code> Keeps the uniform conversion result and interface to the enrichment, but whether it is worth a different description of the picture, which is to be opened on a task basis.</p>
<p>The rest of the parameters are mainly for PDF <code>standard</code> Pipeline:<code>do_ocr</code> The blogger says:<code>force_ocr</code> It is the enforcement of the entire page even if there is a text layer; normally the former is opened, and only when the text layer is hidden is damaged.<code>do_table_structure</code> Controls whether or not to restore the table grid; columns can be closed when they are merged by error <code>do_cell_matching</code>, the cell text predicted by the table model.</p>
<p>The code, formulae, photo classification, picture description and chart are understood as "enrichment," and whether or not downstream is really needed. In the formula that I've been working on, in the dense and complex layout sample, MinerU is often more stable, so I'm used to letting it take over this part, and then map it back and do it alone. This is my sample experience, not the uniform ranking of the two items on all the files.</p>
<h3>MinenerU: Heavy OCR system</h3>
<p>Minernu<a href="https://mineru.net/">Official online services</a>The site is also available at the end of the page and at the API, saving local download models and preparing for display. Prices are relatively friendly, but I'm still used to local deployment when I have a card.</p>
<p>API <code>model_version</code> There are three values:</p>
<table>
<thead>
<tr>
<th align="left"><code>model_version</code></th>
<th align="left">Apply Input</th>
<th align="left">Characteristics</th>
</tr>
</thead>
<tbody><tr>
<td align="left"><code>pipeline</code>(Default)</td>
<td align="left">PDF, Picture, Office</td>
<td align="left">Traditional module resolution, speed and certainty; digital primary PDF, first batch screening priority</td>
</tr>
<tr>
<td align="left"><code>vlm</code>(Officer's web label “Recommended”)</td>
<td align="left">PDF, Picture, Office</td>
<td align="left">Complex layouts, scanned copies, formulae and table-intensive pages are more worthwhile to try; higher costs are also needed to prevent the creation of errors</td>
</tr>
<tr>
<td align="left"><code>MinerU-HTML</code></td>
<td align="left">HTML</td>
<td align="left">HTML Text Extracting Special Mode; not ratio <code>vlm</code> Do not select a higher-level, non-HTML file</td>
</tr>
</tbody></table>
<p>function parameter,<code>is_ocr</code> Default <code>false</code>，<code>enable_formula</code> and <code>enable_table</code> Default <code>true</code>，<code>language</code> Default <code>ch</code>; also available <code>page_ranges</code> Only sample pages are parsed. Formula Switches on <code>vlm</code> Down only affects the intra-line formulae. It's usually Docling who's in the MinerU, which is a non-processed scan or complex document, so I'm in MinerU more often. Choose <code>vlm</code>;digital, generic PDF still to be tried first <code>pipeline</code>There is no need to spend VLM more than once for the sake of uniformity.</p>
<p>Default for the current open-source version <code>hybrid-engine</code>I'm sorry. It combines primary text extraction with VLM, with the goal of reducing hallucinations while retaining high accuracy. It has. <code>medium</code> and <code>high</code> 2-Back: Default <code>medium</code> Faster, but not supported; maximum precision or photo analysis required <code>high</code>I'm sorry. And... <code>pipeline</code>、<code>vlm-engine</code> Two local models.<code>pipeline</code> It can be run on CPU or GPU, emphasizing stability and non-supplegicity;<code>vlm-engine</code> In exchange for precision in complex layouts.</p>
<h2>How to design a unified data expression layer</h2>
<p>After the diversion, two solvers produce two sets of formats, and more will be introduced later. It requires a unified expression before the inquiry can be made on it.</p>
<p>The question that this layer is going to answer is really certain: what are the elements in the document, what type each element is, who is it, who is it, who is it read, what order is it, what is it from, what coordinates are to explain, how the row relationship of the table is, how the model is created, how the content is distinguished from the original text, and whether the original elements can be recovered after cutting off the chunk.</p>
<p>The DoclingDocument, I think, is the better designed one in this kind of structure, and it has clear fields for each of the questions:</p>
<table>
<thead>
<tr>
<th align="left">Means questions to answer at the level</th>
<th align="left">DoclingDocument</th>
</tr>
</thead>
<tbody><tr>
<td align="left">What are the elements?</td>
<td align="left"><code>texts</code> / <code>tables</code> / <code>pictures</code> / <code>key_value_items</code> The stylized packagings are separated; the new version also includes <code>form_items</code>、<code>field_regions</code>、<code>field_items</code></td>
</tr>
<tr>
<td align="left">What kind of elements are they?</td>
<td align="left"><code>DocItemLabel</code> Enumeration, 30+ labels, from <code>SECTION_HEADER</code> Present. <code>FOOTNOTE</code></td>
</tr>
<tr>
<td align="left">Who's got who?</td>
<td align="left"><code>body</code> For root trees, use JSON pointer (e.g. <code>#/texts/1</code>I'm a father and a son.</td>
</tr>
<tr>
<td align="left">Reading Order</td>
<td align="left">Tree depth priority through the field without additional order fields</td>
</tr>
<tr>
<td align="left">Where did it come from?</td>
<td align="left"><code>ProvenanceItem</code>- Yeah. <code>page_no</code>、<code>bbox</code>、<code>charspan</code></td>
</tr>
<tr>
<td align="left">What do you mean, coordinates?</td>
<td align="left"><code>BoundingBox</code> Take your own <code>CoordOrigin</code> Enumeration, left top or left bottom in the data.</td>
</tr>
<tr>
<td align="left">Table Structure</td>
<td align="left"><code>TableItem.data</code> It's structured. <code>TableCell</code>, Ham <code>row_span</code>、<code>col_span</code>、<code>column_header</code></td>
</tr>
<tr>
<td align="left">Content generated by the model</td>
<td align="left">It's in its place. <code>meta</code> barematic fields, separate from original text fields; old versions of pictures, tables <code>annotations</code> Disabled</td>
</tr>
<tr>
<td align="left">chunk, how do you trace it?</td>
<td align="left"><code>BaseChunk.doc_items</code> Point back to the original elements that make it.</td>
</tr>
</tbody></table>
<p>There are a few points worth saying alone.</p>
<p>Trees, not lists. Reading sequences and hierarchy are the same thing in trees, and they are sequenced over and over and over and over and over and over and over and over and over and over again, and you know the chapter path by looking at the parent. A flat list with a whole number can be expressed, but it has to be pushed again every time you need it.</p>
<p>The coordinates are written in the data.<code>CoordOrigin</code> It is an itemization rather than an agreement, because the PDF primary coordinates are the bottom left, the image processing practice is the top left, and neither of the two sides can be connected by default. Of course, most of the search scenes do not need to be precise enough to be a frame, and this field is not a bad idea, but when it is really a good one, it saves a kind of hard-to-see fault.</p>
<p>Tables are structured cells rather than HTML strings. Answer.&quot;What's the column title for this row?&quot;The former read the field directly, while the latter re-deciphered HTML.</p>
<p>Original language and creation of content subfields. Current schema with item <code>meta</code> Saves derived information such as photographic descriptions, classifications and table summaries, and the text remains in the content field. Old pages of pictures and tables <code>annotations</code> Compatible accesses are also maintained but marked as obsolete. This corresponds to the one in front that "can't make the model's story look like a document."</p>
<p>Key premise <code>docling-core</code> It can be installed independently, without dependence. <code>docling</code> Main package, pure Pydantic model, officially stated as being&quot;Interoperability&quot;Designed. In other words, it was intended to be used as a generic representation by third parties, not as an internal realization detail for Docling.</p>
<p>And it's bringing more than schema.<code>HybridChunker</code> The hierarchy and tokenization-aware cut-off also combines small pieces under the same heading, with the output chunk taking headings, captions and position information; sequenced API for Markdown, HTML; Langchain, LlamaIndex, Haystack, CrewAI. Use DoclingDocument as a semantic expression layer to reduce the cost of subsequent adaptation work.</p>
<p>The rest is about the adapter that MinerU output to DoclingDocument. There's no such thing as a smart spot, but there's a lot of little engineering. I'm not gonna let you go, Legacy. <code>content_list.json</code> It's a reading-order list. The title level is on it. <code>text_level</code> (a) Revert; 3.0 from all backends also output grouping by page and uniforming to <code>type + content</code> It's... <code>content_list_v2.json</code>The new adapter is more suitable for priority, but the official V2 is still marked as a development format. And the coordinates can't be mixed:<code>content_list.json</code> And V2 <code>bbox</code> The map is 0-1000.<code>middle.json</code> Use corresponding <code>page_size</code> . The page size coordinates system, VLM Original <code>model.json</code> Use 0-1. Plus, MinerU. <code>page_idx</code> From 0, Docling <code>page_no</code> From 1 onwards, and the conversion of HTML tables to a cell grid with spans, such errors are usually not reported wrong, but only allow the reference to be slushly slurred over a page.</p>
<p>RAGFlow also walked this way. It's in... <a href="https://github.com/infiniflow/ragflow/blob/main/docs/release_notes.md#v0220">v0.22.0</a> Medium experimental to add MinerU as an optional PDF solver, not using the Docling expression, but rather suitable for a section structure that fits into a home. The output particle size of the external solver and the internal chunk structure still need to be aligned, and alignment is not as easy as it could be expected.</p>
<h2>Show layer is not query layer</h2>
<p>With a uniform indication, it seems logical to search the matter. But there's a lot of confusion here.</p>
<p>DoclingDocument.&quot;What's the file like?&quot;, don't solve&quot;How do I find it?&quot;I'm sorry. It's a tree. Trees are good at placing and sequencing. They're not good at answering.&quot;Which documents refer to X and are published after 2025&quot;I'm sorry. These two things require different data structures, and Docling himself has opened them up in different warehouses.</p>
<p>I prefer three: the most traditional keyword queries, PageIndex, DeepRead, which allow models to navigate themselves on document structures, and intensive vector-drive search. That kind of knowledge mapping technique that I used to choose. In the common knowledge base, the physical type is easily driftable and the cost of extraction, discrimination and updating is much higher than that of tree navigation.</p>
<p>Structured queries are indexing the tree into a relationship table or search engine, filtering by element type, chapter path, page number, document ID, and walking BM25 in the body. The most reliable, precise and interpretable answer to the question of clear filtering conditions is the most reliable and reliable.</p>
<p>PageIndex, DeepRead, such techniques are searchable like human beings: first look at the stratification catalogue, then look at the chapter title, move down step by step, and finally find what is needed. PageIndex simply calls itself a vctorless, and the whole process does not need a vector bank to allow the model to determine which branch to go on a tree by node; DeepRead makes it a loop of locate-then-read, and locates it and reads it. Their common advantage is that each step can be explained and the result can be directly on the page and chapter.</p>
<p>Vector search is by itself. <code>HybridChunker</code> The output chunk, which carries its own ancestral titles and annotated drawings, puts this information into embedded text and quantifys it, and it is much better to recall semantics. The chunk retains the reference to the original element, and can go back to the structure and the original page after the hit.</p>
<p>These are not competitions, but different views from the same factual level. Integration should take place at the retrieval level, where each is called back and reordered, rather than forcing a universal index at the resolution level. The function of the typology layer is to keep the facts right, to keep them intact and to be able to trace them back to their original location at any time.</p>
<p>But all these callers have a common premise: the document must have a clearly structured tree. The catalogue is correct, the chapter level is not disordered, the title and the body are attributed to the right, and it is directly determined whether the model can be navigable. That's why I'm willing to show you how much energy there is, not just to save it for good, but to really rely on it on the back floor.</p>
<h2>Concluding remarks</h2>
<p>The whole chain is actually four levels: identification, structure, representation, searching.</p>
<p>The identification is only a small first paragraph, which is followed by the restoration of the page to a structural object, and the consolidation of the output of multiple solvers into a single single expression, with the final turn of query. It means that the layers are designed to determine what can be done directly behind them, but it means that doing it better is not the same as being able to find out.</p>
<p>So the title, and the article, the story is twisted. OCR cannot fit the current name, and now the link is about structural restoration, unity, retroactivity and retrieval interface. But this does not prevent it from being much more important than it was in the past, and this is the first and the easiest to underestimate for Agent, who needs to process real documents.</p>
<p>MOGA。</p>
<h2>References</h2>
<ul>
<li>Docling, <a href="https://docling-project.github.io/docling/concepts/docling_document/">DoclingDocument concept</a></li>
<li>Docling, <a href="https://docling-project.github.io/docling/concepts/chunking/">Chunking</a></li>
<li>Docling, <a href="https://docling-project.github.io/docling/concepts/serialization/">Serialization</a></li>
<li>Docling, <a href="https://docling-project.github.io/docling/usage/enrichments/#picture-description">Enrichments: picture description</a></li>
<li>Docling, <a href="https://docling-project.github.io/docling/reference/document_converter/">Document converter API (<code>SimplePipeline</code> / <code>StandardPdfPipeline</code>)</a></li>
<li>Docling, <a href="https://docling-project.github.io/docling/usage/advanced_options/">Advanced options</a></li>
<li>Docling, <a href="https://docling-project.github.io/docling/usage/vision_models/">Vision models</a></li>
<li>Docling, <a href="https://docling-project.github.io/docling/usage/api_server/rest_api/">REST API</a></li>
<li>Docling, <a href="https://docling-project.github.io/docling/usage/api_server/managed/">Managed service</a></li>
<li>IBM, <a href="https://www.ibm.com/products/docling">Docling for IBM watsonx</a></li>
<li>docling-project, <a href="https://github.com/docling-project/docling-core">docling-core</a></li>
<li>OpenDataLab, <a href="https://github.com/opendatalab/MinerU">MinerU</a></li>
<li>MinerU, <a href="https://mineru.net/">Official online service</a></li>
<li>MinerU, <a href="https://mineru.net/apiManage/docs">Online API documentation</a></li>
<li>MinerU, <a href="https://opendatalab.github.io/MinerU/usage/cli_tools/">CLI tools</a></li>
<li>MinerU, <a href="https://opendatalab.github.io/MinerU/reference/output_files/">Output File Format</a></li>
<li>MinerU, <a href="https://opendatalab.github.io/MinerU/reference/changelog/">Changelog</a></li>
<li>OpenDataLab, <a href="https://arxiv.org/html/2604.04771v1">MinerU2.5-Pro: Pushing the Limits of Data-Centric Document Parsing at Scale</a></li>
<li>RAGFlow, <a href="https://github.com/infiniflow/ragflow/blob/main/docs/release_notes.md#v0220">v0.22.0 release notes</a></li>
<li>Baidu, <a href="https://github.com/baidu/Unlimited-OCR">Unlimited-OCR</a></li>
<li>Baidu, <a href="https://arxiv.org/html/2606.23050v1">Unlimited OCR Works: Welcome the Era of One-shot Long-horizon Parsing</a></li>
<li>vLLM Recipes, <a href="https://recipes.vllm.ai/baidu/Unlimited-OCR">baidu/Unlimited-OCR</a></li>
<li>VectifyAI, <a href="https://github.com/VectifyAI/PageIndex">PageIndex</a></li>
<li><a href="https://github.com/Zhanli-Li/DeepRead">DeepRead</a></li>
<li>Sarthi et al., <a href="https://arxiv.org/abs/2401.18059">RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval</a></li>
<li>Google Cloud, <a href="https://cloud.google.com/document-ai/docs/layout-parse-chunk">Document AI Layout Parser</a></li>
<li>Microsoft Azure, <a href="https://learn.microsoft.com/en-us/azure/ai-services/content-understanding/document/markdown">Document Content Understanding: Markdown Representation</a></li>
<li>Amazon Web Services, <a href="https://docs.aws.amazon.com/textract/latest/dg/how-it-works-tables.html">Tables in Amazon Textract</a></li>
</ul>
