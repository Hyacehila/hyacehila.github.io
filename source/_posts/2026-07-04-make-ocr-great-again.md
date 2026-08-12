---
title: "让 OCR 再次伟大"
title_en: "Make OCR Great Again"
date: 2026-07-04 23:30:00 +0800
categories: ["Agent Systems", "Agent Infrastructure"]
tags: ["OCR", "Document Parsing", "AI Engineering"]
author: Hyacehila
excerpt: "从面向检索的交付标准出发，讨论文字、章节、公式、图片和表格应该如何被结构化与增强，再比较市面上常见的 OCR 解决方案，顺便聊聊我喜欢怎么做。"
excerpt_en: "A retrieval-oriented view of production OCR: what a parser should deliver for text, sections, formulas, images, and tables; how to route documents across Docling and MinerU; why DoclingDocument makes a good unified representation; and why the representation layer is not the query layer."
mathjax: false
permalink: '/blog/2026/07/04/make-ocr-great-again/'
---

我并不研究 OCR。以前提到它，我脑子里想到的还是从图片里把字"抠"出来，比如各大手机相册里的文字提取，已经相当好用。

后来开始研究 Agent，我才发现 OCR 有时会卡在整个系统最前面。现实世界的数据没那么干净，`.png`、`.pdf`、`.docx` 和各种奇怪格式都是系统必须处理的输入。科研可以直接使用清洗好的数据集，工程不行，我们得把这些东西接进来。

文档麻烦的地方通常不止是字。表格在哪里，公式怎么保留，标题和正文是什么关系，阅读顺序有没有乱，图和图注要不要放在一起，页面里的图片是裁掉、描述，还是留一个引用。这些以前更像后处理的问题，现在慢慢成了 OCR 本身的一部分。现在的 OCR 和曾经的定义已经不太一样了，变成了一个用来清理原始数据，方便后面 RAG 的固定管线。

## OCR 不是识别，是文档解析

如果只要纯文本，很多数字 PDF 根本不该走 OCR。直接用 PyMuPDF、pdfplumber 抽文本，便宜、快，也不会把原本干净的文本重新识别一遍再引入错误。

OCR 重新变得有意思，是因为我们开始把文档当作一种结构化对象看。页面不是一串字符，而是正文、标题、公式、表格、图片、脚注、页眉页脚和阅读顺序混在一起的东西。模型如果只吐一段文本，后面还要花很多力气猜它来自哪里。

所以现在这件事更准确的名字是 document parsing。它要把页面拆开，再尽量拼回一个机器能用、人也能读的结果。Markdown 是一个好出口，但不是唯一出口，更有价值的是它保留下来的结构信息。

那么问题就变成了：这个结构该长什么样？

## 模型之外：交付水平的 OCR 应该产出什么

如果下游是搜索、RAG 或 Agent，OCR 的交付物就不能只是一份看起来正确的 Markdown。Markdown 更像预览层，真正应该交付的是一份可以被追踪、切分、增强和索引的文档对象。

大厂已经给了参考答案。[Google Document AI Layout Parser](https://cloud.google.com/document-ai/docs/layout-parse-chunk) 会保留标题、表格、公式、列表和层级关系，再生成带祖先标题信息的 context-aware chunks。[Azure Content Understanding 的 Markdown 表示](https://learn.microsoft.com/en-us/azure/ai-services/content-understanding/document/markdown) 会显式保存章节、表格、公式、图片、页码与目录。它们背后的共同判断是：纯 OCR 文本会把阅读顺序和上下文压平，而检索系统需要知道一段内容属于哪一节、来自哪一页、和哪张图或哪张表相连。

从里面能学到几条设计思路：元素要有稳定标识和位置信息，生成内容要和原文分开存，结构不能被压平，复杂元素要有可搜索的解释，每个结果都能回到原始页面。放弃的是那套完整的托管服务形态，因为我需要能在本地跑，也需要定制其中的处理逻辑。在一个 Data is all you need 的时代，把数据清洗管线外包出去不是个好主意。

### 不同元素需要不同的交付标准

| 对象 | 最低交付结果 | 面向检索的增强 |
|:---|:---|:---|
| 文字 | 段落、列表、阅读顺序、语言和页码，处理断词、乱码与页眉页脚 | 保留章节路径、段落角色、关键实体和前后文，不把所有文本压成一段 |
| 章节结构 | 标题级别、父子关系、目录锚点和章节范围 | 把祖先标题写入 chunk metadata，让"方法""结果"这类重复标题仍有完整语境 |
| 公式 | LaTeX 或 MathML、行内/行间类型、公式编号、原图和坐标 | 添加相邻定义、变量解释和可检索的自然语言释义，并标记释义是否由模型生成 |
| 图片与图表 | 原图裁剪、原始图注、图内 OCR、页码、bbox，以及和正文的引用关系 | 生成描述性文字，说明对象、坐标轴、图例和主要可见关系；同时保留视觉 embedding 的入口 |
| 表格 | 单元格网格、行列标题、合并单元格、单位、标题、脚注和原始坐标 | 同时提供结构化数据与检索摘要，把关键行列语义写成自然语言，但不替原表下结论 |

公式尤其容易被错误处理。只保留一张公式图片，文本检索几乎找不到它；只保留 LaTeX，又可能缺少"这个公式在算什么"的语义。更稳妥的做法是把公式、编号、前后解释和变量定义绑在一起。用户搜索"如何计算长期奖励折扣"时，应该能命中公式所在的小节，而不是要求查询里正好出现 `\gamma`。

图片与表格也不应该只留下一个 `![](images/xxx.png)`。至少要保留原始图注、图内文字、所属章节和正文里的引用句。对于没有图注的图片，可以用 VLM 生成描述，[Docling 的 picture description enrichment](https://docling-project.github.io/docling/examples/pictures_description/) 就是干这个的。但要保留来源标记，因为"图中明确写了什么"和"模型认为图表达了什么"不是同一类证据。

这引出了整份清单里我认为最重要的一条：忠实还原的原文和面向检索的增强文本，必须是两个字段。图片描述、表格摘要、公式释义可以扩大召回，但它们不能伪装成原文事实。混在一起，你就永远说不清一个检索结果到底是文档说的还是模型幻觉。

### 页码和坐标不是所有文档都有

`page_idx` 和 `bbox` 对于 PDF 是必填字段。直到我把 DOCX 接进来才发现问题：Word 文档本身没有分页。你在 Word 里看到的分页是渲染结果，随字体、纸张大小、打印机驱动而变，文件里根本不存这个信息。HTML、EPUB、Markdown 同理。

这不是工具不给你页码，是格式里就没有这个概念。硬要页码只能先转 PDF，那个页码不对应任何真实的东西。

所以定位信息必须是可降级的：分页格式给页码和 bbox，流式格式给结构路径和字符偏移。下游查询也要相应地写成"有 bbox 就画框，没有就跳结构位置"。允许各种不同形式的样式分支，才能尝试去兼容不同的格式。

### 不要先切块，再尝试找回结构

面向检索的切块应该发生在结构恢复之后。固定每 500 个 token 切一刀很容易把标题和正文分开，把表头和数据行分开，也可能让图片描述失去对应图片。

更合理的顺序是：

1. 先恢复页面元素和阅读顺序
2. 再建立章节树、元素关系和跨页关系
3. 按章节、段落和元素边界生成 chunk
4. 给每个 chunk 附上祖先标题、位置信息、文档版本和元素类型
5. 对图片、表格和公式生成独立 chunk，同时保留它们与正文父 chunk 的关系

此时字符准确率仍然重要，但它不是最终验收标准。一个能交付的系统还应该回答这些问题：

- 搜索章节主题时，能否命中正确段落，并带回完整标题路径？
- 搜索表格里的某个条件时，能否同时取回对应行、列标题、单位和脚注？
- 搜索一张图表达的现象时，能否通过图注或生成描述找到图片，并回到原始页？
- 搜索公式含义时，能否返回公式、变量定义和相邻解释，而不是孤立的 LaTeX？
- 每一个结果能否追溯到源文件、位置，并区分原文与生成内容？
- 某一页或某种元素解析失败时，系统能否明确报告失败，而不是静默产出一份看似完整的 Markdown？

最后一条我觉得最容易被忽略。解析失败是可以接受的，静默失败不行，因为你连哪里出了问题都不知道。

## 选型：三个工具，其实不在同一层

带着上面这份清单，尝试去对比我觉得值得思考的方案。

| | Docling | MinerU | Unlimited OCR |
|:---|:---|:---|:---|
| 出品 | IBM Research Zurich / LF AI & Data | 上海 AI Lab / OpenDataLab | 百度 |
| 定位 | 多格式解析器 + 统一表示 | 文档解析框架，三后端可选 | 端到端长文档 VLM |
| 输入 | PDF、Office、HTML、EPUB、邮件、音频、视频、ODF、XBRL 等 20+ | PDF、图片、DOCX、PPTX、XLSX | 仅图片和 PDF |
| 内部结构 | 树，`body` 根节点 + JSON pointer 互指 | 扁平列表，层级靠 `text_level` 隐含 | 模型直出带标记的文本 |
| 长文档 | 逐页 | 逐页 | R-SWA，几十页一次读完 |
| 许可 | MIT | 3.1.0 起为"基于 Apache 2.0 的自定义许可" | MIT |
| 成熟度 | 高，LangChain / LlamaIndex / Haystack 官方集成 | 高，社区大 | 低，主分支仅十余次提交 |

MinerU 更专精于 PDF 和复杂版式，Docling 支持的原始格式更多，还提供了一层统一表示。Unlimited OCR 想解决的是长文档的连续性问题。传统多页 OCR 逐页处理，跨页段落、跨页表格、连续编号都要靠外部系统补回来；它的 R-SWA 让模型在生成时始终能看见视觉输入，但对已生成的文本只保留最近一段窗口，于是 KV cache 被压到一个固定上限附近，几十页可以一次读完。现在它更像技术 Demo，不容易放进生产。

### Office 格式根本不是 OCR 问题

DOCX、XLSX、PPTX 本质是 zip 包着的 XML，结构信息完整地写在文件里。任何工具处理它们都是在做 XML 遍历，OCR 引擎在这条路径上完全不参与。

所以拿三个 OCR 方案去比"Word 支持得好不好"，比的根本不是识别能力，而是各自的中间表示设计和边界情况处理：合并单元格、嵌入图表、修订痕迹、文档内嵌图片（只有这部分才需要 OCR）。想清楚这一点，选型的轴就变了，比的不是谁认得准，是谁的结构表达能力强、格式覆盖广。

### 先判断文档，再选择解析器

按这个思路，路由逻辑其实很朴素：

| 输入 | 交给谁 | 为什么 |
|:---|:---|:---|
| Office / HTML / EPUB / 邮件 | Docling standard | XML 直读，无损，OCR 不参与 |
| 数字原生 PDF（有文本层） | Docling 或 MinerU pipeline/hybrid | 快且省，不必上 VLM |
| 扫描件 / 复杂版面 PDF | MinerU hybrid 或 VLM 后端 | 版面精度是这里的瓶颈 |

PDF 是否有文本层是一个很重要划分，如果有文本层那么 docling 就足够。判断依据是每页平均字符数和图片密度，具体阈值根据项目决定。

MinerU 三类后端里，`hybrid-engine` 最值得考虑。它用原生文本抽取给 VLM 兜底，VLM 认错了还有文本层可以比对；纯端到端 VLM 一旦幻觉，没有任何东西能发现。`pipeline` 后端则可以在 CPU 上跑，无幻觉，适合量大且版面简单的文档。

Excel 有两种用法，有时候人们只用很少的表格量，排一下值班表或者设计一下看板，有时候则是数据表。前者可以作为知识库的一部分，后者则应该放到数据库交给数据中台部门使用。

## 如何设计一个统一的数据表示层

分流之后，两个解析器产出两套格式，以后引入更多解析器还会有更多套。得先有一层统一表示，才谈得上在它之上做查询。

这层东西要回答的问题其实很确定：文档里有哪些元素，每个元素是什么类型，谁包含谁，读的顺序是什么，内容来自原文哪个位置，坐标该怎么解释，表格的行列关系怎么存，模型生成的内容和原文怎么区分，切完 chunk 之后还能不能找回原来的元素。

DoclingDocument 我认为是这类结构里设计得比较好的一个，它对上面每个问题都有明确的字段：

| 表示层要回答的问题 | DoclingDocument 的做法 |
|:---|:---|
| 有哪些元素 | `texts` / `tables` / `pictures` / `key_value_items` 四类容器分开存 |
| 元素是什么类型 | `DocItemLabel` 枚举，30 多个标签，从 `SECTION_HEADER` 到 `FOOTNOTE` |
| 谁包含谁 | `body` 为根的树，用 JSON pointer（如 `#/texts/1`）互指父子 |
| 阅读顺序 | 树的深度优先遍历，不需要额外的 order 字段 |
| 内容来自哪里 | `ProvenanceItem`，带 `page_no`、`bbox`、`charspan` |
| 坐标怎么解释 | `BoundingBox` 自带 `CoordOrigin` 枚举，左上还是左下写在数据里 |
| 表格结构 | `TableItem.data` 是结构化 `TableCell`，含 `row_span`、`col_span`、`column_header` |
| 模型生成的内容 | `annotations`，和原文字段分开 |
| chunk 怎么回溯 | `BaseChunk.doc_items` 指回构成它的原始元素 |

有几个地方值得单独说。

树而不是列表。阅读顺序和层级关系在树里是同一件事，遍历一遍就得到顺序，看一眼父节点就知道章节路径。用扁平列表加一个 level 整数也能表达，但每次要用都得重新推一遍。

坐标原点写在数据里。`CoordOrigin` 是个枚举而不是约定，因为 PDF 原生坐标是左下原点，图像处理惯例是左上原点，两边都要接就不能靠默认值。当然大部分检索场景并不需要精确到画框，这个字段用不上也不亏，但真要做原文高亮的时候，它省掉的是一类很难查的错。

表格是结构化单元格而不是 HTML 字符串。要回答"这一行对应的列标题是什么"，前者直接读字段，后者要重新解析一遍 HTML。

原文和生成内容分字段。`annotations` 独立于文本内容，正好对应前面那条"不能让模型说的伪装成文档说的"。

关键前提是 `docling-core` 可以独立安装，不依赖 `docling` 主包，纯 Pydantic 模型，官方明确说它是为"互操作性"设计的。也就是说，它本来就打算被第三方当作通用表示来用，而不是 Docling 的内部实现细节。

而且它带的东西远不止 schema。`HybridChunker` 做层级分块加 tokenization-aware 切分，还会把相同标题下的小块合并，产出的 chunk 自带 headings、captions 和位置信息；序列化 API 支持 Markdown、HTML；LangChain、LlamaIndex、Haystack、CrewAI 都有官方集成。用 DoclingDocument 作为语义的表示层，可以减少后续适配工作的成本。

剩下要做的就是 MinerU 输出到 DoclingDocument 的适配器。这部分没什么高明的地方，多是琐碎的工程对齐：扁平列表要用一个栈还原成树，MinerU 一家就有三套坐标空间（`content_list.json` 是 0–1000 归一化，`middle.json` 是页面 points，VLM 的 `model.json` 是 0–1）要统一，`page_idx` 从 0 起而 Docling 的 `page_no` 从 1 起，表格从 HTML 字符串还原成带 span 的单元格网格。这类差错不会报错，只会让引用悄悄偏一页。

RAGFlow 也走过这条路。它从 v0.22.0 起支持 MinerU 作为可选 PDF 解析器，但没有采用 Docling 的表示，而是适配成自家 DeepDoc 的 section 结构。官方文档提到 MinerU 的输出比 DeepDoc 更碎，建议用 `HierarchicalMerger` 合并。看来解析器之间的粒度对齐是个普遍问题。

### Markdown 会丢掉表格结构

Docling 的序列化文档提到过导出 Markdown 时合并单元格会被压平，因为 Markdown 语法里没有 rowspan 和 colspan，跨格的单元格会变成空格子。LaTeX 同理。官方明确建议，如果下游依赖表格结构，用 HTML 或字典导出，别用 Markdown。

这正好印证了我在[《事实层与界面层》](/blog/2026/06/08/fact-layer-interface-layer-markdown-html/)里说的那件事。Markdown 是给人读的界面层，JSON 才是给系统用的事实层。一份表格在 Markdown 里"看起来对"，不代表它的结构还在。

## 表示层不是查询层

有了统一表示，检索这件事看起来就顺理成章了。但这里有个容易混淆的地方。

DoclingDocument 解决的是"文档长什么样"，不解决"怎么找到它"。它是一棵树，树擅长表达嵌套和顺序，不擅长回答"哪些文档提到了 X 且发布于 2025 年之后"。这两件事需要不同的数据结构，Docling 官方自己就把它们拆在了不同仓库里。

我比较喜欢的是三条：最传统的关键词查询，PageIndex、DeepRead 这类让模型自己在文档结构上导航的拟人查询，以及密集向量嵌入检索。GraphRAG 那类知识图谱技术不在我常用的选择里，它需要预先定义好实体和关系的类型，这个前提在通用知识库上不太成立。

结构化查询是把树投影成关系表或搜索引擎索引，按元素类型、章节路径、页码、文档 ID 做过滤，正文走 BM25。这条路最土也最可靠，精确、可解释，"找 2025 年 Q3 财报里的所有表格"这种问题只有它能干净地回答。

PageIndex、DeepRead 这类技术像人一样检索：先看分层目录，再看章节标题，一步步向下推进，最后找到需要的内容。PageIndex 干脆自称 vectorless，整个流程不需要向量库，让模型在树上逐节点判断该往哪个分支走；DeepRead 把它做成 locate-then-read 的循环，先定位再读。它们的共同好处是每一步都能解释，结果能直接落到具体的章节和页码上。

向量检索则靠 `HybridChunker` 产出的 chunk，它自带祖先标题和图注，把这些信息拼进待嵌入的文本里再向量化，语义召回会好很多。chunk 里保留了对原始元素的引用，命中之后能一路回到结构和原页。

这几条不是竞品，是同一份事实层的不同视图。融合应该发生在检索层，各自召回再排序合并，而不是在解析层强行合成一个万能索引。解析层的职责是把事实存对、存全，并且随时能回溯到原始位置。

不过这些拟人查询都有一个共同的前提：文档得先有一棵结构清晰的树。目录是不是准的，章节层级有没有乱，标题和正文的归属对不对，直接决定模型能不能导航下去。这也是我愿意在表示层上多花力气的原因，它不只是存下来好看，后面那一层是真的依赖它。

## 结语

整条链路其实是四个层次：识别、结构、表示、查询。

识别只是最前面一小段，往后是把页面还原成有结构的对象，再把多个解析器的产出收敛成一份统一表示，最后才轮到查询。表示层的设计好坏直接决定后面能做什么，但表示做得再好也不等于能查，那是另一套东西。

所以标题这个梗，和文章想说的事是拧巴的。OCR 这个名字装不下现在的事，现在这条链路要管的是结构恢复、表示统一、位置回溯和检索接口。但这不妨碍它比过去重要得多，对于需要处理真实文档的 Agent，这是第一道关，也是最容易被低估的一道。

MOGA。

## 参考资料

- Docling, [DoclingDocument 概念](https://docling-project.github.io/docling/concepts/docling_document/)
- Docling, [Chunking](https://docling-project.github.io/docling/concepts/chunking/)
- Docling, [Serialization](https://docling-project.github.io/docling/concepts/serialization/)
- Docling, [Automatic picture description](https://docling-project.github.io/docling/examples/pictures_description/)
- docling-project, [docling-core](https://github.com/docling-project/docling-core)
- OpenDataLab, [MinerU](https://github.com/opendatalab/MinerU)
- MinerU, [Output File Format](https://opendatalab.github.io/MinerU/reference/output_files/)
- MinerU, [Changelog](https://opendatalab.github.io/MinerU/reference/changelog/)
- OpenDataLab, [MinerU2.5-Pro: Pushing the Limits of Data-Centric Document Parsing at Scale](https://arxiv.org/html/2604.04771v1)
- RAGFlow, [MinerU 集成](https://github.com/infiniflow/ragflow)
- Baidu, [Unlimited-OCR](https://github.com/baidu/Unlimited-OCR)
- Baidu, [Unlimited OCR Works: Welcome the Era of One-shot Long-horizon Parsing](https://arxiv.org/html/2606.23050v1)
- vLLM Recipes, [baidu/Unlimited-OCR](https://recipes.vllm.ai/baidu/Unlimited-OCR)
- VectifyAI, [PageIndex](https://github.com/VectifyAI/PageIndex)
- [DeepRead](https://github.com/Zhanli-Li/DeepRead)
- Sarthi et al., [RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval](https://arxiv.org/abs/2401.18059)
- Google Cloud, [Document AI Layout Parser](https://cloud.google.com/document-ai/docs/layout-parse-chunk)
- Microsoft Azure, [Document Content Understanding: Markdown Representation](https://learn.microsoft.com/en-us/azure/ai-services/content-understanding/document/markdown)
- Amazon Web Services, [Tables in Amazon Textract](https://docs.aws.amazon.com/textract/latest/dg/how-it-works-tables.html)
