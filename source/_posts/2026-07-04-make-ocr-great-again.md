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

图片与表格也不应该只留下一个 `![](images/xxx.png)`。至少要保留原始图注、图内文字、所属章节和正文里的引用句。对于没有图注的图片，可以用 VLM 生成描述，[Docling 的 picture description enrichment](https://docling-project.github.io/docling/usage/enrichments/#picture-description) 就是干这个的。但要保留来源标记，因为"图中明确写了什么"和"模型认为图表达了什么"不是同一类证据。

这引出了整份清单里我认为最重要的一条：忠实还原的原文和面向检索的增强文本，必须是两个字段。图片描述、表格摘要、公式释义可以扩大召回，但它们不能伪装成原文事实。混在一起，你就永远说不清一个检索结果到底是文档说的还是模型幻觉。

### 页码和坐标不是所有文档都有

`page_idx` 和 `bbox` 对于 PDF 是必填字段。Word 文档没有一套可靠、与渲染环境无关的最终分页。文件里可以有显式分页符，但在 Word 里看到的页码仍会随字体、纸张大小和打印机驱动而变。Word 和 HTML、EPUB、Markdown 都更接近连续流式文档。


所以定位信息必须是可降级的：分页格式给页码和 bbox，流式格式给结构路径和字符偏移。下游查询也要相应地写成"有 bbox 就画框，没有就跳结构位置"，按格式保留不同的定位分支。

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

## 如何选择合适的工具

### 一个简单的对比

带着上面这份清单，我挑了三个值得对比的方案。它们都被归到 OCR 这类工具里，适用场景却不一样。


| | Docling | MinerU | Unlimited OCR |
|:---|:---|:---|:---|
| 出品 | IBM Research Zurich / LF AI & Data | 上海 AI Lab / OpenDataLab | 百度 |
| 定位 | 多格式解析器 + 统一表示 | 文档解析框架，三种核心模式 | 端到端长文档 VLM |
| 输入 | PDF、Office、HTML、EPUB、邮件、音频、视频、ODF、XBRL 等 20+ | PDF、图片、DOCX、PPTX、XLSX | 仅图片和 PDF |
| 内部结构 | 树，`body` 根节点 + JSON pointer 互指 | legacy 输出是按阅读顺序的列表；3.0 起另有按页分组的统一结构 | 模型直出带标记的文本 |
| 长文档 | 以页面为基本处理单元 | 页面/滑动窗口编排 | R-SWA，几十页可联合输入 |
| 许可 | MIT | 3.1.0 起为"基于 Apache 2.0 的自定义许可" | MIT |
| 成熟度 | 高，LangChain / LlamaIndex / Haystack 官方集成 | 高，社区大 | 弱，接入了 vllm 等推理生态 |

MinerU 更专精于 PDF 和复杂版式，Docling 支持的原始格式更多，还提供了一层统一表示。Unlimited OCR 想解决的是长文档的连续性问题。

很多文档解析模型仍以页面为基本输入，跨页段落、跨页表格和连续编号要由外部编排补回来；Unlimited OCR 的 R-SWA 让模型在生成时始终能看见视觉输入，但对已生成的文本只保留最近一段窗口，于是 KV cache 被压到一个固定上限附近，几十页可以一次读完。它已经能通过 Transformers、vLLM 和 SGLang 等方式部署，但项目本身难以直接进入生产。

DOCX、XLSX、PPTX 本质上是 ZIP 容器，里面放着 XML、媒体文件和它们之间的关系。大部分结构和内容已经写在文件里，处理重点是读懂原生结构，而不是把每一页重新做 OCR。Docling 在这方面的适配更完整。MinerU 一直更侧重 PDF、图片、公式和复杂版式，这些场景才需要 OCR 模型或 VLM 参与。

Excel 有两种用法，有时候人们只用很少的表格量，排一下值班表或者设计一下看板，有时候则是数据表。前者可以作为知识库的一部分，后者则应该放到数据库交给数据中台部门使用。对于小规模的排班表格，我一般会魔改 Docling 表格 `Chunker`，让一个 Sheet 对应一个 Table。大表则交给 `HybridChunker` 按 token 切分；表格跨 chunk 时，它默认会重复表头。


### 先判断文档，再选择解析器

按这个思路，路由逻辑其实很朴素：

| 输入 | 默认选择 | 什么时候升级 |
|:---|:---|:---|
| Office / HTML / EPUB / markdown | Docling 格式后端；能直出 `DoclingDocument` 时走 `SimplePipeline` | 原生结构读取失败或格式不受支持时，先转 PDF 再走下面的分支 |
| 数字原生 PDF（文本层完整、版面常规） | Docling `standard` 或 MinerU `pipeline` | 阅读顺序、跨栏或复杂表格明显出错时再试 VLM |
| 扫描件 / 乱码 PDF | 先开 OCR；可用 MinerU `pipeline` 或 Docling `standard` 抽样 | OCR 仍无法恢复复杂版面时，再试 MinerU `vlm`、Docling `vlm`  |
| 重排版、复杂表格、公式密集 PDF | MinerU `vlm`，或 Docling `vlm` | 没得升级，注意观察效果 |

PDF 有没有可用的文本层是第一道分界，但有文本层不等于一定不用 OCR。我会抽取每页字符数、可打印字符比例和图片覆盖率，再随机渲染几页对照；有些 PDF 的隐藏文本层乱码、错位，照样应该按扫描件处理。阈值也不适合写死，中文幻灯片、双栏论文和发票的分布差别很大。

### Docling：处理 XML 和 PDF 不应该放到同一个管线

[Docling 官方托管服务](https://docling-project.github.io/docling/usage/api_server/managed/)叫 **Docling for IBM watsonx**，底层沿用 `docling-serve` 的 REST API。考虑到没有免费使用机会，我一般是本地部署。

Docling 要先按输入格式选择后端和 Pipeline，再调 Pipeline 内部的参数：

| 所在层 | 选项 | 实际做什么 |
|:---|:---|:---|
| 原生结构格式 | 格式后端 + `SimplePipeline` | DOCX、PPTX、HTML、Markdown 等后端直接读取文件里的段落、标题、表格和图片关系，产出 `DoclingDocument`；`SimplePipeline` 接收这份结果并执行统一交付，不跑 PDF 的版面分析、OCR 和 TableFormer |
| PDF / 图片处理管线 | `standard` / `vlm` | `standard` 对应 `StandardPdfPipeline`，把布局、OCR、表格结构等专用模型串起来；`vlm` 对应 `VlmPipeline`，让视觉语言模型按页端到端转换，适合扫描件和非常规版面，但速度、成本与生成误差都更高 |
| `standard` 内的表格模式 | `fast` / `accurate` | `fast` 适合表格简单或先做预览；`accurate` 是 TableFormer 的默认模式，复杂表头、合并单元格和错列较多时再用 |

对于能直接生成 `DoclingDocument` 的原生结构形式 backend，`DocumentConverter` 通常会自动选择 `SimplePipeline`：它接过 backend 的结果，不再运行 PDF 那套页面级视觉结构恢复。`SimplePipeline` 仍保留统一的转换结果和 enrichment 接口，但是否值得对其中的图片另做描述，要按任务再开。

其余参数主要作用于 PDF 的 `standard` 管线：`do_ocr` 是允许 OCR，`force_ocr` 是即使已有文本层也强制整页 OCR；通常先开前者，只有隐藏文本层损坏时才开后者。`do_table_structure` 控制是否恢复表格网格；列被错误合并时，可以关闭 `do_cell_matching`，改用表格模型预测的单元格文本。

代码、公式、图片分类、图片描述和图表理解都是 enrichment，按下游是否真的需要来开。在我处理过的公式密集和复杂版面样本里，MinerU 往往更稳定，所以我习惯让它接管这部分，再映射回统一表示，最后单独做 enrichment。这是我的样本经验，不是两个项目在所有文档上的统一排名。

#### MinerU：更重的 OCR 系统

MinerU 的[官方在线服务](https://mineru.net/)同时提供网页端和 API，省掉本地下载模型与准备显存。价格相对友好，但是有卡的时候我依旧习惯于本地部署。

API 的 `model_version` 有三个值：

| `model_version` | 适用输入 | 特点 |
|:---|:---|:---|
| `pipeline`（默认） | PDF、图片、Office | 传统模块化解析，速度和确定性更好；数字原生 PDF、批量初筛优先用它 |
| `vlm`（官网标注“推荐”） | PDF、图片、Office | 复杂版式、扫描件、公式和表格密集页面更值得试；成本更高，也要防生成式误差 |
| `MinerU-HTML` | HTML | HTML 正文抽取专用模式；不是比 `vlm` 更高级的档位，非 HTML 文件不要选 |

功能参数里，`is_ocr` 默认 `false`，`enable_formula` 和 `enable_table` 默认 `true`，`language` 默认 `ch`；还可以用 `page_ranges` 只解析抽样页。公式开关在 `vlm` 下只影响行内公式。进入 MinerU 的通常已经是 Docling 没处理好的扫描件或复杂文档，所以我在 MinerU 里更常选 `vlm`；数字原生、版面常规的 PDF 仍应先试 `pipeline`，没必要为了统一而多花一次 VLM 成本。

当前开源版默认 `hybrid-engine`。它把原生文本提取和 VLM 结合起来，目标是在保留高精度的同时降低幻觉。它有 `medium` 和 `high` 两档：默认的 `medium` 更快，但不支持 image analysis；需要最高精度或图片分析时再用 `high`。另外还有 `pipeline`、`vlm-engine` 两种本地模式。`pipeline` 可以在 CPU 或 GPU 上运行，强调稳定和无幻觉；`vlm-engine` 主要换取复杂版式上的精度。



## 如何设计一个统一的数据表示层

分流之后，两个解析器产出两套格式，以后引入更多解析器还会有更多套。得先有一层统一表示，才谈得上在它之上做查询。

这层东西要回答的问题其实很确定：文档里有哪些元素，每个元素是什么类型，谁包含谁，读的顺序是什么，内容来自原文哪个位置，坐标该怎么解释，表格的行列关系怎么存，模型生成的内容和原文怎么区分，切完 chunk 之后还能不能找回原来的元素。

DoclingDocument 我认为是这类结构里设计得比较好的一个，它对上面每个问题都有明确的字段：

| 表示层要回答的问题 | DoclingDocument 的做法 |
|:---|:---|
| 有哪些元素 | `texts` / `tables` / `pictures` / `key_value_items` 等类型化容器分开存；新版本还包括 `form_items`、`field_regions`、`field_items` |
| 元素是什么类型 | `DocItemLabel` 枚举，30 多个标签，从 `SECTION_HEADER` 到 `FOOTNOTE` |
| 谁包含谁 | `body` 为根的树，用 JSON pointer（如 `#/texts/1`）互指父子 |
| 阅读顺序 | 树的深度优先遍历，不需要额外的 order 字段 |
| 内容来自哪里 | `ProvenanceItem`，带 `page_no`、`bbox`、`charspan` |
| 坐标怎么解释 | `BoundingBox` 自带 `CoordOrigin` 枚举，左上还是左下写在数据里 |
| 表格结构 | `TableItem.data` 是结构化 `TableCell`，含 `row_span`、`col_span`、`column_header` |
| 模型生成的内容 | 放在 item 的 `meta` 等元数据字段中，和原文字段分开；旧版图片、表格的 `annotations` 已弃用 |
| chunk 怎么回溯 | `BaseChunk.doc_items` 指回构成它的原始元素 |

有几个地方值得单独说。

树而不是列表。阅读顺序和层级关系在树里是同一件事，遍历一遍就得到顺序，看一眼父节点就知道章节路径。用扁平列表加一个 level 整数也能表达，但每次要用都得重新推一遍。

坐标原点写在数据里。`CoordOrigin` 是个枚举而不是约定，因为 PDF 原生坐标是左下原点，图像处理惯例是左上原点，两边都要接就不能靠默认值。当然大部分检索场景并不需要精确到画框，这个字段用不上也不亏，但真要做原文高亮的时候，它省掉的是一类很难查的错。

表格是结构化单元格而不是 HTML 字符串。要回答"这一行对应的列标题是什么"，前者直接读字段，后者要重新解析一遍 HTML。

原文和生成内容分字段。当前 schema 用 item 的 `meta` 保存图片描述、分类和表格摘要等派生信息，正文仍留在内容字段里。旧版图片和表格上的 `annotations` 还保留着兼容入口，但已经标记为弃用。这样正好对应前面那条“不能让模型说的伪装成文档说的”。

关键前提是 `docling-core` 可以独立安装，不依赖 `docling` 主包，纯 Pydantic 模型，官方明确说它是为"互操作性"设计的。也就是说，它本来就打算被第三方当作通用表示来用，而不是 Docling 的内部实现细节。

而且它带的东西远不止 schema。`HybridChunker` 做层级分块加 tokenization-aware 切分，还会把相同标题下的小块合并，产出的 chunk 自带 headings、captions 和位置信息；序列化 API 支持 Markdown、HTML；LangChain、LlamaIndex、Haystack、CrewAI 都有官方集成。用 DoclingDocument 作为语义的表示层，可以减少后续适配工作的成本。

剩下要做的就是 MinerU 输出到 DoclingDocument 的适配器。这部分没什么高明的地方，多是琐碎的工程对齐。legacy `content_list.json` 是按阅读顺序的列表，标题层级要靠 `text_level` 还原；3.0 起所有后端还会输出按页分组、统一为 `type + content` 的 `content_list_v2.json`，新适配器更适合优先接它，但官方仍把 V2 标为开发中格式。坐标也不能混用：`content_list.json` 和 V2 的 `bbox` 映射到 0–1000，`middle.json` 使用对应 `page_size` 的页面尺寸坐标系，VLM 原始 `model.json` 使用 0–1。再加上 MinerU 的 `page_idx` 从 0 起、Docling 的 `page_no` 从 1 起，以及 HTML 表格到带 span 单元格网格的转换，这类差错通常不会报错，只会让引用悄悄偏一页。

RAGFlow 也走过这条路。它在 [v0.22.0](https://github.com/infiniflow/ragflow/blob/main/docs/release_notes.md#v0220) 中实验性加入 MinerU 作为可选 PDF 解析器，没有采用 Docling 的表示，而是适配成自家的 section 结构。外部解析器的输出粒度和内部 chunk 结构仍然需要对齐，在这方面做到对齐没有想象中的那么容易。

## 表示层不是查询层

有了统一表示，检索这件事看起来就顺理成章了。但这里有个容易混淆的地方。

DoclingDocument 解决的是"文档长什么样"，不解决"怎么找到它"。它是一棵树，树擅长表达嵌套和顺序，不擅长回答"哪些文档提到了 X 且发布于 2025 年之后"。这两件事需要不同的数据结构，Docling 官方自己就把它们拆在了不同仓库里。

我比较喜欢的是三条：最传统的关键词查询，PageIndex、DeepRead 这类让模型自己在文档结构上导航的拟人查询，以及密集向量嵌入检索。GraphRAG 那类知识图谱技术不在我常用的选择里。在通用知识库里，实体类型容易漂移，抽取、消歧和图更新的成本也比树导航高很多。

结构化查询是把树投影成关系表或搜索引擎索引，按元素类型、章节路径、页码、文档 ID 做过滤，正文走 BM25。这条路最土也最可靠，精确、可解释，最适合稳定地回答“找 2025 年 Q3 财报里的所有表格”这类带明确过滤条件的问题。

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
- Docling, [Enrichments: picture description](https://docling-project.github.io/docling/usage/enrichments/#picture-description)
- Docling, [Document converter API (`SimplePipeline` / `StandardPdfPipeline`)](https://docling-project.github.io/docling/reference/document_converter/)
- Docling, [Advanced options](https://docling-project.github.io/docling/usage/advanced_options/)
- Docling, [Vision models](https://docling-project.github.io/docling/usage/vision_models/)
- Docling, [REST API](https://docling-project.github.io/docling/usage/api_server/rest_api/)
- Docling, [Managed service](https://docling-project.github.io/docling/usage/api_server/managed/)
- IBM, [Docling for IBM watsonx](https://www.ibm.com/products/docling)
- docling-project, [docling-core](https://github.com/docling-project/docling-core)
- OpenDataLab, [MinerU](https://github.com/opendatalab/MinerU)
- MinerU, [Official online service](https://mineru.net/)
- MinerU, [Online API documentation](https://mineru.net/apiManage/docs)
- MinerU, [CLI tools](https://opendatalab.github.io/MinerU/usage/cli_tools/)
- MinerU, [Output File Format](https://opendatalab.github.io/MinerU/reference/output_files/)
- MinerU, [Changelog](https://opendatalab.github.io/MinerU/reference/changelog/)
- OpenDataLab, [MinerU2.5-Pro: Pushing the Limits of Data-Centric Document Parsing at Scale](https://arxiv.org/html/2604.04771v1)
- RAGFlow, [v0.22.0 release notes](https://github.com/infiniflow/ragflow/blob/main/docs/release_notes.md#v0220)
- Baidu, [Unlimited-OCR](https://github.com/baidu/Unlimited-OCR)
- Baidu, [Unlimited OCR Works: Welcome the Era of One-shot Long-horizon Parsing](https://arxiv.org/html/2606.23050v1)
- vLLM Recipes, [baidu/Unlimited-OCR](https://recipes.vllm.ai/baidu/Unlimited-OCR)
- VectifyAI, [PageIndex](https://github.com/VectifyAI/PageIndex)
- [DeepRead](https://github.com/Zhanli-Li/DeepRead)
- Sarthi et al., [RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval](https://arxiv.org/abs/2401.18059)
- Google Cloud, [Document AI Layout Parser](https://cloud.google.com/document-ai/docs/layout-parse-chunk)
- Microsoft Azure, [Document Content Understanding: Markdown Representation](https://learn.microsoft.com/en-us/azure/ai-services/content-understanding/document/markdown)
- Amazon Web Services, [Tables in Amazon Textract](https://docs.aws.amazon.com/textract/latest/dg/how-it-works-tables.html)
