---
title: "让 OCR 再次伟大"
title_en: "Make OCR Great Again"
date: 2026-07-04 23:30:00 +0800
categories: ["Agent Systems", "Agent Infrastructure"]
tags: ["OCR", "Document Parsing", "AI Engineering"]
author: Hyacehila
excerpt: "从面向检索的交付标准出发，讨论文字、章节、公式、图片和表格应该如何被结构化与增强，再比较 MinerU 3.4 的工程框架和 Unlimited OCR 的长文档模型路线。"
excerpt_en: "A retrieval-oriented view of production OCR: structuring and enriching text, sections, formulas, images, and tables, followed by a comparison of the MinerU 3.4 framework and Unlimited OCR's long-document model approach."
mathjax: false
---

我并不研究 OCR。以前提到 OCR，我脑子里想到的还是从图片里把字“抠”出来，比如各大手机厂商相册里的文字提取功能，已经相当好用。

后来开始研究 Agent，我才发现 OCR 有时会成为整个系统的关键一环。现实世界的数据没有那么干净，`.png`、`.pdf`、`.docx` 和各种奇怪的格式都是系统必须处理的输入。科研可以直接使用清洗好的数据集，但工程不行，我们得去做点什么。

文档麻烦的地方通常不止是字。表格在哪里，公式怎么保留，标题和正文是什么关系，阅读顺序有没有乱，图和图注要不要放在一起，页面里的图片是要裁掉、描述，还是留一个引用。这些问题以前更像后处理，现在慢慢成了 OCR 本身的一部分。

这也是我最近看 MinerU 和 Unlimited OCR 时比较感兴趣的地方。它们不是同一类东西。MinerU 是一套文档解析工程框架，MinerU2.5-Pro-2605-1.2B 是它在 VLM 与 hybrid 后端中使用的模型之一；Unlimited OCR 则把长文档连续解析放进模型架构里，试图少依赖逐页循环。

两条路都挺有意思。

## OCR 不再只是文本识别

如果只要纯文本，很多数字 PDF 根本不该走 OCR。直接用 PyMuPDF、pdfplumber 或类似工具抽文本，便宜、快，也不会把原本干净的文本重新识别一遍。

OCR 重新变得有意思，是因为我们开始把文档当作一种结构化对象看。页面不是一串字符，而是正文、标题、公式、表格、图片、脚注、页眉页脚和阅读顺序混在一起的东西。模型如果只吐一段文本，后面还要花很多力气猜它来自哪里。

所以现在的文档 OCR 更像 "document parsing"。它要把页面拆开，再尽量拼回一个机器能用、人也能读的结果。Markdown 是一个好出口，但不是唯一出口。真正有价值的，是 Markdown 形式自带的结构信息与其本身双重可读的特性。

## 模型之外：交付水平的 OCR 应该产出什么

如果下游是搜索、RAG 或 Agent，OCR 的交付物就不能只是一份看起来正确的 Markdown。Markdown 更像预览层，真正应该交付的是一份可以被追踪、切分、增强和索引的文档对象。

[Google Document AI Layout Parser](https://cloud.google.com/document-ai/docs/layout-parse-chunk) 会保留标题、表格、公式、列表和层级关系，再生成带祖先标题信息的 context-aware chunks,同时为图片和表格增加元素批注（或解释）方便检索系统工作；[Azure Content Understanding 的 Markdown 表示](https://learn.microsoft.com/en-us/azure/ai-services/content-understanding/document/markdown) 也会显式保存章节、表格、公式、图片、页码（页面元数据符号）与目录。它们背后的共同判断是：纯 OCR 文本会把阅读顺序和上下文压平，而检索系统需要知道一段内容属于哪一节、来自哪一页、和哪张图或哪张表相连。

### 先交付一个可追踪的文档对象

每一个解析元素都应该有稳定的 `element_id`，并携带页码、bbox、阅读顺序、父节点、章节路径、解析后端和版本。原始识别结果、格式归一化结果和模型生成的解释文本也应该分开保存。

一个最小结构可以类似下面这样：

```json
{
  "element_id": "page_12_table_3",
  "type": "table",
  "page_idx": 12,
  "bbox": [82, 214, 921, 786],
  "bbox_space": "normalized_0_1000",
  "heading_path": ["实验结果", "不同模型的性能对比"],
  "content": {
    "original": "<table>...</table>",
    "normalized": "..."
  },
  "search_text": "表 3 比较了不同模型在三个数据集上的准确率和推理延迟……",
  "relations": ["page_12_caption_3", "page_12_footnote_2"],
  "provenance": {
    "parser": "mineru",
    "parser_version": "3.4.4",
    "search_text_source": "generated"
  }
}
```

这里最重要的是 `original` 和 `search_text` 不能混成一个字段。前者负责忠实还原，后者负责帮助检索。图片描述、表格摘要和公式释义可能由模型生成，它们可以扩大召回，但不能伪装成原文事实。

### 不同元素需要不同的交付标准

| 对象 | 最低交付结果 | 面向检索的增强 |
|:---|:---|:---|
| 文字 | 段落、列表、阅读顺序、语言和页码，处理断词、乱码与页眉页脚 | 保留章节路径、段落角色、关键实体和前后文，不把所有文本压成一段 |
| 章节结构 | 标题级别、父子关系、目录锚点和章节范围 | 把祖先标题写入 chunk metadata，让“方法”“结果”这类重复标题仍有完整语境 |
| 公式 | LaTeX 或 MathML、行内/行间类型、公式编号、原图和坐标 | 添加相邻定义、变量解释和可检索的自然语言释义，并标记释义是否由模型生成 |
| 图片与图表 | 原图裁剪、原始图注、图内 OCR、页码、bbox，以及和正文的引用关系 | 生成描述性文字，说明对象、坐标轴、图例和主要可见关系；同时保留视觉 embedding 的入口 |
| 表格 | 单元格网格、行列标题、合并单元格、单位、标题、脚注和原始坐标 | 同时提供 HTML/结构化 JSON 与检索摘要，把关键行列语义写成自然语言，但不替原表下结论 |

公式尤其容易被错误处理。只保留一张公式图片，文本检索几乎找不到它；只保留 LaTeX，又可能缺少“这个公式在算什么”的语义。更稳妥的做法是把公式、编号、前后解释和变量定义绑在一起。用户搜索“如何计算长期奖励折扣”时，应该能命中公式所在的小节，而不是要求查询里正好出现 `\gamma`。

图片也不应该只留下 `![](images/xxx.png)`。至少要保留原始图注、图内文字、所属章节和正文里的引用句。对于没有图注或图注信息不足的图片，可以用 VLM 生成描述。[Docling 的 picture description enrichment](https://docling-project.github.io/docling/examples/pictures_description/) 就会把生成描述附加到图片元素上；Google 的 Layout Parser 也支持为图表和图片生成文本化描述，帮助后续检索。这里仍然要保留来源标记，因为“图中明确写了什么”和“模型认为图表达了什么”不是同一类证据。

表格则需要同时服务精确查询和语义查询。[Amazon Textract 的表格结果](https://docs.aws.amazon.com/textract/latest/dg/how-it-works-tables.html) 会区分单元格、合并单元格、列标题、标题和脚注，这些结构不应该在转成 Markdown 时丢掉。检索系统既可能要找“2025 年 Q3 的收入是多少”，也可能要找“哪张表比较了三种模型的成本”。前一个问题需要可靠的行列对应关系，后一个问题需要表格标题、章节路径和一段简短摘要。

### 不要先切块，再尝试找回结构

面向检索的切块应该发生在结构恢复之后。固定每 500 个 token 切一刀很容易把标题和正文分开，把表头和数据行分开，也可能让图片描述失去对应图片。

更合理的顺序是：

1. 先恢复页面元素和阅读顺序。
2. 再建立章节树、元素关系和跨页关系。
3. 按章节、段落和元素边界生成 chunk。
4. 给每个 chunk 附上祖先标题、页码、bbox、文档版本和元素类型。
5. 对图片、表格和公式生成独立 chunk，同时保留它们与正文父 chunk 的关系。

此时字符准确率仍然重要，但它不是最终验收标准。一个交付水平的系统还应该回答下面这些问题：

- 搜索章节主题时，能否命中正确段落，并带回完整标题路径？
- 搜索表格里的某个条件时，能否同时取回对应行、列标题、单位和脚注？
- 搜索一张图表达的现象时，能否通过图注或生成描述找到图片，并回到原始页？
- 搜索公式含义时，能否返回公式、变量定义和相邻解释，而不是孤立的 LaTeX？
- 每一个结果能否追溯到源文件、页码和 bbox，并区分原文与生成内容？
- 某一页或某种元素解析失败时，系统能否明确报告失败，而不是静默产出一份看似完整的 Markdown？

真正可交付的 OCR 是把文档转成一种方便检索系统继续工作的表示：结构没有被压平，复杂元素有可搜索的解释，生成内容有明确来源，任何命中结果都能回到原始页面。

## MinerU 3.4：把模型放进文档解析系统

MinerU2.5-Pro-2605-1.2B 是 PDF-to-Markdown 文档解析模型，不是 MinerU 工程框架的版本号。它在 2026 年 5 月发布，随后成为 MinerU 3.3 与 3.4 稳定版本中 VLM 和 hybrid 后端使用的主要模型。MinerU 本身则是负责文件输入、后端选择、解析编排、结构恢复、输出协议和服务部署的完整工具链。

MinerU 3.4 也不再依赖一条单一的模型路线。`pipeline` 后端在 3.4 中把 OCR 模型升级到 PP-OCRv6，强调低资源、CPU 可用和稳定解析；`vlm-engine` 使用 MinerU2.5-Pro-2605-1.2B 完成端到端视觉文档解析；`hybrid-engine` 则把原生文本提取、pipeline 能力和 VLM 结合起来，并通过 `effort=medium/high` 在速度、精度与 image analysis 能力之间取舍。

MinerU2.5-Pro 本身仍然值得单独讨论。2605 模型的更新重点很朴素：处理布局类别误判，降低 `image_block` 漏检，增强图表、流程图、印章等 image analysis 能力。论文和模型卡都强调数据工程：不改原来的 1.2B 架构，而是扩大数据规模、清洗难样本、提高标注质量，再调整训练阶段。模型侧说明数据工程能带来收益，框架侧则负责把不同模型和传统解析能力组织成可部署的系统。

更重要的是它的输出。用完整 MinerU 工具链跑完以后，拿到的不只是一个 `.md` 文件。官方输出文档里列了几类结果：

- `layout.pdf`：把页面版面检测结果画出来，检测框右上角还有阅读顺序编号。
- `span.pdf`：给文本 span 上色，用来检查丢字、行内公式和切分问题；这个文件只由 `pipeline` 后端生成。
- `model.json`：当前后端的原始推理结果，不同后端的结构并不相同。
- `middle.json`：更细的中间结构，包含页面、block、line、span 等层级。
- `content_list.json`：更适合后续使用的简化内容列表。
- `content_list_v2.json`：MinerU 3.0 起新增的跨后端统一结构，目前仍标记为开发中，格式可能继续调整。
- 图片裁剪文件和 Markdown 里的图片引用。

这些东西很像工程系统里的账本。Markdown 负责读起来顺，JSON 负责让程序继续处理，PDF 可视化文件负责排查问题。做知识库、RAG、合同抽取、论文解析时，光有 `.md` 经常不够。你还会想知道某段文字在哪一页，bbox 是什么，表格是不是 HTML，公式是不是 LaTeX，某张图有没有被裁出来。

MinerU 也把这些信息放进结果里。`content_list.json` 里有内容类型、文本层级、页码和 bbox 等字段；表格可以进 HTML，公式可以保留成 LaTeX，图片和图表会以路径引用。新的 `content_list_v2.json` 试图为 pipeline、VLM 和 Office 文档提供更统一的 `type + content` 结构，但目前还不适合被当作永久稳定协议。`layout.pdf` 和 pipeline 后端的 `span.pdf` 则解决了一个重要问题：输出看起来挺完整，其实阅读顺序错了，或者某一块小字早就丢了。可视化调试文件能让人快速发现这种问题。

使用时也要分清两层。`mineru-vl-utils` 是给 MinerU VLM 发请求、处理响应的 Python 包，适合单张图片或 standalone image 的调用。它自己也写得很清楚：Transformers backend 慢，不适合生产；这个 client 不计划支持 PDF/DOCX，也不处理跨页、跨文档操作。正式做文档解析，还是应该用完整 MinerU 工具链。

完整 MinerU 这边已经很像一个服务平台。`mineru` CLI 在没有指定 `--api-url` 时会自动拉起本地临时 `mineru-api`；`mineru-api` 同时提供同步的 `/file_parse` 和异步的 `/tasks` 接口；`mineru-router` 可以把多个服务或多张 GPU 组织到统一入口；Gradio、OpenAI-compatible server 和 http-client 模式则覆盖交互界面与远程推理。生产里可以把 VLM 推理服务拆出去，用 vLLM、SGLang、LMDeploy 或其他 OpenAI-compatible server 提供推理，再让 MinerU 用 `vlm-http-client` 或 `hybrid-http-client` 调用。版面分析、文件处理、输出组织和模型推理不必挤在一个进程里。

当然，工程化也会带来工程化的注意事项。API 的 task 状态仍然是单进程、进程内实现，服务重启、热重载或多进程部署后不保证还能查询历史任务；默认任务状态和输出保留 24 小时，随后会被清理。pipeline、VLM 与 Office 后端的 `model.json`、`middle.json` 和 `content_list.json` 也不是完全相同的协议。如果要基于 JSON 做二次开发，必须固定 MinerU 版本、后端和 schema，而不能只看 Markdown 长得像不像。

这不是坏事。相反，它说明 MinerU 已经把问题暴露在工程边界上了。MinerU2.5-Pro 通过模型侧的数据工程提高解析能力，MinerU 3.4 再用 pipeline、VLM、hybrid、API、router 和多种结构化输出把这些能力组织起来。真正工程化的对象不是某一个模型，而是这整套可以替换后端、检查结果并接入下游系统的文档解析框架。

## Unlimited OCR：让模型连续读下去

Unlimited OCR 就不太一样了。

它关注的不是怎样把一套文档解析系统打磨完整，而是一个更模型侧的问题：OCR 能不能像连续转写一样处理长文档，而不是每页都从头开始。

传统多页 OCR 通常是逐页处理。第一页跑完，第二页再跑。工程上很自然，也容易并行。但这种方式会把文档切成很多独立小任务。跨页段落、跨页表格、连续编号、上下文延续，都要靠外部系统补回来。补得好就是工程能力，补不好就会出现很奇怪的断裂。

Unlimited OCR 的论文把核心放在 R-SWA，也就是 Reference Sliding Window Attention。模型生成每一个新 token 时，始终能看见视觉 token 和 prompt 这些 reference；但对已经生成的文本，它只保留最近一段窗口。视觉前缀是固定的，输出侧窗口滑动。这样标准注意力里会随着输出长度不断增长的 KV cache，就被压到一个固定上限附近。

这个设计很适合 OCR。因为转写文档时，模型最需要的东西其实是原始页面和最近一点输出。它不一定需要反复回看几万 token 以前自己写过什么。人抄书时也差不多，眼睛看原文，手上记着刚写完的几个字，继续往下走。

Unlimited OCR 依赖 DeepSeek-OCR 路线里的 DeepEncoder，把高分辨率页面压成较少的视觉 token。论文里提到 1024 x 1024 的 PDF 图像可以压到 256 个 token。这个压缩很关键，因为 R-SWA 只能控制输出侧 KV cache，视觉前缀本身还是要放进去。前缀太长，长文档照样顶不住。

官方用法里，单图支持 `gundam` 和 `base` 两种模式。多页和 PDF 走 `base`，把 PDF 先转成页面图片，再用 `infer_multi` 连续解析。输出里会用 `<PAGE>` 分隔页面。保存结果时，它会生成 `result.md`，也会处理 `<|ref|>` 和 `<|det|>` 这类定位标记。遇到 `image` 区域，后处理会按坐标从页面图里裁剪出来，放到 `images/` 目录，再在 Markdown 里替换成 `![](images/...)`。还会保存带框的 `result_with_boxes.jpg` 或多页版本，方便看模型到底框了什么。

这也不再是简单的 OCR 文本输出。它同时提供阅读顺序、结构标记、坐标、图片裁剪和 Markdown，将部分原本由工程系统负责的结构恢复能力内化到可以端到端训练的模型中。

用 Unlimited OCR 时，参数也不能随便。vLLM recipe 里写得很明确：prompt 要以字面量 `<image>` 开头；`skip_special_tokens=False`；服务端要注册 no-repeat n-gram logits processor；请求里传 `ngram_size=35`，单页窗口常用 `window_size=128`，多页或 PDF 用 `1024`。没有这些，长文档容易在坐标 token 上循环，或者直接空输出。Unlimited OCR 的论文于 2026 年 6 月 23 日公开；作为一个仍处于早期阶段的模型，它足够强大，但在工程优化层面还是一个孩子。

还有一点要冷静：Unlimited 不是物理意义上的无限。论文自己也说，32K 上下文仍然限制 prefill。页数越多，视觉 token 越多，前缀越长。R-SWA 解决的是长输出时 KV cache 一直长的问题，不是让模型凭空装下无限页面。

但这个方向仍然有价值。它把逐页循环这个工程习惯重新拿出来问了一遍：如果模型本身能连续转写，多页文档是不是可以少一些外部拼接？End-to-end 训练的魅力，我们已经见过不止一次了，它能否改变 OCR 还不好说。

## 它们能给用户什么

这两个项目放在一起看，会发现 OCR 的用户价值已经变了。

以前用户要的是文字。现在用户更想要一份可用的文档对象：正文有顺序，标题有层级，公式是 LaTeX，表格能继续解析，图片被裁出来，坐标还在，必要时能回到原页检查。最好还有调试文件，能告诉我哪里识别错了，而不是只给一份看似完整的 Markdown。

MinerU 3.4 更擅长把这些能力组织成系统，MinerU2.5-Pro 则负责其中的 VLM 解析能力。它让我想到以前写过的事实层和界面层：Markdown 是给人读的，JSON 和中间文件才是后续系统继续工作的事实层。你可以用 Markdown 做展示和索引，用 `content_list.json` 做 chunk，用 `middle.json` 做固定版本下的二次开发，用 `layout.pdf` 做通用版面检查，再在 pipeline 后端使用 `span.pdf` 排查文本片段问题。`content_list_v2.json` 值得关注，但在格式稳定前不宜成为下游系统唯一依赖的接口。

Unlimited OCR 更像是在模型层面补一个长期缺口。长文档不是很多页单页 OCR 的简单相加。连续性本身就是能力。它现在还需要专门 recipe，也有不少推理参数要守，但 R-SWA 这个想法很清楚：固定参考，滑动输出，让模型别被自己越写越长的历史拖住。

## 结语

如果要把它们放进工程里，我会先把期待压低一点。

数字 PDF 能直接抽文本就不要 OCR。扫描件、复杂版面、低质量图片页，再交给 OCR/VLM。MinerU 适合做文档解析主流程，但要认真看输出文件和 backend 差异。Unlimited OCR 适合长文档专项评测，尤其是几十页连续解析，但不能忽略 prompt、special token、no-repeat n-gram 和上下文上限。

更现实的做法，可能是混合管线。普通文档用成熟解析系统处理，长文档和复杂页交给更强的 VLM/OCR 模型；输出统一落到 Markdown 和 JSON，再加一层质量检查。但混合管线的开发成本和维护成本都很高，MinerU 与 Unlimited OCR 这种开箱即用的工具能为我们节省 AI 时代最宝贵的时间。随着推理成本继续下降，更多文档处理步骤可能会交给统一模型，但是否跳过复杂度分流，仍然要由质量、延迟和成本共同决定。

MinerU 3.4 让我看到 OCR 的工程化价值，MinerU2.5-Pro 则说明模型侧的数据工程同样重要。不是所有能力都要塞进模型里，文档解析本来就需要文件处理、版面分析、后处理、输出协议和调试工具；反过来，框架也需要持续吸收更强的模型能力。

Unlimited OCR 提醒我们，模型结构仍然能改变工作流。长文档解析不一定永远是逐页循环加外部拼接。也许以后真正好用的 OCR 系统，会一边有 MinerU 这样的工程外壳，一边吸收 Unlimited OCR 这种长文档模型能力。

当进入真实的工程实践中，[Amazon Textract](https://docs.aws.amazon.com/textract/latest/dg/what-is.html)， [Google Document AI Layout Parser](https://cloud.google.com/document-ai/docs/layout-parse-chunk)，[Azure Document Content Understanding: Markdown Representation](https://learn.microsoft.com/en-us/azure/ai-services/content-understanding/document/markdown)。都提供相关能力，OCR 早已经成为了一项平台化的基础服务。

OCR 以前负责把字认出来。现在它开始负责把文档还原成可以继续使用的结构。对于需要处理真实文档的 Agent，OCR 是离不开的一环，MOGA。

## 参考资料

- Google Cloud, [Document AI Layout Parser](https://cloud.google.com/document-ai/docs/layout-parse-chunk)
- Microsoft Azure, [Document Content Understanding: Markdown Representation](https://learn.microsoft.com/en-us/azure/ai-services/content-understanding/document/markdown)
- Amazon Web Services, [Tables in Amazon Textract](https://docs.aws.amazon.com/textract/latest/dg/how-it-works-tables.html)
- Docling, [Automatic picture description](https://docling-project.github.io/docling/examples/pictures_description/)
- Docling, [Hybrid chunking](https://docling-project.github.io/docling/examples/hybrid_chunking/)
- OpenDataLab, [MinerU Releases](https://github.com/opendatalab/MinerU/releases)
- MinerU, [Changelog](https://opendatalab.github.io/MinerU/reference/changelog/)
- OpenDataLab, [MinerU2.5-Pro-2605-1.2B](https://huggingface.co/opendatalab/MinerU2.5-Pro-2605-1.2B)
- MinerU, [Output File Format](https://opendatalab.github.io/MinerU/reference/output_files/)
- MinerU, [Quick Usage](https://opendatalab.github.io/MinerU/usage/quick_usage/)
- OpenDataLab, [mineru-vl-utils](https://github.com/opendatalab/mineru-vl-utils)
- OpenDataLab, [MinerU2.5-Pro: Pushing the Limits of Data-Centric Document Parsing at Scale](https://arxiv.org/html/2604.04771v1)
- Baidu, [Unlimited-OCR](https://github.com/baidu/Unlimited-OCR)
- Baidu, [Unlimited OCR Works: Welcome the Era of One-shot Long-horizon Parsing](https://arxiv.org/html/2606.23050v1)
- vLLM Recipes, [baidu/Unlimited-OCR](https://recipes.vllm.ai/baidu/Unlimited-OCR)
