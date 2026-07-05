---
title: "让 OCR 再次伟大"
title_en: "Make OCR Great Again"
date: 2026-07-04 23:30:00 +0800
categories: ["Agent Systems", "Agent Infrastructure"]
tags: ["OCR", "Document Parsing", "AI Engineering"]
author: Hyacehila
excerpt: "借 Unlimited OCR 的发布，简单聊聊 OCR 技术的两条演进路线：MinerU2.5-Pro 把模型放进一套工程化文档解析系统里，Unlimited OCR 则把长文档连续解析做进模型本身。"
excerpt_en: "A short note on two OCR directions: MinerU2.5-Pro as an engineering-oriented document parsing system, and Unlimited OCR as a long-horizon OCR model."
mathjax: false
---

我并不研究 OCR ，以前说 OCR，我脑子里想到的还是把图片里的字抠出来，比如各大手机厂商相册里的自动扣字，相当的可用。

现在我来研究 Agent 这个问题以后， 发现 OCR 有时候成为了整个系统的关键一环。现实世界的数据没有那么干净，`.png` 、`.pdf` 或者是 `.docx` 或者各种奇怪的形式都是我们要处理的一环，科研可以去直接用清洗好的数据集，但工程不行，我们得去做点什么。

文档麻烦的地方通常不止是字。表格在哪里，公式怎么保留，标题和正文是什么关系，阅读顺序有没有乱，图和图注要不要放在一起，页面里的图片是要裁掉、描述，还是留一个引用。这些问题以前更像后处理，现在慢慢成了 OCR 本身的一部分。

这也是我最近看 MinerU2.5-Pro 和 Unlimited OCR 时比较感兴趣的地方。它们不是同一类东西。MinerU2.5-Pro 算是最成熟的文档解析工程系统，模型只是其中的一环；Unlimited OCR 则把长文档连续解析放进模型架构里，试图少依赖逐页循环。

两条路都挺有意思。

## OCR 不再只是文本识别

如果只要纯文本，很多数字 PDF 根本不该走 OCR。直接用 PyMuPDF、pdfplumber 或类似工具抽文本，便宜、快，也不会把原本干净的文本重新识别一遍。

OCR 重新变得有意思，是因为我们开始把文档当作一种结构化对象看。页面不是一串字符，而是正文、标题、公式、表格、图片、脚注、页眉页脚和阅读顺序混在一起的东西。模型如果只吐一段文本，后面还要花很多力气猜它来自哪里。

所以现在的文档 OCR 更像 "document parsing"。它要把页面拆开，再尽量拼回一个机器能用、人也能读的结果。Markdown 是一个好出口，但不是唯一出口。真正有价值的，是 Markdown 形式自带的结构信息与其本身双重可读的特性。

## MinerU2.5-Pro：把 OCR 做成系统

MinerU2.5-Pro 走的是一条很工程化的路线。

官方模型卡把它叫做 PDF-to-Markdown 文档解析模型。最近的更新重点也很朴素：处理布局类别误判，降低 `image_block` 漏检，增强图表、流程图、印章等 image analysis 能力。论文和模型卡都强调数据工程：不改原来的 1.2B 架构，扩大数据规模，清洗难样本，提高标注质量，再做训练阶段上的安排。

这和很多模型发布时喜欢讲新结构不太一样。MinerU2.5-Pro 的意思更像是：文档解析这件事，系统和数据本身就能带来很大收益。

更重要的是它的输出。用完整 MinerU 工具链跑完以后，拿到的不只是一个 `.md` 文件。官方输出文档里列了几类结果：

- `layout.pdf`：把页面版面检测结果画出来，检测框右上角还有阅读顺序编号。
- `span.pdf`：给文本 span 上色，用来检查丢字、行内公式和切分问题。
- `model.json`：模型原始推理结果。
- `middle.json`：更细的中间结构，包含页面、block、line、span 等层级。
- `content_list.json`：更适合后续使用的简化内容列表。
- 图片裁剪文件和 Markdown 里的图片引用。

这些东西很像工程系统里的账本。Markdown 负责读起来顺，JSON 负责让程序继续处理，PDF 可视化文件负责排查问题。做知识库、RAG、合同抽取、论文解析时，光有 `.md` 经常不够。你还会想知道某段文字在哪一页，bbox 是什么，表格是不是 HTML，公式是不是 LaTeX，某张图有没有被裁出来。

MinerU 也把这些信息放进结果里。`content_list.json` 里有内容类型、文本层级、页码等字段；表格可以进 HTML，公式可以保留成 LaTeX，图片和图表会以路径引用。`layout.pdf` 和 `span.pdf` 解决了一个重要问题：输出看起来挺完整，其实阅读顺序错了，或者某一块小字早就丢了。可视化调试文件能让人快速发现这种问题。

使用时也要分清两层。`mineru-vl-utils` 是给 MinerU VLM 发请求、处理响应的 Python 包，适合单张图片或 standalone image 的调用。它自己也写得很清楚：Transformers backend 慢，不适合生产；这个 client 不计划支持 PDF/DOCX，也不处理跨页、跨文档操作。正式做文档解析，还是应该用完整 MinerU 工具链。

完整 MinerU 这边就更像一个服务平台了。它有 CLI，也有 `mineru-api`、Gradio、`mineru-router`、OpenAI-compatible server 和 http-client 模式。生产里可以把 VLM 推理服务拆出去，用 vLLM 或 LMDeploy 起服务，再让 MinerU 用 `hybrid-http-client` 调它。版面分析、文件处理、输出组织和模型推理不必挤在一个进程里。

当然，工程化也会带来工程化的注意事项。比如 API 的 task 状态默认只在单个进程内保存，服务重启后不会保留；pipeline backend 和 VLM backend 的结构化输出不完全兼容；如果你要基于 JSON 做二次开发，就不能只看 Markdown 长得像不像。

这不是坏事。相反，它说明 MinerU 已经把问题暴露在工程边界上了。MinerU2.5-Pro 在简单模型架构的基础上，通过大规模的数据工程和 OCR 系统层面的优化，为我们带来了一个工程上高度可用的 OCR 工具包。

## Unlimited OCR：让模型连续读下去

Unlimited OCR 就不太一样了。

它关注的不是怎样把一套文档解析系统打磨完整，而是一个更模型侧的问题：OCR 能不能像连续转写一样处理长文档，而不是每页都从头开始。

传统多页 OCR 通常是逐页处理。第一页跑完，第二页再跑。工程上很自然，也容易并行。但这种方式会把文档切成很多独立小任务。跨页段落、跨页表格、连续编号、上下文延续，都要靠外部系统补回来。补得好就是工程能力，补不好就会出现很奇怪的断裂。

Unlimited OCR 的论文把核心放在 R-SWA，也就是 Reference Sliding Window Attention。模型生成每一个新 token 时，始终能看见视觉 token 和 prompt 这些 reference；但对已经生成的文本，它只保留最近一段窗口。视觉前缀是固定的，输出侧窗口滑动。这样标准注意力里会随着输出长度不断增长的 KV cache，就被压到一个固定上限附近。

这个设计很适合 OCR。因为转写文档时，模型最需要的东西其实是原始页面和最近一点输出。它不一定需要反复回看几万 token 以前自己写过什么。人抄书时也差不多，眼睛看原文，手上记着刚写完的几个字，继续往下走。

Unlimited OCR 依赖 DeepSeek-OCR 路线里的 DeepEncoder，把高分辨率页面压成较少的视觉 token。论文里提到 1024 x 1024 的 PDF 图像可以压到 256 个 token。这个压缩很关键，因为 R-SWA 只能控制输出侧 KV cache，视觉前缀本身还是要放进去。前缀太长，长文档照样顶不住。

官方用法里，单图支持 `gundam` 和 `base` 两种模式。多页和 PDF 走 `base`，把 PDF 先转成页面图片，再用 `infer_multi` 连续解析。输出里会用 `<PAGE>` 分隔页面。保存结果时，它会生成 `result.md`，也会处理 `<|ref|>` 和 `<|det|>` 这类定位标记。遇到 `image` 区域，后处理会按坐标从页面图里裁剪出来，放到 `images/` 目录，再在 Markdown 里替换成 `![](images/...)`。还会保存带框的 `result_with_boxes.jpg` 或多页版本，方便看模型到底框了什么。

这也不是简单的 OCR 文本了，它同时在输出阅读顺序、结构标记、坐标、图片裁剪和 Markdown，将 MinerU 在工程系统里所做的重新放回可以端到端训练的模型中。

用 Unlimited OCR 时，参数也不能随便。vLLM recipe 里写得很明确：prompt 要以字面量 `<image>` 开头；`skip_special_tokens=False`；服务端要注册 no-repeat n-gram logits processor；请求里传 `ngram_size=35`，单页窗口常用 `window_size=128`，多页或 PDF 用 `1024`。没有这些，长文档容易在坐标 token 上循环，或者直接空输出。作为一个发布仅半个月的模型，他足够强大，但在工程优化层面仍旧是一个孩子。

还有一点要冷静：Unlimited 不是物理意义上的无限。论文自己也说，32K 上下文仍然限制 prefill。页数越多，视觉 token 越多，前缀越长。R-SWA 解决的是长输出时 KV cache 一直长的问题，不是让模型凭空装下无限页面。

但这个方向仍然有价值。它把逐页循环这个工程习惯重新拿出来问了一遍：如果模型本身能连续转写，多页文档是不是可以少一些外部拼接？End to End 训练的魅力，我们已经见过不止一次了，他能否改变 OCR 可不好说。

## 它们能给用户什么

这两个项目放在一起看，会发现 OCR 的用户价值已经变了。

以前用户要的是文字。现在用户更想要一份可用的文档对象：正文有顺序，标题有层级，公式是 LaTeX，表格能继续解析，图片被裁出来，坐标还在，必要时能回到原页检查。最好还有调试文件，能告诉我哪里识别错了，而不是只给一份看似完整的 Markdown。

MinerU2.5-Pro 更擅长把这些能力组织成系统。它让我想到以前写过的事实层和界面层：Markdown 是给人读的，JSON 和中间文件才是后续系统继续工作的事实层。你可以用 Markdown 做展示和索引，用 `content_list.json` 做 chunk，用 `middle.json` 做二次开发，用 `layout.pdf` 和 `span.pdf` 做质量排查。

Unlimited OCR 更像是在模型层面补一个长期缺口。长文档不是很多页单页 OCR 的简单相加。连续性本身就是能力。它现在还需要专门 recipe，也有不少推理参数要守，但 R-SWA 这个想法很清楚：固定参考，滑动输出，让模型别被自己越写越长的历史拖住。

## 结语

如果要把它们放进工程里，我会先把期待压低一点。

数字 PDF 能直接抽文本就不要 OCR。扫描件、复杂版面、低质量图片页，再交给 OCR/VLM。MinerU 适合做文档解析主流程，但要认真看输出文件和 backend 差异。Unlimited OCR 适合长文档专项评测，尤其是几十页连续解析，但不能忽略 prompt、special token、no-repeat n-gram 和上下文上限。

更现实的做法，可能是混合管线。普通文档用成熟解析系统处理，长文档和复杂页交给更强的 VLM/OCR 模型；输出统一落到 Markdown 和 JSON，再加一层质量检查。但混合管线的开发成本和维护成本都是昂贵的，MinerU 与 Unlimited OCR 这种开箱即用的工具能够为我们省下来在 AI 时代最为宝贵的时间，**当模型足够便宜的时候，一切工作就都可以交给模型，无需区分复杂度和设计工具**，现在的 LLM 正在接近这一点。

MinerU2.5-Pro 让我看到 OCR 的工程化价值。不是所有能力都要塞进模型里，文档解析本来就需要文件处理、版面分析、后处理、输出协议和调试工具。

Unlimited OCR 提醒我们，模型结构仍然能改变工作流。长文档解析不一定永远是逐页循环加外部拼接。也许以后真正好用的 OCR 系统，会一边有 MinerU 这样的工程外壳，一边吸收 Unlimited OCR 这种长文档模型能力。

OCR 以前负责把字认出来。现在它开始负责把文档还原成可以继续使用的结构。Agent 想要走进现实世界，OCR 是离不开的一环，MOGA。

## 参考资料

- OpenDataLab, [MinerU2.5-Pro-2605-1.2B](https://huggingface.co/opendatalab/MinerU2.5-Pro-2605-1.2B)
- MinerU, [Output File Format](https://opendatalab.github.io/MinerU/reference/output_files/)
- MinerU, [Quick Usage](https://opendatalab.github.io/MinerU/usage/quick_usage/)
- OpenDataLab, [mineru-vl-utils](https://github.com/opendatalab/mineru-vl-utils)
- OpenDataLab, [MinerU2.5-Pro: Pushing the Limits of Data-Centric Document Parsing at Scale](https://arxiv.org/html/2604.04771v1)
- Baidu, [Unlimited-OCR](https://github.com/baidu/Unlimited-OCR)
- Baidu, [Unlimited OCR Works: Welcome the Era of One-shot Long-horizon Parsing](https://arxiv.org/html/2606.23050v1)
- vLLM Recipes, [baidu/Unlimited-OCR](https://recipes.vllm.ai/baidu/Unlimited-OCR)
