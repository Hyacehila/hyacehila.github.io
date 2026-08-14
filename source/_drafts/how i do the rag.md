# 思考一下我在做 RAG 的工程实践。 

## 关于 POPO Bot

### 事实层

在保持目录结构的基础上，我们收集到的原始数据包括 

.pptx  .docx   .pdf   .md   .xlsx  .html

这就是我们目前系统解除的全部数据类型，也只有这些类型的文件会加入数据处理的管线

Docling 可以让我们更好的去完成这种多格式数据的清洗，docling document 是我们最终真实信息源的存储地，他是不可更改的事实层。

对于 docx pdf md html 他们都是去构造一个完整的文档树，用一个树状的结构存储信息

对于 xls 等；他们实际上更多的存储的公司内部的排班，排期表格等内容，我们对他们不进行任何切分，使用相关 python 解析以后，脚本转换为一个 docling document，他只存储一个 table 节点（哪怕表有题头，不出来，脚本有时候不能解析这个，我们就是存储原始表）

对于 pptx 文件，他最可能是缺乏结构的，更多的只是信息与内容的堆叠，但 docling document 永远只作为事实层，我们不可以在事实层让 AI 辅助我们去解析内容，这违背了事实层属于事实的核心要求。

经过了以上处理，事实层本身应该就可以被完整的构建了。

参考内容

PPTX 的参考结构表达，缺乏内部章节结构，因此按页切分章节

DoclingDocument
└── body
    ├── group(label="chapter", name="slide-0")
    │   ├── text
    │   ├── picture
    │   └── table
    ├── group(label="chapter", name="slide-1")
    │   ├── text
    │   └── picture
    └── group(label="chapter", name="slide-2")
        ├── text
        ├── picture
        └── table

{
  "body": {
    "self_ref": "#/body",
    "children": [
      { "$ref": "#/groups/0" }
    ]
  },
  "groups": [
    {
      "self_ref": "#/groups/0",
      "label": "chapter",
      "name": "slide-0",
      "parent": { "$ref": "#/body" },
      "children": [
        { "$ref": "#/texts/0" },
        { "$ref": "#/pictures/0" }
      ]
    }
  ],
  "texts": [
    {
      "self_ref": "#/texts/0",
      "label": "paragraph",
      "text": "幻灯片中的原始文本",
      "parent": { "$ref": "#/groups/0" },
      "prov": [
        {
          "page_no": 1,
          "bbox": {
            "l": 0,
            "t": 0,
            "r": 0,
            "b": 0,
            "coord_origin": "BOTTOMLEFT"
          }
        }
      ]
    }
  ]
}

XLSX 结构示意

DoclingDocument
└── body
    ├── group(label="sheet", name="目录")
    │   ├── table
    │   └── picture
    ├── group(label="sheet", name="排班表")
    │   ├── table
    │   └── picture
    └── group(label="sheet", name="字段说明")
        └── table

表内 tables[].data.table_cells[] 

再嵌套 

{
  "row_span": 1,
  "col_span": 1,
  "start_row_offset_idx": 4,
  "end_row_offset_idx": 5,
  "start_col_offset_idx": 2,
  "end_col_offset_idx": 3,
  "text": "夜班",
  "column_header": false,
  "row_header": false,
  "row_section": false
}

使用 openpyxl 在这里作为构建器，不是第二事实数据库，经过了转写脚本进行合并。

其余事实层采用标准嵌套，这就是标准文档

body
├── text(label="section_header")
├── group(label="section")
├── group(label="list")
├── table
└── picture

DoclingDocument 是唯一结构化事实层；各格式解析器只负责把源文件忠实映射为它。检索、切分、嵌入、重排、Agent 与生成回答都只能是事实层之上的可重建投影或查询时行为，绝不能反向污染事实层。

### 增强层与检索资料

增强层需要和被检索的资料层一起思考进行构建

Enrichment 层是整个系统中不可缺少的部分，图需要 Enrichment 表需要 Enrichment 否则他们难以被召回，如果希望构建结构化索引，则还需要单独的局部 summary 的 enrichment

Enrichment 层 也不应该被和事实层混在一起，否则是对严重的事实与幻觉层混合。

图表解释文字是正常情况下最大的 Enrichment ；他们是默认需要进行的，而关于图表的处理是我唯一一个尝试修改默认切片器的地方，在默认行为下图和 Caption 放到一起，表根据长度切分；在我的自定义行为下，图和表都只包括 Captain 和 AI 的 Summary 作为增强后的可被检索层。

下面内容不检索，用了一个小的 VLM 判断：
Logo
封面背景
装饰性图片
头像
编辑器 chrome
无图注、无正文引用、尺寸很小的图标

我们在其他地方依旧复用 HybridChunker 保持 HybridChunker 的完整上下文感知逻辑： 不需要自己写复杂的 Token 计算、跨页合并逻辑。Metadata 溯源不会丢失： 只要在自定义序列化时保留 span_source=item，生成的 Chunk 的 meta.doc_items 依然会包含指向原文档 TableItem 或 PictureItem 的指针。  最终我们就可以得到一个 Chunker 层，他们是增强层的第一步。 PPTX 格式在正常情况下，都会被按照页生成 Chunker ，默认的 HybridChunker 即使遇到非常小的 Chapter（章节），也绝对不会将跨章节的内容合并到一个 Chunk 中。

增强层的第二步是面向局部 Chunker 可能被查询到的总结，这是针对 Chunker 层级的，表格和图片不考虑二次总结直接复用。每个 Chunker 会让 AI 进行一个简单的 Summary 在 Enrichment 层提前做重度预处理的思路。通过在构建期（Build-time）消耗算力去提纯知识，从而换取检索期（Run-time）极高的召回率和信噪比。

至于他们要如何进入查询库，包含三种思路
* 只检索 Summary
* Summary 和 Content 层拼接
* Summary 和 Content 层独立
我们使用了第二种，但在基础测试下，没有感觉到区别，

注意 Chunker 已经逐渐的离开的事实层，变成了增强后资料层的一部分。参考下面的格式

from docling_core.transforms.chunker.base import DocChunk
from docling_core.transforms.chunker.base import DocMeta

DocChunk(
    # 1. 纯文本载体：这里就是你自定义序列化器重写的占位符文本
    # 它是最终被 Embedding 模型向量化，并存入 Vector DB 的文本内容。
    text="\n\n[🖼️ 图片说明: 2026年Q3财报柱状图 | 注释: Q3总营收达到峰值，环比增长15% | 引用路径: #/pictures/3]\n\n",
    
    # 2. 核心元数据层：记录了这个 Chunk 的身世
    meta=DocMeta(
        # 继承的上下文：说明这张图表存在于文档的哪一个章节下
        headings=["第三部分：财务表现", "3.2 季度营收分析"], 
        
        # 文档来源信息：哪个文件、哈希值等
        origin=DocumentOrigin(
            filename="Q3_Financial_Report.pdf",
            mimetype="application/pdf",
            binary_hash="abc123xyz..."
        ),
        
        # 3. 事实追溯层 (最重要的一环！)
        # 里面包含了这个 Chunk 文本对应的所有物理节点引用
        doc_items=[
            DocItemRef(
                self_ref="#/pictures/3",  # 核心指针：指向原文档中的第3个图片对象
                label="picture",          # 节点类型：明确这是一个图片
                # 物理溯源 (Provenance)：原图片在 PDF 上的精确位置，用于前端高亮或裁剪原图
                prov=[
                    ProvenanceItem(
                        page_no=12,       # 在 PDF 的第 12 页
                        bbox=BoundingBox(l=72.5, t=150.0, r=530.2, b=420.5) # 边框坐标
                    )
                ]
            )
        ]
    )
)

超过 1500 切分，此时 overlap 200，不设置最小，取决于章节划分情况。

结构化 Agentic RAG 单独构建增强层，没有 Chunker 结构，纯抄袭 PageIndex 实现，利用原本的结构树信息，做group 和 section 的 summary。 然后给查询 Agent 暴露一组简单的工具去实现这样的结构化索引。

### 查询层

混合检索（然后回到事实层），自动附带 windows 上下文结构，如果是Agentic RAG 那么则允许自主决定是否需要向前向后获取多少信息，一定要回到事实层。

在这里我们只考虑两个事情，查询的重写和不同源结果的聚合。

重写基于预制提示词实现，需要策划写，我们这随便 ai 写一个

聚合采用 RRF 聚合。粒度为 Chunker 但是需要把最后回传事实层的相关内容，一切最终回答基于事实层。在事实层会允许适当的 context 扩展，在非 Agentic RAG 情况下扩展上下文的 5 条原始信息，在 Agentic 的情况下允许模型自主选择如何扩展。 对于 Agentic 的检索，由默认最终决定那几条信息作为事实层介入回答。  本质上所有系统最后应该都是一个 llm 作为 reranker。

## 关于如何构建一个游戏内部的数据库

这个数据库可能面向策划使用和给玩家回答问题，本质上都是希望让不理解游戏的人能够获得一个更快的理解。

在讨论数据库的时候，最关键的问题依旧是数据，如何或许游戏内部的数据？ 玩法是什么样的，有什么物品，描述又是什么样的。

直接使用策划一直使用的文档仓库是不靠谱的，太多的冲突要处理，策划文档本身存在很多前后矛盾和后面需求修改了以后文档仓库没有对应修改。

这里采用了从 trunk 代码中获取更可靠信息的方法。 我们从 UI 工程文件入手，去看相关策划配表的加载，从代码仓库中提取各种不同类型的实体，然后通过本身的类属性，形成一个大致的分类。比如物品，制造配方，原料，地区，玩法，怪物等。这里的具体分类是 AI 和策划一起做的，最终的结果是构建了一个局部地区有二级分类，大部分地区只有大分类的架构。 我们得到了大量的节点3w，部分节点之间存在边的连接，边全部的单向的且不存在属性。这些边依旧从代码中获取，边本身也可能是节点，但我们在系统中没有处理这个问题，且并不是所有的代码的边的关系都被建模了，类似关卡内的触发器我们就没有处理，总边的数目和节点数目类似。

单个节点大致的信息有节点本身的名字，在策划配表中的描述文字，分类 metadata 和连接边的相关信息。 边的建立更多的基于程序上的一些关系，而不是由 AI 或者 人工进行标注。

从以上结构能看出，他本质是一个类似 LLM Wiki 的图谱，但此时图谱为了降低成本和保持总结可信度是通过代码与程序关系建立的。我们也从使用 markdown wiki 改为使用 JSON Wiki 每个节点依旧保留一定的树状分类层级结构。这就是系统的知识层。 系统后续根据这个只是层进行回答。

为了保证整个系统能够冷启动，节点库总分类约20，让模型结构化索引不现实，因此我们需要类似 LLM Wiki 的冷启动思路，从这里开始，其实整个系统的架构都参考了 LLM Wiki 的相关实现。节点默认成为了最小 Chunk 单元。全程使用 Agentic 检索思路。经典结构如

```json
{"action":"tool","tool":"wiki.search","query":"检索词"}
```

```json
{"action":"tool","tool":"wiki.read_page","path":"wiki/example.md"}
```

graph.search

```json
{"action":"final","answer":"最终回答"}
```

解析动作、执行工具，把结果作为 observation 加入下一轮上下文。模型随后可以：

- 换一个查询词继续搜；
- 阅读已发现页面的完整内容；
- 开始在图上 search；
- 判断证据足够并结束。

wiki.search 重写查询，依旧 sqlite 两路 RRF 聚合关键词排名和向量选出节点后展开图查询。 允许 Agent 在图查询后回答 wiki.search ； 这是一个非常 Agentic 的 RAG 和我们的前一个系统思路并不一样。允许在图谱多跳。

最终的回答层，会给所有路线都预留份额，sqlite 两路 RRF 聚合的结果，单路 top-k 也可以返回。Graph 这里由 Agent 决定是哪些返回（Agentic 是这样的）。  在条件允许的情况下，都是一个llm 作为 reranker 。 

后面的尝试，边可以有属性

可以走 Tree Search

回答需要证据约束，查询需要 budget 