# LLM 记忆基准调研草稿

> 版本：2026-04-30  
> 状态：研究草稿，暂不并入现有 memory 研究。  
> 目标：回答“是否已经存在单独针对 LLM / Agent 记忆模块进行评估的开源基准”，并梳理这些基准暴露出的评价问题。  
> 证据说明：本文优先使用论文、OpenReview / ACL / ICML / arXiv 页面、官方项目页、GitHub 与 Hugging Face 数据集卡。第三方博客和产品页只作为线索，不作为核心结论依据。本轮没有下载所有 arXiv TeX 源码，因此对方法和实验细节只做论文摘要、官方仓库和项目页层面的归纳，不展开逐公式或逐表复核。

## 1. 研究问题与范围

这篇草稿只讨论一件事：**目前是否已经有专门评估 LLM 或 Agent 记忆能力的开源 benchmark，以及这些 benchmark 到底测的是哪一种“记忆”。**

这里的“记忆”不等同于长上下文能力。长上下文模型可以把 100k、1M 甚至更长的 token 塞进上下文窗口，但这只能说明模型有机会访问历史信息；真正的记忆系统还需要决定什么应该写入、如何组织、如何检索、如何更新、如何处理冲突、何时遗忘、如何在生成阶段使用这些信息。把完整聊天记录直接拼进 prompt 是一种极端基线，但不是一个完整的记忆模块。

因此本文把评估对象分成三层：

1. **长期对话记忆**：聊天助手跨 session 回忆用户事实、时间线、偏好、旧承诺和知识更新。
2. **Agent 记忆模块**：带外部 memory store、RAG、summary、scratchpad、graph memory、file memory 或工具化 memory API 的智能体，在持续交互中写入、召回、更新和使用记忆。
3. **记忆操作与结构**：把 memory workflow 拆成 extraction、update、retrieval、state tracking、structured organization、hallucination suppression、continual learning 等可诊断步骤。

本文不把 [LongBench](https://github.com/THUDM/LongBench)、[RULER](https://github.com/NVIDIA/RULER)、Needle-in-a-Haystack 这类长上下文基准作为主体。它们对理解背景很重要，但主要测的是模型在长输入中的检索、阅读、聚合和推理能力，不一定测一个独立 memory module 的生命周期。

## 2. 一页式结论

**结论一：已经存在多个单独面向 LLM / Agent 记忆能力的开源基准。**  
比较确定的开源代表包括 [LongMemEval](https://github.com/xiaowu0162/LongMemEval)、[LoCoMo](https://github.com/snap-research/locomo)、[GoodAI LTM Benchmark](https://github.com/GoodAI/goodai-ltm-benchmark)、[MemBench](https://github.com/import-myself/Membench)、[MemoryAgentBench](https://github.com/HUST-AI-HYZ/MemoryAgentBench)、[Minerva](https://github.com/microsoft/minerva_memory_test)、[HaluMem](https://github.com/MemTensor/HaluMem)、[MemoryBench](https://github.com/LittleDinoC/MemoryBench)、[StructMemEval](https://github.com/yandex-research/StructMemEval)、[MemoryArena](https://memoryarena.github.io/) 和 [MemGUI-Bench](https://lgy0404.github.io/MemGUI-Bench/)。这些工作不是简单的长文本 QA，而是明确把 memory 当成聊天助手或 agent 系统的核心部件来测。

**结论二：这些基准测的“记忆”并不相同。**  
LongMemEval 和 LoCoMo 更接近长期对话记忆；MemBench 和 MemoryAgentBench 更接近 Agent memory 能力横评；Minerva 更像可编程的上下文记忆操作单元测试；HaluMem 把记忆幻觉拆到操作级；MemoryBench 把记忆和服务期反馈、持续学习绑定；StructMemEval 关注记忆组织结构；MemoryArena 和 MemGUI-Bench 则把记忆放进行动任务、网页/购物/旅行规划/移动 GUI 等环境中。它们可以互补，但不能互相替代。

**结论三：当前最严重的问题不是“没有基准”，而是“归因不清”。**  
很多 benchmark 的最终指标仍然是 answer accuracy、F1、success rate 或 LLM-as-judge 分数。系统得分高，可能因为 memory extraction 好，也可能因为 retrieval 好，可能因为 reader LLM 强，也可能因为 prompt 或 judge 偏置。要评估 memory module 本身，必须把写入、压缩、索引、检索、更新、冲突解决、遗忘和生成阶段拆开。

**结论四：开源状态需要分层看。**  
多数代表工作已经发布代码、数据或评估脚本；但也有一些 2026 年新工作处于 preprint / workshop / project-page 阶段。比如 [BEAM](https://openreview.net/forum?id=y59hf5lrMn) 是 ICLR 2026 的重要长期记忆基准，论文公开且提出 100K 到 10M token 的多领域对话记忆评测，但本轮没有确认到官方 GitHub 或 Hugging Face code/data 入口，因此在本文中标为“论文确认、artifact 待确认”，不把它作为 confirmed open-source benchmark 计数。

**结论五：下一步如果要做 memory 研究，最有价值的方向不是再做一个单一 QA 集，而是做分阶段、可插拔、可归因的评估协议。**  
理想 benchmark 应该允许替换 memory writer、memory index、retriever、ranker、updater、consolidator、forgetting policy 和 final reader，并同时报告质量、延迟、成本、记忆污染、错误累积和跨 session 稳定性。

## 3. 检索策略与证据标准

本轮检索使用的英文关键词包括：

| 查询簇 | 目的 |
| --- | --- |
| `LLM agent memory benchmark` | 找 Agent memory 直接评估工作 |
| `long-term conversational memory benchmark` | 找跨 session 对话记忆基准 |
| `memory hallucination benchmark agents` | 找记忆幻觉、错误传播、冲突更新基准 |
| `continual learning memory benchmark LLM systems` | 找服务期反馈和持续学习类 memory benchmark |
| `memory structure LLM agents benchmark` | 找结构化记忆、组织方式、状态追踪评估 |
| `mobile GUI agents memory benchmark` | 找 GUI / 行动任务中的记忆评估 |

证据分层如下：

| 层级 | 使用方式 | 示例 |
| --- | --- | --- |
| confirmed | 论文或 OpenReview / ACL / ICML 页面与官方 repo / HF 数据互相印证 | LongMemEval、LoCoMo、MemBench、Minerva、MemoryBench、HaluMem |
| early | arXiv / OpenReview / workshop / 项目页已有，代码或数据公开，但仍处于早期版本 | StructMemEval、MemoryArena、MemGUI-Bench |
| needs_corroboration | 只有论文摘要、第三方页或 artifact 入口未确认 | BEAM 的开源 artifact 状态 |

这篇草稿不做排行榜。很多论文和项目页会给出模型分数，但不同 benchmark 的任务形态、judge、数据合成方式、上下文长度、成本约束和被测系统差异很大，直接排序会误导。本文只关心：每个 benchmark 能诊断什么，不能诊断什么，以及它在 memory module 评估中填补了哪个空白。

## 4. Benchmark map

### 4.1 长期对话记忆

长期对话记忆基准把用户和助手之间的多轮、多 session 对话作为主场景，考察系统能否跨越时间线保留事实、理解事件变化、回答时间相关问题，并在证据不足时弃权。

[LongMemEval](https://arxiv.org/abs/2410.10813) 是这一类中最清晰的代表。论文将长期聊天助手记忆拆成五类能力：information extraction、multi-session reasoning、temporal reasoning、knowledge updates 和 abstention。官方 [GitHub 仓库](https://github.com/xiaowu0162/LongMemEval) 和 [Hugging Face 数据](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned) 提供了代码和数据。LongMemEval 的重要之处在于它不只是问“历史里有没有某个事实”，还要求系统处理跨 session 推理、时间线和知识更新，并且提供 oracle retrieval 版本，方便区分检索失败和阅读失败。

[LoCoMo](https://snap-research.github.io/locomo/) 更偏向“非常长期的对话理解”。它通过 persona、temporal event graph 和人类校验构造长期对话，平均多 session、多轮，并设置 QA、event graph summarization 和 multi-modal dialog generation 等任务。官方 [GitHub](https://github.com/snap-research/locomo) 释放了数据和脚本。LoCoMo 的价值在于它不只看事实回忆，还把事件、因果、时间和多模态对话一致性放进评估；局限是规模较小，且作为对话数据集，它不能完全隔离 memory module 的内部操作。

[GoodAI LTM Benchmark](https://github.com/GoodAI/goodai-ltm-benchmark) 是更工程化的长期记忆测试库。仓库说明它用于测试 conversational agents 的 long-term memory 和 continual learning，包含诸如 prospective memory、locations/directions、shopping、restaurant、trigger-response 等任务，并支持 OpenAI、Anthropic、Gemini、MemGPT、GoodAI LTM agent 等不同 agent 接口。它的强项是可运行、可扩展、强调长对话里的动态记忆维护；弱项是论文体系和学术基准化程度不如 LongMemEval / LoCoMo 明确，更像可复现实验套件。

### 4.2 Agent 记忆模块综合评估

这一类 benchmark 的对象不只是单个 LLM，而是带 memory mechanism 的 agent。它们关心 agent 在持续交互中如何存储、更新、检索和利用历史信息。

[MemBench: Towards More Comprehensive Evaluation on the Memory of LLM-based Agents](https://aclanthology.org/2025.findings-acl.989/) 是 ACL Findings 2025 工作，官方 [GitHub](https://github.com/import-myself/Membench) 释放数据和项目。它的关键设计是把 memory content 分为 factual memory 与 reflective memory，把 interaction scenario 分为 participation 与 observation。换句话说，它不只测“用户明确说过什么”，也测 agent 能否从历史中推断偏好、画像或高层反思；同时区分 agent 主动参与对话时的记忆与被动观察信息流时的记忆。指标覆盖 effectiveness、efficiency、capacity 等维度。

[MemoryAgentBench](https://arxiv.org/abs/2507.05257) 明确提出“memory agents”概念，将 agent memory 评估组织为 incremental multi-turn interactions。官方 [GitHub](https://github.com/HUST-AI-HYZ/MemoryAgentBench) 已公开。论文把核心能力定义为 accurate retrieval、test-time learning、long-range understanding 和 selective forgetting，并指出静态长上下文 QA 难以反映 agent 在真实交互中逐步积累信息的方式。它的优势是把 memory 看成持续过程；需要注意的是，benchmark 中部分任务来自已有长上下文数据的多轮化改造，因此仍要警惕“数据集改写后是否真的模拟真实记忆流”的问题。

[MemoryArena](https://memoryarena.github.io/) 把 memory 放入 multi-session agentic tasks 中，而不是只在最后问一个回忆问题。它的任务包括 web navigation、preference-constrained planning、progressive information search 和 sequential formal reasoning。项目页提供 [Hugging Face 数据](https://huggingface.co/datasets/ZexueHe/memoryarena) 和 GitHub code 入口。它的核心贡献是强调“记住”和“行动”耦合：agent 在前面 session 中获得经验、反馈和约束，后面必须用这些记忆继续完成任务。这个方向很重要，因为很多实际 agent 失败不是因为它完全忘了事实，而是因为它没有把过去经验转化成后续行动策略。

### 4.3 记忆操作、结构和可编程单元测试

长期记忆的难点不只是信息量大，而是 memory store 必须有组织结构。没有结构的向量召回可以解决部分事实检索，但面对账本、待办、树状关系、状态机、长期偏好和更新日志时，简单 top-k retrieval 很快会暴露局限。

[Minerva: A Programmable Memory Test Benchmark for Language Models](https://www.microsoft.com/en-us/research/publication/minerva-a-programmable-memory-test-benchmark-for-language-models/) 是 Microsoft Research 的 ICML 2025 工作，官方 [GitHub](https://github.com/microsoft/minerva_memory_test) 公开。它与对话记忆 benchmark 不同，更像 memory/context use 的程序化单元测试。Minerva 包含 search、recall and edit、match and compare、spot the differences、compute on sets and lists、stateful processing 六类测试，以及 composite tests。官方仓库说明共有 21 个任务，并支持参数化生成新样本。它的价值在于可解释和可扩展：失败时更容易知道模型到底不会搜索、不会编辑，还是不会在 memory 上维护状态。

[StructMemEval](https://arxiv.org/abs/2602.11243) 是 2026 年 work-in-progress，官方 [GitHub](https://github.com/yandex-research/StructMemEval) 提供补充代码和原始 benchmark data。它关注“agent 是否能把长期记忆组织成合适结构”，而不是只做事实回忆。任务包括 accounting、tree-based、state tracking、recommendation 等结构化场景。它的意义在于把记忆从“检索若干片段”推进到“维护可操作的数据结构”。这对真实 agent 很关键：财务流水、旅行计划、购物约束、项目状态、用户偏好都不是孤立事实，而是持续变化的结构。

### 4.4 记忆幻觉、冲突和更新

记忆系统不只会忘，也会编。更麻烦的是，错误记忆一旦写入 memory store，就可能在后续 retrieval 和 generation 中反复污染输出。

[HaluMem](https://arxiv.org/abs/2511.03506) 专门评估 memory systems of agents 中的 hallucination。官方 [GitHub](https://github.com/MemTensor/HaluMem) 和 Hugging Face 入口已公开。它把评估拆成 memory extraction、memory updating 和 memory question answering 三个任务，并在项目页中报告 extraction recall / precision、false memory resistance、updating correctness、hallucination rate、omission rate 等指标。这个设计比端到端 QA 更有诊断价值：如果最终回答错了，可以进一步定位是提取阶段制造了假记忆、更新阶段没有覆盖旧信息，还是 QA 阶段错误整合了记忆。

这一类工作对 memory 研究特别重要。很多 memory benchmark 把“召回不到”当作主要错误，但真实系统还有几类更危险的问题：把用户没有说过的偏好写入长期记忆；把过期信息当成当前事实；面对新旧冲突时保留两份互相矛盾的记忆；把相似用户或相似事件的记忆串扰到当前用户；在总结压缩时把不确定推断写成确定事实。HaluMem 的价值就在于它把这些问题从终局回答中拆出来，进入 memory operation 层面。

### 4.5 持续学习和服务期反馈

长期记忆不一定只来自显式对话事实，也可能来自用户反馈。一个写作助手、搜索助手或推荐助手在服务期间可能不断收到用户的修改、复制、喜欢、不满意、追问等反馈。如何从这些反馈中形成可迁移的记忆，是另一类 benchmark 的重点。

[MemoryBench: A Benchmark for Memory and Continual Learning in LLM Systems](https://arxiv.org/abs/2510.17281) 关注 memory and continual learning in LLM systems。官方 [GitHub](https://github.com/LittleDinoC/MemoryBench) 和 [Hugging Face 数据集](https://huggingface.co/datasets/THUIR/MemoryBench) 已公开，且有 extended version [MemoryBench-Full](https://huggingface.co/datasets/THUIR/MemoryBench-Full)。它用 user feedback simulation framework 构造多领域、多语言、多任务反馈日志，并评估系统是否能在服务期从积累反馈中改进。它和 LongMemEval 的差异在于：LongMemEval 更像问“你是否记得过去发生了什么”，MemoryBench 更像问“你是否能从过去的用户反馈中学会之后怎么做得更好”。

这类 benchmark 暴露了 memory 和 continual learning 的边界问题。如果系统只是把反馈日志检索出来拼进上下文，那它是不是“学习”？如果系统把反馈归纳成用户偏好或任务策略，那归纳是否可解释、可撤销、可冲突更新？MemoryBench 的价值是把 memory 从事实保存扩展到服务期适应，但也带来新的评估难题：用户模拟器是否逼真、反馈是否覆盖真实产品场景、LLM-as-judge 是否稳定，都会影响结论。

### 4.6 行动环境、GUI 和跨任务记忆

只在文本问答中测 memory，容易低估 agent 使用记忆的难度。真实 agent 往往要在环境中行动：打开网页、填写表单、比较商品、执行移动 GUI 步骤、从失败中恢复、跨应用传递信息。这时 memory 不再只是回答问题的证据，而是控制策略的一部分。

[MemGUI-Bench](https://arxiv.org/abs/2602.06075) 面向 mobile GUI agents。官方 [项目页](https://lgy0404.github.io/MemGUI-Bench/) 提供 paper、code、tasks 和 trajectories 入口。它包含 128 tasks、26 applications，并强调 cross-temporal 与 cross-spatial retention。项目页给出 short-term memory 的 success rate、information retention rate、memory-task proficiency ratio，以及 long-term memory 的 pass@k、failure recovery rate 等指标。这个 benchmark 的价值在于把 memory 能力放到 GUI 操作链路中：agent 不仅要记得某个信息，还要在多个应用、多个步骤和多次尝试中正确使用它。

[MemoryArena](https://memoryarena.github.io/) 也属于这一方向，只是它的环境更偏多 session agentic task gym，而不是移动 GUI。两者共同说明一个趋势：memory benchmark 正在从“最后回答一个问题”移动到“历史经验是否能改变后续行动”。

### 4.7 超长规模对话记忆

[BEAM: Beyond a Million Tokens](https://openreview.net/forum?id=y59hf5lrMn) 是 ICLR 2026 工作，提出 100K 到 10M token 的多领域长期对话记忆 benchmark，并配套 LIGHT memory framework。OpenReview 页面显示 BEAM 包含 100 conversations 和 2,000 validated questions，目标是覆盖比简单 recall 更广的 memory probes。它对长期记忆研究很重要，因为它把规模推到真实长周期交互更接近的区间，也明确指出即便 1M token context window 模型在对话变长时仍会困难。

但本文对 BEAM 做保守标注：本轮确认到 OpenReview 论文入口，但没有确认官方 GitHub / Hugging Face code 或 data 入口。因此，BEAM 在本文中作为重要研究信号和 benchmark 设计参考，不作为“已确认开源 artifact”的例子。后续需要持续跟踪其 supplementary、作者主页或数据集发布状态。

## 5. 代表基准对照表

| Benchmark | 证据状态 | 开源状态 | 任务形态 | 核心评估维度 | 适合评估 | 不适合单独回答 |
| --- | --- | --- | --- | --- | --- | --- |
| [LongMemEval](https://github.com/xiaowu0162/LongMemEval) / [paper](https://arxiv.org/abs/2410.10813) | confirmed | GitHub + HF 数据 | 长期聊天历史中的 QA | 信息抽取、跨 session 推理、时间推理、知识更新、弃权 | 聊天助手长期记忆、RAG memory、oracle retrieval 对照 | 不能完整定位 memory writer / updater 内部错误 |
| [LoCoMo](https://snap-research.github.io/locomo/) / [code](https://github.com/snap-research/locomo) | confirmed | GitHub 数据和脚本 | 多 session 长期对话、事件图、多模态对话 | QA、事件总结、多模态生成、一致性 | 对话长期记忆、时间/因果理解 | 数据规模较小，不是模块级 memory workflow 诊断 |
| [GoodAI LTM Benchmark](https://github.com/GoodAI/goodai-ltm-benchmark) | confirmed | GitHub | 可运行长期记忆任务套件 | 动态记忆维护、长对话任务、成本和报告 | 工程化 agent memory 回归测试 | 学术 benchmark 标准化和人工数据说明较弱 |
| [MemBench](https://aclanthology.org/2025.findings-acl.989/) / [code](https://github.com/import-myself/Membench) | confirmed | GitHub + 数据链接 | Agent 记忆问答和长信息流 | factual / reflective memory，participation / observation，accuracy / recall / capacity / efficiency | Agent 记忆模块综合评估 | 仍以最终任务表现为主，操作级归因有限 |
| [MemoryAgentBench](https://arxiv.org/abs/2507.05257) / [code](https://github.com/HUST-AI-HYZ/MemoryAgentBench) | confirmed | GitHub | incremental multi-turn interactions | accurate retrieval、test-time learning、long-range understanding、selective forgetting | 逐步累积信息的 memory agent | 部分任务由已有数据改造，真实交互性仍需验证 |
| [MemoryArena](https://memoryarena.github.io/) | early | Project page + HF + GitHub 入口 | 多 session agentic tasks | 记忆与行动耦合、跨任务依赖、行动反馈 | Web / planning / search / reasoning 中的 agent memory | 不适合只比较纯聊天助手记忆 |
| [Minerva](https://www.microsoft.com/en-us/research/publication/minerva-a-programmable-memory-test-benchmark-for-language-models/) / [code](https://github.com/microsoft/minerva_memory_test) | confirmed | GitHub + benchmark snapshot | 可编程 memory/context 单元测试 | search、recall/edit、match/compare、diff、set/list、stateful processing | 诊断模型使用上下文记忆的原子能力 | 不直接模拟真实长期用户记忆 |
| [StructMemEval](https://arxiv.org/abs/2602.11243) / [code](https://github.com/yandex-research/StructMemEval) | early | GitHub | 结构化记忆任务 | ledgers、trees、state tracking、recommendations | 记忆组织结构、结构化 memory store | work in progress，结论应保守 |
| [HaluMem](https://arxiv.org/abs/2511.03506) / [code](https://github.com/MemTensor/HaluMem) | confirmed / early | GitHub + HF 入口 | memory extraction / update / QA | recall、precision、false memory resistance、hallucination、omission | 记忆幻觉、冲突更新、错误传播 | 不覆盖所有行动环境中的 memory use |
| [MemoryBench](https://arxiv.org/abs/2510.17281) / [code](https://github.com/LittleDinoC/MemoryBench) / [HF](https://huggingface.co/datasets/THUIR/MemoryBench) | confirmed | GitHub + HF | 服务期用户反馈模拟 | continual learning、多领域、多语言、多任务、反馈利用 | 从用户反馈中学习的 LLM system | 用户模拟器真实性和 judge 稳定性需单独评估 |
| [MemGUI-Bench](https://lgy0404.github.io/MemGUI-Bench/) / [paper](https://arxiv.org/abs/2602.06075) | early | Project page 提供 code/tasks/trajectories 入口 | 移动 GUI agent 任务 | pass@k、IRR、MTPR、FRR、progressive scrutiny judge | GUI agent 的短期/长期记忆和跨应用操作 | 不适合文本-only memory module 排名 |
| [BEAM](https://openreview.net/forum?id=y59hf5lrMn) | needs_corroboration for artifacts | 论文公开；官方 code/data 未确认 | 100K-10M token 长对话记忆 probes | 多领域、长规模、10 类记忆能力 | 超长对话记忆设计参考 | 本轮不把它记为 confirmed open-source benchmark |

## 6. 现有基准的共同问题

### 6.1 “记忆模块”与“基础模型能力”难以解耦

同一个任务分数可能来自很多环节。比如 LongMemEval 上的正确回答可能来自：

1. 记忆系统写入了正确事实。
2. 检索器找到正确 session。
3. reranker 把证据排到前面。
4. reader LLM 有足够长上下文和推理能力。
5. prompt 恰好鼓励模型引用正确证据。

如果 benchmark 只看最终答案，就很难判断到底是哪一层有效。HaluMem 和 Minerva 的价值就在于试图拆解操作，但整个领域还没有形成统一的 memory module trace 标准。

### 6.2 长上下文不是长期记忆

长上下文基线很有必要，因为它是“把所有历史都给模型”的强对照。但真实系统不能无限制依赖它：成本高、延迟高、隐私风险高、上下文污染严重，而且长上下文模型也会出现 lost-in-the-middle、时间混淆和冲突更新失败。GoodAI LTM Benchmark、LongMemEval 和 BEAM 都在不同程度上说明：拥有更长上下文窗口只是必要条件，不是充分条件。

一个更合理的评估协议应同时报告：

- full-context baseline
- oracle evidence baseline
- naive RAG baseline
- memory-system baseline
- memory-system with oracle writer
- memory-system with oracle retriever

这样才能知道系统失败是因为没有写入、没有召回，还是召回后不会读。

### 6.3 很多 benchmark 仍偏向 QA，行动记忆不足

QA 是最容易标准化的评估形式，但 agent memory 的最终价值往往不是“回答一个历史问题”，而是“之后做事时少犯错”。MemoryArena 和 MemGUI-Bench 的出现说明社区正在补这个缺口。未来 benchmark 应该更多测：

- agent 是否能从失败轨迹中学习；
- 是否能把用户偏好迁移到后续任务；
- 是否能跨工具、网页、文件和 GUI 保持一致状态；
- 是否能在执行计划时调用正确历史约束；
- 是否能避免把旧任务的约束污染到新任务。

### 6.4 记忆更新比记忆召回更难

早期 memory benchmark 常把“找回旧事实”当核心任务，但真实用户会改变计划、修正偏好、撤销授权、更新地址、改名、换工作、取消会议。系统必须知道新信息何时覆盖旧信息，何时保留历史版本，何时标记不确定。

LongMemEval 的 knowledge updates、HaluMem 的 memory updating、StructMemEval 的 state tracking 都指向同一个问题：记忆不是静态数据库，而是带时间戳、版本、冲突和置信度的状态。单纯 append-only memory 很容易积累冲突。

### 6.5 记忆幻觉是写入阶段的问题，不只是生成阶段的问题

如果 hallucination 只发生在最终回答中，可以通过 grounded generation 或拒答策略缓解。但如果 hallucination 被写进长期记忆，后续系统会把它当成事实。HaluMem 的操作级设计提醒我们：memory extraction 的 precision 和 false memory resistance 应该成为长期记忆系统的核心指标。

这也意味着 memory writer 不能简单地把“看起来重要的内容”总结成用户画像。它必须区分：

- 用户明确说过的事实；
- 系统推断出的偏好；
- 低置信猜测；
- 临时上下文；
- 已过期信息；
- 不能保存的敏感信息。

现有 benchmark 对这些边界的覆盖还不够系统。

### 6.6 LLM-as-judge 带来新的测量误差

MemoryBench、MemGUI-Bench、HaluMem 等工作都不可避免地引入自动评价或 LLM-as-judge。这样做可以扩展任务复杂度，但也会带来 judge bias、格式敏感、模型家族偏置和复现实验成本问题。一个更稳的 benchmark 应该提供：

- 机器可判定子任务；
- 人工校验集；
- judge prompt 和 judge model 固定版本；
- 多 judge agreement；
- 对答案等价性的明确规范；
- 小规模人工复核协议。

否则 memory 系统的提升可能只是更适配 judge。

### 6.7 数据污染和 benchmark gaming 会越来越严重

一旦 benchmark 公开，模型和 memory 系统都可能围绕它优化。Minerva 的程序化生成是一个方向：动态参数化任务可以降低过拟合。MemoryArena、MemGUI-Bench 这类环境任务也较难通过背数据直接刷分。长期看，memory benchmark 应该有公开 dev set、私有 test set、动态生成 test、版本化 leaderboard 和污染检测。

## 7. 对后续 memory 研究的启发

如果后续要围绕 LLM memory 做研究，建议把问题拆成三条线。

第一条线是**模块化评估协议**。不要只说“我们的 memory 系统在 LongMemEval 上更高”，而要报告每个环节的指标：

| 环节 | 需要测什么 |
| --- | --- |
| memory writing | 是否提取正确事实，是否写入虚假记忆，是否区分事实和推断 |
| memory representation | 是否保留时间、来源、置信度、版本和实体边界 |
| memory indexing | 是否支持实体、时间、事件、任务、偏好和结构化状态检索 |
| retrieval / ranking | 是否召回证据，是否把关键证据排前，是否引入噪声 |
| update / consolidation | 是否覆盖旧信息，是否合并重复，是否保留历史轨迹 |
| forgetting / pruning | 是否删除过期、无用、敏感或冲突信息 |
| final use | 是否忠实使用记忆，是否在证据不足时弃权 |

第二条线是**任务谱系扩展**。已有基准主要覆盖长期对话、agent 记忆、结构化状态、反馈学习和 GUI 行动，但还缺少一些真实产品中高频出现的问题：

- 多用户、多账号、多身份之间的记忆隔离；
- 隐私、授权和可删除记忆；
- 用户纠错后的记忆撤销；
- 低置信偏好和长期偏好之间的区分；
- 团队协作场景中的 shared memory；
- 代码、文档、项目状态这类专业工作记忆；
- agent 在工具调用失败后如何形成可复用经验；
- memory poisoning 与恶意用户注入。

第三条线是**成本和可靠性一起评估**。长期记忆系统不能只看 accuracy。生产系统更关心：

- 每轮写入成本；
- 检索延迟；
- memory store 增长速度；
- 压缩带来的信息损失；
- 不同用户规模下的索引成本；
- 错误记忆的回滚成本；
- 敏感信息误存风险；
- 离线 consolidation 对在线体验的影响。

GoodAI LTM Benchmark 和 MemoryBench 已经开始把成本、效率或持续学习纳入视野，但这个方向还远未统一。

## 8. 一个可用的评估框架草案

如果要基于现有 benchmark 设计一个新的 memory module 评估栈，可以把它做成五层。

### 8.1 Synthetic operation tests

用 Minerva 和 StructMemEval 风格的任务测底层能力：搜索、编辑、集合计算、状态跟踪、树结构、账本、待办列表。这里的优点是可控、可解释、可自动生成。缺点是不够像真实用户。

### 8.2 Long-term conversation QA

用 LongMemEval、LoCoMo、BEAM 风格的任务测长期聊天记忆：用户事实、跨 session 推理、时间线、知识更新、弃权。这里适合比较聊天助手 memory 能力，但需要 oracle retrieval baseline 来拆分检索和阅读。

### 8.3 Operation-level hallucination and update

用 HaluMem 风格拆 extraction、update、QA，重点测 false memory、omission、conflict handling 和 error propagation。这一层应该成为所有长期记忆系统的必测项，因为错误写入比回答错一次更危险。

### 8.4 Feedback and adaptation

用 MemoryBench 风格测服务期反馈：用户显式评价、隐式行为、修改建议、偏好迁移、多领域任务适应。这里需要谨慎设计用户模拟器和 judge。

### 8.5 Agentic environment use

用 MemoryArena 和 MemGUI-Bench 风格测 memory 是否改变行动：跨 session 任务、网页/GUI 操作、失败恢复、计划约束、跨应用信息传递。这里最接近产品 agent，但复现实验成本最高。

这五层不要合成一个总分。更合理的是输出一个 memory profile：

| 维度 | 例子 |
| --- | --- |
| recall-heavy | 能找回旧事实，但不一定会更新 |
| update-aware | 能处理新旧冲突，但可能检索慢 |
| structure-aware | 能维护状态和层级，但对自然对话弱 |
| hallucination-resistant | 写入精度高，但召回覆盖可能低 |
| action-useful | 能把记忆转化为后续行动策略 |
| cost-efficient | 延迟和 token 成本可控 |

## 9. 开放问题与持续跟踪清单

需要继续跟踪的 benchmark 和问题：

- [BEAM](https://openreview.net/forum?id=y59hf5lrMn)：确认官方 code / data 是否公开，若公开应补充 artifact 入口和 license。
- [MemoryArena](https://memoryarena.github.io/)：跟踪 arXiv 正文、GitHub 代码成熟度、HF 数据版本和 leaderboard。
- [MemGUI-Bench](https://lgy0404.github.io/MemGUI-Bench/)：跟踪 code/tasks/trajectories 的实际可复现程度，以及 Progressive Scrutiny judge 的稳定性。
- [StructMemEval](https://github.com/yandex-research/StructMemEval)：关注 work-in-progress 后续版本是否扩大任务规模、增加更多结构类型和人工校验。
- [HaluMem](https://github.com/MemTensor/HaluMem)：关注它是否成为 memory systems 的标准 hallucination suite，以及不同 memory vendor 是否复现实验。
- [MemoryBench](https://huggingface.co/datasets/THUIR/MemoryBench)：关注 user simulator 版本、full dataset、judge 模型和多语言评估一致性。

仍未解决的研究问题：

1. 如何把 memory writer 的错误和 retriever 的错误严格分离？
2. 如何评估 memory deletion、privacy request 和 user correction？
3. 如何构造动态 benchmark，避免公开数据被训练或专门优化？
4. 如何判断“推断出的用户偏好”是否可以写入长期记忆？
5. 如何在多用户、多角色、多项目场景中防止记忆串扰？
6. 如何把 memory benchmark 从 QA 扩展到真实工作流，同时保持可复现？
7. 如何在 accuracy、latency、cost、privacy、controllability 之间做统一报告？

## 10. 推荐阅读路径

如果只是想快速理解 LLM 长期记忆评估，可以先读：

1. [LongMemEval](https://arxiv.org/abs/2410.10813)：理解长期聊天助手记忆的基本能力拆分。
2. [LoCoMo](https://snap-research.github.io/locomo/)：理解长期对话、事件图和多模态对话一致性。
3. [MemoryAgentBench](https://arxiv.org/abs/2507.05257)：理解 agent memory 在 incremental multi-turn setting 下的评价方式。
4. [HaluMem](https://arxiv.org/abs/2511.03506)：理解为什么 memory hallucination 必须拆到 operation level。
5. [Minerva](https://github.com/microsoft/minerva_memory_test)：理解可编程 memory/context 单元测试如何设计。

如果关注产品化 agent，可以再读：

1. [GoodAI LTM Benchmark](https://github.com/GoodAI/goodai-ltm-benchmark)：看工程化长期记忆测试如何落地。
2. [MemoryBench](https://github.com/LittleDinoC/MemoryBench)：看服务期用户反馈和持续学习评估。
3. [MemoryArena](https://memoryarena.github.io/)：看记忆如何影响后续行动。
4. [MemGUI-Bench](https://lgy0404.github.io/MemGUI-Bench/)：看移动 GUI agent 中的短期和长期记忆。
5. [StructMemEval](https://github.com/yandex-research/StructMemEval)：看结构化记忆组织的评估方向。

## 11. 参考文献与官方入口

- LongMemEval: [arXiv](https://arxiv.org/abs/2410.10813), [GitHub](https://github.com/xiaowu0162/LongMemEval), [Hugging Face](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned)
- LoCoMo: [project page](https://snap-research.github.io/locomo/), [GitHub](https://github.com/snap-research/locomo), [arXiv](https://arxiv.org/abs/2402.17753)
- GoodAI LTM Benchmark: [GitHub](https://github.com/GoodAI/goodai-ltm-benchmark)
- MemBench: [ACL Anthology](https://aclanthology.org/2025.findings-acl.989/), [arXiv](https://arxiv.org/abs/2506.21605), [GitHub](https://github.com/import-myself/Membench)
- MemoryAgentBench: [arXiv](https://arxiv.org/abs/2507.05257), [GitHub](https://github.com/HUST-AI-HYZ/MemoryAgentBench)
- Minerva: [Microsoft Research](https://www.microsoft.com/en-us/research/publication/minerva-a-programmable-memory-test-benchmark-for-language-models/), [OpenReview](https://openreview.net/forum?id=ib9drlZllP), [GitHub](https://github.com/microsoft/minerva_memory_test)
- HaluMem: [arXiv](https://arxiv.org/abs/2511.03506), [GitHub](https://github.com/MemTensor/HaluMem)
- MemoryBench: [arXiv](https://arxiv.org/abs/2510.17281), [GitHub](https://github.com/LittleDinoC/MemoryBench), [Hugging Face](https://huggingface.co/datasets/THUIR/MemoryBench), [MemoryBench-Full](https://huggingface.co/datasets/THUIR/MemoryBench-Full)
- StructMemEval: [arXiv](https://arxiv.org/abs/2602.11243), [OpenReview workshop page](https://openreview.net/forum?id=a9vY2sJkf4), [GitHub](https://github.com/yandex-research/StructMemEval)
- MemoryArena: [project page](https://memoryarena.github.io/), [Hugging Face data](https://huggingface.co/datasets/ZexueHe/memoryarena)
- MemGUI-Bench: [project page](https://lgy0404.github.io/MemGUI-Bench/), [arXiv](https://arxiv.org/abs/2602.06075)
- BEAM: [OpenReview](https://openreview.net/forum?id=y59hf5lrMn), [PDF](https://openreview.net/pdf?id=y59hf5lrMn)
