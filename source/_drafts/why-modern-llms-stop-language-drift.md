---
title: "为什么现代大模型不再频繁跳语言：从分词器、表示空间到激活动态"
date: 2026-04-26 22:00:00 +0800
categories: ["AI", "Research"]
tags: [research, llm, multilingual, tokenizer, interpretability, language-drift]
author: Hyacehila
excerpt: "现代大模型很少在中文对话中突然切到英文，并不只是因为系统提示词写了“请用中文回答”。更底层的解释要同时看 tokenizer 成本、跨语言表示空间、语言特异性激活与解码阶段的控制。"
mathjax: true
---

很多人对早期大语言模型还有一个直观印象：用中文问着问着，模型突然切到英文；让它写一篇中文长文，前几段还好，后面夹杂英文术语、英文解释，甚至整段变成英文。现在的主流对话模型很少在普通场景里这样失控。它们通常会沿着用户的语言继续回答，最多是在专业术语、代码、引用标题或固定名词处夹一点英文。

这看起来像一个已经被工程解决的小问题，但其实背后牵涉到一个很深的机制问题：

**模型在生成时到底如何“选择语言”？它为什么会保持在中文、英文、阿拉伯文或俄文上，而不是滑向训练语料最多、token 最省、知识最密的英文？**

我目前更倾向于把答案拆成三层。

第一层是 **tokenizer 层**：现代 tokenizer 对非英语文本的切分不再那么灾难，语言之间的 token 成本差距被缩小，至少不会让中文或其他语言从输入开始就处在极端劣势。

第二层是 **表示空间层**：多语言模型不是简单把所有语言混进一个完全统一的空间，也不是每种语言完全隔离。更像是中间层形成某种共享语义区域，同时保留语言、脚本、词表和输出方向上的子空间结构。

第三层是 **激活动态和解码层**：语言选择并不只是 prompt 里的一个规则，而是可以在 neuron、residual stream、logit mass 和最终词表分布里被观测和干预的动态过程。近年来一批 language-specific neurons、activation steering 和 multilingual RAG drift 的工作，已经开始把这个问题从经验现象推进到可测机制。

这篇文章是一个 blog 草稿，也是一份研究路线笔记。它不会把“模型不跳语言”归因给某一个万能机制。更准确的说法是：现代模型之所以稳定，是因为输入成本、表示几何、语言控制电路、对齐训练和系统工程共同把语言漂移的概率压低了。

## 先定义问题：语言漂移不是一个现象

“跳语言”至少包含几种不同现象。

第一种是 **output language drift**：用户、系统指令和示例都要求中文，模型最后却输出英文。这是最典型的失败。

第二种是 **code-switching**：模型在同一段回答里混用多种语言。它不一定是错误，因为真实人类语言本来就可能夹杂外来词、术语、代码或引用。问题在于模型是否在不该混合时混合。

第三种是 **RAG 场景的证据语言漂移**：用户用中文问，检索到的资料是英文，模型回答时被资料语言带走。2025 年的 [Language Drift in Multilingual Retrieval-Augmented Generation](https://arxiv.org/abs/2511.09984) 就专门研究了这个场景。它的关键设定是固定 query、prompt 和 in-context examples 的目标语言，只改变 retrieved context 的语言，结果发现跨语言 evidence 会显著影响输出语言一致性。

第四种是 **长生成中的 switch point**：模型开头遵守目标语言，生成到某个段落、某个推理步骤或某个领域术语密集区后突然切换。这类问题最适合用 activation dynamics 来分析，因为我们可以寻找切换前后 residual stream、logit mass 或 language-specific neuron 的变化。

第五种是 **隐性英语中介**：模型表面输出中文，但中间层的可解码表示或语义决策更接近英文。这不等于模型“真的先翻译成英文再翻译回来”，但它提示我们，英语可能在表示空间里承担了某种中心吸引子的角色。这个争论可以从 [Do Llamas Work in English?](https://arxiv.org/abs/2402.10588) 和 [Do Multilingual LLMs Think In English?](https://arxiv.org/abs/2502.15603) 开始看。

所以，现代模型“不会跳语言”并不是绝对现象。更准确的命题是：

**在普通对话、低温采样、指令清晰、上下文单语一致的条件下，现代模型的目标语言保持能力已经非常强；但在长 CoT、跨语言 RAG、多脚本输入、低资源语言、弱提示和高温采样下，语言漂移仍然是可观察、可诱发、可研究的问题。**

## 一页式结论

我先给出本文的核心判断。

| 结论 | 证据强度 | 说明 |
| --- | --- | --- |
| Tokenizer 会系统性改变不同语言的计算成本和下游表现 | confirmed | [The Token Tax](https://arxiv.org/abs/2509.05486) 在 AfriMMLU 上发现 fertility 与准确率负相关；[Beyond Fertility / STRR](https://arxiv.org/abs/2510.09947) 进一步指出 fertility 不足以解释词表分配。 |
| 多语言表示空间同时包含语言敏感轴和语言中性轴 | confirmed | [The Geometry of Multilingual Language Model Representations](https://arxiv.org/abs/2205.10964) 在 XLM-R 中发现 language-sensitive 与 language-neutral axes 并存。 |
| 模型中存在可识别、可干预的语言相关神经元或特征 | confirmed / early | [Language-Specific Neurons](https://arxiv.org/abs/2402.16438) 用 LAPE 识别语言特异性神经元并做激活/去激活；后续 language arithmetics、CLAS、Neural FOXP2 做了更强的 steering 设定。 |
| 英语在很多开源模型内部是强吸引子 | early | Llama、Gemma、Mistral 等模型上有 logit lens、latent space、RAG drift 证据，但不同模型和不同任务并不完全一致。 |
| SFT、RLHF、system prompt 很可能显著压低可见语言漂移 | plausible | 这是产品系统中非常合理的工程解释，但对 GPT-4、Gemini、Claude 等专有模型的内部训练细节缺少公开可验证证据。 |
| “现代模型不跳语言”不是一个已经彻底解决的科学问题 | confirmed | 多语言 RAG、长 CoT、低资源语言和跨脚本输入仍会诱发 drift；只是普通 chat 场景被多层机制压住了。 |

最重要的一点是：不要把语言保持理解成一个简单 prompt 规则。它更像一个分布式控制问题。输入端的 tokenizer 决定某种语言进入模型时有多“贵”；中间层决定语义是否能在跨语言空间里稳定表达；输出端的语言特异性神经元、residual direction 和 logits 决定模型最后沿哪条语言路径落到词表上。

## 第一层：tokenizer 改变语言的“物理成本”

早期模型对中文、阿拉伯文、印地语、很多非拉丁脚本语言并不友好。最粗暴的情况是 byte-level fallback：一个语义上很简单的字或词，在 tokenizer 看来会变成多个 byte token。这样一来，同样长度、同样语义密度的一句话，在模型内部会变成更长的 token 序列。

这不是小问题。Transformer 的注意力计算随序列长度增长而变贵；更重要的是，模型在同样上下文窗口里能看到的信息减少了。一个语言如果需要更多 token 才能表达同样内容，就会同时付出三种代价：

- 训练和推理更贵；
- 有效上下文更短；
- 每个语义单位更容易被切碎，导致表示更难稳定。

[The Token Tax: Systematic Bias in Multilingual Tokenization](https://arxiv.org/abs/2509.05486) 把这个问题量化为 token tax。论文在 AfriMMLU 上评估 10 个模型和 16 种非洲语言，使用 fertility，也就是每个词平均被切成多少 token，作为指标。它报告的核心发现是：fertility 越高，模型准确率越低，并且这种关系在多个模型和科目上稳定出现。论文还把 token inflation 转化成训练、推理、延迟和成本问题，强调这不是单纯的 tokenizer 美学，而是会转化成真实能力差距。

但是 fertility 也不是万能指标。[Beyond Fertility: Analyzing STRR as a Metric for Multilingual Tokenization Evaluation](https://arxiv.org/abs/2510.09947) 指出，fertility 只是平均 token 数，容易掩盖“词表到底给哪些语言保留了完整词”的问题。它提出 Single Token Retention Rate，简称 STRR，用来衡量一个词是否被保留为单个 token。这个指标更接近词表容量分配：英语高频词是否经常是单 token？中文、印地语、法语、德语等语言是否得到同样待遇？论文发现英语在很多 tokenizer 中被优先保留，中文在一些现代 tokenizer 中已有较强支持，而印地语等语言仍然存在明显碎片化。

这对语言漂移有什么意义？

我的理解是，tokenizer 不一定直接决定模型会不会跳语言，但它决定了目标语言路径的阻力。如果中文被切得很碎，而英文对应概念只需要一个或少数几个 token，那么在生成阶段，英文路径天然更短、更密、更常见。模型遇到复杂概念时，英文 token 的概率质量可能更容易集中起来。

现代大模型的 tokenizer 往往有更大词表和更好的多语言覆盖。例如 Llama 3 系列、Qwen、DeepSeek、GPT-4o 一类模型都显著扩展了多语言 token 覆盖。这样做带来的效果不是“模型突然懂中文”，而是降低中文进入和离开模型时的结构成本。

不过这里需要保守一点。Tokenizer 改进能解释非英语语言更容易被稳定处理，但它不能单独解释现代 chat 模型为什么遵守“用用户语言回答”。即使 tokenizer 很好，如果输出端语言控制弱，模型仍然会 drift；反过来，tokenizer 很差的语言也可能靠强提示和强对齐勉强维持输出语言。

所以 tokenizer 是必要背景，不是充分原因。

## 第二层：表示空间不是统一世界语，而是共享语义加语言边界

如果只看 tokenizer，问题还停留在输入输出表面。更深的问题是：文本进入模型后，不同语言的表示在哪里相遇？

有一种很诱人的直觉：多语言模型会把“苹果”“apple”“pomme”映射到同一个语言无关概念空间里，然后再从这个概念空间投影回目标语言。这个说法很有解释力，但过于干净。近几年的表示空间研究显示，真实情况更复杂。

[The Geometry of Multilingual Language Model Representations](https://arxiv.org/abs/2205.10964) 以 XLM-R 为例，分析 88 种语言的 contextual representations。它的核心发现是：语言表示在 mean-centering 之后有相似的线性子空间，但不同语言的均值方向仍然携带语言敏感信息，比如词表相关信息。也就是说，模型确实有跨语言共享结构，但语言身份没有消失。论文还区分了 language-sensitive axes 和 language-neutral axes：前者能聚类语言家族或词表信息，后者能编码词性、位置等跨语言共享特征。

这给“为什么不跳语言”一个很重要的几何解释：现代模型并不是在一个完全混合的语义汤里生成。它们在中间层可以共享语义，但在表示空间里仍保留目标语言方向、脚本方向、词表方向和输出方向。只要最后几层能正确沿着目标语言方向投影，模型就能把共享语义落回中文，而不是落到英文。

更近期的 [High-Dimensional Interlingual Representations of Large Language Models](https://arxiv.org/abs/2503.11280) 进一步提醒我们，所谓 interlingua 并不是一个完全统一的区域。论文分析 31 种语言，提出 Interlingual Local Overlap 来衡量不同语言表示的局部邻域重叠。它的结论更像是“部分对齐”：高资源语言、同语系语言、相近地区语言之间可能有更好的局部对齐，低资源或类型差异较大的语言存在更明显的碎片化。

这对语言漂移的启发是：如果目标语言在 interlingual space 中对齐得好，模型可以在中间层共享语义，又能在输出层回到目标语言；如果对齐差，模型可能更容易退回高资源语言，尤其是英文。

另外还有一个新的提醒：[Multilingual Language Models Encode Script Over Linguistic Structure](https://arxiv.org/abs/2604.05090) 发现，语言相关单元很大程度上受正字法和脚本影响。论文用 LAPE、Sparse Autoencoders、romanization、word-order shuffling 和 causal intervention 分析多语言模型，结论是 language-associated units 往往更像 script-bound units。把非拉丁文字 romanize 后，激活集合可能和原脚本几乎分离，也不简单等同于英语。

这意味着我们不能把语言表示只理解成“语义”。中文、日文、韩文、印地语、阿拉伯文这类脚本差异，会在模型内部形成明显的路由边界。现代模型能保持中文，可能不仅因为它“知道用户在说中文”，还因为中文脚本本身提供了强 surface-form anchor。

## “模型在内部用英语思考”这件事要小心说

很多关于语言漂移的讨论会走向一个强说法：模型内部其实在用英语思考，然后最后翻译成中文。

这个说法有一部分证据，也有一部分风险。

[Do Llamas Work in English?](https://arxiv.org/abs/2402.10588) 是这条线的代表。作者构造非英语 prompt，使正确 continuation 可以被设计成单 token，然后用 logit lens 观察中间层。如果 prompt 要求法语到中文的翻译，最终层会把正确中文 token 排到高位，但中间层常常先能解码出英文版本。论文把 forward pass 分成三个阶段：早层还远离 output token embedding，中间层已经有语义正确的 token 但更接近英文，最后层进入输入语言或目标语言特定区域。

[Do Multilingual LLMs Think In English?](https://arxiv.org/abs/2502.15603) 把问题扩展到开放式多 token 生成，并比较 Llama、Gemma、Aya、Mistral 等模型。它发现 lexical words 更常经过英语中心的表示，而 non-lexical words 不一定如此；英语 steering vectors 在非英语生成中有时比目标语言 vectors 更有效；跨语言事实表示有共享区域，但插值时输出往往更偏英文。

这些结果说明：至少在一些开源模型、任务和层级上，英语确实是强中心。

但要避免两个过度结论。

第一，不要说所有模型都“先翻译成英文”。[Do Llamas Work in English?](https://arxiv.org/abs/2402.10588) 自己也更谨慎地把中间层解释为 concept space closer to English，而不是一个离散的翻译再翻译流程。

第二，不同模型差异很大。多语言训练更强、词表更均衡、对齐更充分的模型，可能不像英语中心语料训练出的模型那样明显依赖英文吸引子。比如 Aya 这类多语言导向模型和 Gemma/Llama 的表现就不完全一样。

所以，我会把“内部英语中心性”写成一个重要但非普适的机制假设：

**英语在许多模型的表示空间和解码分布中是高密度吸引子；语言漂移常表现为目标语言约束减弱后，生成轨迹回落到这个吸引子。但这不是所有模型、所有语言、所有任务的固定规律。**

这个说法也解释了为什么现代 chat 模型平时不跳语言：它们不是没有英文吸引子，而是目标语言的输出约束、系统提示和对齐训练足够强，能把生成轨迹从英文吸引子旁边拉回中文路径。

## 第三层：语言选择可以在激活里被观测和操纵

如果语言保持只是 prompt 规则，我们很难解释为什么小小的 activation intervention 能改变输出语言。但最近的机制可解释性研究显示，语言选择确实有可观测的内部控制点。

[Language-Specific Neurons: The Key to Multilingual Capabilities in Large Language Models](https://arxiv.org/abs/2402.16438) 提出 LAPE，也就是 Language Activation Probability Entropy，用来找对特定语言更常激活的神经元。论文在 LLaMA-2、BLOOM、Mistral 等模型上发现，少量语言特异性神经元对多语言能力有显著影响，而且这些神经元主要集中在底层和顶层。作者还通过激活或去激活这些 neurons 来影响输出语言。

这个层级分布非常有意思。底层像是把不同语言的表面符号映射到内部空间；中间层更偏共享语义处理；顶层则把内部状态重新投影回某种语言的词表。它和我们前面说的 tokenizer 与表示空间可以拼起来：输入语言先被编码，中间共享语义，最后输出语言重新被选择。

后续工作把这个思路推进到了更直接的控制。

[Language Arithmetics](https://arxiv.org/abs/2507.22608) 在 Llama-3.1-8B、Mistral-Nemo、Aya-Expanse 等模型上用 LAPE 找语言相关 neurons，并做 additive intervention。它不是简单把激活替换成固定均值，而是在保留原上下文动态的基础上加上目标语言激活模式，或者同时压低源语言模式。论文设计了 language forcing 任务：输入是某种语言的问题，但希望模型不用显式 prompt 就用另一种目标语言回答。结果显示，激活目标语言 neurons 同时去激活源语言 neurons，更容易把输出推到目标语言。

[Cross-Lingual Activation Steering for Multilingual Language Models](https://arxiv.org/abs/2601.16390) 提出 CLAS，在推理时调节 shared、partial-shared、language-specific neurons。它的目标不是单纯强迫输出某种语言，而是提升非主导语言的跨语言迁移能力。它的一个重要观点是，有效迁移不一定来自把所有语言表示拉近，而可能来自功能性分化。换句话说，好的多语言模型不是让所有语言长得一样，而是让共享语义和语言特异路径各司其职。

[Neural FOXP2](https://arxiv.org/abs/2602.00945) 更像一个早期提案式工作，使用 VAE/SAE 字典基来发现语言选择特征，并把目标语言 defaultness 定义为早期解码步骤中目标语言 token mass 与英文 token mass 的差值。这个方向还需要更多独立复现，但它提出的观测量很适合本文问题：如果模型在弱提示下默认英文，那么我们可以直接看最初几个 step 的语言 token mass 是否偏向英文。

这些研究共同说明一件事：

**语言不是只存在于输入文本或最终输出里。它也存在于模型的激活分布、神经元子集、层间路由和 logit mass 中。**

因此，混合语言输出时，我们可以提出一个可测假设：

1. 正常中文生成时，深层或输出前层的中文相关 neurons、中文 token logit mass、中文子空间投影保持高位。
2. 当模型进入复杂推理或跨语言 RAG 时，英文证据、英文术语或高频英文模式抬高英文相关方向。
3. 在某个 switch point 前后，中文路径的优势不再足以压过英文路径。
4. 输出层 softmax 后，英文 token 的局部概率优势持续扩大，模型开始生成英文。
5. 一旦生成出英文 token，新的英文上下文又反过来强化后续英文 token，漂移进入自我稳定状态。

这不是已经被某一篇论文完整证明的机制链条，但每一环都有现有研究可以支持或检验。

## RAG 里的漂移：最接近真实应用的证据

如果要研究“现代模型为什么还会跳语言”，多语言 RAG 是特别好的场景，因为它天然制造了语言冲突。

[Language Drift in Multilingual Retrieval-Augmented Generation](https://arxiv.org/abs/2511.09984) 的实验很直接：用户 query、prompt、ICL examples 都在目标语言中，只有 retrieved context 换成另一种语言。这样就能隔离检索证据语言对输出语言的影响。论文构造 HotpotQA、MuSiQue、DuReader 的多语言版本，覆盖英语、中文、阿拉伯语、俄语，并在 LLaMA3-8B-Instruct、Qwen2.5-7B-Instruct 上测试。

它的几个发现很值得放进我们的研究问题里。

第一，跨语言 retrieved context 会同时降低任务表现和输出语言一致性。也就是说，模型不是只“语言错了但内容对了”，而是在复杂 reasoning 中语言漂移和任务质量会相互影响。

第二，ICL 和 CoT 会让问题更复杂。示例和推理步骤通常提升语义表现，但也拉长生成路径，增加中途漂移机会。

第三，漂移不总是跟随 retrieved context 的语言。即使证据不是英文，模型失败时也经常回到英文。这支持“英文默认吸引子”的说法。

第四，论文提出 Soft Constrained Decoding，在解码时对非目标语言 token 施加软惩罚，而不是硬过滤。它把词表分成 target、neutral、distractor 三类，对 logits 做轻量调整。实验显示，SCD 能提升 language consistency，同时不必牺牲推理流畅性。

这篇论文对本文问题的价值在于：它把“跳语言”从聊天体验变成了一个 decoder-level collapse 问题。模型可能已经理解了任务和证据，但生成阶段的 token prior、语言吸引子和长推理路径把输出带到非目标语言。

这也解释了为什么现代主流 chat 模型平时稳定：普通对话没有强跨语言 evidence，没有很长 CoT 外显输出，采样温度较低，系统 prompt 明确，SFT/RLHF 又惩罚了不按用户语言回答的行为。多语言 RAG 把这些护栏削弱或冲突化，漂移就又出现了。

## SFT、RLHF 和系统提示词：强工程解释，但不能替代机制解释

现在的产品级大模型一般都会遵守“用户用什么语言问，就用什么语言答”。这显然和训练后对齐有关。

SFT 阶段会包含大量指令数据。高质量多语言 instruction data 里通常隐含一个规则：中文问题给中文答案，英文问题给英文答案。模型不仅学习任务，也学习对话礼仪。RLHF 或 RLAIF 阶段，语言错误、无故切换、跟用户语言不一致的回答很可能被偏好模型或人工标注打低分。系统提示词也常常显式要求使用用户语言。

这些解释在工程上非常合理，而且很可能是现代 chat 模型稳定的主要表层原因。

但如果我们的研究目标是“分词器切分与训练后嵌入空间形状如何影响跳语言”，就不能停在这里。原因有三个。

第一，对齐训练解释的是可见行为，不直接解释模型内部为什么能保持语言路径。

第二，对齐训练可能只是把原有语言控制电路调得更强，而不是凭空创造它。没有底层 tokenizer、表示空间和输出层可控性，对齐信号也很难高效地稳定所有语言。

第三，专有模型训练细节不可见。我们可以合理推断 SFT/RLHF 有贡献，但不能把它写成已由公开论文直接证明的因果机制。

所以我会把它放在文章的工程层：

**对齐训练和系统提示词是现代产品体验中最可见的护栏；tokenizer、表示空间和激活动态是这些护栏能发挥作用的底层条件。**

## 一个统一机制图

可以把语言保持理解成一个从输入到输出的控制链。

```text
用户语言 / 系统指令 / 上下文语言
        |
        v
Tokenizer
  - fertility / STRR / byte fallback
  - 目标语言 token 成本
        |
        v
底层表示
  - script / orthography / lexical cues
  - language-associated units
        |
        v
中间层表示
  - shared semantic space
  - interlingual local overlap
  - English-centric or partially language-agnostic concept space
        |
        v
高层与输出前层
  - language-specific neurons
  - target-language residual direction
  - language vector / steering direction
        |
        v
Logits / decoding
  - target token mass vs English token mass
  - SCD / system prompt / sampling temperature
        |
        v
可见输出语言
```

在这个图里，跳语言可以发生在不同位置：

- tokenizer 让目标语言成本太高，输入锚点变弱；
- 中间层共享语义更靠近英文，推理阶段形成英文吸引子；
- RAG evidence 或 CoT 让英文方向逐渐增强；
- 输出层语言特异性激活不足，目标语言 logit mass 被英文 token mass 反超；
- 生成出第一个非目标语言 token 后，上下文自回归地强化错误语言。

现代模型稳定，是因为这些环节都被改善了：tokenizer 更公平，预训练多语言覆盖更好，指令数据更强，对齐训练惩罚漂移，系统 prompt 提供显式约束，解码参数更保守，应用层还能做语言检测和重试。

## 可复现实验路线：如何观察混合语言输出时的激活偏向

如果要把这个问题发展成一条完整研究路线，我会从一个开源模型实验框架开始，而不是直接研究闭源模型。

### 1. 模型选择

建议至少选三类模型：

- 英语中心但有一定多语言能力的模型：Llama-3.1-8B、Mistral-Nemo。
- 多语言导向模型：Qwen2.5-7B/14B、Aya-Expanse-8B。
- 中文能力强的模型：Qwen、DeepSeek、Yi 或 GLM 系列的开源版本。

这样可以比较三个变量：词表设计、预训练语言比例、多语言对齐强度。

### 2. Prompt 设计

需要构造几类任务，而不是只看普通问答。

| 场景 | 目的 |
| --- | --- |
| 单语普通问答 | 建立语言保持 baseline。 |
| 中文长文生成 | 观察长序列中是否存在 switch point。 |
| 中文复杂推理 | 观察 reasoning depth 是否提高 drift 风险。 |
| 中文问题 + 英文证据 RAG | 模拟最常见的跨语言 evidence 干扰。 |
| 中文问题 + 阿拉伯语/俄语证据 | 区分“跟随证据语言”和“回退英文”。 |
| 混合 prompt | 观察代码、术语、引用标题对语言路径的局部影响。 |
| 弱提示/无系统提示 | 放大模型默认语言 prior。 |

每个 prompt 都应该有多个温度设置，比如 greedy、temperature 0.3、0.7、1.0。语言漂移往往在高温和长输出下更明显。

### 3. 可见输出指标

最基础的是文本层指标：

- token-level language ID；
- 第一个非目标语言 token 的位置；
- drift span 的长度；
- drift 后是否恢复目标语言；
- 输出语言一致率；
- 任务正确率；
- 术语、引用、代码等合理 code-switching 的人工标注。

这里要小心：不是所有英文 token 都是错误。`Transformer`、`RLHF`、论文标题、代码变量名都可能应当保留英文。所以需要区分合理 code-switching 和 unintended drift。

### 4. Tokenizer 指标

对每条 prompt 和每个输出，记录：

- target language fertility；
- character-to-token ratio；
- STRR；
- 英文等价表达的 token 数；
- 中英 token 数比例；
- 关键概念在不同语言中的 tokenization pattern。

一个可检验假设是：在同一模型中，更高的 target/English token ratio 会提高 drift 风险，尤其在长输出和复杂推理中。

### 5. Logit 指标

在每个生成 step 上计算：

$$
M_{\text{zh}}(t)=\sum_{u \in V_{\text{zh}}} p(u \mid x_{<t})
$$

$$
M_{\text{en}}(t)=\sum_{u \in V_{\text{en}}} p(u \mid x_{<t})
$$

然后观察：

$$
\Delta M(t)=M_{\text{zh}}(t)-M_{\text{en}}(t)
$$

如果语言漂移真的是 decoder-level collapse，那么 switch point 前后应该能看到目标语言 mass 下降、英文 mass 上升，或者英文 top-k token 从候选边缘进入主候选区。

实际实现中，中文 token set 可以用 Unicode 范围、词表 decode 后的字符比例、fastText/langid 组合得到。英文 token set 也类似。neutral tokens 需要单独处理，包括数字、标点、空格、Markdown、代码符号。

### 6. Hidden-state 和子空间指标

可以在每层取 residual stream 或 MLP activation，构造语言方向：

$$
v_{\text{zh-en}} = \mathbb{E}[h_{\text{zh}}] - \mathbb{E}[h_{\text{en}}]
$$

然后对生成过程中的每个 step 计算投影：

$$
s_l(t)=\langle h_l(t), v_{\text{zh-en},l} \rangle
$$

如果模型保持中文，深层或输出前层的投影应该稳定偏向中文。如果 switch point 前出现投影下降，说明语言方向在内部已经提前漂移。

还可以做 PCA/SVD 子空间分析，对齐 [The Geometry of Multilingual LM Representations](https://arxiv.org/abs/2205.10964) 的思路：比较中文、英文、混合输入在各层的均值、协方差和局部邻域重叠。

### 7. Language-specific neurons 与干预

用 LAPE 或更简单的 activation frequency 方法找中文、英文相关 neurons。然后做三类干预：

1. 激活中文 neurons；
2. 去激活英文 neurons；
3. 同时激活中文、压低英文。

如果干预能推迟或消除 drift，就说明语言选择不是纯输出文本后处理，而是深层激活里有可操纵变量。

更细的实验是 patching：

- 从稳定中文生成样本中取某层 activation；
- patch 到会 drift 的样本同一位置；
- 观察输出语言是否恢复中文；
- 反过来从 drift 样本 patch 到稳定样本，观察是否诱发英文。

这能帮助定位 switch point 之前到底是哪几层开始失去目标语言约束。

### 8. Decoding intervention

最后做解码层对照：

- prompt-only：只加“请用中文回答”；
- hard vocabulary restriction：禁止英文 token；
- soft constrained decoding：轻惩罚非中文 token；
- activation steering：只改内部激活，不改 logits；
- combined：activation steering + SCD。

预期结果可能是：

- hard restriction 语言一致性高，但容易伤害术语、代码和流畅性；
- SCD 在 RAG 场景中性价比高；
- activation steering 更适合研究机制，也可能更自然地保留语义；
- 两者结合可以区分“内部语言状态”与“最终解码偏置”的贡献。

## 关键假设候选

我会把后续研究整理成几个可以检验的假设。

**H1: Token cost hypothesis**

语言漂移概率与目标语言相对英文的 token 成本正相关。高 fertility、低 STRR 的语言更容易在复杂生成中漂移，尤其当任务需要生成很多抽象概念或专业术语。

**H2: Late-layer language gate hypothesis**

目标语言选择主要在中后层或输出前层完成。中间层可以共享语义或接近英文，但只要最后几层的目标语言 gate 足够强，模型仍能稳定输出目标语言。

**H3: English attractor hypothesis**

当目标语言 gate 变弱时，模型不一定跟随上下文中最近的语言，而是回退到训练和词表中最强的默认语言，通常是英文。多语言 RAG drift 的证据支持这个方向。

**H4: Script anchor hypothesis**

脚本不是表面细节，而是重要语言锚点。中文、日文、阿拉伯文、印地语等脚本可以形成强 surface-form routing；romanization 可能破坏原本的语言激活路径。

**H5: Alignment as boundary training hypothesis**

SFT/RLHF 并不只是教模型“礼貌地按用户语言回答”，它可能在输出层附近强化了语言边界，让模型在共享语义推理之后更稳定地投影回目标语言。这个假设需要用开源 instruct/base 模型对照来验证。

## 文献地图

| 主题 | 论文 | 对本文问题的贡献 |
| --- | --- | --- |
| Tokenizer 成本 | [The Token Tax](https://arxiv.org/abs/2509.05486) | fertility 系统性影响多语言任务准确率和成本。 |
| Tokenizer 评估 | [Beyond Fertility / STRR](https://arxiv.org/abs/2510.09947) | fertility 之外还要看词表是否保留单 token 词。 |
| 脚本与表示 | [Multilingual LMs Encode Script Over Linguistic Structure](https://arxiv.org/abs/2604.05090) | 语言相关单元很大程度上被 orthography/script 组织。 |
| 表示空间几何 | [The Geometry of Multilingual LM Representations](https://arxiv.org/abs/2205.10964) | language-sensitive axes 与 language-neutral axes 并存。 |
| 语言无关子空间 | [Discovering Low-rank Subspaces for Language-agnostic Multilingual Representations](https://arxiv.org/abs/2401.05792) | 提供语言无关表示子空间的分析路线。 |
| 隐性英语 | [Do Llamas Work in English?](https://arxiv.org/abs/2402.10588) | logit lens 显示中间层语义 token 常更接近英文。 |
| 英语中心生成 | [Do Multilingual LLMs Think In English?](https://arxiv.org/abs/2502.15603) | 开放式生成中 lexical decisions 更靠近英文空间。 |
| Interlingua | [High-Dimensional Interlingual Representations](https://arxiv.org/abs/2503.11280) | 共享语义区域与碎片化组件并存。 |
| 语言神经元 | [Language-Specific Neurons](https://arxiv.org/abs/2402.16438) | LAPE 识别语言特异性 neurons，并能操控输出语言。 |
| 神经元算术 | [Language Arithmetics](https://arxiv.org/abs/2507.22608) | additive intervention 可做 language forcing。 |
| RAG 漂移 | [Language Drift in Multilingual RAG](https://arxiv.org/abs/2511.09984) | 证明跨语言 evidence 会诱发输出语言漂移，SCD 可缓解。 |
| 激活引导 | [Cross-Lingual Activation Steering](https://arxiv.org/abs/2601.16390) | 通过推理时 neuron modulation 提升非主导语言表现。 |
| 默认语言编辑 | [Neural FOXP2](https://arxiv.org/abs/2602.00945) | 提出以 target-vs-English token mass 衡量 defaultness 的 intervention 思路。 |

## 争议和局限

第一，许多机制论文主要研究开源中小模型，不能直接外推到 GPT-4、Claude、Gemini 这类专有模型。产品模型的 tokenizer、训练语料、SFT/RLHF、system prompt、decoding stack 都不可见。

第二，logit lens 不是完美读心术。中间层用 unembedding 解码出英文 token，不等于模型真的在离散英文句子中推理。它只是说明某些中间表示更容易被输出头解释为英文。

第三，language-specific neurons 也不一定是纯语言 neurons。它们可能混合了脚本、频率、主题、语域、数据来源等因素。[Script over Linguistic Structure](https://arxiv.org/abs/2604.05090) 正是在提醒我们，很多所谓 language units 其实更接近 surface-form units。

第四，中文本身不是最弱势的测试对象。现代模型对中文投入很大，中文 tokenizer 和训练数据都相对充足。若要研究语言漂移的底层机制，应该同时看印地语、孟加拉语、阿姆哈拉语、斯瓦希里语、阿拉伯语、俄语等语言。

第五，语言漂移不总是坏事。真实用户可能希望模型保留论文标题、代码、术语和引用原文。因此评估时不能把所有非目标语言 token 都算错误。

## 我现在的答案

回到最初的问题：为什么现在的大模型在使用时不会频繁跳到其他语言？

我的答案是：

**现代大模型不是突然“更懂规矩”这么简单，而是从输入、表示、激活到输出的整条链路都变得更能维持语言边界。**

Tokenizer 改进降低了非英语文本的 token 成本，让中文等语言不再从输入层就被严重切碎。多语言预训练和更大词表让目标语言有足够知识与表达资源，不必总是借英文路径。表示空间中共享语义与语言特异方向并存，使模型能在中间层做跨语言语义处理，又在输出层回到目标语言。语言特异性 neurons 和 activation steering 研究说明，输出语言选择可以被局部激活模式影响，而不是只由 prompt 文本决定。SFT、RLHF、system prompt 和保守 decoding 则在产品层把这些底层能力进一步约束成稳定的对话行为。

但是，这个问题远没有结束。现在的模型不是不会跳语言，而是在常规使用场景里不容易暴露。只要把场景换成跨语言 RAG、长 CoT、弱提示、多脚本混合、低资源语言或高温采样，语言漂移仍然会出现。也正因为如此，它是一个很好的机制研究入口：它连接了 tokenizer fairness、representation geometry、activation dynamics、decoding control 和 alignment training。

如果要继续深入，我最想做的实验不是再统计“模型回答里有多少英文”，而是抓住语言切换前后的那几个 token：

**在模型即将从中文滑向英文的瞬间，中文 token mass、英文 token mass、深层语言 neurons、residual language direction 和中间层 concept space 到底发生了什么？**

这个问题如果能被系统回答，我们对多语言大模型的理解会比“它会按用户语言回答”深很多。

## 附录：arXiv TeX 源码深读笔记

这部分记录我认为最值得在正式研究中继续追的细节。证据强度以公开论文和 arXiv source archive 为准。

### Tokenizer 与 token tax

[The Token Tax](https://arxiv.org/abs/2509.05486) 的强点是把 tokenizer inefficiency 和任务准确率、经济成本放在同一框架里。它的实验对象是 AfriMMLU，覆盖多个非洲语言和 10 个大模型。它不是直接研究 language drift，但能支持一个前提：如果目标语言 fertility 高，模型在该语言上的处理会更贵、更难、更不稳定。

[Beyond Fertility / STRR](https://arxiv.org/abs/2510.09947) 的关键贡献是指出 fertility 的盲区。一个 tokenizer 平均 tokens/word 不高，不代表它公平地给各语言高频词分配了完整 token。STRR 更适合研究“目标语言是否在词表里有稳定落点”。对语言漂移来说，这个指标可能比 fertility 更接近输出阶段的稳定性。

### 表示空间与 latent English

[The Geometry of Multilingual LM Representations](https://arxiv.org/abs/2205.10964) 的重点不是 LLM chat，而是 XLM-R 表示空间。但它提供了很重要的几何语言：mean-shifted subspaces、language-sensitive axes、language-neutral axes。后续研究可以直接借用这种方法分析 decoder-only LLM 的 residual stream。

[Do Llamas Work in English?](https://arxiv.org/abs/2402.10588) 的实验很干净，因为它把 continuation 设计成单 token，从而能用 logit lens 判断中间层偏向哪种语言。局限是单 token 任务不等于开放式生成。

[Do Multilingual LLMs Think In English?](https://arxiv.org/abs/2502.15603) 弥补了开放式生成，但它的结论也更容易受任务、模型和解码方式影响。它很适合作为“英文中心语义决策”的证据，但不应被写成所有 LLM 的普遍定律。

[High-Dimensional Interlingual Representations](https://arxiv.org/abs/2503.11280) 的 ILO 指标值得借鉴，因为 language drift 很可能和局部邻域结构有关。若中文表示在某层与英文高频概念邻域高度重叠，输出时可能更容易被英文 token 竞争。

### Language neurons 与 activation steering

[Language-Specific Neurons](https://arxiv.org/abs/2402.16438) 的 LAPE 是研究语言激活偏向的基础工具。它的层级发现很适合本文问题：底层和顶层更语言特异，中间层更共享。

[Language Arithmetics](https://arxiv.org/abs/2507.22608) 的 additive intervention 比 replacement 更适合做语言漂移研究，因为 replacement 容易破坏上下文动态，而 additive 更接近“给目标语言路径加一个偏置”。

[Cross-Lingual Activation Steering](https://arxiv.org/abs/2601.16390) 提醒我们，activation steering 不只是把所有语言拉近。有效跨语言迁移可能需要保留功能分化。这对“防止跳语言”也很重要：过强的语言无关化可能会伤害输出语言边界。

[Neural FOXP2](https://arxiv.org/abs/2602.00945) 还需要更谨慎对待，但它提出的 defaultness 指标很有用。用目标语言 token mass 减英文 token mass 来跟踪早期生成偏置，是分析 switch point 的一个低成本入口。

### RAG drift 与解码控制

[Language Drift in Multilingual RAG](https://arxiv.org/abs/2511.09984) 是最接近应用场景的论文。它把 drift 归因到 decoder-stage bias，而不是简单的 comprehension failure，并提出 SCD 这种训练-free 的 soft control。对工程实践来说，这可能是最快能落地的方向：不改模型参数，只在解码时轻推目标语言 token。

但从机制研究角度看，SCD 还只是输出层控制。更有意思的问题是：SCD 成功时，内部 residual state 是否也更偏向目标语言，还是只是最终 logits 被修正？这可以通过 activation logging 来验证。

## 推荐阅读路径

如果只想理解 tokenizer 层，先读 [The Token Tax](https://arxiv.org/abs/2509.05486)，再读 [Beyond Fertility / STRR](https://arxiv.org/abs/2510.09947)。

如果想理解表示空间，先读 [The Geometry of Multilingual LM Representations](https://arxiv.org/abs/2205.10964)，再读 [Do Llamas Work in English?](https://arxiv.org/abs/2402.10588) 和 [High-Dimensional Interlingual Representations](https://arxiv.org/abs/2503.11280)。

如果想做机制可解释性实验，先读 [Language-Specific Neurons](https://arxiv.org/abs/2402.16438)，再读 [Language Arithmetics](https://arxiv.org/abs/2507.22608) 和 [Cross-Lingual Activation Steering](https://arxiv.org/abs/2601.16390)。

如果想做应用层防漂移，先读 [Language Drift in Multilingual RAG](https://arxiv.org/abs/2511.09984)，然后实现一个最小版 SCD，对比 prompt-only、hard restriction 和 activation steering。

## 参考文献

- Lundin et al. [The Token Tax: Systematic Bias in Multilingual Tokenization](https://arxiv.org/abs/2509.05486), arXiv, 2025.
- Nayeem et al. [Beyond Fertility: Analyzing STRR as a Metric for Multilingual Tokenization Evaluation](https://arxiv.org/abs/2510.09947), arXiv, 2025.
- Verma et al. [Multilingual Language Models Encode Script Over Linguistic Structure](https://arxiv.org/abs/2604.05090), arXiv, 2026.
- Chang et al. [The Geometry of Multilingual Language Model Representations](https://arxiv.org/abs/2205.10964), EMNLP, 2022.
- Xie et al. [Discovering Low-rank Subspaces for Language-agnostic Multilingual Representations](https://arxiv.org/abs/2401.05792), arXiv, 2024.
- Wendler et al. [Do Llamas Work in English? On the Latent Language of Multilingual Transformers](https://arxiv.org/abs/2402.10588), ACL, 2024.
- Schut et al. [Do Multilingual LLMs Think In English?](https://arxiv.org/abs/2502.15603), arXiv, 2025.
- Wilie et al. [High-Dimensional Interlingual Representations of Large Language Models](https://arxiv.org/abs/2503.11280), arXiv, 2025.
- Tang et al. [Language-Specific Neurons: The Key to Multilingual Capabilities in Large Language Models](https://arxiv.org/abs/2402.16438), arXiv, 2024.
- Gurgurov et al. [Language Arithmetics: Towards Systematic Language Neuron Identification and Manipulation](https://arxiv.org/abs/2507.22608), arXiv, 2025.
- Li et al. [Language Drift in Multilingual Retrieval-Augmented Generation: Characterization and Decoding-Time Mitigation](https://arxiv.org/abs/2511.09984), arXiv, 2025.
- Pokharel et al. [Cross-Lingual Activation Steering for Multilingual Language Models](https://arxiv.org/abs/2601.16390), arXiv, 2026.
- Saha et al. [Neural FOXP2: Language Specific Neuron Steering for Targeted Language Improvement in LLMs](https://arxiv.org/abs/2602.00945), arXiv, 2026.
