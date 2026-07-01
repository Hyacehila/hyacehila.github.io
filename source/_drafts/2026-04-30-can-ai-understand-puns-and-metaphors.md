---
title: "AI 能听懂谐音梗吗：从 Tokenizer 到后训练"
date: 2026-04-30 20:00:00 +0800
categories: ["Foundation Models"]
tags: [LLM, Tokenizer, Humor, Metaphor, Post-training]
author: Hyacehila
excerpt: "大模型不是完全不懂谐音梗和比喻，而是经常把这种理解建立在文本记忆、表层相似和上下文猜测上。真正难的地方，是让模型同时看到字形、字义、字音和文化语境。"
---

今天的大模型当然能解释很多谐音梗。

你问它“泰裤辣”是什么意思，它大概率知道这是“太酷啦”的谐音化网络表达；你问它英文里的 *two tired* 为什么好笑，它也能说出 *two tired / too tired / two tires* 之间的双关关系。问题不在于模型完全没有这种能力，而在于：它的理解很不稳定。

对于常见、已经在训练语料里反复出现的梗，模型像一个背过段子的学生，能把答案说得很顺。对于新造的、局部上下文很强的、夹杂方言和视觉元素的谐音梗，它就容易开始一本正经地胡说八道。这个现象在近两年的研究里已经被反复观察到：2024 年有论文专门用识别、解释和生成三类任务评估 LLM 对双关语的理解，发现模型会出现“偷懒式生成”与解释不稳的问题；2025 年的 Phunny benchmark 进一步用人工新造的双关问题控制数据污染，结果显示多数 LLM 在双关理解、消解和生成上仍显著落后于人类。

所以更准确的说法是：**AI 可以在一定程度上理解谐音梗，但它的理解常常是统计性的、文本性的、回忆式的，而不是稳定地把“声音”和“意义”接起来。**

## 为什么模型会卡在谐音上

谐音梗的核心是双通道：表面上读到一个词，耳朵里却听到另一个词。中文里是“堡/饱”“裤辣/酷啦”，英文里是 *night/knight*、*two/too/to*、*tired/tires*。人类可以很自然地在字形、语义、语音和上下文之间跳转，但文本大模型的输入起点不是声音，而是 token。

以常见的 BPE tokenizer 为例，OpenAI 的 `tiktoken` 文档直接说明，语言模型看到的不是自然文本，而是一串 token 数字；BPE 的目标是把文本可逆地压缩成常见子词片段。这个过程对建模非常有效，但它不会天然给每个 token 附上一条“这个词怎么读”的并行音系表示。

这就导致一个结构性错位：模型很擅长学习“这个词在文本里通常和什么词一起出现”，却不一定知道“这个词和另一个词听起来很像”。在英语里，*knight* 和 *night* 可能被分到完全不同的 token 序列；在中文里，“理发”和“李发”在语义空间中也没有理由因为发音接近就自动靠近。模型可以通过语料里的拼写错误、歌词、笑话解释、拼音标注间接学到一些发音关系，但这不是输入层保证的能力。

双关语之所以难，还因为它天然违反传统语义模型里的“一个上下文里选一个词义”的假设。SemEval 2017 的英文双关任务就把 pun 定义为利用多义、同形或音系相似性让一个表达同时指向多个意义的文字游戏，并指出它会挑战传统词汇语义中的 one-sense-per-context 假设。换句话说，谐音梗不是让模型选 A 或 B，而是要求它同时保留 A 和 B，再解释两者之间为什么能制造幽默。

## 英语里有没有类似现象

有，而且非常典型。英语里的总称通常是 *puns* 或 *wordplay*，其中同音/近音双关可以叫 *homophonic puns*。

经典例子是：

> Why did the bicycle fall over? Because it was two tired.

这里至少有两层关系：*too tired* 回答“为什么倒下”，*two tires* 又对应自行车有两个轮胎。人类听到以后会把句法上的不自然当作线索，主动寻找另一个读音相近但语义更通顺的解释。模型如果只沿着字面 token 走，很容易解释得很机械。

NLP 里早就有这条研究线。SemEval 2017 Task 7 做的是英文双关的检测、定位和解释；2024 年的 *Can Large Language Models Understand Puns?* 开始系统考察 LLM 的双关识别、解释和生成；2025 年 ACL 的 Phunny benchmark 则强调新造双关与数据污染控制，用来测试模型到底是在泛化，还是在复述见过的段子。

中文方向最近也有类似证据。Chumor 2.0 从“弱智吧”收集中文幽默解释数据，覆盖文化、情境、谐音、字形、跨语言等类型，报告称直接提示和 CoT 提示下的多个 LLM 仍远低于人类。PunMemeCN 则把问题推进到中文梗图：它包含 1,959 张中文 meme 和多轮对话标注，实验显示视觉语言模型尤其容易在同音文字游戏上失败，哪怕给了 CoT。

这些结果共同说明一件事：英文、中文都不是“没有研究”，而是研究越细，越能看到模型在幽默和双关上的脆弱性。

## 比喻为什么又不一样

比喻不主要靠声音，而靠跨域映射。

“时间像河流”不是因为“时间”和“河流”读音像，而是因为它们共享某些抽象结构：流动、不可逆、无法抓住、持续推进。谐音梗要求模型发现音系相似，隐喻则要求模型把一个领域的结构投射到另一个领域。

这类能力大模型也有，但同样不稳定。MiQA 这类早期 benchmark 把隐喻理解和常识推理放在一起，要求模型在字面和隐喻读法之间选择。2024 年的 MUNCH 数据集进一步提供 1 万多个隐喻句转述和 1,500 个不恰当转述，用来区分模型是真的理解隐喻，还是只在利用词汇重叠。2025 年的一个多数据集评估则更尖锐：LLM 在隐喻任务上的表现会受到词汇重叠、句长等表层特征影响，所谓“涌现的隐喻理解”很可能混合了表层线索、上下文学习和已有语言知识。

所以谐音梗和比喻都属于“表面文本之外还有第二层结构”的语言现象。不同点在于，谐音梗缺的是字音通道，比喻缺的是跨域结构映射；前者需要 phonology，后者需要 analogy。

## 怎么让模型真正更懂谐音梗

第一条路是做专门数据。

不要只收集“笑话文本”，而要把每条样本拆成结构化标注：表层表达是什么，可能的谐音候选是什么，拼音或音标是什么，字面解释为什么不顺，隐藏解释为什么顺，最后幽默点在哪里。SFT 时可以让模型学习一种固定解释路径：先发现违和，再展开发音候选，再做语义重构，最后给出双关解释。

这条路已经有现实依据。PhonoThink 在 2025 年针对中文音系歧义提出多阶段训练：先用构造好的子任务数据和合成的逐步推理链做 SFT，再用强化学习稳定推理。它的方向和“让模型学会处理谐音梗”高度相关：不要指望模型从普通语料里自然学会所有发音歧义，而是把音系知识和推理过程显式做成训练对象。

第二条路是工具调用。

对于英文，可以接入 CMU Pronouncing Dictionary，把词转成 ARPAbet 音素，再反查近音词。对于中文，可以接入拼音转换、分词、近音词典、方言音系表。模型遇到一句字面意思不通顺的话时，先不要急着解释，而是调用工具查“这几个词读起来像什么”。

这个范式并不新。Toolformer 早就展示过语言模型可以学习何时调用外部 API、如何传参、如何把工具返回结果纳入后续预测。放到谐音梗上，工具可以很简单：`to_pinyin(text)`、`to_phoneme(word)`、`near_homophones(query)`、`semantic_fit(candidate, context)`。训练时给模型奖励的不是“调用了工具”，而是“调用后找到了更好的双关解释”。

第三条路是改输入表征。

如果从预训练或基础模型层面解决，就不能只给模型 token embedding，还要额外给它 phonetic embedding。中文里可以拼接字形、拼音和常规字符表示；ChineseBERT 就是一个早期代表，它把 glyph 和 pinyin 信息纳入中文预训练，并报告了多项中文任务上的收益。对谐音梗而言，这类思路的价值在于让模型从底层就知道“这些字虽然写法不同，但可能共享发音”。

不过这条路有代价。拼音不是总能一对一决定读音，多音字、变调、儿化、方言、语境读法都会带来噪声。英文也一样，拼写到发音不是完全规则，重音和连读会影响笑点。因此，phonetic embedding 更适合作为一条辅助通道，而不是替代语义建模。

第四条路是把比喻当作结构映射来训练。

比喻数据不应只标“这是比喻/不是比喻”，还要标本体、喻体和 ground，也就是两者共享的抽象属性。比如“时间是河流”的 ground 可以是“流动性”“不可逆性”“无法停驻”。训练目标可以是让模型生成转述、区分恰当和不恰当转述，或者用对比学习把共享抽象维度拉近，而不是简单把两个词的普通语义向量拉近。

## 一个可落地的 Agent 方案

如果今天就要做一个更会解释谐音梗的 Agent，我会把它拆成四层。

第一层是异常检测。模型先判断句子有没有局部不自然：语法怪、语义怪、回答和问题接不上、图文关系不对等。

第二层是音系展开。对可疑片段调用拼音、音素、近音词典，生成多个候选改写。例如“吃堡了撑的”可以展开到“吃饱了撑的”，“two tired”可以展开到 *too tired* 和 *two tires*。

第三层是语义重评分。把候选放回上下文，看哪一个同时解释字面、隐藏义和幽默效果。这里可以用模型自己评分，也可以训练一个小的 reward model。

第四层是解释生成。输出时不要只说“这是谐音”，而要说明两层含义、为什么会造成错位、为什么错位能成立。

这套方案不神秘，但它比“让模型多看笑话”更可靠。因为它把谐音梗拆成了可学习、可调用、可评估的中间步骤。

## 结论

AI 不是完全不懂谐音梗，也不是已经真正“听懂”了谐音梗。它处在中间状态：对高频梗、经典双关、解释型语料里出现过的笑话，它可以表现得很像懂；对新梗、文化梗、方言梗、多模态梗，它又会暴露出文本模型的根本局限。

要补上这个缺口，不能只靠更大的参数。更有希望的方向是：在数据上显式标注双层意义，在后训练中教模型做音系推理，在工具层给它拼音和音标查询能力，在表征层把发音作为输入通道，在评估上用新造样本和人类解释控制背答案。

谐音梗听起来像小问题，其实它很适合拿来观察大模型的边界。因为它逼着模型回答一个基础问题：你到底只是在看字，还是已经学会在字、音、义和语境之间来回切换？

## 参考资料

- [SemEval-2017 Task 7: Detection and Interpretation of English Puns](https://aclanthology.org/S17-2005/)
- ["A good pun is its own reword": Can Large Language Models Understand Puns?](https://arxiv.org/abs/2404.13599)
- ["What do you call a dog that is incontrovertibly true? Dogma": Testing LLM Generalization through Humor](https://aclanthology.org/2025.acl-long.1117/)
- [Chumor 2.0: Towards Better Benchmarking Chinese Humor Understanding from (Ruo Zhi Ba)](https://aclanthology.org/2025.findings-acl.1122/)
- [PunMemeCN: A Benchmark to Explore Vision-Language Models' Understanding of Chinese Pun Memes](https://aclanthology.org/2025.emnlp-main.944/)
- [PhonoThink: Improving Large Language Models' Reasoning on Chinese Phonological Ambiguities](https://aclanthology.org/2025.emnlp-main.961/)
- [Metaphor Understanding Challenge Dataset for LLMs](https://aclanthology.org/2024.acl-long.193/)
- [Metaphor and Large Language Models: When Surface Features Matter More than Deep Understanding](https://aclanthology.org/2025.findings-acl.898/)
- [MiQA: A Benchmark for Inference on Metaphorical Questions](https://arxiv.org/abs/2210.07993)
- [ChineseBERT: Chinese Pretraining Enhanced by Glyph and Pinyin Information](https://aclanthology.org/2021.acl-long.161/)
- [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761)
- [CMU Pronouncing Dictionary](https://github.com/cmusphinx/cmudict)
- [tiktoken: a fast BPE tokeniser for OpenAI models](https://github.com/openai/tiktoken)
