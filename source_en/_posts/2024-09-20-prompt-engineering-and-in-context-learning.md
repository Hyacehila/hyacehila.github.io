---
title: 'Prompt Engineering and In-Context Learning: Foundations, Technique Map, and Practical Workflows'
title_zh: 提示工程与上下文学习：从基础设计到技术图谱与场景实践
date: 2024-09-20 20:00:00 +0800
categories:
- Foundation Models
- Model Mechanics
tags:
- LLM
- Prompt Engineering
- In-Context Learning
- Reasoning
- Context Engineering
author: Hyacehila
mathjax: true
hidden: true
excerpt: A practical guide to prompt design, in-context learning, reasoning prompts, retrieval and verification, automatic
  prompt optimization, and representative workflows.
description: A practical guide to prompt design, in-context learning, reasoning prompts, retrieval and verification, automatic
  prompt optimization, and representative workflows.
excerpt_zh: 从任务、输入、上下文、示例和输出约束出发，整理上下文学习、推理提示、检索与验证、自动提示优化，并给出可直接改写的场景模板与论文索引。
permalink: /blog/2024/09/20/prompt-engineering-and-in-context-learning/
lang: en
translation_key: 2024-09-20-prompt-engineering-and-in-context-learning
translation_status: machine
translation_source_hash: 9d648d5646e91f70d11f3e8b09c43e1b186fd7ae9951a32a4a27f8d933605280
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<h2>Prompt is an interface, not a spell.</h2>
<p>I have previously divided the guidelines on indicative learning, on indicative engineering and on Google. It follows that three of them are actually discussing different aspects of the same matter: the basic articles describe how to write, the synthesis collects a large number of methodological names, and the Guide to Practice tells ordinary users how to communicate their mandates clearly. When separated, the basics, the technological spectrum and the scenes were broken.</p>
<p>This article begins with a simple judgment:<strong>Prompt is the input interface for the current reasoning process of the model.</strong> It can change the information, mission expression and output constraints that the model sees, but it does not modify parameters or add to the knowledge and capabilities that the model has not learned. Thinking about this border clearly makes a lot of "tips" no longer mysterious.</p>
<p>A usable tip usually contains five types of information:</p>
<ol>
<li><strong>Tasks</strong>: What action the model is going to make, what the success criteria are.</li>
<li><strong>Input</strong>: Data, issues or materials to be addressed in this session.</li>
<li><strong>Context</strong>: Background, reference documents and constraints required to complete the task.</li>
<li><strong>Example:</strong>: the input-output relationship that the model is expected to imitate.</li>
<li><strong>Output Contract</strong>: Format, length, fields, audience and unworkable.</li>
</ol>
<p>They do not need to appear in every instance. A request for “translation of this sentence into English” had been sufficient for tasks and inputs; a result of an analysis to be submitted to the process for consumption would require clearer field, type and failure processing. The length of the hint does not mean that the message is more adequate, and the key is whether each paragraph really affects the mandate.</p>
<h2>How do you write a basic tip?</h2>
<p>Google. <a href="https://services.google.com/fh/files/misc/gemini_for_workspace_prompt_guide_october_2024_digital_final.pdf">Gemini for Workspace Prompting Guide</a> The organization of the office scenes is based on the four components: Persona (role), Task, Context, and Format. The framework is practical, but Persona is not an obligatory one; in many missions, it is more useful to identify audiences, materials and criteria for judgement than to require models to “play top-level experts”.</p>
<p>Here is a skeleton that is more suitable for a technical mission:</p>
<pre><code class="language-text">任务：根据给定事故记录，整理一份故障复盘摘要。

输入：
&lt;incident&gt;
&#123;&#123;事故记录&#125;&#125;
&lt;/incident&gt;

上下文与边界：
- 读者是参与值班的后端工程师。
- 只使用记录中出现的事实；无法确认的内容标记为“待核实”。
- 区分直接原因、促成因素和后续风险。

输出：
1. 事件时间线
2. 影响范围
3. 原因分析
4. 已完成和待完成的行动项
</code></pre>
<p>There's no special excuse. The labels are used to separate materials, and the project symbol clearly sets out the criteria for judgement, and the output part gives the model how the results are used. If the record is long, please also indicate which fields are most important, which sources are allowed to be cited and how they should be returned if the material is insufficient.</p>
<h3>Write action, then a correction.</h3>
<p>The “analysis of the report” remains broad. The analysis can refer to generalization, risk identification, numerical reconciliation, comparative versions, or policy recommendations. A better job description will use specific actions:</p>
<ul>
<li>(a) Extracting the three key assumptions in the report and quoting the corresponding paragraphs;</li>
<li>Comparison of cost, dependence and failure patterns of the two programmes;</li>
<li>(a) Identify inconsistencies between conclusions and table data;</li>
<li>Rewriting the original text to a reader who is not aware of the background does not add new facts.</li>
</ul>
<p>The clearer the verb, the easier the subsequent assessment. Conversely, if even people fail to say what is good, continuing the role, tone and “think carefully” will not normally save the task.</p>
<h3>Separate the material from the instructions.</h3>
<p>Long documents, user input, code and web content should preferably be placed in a clear partition. This allows both the model to distinguish between “directions to be followed” and “data to be processed” and the program to replace variables.</p>
<p>The separator is not a secure boundary. External material may contain text that conflicts with the system ' s target, and genuine applications still require privileges, content filters and protection of prompt. The structure of the alert can only reduce ambiguity and cannot replace the boundaries of trust at the system level.</p>
<h3>Output format to be consistent with downstream</h3>
<p>A visible result may require a title, table or brief conclusion; the result given to the program should define the field, type, number and missing value. Only writing "return to JSON" in the prompt may still produce an illegitimate output. When there is a hard-on format correctness, require JSON Schema, function call or restricted decoding, as detailed<a href="/en/blog/2026/03/01/structured-output-and-constrained-decoding/">Large Model Structured Output and Limited Decoding Technologies</a>。</p>
<h2>Context learning: an example is also a temporary training signal</h2>
<p>Context learning (In-Context Learning, ICL) allows models to recognize tasks without updating model parameters by commands and examples in the current context. Only the job description is often called zero-shot; one-shot or new-shot is added to one-shot or one or more examples of input outputs.</p>
<p>It looks like “field teaching”, but the model does not really complete the gradient update. The example is only part of the current sequence and will not be automatically retained after leaving this context. This feature makes ICL well suited to fast-adaptation formats and labels, and it is also limited by context length, illustrative order and the model ' s own capabilities.</p>
<h3>Example selection is more important than the number of examples</h3>
<p>The Few-shot example must satisfy at least three things at once:</p>
<ul>
<li>Use the same field and output format as the actual input;</li>
<li>Covering easily confused borders, not just the simplest formal cases;</li>
<li>The answer is correct in itself, and the way in which it is interpreted is consistent with the final task.</li>
</ul>
<p>If emotional classification is given to only three obvious positive samples, the model does not learn how to deal with irony, mixed evaluation or neutral statements. The more examples, the more they are, the more space they actually enter, and the more chance they can be for models to imitate random details. A more stable approach would be to identify the types of failure from the authentication collection, which would be supplemented by examples of those types.</p>
<p>The order of examples will also affect the outcome. The distribution of labels, recent examples and presentation may all be biased. Where stability is needed, the order of the evaluation can be changed rather than tested for a single ranking that appears to be good.</p>
<h3>RAG does not equal Few-shot</h3>
<p>The Few-shot example tells the model "how this task is done " ; the document that RAG retrieves usually tells the model " what the answer should be based on." Both place information in context, but assume different roles. The RAG also contains engineering issues such as cut-offs, indexing, retrieval, reordering, citation and permissions, which cannot be reduced to “a few additional sections of information”.</p>
<p>The context window is not a warehouse that can be filled with randomly. The more material the information is, the more noise the information is. How to select, compress, isolate and update the context has evolved from a single prompt writing to a separate context engineering issue, see<a href="/en/blog/2026/06/11/agent-context-engineering/">"Context is All You Need: Context Project for Smart Bodies"</a>。</p>
<h2>Technological mapping: first, by problem</h2>
<p>There's a lot of paper on the subject. <code>Chain-of-X</code>I'm sorry. Some methods propose reusable structures, some are only built on specific data sets and models, and others are closer to the reasoning system or the Agent framework. Instead of abbreviated memory, ask what it is trying to change.</p>
<pre><code class="language-mermaid">graph TD
    A[&quot;Prompt 技术&quot;] --&gt; B[&quot;任务表达与示例学习&quot;]
    A --&gt; C[&quot;推理、搜索与分解&quot;]
    A --&gt; D[&quot;检索、工具与验证&quot;]
    A --&gt; E[&quot;自动生成与迭代优化&quot;]
    B --&gt; B1[&quot;Zero/Few-shot · Persona · Step-Back&quot;]
    C --&gt; C1[&quot;CoT · Self-Consistency · ToT/GoT · PoT&quot;]
    D --&gt; D1[&quot;RAG · ReAct · CoVe · Chain-of-Note&quot;]
    E --&gt; E1[&quot;Self-Refine · APE · OPRO · Active-Prompting&quot;]
</code></pre>
<h3>Mission expression and example learning</h3>
<p>Such approaches primarily improve how mandates are expressed. Zero-shot and few-shot decide whether to give examples; roll, scene, format and separator reduce semantic ambiguity; and Refrase and Repond, Step-Back, etc., rewriting questions or abstracting high-level principles before processing specific requests.</p>
<p>They are best suited to situations where the mission ' s intent is not clear, where the presentation is significantly altered or where the output is unstable. If problems arise from lack of knowledge, the inaccessibility of tools or inadequate modelling capacity, continuing to improve the same text usually only leads to more fluid errors.</p>
<h3>Decompose, search and decompose.</h3>
<p><a href="https://arxiv.org/abs/2201.11903">Chain-of-Thought</a> Insert intermediate decomposition steps in the examples to allow models to imitate the process from question to answer;<a href="https://arxiv.org/abs/2205.11916">Zero-shot-CoT</a> Proof of simple step-by-step reasoning directives may also improve some of the tasks. They have implications for mathematics, symbols and multi-step reasoning, but the benefits depend on models, tasks and assessment methods. Simple factual extraction or fixed classification does not require Mr. S.S. to form a large line of reasoning.</p>
<p>In the product, I am more concerned with verifiable intermediates than with the need for models to show all the “thinking processes”. The model can be used to list the evidence used, give the computation, generate the operational codes or return the checklist. This would be more useful than a reasonable but unreconciled interpretation, both to aid in debugging.</p>
<p><a href="https://arxiv.org/abs/2203.11171">Self-Consistency</a> Samp multiple reasoning paths to the same question, and then aggregate the final answer. It is a matter of extra reasoning costs for stability and is suitable for a task where the answer can be consolidated and voted upon. If the output is an open formula or a long text, it is often difficult to define what is called “most answers”.</p>
<p><a href="https://arxiv.org/abs/2305.10601">Tree of Thoughts</a> and <a href="https://arxiv.org/abs/2308.09687">Graph of Thoughts</a> Extending the single chain of reasoning to a search structure: generate candidates, evaluate status, retain or retreat and continue. It's not just a prompt, it's a design of the reasoning controller. There is no status representation, evaluation function and search budget, and simply requiring the model to “use the think tree” usually produces only one tree description.</p>
<p><a href="https://arxiv.org/abs/2211.12588">Program of Thoughts</a> Give the calculation to the program interpreter,<a href="https://arxiv.org/abs/2502.18600">Chain of Draft</a> . Compress the middle step to reduce the cost of the token. They remind each other that verifiable calculations need not be based solely on linguistic models, and that the reasoning text is not as reliable as it is long.</p>
<h3>Search, Tool Call and Results Validation</h3>
<p><a href="https://arxiv.org/abs/2005.11401">RAG</a> (a) Retrieving external material before generating responses based on the material;<a href="https://arxiv.org/abs/2210.03629">ReAct</a> The model is staggered between reasoning and action and can be searched, called upon, read and processed. These methods reduce the pressure for the model to answer questions by using only parameter memory and introduce new failure points: retrieval may miss evidence, tools may return errors, and models may misinterpret observations.</p>
<p><a href="https://arxiv.org/abs/2309.11495">Chain-of-Verification</a> The model is prepared to draft the answers, then to generate validation questions and independently examine them. It is appropriate to disassembly a specific factual answer, but “let one model examine itself” does not automatically amount to independent evidence. More reliable validation is obtained from the original document, rules, tests, calculator or another data path.</p>
<p>If the task requires strict grammar, the validation should preferably occur during the token generation, rather than pray for a successful resolution after the entire text has been written. This is the line between restricted decode and common tip projects. Prompt describes intent, code decoder semantics, both of which can be used together, but should not be impersonated against each other.</p>
<h3>Auto-generated, iterative and self-adaptation tips</h3>
<p><a href="https://arxiv.org/abs/2303.17651">Self-Refine</a> Use of the `revenue-revision-change' cycle improvement results;<a href="https://arxiv.org/abs/2211.01910">APE</a> and <a href="https://arxiv.org/abs/2309.03409">OPRO</a> The model is then used to generate a candidate command, which is then selected or iteratively by the task performance. Active-Prompting gives preference to the unsettled sample of the model for labelling, while Information-imptive Prompting is a hint for different input adjustments.</p>
<p>Such methods are indispensable to the assessment set. One case alone can easily be seen as a reminder of improvement. The candidate prompt should be compared with model decoded parameters, recording accuracy, format pass rate, cost and type of failure in the data that cover the real distribution. If no duplicate rating is available, automatic optimization is only an automatic rewrite.</p>
<h2>Site practice: from generation round to searchable workflow</h2>
<p>Google's guide covers the scenes of administration, communication, marketing, project management and sales. The specific occupation changes and the pattern behind the writing is stable: it is given materials, it describes the action, it defines the audience and the format, and it is then continued to be modified on the basis of results. Only four common categories of tasks are retained below.</p>
<h3>Writing and summary</h3>
<pre><code class="language-text">任务：把下面的技术说明改写成发布说明。

读者：已经使用旧版本、但不了解内部实现的开发者。

要求：
- 先说明用户能观察到的变化，再说明迁移注意事项。
- 保留版本号、命令和兼容性限定。
- 不写“重大升级”“全面赋能”等宣传性结论。
- 控制在 400 字以内。

原始材料：
&lt;source&gt;
&#123;&#123;技术说明&#125;&#125;
&lt;/source&gt;
</code></pre>
<p>The design focus is not “you are a professional technical writer”, but readers, authenticity requirements and the use of expressions is disabled. Where the source material lacks compatible information, the model should identify gaps rather than supplement an apparently reasonable migration proposal.</p>
<h3>Information extraction and structured results</h3>
<pre><code class="language-text">从合同文本中提取以下字段：合同主体、生效日期、终止日期、自动续约、付款周期。

规则：
- 字段没有出现时返回 null。
- 日期统一为 YYYY-MM-DD；原文无法确定具体日期时保留原始表达。
- 每个非空字段附带原文证据。
- 不根据常见合同惯例推断。

返回字段：
parties, effective_date, termination_date, auto_renewal, payment_cycle, evidence
</code></pre>
<p>The key to such a reminder is missing values and evidence, not a sentence “Please extract accurately”. If the result is directly entered, the schema verification and restricted decode should also be used; Prompt is responsible for semantics and the program is responsible for rejecting the unlawful object.</p>
<h3>Analysis and planning</h3>
<pre><code class="language-text">根据提供的需求、人员和截止日期，生成一份两周实施计划。

先列出你从材料中确认的约束，再输出任务依赖图和每日计划。
不要假设未提供的人员可用性。若计划无法在截止日期内完成，指出最小缺口并给出两个调整方案。
</code></pre>
<p>The planning mission is most afraid that the model will be used to provide complete answers. It is more reliable to list what is not feasible to report in a binding and visible manner than to require “a comprehensive plan”. Complex schemes also require calendars, code libraries, worksheets or solvers, which cannot be stopped in a single natural language.</p>
<h3>Multi-temporal</h3>
<p>The multi-round dialogue is appropriate to gradually reduce the problem, but it is not always the model that will remember what has been said. Each round of modifications should be expressly reserved for items and changes, such as:</p>
<pre><code class="language-text">保留上一版的事实、引用和章节顺序，只修改下面三点：
1. 把开头缩短到两段。
2. 将第二节的示例替换为给定的新案例。
3. 删除没有来源的效果判断。

修改后附一份变更清单，不要改动其他部分。
</code></pre>
<p>When the dialogue has accumulated a large amount of scrap and conflicting demands, it is usually more stable to organize a new mission statement than to continue adding a “reform”. This is why context projects are officially operational with respect to both content and reset.</p>
<p>Google Official English Handbook is directly accessible <a href="https://services.google.com/fh/files/misc/gemini_for_workspace_prompt_guide_october_2024_digital_final.pdf">Gemini for Workspace Prompting Guide</a>I'm sorry. We have a copy of the package. <a href="/assets/docs/Gemini_Prompt.pdf">Gemini Bilingual Handbook of Phrases</a>, which is appropriate for quick viewing of the original scene example.</p>
<h2>Prompt solves what, doesn't solve what</h2>
<table>
<thead>
<tr>
<th>Problem</th>
<th>Prompt, what can you do?</th>
<th>What else can I get you?</th>
</tr>
</thead>
<tbody><tr>
<td>The mission is not clear.</td>
<td>Additional action, boundary, examples and success criteria</td>
<td>Sample of real user needs and assessments</td>
</tr>
<tr>
<td>Lack of current facts</td>
<td>Tell the model to be based only on the response to the given material</td>
<td>Retrieving, database, search and reference</td>
</tr>
<tr>
<td>Complex calculations are easily incorrect.</td>
<td>Requires the generation of formulae, codes or check steps</td>
<td>Calculators, interpreters, tests and certifiers</td>
</tr>
<tr>
<td>JSON often parses failed</td>
<td>Description field, type and missing value</td>
<td>Schema, function call or restricted decode</td>
</tr>
<tr>
<td>Forgetting constraints in long missions</td>
<td>Restatement of key rules, compression context</td>
<td>Context management、memory、checkpoint</td>
</tr>
<tr>
<td>The model doesn't even know how to do it.</td>
<td>Provide a few examples for ad hoc adaptation</td>
<td>More appropriate models, SFTs, tools or process re-engineering</td>
</tr>
</tbody></table>
<p>The most comfortable location for the project is to convert an already available and assessable task. It can reduce ambiguity and cannot replace data, training, retrieval, tools and procedural constraints. Knowing when to stop prompt is often more important than learning another acronym.</p>
<h2>Method Index</h2>
<p>The table below retains the name of the method that appeared in the previous synthesis, but does not assign a short, repetitive chapter to each. Many methods have clear mission boundaries and the results of the papers rely on the models and data sets used at the time. Read the original paper and code before using them, and not just by name, to judge whether it fits the current system.</p>
<table>
<thead>
<tr>
<th>Methodology</th>
<th>Category</th>
<th>One word.</th>
<th>Source material</th>
</tr>
</thead>
<tbody><tr>
<td>Zero-shot / Few-shot</td>
<td>Basic expression</td>
<td>Show the model to recognize new tasks by job description or a few examples</td>
<td><a href="https://arxiv.org/abs/2005.14165">GPT-3</a></td>
</tr>
<tr>
<td>Chain-of-Thought（CoT）</td>
<td>Inference</td>
<td>Insert intermediate delineation steps in the illustrative examples</td>
<td><a href="https://arxiv.org/abs/2201.11903">Wei et al.</a></td>
</tr>
<tr>
<td>Zero-shot-CoT</td>
<td>Inference</td>
<td>Trigger intermediate steps with simple step-by-step instructions</td>
<td><a href="https://arxiv.org/abs/2205.11916">Kojima et al.</a></td>
</tr>
<tr>
<td>Auto-CoT</td>
<td>Auto-optimize</td>
<td>Cluster problems and automatically generate CTT examples</td>
<td><a href="https://arxiv.org/abs/2210.03493">Zhang et al.</a></td>
</tr>
<tr>
<td>Self-Consistency</td>
<td>Decoded and decoded</td>
<td>Samp multiple reasoning paths to aggregate answers</td>
<td><a href="https://arxiv.org/abs/2203.11171">Wang et al.</a></td>
</tr>
<tr>
<td>LogiCoT</td>
<td>Training and reasoning</td>
<td>Learning Logic through institution learning COT and adding to the check process</td>
<td><a href="https://arxiv.org/abs/2305.12147">Zhao et al.</a></td>
</tr>
<tr>
<td>Chain-of-Symbol（CoS）</td>
<td>Inference</td>
<td>A compact symbol for space relations and planning steps</td>
<td><a href="https://arxiv.org/abs/2305.10276">Hu et al.</a></td>
</tr>
<tr>
<td>Tree of Thoughts（ToT）</td>
<td>Search</td>
<td>Generate, evaluate and withdraw candidate thinking in tree structures</td>
<td><a href="https://arxiv.org/abs/2305.10601">Yao et al.</a></td>
</tr>
<tr>
<td>Graph of Thoughts（GoT）</td>
<td>Search</td>
<td>Combine, aggregate and improve the middle lines with graphics</td>
<td><a href="https://arxiv.org/abs/2308.09687">Besta et al.</a></td>
</tr>
<tr>
<td>System 2 Attention（S2A）</td>
<td>Context Process</td>
<td>Rewrite the context to reduce the impact of unrelated information on the answers</td>
<td><a href="https://arxiv.org/abs/2311.11829">Weston &amp; Sukhbaatar</a></td>
</tr>
<tr>
<td>Thread of Thought（ThoT）</td>
<td>Context Process</td>
<td>Answers after a summary of long or confusing context subparagraphs</td>
<td><a href="https://arxiv.org/abs/2311.08734">Zhou et al.</a></td>
</tr>
<tr>
<td>Chain-of-Table</td>
<td>Special reasoning.</td>
<td>Complete table asking through continuous table operations Answer</td>
<td><a href="https://arxiv.org/abs/2401.04398">Wang et al.</a></td>
</tr>
<tr>
<td>Self-Refine</td>
<td>Auto-optimize</td>
<td>Use model-generated feedback loop to modify the first draft</td>
<td><a href="https://arxiv.org/abs/2303.17651">Madaan et al.</a></td>
</tr>
<tr>
<td>Code Prompting</td>
<td>The reasoning suggests that</td>
<td>Recast the issue of natural language into code and delineate it as a subsidiary condition.</td>
<td><a href="https://arxiv.org/abs/2401.10065">Madaan et al.</a></td>
</tr>
<tr>
<td>ECHO</td>
<td>Auto-optimize</td>
<td>COT example of cluster and repeatedly coordinated automatically generated</td>
<td><a href="https://arxiv.org/abs/2409.04057">Self-Harmonized CoT</a></td>
</tr>
<tr>
<td>Instance-adaptive Prompting（IAP）</td>
<td>Self-adaptation</td>
<td>Select or reorganize zero-shot COT tips from the current instance</td>
<td><a href="https://arxiv.org/abs/2409.20441">Zhang et al.</a></td>
</tr>
<tr>
<td>Layer-of-Thoughts（LoT）</td>
<td>Specialized search</td>
<td>Organize candidate screens for legal searches at a binding level Choose</td>
<td><a href="https://arxiv.org/abs/2410.12153">Choi et al.</a></td>
</tr>
<tr>
<td>Narrative-of-Thought（NoT）</td>
<td>Special reasoning.</td>
<td>The narrative structure and procedures are used to indicate the time reasoning of the processing</td>
<td><a href="https://arxiv.org/abs/2410.17607">Kim et al.</a></td>
</tr>
<tr>
<td>Buffer of Thoughts（BoT）</td>
<td>Logical reuse</td>
<td>Save and retrieve reusable high-level think templates</td>
<td><a href="https://arxiv.org/abs/2406.04271">Yang et al.</a></td>
</tr>
<tr>
<td>CD-CoT</td>
<td>Luo-Boo-Boo-Boo-Boo-Boo-Boo-Boo-Boo-Boo-Boo-Boo-Boo-Boo-Boo-Boo-Boo-Boo-Boo-Boo-Boo-Boo-Boo-Boo-Boo-Boo-Boo-Boo-Boo-Boo-Boo-Boo-Boo-Boo-Boo-Boo-Boo-Boo-Boo-Boo-Boo-Boo-Boo-Boo-Boo-Boo-Boo-Boo-Boo-Boo-Boo-Boo-Boo-Boo-Boo-Boo-Boo-Boo-Boo-Boo-Boo-Boo-Boo-Boo-Boo-Boo-Boo-Boo-Boo-Boo-Boo-Boo-Boo-Boo-Boo-Boo-Boo-Boo-Boo-B-B-B-B-B-B-B-B-B-B</td>
<td>Rewrite, select and vote on the Noise CTT Example</td>
<td><a href="https://arxiv.org/abs/2410.23856">Zhou et al.</a></td>
</tr>
<tr>
<td>Chain of Draft（CoD）</td>
<td>Efficient reasoning</td>
<td>Use a very short middle step to lower the reasoning token and delay</td>
<td><a href="https://arxiv.org/abs/2502.18600">Xu et al.</a></td>
</tr>
<tr>
<td>Retrieval-Augmented Generation（RAG）</td>
<td>Search</td>
<td>Once you have access to external evidence, you can generate answers.</td>
<td><a href="https://arxiv.org/abs/2005.11401">Lewis et al.</a></td>
</tr>
<tr>
<td>ReAct</td>
<td>Tool Call</td>
<td>Let reasoning, action and environmental observation intersect.</td>
<td><a href="https://arxiv.org/abs/2210.03629">Yao et al.</a></td>
</tr>
<tr>
<td>Chain-of-Verification（CoVe）</td>
<td>Authentication</td>
<td>Generate validation questions and independent answers for first drafts</td>
<td><a href="https://arxiv.org/abs/2309.11495">Dhuliawala et al.</a></td>
</tr>
<tr>
<td>Chain-of-Note（CoN）</td>
<td>Search and Validation</td>
<td>Generate notes for retrieval of documents, filter unrelated or conflicting materials</td>
<td><a href="https://arxiv.org/abs/2311.09210">Yu et al.</a></td>
</tr>
<tr>
<td>Chain-of-Knowledge（CoK）</td>
<td>Knowledge integration</td>
<td>Phased preparation, acquisition and adaptation of external knowledge</td>
<td><a href="https://arxiv.org/abs/2305.13269">Li et al.</a></td>
</tr>
<tr>
<td>Scratchpad Prompting</td>
<td>Inference</td>
<td>Generate any intermediate calculation before the final answer Arguments</td>
<td><a href="https://arxiv.org/abs/2112.00114">Nye et al.</a></td>
</tr>
<tr>
<td>Program of Thoughts（PoT）</td>
<td>Tools and reasoning</td>
<td>Use the program to calculate and give it to the interpreter to execute</td>
<td><a href="https://arxiv.org/abs/2211.12588">Chen et al.</a></td>
</tr>
<tr>
<td>Structured CoT（SCoT）</td>
<td>Code Generation</td>
<td>Planning code by order, branch and circular structure</td>
<td><a href="https://arxiv.org/abs/2305.06599">Li et al.</a></td>
</tr>
<tr>
<td>Chain of Code（CoC）</td>
<td>Code reasoning</td>
<td>Generate pseudocodes and execute them with language model enhanced interpreter</td>
<td><a href="https://arxiv.org/abs/2312.04474">Li et al.</a></td>
</tr>
<tr>
<td>Active-Prompting</td>
<td>Auto-optimize</td>
<td>Prioritize the most uncertain issue of the model to be marked CTT</td>
<td><a href="https://arxiv.org/abs/2302.12246">Diao et al.</a></td>
</tr>
<tr>
<td>Automatic Prompt Engineer（APE）</td>
<td>Auto-optimize</td>
<td>Generate candidate commands and search for task performance tips</td>
<td><a href="https://arxiv.org/abs/2211.01910">Zhou et al.</a></td>
</tr>
<tr>
<td>Automatic Reasoning and Tool-use（ART）</td>
<td>Tool Call</td>
<td>Retrieving examples from the task library and automatically combining reasoning and tool steps</td>
<td><a href="https://arxiv.org/abs/2303.09014">Paranjape et al.</a></td>
</tr>
<tr>
<td>Contrastive CoT（CCoT）</td>
<td>Example Learning</td>
<td>And provide examples of correct and erroneous reasoning as a comparison</td>
<td><a href="https://arxiv.org/abs/2311.09277">Chia et al.</a></td>
</tr>
<tr>
<td>EmotionPrompt</td>
<td>Job expression</td>
<td>Add emotional irritation to the hint and measure mission performance</td>
<td><a href="https://arxiv.org/abs/2307.11760">Li et al.</a></td>
</tr>
<tr>
<td>Optimization by PROmpting（OPRO）</td>
<td>Auto-optimize</td>
<td>Let LLM continue to offer his solution based on his historical candidacy and scores.</td>
<td><a href="https://arxiv.org/abs/2309.03409">Yang et al.</a></td>
</tr>
<tr>
<td>Rephrase and Respond（RaR）</td>
<td>Job expression</td>
<td>Questions are rewritten and extended before the final answer is generated</td>
<td><a href="https://arxiv.org/abs/2311.04205">Deng et al.</a></td>
</tr>
<tr>
<td>Step-Back Prompting</td>
<td>Job expression</td>
<td>First, abstract high-level concepts and principles, then specific examples.</td>
<td><a href="https://arxiv.org/abs/2310.06117">Zheng et al.</a></td>
</tr>
</tbody></table>
<h2>References</h2>
<ul>
<li><a href="https://arxiv.org/abs/2406.06608">The Prompt Report: A Systematic Survey of Prompting Techniques</a></li>
<li><a href="https://arxiv.org/abs/2402.07927">A Systematic Survey of Prompt Engineering in Large Language Models</a></li>
<li><a href="https://services.google.com/fh/files/misc/gemini_for_workspace_prompt_guide_october_2024_digital_final.pdf">Gemini for Workspace Prompting Guide</a></li>
<li><a href="https://arxiv.org/abs/2005.14165">Language Models are Few-Shot Learners</a></li>
<li><a href="https://arxiv.org/abs/2201.11903">Chain-of-Thought Prompting Elicits Reasoning in Large Language Models</a></li>
</ul>
<p>Some of these methods have become generic engineering components, while others are still experimental designs in specific papers. The technical catalogue is intended to help locate, not to imply that all methods should be entered into the same programt. In the face of specific mandates, starting with clear mandates, reliable context and enforceable validation, most of the problems are usually resolved.</p>
