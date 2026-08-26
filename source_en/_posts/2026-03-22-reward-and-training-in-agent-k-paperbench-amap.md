---
title: 'Reward and Training Loops in Real Agents: From Data Governance to Online RL'
title_zh: Reward 与 Training 在真实 Agent 中如何闭环：从数据治理到在线 RL
date: 2026-03-22 20:30:00 +0800
categories:
- Agent Systems
- Agent Training
tags:
- Reward Modeling
- Data Curation
- Verifiers
- Online RL
author: Hyacehila
mathjax: true
excerpt: A rewritten view of agent training pipelines, from data governance, tool environments, and verifier design to trajectories,
  SFT, curriculum, online RL, and benchmark audits.
description: A rewritten view of agent training pipelines, from data governance, tool environments, and verifier design to
  trajectories, SFT, curriculum, online RL, and benchmark audits.
excerpt_zh: 按真实系统的训练流水线重写 Agent 训练闭环：从数据治理、工具环境和 verifier 设计，到 trajectories、SFT、curriculum、online RL 与 benchmark audit。
permalink: /blog/2026/03/22/reward-and-training-in-agent-k-paperbench-amap/
lang: en
translation_key: 2026-03-22-reward-and-training-in-agent-k-paperbench-amap
translation_status: machine
translation_source_hash: d2a1517e0a29c7b4ce0175a0096b242c8b6ba9803e9c88a566f179f1dee60909
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<p>The first two series are about how you're being produced, and the other one about how you're being trained, and how you're becoming a part of it. <code>advantage</code>、<code>update</code> And angstic RL's closed ring. But it's not enough to talk about it, because the reader is not the one who can get caught in the most difficult way, but the more engineering question:<strong>What links does the real agent system usually start to connect?</strong></p>
<p>The questions in this article can also be addressed<a href="/en/blog/2026/03/21/from-sft-to-agentic-rl-training-loop/">Actic RL: Why is training closed rings more important than training algorithms?</a>、<a href="/en/blog/2026/03/23/mineclip-visual-reward-appendix/">MineCLIIP, Visual Signals and Incentives Functions</a>How the concept of a relatively close read together is developed in different contexts.</p>
<p>The answer is not usually " write a reward function first." More authentic sequences are: raw data and history query, followed by data governance and environmental architecture; environment running before the system produces logs, tests, database status, tool outputs and user interactions; these feedbacks are then organized into verifier, judge, rubric or outcome reward; then to veried projects, SFT, curriculum and online; and finally, benchmark is going back to auditing, and we are not optimizing the right goal.</p>
<p>In the real system, reward is not a single score, but a feedback interface that works with data, environment, certification. To clarify this line, I will not summarize it by a single article, but rewrite it by a system stream. Main Line by <code>AMAP</code> Provides because it is closest to the complete recipe from data governance to RL;<code>Agent K</code>、<code>MLE-bench</code>、<code>tau-bench</code>、<code>PaperBench</code> and <code>Benchmark^2</code> The chain is embedded in different parts of the chain, and answers questions about the environment, the assessment structure, deployment targets and benchmark audits.</p>
<p>If, after reading this article, the reader can repeat the whole process of rewarding from data processing, then the article will be the one that's working.</p>
<h2>Why isn't the real system starting with a reward function?</h2>
<p>Much of the RL discussion is easily floated because the default system has an incentive function, and all the work that follows is simply optimising it. But in real anent, that order almost always reverses.</p>
<p>Because the model does not face the target itself, it will only face a world that can interact with each other when it is engineered. The world has original logs, history query, tool protocols, failure codes, user feedback, test scripts, database status, and judicial and verifier. The whole world is just a compression expression on one of the interfaces. If data, environment and assessment structures have not been built, stable rewards, let alone training, are not possible.</p>
<p>That's why I'd rather read the real one into one. <code>feedback production pipeline</code>I'm sorry. The first half of the pipeline is responsible for rewriting the task into an environment and for rewriting the feedback from the environment into a monitorable object; the second half is responsible for sifting the objects into high-value tracks and writing back to SFT and RL. Many papers speak of a trainer, and the place where the ceiling is decided is often before a trainer; algorithms and training are indeed important, but data are at least as important.</p>
<h2>The location of the papers in this waterline.</h2>
<p>The role of several literatures is to be pinned down with a table. The phrase “six paragraphs of the summary” is not to be dropped back when it is expanded.</p>
<table>
<thead>
<tr>
<th>Documentation</th>
<th>Position in the whole process</th>
<th>What's the main supply?</th>
<th>Positioning length in this document</th>
</tr>
</thead>
<tbody><tr>
<td><a href="https://arxiv.org/abs/2512.24957">AMAP</a></td>
<td>Data governance -&gt; Environment -&gt; reward -&gt; SFT -&gt; curriculum -&gt; RL</td>
<td>Complete training</td>
<td>Main</td>
</tr>
<tr>
<td><a href="https://arxiv.org/abs/2411.03562">Agent K</a></td>
<td>Environmentalization and feedbackbackback</td>
<td>How complex missions are first rewritten into learning environments</td>
<td>Focus support</td>
</tr>
<tr>
<td><a href="https://arxiv.org/abs/2410.07095">MLE-bench</a></td>
<td>standardise tasks</td>
<td>How the ML engineering mission became a public commentator</td>
<td>Support</td>
</tr>
<tr>
<td><a href="https://arxiv.org/abs/2406.12045">tau-bench</a></td>
<td>Deployment level interactive environment and outcome response</td>
<td>How user-tool-rule-database interactions are written as objective rewards</td>
<td>Focus support</td>
</tr>
<tr>
<td><a href="https://arxiv.org/abs/2504.01848">PaperBench</a></td>
<td>Judge relies on evaluation structures</td>
<td>Open research re-enactment mission to break down</td>
<td>Support</td>
</tr>
<tr>
<td><a href="https://arxiv.org/abs/2601.03986">Benchmark^2</a></td>
<td>benchmark audit</td>
<td>Benchmark own stable, distinguishable, aligned capability level</td>
<td>- The beaming aid.</td>
</tr>
</tbody></table>
<p>This table also explains a judgment that will be repeated:<strong>Not every paper is directly about the training formula.</strong> Some papers are mainly defining the environment, some are defining benchmark and some are auditing benchmark. They all matter, but in different ways. And that's why the next main body on data governance and training governance will be clearly oriented. <code>AMAP</code>、<code>Agent K</code> and <code>tau-bench</code> Slash, and <code>MLE-bench</code>、<code>PaperBench</code>、<code>Benchmark^2</code> The role of “training targets” will be increased.</p>
<h2>I. From raw data to training Query Pool: data governance and environmental preparedness</h2>
<h3>AMAP's not about RL, it's about 30 million history query.</h3>
<p>If you only read the secondhand, a lot of people would. <code>AMAP Agentic Planning Technical Report</code> Read a map of antgent + RL. But it is difficult to understand it when you read it directly. Because AMAP's first solution is not an optimizer, but...<strong>Data governance</strong>。</p>
<p>The paper uses anonymous user logs in the real world as a starting point, with a three-month window, totalling over 30 million. The problem is that these logs are not training data, but are only traces left by the original world: they are very noiseful, repetitive, and distributed in an uneven manner, and they are completely free of readily available difficulty labels, task type labels or labels that are suitable for angent learning. Directly trained, it only feeds the model with noise and high head demand.</p>
<p>So the first thing AMAP did was build a set for these primitive querys. <code>Intent Taxonomy</code>I'm sorry. This taxonony is not just a few categories, but a layered framework of intent: final harvest in the paper. <code>5</code> Category I (Rules and Policies, Discovery, Planning and Regulation, Dynamic Investment, Application International),<code>16</code> Category II <code>30</code> A fine particle size node. This taxonony isn't a brainshot, it's what the paper says. <code>Seed-Driven Evolutionary Framework</code>- It's growing up. Specifically, there are four phases:</p>
<p><strong>Stage 1: Seed Initialization。</strong> Manually select a small group of high-spectrum seeds prompt()&#36;D_{seed} \subset D_{pool}&#36;) with an open intention label for each query by an expert. Each query is mapped into a group of pieces &#36;S_i = \langle q_i, T_i \rangle&#36;, of which &#36;T_i = {t_{i,1}, t_{i,2}, \ldots, t_{i,k&#125;&#125;&#36; This command covers. &#36;k&#36; A point of intent that is either active or complementary. This is not a quantitative step, but rather a major diversity in the field that allows seed collections to cover as much as possible.</p>
<p><strong>Stage 2: LLM-Driven Category Induction。</strong> Use the marked seed set to get the first class (Level-1 classes) of the submitted. The role of LLM here is:&quot;Abstract structure from specific samples&quot;Instead of creating a classification system in a vacuum.</p>
<p><strong>Stage 3: Iterative Refinement Loop。</strong> To prevent the illusion or omission of LLM in general, the paper designed a strict approach. <code>Tag-Feedback-Correction</code> Cycle: LLM remarks the torrent collection with the current version of taxonony, human experts review the results, identify ambiguities and gaps in coverage, and feed back LLM feedback to fix taxonomy. The paper wrote this process. &#36;T^{(k+1)} = \text{Refine}\big(\text{Annotate}(D_{seed}, T^{(k)}),; \text{Human Feedback}\big)&#36;, the cycle continues to &#36;T^{(k+1)} \approx T^{(k)}&#36;That's the taxonony condensed into a stable structure. &#36;T^*&#36;。</p>
<p><strong>Stage 4: Dynamic Taxonomy Expansion。</strong> The first three steps can only cover the intended space that seed energy reaches. To capture the long end, the paper is mandatory to keep one at each node. <code>Other</code> Category. When large-scale original logs are marked, samples that fall to either are analysed separately, and new and emerging headings are found and consolidated. This mechanism has transformed tuxonomy from a static tree into an evolutionary system that can adapt to the distribution of open world query.</p>
<p>Data governance here is not simply cleaning, but rather,<strong>First, we cut the world into manageable space of intent.</strong>(As they say, first, the sample is classified) And this classification is also adopted.&quot;Seed drive + LLM summation + human correction + Other dynamic expansion&quot;It grows out of it.</p>
<p>This is a crucial step, as training requires more than just more data, but also a manageable distribution of data. Without taxonony, you can only see a blurry ocean; with taxonomy you can talk about coverage, long end proportions, difficult samples, binding density, and which areas are more scarce and which areas are more worthy of expansion.</p>
<h3>2. Quality data governance is not a weighting exercise, but rather a combination of maintaining coverage and difficulty spectrum Yes</h3>
<p>After Taxonony stabilized, AMAP made two more sophisticated types of governance. The paper calls these two steps together. <code>Annotation and Data Curation</code>。</p>
<p><strong>The first category is multi-dimensional labelling (Procise Multi-Dimensional Action).</strong> The paper included intent to understand visual modelling as a multi-label classification task. Each user command is mapd into a composite label vector &#36;V = \langle I_{primary}, I_{secondary}, C_{constraints} \rangle&#36;I'm sorry. of which &#36;I_{primary}&#36; and &#36;I_{secondary}&#36; The following is a list of the most recent events in the world:&#36;C_{constraints}&#36; The auxiliary constraints are captured, such as spatial scope, time budget, transport preferences, etc. The point is that the model will not be “a navigational request” but rather “a primary task, a binding chain, a tool chain”. The paper uses a high-volume teacher LLM to carry out large-scale labelling on the structured log, with specific prompt template in appendix A.2.</p>
<p><strong>The second category is layered filtering (Controlled Sampling via Funnel Filtering).</strong> AMAP has designed a three-tier funnel filter strategy to systematically eliminate redundancy at the vocabulary, semantics, geometry levels:</p>
<ul>
<li><strong>Lexical Redundancy Society:</strong> Do it at the whole language level. <code>Locality-Sensitive Hashing</code>, the close to repeat string and the textual trash are removed efficiently. The goal of this step is to reduce the initial volume significantly and clear the obvious repetitions and noises first.</li>
<li><strong>Semantic Redundancy Community:</strong> Press &#36;\langle I_{primary}, I_{secondary} \rangle&#36; The metagroup splits the data into barrels, does embedding similarity searches in each barrel, and cuts out semantic redundant samples -- those querys that are different in terms but below the threshold in hidden space. The paper specifically mentioned that they used data sets to process data of this scale. <code>Faiss</code> Speed up the search and get the computational complexity from &#36;O(N^2)&#36; The violence has been reduced to sublinear or near linear. This step ensures a difference in the barrel: there is a sufficient difference between the samples left under the same intent category.</li>
<li><strong>Geometric Redundancy:</strong> Use semantics to go to the pool. <code>K-Center-Greedy</code> The algorithm selects the most representative sample. The core idea of this algorithm is to maximize the smallest distance between selected samples in the embeding space, with the effect of prioritizing the retention of those querys distributed over borders and long tails. That is, it's not simply seeking less data, but it's in the...<strong>Reduce redundancy while preserving the shape of the distribution</strong>- Especially those that are essential to the angent rut.</li>
</ul>
<p>After this set of governance, AMAP from <code>3000 万+</code> Original history query sifted around <code>20 万</code> Candidature Qury, underrepresented <code>1%</code>I'm sorry. That's why I keep saying that real angent training never is "a lot of data." Here, data governance itself is already part of the training.</p>
<h3>3. Difficulty labels and negative samples are not accessories, but are the foundation of the subsequent curriculum</h3>
<p>AMAP also has a design that is often overlooked but is important to understanding the whole process: it does not just mark the intent, but also the difficulty. The paper called this part <code>Difficulty and Diversity</code>The motivation is clear: static data distribution leads to inefficient RL training. The early model can only get zero rewards for the difficult task, and the later model can only get full points for the simple task (the difference disappears). AMAP designed a set to solve this problem. <code>Execution-Simulation Scoring Mechanism</code>, from three orthogonal dimensions to assess difficulty:</p>
<ul>
<li><strong>Cognition Road in Tool Management:</strong> Measuring the degree of ambiguity of the intended map. From a visible map (Score 1-2, for example, "navigation to X") to a hidden reasoning (Score 4-5, for example, "see a place to drive in 20 minutes, suitable for quiet reading, and need abstract interpretation".</li>
<li><strong>Export Challenge (Enhancement of the Implementation Chain):</strong> Quantify the logical complexity of the solver path, the number, type and depth of reliance (in sequence vs in parallel).</li>
<li><strong>Construment Complexity:</strong> Assess the density of constraints such as space, time, preferences, and the extent to which angent needs to be jointly optimized.</li>
</ul>
<p>The paper uses a strong teacher LLM as an automatic assessor to give a standard difficulty score to the structured user logs at these three dimensions &#36;r \in {-1, 0, 1, \ldots, 5}&#36;I'm sorry. of which <code>-1/0</code> Special purpose:<code>-1</code> The blogger says that the government is not responsible for the crime.<code>0</code> The blog also shows that the Internet is a tool that is not needed.</p>
<p>More notably, AMAP has built one. <code>Irrelevance Dataset</code>, to teach model recognition “this thing goes beyond the limits of the instrument” or “should refuse to make up”. These Score-1 and 0 samples are clearly proportional in the area of Systems Control. It's very important to remember, because it shows that in the real system,<strong>Negative sample governance is also part of data governance</strong>I'm sorry. If the training package is only capable of accomplishing its tasks, models usually do not learn to have a border perception.</p>
<h3>4. The inspiration of MLE-bench: benchmark also needs to be managed by data</h3>
<p>I'm not sure if I'm going to be able to get a good look at this.<code>MLE-bench</code> What actually does is the same thing in the Benchmark version. It's from Meta Kagle. <code>5673</code> After a closed competition, filtered through community competitions, manually screened representativeness and replicability, and eventually left behind <code>75</code> A test assignment. Each task also marked problem type, complexity rating (experience ML engineer needs) &lt;2h / 2-10h / &gt;10h) and re-lock the explanatory project level threshold with the Kagle Private leaderboard ' s metal logic.</p>
<p>This is a direct inspiration:<strong>Not only is training data to be governed, but benchmark data to be governed.</strong> Once the benchmark has not been organized into a stable mission contract, the latter scores will not be explained. MLE-bench does not take the lead in training in this paper, but it shares the same principle as AMAP ' s data governance: the task space is structured, then the assessment and optimization are discussed.</p>
<h2>Two, make the environment first, then talk about it.</h2>
<p>If the answer to the question of data governance is worth learning, then the answer to the question of environmental architecture is “where exactly do models come from?</p>
<h3>1. AMAP: steady the tol world before training</h3>
<p>AMAP’s environment is much like the kind of work that many people think is not academic, but important in engineering. The paper started with a high-security tool environment covering maps, travel, weather, information retrieval categories, total <code>10</code> Special tools; reuse <code>FastMCP</code> Harmonization of the tool call protocols; then integration of the training infrastructure to support the walkout rollout and walk-through training.</p>
<p>These designs sound like infrastructure details, but they actually determine whether the RLs that follow can happen. Because for the tool-integraded environment is not an abstract container, but a generator of rewards and credit awards. The back end of the tool collapses as long as the return is slow, the protocol is unstable and the trajectory is not synchronized. AMAP is worth talking about the whole process because it doesn't use environment as a default black box, but rather as a training tool.</p>
<h3>2. Agent K: Complex tasks must be operated by scafferd</h3>
<p><code>Agent K</code> The same thing is illustrated from another angle. It deals with a very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, very, <code>Workspace Scaffold</code> and <code>Solution Generation Scaffold</code> Two floors.</p>
<p>Yes. <code>Workspace Scaffold</code> , angent first needs to understand how input output structure, task type, metric, submission format and workspace are organized. This step has not yet been ostensibly modelled, and is in fact recasting the original obscurantious open task into an enforceable, verifiable, multi-wheel-modifiable workspace.</p>
<p>Only the work area is out first, the back. <code>Solution Generation Scaffold</code> This is where the ground is: feature engineering, model selection, training, iterative, debugging, resubmission, and not a bunch of drifting intentions. The most important inspiration that Agent K gives us is not "LLM can do Kaggle," but...<strong>Complex missions must be scafferded in the environment before feedback becomes structural.</strong>。</p>
<h3>3. tau-bench: In a real deployment environment, angent faces users, rules and data at the same time Library</h3>
<p><code>tau-bench</code> Another step towards operationalizing the environment. It's not just a mission statement for the model, but a multiple-wheeled mission. <code>tool-agent-user</code> Interactive world: Database status is hidden, angent can read and write databases through API, and also repeats conversations with the LM simulator users and always follows the domain-special policy document.</p>
<p>This is particularly important because it shows that the real environment is not as simple as the "task + tool" but consists of at least four layers: users, tools, rule files, bottom state. In such an environment, you train and assess angent, and the real learning is not just about “choice-to-function”, but about how to keep up with information, maintain context in multiple rounds, adjust paths under rules and, if necessary, refuse user requests or offer alternatives.</p>
<p>That is, AMAP, Agent K and tau-bench, though different in their field, are telling us the same thing:<strong>Without environmental contracts, there is no stable reward; without a structured environment, there is no interpretative training.</strong></p>
<h2>III. HOW TO REWARD / VERIFier / JUDGE READ THE OBJECT TO BE OVERSIGHTable</h2>
<p>The system is not really going to be able to get into the reward process until the environment is set. The most misleading thing here is that reward is not necessarily a total point that is directly optimized. And so many times, rewards begin with the assessment structure.</p>
<h3>1. AMAP: rubrics-as-reward not black box scores, but first to undivide dimensions</h3>
<p>AMAP reward signature is a good example because it really makes rubric a training interface. The paper disassembled the mass of the angent track into three dimensions:</p>
<ul>
<li><code>Reasoning and Proactive Planning</code></li>
<li><code>Information Fidelity and Integration</code></li>
<li><code>Presentation and Service Loop</code></li>
</ul>
<p>This is a very representative method. The first dimension assesses whether angent has an economic, effective and proactive implementation plan; the second dimension depends on whether it can faithfully extract and integrate the output of the tool; and the third dimension on whether it actually finishes the service closed loop and can structure and clearly deliver the results to the user.</p>
<p>More importantly, AMAP did not give these three-dimensional fixed death weights, but rather redistributed them by type of task. Complex planning types of missions are more reasoned and proactive; information retrieval types are more factual; advisory interpretation types are more expression-delivery and service-closure. This step is crucial because it implies systematic recognition of a fact that is often overlooked:<strong>The good answers to different tasks are not shared with the same set of fixed mandates.</strong></p>
<p>The most important thing to remember alone is that it's about the hallucination. <code>hard veto</code>I'm sorry. The paper is clear: as long as facts that cannot be supported by the output of tools, such as information illusions of time, price, distance, are created, the reward is ultimately zero. That is, AMAP clearly dissociates from the logic of the average of the mistakes that must never be made. This design is so important for real anent that some mistakes should not have been offset by better performance elsewhere.</p>
<h3>2. tau-bench: Some tasks do not require judge, summary is itself reward</h3>
<p>If AMAP shows a rubric reward, then tau-bench shows another important writing:<code>outcome reward</code>。</p>
<p>After its end, the system looked at two things:</p>
<ul>
<li>Whether the final database status is consistent with the only correct target status</li>
<li>Angent answers the necessary information for the user</li>
</ul>
<p>This is more plain than rubric, but it's just about saying:<strong>Not all open interactive missions have to rely on black box judge.</strong> Many key feedbacks can still be objectively verifiable, and RLVR remains the most reliable idea when conditions permit, provided that the mission is well designed, that the final exit is written and the necessary information items are defined.</p>
<p>Tau-bench also wrote very clearly about the deployment level targets. The code is a common feature of the world. <code>pass@k</code>, which means running k at least once; and tau-bench <code>pass^k</code> The question is: can you run k every time? For real customers, retail, booking, and rule-based categories, the latter indicator is closer to deployment requirements. It does not measure “can it be unsolved occasionally”, but “can it be undefended in a stable and disciplined manner”.</p>
<h3>3. PaperSmart: In the open mission, the judge is able to stabilize his work if he has to have an eval tree</h3>
<p><code>PaperBench</code> The idea of how the open mission was judged by the judge is very systematic. It assesses the re-emergence of research projects, which are naturally much more open than the general code, so the paper does not give Judice a direct impression of the total, but starts by tearing the reproduction of each paper into a hierarchical rubric tree. The whole benchmark final overwrite <code>20</code> A paper and unearthed <code>8316</code> A requirement for an independent rating.</p>
<p>More importantly, it clearly divides the leaves into three categories:</p>
<ul>
<li><code>Code Development</code></li>
<li><code>Execution</code></li>
<li><code>Result Match</code></li>
</ul>
<p>This is crucial because it forcibly removes three processes that are often mixed up in a recapitulation: whether the code is correct, whether the script is running, and whether the results are sufficient to support the conclusions of the paper. The heart of the inspiration for PaperSmart is not that LLM Judge is already strong, but that it is not the heart of the story.<strong>In an open mission, the premise of the judge is to break the subject into a partially legible, cumulative and incrementally scalable structure.</strong>。</p>
<p>We'll do it later. <code>JudgeEval</code>The same principle is actually being stated. The best in the paper, Judge, did it on Judge Eval. About <code>F1=0.83</code>And this is not about declaring the judge correct, but about reminding us that the judge is also proxy and that it must be be benchmark.</p>
<p>By this point, the relationship between reward / vvefier / judge is clear:</p>
<ul>
<li>Writes as final consistency, as far as possible, an object exit home</li>
<li>Unable to write to final, undisable as possible</li>
<li>Judge should not face a vague overall objective, but rather a structured response.</li>
</ul>
<h2>IV. VERSITE TRANSFIELD TRANSPORTS HOW TO PRODUCURE, CHECK, BACK TO SFT</h2>
<p>By this point, the system already has data, an environment, a reward structure. The next question is:<strong>How did the training data come out?</strong></p>
<h3>1. AMAP: reward not the gradient, but the data gate.</h3>
<p>AMAP gives a complete recipe at this step. It starts with a high-quality prompt Pool, then double-tracks the data control.</p>
<p>The first path is the generation of offline candidate tracks. The paper clearly states that it uses a strong model. <code>DeepSeek-R1</code> For each query generation <code>K=8</code> Can not open message <code>Gemini-3-Pro-Preview</code> As a verifier, follow the three-dimensional reward to score. The selection rules are strict:<strong>The trajectory will be retained only if it is fully scored on all dimensions.</strong></p>
<p>This means that the first thing that is called is not a policy input, but a training data entry condition. It decides what is worth learning before it decides how to learn. That's a good idea to get cold start data.</p>
<p>The second route is long-tail data synthesis. Because the real distribution of high frequency tasks is always much more complex than complex planning tasks, it is only by historical query recycling that models can learn a lot of navigation, retrieval, short questions and answers, but not rare but high-value multi-barrel planning. To do this, AMAP will sample a complex set of tools that are rare in real distribution, and then use ICL to make the reverse synthesis of powerful models a user that must call these tool chains. And then the synthetic query still needs to go back to the same offline screen line to verify enforceability.</p>
<p>This suggests that the real agent data engine is not usually as simple as recycling logs, but that three things happen simultaneously: recycling from the real world, synthesis from powerful models, and re-use the verifier.</p>
<h3>2. Agent K: Multi-source feedback in complex systems that is suitable for first-time filtering</h3>
<p>If this is the one that's gonna be <code>Agent K</code> And when you look at it together, the logic becomes clearer. The real depend of Agent K is never a single scale, but a set of feedbackbacks: unit testes, meta-unit testes, execution logs, service score, public leaderboard, Private leaderboard. These signals share the functions of gates, diagnostics, bias and final screening.</p>
<p>This is not the same as AMAP's verifier flying, although it is not in a technical form, but it means a high degree:<strong>Complex agent rewards often take the filter and then enter the optimizer.</strong> In other words, deciding which tracks are worth getting into the training set is often more important than discussing how they get into the lost.</p>
<h3>3. Data governance and training governance began to split here</h3>
<p>This is the first step in cutting the whole line in half. The first half is data-based and the second half is training-based.</p>
<table>
<thead>
<tr>
<th>Data governance</th>
<th>Training governance</th>
</tr>
</thead>
<tbody><tr>
<td>Make taxonony, define taskspace</td>
<td>Create a priori for behaviors in the first tier with seen SFT</td>
</tr>
<tr>
<td>Weighted, representative choice, long end additions</td>
<td>Measure what the current policy will and won't</td>
</tr>
<tr>
<td>Plots, constraints, difficulty, border samples</td>
<td>You know, using signature/noise control to decide what samples are worth continuing. Learn.</td>
</tr>
<tr>
<td>Filter tracks with verifier / tests / outside reward</td>
<td>Take over different difficulty areas with second-size SFT and RL</td>
</tr>
<tr>
<td>Construct a negative sample ofirrelevance / hallucination</td>
<td>Leave fronters questions online RL</td>
</tr>
</tbody></table>
<p>The core of this table is simple: the whole process of rewarding and training goes beyond training, and there is an entire section of data governance ahead. Many systems seem strong, and the real gap is often from the fact that the half of them are solid enough.</p>
<h2>V. SFT AND CLASSING</h2>
<h3>1. Multistep tool trajectories SFT, what the hell is it writing?</h3>
<p>AMAP is next designed to correct a common misconception: SFT is not a warmup before RL, but rather the first level of behavior of an individual.</p>
<p>The paper models the angent interactively into a multi-step trajectory, not a single-wheel answer. In other words, the subject of the training is not so simple as the “final response text”, but rather a series of alternating sequences: reasoning, tool call, tool observation, re-advising, re-calling, re-capitulation. The model really needs to learn, it's a multi-step rhythm.</p>
<p>And that's why it specifically deals with tool operations in SFT targets. It was explicitly mentioned that the tool observation text would not be directly contributed to the end of the list, which would mean that the observation was treated as context rather than as a monitoring object. This is a good project detail: the SFT really wants to write into the model, how to make decisions based on the operation, not carry it down.</p>
<h3>2. Seed SFT: Pull the policy to an area that can be assessed first</h3>
<p>AMAP's SFT is not a complete, but a phased process. Phase one is... <code>seed SFT</code>I'm sorry. The way to get the papers is to pick one of the random ones from the front line of the program. <code>10% Tiny Dataset</code>, create ground-truth projects with a strong model, then select the best version with the best-of-8 + verifier, first turning the policy up into an initial policy.</p>
<p>The systemic significance of this step is clear: if policy is so weak that even a sample trajectory is difficult to produce, then the later “complication” “RL testier” is not operational. Seed SFT is tasked with pulling the model to a region that can self-expose boundaries.</p>
<h3>3. Capability programming: difficulty is not the static properties of the sample, but rather the relative properties of the policy</h3>
<p>AMAP's most interesting next step is its treatment of curriculum. The paper expressly rejected the interpretation of difficulty as a static label, because tasks that are easy for strong models may be out of circulation for the current policy; in turn, tasks that are too simple for teacher may still be the decision-making boundary that the student needs to learn.</p>
<p>So it's not the way to get offline to all samples in the can, but to get the current policy model to rollout. Query, system sampling <code>K=8</code> Track, and then the verifier gives the reward of each track, and then the experience value of this query is obtained <code>μ̂</code> Difference <code>σ̂²</code>。</p>
<p>In this step, the average and the difference play a very different role:</p>
<ul>
<li>Mean is "This task is largely inexplicable for the current policy"</li>
<li>The variance means "Current policy is unstable on this mission, has there been any learning but not yet learning signals?"</li>
</ul>
<p>That's why the paper is... <code>σ̂²</code> Almost like uncertify proxy. High-specified but non-zero-average values usually mean that models are not stable but are occasionally right; this is precisely the area of greatest training value.</p>
<h3>4. Signal-to-noise migration: Not all labeled data are worth learning</h3>
<p>Based on the above, AMAP divides the data into three categories:</p>
<ul>
<li><code>Trivial Region</code>: averages close to one, variance close to zero, which means the model has been met, and retracts are just overworked</li>
<li><code>Noise Region</code>: averages close to 0, variance close to 0, which means that the current policy is largely powerless, and forced learning is only highly skewed and even negative.</li>
<li><code>Learnable Region</code>: the variance is high and equal to zero, indicating that the task falls exactly within the decision-making boundaries of the current model Go, go, go!</li>
</ul>
<p>And then the paper was kept. <code>Learnable Region</code>I'm sorry. To quantify this level of learning, it also defined one. <code>Learnability Potential Score</code>The basic idea is to multiply non-zero solvency and instability by prioritizing the retention of samples that are neither tradially solved nor popular noise.</p>
<p>This step is important because it has truly upgraded data governance to training governance. From here on, the question is no longer just "How's this data OK?" It is "Does this data have a gradient value on the current policy?"</p>
<h3>5. Aerial projectory synthesis: Level 2 SFT is actually repairing the base</h3>
<p>After having lostnability score, AMAP will not end SFT immediately. It's going to be in step four, which is... <code>Adaptive Trajectory Synthesis</code>I'm sorry. The strong model is considered expensive, oracle, and the system will allocate more sampling budgets to higher quality query, at best. <code>Kmax = 8</code> Generates opportunities to restore a high-quality recovery path with greater probability.</p>
<p>Finally, these verified projects are assembled, and the system then decides the best mix and trains the RL follow-up backbone model.</p>
<p>In abstraction, this corresponds to the second level of SFT. It's not just a different role than a different one: Seed SFT is responsible for pulling models into a trainingable area; this second level is more like SFT<strong>Fix</strong>And, you know, you know, you know, you know, you know, you know, you know, you know, you know, you know, you know, you know, you know, you know, you know, you know, you know, you know, you know, you know, you know, you know, you know, you know, you know, you know, you know, you know, you know, you know, you know, you know, you know, you know, you know, you know, you know, you know, you know, you know, you know, you know, you know, you know, you know, you know, you know, you know, you know, you know, you know, you know, you know, you know, you know, you know, you know, you know, you know, you know, you know, you know, you know, you know, you know, you know, you know, you know, you know, you know, you know, you know, you know, you know, you know, you know, you know, you know, you know, you know, you know, you know, you know, you know, you,</p>
<h2>VI. ONline RL: When should the sample be handed over to RL</h2>
<h3>RL does not take over everything, but takes over the fronters questions</h3>
<p>And here, a lot of systems will start on RL. But good recide is not just about the data coming, but about the questions that should be left to the RL.</p>
<p>AMAP summed up the story in macro-species:<code>seed SFT -&gt; 再用更确定的样本稳住行为 -&gt; 把低 certainty 的 frontier tasks 留给 RL</code>I'm sorry. If it is read in conjunction with the text, it is clear that RL's role is not to replace all SFTs, but to eat those samples that have been screened by reward, verifeer and the first two rounds of SFTs, and that there are still difficult samples to explore space.</p>
<p>It's not like many people think RLs. RL is not here a universal calculator, but a module that deals with static SFT problems of long-range interaction, exploration, recovery, decision-making times and service links.</p>
<h3>RL for AMAP: Reword structure instead of re-build RM</h3>
<p>Another advantage of AMAP is that it does not suddenly switch sources of surveillance when entering RL. Online RL uses the same three-dimensional rubric, same dynamic weight, same rewrite. Offline screening tracks and online optimization share the same set of value interfaces.</p>
<p>This is critical, because many of the trainings that are most problematic are offline data and online optimization of two different things in school. AMAP avoids this break-up here: reward is the filter first, then the RL returns signal, but the definition of value itself is continuous.</p>
<h3>Algorithmic layer: GRPO is form, and what really matters is what it optimizes in what environment.</h3>
<p>AMAP, in the algorithm, <code>GRPO</code>and use in practice <code>GSPO</code> Come stabilize the training. The approach in the paper can be summarized in three things:</p>
<ul>
<li>A set of tracks for each query sample</li>
<li>Standardized in reward group</li>
<li>Update the reference with crip and KL bound control policy too far</li>
</ul>
<p>These formulas are of course important, but this article would like to highlight another thing more:<strong>Algorithms are not the focus of this paper, but the focus is on what kind of environment, what sort of data compartment RL is in, what sort of data compartment it is taking over.</strong> Without the previous data governance, environmental construction, verified projects and curriculum layers, moving GRPOs in isolation would not naturally grow a strong agent.</p>
<h3>What is RL really learning here?</h3>
<p>If you can sum up from the point of view of ability, RL from AMAP is learning four things:</p>
<ul>
<li>When should we continue to explore, not stop early?</li>
<li>When should the user condition be modified, not passively rejected</li>
<li>When do you need multiple tools to connect, not just to do local searches?</li>
<li>When should we organize the service chain of multiple steps ahead? Road</li>
</ul>
<p>Online RL is supplemented by angent's decision-making ability in the environment, not a simple text style or a template for answers.</p>
<h2>vii. How Benchmark limits training targets, and how benchmark is audited for himself</h2>
<p>By this point, the training closed ring has largely been shaped. But there is the last layer that is often overlooked: how do we know that we are really optimizing the right goal?</p>
<h3>1. MLE-bench, tau-bench, and PaperSmart, respectively, what are they bound by?</h3>
<p><code>MLE-bench</code> The constraint is "complex ML engineering workwork can be standardized by public benchmark". It shows that what is being evaluated is the engineering tasks that are defined together, not the single-wheeling ability.</p>
<p><code>tau-bench</code> What is bound by “the success of the deployment level” is what it means. It uses database end-of-life, consistency and <code>pass^k</code> Remind us that the deployment of angents should not be seeking success only once, but must be one of stability and discipline.</p>
<p><code>PaperBench</code> And the binding is "how does the judge work in an open mission." It reminds us that research on the recurrence of this open task cannot be directly assigned to a general comment, but rather to a general comment, to decompose the re-emergence into rubrics and to decompose the re-emergence into a different kind of process. <code>Code Development / Execution / Result Match</code>。</p>
<p>And that's why these papers will not be discussed in this paper on data governance on the same page as AMAP. The reason is not that they are not important, but that their main value is not to provide training formulas directly, but that they are not.<strong>Define what training should be for.</strong>。</p>
<h3>2. Benchmark #2: Revard is proxy, judge is proxy, benchmark is also proxy</h3>
<p>Finally,<code>Benchmark^2</code> This is a further layer of the problem. It asked not " which model is stronger," but "how is this benchmark actually doing?" The paper proposed three indicators:</p>
<ul>
<li><code>CBRC</code>: a benchmark given model sorting, whether it is broadly consistent with other benchmarks in the field</li>
<li><code>DS</code>: Is this a Benchmark really able to pull different models?</li>
<li><code>CAD</code>: Is there a frequent anomaly at the subject level that "the weaker model in the same family is right, the stronger model is wrong"</li>
</ul>
<p>The value of this framework is that it pulls the benchmark back from default truth to proxy. We have seen it repeatedly in previous articles: reward Mode is proxy, judge is proxy; and Benchmark #2 is just pushing this thing further, telling us that benchmark is also proxy.</p>
<p>That's why I put benchmark audit at the end of the whole line. The training closed not at that moment, but after the benchmark was audited. Otherwise you're probably just making false progress on a vulnerability gauge.</p>
<h2>This article is the last one to keep the judgment.</h2>
<p>If I'm allowed to leave only one word, I'll write:<strong>Real anent is not a six-part module, but a continuous system from data governance, environmental architecture, track screening to training for updating and evaluating audits.</strong></p>
<p>AMAP gave us the most complete training in the system: from 30 million history Query to 200,000-grade candidate Qory Pool through taxony, filtering, difficulty labeling and negative sample formation; then to verifier rating and personal-trajorry retention through strong models; then to verfield projects; then to seed SFT to bring the model into the environment, then to re-establish the curriculum with signature/noise pilot transfer, and finally to hand over the currtier procedures to online RL.<code>Agent K</code>、<code>tau-bench</code>、<code>PaperBench</code>、<code>MLE-bench</code> and <code>Benchmark^2</code> We are told that the chain is based on the environmentalization of the mission, the structuring of the target, the clear definition of the deployment criteria, and the auditing of Benchmark as proxy.</p>
<p>Do not interpret reward with training as two words. For real anent, they are more like the back and back half of the same chain: The first half is responsible for translating the target into learning signals, and the second half is responsible for writing back strategies for those signals. Only by reconnecting the whole chain will we understand the whole process of rewards and training in practice.</p>
<h2>References</h2>
<h3>Main source</h3>
<ul>
<li><a href="https://arxiv.org/abs/2512.24957">AMAP Agentic Planning Technical Report</a></li>
<li><a href="https://arxiv.org/abs/2411.03562">Kolb-Based Experiential Learning for Generalist Agents with Human-Level Kaggle Data Science Performance</a></li>
<li><a href="https://arxiv.org/abs/2410.07095">MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering</a></li>
<li><a href="https://arxiv.org/abs/2406.12045">tau-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains</a></li>
<li><a href="https://arxiv.org/abs/2504.01848">PaperBench: Evaluating AI&#39;s Ability to Replicate AI Research</a></li>
<li><a href="https://arxiv.org/abs/2601.03986">Benchmark^2: Systematic Evaluation of LLM Benchmarks</a></li>
</ul>
<h3>Extending reading</h3>
<ul>
<li><a href="https://openai.com/research/paperbench/">OpenAI PaperSmart Introduction Page</a></li>
<li><a href="https://arxiv.org/abs/2305.20050">Let&#39;s Verify Step by Step</a></li>
<li><a href="https://openreview.net/forum?id=21UFlJrmS2">Rubrics as Rewards</a></li>
<li><a href="https://arxiv.org/abs/2601.06487">ArenaRL</a></li>
</ul>
