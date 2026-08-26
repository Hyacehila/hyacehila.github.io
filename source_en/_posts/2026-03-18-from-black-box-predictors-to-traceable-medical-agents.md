---
title: 'From Black-Box Predictors to Traceable Medical Agents: The Future of Medical AI'
title_zh: 从黑盒预测器到可追溯医疗 Agent：医疗AI的未来
date: 2026-03-18 20:00:00 +0800
categories:
- Agent Systems
- Agent Evaluation & Governance
tags:
- Interpretability
- Multimodality
- Evaluation
author: Hyacehila
excerpt: A technical evolution map for medical AI, from black-box predictors toward traceable medical agents.
description: A technical evolution map for medical AI, from black-box predictors toward traceable medical agents.
excerpt_zh: 按技术演进梳理医疗 AI 如何从黑盒预测器走向可追溯医疗 Agent。
permalink: /blog/2026/03/18/from-black-box-predictors-to-traceable-medical-agents/
lang: en
translation_key: 2026-03-18-from-black-box-predictors-to-traceable-medical-agents
translation_status: machine
translation_source_hash: 8c46ed334b30a286902ed0311df0b835e63c63690d57559967c5cc5a3199bb05
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<p>If you give a familiar picture of the medical care of the last decade, AI, it's probably like this: you give the system a chest plate, an ECG, an EHR, a model that returns a probability.</p>
<p>The questions in this article can also be addressed<a href="/en/blog/2026/03/17/behavior-auditing-and-decoding-beginners-guide/">Behaviour Audit and Decoded Behaviour: From Reward to Agent Observation</a>、<a href="/en/blog/2026/06/06/feedback-driven-agentic-scientific-discovery/">Seeing from feedback loops how Agent turned generation into search</a>How the concept of a relatively close read together is developed in different contexts.</p>
<p>This route is not wrong. Instead, it was successful.<a href="https://www.nature.com/articles/srep26094">DeepPatient</a>、<a href="https://jamanetwork.com/journals/jama/fullarticle/2588763">Gelshan et al. sugar net screening system</a>、<a href="https://arxiv.org/abs/1711.05225">CheXNet</a> This type of work proves that, as long as the mission is clear, labelled and measured, in-depth learning can make a strong predictor of the medical submission.</p>
<p>But the medical scene is not just a little higher. The system also needs to be able to be clinically questioned, reviewed and taken over.</p>
<p>What doctors want is a system that can be asked: it can explain why a disease is placed ahead, it can point to what image regions, what test indicators, what documentary evidence are relied upon for conclusions, and it can admit uncertainty in case of a conflict of evidence.</p>
<p>Following this evolution, the next main line of medical AI is more like a searchable, reviewable, modifiable, traceable-evidence chain of medical Agent system.</p>
<h2>The boundaries of the Black Box Predictor Age</h2>
<p>First, we correct a common misconception: early medical in-depth learning is not the wrong path.</p>
<p>Yeah. <a href="https://www.nature.com/articles/srep26094">DeepPatient</a> This work, published on May 17, 2016, is about high-dimensional, thin, dirty clinical data like EHR, which can also be codified as useful indicators and then used for downstream disease prediction. It was published online on November 29, 2016.<a href="https://jamanetwork.com/journals/jama/fullarticle/2588763">Sugar net screening</a> It is clear that single-mission visual models can already approach a high level of practicality in a given image mission. Later, November 14, 2017.<a href="https://arxiv.org/abs/1711.05225">CheXNet</a> And then you're gonna push the + CNN+ line even further.</p>
<p>These systems are certainly strong, and the problem is that they level the medical complexities too well.</p>
<p>Enter a chart to output a disease probability; enter a wave shape to output an abnormal label; enter a medical file to output a risk fraction. This form is natural for benchmark and for retrospective story, because the evaluation is well targeted, labels are well defined and statistical indicators are clear.</p>
<p>But clinical reasoning is not long.</p>
<p>Diagnosis in the real world is often a continuous process of renewal: The initial assumptions are based on the principal complaint, followed by the context of the examination, the decision to conduct the examination and the reordering of the candidate diagnosis based on the image, test, pathology, genetic results. It is not a question-and-answer exercise, but rather a question-and-answer exercise, a clarification of the conflict.</p>
<p>And that's why, the question that followed was not just whether the model was smart enough, but whether we were always describing the diagnosis with too flat an interface.</p>
<h2>Around 2017, the problem was finally defined.</h2>
<p><a href="https://arxiv.org/abs/1712.09923">XAI</a>A systematic judgement was made on the value of AI in medical treatment. Its value is not to suggest which specific interpretative algorithms are available, but to distribute directly a problem that was often weakened before:<strong>Medical care cannot tolerate a high-profile, non-challenged system for long.</strong> The author was, of course, unable to think of AI Agent in 2017, and interpretability could also be changed from being an extenuating model to being a system, but in the same direction.</p>
<p>Once this matter is made clear, the evaluation criteria for the whole direction are beginning to change.</p>
<p>What you've been more concerned about in the past is that AUC is not much more sensitive, is it possible to overtake a doctor in a fixed set of tests. XAI, this line, turns the question into:</p>
<ul>
<li>What evidence does the system see?</li>
<li>Can this evidence be reviewed?</li>
<li>When the conclusions are wrong, can we know if they are misperception, wrong alignment, wrong reasoning?</li>
<li>When the system recommends the next inspection, is it doing a valid update or is it simply rereading the template of the usual recommendations?</li>
</ul>
<p>Here, one must be restrained: it is not to be interpreted as a white box, nor to be traced as a proof of cause or effect.</p>
<p>But what is needed in the medical landscape is not a mythical model of complete transparency, but a system that can externalize intermediate evidence and allow humans to cross-examine, review and audit. XAI did not directly turn Medical AI into Agent, but it prefaced the demand that the Age of Agent would not be able to get away with:<strong>The system does not just give answers, but leaves the process behind.</strong></p>
<h2>The multi-modular base model was changed not by input of quantity, but by the way the evidence went into reasoning.</h2>
<p>And along the lines of 2022 to 2026, people usually put it in a big medical model and started supporting polymorphism.</p>
<p>But I think more precisely, medical AI finally started to re-respect the original evidence itself.</p>
<p>The early system usually compresses each of the models first. Image models are first given to labels, ECG models are first given to rhythm categories, test indicators for thresholdisation, and then then hand over these intermediate results to another module for synthesis. This pipeline can certainly run, but it has two natural problems: the loss of information, the difficulty of tracking the wrong source.</p>
<p>The value of the MMA model is that it attempts to allow images, text, medical records, and partial time-series signals to enter the unified semantic space. Technically, it is projecting the results of visual or time series encoding into the space of representation available in the language model, allowing the model to continue to study the original evidence during the generation process, rather than simply staring at a summary that has been put on the table.</p>
<p>That's why it looks like it. <a href="https://arxiv.org/abs/2204.09817">BioViL</a>、<a href="https://arxiv.org/abs/2306.00890">LLaVA-Med</a>、<a href="https://arxiv.org/abs/2307.14334">Med-PaLM M</a>、<a href="https://arxiv.org/abs/2405.03162">Advancing Multimodal Medical Capabilities of Gemini</a> and <a href="https://huggingface.co/google/medgemma-1.5-4b-it">MedGemma</a> This line will become more important. They are not just making models talk, but are building a new medical interface: The original pattern no longer serves only a one-time classification, but it has begun to participate in subsequent interpretations, questions and answers, reasoning and the call for tools.</p>
<p>This step seems to be just a model upgrade, laying the undersea for the medical agent.</p>
<p>Because the system cannot accept discrete labels only as long as it wants to work as a doctor in the future. It must be able to deal with the checks between images, reports, pathologies, vital signs, laboratory forms, pathology and genetic results. The MMA model is precisely about reconnecting these evidence to the linguistic space and the decision-making space.</p>
<p>Of course, it's not too full here either.</p>
<p>Not all medical models are equally easily linguisticized. The chest and pathological images are relatively easy to match with reports and descriptions; changes in ECG, dynamic vital signs, pathologies and vertical tests are much more difficult, as the real key information is often distributed in time axes, leadership relationships, context changes, rather than a static area. So I do not want to describe the multiplicity as a unified world that has been completed. By March 2026, we just saw the bottoms begin to form, and the whole family is not yet in place.</p>
<h2>A good time line.</h2>
<table>
<thead>
<tr>
<th>Date</th>
<th>Nodes</th>
<th>It really changed something.</th>
</tr>
</thead>
<tbody><tr>
<td>2008-10-23</td>
<td><a href="https://pubmed.ncbi.nlm.nih.gov/18950739/">HPO</a></td>
<td>The genetic disease pattern is converted into a standard language that is calculable, searchable and compatible.</td>
</tr>
<tr>
<td>2015-11-12</td>
<td><a href="https://www.nature.com/articles/gim2015137">Exomiser</a></td>
<td>Connect the phenotype with the variant langing to create a reusable gene diagnostic tool chain.</td>
</tr>
<tr>
<td>2016-05-17</td>
<td><a href="https://www.nature.com/articles/srep26094">DeepPatient</a></td>
<td>Representing the Black Box Predictor Age: EHR can be directly translated into risk prediction.</td>
</tr>
<tr>
<td>2017-12-28</td>
<td><a href="https://arxiv.org/abs/1712.09923">Medical XAI</a></td>
<td>The first time that the issue was explicitly named as a core issue in medical care, it was highly qualified but not subject to inquiry.</td>
</tr>
<tr>
<td>2022-04-21</td>
<td><a href="https://arxiv.org/abs/2204.09817">BioViL</a></td>
<td>Medical images and reports are being systematically aligned and multi-modular basic forms are being shaped.</td>
</tr>
<tr>
<td>2023-07-26</td>
<td><a href="https://arxiv.org/abs/2307.14334">Med-PaLM M</a></td>
<td>The emergence of generic medical polymodular models.</td>
</tr>
<tr>
<td>2024-05-05</td>
<td><a href="https://arxiv.org/abs/2405.02957">Agent Hospital</a></td>
<td>The subject of the study moved from a single output to a multi-role system in the hospital process.</td>
</tr>
<tr>
<td>2024-05-13</td>
<td><a href="https://arxiv.org/abs/2405.07960">AgentClinic</a></td>
<td>Static medicine questions and answers were noted as being overly optimistic, and sequenced clinical decision-making became a more rational unit of evaluation.</td>
</tr>
<tr>
<td>2025-04-09</td>
<td><a href="https://www.nature.com/articles/s41586-025-08866-7">The Nature version of AMIE</a></td>
<td>Dialogue diagnostic systems have become real bridges: the consultation itself has begun to be included in the definition of competence.</td>
</tr>
<tr>
<td>2025-09-24</td>
<td><a href="https://arxiv.org/abs/2509.20067">MACD</a></td>
<td>The multi-smart body is no longer a multi-person discussion, but begins to emphasize reusable experiences and synergetic processes.</td>
</tr>
<tr>
<td>2026-02-18</td>
<td><a href="https://www.nature.com/articles/s41586-025-10097-9">DeepRare</a></td>
<td>For the first time, the phenotype, genotype and literature were more fully organized into a retroactive rare disease system.</td>
</tr>
</tbody></table>
<h2>From answering questions to following the process?</h2>
<p>If the MMA model addresses “the evidence or not”, then the medical agent Agent addresses whether the system will work when the evidence comes in.</p>
<p>That's why I'm putting <a href="https://www.nature.com/articles/s41586-025-08866-7">AMIE</a> Look at it as a critical bridge. It is not a complete multi-intellectual body system, but it has already done the right thing: medical competence should not be understood as merely the ability to answer medical questions, but rather as a medical examination, medical history gathering, differentiated diagnosis, communication and recommendations for the next steps.</p>
<p>Once this step was set up, then much of the work from 2024 to 2025 was right.</p>
<p><a href="https://arxiv.org/abs/2405.02957">Agent Hospital</a> What is done is to simulate hospital processes so that the roles of “doctors, nurses, patients” form an environment in an interactive manner;<a href="https://arxiv.org/abs/2405.07960">AgentClinic</a> The project is to change the static database to a sequenced mission of consultation, examination, tool call and re-decision;<a href="https://aclanthology.org/2024.findings-acl.33.pdf">MedAgents</a> The MDT-style multidisciplinary discussion has been translated into a multi-role collaborative process;<a href="https://arxiv.org/abs/2509.20067">MACD</a> Further, emphasis was placed on how to settle reusable clinical knowledge between multiple intelligent bodies, rather than always arguing from the beginning.</p>
<p>These systems are really moving together, not so simple as multimodels, but three more things:</p>
<p>First,<strong>Clinical reasoning is orderly.</strong>I'm sorry. A system that does not ask, does not say that I have any key information, is far from a true diagnosis, that we can feed it all to AI once in the algorithmic research laboratory, and that we will not do all the tests in clinical terms.</p>
<p>Second,<strong>The evaluation must interact.</strong>I'm sorry. The static benchmark answers the same question, and it is not the same thing to do in an environment where information is incomplete, updated and tools are also to be adjusted. The available Benchmark technology still does not allow for a precise measurement of the complexity of clinical consultations themselves.</p>
<p>Thirdly,<strong>The value of multi-smart bodies is not in voting, but in the obvious division of labour and questioning.</strong>I'm sorry. Images, tests, pathology, genetics, guidance retrieval, and case retrieval are not the same kinds of work. Combining these functions into a one-size-fits-all model is not necessarily more reliable than separating them from each other.</p>
<p>In other words, the emergence of Medical Agent is actually a process of rewriting the smallest units of the diagnostic system from a model to a process.</p>
<h2>Why genomics became the key puzzle for the next phase.</h2>
<p>If you look at images and medical records only, the medical profile of Agent is quite clear. But what really keeps this route going is probably genomics.</p>
<p>Because complex clinical problems such as rare diagnosis are naturally not a single model topic.</p>
<p>It often requires at least three types of simultaneous presence.</p>
<p>The first is the phenotype, which is the clinical performance of the patient. The most critical infrastructure here is <a href="https://pubmed.ncbi.nlm.nih.gov/18950739/">HPO</a>I'm sorry. The value of HPO is not another database, but rather it transforms “stunting” “low-strength” “vision nervous atrophy” into a standardized language that machines can compare, aggregate, retrieve.</p>
<p>The second category is genotype, i.e. sequence results. For many technical readers, the word VCF, which is a little abstract when first appears, is essentially a “list of patients' genetic mutations”. The question is not whether the list exists, but what is usually too long, and what is more suspicious, what is more rational in genetic patterns, and which genes are more compatible with the current pattern.</p>
<p>It's like this time of year. <a href="https://www.nature.com/articles/gim2015137">Exomiser</a> Such tools are crucial. It can be interpreted as a rough way to read: to sort out the more priority-oriented group of forms of resemblance, mutate pathogenity, genetic patterns and the existing knowledge base.</p>
<p>But it's not enough to be sorted.</p>
<p>The hardest step in medicine is not to give me a top-1, but to tell me why you're in this row. And so is it. <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC7477017/">LIRICAL</a> It's a fun place. It's re-enumerating the scoring process implicit in many phenotype-driven diagnosis tools to achieve a closer clinical intuition in the holihood of the philihood: how much does each graph actually push forward with which diagnosis. This line of thinking and the retrospectable reasoning of 2026 is in fact highly homogeneous.</p>
<p>I'm not sure what I'm talking about.<a href="https://pubmed.ncbi.nlm.nih.gov/32434849/">AMELIE</a> It automates another highly intensive manual labor: it looks not only at genes and forms, but it also matches these candidates with the primary literature. In other words, it is already doing a very agent thing for clinical genetics: retrieving the literature from the outer edge of the diagnostic process to the diagnostic process itself.</p>
<p>The logic is clear when you see it here.</p>
<p>HPO is responsible for making the phenotype sound like the machine, VCF is bringing the genotype in, Exomiser is responsible for preliminary sequencing, LIRICAL is responsible for making the evidence more readable, Ameline is responsible for bringing the latest literature in. The genomics tools that have been seen to be scattered over the years have been preparing for the same future:<strong>Replace the complex diagnosis with “experts manual integration information” to “systematize phenotype, grootype and literature”.</strong></p>
<p>This is one of the most natural stages of medical care, Agent.</p>
<h2>DeepRare: A rare disease that is actually connected, Agent.</h2>
<p>As of March 2026, one of the best cases of this trend was published in Nature on February 18, 2026 <a href="https://www.nature.com/articles/s41586-025-10097-9">DeepRare</a>。</p>
<p>It's not just another rare disease model, but it's a very clear indication of the structural difference between the medical care of Agent and the traditional black box predictor.</p>
<p>From the project instinct, DeepRare looks like a medical MCP.</p>
<p>It does not allow a model to complete a diagnosis from scratch, but it separates the system into three layers: a central host to organize, plan and integrate; a group of specially edited antlers; a group of individually processed external evidence environments such as phenotype excise, disase nonmalization, knowledge search, case search, phenotype anallysis and genotype analis; the outermost layer connects PubMed, OIM, Orphanet, HPO, Crossref and the General Websearch.</p>
<p>The structure is very significant.</p>
<p>Because it is a recognition of the fact that the diagnosis of rare diseases is not a model of knowing or not about the disease, but rather of the system's ability to organize multiple evidence spaces.</p>
<p>DeepRare can be entered into a free text table, structured HPO, and original VCF generated by WES. Subsequently, genotype anallyser will call on existing tools like Exomiser to sort variable notes and priorities, and host recombines phenotype, variant, gene-disase link, inheritance pattern and aidature evolution to form a retrospective diagnostic link.</p>
<p>What is most noteworthy here is not the fractions, but the interfaces.</p>
<p>The traditional system usually gives you the most likely disease to end; DeepRare tries to give you, yes. <code>candidate diagnosis + reasoning chain + evidence links</code>I'm sorry. More importantly, it would do self-refective diagnosis: if the current assumptions were not valid, it would continue to deepen its search and analysis, rather than pretend that the first answer is enough.</p>
<p>And that's the kind of change I've been talking about: the smallest unit of the medical AI is going from an output to a process.</p>
<p>DeepRare was particularly suited as a anchor for the article because it brought together for the first time two seemingly independent technical lines.</p>
<p>One line is the evolution of the medical AI itself: from the black box predictor to the multimodular base model, to the interactive and multi-smart system.</p>
<p>Another line is the evolution of the professional tool chain that can be used by AI: from HPO to Exomiser, LIRICAL, Ameliolee, to the step-by-step transformation of phenotype, genotype and literature into calculable, sortable, interpretable and renewable objects.</p>
<p>DeepRare meant that the two lines finally converged on February 18, 2026.</p>
<p>Of course, it is not the end.</p>
<p>From the published information, many of the capacities of DeepRare are still built within the current tool chain ' s coverage boundaries. For example, it is very friendly to WES/VCF workflows, but the integration of structural variations, repeat enlargements, deep-inline subeffects, RNA layers of evidence, protein cluster evidence is far from being permanent; he supports HPO information and the incorporation of clinical symptoms, but lacks more information that is embedded in the DR, CT, MRI, ECG, etc. The paper also clearly revealed failure patterns such as phenotype mimic and evidence watching error. This is just one thing:<strong>Agent didn't just wipe out the mistake, but exposed it to a more analytical position.</strong></p>
<h2>The next stage is more like a platform than a single model.</h2>
<p>If this route continues, the next phase will be more like a new level of medical infrastructure than a one-size-fits-all model that attempts to cover all tasks.</p>
<p>It may look like this:</p>
<pre><code class="language-mermaid">graph TD
    A[&quot;Patient / EHR / Imaging / Labs / Genomics&quot;] --&gt; B[&quot;Host Agent&quot;]
    B --&gt; C[&quot;Imaging Agent&quot;]
    B --&gt; D[&quot;Genomics Agent&quot;]
    B --&gt; E[&quot;Guideline &amp; Literature Agent&quot;]
    B --&gt; F[&quot;Ordering / Tool Agent&quot;]
    B --&gt; G[&quot;Audit Agent&quot;]
    C --&gt; H[&quot;Evidence Board&quot;]
    D --&gt; H
    E --&gt; H
    F --&gt; H
    H --&gt; I[&quot;Clinician Review&quot;]
    G --&gt; I
    I --&gt; J[&quot;Diagnosis / Differential / Next-step Plan&quot;]
</code></pre>
<p>The key in this picture is not the number of angents, but the boundaries of duty. A more mature medical platform, Agent, will continuously handle cases, deploy specialized capabilities such as video, testing, genes, case retrieval and guidance retrieval, organize intermediate evidence into accessible evidence board, and turn points of disagreement, uncertainty and recommendations for the next step back to clinical practice.</p>
<p>From this perspective, the medical treatment doesn't look like a system to move doctors out of the way, but rather like a set of clinic co-pilots infrestrucure. It has partially liberated doctors from the labour of manual handling of information, repeated documentation and preparation of candidate diagnostics, but final judgement, responsibility and clinical decisions remain in human return.</p>
<p>And that's why the medical field is one of the most suitable environments for growing Agent. It is natural that there is a world of many models, tools, players, multiple rounds of renewal, and strong audit requirements. Many of these capabilities appear to be additional and complex in other industries, not in the medical sector, but in the most basic systems.</p>
<h2>The real boundaries that still need to be faced</h2>
<p>Optimism is optimistic, and so far this direction is far from ripening. There are at least five practical issues that no one can bypass.</p>
<p>First,<strong>There is still a deep gap between the simulated environment and real clinical.</strong>I'm sorry. Neither Agent Hospital nor Agent Clinic can be directly equated with real clinical returns as long as the primary evidence is also derived from simulated patients, constructed environments and offline benchmarks.</p>
<p>Second,<strong>The weight of evidence is still very easy to make mistakes.</strong>I'm sorry. The system may find a lot of relevant material, but finding it is not the same as “weighting right”. DeepRare exposed the question of resonating weighing error, which is essentially the problem.</p>
<p>Thirdly,<strong>The appearance of overlap and multi-modular conflict will not disappear naturally.</strong>I'm sorry. Many diseases are highly similar in their appearances and can easily be biased by text and symptoms alone; the inclusion of genetic information and video screening can alleviate, but can also introduce new conflicts and interpretations of challenges.</p>
<p>Fourth,<strong>The tool chain is a real problem.</strong>I'm sorry. Most systems appear to be becoming more complete today in terms of images, text and conventional genetic variations, but the search and callability in structural variations, vertical pathologies, real hospital information system interfaces, and privacy isolation settings are still far from adequate.</p>
<p>Fifthly,<strong>Audit and accountability boundaries are not yet truly institutionalized</strong>I'm sorry. The chain of retroactivity is important, but it is the subject of compliance, traceable, accountable engineering that can become part of the clinical workflow.</p>
<p>So, the future of Medical Age is not that the model can take over the diagnosis, but that we finally know what the next generation of systems should look like.</p>
<p>There is a long distance between thought and clinical use.</p>
<h2>Final judgment.</h2>
<p>Looking back at this route, Medical Agent is not a new concept that suddenly grew out of the big model. It is more like a few long-accumulated technical lines that finally converge at the same time: one line from black box predictors, multimodular base models and interactive clinical processes, and another line from HPO, Exomiser, LIRICAL, Amellie, a genomic tool chain that gradually transforms phenotype, genotype and literature into calculable objects.</p>
<p>From this perspective, the change is not just a few points higher on a model than on a medical benchmark, but rather a change in the interface of the diagnostic system itself. The system began to be able to retain original evidence, call external tools, split specialist duties, handle differences and leave the chain of reasoning and the basis for reference in the process. This is closer to real clinical than a single prediction.</p>
<p>DeepRare is important here not because it already represents the finale, but because it's specific enough. The phenotype, gentype and literature are no longer three scattered materials, but are organized by host, specialized entities and external evidence environment organizations as a diagnostic chain that can be asked, reviewed and continuously improved.</p>
<p>So what this article is saying is that instead of the medical agent is mature enough to take over the clinical function, the medical AI interface is moving from a one-off predictor to a searchable, collaborative, auditable clinical system.</p>
<h2>References</h2>
<ul>
<li>Blackbox Forecaster and XAI<ul>
<li><a href="https://www.nature.com/articles/srep26094">DeepPatient</a></li>
<li><a href="https://jamanetwork.com/journals/jama/fullarticle/2588763">Development and Validation of a Deep Learning Algorithm for Detection of Diabetic Retinopathy in Retinal Fundus Photographs</a></li>
<li><a href="https://arxiv.org/abs/1711.05225">CheXNet</a></li>
<li><a href="https://arxiv.org/abs/1712.09923">What do we need to build explainable AI systems for the medical domain?</a></li>
</ul>
</li>
<li>Multi-modular Foundation Model<ul>
<li><a href="https://arxiv.org/abs/2204.09817">BioViL</a></li>
<li><a href="https://arxiv.org/abs/2306.00890">LLaVA-Med</a></li>
<li><a href="https://arxiv.org/abs/2307.14334">Towards Generalist Biomedical AI (Med-PaLM M)</a></li>
<li><a href="https://arxiv.org/abs/2405.03162">Advancing Multimodal Medical Capabilities of Gemini</a></li>
<li><a href="https://huggingface.co/google/medgemma-1.5-4b-it">MedGemma 1.5 model card</a></li>
</ul>
</li>
<li>Interactive diagnosis and multi-intellectual body<ul>
<li><a href="https://www.nature.com/articles/s41586-025-08866-7">Towards conversational diagnostic artificial intelligence (AMIE)</a></li>
<li><a href="https://arxiv.org/abs/2405.02957">Agent Hospital</a></li>
<li><a href="https://arxiv.org/abs/2405.07960">AgentClinic</a></li>
<li><a href="https://aclanthology.org/2024.findings-acl.33.pdf">MedAgents</a></li>
<li><a href="https://arxiv.org/abs/2509.20067">MACD</a></li>
</ul>
</li>
<li>Genomics and rare diseases, Agent<ul>
<li><a href="https://pubmed.ncbi.nlm.nih.gov/18950739/">The Human Phenotype Ontology</a></li>
<li><a href="https://www.nature.com/articles/gim2015137">Exomiser</a></li>
<li><a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC7477017/">Interpretable Clinical Genomics with a Likelihood Ratio Paradigm</a></li>
<li><a href="https://pubmed.ncbi.nlm.nih.gov/32434849/">AMELIE</a></li>
<li><a href="https://www.nature.com/articles/s41586-025-10097-9">DeepRare</a></li>
</ul>
</li>
</ul>
