---
title: The Application of Statistics and Statistics of Application
title_zh: 统计学的应用与应用的统计学
date: 2026-02-14 12:00:00 +0800
categories:
- Work & Society
- Research Practice
tags:
- Statistical Thinking
- Research Methods
author: Hyacehila
excerpt: A discussion of why statistics drifted away from applied settings while pursuing theoretical completeness, and how
  data science and machine learning reconnect with learning from data.
description: A discussion of why statistics drifted away from applied settings while pursuing theoretical completeness, and
  how data science and machine learning reconnect with learning from data.
excerpt_zh: 讨论统计学在追求理论完备的过程中为什么逐渐远离应用现场，以及数据科学与机器学习如何在大数据时代重新接上“从数据中学习”的问题。
permalink: /blog/2026/02/14/application-of-statistics-and-applied-statistics/
lang: en
translation_key: 2026-02-14-application-of-statistics-and-applied-statistics
translation_status: machine
translation_source_hash: 35da98ff5cb492711ebbc5fe1c76aeca7c627cd8979d6ccc2e08b3a6d2cddd27
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<blockquote>
<p>&quot;Statistics represents the science of learning from data.&quot;<br>Statistical science should be about learning from data.</p>
</blockquote>
<p>The questions in this article can also be addressed<a href="/en/blog/2026/01/24/the-statistical-crisis-in-science/">The Statistical Crisis in Science</a>、<a href="/en/blog/2025/09/04/research-theory-and-practice/">Scientific theory and practical experience</a>How the concept of a relatively close read together is developed in different contexts.</p>
<p>Today, when big data and artificial intelligence, especially the generation AI, are sweeping the globe, we hear often the argument that statistics pass the test of the times. Is statistics outdated because of the inability to process big data?</p>
<p>It's a hypocritical proposition. Statistics never fear the size of data, making traditional statistics seem to be a drag in modern applications, not data.<strong>Volume</strong>And we're dealing with data.<strong>Attitude</strong>and<strong>Parameter</strong>。</p>
<p>It is intended to discuss why statistics are far removed from application sites on the path to mathematical rigour and how modern data science and machine learning can bring problems back into reality through a “data-driven” approach, focusing on two seemingly entangled concepts: “Application of statistics” and “Statistics of application”.</p>
<h2>Lost in the original: from solving problems to pursuing mathematics.</h2>
<h3>The “leveling” of mathematics and statistics</h3>
<p>The original aim of mathematics was to study the real world (e.g. measuring land, calculating astronomicals) but it gradually developed into a clear hierarchy:</p>
<ol>
<li><strong>Pure mathematician.</strong>: Study abstract structures and logic, pursuing the ultimate breakthrough of theory.</li>
<li><strong>Applied mathematician</strong>: Although in the name of “application”, it is often the highly abstract structure (e.g. the condensity of the PBDE numerical decomposition) that is studied rather than the immediate reality.</li>
<li><strong>Math Applicator</strong>Researchers in specific disciplines (physical, engineering, economic) who abstract the real world, form the theory of their own disciplines and use mathematical tools only at a very small point.</li>
</ol>
<p>Statistics also seem to be unsavory and embark on a similar path.</p>
<p>The purpose of statistical studies, at the beginning of their life (e.g. during Fisher), was to analyse both agricultural experimental data and biogenetic characteristics.<strong>Addressing specific application issues</strong>I'm sorry. As disciplines developed, statisticians began to build a more complete theoretical system. To make the system mathematically sound, fine assumptions are introduced and theoretical reasoning is becoming more complex and sophisticated.</p>
<p>Ultimately, statistics also form a tower structure similar to mathematics: the top-level statisticians no longer process specific data but research abstract distributive, progressive and optimal proof.<strong>The gradual alienation of applied statistics for statistical applications</strong>— That is, to find nails with hammers, to force reality to be fine-tuned to fit the perfect theoretical models.</p>
<p>The application-oriented disciplines gradually distance themselves from the application site; the corresponding student development is also moving away from the application problem and it is ultimately difficult to develop the talent to solve the application problem.</p>
<h3>Leo Breiman's Warning</h3>
<p>As early as 1994, Leo Breiman, a statistical giant at the University of California in Berkeley, had warned of deafening. In one of his speeches, he mentioned:</p>
<blockquote>
<p>&quot;Nor is any discipline as far away from theory and practice as statistics, and most statistical theories and statisticians deal with issues that are far apart, as if they were living in different worlds.&quot;</p>
</blockquote>
<p>Breiman has a keen point that statistics should not be aimed at being second-rate mathematicians, but should be nurtured.<strong>Experts collecting, analysing and drawing conclusions</strong>I'm sorry. If statisticians cannot find pleasure and solve problems in specific applications such as voice processing, astronomy, medicine, etc., then the identity crisis in statistics will never be solved.</p>
<h3>Late awakening: statistics at the crossroads</h3>
<p>Breiman's voice may have been too radical at the time, but 25 years later, it's become a consensus in the academic world.</p>
<p>In 2019, in a major report funded by the National Science Foundation of the United States, Statistics at a Crossroads, several leading statisticians, including Bin Yu, finally collectively acknowledged this. The report notes that statistics are at risk of being marginalized and that fundamental cultural changes are required.</p>
<p>The most striking point in the report is that<strong>Practice must be re-centre of statistical evaluation</strong>I'm sorry. There has long been a false dichotomy in the statistical community: “theory statistics” are considered to be a noble upper-level building, while “application statistics” are considered to be a secondary manual labour. This value leads to the alienation of the discipline: scholars have to prove their intelligence by publishing obscure mathematical theories in order to gain a permanent teaching career (Tenure), and lack interest in solving scientific or social problems.</p>
<p>This late report calls for the evaluation of a statistician standard that should no longer be the complexity of mathematical techniques but rather the solution to actual problems.<strong>Impact</strong>。</p>
<h2>The shackles of the big data age: difficult to verify assumptions</h2>
<p>When time comes to the twenty-first century, the eruption of big data has exposed the embarrassment of traditional statistics.</p>
<p><strong>Statistics can handle big data, but not data that no longer satisfy the assumptions.</strong></p>
<p>Traditional mathematical statistics are often based on a series of fine assumptions, none of which is more classic than:</p>
<ul>
<li><strong>IID Assumptions</strong>: Data are distributed independently and independently.</li>
<li><strong>Normality assumptions</strong>: The error items are subject to normal distribution.</li>
<li><strong>Linear assumptions</strong>: The relationship between variables is linear.</li>
</ul>
<p>In the age of small data, these assumptions are essential to our understanding of the world's crutches. However, in the age of big data, the sources of data became extremely complex, with exponential growth in dimensions, and the correlation between observations became complex.</p>
<p>And now, if we still insist,<strong>Let's assume, then extrapolate.</strong>The traditional paradigm reveals that data from the real world almost never fully satisfy the perfect assumptions that are derived from mathematics.</p>
<ul>
<li>When we use linear regression to force the integration of a highly non-linear complex system;</li>
<li>When we use p to test a data set that does not satisfy the normal distribution and has a large sample mass;</li>
</ul>
<p>We get one of them all the time.<strong>A precise wrong answer</strong>I'm sorry. With difficult assumptions to verify, statistical science has lost its promise of application in a truly complex world.</p>
<p>But applied statistics are not in reality. In many cases, statistical models can also describe some of the "causal and consequentials" with relationships, not because the models identify their own causal structures from the data, but because they are not.<strong>Causation is embedded in the researchers ' understanding of the problem, in their research design and in their selection of variables.</strong>I'm sorry. Often researchers do not “discover causes and consequences” from the data, but first judge what the field knowledge may affect, then use statistical models to estimate strength, comparative direction and quantify uncertainty. Strictly speaking, this is not a causal recognition that is entirely from the data, but it is the reality of many applied statistical studies today: models do not create causal explanations, but they are merely trying to paraphrased and quantified the causal understanding that the researchers have brought into the analysis.</p>
<h2>Applied statistics: data science and machine learning relay</h2>
<p>Since traditional theoretical statistics have touched the wall of application, who has taken the great thing of applied statistics?</p>
<p>Yes.<strong>Data Science or Data Mining<strong>and</strong>Machine Learning (Machine Learning)</strong>。</p>
<h3>The difference between the two: from Inference to Prevention</h3>
<p>People often ask: What is the difference between statistics and machine learning?
One of the excellent answers is:<strong>They can't all be around the same question: how can we learn from data?</strong></p>
<p>But the difference in their focus explains why machine learning is so eccentric in the age of big data:</p>
<ul>
<li>Statistics: Emphasis on statistical extrapolation (Inference). Focus on confidence interval, hypothetical tests, estimation of parameters. It attempts to explain the model (Explainability), by which it has to make strong assumptions about data distribution (e.g. logical regression).</li>
<li>Machine Learning: Emphasis on Forecasting. It treats the data generation mechanism as a black box, not a strong understanding of internal parameters, but a desire to use the data to create a data base. &#36;f(x)&#36; It's the best way to predict it. &#36;y&#36;。</li>
</ul>
<p>In high-dimensional, structural realities (e.g. image identification, referral systems), human beings simply cannot predict the right mathematical distribution. Machine learning has abandoned the quest for perfect form and assessed the good and bad of models by empirical means such as cross-checking (Cross-Validation) rather than relying on theoretical progressive normality.</p>
<p>But to date, there are still a number of statisticians and some social scientists who see “interpretation” as a higher and more scientific objective, while demeaning “predictation” as a less academic, engineering-oriented technical activity. This view is in itself untenable.<strong>Interpretation and prediction are not a science-non-science opposition, but are two equally important objectives in scientific research.</strong> Interpretation helps us understand mechanisms, organizational knowledge, theory; predictions help us to test whether models capture stable structures and help us measure whether a method is useful in unknown samples, future situations and real-life decisions. A model that can only explain but cannot demonstrate stability in new data is hardly truly mastery of the world; and a model that can make accurate predictions on a sustainable basis must not be easily described as “unscientific”.</p>
<p>This is what the first point of the post is:<strong>The approach must first solve the problem.</strong></p>
<h3>The rise of the fourth paradigm</h3>
<p>The data science was born precisely to cope with the explosion and the complexity of the sources of this data dimension. As a scientific researcher,<strong>Fourth paradigm (data-intensive scientific discoveries)</strong>Data science no longer relies on the a priori knowledge required for theoretical simulation or computational simulation, but is based directly on data and patterns.</p>
<p>It is not simply a confluence of disciplines, but a deep convergence of several types of capabilities:</p>
<ul>
<li><strong>Computer science</strong>Infrastructure is provided: from core database technology (storage and extraction) to cloud computing and distribution systems, the rapid leap in computing capacity makes the processing of big data possible.</li>
<li><strong>Statistics</strong>The blog provides an evolution of methodology:<ul>
<li>Computation statistics (e.g., MCMC, EM algorithms) replace some traditional resolution and provide a powerful support for the processing of complex statistical structures.</li>
<li>The exploratory data analysis (EDA) is re-energizing in the age of big data, helping us “sniff” information from the data ocean.</li>
<li>The combination of compression of perception and thinness processing in high-dimensional statistics provides a mathematical antidote to the “dimensional curse”.</li>
</ul>
</li>
<li><strong>Artificial intelligence</strong>Powerful tools are provided: from traditional models to evolutionary learning through modern machines, to the outbreak of in-depth learning, providing a powerful capability for automated characterization extraction and non-linear modelling.</li>
</ul>
<p>This confirms one point:<strong>While statistics have many tools to use to complete the system, the fundamental 0-1 breakthrough in statistics must have been the result of addressing major application problems.</strong> And neither Fisher nor today's data science was created to solve real problems, not to perfect the mathematical structure.</p>
<h3>Statistics and machine learning: The difference between return and return is a repeat of the past.</h3>
<p>Kiri Wagstaff is here. <em>Machine Learning That Matters</em> It says:</p>
<blockquote>
<p>&quot;Much of current machine learning (ML) research has lost its connection to problems of import to the larger world of science and society.&quot;</p>
</blockquote>
<blockquote>
<p>(Most of the machine learning research is now lost to the scientific and social communities. I'm not sure.</p>
</blockquote>
<p>That criticism sounds familiar. If we put that in the first sentence, &quot;Machine Learning&quot; Replace with &quot;Statistics&quot;And it could be seamlessly present in the 1980s, to criticize statisticians who were obsessed with progressive theory and ignored reality data. He seems to be in collusion with Leo Breiman's famous speech at the University of California in Berkeley in 1994.</p>
<p>History seems to be repeating itself at an unprecedented rate: when a field is pursuing simple ** indicators<strong>SOTA, I'm not.</strong>When the problem itself is repeated, it is the same as statistics of the year.</p>
<p>This is also a serious challenge for the AI sector:</p>
<ul>
<li>Engineering overwhelms theory: a great deal of research has focused on “finding” techniques, and the models of End-to-End have become more complex and the black box is becoming more powerful.</li>
<li>Theories lag behind practice: while deep learning sweeps the top lists, the mathematics behind them (e.g. interpretation of generalization, rudge-precision paradox) fall far behind. Academics often have difficulty answering the practical questions posed by industry, and even have the embarrassment of “industry is a leader in academia”.</li>
</ul>
<p>If the crisis in traditional statistics is<strong>The theory is out of line.</strong>And the crisis of modern machine learning is<strong>Reality has abandoned theory.</strong>And what's in common is that<strong>Research is out of touch with real practice</strong>。    </p>
<p>This brings us back to the substance of our discussions in both areas. What's the difference between statistics and machine learning?
In short:<strong>There is no difference in substance. They all focus on the same question — how can we learn from data?</strong></p>
<p>If one wishes to summarize their main differences at this time:</p>
<ul>
<li>Statistics: Statistical inferences (confidence interval, hypothetical tests, optimal estimates) focused on forms in low-dimensional issues, emphasizing<strong>Explanatory</strong>。</li>
<li>Machine learning: Focusing on predictive accuracy in high-dimensional issues, emphasizing<strong>Broadening capacity</strong>。</li>
</ul>
<p>Although the focus is different, the two areas are increasingly being integrated. The core of data science is not whether you're using t or deep nervous networks, but whether you really use these tools to solve one.<strong>Existing scientific or social problems</strong>。</p>
<h3>Direction of application of data science</h3>
<p>Then there is a discussion of how data science should be applied, broadly speaking in two directions:</p>
<ul>
<li>Continuing to serve academia: using data science to solve complex problems that are being addressed in the physical, biological, social sciences (this is what many “calculations X” are doing).</li>
<li>Turning to industry: addressing the last-end reality world applications. Here, we can further break down into two closely related but different-focused roles:<ul>
<li>Data Scientist, DS: Focus on extracting insights from data (Insights) to assist enterprises or organizations in scientific decision-making through analysis. They are closer to “consultants” and “discoverers”.</li>
<li>Machine Learning Engineer (Machine Learning Engineering, MLE): Focus on construction products. Their first task is to convert algorithms into engineering, landing and practical services.</li>
</ul>
</li>
</ul>
<p>In theory, MLE is part of a broad DS, but they represent two different forms of value excavated from data: one is<strong>To understand.</strong>One of them is...<strong>For action.</strong>。</p>
<p>In either direction, the core is designed to solve that real problem.</p>
<h2>Return to the fields: keys to the backyard</h2>
<p>Professor Terry Speed of the University of California at Berkeley once had a famous saying:</p>
<blockquote>
<p>&quot;Statistics should have been the subject of other disciplines, and I'm so interested in statistics, that it's like putting keys in the backyard of any discipline.&quot;</p>
</blockquote>
<p>This is perhaps the most important message to be borne in mind by all data workers, whether they call themselves statisticians or data scientists.</p>
<p>Statistics should not be a mathematical game in ivory towers, but rather a tool for solving practical problems. The core value of this, whether it be called applied statistics or data science, is whether we can use the data in our hands to find definitive answers to the problems of biology, economics, medicine and even social sciences.</p>
<p>When we put aside our belief in assumptions and embrace the true complexity of data, we can truly achieve the application of statistics that give new life to this ancient and fascinating discipline in the data age.</p>
