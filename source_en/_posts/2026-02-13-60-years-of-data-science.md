---
title: The 60th Year of Data Science
title_zh: 数据科学的第60年
date: 2026-02-13 12:00:00 +0800
categories:
- Data Science
- Statistical Thinking
tags:
- Statistical Thinking
- Research Methods
author: Hyacehila
excerpt: A review of 60 years of data science, from John Tukey's prediction to David Donoho's broader view, discussing how
  statistics, computer science, and machine learning converged.
description: A review of 60 years of data science, from John Tukey's prediction to David Donoho's broader view, discussing
  how statistics, computer science, and machine learning converged.
excerpt_zh: 回顾数据科学60年的发展历程，从 John Tukey 的预言到 David Donoho 的广义数据科学。讨论统计学、计算机科学与机器学习如何合流，以及数据科学为什么需要同时处理推断、计算和现实问题。
permalink: /blog/2026/02/13/60-years-of-data-science/
lang: en
translation_key: 2026-02-13-60-years-of-data-science
translation_status: machine
translation_source_hash: b86df8f8d6df3eb627d1719ee68d8c404272316af3ebe40199b919ec3f8b9d8a
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<p>It's a big topic. The main idea of this paper is from David Donoho's 50 Years of Data Science, which also adds something I find interesting. Nearly 10 years after Donoho published this paper, the wave of artificial intelligence generation is transforming society as a whole, and data science is no exception.</p>
<p>The questions in this article can also be addressed<a href="/en/blog/2026/01/10/statistics-and-truth/">Statistics and Truth: How to use the accident (Statistics and Truth)</a>、<a href="/en/blog/2026/01/14/anscombes-quartet/">Anscom Quartet: Visualized power and statistical illusion</a>How the concept of a relatively close read together is developed in different contexts.</p>
<blockquote>
<p>The following elements were taken into account in the preparation of this document, and the rest of the paper will not add any emphasis on the sources of content:</p>
<ul>
<li><a href="https://www.tandfonline.com/doi/full/10.1080/10618600.2017.1384734">50 Years of Data Science</a> (David Donoho)</li>
<li><a href="https://projecteuclid.org/journals/statistical-science/volume-18/issue-3/A-Conversation-with-John-W-Tukey-and-Elizabeth-Tukey/10.1214/ss/1076102422.full">A Conversation with John W. Tukey and Elizabeth Tukey</a></li>
<li><a href="https://cosx.org/2021/08/a-century-in-statistical-science/">C. Radhakrishna Rao: A Century in Statistical Science</a></li>
<li><a href="https://arxiv.org/abs/2012.00174">What are the most important statistical ideas of the past 50 years?</a> (Andrew Gelman &amp; Aki Vehtari)</li>
<li><a href="https://cdss.berkeley.edu/about/history">Berkeley Data Science Planning</a></li>
<li><a href="https://magazine.amstat.org/blog/2019/04/01/statistics-at-a-crossroads-who-is-for-the-challenge/">Statistics at a Crossroads: Who Is for the Challenge?</a></li>
</ul>
</blockquote>
<h2>Before the text begins</h2>
<h3>From the beginning of the development of human science,</h3>
<p>Science is the tool of humanity to understand and interpret the world. Looking back at the evolution of human science, we can see the evolutionary context from concrete to abstract, and from theory to calculation.</p>
<p>The origins of mathematics in ancient China can be summarized in the nine chapters of the book, which are studying us.<strong>How to address some of the specific problems in the real world without addressing the general theory behind them</strong>♪ That can be considered ♪<strong>First paradigm: experiments and observations</strong>I'm sorry. Natural phenomena are recorded and described, and lessons learned from them.</p>
<p>The widely known scholar in physics, Newton, gave the laws of sport and gravity to transform the laws of movement in the real world.<strong>Precision, abstract theory.</strong>I'm sorry. At this stage, scientific research focuses on the theoretically abstracting of the universal formula and theorem, which is known as<strong>II: Theory evolution</strong>。</p>
<p>In the development of modern science, with advances in computer technology,<strong>Simulation</strong>Be a common method. In the face of theoretical models that are too complex to solve by deciphering, computers are used to simulate (e.g. limited meta-analysis) based on known physical patterns. Relying on simulation research and developing science, or scientific computation, makes up<strong>3rd Parameter: Calculator Simulation</strong>。</p>
<h3>The fourth paradigm of scientific research</h3>
<p>Data science defined as after experimental observations, theoretical evolutions, computational simulations<strong>Fourth data-driven scientific research paradigm</strong>It's an inspiring idea.</p>
<p>Now we have more problems: we don't know the theoretical mechanisms, we are constrained by physical conditions that do not allow experimental observations, we have too large a calculation or we have no parameters that make it impossible to rely on computational simulations. In this “uninformed” dilemma, the blogger says:<strong>Data driver</strong>Another path is provided.</p>
<p>The first three paradigms can be called together.<strong>Knowledge paradigm</strong>They are based on a certain a priori knowledge of the problem (experience, theory, equation). Where knowledge is sufficient, they are effective and accurate. However, in the absence of knowledge and incomprehensible mechanisms, the fourth paradigm is based directly on data, discovering patterns, digging linkages and adding to the hard-to-covered aspects of the knowledge paradigm. Data science is concerned with how to build testable and reusable judgements when mechanisms are incomplete.</p>
<h3>Definition of data science</h3>
<p>As an emerging discipline, data science is difficult to sum up accurately in a few sentences.</p>
<p>Some scholars simply define data science as a discipline used to process data and as a tool. This may be an accurate summary of what we do with it, but whether it's a matter of thinking about where it is.</p>
<p>Based on the above discussion of the scientific paradigm, I think<strong>Fourth Parameter</strong>It is an effective perspective for defining data science:</p>
<blockquote>
<p><strong>Data science is the discipline that implements the fourth paradigm of scientific research — the data-driven paradigm of scientific research; it is a discipline that works to make data useful.</strong></p>
</blockquote>
<p>This definition emphasizes two points:</p>
<ol>
<li><strong>Changes in methodological layers</strong>It adds to the problem that traditional scientific research paradigms are difficult to cover.</li>
<li><strong>High Applicability</strong>: Data science should not remain theoretical,<strong>It should be a highly applied discipline.</strong>I'm sorry. We have tried to make the data useful, with the goal of bringing it into the process of awareness, decision-making and intervention in the real world.</li>
</ol>
<p>In this context, discussions could continue on the content of data science, its sources of statistics and computer science, and how it has evolved to this day.</p>
<h2>Data science and statistics</h2>
<h3>A hundred years of statistics.</h3>
<blockquote>
<p>In the final analysis, all knowledge is history; in abstraction, all science is mathematics; in a rational world, all judgments are statistics.
The #C.R. Rao</p>
</blockquote>
<p>This statement by Professor C.R. Rao summarizes the place of statistics in the human knowledge system. As a recent statistical conglomerate, Rao experienced statistics from early descriptive analysis (Pearson era) to modern extrapolation theory (Fisher age) to the full 100-year history of today ' s integration with the depth of computing science. His work includes the Cramer-Rao heterogeneity and the Rao-Blackwell Theorem, which also shows how statistics can extract stable information from uncertain data through rigorous mathematical tools.</p>
<p><strong>The core of statistics is “inferment”.</strong> It deals not only with data collection and collation, but also with how to predict the total from the sample, how to quantify uncertainty, and how to find signals in noise. In the past 100 years, statistics have provided universal language for natural sciences, social sciences and engineering technologies, allowing for clearer judgements in randomity.</p>
<p>With the advent of the big data age, statistics have also come under new pressure. Traditional statistics tend to rely on strict assumptions about data distribution (e.g., independence and distribution, normality), while large data in the real world tend to be high, heterogeneity and dynamic. As Rao considered in late years, statistics needed to embrace computing, embrace more complex models and re-examine what “inferment” meant in the new data environment.</p>
<p>In the picture of data science, statistics are not outdated because of the emergence of big data.<strong>It provides the extrapolation skeleton in data science.</strong>I'm sorry. Only by combining statistical inferences with modern computer computations can we understand the patterns behind the data.</p>
<h3>Why data science?</h3>
<blockquote>
<p>“Data scientists” means professionals who use scientific methods to extract and create meaning from raw data.</p>
</blockquote>
<p>For statisticians, doesn't that sound like the job of applied statisticians? <strong>“Statistics” refers to practice or science that collects and analyses a large amount of data.</strong> </p>
<p>For statisticians, this definition of statistics seems to cover most of the definition of data scientists, but it appears to be limited. The DSI programme is therefore incomprehensible: statisticians believe that their daily work throughout their careers is being packaged as new by managers.</p>
<p>Several perspectives on data science and its relationship to statistics are discussed below.</p>
<h4>Is big data the core cause?</h4>
<p>A widespread voice is that data science has emerged because of Big Data: data are so big that statistics cannot handle them that a new discipline is needed. This view, while popular, cannot be easily refined.</p>
<p>Statistics never fear large-scale data, and their history encompasses theories and practices for processing complex, big data. If only because of the increase in data, we could develop “big data statistics” without having to build another stove. In fact, this view is more derived from misconceptions among non-statistical professionals; the emphasis on the increase in data volumes alone does not explain why a new entity, “data science”, is needed.</p>
<h4>Realistic drivers: skills gaps and the “golden fever” of talent</h4>
<p>Data science is driven largely by industry successes over the past decade. The use of data by technology giants such as Google and Amazon has generated significant commercial returns and has allowed businesses to begin to systematically compete for talent.</p>
<p>In this heat, businesses have discovered an awkward reality: traditional discipline education does not provide the much needed talent.</p>
<ul>
<li><strong>Graduated from traditional statistics Fan.</strong>Premise extrapolation and analysis, but often lack the capacity to process large-scale databases, prepare production-level codes and build complex software systems.</li>
<li><strong>Graduate of Computer Science</strong>The sophistication of engineering and systems is often poorly trained in statistical thinking, such as extracting signals from noise, quantifying uncertainties, etc.</li>
</ul>
<p>Mike Barlow is here. <em>The Culture of Big Data</em> It was noted that this skill gap led to a thirst for “data scientists”. This new title is a yes.<strong>Capacity integration</strong>High demand: a qualified data scientist must be able to think as closely as a statistician, and to process dirty data and build systems as software engineers. The demand for this mix of skills is a watershed in the job market between “data science” and traditional statistics.</p>
<h4>To true science.</h4>
<p>However, if data science is only designed to fill the recruitment gap for commercial companies, it is at best a vocational training orientation rather than a science.</p>
<p>The establishment of the “data science” entity should not be limited to commercial recruitment. As mentioned earlier, the fourth paradigm, we need a door.<strong>Science on learning from data</strong>I'm sorry. This scientific legacy of rigorous statistical inferences, while embracing modern computing capabilities, is used to address the problems of theoretical evolution and computational simulations that are difficult to grasp.</p>
<p>This vision is also the direction in which many statisticians have been building a foundation for the past 50 years.</p>
<p>And now,<strong>Math, statistics, computer science, artificial intelligence (mechanical learning) theory integration</strong>The evolving data science will be an effective tool. Statistics provide assumptions-based extrapolations, computer science provides high performance query and computing tools, and machine learning techniques provide methods for complex data modelling.</p>
<blockquote>
<p><strong>Data science is a highly applied discipline that we use to understand and manipulate the world.</strong>  </p>
</blockquote>
<h3>The Future of Data Analysis</h3>
<h4>1962: The birth of the prophecy</h4>
<p>John Tukey, in The Future of Data Analysis (1962), predicted many of today's data science issues more than 50 years ago. Tukey, to be blunt, said he was disturbed by his status as a mathematical statistician; by observing the development of mathematical statistics, he realized that his interest was in the development of a digital system. <strong>Data analysis</strong>。</p>
<p>The blogger says:<strong>Data analysis is a science.</strong>And not a branch of math. Mathematics seeks logical consistency and probability, while data analysis has three main elements of science:</p>
<ol>
<li><strong>Intellectual Content</strong></li>
<li><strong>Understandable forms of organization</strong></li>
<li><strong>Relying on the empirical test as the ultimate criterion for effectiveness</strong></li>
</ol>
<p>In Tukey ' s vision, the Statistical Form Theory is only a fraction, not all, of this new science. He listed four main factors driving this new scientific development: statistical theory, and the following:<strong>Computation capacity</strong>, Big Data challenges and quantitative trends across disciplines. This 1962 list, which is in today ' s data science discussion, still has many real problems.</p>
<p>For Tukey, whether it optimizes the trajectory of the Nike anti-aircraft missile at Bell Laboratory or analyses the flow data of U-2 aircraft, he always looks from the practical point of view, looking for answers in an empirical way, rather than the assumptions in superstition textbooks. The blogger says:<strong>If a method is not even used in practice, then it's pointless to test whether it's worth it.</strong>。</p>
<h4>From Tukey to Cleveland</h4>
<p>Despite Tukey ' s arms shivers, academic statistics continued to react in a cold manner over the following decades, continuing to be obsessed with purely theoretical evidence. However, Tukey's Bell lab colleagues and a few visionary scholars have taken over the torch and continue to march in the wilderness.</p>
<ul>
<li><strong>John Chambers</strong>In 1993, the S Language Developer called for the establishment of “Grey Statistics”, warning of the risk of marginalization if statistics do not embrace the concept of inclusiveness of learning from data.</li>
<li><strong>Jeff Wu</strong> In his inaugural speech in 1997, the speaker directly introduced “Statistics = Data Science” and advocated renaming statistics as Data Science.</li>
<li><strong>William S. Cleveland</strong> The famous Data Science: An Action Plan was published in 2001 and a road map for action was developed for this discipline. He proposed to allocate academic resources to six areas: interdisciplinary research, modelling methods,<strong>Data Computation</strong>(c) Teaching methods, tools assessment and theory. In addition to theory, five other areas were virtually blank in the traditional statistical faculties of the time.</li>
</ul>
<h4>Counting environmental victories</h4>
<p>Industry and practitioners have defined the future by code when definitions are still being debated by the academic community. The evolution of the computing environment changed the game rules from the early SSS/SAS to the S language developed by John Chambers, to the later R language.</p>
<p><strong>Scripts became a new age's paper.</strong>I'm sorry. It is an accurate and abstract description of the calculation steps. When the quantitative programming environment like R is popular, data analysis is no longer on paper, but rather becomes<strong>Recoverable, shared, verifiable</strong>- The practice. People can directly run others ' codes, validate methods on different data, and improve analytical processes through performance measures.</p>
<p>So far, Tukey about<strong>Data analysis is a science.</strong>The judgement was finally put into practice through the code and the computational environment.</p>
<h2>Forecast-led statistical modelling</h2>
<h3>Two cultures: Generate vs projections</h3>
<p>Leo Breiman published a sensational article in The State of Science in 2001. <em>Statistical Modeling: The Two Cultures</em>I'm sorry. He noted that there were two distinct statistical modelling cultures in the process from data to conclusions:</p>
<ol>
<li><strong>Data Modeling Culture</strong>: Assumes that the data are generated by a known random process (e.g. linear regression model). The task of statisticians is to extrapolate the parameters of the model. Breiman believes that this represents 98 per cent of the academic statistical community.</li>
<li><strong>Algorithmic Modelling Culture</strong>: Treating the data generation mechanism as an unknown and complex "blackbox" does not attempt to decipher its internal mechanisms, but focuses on finding input through algorithms &#36;x&#36; and Output &#36;y&#36; to achieve the preciseest possible<strong>Projections</strong>。</li>
</ol>
<p>The statistical community has long relied heavily on data models, leading to a disconnect between theory and reality. The existing algorithm models, while lacking a critical theoretical underpinning (as in early statistics), have developed rapidly in computer science and industry with the ability to address complex realities. That explains it.<strong>Data Science</strong>Why does it arise: it contains both traditional statistical assumptions and embraces a predictive-centric algorithm model.</p>
<h3>The secret of a culture of prediction: a framework for common tasks (CTF)</h3>
<p>If culture of prediction is an important pillar of data science,** the common mission framework (Common Task Framework, CTF)** is one of the mechanisms through which it can make sustainable progress.</p>
<p>The computational linguist Mark Liberman argues that the CTF is an important driver of machine learning and predicting the success of modelling, but is often overlooked by the mainstream statistical community. A typical CTF has three elements:</p>
<ol>
<li><strong>Open training data sets</strong>: Contains features and labels.</li>
<li><strong>Competing</strong>: Committed to training the best predictive rules.</li>
<li><strong>The adjudicative system</strong>: An objective and automatic assessment of the accuracy of the projections is conducted using the “black box” test set.</li>
</ol>
<p>From Netflix Challenge to Kagle Competition to modern deep learning revolutions like ImageNet, the CTF paradigm has been repeated. It turns the original blurry of research (such as early machine translation) into a new one.<strong>Quantifiable, comparable, re-emergible</strong>The engineering challenge.</p>
<p><strong>Minimize the predicted error + CTF paradigm = the perfect for the performance of experience.</strong></p>
<p>This model not only filters valid algorithms, but also changes talent needs. In the framework of the CTF,<strong>Information technology (IT) skills</strong>(Processing data, constructing systems, preparing scripts) becomes more direct than purely mathematical extrapolation. This explains why today's education in data science must include a great deal of computer science: in a new world dominated by prediction, code capability is as important as statistical thinking.</p>
<h2>Data science now</h2>
<p>As the discipline developed, early controversy over the relationship between data science and statistics gradually subsided. A pragmatic consensus has been reached in practice between academia and industry, first and foremost in the curriculum of higher education.</p>
<h3>Consensus in education</h3>
<p>The system of data science courses at higher education institutions such as the University of California at Berkeley (UC Berkeley) reveals a deep convergence of statistical and computer science skills. This integration is no longer limited to the discussion of the subject ' s attribution, but focuses on the development of practical competencies:</p>
<ol>
<li><strong>Basic status of computing capability</strong>• Unlike traditional statistical education, the curriculum for modern data science considers programming to be a core skill. Students need to master languages such as Python or R for production-level code writing, version control (e.g. Git) and large-scale data processing, not just mathematical extrapolation.</li>
<li><strong>The predictions and extrapolations are weighed equally</strong>: “Application Machine Learning” is a key feature of the curriculum. This marks a shift in the focus of education from a single parameter statistical inference to a parallel approach to processing high-dimensional data and achieving accurate predictions.</li>
<li><strong>Integration of data projects</strong>• Data storage, retrieval and management are integrated into the core curriculum system, complementing the knowledge gap in traditional statistics on the principles of databases and distributed systems.</li>
</ol>
<p>As Donoho has pointed out, there is a consensus that data scientists should be software engineers with statistical thinking and statisticians with expertise in software engineering.</p>
<h3>2019: Statistics at the crossroads</h3>
<p>This integration is reflected not only in education but also in the reflection of the academic community on its own positioning. In 2019, the National Science Foundation of the United States (NSF) funded a landmark report, Statistics at a Crossroads: Who Is for the Challenge?</p>
<p>This report, written by top statisticians like Xuming He, David Madigan, Bin Yu, echoes Tukey's predictions half a century ago. The report states categorically that statistics are at a crossroads: Without fundamental reform, the discipline risks being marginalized.</p>
<p>The core call of the report coincides with the idea of data science:</p>
<ol>
<li><strong>Centre of practice</strong>Statistics must return to the essence of “learning from data”, theoretical research must not exist for the sake of mathematical perfection alone, but for the service of “Better Practice”.</li>
<li><strong>The Revolution of the Evaluation System</strong>The tendency of academia to over-reward the publication of purely theoretical papers for a long time must be changed. The report clearly recommends that<strong>Interdisciplinary cooperation, software code development, data cleansing and collation</strong>It should be seen as an academic contribution of equal importance to the publication of papers.</li>
</ol>
<p>This marks the official recognition of the mainstream statistical community: In addition to extrapolating formulas, writing codes and processing data are an integral part of scientific research.</p>
<h3>Broad Data Science</h3>
<p>To define this discipline more comprehensively, Donoho proposes a framework** for a broad data science (Greater Data Science, GDS)**. The framework divides the scope of data science into six dimensions, noting the limitations of the current focus of academic research and the breadth of the needs for practical application.</p>
<ol>
<li><strong>GDS1: Data collection, preparation and exploration</strong>: Data scientists often devote a significant amount of time to data cleansing and collation in practice. Although this process is often overlooked in traditional academic research, it is the foundation for ensuring data quality and analytical validity.</li>
<li><strong>GDS2: Data expression and conversion</strong>: involves transforming unstructured information (e.g. text, images, audio) and structured data (SQL/NOSQL) into mathematical forms suitable for modelling.</li>
<li><strong>GDS3: Data-based calculations</strong>: Covers programming language proficiency, replicability of analytical stream construction and use of high performance computing resources. In this framework, code scripts that document the full analysis process have the same scientific value as academic papers.</li>
<li><strong>GDS4: Visualization and presentation</strong>: Not limited to the generation of static charts, but also includes the use of modern graphic syntax (e.g. ggpllot2) and interactive tools to discover data patterns and effectively deliver information to audiences.</li>
<li><strong>GDS5: Data modelling</strong>: Traditional, generating models and modern predictive models. This is the area of greatest concentration of current academic research, but it is only part of the broader data science.</li>
<li><strong>GDS6: Science of data science (Science about Data Science)</strong>: This is the more forward-looking dimension of the framework. It advocates the use of scientific methods to study the data analysis process itself and to assess the effectiveness and deviation of different analytical methods.</li>
</ol>
<h3>Core challenge: Science on data science</h3>
<p>Currently, the GDS6 dimension study is responding to a major challenge for the scientific community -<strong>Recoverable Crisis</strong>I'm sorry. As the volume of scientific literature has grown, it has become crucial to validate the findings of the study. Data science works by:</p>
<ul>
<li><strong>Meta-Anallysis</strong>Systematically integrates data from existing literature, assesses the overall effect of a given scientific issue and identifies publication deviations.</li>
<li><strong>Cross-workstream analysis</strong>: Study the impact of different analytical pathways (e.g. choice of data pre-processing methods, differences in model assumptions) on final conclusions, quantify the uncertainty associated with “researcher freedom” and thus seek more robust scientific findings.</li>
</ul>
<p>This trend indicates that data science is moving from empirical practice to a more rigorous scientific framework, which is dedicated to assessing and enhancing the validity of scientific findings from data analysis.</p>
<h3>Reflections: Data-driven limitations</h3>
<p>The case of the long-term capital management company (LTCM) provides a lesson on overdependence on data and models. Despite the existence of the best quantitative models, the impact of low-probability extreme events (i.e., the “black swan” event) was ignored by over-reliance on historical statistics, which eventually led to a systemic collapse.</p>
<p>This reveals the potential risk of purely data-driven thinking: it tends to focus on relevance rather than causality. Models based on historical data training may be ineffective once an unknown structural change occurs in the bottom mechanism for data generation.<strong>Pure data-driven methods are intrinsically difficult to predict black swan events that never appear in historical data.</strong>I'm sorry. This suggests that data science cannot be divorced from the understanding of the knowledge of the field and the mechanisms of causality.</p>
<h2>Future of data science</h2>
<p>Looking ahead, the development of data science should not be seen as a mere tool for industry to improve efficiency. As an independent discipline, the goal is to create<strong>Science of Learning from Data</strong>。</p>
<h3>Knowledge at the heart</h3>
<p>Future data science will go beyond the simple superseding of statistical and computer science to focus on its cognitive issues:<strong>How can the extrapolation from the data be ensured is real, reliable and re-emergible?</strong></p>
<p>Developments in this area will be directed in three directions:</p>
<ol>
<li><strong>Integration of the two cultures</strong>: Breiman proposes a further integration of the “generated model” (explaining mechanism) and “predictional model” (precision prediction). The development of an Explanatory Machine Learning (Ex plainable AI) is the manifestation of this convergence trend, which aims to obtain high predictive precision and model interpretability at the same time.</li>
<li><strong>Evidence-based methodology</strong>: The development of scientific methods for data will be guided by empirical principles. The performance of different algorithms and analytical processes is objectively assessed through the Common Task Framework, CTF and large-scale empirical research, rather than relying solely on theoretical assumptions.</li>
<li><strong>Scientific effectiveness safeguards</strong>• In the data-driven research paradigm, data science will assume responsibility for setting academic standards. To ensure that scientific findings are firmly grounded by developing “sciences on data science” and by establishing rigorous certification systems to remove noise and deviations.</li>
</ol>
<p>As Donoho said:</p>
<blockquote>
<p>“The broad data science is essentially dedicated to understanding and enhancing the validity of research findings and can play a key role in all areas dominated by data analysis modelling.”</p>
</blockquote>
<p>In the future, data science will serve as the basic methodological basis for modern scientific research, providing not only computing tools but also a critical thinking framework that will guide researchers from relevance to causality and more reliable conclusions.</p>
