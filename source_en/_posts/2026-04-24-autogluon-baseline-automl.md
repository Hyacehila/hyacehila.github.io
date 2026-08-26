---
title: 'AutoGluon: Simplifying Machine Learning Baselines to a Few Lines of Code'
title_zh: AutoGluon：把机器学习 Baseline 简化到几行代码
date: 2026-04-24 21:45:00 +0800
categories:
- Machine Learning
- Classical Machine Learning
tags:
- Baselines
- Tabular Data
- Workflow
- Software Engineering
author: Hyacehila
hidden: true
excerpt: AutoGluon does more than reduce code; it turns strong baseline methodology into a unified, comparable, and portable
  machine-learning workflow.
description: AutoGluon does more than reduce code; it turns strong baseline methodology into a unified, comparable, and portable
  machine-learning workflow.
excerpt_zh: AutoGluon 不只是在减少代码量，它把强 baseline 方法论固化成统一、可比较、可迁移的机器学习工作流。
permalink: /blog/2026/04/24/autogluon-baseline-automl/
lang: en
translation_key: 2026-04-24-autogluon-baseline-automl
translation_status: machine
translation_source_hash: c012e03de3bc363f579904e1452c85db94e8ad7fbc2d2310617fbb157b6e8008
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<p>In many machine learning projects, the real time consumed is not the training model itself, but a whole series of engineering frictions before and after the training: field type recognition, missing value processing, category code, feature screening, model selection, cross-checking, results recording, reasoning speed assessment, error sample looking back...</p>
<p>The questions in this article can also be addressed<a href="/en/blog/2026/01/22/tree-based-models-tabular-data/">The table still contains SOTA: XGBoost, LightGBM and CatBoost</a>、<a href="/en/blog/2024/03/28/machine-learning-introduction-supervised-learning/">Introduction to Machine Learning: Monitoring Learning and the Bayesian Approach</a>How the concept of a relatively close read together is developed in different contexts.</p>
<p>These are certainly important, but they should not start from scratch in every new data set. Especially in the early stages of the project, what we usually need is not a perfectly deployed production model, but a sufficiently reliable performance anchor: How high can this data set be? Is the current quality of data worth continuing input? How much marginal gains can be made by artificial profiling?</p>
<p>That's the most useful place I can understand AutoGluon: It's not just helping you to lower the number of lines, it's just fixing a set of strong baseline methods into a default workflow. You give it a table of data and a list of targets, which automatically finish a lot of dirty work and you use a set of comparable model results to tell you how much machine learning caps can be built on current data.</p>
<p><img src="https://raw.githubusercontent.com/Innixma/autogluon-doc-utils/main/docs/cheatsheets/stable/autogluon-cheat-sheet.jpeg" alt="AutoGluon, official Cheat Sheet"></p>
<p><em>The following is a list of the most recent examples of the events in the country:<a href="https://auto.gluon.ai/stable/cheatsheet.html">AutoGluon Cheat Sheet</a>。</em></p>
<h2>The real cost of Baseline</h2>
<p>The traditional machine learns the baseline that usually:</p>
<ol>
<li>Use <code>pandas</code> (a) Cleaning of data, type of repair, missing and abnormal;</li>
<li>Use <code>sklearn.pipeline</code> (a) The logic of processing the adhesive numerical characteristics, category characteristics and text characteristics;</li>
<li>The first is the Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost, etc.</li>
<li>Search for hyperparameters using GridSearch, RandomSearch or Optuna;</li>
<li>(b) Cross-check to confirm stability of results;</li>
<li>Additional training time, reasoning time, model size and validation scores are recorded.</li>
</ol>
<p>The process is not amateur, the problem is that it is too easy to turn early exploration into engineering muddy. You spent two days putting up a decent pipeline, and it took you two days to find that the data itself was not signaled enough, or that the definition of operational indicators was not even right. At this point, the precision of handwritten paperline was not translated into project proceeds.</p>
<p>The idea of AutoGluon is more like the idea of making a product of the case first. It defaults to make you a good enough automatic processing and model combination, and gives you a reference point that is difficult to easily exceed, and then decides whether it is worth continuing with a heavier workforce.</p>
<h2>AutoGluon's Core Design Thought</h2>
<p>I think it's important to understand AutoGluon, not from a parameter, but from the design choices behind it.</p>
<p><strong>Unified abstraction: Predictor as mission entrance.</strong></p>
<p>Many of AutoGluon's modules are organized around similar workflows:</p>
<pre><code class="language-python">predictor.fit(train_data)
predictions = predictor.predict(test_data)
predictor.evaluate(test_data)
predictor.leaderboard(test_data)
</code></pre>
<p>The benefits of this interface are not simply simple. It organizes the different machine learning missions into several stable actions: training, forecasting, evaluation, comparison.</p>
<p>For users, this means that you do not have to over-care the bottom model family, feature processing details and validation processes at an early stage of the project. You run with a single interface, get a baseline of results, and decide whether to drill.</p>
<p><strong>Automation priority: Repetition projects built into the framework.</strong></p>
<p>AutoGluon automatically identifies field types, handles missing values, class characteristics, numerical features and partial text features, and selects the appropriate model collection according to the task. This is particularly important for table data, as real business data are often not a clean matrix but rather a combination of integers, floating point numbers, categories, dates, text, ID, missing values and various odd codes of DataFrame.</p>
<p>It doesn't mean we can ignore the data. AutoGluon is better placed to turn default project treatment to the frame, and to free human attention to more manual questions: is the label reliable? Is the signature leaking? Is the training set and the distribution on line consistent? Are operational indicators correctly defined?</p>
<p><strong>Ensemble-first: Re-integrated, light manual.</strong></p>
<p>Many AutoML tools focus on the narrative of search: looking for the best model in algorithms and hyperparametric space. AutoGluon's philosophy is more biased towards anesimble-first: instead of a monomer model, it is better to train a set of complementary models, which are combined through bgging, staking and weighted ensemble.</p>
<p><img src="https://docs.aws.amazon.com/images/sagemaker/latest/dg/images/autogluon_tabular_illustration.png" alt="AutoGluon Tabular working mechanism Figure"></p>
<p><em>Figure: AutoGluon-Tabular Multi-Story Stacking/ Ensemble working mechanism, source:<a href="https://docs.aws.amazon.com/sagemaker/latest/dg/autogluon-tabular-HowItWorks.html">AWS SageMaker AutoGluon-Tabular Document</a>。</em></p>
<p>This is one of the reasons AutoGluon often gives strong baseline quickly. It does not rely on speculation as to which model should be used this time, but rather on competition and collaboration among different models within a unified certification framework. The costs are clear: integrated models are usually larger, the reasoning chain is longer and the interpretation may be less clear than a single model.</p>
<p><strong>Leaderboard-first: Results must be comparable.</strong></p>
<p>AutoGluon <code>leaderboard()</code> It is important because it transforms the training process from a black box score to a comparable trial sheet. You can see the validation scores for each model, test scores, training hours, reasoning times and stock level.</p>
<p>This table is not just a "rank." It answers the question of engineering decision-making:</p>
<ul>
<li>If only points are sought, which model should be chosen?</li>
<li>If delay in reasoning is more important, can a fraction be sacrificed for a faster model?</li>
<li>Does the gain cost extra after adding bagging/staking?</li>
<li>Is there a model that has good training scores but tests for unstable performance?</li>
</ul>
<p>In other words, AutoGluon training model, while helping you to make your experimental books.
<img src="https://quickchart.io/chart/render/zf-dc974a3f-fca0-4b55-b34b-936e51724672" alt="AutoGluon League score and reasoning time off Figure"></p>
<p><em>Figure: A diagram based on the leaderboard field of the AutoGluon official Tabular curriculum. The crossaxis is... <code>pred_time_test</code>♪ The vertical axis is <code>score_test</code>, the trade-off between the "highest score model" and "faster model" can be seen directly. References:<a href="https://auto.gluon.ai/stable/tutorials/tabular/tabular-indepth.html">AutoGluon Tabular In Depth</a>。</em></p>
<p><strong>Baseline-first: Create performance anchors first.</strong></p>
<p>My favorite use of AutoGluon was to use it as an early performance anchor for the project, not to allow it to do all the modelling work for me.</p>
<p>When AutoGluon gave a strong baseline in a short time, the follow-up discussion became clearer:</p>
<ul>
<li>If the manual model is much worse than it, it indicates that the processing of the pipeline or features may be problematic;</li>
<li>If manual models are somewhat better but are much more complex, the value of the benefits needs to be assessed;</li>
<li>If AutoGluon also performed poorly, the problem may not be in the model, but in the label, feature, sample volume or task definition;</li>
<li>If AutoGluon performed well but with too slow reasoning, it could consider distilling, refit or retaining a lightweight monomer model.</li>
</ul>
<h2>Module architecture: What does AutoGluon really cover?</h2>
<p>AutoGluon has now been extended beyond the Table AutoML. More precisely, it's a set of around-the-clocks. <code>Predictor</code> In abstractally organized automechanical learning modules: upper layers to carry training, prediction, assessment and comparison of results with similar interfaces; intermediate layers to select competencies such as Tabular, Time Series or MultiModal according to task type; bottom-level reassembly feature processing, model libraries, integrated strategies and leaderboard records.</p>
<p>The advantage of this structure is that users do not need to re-learning a completely different engineering paradigm for each mission. The bottom models for tables, time series and multi-modular tasks vary widely, but they are all packaged in AutoGluon as much as possible as “for data, target setting, training, comparison, iterative” workflows.</p>
<p><strong>Tabular: The classic strong baseline scene.</strong></p>
<p><code>autogluon.tabular</code> It's the classic and most reflective of design philosophy of AutoGluon. It is directed towards the classification, regression and sorting of tables and is directly acceptable <code>pandas.DataFrame</code>, autoprocess features and train multiple models.</p>
<p>In many business scenarios, table data remain the most common data pattern: user portraits, transaction records, questionnaire data, operational indicators, wind control features, experimental data, structured logs. The value of AutoGluon Tabular is that it can quickly transform these data into a comparable model baseline.</p>
<p>The smallest example is probably as follows:</p>
<pre><code class="language-python">from autogluon.tabular import TabularDataset, TabularPredictor

train_data = TabularDataset(
    &quot;https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv&quot;
)
test_data = TabularDataset(
    &quot;https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv&quot;
)

label = &quot;class&quot;
predictor = TabularPredictor(label=label, eval_metric=&quot;accuracy&quot;).fit(
    train_data,
    time_limit=300,
    presets=&quot;medium_quality&quot;,
)

predictor.evaluate(test_data)
predictor.leaderboard(test_data)
</code></pre>
<p>If you want to spend more time in training for more robust performance, you can try:</p>
<pre><code class="language-python">predictor = TabularPredictor(label=label, eval_metric=&quot;accuracy&quot;).fit(
    train_data,
    time_limit=1800,
    presets=&quot;best_quality&quot;,
)
</code></pre>
<p>But the key here is not to remember. <code>presets</code> Parameters, but rather understanding the trade-offs behind them: a stronger configuration usually implies more models, more complex integration, longer training hours and higher reasoning costs.</p>
<p><strong>Time Series: Integration of forecasting into the unified workflow.</strong></p>
<p><code>autogluon.timeseries</code> (c) Time series projections. It continues the working methods of Predictor, evaluate, and leaderboard, but the mission objective is becoming forecasting, focusing on historical sequences, predictive windows, compost variables and probabilities.</p>
<p>This means that you can use relatively uniform mental models to address another common type of problem: sales forecasting, flow forecasting, inventory forecasting, indicator trend forecasting, etc. AutoGluon Time Series continues to aim for a strong and comparable baseline, compared to manual integration of traditional statistical models, deep time series models and backtracking processes.</p>
<p><strong>MultiModal: a uniform entry for text, images, tables.</strong></p>
<p><code>autogluon.multimodal</code> For more complex data patterns: text, images, table fields can appear in a task simultaneously. It covers categories, regressions, semantic matching, target testing, embedding extraction, etc.</p>
<img src="https://automl-mm-bench.s3-accelerate.amazonaws.com/cheatsheet/stable/automm.jpeg" alt="AutoGluon MultiModal Official Cheat Sheet" style="max-width:680px;width:100%;height:auto;display:block;margin:18px auto;">

<p><em>The following is a list of the most recent cases of violence against women:<a href="https://auto.gluon.ai/stable/cheatsheet.html">AutoGluon Cheat Sheet</a>。</em></p>
<p>The significance of this module is that many of the actual data are not a single model. For example, commodity data may have both titles, descriptions, prices, headings and pictures; curriculum vitae screening may have structured fields and long text; and qualitative data may have sensor tables and images simultaneously. MultiModalPredictor tried to package these blends into a uniform training process.</p>
<p><strong>Features: the base level for automated feature processing.</strong></p>
<p><code>autogluon.features</code> More like a support layer. It is responsible for the ability to automatically deduce, characterize metadata, feature generation and conversion. Although ordinary users do not necessarily use it directly, it explains why AutoGluon can receive relatively original data sheets without asking you to write a complete pre-processing file first.</p>
<p>Of course, automatic characterization is not magic. ID leakage, time travel, target code leakage, training and online fields are inconsistent and the framework for these issues cannot be fully judged for you. AutoGluon can reduce the sample work, but it cannot replace the data audit.</p>
<p><strong>Cloud / SageMaker: From local experiments to hosting processes.</strong></p>
<p>AutoGluon also has an integrated ecological set with AWS / SageMaker for hosting training, modeling and cloudwork streams. For individual experiments or small projects, local runaways are usually sufficient; for teams and production environments, cloud-end integration is valuable in resource management, replicability training and deployment links.</p>
<p>This does not expand the SageMaker operation details, as this will lead the theme to the cloud platform tutorial. All that needs to be learned is that AutoGluon is not designed for rapid experiments in Notebook, but it can enter a more complete engineering system and is more naturally integrated with the AWS / SageMaker ecology.</p>
<h2>The difference between AutoGluon and other tools: and the AutoML imagination of the Age of Age</h2>
<p>AutoGluon won't hang all the tools. More precisely, different tools serve different stages and constraints.</p>
<table>
<thead>
<tr>
<th>Tools / Routes</th>
<th>What do you do better than that?</th>
<th>Difference between AutoGluon</th>
</tr>
</thead>
<tbody><tr>
<td>Handwritten sklearn / XGBoost / LightGBM</td>
<td>Controllable, light, easily embedded production</td>
<td>Need to process characterization, authentication, referral and experimental management</td>
</tr>
<tr>
<td>auto-sklearn / TPOT</td>
<td>Search AutoML, algorithm selection, pageline search</td>
<td>More emphasis on the best search, AutoGluon more emphasis on integration and strong defaults</td>
</tr>
<tr>
<td>H2O AutoML</td>
<td>Enterprise-level platform, visualization, governance and deployment of ecology</td>
<td>The platform has more complete capabilities, but experience varies from light-script to hard-script experience</td>
</tr>
<tr>
<td>PyTorch / TensorFlow</td>
<td>High-defined models, end-to-end depth learning studies</td>
<td>Flexibility, but the table baseline tends to be more costly</td>
</tr>
<tr>
<td>AutoGluon</td>
<td>Fast, smooth, comparable, baseline</td>
<td>Models may be heavier, reasoning slower, interpretation and deployment control more often</td>
</tr>
</tbody></table>
<p>If your goal is to make a production service that is strictly manageable, very slow and relies on a single model document, the final solution is not necessarily an integrated model for AutoGluon. But if your goal is to answer quickly in the early hours of the project, "Is this data valuable, and what models are likely to do?" AutoGluon is very appropriate.</p>
<p>However, if the time scale is slightly increased, AutoML’s ecology may be rewritten by the language model and Agent.</p>
<p>The former AutoML is more like a searcher: given data, tasks and indicators, which search models, feature processing and hyperparameters in pre-defined pipeline spaces. AutoGluon is more like a strong default workflow: instead of being obsessed with finding a single-body solution, it quickly builds a strong baseline with a set of stick defaults and integrated models.</p>
<p>But the new variable that language models and Agent bring is that they begin to have the ability to read the context and organize experiments. An Agent can first look at data schema, meaning of fields, missing patterns and target variables, then decide that table models, time series models, multi-modular models should be tried, and even automatically write clean codes, run experiments, observe leaderboard, modify the pipeline. In other words, the fact that a strong model is being found by data type is moving from traditional AutoML search problems to end-to-end data science workflow issues where Agent can participate.</p>
<p>This is not a mere imagination. In the last two years, a lot of work has been going in this direction: OpenAI. <a href="https://arxiv.org/abs/2410.07095">MLE-bench</a> Evaluate engineering with Kagle competitions, Agent;<a href="https://arxiv.org/abs/2310.03302">MLAgentBench</a> Concerned about the planning, coding and iterative capabilities of LLM Agent in machine learning experiments;<a href="https://arxiv.org/abs/2402.18679">Data Interpreter</a> Try to get LLM Agent to do the data science task automatically; and... <a href="https://arxiv.org/abs/2410.02958">AutoML-Agent</a> Such work directly introduces multi-intellectual body thinking into automatic machine learning processes.</p>
<p>How does this affect AutoGluon's framework? My judgment is that they are not simply replaced by Agent, but rather could be an Agent tool layer.</p>
<p>The reason is simple. Agent is good at understanding tasks, dismantling steps, writing glue codes and depending on feedback, but it still needs a stable, accessible, comparable bottom-up tool. AutoGluon provides the right capability for a unified Predator interface, automated feature processing, strong baseline, modelboard, model preservation and reuse. Instead of handwritten sklern pipeline, a data science-oriented Agent should give priority to the AutoGluon baseline and decide on the next step based on the results: whether to do data cleansing, feature auditing, model distillation or to use more specialized models.</p>
<p>In this perspective, AutoGluon's value will not disappear, but its role will change: It is not necessarily the final interface that users face directly, but it may be the most trustworthy baseline engine behind Agent.</p>
<h2>Get the John Baseline and what should we do?</h2>
<p>AutoGluon gave strong baseline after the real work started. This baseline should not be seen as the end point, but as a ruler: It helps us judge whether the current data quality, the definition of tasks and the engineering inputs are worth continuing.</p>
<p>The first step is not usually to continue to engage, but to perform error analysis and data auditing. What is wrong first: which are the categories that are far worse? Which samples have high confidence but are wrong to predict? Are errors concentrated on certain time periods, regions, user groups or data sources? If the wrong pattern is clear, the maximum-return action is often not a model change, but rather a labeling, recharging, job dismantling or a redefinition of operational indicators.</p>
<p>Meanwhile, strong baseline can be bad news sometimes. The abnormally high scores may mean that the target leak, time travel or training/testing cut-off is not in line with real business processes. For example, a feature is not actually available at the time of prediction, or data is mixed into the test set and highly duplicated in the training sample. AutoGluon can quickly give high points, but it cannot automatically prove that they are credible; data leak checks remain a human responsibility.</p>
<p>And then it's the project.<code>leaderboard</code> It helps you choose between accuracy, training and reasoning. In many cases, the highest score model may not be the best model to go online; a model with a slightly lower score but much faster reasoning and simpler structure may be a better engineering solution. If the complete integrated model is too large and slow, consideration may be given to retaining the better performing monomer model or to converting AutoML products into a programme that meets operational constraints using distillation, refit, model durability and reasoning optimization.</p>
<p>And finally, Baseline should go into a long, iterative loop. When the model is online, it can serve as a reference for subsequent editions: is the new feature really effective? Is the new model stable more than the old model? Has the distribution of online data deviated from training data? AutoGluon addresses the rapid establishment of a credible starting point, not a permanent alternative model life cycle management.</p>
<h2>Summary</h2>
<p>AutoGluon is a good place to build a strong baseline, but it also has a clear cost:</p>
<ul>
<li>Integrated models may occupy more disks and memory;</li>
<li>The data are not available.</li>
<li>Automatic characterization processes reduce the volume of sample code and may also result in some of the details being less transparent;</li>
<li>Multiple model dependence increases deployment and version management complexity;</li>
<li>Handwritten pipeline may eventually be required for strong operational constraints, causal explanations or extremely low delayed scenes.</li>
</ul>
<p>So I'm going to put AutoGluon in the early high-value position of the machine learning tool chain: first, to build a strong baseline, then to decide whether to do manual profiling, custom model, light quantitative deployment or stricter production governance.</p>
<p>It allows us to move faster through the early mudslides of model selection, characterization and experiment comparison, and to focus our attention back on more important issues: The reliability of the data, the correct definition of the mandate, the operational relevance of the indicator and the value of the model being truly deployed.</p>
<h2>Extending reading</h2>
<ul>
<li><a href="https://auto.gluon.ai/stable/index.html">AutoGluon Official Document</a></li>
<li><a href="https://auto.gluon.ai/stable/tutorials/tabular/index.html">AutoGluon Tabular</a></li>
<li><a href="https://auto.gluon.ai/stable/tutorials/timeseries/index.html">AutoGluon Time Series</a></li>
<li><a href="https://auto.gluon.ai/stable/tutorials/multimodal/index.html">AutoGluon Multimediadal</a></li>
<li><a href="https://docs.aws.amazon.com/sagemaker/latest/dg/autogluon-tabular-HowItWorks.html">AWS：How AutoGluon-Tabular works</a></li>
<li><a href="https://arxiv.org/abs/2003.06505">AutoGluon-Tabular Paper</a></li>
<li><a href="https://arxiv.org/abs/2410.07095">OpenAI MLE-bench</a></li>
<li><a href="https://arxiv.org/abs/2310.03302">MLAgentBench</a></li>
<li><a href="https://arxiv.org/abs/2402.18679">Data Interpreter</a></li>
<li><a href="https://arxiv.org/abs/2410.02958">AutoML-Agent</a></li>
</ul>
