---
title: 'Interpretable Machine Learning: Model Explanations, SHAP, and Counterfactual Methods'
title_zh: 可解释机器学习：模型解释、SHAP 与反事实方法
date: 2024-05-23 23:00:06 +0800
categories:
- Machine Learning
- Classical Machine Learning
tags:
- Machine Learning
- Interpretability
- SHAP
author: Hyacehila
mathjax: true
hidden: true
excerpt: Covers interpretability concepts, interpretable and model-agnostic methods, example-based explanations, counterfactuals,
  influence functions, and a focused entry point for SHAP.
description: Covers interpretability concepts, interpretable and model-agnostic methods, example-based explanations, counterfactuals,
  influence functions, and a focused entry point for SHAP.
excerpt_zh: 整理可解释性概念、可解释模型、模型无关解释方法、基于样本的解释、反事实解释和影响函数，并提供 SHAP 专题入口。
permalink: /blog/2024/05/23/interpretable-machine-learning-notes/
lang: en
translation_key: 2024-05-23-interpretable-machine-learning-notes
translation_status: machine
translation_source_hash: c9af67bdc14d3fdd499a83f98e0bd65d528f05af895af2c873a64aff18551015
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<h2>Between the beginning.</h2>
<h3>Introduction</h3>
<p>Machine learning has great potential for improving products, processes and research. But computers are often unable to explain their predictions, which is an obstacle to learning by machine. And that's what we're trying to solve here.</p>
<p>We'll deal with the basic presentation, then we'll discuss interpretative models, generic methods, and finally a sample-based interpretation.</p>
<p>Our focus is on the mechanical learning model of table format data (also known as relational or structured data), which basically does not address computer visual and natural language processing tasks. It's only about monitoring learning.</p>
<p>This is not about the latest studies that explain machine learning, but about the more mature methods.</p>
<p>Finally, we would like to stress one point: <strong>Multimodel integration has now become a necessary weapon for major machine learning competitions, and most of the winning models are model integration or very complex models, such as upgrading trees or deep neural networks, which actually reduces the model ' s interpretability significantly.</strong></p>
<h3>Terminology</h3>
<p>In order to avoid ambiguity, some agreed terms are given here.</p>
<ul>
<li><strong>Algorithm</strong> It's a set of rules that machines follow to achieve a specific goal.</li>
<li><strong>Machine Learning (Machine Learning)</strong> A methodology that allows computers to learn from data to make and improve projections</li>
<li><strong>Learner (Learner)</strong> or Machine Learning Algorinthm is a program used to learn machine learning models from data.</li>
<li><strong>Machine Learning Model (Machine Learning Model)</strong> It is a learning program that maps input to prediction, which can be a linear model or a group of weights of a neural network. Model (Model) can also be called "Prognator" (Predictor), based on a task that can be divided into "Classifier" or "Regression Model"</li>
<li><strong>Blackbox Model (Black Box Model)</strong> It's a system that doesn't reveal its internal mechanisms.<strong>White Box (white Box)</strong>In this book, it's called an interpretable model. The model's unconnected interpretation treats machine learning models as black boxes.</li>
<li><strong>Interpretable Machine Learning</strong> Means the methods and models that enable the machine to learn the behaviour and predictions of humans.</li>
<li><strong>Dataset (Dataset)</strong> It's a table containing data from which the machine has to learn.</li>
<li><strong>Example</strong> Show as a line in the data set</li>
<li><strong>Characteristics (Features)</strong> is used to predict or classify inputs. Characteristics as columns in data sets</li>
<li><strong>Target (Target)</strong> It's the machine that has to learn to predict.</li>
<li><strong>Machine Learning Job (Machine Learning Task)</strong> A combination of data sets with characteristics and targets</li>
<li><strong>Forecast</strong> It's a machine learning model based on a given feature.</li>
</ul>
<h3>Dataset</h3>
<p>Here we present the data sets that will emerge from the following examples. There are no examples to explain how machine learning is hard to understand. There are many examples that will help us understand how.</p>
<p>We will use different data sets for different tasks: classification, regression and text classification.</p>
<p>Data files already have project folders with original books on Github Medium </p>
<h4>Bicycling (return)</h4>
<p>This data set is derived from the UCI Machine Learning Database, which is publicly provided by Capital-Bikeshare with daily counting of bicycle rentals, Fanee-T and Gama adding weather data and seasonal information</p>
<p>We don't use all the data set features, but the features we use are basically covered.</p>
<ul>
<li>Number of bicycle rentals</li>
<li>Various dates, such as seasons, holidays, calendar days, working days</li>
<li>Various weather markers, such as temperature, humidity, weather (close rain and snow), wind speed</li>
</ul>
<h4>YouTube garbage review (text classification)</h4>
<p>For example, we use 1956 comments from five different YouTube videos, using YouTube API</p>
<p>These comments are manually marked as garbage comments or regular comments. The trash comment is code "1" and the normal comment is code "0".</p>
<h4>Risk factors for cervical cancer (classification)</h4>
<p>The cervical cancer data set contains indicators and risk factors for predicting whether women will suffer from cervical cancer. These characteristics include human and statistical data (e.g. age), lifestyle and medical history.</p>
<p>Our main features are:</p>
<ul>
<li>Basic demographic data: age, age of first sexual intercourse, number of sexual partners, number of pregnancies</li>
<li>Drugs and drug data: smoking, smoking age, hormonal contraception, hormonal contraception, IUD, IUD </li>
<li>Disease data: Sexually Transmitted Diseases (STDs), Number of STDs diagnosed, time since first STD diagnosis, time since last STD diagnosis</li>
<li>Target output: The results of the biopsy are “health” or “cancer”.</li>
</ul>
<h2>Explanatory</h2>
<h3>Explanatory definitions</h3>
<p>There's never been a mathematical definition of interpretativeity. We've given some thought here.</p>
<p>Miller's informal definition of interpretativeity is the extent to which one can understand the reasons for decision-making.</p>
<p>Another explanation is: interpretability means the degree to which people can predict the results of models in a consistent manner.</p>
<p>The higher the interpretability of machine learning models, the easier it is to understand why certain decisions or predictions are made. If decision-making on one model is easier to understand than on another model, it is more explanatory than on another model.</p>
<p>Interpretability can be described in terms of both Interpretable and Explicable terms, but they are distinguished. We will use Expressable to describe the interpretation of individual case predictions. Interpretable is the interpretation of the whole model.</p>
<h3>Importance of interpretability</h3>
<p>When it comes to forecasting models, we need to weigh: we just want to know what the projections are. Or do you want to know why you made such a prediction? Both tend to be at both ends of the balance.</p>
<p>People want models to be explained for a few reasons.</p>
<ul>
<li><strong>Human curiosity and learning ability</strong> We're just curious about the answer to that.</li>
<li>Machine.<strong>Impact of decision-making on people ' s lives</strong>The bigger the machine, the more important it is to interpret its behavior.</li>
<li>The goal of science is<strong>Access to knowledge</strong>The model itself should be a source of knowledge. Interpretability makes it possible to extract these additional knowledge captured by models.</li>
<li>Machine learning models only work when they can be explained.<strong>Debugging and auditing</strong></li>
</ul>
<p>In some cases, we often do not need interpretativeity, but rather a better prediction.</p>
<ul>
<li>Explanatory is not required if the model has no significant impact</li>
<li>We don't need to explain when the problem is being studied in depth.</li>
<li>Interpretability may make it possible for a person or process to manipulate the system, and people may modify their own characteristics in the light of the results of the interpretation, most commonly in the context of credit reviews Nuclear</li>
</ul>
<h3>Classification of explanatory methods</h3>
<p>Methodologies for machine learning interpretability can be classified according to various criteria.</p>
<h4>Introsic or later?</h4>
<p>Essential interpretability refers to machine learning models, such as short decision tree or thin linear models, which are considered to be interpretable because of their simple structure;</p>
<p>Ex post interpretative means the application of an interpretative method after model training, e.g. the importance of a replacement feature is an ex post interpretative method.</p>
<p>The after-action approach could also be applied to models that are essentially interpretable. The sequence behind us is from the essence of the interpretation model to the ex post interpretation problem.</p>
<p>Essential interpretability is usually specific to certain categories of models (Model-specific), and ex post interpretation methods are often irrelevant to models (Model-agnostic).</p>
<h4>Local or global?</h4>
<p>Does the method of interpretation explain individual case predictions or whole model behaviour? Or is it in between?</p>
<h3>Explanatory scope</h3>
<h4>Algorithm Transparency</h4>
<p>Algorithmic training produces predictive models, and each step can be evaluated on the basis of transparency or interpretability</p>
<p>Algorithmic transparency refers to how algorithms learn models from data and what relationships they can learn. Transparency in algorithms requires knowledge of algorithms rather than data or learning models. He's related to interpretability, but he's two concepts.</p>
<h4>Global, corporate model interpretability</h4>
<p>Once you understand the whole model, you can describe it as an interpretable one.</p>
<p>The interpretability of this level is based on an overall understanding of model characteristics and each learning component (e.g. weight, other parameters and structures) how the model is made.</p>
<p>In practice, however, it is difficult to achieve the interpretability of global models, and any model beyond several parameters or weights is unlikely to be suitable for human short-term memory. Normally, when people try to understand a model, they consider only a part of it, such as weights in linear models. This is the section on "Global Model Interpretability at Modular Level."</p>
<h4>Global model interpretability at the modular level</h4>
<p>While global models are usually not interpretable, there is at least an opportunity to understand some models at the modular level.</p>
<p>Not all models can be explained at the parameter level. For linear models, it is explained that the weights are in part those of split nodes and leaf nodes for trees.</p>
<p>For example, linear models seem to be perfectly explained at the modular level, but the interpretation of individual weights is interrelated with all other weights. The interpretation of individual weights has always been accompanied by a footnote, namely, “Other input characteristics remain of the same value”, which is unrealistic in many practical applications. Still, it's a good model.</p>
<p><strong>The explanation at the modular level is that we are focused on research, because human understanding prevents us from understanding a model with a large number of (three or four) parameters.</strong></p>
<h4>Local interpretability of individual projections</h4>
<p>When global models cannot be explained, we may wish to look at an example.</p>
<p>Check the model ' s prediction for a particular input and explain the reasons. If you look at individual predictions, then the behavior of this original complex model may be more pleasant.</p>
<p>Local interpretation is more accurate than global interpretation. In the subsequent introduction, the chapter on “Model-independent methods” could make predictions of individual examples easier to interpret.</p>
<h4>Partial interpretability of a set of projections</h4>
<p>Very naturally, we can now explain individual examples, if we get an example group, using a separate local interpretation method, and then list the results for the whole group or aggregate the results. You can explain the model to a great extent.</p>
<h3>Nature of interpretation</h3>
<p>We have to explain the predictions of machine learning models. To achieve this, we rely on an interpretation method, an algorithm that generates interpretation.<strong>Interpretation usually links the characteristic values of the example to its model prediction in a way that humans understand.</strong></p>
<p>We examine the methods of interpretation and the nature of the interpretation, which is used to assist in determining whether an interpretation is good or not, but there is still a lack of quantitative assessment of whether it is good or not.</p>
<h4>Nature of means of interpretation</h4>
<ul>
<li><strong>Expressive Power</strong>: is the “language” or structure of the interpretation that the method can produce</li>
<li><strong>Translucency</strong>: Describe the extent to which the method of interpretation relies on looking at machine learning models (e.g. their parameters). Since the high-transparent method of interpretation and the model, the low-transparent method needs only to modify the input and then observe the prediction</li>
<li><strong>Portability</strong>: Describes the range of machine learning models using the means of interpretation, the larger the nature, the better</li>
<li><strong>Algorithmic Complexity</strong>: describe the computational complexity of the method of generating the interpretation</li>
</ul>
<h4>Nature of individual interpretation</h4>
<ul>
<li>Accuracy: If an interpretation is to be used to predict, a model of high accuracy is required, which is less accurate if only an explanation is required.</li>
<li>Solidity: explain the approximation of black box models</li>
<li>Consistency: How different is the interpretation between models trained on the same mission and producing similar predictions</li>
<li>Stability: How similar an interpretation can be between similar examples</li>
<li>Understandability: What is human understanding of interpretation?</li>
<li>Determination (Certainty): explain whether it reflects certainty in machine learning models</li>
<li>Importance (Degree of Importance): The extent to which the interpretation reflects the characteristics or partial importance of the interpretation</li>
<li>novelty (Novelty): explain whether the data examples to be explained are from a "new" area far from the distribution of training data</li>
<li>Representation (Reprecentativeness): How many instances can an explanation cover</li>
</ul>
<h3>A humane interpretation.</h3>
<p>The explanation is obvious, and there's nothing that traditional machine learning can do about it.</p>
<p>As an explanation of events, humans prefer short explanations (only 1 or 2 reasons), which compare the current situation with the non-occurrence of events, especially where unusual causes provide a good explanation.</p>
<p>When you consider the need for an interpretation of all the factors of prognosis or behaviour, you do not need a humane interpretation, but rather a complete attribution of cause and effect.</p>
<p>We're here to introduce what we're doing about people, and we're trying to get a quick understanding of what our models are doing.</p>
<ul>
<li><strong>Interpretation is contrasted.</strong> Humans do not usually ask why some predictions are made, but why they are made rather than another. So the good explanation is...<strong>Emphasizing the greatest differences between interested and target audiences</strong>。</li>
<li><strong>Selective interpretation</strong>One does not want an explanation of the actual and complete causes of the events covered. We are accustomed to choosing one or two of the possible reasons for interpretation.<strong>Even though the real situation is complex, only one or three reasons are given.</strong></li>
<li><strong>The explanation is social.</strong> : For a specific group of people, we need different interpretations, different people understand and focus on their audiences</li>
<li><strong>The point of explanation is abnormality.</strong>There is greater interest in explaining the causes of the events, which are very unlikely to occur, but do. Eliminating these unusual causes will significantly change the outcome (anti-fact interpretation). The anomaly is a very good explanation, even if models don't think so, but people think so.</li>
<li><strong>The explanation is true.</strong> : the interpretation should predict events as honestly as possible</li>
<li><strong>A good explanation is consistent with a priori knowledge of the subject.</strong>: Humans tend to ignore information that is inconsistent with their a priori knowledge, an effect known as confirmation deviations (see A/CN.9/WG.III/WP.36, paras. Even if you think your interpretation is true, it's not likely to be accepted if you don't know a priori.</li>
<li><strong>A good explanation is universal and probable.</strong>: Universality can be easily measured by characteristic “support”, i.e. the number of instances to which the interpretation applies divided by the total number of examples. Let's give the most common explanation possible.</li>
</ul>
<h2>Explanatory model</h2>
<p>The simplest way to achieve interpretability is to use only a subset of algorithms that create interpretable models. Linear regression, logical regression and decision-making trees are commonly used as interpretable models.</p>
<p>The contents of this chapter will focus on these interpretive models, we will focus only on our interpretation, not on the rationale, and we will simply review and provide a jumpover for those models that have been studied in the basics.</p>
<p>Characteristics of the model:
If the link between characteristics and objectives is linear, then the model is linear.</p>
<p>Models with single-modular constraints ensure that the relationship between the characteristics and the target results remains in the same direction throughout the characteristics: either the increase in the feature values will always lead to an increase in the target outcome or will always lead to a decrease in the target result. Modularity is useful for model interpretation as it makes understanding relationships easier.</p>
<p>Some models can automatically contain interactions between features to predict target results. You can create interactive features manually, and you can include them in any type of model. Interaction enhances forecasting performance, but too many or too complex interactions undermine interpretability.</p>
<p>Some models only deal with regression, others only with classifications and others both.</p>
<p>The usual models we're going to introduce are the following.</p>
<table>
<thead>
<tr>
<th>Algebra</th>
<th>Linear</th>
<th>Monophonic</th>
<th>Interactive</th>
<th>Tasks</th>
</tr>
</thead>
<tbody><tr>
<td>Linear regression</td>
<td>Yes</td>
<td>Yes</td>
<td>No</td>
<td>regr</td>
</tr>
<tr>
<td>Logical regression</td>
<td>No</td>
<td>Yes</td>
<td>No</td>
<td>class</td>
</tr>
<tr>
<td>Decision Tree</td>
<td>No</td>
<td>No</td>
<td>Yes</td>
<td>class,regr</td>
</tr>
<tr>
<td>RuleFit</td>
<td>Yes</td>
<td>No</td>
<td>Yes</td>
<td>class,regr</td>
</tr>
<tr>
<td>PARK Soo Bayes.</td>
<td>No</td>
<td>Yes</td>
<td>No</td>
<td>class</td>
</tr>
<tr>
<td>k-nearest neighbour</td>
<td>No</td>
<td>No</td>
<td>No</td>
<td>class,regr</td>
</tr>
</tbody></table>
<p><strong>There is no clear line between the interpretable and the non-explainable models and it needs to be viewed rationally</strong></p>
<h3>Linear regression</h3>
<p>The acceptance of linear models does not explain our online regression base.</p>
<h4>Explanatory characteristics</h4>
<p>The interpretation of weights in linear regression models depends on the type of characteristics.</p>
<ul>
<li>Numerical characteristics: Adding a unit of numerical features will change the estimated results according to their weight</li>
<li>Class II characteristics: Each example uses the characteristic of one of two possible values, while the other value is considered a reference category. The reclassification of the feature from the reference category to the other category would change the estimated result according to the weight of the feature.</li>
<li>There are several categories of classification characteristics: characteristics with possible values in fixed quantities. Usually we need to use one-hot code to handle multi-classification features. Then we'll use the explanation of the characteristics of the second classification.</li>
<li>Intersection: Intersection is the characteristic weight of the "continent feature" and for all examples is 1 and is explained as the projection for all numerical characteristics of zero and classification of cases under the reference category</li>
</ul>
<p>Based on the above explanations, we can give some explanation text templates, and use templates to automatically generate model coefficients</p>
<p><strong>Explanation of numerical characteristics</strong>：&#36;当所有其他特征保持不变时,特征x_k\text{ 增加一个单位,预测结果 }y\text{ 增加 }\beta_k\text{。}&#36;
<strong>Characteristics of the classification</strong>：&#36;当所有其他特征保持不变时,将特征x_k\text{ 从参照类别改变为其他类别时,预测结果 }y\text{ 会增加 }\beta_k\text{。}&#36;</p>
<p>In the case of interaction, the question of the interpretability of coefficients is clearly complicated.
We introduced a regression model with interactive entries.
&#36;&#36;Y_i=\beta_0+\beta_1x_i+\beta_2u_i+\beta_3w_i+\beta_4x_iw_i+\epsilon_i&#36;&#36;
The situation is different.</p>
<ul>
<li>Two binary variables interact.</li>
<li>A binary variable and a continuous variable interact.</li>
<li>Two consecutive variables interact</li>
</ul>
<p>Interactivity of two binary variables: first calculation of the expected values for all binary interactions and comparison of these expectations, all regression factors can be explained by the difference between the expected values</p>
<p>Intersection of continuous and binary variables: Depending on the context of the interactive item, different lines of regression can be obtained, and differences of regression lines can explain the significance of the regression factor</p>
<p>(b) The question of the interaction of two consecutive items: if we keep the original two volumes at the same time, any single variation of a variable from one variable will affect the variable by two coefficients;<strong>The coefficient of the joint item reflects the sum of the effects of the two variables at the same time as the two variables were converted more than the single variable.</strong></p>
<h4>Visually interpret linear regression</h4>
<p>Linear regression models have a lot of options for visual interpretation.</p>
<h5>Weight Plot</h5>
<p>Information on the weight table (weight and variance estimates) can be visualized in the weight chart </p>
<p>In order for them to be comparable on the axis, we need to standardize before using this visualization method.</p>
<p>As shown in the figure below:
<img src="/assets/images/interpretable-machine-learning/interpretable-ml-01.png" alt="Explainable Machine Learning Chart 01"></p>
<h5>Effect Plt</h5>
<p>The effect map is another visualization that we do not need to standardize, but multiply the weights of linear regression models with the actual characteristic values; then we draw the Boxplot for each characteristic effect and place it in a comparable axis. <img src="/assets/images/interpretable-machine-learning/interpretable-ml-02.png" alt="Explainable Machine Learning Chart 02"></p>
<h4>Explanation of individual case projections</h4>
<p>All that has been described is the overall interpretation model, and now we would like to be able to explain the prediction of a single case, why his projection is small (very large) and just to study the characteristic effects.</p>
<p>When we calculate the effects of the characteristics of this example, it is clear who has the most significant effect of combining their criteria in the effect map, and Hum is the main reason for the smallness of this example.
<img src="/assets/images/interpretable-machine-learning/interpretable-ml-03.png" alt="Explainable Machine Learning Chart 03"></p>
<h4>Linear regression discussion</h4>
<p>In terms of the nature of what constitutes a "good" interpretation, linear models do not create the best interpretation. They are contrasting, but the reference example is a data point constructed, where all numerical features are zero and the classification feature is set as their reference category, which is usually a manual and meaningless example and is unlikely to appear in your real data or reality.</p>
<p>Linear regression increases prediction modelling to a weight, makes prediction generation transparent and ensures that the best weight can be found, and allows for confidence-building, testing and reliable statistical theory</p>
<p>From the perspective of predictive performance, linear models are usually not so good, and it is not convenient that each non-linear or interactive must be artificial and provided to the model explicitly as an input feature.</p>
<p>The diversity of characteristics in the real world and the absence of those characteristics and interactions will significantly influence the interpretability of models.</p>
<h3>Logical regression</h3>
<p>Basic knowledge to refer to Logistic returns
When we studied the logical regression, we studied its interpretation methods like the Logistic regression coefficient.</p>
<h3>Decision Tree</h3>
<p>The decision tree is a very good machine learning algorithm that works very well on non-linear questions and the interaction of features. <a href="/en/blog/2024/03/28/machine-learning-introduction-supervised-learning/">Introduction to Machine Learning and Monitoring Learning: Decision Tree</a></p>
<h4>Explain the decision tree.</h4>
<h5>Visual decision tree</h5>
<p>The decision tree wants to be explained at the very heart of this.</p>
<p>The major software provides visualization of an ordinary decision tree, which is the basis of our understanding of the decision tree's interpretation.</p>
<h5>Interpretation</h5>
<p>The template to explain the decision tree is:
&#36;\text{“如果特征 }x\text{ 比阈值 }c\text{ [小/大] AND ...,那么预测结果就是节点 }y\text{ 中实例的平均值。”}&#36;</p>
<h5>Characteristic importance</h5>
<p>In the decision tree, the overall importance of a feature can be calculated by passing through all the divisions of the characteristic and measuring how much difference or Gini has been reduced relative to the parent node. The sum of all materiality is scaled to 100, which means that each materiality can be interpreted as part of the importance of the overall model.</p>
<h5>Tree decomposition and individual predictions</h5>
<p>Individual predictions of the decision tree can be explained by dividing the decision path into the composition of each characteristic. We can interpret projections by tracking decisions through trees and by adding contributions to each decision point.</p>
<p>From the predicted mean of the root node, the projection is modified in each division until the leaf node is reached, and the formula can be expressed as
&#36;&#36;\hat{f}(x)=\bar{y}+\sum_{d=1}^D\text{split.contrib}(d,x)=\bar{y}+\sum_{j=1}^p\text{feat.contrib}(j,x)&#36;&#36;
On the basis of the additions and subparagraphs of this formula, it is possible to determine which division has a significant impact, while adding the multiple division of characteristics together, it is possible to determine how much each feature contributes to the projections.</p>
<h4>Decision tree discussion</h4>
<p>The tree is very structural.<strong>Appropriate interaction between features in capture data</strong>I don't know. There's a natural visualization and there's a good interpretability.</p>
<p><strong>Trees can't handle linear relationships.</strong>I don't know. Enter any linear relationship between the feature and the result that must be approximated by partitioning to create a step-step function</p>
<p>The trees, the lack of smoothness, are rather unstable, the trees too deep are hard to understand.</p>
<h3>RuleFit</h3>
<p>The linear regression model doesn't take into account the interaction between the features, so we want to look for something that's...<strong>It's like a linear model that's simple and interpretable, but that's where it comes together.</strong>This is RuleFit.</p>
<p>RuleFit learns rare linear models with original features and many new features (decision-making rules), which capture the interaction between original features.<strong>RuleFit automatically generates these features from the decision tree and can calculate their significance.</strong></p>
<p><strong>The rule of decision-making itself is a sort of classification algorithm, but it's only working.</strong></p>
<h4>Method of interpretation</h4>
<p>Since RuleFit eventually estimates a linear model, its interpretation is the same as that of the “conventional” linear model. The only difference is that the model has new features derived from decision-making rules. The decision-making rule is a binary feature: a value of 1 indicates that all conditions of the rule are met, otherwise the value is 0.</p>
<p>For linear items in RuleFit, the explanation is the same as in linear regression models:<strong>If a unit is added to the feature, the forecast results change with the corresponding characteristic weights.</strong>For decision-making characteristics it should be ** if a decision-making rule &#36;r_k&#36; All conditions are applicable. &#36;\beta_k&#36; **</p>
<h4>RuleFit Discussion</h4>
<p>RuleFit <strong>Automatically cross-add features to linear models</strong>I don't know. It therefore addresses the issue of linear models that must be manually added to the interactive function and helps to model non-linear relationships.</p>
<p>He's a very effective method of automatic interaction, but only those decision-making interactive items are identified, but he'll generate too many interactive items, and we'll need to compress them with something like LASSO.</p>
<p>There's a paper that claims that RuleFit is very good -- close to the predictions of random forests, but it's hard to be recognized that if it was really good, he wouldn't be so natural. First Name</p>
<p>The end product of the RubeFit process is a linear model with additional fancy features (decision-making rules). And the linear model explains the need to keep other features unchanged, and there is a real risk of conflict in decision-making rules.</p>
<h3>PARK Soo Bayes</h3>
<p>PARK Soo Bayes's explanation is based on the distribution of each feature, and we just need to look at the probability.<a href="/en/blog/2024/03/28/machine-learning-introduction-supervised-learning/">Machine Learning Introduction and Monitoring Learning: The Bayesian Catalogue</a> Very easy to understand.</p>
<h3>k-nearest neighbour</h3>
<p>We talked about k-near Neighborhood. <a href="/en/blog/2024/03/28/machine-learning-introduction-supervised-learning/">Introduction to Machine Learning and Monitoring Learning: k - Recent Neighborhood</a>
The "sample-based interpretation" section of this paper is the closest possible reference.</p>
<h2>Method of interpretation not related to the model</h2>
<h3>Before you start.</h3>
<p>Separating the interpretation from the machine learning model (= non-model-related means of interpretation) has the advantage of being very flexible, and we use any model we like. At the same time, it is easier to compare interpretability between models, as the same methods can be used for any type of model.</p>
<p>The alternative to a model's unconnected means of interpretation is to use only an interpretable model, whose shortcomings are clear and at the expense of predictive performance.</p>
<p>The model of interpretation gives more vision to machine learning, and in the traditional machine learning method, we're looking at how we can learn from it.<strong>World</strong>Collection<strong>Data</strong>♪ With ♪<strong>Learning</strong>Method obtained<strong>Model</strong>And finally, using models.<strong>Projections</strong>I don't know. Now we've added a layer of predictions and models that can be made by models themselves.<strong>Human understanding.</strong></p>
<p>This multilayered abstract structure allows us to understand the differences in methodology between statisticians and machine-learning specialists.<strong>The statisticians handle the data layer, they skip the black box model layer, then move to the interpretive method layer.</strong> <strong>Machine learning specialists handle data layers and train black box machine learning models. Over the Explanatory Method layer, humans directly addressed the prediction of the Black Box model.</strong></p>
<p>Of course, such a structure is a one-size-fits-all, and the data may come from simulations, and the black box model may output projections that are not for human use, but it remains an interesting point of view.</p>
<h3>Partially dependent on the figure</h3>
<h4>Definitions</h4>
<p>Partially dependent (Partial Data Plot, short of PDP or PD) shows the marginal effect of one or two characteristics on the projection of machine learning models</p>
<p>Partially dependent on the chart to show linear, monotonous or more complex relationships between objectives and features. Partly dependent functions for regression are defined as:
That's right.<em>{x_S}(x_S)=E</em>{x_C}\left[\hat{f}(x_S,x_C)\right]=\int\hat{f}(x_S,x_C)d\mathbb{P}(x_C)&#36;&#36;</p>
<p>&#36;x_S&#36;is the feature that is partly dependent on the function to be drawn,&#36;x_C&#36;It's a machine learning model.&#36;\hat{f}&#36;Other features used.</p>
<p>Usually, gather.&#36;S&#36; Only one or two of them.&#36;S&#36;The characteristics are the ones we want to know about their impact on projections. Characteristic vector&#36;x_S&#36;and&#36;x_C&#36; Merge into total feature space&#36;x&#36;。</p>
<p>Some dependencies are coming together.&#36;C&#36;It's an output of a learning model for marginalized machines, so...<strong>This function shows a collection that we're interested in.&#36;S&#36;Relationship between characteristics and projected results</strong>I don't know. By marginalizing other features, we get to depend on it.&#36;S&#36;, and functions that interact with other features.</p>
<p>In practical applications, we use the MC method to calculate the partially dependent function (MC method to calculate the points)
That's right.<em>{x_S}(x_S)=\frac{1}{n}\sum</em>{i=1}^{n}\hat{f}(x_S,x_C^{(i)})&#36;&#36;
<strong>We can probably analyze the idea of this method from a visual perspective, which is actually an expectation.</strong></p>
<p>PDP assumes that the characteristics in C are not relevant to the characteristics in S. If this assumption is violated, the averages calculated on a partial basis will include data points that are highly unlikely or even impossible</p>
<p>For the classification of the probability of machine learning model output, the function is used in part to show the probability of a given category under different characteristic values in S. One simple way to deal with multiple categories is to draw a line or map for each category.</p>
<p>The partial reliance on the figure is a global approach: it takes into account all the examples and gives a description of the overall relationship between the characteristics and the projected results.</p>
<p>For classification characteristics, partial reliance is easy to calculate. We do this all the time in the differential analysis, comparing the average of the groups, which is partly based on an experimental design method.</p>
<h4>Example:</h4>
<p>We need to use examples to help us understand exactly what we're doing. Partly, after all, this is a strange way for us to explain.</p>
<p>In fact, the feature set S usually contains only one feature or at most two because one feature produces a 2D figure, while two features produce a 3D figure. Everything else is very difficult.</p>
<p>We consider the models of the number of bicycles and the PDPs for the temperature, humidity and wind speeds, which are shown below, in which the cross-references are raw data and their distribution, even if the coordinates are the number of bicycles projected.
<img src="/assets/images/interpretable-machine-learning/interpretable-ml-04.png" alt="Explainable Machine Learning Chart 04">
As can be seen from the figure, the model predicts a large number of bicycles on average for warm, but not too hot weather, with more than 60 humidity reducing the desire to rent bicycles, and at the same time, wind speeds are largely completely negative, and wind speeds are higher and fewer people are riding.</p>
<p>Let's consider the PDF issue of a sorted variable and the impact of seasonal features on predicting bicycle rentals, as illustrated below.
<img src="/assets/images/interpretable-machine-learning/interpretable-ml-05.png" alt="Explainable Machine Learning Chart 05">
We find that all seasons have similar impacts on model predictions, and in spring alone, models predict fewer bicycle leases. There's a difference between this figure and the graphics of the variance analysis, which only focuses on the average, and the variance analysis tends to use Boxplot to study the general distribution.</p>
<p><strong>The PDP only takes into account the average and reasonable, because many models cannot output the difference estimates for regression.</strong></p>
<p>We gave another example of a partial dependency that simultaneously visualizes both features, using colour maps as a method of distinguishing values and using some sort of discrete method of mapping.
<img src="/assets/images/interpretable-machine-learning/interpretable-ml-06.png" alt="Explainable Machine Learning Chart 06">
It is easy to see that the greater the probabilistic value, the greater the Num, the larger the Age's bias.</p>
<h4>Strengths</h4>
<ul>
<li>Partly dependent on graphic calculations is intuitive and non-professionals usually quickly understand the PDP concept</li>
<li>It's easy to explain the partial reliance on maps. It takes a simple reading ability and one or two examples to learn quickly.</li>
<li>Part of it depends on the map.</li>
<li>Partially dependent on chart calculations with causality</li>
</ul>
<h4>Disadvantages</h4>
<ul>
<li>Partially dependent on actual in function<strong>Maximum feature number is 2</strong> This is because of the two-dimensional medium and our unimaginable high-dimensional space problem.</li>
<li>Some PD charts do not show the distribution of features and strongly recommend using RUG (data point indicator on x axis) or histograms to help us understand the distribution of features. The PD map, which lacks characterizations, is likely to lead us to misinterpret some unprovoked predictions that have no samples.</li>
<li>The assumption of independence is that PD's biggest problem, and we'll consider it later.</li>
<li>The heterogeneity effects may be hidden because the PD curve shows only average marginal effects.</li>
</ul>
<p><strong>Heterogeneity</strong>: Assuming, for one feature, that half of your data point is positively relevant to the projection — the larger the feature, the larger the projection — the less the other half has negative relevance — the larger the projection. The PD curve may be a horizontal line, as the effects of the data set in two parts may be offset by each other. Then you can conclude that this feature has no impact on predictions.<strong>It's because some other variable is interacting.</strong>, drawing an individual curve helps us find this problem</p>
<h3>Individual expectations</h3>
<h4>Definitions</h4>
<p>Individual conditions expectations (Individual Regulation, short ICE) displays a line for each example showing how the prediction of the example changes when the feature changes.</p>
<p>The characterization average effect relies in part on the map as a global approach, as it does not focus on specific examples, but on overall averages. PDPs with an equal value to a single data example are referred to as individual conditions expectations (ICEs) figures. PDP is the average of the ICE figure</p>
<p>All other features are the same, creating the variant of the example by replacing the feature with the value of the value in the grid and projecting these newly created examples using the Black Box model. The result is a set of features from the grid and corresponding predictions. Connecting it is what we want.</p>
<p>The purpose of ICE was to deal with the problem of partial reliance on maps that might mask an interaction-created isomer relationship, and ICE charts are more rational than PDPs when interaction exists.</p>
<p>The official definition of ICE is: &#36;\begin{aligned}&amp;\\text{in ICE {(x S^(i)}, x C^(i)&#125;&#125; {i}\text{in each instance, \hat{f} S^(i)}\text{is about}x S^(i)}\text{, at this time}x C^(i)}\text{&amp;\text{changed. I'm sorry.</p>
<h4>Example:</h4>
<p>We also use examples to help us understand what ICE means and how to use the data set used in this paper for the "Culture Cancer Risks (Classification)" section of the model prediction that gives the probability of classification rather than the ICE graph 01. Icon
<img src="/assets/images/interpretable-machine-learning/interpretable-ml-07.png" alt="Explainable Machine Learning Chart 07">
It can be seen that the age effect follows the trend of average increase at age 50 (the trend for most people) and that for a small number of individuals who have a higher predictive probability at a young age, the predicted cancer probability will not change significantly with age.
Basically following the same pattern, which means ICE and PDP basically reflect the same trend.</p>
<h4>Strengths</h4>
<p>Individual conditions expect curves to be more intuitive than partial reliance maps, and a line represents the projection of an example</p>
<p>ICE Curve reveals heterogeneity</p>
<h4>Disadvantages</h4>
<p>ICE curves can only show a meaningful feature, which is still due to the lack of plane media and human imagination.</p>
<p>If many ICE curves are drawn, the map may be too crowded to see anything, general<strong>It is suggested that some transparency be added and then superimposed</strong>。</p>
<p>It's hard to see averages in ICE drawings.<strong>Suggested mix with PDP</strong> </p>
<p>ICE can help us identify interactions, but it is still not possible to interpret them well, and if the characteristics of interest are linked to other features, some points in the line may be invalid data points</p>
<h3>Cumulative Local Effects Chart</h3>
<h4>Thought</h4>
<p>The cumulative local effect (Accumulated Local Effects Plot) describes how the average feature influences the prediction of machine learning models. The ALE figure is a faster, more unbiased alternative that relies partly on the PDP. Both approaches have the same goal.</p>
<p>We know that if the features of the machine learning model were relevant, then partial reliance would not be credible, and we actually produced some totally impractical samples in the PDP.&#36;x_1&#36;They can't exist in real applications, but we pretend everything's normal.</p>
<p>How do we get it?<strong>Estimation of characterization effects that respect identity relevance</strong>What? We can average the condition distribution of the feature, which is&#36;x_1&#36; , yes. &#36;x_1&#36; Projections of similar examples are average. This method is called the Marginal Plot or M. </p>
<p>However, M-figures are not perfect, and M-figures avoid average predictions of data examples that are unlikely to occur, but they mix the effects of characteristics with those of all relevant characteristics. That is, even if a particular feature has no influence on the target itself, his relevance to an influential feature will be reflected in his influence.</p>
<p>So we bring out the ALE, which calculates the difference in projections rather than the average, based on the condition distribution of characteristics, that is,<strong>ALE charts how model predictions of data examples in this window are around &#36;v&#36; Features &#36;x_j&#36; A small "window" change</strong> </p>
<p>We use small windows to avoid the absorption of the relevant characteristic, which is essentially offset by the impact of that characteristic using differential techniques.</p>
<p><strong>I don't usually use M-charts, so forget it now. Just remember, ALE does.</strong></p>
<h4>estimate</h4>
<p>We just know the idea of the ALE, but we need a specific explanation of the way.</p>
<h5>ALE for individual characteristics</h5>
<p>First of all, we can calculate the uncentralized individual ALE.
That's a good idea.<em>{j,ALE}(x)=\sum\limits</em>\(z k,j}\sum\lits i:x j^(i)}\j(k)}\left[f(z k,j},x {\(i)}-f(z k-1,j},x {\setminus})\
We know that ALE's idea is to calculate the difference in projections, so ours.&#36;z&#36; It's a feature of real interest. We divide the compartments, and we finally calculate the average difference in the projection of this interval.</p>
<p>Centralize the effect. The average effect is zero.
That's right.<em>{j,ALE}(x)=\hat{\tilde{f&#125;&#125;</em>{j,ALE}(x)-\frac{1}{n}\sum_{i=1}^{n}\hat{\tilde{f&#125;&#125;<em>{j,ALE}(x</em>That's right.
After centralisation, the ALE value can be interpreted as the main effect of the characteristic under a given value</p>
<p>The fractional number of the feature distribution is generally used as a grid for defining spacing. The use of fractional numbers ensures that an equal amount of data is available at each interval. The disadvantage of the fraction number is that the length of the interval may be very different. If the characteristics of interest are highly skewed (e.g., many low and only a few very high), this may lead to anomalies in some ALE maps.</p>
<h5>ALE for the interaction of two features</h5>
<p>ALE charts also show the interaction of two features. The principles of calculation are the same as individual features, but we use rectangular units instead of spacing. We omit the overly complex formula and use a chart to visualize it, as follows:
<img src="/assets/images/interpretable-machine-learning/interpretable-ml-08.png" alt="Explainable Machine Learning Chart 08">
We essentially need to calculate the second order of all examples within each grid cell. Bad</p>
<p>Since the ALE estimates for both characteristics show only the second-tier effects of the characteristics, particular attention needs to be paid to interpretation. The secondary effect is the additional interaction of the characteristics after consideration of the main effects of the characteristics.</p>
<p>It is assumed that the two features are not interactive, but each has linear effects on the projection results. In the one-dimensional ALE map of each feature, we will see a straight line as the estimated ALE curve.</p>
<p>But when we draw 2D ALE estimates, they should be close to zero, because the secondary effect is only an additional interaction. ALE and PD figures differ in this respect: PDP always shows total effects, ALE charts show first- or second-tier effects.</p>
<p>The ALE map of the classification features is complex and requires artificially defined distances, and we briefly present the analytical methods of the results when we encounter them, omitting theoretical descriptions.</p>
<h4>Example:</h4>
<p>We're still using examples to help us understand how the ALE diagram works, using the Bike Leasing (Return) part of the data set here to help us explain.</p>
<p>The use of ALE is generally limited to centralized ALE, and if we suspect the presence of characteristics, we should consider the use of ALE charts.</p>
<p>ALE based on temperature, humidity and wind speed prediction models
<img src="/assets/images/interpretable-machine-learning/interpretable-ml-09.png" alt="Explainable Machine Learning Chart 09"></p>
<p>With regard to relevance, consideration of the relevant coefficient is the best option.</p>
<p>It's natural to read ALE and PDP.</p>
<p>The ALE chart of the classification variables is as follows:
<img src="/assets/images/interpretable-machine-learning/interpretable-ml-10.png" alt="Explanatory Machine Learning Chart 10">
We can give the same explanation as the PDP. It's just the ALE's centralization.</p>
<p>We consider the second-order effect of humidity and temperature on the projected number of leased bicycles, noting that he does not include the main effects, but only the issue of interaction.
<img src="/assets/images/interpretable-machine-learning/interpretable-ml-11.png" alt="Explainable Machine Learning Chart 11">
You can see that hot and wet weather increases predictions. In cold and damp weather, it also has negative effects on the number of bicycles projected. Response</p>
<p>The main effects of humidity and temperature indicate that the number of bicycles is projected to decrease in very hot and humid weather. Therefore, in hot and humid weather, the combined effects of temperature and humidity are not the sum of the main effects, but are larger than the sum, and we may consider this time.<strong>Second-order PDP to reflect combined effects</strong>
<img src="/assets/images/interpretable-machine-learning/interpretable-ml-12.png" alt="Explainable Machine Learning Chart 12"></p>
<h4>Strengths</h4>
<p>ALE charts are impartial, which means they remain valid when the characteristics are relevant. And the PDP is not valid.</p>
<p>ALE drawings are calculated faster than PDP</p>
<p>The interpretation of the ALE figure is clear: the relative impact of the change feature on the projection can be read from the ALE chart under the given value.</p>
<p>The ALE chart is centred on 0. That makes them better, because each point of the ALE curve is a difference from the average forecast.</p>
<p>2D ALE drawings show only interactive effects: if the two features are not interactive, they do not show anything.</p>
<p><strong>In most cases, it is proposed to use the ALE figure instead of the PDP figure, as features are usually relevant to some extent.</strong></p>
<h4>Disadvantages</h4>
<p>The ALE chart may become somewhat unstable (many small fluctuations, especially in the second-order ALE) and may be highly spaced, usually because of the problem of spacing setting, and the ALE chart may not be accurate if it is too small. If it's too large, the curve may become unstable.</p>
<p>ALE chart does not have ICE curves, does not deal with individual predictions, isthogenic.</p>
<p>It's a little annoying to explain the second-order effect map, because you always have to remember the main effects, which, like the PDP, make sense in the picture.</p>
<p>Even if the ALE chart does not deviate from the relevant characteristics,<strong>However, when characteristics are strongly relevant, interpretation remains difficult.</strong> We can't eliminate those effects.</p>
<h3>Feature Interactive</h3>
<h4>Definitions</h4>
<p>Characteristic Interaction: When the feature interacts in the predictive model, the projection cannot be expressed as the sum of the feature effects, since the effect of one feature depends on the value of the other. Aristotle's "whole is greater than the sum of the parts" applies where there are interactions.</p>
<p>The PDP is completely incapable of dealing with issues under the interaction, and ICE can allow us to observe the interaction from an intuitive perspective, but it is difficult to explain carefully that ALE has addressed the issue of characteristics and has explored some aspects of second-order interaction, which unfortunately are not sufficient.</p>
<p><strong>We're working on PDP, ALE, ICE, and now we're working on it.</strong> </p>
<p>If the machine learning model projects on the basis of two characteristics, it can be broken down into four: constants, the first feature, the second feature and the interaction between the two. <strong>The interaction between the two features is the projected change by changing the characteristics after considering the effects of a single feature.</strong> The effect of the characteristics on the final projection is no longer independent.</p>
<p>One way to estimate the intensity of interaction is to measure the extent to which projected changes depend on the interaction of characteristics. This measure is called &#36;H&#36; Statistics</p>
<h4>Friedman's H Count</h4>
<p>We will deal with two situations: first, using a two-way interactive measure, which tells us whether and to what extent the two characteristics of the model interact; and second, an overall interactive measure, which tells us whether and to what extent a feature interacts with all other features of the model.</p>
<p>If the two features are not interactive, we can split the partially dependent function as follows:<em>Let's say the PDF's already centralized.</em>
&#36;&#36;PD_{jk}(x_j,x_k)=PD_j(x_j)+PD_k(x_k)&#36;&#36;
That is, the two-way partial reliance function of both characteristics is directly the sum of the function of the individual feature.</p>
<p>If a feature does not interact with any other feature, the entire predictive function can be broken down into the sum of the non-interactive and other characteristics that are partially dependent on the function as follows:
&#36;&#36;\hat{f}(x)=PD_j(x_j)+PD_{-j}(x_{-j})&#36;&#36;</p>
<p>That's why we came up with the features. &#36;j&#36;and &#36;k&#36; The interaction between the &#36;H&#36; The statistics are:
&#36;&#36;H_{jk}^{2}=\sum_{i=1}^{n}\left[PD_{jk}(x_{j}^{(i)},x_{k}^{(i)})-PD_{j}(x_{j}^{(i)})-PD_{k}(x_{k}^{(i)})\right]^{2}/\sum_{i=1}^{n}PD_{jk}^{2}(x_{j}^{(i)},x_{k}^{(i)})&#36;&#36;
The same applies to measurement features. &#36;j&#36; Whether to interact with any other feature:
&#36;&#36;H_{j}^{2}=\sum_{i=1}^{n}\left[\hat{f}(x^{(i)})-PD_{j}(x_{j}^{(i)})-PD_{-j}(x_{-j}^{(i)})\right]^{2}/\sum_{i=1}^{n}\hat{f}^{2}(x^{(i)})&#36;&#36;
<strong>If there is no interaction at all, the statistical amount is 0, if the interaction between the two features is counted as 1 means that each PD function is constant and the effect on the projection is only from interaction.</strong></p>
<p>&#36;H&#36; The assessment costs of statistics are high. &#36;n&#36; Call required &#36;n^2&#36; The prediction function at the secondary level can reduce performance costs by sampling the original data points, but can cause statistical instability.</p>
<h4>Example:</h4>
<p>Let's see what the interactive features are in practice, and we'll explain it in the context of the cycle lease (return) section of this paper, the risk factor for cervical cancer (classification) section of this paper.</p>
<p>We measure the interaction of the SVM model in the regression problem.
<img src="/assets/images/interpretable-machine-learning/interpretable-ml-13.png" alt="Explainable Machine Learning Chart 13">
Projections of the intensity of interaction of each characteristic of the support vector leased by bicycle with all other characteristics (H statistics) are shown in the figure above. Overall, interaction between features is weak (less than 10 per cent of the variance for each characteristic explanation).</p>
<p>Consider classification and use the RF model for prediction
<img src="/assets/images/interpretable-machine-learning/interpretable-ml-14.png" alt="Explainable Machine Learning Chart 14">
A number of indicators, such as HC and NUM, are highly interactive. That's over 30 percent of the explanation.</p>
<p>After looking at the characteristics of each and all other characteristics, we can select one of them and then study in greater depth all the two-way interactions between the selected features and the other. Let's take the example of the NUM on classification.
<img src="/assets/images/interpretable-machine-learning/interpretable-ml-15.png" alt="Explainable Machine Learning Chart 15">
This is the result of our analysis.</p>
<h4>Strengths</h4>
<p>Interaction &#36;H&#36; Statistics are available through partial reliance on disaggregation<strong>Theory Foundation</strong>。</p>
<p>&#36;H&#36; Statistically available<strong>Meaningful explanation.</strong>: Interactivity is defined as the difference in shares explained by the interaction.</p>
<p>Because statistical information is<strong>Unspecified</strong>, and always between 0 and 1, so it is comparable between features and even models.</p>
<p>Statistical Information Society<strong>Test all types of interactions</strong>Whatever their special form.</p>
<p>Use H stat, also<strong>You can analyze any higher-level interaction.</strong>, for example, the intensity of interaction between three or more features.</p>
<h4>Disadvantages</h4>
<p>Interactive H statistics take a long time to calculate because of its<strong>It's a big calculation.</strong>。</p>
<p>This calculation relates to estimated marginal distribution.<strong>If we don't use all the data points, there's a difference in these estimates.</strong>I don't know. This means that when we sample the points, the estimates also vary from operation to operation and the results may be unstable. I propose to repeat the calculations of H statistics several times to see if there are enough data to achieve stable results.</p>
<p>It's not possible to judge if the interaction is greater than zero, and there's no theory to help us deal with it, and it's hard to say when H's statistics are so big that we think the interaction is strong. Which means...<strong>It's all about experience.</strong></p>
<p>Only the intensity of the interaction can be judged, and there is no capacity for more detailed analysis, which requires a return to 2D PDP or ICE analysis</p>
<p>Not applicable to computer visual problems because no image can be processed (in pixels)</p>
<p>It's too relevant to render the method ineffective.</p>
<h3>Replace feature importance</h3>
<h4>Definitions</h4>
<p>The replacement feature importance measures the increase in error in the model that we predict by replacing the feature value, which breaks the relationship between the feature and the real result.</p>
<p>The algorithms have the following ideas.</p>
<p>Training model&#36;f&#36;,Specific Matrix&#36;X&#36; Target vector&#36;y&#36;, Error measure&#36;L(y,f)&#36;Estimated original model error&#36;e^{orig}=&#36; &#36;L(y,f(X))&#36; (e.g., average error) characteristics&#36;j\leftarrow1&#36; to &#36;p&#36;By changing data&#36;X&#36;Characteristics in&#36;j&#36;Generate Feature Matrix&#36;X^{perm}&#36;。</p>
<p>It destroys the character.&#36;j&#36;And the real results.&#36;y&#36;The links are based on data replacement projections, estimated errors&#36;e^{perm}=L(Y,f(X^{perm})&#36;Calculate replacement feature importance&#36;FI^j=e^{perm}/e^{orig}&#36;I don't know. Alternatively, differences can be used:&#36;FI^j=e^{perm}-e^{orig}&#36;Sort features in descending order, in which comparable replacement feature importance is most common</p>
<h4>Importance of using training data or testing data</h4>
<p>At present, there is no complete answer to this question;</p>
<p>In the case of selection of training data, the combination seriously affects our correct thinking about the error in the model, resulting in the invalidity of the final character importance judgement;<strong>The importance of characterization based on training data leads us to mistakenly believe that characterization is important for prediction, whereas in fact models are simply oversyncs and not at all.</strong></p>
<p>For the selection of test data: if all data are used to train the model, this means that there are no available test data, so we propose cross-validation to address this issue, but it means that the significance of the feature is calculated on a subset of the data expressed differently.</p>
<p>Therefore, if we choose the importance of training data computing features, it means that we want to know to what extent the model depends on each feature for prediction; and correspondingly, testing data means that we want to study the extent to which that feature contributes to the model ' s performance on unknown data.</p>
<p><strong>We can't come to a definitive conclusion. We need more experience and research to help us think.</strong></p>
<h4>Example:</h4>
<p>Characteristics of increasing model error to 1 times (= no change) It's not important to predict cervix cancer, and we're going to take the bike lease (return) section of this paper, the risk factor of cervix cancer (classification) section of this paper, and two data sets to discuss our problem.</p>
<p>For classification issues,
<img src="/assets/images/interpretable-machine-learning/interpretable-ml-16.png" alt="Explainable Machine Learning Chart 16">
There's something about return.
<img src="/assets/images/interpretable-machine-learning/interpretable-ml-17.png" alt="Explainable Machine Learning Chart 17">
There's basically nothing to explain.</p>
<h4>Strengths</h4>
<p>Good explanatory: Characteristic importance is the increase in model error when characteristic information is destroyed.</p>
<p>Between questions, the measure of characteristic importance is comparable (provided we use the margin)</p>
<p><strong>The importance of changing features automatically takes into account all interactions with other features.</strong></p>
<p>The replacement feature importance does not require a re-training model, which significantly reduces the operating costs</p>
<h4>Disadvantages</h4>
<p>We don't know whether to choose between training data or testing data to determine the importance of characteristics.</p>
<p>We need real samples, real and marked samples for training models.</p>
<p><strong>If the features are relevant, the importance of changing the features may be biased by unrealistic data examples</strong> So it's still not going to be able to deal with the issue of strong relevance, and it's now the third model that can't be dealt with. </p>
<h3>Global proxy model</h3>
<h4>Definitions</h4>
<p>Now it's time to discuss proxy models, and we want to create an interpretable model that can train him to be close to black box predictions, and then deal with machine learning interpretability by interpreting proxy models.</p>
<p>Understanding proxy models does not actually require much theory. We want to be there. &#36;g&#36; Explanatory constraints, proxy model prediction function &#36;g&#36; Approach our black box prediction function as closely as possible &#36;f&#36;I don't know. For Functions &#36;g&#36;, can use the "explainable model" section of any paper</p>
<p>One way to measure the ability of the agent to copy the Black Box model is to use the R formula as
&#36;R^2=1-\frac{SSE}=1-\frac{sum <em>*^{(i)}-\hat{y}^{(i)})^2}{\sum</em>I don't know.
And when online re-entry, we also looked at the R side, the R side of the linear re-entry base, where we mentioned that R had no true meaning, but with the proxy model,<strong>The R side can be explained by the percentage difference captured by the proxy model, which is the ability of the proxy model to explain the black box.</strong></p>
<p>We don't talk about black box performance here, and if it's not good, then the proxy model's explanation becomes irrelevant.</p>
<p><strong>The whole agency problem is training a new explanatory model. We don't waste time rereading it here.</strong></p>
<h4>Strengths</h4>
<p>The proxy model is very flexible, so we can change the new black box model or replace the new interpretation model, and it's perfectly possible to hand it over to multiple teams in parallel.</p>
<p>Using R, we can easily measure the performance of our proxy model in approaching the black box prediction.</p>
<h4>Disadvantages</h4>
<p>The conclusion is that the model is not the data, since the proxy model will never see the actual results.</p>
<p>It's not clear what the R side's best cut-off point is, like the linear regression.</p>
<p>For a subset of a data set, it is explained that the model may be very close, while for another subset, it may vary considerably. In this case, the interpretation of simple models will vary for all data points. The local effects should therefore be considered.</p>
<p>All the strengths and weaknesses of the depreciable proxy model itself</p>
<h3>Local proxy model LIME</h3>
<h4>Definitions</h4>
<p>Local proxy models (Local interpretable model-agnostic models, LIME) are self-depreciable models for<strong>An individual case prediction explaining the Black Box Machine Learning Model</strong> </p>
<p>The central idea of LIME is that for any complex black box model (Black-box Model), it is difficult to understand its decision-making boundaries from a global perspective, but it is possible to enter samples in a specific context.<strong>Local</strong>, a simple, self-articulated model (e.g. linear regression or decision tree) is used to approach and formulate black box models.</p>
<p>The model studied should be a good approximation of the local predictions of the machine learning model, but it is not necessarily a good global approximation, which is easier to approximate a local nature than the global approximation.</p>
<h4>A mathematical explanation.</h4>
<p>LIME aims to optimize the following target functions:</p>
<p>&#36;&#36;\xi(x) = \arg\min_{g \in G} \mathcal{L}(f, g, \pi_x) + \Omega(g)&#36;&#36;</p>
<ul>
<li>&#36;x&#36;: We need to explain specific input samples.</li>
<li>&#36;f&#36;: Black box models to be explained (e.g. random forest, deep neural network).</li>
<li>&#36;g&#36;: Explanatory model (usually simple model community) &#36;G&#36;, such as linear models).</li>
<li>&#36;\pi_x&#36;: Local proximity measurement, defined in samples &#36;x&#36; Weight of other nearby samples (range) &#36;x&#36; The closer you get, the greater the weight.</li>
<li>&#36;\mathcal{L}(f, g, \pi_x)&#36;: Solidity Loss, measured by &#36;x&#36; It's a local neighborhood.&#36;g&#36; Proposed &#36;f&#36; Errors in forecast results.</li>
<li>&#36;\Omega(g)&#36;: Complexity, we need to limit in order to ensure interpretativeity &#36;g&#36; Complexity (e.g. limiting the number of non-zero characteristics of linear models).</li>
</ul>
<p><strong>Experimental and realization mechanisms:</strong></p>
<ol>
<li><strong>Perturbation:</strong> Right Input &#36;x&#36; Microturbation (e.g., in the case of text, some words are randomly discarded; in the case of images, some hyperpixel blocks are shielded) to generate new samples.</li>
<li><strong>Black Box Forecast:</strong> Enter these new samples into the black box model &#36;f&#36;, obtain probabilities.</li>
<li><strong>Weighted training:</strong> According to the disturbance sample and the original sample, &#36;x&#36; Distance &#36;\pi_x&#36; Calculate weights and then use these data to train one with regularity &#36;\Omega(g)&#36; The White Box Model &#36;g&#36;。</li>
<li><strong>Extract explanation:</strong> White Box Model &#36;g&#36; the weight or structure of the &#36;x&#36; Local interpretation of projected results.</li>
</ol>
<h4>Sample disturbance of data</h4>
<p>We mentioned that LIME needed to create a data set based on an example of disturbance, and here we are presenting the method of disturbance.</p>
<p>For structured data, the basic disturbance is shown in the figure below.
<img src="/assets/images/interpretable-machine-learning/interpretable-ml-18.png" alt="Explainable Machine Learning Chart">
The first is the original projection, the second is the point where some normal samples are generated in the case of interest, the third is the weight assigned to points based on distance and the fourth is the local learning method.</p>
<p>It is very difficult to define a meaningful neighbourhood around a point. The current LIME uses index smooth cores to define a neighbourhood, but there are many possibilities for nuclear parameter regulation, which will affect the outcome, but there is no defined way to help us determine it.</p>
<p>Here's one example: we use the feature effect: we multiply the weight by the actual feature value to judge the local effect of this feature.
<img src="/assets/images/interpretable-machine-learning/interpretable-ml-19.png" alt="Explainable Machine Learning Chart 19"></p>
<h4>Strengths</h4>
<p>Its advantages are broadly consistent with the global proxy model;</p>
<p>An explanation will be simple and effective when using LASO or a short decision tree.</p>
<h4>Disadvantages</h4>
<p>When using LIME for table format data, the correct definition of the neighbourhood is a large unresolved issue</p>
<h3>SHAP: from methodological map to feature article</h3>
<p>SHAP (SHAP) uses Shapley values to explain the characteristic contribution of a given projection to the baseline. It can provide partial attribution, and it can summarize the attributions of multiple samples into global importance, summary and dependency maps. TreeSHAP provides efficient algorithms for tree models, and KernelSHAP is calculated in a non-model manner.</p>
<p>It is appropriate to answer “why does the model give this prediction”, but it cannot be interpreted as a causal link to the real world. Both background data, feature relevance and the definition of “identity loss” change the outcome; the same model changes a sample of benchmarks, which may also be interpreted differently.</p>
<p>Full Shapley extrapolation, KernelSHAP, TreeSHAP, DeepSHAP, GradientSHAP, Background Data Selection and Project Errors were collated<a href="/en/blog/2026/02/28/shapley-and-shap/">Shapley and SHAP - Model Explanatory SOTA Tool</a>I don't know. There's no repetition here, just keeping SHAP on an interpretable map.</p>
<h2>Sample-based interpretation</h2>
<h3>What's based on a sample explanation?</h3>
<p>Sample-based Interpretations Select a specific example of a data set to explain the behaviour of a machine learning model or to explain the bottom data distribution, and he does not explain an example.</p>
<p>The explanation based on the sample is mostly unrelated to the model and the differences in methods not related to the model The reason is that the sample-based approach explains the model by selecting examples of datasets rather than by creating feature summaries (e.g., feature importance or partial dependence).</p>
<p>A sample-based interpretation would be meaningful only if we were able to present data examples in a way that humans could understand. This is very effective for the image.</p>
<p>Let's just give an example based on a sample.</p>
<p>A little cat sits on a window table in a fireless, uninhabited house. The fire department has arrived and one of the firefighters is considering whether he could venture into the building to save the kittens. He remembers what he met when he was a fireman.<strong>Similar situation</strong>The old cabins, which had been burning slowly for some time, were often unstable and eventually collapsed.<strong>Because of the similarity of this situation,</strong>He decided not to enter because the risk of a house collapse was too high.</p>
<p>And that's what we want models to think like humans (at least at the level of interpretation): things like B, things like A, things A lead to Y, so I predict that B will also cause Y.</p>
<h3>Counterfaction</h3>
<h4>Definitions</h4>
<p>Counterfactual Interpretations describe a causal relationship in the following form: "Y would not have happened if X had not happened," thinking against facts requires conceiving a hypothetical reality that contradicts the observed facts, but rather the way humans think frequently.</p>
<p>The counterfact explanation of the model describes the smallest change in the characteristic value of changing the forecast to predefined output (anti-fact), i.e., if I wanted to change the fact that the loan was rejected, i.e., if I wanted to make an extra dollar a year, or if we wanted to raise the rent of our own house, we needed to expand the size of the house.</p>
<p>The counter-facts are human friendly interpretations because they contrast with current examples and because they are selective, which means that they usually focus on a small number of characterizations.</p>
<p>But the counterfact is beset by the "Roshenmon effect." The non-exclusiveness of a counter-fact interpretation deserves our separate consideration, and this multiple counter-fact problem can be resolved by reporting on all counter-fact explanations or by developing criteria to assess the counter-fact and choose the best counter-fact.</p>
<p>We usually ask for satisfaction.</p>
<ul>
<li>Counter-fact cases should produce predefined projections as closely as possible, i.e., do not change too many target values at once, such as classification probability and regression predictions. Value</li>
<li>Counterfact should be as similar as possible to the examples of characteristic values so that as few features as possible can be changed Value</li>
<li>Anti-facts cases should have possible characteristics, and there's no point in those counter-fact explanations that are completely out of step with reality.</li>
</ul>
<h4>Generate a counter-fact explanation</h4>
<p>A simple way of producing counterfacts is to search through repeated experiments, but it's too stupid, so we usually use some methods of optimizing losses to generate counterfacts.</p>
<p>Wachter et al. recommend to minimize the following losses:
&#36;L(x,x)&#39;,y&#39;,\lambda)=\lambda\cdot(\hat{f}(x&#39;)-y&#39;)^2+d(x,x&#39;That's it.
The first is the square example of predictions and expected results, and the second is the distance between examples and counter-facts.&#36;\lambda&#36; It's the coefficient used to mix the two effects, the larger it means we want to have the opposite facts closer to prediction, the smaller it means we want to change the characteristics as little as possible. Value</p>
<p>It's customary to use MAD's deformation as a distance behind.
&#36;d(x,x)&#39;)=\sum_{j=1}^p\frac{|x_j-x&#39;_j|}{MAD_j}&#36;&#36;</p>
<p>As for the anti-fact example, it's very easy to understand.</p>
<h4>Strengths</h4>
<p>The explanations of the counterfacts are clear and very easy to understand.</p>
<p>An anti-fact approach does not require access to data or models. It only needs access to the predictive function of the model.</p>
<p>This method also applies to systems that do not learn by machine. We can create a counter-fact for receiving input and returning any system that is exported.</p>
<p>The counterfact interpretation method is relatively easy to achieve, as it is essentially a loss function that can be optimized using the standard optimized library.</p>
<h4>Disadvantages</h4>
<p>The counterfact interpretation method is relatively easy to achieve, as it is essentially a loss function that can be optimized using the standard optimized library.</p>
<p>Classification characteristics with many different levels cannot be well addressed. The authors of the methodology suggest that the method be operated separately for each combination of characteristic values of the classification characteristics, but if you have multiple classification features with multiple values, this will result in a combination explosion.</p>
<h3>Fight the sample.</h3>
<h4>Definitions</h4>
<p>Anti-sampling (Adversarial Express) is the case when a small change in a given characteristic value of a sample makes a miscalculation of the entire model. This is very similar to the counterfact explanation.<strong>The anti-sampling sample is an example of the opposite.</strong>, designed to defraud models rather than to interpret models.</p>
<p>For example, machine-learning scanners are scanning luggage at the airport. To avoid detection, people invented a knife to make the system think it was an umbrella.</p>
<h4>Methodology and examples</h4>
<p>The approach in this section focuses on image taxonomyrs with deep nervous networks, where extensive research has been carried out, and visualization of anti-images has a strong educational significance.</p>
<p>The anti-image sample is an image with a deliberate disturbance pixel intended to defraud the model during its application. The samples are impressive evidence of how easily people can look at harmless images to deceive deep-seated nervous networks for target identification, and the changes predicted are not understandable to human observers. The anti-sampling samples are like optical illusions to machines.</p>
<p>Since the anti-smuggling sample is concentrated in the CV field, there are no more presentations here.</p>
<h3>Examples of influence</h3>
<p>The machine learning model is ultimately the product of training data, and the removal of one of the training examples may affect the model generated. When training examples are removed from training data, they change the parameters or projections of the model significantly, so we call this example “impacted”. By identifying examples of influential training, we can “debug” machine learning models and better explain their behaviour and predictions.</p>
<p>In this subsection, we do not view models as fixed models, but rather as functions of training data. Examples of influence can help us answer questions about global model behaviour and individual predictions.</p>
<h4>Angular values and influential examples</h4>
<p>An anomaly (also known as entanglement) is an example of other examples of distance from data concentration, and of impact when the anomaly impact model is used.</p>
<p>An example of impact is the example of data, whose deletion has a significant impact on training models. The greater the changes in model parameters or projections, the greater the impact of the example, when the model is retrained after the specific example is removed from the training data.</p>
<h4>Delete Diagnosis</h4>
<p>Examples of how the removal of the model has had a significant impact on the model are generally considered to have had an impact.</p>
<p> DFBETA assessed the coefficient shift after the deletion of an example.
 &#36;&#36;DFBETA_{i}=\beta-\beta^{(-i)}&#36;&#36;
Cook Distance can assess his impact on the overall predictive performance of the model, but only for the LM and GLM models, and the Cook Distance in a broad linear regression</p>
<p>The simplest impact levels for the model ' s projected impact can be written as:
&#36;&#36;US&#36;US&#36;&#36;US&#36;US&#36;&#36;US&#36;US&#36;&#36;US&#36;&#36;US&#36;US&#36;&#36;US&#36;US&#36;&#36;US&#36;&#36;&#36;US&#36;&#36;US&#36;&#36;&#36;&#36;US&#36;&#36;&#36;&#36;US&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;...&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;<em>{j}-\hat{y}</em>\right|&#36;&#36;
It's a very common format.</p>
<h4>Impact Functions</h4>
<p>Sometimes we don't want to train so many models, especially when they're complicated.</p>
<p>So if there is a model with a loss function, the model's parameters have a second-ordered guide, we can consider using the Impact Functions to estimate the impact of the example on model parameters and projections.</p>
<p>The method of influencing functions requires a loss gradient associated with model parameters, which only applies to the subset of machine learning models, all of which are not tree-based, but the neural network is available, which is good news.</p>
<h4>Strengths</h4>
<p>The study of influential examples highlighted the role of training data in the learning process. This makes impact function and elimination of diagnosis one of the best debugging tools for machine learning models.</p>
<p>It's not a model to delete a diagnosis.</p>
<h4>Disadvantages</h4>
<p>The deletion of the calculation of the diagnosis is very expensive, as they require retraining. But the math is going to go way beyond what you think.</p>
<p>The impact function is a good alternative to deleting diagnosis, but only for micro-parameter models</p>
<p>The effect function is only similar, as it is expanded twice around the parameter. The approximation could be wrong.</p>
<p>There are no clear cut-off points for the impact intensity with or without impact</p>
<p>Impact intensity only considers the deletion of individual examples, not the deletion of multiple ones at a time. The group of data examples may have some interaction, but, in dealing with interaction, our computing needs are increasing exponentially.</p>
<h2>Future</h2>
<h3>Prerequisites for forecasting</h3>
<p>"Provision" based on three premises</p>
<ul>
<li>Digital: Any (interesting) information will be digitized</li>
<li>Automation: A mission will be automated when it can be automated and when the cost of automation is lower than the cost of carrying out the mission over time.</li>
<li>Incomplete target norms: We cannot perfect a limited target.</li>
</ul>
<p>Digitalization is certain, automation will conflict with imperfect target norms, and it is therefore difficult for us to train a model that is completely automatic and perfectly achievable.</p>
<p><strong>This conflict is partly mediated by means of interpretation.</strong></p>
<h3>A little story.</h3>
<p><strong>2030: Medical laboratory in Switzerland</strong></p>
<p>"This is definitely not the worst way to die!" Tom concluded, "Trying to find something positive in this tragedy." He removed the pump from the IVF.</p>
<p>Lena added, "He died just for the wrong reasons."</p>
<p>"There are, of course, the wrong morphine pumps! They're adding to our workload!" Tom complained while he was pulling the backboard of the pump. After removing all the screws, he put the plate aside. He inserted the cable into the diagnostic port.</p>
<p>"You're not just complaining about work, are you?" Lena laughed.</p>
<p>“Of course not. Never!” He's sarcasm-bling.</p>
<p>He started the pump computer.</p>
<p>Lena inserts the other end of the cable into the tablet. "Okay, the diagnostic program is running." She said, "I'm really curious about what's going on."</p>
<p>"It did inject our John Doe into Nirvana. That morphine is very high. Dude, I mean... this is the first time, right? Usually, a bad pump only releases little sweetness or no taste. But never, like that crazy injection. Tom explains.</p>
<p>"I know. You don't have to convince me. Hey, look at that. Lena raised her tablet. "Do you see this peak? That's the efficacy of painkillers. Look! This line shows the reference level. This poor guy mixed multiple painkillers in his blood system that could kill him 17 times or more. Here's our pump injection. And then here she goes, "You can see here the moment the patient dies."</p>
<p>"Well, you know what happened, boss?" Tom asked his boss.</p>
<p>"Well... the sensors seem to be very good. Heart rate, oxygen level, glucose etc. Data collected as expected. There are missing values in blood oxygen data, but this is not unusual. Here, sensors also detect a reduction in heart rate and cortex levels caused by morphine derivatives and other painkillers. She continues to view the diagnosis.</p>
<p>Tom was obsessed with staring at the screen. This was his first investigation into the failure of the real equipment.</p>
<p>“Well, that's our first problem. The system failed to send a warning to the hospital's communication channel. The warning was triggered, but the emergency programme did not respond. It could be our fault, but it could be the hospital's fault. Please send the log to the IT team." Lena says to Tom.</p>
<p>Tom noded his head and his eyes were still on the screen.</p>
<p>Lena goes on to say, "It's weird. The warning should also lead to the shutdown of the pumps. But it clearly did not do so, and that must be a mistake. The quality team missed something. It's really bad. It may be related to the emergency programme.”</p>
<p>"So, somehow, the emergency system of the pump went down, but why was the pump so crazy and injected a lot of painkillers into John Doe?" Tom wondered.</p>
<p>“Good question. You're right. Apart from emergency emergencies, pumps should never have used that much medicine. Given the low level of cortical alcohol and other warning signals, the algorithm should have stopped earlier.” Lena explains.</p>
<p>"Maybe there's some misfortune, like a millionth of a millionth of a million?" Tom asked her.</p>
<p>"No, Tom. If you read the files I sent you, you'll know that the pump was tested first in animal experiments, then in humans.<strong>The perfect amount of painkillers is given by learning to input according to the sense. The pumping algorithm may be non-transparent and complex, but it is not random.</strong> This means that in the same circumstances the pump will again operate in exactly the same way and our patients will die again. The combination of the input or the unwanted interaction must have triggered the wrong act of the pump. That's why we have to dig deep to find out what's going on here." Lena explained.</p>
<p>"I understand..." Tom replied with a confused answer, "Isn't the patient dying soon? Because of cancer or something."</p>
<p>Lena noded in reading the analysis.</p>
<p>Tom, get up and go to the window. He looked out and looked at something far away. “Maybe the machine freed him from suffering and helped him to do him a favor and not suffer any more. Maybe it just did the right thing, like lightning, but, you know, a good lightning. I mean like lottery tickets, but not random. But for some reason. I would do the same thing if I were a pump."</p>
<p>She finally looked up at him.</p>
<p>He's been looking outside for something.</p>
<p>They were all silent for a moment.</p>
<p>Lena, keep your head down again and keep analysis. "No, Tom. It was a mistake. It was a fucking mistake."</p>
<h3>Meaning of model interpretability</h3>
<p>According to the previous story, we know that interpretability is an important point.<strong>When the results of the model have a major impact on the real world</strong>The major security problem of AI is very important at this time, and we do not want a black box model to cause enormous damage to society as a whole.</p>
<p>The blogger adds:<strong>When people need to interact with models,</strong> There is also value in explanatory, and more relevant studies may focus more on generating models than predictive models, which are discussed mainly in this chapter, and know why producing such results will guide us in adjusting the next generation to interpretive ones.</p>
<h3>The future of machine learning.</h3>
<p>The discussion explains the future of machine learning, and the future of machine learning.</p>
<ul>
<li>Machine learning will grow slowly and steadily.</li>
<li>Machine learning drives a lot of things.</li>
<li>Explanatory tools facilitate the introduction and research of machine learning</li>
</ul>
<h3>Explanatory future</h3>
<ul>
<li>Emphasis will be placed on interpretive tools that are not relevant to the model</li>
<li>Machine learning will be automated and interpretable.</li>
<li>Data scientists will automate themselves and the program will interpret itself (same as the former).</li>
</ul>
