---
title: 'Exploratory Data Analysis: Descriptive Statistics, Visualization, and Preprocessing'
title_zh: 探索性数据分析：描述统计、可视化与数据预处理
date: 2024-02-29 21:41:47 +0800
categories:
- Data Science
- Data Practice
tags:
- Exploratory Data Analysis
- Data Preprocessing
author: Hyacehila
mathjax: true
hidden: true
excerpt: Covers EDA concepts, descriptive statistics, data visualization, preprocessing, cleaning, integration, transformation,
  feature selection and construction, and data transforms.
description: Covers EDA concepts, descriptive statistics, data visualization, preprocessing, cleaning, integration, transformation,
  feature selection and construction, and data transforms.
excerpt_zh: 整理 EDA 的基本思想、描述性统计、数据可视化、数据预处理、数据清洗、数据融合、数据转换、特征选择与特征构造以及数据变换。
permalink: /blog/2024/02/29/exploratory-data-analysis-learning-notes/
lang: en
translation_key: 2024-02-29-exploratory-data-analysis-learning-notes
translation_status: machine
translation_source_hash: fe588e5c8018ebf502b9be6e31205ad845cf9368a535adc3b1814b0aa41ec24e
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<h2>Overview of exploratory data analysis ideas</h2>
<h3>What's the data analysis?</h3>
<p><strong>Data analysis</strong> It's a new direction in statistics.</p>
<ul>
<li>It does not require us to give an accurate measure of the uncertainty of the method, which is indispensable in the estimation of parameters in classical statistics and in the hypothetical tests;</li>
<li>It values the anti-disturbability of the method and its effectiveness at the same level.</li>
</ul>
<p>Existing mathematical statistics attach great importance to measurement uncertainty, but they tend to rely on assumptions for this precise measure, which then leads to the theory of mathematical statistics being applied lightly;</p>
<p>The emergence of data analysis is a change in the idea that we have shifted the focus of our work to making the data talk, processing our data in some less formal ways, providing a basis for the subsequent stage of validation, without considering the precise measure of their uncertainty.</p>
<p>— Chen Chi-Yu — Explored data analysis in Chinese</p>
<h3>What is exploratory data analysis?</h3>
<p>Simply put, exploratory data analysis is a bridge between classical statistical analysis and popular data mining and machine learning (also those parts of classical statistical modelling).</p>
<p>He's the only thing that makes us understand the data, the data from different angles.</p>
<p>EDA (exploratory data analysis) is an emerging part of statistics with the relative concept of CDA validation data analysis</p>
<p>Data analysis is in two stages, exploratory and authentic, and we can only give the right results in alternate use.</p>
<p>This exploration does not prejudge ideas and assumptions, but rather wishes to use the results of exploration to present some ideas and assumptions for subsequent modelling, i.e. CDAs, which are inseparable.</p>
<p>It's obvious that the EDA should be a non-parametric approach.</p>
<h3>What's in the EDA?</h3>
<p>Scientific visualization is an important part of the EDA. <a href="/en/blog/2024/03/15/r-visualization-learning-notes/">R Statistical visualization</a> in which separate presentations are made</p>
<p>Descriptive statistics are the subject of an introduction to statistics, and we do separate studies, descriptive statistics and visualization, and we could have classified them all in the EDA.</p>
<p>The major part of probability density estimates is presented in descriptive statistics. As far as further validation is concerned, it falls within the category of non-parametric statistics, which can be read in conjunction with the distributional shape in visualization.</p>
<p>Pre-processing of data should also be part of the EDA, which is the basis for subsequent data analysis and will be presented separately later in this paper.</p>
<p>In addition, feature engineering is an important pre-treatment method in the field of machine learning, which includes features screening, feature construction and downscaling.</p>
<h3>Characteristics of exploratory data analysis</h3>
<p>According to David and Tukey, underwriting Robust and Exploratory Data Analysis, an exploratory data analysis was presented. </p>
<p>In the current industry, Robust has been translated into greatness, and Professor Chen has been translated into resistance, and sometimes we also call it robustness, and here we do not dwell on the details of translation, which is central to it.</p>
<p><strong>Validity requires that data analysis methods are insensitive to the local poorness of the data, which is local, but can change dramatically.</strong> </p>
<p>So what's the connection between exploratory data analysis and robustness?
<strong>Validity and exploratory data analysis are holistic rather than explicit.</strong>
All exploratory data analysis methods should be as insensitive as possible.</p>
<p>Therefore, the use of fractionals as a representative of high robust digital characteristics in the EDA is very extensive, including five-digit synopsis, box lines, differentials, and so on.</p>
<h2>Know our Data</h2>
<h3>Descriptive statistics</h3>
<p>Descriptive statistical analysis is the most basic way of understanding data.</p>
<h3>Data visualization</h3>
<p>Besides, data visualization is an important part of the EDA. <a href="/en/blog/2024/03/15/r-visualization-learning-notes/">R Statistical visualization</a></p>
<h2>Data preprocessing</h2>
<h3>Basic introduction to data pre-processing and frequently asked questions</h3>
<h4>Basic description of pre-processing data</h4>
<p>Before we start, let's get a picture of the entire data mining process.</p>
<p><img src="/assets/images/data-science-notes/eda-workflow.png" alt="Explored data analysis process"></p>
<p>Pre-processed data are the most basic elements of data mining, but it also happens to be the most time-consuming part of data mining, the more vulnerable part of data science platforms such as Kagle, which contains a lot of work.</p>
<p><strong>Data and features determine the upper limit of machine learning, and the selected models and algorithms are just going to approach the upper limit.</strong></p>
<h4>Common issues in data preprocessing</h4>
<p>Here's why we're doing pre-processing.</p>
<p>For data objects, we can obtain unprocessed features (attributions), which may be the following:</p>
<ul>
<li>Not part of the same schematic: the specification of the characteristic is different and cannot be compared together (uniform unit)</li>
<li>Information redundancies: For certain quantitative features, the useful information contained is compartmentalized, such as learning achievement, and if only “failure” or “failure”, then quantitative scores need to be converted to “1” and “0” for pass and failure (quantitative characterization)</li>
<li>Qualitative characteristics cannot be directly used: some machine learning algorithms and models can only accept input of quantitative features, and then the qualitative features need to be converted to quantitative features (Qualitative Quantification)</li>
<li>Missing value: missing value needs to be supplemented (missing)</li>
<li>When data dimensions are too high, there is also the problem of what is called "Curse of Demensionality" <strong>(data down)</strong> We'll stay on the feature project and discuss the data downs and downs in the feature project.</li>
</ul>
<h4>Feature extraction and type conversion</h4>
<p>Data are first required for pre-processing. The large amount of data obtained in experimental or operational systems is often not directly relevant to research issues and is often unstructured. At this point, the first step in the data mining exercise is to be carried out: feature extraction and type conversion.</p>
<p><strong>Create a set of feature data that can be used by analysts to convert feature data to a uniform expression</strong></p>
<p>The way in which the feature extraction takes place depends on the problem we're dealing with, and often also on the experience of analysts.</p>
<table>
<thead>
<tr>
<th>Source Data Type</th>
<th>Target data type</th>
<th>Methodology</th>
</tr>
</thead>
<tbody><tr>
<td>Numeric Type</td>
<td>Type</td>
<td>Separated</td>
</tr>
<tr>
<td>Type</td>
<td>Numeric Type</td>
<td>Binary</td>
</tr>
<tr>
<td>Text</td>
<td>Numeric Type</td>
<td>Potential semantic analysis (LSA)</td>
</tr>
<tr>
<td>Organisation</td>
<td>Disconnected sequences</td>
<td>Symbol Convergence Approximation (SAX)</td>
</tr>
<tr>
<td>Organisation</td>
<td>Multi-dimensional Numeric</td>
<td>Dispersed small wave transformation (DWT); discrete Fourier transformation (DFT)</td>
</tr>
<tr>
<td>Disconnected sequences</td>
<td>Multi-dimensional Numeric</td>
<td>Dispersed small wave transformation (DWT); discrete Fourier transformation (DFT)</td>
</tr>
<tr>
<td>Space</td>
<td>Multi-dimensional Numeric</td>
<td>DWT</td>
</tr>
<tr>
<td>Figure</td>
<td>Multi-dimensional Numeric</td>
<td>Multi-dimensional scale (MDS); image conversion</td>
</tr>
<tr>
<td>Any type</td>
<td>Figure</td>
<td>Similar figure (limited availability)</td>
</tr>
</tbody></table>
<p>As a matter of strong empirical reliance, there is no introduction to the fact that we have our own programs in different fields, no common methodology.</p>
<h4>Complex feature breakdown</h4>
<p>Some feature fields may include a large amount of information, such as a field containing the time and time of the year in which our valuable information is hidden, but cannot be used directly, which is the problem of decomposition of complex features.</p>
<p><strong>String Split</strong>: In the case of the Titanic data sets, the class number C123 indicates the passenger identification information from which the information from the C-class is likely to be effective for predictions, which is the string split.</p>
<p><strong>Timetamp Split</strong>Depending on how we feel, consider which of the time-spans of the year to keep, such as four seasons, working days and rest days, etc.; if there are time-zone information, there is also a need to consider how to harmonize them, while time-zone information can also reveal geographic information as an independent feature.</p>
<p><strong>Location Information Split</strong>Longitudes, countries, etc. are part of the location-segregation information; we often wish to consider longitudes separately, so that they generally have a better effect.</p>
<h4>Common data pre-processing methods</h4>
<p>We'll be back with the data pre-processing approach below, which must not be complete, because many of the methods are taught in other courses.</p>
<ul>
<li>Data cleansing: missing certain log values for data cleansing processing data, noise in smoothing data, detection of anomalies, correction of inconsistencies, etc.</li>
<li>Integration of data: Integration of data from different sources of varying quality. Good data integration reduces redundancy and inconsistency in data, thereby increasing the accuracy and speed of subsequent steps</li>
<li>Data conversion: conversion of data into forms applicable to data mining through smooth aggregation, data overview, standardization, etc.</li>
<li>Data downside: to translate high-dimensional data into low-dimensional data and to maintain most of the information in the original data, so that data mining results are the same or almost the same as pre-downgrade results</li>
</ul>
<h4>Relevant theoretical support</h4>
<p>Code Relationship Algebra 3<strong>The data is sorted out to make sense of the structure of the data.</strong>Which means...</p>
<ul>
<li>Each feature is in one column.</li>
<li>Every example is in one line.</li>
<li>Each feature should have only one table.</li>
<li>If the features are in multiple tables, then there's a column that connects them.
That's why we're compiling the data.</li>
</ul>
<h3>Data Cleaning</h3>
<p>Let's just talk about what needs to be done.</p>
<h4>Missing value processing</h4>
<p>Missing values are an unavoidable problem in actual data, and different strategies should be adopted for different data scenarios, starting with the distribution of missing values.</p>
<ul>
<li>If the missing value is minimal and this dimension is not important, normal<strong>Delete</strong>They have little impact on the overall data situation;</li>
<li>If the missing value is higher or the information on this dimension is still important, the immediate deletion will have a bad effect on the results of the algorithm run behind, and we need to consider plugging data.</li>
</ul>
<p>Losses generally fall into three categories. </p>
<ul>
<li>Total random missing MCAR (and any other variables unrelated to yourself) </li>
<li>Random missing MAR (misses associated with other observation variables)</li>
<li>Non-random missing MANR (misses associated with their own extraction)</li>
</ul>
<p>Of these, MAR is the most common, and we generally assume that this is the only way to do it on the basis of modelling plugs.</p>
<h5>Average or Middle Fill</h5>
<p>It will not reduce sample information and process simple, but will cause deviations when missing data are not random;
For normal distribution data, the average can be used instead, and if the data are tilted, the median may be better used</p>
<h5>Integrating technical thinking</h5>
<ul>
<li>Random plug-in - random extraction of a sample from the general population in lieu of missing samples;</li>
<li><strong>Multiple plug-in methods</strong>• Forecasting missing data through the relationship between variables, such as the use of the Monte Carlo methodology to generate complete data sets and, finally, to aggregate the results of the analysis,<strong>It's a relatively robust plug-in technique, highly recommended.</strong></li>
<li>Thermal platform plug-in -- find a sample similar to the missing value in the non-missive data cluster (symmetry sample) and use the observations in it to plug in the missing value, that is, &#36;k&#36;Next plug.</li>
</ul>
<h5>Modelling plug-in technology</h5>
<p>It could be determined by means of a process based on reasoning such as return.</p>
<p>For example, using the properties of data centralizing other data, a determination tree (return equation) can be constructed to predict the value of the missing value</p>
<p><strong>Multi-plugging (Multiple Imputation)</strong></p>
<h5>Retain Missing Technology</h5>
<ul>
<li>In the case of classification problems, the missing can be used as a category</li>
<li>If the model is willing to accept the existence of NA, retain NA as the missing embedded model to the back</li>
</ul>
<h5>Missing is sometimes information.</h5>
<p>For random absences, we can only deal with by plug-in. However, there are many problems involving the extraction of data that are not pure, the absence of regularity in many cases, such as those resulting from a failure in the work of an operational component, the lack of the means of data collection itself, and, if the reasons for the absence can be identified, we can re-target the processing of data to fill the gaps, and sometimes even the lack of data can be extrapolated very reliably through other available data.</p>
<p>Sometimes the lack itself is an information to consider. It would also be valuable to include the missing themselves as information in the model, and the absence of some data implied the characteristics of the users themselves.</p>
<h4>Treatment of anomalies</h4>
<p>Unusual values are also commonly referred to as “outlier”, i.e. points that are not consistent with the general behaviour or characteristics of other sample points in the sample space. In general there may be the following causes:</p>
<ul>
<li>Error in calculation or error in operation (miscal data)</li>
<li>The variability or elasticity of the data itself (real special data)</li>
</ul>
<p><strong>Disconnection points may not necessarily be useless data, but they may be of interest to users, for example in the area of fraud detection, where those that are inconsistent with normal data behaviour tend to signal fraud and thus become a concern for law enforcement.</strong></p>
<p>We're not the only ones dealing with detached values.</p>
<ul>
<li>De-grouping is the most common method of processing</li>
<li>If we really want to keep this sample, we usually use a plug-in that's considered missing.</li>
<li>If algorithms aren't sensitive to entanglement, it's okay not to process entanglement values.</li>
</ul>
<h5>Statistically based discrete site testing</h5>
<p>Such tests assume that all data in the sample space are consistent with a distribution or data model, or
Discordancy test (discordancy test) Points </p>
<p>Normal distribution&#36;3\sigma&#36; Principle. Use of box-line maps for detached site detection.</p>
<h5>Distance-based isolation site testing</h5>
<p>Just as we conduct cluster analysis based on distance, those isolated categories can be considered as discrete points. Here, reference can be made to machine learning about distance calculation and hierarchy.
In practice, those points that are too far away from the other points are considered separate points.</p>
<h5>Density-based isolation site testing</h5>
<p>Similarly, discrete points can be understood from the point of view of density clusters.
In practice, those too low-density points are considered discrete. This method detects both global and local detached sites.</p>
<h4>Data heavy.</h4>
<p>Repetition of data is common in practice, and in some data mining models these redundant data increase the difficulty of data analysis and processing speed, and therefore data need to be addressed Heavy</p>
<p>There are common ways.</p>
<ul>
<li>Cross-referenced data search — highly complex, only in smaller data situations</li>
<li>Hash stated that - generating data fingerprints, simple and efficient, applicable to large-scale data, represents algorithms:<ul>
<li>Bitmap: Bitmap method;</li>
<li>SimHash: similar to Hashi;</li>
<li>Blom Filter</li>
</ul>
</li>
</ul>
<p><strong>Many models don't need data to work hard.</strong></p>
<h4>Data to Noise</h4>
<p>Noise, a random error or equation of the measured variable; most data mining methods treat discrete points as noise or anomaly, but there are data mining methods that specifically study noise</p>
<p><strong>Observation = True Data + Noise</strong></p>
<p>The usual data-noise methods are:</p>
<ul>
<li>Boxing method: Checking data “neighbor” (i.e., surrounding values) for smooth and orderly data values</li>
<li>regression method: smooth data using a function that combines data to help eliminate noise</li>
</ul>
<p><strong>Most models (especially statistical models) don't need data for noise work.</strong></p>
<h4>Manually remove useless features</h4>
<p>The manual removal of useless features is highly dependent on our experience, and here is a summary of some basic experiences. It was removed from the perspective of the data itself, and those business-based manual filter features could not be described here:</p>
<ul>
<li>Delete a clear non-functional amount: if the number of the observation is used, there is no theoretical positive effect on the model</li>
<li>Delete the amount of excess of the missing value ratio: no fixed line, depending on the circumstances</li>
<li>Removes the feature of a difference of almost zero:<ul>
<li>The number of variable values is less than 10% of the variable.</li>
<li>Biggest two withdrawals of frequency, frequency ratio above 20%.</li>
</ul>
</li>
<li>Delete the compound linear amount:<ul>
<li>Found the two strongest variables of the current correlation coefficient</li>
<li>Calculate the relevant coefficients for them and for the rest of the population (combinant coefficients)</li>
<li>Delete the largest variable of the compound coefficient</li>
<li>Whether this step needs to be repeated and continued to be deleted, depending on the data set</li>
</ul>
</li>
<li>Screening based on relevant coefficients to retain more relevant quantities (see<a href="/en/blog/2023/11/05/descriptive-statistics-and-visualization-notes/">Descriptive statistics and visualization: related analysis</a>）</li>
</ul>
<h3>Data Integration</h3>
<p>We're still introducing what we need to do.</p>
<p>Data integration integrates data of different kinds from different sources. Good data integration reduces redundancy and inconsistencies in data</p>
<p>In practical use, data integration is often determined by data, for example: Entity Identification Project: trying to match the records of different data sources pointing to the same entity in the real world.</p>
<p>There's more to it.</p>
<h3>Data Conversion</h3>
<p>The purpose of the data conversion is to move data from one form of expression to another and to meet the conditions for data mining. </p>
<p>And here we're presenting some of the more basic data conversion methods, which in fact has become a subject in statistics, with a lot of research.</p>
<p>Common data conversion methods can be broadly grouped into the following categories:</p>
<ul>
<li>Separated</li>
<li>Binary</li>
<li>Harmonization and standardization</li>
<li>Feature Encoding</li>
</ul>
<h4>Separated</h4>
<p>Some data mining algorithms, in particular some classification algorithms, require that data be in the form of classification attributes or that classification forms be effective in increasing the efficiency of algorithms</p>
<p>In this way, continuous properties often need to be converted to classification properties (dissemination, disscretion) and continuing and discrete properties may need to be converted to one or more binary properties</p>
<p>Common discrete methods include:</p>
<ul>
<li>No monitoring: boxing method (separate/equivalent), visual division, median grouping, etc.</li>
<li>Supervision: Chimerge Law MDPL Law, CAIM Law, etc.
<strong>Monitoring separation is a good method for FE workers who lack intuitive and field knowledge</strong></li>
</ul>
<p><strong>We've told you to be careful.<a href="/en/blog/2024/03/15/r-visualization-learning-notes/">R Discussion on statistical visualization of careful data processing</a>But at this point, we have to be separated. Just be vigilant.</strong></p>
<p><strong>A customary rule of the boxing method is:</strong></p>
<ul>
<li>The number of groups between 5 and 20, the number of data in groups.</li>
<li>Group spacing as far as possible (except when equal frequency groups)</li>
<li>The group distance selects the odd number as far as possible, but even if it's the case.</li>
<li>The maximum group ceiling covers all individuals, but avoid too many spills as much as possible.</li>
<li>Avoid the use of groups with upper or lower limits</li>
</ul>
<h4>Binary</h4>
<p>Characteristic dualisation is the process of converting a numerical feature to a boolean value, the core of which is to set one
Threshold, with 1 value greater than the threshold and 0 value less than the threshold</p>
<h4>Harmonization and standardization</h4>
<p>Normalization is a simple calculation, with a schematic expression transformed into
An expression with no scalding, becomes the cursor </p>
<p>Standardization aimed at<strong>Harmonization of data profiles</strong> Increase the interpretability of algorithms, and at the same time, in those algorithms that require a gradient decline, the uniform scale helps to accelerate the reduction of gradients.</p>
<p>In traditional machine learning and statistics, integration and standardization are essential; in the field of in-depth learning, we have also developed more methods of neutralization to help implement the declining gradients.</p>
<p>There are common ways.</p>
<ul>
<li>Based on polarization:&#36;\hat{x}=\frac{x-x_{\min&#125;&#125;{x_{\max}-x_{\min&#125;&#125;&#36; It causes the anomaly to be squeezed into a small compartment, so it should be done after processing the anomaly.</li>
<li>Maximum unified absolute value:&#36;\hat{x}=\frac x{x_{\max&#125;&#125;&#36;</li>
<li>z-scores convert:&#36;\hat{x} = \frac&#123;&#123;x-\mu&#125;&#125;{\sigma}&#36; </li>
<li>Ten times zoom: &#36;hat==&lt;1&#36;</li>
<li>Robust scaling ：&#36;x_{scaled}=\frac{x-median(x)}{IQR}&#36;</li>
<li>Paradigm homogenization:&#36;\tilde{x}=\frac{x}{\left|x\right|_{2&#125;&#125;&#36;</li>
</ul>
<h4>Data Merge</h4>
<p>Dispersion reduces the size of the category by converting the data from continuous to multiple disaggregated indicators, and thus reduces the cost of modelling operations.</p>
<p>Data amalgamation is another fragmentation of discrete data. Sometimes there are too many categories of disaggregated data and there may be hundreds of them, so that the use of characterization codes leads to too many variables.</p>
<p>The goal of data consolidation is to reduce the number of variable categories. Common methods include:</p>
<ul>
<li>Manual consolidation of certain categories based on our understanding of variables</li>
<li>For those categories whose share is less than 20 per cent (customary rules), they are all merged into a OTHERS category, which can also be used as a supplement, i.e.<strong>Slight Category Merge</strong></li>
</ul>
<h4>Feature Encoding</h4>
<p>The feature coding is about the idea that the classification emerges from variables.</p>
<h4>Characteristic tectonics: Crossing and coordinates</h4>
<p>Characteristic tectonics refers to the processing of the original data based on experience and the acquisition of features that are more meaningful to the model. It requires you to spend a lot of time on sample data, thinking about the nature of the problem, the structure of the data and how best to use them in predictive models.
Calculateable features mean the calculation after input of an existing feature and the result as a new feature. It is more applicable to simple ML models (e.g. linear regression interactives); neural networks are generally considered to be self-explanatory in the learning of calculable features, although the introduction of calculable features into NN can also enhance effects. In general, three types of questions are studied: single characterization transformation (see section on data conversion), characteristic algorithm combinations, and the conceptual transformation of feature coordinates.</p>
<p><strong>Characteristic algorithm combination</strong>: The algorithm combination of features is also called&quot;Multiple Characteristics&quot;I don't know. If a combination of characteristics is meaningful in the problem area, it can be significantly added. For example, if there are two characteristics: length and width, the size (long times width) has the same effect. Adding such a combination would require a certain intuition, and adding all algorithmic combinations would be a mistake — because the characteristic combination is very large and should be guided by as much knowledge as possible.</p>
<p><strong>Descartes.</strong>: If both characteristics are always present (e.g. floor and room numbers), the use of the two characteristics as a separate feature enhances the signal for the ML algorithm. It's the reverse of decomposing complex features.</p>
<p><strong>Changing the concept of the feature coordinate system</strong>: If we think that two vector angles are meaningful, then the use of polar-coordinate systems can be considered to facilitate model learning of angle characteristics; if we want to reduce relevance, then the rotational-coordinate system completed from the Marseille distance can be considered. For color features, we also have RGB, HSV, etc., which have their own coding features.</p>
<h3>Identity selection and learning</h3>
<p><strong>Feature selection selects a small number of useful features from a large number of features</strong>I don't know. Not all characteristics are equal: attributes that are not relevant to the issue need to be deleted, some are more important than others and others are redundant. Characteristic selection is the automatic selection of a subset of the most important features of the problem.</p>
<p>The role of feature selection is:</p>
<ul>
<li>Simplify models and increase their interpretability</li>
<li>Reduction of training time</li>
<li>Avoiding dimensions of disaster</li>
<li>Improve model interoperability and reduce alignment</li>
</ul>
<p>There are three main types of features selected:</p>
<ul>
<li>Embedd: the learning algorithm itself contains features selection steps, such as decision tree;</li>
<li>(a) Encapsulation (wrapper): feature selection is integrated with the training process, and LVW (Las Vegas Wrapper) is used to perform characterizations with the result of a trained model;</li>
<li>Filtering (filter): characterization selection is completely independent of training, selection is based on the characteristics themselves and is not related to a learning device.</li>
</ul>
<h4>Filter</h4>
<p>The filter feature selection considers the link between the variable and the target variable to filter the feature, and the evaluation criteria are derived from the intrinsic nature of the data set itself. According to researchers, the more relevant features or feature sub-assemblies obtain a higher accuracy rate on the classification. The evaluation criteria selected for filter features are divided into four categories: distance measure, information measure, correlation measure and consistency measure.</p>
<p>Advantages: Algorithms are very common; training steps for taxonomics are omitted, and algorithms are of low complexity and therefore applicable to large-scale data sets; a large number of unrelated features can be quickly removed and pre-screeners as features are appropriate.</p>
<p>Disadvantages: Because the evaluation criteria are independent of specific learning algorithms, the selected feature subsets are generally lower than the Wrapper method in terms of classification accuracy.</p>
<p><strong>Relief Method</strong>: Using the Relief method allows for steady filtering (filter); it is essentially studying relevance. The method was designed.&quot;Relevant statistics&quot;to measure the importance of the characteristic. The statistical volume is a vector, each of which corresponds to an initial characteristic, while the importance of the characteristic subset is determined by the sum of the relevant statistical weights corresponding to each characteristic. Only one threshold will eventually be specified &#36;\tau&#36;, choose the match &#36;\tau&#36; Characteristics corresponding to large relevant statistical weights are sufficient; the number of characteristics to be selected can also be specified &#36;k&#36;, select the statistical weight of the relevant statistics &#36;k&#36; A signature.</p>
<p>This statistically relevant amount is calculated as: a given training set.<em>1,y_1),(\boldsymbol{x}<em>2,y_2),\ldots,(\boldsymbol{x}<em>m,y_m)}&#36;，对每个示例 &#36;\boldsymbol x_i&#36;，Relief 先在 &#36;\boldsymbol x_i&#36; 的同类样本中寻找其最近邻 &#36;x</em>It's called&quot;Guessing the nearest neighborhood.&quot;(near-hit), from &#36;x_i&#36; \bardsymbol{</em>It's called&quot;Wrong neighborhood.&quot;(near-miss) The relevant statistics correspond to properties &#36;j&#36; . The value is
&#36; \delta=sum</em>}mathrm{diff} (x j), x i,\mathrm{j}^mathrm{diff} (x i^j}, x mathrm{nm}^j)^:
of which &#36;x_a^j&#36; For sample &#36;x_a&#36; In Properties &#36;j&#36; above the value,&#36;\mathrm{diff}(x_a^j,x_b^j)&#36; Depends on properties &#36;j&#36; Type: if discrete, &#36;x_a^j=x_b^j&#36; 0, otherwise 1; if continuous, &#36;|x_a^j-x_b^j|&#36;I don't know. Attention. &#36;x_a^j, x_b^j&#36; Regulated to [0,1] slots.</p>
<p>It's essentially a calculation of a characteristic in&quot;Wrong guess.&quot;and&quot;Yeah.&quot;Whether there is a clear difference between the two: if there is, there is an increase in the relevant statistical weight, and the final average of each sample is the final output. Relief, which is designed for the issue of the second classification, expands the variant Relief-F to deal with the issue of multi-classifications, although they have little effect on qualitative self-variant.</p>
<h4>Seal</h4>
<p>Wrapper features select the performance of learning algorithms to evaluate the merits of the feature subset. For the feature subset to be evaluated, the Wrapper methodology requires the training of a taxonomyr (which needs to be designated by a person) to evaluate the feature subset based on its performance.</p>
<p>Advantages: The feature subsets found by the Wrapper method are usually better classified than the Filter method.</p>
<p>Disadvantages: The selection of features for the Wrapper method is less common and requires a re-selection when learning algorithms are changed; because each evaluation subset is trained and tested with a taxonomy, algorithm calculations are highly complex, especially for large data sets, and are implemented for a long time.</p>
<p>Specific methods of encapsulation are often studied by means of search, such as:</p>
<ul>
<li>Recursive feature elimination method</li>
<li>Back-and-back search methods for greedy thoughts</li>
<li>Random Search Method</li>
</ul>
<p><strong>Stable Selection</strong>: Stability choice is based on<strong>Secondary sampling and selection algorithm (training model)<strong>In combination, the choice can be regression, classification SVM or similar algorithms. The principle is achieved: the training model is run on different feature subsets, repeats and eventually aggregates the results of the feature selection. For example, it is possible to measure the frequency of a feature considered to be an important feature (the number of times it is selected as an important feature divided by the number of times it is tested in its subset). Ideally, important features will score close to 100 per cent; slightly weaker features will score non-0; the least useful features will score close to zero. It's not especially steep, it's different from the results of pure LASSO and random forests, and it shows that stability choices are not the same.</strong>Overcoming aggregation and understanding of data</strong>It helps. In general,<strong>A good feature is not divided into zeros because of similar characteristics.</strong>I don't know. Stability choices are often one of the best performance methods in many data sets and environments.</p>
<p><strong>Recursive Characteristic Elimination (Recursive Feature Integration, RFE)</strong>The RFE's main idea is to conduct multiple rounds of training using a base model (e.g., SVM or regression model): after each round of training, grade the lowest score on the basis of the coefficient of each feature, remove the smallest score, construct a new feature set with the remaining features, and conduct the next training round until all the features are over. The concrete steps are:</p>
<ol>
<li>Repeated construction of models (e.g. SVM or regression models);</li>
<li>Selecting the best (or worst) features (based on coefficients) and putting aside the selected features;</li>
<li>Repeats the steps 1 and 2 above on the remaining features until all the features are over.</li>
</ol>
<p>The order in which the features are removed in this process is the sort of characteristics that are actually a search.<strong>Best Feature Subset</strong>The greedy algorithm. The stability of RFE depends to a large extent on which model is chosen in the iterative selection:</p>
<ul>
<li>If ordinary returns are used, unregulated returns are unstable, and therefore RFE is unstable;</li>
<li>If the Ridge or Lasso model is used, a regularized return is stable, so RFE is stable.</li>
</ul>
<p><strong>Characteristic Value Sorting</strong>Theoretically, if, after sorting or disrupting a particular feature, the effect of the model (for both positive and negative) (forecast rating) is obvious, this characteristic can be shown to be important to the model; conversely, the absence of such a feature does not affect the model ' s effectiveness. Characteristic value sequencing is the method of design based on this idea.</p>
<h4>Embedded</h4>
<p>The embedded feature selection integrates the feature selection process with the learning device training process.<strong>They're done in the same optimisation.</strong>This means that characterizations are automatically made in the course of learning devices training. In the case of embedded methods, the best example of retrogression is:&#36;L_1&#36; Rectification is easier to dilute, so it's usually based on &#36;L_1&#36; Regularized learning methods are referred to as embedded feature selection methods, which are integrated and completed simultaneously with the learning device training process. Ridge, LASO, ElasticNet.<a href="/en/blog/2023/09/04/linear-regression-basics-notes/">Linear regression base</a>。</p>
<p>In addition to being based on the idea of regularization, embedded characterizations based on tree models are commonly used: decision tree, gradient up tree are embedded characterizations.<strong>Deep learning is also an embedded feature selection method.</strong>。</p>
<h4>Rare expression and dictionary learning</h4>
<p>The issue of identity selection is characterisation.&quot;Rareness&quot;, i.e. many columns in the matrix are not relevant to the current learning task, removing these columns through feature selection, enhancing model effectiveness, interpretability and reducing the difficulty of training.</p>
<p>Now let's think of another sort of thinness:&#36;D&#36; There are many zero elements in the corresponding matrix, but they do not exist as columns or rows. When samples have this thin expression, there are a number of benefits to the learning mission: highly thinness makes most problems linear, so SVM can have a good effect in this data; and, at the same time, thin samples do not create a huge storage burden, because the thin matrix already has a lot of efficient storage methods, so this is what we seek.</p>
<p><strong>It's good for us to build a model.</strong>(Of course too thin data is bad). So, if you give a data set, &#36;D&#36; Is it dense, i.e. ordinary, non-sorted data that can be converted into&quot;Rare expression&quot;(sparse representation) in the form of a rare advantage?</p>
<p>Obviously, there's no modern Chinese-language common vocabulary available for general learning assignments (e.g. image classification). We need to learn one of these.&quot;dictionary&quot;A suitable dictionary was found for a sample of common dense expression, which was converted into a suitable form of thin representation, thus simplifying learning tasks and reducing the complexity of models. It's usually called&quot;Dictionary Learning&quot;(dictionary learning)&quot;Rare Encoding&quot;（sparse coding）。</p>
<p>given data set &#36;{x_1,x_2,\ldots,x_m}&#36;The simplest form of dictionary learning is
&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;  &#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;    <em>2^2+\lambda\sum</em>I'm sorry.
of which &#36;\mathbf{B}\in\mathbb{R}^d\times k&#36; As a dictionary matrix,&#36;k&#36; The amount of vocabulary called dictionary is usually specified by the user.&#36;\boldsymbol{\alpha}_i\in\mathbb{R}^k&#36; It's a sample. &#36;x_i\in\mathbb{R}^d&#36; The thin expression. Obviously, the first thing to optimise is to hope that &#36;\boldsymbol\alpha_i&#36; It's a good remodel. &#36;x_i&#36;The second is hope. &#36;\alpha_i&#36; Try to be thin.</p>
<h2>Data Transformation</h2>
<p>In a lot of cases, we need to change the original data to better characterize the data; it's often because the original data has...</p>
<ul>
<li>Strong asymmetrical</li>
<li>Mass Offset Values</li>
<li>The idea of a simple model presents a large and non-accidental flaw.</li>
<li>The median and four-point differentials derived from the box chart reflect their relationship.
We want to make further analysis more rational by making some shape adjustments to the raw data.
It's a very broad concept, and it's a constant, and it's an order, and it's a variation, but we're just going to look at one of the special variations that he's asking for.</li>
<li>The median order remains unchanged</li>
<li>Continuous and smooth functions</li>
<li>The primary function can be constructed in the form of simple calculations.
<strong>Change is a cost, not always a benefit.</strong></li>
</ul>
<h3>& Change</h3>
<p>The definition, the transformation, has the following expression.
&#36;T p(x)=\begin{cases}ax^p+b&amp;,p\neq0\c\log x+d,p=0\end{cases}&#36;&#36;
里面的参数可以比较自由的选取 但是需要保证满足我们前面要求的变换的性质
幂变换有下面三种比较常用的形式
&#36;&#36;T_p(x)=\begin{cases}x^p,&amp;p&gt;0\\log x~,&amp;p=0\-x^p,&amp;p&lt;0\end{cases}.&#36;&#36;
&#36;&#36;T_p^*(x)=\begin{cases}\dfrac{x^p-1}{p},p\neq0\[2ex]\ln x&amp;,p=0\end{cases}&#36;&#36;</p>
<h3>Reason for the change.</h3>
<h4>Inline data requirements</h4>
<p>For example, we sometimes want to turn degrees centigrade into fahrenheit, and to logarithmize population numbers, which are based on the characteristics of the data themselves, often relying on our experience with this type of data, and they can be useful for further analysis.</p>
<h4>As symmetrical transformation</h4>
<p>The graphs and the coefficients tell us whether there's a systematic bias in the data, which we don't usually want, so we need to study how the data is symmetrical.</p>
<ul>
<li>Basic, we'll try square root changes, which is... &#36;p=\frac{1}{2}&#36;</li>
<li>If, however, there's not enough symmetry, we'll consider a logarithmic variant, which is... &#36;p=0&#36;</li>
<li>The logarithmic variants can also be used. &#36;p=\frac{1}{4}&#36;Replace</li>
</ul>
<h4>To eliminate dependence on the quartile and median</h4>
<p>Eradicating this dependency is often considered to be more appropriate for intuitive and exploratory analysis</p>
<h4>To construct the approximate linear relationship of the variable</h4>
<p>The approximation of linear models makes it easier to make simple drawings and analyze the differences.</p>
<h4>When is it worth changing?</h4>
<p>We don't have a strict standard.</p>
<ul>
<li>The maximum data value and the minimum data value are relatively large.</li>
<li>There's a habit of changing things like this.</li>
<li>It's much worse and it's a pattern.</li>
</ul>
<p> <strong>Change the broadest application from custom.</strong></p>
<h3>Box-Cox Transformation</h3>
<p>The point of Box-Cox transformation is to match the linear regression model in the front with all the assumptions we need.
It includes the Gauss-Markov hypothesis.
It's actually a way to fix the problem with regression diagnosis.
Note that the Box-Cox transformation is not a single variant, but a variant.
Overall expression of the Box-Cox variant
&#36; \begin{array}&amp;Y^{(\lambda)}=\begin{cases}\dfrac{Y^{\lambda}-1}{\lambda},&amp;\lambda\neq0,\\ln Y,&amp;\lambda=0, \end{cases}
Obviously, the core of the Box-Cox shift is the right choice.&#36;\lambda&#36;
In fact, we don't have a way of getting a stable.&#36;\lambda&#36; In practice, we choose a lot.&#36;\lambda&#36; There's been a lot of Box-Cox conversions, based on a variety of information standards. </p>
<ul>
<li>Square difference</li>
<li>Meet normality assumptions</li>
</ul>
<p>Maze distance changes: Maze distance is achieved by European range following the rotation axis, which eliminates self-relevance</p>
<h3>White</h3>
<p><strong>The Ma's distance shift eliminates self-relevance, the standardized shift changes the unit differences, and the combination of the two is bleaching.</strong>I don't know. The data generated by bleaching are well suited for modelling analysis of models, and more common methods of bleaching are PCA (main component analysis) and ZCA bleaching. Albinization is a way of changing data, but because of its advanced level, it is often discussed in feature projects.</p>
<p>In addition to bleaching, another common single feature conversion is the Sigmoid operation. It has an S-type function that preserves the variability of the intermediate part of the numeric domain and reduces the variability of both ends of the range:
&#36;&#36;\frac{1}{1+e^{-x&#125;&#125;&#36;&#36;</p>
