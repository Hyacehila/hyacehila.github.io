---
title: 'Sampling Survey: Sampling Design, Estimation Methods, and Data Quality'
title_zh: 抽样调查：抽样调查基础、抽样设计与估计方法
date: 2025-11-12 16:04:37 +0800
permalink: /blog/2025/11/12/sampling-survey-notes/
categories:
- Data Science
- Data Practice
tags:
- Statistics
- Sampling
- Survey Methods
excerpt: Covers survey foundations, sampling design, estimation methods, error control, and survey data quality.
description: Covers survey foundations, sampling design, estimation methods, error control, and survey data quality.
lang: en
translation_key: 2025-11-12-sampling-survey-notes
translation_status: machine
translation_source_hash: ab48f0b18377ca8cf9cec3ee359037a29d3fa1a4d873153fd033a460b6960a70
hidden: true
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<h2>Sample survey basis</h2>
<h3>On sample surveys</h3>
<p>Purpose of the sample survey: The general characteristics of the sample sample sample sample taken from the aggregate, i.e., the acquisition of statistical needs analysis data, were the first stages of the statistical analysis. </p>
<p>The questions in this article can also be addressed<a href="/en/blog/2023/11/05/descriptive-statistics-and-visualization-notes/">Descriptive statistics and visualization: data measurements, distance and related analysis</a>How the concept of a relatively close read together is developed in different contexts.</p>
<p><strong>In the age of big data, the significance of sample surveys has gradually diminished, and it is more worthwhile to discuss the need to obtain credible conclusions from unstable distributions, although it is also valuable to draw on the technical methods used to obtain better sampling data from big data.</strong></p>
<p>The sample surveys divided the analytical statistics into two categories,<strong>Sample and test data</strong>The former is in the real world, but it needs to be observed to get data. The latter is the data obtained by the test under controlled conditions. The former is required to be obtained using this technology, while the latter is used in the same way as the other one. <a href="/en/blog/2024/02/29/experimental-design-methods-notes/">Pilot design methodology</a> Get...</p>
<p>And we have a comprehensive survey methodology, including a comprehensive census, a statistical statement... that is, a survey of the whole population.</p>
<p>Relatively comprehensive survey, non-exhaustive survey<strong>Focused surveys, typical surveys, sample surveys</strong> ; Focusing on research to study units that, although small, account for a large proportion of the research phenomenon. Typical surveys are based on the surveyer’s choice of a number of typical subjects. Sample surveys are the most widely used methods in non-exhaustive surveys, from which the survey is based on the following:<strong>Some samples were taken from the entire research audience, in accordance with certain rules/procedures</strong>  </p>
<p>Sample surveys can reduce costs and time costs. Governments, businesses and a large number of market players need to obtain data through sample surveys and to purchase related services from statistical offices, market survey companies.</p>
<h3>Probability and non-probability sampling</h3>
<p>Non-probability sampling, also known as non-random sampling, has the core of ** “subjective choice”.** The sample relies on the subjective judgement of the researcher, convenience or voluntary participation of the interviewee. The probability of each unit being drawn in the overall picture is unknown and may even be zero.</p>
<p>Key approaches include:</p>
<ul>
<li>I want to take random samples, select the most accessible people as samples</li>
<li>Purpose/intentional sampling, judging and selecting those units that are “most typical” or “most capable of providing information” based on their own expertise and experience.</li>
<li>Quota sampling, an attempt to “memolify” a stratification in non-probability samples. The researchers first identified the “quota” (e.g. gender, age) of the sample for certain characteristics, so that it was consistent with the overall ratio.</li>
<li>Sample of volunteers, based on voluntary selection</li>
<li>Scramble snowball samples, first a few eligible initial interviewees, then a request for other eligible interviewees</li>
</ul>
<ul>
<li><strong>Advantages</strong>：<ul>
<li><strong>Low cost, fast.</strong>: Simple to operate, suitable for projects with limited budget and time.</li>
<li><strong>Wide applicability</strong>: For general purposes without sampling frames or for exploratory, qualitative studies.</li>
<li><strong>Access to specific groups</strong>: the only viable way to study “hidden” populations.</li>
</ul>
</li>
<li><strong>Disadvantages</strong>：<ul>
<li><strong>It's impossible to extrapolate the total.</strong>This is its fundamental, deadly weakness.</li>
<li><strong>Highly subjective.</strong>: The results are vulnerable to prejudice by researchers or participants.</li>
<li><strong>Question of representation</strong>: Samples are usually subject to systemic deviations.</li>
</ul>
</li>
</ul>
<p>Probability sample, also known as random sample, with the idea of “equal opportunity” at its core<strong>I'm sorry. Each unit of the total was one unit during the sampling process</strong>Known, non-zero** probability of being drawn.</p>
<ol>
<li><strong>The randomity principle</strong>: Sample selection is objective and random, excluding subjective interference by researchers.</li>
<li><strong>Derogability</strong>: The results of the sample can be mathematically used to extrapolate the sum as a result of randomness. This is the most valuable feature of a probability sample.</li>
<li><strong>Calculating Sample Errors</strong>: We can calculate the range of errors (between confidence) that may exist between the sample results and the overall real value by statistical formula.</li>
<li><strong>Scientifically strong.</strong>: The results are objective and re-proven, and are the gold standard for quantitative studies and scientific surveys.</li>
</ol>
<p>The main one is the method below.</p>
<ul>
<li><strong>Simple random sampling</strong>  It's completely randomly extracted from the general population.</li>
<li><strong>System sampling</strong> Calculate a sample interval, choose a random starting point, extract it from the interval</li>
<li><strong>Scattered Samples</strong> First, the grouping of the whole into several non-overlapping “layers” by a particular characteristic, and then the independent and random sampling of each layer (usually simple random or systematic)</li>
<li><strong>Group sample</strong> The group is divided into several “groups” (e.g., classes, communities, streets) and then randomly extracts several “clusters” and conducts a comprehensive survey of all units within the selected groups.</li>
<li>Multistage sampling mixes the methods in front, and the sampling is phased.</li>
</ul>
<h3>Basic concepts</h3>
<p>Before any sample survey can begin, researchers must clearly define several basic elements. These concepts are the cornerstone of the entire survey design, and it is essential to understand the differences and linkages between them. They are: total, sample units, sample frames and samples.</p>
<p>The whole is the ultimate target audience for our research. But in practice, we need to distinguish between two levels of overall: <strong>Overall objective</strong> It's us.<strong>Ideas</strong>Complete collection of all units that meet certain conditions that are intended to be studied. It is the “end group” that we want to extend the findings of the study.<strong>Overall research</strong>Yes.<strong>Actual</strong>We can take the whole sample book. It is usually a group represented by a specific “sampling box”.</p>
<p>Sample units are the total, used for sampling<strong>Basic modules</strong>I'm sorry. It's the thing we take from the whole thing. All sample units combined must cover the entire study in a complete manner and there can be no overlap (i.e. each element of the total is only a sample unit). The selection of sampling units depends on the purpose of the study, which is not necessarily a single person.</p>
<p>The sample frame is<strong>Lists, maps or any form of list of all sample units in the study</strong>I'm sorry. It is our “operational manual” or “maps” for actual sampling. The perfect sampling frame should be satisfied: complete, non-duplication, non-duplication, accurate.</p>
<p>The samples were actually extracted from the sampling frame based on sampling methods.<strong>The group of sample units.</strong>I'm sorry. It is the specific object that we use to carry out data analysis and research.<strong>Sample Volume</strong>: Number of units included in the sample, usually in letters <strong>n</strong> - Show.<strong>Sample ratio</strong>: Sample quantity as a proportion of total research, calculated as <strong>f = n / N</strong>(where N is the overall scale of the study)</p>
<p><strong>The total and sample in the sample survey are completely different from the mathematical statistical definition, and the overall is unlimited and has a defined distribution for mathematical statistics, but the overall no-distribution assumptions and limited number of samples in the sample survey are not guaranteed by I.I.D.</strong></p>
<h3>Sample error versus non-sampling error</h3>
<p>In any sample, we extrapolate the sum of the results from the sample, which always differs from the actual situation in the general picture. This general difference is called<strong>General survey error</strong>It consists of two main categories:<strong>Sample error</strong>and<strong>Non-sampling error</strong>。</p>
<p>The sample error is...<strong>The error that came about just because we only looked at part of it, not all of it.</strong>I'm sorry. Only in probability samples can we quantify this "lucky" by using random principles.
Component. The larger the sample, the more stable the sample is to the general, and the smaller the sampling error.</p>
<p>To measure a good or bad estimate, traditional statistics typically use deviations, differences, and average errors MSE to measure an estimate. Modern statistical learning techniques do not take the latter into account.</p>
<p>Non-sampling error is<strong>Errors in all the other components of the survey except the sample itself</strong>I'm sorry. Such errors are more subtle and may exist in both censuses and sample surveys.<strong>Source</strong>: Investigation of various human or systemic failures in the design, implementation, data processing, etc. It's inevitable and it's hard to measure.</p>
<h3>Implementation steps</h3>
<p><strong>Step 1: Clear definition of the objectives and overall objectives of the survey</strong></p>
<p><strong>Core issues</strong>The blogger says: "What are we looking at?" Who do we want to extend the conclusions to?</p>
<p><strong>Step 2: Build or evaluate sampling frames</strong></p>
<p><strong>Core issues</strong>: Do we have a map? How good is this map?</p>
<p><strong>Step 3: Designing sampling schemes</strong></p>
<p><strong>Core issues</strong>: How do we “draw”? How many?</p>
<p><strong>Step 4: Design and testing questionnaire</strong></p>
<p><strong>Core issues</strong>: How do we ask for real and accurate information?</p>
<p><strong>Step 5: Conducting investigations and managing no response</strong></p>
<p><strong>Core issues</strong>: How can we ensure that investigations are conducted smoothly and that everyone is involved as far as possible?</p>
<p><strong>Step 6: Data collation, analysis and reporting</strong></p>
<p><strong>Core issues</strong>: How can meaningful conclusions be drawn from the data collected?</p>
<h2>Simple Random Sampling</h2>
<h3>SRS</h3>
<p>Simple random sampling (Simple Random Sampling, SRS) is the most basic probability sampling method. In this methodology:</p>
<ul>
<li><strong>Each sample unit</strong>The same opportunities (whether individual, family, company, etc.) are available. Importability random selection</li>
<li><strong>Sample selection</strong>: randomly sampled fixed sizes from the aggregate, each of which has the same probability of selection as the possible sample combination</li>
<li><strong>No returned sample</strong>: Each of the units drawn will not be selected again.
SRS can be achieved by generating random numbers.</li>
</ul>
<h3>Average SRS, variance and standard error</h3>
<p>The difference between the sample average and the sample formula is a key tool for estimating the overall average and the overall difference.
<strong>Sample average</strong>：
&#36;&#36;\bar{y} = \frac{1}{n} \sum_{i=1}^{n} y_i&#36;&#36;
<strong>Sample variance</strong>：
&#36;&#36;s^2 = \frac{1}{n-1} \sum_{i=1}^{n} (y_i - \bar{y})^2&#36;&#36;
An important feature of simple random sampling is the average sample value and sample size. Bad<strong>No bias</strong>, that is, their expectations are equal to the overall average and the overall variance, respectively. This means that the averages and squares calculated through sampling do not systematically deviate from the true values of the whole on average.</p>
<p>The standard error is the average deviation of the sample average from the overall average, and the average deviation from the average is measured by the average difference.<strong>Sample average</strong>The error can be passed.<strong>Standard Error</strong>The Quantification of the Quality of the Quality of the Quality of the Quality of the Quality of the Quality of the Quality of the Quality of the Quality of the Quality of the Quality of the Quality of the Quality of the Quality of the Quality of the Quality of the Quality of the Quality of the Quality of the Quality of the Quality of the Quality of the Quality of the Quality of the Quality of the Quality of the Quality of the Quality of the Quality of the<strong>Standard Error</strong>：
&#36;&#36;\text{S.E.}(\bar{y}) = \frac{s}{\sqrt{n&#125;&#125;&#36;&#36;
of which&#36;s&#36; It's a standard deviation from the sample.&#36;n&#36; The smaller the standard error, the closer the average sample is to the overall average, and therefore the larger the sample size reduces the standard error and thus the accuracy of the estimate.</p>
<h3>Trust Interval, CI</h3>
<p>The range used to estimate the overall parameter (e.g., average) provides a range, indicating the likelihood that the overall parameter will fall into that range ... <strong>Standard error (S.E.)</strong> and <strong>Sample average</strong>, we can calculate the confidence interval of the overall average
&#36;&#36;置信区间 = \bar{y} \pm Z \times \text{S.E.}&#36;&#36;
Of which:</p>
<ul>
<li>&#36;\bar{y}&#36; is the sample average.</li>
<li>&#36;Z&#36; is the fraction of the standard normal distribution, depending on the confidence required degrees</li>
<li>&#36;\text{S.E.}&#36; is standard error.</li>
</ul>
<p>In simple random samples, in addition to estimating the overall average, we can estimate the total amount (e.g., total income, total production, etc.) by way of a sample. For total estimates, use the following formula:
&#36;&#36;
\hat{Y} = \frac{N}{n} \sum_{i=1}^{n} y_i
&#36;&#36;
I'm not sure.&#36;N&#36; In general,&#36;n&#36; It's a sample size.&#36;y_i&#36; is the value observed in the sample unit.</p>
<p><strong>Standard error in total sample:</strong>
&#36;&#36;
\text{S.E.}(\hat{Y}) = \frac{N}{n} \times \text{S.E.}(\bar{y})
&#36;&#36;
This formula indicates that the standard error of the total is proportional to the standard error of the sample average.</p>
<h2>Estimated ratio</h2>
<p>This chapter is structured around the estimation of the proportion in the sample survey and focuses on methods such as single-scale estimation, multi-category analysis, independent sample consolidation and sub-total analysis. Unlike stratification, this chapter highlights the fact that it is not possible to distinguish the whole from the different sub-totals before the sampling. These methods are widely used in the areas of political opinion polls, biostatistics, market research and ecology.</p>
<h3>Overall</h3>
<p>In the overall scale sample, we are concerned about the proportion of individuals with certain characteristics in the total. Define indicator variables if a simple random sample of n size is taken from the sum of N size:
I'm sorry.
♪ I'm not sure I'm gonna be able to do this ♪
The \begin{cases}
One. &amp; \text{if i\text{units have target characteristics} \
Photo by Flickr user @un.org &amp; \text{otherwise}
The next thing I know, I'm not sure.
I'm sorry.</p>
<p><strong>Overall parameters:</strong></p>
<ul>
<li>&#36;C = \sum_{i=1}^{N} y_i&#36; : Total number of units with this characteristic</li>
<li>&#36;P = \frac{C}{N}&#36; : Overall</li>
<li>&#36;Q = 1 - P&#36; : Percentage not having this characteristic</li>
<li>&#36;S^2 = \frac{N}{N-1} PQ&#36; : Total variance</li>
</ul>
<p><strong>Sample statistics:</strong></p>
<ul>
<li>&#36;c = \sum_{i=1}^{n} y_i&#36; : Units in the sample with this characteristic</li>
<li>&#36;p = \frac{c}{n} = \bar{y}&#36; : Sample ratio, yes &#36;P&#36; No-Effort estimate</li>
</ul>
<p><strong>Sampling distribution characteristics:</strong></p>
<ul>
<li>Number of units in the sample with characteristics &#36;c&#36; Subscribe to hypergeometric distribution</li>
<li>When? &#36;N&#36; When it's big, the hypergeometric distribution is almost two-dimensional.</li>
<li>&#36;p&#36; The difference is:&#36;\mathrm{Var}(p) = \left(1 - \frac{n}{N}\right) \frac{PQ}{n}&#36;</li>
</ul>
<p><strong>Difference estimate:</strong></p>
<p>&#36;&#36;
\hat{v}(p) = \left(1 - \frac{n}{N}\right) \frac{pq}{n-1}
&#36;&#36;</p>
<p><strong>Standard error:</strong></p>
<p>&#36;&#36;
\mathrm{SE}(p) = \sqrt{\hat{v}(p)}
&#36;&#36;</p>
<p>Application of the overall ratio estimate: Mark the re-capture method, step as follows</p>
<ol>
<li>First capture &#36;M&#36; Animals only. Mark and put back to the original habitat.</li>
<li>After a while, they catch again. &#36;n&#36; Animals only</li>
<li>Calculate the number of animals marked in it &#36;c&#36;
Based on the overall ratio,
&#36;&#36;
p = \frac{c}{n} \text{ 估计了总体中标记动物的比例 } \frac{M}{N}
&#36;&#36;
Equivalent through &#36;\frac{c}{n} = \frac{M}{N}&#36;, obtains a rectangular estimate of the total size:<br>&#36;&#36;
\hat{N} = \frac{Mn}{c}
&#36;&#36;
We'll add amendments to the application.
&#36;&#36;
\hat{N} = \frac{(n + 0.5)M}{c + 0.5}
&#36;&#36;</li>
</ol>
<h3>Calculations between confidence zones and sample demand</h3>
<p>For the overall ratio &#36;P&#36;When the sample is large enough,&#36;p&#36; The sample distribution is almost subject to the normal distribution. It's built on the very limited logic of the center. &#36;1 - \alpha&#36; Trust interval:
&#36;&#36;
Z = \frac{p - P}{\sqrt{\hat{v}(p)&#125;&#125; \stackrel{d}{\approx} N(0,1)
&#36;&#36;
For 95% confidence level:
&#36;&#36;
P\left(-1.96 \leq \frac{p - P}{\sqrt{\hat{v}(p)&#125;&#125; \leq 1.96\right) \approx 0.95
&#36;&#36;
The modular is trusted:
&#36;&#36;
\left[ p - 1.96 \sqrt{ \left(1 - \frac{n}{N}\right) \frac{pq}{n-1} },; p + 1.96 \sqrt{ \left(1 - \frac{n}{N}\right) \frac{pq}{n-1} } \right]
&#36;&#36;</p>
<p>When the total is large (&#36;N \to \infty&#36;In the following:
&#36;&#36;
\left[ p - 1.96 \sqrt{ \frac{pq}{n} },; p + 1.96 \sqrt{ \frac{pq}{n} } \right]
&#36;&#36;</p>
<hr>
<p>In designing the survey, an appropriate sample volume needs to be determined to achieve the pre-set estimated accuracy. Sample quantities are usually calculated based on the width of the confidence zone.</p>
<p>For 95% confidence level, require error threshold not to exceed &#36;e&#36;：</p>
<p>&#36;&#36;
1.96 \times \sqrt{\frac{PQ}{n&#125;&#125; \leq e
&#36;&#36;</p>
<p>Because &#36;PQ \leq 0.25&#36;♪ When ♪ &#36;P = 0.5&#36; The most conservative estimate is:</p>
<p>&#36;&#36;
n \geq \frac{(1.96)^2 \times 0.25}{e^2}
&#36;&#36;</p>
<p>For example, the error limit is ±3%:</p>
<p>&#36;&#36;
n \geq \frac{(1.96)^2 \times 0.25}{(0.03)^2} \approx 1067
&#36;&#36;</p>
<p>That's why many polls use about 1100 samples, claiming precision at > 3%.</p>
<p>When the overall level is limited, limited overall correction is required:
&#36;&#36;
n = \frac{n_I}{1 + \frac{n_I}{N&#125;&#125;
&#36;&#36;
of which &#36;n_I&#36; The sample size is required on the basis of an unlimited overall assumption.</p>
<p><strong>Why did we introduce the sample count into the scale estimate? He needed a margin. Limit&#36;e&#36; It is only a more fixed requirement in the scale estimate, and other confidence-between problems can be similarly calculated to calculate sample demand, and Z distribution is very common in sample surveys, so many give estimates averages and variance, and many start to consider the confidence-building and sample volume issues.</strong></p>
<h3>Analysis in multi-category situations</h3>
<h4>Multi-category basic analysis</h4>
<p>When the total contains &#36;k&#36; Each category accounts for a certain proportion of the total, and we need to estimate both those proportions and their interrelationships, when one category is one (e.g., multiple candidates for election). Set&#36;C_1, C_2, \ldots, C_k&#36; Indicate the number of categories, satisfied &#36;\sum_{j=1}^{k} C_j = N&#36;I'm sorry. The number of categories in the sample is &#36;c_1, c_2, \ldots, c_k&#36;,Fulfilled &#36;\sum_{j=1}^{k} c_j = n&#36;</p>
<p>Sample distribution is subject to multivariant hypergeometric distribution, with a probabilistic mass function:</p>
<p>&#36;&#36;
p(x_1, \ldots, x_k) = \frac{ \binom{C_1}{x_1} \binom{C_2}{x_2} \cdots \binom{C_k}{x_k} }{ \binom{N}{n} }
&#36;&#36;</p>
<p>I'm sorry. &#36;j&#36; Total ratio of category:&#36;P_j = \frac{C_j}{N}&#36;</p>
<p>Sample scale:&#36;p_j = \frac{c_j}{n}&#36;  Sample ratio is the estimate of the overall ratio.</p>
<p>Specimen:
&#36;&#36;
\mathrm{Var}(p_j) = \left(1 - \frac{n}{N}\right) \frac{N}{N-1} \frac{P_j(1 - P_j)}{n}
&#36;&#36;</p>
<p>When? &#36;N&#36; In large numbers, the multivariant hypergeometric distribution can be similar to multiple distributions, with the variance being simplified:
&#36;&#36;
\mathrm{Var}(p_j) \approx \frac{P_j(1 - P_j)}{n}
&#36;&#36;</p>
<p>Difference estimate:
&#36;&#36;
\hat{v}(p_j) = \left(1 - \frac{n}{N}\right) \frac{p_j(1 - p_j)}{n-1}
&#36;&#36;</p>
<p>Formulas can be used directly when the quantity is estimated instead of the proportion
&#36;&#36;
c_i = N p_i
&#36;&#36;
His difference is...
&#36;&#36;
\mathrm{Var}(c_i) = N^2 \mathrm{Var}(p_i) = N^2 \left(1 - \frac{n}{N}\right) \frac{N}{N-1} \frac{P_i(1 - P_i)}{n}.
&#36;&#36;</p>
<h4>Multi-category differentials</h4>
<p>In multi-category situations, we often need to estimate the difference between the two categories, such as the difference in the support rate for the two candidates. For Category &#36;i&#36; and &#36;j&#36; Difference in ratio:</p>
<p><strong>Estimate:</strong> &#36;&#36; p_i - p_j &#36;&#36;</p>
<p><strong>Specimen:</strong></p>
<p>&#36;&#36;
\mathrm{Var}(p_i - p_j) = \frac{N - n}{n(N - 1)} \left[ P_i(1 - P_i) + P_j(1 - P_j) + 2P_i P_j \right]
&#36;&#36;</p>
<p><strong>Difference estimate:</strong></p>
<p>&#36;&#36;
\hat{v}(p_i - p_j) = \frac{N - n}{N(n - 1)} \left[ p_i(1 - p_i) + p_j(1 - p_j) + 2p_i p_j \right]
&#36;&#36;</p>
<h4>Sample volume under multiple categories</h4>
<p>We're gonna need to figure out what we're gonna do. &#36;j&#36; Percentage of categories &#36;P_j&#36;, require confidence level &#36;1 - \alpha&#36; The estimated error is no more than &#36;e&#36;I'm sorry. That is:</p>
<p>&#36;&#36;
P\left( |p_j - P_j| \leq e \right) = 1 - \alpha
&#36;&#36;</p>
<p>When? &#36;N&#36; The blogger says:&#36;p_j&#36; Approximate to normal distribution, we can use the following variants (<strong>We all use the variance estimates and assume that&#36;N&#36;Largely in calculating confidence interval and sample volumes, when normal distribution assumptions can be used and the denominator is used&#36;n-1&#36;Replace with&#36;n&#36;</strong>)：
&#36;&#36;
z_{\alpha/2} \cdot \sqrt{ \frac{P_j(1 - P_j)}{n} } \leq e
&#36;&#36;</p>
<p>of which &#36;z_{\alpha/2}&#36; It's the top of the standard normal distribution. &#36;\alpha/2&#36; Bits (e. g. 95% confidence level) &#36;z_{0.025} = 1.96&#36;）</p>
<p>The solution is:
&#36;&#36;
n \geq \frac{ z_{\alpha/2}^2 \cdot P_j(1 - P_j) }{ e^2 }
&#36;&#36;</p>
<hr>
<p>When the difference between the two categories is estimated &#36;P_i - P_j&#36; , the error is not greater than &#36;e&#36;, the sample volume formula is:</p>
<p>&#36;&#36;
z_{\alpha/2} \cdot \sqrt{ \mathrm{Var}(p_i - p_j) } \leq e
&#36;&#36;</p>
<p>When? &#36;N&#36; Very often.&#36;n-1&#36;In exchange for...&#36;n&#36;)：</p>
<p>&#36;&#36;
\mathrm{Var}(p_i - p_j) \approx \frac{ P_i(1 - P_i) + P_j(1 - P_j) + 2P_i P_j }{n} = \frac{ P_i + P_j - (P_i - P_j)^2 }{n}
&#36;&#36;</p>
<p>Therefore:</p>
<p>&#36;&#36;
n \geq \frac{ z_{\alpha/2}^2 \cdot \left[ P_i(1 - P_i) + P_j(1 - P_j) + 2P_i P_j \right] }{ e^2 }
&#36;&#36;</p>
<h3>Group of independent samples</h3>
<p>In practical studies, there is a constant need to consolidate data from multiple independent surveys to improve the accuracy of estimates or to enhance the reliability of conclusions. Assumptions &#36;k&#36; An independent survey addresses the same response variables for the same target group.</p>
<p>Set &#36;p_i&#36; As No. &#36;i&#36; The percentage of samples surveyed,&#36;n_i&#36; For its sample volume, the combined estimate is:</p>
<p>&#36;&#36;
\hat{P} = \frac{ \sum_{i=1}^{k} n_i p_i }{ \sum_{i=1}^{k} n_i } = \frac{ \sum_{i=1}^{k} c_i }{ \sum_{i=1}^{k} n_i }
&#36;&#36;</p>
<p>This estimate is reasonable because of the sample size of each survey &#36;n_i&#36; It's not random, it's research design. The combined estimates are essentially weighted averages, with a positive ratio of weight to sample volume.</p>
<hr>
<p>When the total size is large, the difference in the combined estimate is:</p>
<p>&#36;&#36;
\mathrm{Var}(\hat{P}) = \frac{ \sum_{i=1}^{k} n_i^2 \mathrm{Var}(p_i) }{ n^2 }
&#36;&#36;</p>
<p>of which &#36;n = \sum_{i=1}^{k} n_i&#36;</p>
<p>Because &#36;\mathrm{Var}(p_i) \approx \frac{PQ}{n_i}&#36;, received:</p>
<p>&#36;&#36;
\mathrm{Var}(\hat{P}) \approx \frac{ \sum_{i=1}^{k} n_i^2 \cdot \frac{PQ}{n_i} }{ n^2 } = \frac{ PQ \sum_{i=1}^{k} n_i }{ n^2 } = \frac{PQ}{n}
&#36;&#36;</p>
<p>This suggests that the difference between the combined sample and the one size is equal to the one size. &#36;n&#36; the difference of the single sample.</p>
<hr>
<p>The variance is estimated to be:
&#36;&#36;
\hat{v}(\hat{P}) = \frac{ \hat{p} \hat{q} }{ n }
&#36;&#36;
of which &#36;\hat{p} = \hat{P} , \hat{q} = 1 - \hat{p}&#36;。</p>
<h2>General estimate</h2>
<h3>On sub-general issues</h3>
<p>In sample surveys, we often focus not only on the overall parameters but also on the parameters of the specific part (subtotal) of the overall picture. For example:</p>
<ul>
<li>In the population census, we may be concerned about income levels of different age groups</li>
<li>In agricultural surveys, the production of different cultivation methods may need to be understood</li>
<li>Health surveys may need to analyse the incidence of diseases by gender or occupational group</li>
</ul>
<p>The key features of such problems are:<strong>The overall composition consists of several non-overlapping parts, of which we are interested in one or more. These are referred to as sub-total or research domains.</strong></p>
<p>The sub-total estimation problem is divided into two main scenarios:</p>
<ol>
<li><p><strong>Scattered sampling:</strong></p>
<ul>
<li>Each subtotal is known and separated before the sample.</li>
<li>The sample can be done independently within each subtotal.</li>
<li>This is part of the stratification sample and will be discussed in a subsequent section.</li>
</ul>
</li>
<li><p><strong>After-action hierarchy:</strong></p>
<ul>
<li>I can't separate the whole sub-total before sampling.</li>
<li>First, take a simple random sample from the body.</li>
<li>Samples are then classified according to the characteristics of the sample unit</li>
<li>This chapter focuses on the size of the sub-group. &#36;N_j&#36; Unknown, balance the discussion of known</li>
</ul>
</li>
</ol>
<p><strong>Core differences</strong>: In the stratification sample, the subtotal is known and separated in advance; in the case discussed in this chapter, the subtotal is determined after the sampling.</p>
<p>This is common in actual investigations:</p>
<ul>
<li>It is not possible to classify the overall unit before the sample (e.g. by income level)</li>
<li>Lack of subtotal identifier information in sample frames</li>
<li>Afterward, to analyze the sub-groups that were not foreseen in the survey</li>
<li>The analytical needs of certain sub-groups were not taken into account in the survey design</li>
</ul>
<h3>Relevant theoretical framework for sub-total</h3>
<p>- Yes, I do. &#36;N&#36; unit, including &#36;J&#36; A non-overlapping subtotal:</p>
<ul>
<li>&#36;N_j&#36; : No. &#36;j&#36; Size of the total of individuals (unknown)</li>
<li>&#36;n&#36; : Simple random sample volume</li>
<li>&#36;n_j&#36; : in sample &#36;j&#36; Unit numbers for individual aggregates (random variables)</li>
<li>&#36;y_{ij}&#36; : No. &#36;j&#36; The first of the entire population. &#36;i&#36; Unit response variable values</li>
<li>&#36;\bar{Y}<em>j = \frac{1}{N_j} \sum</em>{i=1}^{N_j} y_{ij}&#36; ：第 &#36;j&#36; average for the entire population</li>
<li>&#36;Y_j = \sum_{i=1}^{N_j} y_{ij}&#36; : No. &#36;j&#36; Total number of individual units</li>
</ul>
<p>Draw size from the total &#36;n&#36; After a simple random sample:</p>
<ul>
<li>Sample is no. &#36;j&#36; Number of units per unit &#36;n_j&#36; It's random.</li>
<li><strong>Significant nature</strong>: Prove it, this &#36;n_j&#36; Unit composition in size &#36;N_j&#36; Simple random subsamples taken from the total</li>
</ul>
<h3>Estimated subtotal averages and variance</h3>
<p>I'm sorry. &#36;j&#36; The average value of the individual population is estimated at:
I'm sorry.
♪ I'm not gonna let you go ♪<em>j = \frac{1}{n_j} \sum</em>{i=1}^{n_j} y_{ij}
&#36;&#36;</p>
<p>This is the sub-average. &#36;\bar{Y}_j&#36; . That is:
&#36;&#36;
E(\bar{y}_j) = \bar{Y}_j
&#36;&#36;</p>
<hr>
<p>The difference in the sub-total mean estimate is:</p>
<p>&#36;&#36;
\mathrm{Var}(\bar{y}_j) = \left(1 - \frac{n_j}{N_j}\right) \frac{S_j^2}{n_j}
&#36;&#36;</p>
<p>of which &#36;S_j^2 = \frac{1}{N_j - 1} \sum_{i=1}^{N_j} (y_{ij} - \bar{Y}_j)^2&#36; No. No. &#36;j&#36; The difference of the individual total.</p>
<p>Because &#36;N_j&#36; Unknown. We can't use it directly. &#36;n_j / N_j&#36; As a sample percentage. However, it can be shown that:</p>
<p>&#36;&#36;
E\left( \frac{n_j}{N_j} \right) = \frac{n}{N}
&#36;&#36;</p>
<p>It shows that, although... &#36;n_j / N_j&#36; It's random in itself, but its expectations are equal to the overall sample ratio. &#36;n / N&#36;</p>
<hr>
<p>Therefore, the margin is estimated to be:
&#36;&#36;
\hat{v}(\bar{y}_j) = \left(1 - \frac{n}{N}\right) \frac{s_j^2}{n_j}
&#36;&#36;</p>
<p>of which &#36;s_j^2 = \frac{1}{n_j - 1} \sum_{i=1}^{n_j} (y_{ij} - \bar{y}_j)^2&#36; No. No. &#36;j&#36; The difference in the total sample.</p>
<blockquote>
<p><strong>Attention.</strong>: Here the overall sample ratio is used &#36;n/N&#36; Not the subtotal sample ratio. &#36;n_j / N_j&#36;, because &#36;N_j&#36; Unknown, if he knows, should be replaced accordingly.</p>
</blockquote>
<p><strong>If you want to build the confidence compartment, you can still use the normal distribution method. We have now given estimates of the average and the difference, and you can bring it in.</strong></p>
<h3>Estimates of totals</h3>
<p>Total Subsize &#36;N_j&#36; In unknown circumstances, it cannot be simply used &#36;N_j \times \bar{y}_j&#36; As total estimate, as &#36;N_j&#36; Unknown.</p>
<p>Introduction of supporting variables to address this problem &#36;U_i&#36;：</p>
<p>&#36;&#36;
U_i =
\begin{cases}
y_{ij} &amp; \text{if i\text{units belong to the }j\text{total} \
Photo by Flickr user @un.org &amp; \text{otherwise}
The next thing I know, I'm not sure.
I'm sorry.</p>
<p>There are:
&#36;&#36;
\bar{U} = \frac{1}{N} \sum_{i=1}^{N} U_i = \frac{1}{N} \sum_{i=1}^{N_j} y_{ij} = P_j \bar{Y}_j
&#36;&#36;
of which &#36;P_j = N_j / N&#36; No. No. &#36;j&#36; The overall proportion of each individual in the aggregate.</p>
<p>Based on the supporting variables, the total subtotal is estimated to be:
I'm sorry.
What's wrong with you?<em>j = N \bar{u} = \frac{N}{n} \sum</em>{i=1}^{n_j} y_{ij}
&#36;&#36;
这是一个无偏估计量，即：
&#36;&#36;
E(\hat{Y}_j) = Y_j
&#36;&#36;</p>
<p>Supporting variables &#36;U&#36; The overall variance is:</p>
<p>&#36;&#36;
S_u^2 = \frac{1}{N-1} \left[ \sum_{i=1}^{N} U_i^2 - N \bar{U}^2 \right] = \frac{N_j - 1}{N - 1} S_j^2 + \frac{N}{N - 1} P_j Q_j \bar{Y}_j^2
&#36;&#36;</p>
<p>of which &#36;Q_j = 1 - P_j&#36;。</p>
<p>The difference in the total estimated volume is therefore:</p>
<p>&#36;&#36;
\mathrm{Var}(\hat{Y}_j) = N^2 \mathrm{Var}(\bar{u}) = \frac{N^2(1 - f)}{n} S_u^2 \
= \frac{N^2(1 - f)}{n} \left[ \frac{N_j - 1}{N - 1} S_j^2 + \frac{N}{N - 1} P_j Q_j \bar{Y}_j^2 \right]
&#36;&#36;</p>
<p>of which &#36;f = n/N&#36; For sample ratio.</p>
<p>Sample variance &#36;s_u^2&#36; This can be expressed as:</p>
<p>&#36;&#36;
s_u^2 = \frac{1}{n-1} \sum_{i=1}^{n} (u_i - \bar{u})^2 = \frac{1}{n-1} \left[ (n_j - 1) s_j^2 + n p_j q_j \bar{y}_j^2 \right]
&#36;&#36;</p>
<p>of which &#36;p_j = n_j / n , q_j = 1 - p_j&#36;</p>
<p>Therefore, the variance in the total estimated volume is estimated to be:
I'm sorry.
♪ The world is so full of shit ♪<em>j) = \frac{N^2(1 - f)}{n(n-1)} \left[ \sum</em>{i=1}^{n_j} (y_{ij} - \bar{y}_j)^2 + n p_j q_j \bar{y}_j^2 \right]
&#36;&#36;
<strong>Summing the sum of the subtotals is the sum of the sum of the sum, and the sum of the sum of the sum of the sum of the sum of the sum of the sum of the sum of the sum of the sum of the sum of the sum of the sum of the sum of the sum of the sum of the sum of the sum of the sum of the sum of the sum of the objects of the individual subs, which, without much discussion, can be extended to the sum of the sum of the sum of the objects, and its form before simplification can be used in the case of the known size of the subtotal of the sum of the sum, which is very simple, and we do not need to give any additional formula.</strong></p>
<h3>General average value difference</h3>
<p>Set &#36;y&#36; The blogger says that the government is not a party to the law.&#36;\bar{Y}_i&#36; and &#36;\bar{Y}_j&#36; Subtotals, respectively &#36;i&#36; and &#36;j&#36; average.</p>
<ul>
<li><strong>Estimated</strong>：&#36;\bar{y}_i - \bar{y}_j&#36;</li>
<li><strong>Difference</strong>：&#36;\text{Var}(\bar{y}_i - \bar{y}_j) = \left(1 - \frac{n_i}{N_i}\right)\frac{s_i^2}{n_i} + \left(1 - \frac{n_j}{N_j}\right)\frac{s_j^2}{n_j}&#36;</li>
<li><strong>Difference estimate</strong>：&#36;v(\bar{y}_i - \bar{y}_j) = \left(1 - \frac{n_i}{N_i}\right)\frac{s_i^2}{n_i} + \left(1 - \frac{n_j}{N_j}\right)\frac{s_j^2}{n_j}&#36;</li>
</ul>
<p>If confidence is needed to continue using normality, the sample size is considered to be the same.</p>
<h3>Calculation of sample quantities</h3>
<p>The total mean of the estimated sub-value for the volume of samples is calculated, and the estimated number of samples is calculated as follows: &#36;i&#36; Average value of individual total &#36;\bar{Y}_i&#36;, is given a margin error &#36;d_i&#36; And confidence level &#36;1 - \alpha&#36; in the case of the &#36;n_i&#36; , and the formula is:</p>
<p>&#36;&#36;
n_i = \frac{N_i \cdot z_{\alpha/2}^2 \cdot S_i^2}{(N_i - 1)d_i^2 + z_{\alpha/2}^2 \cdot S_i^2}
&#36;&#36;</p>
<p>Of which:</p>
<ul>
<li>&#36;N_i&#36; No. No. &#36;i&#36; Size of the total individual (usually unknown)</li>
<li>&#36;S_i^2&#36; No. No. &#36;i&#36; Differences in the total number of individuals (usually unknown, pre-estimated)</li>
<li>&#36;z_{\alpha/2}&#36; It's the top of the standard normal distribution. &#36;\alpha/2&#36; Bits</li>
<li>&#36;d_i&#36; is the maximum error allowed</li>
</ul>
<p>If you want to study the ratio (specify of the second classification), you just need to replace the difference.
&#36;&#36;
n_i = \frac{N_i \cdot z_{\alpha/2}^2 \cdot p_i (1 - p_i)}{(N_i - 1) d_i^2 + z_{\alpha/2}^2 \cdot p_i (1 - p_i)}
&#36;&#36;</p>
<p>For the case of polykot in general, there are errors.<strong>Consider prioritizing the most concerned sub-totals</strong>, use<strong>Minimum Max Error Method</strong>, select the total sample volume below &#36;n&#36;, minimizes the maximum relative error in all subtotal estimates:</p>
<p>&#36;&#36;
\min_{n} \max_{i} \left( \frac{z_{\alpha/2} \cdot S_i}{\sqrt{n_i&#125;&#125; \sqrt{1 - \frac{n_i}{N_i&#125;&#125; \right)
&#36;&#36;</p>
<p>of which &#36;n_i&#36; No. No. &#36;i&#36; Total individual sample volume, compared to total sample volume &#36;n&#36; ♪ And the relationship is ♪ &#36;n_i = n \cdot W_i&#36;，&#36;W_i = N_i / N&#36; No. No. &#36;i&#36; Total per unit.</p>
<p>Or use the following:<strong>Practical approaches</strong></p>
<ol>
<li>Estimated overall ratio per sub-unit &#36;W_i&#36;</li>
<li>Identification of the required sample quantity for each subtotal &#36;n_i&#36;</li>
<li>Calculated total sample volume:&#36;n = \sum_{i=1}^{k} \frac{n_i}{W_i}&#36;</li>
</ol>
<p><strong>Example:</strong>: Assuming that the total consists of two sub-totals,&#36;W_1 = 0.7&#36;，&#36;W_2 = 0.3&#36;I'm sorry. To achieve the same precision, each of them is required &#36;n_1 = 100&#36;，&#36;n_2 = 150&#36;I'm sorry. The total sample volume is:</p>
<p>&#36;&#36;
n = \frac{100}{0.7} + \frac{150}{0.3} \approx 143 + 500 = 643
&#36;&#36;</p>
<h2>Scattered sampling methods and their application</h2>
<h3>Basic concepts related to stratification sampling</h3>
<p>In practical issues, the general body can generally be divided into several non-overlapping sub-totals by characteristics (e.g., different regions, different categories, different time periods, etc.) and these are referred to as the general body of the individual.&quot;Layer&quot;(strata). When there is a clear heterogeneity in the overall picture, simple random sampling may result in a lack of representativeness of the sample, thereby reducing the accuracy of the estimate. To address this problem, statisticians have proposed a stratification sampling method.</p>
<p>The basic idea of the stratification sample is to first divide the whole into G layers that do not overlap and cover the whole population (strata), then to conduct the separate (usually simple random) sampling within each layer, and finally to merge the results of the stratification into a layer weight that gives estimates of the overall parameters.</p>
<p>Stratification samples significantly reduce the variance of the estimate when the layer is of high homogeneity and the layer is of high heterogeneity. And it can provide data directly at the research level and, under existing formations, better survey implementation.</p>
<p>Gives the symbol of some sort of stratification sample</p>
<ul>
<li>&#36;N&#36; : Total size</li>
<li>&#36;G&#36; : Layers</li>
<li>&#36;N_g&#36; : Section&#36;g&#36;The size of the layer, &#36;g = 1, 2, \dots, G&#36;</li>
<li>&#36;N = \sum_{g=1}^{G} N_g&#36;</li>
<li>&#36;y_{gi}&#36; : Section&#36;g&#36;Layer 1&#36;i&#36;Unit response value</li>
<li>&#36;W_g = N_g / N&#36; : Section&#36;g&#36;Weight of Layers</li>
<li>&#36;\bar{Y}<em>g = \frac{1}{N_g} \sum</em>{i=1}^{N_g} y_{gi}&#36; : 第&#36;G&#36; level average overall</li>
<li>&#36;Y_g = N_g \bar{Y}_g&#36; : Section&#36;g&#36;Total total of layers</li>
<li>&#36;S_g^2 = \frac{1}{N_g - 1} \sum_{i=1}^{N_g} (y_{gi} - \bar{Y}_g)^2&#36; : Section&#36;g&#36;Overall variance of layers</li>
<li>&#36;\bar{Y} = \frac{1}{N} \sum_{g=1}^{G} \sum_{i=1}^{N_g} y_{gi} = \sum_{g=1}^{G} W_g \bar{Y}_g&#36; : Overall average</li>
<li>&#36;S^2 = \frac{1}{N - 1} \sum_{g=1}^{G} \sum_{i=1}^{N_g} (y_{gi} - \bar{Y})^2&#36; : General variance</li>
</ul>
<p>The overall variance can be broken down into:
&#36;&#36;
S^2 = \frac{1}{N - 1} \left[ \sum_{g=1}^{G} (N_g - 1) S_g^2 + \sum_{g=1}^{G} N_g (\bar{Y}_g - \bar{Y})^2 \right]
&#36;&#36;</p>
<p>In the upper form, the first is the sum of the squares (SSerror) and the second is the sum of the squares (SStrt). This reflects the rationale for the variance analysis:
&#36;&#36;
\text{Var}(Y) = E[\text{Var}(Y|X)] + \text{Var}[E(Y|X)]
&#36;&#36;
of which&#36;X&#36;is a layered variable.</p>
<p>We do simple random stratifications, and we do simple random stratifications in each stratification, so we give the mathematical symbols that follow.</p>
<ul>
<li>&#36;n_g&#36; : Section&#36;g&#36;Sample volume of layers</li>
<li>&#36;\bar{y}<em>g = \frac{1}{n_g} \sum</em>{i \in s_g} y_{gi}&#36; : 第&#36;g&#36;-level sample average</li>
<li>&#36;s_g^2 = \frac{1}{n_g - 1} \sum_{i \in s_g} (y_{gi} - \bar{y}_g)^2&#36; : Section&#36;g&#36;Sample differences for layers</li>
<li>&#36;\text{Var}(\bar{y}_g) = \left(1 - \frac{n_g}{N_g}\right) \frac{S_g^2}{n_g}&#36; : Section&#36;g&#36;Difference of the mean of the layer sample</li>
<li>&#36;v(\bar{y}_g) = \left(1 - \frac{n_g}{N_g}\right) \frac{s_g^2}{n_g}&#36; : Uneven estimate of the above differences</li>
</ul>
<p><strong>And here we're just thinking about each floor as a SRSOR, and it's easy to continue the previous section and draw conclusions directly.</strong></p>
<h3>Parameter estimation of a layer sample versus confidence interval</h3>
<h4>Parameter estimation of stratification samples</h4>
<p>Mean values for the layered sample (also estimated by Horvitz-Thompson):
I'm sorry.
♪ I'm not gonna let you go ♪<em>{st} = \sum</em>{g=1}^{G} W_g \bar{y}<em>g
I'm sorry.
The estimate is impartial, i.e. &#36;E (\y}</em>{st}) = \bar{Y}&#36;。</p>
<p>Specimen:
I'm sorry.
\text{Var} (\bar{y}<em>{st}) = \sum</em>{g=1}^{G} W_g^2 \text{Var}(\bar{y}<em>g) = \sum</em>{g=1}^{G} W_g^2 \left(1 - \frac{n_g}{N_g}\right) \frac{S_g^2}{n_g}
&#36;&#36;</p>
<p>The difference is estimated in no-sided terms:
I'm sorry.
v (\bar{y}<em>{st}) = \sum</em>{g=1}^{G} W_g^2 \left(1 - \frac{n_g}{N_g}\right) \frac{s_g^2}{n_g}
&#36;&#36;</p>
<p>Estimates of total:
I'm sorry.
What's wrong with you?<em>{st} = N \bar{y}</em>{st} = \sum_{g=1}^{G} N_g \bar{y}<em>g
&#36;&#36;
方差:
&#36;&#36;
\text{Var}(\hat{Y}</em>{st}) = N^2 \text{Var}(\bar{y}<em>{st}) = \sum</em>{g=1}^{G} N_g^2 \left(1 - \frac{n_g}{N_g}\right) \frac{S_g^2}{n_g}
&#36;&#36;</p>
<p>The difference is estimated in no-sided terms:
I'm sorry.
v (hat{Y}<em>{st}) = N^2 v(\bar{y}</em>{st}) = \sum_{g=1}^{G} N_g^2 \left(1 - \frac{n_g}{N_g}\right) \frac{s_g^2}{n_g}
&#36;&#36;</p>
<h4>Proportional issue of stratification samples</h4>
<p>In the survey, when the variable takes only 0 and 1 values (e.g., if it owns a product), we usually focus on the ratio parameter. Will respond in general terms &#36;y_{gi}&#36; Replace with the binary variable, and get:</p>
<ul>
<li>&#36;\bar{Y}_g \to P_g&#36; , &#36;S_g^2 \to \frac{N_g}{N_g - 1} P_g Q_g&#36; , of which &#36;Q_g = 1 - P_g&#36;</li>
<li>&#36;\bar{y}_g \to p_g&#36; , &#36;s_g^2 \to \frac{n_g}{n_g - 1} p_g q_g&#36; , of which &#36;q_g = 1 - p_g&#36;</li>
<li>&#36;\text{Var}(p_g) = \left(1 - \frac{n_g}{N_g}\right) \frac{N_g}{N_g - 1} \frac{P_g Q_g}{n_g}&#36;</li>
<li>&#36;v(p_g) = \left(1 - \frac{n_g}{N_g}\right) \frac{p_g q_g}{n_g - 1}&#36;</li>
</ul>
<p><strong>Overall rate:</strong>
&#36;&#36;
P = \sum_{g=1}^{G} W_g P_g
&#36;&#36;</p>
<p><strong>Sample estimate:</strong>
&#36;&#36;
p_{st} = \hat{P}<em>{st} = \sum</em>{g=1}^{G} W_g p_g
&#36;&#36;</p>
<p><strong>Specimen:</strong>
&#36;&#36;
\text{Var}(p_{st}) = \sum_{g=1}^{G} W_g^2 \text{Var}(p_g)
&#36;&#36;</p>
<p><strong>The difference is estimated in no-sided terms:</strong>
&#36;&#36;
v(p_{st}) = \sum_{g=1}^{G} W_g^2 v(p_g)
&#36;&#36;</p>
<h4>Trust interval of a stratification sample</h4>
<p><strong>All confidence-building issues are ultimately resolved by normal distribution, which is very easy because we have calculated the average of the estimates and the difference between the square and the square.</strong></p>
<p>For the overall average &#36;\bar{Y}&#36; It's... &#36;100(1 - \alpha)%&#36; Trust interval:
I'm sorry.
♪ I'm not gonna let you go ♪<em>{st} \pm z</em>{\alpha/2} \sqrt{v(\bar{y}_{st})}
&#36;&#36;</p>
<p>For total &#36;Y&#36; It's... &#36;100(1 - \alpha)%&#36; Trust interval:
I'm sorry.
What's wrong with you?<em>{st} \pm z</em>{\alpha/2} \sqrt{v(\hat{Y}_{st})}
&#36;&#36;</p>
<h3>Distribution of sample quantities</h3>
<p>Total sample quantity given &#36;n = \sum_{g=1}^{G} n_g&#36;The distribution of sample volumes between layers is a key issue. Common methods of allocation include:</p>
<h4>Proportional distribution</h4>
<p>Proportional distribution: &#36;n_g \propto N_g&#36;that is
&#36;&#36;
n_g = W_g n = \frac{N_g}{N} n
&#36;&#36;</p>
<p>The weights are equal in the ratio distribution, with the difference in the mean values of the stratification samples being:
I'm sorry.
\text{Var} (\bar{y}<em>{st}) = \left(1 - \frac{n}{N}\right) \frac{1}{n} \sum</em>{g=1}^{G} W_g S_g^2
&#36;&#36;</p>
<p>The difference between a stratification sample and a simple random sample is &#36;\text{Var}(\bar{y}) = \left(1 - \frac{n}{N}\right) \frac{S^2}{n}&#36;, so when
I'm sorry.
♪ I'm gonna be a little bit more like a little more than a little bit of a ♪ &lt; S2
I'm sorry.
When the stratification sample is more accurate than a simple random sample, this variation is easily proven by the use of the squared-difference angle, and they are close to parity only when the mean inter-stage difference is small.</p>
<h4>Optimal Distribution / Neyman Distribution</h4>
<p>When considering sampling costs, the best-placed target is: at total cost &#36;C = c_0 + \sum_{g=1}^{G} c_g n_g&#36; (of which) &#36;c_0&#36; It's a fixed cost.&#36;c_g&#36; No. No. &#36;g&#36; ) Under the pressure of the sample cost of the layer) &#36;\text{Var}(\bar{y}_{st})&#36;。</p>
<p>Neyman's solution is:
&#36;&#36;
n_g \propto \frac{N_g S_g}{\sqrt{c_g&#125;&#125;
&#36;&#36;
Specifically:
&#36;&#36;
n_g = n \cdot \frac{N_g S_g / \sqrt{c_g&#125;&#125;{\sum_{j=1}^{G} N_j S_j / \sqrt{c_j&#125;&#125;
&#36;&#36;
♪ When all &#36;c_g&#36; In the case of equivalence, simplify the text to read:
&#36;&#36;
n_g = n \cdot \frac{W_g S_g}{\sum_{j=1}^{G} W_j S_j}
&#36;&#36;
At this time, the difference in the average of the stratification samples is:
&#36;&#36;
V_{\text{Neyman&#125;&#125; = \frac{1}{n} \left( \sum_{g=1}^{G} W_g S_g \right)^2 - \frac{1}{N} \sum_{g=1}^{G} W_g S_g^2
&#36;&#36;
Compared to the percentage distribution:
&#36;&#36;
V_{\text{proportion&#125;&#125; - V_{\text{Neyman&#125;&#125; = \frac{1}{n} \sum_{g=1}^{G} W_g \left( S_g - \sum_{j=1}^{G} W_j S_j \right)^2 \geq 0
&#36;&#36;</p>
<p>This suggests that Neyman distribution is always better than or equal to proportional distribution. When? &#36;S_g&#36; The greater the difference, the more obvious the advantage is, because the top is about &#36;S_g&#36; Weights in the market &#36;W_g&#36; The difference below.</p>
<h4>Sample volume determined</h4>
<p>Assumptions &#36;N_g&#36; and &#36;S_g^2&#36; Known, use ratio distribution, request &#36;\text{Var} (\bar{y}<em>{st}) \leq V&#36;，其中 &#36;V.A. is the given value. Unlocking:
I'm sorry.
\text{Var} (\bar{y}</em>{st}) = (n^{-1} - N^{-1}) \sum_{g=1}^{G} W_g S_g^2 \leq V
&#36;&#36;
得：
&#36;&#36;
♪ The world is so full of shit ♪
I'm sorry.
(Assumptions) &#36;N = \infty&#36; (samples at the time)</p>
<p>For a limited total, the sample size required is:
&#36;&#36;
n = \frac{n_l}{1 + n_l / N}
&#36;&#36;</p>
<hr>
<p>Use Neyman distribution, request &#36;\text{Var} (\bar{y}<em>\st}\leq V&#36;. Square formula:
I'm sorry.
\text{Var} (\bar{y}</em>{st}) = n^{-1} \left( \sum_{g=1}^{G} W_g S_g \right)^2 - N^{-1} \sum_{g=1}^{G} W_g S_g^2 \leq V
&#36;&#36;</p>
<p>When? &#36;N = \infty&#36; Other Organiser
&#36;&#36;
n_l = \frac{\left( \sum_{g=1}^{G} W_g S_g \right)^2}{V}
&#36;&#36;</p>
<p>For a limited total:
&#36;&#36;
n = \frac{\left( \sum_{g=1}^{G} W_g S_g \right)^2}{V + N^{-1} \sum_{g=1}^{G} W_g S_g^2}
&#36;&#36;</p>
<h2>Ratio estimate versus regression estimate</h2>
<p>In sample surveys, we usually focus on parameters such as the overall average or the ratio. However, in practical application, there are many other important, limited overall parameters, such as median income, percentage of the population living below the poverty line, etc. This chapter will focus on two methods of estimating the ratio of the overall average:</p>
<ul>
<li><strong>Rate estimate (Ratio Estimation)</strong>: Ratio relationship between the use of supporting and target variables</li>
<li><strong>Return Estimates</strong>: Linear relationships between supporting and target variables</li>
</ul>
<p>These methods are particularly useful in the following cases, where we give examples:</p>
<ul>
<li>Need to estimate the ratio of the two overall averages</li>
<li>Total total needs to be estimated, but total size unknown</li>
<li>Need to use supporting information to improve the accuracy of the estimates</li>
<li>The estimates need to be adjusted to reflect demographic characteristics</li>
<li>Process non-response deviations</li>
</ul>
<h3>Study ratio parameters</h3>
<h4>Estimated percentage</h4>
<p>Consider a limited total, with two variables per sample unit responding &#36;x&#36; and &#36;y&#36;：</p>
<ul>
<li>&#36;X&#36;、&#36;Y&#36;: each indicates the total &#36;x&#36; and &#36;y&#36; Average</li>
<li>Ratio parameters:&#36;R = \frac{Y}{X}&#36;</li>
</ul>
<p>In simple random sample unreleased (SRSOR), sample size is &#36;n&#36;I'm sorry. Yeah. &#36;R&#36; Estimates are:
&#36;&#36;
\hat{R} = \frac{\bar{y&#125;&#125;{\bar{x&#125;&#125;
&#36;&#36;</p>
<h4>Difference in the estimated ratio</h4>
<p>Ratio estimate &#36;\hat{R}&#36; The difference is almost as follows:
&#36;&#36;
V(\hat{R}) = \left(1 - \frac{n}{N}\right) \frac{S_d^2}{n\bar{X}^2}
&#36;&#36;
of which &#36;S_d^2 = S_y^2 + R^2 S_x^2 - 2R S_{xy}&#36;, by definition &#36;d = y - Rx&#36;，&#36;\bar{d} = \bar{y} - R\bar{x}&#36; Producate.</p>
<p>Noting:
&#36;&#36;
\hat{R} - R = \frac{\bar{y&#125;&#125;{\bar{x&#125;&#125; - R = \frac{\bar{d&#125;&#125;{\bar{x&#125;&#125; \approx \frac{\bar{d&#125;&#125;{X}
&#36;&#36;</p>
<p>Therefore:
&#36;&#36;
V(\hat{R}) \approx \left(1 - \frac{n}{N}\right) \frac{S_d^2}{nX^2}
&#36;&#36;</p>
<p>Taking into account relevant factors &#36;\rho = \frac{S_{xy&#125;&#125;{S_x S_y}&#36;, the difference can also be:
&#36;&#36;
V(\hat{R}) \approx \left(1 - \frac{n}{N}\right) \frac{S_y^2 + R^2 S_x^2 - 2R\rho S_x S_y}{nX^2}
&#36;&#36;</p>
<h4>Ratio deviation</h4>
<p>Ratio estimate &#36;\hat{R} = \frac{\bar{y&#125;&#125;{\bar{x&#125;&#125;&#36; The deviation is:
&#36;&#36;
B(\hat{R}) = E(\hat{R}) - R = E\left( \frac{\bar{y} - R\bar{x&#125;&#125;{\bar{x&#125;&#125; \right) = E\left[ \frac{\bar{y} - R\bar{x&#125;&#125;{X(1+\epsilon)} \right]
&#36;&#36;
of which &#36;\epsilon = \frac{\bar{x} - X}{X}&#36;I'm sorry. Using Taylor to expand, the deviation became:</p>
<p>&#36;&#36;
B(\hat{R}) = \frac{\text{Cov}(\bar{x}, \bar{y}) - R \times V(\bar{x})}{X^2} = (1 - f) R \frac{(C_{xx} - C_{xy})}{n}
&#36;&#36;
of which &#36;C_{xx} = \frac{S_x^2}{X^2}&#36;，&#36;C_{xy} = \frac{S_{xy&#125;&#125;{XY}&#36;，&#36;f = \frac{n}{N}&#36;I'm sorry. When a sample &#36;n&#36; When it's bigger, it's negligible.</p>
<h4>Average error in ratio (MSE)</h4>
<p>When? &#36;n&#36; When larger, the deviation is negligible, MSE approximates the difference:
&#36;&#36;
\text{MSE}(\hat{R}) = E(\hat{R} - R)^2 = E\left( \frac{\bar{y} - R\bar{x&#125;&#125;{\bar{x&#125;&#125; \right)^2 \approx V(\hat{R}) = E\left( \frac{\bar{y} - R\bar{x&#125;&#125;{X} \right)^2
&#36;&#36;</p>
<p>Definitions &#36;d_i = y_i - Rx_i&#36;，&#36;\bar{D} = 0&#36;, the overall variance is:
&#36;&#36;
S_d^2 = \sum_{i=1}^{N} \frac{(y_i - Rx_i)^2}{N-1} = \sum_{i=1}^{N} \frac{[(y_i - Y) - R(x_i - X)]^2}{N-1} = S_y^2 + R^2 S_x^2 - 2R S_{xy} = S_y^2 + R^2 S_x^2 - 2R\rho S_x S_y
&#36;&#36;</p>
<p>The natural estimate of the variance is:
&#36;&#36;
v(\hat{R}) = (1 - f) \frac{s_d^2}{n\bar{x}^2}
&#36;&#36;
of which &#36;d_i = y_i - \hat{R}x_i&#36;，&#36;s_d^2&#36; These. &#36;d_i&#36; the sample difference. We can give you the details.&#36;R&#36; The confidence interval is:
&#36;&#36;
\hat{R} \pm 1.96 \sqrt{v(\hat{R})}
&#36;&#36;</p>
<h3>Application of ratio estimates in average and sum of estimates</h3>
<p>In some cases,&#36;x&#36; and &#36;y&#36; There is a clear positive correlation, such as land area and production. In addition, sometimes in the aggregate &#36;x&#36; is known in the average or sum. When? &#36;X&#36; When known, we can adjust the estimates by: &#36;Y&#36;：
&#36;&#36;
\hat{Y}_R = \left( \frac{X}{\bar{x&#125;&#125; \right) \bar{y}
&#36;&#36;
When? &#36;x&#36; and &#36;y&#36; This estimate may be more effective when it is close to a ratio, which we call the ratio estimate.&#36;\hat{Y}_R&#36; It's not impartial. It's almost equal to:
&#36;&#36;
\text{Var}(\hat{Y}_R) = X^2 \text{Var}(\bar{y}/\bar{x}) = X^2 \times (1 - f) \frac{S_y^2 + R^2 S_x^2 - 2R\rho S_x S_y}{nX^2} = (1 - f) \frac{S_y^2 + R^2 S_x^2 - 2R\rho S_x S_y}{n}
&#36;&#36;</p>
<p>The ratio estimate is smaller than the sample average only if:
I'm sorry.
R^2 S x^2 - 2R\rho S x S y &lt; 0
&#36;&#36;
&#36;&#36;
RS_x &lt; 2\rho S_y
&#36;&#36;</p>
<p>Equivalent:
I'm sorry.
\cdot S x &lt; 2\rho S_y
&#36;&#36;
&#36;&#36;
\frac{S_x}{X} &lt; 2\rho \frac{S_y}{Y}
&#36;&#36;</p>
<p>Therefore, for the ratio estimates, it is more effective:
I'm sorry.
\rho &gt; \frac{S_x / X}{2 S_y / Y} = \frac{CV(X)}{2 CV(Y)}
&#36;&#36;
&#36;CV(\bar{x})&#36; and &#36;CV(\bar{y})&#36; Absolute values do not usually vary significantly. So when &#36;rho &gt; At &#36;1/2 a ratio is estimated to be more effective than SRS.</p>
<p>In short: when &#36;x&#36; and &#36;y&#36; The ratior is more suitable for estimation than the sample average when there is a strong positive correlation &#36;y&#36; Overall average.</p>
<p>&#36;\hat{Y}_R&#36; The variance is estimated to be:
I'm sorry.
v (hat{Y}<em>R) = (1 - f) \frac{s_y^2 + \hat{R}^2 s_x^2 - 2\hat{R}s</em>{xy&#125;&#125;{n}
&#36;&#36;</p>
<h4>Why use ratio estimates</h4>
<p>When the parameter is a ratio per se, for example, the average juice content per apple.</p>
<p>As a secondary variable &#36;x&#36; & Study Variables &#36;y&#36; Highly relevant and overall &#36;x&#36; The ratio estimates provide more accurate estimates than simple sample averages when the sum or average of the totals is known.</p>
<p><strong>Total estimated, but unknown</strong></p>
<p><strong>For example:</strong> There's a bunch of apples, and we'd like to estimate the total number of apple juices. Set:</p>
<ul>
<li>&#36;y_1, \dots, y_n&#36;: the amount of juice per apple in the sample</li>
<li>&#36;x_i&#36;The weight of each apple in the sample,&#36;\bar{x}&#36; It's the average mass of the sample.</li>
<li>&#36;N&#36; It's hard to count, so... &#36;N\bar{y}&#36; Hard to get directly</li>
<li>But the total weight of the whole batch of apples. &#36;X&#36; Easy to access.</li>
</ul>
<p>The number of apples can be estimated. &#36;X / \bar{x}&#36;The total volume of apple juice can be estimated at:
&#36;&#36;
\hat{Y}_r = \frac{\bar{y&#125;&#125;{\bar{x&#125;&#125; X
&#36;&#36;</p>
<p><strong>Adjustment of estimates to reflect demographic aggregates</strong></p>
<p><strong>Example:</strong> There are 400 students in a university, taking a sample of 400 SRS:</p>
<ul>
<li>240 women and 160 men in samples</li>
<li>84 women and 40 men are planned to work in teaching</li>
</ul>
<p><strong>Objectives:</strong> The total number of students who are planned to become teachers is estimated.</p>
<p><strong>Estimator 1:</strong> Use SRS information only
&#36;&#36;
N\bar{y} = 4000 \times \frac{124}{400} = 1240
&#36;&#36;</p>
<p><strong>Estimator 2:</strong> Integration of demographic information (schools with 2700 women and 1,300 men)
&#36;&#36;
\frac{84}{240} \times 2700 + \frac{40}{160} \times 1300 = 1270
&#36;&#36;</p>
<p><strong>Key points:</strong></p>
<ul>
<li>Application of estimated rates within each gender group</li>
<li>Sixty per cent of the samples were female, but 67.5 per cent of the total were female. Estimators adjusted to better reflect demographic ratios</li>
</ul>
<p><strong>Process non-response deviations</strong></p>
<p><strong>Example:</strong> Enterprise sample</p>
<ul>
<li>&#36;y_i&#36;: enterprises &#36;i&#36; Expenditure on health insurance</li>
<li>&#36;x_i&#36;: enterprises &#36;i&#36; Number of employees known</li>
<li><strong>Objectives:</strong> Estimated expenditure for general insurance</li>
</ul>
<p><strong>Estimator 1:</strong> &#36;N\bar{y}&#36;</p>
<ul>
<li>Companies with fewer employees are unlikely to respond to the survey</li>
<li>&#36;y_i&#36; and &#36;x_i&#36; Proportional</li>
<li>Estimator 1 overestimates total insurance expenditure. &#36;t_y&#36;</li>
</ul>
<p><strong>Estimator 2:</strong> &#36;X \frac{\bar{y&#125;&#125;{\bar{x&#125;&#125;&#36;</p>
<ul>
<li>Because of the greater likelihood of a response from a company with a large number of employees, &#36;X/\bar{x} &lt; N&#36;</li>
<li>Therefore, the ratio of total health insurance expenditure may be expected to compensate for the lack of impact of companies with small staff Reactions</li>
</ul>
<h3>Regressive estimate in simple random sample</h3>
<h4>For the return estimate</h4>
<p>In survey samples, regression estimates are a statistical technique that uses information from supporting variables to improve the accuracy of estimates. When we study a variable,&#36;y&#36;When (e.g. income, production, etc.) associated supporting variables are often available&#36;x&#36;Information (e.g. population, area, etc.). When x has linear relationships with y and the overall parameters of x (e.g., aggregate average or sum) are known, we can use this relationship to improve the estimates of y.</p>
<p>The core of the regression estimates is the use of the following information:</p>
<ul>
<li><strong>Relationship of supporting and target variables:</strong> Assumptions &#36;y&#36; and &#36;x&#36; Linear relationship exists &#36;E(y) = \beta_0 + \beta_1 x&#36;</li>
<li><strong>General information on the supporting variable:</strong> Known &#36;x&#36; Overall average &#36;\bar{X}&#36; Total or total &#36;t_x&#36;</li>
<li><strong>Sample data:</strong> From the sample. &#36;(x_i, y_i)&#36; Match Data</li>
</ul>
<p>The key insight of this estimation is that if the auxiliary variable is &#36;x&#36; & Study Variables &#36;y&#36; Highly relevant, then. &#36;x&#36; The overall information we know can help us estimate it more precisely. &#36;y&#36; General parameters. For example, if we know the total population of a region (&#36;x&#36;and observed population and household income (&#36;y&#36;There is a strong correlation, and then we can use it to obtain more accurate estimates of household income than the simple sample average.</p>
<p><strong>Comparison with simple and ratio estimates</strong></p>
<ul>
<li><strong>Simple random sampling estimates:</strong> Use only &#36;y&#36; sample information, ignore any supporting variables</li>
<li><strong>Ratio estimate:</strong> Assumptions &#36;y&#36; and &#36;x&#36; Proportional relationship between the point of origin &#36;y = R x&#36;</li>
<li><strong>Return estimate:</strong> Allow &#36;y&#36; and &#36;x&#36; General linear relationships between &#36;y = \beta_0 + \beta_1 x&#36;More flexible</li>
</ul>
<p>The advantage of returning to the estimate is when the relationship does not pass its roots. &#36;x=0&#36; Time &#36;y \neq 0&#36;It is more relevant than the estimated ratio. For example, even photographs count when estimating the number of dead trees (in the case of trees)&#36;x&#36;) Zero, field count (&#36;y&#36;The return estimate is more appropriate than the rate estimate at this time.</p>
<h4>Returning to estimated process</h4>
<p>We're re-presenting the process that's needed here. Because there are only two variables, the equation is simple.</p>
<p>Defines the overall regression parameter:</p>
<ul>
<li>&#36;\beta_1 = B_1 = \frac{\sum_{i=1}^{N} (x_i - \bar{X})(y_i - \bar{Y})}{\sum_{i=1}^{N} (x_i - \bar{X})^2}&#36; (General regression rate)</li>
<li>&#36;\beta_0 = B_0 = \bar{Y} - B_1 \bar{X}&#36; (Current regression transect)
I'm not sure.&#36;\bar{X}&#36; and &#36;\bar{Y}&#36; Auxiliary variables &#36;x&#36; And study variables &#36;y&#36; Overall average.</li>
</ul>
<p>From the sample, we can estimate:</p>
<ul>
<li>&#36;\hat{\beta}<em>1 = \hat{B}<em>1 = \frac{\sum</em>{i \in S} (x_i - \bar{x})(y_i - \bar{y})}{\sum</em>{i \in S} (x_i - \bar{x})^2}&#36;</li>
<li>&#36;\hat{\beta}_0 = \hat{B}_0 = \bar{y} - \hat{B}_1 \bar{x}&#36;</li>
</ul>
<p>&#36;y&#36; Overall average &#36;\bar{Y}&#36; The estimated return is:
&#36;&#36;
\tilde{y}_{reg} = \hat{B}_0 + \hat{B}_1 \bar{X} = \bar{y} + \hat{B}_1 (\bar{X} - \bar{x})
&#36;&#36;
This estimate can be understood as using the average sample first. &#36;\bar{y}&#36; estimate &#36;\bar{Y}&#36;, and then by the sample &#36;x&#36; With known total &#36;x&#36; Variance &#36;(\bar{X} - \bar{x})&#36;, combined with estimated slope &#36;\hat{B}_1&#36; Adjustments.</p>
<h4>Nature of the estimate</h4>
<p><strong>Offset</strong>
&#36;&#36;
\text{bias}(\tilde{y}_{reg}) = \text{cov}(\hat{B}_1, \bar{x})
&#36;&#36;
The regression estimate is usually biased, but when the sample is larger, the deviation is negligible. If the regression line goes through all the overalls, Points &#36;(x_i, y_i)&#36;, and &#36;\hat{B}_1 = B_1&#36;- No, it's zero.</p>
<p><strong>Average error (MSE)</strong>
&#36;&#36;
\text{MSE}(\tilde{y}_{reg}) = \left(1 - \frac{n}{N}\right) \frac{S_d^2}{n}
&#36;&#36;
of which &#36;d_i = y_i - [\bar{Y} + B_1(x_i - \bar{X})]&#36;，&#36;S_d^2&#36; These. &#36;d_i&#36; Overall variance.</p>
<p>Utilization of relevant factors &#36;\rho&#36;MSE can be rewritten as:
&#36;&#36;
\text{MSE}(\tilde{y}_{reg}) = \left(1 - \frac{n}{N}\right) \frac{1}{n} S_y^2 (1 - \rho^2)
&#36;&#36;
This indicates that:</p>
<ul>
<li>When sample amount &#36;n&#36; Increase or sample ratio &#36;n/N&#36; MSE decreases when increased</li>
<li>When? &#36;x&#36; and &#36;y&#36; Correlation factors between &#36;\rho&#36; Close &#36;\pm 1&#36; MSE is significantly reduced at the time</li>
</ul>
<p><strong>Total total return estimate</strong>
&#36;&#36;
\hat{t}<em>{yreg} = \sum</em>{i \in S} y_i + \sum_{i \in S^c} (\hat{B}<em>0 + \hat{B}<em>1 x_i) = \sum</em>{i \in S} y_i + (N - n)\hat{B}<em>0 + \hat{B}<em>1 (t_x - \sum</em>{i \in S} x_i)
&#36;&#36;
当 &#36;n \ll N&#36; 时，可近似为：
&#36;&#36;
\hat{t}</em>{yreg} \approx N(\hat{B}<em>0 + \hat{B}<em>1 \bar{X}) = N \tilde{y}</em>{reg}
&#36;&#36;
<strong>Standard Error</strong>
&#36;&#36;
\text{SE}(\tilde{y}</em>{reg}) = \sqrt{\left(1 - \frac{n}{N}\right) \frac{s_d^2}{n&#125;&#125;
&#36;&#36;
&#36;&#36;
\text{SE}(\hat{t}</em>=sqrt/2003/left1 - \right)\frac{s d^2}
I'm sorry.
of which &#36;s_d^2&#36; It's a cripple. &#36;d_i = y_i - \hat{B}_0 - \hat{B}_1 x_i&#36; the sample difference.</p>
<p><strong>95% confidence interval:</strong>
&#36;&#36;
\tilde{y}<em>{reg} \pm t</em>{n-2}(0.025) \sqrt{\left(1 - \frac{n}{N}\right) \frac{s_d^2}{n&#125;&#125;
&#36;&#36;
&#36;&#36;
\hat{t}<em>{yreg} \pm t</em>{n-2}(0.025) N \sqrt{\left(1 - \frac{n}{N}\right) \frac{s_d^2}{n&#125;&#125;
&#36;&#36;</p>
<h4>Summary of estimates of the return to the ratio</h4>
<p>Comparison of the three estimation methods</p>
<table>
<thead>
<tr>
<th align="left">Estimated methodology</th>
<th align="left">Mean &#36;\bar{Y}&#36; Estimated amount</th>
<th align="left">Total &#36;T_Y&#36; Estimated amount</th>
</tr>
</thead>
<tbody><tr>
<td align="left">SRS</td>
<td align="left">&#36;\bar{y}&#36;</td>
<td align="left">&#36;N\bar{y}&#36;</td>
</tr>
<tr>
<td align="left">Ratio</td>
<td align="left">&#36;\hat{B} \bar{X}&#36;</td>
<td align="left">&#36;\hat{B} t_x&#36;</td>
</tr>
<tr>
<td align="left">Return</td>
<td align="left">&#36;\hat{B}_0 + \hat{B}_1 \bar{X}&#36;</td>
<td align="left">&#36;N(\hat{B}_0 + \hat{B}_1 \bar{X})&#36;</td>
</tr>
</tbody></table>
<hr>
<p><strong>Select the situation for the return estimate:</strong></p>
<ul>
<li>When? &#36;x&#36; and &#36;y&#36; Linear relationships exist, but not necessarily through the original.</li>
<li>As a secondary variable &#36;x&#36; and &#36;y&#36; Heightly Associated (&#36;|rho|) &gt; 0.5&#36;）</li>
<li>When known &#36;x&#36; Total information (average or total)</li>
<li>When there is a systemic deviation, correction is required (e.g., photo counting deviation in dead tree count cases)</li>
</ul>
<hr>
<p><strong>Selection of ratio estimates:</strong></p>
<ul>
<li>When? &#36;x&#36; and &#36;y&#36; Proportional relationship between the point of origin</li>
<li>When Total Size &#36;N&#36; Unknown, but the supporting variable is known</li>
<li>It's particularly useful in the whole group sample.</li>
</ul>
<hr>
<p><strong>Select the SRS estimate:</strong></p>
<ul>
<li>When no supporting information available</li>
<li>When the auxiliary variable is relevant to the study variable Weak</li>
<li>When analysis requires simplicity and ease of interpretation</li>
</ul>
