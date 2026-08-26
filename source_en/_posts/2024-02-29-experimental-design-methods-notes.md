---
title: 'Experimental Design Methods: ANOVA, one-factor experiments, and multi-factor experiments'
title_zh: 试验设计方法：方差分析、单因素试验与多因素试验
date: 2024-02-29 21:54:06 +0800
categories:
- Data Science
- Statistical Modeling & Inference
tags:
- Statistics
- Experimental Design
author: Hyacehila
mathjax: true
hidden: true
excerpt: Covers ANOVA, one-factor experiments, multi-factor experiments, orthogonal designs, and experimental data analysis.
description: Covers ANOVA, one-factor experiments, multi-factor experiments, orthogonal designs, and experimental data analysis.
excerpt_zh: 整理方差分析、单因素试验、多因素试验、正交试验设计和实验数据分析方法。
permalink: /blog/2024/02/29/experimental-design-methods-notes/
lang: en
translation_key: 2024-02-29-experimental-design-methods-notes
translation_status: machine
translation_source_hash: 01b98834a9f52a7ee473f40f6dbc2f726470581375ba09ffeda7032d7451176a
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<h2>Summary of test design</h2>
<h3>General</h3>
<p>Data for differential analysis; data obtained through a certain design experiment, not some randomly measured data</p>
<p>Pilot design and analysis as a branch of mathematical statistics to study how to design experiments and analyse data;</p>
<p>In essence, the variance analysis examines only one issue:
<strong>Characteristic values at different levels of different factors</strong>
We'd like to make some useful judgements by studying this issue.</p>
<p>The purpose of the test design is common.<strong>Less experimental workload and lower cost of obtaining sufficient and reliable useful information</strong>; his core purpose is to collect data rather than analyse them</p>
<p>Nevertheless, finding a generic test design method that is unshowed can give us some of the better methods of designing a given type of problem, because this type of problem is often very special (in statistical theory).<strong>Introduce analysis directly in the test design method</strong></p>
<p>Experiment: Focus on academia
Testing: a method to design experiments and analyses</p>
<p>Pilot design is a very applied discipline, and the entire history of experimental design is industry-led.</p>
<h3>Development experience</h3>
<ul>
<li>Fisher's variance analysis in farm practice</li>
<li>The Japanese Statistician Taguchi I proposed an experimental design for the production of industrial goods</li>
<li>A balanced design proposed by Professor Wang Yuan, Fang Kai-tae, during China ' s missile development
<strong>All design methods are designed to reduce the number and cost of experiments in a given situation.</strong></li>
</ul>
<h3>Common terms and models for experimental design</h3>
<h4>Factors and Levels</h4>
<p>In one trial, the variable to be examined is called a factor.
Level: Different state of the factor
Factors are divided into qualitative and quantitative factors depending on whether the level is continuous or not.</p>
<h4>Response</h4>
<p>Results from tests
The purpose of the pilot design is to study the impact of the interaction of factors and factors on response and the relationship between them
The response is sometimes called a characteristic value.
A good test is the relationship between getting enough responses and factors at a minimum number of tests.</p>
<h4>Random error</h4>
<p>It's possible that in the experiment, the amount produced by the experiment has not been included in our study.
They're not controlled.
Good test designs can significantly reduce random error interference and help us find patterns.</p>
<h4>Common statistical models</h4>
<p>One of the main reasons for the efficiency of the methods designed by the statistical experiment is that they're very, very efficient.
Optimal methods under defined mathematical models;
The models used for experimental design are:</p>
<ul>
<li>Difference Analysis Model</li>
<li>Linear regression analysis model</li>
<li>Non-parameter regression model</li>
<li>Steady regression model
We'll talk about it later.</li>
</ul>
<h2>Difference Analysis</h2>
<p>We're here.<a href="/en/blog/2023/09/04/linear-regression-basics-notes/">Linear regression base</a>He's a regression analysis of an indicative variable, often looking at the effects of the existence or non-existence of a factor.</p>
<p>♪ Though it's ♪<a href="/en/blog/2023/09/04/linear-regression-basics-notes/">Linear regression base</a>We see it as a narrow issue of regression analysis, but our chapter will use a new approach.</p>
<p>By studying the origin of the overall variation between data in comparable arrays; finding significant influence factors on things, interactions between factors and optimal levels  </p>
<p>We need to ensure data independence.
We tend to study the following types of problems in differential analysis models.</p>
<ul>
<li>Visibility test for mean differences</li>
<li>Effect of the separation of relevant factors on total variability</li>
<li>Interaction between analytical factors</li>
<li>Spacing test</li>
</ul>
<h3>The idea of differential analysis</h3>
<p>Here's the data from a trial.</p>
<table>
<thead>
<tr>
<th>1</th>
<th>2</th>
<th>3</th>
<th>4</th>
<th>5</th>
<th>Mean</th>
</tr>
</thead>
<tbody><tr>
<td>89</td>
<td>62</td>
<td>93</td>
<td>71</td>
<td>85</td>
<td>71.4</td>
</tr>
<tr>
<td>75</td>
<td>78</td>
<td>60</td>
<td>61</td>
<td>83</td>
<td>80.0</td>
</tr>
<tr>
<td>We want to determine which conditions are better observed; if there are no errors in observations, the results of multiple tests under different conditions are the same, and it is natural to judge.</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td>But in fact, the test results are always influenced by errors, and we can't tell whether the differences that we're showing are caused by conditions or errors.</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td>We can describe the fluctuations in the averages from the results of the same set of tests, that is,</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td>&#36;&#36;\begin{gathered}</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td>S_{E}^{2}=(75-71.4)^{2}+(78-71.4)^{2}+\cdots+(83-71.4)^{2} \</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td>+(89-80.0)^2+(62-80.0)^2+\cdotp\cdotp\cdotp+(85-80.0)^2</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td>\end{gathered}&#36;&#36;</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td>For overall indicator fluctuations, the error in the overall average of all tests can be described.</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td>&#36;&#36;\begin{aligned}S_T^2&amp;=(75-75.7)^2+(78-75.7)^2+\cdots+(85-75.7)^2\&amp;=1294.10.\end{aligned}&#36;&#36;</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td>In the case of indicator fluctuations caused by this change in factor, the average of two different levels can be described as the average of the average for the whole.</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td>&#36;&#36;S_{A}^{2}=5(71.4-75.7)^{2}+5(80.0-75.7)^{2}=184.90&#36;&#36;</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td>It's natural. We can see that.</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td>&#36;&#36;S_{T}^{2}=S_{A}^{2}+S_{E}^{2}&#36;&#36;</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td>We can see it, either.&#36;S_A&#36;Still?&#36;S_E&#36; They all have to do with the number of tests.</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td>Yeah.&#36;S_E&#36; It uses 10 data, but the two averages were calculated by the remaining ten conditions, two limitations, and he's free. &#36;10-2=8&#36;</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td>Yeah.&#36;S_A&#36; It uses two data.&#36;2-1=1&#36;</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td>The overall freedom is both.&#36;0&#36;</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td>If the change in condition has a significant impact on the change in the indicator, then we can get it.&#36;S_A&#36;It'll be very big, relative.&#36;S_E&#36;It'll be small.</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td>&#36;&#36;F_{A}=\frac{S_{A}^{2}/f_{A&#125;&#125;{S_{E}^{2}/f_{E&#125;&#125;&#36;&#36;</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td>When the margin is large enough, it can be assumed that the difference in the significance of our tests is not explained by the error.</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td><strong>As we can see from the reasoning above, we need one side.&#36;F&#36;The test will be ours.</strong></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td>Let's do a little research.</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
</tbody></table>
<h3>Single-factor equation analysis</h3>
<h4>Model construction</h4>
<p>We'll start with a single factor differential analysis.
We gave a model profile.
&#36;&#36;y_{ij}=\mu_i+e_{ij}&#36;&#36;
of which&#36;y_{ij}&#36;is the specified value &#36;\mu_i&#36;It's the average of several tests at one level and then random error.
Which means we think all variables are the same except for the level of factors to be studied.
So studying all levels is research.&#36;\mu_i&#36;Variance
We can easily transform the above model into the following.
&#36; \begin{aligned}
&amp;y_{ij}=\mu+\alpha_i+e_{ij} \
&amp;e_{ij}\sim N(0,\sigma^2) \
&amp;\sum_{i=1}^an_i\alpha_i=0
\end{aligned}&#36;&#36;
这个模型也是很好理解的 &#36;y_{ij}&#36;是具体的取值  &#36;\mu&#36;是总体的均值 &#36;\alpha_i&#36;是所有的水平对最后取值的影响 最后是随机误差 至于后面的等式 如下就可以证明
&#36;&#36;\sum_{i=1}^an_i(\mu_i-\mu)=\sum_{i=1}^an_i\mu_i-\sum_{i=1}^an_i\mu=n\mu-n\mu=0&#36;&#36;
还是和我们在第一章干的事情一样 改写为矩阵形式
&#36;&#36;\begin{gathered}\mathbf{y}=X\boldsymbol{\beta}+\boldsymbol{e}\e\sim N(0,\sigma^2I_n)\\boldsymbol{h&#39;== sync, corrected by elderman ==
It's in the form of a matrix that we've studied in the first chapter.</p>
<h4>Common statistics in variance analysis</h4>
<p>First, give some statistics that are very common in differential analysis.
General average
&#36;&#36;\overline{y}=\frac{1}{n_ia}\sum_{i=1}^{a}\sum_{j=1}^{n_i}y_{ij}&#36;&#36;
Defines the total squared as follows:
&#36;&#36;SS_{T}=\sum_{i=1}^{\alpha}\sum_{j=1}^{n_{i&#125;&#125;(y_{ij}-\bar{y})^{2}&#36;&#36;
It intuitively describes how the whole data is concentrated and all the tests deviate from the overall average.
Reordered.
&#36;overline{y}<em>{i}=\frac1n\sum</em>{i=1}^ny_{ij}&#36;&#36;
则定义组内平方和
&#36;&#36;SS_{<em>E}=\sum</em>{i=1}^{a}\sum_{j=1}^{n_{i&#125;&#125;(y_{ij}-\overline{y}<em>{i.})^2&#36;&#36;
它直观的描述来某个水平下所有的试验偏离水平均值的情况 也就是随机误差的影响
再定义组间平方和
&#36;&#36;SS_A=\sum</em>{i=1}^a\sum_{j=1}^{n_i}(\overline{y}<em>{i.}-\overline{y}</em>{})^2=\sum_{i=1}^an_i(\overline{y}<em>{i.}-\overline{y}</em>{})^2&#36;&#36;
它解释了各个水平的试验偏离整体试验均值的程度
我们经过一系列运算可以得到下面的定理
&#36;&#36;SS_T=\sum_{i=1}^a\sum_{j=1}^{n_i}\Big[(y_{ij}-\overline{y}<em>{i.})^2+(\overline{y}</em>{i.}-\overline{y})^2\Big]=SS_E+SS_A&#36;&#36;</p>
<h4>Assumptions test</h4>
<p>What we want to test is a model factor.&#36;A&#36;Whether there are any significant differences in the average values below each level, i.e.
&#36;&#36;H_0:\alpha_1=\alpha_2\overset{}{\operatorname*{=&#125;&#125;\cdots=\alpha_a\overset{}{\operatorname*{=&#125;&#125;0&#36;&#36;
Now we can extrapolate the statistics.
You can extrapolate to
&#36;&#36;E(SS_A)=(a-1)\sigma^2+\sum_{i=1}^an_i\alpha_i^2&#36;&#36;
Therefore...
&#36;&#36;E(\frac{SS_A}{a-1})=\sigma^2+\frac1{(a-1)}\sum_{i=i}^cn_i\alpha_i^2&#36;&#36;
It reflects the effects of each level of effect.
When hypothetical&#36;H_0&#36;When it was set up,
Which means...&#36;\alpha_1=\alpha_2\overset{}{\operatorname*{=&#125;&#125;\cdots=\alpha_a\overset{}{\operatorname*{=&#125;&#125;0&#36;
Therefore...
&#36;\frac&#123;&#123;SS}_A}{a-1}\text{是}\sigma^2\text{的无偏估计}.&#36;  &#36;\frac&#123;&#123;S}{S}_E}{n-a}\text{为}\sigma^2\text{的无偏估计}&#36;
Therefore...
&#36;&#36;F=\frac{SS_A/(a-1)}{SS_E/(n-a)}\text{接近于1}&#36;&#36;
When the hypothesis doesn't work, it's very obvious that the balance of the group is squared.&#36;SS_A&#36;It'll increase.&#36;SS_E&#36;It reduces the value of testing statistics.
So all we need to do is be one side.&#36;F&#36;Test
It's in there.&#36;a-1,n-a&#36;It's the expression of freedom.</p>
<h4>Estimates</h4>
<p>When we reject the assumption that the levels of the factor are different, we need to estimate the size of the difference.&#36;\mu_{i}-\mu_{j}&#36;I mean, I'm going to do an inter-sectional estimation and, of course, integrate the data into the regression analysis.
It's easy to know.
&#36; \begin{aligned}\overline{y}<em>{i.}&amp;\sim N(\mu_i,\frac{\sigma^2}{n_i})\\overline{y}</em>{i.}-\overline{y}<em>{j.}&amp;\sim N(\mu_i-\mu_j,(\frac{1}{n_i}+\frac{1}{n_j})\sigma^2)\end{aligned}&#36;&#36;
因此可以导出
&#36;&#36;\begin{aligned}U&amp;=\frac{\overline{y}</em>{i.}-\overline{y}<em>{j.}-(\mu</em>{i}-\mu_{j})}{\sigma\sqrt{\frac1{n_{i&#125;&#125;+\frac1{n_{j&#125;&#125;}-N(0,1)}\SS_{E}/\sigma^{2}&amp;=\sum_{i=1}^{a}\sum_{j=1}^{n_{i&#125;&#125;(\overline{y}<em>{i.}-\overline{y})^{2}\left/\sigma^{2}-\chi</em>{n-a}^{2}\right.\end{aligned}&#36;&#36;
最后导出&#36;t&#36;枢轴量
&#36;&#36;\frac{\overline{\boldsymbol{y&#125;&#125;<em>{i,}-\overline{\boldsymbol{y&#125;&#125;</em>{j,}-(\mu_{i}-\mu_{j})}{\sqrt{\frac{\boldsymbol{S}\boldsymbol{S}<em>{E&#125;&#125;{n-a&#125;&#125;\sqrt{\frac1{n</em>+\frac1{n j} &#36;
We won't give up the last compartment.
<strong>In fact, it's an estimate of the average value difference in mathematical statistics.</strong>
Don't forget, we have to run a hypothetical test before we talk about estimates of the average margin.
<strong>If this confidence zone contains zero, it means the average difference is not significant.</strong></p>
<h4>Multiple comparisons</h4>
<p>The significant differences that we've demonstrated before only show that the levels of this factor have an impact on the indicators, but they don't help us select the most appropriate tests, multiple comparisons and multiple comparisons.&#36;S&#36;The test will help us.
First you have to give the factor down the inverted triangle.
<img src="/assets/images/probability-statistics-notes/experimental-design-methods-notes-01.png" alt="Pilot design methodology">
The specific sequence software will give us a calculation.
Calculate
&#36;&#36;D_{ij}(\alpha)=\sqrt{\left(\frac{1}{n_{i&#125;&#125;+\frac{1}{n_{j&#125;&#125;\right)\frac{S^{2&#125;&#125;{n-r}(r-1)F_{s}(r-1,n-r)}&#36;&#36;
If the average of the horizontal indicator is greater&#36;D_{ij}&#36; Proof of significant differences between the two levels
And then we'll fill in the table.</p>
<h3>Analysis and interaction of factors</h3>
<h4>Model description</h4>
<p>Control of variables is a difficult thing, and that's why we're going to study the differential analysis of both factors.
It's very easy to give the core model of this section.
&#36;&#36;y_{ij}=\mu+\alpha_{i}+\beta_{j}+e_{ij}&#36;&#36;
The number of tests at each level of each factor is still to be determined, and if we want to include it, we inevitably introduce the existence of interactions (which are often rarely taken into account in regression analysis, which is already being addressed by the Cyclops).
&#36;&#36;y_{ijk}=\mu+\alpha_{i}+\beta_{j}+\gamma_{ij}+e_{ijk}&#36;&#36;</p>
<h4>Impact of interactions</h4>
<p>We will be followed by a presentation of two factors that do not take into account the interaction and do not repeat the test differential analysis;
If we find out</p>
<ul>
<li>Neither effect is significant.</li>
<li>That's not obvious. It's too big a difference squared.&#36;SS_E&#36;That's because of the error.</li>
<li>We have more confidence in the error control of the experiment.
It's very likely that there's a new effect in there that has an impact on the test, and it's often an interaction.</li>
</ul>
<h3>Both factors do not repeat test variance analysis</h3>
<p>Or do you have the following definition with the idea of creating squares?
Total Square
&#36;&#36;SS_{T}=\sum_{i=1}^{a}\sum_{j=1}^{n_{i&#125;&#125;(y_{ij}-\overline{y})^{2}&#36;&#36;
Group squared
&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;&#36;US&#36;US&#36;US&#36;US&#36;&#36;US&#36;&#36;&#36;&#36;&#36;US&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;US&#36;&#36;&#36;&#36;&#36;US&#36;&#36;&#36;&#36;&#36;US&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;US&#36;&#36;US&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;US&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;...&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;US&#36;&#36;&#36;&#36;...&#36;&#36;&#36;&#36;&#36;...&#36;&#36;&#36;&#36;&#36;&#36;<em>{i.}-\overline{y}</em>{.j}+\overline{y})^{2}&#36;&#36;
因素A组间平方和
&#36;&#36;SS_{A}=b\sum_{i=1}^{a}(\overline{y}<em>{i.}-\overline{y})^{2}&#36;&#36;
因素B组间平方和
&#36;&#36;&#123;&#123;SS</em>{B}=a\sum_{j=1}^{a}(\overline{y}_{.j}-\overline{y})^{2&#125;&#125;}&#36;&#36;</p>
<h4>New statistics</h4>
<p>An assumption that the levels of both factors have a significant impact on the indicator is examined below
&#36;&#36;H_1:\alpha_1=\alpha_2=\cdots=\alpha_a{=}0&#36;&#36;
&#36;&#36;H_{2}:\beta_{1}=\beta_{2}=\cdots=\beta_{b}=0&#36;&#36;</p>
<h4>Assumptions test</h4>
<p>It's easy to give a test count.
&#36;&#36;F=\frac{SS_A/(a-1)}{SS_E/(a-1)(b-1)}\sim F_{a-1,(a-1)(b-1)}&#36;&#36;
These coefficients are still limited by freedom.
If you want to calculate another factor,&#36;B&#36;The influence only needs the molecule.&#36;A&#36;Replace&#36;B&#36; Synchronize Freedom
Or is it a one-sided test?&#36;F&#36;When you're too big, you reject the assumption.</p>
<h4>Estimates</h4>
<p>The idea is exactly the same, or do you study the spatial estimates of the average difference?
&#36; \frac/2003/overline{y}<em>{i.}-\overline{y}</em>{t.} (\alpha i-\alpha t)\sqrt {SS E}(a-1)(b-1)&#125;&#125;sqrt}\simt {(a-1)(b-1)}&#36;&#36;
It's very easy to export.
<strong>If this confidence zone contains zero, it means the average difference is not significant.</strong></p>
<h3>Two factors and so forth, repeat test variance analysis.</h3>
<p>The model is presented below.
&#36;&#36;y_{ij\mathrm{k&#125;&#125;=\mu+\alpha_{i}+\beta_{j}+\gamma_{ij}+\varepsilon_{ijk}&#36;&#36;
Please note that in models where there are interactions, it makes no sense to look at one level of good or bad of one factor, because its good or bad is related to the value of another factor, so we're just...<strong>All we have to do is test the existence of the interaction.</strong> If there's no way to follow the previous part, if there is, then there's no point in the previous part of the non-interactive study.</p>
<p>We need to test the hypothesis as if
&#36;&#36;H_3:\gamma_{ij}=0,i=1,...,a;j=1,...,b&#36;&#36;
Give new statistics
&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;US&#36;&#36;US&#36;US&#36;&#36;US&#36;&#36;US&#36;&#36;&#36;US&#36;&#36;&#36;US&#36;&#36;US&#36;&#36;&#36;&#36;&#36;US&#36;&#36;US&#36;&#36;US&#36;&#36;US&#36;US&#36;US&#36;<em>{A\times B}=c\sum</em>{i=1}^a\sum_{j=1}^b(\overline{y}+\overline{y}<em>{ij.}-\overline{y}</em>{i..}-\overline{y}<em>{..j.})^2&#36;&#36;
&#36;&#36;SS_E=\sum</em>{i=1}^a\sum_{j=1}^b\sum_{k\operatorname{=}1}^c(y_{ijk}-\overline{y}<em>{ij\operatorname{.&#125;&#125;)^2&#36;&#36;
给出&#36;F&#36;检验统计量有
&#36;&#36;F</em>== sync, corrected by elderman == @elder man
Or do you use a one-sided test to see intuitively the direction of the test is consistent with the previous section?
Now all the work software on differential analysis is available, and no manual calculations are required.
At the end of the variance analysis:</p>
<p><strong>The variance analysis is a comprehensive test method, which, as the number of factors increases with their level, the number of combinations increases dramatically and it is not possible to conduct all tests, and we must introduce some of the tests.</strong> ^35bacd</p>
<h2>Turning over test design.</h2>
<p>See the “relevant paragraphs” section of this paper.
One of the tests is a more efficient design method.</p>
<h3>Arrange tests with a positive tab</h3>
<p>A standard due form is shown below
&#36;&#36;L_{k}(m^n)&#36;&#36;
of which &#36;k&#36;Other Organiser &#36;m&#36;Indicate the level of the factor &#36;n&#36;Number of indicators
Of course, you can use a similar line of thought to indicate a mixed horizontal submission.
&#36;&#36;L_{k}(m^{n}\times p^q)&#36;&#36;
Characteristics of the positive submission (reflecting the idea of its balanced experiment)</p>
<ul>
<li>Write a large matrix, and each row is the horizontal selection of a factor for each test.</li>
<li>Numbers appear as many times as in each column</li>
<li>Each column is balanced by the number, which is the same number of times it appears.</li>
</ul>
<p>When we're experiencing multiple-factor trials, but we can't do enough, it's often done according to a positive tab, which is created by software.</p>
<h3>Direct test design without interaction</h3>
<p>Here's how we're going to analyze the calculations based on test data from the positive test.
An ordinary test data log is as follows:
<img src="/assets/images/probability-statistics-notes/experimental-design-methods-notes-02.png" alt="Test design method 1">
The fourth column does not have letters because we only have three factors, but we have a table of four.
<strong>The blank column reflects the impact of random error, and if the blank column is too different, it must mean that we're missing important influences.</strong></p>
<p>The rightmostbar is the test result, of which&#36;T&#36;It's the sum roman numeral number.&#36;I&#36; Meaning the sum of the indicator values selected for 1 at all levels in the column where the factor is present (reflecting the average of the values in a group and/or group) &#36;R&#36;That's the factor.<del>II</del>Extremely bad third-value</p>
<p>In simple data processing, we rely directly on &#36;I.<del>II</del>The value of the &#36;3.0 to determine the level of our final choice, and the extreme difference reflects the magnitude of this factor.</p>
<p>And, of course, if you find that some of these factors are not significant, you can make the right adjustments, and you can do the best you can at a smaller cost.  </p>
<p><strong>It's obvious that this analysis is too crude, and the conclusions that have been drawn must be retested and never directly accepted.</strong></p>
<h3>Direct test design considering interaction</h3>
<p>If we want to analyse the interaction in the direct test, the experimental design and treatment programme will change slightly;
As a matter of custom, we have the following principles for dealing with interaction.</p>
<ul>
<li>Ignore advanced interactions</li>
<li>Selectively explore a level of interaction. Usually only those whose effects are more visible, or which the test requires (which interactions depend on experience, difficult to handle)</li>
<li>Use 2-factor experiment to the extent feasible</li>
</ul>
<p>The usual positive forms are followed by a two-column interactive list used to analyse whether the interaction exists, as shown below.
<img src="/assets/images/probability-statistics-notes/experimental-design-methods-notes-03.png" alt="Test design method 2">
All figures in the table are column numbers in the positive table:</p>
<p>If you want to see the interactive column of column 12 from&#36;(1)&#36; Looking right across from&#36;(2)&#36; So they look up, and that's how they interact with column three.</p>
<p>It can be seen that if we were to analyse the interaction, many of the columns in the original periodic test could not rank factors, but rather examine the interaction column as a single factor.</p>
<p>At this point in time, the analysis of the existence and size of the interaction would require a study of the obvious impact of the interactive role column on the indicator.</p>
<p>If you look at the interaction, you have to choose a bigger, positive form. <strong>&#36;n&#36;Interactive need for factors&#36;n-1&#36;The column interactive column needs to expand the selected positive log to ensure that no multiplicity occurs.</strong></p>
<p>The analysis of the positive results with interactive effects shows the following changes:
The original analysis required only an analysis of the impact of the very poor judgmental factors in the various factors, and the extreme difference in the interactive column now reflected the magnitude of the impact of the interaction; the analysis of the other factors did not change;</p>
<p>For the final selection of the appropriate test level:</p>
<ul>
<li>Not interactive: or the same method of selection as above</li>
<li>The impact of interaction is small: as no interaction</li>
<li>Interactive factors: The selection of its level cannot be considered separately, requiring the drawing of a binary table and a binary chart to be compared before selecting the priority level for the indicator</li>
</ul>
<h3>Multiple indicator test</h3>
<p>Many of the tests have multiple indicators to measure the results, and there's a simple approach.</p>
<h4>Comprehensive balance approach</h4>
<p>The idea is, we're using it as a single indicator, each of which is being analysed separately.
Finally, we'll sort out the factors for each indicator.
This last step requires subjective observation and trade-offs.</p>
<h4>Comprehensive score method</h4>
<p>Just as we are.<a href="/en/blog/2024/01/01/statistical-decision-theory-notes/">Statistical policymaking</a> The same approach is taken in the “multi-target decision-making” section; some empowerment approach is used to group multiple goals into one, and to carry out related analysis</p>
<h3>An analysis of the differences in the design of the test</h3>
<p>Here we're going to look at the differential analysis of the positive table test; naturally, it's because the method of subjective selection of the results of the test lacks a measure of certainty about uncertainty.
The whole calculation is very simple, or the question of the squared difference.</p>
<p>It's the same rule that gives a high percentage of the statistics to be used this time.
Total Square
&#36;&#36;SS_T=\sum_{i=1}^n(y_i-\bar{y})^2&#36;&#36;</p>
<p>Square sum of a factor (column change squared)
&#36;&#36;SS_{A}=\frac{a}{n}\sum_{j=1}^{a}(K_{j}^{A})^{2}-\frac{1}{n}(\sum_{j=1}^{n}y_{i})^{2}&#36;&#36;
Of which&#36;K_j^A&#36; Representation factor&#36;A&#36;No. No.&#36;j&#36;the sum of all test values at each level
The data in the previous table can be shown as</p>
<p>&#36;&#36;
S_i^2=\frac{\mathrm{I_i}^2+\Pi_i^2+\Pi_i^2+\cdots}{\text{水平重复数&#125;&#125; - \frac { T ^ 2 }{\textbf{数据总个数&#125;&#125;.
&#36;&#36;</p>
<p>Sum of error squared
&#36;&#36;SS_E{=}SS_T{-}(SS_1+SS_2+\cdots+SS_m)&#36;&#36;</p>
<p>About freedom.
&#36;f_T=n-1&#36; of which&#36;n&#36;It's the total number of tests.
&#36;f_i=a-1&#36; of which&#36;a&#36;is the level of the factor
&#36;f_E=n-m(a-1)-1&#36; It's the level of freedom, the level of freedom, the degree of freedom, the degree of freedom, the degree of freedom.
<strong>Interactive interactive column as well as thought analysis above</strong></p>
<p>Then we introduce the concept of parity.
Medium&#36;MS&#36;value is the sum of squares&#36;SS&#36; The penalty is freedom. &#36;f&#36; Which means...
&#36;&#36;MS=\frac{SS}{f}&#36;&#36;
So the difference analysis for the test can be calculated.&#36;F&#36;Test statistics for analysis.</p>
<p>Research factors&#36;A&#36; then
&#36;F A}\frac{MS}<em>{A&#125;&#125;&#123;&#123;MS}</em>That's right.
Other factors are the same.
That's how we do the differential analysis of the positive test. </p>
<p><strong>Difference analysis, all empty columns and the effect of error items</strong></p>
<p>Factors that have a disproportionate impact include errors</p>
<p>A positive table without an empty column is not capable of a variance analysis and requires a repetition of the error in the time estimate.</p>
<h3>Repeated tests and repeat sampling</h3>
<p>There are no empty columns in the active table and it is not possible to use it to calculate the error of the test. In the event of a decision not to use a larger positive submission (the selection of a large positive submission would result in a sharp increase in the number of tests), the error of the repeated test may be used as a test error. Of course, if there's any other reason to repeat the experiment, that's how it works.</p>
<p>The place where the data are being filled in the test form will change in the case of repeated tests or repeated sampling as follows:
<img src="/assets/images/probability-statistics-notes/experimental-design-methods-notes-04.png" alt="Pilot design methodology 3">
The calculation of our data on the table is basically unchanged.
These calculations usually let the software go.</p>
<h3>Flexible use of positive returns</h3>
<h4>Mix positive tabs with parallels</h4>
<p>It's a common method used to construct positive tabs at different levels.
&#36;&#36;L_{k}(m^{n}\times p^q)&#36;&#36;
This part of the job is no longer manual.
Give us the number of factors we need, the level of each factor, and the statistical software will give us the positive forms we need, and we don't need to know.</p>
<p>The only thing we need to notice here is:<strong>When the level of the factor is identical, the primary relationship of the factor is determined entirely by the size of the extreme R. When levels are not exactly the same, direct comparisons are not possible because when quantitative factors have the same impact on indicators, the factors for multiple levels should be much higher. So we need to use the coefficient to convert the extreme.</strong> </p>
<h4>Proposed horizontal approach</h4>
<p>To add some less horizontal factors to some more virtual ones so that the less horizontal ones can be ranked in the positive tables that we need. Medium
It's very simple. <strong>Let a certain level of the factor repeat as a new level. Medium</strong></p>
<p>Additional notes:</p>
<ul>
<li>Given the different levels of factor D and other factors, it would be inappropriate to use the extremely poor R to compare the order of the factor (conversion factor). However, the use of differential analysis can still yield reliable results.</li>
<li>While the horizontal approach has expanded the use of the positive instrument, it is worth noting that it is no longer a positive table after the proposed horizontal changes, and it loses the nature of the balanced mix between the levels of the various factors</li>
</ul>
<h4>Delineation of factors</h4>
<p>The method of putting higher-level factors into lower-level positive tables.</p>
<h4>Immediate Law</h4>
<p>We're going to experience too many different levels of testing in large-scale industrial experiments, too many tests on a positive table, and we need to do it in stages.
There are common ideas.</p>
<ul>
<li>Reduction factor: fix certain factors not tested in the first table and change the level of this factor in the second table</li>
<li>Reduced level: part of the level of a factor does not test in the first and in the second allows the best in the first to be compared with the remaining untested level</li>
<li>Combination factor: to combine multiple factors, if they are not significant, then there is no direct need to continue the split test.
It'll take us to do it in parallel to get the final results.</li>
</ul>
<h4>Flow method</h4>
<p>A differential analysis method focused on the interaction of some and others
For example, in chemical tests there are formulation factors (raw materials, matching, etc.) and process factors (processing time, temperature, etc.) there must be two types of interaction; but their internal interactions can often be ignored.
In the experimental design, we consider only interaction between classes, not interaction between factors within classes.
Finally give the margin analysis table.</p>
<p><strong>The core of the two categories of differential analysis is the proper use of statistical software, and all we need to do is look at the final methodological analysis table and reschedule the tests as appropriate.</strong></p>
<h2>even design</h2>
<h3>evenly designed presentation</h3>
<p>We first presented the variance analysis of the full-scale test, and then the problem of the contemporary design due to the high level and number of elements, which was also presented to address the inadequacy of the original experimental design methodology.</p>
<p>The characteristics of the current table tell us that the number of active tests is double the number of squares of horizontal numbers, which means that the increase in our level is related to a flat increase in the number of tests, which, if the level is too high, will soon reach an amount that we cannot afford.</p>
<p>However.<strong>Multiple-level experiments are also frequent, and we need new experimental methods, which raises the issue of even design.</strong> The flat design is Fong Kai-tae, and Prof. Wang Yuan took up the idea of digital theory in his research on missile design tests. It is an application of the pseudo-Montecaro method (formerly Monte Carlo is a random generator, and the pseudo-Monte Carlo idea is that it is supposed to produce a hyper-equilibrium sequence).</p>
<p>The original experimental design idea.</p>
<ul>
<li>evenly dispersed: the test sites are distributed evenly within the test range, allowing for adequate representation of each test point</li>
<li>Accompanying: makes it easier to analyse the results of the tests, to estimate the impact of the factors on the indicators and to identify the main contradictions affecting the change of things
The pilot design is already the best solution for meeting these two test conditions, but if we ignore the requirement of alignment and comparability, even design is proposed.</li>
</ul>
<h3>even design sheet</h3>
<p>The flat sheet is the
&#36;&#36;{\operatorname*{U_7&#125;&#125;(7^6)&#36;&#36;
Of these, the first corner mark 7 means the number of tests, the number in brackets 7 means the allowed level, the upper mark 6 means the permitted number of factors, the U means the uniform design, and the symbol in the table means that only the L is replaced by the U.</p>
<p>Using a uniform design, each level of the factor is tested only once, and when the level increases, the number of tests increases with the level. The number of tests has changed from squared to linear.</p>
<p>Each flat design table is accompanied by a table of usage, calculated by the predecessor on the basis of some rule, which shows how we select the appropriate columns from the design table and the degree of homogeneity of the test programmes made up of these columns. Give&#36;{\operatorname*{U_6&#125;&#125;(6^6)&#36; And the corresponding usage forms are:</p>
<p><img src="/assets/images/probability-statistics-notes/experimental-design-methods-notes-05.png" alt="Test design methodology 4"></p>
<p>There is no difference between the test table on the left and the interpretation of the active test design table on the right, and the use table on the right tells us that 13 columns and three factors and 123 columns should be used for both factors.</p>
<p>Draw odd tables into the last line and get even tables less than it, and use the table unchanged</p>
<p>In particular: In chemical experiments, low-level and low-level encounters may be unreactive, high-level and high-level encounters may be over-reacting, and we need to adjust the numbering of the level; based on the principle of uniform design tables, he cannot simply change the order of the level to the same level as the original test design table, but only smooth it in the order of a circle, which requires our attention.</p>
<h3>Analysis of the results of the flat design test</h3>
<p>Since the flat design no longer takes into account the full comparability of the present test,<strong>The test results are processed using regression analysis.</strong>• Linear regression or multiple regression analysis.<em>That's using the interpretation model in the regression analysis to achieve the outcome analysis.</em></p>
<p>A regression test may be conducted in the regression analysis to determine the significance of the factor in the model, depending on the size of the regression square;</p>
<p>Where there is no correlation between the factors, the fact that the factor is back to square size also reflects its importance for the impact of the test indicator. It's usually done with computers.</p>
<p><strong>All techniques of regression analysis can be used for uniform design tests, but it is important not to use the results of differential analysis techniques to analyse even design, with too many features and too few observations leading to no results.</strong></p>
