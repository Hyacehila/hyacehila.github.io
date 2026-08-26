---
title: How to Share Data with a Statistician
title_zh: 如何与统计学家分享数据 (Jeff Leek)
date: 2026-02-17 12:00:00 +0800
categories:
- Work & Society
- Research Practice
tags:
- Data Curation
- Reproducibility
- Repost
author: Hyacehila
excerpt: A repost of Jeff Leek's classic guide to preparing and organizing data before working with a statistician, covering
  raw data, tidy data, and code books.
description: A repost of Jeff Leek's classic guide to preparing and organizing data before working with a statistician, covering
  raw data, tidy data, and code books.
excerpt_zh: 本文转载自 Jeff Leek 的经典文章，详细介绍了在与统计学家合作分析数据前，应该如何准备和整理数据（Raw Data, Tidy Data, Code Book）。
permalink: /blog/2026/02/17/how-to-share-data-with-a-statistician/
lang: en
translation_key: 2026-02-17-how-to-share-data-with-a-statistician
translation_status: machine
translation_source_hash: 28f46e87ec8ac5fbdc456c0269423d61483d21584b9fb525bc94acdd1ce57835
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<blockquote>
<p>This post is from a guest article published by Jeff Leek in BMC Blog. <a href="https://blogs.biomedcentral.com/bmcblog/2013/11/26/how-to-share-data-with-a-statistician/">How to Share Data with a Statistician</a>, the detailed guidance is derived from <a href="https://github.com/jtleek/datasharing">GitHub Repository</a>。</p>
<p>It is a practical guide for researchers, students and data compilers who need to work with statisticians. It describes the extent to which data should be collated before they are handed over to statistical analysts in order to reduce communication costs and speed up the analysis process.</p>
</blockquote>
<p>The questions in this article can also be addressed<a href="/en/blog/2025/09/04/research-theory-and-practice/">Scientific theory and practical experience</a>、<a href="/en/blog/2026/01/24/the-statistical-crisis-in-science/">The Statistical Crisis in Science</a>How the concept of a relatively close read together is developed in different contexts.</p>
<h1>How to share data with statisticians</h1>
<p>This is a guide for those who need to share data with statisticians. It is possible that you can benefit from this, whether you are a scientific partner, a student or a doctorate who needs the assistance of statisticians in analysing the data, or a primary statistical student who is responsible for organizing/cleaning the data.</p>
<p>The objective of the guidelines is to provide best practices for data sharing and to avoid the pitfalls and delays that are common in the transition from data collection to data analysis. The Leek team, together with a large number of collaborators, found that the most important factor in the difference in output speed is the state of the data when they reach us. According to my interaction with other statisticians, it's almost a universal truth.</p>
<p>I think statisticians should be able to process data in any state. It is important to view raw data, understand each step of the process and include implicit sources of variability in data analysis. However, for many data types, processing steps have been documented and standardized. Therefore, the conversion of data from the original form to a form that can be analysed directly could be accomplished well before seeking the assistance of statisticians. This would significantly accelerate turnaround time, as statisticians do not need to process all pre-treatment steps first.</p>
<h2>What are you supposed to offer to statisticians?</h2>
<p>To speed up analysis, you should provide the following information to statisticians:</p>
<ol>
<li><strong>Source Data (The Raw Data)</strong></li>
<li><strong>Clean Data Set (A dy data set)</strong></li>
<li><strong>Code book (A code book)</strong>: Describes each variable in the clean data set and its values.</li>
<li><strong>Processing of records</strong>: a clear and precise recipe that records how you moved from step 1 to step 2 and 3.</li>
</ol>
<p>Each part of the package is presented in turn.</p>
<h3>1. Raw data (The Raw data)</h3>
<p>To provide statisticians with what you can get.<strong>Original</strong>Data is important. This ensures that data trace across the entire workflow.</p>
<p>The following are examples of some raw data forms:</p>
<ul>
<li>The odd binary files that the measuring machine spit out.</li>
<li>The company signed the Excel file containing 10 sheets.</li>
<li>The complex JSON data that you can retrieve from Twitter API.</li>
<li>The numbers are observed through microscopes and manually recorded.</li>
</ul>
<p>You can judge whether the original data is in the correct format by the following criteria:</p>
<ol>
<li>No software was run for the data.</li>
<li>No data values have been modified.</li>
<li>No data from the data collection was removed.</li>
<li>Data are not aggregated in any way.</li>
</ol>
<p>If you make any changes to the original data, it is no longer in original form.<strong>Reporting of modified data as raw data is a common cause of slowing down the analysis process</strong>Because analysts usually have to do “cord-in” your data to figure out why the original data look strange. (And imagine what would happen if new data arrived?</p>
<h3>2. Clean data set (A dy data set)</h3>
<p>The general principle of clean data (Tidy Data) is that Hadley Wickham is a man who is a man of the world.<a href="http://vita.had.co.nz/papers/tidy-data.pdf">This paper.</a>Submitted by: Although, as with all truths, these principles are common in R:</p>
<ol>
<li><strong>Each variable</strong>You should measure a column.</li>
<li><strong>Every different observation.</strong>should be a row under this variable.</li>
<li><strong>Each type</strong>. The variable should have a table.</li>
<li>If you have more than one table, they should contain a column that allows them to be connected or merged.</li>
</ol>
<p>Although these are mandatory, there are other things that will make your dataset easier to process:</p>
<ul>
<li>The first line of each data sheet/ spreadsheet contains the full row name. For example, if you measure the age of the patient at the time of diagnosis, name the column as <code>AgeAtDiagnosis</code>♪ Not like ♪ <code>ADx</code> This way others may find it difficult to understand the acronym.</li>
</ul>
<h4>Examples of genomics</h4>
<p>Suppose you have 20 individual RNA-Seq genetic expression measurements. You also collect demographic and clinical information on patients, including age, treatment and diagnosis.</p>
<ul>
<li>You should have a table/ spreadsheet containing clinical/demographic information. It will have 4 columns (patient ID, age, treatment, diagnosis) and 21 rows (one line variable, followed by one row for each patient).</li>
<li>You should have a summary genome data spreadsheet. Usually, such data are aggregated at the count level of each external item. If you have 100,000 externals, you should have a table/ spreadsheet containing 21 rows (one genetic name per patient, one row) and 100,001 columns (one patient ID, one column per data type).</li>
</ul>
<h4>Format Recommendations</h4>
<p>If you share data with partners in Excel, clean data should be<strong>One Excel file per table</strong>Medium. They should not have multiple sheets, should not apply macros to data, and should not highlight any columns/cells.
Alternatively, share data using a CSV or TAB-separated text file. (But be careful when reading CSV files into Excel, which sometimes leads to the unrecurring processing of date and time variables.</p>
<h3>3. Code book (A code book)</h3>
<p>For almost any data set, the measurements you calculate require more detailed descriptions than you can plug in in your spreadsheet. The code book is used to contain this information. It's like a “telegraph” for your data.</p>
<p>At the very least, this note should enable statisticians to understand:</p>
<ol>
<li>Information on variables not contained in the clean data (<strong>Including units!</strong>）。</li>
<li>Information about the sum of options you made.</li>
<li>Information about the experimental research design you used.</li>
</ol>
<p>In our genomics case, analysts would like to know what the unit of measurement is for each clinical/demographic variable (age is year? Is the treatment a name or a dose? What is the level of diagnosis and its heterogeneity? They would also like to know how you chose the outsiders to summarize the genome data (UCSC/Ensembl, etc.). They would also like to know any other information on how you are conducting the data collection/research design. For example, are these the first 20 patients who went into the clinic? Are they carefully selected 20 patients by certain characteristics, such as age? Are they randomly assigned to treatment?</p>
<p>The common format for this document is Word (in 2026 in this Rept, perhaps.md has become a close-mainstream format). There should be a chapter called “Research Design” detailing how you collect data. There is also a chapter called "Codebook " describing each variable and its unit.</p>
<h4>Variable code recommendations</h4>
<p>When you put variables in spreadsheets, you encounter several main types:</p>
<ol>
<li><strong>Continuous Variables (Continuous)</strong>: Anything measured on a quantitative scale may be any fraction (e.g. weightkg).</li>
<li><strong>Order Variable (Ordinal)</strong>: with fixed, small (&lt;100) Level but sequenced data (e.g. survey responses: poor, general, good).</li>
<li><strong>Categorical Variables</strong>• There are several categories, but no order of data (e.g. sex: men, women).</li>
<li><strong>Missing Data (Missing)</strong>: No data observed. Should be coded <code>NA</code>。</li>
<li><strong>Deleted Data</strong>: You know that there is some lack of data on the mechanisms (e.g., the measurement values are below the test limit or the patient is missing).</li>
</ol>
<p><strong>Avoid Encoding Classification or Order Variables into Numbers</strong>I'm sorry. In the clean data, the value for gender should read &quot;male&quot; or &quot;female&quot;, instead of one or two. The orderly value should read &quot;poor&quot;, &quot;fair&quot;, &quot;good&quot;, not one, two, three. This avoids potential confusion about the direction of effects and helps identify coding errors.</p>
<h3>4. Processing records (The processing list/script)</h3>
<p>You may have heard of it,<strong>Repetitivity is a big thing in computing science.</strong>I'm sorry. This means that when you present your paper, the reviewers and others around the world should be able to fully recreate the analysis from the original data to the end result.</p>
<p>If you try to improve efficiency, you are likely to implement some of the aggregation/data analysis steps before the data are considered “clean”.
The ideal thing is to create a<strong>Computer Script</strong>(in R, Python or other languages) The script is entered with the original data and the exact data you share. You can try running scripts several times to see if the code produces the same output.</p>
<p>If you can't write code, you should give a statistician a name for it.<strong>Hypocrite</strong>Something. It should look like this:</p>
<ol>
<li>Step 1 - Get original files, summarised software for running version 3.1.2, with a = 1, b = 2, c = 3.</li>
<li>Step 2 - Run the software for each sample.</li>
<li>Step 3 - The third column of the outputfile.out for each sample is the corresponding row in the output data set.</li>
</ol>
<p>You should also include information on the system you use (Mac/Windows/Linux) and the software version.</p>
<h2>What do you expect from an analyst?</h2>
<p>When you transfer a properly organized data set, it significantly reduces the workload of statisticians. So I hope they can get back to you faster. But most careful statisticians will check your processing records, ask you about the steps and try to ascertain whether they can obtain the same neat data as you through a spot check.</p>
<p>Then you should expect from statisticians:</p>
<ol>
<li>I'm gonna do every analysis.<strong>Analyse Script</strong>(not just an indication).</li>
<li>They're used for running analysis.<strong>The exact computer code.</strong>。</li>
<li>Everything they produce.<strong>Output file/chart</strong>。</li>
</ol>
<p>These are the information you will use to supplement the material to create the repetitivity and accuracy of the results. Every step of the analysis should be clearly explained, and if you do not understand what the analyst has done, you should ask. Understanding statistical analysis is a shared responsibility of statisticians and scientists. You may not be able to do the exact analysis without the code of a statistician, but you should be able to explain to your lab partner/your chief researcher why the statisticians do every step.</p>
