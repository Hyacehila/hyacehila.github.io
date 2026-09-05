---
title: 'R Time Series Analysis: Time Series Objects, ARIMA, and VAR'
title_zh: R 时间序列分析：ts、zoo、xts、ARIMA 与 VAR
date: 2024-05-04 21:38:03 +0800
categories:
- Programming
- Programming Languages
tags:
- R
- Time Series
author: Hyacehila
mathjax: true
hidden: true
excerpt: Covers ts, zoo, xts, financial time series, ARIMA, VAR, state space models, and MARSS.
description: Covers ts, zoo, xts, financial time series, ARIMA, VAR, state space models, and MARSS.
excerpt_zh: 整理 ts、zoo、xts、金融时间序列、ARIMA、VAR、状态空间模型与 MARSS 的基础用法。
permalink: /blog/2024/05/04/r-time-series-analysis-learning-notes/
lang: en
translation_key: 2024-05-04-r-time-series-analysis
translation_status: machine
translation_source_hash: 2f0d544ca072b5e2e7af11fe3f44d6f403d8048b92969d2f860e818833abca10
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<h2>About Time Series Analysis</h2>
<p>The time series analysis is designed to help us deal with a particular type of data. Time series are the most common model for modelling in the economic and financial sphere and are also applied in a wide range of fields;</p>
<p>The most important tools in time series analysis are the ARIMA model and the season ARIMA model that it derived; of course, there are many models that are used for different problems, and then one is presented.</p>
<h2>Type of time series data and necessary basis</h2>
<p>Time series data can be saved in the vector of R, or in one or more columns of the data box of R, which is saved separately or in the same data box at the same time;</p>
<p>However, due to the special nature of time series data, R provides some specific formats for the preservation of time series data, which are essentially a special data box, but provide a more interactive syntax.</p>
<p>The ts category is the basic R time series category, and many functions are performed in the form of ts, and we should use it when we need to call. <code>as.ts</code> Talk about the format of the data
zoo forms are an extension of the ts form, allowing for a series of realizations with varying intervals; the whole zoo form is absorbed by the xts form, and more advanced time series analysis functions are constructed on the xts form
<strong>We usually store data in xts format, convert it to ts format when needed.</strong></p>
<h3>ts type</h3>
<h4>Create tstype sequence</h4>
<p>ts are the type of rule time series supported in the stats package of basic R software, with<code>start</code>and<code>frequency</code>Two Properties</p>
<p>Day data should not be labelled as the type of ts for a specific calendar time, because the day data for financial data are generally not available on weekends and holidays, and the ts type requires that the time is connected every day, and cannot jump directly to Monday on Friday.</p>
<p>Of course, if we connect directly, and we connect directly to the transaction date, then there may be problems with multiple aspects, because some data may be collected on non-trade days.</p>
<p>ts type used to save equal time series of singles or multiples, e.g. monthly, quarterly, annual data, for example, by generating</p>
<pre><code class="language-r">ts(x, start=c(2001, 1), frequency=12)
</code></pre>
<p>of which<code>x</code>is a vector or matrix, and each column of the matrix is a sequence when taking the array values. <code>frequency</code>The monthly data is 12, the quarterly data is 4, and the annual data can be defaulted (value 1)</p>
<p><strong>Ts type is not generally used for daily observations, and if necessary, set annual data frequency 1</strong></p>
<h4>Common function for ts series</h4>
<p>Use<code>ts.intersect</code>Functions and<code>ts.union</code>function to combine two or more time series into multiple time series, and to cross-set or to combine.</p>
<p>For the use of serial data for calculations, drawings, etc., you can use<code>as.vector</code>Convert data from a single-dollar time series to a normal vector</p>
<p>For Multi-Time Series<code>x</code>Yes, I can.<code>x[,1]</code>This format takes out the time series of the timescales, which can be used.<code>as.vector(x[,1])</code>Converts the amount of the mass to the normal vector, which can be used by using the xts type<code>coredata(as.xts(x))</code>Converts data from multiple time series to a normal matrix.</p>
<p>Use<code>start()</code>The time series starts at the beginning. <code>end()</code>The time series end, <code>frequency()</code>Sample frequency</p>
<p><code>aggregate()</code>The function adds monthly data to total adult data, his role and the normal data box.<code>aggregate()</code>function function type calculates a statistical amount by a label The time series sets this classification variable for time</p>
<p><code>time()</code>function returns the time of each time point in the series for the type of ts data, and the result is the same time series as the original time series. <code>cycle()</code>Function returns the month of each time point of the series to the monthly data, and results in the same time series as the original time point</p>
<p><code>window()</code>Function takes out a part of the time series, if specified<code>frequency=TRUE</code>It can also be taken out only for a month (quarterly)</p>
<p><code>filter</code>Function to calculate the filters of an incremental or a volume similar to the apply series of zoo type and xts type</p>
<h3>zoo type</h3>
<p>R's zoo extension provides a more flexible type of time series than the basic R-rings time series type, with time labels (time stamp) that use any date and time type in R, and sequences do not need to be time-spaced to support multiple time series.</p>
<p>If the sequence meets the requirements of the ts type, it is compatible with the ts type and can be converted to one another. zoo also provides functions that are identical or similar to those of the class of ts, to the extent possible. The time series data type provided by the zoo extension is called zoo type</p>
<p><strong>As an extension of the original form of the ts, we can use the syntax rules and functions of the grammaticals that are boldly combined until we consider them separately when we have made a mistake.</strong></p>
<h4>Create a zoo-type sequence</h4>
<p>Generate a time series of zoo types, with two parts of input: a vector or matrix<code>x</code>As an observation, a subscript sequence<code>order.by</code>as sorted variables and time labels. <em>Data box not accepted</em></p>
<p><strong>The design feature of the zoo is to allow any sort of sorted data type to be used as a time tag</strong></p>
<pre><code class="language-r">zoo(x, order.by)
</code></pre>
<p>Succession <code>ts</code> The type of idea that we're going to have each line to a fixed time node, which means a multiple time series of time-sharing times, but we can accept the existence of the NA, so this flexibility is enough.</p>
<p>Examples</p>
<pre><code class="language-r">## 一元的时间序列
set.seed(1)
z.1 &lt;- zoo(sample(3:6, size=12, replace=TRUE),
           make_date(2018, 1, 1) + ddays(0:11));
## 多元的时间序列
set.seed(2)
z.2 &lt;- zoo(cbind(x=sample(5:10, size=12, replace=TRUE),
                 y=sample(8:13, size=12, replace=TRUE)),
           make_date(2018, 1, 1) + ddays(0:11));
</code></pre>
<p>I can see it.<code>order.by</code>Some of the functions generated the time tags, and they were introduced later.</p>
<p>ts, irts (defined in tseries packages), ts, etc.<code>as.zoo()</code>Convert to zoo type (<strong>Extremely inclusive.</strong>) , if the time subscript of the zoo type meets the conversion requirements, you can also use the zoo time series<code>as.xxx()</code>class functions are converted to other time series types.</p>
<p>Time series data saved in text files, strings, data boxes can be used<code>read.zoo()</code>Converts to the zoo time series.</p>
<h4>zoo type generic extension</h4>
<p>The zoo type is a more central time series extension type, and he expands some functions.</p>
<p><code>print(x)</code>Show<code>x</code>, horizontally display a single-serial sequence, vertically display a multi-serial series</p>
<p><code>str(x)</code>Show<code>x</code>This is the original function.<code>str</code>Extension</p>
<p><code>head(x)</code>and<code>tail(x)</code>You can take out several entries at the beginning of the sequence and several at the end, which are also extended and broad functions.</p>
<p><code>summary(x)</code>A simple summary of each sequence and time is also an extended generic function.</p>
<h4>zoo-type subset extraction</h4>
<p>We said that the time series is all in extended data box form so extraction syntax can be done with reference to data frames and arrays.</p>
<p>Neither a dollar nor a multi-sequence is taken in a non-changed manner.</p>
<pre><code class="language-r">## 提取第一行
z[1]
## 提取一些行
z[10:12]
#提取一些行和第二列
z.2[1:3, 2]
## 使用时间下标进行提取行
z.2[ymd(c(&quot;2018-01-01&quot;, &quot;2018-01-12&quot;))]
</code></pre>
<p>Datetime string format as<code>CCYY-MM-DD HH:MM:SS</code>, and can omit the rest of the text, which means taking out all the time that the front part matches Points</p>
<h4>zooreg type</h4>
<p>To match the type of ts of the rule time series, the zoo extension provides the zoo-type zooreg, representing the time series between the rules, which has the same time-spacing information as the ts, but<strong>Allows some internal points to exist</strong></p>
<p>Use<code>zoo()</code>Function plus<code>frequency</code>Options or <code>zooreg()</code>Function generates a time series of zooreg types</p>
<pre><code class="language-r">zoo(x, order.by, freqency)
zooreg(x, start, end, frequency, deltat, ts.eps, order.by)
</code></pre>
<p><strong>The advantage of the zooreg type bits is to allow non-NA deficiencies to remain the same.</strong>  And it's also supported by default, by sky-based observations.</p>
<p>Yes, it is.<code>is.regular(x)</code>Determines whether a zoo type of data is a rule, and uses<code>is.regular(x, strict=TRUE)</code>Rules sequence to determine whether a zoo-type data meets the same conditions as ts</p>
<p>Use<code>as.ts(x)</code>Converts a rule-based zooreg or zoo type data to a time series of ts type. If<code>x</code>There are missing points of time inside, which are filled with observations when converted to ts type<code>NA</code> Use<code>as.zoo(x)</code>Convert time series of ts type to zoo type</p>
<h4>zootype function</h4>
<h5>zootype merge</h5>
<p>Two saved time series of different time periods<code>x</code>, <code>y</code>Yes, it is.<code>c(x,y)</code>Or... <code>rbind(x,y)</code>Merges, two different attribute time series, which can be merged into a multiple time series, time to take together, and value missing. Two sequences don't need to be long.</p>
<h5>zoo-type downs</h5>
<p>Similar to the ts time series, for the zoo time series, it can also be used<code>aggregate()</code>We're going to use the frequency to reduce the data, and we're going to integrate the observations over hours into a longer time frame, which is a frequency change for the time series.</p>
<pre><code class="language-r">z.apy &lt;- aggregate(z.ap, year, sum)
</code></pre>
<p>It's also a function.<code>aggregate</code>Extension</p>
<h5>zoo type filling</h5>
<p>When the sequence is missing, we have a simple way to fill it, using it.<code>na.locf()</code>Fills in the missing value, fills in the last non-missive value in front, and uses<code>na.approx()</code>You can fill missing values as linear plugins, and more ways to check the document to get</p>
<h5>zootype changes</h5>
<p>The two sequences can be multiplied by four operations and logic comparisons, with the result that the corresponding operation is at the time point. Missing value deleted</p>
<p>Actual data of the zoo time series type can be used<code>coredata()</code>Read or modify. The result for a single-sequence sequence when read is a numerical vector, and for a multi-sequence sequence, the matrix, we can use the matrix to make the original data modified, as if</p>
<pre><code class="language-r">z.2b &lt;- z.2
coredata(z.2b) &lt;- 100 + coredata(z.2b)
z.2b
</code></pre>
<h5>zoo-type slide average</h5>
<p>The average of a slide like the seven-day average is often calculated for time series. The function of the basic R for the general vector<code>filter()</code>Weighted average slide and self-regressive iterative calculations can be performed. zoo packages provided time series and ts time series for zoo<code>rollapply()</code>function, you can calculate multiple scroll calculations, including average slides</p>
<p>A separate function is provided for some commonly used scrolling calculationszoo, such as <code>rollmean()</code>, <code>rollmedian()</code>, <code>rollmax()</code>, <code>rollsum()</code>, these are the result points corresponding to the scroll window centre. And...<code>rollmeanr()</code>, <code>rollmedianr()</code>, <code>rollmaxr()</code>, <code>rollsumr()</code> The result points are the end of the scroll window.</p>
<h3>xts type</h3>
<h4>Succession and development</h4>
<p>The goal of the xts package is to make the xts-in-type function compatible with other time series types of input and to easily add properties.<strong>In fact, xts have become the first priority for time series analysis.</strong></p>
<p>Xts package provides the xts time series type, which is essentially the zoo type of the zoo package, so the approach to the zoo type also applies to the xts type. He just made a small change in the zoo type.</p>
<h4>Create xts type</h4>
<p>Yes, it is.<code>xts()</code>Generate new data objects of xts time series type, similar to those used<code>zoo::zoo()</code></p>
<p>Other time series types can be used<code>as.xts()</code>Convert to xts type</p>
<h4>Subset of xts type data</h4>
<p>His seamless connection.<a href="/en/blog/2024/05/04/r-time-series-analysis-learning-notes/">zoo-type subset extraction</a>All the ways</p>
<p>Yes, it is.<code>&quot;from/to&quot;</code>format specifies a date time frame, and does not require data at start and end points as well as:</p>
<pre><code class="language-r">xts.1[&quot;2018-01-10/2018-01-14&quot;]
</code></pre>
<p><code>first(x, n)</code>and<code>last(x, n)</code>Similar to<code>head(x, n)</code>and<code>last(x, n)</code>, but for xts objects<code>x</code>，<code>n</code>In addition to the positive integer values, the string is allowed to specify the length of time, including the secs, seconds, strings, units, days, weeks, months, questions, years. Like what?</p>
<pre><code class="language-r">first(xts.ap, &quot;3 months&quot;)
</code></pre>
<p>When a negative value is taken in a string, it is deducted.</p>
<h4>xtstype function</h4>
<h5>xtstype extension functions</h5>
<p>Expands the generic function <code>plot</code> Query<code>plot.xts</code>Yeah.</p>
<p>Yes, it is.<code>coredata(x)</code>Back<code>x</code>not containing time; using<code>index(x)</code>Back<code>x</code>Time tag</p>
<p><code>periodicity(x)</code>Ask for xts objects<code>x</code>Time frame</p>
<p><code>endpoints(x, on)</code>Gives a point of delimitation by a certain frequency, which includes<code>&quot;us&quot;</code>(microseconds), <code>&quot;microseconds&quot;</code>, <code>&quot;ms&quot;</code>(ms), <code>&quot;milliseconds&quot;</code>, <code>&quot;secs&quot;</code>, <code>&quot;seconds&quot;</code>, <code>&quot;mins&quot;</code>, <code>&quot;minutes&quot;</code>, <code>&quot;hours&quot;</code>, <code>&quot;days&quot;</code>, <code>&quot;weeks&quot;</code>, <code>&quot;months&quot;</code>, <code>&quot;years&quot;</code></p>
<h5>xts Financial Time Series Decline</h5>
<p>For the financial time series in the form of OHLC, i.e. open, top (high), lowest (low) and closing (clos) components, and the composition variable is also used.<code>Open</code>, <code>High</code>, <code>Low</code>, <code>Close</code>Yes, I can.<code>to.period(x, period)</code>Take it down to the frequency.<code>period</code>Specified sampling frequency <strong>Based on the financial time series habits, the selection of the closing price is a default choice.</strong></p>
<p>If<code>x</code>It's the minutes. <code>to.minutes3(x)</code>, <code>to.minutes5(x)</code>, <code>to.minutes10(x)</code>, <code>to.minutes15(x)</code>,<code>to.minutes30(x)</code>, <code>to.hourly(x)</code> Will<code>x</code>Data on down frequency from 3 to 60 minutes.</p>
<p><code>to.daily(x)</code>Will<code>x</code>The drop frequency is the daily data, and the time portion is deleted from the time subscript. <code>to.weekly(x)</code>Will<code>x</code>The drop frequency is the weekly data, and the time portion is deleted from the time mark, the date being the date of the last day of the week (swiss).<code>to.monthly(x)</code>Will<code>x</code>The time is down to the yearbook and the time is changed to the yearmon type. <code>toquarterly(x)</code>Will<code>x</code>Reduces the frequency to the quarterly data and replaces the time down to the type of yearqtr. <code>toyearly(x)</code>Will<code>x</code>Drop frequency to annual data, date used for the last day of the year for which data are available</p>
<h5>xts type slide average</h5>
<p>Use<code>period.apply(x, INDEX, FUN)</code>Yes, it is.<code>INDEX</code>For Time Series<code>x</code>Group, for each group<code>FUN</code>function calculates a function. <code>INDEX</code>Commonly<code>endpoints(x, on=&quot;...&quot;)</code>, give some sort of grouping cycle. Like what?</p>
<pre><code class="language-r">period.apply(xts.ap,
             INDEX=endpoints(xts.ap, on=&quot;years&quot;),
             FUN=mean)
</code></pre>
<p>Common operations such as sum-up have specific functions, e.g.<code>period.sum()</code>, <code>period.min()</code>, <code>period.max()</code>, <code>period.prod()</code>I'm sorry. Special function functions are more efficient.</p>
<p>The usual cycles also have specific functions, such as:<code>apply.daily()</code>, <code>apply.weekly()</code>, <code>apply.monthly()</code>, <code>apply.quarterly()</code>, <code>apply.yearly()</code>I'm sorry. These functions are actually called.<code>period.apply()</code></p>
<h3>Quantmod package</h3>
<p>The purpose of the Quantmod package is to<strong>Development of testing tools for a prototype that facilitates quantification of investors</strong>Instead of providing new statistical methods.</p>
<p>Quantmod packages provide some convenient features of financial time series data, such as loading data from open data sources, stock pattern, time series, etc.</p>
<p>Quantmod package provides<code>getSymbols()</code>function, you can download financial and economic data from multiple open data sources and convert them to R format (mainly xts format)</p>
<p><code>chartSeries()</code>The K-line and curves are all the features that are designed to analyze the financial time series, which are a good visualization of the financial time series.</p>
<h2>Linear Time Series Model</h2>
<p>And we recommend linear time series analysis, and at the same time, an example of linear time series analysis is stored in the R-language study case file that allows us to understand the algorithms that achieve the whole linear time series analysis.</p>
<h3>Basic processing</h3>
<p>Whatever type of time series object we use, these basic processing functions are common.</p>
<p>Expand pane function <code>plot()</code> Do the minimum time series curve</p>
<p>by <strong>stats</strong> Package Provider Functions <code>acf</code> The blog also shows how the sample is used to map the situation. <strong>forecast</strong>The package provides a similar function.<code>Acf()</code>Function It doesn't keep a single step lag to focus on the relevance behind.
<code>pacf</code>The function calculates that the sample is based on the relevant figure.<strong>forecast</strong>Packages provided<code>Pacf</code>It's the same thing.
These functions also give us the value of the correlation coefficient.</p>
<p><code>lag()</code>You can calculate the lag sequence, input to the ts type, <code>lag()</code>The effect is that the serial number remains the same, but the time label adds a unit or uses<code>k=</code>Specified interval</p>
<pre><code class="language-r">## 这才是传统意义上的滞后一个单位 需要设置-1
x2 &lt;- stats::lag(x1, k=-1); x2
</code></pre>
<p><code>diff(x)</code>The first step of the difference is calculated. <code>diff(x, lag, differences)</code>Calculating Delays<code>lag</code>step places<code>differences</code>The difference is the difference between the seasons.</p>
<h3>ARIMA Miscellaneous</h3>
<p><code>arima.sim</code>It's a simulation of the data that generates the ARIMA model.</p>
<p>Supplementary ARFIMA
<code>fracdiff::fdGPH()</code>Geweke-Porter-Hudak estimate for calculating differentials
<code>fracdiff::fracdiff()</code>Function to perform ARFIMA model estimation</p>
<h3>ARIMA Model Identification</h3>
<p>The basic recognition of the ARIMA model requires the use of ACF and PACF curves to assist in<a href="/en/blog/2024/05/04/r-time-series-analysis-learning-notes/">Basic processing</a>
Here we repeat the main points of the basic treatment:<code>acf()</code> The blog is a tool for viewing the samples from the map.<code>pacf()</code> (b) To view samples for the relevant maps;<code>forecast::Acf()</code> and <code>forecast::Pacf()</code> Provides similar features and is more appropriate for quick viewing of the structures at the time of modelling identification.</p>
<p><strong>stats</strong>Package<code>ar()</code>The function can model time series samples in AR.</p>
<p><strong>forecast</strong>The bag's a gift.<code>auto.arima()</code>function, you can automatically make model selections, but often people don't get sick.</p>
<p><strong>TSA</strong>Packaged<code>eacf()</code>Function Identification Model.</p>
<p><strong>TSA</strong>The bag is also available.<code>armasubsets()</code>Function to select the ARMA model step</p>
<h3>ARIMA model preparation</h3>
<p><code>arima()</code>The function estimates the general ARIMA model, but it needs to pre-specify the steps.
<code>arima</code>Function allows the specified coefficients to be fixed to predefined values Use parameters<code>fixed</code>We're sure, we'll set the smaller coefficients directly to zero, so we can achieve a thin estimate.
<code>arima()</code>function can be used<code>seasonal=</code>Specify seasonal models, including seasons of AR, seasonal differentials, seasons of MA and cycles
<code>arima()</code>Function provides <code>xreg=</code> And he's using the retrogressive variables, and he's using the retrogressive variables as the original sequence.
I can see that. <code>arima()</code>Models are very common time series analysis functions.</p>
<blockquote>
<p> Support for smooth and reversible ARMA modelling, which can be used<code>include.mean=TRUE</code>Sets the parameters with the mean value.
Support for ARIMA modelling, not allowed if the margin is greater than zero<code>include.mean=TRUE</code>, that is, the unit root process does not allow drift.
The project is based on a series of projects that support the development of a seasonal ARIMA model. The difference between seasonal and zero is not allowed to drift.
Supports regression modelling with a smooth ARMA sequence as an error.</p>
</blockquote>
<p><strong>Yes.<code>arima()</code>Function to use<code>xreg=</code>Do not specify a differential or seasonal difference when introducing a return from a variable (outside variable)</strong></p>
<p><strong>forecast</strong>Bag.<code>Arima()</code>Function<code>stats::arima()</code>function, but allow drift items when the margin is greater than zero</p>
<h3>ARIMA Model Test</h3>
<p>The normality analysis of the disability is described elsewhere, not repeated here.</p>
<p><code>Box.test()</code>And it's very useful to do a Ljug-Box white noise test, which is to test if the sequence is white noise sequences, because the residuals are calculated from model estimates, and the freedom is lost.<code>fitdf=</code>Select the number of degrees of freedom reduced to&#36;p+q&#36;</p>
<p>Yes, it is.<code>forecast::checkresiduals()</code>It's a model diagnostic, and it's doing all the usual analysis of the disability at the same time.</p>
<p>Will<code>arima()</code>, and then enter the output to<code>tsdiag()</code>function, you can make model diagnostics, which are a direct systematic diagnosis of the model.</p>
<p>The unit root test is a smooth test, and the zero assumption is a unit root, i.e., a flat; the opposing assumption is a smooth one. (b) The frequent use of enhanced Dickey-Fuller tests (ADF tests);<strong>fUnitRoots</strong>Bag.<code>adfTest()</code>function to perform unit root ADF tests. <strong>tseries</strong>Bag.<code>adf.test()</code>function can also perform unit root ADF tests;
Unit Root Check Options<code>type</code>Select the underlying model to take:</p>
<ul>
<li><code>&quot;nc&quot;</code>, which means that no drift or cut-off items are present;</li>
<li><code>&quot;c&quot;</code>, which indicates that it has a drift item or a cut-off item;</li>
<li><code>&quot;ct&quot;</code>, which means that the base model has&#36;a+bt&#36;(a) such linear items;</li>
</ul>
<h3>ARIMA model prediction</h3>
<p>Model predictions are still using classic generic functions. <code>predict()</code> This is... <strong>stats</strong>It's the most basic method to provide a predictive side of the SE.</p>
<p>It's forecast.<code>forecast</code> It's easier to predict functions.</p>
<h3>Time series breakdown</h3>
<p><strong>stats</strong>Bag.<code>decompose()</code>function. The way to move is to slide the average of the central symmetry. Yes, it is.<code>type=&quot;additive&quot;</code>or<code>type=&quot;multiplicative&quot;</code> Specifies whether to add or multiply the subparagraphs</p>
<p><strong>stats</strong>Package provides function<code>stl()</code>, the function is based on the local weighting of the return estimate, which reduces the effect of the anomaly, and is a robust return. The smooth season changes are estimated using the same month (quarterly) values, less seasonal items and then the smoothing method to estimate trends</p>
<p><strong>stats</strong>Bag.<code>StructTS()</code>The function uses a state-space model to represent time-series breakdown, and estimates the components in the largest semblance method</p>
<p><strong>stats</strong>Bag.<code>HoltWinters()</code>Function provides an index smoothing method,<strong>forecast</strong>Bag.<code>ets()</code>function provides the function of automatically selecting and forecasting the appropriate index smoothing method.</p>
<h2>ARCH Series Model</h2>
<h3>Testing of ARCH effects</h3>
<p>We have two ways to test the ARCH effect.</p>
<ul>
<li>Use function to test the balance squared white noise <code>Boxtest()</code></li>
<li>Check the difference using a minimum two-fold method.<code>FinTS::ArchTest()</code>
<strong>Note, first function enters the disability sequence squared, second direct acceptance of the disability sequence</strong></li>
</ul>
<p>We also have visual tests, that is, the ACF curve; we ask that the difference itself is a white noise sequence, and that the square of the difference is self-relevance.</p>
<h3>ARCH Series Modeling</h3>
<p><strong>fGarch</strong>Bag.<code>garchFit()</code>He's the most common function in a volatile model.
<strong>fGarch</strong>Bag.<code>garchFit()</code>Function supports multiple condition distributions, defaults as normal distributions, and uses<code>cond.dist=</code>Specified distribution:</p>
<ul>
<li><code>&quot;norm&quot;</code>(a) Normal;</li>
<li><code>&quot;snorm&quot;</code>: Is biased;</li>
<li><code>&quot;ged&quot;</code>: broad error distribution;</li>
<li><code>&quot;sged&quot;</code>: A wide-ranging error distribution;</li>
<li><code>&quot;std&quot;</code>: t-distribution;</li>
<li><code>&quot;sstd&quot;</code>: Slightly distributed;</li>
<li><code>&quot;snig&quot;</code></li>
<li><code>&quot;QMLE&quot;</code>: the maximum semblance is proposed, assuming normal but applying robust standard error estimates;
After the selection of the conditions, then the normality test is meaningless, and our target is not the normal distribution.</li>
</ul>
<p>More complex ARCH type modelling requires the use of an extension package <strong>rugarch</strong></p>
<h3>Testing of ARCH models</h3>
<p>The test of the model is to study the characteristics of the disability sequence.
<code>residuals(model, standardize=TRUE)</code> Function to calculate a standard disability
<code>summary(model)</code> It's possible to give a summary of the model quickly, including all the routine tests.
<code>plot.garch()</code> Expands the generic function<code>plot</code> You can draw all the normal detection graphics.</p>
<h3>ARCH alignment and projections</h3>
<p><code>volatility()</code> Function to match fluctuations
<code>fitted()</code> Function corresponds to the value of the model itself, which is the mean value item and the rate of fluctuations.
<code>predict()</code> Function is used for multistep predictions, and it is also an extended generic function.</p>
<h3>Two-step estimation method</h3>
<p>It's a theory that we don't need to use anything other than...<a href="/en/blog/2024/05/04/r-time-series-analysis-learning-notes/">Linear Time Series Model</a> The principle of reference financial time series analysis (one dollar): two-step estimation method.</p>
<h2>Multi-temporal series analysis and alignment analysis</h2>
<p>The R-extension package that is used mainly for multiple time series analysis is available <code>MTS</code> <code>vars</code></p>
<h3>Model estimation</h3>
<p><strong>MTS</strong>Bag.<code>VAR()</code>function to estimate VAR models, the second parameter of which sets the number of the model's steps</p>
<p><strong>MTS</strong>The VARorder function of the package can calculate the VAR rankings&#36;M(i)&#36; Statistical volumes and various information guidelines</p>
<p><strong>vars</strong>Bag.<code>VAR()</code>The function can also be used for VAR model estimates, which in different forms and MTS packages, but the calculations are consistent, and this function allows automatic step selection.</p>
<p>MTS bag.<code>refVAR()</code>Function enters unbound VAR modelling results, and<code>thres=1.645</code>or<code>thres=1.96</code>Here's the thing.&#36;t&#36;margin limit, generate binding estimate with a zero factor for the set-up component</p>
<h3>Model testing</h3>
<p><strong>MTS</strong>Bag.<code>mq()</code>function is used for multiple mixing tests, i.e. the Ljung-Box test in a dollar, allowing manual set-up of freedom deductions.</p>
<p><strong>The freedom deduction is determined by coefficient, and when multiple mixing tests are performed using the disability, yes&#36;k^2p&#36;A coefficient is estimated, which is the amount of freedom deduction we set; if some coefficients are simplified to 0, they are not included in the freedom deduction. Medium</strong></p>
<p><strong>MTS</strong>The bag also provided one.<code>MTSdiag()</code>function, input model results and<code>adj=</code>The freedom reduction is a multiple test of the CCM estimate (ACF variant), the map and the disability</p>
<p>MTS Package<code>GrangerTest()</code>Function to perform the Granger Gymmetric Test, in<code>GrangerTest()</code>Center, with<code>locInput=</code>Enter a fraction serial number to test zero assumptions: all other weights are not the cause of Granger's weight. No other tests of the Granger cause can be carried out. The use of document writing for this function is not friendly</p>
<h3>Model predictions</h3>
<p><strong>MTS</strong>Bag.<code>VARpred()</code>Function can calculate the prediction from the VAR model results, without taking into account standard prediction errors (Standard error of assumptions), and for the estimation errors (Root means squared error of assumptions)</p>
<h3>Co-ordinated analysis and vector correction model</h3>
<p>Extension<strong>tseries</strong>Medium<code>po.test()</code>The Phillips-Ouliaris co-ordinated test based on the EG two-stage approach, with zero assuming non-coherence and opposing assumption being the existence of a concoction.</p>
<p>Extension<strong>urca</strong>Yes.<code>ca.jo()</code>Function allows two tests to calculate Johansen's</p>
<h2>State spatial model</h2>
<p>Many extension packages in the R software support the modelling of state space models.</p>
<ul>
<li>Statespacer: Supports linear Gaussian state time series modelling.</li>
<li>KFAS: Supports linear Goss time series modelling, and supports non-Gaz situations in the index distribution group.</li>
<li>dlm: Dynamic linear models using models (West and Harrison 1997). Supports linear Goss time series models, which use the maximum semblance of estimates, and supports time-variable models.</li>
<li>dynr: Support modelling with a break time or continuous duration with a mechanism to switch.</li>
<li>dse: Linear Goss' ARAMA, VAR and state spatial models, using methods that are not very consistent with R's usage.</li>
<li>bssm: No-linear, non-Gross spatial model beyers extrapolation.</li>
<li>MARSS: Multi-dimensional self-regression spatial model. Parameters can vary over time and observations can contain missing values.</li>
<li>MSM: The one dollar self-regression model for the Marcov mechanism, which supports linear and broad linear models.
We're just introducing a few of them here.</li>
</ul>
<h3>statespacer</h3>
<p>Statespacer packages support linear Goss status spatial modelling, document details, model markings and initialization ideas used to refer to financial time series analysis (one dollar): linear Goss status spatial model</p>
<p>A simpler setup function is provided for commonly used structural time series models, ARMA, etc. Insufficient support for the changing situation of the various matrices.</p>
<p><code>statespacer()</code>is the main modelling function. For common models, you can use options to specify the model directly. Need to enter the initial value of the superparameter, <strong>It takes knowing how models are expressed in state space.</strong>。</p>
<p>It was all stored in a deep-snack list.<code>statespacer()</code>List of the results, which are visited with the following components:</p>
<ul>
<li><code>system_matrices</code>System Matrix<ul>
<li><code>H</code>, <code>Z</code>, <code>T</code>, <code>R</code>, <code>Q</code>Wait.</li>
</ul>
</li>
<li><code>predicted</code>: One step predicting distribution<ul>
<li><code>yfit</code>: One step forecast</li>
<li><code>v</code>: Errors in step predictions</li>
<li><code>Fmat</code>: the error range of the step forecast</li>
<li><code>a</code>: One step forecast</li>
<li><code>P</code>: the error range of the step forecast</li>
<li>…………</li>
</ul>
</li>
<li><code>filtered</code>: average filter distribution<code>a</code>The square.<code>P</code>。</li>
<li><code>smoothed</code>: Smooth results<ul>
<li><code>a</code>Average of smooth distribution:</li>
<li><code>V</code>: A smooth distribution square array</li>
<li>…………</li>
</ul>
</li>
</ul>
<h3>MARSS</h3>
<p>The model structure for the reference is financial time series analysis (one dollar): the model of the MARSS package, the MARSS extension, has a detailed hundreds of pages of user manuals that we can refer to when needed.</p>
<p>The basic function of the MARSS package is<code>MARSS()</code>Use as</p>
<pre><code class="language-r">fit &lt;- MARSS(y, model=list(...))
</code></pre>
<p>of which <code>y</code> It's a time series to model, and if it's a dimension, it can be a normal R vector, and more generally it can be entered into one.&#36;n\times T&#36;Matrix, where&#36;n&#36;It's a observation.&#36;\boldsymbol{y}_t&#36;The number of dimensions,&#36;T&#36;It is a time-series observation time point, and each column corresponds to one time point. Observation values allow for missing values.</p>
<p><code>model</code> Enter with a List <code>B </code>, <code>U</code>, <code>0</code> Equivalents, variable names are analysed in financial time series (one dollar): marks in the model of the MARSS package, but&#36;\boldsymbol\pi&#36;Use <code>x0</code> - Show. If you have a specific appointment, <code>V0</code> , and means&#36;\boldsymbol{x}_{0}&#36;Prevaluation distribution is specified as average <code>x0</code> Square range. <code>V0</code></p>
<p><code>MARSS()</code>Return one<code>marseMLE</code>object of type, which can be extracted from or further analysed by various information extract functions:</p>
<ul>
<li><code>print(MLEobj)</code>Shows the main results. <code>summary(MLEobj)</code>Shows less results.</li>
<li><code>coef(MLEobj)</code>Extract parameters estimate. Use<code>tidy::broom(MLEobj)</code>Extracts into the data box format.</li>
<li><code>residuals(MLEobj)</code>The predicted, filtered or smoother difference from the observation or state, returns the data box form.</li>
<li><code>tsSmooth(MLEobj)</code>Extract predictions, filters or smooth results, and use<code>type</code>argument selects, the default is smooth, and returns the data box form. Options<code>interval = &quot;confidence&quot;</code>It can also output smooth projection ranges. <code>fitted(MLEobj)</code>Default to predict by one step, and no noise portion is estimated when predicting, filtering, smoothing, so the observation factor value should be used to codify. Supports estimates of missing data. It's available in the prediction.<code>n.ahead</code>Specifies the number of steps to be used for multi-step projections.</li>
<li><code>logLik(MLEobj)</code>Returns a logarithmic function.</li>
<li><code>AIC(MLEobj)</code>Return AIC value, <code>AICc(MLEobj)</code>It is a variant that amends the circumstances of the small sample. <code>MLEobj &lt;- MARSSaic(MLEobj)</code>Add more AIC-class criteria.</li>
<li><code>MARSSkf(MLEobj)</code>Filter, smooth, and result: <code>xtt1</code>(a) is a step forward in forecasting expectations; <code>xtt</code>It is the state filtering expectation; <code>xtT</code>It's the smooth expectation of the state. <code>Vtt1</code>, <code>Vtt</code>, <code>VtT</code>A step forecast, filtering, smooth range estimation, etc., see the MARSS User Manual §3.3 and §5.10.</li>
<li><code>MARSSparamCIs()</code>Calculate parameter confidence interval, use sea-colored array by default, and use<code>method = &quot;parametric&quot;</code>Specifies that the argument Bootstream method is used, and that<code>method = &quot;innovation&quot;</code>Specifies that you use the new utensils method. For square array parameters, the sea-color array method should not be used. The Bootslap method is long and should be used only when sufficient.</li>
<li><code>MARSSboot(MLEobj)</code>The confidence interval, deviation estimation, etc., can be used using the Bootstream method, either by parameter or by new valor sampling, and only by supporting new valorization when the observation values contain missing values.</li>
<li>There are also further calculated functions, see § 2.4 of the user manual in the MARSS document.</li>
</ul>
<h2>The Herma Model.</h2>
<p>Available R-extension packages:</p>
<ul>
<li>depmixS4;</li>
<li>HiddenMarkov;</li>
<li>msm;</li>
<li>R2OpenBUGS (for Bayesian estimation);</li>
<li>HMM (class values time series only supported).</li>
</ul>
