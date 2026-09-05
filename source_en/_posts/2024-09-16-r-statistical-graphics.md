---
title: 'R Statistical Graphics: Base Graphics, lattice, and ggplot2'
title_zh: R 统计图形：base、lattice 与 ggplot2
date: 2024-09-16 22:32:48 +0800
categories:
- Programming
- Programming Languages
tags:
- R
- Data Visualization
- ggplot2
author: Hyacehila
mathjax: false
hidden: true
excerpt: Covers classic graphics, plotting principles, base R graphics, and ggplot2 layers, geoms, scales, facets, and themes.
description: Covers classic graphics, plotting principles, base R graphics, and ggplot2 layers, geoms, scales, facets, and
  themes.
excerpt_zh: 整理经典统计图形、作图经验、base R 作图系统，以及 ggplot2 的图层、几何对象、标度、分面与主题。
permalink: /blog/2024/09/16/r-graph-learning-notes/
lang: en
translation_key: 2024-09-16-r-statistical-graphics
translation_status: machine
translation_source_hash: bb47bd55676111cc21c0b544593d07e8bd25b40c65ea96e49abb8e902c2e0fa7
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<p>We would like to talk about how we should make the drawings, that is, the principle of drawing them, before they start.</p>
<p>We're moving to the full library. <a href="/en/blog/2024/03/15/r-visualization-learning-notes/">R Visualization</a>It is presented, thus reducing the reading burden.</p>
<h2>Classic Graphics</h2>
<p>The point of statistical graphics is to guide us in observing the information in statistics. The greatest value of a graphic is when it reaches us to notice what we never expected to see. In this sense, the importance of statistical graphics is self-evident.</p>
<p>In the history of statistical graphics, there are not many images that can reach the height of "discovering unforeseeable information".</p>
<h3>The origin of the pie and the line.</h3>
<p>Both the pie and the graphs are based on very long-standing statistics, and they're all invented by Playfair; although they don't look strange now, they're almost 300 years old.
<img src="/assets/images/r-learning-notes/r-language-stat-visualization.png" alt="Language R Statistical visualization">
This chart shows the time series of imports and exports of England between 1700 and 1780, and the left shows that foreign trade is not good for England, while foreign trade has gradually become beneficial after about 1752.
<img src="/assets/images/r-learning-notes/r-language-stat-visualization-01.png" alt="Statistical Visualization-1">
Large maps above show the size of each country ' s territory (in proportion to the circle) and the population (in proportion to the left) and tax revenues (in proportion to the right) and the proportion of the country distributed across all continents.</p>
<h3>Cholera transmission</h3>
<p>John Snow found clear geographical patterns in the location of the deaths, as shown in the figure below.
<img src="/assets/images/r-learning-notes/r-language-stat-visualization-02.png" alt="Statistical visualization-2"></p>
<h3>Rose Chart</h3>
<p>Nightingale was a pioneer in the historical use of polar coordinates. It's like a rose, and it's later called a rose map. The main idea is to use the area of the petal as the size of the statistical value.</p>
<p>The Rose Map not only clearly illustrates the change in the number of military deaths in the two years, but, more importantly, she has marked three deaths per month in different colours: Blue means death from preventable diseases, red means death from war, black means death from other causes. In this way, we can clearly understand the structure of the causes of military casualties, especially “the vast majority of soldiers die from preventable diseases” (the highest petals in the picture). With this important message, she has made the British Government aware that what really affects war casualties is not the war itself, but the lack of effective medical care by the army!</p>
<p><img src="/assets/images/r-learning-notes/r-language-stat-visualization-03.png" alt="Language R Statistical visualization 3"></p>
<h3>Napoleon's Russian expedition.</h3>
<p>The maps produced by Minard show the route (first half) of Napoleon's legions marching into Russia in 1812 and the temperature changes (second half) during the withdrawal. In this historic event, there was a dramatic decline in the number of French troops and an all-embracing picture of harsh weather conditions.</p>
<p>It includes the following information.</p>
<ul>
<li>Position and direction of the army, as well as branching and integration of the army on the way</li>
<li>Reduction in the number of soldiers</li>
<li>The temperature changes during the evacuation are shown in the lower half of the chart.
<img src="/assets/images/r-learning-notes/r-language-stat-visualization-04.png" alt="Statistical Visualization 4"></li>
</ul>
<h3>Summary and beginning</h3>
<p>Each of the first four subsections of statistics was created before the computer was born, all by hand, by the author; but without prejudice, they were of great value.</p>
<p>There are, of course, a large number of successful researchers in the area of statistical graphics who have contributed a lot to the development of statistical graphics; here is a brief introduction.</p>
<p>It's more famous to have data on da Vinci.<strong>Data-Ink Ratio</strong> and <strong>Minimizing Chartjunk</strong> He's got a lot of work worth reading. Look.</p>
<p>The real integration of statistical graphics into the mainstream depends on John.W. Tukey, who's proposed exploratory data analysis technology, has led to a new direction in statistics that has injected a new dynamism into the dominant statistical community, with the main tool for exploratory data analysis being statistical graphics.</p>
<p>Wilkinson offers a good framework for a more theoretical interpretation of statistical graphics, which is the basis for ggplot's birth.</p>
<p>He's probably one of the first statisticians to study the impact of statistical graphics on readers' mental awareness.</p>
<p>M. Friendly and Denis collated and documented more influential statistical images from the centuries before the 17th century to the present.</p>
<p>Recent statistics, using John.W.Tukey's exploratory data analysis as a landmark starting point, have produced a large number of graphic works and graphic types that are mathematically statistically significant and computer-enabled.</p>
<p><strong>The development of modern statistical graphics is more focused on the development of computer tools and the presentation of high-dimensional and dynamic graphics.</strong> One of the classics of high-dimensional graphics is a parallel map of coordinates that breaks the normal Cartesian coordinate system; as for dynamic graphics software, we'll stay at the end of the book to explain it.</p>
<p>Traditional statistical analysis can be divided into about three categories:</p>
<ul>
<li>Descriptive statistical analysis:</li>
<li>Hypothetical statistical analysis:</li>
<li>Explored statistical analysis: Exploratory Statistical Analysis</li>
</ul>
<p>The former are based on a number of statistical models, which are based on statistical methods.<strong>More graphic-based analysis of exploratory data</strong>In order to use this tool, it is necessary to have a clear understanding of how to map and what types of graphics are available in order to truly develop the value of statistical graphics.</p>
<p>The concept of “graphic statistical analysis” is not a new one, and the usual statistical images have been used to a greater or lesser extent, except that we tend to prefer statistical model analysis in a mathematical sense rather than using graphic statistical analysis as the main analytical tool. Of course, the limitations of graphic expression and the availability of statistical data make it impossible to replace model analysis.</p>
<p><strong>In particular, there's subjectivity in data interpretation.</strong></p>
<h2>Before the Tuco</h2>
<h3>Tools</h3>
<p>We make the following three requests for statistical tools:</p>
<ul>
<li>Full statistical computing capability</li>
<li>Statistical elements are easy to control.</li>
<li>Graphic types are diverse</li>
</ul>
<p>The primary function of graphics is indeed to visualize information, but the information here is not necessarily simple.<strong>A good statistical picture may hide important statistics, which are the most critical component of statistical graphics.</strong></p>
<p>Excel is the most common statistical graphic tool, but Excel appears to have only three graphics overall.</p>
<ul>
<li>The first is the expression of absolute value sizes, such as bar charts, column charts, circuit charts, etc.</li>
<li>The second is performance, like pie.</li>
<li>The third is a variable relationship on a 2-D plane, such as an X-Y scattered map;</li>
<li>In a broader sense, Excel presents almost all raw data, and statistical extrapolations based on data mean less.</li>
</ul>
<p><em>Excel now provides the capability of a data lens table, which is essentially an interactive graphical interface that brings together three basic graphics and provides a fixed format for displaying them or for displaying raw data.</em>
<em>On how Excel's graphics fit colours and become stereotypic, they can't skip the three types of limitations above, and they can't fully express the key to statistics.</em></p>
<p><strong>Statistical graphics are intuitive, but they're not simple, they're not speculative, they're not a stack of raw data.</strong></p>
<p>The core of statistics isn't looking at how the data are going to show it, but they're trying to extrapolate it from data, and in classical statistics, distribution is the most important thing that we need to study; concepts like averages, differentials, correlations, and probabilities can be grouped into distribution.</p>
<p>From here, we've found what really matters. <strong>Statistics are at the heart of statistical drawings, and the more complex they are, the more complex they are, or the aggregation of them, they mean statistical graphics that contain more data, and how they are designed is at the heart of how they are designed to remain visual.</strong></p>
<p>The menu-based GUI interface is not a “perfect” statistical mapping method, and the GUI is unlikely to increase indefinitely, but statistical graphics will; closed-source software operated by a single company will not be an excellent statistical mapping method, because they cannot keep up with the fast-growing academic community, but will have to hang on;</p>
<p><strong>An open-source, pure-coded platform is the best way to map statistics, so R and Python will be the end of modern statistical mapping until a new, epoch-making pattern is discovered and abandoned the old system.</strong></p>
<p>We say that graphics do not mean simple, that statistical graphics can be constructed in a very nuanced way (including the selection of statistical quantities and the design of graphic elements), rather than that the mapping process or procedures are complex.
We need to find a balance between efficiency and beauty.</p>
<p>At the end of the tool section, we redraw it with R code.<a href="/en/blog/2024/03/15/r-visualization-learning-notes/">Napoleon's Russian expedition.</a> Figure. Understanding his high self-defined level.</p>
<pre><code class="language-r">troops &lt;- read.table(system.file(&quot;extdata&quot;, &quot;troops.txt&quot;, package = &quot;MSG&quot;), header = TRUE)
cities &lt;- read.table(system.file(&quot;extdata&quot;, &quot;cities.txt&quot;, package = &quot;MSG&quot;), header = TRUE)
library(ggplot2)
p &lt;- ggplot(cities, aes(x = long, y = lat)) # 框架
p &lt;- p + geom_path(aes(size = survivors, colour = direction, group = group),
                   data = troops, lineend = &quot;round&quot;) # 军队路线
p &lt;- p + geom_point() # 城市点
p &lt;- p + geom_text(aes(label = city), hjust = 0, vjust = 1, size = 2.5) # 城市名称
p &lt;- p + scale_colour_manual(values = c(&quot;grey50&quot;, &quot;red&quot;)) +
  scale_size(range = c(1, 10)) +
  theme(legend.position = &quot;none&quot;) +
  xlim(24, 39) # 细节调整工作
print(p) # 打印全图
</code></pre>
<p><img src="/assets/images/r-learning-notes/r-language-stat-visualization-05.png" alt="Language R Statistical visualization 5"></p>
<p>Here are some examples of the creation of statistical graphics and possible problems in their application.</p>
<h3>Price trends</h3>
<p>We're using a manual data set to see what we're going to do with him.</p>
<pre><code class="language-r">year &lt;- c(2006, 2007, 2008, 2009, 2010 + c(1, 4, 7, 10, 13) / 12)
price &lt;- c(12.11, 18.8, 22.09, 18.39, 19.86, 14.89, 16.68, 18.76, 19.57)
</code></pre>
<p>And we can find that the four-sided coordinates of this data set correspond to a year, and the five-sided coordinates turn into three months, and see what happens when you make a straight line.
<img src="/assets/images/r-learning-notes/r-language-stat-visualization-07.png" alt="Language R Statistical visualization 7">
R automatically helps us to compress the back line; if you allow the cross-coordinate equidistance to be mapped, the original growth will be smooth. <strong>We've made some of the tricks in statistical graphics that can change the reader's first feeling.</strong></p>
<p>Besides, in drawing this time series, we might...<strong>I'm thinking about a line or histogram, and I'm thinking about zero starting with zero or the smallest.</strong></p>
<ul>
<li>Bar diagrams easily observe the size of the difference (the length of the comparative bar)</li>
<li>And it's easier to see the slope size.</li>
<li>If we pick zero on the real zero, it's easier to calculate the real ratio.</li>
<li>If the minimum value is selected, then it is easier to see absolute variation (because the difference is “magnified”)</li>
</ul>
<p>Give the corresponding graphics
<img src="/assets/images/r-learning-notes/r-language-stat-visualization-08.png" alt="Statistical visualization-8"></p>
<h3>Actors' pay.</h3>
<p>We gather information about the names of the actors, the names of the TV plays, the categories (rent Drama or comedy), the average income, the sex of the actors, the IMDB rating, etc., to consider the link between pay and gender, pay and work rating (including the type of work).</p>
<p><strong>The direct comparative distribution of gender-specific pay is very intuitive.</strong> He could be seen as a means of similar return to the binary.</p>
<p><strong>LOWESS Local Weighted Return Fragment Smoothing; he extracts local data for multi-formulation, then repeats the process until complete data is prepared, and can be considered as a non-parametric method that has a good advantage over return</strong>
Figure
<img src="/assets/images/r-learning-notes/r-language-stat-visualization-09.png" alt="Statistical Visualization - 9">
Histogram visualizes the gender-reversive pay differential: it can be seen that the average remuneration of actresses is slightly higher than that of male actors.</p>
<p>The IMDB rating seems to have nothing to do with the actor's pay, and the curve is almost a horizontal line; for comedy, it's not a straight line, and it's the highest in the vicinity of 7.9.</p>
<h3>Sound of Music</h3>
<p>The data contained 36 tracks, including classical music such as Mozart and Vivaldi, and rock music such as Abbas and Eels. In addition to the type of track (classical or rocking), we use only three continuous variables: average, maximum and variance of left audio frequency.</p>
<p>For such data, our concern may be whether classical music and rock music differ from audio variables; for this three-dimensional data, our most intuitive idea is a three-dimensional spread. Figure
<img src="/assets/images/r-learning-notes/r-language-stat-visualization-10.png" alt="Statistical visualization - 10">
The parallel coordinates and tunes and curves that we'll be introducing in the future will also be useful for this high-level data.
<img src="/assets/images/r-learning-notes/r-language-stat-visualization-11.png" alt="Statistical Visualization - 11">
<img src="/assets/images/r-learning-notes/r-language-stat-visualization-12.png" alt="Statistical visualization - 12"></p>
<p>All three graphics show differences between different types of music.
<strong>It's not always a good idea to have GVs.</strong></p>
<h3>Word Frequency Cluster</h3>
<p>Each author has his or her own unique style, such as the length of the sentence paragraph, the customary use of words, etc. In literature, the use of statistical methods to study the author's writing style and to classify the author's work, and to judge cases that have long been very successful</p>
<p>We're thinking about dividing the words in the work of different authors by the usual verbs, finding some of the more frequent words, and then grouping the authors according to the verb vector, and getting the following figure.
<img src="/assets/images/r-learning-notes/r-language-stat-visualization-13.png" alt="Statistical Visualization-13">
Our concentration is consistent with literature analysis.</p>
<p>And of course, we can look at the links between words, and see who's more likely to come together.</p>
<p>As traditional unstructured data, we tend to deal with a lot of things differently.</p>
<h2>Library</h2>
<p><a href="/en/blog/2024/03/15/r-visualization-learning-notes/">R Visualization</a></p>
<h2>Mapping experience</h2>
<p>We've been looking at statistical graphics before, and technically, they don't have much technology; but a good picture isn't just what the technology and code can do, it's hard to get the right graphics for the right data.</p>
<p>Let's start from a data point of view and start with a summary of what we're talking about; then we'll start with some drawing principles, and then we'll start with a summary.</p>
<h3>Graphic Selection</h3>
<p>The simplest way to classify statistics is to divide them into qualitative data and quantitative data; in quantitative data, the focus of our research tends to be related to distribution; the defined data tends to start with frequency; the following table is a brief summary of the graphics described above.</p>
<table>
<thead>
<tr>
<th></th>
<th>One-dimensional.</th>
<th>2D</th>
<th>Garvey.</th>
<th>Matrix</th>
</tr>
</thead>
<tbody><tr>
<td><strong>Disaggregated data</strong></td>
<td>Bar Chart</td>
<td>Marcello.</td>
<td>Marcello.</td>
<td></td>
</tr>
<tr>
<td></td>
<td></td>
<td>Linkage, Four Kingdoms</td>
<td></td>
<td></td>
</tr>
<tr>
<td><strong>Continuous data</strong></td>
<td>Histogram</td>
<td>Scatter Chart</td>
<td>Parallel Coordinate Chart</td>
<td>Colour Chart</td>
</tr>
<tr>
<td></td>
<td>Line Chart</td>
<td></td>
<td>Scatter Chart Matrix, 3D Scatter</td>
<td>Hot Chart</td>
</tr>
<tr>
<td></td>
<td>Cleveland Point</td>
<td></td>
<td>Three-dimensional view, smooth breakpoint Figure</td>
<td>Waiting for High Chart</td>
</tr>
<tr>
<td></td>
<td>One-dimensional spreads</td>
<td></td>
<td>Star, symbol, face map</td>
<td></td>
</tr>
<tr>
<td><strong>Mixed data</strong></td>
<td></td>
<td>Conditional Density Chart</td>
<td>Conditional Partition Chart</td>
<td></td>
</tr>
</tbody></table>
<h4>Disaggregated data</h4>
<p>With regard to disaggregated data, we tend to be concerned about the frequency or proportion of each classification, which is often simple, and reading graphics is simply an eye-ordering exercise.</p>
<p>There are few options for one-dimensional disaggregated data, and the most common is a bar chart; in a multi-dimensional data situation, Marseektu is able to clearly express the frequency size of the cells in the column tables, and also to observe the marginal probabilities and probabilities of conditions in the tables.</p>
<p>In addition to graphics of a descriptive nature, we can also use an extrapolated correlation map, which gives us a clear picture of which cells contribute significantly to this “non-independent” conclusion if the row variable in the list is not independent, and a quadrilateral map, which allows us to read quickly whether the row variable in the list is independent, i.e. whether the fanring is overlapping.</p>
<h4>Continuous data</h4>
<p>This compares with a much broader expression of continuous data:</p>
<p>In a one-dimensional situation, we can display the probability distribution of the data in histograms and density curves, a summary of the data in a four-digit graphic (this is a rough distribution expression), an expression of the size of the original values in a Cleveland dot, or an expression of the original values and their rough distribution in a one-dimensional scattered dot;</p>
<p>The most commonly used in two-dimensional situations is a scatterchart, which is usually used to express linear or non-linear relationships between two continuous variables, and is often used in conjunction with other graphic elements, such as reconnecting lines; in three-dimensional situations, we can draw a three-dimensional breakchart, and for special three-dimensional data, we can draw a three-dimensional map.</p>
<h4>High profile</h4>
<p>It's also important to reduce high-dimensional graphics to low-dimensional representation.</p>
<p><strong>Find carrier</strong>
Looking for other dimensions of "carriers" on a 2D plane, these "carriers" have many possibilities, but they are attached to high-dimensional data with some properties of graphic elements, such as the symbol in the symbol chart, the large width of the symbol, the facial features in the face map.</p>
<p><strong>Change Coordinate System</strong>
The Cartesian coordinates are theoretically limited to two-dimensional variables, so the use of other coordinates is also a natural choice that extends to high dimensions, such as the asterisk map using asterisk coordinate systems, with one coordinate for each branch outside the centre; parallel coordinates are also a common tool for the expression of high-dimensional data, which converts vertical coordinates to parallels, while the plane is theoretically capable of containing an infinite number of parallel lines, so parallel coordinates can theoretically also place any number of variables.</p>
<p><strong>Repeat 2D Graphics</strong>
The two-dimensional repetition can also serve the purpose of expressing high-dimensional data, for example, the scatterchart matrix is a double-and-trip scatterchart of all variables, so that all variable combinations can be represented on the plane.</p>
<p><strong>Subtract</strong>
Reducing the high-dimensional data to two-dimensional data is one way, for example, to analyse the main components of the data, and then to draw only scattered maps of the first two components; in fact, the cruise mode of the Gobi system is also a downside.</p>
<h4>Mixed data</h4>
<p>There are not many graphics specific to mixed data, and the condition density maps described above are a rare example. In the vast majority of cases, we recommend the use of “division of conditions”.<strong>Draws two-dimensional graphics using the respective values of the classification variables</strong>, so that we can easily compare the differences between the two-dimensional variables under the different categories of values.</p>
<p>To a certain extent, this approach of summarizing graphics for data types is somewhat rigid, as, for example, two classification variables are generally not able to draw a scattering point because the classification variable takes only a limited number of values, so that the break-up between the two classification variables is usually only a few grid points, and these do not in themselves reflect the true frequency of the position;</p>
<p>It's not like you can't draw a scattering map.<strong>There's too much concentration of classification variables.<code>jitter()</code> Function to disperse raw data is a viable solution</strong> For the latter, we need more attention to the convergence.</p>
<p><strong>In the end, drawings can't be done in style, they can be found freely in the graphics we've learned.</strong></p>
<h3>Drawing principles</h3>
<h4>Data Top</h4>
<p>Data are valuable, and they may come from difficult questionnaires or cumbersome experimental measurements, so we should try to appreciate them as much as possible, but the reality is that we often waste data, intentionally or unintentionally, in such cases:</p>
<ul>
<li>Elements expressing data are overshadowed by secondary graphic elements</li>
<li>Data features cannot be highlighted in the figure</li>
<li>The data was processed manually and inappropriately.</li>
</ul>
<h5>Split Main</h5>
<p>Obviously not all graphic elements are equally important for a graphic. For example, the point in the scatterchart should be the most important element, and the line in the contours is more important, etc. Therefore, we cannot allow secondary graphic elements to interfere with the expression of data and to make it clear and clear.</p>
<p>Which means... <strong>It's about getting the data out.</strong></p>
<p>The default style of the point in the R base graphics system is the hollow spot, which in many cases is not a good option, because the hollow spot looks too low on the map, especially when the data point is small.
You can control parameters. <code>pch</code> To adjust the shape of our points, there are a number of options.</p>
<p>If we want to know more about who each of these points represents, then we need to place text labels on the picture, which are very revealing tools, which can tell us straightaway, but because of its relatively large size, if not properly handled, it can fill the picture with text information, thus losing the value of the data itself.</p>
<p><strong>maptools</strong> Package <code>pointLabel()</code> The function provides a tag-up scheme based on analogue repulsive algorithms and genetic algorithms that do everything possible to avoid overlapping labels; of course, it would be better if the drawing window were expanded.</p>
<p><strong>The multi-dimensional scale analysis of labels (Multidisional Scaling, MDS) was mentioned as a statistical approach that is well suited to the presentation of labels.</strong> Theory Reference Machine Learning Progress and Unsupervised Learning: Multi-dimensional Zooming (MDS)</p>
<p>Because we're concerned about the distance between the individual and the individual, it's better to draw some of the individual's features in the picture, and the most direct idea is, of course, the individual's name, and the focus of the picture is on these labels.</p>
<p>We can use MDS to observe some of the cluster characteristics, which can be effective in reducing the dimensions, as shown below.
<img src="/assets/images/r-learning-notes/r-stat-visualization-08.png" alt="R Statistical visualization-8"></p>
<p>There are also some details about the main relationship of the graphic elements, which is also ours.<a href="/en/blog/2024/09/16/r-graph-learning-notes/">& Base-based drawing system R</a> One of the reasons for all the graphic details.</p>
<p>For example, the direction of the tic shorts of the axis is defaulted to the outward (<code>tcl</code> (a) Arguments) which are reasonable and may interfere with elements in the graph area if the tic line is stretched internally;</p>
<p>Like <code>xaxs</code> and <code>yaxs</code> Parameters, they default to allow the range of the map area to be expanded by 4 per cent, so that a small space is left between the axis and the boundary of the chart data, so that the minimum and maximum values in the data are not closely connected to the axis, and so that the main graphic elements are fully visible and free from the axis lines of the coordinates.</p>
<h5>The symbol is clearly divided.</h5>
<p>We'll use a lot of symbols in a picture, and in order to make the graphic clear and distinguish, we should try to select the more differentiated symbols.</p>
<h5>Carefully processing data</h5>
<p>Usually, data can only be processed to reveal the information we want to know, for example, that giving all of us a height data profile in the country will only drown us in numbers, but an average or median can tell us the average of height. This may be the reason why people develop habits for processing data, but it is often a disaster for statistical graphics, and valuable raw information is destroyed by human processes.</p>
<p>In graphics, we advocate that raw data be expressed as far as possible, rather than being processed by humans, including</p>
<ul>
<li>Do not omit data</li>
<li>Do not separate the data</li>
</ul>
<p>One of the advantages of graphics compared to tables is that it displays much information in smaller spaces, and we hardly have to deliberately delete the raw data and draw them, and even if they are necessary, then it is usually decided to look at local data after looking at the complete picture.</p>
<p>Dispersional data are the more common data-processing tool and its shortcomings are less easily detected, often because of the inertia of previous behaviour and the temptation of those disaggregated data-processing methods;
In fact, discrete data loses the original information and the arbitrary division of the area often leads to errors due to the inappropriate division of the area.</p>
<p>Nor can we absoluteize the principle of "no data processing" and in one case processing data would make the graphics more explicit, i.e. data that are not readily visible from the original data; for example, differences in the two sets of decorative lines, or differences in the two data sets, or observation of growth rates, where the deviation would allow us to better observe differences</p>
<p>There is also a situation where there may be a need to process data slightly, and when there are many overlaps in the data, we have to find ways to allow readers to read these overlaps, and randomly to disrupt the location of the data points is a way to do it, but don't forget to remind readers of our operation, otherwise they will not realize that we've been messing with it.</p>
<h4>Ink saving</h4>
<p>Turning to ink, we have to mention the visualized master Edward Tufte, who invented an interesting word, "chartjunk," which is a surplus in a graphic that does not help in the expression of data, or even conceals or distorts information in data.</p>
<p>We can give a classic chartjunk, and he shows only five figures, five years, the percentage of admission to U.S. universities at 25 years old.
<img src="/assets/images/r-learning-notes/r-stat-visualization-09.png" alt="R Statistical Visualization-9"></p>
<blockquote>
<p>One graphic can be decorated in one of three ways, one in a color that is frowning, two in a 3D effect, and three in a disguise that is as rich as it is, using all three means -- Michael Friendly.</p>
</blockquote>
<h4>Design Layout</h4>
<p>Here we would like to introduce the vertical comparison, which is a concept that is very important for graphic interpretation, especially the contour. In short, the vertical-to-penetrating effect is the ratio of the graphic elements to the width. The figure for "skinny" is very steep, and the sense is that the trend is very sharp, while the figure for "skinny" is that the trend seems to be smoother.</p>
<p>R. Common in basic graphics systems <code>asp</code> Parameter-argument ratio
<img src="/assets/images/r-learning-notes/r-stat-visualization-10.png" alt="R Statistical visualization - 10">
I can see from the following figure. <strong>The number of sun darks rises faster than the rate of decline.</strong> This is the sense of deception of readers through vertical matching.</p>
<p>Cleveland's suggestion on this issue is to adjust the vertical-and-arrange angle average of all the cones to close to 45∘** Because people's eyes are the most accurate of the angles around 45 gills, and they're all too big or too small.</p>
<h4>With explanation</h4>
<p>Although we say "a picture is a success," the graphic itself may be interpreted differently for different readers, even if some readers do not necessarily understand what a picture really means, and it is necessary to provide accompanying explanations at this point.</p>
<p>There are two ways to explain by-product, one by adding a reference to the text, which is very limited because the space in the picture is limited and too many text indications may lose the focus of the graphic itself</p>
<p>The other way is to use the title of the chart, which seems to be relatively good in the English literature, which usually has a clear title and which is like a complete phrase, but in most Chinese literature the picture is as plating as a text, the title of the chart is only one sentence, and the explanation of the chart is usually in the body of the text, which may prevent readers from focusing on the reading of the graphic, as it needs to be read back to the body of the text with the corresponding explanatory text.</p>
<p>Our suggestion is that we can get the reader to understand the implications of the graphic as much as possible by combining the content of the graphics with the simple understanding of the article in the brain; that requires that our title be a short but complete story.</p>
<h4>Think about it.</h4>
<p><img src="/assets/images/r-learning-notes/r-stat-visualization-11.png" alt="R Statistical Visualization-11">
It's a classic visual deception. Red and black are essentially the same length.</p>
<p>And similarly, we have some of the graphic psychology that has been studied.</p>
<ul>
<li>Red is exaggerating, so the red area may look bigger than it really is.</li>
<li>The color of filling in the larger area seems to be deeper than the smaller area.</li>
<li>The same angle may be placed in different directions, which may make it look different, for example, from a horizontal angle and from the same angle of 45 ∘ angles, which may affect the interpretation of the pie.</li>
</ul>
<h3>Final summary</h3>
<p>We've been presenting a lot of statistical data, but they're not completely, or just the tip of the iceberg, for the following reasons.</p>
<ul>
<li>So far, R packages have exceeded 2w, and many authors design the function of drawing, and it's impossible to introduce it completely.</li>
<li>The graphics in many statistical packages are not very different in their use, for example, many packages have chosen extended and extended functions.<code>plot</code> To achieve your own effects</li>
<li>Many of the parameters in the graphics are actually the same or close.</li>
</ul>
<p>Of course, R's graphic system is also very deficient.</p>
<ul>
<li>Graphics are unedited. You want to change them, you have to redraw the whole picture.</li>
<li>We still need to work slowly to get the desired effect.</li>
<li>Lack of interactive graphics</li>
<li>There's something unreasonable about the details. <code>plot</code> Default hollow circle. Data is hard to highlight.</li>
</ul>
<p><strong>Here's the graphics, and then we'll show some of the drawing systems in R.</strong></p>
<h2>R Basic Mapping System</h2>
<p>The statistical graphics of the R base mapping system are generated by a function, and his core is in the R base package. <strong>graphics</strong> A large number of developers have expanded a series of derivative packages based on this, and they all form the grammar rules of the basic system.</p>
<p>Sometimes we have our own personal requirements for statistical graphics, and this is when we fine-tune the details, and here we are talking about the details.</p>
<h3>Two Graphical Functions</h3>
<h4>plot function</h4>
<p>The most common graph function in R is <code>plot()</code> function, it is a generic function that allows many different categories of objects to be accepted as their graphic object parameters; we are here to explain only the graphic parameters, not the graphic object parameters.</p>
<p>First introduce <code>plot()</code> Common parameters:</p>
<p><code>type</code>
Graphic style type, with nine possible values, representing different styles:</p>
<ul>
<li><code>&#39;p&#39;</code>- Drawing points;<code>&#39;l&#39;</code>⇒ Paint lines;</li>
<li><code>&#39;b&#39;</code>(b) The drawing of points and lines at the same time, but not the intersection of the lines;</li>
<li><code>&#39;c&#39;</code>⇒ General <code>type = &#39;b&#39;</code> The midpoint is removed and only the corresponding line is left;</li>
<li><code>&#39;o&#39;</code>♪ And draw points and lines, and overlap, and it's with <code>type = &#39;b&#39;</code> (b) Distinctions;</li>
<li><code>&#39;h&#39;</code>• Draw lead lines;</li>
<li><code>&#39;s&#39;</code>Draws a ladder, from one point to the next, horizontal lines and vertical lines;</li>
<li><code>&#39;S&#39;</code>The tectonic line is also a line of drawing, but the vertical line is drawn from one point to the next, and horizontal lines are drawn;</li>
<li><code>&#39;n&#39;</code>⇒ Make an empty map with no content, but all other elements such as the axis, the title, etc. are shown as such (unless hidden in a different setting)</li>
</ul>
<p><code>main</code> <code>sub</code> <code>xlab</code> <code>ylab</code>
Main title Subtitle x-axis label y-axis label</p>
<p><code>asp</code>
Graphical vertical ratio, i.e. ratio of 1 unit length on y axis and 1 unit length on x axis; normally, this ratio is not 1, and in some cases it is necessary to set up to show better graphic effects, e.g. the slope that needs to be expressed in a straight line from angle: If <code>asp</code> Not equal to one, so 45-mile angles may not look like the real 45-mile angle.</p>
<p><code>x, y</code>
Two vectors to make a scattering chart; if y is missing, x is the position of its elements (in %2)<code>1:n</code> ) to create a scattered map</p>
<p><code>xlim, ylim</code>
Sets the limit of the coordinates system, both parameters take a vector of 2 in length and they work similarly <code>par()</code> Medium <code>usr</code> But we can do it. <code>par()&#36;usr</code> The coordinates of the given map are obtained, and this function is not available for both parameters, as normally the charting function does not return any value (or the return value is empty):<code>NULL</code>）</p>
<p><code>log</code>
Whether coordinates are logarithmic, values are taken <code>&#39;x&#39;</code> The text of the article is reproduced below:<code>&#39;y&#39;</code> The #symmetrical #symmetrical #symmetrical #symmetrical #symmetrical #symmetrical #symmetrical #sympic #symphosmpic #sympic #symphosmphos #symphosympic #sympic #sympic #symphos-ymphos #sympic #sympic #symphos #symphos #sympic #symphos #symphasym<code>&#39;xy&#39;</code> Both coordinates are in logarithmic.</p>
<p><code>ann</code>
Some default marks are shown, such as coordinates of axes and graph titles</p>
<p><code>axes</code>
Whether or not to draw the axis; care will affect only the drawing of the axis and the scale, and not the title of the axis</p>
<p><code>frame.plot</code>
whether to box the graphics;available <code>box()</code> function, functions similar but more detailed</p>
<p><code>panel.first</code>
Work to be done before drawing; this parameter is often used to add a background grid or a smooth curve of a scatter point before drawing, for example <code>panel.first = grid()</code></p>
<p><code>panel.last</code>
Tasks to be completed after drawing; similar to previous parameter</p>
<p>Besides the initial:
<code>col, pch, cex, lty, lwd</code>
Meaning of these parameters <code>par()</code> And the parameters in it are basically the same, and the difference is,<code>par()</code> , and this is the only single value that can be set here, and this vector is applied in turn to each element, and if the vector length is shorter than the number of elements, the vector will be recycled until all elements are drawn, and in fact the recycling of the vector is a major feature of the R graphic parameter.</p>
<p><code>bg</code>
background colour; attention and <code>par()</code> The difference is, it's set only to draw the background color, not the whole picture!</p>
<h4>Par Functions</h4>
<p>The graphic parameter for R can be used both by function <code>par()</code> Pre-global settings, which can also be used for specific graphic functions (e. g. <code>plot()</code>、<code>lines()</code>) sets the temporary parameter values;</p>
<p>The difference between the two is that the former setup will always work in the current graphic device, unless the graphic device is shut down, while the latter setup is only temporary and does not affect the graphic effects of other graphic functions that follow.</p>
<p>Functions <code>par()</code> It covers most of the graphic parameters, and therefore is described in a dedicated section.</p>
<p>Functions <code>par()</code> You can set or get graphic parameters.<code>par()</code> Returns the current graphic parameter settings (a list) to the extent that they are available for setting graphic parameters. <code>par(tag = value)</code> Form</p>
<p>Current <code>par()</code> The function involves approximately 70 graphic parameters, which are used to explain the common and more understandable parameters.</p>
<p><code>adj</code>
Adjusts the relative position of characters in the diagram; values are given in a numerical vector of 1 length, usually in [0,1], 0 for left alignment, 1 for right alignment; in <code>text()</code> function, the length of which can be 2 and the adjustment of the lower left angle relative to the point (x, y) of the character boundary rectangular, respectively, is also generally in the range [0,1] of the vector, which can also be exceeded in some graphic devices, and the ratio of the string to the left, which is based on the lower left, moving to the left and down, depending on its width and height, is defaulted. <code>c(0.5, 0.5)</code>I'm sorry. For example, <code>c(0, 0)</code> The lower left corner of the whole character (string) is the point of the given coordinates, and <code>c(1, 0)</code> is the distance of the string that moves its width horizontally, without vertically affecting it.</p>
<p><code>ask</code>
Switch to the next new graphic device (usually a new one) if the user needs to enter (knock back the car key or click the mouse); TRUE indicates; FALSE indicates no. It is useful when multiple maps are presented on each of them and need to be displayed on graphic devices in sequence, if set <code>ask</code> Yes <code>TRUE</code>, then every new picture will be made before the user enters it, or all the images will be flashed.</p>
<p><code>bg</code>
Sets the graphic background colour;</p>
<p><code>bty</code>
setting graphic border styles; taking values to characters <code>o, l, 7, c, u, ]</code> one;the shapes of these characters themselves correspond to border styles, such as (default values)<code>o</code> It means that all four sides are shown, and <code>c</code> This means that the right side is not shown</p>
<p><code>cex</code>
The zoom factor (text and symbols, etc.) in the figure above is multiplied; the value is a value relative to 1 (default is 1). The specific detail can be scaled by the following parameter settings (the default values are 1:</p>
<p><code>cex.axis</code>
Multiplier of coordinates of coordinates of scale marks</p>
<p><code>cex.lab</code>
Multiplier of coordinates for axis titles</p>
<p><code>cex.main</code>
Multiplier scaling of main title of the figure</p>
<p><code>cex.sub</code>
Multiplication of the figure by subtitle</p>
<p><code>col</code>
(a) the colour of the symbols (points, lines, etc.) in the figure; and <code>cex</code> Parameters similar</p>
<p><code>col.axis</code>
Colour of coordinates tic marks</p>
<p><code>col.lab</code>
Colour of coordinates for axis title</p>
<p><code>col.main</code>
Colour of main title of the figure</p>
<p><code>col.sub</code>
Colour of the byline title of the figure</p>
<p><code>family</code>
Sets the font family of text (breadline, liner, equal width, symbol font, etc.); standard values are:<code>serif, sans, mono, symbol</code>，</p>
<p><code>fg</code>
Sets the foreground colour (if no other color settings are specified later, this parameter affects almost all the subsequent graphic elements colours, and if the subsequent graphic elements have specified colour settings, only the colours of the graphic border and the coordinates of the axis lines);</p>
<p><code>font</code>
Sets text font styles; takes value to an integer; normally 1, 2, 3 and 4 means normal, bold, italics and bold italics respectively; for additions,<code>text()</code> Functions and <code>vfont</code> Parameters can set more detailed font family and font styles; see these two presentations:<code>demo(Hershey)</code> and <code>demo(Japanese)</code>The former demonstrates Hershey vector fonts, the latter expresses Japanese;</p>
<p><code>font.axis</code>
Font style for the coordinate axis tic label</p>
<p><code>font.lab</code>
Font Style for Coordinate Axes Titles</p>
<p><code>font.main</code>
Font Style for the main title of the figure</p>
<p><code>font.sub</code>
Font Styles for the Figure Subheading</p>
<p><code>lab</code>
Sets the number of coordinates of the axes (R will automatically "take" as much as possible, i.e. as close as possible to the arc of 0.5, 1 or 10); the form of the value to be taken <code>c(x, y, len)</code>：<code>x</code> and <code>y</code> Sets the number of tics for each of the two axes.<code>len</code> Currently not in effect in R, setting any value is not affected (but is used) <code>lab</code> This parameter must be written when you are using it)</p>
<p><code>las</code></p>
<p>Coordinate axis label style; take one of the four integers, 0, 1, 2, 3 and 1, respectively, indicating " always parallel to the axis " , " always horizontal " , " always vertically " and " always vertically " .</p>
<p><code>lend</code></p>
<p>style (round or square) at the end of the line; values are either 0, 1 or 2 integers (or the corresponding string) <code>&#39;round&#39;, &#39;mitre&#39;, &#39;bevel&#39;</code>) , watch the fine differences between the two</p>
<p><code>lheight</code>
Line height in figure Chinese; value taken multiple, default 1</p>
<p><code>ljoin</code>
style for the intersection of lines;the value is either 0, 1 or 2 integers (or the string of the string) <code>&#39;round&#39;, &#39;mitre&#39;, &#39;bevel&#39;</code>) means drawing round corners, drawing square corners and cutting the top angles</p>
<p><code>lty</code>
Lines are fake styles: 0 ⇒ without drawing, 1 ⇒ solid, 2 ⇒ dotted, 3 ⇒ dotted, 4 ⇒ dotted, 5 ⇒ long underlined, 6 ⇒ long underlined; or the following string is set accordingly (for the preceding numbers):<code>&#39;blank&#39;, &#39;solid&#39;, &#39;dashed&#39;, &#39;dotted&#39;, &#39;dotdash&#39;, &#39;longdash&#39;, &#39;twodash&#39;</code>; also indicates the length of the line in the line and the blanks in a string consisting of a hexadecimal number, if <code>&#39;F624&#39;</code></p>
<p><code>lwd</code>
line width;default 1</p>
<p><code>mar</code>
Sets the width of the graphic boundary in the whitespace; by default, in the order of " lower, left, top, right " <code>c(5, 4, 4, 2) + 0.1</code></p>
<p><code>mex</code>
Sets the width of the axis to be multiplied by the width of the boundary; default is 1 and this parameter will affect <code>mgp</code> Parameters</p>
<p><code>mfrow, mfcol</code>
setting a page of multi-graphs; taking the form of values <code>c(nrow, ncol)</code> Vector with 2 length, set rows and columns, respectively</p>
<p><code>mgp</code>
Sets the width of the boundary of the axis; the numerical vector with a value of 3 is the width of the axis title, the coordinates tic line label and the coordinates axis, respectively (acceptance) <code>mex</code> ) , Default is <code>c(3, 1, 0)</code>, meaning that the coordinates are 3 and 1 and 0, respectively, from the coordinates ' axes, coordinates ' tic line labels and coordinates ' axes;</p>
<p><code>oma</code>
Sets the width of the outer boundary (Outer Margin); similar <code>mar</code>, Default is <code>c(0, 0, 0, 0)</code>, when only one chart is displayed on a page, the parameter is compared to <code>mar</code> It's not a good distinction, but it's easy to see the difference between <code>mar</code> The difference.</p>
<p><code>pch</code>
(a) The symbol of the point;<code>pch = 19</code>* The point of the square,<code>pch = 20</code>⇒ Small solid dot, pch = 21 ⇒ circle,<code>pch = 22</code>Zirconium square,<code>pch = 23</code>♪ ♪ The oil ♪<code>pch = 24</code>♪ ♪ Right on the triangle, ♪<code>pch = 25</code>⇒ Tile-tip, where 21-25 can fill colours (using <code>bg</code> (parameters)</p>
<p><code>pty</code>
setting shapes for the chart area;defaultly as <code>&#39;m&#39;</code>: Maximize the map area; another value <code>&#39;s&#39;</code> Means that the set-up-charted area is square</p>
<p><code>srt</code>
rotation angle of string; taking an angle value</p>
<p><code>tck</code>
Height of the axis tic line; take value is the ratio to the width of the graphic (between 0 and 1); positive value is the internal graph line, negative value is the outward; default is not using it (set to <code>NA</code>) and uses <code>tcl</code> Parameters</p>
<p><code>tcl</code>
coordinates the height of the axis tic line; taking a ratio to the height of the text line; positive or negative meaning <code>tck</code>, the default value is <code>-0.5</code>, i.e., an outward drawing line, with a height of half-line text;</p>
<p><code>usr</code>
Range limit for the chart area, with a value of 4 value vector <code>c(x1, x2, y1, y2)</code>, which means the right and right limits of the x axis and the lower upper limits of the y axis in the map area; note, if the coordinates are taken as logarithmic (see <code>xlog, ylog</code> The actual limit is set at 10 corresponding tacks. Number of times</p>
<p><code>xaxs, yaxs</code>
coordinates range calculations;default <code>&#39;r&#39;</code>: extend the range of the original data by 4% and then draw the axis of the coordinates with this range; another value `i' indicates the direct use of the original range; there are actually other methods of calculating the range of the coordinates, but they are not presented as they are not currently in force in R</p>
<p><code>xaxt, yaxt</code>
coordinates axis style; default <code>&#39;s&#39;</code> as standard styles;other value <code>&#39;n&#39;</code> It means no coordinates.</p>
<p><code>xlog, ylog</code>
whether coordinates are to be taken as logarithm; default <code>FALSE</code></p>
<p><code>xpd</code>
handling of graphics beyond boundaries; taking values <code>FALSE</code>: Limiting graphics to the graphics area, and removing the graphics from the boundary; take values <code>TRUE</code>: Limiting graphics to graphics, and removing the graphics from the boundary; taking values <code>NA</code>: Limit graphics to the device area Internal</p>
<p><strong>The role of these parameters need not be so well understood; this chapter can be used as a reference only and can be consulted as needed</strong></p>
<p>After we've finished the argument, let's just say <code>par()</code> The usual technique. As mentioned at the beginning of this section, this function changes the pattern setting, and we do not sometimes want this function, especially when we wish to be restored after a picture is finished and the next one is prepared.</p>
<p>And then we need to save the drawing parameters to an object before we start a map, for example. <code>op = par()</code>, and then we can use it in the making of this picture <code>par()</code> Function changes any settings that are appropriate to the needs, and we'll use them after this picture is finished <code>par(op)</code> statement sets the previously saved parameter " Release " out so that the changes to the graphic parameter by the intermediate process no longer affect the next chart.</p>
<p>Of course, every drawing that is done can also turn off the graphics device and then make the next one, which can also serve its purpose, but only to a lesser extent, especially when it is repeated, adjusted and compared, and then it becomes more cumbersome to turn off and open the graphics.</p>
<h3>Colour</h3>
<p>The color is the most important element in the graphic.</p>
<p>By default, the settings for the colour in the R will depend on <strong>grDevices</strong> Package support, which provides a large number of colour selection and generation functions, as well as several preset palettes to express different themes. We'll go down to the next level.</p>
<p><strong>graphics</strong>The package supports three drawing colour parameters: <code>col</code> <code>bg</code> <code>fg</code> The colour of the elements, the background colour, the foreground colour, are used separately; this rule is also absorbed in other drawing packages.</p>
<h4>Fixed Colour Selection Function</h4>
<p>Fixed Colour Selection function is the color that R provides for bringing a fixed type of color, mainly a function <code>colors()</code></p>
<p><code>colors(), colours()</code>The two functions are identical, they are two different spellings in English, they do not require any parameters and generate 657 colour names</p>
<p>We can use names to call these colors, and R, of course, assigns numbers to common colors, one to eight, so the color vector can be used as a color vector.<code>col</code> <code>bg</code> <code>fg</code>value of the parameter</p>
<h4>Colour Theme Palette</h4>
<p>The colour generation process described above may be too complex for the general population. At this point, R offers a third option, the colour palette for a specific color theme. These palettes present specific themes in a series of gradients, such as rainbow colour series, white thermal colour series, topographic colour series, etc.</p>
<p><code>rainbow()</code>By definition, it's the rainbow color that produces a series of colors.</p>
<p><code>heat.colors()</code>Gradual change from red to yellow to white (to reflect "high temperature", "white heat")</p>
<p><code>terrain.colors()</code>Gradient green to yellow to brown to white.</p>
<p><code>topo.colors()</code>Gradually from blue to cyan, to yellow and finally brown.</p>
<p><code>cm.colors()</code>From the cyan to the white to the pink.</p>
<p><strong>These palette functions help us generate some colour pools that we can call directly.</strong></p>
<h4>Type Palette</h4>
<p>Of course we can be more impatient. Package <strong>RColorBrewer</strong>Three types of palette are provided, and users can use the colour palette name in the package <code>brewer.pal()</code> function to generate colour. These three types of palettes include:</p>
<ul>
<li>Continuous palettes generate a series of successive gradients that usually mark the size of the continuum values</li>
<li>Extreme palettes Points</li>
<li>Dispersive palettes generate a series of more distinct colours that are usually used to mark classified data
<code>brewer.pal()</code> You return the colour vector.</li>
</ul>
<p><strong>RColorBrewer</strong> The bag is also available. <code>display.brewer.pal()</code> and <code>display.brewer.all()</code> function that shows the selected palette or all palettes in the graphic window.</p>
<p>The blog also provides a detailed explanation of the situation.<strong>RColorBrewer</strong>It's very easy to use.</p>
<h4>Colour Generation and Conversion Functions</h4>
<p>R A range of colour-generated models is available, such as the RGB model (Red Green Blue Three Colors Mixed), the HSV colour model (Colour, Saturation and Purity), the HCL colour model (Colour, Color and Brightness) and the Grey Generation model. The structure of the colour is complex and beyond the scope of the book, so here only the use of the function is described.</p>
<p><code>rgb()</code>The three-plattered blues and red. <code>rgb(red, green, blue, alpha, names = NULL, maxColorValue = 1)</code></p>
<p><code>hsv()</code>Construct colours with Hue, Satouration and Value <code>hsv(h = 1, s = 1, v = 1, alpha)</code>；</p>
<p><code>hcl()</code>Construct colours with Hue, Chroma and Luminance, as <code>hcl(h = 0, c = 35, l = 85, alpha, fixup = TRUE)</code>；</p>
<p><code>gray(), grey()</code>Generate grey series; only one parameter <code>level</code>, indicating the greyscale level, with values ranging from 0 to 1</p>
<p><code>rgb2hsv()</code>Convert RGB colour to HSV colour, usage <code>rgb2hsv(r, g = NULL, b = NULL, maxColorValue = 255)</code></p>
<p><code>col2rgb()</code>Convert any R colour value to RGB, usage <code>col2rgb(col, alpha = FALSE)</code></p>
<h3>Drawing Elements</h3>
<p>The statistical graphics are made of elements, and here we are presenting the bottom elements, the advanced graphic functions we use are essentially to generate these elements by certain patterns and wrap them up into a function for us to use, which is useful for people who want to get to the bottom of R.</p>
<h4>Points</h4>
<p>For point settings, we can use both many graphic functions <code>pch</code> And so on, it can be done with lower layers. <code>points()</code> Add a point to an existing graphic to achieve</p>
<p>We're here to highlight the importance of this.</p>
<ul>
<li><code>lwd</code> Parameters, which we know are the width of the line, can also set the edge "line" width of the point for point purposes;</li>
<li><code>pch</code> Parameters can also accept characters as parameter values, not just numbers;</li>
<li>Finally, parameters <code>pch</code> The point from 21-25 can fill the background colour</li>
</ul>
<h4>Line</h4>
<p>We can use a function. <code>lines()</code> to add a curve to the chart (the curves in this context are essentially links to some segments of the line, not smooth curves); the following is a brief addition to the line style: <code>lty</code> Settings</p>
<p>R can achieve almost a million lines because of its <code>lty</code> The parameters are flexible, and apart from the values 0-6, the lines can be set in the form of a hexadecimal digital string (digits must be even and not zero).</p>
<p>For the straight line, we need only determine the position of the plane coordinates by two factors: the slope and the cut-off. Functions <code>abline()</code> It's for adding a straight line.</p>
<p>Lines can be used for functions <code>segments()</code> Generate</p>
<p>The sample is a curve that links several data points with a smooth curve  <code>xspline</code> It's the method of generation.</p>
<p>Arrows can be used for functions <code>arrows()</code>Generate</p>
<h4>Rectangle, Polygon</h4>
<p>R is also easy to draw polygons, mainly for use <code>polygon()</code> Function, rectangle is a special case of polygon, but R also provides a specific function <code>rect()</code> Here, draw it.</p>
<p>There's a special rectangle, which is the frame of the whole picture, and it can be used. <code>box()</code> Function to complete</p>
<h4>Grid Lines</h4>
<p>Sometimes, in order to facilitate graphic readers to know the more precise location of the elements in the diagram, we can help the reader to align the axis with the view of the grid by adding a background grid. Functions <code>grid()</code> And that's what it's all about.</p>
<h4>Title Any text Periphery text</h4>
<p>All text in the graphic can be divided into three categories: title (main subheading and coordinate axis heading), any text and graphic surrounding text.<code>title()</code> function to add a title,<code>text()</code> function to add text to any position in the graphic.<code>mtext()</code> Function to add text to the four sides of the chart</p>
<h4>Legend</h4>
<p>Functions <code>legend()</code> The function is to add legends</p>
<h4>Coordinate axis</h4>
<p>Sometimes we need to make special arrangements for coordinates, for example, to make a two-coordinate axis, or to use special text in coordinates markings, so we have to use it. <code>axis()</code> Function to support the completion of the alignment and adjustment of the axis</p>
<h4>Gradient Declination Algorithm</h4>
<p>We should understand R's freedom to make drawings -- we can control almost all the details of the graphics -- and consider using them to visualize the gradients.</p>
<p><img src="/assets/images/r-learning-notes/r-stat-visualization-12.png" alt="R Statistical Visualization-12"></p>
<p>Specific function realization to refer to in MSG package <code>msg(&quot;9.22&quot;)</code></p>
<h3>One page of multi-graph</h3>
<p>Sometimes we need to place multiple graphics in the same page to compare them or to make them more beautifully organized. In this case, we have at least three options:</p>
<h4>Set graphic parameters</h4>
<p>We talked about it once. <code>mfrow</code> and <code>mfcol</code> Two parameters if we're in <code>par()</code> function, then the next graphics will be created by the number of rows and columns set by these parameters, which is the most common graphic layout method</p>
<p>The limitation of these two parameters is that they can only split the graphic area into grids, each of which must be equal in length and width, and each of which must have a graphic that does not allow for the function of a graphic in multiple cells.</p>
<h4>Set Graphical Layout</h4>
<p>R provided <code>layout()</code>Function as a tool to set the split of graphic layouts</p>
<ul>
<li><code>mat</code> Parameters are a matrix that provides the order of the drawings and the arrangements for the graphic layout</li>
<li><code>widths</code> and <code>heights</code> Proportion of rectangular regions long and wide</li>
<li><code>respect</code> Controls whether the scale of the length of the vertical axis inside the graphics is the same</li>
<li><code>n</code> Serial number of the area to display</li>
</ul>
<h3>Graphical Devices</h3>
<p>Utilization <strong>grDevices</strong> Several graphic devices in the package, which we can output the R graphics into files in various formats, including bitmap files (BMP, JPEG, PNG, TIFF) and vector chart files (PDF, EPS) and TeX or LaTeX files</p>
<p>Basic graphic device function has bitmap device <code>bmp()</code>、<code>jpeg()</code>、<code>png()</code> and <code>tiff()</code>and vector-mapping devices <code>svg()</code>、<code>postscript()</code> and <code>pdf()</code>, all R graphics are generated in the graphic device after opening the graphic device, and will not be shown in the window until the graphic device is shut down.</p>
<p><strong>tikzDevice</strong> The package is more friendly to the output of the graphics, and it's better to use it to control the output of our graphics.</p>
<p>Diagram Devices support the use of Chinese or other CJK characters in graphics, but font family parameters are required when using Chinese characters in vector chart devices <code>family</code>Otherwise, Chinese will not be shown (e.g., Chinese should be used) <code>pdf(family = &#39;GB1&#39;)</code> The problem arises in exporting EPS formats.</p>
<h3>Math Formula</h3>
<p>Since mathematical symbols are often required in statistical theory, adding some mathematical descriptions to statistical graphics not only makes the graphics look more professional, but also adds significantly to the theory behind the graphics.</p>
<p>R's. <strong>grDevices</strong> The package provides a series of mathematical formulae with symbols that he can help us insert into the graphics. <code>?plotmath</code> You can see help.</p>
<p>If you want to add a text label to the map for mathematical expression, you just have to set the text to the expression type.</p>
<p><strong>tikzDevice</strong> The package can also help us generate better-quality mathematical formulae, and this extension was released in 2020, which is more friendly to the Latex syntax.</p>
<h2>ggpllot2 graphics system</h2>
<p>The basic graphics system is flexible, but the options are varied and dispersed. ggpllot2 to <em>The Grammar of Graphics</em> , which is based on the theory that the graphics are to be broken down into components that can be assembled layer by layer. It expands the generic function. <code>+</code>, data, visual maps, geometric objects and other settings can be stacked by layers.</p>
<h3>Basic Syntax:</h3>
<p><code>ggplot()</code> The project is designed to provide data and default visual maps.<code>aes()</code> Describe how the variable is mapped to cross-axis, vertical axis, colour or shape.<code>geom_*()</code> Add specific geometric objects,<code>labs()</code> , and then set the title and label. A basic example is as follows:</p>
<pre><code class="language-r">library(ggplot2)

p &lt;- ggplot(data = mtcars, aes(x = wt, y = mpg)) +
  geom_point() +
  labs(
    title = &quot;Automobile Data&quot;,
    x = &quot;Weight&quot;,
    y = &quot;Miles Per Gallon&quot;
  )

# 在已有图形上增加平滑曲线
p + geom_smooth(method = &quot;loess&quot;)

# 输出没有平滑曲线的原始图形
print(p)
</code></pre>
<p>ggpllot2 automatically adjusts the margins, selects the colour from the palette and generates legends when needed. It is usually only necessary to clarify the data and mapping relationship, and the remaining details can be progressively supplemented in the various layers.</p>
<p>The system consists mainly of geometric objects (geom), statistical transformation (stat), scale (scale), coordinates (coordinate system), facet and theme (theme). There are many extension packages around this syntax that can add new graphic types, scales and themes.</p>
<h3>Geometric Object</h3>
<p>Geometric object abbreviations géom, including points, bars, lines, box charts and text. They are similar to basic graphic elements, but more encapsulated. For example, the box diagram, smooth curves and slides are all statistically calculated, and in ggpllot2 only the corresponding call is required <code>geom_*()</code> function.</p>
<p><img src="/assets/images/r-learning-notes/r-stat-visualization-13.png" alt="R Statistical Visualization-13"></p>
<pre><code class="language-r"># 汽车马力与每加仑汽油行驶里程的关系
library(ggplot2)

p &lt;- ggplot(data = mtcars, aes(x = hp, y = mpg)) +
  geom_point() +
  geom_smooth(method = &quot;loess&quot;) +
  labs(x = &quot;马力&quot;, y = &quot;每加仑汽油行驶里程&quot;)

print(p)
</code></pre>
<p><code>ggplot()</code> Specifies the data source and variables, the geometry function determines how these variables are expressed as points, bars, lines or shadow areas. Common functions are as follows:</p>
<table>
<thead>
<tr>
<th>Functions</th>
<th>Geometric Object</th>
</tr>
</thead>
<tbody><tr>
<td><code>geom_bar()</code></td>
<td>Bar Chart</td>
</tr>
<tr>
<td><code>geom_boxplot()</code></td>
<td>Line Chart</td>
</tr>
<tr>
<td><code>geom_density()</code></td>
<td>Density Chart</td>
</tr>
<tr>
<td><code>geom_histogram()</code></td>
<td>Histogram</td>
</tr>
<tr>
<td><code>geom_hline()</code></td>
<td>Horizontal Line</td>
</tr>
<tr>
<td><code>geom_jitter()</code></td>
<td>Shake point</td>
</tr>
<tr>
<td><code>geom_line()</code></td>
<td>Line</td>
</tr>
<tr>
<td><code>geom_point()</code></td>
<td>Scatter Chart</td>
</tr>
<tr>
<td><code>geom_rug()</code></td>
<td>Axis of coordinates</td>
</tr>
<tr>
<td><code>geom_smooth()</code></td>
<td>Compressing Curves</td>
</tr>
<tr>
<td><code>geom_text()</code></td>
<td>Text Notes</td>
</tr>
<tr>
<td><code>geom_violin()</code></td>
<td>Fiddle Chart</td>
</tr>
<tr>
<td><code>geom_vline()</code></td>
<td>Line</td>
</tr>
</tbody></table>
<p>Geometric functions also accept a set of common parameters:</p>
<table>
<thead>
<tr>
<th>Parameters</th>
<th>Meaning</th>
</tr>
</thead>
<tbody><tr>
<td><code>color</code></td>
<td>colour of the object boundary or line; color map refers<a href="#%E6%A0%87%E5%BA%A6%E4%B8%8E%E5%88%86%E7%BB%84">Scales and Grouping</a></td>
</tr>
<tr>
<td><code>fill</code></td>
<td>Fill colour inside the object</td>
</tr>
<tr>
<td><code>alpha</code></td>
<td>Transparency, values range 0 to 1</td>
</tr>
<tr>
<td><code>linetype</code></td>
<td>Line type, as solid, dotted and nodal</td>
</tr>
<tr>
<td><code>size</code></td>
<td>size of points; also for width in old version ggpllot2</td>
</tr>
<tr>
<td><code>shape</code></td>
<td>The shape of the dot, see<a href="#%E7%82%B9">Points in Basic Graphics</a></td>
</tr>
<tr>
<td><code>position</code></td>
<td>Bar-forming, or overlap-resistant points</td>
</tr>
<tr>
<td><code>sides</code></td>
<td>Position of coordinates</td>
</tr>
<tr>
<td><code>width</code></td>
<td>Geometric width of objects</td>
</tr>
</tbody></table>
<h3>Statistical transformation</h3>
<p>Statistical changes specify how to process raw data and then hand over the results to geometric objects. Common operations include the statistical frequency of the histogram, the calculation of the fractional number and the estimated density. ggplot2 also supports two-dimensional boxes, such as the measurement of observations in each region after dividing the plane into hexagonal.</p>
<p><img src="/assets/images/r-learning-notes/r-stat-visualization-14.png" alt="R Statistical visualization 14"></p>
<pre><code class="language-r"># 钻石重量与价格的蜂巢图
library(ggplot2)

p &lt;- ggplot(data = diamonds, aes(x = carat, y = price)) +
  geom_hex() +
  labs(x = &quot;重量&quot;, y = &quot;价格&quot;, fill = &quot;频数&quot;)

print(p)
</code></pre>
<h3>Scales and Grouping</h3>
<p>The measure controls how the data is mapped to visual properties such as colour, shape, size and coordinate axis. Most of the time, just in the... <code>aes()</code> , ggpllot2 automatically selects the appropriate scale.</p>
<p><img src="/assets/images/r-learning-notes/r-stat-visualization-15.png" alt="R Statistical visualization 15"></p>
<pre><code class="language-r">library(ggplot2)

p &lt;- ggplot(
  data = iris,
  aes(x = Petal.Length, y = Petal.Width)
) +
  geom_point(aes(color = Species, shape = Species)) +
  labs(
    x = &quot;花瓣长度&quot;,
    y = &quot;花瓣宽度&quot;,
    color = &quot;种类&quot;,
    shape = &quot;种类&quot;
  )

print(p)
</code></pre>
<p>Grouping is used to compare observations for two or more groups in the same picture. Group Variables Generally In <code>aes()</code> medium; constants are written in <code>aes()</code> is not interpreted as a data map. The following is a coloured colour for the job title to compare the pay distribution for different job titles:</p>
<pre><code class="language-r">data(&quot;Salaries&quot;, package = &quot;car&quot;)
library(ggplot2)

ggplot(data = Salaries, aes(x = salary, fill = rank)) +
  geom_density(alpha = 0.3)
</code></pre>
<p>You can also call again in a separate geometric layer <code>aes()</code>Let a map affect only this layer.</p>
<h3>Coordinate System</h3>
<p>ggpllot2 defaults to use the Cartesian coordinates system, which also provides polar and map coordinates.<code>coord_flip()</code> You can exchange x-axis and y-axis. Since the graphic is structured by layer, you can save the base graphics and add coordinates to the change.</p>
<p><img src="/assets/images/r-learning-notes/r-stat-visualization-16.png" alt="R Statistical visualization 16"></p>
<pre><code class="language-r"># 钻石切工与对数价格的关系
library(ggplot2)
library(patchwork)

diamonds_zh &lt;- diamonds
levels(diamonds_zh&#36;cut) &lt;- c(&quot;一般&quot;, &quot;良好&quot;, &quot;优质&quot;, &quot;珍贵&quot;, &quot;完美&quot;)

p &lt;- ggplot(diamonds_zh, aes(x = cut, y = log(price))) +
  geom_boxplot() +
  labs(x = &quot;切工&quot;, y = &quot;log(价格)&quot;)

print(p / (p + coord_flip()))
</code></pre>
<h3>Partition</h3>
<p>The idea of the fraction comes from the Trellis graphics: first, to tear the data into a subset by one or two classification variables, then to draw separate maps using the same rules. The segment is more appropriate to observe the pattern within the groups than to overlap the grouping of multiple data sets in the same chart.</p>
<p><img src="/assets/images/r-learning-notes/r-stat-visualization-17.png" alt="R Statistical visualization-17"></p>
<pre><code class="language-r"># 按切工分面后的钻石重量密度曲线
library(ggplot2)

diamonds_zh &lt;- diamonds
levels(diamonds_zh&#36;cut) &lt;- c(&quot;一般&quot;, &quot;良好&quot;, &quot;优质&quot;, &quot;珍贵&quot;, &quot;完美&quot;)

p &lt;- ggplot(diamonds_zh, aes(x = carat)) +
  geom_density() +
  labs(x = &quot;重量&quot;, y = &quot;分布密度&quot;) +
  facet_grid(cut ~ .)

print(p)
</code></pre>
<p><code>facet_wrap()</code> and <code>facet_grid()</code> It's two main sub-functions, of which <code>var</code>、<code>rowvar</code> and <code>colvar</code> Both represent classification variables:</p>
<table>
<thead>
<tr>
<th>Functions</th>
<th>Split</th>
</tr>
</thead>
<tbody><tr>
<td><code>facet_wrap(~ var, ncol = n)</code></td>
<td>Press <code>var</code> Split up and line up. <code>n</code> Columns</td>
</tr>
<tr>
<td><code>facet_wrap(~ var, nrow = n)</code></td>
<td>Press <code>var</code> Split up and line up. <code>n</code> Okay.</td>
</tr>
<tr>
<td><code>facet_grid(rowvar ~ colvar)</code></td>
<td>Press <code>rowvar</code> Branch,<code>colvar</code> Breakdown</td>
</tr>
<tr>
<td><code>facet_grid(rowvar ~ .)</code></td>
<td>Every one. <code>rowvar</code> One line at the horizontal level</td>
</tr>
<tr>
<td><code>facet_grid(. ~ colvar)</code></td>
<td>Every one. <code>colvar</code> Level One</td>
</tr>
</tbody></table>
<h3>Theme and Appearance</h3>
<p>The default theme for ggpllot2 uses the grey background and grid lines. Grid lines help readers cross coordinates, and grey background also distinguishes the graphic from the black text in the body. Default style is not a fixed requirement and can be used <code>theme()</code> Changes to individual elements can also be made by switching to other built-in themes or to topics provided by the extension package.</p>
<p>Basic Graphics System <code>par()</code> It won't affect ggpllot2. The appearance of coordinates, legends and background requires the ggpllot2 own scale and theme function settings.</p>
<h4>Coordinate axis</h4>
<table>
<thead>
<tr>
<th>Functions</th>
<th>Common Options</th>
</tr>
</thead>
<tbody><tr>
<td><code>scale_x_continuous()</code> and <code>scale_y_continuous()</code></td>
<td><code>breaks</code> Specify the scale,<code>labels</code> Specify the tic label,<code>limits</code> Control the range of the continuum axis</td>
</tr>
<tr>
<td><code>scale_x_discrete()</code> and <code>scale_y_discrete()</code></td>
<td><code>breaks</code> Select and rank the factor level,<code>labels</code> Specify labels,<code>limits</code> Control the level displayed</td>
</tr>
<tr>
<td><code>coord_flip()</code></td>
<td>Swap cross and vertical axes</td>
</tr>
</tbody></table>
<h4>Legend</h4>
<p>ggpllot2 automatically generates legends based on visual mapping. Legend title usually <code>labs()</code> , position is passed <code>theme()</code> Adjustments:</p>
<pre><code class="language-r">data(&quot;Salaries&quot;, package = &quot;car&quot;)
library(ggplot2)

ggplot(Salaries, aes(x = rank, y = salary, fill = sex)) +
  geom_boxplot() +
  labs(
    title = &quot;Faculty Salary by Rank and Gender&quot;,
    x = NULL,
    y = NULL,
    fill = &quot;Gender&quot;
  ) +
  theme(legend.position = c(0.1, 0.8))
</code></pre>
<p>Legend location can be set to <code>&quot;left&quot;</code>、<code>&quot;top&quot;</code>、<code>&quot;right&quot;</code> or <code>&quot;bottom&quot;</code>, you can also specify the position inside the chart in a binary vector. The previous example indicates the position of 10% from the left edge and 80% from the bottom edge. Use <code>theme(legend.position = &quot;none&quot;)</code> The legend can be deleted.</p>
<h3>Save Graphics</h3>
<p><code>ggsave()</code> You can save a graphic as a file. File extensions determine output format,<code>plot</code> Specify the graphic object to save,<code>width</code> and <code>height</code> Sets the dimensions.</p>
<pre><code class="language-r">myplot &lt;- ggplot(data = mtcars, aes(x = mpg)) +
  geom_histogram()

ggsave(
  filename = &quot;mygraph.png&quot;,
  plot = myplot,
  width = 5,
  height = 4
)
</code></pre>
