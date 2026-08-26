---
title: Spatial Data Analysis
title_zh: 空间数据分析
date: 2026-03-02 12:13:58 +0800
categories:
- Data Science
- Time Series & Spatial Data
tags:
- Spatial Statistics
- Sampling
author: Hyacehila
mathjax: true
hidden: true
excerpt: A systematic overview of spatial data analysis, including core concepts, population properties, sampling, and interpolation
  methods.
description: A systematic overview of spatial data analysis, including core concepts, population properties, sampling, and
  interpolation methods.
excerpt_zh: 对空间数据分析的基本概念、空间总体特性、抽样和插值方法进行了系统的整理和详细介绍。
permalink: /blog/2026/03/02/spatial-data-analysis/
lang: en
translation_key: 2026-03-02-spatial-data-analysis
translation_status: machine
translation_source_hash: 47ffbd8844713859947d6ee02ff67cda961e2218940794365509c27eb88c3fa8
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<h2>Introduction</h2>
<h3>Introduction of spatial data analysis</h3>
<p>Data with spatial coordinates or relative positions are known as spatial data. The classic statistical methodology requires, in most cases, that samples be independent of each other, large samples and repeated repeatedly. Spatial data, on the other hand, generally do not meet the requirements of independence, while spatial heterogeneity exists and cannot be repeated.</p>
<p>The questions in this article can also be addressed<a href="/en/blog/2023/03/18/stochastic-process-basics-notes/">Random process basis: random process definition, digital characteristics and smooth process</a>、<a href="/en/blog/2024/01/01/statistical-forecasting-notes/">Statistical projections: qualitative projections, quantitative projections and extrapolations of trends</a>How the concept of a relatively close read together is developed in different contexts.</p>
<p>Migration of classical statistical methods to spatial data needs to be discussed separately. After many years of research, space data have developed their own theoretical systems.<strong>The entire spatial data analysis system is based on spatial self-relevance</strong></p>
<h3>Origin and development of spatial data analysis</h3>
<p>In 1854, John Show, a spatial analysis of cholera data from London, identified sources of infection and became a common source of both spatial data analysis and epidemiology disciplines.</p>
<p>The spatial data analysis originated from spatial interpolation values of mining drilling data, spatial polygonal data analysis from spatially relevant and retrogressive and metrological geology of socio-economic statistics unit data, and spatial point data analysis from ecological sample analysis.</p>
<p>The field of spatial data analysis is being used extensively by many institutions/areas of social science, and machine learning algorithms are being used extensively in spatial data analysis. As the amount of time and space data is generated, we are also entering the age of time and space combined analysis, that is, analysis of space and time data.</p>
<h3>Spatial data type</h3>
<p>Spatial data are divided into three categories, and we have different spatial data analysis methods for three different spatial data types.</p>
<h4>Spatial continuity data</h4>
<p>Spatial continuity data is also known as geostatistical data</p>
<p>The representative example is surface temperature analysis, soil drilling distribution data, which can generate continuous data via spatial plug-in values.</p>
<h4>Polygon Data</h4>
<p>Polygons, also known as face data (real data) or regional data (regional data)
He's a graphic information in space, either a ruled (remotely sensed image) or an irregular administrative planning map, which is not continuous, with a specific division of blocks, each of which corresponds to a specific attribute value.</p>
<h4>Point Data</h4>
<p>Point data. Focus on spatial location, not property values, e.g. spatial distribution of settlements, spatial distribution of outbreak sites.</p>
<h4>General form of spatial data</h4>
<p>We can record spatial data in the following form.
&#36;&#36;{z(s):s\in D}&#36;&#36;
of which&#36;D&#36;It's our research area, a subset of the full-coordinate space, an unlimited but generally 2-dimensional plane or 3-dimensional space information.
At every point&#36;s&#36;The real value on it is a random variable, all of it.&#36;Z(s)&#36;It's the generality of our population. We get samples of the total space because of the infinite spatial spectrometry (sample) </p>
<p>Based on this general pattern, we can harmonize the original spatial data type.</p>
<ul>
<li>For spatial data continuity,&#36;D&#36;Yes.&#36;R^d&#36;A fixed continuum subset</li>
<li>For polygonal data,&#36;D&#36;Yes.&#36;R^d&#36;A constant set of specs, which no longer have continuous properties.</li>
<li>For point data,&#36;D&#36;Yes.&#36;R^d&#36;A random subset of,&#36;Z(s)&#36; It's degraded, no attribute value.</li>
</ul>
<h3>Space data analysis methods</h3>
<h4>On spatial data</h4>
<p>In general, we measure spatial point data and spatial continuity data using point spacing and semi-variant functions. For polygonal data, however, a connectivity matrix is generally used to achieve this. The two forms of expression are different and the thinking is similar.</p>
<p>Spatial data types can be converted to each other and used to express different issues. Forthringham integrates the Kriging model and SAR methods in the polygonal data in a continuous data analysis into a system. We can also convert the number of cases in the original polygonal data into a disease and convert them into point data, or construct the regional equivalence to a continuous data, using different methods for analysis.</p>
<h4>Spatial statistical flow</h4>
<p>Spatial data analysis has a thinking on the problem that is close to classical statistical analysis, and we all need to extrapolate from the total sample and then the total.</p>
<p>But there is no so-called I.I.D., and when there is a large spatial heterogeneity and the number of samples is insufficient, the location of the sample will significantly influence the statistical inference, i.e., the close connection between the sampling method and the statistical inference.</p>
<h3>Model selection</h3>
<p>We'll be back with a model by model. This is a general overview.</p>
<ul>
<li>There may be space-related overall spatial distribution objects (using Moran)&#39;s &#36;I&#36;Or semi-variant function test) </li>
<li>There may be spatial heterogeneity.&#36;q&#36;Test) </li>
<li>Variable acetal issues (different layers or levels)&#36;q&#36;Different, depending on the expertise or maximum&#36;q&#36;(Crowding)</li>
</ul>
<p>If the model assumes that the subject is of the same general nature, it is the appropriate model, giving the corresponding relationship</p>
<ul>
<li>Independent and distributed, using classic statistical methods for research</li>
<li>The space is highly relevant, but it's not very different.</li>
<li>Space is highly differentiated, relevant, no explanation variables are not available or explained, and the Sandwich model is appropriate</li>
<li>Space segmentation and related are very different, with no clear or unaccessible explanation variables, and are broken down into three scenarios<ul>
<li>There are samples of each layer (strata), MSN or P-MSN models.</li>
<li>No samples for certain layers, BSHADE, P-BSHADE model</li>
<li>There's only one sample unit with a supporting variable, the SPA model.</li>
</ul>
</li>
<li>If the explanation variables are clear and accessible, and space is less differentiated and relevant, then the Beyers Level Model (BHM) or multiple regressions are all possible.</li>
</ul>
<p><strong>We'll go back to the whole story of the system.</strong></p>
<h3>Precision assessment</h3>
<p>If real values are available, the test is conducted using real values, and consideration may be given to leaving some sample tests; the sample size is insufficient and the classic cross-certification method or one can be used.</p>
<p>If real values are missing, the selection of variables whose nature is close to the target variable is tested; modern real values can be tested by means of historical or future plug-ins without real values.</p>
<p>Overall, improvements in spatial data analysis are not evident in the methodology for precision assessment.</p>
<h2>Space Exploration Data Analysis</h2>
<h3>GIS Profile</h3>
<p>The large number of problems in the real world are related to spatial data, and addressing this type of problem requires access to multidimensional spatial coordinates. Traditional statistical software is not very good at this, but, as representative of open source software, R and Python have raised the issue of space data analysis packages, except that there is clearly no more efficient software for processing.</p>
<p>The Geographic Information System (GIS) is a system for spatial data storage, display, management, query, analysis and decision support. This is characterized by the geo-coded data processed as part of the retrieval and processing of information. GIS has a number of specialized software, which need not be limited to traditional statistical analysis software. And...<strong>Most GIS also features commonly used spatial data analysis.</strong>。</p>
<p>We don't have a presentation here, but we need to study it. GIS is basically the best solution for space exploration data analysis, and SEDA, and R of course, has offered us its own solution.</p>
<p><strong>The most common special GIS is ArcGIS.</strong></p>
<h3>GIS principles</h3>
<p>The creation of a GIS involves geographical expression, spatial reference, and spatial data models in three parts, and we have here a brief introduction.</p>
<h4>Geographical expression of elements</h4>
<p>Common geofactory spatial expression of vectors, grids, grids, Voronoi, etc., we understand when we come to the real GIS solution that the pure theory is not clear.</p>
<h4>Space reference system</h4>
<p>The more common coordinate systems are the geocentric coordinates system, the ball coordinates system and the most common Cartesian coordinates system. The Cartesian coordinate system is the most common of these.</p>
<p>Among the spatial data analysis problems, we need to establish a partial 3D coordinate system, which is normally directly based on the Cartesian coordinate system. There is no need for too much GIS discussion.</p>
<p>In real-world GIS, we most often establish a system of coordinates of the surface, and often global. We know that the Earth is an elliptical body and that the Cartesian coordinate system wants to create a flat coordinate system. So we usually need to do the flattening with a transverse Mercator. Local maps, of course, generally use local plane projection systems. With a projection system, we can easily measure the map, calculate the length, size, length, and properties.</p>
<p>In real-world research, we generally see the Earth as an elliptical sphere, and in order to harmonize standards we introduce a baseline as a measurement benchmark, with the usual baseline being WGS84 ED50 NAD83 for global positioning, European positioning, North American positioning, and 2,000-coordinate systems for domestic use.</p>
<p>The usual projection methods (two concepts with the projection system) have cylinders, cone projection, and azimuth projection, three types of which are unavoidable variations in their angles and directions, and are used in different fields, as shown below.
[Spatial data analysis.png]</p>
<p>Space reference is the synthesis of the preceding narratives, and we need to select the reference surfaces that are projected in the coordinates system, and then get a map of the plane.</p>
<h4>Spatial data models</h4>
<p>For the storage and use of computers, we need to build spatial data models.</p>
<p>In spatial data models, we need to store spatial location data, time data, attribute feature data (coding data) and normal feature data. They usually use vectors to store them.</p>
<p>We will study spatial data models when we present the spatial data analysis in R, which do not need our consideration in most GIS systems with visual interfaces.</p>
<h2>General characteristics of space</h2>
<p>Spatial data are unique in relation to general data</p>
<ul>
<li>Space self-relevance</li>
<li>Space heterogeneity</li>
<li>Modified area unit problem</li>
</ul>
<p>The properties that are relatively close to space tend to be more similar than those that are more distant. It's called Tobler, the first rule of geography, and it's a self-relevance of space.
The world is not evenly distributed.
As space is divided, the correlation and regression factors change.</p>
<p>These three characteristics are distinguished from the sampling of I.I.D required in classical statistics, and they also give rise to spatial statistics. We'll give you the overall characteristics of the space below.</p>
<h3>Space self-relevance</h3>
<h4>Definitions and impacts</h4>
<p>If the vicinity and surrounding areas are more similar to the centre, this is called space-related. If similar values tend to be adjacent to each other, they are referred to as negative spatial correlations.</p>
<p>Non-independent spatial data can affect statistical methods based on independent and distribution assumptions, and in general we can consider</p>
<ul>
<li>Scattered samples, reducing correlation between sample points</li>
<li>Use of space-connected matrix feature values for regression models</li>
<li>Add space as a variable to the regression model, i.e. the spatial regression method</li>
</ul>
<p>Space is not only a disadvantage. It makes it possible to [[Spatial data analysis #space plug-in]; at the same time, space regression models can directly use this spatial dependency to improve predictions. ]</p>
<p>Space-related interpretations need to be combined with the value of the indicator itself and knowledge of the relevant area, usually involving experts in the field</p>
<h4>Metric</h4>
<p>To study space self-relevance, we need to give a space connectivity matrix. &#36;W&#36;
&#36;&#36;W={w_{ij&#125;&#125;&#36;&#36;
When a polygon &#36;i,j&#36; Take one when you're next to it or zero.</p>
<p>The most common measure of spatial self-relevance is the Moran's I index.&#36;y&#36;Use the next one.&#36;x&#36; Replace it with a simple mathematical correction to get the following formula.</p>
<p>&#36;&#36;Moran&#39;{\i1 \cH30D3F4}Si=si=si=si}si}si}si}si}si}si^si^si}si}si}si}si}si^si^si^si^si}si}si}si}si}si}si}si}si}si{si{\\si{si{linelinelinesi{si}si}si}si{si}si^si}si^si}si}si^si^si^si}rsi{x{x{x^right^&#125;&#125;&#125;&#125;}&#36;&#36;&#36;&#36;&#36;&#36;&#36;&#36;
We can calculate it. &#36;I&#36; Mostly in &#36;[-1,1]&#36; between positives for positives and negatives for negatives and zeros for non-relevance. &#36;x_i,\bar{x}&#36;  The observations are for a point and for the whole.</p>
<p>Moran's I index has its own hypothetical test method.</p>
<p>Space self-relevance can also be measured by a variable function (semi - varigram)
&#36;&#36;\gamma\left(h\right)=\frac{1}{2n\left(h\right)}\sum_{s=1}^{n\left(h\right)}\left[x\left(s\right)-x\left(s+h\right)\right]^{2}&#36;&#36;
of which&#36;n(h)&#36; ♪ To express the distance ♪ &#36;h&#36; Points logarithmic &#36;x&#36;An image that represents a point observation, and a variable function is generally measured by a variable curve, representing a variable function at a certain distance</p>
<p>We usually set a threshold.&#36;a&#36; When a variable function is less than &#36;a&#36; , which is considered relevant, and which is not, the variable function does not assume the test method</p>
<h3>Space stratification heterogeneity</h3>
<h4>Definitions and impacts</h4>
<p>Space heterogeneity refers to a variation in the properties over random fluctuations in space, while stratification is a difference in the layers of the layers of the atmosphere of the atmosphere of the atmosphere of the atmosphere of the atmosphere of the Earth.</p>
<p>Space-species heterogeneity is a regularity of space heterogeneity. Heterogeneity itself is the foundation of geography: uniqueness vis-à-vis other locations exists in almost every location.</p>
<p>The presence of spatial stratification is causing the properties of the space environment to be inaccurately depicted as local features, and there are ways in which we use this to be more than a few.</p>
<ul>
<li>Classification or zoning, study of regional characteristics</li>
<li>Local model construction</li>
</ul>
<p>Space-species heterogeneity is the regularity of the heterogeneity, and stratification modelling may continue to improve the effects of the model that has been created; at the same time, space-plugging values are also dependent on the study of the heterogeneity of the stroposphere. The presence of layers can also help us with sampling.</p>
<h4>Metric</h4>
<p>The spatial stratification is expressed in classification or partitioning, and is statistically structured as the principle of the smallest difference in the inner layer and the largest difference in the interstory.</p>
<p>So we define the layer of heterogeneity.&#36;q&#36; We're counting it.
&#36;&#36;q=1-\frac{1}{N\sigma^{2&#125;&#125;\sum_{h=1}^{L}N_{h}\sigma_{h}^{2}&#36;&#36;
The range to be taken is &#36;[0,1]&#36; When it's close to zero, it's the best.</p>
<h3>Space General Characteristics Summary</h3>
<p>When the whole is independent and stable, we should use classical statistics, and at this point i.i.d. is all data content.</p>
<p>When we test space self-relevance or spatial heterogeneity, we need to use spatial statistics/spatial data analysis for this.</p>
<p>The spatial second-stage smooth assumption is that the attribute value of each point is a random variable, that the mathematical expectations of each point are equal and that the correlation between the two points randomly variable is related only to the distance between them, and not to the absolute location of the two, and that based on the second-stage smooth assumption, we have a space-based self-relevance approach represented by the Kring space plug.</p>
<p>Space segmentation does not satisfy the second-tier smooth assumption, resulting in a spatially differentiating approach represented by geographic detectors and Sandwich space plugs.</p>
<p>All right, all right.</p>
<ul>
<li>Classic statistics are used if spatially related and spatially differentiated differences are not significant</li>
<li>If only space-related is significant, use Kwiding plugs and space regression</li>
<li>If only space is significant, use Sandwich plugs and stratifications</li>
<li>If all are significant, use models such as MSN SPA</li>
</ul>
<h2>Space sampling</h2>
<p>Spatial sampling is the method of obtaining statistical extrapolation samples. The classic statistical studies of sampling methods are no longer fully applicable in the face of the self-relevance and heterogeneity of spatial data. There is therefore a need to discuss the sampling in spatial statistics separately</p>
<p>In the big data age, the position of spatial sampling and even of the entire sample survey system has declined because we are better able to make complete samples, so here we are briefly presented.</p>
<p>As for the form of sample statistics, we have selected the statistical data for the estimated overall average.</p>
<h3>Simple sample in space</h3>
<p>A simple sample of space means a number of units drawn from geo-spatial probability, in the form of basic statistical quantities.
&#36;&#36;\overline{y}=\frac{1}{n}\sum_{i=1}^{n}y_{i}&#36;&#36;</p>
<p>The exact extract is a point, a sample, an administrative unit acceptable.</p>
<h3>Specimen of space systems</h3>
<p>The basic idea of system sampling is to extract units at fixed intervals; space system sampling is to spread them slightly as evenly as far as possible into two dimensions. Medium</p>
<p>The basic statistical form is still the same.
&#36;&#36;\overline{y}=\frac{1}{n}\sum_{i=1}^{n}y_{i}&#36;&#36;</p>
<p>Space system sampling is more even than simple space sampling, and space system sampling is more appropriate than simple space sampling when the need for interpolation values using space self-relevance is required</p>
<h3>Spatial stratification sampling</h3>
<p>In the face of spatial heterogeneity, the Specimetric Specimetric Sample and the Sandwich Sample that we're after are more appropriate than the system sample and simple sampling.</p>
<p>The principle of layering is that the difference is the smallest within the layer, the difference between the layers is the largest, and the point where the attribute value is close is divided into the same layer (stratum)</p>
<p>After the layering has been completed, we generally allocate samples to the layers according to some sort of distribution principle. There are common principles.</p>
<ul>
<li>Equal distribution of levels</li>
<li>Distribution according to the number of units in the layer</li>
<li>Distribution according to the multiplier ratio of standard discrete differences and unit numbers for a layer (with a focus on sampling for the discrete power)</li>
</ul>
<p>The formula of the statistics remains unchanged. We can calculate the average of the layers.</p>
<h3>Sampling of Sandwich</h3>
<p>The current sampling method is more often based on sampling of the reporting modules, which in principle should be at least twice the number of reporting units, each of which is drawn at least once. This was detrimental to the compression of sample volumes and the control of sampling costs, and the space Sandwich sample was presented.</p>
<p>The Sandwich sample is an improved layered sampling method that has a good effect on spatial heterogeneity.</p>
<p>First, we still need to stratification and create multiple layers of knowledge. Samples are then distributed to samples based on the knowledge layer, which calculates the average of the statistical volume and the difference. Finally, we match the knowledge layer to the reporting layer, and get the average and the difference between the reporting layer</p>
<p><strong>So we'll do statistical extrapolation in a larger sample and then apply it to smaller units. Go, go, go!</strong></p>
<h2>Space Plugin Value</h2>
<p>Spatial interpolation is an important part of spatial statistics and can be extrapolated from known points to extract a large number of unknown locations.</p>
<p>Plugin values are also used in classical statistics, but are not widely applied; space-plugging values are more commonly used, and they are useful in practice by extrapolating a small sample value to a larger range of spatial properties.</p>
<p><strong>The method of spatial insertion is closely linked to the overall characteristics of space, and the strength and weakness of the differentness and self-relevance influence our choice of interpolation methods</strong></p>
<h3>Nuclear density estimates</h3>
<p>The nuclear density estimate is calculated on the basis of the sample point population of the single variable, and its spatially smoothing estimate is calculated.</p>
<p>Use&#36;s&#36;represent any point in space, use&#36;s_i&#36; It means that the point is known, so it's possible to calculate.&#36;\lambda(s)&#36; Based on
That's a good idea.<em>{\tau}\left(s\right)=\sum</em>I'm sorry, I'm sorry.
%1 nuclear function&#36;k&#36; A predefined inverted U function, which achieves a given&#36;\tau&#36; As a defined value for smoothing, it is in fact a defined smooth radius. Off&#36;s&#36;The further away, the less it will affect them.</p>
<p>Too big.&#36;\tau&#36; It's a big effect on the flat local value.</p>
<p><strong>Nuclear density estimates are conducted using space-related self-relevance</strong></p>
<h3>Trends Plugin Value</h3>
<p>The idea of trend-faced values is straightforward: using a known function to aggregate the distribution, the basic form of the function is determined in advance, with only the unknown parameters estimated.</p>
<p>Trends in the penetration of trends are heavily dependent on the selection of trends; low trends tend to be less likely to produce results, while high trends are more heavily valued</p>
<p><strong>Use of space self-relevance</strong></p>
<h3>Inverse distance weight</h3>
<p>The value of the feature value of the point to be inserted is the weight of the feature value of the point around it, and weight is inversely proportional to the two-point distance function</p>
<p>Inverse distance is often dependent on manual selection, which tends to lead to a situation where the point of value to be inserted is significantly higher than the surrounding sample point</p>
<p><strong>Use self-relevance to do so</strong></p>
<h3>Kriging Method</h3>
<p>The Kringing method uses a linear combination of values of several known sample points within the peri-effect range, as follows:
&#36;&#36;z_{0}=\sum_{1}^{n}\lambda_{1}z_{1}&#36;&#36;
The Kringing method relies on a second-order smooth assumption.</p>
<ul>
<li>The first steps of each point are unknown but consistent.</li>
<li>The difference between the two points is related to the distance, not to the absolute position.</li>
</ul>
<p>The Kriging plug is the best way to get it under the same assumptions as above.&#36;\lambda_i&#36;  Make the projections neutral, the calculations.&#36;\lambda_i&#36; The algebra equation is
&#36;&#36;\begin{cases}\sum_{j}^{n}\lambda_{j}C\left(z_{i},z_{j}\right)+\mu=C\left(z_{i},z_{0}\right)\\sum_{j}^{n}\lambda_{j}=1\end{cases}&#36;&#36;
Or...
&#36; \begin{cases}sum^<em>{j}\lambda</em>\gamma\left({i), z \right)+mu=gamma\left({i}, z zright)\sum\ \lambda j}=end{cases}. &#36;
Functions&#36;\gamma&#36; He and the co-conforming difference are used to measure the degree of correlation.</p>
<h3>CoKriging Method</h3>
<p>Estimated value&#36;z&#36;& Other Variables&#36;x&#36; When it's relevant, these compost variables also contain information on the main variable that we can use to supplement our efforts to achieve the following:&#36;z&#36;The estimates, the CoKriging method.
&#36;&#36;&#36;220<em>{0}=\sum</em>I'm not gonna let you go.
The specific coefficient solver is not repeated here.</p>
<h3>Sandwich Plugin</h3>
<p>When space relevance is weak, the interpolation method based on space self-relevance cannot be implemented, and here we present Sandwich interpolation, which is the only interpolation method used in this chapter when space is less relevant and space is highly differentiated.</p>
<p>If the correlation is highly different, there is a specific treatment, but not the scope of the study in this section. ] Internal</p>
<p>The steps in calculating the Sandwich plug-in are as follows:</p>
<ul>
<li>Layer the target by the minimum intra-group deviation and the largest inter-group deviation to obtain the knowledge layer</li>
<li>Calculate the average of knowledge layers to the difference</li>
<li>The report layer of knowledge and more finer grains is superseded to obtain the values of the various reporting units</li>
</ul>
<p>Method and [(Spatial data analysis#Sandwich sample] The idea is exactly the same, actually, and they do the same thing, combining sampling and extrapolation, not separate.</p>
<h2>Space patterns</h2>
<p>Space-formulation studies of space differences that are completely beyond random differences fall within the category of [[Spatial Data Analysis #Space Exploration Data Analysis SEDA]], although we do not present our theoretical ideas in those areas.</p>
<h3>Space point pattern</h3>
<p>Four main methods of identifying the point pattern are different, with input and output, and it is sufficient to select the method according to the form and demand of the data. The data we're analysing are for the [Spatial Data Analysis # Point Data] described earlier.</p>
<h4>Sample Analysis</h4>
<p>Sample analysis (quadrant anonysis QA) uses a set of square grids to measure the average points and square differences in the grids, and then studies spatial patterns in random, scattered or conglomerate terms using the average and square differences</p>
<p>We usually use VMRs as a specific indicator.
&#36;&#36;VRM=\frac{\sqrt{Var(X)&#125;&#125;{\bar{X&#125;&#125;&#36;&#36;
&#36;&#36;VRM\sim \chi^2(n-1)&#36;&#36;
And when the balance is evenly spread,&#36;VRM=0&#36;  When randomly distributed &#36;VRM=1&#36; When &#36;VRM&gt;At &#36;1, there is a gathering</p>
<h4>Periphery Index</h4>
<p>The closest neighbor index method (nearest neighbor indicator NNI) judge distribution patterns by point to the nearest distance. The idea is to compare the average distance from the nearest point of view to the nearest point of view of the actual observation and the nearest point of view of the pattern of the random distribution.</p>
<p>The calculation of the nearest distance is
&#36;&#36;r=\frac{1}{n}\sum_{i=1}^{n}\min\left(d_{ij}\mid\forall j\right)&#36;&#36;</p>
<p>The nearest range randomly distributed is &#36;Er = 0.5\sqrt{A/n}&#36;  of which&#36;A&#36; It's the size of the study area.</p>
<p>Define the nearest index NNI has
&#36;&#36;NNI = \frac{r}{Er}&#36;&#36;</p>
<p>NNI uses one to divide the line more than one, which means the sample is scattered. Less than one means sample gathering.</p>
<h4>Levels</h4>
<p>Search for spatially present congregates according to the distance-group approach</p>
<h4>Ripley&#39;sK function</h4>
<p>The distribution characteristics of the point elements may vary depending on the observation scale, and the concentration of small scales may be random or evenly distributed on a larger scale. Ripley.&#39;sK function can analyse space distribution patterns at any scale, and is therefore the most common method of analysis for space point patterns</p>
<p>Variable Ripley&#39;sK(d) function for distance&#36;d&#36;Average of time within and ratio of event density within the region
&#36;&#36;K\left(d\right)=\frac{\sum_{i=1}^{n}N\left(i,d\right)}{n}/\frac{n}{A}=\frac{A}{n^{2&#125;&#125;\sum_{i=1}^{n}N\left(i,d\right)&#36;&#36;
In this formula&#36;n&#36; The number of incidents in the area is estimated to be about 50 percent.&#36;N(i,d)&#36;Yes and&#36;i&#36; Distance is&#36;d&#36; The number of incidents in the range,&#36;A&#36;The area of the study area is, &#36;\lambda=n/A&#36; It's the spatial density of events.</p>
<p>We can calculate if the situation is evenly distributed.&#36;K(d) = \pi d^2&#36; So we can construct the following indicators.
&#36;&#36;\Delta\left(d\right)=K\left(d\right)-\pi d^{2}或L\left(d\right)=\sqrt{\frac{K\left(d\right)}{\pi&#125;&#125;-d&#36;&#36;
When the indicator above is greater than 0, the point element is concentrated, and less than 0 reflects the spread</p>
<h3>Hotspots.</h3>
<p>Hotspot study properties are significantly higher than the sub-areas elsewhere</p>
<p><strong>The results of the various hotspot studies are in practice very different.</strong></p>
<h4>Gi</h4>
<p>The Getis-Ord Gi statistical method identifies hot and cold points by calculating the Gi value for each element. If the Gi value of one region is significantly higher than that of other regions, the region may be considered a hot spot. Conversely, if Gi values are significantly lower than in other regions, it could be a cold spot.</p>
<p>The Gi value formula is
&#36;&#36;G_{i}^{*}\left(d\right)=\frac{\sum_{j=1}^{N}w_{ij}\left(d\right)y_{j&#125;&#125;{\sum_{j=1}^{N}y_{j&#125;&#125;&#36;&#36;</p>
<p>All GIS offers Gi values for hotspot assessment, which is likely to be measured by us.&#36;d&#36;Impact</p>
<h4>LISA</h4>
<p>LISA (local indicator of special association) specifically known as the Moran region&#39;s I index to measure spatial self-relevance in local space, i.e. hotspot issues</p>
<p>whose formula is
&#36;&#36;I_{i}=\frac{y_{i}-\overline{y&#125;&#125;{S^{2&#125;&#125;\sum_{j}^{n}w_{ij}\left(y_{j}-\overline{y}\right)&#36;&#36;</p>
<h4>Spatial scanning statistics SatScan</h4>
<p>Space scanning is a method of detecting concentration within the study area using a series of scanning circles. The ratio is calculated by the actual and expected value of the cases in and out of the circle. Depending on the probability distribution of different cases (this is the concentration method for studying diseases), using different seemingly comparable formulas,</p>
<p>Porcelain Appearance Ratio Calculation Formula is
&#36;&#36;LR=\left(\frac{c}{\mu}\right)^{c}\left(\frac{C-c}{C-\mu}\right)^{C-c}=\left(\frac{c}{n\frac{C}{N&#125;&#125;\right)^{c}\left(\frac{C-c}{C-n\frac{C}{N&#125;&#125;\right)^{C-c}&#36;&#36;</p>
<h3>Space is different.</h3>
<p>Space segmentation means studying space stratification heterogeneity, which they can use.&#36;q&#36; Statistically, we'll study this spatial pattern properties in [Spatial Data Analysis #geographic probe]</p>
<h2>Space return</h2>
<p>Space self-relevance influences the results of classical linear regression models, and at this point we need to adopt regression models that take into account space relevance</p>
<h3>Generic regression model for grid data</h3>
<p>The common form of the space regression equation is given by Anselin. Out
&#36;0 = \rho W  ~ \ varepsilon=\lambda W} \ \ varepsilon+ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ right } } } } } } } } } } } } } } } } } } } \ \ \ \ \ \ \ \ \ } } \ } \ } } } \ \ \ } } } } } } } } } } } } } } } } } } } } } } &gt;&#36;0.00
of which &#36;X&#36; It's a matrix of variables from the traditional regression model. &#36;y&#36; It's a observation vector. &#36;W_1&#36; Connection between reaction samples &#36;p&#36; It is the coefficient of the spatial lag variable. &#36;W_2&#36; The spatial connection between the reactional disability can be set up and &#36;W_1&#36; Exactly the same.</p>
<p>In all, there are three super-parameters in control of the entire regression equation. &#36;\lambda ,\rho, a&#36;<br>When they took all 0, this was the classic regression equation, which, on the basis of the generic equation, produced two space regression models -- space lag and space error models.</p>
<h3>Space lag model</h3>
<p>We build on the universal model, the basic form of the space lag model is
&#36;&#36;y=\rho Wy+X\beta+\mu &#36;&#36;</p>
<p>This model takes into account the self-relevance of space connections, and actually we're making superparameters at this point. &#36;\lambda = 0&#36; </p>
<h3>Space error model</h3>
<p>If space dependence is caused by the self-variant that ignores a space impact, then the space error model can model it, and we can then let it be a model for the space impact. &#36;\rho = 0&#36;  Modelling using self-relevance between the disabilities, the basic form of the model is
&#36;&#36;00begin{gathered}
\y=X\beta+\varepsilon \
\varepsilon \lambda W\varepsilon+mu<br>I'm sorry, I'm sorry.
Anselin suggested that the two models be selected by conducting LMG-error tests, which are more visible, and then reverting to OLS for modelling.</p>
<h3>Geographic weighted regression (GWR)</h3>
<p>The idea of geo-gravative re-entry is essentially local re-entry, with a local linear re-entry model modelled, and his regression coefficient.&#36;a&#36; It's no longer a single unit of globality, but a single unit of space.</p>
<p>The idea of a geographically weighted regression is close to [(machine learning introduction and supervision learning#integrated learning# dynamic sorter selection (DCS)], but there are still differences. ]</p>
<p>GWR solvers use a local weighted minimum two-fold regression, which is a function of the geographic distance from the point to be estimated to the other point of observation. The mathematical model is in the form of
&#36;&#36;y_{i}=a_{0}\left(u_{i},v_{i}\right)+\sum_{k}a_{k}\left(u_{i},v_{i}\right)x_{ik}+\varepsilon_{i}&#36;&#36;
of which&#36;u_i,v_i&#36; It's a space coordinate. </p>
<h2>Geographic probe</h2>
<p>The spatial regression is the process of the variable.&#36;Y&#36;& and self-variant&#36;X&#36; Linkage, which in fact is also reflected in the consistency of the spatial distribution of variables and self-variant, requires excavation by geographical detectors.</p>
<p>When linear regression models are significant, geographic detectors are necessarily significant, but not necessarily, and as long as there is a correlation between variables, geo-detectors can detect them.</p>
<p>The geo-detectors are subject to space-specific heterogeneity, the basic idea being: <strong>So long as the variables have an impact on the variables, there should be consistency in spatial distribution.</strong> Here, space can be geospace, attribute space, time classification, and so on.</p>
<p>The geoprospectator contains four detectors.</p>
<ul>
<li>Is there space-species heterogeneity, and what factors cause it?</li>
<li>Variables &#36;Y&#36; Is there a significant difference?</li>
<li>&#36;X&#36; What's the relative importance of it?</li>
<li>Factors &#36;X&#36; Yeah. &#36;Y&#36; Is it independent or is it interactive in any sense?</li>
</ul>
<h3>Space stratification heterogeneity and factor detection</h3>
<p>For the first question, we use&#36;q&#36; Value Metrics
&#36;&#36;q=1-\frac{\sum_{h=1}^{L}N_{h}\sigma_{h}^{2&#125;&#125;{N\sigma^{2&#125;&#125;=1-\frac{SSW}{SST}&#36;&#36;</p>
<p>of which&#36;h&#36; It's a layer number. </p>
<p>&#36;q&#36; The greater the value, the greater it is.&#36;Y&#36;The more obvious the spatial divide (if the layers are made using Y)
When used in the layer &#36;X&#36; And when it goes on,&#36;q&#36; The larger the value, the greater the explanation of the variable from the variable.</p>
<h3>Risk zone detection</h3>
<p>♪ Want to answer the second question ♪&#36;t&#36; Statistical testing
&#36;t (overline{y}<em>{h-1}-\overline{y}</em>{h-2&#125;&#125;=\frac{\overline{Y}<em>{h=1}-\overline{Y}</em>{h=2&#125;&#125;{\left[\frac{Var\left(\overline{Y}<em>{h=1}\right)}{n</em>{h1&#125;&#125;+\frac{Var\left(\overline{Y}<em>{h=2}\right)}{n</em>- I'm sorry, I'm sorry.
of which&#36;h&#36; It's a layer number. </p>
<h3>Ecoprospecting</h3>
<p>For the second question, we can compare which of the two self-variant variables is more important by constructing F statistics.
&#36;&#36;F=\frac{n_{X1}\left(n_{x2}-1\right)SSW_{X1&#125;&#125;{n_{X2}\left(n_{x1}-1\right)SSW_{X2&#125;&#125;&#36;&#36;</p>
<h3>Interactive testing</h3>
<p>For the fourth question, the method we use is</p>
<ol>
<li>Calculate two factors for each. &#36;Y&#36; Yes. &#36;q&#36; Value </li>
<li>The layer that folds two factors into the same layer is new.&#36;q&#36; Value</li>
<li>Compare three.&#36;q&#36; Values.</li>
</ol>
<p>of which the judgement form is</p>
<table>
<thead>
<tr>
<th>The judgement</th>
<th>Interactive</th>
</tr>
</thead>
<tbody><tr>
<td>&#36;q_{12} &lt; min {q_1,q_2}&#36;</td>
<td>Non-linear weakening</td>
</tr>
<tr>
<td>&#36;min {q_1,q_2}&lt;q_{12} &lt; max {q_1,q_2}&#36;</td>
<td>Single factor is less linear</td>
</tr>
<tr>
<td>&#36;max {q_1,q_2}&lt;q_{12}&#36;</td>
<td>Double Factor Enhancement</td>
</tr>
<tr>
<td>&#36;q_{12} = q_1+q_2&#36;</td>
<td>Independence</td>
</tr>
<tr>
<td>&#36;q_{12} &gt; q_1+q_2&#36;</td>
<td>Non-linear enhancements</td>
</tr>
</tbody></table>
<h2>Space and time analysis methods</h2>
<p>Time and space analysis requires a combination of time and space data analysis, which is left for future learning, mainly to include</p>
<ul>
<li>EOF and Small Wave Analysis</li>
<li>The biggest entropy in Bayes.</li>
<li>Bayesian Level Model</li>
<li>Geo-expulsion Tree</li>
<li>Genbank Sequence-Sequence Evolution Analysis</li>
</ul>
