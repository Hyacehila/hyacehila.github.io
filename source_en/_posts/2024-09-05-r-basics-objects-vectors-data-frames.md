---
title: 'R Basics: Objects, Vectors, Data Frames, Functions, and Data Analysis'
title_zh: R 基础：对象、向量、数据框、函数与数据分析
date: 2024-09-05 10:29:12 +0800
categories:
- Programming
- Programming Languages
tags:
- R
- Data Management
- Exploratory Data Analysis
author: Hyacehila
mathjax: true
hidden: true
excerpt: A practical guide to R objects, vectors, matrices, data frames, lists, time series, functions, data import, data
  management, missing-data handling, and exploratory data analysis.
description: A practical guide to R objects, vectors, matrices, data frames, lists, time series, functions, data import, data
  management, missing-data handling, and exploratory data analysis.
excerpt_zh: 系统整理 R 的对象、向量、矩阵、数据框、列表、时间序列、函数与常用内置函数，并覆盖数据导入、数据管理、缺失值处理和探索性数据分析。
permalink: /blog/2024/09/05/r-basic-learning-notes/
lang: en
translation_key: 2024-09-05-r-basics-objects-vectors-data-frames
translation_status: machine
translation_source_hash: 597eee958cdcae9e6c62166ec8598f5f3deaad8ecbb0819af1d9447926095673
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<h2>R. Rationale</h2>
<p>R is a programming language for data analysis. Its core is not complex: objects save data, functions process objects, and packages provide additional functions and data sets. The note is based on these basic concepts and goes back into data import, collation and exploration.</p>
<p>R itself contains an interpreter and a standard library, RStudio is a common integrated development environment. Both need to be installed separately; RStudio edits the code, manages the project and display objects, and the actual calculation is still R.</p>
<p>function. Enter the name of the function object directly, and R will print its definition; the function will only be called if it is in parentheses.</p>
<pre><code class="language-r">ls       # 查看函数对象
ls()     # 调用函数
</code></pre>
<p>function is also an object, so an object cannot be judged by whether it is a function by whether it is in round brackets. Available for inspection <code>is.function()</code>。</p>
<pre><code class="language-r">is.function(ls)
</code></pre>
<p>The R package is installed in the local library and passed before use <code>library()</code> Loads the package to the current session. The package will only be installed once, but it will be reloaded every time a new R session starts.</p>
<pre><code class="language-r">install.packages(&quot;readxl&quot;)  # 只需安装一次
library(readxl)             # 每个新会话都要加载
</code></pre>
<h3>Object & Value</h3>
<p>R Common <code>&lt;-</code> Granted.<code>=</code>、<code>assign()</code> And from right to left. <code>-&gt;</code> It can be given value, but mixing reduces readability.</p>
<pre><code class="language-r">n &lt;- 10
n = 10
assign(&quot;n&quot;, 10)
10 -&gt; n
n
</code></pre>
<p>Object name is case sensitive. The generic name starts with the letter or does not follow the number, and can be followed by letters, numbers, points and underlineds.<code>x</code> and <code>X</code> Two different objects.</p>
<p>When the expression is not given a value, the result will only be printed to the console and will not be automatically saved. Comment from <code>#</code> Start, continue until the end of the line.</p>
<pre><code class="language-r">2 + 3  # 打印 5，但不保存结果

result &lt;- 2 + 3
result
</code></pre>
<h2>Data Structure for R</h2>
<p>R Object has both a bottom storage type and a possible possession <code>class</code>、<code>dim</code>、<code>names</code> equals. It is more accurate to understand these two layers of information than to create a rigid “type” of mutually exclusive objects.</p>
<h3>Object Properties and Information</h3>
<p>The bottom types common to atomic vectors include logical, integer, double-precision, plural, character and original bytes. The factor is not a separate bottom type, but has <code>levels</code> and <code>class = &quot;factor&quot;</code> attribute's integer vector.</p>
<pre><code class="language-r">x &lt;- 1:5

typeof(x)       # 底层存储类型
class(x)        # 对象类别
length(x)       # 元素个数
attributes(x)   # 对象属性
</code></pre>
<p><code>is.*()</code> function is used to judge objects,<code>as.*()</code> function. Call with the actual object.</p>
<pre><code class="language-r">is.numeric(x)
digits &lt;- as.character(x)
</code></pre>
<p>R uses several special values to describe situations that cannot be treated in normal values:</p>
<ul>
<li><code>NA</code> is the missing value.</li>
<li><code>NaN</code> indicates undefined numerical results, e.g. <code>0 / 0</code>。</li>
<li><code>Inf</code> and <code>-Inf</code> It means positive and negative.</li>
<li><code>NULL</code> Usually indicates that the object or result does not exist and the missing value is one length <code>NA</code> Different.</li>
</ul>
<p>Character values are placed in single or double quotation marks. When similar quotation marks are required in a string, you can transpose them with a backslash.</p>
<pre><code class="language-r">message &lt;- &quot;Double quotes \&quot; delimit R strings.&quot;
</code></pre>
<p>Common data structures are as follows:</p>
<table>
<thead>
<tr>
<th>Data structure</th>
<th>Main features</th>
<th>Whether columns or elements are allowed to have different types</th>
</tr>
</thead>
<tbody><tr>
<td>Vector</td>
<td>One dimension, homogeneity</td>
<td>Yes</td>
</tr>
<tr>
<td>Factor</td>
<td>Encoding classification levels with integer</td>
<td>Yes</td>
</tr>
<tr>
<td>Array</td>
<td>Multi-dimensional, homogenous</td>
<td>Yes</td>
</tr>
<tr>
<td>Matrix</td>
<td>2-D array</td>
<td>Yes</td>
</tr>
<tr>
<td>Data Box</td>
<td>2D table, each column is a vector</td>
<td>Yes.</td>
</tr>
<tr>
<td>Time series</td>
<td>Vector or matrix with time index properties</td>
<td>Yes</td>
</tr>
<tr>
<td>List</td>
<td>Can accommodate any object</td>
<td>Yes.</td>
</tr>
</tbody></table>
<p>Some common operators:</p>
<table>
<thead>
<tr>
<th>Operations</th>
<th>Operators</th>
</tr>
</thead>
<tbody><tr>
<td>Multiplication</td>
<td><code>^</code></td>
</tr>
<tr>
<td>Modelling</td>
<td><code>%%</code></td>
</tr>
<tr>
<td>Division</td>
<td><code>%/%</code></td>
</tr>
<tr>
<td>Matrix Multiplication</td>
<td><code>%*%</code></td>
</tr>
</tbody></table>
<p><code>ls()</code> You can list objects in the current environment or you can filter them by name.<code>rm()</code> to remove the object.</p>
<pre><code class="language-r">ls()
ls(pattern = &quot;^m&quot;)

rm(x)
rm(list = ls(pattern = &quot;^m&quot;))
</code></pre>
<h3>Vector</h3>
<p>Vector is a set of elements of the same type.<code>c()</code>、<code>:</code>、<code>seq()</code> and <code>rep()</code> The most common way to create it.</p>
<h4>Numerical vector</h4>
<pre><code class="language-r">1:10
seq(1, 5, by = 0.5)
rep(2:5, times = 2)
c(42, 7, 64, 9)
</code></pre>
<p><code>scan()</code> Data can be read from the console or text connection, but the script is usually more suitable for using clear data import functions.</p>
<h4>Character vector</h4>
<pre><code class="language-r">colors &lt;- c(&quot;green&quot;, &quot;blue sky&quot;, &quot;-99&quot;)
paste(c(&quot;X&quot;, &quot;Y&quot;), 1:2, sep = &quot;&quot;)
</code></pre>
<p><code>paste()</code> It connects characters,<code>paste0()</code> Equivalent <code>paste(..., sep = &quot;&quot;)</code>。</p>
<h4>Logical vector</h4>
<p>Logical values include <code>TRUE</code>、<code>FALSE</code> and <code>NA</code>I don't know. Comparative calculations usually generate a logical vector.</p>
<pre><code class="language-r">x &lt;- c(10.4, 5.6, 3.1, 6.4, 21.7)
x &gt; 13
7 != 6

all(1:7 &gt; 3)
any(1:7 &gt; 3)
</code></pre>
<p><code>T</code> and <code>F</code> It can be revalued and not suitable for replacement in the official code. <code>TRUE</code> and <code>FALSE</code>。</p>
<h4>Factor</h4>
<p>The factor is used to represent the classification variable. It saves the integer coding for observation and passes <code>levels</code> Record category name. Sets when categories are sequential <code>ordered = TRUE</code>。</p>
<pre><code class="language-r">colors &lt;- c(&quot;green&quot;, &quot;blue&quot;, &quot;green&quot;, &quot;yellow&quot;)
color_factor &lt;- factor(colors)

scores &lt;- factor(
  c(1, 2, 3, 1),
  levels = c(1, 2, 3),
  labels = c(&quot;low&quot;, &quot;middle&quot;, &quot;high&quot;),
  ordered = TRUE
)
</code></pre>
<p><code>gl()</code> A rule-based factor can be generated.<code>n</code> It's horizontal.<code>k</code> Is the number of consecutive repetitions at each level.<code>length</code> is the length of the result.</p>
<pre><code class="language-r">gl(n = 3, k = 2, length = 6, labels = c(&quot;A&quot;, &quot;B&quot;, &quot;C&quot;))
</code></pre>
<p>The following functions are commonly used to check and summarize factors:</p>
<pre><code class="language-r">sex &lt;- factor(c(&quot;M&quot;, &quot;F&quot;, &quot;M&quot;, &quot;M&quot;, &quot;F&quot;))
height &lt;- c(174, 165, 180, 171, 160)

is.factor(sex)
levels(sex)
table(sex)
tapply(height, sex, mean)
</code></pre>
<h4>Vector Operations & Cycle Completion</h4>
<p>Vector operations are done on an element-by-element basis. The length of the two vectors is different and the shorter vectors are reused, which is referred to as the recycling rule. If the length of the longer vector is not a multiple of the integer length of the shorter vector, R usually gives a warning; relying on this incomplete loop to fill in can easily hide an error.</p>
<pre><code class="language-r">x &lt;- c(10.4, 5.6, 3.1, 6.4, 21.7)
x * 2 + 1

c(1, 2, 3, 4) + c(10, 20)
</code></pre>
<p>Common summary functions:</p>
<table>
<thead>
<tr>
<th>Functions</th>
<th>Role</th>
</tr>
</thead>
<tbody><tr>
<td><code>min(x)</code>、<code>max(x)</code></td>
<td>Min & Maximum</td>
</tr>
<tr>
<td><code>which.min(x)</code>、<code>which.max(x)</code></td>
<td>Location of minimum and maximum values</td>
</tr>
<tr>
<td><code>mean(x)</code>、<code>median(x)</code></td>
<td>Mean to Medium</td>
</tr>
<tr>
<td><code>var(x)</code>、<code>sd(x)</code></td>
<td>Difference to Standard</td>
</tr>
<tr>
<td><code>quantile(x)</code></td>
<td>Bits</td>
</tr>
<tr>
<td><code>summary(x)</code></td>
<td>Returns common summary by object category</td>
</tr>
<tr>
<td><code>sort(x)</code></td>
<td>Sort</td>
</tr>
<tr>
<td><code>sum(x)</code>、<code>prod(x)</code></td>
<td>Summation and Multiplication</td>
</tr>
<tr>
<td><code>cov(x, y)</code>、<code>cor(x, y)</code></td>
<td>Differences and related factors</td>
</tr>
</tbody></table>
<p>A lot of summary functions. <code>na.rm</code> parameter. When data contains missing values, it is necessary to clearly decide whether to exclude the missing values, rather than mechanically adding <code>na.rm = TRUE</code>。</p>
<pre><code class="language-r">mean(c(1, 2, NA), na.rm = TRUE)
</code></pre>
<h4>The extraction of vector elements</h4>
<p>The R index starts at 1. In square brackets you can use positive integers, negative integers, logical vectors or names; positive and negative indices cannot be mixed.<code>0</code> Except for.</p>
<pre><code class="language-r">x &lt;- c(42, 7, 64, 9, 10, 8)

x[1:5]
x[c(1, 4)]
x[x &gt; 10]
x[-(1:5)]

x[x &gt; 10] &lt;- 10
</code></pre>
<p>When checking special values, pass the object to the judgement function:</p>
<pre><code class="language-r">values &lt;- c(1, NA, NaN, Inf)

is.na(values)
is.nan(values)
is.finite(values)
is.infinite(values)
</code></pre>
<h3>Numeric & Matrix</h3>
<p>The array is tape. <code>dim</code> property. The matrix is a two-dimensional array. They can only save one lower type; if you mix a character value, the value is usually converted to a character.</p>
<h4>Create arrays and matrices</h4>
<pre><code class="language-r">A &lt;- array(1:8, dim = c(2, 2, 2))
A

X &lt;- matrix(1:8, nrow = 2, ncol = 4)
X_by_row &lt;- matrix(1:8, nrow = 2, ncol = 4, byrow = TRUE)

diagonal &lt;- diag(c(10, 20, 30))
</code></pre>
<p>R Default to fill the matrix by column.<code>byrow = TRUE</code> will be filled by row.</p>
<h4>Matrix and array index</h4>
<p>Matrix Use <code>[行, 列]</code> Index. Ignores one dimension for all elements that select the dimension.</p>
<pre><code class="language-r">x &lt;- matrix(1:6, nrow = 2, ncol = 3)

x[2, 2]
x[2, ]
x[, 3] &lt;- NA
x[is.na(x)] &lt;- 1

x[-1, ]
x[, -2]
</code></pre>
<p>The default of a single line or column result may be reduced to a vector. Use when retaining matrix structure <code>drop = FALSE</code>。</p>
<pre><code class="language-r">x[1, , drop = FALSE]
</code></pre>
<h4>Matrix Operations</h4>
<pre><code class="language-r">X &lt;- matrix(1:4, nrow = 2)

t(X)       # 转置
diag(X)    # 对角线元素
det(X)     # 行列式
X * X      # 对应元素相乘
X %*% X    # 矩阵乘法
</code></pre>
<h3>Data Box</h3>
<p>Data box is the most common table structure in R. Each row usually corresponds to one observation and each column corresponds to one variable. The length of each column must be consistent, but a value, character, logical value or factor can be saved separately.</p>
<h4>Create and check data boxes</h4>
<pre><code class="language-r">measurements &lt;- data.frame(
  id = 1:4,
  value = c(42, 7, 64, 9),
  group = c(&quot;A&quot;, &quot;A&quot;, &quot;B&quot;, &quot;B&quot;)
)

str(measurements)
summary(measurements)
head(measurements)
</code></pre>
<p>In actual analysis, data boxes are more derived from external documents. Import methods are followed by the section on Data Import, Storage and Analysis Preparedness.</p>
<p>Direct Use <code>data&#36;column</code> or function <code>data</code> Parameters, ratio <code>attach()</code> More clearly, the data will not be wrong because of the existence of the same name object in the search path.</p>
<pre><code class="language-r">cor(Puromycin&#36;conc, Puromycin&#36;rate)
pairs(Puromycin, panel = panel.smooth)
xtabs(~ state, data = Puromycin)
</code></pre>
<h4>Select Rows and Columns</h4>
<p>The data boxes support both matrix indexes and listing access.</p>
<pre><code class="language-r">Puromycin[1, 1]
Puromycin[c(1, 3, 5), c(&quot;conc&quot;, &quot;rate&quot;)]
Puromycin&#36;conc

subset(Puromycin, state == &quot;treated&quot; &amp; rate &gt; 160)
</code></pre>
<p><code>subset()</code> Appropriate for interactive analysis. When a function that requires strict control of the value-seeking environment is prepared, it is more prudent to use a visible bracketed index.</p>
<pre><code class="language-r">selected &lt;- leadership[
  leadership&#36;age &gt;= 35 | leadership&#36;age &lt; 24,
  c(&quot;q1&quot;, &quot;q2&quot;, &quot;q3&quot;, &quot;q4&quot;)
]
</code></pre>
<h4>Create and Rename Variables</h4>
<pre><code class="language-r">Puromycin&#36;inverse_conc &lt;- 1 / Puromycin&#36;conc

Puromycin &lt;- transform(
  Puromycin,
  inverse_conc = 1 / conc,
  sqrt_conc = sqrt(conc)
)

names(Puromycin)[names(Puromycin) == &quot;rate&quot;] &lt;- &quot;reaction_rate&quot;
</code></pre>
<p><code>with()</code> is appropriate to read the columns in the data box, but the assigned value does not automatically write back to the original data box.<code>fix()</code> It opens an interactive editor, which is not conducive to recurrence and is therefore not a regular data management method.</p>
<h4>Merge and add data</h4>
<p>A data box with a common key can be used. <code>merge()</code> Connect. The default result is an internal connection, with only the keys that match both sides;<code>all.x = TRUE</code> You can get a left link.</p>
<pre><code class="language-r">total &lt;- merge(dataframe_a, dataframe_b, by = &quot;ID&quot;)
left_total &lt;- merge(dataframe_a, dataframe_b, by = &quot;ID&quot;, all.x = TRUE)
</code></pre>
<p><code>cbind()</code> If you spell objects horizontally in the current line order, you will not be aligned by key, so you should first confirm that the number of lines and the order of lines are consistent.<code>rbind()</code> A vertical data box is added to require that the type of listing and column correspond.</p>
<pre><code class="language-r">wide &lt;- cbind(dataframe_a, extra_columns)
long &lt;- rbind(dataframe_a, dataframe_b)
</code></pre>
<h4>Line identifier</h4>
<p>Line names can save instance identifiers, but are usually better suited to keep identifiers as normal columns, so they are more directly exported, linked and checked for duplicate values.</p>
<pre><code class="language-r">patient_data &lt;- data.frame(
  patient_id = patient_id,
  age = age,
  diabetes = diabetes,
  status = status
)

anyDuplicated(patient_data&#36;patient_id)
</code></pre>
<h3>List</h3>
<p>Lists can accommodate objects of different types and lengths, including vectors, matrices, data frames, functions and other lists. Many modelling functions return to the list, as the adhesion results usually contain coefficients, disabilities and diagnostic information at the same time.</p>
<pre><code class="language-r">results &lt;- list(
  values = 1:6,
  matrix = matrix(1:4, nrow = 2)
)

results&#36;values
results[[1]]  # 提取第一个元素本身
results[1]    # 返回只含第一个元素的子列表
</code></pre>
<h3>Time series</h3>
<p><code>ts()</code> Adds a rule time index to the vector or matrix. It is suitable for an intervalent sequence; data with irregular dates or time zone information are usually used for other time objects.</p>
<pre><code class="language-r">ts(
  data = NA,
  start = 1,
  end = numeric(0),
  frequency = 1,
  deltat = 1,
  names = NULL
)
</code></pre>
<ul>
<li><code>data</code> is a one-dollar vector or a multiple matrix.</li>
<li><code>start</code> and <code>end</code> Specifies the position of the end-of-pipe observations.</li>
<li><code>frequency</code> Indicates the number of observations in each time unit, e.g., quarterly data extraction 4 and monthly data acquisition 12.</li>
<li><code>deltat</code> is the time interval between adjacent observations and <code>frequency</code> Two or one.</li>
<li><code>names</code> Listing for multiple sequences.</li>
</ul>
<pre><code class="language-r">annual &lt;- ts(1:10, start = 1959)
monthly &lt;- ts(1:47, frequency = 12, start = c(1959, 2))
quarterly &lt;- ts(1:10, frequency = 4, start = c(1959, 2))

multivariate &lt;- ts(
  matrix(rpois(36, lambda = 5), nrow = 12, ncol = 3),
  start = c(1961, 1),
  frequency = 12
)
</code></pre>
<h2>Date and time</h2>
<p><code>Date</code> The object is stored at the bottom in numerical terms, representing the relative number of days 1970-01-01. Character Date Required <code>as.Date()</code> Convert to actual format.</p>
<pre><code class="language-r">date_strings &lt;- c(&quot;01/05/1965&quot;, &quot;08/16/1975&quot;)
dates &lt;- as.Date(date_strings, format = &quot;%m/%d/%Y&quot;)
</code></pre>
<p>Common formatrs:</p>
<table>
<thead>
<tr>
<th>Symbol</th>
<th>Meaning</th>
<th>Example:</th>
</tr>
</thead>
<tbody><tr>
<td><code>%d</code></td>
<td>Two dates</td>
<td><code>01</code> to <code>31</code></td>
</tr>
<tr>
<td><code>%a</code></td>
<td>Weekly abbreviations</td>
<td><code>Mon</code></td>
</tr>
<tr>
<td><code>%A</code></td>
<td>Full week name</td>
<td><code>Monday</code></td>
</tr>
<tr>
<td><code>%m</code></td>
<td>Two months.</td>
<td><code>01</code> to <code>12</code></td>
</tr>
<tr>
<td><code>%b</code></td>
<td>Month abbreviation</td>
<td><code>Jan</code></td>
</tr>
<tr>
<td><code>%B</code></td>
<td>Full Month Name</td>
<td><code>January</code></td>
</tr>
<tr>
<td><code>%y</code></td>
<td>Two years.</td>
<td><code>07</code></td>
</tr>
<tr>
<td><code>%Y</code></td>
<td>Four years.</td>
<td><code>2007</code></td>
</tr>
</tbody></table>
<p>Date of acquisition, formatting and comparison:</p>
<pre><code class="language-r">today &lt;- Sys.Date()
now &lt;- Sys.time()

format(today, format = &quot;%B %d %Y&quot;)

dob &lt;- as.Date(&quot;1956-10-12&quot;)
difftime(today, dob, units = &quot;weeks&quot;)
</code></pre>
<p><code>date()</code> returns the character of the current date and time; should continue to calculate <code>Sys.Date()</code> or <code>Sys.time()</code>。</p>
<h2>Data import, storage and analysis readiness</h2>
<p>Recurring analyses should use the project catalogue and relative paths to the extent possible.<code>getwd()</code> You can view the current working directory.<code>file.path()</code> It is possible to clutter paths across platforms. Do not write the absolute path on the PC in shared scripts.</p>
<pre><code class="language-r">getwd()
data_path &lt;- file.path(&quot;data&quot;, &quot;measurements.csv&quot;)
</code></pre>
<h3>Storage data</h3>
<p>Text, CSV and RData are common storage formats. CSV is easy to trade with other software, and RData can save multiple R objects in one file.</p>
<pre><code class="language-r">d &lt;- data.frame(
  observation = c(1, 2, 3),
  treatment = c(&quot;A&quot;, &quot;B&quot;, &quot;A&quot;),
  weight = c(2.3, NA, 9)
)

write.table(
  d,
  file = file.path(&quot;data&quot;, &quot;observations.txt&quot;),
  row.names = FALSE,
  quote = FALSE
)

write.csv(
  d,
  file = file.path(&quot;data&quot;, &quot;observations.csv&quot;),
  row.names = FALSE
)

save(d, file = file.path(&quot;data&quot;, &quot;objects.RData&quot;))
</code></pre>
<h3>Read Data</h3>
<p>Base R can read text and CSV files directly, and can load data sets and RData files with packages.</p>
<pre><code class="language-r">houses &lt;- read.table(&quot;houses.txt&quot;, header = TRUE)
scores &lt;- read.csv(&quot;educ_scores.csv&quot;)

data(&quot;mtcars&quot;)
load(file.path(&quot;data&quot;, &quot;objects.RData&quot;))
</code></pre>
<p><code>read.delim(&quot;clipboard&quot;)</code> The clipboard is readable in part of the desktop environment, but it relies on the operating system and is not suitable for scripts that require stable recurrence.</p>
<p>When reading formats such as SSS, SAS Transport and Stata, you can use <code>foreign</code> Bag. It is a traditional scheme in the basic workflow, and the level of support for different formats in specific functions is not entirely consistent.</p>
<pre><code class="language-r">library(foreign)

spss_data &lt;- read.spss(&quot;educ_scores.sav&quot;, to.data.frame = TRUE)
sas_data &lt;- read.xport(&quot;educ_scores.xpt&quot;)
stata_data &lt;- read.dta(&quot;educ_scores.dta&quot;)
</code></pre>
<p>Excel file not valid <code>foreign</code> Read, save as CSV, or <code>readxl</code>。</p>
<pre><code class="language-r">library(readxl)
so2_data &lt;- read_excel(file.path(&quot;data&quot;, &quot;SO2.xlsx&quot;))
</code></pre>
<p>After import, check the structure, rows and key variables before entering data management. The ability of the file to read does not mean that column type, missing value code and identifier are correct.</p>
<pre><code class="language-r">str(so2_data)
dim(so2_data)
summary(so2_data)
</code></pre>
<h3>Missing data</h3>
<p>Missing values often appear in real data. Before processing, it is necessary to determine where, why and how the observations were used in the analytical methods.<code>is.na()</code> Return the element-by-component result.<code>complete.cases()</code> Mark full row.</p>
<pre><code class="language-r">is.na(d)
colSums(is.na(d))
complete.cases(d)
</code></pre>
<h4>Try Restore</h4>
<p>If the original questionnaire, log or other field can determine the missing value, it can be restored to the data source. For example, when there is a definite relationship between the total score and the sub-item, the missing sub-item can sometimes be checked against it. The recovery must be based on clear grounds and cannot return speculation to original data.</p>
<h4>Full case analysis</h4>
<p><code>na.omit()</code> Deletes the entire line containing the missing value. Many statistical functions will pass. <code>na.action</code> Adopt a similar treatment. This is simple, but it may reduce the volume of samples; it may also introduce deviations when the missing are not entirely random.</p>
<pre><code class="language-r">complete_data &lt;- na.omit(d)
</code></pre>
<p>The deletion of rows should be determined jointly by missing mechanisms, missing proportions and subsequent analysis, rather than as a default cleansing step.</p>
<h4>Multiple plugs</h4>
<p>Multiple plug-ins generate a number of reasonably complete data sets, combining models and consolidating estimates and uncertainties.<code>mice</code> The package provides common realization.</p>
<pre><code class="language-r">library(mice)

imp &lt;- mice(data, m = 5, seed = 123)

fits &lt;- with(imp, lm(y ~ x1 + x2))
pooled &lt;- pool(fits)
summary(pooled)

completed_data &lt;- complete(imp, action = 1)
</code></pre>
<p>Of which:</p>
<ul>
<li><code>data</code> is a data box or matrix with missing values.</li>
<li><code>imp</code> Saves information on multiple plug-in data sets and plug-in processes.</li>
<li><code>with()</code> Implement the same analytical expression on each plug-in data set.</li>
<li><code>pool()</code> Merge model results with multiple plug-in rules.</li>
<li><code>complete()</code> You can extract a complete data set.</li>
</ul>
<p>Only one plug-in data set continues to be extrapolated, and uncertainties between plug-ins are lost. If the follow-up is not directly compatible <code>with()</code> and <code>pool()</code>This limitation and the consolidation strategy used should be clearly documented.</p>
<h3>Explored data analysis</h3>
<p>The exploratory data analysis (EDA) took place before formal modelling. The objective is to understand the relationship between the distribution of variables, anomalies, missing patterns and variables and to check whether the data fit into the research design.</p>
<pre><code class="language-r">str(data)
summary(data)
table(data&#36;group, useNA = &quot;ifany&quot;)

numeric_data &lt;- data[vapply(data, is.numeric, logical(1))]
cor(numeric_data, use = &quot;pairwise.complete.obs&quot;)
</code></pre>
<p>Numerical summary cannot replace graphics. Histograms, box charts, scatter charts and group charts often reveal structures that are invisible to the coefficient. See more complete statistical methods. <a href="/en/blog/2024/09/05/r-classical-statistics-learning-notes/">EDA and descriptive statistics</a>See you at the drawing. <a href="/en/blog/2024/03/15/r-visualization-learning-notes/">R Statistical visualization</a> and <a href="/en/blog/2024/09/16/r-graph-learning-notes/">R Statistical Graphics</a>。</p>
<h2>R Programming</h2>
<h3>Conditions and Looping</h3>
<p>R Provided <code>if</code>、<code>else</code>、<code>switch</code>、<code>for</code>、<code>while</code> and <code>repeat</code> Like control structures. The brackets avoid ambiguity in multi-line branches.</p>
<pre><code class="language-r">if (condition_1) {
  statement_1
} else if (condition_2) {
  statement_2
} else {
  statement_3
}

for (i in 1:5) {
  print(i)
}

i &lt;- 1
while (i &lt;= 5) {
  print(i)
  i &lt;- i + 1
}
</code></pre>
<p><code>switch()</code> Select a branch according to the character or position to handle a small number of fixed options.</p>
<pre><code class="language-r">operation &lt;- &quot;mean&quot;

switch(
  operation,
  mean = mean(1:5),
  sum = sum(1:5),
  stop(&quot;Unknown operation&quot;)
)
</code></pre>
<h3>Quantified</h3>
<p>Quantification is given to a vector function or logical index on an element-by-element basis, rather than a prominent writing cycle. It is usually simpler, and some functions can also call on optimized bottoms, but quantification itself is not equivalent to automatic parallels.</p>
<pre><code class="language-r">y &lt;- numeric(length(x))
y[x == b] &lt;- 0
y[x != b] &lt;- 1

y &lt;- ifelse(x == b, 0, 1)
</code></pre>
<p>The cycle is not wrong. Clear cycles are often more appropriate when the operation is pre- and post-existing, each time it returns complex objects, or when there is no suitable vector function.</p>
<h3>Custom Functions</h3>
<p>function consists of a list of parameters and functions. It can return any R object; no visible call <code>return()</code> , the last expression returns the value.</p>
<pre><code class="language-r">plot_file &lt;- function(title_text, file_path) {
  data &lt;- read.table(file_path, header = TRUE)
  plot(data[[1]], data[[2]], type = &quot;l&quot;)
  title(title_text)

  invisible(data)
}
</code></pre>
<p>You can call a function by location or name. Naming parameters need not follow the order of definitions, but full names should be used to avoid the ambiguity of partial matching.</p>
<pre><code class="language-r">foo1(u, v, w)
foo1(arg3 = w, arg2 = v, arg1 = u)
</code></pre>
<p>Parameters can have default values or pass <code>...</code> Receives additional parameters.</p>
<pre><code class="language-r">foo2 &lt;- function(arg1, arg2 = 5, arg3 = FALSE, ...) {
  list(arg1 = arg1, arg2 = arg2, arg3 = arg3, extra = list(...))
}
</code></pre>
<p>R supports a recursive function, but a deeper regression may be limited by the call stack. Many data processing tasks are more direct using circular or vector functions.</p>
<h3>Help system</h3>
<p>R Help system.<code>help.start()</code> It opens the front page of the help.<code>help()</code> and <code>?</code> to query functions or special syntax.</p>
<pre><code class="language-r">help.start()

help(&quot;lm&quot;)
?lm

help(&quot;bs&quot;, try.all.packages = TRUE)
help(&quot;bs&quot;, package = &quot;splines&quot;)
</code></pre>
<p>Part of the package also provides a vignette document that describes design thinking or complete workflow.</p>
<pre><code class="language-r">vignette()
vignette(package = &quot;survival&quot;)
</code></pre>
<h3>Normal functions, broad and method</h3>
<p>Normal functions directly perform fixed realizations, and generic functions select methods according to object categories. S3 General <code>UseMethod()</code> Distribution. For example:<code>mean()</code> It's based on <code>class(x)</code> Selection <code>mean.default</code>、<code>mean.Date</code> When it's done, it's still called in form. <code>mean(x)</code>。</p>
<pre><code class="language-r">mean
methods(&quot;mean&quot;)
</code></pre>
<p>The typical output shows a broad definition and registered method:</p>
<pre><code class="language-text">function (x, ...)
UseMethod(&quot;mean&quot;)

[1] mean.Date     mean.default  mean.difftime mean.POSIXct  mean.POSIXlt
</code></pre>
<h3>View function source</h3>
<p>For functions performed by R code and currently visible, you can print a function name.</p>
<pre><code class="language-r">lm
</code></pre>
<p>When viewing S3 methods, you can use <code>getS3method()</code>。<code>methods()</code> The output's asterisk means the method is invisible. <code>getAnywhere()</code> You can search for objects and their defined location.</p>
<pre><code class="language-r">getS3method(&quot;mean&quot;, &quot;default&quot;)

methods(&quot;predict&quot;)
getAnywhere(&quot;predict.Arima&quot;)
</code></pre>
<p>If the core of the function is achieved by C or Fortran, the printing R function usually only sees the containment layer that calls the compiler code. The source code of the R or the corresponding package needs to be viewed while continuing the tracking.</p>
<h2>Common Internal Functions</h2>
<h3>Math Functions</h3>
<pre><code class="language-r">abs(x)             # 绝对值
acos(x)            # 反余弦
asin(x)            # 反正弦
atan(x)            # 反正切
atan2(y, x)        # 根据 x、y 坐标计算反正切
ceiling(x)         # 向上取整
floor(x)           # 向下取整
round(x, digits)   # 四舍五入
cos(x)             # 余弦
cosh(x)            # 双曲余弦
exp(x)             # 指数函数
log(x)             # 自然对数
log10(x)           # 以 10 为底的对数
logb(x, base)      # 指定底数的对数
sin(x)             # 正弦
sinh(x)            # 双曲正弦
sqrt(x)            # 平方根
tan(x)             # 正切
tanh(x)            # 双曲正切
</code></pre>
<h3>Statistical Functions</h3>
<pre><code class="language-r">mean(x)             # 均值
median(x)           # 中位数
sum(x)              # 总和
min(x)              # 最小值
max(x)              # 最大值
range(x)            # 返回最小值和最大值
diff(range(x))      # 极差
diff(x)             # 相邻元素之差
prod(x)             # 连乘
var(x)              # 样本方差
sd(x)               # 样本标准差
cor(x, y)           # 相关系数
cov(x, y)           # 协方差
quantile(x, probs)  # 分位数

t.test(x, y)
chisq.test(x)
cor.test(x, y)
</code></pre>
<h3>Probability distribution function</h3>
<p>R Use uniform naming for common distributions. Add abbreviations before distribution <code>d</code>、<code>p</code>、<code>q</code> or <code>r</code>, obtains a probability density or probability mass, cumulative distribution, fraction and random number functions, respectively. Only part of the distribution is achieved, such as multiple distributions <code>dmultinom()</code> and <code>rmultinom()</code>But there's no match. <code>p*()</code> and <code>q*()</code> function.</p>
<table>
<thead>
<tr>
<th>Distribution</th>
<th>R Abbreviations</th>
<th>Common parameters</th>
</tr>
</thead>
<tbody><tr>
<td>Beta</td>
<td><code>beta</code></td>
<td><code>shape1</code>, <code>shape2</code></td>
</tr>
<tr>
<td>Two</td>
<td><code>binom</code></td>
<td><code>size</code>, <code>prob</code></td>
</tr>
<tr>
<td>Cauchy</td>
<td><code>cauchy</code></td>
<td><code>location</code>, <code>scale</code></td>
</tr>
<tr>
<td>Index</td>
<td><code>exp</code></td>
<td><code>rate</code></td>
</tr>
<tr>
<td>Kafane.</td>
<td><code>chisq</code></td>
<td><code>df</code>, <code>ncp</code></td>
</tr>
<tr>
<td>F</td>
<td><code>f</code></td>
<td><code>df1</code>, <code>df2</code>, <code>ncp</code></td>
</tr>
<tr>
<td>Gamma</td>
<td><code>gamma</code></td>
<td><code>shape</code>, <code>rate</code> or <code>scale</code></td>
</tr>
<tr>
<td>Geometry</td>
<td><code>geom</code></td>
<td><code>prob</code></td>
</tr>
<tr>
<td>Super Geometry</td>
<td><code>hyper</code></td>
<td><code>m</code>, <code>n</code>, <code>k</code></td>
</tr>
<tr>
<td>logarithmic normal</td>
<td><code>lnorm</code></td>
<td><code>meanlog</code>, <code>sdlog</code></td>
</tr>
<tr>
<td>Logistic</td>
<td><code>logis</code></td>
<td><code>location</code>, <code>scale</code></td>
</tr>
<tr>
<td>Multiple</td>
<td><code>multinom</code></td>
<td><code>size</code>, <code>prob</code></td>
</tr>
<tr>
<td>Normal</td>
<td><code>norm</code></td>
<td><code>mean</code>, <code>sd</code></td>
</tr>
<tr>
<td>Negative 2</td>
<td><code>nbinom</code></td>
<td><code>size</code>, <code>prob</code> or <code>mu</code></td>
</tr>
<tr>
<td>Poisson</td>
<td><code>pois</code></td>
<td><code>lambda</code></td>
</tr>
<tr>
<td>Student t</td>
<td><code>t</code></td>
<td><code>df</code></td>
</tr>
</tbody></table>
<p>For example:</p>
<pre><code class="language-r">dnorm(0)                    # x = 0 处的密度
pnorm(1.96)                 # P(X &lt;= 1.96)
qnorm(0.975)                # 97.5% 分位数
rnorm(100, mean = 0, sd = 1)  # 生成 100 个随机数
</code></pre>
<p>Continuously distributed <code>d*()</code> Return density, scattered <code>d*()</code> Returns the probability quality.<code>r*()</code> The first parameter is usually a random number to generate.<code>p*()</code> and <code>q*()</code> Common <code>lower.tail</code> Control left or right end of calculation. Use <code>log.p</code> Controls whether logarithmic probability is used.</p>
<pre><code class="language-r">pnorm(1.96, lower.tail = FALSE)
qnorm(log(0.025), log.p = TRUE)
</code></pre>
<h3>Sample and grouping</h3>
<p><code>sample()</code> Sample from vector.<code>size</code> It's the extraction amount.<code>replace</code> Whether or not to put it back,<code>prob</code> You can specify the sampling weight corresponding to one element.</p>
<pre><code class="language-r">sample(x, size = 5)
sample(x, size = 5, replace = TRUE)
sample(x, size = 5, replace = TRUE, prob = weights)
</code></pre>
<p>Arrange group calculations that can be used <code>factorial()</code> and <code>choose()</code>。</p>
<pre><code class="language-r">factorial(5)
choose(52, 4)

# 从 52 个对象中依次取 4 个且不放回的排列数
prod(52:49)
</code></pre>
<h3>Character processing function</h3>
<pre><code class="language-r">nchar(x)                          # 字符数
substr(x, start, stop)            # 提取或替换子串
substring(x, first, last)         # 向量化的子串操作
paste(..., sep = &quot; &quot;, collapse = NULL)
paste0(..., collapse = NULL)
sprintf(format, ...)              # 格式化字符串
toupper(x)
tolower(x)
trimws(x)                         # 去除两端空白
strtrim(x, width)                 # 截断到指定显示宽度
cat(..., sep = &quot;&quot;, file = &quot;&quot;)    # 连接并输出
gsub(pattern, replacement, x)     # 替换全部匹配
sub(pattern, replacement, x)      # 替换首个匹配
chartr(old, new, x)               # 逐字符替换
strsplit(x, split, fixed = FALSE)
grep(pattern, x, value = FALSE)
regexpr(pattern, text)
gregexpr(pattern, text)
</code></pre>
<h3>Apply function to matrix and data box</h3>
<p><code>apply()</code> Call a function along the specified dimensions of the matrix or array.<code>MARGIN = 1</code> This means yes, yes.<code>MARGIN = 2</code> indicates the column.</p>
<pre><code class="language-r">apply(x, MARGIN, FUN, ...)

apply(matrix_data, 1, mean)
apply(matrix_data, 2, sd)
</code></pre>
<p>Use for mixed data frames <code>apply()</code> , the data box may first be converted into a matrix, resulting in a uniform column type. When processing data boxes by columns,<code>lapply()</code> or <code>vapply()</code> Usually more appropriate.</p>
<pre><code class="language-r">lapply(dataframe, class)
vapply(dataframe, is.numeric, logical(1))
</code></pre>
<p><code>aggregate()</code> Summarizes the numerical columns by one or more grouping variables.</p>
<pre><code class="language-r">aggregate(x, by, FUN, ...)

aggregate(
  reaction_rate ~ state,
  data = Puromycin,
  FUN = mean
)
</code></pre>
<p>Formula interface will be pressed <code>state</code> Group, calculate each group <code>reaction_rate</code> average. Returns value remains a data box that allows continued connection or drawing.</p>
