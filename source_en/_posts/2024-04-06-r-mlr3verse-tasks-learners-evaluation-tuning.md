---
title: 'R mlr3verse: Tasks, Learners, Evaluation, and Tuning'
title_zh: R mlr3verse：任务、学习器、评估与调参
date: 2024-04-06 20:28:48 +0800
categories:
- Programming
- R
tags:
- R
- Machine Learning
author: Hyacehila
mathjax: true
hidden: true
excerpt: Covers tasks, learners, prediction, evaluation, resampling, benchmarking, tuning, and classification metrics.
description: Covers tasks, learners, prediction, evaluation, resampling, benchmarking, tuning, and classification metrics.
excerpt_zh: 整理 mlr3 的任务、学习器、预测、评估、重抽样、基准实验、调参与分类评估等内容。
permalink: /blog/2024/04/06/r-mlr3verse-learning-notes/
lang: en
translation_key: 2024-04-06-r-mlr3verse-tasks-learners-evaluation-tuning
translation_status: machine
translation_source_hash: 620774fba271aeb94cfbb9e5f873163bbb95357f07bed8a98cfafe50b480632a
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<h2>Preparatory knowledge and overview</h2>
<h3>mlr3</h3>
<p><code>mlr3</code>  Packages and broader <code>mlr3verse</code> A common, object-oriented and scalable framework is provided for R language return, classification and other machine learning tasks.</p>
<p>At the most basic level, the unified interface provides training, testing and evaluation of algorithms for many machines. It could go further by over-parameter optimization, computing conduits, model interpretation, etc.</p>
<p><code>mlr3</code> and <code>scikit-learn</code>  <code>caret</code>  <code>tidymodels</code> With similar overall goals</p>
<p><code>mlr3</code> It is designed to provide greater flexibility than any other ML framework. <code>mlr3</code> A simple method of using advanced functionality is still available. Although... <code>tidymodels</code> In particular, it makes it easier to carry out simple ML tasks. <code>mlr3</code> But it's better for senior ML.</p>
<p><strong><code>mlr3</code> The data box should be secured.</strong></p>
<h3>Examples</h3>
<p>Now, we're going to use two examples of what we're looking at later.</p>
<p>A simple decision tree.</p>
<pre><code class="language-r">## 训练
library(mlr3)
task = tsk(&quot;penguins&quot;)
split = partition(task)
learner = lrn(&quot;classif.rpart&quot;)

learner&#36;train(task, row_ids = split&#36;train)
learner&#36;model

## 预测
prediction = learner&#36;predict(task, row_ids = split&#36;test)
prediction

## 评估
prediction&#36;score(msr(&quot;classif.acc&quot;))
</code></pre>
<p>More complex examples</p>
<pre><code class="language-r">library(mlr3verse)

tasks = tsks(c(&quot;breast_cancer&quot;, &quot;sonar&quot;))

glrn_rf_tuned = as_learner(ppl(&quot;robustify&quot;) %&gt;&gt;% auto_tuner(
    tnr(&quot;grid_search&quot;, resolution = 5),
    lrn(&quot;classif.ranger&quot;, num.trees = to_tune(200, 500)),
    rsmp(&quot;holdout&quot;)
))
glrn_rf_tuned&#36;id = &quot;RF&quot;

glrn_stack = as_learner(ppl(&quot;robustify&quot;) %&gt;&gt;% ppl(&quot;stacking&quot;,
    lrns(c(&quot;classif.rpart&quot;, &quot;classif.kknn&quot;)),
    lrn(&quot;classif.log_reg&quot;)
))
glrn_stack&#36;id = &quot;Stack&quot;

learners = c(glrn_rf_tuned, glrn_stack)
bmr = benchmark(benchmark_grid(tasks, learners, rsmp(&quot;cv&quot;, folds = 3)))

bmr&#36;aggregate(msr(&quot;classif.acc&quot;))
</code></pre>
<h3>2 required packages</h3>
<p>We need to add software to R in order to make the ML more objects-oriented. Package<code>R6</code>
Meanwhile, in order to process a lot of data more efficiently, we introduced<code>data.frame</code>Improvements<code>data.table</code></p>
<h4>R6</h4>
<p><code>R6</code> It's one of the most recent examples of object-oriented programming, and it's identical to the other object-oriented languages.</p>
<p><code>&#36;new()</code> It's an initialization method for creating R6-type objects.</p>
<pre><code class="language-r">foo = Foo&#36;new(bar = 1)
</code></pre>
<p>We'll use it.<code>Foo</code>Class created one.<code>foo</code>Object and set parameters
Yes.<code>mlr3 </code>We've made a lot of progress in creating what we need.</p>
<p>And for objects that have a variable state, we also offer a way to modify it.<code>&#36;</code></p>
<pre><code class="language-r">foo&#36;bar = 2
</code></pre>
<p>We visited the variable state of the object and modified it.</p>
<p>Besides, we certainly have a method (methods) that allows users to check the status of objects, retrieve information or perform changes to the internal state of objects.
For example, for learning machines <code>&#36;train()</code> The method is to modify the internal state of the learning device. <code>R6</code>The object, of course, has a way to give something else, and the way back is quite free, and we can, of course, enter multiple ways at once, and then call in, like,</p>
<pre><code class="language-r">Foo&#36;bar()&#36;hello_world()
</code></pre>
<p>Object First<code>Foo </code>It's done. <code>bar</code> And then we run the way they return. <code>hello_world</code></p>
<p>Finally. <code>R6</code>The target.<code>environments</code> The direct value is quoted instead of creating a new object as follows:</p>
<pre><code class="language-r">foo2 = foo
foo2 = foo&#36;clone(deep = TRUE)
</code></pre>
<p>The former does not create new objects, but quotes.<code>foo</code> Yeah.<code>foo2</code>The change is the same as the direct pair.<code>foo</code>Modify
Other Organiser<code>clone</code>Method See Article 2</p>
<h4>data.table</h4>
<p>It's right.<code>R</code>Medium<code>data.frame</code> And that's why we're here.<code>mlr3</code>Use it.</p>
<p>He's on basic grammar rules.<code>data.frame</code>There's no difference. It's an extension, but it's based entirely on the syntax rule of R.</p>
<p><code>data.table</code> Also use semantics of quotes, which require use <code>copy()</code> Cloning <code>data.table</code></p>
<h3>Practical applications</h3>
<p><code>mlr3</code> Including some important practical procedures that are essential to simplify codes in the mlr3 ecosystem</p>
<h4>Sugar Functions Sugar</h4>
<p><code>mlr3</code> Most objects can be created by a facilitative function called a auxiliary function or a sugar function. They provide a shortcut to common code customary terms and reduce the number of codes that users must prepare. For example: <code>lrn(&quot;regr.rpart&quot;)</code> , return the learner without creating a new R6 object</p>
<p>SugarFunctions is designed to cover most users, and complete only when building custom objects or extensions <code>R6</code> Backend knowledge</p>
<p><code>mlr3 </code>to be standardized according to agreement  <code>mlr_&lt;type&gt;_&lt;key&gt;</code></p>
<h4>Dictionaries</h4>
<p><code>mlr3</code> Use the dictionary to store values in the R6 category dictionaries usually accessed through the sugar function where objects are retrieved from the dictionaries
For example: <code>lrn(&quot;regr.rpart&quot;)</code> It's...  <code>mlr_learners&#36;get(&quot;regr.rpart&quot;)</code>Packaging, so from <code>mlr_learners</code> A simpler way to load the decision tree learner</p>
<p>The dictionary groupes a large number of clusters of objects so that they can be easily listed and retrieved, for example.
<code>as.data.table(mlr_learners)  lrn() </code>You can view available learners in loaded packages</p>
<h4>mlr3viz</h4>
<p><code>mlr3viz</code> Including all the drawing functions of mlr3 and some of the ggpllot2 functions used.
We use <code>theme_minimal()</code> To unify our aesthetics, but with all <code>ggplot</code> As with output, users can be completely customised It's...</p>
<p><code>mlr3viz</code> Extension <code>fortify</code> and <code>autoplot</code> They're used for common use. <code>mlr3</code> Output, including <code>Prediction</code> 、 <code>Learner</code> and <code>BenchmarkResult</code> Object</p>
<p>I understand. <code>mlr3viz</code> The best way to do this is through experiments; load the bag, see where <code>mlr3</code> Run on objects <code>autoplot</code> What happens sometimes?
The drawing type is recorded on the corresponding manual page and can be accessed through <code>?autoplot.&lt;class&gt;</code> , for example, by running <code>?autoplot.TaskRegr</code> to find different types of drawings for returning tasks</p>
<h3>Design principles</h3>
<p>Here's what the developers are saying.<code>mlr3</code>The principles of design, knowledge of them, help us develop better code habits and understand.<code>mlr3</code>Basic composition logic</p>
<ul>
<li>Object-oriented programming: we embrace <code>R6</code> Clean, object-oriented design, object status changes and references</li>
<li>Table data: Embrace <code>data.table</code> Its first-class computing performance and table data as a structure that can easily be further processed</li>
<li>Uniform table input and output data formats: greatly simplified API</li>
<li>Defense programming and type security: all user input is used <code>checkmate</code>  Inspection</li>
<li>Reduction of dependency relationships: main <code>mlr</code> One of the maintenance burdens is to keep up with the changing learning tool interface and the many software packages on which it relies. What we need. <code>mlr3</code> The significantly fewer software packages make installation and maintenance easier. We still provide the same function, but it's broken down into more packages with fewer dependencies.</li>
<li>Calculates and indicates separation. <code>mlr3</code> Most of the ecosystem packages focus on processing and converting data, applying ML algorithms and computations, and visualization of data and results <code>mlr3viz</code> Available</li>
</ul>
<h2>Data and Basic Modelling</h2>
<p>In this chapter, we will present the basic building blocks of machine learning. <code>mlr3</code> Object and Correspond <code>R6</code> Class. These building blocks include data (and methods for creating training and testing sets), machine learning algorithms (and their training and prediction processes), machine learning algorithms through over-parametric configuration and assessment measures to assess the quality of projections</p>
<p>This chapter will be ours.<code>mlr3verse</code>The rest of this book will be based on the basic elements seen in this chapter.</p>
<p>Our presentation will begin with the fundamentals of return and then gradually expand to the classification issue, which is the core of our monitoring studies.</p>
<h3>Tasks</h3>
<p>The task is to include the object of data (usually tables) and additional metadata (metala); the additional metadata contains the name of the target characteristic for monitoring machine learning
The information is automatically extracted when required, so users do not have to specify the target for each training model</p>
<h4>Build Task</h4>
<p><code>mlr3</code> Yes. <code>mlr_tasks</code>  <code>Dictionary</code> Include some predefined machine learning tasks
To get jobs from the dictionary, use <code>tsk()</code> Function and return value to new variable
Run without any arguments <code>tsk()</code> Lists all the jobs in the dictionary, which also applies to other sugar functions</p>
<pre><code class="language-r">#查看字典中存储的预先定义的任务
mlr_tasks

#建立一个任务 我们是从字典中提取的
tsk_mtcars = tsk(&quot;mtcars&quot;)
#打印任务简报
tsk_mtcars

#查看字典中存储的预先定义的任务
tsk()
</code></pre>
<p>To create your own return task, you need to construct a new one <code>TaskRegr</code> Example. The simplest way to use a function <code>as_task_regr()</code>Will <code>data.frame</code> Type of object converted to return task by passing it to <code>target</code> Parameters to specify target characteristics.<code>mtcars</code>Dataset</p>
<pre><code class="language-r">#引入数据 提取其中的一部分作为我们需要的数据集 然后简单展示
data(&quot;mtcars&quot;, package = &quot;datasets&quot;)
mtcars_subset = subset(mtcars, select = c(&quot;mpg&quot;, &quot;cyl&quot;, &quot;disp&quot;))
str(mtcars_subset)

## 用数据集建立一个回归任务 其中预测目标是 &#39;mpg&#39; id 是任务的简述 后面绘图会用它称呼我们的任务
tsk_mtcars = as_task_regr(mtcars_subset, target = &quot;mpg&quot;, id = &quot;cars&quot;)
</code></pre>
<p><code>as_task_regr()</code> Various types of data boxes are acceptable, including<code>data.frame data.table tibble</code> It's very compatible.</p>
<p>Special <code>as_task_regr()</code> In many cases, the UTF-8 code name is not accepted.</p>
<p>For the mission, we can use it.<code>mlr3viz</code>It fits directly with the mlr3 object, and it's all automatic.</p>
<pre><code class="language-r">library(mlr3viz)
autoplot(tsk_mtcars, type = &quot;pairs&quot;)
</code></pre>
<h4>Retrieving data</h4>
<p>We've learned how to create jobs to store data and metadata, and now we'll know how to retrieve stored data.</p>
<p>You can use a variety of fields to retrieve metadata about tasks like</p>
<ul>
<li>The names of the functional and target columns are stored separately in <code>&#36;feature_names</code> and <code>&#36;target_names</code> Inside.</li>
<li>Available <code>&#36;nrow</code> and <code>&#36;ncol</code> Retrieval dimensions</li>
<li>Task column has only one <code>character</code> value, the row is marked by the only natural number (known as line ID). Yes. <code>&#36;row_ids</code> Field Access</li>
<li>The data contained in the task can be passed <code>&#36;data()</code> Visit, it returns one. <code>data.table</code> object. There are options. <code>rows</code> and <code>cols</code> Parameters to specify a subset of data to be retrieved</li>
</ul>
<pre><code class="language-r">## 回报维数 也就是行数（观测数）列数（特征数）
c(tsk_mtcars&#36;nrow, tsk_mtcars&#36;ncol)

#回报功能列和目标列名称
c(Features = tsk_mtcars&#36;feature_names,
  Target = tsk_mtcars&#36;target_names)

#回报行ID head用于限定返回前几个元素（默认6个）
head(tsk_mtcars&#36;row_ids)
</code></pre>
<p>When we filter some lines, the lines change, but the lines don't change.</p>
<pre><code class="language-r">task = as_task_regr(data.frame(x = runif(5), y = runif(5)),
  target = &quot;y&quot;)
task&#36;row_ids
## 返回 1 2 3 4 5
task&#36;filter(c(4, 1, 3))
task&#36;row_ids
## 返回 1 3 4
</code></pre>
<p>The data contained in the task can be passed <code>&#36;data()</code> Visit, it returns one. <code>data.table</code> object. There are options. <code>rows</code> and <code>cols</code> Parameters to specify a subset of data to be retrieved
<strong>Default lines are searched with line ID</strong></p>
<pre><code class="language-r">## 返回全部数据子集
tsk_mtcars&#36;data()

## 根据行ID以及列名称返回数据子集
tsk_mtcars&#36;data(rows = c(1, 5, 10), cols = tsk_mtcars&#36;feature_names)

## 这就起到了用行号检索数据子集的效果
tsk_mtcars&#36;data(rows = task&#36;row_ids[2])
</code></pre>
<h4>Task changes</h4>
<p>It works by modifying the mission after it was created.<code>&#36;data()</code>The difference is... <strong>It changed the mission directly.</strong></p>
<p>Use <code>&#36;select()</code> Sub-assembly by feature (column), with the desired feature name transmitted as a character vector, using <code>&#36;filter()</code>Observation sub-unitation by using line ID as a digital vector
<strong>These methods directly modified the mission.</strong>
Because of the R6 quote, if you want to keep the original job at the same time, you need to <code>&#36;clone()</code> <a href="/en/blog/2024/04/06/r-mlr3verse-learning-notes/">R6 Example of semantic references</a></p>
<pre><code class="language-r">## 创建任务
tsk_mtcars_small = tsk(&quot;mtcars&quot;)
## 只选取一个列（特征） 它不可以移除target
tsk_mtcars_small&#36;select(&quot;cyl&quot;)
## 选取下面的行
tsk_mtcars_small&#36;filter(2:3)
</code></pre>
<p>To add additional rows and columns to the task, you can use them separately. <code>&#36;rbind()</code> and <code>&#36;cbind()</code></p>
<pre><code class="language-r">tsk_mtcars_small&#36;cbind(
  data.frame(disp = c(150, 160))
)

tsk_mtcars_small&#36;rbind(
  data.frame(mpg = 23, cyl = 5, disp = 170)
)
</code></pre>
<h3>Learner Learners</h3>
<h4>Introduction to the learner</h4>
<p>Category <code>Learner</code> The object provides a unified interface for many commonly used machine learning algorithms in R. <code>mlr_learners</code> Dictionary contains <code>mlr3</code>. We will discuss the available learning tools later; Now we're going to use the tree-returner as an example. <code>Learner</code>Interface. As with tasks, you can use a single sugar function to access learners from the dictionary, as in this case <code>lrn()</code></p>
<pre><code class="language-r">## 查看可用的学习器
mlr_learners
lrn()

#从字典中建立一个学习器 作学习器的简要报告（因为没存储）
lrn(&quot;regr.rpart&quot;)
</code></pre>
<p>The learning device is at the core of the algorithm and, unlike the mission, the learning device has nothing to do with the training data, and in most cases we only have to call on the already existing learning device, and we only need to customize the new learning device when we design the new algorithm from the bottom, so it's enough to use it from the dictionary.</p>
<p>All <code>Learner</code> Object contains the following metadata, which can be seen in the output of the summary report of the learner</p>
<ul>
<li><code>&#36;feature_types</code> : Characteristic types that can be handled by learners</li>
<li><code>&#36;packages</code> : Use the software that the learners need to install Package</li>
<li><code>&#36;properties</code> : learner attributes, for example, “missings” properties mean that models can process lost data, while “importance” means that it can calculate the relative importance of each characteristic</li>
<li><code>&#36;predict_types</code> : Type of prediction that the model can make</li>
<li><code>&#36;param_set</code> : Available Super Parameter Set</li>
</ul>
<p>A full-fledged machine learning experiment, with learners going through two stages:</p>
<ul>
<li>Training: training <code>Task</code> Passed to learners <code>&#36;train()</code> function, which trains and stores models, i.e. learning relationships between features and targets</li>
<li>Forecast: New data, probably different divisions of the original data set, passed to trained learners <code>&#36;predict()</code> Methodology for projecting target values</li>
</ul>
<p><strong>The method of training and forecasting is a learning device, not a mission.</strong></p>
<h4>Training</h4>
<p>By Usage <code>&#36;train()</code> The method is to transfer the task to the learner to train the model.<code>&#36;model</code>Medium</p>
<pre><code class="language-r">## load mtcars task
tsk_mtcars = tsk(&quot;mtcars&quot;)
## load a regression tree
lrn_rpart = lrn(&quot;regr.rpart&quot;)
## 训练学习器 任务和学习器均已经给出
lrn_rpart&#36;train(tsk_mtcars)
## 查看训练出的模型 具体怎么理解可以参考模型的help 我们的帮助是可以对对象使用的
lrn_rpart&#36;model
</code></pre>
<p>In many cases, we want not to use all the raw data to train models, and here we're going to introduce a simple division that corresponds to what we're talking about in machine learning theory, machine learning introductory and supervisory learning: setting aside.</p>
<p><code>partition()</code> Function creates index set, which randomly divides tasks into two discrete sets: training set (67 per cent of total data by default) and testing set (remaining data)</p>
<pre><code class="language-r">## 划分数据 返回的是两个集合
splits = partition(tsk_mtcars)
splits

## 借助前面的划分来实现 row_ids让我们可以选择一部分行号进行训练
lrn_rpart&#36;train(tsk_mtcars, row_ids = splits&#36;train)
</code></pre>
<h4>Projections</h4>
<p>The prediction from a trained model is like data. <code>Task</code> Pass it to the training. <code>Learner</code>Yes. <code>&#36;predict()</code> It's as simple as that.</p>
<h5>Foundation projections</h5>
<p>Because projections also require data, and according to machine learning habits, training and testing data will be combined.<code>tasks</code>That's why we moved in.<code>tasks</code> Specify the forecast line number</p>
<pre><code class="language-r">prediction = lrn_rpart&#36;predict(tsk_mtcars, row_ids = splits&#36;test)
</code></pre>
<p><code>&#36;predict()</code> Method returns one from <code>Prediction</code> The object of succession will vary according to the object of our mission.<code>row_ids</code> Column corresponds to the line ID of the predicted observation. <code>truth</code>Column contains real data taken from the task by the object (if available)<code>response</code> Column contains values projected by the model</p>
<p>Use <code>as.data.table()</code> The function can easily be <code>Prediction</code> Convert Object to <code>data.table</code> or <code>data.frame</code> Object</p>
<h5>Special treatment of forecasted objects</h5>
<p>As<code>mlr3verse</code>It also has its own special ways of doing things.</p>
<pre><code class="language-r">## 直接访问 如果访问的不是很多没必要调用数据框相关函数 语法也非常的自然
prediction&#36;response[1:2]

#`mlr3viz` 为 所有继承`Prediction`类的对象提供了一个 `autoplot()` 方法
library(mlr3viz)
prediction = lrn_rpart&#36;predict(tsk_mtcars, splits&#36;test)
autoplot(prediction)
</code></pre>
<h5>Forecast new data</h5>
<p>The machine learns the habit of bringing all the data together into the tasks, but sometimes we do have a need to predict some new data, and at this point there's no need to re-establish the task. mlr3 takes this into account. <code>&#36;predict_newdata()</code> That's it.</p>
<pre><code class="language-r">mtcars_new = data.table(cyl = c(5, 6), disp = c(100, 120),
  hp = c(100, 150), drat = c(4, 3.9), wt = c(3.8, 4.1),
  qsec = c(18, 19.5), vs = c(1, 0), am = c(1, 1),
  gear = c(6, 4), carb = c(3, 5))
prediction = lrn_rpart&#36;predict_newdata(mtcars_new)
prediction
</code></pre>
<p>At this point,<code>truth</code>It's all empty.<code>row_ids</code>Continue New Data Box
<strong>Note that when predicting new data, make sure the column name and<code>tasks</code>It's the same thing.</strong></p>
<h5>Change projection type</h5>
<p>While individual values are the most common projection type in regression, they are not the only projection type. Some regression models can also give the prediction standard error at the same time as we do in the traditional linear regression.</p>
<p>In order to predict this, before training, we have to... <code>LearnerRegr</code> Yes. <code>&#36;predict_type</code> Field changed from 'response' (default value) to <code>&quot;se&quot;</code></p>
<p>We use it up there. <code>&quot;rpart&quot;</code> The learner does not support predicting standard errors, so in the example below we will use linear regression models<code>lrn(&quot;regr.lm&quot;)</code></p>
<pre><code class="language-r">## 导入需要的包并且建立新的学习器
library(mlr3learners)
lrn_lm = lrn(&quot;regr.lm&quot;, predict_type = &quot;se&quot;)
## 训练与预测
lrn_lm&#36;train(tsk_mtcars, splits&#36;train)
lrn_lm&#36;predict(tsk_mtcars, splits&#36;test)
</code></pre>
<h4>Hyperparameters</h4>
<p><code>Learners</code> A machine learning algorithm and its super-parameters are encapsulated and can be set by the user.</p>
<p>Superparameters may affect the mode of training or prediction of models and may require expertise to determine how to set them</p>
<p>We're going to be able to optimise the superparameters, and we're going to be able to talk about how to optimise them automatically, but in this chapter, we're going to focus on how to set them manually, and that's the basis for setting auto-optimizations behind us, and in actual machine learning, super-parameters are often manually set.</p>
<h5>Parameters and Arguments Set</h5>
<p>We've explained before that the learning machine should be used.<code>&#36;param_set</code>Visits <a href="/en/blog/2024/04/06/r-mlr3verse-learning-notes/">A description of the learner's hyperparameter</a>  As follows:</p>
<pre><code class="language-r">lrn_rpart&#36;param_set
</code></pre>
<p>The output is one. <code>ParamSet</code> object, by <code>paradox</code> Package provided. These objects provide information about the super-parameters, including their names. <code>id</code> Type of data () <code>class</code> ), technical validity range of ultra-parameter values () <code>lower</code> 、 <code>upper</code> ) possible levels if the type of data is classified ( <code>nlevels</code> Default value from bottom package ( <code>default</code> ) and last set value ( <code>value</code> ）</p>
<p><code>class</code> Inheritance is <code>paradox</code> the categories in which the parameters are determined and the possible values that it is possible to use.</p>
<table>
<thead>
<tr>
<th>Hyperparameter Class</th>
<th>Hyperparameter Type</th>
</tr>
</thead>
<tbody><tr>
<td><a href="https://paradox.mlr-org.com/reference/ParamDbl.html"><code>ParamDbl</code></a></td>
<td>Real value (value)</td>
</tr>
<tr>
<td><a href="https://paradox.mlr-org.com/reference/ParamInt.html"><code>ParamInt</code></a></td>
<td>Integer</td>
</tr>
<tr>
<td><a href="https://paradox.mlr-org.com/reference/ParamFct.html"><code>ParamFct</code></a></td>
<td>Factor</td>
</tr>
<tr>
<td><a href="https://paradox.mlr-org.com/reference/ParamLgl.html"><code>ParamLgl</code></a></td>
<td>Boolean Type (T or F)</td>
</tr>
<tr>
<td><a href="https://paradox.mlr-org.com/reference/ParamUty.html"><code>ParamUty</code></a></td>
<td>No type</td>
</tr>
</tbody></table>
<p>In most cases, the super-parameters are properly initialized into the defaults that they should have, but in some cases they can be misinformed, i.e., when creating the learning device, the super-parameters are not in the situation that they should be, and at this point, they usually give a hint.<code>bug</code>It's a development that avoids it as much as possible, but not entirely.</p>
<h5>Fetch and set hyperparameter values</h5>
<p>Now that we know how the super-parameter sets are stored, we can think about getting and setting them. Back to our decision-making tree, assuming we're interested in growing a deep one, <code>1</code> and the tree, also known as the decision stake, in which the data are divided into only two terminal nodes</p>
<p>There are several different ways to change this parameter. The simplest way is to transmit the name and new value of the super-parameter during the construction of the learning device. Here. <code>lrn()</code> Like</p>
<pre><code class="language-r">## 建立学习器的时候设置超参数
lrn_rpart = lrn(&quot;regr.rpart&quot;, maxdepth = 1)

## 返回那些非默认超参数的列表 本质上就是在超参数集上多了一层访问
lrn_rpart&#36;param_set&#36;values
</code></pre>
<p><strong>Manual setting of super-parameters directly when constructing a learning device is the most practical method</strong></p>
<p>Just now, we've introduced another way to access the super-parameter assembly. <code>&#36;value</code>Start to modify the hyperparameter</p>
<pre><code class="language-r">lrn_rpart&#36;param_set&#36;values&#36;maxdepth = 2
lrn_rpart&#36;param_set&#36;values
</code></pre>
<p>There's only one thing we can do about it at a time.</p>
<pre><code class="language-r">lrn_rpart&#36;param_set&#36;set_values(xval = 2, cp = 0.5)
lrn_rpart&#36;param_set&#36;values
</code></pre>
<p><code>lrn_rpart&#36;param_set&#36;values</code> Return one <code>list</code> But don't use new ones. <code>list</code>And the way to change the parametrics, he'll cause some of them to be erased, because we're always building them. <code>list</code>, and only include the hyperparameters you want to modify.</p>
<p><strong>All modifications to the hyperparameters are subject to relevant cross-border checks to ensure the type of compliance.</strong></p>
<h5>Superparameter Dependence</h5>
<p>More complex hyper-parameter space may contain dependency relationships, which occurs when the setting of one super-parameter is conditional on the value of another;
An example of this is support for vector machines. <code>lrn(&quot;regr.svm&quot;)</code> I don't know. Fields <code>&#36;deps</code> Return one <code>data.table</code> It's listed. <code>Learner</code> Overparameter dependency in</p>
<pre><code class="language-r">lrn(&quot;regr.svm&quot;)&#36;param_set&#36;deps
</code></pre>
<p>of which <code>id</code> The column indicates who depends on other super-parameters <code>on</code> The column tells us who's dependent. <code>cond</code> Column tells us what the deal is.</p>
<pre><code class="language-r">#访问cond列内容
lrn(&quot;regr.svm&quot;)&#36;param_set&#36;deps[[1, &quot;cond&quot;]]
lrn(&quot;regr.svm&quot;)&#36;param_set&#36;deps[[3, &quot;cond&quot;]]
</code></pre>
<p><code>CondAnyOf</code> Meaning<code>on</code>It's one of the numbers in the pool.
<code>CondEqual</code> Meaning<code>on</code> Equals to a value
It means our condition.</p>
<p>If the conditions for the relevant superparameters are not met, then <code>Learner</code> Error</p>
<h4>Benchmark learners</h4>
<p>Before we continue with the learning machine assessment, we will highlight an important learning tool. These are very simple or “weak” learners, referred to as benchmarks;</p>
<p>For the return, we have achieved the benchmark. <code>lrn(&quot;regr.featureless&quot;)</code> , it always predicts that the new value is the average (or median) of the target in the training data if <code>robust</code> Set Hyperparameter to <code>TRUE</code> ）</p>
<p>If a model works worse than a benchmark learner, then it's a bad model.</p>
<pre><code class="language-r">df = as_task_regr(data.frame(x = runif(1000), y = rnorm(1000, 2, 1)),
  target = &quot;y&quot;)
lrn(&quot;regr.featureless&quot;)&#36;train(df, 1:995)&#36;predict(df, 996:1000)
</code></pre>
<h3>Evaluation</h3>
<p>Perhaps the most important step in the application machine learning workflow is to assess model performance. Without that, we will not know whether our training models can make very accurate predictions, whether they are worse than random speculation, or whether they are in between; and here is an example of our code, which can also be seen as a review of some of the previous code contents.</p>
<pre><code class="language-r">lrn_rpart = lrn(&quot;regr.rpart&quot;)
tsk_mtcars = tsk(&quot;mtcars&quot;)
splits = partition(tsk_mtcars)
lrn_rpart&#36;train(tsk_mtcars, splits&#36;train)
prediction = lrn_rpart&#36;predict(tsk_mtcars, splits&#36;test)
</code></pre>
<h4>Evaluator</h4>
<p>The quality of projections is assessed using measures that compare them with real data on monitoring learning assignments
and <code>Tasks</code> and <code>Learners</code>Like, <code>mlr3</code> . The available measures are stored in the name <code>mlr_measures</code> , and can be used <code>msr()</code> Visits</p>
<pre><code class="language-r">## 访问评价器的字典
mlr_measures
msr()
</code></pre>
<p>Because the idea of evaluating a model is often fixed, our evaluators rarely need to create new ones.</p>
<p><code>mlr3</code> All measures achieved are defined mainly by three components</p>
<ul>
<li>Function to measure</li>
<li>Whether lower or higher values are considered “good”</li>
<li>Scope of possible values for measurement
Besides this, an evaluator has some metadata like</li>
<li>Measure any special properties</li>
<li>Type of projection that measures can assess</li>
<li>Measurement with any "control parameters"</li>
</ul>
<p>If you look directly at an evaluator, you can get all the data you need.</p>
<pre><code class="language-r">measure = msr(&quot;regr.mae&quot;)
measure
</code></pre>
<h4>Projection scores</h4>
<p>To calculate model performance, we just have to call. <code>Prediction</code> object <code>&#36;score()</code> The method and the measure we want to calculate as a single parameter conveys the facts. Let's go. <code>Prediction</code> It stores all the data we need to evaluate a model, including real values and projections. Value</p>
<pre><code class="language-r">prediction&#36;score(measure)
</code></pre>
<p>All job types have default evaluators, for example, re-entry models using average MSE as default evaluators if we don't pass in<code>&#36;score()</code> Parameters, then use the default evaluator</p>
<p><strong>The evaluator evaluates only the test data, and we focus on generalization performance rather than the intended effects, as do other tests later.</strong></p>
<p>By passing multiple ratingrs to <code>&#36;score()</code> , multiple evaluations can be counted simultaneously</p>
<pre><code class="language-r">## 同时把多个评价器给了这个变量 然后一起传入
measures = msrs(c(&quot;regr.mse&quot;, &quot;regr.mae&quot;))
prediction&#36;score(measures)
</code></pre>
<h4>Other evaluations</h4>
<p><code>mlr3</code> Measurements of the quality of modelling projections are also provided, not quantitative, but rather “meta-information” on models. These include:</p>
<ul>
<li><code>msr(&quot;time_train&quot;)</code> - Time for training models.</li>
<li><code>msr(&quot;time_predict&quot;)</code> - Time taken to predict the model</li>
<li><code>msr(&quot;time_both&quot;)</code> - The total time spent on training models and forecasting.</li>
<li><code>msr(&quot;selected_features&quot;)</code> - Number of features selected for the model only if the model has a "selected features" attribute</li>
</ul>
<p>One simple example:</p>
<pre><code class="language-r">measures = msrs(c(&quot;time_train&quot;, &quot;time_predict&quot;, &quot;time_both&quot;))
prediction&#36;score(measures, learner = lrn_rpart)
</code></pre>
<p>We put the learning device in together.<code>&#36;score()</code> It's a special attribute of the evaluator.</p>
<p>For the number of model selection features there are</p>
<pre><code class="language-r">## 查看评估器的元数据
msr_sf = msr(&quot;selected_features&quot;)
msr_sf
</code></pre>
<p>In particular, there are two of these in the metadata of this evaluator.</p>
<ul>
<li>Parameters: normalize=FALSE</li>
<li>Documents: references task, references learner, references model
That is, this evaluator has parameters that can be set up for direct reference to evaluation parameters. <a href="/en/blog/2024/04/06/r-mlr3verse-learning-notes/">Hyperparameters</a> All methods are the same.</li>
<li><code>normalize</code> Superparameter specifies whether the selected number of returns should be standardized according to the total number of features</li>
<li><code>Properties</code> Tells us that this assessor needs a mission, a learning machine, to travel together.</li>
</ul>
<p>Displays the code for using this assessor as</p>
<pre><code class="language-r">## 设置了评估器参数
msr_sf&#36;param_set&#36;values&#36;normalize = TRUE
## 调用了评估器 它需要任务和学习器
prediction&#36;score(msr_sf, task = tsk_mtcars, learner = lrn_rpart)
</code></pre>
<h3>Return Experiment</h3>
<p>Research. <code>mlr3</code> We'll suspend all the above in a short experiment to assess the quality of our projections.</p>
<p>Independent review of the code below, understanding usage, and learning to expand.</p>
<pre><code class="language-r">library(mlr3)
set.seed(349)
## 任务构建和划分 并没有自建任务
tsk_mtcars = tsk(&quot;mtcars&quot;)
splits = partition(tsk_mtcars)
## 加载学习器 这是基准学习器
lrn_featureless = lrn(&quot;regr.featureless&quot;)
## 加载学习器 这是决策树
lrn_rpart = lrn(&quot;regr.rpart&quot;, cp = 0.2, maxdepth = 5)
## 加载评估器 两种评估方法
measures = msrs(c(&quot;regr.mse&quot;, &quot;regr.mae&quot;))
## 对两个学习器训练 使用训练数据
lrn_featureless&#36;train(tsk_mtcars, splits&#36;train)
lrn_rpart&#36;train(tsk_mtcars, splits&#36;train)
## 对两个学习器预测 使用预测数据 同时对预测的结果进行评价
lrn_featureless&#36;predict(tsk_mtcars, splits&#36;test)&#36;score(measures)
lrn_rpart&#36;predict(tsk_mtcars, splits&#36;test)&#36;score(measures)
</code></pre>
<p>You'll notice that our learning tools and measurements are available. <code>&quot;regr.&quot;</code> Prefix, which is a convenient way to remind us that we are dealing with a return mission and that it is necessary to use learning devices and metrics built for a return.</p>
<p>In the next section, we'll use <code>mlr3</code> It's just a slight change to consider the classification task.</p>
<h3>Classes Classif</h3>
<p>The classification issue is a model that predicts a discrete, disaggregated target rather than a continuous, numerical volume. For example, predicting the species from the physical characteristics of penguins will be a classification problem because there is a defined group of species</p>
<p><code>mlr3</code> Ensure that the interface of all tasks is as similar (if not identical) as possible, so focus only on differences that make classification a unique machine learning issue</p>
<p>We'll start by implementing one of the<a href="/en/blog/2024/04/06/r-mlr3verse-learning-notes/">Return Experiment</a>A very similar experiment to demonstrate the similarities between regression and classification.</p>
<p>Then we will discuss the differences between tasks, learners and projections, and then the threshold, which is a method specific to classification.</p>
<h4>Classification test</h4>
<p>Here's the code.</p>
<pre><code class="language-r">library(mlr3)
set.seed(349)
## 构建任务 这里我们的任务还是直接用已有的 建立一个新的预测任务的代码与回归存在的差异后面来解释 划分了集合
tsk_penguins = tsk(&quot;penguins&quot;)
splits = partition(tsk_penguins)
## 加载一个分类学习器 它是基准学习器
lrn_featureless = lrn(&quot;classif.featureless&quot;)
## 加载一个分类学习器 还是决策树 但是是分类专用
lrn_rpart = lrn(&quot;classif.rpart&quot;, cp = 0.2, maxdepth = 5)
## 加载分类评价器
measure = msr(&quot;classif.acc&quot;)
## 训练
lrn_featureless&#36;train(tsk_penguins, splits&#36;train)
lrn_rpart&#36;train(tsk_penguins, splits&#36;train)
## 预测并评价
lrn_featureless&#36;predict(tsk_penguins, splits&#36;test)&#36;score(measure)
lrn_rpart&#36;predict(tsk_penguins, splits&#36;test)&#36;score(measure)
</code></pre>
<h4>Classification Tasks</h4>
<p>The classification task is from <code>TaskClassif</code> The object of succession, except for the target variable, which is a factor type, is very similar to a return mission.</p>
<p>Filtered <code>mlr_tasks</code> Dictionary View <code>mlr3</code> sort tasks predefined in</p>
<pre><code class="language-r">as.data.table(mlr_tasks)[task_type == &quot;classif&quot;]
</code></pre>
<p>Available <code>as_task_classif</code> Create your own category task</p>
<pre><code class="language-r">as_task_classif(palmerpenguins::penguins, target = &quot;species&quot;)
</code></pre>
<p><code>mlr3</code> In support of two types of classification tasks: dual classification, where the result could be one of two categories, and multiple classifications, where the result could be one of three or more categories</p>
<p>We can see in the mission's brief report all the relevant attributes and use the most natural habits to access them.</p>
<p>An important difference between these tasks is that the binary classification task is named <code>&#36;positive</code> , which defines the " positive " category. In the binary classification, since there are only two possible categories, as is customary, one is referred to as the “positive” category and the other as the “negative” category. Category</p>
<pre><code class="language-r">## 加载数据
data(Sonar, package = &quot;mlbench&quot;)
## 建立tasks
tsk_classif = as_task_classif(Sonar, target = &quot;Class&quot;, positive = &quot;R&quot;)
## 查看正类
tsk_classif&#36;positive
## 修改正类
tsk_classif&#36;positive = &quot;M&quot;
</code></pre>
<p>Although the choice of categories is arbitrary, they are essential to ensure that the results of models and performance indicators are interpreted as intended - as demonstrated when we discuss thresholds and ROC indicators</p>
<p>Finally, it's available. <code>autoplot.TaskClassif</code> Draw</p>
<pre><code class="language-r">library(ggplot2)
autoplot(tsk(&quot;penguins&quot;), type = &quot;duo&quot;) +
  theme(strip.text.y = element_text(angle = -45, size = 8))
</code></pre>
<h4>Classification Learner</h4>
<p>From <code>LearnerClassif</code> The classification learners have almost identical interfaces with regression learners;</p>
<p>But the possible predictions in the classification are not the only ones.<code>&quot;response&quot;</code> That's the type of predictive observation.<code>&quot;prob&quot;</code> Projections of the probability vectors of observation for each category (or a lateral probability)
<code>response</code> The default is the highest predictive probability. Category</p>
<h4>Classification Evaluator</h4>
<p>Classification measures (categorys) <code>MeasureClassif</code> the same interface as the regression measure.</p>
<p>But we found that the task type of the classification is divided into two categories and multiple categories, and the predictive type of the classification is divided into probability and class predictions, and they all relate to the gap in the evaluator, and we need to make a choice based on looking at all the evaluators in the first place.</p>
<pre><code class="language-r">as.data.table(msr())[
    task_type == &quot;classif&quot; &amp; predict_type == &quot;prob&quot; ]
</code></pre>
<p>The first part limits the type of task for the evaluator:<code>classif</code>Still?<code>regr</code>
The second part limits the type of projection.<code>prob</code>Still?<code>response</code></p>
<p>The example of the code is that the whole interface is the same.</p>
<pre><code class="language-r">measures = msrs(c(&quot;classif.mbrier&quot;, &quot;classif.logloss&quot;, &quot;classif.acc&quot;))
prediction&#36;score(measures)
</code></pre>
<h4>Projections in the classification</h4>
<p><code>PredictionClassif</code> There are two important differences between objects and their regression simulations.</p>
<ul>
<li>Add field first <code>&#36;confusion</code></li>
<li>Next to add method <code>&#36;set_threshold()</code></li>
</ul>
<p>They wouldn't have been.</p>
<pre><code class="language-r">prediction
</code></pre>
<p>The code directly accesses them. They're all special amounts of classification problem predictions.</p>
<h5>Confusion matrix</h5>
<p>Confusion matrix is a popular way of showing in more detail the quality of classification (response) forecasts by seeing whether the model is good at categorizing observations in a given category (in error)</p>
<p>For binary and multi-classification, confusion matrix stored in <code>PredictionClassif</code> object <code>&#36;confusion</code> Field Access Code</p>
<pre><code class="language-r">prediction&#36;confusion
</code></pre>
<p>On the theoretical interpretation, we can look at the introduction to machine learning and supervision of learning: calibration rate, full rate and F1.</p>
<p>Specifically, we can visualize the graphics from the confusing matrix.</p>
<pre><code class="language-r">autoplot(prediction)
</code></pre>
<h5>Threshold threshold</h5>
<p>Another issue raised by classification as compared to returns is the issue of thresholds;</p>
<p>Default <code>response</code> The projection type is the highest predictive probability category, and if the maximum probability is not the only, that is, multiple categories are projected to have the highest probability and are then randomly selected from these categories;</p>
<p>In the binary classification, this means that if the projected category is more than 50%, the positive category will be selected, otherwise the negative category will be selected</p>
<p><strong>This value of 50 per cent is referred to as the threshold, which may be useful if there is a class imbalance (when a class is over- or under-centralized), or if there are different costs associated with the class, or if only if there is a preference for a class “over” predicting.</strong></p>
<p>It's easy to set a threshold in a category II problem.</p>
<pre><code class="language-r">prediction&#36;set_threshold(0.7)
</code></pre>
<p>At this point, there's just...<code>prob&gt;0.7</code>That's when it's supposed to be positive.</p>
<p>In multiple categories, the working principle for threshold processing is to start with each <code>n</code> Class allocates a threshold by dividing the predicted probability of each category by these thresholds to return <code>n</code> And at this point, the threshold still reflects the preferences that we choose, and the bigger the threshold, the more we deviate from it.</p>
<p>Yes. <code>mlr3</code> ♪ Medium, it's through ♪ <code>&#36;set_threshold()</code> Pass naming list to achieve</p>
<pre><code class="language-r">library(ggplot2)
library(patchwork)

tsk_zoo = tsk(&quot;zoo&quot;)
splits = partition(tsk_zoo)
lrn_rpart = lrn(&quot;classif.rpart&quot;, predict_type = &quot;prob&quot;)
lrn_rpart&#36;train(tsk_zoo, splits&#36;train)
prediction = lrn_rpart&#36;predict(tsk_zoo, splits&#36;test)
before = autoplot(prediction) + ggtitle(&quot;Default thresholds&quot;)
new_thresh = proportions(table(tsk_zoo&#36;truth(splits&#36;train)))
new_thresh
prediction&#36;set_threshold(new_thresh)
after = autoplot(prediction) + ggtitle(&quot;Inverse weighting thresholds&quot;)
before + after + plot_layout(guides = &quot;collect&quot;)
</code></pre>
<p>It's usually called reverse weighting.</p>
<h3>Taskbar Roles</h3>
<p>Now that we have described regression and classification, we will briefly return to the task; the role is the most important metadata that learners and other objects can use to interact with the task; there are seven roles:</p>
<ul>
<li><code>&quot;feature&quot;</code> : function for prediction</li>
<li><code>&quot;target&quot;</code> Target variable to predict</li>
<li><code>&quot;name&quot;</code> : row name/observation label, e.g., for <code>mtcars</code> This is... <code>&quot;model&quot;</code> Columns</li>
<li><code>&quot;order&quot;</code> : For right <code>&#36;data()</code> variables for sorting returned data; using <code>order()</code></li>
<li><code>&quot;group&quot;</code> : variable used to keep observations together during redistribution</li>
<li><code>&quot;stratum&quot;</code> : Layered variables during re-sampling</li>
<li><code>&quot;weight&quot;</code> : Observation weight. Only one numerical column can have this role</li>
</ul>
<p><code>feature</code> and <code>target</code> We've talked about it before.<a href="/en/blog/2024/04/06/r-mlr3verse-learning-notes/">Tasks</a>
<code>stratum</code> and <code>group</code> We'll introduce it later in the section.
We're not going to give you details. <code>name</code> , it's mainly used for mapping, and it's almost always the bottom data. <code>rownames()</code></p>
<p><strong>Use <code>&#36;set_col_roles()</code> Update Column Roles</strong> When the column is updated, it will not be used as another column, which means that each column has only one column.</p>
<h4>order</h4>
<p>Yeah.<code>&quot;order&quot;</code> Role Data sorted according to this column
♪ When we run ♪ <code>&#36;data()</code> It is no longer used as a feature, but rather to rank observations according to their values. This metadata will not be passed to learners</p>
<pre><code class="language-r">df = data.frame(mtcars[1:2, ], idx = 2:1)
tsk_mtcars_order = as_task_regr(df, target = &quot;mpg&quot;)
## 初始排序
tsk_mtcars_order&#36;data(ordered = TRUE)

## 根据列 idx 进行排序
tsk_mtcars_order&#36;set_col_roles(&quot;idx&quot;, roles = &quot;order&quot;)
tsk_mtcars_order&#36;data(ordered = TRUE)
</code></pre>
<h4>weight</h4>
<p><code>weights</code> Column roles are used to weight data points differently; in classification tasks with serious category imbalances, a more weighted minority category may increase the predictive performance of the model for that category</p>
<p>Example code:</p>
<pre><code class="language-r">cancer_unweighted = tsk(&quot;breast_cancer&quot;)
summary(cancer_unweighted&#36;data()&#36;class)

## add column where weight is 2 if class &quot;malignant&quot;, and 1 otherwise
df = cancer_unweighted&#36;data()
df&#36;weights = ifelse(df&#36;class == &quot;malignant&quot;, 2, 1)

## create new task and role
cancer_weighted = as_task_classif(df, target = &quot;class&quot;)
cancer_weighted&#36;set_col_roles(&quot;weights&quot;, roles = &quot;weight&quot;)

## compare weighted and unweighted predictions
split = partition(cancer_unweighted)
lrn_rf = lrn(&quot;classif.ranger&quot;)
lrn_rf&#36;train(cancer_unweighted, split&#36;train)&#36;
  predict(cancer_unweighted, split&#36;test)&#36;score()

lrn_rf&#36;train(cancer_weighted, split&#36;train)&#36;
  predict(cancer_weighted, split&#36;test)&#36;score()
</code></pre>
<p>In this example, the weighting increases the overall performance of the model; not all models can handle the weights of the task, so please check the attributes of the learners to ensure that this role is used as expected</p>
<h3>Supported learning algorithms</h3>
<p><code>mlr3</code>Support many learning algorithms; these are mainly <code>mlr3</code> 、 <code>mlr3learners</code> and <code>mlr3extralearners</code> Packages are available; of course, the newer packages usually contain some of the newer algorithms.</p>
<h4>mlr3</h4>
<p><code>mlr3</code> The list of learning devices included is deliberately small, thereby reducing reliance on other packages;</p>
<ul>
<li>Featureless learners (<code>&quot;regr.featureless&quot;</code>/<code>&quot;classif.featureless&quot;</code>) Use as a benchmark learner</li>
<li>Debug learners (<code>&quot;regr.debug&quot;</code>/<code>&quot;classif.debug&quot;</code>For code debugging</li>
<li>Classification and regression trees (also known as CART: <code>&quot;regr.rpart&quot;</code>/<code>&quot;classif.rpart&quot;</code>It's also known as CRAT.</li>
</ul>
<h4>mlr3learners</h4>
<p><code>mlr3learners</code> The package contains a series of algorithms chosen by the mlr team.</p>
<ul>
<li>Linear <code>&quot;regr.lm&quot;</code> ) and logic ( ) <code>&quot;classif.log_reg&quot;</code> ♪ Back</li>
<li>The punishment is a broad linear model in which the punishment is either used as a super-parameter (para. <code>&quot;regr.glmnet&quot;</code> / <code>&quot;classif.glmnet&quot;</code> ), or optimise automatically ( <code>&quot;regr.cv_glmnet&quot;</code> / <code>&quot;classif.cv_glmnet&quot;</code> ）</li>
<li>Weighted&#36;k&#36;Near Neighbors <code>&quot;regr.kknn&quot;</code> / <code>&quot;classif.kknn&quot;</code> ）</li>
<li>Kriging / Gaussian process regression（ <code>&quot;regr.km&quot;</code> ）</li>
<li>Linear <code>&quot;classif.lda&quot;</code> ) and secondary ( ) <code>&quot;classif.qda&quot;</code> Other Organiser</li>
<li>PARK Soo Bayes <code>&quot;classif.naive_bayes&quot;</code> ）</li>
<li>Support vectors <code>&quot;regr.svm&quot;</code> / <code>&quot;classif.svm&quot;</code> ）</li>
<li>Gradient Enhancement <code>&quot;regr.xgboost&quot;</code> / <code>&quot;classif.xgboost&quot;</code> ）</li>
<li>Return and classification of random forests <code>&quot;regr.ranger&quot;</code> / <code>&quot;classif.ranger&quot;</code> ）</li>
</ul>
<h4>View learning algorithms</h4>
<p>Normally, we would take all available learners and convert them to a data frame and briefly review their format.</p>
<pre><code class="language-r">learners_dt = as.data.table(mlr_learners)
learners_dt
</code></pre>
<p>Generated <code>data.table</code> It contains a large amount of metadata that are very useful for identifying learners with specific attributes</p>
<p>Lists all learners who support classification questions:</p>
<pre><code class="language-r">learners_dt[task_type == &quot;classif&quot;]
</code></pre>
<p>Several conditions are filtered, listing all retrogressors that can predict standard errors:</p>
<pre><code class="language-r">learners_dt[task_type == &quot;regr&quot; &amp;
  sapply(predict_types, function(x) &quot;se&quot; %in% x)]
</code></pre>
<h2>Evaluation and Benchmarking</h2>
<p>Monitoring machine learning models can only be deployed in practice if they have good generalization capabilities, so accurate estimates of generalization performance are essential for many aspects of machine learning applications and research, which will be an important basis for our selection in multiple models and for over-parametric adjustments;</p>
<p>We know that the use of the same data to train and test models is a bad strategy and that it is simply impossible to solve the problem of alignment. In the previous section, we've introduced <code>partition()</code> It divides the data sets into training data (for training models) and test data (for testing models and estimating generalization performance)<a href="/en/blog/2024/04/06/r-mlr3verse-learning-notes/">Partition() Description</a> This is called holdout strategy, and it will be the beginning of this chapter. We will then consider a more advanced strategy for assessing generalization performance.</p>
<p>A common misunderstanding is that holdout and other, more advanced, resampling models can prevent over-formulation, and in fact these methods simply make it visible because we can evaluate training/test performance separately. He allowed us to make almost impartial estimates of general errors.</p>
<h3>Holdout Policy</h3>
<p>An important objective of ML is to learn a model that can then be used to predict new data. In order to make the model as accurate as possible, we would ideally use as much data as possible to train it. However, the data are limited and, as we discussed, we cannot train and test models on the same data.</p>
<p>In practice, one usually creates an intermediate model that is trained on a subset of available data and then tested on the remaining data. The performance of the intermediate model obtained by comparing model predictions with real data is an estimate of the generalization of the final model. And finally, we get intermediate model information and superparameter information to train models on all data, which is the result of our final output.</p>
<p>The holdout strategy is a simple way of creating a division between training and testing data sets. Ideally, training data sets should be as large as possible so that intermediate models represent the final model as possible, while test data sets should be as large as possible to allow accurate estimates of general error.</p>
<p>Based on experience, two thirds of the data are usually used for training and one third for testing, as this provides a reasonable balance between deviations and differences in generalized performance estimates</p>
<p>We've already introduced the code he needs to use. Down</p>
<pre><code class="language-r">tsk_penguins = tsk(&quot;penguins&quot;)
splits = partition(tsk_penguins)
lrn_rpart = lrn(&quot;classif.rpart&quot;)
## 在训练集上训练 测试集上预测
lrn_rpart&#36;train(tsk_penguins, splits&#36;train)
prediction = lrn_rpart&#36;predict(tsk_penguins, splits&#36;test)
## 对测试集的预测结果进行评分
prediction&#36;score(msr(&quot;classif.acc&quot;))
</code></pre>
<p>When dividing data, observation values must be confused to remove any information coded in the data sorting. Because... <code>tasks</code> Establishing the data used is likely to be based on some regular data, which is also a common practice in data collection. But it will affect the division of our tests and training sets.</p>
<p><code>partition()</code> And all the restampling strategies discussed below will automatically and randomly divide the data to prevent any bias, to ensure that our training in models, projections, and generalized error estimates are valid.</p>
<p>Many performance indicators are based on “decompositionable” losses, which means that they calculate differences between projections and real values first at the observation level, and then summarize the loss values in the test set into single value fractions.</p>
<p>In fact, we have a more sophisticated assessment strategy, which is a non-dissolved performance measure, and we'll talk about it later.</p>
<h3>Resampling Policy</h3>
<p>The Resampling strategy repeats all available data into multiple training and testing sets. One of the repetitions corresponds to the other. <code>mlr3</code> Resampling itseration or re-sampling Subaru.</p>
<p><strong>The panoramic performance is ultimately estimated by aggregating the performance scores of multiple retraces.</strong></p>
<p>By repeating the data split process, the data points can be reused for training and testing, making it more efficient to use all available data for performance estimates. In addition, a large number of reclassifications can reduce the difference in fractions, resulting in more reliable performance estimates. This means that performance estimates are unlikely to be affected by “unfortunate” fragmentation.</p>
<p><strong>We can generally think that Resampling's strategy provides a better general error estimate than the Holdout strategy described earlier. But at the same time, he'll bring more performance costs because we trained and tested multiple models.</strong></p>
<h4>Resampling Policy Theory</h4>
<h5>CV</h5>
<p>A very common strategy is k-fold cross-validation.
It randomly divides the data into&#36;k&#36;A non-overlapping subset, called discounts; &#36;k&#36;The models are always there.&#36;k-1&#36;Collapse training, with the remaining fold being used to test data; repeat the process until each fold is accurately implemented as a test set. Finally, a summary of performance estimates for each fold is usually sought in mean. CV ensures that each observation is used only once in the test concentration, thus effectively using available data for performance estimates</p>
<p>&#36;k&#36;Common values are 5 and 10, which means that each set will consist of 4/5 or 9/10 of raw data.</p>
<p>CV has several variants, including repeat k-clip cross-certification (where k-cV is repeated) and a cross-certification (LOO-CV), with a discount equal to the number of observations, resulting in a test set consisting of only one observation per discount</p>
<p>Theory can be used to refer to the introduction to machine learning and supervisory learning: cross-certification.</p>
<h5>Subsampling and Bootstrapping</h5>
<p>Subsampling randomly selects the data of the given ratio (commonly 4/5 and 9/10) for use in the training data set, where each observation in the data set is extracted from the original data collection and does not need to be replaced. The model is trained on this data and tested on the remaining data, and this process is repeated&#36;k&#36;Minor</p>
<p>Bootstrapping, the strategy is self-help.</p>
<h5>Policy Selection</h5>
<p>Resampling strategy choices usually depend on the specific tasks at hand and the objectives of the performance assessment, but there are some empirical rules.</p>
<p>If available data are small (&#36;N)&lt;&#36;500, with a large number of duplicate cross-checks that can be used to keep performance estimates low (10 times and 10 times as a good starting point)</p>
<p>The LOO-CV were also recommended for these small sample quantities, but the estimated cost is very high (except in exceptional cases where a shortcut exists) and is in violation of a fairly high range of instincts. At the same time, he has problems with the unbalanced binary classification task.</p>
<p>For &#36;500.&lt;N&lt;&#36;5,000 range, usually 5-10 CV</p>
<p>Bootstrapping has become less common because repeated sampling can cause problems in machine learning algorithms.</p>
<p>Later, we'll give you details of how these Resampling strategies are implemented in R.</p>
<h4>Create Resampling Policy</h4>
<p>All achieved Resampling policies are stored in <code>mlr_resamplings</code> Dictionary</p>
<pre><code class="language-r">as.data.table(mlr_resamplings)
</code></pre>
<p><code>params</code> Shows the parameters of each Resampling policy that can be constructed from behind<code>Resampling</code> Modify Parameters When Object
<code>iters</code>Column shows the default number of Resampling iterative times we do not normally need to adjust</p>
<p><code>Resampling</code> Object can pass the policy "key" to the sugar function <code>rsmp()</code> To construct, for example.</p>
<pre><code class="language-r">rsmp(&quot;holdout&quot;, ratio = 0.8)
</code></pre>
<p>It's built. <code>holdout</code> We modified the default ratio from two thirds of the training set to four fifths of the training. Set</p>
<p>From <code>Resampling</code> The calibration and evaluation of the parameters of the object of succession. Let's go.
<code>Resampling</code> The grammatical rule that you're constructing is simply to replace the SugarFunction with <code>rsmp()</code></p>
<pre><code class="language-r">## three-fold CV
cv3 = rsmp(&quot;cv&quot;, folds = 3)
## Subsampling with 3 repeats and 9/10 ratio
ss390 = rsmp(&quot;subsampling&quot;, repeats = 3, ratio = 0.9)
## 2-repeats 5-fold CV
rcv25 = rsmp(&quot;repeated_cv&quot;, repeats = 2, folds = 5)
</code></pre>
<p>We can do this manually, but this operation is cumbersome and often ineffective.</p>
<h4>Resampling objects for practical learning</h4>
<p><code>partition()</code> The function accepts the task automatically dividing the training and testing set for us and returns the line numbers; of course, Resampling objects should have the same function;</p>
<p><code>resample()</code> Function accepts given <code>Task</code> 、 <code>Learner</code> and <code>Resampling</code> object to run a given Resampling policy. <code>resample()</code> Repeat the assembly model on the training set, predict it on the corresponding test set and store it in <code>ResampleResult</code> object, the object contains all the information needed to estimate the generalized performance</p>
<pre><code class="language-r">rr = resample(tsk_penguins, lrn_rpart, cv3)
rr
</code></pre>
<p><strong>We changed the process of learning the front learners while we were at Resampling, while changing the training and prediction steps.</strong></p>
<p>Of course, we still need an evaluator to evaluate it.</p>
<pre><code class="language-r">## 返回在每次迭代中的性能
acc = rr&#36;score(msr(&quot;classif.ce&quot;))
acc[, .(iteration, classif.ce)]

## 聚合多次迭代 给出更加常用的结果
rr&#36;aggregate(msr(&quot;classif.ce&quot;))
</code></pre>
<p>By default, most measures will use macro averages (the average of scores directly to each test set) to aggregate fractions, but we can specify micromeans (he considers different sizes of each test set) to aggregate fractions.</p>
<pre><code class="language-r">## 这就是采用了微平均值 需要直接修改我们的评价器
rr&#36;aggregate(msr(&quot;classif.ce&quot;, average = &quot;micro&quot;))
</code></pre>
<p>Through Query <code>Measure</code> object <code>&#36;average</code> Fields can find the default type of aggregation method</p>
<p>Visual reordering results, available <code>autoplot.ResampleResult()</code> function. Histograms can be used to measure intuitively the variance of the intermediate performance of the rearranged trajectories, while box charts are usually used to compare multiple studies in parallel Device</p>
<pre><code class="language-r">## 训练模型 返回ResampleResult对象
rr = resample(tsk_penguins, lrn_rpart, rsmp(&quot;cv&quot;, folds = 10))
## 使用 autoplot 函数 和前面一样 他为各种 mlr3 对象绘图 此时可以选择绘图的类型
autoplot(rr, measure = msr(&quot;classif.acc&quot;), type = &quot;boxplot&quot;)
autoplot(rr, measure = msr(&quot;classif.acc&quot;), type = &quot;histogram&quot;)
</code></pre>
<h4>ResampleResult Object</h4>
<p>We changed the learning function of the learning device to use the Resampling strategy, so the result of the study became a ResampleResult object.</p>
<p>We discussed how to use ResampleResult objects to calculate general errors, but ResampleResult objects cannot be used for this purpose only.</p>
<p>We can use it. <code>&#36;predictions()</code>The method of obtaining a corresponding forecast for each resource <code>Prediction</code> List of Objects  <strong>The target.<code>Prediction</code></strong></p>
<pre><code class="language-r">## 返回结果是一个列表 里面含有迭代次数个元素
rrp = rr&#36;predictions()
</code></pre>
<p>By default, intermediate models produced in each Resampling policy trajectories are discarded after predicting steps to reduce <code>ResampleResult</code>Memory consumption of objects (the greatest effect of which is performance measurement)</p>
<p>But we can go through the settings. <code>store_models = TRUE</code> Configure <code>resample()</code> function to maintain the proposed intermediate model. And then, through <code>&#36;learnersi&#36;model</code> Visit every model trained in a particular recovery trajectories, where <code>i</code> It means no. <code>i</code>Other Organiser</p>
<pre><code class="language-r">rr = resample(tsk_penguins, lrn_rpart, cv3, store_models = TRUE)
## 得到各个学习器 后面的&#36;model查看学习器的模型
rr&#36;learners[[1]]&#36;model
</code></pre>
<p>We'll be able to access all the information about the model, if we need it.</p>
<h4>Custom Resampling</h4>
<p>Self-defining Resampling is something that might be needed. <code>mlr3</code> It provides the appropriate way.</p>
<p>Custom <code>holdout</code> For information</p>
<pre><code class="language-r">rsmp_custom = rsmp(&quot;custom&quot;)

## resampling strategy with two iterations
train_sets = c(1:5, 153:158, 277:280)
rsmp_custom&#36;instantiate(tsk_penguins,
  train = list(train_sets, train_sets + 5),
  test = list(train_sets + 15, train_sets + 25)
)
resample(tsk_penguins, lrn_rpart, rsmp_custom)&#36;prediction()
</code></pre>
<p>Custom<code>cv</code>  For information</p>
<pre><code class="language-r">tsk_small = tsk(&quot;penguins&quot;)&#36;filter(c(1, 100, 200, 300))
rsmp_customcv = rsmp(&quot;custom_cv&quot;)
folds = as.factor(c(1, 2, 1, 2))
rsmp_customcv&#36;instantiate(tsk_small, f = folds)
resample(tsk_small, lrn_rpart, rsmp_customcv)&#36;predictions()
</code></pre>
<h4>Layers and Layers</h4>
<p>Use taskbar roles to group or layer observations according to specific columns in the data</p>
<h5>Grouped Resampling</h5>
<p>In longitudinal studies, measurements are made from the same body at multiple points of time. If we do not group these data, we may overestimate the model's ability to extend to unknown individuals, since observations of the same individuals may be present at the same time in the training set and in the concentration of testing.</p>
<p><code>&quot;group&quot;</code> Column roles allow us to specify columns in the data that define the group structure for observation. At this point in the construction of Resampling, the folding of each observation becomes a folding of groups.</p>
<pre><code class="language-r">rsmp_loo = rsmp(&quot;loo&quot;)
tsk_grp = tsk(&quot;penguins&quot;)
tsk_grp&#36;set_col_roles(&quot;year&quot;, &quot;group&quot;) rsmp_loo&#36;instantiate(tsk_grp)
</code></pre>
<h5>Stratified Sampling</h5>
<p>A layer sample ensures that one or more discrete features of the training set and test concentration will have a distribution similar to that of the original mission covering all observations; this ensures that multiple and iterative estimates are accurate in cross-checking;</p>
<p>Different from grouping, can be used <code>&quot;stratum&quot;</code> Column roles are layered according to multiple discrete characteristics. In this case, the layer will be formed by each combination of the layers, as follows:</p>
<pre><code class="language-r">tsk_str = tsk(&quot;penguins&quot;)
## 设定 species 同时作为分层用的`&quot;stratum&quot;` 列 和&quot;target&quot;列
tsk_str&#36;set_col_roles(&quot;species&quot;, c(&quot;target&quot;, &quot;stratum&quot;))
rsmp_cv10&#36;instantiate(tsk_str)
</code></pre>
<h3>Benchmark Test Benchmarking</h3>
<p>Benchmark tests in machine learning are learning devices that compare different tasks.</p>
<p>When comparing multiple learning devices on a single mission or on more than one similar task, the main purpose is usually to rank learning devices according to predefined performance measures and to determine the best learning instrument for the given task.</p>
<p>When multiple learners are compared on multiple assignments, the main purpose is often less simple than before.
For example, an in-depth understanding of the performance of different learners in different data situations, or the existence of certain data attributes that significantly affect the performance of some learners (or some over-parameters of learners).</p>
<p>Since baseline tests usually consist of many assessments that can operate independently of each other, <code>mlr3</code> The possibility of automatic parallelization is thus provided. In this section, we present the most extensive baseline tests used and discuss more complex benchmarking issues later.</p>
<h4>Benchmark</h4>
<p><code>mlr3</code> The baseline experiment is used. <code>benchmark()</code> It's done, it's only run to each task and learner separately <code>resample()</code> , then collect the results. The re-sampling strategy provided will be automatically exemplified for each assignment to ensure that all learners are compared with the same training and testing data</p>
<p>It's obvious that for benchmarking, we need to introduce multiple tasks, multiple learners, possibly multiple Resample methods, so the code has</p>
<pre><code class="language-r">## 建立两个任务
tasks = tsks(c(&quot;german_credit&quot;, &quot;sonar&quot;))
## 建立三个学习器
learners = lrns(c(&quot;classif.rpart&quot;, &quot;classif.ranger&quot;,
  &quot;classif.featureless&quot;), predict_type = &quot;prob&quot;)
## 建立一种 Resample 方法
rsmp_cv5 = rsmp(&quot;cv&quot;, folds = 5)

## 构造`benchmark()`方案并审阅
design = benchmark_grid(tasks, learners, rsmp_cv5)
head(design)
</code></pre>
<p>It's essentially just a design. <code>data.table</code> , if you want to delete a particular combination, you can modify it, even without it <code>benchmark_grid()</code> Create from scratch in a function</p>
<p>Then you can pass the constructed baseline design to <code>benchmark()</code> Run the experiment. The result is one. <code>BenchmarkResult</code> Object:</p>
<pre><code class="language-r">bmr = benchmark(design)
bmr
</code></pre>
<p>Because <code>benchmark()</code> It's just... <code>resample()</code> The extension we can use again <code>&#36;score()</code> or <code>&#36;aggregate()</code> This is how we look at the results.</p>
<pre><code class="language-r">bmr&#36;score()[c(1, 7, 13), .(iteration, task_id, learner_id, classif.ce)]

bmr&#36;aggregate()[, .(task_id, learner_id, classif.ce)]
</code></pre>
<p>We don't have a rigorous statistical hypothesis test here, so we have to be careful to draw conclusions about which model is better.</p>
<h4>BenchmarkResult Object</h4>
<p>Object <code>BenchmarkResult</code> Multiple <code>ResampleResult</code> Collection of Objects</p>
<p>We can extract the BenchmarkResult object<code>ResampleResult</code> Object has</p>
<pre><code class="language-r">rr1 = bmr&#36;resample_result(1)
rr1
</code></pre>
<p>And then if you need more detailed access, you can pass.<a href="/en/blog/2024/04/06/r-mlr3verse-learning-notes/">ResampleResult Object</a> Codes in order to achieve access</p>
<p>In addition, <code>as_benchmark_result()</code> It can also be used to direct objects from <code>ResampleResult</code> Convert to <code>BenchmarkResult</code> 。<code>c()</code> Available for grouping multiples <code>BenchmarkResult</code> Object</p>
<pre><code class="language-r">bmr1 = as_benchmark_result(rr1)
bmr2 = as_benchmark_result(rr2)

c(bmr1, bmr2)
</code></pre>
<p>The BenchmarkResult object also has an exclusive visualization method, which gives boxplot to compare the effects of multiple algorithms.</p>
<pre><code class="language-r">autoplot(bmr, measure = msr(&quot;classif.acc&quot;))
</code></pre>
<h3>Assessment of the binary taxonomyr</h3>
<p>We're here.<a href="/en/blog/2024/04/06/r-mlr3verse-learning-notes/">Classification Evaluator</a> <a href="/en/blog/2024/04/06/r-mlr3verse-learning-notes/">Projections in the classification</a> It's about taxonomic evaluation; now we're going to look into it.</p>
<p>In the theoretical part of machine learning, we're introducing some of the knowledge that machine learning leads and supervises learning: performance measurement, simple re-reading and code realization.</p>
<p><code>mlr3measures</code> Package allows you to use the following: <code>confusion_matrix()</code> Function calculates several common measures based on confusing matrices</p>
<pre><code class="language-r">mlr3measures::confusion_matrix(truth = prediction&#36;truth,
  response = prediction&#36;response, positive = tsk_german&#36;positive)
</code></pre>
<p>Draw ROC curves needs</p>
<pre><code class="language-r">autoplot(prediction, type = &quot;roc&quot;)
</code></pre>
<p>It's not a common measure derived from a confusing matrix.</p>
<pre><code class="language-r">prediction&#36;score(msr(&quot;classif.auc&quot;))
</code></pre>
<p>Draw a PRC curve (precision-recall rate curve) that requires</p>
<pre><code class="language-r">autoplot(prediction, type = &quot;prc&quot;)
</code></pre>
<p>Finally, we consider the relationship between thresholds and indicators.</p>
<pre><code class="language-r">autoplot(prediction, type = &quot;threshold&quot;, measure = msr(&quot;classif.fpr&quot;))
autoplot(prediction, type = &quot;threshold&quot;, measure = msr(&quot;classif.acc&quot;))
</code></pre>
<p>These visualizations are perfectly usable. <code>ResampleResult BenchmarkResult 对象 替换原本的</code> Just do it.</p>
<h2>Hyperparameter Optimization</h2>
<p>Starting with this chapter, we're working on three consecutive chapters on how to upgrade learning machines; including the most basic automatic over-parameter regulation; and on further modulation methods and feature engineering; that's what we're doing.<code>mlr3</code>Then we started learning how to build a better model.</p>
