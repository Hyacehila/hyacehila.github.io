---
title: 'Python Iterators, Generators, and Lambda: Lazy Evaluation and Functional Tools'
title_zh: Python 迭代器、生成器与 Lambda：惰性计算和函数式工具
date: 2025-04-19 19:53:29 +0800
categories:
- Programming
- Python
tags:
- Python
- Iterators
author: Hyacehila
mathjax: false
hidden: true
excerpt: Covers iterables, iterators, map, filter, generators, sort key functions, and lambda functions.
description: Covers iterables, iterators, map, filter, generators, sort key functions, and lambda functions.
excerpt_zh: 整理可迭代对象、迭代器、map、filter、生成器、排序 key 参数和 lambda 函数。
permalink: /blog/2025/04/19/python-iterators-generators-lambda-learning-notes/
lang: en
translation_key: 2025-04-19-python-iterators-generators-lambda
translation_status: machine
translation_source_hash: 49cb84c4e8bdef3372b24cef1f53ede08539451fadd173946af5a51791c2090c
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<h2>About the iterative</h2>
<p>The questions in this article can also be addressed<a href="/en/blog/2023/03/18/python-basics-learning-notes/">Python Foundation: Syntax: Data Structure and File Processing</a>、<a href="/en/blog/2025/03/04/python-oop-and-decorators-learning-notes/">Python For Object and Decorator: Classes, Inheritance, Policy and Closed</a>How the concept of a relatively close read together is developed in different contexts.</p>
<h3>Invertable objects and anipher</h3>
<p><strong>Reversible objects</strong>It means that it's done. <code>__iter__()</code> The object of the method. In short, it's an object that can be repeated, that is, it can be used. <code>for</code> Cycle through objects. Common iterative objects include lists, groups, strings, dictionaries, collections, etc.</p>
<p>The object must be an iterative object <code>__iter__()</code> Method, method returns one<strong>Organisation</strong>I'm sorry. When used <code>for</code> When you recycle an iterative object, you actually call the object first <code>__iter__()</code> method to get an iterative device, and then access the elements one by one through an iterative device.</p>
<p>The anecdote is a special object, and it's achieved. <code>__iter__()</code> and <code>__next__()</code> Method (and therefore an iterative object can also be considered).<code>__iter__()</code> Method returns the iterative device itself, and <code>__next__()</code> Method to return the next element of the iterative. When there are no more elements,<code>__next__()</code> The way it's going to be thrown out. <code>StopIteration</code> Unusual.</p>
<p>Like the effect of the code.</p>
<pre><code class="language-python">my_list = [1, 2, 3]
# 获取列表的迭代器
iterator = iter(my_list)

# 使用 next() 函数逐个获取元素
print(next(iterator))  # 输出 1
print(next(iterator))  # 输出 2
print(next(iterator))  # 输出 3

# 再次调用 next() 会抛出 StopIteration 异常
try:
    print(next(iterator))
except StopIteration:
    print(&quot;已经没有更多元素了&quot;)
</code></pre>
<p>For the reference <code>for</code> Cycles functions, and the iterative and iterative objects are important, and many functions are developed on the basis of the iterative objects and are important for their realization.</p>
<h3>Intersectional Relevant Functions</h3>
<h4><code>map</code> Functions</h4>
<p><code>map</code> function is applied to each element of an iterative object and returns a new iterative device, the element of which is the result of each element of the original avoidable object being processed by the specified function. Its basic grammar is as follows:</p>
<pre><code class="language-python">map(function, iterable, ...)
</code></pre>
<ul>
<li><code>function</code>: This is the function to be applied, it will be applied to <code>iterable</code> - Every element of it.</li>
<li><code>iterable</code>: This is one or more iterative objects, such as lists, groups of elements, collections, etc. If you can enter multiple, it is possible to enter several, many, many, many, many, many, many, many, many, many, many, many, many, many, many, many, many, many, many, many, many, many, many, many, many, many, many, many, many, many, many, many, many, many, many, many, many, many, many, many, many, many, many, many, many, many, many, many, many, many, many, many, many, many, many, many, many, many, many, many, many, many, many, many, others, others, others, others, others, others, and others, others, others, and others.<code>function</code> The same number of parameters as the number of reversible objects must be acceptable. Otherwise, it would be a mistake.</li>
</ul>
<p>A simple example</p>
<pre><code class="language-python"># 定义一个将元素平方的函数
def square(x):
    return x ** 2

numbers = [1, 2, 3, 4, 5]
# 使用 map 函数将 square 函数应用到 numbers 列表的每个元素上
squared_numbers = map(square, numbers)
# 此时返回 map 对象，无法直接打印，需要转换回我们熟悉的数据结构上
# 将 map 对象转换为列表
result = list(squared_numbers)
print(result)  # 输出: [1, 4, 9, 16, 25]
</code></pre>
<p>About <code>map</code> Function We use often <code>lambda</code> function to simplify the related issues, as follows:</p>
<pre><code class="language-python">numbers = [1, 2, 3, 4, 5]
# map 函数自身会帮助我们把各个元素传入给函数的，lambda仅仅用于定义函数
squared_numbers = map(lambda x: x ** 2, numbers)
result = list(squared_numbers)
print(result)  # 输出: [1, 4, 9, 16, 25]
</code></pre>
<p>Deals with multiple iterative objects simultaneously <code>map</code> And very naturally.</p>
<pre><code class="language-python">numbers1 = [1, 2, 3]
numbers2 = [4, 5, 6]
# 定义一个将两个元素相加的函数
def add(x, y):
    return x + y

result = map(add, numbers1, numbers2)
print(list(result))  # 输出: [5, 7, 9]
</code></pre>
<h4><code>filter</code> Functions</h4>
<p><code>filter</code> Function serves to filter elements in an iterative object, leaving only those that return the specified function <code>True</code> and returns a new iterative. Its basic grammar is as follows:</p>
<pre><code class="language-python">filter(function, iterable)
</code></pre>
<ul>
<li><code>function</code>: This is a filter function that accepts a parameter and returns a boolean value. If Back <code>True</code>, the element will be retained; if returned <code>False</code>, the element will be filtered out.</li>
<li><code>iterable</code>: This is an iterative object to filter, such as lists, groups, collections, etc.</li>
</ul>
<p>The overall grammar rule and <code>map</code> The function is very similar, and we need only one more example.</p>
<pre><code class="language-python"># 定义一个判断元素是否为偶数的函数
def is_even(x):
    return x % 2 == 0

numbers = [1, 2, 3, 4, 5, 6]
# 使用 filter 函数过滤出偶数
even_numbers = filter(is_even, numbers)
# 将 filter 对象转换为列表
result = list(even_numbers)
print(result)  # 输出: [2, 4, 6]
</code></pre>
<h3>Generator</h3>
<p><code>yield</code> Keywords for a label<code>generator</code> ♪ When a function contains <code>yield</code> So he automatically becomes a generator function and life cycle and character changes.</p>
<p><code>yield</code> The core of the keyword - “Pause and hand over”</p>
<ul>
<li><strong>Normal Functions (<code>return</code>)</strong>"Just like you read 20 episodes of the season and told your friend I'm finished." You only have one option to finish all the stories in one shot.</li>
<li><strong>Generator Function (<code>yield</code>)</strong>: Just like you look at a episode and then press the Pause and Hand over button. You give the remote control to your friend, and he goes to see something else. When he comes back to watch, you play him the next episode from where he's just suspended.</li>
</ul>
<p><code>yield</code> That's it."<strong>Pause and hand over</strong>. It did two very important things:</p>
<ol>
<li><strong>Hand over value</strong>: will <code>yield</code> The later expression results as the return value for this iterative period.</li>
<li><strong>Pause Functions</strong>: the performance status of the function (including the value of all local variables) is<strong>Freeze</strong>function " Go to sleep" and wait for the next wake-up call.</li>
</ol>
<p>When a function contains <code>yield</code> And it changed completely:</p>
<ol>
<li><strong>Call</strong>：<code>my_generator = my_func()</code> No function will be executed. It just created and returned one.<strong>Generator Object</strong>I'm sorry. This object is like a list of to-dos, which records the code of the function and its current status.</li>
<li><strong>First time in an iterative fashion</strong>: When you go through this generator for the first time (e.g., with <code>for</code> ) , function starts until the first one is encountered <code>yield</code>。</li>
<li><strong>I met him. <code>yield</code> Time</strong>: Function Handover <code>yield</code> The value in the back, then<strong>Time out immediately.</strong>, all internal status is saved.</li>
<li><strong>Next time it's an iterative one.</strong>: Function from last pause<strong>Awakening.</strong>Continue until you meet the next one. <code>yield</code>。</li>
<li><strong>Circumpolarally</strong>: This process continues until the function is performed (no more code) or a single one is encountered <code>return</code> statement.</li>
</ol>
<p>Since... <code>yield</code> It's the lead, that. <code>return</code> Is it still working?</p>
<p>Yes, but its role has changed. In the generator function,<code>return</code> The effect is...<strong>Early termination generator</strong>。</p>
<p>When the generator function executes one <code>return</code> When a statement is made, it will stop and trigger a <code>StopIteration</code> Unusual. This anomaly can be captured.<code>return</code> The value in the back will be this unusual. <code>value</code> attribute.</p>
<p>This is a relatively advanced use, usually used to transmit an additional "end state" or "misinformation" to the generator's users.</p>
<p>That's using the structures below to make unusual seizures.</p>
<pre><code class="language-python">def generator_with_return(n):
    i = 0
    while i &lt; n:
        yield i
        i += 1
    return &quot;我处理完了所有数字！&quot; # &lt;-- return 在这里

# 创建生成器
gen = generator_with_return(3)

# 手动迭代，以便捕获 StopIteration
while True:
    try:
        value = next(gen) # next() 函数获取下一个值
        print(f&quot;从生成器拿到: {value}&quot;)
    except StopIteration as e:
        print(f&quot;生成器结束了！&quot;)
        print(f&quot;它 return 的值是: {e.value}&quot;) # &lt;-- 在这里获取 return 的值
        break
</code></pre>
<p>In use<code>generator</code>, and then select the <code>for</code> It's a loop to read. He'll do it automatically.<code>StopIteration</code> The problem is that it ends naturally by going through all the elements.</p>
<p><strong>The generator can only be repeated once</strong></p>
<h2>Other</h2>
<h3>Sort Functions<code>key</code>Parameters</h3>
<p>In the most common <code>sorted()</code> function <code>key</code> We've decided what we use as a basis for sorting, in many cases, into. <code>sorted()</code> The function is often not a list but a complex dictionary, so we need to specify which key to use for sorting.</p>
<pre><code class="language-python">students = [
    {&#39;name&#39;: &#39;Alice&#39;, &#39;age&#39;: 20},
    {&#39;name&#39;: &#39;Bob&#39;, &#39;age&#39;: 18},
    {&#39;name&#39;: &#39;Charlie&#39;, &#39;age&#39;: 22}
]
# 按照年龄从小到大排序
sorted_students = sorted(students, key=lambda student: student[&#39;age&#39;])
print(sorted_students)
</code></pre>
<p><code>key</code> The actual working principle of the parameters is not that simple. He is a function.<code>sorted</code> Function <code>iterable</code> Each element of this calls this <code>key</code> function, then by <code>key</code> function to compare the size of the element, rather than directly to the element itself.</p>
<p>In this code, we have to consider the question of the code.<code>key</code> It's a function that fits the usual definition.<code>lambda</code>Function, we're<code>iterable</code> Each element of this calls this<code>lambda</code>function, get a value for sorting, here's the<code>lambda</code>Function to extract the key pairs entered into the function<code>age</code>The value of the key, that's what makes it understandable.</p>
<h3><code>lambda</code>Functions</h3>
<p><code>lambda</code> function is a simple anonymous function in Python that can be defined and used temporarily where the function object is needed, without the need to define a full function in a visible manner.<code>lambda</code> Keywords<strong>Create small, one-time, anonymous functions</strong>I'm sorry. Its basic grammar is as follows:</p>
<pre><code class="language-python">lambda 参数列表: 表达式
</code></pre>
<ul>
<li><strong>List of Parameters</strong>: This is the parameter that is passed to the function, with zero or more parameters, separated by commas.</li>
<li><strong>Expression</strong>: This is the value that the function is going to return,<code>lambda</code> function can only contain one expression, and the result of that expression will be returned automatically.</li>
</ul>
<p>This is the right place. <code>lambda</code> function is easily understood as " sorting function " <code>key</code> Why is it often used in Parameters: it can create simple functions quickly without redefinition and maintenance in the front.</p>
<p>Multiparameters<code>lambda</code> It's natural to define functions.</p>
<pre><code class="language-python">lambda x, y: x &lt; y
</code></pre>
<p>Like normal functions,<code>lambda</code> function can be used to return the function value (which can be called after it is called as normal), thus achieving some interesting functions, such as:</p>
<pre><code class="language-python">def multiplier(factor):
    return lambda x: x * factor

double = multiplier(2)
triple = multiplier(3)

print(double(5))  # 输出 10
print(triple(5))  # 输出 15
</code></pre>
