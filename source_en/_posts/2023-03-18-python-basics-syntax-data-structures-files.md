---
title: 'Python Basics: Syntax, Data Structures, and File Handling'
title_zh: Python 基础：语法、数据结构与文件处理
date: 2023-03-18 21:32:22 +0800
categories:
- Programming
- Programming Languages
tags:
- Python
author: Hyacehila
mathjax: false
hidden: true
excerpt: Covers variables, strings, lists, branches, sets, dictionaries, loops, functions, regular expressions, files, exceptions,
  and testing.
description: Covers variables, strings, lists, branches, sets, dictionaries, loops, functions, regular expressions, files,
  exceptions, and testing.
excerpt_zh: 整理变量、字符串、列表、分支、集合、字典、循环、函数、正则表达式、文件异常与测试等入门内容。
permalink: /blog/2023/03/18/python-basics-learning-notes/
lang: en
translation_key: 2023-03-18-python-basics-syntax-data-structures-files
translation_status: machine
translation_source_hash: bd1e6f836ce088e8b78b87d0df3c360ad05899132ff69dfa9f08c1435bfe35b6
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<p>The questions in this article can also be addressed<a href="/en/blog/2025/03/04/python-oop-and-decorators-learning-notes/">Python For Object and Decorator: Classes, Inheritance, Policy and Closed</a>、<a href="/en/blog/2025/04/19/python-iterators-generators-lambda-learning-notes/">Python, Generator and Lambda: Inert Calculator and Function Tool</a>How the concept of a relatively close read together is developed in different contexts.</p>
<p>The whole structure will be more practice-oriented, with a better foundation and practice to ensure that there is more experience, and it will be easier to look at official documents at the moment.</p>
<h2>- Open.</h2>
<h3>Analysis of Python applications and career development</h3>
<p>In short, Python is a programming language of “excellency”, “unequivocal”, and “simple”.</p>
<ul>
<li>You can learn a low curve, and you can't be a professional.</li>
<li>Open source systems with powerful eco-circles.</li>
<li>Explanatory language, perfect platform portability</li>
<li>Dynamic type language, support object-oriented and function-based programming</li>
<li>High code specification, readability</li>
</ul>
<p>Python has a place of use in the following areas.</p>
<ul>
<li>Backend Development - Python / Java / Go / PHP</li>
<li>DevOps - Python / Shell / Ruby</li>
<li>Data Collection - Python / C++ / Java</li>
<li>Quantified transactions - Python / C++ / R</li>
<li>Data Science - Python / R / Julia / Matlab</li>
<li>Machine Learning - Python / R / C++ / Julia</li>
<li>Automation Test - Python / Shell</li>
</ul>
<p>As a Python developer, there are also a number of employment options available to individuals based on their preferences and career plans.</p>
<ul>
<li>Python backend development engineer (server, cloud platform, data interface)</li>
<li>Python Transport Engineer (Automated Transport, SRE, DevOps)</li>
<li><strong>Python Data Analyst (data analysis, business intelligence, digital operations)</strong></li>
<li><strong>Python Data Excavator (mechanical learning, in-depth learning, algorithms specialist)</strong></li>
<li>Python reptile project Division</li>
<li>Python Test Engineer (Automation Testing, Test Development)</li>
</ul>
<p>Several recommendations for beginners:</p>
<ul>
<li>Make English as your working language.</li>
<li>Practice makes perfect.</li>
<li>All experience comes from mistakes.</li>
<li>Don&#39;I'm not a big fan of this, but I'm not a big fan of this.</li>
<li>Either outside or out.</li>
</ul>
<h3>Simple description and advantages and disadvantages</h3>
<ol>
<li>Simple and simple, learning curves are lower than many programming languages.</li>
<li>Open source, with strong communities and eco-circles, especially in the area of data analysis and machine learning</li>
<li>The interpretation language, which is inherently platform portable, is the code that works on different operating systems.</li>
<li>Both mainstream programming paradigms (object-oriented and functional programming) are supported.</li>
<li>The code is highly coded and readable and suitable for people with code-cleaning and coercive disorders.</li>
</ol>
<h2>Variables and simple data types</h2>
<h3>Some variable naming rules</h3>
<ul>
<li>Variable names consist of letters, numbers and underlineds, and numbers cannot begin</li>
<li>Sensitivity to case</li>
<li>Space-Enabled</li>
<li>Avoiding the use of reserved functions and grammar keywords</li>
<li>Lower and uppercase O are easily misread as numbers, carefully used.</li>
<li>All letters in capital are constants. All lowercases of normal variables are connected by underlined</li>
<li>The protected case properties begin with a single underline (discussed later)</li>
<li>Private case properties start with two underlineds (to be discussed later)</li>
</ul>
<pre><code class="language-python"># type函数可以查看变量的类型 由于不要求提前声明变量，我们可以直接进行赋值，类型交给解释器选择
type(a)
</code></pre>
<h3>Number</h3>
<p>Python is a fairly free language, and without any kind of definition, it's very easy for us to use it, either integer or floating point, and Python will provide you with an answer that will satisfy you.</p>
<p>The difference between the more important and the C is that he owns ** for multipliers and / for normal division % for residuals</p>
<h3>String</h3>
<h4>Simple String Introduction</h4>
<p>The single quotes, the double quotes are wrapped in strings, and these different quotes are meant to make it possible for you to use them successfully in the sentence.</p>
<pre><code class="language-python">s1 = &#39;hello, world!&#39;
s2 = &quot;hello, world!&quot;
</code></pre>
<p>Use in string<code>\</code>(reverse slash) to indicate a transposition, which means<code>\</code>The characters behind are no longer what they meant.<code>\</code>And then you can use a octal or hexadecimal to indicate the word. Arguments</p>
<pre><code class="language-python">s1 = &#39;\141\142\143\x61\x62\x63&#39;
s2 = &#39;\u9a86\u660a&#39;
print(s1, s2)
#各种编码格式都可以通过\进行显示
</code></pre>
<p>If you want to use &#39; \ &#39;In a string, he needs a transfer.</p>
<pre><code class="language-python">s1 = &quot;\\dsadasdasd\\&quot;
</code></pre>
<h4>String Method</h4>
<p>The way Python does it, he's not the same as the function.</p>
<p>There are basic ways to use a string</p>
<pre><code class="language-python">.title() .upper() .lower()
name=&quot;ada fesasd&quot;
print(name.title())
# 他们会对字符串进行首字母大写  全部大写 全部小写
name.split()
# 把空格当做划分符 将字符串拆解成字符列表
print(&#39;ll&#39; in s1)
# 判断字符串的包含关系
print(str2[2:5])
# 字符串的正常切片 和列表一样处理
</code></pre>
<h4>Variables in Strings</h4>
<p>Think about adding some variables to the output of the string, that's what it takes.</p>
<pre><code class="language-python">first_name=&quot;ada&quot;
last_name=&quot;love&quot;
full_name=f&quot;{first_name} {last_name}&quot;
print(full_name)
#f 是 format 的缩写 是对字符串个格式设置 他会替换变量生成新的字符串
</code></pre>
<p><code>f 字符串</code>It was introduced after python 3.6, and his previous format was a way of doing a string.</p>
<pre><code class="language-python">full_name=&quot;{} {}&quot;.format(first_name,last_name)
</code></pre>
<p>It's like the use of our research. <code>%d</code> String output for equal placeholder, e. g.</p>
<pre><code class="language-python">a, b = 5, 10
print(&#39;%d * %d = %d&#39; % (a, b, a * b))
</code></pre>
<h3>Force type conversion</h3>
<p>The function embedded in Python converts the variable type to deal with an inappropriate situation of the type, as follows:</p>
<ul>
<li><code>int()</code>: Converts a value or string to an integer, and you can specify an integer.</li>
<li><code>float()</code>: Converts a string to a floating point number.</li>
<li><code>str()</code>: Converts the specified object to a string form, which allows you to specify the code.</li>
<li><code>chr()</code>: Converts the integer to the string (a character) corresponding to the code.</li>
<li><code>ord()</code>: Converts a string (a character) to the corresponding code (integer number).</li>
</ul>
<h3>Comment</h3>
<pre><code class="language-python">import this
# 这是一些开发者留下的寄语

&quot;&quot;&quot;
这里也是注释，不过大部分人还是喜欢#
甚至使用 Command + / 添加批量的#
&quot;&quot;&quot;
</code></pre>
<h2>List</h2>
<p>The list is a sequence of elements and arrays that differ in order without any restriction on the type of element, but, as the opening of the collection says, there are few unrelated quantities that are combined, so the names of the lists are more terminological than word complex.</p>
<p>Actually, a string is a list.</p>
<pre><code class="language-python">bicycles=[&#39;trek&#39;,&#39;redline&#39;,&#39;cannondale&#39;]
# 这是列表的定义

print(bicycles)
# 这是直接打印列表本身 会有方括号和里面的逗号 以及字符串的引号

bicycles[0]
# 这是对列表元素的访问 还是从0开始 事实上我们也可以用 -1这些量表示倒着数 无论是使用还是修改都是从访问入手的

bicycles2 = bicycles[1:4]
#切片列表的一部分

bicycles3 = bicycles[:]
#切片全部就是复制
</code></pre>
<h3>Modify List</h3>
<pre><code class="language-python">bicycles[0]=bus
#对列表元素的修改就是通过访问实现的

bicycles.append(&#39;bus&#39;)
#方法append适用于在列表的尾端附件元素的 他是一个有参数的方法；我们可以先创建空列表 然后用append实现元素添加

bicycles.insert(0,&#39;bus&#39;)
#这个方法是对元素的插入 0代表位置参数 其余元素按需移动

del bicycles[0]
#del语句实现对元素的删除 他不是方法 后面的元素自动向前补位

print(bicycles.pop())
#pop方法是弹出 无参数时他会把栈顶的元素作为返回量 并且移除这个元素 所以是弹出 弹出用于你还想使用一下这个量的时候

print(bicycles.pop(0))
#弹出也可以指定索引 原理一模一样

bicycles.remove(&#39;trek&#39;)
#remove是为了处理你不知索引的情况 直接对元素下手 他只会删除第一个被查找到的值
</code></pre>
<h3>Organisation</h3>
<pre><code class="language-python">bicycles.sort()
#sort方法是对列表的永久性重新排序 他默认从小到大的顺序 如果有字母使用ASCII编码比较

print(sorted(bicycles))
#sorted函数是对列表的临时重新排序 很明显他有返回值
#sorted和sort虽然一个是方法一个是函数 但是均可以使用参数reverse=Ture进行翻转排序

bicycles.reverse()
#方法reverse是对列表原本顺序的永久性倒转 和排序无关

len(bicycles)
#函数len是对列表元素数量的测量
</code></pre>
<h3>Value List</h3>
<p>The list is very good for storing numbers, just like arrays, but Python has given us a lot of simpler tools to achieve him.</p>
<pre><code class="language-python">for value in range(1,5):
    print(value)
# 他会打印1 2 3 4  这是range这一函数的特征 不包括最后一位

range(6)
# 生成0到5 没有开头默认为0

numbers=list(range(1,5))
# list函数可以把range生成的结果列表化 此时numbers就是一个好用的列表

range(0,5,2)
# 第三个参数时生成列表对象时候的步长 不写的话默认为1

squares=[]
for value in range(1,11):
    squares.append(value**2)
print(squares)
#这是非常简单的一个思路 但是这样生成一个列表占用的代码行数是在是太多了 所以我们引入了列表解析

squares=[value**2 for value in range(1,11)]
print(squares)
#列表解析只是把循环和生成数组放到一起了 当觉得生成列表占用这么多行不划算的时候 列表解析自然的产生了
</code></pre>
<h3>Use part of the list</h3>
<pre><code class="language-python">players=list(range(1,11))
print(players[0:3])
#这是对列表一个部分的切片 按照切片的规定 他应该包括前边界 不含后边界

players[:3]
players[0:]
#不含开头和结尾索引的时候 默认从开头开始 或者到列表末尾结束

players[-3:]
#使用负数索引也是被允许的 切片也可以制定第三个变量 用于表示间隔 一般不写默认为1

sun_players=players[:]
#这表示对原本列表的复制 因为我们不允许直接对列表进行赋值

sun_players=players
#这实际上不会创建新的列表 和数组直接赋值一样 他只是一个索引 而非空间本身
</code></pre>
<h3>Division</h3>
<p>The only difference between the group and the list is that the group can't be modified when it's created.</p>
<pre><code class="language-python">players=(23,434)
# 这就是创建了一个元组
#访问和遍历与列表完全一致

players=(23,434)
players=(23,23434)
#虽然我们不能修改元组里面的变量 但重新给元组赋值是合法的


person = list(players)
print(person)
# 将元组转换成列表


fruits_list = [&#39;apple&#39;, &#39;banana&#39;, &#39;orange&#39;]
fruits_tuple = tuple(fruits_list)
print(fruits_tuple)
# 将列表转换成元组
</code></pre>
<h2>Branch</h2>
<h3>Elements of a basis</h3>
<p>In Python, the branch structure can be used to construct<code>if</code>、<code>elif</code>and<code>else</code>Keywords. So-called<strong>Keywords</strong>It's a word that has a special meaning, like...<code>if</code>and<code>else</code>It's a word that's used to construct a branch structure, which obviously cannot be used as a variable. Synchronising folder</p>
<pre><code class="language-python">for car in cars:
    if car==&#39;bwm&#39;:
        print(car.upper())
    else:
        print(car.title())
#Python中没有用花括号来构造代码块而是**使用了缩进的方式来表示代码的层次结构**
#一切判断的核心都是True和False == 和&gt; &lt; !=是最重要的判断语句

if &#39;Audi&#39;==&#39;audi&#39;
#python默认区分大小写 如果不需要可以用前面使用的upper和lower方法处理 这几个方法不改变原有变量

if 1&gt;2 and 3&gt;4:
if 1&gt;2 or 3&gt;4:
# 针对Bool型变量我们也有运算符号

for car in cars:
for car not in cars:
#这是关于事物是否在列表里面的判断语句

if car==&#39;bwm&#39;:
    print(car.upper())
if car==&#39;bwm&#39;:
    print(car.upper())
else:
    print(car.title())
if car==&#39;bwm&#39;:
    print(car.upper())
elif car==&#39;ct&#39;:
    print(car.lower())
else:
    print(car.title())
#以上是if语句的三种结构 他的自由程度也是非常高的
#最后提到一点 这三种语句块共同之处是只执行其中的一块 如果想要执行多次判断 要用大量的if语句 而非elif结构
</code></pre>
<h3>List and PEP 8</h3>
<pre><code class="language-python">if cars:
    *********
else:
    **********
#这个if语句会检验列表是不是空的 如果确实是空的 执行else语句内容
if age &lt; 4:
#这是PEP8向我们建议的书写方式，四个空间作为区分层级的符号，对于判断符和变量之间使用空格间隔一下
</code></pre>
<h2>♪ Gathering ♪ ♪ Dictionary</h2>
<p>The dictionary is a C-language structure that aims to make different types of arrays but associated ones into a unit that is accessible, not a list that typically stores lateral data; the collection is consistent with the mathematical aggregation.</p>
<h3>Gather!</h3>
<p>The collection in Python is consistent with the mathematical aggregation, does not allow for duplicate elements and can be interlocked, combined, differential, etc.
We're looking at some of the more statistically single items, set is a good choice.</p>
<pre><code class="language-python"># 创建集合的字面量语法
set1 = {1, 2, 3, 3, 3, 2}
print(set1)
print(&#39;Length =&#39;, len(set1))
# 创建集合的构造器语法(面向对象部分会进行详细讲解)
set2 = set(range(1, 10))
set3 = set((1, 2, 3, 3, 2, 1))
print(set2, set3)
# 创建集合的推导式语法(推导式也可以用于推导集合)
set4 = {num for num in range(1, 100) if num % 3 == 0 or num % 5 == 0}
print(set4)

#添加与删除集合中的元素
set1.add(4)
set1.add(5)
set2.update([11, 12])
set2.discard(5)
if 4 in set2:
    set2.remove(4)
print(set1, set2)
print(set3.pop())
print(set3)

# 集合的交集、并集、差集、对称差运算
print(set1 &amp; set2)
# print(set1.intersection(set2))
print(set1 | set2)
# print(set1.union(set2))
print(set1 - set2)
# print(set1.difference(set2))
print(set1 ^ set2)
# print(set1.symmetric_difference(set2))

# 判断子集和超集
print(set2 &lt;= set1)
# print(set2.issubset(set1))
print(set3 &lt;= set1)
# print(set3.issubset(set1))
print(set1 &gt;= set2)
# print(set1.issuperset(set2))
print(set1 &gt;= set3)
# print(set1.issuperset(set3))
</code></pre>
<h3>Use dictionary</h3>
<pre><code class="language-python">alien_0={&#39;color&#39;:&#39;green&#39;,&#39;points&#39;:5}
print(alien0[&#39;color&#39;])
print(alien0[&#39;points&#39;])
#这就是最基本的字典定义的方式和使用的方式 借助原本学习结构的思路我们能很快的理解 字典这个编程概念的重要性
#Python的字典是键值对的结构 键与值相关联 值的数据类型不进行任何限制 任何对象都可以

# 创建字典的构造器语法
items1 = dict(one=1, two=2, three=3, four=4)

print(alien0)
#直接对字典的打印会显示他的信息快照 和数组接近

# 通过键可以获取字典中对应的值，无论构造的时候键名，只要是字符串就需要引号
print(alien0[&#39;x_position&#39;])
print(alien0[&#39;x_position&#39;])

alien0[&#39;x_position&#39;]=0
alien0[&#39;y_position&#39;]=25
#这就是新增键值对的过程 我们新增了两个和位置相关的键并给了值

alien0={}
#这定义了空字典 方便我们后面添加键值对

alien0[&#39;x_position&#39;]=alien0[&#39;x_position&#39;]+5
#对值的修改是直接访问进行的 这很自然

del alien0[&#39;points&#39;]
#这会直接删除键和他对应的值 有时候留着这个键没用

alien0.clear()
# 清空字典
</code></pre>
<pre><code class="language-python">favorite_languages = {
    &#39;jen&#39;:&#39;python&#39;,
    &#39;sarah&#39;:&#39;c&#39;,
    &#39;edward&#39;:&#39;ruby&#39;,
    &#39;phil&#39;:&#39;python&#39;,
}
#这个由于字典太长才进行的分行定义 这是我们的一个习惯 别忘了逗号 不能省略的

print(alien0.get(&#39;height&#39;,&#39;No height value assigned&#39;))
#get也是一个用于访问字典的方法 他会用来处理想要访问的键不存在的问题 正常情况下 此时程序会崩溃 但是用get就可以避免崩溃
</code></pre>
<h3>Walking through Dictionary</h3>
<p>It's obvious that a dictionaries-- a cross-section of the dictionary is not a problem to be ignored, but how can a dictionary with only the keys?</p>
<pre><code class="language-python">for key,value in favorite_languages.items():
    print(f&quot;\n Key:{key}&quot;)
    print(f&quot;\n Value:{value}&quot;)
#这里的for循环使用两个变量表示键与值 items方法返回一个键值对列表 循环负责不断的使用列表

#变量的命名是随意的 items方法是本部分的重点
for name in favorite_languages.keys():
    print(name.title())
#这是用方法keys实现了对键的遍历 事实上方法keys也是把键单独进行列表化 事实上这种遍历提供了非常大的可操作性空间 在后面使用此时遍历到的键就可以实现对值的访问和操作

for name in sorted(favorite_languages.keys()):
#这个以让我们的遍历有顺序 如果要有其他顺序 再编制一些别的排序函数就可以了

for name in favorite_languages.values():
#values方法是对值的列表化 当然众所周知 列表化是从来不考虑重复的

# 对字典中所有键值对进行遍历，什么方法都不用也是可以接受的
for key in scores:
    print(f&#39;{key}: {scores[key]}&#39;)
</code></pre>
<h3>Embedded</h3>
<p>It's obvious that the dictionary and the list make it possible to learn so far about very useful data types, even if it's a good data structure, and the embedded suites between them are natural and complex.</p>
<pre><code class="language-python">alien_0={&#39;color&#39;:&#39;green&#39;,&#39;points&#39;:5}
alien_2={&#39;color&#39;:&#39;black&#39;,&#39;points&#39;:4}
alien_1={&#39;color&#39;:&#39;green&#39;,&#39;points&#39;:2}
aliens=[alien_0,alien_1,alien_2]
# 这是一个字典列表 列表的所有元素都是字典
pizza = {
    &#39;crust&#39;:&#39;thick&#39;,
    &#39;topping&#39;:[&#39;muanroom&#39;,&#39;cheese&#39;],
}
#很明显的 字典里面的某个元素是列表 这也是按需设置的 访问的时候按照程序设计的逻辑就可以
#字典当然可以和字典嵌套 好好设计结构就可以 千万别忘了考虑访问
#记住 列表一般存储同类型信息 字典用来存储不同类型的信息 按照需求进行嵌套
</code></pre>
<h2>User Input & Cycle</h2>
<h3>Walk Through List</h3>
<pre><code class="language-python">bicycles=[&#39;trek&#39;,&#39;redline&#39;,&#39;cannondale&#39;]
for bicycle in bicycles:
    print(bicycle)
#这就是for in循环的基本样式 ，针对列表实现for in 循环很自然
</code></pre>
<p>The colon and indentation are characteristic of the Python cycle, and the indentation part ends the cycle naturally, which means that the indentation of Python is closer to the natural language for in-cycle based on the list of accomplishments.</p>
<pre><code class="language-python"># 如果应该缩进的地方完全没有缩进 Python会提供报错提示
# 如果只是漏掉了一部分需要的缩进 这属于逻辑错误 编辑器无法排查
# 不需要缩进的地方在Python严禁缩进 因为缩进是Python识别语言的依据之一
# 不要忘记冒号 这是重要的标识符
</code></pre>
<h3>Input function</h3>
<pre><code class="language-python">message = input(&quot;Tell me some thing : &quot;)
#input函数只接受一个参数 也就是提示prompt 他会让用户键入数据 并且返回给函数返回值

prompt = &quot;****************&quot;
prompt+=&quot;\n****&quot;
name=input(prompt)
#这告诉我们一定要给与用户清晰的prompt 上面的字符串加法是一种连接手段 分行的提示让提示更加清楚 冒号的存在让键入更加舒适

age = int(input())
#input函数是默认用字符串来处理内容的 int函数是为了把它变成数 数字是方便各种使用的
</code></pre>
<h3>♪ While the cycle ♪</h3>
<p>For in-the-list-based running in Python, of course, he's useful, but sometimes for only for the cycle is not enough if we want to construct a loop structure that does not know the number of cycles.<code>while</code>Loop.<code>while</code>Cycle by a capable of generating or converting<code>bool</code>Value expression to control the cycle, the value of the expression is<code>True</code>;expression is<code>False</code>.</p>
<pre><code class="language-python">currrent = 1
while current &lt;= 5:
    print(current)
    current += 1
#这就是while循环的核心 设定停止条件 原本C中的for循环也是使用停止条件的 但是Python对这方面进行了改动，改为了一个针对列表的操作

message = &quot;&quot;
while message != &#39;quit&#39;:
    message = input(输入&#39;quit&#39;退出循环)
    if message != &#39;quit&#39;:
        print(message)
#这是一个非常常用的退出循环 死去的记忆死灰复燃

while flag != Ture:
    ***********
#使用flag标志用于多个因素均会导致这个循环出现变化的情况 flag的运用是非常重要的

while flag:
#这是简化的形式flag形式，flag==Ture的时候自动脱离

while Ture:
    message = input()
    if message == &#39;quit&#39;:
        break
    else:
        print(message)
#使用break命令可以随时跳出循环 开头的while ture实际上是无限循环的意思 这种代码一定要搭配跳出手段 无限循环可不是什么好事情

currrent = 1
while current &lt;= 10:
    current += 1
    if current % 2 == 0:
        continue
    print(current)
#这是对continue的使用 注意 break 和 continue 会选择离他最近的循环 并且只选择这一个循环
</code></pre>
<h3># When the dictionaries and the list #</h3>
<p>Why do we draw this subheading for the cycling is very compatible with the list? Far</p>
<p>For the loop is certainly very effective, but we don't suggest changing elements in the loop in the for-cycle, which should be used only for the walk-through.</p>
<pre><code class="language-python">while list:
    ***********
#如上所示的while循环将会一直运行 直到list变成一个空列表 也就是我们在后面的代码块中要记得不断减少列表中的元素 pop方法 del都可以

while &#39;cat&#39; in pets:
    pets.remove(&#39;cats&#39;)
#这也是一个重要的判断 他其实起到了remove移除多个元素的作用 不要忘记in这个在Python中重要的变量

#键值对的while循环也是容易的 事实上我们大多会引入flag来帮助循环 灵活运用前面提到的方法就可以
</code></pre>
<h2>Functions</h2>
<h3>Define Functions and their Parameters</h3>
<p>The function is the important thing that increases the efficiency of the writing.</p>
<pre><code class="language-python">def greet_user():
    print(&quot;hello!&quot;)
#这个非常简单的代买解释了Python函数的最简单定义方式 也就是def语句 后面的greet_uesr是我们的函数名 缩进部分是函数体 括号里面是可有可无的参数 调用函数一定要带括号 实际上我们使用的基本函数都是别人帮你定义好的

def greet_user(name):
    print(f&quot;hello! {name.title()}&quot;)
#添加参数也非常简单 此时调用函数的时候也要记得加函数需求的变量 否则肯定会报错
#多个参数的顺序方面 我们更应该灵活的选取调用方法 位置调用更加的简单迅速  关键字在参数数量增加的时候才会更加的使用

def greet_user(first_name,last_name=&#39;Wang&#39;):
#这其实给了参数默认值 此时哪怕你在使用的时候传入的参数不够 Python也不会报错而是使用默认参数 当然你传入新的参数的时候会按照传入的执行
#位置调用其实引入了另一个问题 导致有时候使用参数会用如 greet(,Jue) 这就是设定了默认参数并且选择了位置调用 不过熟能生巧 出错了积极修改就好 一般会把最容易不选的参数放在最后 来给调用者省事
</code></pre>
<h3>Functions with returned values and parameters</h3>
<p>Obviously we've seen a function with a return value, but the previous function was not written to say how to return the value.</p>
<pre><code class="language-python">def greet_user():
    print(&quot;hello!&quot;)
    return 0
#这就是最简单的有返回值的函数 我们不对返回值的类型进行限制（列表 字典也可以） 编写的时候会发现 一旦return 编辑器的换行辅助就会帮助我们重启一行 意思是return是一个函数的结尾 之后别再编写了 （使用函数的返回值是很重要的）

def build_person(first_name,last_name,age=None):
#None 是一个特殊的占位符 他的意思是什么都没有 条件测试的时候None是False

def greet_users(names):
    for name in names:
        print(f&quot;Hello! {name}&quot;)
#函数对于直接传入或者传出列表没有任何特殊要求 但是我们要知道 传入列表的函数没有形式参数的概念 被改变的就是原始列表

function_name(names[:])
#Python程序员总是要求许多 使用以上的调用方式就可以避免原始列表的修改 此时只会影响副本 注意 大型列表的拷贝是一件非常耗费资源的事情 不要随便这样拷贝 除非一定需要

</code></pre>
<h3>Any quantitative parameters</h3>
<pre><code class="language-python">def make_pizza(*toppings):
  print(toppings)
#这个函数的定义方法允许我们在函数内使用任意数量的实参 并且把他们存储在一个元组里面 这是非常有用的

def make_pizza(size,*toppings):
#混合使用当然是允许的 在位置实参中 我们肯定是先满足前面的 最后把所有多的放到后面的整个toppings里面形成元组 当然使用关键词也是一个不错的主意

def build_profile(first,last,**info):、
build_profile(Wang,Jue,location=&#39;LUOYANG&#39;,filed=&#39;MATH&#39;)
#容易看到我们又多了一个星号 此时info会接受任意数量的键值对并生成字典
</code></pre>
<h3>Manage functions with modules</h3>
<p>(a) On modularization: since Python does not have the concept of reloading functions, the latter definition would cover the previous definition, which means that only one function with the same name actually exists;</p>
<p>How then can such a naming conflict be resolved? The answer is simple, and each file in Python represents a module, and we can have a function with the same name in different modules, and we can use it.<code>import</code>Keyword import specified modules to distinguish which module functions are to be used</p>
<pre><code class="language-python">import pizza
pizza.make_pizza()
#import命令能够让我们引入一个新的模块（后缀为.py的文件） 使用import后 模块里面函数之类的就可以间接调用(要前缀) 这是跨文件编程的第一步

from pizza import make_pizza
#导入单个函数的语法 这是后调用函数直接用make_pizza就可以 不需要pizza.

from pizza import make_pizza as mp
import pizza as p
#这是对函数和模块起了一个其他的名字 后面直接使用就可以

from pizza import *
#这是复制所有函数到这个文件里面 可以直接用名字调用 不需要pizza. 不过请尽量节制的使用这个导入方法
</code></pre>
<p>We need to introduce an important judgement on modularization.</p>
<pre><code class="language-python">if __name__ == &#39;__main__&#39;:
#用来判定是否是直接执行的模块

#在模块化中 这是重要的判定, 如果模块包含可执行代码 那么在导入模块的时候他们会被运行, 我们需要避免导入某个模块时错误的执行这些代码
</code></pre>
<h3>About Variable Action Fields</h3>
<p>No function defined This is a global variable
Defines the variable in a function as a local variable
But if the function is embedded, a local variable is accessed by another domain, which is called the nested domain.
The function will search for variables according to the order of local domain, embedded domain, embedded field, internal field, which is the one that is the highest priority, and now it should be understood that we have an inaccurate description and understanding of the variable field.</p>
<pre><code class="language-python">global a
nonlocal a
#他们的意思分别是访问全局作用域变量a 和嵌套作用域变量a
#如果没有这样的变量 就创建一个
#降低对全局变量的依赖是降低程序耦合程度的关键

def main():
    # Todo: Add your code here
    pass


if __name__ == &#39;__main__&#39;:
    main()
#这样的一份主程序 就符合一个专业的开发者的习惯
</code></pre>
<h2>Regular Expression</h2>
<p>When preparing a program or web page to process strings, there is often a need to find strings that meet certain complex rules, and the regular expression is a tool to describe these rules, in other words, a regular expression is a tool that defines the matching pattern of a string (how to check whether a string has parts matching a pattern or to extract or replace those that match a pattern in a string).</p>
<p>Regular expression is understood as a set of languages used to match strings, and Python supports regular expression.</p>
<h3>Regular Expression</h3>
<table>
<thead>
<tr>
<th>Symbol</th>
<th>Explanation</th>
<th>Example:</th>
<th>Annotations</th>
</tr>
</thead>
<tbody><tr>
<td>.</td>
<td>Match Any Character</td>
<td>b.t</td>
<td>Matches bat / but / b#t / b1t, etc.</td>
</tr>
<tr>
<td>\w</td>
<td>Match letters/numbers/underlines</td>
<td>b\wt</td>
<td>Matches bat / b1t / b t, etc.<br>But not a match for b#t</td>
</tr>
<tr>
<td>\s</td>
<td>Match whitespace characters (including \\r, \n, \t, etc.)</td>
<td>love\syou</td>
<td>♪ Can match you ♪</td>
</tr>
<tr>
<td>\d</td>
<td>Match numbers</td>
<td>\d\d</td>
<td>It matches 01/ 23/ 99 etc.</td>
</tr>
<tr>
<td>\b</td>
<td>Boundaries Matching Words</td>
<td>\bThe\b</td>
<td></td>
</tr>
<tr>
<td>^</td>
<td>Start of matching string</td>
<td>^The</td>
<td>You can match the string that begins with The</td>
</tr>
<tr>
<td>&#36;</td>
<td>End of matching string</td>
<td>.exe&#36;</td>
<td>You can match the string at the end of.exe.</td>
</tr>
<tr>
<td>\W</td>
<td>Match non-letter/number/underline</td>
<td>b\Wt</td>
<td>Matches b#t / b@t etc.<br>But not matching But / b1t / b t etc.</td>
</tr>
<tr>
<td>\S</td>
<td>Match non-empty characters</td>
<td>love\Syou</td>
<td>It matches love#you and so on.<br>But not to match you</td>
</tr>
<tr>
<td>\D</td>
<td>Match Non-numbers</td>
<td>\d\D</td>
<td>Matches 9a / 3# / 0F etc.</td>
</tr>
<tr>
<td>\B</td>
<td>Match non-word boundary</td>
<td>\Bio\B</td>
<td></td>
</tr>
<tr>
<td>[]</td>
<td>Match any single word from the character set Arguments</td>
<td>[aeiou]</td>
<td>You can match any vowel letter</td>
</tr>
<tr>
<td>[^]</td>
<td>Match any single word that is not in the character set Arguments</td>
<td>[^aeiou]</td>
<td>Matches any non-metaphonic letter</td>
</tr>
<tr>
<td>*</td>
<td>Match 0 or more times</td>
<td>\w*</td>
<td></td>
</tr>
<tr>
<td>+</td>
<td>Matches one or more times</td>
<td>\w+</td>
<td></td>
</tr>
<tr>
<td>?</td>
<td>Match 0 or 1 times</td>
<td>\w?</td>
<td></td>
</tr>
<tr>
<td>{N}</td>
<td>Match N</td>
<td>\w{3}</td>
<td></td>
</tr>
<tr>
<td>{M,}</td>
<td>Match at least M times</td>
<td>\w{3,}</td>
<td></td>
</tr>
<tr>
<td>{M,N}</td>
<td>Match at least M times up to N</td>
<td>\w{3,6}</td>
<td></td>
</tr>
<tr>
<td>|</td>
<td>Branch</td>
<td>foo|bar</td>
<td>You can match a foo or a bar.</td>
</tr>
<tr>
<td>(?#)</td>
<td>Comment</td>
<td></td>
<td></td>
</tr>
<tr>
<td>(exp)</td>
<td>Match exp and capture to automatically named group</td>
<td></td>
<td></td>
</tr>
<tr>
<td>(?&lt;name&gt;exp)</td>
<td>Matches exp and captures to a group named name</td>
<td></td>
<td></td>
</tr>
<tr>
<td>(?:exp)</td>
<td>Matches exp but does not capture matching text</td>
<td></td>
<td></td>
</tr>
<tr>
<td>(?=exp)</td>
<td>Match position in front of exp</td>
<td>\b\w+(?=ing)</td>
<td>It's a match for I.&#39;♪ I'm danc in dancing ♪</td>
</tr>
<tr>
<td>(?&lt;=exp)</td>
<td>Match position behind exp</td>
<td>(?&lt;=\bdanc)\w+\b</td>
<td>I love dancing and reading</td>
</tr>
<tr>
<td>(?!exp)</td>
<td>Matching the back is not exp's position.</td>
<td></td>
<td></td>
</tr>
<tr>
<td>(?&lt;!exp)</td>
<td>Matching a position that is not exp in front</td>
<td></td>
<td></td>
</tr>
<tr>
<td>*?</td>
<td>Repeat any time, but with as little repetition as possible</td>
<td>a.*b<br>a.*?b</td>
<td>Apply regular expression to aabab, which matches the whole string aabab, and the latter to both aab and ab</td>
</tr>
<tr>
<td>+?</td>
<td>Repeated once or more, but as little as possible</td>
<td></td>
<td></td>
</tr>
<tr>
<td>??</td>
<td>0 or 1 repeats, but as little as possible</td>
<td></td>
<td></td>
</tr>
<tr>
<td>{M,N}?</td>
<td>Repeat M to N, but as little as possible.</td>
<td></td>
<td></td>
</tr>
<tr>
<td>{M,}?</td>
<td>Repeated more than M, but less than possible</td>
<td></td>
<td></td>
</tr>
</tbody></table>
<blockquote>
<p><strong>Note:</strong> If the character to be matched is a special character in a regular expression, then you can transpose it using \\, for example, to match the decimal points to be written as \\..., because writing directly... will match any character; similarly, to match a round brackets must be written as \\ (and \\), otherwise the brackets will be considered as a grouping in a regular expression.</p>
</blockquote>
<h3>Regular expression in Python</h3>
<p>Python provides the re module to support regular expression-related operations, and the core functions in the re module are listed below.</p>
<table>
<thead>
<tr>
<th>Functions</th>
<th>Annotations</th>
</tr>
</thead>
<tbody><tr>
<td>compile(pattern, flags=0)</td>
<td>Compile regular expression to return regular expression object</td>
</tr>
<tr>
<td>match(pattern, string, flags=0)</td>
<td>Matching string with regular expression successfully returning the match object or not returning the Noone</td>
</tr>
<tr>
<td>search(pattern, string, flags=0)</td>
<td>Search for the first regular expression in a string</td>
</tr>
<tr>
<td>split(pattern, string, maxsplit=0, flags=0)</td>
<td>Split strings with the mode separator specified in regular expression</td>
</tr>
<tr>
<td>sub(pattern, repl, string, count=0, flags=0)</td>
<td>Replace the pattern matching the regular expression in the original string with the specified string.</td>
</tr>
<tr>
<td>fullmatch(pattern, string, flags=0)</td>
<td>Full Match (start to end of string) version of the Match function</td>
</tr>
<tr>
<td>findall(pattern, string, flags=0)</td>
<td>Find all modes of string matching regular expression Returns the string list</td>
</tr>
<tr>
<td>finditer(pattern, string, flags=0)</td>
<td>Find all modes of string matching regular expressions and return an iterative device</td>
</tr>
<tr>
<td>purge()</td>
<td>Clear the cache of hidden regular expressions</td>
</tr>
<tr>
<td>re.I / re.IGNORECASE</td>
<td>Ignore case matching tags</td>
</tr>
<tr>
<td>re.M / re.MULTILINE</td>
<td>Multiline Matching Tags</td>
</tr>
</tbody></table>
<blockquote>
<p><strong>Note:</strong> These functions in the re module mentioned above can also be replaced by regular expression objects in actual development. If a regular expression needs to be repeated, it is certainly a wiser option to compile regular expression first through the function of the compile and create regular expression objects.</p>
</blockquote>
<h3>Examples</h3>
<pre><code class="language-python">&quot;&quot;&quot;
验证输入用户名和QQ号是否有效并给出对应的提示信息

要求：用户名必须由字母、数字或下划线构成且长度在6~20个字符之间，QQ号是5~12的数字且首位不能为0
&quot;&quot;&quot;
import re


def main():
    username = input(&#39;请输入用户名: &#39;)
    qq = input(&#39;请输入QQ号: &#39;)
    # match函数的第一个参数是正则表达式字符串或正则表达式对象
    # 第二个参数是要跟正则表达式做匹配的字符串对象
    m1 = re.match(r&#39;^[0-9a-zA-Z_]{6,20}&#36;&#39;, username)
    if not m1:
        print(&#39;请输入有效的用户名.&#39;)
    m2 = re.match(r&#39;^[1-9]\d{4,11}&#36;&#39;, qq)
    if not m2:
        print(&#39;请输入有效的QQ号.&#39;)
    if m1 and m2:
        print(&#39;你输入的信息是有效的!&#39;)


if __name__ == &#39;__main__&#39;:
    main()
</code></pre>
<p>It is written in the regular expression (replacing the original string) and the term "original string " is the original meaning of each character in the string, which is more directly the absence of the so-called transliteration in the string. Arguments</p>
<p>If we are to develop reptile applications, then regular expression must be a very good assistant, because it helps us quickly to find some of the patterns we have specified and extract the information we need, and of course it may not be easy for starters to develop a proper regular expression right (some of the usual regular expressions can be found directly online).</p>
<h2>File and anomaly</h2>
<p>Neither this nor the next chapter will introduce new programming knowledge.</p>
<h3>Read data from file</h3>
<pre><code class="language-python">#Python默认从当前运行的程序所在的位置寻找要打开的文件
with open(&#39;file_name.txt&#39;,encoding=&#39;utf-8&#39;) as file_object:
    contents = file_object.read()
print(contents.rstrip())
#一切文件操作的核心都是打开文件 open寻找并打开了参数内的文件 并且把这个对象返回给了file_object 关键词with会在不需要访问文件后将其关闭 此时我们就不需要使用close() 避免了一些问题 Python会帮我们处理关闭文件的问题
#拥有文件对象以后（面向对象的程序设计）我们对他使用了方法read 他会读取这个文件的全部内容 并且用字符串返回他 请注意read函数会在读取到的所有内容后面添加一个空字符串 所以添加了rstrip的方法进行处理
#encoding部分是对文件读取编码的要求 如果默认编码和文件使用的编码不一样 直接读取会出现乱码
with open(&#39;text_files/filename.txt&#39;) as file_object:
#这是相对文件路径寻找 是从目前程序所在的文件夹的子文件夹寻找 我们使用/ 而不是\这一标准的文件路径符号是因为代码的特殊规定
with open(&#39;/home/ehmatthes/others/text_files/filename.txt&#39;) as file_object
#这是绝对文件路径寻找 要从盘符开始
with open(&#39;file_name.txt&#39;) as file_object:
    for line in file_object:
        print(line.rstrip())
#这是逐行读取文件的内容 rstrip的存在是因为 逐行读取是包括结尾的换行符的 print又会增加换行符号 导致换行符号过多
with open(&#39;file_name.txt&#39;) as file_object:
    lines = file_object.readlines()
print(lines)
#read函数打开的文件在函数结束后自动关闭 前面我们是吧整个文件当成一个巨大的字符串存储 实际上readlines方法能够把它转换成一个按行为单位的列表 当然别忘了rstrip
#有了上面研究的这几个方法 使用txt文件应该不是一个困难的事情了 注意 Python 把txt内的所有文本都解读为字符串 要进行别的数字运算的时候要格式转换
</code></pre>
<h3>Writing files</h3>
<pre><code class="language-python">with open(&#39;file_name.txt&#39;,&#39;w&#39;) as file_object:
    file_object.write(&#39;I love programming.\n&#39;)
#对于写入文件 我们要给Python的open额外的参数 也就是这个w 它意味着我们打开后可能对文件进行写入
#实际上 &#39;r&#39;读取模式 &#39;w&#39;写入模式 &#39;a&#39;附加模式 &#39;r+&#39;读写模式 不写参数 默认&#39;r&#39;
#写入模式w要专门注意 如果要写入的文件不存在 Python会创建他 如果已经存在 Python会先对他进行格式化
#write方法就是写入字符串使用的 Python只能写入字符串 他不会默认加上换行符 所以我们一般会人为加上
with open(&#39;file_name.txt&#39;,&#39;a&#39;) as file_object:
    file_object.write(&#39;I love programming.\n&#39;)
#附加模式不会对原始文件格式化 他会在原本的文件后面添加东西
#对文件的读取和写入本质上与对终端的读取和写入没有差别
</code></pre>
<h3>Unusual</h3>
<p>It's impossible to get a code wrong, most times, the code stops running when it's wrong, but it's not a good experience for users to use it to crash, so Python offers this special object that's used to handle these errors when he encounters a glitch error, and if we do the code for dealing with an abnormal object, the program will continue to run, not backtrack.</p>
<p>Use the try-except code block to handle it, and remember to tell the user what we've done, not a trackback that a normal person can't understand.</p>
<pre><code class="language-python">try:
    print(5/0)
except ZeroDivisionError:
    print(&quot;You can&#39;t divide by zero&quot;)
#这就是最基础的模块 如果try部分的代码运行时出现了错误 python会选择对应的except代码块并运行 这样的话我们就避免了trackback被显示出来 并且程序可以继续向后面运行 事实上 trackback的显示无论是对于使用者还是专业程序员看到都不好 使用者会感到疑惑 专业的程序员会从你的trackback中看到很多程序相关的信息
try:
    answer = a/b
except ZeroDivisionError:
    print(&quot;You can&#39;t divide by zero&quot;)
else:
    print(answer)
#else 代码块是在try运行中没有发现问题才继续执行的代码 他依赖于try的正确执行 这是他和直接写在后面的代码的区别
except ZeroDivisionError:
    pass
#这适用于你不想给使用者看到任何信息的情况 pass的意思就是什么都不执行 这往往称为静默失败 是否选择静默失败是程序设计者应该仔细思考的 显示一些没用的信息有时候会减小程序的易用性
#研究异常不是说让自己编写的代码出错 而是为了避免一些外部因素影响程序的可用性 比如用户输出错误 文件被以外删除 网络连接异常等等 程序的设计者往往凭借自己的经验判断程序什么时候可能出现异常 并且加以处理
</code></pre>
<h3>Store data</h3>
<p>Any program that is shut down always has to store some data into the file.</p>
<pre><code class="language-python">import json
numbers = [1,2,3,42,1]
with open (&#39;filename.json&#39;,&#39;w&#39;) as f:
    json.dump(numbers,f)
#实际上json是一个很实用的数据格式 我们把一个列表借助dump函数写到了一个json文件里面 dump写入函数支持两个参数 要写入的内容和文件名
with open (&#39;filename.json&#39;) as f:
    numbers = json.load(f)
#load函数接受一个参数 文件名
#事实上使用这个json格式能够存储很多txt不能存储的数据类型 他对成程序设计者非常有用
#熟练地使用前面所学习的内容 我们就可以开始设计一些比较复杂的程序了 要记住我们的核心编程思想 一旦主程序变得臃肿 就要重构代码 建立函数进行封装 通过完善的注释来让程序有着优良的可读性
</code></pre>
<h2>Test</h2>
<p>The purpose of the test code is very simple, and you want to try to do the same without any bugs, or even if there's a problem, your code will work as expected.</p>
<pre><code class="language-python">#首先我们需要被测试的代码
def get_formatted_name(first,last):
    full_name = f&quot;{first} {last}&quot;
    return full_name.title()
#如何测试这个函数能不能像我们设计的一样工作 当然我们可以选择手动输入 但是这太麻烦了 Python为我们提供了一些自动测试的有效方法
</code></pre>
<p>The Python Standard Library provides the module unitst as a code test tool  <strong>Unit Test</strong>There's something about measuring functions that's fine. <strong>Test Example</strong>It's a set of unit tests. <strong>Overwrite All</strong>The test is a set of tests that we're normally only going to cover when the project is widely used.</p>
<pre><code class="language-python">import unittest
from fuction import get_formatted_name
#引入测试用模块和待测试函数
class NamesTestCase(unittest.TestCase):
    def test_first_last_name(self):
        formatted_name = get_formatted_name(&#39;janis&#39;,&#39;joplin&#39;)
        self.assertEqual(formatted_name,&#39;Janis Joplin&#39;)

#创建一个类（测试用例）用于储存测试单元 命名一定要能看出来他是在测试什么 注意 这个类继承了一个unittest模块中的类
#对于这个测试才被使用的类 我们只为他创建里除继承以外的一个方法 运行这个方法的时候 我们会使用一次待测试函数 然后借助unittest本来就存在的断言方法核实我们期望得到的结果和实际运行生成的结果 运行这段代码 我们会显示一个明显的测试提示无论通过与否 告诉你测试的结果如何 现在我们就编写了一个测试单元 随时可以使用这个测试单元对函数进行检验
#很明显的 仅仅一个测试单元肯定不能满足我们的要求 实际上测试工程师的任务就是编写测试单元 整合各种测试用例让架构师使用
	def test_first_last_middle_name(self):
        formatted_name = get_formatted_name(&#39;wolfgang&#39;,&#39;mozart&#39;,&#39;amadeus&#39;)
        self.assertEqual(formatted_name,&#39;Wolfgang Mozart Amadeus&#39;)
#我们又编写了一个测试单元 实际上测试模块的规则非常的特殊 一切test_开头的方法都会被自动调用 当这个模块被使用时 这就是为了让架构师能够轻松地调用 一切测试单元的方法名都要是描述性的 无论长度 你要让架构师能快速的定位是哪里的程序出了问题
</code></pre>
<p>The test of the function is now to consider whether the test class works well.</p>
<pre><code class="language-python">#下面是一些比较常用的unittest模块断言方法
assertEqual(a,b)
assertNotEqual(a,b)
assertTure(a)
assertFalse(a)
assertIn(item,list)
assertNotIn(item,list)
#他们分别合适相等与否 真假与否 所有与否 实际上就是一些布尔运算 只是在设计测试的时候比较重要 断言是测试结果是否通过的判别
</code></pre>
<p>The test classes are often the same methods in the test classes, so there are many places in which the test functions are similar to the same starting-- the exact same way in which the test classes are defined -- and the same way in which they are used as part of the function of the class or function in the method -- actually the test examples are the complete use of the original function or class, and the final use of the assertion to judge it is actually the same code, as follows:</p>
<pre><code class="language-python">class AnonymousSurvey():
    &quot;&quot;&quot;收集匿名调查问卷的答案&quot;&quot;&quot;

    def __init__(self, question):
        &quot;&quot;&quot;存储一个问题，并为存储答案做准备&quot;&quot;&quot;
        self.question = question
        self.responses = []

    def show_question(self):
        &quot;&quot;&quot;显示调查问卷&quot;&quot;&quot;
        print(self.question)

    def store_respond(self, new_response):
        &quot;&quot;&quot;存储单份调查答卷&quot;&quot;&quot;
        self.response.append(nes_response)

    def show_result(self):
        &quot;&quot;&quot;显示收集到的所有答卷&quot;&quot;&quot;
        print(&quot;Survey results:&quot;)
        for response in self.responses:
            print(&#39;- &#39; + response)
# A new file
import unittest
from survey import AnonymousSurvey

class TestAnonymousSurvey(unittest.TestCase):
     &quot;&quot;&quot;针对AnonymousSurvey类的测试&quot;&quot;&quot;

     def test_single_response(self):
         &quot;&quot;&quot;测试单个答案会被妥善的存储&quot;&quot;&quot;
         question = &quot;What language did you first learn to speank?&quot;
         my_survey = AnonymousSurvey(question)
         my_survey.store_response(&#39;English&#39;)

         self.assertIn(&#39;English&#39;, my_survey.response)
     def test_store_three_response(self):
        &quot;&quot;&quot;测试三个答案会被被妥善地存储&quot;&quot;&quot;
        question = &quot;What language did you first learn to speak?&quot;
        my_survey = AnonymousSurvey(question)
        response = [&#39;English&#39;, &#39;Spanish&#39;, &#39;Mandarin&#39;]
        for response in responses:
	my_survey.store_response(response)

	for response in responses:
	self.assertIn(response, my_survey.response)
#容易发现测试类总是需要建立好多次实例  一直CV确实无聊 所有有个setup
        def setUp(self):
            question = &quot;What language did you first learn to speak?&quot;
            self.my_survey = AnonymousSurvey(question)
            self.responses = [&#39;English&#39;, &#39;Spanish&#39;, &#39;Mandarin&#39;]
#创建一个调查对象和一组答案，共使用的测试方法使用 按照管理建立Setup要在所有测试用例之前
</code></pre>
<p>Python's test will give some feedback on running.</p>
<p>Each unit test is completed with a number of characters printed by printing a stop point at the time testing triggers an error printing E when an assertion that F has failed</p>
<p><strong>The foundational elements of Python are over here, and we need to finish some projects as a practical exercise and to review the knowledge we have learned, but then we don't have to continue with this document.</strong></p>
