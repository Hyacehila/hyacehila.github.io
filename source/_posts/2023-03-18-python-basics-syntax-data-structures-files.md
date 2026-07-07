---
title: "Python 基础：语法、数据结构与文件处理"
title_en: "Python Basics: Syntax, Data Structures, and File Handling"
date: 2023-03-18 21:32:22 +0800
categories: ["Programming", "Python"]
tags: ["Python"]
author: Hyacehila
excerpt: "整理变量、字符串、列表、分支、集合、字典、循环、函数、正则表达式、文件异常与测试等入门内容。"
excerpt_en: "Covers variables, strings, lists, branches, sets, dictionaries, loops, functions, regular expressions, files, exceptions, and testing."
mathjax: false
hidden: true
permalink: '/blog/2023/03/18/python-basics-learning-notes/'
---


和C语言学习的不同之处，这次的整体构建会更加的重视实践，一个比较优秀的基础和实践的练习才能够保证有比较充足的经验，此时看各种官方的文档就会更加的容易了

## 开篇
### Python应用领域和职业发展分析

简单的说，Python是一个“优雅”、“明确”、“简单”的编程语言。

-   学习曲线低，非专业人士也能上手
-   开源系统，拥有强大的生态圈
-   解释型语言，完美的平台可移植性
-   动态类型语言，支持面向对象和函数式编程
-   代码规范程度高，可读性强

Python在以下领域都有用武之地。

-   后端开发 - Python / Java / Go / PHP
-   DevOps - Python / Shell / Ruby
-   数据采集 - Python / C++ / Java
-   量化交易 - Python / C++ / R
-   数据科学 - Python / R / Julia / Matlab
-   机器学习 - Python / R / C++ / Julia
-   自动化测试 - Python / Shell

作为一名Python开发者，根据个人的喜好和职业规划，可以选择的就业领域也非常多。

-   Python后端开发工程师（服务器、云平台、数据接口）
-   Python运维工程师（自动化运维、SRE、DevOps）
-   **Python数据分析师（数据分析、商业智能、数字化运营）**
-   **Python数据挖掘工程师（机器学习、深度学习、算法专家）**
-   Python爬虫工程师
-   Python测试工程师（自动化测试、测试开发）

给初学者的几个建议：

-   Make English as your working language. （让英语成为你的工作语言）
-   Practice makes perfect. （熟能生巧）
-   All experience comes from mistakes. （所有的经验都源于你犯过的错误）
-   Don't be one of the leeches. （不要当伸手党）
-   Either outstanding or out. （要么出众，要么出局）

### 简单的描述和优缺点
1.  简单明了，学习曲线低，比很多编程语言都容易上手。
2.  开放源代码，拥有强大的社区和生态圈，尤其是在数据分析和机器学习领域
3.  解释型语言，天生具有平台可移植性，代码可以工作于不同的操作系统  Cpython仍然占据主流 也就是解释器是C写的
4.  对两种主流的编程范式（面向对象编程和函数式编程）都提供了支持。
5.  代码规范程度高，可读性强，适合有代码洁癖和强迫症的人群。

## 变量和简单的数据类型

### 一些变量命名规则
*  变量名由字母、数字和下划线构成，数字不能开头
* 对大小写敏感
* 含空格禁止
* 避免使用保留函数和语法关键字
* 小写l和大写O容易被误读为数字 谨慎使用
* 大写全部字母表示常量 普通变量全部小写使用下划线连接
* 受保护的实例属性用单个下划线开头（后面会讲到）
* 私有的实例属性用两个下划线开头（后面会讲到）

```python
# type函数可以查看变量的类型 由于不要求提前声明变量，我们可以直接进行赋值，类型交给解释器选择
type(a)
```
### 数

Python是一个相当自由的语言 没有各种数定义的约束让我们使用起来非常的轻松 无论是整数还是浮点都可以自由使用 Python会提供给你一个让你满意的答案

比较重要的和C的区别是 他拥有 ** 表示乘方 并且 / 表示普通的除法 %表示余数
### 字符串
#### 简单的字符串介绍

使用单引号 双引号括起来的是字符串 这些不同的引号规则是为了你能在句子里面顺利的使用引号

```python
s1 = 'hello, world!'
s2 = "hello, world!"
```

可以在字符串中使用`\`（反斜杠）来表示转义，也就是说`\`后面的字符不再是它原来的意义,在`\`后面还可以跟一个八进制或者十六进制数来表示字符
```python
s1 = '\141\142\143\x61\x62\x63'
s2 = '\u9a86\u660a'
print(s1, s2)
#各种编码格式都可以通过\进行显示
```

如果想要使用 ' \ '在字符串里，则需要给他也来一层转义有
```python
s1 = "\\dsadasdasd\\"
```
#### 字符串方法

方法是Python对数据执行的操作 他和函数不一样  具体的区别我们会在以后再说

对字符串的基础方法有

``` python
.title() .upper() .lower()
name="ada fesasd"
print(name.title())
# 他们会对字符串进行首字母大写  全部大写 全部小写
name.split()
# 把空格当做划分符 将字符串拆解成字符列表
print('ll' in s1)
# 判断字符串的包含关系
print(str2[2:5])
# 字符串的正常切片 和列表一样处理
```

#### 字符串中的变量
如想想要在字符串的输出中增加一些变量的内容，就需要这点

``` python
first_name="ada"
last_name="love"
full_name=f"{first_name} {last_name}"
print(full_name)
#f 是 format 的缩写 是对字符串个格式设置 他会替换变量生成新的字符串
```

`f 字符串`是在python3.6以后引入的 他之前的格式是 是一个关于字符串的方法

``` python
full_name="{} {}".format(first_name,last_name)
```

这类似于我们前面研究的使用 `%d` 等占位符的字符串输出，如
```python
a, b = 5, 10
print('%d * %d = %d' % (a, b, a * b))
```
### 强制类型转换
Python中内置的函数对变量类型进行转换，用于处理类型不合适的情况，函数如下
- `int()`：将一个数值或字符串转换成整数，可以指定进制。
- `float()`：将一个字符串转换成浮点数。
- `str()`：将指定的对象转换成字符串形式，可以指定编码。
- `chr()`：将整数转换成该编码对应的字符串（一个字符）。
- `ord()`：将字符串（一个字符）转换成对应的编码（整数）。
### 注释

```python
import this
# 这是一些开发者留下的寄语

"""
这里也是注释，不过大部分人还是喜欢#
甚至使用 Command + / 添加批量的#
"""

```

## 列表

列表是一系列按照一定顺序排列的元素 和数组的不同之处是对元素的类型没有任何限制 但是正如集合论的开篇所讲 人们很少把一些毫无关系的量放在一起 所以列表的命名多采用名词复数；

事实上，字符串就是一种列表

``` python
bicycles=['trek','redline','cannondale']
# 这是列表的定义

print(bicycles)
# 这是直接打印列表本身 会有方括号和里面的逗号 以及字符串的引号

bicycles[0]
# 这是对列表元素的访问 还是从0开始 事实上我们也可以用 -1这些量表示倒着数 无论是使用还是修改都是从访问入手的

bicycles2 = bicycles[1:4]
#切片列表的一部分

bicycles3 = bicycles[:]
#切片全部就是复制
```

### 修改列表

``` python
bicycles[0]=bus
#对列表元素的修改就是通过访问实现的

bicycles.append('bus')
#方法append适用于在列表的尾端附件元素的 他是一个有参数的方法；我们可以先创建空列表 然后用append实现元素添加

bicycles.insert(0,'bus')
#这个方法是对元素的插入 0代表位置参数 其余元素按需移动

del bicycles[0]
#del语句实现对元素的删除 他不是方法 后面的元素自动向前补位

print(bicycles.pop())
#pop方法是弹出 无参数时他会把栈顶的元素作为返回量 并且移除这个元素 所以是弹出 弹出用于你还想使用一下这个量的时候

print(bicycles.pop(0))
#弹出也可以指定索引 原理一模一样

bicycles.remove('trek')
#remove是为了处理你不知索引的情况 直接对元素下手 他只会删除第一个被查找到的值
```

### 列表的组织

``` python
bicycles.sort()
#sort方法是对列表的永久性重新排序 他默认从小到大的顺序 如果有字母使用ASCII编码比较

print(sorted(bicycles))
#sorted函数是对列表的临时重新排序 很明显他有返回值
#sorted和sort虽然一个是方法一个是函数 但是均可以使用参数reverse=Ture进行翻转排序

bicycles.reverse()
#方法reverse是对列表原本顺序的永久性倒转 和排序无关

len(bicycles)
#函数len是对列表元素数量的测量
```

### 数值列表

列表非常适合存储数字 正如数组一样 但是Python为我们提供了很多更简单的工具来实现他

``` python
for value in range(1,5):
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
```

### 使用列表的一部分

``` python
players=list(range(1,11))
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
```
### 元组

元组和列表是非常相似的 实际上他和列表唯一的区别就是元组创建后不可以被修改 实际上元组就是为了避免列表被不小心修改而产生的

``` python
players=(23,434)
# 这就是创建了一个元组
#访问和遍历与列表完全一致

players=(23,434)
players=(23,23434)
#虽然我们不能修改元组里面的变量 但重新给元组赋值是合法的


person = list(players)
print(person)
# 将元组转换成列表


fruits_list = ['apple', 'banana', 'orange']
fruits_tuple = tuple(fruits_list)
print(fruits_tuple)
# 将列表转换成元组
```


## 分支

### 基础的内容
在Python中，要构造分支结构可以使用`if`、`elif`和`else`关键字。所谓**关键字**就是有特殊含义的单词，像`if`和`else`就是专门用于构造分支结构的关键字，很显然不能够使用它作为变量名

``` python
for car in cars:
    if car=='bwm':
        print(car.upper())
    else:
        print(car.title())
#Python中没有用花括号来构造代码块而是**使用了缩进的方式来表示代码的层次结构**
#一切判断的核心都是True和False == 和> < !=是最重要的判断语句

if 'Audi'=='audi'
#python默认区分大小写 如果不需要可以用前面使用的upper和lower方法处理 这几个方法不改变原有变量

if 1>2 and 3>4:
if 1>2 or 3>4:
# 针对Bool型变量我们也有运算符号

for car in cars:
for car not in cars:
#这是关于事物是否在列表里面的判断语句

if car=='bwm':
    print(car.upper())
if car=='bwm':
    print(car.upper())
else:
    print(car.title())
if car=='bwm':
    print(car.upper())
elif car=='ct':
    print(car.lower())
else:
    print(car.title())
#以上是if语句的三种结构 他的自由程度也是非常高的
#最后提到一点 这三种语句块共同之处是只执行其中的一块 如果想要执行多次判断 要用大量的if语句 而非elif结构
```

### 列表和PEP 8

```python
if cars:
    *********
else:
    **********
#这个if语句会检验列表是不是空的 如果确实是空的 执行else语句内容
if age < 4:
#这是PEP8向我们建议的书写方式，四个空间作为区分层级的符号，对于判断符和变量之间使用空格间隔一下
```

## 集合 字典

字典类似C语言的结构 他旨在把各种不同类型但是相关的数组组成一个便于访问的单元 而非列表一般存储平级数据；集合则和数学上的集合是一致的。

### 集合
Python中的集合跟数学上的集合是一致的，不允许有重复元素，而且可以进行交集、并集、差集等运算。
我们在研究一些比较单一的统计项 set是一个很好的选择
```python
# 创建集合的字面量语法
set1 = {1, 2, 3, 3, 3, 2}
print(set1)
print('Length =', len(set1))
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
print(set1 & set2)
# print(set1.intersection(set2))
print(set1 | set2)
# print(set1.union(set2))
print(set1 - set2)
# print(set1.difference(set2))
print(set1 ^ set2)
# print(set1.symmetric_difference(set2))

# 判断子集和超集
print(set2 <= set1)
# print(set2.issubset(set1))
print(set3 <= set1)
# print(set3.issubset(set1))
print(set1 >= set2)
# print(set1.issuperset(set2))
print(set1 >= set3)
# print(set1.issuperset(set3))
```
### 使用字典

```python
alien_0={'color':'green','points':5}
print(alien0['color'])
print(alien0['points'])
#这就是最基本的字典定义的方式和使用的方式 借助原本学习结构的思路我们能很快的理解 字典这个编程概念的重要性
#Python的字典是键值对的结构 键与值相关联 值的数据类型不进行任何限制 任何对象都可以

# 创建字典的构造器语法
items1 = dict(one=1, two=2, three=3, four=4)

print(alien0)
#直接对字典的打印会显示他的信息快照 和数组接近

# 通过键可以获取字典中对应的值，无论构造的时候键名，只要是字符串就需要引号
print(alien0['x_position'])
print(alien0['x_position'])

alien0['x_position']=0
alien0['y_position']=25
#这就是新增键值对的过程 我们新增了两个和位置相关的键并给了值

alien0={}
#这定义了空字典 方便我们后面添加键值对

alien0['x_position']=alien0['x_position']+5
#对值的修改是直接访问进行的 这很自然

del alien0['points']
#这会直接删除键和他对应的值 有时候留着这个键没用

alien0.clear()
# 清空字典
```

``` python
favorite_languages = {
    'jen':'python',
    'sarah':'c',
    'edward':'ruby',
    'phil':'python',
}
#这个由于字典太长才进行的分行定义 这是我们的一个习惯 别忘了逗号 不能省略的

print(alien0.get('height','No height value assigned'))
#get也是一个用于访问字典的方法 他会用来处理想要访问的键不存在的问题 正常情况下 此时程序会崩溃 但是用get就可以避免崩溃
```

### 遍历字典

很明显对字典进行遍历是不能忽视的一个问题 但是只有键值对的字典该怎么遍历呢

```python
for key,value in favorite_languages.items():
    print(f"\n Key:{key}")
    print(f"\n Value:{value}")
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
    print(f'{key}: {scores[key]}')
```

### 嵌套

很明显的 字典和列表使我们目前为止学到的非常有用的数据类型 甚至可以说是一种好用的数据结构 他们之间的嵌套是自然的 也是复杂的

```python
alien_0={'color':'green','points':5}
alien_2={'color':'black','points':4}
alien_1={'color':'green','points':2}
aliens=[alien_0,alien_1,alien_2]
# 这是一个字典列表 列表的所有元素都是字典
pizza = {
    'crust':'thick',
    'topping':['muanroom','cheese'],
}
#很明显的 字典里面的某个元素是列表 这也是按需设置的 访问的时候按照程序设计的逻辑就可以
#字典当然可以和字典嵌套 好好设计结构就可以 千万别忘了考虑访问
#记住 列表一般存储同类型信息 字典用来存储不同类型的信息 按照需求进行嵌套
```

## 用户输入与循环
### 遍历列表

``` python
bicycles=['trek','redline','cannondale']
for bicycle in bicycles:
    print(bicycle)
#这就是for in循环的基本样式 ，针对列表实现for in 循环很自然
```

冒号和缩进是Python循环的特征 缩进部分结束循环自然结束 也就是说Python的缩进更加接近自然语言 for in循环基于列表实现


```python
# 如果应该缩进的地方完全没有缩进 Python会提供报错提示
# 如果只是漏掉了一部分需要的缩进 这属于逻辑错误 编辑器无法排查
# 不需要缩进的地方在Python严禁缩进 因为缩进是Python识别语言的依据之一
# 不要忘记冒号 这是重要的标识符
```
### input函数

```python
message = input("Tell me some thing : ")
#input函数只接受一个参数 也就是提示prompt 他会让用户键入数据 并且返回给函数返回值

prompt = "****************"
prompt+="\n****"
name=input(prompt)
#这告诉我们一定要给与用户清晰的prompt 上面的字符串加法是一种连接手段 分行的提示让提示更加清楚 冒号的存在让键入更加舒适

age = int(input())
#input函数是默认用字符串来处理内容的 int函数是为了把它变成数 数字是方便各种使用的
```

### while循环

for in 循环在 Python 中基于列表运行 当然他是有用的 但是有时候仅仅 for 循环并不够用 如果要构造不知道具体循环次数的循环结构，我们推荐使用`while`循环。`while`循环通过一个能够产生或转换出`bool`值的表达式来控制循环，表达式的值为`True`则继续循环；表达式的值为`False`则结束循环。

``` python
currrent = 1
while current <= 5:
    print(current)
    current += 1
#这就是while循环的核心 设定停止条件 原本C中的for循环也是使用停止条件的 但是Python对这方面进行了改动，改为了一个针对列表的操作

message = ""
while message != 'quit':
    message = input(输入'quit'退出循环)
    if message != 'quit':
        print(message)
#这是一个非常常用的退出循环 死去的记忆死灰复燃

while flag != Ture:
    ***********
#使用flag标志用于多个因素均会导致这个循环出现变化的情况 flag的运用是非常重要的

while flag:
#这是简化的形式flag形式，flag==Ture的时候自动脱离

while Ture:
    message = input()
    if message == 'quit':
        break
    else:
        print(message)
#使用break命令可以随时跳出循环 开头的while ture实际上是无限循环的意思 这种代码一定要搭配跳出手段 无限循环可不是什么好事情

currrent = 1
while current <= 10:
    current += 1
    if current % 2 == 0:
        continue
    print(current)
#这是对continue的使用 注意 break 和 continue 会选择离他最近的循环 并且只选择这一个循环
```

### while循环与字典和列表

我们为什么会引出这一个分标题 for 循环对于列表的兼容性非常好 为什么要舍近求远

for 循环当然非常有效 但是我们不建议在for 循环的过程中修改元素 他应该只用于遍历 如果要在遍历的同时修改列表 字典 请使用while循环结构 便于Python追踪

```python
while list:
    ***********
#如上所示的while循环将会一直运行 直到list变成一个空列表 也就是我们在后面的代码块中要记得不断减少列表中的元素 pop方法 del都可以

while 'cat' in pets:
    pets.remove('cats')
#这也是一个重要的判断 他其实起到了remove移除多个元素的作用 不要忘记in这个在Python中重要的变量

#键值对的while循环也是容易的 事实上我们大多会引入flag来帮助循环 灵活运用前面提到的方法就可以
```

## 函数
### 定义函数及其参数

函数是提升编写效率的重要存在 我们前面使用的 input sorted 都是函数 现在我们来讲讲编写属于自己的函数

``` python
def greet_user():
    print("hello!")
#这个非常简单的代买解释了Python函数的最简单定义方式 也就是def语句 后面的greet_uesr是我们的函数名 缩进部分是函数体 括号里面是可有可无的参数 调用函数一定要带括号 实际上我们使用的基本函数都是别人帮你定义好的

def greet_user(name):
    print(f"hello! {name.title()}")
#添加参数也非常简单 此时调用函数的时候也要记得加函数需求的变量 否则肯定会报错
#多个参数的顺序方面 我们更应该灵活的选取调用方法 位置调用更加的简单迅速  关键字在参数数量增加的时候才会更加的使用

def greet_user(first_name,last_name='Wang'):
#这其实给了参数默认值 此时哪怕你在使用的时候传入的参数不够 Python也不会报错而是使用默认参数 当然你传入新的参数的时候会按照传入的执行
#位置调用其实引入了另一个问题 导致有时候使用参数会用如 greet(,Jue) 这就是设定了默认参数并且选择了位置调用 不过熟能生巧 出错了积极修改就好 一般会把最容易不选的参数放在最后 来给调用者省事
```

### 有返回值和参数的函数

很明显的我们见过有返回值的函数 但是前面编写函数的时候没有说怎么又返回值 so now

``` python
def greet_user():
    print("hello!")
    return 0
#这就是最简单的有返回值的函数 我们不对返回值的类型进行限制（列表 字典也可以） 编写的时候会发现 一旦return 编辑器的换行辅助就会帮助我们重启一行 意思是return是一个函数的结尾 之后别再编写了 （使用函数的返回值是很重要的）

def build_person(first_name,last_name,age=None):
#None 是一个特殊的占位符 他的意思是什么都没有 条件测试的时候None是False

def greet_users(names):
    for name in names:
        print(f"Hello! {name}")
#函数对于直接传入或者传出列表没有任何特殊要求 但是我们要知道 传入列表的函数没有形式参数的概念 被改变的就是原始列表

function_name(names[:])
#Python程序员总是要求许多 使用以上的调用方式就可以避免原始列表的修改 此时只会影响副本 注意 大型列表的拷贝是一件非常耗费资源的事情 不要随便这样拷贝 除非一定需要


```
### 任意数量参数
  ```python
def make_pizza(*toppings):
    print(toppings)
#这个函数的定义方法允许我们在函数内使用任意数量的实参 并且把他们存储在一个元组里面 这是非常有用的

def make_pizza(size,*toppings):
#混合使用当然是允许的 在位置实参中 我们肯定是先满足前面的 最后把所有多的放到后面的整个toppings里面形成元组 当然使用关键词也是一个不错的主意

def build_profile(first,last,**info):、
build_profile(Wang,Jue,location='LUOYANG',filed='MATH')
#容易看到我们又多了一个星号 此时info会接受任意数量的键值对并生成字典
  ```
### 用模块管理函数
关于模块化：由于Python没有函数重载的概念，那么后面的定义会覆盖之前的定义，也就意味着两个函数同名函数实际上只有一个是存在的；

那么怎么解决这种命名冲突呢？答案其实很简单，Python中每个文件就代表了一个模块（module），我们在不同的模块中可以有同名的函数，在使用函数的时候我们通过`import`关键字导入指定的模块就可以区分到底要使用的是哪个模块中的函数

```python
import pizza
pizza.make_pizza()
#import命令能够让我们引入一个新的模块（后缀为.py的文件） 使用import后 模块里面函数之类的就可以间接调用(要前缀) 这是跨文件编程的第一步

from pizza import make_pizza
#导入单个函数的语法 这是后调用函数直接用make_pizza就可以 不需要pizza.

from pizza import make_pizza as mp
import pizza as p
#这是对函数和模块起了一个其他的名字 后面直接使用就可以

from pizza import *
#这是复制所有函数到这个文件里面 可以直接用名字调用 不需要pizza. 不过请尽量节制的使用这个导入方法
```

对于模块化问题，我们需要引入一个重要的判断
```python
if __name__ == '__main__':
#用来判定是否是直接执行的模块

#在模块化中 这是重要的判定, 如果模块包含可执行代码 那么在导入模块的时候他们会被运行, 我们需要避免导入某个模块时错误的执行这些代码
```



### 关于变量作用域
没有定义在任何一个函数中 这是一个全局变量（global variable）
定义在某个函数中的变量 是一个局部作用域的变量
但是如果函数中嵌套了函数 一个局部变量被另一个作用域访问了 此时称为嵌套作用域
函数会根据局部作用域 嵌套作用域 全局作用域 内置作用域（语言保留的标识符）的顺序对变量进行寻找 以最优先找到的那个为准 现在应该能理解我们对变量作用域做的那些不精确描述和理解了

```python
global a
nonlocal a
#他们的意思分别是访问全局作用域变量a 和嵌套作用域变量a
#如果没有这样的变量 就创建一个
#降低对全局变量的依赖是降低程序耦合程度的关键

def main():
    # Todo: Add your code here
    pass


if __name__ == '__main__':
    main()
#这样的一份主程序 就符合一个专业的开发者的习惯
```

## 正则表达式
在编写处理字符串的程序或网页时，经常会有查找符合某些复杂规则的字符串的需要，正则表达式就是用于描述这些规则的工具，换句话说正则表达式是一种工具，它定义了字符串的匹配模式（如何检查一个字符串是否有跟某种模式匹配的部分或者从一个字符串中将与模式匹配的部分提取出来或者替换掉）。

正则表达式很好理解，就是一套用于匹配字符串的语言，Python对正则表达式也提供了支持。

### 正则表示式
| 符号                 | 解释                  | 示例                  | 说明                                              |
| ------------------ | ------------------- | ------------------- | ----------------------------------------------- |
| .                  | 匹配任意字符              | b.t                 | 可以匹配bat / but / b#t / b1t等                      |
| \\w                | 匹配字母/数字/下划线         | b\\wt               | 可以匹配bat / b1t / b_t等<br>但不能匹配b#t                |
| \\s                | 匹配空白字符（包括\r、\n、\t等） | love\\syou          | 可以匹配love you                                    |
| \\d                | 匹配数字                | \\d\\d              | 可以匹配01 / 23 / 99等                               |
| \\b                | 匹配单词的边界             | \\bThe\\b           |                                                 |
| ^                  | 匹配字符串的开始            | ^The                | 可以匹配The开头的字符串                                   |
| $                  | 匹配字符串的结束            | .exe$               | 可以匹配.exe结尾的字符串                                  |
| \\W                | 匹配非字母/数字/下划线        | b\\Wt               | 可以匹配b#t / b@t等<br>但不能匹配but / b1t / b_t等         |
| \\S                | 匹配非空白字符             | love\\Syou          | 可以匹配love#you等<br>但不能匹配love you                  |
| \\D                | 匹配非数字               | \\d\\D              | 可以匹配9a / 3# / 0F等                               |
| \\B                | 匹配非单词边界             | \\Bio\\B            |                                                 |
| []                 | 匹配来自字符集的任意单一字符      | [aeiou]             | 可以匹配任一元音字母字符                                    |
| [^]                | 匹配不在字符集中的任意单一字符     | [^aeiou]            | 可以匹配任一非元音字母字符                                   |
| *                  | 匹配0次或多次             | \\w*                |                                                 |
| +                  | 匹配1次或多次             | \\w+                |                                                 |
| ?                  | 匹配0次或1次             | \\w?                |                                                 |
| {N}                | 匹配N次                | \\w{3}              |                                                 |
| {M,}               | 匹配至少M次              | \\w{3,}             |                                                 |
| {M,N}              | 匹配至少M次至多N次          | \\w{3,6}            |                                                 |
| \|                 | 分支                  | foo\|bar            | 可以匹配foo或者bar                                    |
| (?#)               | 注释                  |                     |                                                 |
| (exp)              | 匹配exp并捕获到自动命名的组中    |                     |                                                 |
| (?&lt;name&gt;exp) | 匹配exp并捕获到名为name的组中  |                     |                                                 |
| (?:exp)            | 匹配exp但是不捕获匹配的文本     |                     |                                                 |
| (?=exp)            | 匹配exp前面的位置          | \\b\\w+(?=ing)      | 可以匹配I'm dancing中的danc                           |
| (?<=exp)           | 匹配exp后面的位置          | (?<=\\bdanc)\\w+\\b | 可以匹配I love dancing and reading中的第一个ing          |
| (?!exp)            | 匹配后面不是exp的位置        |                     |                                                 |
| (?<!exp)           | 匹配前面不是exp的位置        |                     |                                                 |
| *?                 | 重复任意次，但尽可能少重复       | a.\*b<br>a.\*?b     | 将正则表达式应用于aabab，前者会匹配整个字符串aabab，后者会匹配aab和ab两个字符串 |
| +?                 | 重复1次或多次，但尽可能少重复     |                     |                                                 |
| ??                 | 重复0次或1次，但尽可能少重复     |                     |                                                 |
| {M,N}?             | 重复M到N次，但尽可能少重复      |                     |                                                 |
| {M,}?              | 重复M次以上，但尽可能少重复      |                     |                                                 |

> **说明：** 如果需要匹配的字符是正则表达式中的特殊字符，那么可以使用\\进行转义处理，例如想匹配小数点可以写成\\.就可以了，因为直接写.会匹配任意字符；同理，想匹配圆括号必须写成\\(和\\)，否则圆括号被视为正则表达式中的分组。

### Python中的正则表达式
Python提供了re模块来支持正则表达式相关操作，下面是re模块中的核心函数。

| 函数                                         | 说明                                                         |
| -------------------------------------------- | ------------------------------------------------------------ |
| compile(pattern, flags=0)                    | 编译正则表达式返回正则表达式对象                             |
| match(pattern, string, flags=0)              | 用正则表达式匹配字符串 成功返回匹配对象 否则返回None         |
| search(pattern, string, flags=0)             | 搜索字符串中第一次出现正则表达式的模式 成功返回匹配对象 否则返回None |
| split(pattern, string, maxsplit=0, flags=0)  | 用正则表达式指定的模式分隔符拆分字符串 返回列表              |
| sub(pattern, repl, string, count=0, flags=0) | 用指定的字符串替换原字符串中与正则表达式匹配的模式 可以用count指定替换的次数 |
| fullmatch(pattern, string, flags=0)          | match函数的完全匹配（从字符串开头到结尾）版本                |
| findall(pattern, string, flags=0)            | 查找字符串所有与正则表达式匹配的模式 返回字符串的列表        |
| finditer(pattern, string, flags=0)           | 查找字符串所有与正则表达式匹配的模式 返回一个迭代器          |
| purge()                                      | 清除隐式编译的正则表达式的缓存                               |
| re.I / re.IGNORECASE                         | 忽略大小写匹配标记                                           |
| re.M / re.MULTILINE                          | 多行匹配标记                                                 |

> **说明：** 上面提到的re模块中的这些函数，实际开发中也可以用正则表达式对象的方法替代对这些函数的使用，如果一个正则表达式需要重复的使用，那么先通过compile函数编译正则表达式并创建出正则表达式对象无疑是更为明智的选择。

### 例子
```python
"""
验证输入用户名和QQ号是否有效并给出对应的提示信息

要求：用户名必须由字母、数字或下划线构成且长度在6~20个字符之间，QQ号是5~12的数字且首位不能为0
"""
import re


def main():
    username = input('请输入用户名: ')
    qq = input('请输入QQ号: ')
    # match函数的第一个参数是正则表达式字符串或正则表达式对象
    # 第二个参数是要跟正则表达式做匹配的字符串对象
    m1 = re.match(r'^[0-9a-zA-Z_]{6,20}$', username)
    if not m1:
        print('请输入有效的用户名.')
    m2 = re.match(r'^[1-9]\d{4,11}$', qq)
    if not m2:
        print('请输入有效的QQ号.')
    if m1 and m2:
        print('你输入的信息是有效的!')


if __name__ == '__main__':
    main()
```

上面在书写正则表达式时使用了“原始字符串”的写法（在字符串前面加上了r），所谓“原始字符串”就是字符串中的每个字符都是它原始的意义，说得更直接一点就是字符串中没有所谓的转义字符

如果要从事爬虫类应用的开发，那么正则表达式一定是一个非常好的助手，因为它可以帮助我们迅速的从网页代码中发现某种我们指定的模式并提取出我们需要的信息，当然对于初学者，要编写一个正确的适当的正则表达式可能并不是一件容易的事情（当然有些常用的正则表达式可以直接在网上找找）
## 文件与异常

这一章和下一章都不会引入新的程序设计知识 他们只是为了提高程序的适用性 可用性 稳定性

### 从文件中读取数据

``` python
#Python默认从当前运行的程序所在的位置寻找要打开的文件
with open('file_name.txt',encoding='utf-8') as file_object:
    contents = file_object.read()
print(contents.rstrip())
#一切文件操作的核心都是打开文件 open寻找并打开了参数内的文件 并且把这个对象返回给了file_object 关键词with会在不需要访问文件后将其关闭 此时我们就不需要使用close() 避免了一些问题 Python会帮我们处理关闭文件的问题
#拥有文件对象以后（面向对象的程序设计）我们对他使用了方法read 他会读取这个文件的全部内容 并且用字符串返回他 请注意read函数会在读取到的所有内容后面添加一个空字符串 所以添加了rstrip的方法进行处理
#encoding部分是对文件读取编码的要求 如果默认编码和文件使用的编码不一样 直接读取会出现乱码
with open('text_files/filename.txt') as file_object:
#这是相对文件路径寻找 是从目前程序所在的文件夹的子文件夹寻找 我们使用/ 而不是\这一标准的文件路径符号是因为代码的特殊规定
with open('/home/ehmatthes/others/text_files/filename.txt') as file_object
#这是绝对文件路径寻找 要从盘符开始
with open('file_name.txt') as file_object:
    for line in file_object:
        print(line.rstrip())
#这是逐行读取文件的内容 rstrip的存在是因为 逐行读取是包括结尾的换行符的 print又会增加换行符号 导致换行符号过多
with open('file_name.txt') as file_object:
    lines = file_object.readlines()
print(lines)
#read函数打开的文件在函数结束后自动关闭 前面我们是吧整个文件当成一个巨大的字符串存储 实际上readlines方法能够把它转换成一个按行为单位的列表 当然别忘了rstrip
#有了上面研究的这几个方法 使用txt文件应该不是一个困难的事情了 注意 Python 把txt内的所有文本都解读为字符串 要进行别的数字运算的时候要格式转换
```

### 写入文件

``` python
with open('file_name.txt','w') as file_object:
    file_object.write('I love programming.\n')
#对于写入文件 我们要给Python的open额外的参数 也就是这个w 它意味着我们打开后可能对文件进行写入
#实际上 'r'读取模式 'w'写入模式 'a'附加模式 'r+'读写模式 不写参数 默认'r'
#写入模式w要专门注意 如果要写入的文件不存在 Python会创建他 如果已经存在 Python会先对他进行格式化
#write方法就是写入字符串使用的 Python只能写入字符串 他不会默认加上换行符 所以我们一般会人为加上
with open('file_name.txt','a') as file_object:
    file_object.write('I love programming.\n')
#附加模式不会对原始文件格式化 他会在原本的文件后面添加东西
#对文件的读取和写入本质上与对终端的读取和写入没有差别
```

### 异常

代码不出现错误是不可能的 大多时候 代码遇到错误的时候会终止运行 但是用户用着用着突然崩溃了并不是什么好体验 所以Python提供了异常这一特殊的对象用来处理这些错误 当他遇到不知所措的错误时 会创建一个异常对象 如果我们编写了处理异常对象的代码 程序会继续运行 而不是反馈trackback

异常使用try-except代码块来处理 记得告诉用户我们出了什么错误 而不是一个正常人完全看不懂的trackback

``` python
try:
    print(5/0)
except ZeroDivisionError:
    print("You can't divide by zero")
#这就是最基础的模块 如果try部分的代码运行时出现了错误 python会选择对应的except代码块并运行 这样的话我们就避免了trackback被显示出来 并且程序可以继续向后面运行 事实上 trackback的显示无论是对于使用者还是专业程序员看到都不好 使用者会感到疑惑 专业的程序员会从你的trackback中看到很多程序相关的信息
try:
    answer = a/b
except ZeroDivisionError:
    print("You can't divide by zero")
else:
    print(answer)
#else 代码块是在try运行中没有发现问题才继续执行的代码 他依赖于try的正确执行 这是他和直接写在后面的代码的区别
except ZeroDivisionError:
    pass
#这适用于你不想给使用者看到任何信息的情况 pass的意思就是什么都不执行 这往往称为静默失败 是否选择静默失败是程序设计者应该仔细思考的 显示一些没用的信息有时候会减小程序的易用性
#研究异常不是说让自己编写的代码出错 而是为了避免一些外部因素影响程序的可用性 比如用户输出错误 文件被以外删除 网络连接异常等等 程序的设计者往往凭借自己的经验判断程序什么时候可能出现异常 并且加以处理
```

### 存储数据

任何的程序被关闭的时候总要存储一些数据到文件中 模块json是一个很好的模块 他原本是java开发的 但是很快变成了一种常见的格式 被很多开发者使用

``` python
import json
numbers = [1,2,3,42,1]
with open ('filename.json','w') as f:
    json.dump(numbers,f)
#实际上json是一个很实用的数据格式 我们把一个列表借助dump函数写到了一个json文件里面 dump写入函数支持两个参数 要写入的内容和文件名
with open ('filename.json') as f:
    numbers = json.load(f)
#load函数接受一个参数 文件名
#事实上使用这个json格式能够存储很多txt不能存储的数据类型 他对成程序设计者非常有用
#熟练地使用前面所学习的内容 我们就可以开始设计一些比较复杂的程序了 要记住我们的核心编程思想 一旦主程序变得臃肿 就要重构代码 建立函数进行封装 通过完善的注释来让程序有着优良的可读性
```

## 测试

测试代码的目的非常简单 你希望你提交的代码里面尽量不含有任何bug 或者说哪怕出现问题 你的代码也能向预期一样工作 这里我们只会介绍简单的测试方法

``` python
#首先我们需要被测试的代码
def get_formatted_name(first,last):
    full_name = f"{first} {last}"
    return full_name.title()
#如何测试这个函数能不能像我们设计的一样工作 当然我们可以选择手动输入 但是这太麻烦了 Python为我们提供了一些自动测试的有效方法
```

Python标准库提供了模块unittest作为代码测试工具  **单元测试**衡量函数的某个方面没问题 **测试用例**是一组单元测试 **全覆盖**的测试是一整套测试 一般我们只会在项目广泛使用的时候进行全覆盖 正常只需要对程序的重要行为编写测试

``` python
import unittest
from fuction import get_formatted_name
#引入测试用模块和待测试函数
class NamesTestCase(unittest.TestCase):
    def test_first_last_name(self):
        formatted_name = get_formatted_name('janis','joplin')
        self.assertEqual(formatted_name,'Janis Joplin')

#创建一个类（测试用例）用于储存测试单元 命名一定要能看出来他是在测试什么 注意 这个类继承了一个unittest模块中的类
#对于这个测试才被使用的类 我们只为他创建里除继承以外的一个方法 运行这个方法的时候 我们会使用一次待测试函数 然后借助unittest本来就存在的断言方法核实我们期望得到的结果和实际运行生成的结果 运行这段代码 我们会显示一个明显的测试提示无论通过与否 告诉你测试的结果如何 现在我们就编写了一个测试单元 随时可以使用这个测试单元对函数进行检验
#很明显的 仅仅一个测试单元肯定不能满足我们的要求 实际上测试工程师的任务就是编写测试单元 整合各种测试用例让架构师使用
	def test_first_last_middle_name(self):
        formatted_name = get_formatted_name('wolfgang','mozart','amadeus')
        self.assertEqual(formatted_name,'Wolfgang Mozart Amadeus')
#我们又编写了一个测试单元 实际上测试模块的规则非常的特殊 一切test_开头的方法都会被自动调用 当这个模块被使用时 这就是为了让架构师能够轻松地调用 一切测试单元的方法名都要是描述性的 无论长度 你要让架构师能快速的定位是哪里的程序出了问题
```

前面是对函数的测试 现在来考虑测试类能否顺利的工作（测试一般是因为你修改了原始代码 你需要保证你的修改不影响他应该有的功能）

```python
#下面是一些比较常用的unittest模块断言方法
assertEqual(a,b)
assertNotEqual(a,b)
assertTure(a)
assertFalse(a)
assertIn(item,list)
assertNotIn(item,list)
#他们分别合适相等与否 真假与否 所有与否 实际上就是一些布尔运算 只是在设计测试的时候比较重要 断言是测试结果是否通过的判别
```

测试类往往是在测试类中的方法 所以和测试函数有很多相近的地方 他们使用完全相同的开头 完全一样的定义测试类（实例）的方式 也是选择在方法中使用这个类或者函数的一部分功能 实际上测试用例里面就是对原始函数或者类的完全使用 最后使用断言来判断 实际上是完全一样的 代码如下

``` python
class AnonymousSurvey():
    """收集匿名调查问卷的答案"""

    def __init__(self, question):
        """存储一个问题，并为存储答案做准备"""
        self.question = question
        self.responses = []

    def show_question(self):
        """显示调查问卷"""
        print(self.question)

    def store_respond(self, new_response):
        """存储单份调查答卷"""
        self.response.append(nes_response)

    def show_result(self):
        """显示收集到的所有答卷"""
        print("Survey results:")
        for response in self.responses:
            print('- ' + response)
# A new file
import unittest
from survey import AnonymousSurvey

class TestAnonymousSurvey(unittest.TestCase):
     """针对AnonymousSurvey类的测试"""

     def test_single_response(self):
         """测试单个答案会被妥善的存储"""
         question = "What language did you first learn to speank?"
         my_survey = AnonymousSurvey(question)
         my_survey.store_response('English')

         self.assertIn('English', my_survey.response)
     def test_store_three_response(self):
        """测试三个答案会被被妥善地存储"""
        question = "What language did you first learn to speak?"
        my_survey = AnonymousSurvey(question)
        response = ['English', 'Spanish', 'Mandarin']
        for response in responses:
        	my_survey.store_response(response)

    	for response in responses:
        	self.assertIn(response, my_survey.response)
#容易发现测试类总是需要建立好多次实例  一直CV确实无聊 所有有个setup
        def setUp(self):
            question = "What language did you first learn to speak?"
            self.my_survey = AnonymousSurvey(question)
            self.responses = ['English', 'Spanish', 'Mandarin']
#创建一个调查对象和一组答案，共使用的测试方法使用 按照管理建立Setup要在所有测试用例之前
```

Python的测试会在运行时给与一些反馈

每完成一个单元测试都会打印一些字符 通过时打印句点 测试引发错误打印E 断言失败时打印F

**Python的基础内容到这里就结束了 我们需要完成一些项目来作为实践并且复习学到的知识 不过那就不必在这个文件里面继续进行了  编辑器才是真正应该待的地方**
