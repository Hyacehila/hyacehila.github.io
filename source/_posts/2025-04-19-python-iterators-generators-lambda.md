---
title: "Python 迭代器、生成器与 Lambda：惰性计算和函数式工具"
title_en: "Python Iterators, Generators, and Lambda: Lazy Evaluation and Functional Tools"
date: 2025-04-19 19:53:29 +0800
categories: ["Programming", "Python"]
tags: ["Learning Notes", "Python", "Iterators"]
author: Hyacehila
excerpt: "一篇 Python 进阶基础学习笔记，整理可迭代对象、迭代器、map、filter、生成器、排序 key 参数和 lambda 函数。"
excerpt_en: "A study note on intermediate Python basics, covering iterables, iterators, map, filter, generators, sort key functions, and lambda functions."
mathjax: false
hidden: true
permalink: '/blog/2025/04/19/python-iterators-generators-lambda-learning-notes/'
---

## 关于迭代

### 可迭代对象与迭代器

**可迭代对象**是指那些实现了 `__iter__()` 方法的对象。简单来说，可迭代对象就是可以被迭代的对象，也就是可以使用 `for` 循环遍历的对象。常见的可迭代对象包括列表、元组、字符串、字典、集合等。

可迭代对象必须实现 `__iter__()` 方法，该方法返回一个**迭代器对象**。当使用 `for` 循环遍历可迭代对象时，实际上是先调用该对象的 `__iter__()` 方法获取迭代器，然后通过迭代器来逐个访问元素。

迭代器是一种特殊的对象，它实现了 `__iter__()` 和 `__next__()` 方法（因此也可以看成一个可迭代对象）。`__iter__()` 方法返回迭代器本身，而 `__next__()` 方法用于返回迭代器的下一个元素。当没有更多元素时，`__next__()` 方法会抛出 `StopIteration` 异常。

比如如下代码的效果

```python
my_list = [1, 2, 3]
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
    print("已经没有更多元素了")
```

对于涉及 `for` 循环遍历的函数，迭代器与可迭代对象还是很重要的，还有很多函数基于迭代对象开发，并且在实现上很重要。

### 迭代相关函数

#### `map` 函数

`map` 函数的作用是将一个函数应用到可迭代对象的每个元素上，并返回一个新的迭代器，该迭代器中的元素是原可迭代对象中每个元素经过指定函数处理后的结果。其基本语法如下：

```python
map(function, iterable, ...)
```

- `function`：这是要应用的函数，它会被应用到 `iterable` 中的每个元素上。
- `iterable`：这是一个或多个可迭代对象，如列表、元组、集合等。如果传入多个可迭代对象，`function` 必须能接受和可迭代对象数量相同的参数。否则将会出错。

一个简单的示例

```python
# 定义一个将元素平方的函数
def square(x):
    return x ** 2

numbers = [1, 2, 3, 4, 5]
# 使用 map 函数将 square 函数应用到 numbers 列表的每个元素上
squared_numbers = map(square, numbers)
# 此时返回 map 对象，无法直接打印，需要转换回我们熟悉的数据结构上
# 将 map 对象转换为列表
result = list(squared_numbers)
print(result)  # 输出: [1, 4, 9, 16, 25]
```

关于 `map` 函数 我们经常使用 `lambda` 函数来简化相关问题，如下

```python
numbers = [1, 2, 3, 4, 5]
# map 函数自身会帮助我们把各个元素传入给函数的，lambda仅仅用于定义函数
squared_numbers = map(lambda x: x ** 2, numbers)
result = list(squared_numbers)
print(result)  # 输出: [1, 4, 9, 16, 25]
```

同时处理多个可迭代对象的 `map` 也非常自然

```python
numbers1 = [1, 2, 3]
numbers2 = [4, 5, 6]
# 定义一个将两个元素相加的函数
def add(x, y):
    return x + y

result = map(add, numbers1, numbers2)
print(list(result))  # 输出: [5, 7, 9]
```

#### `filter` 函数

`filter` 函数的作用是过滤可迭代对象中的元素，只保留那些使指定函数返回 `True` 的元素，并返回一个新的迭代器。其基本语法如下：

```python
filter(function, iterable)
```

- `function`：这是一个用于过滤的函数，它接受一个参数，并返回一个布尔值。如果返回 `True`，则该元素会被保留；如果返回 `False`，则该元素会被过滤掉。
- `iterable`：这是要过滤的可迭代对象，如列表、元组、集合等。

整体的语法规则与 `map` 函数非常类似，我们只需要再单独介绍一个例子。

```python
# 定义一个判断元素是否为偶数的函数
def is_even(x):
    return x % 2 == 0

numbers = [1, 2, 3, 4, 5, 6]
# 使用 filter 函数过滤出偶数
even_numbers = filter(is_even, numbers)
# 将 filter 对象转换为列表
result = list(even_numbers)
print(result)  # 输出: [2, 4, 6]
```

### 生成器 generator

`yield` 关键字用来标注一个`generator` 当一个函数中包含 `yield` 那么他自动变为一个生成器函数.并且生命周期以及特性就都发生了变换.

`yield` 关键字的核心—— “暂停并交出”

- **普通函数 (`return`)**：就像你一口气看完一整季20集，然后告诉朋友“我看完了”。你只有一次性讲完所有剧情这一个选项。
- **生成器函数 (`yield`)**：就像你看一集，然后按下“暂停并交出”按钮。你把遥控器（控制权）交给朋友，让他去看别的。等他回来想继续看时，你再从刚才暂停的地方，播放下一集给他看。

`yield` 就是那个“**暂停并交出**”按钮。它做了两件至关重要的事：

1. **交出值**：将 `yield` 后面的表达式结果，作为本次迭代的返回值，交出去。
2. **暂停函数**：函数的执行状态（包括所有局部变量的值）被**冻结**，函数“睡去”，等待下一次被唤醒。

当一个函数包含 `yield` 时，它的生命周期就完全改变了：

1. **调用时**：`my_generator = my_func()` 不会执行函数体。它只是创建并返回一个**生成器对象**。这个对象就像一个“待办事项清单”，记录了函数的代码和当前状态。
2. **首次迭代时**：当你第一次遍历这个生成器（例如用 `for` 循环），函数开始执行，直到遇到第一个 `yield`。
3. **遇到 `yield` 时**：函数交出 `yield` 后面的值，然后**立即暂停**，所有内部状态都被保存。
4. **下次迭代时**：函数从上次暂停的地方**苏醒**，继续执行，直到遇到下一个 `yield`。
5. **循环往复**：这个过程持续进行，直到函数执行完毕（没有更多代码了）或者遇到一个 `return` 语句。

既然 `yield` 是主角，那 `return` 还有用吗？

有的，但它的作用变了。在生成器函数中，`return` 的作用是**提前终止生成器**。

当生成器函数执行到一个 `return` 语句时，它会直接停止，并引发一个 `StopIteration` 异常。这个异常可以被捕获，`return` 后面的值会成为这个异常的 `value` 属性。

这是一种比较高级的用法，通常用于向生成器的使用者传递一个额外的“最终状态”或“错误信息”。

也就是使用下面的结构来进行异常的捕捉.

```python
def generator_with_return(n):
    i = 0
    while i < n:
        yield i
        i += 1
    return "我处理完了所有数字！" # <-- return 在这里

# 创建生成器
gen = generator_with_return(3)

# 手动迭代，以便捕获 StopIteration
while True:
    try:
        value = next(gen) # next() 函数获取下一个值
        print(f"从生成器拿到: {value}")
    except StopIteration as e:
        print(f"生成器结束了！")
        print(f"它 return 的值是: {e.value}") # <-- 在这里获取 return 的值
        break

```

在使用`generator`的时候一般选择使用 `for` 循环来读取,他会自动的处理`StopIteration` 的问题,只会遍历全部的元素就自然结束.

**生成器只可以被迭代一次**



## 其他

### 排序函数的`key`参数

在最常用的 `sorted()` 函数中 参数 `key` 决定了我们使用什么东西作为排序的依据，在很多情况下，传入 `sorted()` 函数的内容往往并不是一个列表而是一个复杂的字典，因此我们需要指定使用字典的什么键来排序。

```python
students = [
    {'name': 'Alice', 'age': 20},
    {'name': 'Bob', 'age': 18},
    {'name': 'Charlie', 'age': 22}
]
# 按照年龄从小到大排序
sorted_students = sorted(students, key=lambda student: student['age'])
print(sorted_students)
```

`key` 参数的实际工作原理并没有那么简单，他是一个函数。`sorted` 函数会对 `iterable` 中的每个元素调用这个 `key` 函数，然后依据 `key` 函数的返回值来比较元素的大小，而不是直接比较元素本身。

在这一段代码考虑的问题中，`key` 函数符合惯例的定义了一个`lambda`函数，我们对`iterable` 中的每个元素调用这个`lambda`函数，得到了一个值用于排序，这里的`lambda`函数用于提取输入到函数的键值对的`age`键的值，据此就可以理解了。

### `lambda`函数

`lambda` 函数是 Python 中一种简洁的匿名函数，它可以在需要函数对象的地方临时定义并使用，无需显式定义一个完整的函数。`lambda` 关键字**用于创建小型的、一次性使用的匿名函数**。其基本语法如下：

```python
lambda 参数列表: 表达式
```

- **参数列表**：这是传递给函数的参数，可以有零个或多个参数，参数之间用逗号分隔。
- **表达式**：这是函数要返回的值，`lambda` 函数只能包含一个表达式，且该表达式的结果会被自动返回。

结合这里对 `lambda` 函数的表达，就容易理解“排序函数的 `key` 参数”里为什么经常使用到它了：它可以快捷地创建简单函数，无须在前面重新定义与维护。

多参数的`lambda` 函数定义也是很自然的

```python
lambda x, y: x < y
```

和普通的函数一样，`lambda` 函数可以用于函数的返回值（后面可以像正常的函数一样调用），从而实现一些有趣的功能，如

```python
def multiplier(factor):
    return lambda x: x * factor

double = multiplier(2)
triple = multiplier(3)

print(double(5))  # 输出 10
print(triple(5))  # 输出 15
```
