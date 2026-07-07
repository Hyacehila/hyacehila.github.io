---
title: "JSON 与 JSONL 速成"
title_en: "JSON and JSONL: Objects, Arrays, and Streaming Records"
date: 2025-10-14 23:54:21 +0800
categories: ["Programming", "Data & Databases"]
tags: ["JSON", "Data Engineering"]
author: Hyacehila
excerpt: "整理对象、数组、流式处理，以及 Python 中 json.load、json.loads、dump、dumps 的基本用法。"
excerpt_en: "Covers objects, arrays, streaming processing, and basic Python json.load, json.loads, dump, and dumps usage."
mathjax: false
hidden: true
permalink: '/blog/2025/10/14/json-and-jsonl-learning-notes/'
---

## 什么是JSON
JS = JavaScript ; ON = Object Notation , 故JSON的本身含义是JavaScript 对象符号系统,但是他已经兼容了几乎一切语言形式,在任何同时需要机器和人类进行双重阅读的时候,都会使用JSON 其使用`.json` 表示作为后缀.

JSON使用 `{}`作为核心对象表示符号,并且全部使用键值对组织数据,对于一个对象里面的各个子列,我们使用逗号分隔,除了最后一个.
```json
{
"name" : "Jake", //这是一个普通字符串
"age" : 25,  //这是一个数字
"hobbies":["Swimming","Basketball"] //这是一个数组结构
}
```

JSON允许嵌套,可以在一个JSON对象的值中嵌套其他对象,这使得我们可以使用多层的嵌套来管理和使用复杂的数据.

除了这种使用`{}`的结构以外, JSON还有一种特殊的对象数组结构,专门为存储大量类似格式的重复条目设计,他直接在JSON文件中使用 `[]` 存储一个个数组中的对象,由于支持自然的嵌套,这些对象往往也是JSON形式的.
```json
[
  {
    "id": 1,
    "text": "这是我的第一篇博文，内容很精彩。",
    "images": [
      "/posts/2023/10/20/img1.jpg",
      "/posts/2023/10/20/img2.jpg"
    ],
    "timestamp": "2023-10-20T10:00:00Z",
    "location": "北京"
  },
  {
    "id": 2,
    "text": "今天去了长城，风景真不错。",
    "images": [
      "/posts/2023/10/21/great_wall_a.jpg",
      "/posts/2023/10/21/great_wall_b.jpg",
      "/posts/2023/10/21/great_wall_c.jpg"
    ],
    "timestamp": "2023-10-21T15:30:00Z",
    "location": "北京, 八达岭长城"
  },
  // ... 省略 9997 条
]
```

## JSONL文件
当单个JSON文件(一般是对象数组的JSON文件)变得非常巨大（例如几百MB或几GB）,一次性读入内存不现实时,可以使用JSON Lines格式, `.jsonl` 他用来处理类似对象数组的问题.

`.jsonl` 文件中,每一行都是一个独立的、完整的JSON对象,行与行之间没有逗号,整个文件也没有外层的方括号 `[]`. **注意JSONL文件中的JSON对象不嵌套使用,我们不用他来处理需要多层解析的问题**

```json
{"id": 1, "text": "这是我的第一篇博文，内容很精彩。", "images": ["/posts/2023/10/20/img1.jpg", "/posts/2023/10/20/img2.jpg"], "timestamp": "2023-10-20T10:00:00Z", "location": "北京"}

{"id": 2, "text": "今天去了长城，风景真不错。", "images": ["/posts/2023/10/21/great_wall_a.jpg", "/posts/2023/10/21/great_wall_b.jpg", "/posts/2023/10/21/great_wall_c.jpg"], "timestamp": "2023-10-21T15:30:00Z", "location": "北京, 八达岭长城"}

{"id": 3, "text": "分享一个技术心得...", "images": [], "timestamp": "2023-10-22T09:00:00Z", "location": "上海"}
// ... 每行一篇博文
```

`.jsonl` 格式支持**流式处理**,可以逐行读取和处理,不需要一次性加载整个文件到内存,内存占用极低.并且**易于追加**,直接在末尾追加一行就可以.

`.jsonl` 格式本质上是多个 JSON 文件, 无法随机访问,只能从第一行开始逐个访问. 并且无法直接使用类似JSON的方法解析.

## 如何处理JSON
我们最经常使用JS和Python来处理JSON格式的数据,其中`JSON.parse`这个原生支持的JS函数可以将它处理为JS支持的对象,之后可以选择使用方括号或者点号来根据访问其中的嵌套对象.

下面详细解释一下Python中`json`模块里这两个非常重要且容易混淆的函数：`json.loads()` 和 `json.load()`. 他们将JSON文件转换为Python字典来处理,后续使用Python字典的方括号来处理. 解析错误的时候发生 `json.JSONDecodeError`

一句话总结核心区别：**`s` 代表 `string`（字符串）. `json.loads()` 用于解析字符串,而 `json.load()` 用于解析文件.** 

`json.loads()` 将 `str`, `bytes` 或 `bytearray` 类型的实例,其中包含JSON文档,反序列化为Python对象. 一般用于API的响应的处理.

```python
import json
# 其中 s 是我们希望解析的字符串,他符合JSON的结构语言
json.loads(s, *, cls=None, object_hook=None, parse_float=None, parse_int=None, parse_constant=None, object_pairs_hook=None, **kw)

```

一般遵循下面的转化方法
* object `{}` -> dict `dict`
* array `[]` -> list `list`
* string `""` -> str `str`
* number (int) -> int `int`
* number (real) -> float `float`
* `true` -> `True`
* `false` -> `False`
* `null` -> `None`

`load` 的作用是从一个**文件类对象**中读取JSON数据,并将其解析成Python对象.这里的“文件类对象”指的是任何支持 `.read()` 方法的对象,最常见的就是通过 `open()` 函数打开的文件.

```python
import json
# 其中 fp 是我们希望解析的文件,他符合JSON的结构语言
json.load(fp, *, cls=None, object_hook=None, parse_float=None, parse_int=None, parse_constant=None, object_pairs_hook=None, **kw)

```

使用下面的代码,需要先进行打开文件,然后再使用 `json.load` 处理
```python
import json

# 文件路径
file_path = 'data.json'

# 使用 'with open' 语句打开文件，这是最佳实践
# 它能确保文件在操作完成后被自动关闭，即使发生错误也不例外
try:
    with open(file_path, 'r', encoding='utf-8') as f:
        # 使用 json.load() 从文件对象 f 中解析JSON数据
        python_data = json.load(f)

        # 查看解析后的Python对象及其类型
        print("从文件解析后的Python对象:")
        print(python_data)
        print(f"类型: {type(python_data)}")

        # 操作数据
        print(f"用户邮箱: {python_data['email']}")
        print(f"标签数量: {len(python_data['tags'])}")

except FileNotFoundError:
    print(f"错误：文件 {file_path} 未找到。")
except json.JSONDecodeError as e:
    print(f"JSON解析错误: {e}")

```

## 如何处理JSONL
首先明确一点,整个JSONL文件本身**不是**一个有效的 JSON 对象或数组.不能用 `json.load()` 一次性读取整个文件,因为它会报错,说文件格式不正确.正确的做法是**逐行读取,并对每一行使用 `json.loads()`**.

假设有JSONL文件
```json
# data.jsonl
{"name": "Alice", "age": 30, "city": "New York"}
{"name": "Bob", "age": 25, "city": "Los Angeles"}
{"name": "Charlie", "age": 35, "city": "Chicago"}

```

用 Python 来读取它,当然这只是一个例子,不够Robustness：
```python
import json

# 存储解析后的数据
data_list = []

# 使用 'with' 语句可以确保文件被正确关闭
# 指定 encoding='utf-8' 是一个好习惯，可以避免编码问题
with open('data.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        # json.loads() 将每一行的字符串转换为 Python 字典
        data = json.loads(line)
        data_list.append(data)
        print(f"读取到一行数据: {data}")

print("\n所有数据已加载到列表中:")
print(data_list)

```
**如同JSONL本身的特性一样,我们最终获得了一个Python中的字典列表**

## 写JSON

写标准`.json` , 当你有一个完整的数据结构（比如一个包含所有博文的列表）,并希望将它存储为一个单一的、格式化的JSON文件时,使用此方法.

**关键函数**: `json.dump(data, file_object)` 
- `data`: 你要写入的Python对象（字典、列表等）
- `file_object`: 一个已打开的文件对象

```python
import json

# 我们的Python数据结构：一个包含多个字典的列表
blog_posts = [
    {
        "id": 1,
        "text": "今天天气真好",
        "images": ["sunny.jpg"],
        "timestamp": "2023-10-20T10:00:00Z"
    },
    {
        "id": 2,
        "text": "分享一篇好文章",
        "images": [],
        "timestamp": "2023-10-20T11:30:00Z"
    },
    {
        "id": 3,
        "text": "我的新宠物",
        "images": ["cat1.jpg", "cat2.jpg"],
        "timestamp": "2023-10-20T12:15:00Z"
    }
]

# 将数据写入 posts.json 文件
with open('posts.json', 'w', encoding='utf-8') as f:
    # 使用 json.dump() 写入数据
    # indent=4 使文件格式化，易于阅读
    # ensure_ascii=False 确保中文字符能正常写入，而不是被转义
    json.dump(blog_posts, f, indent=4, ensure_ascii=False)

print("成功写入 posts.json 文件")

```

生成的结果
```json
[
    {
        "id": 1,
        "text": "今天天气真好",
        "images": [
            "sunny.jpg"
        ],
        "timestamp": "2023-10-20T10:00:00Z"
    },
    {
        "id": 2,
        "text": "分享一篇好文章",
        "images": [],
        "timestamp": "2023-10-20T11:30:00Z"
    },
    {
        "id": 3,
        "text": "我的新宠物",
        "images": [
            "cat1.jpg",
            "cat2.jpg"
        ],
        "timestamp": "2023-10-20T12:15:00Z"
    }
]

```

---

写`.jsonl`  当需要流式处理数据,或者数据量非常大,不希望一次性加载到内存时,使用JSONL.每一行都是一个独立的JSON对象.

**关键函数**: `json.dumps(data)`
- `data`: 你要转换的Python对象（通常是单个字典）
- 该函数返回一个JSON格式的字符串

```python
import json

# 相同的博文数据
blog_posts = [
    {
        "id": 1,
        "text": "今天天气真好",
        "images": ["sunny.jpg"],
        "timestamp": "2023-10-20T10:00:00Z"
    },
    {
        "id": 2,
        "text": "分享一篇好文章",
        "images": [],
        "timestamp": "2023-10-20T11:30:00Z"
    },
    {
        "id": 3,
        "text": "我的新宠物",
        "images": ["cat1.jpg", "cat2.jpg"],
        "timestamp": "2023-10-20T12:15:00Z"
    }
]

# 将数据逐行写入 posts.jsonl 文件
with open('posts.jsonl', 'w', encoding='utf-8') as f:
    # 遍历列表中的每一个字典（博文）
    for post in blog_posts:
        # 1. 使用 json.dumps() 将单个字典转换为JSON字符串
        json_string = json.dumps(post, ensure_ascii=False)
        
        # 2. 将字符串写入文件，并在末尾添加一个换行符 \n
        f.write(json_string + '\n')

print("成功写入 posts.jsonl 文件")

```

生成的结果
```json
{"id": 1, "text": "今天天气真好", "images": ["sunny.jpg"], "timestamp": "2023-10-20T10:00:00Z"}
{"id": 2, "text": "分享一篇好文章", "images": [], "timestamp": "2023-10-20T11:30:00Z"}
{"id": 3, "text": "我的新宠物", "images": ["cat1.jpg", "cat2.jpg"], "timestamp": "2023-10-20T12:15:00Z"}

```
