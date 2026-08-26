---
title: 'JSON and JSONL: Objects, Arrays, and Streaming Records'
title_zh: JSON 与 JSONL 速成
date: 2025-10-14 23:54:21 +0800
categories:
- Programming
- Computer Science Fundamentals
tags:
- JSON
- Data Engineering
author: Hyacehila
mathjax: false
hidden: true
excerpt: Covers objects, arrays, streaming processing, and basic Python json.load, json.loads, dump, and dumps usage.
description: Covers objects, arrays, streaming processing, and basic Python json.load, json.loads, dump, and dumps usage.
excerpt_zh: 整理对象、数组、流式处理，以及 Python 中 json.load、json.loads、dump、dumps 的基本用法。
permalink: /blog/2025/10/14/json-and-jsonl-learning-notes/
lang: en
translation_key: 2025-10-14-json-and-jsonl-objects-arrays-streaming
translation_status: machine
translation_source_hash: c0736d7c94ebee85a1e83b7e86801393f92a2fa3c33bc860abe37e86eac32c91
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<h2>What's JSON?</h2>
<p>JS = JavaScript; ON =Object Notation, so JSON means the JavaScript object symbol system, but he already has almost all language versions that are used in any time that both machines and humans are required to read twice.<code>.json</code> It means as a suffix.</p>
<p>The questions in this article can also be addressed<a href="/en/blog/2025/10/14/yaml-format-usage-learning-notes/">YAML Format and Usage Speed</a>、<a href="/en/blog/2024/07/29/sql-learning-notes/">SQL Foundation: Query, Aggregation, JOIN and Window Functions</a>How the concept of a relatively close read together is developed in different contexts.</p>
<p>JSON uses <code>{}</code>As a core object, a symbol, and using all key values for tissue data, we use commas to separate the columns within an object, except for the last.</p>
<pre><code class="language-json">{
&quot;name&quot; : &quot;Jake&quot;, //这是一个普通字符串
&quot;age&quot; : 25,  //这是一个数字
&quot;hobbies&quot;:[&quot;Swimming&quot;,&quot;Basketball&quot;] //这是一个数组结构
}
</code></pre>
<p>JSON allows nesting, which allows you to embed other objects in a JSON object value, which allows us to use multilayered nesting to manage and use complex data.</p>
<p>Except for this use.<code>{}</code>Aside from the structure, JSON has a special object array structure designed to store a large number of duplicate entries in similar formats, which he uses directly in JSON files. <code>[]</code> Store objects in a single array, which are often in the form of JSON because of the embedded devices that support nature.</p>
<pre><code class="language-json">[
  {
    &quot;id&quot;: 1,
    &quot;text&quot;: &quot;这是我的第一篇博文，内容很精彩。&quot;,
    &quot;images&quot;: [
      &quot;/posts/2023/10/20/img1.jpg&quot;,
      &quot;/posts/2023/10/20/img2.jpg&quot;
    ],
    &quot;timestamp&quot;: &quot;2023-10-20T10:00:00Z&quot;,
    &quot;location&quot;: &quot;北京&quot;
  },
  {
    &quot;id&quot;: 2,
    &quot;text&quot;: &quot;今天去了长城，风景真不错。&quot;,
    &quot;images&quot;: [
      &quot;/posts/2023/10/21/great_wall_a.jpg&quot;,
      &quot;/posts/2023/10/21/great_wall_b.jpg&quot;,
      &quot;/posts/2023/10/21/great_wall_c.jpg&quot;
    ],
    &quot;timestamp&quot;: &quot;2023-10-21T15:30:00Z&quot;,
    &quot;location&quot;: &quot;北京, 八达岭长城&quot;
  },
  // ... 省略 9997 条
]
</code></pre>
<h2>JSONL File</h2>
<p>When a single JSON file (usually a JSON file for object arrays) becomes very large (e.g. hundreds of MB or several GBs), it is possible to use JSON Lines format when the memory is unrealistic. <code>.jsonl</code> He's used to deal with questions like the object array.</p>
<p><code>.jsonl</code> Each line of the file is a separate, complete JSON object, with no commas between the rows and the line, and the entire file is not squared. <code>[]</code>. <strong>Note that the JSON objects in the JSONL file are not embedded in the system, and we don't need him to handle the multi-layer resolution.</strong></p>
<pre><code class="language-json">{&quot;id&quot;: 1, &quot;text&quot;: &quot;这是我的第一篇博文，内容很精彩。&quot;, &quot;images&quot;: [&quot;/posts/2023/10/20/img1.jpg&quot;, &quot;/posts/2023/10/20/img2.jpg&quot;], &quot;timestamp&quot;: &quot;2023-10-20T10:00:00Z&quot;, &quot;location&quot;: &quot;北京&quot;}

{&quot;id&quot;: 2, &quot;text&quot;: &quot;今天去了长城，风景真不错。&quot;, &quot;images&quot;: [&quot;/posts/2023/10/21/great_wall_a.jpg&quot;, &quot;/posts/2023/10/21/great_wall_b.jpg&quot;, &quot;/posts/2023/10/21/great_wall_c.jpg&quot;], &quot;timestamp&quot;: &quot;2023-10-21T15:30:00Z&quot;, &quot;location&quot;: &quot;北京, 八达岭长城&quot;}

{&quot;id&quot;: 3, &quot;text&quot;: &quot;分享一个技术心得...&quot;, &quot;images&quot;: [], &quot;timestamp&quot;: &quot;2023-10-22T09:00:00Z&quot;, &quot;location&quot;: &quot;上海&quot;}
// ... 每行一篇博文
</code></pre>
<p><code>.jsonl</code> Format support<strong>Fluid processing</strong>, read and processed by line, without a single load of the entire file to the memory, which is extremely low ... and<strong>Easy to add</strong>Just add a line to the end.</p>
<p><code>.jsonl</code> The format is essentially multiple JSON files, which cannot be accessed at random, only from the first line, individually, and cannot be analysed directly using a JSON-like method.</p>
<h2>How to deal with JSON</h2>
<p>We used JS and Python most often to process JSON data, of which<code>JSON.parse</code>This originally supported JS function can handle it as an JS-supported object, and then choose to use square brackets or dots to access the embedded object.</p>
<p>Here's a more detailed explanation of Python's position.<code>json</code>These two very important and easily confused functions in the module:<code>json.loads()</code> and <code>json.load()</code>They converted the JSON files to the Python dictionary, then used the square brackets in the Python dictionary. <code>json.JSONDecodeError</code></p>
<p>A sentence summing up the core distinction:<strong><code>s</code> Representative <code>string</code>(strings) <code>json.loads()</code> to parse strings, and <code>json.load()</code> For parse files.</strong> </p>
<p><code>json.loads()</code> Will <code>str</code>, <code>bytes</code> or <code>bytearray</code> Type of example, containing JSON documents, inverse sequence to Python objects, generally used for API response processing.</p>
<pre><code class="language-python">import json
# 其中 s 是我们希望解析的字符串,他符合JSON的结构语言
json.loads(s, *, cls=None, object_hook=None, parse_float=None, parse_int=None, parse_constant=None, object_pairs_hook=None, **kw)
</code></pre>
<p>The following conversion methods are generally followed:</p>
<ul>
<li>object <code>{}</code> -&gt; dict <code>dict</code></li>
<li>array <code>[]</code> -&gt; list <code>list</code></li>
<li>string <code>&quot;&quot;</code> -&gt; str <code>str</code></li>
<li>number (int) -&gt; int <code>int</code></li>
<li>number (real) -&gt; float <code>float</code></li>
<li><code>true</code> -&gt; <code>True</code></li>
<li><code>false</code> -&gt; <code>False</code></li>
<li><code>null</code> -&gt; <code>None</code></li>
</ul>
<p><code>load</code> It's a function of one.<strong>File-Category Object</strong>. Here's the file class object' for any support <code>.read()</code> The most common object is the person who passed the method. <code>open()</code> function opens the file.</p>
<pre><code class="language-python">import json
# 其中 fp 是我们希望解析的文件,他符合JSON的结构语言
json.load(fp, *, cls=None, object_hook=None, parse_float=None, parse_int=None, parse_constant=None, object_pairs_hook=None, **kw)
</code></pre>
<p>Use the code below, you need to open the file first, then use it <code>json.load</code> Processing</p>
<pre><code class="language-python">import json

# 文件路径
file_path = &#39;data.json&#39;

# 使用 &#39;with open&#39; 语句打开文件，这是最佳实践
# 它能确保文件在操作完成后被自动关闭，即使发生错误也不例外
try:
    with open(file_path, &#39;r&#39;, encoding=&#39;utf-8&#39;) as f:
        # 使用 json.load() 从文件对象 f 中解析JSON数据
        python_data = json.load(f)

        # 查看解析后的Python对象及其类型
        print(&quot;从文件解析后的Python对象:&quot;)
        print(python_data)
        print(f&quot;类型: {type(python_data)}&quot;)

        # 操作数据
        print(f&quot;用户邮箱: {python_data[&#39;email&#39;]}&quot;)
        print(f&quot;标签数量: {len(python_data[&#39;tags&#39;])}&quot;)

except FileNotFoundError:
    print(f&quot;错误：文件 {file_path} 未找到。&quot;)
except json.JSONDecodeError as e:
    print(f&quot;JSON解析错误: {e}&quot;)
</code></pre>
<h2>How to deal with JSONL</h2>
<p>First of all, let's be clear. The whole JSONL document itself.<strong>No, it's not.</strong>A valid JSON object or array. Not available <code>json.load()</code> Read the whole document once, because it's wrong, saying it's not in the right format.<strong>Read by line and use each line <code>json.loads()</code></strong>.</p>
<p>Assuming there's a JSONL file.</p>
<pre><code class="language-json"># data.jsonl
{&quot;name&quot;: &quot;Alice&quot;, &quot;age&quot;: 30, &quot;city&quot;: &quot;New York&quot;}
{&quot;name&quot;: &quot;Bob&quot;, &quot;age&quot;: 25, &quot;city&quot;: &quot;Los Angeles&quot;}
{&quot;name&quot;: &quot;Charlie&quot;, &quot;age&quot;: 35, &quot;city&quot;: &quot;Chicago&quot;}
</code></pre>
<p>Read it with Python, of course, but it's just one example, not enough for Robustness:</p>
<pre><code class="language-python">import json

# 存储解析后的数据
data_list = []

# 使用 &#39;with&#39; 语句可以确保文件被正确关闭
# 指定 encoding=&#39;utf-8&#39; 是一个好习惯，可以避免编码问题
with open(&#39;data.jsonl&#39;, &#39;r&#39;, encoding=&#39;utf-8&#39;) as f:
    for line in f:
        # json.loads() 将每一行的字符串转换为 Python 字典
        data = json.loads(line)
        data_list.append(data)
        print(f&quot;读取到一行数据: {data}&quot;)

print(&quot;\n所有数据已加载到列表中:&quot;)
print(data_list)
</code></pre>
<p><strong>Like JSONL itself, we finally got a list of dictionarys in Python.</strong></p>
<h2>Write JSON.</h2>
<p>Write Standard<code>.json</code> , use this method when you have a complete data structure (e.g. a list of all the Bodicals) and you want to store it as a single, formatted JSON file.</p>
<p><strong>Key Functions</strong>: <code>json.dump(data, file_object)</code> </p>
<ul>
<li><code>data</code>: Python objects (dictionaries, lists, etc.) that you want to write</li>
<li><code>file_object</code>: An open file object</li>
</ul>
<pre><code class="language-python">import json

# 我们的Python数据结构：一个包含多个字典的列表
blog_posts = [
    {
        &quot;id&quot;: 1,
        &quot;text&quot;: &quot;今天天气真好&quot;,
        &quot;images&quot;: [&quot;sunny.jpg&quot;],
        &quot;timestamp&quot;: &quot;2023-10-20T10:00:00Z&quot;
    },
    {
        &quot;id&quot;: 2,
        &quot;text&quot;: &quot;分享一篇好文章&quot;,
        &quot;images&quot;: [],
        &quot;timestamp&quot;: &quot;2023-10-20T11:30:00Z&quot;
    },
    {
        &quot;id&quot;: 3,
        &quot;text&quot;: &quot;我的新宠物&quot;,
        &quot;images&quot;: [&quot;cat1.jpg&quot;, &quot;cat2.jpg&quot;],
        &quot;timestamp&quot;: &quot;2023-10-20T12:15:00Z&quot;
    }
]

# 将数据写入 posts.json 文件
with open(&#39;posts.json&#39;, &#39;w&#39;, encoding=&#39;utf-8&#39;) as f:
    # 使用 json.dump() 写入数据
    # indent=4 使文件格式化，易于阅读
    # ensure_ascii=False 确保中文字符能正常写入，而不是被转义
    json.dump(blog_posts, f, indent=4, ensure_ascii=False)

print(&quot;成功写入 posts.json 文件&quot;)
</code></pre>
<p>Result generated</p>
<pre><code class="language-json">[
    {
        &quot;id&quot;: 1,
        &quot;text&quot;: &quot;今天天气真好&quot;,
        &quot;images&quot;: [
            &quot;sunny.jpg&quot;
        ],
        &quot;timestamp&quot;: &quot;2023-10-20T10:00:00Z&quot;
    },
    {
        &quot;id&quot;: 2,
        &quot;text&quot;: &quot;分享一篇好文章&quot;,
        &quot;images&quot;: [],
        &quot;timestamp&quot;: &quot;2023-10-20T11:30:00Z&quot;
    },
    {
        &quot;id&quot;: 3,
        &quot;text&quot;: &quot;我的新宠物&quot;,
        &quot;images&quot;: [
            &quot;cat1.jpg&quot;,
            &quot;cat2.jpg&quot;
        ],
        &quot;timestamp&quot;: &quot;2023-10-20T12:15:00Z&quot;
    }
]
</code></pre>
<hr>
<p>Write<code>.jsonl</code>  When it takes a flow of data, or when it's very large, and it doesn't want to be loaded into memory once, use JSONL. Each line is an independent JSON object.</p>
<p><strong>Key Functions</strong>: <code>json.dumps(data)</code></p>
<ul>
<li><code>data</code>: Python objects you want to convert (usually a single dictionary)</li>
<li>This function returns a string in JSON format</li>
</ul>
<pre><code class="language-python">import json

# 相同的博文数据
blog_posts = [
    {
        &quot;id&quot;: 1,
        &quot;text&quot;: &quot;今天天气真好&quot;,
        &quot;images&quot;: [&quot;sunny.jpg&quot;],
        &quot;timestamp&quot;: &quot;2023-10-20T10:00:00Z&quot;
    },
    {
        &quot;id&quot;: 2,
        &quot;text&quot;: &quot;分享一篇好文章&quot;,
        &quot;images&quot;: [],
        &quot;timestamp&quot;: &quot;2023-10-20T11:30:00Z&quot;
    },
    {
        &quot;id&quot;: 3,
        &quot;text&quot;: &quot;我的新宠物&quot;,
        &quot;images&quot;: [&quot;cat1.jpg&quot;, &quot;cat2.jpg&quot;],
        &quot;timestamp&quot;: &quot;2023-10-20T12:15:00Z&quot;
    }
]

# 将数据逐行写入 posts.jsonl 文件
with open(&#39;posts.jsonl&#39;, &#39;w&#39;, encoding=&#39;utf-8&#39;) as f:
    # 遍历列表中的每一个字典（博文）
    for post in blog_posts:
        # 1. 使用 json.dumps() 将单个字典转换为JSON字符串
        json_string = json.dumps(post, ensure_ascii=False)

        # 2. 将字符串写入文件，并在末尾添加一个换行符 \n
        f.write(json_string + &#39;\n&#39;)

print(&quot;成功写入 posts.jsonl 文件&quot;)
</code></pre>
<p>Result generated</p>
<pre><code class="language-json">{&quot;id&quot;: 1, &quot;text&quot;: &quot;今天天气真好&quot;, &quot;images&quot;: [&quot;sunny.jpg&quot;], &quot;timestamp&quot;: &quot;2023-10-20T10:00:00Z&quot;}
{&quot;id&quot;: 2, &quot;text&quot;: &quot;分享一篇好文章&quot;, &quot;images&quot;: [], &quot;timestamp&quot;: &quot;2023-10-20T11:30:00Z&quot;}
{&quot;id&quot;: 3, &quot;text&quot;: &quot;我的新宠物&quot;, &quot;images&quot;: [&quot;cat1.jpg&quot;, &quot;cat2.jpg&quot;], &quot;timestamp&quot;: &quot;2023-10-20T12:15:00Z&quot;}
</code></pre>
