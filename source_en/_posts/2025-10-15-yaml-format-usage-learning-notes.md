---
title: 'YAML: Syntax, Use Cases, and Practical Parsing'
title_zh: YAML 格式与使用速成
date: 2025-10-15 23:55:21 +0800
categories:
- Programming
- Computer Science Fundamentals
tags:
- YAML
- Data Engineering
author: Hyacehila
mathjax: false
hidden: true
excerpt: A practical note on YAML syntax, configuration use cases, Python/JavaScript parsing, and common pitfalls such as
  indentation, implicit typing, anchors, and duplicate keys.
description: A practical note on YAML syntax, configuration use cases, Python/JavaScript parsing, and common pitfalls such
  as indentation, implicit typing, anchors, and duplicate keys.
excerpt_zh: 整理 YAML 的基本语法、配置文件场景、Python/JS 读写方法，以及缩进、类型推断、锚点、重复 key 等常见坑。
permalink: /blog/2025/10/14/yaml-format-usage-learning-notes/
lang: en
translation_key: 2025-10-15-yaml-format-usage-learning-notes
translation_status: machine
translation_source_hash: 6a747caddf71050820894634111eb50ba1814e97d76bab2873e2d23953f0fd5e
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<h2>What's a Yaml?</h2>
<p>The questions in this article can also be addressed<a href="/en/blog/2025/10/14/json-and-jsonl-learning-notes/">JSON and JSONL Speed</a>、<a href="/en/blog/2024/07/29/sql-learning-notes/">SQL Foundation: Query, Aggregation, JOIN and Window Functions</a>How the concept of a relatively close read together is developed in different contexts.</p>
<p>YAML's full name was first understood as Yet Another Markup Language, later interpreted as YaML Ain&#39;t Markup Language. The name itself is interesting: it starts as a sign language, and then stresses that it is not a mark language, but a data sequence format. In short, the YamL is used to express the data structure, the most common suffix is <code>.yaml</code> or <code>.yml</code>。</p>
<p>If JSON's goal is to stabilize the machine and exchange data, YAML is more comfortable with configuration. It does not require a double quote for each key, nor does it need to write commas at the end of each, a hierarchy by indentation, and an annotated note can be written directly in the document. These advantages are clear for the configuration document: one can see the hierarchy at a glance, and changing one parameter is not easily interrupted by a pile of brackets and commas.</p>
<p>So we often see YAML in the configuration scene: the front matter of blog articles, GitHub Actions workflow, Docker Company, Kubernets Manifest, Ansible Playbook, model training configuration, service deployment configuration, all of which are preferred. It's not a core value "more advanced than JSON" but "more suitable for handwritten and maintenance."</p>
<p>YAML and JSON are close. Many Yaml files express data structures that are eventually deciphered to the normal objects of the program Map, List, String, Nuber, Boolean, Null. So, YAML is not a mysterious format, and it's still going back to the data structure that the program can handle.</p>
<h2>Basic YamL</h2>
<p>The most basic structure for YAML is the key pair. The colon is key on the left, and the colon is value on the right.</p>
<pre><code class="language-yaml">name: Jake
age: 25
city: Wuhan
is_student: true
</code></pre>
<p>After this, it becomes, in general, the object:</p>
<pre><code class="language-json">{
  &quot;name&quot;: &quot;Jake&quot;,
  &quot;age&quot;: 25,
  &quot;city&quot;: &quot;Wuhan&quot;,
  &quot;is_student&quot;: true
}
</code></pre>
<p>The level of YAML is indented. Usually use two spaces, not tab. This is a very important habit, because the YamL is sensitive to indentation, and the data structure is wrong.</p>
<pre><code class="language-yaml">user:
  name: Jake
  age: 25
  profile:
    email: jake@example.com
    location: Wuhan
</code></pre>
<p>In this example,<code>name</code> and <code>age</code> belong <code>user</code>，<code>email</code> and <code>location</code> belong <code>profile</code>I'm sorry. YAML does not need <code>{}</code> To wrap objects, to indent the structure to naturally expand.</p>
<p>List with Short Line <code>-</code> - Show. Short-wire lines themselves follow indentation rules.</p>
<pre><code class="language-yaml">hobbies:
  - Swimming
  - Basketball
  - Reading
</code></pre>
<p>Object arrays are also common. A group of blog posts can be written as follows:</p>
<pre><code class="language-yaml">posts:
  - id: 1
    title: 第一篇文章
    tags:
      - Python
      - Data
    published: true
  - id: 2
    title: 第二篇文章
    tags:
      - YAML
      - Config
    published: false
</code></pre>
<p>It's like the object array in JSON, but YAML saved brackets, commas and quotes. The omission of those symbols made the document easier to read, but the price was that indentation had to be more serious.</p>
<p>YamL supports the comment. Comment from <code>#</code> Start, end of line.</p>
<pre><code class="language-yaml"># 站点基本信息
site:
  title: Hyacehila
  language: zh-CN  # 默认语言
</code></pre>
<p>The comment is one of the important reasons why YaML is well suited to the profile. JSON cannot write an comment in standard formats, so many items are added to JSONC or are written in the document. YAML can place the explanation directly next to the configuration.</p>
<p>The string in YAML is usually not accompanied by quotation marks.</p>
<pre><code class="language-yaml">title: YAML 格式与使用速成
path: /blog/2026/07/06/yaml-format-usage-learning-notes/
</code></pre>
<p>But when the string contains elements that are easily misunderstood, it is advisable to add a reference sign. Examples include colons, wells, special booleans, back and forth spaces, version numbers, times, dates, etc.</p>
<pre><code class="language-yaml">title: &quot;YAML: Syntax, Use Cases, and Practical Parsing&quot;
version: &quot;1.0&quot;
answer: &quot;no&quot;
created_at: &quot;2026-07-06 20:04:07&quot;
</code></pre>
<p>YAML automatically resolves some values into numbers, booleans or empty values.</p>
<pre><code class="language-yaml">count: 10
ratio: 0.8
enabled: true
disabled: false
empty_value: null
also_empty:
</code></pre>
<p>Here. <code>count</code> It's the integer number.<code>ratio</code> It's floating point number.<code>enabled</code> It's a boolean value.<code>empty_value</code> and <code>also_empty</code> Could be parsed to empty values. This automatic type of extrapolation is convenient, but it is also one of the most common pits in the YamL, which will be discussed separately later.</p>
<h2>Multiline String</h2>
<p>YAML handles long text much more comfortable than JSON. The two most common ways are: <code>|</code> and <code>&gt;</code>。</p>
<p><code>|</code> indicates that the line is reserved. The text is basically retained as it is transposed.</p>
<pre><code class="language-yaml">description: |
  这是第一行。
  这是第二行。
  这是第三行。
</code></pre>
<p>This is appropriate to save scripts, certificates, hint templates, email body, Markdown clips, etc., which require a line break.</p>
<pre><code class="language-yaml">script: |
  npm install
  npm run build
  npm run check:i18n
</code></pre>
<p><code>&gt;</code> is a folding line. Multiline text is folded into a more nuanced form.</p>
<pre><code class="language-yaml">summary: &gt;
  YAML 很适合写配置文件，
  因为它比 JSON 更方便人类阅读和编辑，
  但它也更依赖缩进和解析规则。
</code></pre>
<p>If you want to keep a long description in the profile file from being too long, you can't be sure that the profile is too long.<code>&gt;</code> It'll be more natural. If you really need to keep the boundaries of each line, such as shell scripts, prompt templates, Nginx configurations, you should use <code>|</code>。</p>
<p>The difference is important in engineering. Many times the profile does not contain simple parameters, but rather a command, a template, a description text. Wrong. <code>|</code> and <code>&gt;</code>And the strings that the program sees will be different from what you think.</p>
<h2>YAML common use of scenes</h2>
<p>The first very daily scene is the front matter of blog articles. Many static blog systems will start with a YamL metadata section in the Markdown file, informing the system about the title, date, classification, label, summary of the article.</p>
<pre><code class="language-yaml">---
title: &quot;YAML 格式与使用速成&quot;
date: 2026-07-06 20:04:07 +0800
categories: [&quot;Programming&quot;, &quot;Computer Science Fundamentals&quot;]
tags: [&quot;Learning Notes&quot;, &quot;YAML&quot;]
hidden: true
---
</code></pre>
<p>The text here is still Markdown, but this opening part. <code>---</code> The area will be analyzed as YAML first. The fields are read in the theme, archive, tab page, front page card.</p>
<p>The second common scene is GitHub Actions. CI/CD configuration is particularly appropriate for YAML because it has a clear hierarchy: what triggers an event, which system runs, which job, which steps each job has.</p>
<pre><code class="language-yaml">name: build

on:
  push:
    branches:
      - master

jobs:
  site:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4
      - name: Install dependencies
        run: npm install
      - name: Build
        run: npm run build
</code></pre>
<p>This configuration, if written in JSON, will have a large number of brackets and string quotation marks; it will be written in YAML, more like a readable implementation list.</p>
<p>The third scene is Docker Company. It uses YAML to describe the relationship between multiple services.</p>
<pre><code class="language-yaml">services:
  web:
    image: nginx:latest
    ports:
      - &quot;8080:80&quot;
    volumes:
      - ./site:/usr/share/nginx/html:ro

  redis:
    image: redis:7
    restart: unless-stopped
</code></pre>
<p>Here, the YAML expression is just enough: the service name is key, and mirrors, ports, rolls, restarts are fields. It's very close to "What components are the systems?"</p>
<p>The fourth scene is Kubernetes. The Kubernetes resource audience is usually YAML prime.</p>
<pre><code class="language-yaml">apiVersion: apps/v1
kind: Deployment
metadata:
  name: blog
spec:
  replicas: 2
  selector:
    matchLabels:
      app: blog
  template:
    metadata:
      labels:
        app: blog
    spec:
      containers:
        - name: blog
          image: hyacehila/blog:latest
          ports:
            - containerPort: 4000
</code></pre>
<p>The Kubernetes Yaml also explains another aspect of YAML: When the configuration is too complex, it becomes long and easily duplicates the paste. YAML is not a silver bullet in a complex system, it just makes the configuration object more easily edited directly.</p>
<p>The fifth scenario is to apply its own configuration file.</p>
<pre><code class="language-yaml">server:
  host: 0.0.0.0
  port: 8080

database:
  url: postgresql://localhost:5432/app
  pool_size: 10
  timeout_seconds: 30

features:
  enable_cache: true
  enable_experiment: false
</code></pre>
<p>This configuration is usually eventually read into a dictionary or object. The application should not simply treat YAML as a string, but should read it in the solver, then bind it with type check, schema or default value logic.</p>
<h2>How to deal with YAML</h2>
<p>The most common YamL library in Python is Pyyaml. Use YAML first when reading <code>safe_load</code>, do not use unsafe loads.</p>
<pre><code class="language-python">import yaml

raw = &quot;&quot;&quot;
server:
  host: 0.0.0.0
  port: 8080
features:
  enable_cache: true
&quot;&quot;&quot;

config = yaml.safe_load(raw)

print(config[&quot;server&quot;][&quot;host&quot;])
print(config[&quot;server&quot;][&quot;port&quot;])
print(config[&quot;features&quot;][&quot;enable_cache&quot;])
</code></pre>
<p><code>yaml.safe_load()</code> The YAML string is to be resolved to Python objects. The general correspondence is as follows:</p>
<ul>
<li>YAML mapping -&gt; Python <code>dict</code></li>
<li>YAML sequence -&gt; Python <code>list</code></li>
<li>YAML string -&gt; Python <code>str</code></li>
<li>YAML integer -&gt; Python <code>int</code></li>
<li>YAML float -&gt; Python <code>float</code></li>
<li>YAML boolean -&gt; Python <code>bool</code></li>
<li>YAML null -&gt; Python <code>None</code></li>
</ul>
<p>If it is read from a document, it is written directly.</p>
<pre><code class="language-python">import yaml

file_path = &quot;config.yaml&quot;

with open(file_path, &quot;r&quot;, encoding=&quot;utf-8&quot;) as f:
    config = yaml.safe_load(f)

print(config)
</code></pre>
<p>Write Yaml to use <code>safe_dump</code>I'm sorry. If you have Chinese, you usually have to set it up. <code>allow_unicode=True</code>Otherwise, Chinese might be replaced.</p>
<pre><code class="language-python">import yaml

config = {
    &quot;site&quot;: {
        &quot;title&quot;: &quot;YAML 格式与使用速成&quot;,
        &quot;language&quot;: &quot;zh-CN&quot;,
    },
    &quot;features&quot;: {
        &quot;search&quot;: True,
        &quot;comments&quot;: False,
    },
}

with open(&quot;config.yaml&quot;, &quot;w&quot;, encoding=&quot;utf-8&quot;) as f:
    yaml.safe_dump(
        config,
        f,
        allow_unicode=True,
        sort_keys=False,
    )
</code></pre>
<p><code>sort_keys=False</code> It's common. The configuration file is often readable, the field order is meaningful and does not necessarily want the library to be automatically alphabetical.</p>
<p>JavaScript or Node.js, available <code>js-yaml</code> This kind of library.</p>
<pre><code class="language-javascript">const yaml = require(&quot;js-yaml&quot;);

const raw = `
server:
  host: 0.0.0.0
  port: 8080
features:
  enable_cache: true
`;

const config = yaml.load(raw);

console.log(config.server.host);
console.log(config.server.port);
console.log(config.features.enable_cache);
</code></pre>
<p>When read from a file, you can cooperate <code>fs.readFileSync</code>。</p>
<pre><code class="language-javascript">const fs = require(&quot;fs&quot;);
const yaml = require(&quot;js-yaml&quot;);

const raw = fs.readFileSync(&quot;config.yaml&quot;, &quot;utf8&quot;);
const config = yaml.load(raw);

console.log(config);
</code></pre>
<p>Write YAML <code>dump</code>。</p>
<pre><code class="language-javascript">const fs = require(&quot;fs&quot;);
const yaml = require(&quot;js-yaml&quot;);

const config = {
  site: {
    title: &quot;YAML 格式与使用速成&quot;,
    language: &quot;zh-CN&quot;,
  },
  features: {
    search: true,
    comments: false,
  },
};

const output = yaml.dump(config, {
  lineWidth: 100,
  noRefs: true,
});

fs.writeFileSync(&quot;config.yaml&quot;, output, &quot;utf8&quot;);
</code></pre>
<p>When you write a profile, reading and writing is not usually the hardest. And even harder: how to figure out how to prove it is the structure you want. In the real project, it is advisable to add a schema or a visible validation to the YaML configuration. Like what? <code>port</code> It must be numbers,<code>host</code> Must be a string,<code>features</code> Only allowed switches can appear below. Do not default on the fact that the YamL file must be a valid configuration because it looks like a configuration.</p>
<h2>Yaml's pit</h2>
<p>YAML's first pit is indented. JSON expresses the hierarchy in brackets, and the error in brackets is obvious; YAML expresses the hierarchy in indentation, and the error in a space is sometimes difficult to find in the naked eye.</p>
<pre><code class="language-yaml">user:
  name: Jake
  profile:
    email: jake@example.com
    city: Wuhan
</code></pre>
<p>If you don't, it's written like this:</p>
<pre><code class="language-yaml">user:
  name: Jake
  profile:
    email: jake@example.com
  city: Wuhan
</code></pre>
<p><code>city</code> I'm not a part of it anymore. <code>profile</code>, instead of the peace <code>profile</code> Same grade. It may still be legal, but the semantics have changed. This is the most problematic because the solver will not always report the error, but will only faithfully decipher the other structure.</p>
<p>The second pit is tab. YAML Indents Do Not Use Tab, Unique Space. Team best configure editor, auto-convert tab to spaces and display invisible characters. It looks like a little habit, and it's gonna take a lot of weird mistakes.</p>
<p>The third pit is of a hidden type. YAML automatically guesses the type, which is convenient in simple configuration, dangerous in border scenes. For example, the following values, preferably with quotation marks:</p>
<pre><code class="language-yaml">version: &quot;1.0&quot;
answer: &quot;no&quot;
switch: &quot;on&quot;
date: &quot;2026-07-06&quot;
time: &quot;20:04:07&quot;
hex_like: &quot;0x10&quot;
</code></pre>
<p>Different YaML versions, different solvers, different library options may not be interpreted in the same way as certain values. To reduce uncertainty, you can add a quote directly to any you want it to keep the value of the string. In particular, ID, version number, date, count, command parameters should not be given to the decryptator to guess.</p>
<p>The fourth pit is the colon and the well. Yaml <code>:</code> and <code>#</code> It's a grammar. If they appear in a string, it is advisable to add quotation marks.</p>
<pre><code class="language-yaml">title: &quot;YAML: Syntax, Use Cases, and Practical Parsing&quot;
command: &quot;echo hello # this is not a yaml comment&quot;
url: &quot;https://example.com/a:b&quot;
</code></pre>
<p>The fifth pit is a repeat key. This configuration below looks like it's only written twice. <code>port</code>：</p>
<pre><code class="language-yaml">server:
  port: 8080
  port: 9000
</code></pre>
<p>But many of the solvers will simply keep the latter value, the previous one being overwritten. More troublesomely, this may not be a mistake. For the configuration file, repetition of key often means that there is a problem with copying paste or merging configurations. Better use the Linter or schema tools to stop early.</p>
<p>The sixth pit is anchor and aliases. YamL support <code>&amp;</code> Define anchor, use <code>*</code> It's still working. <code>&lt;&lt;</code> Merges fields.</p>
<pre><code class="language-yaml">defaults: &amp;defaults
  image: node:22
  restart: unless-stopped
  environment:
    NODE_ENV: production

services:
  api:
    &lt;&lt;: *defaults
    command: npm run start:api
  worker:
    &lt;&lt;: *defaults
    command: npm run start:worker
</code></pre>
<p>This capacity is useful and reduces duplication. But it also allows the document to move slowly from " configuration " to " configuration with a logic of expansion " . If a YamL needs readers to leap, merge, and cover in their minds, the cost of maintenance increases. The anchor is suitable for a small number of reuses and does not fit the configuration into a puzzle.</p>
<p>The seventh pit is multi-document. YAML, a file can be used. <code>---</code> Splits multiple documents.</p>
<pre><code class="language-yaml">---
kind: ConfigMap
metadata:
  name: app-config
---
kind: Secret
metadata:
  name: app-secret
</code></pre>
<p>It's common in Kubernetes, but when you're deciphered, it's normal. <code>safe_load</code> Could be suitable for single documents only. If there are multiple documents in the file, it's usually used in Python <code>safe_load_all</code>。</p>
<pre><code class="language-python">import yaml

with open(&quot;resources.yaml&quot;, &quot;r&quot;, encoding=&quot;utf-8&quot;) as f:
    documents = list(yaml.safe_load_all(f))

for doc in documents:
    print(doc[&quot;kind&quot;])
</code></pre>
<p>The eighth pit is safe. Do not use unsafe resolution for untrustworthy sources of Yaml. Some of the YAML advanced abilities and object construction capabilities may allow the process to go beyond “reading data”. For the ordinary profile,<code>safe_load</code> Such a safe resolution is sufficient. The more YAML is uploaded by users, external services, network input, the more conservative it should be.</p>
<h2>When won't you need it, Yaml?</h2>
<p>YAML is a good human-written configuration, but does not mean that all structured data should be used.</p>
<p>If data is transmitted between services, give priority to JSON. JSON is more stringent, solvers are more uniform, more ecologically stable, and more suitable for API requests and responses. HTTP API returns to YamL is not an option, but in most of the scenes, JSON will make the caller easier.</p>
<p>For large-scale logs, training data, reptile results, model input output records, priority is given to JSONL. JSONL is a complete JSON object in each line, which is natural for addition, streaming and segment processing. YAML can express object arrays, but if the data is large, an entire YAML file will become less suitable for current processing.</p>
<p>If it's a very complex configuration that needs to be robust, be careful with YamL. Many of the engineering accidents were not because YAML was unable to express them, but because it was too expressive, and the final project used a configuration format as half a DSL. Once a large number of anchors, templates, conditions, inheritance, rules of coverage have begun to emerge in the configuration, it is difficult for one to judge what the configuration ultimately takes effect.</p>
<p>There are several more stable approaches at this time: binding the configuration with tools such as JSON Schema, OpenAPI, Pydantic, Zod; moving complex logic into the normal code; or providing the generator and the checker for the configuration, rather than allowing the person to write all the details by hand.</p>
<p>So the YamL border can be understood as follows:<strong>It is appropriate to express static, hierarchically clear, primarily manned configurations; it is not suitable to carry high frequency transmissions, large-scale current data and overly complex business logic.</strong></p>
<h2>Summary</h2>
<p>The advantages of YAML are straightforward: less brackets, less quotation marks, more commentable, clear hierarchy, and appropriate configuration. It allows for the natural expression of objects, lists, strings, numbers, booleans and empty values in a text file, and for the comfortable writing of multiple lines of text.</p>
<p>But the dangers of YAML are also due to these advantages. It relies too much on indentation, it is too confident in people's reading instincts, and it also makes automatic type extrapolations. A lot of Yaml documents look so clear, they don't really solve the tree in your head. Several habits are to be developed when writing YAML: to unify two spaces indents without tab; to add quotation marks if the string is uncertain; to complex configurations plus schema; to check repeat key with the loger; to use security resolution when reading external YAML.</p>
<p>YAML is more like a hand-held configuration note than JSON; YAML is less suitable for current data than JSON; and YAML should not have too much logic than code. Put it in the right place, it'll be very easy. It becomes a maintenance burden, slowly, as an inescapable configuration language.</p>
<p>I think the simplest way to judge is if this document is written primarily by people, read by programs, and is structured at a more complex level than a plain text, and not so complicated as to require code writing, then Yamal is a good choice. If this document is mainly machine-generated, machine-consuming, HF-exchange, strong schema, or fluent processing, then it should be carefully considered, JSON, JSONL, or a more explicit configuration system.</p>
