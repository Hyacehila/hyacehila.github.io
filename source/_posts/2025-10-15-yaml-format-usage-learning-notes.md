---
title: "YAML 格式与使用速成"
title_en: "YAML: Syntax, Use Cases, and Practical Parsing"
date: 2025-10-15 23:55:21 +0800
categories: ["Programming", "Data & Databases"]
tags: ["YAML", "Data Engineering"]
author: Hyacehila
excerpt: "整理 YAML 的基本语法、配置文件场景、Python/JS 读写方法，以及缩进、类型推断、锚点、重复 key 等常见坑。"
excerpt_en: "A practical note on YAML syntax, configuration use cases, Python/JavaScript parsing, and common pitfalls such as indentation, implicit typing, anchors, and duplicate keys."
mathjax: false
hidden: true
permalink: '/blog/2025/10/14/yaml-format-usage-learning-notes/'
---

## 什么是 YAML

YAML 的全称最早可以理解为 Yet Another Markup Language，后来更常被解释成 YAML Ain't Markup Language。这个名字本身就很有意思：它一开始像一种标记语言，后来又强调自己不是标记语言，而是一种数据序列化格式。简单说，YAML 用来表达数据结构，最常见的后缀是 `.yaml` 或 `.yml`。

如果说 JSON 的目标是让机器稳定交换数据，那么 YAML 更偏向让人舒服地写配置。它不需要每个 key 都加双引号，也不需要在每一项后面写逗号，层级靠缩进表达，注释可以直接写在文件里。对于配置文件来说，这些优点非常明显：人可以一眼看到层级，改一个参数也不容易被一堆括号和逗号打断。

所以我们经常在配置场景里看到 YAML：博客文章的 front matter、GitHub Actions 工作流、Docker Compose、Kubernetes manifest、Ansible playbook、模型训练配置、服务部署配置，都喜欢用它。它的核心价值不是“比 JSON 更高级”，而是“更适合人手写和维护”。

YAML 和 JSON 的关系也很近。很多 YAML 文件表达出来的数据结构，最后都会被解析成程序里的 map、list、string、number、boolean、null 这些普通对象。也就是说，YAML 不是一种神秘格式，它最后仍然会回到程序能够处理的数据结构里。

## YAML 的基本写法

YAML 最基本的结构是键值对。冒号左边是 key，冒号右边是 value。

```yaml
name: Jake
age: 25
city: Wuhan
is_student: true
```

解析以后，它大致会变成这样的对象：

```json
{
  "name": "Jake",
  "age": 25,
  "city": "Wuhan",
  "is_student": true
}
```

YAML 的层级靠缩进表示。一般使用两个空格，不使用 tab。这个习惯非常重要，因为 YAML 对缩进敏感，缩进错了，数据结构就会变。

```yaml
user:
  name: Jake
  age: 25
  profile:
    email: jake@example.com
    location: Wuhan
```

上面这个例子里，`name` 和 `age` 属于 `user`，`email` 和 `location` 属于 `profile`。YAML 不需要 `{}` 来包住对象，而是用缩进让结构自然展开。

列表用短横线 `-` 表示。短横线本身也要遵守缩进规则。

```yaml
hobbies:
  - Swimming
  - Basketball
  - Reading
```

对象数组也很常见。比如一组博客文章，可以这样写：

```yaml
posts:
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
```

这和 JSON 里的对象数组很像，只是 YAML 把括号、逗号和引号都省掉了。省掉这些符号以后，文件更容易阅读，但代价是缩进必须更认真。

YAML 支持注释。注释从 `#` 开始，到这一行结束。

```yaml
# 站点基本信息
site:
  title: Hyacehila
  language: zh-CN  # 默认语言
```

注释是 YAML 很适合配置文件的重要原因之一。JSON 标准格式里不能写注释，所以很多项目会额外引入 JSONC 或者把说明写到文档里。YAML 则可以直接把解释放在配置旁边。

YAML 里的字符串通常不用加引号。

```yaml
title: YAML 格式与使用速成
path: /blog/2026/07/06/yaml-format-usage-learning-notes/
```

但是当字符串里包含容易被误解的内容时，最好主动加引号。比如冒号、井号、特殊布尔值、前后空格、版本号、时间、日期等。

```yaml
title: "YAML: Syntax, Use Cases, and Practical Parsing"
version: "1.0"
answer: "no"
created_at: "2026-07-06 20:04:07"
```

YAML 会把一些值自动解析成数字、布尔值或空值。

```yaml
count: 10
ratio: 0.8
enabled: true
disabled: false
empty_value: null
also_empty:
```

这里 `count` 是整数，`ratio` 是浮点数，`enabled` 是布尔值，`empty_value` 和 `also_empty` 都可能被解析成空值。这个自动类型推断很方便，但也是 YAML 最常见的坑之一，后面会单独讲。

## 多行字符串

YAML 处理长文本时，比 JSON 舒服很多。最常见的两种写法是 `|` 和 `>`。

`|` 表示保留换行。文本怎么换行，解析以后基本就怎么保留。

```yaml
description: |
  这是第一行。
  这是第二行。
  这是第三行。
```

这适合保存脚本、证书、提示词模板、邮件正文、Markdown 片段等需要保留换行的内容。

```yaml
script: |
  npm install
  npm run build
  npm run check:i18n
```

`>` 表示折叠换行。多行文本会被折叠成更接近一段话的形式。

```yaml
summary: >
  YAML 很适合写配置文件，
  因为它比 JSON 更方便人类阅读和编辑，
  但它也更依赖缩进和解析规则。
```

如果只是想让配置文件里的一段长描述不要横向太长，`>` 会更自然。如果你真的需要保留每一行的边界，比如 shell 脚本、prompt 模板、Nginx 配置片段，就应该用 `|`。

这个区别在工程里很重要。很多时候配置文件不是只存简单参数，而是会存一段命令、一段模板、一段说明文字。用错了 `|` 和 `>`，程序看到的字符串就会和你以为的不一样。

## YAML 常见使用场景

第一个非常日常的场景是博客文章的 front matter。很多静态博客系统都会在 Markdown 文件开头放一段 YAML 元数据，告诉系统这篇文章的标题、日期、分类、标签、摘要。

```yaml
---
title: "YAML 格式与使用速成"
date: 2026-07-06 20:04:07 +0800
categories: ["Programming", "Data & Databases"]
tags: ["Learning Notes", "YAML"]
hidden: true
---
```

这里的正文仍然是 Markdown，但开头这段 `---` 包住的区域会先被当成 YAML 解析。主题、归档、标签页、首页卡片都会读取这些字段。

第二个常见场景是 GitHub Actions。CI/CD 配置特别适合 YAML，因为它有明显的层级结构：什么事件触发、跑在哪个系统上、有哪些 job、每个 job 有哪些 step。

```yaml
name: build

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
```

这类配置如果写成 JSON，会有大量括号和字符串引号；写成 YAML，就更像一个可读的执行清单。

第三个场景是 Docker Compose。它用 YAML 描述多个服务之间的关系。

```yaml
services:
  web:
    image: nginx:latest
    ports:
      - "8080:80"
    volumes:
      - ./site:/usr/share/nginx/html:ro

  redis:
    image: redis:7
    restart: unless-stopped
```

这里的 YAML 表达力刚好够用：服务名是 key，镜像、端口、卷、重启策略都是字段。人读起来很接近“这个系统由哪些组件组成”。

第四个场景是 Kubernetes。Kubernetes 的资源对象通常写成 YAML manifest。

```yaml
apiVersion: apps/v1
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
```

Kubernetes YAML 也说明了 YAML 的另一面：当配置太复杂时，它会变得很长，很容易复制粘贴出错。YAML 不是复杂系统的银弹，它只是让配置对象更容易被人直接编辑。

第五个场景是应用自己的配置文件。

```yaml
server:
  host: 0.0.0.0
  port: 8080

database:
  url: postgresql://localhost:5432/app
  pool_size: 10
  timeout_seconds: 30

features:
  enable_cache: true
  enable_experiment: false
```

这类配置最后通常会被程序读成一个字典或对象。程序不应该直接把 YAML 当字符串处理，而应该通过解析器读入，再用类型检查、schema 或默认值逻辑来约束它。

## 如何处理 YAML

Python 里最常见的 YAML 库是 PyYAML。读取 YAML 时，应该优先使用 `safe_load`，不要随手使用不安全的加载方式。

```python
import yaml

raw = """
server:
  host: 0.0.0.0
  port: 8080
features:
  enable_cache: true
"""

config = yaml.safe_load(raw)

print(config["server"]["host"])
print(config["server"]["port"])
print(config["features"]["enable_cache"])
```

`yaml.safe_load()` 会把 YAML 字符串解析成 Python 对象。一般对应关系如下：

* YAML mapping -> Python `dict`
* YAML sequence -> Python `list`
* YAML string -> Python `str`
* YAML integer -> Python `int`
* YAML float -> Python `float`
* YAML boolean -> Python `bool`
* YAML null -> Python `None`

如果是从文件读取，写法也很直接。

```python
import yaml

file_path = "config.yaml"

with open(file_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

print(config)
```

写 YAML 可以使用 `safe_dump`。如果有中文，通常要设置 `allow_unicode=True`，否则中文可能会被转义。

```python
import yaml

config = {
    "site": {
        "title": "YAML 格式与使用速成",
        "language": "zh-CN",
    },
    "features": {
        "search": True,
        "comments": False,
    },
}

with open("config.yaml", "w", encoding="utf-8") as f:
    yaml.safe_dump(
        config,
        f,
        allow_unicode=True,
        sort_keys=False,
    )
```

`sort_keys=False` 也很常用。配置文件往往是人读的，字段顺序有意义，不一定希望库自动按字母排序。

JavaScript 或 Node.js 里，可以使用 `js-yaml` 这类库。

```javascript
const yaml = require("js-yaml");

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
```

从文件读取时，可以配合 `fs.readFileSync`。

```javascript
const fs = require("fs");
const yaml = require("js-yaml");

const raw = fs.readFileSync("config.yaml", "utf8");
const config = yaml.load(raw);

console.log(config);
```

写出 YAML 则使用 `dump`。

```javascript
const fs = require("fs");
const yaml = require("js-yaml");

const config = {
  site: {
    title: "YAML 格式与使用速成",
    language: "zh-CN",
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

fs.writeFileSync("config.yaml", output, "utf8");
```

写配置文件时，读写本身通常不是最难的。更难的是：解析以后怎么验证它是你想要的结构。真实项目里，最好给 YAML 配置加一层 schema 或显式校验。比如 `port` 必须是数字，`host` 必须是字符串，`features` 下面只能出现允许的开关。不要因为 YAML 文件看起来像配置，就默认它一定是合法配置。

## YAML 的坑

YAML 第一个坑是缩进。JSON 用括号表达层级，括号错了会很明显；YAML 用缩进表达层级，错一个空格有时候肉眼不容易发现。

```yaml
user:
  name: Jake
  profile:
    email: jake@example.com
    city: Wuhan
```

如果不小心写成这样：

```yaml
user:
  name: Jake
  profile:
    email: jake@example.com
  city: Wuhan
```

`city` 就不再属于 `profile`，而是和 `profile` 同级。它仍然可能是合法 YAML，但语义已经变了。这种错误最麻烦，因为解析器不会一定报错，它只会忠实地解析成另一个结构。

第二个坑是 tab。YAML 缩进不要用 tab，统一用空格。团队里最好配置编辑器，把 tab 自动转换为空格，并且显示不可见字符。看起来很小的习惯，能少掉很多奇怪错误。

第三个坑是隐式类型。YAML 会自动猜类型，这在简单配置里方便，在边界场景里危险。比如下面这些值，最好加引号：

```yaml
version: "1.0"
answer: "no"
switch: "on"
date: "2026-07-06"
time: "20:04:07"
hex_like: "0x10"
```

不同 YAML 版本、不同解析器、不同库选项，对某些值的解释可能不完全一样。为了减少不确定性，凡是你希望它保持字符串的值，都可以直接加引号。尤其是 ID、版本号、日期、枚举、命令参数，不要把解释权交给解析器猜。

第四个坑是冒号和井号。YAML 里 `:` 和 `#` 有语法含义。如果它们出现在字符串中，最好加引号。

```yaml
title: "YAML: Syntax, Use Cases, and Practical Parsing"
command: "echo hello # this is not a yaml comment"
url: "https://example.com/a:b"
```

第五个坑是重复 key。下面这个配置看起来只是写了两次 `port`：

```yaml
server:
  port: 8080
  port: 9000
```

但是很多解析器会直接保留后一个值，前一个值被覆盖。更麻烦的是，这可能不会报错。对于配置文件来说，重复 key 往往意味着复制粘贴或者合并配置时出了问题。最好使用 linter 或 schema 工具提前拦住。

第六个坑是锚点和别名。YAML 支持用 `&` 定义锚点，用 `*` 引用它，还可以用 `<<` 合并字段。

```yaml
defaults: &defaults
  image: node:22
  restart: unless-stopped
  environment:
    NODE_ENV: production

services:
  api:
    <<: *defaults
    command: npm run start:api
  worker:
    <<: *defaults
    command: npm run start:worker
```

这个能力很有用，可以减少重复。但它也会让文件从“配置”慢慢变成“带展开逻辑的配置”。如果一个 YAML 需要读者在脑子里不断跳转、合并、覆盖，维护成本就会上升。锚点适合少量复用，不适合把配置写成谜题。

第七个坑是多文档。YAML 一个文件里可以用 `---` 分隔多个文档。

```yaml
---
kind: ConfigMap
metadata:
  name: app-config
---
kind: Secret
metadata:
  name: app-secret
```

这在 Kubernetes 里很常见，但解析时要注意：普通的 `safe_load` 可能只适合单个文档。如果文件里有多个文档，在 Python 里通常要用 `safe_load_all`。

```python
import yaml

with open("resources.yaml", "r", encoding="utf-8") as f:
    documents = list(yaml.safe_load_all(f))

for doc in documents:
    print(doc["kind"])
```

第八个坑是安全。不要对不可信来源的 YAML 使用不安全的解析方式。YAML 的一些高级能力和对象构造能力，可能会让解析过程不只是“读数据”。对普通配置文件来说，`safe_load` 这样的安全解析方式已经足够。越是来自用户上传、外部服务、网络输入的 YAML，越应该保守处理。

## 什么时候不用 YAML

YAML 很适合人写配置，但不代表所有结构化数据都应该用 YAML。

如果是服务之间传输数据，优先考虑 JSON。JSON 更严格，解析器行为更统一，生态更稳定，也更适合 API 请求和响应。HTTP API 返回 YAML 并不是不可以，但绝大多数场景下，JSON 会让调用方轻松很多。

如果是大规模日志、训练数据、爬虫结果、模型输入输出记录，优先考虑 JSONL。JSONL 每一行都是一个完整 JSON 对象，天然适合追加、流式读取和分块处理。YAML 可以表达对象数组，但如果数据量很大，一整个 YAML 文件会越来越不适合流式处理。

如果是非常复杂、需要强校验的配置，也要小心 YAML。很多工程事故不是因为 YAML 不能表达，而是因为它太能表达，最后项目把一门配置格式用成了半门 DSL。配置里开始出现大量锚点、模板、条件、继承、覆盖规则之后，人就很难判断最终生效的配置到底是什么。

这时候有几种更稳的做法：用 JSON Schema、OpenAPI、Pydantic、Zod 等工具给配置加约束；把复杂逻辑移到普通代码里；或者为配置提供生成器和校验器，而不是让人手写所有细节。

所以 YAML 的边界可以这样理解：**它适合表达静态、层级清楚、主要由人维护的配置；不适合承载高频传输、大规模流式数据和过于复杂的业务逻辑。**

## 总结

YAML 的优势很直接：少括号、少引号、能写注释、层级清楚、适合配置。它让人能够在一个文本文件里自然地表达对象、列表、字符串、数字、布尔值和空值，也能舒服地写多行文本。

但 YAML 的危险也来自这些优势。它太依赖缩进，太相信人的阅读直觉，也会进行自动类型推断。很多 YAML 文件看起来很好懂，真正解析出来却未必是你脑子里那棵树。写 YAML 时要养成几个习惯：统一两个空格缩进，不用 tab；字符串不确定就加引号；复杂配置加 schema；用 linter 检查重复 key；读取外部 YAML 时使用安全解析。

和 JSON 相比，YAML 更像是给人看的配置笔记；和 JSONL 相比，YAML 更不适合流式数据；和代码相比，YAML 不应该承担太多逻辑。把它放在合适的位置，它会非常顺手。把它当成无所不能的配置语言，它就会慢慢变成维护负担。

我觉得最朴素的判断方式是：如果这个文件主要由人来写、由程序来读，并且结构层级比纯文本复杂，又没有复杂到需要写代码，那么 YAML 是一个很好的选择。如果这个文件主要由机器生成、机器消费、需要高频交换、需要强 schema 或者需要流式处理，那就应该认真考虑 JSON、JSONL 或者更明确的配置系统。
