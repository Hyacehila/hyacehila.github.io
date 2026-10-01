---
title: "SQL 基础：查询、聚合、JOIN 与窗口函数"
title_en: "SQL Basics: Queries, Aggregation, JOINs, and Window Functions"
date: 2024-07-29 22:29:30 +0800
categories: ["Programming", "CS Foundations"]
tags: ["SQL"]
author: Hyacehila
excerpt: "围绕同一套客户、商品、订单与明细数据，学习建表和增删改查，再理解 NULL、聚合、JOIN、子查询、集合运算和窗口函数，并保留视图、索引及常用函数的查阅入口。"
excerpt_en: "Use one customer, product, order, and order-item dataset to learn table definitions and CRUD, then NULL, aggregation, JOINs, subqueries, set operations, and window functions, with reference sections for views, indexes, and common functions."
description: "以 MySQL 8.4 和 InnoDB 为示例基准，从可复用的订单数据学习 SQL，并区分查询语义、数据库特有行为和基本事务操作。"
mathjax: false
hidden: true
permalink: '/blog/2024/07/29/sql-learning-notes/'
---

表设计好以后，下一步是把数据放进去，再把需要的结果取出来。我们已经在[《数据库基础：关系模型、表设计与规范化》](/blog/2024/12/23/database-systems-concepts/)里分出了客户、商品、订单和明细。现在沿着同一套数据看看：怎样查库存，怎样算订单金额，怎样找到从未下单的客户？

这篇既可以顺着学习，也可以按目录查语法。示例基准是 **MySQL 8.4、InnoDB**，通用 SQL 的思想与 MySQL 的具体规则会分别说明。先在一个新的练习库运行建表和种子数据；查询示例默认读取这份初始数据，后面的增删改与结构变更则使用独立练习表。代码块里的反例会明确标注，不需要把它们和正常示例一起执行。

## 先准备一套能反复查询的数据

### 建表：让规则真正进入数据库

下面是完整的初始化 SQL。主键、外键和数量约束，都是上一篇中业务规则的具体声明。

```sql
CREATE DATABASE database_foundations CHARACTER SET utf8mb4;
USE database_foundations;

CREATE TABLE customers (
    customer_id INT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    city VARCHAR(100) NOT NULL,
    email VARCHAR(255) UNIQUE
) ENGINE=InnoDB;

CREATE TABLE products (
    product_id INT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    category VARCHAR(50) NOT NULL,
    price DECIMAL(10, 2) NOT NULL DEFAULT 0,
    stock INT NOT NULL DEFAULT 0,
    CHECK (price >= 0),
    CHECK (stock >= 0)
) ENGINE=InnoDB;

CREATE TABLE orders (
    order_id INT PRIMARY KEY,
    customer_id INT NOT NULL,
    ordered_at DATETIME NOT NULL,
    status VARCHAR(16) NOT NULL,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
    CHECK (status IN ('paid', 'pending', 'cancelled'))
) ENGINE=InnoDB;

CREATE TABLE order_items (
    order_id INT NOT NULL,
    product_id INT NOT NULL,
    quantity INT NOT NULL,
    unit_price DECIMAL(10, 2) NOT NULL,
    PRIMARY KEY (order_id, product_id),
    FOREIGN KEY (order_id) REFERENCES orders(order_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id),
    CHECK (quantity > 0),
    CHECK (unit_price >= 0)
) ENGINE=InnoDB;
```

`INT` 用来保存整数；`DECIMAL(10, 2)` 是最多十位、其中两位小数的精确数值，适合这里的金额；`DATETIME` 保存日期与时间，`DATE` 则只保存日期。`VARCHAR` 保存可变长字符串；`CHAR` 是定长字符串，定长用的不多。

`NOT NULL` 禁止空值，`UNIQUE` 限制重复，主键同时保证唯一与非空。`CHECK` 在 MySQL 8.4 中会检查条件，不过条件得到 UNKNOWN 时也能通过，因此非空需求仍要用 `NOT NULL` 表达，`DEFAULT` 提供默认值，不能代替非空或业务约束。

### 种子数据：后面的结果从这里来

```sql
INSERT INTO customers (customer_id, name, city, email) VALUES
(1, 'Alice', 'Shanghai', 'alice@example.com'),
(2, 'Bob', 'Beijing', NULL),
(3, 'Carol', 'Shanghai', 'carol@example.com'),
(4, 'Dana', 'Shenzhen', NULL);

INSERT INTO products (product_id, name, category, price, stock) VALUES
(101, 'Keyboard', 'accessories', 200.00, 10),
(102, 'Mouse', 'accessories', 100.00, 20),
(103, 'Database Book', 'books', 100.00, 30),
(104, 'Monitor', 'hardware', 1000.00, 5);

INSERT INTO orders (order_id, customer_id, ordered_at, status) VALUES
(1001, 1, '2024-07-01 10:00:00', 'paid'),
(1002, 1, '2024-07-02 11:00:00', 'pending'),
(1003, 2, '2024-07-03 12:00:00', 'paid'),
(1004, 3, '2024-07-04 13:00:00', 'cancelled');

INSERT INTO order_items (order_id, product_id, quantity, unit_price) VALUES
(1001, 101, 1, 180.00),
(1001, 102, 2, 90.00),
(1002, 103, 1, 100.00),
(1003, 102, 1, 100.00),
(1003, 103, 3, 90.00),
(1004, 101, 1, 200.00);
```

Alice 有两张订单，Dana 没有订单，Monitor 从未出现在明细中。商品 102 与 103 的当前价格相同，客户中也有缺失的邮箱。这些情况是特意留下的，后面会用它们检查重复值、并列排名、空值和未匹配记录。

### 书写约定与语句种类

本文把关键字写成大写，表名与列名写成小写，用分号结束语句。这是便于阅读的约定。

`CREATE`、`ALTER`、`DROP` 通常归为数据定义语言 DDL；`INSERT`、`UPDATE`、`DELETE` 是数据操作语言 DML，`SELECT` 有时归入 DML，有时单独称 DQL；`GRANT`、`REVOKE` 管理权限，常称 DCL；提交和回滚属于事务控制。

注释可写成 `-- 后面有空格的单行注释`，或者 `/* 多行注释 */`。MySQL 还支持 `#` 单行注释，但它不是所有数据库都通用的形式。

## 写入和修改：先在练习表里观察变化

### INSERT、UPDATE、DELETE

先复制商品表结构与数据。`CREATE TABLE ... LIKE` 在 MySQL 中复制结构，`INSERT ... SELECT` 才把已有行复制进去。

```sql
CREATE TABLE products_demo LIKE products;
INSERT INTO products_demo SELECT * FROM products;

INSERT INTO products_demo (product_id, name, category, price, stock)
VALUES (105, 'Cable', 'accessories', 20.00, 8),
       (106, 'Stand', 'accessories', 50.00, 6);

UPDATE products_demo
SET price = price + 10
WHERE product_id = 105;

DELETE FROM products_demo
WHERE product_id = 106;

SELECT product_id, name, price
FROM products_demo
WHERE product_id >= 105
ORDER BY product_id;
```

最后只剩 Cable，价格是 30.00。一次 `INSERT` 可以插入多行，不需要为了每行单独循环。显式写列名也能避免表结构变化后，值的位置悄悄对错。

`UPDATE` 和 `DELETE` 中的 `WHERE` 决定影响哪些行，省略就会处理整个表的行。`DROP TABLE` 删除表本身，`DELETE` 删除表中的行；MySQL 的 `TRUNCATE TABLE` 也是另一种清空表的操作，不能把它当成可以按行筛选、随意回滚的 `DELETE`。

### 默认值不是把 NULL 自动变成零

用一张允许空价格的练习表，把两种情况分开看：

```sql
CREATE TABLE defaults_demo (
    id INT PRIMARY KEY,
    price DECIMAL(10, 2) DEFAULT 0
) ENGINE=InnoDB;

INSERT INTO defaults_demo (id) VALUES (1);
INSERT INTO defaults_demo (id, price) VALUES (2, DEFAULT), (3, NULL);

SELECT id, price FROM defaults_demo ORDER BY id;
```

结果依次是 `0.00`、`0.00`、`NULL`。省略价格或者使用 `DEFAULT`，才取这里的默认值；显式传入 `NULL` 会保存空值。核心商品表的价格还声明了 `NOT NULL`，传入空值在严格模式下会报错。

### ALTER：改的是结构，不是某一行

```sql
ALTER TABLE products_demo ADD COLUMN note VARCHAR(100);
ALTER TABLE products_demo MODIFY COLUMN note VARCHAR(200);
ALTER TABLE products_demo DROP COLUMN note;
```

增加列、修改定义、删除列都是结构变更。只想修改某件商品的价格，应该用前面的 `UPDATE`。

## 查询：先说清楚要哪些行、哪些列

### SELECT、WHERE 与表达式

```sql
SELECT product_id, name, price, stock,
       price * stock AS inventory_value
FROM products
WHERE price >= 100 AND stock >= 10
ORDER BY product_id;
```

返回 101、102、103 三种商品，库存金额分别为 2000.00、2000.00、3000.00。`SELECT` 定义返回的列，也可以计算表达式；`AS` 给表达式起别名；`FROM` 指定数据来源；`WHERE` 只保留条件为 TRUE 的行。

比较运算可使用 `=`、`<>`、`<`、`<=`、`>`、`>=`。`AND`、`OR`、`NOT` 组合条件，复杂表达式显式加括号。例如 `category = 'books' OR (category = 'accessories' AND stock >= 20)`，与先选两个类别再检查库存不是同一个意思。

### 重复、排序与分页

```sql
SELECT DISTINCT category FROM products ORDER BY category;

SELECT product_id, name, price
FROM products
ORDER BY price DESC, product_id ASC
LIMIT 2 OFFSET 1;
```

第一条返回三种类别，`DISTINCT` 对整组返回列去重，不是只对其中某一列单独去重，只有返回列完全一致的时候才会去重。第二条按价格从高到低排序，跳过第一行，取两行，得到 101 与 102，这和正式后端开发中的分页联系密切，我们通过这个让用户在数据库索引的时候能够逐页的查看。价格相同的时候再按 ID 排序，结果才有明确的先后。

`LIMIT ... OFFSET ...` 是这里采用的 MySQL 分页写法。没有 `ORDER BY`，不能假定结果沿着主键或者插入顺序返回；深分页的性能问题则留到查询优化时讨论。

### NULL 与三值逻辑

```sql
SELECT customer_id, name
FROM customers
WHERE email IS NULL
ORDER BY customer_id;

SELECT COUNT(*) AS all_customers,
       COUNT(email) AS known_emails
FROM customers;
```

第一条得到 Bob 与 Dana；第二条得到 4 与 2。`email = NULL` 不能代替 `IS NULL`，因为普通比较涉及 NULL 时通常得到 UNKNOWN。`WHERE` 不保留 UNKNOWN，也不保留 FALSE。

`SUM`、`AVG`、`MIN`、`MAX` 通常忽略空值；`COUNT(column)` 只数该列非空的行，`COUNT(*)` 数所有行，`COUNT(1)` 也是每行计算一个非空常量再计数。空输入下 `COUNT` 返回 0，而 `SUM` 等通常返回 NULL。`COUNT(DISTINCT email)` 则统计不同的非空邮箱。

空值也会影响排序，但不是“随机放在头尾”。MySQL 升序时把 NULL 放在非空值前面，降序时放在后面；分组时多个 NULL 会被视为同一组。

### LIKE、BETWEEN 与 IN

```sql
SELECT product_id, name FROM products WHERE name LIKE '%Book%';
SELECT product_id, price FROM products WHERE price BETWEEN 100 AND 200;
SELECT product_id, category FROM products WHERE category IN ('books', 'hardware');
```

`LIKE` 中 `%` 匹配零个或多个字符，`_` 匹配一个字符，是否区分大小写受排序规则影响。`BETWEEN` 包含两个端点，因此第二条包含 100 和 200。`IN` 检查是否匹配集合中的值；涉及 NULL 的反向判断需要额外小心，后面单独展示。

## 汇总：把一组行变成一条结果

### 聚合与 GROUP BY

```sql
SELECT category,
       COUNT(*) AS product_count,
       SUM(stock) AS total_stock,
       AVG(price) AS average_price
FROM products
GROUP BY category
ORDER BY category;
```

| category | product_count | total_stock | average_price |
| --- | --- | --- | --- |
| accessories | 2 | 30 | 150.00 |
| books | 1 | 30 | 100.00 |
| hardware | 1 | 5 | 1000.00 |

`GROUP BY` 定义哪些行属于同一组。分组后，输出里的普通列要能在组内确定一个值。MySQL 默认启用 `ONLY_FULL_GROUP_BY`：普通列需要出现在分组列中、能由分组列函数确定，或者满足它允许的特定单值条件。**一般情况下，分组聚合查询 GROUP BY 的 SELECT 里面可以涉及分组键本身以及聚合函数，其他都不太合法。**

### WHERE 与 HAVING 作用在不同位置

```sql
SELECT category, COUNT(*) AS product_count
FROM products
WHERE stock >= 10
GROUP BY category
HAVING COUNT(*) >= 2;

SELECT SUM(stock) AS total_stock
FROM products
HAVING SUM(stock) > 50;
```

第一条先筛选库存，再分组，只剩 accessories 这一组。`WHERE` 对输入行筛选，`HAVING` 对分组后的结果筛选。第二条没有显式 `GROUP BY`，整份输入作为一组，得到总库存 65；HAVING 并不是必须和 GROUP BY 一起出现。

### 用 CASE 把条件放进计算里

```sql
SELECT
    SUM(CASE WHEN status = 'paid' THEN 1 ELSE 0 END) AS paid_orders,
    SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) AS pending_orders,
    SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END) AS cancelled_orders
FROM orders;
```

结果是 2、1、1。`CASE WHEN ... THEN ... ELSE ... END` 是表达式，可以放在 SELECT、聚合参数、排序等需要值的位置。它不修改表，只根据条件产生一个值；省略 ELSE 时，未命中的情况返回 NULL。简单形式也可以写成 `CASE status WHEN 'paid' THEN ... END`。

### 逻辑处理顺序

理解一条普通分组查询时，可以先按下面的逻辑顺序推导结果：

```text
FROM / JOIN → WHERE → GROUP BY → HAVING
→ 窗口计算 → SELECT 输出与 DISTINCT → ORDER BY → LIMIT
```

这是一种解释结果的方式，不是要求数据库必须逐步生成这些中间表。优化器可以重排连接或下推条件，只要结果语义保持一致。

## 多张表：连接时要先看匹配关系

### INNER JOIN：取匹配成功的组合

```sql
SELECT o.order_id, c.name AS customer_name, o.status
FROM orders AS o
INNER JOIN customers AS c ON c.customer_id = o.customer_id
ORDER BY o.order_id;
```

返回四张订单；Alice 出现两次，因为她有两张订单。JOIN 不是单纯“给表加几列”，而是根据条件组合行，一对多关系会扩大某一侧的行数。这也解释了为什么连接明细后直接数订单，可能会重复计数。

把订单和明细接起来再汇总：

```sql
SELECT o.order_id, c.name AS customer_name,
       SUM(i.quantity * i.unit_price) AS total_amount
FROM orders AS o
JOIN customers AS c ON c.customer_id = o.customer_id
JOIN order_items AS i ON i.order_id = o.order_id
GROUP BY o.order_id, c.name
ORDER BY o.order_id;
```

金额依次是 360.00、100.00、370.00、200.00。用的是成交单价 `unit_price`，不是商品当前价格。若只统计已支付订单，可以在 GROUP BY 前加入 `WHERE o.status = 'paid'`，总金额就是 730.00。

### LEFT JOIN：未匹配的行也要留下

```sql
SELECT c.customer_id, c.name, COUNT(o.order_id) AS order_count
FROM customers AS c
LEFT JOIN orders AS o ON o.customer_id = c.customer_id
GROUP BY c.customer_id, c.name
ORDER BY c.customer_id;
```

四位客户的订单数是 2、1、1、0。Dana 的右侧列为 NULL，`COUNT(o.order_id)` 不计入它；换成 `COUNT(*)`，她那条保留下来的行就会被计为 1。

如果只想附上已支付订单、但仍保留所有客户，条件应该写在 ON 中：

```sql
SELECT c.customer_id, c.name, o.order_id
FROM customers AS c
LEFT JOIN orders AS o
    ON o.customer_id = c.customer_id AND o.status = 'paid'
ORDER BY c.customer_id, o.order_id;
```

把 `o.status = 'paid'` 移到 WHERE，会去掉没有匹配订单的客户，因为它们的条件结果不是 TRUE。RIGHT JOIN 保留右侧未匹配行，也可以通过交换左右表写成 LEFT JOIN。

### 自连接、非等值连接与其他形式

```sql
SELECT a.name AS cheaper_product, b.name AS more_expensive_product
FROM products AS a
JOIN products AS b ON a.price < b.price
WHERE a.product_id = 102
ORDER BY b.product_id;
```

Mouse 比 Keyboard 和 Monitor 便宜。这里对同一张表取两个别名，做的是自连接；ON 使用 `<`，所以也是非等值连接。自连接与内、外连接是不同维度的描述。

CROSS JOIN 得到所有行组合，两张各有四行的表会得到 16 行。NATURAL JOIN 自动拿同名列作匹配条件，USING 则显式列出同名连接列；随着表增加同名字段，自然连接的含义可能变化，所以本文优先明确写 ON。ON 也不是每种连接形式都必须具备的语法。

MySQL 8.4 没有直接的 FULL OUTER JOIN 语法。若要保留两侧未匹配行，可以组合左连接和反向的未匹配部分。下面是语法示意，A、B 是占位表名，假设各自 id 都非空：

```text
SELECT a.id AS a_id, b.id AS b_id
FROM A AS a LEFT JOIN B AS b ON a.id = b.id
UNION ALL
SELECT a.id, b.id
FROM B AS b LEFT JOIN A AS a ON a.id = b.id
WHERE a.id IS NULL;
```

第二部分只补未匹配的 B 行，避免把匹配行加两次。

## 子查询、存在性与集合运算

### 标量、派生表和关联子查询

```sql
SELECT product_id, name, price
FROM products
WHERE price > (SELECT AVG(price) FROM products);
```

平均价格是 350.00，因此只有 Monitor。括号里的查询作为一个值使用，叫标量子查询；多行结果会报错，没有行时得到 NULL。

把查询结果放在 FROM 中，就得到派生表，MySQL 要给它一个别名。还可以用 CTE 给一段查询命名：

```sql
WITH order_totals AS (
    SELECT order_id, SUM(quantity * unit_price) AS total_amount
    FROM order_items
    GROUP BY order_id
)
SELECT order_id, total_amount
FROM order_totals
WHERE total_amount > 300
ORDER BY order_id;
```

得到 1001 与 1003。CTE 只对紧随其后的一条语句有效，不会创建持久表。

关联子查询引用外层的列：

```sql
SELECT p.product_id, p.name, p.price
FROM products AS p
WHERE p.price > (
    SELECT AVG(q.price)
    FROM products AS q
    WHERE q.category = p.category
);
```

### EXISTS 与 NOT IN 的空值陷阱

```sql
SELECT p.product_id, p.name
FROM products AS p
WHERE NOT EXISTS (
    SELECT 1 FROM order_items AS i
    WHERE i.product_id = p.product_id
);

SELECT 3 NOT IN (1, 2, NULL) AS result;
```

第一条找到从未出现在订单中的 Monitor。EXISTS 只检查子查询有没有行，返回的列写 `1` 或 `*` 都不改变这个含义；拼写是 EXISTS，不是 EXIST。

第二条结果为 NULL，即 UNKNOWN。虽然 3 不等于 1 或 2，但与 NULL 的比较无法确定，因此不能把 NOT IN 当成“肯定没有出现”。当列表或子查询可能含 NULL 时，NOT IN 与 NOT EXISTS 不一定等价；外层值自身为 NULL 时，也要根据业务明确处理方式。

### UNION、INTERSECT、EXCEPT

```sql
SELECT product_id FROM products WHERE price <= 200
UNION
SELECT product_id FROM products WHERE category = 'accessories'
ORDER BY product_id;

SELECT product_id FROM products WHERE price <= 200
INTERSECT
SELECT product_id FROM products WHERE category = 'accessories'
ORDER BY product_id;

SELECT product_id FROM products WHERE price <= 200
EXCEPT
SELECT product_id FROM products WHERE category = 'accessories'
ORDER BY product_id;
```

分别得到 `{101,102,103}`、`{101,102}` 和 `{103}`。默认进行去重；UNION ALL 保留重复，上面第一条换成 UNION ALL 会得到五行。MySQL 8.4 还支持 INTERSECT ALL 和 EXCEPT ALL，其含义涉及重复次数，不能按普通集合的去重方式理解。

两边 SELECT 的列数要一致，各位置类型需要兼容，不是要求完全相同。MySQL 从 8.0.31 开始支持 INTERSECT 和 EXCEPT，较旧版本需要其他写法。INTERSECT 优先于 UNION 与 EXCEPT，复杂组合用括号明确含义，最终排序放在整个结果后面。

## 窗口函数：保留每行，再看它周围的数据

### OVER() 与分区

```sql
SELECT product_id, name, category, price,
       SUM(price) OVER () AS all_price_sum,
       AVG(price) OVER (PARTITION BY category) AS category_average
FROM products
ORDER BY product_id;
```

每种商品仍然有一行，总价 1400.00 出现在每行；配件两行的类别均价都是 150.00。GROUP BY 会合并行，窗口函数则对查询结果中的每行计算一个值。空的 `OVER()` 合法且有意义，表示整个查询结果作为窗口。

窗口函数在 MySQL 中可以出现在 SELECT 列表和 ORDER BY 中，不能直接放在 WHERE 或 HAVING。想按窗口结果筛选，需要再套一层派生表或 CTE。

### 排名与并列

```sql
SELECT product_id, price,
       RANK() OVER (ORDER BY price) AS price_rank,
       DENSE_RANK() OVER (ORDER BY price) AS dense_rank,
       ROW_NUMBER() OVER (ORDER BY price, product_id) AS row_num
FROM products
ORDER BY price, product_id;
```

| product_id | price | price_rank | dense_rank | row_num |
| --- | --- | --- | --- | --- |
| 102 | 100.00 | 1 | 1 | 1 |
| 103 | 100.00 | 1 | 1 | 2 |
| 101 | 200.00 | 3 | 2 | 3 |
| 104 | 1000.00 | 4 | 3 | 4 |

RANK 保留并列并跳过后续名次，DENSE_RANK 不跳号，ROW_NUMBER 为每行分配序号。这里故意不给 RANK 加商品 ID，否则两件同价商品也不再算并列；ROW_NUMBER 加 ID 则是为了给同价行确定顺序。窗口里的 ORDER BY 决定计算顺序，最终输出顺序仍由最外层 ORDER BY 指定。

### 累计金额和移动平均

```sql
WITH order_totals AS (
    SELECT order_id, SUM(quantity * unit_price) AS total_amount
    FROM order_items
    GROUP BY order_id
)
SELECT order_id, total_amount,
       SUM(total_amount) OVER (
           ORDER BY order_id ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
       ) AS running_total,
       AVG(total_amount) OVER (
           ORDER BY order_id ROWS BETWEEN 1 PRECEDING AND CURRENT ROW
       ) AS moving_average
FROM order_totals
ORDER BY order_id;
```

累计金额依次是 360、460、830、1030；两行移动平均依次是 360、230、235、285。第一行前面没有行，所以只用自身计算平均。

窗口聚合不总是累计计算：前面的 OVER() 就计算整个窗口。框架定义参与本行计算的范围，ROWS 按行位置，RANGE 按排序值及其同值行等规则解释。有 ORDER BY 而省略框架时，MySQL 对这类聚合的默认框架是 `RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`，会包括与当前行排序值相同的行；不能一概理解为“刚好到当前这一行”。希望逐行累计时，把排序和 ROWS 框架写清楚。

### 练习：找出至少连续三天登录的区间

这道题单独使用登录表。时间保留到秒，但统计时按日期去重：

```sql
CREATE TABLE login_records (
    user_id INT NOT NULL,
    login_at DATETIME NOT NULL
) ENGINE=InnoDB;

INSERT INTO login_records VALUES
(1, '2024-07-30 09:00:00'),
(1, '2024-07-30 18:00:00'),
(1, '2024-07-31 09:00:00'),
(1, '2024-08-01 09:00:00'),
(1, '2024-08-03 09:00:00'),
(2, '2024-07-30 10:00:00'),
(2, '2024-08-01 10:00:00'),
(3, '2024-07-29 10:00:00'),
(3, '2024-07-30 10:00:00'),
(3, '2024-07-31 10:00:00'),
(3, '2024-08-01 10:00:00');

WITH login_days AS (
    SELECT DISTINCT user_id, DATE(login_at) AS login_day
    FROM login_records
), ranked_days AS (
    SELECT user_id, login_day,
           ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY login_day) AS rn
    FROM login_days
), streak_groups AS (
    SELECT user_id, login_day,
           DATE_SUB(login_day, INTERVAL rn DAY) AS group_day
    FROM ranked_days
)
SELECT user_id, MIN(login_day) AS start_date,
       MAX(login_day) AS end_date, COUNT(*) AS consecutive_days
FROM streak_groups
GROUP BY user_id, group_day
HAVING COUNT(*) >= 3
ORDER BY user_id, start_date;
```

| user_id | start_date | end_date | consecutive_days |
| --- | --- | --- | --- |
| 1 | 2024-07-30 | 2024-08-01 | 3 |
| 3 | 2024-07-29 | 2024-08-01 | 4 |

同一天两次登录先合并，再编号。连续日期每前进一步，序号也增加一，因此“日期减序号”在一段连续区间中相同；缺一天就产生新的组。日期运算会处理跨月，而不是把日号当整数相减。

CTE 是给每一步起名字。也可以把 login_days、ranked_days、streak_groups 逐层改成 FROM 中的派生表，保留每层别名和同样的列，逻辑不会改变。这里保留一种完整解法，避免两份重复代码使用不同表名，却让读者误以为它们对应同一份数据。
