---
title: 'SQL Basics: Queries, Aggregation, JOINs, and Window Functions'
title_zh: SQL 基础：查询、聚合、JOIN 与窗口函数
date: 2024-07-29 22:29:30 +0800
categories:
- Programming
- CS Foundations
tags:
- SQL
author: Hyacehila
mathjax: false
hidden: true
excerpt: Use one customer, product, order, and order-item dataset to learn table definitions and CRUD, then NULL, aggregation,
  JOINs, subqueries, set operations, and window functions, with reference sections for views, indexes, and common functions.
description: Use one customer, product, order, and order-item dataset to learn table definitions and CRUD, then NULL, aggregation,
  JOINs, subqueries, set operations, and window functions, with reference sections for views, indexes, and common functions.
excerpt_zh: 围绕同一套客户、商品、订单与明细数据，学习建表和增删改查，再理解 NULL、聚合、JOIN、子查询、集合运算和窗口函数，并保留视图、索引及常用函数的查阅入口。
permalink: /blog/2024/07/29/sql-learning-notes/
lang: en
translation_key: 2024-07-29-sql-queries-joins-window-functions
translation_status: machine
translation_source_hash: 09a9557938a1b4c53bb63c100f946259d9364b41c41a16a41b3428dc6f66d02a
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

Once tables are designed, the next step is to insert data and retrieve the results we need. In [Database Foundations: Relational Models, Table Design, and Normalization](/en/blog/2024/12/23/database-systems-concepts/), we separated customers, products, orders, and items. Now we use the same data to ask: how do we check stock, calculate order totals, or find customers who have never placed an order?

You can read this post in sequence or use the contents to look up syntax. Examples target **MySQL 8.4 and InnoDB**. General SQL ideas and MySQL-specific rules are distinguished. Run the table definitions and seed data in a new practice database first. Query examples assume this initial data, while modifications and structural changes use separate practice tables. Counterexamples are explicitly identified and do not need to be executed alongside normal examples.

## Prepare One Reusable Dataset

### Table Definitions: Put the Rules into the Database

This is the complete initialization SQL. Primary keys, foreign keys, and quantity constraints implement the business rules from the companion post.

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

`INT` stores integers. `DECIMAL(10, 2)` is an exact numeric type with at most ten digits, two after the decimal point, suitable for these amounts. `DATETIME` stores a date and time, while `DATE` stores only a date. `VARCHAR` stores variable-length strings; `CHAR` stores fixed-length strings and is less often used here.

`NOT NULL` rejects null values, `UNIQUE` restricts duplicates, and a primary key guarantees both uniqueness and non-nullability. MySQL 8.4 enforces `CHECK` conditions, but a condition evaluating to UNKNOWN also passes, so non-null requirements still need `NOT NULL`. `DEFAULT` supplies a default value and cannot replace non-null or business constraints.

### Seed Data: Where Later Results Come From

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

Alice has two orders, Dana has none, and Monitor never appears in an item. Products 102 and 103 have equal current prices, and some customer emails are missing. These cases let us examine duplicates, tied ranks, nulls, and unmatched records.

### Writing Conventions and Statement Categories

Keywords are uppercase here, table and column names lowercase, and statements end with semicolons. This is a readability convention.

`CREATE`, `ALTER`, and `DROP` are usually classified as data definition language, DDL. `INSERT`, `UPDATE`, and `DELETE` are data manipulation language, DML. `SELECT` is sometimes included in DML and sometimes called DQL separately. `GRANT` and `REVOKE` manage permissions and are commonly called DCL; commit and rollback are transaction control.

Comments can use `-- followed by a space` for one line or `/* multiple lines */`. MySQL also supports `#` comments, but not every database does.

## Insert and Modify Data in Practice Tables

### INSERT, UPDATE, and DELETE

First copy the product structure and rows. In MySQL, `CREATE TABLE ... LIKE` copies the structure; `INSERT ... SELECT` copies the existing rows.

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

Only Cable remains, priced at 30.00. One `INSERT` can insert multiple rows without a separate loop for each. Explicit column names also prevent values silently being assigned to the wrong positions after schema changes.

In `UPDATE` and `DELETE`, `WHERE` determines which rows are affected. Omitting it processes all rows. `DROP TABLE` removes the table itself, while `DELETE` removes its rows. MySQL's `TRUNCATE TABLE` is another way to empty a table; it is not a row-filterable, freely rollbackable version of `DELETE`.

### A Default Does Not Automatically Turn NULL into Zero

A practice table with a nullable price separates the cases:

```sql
CREATE TABLE defaults_demo (
    id INT PRIMARY KEY,
    price DECIMAL(10, 2) DEFAULT 0
) ENGINE=InnoDB;

INSERT INTO defaults_demo (id) VALUES (1);
INSERT INTO defaults_demo (id, price) VALUES (2, DEFAULT), (3, NULL);

SELECT id, price FROM defaults_demo ORDER BY id;
```

The results are `0.00`, `0.00`, and `NULL`. Omitting price or writing `DEFAULT` uses the default. An explicit `NULL` stores a null. The core product table also declares price `NOT NULL`, so inserting a null fails in strict mode.

### ALTER Changes Structure, Not an Individual Row

```sql
ALTER TABLE products_demo ADD COLUMN note VARCHAR(100);
ALTER TABLE products_demo MODIFY COLUMN note VARCHAR(200);
ALTER TABLE products_demo DROP COLUMN note;
```

Adding a column, changing its definition, and dropping it are structural changes. Changing one product's price calls for `UPDATE`.

## Queries: Decide Which Rows and Columns You Want

### SELECT, WHERE, and Expressions

```sql
SELECT product_id, name, price, stock,
       price * stock AS inventory_value
FROM products
WHERE price >= 100 AND stock >= 10
ORDER BY product_id;
```

This returns products 101, 102, and 103, with inventory values of 2000.00, 2000.00, and 3000.00. `SELECT` specifies columns and computed expressions; `AS` names an expression; `FROM` supplies the input; `WHERE` keeps only rows whose condition is TRUE.

Comparisons use `=`, `<>`, `<`, `<=`, `>`, and `>=`. `AND`, `OR`, and `NOT` combine conditions. Use parentheses for complex expressions. For example, `category = 'books' OR (category = 'accessories' AND stock >= 20)` is not the same as selecting both categories and then applying the stock condition to both.

### Duplicates, Ordering, and Pagination

```sql
SELECT DISTINCT category FROM products ORDER BY category;

SELECT product_id, name, price
FROM products
ORDER BY price DESC, product_id ASC
LIMIT 2 OFFSET 1;
```

The first query returns three categories. `DISTINCT` removes duplicates across the complete selected column list, rather than independently deduplicating just one column: all returned column values must match for a row to be deduplicated. The second sorts prices descending, skips the first row, and takes two, returning 101 and 102. This is closely related to pagination in backend development, allowing users to browse database results page by page. Adding ID ordering makes ties deterministic.

`LIMIT ... OFFSET ...` is the MySQL pagination form used here. Without `ORDER BY`, do not assume primary-key or insertion order. The performance of deep pagination belongs to the later query optimization discussion.

### NULL and Three-Valued Logic

```sql
SELECT customer_id, name
FROM customers
WHERE email IS NULL
ORDER BY customer_id;

SELECT COUNT(*) AS all_customers,
       COUNT(email) AS known_emails
FROM customers;
```

The first query returns Bob and Dana; the second returns 4 and 2. `email = NULL` does not replace `IS NULL`, because ordinary comparisons involving null usually evaluate to UNKNOWN. `WHERE` discards both UNKNOWN and FALSE.

`SUM`, `AVG`, `MIN`, and `MAX` generally ignore nulls. `COUNT(column)` counts non-null values in that column; `COUNT(*)` counts rows; `COUNT(1)` counts a non-null constant for each row. On empty input, `COUNT` returns zero, while `SUM` and similar aggregates generally return NULL. `COUNT(DISTINCT email)` counts distinct non-null email values.

Nulls also affect sorting, but are not randomly placed at the beginning or end. MySQL places NULL before non-null values in ascending order and after them in descending order. Multiple nulls form one group when grouping.

### LIKE, BETWEEN, and IN

```sql
SELECT product_id, name FROM products WHERE name LIKE '%Book%';
SELECT product_id, price FROM products WHERE price BETWEEN 100 AND 200;
SELECT product_id, category FROM products WHERE category IN ('books', 'hardware');
```

For `LIKE`, `%` matches zero or more characters and `_` matches one; case sensitivity depends on collation. `BETWEEN` includes both endpoints, so the second query includes prices of 100 and 200. `IN` tests membership among values. Its negative form needs extra care around nulls, as shown later.

## Aggregation: Turn a Group of Rows into One Result

### Aggregates and GROUP BY

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

`GROUP BY` determines group membership. A nonaggregated output column must have a determinate value within the group. MySQL enables `ONLY_FULL_GROUP_BY` by default: nonaggregated columns must be grouped, functionally determined by grouped columns, or satisfy certain permitted single-value conditions. **As a general rule, the SELECT list of a grouped aggregate query contains grouping keys and aggregate expressions; other expressions are usually invalid.**

### WHERE and HAVING Filter Different Stages

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

The first filters stock before grouping and retains only accessories. `WHERE` filters input rows; `HAVING` filters grouped results. The second has no explicit `GROUP BY`: all input is one group, producing total stock 65. HAVING does not require an accompanying GROUP BY.

### Put Conditions into Calculations with CASE

```sql
SELECT
    SUM(CASE WHEN status = 'paid' THEN 1 ELSE 0 END) AS paid_orders,
    SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) AS pending_orders,
    SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END) AS cancelled_orders
FROM orders;
```

The result is 2, 1, and 1. `CASE WHEN ... THEN ... ELSE ... END` is an expression usable in SELECT, aggregate arguments, ordering, and other value positions. It does not modify a table; it produces a value from conditions. Without ELSE, unmatched cases yield NULL. The simple form can be written `CASE status WHEN 'paid' THEN ... END`.

### Logical Processing Order

For an ordinary grouped query, this logical sequence helps derive the result:

```text
FROM / JOIN → WHERE → GROUP BY → HAVING
→ window calculations → SELECT output and DISTINCT → ORDER BY → LIMIT
```

It explains results rather than requiring the database to materialize each intermediate table in that order. The optimizer can reorder joins or push predicates down while preserving semantics.



## Multiple Tables: Understand Matching Before Joining

### INNER JOIN Keeps Matching Combinations

```sql
SELECT o.order_id, c.name AS customer_name, o.status
FROM orders AS o
INNER JOIN customers AS c ON c.customer_id = o.customer_id
ORDER BY o.order_id;
```

Four orders are returned. Alice appears twice because she has two orders. A JOIN is not merely adding columns: it combines rows according to conditions, and one-to-many relationships multiply rows on one side. Counting orders directly after joining items can therefore count an order more than once.

Join orders to items, then aggregate:

```sql
SELECT o.order_id, c.name AS customer_name,
       SUM(i.quantity * i.unit_price) AS total_amount
FROM orders AS o
JOIN customers AS c ON c.customer_id = o.customer_id
JOIN order_items AS i ON i.order_id = o.order_id
GROUP BY o.order_id, c.name
ORDER BY o.order_id;
```

Totals are 360.00, 100.00, 370.00, and 200.00. They use sale unit prices, not current product prices. To keep only paid orders, add `WHERE o.status = 'paid'` before GROUP BY; their combined total is 730.00.

### LEFT JOIN Also Keeps Unmatched Rows

```sql
SELECT c.customer_id, c.name, COUNT(o.order_id) AS order_count
FROM customers AS c
LEFT JOIN orders AS o ON o.customer_id = c.customer_id
GROUP BY c.customer_id, c.name
ORDER BY c.customer_id;
```

The four customers have 2, 1, 1, and 0 orders. Dana's right-side columns are NULL, so `COUNT(o.order_id)` does not count them. `COUNT(*)` would count her preserved row as one.

To attach only paid orders while retaining every customer, put the condition in ON:

```sql
SELECT c.customer_id, c.name, o.order_id
FROM customers AS c
LEFT JOIN orders AS o
    ON o.customer_id = c.customer_id AND o.status = 'paid'
ORDER BY c.customer_id, o.order_id;
```

Moving `o.status = 'paid'` to WHERE removes customers with no matching order because their condition does not evaluate to TRUE. RIGHT JOIN retains unmatched rows on the right and can also be written as LEFT JOIN by exchanging the tables.

### Self-Joins, Nonequality Joins, and Other Forms

```sql
SELECT a.name AS cheaper_product, b.name AS more_expensive_product
FROM products AS a
JOIN products AS b ON a.price < b.price
WHERE a.product_id = 102
ORDER BY b.product_id;
```

Mouse is cheaper than Keyboard and Monitor. Using two aliases of the same table makes this a self-join; comparing with `<` makes it a nonequality join. Self-join and inner/outer join describe different dimensions.

CROSS JOIN produces all row combinations: two four-row tables produce 16 rows. NATURAL JOIN automatically matches same-named columns; USING explicitly lists shared column names. Adding same-named fields can change a natural join's meaning, so examples here prefer explicit ON conditions. ON is not mandatory in every join form.

MySQL 8.4 has no direct FULL OUTER JOIN syntax. To keep unmatched rows from both sides, combine a left join with the unmatched part of the reverse join. This is a syntax illustration: A and B are placeholder table names, and both id columns are assumed non-null:

```text
SELECT a.id AS a_id, b.id AS b_id
FROM A AS a LEFT JOIN B AS b ON a.id = b.id
UNION ALL
SELECT a.id, b.id
FROM B AS b LEFT JOIN A AS a ON a.id = b.id
WHERE a.id IS NULL;
```

The second part adds only unmatched B rows, avoiding duplicate addition of matches.

## Subqueries, Existence Tests, and Set Operations

### Scalar Subqueries, Derived Tables, and Correlated Subqueries

```sql
SELECT product_id, name, price
FROM products
WHERE price > (SELECT AVG(price) FROM products);
```

The average price is 350.00, so only Monitor qualifies. The parenthesized query supplies a single value and is a scalar subquery. More than one row causes an error; no rows yields NULL.

A query placed in FROM is a derived table, and MySQL requires an alias. A CTE can also name a query:

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

Orders 1001 and 1003 qualify. A CTE belongs only to the immediately following statement and does not create a persistent table.

A correlated subquery references an outer column:

```sql
SELECT p.product_id, p.name, p.price
FROM products AS p
WHERE p.price > (
    SELECT AVG(q.price)
    FROM products AS q
    WHERE q.category = p.category
);
```



### EXISTS and the NOT IN Null Trap

```sql
SELECT p.product_id, p.name
FROM products AS p
WHERE NOT EXISTS (
    SELECT 1 FROM order_items AS i
    WHERE i.product_id = p.product_id
);

SELECT 3 NOT IN (1, 2, NULL) AS result;
```

The first finds Monitor, which never occurs in an order item. EXISTS asks only whether the subquery returns a row; selecting `1` or `*` does not change that meaning. Its spelling is EXISTS, not EXIST.

The second produces NULL, or UNKNOWN. Although 3 is different from 1 and 2, comparison with NULL is indeterminate. Thus NOT IN does not always establish definite absence. NOT IN and NOT EXISTS need not be equivalent when the list or subquery can contain NULL. A null outer value also requires explicit business handling.

### UNION, INTERSECT, and EXCEPT

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

The results are `{101,102,103}`, `{101,102}`, and `{103}`. These operations deduplicate by default. UNION ALL retains duplicates; replacing the first UNION with UNION ALL gives five rows. MySQL 8.4 also supports INTERSECT ALL and EXCEPT ALL, whose meanings account for duplicate multiplicities rather than ordinary deduplicated sets.

Both SELECT lists must have the same number of columns, with compatible types in corresponding positions, rather than necessarily identical types. MySQL added INTERSECT and EXCEPT in 8.0.31; older versions need alternatives. INTERSECT takes precedence over UNION and EXCEPT. Use parentheses for complex combinations and place final ordering after the complete result.



## Window Functions: Keep Each Row and Examine Related Rows

### OVER() and Partitions

```sql
SELECT product_id, name, category, price,
       SUM(price) OVER () AS all_price_sum,
       AVG(price) OVER (PARTITION BY category) AS category_average
FROM products
ORDER BY product_id;
```

Every product still has a row. Total price 1400.00 appears on each, and both accessory rows show a category average of 150.00. GROUP BY collapses rows; a window function computes a value for each query row. Empty `OVER()` is valid and meaningful: it uses the entire query result as a window.

MySQL permits window functions in the SELECT list and ORDER BY, not directly in WHERE or HAVING. To filter a window result, wrap it in a derived table or CTE.

### Ranking and Ties

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

RANK keeps ties and leaves gaps afterward; DENSE_RANK leaves no gaps; ROW_NUMBER assigns a sequence to individual rows. Product ID is deliberately omitted from RANK ordering, since including it would break the price tie. It is added to ROW_NUMBER for deterministic ordering of equal prices. Window ORDER BY controls calculations; final output order still requires the outer ORDER BY.

### Running Totals and Moving Averages

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

Running totals are 360, 460, 830, and 1030. Two-row moving averages are 360, 230, 235, and 285. The first row has no preceding row, so its average uses itself alone.

Window aggregation is not always cumulative: the earlier OVER() covers the complete window. The frame specifies which rows participate in each calculation. ROWS uses row positions, while RANGE uses ordering values and peers according to its rules. For these aggregates, an ORDER BY with no explicit frame gives MySQL's default `RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`, including rows with the current row's ordering value. This is not necessarily exactly through the current physical row. Specify ordering and a ROWS frame for row-by-row accumulation.

### Exercise: Find Login Streaks Lasting at Least Three Days

This exercise uses a separate login table. Timestamps retain seconds, but the analysis deduplicates dates:

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

Two logins on one day are merged before numbering. Within a streak, both the date and row number increase by one each day, so subtracting the row number from the date yields a constant group key. A missing day starts a different group. Date arithmetic handles month boundaries rather than subtracting integer day-of-month values.

The CTEs name each step. They can also be rewritten as nested FROM derived tables, retaining each alias and the same columns without changing the logic. One complete solution is kept here to avoid duplicated code with different table names suggesting it uses the same dataset.
