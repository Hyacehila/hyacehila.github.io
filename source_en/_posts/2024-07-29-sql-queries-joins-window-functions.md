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
excerpt: Covers database concepts, SQL statements, table creation and updates, querying, aggregation, views, subqueries, functions,
  predicates, CASE, set operations, JOINs, and window functions.
description: Covers database concepts, SQL statements, table creation and updates, querying, aggregation, views, subqueries,
  functions, predicates, CASE, set operations, JOINs, and window functions.
excerpt_zh: 覆盖数据库与 SQL 语句基础、创建和更新、查询排序、聚合分组、视图、子查询、函数、谓词、CASE、集合运算、JOIN 和窗口函数。
permalink: /blog/2024/07/29/sql-learning-notes/
lang: en
translation_key: 2024-07-29-sql-queries-joins-window-functions
translation_status: machine
translation_source_hash: 9840af0aeec3f010a391f39fc1a8d765801ecf28d326d4a3930d977db15bd178
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

<p>SQL is widely used in a wide range of areas, including data analysis, development, testing, maintenance, product manager, etc. Course content is used, considering accessibility and accessibility<code>MySql</code> The database is presented.</p>
<p>The questions in this article can also be addressed<a href="/en/blog/2024/12/18/search-and-sorting-algorithms/">Find and Sort Algorithms: Retrieval, Dispersed Lists and Sort</a>、<a href="/en/blog/2024/12/23/database-systems-concepts/">Database system concept: data models, services and query processing</a>How the concept of a relatively close read together is developed in different contexts.</p>
<p>The environment in which the relevant databases are built is not recorded here, and we begin with an understanding of the databases by introducing basic queries and sequencing, and then by studying more complex calculations and advanced processing. Final study of some of the topics.</p>
<h2>Initial knowledge database</h2>
<h3>Introduction to the database</h3>
<p>The database is a computerized collection of data that is efficiently accessible by preserving large amounts of data. The data set is referred to as the database (Database, DB). The computer system used to manage the database is known as the database management system (Database Management System, DBMS).</p>
<p>DBMS is classified mainly through data preservation formats (type of database) and at this stage there are mainly the following five types.</p>
<ul>
<li>Level database (Hierarchical Database, HDB)</li>
<li>Relationship database (Relational Database, RDB)<ul>
<li>Oracle Database: Oracle RDBMS</li>
<li>SQL Server: Microsoft RDBMS</li>
<li>DB2: RDBMS</li>
<li>PostgreSQL: Open Source RDBMS</li>
<li>MySQL: Open Source RDBMS</li>
</ul>
</li>
<li>Object-oriented database (Object Oriented Database, OODB)</li>
<li>XML Database (XML Database, XMLDB)</li>
<li>Key-Value Store, KVS, for example: MongoDB</li>
</ul>
<p>The most common of these is the relationship database RDB, which is characterized by a two-dimensional table of rows and columns to manage data, known as the relationship database management system (Relational Data Management System, RDBMS). That's what we're going to be working on.</p>
<p>For RDBMS, the most based system structure is that where the service side is separated from the client, we use SQL to request data from the service side</p>
<h3>SQL Statement Basis</h3>
<p>Stored in database<strong>Table structure</strong>like rows and columns in excel, in the database, as<strong>Records</strong>It's like a record. It's called<strong>Fields</strong>, which represents the data items stored in the table. The place where rows and columns intersect is called a cell, and only one record can be entered in a cell.</p>
<p>SQL is the language developed to operate the database. The International Organization for Standardization (ISO) has developed the corresponding standard for SQL, which is referred to as the standard SQL. We're just introducing standards, SQLs, smart learning for those SQLs that have been modified by the company.</p>
<p>Depending on the type of command given to RDBMS, SQL statements can be divided into three categories.</p>
<ul>
<li><strong>DDL</strong> : DDL (Data Defense Language, Data Definition Language) is used to create or delete databases for storing data and tables in databases. DDL contains the following types of instructions.<ul>
<li>CREATE: Create objects such as databases and tables</li>
<li>DROP: Delete objects such as databases and tables</li>
<li>ALTER: Modify the structure of objects such as databases and tables</li>
</ul>
</li>
<li><strong>DML</strong> : DML (Data Manipulation Language, Data Manipulation Language) is used to query or change the records in the table. DML contains the following types of instructions.<ul>
<li>SELECT: Data from the query table</li>
<li>INSERT: Insert new data into the table</li>
<li>UPDATE: Data in updated tables</li>
<li>DELETE: Delete data from the table</li>
</ul>
</li>
<li><strong>DCL</strong> DCL (Data Control Language, Data Control Language) is used to confirm or cancel changes to data in databases. In addition to this, it is possible to set if the users of RDBMS have the permission to operate objects (database tables, etc.) in the database. DCL contains the following types of instructions.<ul>
<li>COMMIT: Confirm changes to data in the database</li>
<li>ROLLBACK: Cancel changes to data in the database</li>
<li>GRANT: Give user access</li>
<li>REVOKE: Undo user permissions</li>
</ul>
</li>
</ul>
<p><strong>90% of the SQL statements actually used are DML</strong></p>
<h3>Basic writing rules for SQL</h3>
<h4>Mandatory grammar rule</h4>
<ul>
<li>SQL statements should end with semicolons (;)</li>
<li>SQL does not distinguish between case of keywords, but the data inserted into the table is case-sensitive</li>
<li>The win system default does not distinguish the size of a table name from a field name Write</li>
<li>linux / Mac defaults strictly to distinguish the size of the table and field names Write</li>
<li>The writing of constants is fixed, as if&#39;abc&#39;, 1234, &#39;26 Jan 2010&#39;, &#39;10/01/26&#39;, &#39;2010-01-26&#39;......</li>
<li>Words need to be separated by half-angled spaces or line breaks</li>
</ul>
<p>The words of the SQL statement need to be separated by a half-angled space or a line break, and the full-angled space cannot be used as a separator for a word, otherwise there will be an error and an unexpected result.</p>
<h4>Write Recommendations</h4>
<p>** The general principle of the SQL syntax norm is that it is clear, readable and hierarchical. ** In the actual scene, thousands of SQL words are often used, and if not clearly written, life is questioned when review or someone else takes over.</p>
<p><strong>The common concerns are as follows:</strong></p>
<ol>
<li>MySQL does not distinguish case by case per se, but strongly requires capitalisation of keywords, listings, etc.;</li>
<li>When creating tables, use uniform, descriptive field naming rules to ensure that field names are unique and not reserved, do not use continuous underlined endings; preferably start with letters</li>
<li>Right alignment of keywords, with space or indentation control at different levels to separate them, as in Example 2;</li>
<li>In the case of low listings, write in a row without harm; in many cases, and in relation to CASE WhEN or aggregation calculations, recommend line writing; and in the case of individuals, commas are used to pre-listed, to remove certain columns at ease and to list them;</li>
<li>(a) The use of aliases and aliases with as much meaning as possible, rather than a b c, otherwise the review will be painful;</li>
<li>A space is added to the operator;</li>
<li>When multiple forms are used, please write the aliases quoted before all listings, so as not to be cumbersome;</li>
<li>Each order ends with a semicolon;</li>
<li>A habit of hand-written notes.</li>
</ol>
<pre><code class="language-plain">单行注释 #注释文字 （MySQL专属）
单行注释 -- 注释文字
多行注释：/* 注释文字 */
</code></pre>
<h3>Create grammar knowledge with it</h3>
<h4>Create CREATE Statement</h4>
<p>The language code for creating the database is</p>
<pre><code class="language-sql">CREATE DATABASE &lt; 数据库名称 &gt; ;
</code></pre>
<p>The language code for creating the form is</p>
<pre><code class="language-sql">CREATE TABLE &lt; 表名 &gt;
( &lt; 列名 1&gt; &lt; 数据类型 &gt; &lt; 该列所需约束 &gt; ,
  &lt; 列名 2&gt; &lt; 数据类型 &gt; &lt; 该列所需约束 &gt; ,
  &lt; 列名 3&gt; &lt; 数据类型 &gt; &lt; 该列所需约束 &gt; ,
  &lt; 列名 4&gt; &lt; 数据类型 &gt; &lt; 该列所需约束 &gt; ,
  .
  .
  .
  &lt; 该表的约束 1&gt; , &lt; 该表的约束 2&gt; ,……);
</code></pre>
<p>Of which <code>&lt; &gt;</code> Just to emphasize that this is a statement that you need to enter yourself.</p>
<h4>Naming rules</h4>
<ul>
<li>Use only<strong>Half-angle English letters, numbers, underlining</strong> As<strong>Databases, tables and columns</strong>Name</li>
<li>The name must start with a semi-English letter.</li>
<li>It's a habit. We shouldn't end with a underlined.</li>
</ul>
<h4>Data Type</h4>
<p>The table created by the database must specify the type of data, and each column cannot store data that does not match the type of data in the column.</p>
<p>Four minimum data types</p>
<ul>
<li>INTEGER type: Specifies the type of data (number type) used to store integer columns, which cannot store decimals.</li>
<li>CHAR type: When the length of the string stored in a column is less than the maximum length, it is supplemented by a half-angle space, which is not generally used because storage space is wasted.<strong>To Set Length</strong></li>
<li>VARCHAR type: A long string is used to store a variable length string, which is filled with a half-angle space when the number of characters does not reach the maximum length, but with a variable-long string, even if the number of characters does not reach the maximum length.<strong>To set maximum length</strong></li>
<li>DATE type: Data type (date type) for columns that specify the date of storage (month of year).</li>
</ul>
<p>In practical use, we use the VARCHAR type most often</p>
<h4>Binding Settings</h4>
<p>Constraint is the function of limiting or adding conditions to data stored in columns in addition to data types.</p>
<p><code>NOT NULL</code>is a non-empty constraint, i.e. the column must enter data.</p>
<p><code>PRIMARY KEY</code>is the primary key constraint, representing the only value of the column, from which you can extract data for a particular row. This sentence should be used as a binding form. <code>PRIMARY KEY (product_id)</code></p>
<p>The main key constraint is an important part of the follow-up interview.</p>
<h3>Delete and update</h3>
<h4>Changed database structure</h4>
<p>Delete table DROP TABLE statement</p>
<pre><code class="language-sql">DROP TABLE &lt; 表名 &gt; ;
</code></pre>
<p>It should be noted, in particular, that the deleted tables cannot be restored and can only be reinserted, with particular care being taken when executing the deletions.</p>
<p>Add column's ALTER TABLE statement</p>
<pre><code class="language-sql">ALTER TABLE &lt; 表名 &gt; ADD COLUMN &lt; 列的定义 &gt;;
</code></pre>
<p>The definition of the column would need to include both the type of data listed and the column and the constraints</p>
<p>Delete column ALTER TABLE statement</p>
<pre><code class="language-sql">ALTER TABLE &lt; 表名 &gt; DROP COLUMN &lt; 列名 &gt;;
</code></pre>
<h4>Changes to database contents</h4>
<p>Removes the specific row of the table</p>
<pre><code class="language-sql">-- 一定注意添加 WHERE 条件，否则将会删除所有的数据
DELETE FROM &lt; 表名 &gt; WHERE COLUMN_NAME=&#39;XXX&#39;;
</code></pre>
<p><strong>The WHERE statement is used to select a line.</strong></p>
<p>Update of data</p>
<pre><code class="language-sql">UPDATE &lt;表名&gt;
   SET &lt;列名&gt; = &lt;表达式&gt; [, &lt;列名2&gt;=&lt;表达式2&gt;...]  
 WHERE &lt;条件&gt;  -- 可选，非常重要
 ORDER BY 子句  --可选
 LIMIT 子句; --可选
</code></pre>
<p><strong>where condition select which line to operate, otherwise all lines will be modified by statement</strong></p>
<p><strong>SET in the UPDATE statement is the core statement that determines the update.</strong></p>
<p>Multi-column update: The UPDATE statement's SET subword supports multiple columns as an update object. Here are some examples.</p>
<pre><code class="language-sql">-- 基础写法，一条UPDATE语句只更新一列
UPDATE product
   SET sale_price = sale_price * 10
 WHERE product_type = &#39;厨房用具&#39;;
UPDATE product
   SET purchase_price = purchase_price / 2
 WHERE product_type = &#39;厨房用具&#39;;  
</code></pre>
<p>It can get the right results, but the code is cumbersome. A consolidation approach could be used to simplify the code.</p>
<pre><code class="language-sql">-- 合并后的写法
UPDATE product
   SET sale_price = sale_price * 10,
       purchase_price = purchase_price / 2
 WHERE product_type = &#39;厨房用具&#39;;  
</code></pre>
<p>It needs to be made clear that the columns in the SET clause can be not only two columns, but also three columns or more. One row to update one column</p>
<h3>Insert</h3>
<h4>Database background for this section</h4>
<p>To learn.  <code>INSERT</code> The statement is used to create a table called producins, which reads as follows:</p>
<pre><code class="language-sql">CREATE TABLE productins
(product_id    CHAR(4)      NOT NULL,
product_name   VARCHAR(100) NOT NULL,
product_type   VARCHAR(32)  NOT NULL,
sale_price     INTEGER      DEFAULT 0,
purchase_price INTEGER ,
regist_date    DATE ,
PRIMARY KEY (product_id));
</code></pre>
<p><em>Here.<code>sale_price</code>Columns are bound by default and will be replaced with default when NULL is inserted</em></p>
<h4>Insert INSERT statement</h4>
<p><code>INSERT</code> statement Insert a line Basic syntax:</p>
<pre><code class="language-sql">INSERT INTO &lt;表名&gt; (列1, 列2, 列3, ……) VALUES (值1, 值2, 值3, ……);  
</code></pre>
<p>When you complete the table with INSERT, you can omit the list after the name. The value of the VALUES clause is then given to each column by default in a left-to-right order.</p>
<p>An example to insert is</p>
<pre><code class="language-sql">-- 包含列清单
INSERT INTO productins (product_id, product_name, product_type, sale_price, purchase_price, regist_date) VALUES (&#39;0005&#39;, &#39;高压锅&#39;, &#39;厨房用具&#39;, 6800, 5000, &#39;2009-01-15&#39;);
-- 省略列清单
INSERT INTO productins VALUES (&#39;0005&#39;, &#39;高压锅&#39;, &#39;厨房用具&#39;, 6800, 5000, &#39;2009-01-15&#39;);  
</code></pre>
<p>In principle, an INSERT statement is inserted in a row. When you insert multiple rows, you usually need to recycle the corresponding number of INSERT statements.</p>
<p>In INSERT statements you can write NULL directly in the list of values of the VALUES sub-rules if you want to give a NULL value to a column. Columns that want to insert NULL must not set the NOT NULL constraint.</p>
<h4>Copy Insert From Other Tables</h4>
<p>You can copy data from other tables using the INSERT... SELECT statement.</p>
<pre><code class="language-sql">-- 将商品表中的数据复制到商品复制表中
INSERT INTO productcopy (product_id, product_name, product_type, sale_price, purchase_price, regist_date)
SELECT product_id, product_name, product_type, sale_price, purchase_price, regist_date
  FROM Product;  
</code></pre>
<h3>Index</h3>
<p>MySQL index is important for the efficient operation of MySQL, and the index can significantly increase MySQL's retrieval speed. In fact,<strong>The index is for a quick search behind us.</strong></p>
<p>When creating a table, you can directly create an index with the following syntax:</p>
<pre><code class="language-sql">CREATE TABLE mytable(

ID INT NOT NULL,

username VARCHAR(16) NOT NULL,

INDEX [indexName] (username(length))

);
</code></pre>
<p>You can also create:</p>
<pre><code class="language-sql">-- 方法1
CREATE INDEX indexName ON table_name (column_name)

-- 方法2
ALTER table tableName ADD INDEX indexName(columnName)
</code></pre>
<p><strong>The index is more complex, and we'll introduce it later.</strong></p>
<h2>Basic queries and sorting</h2>
<h3>SELECT Statement Basis</h3>
<h4>Basic queries</h4>
<p>When selecting data from a table, you need to use a SELECT statement, which means that only necessary data are selected from the table. The process of searching and extracting the necessary data through a SELECT statement is called matching query or query (query).</p>
<p>The underlying SELECT statements include both SELECT and FROM.</p>
<pre><code class="language-sql">SELECT &lt;列名&gt;,
  FROM &lt;表名&gt;;
</code></pre>
<p>Of these, the SELECT sentence lists the names of the columns that wish to be consulted from the table, while the FROM sentence specifies the name of the table from which the data are selected. It's acceptable to choose more than one column.<strong>No comma between words</strong>It's a split symbol.</p>
<h4>Selective queries</h4>
<p>WHERE statements are used when the full data need not be removed, but when the data are selected to meet certain conditions such as “commodity type is clothing” “sale unit price above Yen1,000”.
In **WHERE sub- sentences you can specify conditions such as " the value of a column is equal to that string " or " the value of a column is greater than that number " . ** The execution of a SELECT statement containing these conditions makes it possible to search for records that only meet that condition.</p>
<pre><code class="language-sql">SELECT &lt;列名&gt;, ……
  FROM &lt;表名&gt;
 WHERE &lt;条件表达式&gt;;
</code></pre>
<h4>Some additional knowledge</h4>
<ul>
<li>The asterisk represents the meaning of all the columns.</li>
<li>SQL is free to use line breaks without prejudice to execution</li>
<li>Double quotes are required when setting a Chinese aliases&quot;It's covered up.</li>
<li>Use DISTINCT in SELECT statements to delete duplicate lines.</li>
</ul>
<pre><code class="language-sql">-- 想要查询出全部列时，可以使用代表所有列的星号（*）。
SELECT *
  FROM &lt;表名&gt;；
-- SQL语句可以使用AS关键字为列设定别名（用中文时需要双引号（“”））。
SELECT product_id     As id,
       product_name   As name,
       purchase_price AS &quot;进货单价&quot;
  FROM product;
-- 使用DISTINCT删除product_type列中重复的数据
SELECT DISTINCT product_type
  FROM product;
</code></pre>
<h3>arithmetical and comparative operators</h3>
<h4>Count Operators</h4>
<p>The four main operators available in the SQL statement are as follows:</p>
<table>
<thead>
<tr>
<th align="left">Meaning</th>
<th align="left">Operators</th>
</tr>
</thead>
<tbody><tr>
<td align="left">Add</td>
<td align="left">+</td>
</tr>
<tr>
<td align="left">Subtract</td>
<td align="left">-</td>
</tr>
<tr>
<td align="left">Multiplication</td>
<td align="left">*</td>
</tr>
<tr>
<td align="left">Division</td>
<td align="left">/</td>
</tr>
</tbody></table>
<p>They only run calculations for INTEGER.</p>
<h4>Compare operators</h4>
<table>
<thead>
<tr>
<th>Operators</th>
<th>Meaning</th>
</tr>
</thead>
<tbody><tr>
<td>=</td>
<td>Equal</td>
</tr>
<tr>
<td>&lt;&gt;</td>
<td>Unequal</td>
</tr>
<tr>
<td>&gt;=</td>
<td>greater than or equal to</td>
</tr>
<tr>
<td>&lt;=</td>
<td>less than or equal to</td>
</tr>
<tr>
<td>&gt;</td>
<td>Greater than</td>
</tr>
<tr>
<td>&lt;</td>
<td>less than</td>
</tr>
</tbody></table>
<p>They only run calculations for INTEGER.</p>
<h4>Common Law</h4>
<ul>
<li>A constant or expression can be used in the SELECT sub-statement.</li>
<li>String type data are in principle sorted according to dictionary order and cannot be confused with the numerical size order. If we can avoid this as much as we can.</li>
<li>If you want to select a NULL record, use IS NULL operator in a condition expression. If you want to select a record that is not a NULL, use IS NOT NULL operator in a condition expression.</li>
</ul>
<pre><code class="language-sql">-- SQL语句中也可以使用运算表达式
SELECT product_name, sale_price, sale_price * 2 AS &quot;sale_price x2&quot;
  FROM product;
-- WHERE子句的条件表达式中也可以使用计算表达式
SELECT product_name, sale_price, purchase_price
  FROM product
 WHERE sale_price - purchase_price &gt;= 500;
/* 对字符串使用不等号
首先创建chars并插入数据
选取出大于‘2’的SELECT语句*/
-- DDL：创建表
CREATE TABLE chars
（chr CHAR（3）NOT NULL,
PRIMARY KEY（chr））;
-- 选取出大于&#39;2&#39;的数据的SELECT语句(&#39;2&#39;为字符串)
SELECT chr
  FROM chars
 WHERE chr &gt; &#39;2&#39;;
-- 选取NULL的记录
SELECT product_name, purchase_price
  FROM product
 WHERE purchase_price IS NULL;
-- 选取不为NULL的记录
SELECT product_name, purchase_price
  FROM product
 WHERE purchase_price IS NOT NULL;
</code></pre>
<h3>Logical Operators</h3>
<p>Logical operators help us continue the conditions in the WHERE statement of complex inquiries, giving us greater freedom</p>
<h4>NOT Operator</h4>
<p>Wants to indicate <code>不是……</code> Except for the previous one.&lt;&gt;There is another negative operator in addition to the operator: NOT.</p>
<p>NOT cannot be used alone and must be used in combination with other query conditions such as</p>
<pre><code class="language-sql">SELECT product_name, product_type, sale_price
  FROM product
 WHERE NOT sale_price &gt;= 1000;
</code></pre>
<p>If it's not necessarily necessary to use the comparative operator directly, it's more intuitive.</p>
<h4>AND and OR operators</h4>
<p>An AND or OR operator can be used if you want to use multiple search conditions at the same time.</p>
<p>AND is equivalent to "and" similar to the intersection in mathematics;
OR is equivalent to "or", similar to the combination in mathematics.</p>
<p>Note that, following the complexity of the operators, we need to express the priorities of the various calculations well in brackets and indents, so as not to lead to errors in results due to the priority of the logical operations, as follows:</p>
<pre><code class="language-sql">-- 通过使用括号让OR运算符先于AND运算符执行
SELECT product_name, product_type, regist_date
  FROM product
 WHERE product_type = &#39;办公用品&#39;
   AND ( regist_date = &#39;2009-09-11&#39;
        OR regist_date = &#39;2009-09-20&#39;);
</code></pre>
<p>Apparently, we are. <code>regist_date</code> There's a clear choice, then he and ours. <code>product_type</code>Column conditions side by side</p>
<h3>Aggregation queries</h3>
<p>The function used in SQL for aggregation is called a polymer function. The following five aggregate functions are most commonly used:</p>
<ul>
<li>SUM: Computes total values in a value column in a value table</li>
<li>AVG: Calculation of averages in a value column in the table</li>
<li>MAX: Maximum values for data in any column in the calculation table, including text type and number type</li>
<li>MIN: Calculate the minimum values for data in any column in the table, including text type and number type</li>
<li>COUNT: Number of entries in the calculation table (lines)</li>
</ul>
<p>Their grammar rules have changed somewhat more than the basic SELECTs.</p>
<pre><code class="language-sql">-- 计算销售单价和进货单价的合计值
SELECT SUM(sale_price), SUM(purchase_price)
  FROM product;
-- 计算销售单价和进货单价的平均值
SELECT AVG(sale_price), AVG(purchase_price)
  FROM product;
-- 计算销售单价的最大值和最小值
SELECT MAX(sale_price), MIN(sale_price)
  FROM product;
-- MAX和MIN也可用于非数值型数据
SELECT MAX(regist_date), MIN(regist_date)
  FROM product;
-- 计算全部数据的行数（包含 NULL 所在行）
SELECT COUNT(*)
  FROM product;
-- 计算 NULL 以外数据的行数
SELECT COUNT(purchase_price)
  FROM product;
</code></pre>
<p>Sometimes, multiple lines may contain exactly the same data, and we may want to delete duplicate data.</p>
<pre><code class="language-sql">SELECT COUNT(DISTINCT product_type)
  FROM product;
</code></pre>
<ul>
<li>The COUNT function operation results are associated with parameters, COUNT(asterisk) / COUNT(1) gets all rows containing NULL values, COUNT()&lt;Listing&gt;) Gets all rows that do not contain NULL values.</li>
<li>The polymer function does not process rows containing NULL values, except for COUNT.</li>
<li>The MAX/ MIN function applies to columns of text type and number type, while the SUM/ AVG function only applies to columns of digital type.</li>
<li>Use DISTINCT keywords in the parameters of the polymer function, and you can remove the polymer result of duplicate values.</li>
</ul>
<h3>Cluster statistics</h3>
<p>The previous use of the polymer function would have processed the entire table, and when you wanted to group the data together (i.e. aggregate the existing data by a column), GROUP BY could have helped us.</p>
<p>Syntax:</p>
<pre><code class="language-sql">-- 在SELECT中使用的全部非聚合列，都应该出现在GROUP BY中，否则逻辑出错
SELECT &lt;列名1&gt;,&lt;列名2&gt;, &lt;列名3&gt;, ……
  FROM &lt;表名&gt;
 GROUP BY &lt;列名1&gt;, &lt;列名2&gt;, &lt;列名3&gt;, ……;
</code></pre>
<p>A very basic example.</p>
<pre><code class="language-sql">-- 按照商品种类统计数据行数
SELECT product_type, COUNT(*)
FROM product
WHERE product_type IN (&#39;衣服&#39;, &#39;鞋类&#39;)
GROUP BY product_type;
 -- 不含GROUP BY 此代码非法，因为混合使用了聚合列以及普通列且无分组
SELECT product_type, COUNT(*)
FROM product
WHERE product_type IN (&#39;衣服&#39;, &#39;鞋类&#39;)
</code></pre>
<p>At this point in time, we have chosen a statement based on <code>GROUP BY</code> in which the specified columns are aggregated</p>
<p>NULL aggregates as a special set of data</p>
<p>When using normal columns and polymer functions,<strong>Must use <code>GROUP BY</code> To specify how to group data</strong> Otherwise, the database does not know how to handle multiple rows of data for a polymer.</p>
<h3>Specify conditions for group results</h3>
<p><code>GROUP BY</code> To help us achieve clusters, how can we remove special groups from them? That's what we're here to study.</p>
<p>You can use the HAVING substatement after GRUP BY. HAVING uses similar to WHERE.</p>
<p>It is noteworthy that:<strong>HAVING sub-words must be used in conjunction with GRUP BY sub-phrases and limited to group aggregation results, i.e. keys used here need to be included in SELECT</strong>, the WHERE clause is a limited data line (including grouping columns), each of which has its own function and in which the keys used are not required by SELECT.</p>
<p>HAVING'S IMPLEMENTING CLASS, JUST BEFORE ORDER BY</p>
<p>Here are some examples.</p>
<pre><code class="language-sql">-- 只要行数大于2的分组
SELECT product_type, COUNT(*)
  FROM product
 GROUP BY product_type
HAVING COUNT(*) = 2;

-- 错误形式（因为product_name不包含在GROUP BY聚合键中）
SELECT product_type, COUNT(*)
  FROM product
 GROUP BY product_type
HAVING product_name = &#39;圆珠笔&#39;;
</code></pre>
<h3>Sort Query Results</h3>
<p>In some scenarios, a sorted result is required. And the SQL execution results are by default randomly sorted in order, using <strong>ORDER BY</strong> Second sentence. And the language code is...</p>
<pre><code class="language-sql">SELECT &lt;列名1&gt;, &lt;列名2&gt;, &lt;列名3&gt;, ……
  FROM &lt;表名&gt;
 ORDER BY &lt;排序基准列1&gt; [ASC, DESC], &lt;排序基准列2&gt; [ASC, DESC], ……
</code></pre>
<p>of which the parameter ASC is an ascending order, the parameter ASC is an descending order, and the default is an ascending order, at which point the parameter ASC can be defaulted.</p>
<p>Example:</p>
<pre><code class="language-sql">-- 降序排列
SELECT product_id, product_name, sale_price, purchase_price
  FROM product
 ORDER BY sale_price DESC;

-- 多个排序键
SELECT product_id, product_name, sale_price, purchase_price
  FROM product
 ORDER BY sale_price DESC, product_id DESC;
</code></pre>
<p>NULL cannot compare, so the sorting randomly places them at the beginning or at the end. In MySQL,<code>NULL</code> Value considered compared to any <code>非NULL</code> Low value, so when the order is ASC,<code>NULL</code> value appears in first place, and when the order is DESC, the order is at last.</p>
<h3>About the order of execution of statements</h3>
<p>GROUP BY mentions that the aliases defined in the SELECT clause cannot be used in GROUP BY, but may be used in ORDER BY.</p>
<p>This is because SQL executes the SELECT statement in the following order when using the HAVING clause:
FORM WHERE GRUP BY SELECT HAVING ORDER BY</p>
<p>Of which SELECT is executed in the order that follows the GRUP BY clause, before the ORDER BY clause. Therefore, aliases can be used in the ORDER BY sentence, but not in GRUP BY.</p>
<h2>Complex queries</h2>
<h3>View</h3>
<h4>Basic knowledge of view</h4>
<p>View is a virtual table, different from a direct operating data sheet, created on the basis of a SELECT statement (specifically described below), so the operating view generates a virtual table based on the SELECT statement that created the view, and then does SQL on this virtual table.</p>
<p><strong>Difference between view and table - - Whether actual data are saved</strong> So view is not a data sheet that is actually stored in the database; it can be seen as a window through which we can see the real data in the database table.</p>
<p>Now that we have the data sheets, why do we need a view? The main reasons are the following:</p>
<ol>
<li>By defining the view<strong>Save frequently used SELECT statements</strong>To increase efficiency.</li>
<li>By defining the view, the data seen by the user can be made clearer.</li>
<li>By defining the view, the entire field of the data sheet can be kept closed and the confidentiality of the data enhanced.</li>
<li>By defining the view, you can reduce the redundancy of the data.</li>
</ol>
<h4>Create View</h4>
<p>The basic syntax for creating view is as follows:</p>
<pre><code class="language-sql">CREATE VIEW &lt;视图名称&gt;(&lt;列名1&gt;,&lt;列名2&gt;,...) AS &lt;SELECT语句&gt;
</code></pre>
<p>Of which SELECT statements need to be written after AS keywords. The order of the columns in the SELECT statement and the order of the columns in the view are the same. Column 1 in the SELECT statement is column 1 in the view, and column 2 in the SELECT statement is column 2 in the view, so push. And the view listing is defined in the list after the view name.</p>
<p>Note that the view name needs to be the only one in the database and cannot be renamed with other views and tables.</p>
<p>View can be based not only on a real table, but also on a view. But multiple views reduce the performance of SQL.</p>
<p>Unable to use during view creation <code>SELECT</code> Yes. <code>ORDER BY</code> Sub sentences because the lines in the view he created should not be sequential <em>The definition of view in MySQL allows the use of ORDER BY statement but avoids it as much as possible</em></p>
<p>Create an example of a view based on a table below</p>
<pre><code class="language-sql">CREATE VIEW productsum (product_type, cnt_product)
AS
SELECT product_type, COUNT(*)
  FROM product
 GROUP BY product_type ;
</code></pre>
<p>Create an example of a view based on multiple tables</p>
<pre><code class="language-sql">CREATE VIEW view_shop_product(product_type, sale_price, shop_name)
AS
SELECT product_type, sale_price, shop_name
  FROM product,
       shop_product
 WHERE product.product_id = shop_product.product_id;
--这使用的查询方法是隐式内连接，是模仿INNER JOIN的语句，效果一样
</code></pre>
<p>The views we create are well understood in code, and it's natural to take the data in multiple tables at the same time.</p>
<h4>Modify View Structure</h4>
<p>Syntax:</p>
<pre><code class="language-sql">ALTER VIEW &lt;视图名&gt; AS &lt;SELECT语句&gt;
</code></pre>
<p>It's basically the same as when we delete and create.</p>
<p>Delete the basic syntax of the view as follows:</p>
<pre><code class="language-sql">DROP VIEW &lt;视图名1&gt; [ , &lt;视图名2&gt; …]
</code></pre>
<h4>Update View</h4>
<p>Because the view is a virtual table, the action on the view is the action on the bottom base table, so the change can only be made successfully if the definition of the bottom base table is met.</p>
<p>Some changes can be made, but...<strong>However, this use is not recommended. And when we create the view, we try to use the limit not to allow changes to the table through the view.</strong></p>
<p>Cannot be updated when view contains the following statement</p>
<ul>
<li>Aggregation functions SUM(), MIN(), MAX(), COUNT() etc.</li>
<li>DISTINCT keyword.</li>
<li>GRUP BY sentence.</li>
<li>HAVING subword.</li>
<li>Union or Union All Operator.</li>
<li>FROM sub sentences contain multiple tables.</li>
</ul>
<h3>SubQuery</h3>
<h4>What's a sub-survey?</h4>
<p>A simple example.</p>
<pre><code class="language-sql">SELECT stu_name
FROM (
         SELECT stu_name, COUNT(*) AS stu_cnt
          FROM students_info
          GROUP BY stu_age) AS studentSum;
</code></pre>
<p>The statement appears to be well understood, with the use of bracketed sql statements to be executed first, and then outside sql to be executed after success. This is the statement of the sub-survey. (There is a logical problem with the code itself, which does not meet the need for cluster aggregation)</p>
<p>Sub-Query refers to a query where a query statement is embedded within another query statement, which calculates a sub-Query in a SELECT sub-statement, and the sub-Query results are a filter condition for another query in the outer layer, which can be based on a table or tables.</p>
<p>Sub-Query is the direct use of the SELECT statement used to define the view in the FROM sub-rule. Of these, AS studentsum can be considered a sub-searcher and the sub-searcher is one-off.</p>
<p><strong>It is true that embedded multi-layer queries can yield results, but they reduce the readability of SQL statements and result in less efficient implementation and avoid them as much as possible</strong></p>
<h4>Standard Quantum Query</h4>
<p>The scale is a single, then the quantum query is a single sub-search, and the single is the SQL statement that we are required to execute returns only one value, that is, returns the specifics of the table.<strong>A column in a row</strong>。</p>
<p>All places where a single value is required can be searched with a standard quantum, for example.</p>
<pre><code class="language-sql">SELECT product_id, product_name, sale_price
  FROM product
 WHERE sale_price &gt; (SELECT AVG(sale_price) FROM product);
</code></pre>
<h4>Association Sub-Query</h4>
<p>Since the association sub-inquiries contain two words, they must mean that there is a link between the query and the sub-inquiries.</p>
<p>Requirements<code>选取出各商品种类中高于该商品种类的平均销售单价的商品</code>I don't know. The SQL statement reads as follows:</p>
<pre><code class="language-sql">SELECT product_type, product_name, sale_price
  FROM product AS p1
 WHERE sale_price &gt; (SELECT AVG(sale_price)
                       FROM product AS p2
                      WHERE p1.product_type =p2.product_type
                      GROUP BY product_type);
</code></pre>
<p>The so-called connection is the use of a number of markers to connect both internal and external layers of queries for the purpose of filtering data. We mark the outside product form as p1, set the internal product as p2 and connect two queries through the WERERE statement.</p>
<p>The basic implementation line is</p>
<ol>
<li>First do the main query without WERE</li>
<li><strong>Match program type with main query results, get sub-search results</strong></li>
<li>Execute the complete SQL statement in conjunction with the main query</li>
</ol>
<p><strong>View and sub-Query are more based elements in database operations, and some complex queries require a combination of sub-Query conditions to get the correct result. In any case, however, the SQL language should not be designed in such deep and particularly complex layers that not only readability, but also the efficiency of implementation, is difficult to ensure, so as to be as concise as possible to perform the required functions.</strong></p>
<h3>Various Functions</h3>
<p>Sql carries a variety of functions, which greatly enhances the convenience of the sql language.</p>
<p>The functions are broadly divided into the following categories:</p>
<ul>
<li>Algorithmic functions (functions used for numerical calculations)</li>
<li>String Functions (functions used for string operations)</li>
<li>Date Functions (functions used for date operations)</li>
<li>Convert Functions (functions used to convert data types and values)</li>
<li>Aggregation Functions (letters used for data aggregation)</li>
</ul>
<p>The total number of functions is over 200 and does not need to be fully remembered. The number of commonly used functions is 30-50, so that you can look at the document when other non-used functions are used.</p>
<h4>Count Functions</h4>
<ul>
<li><p>ABS - Absolute Value
Syntax:<code>ABS( 数值 )</code>
The ABS function is used to calculate the absolute value of a number, representing the distance from a number to its original point. When the parameter for the ABS function is<code>NULL</code>, return value<code>NULL</code>。</p>
</li>
<li><p>MOD -- Ask for balance
Syntax:<code>MOD( 被除数，除数 )</code>
The MOD is a function of calculating the residual number (excess) and is the abbreviation of the modulo. There is no concept of the number of decimals, but only the number of integer columns.
Note: Mainstream DBMS supports the MOD function, only SQL Server does not support it, and it is used<code>%</code>symbol to calculate the balance.</p>
</li>
<li><p>ROUND -- rounded
Syntax:<code>ROUND( 对象数值，保留小数的位数 )</code>
The ROUND function is used for rounding operations.
Note: when parameters <strong>Keep decimal places</strong> When you are a variable, you may encounter a mistake, so be careful with the variable.</p>
</li>
</ul>
<h4>String Functions</h4>
<ul>
<li><p>CONCAT -- Spelling
Syntax:<code>CONCAT(str1, str2, str3)</code>
MySQL uses the CONCAT function to spell.</p>
</li>
<li><p>LENGTH - String Length
Syntax:<code>LENGTH( 字符串 )</code></p>
</li>
<li><p>LOWER -- lowercase conversion
The LOWER function, which can only be used for letters, converts all strings in parameters to lowercase. This function does not apply to places other than English letters, without prejudice to characters that are originally lowercase. Similarly, the UPPER function is used for capitalisation.</p>
</li>
<li><p>REPLACE - Replace String
Syntax:<code>REPLACE( 对象字符串，替换前的字符串，替换后的字符串 )</code></p>
</li>
<li><p>SUBSTRING - Interception of Strings
Syntax:<code>SUBSTRING （对象字符串 FROM 截取的起始位置 FOR 截取的字符数）</code>
Use the SUBSTRING function to intercept a part of a string. The starting position of the intercept starts with the leftmost side of the string and the index value starts with 1.</p>
</li>
<li><p>SUBSTRING INDEX - Indexed
Syntax:<code>SUBSTRING_INDEX (原始字符串， 分隔符，n)</code>
This function supports positive and reverse indexing with initial values of 1 and -1 respectively, after the original string is separated by separator.</p>
</li>
<li><p>REPEAT - String repeats repeatedly as required
Syntax:<code>REPEAT(string, number)</code></p>
</li>
</ul>
<h4>Date Functions</h4>
<p>Different DBMS date functions vary, and this course presents a number of functions recognized by standard SQL that can be applied to most DBMS. A specific DBMS date function is sufficient to access the document.</p>
<ul>
<li><p>Current DATE - Get the current date</p>
</li>
<li><p>CURRENT TIME -- Current Time</p>
</li>
<li><p>CURRENT TIMESTAMP - Current date and time</p>
</li>
<li><p>EXTRACT - Intercept Date Elements
Syntax:<code>EXTRACT(日期元素 FROM 日期)</code>
Use the EXTRACT function to intercept part of the date data, e.g. " year " month, " hour " seconds, etc. The return value of the function is not a date type but a value type</p>
</li>
</ul>
<h4>Convert Functions</h4>
<p>The term "conversion" has a very broad meaning, and in SQL there are two main meanings: the first is the conversion of data types, abbreviated as the conversion of types, referred to in English as<code>cast</code>; the other level means value conversion.</p>
<ul>
<li><p>CAST - Type Conversion
Syntax:<code>CAST（转换前的值 AS 想要转换的数据类型）</code>
Of particular note is the need to specify SIGNED or UNSIGNED when converting to integers</p>
</li>
<li><p>COALESSE - Find non-NULLs for ease of conversion
Syntax:<code>COALESCE(数据1，数据2，数据3……)</code>
COALESE is a function specific to SQL. This function returns the value of NULL that begins on the left side of variable A. The number of parameters is variable and can therefore be increased indefinitely as required.</p>
</li>
</ul>
<h3>Word</h3>
<p>The word returns a function whose value is real. Including<code>TRUE / FALSE / UNKNOWN</code>I don't know. Besides the functions that we described in the Comparative Operators section, there are functions that produce a boolean value.</p>
<p>The terms are as follows:</p>
<ul>
<li>LIKE</li>
<li>BETWEEN</li>
<li>IS NULL、IS NOT NULL</li>
<li>IN</li>
<li>EXISTS</li>
</ul>
<h4>LIKE</h4>
<p>The word is used when partial consistency of a string is required.</p>
<p>Partial consistency can broadly be divided into three categories: forward, intermediate and rear.</p>
<h5>Unanimously forward.</h5>
<p>The syntax structure of the front is as follows:<code>WHERE strcol LIKE &#39;ddd%&#39;</code>  </p>
<p>The string (here is 'dddd) that is the same as the beginning of the string of the query object, which is the condition of the query.</p>
<p>Of which<code>%</code>is a special symbol for " Zero or more arbitrary strings " , which in this case is for " all strings starting with ddd " .</p>
<h5>Midway</h5>
<p>Intermediately, the query object string contains a string as a condition for the query, whether it appears at the end or in the middle of the object string.</p>
<p>Syntax:<code>WHERE strcol LIKE &#39;%ddd%&#39;</code></p>
<h5>Rear Unanimously</h5>
<p>The string (here'dddd) that is the condition of the query is the same as the end of the string of the query object.</p>
<p>Syntax:<code>WHERE strcol LIKE &#39;%ddd&#39;</code></p>
<h5>Any Character</h5>
<p>Use the underlined instead of %, unlike %, it represents "any one character".</p>
<p>Syntax:<code>WHERE strcol LIKE &#39;abc__&#39;</code></p>
<h4>Between.</h4>
<p>Use BETWEEN to make range queries. The term differs from other terms or functions in that it uses three parameters.</p>
<p>Let's give you one example:</p>
<pre><code class="language-sql">-- 选取销售单价为100～ 1000元的商品
SELECT product_name, sale_price
FROM product
WHERE sale_price BETWEEN 100 AND 1000;
</code></pre>
<p>The Between feature is that the result will contain 100 and 1000 thresholds, i.e., the closed space. If you don't want the result to contain a threshold, you have to use it. &lt; and &gt;</p>
<h4>IS NULL、 IS NOT NULL</h4>
<p>In order to select the data for certain columns with values NULL, the = cannot be used, but only the specific term IS NULL can be used.</p>
<p>On the contrary, if you want to select data other than NULL, use IS NOT NULL.</p>
<p>The reason for this rule is that SQL uses a three-value boolean, that NULL means unknown, and that any non-relative calculation he makes of UNKNOWN.</p>
<h4>In word</h4>
<p>You can choose to use multiple query conditions when taken together<code>or</code>statement.</p>
<p>That is, as more and more people want to choose, use<code>or</code>The SQL statement will be longer and harder to read. So we can replace the OR statement with the IN word `IN.</p>
<p>One simple example:</p>
<pre><code class="language-sql">SELECT product_name, purchase_price
FROM product
WHERE purchase_price IN (320, 500, 5000);
</code></pre>
<p>By the same token, we can use the negative form NOT IN. </p>
<p>It should be noted that Null data cannot be selected when using IN and NOT IN.</p>
<p>In particular, the IN statements are often used in combination with sub-references, for example.</p>
<pre><code class="language-sql">-- 取出大阪门店在售商品的销售单价 `sale_price`
SELECT product_name, sale_price
FROM product
WHERE product_id IN (SELECT product_id
  FROM shopproduct
                       WHERE shop_id = &#39;000C&#39;);
</code></pre>
<p>Sub-Query provides a new table that gives IN the word space to search.</p>
<h4>EXIST term</h4>
<p>The use of the EXIST word is difficult to understand.</p>
<p>1 EXIST is used differently than before</p>
<p>2 Syntax:</p>
<p>In fact, if you don't use EXIST, you can basically use IN instead.</p>
<p>In that case, is there a need to learn the EXIST word? The answer is yes, because once the EXIST word is used, it's extremely convenient.</p>
<p>You need not be too worried, however, that this course presents some basic usages, and more attention can be paid to the use of the EXIST term in future studies, so that it can be used when reaching the SQL intermediate level.</p>
<ul>
<li>Methodology for the use of EXIST</li>
</ul>
<p>That's the word. <strong>“to determine whether there is a record of fulfilment of certain conditions”</strong>。</p>
<p>If such a record exists, it returns true (TRAE) or, if it does not exist, false (FALSE).</p>
<p>The main word for the term EXIST (existence) is “record”.</p>
<p>We continue to use the example of IN and Sub-Query, using EXIST to select the unit price of the sale of goods at Osaka Gate.</p>
<pre><code class="language-sql">SELECT product_name, sale_price
  FROM product AS p
 WHERE EXISTS (SELECT *
                 FROM shopproduct AS sp
                WHERE sp.shop_id = &#39;000C&#39;
                  AND sp.product_id = p.product_id);
+--------------+------------+
| product_name | sale_price |
+--------------+------------+
| 运动T恤      |       4000 |
| 菜刀         |       3000 |
| 叉子         |        500 |
| 擦菜板       |        880 |
+--------------+------------+
4 rows in set (0.00 sec)
</code></pre>
<ul>
<li>EXIST parameters</li>
</ul>
<p>The terms we have learned before are basically like " Column LIKE string " or " Column BETWEEN value 1 AND value 2 " , which requires the specification of more than two parameters, while the left side of EXIST does not have any parameters. Because EXIST is a word for only one parameter. So EXIST only needs to write one parameter on the right, which is usually a sub-search.</p>
<pre><code class="language-sql">(SELECT *
   FROM shopproduct AS sp
  WHERE sp.shop_id = &#39;000C&#39;
    AND sp.product_id = p.product_id)  
</code></pre>
<p>Such a sub-Query above is the only parameter. To be precise, as the link between the produd and the shopprodud tables is made through the condition “SP.produtt id = P.produtt id”, the associated sub-reference is the parameter. EXIST usually uses associated sub-inquiries as parameters.</p>
<ul>
<li>SELECT*</li>
</ul>
<p>Since EXIST only cares about the existence of records, it is irrelevant to return any columns. EXIST only determines whether there is a condition specified in the WHERE sub-rule "shop number (shop id) as &#39;000C&#39;, product forms and stores</p>
<p>The record of the commodity number (product id) in the list of commodities is identical and returns only when such a record exists.</p>
<p>Therefore, using the query statement below, the search results will not change.</p>
<pre><code class="language-sql">SELECT product_name, sale_price
  FROM product AS p
 WHERE EXISTS (SELECT 1 -- 这里可以书写适当的常数
                 FROM shopproduct AS sp
                WHERE sp.shop_id = &#39;000C&#39;
                  AND sp.product_id = p.product_id);
+--------------+------------+
| product_name | sale_price |
+--------------+------------+
| 运动T恤      |       4000 |
| 菜刀         |       3000 |
| 叉子         |        500 |
| 擦菜板       |        880 |
+--------------+------------+
4 rows in set (0.00 sec)
</code></pre>
<blockquote>
<p>You can use writing SELECT* in an EXIST sub-reference as a custom for SQL.</p>
</blockquote>
<ul>
<li>Replace NOT IN with NOT EXIST</li>
</ul>
<p>Just like EXIST can replace IN, NOT IN can also be replaced by NOT EXIST.</p>
<p>The following code examples are taken out of the unit prices for the sale of goods not sold at the Tokyo Gate.</p>
<pre><code class="language-sql">SELECT product_name, sale_price
  FROM product AS p
 WHERE NOT EXISTS (SELECT *
                     FROM shopproduct AS sp
                    WHERE sp.shop_id = &#39;000A&#39;
                      AND sp.product_id = p.product_id);
+--------------+------------+
| product_name | sale_price |
+--------------+------------+
| 菜刀         |       3000 |
| 高压锅       |       6800 |
| 叉子         |        500 |
| 擦菜板       |        880 |
| 圆珠笔       |        100 |
+--------------+------------+
5 rows in set (0.00 sec)
</code></pre>
<p>NOT EXIST, unlike EXIST, returns real (TRUE) when "does not exist " fulfils the record specified in the sub-search.</p>
<h3>CASE Expression</h3>
<h4>What's a CASE expression?</h4>
<p>CASE expression is a function. It's an important function of the SQL medium count. It's important to learn.</p>
<p>CASE expressions are used to distinguish situations, which are commonly referred to as branches (conditions) in programming.</p>
<p>The syntax of CASE expression is divided into simple CASE expression and search for CASE expression. Because the search for CASE expression contains all the functions of a simple CASE expression. This course will highlight the search for CASE expressions.</p>
<p>Syntax:</p>
<pre><code class="language-sql">CASE WHEN &lt;求值表达式&gt; THEN &lt;表达式&gt;
     WHEN &lt;求值表达式&gt; THEN &lt;表达式&gt;
     WHEN &lt;求值表达式&gt; THEN &lt;表达式&gt;
     .
     .
     .
ELSE &lt;表达式&gt;
END  
</code></pre>
<p>When the expression is executed, the statement after THEEN is executed, and if all the expressions are false, the statement after ELSE is executed.
No matter how large the CASE expression, it will only return one value.</p>
<h4>CASE use method</h4>
<p><strong>Apply scene 1: Different column values obtained from different branches</strong></p>
<pre><code class="language-sql">SELECT  product_name,
        CASE WHEN product_type = &#39;衣服&#39; THEN CONCAT(&#39;A ： &#39;,product_type)
             WHEN product_type = &#39;办公用品&#39;  THEN CONCAT(&#39;B ： &#39;,product_type)
             WHEN product_type = &#39;厨房用具&#39;  THEN CONCAT(&#39;C ： &#39;,product_type)
             ELSE NULL
        END AS abc_product_type
  FROM  product;

-- 下面是SELECT语句返回的表
+--------------+------------------+
| product_name | abc_product_type |
+--------------+------------------+
| T恤          | A ： 衣服        |
| 打孔器       | B ： 办公用品    |
| 运动T恤      | A ： 衣服        |
| 菜刀         | C ： 厨房用具    |
| 高压锅       | C ： 厨房用具    |
| 叉子         | C ： 厨房用具    |
| 擦菜板       | C ： 厨房用具    |
| 圆珠笔       | B ： 办公用品    |
+--------------+------------------+
</code></pre>
<p>The ELSE clause can also be omitted and will be defaulted as ELSE Null. But in order to prevent people from reading out, it is hoped that you will be able to write the ELSE subword in a prominent way.</p>
<p>In addition, the final “END” of the CASE expression cannot be omitted, and please be careful not to be omitted. Forget writing about ENDs can cause grammatical errors, which are the easiest mistakes at first school.</p>
<p><strong>Apply scenario 2: no change in aggregation/content in column direction and modify the presentation</strong>
Aggregation in the line direction is achieved using the method in the Syndication Query section, and convergence in the column direction requires reliance on CASE expression</p>
<p>Example 1</p>
<pre><code class="language-sql">-- 对按照商品种类计算出的销售单价合计值进行行列转换
SELECT SUM(CASE WHEN product_type = &#39;衣服&#39; THEN sale_price ELSE 0 END) AS sum_price_clothes,
       SUM(CASE WHEN product_type = &#39;厨房用具&#39; THEN sale_price ELSE 0 END) AS sum_price_kitchen,
       SUM(CASE WHEN product_type = &#39;办公用品&#39; THEN sale_price ELSE 0 END) AS sum_price_office
  FROM product;

-- 返回表结构
+-------------------+-------------------+------------------+
| sum_price_clothes | sum_price_kitchen | sum_price_office |
+-------------------+-------------------+------------------+
|              5000 |             11180 |              600 |
+-------------------+-------------------+------------------+
</code></pre>
<p>In fact, if we choose the traditional one-action structure of a column, direct use of GROUP BY Cluster Convergence, but transversalization requires this CASE expression. Pattern</p>
<p>Example 2</p>
<pre><code class="language-sql">-- CASE WHEN 实现数字列 score 行转列
SELECT name,
       SUM(CASE WHEN subject = &#39;语文&#39; THEN score ELSE null END) as chinese,
       SUM(CASE WHEN subject = &#39;数学&#39; THEN score ELSE null END) as math,
       SUM(CASE WHEN subject = &#39;外语&#39; THEN score ELSE null END) as english
  FROM score
 GROUP BY name;
+------+---------+------+---------+
| name | chinese | math | english |
+------+---------+------+---------+
| 张三 |      93 |   88 |      91 |
| 李四 |      87 |   90 |      77 |
+------+---------+------+---------+
</code></pre>
<p>The original table structure is a three-column structure of name subject score, and we've been transformed into such a line structure, inside. <code>SUM</code> This polymer function is only designed to convert a list returned by CASE to a number</p>
<p>Example 3</p>
<pre><code class="language-sql">-- CASE WHEN 实现文本列 subject 行转列
SELECT name,
       MAX(CASE WHEN subject = &#39;语文&#39; THEN subject ELSE null END) as chinese,
       MAX(CASE WHEN subject = &#39;数学&#39; THEN subject ELSE null END) as math,
       MIN(CASE WHEN subject = &#39;外语&#39; THEN subject ELSE null END) as english
  FROM score
 GROUP BY name;
+------+---------+------+---------+
| name | chinese | math | english |
+------+---------+------+---------+
| 张三 | 语文    | 数学 | 外语    |
| 李四 | 语文    | 数学 | 外语    |
+------+---------+------+---------+
</code></pre>
<p>It's not a practical example. It's just that we can use the text column if we want it. <code>MIN，MAX</code>This aggregate query function performs</p>
<h2>Pool Operations</h2>
<h3>About the assembly itself</h3>
<p><code>集合</code>In the mathematical field, the expression “summary of all things” and in the database area the expression of a collection of records. Specifically, the results of the execution of tables, views and queries are a collection of records, with the elements being a table or each line of the query results.</p>
<p>In standard SQL, use separate search results <code>UNION</code>，<code>INTERSECT</code>， <code>EXCEPT</code> to perform the search results in combination, intersection and differential calculations. The operator used to perform the pool operation is referred to as the pool operator.</p>
<p>In the database, all tables — as well as the search results — can be considered as a collection, so that the tables can also be considered as a collection for the above-mentioned aggregation operation, and in many cases this abstraction is very helpful in providing a workable idea of complex queries.</p>
<h3>Add to Table - UNION</h3>
<p>Let's start with an example.</p>
<pre><code class="language-sql">SELECT product_id, product_name
  FROM product
 UNION
SELECT product_id, product_name
  FROM product2;
</code></pre>
<p>As you can see, we remove the symbol that symbolizes the end of the sentence from the end of the preceding sentence, using<code>UNION</code>Connect them, and we'll end up with two teams.</p>
<p> <strong>UNION and others usually remove duplicate records.</strong></p>
<p>This weighting will remove not only the repetition of the two result sets, but also the repetition of the concentration of results. In practice, however, there is sometimes a need not to be heavy, and it is very simple to keep the grammatical of repeat lines in the results of UNION, just add an all-word to the UNION. As follows:</p>
<pre><code class="language-sql">-- 保留重复行
SELECT product_type
  FROM product
 UNION ALL
SELECT product_type
  FROM product2;
</code></pre>
<h3>Intersection Operator INTERSECT</h3>
<p>The condensation is the public part of the conglomerate, which, because of the differentity of the conglomerate elements, can be seen in a very intuitive way by means of the scribe.</p>
<p>Use <code>INTERSECT </code> The cross-coding code is as follows:</p>
<pre><code class="language-sql">TABLE product INTERSECT TABLE product2;
</code></pre>
<p>Use <code>INTERSECT</code> The number of columns in the two tables for which operators perform cross-counting must be the same.<strong>The same type of field is required.</strong></p>
<p><code>INTERSECT</code> Operator priority above <code>UNION</code> and <code>EXCEPT</code> , and when they appear, they give priority to cross-counting</p>
<p>For the two search results of the same table, their surrender to INTERSECT actually enables the search conditions for the two queries to be matched by the AND word.</p>
<h3>Discrepancies, patches and table subtractions</h3>
<p>The reduction of the sum difference is somewhat different from the reduction of the actual amount. When one pool A minus another B is used, a policy of direct neglect is applied to those elements that exist only in the pool B and not in the pool A, so that the reduction of A and B is only the reduction of those elements in the pool A that also belong to the pool B.</p>
<p>The statement using MySQL version 8.0.31 for differential computation is as follows:</p>
<pre><code class="language-sql">TABLE product EXCEPT TABLE product2;
</code></pre>
<p>In fact, the use of the NOT IN term can essentially achieve the same effect as the EXCEPT operation in SQL standard syntax.</p>
<p><strong>For those operations that do not provide direct operators, we can do this by combining them with terminologies and common query codes.</strong></p>
<h3>JOIN Foundation</h3>
<p>We've discussed a lot of ways to search and process data, but they can't get us to add the information that we have.</p>
<ul>
<li>Pool operations such as Union and INTERSECT operate in line directions</li>
<li>function or CASE expression equation, which increases the number of columns and does not provide more information in substance</li>
</ul>
<p>As we mentioned in the previous section on Linked Sub-Query, associated sub-Query allows us to obtain information from multiple tables at the same time, but<strong>Link</strong>Better to get information from multiple tables</p>
<p>JOIN is a connection.<strong>It's usually the equivalent of judgement.&quot;=&quot;</strong>), add columns in other tables for the " Add Columns " pool operation. It can be said that the connection is the core of the SQL query, that it is connected, that it is possible to obtain columns from two or more tables, that it is possible to simplify the past to a more readable form of an overly complex query, such as the use of associated sub-inquiries, and that it involves more complex queries.</p>
<h3>Interconnection (INNER JOIN)</h3>
<h4>Basic INNER JOIN</h4>
<p>The syntax format of the connection is:</p>
<pre><code class="language-sql">-- 内连结
FROM &lt;tb_1&gt; INNER JOIN &lt;tb_2&gt; ON &lt;condition(s)&gt;
</code></pre>
<p>Among them, the INNER keyword indicates the use of interconnectivity, which, for the time being, is not subject to scrutiny.</p>
<p>Using INNER JOIN in the FROM subword to connect two tables and to specify the link condition for the ON subphrase is shopprodud.produd id=product.produd id, the following query statement was obtained:</p>
<pre><code class="language-sql">SELECT SP.shop_id
       ,SP.shop_name
       ,SP.product_id
       ,P.product_name
       ,P.product_type
       ,P.sale_price
       ,SP.quantity
  FROM shopproduct AS SP
 INNER JOIN product AS P
    ON SP.product_id = P.product_id;
</code></pre>
<p>We have a simple aliases for each of the two tables. This is very common when using a connection.</p>
<p><strong>Point 1: Need to use multiple tables in the FROM sentence when connecting</strong>
<strong>Element 2: The link condition must be specified by using the ON clause</strong>
<strong>Point three: The column in the SELECT clause should preferably be used in the form of a lister.</strong></p>
<h4>Use an inline link in conjunction with the WHERE clause</h4>
<p>If you need to filter the search results with a WHERE sub-phrase at the same time as using an inline link, you need to write the WHERE sub-phrase behind the ON sub-phrase.</p>
<p>There are several ways to add the WHERE clause.</p>
<p>The first way to add the WEHRE clause is to use the above-mentioned query as a sub-search, encapsulating it in brackets, and then adding filter conditions to the outer layer of the query.</p>
<pre><code class="language-sql">SELECT *
  FROM (-- 第一步查询的结果
        SELECT SP.shop_id
               ,SP.shop_name
               ,SP.product_id
               ,P.product_name
               ,P.product_type
               ,P.sale_price
               ,SP.quantity
          FROM shopproduct AS SP
         INNER JOIN product AS P
            ON SP.product_id = P.product_id) AS STEP1
 WHERE shop_name = &#39;东京&#39;
   AND product_type = &#39;衣服&#39;;
</code></pre>
<p>If we know that the WHERE clause will be implemented after the FROM clause, that is, after the INNER JOIN ON gets a new table, we get the standard formula:</p>
<pre><code class="language-sql">SELECT  SP.shop_id
       ,SP.shop_name
       ,SP.product_id
       ,P.product_name
       ,P.product_type
       ,P.sale_price
       ,SP.quantity
  FROM shopproduct AS SP
 INNER JOIN product AS P
    ON SP.product_id = P.product_id
 WHERE SP.shop_name = &#39;东京&#39;
   AND P.product_type = &#39;衣服&#39;;
</code></pre>
<p>Implementation order: FROM sub-rule -&gt; What's the matter?&gt; SELEECT sub-rules; the two tables are linked by a linked column, and a new form is obtained, and the WERERE sub-rules screen the rows of this new table on two conditions, and finally the SELEECT sub-rule selects those columns that we need.</p>
<p>And of course, we can also use the WHERE to screen first in two tables, and then connect the two sub-inquiries.</p>
<pre><code class="language-sql">SELECT SP.shop_id
       ,SP.shop_name
       ,SP.product_id
       ,P.product_name
       ,P.product_type
       ,P.sale_price
       ,SP.quantity
  FROM (-- 子查询 1:从 shopproduct 表筛选出东京商店的信息
        SELECT *
          FROM shopproduct
         WHERE shop_name = &#39;东京&#39; ) AS SP
 INNER JOIN -- 子查询 2:从 product 表筛选出衣服类商品的信息
   (SELECT *
      FROM product
     WHERE product_type = &#39;衣服&#39;) AS P
    ON SP.product_id = P.product_id;
</code></pre>
<h4>Use inline connection in conjunction with GRUP BY sub-words</h4>
<p>The use of an inline link in conjunction with the GRUP BY sub-word requires a distinction according to which table is located in the cluster column.</p>
<p>In the simplest case, the GRUP BY clause was used before the inline link. </p>
<p>However, if the columns are not identical to the one that is consolidated and neither is used to link the two tables, they can only be linked and then aggregated.</p>
<p><strong>Just write the code according to our feelings. Group first and group later are on demand.</strong></p>
<h4>Self-connection (SELF JOIN)</h4>
<p>The previous interlinks were all linked to two different tables. But in practice a table can also be linked to itself, which is called self-connection. It is important to note that self-connection is not the third link that is distinguished from interconnection and interconnection, and that self-linking can be either external or internal, but is another classification that differs from interconnection.</p>
<h4>Inline links and association sub-inquiries</h4>
<p><strong>In thinking, the association sub-inquiries, which search for the same lines in the association column in table B, each of the rows of the table A takes, result in repeated queries resulting in a high cost calculation, and the inline links, although not functionally different, are significantly improved, both in performance and in grammar structure</strong></p>
<h4>Natural connection (NATURAL JOIN)</h4>
<p>Natural connections are not a third link, distinct from internal and external connections, but are a special case of interconnections -- when the two tables are naturally connected, the intra-value link is made by reference to the listings contained in both tables, and the use of ON is not required to specify the connection conditions.</p>
<p>Syntax:</p>
<pre><code class="language-sql">SELECT *  FROM shopproduct NATURAL JOIN product
</code></pre>
<h3>Extra JOIN</h3>
<p>The connection is to abandon the line of two tables that do not meet the requirements of the O.N., and the connection is to be linked. The outer link will selectively retain lines that cannot be matched according to the type of connection.</p>
<p>The outer link takes three forms according to which line is maintained: left link, right link and full external link.</p>
<p>Left-link saves lines that cannot be matched by an ON sub-rule in the left table, where the corresponding right-table lines are missing; right-link saves lines that cannot be matched by an ON sub-rule, where the corresponding left-table lines are missing; and full-out links save both lines that cannot be matched by an ON sub-rule and the corresponding rows are filled with missing values in the corresponding one.</p>
<p>Since the link can be linked by exchanging the position of the left and right watch, there is no difference in the essence between the left and right links. Whatever you want. It's easy to see that the whole outer link is the result of the left and right connections.</p>
<p>The three external syntaxes are:</p>
<pre><code class="language-sql">-- 左连结
FROM &lt;tb_1&gt; LEFT  OUTER JOIN &lt;tb_2&gt; ON &lt;condition(s)&gt;
-- 右连结
FROM &lt;tb_1&gt; RIGHT OUTER JOIN &lt;tb_2&gt; ON &lt;condition(s)&gt;
-- 全外连结
FROM &lt;tb_1&gt; FULL  OUTER JOIN &lt;tb_2&gt; ON &lt;condition(s)&gt;
</code></pre>
<p><strong>Except for the difference between de-linking and interlinking, the connection is not too different from the interlinking, so we can just choose precisely according to the changing circumstances.</strong></p>
<h3>Multi-table links</h3>
<p>The link usually involves only 2 tables, but sometimes there are instances where more than 3 tables must be linked simultaneously, and in principle the number of such tables is not limited. Of course, we should distinguish the concept of the master table at this time.<code>FROM</code>statement.</p>
<p>Give us a simple example of how to write multiple forms.</p>
<pre><code class="language-sql">SELECT SP.shop_id
       ,SP.shop_name
       ,SP.product_id
       ,P.product_name
       ,P.sale_price
       ,IP.inventory_quantity
  FROM shopproduct AS SP
 INNER JOIN product AS P
    ON SP.product_id = P.product_id
 INNER JOIN Inventoryproduct AS IP
    ON SP.product_id = IP.product_id
 WHERE IP.inventory_id = &#39;P001&#39;;
</code></pre>
<p>Even if you want to increase the number of linked tables to four, five...the same way you add them using INNER JOIN.</p>
<h3>ON sub-speech step - non-equivalent link</h3>
<p>At the beginning of the introduction of the link, it was mentioned that, in addition to using the equivalent of the judgement, a comparative operator could be used to connect. In fact, including comparative operators (%)&lt;、&lt;=、&gt;、&gt;=, BETWEEN, and all logical calculations (LIKE, IN, NOT, etc.) can be placed in the ON sub-word as a condition for connection.</p>
<p>Like what?</p>
<pre><code class="language-sql">SELECT  product_id
       ,product_name
       ,sale_price
       ,COUNT(p2_id) AS my_rank
  FROM (--使用自左连结对每种商品找出价格不低于它的商品
        SELECT P1.product_id
               ,P1.product_name
               ,P1.sale_price
               ,P2.product_id AS P2_id
               ,P2.product_name AS P2_name
               ,P2.sale_price AS P2_price 
          FROM product AS P1
          LEFT OUTER JOIN product AS P2 
            ON P1.sale_price &lt;= P2.sale_price 
        ) AS X
 GROUP BY product_id, product_name, sale_price
 ORDER BY my_rank; 
</code></pre>
<p><strong>Understand that the non-equivalent connection is not a line addition, but a condition for those who fulfil our conditions for association and are selected in the SELECT statement to expand the information in the column</strong></p>
<h3>Cross-link - COSS JOIN</h3>
<p>The first condition of a connection, whether external or external, is a condition of connection -- the "ON clause" -- which specifies the condition of the connection.</p>
<p>If you try not to use this connection, you may have found that there are many lines of action. The link removes the "on" clause, which is called "Cross JOIN."</p>
<p>Cross-links create a lot of rows and columns that combine each row of the left and right tables, which often leads to many meaningless lines appearing in search results. Of course, there is some utility in cross-linking some of the queries.</p>
<p>The syntax of the interconnection takes several forms:</p>
<pre><code class="language-sql">-- 1.使用关键字 CROSS JOIN 显式地进行交叉连结
SELECT SP.shop_id
       ,SP.shop_name
       ,SP.product_id
       ,P.product_name
       ,P.sale_price
  FROM shopproduct AS SP
 CROSS JOIN product AS P;
--2.使用逗号分隔两个表,并省略 ON 子句
SELECT SP.shop_id
       ,SP.shop_name
       ,SP.product_id
       ,P.product_name
       ,P.sale_price
  FROM shopproduct AS SP , product AS P;
</code></pre>
<p>Cross-linking is not applied to actual operations for two reasons. The first is that the results are of no practical value and the second is that they are too numerous and require support from a large amount of computing time and high performance equipment.</p>
<h2>Advanced processing</h2>
<h3>Window Functions</h3>
<h4>The concept of the function and the underlying method of use</h4>
<p>The function of the window is also known as<strong>ORAP function</strong>I'm sorry. ORAP, yes. <code>OnLine AnalyticalProcessing</code> The abbreviations are for real-time analysis of database data processing.</p>
<p>window function:</p>
<pre><code class="language-sql">&lt;窗口函数&gt; OVER ([ PARTITION BY &lt;列名&gt; ]
                     [ ORDER BY &lt;排序用列名&gt; ])  
</code></pre>
<p>[ ] The content in [ ] could be omitted.</p>
<p><strong>PARITITON BY Sub-Parat</strong> Optional parameters, which indicate how to group query lines into groups, similar to GRUP BY sub-rules, but the PARTITON BY sub-rules do not have the grouping function for GRUP BY sub-rules and do not change the number of lines recorded in the original table.</p>
<p><strong>ORDER BY Sub-word</strong> Optional parameters, indicating how to sort rows in each partition, i.e., to determine which rules (fields) are to be sorted in the window.</p>
<p>Although... <strong>PARITITON BY Sub-Parat</strong> and <strong>ORDER BY Sub-word</strong> Both are optional parameters, but both cannot be absent at the same time (at least two or two). No, I'm not. <code>&lt;窗口函数&gt; OVER( )</code> This usage is not meaningful (windows consist of all query lines, and window functions use all rows to calculate results).</p>
<p>window function, as in the end of the SELECT statement, you can specify the order of ascending/downs by key word ASC/DECC. Default to default the keyword by ASC</p>
<p>In principle, the window function can only be used in the SELECT sub-rule.</p>
<p>Window function OVER does not affect the sorting of the final result. It is only used to determine the order in which the function is calculated.</p>
<h4>Window Function Type</h4>
<p>In general, the functions of the window can be divided into two categories.</p>
<ul>
<li>Use SUM, MAX, MIN, etc. in window functions</li>
<li>Specialized window functions for RANK, DENSE RANK etc.</li>
</ul>
<p><strong>Special Window Functions</strong></p>
<p><strong>RANK function</strong></p>
<p>When you are calculating the sorting, the subsequent place is skipped if you have a record of the same place.</p>
<p>3 entries at 1st place: 1 bit, 1 bit, 4 bit...</p>
<p><strong>DENSE RANK function</strong></p>
<p>Also calculates the order of the order, even if a record of the same place is in place, it does not skip the subsequent place.</p>
<p>Example: 3 entries at 1st place: 1 bit, 1 bit, 2 bit...</p>
<p><strong>ROW NUMBER function</strong></p>
<p>Gives only consecutive places.</p>
<p>3 entries at 1st place: 1 bit, 2 bit, 3 bit, 4 bit</p>
<p>One example of a code is</p>
<pre><code class="language-sql">SELECT  product_name
       ,product_type
       ,sale_price
       ,RANK() OVER (ORDER BY sale_price) AS ranking
       ,DENSE_RANK() OVER (ORDER BY sale_price) AS dense_ranking
       ,ROW_NUMBER() OVER (ORDER BY sale_price) AS row_num
  FROM product;  
</code></pre>
<p><strong>Use of the polymer function on the window function</strong></p>
<p>The polymer function is used in the window function by the same method as the previous dedicated window function, except that the result is one<strong>Cumulative</strong>. The syntax function value.</p>
<p>Gives an example of a code as</p>
<pre><code class="language-sql">SELECT  product_id
       ,product_name
       ,sale_price
       ,SUM(sale_price) OVER (ORDER BY product_id) AS current_sum
       ,AVG(sale_price) OVER (ORDER BY product_id) AS current_avg  
  FROM product;  
</code></pre>
<p>The result of the polymer function is, in order of our order, this is the protocol id,<strong>Current and All Previous Lines</strong>. is the aggregate of the current line.</p>
<h4>Move Average</h4>
<p>As mentioned above, the polymer function calculates the aggregation of all data accumulated in the current row when the window function is used. Actually, you could have specified more details.<strong>Summary scope</strong>I'm sorry. The aggregation is described as <strong>Frame</strong> (frame)。</p>
<p>Syntax:</p>
<pre><code class="language-sql">&lt;窗口函数&gt; OVER (ORDER BY &lt;排序用列名&gt;
                 ROWS n PRECEDING )  

&lt;窗口函数&gt; OVER (ORDER BY &lt;排序用列名&gt;
                 ROWS BETWEEN n PRECEDING AND n FOLLOWING)
</code></pre>
<p>PRECEDING (“Previous”), assigns the frame to "n rows before the deadline" and adds its own line
FOLLOWING (“after”), assigns the frame to "n rows after the deadline" plus its own line</p>
<p>Example:</p>
<p>BETWEEN 1 PRECEDING AND 1 FOLLOWING, specify the frame as "Preceive 1 line" + "Post 1 line" + "Beyond itself"</p>
<h3>GRUPING Operator</h3>
<p>The regular GRUP BY receives only a subtotal for each classification, and sometimes the sum of the classification is calculated and can be used as ROLLUP keywords.</p>
<p>Syntax:</p>
<pre><code class="language-sql">SELECT  product_type
       ,regist_date
       ,SUM(sale_price) AS sum_price
  FROM product
 GROUP BY product_type, regist_date WITH ROLLUP;  
</code></pre>
<h2>Title</h2>
<h3>Continuous login user tags</h3>
<p>Original table structure is the user login information, created by</p>
<pre><code class="language-sql">CREATE TABLE login_records
(user_id INT,
 login_date DATE);
</code></pre>
<p>The information created for the login may relate to the exact time of the day, but we will last only consider the day of the login;</p>
<p>Marks the user (user id) who logs in for at least three consecutive days and generates the following fields for these users:</p>
<ul>
<li>Userid</li>
<li>Start of continuous login</li>
<li>End of consecutive login</li>
<li>Number of consecutive login days</li>
</ul>
<p>Here is an example of an AI-generated code.</p>
<pre><code class="language-sql">WITH ranked_logins AS (
  SELECT
    user_id,
    DATE(login_date) AS login_date,
    ROW_NUMBER() OVER(PARTITION BY user_id ORDER BY DATE(login_date)) AS rn
  FROM login_records
  GROUP BY user_id, DATE(login_date)
),
grouped_logins AS (
  SELECT
    user_id,
    login_date,
    DATE_SUB(login_date, INTERVAL rn DAY) AS base_dt
  FROM ranked_logins
),
consecutive_streaks AS (
  SELECT
    user_id,
    base_dt,
    MIN(login_date) AS start_date,  -- 连续登录的第一天
    MAX(login_date) AS end_date,    -- 连续登录的最后一天
    COUNT(1) AS days
  FROM grouped_logins
  GROUP BY user_id, base_dt
  HAVING COUNT(1) &gt;= 3
)
SELECT
  user_id,
  start_date,                   -- 直接使用第一天，无需+1
  end_date,                     -- 直接使用最后一天，无需+1
  days
FROM consecutive_streaks
ORDER BY user_id, start_date;
</code></pre>
<p>This is an example of what can be achieved through multilayer queries.</p>
<pre><code class="language-sql">SELECT
  user_id,
  base_dt,
  COUNT(1)
FROM (
  SELECT
    *,
    DATE_SUB(dt, INTERVAL rn DAY) AS base_dt
  FROM (
    SELECT
      *,
      ROW_NUMBER() OVER(PARTITION BY a.user_id ORDER BY a.dt) AS rn
    FROM (
      SELECT
        user_id,
        DATE(dt) AS dt  -- 使用 MySQL 的 DATE() 函数
      FROM log_table
      GROUP BY
        user_id,
        DATE(dt)  -- 分组依据同步修改为 DATE(dt)
    ) a
  ) b
) c
GROUP BY user_id, base_dt
HAVING COUNT(1) &gt;= 3;
</code></pre>
