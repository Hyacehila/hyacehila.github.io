---
title: "SQL 基础学习笔记"
title_en: "SQL Basics Study Notes"
date: 2024-07-29 22:29:30 +0800
categories: ["Programming", "Data & Databases"]
tags: ["Learning Notes", "SQL"]
author: Hyacehila
excerpt: "一篇 SQL 基础学习笔记，覆盖数据库与 SQL 语句基础、创建和更新、查询排序、聚合分组、视图、子查询、函数、谓词、CASE、集合运算、JOIN 和窗口函数。"
excerpt_en: "A study note on SQL basics, covering database concepts, SQL statements, table creation and updates, querying, aggregation, views, subqueries, functions, predicates, CASE, set operations, JOINs, and window functions."
mathjax: false
hidden: true
permalink: '/blog/2024/07/29/sql-learning-notes/'
---

在诸多领域中 SQL 应用广泛，数据分析、开发、测试、维护、产品经理等都有可能会用到SQL。考虑到易用性和普及度，课程内容采用`MySql` 数据库进行介绍。

相关数据库的环境搭建这里不进行记录，我们从认识数据库开始，介绍基本的查询与排序，然后研究更加复杂的运算和高级的处理。最后研究一部分题目。

## 初识数据库
### 数据库的介绍
数据库是将大量数据保存起来，通过计算机加工而成的可以进行高效访问的数据集合。该数据集合称为数据库（Database，DB）。用来管理数据库的计算机系统称为数据库管理系统（Database Management System，DBMS）。

DBMS 主要通过数据的保存格式（数据库的种类）来进行分类，现阶段主要有以下 5 种类型.

- 层次数据库（Hierarchical Database，HDB）
- 关系数据库（Relational Database，RDB）
    - Oracle Database：甲骨文公司的RDBMS
    - SQL Server：微软公司的RDBMS
    - DB2：IBM公司的RDBMS
    - PostgreSQL：开源的RDBMS
    - MySQL：开源的RDBMS
- 面向对象数据库（Object Oriented Database，OODB）
- XML数据库（XML Database，XMLDB）
- 键值存储系统（Key-Value Store，KVS），举例：MongoDB

其中最为常用的是关系数据库RDB，其特点是由行和列组成的二维表来管理数据，这种类型的 DBMS 称为关系数据库管理系统（Relational Database Management System，RDBMS）。这也是我们在后面将要研究的

对于RDBMS 最为基础的系统结构是服务端与客户端分离的结构，我们在客户端使用SQL向服务端请求数据

### SQL语句基础
数据库中存储的**表结构**类似于excel中的行和列，在数据库中，行称为**记录**，它相当于一条记录，列称为**字段**，它代表了表中存储的数据项目。行和列交汇的地方称为单元格，一个单元格中只能输入一条记录。

SQL是为操作数据库而开发的语言。国际标准化组织（ISO）为 SQL 制定了相应的标准，以此为基准的SQL 称为标准 SQL。我们仅仅介绍标准 SQL，对于那些被公司进行修改的SQL，智能根据后续的需要进行学习


根据对 RDBMS 赋予的指令种类的不同，SQL 语句可以分为以下三类.
- **DDL** ：DDL（Data Definition Language，数据定义语言） 用来创建或者删除存储数据用的数据库以及数据库中的表等对象。DDL 包含以下几种指令。
    - CREATE ： 创建数据库和表等对象
    - DROP ： 删除数据库和表等对象
    - ALTER ： 修改数据库和表等对象的结构
- **DML** :DML（Data Manipulation Language，数据操纵语言） 用来查询或者变更表中的记录。DML 包含以下几种指令。
    - SELECT ：查询表中的数据
    - INSERT ：向表中插入新数据
    - UPDATE ：更新表中的数据
    - DELETE ：删除表中的数据
- **DCL** ：DCL（Data Control Language，数据控制语言） 用来确认或者取消对数据库中的数据进行的变更。除此之外，还可以对 RDBMS 的用户是否有权限操作数据库中的对象（数据库表等）进行设定。DCL 包含以下几种指令。
    - COMMIT ： 确认对数据库中的数据进行的变更
    - ROLLBACK ： 取消对数据库中的数据进行的变更
    - GRANT ： 赋予用户操作权限
    - REVOKE ： 取消用户的操作权限

**实际使用的 SQL 语句当中有 90% 属于 DML**

### SQL的基本书写规则
#### 强制语法规则
* SQL语句要以分号（ ; ）结尾
* SQL 不区分关键字的大小写，但是插入到表中的数据是区分大小写的
* win 系统默认不区分表名及字段名的大小写
* linux / mac 默认严格区分表名及字段名的大小写
* 常数的书写方式是固定的 如'abc', 1234, '26 Jan 2010', '10/01/26', '2010-01-26'......
* 单词需要用半角空格或者换行来分隔

SQL 语句的单词之间需使用半角空格或换行符来进行分隔，且不能使用全角空格作为单词的分隔符，否则会发生错误，出现无法预期的结果。

#### 书写建议
**SQL语法规范总得的原则是，清楚、易读并且层次清晰。**实际场景中常常动辄几百上千行的SQL语句，如果不写清楚，事后review或者别人接手的时候，会让人怀疑人生。

**常见注意事项如下：**

1. MySQL本身不区分大小写，但强烈要求关键字大写，表名、列名用小写；
2. 创建表时，使用统一的、描述性强的字段命名规则保证字段名是独一无二且不是保留字的，不要使用连续的下划线，不用下划线结尾；最好以字母开头
3. 关键字右对齐，且不同层级的用空格或缩进控制，使其区分开，见样例二；
4. 列名少的时候写在一行里无伤大雅；多的时候以及涉及到CASE WHEN 或者聚合计算的时候，建议分行写；个人习惯是逗号在列名前面，方便之后删除某些列，放列名后亦可；
5. 表别名和列别名尽量用有具体含义的词组，不要用a b c，不然以后review的时候会非常痛苦；
6. 运算符前后都加一个空格；
7. 当用到多个表时，请在所有列名前写上引用的表别名，不要嫌麻烦；
8. 每条命令用分号结尾；
9. 养成随手写注释的习惯，注释方法：
```plain
单行注释 #注释文字 （MySQL专属）
单行注释 -- 注释文字
多行注释：/* 注释文字 */
```

### 创建与其中的语法知识
#### 创建 CREATE 语句
创建数据库的语法规则为
```sql
CREATE DATABASE < 数据库名称 > ;
```

创建表的语法规则为
```sql
CREATE TABLE < 表名 >
( < 列名 1> < 数据类型 > < 该列所需约束 > ,
  < 列名 2> < 数据类型 > < 该列所需约束 > ,
  < 列名 3> < 数据类型 > < 该列所需约束 > ,
  < 列名 4> < 数据类型 > < 该列所需约束 > ,
  .
  .
  .
  < 该表的约束 1> , < 该表的约束 2> ,……);
```

其中的 `< >` 仅仅是为了强调这里是一个需要自行输入的语句 实际的代码中不包含

#### 命名规则
* 只能使用**半角英文字母、数字、下划线** 作为**数据库、表和列**的名称
* 名称必须以半角英文字母开头
* 习惯上，我们不应该用下划线结尾

#### 数据类型
数据库创建的表，所有的列都必须指定数据类型，每一列都不能存储与该列数据类型不符的数据。

四种最基本的数据类型
* INTEGER 型：用来指定存储整数的列的数据类型（数字型），不能存储小数。
* CHAR 型：用来存储定长字符串，当列中存储的字符串长度达不到最大长度的时候，使用半角空格进行补足，由于会浪费存储空间，所以一般不使用。**要设置长度**
* VARCHAR 型：用来存储可变长度字符串，定长字符串在字符数未达到最大长度时会用半角空格补足，但可变长字符串不同，即使字符数未达到最大长度，也不会用半角空格补足。**要设置最大长度**
* DATE 型：用来指定存储日期（年月日）的列的数据类型（日期型）。

在实际的使用中，我们最常用的是VARCHAR 型
#### 约束设置
约束是除了数据类型之外，对列中存储的数据进行限制或者追加条件的功能。

`NOT NULL`是非空约束，即该列必须输入数据。

`PRIMARY KEY`是主键约束，代表该列是唯一值，可以通过该列取出特定的行的数据。这一句应该作为表的约束使用 语法为 `PRIMARY KEY (product_id)`

其中主键约束是后续用来访问中很重要的一个环节

### 删除和更新
#### 对数据库结构进行更改
删除表的 DROP TABLE 语句
```sql
DROP TABLE < 表名 > ;
```

需要特别注意的是，删除的表是无法恢复的，只能重新插入，请执行删除操作时要特别谨慎。

添加列的 ALTER TABLE 语句
```sql
ALTER TABLE < 表名 > ADD COLUMN < 列的定义 >;
```
其中列的定义需要同时包含列名和列的数据类型以及约束

删除列的 ALTER TABLE 语句
```sql
ALTER TABLE < 表名 > DROP COLUMN < 列名 >;
```

#### 对数据库内容的修改
删除表中特定的行
```sql
-- 一定注意添加 WHERE 条件，否则将会删除所有的数据
DELETE FROM < 表名 > WHERE COLUMN_NAME='XXX';
```
**WHERE语句是用来选定行的 后面会经常继续遇到**

数据的更新
```sql
UPDATE <表名>
   SET <列名> = <表达式> [, <列名2>=<表达式2>...]  
 WHERE <条件>  -- 可选，非常重要
 ORDER BY 子句  --可选
 LIMIT 子句; --可选
```
**where 条件选定对哪一行进行操作，否则将会将所有的行按照语句修改**

**UPDATE语句中的SET是决定了更新的进行的核心语句**

多列更新：UPDATE 语句的 SET 子句支持同时将多个列作为更新对象。例子如下
```sql
-- 基础写法，一条UPDATE语句只更新一列
UPDATE product
   SET sale_price = sale_price * 10
 WHERE product_type = '厨房用具';
UPDATE product
   SET purchase_price = purchase_price / 2
 WHERE product_type = '厨房用具';  
```
该写法可以得到正确结果，但是代码较为繁琐。可以采用合并的方法来简化代码。

```sql
-- 合并后的写法
UPDATE product
   SET sale_price = sale_price * 10,
       purchase_price = purchase_price / 2
 WHERE product_type = '厨房用具';  
```
需要明确的是，SET 子句中的列不仅可以是两列，还可以是三列或者更多。一行用来更新一列

### 插入
#### 本节的数据库背景
为了学习  `INSERT` 语句用法，我们首先创建一个名为  productins 的表，建表语句如下：
```sql
CREATE TABLE productins
(product_id    CHAR(4)      NOT NULL,
product_name   VARCHAR(100) NOT NULL,
product_type   VARCHAR(32)  NOT NULL,
sale_price     INTEGER      DEFAULT 0,
purchase_price INTEGER ,
regist_date    DATE ,
PRIMARY KEY (product_id)); 
```
*这里的`sale_price`列被设置了默认值约束，插入NULL的时候会被替换为默认*
#### 插入INSERT语句
`INSERT` 语句 插入一行 基本语法：
```sql
INSERT INTO <表名> (列1, 列2, 列3, ……) VALUES (值1, 值2, 值3, ……);  
```
对表进行全列 INSERT 时，可以省略表名后的列清单。这时 VALUES子句的值会默认按照从左到右的顺序赋给每一列。

一个插入的例子为
```sql
-- 包含列清单
INSERT INTO productins (product_id, product_name, product_type, sale_price, purchase_price, regist_date) VALUES ('0005', '高压锅', '厨房用具', 6800, 5000, '2009-01-15');
-- 省略列清单
INSERT INTO productins VALUES ('0005', '高压锅', '厨房用具', 6800, 5000, '2009-01-15');  
```

原则上，执行一次 INSERT 语句会插入一行数据。插入多行时，通常需要循环执行相应次数的 INSERT 语句。

INSERT 语句中想给某一列赋予 NULL 值时，可以直接在 VALUES子句的值清单中写入 NULL。想要插入 NULL 的列一定不能设置 NOT NULL 约束。

#### 从其他表中复制插入
可以使用INSERT … SELECT 语句从其他表复制数据。
```sql
-- 将商品表中的数据复制到商品复制表中
INSERT INTO productcopy (product_id, product_name, product_type, sale_price, purchase_price, regist_date)
SELECT product_id, product_name, product_type, sale_price, purchase_price, regist_date
  FROM Product;  
```

### 索引
MySQL索引的建立对于MySQL的高效运行是很重要的，索引可以大大提高MySQL的检索速度。 实际上，**索引是为了我们后面快捷的进行查询**

创建表时可以直接创建索引，语法如下：
```sql
CREATE TABLE mytable(  
 
ID INT NOT NULL,   
 
username VARCHAR(16) NOT NULL,  
 
INDEX [indexName] (username(length))  
 
);  
```

也可以使用如下语句创建：
```sql
-- 方法1
CREATE INDEX indexName ON table_name (column_name)

-- 方法2
ALTER table tableName ADD INDEX indexName(columnName)
```

**索引的比较复杂的，我们后面再逐渐的进行介绍**

## 基础查询与排序
### SELECT语句基础
#### 基本查询
从表中选取数据时需要使用SELECT语句，也就是只从表中选出（SELECT）必要数据的意思。通过SELECT语句查询并选取出必要数据的过程称为匹配查询或查询（query）。

基础的SELECT语句包含了SELECT和FROM两个子句
```sql
SELECT <列名>, 
  FROM <表名>;
```
其中，SELECT子句中列举了希望从表中查询出的列的名称，而FROM子句则指定了选取出数据的表的名称。想要选取多个列也是可以接受的，只需要注意，**字句之间不需要逗号**，它是分割符号

#### 选择性的查询
当不需要取出全部数据，而是选取出满足“商品种类为衣服”“销售单价在1000日元以上”等某些条件的数据时，使用WHERE语句。
在**WHERE 子句中可以指定“某一列的值和这个字符串相等”或者“某一列的值大于这个数字”等条件。**执行含有这些条件的SELECT语句，就可以查询出只符合该条件的记录了。
```sql
SELECT <列名>, ……
  FROM <表名>
 WHERE <条件表达式>;
```

#### 一些补充知识
* 星号 代表全部列的意思。
* SQL中可以随意使用换行符，不影响语句执行
* 设定汉语别名时需要使用双引号（"）括起来。
* 在SELECT语句中使用DISTINCT可以删除重复行。
```sql
-- 想要查询出全部列时，可以使用代表所有列的星号（*）。
SELECT *
  FROM <表名>；
-- SQL语句可以使用AS关键字为列设定别名（用中文时需要双引号（“”））。
SELECT product_id     As id,
       product_name   As name,
       purchase_price AS "进货单价"
  FROM product;
-- 使用DISTINCT删除product_type列中重复的数据
SELECT DISTINCT product_type
  FROM product;
```

### 算术运算符和比较运算符
#### 算数运算符
SQL语句中可以使用的四则运算的主要运算符如下：

|含义|运算符|
|:----|:----|
|加法|+|
|减法|-|
|乘法|*|
|除法|/|

他们只针对INTEGER进行运算
#### 比较运算符

| 运算符 | 含义   |
| --- | ---- |
| =   | 相等   |
| <>  | 不相等  |
| >=  | 大于等于 |
| <=  | 小于等于 |
| >   | 大于   |
| <   | 小于   |

他们只针对INTEGER进行运算
#### 常用法则
* SELECT子句中可以使用常数或者表达式。
* 字符串类型的数据原则上按照字典顺序进行排序，不能与数字的大小顺序混淆。如果可以尽量避免这样使用
* 希望选取NULL记录时，需要在条件表达式中使用IS NULL运算符。希望选取不是NULL的记录时，需要在条件表达式中使用IS NOT NULL运算符。

```sql
-- SQL语句中也可以使用运算表达式
SELECT product_name, sale_price, sale_price * 2 AS "sale_price x2"
  FROM product;
-- WHERE子句的条件表达式中也可以使用计算表达式
SELECT product_name, sale_price, purchase_price
  FROM product
 WHERE sale_price - purchase_price >= 500;
/* 对字符串使用不等号
首先创建chars并插入数据
选取出大于‘2’的SELECT语句*/
-- DDL：创建表
CREATE TABLE chars
（chr CHAR（3）NOT NULL, 
PRIMARY KEY（chr））;
-- 选取出大于'2'的数据的SELECT语句('2'为字符串)
SELECT chr
  FROM chars
 WHERE chr > '2';
-- 选取NULL的记录
SELECT product_name, purchase_price
  FROM product
 WHERE purchase_price IS NULL;
-- 选取不为NULL的记录
SELECT product_name, purchase_price
  FROM product
 WHERE purchase_price IS NOT NULL;
```

### 逻辑运算符
逻辑运算符帮助我们继续复杂查询的WHERE语句中的条件，给予我们更高的自由度

#### NOT运算符
想要表示 `不是……` 时，除了前文的<>运算符外，还存在另外一个表示否定、使用范围更广的运算符：NOT。

NOT不能单独使用，必须和其他查询条件组合起来使用，比如
```sql
SELECT product_name, product_type, sale_price
  FROM product
 WHERE NOT sale_price >= 1000;
```

要注意 如果并非一定需要 直接使用比较运算符显然更加符合我们的直观。

#### AND 与 OR 运算符
当希望同时使用多个查询条件时，可以使用AND或者OR运算符。

AND 相当于“并且”，类似数学中的取交集；
OR 相当于“或者”，类似数学中的取并集。

要注意，在运算符逐渐复杂的后，我们需要善用括号和缩进表达各种运算的优先级，避免因为逻辑运算的优先级导致结果错误，如下
```sql
-- 通过使用括号让OR运算符先于AND运算符执行
SELECT product_name, product_type, regist_date
  FROM product
 WHERE product_type = '办公用品'
   AND ( regist_date = '2009-09-11'
        OR regist_date = '2009-09-20');
```

很明显，我们在 `regist_date` 列有着明显的二选一 然后他和我们的 `product_type`列条件并列 这种使用务必使用OR运算符

### 聚合查询
SQL中用于汇总的函数叫做聚合函数。以下五个是最常用的聚合函数：
- SUM：计算表中某数值列中的合计值
- AVG：计算表中某数值列中的平均值
- MAX：计算表中任意列中数据的最大值，包括文本类型和数字类型
- MIN：计算表中任意列中数据的最小值，包括文本类型和数字类型
- COUNT：计算表中的记录条数（行数）

他们的语法规则相较于基本的SELECT又有了一些变化，当我们只想进行一些很简单的分析的时候聚合查询会派上一些用处


```sql
-- 计算销售单价和进货单价的合计值
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
```

有时候，多个行可能在同一列有着完全一样的数据，我们可能想要删除重复数据。
```sql
SELECT COUNT(DISTINCT product_type)
  FROM product;
```


- COUNT 聚合函数运算结果与参数有关，COUNT(星号) / COUNT(1) 得到包含 NULL 值的所有行，COUNT(<列名>) 得到不包含 NULL 值的所有行。
- 聚合函数不处理包含 NULL 值的行，但是 COUNT除外。
- MAX / MIN 函数适用于文本类型和数字类型的列，而 SUM / AVG 函数仅适用于数字类型的列。
- 在聚合函数的参数中使用 DISTINCT 关键字，可以得到删除重复值的聚合结果。

### 分组统计
之前使用聚合函数都是会将整个表的数据进行处理，当你想将进行分组汇总时（即：将现有的数据按照某列来汇总统计），GROUP BY可以帮助我们。

语法结构为
```sql
-- 在SELECT中使用的全部非聚合列，都应该出现在GROUP BY中，否则逻辑出错
SELECT <列名1>,<列名2>, <列名3>, ……
  FROM <表名>
 GROUP BY <列名1>, <列名2>, <列名3>, ……;
```

一个很基础的例子为
```sql
-- 按照商品种类统计数据行数
SELECT product_type, COUNT(*)
FROM product
WHERE product_type IN ('衣服', '鞋类')
GROUP BY product_type;
 -- 不含GROUP BY 此代码非法，因为混合使用了聚合列以及普通列且无分组
SELECT product_type, COUNT(*)
FROM product
WHERE product_type IN ('衣服', '鞋类')
```
此时我们选出的语句根据 `GROUP BY` 中指定的列进行分类汇总

NULL作为一组特殊数据进行聚合运算

当混合使用普通列和聚合函数时，**必须使用 `GROUP BY` 来指定如何分组数据** ，否则数据库不知道如何处理多行数据对应一个聚合值的情况。
### 为分组结果指定条件
`GROUP BY` 帮助我们实现了分组，那么如何取出其中特殊的组呢？ 这就是我们这里要研究的。

可以在 GROUP BY 后使用 HAVING 子句。HAVING 的用法类似 WHERE。

值得注意的是：**HAVING 子句必须与 GROUP BY 子句配合使用，且限定的是分组聚合结果，即这里使用的键需要包括在SELECT中**，WHERE 子句是限定数据行（包括分组列），二者各司其职，其中使用的键无需被SELECT。

HAVING的执行逻辑相当靠后，仅仅在ORDER BY之前

例子如下
```sql
-- 只要行数大于2的分组
SELECT product_type, COUNT(*)
  FROM product
 GROUP BY product_type
HAVING COUNT(*) = 2;

-- 错误形式（因为product_name不包含在GROUP BY聚合键中）
SELECT product_type, COUNT(*)
  FROM product
 GROUP BY product_type
HAVING product_name = '圆珠笔';
```

### 对查询结果排序
在某些场景下，需要得到一个排序之后的结果。而 SQL 语句执行结果默认随机排列，想要按照顺序排序，需使用 **ORDER BY** 子句。语法规则为
```sql
SELECT <列名1>, <列名2>, <列名3>, ……
  FROM <表名>
 ORDER BY <排序基准列1> [ASC, DESC], <排序基准列2> [ASC, DESC], ……
```
其中，参数 ASC 表示升序排列，DESC 表示降序排列，默认为升序，此时，参数 ASC 可以缺省。

例子为
```sql
-- 降序排列
SELECT product_id, product_name, sale_price, purchase_price
  FROM product
 ORDER BY sale_price DESC;

-- 多个排序键
SELECT product_id, product_name, sale_price, purchase_price
  FROM product
 ORDER BY sale_price DESC, product_id DESC;
```

NULL无法进行比较，因此排序的随机把他们放在开头或者结尾。在MySQL中，`NULL` 值被认为比任何 `非NULL` 值低，因此，当顺序为 ASC（升序）时，`NULL` 值出现在第一位，而当顺序为 DESC（降序）时，则排序在最后。

### 关于语句的执行顺序
GROUP BY中提到，GROUP BY 子句中不能使用SELECT 子句中定义的别名，但是在 ORDER BY 子句中却可以使用别名。

这是因为 SQL 在使用 HAVING 子句时 SELECT 语句的执行顺序为：
FROM → WHERE → GROUP BY → SELECT → HAVING → ORDER BY

其中 SELECT 的执行顺序在 GROUP BY 子句之后，ORDER BY 子句之前。因此在 ORDER BY 子句中可以使用别名，但是在GROUP BY中不能使用别名。
## 复杂的查询
### 视图
#### 视图的基本知识
视图是一个虚拟的表，不同于直接操作数据表，视图是依据SELECT语句来创建的（会在下面具体介绍），所以操作视图时会根据创建视图的SELECT语句生成一张虚拟表，然后在这张虚拟表上做SQL操作。

**视图与表的区别---“是否保存了实际的数据”** 所以视图并不是数据库真实存储的数据表，它可以看作是一个窗口，通过这个窗口我们可以看到数据库表中真实存在的数据。

那既然已经有数据表了，为什么还需要视图呢？主要有以下几点原因：
1. 通过定义视图可以**将频繁使用的SELECT语句保存**以提高效率。
2. 通过定义视图可以使用户看到的数据更加清晰。
3. 通过定义视图可以不对外公开数据表全部字段，增强数据的保密性。
4. 通过定义视图可以降低数据的冗余。

#### 创建视图
创建视图的基本语法如下：

```sql
CREATE VIEW <视图名称>(<列名1>,<列名2>,...) AS <SELECT语句>
```
其中SELECT 语句需要书写在 AS 关键字之后。 SELECT 语句中列的排列顺序和视图中列的排列顺序相同， SELECT 语句中的第 1 列就是视图中的第 1 列， SELECT 语句中的第 2 列就是视图中的第 2 列，以此类推。而且视图的列名是在视图名称之后的列表中定义的。

需要注意的是视图名在数据库中需要是唯一的，不能与其他视图和表重名。

视图不仅可以基于真实表，我们也可以在视图的基础上继续创建视图。但是多重视图会降低 SQL 的性能。

在视图创建的过程中不能使用 `SELECT` 的 `ORDER BY` 子句 因为他创建的视图中的行不应该有顺序 *在 MySQL中视图的定义是允许使用 ORDER BY 语句的，但是尽量规避*

基于单表创建一个视图例子如下
```sql
CREATE VIEW productsum (product_type, cnt_product)
AS
SELECT product_type, COUNT(*)
  FROM product
 GROUP BY product_type ;
```

基于多表创建一个视图例子如下
```sql
CREATE VIEW view_shop_product(product_type, sale_price, shop_name)
AS
SELECT product_type, sale_price, shop_name
  FROM product,
       shop_product
 WHERE product.product_id = shop_product.product_id;
--这使用的查询方法是隐式内连接，是模仿INNER JOIN的语句，效果一样
```

我们创建的视图在代码上都是很好理解的，语句同时在多个表中取数据也是非常自然的想法

#### 修改视图结构
修改视图结构的语法是

```sql
ALTER VIEW <视图名> AS <SELECT语句>
```

和我们删除再创建基本上也没什么区别

删除视图的基本语法如下：

```sql
DROP VIEW <视图名1> [ , <视图名2> …]
```
#### 更新视图内容
因为视图是一个虚拟表，所以对视图的操作就是对底层基础表的操作，所以在修改时只有满足底层基本表的定义才能成功修改。

有些修改是可以进行的，但是**但是并不推荐这种使用方式。而且我们在创建视图时也尽量使用限制不允许通过视图来修改表**

当视图包含下面的语句的时候，无法被更新
* 聚合函数 SUM()、MIN()、MAX()、COUNT() 等。
* DISTINCT 关键字。
* GROUP BY 子句。
* HAVING 子句。
* UNION 或 UNION ALL 运算符。
* FROM 子句中包含多个表。

### 子查询
#### 什么是子查询
一个简单的例子
```sql
SELECT stu_name
FROM (
         SELECT stu_name, COUNT(*) AS stu_cnt
          FROM students_info
          GROUP BY stu_age) AS studentSum;
```
这个语句看起来很好理解，其中使用括号括起来的sql语句首先执行，执行成功后再执行外面的sql语句。这就是子查询的语句。（这段代码本身存在一定逻辑问题，不满足分组聚合的需求）

子查询指一个查询语句嵌套在另一个查询语句内部的查询，在 SELECT 子句中先计算子查询，子查询结果作为外层另一个查询的过滤条件，查询可以基于一个表或者多个表。

子查询就是将用来定义视图的 SELECT 语句直接用于 FROM 子句当中。其中AS studentSum可以看作是子查询的名称，而且子查询是一次性的。

**嵌套的多层子查询确实可以获得结果，但是会降低SQL语句的可读性并且导致执行效率降低，尽量避免**

#### 标量子查询
标量就是单一的意思，那么标量子查询也就是单一的子查询，所谓单一就是要求我们执行的SQL语句只能返回一个值，也就是要返回表中具体的**某一行的某一列**。

所有需要使用单一值的地方都可以用标量子查询，比如
```sql
SELECT product_id, product_name, sale_price
  FROM product
 WHERE sale_price > (SELECT AVG(sale_price) FROM product);
```

#### 关联子查询
关联子查询既然包含关联两个字，那么一定意味着查询与子查询之间存在着联系。

需求`选取出各商品种类中高于该商品种类的平均销售单价的商品`。SQL语句如下：
```sql
SELECT product_type, product_name, sale_price
  FROM product AS p1
 WHERE sale_price > (SELECT AVG(sale_price)
                       FROM product AS p2
                      WHERE p1.product_type =p2.product_type
                      GROUP BY product_type);
```

所谓的关联，就是使用一些标志将内外两层的查询连接起来起到过滤数据的目的。我们将外面的product表标记为p1，将内部的product设置为p2，而且通过WHERE语句连接了两个查询。

基本的执行思路是
1. 首先执行不带WHERE的主查询
2. **根据主查询讯结果匹配product_type，获取子查询结果**
3. 将子查询结果再与主查询结合执行完整的SQL语句

**视图和子查询是数据库操作中较为基础的内容，对于一些复杂的查询需要使用子查询加一些条件语句组合才能得到正确的结果。但是无论如何对于一个SQL语句来说都不应该设计的层数非常深且特别复杂，不仅可读性差而且执行效率也难以保证，所以尽量有简洁的语句来完成需要的功能。**

### 各种各样的函数
sql 自带了各种各样的函数，极大提高了 sql 语言的便利性。

函数大致分为如下几类：

* 算术函数    （用来进行数值计算的函数）
* 字符串函数 （用来进行字符串操作的函数）
* 日期函数     （用来进行日期操作的函数）
* 转换函数     （用来转换数据类型和值的函数）
* 聚合函数     （用来进行数据聚合的函

函数总个数超过200个，不需要完全记住，常用函数有 30~50 个，其他不常用的函数使用时查阅文档即可。

#### 算数函数
* ABS -- 绝对值
语法：`ABS( 数值 )`
ABS 函数用于计算一个数字的绝对值，表示一个数到原点的距离。当 ABS 函数的参数为`NULL`时，返回值也是`NULL`。

* MOD -- 求余数
语法：`MOD( 被除数，除数 )`
MOD 是计算除法余数（求余）的函数，是 modulo 的缩写。小数没有余数的概念，只能对整数列求余数。
注意：主流的 DBMS 都支持 MOD 函数，只有SQL Server 不支持该函数，其使用`%`符号来计算余数。

* ROUND -- 四舍五入
语法：`ROUND( 对象数值，保留小数的位数 )`
ROUND 函数用来进行四舍五入操作。
注意：当参数 **保留小数的位数** 为变量时，可能会遇到错误，请谨慎使用变量。

#### 字符串函数
* CONCAT -- 拼接
语法：`CONCAT(str1, str2, str3)`
MySQL中使用 CONCAT 函数进行拼接。

* LENGTH -- 字符串长度
语法：`LENGTH( 字符串 )`

* LOWER -- 小写转换
LOWER 函数只能针对英文字母使用，它会将参数中的字符串全都转换为小写。该函数不适用于英文字母以外的场合，不影响原本就是小写的字符。类似的， UPPER 函数用于大写转换。

* REPLACE -- 字符串的替换
语法：`REPLACE( 对象字符串，替换前的字符串，替换后的字符串 )`

* SUBSTRING -- 字符串的截取
语法：`SUBSTRING （对象字符串 FROM 截取的起始位置 FOR 截取的字符数）`
使用 SUBSTRING 函数 可以截取出字符串中的一部分字符串。截取的起始位置从字符串最左侧开始计算，索引值起始为1。

* SUBSTRING_INDEX -- 字符串按索引截取
语法：`SUBSTRING_INDEX (原始字符串， 分隔符，n)`
该函数用来获取原始字符串按照分隔符分割后，第 n 个分隔符之前（或之后）的子字符串，支持正向和反向索引，索引起始值分别为 1 和 -1。

* REPEAT -- 字符串按需重复多次
语法：`REPEAT(string, number)`

#### 日期函数
不同DBMS的日期函数语法各有不同，本课程介绍一些被标准 SQL 承认的可以应用于绝大多数 DBMS 的函数。特定DBMS的日期函数查阅文档即可。

* CURRENT_DATE -- 获取当前日期
* CURRENT_TIME -- 当前时间
* CURRENT_TIMESTAMP -- 当前日期和时间

* EXTRACT -- 截取日期元素
语法：`EXTRACT(日期元素 FROM 日期)`
使用 EXTRACT 函数可以截取出日期数据中的一部分，例如“年” “月”，或者“小时”“秒”等。该函数的返回值并不是日期类型而是数值类型

#### 转换函数
“转换”这个词的含义非常广泛，在 SQL 中主要有两层意思：一是数据类型的转换，简称为类型转换，在英语中称为`cast`；另一层意思是值的转换。

* CAST -- 类型转换
语法：`CAST（转换前的值 AS 想要转换的数据类型）`
需要特别注意的是，当要转换为整型时，需要指定为 SIGNED（有符号） 或者 UNSIGNED（无符号）

* COALESCE -- 找非NULL，方便转换
语法：`COALESCE(数据1，数据2，数据3……)`
COALESCE 是 SQL 特有的函数。该函数会返回可变参数 A 中左侧开始第 1个不是NULL的值。参数个数是可变的，因此可以根据需要无限增加。

### 谓词
谓词就是返回值为真值的函数。包括`TRUE / FALSE / UNKNOWN`。除去我们在“比较运算符”一节中介绍的，还有一些产生布尔值的函数

谓词主要有以下几个：
* LIKE
* BETWEEN
* IS NULL、IS NOT NULL
* IN
* EXISTS
#### LIKE
当需要进行字符串的部分一致查询时需要使用该谓词。

部分一致大体可以分为前方一致、中间一致和后方一致三种类型。
##### 前方一致
前方一致的语法结构为`WHERE strcol LIKE 'ddd%'`  

前方一致即作为查询条件的字符串（这里是“ddd”）与查询对象字符串起始部分相同。

其中的`%`是代表“零个或多个任意字符串”的特殊符号，本例中代表“以ddd开头的所有字符串”。

##### 中间一致
中间一致即查询对象字符串中含有作为查询条件的字符串，无论该字符串出现在对象字符串的最后还是中间都没有关系。

语法结构为`WHERE strcol LIKE '%ddd%'`

##### 后方一致
后方一致即作为查询条件的字符串（这里是“ddd”）与查询对象字符串的末尾部分相同。

语法结构为`WHERE strcol LIKE '%ddd'`

##### 任意字符
使用 下划线 来代替 %，与 % 不同的是，它代表了“任意 1 个字符”。

语法结构为`WHERE strcol LIKE 'abc__'`

#### BETWEEN谓词
使用 BETWEEN 可以进行范围查询。该谓词与其他谓词或者函数的不同之处在于它使用了 3 个参数。

我们举一个例子为
```sql
-- 选取销售单价为100～ 1000元的商品
SELECT product_name, sale_price
FROM product
WHERE sale_price BETWEEN 100 AND 1000;

```

BETWEEN 的特点就是结果中会包含 100 和 1000 这两个临界值，也就是闭区间。如果不想让结果中包含临界值，那就必须使用 < 和 >

#### IS NULL、 IS NOT NULL
为了选取出某些值为 NULL 的列的数据，不能使用 =，而只能使用特定的谓词IS NULL。

与此相反，想要选取 NULL 以外的数据时，需要使用IS NOT NULL。

之所以有这条规则是因为， SQL使用三值布尔，NULL意味着未知，和他进行任何非相关谓词的运算都产生UNKNOWN

#### IN谓词
多个查询条件取并集时可以选择使用`or`语句。

那就是随着希望选取的对象越来越多， 使用`or`语句的SQL 语句也会越来越长，阅读起来也会越来越困难。这时， 我们就可以使用IN 谓词`IN(值1, 值2, 值3, ......)来替换OR语句。

一个简单的例子为
```sql
SELECT product_name, purchase_price
FROM product
WHERE purchase_price IN (320, 500, 5000);
```

同理我们可以使用否定形式NOT IN来实现 

需要注意的是，在使用IN 和 NOT IN 时是无法选取出NULL数据的。

特别的，IN语句经常和子查询搭配使用，比如
```sql
-- 取出大阪门店在售商品的销售单价 `sale_price`
SELECT product_name, sale_price
FROM product
WHERE product_id IN (SELECT product_id
  FROM shopproduct
                       WHERE shop_id = '000C');
```

子查询提供了一个新的表让IN谓词用法有了查询的空间

#### EXIST 谓词
EXIST 谓词的用法理解起来有些难度。

① EXIST 的使用方法与之前的都不相同

② 语法理解起来比较困难

③ 实际上即使不使用 EXIST，基本上也都可以使用 IN（或者 NOT IN）来代替

这么说的话，还有学习 EXIST 谓词的必要吗？答案是肯定的，因为一旦能够熟练使用 EXIST 谓词，就能体会到它极大的便利性。

不过，你不用过于担心，本课程介绍一些基本用法，日后学习时可以多多留意 EXIST 谓词的用法，以期能够在达到 SQL 中级水平时掌握此用法。

* EXIST谓词的使用方法

谓词的作用就是 **“判断是否存在满足某种条件的记录”**。

如果存在这样的记录就返回真（TRUE），如果不存在就返回假（FALSE）。

EXIST（存在）谓词的主语是“记录”。

我们继续以 IN和子查询 中的示例，使用 EXIST 选取出大阪门店在售商品的销售单价。

```sql
SELECT product_name, sale_price
  FROM product AS p
 WHERE EXISTS (SELECT *
                 FROM shopproduct AS sp
                WHERE sp.shop_id = '000C'
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
```
* EXIST的参数

之前我们学过的谓词，基本上都是像“列 LIKE 字符串”或者“ 列 BETWEEN 值 1 AND 值 2”这样需要指定 2 个以上的参数，而 EXIST 的左侧并没有任何参数。因为 EXIST 是只有 1 个参数的谓词。 所以，EXIST 只需要在右侧书写 1 个参数，该参数通常都会是一个子查询。

```sql
(SELECT *
   FROM shopproduct AS sp
  WHERE sp.shop_id = '000C'
    AND sp.product_id = p.product_id)  
```
上面这样的子查询就是唯一的参数。确切地说，由于通过条件“SP.product_id = P.product_id”将 product 表和 shopproduct表进行了联接，因此作为参数的是关联子查询。 EXIST 通常会使用关联子查询作为参数。
* 子查询中的SELECT *

由于 EXIST 只关心记录是否存在，因此返回哪些列都没有关系。 EXIST 只会判断是否存在满足子查询中 WHERE 子句指定的条件“商店编号（shop_id）为 '000C'，商品（product）表和商店

商品（shopproduct）表中商品编号（product_id）相同”的记录，只有存在这样的记录时才返回真（TRUE）。

因此，使用下面的查询语句，查询结果也不会发生变化。

```sql
SELECT product_name, sale_price
  FROM product AS p
 WHERE EXISTS (SELECT 1 -- 这里可以书写适当的常数
                 FROM shopproduct AS sp
                WHERE sp.shop_id = '000C'
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
```
>大家可以把在 EXIST 的子查询中书写 SELECT * 当作 SQL 的一种习惯。
* 使用NOT EXIST替换NOT IN

就像 EXIST 可以用来替换 IN 一样， NOT IN 也可以用NOT EXIST来替换。

下面的代码示例取出，不在东京门店销售的商品的销售单价。

```sql
SELECT product_name, sale_price
  FROM product AS p
 WHERE NOT EXISTS (SELECT *
                     FROM shopproduct AS sp
                    WHERE sp.shop_id = '000A'
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
```
NOT EXIST 与 EXIST 相反，当“不存在”满足子查询中指定条件的记录时返回真（TRUE）。

### CASE 表达式
#### 什么是CASE表达式
CASE 表达式是函数的一种。是 SQL 中数一数二的重要功能，有必要好好学习一下。

CASE 表达式是在区分情况时使用的，这种情况的区分在编程中通常称为（条件）分支。

CASE表达式的语法分为简单CASE表达式和搜索CASE表达式两种。由于搜索CASE表达式包含简单CASE表达式的全部功能。本课程将重点介绍搜索CASE表达式。

语法结构为
```sql
CASE WHEN <求值表达式> THEN <表达式>
     WHEN <求值表达式> THEN <表达式>
     WHEN <求值表达式> THEN <表达式>
     .
     .
     .
ELSE <表达式>
END  
```

上述语句执行时，依次判断 when 表达式是否为真值，是则执行 THEN 后的语句，如果所有的 when 表达式均为假，则执行 ELSE 后的语句。
无论多么庞大的 CASE 表达式，最后也只会返回一个值。

#### CASE使用方法
**应用场景1：根据不同分支得到不同列值**
```sql
SELECT  product_name,
        CASE WHEN product_type = '衣服' THEN CONCAT('A ： ',product_type)
             WHEN product_type = '办公用品'  THEN CONCAT('B ： ',product_type)
             WHEN product_type = '厨房用具'  THEN CONCAT('C ： ',product_type)
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
```
ELSE 子句也可以省略不写，这时会被默认为 ELSE NULL。但为了防止有人漏读，还是希望大家能够显式地写出 ELSE 子句。

此外， CASE 表达式最后的“END”是不能省略的，请大家特别注意不要遗漏。忘记书写 END 会发生语法错误，这也是初学时最容易犯的错误。

**应用场景2：实现列方向上的聚合/内容不变，修改排列的形式**
行方向的聚合使用“聚合查询”一节中的方法实现，列方向上的聚合就需要依靠CASE表达式实现

例子1
```sql
-- 对按照商品种类计算出的销售单价合计值进行行列转换
SELECT SUM(CASE WHEN product_type = '衣服' THEN sale_price ELSE 0 END) AS sum_price_clothes,
       SUM(CASE WHEN product_type = '厨房用具' THEN sale_price ELSE 0 END) AS sum_price_kitchen,
       SUM(CASE WHEN product_type = '办公用品' THEN sale_price ELSE 0 END) AS sum_price_office
  FROM product;

-- 返回表结构
+-------------------+-------------------+------------------+
| sum_price_clothes | sum_price_kitchen | sum_price_office |
+-------------------+-------------------+------------------+
|              5000 |             11180 |              600 |
+-------------------+-------------------+------------------+
```

实际上，如果我们选择呈现一个列方向的传统的一行为一个观测的结构，直接使用GROUP BY分组聚合就可以，但是转换成横向就需要这种CASE表达式

例子2
```sql
-- CASE WHEN 实现数字列 score 行转列
SELECT name,
       SUM(CASE WHEN subject = '语文' THEN score ELSE null END) as chinese,
       SUM(CASE WHEN subject = '数学' THEN score ELSE null END) as math,
       SUM(CASE WHEN subject = '外语' THEN score ELSE null END) as english
  FROM score
 GROUP BY name;
+------+---------+------+---------+
| name | chinese | math | english |
+------+---------+------+---------+
| 张三 |      93 |   88 |      91 |
| 李四 |      87 |   90 |      77 |
+------+---------+------+---------+
```

原始的表结构为 name subject score 的三列结构，我们转化为了这样的一种行结构，里面的 `SUM` 这样的聚合函数仅仅是为了把CASE返回的一个列表转换为了一个数

例子3
```sql
-- CASE WHEN 实现文本列 subject 行转列
SELECT name,
       MAX(CASE WHEN subject = '语文' THEN subject ELSE null END) as chinese,
       MAX(CASE WHEN subject = '数学' THEN subject ELSE null END) as math,
       MIN(CASE WHEN subject = '外语' THEN subject ELSE null END) as english
  FROM score
 GROUP BY name;
+------+---------+------+---------+
| name | chinese | math | english |
+------+---------+------+---------+
| 张三 | 语文    | 数学 | 外语    |
| 李四 | 语文    | 数学 | 外语    |
+------+---------+------+---------+
```

这个例子没什么实际意义，仅仅是告诉我们如果想要进行文本列，可以用 `MIN，MAX`这种聚合查询函数实现

## 集合运算
### 关于集合本身
`集合`在数学领域表示“各种各样的事物的总和”，在数据库领域表示记录的集合。具体来说，表、视图和查询的执行结果都是记录的集合，其中的元素为表或者查询结果中的每一行。

在标准 SQL 中，分别对检索结果使用 `UNION`，`INTERSECT`， `EXCEPT` 来将检索结果进行并集、交集和差集运算。用来进行集合运算的运算符称为集合运算符。

在数据库中，所有的表--以及查询结果--都可以视为集合，因此也可以把表视为集合进行上述集合运算，在很多时候，这种抽象非常有助于对复杂查询问题给出一个可行的思路。
### 表的加法--UNION
我们首先给出一个例子
```sql
SELECT product_id, product_name
  FROM product
 UNION
SELECT product_id, product_name
  FROM product2;
```

能看出，我们把象征着语句结束的符号从前面语句末尾移除，用`UNION`连接他们，最后让两个集合做了并

 **UNION 等集合运算符通常都会除去重复的记录**

这种去重不仅会去掉两个结果集相互重复的，还会去掉一个结果集中的重复行。但在实践中有时候需要需要不去重的并集，在 UNION 的结果中保留重复行的语法其实非常简单，只需要在 UNION 后面添加 ALL 关键字就可以了。如下
```sql
-- 保留重复行
SELECT product_type
  FROM product
 UNION ALL
SELECT product_type
  FROM product2;
```

### 交集运算符 INTERSECT
集合的交，就是两个集合的公共部分，由于集合元素的互异性，集合的交只需通过文氏图就可以很直观地看到它的意义。

使用 `INTERSECT ` 求交集代码如下：
```sql
TABLE product INTERSECT TABLE product2;
```

使用 `INTERSECT` 运算符进行交集运算的两张表的列数必须相同，**字段类型也需要相同。**

`INTERSECT` 运算符优先级高于 `UNION` 和 `EXCEPT` ，同时出现时会优先进行交集运算

对于同一个表的两个查询结果而言，他们的交INTERSECT实际上可以等价地将两个查询的检索条件用AND谓词连接来实现。
### 差集，补集与表的减法
求集合差集的减法运算和实数的减法运算有些不同，当使用一个集合A减去另一个集合B的时候，对于只存在于集合B而不存在于集合A的元素，采取直接忽略的策略，因此集合A和B做减法只是将集合A中也同时属于集合B的元素减掉。

使用 MySQL 8.0.31 版本进行差集运算的语句如下：
```sql
TABLE product EXCEPT TABLE product2;
```

实际上，使用 NOT IN 谓词，基本上可以实现和SQL标准语法中的EXCEPT运算相同的效果。

**对于那些不提供直接的运算符的运算，我们使用谓词和常见的查询代码进行组合就可以实现了**
### 联结(JOIN)基础
我们在前面介绍了非常多中查询与处理数据的方法，但是他们都不能让我们增加列的信息
* UNION和INTERSECT 等集合运算以行方向为单位进行操作
* 函数或者 CASE表达式等列运算，可以增加列的数量，本质上并不能提供更多的信息

我们在前面的“关联子查询”一节介绍过，关联子查询允许我们同时从多个表中获取信息，不过**连结**更适合从多张表获取信息

连结(JOIN)就是使用某种关联条件(**一般是使用相等判断谓词"="**)， 将其他表中的列添加过来，进行“添加列”的集合运算。可以说，连结是 SQL 查询的核心操作，掌握了连结，能够从两张甚至多张表中获取列，能够将过去使用关联子查询等过于复杂的查询简化为更加易读的形式，以及进行一些更加复杂的查询。

### 内连结(INNER JOIN)
#### 基本的INNER JOIN
内连结的语法格式是:

```sql
-- 内连结
FROM <tb_1> INNER JOIN <tb_2> ON <condition(s)>
```

其中 INNER 关键词表示使用了内连结，至于内连结的涵义，目前暂时可以不必细究。

按照内连结的语法，在 FROM 子句中使用 INNER JOIN 将两张表连接起来，并为 ON 子句指定连结条件为 shopproduct.product_id=product.product_id, 就得到了如下的查询语句：
```sql
SELECT SP.shop_id
       ,SP.shop_name
       ,SP.product_id
       ,P.product_name
       ,P.product_type
       ,P.sale_price
       ,SP.quantity
  FROM shopproduct AS SP
 INNER JOIN product AS P
    ON SP.product_id = P.product_id;
```
我们分别为两张表指定了简单的别名，这种操作在使用连结时是非常常见的

**要点一: 进行连结时需要在 FROM 子句中使用多张表**
**要点二:必须使用 ON 子句来指定连结条件**
**要点三: SELECT 子句中的列最好按照 表名.列名 的格式来使用**

#### 结合 WHERE 子句使用内连结
如果需要在使用内连结的时候同时使用 WHERE 子句对检索结果进行筛选，则需要把 WHERE 子句写在 ON 子句的后边。

增加 WHERE 子句的方式有好几种

第一种增加 WEHRE 子句的方式，就是把上述查询作为子查询，用括号封装起来，然后在外层查询增加筛选条件。
```sql
SELECT *
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
 WHERE shop_name = '东京'
   AND product_type = '衣服';
```

如果我们熟知 WHERE 子句将在 FROM 子句之后执行，也就是说，在做完 INNER JOIN ... ON 得到一个新表后，才会执行 WHERE 子句，那么就得到标准的写法：
```sql
SELECT  SP.shop_id
       ,SP.shop_name
       ,SP.product_id
       ,P.product_name
       ,P.product_type
       ,P.sale_price
       ,SP.quantity
  FROM shopproduct AS SP
 INNER JOIN product AS P
    ON SP.product_id = P.product_id
 WHERE SP.shop_name = '东京'
   AND P.product_type = '衣服';
```

执行顺序：FROM 子句 -> WHERE 子句 -> SELECT 子句；两张表是先按照连结列进行了连结，得到了一张新表，然后 WHERE 子句对这张新表的行按照两个条件进行了筛选，最后，SELECT 子句选出了那些我们需要的列。

当然，我们也可以先分别在两个表使用 WHERE 进行筛选，然后把上述两个子查询连结起来。
```sql
SELECT SP.shop_id
       ,SP.shop_name
       ,SP.product_id
       ,P.product_name
       ,P.product_type
       ,P.sale_price
       ,SP.quantity
  FROM (-- 子查询 1:从 shopproduct 表筛选出东京商店的信息
        SELECT *
          FROM shopproduct
         WHERE shop_name = '东京' ) AS SP
 INNER JOIN -- 子查询 2:从 product 表筛选出衣服类商品的信息
   (SELECT *
      FROM product
     WHERE product_type = '衣服') AS P
    ON SP.product_id = P.product_id;
```

#### 结合 GROUP BY 子句使用内连结
结合 GROUP BY 子句使用内连结，需要根据分组列位于哪个表区别对待。

最简单的情形，是在内连结之前就使用 GROUP BY 子句。 

但是如果分组列和被聚合的列不在同一张表，且二者都未被用于连结两张表，则只能先连结，再聚合。

**根据我们的感觉进行代码书写即可，先分组和后分组都是根据需求进行的**

#### 自连结(SELF JOIN)
之前的内连结，连结的都是不一样的两个表。但实际上一张表也可以与自身作连结，这种连接称之为自连结。需要注意，自连结并不是区分于内连结和外连结的第三种连结，自连结可以是外连结也可以是内连结，它是不同于内连结外连结的另一个连结的分类方法。

#### 内连结与关联子查询
**思路上，关联子查询根据表A的每一行取值，逐个到表 B 中的关联列中去查找取值相等的行，这种查询方式会导致不断重复查询导致计算开销较大，内连结虽然在功能上区别不大，但是性能和语法结构都提升显著**

#### 自然连结(NATURAL JOIN)
自然连结并不是区别于内连结和外连结的第三种连结，它其实是内连结的一种特例--当两个表进行自然连结时，会按照两个表中都包含的列名来进行等值内连结，此时无需使用 ON 来指定连接条件。

语法结构为
```sql
SELECT *  FROM shopproduct NATURAL JOIN product
```

### 外连结(OUTER JOIN)
内连结会丢弃两张表中不满足 ON 条件的行，和内连结相对的就是外连结。外连结会根据外连结的种类有选择地保留无法匹配到的行。

按照保留的行位于哪张表,外连结有三种形式：左连结，右连结和全外连结。

左连结会保存左表中无法按照 ON 子句匹配到的行，此时对应右表的行均为缺失值；右连结则会保存右表中无法按照 ON 子句匹配到的行，此时对应左表的行均为缺失值；而全外连结则会同时保存两个表中无法按照 ON子句匹配到的行，相应的另一张表中的行用缺失值填充。

由于连结时可以交换左表和右表的位置，因此左连结和右连结并没有本质区别。怎么使用都可以。我们可以容易看出，全外连结就是左连结和右连结的结果进行 UNION

三种外连结的对应语法分别为：
```sql
-- 左连结     
FROM <tb_1> LEFT  OUTER JOIN <tb_2> ON <condition(s)>
-- 右连结     
FROM <tb_1> RIGHT OUTER JOIN <tb_2> ON <condition(s)>
-- 全外连结
FROM <tb_1> FULL  OUTER JOIN <tb_2> ON <condition(s)>
```

**除去和内连结的区别点以外，外连结在使用上和内连结没太大的区别，我们只要能根据情况的变换准确选择就可以了**

### 多表连结
通常连结只涉及 2 张表，但有时也会出现必须同时连结 3 张以上的表的情况，原则上连结表的数量并没有限制。当然，我们应该在此时区分主表的概念，他是我们在联结中写在`FROM`语句后面的表。

给出一段简单的例子就可以让我们理解怎么书写多表联结
```sql
SELECT SP.shop_id
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
 WHERE IP.inventory_id = 'P001';
```

即使想要把连结的表增加到 4 张、5 张……使用 INNER JOIN 进行添加的方式也是完全相同的。

### ON 子句进阶--非等值连结
在刚开始介绍连结的时候，书上提到过，除了使用相等判断的等值连结，也可以使用比较运算符来进行连接。实际上，包括比较运算符(<、<=、>、>=、BETWEEN)和谓词运算(LIKE、IN、NOT 等等)在内的所有的逻辑运算都可以放在 ON 子句内作为连结条件。

比如
```sql
SELECT  product_id
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
            ON P1.sale_price <= P2.sale_price 
        ) AS X
 GROUP BY product_id, product_name, sale_price
 ORDER BY my_rank; 
```

**要理解，非等值联结不是添加行，而是条件那些满足了我们联结条件的，并且被我们在SELECT语句中选中的列，扩充列的信息**

### 交叉连结—— CROSS JOIN(笛卡尔积)
之前的无论是外连结内连结，一个共同的必备条件就是连结条件--ON 子句，用来指定连结的条件。

如果你试过不使用这个连结条件的连结查询，你可能已经发现，结果会有很多行。在连结去掉 ON 子句，就是所谓的交叉连结(CROSS JOIN)。

交叉联结会创建非常多的行和列，会对左表和右表的每一行进行组合，这经常会导致很多无意义的行出现在检索结果中。当然，在某些查询需求中，交叉连结也有一些用处。

交叉连结的语法有如下几种形式：

```sql
-- 1.使用关键字 CROSS JOIN 显式地进行交叉连结
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
```

交叉连结没有应用到实际业务之中的原因有两个。一是其结果没有实用价值，二是由于其结果行数太多，需要花费大量的运算时间和高性能设备的支持。

## 高级处理

### 窗口函数

#### 窗口函数概念及基本的使用方法

窗口函数也称为**OLAP函数**。OLAP 是 `OnLine AnalyticalProcessing` 的简称，意思是对数据库数据进行实时分析处理。

窗口函数的通用形式：

```sql
<窗口函数> OVER ([ PARTITION BY <列名> ]
                     [ ORDER BY <排序用列名> ])  
```

[   ]中的内容可以省略。

**PARTITON BY 子句** 可选参数，指示如何将查询行划分为组，类似于 GROUP BY 子句的分组功能，但是 PARTITION BY 子句并不具备 GROUP BY 子句的汇总功能，并不会改变原始表中记录的行数。

**ORDER BY 子句** 可选参数，指示如何对每个分区中的行进行排序，即决定窗口内，是按那种规则(字段)来排序的。

虽然 **PARTITON BY 子句** 和 **ORDER BY 子句** 都是可选参数，但是两个参数不能同时没有（最少二选一）。不然， `<窗口函数> OVER( )` 这种用法没用实际意义（窗口由所有查询行组成，窗口函数使用所有行计算结果）。

窗口函数中的ORDER BY与SELECT语句末尾的ORDER BY一样，可以通过关键字ASC/DESC来指定升序/降序。省略该关键字时会默认按照ASC

原则上，窗口函数只能在SELECT子句中使用。

窗口函数OVER 中的ORDER BY 子句并不会影响最终结果的排序。其只是用来决定窗口函数按何种顺序计算。

#### 窗口函数种类

大致来说，窗口函数可以分为两类。

* 将SUM、MAX、MIN等聚合函数用在窗口函数中
* RANK、DENSE_RANK等排序用的专用窗口函数

**专用窗口函数**

**RANK函数**

计算排序时，如果存在相同位次的记录，则会跳过之后的位次。

例）有 3 条记录排在第 1 位时：1 位、1 位、1 位、4 位……

**DENSE_RANK函数**

同样是计算排序，即使存在相同位次的记录，也不会跳过之后的位次。

例）有 3 条记录排在第 1 位时：1 位、1 位、1 位、2 位……

**ROW_NUMBER函数**

赋予唯一的连续位次。

例）有 3 条记录排在第 1 位时：1 位、2 位、3 位、4 位

一个代码实例为

```sql
SELECT  product_name
       ,product_type
       ,sale_price
       ,RANK() OVER (ORDER BY sale_price) AS ranking
       ,DENSE_RANK() OVER (ORDER BY sale_price) AS dense_ranking
       ,ROW_NUMBER() OVER (ORDER BY sale_price) AS row_num
  FROM product;  
```

**聚合函数在窗口函数上的使用**

聚合函数在窗口函数中的使用方法和之前的专用窗口函数一样，只是出来的结果是一个**累计**的聚合函数值。

给出一个代码实例为

```sql
SELECT  product_id
       ,product_name
       ,sale_price
       ,SUM(sale_price) OVER (ORDER BY product_id) AS current_sum
       ,AVG(sale_price) OVER (ORDER BY product_id) AS current_avg  
  FROM product;  
```

聚合函数结果是，按我们指定的排序，这里是product_id，**当前所在行及之前所有的行**的合计或均值。即累计到当前行的聚合。

#### 移动平均

在上面提到，聚合函数在窗口函数使用时，计算的是累积到当前行的所有的数据的聚合。 实际上，还可以指定更加详细的**汇总范围**。该汇总范围称为 **框架** (frame)。

语法

```sql
<窗口函数> OVER (ORDER BY <排序用列名>
                 ROWS n PRECEDING )  
                 
<窗口函数> OVER (ORDER BY <排序用列名>
                 ROWS BETWEEN n PRECEDING AND n FOLLOWING)
```

PRECEDING（“之前”）， 将框架指定为 “截止到之前 n 行”，加上自身行
FOLLOWING（“之后”）， 将框架指定为 “截止到之后 n 行”，加上自身行

例子：

BETWEEN 1 PRECEDING AND 1 FOLLOWING，将框架指定为 “之前1行” + “之后1行” + “自身”

### GROUPING运算符

常规的GROUP BY 只能得到每个分类的小计，有时候还需要计算分类的合计，可以用 ROLLUP关键字。

语法示例

```sql
SELECT  product_type
       ,regist_date
       ,SUM(sale_price) AS sum_price
  FROM product
 GROUP BY product_type, regist_date WITH ROLLUP;  
```

## 题目

### 连续登录用户标记

原始的表结构是用户登录信息，创建方法为

```sql
CREATE TABLE login_records 
(user_id INT, 
 login_date DATE);
```

创建的登录信息可能涉及一天中的精确时间，但是我们最后只考虑登录的日信息；

标记至少连续3天登录的用户 （user_id），并为这些用户生成以下字段：

* 用户id
* 连续登录开始日期
* 连续登录结束日期
* 连续登录天数

下面是一个AI生成的代码示例

```sql
WITH ranked_logins AS (
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
  HAVING COUNT(1) >= 3
)
SELECT 
  user_id,
  start_date,                   -- 直接使用第一天，无需+1
  end_date,                     -- 直接使用最后一天，无需+1
  days
FROM consecutive_streaks
ORDER BY user_id, start_date;

```

这是一个通过多层子查询实现的示例

```sql
SELECT 
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
HAVING COUNT(1) >= 3;

```
