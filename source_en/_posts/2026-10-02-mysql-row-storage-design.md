---
title: Understanding MySQL Storage Design Through a Single Row
title_zh: 从一行记录理解 MySQL 的存储设计
date: 2026-10-02
categories:
  - Programming
  - CS Foundations
tags:
  - MySQL
  - Database
  - Software Engineering
author: Hyacehila
mathjax: false
hidden: false
excerpt: A short look at data pages, row formats, and off-page storage to understand how a database represents a row and why column design affects read costs.
description: A short look at data pages, row formats, and off-page storage to understand how a database represents a row and why column design affects read costs.
excerpt_zh: 从数据页、行格式和行溢出出发，理解数据库怎样描述一行数据，以及字段设计为什么会影响读取成本。
permalink: /blog/2026/10/02/mysql-row-storage-design/
lang: en
translation_key: 2026-10-02-mysql-row-storage-design
translation_status: machine
translation_source_hash: 0d51bfe9b84509a6d142209323761359e496a88b041d5a564131552ebedaa9c3
---

<aside class="translation-notice" role="note">This English version was machine-translated from the Chinese original. Technical terms may require verification.</aside>

## First, understand why pages exist

SQL presents data as individual rows. When InnoDB manages that data, an important unit is the page, which defaults to 16KB. Pages are grouped into extents and segments, all managed within a tablespace. With the default page size, an extent contains 64 consecutive pages, totaling 1MB. Allocating space by extent helps keep data more contiguous on disk.

What I find most useful here is that looking up a row means considering the page that contains it. If the page is already cached in memory, it may be read directly. If it needs to be read from disk, that read also brings in other contents of the page. So, with other conditions being similar, smaller rows allow more records to fit on a page and more data to fit in a cache of the same size.

This is where column size starts to connect with query costs. An article table, for example, might store both titles and long article bodies, while the listing page only needs titles. Understanding where these values are actually stored helps explain which data takes up space in frequently accessed pages.

## A row needs information about how to read it

Storing column values next to one another also requires a way for the database to identify their boundaries. A rough picture of the COMPACT row format looks like this:

```text
Variable-length field length list | NULL bitmap | Record header | System and user fields
```

The variable-length field length list records how many bytes each non-NULL variable-length field actually occupies, allowing the database to parse it. Fields whose fixed lengths can be determined from the table definition do not need their lengths recorded here again. The NULL bitmap uses bits to mark nullable columns: one bit per column, allocated in whole bytes. With 9 nullable columns, the bitmap requires 2 bytes, and it is still present even when all 9 columns in a particular row have values. A column whose value is NULL does not occupy column data space. The bitmap overhead depends on the number of nullable columns.

The record header also stores a deletion flag, the record type, and the position of the next record within the page. These allow records within a page to be organized and traversed.

A row also contains system fields that support transactions. Clustered index records store a transaction ID and a rollback pointer. The latter points to an undo log record, which the database can use to reconstruct an earlier version. When a record is deleted, it is first marked for deletion. Once its earlier versions are no longer needed, purge removes it. This makes it easier for me to understand why supporting transactions requires additional information alongside the data.

## varchar length has two meanings

The 100 in `VARCHAR(100)` specifies a character count, while actual storage is measured in bytes. With utf8mb4, a character can require up to 4 bytes, so 100 characters can require up to 400 bytes of data space. If the value is just `abc`, the data itself takes 3 bytes, plus length information and other overhead. It does not reserve all 400 bytes in advance. This also explains why there is no single number to memorize for the maximum length of a varchar.

MySQL has a row size limit of 65,535 bytes, including the relevant storage overhead. If a table contains only one nullable varchar column and uses the ASCII character set, its maximum character count can be calculated as follows:

```text
65,535 - 2 bytes of length information - 1 byte of NULL information = 65,532
```

## Long fields can be stored outside the page

The previous limit concerns the size of the whole row. InnoDB also needs to consider how much space the record occupies within a page. With the default 16KB page size, the limit for the portion stored within the page is slightly less than 8KB, so long fields may need to be moved off-page.

To handle row overflow, InnoDB moves selected long field values to additional overflow pages and leaves information in the original record to locate them. Different row formats make different choices about what stays in the original page:

| Row format | What remains in the original record for a field selected for off-page storage |
| --- | --- |
| REDUNDANT, COMPACT | The first 768 bytes and a 20-byte off-page reference |
| DYNAMIC, COMPRESSED | A 20-byte off-page reference, with the field value stored on overflow pages |

This applies to fields that have already been selected for off-page storage. Short fields can still remain in the original page. Whether a field needs to be moved depends on both the total row size and the page size. MySQL 8.4 uses DYNAMIC by default.

As I understand this structure, moving long values off-page allows more records to fit in the original page. Reading a complete value, however, may require accessing overflow pages as well. Saving space in frequently accessed pages adds another step to retrieving long values. The tradeoff depends on how the data is accessed.

Returning to the article table, I would first check whether the listing page only queries titles and the detail page is where the body is needed. Then I would decide whether the body should be moved to a separate table. DYNAMIC can already store long fields off-page, so splitting the table still needs to account for query patterns and maintenance costs. The length of a field alone is not enough to decide.

Likewise, whether a column should be `NOT NULL` depends first on whether the application allows the value to be missing. Replacing an unknown value with an empty string might save a little bitmap space, but it changes the meaning of the data. Primary keys also deserve attention from a storage perspective: InnoDB secondary index records include the primary key, so longer primary keys generally make secondary indexes larger.

For me, this gives a few concrete questions to ask when looking at a table: how large is each row in practice, which columns do common queries need, will those values remain in the same page, and where else must the database look to retrieve the complete data? Thinking through these questions makes database design advice easier to understand.
