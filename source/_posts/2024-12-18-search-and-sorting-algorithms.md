---
title: "查找与排序算法：二分查找、散列表与排序实现"
title_en: "Search and Sorting Algorithms: Binary Search, Hash Tables, and Sorting Implementations"
date: 2024-12-18 21:37:43 +0800
categories: ["Programming", "CS Foundations"]
tags: ["Algorithms", "Search", "Sorting"]
author: Hyacehila
excerpt: "整理顺序查找、二分查找、二叉排序树、散列表、排序算法及其 Python 实现。"
excerpt_en: "Covers sequential search, binary search, binary search trees, hash tables, sorting algorithms, and related Python implementations."
mathjax: true
hidden: true
permalink: '/blog/2024/12/18/search-and-sorting-algorithms/'
---

## 查找

这篇文章可以和[数据结构导论：线性表、树、图与查找排序](/blog/2025/05/12/data-structures-introduction/)、[算法设计与分析：分治、动态规划与图算法](/blog/2025/05/13/algorithm-design-and-analysis/)一起阅读。几篇文章涉及的概念相近，但侧重点不同。

查找是程序中经常出现的操作，也是学习数据结构时绕不开的主题。问题通常很直接：给定一组数据和一个关键字，如何尽快找到对应记录？数据规模、是否有序、是否需要插入或删除，都会影响查找方法和数据结构的选择。

### 查找概论

存放待查记录的集合称为**查找表**（search table）。

**关键字**（key）是数据元素中的某个字段，可以用来标识一个数据元素，也可以用来标识一个记录中的字段。

如果一个关键字能够唯一标识一条记录，就称为**主关键字**（primary key）；否则称为**次关键字**（secondary key）。这两个概念也常被简称为主码和次码。

查找就是根据给定的关键字，找到对应记录。查找成功时返回记录的位置或记录本身；查找失败时，通常返回 `None` 或约定的无效下标。

按照操作过程中是否允许修改数据，查找表可以分为静态查找表和动态查找表：

- **静态查找表**：只进行查询，不插入或删除数据。
- **动态查找表**：在查询过程中还要插入或删除数据元素。

为了提高查找效率，需要为查找操作选择合适的数据结构。静态查找通常可以使用线性表；动态查找可以考虑二叉搜索树；如果只需要根据关键字直接定位，而不需要范围查询，则可以考虑散列表。

### 线性表查找

#### 顺序查找

当数据元素存放在无序的线性表中时，没有可以利用的顺序信息，最直接的方法就是从头到尾逐个比较。下面的代码定义了一个简单的顺序表，然后分别实现普通查找和带哨兵的查找。

```python
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass
class SequenceTable(Generic[T]):
    """用 Python 列表封装的顺序表。"""

    items: list[T]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> T:
        return self.items[index]


def linear_search(table: SequenceTable[int], target: int) -> int:
    """返回 target 的下标；查找失败时返回 -1。"""
    for index, value in enumerate(table.items):
        if value == target:
            return index
    return -1


def linear_search_with_sentinel(table: SequenceTable[int], target: int) -> int:
    """在表尾临时添加哨兵，省去循环中的一次边界判断。"""
    table.items.append(target)
    index = 0
    while table.items[index] != target:
        index += 1
    table.items.pop()
    return index if index < len(table) else -1


table = SequenceTable([7, 3, 9, 1])
print(linear_search(table, 9))
print(linear_search_with_sentinel(table, 8))
```

顺序查找不要求数据有序，时间复杂度为 $O(n)$。哨兵只能减少边界判断，不能改变最坏时间复杂度。

#### 有序表查找

如果元素已经按照某个关键字排序，就可以利用这种顺序缩小查找范围。二分查找、插值查找和斐波那契查找都建立在有序表之上。

#### 二分查找

二分查找每次检查区间中间的元素。如果中间元素大于目标值，就继续查找左半区间；如果小于目标值，就查找右半区间。下面的实现使用左闭右闭区间 `[left, right]`。

```python
from dataclasses import dataclass


@dataclass
class OrderedTable:
    """保存升序整数的有序表。"""

    items: list[int]

    def __post_init__(self) -> None:
        if self.items != sorted(self.items):
            raise ValueError("items 必须按升序排列")

    def __len__(self) -> int:
        return len(self.items)


def binary_search(table: OrderedTable, target: int) -> int:
    left, right = 0, len(table) - 1
    while left <= right:
        middle = left + (right - left) // 2
        value = table.items[middle]
        if value < target:
            left = middle + 1
        elif value > target:
            right = middle - 1
        else:
            return middle
    return -1


numbers = OrderedTable([1, 3, 7, 9, 12])
print(binary_search(numbers, 9))
```

二分查找的时间复杂度为 $O(\log n)$，前提是表中的元素已经有序，并且能够通过下标快速访问。

#### 插值查找

二分查找总是取中间位置。对于关键字分布比较均匀的有序表，可以根据目标值在首尾关键字之间的相对位置估计查找位置，这就是插值查找。估计位置为：

$$
mid = low + \frac{(high-low)(key-a[low])}{a[high]-a[low]}
$$

当首尾元素相等时不能使用这个公式，否则会发生除零错误。插值查找适合关键字分布均匀的场景；分布不均匀时，它不一定优于二分查找。

```python
from dataclasses import dataclass


@dataclass
class InterpolationTable:
    """保存升序整数，并提供插值查找所需的下标访问。"""

    items: list[int]

    def __post_init__(self) -> None:
        if self.items != sorted(self.items):
            raise ValueError("items 必须按升序排列")


def interpolation_search(table: InterpolationTable, target: int) -> int:
    low, high = 0, len(table.items) - 1
    while low <= high and table.items[low] <= target <= table.items[high]:
        if table.items[low] == table.items[high]:
            return low if table.items[low] == target else -1

        mid = low + (high - low) * (target - table.items[low]) // (
            table.items[high] - table.items[low]
        )
        if table.items[mid] < target:
            low = mid + 1
        elif table.items[mid] > target:
            high = mid - 1
        else:
            return mid
    return -1


table = InterpolationTable([10, 20, 30, 40, 50, 60])
print(interpolation_search(table, 40))
```

#### 斐波那契查找

斐波那契查找使用斐波那契数列确定分隔位置。数列满足 $F(n)=F(n-1)+F(n-2)$，查找区间也按照相应比例缩小。下面的代码在必要时用最后一个元素补齐逻辑数组，避免访问越界；返回值仍然是原表中的下标。

```python
from dataclasses import dataclass


@dataclass
class FibonacciTable:
    """保存升序整数的有序表，查找时不修改原始数据。"""

    items: list[int]

    def __post_init__(self) -> None:
        if self.items != sorted(self.items):
            raise ValueError("items 必须按升序排列")


def fibonacci_search(table: FibonacciTable, target: int) -> int:
    size = len(table.items)
    if size == 0:
        return -1

    fib_previous, fib_current = 0, 1
    while fib_current < size:
        fib_previous, fib_current = fib_current, fib_previous + fib_current

    offset = -1
    while fib_current > 1:
        index = min(offset + fib_previous, size - 1)
        if table.items[index] < target:
            fib_current, fib_previous = fib_previous, fib_current - fib_previous
            offset = index
        elif table.items[index] > target:
            fib_current, fib_previous = fib_current - fib_previous, fib_previous - (fib_current - fib_previous)
        else:
            return index

    if fib_previous and offset + 1 < size and table.items[offset + 1] == target:
        return offset + 1
    return -1


table = FibonacciTable([1, 3, 5, 8, 13, 21])
print(fibonacci_search(table, 13))
```

### 线性索引查找

当数据量很大、无法直接把所有记录排成一个整体有序序列时，可以为数据建立索引。索引把关键字与记录位置关联起来，查找时先定位索引，再到原始数据中查找记录。

索引结构通常分为**线性索引、树形索引和多级索引**。这里重点介绍线性索引。

#### 稠密索引

数据集中的每条记录都有对应的索引项，索引项按照关键字有序排列。稠密索引查找方便，但记录很多时，索引本身也会占用较多空间。

#### 分块索引

分块索引把数据分成若干块，块内可以无序，块间按照关键字范围有序。查找时先定位数据块，再在块内进行顺序查找。图书馆按书架整理图书，就是这种思想的直观例子。

#### 倒排索引

倒排索引是搜索系统常用的结构。它把文档中出现的关键词提取出来，再记录每个关键词对应的文档编号。用户输入关键词后，系统可以直接通过关键词索引找到相关文档，而不必逐个扫描所有文档。

倒排索引（inverted index）的核心是“关键字 → 记录号列表”的映射。关键字表和记录号列表可以继续使用数组、树或散列表组织。

### 二叉排序树

动态查找表既要支持查找，也要方便插入和删除。二叉排序树（binary search tree，BST）通过比较关键字建立这种结构：较小的值放在左子树，较大的值放在右子树。对二叉排序树进行中序遍历，可以得到一个有序序列。

二叉排序树具有以下性质：

- 如果左子树不为空，左子树中所有结点的值都小于根结点的值。
- 如果右子树不为空，右子树中所有结点的值都大于根结点的值。
- 左、右子树也分别是二叉排序树。

构造二叉排序树的目的不是单纯排序，而是让查找、插入和删除可以围绕树结构进行。下面的代码先定义结点和树，再实现查找、插入、最小值查找和删除。

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BSTNode:
    """二叉排序树结点：一个关键字和左右子树引用。"""

    key: int
    left: BSTNode | None = None
    right: BSTNode | None = None


class BinarySearchTree:
    """使用 BSTNode 维护一棵二叉排序树。"""

    def __init__(self) -> None:
        self.root: BSTNode | None = None

    def search(self, key: int) -> BSTNode | None:
        node = self.root
        while node is not None:
            if key < node.key:
                node = node.left
            elif key > node.key:
                node = node.right
            else:
                return node
        return None

    def insert(self, key: int) -> None:
        def insert_node(node: BSTNode | None) -> BSTNode:
            if node is None:
                return BSTNode(key)
            if key < node.key:
                node.left = insert_node(node.left)
            elif key > node.key:
                node.right = insert_node(node.right)
            return node

        self.root = insert_node(self.root)

    @staticmethod
    def _minimum(node: BSTNode) -> BSTNode:
        while node.left is not None:
            node = node.left
        return node

    def delete(self, key: int) -> None:
        def delete_node(node: BSTNode | None, target: int) -> BSTNode | None:
            if node is None:
                return None
            if target < node.key:
                node.left = delete_node(node.left, target)
            elif target > node.key:
                node.right = delete_node(node.right, target)
            elif node.left is None:
                return node.right
            elif node.right is None:
                return node.left
            else:
                successor = self._minimum(node.right)
                node.key = successor.key
                node.right = delete_node(node.right, successor.key)
            return node

        self.root = delete_node(self.root, key)

    def inorder(self) -> list[int]:
        result: list[int] = []

        def visit(node: BSTNode | None) -> None:
            if node is None:
                return
            visit(node.left)
            result.append(node.key)
            visit(node.right)

        visit(self.root)
        return result


tree = BinarySearchTree()
for value in [7, 3, 9, 1, 5, 8]:
    tree.insert(value)
tree.delete(3)
print(tree.search(8) is not None)
print(tree.inorder())
```

二叉排序树的查找、插入和删除操作的时间复杂度与树高有关。树较平衡时接近 $O(\log n)$；如果数据本身有序，树可能退化成链表，最坏时间复杂度为 $O(n)$。这也是平衡二叉树要解决的问题。

### 平衡二叉树：AVL 树

AVL 树要求任意结点的左、右子树高度之差最多为 1。这个差值称为平衡因子（balance factor，BF）：

$$
BF = height(left) - height(right)
$$

在平衡状态下，BF 只能是 -1、0 或 1。插入或删除结点后，如果某个结点的平衡因子绝对值大于 1，就要调整最小不平衡子树，使其恢复平衡。

AVL 树通过旋转完成调整，常见情况有 LL、RR、LR 和 RL：LL 使用右旋，RR 使用左旋，LR 先左旋再右旋，RL 先右旋再左旋。

下面的代码先定义带高度字段的 `AVLNode`，再实现高度更新、左右旋转和重新平衡。

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AVLNode:
    """AVL 树结点：关键字、左右孩子和以该结点为根的高度。"""

    key: int
    left: AVLNode | None = None
    right: AVLNode | None = None
    height: int = 1


class AVLTree:
    """通过旋转维持平衡的二叉搜索树。"""

    @staticmethod
    def height(node: AVLNode | None) -> int:
        return node.height if node is not None else 0

    @classmethod
    def update_height(cls, node: AVLNode) -> None:
        node.height = 1 + max(cls.height(node.left), cls.height(node.right))

    @classmethod
    def balance_factor(cls, node: AVLNode | None) -> int:
        if node is None:
            return 0
        return cls.height(node.left) - cls.height(node.right)

    @classmethod
    def rotate_right(cls, root: AVLNode) -> AVLNode:
        new_root = root.left
        if new_root is None:
            return root
        root.left = new_root.right
        new_root.right = root
        cls.update_height(root)
        cls.update_height(new_root)
        return new_root

    @classmethod
    def rotate_left(cls, root: AVLNode) -> AVLNode:
        new_root = root.right
        if new_root is None:
            return root
        root.right = new_root.left
        new_root.left = root
        cls.update_height(root)
        cls.update_height(new_root)
        return new_root

    @classmethod
    def rebalance(cls, node: AVLNode) -> AVLNode:
        cls.update_height(node)
        factor = cls.balance_factor(node)

        if factor > 1:
            if cls.balance_factor(node.left) < 0:
                node.left = cls.rotate_left(node.left)  # LR
            return cls.rotate_right(node)  # LL 或 LR
        if factor < -1:
            if cls.balance_factor(node.right) > 0:
                node.right = cls.rotate_right(node.right)  # RL
            return cls.rotate_left(node)  # RR 或 RL
        return node

    @classmethod
    def insert_node(cls, node: AVLNode | None, key: int) -> AVLNode:
        if node is None:
            return AVLNode(key)
        if key < node.key:
            node.left = cls.insert_node(node.left, key)
        elif key > node.key:
            node.right = cls.insert_node(node.right, key)
        else:
            return node
        return cls.rebalance(node)

    @classmethod
    def _minimum(cls, node: AVLNode) -> AVLNode:
        while node.left is not None:
            node = node.left
        return node

    @classmethod
    def delete_node(cls, node: AVLNode | None, key: int) -> AVLNode | None:
        if node is None:
            return None
        if key < node.key:
            node.left = cls.delete_node(node.left, key)
        elif key > node.key:
            node.right = cls.delete_node(node.right, key)
        elif node.left is None:
            return node.right
        elif node.right is None:
            return node.left
        else:
            successor = cls._minimum(node.right)
            node.key = successor.key
            node.right = cls.delete_node(node.right, successor.key)
        return cls.rebalance(node)

    @staticmethod
    def inorder(node: AVLNode | None) -> list[int]:
        if node is None:
            return []
        return AVLTree.inorder(node.left) + [node.key] + AVLTree.inorder(node.right)


root: AVLNode | None = None
for value in [3, 2, 1, 4, 5, 6, 7, 10, 9, 8]:
    root = AVLTree.insert_node(root, value)
root = AVLTree.delete_node(root, 5)
print(AVLTree.inorder(root))
```

把平衡调整放进插入和删除的递归回溯过程后，每层结点都会更新高度并检查平衡因子。AVL 树的查找、插入和删除时间复杂度都能保持在 $O(\log n)$。

### 多路查找树：B 树

二叉树限制每个结点最多有两个孩子。当数据规模很大、数据主要存放在磁盘上时，树高会直接影响磁盘访问次数，而磁盘访问速度通常低于内存和高速缓存。多路查找树让一个结点保存多个关键字，并拥有多个孩子，从而降低树高。

下面会依次介绍 2-3 树、2-3-4 树、B 树和 B+ 树。

#### 2-3 树

2-3 树中的结点有两种：

- 2 结点包含一个关键字和两个孩子，也可以没有孩子。
- 3 结点包含两个有序关键字和三个孩子，也可以没有孩子。

2 结点的左子树小于关键字，右子树大于关键字。3 结点的左、中、右子树分别保存小于较小关键字、介于两个关键字之间、大于较大关键字的元素。

2-3 树还要求所有叶子结点位于同一层。插入只发生在叶子处：空树直接插入一个 2 结点；插入到 2 结点后可以变成 3 结点；插入到 3 结点后需要拆分结点，并把中间关键字向上层传递。删除时，删除 3 结点中的一个关键字较简单；删除 2 结点中的关键字则可能需要借关键字或合并结点。

#### 2-3-4 树

2-3-4 树是 2-3 树的扩展，一个结点最多可以包含三个关键字，并拥有四个孩子。它的插入和删除规则更复杂，但仍然要求所有叶子在同一层。

#### B 树

B 树是平衡的多路查找树，2-3 树和 2-3-4 树都可以看作它的特殊情况。B 树的阶（order）通常用于描述一个结点最多可以拥有的孩子数。实际应用中可以根据页大小、记录大小和内存容量选择合适的阶数。阶数越大，树高通常越低；如果根结点常驻内存，查找时就能减少外存访问次数。

#### B+ 树

B+ 树是 B 树的一种改进形式。内部结点主要保存索引，记录通常存放在叶子结点中，叶子结点还会按顺序连接起来。因此，B+ 树特别适合范围查找和顺序扫描。

### 散列表查找：哈希表概述

顺序查找和树形查找都需要进行关键字比较。散列表试图直接根据关键字计算存储位置：

$$
address = h(key)
$$

函数 $h$ 称为散列函数（hash function），存放记录的连续空间称为散列表（hash table）。理想情况下，不同关键字对应不同地址；实际中多个关键字映射到同一地址的情况不可避免，这称为**冲突**（collision），发生冲突的不同关键字互称**同义词**（synonyms）。

散列表既是一种存储结构，也是一种查找结构。它不强调数据元素之间的逻辑顺序，而是面向关键字定位。因此，散列表不适合单个关键字对应多条记录的场景，也不适合范围查找。

### 散列函数的构造

一个常用的散列函数通常需要满足两个条件：计算简单，并且让地址尽量均匀分布，以减少冲突。常见方法如下。

#### 直接定址法

直接把关键字或关键字的线性函数作为地址。例如，统计不同年龄的人数时，可以直接用年龄作为地址；统计不同年份的出生人数时，也可以用年份作为地址。

这种方法简单、地址分布容易理解，且在关键字不重复时不会冲突，但需要预先知道关键字的范围，空间利用率可能较低。

#### 数字分析法

从关键字中抽取若干位作为地址。例如，电话号码或身份证号中的某些位可能具有较好的区分度。使用这种方法前，需要了解关键字的特征，并确认抽取出的数字分布较均匀。

#### 平方取中法

先计算关键字的平方，再取结果中间的若干位作为地址。这种方法不太依赖关键字本身的分布，适合关键字位数不大、难以预先分析分布的情况。

#### 折叠法

把关键字分成位数相等的几段，再将各段相加作为地址。最后一段长度不足时，可以直接使用较短的一段。只从一个方向折叠可能仍然不均匀，也可以从另一端再折叠一次后合并结果。

#### 除留余数法

用关键字除以表长 $m$ 后的余数作为地址：

$$
h(key) = key \bmod m
$$

实际选择表长时，常考虑使用接近表长的质数，以减少特定数据分布带来的冲突。

#### 随机数法

使用伪随机函数根据关键字生成地址。伪随机函数必须保证同一个关键字每次得到相同结果，否则无法再次定位记录。

### 处理散列冲突

发现冲突后，需要使用冲突处理方法寻找其他存储位置。

#### 开放定址法

遇到冲突时，按照某种探测序列继续寻找空地址：

$$
h_i(key) = (h(key)+d_i) \bmod m
$$

其中 $d_i$ 的取法不同，就形成不同的开放定址方法：

- 线性探测：$d_i=i$。实现简单，但容易产生聚集。
- 平方探测：$d_i=\pm i^2$。它可以减轻线性探测的聚集现象。
- 随机探测：$d_i$ 由可复现的伪随机序列产生。

#### 再散列函数法

使用第二个散列函数计算探测步长，例如：

$$
d_i=i\times h_2(key)
$$

发生冲突时改变探测步长，直到找到空位置或确认表已满。

#### 链地址法

每个散列地址对应一个链表或其他容器。发生冲突时，把记录放进同一地址对应的容器中；查找时先定位地址，再在容器内查找。

#### 公共溢出区法

为发生冲突的记录单独设置溢出区。主表中找不到记录时，再到溢出区查找。当冲突记录较少时，这种方法比较容易实现。

### 散列表的查找

下面的示例使用长度为 16 的线性探测散列表。代码开头定义了散列表槽位和哈希表本身；空槽使用 `None` 表示，因此关键字不必用 0 作为特殊值。

```python
from dataclasses import dataclass


@dataclass
class HashEntry:
    """散列表槽位中的记录。"""

    key: int
    value: object = None


class LinearProbingHashTable:
    """使用开放定址法和线性探测保存键值对。"""

    def __init__(self, capacity: int = 16) -> None:
        if capacity <= 0:
            raise ValueError("capacity 必须为正数")
        self.slots: list[HashEntry | None] = [None] * capacity

    def _index(self, key: int) -> int:
        return key % len(self.slots)

    def insert(self, key: int, value: object = None) -> bool:
        start = self._index(key)
        for step in range(len(self.slots)):
            index = (start + step) % len(self.slots)
            entry = self.slots[index]
            if entry is None or entry.key == key:
                self.slots[index] = HashEntry(key, value)
                return True
        return False

    def search(self, key: int) -> tuple[int, object] | None:
        start = self._index(key)
        for step in range(len(self.slots)):
            index = (start + step) % len(self.slots)
            entry = self.slots[index]
            if entry is None:
                return None
            if entry.key == key:
                return index, entry.value
        return None

    def items_by_address(self) -> list[tuple[int, int, object] | None]:
        return [
            None if entry is None else (index, entry.key, entry.value)
            for index, entry in enumerate(self.slots)
        ]


table = LinearProbingHashTable(capacity=16)
for key in [19, 14, 23, 1, 68, 20, 84, 27, 55, 11, 10, 79]:
    table.insert(key, f"record-{key}")

print(table.items_by_address())
print(table.search(68))
print(table.search(100))
```

Python 的列表本身也是一种数组结构，但在示例中使用 `HashEntry` 和 `LinearProbingHashTable` 明确表示了槽位、冲突探测和键值记录之间的关系。

前面的示例主要针对整数关键字。计算机中的字符串等数据最终也会以编码形式表示，因此可以先把它们转换为整数，或直接使用语言提供的哈希函数。需要注意的是，哈希表的哈希值和槽位布局不应被当作持久化数据格式。

## 排序

网络检索、报表生成和数据分析都经常需要有序数据。排序的任务是重新排列线性表中的元素，使其关键字满足非递增或非递减关系。后文默认使用非递减顺序。

### 排序的基本概念和分类

多关键字排序可以看作按多个关键字依次比较。实际实现中，常把多个关键字组合成比较规则；下面主要讨论单关键字排序。

当两个记录的排序关键字相等时，如果排序前后的相对顺序保持不变，就称算法是**稳定的**；否则称为**不稳定的**。例如，两名学生的成绩相同，排序后仍保持原来的先后顺序，就说明排序稳定。

按照数据是否全部装入内存，排序可以分为**内排序**和**外排序**。本文重点介绍内排序。

内排序通常从时间复杂度、辅助空间和稳定性等方面评价。算法复杂度是一个更宽泛的概念，时间复杂度只是其中的一部分。

按照主要操作，排序可以分为插入排序、交换排序、选择排序和归并排序。按照实现和复杂度，又常把冒泡排序、简单选择排序、直接插入排序归为简单排序，把希尔排序、堆排序、归并排序和快速排序归为改进排序。

排序通常作用于线性表。为了让代码清楚地表达“交换元素”这个操作，下面的示例会用一个小型自定义数组类封装 Python 列表。

### 冒泡排序（Bubble Sort）

冒泡排序反复比较相邻记录的关键字，如果顺序相反就交换。每一轮会把当前未排序部分的最大元素移动到末尾。它实现简单，时间复杂度为 $O(n^2)$，并且稳定。

```python
from dataclasses import dataclass


@dataclass
class IntArray:
    """封装可变整数序列，并提供统一的交换操作。"""

    data: list[int]

    def swap(self, left: int, right: int) -> None:
        self.data[left], self.data[right] = self.data[right], self.data[left]


def bubble_sort(values: IntArray) -> None:
    for end in range(len(values.data) - 1, 0, -1):
        swapped = False
        for index in range(end):
            if values.data[index] > values.data[index + 1]:
                values.swap(index, index + 1)
                swapped = True
        if not swapped:
            break


values = IntArray([5, 2, 8, 2, 1])
bubble_sort(values)
print(values.data)
```

如果某一轮没有发生交换，说明剩余部分已经有序，可以提前结束。

### 简单选择排序（Simple Selection Sort）

简单选择排序每轮从未排序部分找到最小元素，再把它交换到当前起始位置。它的比较次数通常为 $O(n^2)$，交换次数少于冒泡排序，但它不稳定。

```python
from dataclasses import dataclass


@dataclass
class SelectionArray:
    """保存待排序整数，并集中定义交换行为。"""

    data: list[int]

    def swap(self, left: int, right: int) -> None:
        self.data[left], self.data[right] = self.data[right], self.data[left]


def selection_sort(values: SelectionArray) -> None:
    for start in range(len(values.data) - 1):
        minimum = start
        for index in range(start + 1, len(values.data)):
            if values.data[index] < values.data[minimum]:
                minimum = index
        if minimum != start:
            values.swap(start, minimum)


values = SelectionArray([5, 2, 8, 2, 1])
selection_sort(values)
print(values.data)
```

### 直接插入排序（Straight Insertion Sort）

直接插入排序逐个取出未排序元素，把它插入已经有序的前缀中。它适合数据量较小或原本已经接近有序的序列，时间复杂度为 $O(n^2)$，并且稳定。

```python
from dataclasses import dataclass


@dataclass
class InsertionArray:
    """保存可变序列；插入排序会直接修改 data。"""

    data: list[int]


def insertion_sort(values: InsertionArray) -> None:
    for index in range(1, len(values.data)):
        current = values.data[index]
        position = index - 1
        while position >= 0 and values.data[position] > current:
            values.data[position + 1] = values.data[position]
            position -= 1
        values.data[position + 1] = current


values = InsertionArray([5, 2, 8, 2, 1])
insertion_sort(values)
print(values.data)
```

### 希尔排序（Shell Sort）

希尔排序是直接插入排序的改进。它先选择一个间隔，把相隔该间隔的元素分为一组，分别进行插入排序；随后逐步缩小间隔，直到间隔为 1。前几轮让序列接近有序，最后一轮插入排序就能少做很多移动。

希尔排序不稳定，时间复杂度取决于间隔序列，不能简单地用一个固定表达式概括。下面使用常见的 `gap // 2` 间隔序列。

```python
from dataclasses import dataclass


@dataclass
class ShellArray:
    """保存希尔排序使用的可变整数序列。"""

    data: list[int]


def shell_sort(values: ShellArray) -> None:
    gap = len(values.data) // 2
    while gap > 0:
        for index in range(gap, len(values.data)):
            current = values.data[index]
            position = index
            while position >= gap and values.data[position - gap] > current:
                values.data[position] = values.data[position - gap]
                position -= gap
            values.data[position] = current
        gap //= 2


values = ShellArray([9, 1, 8, 2, 7, 3, 6, 4, 5])
shell_sort(values)
print(values.data)
```

### 堆排序（Heap Sort）

堆排序改进了简单选择排序。简单选择排序每轮都要重新扫描未排序部分；堆排序把这些元素组织成堆，直接从堆顶取出最大值或最小值。

堆是满足特定序关系的完全二叉树：大顶堆中每个结点都大于或等于孩子，小顶堆中每个结点都小于或等于孩子。若使用从 0 开始的数组表示完全二叉树，结点 `index` 的孩子下标为 `2 * index + 1` 和 `2 * index + 2`。

```python
from dataclasses import dataclass


@dataclass
class MaxHeap:
    """用列表保存大顶堆；堆的有效区间由 size 指定。"""

    data: list[int]
    size: int = 0

    def __post_init__(self) -> None:
        self.size = len(self.data)

    def sift_down(self, root: int) -> None:
        while True:
            largest = root
            left = 2 * root + 1
            right = left + 1
            if left < self.size and self.data[left] > self.data[largest]:
                largest = left
            if right < self.size and self.data[right] > self.data[largest]:
                largest = right
            if largest == root:
                return
            self.data[root], self.data[largest] = self.data[largest], self.data[root]
            root = largest

    def sort(self) -> None:
        for root in range(self.size // 2 - 1, -1, -1):
            self.sift_down(root)
        for end in range(len(self.data) - 1, 0, -1):
            self.data[0], self.data[end] = self.data[end], self.data[0]
            self.size = end
            self.sift_down(0)


heap = MaxHeap([5, 2, 8, 2, 1])
heap.sort()
print(heap.data)
```

堆排序的时间复杂度为 $O(n\log n)$，辅助空间为 $O(1)$，但通常不稳定。

### 归并排序（Merge Sort）

归并排序把序列不断拆分，直到每个子序列只剩一个元素，再将两个有序子序列合并成一个更长的有序序列。下面的实现使用临时数组保存合并结果。

```python
from dataclasses import dataclass


@dataclass
class MergeArray:
    """归并排序操作的整数序列。"""

    data: list[int]


def merge_sort(values: MergeArray) -> None:
    def sort_range(left: int, right: int) -> None:
        if left >= right:
            return
        middle = (left + right) // 2
        sort_range(left, middle)
        sort_range(middle + 1, right)

        merged: list[int] = []
        first, second = left, middle + 1
        while first <= middle and second <= right:
            if values.data[first] <= values.data[second]:
                merged.append(values.data[first])
                first += 1
            else:
                merged.append(values.data[second])
                second += 1
        merged.extend(values.data[first : middle + 1])
        merged.extend(values.data[second : right + 1])
        values.data[left : right + 1] = merged

    sort_range(0, len(values.data) - 1)


values = MergeArray([5, 2, 8, 2, 1])
merge_sort(values)
print(values.data)
```

归并排序的时间复杂度为 $O(n\log n)$，需要 $O(n)$ 的辅助空间。合并时使用小于等于比较，可以保持相等元素的相对顺序，因此该实现稳定。

### 计数排序（Counting Sort）

计数排序不是基于比较的排序算法。它把输入整数作为计数数组的下标，统计每个值出现的次数，再按照下标顺序还原结果。因此，输入必须是整数，并且关键字范围不能远大于元素数量。

### 桶排序（Bucket Sort）

桶排序把元素按照取值范围分配到有限个桶中，再分别对每个桶排序，最后依次连接所有桶。桶的数量和分配规则会明显影响性能；数据分布均匀时更容易发挥优势。

### 基数排序（Radix Sort）

基数排序不直接比较完整关键字，而是按位处理整数。LSD（least significant digit）从最低位开始，MSD（most significant digit）从最高位开始。每一轮通常需要稳定的计数排序或桶排序作为子过程。

- **MSD**：从高位开始处理，适合按前缀划分数据。
- **LSD**：从低位开始处理，要求每一轮排序保持稳定。

### 快速排序（Quick Sort）

快速排序选择一个基准值（pivot），通过分区操作把序列分成两部分：左侧元素不大于基准值，右侧元素不小于基准值，然后递归处理两部分。

快速排序是不稳定的，平均时间复杂度为 $O(n\log n)$；如果每次选择的基准值都接近最小值或最大值，最坏时间复杂度会退化为 $O(n^2)$。

下面的代码使用原地分区。代码开头的 `QuickArray` 明确表示待排序的可变线性表。

```python
from dataclasses import dataclass


@dataclass
class QuickArray:
    """快速排序直接修改其中的 data。"""

    data: list[int]


def quick_sort(values: QuickArray) -> None:
    def partition(left: int, right: int) -> int:
        pivot = values.data[right]
        boundary = left
        for index in range(left, right):
            if values.data[index] <= pivot:
                values.data[boundary], values.data[index] = (
                    values.data[index],
                    values.data[boundary],
                )
                boundary += 1
        values.data[boundary], values.data[right] = (
            values.data[right],
            values.data[boundary],
        )
        return boundary

    def sort_range(left: int, right: int) -> None:
        if left >= right:
            return
        pivot_index = partition(left, right)
        sort_range(left, pivot_index - 1)
        sort_range(pivot_index + 1, right)

    sort_range(0, len(values.data) - 1)


values = QuickArray([5, 2, 8, 2, 1])
quick_sort(values)
print(values.data)
```

#### 基准值的选择

快速排序的性能很大程度上取决于基准值。可以使用随机选取、三数取中等方法，让分区结果尽量接近均衡，降低退化的可能性。

#### 小数组优化

对很小的子数组继续递归分区，额外开销可能超过排序本身。实际实现中，可以在子数组长度较小时改用直接插入排序。

#### 递归优化

递归深度过大时会增加栈空间消耗。可以优先递归较短的一侧，对较长的一侧使用循环处理，从而把额外栈空间控制在较小范围内。

## 结语

快速排序的名字来自它在许多实际场景中的高效表现，但它并不是所有输入下都最快。基准值选择、数据分布、稳定性要求和可用内存都会影响算法选择。理解这些取舍，比单独记住某个排序算法的名字更重要。
